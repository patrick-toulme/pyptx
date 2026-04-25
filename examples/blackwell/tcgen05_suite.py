"""Blackwell tcgen05 bring-up suite.

This is a small runtime harness for the current Blackwell/B200 work. It keeps
the instruction-level probes and the current GEMM diagnostics in one place so
we can rerun them quickly after each lowering/runtime change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import torch

try:
    from pyptx.examples.blackwell.gemm_experimental_blackwell import build_gemm_no_tma_debug
    from pyptx.examples.blackwell.tcgen05_accum_probe import build as build_accum
    from pyptx.examples.blackwell.tcgen05_ld_register_probe import build as build_ld_register
    from pyptx.examples.blackwell.tcgen05_mma_probe import build as build_single_mma
    from pyptx.examples.blackwell.tcgen05_roundtrip import build as build_roundtrip
    from pyptx.examples.blackwell.tcgen05_smoke import (
        build_alloc_only,
        build_ld_only,
        build_mma_only,
    )
except ImportError:
    from examples.blackwell.gemm_experimental_blackwell import build_gemm_no_tma_debug
    from examples.blackwell.tcgen05_accum_probe import build as build_accum
    from examples.blackwell.tcgen05_ld_register_probe import build as build_ld_register
    from examples.blackwell.tcgen05_mma_probe import build as build_single_mma
    from examples.blackwell.tcgen05_roundtrip import build as build_roundtrip
    from examples.blackwell.tcgen05_smoke import (
        build_alloc_only,
        build_ld_only,
        build_mma_only,
    )


@dataclass
class CaseResult:
    name: str
    ok: bool
    detail: str


def _run_scalar(builder) -> float:
    x = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
    out = builder()(x)
    torch.cuda.synchronize()
    return float(out[0, 0].item())


def case_smoke_alloc() -> CaseResult:
    got = _run_scalar(build_alloc_only)
    return CaseResult("smoke_alloc", got == 11.0, f"got={got:.1f} expected=11.0")


def case_smoke_mma() -> CaseResult:
    got = _run_scalar(build_mma_only)
    return CaseResult("smoke_mma", got == 22.0, f"got={got:.1f} expected=22.0")


def case_smoke_ld() -> CaseResult:
    got = _run_scalar(build_ld_only)
    return CaseResult("smoke_ld", got == 33.0, f"got={got:.1f} expected=33.0")


def case_single_mma() -> CaseResult:
    x = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
    out = build_single_mma()(x)
    torch.cuda.synchronize()
    diff = float((out - 16.0).abs().max())
    return CaseResult("single_mma", diff == 0.0, f"max_abs_to_16={diff:.1f}")


def case_accum_sweep() -> CaseResult:
    x = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
    got: list[float] = []
    for num_mmas in (1, 2, 3, 4):
        out = build_accum(num_mmas=num_mmas, advance_descs=False)(x)
        torch.cuda.synchronize()
        got.append(float(out[0, 0].item()))
    ok = got == [16.0, 32.0, 48.0, 48.0]
    detail = "got=[" + ", ".join(f"{x:.1f}" for x in got) + "] expected=[16,32,48,48]"
    return CaseResult("accum_sweep", ok, detail)


def case_accum_split_commit() -> CaseResult:
    x = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
    out = build_accum(num_mmas=4, advance_descs=False, split_after=2)(x)
    torch.cuda.synchronize()
    got = float(out[0, 0].item())
    return CaseResult("accum_split_commit", got == 64.0, f"got={got:.1f} expected=64.0")


def case_roundtrip() -> CaseResult:
    x = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
    out = build_roundtrip()(x)
    torch.cuda.synchronize()
    ref = torch.arange(1, 65, device="cuda", dtype=torch.float32).repeat(32, 1)
    diff = float((out - ref).abs().max())
    return CaseResult("roundtrip", diff == 0.0, f"max_abs_to_pattern={diff:.1f}")


def case_ld_register_probe() -> CaseResult:
    a = torch.zeros((128, 64), device="cuda", dtype=torch.bfloat16)
    b = torch.zeros((64, 256), device="cuda", dtype=torch.bfloat16)
    a[:, 0] = 1
    b[0, :] = torch.arange(1, 257, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    got: dict[int, tuple[list[int], list[float]]] = {}
    for load_count in (16, 32, 64, 128):
        out = build_ld_register(load_count=load_count)(a, b.t().contiguous())
        torch.cuda.synchronize()
        row0 = out[0].float().cpu().numpy()
        nz = np.nonzero(row0)[0].tolist()
        vals = row0[nz].tolist()
        got[load_count] = (nz, vals)
    ok = (
        got[16] == ([0, 8], [1.0, 9.0])
        and got[32] == ([0, 8, 16, 24], [1.0, 9.0, 17.0, 25.0])
        and got[64] == (
            [0, 8, 16, 24, 32, 40, 48, 56],
            [1.0, 9.0, 17.0, 25.0, 33.0, 41.0, 49.0, 57.0],
        )
        and got[128] == (
            [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120],
            [1.0, 9.0, 17.0, 25.0, 33.0, 41.0, 49.0, 57.0, 65.0, 73.0, 81.0, 89.0, 97.0, 105.0, 113.0, 121.0],
        )
    )
    detail = (
        f"x16={got[16]} "
        f"x32={got[32]} "
        f"x64={got[64]} "
        f"x128={got[128]}"
    )
    return CaseResult("ld_register_probe", ok, detail)


def case_ld_x128_row_residue_map() -> CaseResult:
    a = torch.zeros((128, 64), device="cuda", dtype=torch.bfloat16)
    b = torch.zeros((64, 256), device="cuda", dtype=torch.bfloat16)
    a[:, 0] = torch.arange(1, 129, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    b[0, :] = 1
    out = build_ld_register(load_count=128)(a, b.t().contiguous())
    torch.cuda.synchronize()
    out = out.float().cpu().numpy()
    got: list[tuple[list[int], list[float]]] = []
    for row in range(8):
        nz = np.nonzero(out[row])[0].tolist()
        vals = out[row][nz].tolist()
        got.append((nz[:4], vals[:4]))
    expect = [
        ([row, row + 8, row + 16, row + 24], [float(row + 1)] * 4)
        for row in range(8)
    ]
    ok = got == expect
    return CaseResult("ld_x128_row_residue_map", ok, f"got={got}")


def case_gemm_ones() -> CaseResult:
    kfun = build_gemm_no_tma_debug(128, 256, 64)
    a = torch.ones((128, 64), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((64, 256), device="cuda", dtype=torch.bfloat16)
    out = kfun(a, b.t().contiguous())
    torch.cuda.synchronize()
    diff = float((out - 64.0).abs().max())
    return CaseResult("gemm_ones", diff == 0.0, f"max_abs_to_64={diff:.1f}")


def case_gemm_random() -> CaseResult:
    kfun = build_gemm_no_tma_debug(128, 256, 64)
    rng = np.random.default_rng(128 * 10007 + 256 * 313 + 64)
    a = torch.tensor(
        (rng.standard_normal((128, 64)) * 0.1).astype(np.float32),
        device="cuda",
        dtype=torch.bfloat16,
    )
    b = torch.tensor(
        (rng.standard_normal((64, 256)) * 0.1).astype(np.float32),
        device="cuda",
        dtype=torch.bfloat16,
    )
    out = kfun(a, b.t().contiguous())
    torch.cuda.synchronize()
    ref = a.float() @ b.float()
    diff = float((out - ref).abs().max())
    ok = bool(torch.allclose(out, ref, atol=5e-2, rtol=5e-2))
    return CaseResult("gemm_random", ok, f"max_abs={diff:.6f}")


def case_gemm_k_lane_map() -> CaseResult:
    kfun = build_gemm_no_tma_debug(128, 256, 64)
    b = np.zeros((64, 256), dtype=np.float32)
    for kk in range(64):
        b[kk, :] = kk + 1
    bt = torch.tensor(b, device="cuda", dtype=torch.bfloat16)
    got = []
    for kk in range(16):
        a = np.zeros((128, 64), dtype=np.float32)
        a[:, kk] = 1.0
        at = torch.tensor(a, device="cuda", dtype=torch.bfloat16)
        out = kfun(at, bt.t().contiguous())
        torch.cuda.synchronize()
        got.append(float(out[0, 0].item()))
    expect = [float(x) for x in range(1, 17)]
    ok = got == expect
    detail = "got=[" + ", ".join(f"{x:.1f}" for x in got[:8]) + ", ...]"
    return CaseResult("gemm_k_lane_map", ok, detail)


def case_gemm_row_col_partition() -> CaseResult:
    kfun = build_gemm_no_tma_debug(128, 256, 64)
    kk = 0

    a = np.zeros((128, 64), dtype=np.float32)
    a[:, kk] = np.arange(1, 129, dtype=np.float32)
    b = np.zeros((64, 256), dtype=np.float32)
    b[kk, :] = 1.0
    out_rows = kfun(
        torch.tensor(a, device="cuda", dtype=torch.bfloat16),
        torch.tensor(b, device="cuda", dtype=torch.bfloat16).t().contiguous(),
    )
    torch.cuda.synchronize()
    rows = [float(out_rows[i, 0].item()) for i in range(16)]

    a = np.zeros((128, 64), dtype=np.float32)
    a[:, kk] = 1.0
    b = np.zeros((64, 256), dtype=np.float32)
    b[kk, :] = np.arange(1, 257, dtype=np.float32)
    out_cols = kfun(
        torch.tensor(a, device="cuda", dtype=torch.bfloat16),
        torch.tensor(b, device="cuda", dtype=torch.bfloat16).t().contiguous(),
    )
    torch.cuda.synchronize()
    cols = [float(out_cols[0, j].item()) for j in range(16)]

    ok = rows == [float(i) for i in range(1, 17)] and cols == [float(i) for i in range(1, 17)]
    detail = f"rows0..15={rows} cols0..15={cols}"
    return CaseResult("gemm_row_col_partition", ok, detail)


def run_suite() -> list[CaseResult]:
    return [
        case_smoke_alloc(),
        case_smoke_mma(),
        case_smoke_ld(),
        case_single_mma(),
        case_accum_sweep(),
        case_accum_split_commit(),
        case_roundtrip(),
        case_ld_register_probe(),
        case_ld_x128_row_residue_map(),
        case_gemm_ones(),
        case_gemm_random(),
        case_gemm_k_lane_map(),
        case_gemm_row_col_partition(),
    ]


def main() -> int:
    results = run_suite()
    failures = 0
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
        if not result.ok:
            failures += 1
    print(f"summary: {len(results) - failures} passed, {failures} failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
