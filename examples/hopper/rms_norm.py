"""Fused RMS normalization, written in pyptx, callable from JAX and PyTorch.

Reaches **2.6 TB/s** at B=2048 N=8192 f32 on H100 (88% of HBM3 peak),
**3.9x** faster than the reference PyTorch eager implementation.

``Y[b, i] = X[b, i] * W[i] / sqrt(mean(X[b, :]^2) + eps)``

Structure mirrors Triton's ``_layer_norm_fwd_fused`` tutorial (tutorial 05):
one CTA per row, each thread accumulates a strided slice of the row into a
per-thread sum-of-squares, a warp-level butterfly reduction turns that into
one partial per warp, and a final warp reduces those partials to the full-row
sum. Then every thread reuses the values it already loaded and writes the
normalized result.

This file is deliberately written to show off the pyptx DSL sugar:

* ``ptx.global_ptrs(X, W, Y)`` — one call for the three param-ptr
  prologues.
* ``reg.scalar(f32, init=0.0)`` — Python float literals are encoded to
  PTX's ``0fXXXXXXXX`` form automatically.
* ``Reg`` arithmetic operators — ``ptr + off`` and ``idx * 4`` emit
  exactly one PTX instruction each, just like hand-writing the ``add.s64``
  and ``mul.wide.u32`` would.
* ``ptx.warp.reduce_sum(sum_sq)`` — canonical butterfly shfl reduction
  across the warp, no hand-rolled helper.

Run ``python examples/hopper/rms_norm.py`` to execute both a ``jax.jit``
path and a PyTorch eager path against framework references.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pyptx import kernel, reg, ptx, smem, Tile
from pyptx.types import f32, u32


WARP_SIZE = 32
BLOCK_CANDIDATES = (512, 256, 128, 64, 32)


def _pick_block(n: int) -> int:
    # Prefer more items-per-thread (smaller blocks) for memory-level
    # parallelism — more outstanding v4 loads hide DRAM latency.
    best = None
    for block in BLOCK_CANDIDATES:
        ipt = n // block
        if n % block == 0 and ipt >= 4 and ipt % 4 == 0 and block >= 128:
            if best is None or ipt > best[1]:
                best = (block, ipt)
    if best is not None:
        return best[0]
    for block in BLOCK_CANDIDATES:
        if n % block == 0:
            return block
    raise AssertionError(
        f"N={n} must be divisible by one of {BLOCK_CANDIDATES}"
    )


def build_rms_norm(B: int, N: int, *, eps: float = 1e-6, arch: str = "sm_90a"):
    """Build an RMS-norm kernel specialized for batch ``B`` and feature
    dim ``N``. ``N`` must be divisible by the CTA size."""
    block = _pick_block(N)
    num_warps = block // WARP_SIZE
    items_per_thread = N // block
    use_v4 = items_per_thread >= 4 and items_per_thread % 4 == 0
    v4_iters = items_per_thread // 4 if use_v4 else 0

    version = (8, 7) if arch.startswith("sm_100") else None
    @kernel(
        in_specs=(
            Tile(B, N, f32),   # X: (batch, features)
            Tile(N, f32),      # W: (features,) — learned scale
        ),
        out_specs=(Tile(B, N, f32),),
        grid=(B, 1, 1),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def rms_norm(X, W, Y):
        partials = smem.alloc(f32, (num_warps, 1))
        stats = smem.alloc(f32, (1, 1))

        # --- prologue: three param pointers in one line ---------------
        px, pw, py = ptx.global_ptrs(X, W, Y)

        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        row = reg.scalar(u32)
        ptx.inst.mov.u32(row, ptx.special.ctaid.x())
        row_byte_off = row * (N * 4)
        px += row_byte_off
        py += row_byte_off
        lane = tid & (WARP_SIZE - 1)
        warp_id = tid >> 5

        # --- pass 1: load items_per_thread strided values per thread --
        sum_sq = reg.scalar(f32, init=0.0)
        x_vals = reg.array(f32, items_per_thread)

        if use_v4:
            elem_base = tid << 2
            for j in range(v4_iters):
                idx = elem_base if j == 0 else elem_base + (j * block * 4)
                ptr = px + idx * 4
                ptx.inst.ld.global_.v4.f32(
                    [x_vals[j*4], x_vals[j*4+1], x_vals[j*4+2], x_vals[j*4+3]],
                    ptx.addr(ptr),
                )
                for sub in range(4):
                    ptx.inst.fma.rn.f32(sum_sq, x_vals[j*4+sub], x_vals[j*4+sub], sum_sq)
        else:
            for i in range(items_per_thread):
                idx = reg.scalar(u32)
                ptx.inst.add.u32(idx, tid, i * block)
                ptr = px + idx * 4
                ptx.inst.ld.global_.f32(x_vals[i], ptx.addr(ptr))
                ptx.inst.fma.rn.f32(sum_sq, x_vals[i], x_vals[i], sum_sq)

        ptx.warp.reduce_sum(sum_sq)

        with ptx.if_(lane == 0):
            partials[warp_id, 0] = sum_sq
        ptx.bar.sync(0)

        with ptx.if_(tid == 0):
            block_sum = reg.scalar(f32, init=0.0)
            for i in range(num_warps):
                ptx.inst.add.f32(block_sum, block_sum, partials[i, 0])
            stats[0, 0] = block_sum
        ptx.bar.sync(0)

        ptx.inst.mov.f32(sum_sq, stats[0, 0])

        mean_sq = reg.scalar(f32)
        inv_n = reg.scalar(f32, init=1.0 / N)
        ptx.inst.mul.f32(mean_sq, sum_sq, inv_n)
        eps_reg = reg.scalar(f32, init=eps)
        ptx.inst.add.f32(mean_sq, mean_sq, eps_reg)
        rstd = reg.scalar(f32)
        ptx.inst.rsqrt.approx.f32(rstd, mean_sq)

        if use_v4:
            for j in range(v4_iters):
                idx = elem_base if j == 0 else elem_base + (j * block * 4)
                off = idx * 4

                w_vals = [reg.scalar(f32) for _ in range(4)]
                ptx.inst.ld.global_.v4.f32(w_vals, ptx.addr(pw + off))

                y_vals = []
                for sub in range(4):
                    y_val = reg.scalar(f32)
                    ptx.inst.mul.f32(y_val, x_vals[j*4+sub], rstd)
                    ptx.inst.mul.f32(y_val, y_val, w_vals[sub])
                    y_vals.append(y_val)

                ptx.inst.st.global_.v4.f32(ptx.addr(py + off), y_vals)
        else:
            for i in range(items_per_thread):
                idx = reg.scalar(u32)
                ptx.inst.add.u32(idx, tid, i * block)
                off = idx * 4

                pw_el = pw + off
                w_val = reg.scalar(f32)
                ptx.inst.ld.global_.f32(w_val, ptx.addr(pw_el))

                y_val = reg.scalar(f32)
                ptx.inst.mul.f32(y_val, x_vals[i], rstd)
                ptx.inst.mul.f32(y_val, y_val, w_val)

                py_el = py + off
                ptx.inst.st.global_.f32(ptx.addr(py_el), y_val)

        ptx.ret()

    return rms_norm


# ---------------------------------------------------------------------------
# JAX reference + test harness
# ---------------------------------------------------------------------------

def rms_norm_ref(x: jnp.ndarray, w: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Plain JAX reference for correctness checking."""
    mean_sq = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(mean_sq + eps) * w


def _run_jax_case(B: int, N: int) -> None:
    k = build_rms_norm(B, N)
    np.random.seed(B * 7919 + N)
    x_np = np.random.randn(B, N).astype(np.float32) * 0.3
    w_np = (np.random.randn(N) * 0.1 + 1.0).astype(np.float32)
    x = jnp.asarray(x_np)
    w = jnp.asarray(w_np)

    @jax.jit
    def fn(x, w):
        return k(x, w)

    out = np.asarray(fn(x, w))
    ref = np.asarray(rms_norm_ref(x, w))
    diff = float(np.abs(out - ref).max())
    ok = bool(np.allclose(out, ref, atol=1e-4, rtol=1e-3))
    status = "OK  " if ok else "FAIL"
    print(f"[JAX  {status}] B={B:4d} N={N:5d}  max_abs={diff:.3e}")


def _run_torch_case(B: int, N: int) -> None:
    import torch

    k = build_rms_norm(B, N)
    np.random.seed(B * 7919 + N)
    x_np = np.random.randn(B, N).astype(np.float32) * 0.3
    w_np = (np.random.randn(N) * 0.1 + 1.0).astype(np.float32)
    x = torch.tensor(x_np, device="cuda")
    w = torch.tensor(w_np, device="cuda")

    out = k(x, w)
    torch.cuda.synchronize()
    ms = (x * x).mean(dim=-1, keepdim=True)
    ref = x * torch.rsqrt(ms + 1e-6) * w
    diff = float((out - ref).abs().max())
    ok = bool(torch.allclose(out, ref, atol=1e-4, rtol=1e-3))
    status = "OK  " if ok else "FAIL"
    print(f"[Torch{status}] B={B:4d} N={N:5d}  max_abs={diff:.3e}")


def main() -> None:
    _ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()

    for B, N in [(4, 64), (16, 512), (32, 1024), (128, 2048), (256, 4096)]:
        _run_jax_case(B, N)
        _run_torch_case(B, N)


if __name__ == "__main__":
    main()
