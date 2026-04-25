"""Blackwell sm_100a GEMM benchmark.

Runs ``build_gemm`` from ``examples/blackwell/gemm_highperf_blackwell.py``
against ``torch.matmul`` (cuBLAS) on a handful of shapes. Emits a table of
median pyptx / cuBLAS TFLOPS and the ratio.

Usage:
    python benchmarks/bench_blackwell_gemm.py
    python benchmarks/bench_blackwell_gemm.py --size 8192
"""
from __future__ import annotations

import argparse
import os
import statistics

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torch

from examples.blackwell.gemm_highperf_blackwell import build_gemm, build_gemm_2sm


def _time(fn, iters: int = 20) -> float:
    """Median-over-3 timing in ms per invocation."""
    times = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / iters)
    return statistics.median(times)


def benchmark(M: int, N: int, K: int) -> tuple[float, float, float]:
    k = build_gemm(M, N, K)
    k2 = build_gemm_2sm(M, N, K) if M % 256 == 0 else None
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    bt = b.transpose(0, 1).contiguous()

    for _ in range(5):
        k(a, bt)
        if k2 is not None:
            k2(a, bt)
        torch.matmul(a, b)
    torch.cuda.synchronize()

    ms_pyptx = _time(lambda: k(a, bt))
    ms_2sm = _time(lambda: k2(a, bt)) if k2 is not None else float("inf")
    ms_cublas = _time(lambda: torch.matmul(a, b))
    flops = 2 * M * N * K
    return (
        flops / (ms_pyptx * 1e-3) / 1e12,
        flops / (ms_2sm * 1e-3) / 1e12,
        flops / (ms_cublas * 1e-3) / 1e12,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=None,
                        help="Run only a single square M=N=K at this size.")
    args = parser.parse_args()

    if args.size is not None:
        shapes = [(args.size, args.size, args.size)]
    else:
        shapes = [
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 4096, 8192),
            (4096, 8192, 8192),
            (8192, 8192, 8192),
        ]

    header = (f'{"Shape":>18s} | {"1SM":>8s} | {"2SM":>8s} | '
              f'{"cuBLAS":>8s} | {"best/cuBLAS":>12s}')
    print(header)
    print("-" * len(header))
    for M, N, K in shapes:
        one_tf, two_tf, cublas_tf = benchmark(M, N, K)
        best_tf = max(one_tf, two_tf)
        shape = f"{M}x{N}x{K}"
        ratio = best_tf / cublas_tf * 100
        two_str = f"{two_tf:.0f}" if two_tf != float("inf") else "n/a"
        print(f"{shape:>18s} | {one_tf:>8.0f} | {two_str:>8s} | "
              f"{cublas_tf:>8.0f} | {ratio:>11.0f}%", flush=True)


if __name__ == "__main__":
    main()
