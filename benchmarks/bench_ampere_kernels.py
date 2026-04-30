"""Ampere (A100) benchmarks — first-class arch="sm_80" support.

Mirrors ``bench_final.py`` but uses the ``examples.ampere.*`` wrappers,
which build the same Hopper kernels with ``arch="sm_80"``. The GEMM
benchmark uses the dedicated A100 cp.async + mma.sync pipeline from
``examples/ampere/gemm_pipelined.py``.

Usage::

    python benchmarks/bench_ampere_kernels.py            # all
    python benchmarks/bench_ampere_kernels.py rms layer  # subset

All numbers measured on a single A100 80GB SXM4 with PyTorch CUDA events.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch


ITERS = 200
WARMUP = 20


def _time_events(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn(*args)
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / ITERS


def bw(byte_count, us):
    return byte_count / (us * 1e-6) / 1e9


def hdr(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def rms_norm():
    from examples.ampere.rms_norm import build_rms_norm
    hdr("[A100] RMS NORM (fp32, sm_80)")
    print(f"{'B':>5} {'N':>6} {'pyptx':>9} {'torch':>9} {'pyptx GB/s':>11} {'torch GB/s':>11} {'speedup':>8}")
    for B, N in [(32, 1024), (256, 4096), (1024, 8192), (2048, 8192)]:
        k = build_rms_norm(B, N)
        x = torch.randn(B, N, device="cuda")
        w = torch.randn(N, device="cuda")
        k(x, w); torch.cuda.synchronize()
        t_p = _time_events(lambda x, w: k(x, w), x, w) * 1e3

        def ref(x, w):
            return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6) * w
        ref(x, w); torch.cuda.synchronize()
        t_t = _time_events(ref, x, w) * 1e3
        bytes_ = 2 * B * N * 4 + N * 4
        print(f"{B:5d} {N:6d} {t_p:8.2f}us {t_t:8.2f}us {bw(bytes_, t_p):11.1f} {bw(bytes_, t_t):11.1f} {t_t / t_p:8.2f}x")


def layer_norm():
    from examples.ampere.layer_norm import build_layer_norm
    hdr("[A100] LAYER NORM (fp32, sm_80)")
    print(f"{'B':>5} {'N':>6} {'pyptx':>9} {'torch':>9} {'pyptx GB/s':>11} {'torch GB/s':>11} {'speedup':>8}")
    for B, N in [(32, 1024), (256, 4096), (1024, 8192), (2048, 8192)]:
        k = build_layer_norm(B, N)
        x = torch.randn(B, N, device="cuda")
        w = torch.randn(N, device="cuda") * 0.1 + 1.0
        b = torch.randn(N, device="cuda") * 0.1
        k(x, w, b); torch.cuda.synchronize()
        t_p = _time_events(lambda x, w, b: k(x, w, b), x, w, b) * 1e3

        def ref(x, w, b):
            return torch.nn.functional.layer_norm(x, (N,), w, b)
        ref(x, w, b); torch.cuda.synchronize()
        t_t = _time_events(ref, x, w, b) * 1e3
        bytes_ = 2 * B * N * 4 + 2 * N * 4
        print(f"{B:5d} {N:6d} {t_p:8.2f}us {t_t:8.2f}us {bw(bytes_, t_p):11.1f} {bw(bytes_, t_t):11.1f} {t_t / t_p:8.2f}x")


def swiglu():
    from examples.ampere.swiglu import build_fused_silu_mul
    hdr("[A100] FUSED SILU * MUL (fp32, sm_80)")
    print(f"{'M':>5} {'F':>6} {'pyptx':>9} {'torch':>9} {'pyptx GB/s':>11} {'torch GB/s':>11} {'speedup':>8}")
    for M, F in [(32, 1024), (256, 4096), (1024, 8192), (2048, 8192)]:
        k = build_fused_silu_mul(M, F)
        g = torch.randn(M, F, device="cuda") * 0.5
        u = torch.randn(M, F, device="cuda") * 0.5
        k(g, u); torch.cuda.synchronize()
        t_p = _time_events(lambda g, u: k(g, u), g, u) * 1e3

        def ref(g, u):
            return torch.nn.functional.silu(g) * u
        ref(g, u); torch.cuda.synchronize()
        t_t = _time_events(ref, g, u) * 1e3
        bytes_ = 3 * M * F * 4
        print(f"{M:5d} {F:6d} {t_p:8.2f}us {t_t:8.2f}us {bw(bytes_, t_p):11.1f} {bw(bytes_, t_t):11.1f} {t_t / t_p:8.2f}x")


def gemm_highperf():
    from examples.ampere.gemm_highperf_ampere import build_gemm_highperf
    hdr("[A100] GEMM (bf16, ldmatrix + cp.async 3-stage + mma.sync m16n8k16, sm_80)")
    print(f"{'M':>5} {'N':>5} {'K':>5} {'pyptx':>9} {'cublas':>9} {'pyptx TFLOPS':>13} {'cublas TFLOPS':>14} {'ratio':>7}")
    for M, N, K in [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048),
                    (4096, 4096, 4096)]:
        k = build_gemm_highperf(M, N, K)
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
        b_t = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
        b = b_t.transpose(0, 1).contiguous()
        k(a, b_t); torch.cuda.synchronize()
        t_p = _time_events(lambda a, b_t: k(a, b_t), a, b_t) * 1e3

        def ref(a, b):
            return torch.matmul(a, b)
        ref(a, b); torch.cuda.synchronize()
        t_t = _time_events(ref, a, b) * 1e3
        flops = 2 * M * N * K
        tf_p = flops / (t_p * 1e-6) / 1e12
        tf_t = flops / (t_t * 1e-6) / 1e12
        print(f"{M:5d} {N:5d} {K:5d} {t_p:8.2f}us {t_t:8.2f}us {tf_p:13.1f} {tf_t:14.1f} {tf_p / tf_t:7.2f}x")


def main():
    rms_norm()
    layer_norm()
    swiglu()
    gemm_highperf()


if __name__ == "__main__":
    import sys

    dispatch = {
        "rms": rms_norm,
        "layer": layer_norm,
        "silu": swiglu,
        "gemm": gemm_highperf,
    }
    targets = sys.argv[1:]
    if not targets:
        main()
    else:
        for t in targets:
            dispatch[t]()
