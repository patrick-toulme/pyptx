"""Blackwell kernel benchmark suite.

Measures the maintained Blackwell kernels against simple PyTorch references:

- RMSNorm
- LayerNorm
- SwiGLU
- Grouped GEMM

Usage:
    python benchmarks/bench_blackwell_kernels.py
    python benchmarks/bench_blackwell_kernels.py rms
    python benchmarks/bench_blackwell_kernels.py grouped
"""

from __future__ import annotations

import sys
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
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def rms_norm():
    from examples.blackwell.rms_norm import build_rms_norm

    hdr("BLACKWELL RMS NORM (fp32)")
    print(f"{'B':>5} {'N':>6} {'pyptx':>9} {'torch':>9} {'pyptx GB/s':>11} {'torch GB/s':>11} {'speedup':>8}")
    for B, N in [(32, 1024), (256, 4096), (1024, 8192), (2048, 8192)]:
        k = build_rms_norm(B, N)
        x = torch.randn(B, N, device="cuda")
        w = torch.randn(N, device="cuda")
        k(x, w)
        torch.cuda.synchronize()
        t_p = _time_events(lambda x, w: k(x, w), x, w) * 1e3

        def ref(x, w):
            return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6) * w

        ref(x, w)
        torch.cuda.synchronize()
        t_t = _time_events(ref, x, w) * 1e3
        bytes_ = 2 * B * N * 4 + N * 4
        print(f"{B:5d} {N:6d} {t_p:8.2f}us {t_t:8.2f}us {bw(bytes_, t_p):11.1f} {bw(bytes_, t_t):11.1f} {t_t / t_p:8.2f}x")


def layer_norm():
    from examples.blackwell.layer_norm import build_layer_norm

    hdr("BLACKWELL LAYER NORM (fp32)")
    print(f"{'B':>5} {'N':>6} {'pyptx':>9} {'torch':>9} {'pyptx GB/s':>11} {'torch GB/s':>11} {'speedup':>8}")
    for B, N in [(32, 1024), (256, 4096), (1024, 8192), (2048, 8192)]:
        k = build_layer_norm(B, N)
        x = torch.randn(B, N, device="cuda")
        w = torch.randn(N, device="cuda") * 0.1 + 1.0
        b = torch.randn(N, device="cuda") * 0.1
        k(x, w, b)
        torch.cuda.synchronize()
        t_p = _time_events(lambda x, w, b: k(x, w, b), x, w, b) * 1e3

        def ref(x, w, b):
            return torch.nn.functional.layer_norm(x, (N,), w, b)

        ref(x, w, b)
        torch.cuda.synchronize()
        t_t = _time_events(ref, x, w, b) * 1e3
        bytes_ = 2 * B * N * 4 + 2 * N * 4
        print(f"{B:5d} {N:6d} {t_p:8.2f}us {t_t:8.2f}us {bw(bytes_, t_p):11.1f} {bw(bytes_, t_t):11.1f} {t_t / t_p:8.2f}x")


def swiglu():
    from examples.blackwell.swiglu import build_fused_silu_mul

    hdr("BLACKWELL FUSED SILU * MUL (fp32)")
    print(f"{'M':>5} {'F':>6} {'pyptx':>9} {'torch':>9} {'pyptx GB/s':>11} {'torch GB/s':>11} {'speedup':>8}")
    for M, F in [(32, 1024), (256, 4096), (1024, 8192), (2048, 8192)]:
        k = build_fused_silu_mul(M, F)
        g = torch.randn(M, F, device="cuda") * 0.5
        u = torch.randn(M, F, device="cuda") * 0.5
        k(g, u)
        torch.cuda.synchronize()
        t_p = _time_events(lambda g, u: k(g, u), g, u) * 1e3

        def ref(g, u):
            return torch.nn.functional.silu(g) * u

        ref(g, u)
        torch.cuda.synchronize()
        t_t = _time_events(ref, g, u) * 1e3
        bytes_ = 3 * M * F * 4
        print(f"{M:5d} {F:6d} {t_p:8.2f}us {t_t:8.2f}us {bw(bytes_, t_p):11.1f} {bw(bytes_, t_t):11.1f} {t_t / t_p:8.2f}x")


def grouped_gemm():
    from examples.blackwell.grouped_gemm import build_grouped_gemm

    hdr("BLACKWELL GROUPED GEMM (bf16 -> fp32)")
    print(f"{'G':>3} {'M':>5} {'N':>4} {'K':>5} {'pyptx':>9} {'torch':>9} {'pyptx TF':>10} {'torch TF':>10} {'speedup':>8}")
    for G, M, N, K in [(8, 512, 128, 512), (8, 1024, 128, 1024), (4, 2048, 256, 2048)]:
        k = build_grouped_gemm(G, M, N, K)
        a3 = torch.randn(G, M, K, device="cuda", dtype=torch.bfloat16) * 0.1
        b3 = torch.randn(G, K, N, device="cuda", dtype=torch.bfloat16) * 0.1
        a2 = a3.reshape(G * M, K)
        bt2 = b3.transpose(1, 2).contiguous().reshape(G * N, K)

        k(a2, bt2)
        torch.cuda.synchronize()
        t_p = _time_events(lambda a, b: k(a, b), a2, bt2) * 1e3

        def ref(a, b):
            return torch.einsum("gmk,gkn->gmn", a.float(), b.float())

        ref(a3, b3)
        torch.cuda.synchronize()
        t_t = _time_events(ref, a3, b3) * 1e3

        flops = 2 * G * M * N * K
        tf_p = flops / (t_p * 1e-6) / 1e12
        tf_t = flops / (t_t * 1e-6) / 1e12
        print(f"{G:3d} {M:5d} {N:4d} {K:5d} {t_p:8.2f}us {t_t:8.2f}us {tf_p:10.1f} {tf_t:10.1f} {t_t / t_p:8.2f}x")


def main():
    dispatch = {
        "rms": rms_norm,
        "layer": layer_norm,
        "silu": swiglu,
        "grouped": grouped_gemm,
    }
    targets = sys.argv[1:]
    if not targets:
        rms_norm()
        layer_norm()
        swiglu()
        grouped_gemm()
        return
    for target in targets:
        dispatch[target]()


if __name__ == "__main__":
    main()
