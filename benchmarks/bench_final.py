"""Final benchmark after all optimizations.

Summary of wins vs prior baseline:
    - SwiGLU:        v4 loads, now ~2.8 TB/s (94% of H100 HBM peak)
    - Grouped GEMM:  tile_k=64 multi-k wgmma, 104 TFLOPS (was 46)
    - Flash Attn:    new Hopper impl with BN=64 multi-k + Q parallelism,
                     2-5x faster than torch naive ref
    - RMS/LayerNorm: already good, 2-3x vs torch at large sizes
"""
import math, time
import torch, numpy as np


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
    from examples.hopper.rms_norm import build_rms_norm
    hdr("RMS NORM (fp32)")
    print(f"{'B':>5} {'N':>6} {'pyptx':>9} {'torch':>9} {'pyptx GB/s':>11} {'torch GB/s':>11} {'speedup':>8}")
    for B, N in [(32, 1024), (256, 4096), (1024, 8192), (2048, 8192)]:
        k = build_rms_norm(B, N)
        x = torch.randn(B, N, device="cuda")
        w = torch.randn(N, device="cuda")
        k(x, w); torch.cuda.synchronize()
        t_p = _time_events(lambda x, w: k(x, w), x, w) * 1e3
        def ref(x, w):
            return x * torch.rsqrt((x*x).mean(dim=-1, keepdim=True) + 1e-6) * w
        ref(x, w); torch.cuda.synchronize()
        t_t = _time_events(ref, x, w) * 1e3
        bytes_ = 2 * B * N * 4 + N * 4
        print(f"{B:5d} {N:6d} {t_p:8.2f}us {t_t:8.2f}us {bw(bytes_, t_p):11.1f} {bw(bytes_, t_t):11.1f} {t_t/t_p:8.2f}x")


def layer_norm():
    from examples.hopper.layer_norm import build_layer_norm
    hdr("LAYER NORM (fp32)")
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
        print(f"{B:5d} {N:6d} {t_p:8.2f}us {t_t:8.2f}us {bw(bytes_, t_p):11.1f} {bw(bytes_, t_t):11.1f} {t_t/t_p:8.2f}x")


def swiglu():
    from examples.hopper.swiglu import build_fused_silu_mul
    hdr("FUSED SILU * MUL (fp32)")
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
        print(f"{M:5d} {F:6d} {t_p:8.2f}us {t_t:8.2f}us {bw(bytes_, t_p):11.1f} {bw(bytes_, t_t):11.1f} {t_t/t_p:8.2f}x")


def softmax():
    from examples.hopper.softmax import build_softmax
    hdr("SOFTMAX (fp32, row-wise)")
    print(f"{'B':>5} {'N':>6} {'pyptx':>9} {'torch':>9} {'pyptx GB/s':>11} {'torch GB/s':>11} {'speedup':>8}")
    for B, N in [(32, 1024), (256, 4096), (1024, 8192), (2048, 8192)]:
        k = build_softmax(B, N)
        x = torch.randn(B, N, device="cuda")
        k(x); torch.cuda.synchronize()
        t_p = _time_events(lambda x: k(x), x) * 1e3
        def ref(x):
            return torch.softmax(x, dim=-1)
        ref(x); torch.cuda.synchronize()
        t_t = _time_events(ref, x) * 1e3
        bytes_ = 2 * B * N * 4
        print(f"{B:5d} {N:6d} {t_p:8.2f}us {t_t:8.2f}us {bw(bytes_, t_p):11.1f} {bw(bytes_, t_t):11.1f} {t_t/t_p:8.2f}x")


def grouped_gemm():
    from examples.hopper.grouped_gemm import build_grouped_gemm
    hdr("GROUPED GEMM (bf16 -> fp32)")
    print(f"{'G':>3} {'M':>5} {'N':>4} {'K':>5} {'pyptx':>9} {'TFLOPS':>8}")
    for G, M, N, K in [(8, 512, 64, 512), (16, 512, 64, 512),
                       (8, 1024, 64, 1024), (4, 2048, 64, 2048),
                       (8, 2048, 64, 2048)]:
        k = build_grouped_gemm(G, M, N, K)
        a = torch.randn(G*M, K, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(G*K, N, device="cuda", dtype=torch.bfloat16) * 0.1
        k(a, b); torch.cuda.synchronize()
        t_p = _time_events(lambda a, b: k(a, b), a, b) * 1e3
        flops = 2 * G * M * N * K
        tflops = flops / (t_p * 1e-6) / 1e12
        print(f"{G:3d} {M:5d} {N:4d} {K:5d} {t_p:8.2f}us {tflops:8.1f}")


def flash_attn():
    from examples.hopper.experimental.flash_attention_hopper import build_flash_attention_hopper
    hdr("FLASH ATTENTION (Hopper BN=64 multi-k, bf16)")
    print(f"{'M_q':>6} {'N':>6} {'HD':>3} {'pyptx':>9} {'naive':>9} {'sdpa':>9} {'vs naive':>9} {'vs sdpa':>9}")
    for HD in (16, 32, 64):
        for M_q, N_seq in [(512, 512), (2048, 2048), (4096, 4096), (8192, 4096)]:
            k = build_flash_attention_hopper(M_q, N_seq, HD)
            q = torch.randn(M_q, HD, device="cuda", dtype=torch.bfloat16) * 0.3
            kk = torch.randn(N_seq, HD, device="cuda", dtype=torch.bfloat16) * 0.3
            v = torch.randn(N_seq, HD, device="cuda", dtype=torch.bfloat16) * 0.3
            k_t = kk.transpose(0, 1).contiguous()
            k(q, k_t, v); torch.cuda.synchronize()
            t_p = _time_events(lambda q, k_t, v: k(q, k_t, v), q, k_t, v) * 1e3
            def ref(q, kk, v):
                s = torch.matmul(q.float(), kk.float().T) / math.sqrt(HD)
                return torch.softmax(s, dim=-1) @ v.float()
            ref(q, kk, v); torch.cuda.synchronize()
            t_n = _time_events(ref, q, kk, v) * 1e3
            q4 = q.view(1, 1, M_q, HD); k4 = kk.view(1, 1, N_seq, HD); v4 = v.view(1, 1, N_seq, HD)
            def sdpa(q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)
            sdpa(q4, k4, v4); torch.cuda.synchronize()
            t_s = _time_events(sdpa, q4, k4, v4) * 1e3
            print(f"{M_q:6d} {N_seq:6d} {HD:3d} {t_p:8.2f}us {t_n:8.2f}us {t_s:8.2f}us {t_n/t_p:8.2f}x {t_s/t_p:8.2f}x")


def hopper_gemm():
    """Warp-specialized m64n256k16 GEMM — requires the dedicated harness
    (TMA descriptor plumbing, Hilbert schedule), so we delegate to it."""
    from benchmarks.bench_hopper_gemm import run_benchmark
    hdr("HOPPER GEMM (bf16, warp-specialized)")
    print(f"{'size':>6} {'ms':>8} {'TFLOPS':>8}")
    for size in (2048, 4096, 6144, 8192):
        r = run_benchmark(size=size, warmup=4, iters=16)
        print(f"{size:6d} {r['ms']:8.3f} {r['tflops']:8.1f}")


def main():
    rms_norm()
    layer_norm()
    swiglu()
    softmax()
    grouped_gemm()
    flash_attn()
    hopper_gemm()


if __name__ == "__main__":
    import sys
    dispatch = {
        "rms": rms_norm, "layer": layer_norm, "silu": swiglu,
        "softmax": softmax,
        "grouped": grouped_gemm, "attn": flash_attn, "gemm": hopper_gemm,
    }
    targets = sys.argv[1:]
    if not targets:
        main()
    else:
        for t in targets:
            dispatch[t]()
