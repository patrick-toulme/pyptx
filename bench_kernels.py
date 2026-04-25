"""Benchmark pyptx kernels vs JAX/PyTorch references on H100."""
import time
import numpy as np
import jax
import jax.numpy as jnp
import torch

def bench_fn(fn, *args, warmup=5, iters=100):
    for _ in range(warmup):
        fn(*args)
    if hasattr(args[0], 'block_until_ready'):
        fn(*args).block_until_ready()
    else:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    if hasattr(args[0], 'block_until_ready'):
        out.block_until_ready()
    else:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    return elapsed * 1e6  # microseconds


def bench_torch_fn(fn, *args, warmup=5, iters=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    return elapsed * 1e6


print("=" * 70)
print("RMS NORM")
print("=" * 70)
from examples.hopper.rms_norm import build_rms_norm, rms_norm_ref

for B, N in [(32, 1024), (128, 2048), (256, 4096), (512, 8192)]:
    k = build_rms_norm(B, N)
    x = jnp.asarray(np.random.randn(B, N).astype(np.float32) * 0.3)
    w = jnp.asarray((np.random.randn(N) * 0.1 + 1.0).astype(np.float32))

    @jax.jit
    def pyptx_rms(x, w):
        return k(x, w)

    @jax.jit
    def jax_rms(x, w):
        return rms_norm_ref(x, w)

    t_pyptx = bench_fn(pyptx_rms, x, w)
    t_jax = bench_fn(jax_rms, x, w)

    xt = torch.tensor(np.asarray(x), device="cuda")
    wt = torch.tensor(np.asarray(w), device="cuda")
    t_pyptx_torch = bench_torch_fn(k, xt, wt)
    t_torch = bench_torch_fn(lambda x, w: x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6) * w, xt, wt)

    print(f"B={B:4d} N={N:5d}  pyptx={t_pyptx:7.1f}us  jax={t_jax:7.1f}us  speedup={t_jax/t_pyptx:.2f}x | torch_pyptx={t_pyptx_torch:7.1f}us  torch={t_torch:7.1f}us  speedup={t_torch/t_pyptx_torch:.2f}x")


print("\n" + "=" * 70)
print("LAYER NORM")
print("=" * 70)
from examples.hopper.layer_norm import build_layer_norm, layer_norm_ref

for B, N in [(32, 1024), (128, 2048), (256, 4096), (512, 8192)]:
    k = build_layer_norm(B, N)
    x = jnp.asarray(np.random.randn(B, N).astype(np.float32))
    w = jnp.asarray((np.random.randn(N) * 0.1 + 1.0).astype(np.float32))
    b = jnp.asarray((np.random.randn(N) * 0.1).astype(np.float32))

    @jax.jit
    def pyptx_ln(x, w, b):
        return k(x, w, b)

    @jax.jit
    def jax_ln(x, w, b):
        return layer_norm_ref(x, w, b)

    t_pyptx = bench_fn(pyptx_ln, x, w, b)
    t_jax = bench_fn(jax_ln, x, w, b)

    xt = torch.tensor(np.asarray(x), device="cuda")
    wt = torch.tensor(np.asarray(w), device="cuda")
    bt = torch.tensor(np.asarray(b), device="cuda")
    t_pyptx_torch = bench_torch_fn(k, xt, wt, bt)
    t_torch = bench_torch_fn(lambda x, w, b: torch.nn.functional.layer_norm(x, (N,), w, b), xt, wt, bt)

    print(f"B={B:4d} N={N:5d}  pyptx={t_pyptx:7.1f}us  jax={t_jax:7.1f}us  speedup={t_jax/t_pyptx:.2f}x | torch_pyptx={t_pyptx_torch:7.1f}us  torch={t_torch:7.1f}us  speedup={t_torch/t_pyptx_torch:.2f}x")


print("\n" + "=" * 70)
print("GROUPED GEMM")
print("=" * 70)
from examples.hopper.grouped_gemm import build_grouped_gemm

for G, M, N, K in [(4, 256, 32, 128), (8, 512, 64, 512), (16, 512, 64, 512), (8, 1024, 64, 1024), (4, 2048, 64, 2048)]:
    k_fn = build_grouped_gemm(G, M, N, K)
    a3 = jnp.asarray((np.random.randn(G, M, K) * 0.1).astype(np.float32), dtype=jnp.bfloat16)
    b3 = jnp.asarray((np.random.randn(G, K, N) * 0.1).astype(np.float32), dtype=jnp.bfloat16)
    a2 = a3.reshape(G * M, K)
    b2 = b3.reshape(G * K, N)

    @jax.jit
    def pyptx_gemm(a, b):
        return k_fn(a, b)

    @jax.jit
    def jax_gemm(a3, b3):
        return jnp.einsum("gmk,gkn->gmn", a3, b3, preferred_element_type=jnp.float32)

    t_pyptx = bench_fn(pyptx_gemm, a2, b2)
    t_jax = bench_fn(jax_gemm, a3, b3)

    flops = 2 * G * M * N * K
    print(f"G={G:2d} M={M:4d} N={N:4d} K={K:4d}  pyptx={t_pyptx:7.1f}us  jax={t_jax:7.1f}us  speedup={t_jax/t_pyptx:.2f}x  TFLOPS_pyptx={flops/t_pyptx/1e6:.1f}  TFLOPS_jax={flops/t_jax/1e6:.1f}")


print("\n" + "=" * 70)
print("FLASH ATTENTION")
print("=" * 70)
from examples.hopper.flash_attention_wgmma_kloop import build_flash_attention_kloop, attention_ref, BM, HEAD_DIM

for N_seq in [64, 128, 256, 512, 1024, 2048, 4096]:
    k_fn = build_flash_attention_kloop(N_seq)
    q = jnp.asarray((np.random.randn(BM, HEAD_DIM) * 0.3).astype(np.float32), dtype=jnp.bfloat16)
    k_np = (np.random.randn(N_seq, HEAD_DIM) * 0.3).astype(np.float32)
    k_arr = jnp.asarray(k_np, dtype=jnp.bfloat16)
    v = jnp.asarray((np.random.randn(N_seq, HEAD_DIM) * 0.3).astype(np.float32), dtype=jnp.bfloat16)
    k_t = jnp.asarray(np.ascontiguousarray(np.asarray(k_arr).T), dtype=jnp.bfloat16)

    @jax.jit
    def pyptx_attn(q, k_t, v):
        return k_fn(q, k_t, v)

    @jax.jit
    def jax_attn(q, k, v):
        return attention_ref(q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32))

    t_pyptx = bench_fn(pyptx_attn, q, k_t, v)
    t_jax = bench_fn(jax_attn, q, k_arr, v)

    print(f"N_seq={N_seq:5d}  pyptx={t_pyptx:7.1f}us  jax={t_jax:7.1f}us  speedup={t_jax/t_pyptx:.2f}x")
