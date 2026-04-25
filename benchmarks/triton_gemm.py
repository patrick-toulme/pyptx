"""Compile Triton's matmul to PTX, dump it, then load via pyptx."""
import torch
import triton
import triton.language as tl
import time


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)


M = N = K = 2048
a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
b = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

# Compile
matmul_kernel[grid](a, b, c, M, N, K,
    a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64)
torch.cuda.synchronize()

ref = torch.matmul(a, b)
print(f"triton diff: {float((c.float() - ref.float()).abs().max()):g}")

# Benchmark Triton
for _ in range(10):
    matmul_kernel[grid](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(200):
    matmul_kernel[grid](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64)
torch.cuda.synchronize()
t = (time.perf_counter() - t0) / 200
flops = 2 * M * N * K
print(f"triton: {t*1000:.3f}ms {flops/t/1e12:.0f}TFLOPS")

# Benchmark cuBLAS
for _ in range(10): torch.matmul(a, b); torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(200): torch.matmul(a, b)
torch.cuda.synchronize()
t2 = (time.perf_counter() - t0) / 200
print(f"cublas: {t2*1000:.3f}ms {flops/t2/1e12:.0f}TFLOPS")
print(f"triton/cublas: {t/t2:.2f}x")

# Try to get the compiled PTX
try:
    # Triton stores compiled artifacts — find the PTX
    import triton.compiler as tc
    key = matmul_kernel.cache_key
    print(f"cache key type: {type(key)}")
except Exception as e:
    print(f"cache access: {e}")
