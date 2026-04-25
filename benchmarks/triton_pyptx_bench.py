"""Benchmark: Triton PTX → pyptx round-trip → driver JIT → cuLaunchKernel.

Proves pyptx can:
1. Parse Triton's compiled matmul PTX byte-identically
2. Emit it back (round-trip)
3. Load via cuModuleLoadData (same as pyptx's compile path)
4. Launch via cuLaunchKernel at full speed
5. Transpile to pyptx DSL code

Performance should be identical to Triton's native launch since it's
the same PTX → same cubin → same SASS.

Usage (on H100):
    python benchmarks/triton_pyptx_bench.py
"""

import ctypes
import os
import sys
import time

import torch
import triton
import triton.language as tl
from cuda.bindings import driver

# Add pyptx to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyptx.parser import parse
from pyptx.emitter import emit
from pyptx.codegen.codegen import ptx_to_python


# ---------------------------------------------------------------------------
# 1. Triton matmul kernel (standard from tutorials)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. Setup
# ---------------------------------------------------------------------------

M = N = K = 2048
BM, BN, BK = 128, 128, 64

a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
b = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
c_pyptx = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

# Warm up Triton (compiles the kernel)
matmul_kernel[grid](a, b, c, M, N, K,
    a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK)
torch.cuda.synchronize()

# Verify correctness
ref = torch.matmul(a, b)
print(f"Triton diff from cuBLAS: {float((c.float() - ref.float()).abs().max()):g}")


# ---------------------------------------------------------------------------
# 3. Extract Triton's compiled PTX
# ---------------------------------------------------------------------------

print("\n--- Extracting Triton PTX ---")

# Triton stores compiled kernels in its cache
ptx_source = None
kernel_name = None
num_warps = None
shared_mem = None

# Method 1: iterate over the cache
for key, compiled_kernel in matmul_kernel.cache.items():
    try:
        ptx_source = compiled_kernel.asm.get("ptx", None)
        if ptx_source is None:
            ptx_source = compiled_kernel.asm.get("source", None)
    except (AttributeError, TypeError):
        pass

    # Try to get metadata
    try:
        metadata = compiled_kernel.metadata
        num_warps = getattr(metadata, "num_warps", None)
        shared_mem = getattr(metadata, "shared", None)
        kernel_name = getattr(compiled_kernel, "name", None)
        if kernel_name is None:
            kernel_name = getattr(metadata, "name", None)
    except (AttributeError, TypeError):
        pass

    if ptx_source is not None:
        break

if ptx_source is None:
    # Method 2: try the compiled_kernel directly
    try:
        compiled_kernel = list(matmul_kernel.cache.values())[0]
        ptx_source = compiled_kernel.asm["ptx"]
    except Exception:
        pass

if ptx_source is None:
    print("ERROR: Could not extract PTX from Triton cache")
    print(f"Cache keys: {list(matmul_kernel.cache.keys())}")
    print(f"Cache type: {type(matmul_kernel.cache)}")
    for key, val in matmul_kernel.cache.items():
        print(f"  key={key}, val type={type(val)}")
        if hasattr(val, 'asm'):
            print(f"  asm keys: {list(val.asm.keys()) if isinstance(val.asm, dict) else type(val.asm)}")
        if hasattr(val, '__dict__'):
            print(f"  attrs: {list(val.__dict__.keys())}")
    sys.exit(1)

# Save the raw PTX
ptx_path = "/tmp/triton_gemm.ptx"
with open(ptx_path, "w") as f:
    f.write(ptx_source)

print(f"PTX size: {len(ptx_source)} bytes, {len(ptx_source.splitlines())} lines")
print(f"Saved to: {ptx_path}")
if kernel_name:
    print(f"Kernel name: {kernel_name}")
if num_warps:
    print(f"Num warps: {num_warps}")
if shared_mem:
    print(f"Shared memory: {shared_mem} bytes")

# Extract entry name from PTX if not found above
if kernel_name is None:
    for line in ptx_source.splitlines():
        s = line.strip()
        if ".entry" in s:
            after = s.split(".entry", 1)[1].lstrip()
            name = ""
            for ch in after:
                if ch.isalnum() or ch in "_$":
                    name += ch
                else:
                    break
            if name:
                kernel_name = name
                break

if kernel_name is None:
    print("ERROR: Could not determine kernel entry name")
    sys.exit(1)

print(f"Entry point: {kernel_name}")


# ---------------------------------------------------------------------------
# 4. Round-trip through pyptx parser/emitter
# ---------------------------------------------------------------------------

print("\n--- pyptx Round-Trip ---")

t0 = time.perf_counter()
module = parse(ptx_source)
parse_time = time.perf_counter() - t0

t0 = time.perf_counter()
emitted = emit(module)
emit_time = time.perf_counter() - t0

roundtrip_ok = (emitted == ptx_source)
print(f"Parse time:   {parse_time*1000:.1f}ms")
print(f"Emit time:    {emit_time*1000:.1f}ms")
print(f"Round-trip:   {'BYTE-IDENTICAL' if roundtrip_ok else 'DIFF (see below)'}")

if not roundtrip_ok:
    # Show first difference
    orig_lines = ptx_source.splitlines()
    emit_lines = emitted.splitlines()
    for i, (a_line, b_line) in enumerate(zip(orig_lines, emit_lines)):
        if a_line != b_line:
            print(f"  First diff at line {i+1}:")
            print(f"    orig: {a_line!r}")
            print(f"    emit: {b_line!r}")
            break
    if len(orig_lines) != len(emit_lines):
        print(f"  Line count: orig={len(orig_lines)}, emit={len(emit_lines)}")

# Use the emitted PTX (proves the round-trip) for loading
load_ptx = emitted if roundtrip_ok else ptx_source


# ---------------------------------------------------------------------------
# 5. Load via cuModuleLoadData (pyptx's compile path)
# ---------------------------------------------------------------------------

print("\n--- Loading PTX via CUDA Driver API ---")

driver.cuInit(0)

err, cu_module = driver.cuModuleLoadData(load_ptx.encode())
assert err == driver.CUresult.CUDA_SUCCESS, f"cuModuleLoadData: {err}"

err, cu_function = driver.cuModuleGetFunction(cu_module, kernel_name.encode())
assert err == driver.CUresult.CUDA_SUCCESS, f"cuModuleGetFunction({kernel_name}): {err}"

print(f"CUfunction: 0x{int(cu_function):x}")

# Set dynamic shared memory if needed
if shared_mem and shared_mem > 0:
    attr = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    err, = driver.cuFuncSetAttribute(cu_function, attr, shared_mem)
    if err != driver.CUresult.CUDA_SUCCESS:
        print(f"WARNING: cuFuncSetAttribute(SMEM={shared_mem}) failed: {err}")
    else:
        print(f"Dynamic SMEM set to {shared_mem} bytes")


# ---------------------------------------------------------------------------
# 6. Launch via cuLaunchKernel
# ---------------------------------------------------------------------------

print("\n--- Direct Launch via cuLaunchKernel ---")

# Build kernel params matching Triton's signature:
# (a_ptr:u64, b_ptr:u64, c_ptr:u64, M:i32, N:i32, K:i32,
#  stride_am:i32, stride_ak:i32, stride_bk:i32, stride_bn:i32,
#  stride_cm:i32, stride_cn:i32)
a_ptr = a.data_ptr()
b_ptr = b.data_ptr()
c_ptr = c_pyptx.data_ptr()

# Pack params: 3 u64 pointers + 9 i32 values
param_ptrs = (ctypes.c_uint64 * 3)(a_ptr, b_ptr, c_ptr)
param_i32s = (ctypes.c_int32 * 9)(
    M, N, K,
    a.stride(0), a.stride(1),  # stride_am, stride_ak
    b.stride(0), b.stride(1),  # stride_bk, stride_bn
    c_pyptx.stride(0), c_pyptx.stride(1),  # stride_cm, stride_cn
)

# Build void* array for kernelParams
# Layout: [&ptr0, &ptr1, &ptr2, &M, &N, &K, &s_am, &s_ak, &s_bk, &s_bn, &s_cm, &s_cn]
param_void_ptrs = (ctypes.c_void_p * 12)()
for i in range(3):
    param_void_ptrs[i] = ctypes.cast(
        ctypes.byref(param_ptrs, i * 8), ctypes.c_void_p
    ).value
for i in range(9):
    param_void_ptrs[3 + i] = ctypes.cast(
        ctypes.byref(param_i32s, i * 4), ctypes.c_void_p
    ).value

kernel_params_addr = ctypes.addressof(param_void_ptrs)

# Grid: same as Triton
grid_x = triton.cdiv(M, BM) * triton.cdiv(N, BN)
block_x = (num_warps or 4) * 32
smem = shared_mem or 0

print(f"Grid: ({grid_x}, 1, 1), Block: ({block_x}, 1, 1), SMEM: {smem}")

# Get the current CUDA stream
stream = torch.cuda.current_stream()
cu_stream = stream.cuda_stream

# Launch!
err, = driver.cuLaunchKernel(
    cu_function,
    grid_x, 1, 1,
    block_x, 1, 1,
    smem,
    cu_stream,
    kernel_params_addr,
    0,
)
assert err == driver.CUresult.CUDA_SUCCESS, f"cuLaunchKernel: {err}"
torch.cuda.synchronize()

# Verify correctness
diff = float((c_pyptx.float() - ref.float()).abs().max())
print(f"pyptx-launched diff from cuBLAS: {diff:g}")
bit_match = torch.equal(c, c_pyptx)
print(f"Bit-identical to Triton result: {bit_match}")


# ---------------------------------------------------------------------------
# 7. Benchmark: Triton native vs pyptx-launched vs cuBLAS
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("BENCHMARK: 2048x2048x2048 bf16 matmul")
print("="*60)

WARMUP = 20
ITERS = 200
flops = 2 * M * N * K


def bench_triton():
    for _ in range(WARMUP):
        matmul_kernel[grid](a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        matmul_kernel[grid](a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS


def bench_pyptx_launch():
    for _ in range(WARMUP):
        driver.cuLaunchKernel(
            cu_function, grid_x, 1, 1, block_x, 1, 1,
            smem, cu_stream, kernel_params_addr, 0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        driver.cuLaunchKernel(
            cu_function, grid_x, 1, 1, block_x, 1, 1,
            smem, cu_stream, kernel_params_addr, 0)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS


def bench_cublas():
    for _ in range(WARMUP):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS


t_triton = bench_triton()
t_pyptx = bench_pyptx_launch()
t_cublas = bench_cublas()

print(f"\n{'Method':<25} {'Time (ms)':>10} {'TFLOPS':>10} {'vs cuBLAS':>10}")
print("-" * 60)
print(f"{'cuBLAS':<25} {t_cublas*1000:10.3f} {flops/t_cublas/1e12:10.0f} {'1.00x':>10}")
print(f"{'Triton (native)':<25} {t_triton*1000:10.3f} {flops/t_triton/1e12:10.0f} {t_triton/t_cublas:10.2f}x")
print(f"{'pyptx (Triton PTX)':<25} {t_pyptx*1000:10.3f} {flops/t_pyptx/1e12:10.0f} {t_pyptx/t_cublas:10.2f}x")

pyptx_vs_triton = t_pyptx / t_triton
print(f"\npyptx overhead vs Triton native: {(pyptx_vs_triton - 1)*100:+.1f}%")


# ---------------------------------------------------------------------------
# 8. Transpile to pyptx DSL
# ---------------------------------------------------------------------------

print("\n--- Transpiling to pyptx DSL ---")
t0 = time.perf_counter()
dsl_code = ptx_to_python(ptx_source)
transpile_time = time.perf_counter() - t0
print(f"Transpile time: {transpile_time*1000:.1f}ms")
print(f"DSL code: {len(dsl_code)} bytes, {len(dsl_code.splitlines())} lines")

dsl_path = "/tmp/triton_gemm_pyptx.py"
with open(dsl_path, "w") as f:
    f.write(dsl_code)
print(f"Saved to: {dsl_path}")

# Show first 30 lines
print("\n--- DSL Preview (first 30 lines) ---")
for i, line in enumerate(dsl_code.splitlines()[:30], 1):
    print(f"  {i:3d}  {line}")
print("  ...")


# ---------------------------------------------------------------------------
# 9. Larger sizes
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("SCALING: larger matrix sizes")
print("="*60)

for sz in [4096, 8192]:
    a_big = torch.randn((sz, sz), device="cuda", dtype=torch.bfloat16)
    b_big = torch.randn((sz, sz), device="cuda", dtype=torch.bfloat16)
    c_big = torch.empty((sz, sz), device="cuda", dtype=torch.bfloat16)
    flops_big = 2 * sz * sz * sz

    # Update params for the pyptx launch
    p_ptrs_big = (ctypes.c_uint64 * 3)(a_big.data_ptr(), b_big.data_ptr(), c_big.data_ptr())
    p_i32_big = (ctypes.c_int32 * 9)(
        sz, sz, sz,
        a_big.stride(0), a_big.stride(1),
        b_big.stride(0), b_big.stride(1),
        c_big.stride(0), c_big.stride(1),
    )
    pvp_big = (ctypes.c_void_p * 12)()
    for i in range(3):
        pvp_big[i] = ctypes.cast(ctypes.byref(p_ptrs_big, i * 8), ctypes.c_void_p).value
    for i in range(9):
        pvp_big[3 + i] = ctypes.cast(ctypes.byref(p_i32_big, i * 4), ctypes.c_void_p).value
    kp_big = ctypes.addressof(pvp_big)
    grid_big = triton.cdiv(sz, BM) * triton.cdiv(sz, BN)

    # cuBLAS
    for _ in range(10):
        torch.matmul(a_big, b_big)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        torch.matmul(a_big, b_big)
    torch.cuda.synchronize()
    t_cb = (time.perf_counter() - t0) / 50

    # pyptx launch
    for _ in range(10):
        driver.cuLaunchKernel(cu_function, grid_big, 1, 1, block_x, 1, 1,
                              smem, cu_stream, kp_big, 0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        driver.cuLaunchKernel(cu_function, grid_big, 1, 1, block_x, 1, 1,
                              smem, cu_stream, kp_big, 0)
    torch.cuda.synchronize()
    t_pp = (time.perf_counter() - t0) / 50

    print(f"\n{sz}x{sz}x{sz}:")
    print(f"  cuBLAS:          {t_cb*1000:.3f}ms  {flops_big/t_cb/1e12:.0f} TFLOPS")
    print(f"  pyptx (Triton):  {t_pp*1000:.3f}ms  {flops_big/t_pp/1e12:.0f} TFLOPS  ({t_pp/t_cb:.2f}x)")


print("\n" + "="*60)
print("DONE")
print("="*60)
