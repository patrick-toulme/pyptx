"""Directly launch a PTX file via the CUDA driver, bypassing pyptx.

Takes the path as argv[1]. Expected kernel signature:
    kloop_gemm(A, B, C, A_tma_desc, B_tma_desc)    for NVRTC
    gemm(A, B, C, A_tma_desc, B_tma_desc)          for pyptx

Both expect: A (64,32) bf16, B (32,8) bf16, C (64,8) f32, TMA descs on device.
"""
import ctypes
import sys

import jax
import jax.numpy as jnp
import numpy as np
from cuda.bindings import driver

PTX_PATH = sys.argv[1]
KERNEL_NAME = sys.argv[2].encode() if len(sys.argv) > 2 else b"kloop_gemm"

# Force CUDA init via JAX
_ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()

with open(PTX_PATH) as f:
    ptx = f.read().rstrip("\x00").rstrip() + "\n"

err, module = driver.cuModuleLoadData(ptx.encode())
assert err == driver.CUresult.CUDA_SUCCESS, f"cuModuleLoadData: {err}"
err, kernel = driver.cuModuleGetFunction(module, KERNEL_NAME)
assert err == driver.CUresult.CUDA_SUCCESS, f"cuModuleGetFunction({KERNEL_NAME}): {err}"
print(f"loaded {KERNEL_NAME.decode()}: cu_fn = 0x{int(kernel):x}", flush=True)

# Allocate via JAX
M, K, N = 64, 32, 8
np.random.seed(1)
a_np = (np.random.randn(M, K) * 0.1).astype(np.float32)
b_np = (np.random.randn(K, N) * 0.1).astype(np.float32)
A = jnp.asarray(a_np, dtype=jnp.bfloat16)
B = jnp.asarray(b_np, dtype=jnp.bfloat16)
C = jnp.zeros((M, N), dtype=jnp.float32)
A.block_until_ready(); B.block_until_ready(); C.block_until_ready()

A_ptr = int(A.unsafe_buffer_pointer())
B_ptr = int(B.unsafe_buffer_pointer())
C_ptr = int(C.unsafe_buffer_pointer())

# TMA descriptors
u64 = driver.cuuint64_t
u32 = driver.cuuint32_t


def make_desc(data_ptr, rows, cols, box_rows, box_cols, swizzle):
    err, tmap = driver.cuTensorMapEncodeTiled(
        driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        data_ptr,
        [u64(cols), u64(rows)],
        [u64(cols * 2)],
        [u32(box_cols), u32(box_rows)],
        [u32(1), u32(1)],
        driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    assert err == driver.CUresult.CUDA_SUCCESS, err
    return tmap


# A: (64, 32) bf16 row-major. Box = (16 cols, 64 rows)  -> a 64x16 tile
A_tmap = make_desc(A_ptr, 64, 32, 64, 16, driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B)
# B: (32, 8) bf16 row-major. Box = (8 cols, 16 rows) -> a 16x8 tile
B_tmap = make_desc(B_ptr, 32, 8, 16, 8, driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE)

# Pin descriptors to device memory
A_tmap_host_ptr = int(A_tmap.getPtr())
B_tmap_host_ptr = int(B_tmap.getPtr())
err, A_dev_desc = driver.cuMemAlloc(128); assert err == driver.CUresult.CUDA_SUCCESS
err, B_dev_desc = driver.cuMemAlloc(128); assert err == driver.CUresult.CUDA_SUCCESS
A_host_blob = ctypes.string_at(A_tmap_host_ptr, 128)
B_host_blob = ctypes.string_at(B_tmap_host_ptr, 128)
err, = driver.cuMemcpyHtoD(A_dev_desc, A_host_blob, 128); assert err == driver.CUresult.CUDA_SUCCESS
err, = driver.cuMemcpyHtoD(B_dev_desc, B_host_blob, 128); assert err == driver.CUresult.CUDA_SUCCESS

# Build kernel params: 5 u64 values packed into a ctypes array.
# cuda-python wants kernelParams as a single int: address of an array of void*,
# each void* pointing to the corresponding param value.
vals = (ctypes.c_uint64 * 5)(A_ptr, B_ptr, C_ptr, int(A_dev_desc), int(B_dev_desc))
ptrs = (ctypes.c_void_p * 5)(
    *[ctypes.cast(ctypes.byref(vals, i * 8), ctypes.c_void_p) for i in range(5)]
)
kernel_params_addr = ctypes.addressof(ptrs)

print("launching...", flush=True)
err, = driver.cuLaunchKernel(
    kernel,
    1, 1, 1,        # grid
    128, 1, 1,      # block
    0,              # smem bytes
    0,              # stream (default)
    kernel_params_addr,
    0,              # extra
)
print(f"launch err: {err}", flush=True)
err, = driver.cuCtxSynchronize()
print(f"sync err:   {err}", flush=True)

out = np.asarray(C)
ref = np.asarray(
    jax.lax.dot_general(A, B, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
)
print(f"max diff:   {float(np.abs(out - ref).max())}")
print(f"bit-exact:  {bool(np.array_equal(out, ref))}")
print(f"out[0]:     {out[0]}")
print(f"ref[0]:     {ref[0]}")
