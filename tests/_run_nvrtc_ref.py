"""Actually run the NVRTC-compiled reference kernel to prove the
pattern works — or doesn't.

Allocates buffers, synthesizes TMA descriptors, and launches via
cuLaunchKernel directly. Bypasses pyptx codegen entirely; this is the
"golden" PTX.
"""
import ctypes
import jax, jax.numpy as jnp, numpy as np
from cuda.bindings import driver, nvrtc

# Force CUDA init
_ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()

# Load the PTX NVRTC produced
with open("/tmp/nvrtc_kloop.ptx") as f:
    ptx = f.read()

err, module = driver.cuModuleLoadData(ptx.encode())
assert err == driver.CUresult.CUDA_SUCCESS, err
err, kernel = driver.cuModuleGetFunction(module, b"kloop_gemm")
assert err == driver.CUresult.CUDA_SUCCESS, err
print(f"loaded kernel: cu_fn = 0x{int(kernel):x}")

# Allocate input/output via JAX
M, K, N = 64, 32, 8
np.random.seed(1)
a_np = (np.random.randn(M, K) * 0.1).astype(np.float32)
b_np = (np.random.randn(K, N) * 0.1).astype(np.float32)
A = jnp.asarray(a_np, dtype=jnp.bfloat16)
B = jnp.asarray(b_np, dtype=jnp.bfloat16)
C = jnp.zeros((M, N), dtype=jnp.float32)
A.block_until_ready()
B.block_until_ready()
C.block_until_ready()

A_ptr = int(A.unsafe_buffer_pointer())
B_ptr = int(B.unsafe_buffer_pointer())
C_ptr = int(C.unsafe_buffer_pointer())

# Synthesize TMA descriptors
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

# A: (64, 32) bf16 row-major. Box = (16 cols, 64 rows) = 16 K-slice
A_tmap = make_desc(A_ptr, 64, 32, 64, 16,
                   driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B)
# B: (32, 8) bf16 row-major. Box = (8 cols, 16 rows)
B_tmap = make_desc(B_ptr, 32, 8, 16, 8,
                   driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE)

# Pin the descriptors to device memory (each is 128 bytes aligned)
import ctypes
A_tmap_host_ptr = int(A_tmap.getPtr())
B_tmap_host_ptr = int(B_tmap.getPtr())

err, A_dev_desc = driver.cuMemAlloc(128)
err, B_dev_desc = driver.cuMemAlloc(128)

A_host_blob = ctypes.string_at(A_tmap_host_ptr, 128)
B_host_blob = ctypes.string_at(B_tmap_host_ptr, 128)

err, = driver.cuMemcpyHtoD(A_dev_desc, A_host_blob, 128)
err, = driver.cuMemcpyHtoD(B_dev_desc, B_host_blob, 128)

# Build kernel params: 5 u64 values
params_array = (ctypes.c_uint64 * 5)(A_ptr, B_ptr, C_ptr, int(A_dev_desc), int(B_dev_desc))
kernel_param_ptrs = (ctypes.c_void_p * 5)(
    *[ctypes.addressof(ctypes.c_uint64.from_buffer(params_array, i*8)) for i in range(5)]
)

print("launching kernel...", flush=True)
err, = driver.cuLaunchKernel(
    kernel,
    1, 1, 1,       # grid
    128, 1, 1,     # block
    0,             # smem bytes (static)
    None,          # stream (default)
    kernel_param_ptrs,
    None,
)
print("launch err:", err)
err, = driver.cuCtxSynchronize()
print("sync err:", err)

out = np.asarray(C)
ref = np.asarray(
    jax.lax.dot_general(A, B, (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32)
)
print(f"max diff: {float(np.abs(out - ref).max())}")
print(f"bit-exact: {bool(np.array_equal(out, ref))}")
print(f"out[0]: {out[0]}")
print(f"ref[0]: {ref[0]}")
