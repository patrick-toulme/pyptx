"""Use NVRTC to compile a reference CUDA C++ kernel with inline asm,
then dump the PTX so we can see the canonical K-loop wgmma pattern.

We write a 2-iteration K-loop wgmma by hand in inline asm, mimicking
what CUTLASS would generate. NVRTC compiles it with full
optimization, so ptxas will have resolved any hazards correctly.
"""
import jax, jax.numpy as jnp, numpy as np
from cuda.bindings import driver, nvrtc

# Force JAX to init CUDA first so our cuModuleLoadData runs in the
# right context.
_ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()

CUDA_SRC = r"""
// Avoid libc++/libstdc++ headers (NVRTC can't find them without cudatoolkit
// in the include path). Use raw typedefs and ignore bf16 type semantics —
// inline asm handles everything.
typedef unsigned int  uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned short __nv_bfloat16;
struct CUtensorMap { unsigned long long opaque[16]; };

// 2-iteration K-loop wgmma GEMM:
//   A: 64 x 32 bf16 row-major
//   B: 32 x 8  bf16 row-major
//   C: 64 x 8  f32  row-major
//
// Single CTA, single warpgroup (128 threads).
// Two K=16 slices. One SMEM buffer reused.

extern "C" __global__ __launch_bounds__(128, 1)
void kloop_gemm(
    __nv_bfloat16 const* __restrict__ A_global,
    __nv_bfloat16 const* __restrict__ B_global,
    float* __restrict__ C_global,
    const CUtensorMap* __restrict__ A_desc,
    const CUtensorMap* __restrict__ B_desc
) {
    // 64x16 bf16 = 2048 bytes, aligned to 128B for TMA
    __shared__ alignas(128) __nv_bfloat16 sA[64 * 16];
    __shared__ alignas(128) __nv_bfloat16 sB[16 * 8];
    __shared__ alignas(8)   uint64_t      bar[2];

    const int tid = threadIdx.x;

    // Init mbarriers. Only thread 0.
    if (tid == 0) {
        asm volatile(
            "mbarrier.init.shared.b64 [%0], 1;\n"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(&bar[0]))
        );
        asm volatile(
            "mbarrier.init.shared.b64 [%0], 1;\n"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(&bar[1]))
        );
        asm volatile("fence.proxy.async.shared::cta;\n");
    }
    __syncthreads();

    float acc[4] = {0, 0, 0, 0};
    uint32_t phase0 = 0;
    uint32_t phase1 = 0;

    // ===== Iteration 0 =====
    if (tid == 0) {
        uint32_t bar_addr = (uint32_t)__cvta_generic_to_shared(&bar[0]);
        uint32_t sA_addr  = (uint32_t)__cvta_generic_to_shared(sA);
        uint32_t sB_addr  = (uint32_t)__cvta_generic_to_shared(sB);

        // arrive_expect_tx
        asm volatile(
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
            :
            : "r"(bar_addr), "r"(64*16*2 + 16*8*2)
        );

        // TMA load A[0..63, 0..15]
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global"
            ".mbarrier::complete_tx::bytes [%0], [%1, {0, 0}], [%2];\n"
            :
            : "r"(sA_addr), "l"(A_desc), "r"(bar_addr)
        );
        // TMA load B[0..15, 0..7]
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global"
            ".mbarrier::complete_tx::bytes [%0], [%1, {0, 0}], [%2];\n"
            :
            : "r"(sB_addr), "l"(B_desc), "r"(bar_addr)
        );
    }
    __syncthreads();

    // Wait on bar0
    {
        uint32_t bar_addr = (uint32_t)__cvta_generic_to_shared(&bar[0]);
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "loop0:\n"
            "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
            "@!p bra loop0;\n"
            "}\n"
            :
            : "r"(bar_addr), "r"(phase0)
        );
    }

    // wgmma iter 0
    {
        uint32_t sA_addr = (uint32_t)__cvta_generic_to_shared(sA);
        uint32_t sB_addr = (uint32_t)__cvta_generic_to_shared(sB);
        // Build descriptors
        uint64_t desc_a = ((uint64_t)(sA_addr >> 4) & 0x3FFFULL)
                        | ((uint64_t)1 << 16)       // LBO = 16 bytes
                        | ((uint64_t)16 << 32)      // SBO = 256 bytes
                        | ((uint64_t)3 << 62);      // B32 swizzle
        uint64_t desc_b = ((uint64_t)(sB_addr >> 4) & 0x3FFFULL)
                        | ((uint64_t)8 << 16)       // LBO = 128 bytes
                        | ((uint64_t)1 << 32)       // SBO = 16 bytes
                        | ((uint64_t)0 << 62);      // INTERLEAVE

        asm volatile("wgmma.fence.sync.aligned;\n");
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %4, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
            "{%0, %1, %2, %3}, %5, %6, p, 1, 1, 0, 1;\n"
            "}\n"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(0), "l"(desc_a), "l"(desc_b)
        );
        asm volatile("wgmma.commit_group.sync.aligned;\n");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n");
    }

    // ===== Iteration 1 =====
    if (tid == 0) {
        uint32_t bar_addr = (uint32_t)__cvta_generic_to_shared(&bar[1]);
        uint32_t sA_addr  = (uint32_t)__cvta_generic_to_shared(sA);
        uint32_t sB_addr  = (uint32_t)__cvta_generic_to_shared(sB);

        asm volatile(
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
            :
            : "r"(bar_addr), "r"(64*16*2 + 16*8*2)
        );
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global"
            ".mbarrier::complete_tx::bytes [%0], [%1, {16, 0}], [%2];\n"
            :
            : "r"(sA_addr), "l"(A_desc), "r"(bar_addr)
        );
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global"
            ".mbarrier::complete_tx::bytes [%0], [%1, {0, 16}], [%2];\n"
            :
            : "r"(sB_addr), "l"(B_desc), "r"(bar_addr)
        );
    }
    __syncthreads();

    {
        uint32_t bar_addr = (uint32_t)__cvta_generic_to_shared(&bar[1]);
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "loop1:\n"
            "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
            "@!p bra loop1;\n"
            "}\n"
            :
            : "r"(bar_addr), "r"(phase1)
        );
    }

    {
        uint32_t sA_addr = (uint32_t)__cvta_generic_to_shared(sA);
        uint32_t sB_addr = (uint32_t)__cvta_generic_to_shared(sB);
        uint64_t desc_a = ((uint64_t)(sA_addr >> 4) & 0x3FFFULL)
                        | ((uint64_t)1 << 16)
                        | ((uint64_t)16 << 32)
                        | ((uint64_t)3 << 62);
        uint64_t desc_b = ((uint64_t)(sB_addr >> 4) & 0x3FFFULL)
                        | ((uint64_t)8 << 16)
                        | ((uint64_t)1 << 32)
                        | ((uint64_t)0 << 62);

        asm volatile("wgmma.fence.sync.aligned;\n");
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %4, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
            "{%0, %1, %2, %3}, %5, %6, p, 1, 1, 0, 1;\n"
            "}\n"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(1), "l"(desc_a), "l"(desc_b)
        );
        asm volatile("wgmma.commit_group.sync.aligned;\n");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n");
    }

    // Epilogue (same as pyptx m64n8 thread layout)
    uint32_t warp_id = tid >> 5;
    uint32_t lane = tid & 31;
    uint32_t row = warp_id * 16 + (lane >> 2);
    uint32_t col = (lane & 3) * 2;

    C_global[row * 8 + col    ] = acc[0];
    C_global[row * 8 + col + 1] = acc[1];
    C_global[(row + 8) * 8 + col    ] = acc[2];
    C_global[(row + 8) * 8 + col + 1] = acc[3];
}
"""

# Compile with NVRTC
opts = [
    b"--gpu-architecture=sm_90a",
    b"-std=c++17",
    b"-default-device",
    b"-I/root/pyptx/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/include",
]
err, prog = nvrtc.nvrtcCreateProgram(CUDA_SRC.encode(), b"kloop.cu", 0, [], [])
assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, err
err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
    err2, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
    log = b" " * log_size
    err3, = nvrtc.nvrtcGetProgramLog(prog, log)
    print("NVRTC compile failed:")
    print(log.decode())
    raise SystemExit(1)

err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
ptx = b" " * ptx_size
err, = nvrtc.nvrtcGetPTX(prog, ptx)
ptx_str = ptx.decode()
print("=== NVRTC PTX ===")
print(ptx_str)
with open("/tmp/nvrtc_kloop.ptx", "w") as f:
    f.write(ptx_str)
print(f"\n\n(saved to /tmp/nvrtc_kloop.ptx, {len(ptx_str)} bytes)")
