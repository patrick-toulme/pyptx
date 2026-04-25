"""K-loop wgmma GEMM through jax.jit — same kernel we just verified works
via direct driver launch. Goal: prove the full pyptx jax.jit path works."""
import jax, jax.numpy as jnp, numpy as np
from pyptx import kernel, reg, smem, ptx, Tile, Layout
from pyptx.types import bf16, f32, b32, u32, u64, b64

_ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()


@kernel(
    in_specs=(
        Tile.wgmma_a(64, 32, bf16, tile_k=16),
        Tile.wgmma_b(32, 8,  bf16, tile_k=16, tile_n=8),
    ),
    out_specs=(Tile(64, 8, f32),),
    grid=(1, 1, 1), block=(128, 1, 1), arch="sm_90a",
)
def gemm(A, B, C):
    sA = smem.wgmma_tile(bf16, (64, 16), major="K")
    sB = smem.wgmma_tile(bf16, (16, 8),  major="MN")
    bar0 = smem.mbarrier(1)
    bar1 = smem.mbarrier(1)
    phase0 = reg.scalar(b32, init=0)
    phase1 = reg.scalar(b32, init=0)
    acc = reg.array(f32, 4)

    tid = ptx.special.tid.x()
    with ptx.if_(tid == 0):
        ptx.mbarrier.init(bar0[0], 1)
        ptx.mbarrier.init(bar1[0], 1)
        ptx.fence.proxy_async_shared_cta()

    with ptx.if_(tid == 0):
        ptx.mbarrier.arrive_expect_tx(bar0[0], 64*16*2 + 16*8*2)
        ptx.cp.async_.bulk.tensor_2d(dst=sA[0], src=A.tma_desc(), coord=(0, 0), mbar=bar0[0])
        ptx.cp.async_.bulk.tensor_2d(dst=sB[0], src=B.tma_desc(), coord=(0, 0), mbar=bar0[0])
    ptx.bar.sync(0)

    ptx.mbarrier.wait(bar0[0], phase0)

    ptx.wgmma.fence()
    ptx.wgmma.mma_async(shape=(64, 8, 16), dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
        d=acc, a=sA, b=sB, scale_d=False, trans_a=0, trans_b=1)
    ptx.wgmma.commit_group()
    ptx.wgmma.wait_group(0)

    with ptx.if_(tid == 0):
        ptx.mbarrier.arrive_expect_tx(bar1[0], 64*16*2 + 16*8*2)
        ptx.cp.async_.bulk.tensor_2d(dst=sA[0], src=A.tma_desc(), coord=(16, 0), mbar=bar1[0])
        ptx.cp.async_.bulk.tensor_2d(dst=sB[0], src=B.tma_desc(), coord=(0, 16), mbar=bar1[0])
    ptx.bar.sync(0)

    ptx.mbarrier.wait(bar1[0], phase1)

    ptx.wgmma.fence()
    ptx.wgmma.mma_async(shape=(64, 8, 16), dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
        d=acc, a=sA, b=sB, scale_d=True, trans_a=0, trans_b=1)
    ptx.wgmma.commit_group()
    ptx.wgmma.wait_group(0)

    # Epilogue — NVRTC's thread→row/col layout for m64n8
    tid2 = reg.scalar(u32); ptx.inst.mov.u32(tid2, ptx.special.tid.x())
    group = reg.scalar(u32); lane = reg.scalar(u32)
    ptx.inst.shr.u32(group, tid2, 5)
    ptx.inst.and_.b32(lane, tid2, 31)
    row = reg.scalar(u32); col = reg.scalar(u32); tmp = reg.scalar(u32)
    ptx.inst.shl.b32(row, group, 4)
    ptx.inst.shr.u32(tmp, lane, 2)
    ptx.inst.add.u32(row, row, tmp)
    ptx.inst.and_.b32(col, lane, 3)
    ptx.inst.shl.b32(col, col, 1)
    pc = reg.scalar(b64); ptx.inst.ld.param.u64(pc, ptx.addr(C))
    ptx.inst.cvta.to.global_.u64(pc, pc)
    row_col = reg.scalar(u32)
    ptx.inst.shl.b32(tmp, row, 3)
    ptx.inst.add.u32(row_col, tmp, col)
    off = reg.scalar(u64)
    ptx.inst.mul.wide.u32(off, row_col, 4)
    p0 = reg.scalar(b64)
    ptx.inst.add.s64(p0, pc, off)
    ptx.inst.st.global_.f32(ptx.addr(p0), acc[0])
    ptx.inst.st.global_.f32(ptx.addr(p0, 4), acc[1])
    row8 = reg.scalar(u32)
    ptx.inst.add.u32(row8, row, 8)
    ptx.inst.shl.b32(tmp, row8, 3)
    ptx.inst.add.u32(row_col, tmp, col)
    ptx.inst.mul.wide.u32(off, row_col, 4)
    p1 = reg.scalar(b64)
    ptx.inst.add.s64(p1, pc, off)
    ptx.inst.st.global_.f32(ptx.addr(p1), acc[2])
    ptx.inst.st.global_.f32(ptx.addr(p1, 4), acc[3])
    ptx.ret()


np.random.seed(1)
a = (np.random.randn(64, 32) * 0.1).astype(np.float32)
b = (np.random.randn(32, 8)  * 0.1).astype(np.float32)
A = jnp.asarray(a, dtype=jnp.bfloat16)
B = jnp.asarray(b, dtype=jnp.bfloat16)


@jax.jit
def fn(A, B):
    return gemm(A, B)


print("running via jax.jit...", flush=True)
out = np.asarray(fn(A, B))
ref = np.asarray(jax.lax.dot_general(A, B, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32))
print(f"max diff:  {float(np.abs(out - ref).max())}")
print(f"bit-exact: {bool(np.array_equal(out, ref))}")
print(f"out[0]:    {out[0]}")
print(f"ref[0]:    {ref[0]}")
