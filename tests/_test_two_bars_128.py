"""Block=128 threads, two TMAs with two separate mbarriers. No wgmma."""
import jax, jax.numpy as jnp, numpy as np
from pyptx import kernel, reg, smem, ptx, Tile, Layout
from pyptx.types import bf16, f32, b32, u32, u64, b64

_ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()

@kernel(
    in_specs=(Tile.wgmma_a(64, 32, bf16, tile_k=16),),
    out_specs=(Tile(64, 16, bf16),),
    grid=(1, 1, 1), block=(128, 1, 1), arch="sm_90a",
)
def k(A, out):
    sA = smem.wgmma_tile(bf16, (64, 16), major="K")
    bar0 = smem.mbarrier(1)
    bar1 = smem.mbarrier(1)
    phase0 = reg.scalar(b32, init=0)
    phase1 = reg.scalar(b32, init=0)

    tid = ptx.special.tid.x()
    with ptx.if_(tid == 0):
        ptx.mbarrier.init(bar0[0], 1)
        ptx.mbarrier.init(bar1[0], 1)
        ptx.fence.proxy_async_shared_cta()

    with ptx.if_(tid == 0):
        ptx.mbarrier.arrive_expect_tx(bar0[0], 64*16*2)
        ptx.cp.async_.bulk.tensor_2d(dst=sA[0], src=A.tma_desc(), coord=(0, 0), mbar=bar0[0])
    ptx.bar.sync(0)
    ptx.mbarrier.wait(bar0[0], phase0)

    with ptx.if_(tid == 0):
        ptx.mbarrier.arrive_expect_tx(bar1[0], 64*16*2)
        ptx.cp.async_.bulk.tensor_2d(dst=sA[0], src=A.tma_desc(), coord=(16, 0), mbar=bar1[0])
    ptx.bar.sync(0)
    ptx.mbarrier.wait(bar1[0], phase1)

    with ptx.if_(tid == 0):
        ptx.cp.async_.bulk.tensor_2d.store(dst=out.tma_desc(), src=sA[0], coord=(0, 0))
        ptx.cp.async_.bulk.commit_group()
        ptx.cp.async_.bulk.wait_group(0)
    ptx.ret()

a = jnp.asarray(np.arange(64*32, dtype=np.float32).reshape(64, 32), dtype=jnp.bfloat16)

@jax.jit
def fn(a):
    return k(a)

print("running...", flush=True)
out = np.asarray(fn(a))
print("shape:", out.shape)
print("first 4:", out[0, :4])
