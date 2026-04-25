"""TMA load -> wgmma -> TMA store. Does TMA store work after wgmma?"""
import jax, jax.numpy as jnp, numpy as np
from pyptx import kernel, reg, smem, ptx, Tile, Layout
from pyptx.types import bf16, f32, b32, u32, u64, b64

_ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()

@kernel(
    in_specs=(Tile.wgmma_a(64, 16, bf16), Tile.wgmma_b(16, 8, bf16)),
    out_specs=(Tile(64, 16, bf16),),   # TMA store target
    grid=(1, 1, 1), block=(128, 1, 1), arch="sm_90a",
)
def k(A, B, out):
    sA = smem.wgmma_tile(bf16, (64, 16), major="K")
    sB = smem.wgmma_tile(bf16, (16, 8),  major="MN")
    bar = smem.mbarrier(1)
    phase = reg.scalar(b32, init=0)
    acc = reg.array(f32, 4)

    tid = ptx.special.tid.x()
    with ptx.if_(tid == 0):
        ptx.mbarrier.init(bar[0], 1)
        ptx.fence.proxy_async_shared_cta()
        ptx.mbarrier.arrive_expect_tx(bar[0], 2304)
        ptx.cp.async_.bulk.tensor_2d(dst=sA[0], src=A.tma_desc(), coord=(0, 0), mbar=bar[0])
        ptx.cp.async_.bulk.tensor_2d(dst=sB[0], src=B.tma_desc(), coord=(0, 0), mbar=bar[0])
    ptx.bar.sync(0)
    ptx.mbarrier.wait(bar[0], phase)

    ptx.wgmma.fence()
    ptx.wgmma.mma_async(shape=(64, 8, 16), dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
        d=acc, a=sA, b=sB, scale_d=False, trans_a=0, trans_b=1)
    ptx.wgmma.commit_group()
    ptx.wgmma.wait_group(0)

    # Now TMA store sA (which holds bf16 data still) to `out`
    with ptx.if_(tid == 0):
        ptx.cp.async_.bulk.tensor_2d.store(dst=out.tma_desc(), src=sA[0], coord=(0, 0))
        ptx.cp.async_.bulk.commit_group()
        ptx.cp.async_.bulk.wait_group(0)
    ptx.ret()

a = jnp.asarray(np.arange(64*16, dtype=np.float32).reshape(64, 16), dtype=jnp.bfloat16)
b = jnp.zeros((16, 8), dtype=jnp.bfloat16)

@jax.jit
def fn(a, b):
    return k(a, b)

print("running...", flush=True)
out = np.asarray(fn(a, b))
print("ran!  out[0, :4]:", out[0, :4])
