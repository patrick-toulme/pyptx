"""Test wgmma chaining: store f32 fragment as bf16 to SMEM with B32 swizzle,
then read via wgmma descriptor.

Validates that ``smem.apply_swizzle("32B")`` produces the correct physical
byte offsets for the GMMA hardware to read back via ``auto_descriptor``.
"""
import pytest
import numpy as np

jax = pytest.importorskip("jax")
jnp = jax.numpy

from pyptx import kernel, reg, smem, ptx, Tile
from pyptx.smem import apply_swizzle
from pyptx.types import bf16, f32, b16, b32, u32


def build_swizzle_roundtrip():
    """Store all-ones as bf16 with B32 swizzle → wgmma reads → output should be column sums."""

    @kernel(
        in_specs=(Tile.wgmma_b(16, 16, bf16),),
        out_specs=(Tile(64, 16, f32),),
        grid=(1, 1, 1),
        block=(128, 1, 1),
        arch="sm_90a",
    )
    def roundtrip(D, C):
        sP = smem.alloc(b16, 64 * 16)
        sD = smem.wgmma_tile(bf16, (16, 16), major="MN")
        bar = smem.mbarrier(1)
        phase = reg.scalar(b32, init=0)
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())

        with ptx.if_(tid == 0):
            ptx.mbarrier.init(bar[0], 1)
            ptx.fence.proxy_async_shared_cta()
            ptx.mbarrier.arrive_expect_tx(bar[0], 16 * 16 * 2)
            ptx.cp.async_.bulk.tensor_2d(
                dst=sD[0], src=D.tma_desc(), coord=(0, 0), mbar=bar[0],
            )

        sP_base = reg.scalar(u32)
        ptx.inst.mov.b32(sP_base, sP.name)
        one_bf = reg.scalar(b16)
        one_f = reg.scalar(f32, init=1.0)
        ptx.inst.cvt.rn.bf16.f32(one_bf, one_f)
        for i in range(8):
            logical = reg.scalar(u32)
            ptx.inst.mad.lo.u32(logical, tid, 16, i * 2)
            physical = apply_swizzle(logical, "32B")
            addr = reg.scalar(u32)
            ptx.inst.add.u32(addr, sP_base, physical)
            ptx.inst.st.shared.b16(ptx.addr(addr), one_bf)

        ptx.bar.sync(0)
        ptx.mbarrier.wait(bar[0], phase)

        c_frag = reg.array(f32, 8)
        desc_p = ptx.wgmma.auto_descriptor(
            sP_base, dtype=bf16, shape=(64, 16), major="K",
        )
        ptx.wgmma.fence()
        ptx.wgmma.mma_async(
            shape=(64, 16, 16),
            dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
            d=c_frag, a=desc_p, b=sD,
            scale_d=False, trans_a=0, trans_b=1,
        )
        ptx.wgmma.commit_group()
        ptx.wgmma.wait_group(0)

        warp_id = tid >> 5
        lane = reg.scalar(u32)
        ptx.inst.and_.b32(lane, tid, 31)
        row_a = (warp_id << 4) + (lane >> 2)
        row_b = row_a + 8
        col_base = (lane & 3) << 1
        (pc,) = ptx.global_ptrs(C)
        from pyptx.types import pred as pred_t  # noqa: F811

        frag_pos = [
            (row_a, 0), (row_a, 1), (row_b, 0), (row_b, 1),
            (row_a, 8), (row_a, 9), (row_b, 8), (row_b, 9),
        ]
        for i, (row, c_off) in enumerate(frag_pos):
            col = col_base + c_off
            off = (row * 16 + col) * 4
            ptx.inst.st.global_.f32(ptx.addr(pc + off), c_frag[i])
        ptx.ret()

    return roundtrip


def main():
    _ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()

    k = build_swizzle_roundtrip()
    d = jnp.eye(16, dtype=jnp.bfloat16)

    @jax.jit
    def fn(d):
        return k(d)

    out = np.asarray(fn(d))
    expected = np.ones((64, 16), dtype=np.float32)
    diff = float(np.abs(out - expected).max())
    ok = diff < 0.01
    print(f"[{'OK  ' if ok else 'FAIL'}] swizzle roundtrip: max_abs={diff:.4f}")
    assert ok, f"B32 swizzle roundtrip failed: max_abs={diff}"


if __name__ == "__main__":
    main()
