"""Diagnose m64n16k16 wgmma f32 output frag layout by probing specific outputs."""
import jax, jax.numpy as jnp, numpy as np
import sys; sys.path.insert(0, ".")
_ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()
from pyptx import kernel, reg, smem, ptx, Tile
from pyptx.types import bf16, f32, b32, u32, u64, b64

BM, BN, D = 64, 16, 16


def build_qk(layout_variant: str):
    """Try a specific frag→(row, col) mapping. ``layout_variant`` is
    one of 'abcd-abcd' (my current guess) or 'abab-cdcd' (alternative)."""

    @kernel(
        in_specs=(Tile.wgmma_a(BM, D, bf16), Tile.wgmma_b(D, BN, bf16)),
        out_specs=(Tile(BM, BN, f32),),
        grid=(1, 1, 1), block=(128, 1, 1), arch="sm_90a",
    )
    def qk(Q, Kt, O):
        sQ = smem.wgmma_tile(bf16, (BM, D), major="K")
        sK = smem.wgmma_tile(bf16, (D, BN), major="MN")
        bar = smem.mbarrier(1); ph = reg.scalar(b32, init=0)
        qk_acc = reg.array(f32, 8)
        tid = reg.scalar(u32); ptx.inst.mov.u32(tid, ptx.special.tid.x())
        with ptx.if_(tid == 0):
            ptx.mbarrier.init(bar[0], 1); ptx.fence.proxy_async_shared_cta()
            ptx.mbarrier.arrive_expect_tx(bar[0], BM*D*2 + D*BN*2)
            ptx.cp.async_.bulk.tensor_2d(dst=sQ[0], src=Q.tma_desc(), coord=(0, 0), mbar=bar[0])
            ptx.cp.async_.bulk.tensor_2d(dst=sK[0], src=Kt.tma_desc(), coord=(0, 0), mbar=bar[0])
        ptx.bar.sync(0); ptx.mbarrier.wait(bar[0], ph)
        ptx.wgmma.fence()
        ptx.wgmma.mma_async(shape=(64, 16, 16), dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
            d=qk_acc, a=sQ, b=sK, scale_d=False, trans_a=0, trans_b=1)
        ptx.wgmma.commit_group(); ptx.wgmma.wait_group(0)

        wid = reg.scalar(u32); ln = reg.scalar(u32)
        ptx.inst.shr.u32(wid, tid, 5); ptx.inst.and_.b32(ln, tid, 31)
        row_a = reg.scalar(u32); tmp = reg.scalar(u32)
        ptx.inst.shl.b32(row_a, wid, 4); ptx.inst.shr.u32(tmp, ln, 2); ptx.inst.add.u32(row_a, row_a, tmp)
        row_b = reg.scalar(u32); ptx.inst.add.u32(row_b, row_a, 8)
        col_base = reg.scalar(u32); ptx.inst.and_.b32(col_base, ln, 3); ptx.inst.shl.b32(col_base, col_base, 1)
        po = reg.scalar(b64); ptx.inst.ld.param.u64(po, ptx.addr(O)); ptx.inst.cvta.to.global_.u64(po, po)

        def st(row, col_off, val):
            col = reg.scalar(u32); ptx.inst.add.u32(col, col_base, col_off)
            rxw = reg.scalar(u32); ptx.inst.mul.lo.u32(rxw, row, BN)
            idx = reg.scalar(u32); ptx.inst.add.u32(idx, rxw, col)
            off = reg.scalar(u64); ptx.inst.mul.wide.u32(off, idx, 4)
            ptr = reg.scalar(b64); ptx.inst.add.s64(ptr, po, off)
            ptx.inst.st.global_.f32(ptx.addr(ptr), val)

        if layout_variant == "abcd-abcd":
            # Current guess: [ra, rb] each have all their cols in sequence
            st(row_a, 0, qk_acc[0]); st(row_a, 1, qk_acc[1])
            st(row_a, 8, qk_acc[2]); st(row_a, 9, qk_acc[3])
            st(row_b, 0, qk_acc[4]); st(row_b, 1, qk_acc[5])
            st(row_b, 8, qk_acc[6]); st(row_b, 9, qk_acc[7])
        elif layout_variant == "abab-abab":
            # Alternative: [ra_lo, rb_lo, ra_hi, rb_hi] interleave
            st(row_a, 0, qk_acc[0]); st(row_a, 1, qk_acc[1])
            st(row_b, 0, qk_acc[2]); st(row_b, 1, qk_acc[3])
            st(row_a, 8, qk_acc[4]); st(row_a, 9, qk_acc[5])
            st(row_b, 8, qk_acc[6]); st(row_b, 9, qk_acc[7])
        ptx.ret()

    return qk


def test(layout):
    k_fn = build_qk(layout)
    np.random.seed(42)
    q = jnp.asarray((np.random.randn(BM, D) * 0.3).astype(np.float32), dtype=jnp.bfloat16)
    k = jnp.asarray((np.random.randn(BN, D) * 0.3).astype(np.float32), dtype=jnp.bfloat16)
    k_t = jnp.asarray(np.ascontiguousarray(np.asarray(k).T), dtype=jnp.bfloat16)

    @jax.jit
    def fn(q, k_t):
        return k_fn(q, k_t)

    out = np.asarray(fn(q, k_t))
    ref = np.asarray(jnp.matmul(q, k.T)).astype(np.float32)
    diff = float(np.abs(out - ref).max())
    print(f"layout={layout}: max_diff={diff:g}")
    if diff < 1e-2:
        print("  MATCHES — this is the correct frag layout")
    else:
        # Show row 0 full
        print(f"  row 0 out: {out[0].tolist()}")
        print(f"  row 0 ref: {ref[0].tolist()}")
        print(f"  row 8 out: {out[8].tolist()}")
        print(f"  row 8 ref: {ref[8].tolist()}")


for layout in ["abcd-abcd", "abab-abab"]:
    test(layout)
    print()
