"""Hopper FlashAttention forward with BM=128 query tiles.

This kernel is a heavier Hopper example than the tutorial FA kernel:

- BM = 128 rows per CTA
- one warpgroup (128 threads) per CTA
- BN = 64 KV tile
- K/V are double-buffered with a 2-stage TMA pipeline
- each staged K/V tile is reused for two independent 64-row query slices
- online softmax state carried separately for the top and bottom 64-row slices

The important structural improvement over the single-block tutorial kernel is
that each K/V tile is amortized across 128 query rows instead of 64, while
still staying within Hopper's static shared-memory budget.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from pyptx import Tile, kernel, ptx, reg, smem
from pyptx.smem import apply_swizzle
from pyptx.types import b16, b32, bf16, f32, pred, u32


BM = 128
WG_M = 64
BN = 64
LOG2E = 1.4426950408889634


def _frag_positions(n: int):
    assert n % 8 == 0
    out = []
    for g in range(n // 8):
        c = 8 * g
        out.extend(((False, c), (False, c + 1), (True, c), (True, c + 1)))
    return out


def build_flash_attention_hopper(
    M_q: int,
    N_seq: int,
    head_dim: int = 64,
    *,
    sm_scale: float | None = None,
):
    """Build a BM=128 Hopper FlashAttention kernel."""
    assert M_q % BM == 0, f"M_q={M_q} must be divisible by BM={BM}"
    assert head_dim in (16, 32, 64), f"head_dim must be 16/32/64, got {head_dim}"
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * LOG2E

    HD = head_dim
    assert N_seq % BN == 0, f"N_seq={N_seq} must be divisible by BN={BN}"
    qk_k_iters = HD // 16
    pv_k_iters = BN // 16
    QK_FRAG = BN // 2
    ACC_FRAG = HD // 2
    QK_POS = _frag_positions(BN)
    ACC_POS = _frag_positions(HD)
    ROW_A_QK = [i for i, (is_b, _) in enumerate(QK_POS) if not is_b]
    ROW_B_QK = [i for i, (is_b, _) in enumerate(QK_POS) if is_b]
    ROW_A_ACC = [i for i, (is_b, _) in enumerate(ACC_POS) if not is_b]
    ROW_B_ACC = [i for i, (is_b, _) in enumerate(ACC_POS) if is_b]

    p_row_bytes = BN * 2
    v_row_bytes = HD * 2
    p_swizzle = "128B" if p_row_bytes >= 128 else "64B" if p_row_bytes == 64 else "32B"

    @kernel(
        in_specs=(
            Tile.wgmma_a(M_q, HD, bf16, tile_m=WG_M),
            Tile.wgmma_b(HD, N_seq, bf16, tile_n=BN),
            Tile.wgmma_b(N_seq, HD, bf16, tile_k=BN, tile_n=HD),
        ),
        out_specs=(Tile(M_q, HD, f32),),
        grid=(M_q // BM, 1, 1),
        block=(128, 1, 1),
        arch="sm_90a",
    )
    def flash_attn(Q, K_t, V, O):
        sQ0 = smem.wgmma_tile(bf16, (WG_M, HD), major="K")
        sQ1 = smem.wgmma_tile(bf16, (WG_M, HD), major="K")
        sK0 = smem.wgmma_tile(bf16, (HD, BN), major="MN")
        sK1 = smem.wgmma_tile(bf16, (HD, BN), major="MN")
        sV0 = smem.wgmma_tile(bf16, (BN, HD), major="MN")
        sV1 = smem.wgmma_tile(bf16, (BN, HD), major="MN")
        sP = smem.wgmma_tile(bf16, (WG_M, BN), major="K")

        bar_q = smem.mbarrier(1)
        bar_k = smem.mbarrier(2)
        phase_q = reg.scalar(b32, init=0)
        phase_k0 = reg.scalar(b32, init=0)
        phase_k1 = reg.scalar(b32, init=0)

        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        q_block = reg.scalar(u32)
        ptx.inst.mov.u32(q_block, ptx.special.ctaid.x())
        q_row_base = q_block * BM

        with ptx.if_(tid == 0):
            ptx.mbarrier.init(bar_q[0], 1)
            ptx.mbarrier.init(bar_k[0], 1)
            ptx.mbarrier.init(bar_k[1], 1)
            ptx.fence.proxy_async_shared_cta()
            ptx.mbarrier.arrive_expect_tx(bar_q[0], BM * HD * 2)
            ptx.cp.async_.bulk.tensor_2d(
                dst=sQ0[0], src=Q.tma_desc(), coord=(0, q_row_base), mbar=bar_q[0],
            )
            q_row_base_1 = reg.scalar(u32)
            ptx.inst.add.u32(q_row_base_1, q_row_base, WG_M)
            ptx.cp.async_.bulk.tensor_2d(
                dst=sQ1[0], src=Q.tma_desc(), coord=(0, q_row_base_1), mbar=bar_q[0],
            )
        ptx.bar.sync(0)
        ptx.mbarrier.wait(bar_q[0], phase_q)

        warp = tid >> 5
        lane = reg.scalar(u32)
        ptx.inst.and_.b32(lane, tid, 31)
        row_a = reg.scalar(u32)
        ptx.inst.shl.b32(row_a, warp, 4)
        tmp = reg.scalar(u32)
        ptx.inst.shr.u32(tmp, lane, 2)
        ptx.inst.add.u32(row_a, row_a, tmp)
        row_b = reg.scalar(u32)
        ptx.inst.add.u32(row_b, row_a, 8)
        col_base = reg.scalar(u32)
        ptx.inst.and_.b32(col_base, lane, 3)
        ptx.inst.shl.b32(col_base, col_base, 1)

        qk_scale_reg = reg.scalar(f32, init=qk_scale)
        zero_f = reg.scalar(f32, init=0.0)
        sP_base = reg.scalar(u32)
        ptx.inst.mov.b32(sP_base, sP.name)

        def make_state():
            m_a = reg.scalar(f32, init=-1e30)
            m_b = reg.scalar(f32, init=-1e30)
            l_a = reg.scalar(f32, init=0.0)
            l_b = reg.scalar(f32, init=0.0)
            acc = reg.array(f32, ACC_FRAG)
            for i in range(ACC_FRAG):
                ptx.inst.mov.f32(acc[i], zero_f)
            return m_a, m_b, l_a, l_b, acc

        m0_a, m0_b, l0_a, l0_b, acc0 = make_state()
        m1_a, m1_b, l1_a, l1_b, acc1 = make_state()

        def emit_group_stage0(q_tile, m_a, m_b, l_a, l_b, acc):
            qk_frag = reg.array(f32, QK_FRAG)
            ptx.wgmma.fence()
            for kk in range(qk_k_iters):
                ptx.wgmma.mma_async(
                    shape=(64, BN, 16),
                    dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                    d=qk_frag, a=q_tile, b=sK0,
                    scale_d=(kk != 0), trans_a=0, trans_b=1,
                    a_k_offset=kk * 32, b_k_offset=kk * 16 * BN * 2,
                )
            ptx.wgmma.commit_group()
            ptx.wgmma.wait_group(0)

            for i in range(QK_FRAG):
                ptx.inst.mul.f32(qk_frag[i], qk_frag[i], qk_scale_reg)

            row_a_new = reg.scalar(f32, init=-1e30)
            row_b_new = reg.scalar(f32, init=-1e30)
            for i in ROW_A_QK:
                ptx.inst.max.f32(row_a_new, row_a_new, qk_frag[i])
            for i in ROW_B_QK:
                ptx.inst.max.f32(row_b_new, row_b_new, qk_frag[i])
            ptx.warp.reduce_max(row_a_new, width=4)
            ptx.warp.reduce_max(row_b_new, width=4)

            m_a_new = reg.scalar(f32)
            m_b_new = reg.scalar(f32)
            ptx.inst.max.f32(m_a_new, m_a, row_a_new)
            ptx.inst.max.f32(m_b_new, m_b, row_b_new)

            d_a = reg.scalar(f32)
            d_b = reg.scalar(f32)
            alpha_a = reg.scalar(f32)
            alpha_b = reg.scalar(f32)
            ptx.inst.sub.f32(d_a, m_a, m_a_new)
            ptx.inst.sub.f32(d_b, m_b, m_b_new)
            ptx.inst.ex2.approx.f32(alpha_a, d_a)
            ptx.inst.ex2.approx.f32(alpha_b, d_b)

            for i in ROW_A_QK:
                diff = reg.scalar(f32)
                ptx.inst.sub.f32(diff, qk_frag[i], m_a_new)
                ptx.inst.ex2.approx.f32(qk_frag[i], diff)
            for i in ROW_B_QK:
                diff = reg.scalar(f32)
                ptx.inst.sub.f32(diff, qk_frag[i], m_b_new)
                ptx.inst.ex2.approx.f32(qk_frag[i], diff)

            ra_sum = reg.scalar(f32, init=0.0)
            rb_sum = reg.scalar(f32, init=0.0)
            for i in ROW_A_QK:
                ptx.inst.add.f32(ra_sum, ra_sum, qk_frag[i])
            for i in ROW_B_QK:
                ptx.inst.add.f32(rb_sum, rb_sum, qk_frag[i])
            ptx.warp.reduce_sum(ra_sum, width=4)
            ptx.warp.reduce_sum(rb_sum, width=4)
            ptx.inst.fma.rn.f32(l_a, l_a, alpha_a, ra_sum)
            ptx.inst.fma.rn.f32(l_b, l_b, alpha_b, rb_sum)

            for i in ROW_A_ACC:
                ptx.inst.mul.f32(acc[i], acc[i], alpha_a)
            for i in ROW_B_ACC:
                ptx.inst.mul.f32(acc[i], acc[i], alpha_b)

            row_a_byte = reg.scalar(u32)
            row_b_byte = reg.scalar(u32)
            ptx.inst.mul.lo.u32(row_a_byte, row_a, p_row_bytes)
            ptx.inst.mul.lo.u32(row_b_byte, row_b, p_row_bytes)
            col_byte = reg.scalar(u32)
            ptx.inst.shl.b32(col_byte, col_base, 1)
            for fi in range(0, QK_FRAG, 2):
                is_b, c_off = QK_POS[fi]
                logical = reg.scalar(u32)
                ptx.inst.add.u32(logical, row_b_byte if is_b else row_a_byte, col_byte)
                if c_off:
                    ptx.inst.add.u32(logical, logical, c_off * 2)
                physical = apply_swizzle(logical, p_swizzle)
                st_addr = reg.scalar(u32)
                ptx.inst.add.u32(st_addr, sP_base, physical)
                bf_lo = reg.scalar(b16)
                bf_hi = reg.scalar(b16)
                pack = reg.scalar(b32)
                ptx.inst.cvt.rn.bf16.f32(bf_lo, qk_frag[fi])
                ptx.inst.cvt.rn.bf16.f32(bf_hi, qk_frag[fi + 1])
                ptx.inst.mov.b32(pack, [bf_lo, bf_hi])
                ptx.inst.st.shared.b32(ptx.addr(st_addr), pack)

            ptx.bar.sync(0, 128)

            ptx.wgmma.fence()
            for kk in range(pv_k_iters):
                ptx.wgmma.mma_async(
                    shape=(64, HD, 16),
                    dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                    d=acc, a=sP, b=sV0,
                    scale_d=True, trans_a=0, trans_b=1,
                    a_k_offset=kk * 32, b_k_offset=kk * 16 * v_row_bytes,
                )
            ptx.wgmma.commit_group()
            ptx.wgmma.wait_group(0)
            ptx.bar.sync(0, 128)

            ptx.inst.mov.f32(m_a, m_a_new)
            ptx.inst.mov.f32(m_b, m_b_new)

        def emit_group_stage1(q_tile, m_a, m_b, l_a, l_b, acc):
            qk_frag = reg.array(f32, QK_FRAG)
            ptx.wgmma.fence()
            for kk in range(qk_k_iters):
                ptx.wgmma.mma_async(
                    shape=(64, BN, 16),
                    dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                    d=qk_frag, a=q_tile, b=sK1,
                    scale_d=(kk != 0), trans_a=0, trans_b=1,
                    a_k_offset=kk * 32, b_k_offset=kk * 16 * BN * 2,
                )
            ptx.wgmma.commit_group()
            ptx.wgmma.wait_group(0)

            for i in range(QK_FRAG):
                ptx.inst.mul.f32(qk_frag[i], qk_frag[i], qk_scale_reg)

            row_a_new = reg.scalar(f32, init=-1e30)
            row_b_new = reg.scalar(f32, init=-1e30)
            for i in ROW_A_QK:
                ptx.inst.max.f32(row_a_new, row_a_new, qk_frag[i])
            for i in ROW_B_QK:
                ptx.inst.max.f32(row_b_new, row_b_new, qk_frag[i])
            ptx.warp.reduce_max(row_a_new, width=4)
            ptx.warp.reduce_max(row_b_new, width=4)

            m_a_new = reg.scalar(f32)
            m_b_new = reg.scalar(f32)
            ptx.inst.max.f32(m_a_new, m_a, row_a_new)
            ptx.inst.max.f32(m_b_new, m_b, row_b_new)

            d_a = reg.scalar(f32)
            d_b = reg.scalar(f32)
            alpha_a = reg.scalar(f32)
            alpha_b = reg.scalar(f32)
            ptx.inst.sub.f32(d_a, m_a, m_a_new)
            ptx.inst.sub.f32(d_b, m_b, m_b_new)
            ptx.inst.ex2.approx.f32(alpha_a, d_a)
            ptx.inst.ex2.approx.f32(alpha_b, d_b)

            for i in ROW_A_QK:
                diff = reg.scalar(f32)
                ptx.inst.sub.f32(diff, qk_frag[i], m_a_new)
                ptx.inst.ex2.approx.f32(qk_frag[i], diff)
            for i in ROW_B_QK:
                diff = reg.scalar(f32)
                ptx.inst.sub.f32(diff, qk_frag[i], m_b_new)
                ptx.inst.ex2.approx.f32(qk_frag[i], diff)

            ra_sum = reg.scalar(f32, init=0.0)
            rb_sum = reg.scalar(f32, init=0.0)
            for i in ROW_A_QK:
                ptx.inst.add.f32(ra_sum, ra_sum, qk_frag[i])
            for i in ROW_B_QK:
                ptx.inst.add.f32(rb_sum, rb_sum, qk_frag[i])
            ptx.warp.reduce_sum(ra_sum, width=4)
            ptx.warp.reduce_sum(rb_sum, width=4)
            ptx.inst.fma.rn.f32(l_a, l_a, alpha_a, ra_sum)
            ptx.inst.fma.rn.f32(l_b, l_b, alpha_b, rb_sum)

            for i in ROW_A_ACC:
                ptx.inst.mul.f32(acc[i], acc[i], alpha_a)
            for i in ROW_B_ACC:
                ptx.inst.mul.f32(acc[i], acc[i], alpha_b)

            row_a_byte = reg.scalar(u32)
            row_b_byte = reg.scalar(u32)
            ptx.inst.mul.lo.u32(row_a_byte, row_a, p_row_bytes)
            ptx.inst.mul.lo.u32(row_b_byte, row_b, p_row_bytes)
            col_byte = reg.scalar(u32)
            ptx.inst.shl.b32(col_byte, col_base, 1)
            for fi in range(0, QK_FRAG, 2):
                is_b, c_off = QK_POS[fi]
                logical = reg.scalar(u32)
                ptx.inst.add.u32(logical, row_b_byte if is_b else row_a_byte, col_byte)
                if c_off:
                    ptx.inst.add.u32(logical, logical, c_off * 2)
                physical = apply_swizzle(logical, p_swizzle)
                st_addr = reg.scalar(u32)
                ptx.inst.add.u32(st_addr, sP_base, physical)
                bf_lo = reg.scalar(b16)
                bf_hi = reg.scalar(b16)
                pack = reg.scalar(b32)
                ptx.inst.cvt.rn.bf16.f32(bf_lo, qk_frag[fi])
                ptx.inst.cvt.rn.bf16.f32(bf_hi, qk_frag[fi + 1])
                ptx.inst.mov.b32(pack, [bf_lo, bf_hi])
                ptx.inst.st.shared.b32(ptx.addr(st_addr), pack)

            ptx.bar.sync(0, 128)

            ptx.wgmma.fence()
            for kk in range(pv_k_iters):
                ptx.wgmma.mma_async(
                    shape=(64, HD, 16),
                    dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                    d=acc, a=sP, b=sV1,
                    scale_d=True, trans_a=0, trans_b=1,
                    a_k_offset=kk * 32, b_k_offset=kk * 16 * v_row_bytes,
                )
            ptx.wgmma.commit_group()
            ptx.wgmma.wait_group(0)
            ptx.bar.sync(0, 128)

            ptx.inst.mov.f32(m_a, m_a_new)
            ptx.inst.mov.f32(m_b, m_b_new)

        with ptx.if_(tid == 0):
            ptx.mbarrier.arrive_expect_tx(bar_k[0], HD * BN * 2 + BN * HD * 2)
            ptx.cp.async_.bulk.tensor_2d(
                dst=sK0[0], src=K_t.tma_desc(), coord=(0, 0), mbar=bar_k[0],
            )
            ptx.cp.async_.bulk.tensor_2d(
                dst=sV0[0], src=V.tma_desc(), coord=(0, 0), mbar=bar_k[0],
            )
        ptx.bar.sync(0)
        ptx.mbarrier.wait(bar_k[0], phase_k0)
        phase_k0 ^= 1

        k_col = reg.scalar(u32, init=0)
        next_k_col = reg.scalar(u32, init=BN)
        stage = reg.scalar(u32, init=0)
        keep_going = reg.scalar(pred)
        ptx.inst.setp.lt.u32(keep_going, k_col, N_seq)
        with ptx.loop("kv_loop", pred=keep_going):
            have_next = reg.scalar(pred)
            ptx.inst.setp.lt.u32(have_next, next_k_col, N_seq)
            next_stage = reg.scalar(u32)
            ptx.inst.xor.b32(next_stage, stage, 1)

            with ptx.if_(tid == 0):
                with ptx.if_(have_next):
                    with ptx.if_(next_stage == 0):
                        ptx.mbarrier.arrive_expect_tx(bar_k[0], HD * BN * 2 + BN * HD * 2)
                        ptx.cp.async_.bulk.tensor_2d(
                            dst=sK0[0], src=K_t.tma_desc(), coord=(next_k_col, 0), mbar=bar_k[0],
                        )
                        ptx.cp.async_.bulk.tensor_2d(
                            dst=sV0[0], src=V.tma_desc(), coord=(0, next_k_col), mbar=bar_k[0],
                        )
                    with ptx.else_():
                        ptx.mbarrier.arrive_expect_tx(bar_k[1], HD * BN * 2 + BN * HD * 2)
                        ptx.cp.async_.bulk.tensor_2d(
                            dst=sK1[0], src=K_t.tma_desc(), coord=(next_k_col, 0), mbar=bar_k[1],
                        )
                        ptx.cp.async_.bulk.tensor_2d(
                            dst=sV1[0], src=V.tma_desc(), coord=(0, next_k_col), mbar=bar_k[1],
                        )

            with ptx.if_(stage == 0):
                emit_group_stage0(sQ0, m0_a, m0_b, l0_a, l0_b, acc0)
                emit_group_stage0(sQ1, m1_a, m1_b, l1_a, l1_b, acc1)
            with ptx.else_():
                emit_group_stage1(sQ0, m0_a, m0_b, l0_a, l0_b, acc0)
                emit_group_stage1(sQ1, m1_a, m1_b, l1_a, l1_b, acc1)

            k_col += BN
            next_k_col += BN
            stage ^= 1
            ptx.inst.setp.lt.u32(keep_going, k_col, N_seq)
            with ptx.if_(keep_going):
                with ptx.if_(stage == 0):
                    ptx.mbarrier.wait(bar_k[0], phase_k0)
                    phase_k0 ^= 1
                with ptx.else_():
                    ptx.mbarrier.wait(bar_k[1], phase_k1)
                    phase_k1 ^= 1

        inv0_a = reg.scalar(f32)
        inv0_b = reg.scalar(f32)
        inv1_a = reg.scalar(f32)
        inv1_b = reg.scalar(f32)
        ptx.inst.rcp.approx.f32(inv0_a, l0_a)
        ptx.inst.rcp.approx.f32(inv0_b, l0_b)
        ptx.inst.rcp.approx.f32(inv1_a, l1_a)
        ptx.inst.rcp.approx.f32(inv1_b, l1_b)
        for i in ROW_A_ACC:
            ptx.inst.mul.f32(acc0[i], acc0[i], inv0_a)
            ptx.inst.mul.f32(acc1[i], acc1[i], inv1_a)
        for i in ROW_B_ACC:
            ptx.inst.mul.f32(acc0[i], acc0[i], inv0_b)
            ptx.inst.mul.f32(acc1[i], acc1[i], inv1_b)

        (po,) = ptx.global_ptrs(O)
        q_row_base_1 = reg.scalar(u32)
        ptx.inst.add.u32(q_row_base_1, q_row_base, WG_M)

        def store_group(acc, row_base):
            row_a_global = reg.scalar(u32)
            row_b_global = reg.scalar(u32)
            ptx.inst.add.u32(row_a_global, row_a, row_base)
            ptx.inst.add.u32(row_b_global, row_b, row_base)
            for i in range(0, ACC_FRAG, 2):
                is_b, c_off = ACC_POS[i]
                row = row_b_global if is_b else row_a_global
                col = reg.scalar(u32)
                ptx.inst.add.u32(col, col_base, c_off)
                off = (row * HD + col) * 4
                ptx.inst.st.global_.v2.f32(ptx.addr(po + off), [acc[i], acc[i + 1]])

        store_group(acc0, q_row_base)
        store_group(acc1, q_row_base_1)
        ptx.ret()

    return flash_attn


def _run_torch_case(M_q: int, N_seq: int, HD: int) -> None:
    import torch

    k_fn = build_flash_attention_hopper(M_q, N_seq, HD)
    torch.manual_seed(M_q * 131 + N_seq + HD * 7)
    q = (torch.randn(M_q, HD, device="cuda") * 0.3).to(torch.bfloat16)
    k = (torch.randn(N_seq, HD, device="cuda") * 0.3).to(torch.bfloat16)
    v = (torch.randn(N_seq, HD, device="cuda") * 0.3).to(torch.bfloat16)
    k_t = k.T.contiguous()
    out = k_fn(q, k_t, v)
    torch.cuda.synchronize()
    ref = torch.softmax(q.float() @ k.float().T / math.sqrt(HD), dim=-1) @ v.float()
    diff = float((out - ref).abs().max())
    ok = bool(torch.allclose(out, ref, atol=5e-2, rtol=1e-2))
    status = "OK  " if ok else "FAIL"
    print(f"[Torch{status}] M_q={M_q:5d} N={N_seq:5d} D={HD:3d}  max_abs={diff:.3e}")


def _run_jax_case(M_q: int, N_seq: int, HD: int) -> None:
    k_fn = build_flash_attention_hopper(M_q, N_seq, HD)
    key = jax.random.PRNGKey(M_q * 17 + N_seq * 31 + HD)
    q = (jax.random.normal(key, (M_q, HD), dtype=jnp.float32) * 0.3).astype(jnp.bfloat16)
    k = (jax.random.normal(key + 1, (N_seq, HD), dtype=jnp.float32) * 0.3).astype(jnp.bfloat16)
    v = (jax.random.normal(key + 2, (N_seq, HD), dtype=jnp.float32) * 0.3).astype(jnp.bfloat16)
    run = jax.jit(lambda q_, k_, v_: k_fn(q_, k_.T, v_))
    out = np.asarray(run(q, k, v).block_until_ready())
    ref = np.asarray(jax.nn.softmax((q.astype(jnp.float32) @ k.astype(jnp.float32).T) / math.sqrt(HD), axis=-1) @ v.astype(jnp.float32))
    diff = float(np.max(np.abs(out - ref)))
    ok = bool(np.allclose(out, ref, atol=5e-2, rtol=1e-2))
    status = "OK " if ok else "FAIL"
    print(f"[JAX {status}] M_q={M_q:5d} N={N_seq:5d} D={HD:3d}  max_abs={diff:.3e}")


def main() -> None:
    for HD in (16, 32, 64):
        for M_q, N_seq in ((128, 128), (512, 512), (1024, 1024), (2048, 2048)):
            _run_torch_case(M_q, N_seq, HD)
            _run_jax_case(M_q, N_seq, HD)


if __name__ == "__main__":
    main()
