"""Blackwell tcgen05 FlashAttention forward (bf16, head_dim 128).

Warp-specialized sm_100a attention forward. The architecture follows
FlashAttention-4 / CUTLASS 77_blackwell_fmha, with a pyptx twist in how the
softmax work is laid out:

- 256 query rows per CTA as two 128-row subtiles ("stages"), BN=128 KV tile
- ``tcgen05.mma`` computes S = Q@K^T into TMEM; P@V accumulates onto O in
  TMEM with the A operand sourced directly from TMEM
- **column-split softmax pool**: all 8 softmax warps work on one S subtile
  at a time. Warp w (rows 32w..32w+31, columns 0-63) pairs with warp w+4
  (same rows, columns 64-127). Partial row maxima are exchanged through
  shared memory at a per-warp-pair named barrier, then each side exps,
  packs bf16, and stores its own quarter of P. This halves the S -> P
  latency compared to one warpgroup per stage and lets P@V start after the
  first 16 P columns land (4-way split-P).
- 16 warps: softmax pool (w0-7), correction (w8-11), MMA dispatch (w12),
  TMA load (w14); w13/w15 idle
- K/V staged through a 4-slot SMEM ring (K,V alternating) fed by TMA
- correction warps rescale O in TMEM only when the running row max moves by
  more than ``RESCALE_THRESHOLD`` in scaled-log2 units (FA4 "skip
  correction"). In the common no-rescale case the softmax warps release the
  O barrier themselves so the correction round-trip stays off the critical
  path; the correction warps re-derive the same per-warp ballot from the
  alphas in shared memory, so barrier arrival counts stay exact.
- epilogue: correction warps normalize O by the row sum and store bf16

TMEM layout (512 columns of 32-bit):

    cols   0-127   S0 (f32) — P0 (packed bf16) overwrites cols 0-63
    cols 128-255   S1 (f32) — P1 overwrites cols 128-191
    cols 256-383   O0 (f32 accumulator)
    cols 384-511   O1 (f32 accumulator)

Tensors are passed 2D as (batch*heads*seqlen, head_dim), i.e. a contiguous
(B, H, S, D) view. grid = (seqlen//256, batch*heads).
"""
from __future__ import annotations

import math

from pyptx import Tile, kernel, ptx, reg, smem
from pyptx.specs import Layout
from pyptx.types import b32, b64, bf16, f32, pred, u32, u64

LOG2E = 1.4426950408889634

BM = 128          # rows per Q subtile
Q_STAGE = 2       # subtiles per CTA -> 256 rows
BN = 128          # KV tile size
HD = 128          # head dim (v1: fixed 128)
KV_SLOTS = 4      # SMEM ring slots, each one K or V tile

TILE_BYTES = BN * HD * 2          # 32 KB per K/V/Q tile
STRIPE_BYTES = 16384              # TMA 128B-swizzle stripe: 128 rows x 64 cols bf16
Q_BYTES = Q_STAGE * TILE_BYTES
SLOT_UNITS = TILE_BYTES // 16     # ring-slot stride in 16B descriptor units

# SMEM map (bytes)
SMEM_Q = 0
SMEM_KV = SMEM_Q + Q_BYTES                    # 65536
SMEM_STATS = SMEM_KV + KV_SLOTS * TILE_BYTES  # 196608
#   alpha[2][128]   @ +0     (stage-major, f32)
#   sumA[2][128]    @ +1024
#   sumB[2][128]    @ +2048
#   xchg[2][2][128] @ +3072  (stage, half, row)
# stats area: alpha[2][128] @0, sums[2][2][128] @1024, and the pair-max
# exchange slots @3072 double-buffered by KV-tile parity
# ([stage][parity][half][row] = stage*2048 + parity*1024 + half*512)
SMEM_BARS = SMEM_STATS + 7168                 # 203776
BAR_Q = SMEM_BARS + 0             # 2 x 8
BAR_KV_FULL = SMEM_BARS + 16      # 4 x 8
BAR_KV_FREE = SMEM_BARS + 48      # 4 x 8
BAR_S_FULL = SMEM_BARS + 80       # 2 x 8
BAR_P_Q = SMEM_BARS + 96          # 2 stages x 4 quarters x 8 = 64
# O_RESC is split by KV-tile parity: [stage][tile&1]. The alpha stream
# can run up to two tiles ahead of the MMA warp's consumption, which
# would alias a single barrier's phase parity; per-parity barriers keep
# every waiter within one phase of its producer.
BAR_O_RESC = SMEM_BARS + 160      # 2 stages x 2 parities x 8
BAR_O_DONE = SMEM_BARS + 192      # 8
BAR_STATS_FREE = SMEM_BARS + 200  # 2 x 8
BAR_PV_DONE = SMEM_BARS + 216     # 2 x 8
BAR_Q_FREE = SMEM_BARS + 232      # 8 (persistent: Q smem free for next work item)
BAR_ITER = SMEM_BARS + 248        # 8 (dbg_lockstep only)
BAR_EPI = SMEM_BARS + 256         # 8 (softmax item-start gate)
BAR_O_DONE2 = SMEM_BARS + 264     # 8 (stage-1's final PV done)
SMEM_TMEM_SLOT = SMEM_BARS + 240
SMEM_BYTES = BAR_O_DONE2 + 8

# TMEM columns
TM_S = (0, 128)
TM_P = (0, 128)      # P overwrites the low half of S
TM_O = (256, 384)

# FA4-style skip-correction threshold, in scaled log2 units.
RESCALE_THRESHOLD = 8.0

# Constants for the FMA-pipe exp2 emulation (degree-3 polynomial, FA4's
# add.rm floor trick). Bit-exact f32 values.
import struct as _struct

def _f32c(hexval: str) -> float:
    return _struct.unpack(">f", bytes.fromhex(hexval))[0]

EX2_BIG = _f32c("4B400000")     # 2^23 + 2^22
EX2_CLAMP = _f32c("C2FE0000")   # -127.0
EX2_C3 = _f32c("3D9DF09D")
EX2_C2 = _f32c("3E6906A4")
EX2_C1 = _f32c("3F31F519")
EX2_C0 = _f32c("3F800000")


def _umma_idesc(m, n, a_fmt, b_fmt, c_fmt, a_major=0, b_major=0):
    """Blackwell UMMA instruction descriptor (cute mma_sm100_desc layout).

    formats: f16=0, bf16=1 (A/B); c: f16=0, f32=1. majors: K=0, MN=1.
    """
    desc = 0
    desc |= c_fmt << 4
    desc |= a_fmt << 7
    desc |= b_fmt << 10
    desc |= a_major << 15
    desc |= b_major << 16
    desc |= (n >> 3) << 17
    desc |= (m >> 4) << 24
    return desc

MMA_TID = 384        # warp 12 lane 0
LOAD_TID = 448       # warp 14 lane 0


def _kmajor_desc_off(kk: int) -> int:
    """Descriptor offset (16B units) for k-atom kk of a K-major 128-col bf16 tile.

    The tile is stored as two TMA swizzle stripes of 128 rows x 128 bytes;
    k-atoms 0-3 live in stripe 0 (+32B each), 4-7 in stripe 1 (+16KB).
    """
    return (kk // 4) * 1024 + (kk % 4) * 2


def build_flash_attention_blackwell(
    seqlen: int,
    batch_heads: int,
    head_dim: int = 128,
    *,
    sm_scale: float | None = None,
    arch: str = "sm_100a",
    rescale_threshold: float = RESCALE_THRESHOLD,
    emu_pairs: tuple = (1, 3, 5, 7),
    s_f16: bool = False,   # experimental fp16 mode; see notes in docstring
    dbg_f16_noexp: bool = False,
    waitmap: bool = False,
    num_sms: int = 148,
    # Work items overlap freely across all roles, with ONE cross-item
    # gate: the softmax pool's item start waits for the previous
    # epilogue's O readout (BAR_EPI). Without it, the softmax warps'
    # TMEM loads/stores concurrently with the epilogue's O reads corrupt
    # single rows (~1e-2 rate, stage 1) — a pairing the mainloop never
    # exercises since only rescale tiles read O there. dbg_lockstep=True
    # (or "no_<role>") additionally locks roles at item boundaries for
    # bisection; dbg_epi_gate="mma" moves the gate to the MMA warp.
    dbg_lockstep=False,
    dbg_epi_gate="softmax",
):
    assert head_dim == HD, "v1 supports head_dim=128 only"
    assert seqlen % (Q_STAGE * BM) == 0, f"seqlen must be a multiple of 256, got {seqlen}"
    n_tiles = seqlen // BN
    assert n_tiles % 2 == 0
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * LOG2E
    total_rows = batch_heads * seqlen
    from pyptx.types import f16 as _f16t
    io_t = _f16t if s_f16 else bf16

    # persistent scheduling: one wave of CTAs, each looping over
    # (m_block, bh) work items. m_block varies fastest so consecutive
    # CTAs share K/V in L2.
    n_mblocks = seqlen // (Q_STAGE * BM)
    assert n_mblocks & (n_mblocks - 1) == 0, "n_mblocks must be a power of 2"
    mb_shift = n_mblocks.bit_length() - 1
    total_work = n_mblocks * batch_heads
    num_ctas = min(total_work, num_sms)

    from pyptx.debugkit import WAITMAP_MAX_SITES, WaitMap
    wm = WaitMap() if waitmap else None

    kernel_kwargs = dict(
        in_specs=(
            Tile(total_rows, HD, io_t, Layout.TMA_128B, tma_box=(BM, 64)),
            Tile(total_rows, HD, io_t, Layout.TMA_128B, tma_box=(BN, 64)),
            Tile(total_rows, HD, io_t, Layout.TMA_128B, tma_box=(BN, 64)),
        ),
        grid=(num_ctas, 1, 1),
        block=(512, 1, 1),
        arch=arch,
        smem=SMEM_BYTES,
        extern_smem=True,
        reqntid=(512, 1, 1),
    )
    out_o = Tile(total_rows, HD, io_t, Layout.ROW)
    if waitmap:
        kernel_kwargs["out_specs"] = (
            out_o, Tile(512, WAITMAP_MAX_SITES, b32, Layout.ROW))
    else:
        kernel_kwargs["out_specs"] = (out_o,)

    def _fa_body(Q, K, V, O, WM=None):
        base = smem.base()
        tid = reg.scalar(u32); ptx.inst.mov.u32(tid, ptx.special.tid.x())
        cta_id = reg.scalar(u32); ptx.inst.mov.u32(cta_id, ptx.special.ctaid.x())
        if waitmap:
            (pwm,) = ptx.global_ptrs(WM)
            wm.attach(pwm, tid)
            _wm_ctx = wm.active()
            _wm_ctx.__enter__()

        warp = tid >> 5
        row128 = reg.scalar(u32); ptx.inst.and_.b32(row128, tid, 127)
        # TMEM lane bits for this thread's warp: lanes (warp%4)*32
        lane_addr_bits = reg.scalar(u32)
        ptx.inst.and_.b32(lane_addr_bits, tid, 96)
        ptx.inst.shl.b32(lane_addr_bits, lane_addr_bits, 16)
        warp_in_group = reg.scalar(u32)
        ptx.inst.and_.b32(warp_in_group, warp, 3)

        is_softmax = reg.scalar(pred); ptx.inst.setp.lt.u32(is_softmax, tid, 256)
        is_correction = reg.scalar(pred)
        with ptx.scope():
            ge = reg.scalar(pred); lt = reg.scalar(pred)
            ptx.inst.setp.ge.u32(ge, tid, 256)
            ptx.inst.setp.lt.u32(lt, tid, 384)
            ptx.inst.and_.pred(is_correction, ge, lt)
        is_mma = reg.scalar(pred); ptx.inst.setp.eq.u32(is_mma, tid, MMA_TID)
        is_load = reg.scalar(pred); ptx.inst.setp.eq.u32(is_load, tid, LOAD_TID)
        alloc_warp = reg.scalar(pred)
        ptx.inst.setp.eq.u32(alloc_warp, warp, 12)

        # dbg_lockstep: participating roles arrive/wait this barrier at
        # the end of each work item, serializing cross-item overlap.
        # dbg_lockstep=True locks all four roles; "no_<role>" releases one
        # (bisection aid).
        lock_roles = set()
        if dbg_lockstep:
            lock_roles = {"softmax", "corr", "mma", "load"}
            if isinstance(dbg_lockstep, str) and dbg_lockstep.startswith("no_"):
                lock_roles.discard(dbg_lockstep[3:])
        lock_count = (
            256 * ("softmax" in lock_roles) + 128 * ("corr" in lock_roles)
            + ("mma" in lock_roles) + ("load" in lock_roles)
        )

        def iter_sync(ph, role):
            if role not in lock_roles:
                return
            ptx.mbarrier.arrive(base + BAR_ITER)
            ptx.mbarrier.wait(base + BAR_ITER, ph)
            ptx.inst.xor.b32(ph, ph, 1)

        # Per-work-item Q/KV row bases, derived from a linear work id.
        # m_block = w & (n_mblocks-1), bh = w >> mb_shift.
        def derive_rows(w, want_kv: bool):
            q_row = reg.scalar(u32)
            kv_row = reg.scalar(u32) if want_kv else None
            with ptx.scope():
                mb = reg.scalar(u32)
                ptx.inst.and_.b32(mb, w, n_mblocks - 1)
                bhw = reg.scalar(u32)
                ptx.inst.shr.u32(bhw, w, mb_shift)
                ptx.inst.mul.lo.u32(q_row, bhw, seqlen)
                if want_kv:
                    ptx.inst.mov.u32(kv_row, q_row)
                ptx.inst.mul.lo.u32(mb, mb, Q_STAGE * BM)
                ptx.inst.add.u32(q_row, q_row, mb)
            return q_row, kv_row

        # ---- init barriers, allocate TMEM ----
        with ptx.if_(tid == 0):
            for i in range(2):
                ptx.mbarrier.init(base + BAR_Q + 8 * i, 1)
                ptx.mbarrier.init(base + BAR_S_FULL + 8 * i, 1)
                for par in range(2):
                    ptx.mbarrier.init(base + BAR_O_RESC + 16 * i + 8 * par, 128)
                ptx.mbarrier.init(base + BAR_STATS_FREE + 8 * i, 128)
                ptx.mbarrier.init(base + BAR_PV_DONE + 8 * i, 1)
                for qtr in range(4):
                    ptx.mbarrier.init(base + BAR_P_Q + 32 * i + 8 * qtr, 128)
            for s in range(KV_SLOTS):
                ptx.mbarrier.init(base + BAR_KV_FULL + 8 * s, 1)
                ptx.mbarrier.init(base + BAR_KV_FREE + 8 * s, 1)
            ptx.mbarrier.init(base + BAR_O_DONE, 1)
            ptx.mbarrier.init(base + BAR_O_DONE2, 1)
            ptx.mbarrier.init(base + BAR_Q_FREE, 1)
            if dbg_lockstep:
                ptx.mbarrier.init(base + BAR_ITER, lock_count)
            if dbg_epi_gate:
                # arrived by the 128 correction threads per epilogue
                ptx.mbarrier.init(base + BAR_EPI, 128)
            ptx.fence.proxy_async_shared_cta()
        with ptx.if_(alloc_warp):
            ptx.tcgen05.alloc(base + SMEM_TMEM_SLOT, 512)
        ptx.bar.sync(0, 512)
        tmem = smem.load(b32, ptx.addr(base + SMEM_TMEM_SLOT))

        # Register-file repartition. setmaxnreg is warpgroup-collective, so
        # each aligned warpgroup executes its own dec/inc at the top of its
        # role block. 136*256 + 96*128 + 144*128 = 65536. (Correction at 96
        # fits 64-wide epilogue O loads — half the wait_ld round trips of
        # the 32-wide quarters; the softmax item start is gated on exactly
        # this readout.)
        NREG_SOFTMAX = 136
        NREG_CORR = 96
        NREG_OTHER = 144

        # =================================================================
        # TMA load warp (single lane; warpgroup 12-15 takes NREG_OTHER)
        # =================================================================
        is_other_wg = reg.scalar(pred)
        ptx.inst.setp.ge.u32(is_other_wg, tid, 384)
        with ptx.if_(is_other_wg):
            ptx.setmaxnreg(NREG_OTHER, inc=True)

        with ptx.if_(is_load):
            def load_tile(dst_off: int, tensor, row_reg, mbar_off: int):
                for stripe in range(2):
                    ptx.cp.async_.bulk.tensor_2d(
                        dst=base + dst_off + stripe * STRIPE_BYTES,
                        src=tensor.tma_desc(),
                        coord=(stripe * 64, row_reg),
                        mbar=base + mbar_off,
                    )

            # self-credit the ring and the Q buffer once: every load below
            # (including each work item's prologue) waits FREE first, so
            # the accounting is uniform across persistent iterations
            for s in range(KV_SLOTS):
                ptx.mbarrier.arrive(base + BAR_KV_FREE + 8 * s)
            ptx.mbarrier.arrive(base + BAR_Q_FREE)
            free_phase = [reg.scalar(b32, init=0) for _ in range(KV_SLOTS)]
            q_free_phase = reg.scalar(b32, init=0)

            def load_kv(slot: int, tensor, row_reg):
                ptx.mbarrier.wait(base + BAR_KV_FREE + 8 * slot, free_phase[slot])
                free_phase[slot] ^= 1
                ptx.mbarrier.arrive_expect_tx(base + BAR_KV_FULL + 8 * slot, TILE_BYTES)
                load_tile(SMEM_KV + slot * TILE_BYTES, tensor, row_reg, BAR_KV_FULL + 8 * slot)
                ptx.inst.add.u32(row_reg, row_reg, BN)

            lw = reg.scalar(u32); ptx.inst.mov.u32(lw, cta_id)
            lgo = reg.scalar(pred)
            liter_ph = reg.scalar(b32, init=0)
            ptx.inst.setp.lt.u32(lgo, lw, total_work)
            with ptx.loop("load_work_loop", pred=lgo):
                q_row0, kv_row0 = derive_rows(lw, want_kv=True)
                q_row1 = reg.scalar(u32)
                ptx.inst.add.u32(q_row1, q_row0, BM)
                krow = reg.scalar(u32); ptx.inst.mov.u32(krow, kv_row0)
                vrow = reg.scalar(u32); ptx.inst.mov.u32(vrow, kv_row0)

                # prologue, ordered by first use: QK(stage 0) needs only
                # Q0+K0, so K0 goes ahead of the second Q tile
                ptx.mbarrier.wait(base + BAR_Q_FREE, q_free_phase)
                ptx.inst.xor.b32(q_free_phase, q_free_phase, 1)
                ptx.mbarrier.arrive_expect_tx(base + BAR_Q + 0, TILE_BYTES)
                load_tile(SMEM_Q, Q, q_row0, BAR_Q + 0)
                load_kv(0, K, krow)
                ptx.mbarrier.arrive_expect_tx(base + BAR_Q + 8, TILE_BYTES)
                load_tile(SMEM_Q + TILE_BYTES, Q, q_row1, BAR_Q + 8)
                load_kv(1, V, vrow)
                load_kv(2, K, krow)
                load_kv(3, V, vrow)

                # steady state: 4 loads per trip into slots 0..3
                trips = (n_tiles - 2) // 2
                if trips > 0:
                    trip = reg.scalar(u32, init=0)
                    go = reg.scalar(pred)
                    ptx.inst.setp.lt.u32(go, trip, trips)
                    with ptx.loop("kv_load_loop", pred=go):
                        for slot, tensor, row_reg in (
                            (0, K, krow), (1, V, vrow), (2, K, krow), (3, V, vrow),
                        ):
                            load_kv(slot, tensor, row_reg)
                        trip += 1
                        ptx.inst.setp.lt.u32(go, trip, trips)

                iter_sync(liter_ph, "load")
                ptx.inst.add.u32(lw, lw, num_ctas)
                ptx.inst.setp.lt.u32(lgo, lw, total_work)

        # =================================================================
        # MMA dispatch warp (single thread). All operand descriptors are
        # loop-invariant and precomputed into registers.
        # =================================================================

        with ptx.if_(is_mma):
            if s_f16:
                idesc_qk = reg.scalar(
                    b32, init=_umma_idesc(128, BN, 0, 0, 0, 0, 0)  # f16 in, f16 acc
                )
                idesc_pv = reg.scalar(
                    b32, init=_umma_idesc(128, HD, 0, 0, 1, 0, 1)  # f16 A/B, f32 acc
                )
            else:
                idesc_qk = reg.scalar(
                    b32,
                    init=ptx.tcgen05.make_instr_desc_f16bf16_f32(
                        m=128, n=BN, ab_dtype="bf16", a_major="K", b_major="K"
                    ),
                )
                idesc_pv = reg.scalar(
                    b32,
                    init=ptx.tcgen05.make_instr_desc_f16bf16_f32(
                        m=128, n=HD, ab_dtype="bf16", a_major="K", b_major="MN"
                    ),
                )
            p_acc = reg.scalar(pred)
            p_noacc = reg.scalar(pred)
            with ptx.scope():
                one = reg.scalar(u32, init=1)
                zero = reg.scalar(u32, init=0)
                ptx.inst.setp.ne.b32(p_acc, one, 0)
                ptx.inst.setp.ne.b32(p_noacc, zero, 0)

            dq0 = ptx.tcgen05.masked_descriptor(base + SMEM_Q)
            dk0 = ptx.tcgen05.masked_descriptor(base + SMEM_KV)
            dv0 = ptx.tcgen05.descriptor(
                base + SMEM_KV + TILE_BYTES,
                stride_bytes=1024, leading_bytes=16384, swizzle="128B",
            )

            def derive(base_desc, off_units):
                if off_units == 0:
                    return base_desc
                d = reg.scalar(b64)
                ptx.inst.add.s64(d, base_desc, off_units)
                return d

            # [stage][kk] / [slot_idx][kk] descriptor registers, hoisted
            desc_q = [
                [derive(dq0, s * SLOT_UNITS + _kmajor_desc_off(kk)) for kk in range(8)]
                for s in range(Q_STAGE)
            ]
            desc_k = [
                [derive(dk0, k_idx * 2 * SLOT_UNITS + _kmajor_desc_off(kk)) for kk in range(8)]
                for k_idx in range(2)
            ]
            desc_v = [
                [derive(dv0, v_idx * 2 * SLOT_UNITS + kk * 128) for kk in range(8)]
                for v_idx in range(2)
            ]

            def qk_mma(stage: int, k_idx: int):
                d = tmem + TM_S[stage]
                for kk in range(8):
                    ptx.tcgen05.mma(
                        d, desc_q[stage][kk], desc_k[k_idx][kk], idesc_qk,
                        kind="f16", pred_operand=(p_noacc if kk == 0 else p_acc),
                    )
                ptx.tcgen05.commit(base + BAR_S_FULL + 8 * stage)

            ph_kv = [reg.scalar(b32, init=0) for _ in range(KV_SLOTS)]
            ph_pq = [
                [reg.scalar(b32, init=0) for _ in range(4)] for _ in range(Q_STAGE)
            ]
            ph_or = [
                [reg.scalar(b32, init=0) for _ in range(2)] for _ in range(Q_STAGE)
            ]

            def wait_kv(slot: int):
                ptx.mbarrier.wait(base + BAR_KV_FULL + 8 * slot, ph_kv[slot])
                ph_kv[slot] ^= 1

            def pv_mma(stage: int, v_idx: int, first_accum: bool, par: int):
                d = tmem + TM_O[stage]
                for q in range(4):
                    ptx.mbarrier.wait(
                        base + BAR_P_Q + 32 * stage + 8 * q,
                        ph_pq[stage][q],
                    )
                    ph_pq[stage][q] ^= 1
                    if q == 0:
                        ptx.mbarrier.wait(
                            base + BAR_O_RESC + 16 * stage + 8 * par,
                            ph_or[stage][par],
                        )
                        ph_or[stage][par] ^= 1
                    ptx.tcgen05.fence_after_thread_sync()
                    for kk in (2 * q, 2 * q + 1):
                        a = tmem + (TM_P[stage] + kk * 8)
                        ptx.tcgen05.mma(
                            d, a, desc_v[v_idx][kk], idesc_pv,
                            kind="f16", a_is_tmem=True,
                            pred_operand=(p_noacc if (first_accum and kk == 0) else p_acc),
                        )
                # deferred-rescale handshake: correction may only touch O
                # after this tile's PV has fully accumulated
                ptx.tcgen05.commit(base + BAR_PV_DONE + 8 * stage)

            ph_qa = reg.scalar(b32, init=0)
            ph_qb = reg.scalar(b32, init=0)
            miter_ph = reg.scalar(b32, init=0)
            mw = reg.scalar(u32); ptx.inst.mov.u32(mw, cta_id)
            mgo = reg.scalar(pred)
            ptx.inst.setp.lt.u32(mgo, mw, total_work)
            if dbg_epi_gate == "mma":
                epi_ph = reg.scalar(b32, init=0)
            with ptx.loop("mma_work_loop", pred=mgo):
                if dbg_epi_gate == "mma":
                    ptx.mbarrier.wait(base + BAR_EPI, epi_ph)
                    ptx.inst.xor.b32(epi_ph, epi_ph, 1)
                # ---- prologue: S(0) for stage 0 as soon as Q0+K0 land;
                # the second Q tile's TMA overlaps the first QK ----
                ptx.mbarrier.wait(base + BAR_Q + 0, ph_qa)
                ptx.inst.xor.b32(ph_qa, ph_qa, 1)
                wait_kv(0)
                qk_mma(0, 0)
                ptx.mbarrier.wait(base + BAR_Q + 8, ph_qb)
                ptx.inst.xor.b32(ph_qb, ph_qb, 1)
                qk_mma(1, 0)
                ptx.tcgen05.commit(base + BAR_KV_FREE + 0)

                # ---- peeled j=0: PV(0) with V slot 1, S(1) with K slot 2 ----
                wait_kv(1)
                wait_kv(2)
                pv_mma(0, 0, first_accum=True, par=0)
                qk_mma(0, 1)
                pv_mma(1, 0, first_accum=True, par=0)
                qk_mma(1, 1)
                ptx.tcgen05.commit(base + BAR_KV_FREE + 8 * 2)
                ptx.tcgen05.commit(base + BAR_KV_FREE + 8 * 1)

                # ---- steady: j = 1 .. n_tiles-2, two iterations per trip ----
                trips = (n_tiles - 2) // 2
                if trips > 0:
                    trip = reg.scalar(u32, init=0)
                    go = reg.scalar(pred)
                    ptx.inst.setp.lt.u32(go, trip, trips)
                    with ptx.loop("mma_loop", pred=go):
                        # j odd: V in slot 3, next K in slot 0. KV waits
                        # are hoisted: KV normally arrives early (the
                        # load warp runs ahead) while P quarters are the
                        # tight resource, so the warp should not block
                        # between a PV and its following QK — the QK
                        # UMMAs queue behind the PVs and keep the tensor
                        # core fed through quarter stalls.
                        wait_kv(3)
                        wait_kv(0)
                        for stage in range(Q_STAGE):
                            pv_mma(stage, 1, first_accum=False, par=1)
                            qk_mma(stage, 0)
                        ptx.tcgen05.commit(base + BAR_KV_FREE + 8 * 0)
                        ptx.tcgen05.commit(base + BAR_KV_FREE + 8 * 3)
                        # j even: V in slot 1, next K in slot 2
                        wait_kv(1)
                        wait_kv(2)
                        for stage in range(Q_STAGE):
                            pv_mma(stage, 0, first_accum=False, par=0)
                            qk_mma(stage, 1)
                        ptx.tcgen05.commit(base + BAR_KV_FREE + 8 * 2)
                        ptx.tcgen05.commit(base + BAR_KV_FREE + 8 * 1)
                        trip += 1
                        ptx.inst.setp.lt.u32(go, trip, trips)

                # all QKs for this work item are issued: release the Q
                # buffer so the next item's Q TMA overlaps the final PV
                # and the epilogue
                ptx.tcgen05.commit(base + BAR_Q_FREE)

                # ---- peeled j = n_tiles-1 (odd): PV only, V in slot 3 ----
                wait_kv(3)
                for stage in range(Q_STAGE):
                    pv_mma(stage, 1, first_accum=False, par=1)
                ptx.tcgen05.commit(base + BAR_KV_FREE + 8 * 3)
                ptx.tcgen05.commit(base + BAR_O_DONE)

                iter_sync(miter_ph, "mma")
                ptx.inst.add.u32(mw, mw, num_ctas)
                ptx.inst.setp.lt.u32(mgo, mw, total_work)

            # tail padding: correction's PV_DONE parity waits may lag the
            # completion stream while idling through skip tiles; extra
            # completions per stage guarantee every pending parity test
            # still sees a phase flip after the last real PV
            for _ in range(8):
                for stage in range(Q_STAGE):
                    ptx.tcgen05.commit(base + BAR_PV_DONE + 8 * stage)

        # =================================================================
        # Softmax: two stage-parallel warpgroups (warps 0-3 -> stage 0,
        # warps 4-7 -> stage 1). Thread t owns row t&127 and streams its
        # 128 columns in four 32-column chunks, ping-ponging two register
        # buffers so each chunk's TMEM load flies under the previous
        # chunk's exp. Stages no longer queue behind each other in a
        # shared pool, and no cross-thread max exchange exists: every
        # thread sees its whole row.
        # =================================================================
        with ptx.if_(is_softmax):
            ptx.setmaxnreg(NREG_SOFTMAX, inc=True)
            from pyptx.types import b16 as _b16

            stage_b = reg.scalar(u32)            # this warpgroup's Q stage
            ptx.inst.and_.b32(stage_b, tid, 128)
            ptx.inst.shr.u32(stage_b, stage_b, 7)

            qk_scale_reg = reg.scalar(f32, init=qk_scale)
            thresh_p = reg.scalar(f32, init=float(2.0 ** rescale_threshold))
            one_f = reg.scalar(f32, init=1.0)
            zero_f = reg.scalar(f32, init=0.0)
            # per-thread state for this thread's row (stage implicit)
            mneg = reg.scalar(f32, init=0.0)     # -(basis * qk_scale)
            pmax_pend = reg.scalar(f32, init=0.0)
            sums_pend = reg.scalar(f32, init=0.0)
            l_run = reg.scalar(f32, init=0.0)
            if emu_pairs:
                clamp_f = reg.scalar(f32, init=EX2_CLAMP)
                big_f = reg.scalar(f32, init=EX2_BIG)
                c3_f = reg.scalar(f32, init=EX2_C3)
                c2_f = reg.scalar(f32, init=EX2_C2)
                c1_f = reg.scalar(f32, init=EX2_C1)
                c0_f = reg.scalar(f32, init=EX2_C0)
                big2 = reg.scalar(b64); ptx.inst.mov.b64(big2, [big_f, big_f])
                c3_2 = reg.scalar(b64); ptx.inst.mov.b64(c3_2, [c3_f, c3_f])
                c2_2 = reg.scalar(b64); ptx.inst.mov.b64(c2_2, [c2_f, c2_f])
                c1_2 = reg.scalar(b64); ptx.inst.mov.b64(c1_2, [c1_f, c1_f])
                c0_2 = reg.scalar(b64); ptx.inst.mov.b64(c0_2, [c0_f, c0_f])
                qk2 = reg.scalar(b64)
                ptx.inst.mov.b64(qk2, [qk_scale_reg, qk_scale_reg])

            # stage-relative addresses (runtime: stage differs per warp)
            st_col = reg.scalar(u32)
            ptx.inst.shl.b32(st_col, stage_b, 7)          # S/P column offset
            sp_base = reg.scalar(b32)                     # S (and P) row base
            ptx.inst.mov.b32(sp_base, tmem)
            ptx.inst.add.u32(sp_base, sp_base, lane_addr_bits)
            ptx.inst.add.u32(sp_base, sp_base, st_col)

            st8 = reg.scalar(u32)
            ptx.inst.shl.b32(st8, stage_b, 3)             # 8*stage
            sfull_ad = reg.scalar(u32)
            ptx.inst.add.u32(sfull_ad, st8, base + BAR_S_FULL)
            sfree_ad = reg.scalar(u32)
            ptx.inst.add.u32(sfree_ad, st8, base + BAR_STATS_FREE)
            pq_ad = reg.scalar(u32)
            with ptx.scope():
                st32 = reg.scalar(u32)
                ptx.inst.shl.b32(st32, stage_b, 5)
                ptx.inst.add.u32(pq_ad, st32, base + BAR_P_Q)
            oresc_ad = reg.scalar(u32)
            with ptx.scope():
                st16 = reg.scalar(u32)
                ptx.inst.shl.b32(st16, stage_b, 4)
                ptx.inst.add.u32(oresc_ad, st16, base + BAR_O_RESC)
            stats_bar = reg.scalar(u32)          # named bar 1+w (s0) / 5+w (s1)
            with ptx.scope():
                st4 = reg.scalar(u32)
                ptx.inst.shl.b32(st4, stage_b, 2)
                ptx.inst.add.u32(stats_bar, warp_in_group, 1)
                ptx.inst.add.u32(stats_bar, stats_bar, st4)
            alpha_ad = reg.scalar(u32)
            sum_ad = reg.scalar(u32)
            with ptx.scope():
                row_b = reg.scalar(u32)
                ptx.inst.shl.b32(row_b, row128, 2)
                st512 = reg.scalar(u32)
                ptx.inst.shl.b32(st512, stage_b, 9)
                ptx.inst.add.u32(alpha_ad, row_b, base + SMEM_STATS)
                ptx.inst.add.u32(alpha_ad, alpha_ad, st512)
                ptx.inst.add.u32(sum_ad, alpha_ad, 1024)

            ph_s = reg.scalar(b32, init=0)
            ph_stats = reg.scalar(b32, init=0)

            if s_f16:
                # packed f16x2: a 64-column chunk is 32 registers, so the
                # tile streams in TWO double-buffered chunks — half the
                # load cadence of the f32 path (which can't afford 64-col
                # buffers in registers)
                pwb = (reg.array(b32, 32), reg.array(b32, 32))
                qh16 = reg.scalar(_b16)
                ptx.inst.cvt.rn.f16.f32(qh16, qk_scale_reg)
                q2h = reg.scalar(b32)
                ptx.inst.mov.b32(q2h, [qh16, qh16])
                macc16 = [reg.scalar(b32) for _ in range(4)]
                sacc16 = [reg.scalar(b32) for _ in range(4)]
            else:
                bufs = (reg.array(f32, 32), reg.array(f32, 32))
                pk = reg.array(b32, 16)
            macc = [reg.scalar(f32) for _ in range(4)]
            sacc = [reg.scalar(f32) for _ in range(4)]

            def exp_pair_emu(v, i):
                # exp2 via degree-3 poly on the packed f32x2 pipe
                a0 = reg.scalar(f32); a1 = reg.scalar(f32)
                ptx.inst.max.ftz.f32(a0, v[i], clamp_f)
                ptx.inst.max.ftz.f32(a1, v[i + 1], clamp_f)
                l1 = reg.scalar(b64)
                ptx.inst.mov.b64(l1, [a0, a1])
                ptx.inst.fma.rn.ftz.f32x2(l1, l1, qk2, m2)
                l7 = reg.scalar(b64); l8 = reg.scalar(b64); l9 = reg.scalar(b64)
                ptx.inst.add.rm.ftz.f32x2(l7, l1, big2)
                ptx.inst.sub.rn.ftz.f32x2(l8, l7, big2)
                ptx.inst.sub.rn.ftz.f32x2(l9, l1, l8)
                l10 = reg.scalar(b64)
                ptx.inst.fma.rn.ftz.f32x2(l10, l9, c3_2, c2_2)
                ptx.inst.fma.rn.ftz.f32x2(l10, l10, l9, c1_2)
                ptx.inst.fma.rn.ftz.f32x2(l10, l10, l9, c0_2)
                r1 = reg.scalar(b32); r2 = reg.scalar(b32)
                r3 = reg.scalar(b32); r4 = reg.scalar(b32)
                ptx.inst.mov.b64([r1, r2], l7)
                ptx.inst.mov.b64([r3, r4], l10)
                ptx.inst.shl.b32(r1, r1, 23)
                ptx.inst.add.s32(r1, r1, r3)
                ptx.inst.shl.b32(r2, r2, 23)
                ptx.inst.add.s32(r2, r2, r4)
                ptx.inst.mov.b32(v[i], r1)
                ptx.inst.mov.b32(v[i + 1], r2)

            def exp_chunk(v):
                for pair in range(16):
                    i = 2 * pair
                    if pair in emu_pairs:
                        exp_pair_emu(v, i)
                        continue
                    ptx.inst.fma.rn.f32(v[i], v[i], qk_scale_reg, mneg)
                    ptx.inst.ex2.approx.ftz.f32(v[i], v[i])
                    ptx.inst.fma.rn.f32(v[i + 1], v[i + 1], qk_scale_reg, mneg)
                    ptx.inst.ex2.approx.ftz.f32(v[i + 1], v[i + 1])

            def chunk_reduce(v, c):
                # fold this chunk into the running per-thread max/sum accs
                for a in range(4):
                    t = reg.scalar(f32)
                    ptx.inst.max.f32(t, v[a], v[4 + a], v[8 + a])
                    ptx.inst.max.f32(t, t, v[12 + a], v[16 + a])
                    ptx.inst.max.f32(t, t, v[20 + a], v[24 + a])
                    ptx.inst.max.f32(t, t, v[28 + a])
                    if c == 0:
                        ptx.inst.mov.f32(macc[a], t)
                    else:
                        ptx.inst.max.f32(macc[a], macc[a], t)
                    s = reg.scalar(f32)
                    ptx.inst.add.f32(s, v[a], v[4 + a])
                    for i in range(8, 32, 4):
                        ptx.inst.add.f32(s, s, v[i + a])
                    if c == 0:
                        ptx.inst.mov.f32(sacc[a], s)
                    else:
                        ptx.inst.add.f32(sacc[a], sacc[a], s)

            def body(is_first: bool, po=None):
                ptx.mbarrier.wait(sfull_ad, ph_s)
                ptx.inst.xor.b32(ph_s, ph_s, 1)
                ptx.tcgen05.fence_after_thread_sync()

                # first chunk's load flies under the deferred tail
                a0 = reg.scalar(b32)
                ptx.inst.mov.b32(a0, sp_base)
                if s_f16:
                    # 64 unpacked f16-D columns -> 32 packed registers
                    ptx.tcgen05.ld([pwb[0][i] for i in range(32)], a0,
                                   shape="32x32b", count=32, dtype="b32",
                                   pack=True)
                else:
                    ptx.tcgen05.ld(bufs[0], a0, shape="32x32b", count=32, dtype="b32")

                if not is_first:
                    # ---- deferred tail of tile k-1: basis decision from
                    # this thread's own row max (no exchange) ----
                    alpha = reg.scalar(f32)
                    ptx.inst.mov.f32(alpha, one_f)
                    move = reg.scalar(pred)
                    ptx.inst.setp.gt.f32(move, pmax_pend, thresh_p)
                    with ptx.if_(move):
                        ptx.inst.rcp.approx.f32(alpha, pmax_pend)
                        lgp = reg.scalar(f32)
                        ptx.inst.lg2.approx.f32(lgp, pmax_pend)
                        ptx.inst.sub.f32(mneg, mneg, lgp)
                    ptx.inst.add.f32(l_run, l_run, sums_pend)
                    ptx.inst.mul.f32(l_run, l_run, alpha)

                    # publish alpha_{k-1}: gates PV(k) at parity k&1. The
                    # common-case O_RESC release goes first; the stats-slot
                    # publish (paced by correction) follows.
                    needs = reg.scalar(pred)
                    ptx.inst.setp.lt.f32(needs, alpha, one_f)
                    blt = reg.scalar(b32)
                    ptx.inst.vote.sync.ballot.b32(blt, needs, 0xFFFFFFFF)
                    all_skip = reg.scalar(pred)
                    ptx.inst.setp.eq.b32(all_skip, blt, 0)
                    ora = reg.scalar(u32)
                    if po is None:
                        ptx.inst.add.u32(ora, oresc_ad, 8)
                    else:
                        ptx.inst.add.u32(ora, oresc_ad, po)
                    ptx.mbarrier.arrive(ora, pred=all_skip)
                    ptx.mbarrier.wait(sfree_ad, ph_stats)
                    ptx.inst.xor.b32(ph_stats, ph_stats, 1)
                    ptx.inst.st.shared.b32(ptx.addr(alpha_ad), alpha)
                    ptx.inst.bar.arrive(stats_bar, 64)

                if emu_pairs:
                    ptx.inst.mov.b64(m2, [mneg, mneg])

                if s_f16:
                    # ---- stream 2 half-tile chunks (64 packed cols) ----
                    mh = reg.scalar(_b16)
                    ptx.inst.cvt.rn.f16.f32(mh, mneg)
                    m2h = reg.scalar(b32)
                    ptx.inst.mov.b32(m2h, [mh, mh])
                    for c in range(2):
                        ptx.tcgen05.wait_ld()
                        if c < 1:
                            an = reg.scalar(b32)
                            ptx.inst.add.u32(an, sp_base, 64)
                            ptx.tcgen05.ld([pwb[1][i] for i in range(32)], an,
                                           shape="32x32b", count=32,
                                           dtype="b32", pack=True)
                        w = pwb[c & 1]
                        for i in range(32):
                            ptx.inst.fma.rn.f16x2(w[i], w[i], q2h, m2h)
                            ptx.inst.ex2.approx.f16x2(w[i], w[i])
                        # results are already packed pairs: store directly
                        pa = reg.scalar(b32)
                        ptx.inst.add.u32(pa, sp_base, c * 32)
                        ptx.tcgen05.st(pa, [w[i] for i in range(32)],
                                       shape="32x32b", count=32, dtype="b32")
                        ptx.tcgen05.wait_st()
                        ptx.tcgen05.fence_before_thread_sync()
                        # this chunk is a HALF: arrive both of its quarter
                        # barriers so the consumer stays quarter-granular
                        for qq in (2 * c, 2 * c + 1):
                            qb = reg.scalar(u32)
                            ptx.inst.add.u32(qb, pq_ad, 8 * qq)
                            ptx.mbarrier.arrive(qb)
                        # fold into running f16x2 max/sum accs
                        for a in range(4):
                            t = reg.scalar(b32)
                            ptx.inst.max.f16x2(t, w[a], w[4 + a])
                            for i in range(8, 32, 4):
                                ptx.inst.max.f16x2(t, t, w[i + a])
                            if c == 0:
                                ptx.inst.mov.b32(macc16[a], t)
                            else:
                                ptx.inst.max.f16x2(macc16[a], macc16[a], t)
                            s = reg.scalar(b32)
                            ptx.inst.add.rn.f16x2(s, w[a], w[4 + a])
                            for i in range(8, 32, 4):
                                ptx.inst.add.rn.f16x2(s, s, w[i + a])
                            if c == 0:
                                ptx.inst.mov.b32(sacc16[a], s)
                            else:
                                ptx.inst.add.rn.f16x2(sacc16[a], sacc16[a], s)

                    # horizontal reduce (f16x2 -> f32 pending carries)
                    ptx.inst.max.f16x2(macc16[0], macc16[0], macc16[1])
                    ptx.inst.max.f16x2(macc16[2], macc16[2], macc16[3])
                    ptx.inst.max.f16x2(macc16[0], macc16[0], macc16[2])
                    h0 = reg.scalar(_b16); h1 = reg.scalar(_b16)
                    ptx.inst.mov.b32([h0, h1], macc16[0])
                    f0 = reg.scalar(f32); f1 = reg.scalar(f32)
                    ptx.inst.cvt.f32.f16(f0, h0)
                    ptx.inst.cvt.f32.f16(f1, h1)
                    ptx.inst.max.f32(pmax_pend, f0, f1)
                    ptx.inst.add.rn.f16x2(sacc16[0], sacc16[0], sacc16[1])
                    ptx.inst.add.rn.f16x2(sacc16[2], sacc16[2], sacc16[3])
                    sh = [reg.scalar(_b16) for _ in range(4)]
                    ptx.inst.mov.b32([sh[0], sh[1]], sacc16[0])
                    ptx.inst.mov.b32([sh[2], sh[3]], sacc16[2])
                    sf = [reg.scalar(f32) for _ in range(4)]
                    for k in range(4):
                        ptx.inst.cvt.f32.f16(sf[k], sh[k])
                    ptx.inst.add.f32(sf[0], sf[0], sf[1])
                    ptx.inst.add.f32(sf[2], sf[2], sf[3])
                    ptx.inst.add.f32(sums_pend, sf[0], sf[2])
                    return

                # ---- stream 4 chunks: ld(c+1) flies under exp/store(c) --
                for c in range(4):
                    ptx.tcgen05.wait_ld()
                    if c < 3:
                        an = reg.scalar(b32)
                        ptx.inst.add.u32(an, sp_base, (c + 1) * 32)
                        ptx.tcgen05.ld(bufs[(c + 1) & 1], an,
                                       shape="32x32b", count=32, dtype="b32")
                    v = bufs[c & 1]
                    exp_chunk(v)
                    for k in range(16):
                        ptx.inst.cvt.rn.bf16x2.f32(pk[k], v[2 * k + 1], v[2 * k])
                    pa = reg.scalar(b32)
                    ptx.inst.add.u32(pa, sp_base, c * 16)
                    ptx.tcgen05.st(pa, pk, shape="32x32b", count=16, dtype="b32")
                    # publish this quarter (32 columns) immediately; the
                    # wait_st round trip hides under the next chunk's
                    # in-flight load
                    ptx.tcgen05.wait_st()
                    ptx.tcgen05.fence_before_thread_sync()
                    qb = reg.scalar(u32)
                    ptx.inst.add.u32(qb, pq_ad, 8 * c)
                    ptx.mbarrier.arrive(qb)
                    chunk_reduce(v, c)

                # horizontal reduce into the pending carries
                ptx.inst.max.f32(macc[0], macc[0], macc[1], macc[2])
                ptx.inst.max.f32(pmax_pend, macc[0], macc[3])
                ptx.inst.add.f32(sacc[0], sacc[0], sacc[1])
                ptx.inst.add.f32(sacc[2], sacc[2], sacc[3])
                ptx.inst.add.f32(sums_pend, sacc[0], sacc[2])

            if emu_pairs:
                m2 = reg.scalar(b64)

            assert n_tiles >= 2, "stale-basis softmax needs n_tiles >= 2"
            siter_ph = reg.scalar(b32, init=0)
            sw = reg.scalar(u32); ptx.inst.mov.u32(sw, cta_id)
            sgo = reg.scalar(pred)
            ptx.inst.setp.lt.u32(sgo, sw, total_work)
            if dbg_epi_gate == "softmax":
                epi_ph = reg.scalar(b32, init=0)
            with ptx.loop("softmax_work_loop", pred=sgo):
                if dbg_epi_gate == "softmax":
                    ptx.mbarrier.wait(base + BAR_EPI, epi_ph)
                    ptx.inst.xor.b32(epi_ph, epi_ph, 1)
                ptx.inst.mov.f32(mneg, zero_f)
                ptx.inst.mov.f32(l_run, zero_f)
                body(is_first=True)
                it = reg.scalar(u32, init=1)
                go = reg.scalar(pred)
                ptx.inst.setp.lt.u32(go, it, n_tiles)
                with ptx.loop("softmax_loop", pred=go):
                    po = reg.scalar(u32)
                    ptx.inst.and_.b32(po, it, 1)
                    ptx.inst.shl.b32(po, po, 3)
                    body(is_first=False, po=po)
                    it += 1
                    ptx.inst.setp.lt.u32(go, it, n_tiles)

                # drain: fold the last tile's sums into l (its basis
                # decision never runs — the pending rescale cancels between
                # O and l in the epilogue normalize), publish the row sum,
                # STATS_FREE-gated like every body publish.
                ptx.inst.add.f32(l_run, l_run, sums_pend)
                ptx.inst.st.shared.b32(ptx.addr(sum_ad), l_run)
                ptx.mbarrier.wait(sfree_ad, ph_stats)
                ptx.inst.xor.b32(ph_stats, ph_stats, 1)
                ptx.inst.bar.arrive(stats_bar, 64)

                iter_sync(siter_ph, "softmax")
                ptx.inst.add.u32(sw, sw, num_ctas)
                ptx.inst.setp.lt.u32(sgo, sw, total_work)

        # =================================================================
        # Correction warps (tids 256-383): O rescale + epilogue
        # =================================================================
        with ptx.if_(is_correction):
            ptx.setmaxnreg(NREG_CORR, inc=False)
            from pyptx.types import b16 as _b16c
            one_f = reg.scalar(f32, init=1.0)
            stats_bar = [reg.scalar(u32) for _ in range(Q_STAGE)]
            alpha_addr = [reg.scalar(u32) for _ in range(Q_STAGE)]
            o_addr = [reg.scalar(b32) for _ in range(Q_STAGE)]
            for s in range(Q_STAGE):
                ptx.inst.add.u32(stats_bar[s], warp_in_group, 1 + s * 4)
                ptx.inst.shl.b32(alpha_addr[s], row128, 2)
                ptx.inst.add.u32(alpha_addr[s], alpha_addr[s], base + SMEM_STATS + s * 512)
                ptx.inst.mov.b32(o_addr[s], tmem)
                ptx.inst.add.u32(o_addr[s], o_addr[s], TM_O[s])
                ptx.inst.add.u32(o_addr[s], o_addr[s], lane_addr_bits)

            # first KV tile needs no correction: release both stages once
            # (parity-0 barrier gates PV(0)); also pre-arrive STATS_FREE so
            # the softmax's pre-write wait at tile 0 is already satisfied
            for s in range(Q_STAGE):
                ptx.mbarrier.arrive(base + BAR_O_RESC + 16 * s)
                ptx.mbarrier.arrive(base + BAR_STATS_FREE + 8 * s)
            if dbg_epi_gate:
                ptx.mbarrier.arrive(base + BAR_EPI)

            ovals = reg.array(f32, 64)
            ph_pv = [reg.scalar(b32, init=0) for _ in range(Q_STAGE)]
            done_phase = [reg.scalar(b32, init=0) for _ in range(Q_STAGE)]
            (po,) = ptx.global_ptrs(O)
            opacked = reg.array(b32, 16)

            citer_ph = reg.scalar(b32, init=0)
            cw = reg.scalar(u32); ptx.inst.mov.u32(cw, cta_id)
            cgo = reg.scalar(pred)
            ptx.inst.setp.lt.u32(cgo, cw, total_work)
            with ptx.loop("corr_work_loop", pred=cgo):
              # alpha published at tile i (i = 0..n_tiles-2; the last tile
              # publishes nothing) rescales O between PV(i) and PV(i+1).
              # The wait for PV(i)'s completion happens EVERY iteration
              # (correction idles here anyway) so the phase var tracks the
              # true completion count — a parity-only wait inside the rare
              # branch could alias an older same-parity completion and
              # race the in-flight PV.
              it = reg.scalar(u32, init=0)
              go = reg.scalar(pred)
              ptx.inst.setp.lt.u32(go, it, n_tiles - 1)
              with ptx.loop("corr_loop", pred=go):
                ponc = reg.scalar(u32)
                ptx.inst.add.u32(ponc, it, 1)
                ptx.inst.and_.b32(ponc, ponc, 1)
                ptx.inst.shl.b32(ponc, ponc, 3)
                for stage in range(Q_STAGE):
                    ptx.bar.sync(stats_bar[stage], 64)
                    alpha = reg.scalar(f32)
                    av = smem.load(b32, ptx.addr(alpha_addr[stage]))
                    ptx.inst.mov.b32(alpha, av)
                    ptx.mbarrier.arrive(base + BAR_STATS_FREE + 8 * stage)
                    ptx.mbarrier.wait(base + BAR_PV_DONE + 8 * stage, ph_pv[stage])
                    ptx.inst.xor.b32(ph_pv[stage], ph_pv[stage], 1)
                    need = reg.scalar(pred)
                    ptx.inst.setp.lt.f32(need, alpha, one_f)
                    ballot = reg.scalar(b32)
                    ptx.inst.vote.sync.ballot.b32(ballot, need, 0xFFFFFFFF)
                    any_need = reg.scalar(pred)
                    ptx.inst.setp.ne.b32(any_need, ballot, 0)
                    # softmax already released O_RESC when nothing rescales
                    with ptx.if_(any_need):
                        ptx.tcgen05.fence_after_thread_sync()
                        for half in range(2):
                            addr = reg.scalar(b32)
                            ptx.inst.add.u32(addr, o_addr[stage], half * 64)
                            ptx.tcgen05.ld(ovals, addr, shape="32x32b", count=64, dtype="b32")
                            ptx.tcgen05.wait_ld()
                            for i in range(64):
                                ptx.inst.mul.f32(ovals[i], ovals[i], alpha)
                            ptx.tcgen05.st(addr, ovals, shape="32x32b", count=64, dtype="b32")
                        ptx.tcgen05.wait_st()
                        ptx.tcgen05.fence_before_thread_sync()
                        ora = reg.scalar(u32)
                        ptx.inst.add.u32(
                            ora, ponc, base + BAR_O_RESC + 16 * stage
                        )
                        ptx.mbarrier.arrive(ora)
                it += 1
                ptx.inst.setp.lt.u32(go, it, n_tiles - 1)

              # drain: consume the last tile's PV_DONE completion so the
              # per-work-item wait/completion counts stay exactly balanced
              # (unconsumed completions would drift the parity waits)
              for stage in range(Q_STAGE):
                  ptx.mbarrier.wait(base + BAR_PV_DONE + 8 * stage, ph_pv[stage])
                  ptx.inst.xor.b32(ph_pv[stage], ph_pv[stage], 1)

              # arrive the work-item boundary BEFORE the epilogue: the
              # DRAM-bound O readout then overlaps the next item's QK and
              # softmax, while the next PV(0) stays gated on the
              # post-epilogue O_RESC re-arm
              iter_sync(citer_ph, "corr")

              # ---- epilogue: wait all PV done, normalize, store bf16 ----
              ptx.mbarrier.wait(base + BAR_O_DONE, done_phase[0])
              ptx.inst.xor.b32(done_phase[0], done_phase[0], 1)
              ptx.tcgen05.fence_after_thread_sync()

              q_row0, _ = derive_rows(cw, want_kv=False)
              out_row = reg.scalar(u32)
              ptx.inst.add.u32(out_row, q_row0, row128)

              for stage in range(Q_STAGE):
                ptx.bar.sync(stats_bar[stage], 64)
                # streaming softmax publishes one full-row sum per row
                la = smem.load(b32, ptx.addr(alpha_addr[stage] + 1024))
                l = reg.scalar(f32)
                ptx.inst.mov.b32(l, la)
                inv_l = reg.scalar(f32)
                ptx.inst.rcp.approx.f32(inv_l, l)

                row = reg.scalar(u32)
                ptx.inst.add.u32(row, out_row, stage * BM)
                byte_off = reg.scalar(u64)
                ptx.inst.mul.wide.u32(byte_off, row, HD * 2)
                gptr = po + byte_off
                if dbg_f16_noexp:
                    # debug build: store raw O (no normalization)
                    ptx.inst.mov.f32(inv_l, reg.scalar(f32, init=1.0))

                for half in range(2):
                    addr = reg.scalar(b32)
                    ptx.inst.add.u32(addr, o_addr[stage], half * 64)
                    ptx.tcgen05.ld(ovals, addr, shape="32x32b", count=64, dtype="b32")
                    ptx.tcgen05.wait_ld()
                    if stage == 1 and half == 1:
                        # the LAST TMEM read is complete: everything the
                        # gates protect (the readout — TC writes to O or
                        # softmax TMEM traffic during it corrupt rows) is
                        # done, and the remaining normalize/pack/store is
                        # register->global work. Release the next item
                        # now so it overlaps the store tail.
                        ptx.mbarrier.arrive(base + BAR_STATS_FREE + 8)
                        if dbg_epi_gate:
                            ptx.mbarrier.arrive(base + BAR_EPI)
                        for s in range(Q_STAGE):
                            ptx.mbarrier.arrive(base + BAR_O_RESC + 16 * s)
                    for i in range(64):
                        ptx.inst.mul.f32(ovals[i], ovals[i], inv_l)
                    for q in range(2):
                        if s_f16:
                            for c in range(16):
                                ha = reg.scalar(_b16c); hb = reg.scalar(_b16c)
                                ptx.inst.cvt.rn.f16.f32(ha, ovals[32 * q + 2 * c])
                                ptx.inst.cvt.rn.f16.f32(hb, ovals[32 * q + 2 * c + 1])
                                ptx.inst.mov.b32(opacked[c], [ha, hb])
                        else:
                            for c in range(16):
                                ptx.inst.cvt.rn.bf16x2.f32(
                                    opacked[c],
                                    ovals[32 * q + 2 * c + 1], ovals[32 * q + 2 * c]
                                )
                        for vec in range(4):
                            ptx.inst.st.global_.v4.b32(
                                ptx.addr(gptr, half * 128 + q * 64 + vec * 16),
                                [opacked[vec * 4 + k] for k in range(4)],
                            )
                if stage == 0:
                    # stage-0 sums slot consumed: return the credit
                    ptx.mbarrier.arrive(base + BAR_STATS_FREE)

              ptx.inst.add.u32(cw, cw, num_ctas)
              ptx.inst.setp.lt.u32(cgo, cw, total_work)

        ptx.bar.sync(0, 512)
        if waitmap:
            _wm_ctx.__exit__(None, None, None)
            is_cta0 = reg.scalar(pred)
            ptx.inst.setp.eq.u32(is_cta0, cta_id, 0)
            wm.flush(pred=is_cta0)
        with ptx.if_(alloc_warp):
            ptx.tcgen05.dealloc(tmem, 512)
            ptx.tcgen05.relinquish_alloc_permit()
        ptx.ret()

    if waitmap:
        @kernel(**kernel_kwargs)
        def flash_attn_fwd(Q, K, V, O, WM):
            _fa_body(Q, K, V, O, WM)
    else:
        @kernel(**kernel_kwargs)
        def flash_attn_fwd(Q, K, V, O):
            _fa_body(Q, K, V, O, None)

    flash_attn_fwd._waitmap = wm  # decoder access for tooling
    return flash_attn_fwd




# =====================================================================
# 2-CTA (cta_group::2) variant — FA4's shipping configuration for
# non-causal head_dim=128. A cluster of 2 CTAs covers 512 query rows;
# each K/V tile is split across the pair (K by KV rows, V by head-dim
# columns), halving per-CTA SMEM traffic. The lead CTA's MMA warp
# drives 256-row tcgen05.mma instructions for the whole pair.
# =====================================================================

KV_HALF_BYTES = TILE_BYTES // 2      # 16 KB: one CTA's half of a K/V tile
SLOT_UNITS_2 = KV_HALF_BYTES // 16   # ring-slot stride in descriptor units

SMEM2_Q = 0
SMEM2_KV = SMEM2_Q + Q_BYTES                     # 65536
SMEM2_STATS = SMEM2_KV + KV_SLOTS * KV_HALF_BYTES  # 131072
#   alpha[2][128] @ +0, sum[2][128] @ +1024
SMEM2_BARS = SMEM2_STATS + 2048
B2_Q = SMEM2_BARS + 0             # 8   (lead-tracked, both CTAs' Q)
B2_KV_FULL = SMEM2_BARS + 8       # 4 x 8 (lead-tracked)
B2_KV_FREE = SMEM2_BARS + 40      # 4 x 8 (local, multicast commit)
B2_S_FULL = SMEM2_BARS + 72       # 2 x 8 (local, multicast commit)
B2_P_FULL = SMEM2_BARS + 88       # [stage][half]: 4 x 8 (lead, count 256)
B2_O_RESC = SMEM2_BARS + 120      # 2 x 8 (lead, count 256)
B2_O_DONE = SMEM2_BARS + 136      # 8   (local, multicast commit)
B2_STATS_FREE = SMEM2_BARS + 144  # 2 x 8 (local)
B2_STATS_FULL = SMEM2_BARS + 160  # 2 x 8 (local)
B2_Q_FREE = SMEM2_BARS + 176      # 8 (persistent: Q smem free, local per CTA)
B2_EPI = SMEM2_BARS + 184         # 8 (softmax item-start gate, local per CTA)
SMEM2_TMEM_SLOT = SMEM2_BARS + 192
SMEM2_BYTES = SMEM2_TMEM_SLOT + 16


def _khalf_desc_off(kk: int) -> int:
    """K-atom offset for a CTA's 64-row K half (two 8 KB stripes)."""
    return (kk // 4) * 512 + (kk % 4) * 2


def build_flash_attention_blackwell_2cta(
    seqlen: int,
    batch_heads: int,
    head_dim: int = 128,
    *,
    sm_scale: float | None = None,
    arch: str = "sm_100a",
    rescale_threshold: float = RESCALE_THRESHOLD,
    debug_plumbing: bool = False,
    debug_blind_feed: bool = False,
    debug_pv_smem_a: bool = False,
    debug_paced_sim: bool = False,
    debug_mma_level: int = 4,
    debug_beacon: bool = False,
    waitmap: bool = False,
):
    assert head_dim == HD, "2-CTA variant supports head_dim=128 only"
    assert seqlen % 512 == 0, f"seqlen must be a multiple of 512, got {seqlen}"
    n_tiles = seqlen // BN
    assert n_tiles % 2 == 0
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * LOG2E
    total_rows = batch_heads * seqlen

    # persistent scheduling over CLUSTER work items (512 Q rows each)
    n_mblocks_c = seqlen // 512
    assert n_mblocks_c & (n_mblocks_c - 1) == 0
    mbc_shift = n_mblocks_c.bit_length() - 1
    total_work = n_mblocks_c * batch_heads
    num_clusters = min(total_work, 74)

    from pyptx.debugkit import WAITMAP_MAX_SITES, WaitMap
    wm = WaitMap() if waitmap else None

    kernel_kwargs = dict(
        in_specs=(
            Tile(total_rows, HD, bf16, Layout.TMA_128B, tma_box=(BM, 64)),
            Tile(total_rows, HD, bf16, Layout.TMA_128B, tma_box=(64, 64)),
            Tile(total_rows, HD, bf16, Layout.TMA_128B, tma_box=(BN, 64)),
        ),
        grid=(2 * num_clusters, 1, 1),
        cluster=(2, 1, 1),
        block=(512, 1, 1),
        arch=arch,
        smem=SMEM2_BYTES,
        extern_smem=True,
        raw_directives=[
            ("reqntid", (512, 1, 1)),
            ("explicitcluster", ()),
            ("reqnctapercluster", (2, 1, 1)),
        ],
    )
    out_o = Tile(total_rows, HD, bf16, Layout.ROW)
    if waitmap:
        kernel_kwargs["out_specs"] = (
            out_o, Tile(512, WAITMAP_MAX_SITES, b32, Layout.ROW))
    else:
        kernel_kwargs["out_specs"] = (out_o,)

    def _fa2_body(Q, K, V, O, WM=None):
        base = smem.base()
        tid = reg.scalar(u32); ptx.inst.mov.u32(tid, ptx.special.tid.x())
        if waitmap:
            (pwm,) = ptx.global_ptrs(WM)
            wm.attach(pwm, tid)
            _wm_ctx = wm.active()
            _wm_ctx.__enter__()
        cluster_x = reg.scalar(u32)
        ptx.inst.mov.u32(cluster_x, ptx.special.ctaid.x())
        ptx.inst.shr.u32(cluster_x, cluster_x, 1)
        cta_rank = reg.scalar(u32)
        ptx.inst.mov.u32(cta_rank, ptx.sreg("%cluster_ctarank"))
        is_lead = reg.scalar(pred)
        ptx.inst.setp.eq.u32(is_lead, cta_rank, 0)
        rank0 = reg.scalar(u32, init=0)

        warp = tid >> 5
        row128 = reg.scalar(u32); ptx.inst.and_.b32(row128, tid, 127)
        lane_addr_bits = reg.scalar(u32)
        ptx.inst.and_.b32(lane_addr_bits, tid, 96)
        ptx.inst.shl.b32(lane_addr_bits, lane_addr_bits, 16)
        warp_in_group = reg.scalar(u32)
        ptx.inst.and_.b32(warp_in_group, warp, 3)

        is_softmax0 = reg.scalar(pred); ptx.inst.setp.lt.u32(is_softmax0, tid, 128)
        is_softmax1 = reg.scalar(pred)
        is_correction = reg.scalar(pred)
        with ptx.scope():
            ge = reg.scalar(pred); lt = reg.scalar(pred)
            ptx.inst.setp.ge.u32(ge, tid, 128)
            ptx.inst.setp.lt.u32(lt, tid, 256)
            ptx.inst.and_.pred(is_softmax1, ge, lt)
            ptx.inst.setp.ge.u32(ge, tid, 256)
            ptx.inst.setp.lt.u32(lt, tid, 384)
            ptx.inst.and_.pred(is_correction, ge, lt)
        is_mma = reg.scalar(pred)
        with ptx.scope():
            t = reg.scalar(pred)
            ptx.inst.setp.eq.u32(t, tid, MMA_TID)
            ptx.inst.and_.pred(is_mma, t, is_lead)
        is_load = reg.scalar(pred); ptx.inst.setp.eq.u32(is_load, tid, LOAD_TID)
        alloc_warp = reg.scalar(pred)
        ptx.inst.setp.eq.u32(alloc_warp, warp, 12)
        _never = reg.scalar(pred)
        with ptx.scope():
            zz = reg.scalar(u32, init=0)
            ptx.inst.setp.ne.u32(_never, zz, 0)

        # Row bases per work item. A cluster covers 512 query rows: stage
        # s of the pair is rows [pair_base + s*256, +256), and this CTA
        # owns the rank half. bh = w >> mbc_shift, mb = w & (n_mblocks_c-1).
        rank_off = reg.scalar(u32)
        ptx.inst.mul.lo.u32(rank_off, cta_rank, BM)

        def derive_rows2(w):
            q_row = reg.scalar(u32)
            kv_row = reg.scalar(u32)
            with ptx.scope():
                mb = reg.scalar(u32)
                ptx.inst.and_.b32(mb, w, n_mblocks_c - 1)
                bhw = reg.scalar(u32)
                ptx.inst.shr.u32(bhw, w, mbc_shift)
                ptx.inst.mul.lo.u32(kv_row, bhw, seqlen)
                ptx.inst.mul.lo.u32(mb, mb, 2 * Q_STAGE * BM)
                ptx.inst.add.u32(q_row, kv_row, mb)
                ptx.inst.add.u32(q_row, q_row, rank_off)
            return q_row, kv_row

        # ---- init barriers, allocate cluster TMEM ----
        with ptx.if_(tid == 0):
            ptx.mbarrier.init(base + B2_Q, 1)
            for s in range(KV_SLOTS):
                ptx.mbarrier.init(base + B2_KV_FULL + 8 * s, 1)
                ptx.mbarrier.init(base + B2_KV_FREE + 8 * s, 1)
            for i in range(2):
                ptx.mbarrier.init(base + B2_S_FULL + 8 * i, 1)
                for h in range(2):
                    ptx.mbarrier.init(base + B2_P_FULL + 16 * i + 8 * h, 256)
                ptx.mbarrier.init(base + B2_O_RESC + 8 * i, 256)
                ptx.mbarrier.init(base + B2_STATS_FREE + 8 * i, 128)
                ptx.mbarrier.init(base + B2_STATS_FULL + 8 * i, 128)
            ptx.mbarrier.init(base + B2_O_DONE, 1)
            ptx.mbarrier.init(base + B2_Q_FREE, 1)
            ptx.mbarrier.init(base + B2_EPI, 128)
            ptx.fence.proxy_async_shared_cta()
        with ptx.if_(alloc_warp):
            ptx.tcgen05.alloc(base + SMEM2_TMEM_SLOT, 512, cta_group=2)
        ptx.cluster.sync()
        tmem = smem.load(b32, ptx.addr(base + SMEM2_TMEM_SLOT))

        _bw_cnt = [0]
        beaconed = reg.scalar(pred) if debug_beacon else None
        if debug_beacon:
            with ptx.scope():
                z = reg.scalar(u32, init=0)
                ptx.inst.setp.ne.u32(beaconed, z, 0)

        def progress(marker: int, itreg=None):
            if not debug_beacon:
                return
            (pob,) = ptx.global_ptrs(O)
            idx = reg.scalar(u32)
            ptx.inst.mad.lo.u32(idx, m_block, 512, tid)
            ptx.inst.add.u32(idx, idx, 1024)
            off = reg.scalar(u64)
            ptx.inst.mul.wide.u32(off, idx, 4)
            val = reg.scalar(b32, init=0xB0B00000 | (marker << 8))
            if itreg is not None:
                ptx.inst.or_.b32(val, val, itreg)
            ptx.inst.st.global_.b32(ptx.addr(pob + off), val)

        def bwait(addr, ph, site: int, extra: int = 0):
            if not debug_beacon:
                ptx.mbarrier.wait(addr, ph)
                return
            _bw_cnt[0] += 1
            lbl = f"bw_{site}_{extra}_{_bw_cnt[0]}"
            tries = reg.scalar(u32, init=0)
            ptx.label(lbl)
            pdone = ptx.mbarrier.try_wait(addr, ph)
            ptx.bra(lbl + "_ok", pred=pdone)
            ptx.inst.add.u32(tries, tries, 1)
            keep = reg.scalar(pred)
            ptx.inst.setp.lt.u32(keep, tries, 3000000)
            ptx.bra(lbl, pred=keep)
            # timed out: record first-blame beacon, then pretend success so
            # the pipeline drains and the kernel exits cleanly
            ptx.bra(lbl + "_skipw", pred=beaconed)
            (pob,) = ptx.global_ptrs(O)
            idx = reg.scalar(u32)
            ptx.inst.mad.lo.u32(idx, m_block, 512, tid)
            off = reg.scalar(u64)
            ptx.inst.mul.wide.u32(off, idx, 4)
            val = reg.scalar(b32, init=0xBEAC0000 | (site << 8) | extra)
            ptx.inst.st.global_.b32(ptx.addr(pob + off), val)
            with ptx.scope():
                o1 = reg.scalar(u32, init=1)
                ptx.inst.setp.ne.u32(beaconed, o1, 0)
            ptx.label(lbl + "_skipw")
            ptx.label(lbl + "_ok")

        NREG_SOFTMAX = 192
        NREG_CORR = 72
        NREG_OTHER = 56

        # =================================================================
        # TMA load warp: each CTA loads its own Q rows, its 64-KV-row half
        # of K, and its 64-head-dim-col half of V. Completions land on the
        # lead CTA's barriers (collective TMA).
        # =================================================================
        is_other_wg = reg.scalar(pred)
        ptx.inst.setp.ge.u32(is_other_wg, tid, 384)
        with ptx.if_(is_other_wg):
            ptx.setmaxnreg(NREG_OTHER, inc=False)

        with ptx.if_(is_load):
            mapped_q = ptx.cluster.map_shared_u32(base + B2_Q, 0)

            def collective_load(dst_off, tensor, coord, mapped_bar):
                ptx.cp.async_.bulk.tensor_2d.shared_cta_global_tile(
                    dst=base + dst_off,
                    src=tensor.tma_desc(),
                    coord=coord,
                    mbar=mapped_bar,
                    cta_group=2,
                )

            mapped_kv = [
                ptx.cluster.map_shared_u32(base + B2_KV_FULL + 8 * s, 0)
                for s in range(KV_SLOTS)
            ]

            # self-credit the ring and the Q buffer once: every load
            # (including each item's prologue) waits FREE first, keeping
            # the accounting uniform across persistent work items
            for s in range(KV_SLOTS):
                ptx.mbarrier.arrive(base + B2_KV_FREE + 8 * s)
            ptx.mbarrier.arrive(base + B2_Q_FREE)
            free_phase = [reg.scalar(b32, init=0) for _ in range(KV_SLOTS)]
            q_free_phase = reg.scalar(b32, init=0)

            krow = reg.scalar(u32)
            vrow = reg.scalar(u32)

            def load_k_half(slot):
                with ptx.if_(is_lead):
                    ptx.mbarrier.arrive_expect_tx(base + B2_KV_FULL + 8 * slot, TILE_BYTES)
                for stripe in range(2):
                    collective_load(
                        SMEM2_KV + slot * KV_HALF_BYTES + stripe * (KV_HALF_BYTES // 2),
                        K, (stripe * 64, krow), mapped_kv[slot],
                    )
                ptx.inst.add.u32(krow, krow, BN)

            def load_v_half(slot):
                # constant inner coordinate per rank (rank 1 takes head-dim
                # columns 64-127)
                with ptx.if_(is_lead):
                    ptx.mbarrier.arrive_expect_tx(base + B2_KV_FULL + 8 * slot, TILE_BYTES)
                with ptx.if_(is_lead):
                    collective_load(
                        SMEM2_KV + slot * KV_HALF_BYTES, V, (0, vrow), mapped_kv[slot],
                    )
                with ptx.else_():
                    collective_load(
                        SMEM2_KV + slot * KV_HALF_BYTES, V, (64, vrow), mapped_kv[slot],
                    )
                ptx.inst.add.u32(vrow, vrow, BN)

            def load_kv_gated(slot, fn):
                bwait(base + B2_KV_FREE + 8 * slot, free_phase[slot], 8, slot)
                free_phase[slot] ^= 1
                # cluster-scope proxy acquire before reusing a
                # collective TMA slot (Mosaic/CUTLASS pattern)
                ptx.fence.proxy_async_generic_acquire_shared_cluster()
                fn(slot)

            lw = reg.scalar(u32); ptx.inst.mov.u32(lw, cluster_x)
            lgo = reg.scalar(pred)
            ptx.inst.setp.lt.u32(lgo, lw, total_work)
            with ptx.loop("load2_work_loop", pred=lgo):
                q_row0, kv_row0 = derive_rows2(lw)
                ptx.inst.mul.lo.u32(krow, cta_rank, 64)
                ptx.inst.add.u32(krow, krow, kv_row0)  # +64 rows for rank 1
                ptx.inst.mov.u32(vrow, kv_row0)

                # Q: 2 stages x 2 stripes for this CTA's rows, gated on
                # the previous item's QKs having drained
                ptx.mbarrier.wait(base + B2_Q_FREE, q_free_phase)
                ptx.inst.xor.b32(q_free_phase, q_free_phase, 1)
                ptx.fence.proxy_async_generic_acquire_shared_cluster()
                with ptx.if_(is_lead):
                    ptx.mbarrier.arrive_expect_tx(base + B2_Q, 4 * TILE_BYTES)
                q_row1 = reg.scalar(u32)
                ptx.inst.add.u32(q_row1, q_row0, 2 * BM)   # stage 1 rows
                for s, rr in ((0, q_row0), (1, q_row1)):
                    for stripe in range(2):
                        collective_load(
                            SMEM2_Q + s * TILE_BYTES + stripe * STRIPE_BYTES,
                            Q, (stripe * 64, rr), mapped_q,
                        )

                load_kv_gated(0, load_k_half)
                load_kv_gated(1, load_v_half)
                load_kv_gated(2, load_k_half)
                load_kv_gated(3, load_v_half)

                trips = (n_tiles - 2) // 2
                if trips > 0:
                    trip = reg.scalar(u32, init=0)
                    go = reg.scalar(pred)
                    ptx.inst.setp.lt.u32(go, trip, trips)
                    with ptx.loop("kv2_load_loop", pred=go):
                        for slot, fn in ((0, load_k_half), (1, load_v_half),
                                         (2, load_k_half), (3, load_v_half)):
                            load_kv_gated(slot, fn)
                        trip += 1
                        ptx.inst.setp.lt.u32(go, trip, trips)

                ptx.inst.add.u32(lw, lw, num_clusters)
                ptx.inst.setp.lt.u32(lgo, lw, total_work)

        # =================================================================
        # MMA dispatch warp — lead CTA only. m=256 instructions span the
        # pair; descriptors address each CTA's local SMEM symmetrically.
        # =================================================================
        with ptx.if_(is_mma):
            idesc_qk = reg.scalar(
                b32,
                init=ptx.tcgen05.make_instr_desc_f16bf16_f32(
                    m=256, n=BN, ab_dtype="bf16", a_major="K", b_major="K"
                ),
            )
            idesc_pv = reg.scalar(
                b32,
                init=ptx.tcgen05.make_instr_desc_f16bf16_f32(
                    m=256, n=HD, ab_dtype="bf16", a_major="K", b_major="MN"
                ),
            )
            p_acc = reg.scalar(pred)
            p_noacc = reg.scalar(pred)
            with ptx.scope():
                one = reg.scalar(u32, init=1)
                zero = reg.scalar(u32, init=0)
                ptx.inst.setp.ne.b32(p_acc, one, 0)
                ptx.inst.setp.ne.b32(p_noacc, zero, 0)
            from pyptx.types import b16
            mcast = reg.scalar(b16)
            ptx.inst.mov.b16(mcast, 3)

            dq0 = ptx.tcgen05.masked_descriptor(base + SMEM2_Q)
            dk0 = ptx.tcgen05.masked_descriptor(base + SMEM2_KV)
            dv0 = ptx.tcgen05.descriptor(
                base + SMEM2_KV + KV_HALF_BYTES,
                stride_bytes=1024, leading_bytes=16384, swizzle="128B",
            )

            def qk_mma(stage: int, k_idx: int):
                d = tmem + TM_S[stage]
                for kk in range(8):
                    if debug_mma_level < 1:
                        break
                    off_a = stage * SLOT_UNITS + _kmajor_desc_off(kk)
                    off_b = k_idx * 2 * SLOT_UNITS_2 + _khalf_desc_off(kk)
                    da = dq0 if off_a == 0 else reg.scalar(b64)
                    if off_a:
                        ptx.inst.add.s64(da, dq0, off_a)
                    db = dk0 if off_b == 0 else reg.scalar(b64)
                    if off_b:
                        ptx.inst.add.s64(db, dk0, off_b)
                    ptx.tcgen05.mma(
                        d, da, db, idesc_qk, kind="f16", cta_group=2,
                        pred_operand=(p_noacc if kk == 0 else p_acc),
                    )
                ptx.tcgen05.commit(
                    base + B2_S_FULL + 8 * stage, cta_group=2,
                    multicast=True, multicast_mask=mcast, space="cluster",
                )

            ph_kv = [reg.scalar(b32, init=0) for _ in range(KV_SLOTS)]
            ph_p = [
                [reg.scalar(b32, init=0) for _ in range(2)] for _ in range(Q_STAGE)
            ]
            ph_or = [reg.scalar(b32, init=0) for _ in range(Q_STAGE)]

            def wait_kv(slot: int):
                bwait(base + B2_KV_FULL + 8 * slot, ph_kv[slot], 2, slot)
                ph_kv[slot] ^= 1

            def commit_free(slot: int):
                ptx.tcgen05.commit(
                    base + B2_KV_FREE + 8 * slot, cta_group=2,
                    multicast=True, multicast_mask=mcast, space="cluster",
                )

            def pv_mma(stage: int, v_idx: int, first_accum: bool):
                d = tmem + TM_O[stage]
                # consume P per 64-column half: kk 0-3 start while the
                # softmax groups still stream the second half
                bwait(base + B2_P_FULL + 16 * stage, ph_p[stage][0], 3, stage)
                ph_p[stage][0] ^= 1
                bwait(base + B2_O_RESC + 8 * stage, ph_or[stage], 4, stage)
                ph_or[stage] ^= 1
                ptx.tcgen05.fence_after_thread_sync()
                for kk in range(8):
                    if debug_mma_level < 2:
                        break
                    if kk == 4:
                        bwait(base + B2_P_FULL + 16 * stage + 8,
                              ph_p[stage][1], 3, stage)
                        ph_p[stage][1] ^= 1
                        ptx.tcgen05.fence_after_thread_sync()
                    a = tmem + (TM_P[stage] + kk * 8)
                    off_b = v_idx * 2 * SLOT_UNITS_2 + kk * 128
                    db = dv0 if off_b == 0 else reg.scalar(b64)
                    if off_b:
                        ptx.inst.add.s64(db, dv0, off_b)
                    if debug_mma_level == 2:
                        # PV with all-K-major operands (dq0/dk0, QK idesc)
                        ptx.tcgen05.mma(
                            d, dq0, dk0, idesc_qk, kind="f16", cta_group=2,
                            pred_operand=(p_noacc if (first_accum and kk == 0) else p_acc),
                        )
                    elif debug_mma_level == 3:
                        # PV with SMEM A + real MN-major V descriptor
                        ptx.tcgen05.mma(
                            d, dq0, db, idesc_pv, kind="f16", cta_group=2,
                            pred_operand=(p_noacc if (first_accum and kk == 0) else p_acc),
                        )
                    elif debug_pv_smem_a:
                        ptx.tcgen05.mma(
                            d, dq0, db, idesc_pv, kind="f16", cta_group=2,
                            pred_operand=(p_noacc if (first_accum and kk == 0) else p_acc),
                        )
                    else:
                        ptx.tcgen05.mma(
                            d, a, db, idesc_pv, kind="f16", cta_group=2,
                            a_is_tmem=True,
                            pred_operand=(p_noacc if (first_accum and kk == 0) else p_acc),
                        )

            ph_q = reg.scalar(b32, init=0)
            mw = reg.scalar(u32); ptx.inst.mov.u32(mw, cluster_x)
            mgo = reg.scalar(pred)
            ptx.inst.setp.lt.u32(mgo, mw, total_work)
            with ptx.loop("mma2_work_loop", pred=mgo):
                # ---- prologue ----
                bwait(base + B2_Q, ph_q, 1)
                ptx.inst.xor.b32(ph_q, ph_q, 1)
                wait_kv(0)
                qk_mma(0, 0)
                qk_mma(1, 0)
                commit_free(0)

                wait_kv(1)
                wait_kv(2)
                pv_mma(0, 0, first_accum=True)
                qk_mma(0, 1)
                pv_mma(1, 0, first_accum=True)
                qk_mma(1, 1)
                commit_free(2)
                commit_free(1)

                trips = (n_tiles - 2) // 2
                if trips > 0:
                    trip = reg.scalar(u32, init=0)
                    go = reg.scalar(pred)
                    ptx.inst.setp.lt.u32(go, trip, trips)
                    with ptx.loop("mma2_loop", pred=go):
                        wait_kv(3)
                        wait_kv(0)
                        for stage in range(Q_STAGE):
                            pv_mma(stage, 1, first_accum=False)
                            qk_mma(stage, 0)
                        commit_free(0)
                        commit_free(3)
                        wait_kv(1)
                        wait_kv(2)
                        for stage in range(Q_STAGE):
                            pv_mma(stage, 0, first_accum=False)
                            qk_mma(stage, 1)
                        commit_free(2)
                        commit_free(1)
                        trip += 1
                        ptx.inst.setp.lt.u32(go, trip, trips)

                # all QKs for this item are issued: release BOTH CTAs' Q
                # buffers so the next item's Q TMA overlaps the tail
                ptx.tcgen05.commit(
                    base + B2_Q_FREE, cta_group=2,
                    multicast=True, multicast_mask=mcast, space="cluster",
                )

                wait_kv(3)
                for stage in range(Q_STAGE):
                    pv_mma(stage, 1, first_accum=False)
                commit_free(3)
                ptx.tcgen05.commit(
                    base + B2_O_DONE, cta_group=2,
                    multicast=True, multicast_mask=mcast, space="cluster",
                )

                ptx.inst.add.u32(mw, mw, num_clusters)
                ptx.inst.setp.lt.u32(mgo, mw, total_work)

        # =================================================================
        # Softmax groups (per CTA): stage 0 = tids 0-127, stage 1 = 128-255.
        # Single pass, thread-local full row. P_FULL arrives go to the lead
        # CTA's barrier (count 256 across the pair).
        # =================================================================
        def softmax_group(stage: int):
            s_lo = reg.scalar(b32)
            ptx.inst.mov.b32(s_lo, tmem)
            ptx.inst.add.u32(s_lo, s_lo, TM_S[stage])
            ptx.inst.add.u32(s_lo, s_lo, lane_addr_bits)
            s_hi = reg.scalar(b32)
            ptx.inst.add.u32(s_hi, s_lo, 64)
            p_lo = reg.scalar(b32)
            ptx.inst.mov.b32(p_lo, tmem)
            ptx.inst.add.u32(p_lo, p_lo, TM_P[stage])
            ptx.inst.add.u32(p_lo, p_lo, lane_addr_bits)

            p_half_addr = [reg.scalar(u32, init=0) for _ in range(2)]
            for h in range(2):
                ptx.inst.add.u32(
                    p_half_addr[h], p_half_addr[h],
                    base + B2_P_FULL + 16 * stage + 8 * h,
                )

            qk_scale_reg = reg.scalar(f32, init=qk_scale)
            thresh_p = reg.scalar(f32, init=float(2.0 ** rescale_threshold))
            one_f = reg.scalar(f32, init=1.0)
            # stale-basis state (see the 1-CTA builder): exp against the
            # pending basis immediately; the max/decision defers one tile
            mneg = reg.scalar(f32, init=0.0)
            pmax_pend = reg.scalar(f32, init=0.0)
            sums_pend = reg.scalar(f32, init=0.0)
            l_run = reg.scalar(f32, init=0.0)

            alpha_addr = reg.scalar(u32)
            ptx.inst.shl.b32(alpha_addr, row128, 2)
            ptx.inst.add.u32(alpha_addr, alpha_addr, base + SMEM2_STATS + stage * 512)
            sum_addr = reg.scalar(u32)
            ptx.inst.add.u32(sum_addr, alpha_addr, 1024)

            stats_bar = reg.scalar(u32)
            ptx.inst.add.u32(stats_bar, warp_in_group, 1 + stage * 4)

            ph_s = reg.scalar(b32, init=0)
            ph_stats = reg.scalar(b32, init=0)
            tcount = reg.scalar(u32, init=0)

            bufs = (reg.array(f32, 32), reg.array(f32, 32))
            packed = reg.array(b32, 16)
            macc = [reg.scalar(f32) for _ in range(4)]
            sacc = [reg.scalar(f32) for _ in range(4)]

            def exp_chunk(v):
                for i in range(32):
                    ptx.inst.fma.rn.f32(v[i], v[i], qk_scale_reg, mneg)
                    ptx.inst.ex2.approx.ftz.f32(v[i], v[i])

            def chunk_reduce(v, c):
                # fold this chunk into the running per-thread max/sum accs
                for a in range(4):
                    t = reg.scalar(f32)
                    ptx.inst.max.f32(t, v[a], v[4 + a], v[8 + a])
                    ptx.inst.max.f32(t, t, v[12 + a], v[16 + a])
                    ptx.inst.max.f32(t, t, v[20 + a], v[24 + a])
                    ptx.inst.max.f32(t, t, v[28 + a])
                    if c == 0:
                        ptx.inst.mov.f32(macc[a], t)
                    else:
                        ptx.inst.max.f32(macc[a], macc[a], t)
                    s = reg.scalar(f32)
                    ptx.inst.add.f32(s, v[a], v[4 + a])
                    for i in range(8, 32, 4):
                        ptx.inst.add.f32(s, s, v[i + a])
                    if c == 0:
                        ptx.inst.mov.f32(sacc[a], s)
                    else:
                        ptx.inst.add.f32(sacc[a], sacc[a], s)

            def body(is_first: bool):
                bwait(base + B2_S_FULL + 8 * stage, ph_s, 5, stage)
                ptx.inst.xor.b32(ph_s, ph_s, 1)
                ptx.tcgen05.fence_after_thread_sync()

                # issue the first chunk's load, then run the PREVIOUS
                # tile's deferred basis decision in its latency shadow
                a0 = reg.scalar(b32)
                ptx.inst.mov.b32(a0, s_lo)
                ptx.tcgen05.ld(bufs[0], a0, shape="32x32b", count=32, dtype="b32")

                alpha = reg.scalar(f32)
                if not is_first:
                    ptx.inst.mov.f32(alpha, one_f)
                    move = reg.scalar(pred)
                    ptx.inst.setp.gt.f32(move, pmax_pend, thresh_p)
                    with ptx.if_(move):
                        ptx.inst.rcp.approx.f32(alpha, pmax_pend)
                        lgp = reg.scalar(f32)
                        ptx.inst.lg2.approx.f32(lgp, pmax_pend)
                        ptx.inst.sub.f32(mneg, mneg, lgp)
                    ptx.inst.add.f32(l_run, l_run, sums_pend)
                    ptx.inst.mul.f32(l_run, l_run, alpha)
                    bwait(base + B2_STATS_FREE + 8 * stage, ph_stats, 6, stage)
                    ptx.inst.xor.b32(ph_stats, ph_stats, 1)
                    ptx.inst.st.shared.b32(ptx.addr(alpha_addr), alpha)
                    ptx.mbarrier.arrive(base + B2_STATS_FULL + 8 * stage)

                # ---- stream 4 chunks: ld(c+1) flies under exp/store(c);
                # stale basis, no tile max in front of the exps. Each
                # 64-column HALF publishes to the lead as soon as its two
                # chunks are stored, so the pair-wide PV starts while the
                # second half is still streaming ----
                for c in range(4):
                    ptx.tcgen05.wait_ld()
                    if c < 3:
                        an = reg.scalar(b32)
                        ptx.inst.add.u32(an, s_lo, (c + 1) * 32)
                        ptx.tcgen05.ld(bufs[(c + 1) & 1], an,
                                       shape="32x32b", count=32, dtype="b32")
                    v = bufs[c & 1]
                    exp_chunk(v)
                    for k in range(16):
                        ptx.inst.cvt.rn.bf16x2.f32(packed[k], v[2 * k + 1], v[2 * k])
                    pa = reg.scalar(b32)
                    ptx.inst.add.u32(pa, p_lo, c * 16)
                    ptx.tcgen05.st(pa, packed, shape="32x32b", count=16, dtype="b32")
                    if c == 1 or c == 3:
                        ptx.tcgen05.wait_st()
                        ptx.tcgen05.fence_before_thread_sync()
                        ptx.cluster.arrive_remote(p_half_addr[c // 2], rank0)
                    chunk_reduce(v, c)
                progress(1 + stage, tcount)
                progress(3 + stage, tcount)
                if debug_beacon:
                    ptx.inst.add.u32(tcount, tcount, 1)
                progress(8, tcount)
                progress(9, tcount)

                # horizontal reduce into the pending carries (exp2 is
                # monotonic; pmax > 2^thresh means the basis drifted)
                ptx.inst.max.f32(macc[0], macc[0], macc[1], macc[2])
                ptx.inst.max.f32(pmax_pend, macc[0], macc[3])
                ptx.inst.add.f32(sacc[0], sacc[0], sacc[1])
                ptx.inst.add.f32(sacc[2], sacc[2], sacc[3])
                ptx.inst.add.f32(sums_pend, sacc[0], sacc[2])
                progress(10, tcount)

            zero_f = reg.scalar(f32, init=0.0)
            epi_ph = reg.scalar(b32, init=0)
            sw = reg.scalar(u32); ptx.inst.mov.u32(sw, cluster_x)
            sgo = reg.scalar(pred)
            ptx.inst.setp.lt.u32(sgo, sw, total_work)
            with ptx.loop(f"softmax2_work_loop_{stage}", pred=sgo):
                # gate the item start on the previous epilogue's O readout
                # (softmax TMEM traffic during a readout corrupts rows)
                ptx.mbarrier.wait(base + B2_EPI, epi_ph)
                ptx.inst.xor.b32(epi_ph, epi_ph, 1)
                ptx.inst.mov.f32(mneg, zero_f)
                ptx.inst.mov.f32(l_run, zero_f)
                body(is_first=True)
                it = reg.scalar(u32, init=1)
                go = reg.scalar(pred)
                ptx.inst.setp.lt.u32(go, it, n_tiles)
                with ptx.loop(f"softmax2_loop_{stage}", pred=go):
                    body(is_first=False)
                    it += 1
                    ptx.inst.setp.lt.u32(go, it, n_tiles)

                progress(6)
                # drain: the last tile's basis decision never runs — the
                # pending rescale cancels between O and l in the epilogue
                ptx.inst.add.f32(l_run, l_run, sums_pend)
                ptx.inst.st.shared.b32(ptx.addr(sum_addr), l_run)
                # STATS_FREE-gated like the body publishes so the final
                # arrive can never stack ahead of correction
                bwait(base + B2_STATS_FREE + 8 * stage, ph_stats, 6, stage)
                ptx.inst.xor.b32(ph_stats, ph_stats, 1)
                ptx.mbarrier.arrive(base + B2_STATS_FULL + 8 * stage)

                ptx.inst.add.u32(sw, sw, num_clusters)
                ptx.inst.setp.lt.u32(sgo, sw, total_work)
            progress(7)

        def plumbing_softmax(stage: int):
            pf = reg.scalar(u32, init=0)
            ptx.inst.add.u32(pf, pf, base + B2_P_FULL + 16 * stage)
            pf2 = reg.scalar(u32, init=0)
            ptx.inst.add.u32(pf2, pf2, base + B2_P_FULL + 16 * stage + 8)
            ph_s = reg.scalar(b32, init=0)
            it = reg.scalar(u32, init=0)
            go = reg.scalar(pred)
            ptx.inst.setp.lt.u32(go, it, n_tiles)
            with ptx.loop(f"plumb_sm_{stage}", pred=go):
                if not debug_blind_feed:
                    ptx.mbarrier.wait(base + B2_S_FULL + 8 * stage, ph_s)
                    ptx.inst.xor.b32(ph_s, ph_s, 1)
                ptx.cluster.arrive_remote(pf, rank0)
                ptx.cluster.arrive_remote(pf2, rank0)
                it += 1
                ptx.inst.setp.lt.u32(go, it, n_tiles)

        def paced_sim_softmax(stage: int):
            # real pacing: wait local S_FULL then remote-arrive P_FULL
            pf = reg.scalar(u32, init=0)
            ptx.inst.add.u32(pf, pf, base + B2_P_FULL + 16 * stage)
            pf2 = reg.scalar(u32, init=0)
            ptx.inst.add.u32(pf2, pf2, base + B2_P_FULL + 16 * stage + 8)
            ph_s = reg.scalar(b32, init=0)
            it = reg.scalar(u32, init=0)
            go = reg.scalar(pred)
            ptx.inst.setp.lt.u32(go, it, n_tiles)
            with ptx.loop(f"paced_sm_{stage}", pred=go):
                ptx.mbarrier.wait(base + B2_S_FULL + 8 * stage, ph_s)
                ptx.inst.xor.b32(ph_s, ph_s, 1)
                ptx.cluster.arrive_remote(pf, rank0)
                ptx.cluster.arrive_remote(pf2, rank0)
                it += 1
                ptx.inst.setp.lt.u32(go, it, n_tiles)

        def paced_sim_correction():
            orc = [reg.scalar(u32, init=0) for _ in range(Q_STAGE)]
            ph_s = [reg.scalar(b32, init=0) for _ in range(Q_STAGE)]
            for s in range(Q_STAGE):
                ptx.inst.add.u32(orc[s], orc[s], base + B2_O_RESC + 8 * s)
                ptx.cluster.arrive_remote(orc[s], rank0)
            it = reg.scalar(u32, init=0)
            go = reg.scalar(pred)
            ptx.inst.setp.lt.u32(go, it, n_tiles)
            with ptx.loop("paced_corr", pred=go):
                for s in range(Q_STAGE):
                    ptx.mbarrier.wait(base + B2_S_FULL + 8 * s, ph_s[s])
                    ptx.inst.xor.b32(ph_s[s], ph_s[s], 1)
                    ptx.cluster.arrive_remote(orc[s], rank0)
                it += 1
                ptx.inst.setp.lt.u32(go, it, n_tiles)
            dph = reg.scalar(b32, init=0)
            ptx.mbarrier.wait(base + B2_O_DONE, dph)

        if debug_paced_sim:
            with ptx.if_(is_softmax0):
                ptx.setmaxnreg(NREG_SOFTMAX, inc=True)
                paced_sim_softmax(0)
            with ptx.if_(is_softmax1):
                ptx.setmaxnreg(NREG_SOFTMAX, inc=True)
                paced_sim_softmax(1)
            with ptx.if_(is_correction):
                ptx.setmaxnreg(NREG_CORR, inc=False)
                paced_sim_correction()
        elif debug_plumbing:
            with ptx.if_(is_softmax0):
                ptx.setmaxnreg(NREG_SOFTMAX, inc=True)
                plumbing_softmax(0)
            with ptx.if_(is_softmax1):
                ptx.setmaxnreg(NREG_SOFTMAX, inc=True)
                plumbing_softmax(1)
        else:
            with ptx.if_(is_softmax0):
                ptx.setmaxnreg(NREG_SOFTMAX, inc=True)
                softmax_group(0)
            with ptx.if_(is_softmax1):
                ptx.setmaxnreg(NREG_SOFTMAX, inc=True)
                softmax_group(1)

        # =================================================================
        # Correction warps: O rescale + epilogue (per CTA, O_RESC arrives
        # go to the lead CTA's barrier)
        # =================================================================
        def plumbing_correction():
            ptx.setmaxnreg(NREG_CORR, inc=False)
            oresc = [reg.scalar(u32, init=0) for _ in range(Q_STAGE)]
            for s in range(Q_STAGE):
                ptx.inst.add.u32(oresc[s], oresc[s], base + B2_O_RESC + 8 * s)
            it = reg.scalar(u32, init=0)
            go = reg.scalar(pred)
            ptx.inst.setp.lt.u32(go, it, n_tiles)
            with ptx.loop("plumb_corr", pred=go):
                for s in range(Q_STAGE):
                    ptx.cluster.arrive_remote(oresc[s], rank0)
                it += 1
                ptx.inst.setp.lt.u32(go, it, n_tiles)
            dphase = reg.scalar(b32, init=0)
            ptx.mbarrier.wait(base + B2_O_DONE, dphase)

        if debug_plumbing and not debug_paced_sim:
            with ptx.if_(is_correction):
                plumbing_correction()

        with ptx.if_(is_correction if not (debug_plumbing or debug_paced_sim) else _never):
            ptx.setmaxnreg(NREG_CORR, inc=False)
            one_f = reg.scalar(f32, init=1.0)
            stats_bar = [reg.scalar(u32) for _ in range(Q_STAGE)]
            alpha_addr = [reg.scalar(u32) for _ in range(Q_STAGE)]
            o_addr = [reg.scalar(b32) for _ in range(Q_STAGE)]
            oresc_addr = [reg.scalar(u32, init=0) for _ in range(Q_STAGE)]
            for s in range(Q_STAGE):
                ptx.inst.add.u32(stats_bar[s], warp_in_group, 1 + s * 4)
                ptx.inst.shl.b32(alpha_addr[s], row128, 2)
                ptx.inst.add.u32(alpha_addr[s], alpha_addr[s], base + SMEM2_STATS + s * 512)
                ptx.inst.mov.b32(o_addr[s], tmem)
                ptx.inst.add.u32(o_addr[s], o_addr[s], TM_O[s])
                ptx.inst.add.u32(o_addr[s], o_addr[s], lane_addr_bits)
                ptx.inst.add.u32(oresc_addr[s], oresc_addr[s], base + B2_O_RESC + 8 * s)

            for s in range(Q_STAGE):
                ptx.cluster.arrive_remote(oresc_addr[s], rank0)
                # pre-arrive so the softmax's pre-write STATS_FREE wait at
                # tile 1 is already satisfied (protocol is one tile deep)
                ptx.mbarrier.arrive(base + B2_STATS_FREE + 8 * s)
            ptx.mbarrier.arrive(base + B2_EPI)

            ovals = reg.array(f32, 64)
            opacked = reg.array(b32, 16)
            (po,) = ptx.global_ptrs(O)
            ph_sfull = [reg.scalar(b32, init=0) for _ in range(Q_STAGE)]
            done_phase = reg.scalar(b32, init=0)

            cw = reg.scalar(u32); ptx.inst.mov.u32(cw, cluster_x)
            cgo = reg.scalar(pred)
            ptx.inst.setp.lt.u32(cgo, cw, total_work)
            with ptx.loop("corr2_work_loop", pred=cgo):
                it = reg.scalar(u32, init=1)
                go = reg.scalar(pred)
                ptx.inst.setp.lt.u32(go, it, n_tiles)
                with ptx.loop("corr2_loop", pred=go):
                    for stage in range(Q_STAGE):
                        bwait(base + B2_STATS_FULL + 8 * stage, ph_sfull[stage], 9, stage)
                        ptx.inst.xor.b32(ph_sfull[stage], ph_sfull[stage], 1)
                        alpha = reg.scalar(f32)
                        av = smem.load(b32, ptx.addr(alpha_addr[stage]))
                        ptx.inst.mov.b32(alpha, av)
                        ptx.mbarrier.arrive(base + B2_STATS_FREE + 8 * stage)
                        need = reg.scalar(pred)
                        ptx.inst.setp.lt.f32(need, alpha, one_f)
                        ballot = reg.scalar(b32)
                        ptx.inst.vote.sync.ballot.b32(ballot, need, 0xFFFFFFFF)
                        any_need = reg.scalar(pred)
                        ptx.inst.setp.ne.b32(any_need, ballot, 0)
                        with ptx.if_(any_need):
                            ptx.tcgen05.fence_after_thread_sync()
                            for half in range(2):
                                addr = reg.scalar(b32)
                                ptx.inst.add.u32(addr, o_addr[stage], half * 64)
                                ptx.tcgen05.ld(ovals, addr, shape="32x32b", count=64, dtype="b32")
                                ptx.tcgen05.wait_ld()
                                for i in range(64):
                                    ptx.inst.mul.f32(ovals[i], ovals[i], alpha)
                                ptx.tcgen05.st(addr, ovals, shape="32x32b", count=64, dtype="b32")
                            ptx.tcgen05.wait_st()
                            ptx.tcgen05.fence_before_thread_sync()
                        ptx.cluster.arrive_remote(oresc_addr[stage], rank0)
                    it += 1
                    ptx.inst.setp.lt.u32(go, it, n_tiles)

                bwait(base + B2_O_DONE, done_phase, 7)
                ptx.inst.xor.b32(done_phase, done_phase, 1)
                ptx.tcgen05.fence_after_thread_sync()

                q_row0, _ = derive_rows2(cw)
                out_row = reg.scalar(u32)
                ptx.inst.add.u32(out_row, q_row0, row128)

                for stage in range(Q_STAGE):
                    bwait(base + B2_STATS_FULL + 8 * stage, ph_sfull[stage], 10, stage)
                    ptx.inst.xor.b32(ph_sfull[stage], ph_sfull[stage], 1)
                    l = reg.scalar(f32)
                    lv = smem.load(b32, ptx.addr(alpha_addr[stage] + 1024))
                    ptx.inst.mov.b32(l, lv)
                    # return the final publish's credit (next item's
                    # standing STATS_FREE credit)
                    ptx.mbarrier.arrive(base + B2_STATS_FREE + 8 * stage)
                    inv_l = reg.scalar(f32)
                    ptx.inst.rcp.approx.f32(inv_l, l)

                    row = reg.scalar(u32)
                    ptx.inst.add.u32(row, out_row, stage * 2 * BM)
                    byte_off = reg.scalar(u64)
                    ptx.inst.mul.wide.u32(byte_off, row, HD * 2)
                    gptr = po + byte_off

                    for quarter in range(4):
                        addr = reg.scalar(b32)
                        ptx.inst.add.u32(addr, o_addr[stage], quarter * 32)
                        part = [ovals[i] for i in range(32)]
                        ptx.tcgen05.ld(part, addr, shape="32x32b", count=32, dtype="b32")
                        ptx.tcgen05.wait_ld()
                        for i in range(32):
                            ptx.inst.mul.f32(ovals[i], ovals[i], inv_l)
                        for c in range(16):
                            ptx.inst.cvt.rn.bf16x2.f32(
                                opacked[c], ovals[2 * c + 1], ovals[2 * c]
                            )
                        if not debug_beacon:
                            for vec in range(4):
                                ptx.inst.st.global_.v4.b32(
                                    ptx.addr(gptr, quarter * 64 + vec * 16),
                                    [opacked[vec * 4 + k] for k in range(4)],
                                )

                # O read out: release the softmax item-start gate and
                # re-arm the next item's first PV gate
                ptx.mbarrier.arrive(base + B2_EPI)
                for s in range(Q_STAGE):
                    ptx.cluster.arrive_remote(oresc_addr[s], rank0)

                ptx.inst.add.u32(cw, cw, num_clusters)
                ptx.inst.setp.lt.u32(cgo, cw, total_work)

        ptx.bar.sync(0, 512)
        ptx.cluster.sync()
        if waitmap:
            _wm_ctx.__exit__(None, None, None)
            is_cta0 = reg.scalar(pred)
            with ptx.scope():
                cx = reg.scalar(u32)
                ptx.inst.mov.u32(cx, ptx.special.ctaid.x())
                ptx.inst.setp.eq.u32(is_cta0, cx, 0)
            wm.flush(pred=is_cta0)
        with ptx.if_(alloc_warp):
            ptx.tcgen05.dealloc(tmem, 512, cta_group=2)
            ptx.tcgen05.relinquish_alloc_permit(cta_group=2)
        ptx.ret()

    if waitmap:
        @kernel(**kernel_kwargs)
        def flash_attn_fwd_2cta(Q, K, V, O, WM):
            _fa2_body(Q, K, V, O, WM)
    else:
        @kernel(**kernel_kwargs)
        def flash_attn_fwd_2cta(Q, K, V, O):
            _fa2_body(Q, K, V, O, None)

    flash_attn_fwd_2cta._waitmap = wm
    return flash_attn_fwd_2cta




# =====================================================================
# Occupancy-2 variant: two independent 384-thread CTAs per SM, each
# driving a single 128-row Q tile with 256 TMEM columns. While one
# CTA's softmax runs, the other CTA's QK/PV keeps the tensor core hot —
# cross-CTA interleaving hides the softmax latency that caps the
# single-CTA pipelines.
# =====================================================================

O2_SMEM_Q = 0                                # 32 KB
O2_SMEM_H = 32768                            # 4 x 16 KB half-tile slots:
#   slot 0: K rows 0-63, slot 1: K rows 64-127,
#   slot 2: V rows 0-63, slot 3: V rows 64-127
O2_HALF_BYTES = 16384
O2_SMEM_STATS = 98304                        # alpha[128], sum[128]
O2_BARS = O2_SMEM_STATS + 1024
O2B_Q = O2_BARS + 0
O2B_H_FULL = O2_BARS + 8        # 4 x 8
O2B_H_FREE = O2_BARS + 40       # 4 x 8
O2B_S_FULL = O2_BARS + 72
O2B_P_FULL = O2_BARS + 80       # count 128
O2B_O_RESC = O2_BARS + 88       # count 128
O2B_O_DONE = O2_BARS + 96
O2B_STATS_FREE = O2_BARS + 104  # count 128
O2B_STATS_FULL = O2_BARS + 112  # count 128
O2_TMEM_SLOT = O2_BARS + 128
O2_SMEM_BYTES = O2_TMEM_SLOT + 16

O2_TM_S = 0     # S/P: cols 0-127 (P packed bf16 in cols 0-63)
O2_TM_O = 128   # O accumulator: cols 128-255

O2_MMA_TID = 256   # warp 8 lane 0
O2_LOAD_TID = 288  # warp 9 lane 0


def build_flash_attention_blackwell_occ2(
    seqlen: int,
    batch_heads: int,
    head_dim: int = 128,
    *,
    sm_scale: float | None = None,
    arch: str = "sm_100a",
    rescale_threshold: float = RESCALE_THRESHOLD,
):
    assert head_dim == HD
    assert seqlen % BM == 0, f"seqlen must be a multiple of {BM}"
    n_tiles = seqlen // BN
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * LOG2E
    total_rows = batch_heads * seqlen

    @kernel(
        in_specs=(
            Tile(total_rows, HD, bf16, Layout.TMA_128B, tma_box=(BM, 64)),
            Tile(total_rows, HD, bf16, Layout.TMA_128B, tma_box=(64, 64)),
            Tile(total_rows, HD, bf16, Layout.TMA_128B, tma_box=(64, 64)),
        ),
        out_specs=(Tile(total_rows, HD, bf16, Layout.ROW),),
        grid=(seqlen // BM, batch_heads, 1),
        block=(384, 1, 1),
        arch=arch,
        smem=O2_SMEM_BYTES,
        extern_smem=True,
        raw_directives=[
            ("reqntid", (384, 1, 1)),
            ("maxnreg", (80,)),
        ],
    )
    def flash_attn_fwd_occ2(Q, K, V, O):
        base = smem.base()
        tid = reg.scalar(u32); ptx.inst.mov.u32(tid, ptx.special.tid.x())
        m_block = reg.scalar(u32); ptx.inst.mov.u32(m_block, ptx.special.ctaid.x())
        bh = reg.scalar(u32); ptx.inst.mov.u32(bh, ptx.special.ctaid.y())

        warp = tid >> 5
        is_softmax = reg.scalar(pred); ptx.inst.setp.lt.u32(is_softmax, tid, 128)
        is_correction = reg.scalar(pred)
        with ptx.scope():
            ge = reg.scalar(pred); lt = reg.scalar(pred)
            ptx.inst.setp.ge.u32(ge, tid, 128)
            ptx.inst.setp.lt.u32(lt, tid, 256)
            ptx.inst.and_.pred(is_correction, ge, lt)
        is_mma = reg.scalar(pred); ptx.inst.setp.eq.u32(is_mma, tid, O2_MMA_TID)
        is_load = reg.scalar(pred); ptx.inst.setp.eq.u32(is_load, tid, O2_LOAD_TID)
        alloc_warp = reg.scalar(pred)
        ptx.inst.setp.eq.u32(alloc_warp, warp, 8)

        def make_q_row0():
            r = reg.scalar(u32)
            ptx.inst.mul.lo.u32(r, bh, seqlen)
            with ptx.scope():
                mb_rows = reg.scalar(u32)
                ptx.inst.mul.lo.u32(mb_rows, m_block, BM)
                ptx.inst.add.u32(r, r, mb_rows)
            return r

        def make_row128():
            r = reg.scalar(u32)
            ptx.inst.and_.b32(r, tid, 127)
            return r

        def make_lane_bits():
            r = reg.scalar(u32)
            ptx.inst.and_.b32(r, tid, 96)
            ptx.inst.shl.b32(r, r, 16)
            return r

        with ptx.if_(tid == 0):
            ptx.mbarrier.init(base + O2B_Q, 1)
            for h in range(4):
                ptx.mbarrier.init(base + O2B_H_FULL + 8 * h, 1)
                ptx.mbarrier.init(base + O2B_H_FREE + 8 * h, 1)
            ptx.mbarrier.init(base + O2B_S_FULL, 1)
            ptx.mbarrier.init(base + O2B_P_FULL, 128)
            ptx.mbarrier.init(base + O2B_O_RESC, 128)
            ptx.mbarrier.init(base + O2B_O_DONE, 1)
            ptx.mbarrier.init(base + O2B_STATS_FREE, 128)
            ptx.mbarrier.init(base + O2B_STATS_FULL, 128)
            ptx.fence.proxy_async_shared_cta()
        with ptx.if_(alloc_warp):
            ptx.tcgen05.alloc(base + O2_TMEM_SLOT, 256)
            # release the SM-wide allocation permit immediately so the
            # co-resident CTA can allocate its own 256 columns
            ptx.tcgen05.relinquish_alloc_permit()
        ptx.bar.sync(0, 384)
        tmem = smem.load(b32, ptx.addr(base + O2_TMEM_SLOT))

        # =============================================================
        # TMA load warp
        # =============================================================
        with ptx.if_(is_load):
            def load_tile(dst_off, tensor, row_reg, mbar_off):
                for stripe in range(2):
                    ptx.cp.async_.bulk.tensor_2d(
                        dst=base + dst_off + stripe * STRIPE_BYTES,
                        src=tensor.tma_desc(),
                        coord=(stripe * 64, row_reg),
                        mbar=base + mbar_off,
                    )

            q_row_ld = make_q_row0()
            ptx.mbarrier.arrive_expect_tx(base + O2B_Q, TILE_BYTES)
            load_tile(O2_SMEM_Q, Q, q_row_ld, O2B_Q)

            krow = reg.scalar(u32)
            ptx.inst.mul.lo.u32(krow, bh, seqlen)
            vrow = reg.scalar(u32); ptx.inst.mov.u32(vrow, krow)

            def load_half(slot, tensor, row_reg, row_off):
                # one 64-row x 128-col half: two 8 KB stripe loads
                ptx.mbarrier.arrive_expect_tx(
                    base + O2B_H_FULL + 8 * slot, O2_HALF_BYTES
                )
                r = reg.scalar(u32)
                ptx.inst.add.u32(r, row_reg, row_off)
                for stripe in range(2):
                    ptx.cp.async_.bulk.tensor_2d(
                        dst=base + O2_SMEM_H + slot * O2_HALF_BYTES + stripe * 8192,
                        src=tensor.tma_desc(),
                        coord=(stripe * 64, r),
                        mbar=base + O2B_H_FULL + 8 * slot,
                    )

            def load_tile_halves():
                load_half(0, K, krow, 0)
                load_half(1, K, krow, 64)
                load_half(2, V, vrow, 0)
                load_half(3, V, vrow, 64)
                ptx.inst.add.u32(krow, krow, BN)
                ptx.inst.add.u32(vrow, vrow, BN)

            load_tile_halves()
            if n_tiles > 1:
                ph = [reg.scalar(b32, init=0) for _ in range(4)]
                it = reg.scalar(u32, init=1)
                go = reg.scalar(pred)
                ptx.inst.setp.lt.u32(go, it, n_tiles)
                with ptx.loop("o2_load_loop", pred=go):
                    for h, (tensor, row_reg, row_off) in enumerate(
                        ((K, krow, 0), (K, krow, 64), (V, vrow, 0), (V, vrow, 64))
                    ):
                        ptx.mbarrier.wait(base + O2B_H_FREE + 8 * h, ph[h])
                        ptx.inst.xor.b32(ph[h], ph[h], 1)
                        load_half(h, tensor, row_reg, row_off)
                    ptx.inst.add.u32(krow, krow, BN)
                    ptx.inst.add.u32(vrow, vrow, BN)
                    it += 1
                    ptx.inst.setp.lt.u32(go, it, n_tiles)

        # =============================================================
        # MMA dispatch warp
        # =============================================================
        with ptx.if_(is_mma):
            idesc_qk = reg.scalar(
                b32,
                init=ptx.tcgen05.make_instr_desc_f16bf16_f32(
                    m=128, n=64, ab_dtype="bf16", a_major="K", b_major="K"
                ),
            )
            idesc_pv = reg.scalar(
                b32,
                init=ptx.tcgen05.make_instr_desc_f16bf16_f32(
                    m=128, n=HD, ab_dtype="bf16", a_major="K", b_major="MN"
                ),
            )
            p_acc = reg.scalar(pred)
            p_noacc = reg.scalar(pred)
            with ptx.scope():
                one = reg.scalar(u32, init=1)
                zero = reg.scalar(u32, init=0)
                ptx.inst.setp.ne.b32(p_acc, one, 0)
                ptx.inst.setp.ne.b32(p_noacc, zero, 0)

            dq0 = ptx.tcgen05.masked_descriptor(base + O2_SMEM_Q)
            dh = [
                ptx.tcgen05.masked_descriptor(base + O2_SMEM_H + h * O2_HALF_BYTES)
                for h in range(2)
            ]
            dv = [
                ptx.tcgen05.descriptor(
                    base + O2_SMEM_H + (2 + h) * O2_HALF_BYTES,
                    stride_bytes=1024, leading_bytes=8192, swizzle="128B",
                )
                for h in range(2)
            ]

            ph_h = [reg.scalar(b32, init=0) for _ in range(4)]
            ph_p = reg.scalar(b32, init=0)
            ph_or = reg.scalar(b32, init=0)
            ph_q = reg.scalar(b32, init=0)

            def wait_h(h):
                ptx.mbarrier.wait(base + O2B_H_FULL + 8 * h, ph_h[h])
                ptx.inst.xor.b32(ph_h[h], ph_h[h], 1)

            def khalf_off(kk):
                # K half: 64 rows x 128 cols as two 8 KB stripes
                return (kk // 4) * 512 + (kk % 4) * 2

            def qk_mma_half(h):
                d = tmem + (O2_TM_S + 64 * h)
                for kk in range(8):
                    offa = _kmajor_desc_off(kk)
                    offb = khalf_off(kk)
                    da = dq0 if offa == 0 else reg.scalar(b64)
                    if offa:
                        ptx.inst.add.s64(da, dq0, offa)
                    db = dh[h] if offb == 0 else reg.scalar(b64)
                    if offb:
                        ptx.inst.add.s64(db, dh[h], offb)
                    ptx.tcgen05.mma(
                        d, da, db, idesc_qk, kind="f16",
                        pred_operand=(p_noacc if kk == 0 else p_acc),
                    )
                ptx.tcgen05.commit(base + O2B_H_FREE + 8 * h)

            def pv_mma_half(h, first_tile):
                d = tmem + O2_TM_O
                for kk in range(4 * h, 4 * h + 4):
                    a = tmem + (O2_TM_S + kk * 8)
                    db = dv[h] if kk % 4 == 0 else reg.scalar(b64)
                    if kk % 4:
                        ptx.inst.add.s64(db, dv[h], (kk % 4) * 128)
                    ptx.tcgen05.mma(
                        d, a, db, idesc_pv, kind="f16", a_is_tmem=True,
                        pred_operand=(p_noacc if (first_tile and kk == 0) else p_acc),
                    )
                ptx.tcgen05.commit(base + O2B_H_FREE + 8 * (2 + h))

            def tile_mma(first_tile):
                wait_h(0)
                qk_mma_half(0)
                wait_h(1)
                qk_mma_half(1)
                ptx.tcgen05.commit(base + O2B_S_FULL)
                ptx.mbarrier.wait(base + O2B_P_FULL, ph_p)
                ptx.inst.xor.b32(ph_p, ph_p, 1)
                ptx.mbarrier.wait(base + O2B_O_RESC, ph_or)
                ptx.inst.xor.b32(ph_or, ph_or, 1)
                ptx.tcgen05.fence_after_thread_sync()
                wait_h(2)
                pv_mma_half(0, first_tile)
                wait_h(3)
                pv_mma_half(1, first_tile)

            ptx.mbarrier.wait(base + O2B_Q, ph_q)
            tile_mma(True)
            if n_tiles > 1:
                it = reg.scalar(u32, init=1)
                go = reg.scalar(pred)
                ptx.inst.setp.lt.u32(go, it, n_tiles)
                with ptx.loop("o2_mma_loop", pred=go):
                    tile_mma(False)
                    it += 1
                    ptx.inst.setp.lt.u32(go, it, n_tiles)
            ptx.tcgen05.commit(base + O2B_O_DONE)

        # =============================================================
        # Softmax warps (tids 0-127, one full row each; 112 regs, so the
        # row is processed in 64-column halves with a reload pass)
        # =============================================================
        with ptx.if_(is_softmax):
            row128 = make_row128()
            lane_addr_bits = make_lane_bits()
            s_lo = reg.scalar(b32)
            ptx.inst.mov.b32(s_lo, tmem)
            ptx.inst.add.u32(s_lo, s_lo, lane_addr_bits)
            p_lo = reg.scalar(b32)
            ptx.inst.mov.b32(p_lo, s_lo)

            qk_scale_reg = reg.scalar(f32, init=qk_scale)
            neg_thresh = reg.scalar(f32, init=-rescale_threshold)
            one_f = reg.scalar(f32, init=1.0)
            m_run = reg.scalar(f32, init=-1e30)
            l_run = reg.scalar(f32, init=0.0)

            alpha_addr = reg.scalar(u32)
            ptx.inst.shl.b32(alpha_addr, row128, 2)
            ptx.inst.add.u32(alpha_addr, alpha_addr, base + O2_SMEM_STATS)
            sum_addr = reg.scalar(u32)
            ptx.inst.add.u32(sum_addr, alpha_addr, 512)

            ph_s = reg.scalar(b32, init=0)
            ph_stats = reg.scalar(b32, init=0)

            vals = reg.array(f32, 32)
            packed = reg.array(b32, 16)

            def ld_quarter(q):
                addr = reg.scalar(b32)
                ptx.inst.add.u32(addr, s_lo, q * 32)
                ptx.tcgen05.ld(vals, addr, shape="32x32b", count=32, dtype="b32")
                ptx.tcgen05.wait_ld()

            def q_max(dst, first):
                acc = [reg.scalar(f32) for _ in range(4)]
                for a in range(4):
                    ptx.inst.max.f32(acc[a], vals[a], vals[4 + a], vals[8 + a])
                for a in range(4):
                    ptx.inst.max.f32(acc[a], acc[a], vals[12 + a], vals[16 + a])
                for a in range(4):
                    ptx.inst.max.f32(acc[a], acc[a], vals[20 + a], vals[24 + a])
                for a in range(4):
                    ptx.inst.max.f32(acc[a], acc[a], vals[28 + a])
                ptx.inst.max.f32(acc[0], acc[0], acc[1], acc[2])
                if first:
                    ptx.inst.max.f32(dst, acc[0], acc[3])
                else:
                    ptx.inst.max.f32(dst, dst, acc[0], acc[3])

            def body(is_first: bool):
                ptx.mbarrier.wait(base + O2B_S_FULL, ph_s)
                ptx.inst.xor.b32(ph_s, ph_s, 1)
                ptx.tcgen05.fence_after_thread_sync()

                # max pass over four 32-column quarters
                pmax = reg.scalar(f32)
                for q in range(4):
                    ld_quarter(q)
                    q_max(pmax, q == 0)

                alpha = reg.scalar(f32)
                if not is_first:
                    m_new = reg.scalar(f32)
                    ptx.inst.max.f32(m_new, m_run, pmax)
                    d = reg.scalar(f32)
                    ptx.inst.sub.f32(d, m_run, m_new)
                    ptx.inst.mul.f32(d, d, qk_scale_reg)
                    keep = reg.scalar(pred)
                    ptx.inst.setp.ge.f32(keep, d, neg_thresh)
                    with ptx.if_(keep):
                        ptx.inst.mov.f32(alpha, one_f)
                    with ptx.else_():
                        ptx.inst.ex2.approx.ftz.f32(alpha, d)
                        ptx.inst.mov.f32(m_run, m_new)
                    ptx.mbarrier.wait(base + O2B_STATS_FREE, ph_stats)
                    ptx.inst.xor.b32(ph_stats, ph_stats, 1)
                    ptx.inst.st.shared.b32(ptx.addr(alpha_addr), alpha)
                    ptx.mbarrier.arrive(base + O2B_STATS_FULL)
                else:
                    ptx.inst.mov.f32(m_run, pmax)

                m_scaled = reg.scalar(f32)
                ptx.inst.mul.f32(m_scaled, m_run, qk_scale_reg)
                ptx.inst.neg.f32(m_scaled, m_scaled)

                # exp pass, quarter q's P words land on S columns already
                # consumed by quarter q (sequential overwrite is safe)
                lsum = reg.scalar(f32, init=0.0)
                for q in range(4):
                    ld_quarter(q)
                    for i in range(32):
                        ptx.inst.fma.rn.f32(vals[i], vals[i], qk_scale_reg, m_scaled)
                        ptx.inst.ex2.approx.ftz.f32(vals[i], vals[i])
                    for c in range(16):
                        ptx.inst.cvt.rn.bf16x2.f32(
                            packed[c], vals[2 * c + 1], vals[2 * c]
                        )
                    pa = reg.scalar(b32)
                    ptx.inst.add.u32(pa, p_lo, q * 16)
                    ptx.tcgen05.st(pa, packed, shape="32x32b", count=16, dtype="b32")
                    acc = [reg.scalar(f32) for _ in range(4)]
                    for a in range(4):
                        ptx.inst.add.f32(acc[a], vals[a], vals[4 + a])
                    for i in range(8, 32, 4):
                        for a in range(4):
                            ptx.inst.add.f32(acc[a], acc[a], vals[i + a])
                    ptx.inst.add.f32(acc[0], acc[0], acc[1])
                    ptx.inst.add.f32(acc[2], acc[2], acc[3])
                    ptx.inst.add.f32(acc[0], acc[0], acc[2])
                    ptx.inst.add.f32(lsum, lsum, acc[0])
                ptx.tcgen05.wait_st()
                ptx.tcgen05.fence_before_thread_sync()
                ptx.mbarrier.arrive(base + O2B_P_FULL)

                if is_first:
                    ptx.inst.mov.f32(l_run, lsum)
                else:
                    ptx.inst.fma.rn.f32(l_run, l_run, alpha, lsum)

            body(is_first=True)
            if n_tiles > 1:
                it = reg.scalar(u32, init=1)
                go = reg.scalar(pred)
                ptx.inst.setp.lt.u32(go, it, n_tiles)
                with ptx.loop("o2_softmax_loop", pred=go):
                    body(is_first=False)
                    it += 1
                    ptx.inst.setp.lt.u32(go, it, n_tiles)

            ptx.inst.st.shared.b32(ptx.addr(sum_addr), l_run)
            ptx.mbarrier.arrive(base + O2B_STATS_FULL)

        # =============================================================
        # Correction warps (tids 128-255): O rescale + epilogue
        # =============================================================
        with ptx.if_(is_correction):
            row128 = make_row128()
            lane_addr_bits = make_lane_bits()
            q_row0 = make_q_row0()
            one_f = reg.scalar(f32, init=1.0)
            alpha_addr = reg.scalar(u32)
            ptx.inst.shl.b32(alpha_addr, row128, 2)
            ptx.inst.add.u32(alpha_addr, alpha_addr, base + O2_SMEM_STATS)
            o_addr = reg.scalar(b32)
            ptx.inst.mov.b32(o_addr, tmem)
            ptx.inst.add.u32(o_addr, o_addr, O2_TM_O)
            ptx.inst.add.u32(o_addr, o_addr, lane_addr_bits)

            ptx.mbarrier.arrive(base + O2B_O_RESC)
            ptx.mbarrier.arrive(base + O2B_STATS_FREE)

            ovals = reg.array(f32, 32)

            if n_tiles > 1:
                ph_sf = reg.scalar(b32, init=0)
                it = reg.scalar(u32, init=1)
                go = reg.scalar(pred)
                ptx.inst.setp.lt.u32(go, it, n_tiles)
                with ptx.loop("o2_corr_loop", pred=go):
                    ptx.mbarrier.wait(base + O2B_STATS_FULL, ph_sf)
                    ptx.inst.xor.b32(ph_sf, ph_sf, 1)
                    alpha = reg.scalar(f32)
                    av = smem.load(b32, ptx.addr(alpha_addr))
                    ptx.inst.mov.b32(alpha, av)
                    ptx.mbarrier.arrive(base + O2B_STATS_FREE)
                    need = reg.scalar(pred)
                    ptx.inst.setp.lt.f32(need, alpha, one_f)
                    ballot = reg.scalar(b32)
                    ptx.inst.vote.sync.ballot.b32(ballot, need, 0xFFFFFFFF)
                    any_need = reg.scalar(pred)
                    ptx.inst.setp.ne.b32(any_need, ballot, 0)
                    with ptx.if_(any_need):
                        ptx.tcgen05.fence_after_thread_sync()
                        for quarter in range(4):
                            addr = reg.scalar(b32)
                            ptx.inst.add.u32(addr, o_addr, quarter * 32)
                            ptx.tcgen05.ld(ovals, addr, shape="32x32b", count=32, dtype="b32")
                            ptx.tcgen05.wait_ld()
                            for i in range(32):
                                ptx.inst.mul.f32(ovals[i], ovals[i], alpha)
                            ptx.tcgen05.st(addr, ovals, shape="32x32b", count=32, dtype="b32")
                        ptx.tcgen05.wait_st()
                        ptx.tcgen05.fence_before_thread_sync()
                    ptx.mbarrier.arrive(base + O2B_O_RESC)
                    it += 1
                    ptx.inst.setp.lt.u32(go, it, n_tiles)

            done_phase = reg.scalar(b32, init=0)
            ptx.mbarrier.wait(base + O2B_O_DONE, done_phase)
            ptx.tcgen05.fence_after_thread_sync()

            # final stats
            ptx.mbarrier.wait(base + O2B_STATS_FULL, (n_tiles - 1) & 1)
            (po,) = ptx.global_ptrs(O)
            out_row = reg.scalar(u32)
            ptx.inst.add.u32(out_row, q_row0, row128)
            l = reg.scalar(f32)
            lv = smem.load(b32, ptx.addr(alpha_addr + 512))
            ptx.inst.mov.b32(l, lv)
            inv_l = reg.scalar(f32)
            ptx.inst.rcp.approx.f32(inv_l, l)
            byte_off = reg.scalar(u64)
            ptx.inst.mul.wide.u32(byte_off, out_row, HD * 2)
            gptr = po + byte_off
            opacked = reg.array(b32, 16)
            for quarter in range(4):
                addr = reg.scalar(b32)
                ptx.inst.add.u32(addr, o_addr, quarter * 32)
                ptx.tcgen05.ld(ovals, addr, shape="32x32b", count=32, dtype="b32")
                ptx.tcgen05.wait_ld()
                for i in range(32):
                    ptx.inst.mul.f32(ovals[i], ovals[i], inv_l)
                for c in range(16):
                    ptx.inst.cvt.rn.bf16x2.f32(
                        opacked[c], ovals[2 * c + 1], ovals[2 * c]
                    )
                for vec in range(4):
                    ptx.inst.st.global_.v4.b32(
                        ptx.addr(gptr, quarter * 64 + vec * 16),
                        [opacked[vec * 4 + k] for k in range(4)],
                    )

        ptx.bar.sync(0, 384)
        with ptx.if_(alloc_warp):
            ptx.tcgen05.dealloc(tmem, 256)
        ptx.ret()

    return flash_attn_fwd_occ2


# =====================================================================
# test / benchmark
# =====================================================================

def _test_case(B: int, H: int, S: int, D: int = 128) -> bool:
    import torch

    k_fn = build_flash_attention_blackwell(S, B * H, D)
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    out = k_fn(q.reshape(-1, D), k.reshape(-1, D), v.reshape(-1, D))
    out = out.reshape(B, H, S, D)
    torch.cuda.synchronize()
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float()
    )
    diff = (out.float() - ref).abs()
    rel = diff.max() / ref.abs().max()
    ok = bool(diff.max() < 0.05)
    print(
        f"[{'OK  ' if ok else 'FAIL'}] B={B} H={H} S={S} D={D} "
        f"max_abs={diff.max():.4e} mean_abs={diff.mean():.4e} rel={rel:.3e}"
    )
    return ok


def main() -> None:
    ok = True
    for B, H, S in ((1, 1, 256), (1, 2, 512), (2, 4, 1024), (1, 8, 4096)):
        ok &= _test_case(B, H, S)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
