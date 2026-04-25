"""Blackwell tcgen05 LD phase probe.

This isolates the TMEM load walk used by the current no-TMA GEMM epilogue.
It emits one 128x256x64 GEMM tile with row-coded inputs and then reads the
accumulator back in two ways:

- current path: base address + 64-column offset
- shift path: repeated tcgen05.ld with tcgen05.shift between loads

The output is intentionally diagnostic rather than polished.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pyptx import Tile, kernel, ptx, reg, smem
from pyptx.smem import apply_swizzle
from pyptx.specs import Layout
from pyptx.types import b32, b64, bf16, f32, pred, u32, u64


BM = 128
BN = 256
BK = 64
A_BYTES = BM * BK * 2
B_BYTES = BN * BK * 2
MMA_BAR_OFF = A_BYTES + B_BYTES
TMEM_SLOT_OFF = MMA_BAR_OFF + 16
SMEM_BYTES = TMEM_SLOT_OFF + 16


def kmajor_swizzle(row_stride_elems: int) -> str:
    row_bytes = row_stride_elems * 2
    if row_bytes >= 128:
        return "128B"
    if row_bytes >= 64:
        return "64B"
    if row_bytes >= 32:
        return "32B"
    raise ValueError(f"unsupported Blackwell K-major row width: {row_stride_elems} elems")


def kmajor_swizzled_logical_bytes(row, k_elem, row_stride_elems):
    contig_elems = {"32B": 16, "64B": 32, "128B": 64}[kmajor_swizzle(row_stride_elems)]
    row_group = row >> 3
    row_in_group = row & 7
    return ((row_group * (contig_elems * 8)) + (row_in_group * contig_elems) + k_elem) << 1


def build(*, use_shift: bool, arch: str = "sm_100a"):
    operand_swizzle = kmajor_swizzle(BK)

    @kernel(
        in_specs=(
            Tile(BM, BK, bf16, Layout.ROW),
            Tile(BN, BK, bf16, Layout.ROW),
        ),
        out_specs=(Tile(BM, BN, f32, Layout.ROW),),
        grid=(1, 1, 1),
        block=(128, 1, 1),
        arch=arch,
        smem=SMEM_BYTES,
        extern_smem=True,
    )
    def k(A, B_T, D):
        base = smem.base()
        mma_bar = base + MMA_BAR_OFF
        tmem_slot = base + TMEM_SLOT_OFF

        tid = reg.scalar(u32)
        lane = reg.scalar(u32)
        warp = reg.scalar(u32)
        alloc_warp = reg.scalar(pred)
        epilogue_thread = reg.scalar(pred)
        ready = reg.scalar(pred)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        ptx.inst.and_.b32(lane, tid, 31)
        ptx.inst.shr.u32(warp, tid, 5)
        ptx.inst.setp.lt.u32(alloc_warp, tid, 32)
        ptx.inst.setp.lt.u32(epilogue_thread, tid, 128)

        with ptx.if_(tid == 0):
            ptx.mbarrier.init(mma_bar, 1)
        with ptx.if_(alloc_warp):
            ptx.tcgen05.alloc(tmem_slot, 512)
        ptx.bar.sync(0)

        tmem_base = smem.load(b32, ptx.addr(tmem_slot))
        idesc = reg.scalar(b32, init=ptx.tcgen05.make_instr_desc_f16bf16_f32())

        pa, pb, pd = ptx.global_ptrs(A, B_T, D)

        a_words = A_BYTES // 4
        a_idx = reg.scalar(u32)
        ptx.inst.mov.u32(a_idx, tid)
        a_keep = reg.scalar(pred)
        ptx.inst.setp.lt.u32(a_keep, a_idx, a_words)
        with ptx.loop("copy_a_loop", pred=a_keep):
            word_index = reg.scalar(u64)
            ptx.inst.cvt.u64.u32(word_index, a_idx)
            g_ptr = pa + (word_index << 2)
            word = reg.scalar(b32)
            ptx.inst.ld.global_.b32(word, ptx.addr(g_ptr))
            row = reg.scalar(u32)
            k_word = reg.scalar(u32)
            logical = reg.scalar(u32)
            ptx.inst.shr.u32(row, a_idx, 5)
            ptx.inst.and_.b32(k_word, a_idx, (BK // 2) - 1)
            logical = kmajor_swizzled_logical_bytes(row, k_word << 1, BK)
            physical = apply_swizzle(logical, operand_swizzle)
            ptx.inst.st.shared.b32(ptx.addr(base + physical), word)
            a_idx += 128
            ptx.inst.setp.lt.u32(a_keep, a_idx, a_words)

        b_words = B_BYTES // 4
        b_idx = reg.scalar(u32)
        ptx.inst.mov.u32(b_idx, tid)
        b_keep = reg.scalar(pred)
        ptx.inst.setp.lt.u32(b_keep, b_idx, b_words)
        with ptx.loop("copy_b_loop", pred=b_keep):
            word_index = reg.scalar(u64)
            ptx.inst.cvt.u64.u32(word_index, b_idx)
            g_ptr = pb + (word_index << 2)
            word = reg.scalar(b32)
            ptx.inst.ld.global_.b32(word, ptx.addr(g_ptr))
            row = reg.scalar(u32)
            k_word = reg.scalar(u32)
            logical = reg.scalar(u32)
            ptx.inst.shr.u32(row, b_idx, 5)
            ptx.inst.and_.b32(k_word, b_idx, (BK // 2) - 1)
            logical = kmajor_swizzled_logical_bytes(row, k_word << 1, BK)
            physical = apply_swizzle(logical, operand_swizzle)
            ptx.inst.st.shared.b32(ptx.addr(base + A_BYTES + physical), word)
            b_idx += 128
            ptx.inst.setp.lt.u32(b_keep, b_idx, b_words)

        ptx.bar.sync(0)

        desc_a0 = ptx.tcgen05.descriptor(
            base,
            stride_bytes=BK * 16,
            leading_bytes=16,
            swizzle=operand_swizzle,
        )
        desc_b0 = ptx.tcgen05.descriptor(
            base,
            byte_offset=A_BYTES,
            stride_bytes=BK * 16,
            leading_bytes=16,
            swizzle=operand_swizzle,
        )

        for phase in range(2):
            for kk in range(phase * 2, phase * 2 + 2):
                if kk == 0:
                    desc_a = desc_a0
                    desc_b = desc_b0
                else:
                    desc_a = reg.scalar(b64)
                    desc_b = reg.scalar(b64)
                    ptx.inst.add.s64(desc_a, desc_a0, kk * 2)
                    ptx.inst.add.s64(desc_b, desc_b0, kk * 2)
                with ptx.if_(tid == 0):
                    ptx.tcgen05.mma(
                        tmem_base,
                        desc_a,
                        desc_b,
                        idesc,
                        kind="f16",
                        pred_operand=(kk != 0),
                    )
            with ptx.if_(tid == 0):
                ptx.tcgen05.commit(mma_bar, space="cluster")
            ptx.label(f"wait_{phase}")
            ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(ready, ptx.addr(mma_bar), 1)
            ptx.bra(f"done_{phase}", pred=ready)
            ptx.bra(f"wait_{phase}")
            ptx.label(f"done_{phase}")
            if phase == 0:
                with ptx.if_(tid == 0):
                    ptx.mbarrier.init(mma_bar, 1)
                ptx.bar.sync(0)

        out = reg.array(b32, BN)
        with ptx.if_(epilogue_thread):
            row = reg.scalar(u32)
            ptx.inst.shl.b32(row, warp, 5)
            ptx.inst.add.u32(row, row, lane)
            row_off = reg.scalar(u64)
            ptx.inst.mul.wide.u32(row_off, row, BN)
            d_ptr = pd + (row_off << 2)

            tmem_addr = reg.scalar(b32)
            ptx.inst.mov.b32(tmem_addr, tmem_base + ((tid << 16) & 0x600000))
            for tile in range(4):
                base_col = tile * 64
                load_addr = tmem_addr
                if not use_shift:
                    load_addr = tmem_addr + base_col
                ptx.tcgen05.ld(
                    [out[base_col + i] for i in range(64)],
                    load_addr,
                    shape="32x32b",
                    count=64,
                    dtype="b32",
                )
                ptx.tcgen05.wait_ld()
                if use_shift and tile != 3:
                    ptx.tcgen05.shift(tmem_addr)

            for col in range(BN):
                ptx.inst.st.global_.b32(ptx.addr(d_ptr, col * 4), out[col])

        with ptx.if_(alloc_warp):
            ptx.tcgen05.dealloc(tmem_base, 512)
            ptx.tcgen05.relinquish_alloc_permit()
        ptx.ret()

    return k


def run():
    import torch

    A = torch.zeros((BM, BK), device="cuda", dtype=torch.bfloat16)
    B = torch.zeros((BK, BN), device="cuda", dtype=torch.bfloat16)
    A[:, 0] = torch.arange(1, BM + 1, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    B[0, :] = 1
    for use_shift in (False, True):
        out = build(use_shift=use_shift)(A, B.t().contiguous())
        torch.cuda.synchronize()
        mode = "shift" if use_shift else "base_offset"
        print(f"[{mode}] row0 cols0..31", out[0, :32].float().cpu().tolist())
        print(f"[{mode}] row1 cols0..31", out[1, :32].float().cpu().tolist())


if __name__ == "__main__":
    run()
