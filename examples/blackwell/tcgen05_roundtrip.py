"""Blackwell tcgen05 TMEM roundtrip diagnostic.

Write a known per-thread pattern to TMEM with tcgen05.st, then load it back
with tcgen05.ld and store to GMEM. This isolates TMEM addressing and epilogue
mapping from UMMA correctness.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torch

from pyptx import Tile, kernel, ptx, reg, smem
from pyptx.specs import Layout
from pyptx.types import b32, f32, pred, u32


ROWS = 32
COLS = 64
TMEM_SLOT_OFF = 0
SMEM_BYTES = 16


def build(*, arch: str = "sm_100a"):
    @kernel(
        in_specs=(Tile(1, 1, f32, Layout.ROW),),
        out_specs=(Tile(ROWS, COLS, f32, Layout.ROW),),
        grid=(1, 1, 1),
        block=(128, 1, 1),
        arch=arch,
        smem=SMEM_BYTES,
        extern_smem=True,
    )
    def k(_x, O):
        base = smem.base()
        tmem_slot = base + TMEM_SLOT_OFF

        tid = reg.scalar(u32)
        lane = reg.scalar(u32)
        alloc_warp = reg.scalar(pred)
        active_lane = reg.scalar(pred)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        ptx.inst.and_.b32(lane, tid, 31)
        ptx.inst.setp.lt.u32(alloc_warp, tid, 32)
        ptx.inst.setp.lt.u32(active_lane, tid, ROWS)

        with ptx.if_(alloc_warp):
            ptx.tcgen05.alloc(tmem_slot, 512)
        ptx.bar.sync(0)

        tmem_base = smem.load(b32, ptx.addr(tmem_slot))
        tmem_addr = tmem_base + ((lane << 16) & 0x1F0000)

        src_bits = reg.array(b32, COLS)
        src_vals = reg.array(f32, COLS)
        dst_bits = reg.array(b32, COLS)
        dst_vals = reg.array(f32, COLS)
        for col in range(COLS):
            val = reg.scalar(f32, init=float(col + 1))
            ptx.inst.mov.b32(src_vals[col], val)
            ptx.inst.mov.b32(src_bits[col], src_vals[col])

        with ptx.if_(active_lane):
            ptx.tcgen05.st(
                tmem_addr,
                [src_bits[i] for i in range(COLS)],
                shape="32x32b",
                count=64,
                dtype="b32",
            )
            ptx.tcgen05.wait_st()
            ptx.tcgen05.ld(
                [dst_bits[i] for i in range(COLS)],
                tmem_addr,
                shape="32x32b",
                count=64,
                dtype="b32",
            )

        (po,) = ptx.global_ptrs(O)
        row_off = reg.scalar(u32)
        ptx.inst.mul.lo.u32(row_off, tid, COLS)
        base_ptr = po + ((row_off) << 2)
        with ptx.if_(active_lane):
            for col in range(COLS):
                ptx.inst.mov.b32(dst_vals[col], dst_bits[col])
                ptx.inst.st.global_.f32(ptx.addr(base_ptr, col * 4), dst_vals[col])

        with ptx.if_(alloc_warp):
            ptx.tcgen05.dealloc(tmem_base, 512)
            ptx.tcgen05.relinquish_alloc_permit()
        ptx.ret()

    return k


def run_torch():
    x = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
    out = build()(x)
    torch.cuda.synchronize()
    print(out[:4, :16].cpu())


if __name__ == "__main__":
    run_torch()
