"""Tiny Blackwell tcgen05 smoke kernels.

These isolate alloc/commit/ld behavior from the full GEMM path.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from pyptx import Tile, kernel, ptx, reg, smem
from pyptx.smem import apply_swizzle
from pyptx.specs import Layout
from pyptx.types import b32, b64, f32, pred, u32


A_BYTES = 128 * 64 * 2
B_BYTES = 256 * 64 * 2
BAR_OFF = A_BYTES + B_BYTES
TMEM_SLOT_OFF = BAR_OFF + 16
SMEM_BYTES = TMEM_SLOT_OFF + 16


def build_alloc_only(*, arch: str = "sm_100a"):
    @kernel(
        in_specs=(Tile(1, 1, f32, Layout.ROW),),
        out_specs=(Tile(1, 1, f32, Layout.ROW),),
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
        alloc_warp = reg.scalar(pred)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        ptx.inst.setp.lt.u32(alloc_warp, tid, 32)
        with ptx.if_(alloc_warp):
            ptx.tcgen05.alloc(tmem_slot, 512)
        ptx.bar.sync(0)
        tmem = smem.load(b32, ptx.addr(tmem_slot))
        with ptx.if_(alloc_warp):
            ptx.tcgen05.dealloc(tmem, 512)
            ptx.tcgen05.relinquish_alloc_permit()
        with ptx.if_(tid == 0):
            (po,) = ptx.global_ptrs(O)
            val = reg.scalar(f32, init=11.0)
            ptx.inst.st.global_.f32(ptx.addr(po), val)
        ptx.ret()
    return k


def build_mma_only(*, arch: str = "sm_100a"):
    @kernel(
        in_specs=(Tile(1, 1, f32, Layout.ROW),),
        out_specs=(Tile(1, 1, f32, Layout.ROW),),
        grid=(1, 1, 1),
        block=(128, 1, 1),
        arch=arch,
        smem=SMEM_BYTES,
        extern_smem=True,
    )
    def k(_x, O):
        base = smem.base()
        bar = base + BAR_OFF
        tmem_slot = base + TMEM_SLOT_OFF
        tid = reg.scalar(u32)
        alloc_warp = reg.scalar(pred)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        ptx.inst.setp.lt.u32(alloc_warp, tid, 32)
        with ptx.if_(tid == 0):
            ptx.mbarrier.init(bar, 1)
        with ptx.if_(alloc_warp):
            ptx.tcgen05.alloc(tmem_slot, 512)
        # Materialize aligned B128-swizzled operand tiles in shared memory.
        a_idx = reg.scalar(u32, init=0)
        a_keep = reg.scalar(pred)
        ptx.inst.setp.lt.u32(a_keep, a_idx, A_BYTES // 4)
        with ptx.loop("fill_a_loop", pred=a_keep):
            logical = a_idx << 2
            physical = apply_swizzle(logical, "128B")
            ptx.inst.st.shared.b32(ptx.addr(base + physical), 0)
            a_idx += 128
            ptx.inst.setp.lt.u32(a_keep, a_idx, A_BYTES // 4)
        b_idx = reg.scalar(u32, init=0)
        b_keep = reg.scalar(pred)
        ptx.inst.setp.lt.u32(b_keep, b_idx, B_BYTES // 4)
        with ptx.loop("fill_b_loop", pred=b_keep):
            logical = b_idx << 2
            physical = apply_swizzle(logical, "128B")
            ptx.inst.st.shared.b32(ptx.addr(base + A_BYTES + physical), 0)
            b_idx += 128
            ptx.inst.setp.lt.u32(b_keep, b_idx, B_BYTES // 4)
        ptx.bar.sync(0)
        tmem = smem.load(b32, ptx.addr(tmem_slot))
        idesc = reg.scalar(b32, init=ptx.tcgen05.make_instr_desc_f16bf16_f32())
        desc_a = ptx.tcgen05.masked_descriptor(base)
        desc_b = ptx.tcgen05.masked_descriptor(base, byte_offset=A_BYTES)
        with ptx.if_(tid == 0):
            ptx.tcgen05.mma(
                tmem,
                desc_a,
                desc_b,
                idesc,
                kind="f16",
                pred_operand=True,
                enable_input_d=False,
            )
            ptx.tcgen05.commit(bar, space="cluster")
        with ptx.scope():
            ready = reg.scalar(pred)
            ptx.label("wait")
            ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(ready, ptx.addr(bar), 0)
            ptx.bra("done", pred=ready)
            ptx.bra("wait")
            ptx.label("done")
        with ptx.if_(alloc_warp):
            ptx.tcgen05.dealloc(tmem, 512)
            ptx.tcgen05.relinquish_alloc_permit()
        with ptx.if_(tid == 0):
            (po,) = ptx.global_ptrs(O)
            val = reg.scalar(f32, init=22.0)
            ptx.inst.st.global_.f32(ptx.addr(po), val)
        ptx.ret()
    return k


def build_ld_only(*, arch: str = "sm_100a"):
    @kernel(
        in_specs=(Tile(1, 1, f32, Layout.ROW),),
        out_specs=(Tile(1, 1, f32, Layout.ROW),),
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
        alloc_warp = reg.scalar(pred)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        ptx.inst.setp.lt.u32(alloc_warp, tid, 32)
        with ptx.if_(alloc_warp):
            ptx.tcgen05.alloc(tmem_slot, 512)
        ptx.bar.sync(0)
        tmem = smem.load(b32, ptx.addr(tmem_slot))
        out = reg.array(b32, 1)
        ptx.tcgen05.ld([out[0]], tmem, shape="32x32b", count=1, dtype="b32")
        with ptx.if_(alloc_warp):
            ptx.tcgen05.dealloc(tmem, 512)
            ptx.tcgen05.relinquish_alloc_permit()
        with ptx.if_(tid == 0):
            (po,) = ptx.global_ptrs(O)
            val = reg.scalar(f32, init=33.0)
            ptx.inst.st.global_.f32(ptx.addr(po), val)
        ptx.ret()
    return k


def run_torch(builder):
    import torch
    x = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
    out = builder()(x)
    torch.cuda.synchronize()
    return int(out[0, 0].item())


def run_jax(builder):
    k = builder()
    x = jnp.zeros((1, 1), dtype=jnp.float32)
    out = np.asarray(jax.jit(lambda y: k(y))(x))
    return int(out[0, 0])
