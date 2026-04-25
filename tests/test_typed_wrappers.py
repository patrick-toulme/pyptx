"""Tests for the typed wrapper layer in pyptx/ptx.py.

For each new wrapper:
  1. Trace it inside a @kernel and assert the emitted PTX matches.
  2. Parse the same PTX and run the corresponding _codegen_* function
     against the resulting Instruction; assert the produced Python source
     is what we expect.
"""

from __future__ import annotations

import pytest

from pyptx import kernel, ptx, reg
from pyptx.codegen.codegen import _CodeGen
from pyptx.ir.nodes import Instruction
from pyptx.parser import parse
from pyptx.types import (
    b32,
    b64,
    bf16,
    f16,
    f32,
    pred as pred_t,
    s32,
    u32,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kernel_body(src: str) -> tuple[Instruction, ...]:
    """Parse a snippet wrapped in an entry function and return its body."""
    wrapper = (
        ".version 8.5\n.target sm_100a\n.address_size 64\n"
        ".visible .entry t()\n{\n" + src + "\n}\n"
    )
    module = parse(wrapper)
    return tuple(
        s for s in module.directives[0].body
        if isinstance(s, Instruction)
    )


def _make_cg() -> _CodeGen:
    """A CodeGen pre-populated with the conventional reg-array mappings."""
    cg = _CodeGen()
    cg._reg_arrays = {
        "%r": "r",
        "%rd": "rd",
        "%f": "f",
        "%h": "h",
        "%p": "p",
        "%rs": "rs",
    }
    cg._reg_singles = {}
    return cg


def _kernel_lines(fn) -> list[str]:
    """Return only the instruction lines from a kernel's emitted PTX."""
    text = fn.ptx()
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(".") or s.startswith(".visible"):
            continue
        if s.startswith("{") or s.startswith("}"):
            continue
        out.append(s)
    # drop the .reg lines that survived (they start with .reg)
    return [ln for ln in out if not ln.startswith(".reg")]


# ---------------------------------------------------------------------------
# wgmma
# ---------------------------------------------------------------------------


class TestWgmmaWrappers:
    def test_fence_emits(self):
        @kernel(arch="sm_90a")
        def k():
            ptx.wgmma.fence()
        assert "wgmma.fence.sync.aligned;" in k.ptx()

    def test_commit_group_emits(self):
        @kernel(arch="sm_90a")
        def k():
            ptx.wgmma.commit_group()
        assert "wgmma.commit_group.sync.aligned;" in k.ptx()

    def test_wait_group_emits(self):
        @kernel(arch="sm_90a")
        def k():
            ptx.wgmma.wait_group(2)
        assert "wgmma.wait_group.sync.aligned 2;" in k.ptx()

    def test_codegen_fence(self):
        (inst,) = _kernel_body("wgmma.fence.sync.aligned;")
        cg = _make_cg()
        assert ptx._codegen_wgmma_fence(inst, cg) == "ptx.wgmma.fence()"

    def test_codegen_commit_group(self):
        (inst,) = _kernel_body("wgmma.commit_group.sync.aligned;")
        cg = _make_cg()
        assert ptx._codegen_wgmma_commit_group(inst, cg) == "ptx.wgmma.commit_group()"

    def test_codegen_wait_group(self):
        (inst,) = _kernel_body("wgmma.wait_group.sync.aligned 2;")
        cg = _make_cg()
        assert ptx._codegen_wgmma_wait_group(inst, cg) == "ptx.wgmma.wait_group(2)"


# ---------------------------------------------------------------------------
# cp.async.bulk.tensor — tile_Nd / store_Nd / im2col / scatter4 / gather4
# ---------------------------------------------------------------------------


class TestCpAsyncBulkTensorWrappers:
    def test_tile_2d_load(self):
        @kernel(arch="sm_90a")
        def k():
            rd = reg.array(b64, 4)
            r = reg.array(b32, 4)
            ptx.cp.async_.bulk.tensor.tile_2d(
                rd[0], rd[1], (r[0], r[1]), mbar=rd[2]
            )
        text = k.ptx()
        assert (
            "cp.async.bulk.tensor.2d.shared::cluster.global"
            ".mbarrier::complete_tx::bytes "
            "[%rd0, {%r0, %r1}], [%rd1], [%rd2];" in text
        )

    def test_tile_3d_load(self):
        @kernel(arch="sm_90a")
        def k():
            rd = reg.array(b64, 4)
            r = reg.array(b32, 4)
            ptx.cp.async_.bulk.tensor.tile_3d(
                rd[0], rd[1], (r[0], r[1], r[2]), mbar=rd[2]
            )
        assert (
            "cp.async.bulk.tensor.3d.shared::cluster.global"
            ".mbarrier::complete_tx::bytes "
            "[%rd0, {%r0, %r1, %r2}], [%rd1], [%rd2];" in k.ptx()
        )

    def test_store_2d(self):
        @kernel(arch="sm_90a")
        def k():
            rd = reg.array(b64, 4)
            r = reg.array(b32, 4)
            ptx.cp.async_.bulk.tensor.store_2d(
                rd[0], rd[1], (r[0], r[1])
            )
        assert (
            "cp.async.bulk.tensor.2d.global.shared::cta "
            "[%rd0, {%r0, %r1}], [%rd1];" in k.ptx()
        )

    def test_im2col_3d(self):
        @kernel(arch="sm_100a")
        def k():
            rd = reg.array(b64, 4)
            r = reg.array(b32, 4)
            ptx.cp.async_.bulk.tensor.im2col_3d(
                rd[0], rd[1], (r[0], r[1], r[2]), mbar=rd[2]
            )
        assert ".im2col" in k.ptx()
        assert "[%rd0, {%r0, %r1, %r2}]" in k.ptx()

    def test_gather4_2d(self):
        @kernel(arch="sm_100a")
        def k():
            rd = reg.array(b64, 4)
            r = reg.array(b32, 4)
            ptx.cp.async_.bulk.tensor.gather4_2d(
                rd[0], rd[1], (r[0], r[1]), mbar=rd[2]
            )
        assert ".gather4" in k.ptx()

    def test_scatter4_2d(self):
        @kernel(arch="sm_100a")
        def k():
            rd = reg.array(b64, 4)
            r = reg.array(b32, 4)
            ptx.cp.async_.bulk.tensor.scatter4_2d(
                rd[0], rd[1], (r[0], r[1]), mbar=rd[2]
            )
        assert ".scatter4" in k.ptx()

    def test_codegen_tile_2d_load(self):
        (inst,) = _kernel_body(
            ".reg .b64 %rd<4>;\n.reg .b32 %r<4>;\n"
            "cp.async.bulk.tensor.2d.shared::cluster.global"
            ".mbarrier::complete_tx::bytes "
            "[%rd0, {%r0, %r1}], [%rd1], [%rd2];"
        )
        cg = _make_cg()
        out = ptx._codegen_cp_async_bulk_tensor(inst, cg)
        assert out == (
            "ptx.cp.async_.bulk.tensor.tile_2d(rd[0], [%rd1], (r[0], r[1]), mbar=[%rd2])"
        ) or "tile_2d" in out and "rd[0]" in out and "(r[0], r[1])" in out

    def test_codegen_store_2d(self):
        (inst,) = _kernel_body(
            ".reg .b64 %rd<4>;\n.reg .b32 %r<4>;\n"
            "cp.async.bulk.tensor.2d.global.shared::cta "
            "[%rd0, {%r0, %r1}], [%rd1];"
        )
        cg = _make_cg()
        out = ptx._codegen_cp_async_bulk_tensor_store(inst, cg)
        assert out is not None
        assert "store_2d" in out
        assert "(r[0], r[1])" in out

    def test_codegen_load_returns_none_for_store(self):
        (inst,) = _kernel_body(
            ".reg .b64 %rd<4>;\n.reg .b32 %r<4>;\n"
            "cp.async.bulk.tensor.2d.global.shared::cta "
            "[%rd0, {%r0, %r1}], [%rd1];"
        )
        cg = _make_cg()
        # The load codegen should NOT match a store
        assert ptx._codegen_cp_async_bulk_tensor(inst, cg) is None


# ---------------------------------------------------------------------------
# tcgen05
# ---------------------------------------------------------------------------


class TestTcgen05Wrappers:
    def test_alloc(self):
        @kernel(arch="sm_100a")
        def k():
            rd = reg.array(b64, 4)
            ptx.tcgen05.alloc(rd[0], 32)
        assert (
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%rd0], 32;"
            in k.ptx()
        )

    def test_dealloc(self):
        @kernel(arch="sm_100a")
        def k():
            r = reg.array(b32, 4)
            ptx.tcgen05.dealloc(r[0], 64)
        assert (
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r0, 64;" in k.ptx()
        )

    def test_relinquish_alloc_permit(self):
        @kernel(arch="sm_100a")
        def k():
            ptx.tcgen05.relinquish_alloc_permit()
        assert (
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
            in k.ptx()
        )

    def test_fence_before_thread_sync(self):
        @kernel(arch="sm_100a")
        def k():
            ptx.tcgen05.fence_before_thread_sync()
        assert "tcgen05.fence::before_thread_sync;" in k.ptx()

    def test_fence_after_thread_sync(self):
        @kernel(arch="sm_100a")
        def k():
            ptx.tcgen05.fence_after_thread_sync()
        assert "tcgen05.fence::after_thread_sync;" in k.ptx()

    def test_commit(self):
        @kernel(arch="sm_100a")
        def k():
            rd = reg.array(b64, 4)
            ptx.tcgen05.commit(rd[1])
        assert (
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
            ".shared::cluster.b64 [%rd1];" in k.ptx()
        )

    def test_commit_cta_scope(self):
        @kernel(arch="sm_100a")
        def k():
            rd = reg.array(b64, 4)
            ptx.tcgen05.commit(rd[1], space="cta")
        assert ".shared::cta.b64 [%rd1];" in k.ptx()

    def test_commit_multicast(self):
        @kernel(arch="sm_100a")
        def k():
            rd = reg.array(b64, 4)
            ptx.tcgen05.commit(rd[1], multicast=True)
        assert ".multicast::cluster" in k.ptx()

    def test_wait_ld(self):
        @kernel(arch="sm_100a")
        def k():
            ptx.tcgen05.wait_ld()
        assert "tcgen05.wait::ld.sync.aligned;" in k.ptx()

    def test_wait_st(self):
        @kernel(arch="sm_100a")
        def k():
            ptx.tcgen05.wait_st()
        assert "tcgen05.wait::st.sync.aligned;" in k.ptx()

    def test_ld(self):
        @kernel(arch="sm_100a")
        def k():
            r = reg.array(b32, 4, name="%r")
            dst = reg.array(b32, 4, name="%dst")
            ptx.tcgen05.ld(dst, r[0], shape="16x128b", count=1)
        text = k.ptx()
        assert "tcgen05.ld.sync.aligned.16x128b.x1.b32" in text
        assert "[%r0]" in text

    def test_st(self):
        @kernel(arch="sm_100a")
        def k():
            r = reg.array(b32, 4, name="%r")
            src = reg.array(b32, 4, name="%src")
            ptx.tcgen05.st(r[0], src, shape="16x128b", count=1)
        text = k.ptx()
        assert "tcgen05.st.sync.aligned.16x128b.x1.b32" in text
        assert "[%r0]" in text

    def test_cp(self):
        @kernel(arch="sm_100a")
        def k():
            r = reg.array(b32, 2)
            rd = reg.array(b64, 2)
            ptx.tcgen05.cp(r[0], rd[0])
        assert "tcgen05.cp.cta_group::1.128x256b [%r0], %rd0;" in k.ptx()

    def test_cp_variant(self):
        @kernel(arch="sm_100a")
        def k():
            r = reg.array(b32, 2)
            rd = reg.array(b64, 2)
            ptx.tcgen05.cp(r[0], rd[0], size="64x128b.warpx2::02_13")
        assert "tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%r0], %rd0;" in k.ptx()

    def test_shift(self):
        @kernel(arch="sm_100a")
        def k():
            r = reg.array(b32, 2)
            ptx.tcgen05.shift(r[0])
        assert "tcgen05.shift.cta_group::1.down [%r0];" in k.ptx()

    def test_mma(self):
        @kernel(arch="sm_100a")
        def k():
            rd = reg.array(b64, 4)
            ptx.tcgen05.mma(rd[0], rd[1], rd[2], rd[3], cta_group=1, kind="f16")
        text = k.ptx()
        assert "tcgen05.mma.cta_group::1.kind::f16" in text
        assert "[%rd0]" in text
        assert ", %p" in text
        assert "{%r" in text

    def test_mma_newer_dense_form_uses_mask_tuple(self):
        @kernel(arch="sm_100a", version=(9, 2))
        def k():
            rd = reg.array(b64, 4)
            ptx.tcgen05.mma(rd[0], rd[1], rd[2], rd[3], cta_group=1, kind="f16")

        text = k.ptx()
        assert "tcgen05.mma.cta_group::1.kind::f16" in text
        assert "{%r" in text

    def test_mma_sparse_collector(self):
        @kernel(arch="sm_100a")
        def k():
            r = reg.array(b32, 8)
            rd = reg.array(b64, 4)
            p = reg.scalar(pred_t)
            ptx.tcgen05.mma(
                r[0],
                r[1],
                rd[2],
                r[2],
                cta_group=1,
                kind="f16",
                sparse=True,
                ashift=True,
                collector_a="discard",
                a_is_tmem=True,
                sparse_metadata=r[3],
                scale_d=p,
                enable_input_d=None,
            )
        text = k.ptx()
        assert "tcgen05.mma.sp.cta_group::1.kind::f16.ashift.collector::a::discard" in text
        assert "[%r0], [%r1], %rd2, [%r3], %r2, %p0;" in text
        assert ", %p0;" in text

    @pytest.mark.parametrize(
        ("ab_dtype", "kwargs", "expected"),
        [
            ("bf16", {}, 0x08400490),
            ("f16", {}, 0x08400010),
            ("f16", {"scale_a": -1}, 0x08402010),
            ("tf32", {}, 0x08400910),
        ],
    )
    def test_make_instr_desc_f16bf16_f32(self, ab_dtype, kwargs, expected):
        desc = ptx.tcgen05.make_instr_desc_f16bf16_f32(
            m=128,
            n=256,
            ab_dtype=ab_dtype,
            a_major="K",
            b_major="K",
            **kwargs,
        )
        assert desc == expected

    def test_masked_descriptor(self):
        @kernel(arch="sm_100a")
        def k():
            base = reg.scalar(b32)
            ptx.inst.mov.b32(base, "dyn_smem")
            _ = ptx.tcgen05.masked_descriptor(base)
            ptx.ret()

        text = k.ptx()
        assert "and.b32" in text
        assert "cvt.u64.u32" in text
        assert "0x4000404000010000" in text or "4611756662049538048" in text

    def test_mma_invalid_kind(self):
        @kernel(arch="sm_100a")
        def k():
            rd = reg.array(b64, 4)
            ptx.tcgen05.mma(rd[0], rd[1], rd[2], rd[3], kind="bad")
        with pytest.raises(ValueError):
            k.ptx()

    # ----- codegens -----

    def test_codegen_alloc(self):
        (inst,) = _kernel_body(
            ".reg .b64 %rd<4>;\n"
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%rd0], 32;"
        )
        cg = _make_cg()
        out = ptx._codegen_tcgen05_alloc(inst, cg)
        assert out is not None and "ptx.tcgen05.alloc" in out
        assert "cta_group=1" in out

    def test_codegen_fence(self):
        (inst,) = _kernel_body("tcgen05.fence::before_thread_sync;")
        cg = _make_cg()
        assert (
            ptx._codegen_tcgen05_fence(inst, cg)
            == "ptx.tcgen05.fence_before_thread_sync()"
        )

    def test_codegen_wait_ld(self):
        (inst,) = _kernel_body("tcgen05.wait::ld.sync.aligned;")
        cg = _make_cg()
        assert ptx._codegen_tcgen05_wait(inst, cg) == "ptx.tcgen05.wait_ld()"

    def test_codegen_shift(self):
        (inst,) = _kernel_body(
            ".reg .b32 %r<4>;\n"
            "tcgen05.shift.cta_group::1.down [%r0];"
        )
        cg = _make_cg()
        out = ptx._codegen_tcgen05_shift(inst, cg)
        assert out is not None
        assert "ptx.tcgen05.shift" in out
        assert "cta_group=1" in out


# ---------------------------------------------------------------------------
# setmaxnreg / elect / cluster
# ---------------------------------------------------------------------------


class TestSetmaxnregWrappers:
    def test_inc(self):
        @kernel(arch="sm_90a")
        def k():
            ptx.setmaxnreg_inc(232)
        assert "setmaxnreg.inc.sync.aligned.u32 232;" in k.ptx()

    def test_dec(self):
        @kernel(arch="sm_90a")
        def k():
            ptx.setmaxnreg_dec(96)
        assert "setmaxnreg.dec.sync.aligned.u32 96;" in k.ptx()

    def test_codegen_inc(self):
        (inst,) = _kernel_body("setmaxnreg.inc.sync.aligned.u32 232;")
        cg = _make_cg()
        assert ptx._codegen_setmaxnreg(inst, cg) == "ptx.setmaxnreg_inc(232)"


class TestElectWrapper:
    def test_emit(self):
        @kernel(arch="sm_90a")
        def k():
            r = reg.array(u32, 4)
            p = reg.scalar(pred_t)
            ptx.elect_sync(r[0], p, -1)
        text = k.ptx()
        assert "elect.sync %r0|%p0, -1;" in text

    def test_codegen(self):
        (inst,) = _kernel_body(
            ".reg .u32 %r<4>;\n.reg .pred %p<4>;\n"
            "elect.sync %r0|%p0, -1;"
        )
        cg = _make_cg()
        out = ptx._codegen_elect_sync(inst, cg)
        assert out is not None
        assert out.startswith("ptx.elect_sync(")
        assert "r[0]" in out
        assert "p[0]" in out


class TestClusterWrapper:
    def test_arrive(self):
        @kernel(arch="sm_90a")
        def k():
            ptx.cluster.arrive()
        assert "barrier.cluster.arrive;" in k.ptx()

    def test_wait_aligned(self):
        @kernel(arch="sm_90a")
        def k():
            ptx.cluster.wait(aligned=True)
        assert "barrier.cluster.wait.aligned;" in k.ptx()

    def test_codegen_arrive(self):
        (inst,) = _kernel_body("barrier.cluster.arrive;")
        cg = _make_cg()
        out = ptx._codegen_barrier_cluster(inst, cg)
        assert out == "ptx.cluster.arrive()"

    def test_codegen_wait_aligned(self):
        (inst,) = _kernel_body("barrier.cluster.wait.aligned;")
        cg = _make_cg()
        out = ptx._codegen_barrier_cluster(inst, cg)
        assert out == "ptx.cluster.wait(aligned=True)"


# ---------------------------------------------------------------------------
# Common arithmetic / memory wrappers
# ---------------------------------------------------------------------------


class TestArithWrappers:
    def test_sub(self):
        @kernel(arch="sm_90a")
        def k():
            r = reg.array(s32, 4)
            ptx.sub(s32, r[0], r[1], r[2])
        assert "sub.s32 %r0, %r1, %r2;" in k.ptx()

    def test_mul_lo(self):
        @kernel(arch="sm_90a")
        def k():
            r = reg.array(s32, 4)
            ptx.mul(s32, r[0], r[1], r[2], mode="lo")
        assert "mul.lo.s32 %r0, %r1, %r2;" in k.ptx()

    def test_mul_no_mode(self):
        @kernel(arch="sm_90a")
        def k():
            f = reg.array(f32, 4)
            ptx.mul(f32, f[0], f[1], f[2])
        assert "mul.f32 %f0, %f1, %f2;" in k.ptx()

    def test_mad(self):
        @kernel(arch="sm_90a")
        def k():
            r = reg.array(s32, 4)
            ptx.mad(s32, r[0], r[1], r[2], r[3])
        assert "mad.lo.s32 %r0, %r1, %r2, %r3;" in k.ptx()

    def test_shl(self):
        @kernel(arch="sm_90a")
        def k():
            r = reg.array(b32, 4)
            ptx.shl(b32, r[0], r[1], 4)
        assert "shl.b32 %r0, %r1, 4;" in k.ptx()

    def test_shr(self):
        @kernel(arch="sm_90a")
        def k():
            r = reg.array(s32, 4)
            ptx.shr(s32, r[0], r[1], 4)
        assert "shr.s32 %r0, %r1, 4;" in k.ptx()

    def test_setp(self):
        @kernel(arch="sm_90a")
        def k():
            r = reg.array(s32, 4)
            p = reg.scalar(pred_t)
            ptx.setp("lt", s32, p, r[0], r[1])
        assert "setp.lt.s32 %p0, %r0, %r1;" in k.ptx()

    def test_setp_invalid_op(self):
        @kernel(arch="sm_90a")
        def k():
            r = reg.array(s32, 4)
            p = reg.scalar(pred_t)
            ptx.setp("zz", s32, p, r[0], r[1])
        with pytest.raises(ValueError):
            k.ptx()

    def test_cvt(self):
        @kernel(arch="sm_90a")
        def k():
            r = reg.array(b32, 4)
            f = reg.array(f32, 4)
            ptx.cvt(f16, f32, r[0], f[0], rounding="rn")
        assert "cvt.rn.f16.f32 %r0, %f0;" in k.ptx()

    def test_ld(self):
        @kernel(arch="sm_90a")
        def k():
            r = reg.array(b32, 4)
            ptx.ld(b32, r[0], ptx.addr(r[1]))
        assert "ld.global.b32 %r0, [%r1];" in k.ptx()

    def test_st_shared(self):
        @kernel(arch="sm_90a")
        def k():
            r = reg.array(b32, 4)
            ptx.st(b32, ptx.addr(r[1]), r[0], space="shared")
        assert "st.shared.b32 [%r1], %r0;" in k.ptx()

    def test_logical_ops(self):
        @kernel(arch="sm_90a")
        def k():
            r = reg.array(b32, 4)
            ptx.and_(b32, r[0], r[1], r[2])
            ptx.or_(b32, r[0], r[1], r[2])
            ptx.xor_(b32, r[0], r[1], r[2])
            ptx.not_(b32, r[0], r[1])
        text = k.ptx()
        assert "and.b32 %r0, %r1, %r2;" in text
        assert "or.b32 %r0, %r1, %r2;" in text
        assert "xor.b32 %r0, %r1, %r2;" in text
        assert "not.b32 %r0, %r1;" in text


# ---------------------------------------------------------------------------
# Codegen functions for arithmetic / memory ops
# ---------------------------------------------------------------------------


class TestArithCodegens:
    def test_codegen_mov(self):
        (inst,) = _kernel_body(
            ".reg .b32 %r<4>;\n"
            "mov.b32 %r0, %r1;"
        )
        cg = _make_cg()
        assert ptx._codegen_mov(inst, cg) == "ptx.mov(b32, r[0], r[1])"

    def test_codegen_add(self):
        (inst,) = _kernel_body(
            ".reg .s32 %r<4>;\n"
            "add.s32 %r0, %r1, %r2;"
        )
        cg = _make_cg()
        assert ptx._codegen_add(inst, cg) == "ptx.add(s32, r[0], r[1], r[2])"

    def test_codegen_sub(self):
        (inst,) = _kernel_body(
            ".reg .s32 %r<4>;\n"
            "sub.s32 %r0, %r1, %r2;"
        )
        cg = _make_cg()
        assert ptx._codegen_sub(inst, cg) == "ptx.sub(s32, r[0], r[1], r[2])"

    def test_codegen_mul_lo(self):
        (inst,) = _kernel_body(
            ".reg .s32 %r<4>;\n"
            "mul.lo.s32 %r0, %r1, %r2;"
        )
        cg = _make_cg()
        assert (
            ptx._codegen_mul(inst, cg)
            == 'ptx.mul(s32, r[0], r[1], r[2], mode="lo")'
        )

    def test_codegen_mul_no_mode(self):
        (inst,) = _kernel_body(
            ".reg .f32 %f<4>;\n"
            "mul.f32 %f0, %f1, %f2;"
        )
        cg = _make_cg()
        assert (
            ptx._codegen_mul(inst, cg) == "ptx.mul(f32, f[0], f[1], f[2])"
        )

    def test_codegen_mad(self):
        (inst,) = _kernel_body(
            ".reg .s32 %r<4>;\n"
            "mad.lo.s32 %r0, %r1, %r2, %r3;"
        )
        cg = _make_cg()
        assert (
            ptx._codegen_mad(inst, cg)
            == 'ptx.mad(s32, r[0], r[1], r[2], r[3], mode="lo")'
        )

    def test_codegen_shl(self):
        (inst,) = _kernel_body(
            ".reg .b32 %r<4>;\n"
            "shl.b32 %r0, %r1, 4;"
        )
        cg = _make_cg()
        assert ptx._codegen_shl(inst, cg) == "ptx.shl(b32, r[0], r[1], 4)"

    def test_codegen_shr(self):
        (inst,) = _kernel_body(
            ".reg .s32 %r<4>;\n"
            "shr.s32 %r0, %r1, 4;"
        )
        cg = _make_cg()
        assert ptx._codegen_shr(inst, cg) == "ptx.shr(s32, r[0], r[1], 4)"

    def test_codegen_setp(self):
        (inst,) = _kernel_body(
            ".reg .s32 %r<4>;\n.reg .pred %p<4>;\n"
            "setp.lt.s32 %p0, %r0, %r1;"
        )
        cg = _make_cg()
        assert (
            ptx._codegen_setp(inst, cg)
            == 'ptx.setp("lt", s32, p[0], r[0], r[1])'
        )

    def test_codegen_cvt(self):
        (inst,) = _kernel_body(
            ".reg .b32 %r<4>;\n.reg .f32 %f<4>;\n"
            "cvt.rn.f16.f32 %r0, %f0;"
        )
        cg = _make_cg()
        out = ptx._codegen_cvt(inst, cg)
        assert out is not None
        assert "ptx.cvt(f16, f32" in out
        assert 'rounding="rn"' in out

    def test_codegen_ld(self):
        (inst,) = _kernel_body(
            ".reg .b32 %r<4>;\n"
            "ld.global.b32 %r0, [%r1];"
        )
        cg = _make_cg()
        out = ptx._codegen_ld(inst, cg)
        assert out is not None
        assert "ptx.ld(b32" in out

    def test_codegen_st_shared(self):
        (inst,) = _kernel_body(
            ".reg .b32 %r<4>;\n"
            "st.shared.b32 [%r1], %r0;"
        )
        cg = _make_cg()
        out = ptx._codegen_st(inst, cg)
        assert out is not None
        assert "ptx.st(b32" in out
        assert 'space="shared"' in out

    def test_codegen_and(self):
        (inst,) = _kernel_body(
            ".reg .b32 %r<4>;\n"
            "and.b32 %r0, %r1, %r2;"
        )
        cg = _make_cg()
        assert ptx._codegen_and(inst, cg) == "ptx.and_(b32, r[0], r[1], r[2])"

    def test_codegen_or(self):
        (inst,) = _kernel_body(
            ".reg .b32 %r<4>;\n"
            "or.b32 %r0, %r1, %r2;"
        )
        cg = _make_cg()
        assert ptx._codegen_or(inst, cg) == "ptx.or_(b32, r[0], r[1], r[2])"

    def test_codegen_xor(self):
        (inst,) = _kernel_body(
            ".reg .b32 %r<4>;\n"
            "xor.b32 %r0, %r1, %r2;"
        )
        cg = _make_cg()
        assert ptx._codegen_xor(inst, cg) == "ptx.xor_(b32, r[0], r[1], r[2])"

    def test_codegen_not(self):
        (inst,) = _kernel_body(
            ".reg .b32 %r<4>;\n"
            "not.b32 %r0, %r1;"
        )
        cg = _make_cg()
        assert ptx._codegen_not(inst, cg) == "ptx.not_(b32, r[0], r[1])"

    def test_codegen_ret(self):
        (inst,) = _kernel_body("ret;")
        cg = _make_cg()
        assert ptx._codegen_ret(inst, cg) == "ptx.ret()"

    def test_codegen_bra(self):
        (inst,) = _kernel_body("bra L0;")
        cg = _make_cg()
        out = ptx._codegen_bra(inst, cg)
        assert out is not None
        assert out.startswith("ptx.bra(")


# ---------------------------------------------------------------------------
# mbarrier / fence / bar codegens
# ---------------------------------------------------------------------------


class TestMbarrierFenceBarCodegens:
    def test_codegen_mbarrier_init(self):
        (inst,) = _kernel_body(
            ".reg .b64 %rd<4>;\n"
            "mbarrier.init.shared.b64 [%rd0], 32;"
        )
        cg = _make_cg()
        out = ptx._codegen_mbarrier_init(inst, cg)
        assert out is not None
        assert "ptx.mbarrier.init" in out
        assert "32" in out

    def test_codegen_mbarrier_arrive(self):
        (inst,) = _kernel_body(
            ".reg .b64 %rd<4>;\n"
            "mbarrier.arrive.shared.b64 [%rd0];"
        )
        cg = _make_cg()
        out = ptx._codegen_mbarrier_arrive(inst, cg)
        assert out is not None
        assert "ptx.mbarrier.arrive" in out

    def test_codegen_mbarrier_try_wait(self):
        (inst,) = _kernel_body(
            ".reg .b64 %rd<4>;\n.reg .b32 %r<4>;\n"
            "mbarrier.try_wait.shared.b64 [%rd0], %r0;"
        )
        cg = _make_cg()
        out = ptx._codegen_mbarrier_try_wait(inst, cg)
        assert out is not None
        assert "ptx.mbarrier.try_wait" in out

    def test_codegen_fence_proxy_async(self):
        (inst,) = _kernel_body("fence.proxy.async;")
        cg = _make_cg()
        assert ptx._codegen_fence_proxy_async(inst, cg) == "ptx.fence.proxy_async()"

    def test_codegen_fence_mbarrier_init(self):
        (inst,) = _kernel_body("fence.mbarrier_init.release.cluster;")
        cg = _make_cg()
        assert (
            ptx._codegen_fence_mbarrier_init(inst, cg)
            == "ptx.fence.mbarrier_init()"
        )

    def test_codegen_bar_sync_no_args(self):
        (inst,) = _kernel_body("bar.sync 0;")
        cg = _make_cg()
        out = ptx._codegen_bar_sync(inst, cg)
        assert out is not None
        assert "ptx.bar.sync(0)" == out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestCodegenRegistry:
    def test_registry_keys_present(self):
        expected = {
            "wgmma", "cp", "tcgen05", "setmaxnreg", "elect", "barrier",
            "mbarrier", "fence", "bar", "stmatrix", "ldmatrix",
            "mov", "add", "sub", "mul", "mad", "shl", "shr", "setp",
            "cvt", "ld", "st", "and", "or", "xor", "not", "ret", "bra",
        }
        assert expected.issubset(set(ptx.TYPED_WRAPPER_CODEGEN.keys()))

    def test_registry_entries_callable(self):
        for opcode, fns in ptx.TYPED_WRAPPER_CODEGEN.items():
            assert isinstance(fns, list)
            for fn in fns:
                assert callable(fn), f"non-callable in {opcode!r}: {fn!r}"

    def test_registry_returns_none_for_mismatch(self):
        # An add instruction should not be matched by any sub codegen.
        (inst,) = _kernel_body(
            ".reg .s32 %r<4>;\n"
            "add.s32 %r0, %r1, %r2;"
        )
        cg = _make_cg()
        for fn in ptx.TYPED_WRAPPER_CODEGEN["sub"]:
            assert fn(inst, cg) is None
