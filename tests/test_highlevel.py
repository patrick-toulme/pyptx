"""Tests for the high-level API: @kernel, reg, smem, ptx."""

import pytest

from pyptx import kernel, reg, smem, ptx
from pyptx.types import bf16, f32, b32, u32, u64, s32, pred
from pyptx.reg import Reg, NegPred, RegArray
from pyptx.smem import SharedAlloc, MbarrierArray
from pyptx._trace import trace_scope


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class TestTypes:
    def test_ptx_property(self):
        assert f32.ptx == ".f32"
        assert bf16.ptx == ".bf16"
        assert pred.ptx == ".pred"

    def test_repr(self):
        assert repr(f32) == "f32"

    def test_bits(self):
        assert f32.bits == 32
        assert bf16.bits == 16
        assert u64.bits == 64


# ---------------------------------------------------------------------------
# Registers
# ---------------------------------------------------------------------------

class TestReg:
    def test_array(self):
        with trace_scope():
            acc = reg.array(f32, 64)
            assert isinstance(acc, RegArray)
            assert len(acc) == 64
            assert acc[0].dtype == f32
            assert acc[63].dtype == f32

    def test_array_indexing(self):
        with trace_scope():
            r = reg.array(b32, 10)
            assert r[0].name.endswith("0")
            assert r[9].name.endswith("9")

    def test_array_out_of_bounds(self):
        with trace_scope():
            r = reg.array(b32, 5)
            with pytest.raises(IndexError):
                r[5]

    def test_scalar(self):
        with trace_scope():
            p = reg.scalar(pred)
            assert isinstance(p, Reg)
            assert p.dtype == pred

    def test_scalar_with_init(self):
        with trace_scope() as ctx:
            phase = reg.scalar(b32, init=0)
            assert isinstance(phase, Reg)
            # Should have emitted a mov instruction
            assert len(ctx.statements) == 1
            assert ctx.statements[0].opcode == "mov"

    def test_comparison_emits_setp(self):
        with trace_scope() as ctx:
            r = reg.array(u32, 5)
            result = r[0] < 32
            assert isinstance(result, Reg)
            assert result.dtype == pred
            assert len(ctx.statements) == 1
            assert ctx.statements[0].opcode == "setp"
            assert ".lt" in ctx.statements[0].modifiers

    def test_negate(self):
        with trace_scope():
            p = reg.scalar(pred)
            neg = ~p
            assert isinstance(neg, NegPred)
            assert neg.reg is p

    def test_outside_trace_raises(self):
        with pytest.raises(RuntimeError, match="No active kernel trace"):
            reg.array(f32, 10)

    def test_inplace_integer_ops_emit_to_same_register(self):
        with trace_scope() as ctx:
            r = reg.scalar(u32, init=0)
            r += 1
            r ^= 3
            r <<= 2
            assert ctx.statements[1].opcode == "add"
            assert ctx.statements[1].operands[0].name == r.name
            assert ctx.statements[2].opcode == "xor"
            assert ctx.statements[2].operands[0].name == r.name
            assert ctx.statements[3].opcode == "shl"
            assert ctx.statements[3].operands[0].name == r.name


# ---------------------------------------------------------------------------
# Shared memory
# ---------------------------------------------------------------------------

class TestSmem:
    def test_alloc(self):
        with trace_scope() as ctx:
            sA = smem.alloc(bf16, (3, 128, 64))
            assert isinstance(sA, SharedAlloc)
            assert sA.dtype == bf16
            assert sA.shape == (3, 128, 64)
            assert len(ctx.var_decls) == 1
            # bf16 is 2 bytes, 3*128*64 = 24576 elements, 49152 bytes
            assert ctx.var_decls[0].array_size == 49152

    def test_alloc_indexing(self):
        with trace_scope():
            sA = smem.alloc(bf16, (3, 128, 64))
            s = sA[0]
            assert s.stage == 0
            assert s.name == sA.name

    def test_mbarrier(self):
        with trace_scope() as ctx:
            bar = smem.mbarrier(3)
            assert isinstance(bar, MbarrierArray)
            assert bar.count == 3
            assert len(ctx.var_decls) == 1

    def test_mbarrier_indexing(self):
        with trace_scope():
            bar = smem.mbarrier(3)
            ref = bar[0]
            assert ref.idx == 0
            with pytest.raises(IndexError):
                bar[3]


# ---------------------------------------------------------------------------
# PTX instructions
# ---------------------------------------------------------------------------

class TestPtxInstructions:
    def test_special_tid(self):
        tid = ptx.special.tid.x()
        assert isinstance(tid, Reg)
        assert tid.name == "%tid.x"

    def test_special_ctaid(self):
        ctaid = ptx.special.ctaid.y()
        assert ctaid.name == "%ctaid.y"

    def test_wgmma_mma_async(self):
        with trace_scope() as ctx:
            acc = reg.array(f32, 16)
            rd = reg.array(u64, 5)
            ptx.wgmma.mma_async(
                shape=(64, 256, 16),
                dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                d=acc, a=rd[0], b=rd[1],
                scale_d=1, scale_a=1, scale_b=1,
            )
            # Passing scale_d as a Python int emits a tiny 2-instr
            # prologue (mov + setp) to build the .pred operand the
            # wgmma instruction expects. The actual wgmma is the last
            # statement in the trace.
            wgmma_insts = [s for s in ctx.statements if s.opcode == "wgmma"]
            assert len(wgmma_insts) == 1
            inst = wgmma_insts[0]
            assert ".mma_async" in inst.modifiers
            assert ".m64n256k16" in inst.modifiers
            assert ".f32" in inst.modifiers
            assert ".bf16" in inst.modifiers
            # Check the operand list: d-vec, a, b, scale-d pred,
            # scale_a=1, scale_b=1, trans_a=0, trans_b=0 → 8 ops total.
            assert len(inst.operands) == 8

    def test_wgmma_fence(self):
        with trace_scope() as ctx:
            ptx.wgmma.fence()
            assert ctx.statements[0].opcode == "wgmma"
            assert ".fence" in ctx.statements[0].modifiers

    def test_wgmma_commit_group(self):
        with trace_scope() as ctx:
            ptx.wgmma.commit_group()
            assert ".commit_group" in ctx.statements[0].modifiers

    def test_wgmma_wait_group(self):
        with trace_scope() as ctx:
            ptx.wgmma.wait_group(0)
            assert ".wait_group" in ctx.statements[0].modifiers

    def test_bar_sync(self):
        with trace_scope() as ctx:
            ptx.bar.sync(0)
            assert ctx.statements[0].opcode == "bar"
            assert ".sync" in ctx.statements[0].modifiers

    def test_raw(self):
        with trace_scope() as ctx:
            ptx.raw("fence.proxy.async;")
            assert len(ctx.statements) == 1
            assert ctx.statements[0].opcode == "fence"

    def test_mov(self):
        with trace_scope() as ctx:
            r = reg.array(b32, 5)
            ptx.mov(b32, r[0], 42)
            assert ctx.statements[0].opcode == "mov"
            assert ".b32" in ctx.statements[0].modifiers

    def test_ret(self):
        with trace_scope() as ctx:
            ptx.ret()
            assert ctx.statements[0].opcode == "ret"

    def test_generic_inst(self):
        with trace_scope() as ctx:
            r = reg.array(u64, 5)
            ptx.inst.shl.b64(r[0], r[1], 2)
            assert ctx.statements[0].opcode == "shl"
            assert ".b64" in ctx.statements[0].modifiers

    def test_loop_carried_inplace_sugar(self):
        def k():
            i = reg.scalar(u32, init=0)
            keep_going = reg.scalar(pred)
            with ptx.loop("L", pred=keep_going):
                i += 1
                ptx.inst.setp.lt.u32(keep_going, i, 4)
            ptx.ret()

        text = kernel(arch="sm_90a")(k).ptx()
        assert "add.u32 %r0, %r0, 1;" in text
        assert "setp.lt.u32 %p0, %r0, 4;" in text


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------

class TestControlFlow:
    def test_if(self):
        with trace_scope() as ctx:
            p = reg.scalar(pred)
            with ptx.if_(p):
                ptx.ret()
            # @!p bra $else, ret, bra $endif, $else:, $endif:
            # (the tentative $endif: is needed so an if_() without a
            #  following else_() is well-formed — else_() pops it.)
            assert len(ctx.statements) == 5
            assert ctx.statements[0].opcode == "bra"
            assert ctx.statements[0].predicate.negated is True
            assert ctx.statements[1].opcode == "ret"
            # Last two statements are the $else: and $endif: labels.
            from pyptx.ir.nodes import Label
            assert isinstance(ctx.statements[-1], Label)
            assert isinstance(ctx.statements[-2], Label)

    def test_if_else(self):
        with trace_scope() as ctx:
            p = reg.scalar(pred)
            r = reg.array(b32, 5)
            with ptx.if_(p):
                ptx.mov(b32, r[0], 1)
            with ptx.else_():
                ptx.mov(b32, r[0], 0)
            # @!p bra $else, mov 1, bra $end, $else:, mov 0, $end:
            assert len(ctx.statements) == 6

    def test_range(self):
        with trace_scope() as ctx:
            for k in ptx.range_(0, 256, 64):
                assert isinstance(k, Reg)
                ptx.bar.sync(0)
            # mov, label, setp, @p bra endloop, bar.sync, add, bra loop, endloop:
            assert len(ctx.statements) == 8


# ---------------------------------------------------------------------------
# @kernel decorator
# ---------------------------------------------------------------------------

class TestKernel:
    def test_basic(self):
        @kernel(arch="sm_90a")
        def simple():
            r = reg.array(b32, 5)
            ptx.mov(b32, r[0], 42)
            ptx.ret()

        text = simple.ptx()
        assert ".version 8.5" in text
        assert ".target sm_90a" in text
        assert ".visible .entry simple()" in text
        assert "mov.b32" in text
        assert "ret;" in text

    def test_with_template_kwargs(self):
        @kernel(arch="sm_90a")
        def tiled(*, BM=128, BN=256):
            acc = reg.array(f32, BM * BN // 128)
            ptx.ret()

        text = tiled.ptx(BM=128, BN=256)
        assert ".reg .f32" in text
        assert "ret;" in text

    def test_inspection_is_parseable(self):
        """PTX emitted by @kernel should round-trip through the parser."""
        from pyptx.parser import parse
        from pyptx.emitter import emit

        @kernel
        def roundtrip_test():
            r = reg.array(b32, 10)
            p = reg.scalar(pred)
            ptx.mov(b32, r[0], 42)
            result = r[0] < 100
            with ptx.if_(result):
                ptx.mov(b32, r[1], 1)
            with ptx.else_():
                ptx.mov(b32, r[1], 0)
            ptx.ret()

        text = roundtrip_test.ptx()
        module = parse(text)
        assert emit(module) == text

    def test_call_raises_without_in_specs(self):
        @kernel
        def not_yet():
            ptx.ret()

        with pytest.raises(NotImplementedError, match="in_specs"):
            not_yet()

    def test_wgmma_kernel(self):
        """Build a realistic wgmma kernel and verify PTX output."""
        @kernel(arch="sm_90a")
        def wgmma_test():
            acc = reg.array(f32, 16)
            rd = reg.array(u64, 5)
            phase = reg.scalar(b32, init=0)

            tid = ptx.special.tid.x()
            is_producer = tid < 32

            ptx.wgmma.fence()
            ptx.wgmma.mma_async(
                shape=(64, 256, 16),
                dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                d=acc, a=rd[0], b=rd[1],
                scale_d=1, scale_a=1, scale_b=1,
            )
            ptx.wgmma.commit_group()
            ptx.wgmma.wait_group(0)
            ptx.ret()

        text = wgmma_test.ptx()
        assert "wgmma.fence.sync.aligned;" in text
        assert "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16" in text
        assert "wgmma.commit_group.sync.aligned;" in text
        assert "wgmma.wait_group.sync.aligned 0;" in text

    def test_repr(self):
        @kernel
        def my_fn():
            ptx.ret()
        assert "my_fn" in repr(my_fn)
