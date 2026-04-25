"""Tests for the opt-in auto register allocation API: reg.alloc(), reg.alloc_array().

These are additive APIs that coexist with reg.scalar() and reg.array() in the
same kernel without name collisions.
"""

from __future__ import annotations

import pytest

from pyptx import kernel, ptx, reg
from pyptx._trace import trace_scope
from pyptx.ir.nodes import RegDecl
from pyptx.reg import Reg, RegArray
from pyptx.types import b32, bf16, f16, f32, f64, pred, s32, u32, u64


# ---------------------------------------------------------------------------
# reg.alloc(dtype)
# ---------------------------------------------------------------------------

class TestAllocScalar:
    def test_returns_reg(self):
        with trace_scope():
            r = reg.alloc(f32)
            assert isinstance(r, Reg)
            assert r.dtype == f32

    def test_emits_reg_decl(self):
        with trace_scope() as ctx:
            r = reg.alloc(f32)
            # Exactly one decl appended.
            assert len(ctx.reg_decls) == 1
            decl = ctx.reg_decls[0]
            assert isinstance(decl, RegDecl)
            assert decl.name == r.name
            # Scalar (not array): count is None.
            assert decl.count is None
            # Type matches dtype.
            assert decl.type.ptx == ".f32"

    def test_does_not_emit_instruction(self):
        """alloc() only allocates; it should not emit any instructions."""
        with trace_scope() as ctx:
            reg.alloc(f32)
            assert len(ctx.statements) == 0

    def test_multiple_unique_names(self):
        with trace_scope() as ctx:
            r0 = reg.alloc(f32)
            r1 = reg.alloc(f32)
            r2 = reg.alloc(f32)
            assert r0.name != r1.name
            assert r1.name != r2.name
            assert r0.name != r2.name
            # Three decls.
            assert len(ctx.reg_decls) == 3

    def test_sequential_indices(self):
        """Auto names should index from 0 upward per dtype."""
        with trace_scope():
            r0 = reg.alloc(f32)
            r1 = reg.alloc(f32)
            # Names should differ in suffix and contain 0 and 1.
            assert r0.name.endswith("0")
            assert r1.name.endswith("1")

    def test_different_dtypes_distinct_prefixes(self):
        with trace_scope():
            f = reg.alloc(f32)
            d = reg.alloc(f64)
            i = reg.alloc(b32)
            p = reg.alloc(pred)
            # All distinct names.
            names = {f.name, d.name, i.name, p.name}
            assert len(names) == 4
            # Different dtype -> different prefix family.
            assert f.name != d.name
            assert f.name != i.name
            assert i.name != p.name

    def test_separate_counters_per_dtype(self):
        """Counters per dtype should be independent."""
        with trace_scope():
            f0 = reg.alloc(f32)
            i0 = reg.alloc(b32)
            f1 = reg.alloc(f32)
            i1 = reg.alloc(b32)
            # Each pair starts at 0 and increments.
            assert f0.name.endswith("0")
            assert f1.name.endswith("1")
            assert i0.name.endswith("0")
            assert i1.name.endswith("1")

    def test_pred_alloc(self):
        with trace_scope() as ctx:
            p = reg.alloc(pred)
            assert isinstance(p, Reg)
            assert p.dtype == pred
            assert ctx.reg_decls[0].type.ptx == ".pred"

    def test_outside_trace_raises(self):
        with pytest.raises(RuntimeError, match="No active kernel trace"):
            reg.alloc(f32)


# ---------------------------------------------------------------------------
# reg.alloc operator overloading
# ---------------------------------------------------------------------------

class TestAllocOperators:
    def test_lt_emits_setp(self):
        with trace_scope() as ctx:
            r = reg.alloc(u32)
            result = r < 32
            assert isinstance(result, Reg)
            assert result.dtype == pred
            # Should have emitted a setp instruction.
            setps = [s for s in ctx.statements if getattr(s, "opcode", None) == "setp"]
            assert len(setps) == 1
            assert ".lt" in setps[0].modifiers

    def test_eq_emits_setp(self):
        with trace_scope() as ctx:
            r = reg.alloc(s32)
            result = r == 0
            assert isinstance(result, Reg)
            assert result.dtype == pred
            setps = [s for s in ctx.statements if getattr(s, "opcode", None) == "setp"]
            assert len(setps) == 1
            assert ".eq" in setps[0].modifiers

    def test_invert_returns_negpred(self):
        from pyptx.reg import NegPred
        with trace_scope():
            p = reg.alloc(pred)
            neg = ~p
            assert isinstance(neg, NegPred)
            assert neg.reg is p


# ---------------------------------------------------------------------------
# reg.alloc_array(dtype, count)
# ---------------------------------------------------------------------------

class TestAllocArray:
    def test_returns_reg_array(self):
        with trace_scope():
            arr = reg.alloc_array(f32, 64)
            assert isinstance(arr, RegArray)
            assert len(arr) == 64
            assert arr.dtype == f32

    def test_indexable(self):
        with trace_scope():
            arr = reg.alloc_array(f32, 16)
            r0 = arr[0]
            r15 = arr[15]
            assert isinstance(r0, Reg)
            assert isinstance(r15, Reg)
            assert r0.dtype == f32
            assert r15.dtype == f32
            assert r0.name != r15.name

    def test_index_out_of_range(self):
        with trace_scope():
            arr = reg.alloc_array(f32, 4)
            with pytest.raises(IndexError):
                arr[4]

    def test_emits_array_reg_decl(self):
        with trace_scope() as ctx:
            arr = reg.alloc_array(f32, 32)
            assert len(ctx.reg_decls) == 1
            decl = ctx.reg_decls[0]
            assert decl.count == 32
            assert decl.type.ptx == ".f32"
            # Base name should match the array's base.
            assert decl.name == arr.base

    def test_multiple_arrays_unique_bases(self):
        with trace_scope() as ctx:
            a = reg.alloc_array(f32, 16)
            b = reg.alloc_array(f32, 8)
            assert a.base != b.base
            assert a[0].name != b[0].name
            assert len(ctx.reg_decls) == 2

    def test_zero_count_rejected(self):
        with trace_scope():
            with pytest.raises(ValueError):
                reg.alloc_array(f32, 0)

    def test_outside_trace_raises(self):
        with pytest.raises(RuntimeError, match="No active kernel trace"):
            reg.alloc_array(f32, 8)


# ---------------------------------------------------------------------------
# Mixing alloc with the existing scalar/array API
# ---------------------------------------------------------------------------

class TestInteropWithExistingAPI:
    def test_no_name_collision_scalar(self):
        with trace_scope() as ctx:
            # Existing API
            old = reg.scalar(f32)
            old_array = reg.array(f32, 8)
            # Auto API
            new = reg.alloc(f32)
            new_arr = reg.alloc_array(f32, 4)

            all_names = {old.name, old_array.base, new.name, new_arr.base}
            assert len(all_names) == 4

            # Indexed names also distinct from old single-scalar name.
            for i in range(4):
                assert new_arr[i].name != old.name
                # And distinct from existing array indices.
                for j in range(8):
                    assert new_arr[i].name != old_array[j].name

    def test_alloc_then_scalar_no_collision(self):
        """reg.alloc(f32) should not consume the %f counter."""
        with trace_scope():
            auto = reg.alloc(f32)
            scal = reg.scalar(f32)  # uses %f
            # Distinct prefixes -> distinct names guaranteed.
            assert auto.name != scal.name
            # The scalar should start at %f0.
            assert scal.name == "%f0"

    def test_alloc_then_setp_predicates_distinct(self):
        """reg.alloc(pred) and setp-allocated %p must not collide."""
        with trace_scope():
            ap = reg.alloc(pred)
            r = reg.scalar(u32)
            from_setp = r < 5  # uses %p
            assert ap.name != from_setp.name


# ---------------------------------------------------------------------------
# End-to-end through the @kernel decorator
# ---------------------------------------------------------------------------

class TestKernelIntegration:
    def test_alloc_in_kernel(self):
        @kernel(arch="sm_90a")
        def k():
            a = reg.alloc(f32)
            b = reg.alloc(f32)
            ptx.inst.add.f32(a, a, b)
            ptx.ret()

        text = k.ptx()
        assert ".reg .f32" in text
        assert "add.f32" in text
        assert "ret;" in text

    def test_alloc_array_in_kernel(self):
        @kernel(arch="sm_90a")
        def k():
            arr = reg.alloc_array(f32, 16)
            ptx.inst.add.f32(arr[0], arr[1], arr[2])
            ptx.ret()

        text = k.ptx()
        # The base appears with <16> in the .reg decl.
        assert ".reg .f32" in text
        assert "<16>" in text
        assert "add.f32" in text

    def test_alloc_kernel_roundtrips(self):
        """PTX emitted from a reg.alloc-based kernel should round-trip."""
        from pyptx.parser import parse
        from pyptx.emitter import emit

        @kernel
        def k():
            x = reg.alloc(b32)
            y = reg.alloc(b32)
            ptx.mov(b32, x, 7)
            ptx.mov(b32, y, 11)
            ptx.ret()

        text = k.ptx()
        module = parse(text)
        assert emit(module) == text

    def test_alloc_kernel_with_predicate(self):
        @kernel
        def k():
            r = reg.alloc(u32)
            p = reg.alloc(pred)
            tid = ptx.special.tid.x()
            cond = tid < 32  # uses %p (setp)
            with ptx.if_(cond):
                ptx.mov(u32, r, 1)
            with ptx.else_():
                ptx.mov(u32, r, 0)
            ptx.ret()

        text = k.ptx()
        # Both auto pred and setp pred should be declared somewhere.
        assert ".reg .pred" in text
        assert "ret;" in text

    def test_mixed_old_and_new_in_kernel(self):
        @kernel
        def k():
            # Old API
            old = reg.array(f32, 4)
            # New API
            new0 = reg.alloc(f32)
            new1 = reg.alloc(f32)
            new_arr = reg.alloc_array(f32, 8)

            ptx.mov(f32, old[0], 1)
            ptx.mov(f32, new0, 2)
            ptx.mov(f32, new1, 3)
            ptx.mov(f32, new_arr[0], 4)
            ptx.ret()

        text = k.ptx()
        # All four .reg .f32 lines should be present (no collision crash).
        # Just confirm it emits without error and contains the expected ops.
        assert text.count(".reg .f32") >= 3
        assert "mov.f32" in text
        assert "ret;" in text

    def test_alloc_many_dtypes(self):
        @kernel
        def k():
            af = reg.alloc(f32)
            ad = reg.alloc(f64)
            ai = reg.alloc(b32)
            aL = reg.alloc(u64)
            aH = reg.alloc(f16)
            aB = reg.alloc(bf16)
            ap = reg.alloc(pred)
            ptx.ret()

        text = k.ptx()
        for t in (".f32", ".f64", ".b32", ".u64", ".f16", ".bf16", ".pred"):
            assert t in text
