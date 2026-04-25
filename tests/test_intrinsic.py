"""Tests for @ptx.intrinsic scope annotations."""

import pytest

from pyptx import kernel, reg, intrinsic
from pyptx.ir.nodes import Function, Instruction, IntrinsicScope
from pyptx.kernel import Kernel
from pyptx.types import b32, f32, pred
import pyptx.ptx as ptx


# ---------------------------------------------------------------------------
# Basic intrinsic behavior
# ---------------------------------------------------------------------------

class TestIntrinsicBasic:
    def test_decorator_marker(self):
        @intrinsic
        def foo():
            pass
        assert getattr(foo, "__is_ptx_intrinsic__", False) is True

    def test_intrinsic_outside_trace_just_calls_fn(self):
        """Calling an intrinsic outside a kernel trace should just run the function."""
        called = []

        @intrinsic
        def noop():
            called.append(1)
            return 42

        result = noop()
        assert result == 42
        assert called == [1]

    def test_intrinsic_preserves_return_value(self):
        @intrinsic
        def compute():
            return "hello"

        @kernel(arch="sm_90a")
        def k():
            result = compute()
            assert result == "hello"
            ptx.ret()

        k.ptx()  # trigger the trace


# ---------------------------------------------------------------------------
# Scope IR node creation
# ---------------------------------------------------------------------------

class TestIntrinsicScope:
    def test_intrinsic_wraps_body_in_scope(self):
        # Register declarations are emitted into ctx.reg_decls (hoisted to
        # the top of the function), NOT ctx.statements, so they don't end
        # up inside the intrinsic scope. Only instructions do.
        r_outer = None

        @intrinsic
        def my_pattern(r):
            ptx.inst.mov.b32(r[0], 42)
            ptx.inst.mov.b32(r[1], 99)

        @kernel(arch="sm_90a")
        def k():
            r = reg.array(b32, 5)
            my_pattern(r)
            ptx.ret()

        module = k.module()
        func = [d for d in module.directives if isinstance(d, Function)][0]

        # Find the IntrinsicScope in the body
        scopes = [s for s in func.body if isinstance(s, IntrinsicScope)]
        assert len(scopes) == 1, f"Expected 1 IntrinsicScope, got {len(scopes)}"
        scope = scopes[0]
        assert scope.name == "my_pattern"

        # The scope body should contain the two mov instructions
        insts = [s for s in scope.body if isinstance(s, Instruction)]
        assert len(insts) == 2
        assert all(i.opcode == "mov" for i in insts)

    def test_intrinsic_name_matches_function(self):
        @intrinsic
        def load_async_tile():
            ptx.inst.bar.sync(0)

        @kernel(arch="sm_90a")
        def k():
            load_async_tile()
            ptx.ret()

        module = k.module()
        func = [d for d in module.directives if isinstance(d, Function)][0]
        scopes = [s for s in func.body if isinstance(s, IntrinsicScope)]
        assert scopes[0].name == "load_async_tile"

    def test_multiple_intrinsic_calls_produce_multiple_scopes(self):
        @intrinsic
        def inc_sync():
            ptx.inst.bar.sync(0)

        @kernel(arch="sm_90a")
        def k():
            inc_sync()
            inc_sync()
            inc_sync()
            ptx.ret()

        module = k.module()
        func = [d for d in module.directives if isinstance(d, Function)][0]
        scopes = [s for s in func.body if isinstance(s, IntrinsicScope)]
        assert len(scopes) == 3
        assert all(s.name == "inc_sync" for s in scopes)

    def test_nested_intrinsics(self):
        @intrinsic
        def inner():
            ptx.inst.bar.sync(0)

        @intrinsic
        def outer():
            inner()
            inner()

        @kernel(arch="sm_90a")
        def k():
            outer()
            ptx.ret()

        module = k.module()
        func = [d for d in module.directives if isinstance(d, Function)][0]
        outer_scopes = [s for s in func.body if isinstance(s, IntrinsicScope)]
        assert len(outer_scopes) == 1
        assert outer_scopes[0].name == "outer"

        # Inside the outer scope should be two inner scopes
        inner_scopes = [
            s for s in outer_scopes[0].body if isinstance(s, IntrinsicScope)
        ]
        assert len(inner_scopes) == 2
        assert all(s.name == "inner" for s in inner_scopes)

    def test_intrinsic_with_args_records_args_repr(self):
        @intrinsic
        def sized_pattern(count, name="default"):
            r = reg.array(b32, count)

        @kernel(arch="sm_90a")
        def k():
            sized_pattern(5)
            sized_pattern(10, name="bigger")
            ptx.ret()

        module = k.module()
        func = [d for d in module.directives if isinstance(d, Function)][0]
        scopes = [s for s in func.body if isinstance(s, IntrinsicScope)]
        assert len(scopes) == 2
        assert "5" in scopes[0].args_repr
        assert "10" in scopes[1].args_repr
        assert "name=" in scopes[1].args_repr


# ---------------------------------------------------------------------------
# Emitter rendering
# ---------------------------------------------------------------------------

class TestIntrinsicEmitter:
    def test_emitted_ptx_contains_begin_end_markers(self):
        @intrinsic
        def load_tile():
            ptx.inst.bar.sync(0)

        @kernel(arch="sm_90a")
        def k():
            load_tile()
            ptx.ret()

        text = k.ptx()
        assert "// BEGIN load_tile" in text
        assert "// END load_tile" in text
        assert "bar.sync 0;" in text

    def test_instructions_inside_scope_still_emitted(self):
        @intrinsic
        def multi_op():
            r = reg.array(b32, 3)
            ptx.inst.mov.b32(r[0], 1)
            ptx.inst.mov.b32(r[1], 2)
            ptx.inst.mov.b32(r[2], 3)

        @kernel(arch="sm_90a")
        def k():
            multi_op()
            ptx.ret()

        text = k.ptx()
        # All three movs should be present between BEGIN/END
        lines = text.split("\n")
        begin_idx = next(i for i, l in enumerate(lines) if "BEGIN multi_op" in l)
        end_idx = next(i for i, l in enumerate(lines) if "END multi_op" in l)
        between = "\n".join(lines[begin_idx:end_idx + 1])
        assert "mov.b32" in between
        assert between.count("mov.b32") == 3

    def test_scope_markers_are_comments(self):
        """Scope markers should be valid PTX comments so ptxas doesn't care."""
        @intrinsic
        def pattern():
            ptx.inst.bar.sync(0)

        @kernel(arch="sm_90a")
        def k():
            pattern()
            ptx.ret()

        text = k.ptx()
        for line in text.split("\n"):
            if "BEGIN pattern" in line or "END pattern" in line:
                assert line.strip().startswith("//"), (
                    f"scope marker should be a // comment: {line!r}"
                )


# ---------------------------------------------------------------------------
# Normalizer flattening
# ---------------------------------------------------------------------------

class TestIntrinsicNormalizer:
    def test_normalized_ir_flattens_intrinsic_scopes(self):
        """Normalization strips IntrinsicScope so structural IR comparison
        works the same whether or not intrinsics were used."""
        from pyptx.ir.normalize import normalize_module

        @intrinsic
        def wrapped():
            ptx.inst.bar.sync(0)
            ptx.inst.bar.sync(1)

        @kernel(arch="sm_90a")
        def with_intrinsic():
            wrapped()
            ptx.ret()

        @kernel(arch="sm_90a")
        def without_intrinsic():
            ptx.inst.bar.sync(0)
            ptx.inst.bar.sync(1)
            ptx.ret()

        canon_a = normalize_module(with_intrinsic.module())
        canon_b = normalize_module(without_intrinsic.module())

        # After normalization the two should have the same instruction sequence
        fn_a = [d for d in canon_a.directives if isinstance(d, Function)][0]
        fn_b = [d for d in canon_b.directives if isinstance(d, Function)][0]

        a_insts = [s for s in fn_a.body if isinstance(s, Instruction)]
        b_insts = [s for s in fn_b.body if isinstance(s, Instruction)]
        assert len(a_insts) == len(b_insts)
        for ia, ib in zip(a_insts, b_insts):
            assert ia.opcode == ib.opcode
            assert ia.modifiers == ib.modifiers


# ---------------------------------------------------------------------------
# Realistic usage pattern
# ---------------------------------------------------------------------------

class TestRealisticIntrinsic:
    def test_async_load_tile_pattern(self):
        """A realistic 'mega instruction' combining multiple PTX ops."""
        @intrinsic
        def async_barrier_arrive(bar_idx):
            ptx.inst.bar.sync(bar_idx)

        @intrinsic
        def compute_tile(acc_idx):
            r = reg.array(f32, 4)
            ptx.inst.add.f32(r[0], r[1], r[2])
            ptx.inst.mul.f32(r[3], r[0], r[0])

        @kernel(arch="sm_90a")
        def gemm_tile():
            async_barrier_arrive(0)
            compute_tile(0)
            async_barrier_arrive(1)
            compute_tile(1)
            ptx.ret()

        text = gemm_tile.ptx()
        assert text.count("BEGIN async_barrier_arrive") == 2
        assert text.count("BEGIN compute_tile") == 2
        assert text.count("END async_barrier_arrive") == 2
        assert text.count("END compute_tile") == 2
