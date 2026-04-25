"""Tests for trace-time spec-driven validation in pyptx.

These tests verify that:

  * Typed wrappers (``ptx.wgmma.*``, ``ptx.add``, ...) get validated
    against the spec at trace time and raise ``PtxValidationError`` on
    bad inputs.
  * The dot-chain escape hatch ``ptx.inst.*`` is exempt from strict
    validation so it can carry arbitrary opcodes.
  * The error message names the offending opcode/modifier chain, lists
    legal options, and pinpoints the user's source line.
  * ``set_strict(False)`` and ``with strict(False):`` disable raising
    while still recording the issues on the trace context.
  * Newly added overload specs (``cp.async.bulk.tensor``, the full
    ``tcgen05`` family, ``barrier.cluster.*``, mbarrier compound forms)
    accept their canonical modifier chains.
"""

from __future__ import annotations

import warnings

import pytest

from pyptx import kernel, ptx, reg
from pyptx.ir.nodes import (
    ImmediateOperand,
    Instruction,
    RegisterOperand,
    VectorOperand,
)
from pyptx.spec import (
    PtxValidationError,
    UnvalidatedInstructionWarning,
    ValidationIssue,
    get_specs,
    is_strict,
    register_overload,
    set_strict,
    strict,
    validate_instruction,
    validate_or_raise,
)
from pyptx.types import b32, bf16, f32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_strict_mode():
    """Each test starts in the default strict-on state."""
    set_strict(True)
    yield
    set_strict(True)


def _wgmma_inst(*, n: int) -> Instruction:
    return Instruction(
        opcode="wgmma",
        modifiers=(
            ".mma_async", ".sync", ".aligned",
            f".m64n{n}k16",
            ".f32", ".bf16", ".bf16",
        ),
        operands=(
            VectorOperand((RegisterOperand("%d0"),)),
            ImmediateOperand("0"),
            ImmediateOperand("0"),
            ImmediateOperand("1"),
            ImmediateOperand("1"),
            ImmediateOperand("1"),
        ),
    )


# ===========================================================================
# Strict-mode plumbing
# ===========================================================================


class TestStrictModeControls:
    def test_default_is_strict(self):
        assert is_strict() is True

    def test_set_strict_toggles(self):
        set_strict(False)
        assert is_strict() is False
        set_strict(True)
        assert is_strict() is True

    def test_strict_context_manager_restores(self):
        assert is_strict() is True
        with strict(False):
            assert is_strict() is False
        assert is_strict() is True

    def test_strict_context_manager_nested(self):
        with strict(False):
            with strict(True):
                assert is_strict() is True
            assert is_strict() is False
        assert is_strict() is True

    def test_strict_context_restores_on_exception(self):
        try:
            with strict(False):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert is_strict() is True


# ===========================================================================
# Direct validator behavior
# ===========================================================================


class TestValidatorOverloads:
    def test_valid_wgmma_passes(self):
        issues = validate_instruction(_wgmma_inst(n=128))
        errors = [i for i in issues if i.severity == "error"]
        assert errors == [], f"unexpected errors: {errors}"

    def test_invalid_wgmma_shape_is_error(self):
        issues = validate_instruction(_wgmma_inst(n=999))
        errors = [i for i in issues if i.severity == "error"]
        assert errors, "expected an error for n=999"
        assert any(".m64n999k16" in i.message for i in errors)

    def test_unknown_opcode_is_warning_not_error(self):
        inst = Instruction(opcode="not_a_real_opcode", modifiers=(), operands=())
        issues = validate_instruction(inst)
        assert all(i.severity == "warning" for i in issues)
        assert any("Unknown opcode" in i.message for i in issues)

    def test_cp_async_bulk_tensor_validates(self):
        # The base table holds only the cp.reduce variant — without the
        # overload registered in validate.py, this would fail.
        inst = Instruction(
            opcode="cp",
            modifiers=(
                ".async", ".bulk", ".tensor", ".2d",
                ".global", ".shared::cta",
            ),
            operands=(
                RegisterOperand("%rd0"),
                RegisterOperand("%rd1"),
            ),
        )
        errors = [
            i for i in validate_instruction(inst) if i.severity == "error"
        ]
        assert errors == [], f"unexpected errors: {errors}"

    def test_cp_reduce_async_bulk_validates(self):
        inst = Instruction(
            opcode="cp",
            modifiers=(
                ".reduce", ".async", ".bulk", ".tensor", ".2d",
                ".global", ".shared::cta", ".add",
            ),
            operands=(
                RegisterOperand("%rd0"),
                RegisterOperand("%rd1"),
            ),
        )
        errors = [
            i for i in validate_instruction(inst) if i.severity == "error"
        ]
        assert errors == [], f"unexpected errors: {errors}"

    def test_tcgen05_alloc_validates(self):
        inst = Instruction(
            opcode="tcgen05",
            modifiers=(".alloc", ".cta_group::1", ".sync", ".aligned", ".b32"),
            operands=(RegisterOperand("%rd0"), ImmediateOperand("128")),
        )
        errors = [
            i for i in validate_instruction(inst) if i.severity == "error"
        ]
        assert errors == [], f"unexpected errors: {errors}"

    def test_tcgen05_relinquish_alloc_permit_validates(self):
        inst = Instruction(
            opcode="tcgen05",
            modifiers=(".relinquish_alloc_permit", ".cta_group::1", ".sync"),
            operands=(),
        )
        errors = [
            i for i in validate_instruction(inst) if i.severity == "error"
        ]
        assert errors == [], f"unexpected errors: {errors}"

    def test_tcgen05_ld_validates(self):
        inst = Instruction(
            opcode="tcgen05",
            modifiers=(".ld", ".sync", ".aligned", ".16x64b", ".x4", ".b32"),
            operands=(RegisterOperand("%r0"), RegisterOperand("%rd0")),
        )
        errors = [
            i for i in validate_instruction(inst) if i.severity == "error"
        ]
        assert errors == [], f"unexpected errors: {errors}"

    def test_barrier_cluster_arrive_validates(self):
        inst = Instruction(
            opcode="barrier",
            modifiers=(".cluster", ".arrive"),
            operands=(),
        )
        errors = [
            i for i in validate_instruction(inst) if i.severity == "error"
        ]
        assert errors == [], f"unexpected errors: {errors}"

    def test_barrier_cluster_wait_validates(self):
        inst = Instruction(
            opcode="barrier",
            modifiers=(".cluster", ".wait", ".aligned"),
            operands=(),
        )
        errors = [
            i for i in validate_instruction(inst) if i.severity == "error"
        ]
        assert errors == [], f"unexpected errors: {errors}"

    def test_setmaxnreg_inc_validates(self):
        inst = Instruction(
            opcode="setmaxnreg",
            modifiers=(".inc", ".sync", ".aligned", ".u32"),
            operands=(ImmediateOperand("232"),),
        )
        errors = [
            i for i in validate_instruction(inst) if i.severity == "error"
        ]
        assert errors == [], f"unexpected errors: {errors}"

    def test_setmaxnreg_missing_action_errors(self):
        inst = Instruction(
            opcode="setmaxnreg",
            modifiers=(".sync", ".aligned", ".u32"),
            operands=(ImmediateOperand("232"),),
        )
        errors = [
            i for i in validate_instruction(inst) if i.severity == "error"
        ]
        assert any("action" in i.message for i in errors)

    def test_elect_sync_validates(self):
        inst = Instruction(
            opcode="elect",
            modifiers=(".sync",),
            operands=(RegisterOperand("%p0"), ImmediateOperand("0xffffffff")),
        )
        errors = [
            i for i in validate_instruction(inst) if i.severity == "error"
        ]
        assert errors == [], f"unexpected errors: {errors}"

    def test_mbarrier_arrive_expect_tx_validates(self):
        inst = Instruction(
            opcode="mbarrier",
            modifiers=(".arrive", ".expect_tx", ".shared::cta", ".b64"),
            operands=(RegisterOperand("%rd0"), ImmediateOperand("16")),
        )
        errors = [
            i for i in validate_instruction(inst) if i.severity == "error"
        ]
        assert errors == [], f"unexpected errors: {errors}"

    def test_mbarrier_try_wait_parity_validates(self):
        inst = Instruction(
            opcode="mbarrier",
            modifiers=(".try_wait", ".parity", ".shared", ".b64"),
            operands=(RegisterOperand("%rd0"), RegisterOperand("%r0")),
        )
        errors = [
            i for i in validate_instruction(inst) if i.severity == "error"
        ]
        assert errors == [], f"unexpected errors: {errors}"

    def test_overload_registry_returns_multiple_specs(self):
        specs = get_specs("cp")
        assert len(specs) >= 2, (
            f"cp should have at least 2 overload specs; got {len(specs)}"
        )

        specs = get_specs("tcgen05")
        assert len(specs) >= 2, (
            f"tcgen05 should have at least 2 overload specs; got {len(specs)}"
        )

    def test_register_overload_is_picked_up(self):
        from pyptx.spec.ptx import InstructionSpec, ModifierGroup

        before = len(get_specs("test_overload_opcode"))
        register_overload(InstructionSpec(
            opcode="test_overload_opcode",
            modifier_groups=(
                ModifierGroup("kind", (".foo",), required=True),
            ),
            min_operands=0,
            max_operands=0,
            description="test-only overload",
        ))
        after = len(get_specs("test_overload_opcode"))
        assert after == before + 1


# ===========================================================================
# Trace-time integration via @kernel
# ===========================================================================


class TestTraceTimeStrictRaises:
    def test_invalid_wgmma_shape_raises(self):
        @kernel(arch="sm_90a")
        def bad():
            d = reg.array(f32, 64, name="%d")
            ptx.wgmma.mma_async(
                shape=(64, 999, 16),
                dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                d=d, a=0, b=0,
            )
            ptx.ret()

        with pytest.raises(PtxValidationError) as exc_info:
            bad.ptx()

        msg = str(exc_info.value)
        assert "wgmma" in msg
        assert ".m64n999k16" in msg
        # The opcode chain should appear on the first line.
        assert "wgmma.mma_async" in msg.splitlines()[0]
        # Should mention the legal options ("shape" group).
        assert "shape" in msg or ".m64n128k16" in msg

    def test_too_few_operands_raises(self):
        # Use the inst.* path so we can construct a deliberately broken
        # instruction; first we have to opt back into strict for inst.*
        # by registering a one-shot wrapper.
        from pyptx.ptx import _emit  # type: ignore[attr-defined]

        @kernel(arch="sm_90a")
        def bad():
            # Bypass the inst.* escape hatch by calling _emit directly,
            # so the strict path is exercised.
            r = reg.array(b32, 4, name="%r")
            _emit("add", (".s32",), (r[0], r[1]), pred=None)
            ptx.ret()

        with pytest.raises(PtxValidationError) as exc_info:
            bad.ptx()
        assert "Too few operands" in str(exc_info.value)

    def test_too_many_operands_raises(self):
        from pyptx.ptx import _emit  # type: ignore[attr-defined]

        @kernel(arch="sm_90a")
        def bad():
            r = reg.array(b32, 4, name="%r")
            _emit("mov", (".b32",),
                  (r[0], r[1], r[2]), pred=None)
            ptx.ret()

        with pytest.raises(PtxValidationError) as exc_info:
            bad.ptx()
        assert "Too many operands" in str(exc_info.value)

    def test_missing_required_modifier_raises(self):
        from pyptx.ptx import _emit  # type: ignore[attr-defined]

        @kernel(arch="sm_90a")
        def bad():
            r = reg.array(b32, 4, name="%r")
            _emit("mov", (), (r[0], r[1]), pred=None)
            ptx.ret()

        with pytest.raises(PtxValidationError) as exc_info:
            bad.ptx()
        assert "Missing required modifier" in str(exc_info.value)

    def test_unknown_opcode_does_not_raise_just_warns(self):
        from pyptx.ptx import _emit  # type: ignore[attr-defined]

        @kernel(arch="sm_90a")
        def maybe():
            r = reg.array(b32, 4, name="%r")
            _emit("totally_made_up_opcode", (".b32",), (r[0],), pred=None)
            ptx.ret()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UnvalidatedInstructionWarning)
            # Should NOT raise.
            maybe.ptx()

        unvalidated = [
            w for w in caught
            if issubclass(w.category, UnvalidatedInstructionWarning)
        ]
        assert unvalidated, "expected an UnvalidatedInstructionWarning"
        assert "totally_made_up_opcode" in str(unvalidated[0].message)


class TestTraceTimeEscapeHatch:
    def test_inst_chain_does_not_raise(self):
        # ptx.inst.* is the escape-hatch within the escape-hatch and
        # should never raise even if the modifier chain is bogus.
        @kernel(arch="sm_90a")
        def k():
            r = reg.array(b32, 4, name="%r")
            ptx.inst.mov.b128(r[0], r[1])  # b128 not in mov spec
            ptx.inst.totally_unknown.foo(r[0])  # unknown opcode
            ptx.ret()

        # Should produce valid PTX text without raising.
        out = k.ptx()
        assert "mov.b128" in out
        assert "totally_unknown.foo" in out

    def test_inst_chain_records_issues_when_strict_off(self):
        # With strict mode off, both typed wrappers AND inst.* still
        # record validation issues on the trace context.
        @kernel(arch="sm_90a")
        def k():
            r = reg.array(b32, 4, name="%r")
            ptx.inst.mov.b999(r[0], r[1])  # bogus modifier
            ptx.ret()

        with strict(False):
            out = k.ptx()
        assert "mov.b999" in out


class TestTraceTimeStrictDisabled:
    def test_set_strict_false_does_not_raise(self):
        @kernel(arch="sm_90a")
        def bad():
            d = reg.array(f32, 64, name="%d")
            ptx.wgmma.mma_async(
                shape=(64, 999, 16),
                dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                d=d, a=0, b=0,
            )
            ptx.ret()

        set_strict(False)
        try:
            out = bad.ptx()
            assert ".m64n999k16" in out
        finally:
            set_strict(True)

    def test_strict_context_manager_disables_raising(self):
        @kernel(arch="sm_90a")
        def bad_a():
            d = reg.array(f32, 64, name="%d")
            ptx.wgmma.mma_async(
                shape=(64, 999, 16),
                dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                d=d, a=0, b=0,
            )
            ptx.ret()

        with strict(False):
            out = bad_a.ptx()
        assert ".m64n999k16" in out

        # After leaving the context, strict mode is restored. Define a
        # fresh kernel so we exercise tracing again instead of hitting
        # the per-kernel module cache.
        @kernel(arch="sm_90a")
        def bad_b():
            d = reg.array(f32, 64, name="%d")
            ptx.wgmma.mma_async(
                shape=(64, 999, 16),
                dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                d=d, a=0, b=0,
            )
            ptx.ret()

        with pytest.raises(PtxValidationError):
            bad_b.ptx()


# ===========================================================================
# PtxValidationError shape
# ===========================================================================


class TestPtxValidationError:
    def test_carries_issues_list(self):
        issues = [ValidationIssue(
            instruction=_wgmma_inst(n=999),
            message="bogus",
            severity="error",
        )]
        err = PtxValidationError(issues, user_frame="kernel.py:42 in foo()")
        assert err.issues == issues
        assert err.user_frame == "kernel.py:42 in foo()"
        assert "wgmma" in str(err)
        assert "bogus" in str(err)
        assert "kernel.py:42" in str(err)

    def test_subclass_of_exception(self):
        assert issubclass(PtxValidationError, Exception)

    def test_validate_or_raise_returns_issues_when_strict_off(self):
        set_strict(False)
        try:
            issues = validate_or_raise(_wgmma_inst(n=999))
            assert any(i.severity == "error" for i in issues)
        finally:
            set_strict(True)

    def test_validate_or_raise_raises_when_strict_on(self):
        with pytest.raises(PtxValidationError):
            validate_or_raise(_wgmma_inst(n=999))
