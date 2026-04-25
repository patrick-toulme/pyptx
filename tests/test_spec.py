"""Tests for the declarative ISA spec and validator."""

from pyptx.ir.nodes import (
    AddressOperand,
    ImmediateOperand,
    Instruction,
    RegisterOperand,
    VectorOperand,
)
from pyptx.spec.ptx import INSTRUCTIONS, InstructionSpec, ModifierGroup
from pyptx.spec.validate import validate_instruction


class TestSpecConsistency:
    def test_all_specs_have_opcode(self):
        for key, spec in INSTRUCTIONS.items():
            assert spec.opcode, f"Spec {key!r} has empty opcode"

    def test_all_specs_have_description(self):
        for key, spec in INSTRUCTIONS.items():
            assert spec.description, f"Spec {key!r} has empty description"

    def test_no_duplicate_options_in_groups(self):
        for key, spec in INSTRUCTIONS.items():
            for group in spec.modifier_groups:
                assert len(group.options) == len(set(group.options)), (
                    f"Spec {key!r}, group {group.name!r} has duplicate options"
                )

    def test_required_groups_have_options(self):
        for key, spec in INSTRUCTIONS.items():
            for group in spec.modifier_groups:
                if group.required:
                    assert len(group.options) > 0, (
                        f"Spec {key!r}, required group {group.name!r} has no options"
                    )

    def test_min_le_max_operands(self):
        for key, spec in INSTRUCTIONS.items():
            assert spec.min_operands <= spec.max_operands, (
                f"Spec {key!r}: min_operands > max_operands"
            )

    def test_spec_count(self):
        """Ensure we have a reasonable number of specs defined."""
        assert len(INSTRUCTIONS) >= 20, (
            f"Only {len(INSTRUCTIONS)} specs defined — expected at least 20"
        )


class TestValidation:
    def test_valid_mov(self):
        inst = Instruction(
            opcode="mov",
            modifiers=(".b32",),
            operands=(RegisterOperand("%r0"), RegisterOperand("%r1")),
        )
        errors = validate_instruction(inst)
        assert len(errors) == 0

    def test_missing_required_modifier(self):
        inst = Instruction(
            opcode="mov",
            modifiers=(),
            operands=(RegisterOperand("%r0"), RegisterOperand("%r1")),
        )
        errors = validate_instruction(inst)
        assert any("Missing required modifier" in str(e) for e in errors)

    def test_unknown_opcode(self):
        inst = Instruction(
            opcode="bogus_instr",
            modifiers=(".b32",),
            operands=(RegisterOperand("%r0"),),
        )
        errors = validate_instruction(inst)
        assert any("Unknown opcode" in str(e) for e in errors)

    def test_too_few_operands(self):
        inst = Instruction(
            opcode="add",
            modifiers=(".s32",),
            operands=(RegisterOperand("%r0"), RegisterOperand("%r1")),
        )
        errors = validate_instruction(inst)
        assert any("Too few operands" in str(e) for e in errors)

    def test_too_many_operands(self):
        inst = Instruction(
            opcode="mov",
            modifiers=(".b32",),
            operands=(
                RegisterOperand("%r0"),
                RegisterOperand("%r1"),
                RegisterOperand("%r2"),
            ),
        )
        errors = validate_instruction(inst)
        assert any("Too many operands" in str(e) for e in errors)

    def test_valid_add(self):
        inst = Instruction(
            opcode="add",
            modifiers=(".s32",),
            operands=(
                RegisterOperand("%r0"),
                RegisterOperand("%r1"),
                RegisterOperand("%r2"),
            ),
        )
        errors = validate_instruction(inst)
        assert len(errors) == 0

    def test_valid_setp(self):
        inst = Instruction(
            opcode="setp",
            modifiers=(".lt", ".f32"),
            operands=(
                RegisterOperand("%p0"),
                RegisterOperand("%r0"),
                RegisterOperand("%r1"),
            ),
        )
        errors = validate_instruction(inst)
        assert len(errors) == 0

    def test_valid_vector_ld_and_st(self):
        ld = Instruction(
            opcode="ld",
            modifiers=(".global", ".v4", ".f32"),
            operands=(
                VectorOperand((
                    RegisterOperand("%f0"),
                    RegisterOperand("%f1"),
                    RegisterOperand("%f2"),
                    RegisterOperand("%f3"),
                )),
                AddressOperand(base="%rd0"),
            ),
        )
        st = Instruction(
            opcode="st",
            modifiers=(".global", ".v4", ".f32"),
            operands=(
                AddressOperand(base="%rd0"),
                VectorOperand((
                    RegisterOperand("%f0"),
                    RegisterOperand("%f1"),
                    RegisterOperand("%f2"),
                    RegisterOperand("%f3"),
                )),
            ),
        )
        assert validate_instruction(ld) == []
        assert validate_instruction(st) == []
