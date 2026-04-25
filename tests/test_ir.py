"""Tests for IR node construction, immutability, and equality."""

import pytest

from pyptx.ir.types import LinkingDirective, ScalarType, StateSpace
from pyptx.ir.nodes import (
    AddressOperand,
    AddressSize,
    FormattingInfo,
    Function,
    ImmediateOperand,
    Instruction,
    Label,
    LabelOperand,
    Module,
    NegatedOperand,
    ParenthesizedOperand,
    Param,
    PipeOperand,
    Predicate,
    RegDecl,
    RegisterOperand,
    Target,
    VarDecl,
    VectorOperand,
    Version,
)


class TestScalarType:
    def test_ptx_property(self):
        assert ScalarType.B32.ptx == ".b32"
        assert ScalarType.F16X2.ptx == ".f16x2"
        assert ScalarType.PRED.ptx == ".pred"

    def test_from_ptx_with_dot(self):
        assert ScalarType.from_ptx(".b32") == ScalarType.B32

    def test_from_ptx_without_dot(self):
        assert ScalarType.from_ptx("f64") == ScalarType.F64

    def test_from_ptx_invalid(self):
        with pytest.raises(ValueError):
            ScalarType.from_ptx(".bogus")


class TestStateSpace:
    def test_ptx_property(self):
        assert StateSpace.SHARED.ptx == ".shared"
        assert StateSpace.SHARED_CTA.ptx == ".shared::cta"
        assert StateSpace.GLOBAL.ptx == ".global"

    def test_from_ptx(self):
        assert StateSpace.from_ptx(".shared::cta") == StateSpace.SHARED_CTA
        assert StateSpace.from_ptx("global") == StateSpace.GLOBAL


class TestNodes:
    def test_version(self):
        v = Version(major=8, minor=5)
        assert v.major == 8
        assert v.minor == 5

    def test_frozen(self):
        v = Version(major=8, minor=5)
        with pytest.raises(AttributeError):
            v.major = 9  # type: ignore[misc]

    def test_equality(self):
        v1 = Version(major=8, minor=5)
        v2 = Version(major=8, minor=5)
        assert v1 == v2

    def test_instruction_basic(self):
        inst = Instruction(
            opcode="mov",
            modifiers=(".b32",),
            operands=(RegisterOperand("%r0"), RegisterOperand("%r1")),
        )
        assert inst.opcode == "mov"
        assert inst.modifiers == (".b32",)
        assert len(inst.operands) == 2
        assert inst.predicate is None

    def test_instruction_predicated(self):
        inst = Instruction(
            opcode="bra",
            operands=(LabelOperand("DONE"),),
            predicate=Predicate(register="%p0", negated=False),
        )
        assert inst.predicate is not None
        assert inst.predicate.register == "%p0"
        assert not inst.predicate.negated

    def test_vector_operand(self):
        vec = VectorOperand(
            elements=(
                RegisterOperand("%r0"),
                RegisterOperand("%r1"),
                RegisterOperand("%r2"),
                RegisterOperand("%r3"),
            )
        )
        assert len(vec.elements) == 4

    def test_address_operand(self):
        addr = AddressOperand(base="%rd0", offset="16")
        assert addr.base == "%rd0"
        assert addr.offset == "16"

    def test_address_operand_no_offset(self):
        addr = AddressOperand(base="%rd0")
        assert addr.offset is None

    def test_pipe_operand(self):
        pipe = PipeOperand(
            left=RegisterOperand("%p0"),
            right=RegisterOperand("%p1"),
        )
        assert pipe.left == RegisterOperand("%p0")

    def test_reg_decl(self):
        rd = RegDecl(type=ScalarType.B32, name="%r", count=100)
        assert rd.count == 100

    def test_param(self):
        p = Param(
            state_space=StateSpace.PARAM,
            type=ScalarType.U64,
            name="param0",
        )
        assert p.state_space == StateSpace.PARAM
        assert p.array_size is None

    def test_module(self):
        m = Module(
            version=Version(8, 5),
            target=Target(("sm_90a",)),
            address_size=AddressSize(64),
            directives=(
                Function(is_entry=True, name="test", body=(
                    Instruction(opcode="ret"),
                )),
            ),
        )
        assert len(m.directives) == 1
        assert isinstance(m.directives[0], Function)

    def test_hashable(self):
        """Frozen dataclasses with tuples should be hashable."""
        inst = Instruction(
            opcode="mov",
            modifiers=(".b32",),
            operands=(RegisterOperand("%r0"), ImmediateOperand("42")),
        )
        # Should not raise
        hash(inst)
        s = {inst}
        assert inst in s
