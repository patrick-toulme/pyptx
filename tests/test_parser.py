"""Tests for the PTX parser."""

import pytest

from pyptx.ir.nodes import (
    AddressOperand,
    AddressSize,
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
from pyptx.ir.types import LinkingDirective, ScalarType, StateSpace
from pyptx.parser.parser import ParseError, parse


from pyptx.ir.nodes import BlankLine, Comment, RawLine


def _parse_module(source: str) -> Module:
    return parse(source)


def _statements(func: Function) -> list:
    """Filter out Comment/BlankLine from body for test assertions."""
    return [s for s in func.body if not isinstance(s, (Comment, BlankLine, RawLine))]


def _parse_single_function(source: str) -> Function:
    """Parse source and return the first Function directive."""
    m = _parse_module(source)
    for d in m.directives:
        if isinstance(d, Function):
            return d
    raise AssertionError("No Function directive found")


_HEADER = """\
.version 8.5
.target sm_90a
.address_size 64
"""


class TestModuleHeader:
    def test_version(self):
        m = _parse_module(_HEADER + "\n.visible .entry test()\n{\n\tret;\n}\n")
        assert m.version == Version(8, 5)

    def test_target(self):
        m = _parse_module(_HEADER + "\n.visible .entry test()\n{\n\tret;\n}\n")
        assert m.target == Target(("sm_90a",))

    def test_address_size(self):
        m = _parse_module(_HEADER + "\n.visible .entry test()\n{\n\tret;\n}\n")
        assert m.address_size == AddressSize(64)


class TestFunctions:
    def test_entry_no_params(self):
        f = _parse_single_function(
            _HEADER + "\n.visible .entry test()\n{\n\tret;\n}\n"
        )
        assert f.is_entry is True
        assert f.name == "test"
        assert f.params == ()
        assert f.linking == LinkingDirective.VISIBLE

    def test_entry_with_params(self):
        f = _parse_single_function(
            _HEADER + "\n.visible .entry k(\n\t.param .u64 p0,\n\t.param .u32 p1\n)\n{\n\tret;\n}\n"
        )
        assert len(f.params) == 2
        assert f.params[0].type == ScalarType.U64
        assert f.params[0].name == "p0"
        assert f.params[1].type == ScalarType.U32

    def test_func_with_return(self):
        src = _HEADER + "\n.func (.reg .b32 rv) add_one(\n\t.reg .b32 x\n)\n{\n\tret;\n}\n"
        f = _parse_single_function(src)
        assert f.is_entry is False
        assert f.return_params is not None
        assert len(f.return_params) == 1
        assert f.return_params[0].name == "rv"


class TestRegDecl:
    def test_reg_with_count(self):
        f = _parse_single_function(
            _HEADER + "\n.visible .entry t()\n{\n\t.reg .b32 %r<100>;\n\tret;\n}\n"
        )
        rd = _statements(f)[0]
        assert isinstance(rd, RegDecl)
        assert rd.type == ScalarType.B32
        assert rd.name == "%r"
        assert rd.count == 100

    def test_reg_pred(self):
        f = _parse_single_function(
            _HEADER + "\n.visible .entry t()\n{\n\t.reg .pred %p<5>;\n\tret;\n}\n"
        )
        rd = _statements(f)[0]
        assert isinstance(rd, RegDecl)
        assert rd.type == ScalarType.PRED
        assert rd.name == "%p"
        assert rd.count == 5


class TestVarDecl:
    def test_shared_with_alignment(self):
        f = _parse_single_function(
            _HEADER + "\n.visible .entry t()\n{\n\t.shared .align 128 .b8 smem[49152];\n\tret;\n}\n"
        )
        vd = _statements(f)[0]
        assert isinstance(vd, VarDecl)
        assert vd.state_space == StateSpace.SHARED
        assert vd.alignment == 128
        assert vd.type == ScalarType.B8
        assert vd.name == "smem"
        assert vd.array_size == 49152


class TestInstructions:
    def test_simple_mov(self):
        f = _parse_single_function(
            _HEADER + "\n.visible .entry t()\n{\n\tmov.b32 %r0, %r1;\n}\n"
        )
        inst = _statements(f)[0]
        assert isinstance(inst, Instruction)
        assert inst.opcode == "mov"
        assert inst.modifiers == (".b32",)
        assert len(inst.operands) == 2
        assert inst.operands[0] == RegisterOperand("%r0")
        assert inst.operands[1] == RegisterOperand("%r1")

    def test_ret(self):
        f = _parse_single_function(
            _HEADER + "\n.visible .entry t()\n{\n\tret;\n}\n"
        )
        inst = _statements(f)[0]
        assert isinstance(inst, Instruction)
        assert inst.opcode == "ret"
        assert inst.operands == ()

    def test_predicated_branch(self):
        f = _parse_single_function(
            _HEADER + "\n.visible .entry t()\n{\n\t@%p0 bra DONE;\nDONE:\n\tret;\n}\n"
        )
        inst = _statements(f)[0]
        assert isinstance(inst, Instruction)
        assert inst.predicate is not None
        assert inst.predicate.register == "%p0"
        assert not inst.predicate.negated
        assert inst.opcode == "bra"

    def test_negated_predicate(self):
        f = _parse_single_function(
            _HEADER + "\n.visible .entry t()\n{\n\t@!%p0 mov.b32 %r1, 0;\n}\n"
        )
        inst = _statements(f)[0]
        assert isinstance(inst, Instruction)
        assert inst.predicate is not None
        assert inst.predicate.negated

    def test_long_modifier_chain(self):
        f = _parse_single_function(
            _HEADER + "\n.visible .entry t()\n{\n\twgmma.mma_async.sync.aligned.m64n256k16.f32.e4m3.e4m3 %r0, %r1, 1;\n}\n"
        )
        inst = _statements(f)[0]
        assert isinstance(inst, Instruction)
        assert inst.opcode == "wgmma"
        assert inst.modifiers == (
            ".mma_async", ".sync", ".aligned",
            ".m64n256k16", ".f32", ".e4m3", ".e4m3",
        )

    def test_address_operand_no_offset(self):
        f = _parse_single_function(
            _HEADER + "\n.visible .entry t()\n{\n\tld.param.u64 %rd0, [param0];\n}\n"
        )
        inst = _statements(f)[0]
        assert isinstance(inst, Instruction)
        assert isinstance(inst.operands[1], AddressOperand)
        assert inst.operands[1].base == "param0"
        assert inst.operands[1].offset is None

    def test_address_operand_with_offset(self):
        f = _parse_single_function(
            _HEADER + "\n.visible .entry t()\n{\n\tst.shared.b32 [%rd0+4], %r0;\n}\n"
        )
        inst = _statements(f)[0]
        assert isinstance(inst, Instruction)
        assert isinstance(inst.operands[0], AddressOperand)
        assert inst.operands[0].base == "%rd0"
        assert inst.operands[0].offset == "4"

    def test_immediate_operand(self):
        f = _parse_single_function(
            _HEADER + "\n.visible .entry t()\n{\n\tmov.b32 %r0, 42;\n}\n"
        )
        inst = _statements(f)[0]
        assert isinstance(inst, Instruction)
        assert inst.operands[1] == ImmediateOperand("42")

    def test_hex_immediate(self):
        f = _parse_single_function(
            _HEADER + "\n.visible .entry t()\n{\n\tmov.b32 %r0, 0xFF;\n}\n"
        )
        inst = _statements(f)[0]
        assert isinstance(inst, Instruction)
        assert inst.operands[1] == ImmediateOperand("0xFF")

    def test_vector_operand(self):
        f = _parse_single_function(
            _HEADER + "\n.visible .entry t()\n{\n\tldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r0, %r1, %r2, %r3}, [%rd0];\n}\n"
        )
        inst = _statements(f)[0]
        assert isinstance(inst, Instruction)
        assert isinstance(inst.operands[0], VectorOperand)
        assert len(inst.operands[0].elements) == 4

    def test_pipe_operand(self):
        f = _parse_single_function(
            _HEADER + "\n.visible .entry t()\n{\n\tsetp.lt.f32 %p1|%p2, %r2, %r3;\n}\n"
        )
        inst = _statements(f)[0]
        assert isinstance(inst, Instruction)
        assert isinstance(inst.operands[0], PipeOperand)
        assert inst.operands[0].left == RegisterOperand("%p1")
        assert inst.operands[0].right == RegisterOperand("%p2")


class TestLabels:
    def test_label(self):
        f = _parse_single_function(
            _HEADER + "\n.visible .entry t()\n{\nLOOP:\n\tret;\n}\n"
        )
        assert isinstance(_statements(f)[0], Label)
        assert _statements(f)[0].name == "LOOP"


class TestCallInstruction:
    def test_call_with_return_and_args(self):
        src = (
            _HEADER
            + "\n.func (.reg .b32 rv) add_one(\n\t.reg .b32 x\n)\n{\n\tret;\n}\n"
            + "\n.visible .entry t()\n{\n\t.reg .b32 %r<5>;\n\tcall (%r1), add_one, (%r0);\n\tret;\n}\n"
        )
        m = _parse_module(src)
        functions = [d for d in m.directives if isinstance(d, Function)]
        func = functions[1]
        call_inst = _statements(func)[1]
        assert isinstance(call_inst, Instruction)
        assert call_inst.opcode == "call"
        # First operand: parenthesized return
        assert isinstance(call_inst.operands[0], ParenthesizedOperand)
        # Second operand: function name
        assert isinstance(call_inst.operands[1], LabelOperand)
        assert call_inst.operands[1].name == "add_one"
        # Third operand: parenthesized args
        assert isinstance(call_inst.operands[2], ParenthesizedOperand)


class TestParseError:
    def test_bad_token(self):
        with pytest.raises(ParseError):
            _parse_module("not valid ptx")
