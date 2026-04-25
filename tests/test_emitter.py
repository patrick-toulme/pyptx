"""Tests for the PTX emitter: hand-built IR → expected text."""

from pyptx.emitter import emit
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


def _minimal_module(**kwargs) -> Module:
    defaults = dict(
        version=Version(8, 5),
        target=Target(("sm_90a",)),
        address_size=AddressSize(64),
        directives=(),
    )
    defaults.update(kwargs)
    return Module(**defaults)


class TestEmitModuleHeader:
    def test_version(self):
        m = _minimal_module()
        text = emit(m)
        assert ".version 8.5" in text

    def test_target(self):
        m = _minimal_module()
        text = emit(m)
        assert ".target sm_90a" in text

    def test_address_size(self):
        m = _minimal_module()
        text = emit(m)
        assert ".address_size 64" in text


class TestEmitRegDecl:
    def test_reg_with_count(self):
        m = _minimal_module(
            directives=(
                Function(
                    is_entry=True,
                    name="test",
                    body=(RegDecl(type=ScalarType.B32, name="%r", count=100),),
                ),
            )
        )
        text = emit(m)
        assert "\t.reg .b32 %r<100>;" in text

    def test_reg_single(self):
        m = _minimal_module(
            directives=(
                Function(
                    is_entry=True,
                    name="test",
                    body=(RegDecl(type=ScalarType.PRED, name="%p0"),),
                ),
            )
        )
        text = emit(m)
        assert "\t.reg .pred %p0;" in text


class TestEmitInstruction:
    def test_simple_mov(self):
        m = _minimal_module(
            directives=(
                Function(
                    is_entry=True,
                    name="test",
                    body=(
                        Instruction(
                            opcode="mov",
                            modifiers=(".b32",),
                            operands=(
                                RegisterOperand("%r0"),
                                RegisterOperand("%r1"),
                            ),
                        ),
                    ),
                ),
            )
        )
        text = emit(m)
        assert "\tmov.b32 %r0, %r1;" in text

    def test_predicated_branch(self):
        m = _minimal_module(
            directives=(
                Function(
                    is_entry=True,
                    name="test",
                    body=(
                        Instruction(
                            opcode="bra",
                            operands=(LabelOperand("DONE"),),
                            predicate=Predicate(register="%p0"),
                        ),
                    ),
                ),
            )
        )
        text = emit(m)
        assert "\t@%p0 bra DONE;" in text

    def test_negated_predicate(self):
        m = _minimal_module(
            directives=(
                Function(
                    is_entry=True,
                    name="test",
                    body=(
                        Instruction(
                            opcode="mov",
                            modifiers=(".b32",),
                            operands=(
                                RegisterOperand("%r1"),
                                ImmediateOperand("0"),
                            ),
                            predicate=Predicate(register="%p0", negated=True),
                        ),
                    ),
                ),
            )
        )
        text = emit(m)
        assert "\t@!%p0 mov.b32 %r1, 0;" in text

    def test_ret(self):
        m = _minimal_module(
            directives=(
                Function(
                    is_entry=True,
                    name="test",
                    body=(Instruction(opcode="ret"),),
                ),
            )
        )
        text = emit(m)
        assert "\tret;" in text

    def test_address_operand_with_offset(self):
        m = _minimal_module(
            directives=(
                Function(
                    is_entry=True,
                    name="test",
                    body=(
                        Instruction(
                            opcode="ld",
                            modifiers=(".global", ".f32"),
                            operands=(
                                RegisterOperand("%r0"),
                                AddressOperand(base="%rd0", offset="16"),
                            ),
                        ),
                    ),
                ),
            )
        )
        text = emit(m)
        assert "\tld.global.f32 %r0, [%rd0+16];" in text

    def test_vector_operand(self):
        m = _minimal_module(
            directives=(
                Function(
                    is_entry=True,
                    name="test",
                    body=(
                        Instruction(
                            opcode="wgmma",
                            modifiers=(".mma_async", ".sync", ".aligned",
                                       ".m64n256k16", ".f32", ".e4m3", ".e4m3"),
                            operands=(
                                VectorOperand(elements=(
                                    RegisterOperand("%r0"),
                                    RegisterOperand("%r1"),
                                )),
                                RegisterOperand("%rd0"),
                                RegisterOperand("%rd1"),
                                ImmediateOperand("1"),
                                ImmediateOperand("1"),
                                ImmediateOperand("1"),
                            ),
                        ),
                    ),
                ),
            )
        )
        text = emit(m)
        assert "wgmma.mma_async.sync.aligned.m64n256k16.f32.e4m3.e4m3 {%r0, %r1}" in text

    def test_pipe_operand(self):
        m = _minimal_module(
            directives=(
                Function(
                    is_entry=True,
                    name="test",
                    body=(
                        Instruction(
                            opcode="setp",
                            modifiers=(".lt", ".f32"),
                            operands=(
                                PipeOperand(
                                    left=RegisterOperand("%p1"),
                                    right=RegisterOperand("%p2"),
                                ),
                                RegisterOperand("%r2"),
                                RegisterOperand("%r3"),
                            ),
                        ),
                    ),
                ),
            )
        )
        text = emit(m)
        assert "\tsetp.lt.f32 %p1|%p2, %r2, %r3;" in text


class TestEmitFunction:
    def test_entry_with_params(self):
        m = _minimal_module(
            directives=(
                Function(
                    is_entry=True,
                    name="my_kernel",
                    params=(
                        Param(
                            state_space=StateSpace.PARAM,
                            type=ScalarType.U64,
                            name="param0",
                        ),
                        Param(
                            state_space=StateSpace.PARAM,
                            type=ScalarType.U32,
                            name="param1",
                        ),
                    ),
                    body=(Instruction(opcode="ret"),),
                    linking=LinkingDirective.VISIBLE,
                ),
            )
        )
        text = emit(m)
        assert ".visible .entry my_kernel(" in text
        assert "\t.param .u64 param0," in text
        assert "\t.param .u32 param1" in text

    def test_label_in_body(self):
        m = _minimal_module(
            directives=(
                Function(
                    is_entry=True,
                    name="test",
                    body=(
                        Label(name="LOOP"),
                        Instruction(opcode="ret"),
                    ),
                ),
            )
        )
        text = emit(m)
        assert "LOOP:" in text


class TestEmitVarDecl:
    def test_shared_with_alignment(self):
        m = _minimal_module(
            directives=(
                Function(
                    is_entry=True,
                    name="test",
                    body=(
                        VarDecl(
                            state_space=StateSpace.SHARED,
                            type=ScalarType.B8,
                            name="smem",
                            array_size=49152,
                            alignment=128,
                        ),
                    ),
                ),
            )
        )
        text = emit(m)
        assert "\t.shared .align 128 .b8 smem[49152];" in text
