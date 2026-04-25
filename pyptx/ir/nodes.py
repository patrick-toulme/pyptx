"""Frozen dataclass IR nodes for lossless PTX representation.

Every node is immutable. Collections are tuples (hashable for frozen dataclasses).
The IR is a value type: equality and hashing work by structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from pyptx.ir.types import LinkingDirective, ScalarType, StateSpace


# ---------------------------------------------------------------------------
# Formatting metadata (for byte-identical round-trip)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormattingInfo:
    """Whitespace metadata attached to statements for round-trip fidelity.

    When present, the emitter reproduces this formatting exactly.
    When absent (None on the node), the emitter uses sensible defaults.
    """

    indent: str = ""
    trailing: str = ""
    blank_lines_before: int = 0
    preceding_comments: tuple[str, ...] = ()  # comment lines before this statement
    raw_line: str | None = None  # if set, emitter uses this verbatim instead of reconstructing


# ---------------------------------------------------------------------------
# Operand types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegisterOperand:
    """A register reference: %r0, %rd1, %p0, %tid.x."""

    name: str


@dataclass(frozen=True)
class ImmediateOperand:
    """A numeric literal, stored as raw text for lossless round-trip.

    Examples: '42', '0xFF', '0d3FF0000000000000', '1.0', '-1'
    """

    text: str


@dataclass(frozen=True)
class LabelOperand:
    """A label name used as an operand (branch target, call target)."""

    name: str


@dataclass(frozen=True)
class VectorOperand:
    """{%r0, %r1, %r2, %r3} — a vector of operands."""

    elements: tuple[Operand, ...]


@dataclass(frozen=True)
class AddressOperand:
    """[base], [base+offset], [base+-offset] — memory address expression.

    base is a register or symbol name (e.g. '%rd0', 'param0', 'smem').
    offset is the raw text of the offset (e.g. '16', '%rd1') or None.
    """

    base: str
    offset: str | None = None


@dataclass(frozen=True)
class ParenthesizedOperand:
    """(op1, op2, ...) — used in call instruction return/argument lists."""

    elements: tuple[Operand, ...]


@dataclass(frozen=True)
class NegatedOperand:
    """!operand — logical negation (used in setp, logical ops)."""

    operand: Operand


@dataclass(frozen=True)
class PipeOperand:
    """%p0|%p1 — dual predicate output in setp."""

    left: Operand
    right: Operand


# Union of all operand types.
Operand = Union[
    RegisterOperand,
    ImmediateOperand,
    LabelOperand,
    VectorOperand,
    AddressOperand,
    ParenthesizedOperand,
    NegatedOperand,
    PipeOperand,
]


# ---------------------------------------------------------------------------
# Predication
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Predicate:
    """@%p0 or @!%p0 — instruction predication."""

    register: str
    negated: bool = False


# ---------------------------------------------------------------------------
# Module header
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Version:
    """.version major.minor"""

    major: int
    minor: int


@dataclass(frozen=True)
class Target:
    """.target sm_90a[, feature, ...]"""

    targets: tuple[str, ...]


@dataclass(frozen=True)
class AddressSize:
    """.address_size 32 or .address_size 64"""

    size: int


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegDecl:
    """.reg .type name<count>; or .reg .type name;

    name is the base name (e.g. '%r' for '%r<100>', or '%rd1' for a single).
    count is None for a single register declaration.
    """

    type: ScalarType
    name: str
    count: int | None = None
    formatting: FormattingInfo | None = None


@dataclass(frozen=True)
class VarDecl:
    """Variable declaration: .shared, .global, .local, .const.

    Covers both function-body and module-level declarations.
    """

    state_space: StateSpace
    type: ScalarType
    name: str
    array_size: int | None = None
    alignment: int | None = None
    initializer: tuple[str, ...] | None = None
    linking: LinkingDirective | None = None
    formatting: FormattingInfo | None = None


# ---------------------------------------------------------------------------
# Statements (inside function body)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Instruction:
    """A single PTX instruction with optional predication.

    opcode: the base opcode (e.g. 'mov', 'ld', 'wgmma', 'bra')
    modifiers: dot-prefixed modifiers in order (e.g. ('.b32',) or
               ('.global', '.nc', '.b32') or ('.mma_async', '.sync', ...))
    operands: destination(s) first, then source(s)
    """

    opcode: str
    modifiers: tuple[str, ...] = ()
    operands: tuple[Operand, ...] = ()
    predicate: Predicate | None = None
    formatting: FormattingInfo | None = None


@dataclass(frozen=True)
class Label:
    """label_name: — a branch target label."""

    name: str
    formatting: FormattingInfo | None = None


@dataclass(frozen=True)
class PragmaDirective:
    """.pragma "string";"""

    value: str
    formatting: FormattingInfo | None = None


@dataclass(frozen=True)
class Comment:
    """A comment line (// ... or /* ... */).

    Stored as a first-class IR node so the emitter can reproduce it.
    """

    text: str  # full comment text including // or /* */ delimiters


@dataclass(frozen=True)
class BlankLine:
    """An empty line in the source. Preserved for formatting fidelity."""
    pass


@dataclass(frozen=True)
class RawLine:
    """A line the parser couldn't structurally parse.

    This is the escape hatch: instead of crashing, the parser captures
    the raw text. The emitter emits it verbatim.
    """

    text: str


@dataclass(frozen=True)
class Block:
    """A nested { } scope block inside a function body.

    PTX allows arbitrary scoping for register lifetime management.
    """

    body: tuple["Statement", ...]
    formatting: FormattingInfo | None = None


@dataclass(frozen=True)
class IntrinsicScope:
    """A named scope created by @ptx.intrinsic.

    Wraps a sequence of statements emitted by an intrinsic function call.
    The emitter renders this as PTX comments (BEGIN/END markers) so the
    emitted PTX is unchanged in semantics, but inspection tools can see
    which high-level intrinsic produced which instructions.

    Intrinsic scopes are construction-time only — they're created when a
    @ptx.intrinsic-decorated function is called inside a kernel trace.
    The parser never produces them (it just sees comments).
    """

    name: str
    args_repr: str  # repr of the args passed to the intrinsic
    body: tuple["Statement", ...]
    formatting: FormattingInfo | None = None


# ---------------------------------------------------------------------------
# Function-level
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Param:
    """A single parameter in a function signature.

    For .entry params: state_space is always PARAM.
    For .func params with .ptr: ptr_state_space indicates the pointer's space.
    """

    state_space: StateSpace
    type: ScalarType
    name: str
    array_size: int | None = None
    alignment: int | None = None
    ptr_state_space: StateSpace | None = None
    ptr_alignment: int | None = None


@dataclass(frozen=True)
class FunctionDirective:
    """Performance hint directive on a function (.maxnreg, .maxntid, etc.)."""

    name: str
    values: tuple[int | str, ...]


# Statement union: anything that can appear in a function body.
Statement = Union[
    Instruction, Label, RegDecl, VarDecl, PragmaDirective,
    Comment, BlankLine, RawLine, Block, IntrinsicScope,
]

# Directive union: anything at module level.
Directive = Union["Function", VarDecl, PragmaDirective, Comment, BlankLine, RawLine]


@dataclass(frozen=True)
class Function:
    """.func or .entry definition."""

    is_entry: bool
    name: str
    params: tuple[Param, ...] = ()
    return_params: tuple[Param, ...] | None = None
    body: tuple[Statement, ...] = ()
    linking: LinkingDirective | None = None
    directives: tuple[FunctionDirective, ...] = ()
    formatting: FormattingInfo | None = None


# ---------------------------------------------------------------------------
# Module (root node)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Module:
    """Root node representing an entire .ptx file."""

    version: Version
    target: Target
    address_size: AddressSize
    directives: tuple[Directive, ...] = ()
    raw_header: str | None = None  # original source for header lines
    raw_source: str | None = None  # complete original source for lossless round-trip
