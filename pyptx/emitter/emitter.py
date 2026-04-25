"""PTX emitter: IR → source text.

Produces PTX text from IR nodes. When FormattingInfo is present on a node,
the emitter reproduces that formatting exactly (for byte-identical round-trip).
When absent, it uses sensible defaults.
"""

from __future__ import annotations

from pyptx.ir.nodes import (
    AddressOperand,
    AddressSize,
    BlankLine,
    Block,
    Comment,
    Directive,
    Function,
    FunctionDirective,
    ImmediateOperand,
    Instruction,
    IntrinsicScope,
    Label,
    LabelOperand,
    Module,
    NegatedOperand,
    Operand,
    ParenthesizedOperand,
    Param,
    PipeOperand,
    Predicate,
    PragmaDirective,
    RawLine,
    RegDecl,
    RegisterOperand,
    Statement,
    Target,
    VarDecl,
    VectorOperand,
    Version,
)
from pyptx.ir.types import LinkingDirective


def emit(module: Module) -> str:
    """Emit a complete PTX module as text.

    If the module was parsed from source and hasn't been modified,
    raw_source provides lossless round-trip. For programmatically
    constructed modules, the emitter reconstructs from the IR.
    """
    # Lossless round-trip: if raw_source is available, use it directly
    if module.raw_source is not None:
        return module.raw_source

    parts: list[str] = []

    # Module header (version, target, address_size)
    if module.raw_header is not None:
        parts.append(module.raw_header)
    else:
        parts.append(_emit_version(module.version))
        parts.append(_emit_target(module.target))
        parts.append(_emit_address_size(module.address_size))

    for directive in module.directives:
        parts.append(_emit_directive(directive))

    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Module header
# ---------------------------------------------------------------------------

def _emit_version(v: Version) -> str:
    return f".version {v.major}.{v.minor}"


def _emit_target(t: Target) -> str:
    return f".target {', '.join(t.targets)}"


def _emit_address_size(a: AddressSize) -> str:
    return f".address_size {a.size}"


# ---------------------------------------------------------------------------
# Top-level directives
# ---------------------------------------------------------------------------

def _emit_directive(d: Directive) -> str:
    match d:
        case Function():
            return _emit_function(d)
        case VarDecl():
            return _emit_var_decl_toplevel(d)
        case PragmaDirective():
            return _emit_pragma(d, indent="")
        case Comment(text=text):
            return text
        case BlankLine():
            return ""
        case RawLine(text=text):
            return text
        case _:
            raise TypeError(f"Unknown directive type: {type(d)}")


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def _emit_function(f: Function) -> str:
    parts: list[str] = []

    # Preceding comments (block comments between functions)
    if f.formatting and f.formatting.preceding_comments:
        for comment in f.formatting.preceding_comments:
            parts.append(comment)
    else:
        # Default: blank line before function
        parts.append("")

    # Header line: [.linking] .entry/.func name(params)
    header = ""
    if f.linking is not None:
        header += f"{f.linking.ptx} "

    # Return params for .func
    if f.return_params is not None:
        ret_params = ", ".join(_emit_param(p) for p in f.return_params)
        header += f".func ({ret_params}) {f.name}"
    elif f.is_entry:
        header += f".entry {f.name}"
    else:
        header += f".func {f.name}"

    # Parameters
    if f.params:
        header += "(\n"
        param_lines = []
        for i, p in enumerate(f.params):
            suffix = "," if i < len(f.params) - 1 else ""
            param_lines.append(f"\t{_emit_param(p)}{suffix}")
        header += "\n".join(param_lines)
        header += "\n)"
    else:
        header += "()"

    parts.append(header)

    # Function directives (.maxnreg, .maxntid, etc.)
    for fd in f.directives:
        parts.append(_emit_function_directive(fd))

    # Body
    parts.append("{")
    for stmt in f.body:
        parts.append(_emit_statement(stmt))
    parts.append("}")

    return "\n".join(parts)


def _emit_param(p: Param) -> str:
    parts: list[str] = []
    parts.append(p.state_space.ptx)

    if p.alignment is not None:
        parts.append(f".align {p.alignment}")

    parts.append(p.type.ptx)

    if p.ptr_state_space is not None:
        parts.append(".ptr")
        parts.append(p.ptr_state_space.ptx)
        if p.ptr_alignment is not None:
            parts.append(f".align {p.ptr_alignment}")

    if p.array_size is not None:
        parts.append(f"{p.name}[{p.array_size}]")
    else:
        parts.append(p.name)

    return " ".join(parts)


def _emit_function_directive(fd: FunctionDirective) -> str:
    values = ", ".join(str(v) for v in fd.values)
    return f".{fd.name} {values}"


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

def _format_prefix(formatting) -> str:
    """Emit preceding comments and blank lines before a statement."""
    if formatting is None:
        return ""
    parts: list[str] = []
    for comment in formatting.preceding_comments:
        parts.append(comment)
        parts.append("\n")
    return "".join(parts)


def _emit_statement(s: Statement) -> str:
    # Check for raw_line fallback
    fmt = getattr(s, "formatting", None)
    if fmt and fmt.raw_line is not None:
        prefix = _format_prefix(fmt)
        return prefix + fmt.raw_line

    match s:
        case Instruction():
            return _emit_instruction(s)
        case Label():
            return _emit_label(s)
        case RegDecl():
            return _emit_reg_decl(s)
        case VarDecl():
            return _emit_var_decl(s)
        case PragmaDirective():
            return _emit_pragma(s, indent="\t")
        case Comment(text=text):
            return text
        case BlankLine():
            return ""
        case RawLine(text=text):
            return text
        case Block(body=body, formatting=bfmt):
            indent = bfmt.indent if bfmt else "    "
            lines = [f"{indent}{{"]
            for inner in body:
                lines.append(_emit_statement(inner))
            lines.append(f"{indent}}}")
            return "\n".join(lines)
        case IntrinsicScope(name=name, args_repr=args, body=body, formatting=ifmt):
            indent = ifmt.indent if ifmt else "\t"
            lines = [f"{indent}// BEGIN {name}({args})"]
            for inner in body:
                lines.append(_emit_statement(inner))
            lines.append(f"{indent}// END {name}")
            return "\n".join(lines)
        case _:
            # CompoundExpr or other extension nodes
            if hasattr(s, "instructions"):
                return "\n".join(_emit_instruction(inst) for inst in s.instructions)
            raise TypeError(f"Unknown statement type: {type(s)}")


def _emit_instruction(inst: Instruction) -> str:
    indent = "\t"
    if inst.formatting:
        indent = inst.formatting.indent

    parts: list[str] = []

    # Predicate
    if inst.predicate is not None:
        parts.append(_emit_predicate(inst.predicate))

    # Opcode + modifiers
    opcode_str = inst.opcode + "".join(inst.modifiers)
    parts.append(opcode_str)

    # Operands
    if inst.operands:
        operand_strs = []
        for op in inst.operands:
            operand_strs.append(_emit_operand(op))
        parts.append(" " + ", ".join(operand_strs))

    # Join: indent + predicate + space + opcode + space + operands + ;
    if inst.predicate is not None:
        line = f"{indent}{parts[0]} {parts[1]}"
        if len(parts) > 2:
            line += parts[2]
    else:
        line = f"{indent}{parts[0]}"
        if len(parts) > 1:
            line += parts[1]

    trailing = ";"
    if inst.formatting and inst.formatting.trailing:
        trailing = inst.formatting.trailing

    return line + trailing


def _emit_predicate(p: Predicate) -> str:
    neg = "!" if p.negated else ""
    return f"@{neg}{p.register}"


def _emit_label(label: Label) -> str:
    indent = ""
    if label.formatting:
        indent = label.formatting.indent
    return f"{indent}{label.name}:"


def _emit_reg_decl(rd: RegDecl) -> str:
    indent = "\t"
    if rd.formatting:
        indent = rd.formatting.indent

    if rd.count is not None:
        return f"{indent}.reg {rd.type.ptx} {rd.name}<{rd.count}>;"
    else:
        return f"{indent}.reg {rd.type.ptx} {rd.name};"


def _emit_var_decl(vd: VarDecl) -> str:
    indent = "\t"
    if vd.formatting:
        indent = vd.formatting.indent
    body = _emit_var_decl_body(vd)
    if vd.linking is not None:
        return f"{indent}{vd.linking.ptx} {body}"
    return indent + body


def _emit_var_decl_toplevel(vd: VarDecl) -> str:
    # Use raw_line if available for lossless round-trip
    if vd.formatting and vd.formatting.raw_line is not None:
        prefix = _format_prefix(vd.formatting)
        return prefix + vd.formatting.raw_line

    prefix = ""
    if vd.formatting and vd.formatting.blank_lines_before > 0:
        prefix = "\n" * vd.formatting.blank_lines_before
    indent = ""
    if vd.formatting:
        indent = vd.formatting.indent
    if vd.linking is not None:
        return f"{prefix}{indent}{vd.linking.ptx} {_emit_var_decl_body(vd)}"
    return f"{prefix}{indent}{_emit_var_decl_body(vd)}"


def _emit_var_decl_body(vd: VarDecl) -> str:
    parts: list[str] = []
    parts.append(vd.state_space.ptx)

    if vd.alignment is not None:
        parts.append(f".align {vd.alignment}")

    parts.append(vd.type.ptx)

    if vd.array_size is not None:
        parts.append(f"{vd.name}[{vd.array_size}]")
    elif vd.linking is not None and vd.linking.value == "extern":
        # Extern arrays: ``extern .shared .b8 name[];``
        parts.append(f"{vd.name}[]")
    else:
        parts.append(vd.name)

    result = " ".join(parts) + ";"

    if vd.initializer is not None:
        pass

    return result


def _emit_pragma(p: PragmaDirective, indent: str) -> str:
    if p.formatting:
        indent = p.formatting.indent
    return f'{indent}.pragma "{p.value}";'


# ---------------------------------------------------------------------------
# Operands
# ---------------------------------------------------------------------------

def _emit_operand(op: Operand) -> str:
    match op:
        case RegisterOperand(name=name):
            return name
        case ImmediateOperand(text=text):
            return text
        case LabelOperand(name=name):
            return name
        case VectorOperand(elements=elems):
            inner = ", ".join(_emit_operand(e) for e in elems)
            return "{" + inner + "}"
        case AddressOperand(base=base, offset=offset):
            if offset is not None:
                # Offset contains its own leading operator (+, -, *, ,, etc.)
                # If it starts with a digit or identifier, prepend +
                stripped = offset.lstrip()
                if stripped and (stripped[0].isalnum() or stripped[0] == '_' or stripped[0] == '%'):
                    return f"[{base}+{offset}]"
                return f"[{base}{offset}]"
            return f"[{base}]"
        case ParenthesizedOperand(elements=elems):
            inner = ", ".join(_emit_operand(e) for e in elems)
            return f"({inner})"
        case NegatedOperand(operand=inner):
            return f"!{_emit_operand(inner)}"
        case PipeOperand(left=left, right=right):
            return f"{_emit_operand(left)}|{_emit_operand(right)}"
        case _:
            raise TypeError(f"Unknown operand type: {type(op)}")
