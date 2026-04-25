"""IR normalizer: strip formatting noise for structural comparison.

Usage:
    from pyptx.ir.normalize import normalize_module
    canon = normalize_module(module)

Normalization strips:
- Comment nodes
- BlankLine nodes
- FormattingInfo (indent, trailing, raw_line, preceding_comments)
- raw_source / raw_header on Module

What's preserved (the semantic content):
- All Instructions (opcode, modifiers, operands, predicate)
- All RegDecls (type, name, count)
- All VarDecls (state_space, type, name, array_size, alignment)
- All Labels (name)
- All Functions (is_entry, name, params, return_params, body, linking)
- All Blocks (body)
- Module header (version, target, address_size)
"""

from __future__ import annotations

from pyptx.ir.nodes import (
    AddressOperand,
    AddressSize,
    BlankLine,
    Block,
    Comment,
    Function,
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
    PragmaDirective,
    Predicate,
    RawLine,
    RegDecl,
    RegisterOperand,
    Statement,
    Target,
    VarDecl,
    VectorOperand,
    Version,
)


def normalize_module(module: Module) -> Module:
    """Strip all formatting metadata, return a canonical IR for comparison."""
    return Module(
        version=module.version,
        target=module.target,
        address_size=module.address_size,
        directives=tuple(_normalize_directives(module.directives)),
    )


def _normalize_directives(directives):
    result = []
    for d in directives:
        if isinstance(d, (Comment, BlankLine, RawLine)):
            continue
        if isinstance(d, Function):
            result.append(_normalize_function(d))
        elif isinstance(d, VarDecl):
            result.append(_normalize_var_decl(d))
        elif isinstance(d, PragmaDirective):
            result.append(PragmaDirective(value=d.value))
        else:
            result.append(d)
    return result


def _normalize_function(f: Function) -> Function:
    return Function(
        is_entry=f.is_entry,
        name=f.name,
        params=f.params,
        return_params=f.return_params,
        body=tuple(_normalize_body(f.body)),
        linking=f.linking,
        directives=f.directives,
    )


def _normalize_body(body) -> list:
    result = []
    for s in body:
        if isinstance(s, (Comment, BlankLine)):
            continue
        if isinstance(s, RawLine):
            # Keep RawLines — they represent real instructions the parser couldn't model
            result.append(s)
            continue
        if isinstance(s, Instruction):
            result.append(_normalize_instruction(s))
        elif isinstance(s, RegDecl):
            result.append(RegDecl(type=s.type, name=s.name, count=s.count))
        elif isinstance(s, VarDecl):
            result.append(_normalize_var_decl(s))
        elif isinstance(s, Label):
            result.append(Label(name=s.name))
        elif isinstance(s, Block):
            # Flatten nested blocks — the { } scope is lost but the
            # instruction sequence is preserved for semantic comparison
            result.extend(_normalize_body(s.body))
        elif isinstance(s, IntrinsicScope):
            # Flatten intrinsic scopes — they're annotations, not semantic content
            result.extend(_normalize_body(s.body))
        elif isinstance(s, PragmaDirective):
            result.append(PragmaDirective(value=s.value))
        else:
            result.append(s)
    return result


def _normalize_instruction(inst: Instruction) -> Instruction:
    return Instruction(
        opcode=inst.opcode,
        modifiers=inst.modifiers,
        operands=inst.operands,
        predicate=inst.predicate,
    )


def _normalize_var_decl(vd: VarDecl) -> VarDecl:
    return VarDecl(
        state_space=vd.state_space,
        type=vd.type,
        name=vd.name,
        array_size=vd.array_size,
        alignment=vd.alignment,
        initializer=vd.initializer,
        linking=vd.linking,
    )


def diff_modules(a: Module, b: Module, entry_only: bool = False) -> list[str]:
    """Compare two normalized modules and return a list of differences.

    Args:
        entry_only: If True, only compare .entry functions (ignore .func helpers).

    Returns empty list if they're identical.
    """
    diffs: list[str] = []

    if a.version != b.version:
        diffs.append(f"version: {a.version} vs {b.version}")
    if a.target != b.target:
        diffs.append(f"target: {a.target} vs {b.target}")
    if a.address_size != b.address_size:
        diffs.append(f"address_size: {a.address_size} vs {b.address_size}")

    a_funcs = [d for d in a.directives if isinstance(d, Function)]
    b_funcs = [d for d in b.directives if isinstance(d, Function)]

    if entry_only:
        a_funcs = [f for f in a_funcs if f.is_entry]
        b_funcs = [f for f in b_funcs if f.is_entry]

    if len(a_funcs) != len(b_funcs):
        diffs.append(f"function count: {len(a_funcs)} vs {len(b_funcs)}")
        return diffs

    for i, (af, bf) in enumerate(zip(a_funcs, b_funcs)):
        prefix = f"func[{i}] ({af.name})"
        if af.name != bf.name:
            diffs.append(f"{prefix} name: {af.name!r} vs {bf.name!r}")
        if af.is_entry != bf.is_entry:
            diffs.append(f"{prefix} is_entry: {af.is_entry} vs {bf.is_entry}")

        # Compare body (skip Comment/BlankLine)
        a_stmts = [s for s in af.body if not isinstance(s, (Comment, BlankLine))]
        b_stmts = [s for s in bf.body if not isinstance(s, (Comment, BlankLine))]

        if len(a_stmts) != len(b_stmts):
            diffs.append(f"{prefix} body length: {len(a_stmts)} vs {len(b_stmts)}")
            # Show first few of each
            for j, s in enumerate(a_stmts[:5]):
                diffs.append(f"  a[{j}]: {_stmt_summary(s)}")
            for j, s in enumerate(b_stmts[:5]):
                diffs.append(f"  b[{j}]: {_stmt_summary(s)}")
            continue

        for j, (sa, sb) in enumerate(zip(a_stmts, b_stmts)):
            if type(sa) != type(sb):
                diffs.append(f"{prefix} body[{j}] type: {type(sa).__name__} vs {type(sb).__name__}")
                continue
            if isinstance(sa, Instruction) and isinstance(sb, Instruction):
                if sa.opcode != sb.opcode:
                    diffs.append(f"{prefix} body[{j}] opcode: {sa.opcode!r} vs {sb.opcode!r}")
                if sa.modifiers != sb.modifiers:
                    diffs.append(f"{prefix} body[{j}] mods: {sa.modifiers} vs {sb.modifiers}")
                if sa.operands != sb.operands:
                    diffs.append(f"{prefix} body[{j}] operands differ")
                    diffs.append(f"  a: {sa.operands}")
                    diffs.append(f"  b: {sb.operands}")
                if sa.predicate != sb.predicate:
                    diffs.append(f"{prefix} body[{j}] pred: {sa.predicate} vs {sb.predicate}")
            elif sa != sb:
                diffs.append(f"{prefix} body[{j}]: {_stmt_summary(sa)} vs {_stmt_summary(sb)}")

    return diffs


def _stmt_summary(s) -> str:
    match s:
        case Instruction(opcode=op, modifiers=mods):
            return f"{op}{''.join(mods)}"
        case RegDecl(type=t, name=n, count=c):
            return f".reg {t.ptx} {n}<{c}>"
        case VarDecl(name=n):
            return f"VarDecl({n})"
        case Label(name=n):
            return f"{n}:"
        case _:
            return repr(s)
