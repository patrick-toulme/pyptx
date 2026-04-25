"""Sugar pass: rewrite IR for cleaner codegen output.

Runs on the parsed Module IR before codegen to Python. Rewrites:
  1. Demangle C++ symbol names to short readable names
  2. Shorten label names ($L__BB13_2 → BB2)
  3. Clean up extern shared memory symbol names
  4. Rename kernel function to a clean name

This is a Module → Module transformation (returns a new frozen Module).

Usage:
    from pyptx.codegen.sugar import apply_sugar
    module = parse(ptx_source)
    module = apply_sugar(module)  # rewrite in place
    python_code = ir_to_python(module)
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Sequence

from pyptx.ir.nodes import (
    AddressOperand,
    AddressSize,
    BlankLine,
    Block,
    Comment,
    Function,
    FunctionDirective,
    ImmediateOperand,
    Instruction,
    Label,
    LabelOperand,
    Module,
    NegatedOperand,
    Operand,
    Param,
    ParenthesizedOperand,
    PipeOperand,
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
from pyptx.ir.types import LinkingDirective, ScalarType, StateSpace


def _find_loops(body: tuple[Statement, ...]) -> dict[str, str]:
    """Find loop back-edges: labels that are targets of backward branches.

    Returns a dict mapping label_name → annotation string.
    """
    from pyptx.ir.nodes import Instruction, Label
    annotations: dict[str, str] = {}

    # Collect label positions
    label_positions: dict[str, int] = {}
    for i, stmt in enumerate(body):
        if isinstance(stmt, Label):
            label_positions[stmt.name] = i

    # Find backward branches (bra to a label that appeared earlier)
    for i, stmt in enumerate(body):
        if isinstance(stmt, Instruction) and stmt.opcode == "bra":
            for op in stmt.operands:
                if isinstance(op, LabelOperand):
                    target = op.name
                    if target in label_positions and label_positions[target] < i:
                        # Backward branch = loop. Try to identify the counter.
                        # Look at the setp right before the bra
                        loop_info = _analyze_loop(body, label_positions[target], i)
                        annotations[target] = loop_info
    return annotations


def _analyze_loop(body: tuple[Statement, ...], start: int, end: int) -> str:
    """Try to extract loop counter info from the loop body."""
    from pyptx.ir.nodes import Instruction
    # Look for setp near the end (the loop condition)
    n_instrs = sum(1 for s in body[start:end] if isinstance(s, Instruction))
    # Look for sub/add of a counter near the bra
    for i in range(max(start, end - 10), end):
        s = body[i]
        if isinstance(s, Instruction) and s.opcode == "sub" and ".s32" in s.modifiers:
            # Likely loop counter decrement
            return f"# === LOOP ({n_instrs} instructions, counts down) ==="
    # Check for unconditional backward branch (persistent tile loop)
    s = body[end]
    if isinstance(s, Instruction) and s.opcode == "bra" and s.predicate is None:
        return f"# === PERSISTENT TILE LOOP ({n_instrs} instructions) ==="
    return f"# === LOOP ({n_instrs} instructions) ==="


def _find_warp_split(body: tuple[Statement, ...]) -> dict[str, str]:
    """Find warp specialization split points (setmaxnreg).

    Returns label annotations for producer/consumer sections.
    """
    from pyptx.ir.nodes import Instruction, Label
    annotations: dict[str, str] = {}
    for i, stmt in enumerate(body):
        if isinstance(stmt, Instruction) and stmt.opcode == "setmaxnreg":
            if ".inc" in stmt.modifiers:
                # Consumer path (increases register count)
                annotations[f"_after_setmaxnreg_inc_{i}"] = "# --- CONSUMER WARP GROUP (increased registers) ---"
            elif ".dec" in stmt.modifiers:
                annotations[f"_after_setmaxnreg_dec_{i}"] = "# --- PRODUCER WARP GROUP (reduced registers) ---"
    return annotations


def apply_sugar(module: Module, *, kernel_name: str = "matmul_kernel") -> Module:
    """Apply sugar rewrites to a parsed Module.

    Returns a new Module with cleaned-up names and simplified labels.
    The emitted PTX from the sugared module is semantically identical
    (same instructions, same register assignments) — only symbol names
    and labels differ.
    """
    # Step 1: Build a renaming table for all symbols in the module
    renames: dict[str, str] = {}

    # Collect all names that need renaming
    for d in module.directives:
        if isinstance(d, Function):
            # Demangle the kernel function name
            renames[d.name] = kernel_name

            # Rename params: detect C++ mangled names and assign clean ones
            _rename_params(d.params, d.name, renames)

            # Rename labels
            _rename_labels(d.body, renames)

        elif isinstance(d, VarDecl):
            # Clean up extern shared memory names
            if d.state_space == StateSpace.SHARED and d.linking == LinkingDirective.EXTERN:
                clean = _demangle_smem(d.name)
                if clean != d.name:
                    renames[d.name] = clean
                    # Also rename without [] suffix (instructions reference the bare name)
                    bare = d.name.rstrip("[]")
                    if bare != d.name:
                        renames[bare] = clean

    # Step 2: Apply renames throughout the module
    new_directives = tuple(_rewrite_directive(d, renames) for d in module.directives)
    return replace(module, directives=new_directives)


# ---------------------------------------------------------------------------
# Name demangling
# ---------------------------------------------------------------------------

def _demangle_symbol(name: str) -> str:
    """Demangle a C++ mangled symbol name.

    _ZN3M124smemE → smem (namespace M12, identifier smem)
    _ZN3M1214matmulKernel12ILi128E...E → matmulKernel12
    """
    clean = name.rstrip("[]")
    if not clean.startswith("_ZN"):
        if "global_smem" in clean:
            return "smem"
        return clean

    # Parse itanium ABI name mangling: _ZN (length name)* E
    # Each component is a decimal length followed by that many chars
    pos = 3  # skip _ZN
    components: list[str] = []
    while pos < len(clean) and clean[pos] != "E":
        # Read length
        num_start = pos
        while pos < len(clean) and clean[pos].isdigit():
            pos += 1
        if pos == num_start:
            break  # not a length-prefixed component
        length = int(clean[num_start:pos])
        if pos + length > len(clean):
            break
        ident = clean[pos:pos + length]
        components.append(ident)
        pos += length

    if components:
        return components[-1]  # last component is the actual name
    return clean


def _demangle_smem(name: str) -> str:
    """Demangle a shared memory symbol name."""
    return _demangle_symbol(name)


_PARAM_HEURISTICS = {
    # Map param index → likely name based on common CUDA kernel conventions
    # For matmul: (M, N, K, tma_C, tma_A, tma_B, hilbert_ptr)
}


def _rename_params(
    params: tuple[Param, ...],
    func_name: str,
    renames: dict[str, str],
) -> None:
    """Assign clean names to kernel parameters."""
    # Strategy: use param type to guess purpose
    u32_idx = 0
    tma_idx = 0
    ptr_idx = 0
    u32_names = ["M", "N", "K", "dim3", "dim4", "dim5"]
    tma_names = ["tma_C", "tma_A", "tma_B", "tma_3", "tma_4", "tma_5"]

    for p in params:
        old_name = p.name
        if old_name in renames:
            continue

        if p.type.value == "u32" and p.array_size is None:
            new_name = u32_names[u32_idx] if u32_idx < len(u32_names) else f"param_u32_{u32_idx}"
            u32_idx += 1
        elif p.array_size is not None and p.array_size == 128:
            # 128-byte param = CUtensorMap by value
            new_name = tma_names[tma_idx] if tma_idx < len(tma_names) else f"tma_{tma_idx}"
            tma_idx += 1
        elif p.type.value == "u64":
            new_name = "hilbert_ptr" if ptr_idx == 0 else f"ptr_{ptr_idx}"
            ptr_idx += 1
        elif p.ptr_state_space is not None:
            new_name = f"buf_{ptr_idx}"
            ptr_idx += 1
        else:
            new_name = f"param_{old_name.split('_')[-1]}" if "_param_" in old_name else old_name
            # Skip if we can't improve
            continue

        renames[old_name] = new_name


def _rename_labels(body: tuple[Statement, ...], renames: dict[str, str]) -> None:
    """Assign short label names."""
    label_counter = 0
    for stmt in body:
        if isinstance(stmt, Label):
            old = stmt.name
            if old not in renames:
                # $L__BB13_2 → BB2, $L__BB13_4 → BB4
                m = re.search(r'_(\d+)$', old)
                if m:
                    renames[old] = f"BB{m.group(1)}"
                else:
                    renames[old] = f"L{label_counter}"
                label_counter += 1
        elif isinstance(stmt, Block):
            _rename_labels(stmt.body, renames)


# ---------------------------------------------------------------------------
# IR rewriting
# ---------------------------------------------------------------------------

def _rewrite_directive(d, renames: dict[str, str]):
    """Rewrite a module-level directive with the rename table."""
    if isinstance(d, Function):
        new_name = renames.get(d.name, d.name)
        new_params = tuple(_rewrite_param(p, renames) for p in d.params)
        new_body = tuple(_rewrite_stmt(s, renames) for s in d.body)
        return replace(d, name=new_name, params=new_params, body=new_body)
    elif isinstance(d, VarDecl):
        new_name = renames.get(d.name, d.name)
        # Also rewrite name in formatting if present
        fmt = d.formatting
        if fmt and fmt.raw_line and d.name in fmt.raw_line:
            new_raw = fmt.raw_line.replace(d.name, new_name)
            fmt = replace(fmt, raw_line=new_raw)
        return replace(d, name=new_name, formatting=fmt)
    elif isinstance(d, (Comment, BlankLine, RawLine)):
        return d
    return d


def _rewrite_param(p: Param, renames: dict[str, str]) -> Param:
    new_name = renames.get(p.name, p.name)
    return replace(p, name=new_name)


def _rewrite_stmt(s: Statement, renames: dict[str, str]) -> Statement:
    if isinstance(s, Instruction):
        new_ops = tuple(_rewrite_operand(op, renames) for op in s.operands)
        # Also rewrite predicate register names (though these are typically
        # short already: %p0, %p1, etc.)
        new_pred = s.predicate
        fmt = s.formatting
        if fmt and fmt.raw_line:
            new_raw = _apply_renames_to_text(fmt.raw_line, renames)
            fmt = replace(fmt, raw_line=new_raw)
        return replace(s, operands=new_ops, formatting=fmt)
    elif isinstance(s, Label):
        new_name = renames.get(s.name, s.name)
        fmt = s.formatting
        if fmt and fmt.raw_line:
            new_raw = fmt.raw_line.replace(s.name, new_name)
            fmt = replace(fmt, raw_line=new_raw)
        return replace(s, name=new_name, formatting=fmt)
    elif isinstance(s, Block):
        new_body = tuple(_rewrite_stmt(inner, renames) for inner in s.body)
        return replace(s, body=new_body)
    elif isinstance(s, (RegDecl, VarDecl, PragmaDirective, Comment, BlankLine)):
        return s
    elif isinstance(s, RawLine):
        if s.text and any(old in s.text for old in renames):
            new_text = _apply_renames_to_text(s.text, renames)
            return replace(s, text=new_text)
        return s
    return s


def _rewrite_operand(op: Operand, renames: dict[str, str]) -> Operand:
    if isinstance(op, LabelOperand):
        new_name = renames.get(op.name, op.name)
        return LabelOperand(name=new_name)
    elif isinstance(op, RegisterOperand):
        # Register names (%r0, %p1) don't get renamed — only symbols
        new_name = renames.get(op.name, op.name)
        return RegisterOperand(name=new_name)
    elif isinstance(op, AddressOperand):
        new_base = renames.get(op.base, op.base)
        return AddressOperand(base=new_base, offset=op.offset)
    elif isinstance(op, VectorOperand):
        new_elems = tuple(_rewrite_operand(e, renames) for e in op.elements)
        return VectorOperand(elements=new_elems)
    elif isinstance(op, NegatedOperand):
        return NegatedOperand(operand=_rewrite_operand(op.operand, renames))
    elif isinstance(op, PipeOperand):
        return PipeOperand(
            left=_rewrite_operand(op.left, renames),
            right=_rewrite_operand(op.right, renames),
        )
    elif isinstance(op, ParenthesizedOperand):
        new_elems = tuple(_rewrite_operand(e, renames) for e in op.elements)
        return ParenthesizedOperand(elements=new_elems)
    elif isinstance(op, ImmediateOperand):
        # Immediates can contain symbol references as strings
        new_text = renames.get(op.text, op.text)
        return ImmediateOperand(text=new_text)
    return op


def _apply_renames_to_text(text: str, renames: dict[str, str]) -> str:
    """Apply all renames to a raw text string (for formatting raw_lines)."""
    # Sort by length descending to avoid partial matches
    for old, new in sorted(renames.items(), key=lambda x: -len(x[0])):
        if old in text:
            text = text.replace(old, new)
    return text
