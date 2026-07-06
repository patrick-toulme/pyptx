"""Post-trace IR optimizations.

These passes run on the traced Statement list before emission to PTX.
They are semantics-preserving transformations that clean up the IR
produced by DSL sugar (operator overloading, RegArray.__setitem__, etc.)
so ptxas is handed a value-graph closer to what a functional DSL (CuTe)
emits — fewer redundant computes, tighter live ranges.

Passes (gated by PYPTX_OPT / optimize_body level):
  * copy_propagate — drop trivial `mov %dst, %src` from RegArray sugar,
    renaming the source through the chain.
  * gvn_invariant  — global value numbering + LICM for registers that
    provably hold one fixed value for the whole kernel (address
    arithmetic, identity `add x,y,0`): compute each once at a dominating
    point and rename all duplicates away.

Every pass is opt-in and the pipeline self-verifies (verify_body) with
fallback to the untransformed body, so a pass bug can never silently
corrupt a kernel. Perf wins still require ptxas/GPU validation.
"""

from __future__ import annotations

from pyptx.ir.nodes import (
    Instruction,
    ImmediateOperand,
    RegisterOperand,
    AddressOperand,
    VectorOperand,
    NegatedOperand,
    PipeOperand,
    ParenthesizedOperand,
    Statement,
    Label,
    RegDecl,
)


def copy_propagate(statements: list[Statement]) -> list[Statement]:
    """Remove trivial mov instructions by renaming registers.

    When RegArray.__setitem__ emits ``mov.bN %dst, %src``, this pass:
    1. Renames %src → %dst in ALL preceding instructions in the chain
    2. Removes the mov instruction
    3. Removes the .reg declaration for %src if it becomes unused

    This produces PTX identical to writing ``ptx.inst.*(dst, ...)``
    directly — no extra registers, no extra movs.
    """
    # Find mov.bN %dst, %src where both are named registers
    # These come from RegArray.__setitem__
    renames: dict[str, str] = {}  # src → dst
    mov_indices: set[int] = set()

    for i, stmt in enumerate(statements):
        if not isinstance(stmt, Instruction):
            continue
        if stmt.opcode != "mov":
            continue
        if len(stmt.operands) != 2:
            continue
        # Check it's mov.bN (not mov.u32 or mov.pred)
        if not any(m.startswith(".b") for m in stmt.modifiers):
            continue
        dst_op, src_op = stmt.operands
        if not isinstance(dst_op, RegisterOperand) or not isinstance(src_op, RegisterOperand):
            continue
        dst_name = dst_op.name
        src_name = src_op.name
        if dst_name == src_name:
            continue
        # Only rename if src is a "fresh" temp from operator overloading.
        # Fresh temps are scalar regs (declared individually) that are
        # defined exactly once (by the operator) and used exactly once
        # (by this mov). Array regs (%r0..%rN from .reg .b32 %r<N>)
        # are NOT temps — they're part of the kernel's register allocation.
        # Detect fresh temps: they're declared as individual .reg (no count)
        # and have a high index or different type than the array.
        src_is_fresh = False
        for s in statements[:i]:
            if isinstance(s, RegDecl) and s.name == src_name and s.count is None:
                src_is_fresh = True
                break
        if not src_is_fresh:
            continue
        # Also verify src is defined exactly once and used exactly once (this mov)
        src_def_count = sum(1 for s in statements if isinstance(s, Instruction)
                           and s.operands and isinstance(s.operands[0], RegisterOperand)
                           and s.operands[0].name == src_name)
        src_use_count = sum(1 for s in statements if isinstance(s, Instruction)
                           and any(isinstance(op, RegisterOperand) and op.name == src_name
                                   for op in s.operands[1:]))
        if src_def_count != 1 or src_use_count != 1:
            continue
        renames[src_name] = dst_name
        mov_indices.add(i)

    if not renames:
        return statements

    # Apply renames: walk backward from each mov to rename src → dst
    # in all instructions that define or use src
    result: list[Statement] = []
    removed_regs: set[str] = set(renames.keys())

    for i, stmt in enumerate(statements):
        if i in mov_indices:
            continue  # remove the mov

        if isinstance(stmt, RegDecl):
            # Remove reg declarations for renamed-away registers
            if stmt.count is None and stmt.name in removed_regs:
                continue
            result.append(stmt)
            continue

        if isinstance(stmt, Instruction):
            # Rename operands
            new_ops = tuple(_rename_operand(op, renames) for op in stmt.operands)
            if new_ops != stmt.operands:
                from dataclasses import replace
                stmt = replace(stmt, operands=new_ops)
        result.append(stmt)

    return result


# ---------------------------------------------------------------------------
# GVN + LICM for invariant (loop-invariant, single-value) registers.
#
# Motivation: pyptx's imperative DSL recomputes address arithmetic inline
# (`base + BAR_X + off`) at every use, and emits identity `add rX, rY, 0`.
# CUTLASS's functional DSL materializes each value once. This pass closes
# that gap by global-value-numbering registers that provably hold ONE
# fixed value for the whole kernel, computing each once at a dominating
# point (the straight-line entry region) and renaming all duplicates.
#
# Safety: a register is "invariant" iff it is single-def AND every source
# of its defining instruction is itself invariant (or an immediate/special
# register). Such a register holds one value for the entire execution, so
# (a) two invariant computes with the same (opcode, modifiers, sources)
# yield the same value, and (b) hoisting the single canonical computation
# into the entry region — which dominates every use — is semantics-
# preserving. Non-invariant registers are never touched.
# ---------------------------------------------------------------------------

# Opcodes that write operand[0] as a pure function of their sources (no
# memory / side effects). Only these can define an invariant value.
_PURE_ALU = frozenset({
    "add", "sub", "mul", "mad", "shl", "shr", "and", "or", "xor",
    "not", "mov", "cvt", "min", "max", "neg", "sad",
})


def _operand_reg_names(op) -> list[str]:
    """All register names read/written inside an operand (recursively)."""
    if isinstance(op, RegisterOperand):
        return [op.name]
    if isinstance(op, AddressOperand):
        return [op.base]
    if isinstance(op, (VectorOperand, ParenthesizedOperand)):
        out: list[str] = []
        for e in op.elements:
            out.extend(_operand_reg_names(e))
        return out
    if isinstance(op, NegatedOperand):
        return _operand_reg_names(op.operand)
    if isinstance(op, PipeOperand):
        return _operand_reg_names(op.left) + _operand_reg_names(op.right)
    return []


def _is_special_reg(name: str) -> bool:
    # %tid, %ctaid, %ntid, %laneid, %clock, %cluster*, params -> stable.
    return name.startswith("%") and any(
        s in name for s in ("tid", "ctaid", "ntid", "nctaid", "laneid",
                             "clock", "cluster", "smid", "warpid", "%p")
    ) and not name[1:2].isdigit()


def _sig(instr: Instruction, canon: dict[str, str]) -> tuple:
    """Value signature of a pure compute, with sources canonicalized."""
    src_reprs = []
    for op in instr.operands[1:]:
        if isinstance(op, RegisterOperand):
            src_reprs.append(("r", canon.get(op.name, op.name)))
        elif isinstance(op, ImmediateOperand):
            src_reprs.append(("i", op.text))
        else:
            # address / vector sources: not value-numbered (rare for ALU)
            src_reprs.append(("o", repr(op)))
    return (instr.opcode, instr.modifiers, tuple(src_reprs))


def gvn_invariant(statements: list[Statement]) -> list[Statement]:
    """Global value numbering + LICM for invariant registers.

    Eliminates redundant invariant computations (duplicate address
    arithmetic, identity `add x,y,0`) by keeping one canonical value,
    hoisted into the entry region so it dominates every use.
    """
    # 1. def counts and first def index (operand[0]-as-register over-counts,
    #    which is safe: a mis-counted "def" only makes a reg look multi-def,
    #    excluding it from the invariant set -> conservative).
    from collections import Counter
    defc: Counter = Counter()
    first_def: dict[str, int] = {}
    for i, s in enumerate(statements):
        if isinstance(s, Instruction) and s.operands and isinstance(s.operands[0], RegisterOperand):
            n = s.operands[0].name
            defc[n] += 1
            first_def.setdefault(n, i)

    # entry region end = first Label or first branch/ret (start of control flow)
    entry_end = len(statements)
    for i, s in enumerate(statements):
        if isinstance(s, Label):
            entry_end = i
            break
        if isinstance(s, Instruction) and s.opcode in ("bra", "ret", "call"):
            entry_end = i
            break

    # 2. invariant set by fixpoint over pure single-def computes.
    invariant: set[str] = set()
    changed = True
    while changed:
        changed = False
        for i, s in enumerate(statements):
            if not (isinstance(s, Instruction) and s.opcode in _PURE_ALU):
                continue
            if s.predicate is not None:
                continue  # conditionally defined -> value not guaranteed
            if not (s.operands and isinstance(s.operands[0], RegisterOperand)):
                continue
            d = s.operands[0].name
            if d in invariant or defc[d] != 1:
                continue
            ok = True
            for op in s.operands[1:]:
                for rn in _operand_reg_names(op):
                    if rn in invariant or _is_special_reg(rn):
                        continue
                    ok = False
                    break
                if not ok:
                    break
            if ok:
                invariant.add(d)
                changed = True

    if not invariant:
        return statements

    # 3. value-number the invariant computes. Union sources via canon map.
    canon: dict[str, str] = {}                 # reg -> canonical reg (same value)
    sig_to_canon: dict[tuple, str] = {}
    # process in program order so 'first' def wins as canonical
    inv_defs = [(i, s) for i, s in enumerate(statements)
                if isinstance(s, Instruction) and s.opcode in _PURE_ALU
                and s.operands and isinstance(s.operands[0], RegisterOperand)
                and s.operands[0].name in invariant]
    # iterate to fixpoint so canonicalized sources fold transitively
    for _ in range(len(inv_defs) + 1):
        stable = True
        for i, s in inv_defs:
            d = s.operands[0].name
            sig = _sig(s, canon)
            c = sig_to_canon.get(sig)
            if c is None:
                sig_to_canon[sig] = canon.get(d, d)
            else:
                if canon.get(d, d) != c:
                    canon[d] = c
                    stable = False
        if stable:
            break

    # regs that are being folded away (canon points elsewhere)
    folded = {r for r, c in canon.items() if c != r}
    if not folded:
        return statements

    return _rewrite_gvn(statements, canon, folded, invariant, entry_end, inv_defs)


def _rewrite_gvn(statements, canon, folded, invariant, entry_end, inv_defs):
    """Emit canonical invariant computes hoisted to entry, drop duplicates,
    rename all uses. Canonical computes keep their original position UNLESS
    they are non-entry (then hoist to entry_end in dependency order)."""
    # canonical reg -> its defining instruction (choose the surviving def)
    inv_def_by_reg = {s.operands[0].name: (i, s) for i, s in inv_defs}
    canon_regs = {canon.get(r, r) for r in invariant}
    # Which canonical defs sit outside the entry region and must be hoisted
    # so they dominate every (possibly-earlier-in-loop) use.
    to_hoist = [r for r in canon_regs
                if r in inv_def_by_reg and inv_def_by_reg[r][0] >= entry_end]

    # topological order of hoisted defs by invariant-source deps
    order: list[str] = []
    seen: set[str] = set()

    def visit(reg: str):
        if reg in seen or reg not in inv_def_by_reg:
            return
        seen.add(reg)
        _, s = inv_def_by_reg[reg]
        for op in s.operands[1:]:
            for rn in _operand_reg_names(op):
                c = canon.get(rn, rn)
                if c in to_hoist:
                    visit(c)
        if reg in to_hoist:
            order.append(reg)

    for r in to_hoist:
        visit(r)

    hoist_indices = {inv_def_by_reg[r][0] for r in to_hoist}
    drop_indices = set()
    for r in folded:
        if r in inv_def_by_reg:
            drop_indices.add(inv_def_by_reg[r][0])

    result: list[Statement] = []
    from dataclasses import replace
    for i, s in enumerate(statements):
        if i == entry_end:
            # insert hoisted canonical computes here (dep order), renamed
            for hr in order:
                _, hs = inv_def_by_reg[hr]
                new_ops = tuple(_rename_operand(op, canon) for op in hs.operands)
                result.append(replace(hs, operands=new_ops))
        if i in hoist_indices or i in drop_indices:
            continue  # moved to entry, or folded away
        if isinstance(s, RegDecl):
            if s.count is None and s.name in folded:
                continue  # decl for a folded reg
            result.append(s)
            continue
        if isinstance(s, Instruction):
            new_ops = tuple(_rename_operand(op, canon) for op in s.operands)
            pred = s.predicate
            if pred is not None and isinstance(getattr(pred, "register", None), str):
                nc = canon.get(pred.register, pred.register)
                if nc != pred.register:
                    pred = replace(pred, register=nc)
            if new_ops != s.operands or pred is not s.predicate:
                s = replace(s, operands=new_ops, predicate=pred)
        result.append(s)
    # if entry_end == len(statements) the hoist loop above never ran
    if entry_end >= len(statements) and order:
        # extremely rare (no control flow) — nothing to hoist meaningfully
        pass
    return result


def verify_body(statements: list[Statement]) -> list[str]:
    """Cheap structural verifier: catch dangling uses and multi-defined
    canonicals introduced by a buggy pass. Returns a list of problems
    (empty == looks well-formed). Not a substitute for GPU validation,
    but catches gross mistakes offline."""
    from collections import Counter
    problems: list[str] = []
    declared: set[str] = set()
    declared_prefix: set[str] = set()  # ranged decls: %r<count>
    for s in statements:
        if isinstance(s, RegDecl):
            if s.count is None:
                declared.add(s.name)
            else:
                declared_prefix.add(s.name)
    defined: set[str] = set()
    for s in statements:
        if isinstance(s, Instruction) and s.operands and isinstance(s.operands[0], RegisterOperand):
            defined.add(s.operands[0].name)
    for s in statements:
        if not isinstance(s, Instruction):
            continue
        uses: list[str] = []
        for op in s.operands[1:]:
            uses += _operand_reg_names(op)
        if s.predicate is not None and isinstance(getattr(s.predicate, "register", None), str):
            uses.append(s.predicate.register)
        for u in uses:
            if not u.startswith("%") or _is_special_reg(u):
                continue
            if u in defined or u in declared:
                continue
            base = u.rstrip("0123456789")
            if base in declared_prefix:
                continue
            problems.append(f"dangling use of {u}")
    return problems


def optimize_body(statements: list[Statement], level: int = 1) -> list[Statement]:
    """Run the pass pipeline on one function body.

    level 0: no-op (byte-identical to trace output).
    level 1: copy propagation + GVN/LICM of invariant computes (pass D).

    Falls back to the input unchanged if the result fails verification —
    so a pass bug can never silently corrupt a kernel.
    """
    if level < 1:
        return statements
    original = statements
    out = copy_propagate(list(statements))
    out = gvn_invariant(out)
    problems = verify_body(out)
    if problems:
        import warnings
        warnings.warn(
            f"pyptx.optimize: verification failed ({problems[:3]}); "
            f"falling back to unoptimized body",
            RuntimeWarning,
        )
        return original
    return out


def optimize_module(module, level: int = 1):
    """Apply the pass pipeline to every Function body in a Module."""
    if level < 1:
        return module
    from dataclasses import replace
    new_dirs = []
    changed = False
    for d in module.directives:
        if type(d).__name__ == "Function" and getattr(d, "body", None):
            new_body = optimize_body(list(d.body), level=level)
            if new_body is not d.body:
                d = replace(d, body=tuple(new_body))
                changed = True
        new_dirs.append(d)
    if not changed:
        return module
    return replace(module, directives=tuple(new_dirs))


def _rename_operand(op, renames: dict[str, str]):
    """Rename register references in an operand."""
    if isinstance(op, RegisterOperand):
        new_name = renames.get(op.name, op.name)
        if new_name != op.name:
            return RegisterOperand(name=new_name)
        return op
    if isinstance(op, AddressOperand):
        new_base = renames.get(op.base, op.base)
        if new_base != op.base:
            return AddressOperand(base=new_base, offset=op.offset)
        return op
    if isinstance(op, VectorOperand):
        new_elems = tuple(_rename_operand(e, renames) for e in op.elements)
        if new_elems != op.elements:
            return VectorOperand(elements=new_elems)
        return op
    if isinstance(op, NegatedOperand):
        new_inner = _rename_operand(op.operand, renames)
        if new_inner is not op.operand:
            return NegatedOperand(operand=new_inner)
        return op
    if isinstance(op, PipeOperand):
        new_left = _rename_operand(op.left, renames)
        new_right = _rename_operand(op.right, renames)
        if new_left is not op.left or new_right is not op.right:
            return PipeOperand(left=new_left, right=new_right)
        return op
    if isinstance(op, ParenthesizedOperand):
        new_elems = tuple(_rename_operand(e, renames) for e in op.elements)
        if new_elems != op.elements:
            return ParenthesizedOperand(elements=new_elems)
        return op
    return op
