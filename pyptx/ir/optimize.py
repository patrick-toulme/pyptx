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

    UNSOUND IN GENERAL — kept only for its narrow single-def use and its unit
    tests. The backward %src->%dst rename moves %dst's definition earlier; if
    %dst is a reused (multiply-defined) register that is read between %src's
    definition and the mov, the rename clobbers that read. Differential
    validation (pyptx.ir.simulate) catches this on the FA kernel, so this pass
    is deliberately NOT in the optimization pipeline — gvn + dce achieve the
    same cleanup soundly. Do not add it back without a liveness-based guard
    (rename only when %dst is dead at %src's definition point).
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


import re as _re
# Register tokens embedded in a raw-string operand field (e.g. the TMA
# coordinate vector pyptx stores in AddressOperand.offset: ", {0, %r98}").
_REG_TOKEN = _re.compile(r"%[A-Za-z_][A-Za-z0-9_$]*")


def _operand_reg_names(op) -> list[str]:
    """All register names read/written inside an operand (recursively)."""
    if isinstance(op, RegisterOperand):
        return [op.name]
    if isinstance(op, AddressOperand):
        names = [op.base]
        if isinstance(op.offset, str):
            names += _REG_TOKEN.findall(op.offset)
        return names
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


# ---------------------------------------------------------------------------
# False-dependency breaking (value splitting / SSA-lite).
#
# pyptx's imperative DSL reuses one scratch register for many *independent*
# values (`%r = a; ...; %r = b; ...`). Each reuse is a WAR/WAW hazard that
# forces ptxas to serialize the two computations and stretches one live
# range, whereas CuTe's functional style names each value once and lets
# ptxas schedule/allocate them freely. This pass restores that freedom:
# inside a straight-line region, when a register is redefined by an
# instruction that does NOT read it (a "killing" def), the previous value
# is provably dead, so that value's def+uses are renamed to a fresh
# register. The first value in a region (may read a live-in value) and the
# last value (may be live-out) keep the original name — proving otherwise
# needs cross-block liveness, so we conservatively don't touch them.
#
# Safety: only unpredicated, plain-scalar-register defs split a value; any
# predicated or vector/complex-dest definition of a register makes that
# register untouchable in the whole region. Fresh registers are declared
# with the original's type; a register whose type can't be resolved from a
# declaration is skipped. The result still passes verify_body, and — since
# it grows the virtual-register count — needs ptxas/GPU validation before
# it should be trusted to help rather than raise register pressure.
# ---------------------------------------------------------------------------

_BLOCK_ENDERS = frozenset({"bra", "ret", "call", "exit", "brkpt", "trap"})


def _ty_str(ty) -> str:
    """Normalize a decl type (ScalarType enum or raw string) to its PTX text."""
    return ty.ptx if hasattr(ty, "ptx") else str(ty)


def _reg_type_lookup(statements):
    """Return a fn name->declared-type, covering scalar and ranged decls."""
    scalar: dict[str, str] = {}
    ranged: list[tuple[str, str]] = []
    for s in statements:
        if isinstance(s, RegDecl):
            if s.count is None:
                scalar[s.name] = s.type
            else:
                ranged.append((s.name, s.type))

    def lookup(name: str):
        if name in scalar:
            return scalar[name]
        for pfx, ty in ranged:
            if name.startswith(pfx) and name[len(pfx):].isdigit():
                return ty
        return None

    return lookup


def _def_and_uses(instr: Instruction):
    """(def_reg | None, complex_def_regs, use_regs) for an instruction.

    def_reg      — plain scalar destination register name (write-only), else None.
    complex_defs — registers written via a vector/paren destination (multi-dest).
    use_regs     — every register read: operands[1:], plus operand[0]'s registers
                   when operand[0] is not a plain destination (e.g. st's address).
    """
    if not instr.operands:
        return None, [], []
    op0 = instr.operands[0]
    def_reg = None
    complex_defs: list[str] = []
    op0_uses: list[str] = []
    if isinstance(op0, RegisterOperand):
        def_reg = op0.name
    elif isinstance(op0, (VectorOperand, ParenthesizedOperand)):
        complex_defs = _operand_reg_names(op0)
    else:
        # address / negated / pipe in dest position -> these registers are read
        op0_uses = _operand_reg_names(op0)
    uses: list[str] = list(op0_uses)
    for op in instr.operands[1:]:
        uses.extend(_operand_reg_names(op))
    return def_reg, complex_defs, uses


def split_false_deps(statements: list[Statement]) -> list[Statement]:
    """Break WAR/WAW false dependencies by renaming dead reused values.

    Within each straight-line region, a register's timeline is split into
    independent values at every killing redefinition; each value that is
    neither the region's first nor last (and whose defining instruction
    does not read the register) is renamed to a fresh register.
    """
    reg_type = _reg_type_lookup(statements)

    # existing names, to guarantee fresh ones don't collide
    existing: set[str] = set()
    for s in statements:
        if isinstance(s, RegDecl):
            existing.add(s.name)
        elif isinstance(s, Instruction):
            for op in s.operands:
                existing.update(_operand_reg_names(op))
    prefix = "%vs"
    while any(n.startswith(prefix) for n in existing):
        prefix += "_"

    # per-index register renames to apply, and fresh decls to add
    rename_at: dict[int, dict[str, str]] = {}
    fresh_decls: list[RegDecl] = []
    counter = 0

    n = len(statements)
    i = 0
    while i < n:
        # find the extent of this straight-line block [i, j)
        # a block runs until (and including) a block-ender, or until just
        # before a Label (which starts a new block / is a join point).
        j = i
        started = False
        while j < n:
            s = statements[j]
            if isinstance(s, Label):
                if started:
                    break  # label begins a new block
                # leading label(s) belong to this block start
                j += 1
                continue
            started = True
            j += 1
            if isinstance(s, Instruction) and s.opcode in _BLOCK_ENDERS:
                break
        block = range(i, j)

        # collect per-register def events + poison flags within the block
        # def event: (index, kind) kind in {"kill","carry"}
        defs: dict[str, list[tuple[int, str]]] = {}
        poison: set[str] = set()
        for k in block:
            s = statements[k]
            if not isinstance(s, Instruction):
                continue
            def_reg, complex_defs, uses = _def_and_uses(s)
            for r in complex_defs:
                poison.add(r)  # written via vector/paren dest -> hands off
            if def_reg is None:
                continue
            if s.predicate is not None:
                poison.add(def_reg)  # conditional write -> value not guaranteed
                continue
            kind = "carry" if def_reg in uses else "kill"
            defs.setdefault(def_reg, []).append((k, kind))

        for r, events in defs.items():
            if r in poison:
                continue
            ty = reg_type(r)
            if ty is None or _ty_str(ty) == ".pred" or _is_special_reg(r):
                continue
            # value starts = first def, plus every killing def
            starts: list[int] = []
            for idx, kind in events:
                if not starts or kind == "kill":
                    starts.append(idx)
            if len(starts) < 2:
                continue  # single value in block -> nothing to split
            # value spans; last value is kept (possible live-out)
            for vi in range(len(starts) - 1):
                start = starts[vi]
                end = starts[vi + 1]  # exclusive
                # only rename values whose defining instruction is a killing
                # def (doesn't read r); a first value that reads a live-in is
                # left alone.
                def_kind = next(kind for idx, kind in events if idx == start)
                if def_kind != "kill":
                    continue
                fresh = f"{prefix}{counter}"
                counter += 1
                fresh_decls.append(RegDecl(type=ty, name=fresh, count=None))
                for k in range(start, end):
                    s = statements[k]
                    if not isinstance(s, Instruction):
                        continue
                    if r in _stmt_reg_names(s):
                        rename_at.setdefault(k, {})[r] = fresh
        i = j

    if not rename_at:
        return statements

    from dataclasses import replace
    result: list[Statement] = list(fresh_decls)
    for k, s in enumerate(statements):
        if k in rename_at and isinstance(s, Instruction):
            renames = rename_at[k]
            new_ops = tuple(_rename_operand(op, renames) for op in s.operands)
            pred = s.predicate
            if pred is not None and isinstance(getattr(pred, "register", None), str):
                nc = renames.get(pred.register, pred.register)
                if nc != pred.register:
                    pred = replace(pred, register=nc)
            if new_ops != s.operands or pred is not s.predicate:
                s = replace(s, operands=new_ops, predicate=pred)
        result.append(s)
    return result


def _stmt_reg_names(instr: Instruction) -> set[str]:
    out: set[str] = set()
    for op in instr.operands:
        out.update(_operand_reg_names(op))
    if instr.predicate is not None and isinstance(getattr(instr.predicate, "register", None), str):
        out.add(instr.predicate.register)
    return out


# ---------------------------------------------------------------------------
# Pass A: dead-code elimination + liveness-driven register allocation.
#
# DCE removes pure computes whose result is never used (a strict win — fewer
# instructions, fewer live values). Register allocation then builds an
# interference graph from liveness and greedily colors it per register type,
# packing the virtual registers onto a minimal set. This is the "linear-scan
# allocator handing ptxas tight live ranges" lever, and it reports the
# concrete metric it targets (distinct vregs before/after, vs the max
# simultaneously-live lower bound).
#
# Coloring MERGES non-interfering registers, so unlike value-splitting it can
# reintroduce WAR/WAW anti-deps — the two passes are opposing levers, to be
# A/B'd on hardware. It is also the one pass verify_body cannot fully police
# (a bad coloring produces no dangling use), so it is guarded hard: it bails
# to the input on any unresolved control flow, and asserts every computed
# interference edge is respected by the final coloring before returning.
# ---------------------------------------------------------------------------

def dead_code_eliminate(statements: list[Statement]) -> list[Statement]:
    """Remove pure instructions whose (non-special, scalar) result is dead."""
    from pyptx.ir.analysis import (
        build_cfg, compute_liveness, liveness_at_instructions,
        instr_def_uses, is_special_reg, SIDE_EFFECTING,
    )
    stmts = list(statements)
    changed = True
    while changed:
        changed = False
        cfg = build_cfg(stmts)
        _, live_out = compute_liveness(cfg)
        live_after = liveness_at_instructions(cfg, live_out)
        dead: set[int] = set()
        for idx, instr in enumerate(stmts):
            if not isinstance(instr, Instruction):
                continue
            if instr.opcode in SIDE_EFFECTING:
                continue
            sdef, cdefs, uses, pred = instr_def_uses(instr)
            if sdef is None or cdefs or is_special_reg(sdef):
                continue  # only pure single-scalar-dest, non-predicate defs
            if sdef not in live_after.get(idx, ()):
                dead.add(idx)
        if dead:
            stmts = [s for k, s in enumerate(stmts) if k not in dead]
            changed = True
    return _drop_unused_scalar_decls(stmts)


def _drop_unused_scalar_decls(statements: list[Statement]) -> list[Statement]:
    used: set[str] = set()
    for s in statements:
        if isinstance(s, Instruction):
            used |= _stmt_reg_names(s)
    out: list[Statement] = []
    for s in statements:
        if isinstance(s, RegDecl) and s.count is None and s.name not in used:
            continue
        out.append(s)
    return out


def allocate_registers(statements: list[Statement]) -> list[Statement]:
    """Interference-graph register coloring. Bails (returns input unchanged)
    on any control flow it can't prove it modeled correctly."""
    from pyptx.ir.analysis import (
        build_cfg, compute_liveness, liveness_at_instructions,
        instr_def_uses, is_special_reg, _branch_target,
    )
    from collections import defaultdict

    cfg = build_cfg(statements)
    # SAFETY: every branch target must resolve, or liveness is unsound; and
    # `call` has a nonstandard def/use shape we don't model -> bail on both.
    for blk in cfg.blocks:
        for k in blk.insts:
            instr = statements[k]
            if instr.opcode == "call":
                return statements
            if instr.opcode == "bra":
                tgt = _branch_target(instr)
                if tgt is None or tgt not in cfg.label_to_block:
                    return statements  # unresolved / indirect branch -> bail

    live_in, live_out = compute_liveness(cfg)
    live_after = liveness_at_instructions(cfg, live_out)
    type_lookup = _reg_type_lookup(statements)

    # allocatable regs = non-special regs with a known declared type
    regs: set[str] = set()
    for instr in statements:
        if not isinstance(instr, Instruction):
            continue
        sdef, cdefs, uses, _ = instr_def_uses(instr)
        for r in ([sdef] if sdef else []) + cdefs + uses:
            if r and not is_special_reg(r):
                regs.add(r)
    regs = {r for r in regs if type_lookup(r) is not None}
    if not regs:
        return statements

    adj: dict[str, set[str]] = defaultdict(set)

    def add_clique(names):
        members = [r for r in names if r in regs]
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                x, y = members[a], members[b]
                adj[x].add(y)
                adj[y].add(x)

    add_clique(live_in[cfg.blocks[0].id])
    for idx, instr in enumerate(statements):
        if not isinstance(instr, Instruction):
            continue
        pointset = set(live_after.get(idx, ()))
        sdef, cdefs, uses, _ = instr_def_uses(instr)
        if sdef and not is_special_reg(sdef):
            pointset.add(sdef)
        for r in cdefs:
            if not is_special_reg(r):
                pointset.add(r)
        add_clique(pointset)

    # color each type's subgraph greedily (highest degree first)
    by_type: dict[str, list[str]] = defaultdict(list)
    for r in regs:
        by_type[type_lookup(r)].append(r)
    color: dict[str, int] = {}
    ncolors: dict[str, int] = {}
    for ty, rs in by_type.items():
        for r in sorted(rs, key=lambda r: (-len(adj[r]), r)):
            taken = {color[n] for n in adj[r] if n in color}
            c = 0
            while c in taken:
                c += 1
            color[r] = c
        ncolors[ty] = (max((color[r] for r in rs), default=-1) + 1)

    # POST-CONDITION: every interference edge must be a different color.
    for x, neigh in adj.items():
        for y in neigh:
            if type_lookup(x) == type_lookup(y) and color[x] == color[y]:
                return statements  # coloring unsound -> bail rather than corrupt

    rename: dict[str, str] = {}
    for r in regs:
        ty = type_lookup(r)
        rename[r] = f"%ra_{_ty_str(ty).replace('.', '')}_{color[r]}"
    if all(rename[r] == r for r in regs):
        return statements

    from dataclasses import replace
    # rebuild: fresh compact decls for the colored set + renamed instructions,
    # dropping the old decls of allocated regs.
    new_decls: list[RegDecl] = []
    for ty, k in ncolors.items():
        for c in range(k):
            new_decls.append(RegDecl(type=ty, name=f"%ra_{_ty_str(ty).replace('.', '')}_{c}", count=None))

    result: list[Statement] = list(new_decls)
    for s in statements:
        if isinstance(s, RegDecl):
            # drop decls fully covered by the allocation (their regs are renamed)
            if s.count is None and s.name in rename:
                continue
            if s.count is not None:
                # ranged decl: keep only if some covered reg escaped allocation
                pfx = s.name
                covered = any(
                    r.startswith(pfx) and r[len(pfx):].isdigit() and r in rename
                    for r in regs
                )
                escaped = any(
                    r.startswith(pfx) and r[len(pfx):].isdigit() and r not in rename
                    for r in regs
                )
                if covered and not escaped:
                    continue
            result.append(s)
            continue
        if isinstance(s, Instruction):
            new_ops = tuple(_rename_operand(op, rename) for op in s.operands)
            pred = s.predicate
            if pred is not None and isinstance(getattr(pred, "register", None), str):
                nc = rename.get(pred.register, pred.register)
                if nc != pred.register:
                    pred = replace(pred, register=nc)
            if new_ops != s.operands or pred is not s.predicate:
                s = replace(s, operands=new_ops, predicate=pred)
        result.append(s)
    return result


# ---------------------------------------------------------------------------
# Pass B: list scheduling.
#
# Reorders instructions within each basic block to interleave independent
# dependency chains, so a long-latency producer (a TMEM/global load, an SFU
# ex2) issues well before its consumer with other work in between, instead of
# the strict Python program order the trace emits. Preserves semantics by
# honoring every register RAW/WAR/WAW dependence and keeping all
# side-effecting instructions (memory, barriers, control) in their original
# relative order. Priority is the latency-weighted critical path (longest-
# path-first), the textbook list-scheduling heuristic.
#
# ptxas re-schedules too, so this is a modest lever; but a cleaner starting
# order can only help its (finite) scheduling window. Opt-in; the schedule is
# checked to be a valid topological order before it is accepted.
# ---------------------------------------------------------------------------

def _op_latency(opcode: str) -> int:
    if opcode in ("ld", "atom", "cp", "mbarrier"):
        return 8
    if opcode.startswith("tcgen05") or opcode in ("mma", "wgmma"):
        return 8
    if opcode in ("ex2", "lg2", "rcp", "sqrt", "rsqrt", "sin", "cos", "div", "rem"):
        return 6
    if opcode in ("mul", "mad", "fma", "dp4a", "dp2a"):
        return 3
    return 1


def list_schedule(statements: list[Statement]) -> list[Statement]:
    """Latency-oriented list scheduling within each basic block."""
    from pyptx.ir.analysis import build_cfg, instr_def_uses, is_special_reg, SIDE_EFFECTING

    cfg = build_cfg(statements)
    result = list(statements)

    for blk in cfg.blocks:
        slots = list(blk.insts)  # statement indices holding Instructions
        if len(slots) < 3:
            continue
        instrs = [statements[k] for k in slots]
        m = len(instrs)

        # reads/writes per instruction (conservative for predication)
        reads: list[set[str]] = []
        writes: list[set[str]] = []
        for ins in instrs:
            sdef, cdefs, uses, pred = instr_def_uses(ins)
            r = {x for x in uses if not is_special_reg(x)}
            w = set()
            if sdef is not None and not is_special_reg(sdef):
                w.add(sdef)
            for x in cdefs:
                if not is_special_reg(x):
                    w.add(x)
            if pred:
                r |= w  # conditional write also depends on the old value
            reads.append(r)
            writes.append(w)

        se = [i for i in range(m) if instrs[i].opcode in SIDE_EFFECTING]
        term_local = None
        if instrs and instrs[-1].opcode in ("bra", "ret", "exit", "trap"):
            term_local = m - 1

        # dependence edges pred -> succ (succ must come after pred)
        succ: list[set[int]] = [set() for _ in range(m)]
        indeg = [0] * m

        def add_edge(a: int, b: int):
            if a < b and b not in succ[a]:
                succ[a].add(b)

        for a in range(m):
            for b in range(a + 1, m):
                if (writes[a] & reads[b]) or (reads[a] & writes[b]) or (writes[a] & writes[b]):
                    add_edge(a, b)
        # total order among side-effecting ops (consecutive chain suffices)
        for x, y in zip(se, se[1:]):
            add_edge(x, y)
        # pin a terminator strictly last
        if term_local is not None:
            for a in range(m):
                if a != term_local:
                    add_edge(a, term_local)

        for a in range(m):
            for b in succ[a]:
                indeg[b] += 1

        # latency-weighted critical path (priority)
        prio = [0] * m
        for a in reversed(range(m)):
            best = 0
            for b in succ[a]:
                best = max(best, prio[b])
            prio[a] = _op_latency(instrs[a].opcode) + best

        # list schedule: among ready (indeg 0), pick highest prio, then
        # earliest original index (stable).
        import heapq
        ready = [(-prio[i], i) for i in range(m) if indeg[i] == 0]
        heapq.heapify(ready)
        order: list[int] = []
        remaining_indeg = list(indeg)
        while ready:
            _, i = heapq.heappop(ready)
            order.append(i)
            for b in succ[i]:
                remaining_indeg[b] -= 1
                if remaining_indeg[b] == 0:
                    heapq.heappush(ready, (-prio[b], b))

        if len(order) != m:
            continue  # cycle (shouldn't happen) -> leave block as-is

        # validate topological order before committing
        pos = {i: p for p, i in enumerate(order)}
        ok = all(pos[a] < pos[b] for a in range(m) for b in succ[a])
        if not ok:
            continue
        if order == list(range(m)):
            continue  # no change

        for slot, local in zip(slots, order):
            result[slot] = instrs[local]

    return result


# ---------------------------------------------------------------------------
# Pass C: SSA-based global value renaming.
#
# The principled, global version of the block-local value-splitting pass.
# Full phi-based SSA with out-of-SSA destruction (critical-edge splitting,
# parallel-copy sequentialization) is the textbook form, but its correctness
# hinges on details that can't be exercised without running the kernel — so
# here it is realized as the provably-safe subset that needs NO phi nodes and
# NO destruction: reaching-definitions analysis isolates every *exclusive*
# def->use web (an unconditional def whose value is the sole reaching
# definition at each of its uses) and alpha-renames it to a fresh register.
# That is exactly what SSA renaming produces for a variable that never merges;
# variables that would require a phi are left untouched. Result is a strictly
# larger set of independent live ranges (broken false deps) across block
# boundaries, complementing the intra-block `split` pass. The "coalescing"
# half of "SSA + coalescing" is the `regalloc` pass.
# ---------------------------------------------------------------------------

def ssa_reconstruct(statements: list[Statement]) -> list[Statement]:
    """Pass C: isolate exclusive def->use webs into fresh registers."""
    return _ssa_reconstruct_impl(statements)


def _ssa_reconstruct_impl(statements: list[Statement]) -> list[Statement]:
    from pyptx.ir.analysis import build_cfg, instr_def_uses, is_special_reg

    cfg = build_cfg(statements)
    type_lookup = _reg_type_lookup(statements)

    # per-instruction reads / (unconditional, conditional) writes
    reads: dict[int, set[str]] = {}
    uwrites: dict[int, set[str]] = {}   # unconditional (killing) writes
    cwrites: dict[int, set[str]] = {}   # conditional (predicated) writes
    for idx, ins in enumerate(statements):
        if not isinstance(ins, Instruction):
            continue
        sdef, cdefs, uses, pred = instr_def_uses(ins)
        reads[idx] = {x for x in uses if not is_special_reg(x)}
        w = set()
        if sdef is not None and not is_special_reg(sdef):
            w.add(sdef)
        for x in cdefs:
            if not is_special_reg(x):
                w.add(x)
        if pred:
            cwrites[idx] = w
            uwrites[idx] = set()
        else:
            uwrites[idx] = w
            cwrites[idx] = set()

    # reaching definitions: a def-site is (idx, reg). Forward dataflow.
    reach_out: dict[int, set[tuple[int, str]]] = {b.id: set() for b in cfg.blocks}
    reach_in: dict[int, set[tuple[int, str]]] = {b.id: set() for b in cfg.blocks}
    reach_before: dict[int, set[tuple[int, str]]] = {}

    def transfer(blk, cur):
        cur = set(cur)
        for idx in blk.insts:
            reach_before[idx] = set(cur)
            for r in uwrites.get(idx, ()):  # kill prior defs of r
                cur = {(i, rr) for (i, rr) in cur if rr != r}
                cur.add((idx, r))
            for r in cwrites.get(idx, ()):  # predicated: add, do not kill
                cur.add((idx, r))
        return cur

    changed = True
    while changed:
        changed = False
        for blk in cfg.blocks:
            inn = set()
            for p in blk.preds:
                inn |= reach_out[p]
            reach_in[blk.id] = inn
            out = transfer(blk, inn)
            if out != reach_out[blk.id]:
                reach_out[blk.id] = out
                changed = True

    # for each use-site, the reaching defs of the used reg
    # def_site -> set of use indices it reaches; also validity flags
    from collections import Counter, defaultdict
    # total unconditional-def count per reg — a single-def reg is already SSA,
    # so renaming it is pointless churn; only isolate reused (>=2 def) regs.
    defcount: Counter = Counter()
    for idx in uwrites:
        for r in uwrites[idx]:
            defcount[r] += 1
    def_uses: dict[tuple[int, str], set[int]] = defaultdict(set)
    def_isolatable: dict[tuple[int, str], bool] = {}
    # seed candidates: an unconditional def-site of a reused (>=2 def) reg that
    # is a KILLING def — it must not read its own register (a carrying def like
    # `shl r, r, n` reads the *previous* value at the same site, so renaming r
    # there would corrupt that read).
    for idx in uwrites:
        for r in uwrites[idx]:
            if defcount[r] >= 2 and r not in reads.get(idx, ()):
                def_isolatable[(idx, r)] = True

    for j in reads:
        rb = reach_before.get(j, set())
        for r in reads[j]:
            reaching = [(i, rr) for (i, rr) in rb if rr == r]
            carrying = r in uwrites.get(j, ()) or r in cwrites.get(j, ())
            if len(reaching) == 1 and reaching[0] in def_isolatable and not carrying:
                def_uses[reaching[0]].add(j)
            else:
                # this use is a merge / carrying / from a predicated def:
                # every def that reaches it becomes non-isolatable
                for d in reaching:
                    if d in def_isolatable:
                        def_isolatable[d] = False

    # Dominance gate: exclusive reaching-def is NECESSARY but not SUFFICIENT
    # for phi-free isolation — the def must also DOMINATE every use, or a
    # loop-carried use across a back-edge would read the fresh register before
    # it is ever written (used-before-def). Compute the dominator tree and
    # require def-block dom use-block (and def-before-use within a block).
    from pyptx.ir.analysis import compute_dominators
    dom = compute_dominators(cfg)
    blk_of = [-1] * len(statements)
    for b in cfg.blocks:
        for i in range(b.start, b.end):
            blk_of[i] = b.id

    def _dominates_all(def_idx, uses):
        db = blk_of[def_idx]
        for j in uses:
            ub = blk_of[j]
            if db not in dom.get(ub, ()):  # def-block must dominate use-block
                return False
            if ub == db and j < def_idx:   # same block: def must precede use
                return False
        return True

    # build renames for isolatable webs with a known type and >=1 use
    rename_at: dict[int, dict[str, str]] = {}
    fresh_decls: list[RegDecl] = []
    counter = 0
    # fresh name prefix guaranteed novel
    existing = {s.name for s in statements if isinstance(s, RegDecl)}
    prefix = "%ssa"
    while any(n.startswith(prefix) for n in existing):
        prefix += "_"

    for (idx, r), ok in def_isolatable.items():
        if not ok:
            continue
        us = def_uses.get((idx, r), set())
        if not us:
            continue  # dead def (DCE's job) — nothing to isolate
        if not _dominates_all(idx, us):
            continue  # def doesn't dominate a use -> would orphan it
        ty = type_lookup(r)
        if ty is None or _ty_str(ty) == ".pred":
            continue
        fresh = f"{prefix}{counter}"
        counter += 1
        fresh_decls.append(RegDecl(type=ty, name=fresh, count=None))
        rename_at.setdefault(idx, {})[r] = fresh   # the def
        for j in us:
            rename_at.setdefault(j, {})[r] = fresh  # its exclusive uses

    if not rename_at:
        return statements

    from dataclasses import replace
    result: list[Statement] = list(fresh_decls)
    for k, s in enumerate(statements):
        if k in rename_at and isinstance(s, Instruction):
            ren = rename_at[k]
            new_ops = tuple(_rename_operand(op, ren) for op in s.operands)
            pred = s.predicate
            if pred is not None and isinstance(getattr(pred, "register", None), str):
                nc = ren.get(pred.register, pred.register)
                if nc != pred.register:
                    pred = replace(pred, register=nc)
            if new_ops != s.operands or pred is not s.predicate:
                s = replace(s, operands=new_ops, predicate=pred)
        result.append(s)
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


# Named pass registry — every transform is individually addressable so the
# pipeline can be composed explicitly (env PYPTX_PASSES="gvn,dce,regalloc").
# NOTE: copy_propagate is intentionally absent — its backward src->dst rename
# is unsound when the destination is a reused (multiply-defined) register, and
# differential validation (pyptx.ir.simulate) flags it on the FA kernel. Its
# intended cleanup (RegArray.__setitem__ movs) is subsumed by gvn + dce.
def _pass_registry():
    return {
        "gvn": gvn_invariant,
        "dce": dead_code_eliminate,
        "split": split_false_deps,
        "regalloc": allocate_registers,
        "schedule": list_schedule,
        "ssa": ssa_reconstruct,
    }

# Numeric-level presets (env PYPTX_OPT). Level 1 is the low-risk set that only
# removes work; higher levels add the opposing false-dep / allocation levers
# which change the vreg count and want hardware A/B before being trusted.
_LEVEL_PRESETS = {
    1: ("gvn", "dce"),
    2: ("gvn", "dce", "split"),
    3: ("gvn", "dce", "regalloc"),
    4: ("gvn", "dce", "schedule"),
    5: ("gvn", "ssa", "dce", "schedule"),
}


def _hoist_declarations(statements: list[Statement]) -> list[Statement]:
    """Float every register declaration to the top of the body.

    pyptx emits each ``.reg`` declaration just before the register's first
    use. That is valid only while instructions keep their traced order — any
    pass that hoists or reorders an instruction above its register's
    declaration produces PTX that ptxas rejects ("Arguments mismatch", the
    register being untyped at that point), even though pyptx's own parser and
    verify_body accept it. Declaring all registers up front (standard PTX) is
    order-independent and makes the body robust to every reordering pass.
    """
    decls = [s for s in statements if isinstance(s, RegDecl)]
    rest = [s for s in statements if not isinstance(s, RegDecl)]
    if not decls:
        return statements
    return decls + rest


def optimize_body(statements, level: int = 1, passes=None, force: bool = False) -> list[Statement]:
    """Run the optimization pipeline on one function body.

    Pass selection: an explicit `passes` name list wins; otherwise the
    `level` preset is used (level 0 / empty = no-op). Each pass self-verifies
    (verify_body) and is skipped — leaving the pre-pass IR intact — if it
    would introduce a structural error, so a single buggy pass can neither
    corrupt the kernel nor sink the rest of the pipeline.
    """
    if passes is None:
        if level < 1:
            return statements
        passes = _LEVEL_PRESETS.get(level, _LEVEL_PRESETS[max(_LEVEL_PRESETS)])
    registry = _pass_registry()
    import warnings

    # SAFETY: a kernel that hand-tunes per-warp register budgets via
    # `setmaxnreg` depends on ptxas allocating a SPECIFIC number of registers
    # per warpgroup — a `setmaxnreg.inc N` becomes an illegal instruction at
    # runtime if the warp is already allocated more than N. These passes are
    # semantics-preserving at the register-dataflow level (differentially
    # validated), but any of them — even DCE removing provably-dead code —
    # perturbs ptxas's allocation and breaks that balance. Verified on the
    # Blackwell FA kernel (B200, CUDA 13): dce/regalloc/ssa/schedule all fault
    # at runtime while producing dataflow-equivalent PTX. So refuse to touch a
    # body that uses setmaxnreg; the optimization can't help it (the gap is
    # ptxas SASS quality, below the PTX layer) and can only break its tuning.
    if not force and any(isinstance(s, Instruction) and s.opcode == "setmaxnreg"
                         for s in statements):
        warnings.warn(
            "pyptx.optimize: kernel uses setmaxnreg (hand-tuned register "
            "budgets); skipping optimization — allocation-perturbing passes "
            "break the setmaxnreg balance at runtime (set PYPTX_FORCE=1 to "
            "override, e.g. for measurement)",
            RuntimeWarning,
        )
        return statements

    out = list(statements)
    for name in passes:
        fn = registry.get(name)
        if fn is None:
            warnings.warn(f"pyptx.optimize: unknown pass {name!r}; skipping",
                          RuntimeWarning)
            continue
        try:
            candidate = fn(list(out))
            problems = verify_body(candidate)
        except Exception as exc:  # a pass bug must never break emission
            warnings.warn(f"pyptx.optimize: pass {name!r} raised {exc!r}; "
                          f"skipping", RuntimeWarning)
            continue
        if problems:
            warnings.warn(f"pyptx.optimize: pass {name!r} failed verification "
                          f"({problems[:3]}); skipping", RuntimeWarning)
            continue
        out = candidate
    # Normalize: any pass may have reordered an instruction above its
    # register's declaration; float all declarations to the top so ptxas
    # (which requires declare-before-use) accepts the result.
    if out is not statements:
        out = _hoist_declarations(out)
    return out


def optimize_module(module, level: int = 1, passes=None, force: bool = False):
    """Apply the pipeline to every Function body in a Module."""
    if passes is None and level < 1:
        return module
    from dataclasses import replace
    new_dirs = []
    changed = False
    for d in module.directives:
        if type(d).__name__ == "Function" and getattr(d, "body", None):
            new_body = optimize_body(list(d.body), level=level, passes=passes, force=force)
            if tuple(new_body) != d.body:
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
        new_offset = op.offset
        if isinstance(op.offset, str):
            new_offset = _REG_TOKEN.sub(
                lambda mm: renames.get(mm.group(0), mm.group(0)), op.offset)
        if new_base != op.base or new_offset != op.offset:
            return AddressOperand(base=new_base, offset=new_offset)
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
