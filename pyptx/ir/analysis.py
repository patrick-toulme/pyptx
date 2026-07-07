"""Control-flow and dataflow analysis over a traced function body.

The optimization passes in :mod:`pyptx.ir.optimize` (register allocation,
list scheduling, SSA) share this machinery: a basic-block CFG built from the
flat Statement list, liveness by backward dataflow, and a dominator tree +
dominance frontiers for SSA construction.

Everything here is analysis only — no Statement is mutated. Registers that
are not declared with a ``.reg`` (special registers like ``%tid``, kernel
parameters) are treated as always-available and are never reported as
allocatable defs/uses, so the passes only ever touch real virtual registers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pyptx.ir.nodes import (
    AddressOperand,
    Instruction,
    Label,
    LabelOperand,
    NegatedOperand,
    ParenthesizedOperand,
    PipeOperand,
    RegDecl,
    RegisterOperand,
    Statement,
    VectorOperand,
)

# Opcodes that end a basic block by transferring control.
_TERMINATORS = frozenset({"bra", "ret", "exit", "trap"})
# Opcodes with side effects / ordering constraints: never dead, never
# reordered across each other (see optimize.list_schedule).
SIDE_EFFECTING = frozenset({
    "st", "atom", "red", "bar", "barrier", "membar", "fence", "bra", "ret",
    "exit", "trap", "call", "vote", "mbarrier", "cp", "tcgen05", "mma", "wgmma",
    "ld",  # loads can fault / alias; keep ordered w.r.t. stores conservatively
    "griddepcontrol", "elect", "shfl", "prmt.nc", "createpolicy",
})


# Register tokens embedded in a raw-string operand field (e.g. the TMA
# coordinate vector pyptx stores in AddressOperand.offset: ", {0, %r98}").
_REG_TOKEN = re.compile(r"%[A-Za-z_][A-Za-z0-9_$]*")


def operand_reg_names(op) -> list[str]:
    """Every register name referenced inside an operand (recursively)."""
    if isinstance(op, RegisterOperand):
        return [op.name]
    if isinstance(op, AddressOperand):
        names = [op.base]
        if isinstance(op.offset, str):
            # TMA / vector coordinates are stashed here as raw text
            names += _REG_TOKEN.findall(op.offset)
        return names
    if isinstance(op, (VectorOperand, ParenthesizedOperand)):
        out: list[str] = []
        for e in op.elements:
            out.extend(operand_reg_names(e))
        return out
    if isinstance(op, NegatedOperand):
        return operand_reg_names(op.operand)
    if isinstance(op, PipeOperand):
        return operand_reg_names(op.left) + operand_reg_names(op.right)
    return []


def is_special_reg(name: str) -> bool:
    """True for hardware/special registers that are never allocated."""
    if not name.startswith("%"):
        return True
    return any(
        s in name for s in ("tid", "ctaid", "ntid", "nctaid", "laneid",
                            "clock", "cluster", "smid", "warpid", "lanemask",
                            "gridid", "%p")
    ) and not name[1:2].isdigit()


def instr_def_uses(instr: Instruction):
    """(scalar_def | None, complex_defs, uses, predicated) for an instruction.

    scalar_def   — plain scalar destination register (write-only), else None.
    complex_defs — registers written through a vector/paren destination.
    uses         — every register read (operands[1:], plus operand[0]'s regs
                   when it is an address/other non-destination operand, plus
                   the predicate register).
    predicated   — True if the write is guarded by a predicate (conditional).
    """
    if not instr.operands:
        uses = []
        if instr.predicate is not None:
            reg = getattr(instr.predicate, "register", None)
            if isinstance(reg, str):
                uses.append(reg)
        return None, [], uses, instr.predicate is not None
    op0 = instr.operands[0]
    scalar_def = None
    complex_defs: list[str] = []
    op0_uses: list[str] = []
    if isinstance(op0, RegisterOperand):
        scalar_def = op0.name
    elif isinstance(op0, (VectorOperand, ParenthesizedOperand)):
        complex_defs = operand_reg_names(op0)
    else:
        op0_uses = operand_reg_names(op0)
    uses = list(op0_uses)
    for op in instr.operands[1:]:
        uses.extend(operand_reg_names(op))
    if instr.predicate is not None:
        reg = getattr(instr.predicate, "register", None)
        if isinstance(reg, str):
            uses.append(reg)
    return scalar_def, complex_defs, uses, instr.predicate is not None


@dataclass
class BasicBlock:
    id: int
    start: int                       # first statement index (inclusive)
    end: int                         # last statement index (exclusive)
    label: str | None = None         # label name if the block begins with one
    insts: list[int] = field(default_factory=list)   # indices of Instructions
    succs: list[int] = field(default_factory=list)
    preds: list[int] = field(default_factory=list)


@dataclass
class CFG:
    blocks: list[BasicBlock]
    label_to_block: dict[str, int]
    statements: list[Statement]

    def block_of_index(self, idx: int) -> int:
        for b in self.blocks:
            if b.start <= idx < b.end:
                return b.id
        return -1


def build_cfg(statements: list[Statement]) -> CFG:
    """Partition the flat Statement list into basic blocks and link them."""
    n = len(statements)
    # 1. find block boundaries. A new block starts at index 0, at every Label,
    #    and right after every terminator.
    starts = {0}
    for i, s in enumerate(statements):
        if isinstance(s, Label):
            starts.add(i)
        elif isinstance(s, Instruction) and s.opcode in _TERMINATORS:
            if i + 1 < n:
                starts.add(i + 1)
    boundaries = sorted(starts)

    blocks: list[BasicBlock] = []
    label_to_block: dict[str, int] = {}
    for bi, start in enumerate(boundaries):
        end = boundaries[bi + 1] if bi + 1 < len(boundaries) else n
        blk = BasicBlock(id=bi, start=start, end=end)
        for k in range(start, end):
            s = statements[k]
            if isinstance(s, Label) and blk.label is None:
                blk.label = s.name
                label_to_block[s.name] = bi
            elif isinstance(s, Instruction):
                blk.insts.append(k)
        blocks.append(blk)

    # 2. link successors.
    for blk in blocks:
        term = None
        for k in reversed(blk.insts):
            term = statements[k]
            break
        last_is_term = (
            term is not None and term.opcode in _TERMINATORS
        )
        if term is not None and term.opcode == "bra":
            tgt = _branch_target(term)
            if tgt is not None and tgt in label_to_block:
                blk.succs.append(label_to_block[tgt])
            if term.predicate is not None and blk.id + 1 < len(blocks):
                blk.succs.append(blk.id + 1)  # conditional: fallthrough too
        elif term is not None and term.opcode in ("ret", "exit", "trap"):
            pass  # no successors
        else:
            if blk.id + 1 < len(blocks):
                blk.succs.append(blk.id + 1)  # fallthrough
        # de-dup while preserving order
        seen: set[int] = set()
        blk.succs = [x for x in blk.succs if not (x in seen or seen.add(x))]

    for blk in blocks:
        for s in blk.succs:
            blocks[s].preds.append(blk.id)

    return CFG(blocks=blocks, label_to_block=label_to_block, statements=statements)


def _branch_target(instr: Instruction) -> str | None:
    for op in instr.operands:
        if isinstance(op, LabelOperand):
            return op.name
    return None


def compute_liveness(cfg: CFG):
    """Backward dataflow. Returns (live_in, live_out) per block id: sets of
    allocatable register names live at block entry / exit.

    A conditionally-written register (predicated def) does NOT kill — the old
    value may flow through — so it is added to the block's use set and left out
    of the kill set (a safe over-approximation of liveness).
    """
    stmts = cfg.statements
    use: dict[int, set[str]] = {}
    kill: dict[int, set[str]] = {}
    for blk in cfg.blocks:
        u: set[str] = set()
        k: set[str] = set()
        for idx in blk.insts:
            instr = stmts[idx]
            sdef, cdefs, uses, pred = instr_def_uses(instr)
            for r in uses:
                if not is_special_reg(r) and r not in k:
                    u.add(r)
            if pred:
                # conditional write: reg read-ish, do not kill
                if sdef is not None and not is_special_reg(sdef) and sdef not in k:
                    u.add(sdef)
            else:
                if sdef is not None and not is_special_reg(sdef):
                    k.add(sdef)
                for r in cdefs:
                    if not is_special_reg(r):
                        k.add(r)
        use[blk.id] = u
        kill[blk.id] = k

    live_in: dict[int, set[str]] = {b.id: set() for b in cfg.blocks}
    live_out: dict[int, set[str]] = {b.id: set() for b in cfg.blocks}
    changed = True
    while changed:
        changed = False
        for blk in reversed(cfg.blocks):
            out: set[str] = set()
            for s in blk.succs:
                out |= live_in[s]
            inn = use[blk.id] | (out - kill[blk.id])
            if out != live_out[blk.id] or inn != live_in[blk.id]:
                live_out[blk.id] = out
                live_in[blk.id] = inn
                changed = True
    return live_in, live_out


def liveness_at_instructions(cfg: CFG, live_out: dict[int, set[str]]):
    """Per-instruction live-out sets (register names live immediately AFTER
    each instruction), keyed by statement index. Enables DCE and interference.
    """
    stmts = cfg.statements
    result: dict[int, set[str]] = {}
    for blk in cfg.blocks:
        live = set(live_out[blk.id])
        for idx in reversed(blk.insts):
            result[idx] = set(live)
            instr = stmts[idx]
            sdef, cdefs, uses, pred = instr_def_uses(instr)
            if not pred:
                if sdef is not None and not is_special_reg(sdef):
                    live.discard(sdef)
                for r in cdefs:
                    live.discard(r)
            for r in uses:
                if not is_special_reg(r):
                    live.add(r)
    return result


def compute_dominators(cfg: CFG) -> dict[int, set[int]]:
    """Dominator sets per block id (block 0 is the entry)."""
    ids = [b.id for b in cfg.blocks]
    entry = cfg.blocks[0].id if cfg.blocks else None
    dom: dict[int, set[int]] = {i: set(ids) for i in ids}
    if entry is None:
        return dom
    dom[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for blk in cfg.blocks:
            if blk.id == entry:
                continue
            new = set(ids)
            for p in blk.preds:
                new &= dom[p]
            new.add(blk.id)
            if new != dom[blk.id]:
                dom[blk.id] = new
                changed = True
    return dom


def immediate_dominators(cfg: CFG, dom: dict[int, set[int]]) -> dict[int, int]:
    """idom[b] = the unique strict dominator of b closest to b (entry -> -1)."""
    idom: dict[int, int] = {}
    entry = cfg.blocks[0].id if cfg.blocks else -1
    for blk in cfg.blocks:
        if blk.id == entry:
            idom[blk.id] = -1
            continue
        strict = dom[blk.id] - {blk.id}
        chosen = -1
        for d in strict:
            # d is the idom iff it is dominated by every other strict dominator
            if all(d == o or d in dom[o] for o in strict):
                # closest: idom is dominated by all others -> pick the one whose
                # dominator set is largest (nearest to b)
                if chosen == -1 or len(dom[d]) > len(dom[chosen]):
                    chosen = d
        idom[blk.id] = chosen
    return idom


def dominance_frontiers(cfg: CFG, idom: dict[int, int]) -> dict[int, set[int]]:
    """Cytron dominance frontiers, used for SSA phi placement."""
    df: dict[int, set[int]] = {b.id: set() for b in cfg.blocks}
    for blk in cfg.blocks:
        if len(blk.preds) < 2:
            continue
        for p in blk.preds:
            runner = p
            while runner != -1 and runner != idom[blk.id]:
                df[runner].add(blk.id)
                runner = idom.get(runner, -1)
    return df
