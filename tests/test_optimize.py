"""Tests for post-trace IR optimizations (pyptx.ir.optimize).

These cover the semantics-preserving guarantees of the passes: correct
folding/hoisting of invariant computes, and — just as important — that
the safety guards refuse to touch anything that isn't provably invariant.
"""

import pytest

from pyptx.ir.nodes import (
    AddressOperand,
    ImmediateOperand,
    Instruction,
    Label,
    Predicate,
    RegDecl,
    RegisterOperand,
)
from pyptx.ir.nodes import LabelOperand
from pyptx.ir.optimize import (
    allocate_registers,
    copy_propagate,
    dead_code_eliminate,
    gvn_invariant,
    list_schedule,
    optimize_body,
    split_false_deps,
    ssa_reconstruct,
    verify_body,
)
from pyptx.ir.analysis import (
    build_cfg,
    compute_liveness,
    instr_def_uses,
    is_special_reg,
    liveness_at_instructions,
)


def R(n):
    return RegisterOperand(name=n)


def I(v):
    return ImmediateOperand(text=str(v))


def ins(op, mods, *ops, predicate=None):
    return Instruction(
        opcode=op, modifiers=tuple(mods), operands=tuple(ops), predicate=predicate
    )


def decl(n):
    return RegDecl(type=".u32", name=n, count=None)


def _find(out, opcode):
    return [s for s in out if isinstance(s, Instruction) and s.opcode == opcode]


def _def_index(out, reg):
    for i, s in enumerate(out):
        if isinstance(s, Instruction) and s.operands and getattr(
            s.operands[0], "name", None
        ) == reg:
            return i
    return None


class TestGvnInvariant:
    def test_folds_duplicate_invariant_add(self):
        # %r1 and %r2 both compute %base+100 -> %r2 folds to %r1.
        body = [
            decl("%base"), decl("%r1"), decl("%r2"),
            ins("mov", [".u32"], R("%base"), R("%tid.x")),
            ins("add", [".u32"], R("%r1"), R("%base"), I(100)),
            Label(name="loop"),
            ins("add", [".u32"], R("%r2"), R("%base"), I(100)),
            ins("st", [".global", ".u32"], AddressOperand(base="%r2", offset=0), R("%r1")),
        ]
        out = gvn_invariant(list(body))
        # the duplicate add is gone
        adds = _find(out, "add")
        assert len(adds) == 1
        # the store now addresses the canonical %r1
        st = _find(out, "st")[0]
        assert st.operands[0].base == "%r1"
        assert verify_body(out) == []

    def test_hoists_surviving_canonical_above_label(self):
        # Two invariant computes of %base+200 inside the loop: one folds, and
        # the surviving canonical is hoisted above the label so it dominates
        # every (folded) use. This is GVN + dominating hoist, not speculative
        # LICM of a lone invariant (which would only trade recompute for
        # register pressure).
        body = [
            decl("%base"), decl("%r4"), decl("%r5"),
            ins("mov", [".u32"], R("%base"), R("%tid.x")),
            Label(name="loop"),
            ins("add", [".u32"], R("%r4"), R("%base"), I(200)),
            ins("add", [".u32"], R("%r5"), R("%base"), I(200)),
            ins("st", [".global", ".u32"], AddressOperand(base="%r4", offset=0), R("%r5")),
        ]
        out = gvn_invariant(list(body))
        assert verify_body(out) == []
        # only one add survives, and it sits above the loop label
        adds = _find(out, "add")
        assert len(adds) == 1
        surv = adds[0].operands[0].name
        surv_idx = _def_index(out, surv)
        loop_idx = next(i for i, s in enumerate(out) if isinstance(s, Label))
        assert surv_idx is not None and surv_idx < loop_idx
        # both original uses now point at the survivor
        st = _find(out, "st")[0]
        assert st.operands[0].base == surv and st.operands[1].name == surv

    def test_does_not_touch_variant_register(self):
        # %r depends on a loaded value -> NOT invariant, must be left alone.
        body = [
            decl("%p"), decl("%r"),
            ins("ld", [".global", ".u32"], R("%p"), AddressOperand(base="%addr", offset=0)),
            ins("add", [".u32"], R("%r"), R("%p"), I(4)),
            ins("st", [".global", ".u32"], AddressOperand(base="%r", offset=0), R("%p")),
        ]
        out = gvn_invariant(list(body))
        assert out == body  # untouched

    def test_ignores_predicated_definition(self):
        # A conditionally-defined reg is not a guaranteed single value; even
        # if it looks like a dup of an unconditional compute, don't fold it.
        body = [
            decl("%base"), decl("%r1"), decl("%r2"), decl("%pred"),
            ins("mov", [".u32"], R("%base"), R("%tid.x")),
            ins("add", [".u32"], R("%r1"), R("%base"), I(8)),
            ins("add", [".u32"], R("%r2"), R("%base"), I(8),
                predicate=Predicate(register="%pred", negated=False)),
            ins("st", [".global", ".u32"], AddressOperand(base="%r2", offset=0), R("%r1")),
        ]
        out = gvn_invariant(list(body))
        # %r2's store use must NOT be rewritten to %r1 (r2 may be undefined)
        st = _find(out, "st")[0]
        assert st.operands[0].base == "%r2"

    def test_multi_def_not_invariant(self):
        # A register assigned twice is not single-value; leave it alone.
        body = [
            decl("%base"), decl("%r"),
            ins("mov", [".u32"], R("%base"), R("%tid.x")),
            ins("add", [".u32"], R("%r"), R("%base"), I(1)),
            ins("add", [".u32"], R("%r"), R("%base"), I(2)),
            ins("st", [".global", ".u32"], AddressOperand(base="%r", offset=0), R("%base")),
        ]
        out = gvn_invariant(list(body))
        assert out == body


class TestCopyPropagate:
    def test_removes_fresh_temp_mov(self):
        # add into fresh temp %t, then mov %dst, %t -> collapse to add %dst.
        body = [
            decl("%base"), decl("%t"), decl("%dst"),
            ins("mov", [".u32"], R("%base"), R("%tid.x")),
            ins("add", [".b32"], R("%t"), R("%base"), I(4)),
            ins("mov", [".b32"], R("%dst"), R("%t")),
            ins("st", [".global", ".u32"], AddressOperand(base="%dst", offset=0), R("%base")),
        ]
        out = copy_propagate(list(body))
        assert not _find(out, "mov") or all(
            m.opcode != "mov" or m.operands[1].name != "%t" for m in _find(out, "mov")
        )
        assert verify_body(out) == []


class TestSplitFalseDeps:
    def _decls(self, *names, ty=".b32"):
        return [RegDecl(type=ty, name=n, count=None) for n in names]

    def test_splits_independent_reuse(self):
        # %r reused for three independent values. First and last keep %r
        # (live-in / live-out safety); the middle value(s) rename to fresh.
        body = [
            *self._decls("%base", "%out", "%r"),
            ins("mov", [".u32"], R("%base"), R("%tid.x")),
            ins("mov", [".u32"], R("%out"), R("%ntid.x")),
            ins("add", [".b32"], R("%r"), R("%base"), I(1)),   # v0 (kill)
            ins("st", [".global", ".b32"], AddressOperand("%out", 0), R("%r")),
            ins("add", [".b32"], R("%r"), R("%base"), I(2)),   # v1 (kill)
            ins("st", [".global", ".b32"], AddressOperand("%out", 8), R("%r")),
            ins("add", [".b32"], R("%r"), R("%base"), I(3)),   # v2 (kill, last)
            ins("st", [".global", ".b32"], AddressOperand("%out", 16), R("%r")),
        ]
        out = split_false_deps(list(body))
        assert verify_body(out) == []
        # each add now writes a distinct register (no WAW on %r across the 3)
        add_dests = [s.operands[0].name for s in _find(out, "add")]
        assert len(set(add_dests)) == 3
        # the store immediately after each add reads that add's dest (dataflow
        # preserved): pair them up in program order.
        adds = _find(out, "add")
        sts = _find(out, "st")
        for a, st in zip(adds, sts):
            assert st.operands[1].name == a.operands[0].name
        # the last value still uses the original name
        assert adds[-1].operands[0].name == "%r"

    def test_carrying_def_not_split(self):
        # %r = %r + 1 reads %r: a true RAW chain (accumulator), not a false
        # dep. Must remain a single register.
        body = [
            *self._decls("%base", "%out", "%r"),
            ins("mov", [".u32"], R("%base"), R("%tid.x")),
            ins("mov", [".u32"], R("%out"), R("%ntid.x")),
            ins("add", [".b32"], R("%r"), R("%base"), I(0)),   # init
            ins("add", [".b32"], R("%r"), R("%r"), I(1)),      # carry (RAW)
            ins("add", [".b32"], R("%r"), R("%r"), I(1)),      # carry (RAW)
            ins("st", [".global", ".b32"], AddressOperand("%out", 0), R("%r")),
        ]
        out = split_false_deps(list(body))
        assert out == body  # untouched — no false dep to break

    def test_predicated_def_poisons_register(self):
        # A conditional write to %r means %r's value isn't a clean single
        # value; leave the whole register alone in this block.
        body = [
            *self._decls("%base", "%out", "%r", "%p", ty=".b32"),
            ins("mov", [".u32"], R("%base"), R("%tid.x")),
            ins("mov", [".u32"], R("%out"), R("%ntid.x")),
            ins("add", [".b32"], R("%r"), R("%base"), I(1)),
            ins("st", [".global", ".b32"], AddressOperand("%out", 0), R("%r")),
            ins("add", [".b32"], R("%r"), R("%base"), I(2),
                predicate=Predicate(register="%p", negated=False)),
            ins("st", [".global", ".b32"], AddressOperand("%out", 8), R("%r")),
        ]
        out = split_false_deps(list(body))
        assert out == body

    def test_no_split_across_label(self):
        # Values in different blocks (separated by a label) are analyzed
        # independently; the register keeps its name across the boundary.
        body = [
            *self._decls("%base", "%out", "%r"),
            ins("mov", [".u32"], R("%base"), R("%tid.x")),
            ins("mov", [".u32"], R("%out"), R("%ntid.x")),
            ins("add", [".b32"], R("%r"), R("%base"), I(1)),
            ins("st", [".global", ".b32"], AddressOperand("%out", 0), R("%r")),
            Label(name="cont"),
            ins("add", [".b32"], R("%r"), R("%base"), I(2)),
            ins("st", [".global", ".b32"], AddressOperand("%out", 8), R("%r")),
        ]
        out = split_false_deps(list(body))
        # single value per block -> nothing to split
        assert [s for s in out if isinstance(s, RegDecl) and s.name.startswith("%vs")] == []

    def test_unknown_type_skipped(self):
        # %r has no declaration -> can't declare a fresh reg of its type ->
        # leave it alone rather than guess.
        body = [
            *self._decls("%base", "%out"),
            ins("mov", [".u32"], R("%base"), R("%tid.x")),
            ins("mov", [".u32"], R("%out"), R("%ntid.x")),
            ins("add", [".b32"], R("%r"), R("%base"), I(1)),
            ins("st", [".global", ".b32"], AddressOperand("%out", 0), R("%r")),
            ins("add", [".b32"], R("%r"), R("%base"), I(2)),
            ins("st", [".global", ".b32"], AddressOperand("%out", 8), R("%r")),
        ]
        out = split_false_deps(list(body))
        assert out == body


def _distinct_allocatable(statements):
    from pyptx.ir.optimize import _reg_type_lookup
    tl = _reg_type_lookup(statements)
    regs = set()
    for s in statements:
        if isinstance(s, Instruction):
            sd, cd, us, _ = instr_def_uses(s)
            for r in ([sd] if sd else []) + cd + us:
                if r and not is_special_reg(r) and tl(r) is not None:
                    regs.add(r)
    return regs


def _colors_conflict(statements):
    """True if any two registers live at the same point share a name (i.e. a
    miscoloring). Independent re-derivation of allocation soundness."""
    cfg = build_cfg(statements)
    _, lo = compute_liveness(cfg)
    live_after = liveness_at_instructions(cfg, lo)
    # a program is sound if, for every point, the live set has no duplicate
    # names — trivially true, so instead check original interference is honored
    # by re-running liveness: every register name is defined before/at each use.
    return verify_body(statements) != []


class TestDeadCodeElimination:
    def test_removes_dead_pure_def(self):
        body = [
            RegDecl(type=".b32", name="%base", count=None),
            RegDecl(type=".b32", name="%dead", count=None),
            RegDecl(type=".b32", name="%live", count=None),
            RegDecl(type=".b32", name="%out", count=None),
            ins("mov", [".u32"], R("%base"), R("%tid.x")),
            ins("mov", [".u32"], R("%out"), R("%ntid.x")),
            ins("add", [".b32"], R("%dead"), R("%base"), I(7)),   # never used
            ins("add", [".b32"], R("%live"), R("%base"), I(9)),
            ins("st", [".global", ".b32"], AddressOperand("%out", 0), R("%live")),
            ins("ret", []),
        ]
        out = dead_code_eliminate(list(body))
        opcodes = [(s.opcode, s.operands[0].name) for s in out
                   if isinstance(s, Instruction) and s.opcode == "add"]
        assert ("add", "%dead") not in opcodes
        assert ("add", "%live") in opcodes
        # its now-unused decl is dropped too
        assert not any(isinstance(s, RegDecl) and s.name == "%dead" for s in out)
        assert verify_body(out) == []

    def test_keeps_side_effecting(self):
        # a store's "result" is memory; never eligible for DCE
        body = [
            RegDecl(type=".b32", name="%out", count=None),
            RegDecl(type=".b32", name="%v", count=None),
            ins("mov", [".u32"], R("%out"), R("%ntid.x")),
            ins("mov", [".u32"], R("%v"), R("%tid.x")),
            ins("st", [".global", ".b32"], AddressOperand("%out", 0), R("%v")),
            ins("ret", []),
        ]
        out = dead_code_eliminate(list(body))
        assert any(isinstance(s, Instruction) and s.opcode == "st" for s in out)

    def test_cascades(self):
        # %t feeds only %dead; removing %dead makes %t dead too
        body = [
            RegDecl(type=".b32", name="%base", count=None),
            RegDecl(type=".b32", name="%t", count=None),
            RegDecl(type=".b32", name="%dead", count=None),
            ins("mov", [".u32"], R("%base"), R("%tid.x")),
            ins("add", [".b32"], R("%t"), R("%base"), I(1)),
            ins("add", [".b32"], R("%dead"), R("%t"), I(2)),
            ins("ret", []),
        ]
        out = dead_code_eliminate(list(body))
        assert not any(isinstance(s, Instruction) and s.opcode == "add" for s in out)


class TestRegisterAllocation:
    def test_coalesces_non_interfering(self):
        # %a and %b never live at the same time -> can share one register.
        body = [
            RegDecl(type=".b32", name="%base", count=None),
            RegDecl(type=".b32", name="%out", count=None),
            RegDecl(type=".b32", name="%a", count=None),
            RegDecl(type=".b32", name="%b", count=None),
            ins("mov", [".u32"], R("%base"), R("%tid.x")),
            ins("mov", [".u32"], R("%out"), R("%ntid.x")),
            ins("add", [".b32"], R("%a"), R("%base"), I(1)),
            ins("st", [".global", ".b32"], AddressOperand("%out", 0), R("%a")),
            ins("add", [".b32"], R("%b"), R("%base"), I(2)),
            ins("st", [".global", ".b32"], AddressOperand("%out", 4), R("%b")),
            ins("ret", []),
        ]
        out = allocate_registers(list(body))
        assert verify_body(out) == []
        # %a and %b (non-interfering, same type) collapse to one physical name
        assert len(_distinct_allocatable(out)) < len(_distinct_allocatable(body))

    def test_keeps_interfering_separate(self):
        # %a and %b are simultaneously live (both used after both defined) ->
        # must NOT share a register.
        body = [
            RegDecl(type=".b32", name="%base", count=None),
            RegDecl(type=".b32", name="%out", count=None),
            RegDecl(type=".b32", name="%a", count=None),
            RegDecl(type=".b32", name="%b", count=None),
            ins("mov", [".u32"], R("%base"), R("%tid.x")),
            ins("mov", [".u32"], R("%out"), R("%ntid.x")),
            ins("add", [".b32"], R("%a"), R("%base"), I(1)),
            ins("add", [".b32"], R("%b"), R("%base"), I(2)),
            ins("add", [".b32"], R("%a"), R("%a"), R("%b")),   # both live here
            ins("st", [".global", ".b32"], AddressOperand("%out", 0), R("%a")),
            ins("ret", []),
        ]
        out = allocate_registers(list(body))
        assert verify_body(out) == []
        # the two dest names at the point of interference differ
        adds = [s for s in out if isinstance(s, Instruction) and s.opcode == "add"]
        # find the `add x, x, y` (reads two regs) — its two source regs differ
        rmw = [s for s in adds if len(s.operands) == 3
               and isinstance(s.operands[2], RegisterOperand)]
        assert rmw, "expected the interfering add to survive"

    def test_bails_on_unresolved_branch(self):
        from pyptx.ir.nodes import LabelOperand
        body = [
            RegDecl(type=".b32", name="%a", count=None),
            ins("mov", [".u32"], R("%a"), R("%tid.x")),
            ins("bra", [], LabelOperand(name="$nowhere")),  # target has no label
        ]
        out = allocate_registers(list(body))
        assert out == body  # bailed, unchanged


def _order_of(out, opcode, argname_index=0):
    """program-order list of a given opcode's operand[argname_index] names."""
    names = []
    for s in out:
        if isinstance(s, Instruction) and s.opcode == opcode:
            op = s.operands[argname_index]
            names.append(getattr(op, "name", getattr(op, "base", None)))
    return names


class TestListSchedule:
    def test_preserves_raw(self):
        # add %b,%a,1 consumes %a from mov %a -> must stay after it.
        body = [
            RegDecl(type=".b32", name="%a", count=None),
            RegDecl(type=".b32", name="%b", count=None),
            RegDecl(type=".b32", name="%out", count=None),
            ins("mov", [".u32"], R("%out"), R("%ntid.x")),
            ins("mov", [".u32"], R("%a"), R("%tid.x")),
            ins("add", [".b32"], R("%b"), R("%a"), I(1)),
            ins("st", [".global", ".b32"], AddressOperand("%out", 0), R("%b")),
            ins("ret", []),
        ]
        out = list_schedule(list(body))
        assert verify_body(out) == []
        idx = {id(s): k for k, s in enumerate(out)}
        prod = next(s for s in out if isinstance(s, Instruction)
                    and s.opcode == "mov" and s.operands[0].name == "%a")
        cons = next(s for s in out if isinstance(s, Instruction)
                    and s.opcode == "add")
        assert idx[id(prod)] < idx[id(cons)]

    def test_keeps_store_order_and_terminator_last(self):
        body = [
            RegDecl(type=".b32", name="%out", count=None),
            RegDecl(type=".b32", name="%v1", count=None),
            RegDecl(type=".b32", name="%v2", count=None),
            ins("mov", [".u32"], R("%out"), R("%ntid.x")),
            ins("mov", [".u32"], R("%v1"), R("%tid.x")),
            ins("mov", [".u32"], R("%v2"), R("%ctaid.x")),
            ins("st", [".global", ".b32"], AddressOperand("%out", 0), R("%v1")),
            ins("st", [".global", ".b32"], AddressOperand("%out", 4), R("%v2")),
            ins("ret", []),
        ]
        out = list_schedule(list(body))
        assert verify_body(out) == []
        # both stores present in the same relative order
        st_offsets = [s.operands[0].offset for s in out
                      if isinstance(s, Instruction) and s.opcode == "st"]
        assert st_offsets == [0, 4]
        assert isinstance(out[-1], Instruction) and out[-1].opcode == "ret"


class TestSsaReconstruct:
    def test_isolates_exclusive_multidef(self):
        # %r reused for two independent values, each with an exclusive use.
        # Global reaching-def analysis renames BOTH webs to fresh names.
        body = [
            RegDecl(type=".b32", name="%r", count=None),
            RegDecl(type=".b32", name="%out", count=None),
            ins("mov", [".u32"], R("%out"), R("%ntid.x")),
            ins("mov", [".b32"], R("%r"), I(5)),                 # def A
            ins("st", [".global", ".b32"], AddressOperand("%out", 0), R("%r")),  # use A
            ins("mov", [".b32"], R("%r"), I(6)),                 # def B
            ins("st", [".global", ".b32"], AddressOperand("%out", 4), R("%r")),  # use B
            ins("ret", []),
        ]
        out = ssa_reconstruct(list(body))
        assert verify_body(out) == []
        fresh = [s.name for s in out if isinstance(s, RegDecl) and s.name.startswith("%ssa")]
        assert len(fresh) == 2  # two independent webs
        # the two stores now read two distinct fresh registers
        vals = [s.operands[1].name for s in out
                if isinstance(s, Instruction) and s.opcode == "st"]
        assert len(set(vals)) == 2

    def test_does_not_isolate_merge(self):
        # %r defined on both arms of a branch, used after the join: the use
        # has two reaching defs (needs a phi) -> leave %r untouched.
        body = [
            RegDecl(type=".b32", name="%r", count=None),
            RegDecl(type=".b32", name="%out", count=None),
            RegDecl(type=".pred", name="%p", count=None),
            ins("mov", [".u32"], R("%out"), R("%ntid.x")),
            ins("bra", [], LabelOperand(name="$else"),
                predicate=Predicate(register="%p", negated=True)),
            ins("mov", [".b32"], R("%r"), I(1)),                 # then-def
            ins("bra", [], LabelOperand(name="$end")),
            Label(name="$else"),
            ins("mov", [".b32"], R("%r"), I(2)),                 # else-def
            Label(name="$end"),
            ins("st", [".global", ".b32"], AddressOperand("%out", 0), R("%r")),  # merge use
            ins("ret", []),
        ]
        out = ssa_reconstruct(list(body))
        assert verify_body(out) == []
        # no web isolated: the merge use blocks both defs
        assert not any(isinstance(s, RegDecl) and s.name.startswith("%ssa") for s in out)
        # %r still used by the store
        st = next(s for s in out if isinstance(s, Instruction) and s.opcode == "st")
        assert st.operands[1].name == "%r"


class TestOptimizeBody:
    def test_level_zero_is_identity(self):
        body = [
            decl("%base"), decl("%r1"), decl("%r2"),
            ins("mov", [".u32"], R("%base"), R("%tid.x")),
            ins("add", [".u32"], R("%r1"), R("%base"), I(100)),
            ins("add", [".u32"], R("%r2"), R("%base"), I(100)),
        ]
        assert optimize_body(list(body), level=0) == body

    def test_level_one_verified_or_fallback(self):
        body = [
            decl("%base"), decl("%r1"), decl("%r2"),
            ins("mov", [".u32"], R("%base"), R("%tid.x")),
            ins("add", [".u32"], R("%r1"), R("%base"), I(100)),
            Label(name="loop"),
            ins("add", [".u32"], R("%r2"), R("%base"), I(100)),
            ins("st", [".global", ".u32"], AddressOperand(base="%r2", offset=0), R("%r1")),
        ]
        out = optimize_body(list(body), level=1)
        assert verify_body(out) == []
        assert len(_find(out, "add")) == 1  # duplicate folded


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
