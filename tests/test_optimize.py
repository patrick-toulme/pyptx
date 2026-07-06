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
from pyptx.ir.optimize import (
    copy_propagate,
    gvn_invariant,
    optimize_body,
    split_false_deps,
    verify_body,
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
