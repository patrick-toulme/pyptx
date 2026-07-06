"""Tests for the differential equivalence checker (pyptx.ir.simulate) and
the correctness of the optimization passes it validates.

Two responsibilities:
  * the checker must call a correct transformation equivalent AND must catch a
    corrupting one (a checker that never fails is worthless);
  * the optimization passes must preserve observable behavior on the real
    Blackwell FA kernel — a regression guard for the three bugs differential
    validation surfaced (SSA use-before-def, SSA carrying-def corruption, and
    copy_propagate's unsound rename).
"""

from dataclasses import replace

import pytest

from pyptx.ir.nodes import (
    AddressOperand,
    ImmediateOperand,
    Instruction,
    Label,
    LabelOperand,
    Predicate,
    RegDecl,
    RegisterOperand,
)
from pyptx.ir.simulate import equivalent, symbolic_trace


def R(n):
    return RegisterOperand(name=n)


def I(v):
    return ImmediateOperand(text=str(v))


def ins(op, mods, *ops, predicate=None):
    return Instruction(opcode=op, modifiers=tuple(mods), operands=tuple(ops), predicate=predicate)


def _straightline():
    return [
        RegDecl(type=".b32", name="%a", count=None),
        RegDecl(type=".b32", name="%b", count=None),
        RegDecl(type=".b32", name="%out", count=None),
        ins("mov", [".u32"], R("%out"), R("%ntid.x")),
        ins("add", [".b32"], R("%a"), R("%tid.x"), I(1)),
        ins("mul", [".b32"], R("%b"), R("%a"), I(3)),
        ins("st", [".global", ".b32"], AddressOperand("%out", 0), R("%b")),
        ins("ret", []),
    ]


class TestChecker:
    def test_identity_is_equivalent(self):
        body = _straightline()
        ok, _ = equivalent(body, list(body))
        assert ok

    def test_pure_rename_is_equivalent(self):
        # rename %a -> %z consistently: same computation, must be equivalent
        body = _straightline()
        renamed = []
        for s in body:
            if isinstance(s, RegDecl) and s.name == "%a":
                renamed.append(replace(s, name="%z"))
            elif isinstance(s, Instruction):
                ops = tuple(R("%z") if isinstance(o, RegisterOperand) and o.name == "%a" else o
                           for o in s.operands)
                renamed.append(replace(s, operands=ops))
            else:
                renamed.append(s)
        ok, _ = equivalent(body, renamed)
        assert ok

    def test_catches_wrong_stored_value(self):
        # corrupt the value the store writes -> must be reported non-equivalent
        body = _straightline()
        bad = list(body)
        for i, s in enumerate(bad):
            if isinstance(s, Instruction) and s.opcode == "st":
                bad[i] = replace(s, operands=(s.operands[0], R("%tid.x")))
                break
        ok, detail = equivalent(body, bad)
        assert not ok, detail

    def test_catches_swapped_source(self):
        # change a source operand of the mul -> different value flows to store
        body = _straightline()
        bad = list(body)
        for i, s in enumerate(bad):
            if isinstance(s, Instruction) and s.opcode == "mul":
                bad[i] = replace(s, operands=(s.operands[0], R("%tid.x"), s.operands[2]))
                break
        ok, detail = equivalent(body, bad)
        assert not ok, detail

    def test_catches_use_before_def(self):
        # the SSA-orphan bug shape: a value read before it is ever written.
        body = _straightline()
        bad = list(body)
        # make the mul read a never-defined %ghost instead of %a
        for i, s in enumerate(bad):
            if isinstance(s, Instruction) and s.opcode == "mul":
                bad[i] = replace(s, operands=(s.operands[0], R("%ghost"), s.operands[2]))
                break
        bad.insert(0, RegDecl(type=".b32", name="%ghost", count=None))
        ok, detail = equivalent(body, bad)
        assert not ok, detail


def _fa_body(seqlen=1024):
    mod = pytest.importorskip("examples.blackwell.flash_attention_blackwell")
    m = mod.build_flash_attention_blackwell(seqlen, 4, arch="sm_100a").module()
    fn = [d for d in m.directives if type(d).__name__ == "Function"][0]
    return list(fn.body)


class TestPassesPreserveKernelBehavior:
    """Every pass and preset must preserve the FA kernel's observable trace."""

    @pytest.mark.parametrize("pass_name", ["gvn", "dce", "split", "regalloc", "ssa", "schedule"])
    def test_single_pass_equivalent(self, pass_name):
        from pyptx.ir.optimize import _pass_registry
        body = _fa_body()
        out = _pass_registry()[pass_name](list(body))
        ok, detail = equivalent(body, out, seeds=6)
        assert ok, f"{pass_name}: {detail}"

    def test_all_presets_equivalent(self):
        from pyptx.ir.optimize import optimize_body, _LEVEL_PRESETS
        body = _fa_body()
        for level, passes in _LEVEL_PRESETS.items():
            out = optimize_body(list(body), passes=list(passes))
            ok, detail = equivalent(body, out, seeds=6)
            assert ok, f"level {level} {passes}: {detail}"

    def test_ssa_introduces_no_use_before_def(self):
        # regression: SSA must not orphan a renamed value (dominance + killing-
        # def gates). Symbolically, the base kernel has no used-before-def
        # register, and SSA must not add any.
        from pyptx.ir.optimize import ssa_reconstruct
        from pyptx.ir.analysis import build_cfg, compute_liveness
        body = _fa_body()
        out = ssa_reconstruct(list(body))
        # entry live-in of a fresh %ssa register would mean use-before-def
        cfg = build_cfg(out)
        li, _ = compute_liveness(cfg)
        entry_livein = li[cfg.blocks[0].id]
        assert not any(r.startswith("%ssa") for r in entry_livein), entry_livein


class TestCopyPropRemovedFromPipeline:
    def test_not_in_registry(self):
        from pyptx.ir.optimize import _pass_registry, _LEVEL_PRESETS
        assert "copyprop" not in _pass_registry()
        for passes in _LEVEL_PRESETS.values():
            assert "copyprop" not in passes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
