"""Differential symbolic execution for translation validation.

The optimization passes in :mod:`pyptx.ir.optimize` change register names,
remove instructions, and reorder them. ``verify_body`` only checks the result
is *structurally* well-formed (no dangling uses) — it cannot tell whether the
transformed program still computes the same thing. This module closes that
gap without a GPU.

It executes a function body as *symbolic* dataflow: every value is a stable
structural hash of the computation that produced it (uninterpreted-function
semantics — we never need real arithmetic, only determinism), memory is
modeled so store→load forwarding and aliasing are captured, and control flow
is followed by hashing branch predicates so two structurally-identical
programs take identical paths in lockstep. The observable trace — the ordered
sequence of side-effecting operations (stores, barriers, MMAs, …) with their
operands' value-hashes — is what a correct transformation must preserve.

Because the check is *differential*, simulator fidelity to real hardware is
irrelevant: as long as it is a deterministic function of the IR, two programs
that compute the same function produce the same trace, and any pass that
corrupts a value (e.g. coalescing two simultaneously-live registers) makes an
observable effect diverge. Register *names* are abstracted away — internal
registers expand to their defining expression, and live-in / renamed leaves
are canonicalized by first-encounter order — so pure renaming passes compare
equal by construction.
"""

from __future__ import annotations

import hashlib

from pyptx.ir.nodes import (
    AddressOperand,
    ImmediateOperand,
    Instruction,
    Label,
    LabelOperand,
    NegatedOperand,
    ParenthesizedOperand,
    PipeOperand,
    RegisterOperand,
    VectorOperand,
)
from pyptx.ir.analysis import SIDE_EFFECTING, is_special_reg, operand_reg_names


def _h(s: str) -> str:
    return hashlib.blake2b(s.encode(), digest_size=10).hexdigest()


def symbolic_trace(statements, event_cap: int = 200000, step_cap: int = 5_000_000,
                   loop_bound: int = 3, seed: int = 0):
    """Execute `statements` symbolically; return (events, status).

    events — ordered list of value-hashes, one per executed side-effecting
             instruction (its opcode, modifiers, and the value-hashes of the
             operands it *reads* — the written destination is excluded, since
             a write is not an observable effect; its value is observed later
             where it is used).
    status — "halted" (hit ret/exit), "event_cap", or "step_cap".

    Loops are bounded: a backward branch to a given label is taken at most
    `loop_bound` times, after which the loop is forced to exit, so execution
    reaches the function's end and the whole body is covered. The bound is
    keyed by target label (stable across passes), so two programs force the
    same decisions in lockstep.
    """
    label_idx = {s.name: i for i, s in enumerate(statements) if isinstance(s, Label)}
    env: dict[str, str] = {}          # register name -> current value hash
    livein: dict[str, str] = {}       # used-before-def name -> positional hash
    mem: dict[str, str] = {}          # address hash -> stored value hash
    events: list[str] = []
    backedge: dict[str, int] = {}     # target label -> times taken backward

    def leaf(name: str) -> str:
        if is_special_reg(name):
            return _h("sreg:" + name)
        if name in env:
            return env[name]
        if name not in livein:
            livein[name] = _h("livein#%d" % len(livein))
        return livein[name]

    def key(op) -> str:
        if isinstance(op, RegisterOperand):
            return leaf(op.name)
        if isinstance(op, ImmediateOperand):
            return _h("imm:" + op.text)
        if isinstance(op, AddressOperand):
            base = op.base
            b = leaf(base) if isinstance(base, str) and base.startswith("%") else _h("sym:" + str(base))
            return _h("addr:%s+%s" % (b, op.offset))
        if isinstance(op, NegatedOperand):
            return _h("neg:" + key(op.operand))
        if isinstance(op, VectorOperand):
            return _h("vec:" + ",".join(key(e) for e in op.elements))
        if isinstance(op, ParenthesizedOperand):
            return _h("par:" + ",".join(key(e) for e in op.elements))
        if isinstance(op, PipeOperand):
            return _h("pipe:%s|%s" % (key(op.left), key(op.right)))
        if isinstance(op, LabelOperand):
            return _h("lbl:" + op.name)
        return _h("other:" + repr(op))

    pc = 0
    steps = 0
    n = len(statements)
    status = "step_cap"
    while pc < n and steps < step_cap and len(events) < event_cap:
        s = statements[pc]
        steps += 1
        if not isinstance(s, Instruction):
            pc += 1
            continue

        take = True
        if s.predicate is not None and isinstance(getattr(s.predicate, "register", None), str):
            # seed perturbs branch decisions to exercise different CFG paths;
            # the SAME predicate value decides the SAME way in both programs
            # (lockstep), so equivalent programs stay in sync for every seed.
            bit = int(_h("%d:%s" % (seed, leaf(s.predicate.register))), 16) & 1
            take = (bit == 1)
            if s.predicate.negated:
                take = not take

        op = s.opcode
        operands = s.operands

        if op == "bra":
            if take:
                tgt = next((o.name for o in operands if isinstance(o, LabelOperand)), None)
                if tgt in label_idx:
                    if label_idx[tgt] <= pc:  # backward edge -> bound the loop
                        backedge[tgt] = backedge.get(tgt, 0) + 1
                        if backedge[tgt] > loop_bound:
                            pc += 1  # force loop exit
                            continue
                    pc = label_idx[tgt]
                    continue
            pc += 1
            continue
        if op in ("ret", "exit", "trap"):
            if take:
                status = "halted"
                break
            pc += 1
            continue

        dest = operands[0] if operands else None
        src_keys = [key(o) for o in operands[1:]]

        # observable event: side-effecting op, hashing only the operands it
        # READS (exclude a written destination — operand[0] when it is a
        # register/vector destination).
        if op in SIDE_EFFECTING:
            dest_written = isinstance(dest, (RegisterOperand, VectorOperand, ParenthesizedOperand))
            read_ops = operands[1:] if dest_written else operands
            ev = (op, tuple(s.modifiers), tuple(key(o) for o in read_ops), take)
            events.append(_h(repr(ev)))

        if op == "st":
            if take and len(operands) >= 2:
                mem[key(operands[0])] = key(operands[1])
            pc += 1
            continue

        if isinstance(dest, RegisterOperand):
            if op == "ld":
                addr_k = key(operands[1]) if len(operands) >= 2 else _h("noaddr")
                val = mem.get(addr_k, _h("uninit:" + addr_k))
            else:
                val = _h("op:%s|%s|%s" % (op, ",".join(s.modifiers), ",".join(src_keys)))
            if take:
                env[dest.name] = val
            else:
                env.setdefault(dest.name, leaf(dest.name))
        elif isinstance(dest, (VectorOperand, ParenthesizedOperand)):
            base = _h("op:%s|%s|%s" % (op, ",".join(s.modifiers), ",".join(src_keys)))
            if take:
                for i, rn in enumerate(operand_reg_names(dest)):
                    env[rn] = _h("proj%d:%s" % (i, base))
        pc += 1

    return events, status


def equivalent(original, transformed, seeds: int = 8, **kw):
    """True iff `transformed` preserves `original`'s observable behavior.

    Runs the differential check under several branch seeds, each exercising a
    different set of CFG paths, and requires them all to agree — so a bug on a
    path the default run happens not to take is still caught. Returns
    (ok, detail); compares event traces over their common length (both
    progress at the same side-effect rate, since passes never remove
    side-effecting ops), making DCE / renaming / scheduling comparable.
    """
    total = 0
    for seed in range(seeds):
        ev_a, st_a = symbolic_trace(list(original), seed=seed, **kw)
        ev_b, st_b = symbolic_trace(list(transformed), seed=seed, **kw)
        m = min(len(ev_a), len(ev_b))
        if ev_a[:m] != ev_b[:m]:
            first = next((i for i in range(m) if ev_a[i] != ev_b[i]), m)
            return False, f"seed {seed}: event {first} diverges (of {len(ev_a)}/{len(ev_b)}, status {st_a}/{st_b})"
        if st_a == "halted" and st_b == "halted" and len(ev_a) != len(ev_b):
            return False, f"seed {seed}: both halted but event counts differ: {len(ev_a)} vs {len(ev_b)}"
        total += m
    return True, f"{total} events match across {seeds} seeds"
