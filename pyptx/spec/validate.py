"""Spec-driven validation for PTX IR instructions.

Validation is optional and separate from parsing. The parser accepts
anything syntactically valid. This module checks parsed Instruction nodes
against the declarative spec in pyptx.spec.ptx.

The validator supports:

  * Multiple specs (overloads) per opcode. The base table in
    ``pyptx.spec.ptx`` only stores one spec per opcode (last write wins),
    so this module also maintains an overload registry where additional
    specs can be registered. Validation tries every spec for an opcode
    and reports the issues from the best-matching one.
  * Strict mode (the default). When strict mode is on, calling
    :func:`validate_or_raise` raises :class:`PtxValidationError` on any
    error-severity issue. Warnings (e.g. unknown opcodes) never raise.
  * A context manager :func:`strict` for temporary toggling.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Iterable

from pyptx.ir.nodes import Instruction
from pyptx.spec.ptx import INSTRUCTIONS, InstructionSpec, ModifierGroup


# ---------------------------------------------------------------------------
# Issues / exceptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation issue found in an instruction."""

    instruction: Instruction
    message: str
    severity: str = "error"  # "error" or "warning"

    def __str__(self) -> str:
        opcode = self.instruction.opcode + "".join(self.instruction.modifiers)
        return f"[{self.severity}] {opcode}: {self.message}"


# Backwards compatibility alias — older code (and test_spec.py) imports
# ``ValidationError`` as the dataclass type. Keep the old name pointing at
# the new dataclass so nothing breaks.
ValidationError = ValidationIssue


class PtxValidationError(Exception):
    """Raised when an instruction fails strict validation.

    Wraps a list of :class:`ValidationIssue` objects with a readable
    aggregate message that names the offending opcode, lists each issue,
    and pinpoints the user's source line if it could be determined.
    """

    def __init__(
        self,
        issues: Iterable[ValidationIssue],
        *,
        user_frame: str | None = None,
    ) -> None:
        self.issues: list[ValidationIssue] = list(issues)
        self.user_frame: str | None = user_frame
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if not self.issues:
            return "PTX validation failed (no issues recorded)"
        first = self.issues[0].instruction
        opcode_chain = first.opcode + "".join(first.modifiers)
        lines = [opcode_chain]
        for issue in self.issues:
            if issue.severity == "error":
                lines.append(f"  {issue.message}")
        for issue in self.issues:
            if issue.severity == "warning":
                lines.append(f"  [warning] {issue.message}")
        if self.user_frame:
            lines.append(f"  at {self.user_frame}")
        return "\n".join(lines)


class UnvalidatedInstructionWarning(UserWarning):
    """Emitted when an opcode has no spec to validate against.

    Silenced by default (use ``warnings.simplefilter('always',
    UnvalidatedInstructionWarning)`` in tests / debug to surface them).
    """


warnings.simplefilter("ignore", UnvalidatedInstructionWarning)


# ---------------------------------------------------------------------------
# Strict mode controls
# ---------------------------------------------------------------------------


_strict_mode: bool = True


def set_strict(enabled: bool) -> None:
    """Enable or disable strict validation globally.

    When strict mode is on (the default), :func:`validate_or_raise` raises
    :class:`PtxValidationError` on any error-severity issue. When off,
    issues are collected but no exception is raised.
    """
    global _strict_mode
    _strict_mode = bool(enabled)


def is_strict() -> bool:
    """Return whether strict validation is currently enabled."""
    return _strict_mode


@contextmanager
def strict(enabled: bool) -> Generator[None, None, None]:
    """Temporarily enable or disable strict validation.

    Usage::

        with strict(False):
            kernel(...)  # validation issues are collected, never raised
    """
    global _strict_mode
    prev = _strict_mode
    _strict_mode = bool(enabled)
    try:
        yield
    finally:
        _strict_mode = prev


# ---------------------------------------------------------------------------
# Overload registry: opcode -> list[InstructionSpec]
# ---------------------------------------------------------------------------
#
# The base table in ``pyptx.spec.ptx`` keys specs by opcode, so an opcode
# like ``cp`` (which has both a normal ``cp.async.bulk.tensor.*`` form and
# a ``cp.reduce.async.bulk.tensor.*`` form) only retains the last
# registration. This registry holds *all* specs for an opcode (including
# the one in the base table) so the validator can pick the best match.

_OVERLOADS: dict[str, list[InstructionSpec]] = {}


def register_overload(spec: InstructionSpec) -> None:
    """Register an additional spec for an opcode."""
    _OVERLOADS.setdefault(spec.opcode, []).append(spec)


def get_specs(opcode: str) -> list[InstructionSpec]:
    """Return every spec registered for ``opcode`` (overloads + base)."""
    specs: list[InstructionSpec] = []
    base = INSTRUCTIONS.get(opcode)
    if base is not None:
        specs.append(base)
    specs.extend(_OVERLOADS.get(opcode, ()))
    return specs


# ---------------------------------------------------------------------------
# Additional specs for instructions that get clobbered or are missing
# ---------------------------------------------------------------------------
#
# These are registered as overloads in this module so they coexist with
# the base table without requiring edits to ``pyptx/spec/ptx.py``.

_BIT_TYPE = ModifierGroup("type", (".b16", ".b32", ".b64"), required=True)


def _seed_overloads() -> None:
    # ---- cp.async.bulk.tensor (TMA) -------------------------------------
    # The base table currently keeps only the cp.reduce variant; register
    # the plain TMA copy as an overload so the typed cp.async.bulk wrappers
    # validate.
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("bulk", (".bulk",)),
            ModifierGroup("tensor", (".tensor",)),
            ModifierGroup("dim", (".1d", ".2d", ".3d", ".4d", ".5d")),
            ModifierGroup("cta_group", (".cta_group::1", ".cta_group::2")),
            ModifierGroup("tensor_kind", (".tile", ".im2col", ".im2col_no_offs", ".gather4")),
            ModifierGroup("dst", (".global", ".shared::cta", ".shared::cluster")),
            ModifierGroup("src", (".global", ".shared::cta", ".shared::cluster")),
            ModifierGroup("completion", (
                ".mbarrier::complete_tx::bytes",
                ".bulk_group",
            )),
            ModifierGroup("multicast", (".multicast::cluster",)),
            ModifierGroup("cache_hint", (".L2::cache_hint",)),
        ),
        operand_pattern="[dst], [src], size|tensorCoords, [mbar] [, ctaMask] [, policy]",
        min_operands=2,
        max_operands=10,
        description="Asynchronous bulk copy / TMA tensor load-store (Hopper)",
        since_version=(8, 0),
        arch="sm_90",
    ))

    # Plain cp.async (Ampere-style cp.async.cg/.ca to shared memory).
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("cache", (".ca", ".cg")),
            ModifierGroup("dst", (".shared", ".shared::cta", ".shared::cluster")),
            ModifierGroup("src", (".global",)),
            ModifierGroup("completion", (".mbarrier::complete_tx::bytes",)),
        ),
        operand_pattern="[dst], [src], byteCount",
        min_operands=2,
        max_operands=4,
        description="cp.async (Ampere) — async copy global → shared",
        since_version=(7, 0),
        arch="sm_80",
    ))

    # cp.async.bulk.commit_group — closes all preceding
    # cp.async.bulk.*.bulk_group operations into a single commit group.
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("bulk", (".bulk",), required=True),
            ModifierGroup("action", (".commit_group",), required=True),
        ),
        operand_pattern="",
        min_operands=0,
        max_operands=0,
        description="Commit pending cp.async.bulk.*.bulk_group ops",
        since_version=(8, 0),
        arch="sm_90",
    ))

    # cp.async.bulk.wait_group N  (optionally cp.async.bulk.wait_group.read)
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("bulk", (".bulk",), required=True),
            ModifierGroup("action", (".wait_group",), required=True),
            ModifierGroup("relax", (".read",)),
        ),
        operand_pattern="N",
        min_operands=1,
        max_operands=1,
        description="Wait until at most N bulk commit groups remain",
        since_version=(8, 0),
        arch="sm_90",
    ))

    # ---- Ampere plain cp.async commit/wait (no .bulk) ---------------------
    # cp.async.commit_group — closes pending cp.async.{cg,ca} into a group.
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("action", (".commit_group",), required=True),
        ),
        operand_pattern="",
        min_operands=0,
        max_operands=0,
        description="cp.async.commit_group (Ampere) — close pending cp.async into a group",
        since_version=(7, 0),
        arch="sm_80",
    ))

    # cp.async.wait_group N — wait until at most N groups remain pending.
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("action", (".wait_group",), required=True),
        ),
        operand_pattern="N",
        min_operands=1,
        max_operands=1,
        description="cp.async.wait_group N (Ampere) — wait until <= N groups pending",
        since_version=(7, 0),
        arch="sm_80",
    ))

    # cp.async.wait_all — wait for all pending cp.async to complete.
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("action", (".wait_all",), required=True),
        ),
        operand_pattern="",
        min_operands=0,
        max_operands=0,
        description="cp.async.wait_all (Ampere) — wait for all pending cp.async",
        since_version=(7, 0),
        arch="sm_80",
    ))

    # ---- tcgen05 (Blackwell) ---------------------------------------------
    # The base table holds a stripped-down tcgen05 spec; the full family
    # (alloc/dealloc/ld/st/cp/shift/commit/fence/wait/relinquish) is
    # registered here as an overload.
    register_overload(InstructionSpec(
        opcode="tcgen05",
        modifier_groups=(
            ModifierGroup("op", (
                ".mma", ".ld", ".st", ".cp", ".shift",
                ".fence", ".commit", ".wait",
                ".alloc", ".dealloc", ".relinquish_alloc_permit",
            ), required=True),
            ModifierGroup("sp", (".sp",)),
            ModifierGroup("cta_group", (".cta_group::1", ".cta_group::2")),
            ModifierGroup("kind", (
                ".kind::tf32", ".kind::f16", ".kind::i8",
                ".kind::f8f6f4", ".kind::mxf8f6f4",
                ".kind::mxf4", ".kind::mxf4nvf4",
            )),
            ModifierGroup("block_scale", (".block_scale",)),
            ModifierGroup("scale_vec_size",
                          (".scale_vec_size::2X", ".scale_vec_size::4X")),
            ModifierGroup("sync", (".sync",)),
            ModifierGroup("aligned", (".aligned",)),
            ModifierGroup("shape", (
                ".16x64b", ".16x128b", ".16x256b", ".32x32b",
            )),
            ModifierGroup("num", (
                ".x1", ".x2", ".x4", ".x8", ".x16", ".x32", ".x64", ".x128",
            )),
            ModifierGroup("direction", (".down",)),
            ModifierGroup("space", (".shared::cta", ".shared::cluster")),
            ModifierGroup("type", (".b32", ".b64")),
            ModifierGroup("completion", (
                ".mbarrier::arrive::one",
                ".multicast::cluster",
            )),
        ),
        operand_pattern="varies by op (tmem addr, descriptors, mbar, regs)",
        min_operands=0,
        max_operands=128,
        description="Fifth-generation tensor core operations (Blackwell)",
        since_version=(8, 7),
        arch="sm_100a",
    ))

    # ---- mbarrier (extended Hopper variants) -----------------------------
    # The base spec already covers .init/.arrive/.try_wait/etc.; this
    # overload allows the compound forms used on Hopper such as
    # ``mbarrier.arrive.expect_tx`` and ``mbarrier.try_wait.parity``.
    register_overload(InstructionSpec(
        opcode="mbarrier",
        modifier_groups=(
            ModifierGroup("op", (
                ".init", ".inval",
                ".arrive", ".arrive_drop",
                ".test_wait", ".try_wait",
                ".pending_count",
                ".expect_tx", ".complete_tx",
            ), required=True),
            ModifierGroup("variant", (
                ".expect_tx", ".complete_tx",
                ".noComplete", ".parity",
            )),
            ModifierGroup("sem", (".acquire", ".release", ".relaxed")),
            ModifierGroup("scope", (".cta", ".cluster")),
            ModifierGroup("space",
                          (".shared", ".shared::cta", ".shared::cluster")),
            ModifierGroup("type", (".b64",)),
        ),
        operand_pattern="varies by op",
        min_operands=0,
        max_operands=4,
        description=(
            "Memory barrier object operations (Hopper compound forms: "
            "arrive.expect_tx, try_wait.parity, ...)"
        ),
        since_version=(7, 8),
        arch="sm_90",
    ))

    # ---- barrier.cluster (Hopper) ---------------------------------------
    # The base ``barrier`` spec covers .sync/.arrive with optional .cta
    # /.cluster scope. This overload models the explicit
    # ``barrier.cluster.arrive`` and ``barrier.cluster.wait`` forms.
    register_overload(InstructionSpec(
        opcode="barrier",
        modifier_groups=(
            ModifierGroup("level", (".cluster",), required=True),
            ModifierGroup("op", (".arrive", ".wait"), required=True),
            ModifierGroup("aligned", (".aligned",)),
            ModifierGroup("release", (".release", ".acquire", ".relaxed")),
        ),
        operand_pattern="",
        min_operands=0,
        max_operands=1,
        description="Cluster barrier arrive/wait (Hopper)",
        since_version=(7, 8),
        arch="sm_90",
    ))

    # ---- bar.warp.sync ---------------------------------------------------
    register_overload(InstructionSpec(
        opcode="bar",
        modifier_groups=(
            ModifierGroup("level", (".warp",), required=True),
            ModifierGroup("op", (".sync",), required=True),
        ),
        operand_pattern="memberMask",
        min_operands=0,
        max_operands=1,
        description="Warp-level barrier sync (Volta+)",
        since_version=(6, 0),
    ))


_seed_overloads()


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------


def _hint_group_for(
    spec: InstructionSpec,
    mod: str,
) -> ModifierGroup | None:
    """Find the group most likely intended for an unrecognized modifier.

    Heuristic: the group whose option with the longest common prefix
    with ``mod`` wins. Falls back to ``None`` if no overlap exists.
    """
    best: tuple[int, ModifierGroup] | None = None
    for group in spec.modifier_groups:
        for option in group.options:
            i = 0
            limit = min(len(option), len(mod))
            while i < limit and option[i] == mod[i]:
                i += 1
            if i >= 2 and (best is None or i > best[0]):
                best = (i, group)
    return best[1] if best is not None else None


def _validate_against(
    inst: Instruction,
    spec: InstructionSpec,
) -> list[ValidationIssue]:
    """Validate an instruction against a single spec."""
    issues: list[ValidationIssue] = []

    # Track which positions in inst.modifiers have already been consumed
    # by an earlier group, so the same modifier value (e.g. ``.bf16``) can
    # legitimately satisfy two adjacent groups (dtype_a and dtype_b).
    consumed: list[bool] = [False] * len(inst.modifiers)

    for group in spec.modifier_groups:
        match_idx: int | None = None
        for idx, mod in enumerate(inst.modifiers):
            if consumed[idx]:
                continue
            if mod in group.options:
                match_idx = idx
                break  # consume the first match; same value can recur

        if match_idx is not None:
            consumed[match_idx] = True
        elif group.required:
            issues.append(ValidationIssue(
                instruction=inst,
                message=(
                    f"Missing required modifier from group '{group.name}': "
                    f"expected one of {group.options}"
                ),
            ))

    remaining_mods = [
        mod for mod, used in zip(inst.modifiers, consumed) if not used
    ]

    # Note: leftover modifiers are treated as *errors* below — when the
    # opcode is in the spec, an unknown modifier means either the user
    # made a typo or our spec is incomplete. Either way the typed surface
    # should hear about it. The escape-hatch path in ``ptx._emit`` calls
    # ``validate_instruction`` directly (not ``validate_or_raise``) and
    # therefore does not surface these as exceptions.

    all_known: set[str] = set()
    for group in spec.modifier_groups:
        all_known.update(group.options)
    for mod in remaining_mods:
        if mod not in all_known:
            # Find the group whose name best describes what this slot
            # is *supposed* to hold, to help the user understand what
            # the legal values are.
            hint_group = _hint_group_for(spec, mod)
            if hint_group is not None:
                msg = (
                    f"Unrecognized modifier {mod!r}; "
                    f"expected a value from group {hint_group.name!r} "
                    f"(one of {hint_group.options})"
                )
            else:
                msg = f"Unrecognized modifier {mod!r} for opcode {inst.opcode!r}"
            issues.append(ValidationIssue(
                instruction=inst,
                message=msg,
                severity="error",
            ))

    n_operands = len(inst.operands)
    if n_operands < spec.min_operands:
        issues.append(ValidationIssue(
            instruction=inst,
            message=(
                f"Too few operands: got {n_operands}, "
                f"expected at least {spec.min_operands}"
            ),
        ))
    if n_operands > spec.max_operands:
        issues.append(ValidationIssue(
            instruction=inst,
            message=(
                f"Too many operands: got {n_operands}, "
                f"expected at most {spec.max_operands}"
            ),
        ))

    return issues


def _error_count(issues: Iterable[ValidationIssue]) -> int:
    return sum(1 for i in issues if i.severity == "error")


def validate_instruction(
    inst: Instruction,
    spec_table: dict[str, InstructionSpec] | None = None,
) -> list[ValidationIssue]:
    """Validate an Instruction node against the ISA spec.

    Returns a list of :class:`ValidationIssue`s (empty if the instruction
    is valid). When multiple specs are registered for the opcode, the
    spec yielding the fewest error-severity issues is used.

    If ``spec_table`` is supplied, it overrides the global base table but
    overload entries from this module are still consulted.
    """
    if spec_table is None:
        candidate_specs = get_specs(inst.opcode)
    else:
        candidate_specs = []
        base = spec_table.get(inst.opcode)
        if base is not None:
            candidate_specs.append(base)
        candidate_specs.extend(_OVERLOADS.get(inst.opcode, ()))

    if not candidate_specs:
        return [ValidationIssue(
            instruction=inst,
            message=f"Unknown opcode: {inst.opcode!r}",
            severity="warning",
        )]

    best: list[ValidationIssue] | None = None
    best_errors = None
    for spec in candidate_specs:
        issues = _validate_against(inst, spec)
        n_err = _error_count(issues)
        if best is None or n_err < best_errors:
            best = issues
            best_errors = n_err
            if n_err == 0:
                break

    return best or []


# ---------------------------------------------------------------------------
# Strict-mode entry point used by ptx._emit
# ---------------------------------------------------------------------------


def _find_user_frame() -> str | None:
    """Walk up the call stack and find the first frame outside of pyptx.

    Returns a string like ``"file.py:42 in fn_name()"`` or ``None`` if
    nothing useful was found.
    """
    import os
    import sys

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    ).replace("\\", "/")
    pkg_root = f"{repo_root}/pyptx"
    preferred: str | None = None
    fallback: str | None = None
    test_frame: str | None = None

    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename
        norm = os.path.abspath(filename).replace("\\", "/")
        # Skip frames inside pyptx itself.
        if "/pyptx/pyptx/" in norm or norm.endswith("/pyptx/spec/validate.py"):
            frame = frame.f_back
            continue
        if "/pyptx/" in norm and "/site-packages/" not in norm:
            # In-tree pyptx files: still skip
            if any(part in norm for part in (
                "/pyptx/ptx.py", "/pyptx/_trace.py", "/pyptx/kernel.py",
                "/pyptx/reg.py", "/pyptx/smem.py", "/pyptx/codegen/",
                "/pyptx/ir/", "/pyptx/parser/", "/pyptx/emitter/",
                "/pyptx/tracer/", "/pyptx/spec/",
            )):
                frame = frame.f_back
                continue
        base = os.path.basename(filename)
        lineno = frame.f_lineno
        funcname = frame.f_code.co_name
        rendered = f"line {lineno} in {funcname}() ({base}:{lineno})"

        # Strongest signal: an actual test file or test function frame.
        if (
            "/tests/" in norm
            or base.startswith("test_")
            or funcname.startswith("test_")
        ):
            if test_frame is None:
                test_frame = rendered
            frame = frame.f_back
            continue

        # Prefer non-library files inside the current repo, such as tests,
        # scripts, or user kernels authored alongside the package.
        if (
            norm.startswith(repo_root + "/")
            and not norm.startswith(pkg_root + "/")
            and "/site-packages/" not in norm
            and "/.venv/" not in norm
        ):
            if preferred is None:
                preferred = rendered
            frame = frame.f_back
            continue

        # Skip common pytest/pluggy wrappers so we can keep walking toward
        # the real user frame instead of stopping at pytest_pyfunc_call().
        if any(part in norm for part in (
            "/site-packages/_pytest/",
            "/site-packages/pluggy/",
            "/site-packages/pytest/",
        )) or base == "pytest":
            if fallback is None:
                fallback = rendered
            frame = frame.f_back
            continue

        if preferred is None:
            preferred = rendered
        frame = frame.f_back
    return test_frame or preferred or fallback


def validate_or_raise(inst: Instruction) -> list[ValidationIssue]:
    """Validate ``inst`` and, in strict mode, raise on errors.

    Always returns the full list of issues. Warnings (e.g. unknown
    opcodes) are surfaced via :class:`UnvalidatedInstructionWarning` and
    never cause an exception.
    """
    issues = validate_instruction(inst)

    # Surface unknown-opcode warnings as Python warnings (suppressed by
    # default — users opt in by adjusting the warning filter).
    has_unknown_opcode = any(
        i.severity == "warning" and i.message.startswith("Unknown opcode")
        for i in issues
    )
    if has_unknown_opcode:
        opcode_chain = inst.opcode + "".join(inst.modifiers)
        warnings.warn(
            f"No spec registered for {opcode_chain!r}; skipping validation",
            UnvalidatedInstructionWarning,
            stacklevel=3,
        )
        return issues

    error_issues = [i for i in issues if i.severity == "error"]
    if error_issues and _strict_mode:
        raise PtxValidationError(
            issues, user_frame=_find_user_frame()
        )

    return issues
