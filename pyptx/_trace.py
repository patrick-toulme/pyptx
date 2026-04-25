"""Thread-local tracing context for kernel construction.

When a @kernel-decorated function executes, a TraceContext is active.
All reg, smem, and ptx calls record into the current context.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Generator

from pyptx.ir.nodes import (
    Instruction,
    Label,
    RegDecl,
    Statement,
    VarDecl,
)


class TraceContext:
    """Accumulates IR nodes during kernel tracing."""

    def __init__(self, *, ptx_version: tuple[int, int] | None = None) -> None:
        self.reg_decls: list[RegDecl] = []
        self.var_decls: list[VarDecl] = []
        self.statements: list[Statement] = []
        self.ptx_version: tuple[int, int] | None = ptx_version
        self._label_counter = 0
        self._reg_counter: dict[str, int] = {}  # prefix -> next id
        self._if_stack: list[tuple[str, str]] = []  # (else_lbl, end_lbl)
        # Dynamic SMEM tracking: when total exceeds 48KB, kernel switches
        # to extern .shared mode with cuFuncSetAttribute at launch time.
        self.dyn_smem_offset: int = 0  # running byte offset
        self.force_dynamic_smem: bool = False  # set True to name allocs dyn_smem
        self._scope_depth: int = 0  # >0 means inside a ptx.scope() block

    def emit(self, stmt: Statement) -> None:
        """Record a statement (instruction, label, etc.)."""
        self.statements.append(stmt)

    def emit_reg_decl(self, decl: RegDecl) -> None:
        """Record a register declaration.

        When inside a ``ptx.scope()`` block, the decl goes into
        ``statements`` (block-local). Otherwise it goes into
        ``reg_decls`` (hoisted to function top).
        """
        if self._scope_depth > 0:
            self.statements.append(decl)
        else:
            self.reg_decls.append(decl)

    def fresh_label(self, prefix: str = "L") -> str:
        """Generate a unique label name."""
        n = self._label_counter
        self._label_counter += 1
        return f"$_{prefix}_{n}"

    def alloc_reg_name(self, prefix: str = "%r") -> int:
        """Return the next available index for a register prefix."""
        n = self._reg_counter.get(prefix, 0)
        self._reg_counter[prefix] = n + 1
        return n

    def body(self) -> tuple[Statement, ...]:
        """Return the full function body: decls then statements.

        Applies copy propagation to remove unnecessary mov instructions
        from RegArray.__setitem__ and operator sugar.
        """
        parts: list[Statement] = []
        parts.extend(self.reg_decls)
        parts.extend(self.var_decls)
        parts.extend(self.statements)
        return tuple(parts)


# -- Thread-local storage ---------------------------------------------------

_local = threading.local()


def get_ctx() -> TraceContext:
    """Get the active trace context. Raises if not inside a @kernel."""
    ctx: TraceContext | None = getattr(_local, "ctx", None)
    if ctx is None:
        raise RuntimeError(
            "No active kernel trace. "
            "This function must be called inside a @kernel-decorated function."
        )
    return ctx


@contextmanager
def trace_scope(*, ptx_version: tuple[int, int] | None = None) -> Generator[TraceContext, None, None]:
    """Context manager that activates a fresh TraceContext."""
    ctx = TraceContext(ptx_version=ptx_version)
    old = getattr(_local, "ctx", None)
    _local.ctx = ctx
    try:
        yield ctx
    finally:
        _local.ctx = old
