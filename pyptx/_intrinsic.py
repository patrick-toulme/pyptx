"""@ptx.intrinsic — scope annotation for reusable PTX patterns.

A lightweight decorator that marks a Python function as a "mega instruction"
composed of several smaller PTX ops. When called inside a kernel trace, the
statements emitted by the function are wrapped in an IntrinsicScope IR node
so inspection tools can see which high-level operation produced which PTX
lines.

The emitted PTX is unchanged in semantics — intrinsic scopes render as
// BEGIN / // END comment markers.

Usage:
    @ptx.intrinsic
    def async_load_tile(dst, src_desc, coord, mbar):
        ptx.mbarrier.wait(mbar, phase=0)
        ptx.cp.async.bulk.tensor.tile_2d(dst=dst, src=src_desc, coord=coord, mbar=mbar)

    @kernel
    def gemm():
        sA = smem.alloc(bf16, (3, 128, 64))
        bar = smem.mbarrier(3)
        async_load_tile(sA[0], A_desc, (0, 0), bar[0])  # 2 PTX ops, named scope
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from pyptx.ir.nodes import IntrinsicScope

F = TypeVar("F", bound=Callable[..., Any])


def intrinsic(fn: F) -> F:
    """Mark a function as a PTX intrinsic (named scope of PTX instructions).

    The decorator wraps the function so that when it's called inside a
    kernel trace, the statements it emits are collected into an
    IntrinsicScope IR node named after the function.

    The function's return value is preserved — this is purely a scope
    annotation layer, not a transformation.

    Nesting works: one intrinsic can call another, and both scopes will
    show up in the IR.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Lazy import to avoid circular dependency
        from pyptx._trace import get_ctx

        try:
            ctx = get_ctx()
        except RuntimeError:
            # Not inside a trace — just call the function directly
            return fn(*args, **kwargs)

        # Save current statements list, install a fresh one so the
        # intrinsic's emissions are isolated.
        saved = ctx.statements
        ctx.statements = []
        try:
            result = fn(*args, **kwargs)
            body = tuple(ctx.statements)
        finally:
            ctx.statements = saved

        # Wrap the collected body in an IntrinsicScope and append it
        ctx.emit(IntrinsicScope(
            name=fn.__name__,
            args_repr=_format_args(args, kwargs),
            body=body,
        ))

        return result

    # Mark the wrapper so introspection tools can detect it
    wrapper.__is_ptx_intrinsic__ = True  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def _format_args(args: tuple, kwargs: dict) -> str:
    """Format args/kwargs as a short repr for the scope marker.

    Keeps the output short so PTX comments stay readable. Register/type
    objects render via their __repr__; tensors/specs show just the name.
    """
    parts: list[str] = []
    for a in args:
        parts.append(_short_repr(a))
    for k, v in kwargs.items():
        parts.append(f"{k}={_short_repr(v)}")
    return ", ".join(parts)


def _short_repr(val: Any) -> str:
    """Short repr for an arg — avoids dumping huge objects."""
    if hasattr(val, "name") and isinstance(getattr(val, "name", None), str):
        return val.name
    r = repr(val)
    if len(r) > 40:
        return r[:37] + "..."
    return r
