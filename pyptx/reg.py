"""Register allocation and register-level DSL sugar.

This module is responsible for turning Python values into explicit PTX
registers. It provides:

- ``reg.scalar`` for a single register
- ``reg.array`` for a declared PTX register array
- ``reg.from_`` for the common “allocate + mov” pattern
- ``reg.wgmma_frag`` for accumulator fragments sized to dense Hopper WGMMA

The returned :class:`Reg` objects support comparison and integer
operator sugar. Those operators emit PTX instructions immediately during
tracing and return new symbolic registers.

Typical usage:

```python
from pyptx import reg, ptx
from pyptx.types import f32, u32, pred

tid = reg.from_(ptx.special.tid.x(), u32)
acc = reg.array(f32, 8)
p = reg.scalar(pred)
lane = tid & 31
is_lane_zero = lane == 0
```

This module is intentionally not a general-purpose algebra layer. The
operator overloads only cover the cases that are common in handwritten
PTX kernels: pointer math, integer loop state, predicates, and a few
frequently repeated idioms.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

from pyptx.ir.nodes import (
    ImmediateOperand,
    Instruction,
    LabelOperand,
    Predicate,
    RegDecl,
    RegisterOperand,
)
from pyptx.ir.types import ScalarType
from pyptx.types import PtxType, pred as pred_type


# Float dtype names that accept ``init=<python float>`` via PTX hex encoding.
_FLOAT_DTYPE_NAMES = frozenset({"f32", "f64", "tf32", "f16", "bf16"})


def _encode_float_init(dtype: PtxType, value: float) -> str:
    """Encode a Python float as a PTX immediate literal for ``mov``.

    PTX accepts:
      * ``0fXXXXXXXX`` — 8 hex digits = big-endian f32 bit pattern
      * ``0dXXXXXXXXXXXXXXXX`` — 16 hex digits = big-endian f64 bit pattern

    For f16 / bf16 there's no ``0f``-style literal so the caller has to
    go through an intermediate b16 register — we reject ``init=<float>``
    there with a clear error.
    """
    if dtype.name in ("f32", "tf32"):
        return "0f" + struct.pack(">f", float(value)).hex().upper()
    if dtype.name == "f64":
        return "0d" + struct.pack(">d", float(value)).hex().upper()
    raise TypeError(
        f"reg.scalar({dtype.name}, init=<float>) not supported — PTX has no "
        f"literal form for {dtype.name}. Use reg.scalar(b16, init=<int>) "
        f"and a mov.b16 + cvt.{dtype.name}.b16 pair instead."
    )

if TYPE_CHECKING:
    from pyptx._trace import TraceContext


class Reg:
    """A symbolic register reference.

    Supports arithmetic and comparison operators that emit PTX instructions
    and return new Reg values.
    """

    __slots__ = ("name", "dtype")

    def __init__(self, name: str, dtype: PtxType) -> None:
        self.name = name
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"Reg({self.name}, {self.dtype})"

    # -- Comparison operators (emit setp, return pred Reg) ------------------

    def __lt__(self, other: Any) -> Reg:
        return _emit_setp("lt", self, other)

    def __le__(self, other: Any) -> Reg:
        return _emit_setp("le", self, other)

    def __gt__(self, other: Any) -> Reg:
        return _emit_setp("gt", self, other)

    def __ge__(self, other: Any) -> Reg:
        return _emit_setp("ge", self, other)

    def __eq__(self, other: Any) -> Reg:  # type: ignore[override]
        return _emit_setp("eq", self, other)

    def __ne__(self, other: Any) -> Reg:  # type: ignore[override]
        return _emit_setp("ne", self, other)

    # -- Predicate helpers --------------------------------------------------

    def __or__(self, other: Any) -> "Reg | PipeRef":
        """Bitwise OR for integer regs, pipe operand for predicates.

        ``r[0] | r[1]`` → emits ``or.b32``
        ``p[0] | p[1]`` → returns PipeRef for setp
        ``r[0] | p[0]`` → returns PipeRef for elect.sync
        ``rd[0] | 0x4000...`` → emits ``or.b64``
        """
        from pyptx.types import pred as pred_type
        # PipeRef if either side is a predicate reg
        if self.dtype == pred_type:
            if isinstance(other, Reg):
                return PipeRef(self, other)
            return NotImplemented
        if isinstance(other, Reg) and other.dtype == pred_type:
            return PipeRef(self, other)
        return _emit_int_or(self, other)

    def __invert__(self) -> NegPred:
        """~pred → negated predicate for use in ptx.if_(~pred)."""
        return NegPred(self)

    def __hash__(self) -> int:
        return hash(self.name)

    # ------------------------------------------------------------------
    # Arithmetic operators (pointer / integer math shortcuts)
    # ------------------------------------------------------------------
    # Only defined for integer / pointer dtypes. Each operator emits one
    # instruction and returns a fresh register. The "10 calls = 10
    # instructions" contract from CLAUDE.md is preserved — each ``+`` is
    # exactly one ``add`` instruction, each ``*`` is exactly one ``mul``.
    # Float arithmetic still goes through ``ptx.inst.*`` for now (f32
    # arithmetic is much less boilerplate-y than pointer math, and the
    # operator overloads would hide the choice of ``add.f32`` vs
    # ``add.rn.f32`` vs ``add.sat.f32``).
    #
    # For integer ops, the result dtype is chosen automatically:
    #   * b64/u64/s64 + anything → u64 (pointer offset math)
    #   * u32/s32 + int → u32 / s32 (preserving the left operand's dtype)
    #   * u32 * int → u64 if the int is 4/8/16 (byte-stride multiply —
    #     this hits ``mul.wide.u32`` which is the one everyone needs)
    #
    # This isn't a general math library — it's the *specific* patterns
    # kernel-writers type dozens of times when they do pointer offsets.

    def __add__(self, other: Any) -> "Reg":
        return _emit_int_add(self, other)

    def __radd__(self, other: Any) -> "Reg":
        return _emit_int_add(self, other)

    def __iadd__(self, other: Any) -> "Reg":
        _emit_int_iadd(self, other)
        return self

    def __sub__(self, other: Any) -> "Reg":
        return _emit_int_sub(self, other)

    def __isub__(self, other: Any) -> "Reg":
        _emit_int_isub(self, other)
        return self

    def __mul__(self, other: Any) -> "Reg":
        return _emit_int_mul(self, other)

    def __rmul__(self, other: Any) -> "Reg":
        return _emit_int_mul(self, other)

    def __imul__(self, other: Any) -> "Reg":
        _emit_int_imul(self, other)
        return self

    def __lshift__(self, other: Any) -> "Reg":
        return _emit_int_shl(self, other)

    def __ilshift__(self, other: Any) -> "Reg":
        _emit_int_ishl(self, other)
        return self

    def __rshift__(self, other: Any) -> "Reg":
        return _emit_int_shr(self, other)

    def __irshift__(self, other: Any) -> "Reg":
        _emit_int_ishr(self, other)
        return self

    def __and__(self, other: Any) -> "Reg":
        return _emit_int_and(self, other)

    def __rand__(self, other: Any) -> "Reg":
        return _emit_int_and(self, other)

    def __iand__(self, other: Any) -> "Reg":
        _emit_int_iand(self, other)
        return self

    def __xor__(self, other: Any) -> "Reg":
        return _emit_int_xor(self, other)

    def __rxor__(self, other: Any) -> "Reg":
        return _emit_int_xor(self, other)

    def __ixor__(self, other: Any) -> "Reg":
        _emit_int_ixor(self, other)
        return self

    def max(self, other: Any) -> "Reg":
        """Emit an integer ``max`` against ``other`` and return the result."""
        return _emit_int_max(self, other)


class PipeRef:
    """Pipe operand for setp dual predicate: %p0|%p1."""

    __slots__ = ("left", "right")

    def __init__(self, left: Reg, right: Reg) -> None:
        self.left = left
        self.right = right


class NegPred:
    """Negated predicate: ~p → @!p."""

    __slots__ = ("reg",)

    def __init__(self, reg: Reg) -> None:
        self.reg = reg


class RegArray:
    """Array of registers from .reg .type %name<count>.

    Indexing returns Reg objects: acc[0] → Reg('%f0', f32).
    """

    __slots__ = ("_base", "_count", "_dtype")

    def __init__(self, base: str, count: int, dtype: PtxType) -> None:
        self._base = base
        self._count = count
        self._dtype = dtype

    def __getitem__(self, idx: int) -> Reg:
        if not 0 <= idx < self._count:
            raise IndexError(
                f"Register index {idx} out of range for "
                f"{self._base}<{self._count}>"
            )
        return Reg(f"{self._base}{idx}", self._dtype)

    def __len__(self) -> int:
        return self._count

    def __iter__(self):
        for idx in range(self._count):
            yield self[idx]

    def regs(self) -> list[Reg]:
        """Materialize the array as a Python list of ``Reg`` objects."""
        return list(self)

    @property
    def count(self) -> int:
        """Number of registers in the declared array."""
        return self._count

    def hw_order(self, *, reverse: bool = False) -> list[Reg]:
        """Return the register list in declaration or reversed order.

        Hopper tensor-core instructions often consume fragments in the
        opposite order from the natural ``reg.array`` declaration order.
        Naming that choice is clearer than open-coding
        ``list(reversed(acc))`` at every call site.
        """
        regs = self.regs()
        return list(reversed(regs)) if reverse else regs

    def __setitem__(self, idx: int, value: "Reg") -> None:
        """Retarget the last emitted instruction to write to slot *idx*.

        When the user writes ``r[90] = (r[89] << 7)``, the ``<<``
        operator emits ``shl.b32 %r_fresh, %r89, 7`` into the trace.
        This method finds that instruction and renames ``%r_fresh`` →
        ``%r90`` in its destination operand, avoiding an extra ``mov``.

        Falls back to emitting a ``mov`` if the last instruction can't
        be retargeted (e.g. the value didn't come from an operator).
        """
        from pyptx._trace import get_ctx
        from pyptx.ir.nodes import Instruction, RegisterOperand
        from dataclasses import replace

        if not isinstance(value, Reg):
            raise TypeError(
                f"Can only assign a Reg to a RegArray slot, got {type(value).__name__}"
            )

        dst_name = f"{self._base}{idx}"
        if value.name == dst_name:
            return  # already targeting the right register

        ctx = get_ctx()

        # Try to retarget the last emitted instruction's destination
        if ctx.statements:
            last = ctx.statements[-1]
            if (isinstance(last, Instruction)
                    and last.operands
                    and isinstance(last.operands[0], RegisterOperand)
                    and last.operands[0].name == value.name):
                # Rename the destination in-place
                new_dst = RegisterOperand(name=dst_name)
                new_ops = (new_dst,) + last.operands[1:]
                ctx.statements[-1] = replace(last, operands=new_ops)
                # Also rename the .reg declaration if it exists
                from pyptx.ir.nodes import RegDecl
                for i, decl in enumerate(ctx.reg_decls):
                    if isinstance(decl, RegDecl) and decl.name == value.name:
                        ctx.reg_decls.pop(i)
                        break
                # For scoped reg decls in statements
                for i, s in enumerate(ctx.statements):
                    if isinstance(s, RegDecl) and s.name == value.name:
                        ctx.statements.pop(i)
                        break
                return

        # Fallback: emit a mov
        dst = Reg(dst_name, self._dtype)
        bits = self._dtype.bits
        mod = f".b{bits}" if bits in (16, 32, 64, 128) else f".b32"
        ctx.emit(Instruction(
            opcode="mov",
            modifiers=(mod,),
            operands=(
                RegisterOperand(dst.name),
                RegisterOperand(value.name),
            ),
        ))

    def __len__(self) -> int:
        return self._count

    @property
    def count(self) -> int:
        """Number of registers in the declared array."""
        return self._count

    @property
    def base(self) -> str:
        """Base PTX register name used for this array declaration."""
        return self._base

    @property
    def dtype(self) -> PtxType:
        """Element dtype of each register in the array."""
        return self._dtype

    def __repr__(self) -> str:
        return f"RegArray({self._base}, {self._count}, {self._dtype})"


# -- Public API (module-level, use implicit trace context) ------------------

# Map PtxType name to register prefix convention
_REG_PREFIX: dict[str, str] = {
    "pred": "%p",
    "b16": "%rs", "b32": "%r", "b64": "%rd", "b128": "%rq",
    "u16": "%rs", "u32": "%r", "u64": "%rd",
    "s16": "%rs", "s32": "%r", "s64": "%rd",
    "f16": "%h", "f16x2": "%hh", "bf16": "%h", "bf16x2": "%hh",
    "f32": "%f", "f64": "%fd",
    "e4m3": "%b", "e5m2": "%b",
    "tf32": "%f",
}


def _prefix_for(dtype: PtxType) -> str:
    return _REG_PREFIX.get(dtype.name, "%r")


def array(dtype: PtxType, count: int, name: str | None = None) -> RegArray:
    """Allocate an array of registers.

    Emits: .reg .{dtype} %prefix<count>;
    Returns a RegArray that can be indexed to get individual Reg refs.

    Args:
        dtype: Element type.
        count: Number of registers.
        name: Optional explicit base name (e.g. '%r', '%rd', '%dtmp').
              If None, uses a default prefix based on dtype.

    When ``name`` is not given and this is the first array for this
    dtype (idx=0), the array uses the bare prefix (e.g. ``%f<count>``,
    which declares ``%f0..%f(count-1)``). We then burn ``count-1`` more
    slots in the scalar counter for the same prefix so that subsequent
    ``reg.scalar()`` calls start at ``%f{count}`` and don't collide
    with the bulk decl. Without this bump, calling ``reg.array(f32, 32)``
    first and ``reg.scalar(f32)`` afterward would hand out ``%f1``,
    ``%f2``, ... which are all already declared as part of ``%f<32>``
    and ptxas rejects the duplicates.
    """
    from pyptx._trace import get_ctx

    ctx = get_ctx()
    if name is not None:
        base = name
        # Bump the counter past the array size so scalar() allocations
        # using the same prefix don't collide with this array's indices.
        current = ctx._reg_counter.get(base, 0)
        if current < count:
            ctx._reg_counter[base] = count
    else:
        prefix = _prefix_for(dtype)
        idx = ctx.alloc_reg_name(prefix)
        if idx == 0:
            base = prefix
            # Reserve the remaining count-1 indices in this prefix's
            # scalar counter so scalar() can't hand out names that
            # collide with the bulk .reg .{type} %prefix<count>; decl.
            for _ in range(count - 1):
                ctx.alloc_reg_name(prefix)
        else:
            # Use a distinct sub-base so elements (``%farr1_0``,
            # ``%farr1_1``, ...) don't collide with scalar ``%f{N}``
            # names. The old ``%f`` + ``chr('a' + idx)`` scheme broke
            # at idx ≥ 26 by producing non-ASCII characters.
            arr_idx = ctx.alloc_reg_name(prefix + "_arr")
            base = f"{prefix}arr{arr_idx}_"

    scalar_type = ScalarType.from_ptx(dtype.ptx)
    ctx.emit_reg_decl(RegDecl(type=scalar_type, name=base, count=count))
    return RegArray(base, count, dtype)


def scalar(
    dtype: PtxType,
    init: int | float | None = None,
    name: str | None = None,
) -> Reg:
    """Allocate a single register, optionally initialized.

    Emits: .reg .{dtype} %name;
    If init is given, also emits: mov.{dtype} %name, init;

    Predicate registers share a single bulk ``.reg .pred %p<N>;``
    declaration (grown as needed) so they don't collide with
    ``_emit_setp``'s pred allocation path — otherwise we'd get both
    an individual ``.reg .pred %p1;`` and a bulk
    ``.reg .pred %p<2>;`` and ptxas rejects the duplicate.
    """
    from pyptx._trace import get_ctx

    ctx = get_ctx()
    if name is not None:
        reg_name = name
    else:
        prefix = _prefix_for(dtype)
        idx = ctx.alloc_reg_name(prefix)
        reg_name = f"{prefix}{idx}"

    # Pred registers allocated out of the default "%p" pool share a
    # single bulk ``.reg .pred %p<N>;`` declaration with ``_emit_setp``,
    # so a mix of reg.scalar(pred) and setp() calls in the same kernel
    # doesn't produce both individual and bulk decls for the same
    # index (which ptxas rejects as duplicates).
    #
    # Callers that pass an explicit ``name=`` (e.g. from PTX→Python
    # codegen that's trying to reproduce a specific corpus file) keep
    # the old individual-decl behavior so corpus roundtrip tests stay
    # byte-exact.
    use_pred_pool = (
        dtype is pred_type
        and name is None
        and reg_name.startswith("%p")
        and reg_name[2:].isdigit()
    )
    if use_pred_pool:
        idx = int(reg_name[2:])
        _ensure_pred_decl(ctx, idx + 1)
    else:
        scalar_type = ScalarType.from_ptx(dtype.ptx)
        ctx.emit_reg_decl(RegDecl(type=scalar_type, name=reg_name))

    reg = Reg(reg_name, dtype)

    if init is not None:
        # Float dtypes with a Python ``float`` init get encoded to the
        # PTX ``0fXXXXXXXX`` / ``0dXXXXXXXXXXXXXXXX`` hex immediate form.
        # Integer dtypes fall through to ``str(init)``.
        if isinstance(init, float) and dtype.name in _FLOAT_DTYPE_NAMES:
            init_text = _encode_float_init(dtype, init)
        else:
            init_text = str(init)
        ctx.emit(Instruction(
            opcode="mov",
            modifiers=(dtype.ptx,),
            operands=(RegisterOperand(reg_name), ImmediateOperand(init_text)),
        ))

    return reg


def from_(src: Any, dtype: PtxType) -> Reg:
    """Allocate ``dtype`` and emit a single ``mov`` from ``src``.

    This is the common prologue/helper pattern for special registers and
    symbolic operands like ``"smem"``:

        tid = reg.from_(ptx.special.tid.x(), u32)
        smem_base = reg.from_("smem", u32)
    """
    from pyptx._trace import get_ctx

    ctx = get_ctx()
    out = scalar(dtype)
    rhs = LabelOperand(src) if isinstance(src, str) else _to_reg_operand(src)
    ctx.emit(Instruction(
        opcode="mov",
        modifiers=(dtype.ptx,),
        operands=(RegisterOperand(out.name), rhs),
    ))
    return out


def wgmma_frag(*, m: int, n: int, dtype: PtxType, name: str | None = None) -> RegArray:
    """Allocate an accumulator fragment sized for dense Hopper WGMMA.

    For the common dense accumulator shapes, Hopper uses ``m * n / 128``
    registers of the accumulator dtype.
    """
    count = (m * n) // 128
    if count <= 0 or (m * n) % 128 != 0:
        raise ValueError(f"invalid WGMMA fragment shape m={m} n={n}")
    return array(dtype, count, name=name)


# -- Internal: emit arithmetic from Reg operator overloads -----------------


def _int_dtype_kind(dtype: PtxType) -> str:
    """Return ``"64"`` / ``"32"`` / ``"16"`` / ``"8"`` / ``"pred"`` for an
    integer dtype, or ``"float"`` for float types, or ``"other"``."""
    n = dtype.name
    if n in ("b64", "u64", "s64"):
        return "64"
    if n in ("b32", "u32", "s32"):
        return "32"
    if n in ("b16", "u16", "s16"):
        return "16"
    if n in ("b8", "u8", "s8"):
        return "8"
    if n == "pred":
        return "pred"
    if n in ("f32", "f64", "f16", "bf16", "tf32"):
        return "float"
    return "other"


def _arith_modifier_32(dtype: PtxType) -> str:
    """Return the PTX arithmetic type modifier for 32-bit integer ops."""
    if dtype.name == "s32":
        return ".s32"
    return ".u32"


def _emit_int_add(left: Reg, right: Any) -> Reg:
    """Emit ``add.s64`` / ``add.u32`` / ``add.u64`` depending on left.dtype.

    Pointer math idiom: ``px + off`` where ``px`` is b64 and ``off`` is u64
    lowers to ``add.s64`` (signed 64-bit add, what ptxas wants for pointer
    offset calculations).
    """
    from pyptx._trace import get_ctx
    ctx = get_ctx()
    kind = _int_dtype_kind(left.dtype)
    if kind == "64":
        # Pointer / index 64-bit add → use add.s64 (what ptxas wants for
        # pointer offset math). If the right operand is a 32-bit Reg,
        # widen it with mul.wide or a fresh 64-bit mov first — for now
        # require a 64-bit Reg or Python int on the right.
        result = scalar(left.dtype)
        right_op = _widen_to_64(ctx, right)
        ctx.emit(Instruction(
            opcode="add", modifiers=(".s64",),
            operands=(
                RegisterOperand(result.name),
                RegisterOperand(left.name),
                right_op,
            ),
        ))
        return result
    if kind == "32":
        # Check if right is a 64-bit Reg — if so, promote to 64-bit add
        if isinstance(right, Reg) and _int_dtype_kind(right.dtype) == "64":
            from pyptx.types import u64
            result = scalar(u64)
            # Widen left to 64-bit
            left_wide = scalar(u64)
            ctx.emit(Instruction(
                opcode="cvt", modifiers=(".u64", ".u32"),
                operands=(RegisterOperand(left_wide.name), RegisterOperand(left.name)),
            ))
            ctx.emit(Instruction(
                opcode="add", modifiers=(".s64",),
                operands=(
                    RegisterOperand(result.name),
                    RegisterOperand(left_wide.name),
                    RegisterOperand(right.name),
                ),
            ))
            return result
        result = scalar(left.dtype)
        mod = _arith_modifier_32(left.dtype)
        ctx.emit(Instruction(
            opcode="add", modifiers=(mod,),
            operands=(
                RegisterOperand(result.name),
                RegisterOperand(left.name),
                _to_reg_operand(right),
            ),
        ))
        return result
    raise TypeError(
        f"Reg.__add__ only supports integer dtypes (b/u/s 32/64); "
        f"got {left.dtype.name}. Use ptx.inst.add.* for floats."
    )


def _emit_int_iadd(left: Reg, right: Any) -> None:
    """In-place: ``left += right``. Writes to ``left.name`` directly."""
    from pyptx._trace import get_ctx
    ctx = get_ctx()
    kind = _int_dtype_kind(left.dtype)
    if kind == "64":
        right_op = _widen_to_64(ctx, right)
        ctx.emit(Instruction(
            opcode="add", modifiers=(".s64",),
            operands=(
                RegisterOperand(left.name),
                RegisterOperand(left.name),
                right_op,
            ),
        ))
        return
    if kind == "32":
        mod = _arith_modifier_32(left.dtype)
        ctx.emit(Instruction(
            opcode="add", modifiers=(mod,),
            operands=(
                RegisterOperand(left.name),
                RegisterOperand(left.name),
                _to_reg_operand(right),
            ),
        ))
        return
    raise TypeError(
        f"Reg.__iadd__ only supports integer dtypes; got {left.dtype.name}"
    )


def _emit_int_sub(left: Reg, right: Any) -> Reg:
    from pyptx._trace import get_ctx
    ctx = get_ctx()
    kind = _int_dtype_kind(left.dtype)
    if kind == "64":
        result = scalar(left.dtype)
        right_op = _widen_to_64(ctx, right)
        ctx.emit(Instruction(
            opcode="sub", modifiers=(".s64",),
            operands=(
                RegisterOperand(result.name),
                RegisterOperand(left.name),
                right_op,
            ),
        ))
        return result
    if kind == "32":
        result = scalar(left.dtype)
        ctx.emit(Instruction(
            opcode="sub", modifiers=(".s32",),
            operands=(
                RegisterOperand(result.name),
                RegisterOperand(left.name),
                _to_reg_operand(right),
            ),
        ))
        return result
    raise TypeError(
        f"Reg.__sub__ only supports integer dtypes; got {left.dtype.name}"
    )


def _emit_int_isub(left: Reg, right: Any) -> None:
    """In-place: ``left -= right``. Writes to ``left.name`` directly."""
    from pyptx._trace import get_ctx
    ctx = get_ctx()
    kind = _int_dtype_kind(left.dtype)
    if kind == "64":
        right_op = _widen_to_64(ctx, right)
        ctx.emit(Instruction(
            opcode="sub", modifiers=(".s64",),
            operands=(
                RegisterOperand(left.name),
                RegisterOperand(left.name),
                right_op,
            ),
        ))
        return
    if kind == "32":
        ctx.emit(Instruction(
            opcode="sub", modifiers=(".s32",),
            operands=(
                RegisterOperand(left.name),
                RegisterOperand(left.name),
                _to_reg_operand(right),
            ),
        ))
        return
    raise TypeError(
        f"Reg.__isub__ only supports integer dtypes; got {left.dtype.name}"
    )


def _emit_int_mul(left: Reg, right: Any) -> Reg:
    """Integer multiply.

    For u32 * Python-int we emit ``mul.wide.u32`` producing a u64 result
    (that's the pointer-stride-multiply pattern — indexing into an f32
    array is ``idx * 4`` and the result is a byte offset added to a
    b64 base). For u32 * u32 reg we emit ``mul.lo.u32`` (keep 32 bits).
    For 64-bit left, we emit ``mul.lo.u64``.
    """
    from pyptx._trace import get_ctx
    from pyptx.types import u32, u64
    ctx = get_ctx()
    kind = _int_dtype_kind(left.dtype)
    if kind == "32":
        # u32 * Python-int → mul.lo.s32 → u32 result (stays 32-bit)
        # Use mul.wide explicitly when 64-bit result is needed.
        if isinstance(right, int):
            result = scalar(left.dtype)
            ctx.emit(Instruction(
                opcode="mul", modifiers=(".lo", ".s32"),
                operands=(
                    RegisterOperand(result.name),
                    RegisterOperand(left.name),
                    ImmediateOperand(str(right)),
                ),
            ))
            return result
        # u32 * u32 Reg → mul.lo.u32 → u32 result
        result = scalar(left.dtype)
        ctx.emit(Instruction(
            opcode="mul", modifiers=(".lo", ".u32"),
            operands=(
                RegisterOperand(result.name),
                RegisterOperand(left.name),
                _to_reg_operand(right),
            ),
        ))
        return result
    if kind == "64":
        result = scalar(left.dtype)
        right_op = _widen_to_64(ctx, right)
        ctx.emit(Instruction(
            opcode="mul", modifiers=(".lo", ".u64"),
            operands=(
                RegisterOperand(result.name),
                RegisterOperand(left.name),
                right_op,
            ),
        ))
        return result
    raise TypeError(
        f"Reg.__mul__ only supports integer dtypes; got {left.dtype.name}"
    )


def _emit_int_imul(left: Reg, right: Any) -> None:
    """In-place: ``left *= right``. Writes to ``left.name`` directly."""
    from pyptx._trace import get_ctx
    ctx = get_ctx()
    kind = _int_dtype_kind(left.dtype)
    if kind == "32":
        if isinstance(right, int):
            ctx.emit(Instruction(
                opcode="mul", modifiers=(".lo", ".s32"),
                operands=(
                    RegisterOperand(left.name),
                    RegisterOperand(left.name),
                    ImmediateOperand(str(right)),
                ),
            ))
            return
        ctx.emit(Instruction(
            opcode="mul", modifiers=(".lo", ".u32"),
            operands=(
                RegisterOperand(left.name),
                RegisterOperand(left.name),
                _to_reg_operand(right),
            ),
        ))
        return
    if kind == "64":
        right_op = _widen_to_64(ctx, right)
        ctx.emit(Instruction(
            opcode="mul", modifiers=(".lo", ".u64"),
            operands=(
                RegisterOperand(left.name),
                RegisterOperand(left.name),
                right_op,
            ),
        ))
        return
    raise TypeError(
        f"Reg.__imul__ only supports integer dtypes; got {left.dtype.name}"
    )


def _emit_int_shl(left: Reg, right: Any) -> Reg:
    from pyptx._trace import get_ctx
    ctx = get_ctx()
    kind = _int_dtype_kind(left.dtype)
    if kind == "32":
        result = scalar(left.dtype)
        ctx.emit(Instruction(
            opcode="shl", modifiers=(".b32",),
            operands=(
                RegisterOperand(result.name),
                RegisterOperand(left.name),
                _to_reg_operand(right),
            ),
        ))
        return result
    if kind == "64":
        result = scalar(left.dtype)
        ctx.emit(Instruction(
            opcode="shl", modifiers=(".b64",),
            operands=(
                RegisterOperand(result.name),
                RegisterOperand(left.name),
                _to_reg_operand(right),
            ),
        ))
        return result
    raise TypeError(
        f"Reg.__lshift__ only supports integer dtypes; got {left.dtype.name}"
    )


def _emit_int_ishl(left: Reg, right: Any) -> None:
    """In-place: ``left <<= right``. Writes to ``left.name`` directly."""
    from pyptx._trace import get_ctx
    ctx = get_ctx()
    kind = _int_dtype_kind(left.dtype)
    if kind == "32":
        ctx.emit(Instruction(
            opcode="shl", modifiers=(".b32",),
            operands=(
                RegisterOperand(left.name),
                RegisterOperand(left.name),
                _to_reg_operand(right),
            ),
        ))
        return
    if kind == "64":
        ctx.emit(Instruction(
            opcode="shl", modifiers=(".b64",),
            operands=(
                RegisterOperand(left.name),
                RegisterOperand(left.name),
                _to_reg_operand(right),
            ),
        ))
        return
    raise TypeError(
        f"Reg.__ilshift__ only supports integer dtypes; got {left.dtype.name}"
    )


def _emit_int_bitop(opcode: str, left: Reg, right: Any) -> Reg:
    """Shared implementation of ``&`` / ``^`` / ``|`` on integer regs.

    Emits ``and.bN`` / ``xor.bN`` / ``or.bN`` with the dtype's bit width
    picked from left.dtype."""
    from pyptx._trace import get_ctx
    ctx = get_ctx()
    kind = _int_dtype_kind(left.dtype)
    if kind == "32":
        result = scalar(left.dtype)
        ctx.emit(Instruction(
            opcode=opcode, modifiers=(".b32",),
            operands=(
                RegisterOperand(result.name),
                RegisterOperand(left.name),
                _to_reg_operand(right),
            ),
        ))
        return result
    if kind == "64":
        result = scalar(left.dtype)
        right_op = _widen_to_64(ctx, right)
        ctx.emit(Instruction(
            opcode=opcode, modifiers=(".b64",),
            operands=(
                RegisterOperand(result.name),
                RegisterOperand(left.name),
                right_op,
            ),
        ))
        return result
    raise TypeError(
        f"Reg.{opcode} only supports 32/64-bit integer dtypes; got {left.dtype.name}"
    )


def _emit_int_ibitop(opcode: str, left: Reg, right: Any) -> None:
    """In-place bitop: writes to ``left.name`` directly."""
    from pyptx._trace import get_ctx
    ctx = get_ctx()
    kind = _int_dtype_kind(left.dtype)
    if kind == "32":
        ctx.emit(Instruction(
            opcode=opcode, modifiers=(".b32",),
            operands=(
                RegisterOperand(left.name),
                RegisterOperand(left.name),
                _to_reg_operand(right),
            ),
        ))
        return
    if kind == "64":
        right_op = _widen_to_64(ctx, right)
        ctx.emit(Instruction(
            opcode=opcode, modifiers=(".b64",),
            operands=(
                RegisterOperand(left.name),
                RegisterOperand(left.name),
                right_op,
            ),
        ))
        return
    raise TypeError(
        f"Reg.__i{opcode}__ only supports 32/64-bit integer dtypes; got {left.dtype.name}"
    )


def _emit_int_and(left: Reg, right: Any) -> Reg:
    return _emit_int_bitop("and", left, right)


def _emit_int_iand(left: Reg, right: Any) -> None:
    _emit_int_ibitop("and", left, right)


def _emit_int_or(left: Reg, right: Any) -> Reg:
    return _emit_int_bitop("or", left, right)


def _emit_int_xor(left: Reg, right: Any) -> Reg:
    return _emit_int_bitop("xor", left, right)


def _emit_int_ixor(left: Reg, right: Any) -> None:
    _emit_int_ibitop("xor", left, right)


def _emit_int_max(left: Reg, right: Any) -> Reg:
    from pyptx._trace import get_ctx
    ctx = get_ctx()
    kind = _int_dtype_kind(left.dtype)
    if kind != "32":
        raise TypeError(
            f"Reg.max only supports 32-bit integer dtypes; got {left.dtype.name}"
        )
    cmp_type = ".u32" if left.dtype.ptx in (".u32", ".b32") else ".s32"
    result = scalar(left.dtype)
    ctx.emit(Instruction(
        opcode="max", modifiers=(cmp_type,),
        operands=(
            RegisterOperand(result.name),
            RegisterOperand(left.name),
            _to_reg_operand(right),
        ),
    ))
    return result


def _emit_int_shr(left: Reg, right: Any) -> Reg:
    from pyptx._trace import get_ctx
    ctx = get_ctx()
    kind = _int_dtype_kind(left.dtype)
    if kind == "32":
        # Default to signed shift (.s32) which matches most compiled
        # PTX output (nvcc/triton). For positive values (addresses,
        # offsets), signed and unsigned shift produce identical results.
        result = scalar(left.dtype)
        ctx.emit(Instruction(
            opcode="shr", modifiers=(".u32",),
            operands=(
                RegisterOperand(result.name),
                RegisterOperand(left.name),
                _to_reg_operand(right),
            ),
        ))
        return result
    if kind == "64":
        result = scalar(left.dtype)
        ctx.emit(Instruction(
            opcode="shr", modifiers=(".u64",),
            operands=(
                RegisterOperand(result.name),
                RegisterOperand(left.name),
                _to_reg_operand(right),
            ),
        ))
        return result
    raise TypeError(
        f"Reg.__rshift__ only supports integer dtypes; got {left.dtype.name}"
    )


def _emit_int_ishr(left: Reg, right: Any) -> None:
    """In-place: ``left >>= right``. Writes to ``left.name`` directly."""
    from pyptx._trace import get_ctx
    ctx = get_ctx()
    kind = _int_dtype_kind(left.dtype)
    if kind == "32":
        ctx.emit(Instruction(
            opcode="shr", modifiers=(".u32",),
            operands=(
                RegisterOperand(left.name),
                RegisterOperand(left.name),
                _to_reg_operand(right),
            ),
        ))
        return
    if kind == "64":
        ctx.emit(Instruction(
            opcode="shr", modifiers=(".u64",),
            operands=(
                RegisterOperand(left.name),
                RegisterOperand(left.name),
                _to_reg_operand(right),
            ),
        ))
        return
    raise TypeError(
        f"Reg.__irshift__ only supports integer dtypes; got {left.dtype.name}"
    )


def _widen_to_64(ctx, val: Any) -> RegisterOperand | ImmediateOperand:
    """Coerce a value for use as the RHS of a 64-bit add/sub/mul.

    64-bit regs and Python ints pass through. 32-bit u32 regs get
    zero-extended to u64 via ``cvt.u64.u32``.
    """
    if isinstance(val, Reg):
        if _int_dtype_kind(val.dtype) == "64":
            return RegisterOperand(val.name)
        if _int_dtype_kind(val.dtype) == "32":
            from pyptx.types import u64
            widened = scalar(u64)
            ctx.emit(Instruction(
                opcode="cvt", modifiers=(".u64", ".u32"),
                operands=(
                    RegisterOperand(widened.name),
                    RegisterOperand(val.name),
                ),
            ))
            return RegisterOperand(widened.name)
        raise TypeError(
            f"Cannot widen {val.dtype.name} Reg to 64-bit for pointer math"
        )
    if isinstance(val, int):
        return ImmediateOperand(str(val))
    raise TypeError(f"Cannot use {type(val).__name__} as 64-bit operand")


# -- Internal: emit setp from comparison operators --------------------------

def _emit_setp(cmp_op: str, left: Reg, right: Any) -> Reg:
    """Emit setp.{cmp}.{type} %pN, left, right; and return the pred Reg.

    ptxas rejects special registers (``%tid.x``, ``%ctaid.y``, ...) as
    direct operands to ``setp``. When ``left`` is such a register we
    first ``mov`` it into a fresh general-purpose register and use that
    instead.
    """
    from pyptx._trace import get_ctx

    ctx = get_ctx()

    # If left is a special register (name has a dotted suffix like
    # ``%tid.x``), we need to stage it into a regular register first.
    left_name = left.name
    if "." in left_name:
        staged = scalar(left.dtype)
        ctx.emit(Instruction(
            opcode="mov",
            modifiers=(left.dtype.ptx,),
            operands=(
                RegisterOperand(staged.name),
                RegisterOperand(left_name),
            ),
        ))
        left_name = staged.name

    # Allocate a predicate register
    p_idx = ctx.alloc_reg_name("%p")
    p_name = f"%p{p_idx}"
    # Ensure pred decl exists (we track this lazily)
    _ensure_pred_decl(ctx, p_idx + 1)

    result = Reg(p_name, pred_type)

    right_op = _to_reg_operand(right)

    # Map bitwise types (b32, b64) to unsigned for setp comparisons
    # since setp.lt.b32 is invalid PTX — needs .u32 or .s32.
    cmp_type = left.dtype.ptx
    if cmp_type == ".b32":
        cmp_type = ".u32"
    elif cmp_type == ".b64":
        cmp_type = ".u64"

    ctx.emit(Instruction(
        opcode="setp",
        modifiers=(f".{cmp_op}", cmp_type),
        operands=(
            RegisterOperand(p_name),
            RegisterOperand(left_name),
            right_op,
        ),
    ))

    return result


def _ensure_pred_decl(ctx: TraceContext, min_count: int) -> None:
    """Ensure there's a .reg .pred %p<N> decl with at least min_count.

    Always hoists to function-level ``reg_decls`` — ``%p<N>`` is a shared
    predicate pool used across the whole kernel, so it must live at the
    top of the function body. Going through ``emit_reg_decl`` while inside
    a ``ptx.scope()`` block would route it into the scope's statement
    stream and produce a duplicate ``.reg .pred`` at top level.
    """
    for decl in ctx.reg_decls:
        if decl.name == "%p" and decl.count is not None:
            if decl.count >= min_count:
                return
            ctx.reg_decls.remove(decl)
            break
    ctx.reg_decls.append(RegDecl(
        type=ScalarType.from_ptx(".pred"),
        name="%p",
        count=min_count,
    ))


def _to_reg_operand(val: Any) -> RegisterOperand | ImmediateOperand:
    if isinstance(val, Reg):
        return RegisterOperand(val.name)
    if isinstance(val, int):
        return ImmediateOperand(str(val))
    if isinstance(val, float):
        return ImmediateOperand(str(val))
    raise TypeError(f"Cannot convert {type(val).__name__} to operand: {val!r}")


# ---------------------------------------------------------------------------
# Opt-in auto-allocation API: reg.alloc(), reg.alloc_array()
# ---------------------------------------------------------------------------
#
# Unlike reg.scalar() / reg.array(), the alloc* family auto-generates a unique
# register name from the dtype plus a per-trace counter. Users don't pick the
# name. Each call immediately appends a RegDecl to the trace context, so no
# finalization step is needed and no _trace.py changes are required.
#
# Auto names use distinct prefixes from _REG_PREFIX so that mixing reg.alloc()
# with reg.scalar() / reg.array() in the same kernel never collides.

# Map PtxType name -> auto-alloc register prefix (distinct from _REG_PREFIX).
_AUTO_PREFIXES: dict[str, str] = {
    "b8":     "%ab",
    "b16":    "%ah",
    "b32":    "%ar",
    "b64":    "%ad",
    "b128":   "%aq",
    "u8":     "%au8",
    "u16":    "%au16",
    "u32":    "%au",
    "u64":    "%aud",
    "s8":     "%as8",
    "s16":    "%as16",
    "s32":    "%as",
    "s64":    "%asd",
    "f16":    "%af16",
    "f16x2":  "%af16x2",
    "bf16":   "%abf",
    "bf16x2": "%abfx2",
    "tf32":   "%atf",
    "f32":    "%af",
    "f64":    "%afd",
    "e4m3":   "%ae4",
    "e5m2":   "%ae5",
    "pred":   "%ap",
}


def _auto_prefix_for(dtype: PtxType) -> str:
    return _AUTO_PREFIXES.get(dtype.name, "%ax")


def alloc(dtype: PtxType) -> Reg:
    """Allocate a single register with an auto-assigned index.

    Unlike reg.scalar(), you don't need to pick a name -- the DSL
    picks one based on the type and a per-function counter. Each
    call returns a Reg with a unique name like %af0, %af1, ... for
    f32, %ar0, %ar1, ... for b32, %ap0, %ap1, ... for pred, etc.

    Registers allocated via reg.alloc() are immediately declared in
    ctx.reg_decls, so they show up in the emitted .reg section at
    the top of the function body.

    Auto-allocated names are distinct from those used by reg.scalar()
    and reg.array(), so mixing the two APIs in the same kernel is safe.

    Usage:
        acc0 = reg.alloc(f32)
        acc1 = reg.alloc(f32)
        ptx.inst.add.f32(acc0, acc0, acc1)
    """
    from pyptx._trace import get_ctx

    ctx = get_ctx()
    prefix = _auto_prefix_for(dtype)
    idx = ctx.alloc_reg_name(prefix)
    reg_name = f"{prefix}{idx}"

    scalar_type = ScalarType.from_ptx(dtype.ptx)
    ctx.emit_reg_decl(RegDecl(type=scalar_type, name=reg_name))
    return Reg(reg_name, dtype)


def alloc_array(dtype: PtxType, count: int) -> RegArray:
    """Allocate an array of registers with an auto-assigned name.

    Unlike reg.array(dtype, count, name=...), you don't pick the name.
    Returns a RegArray that can be indexed to get individual Reg refs.

    Usage:
        accs = reg.alloc_array(f32, 64)
        ptx.inst.add.f32(accs[0], accs[1], accs[2])
    """
    if count <= 0:
        raise ValueError(f"reg.alloc_array count must be > 0, got {count}")

    from pyptx._trace import get_ctx

    ctx = get_ctx()
    prefix = _auto_prefix_for(dtype)
    # Allocate a fresh sub-base by burning one counter slot per array.
    # We tag the base with 'arr' so it cannot collide with scalar
    # auto names which are %{prefix}{int}.
    arr_idx = ctx.alloc_reg_name(prefix + "_arr")
    base = f"{prefix}arr{arr_idx}_"

    scalar_type = ScalarType.from_ptx(dtype.ptx)
    ctx.emit_reg_decl(RegDecl(type=scalar_type, name=base, count=count))
    return RegArray(base, count, dtype)
