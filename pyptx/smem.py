"""Shared-memory allocation, addressing, and barrier objects.

This module covers the shared-memory side of handwritten kernels:

- ``smem.alloc`` allocates shared-memory regions
- ``smem.wgmma_tile`` allocates canonical GMMA/WGMMA operand layouts
- ``smem.mbarrier`` allocates mbarrier arrays in shared memory
- ``smem.base`` / ``smem.load`` / ``smem.store`` provide common address
  and access helpers

Typical usage:

```python
from pyptx import smem
from pyptx.types import bf16

sA = smem.alloc(bf16, (STAGES, BM, BK), swizzle="128B")
bar_full = smem.mbarrier(STAGES)
```

The design here is deliberately pragmatic: shared-memory regions are
described just enough for PTX emission, and some allocations carry extra
metadata for higher-level helpers such as WGMMA descriptor synthesis.
"""

from __future__ import annotations

import math
from typing import Sequence

from pyptx.ir.nodes import RegDecl, VarDecl
from pyptx.ir.types import ScalarType, StateSpace
from pyptx.types import PtxType


class SharedAlloc:
    """Handle to a shared memory allocation.

    Indexing with a stage index returns a SharedSlice representing an offset
    into the allocation, suitable for passing to ptx.cp.async instructions.

    ``gmma_layout`` is set (non-None) when this alloc was produced by
    ``smem.wgmma_tile`` — it carries the ``GmmaLayout`` needed to
    auto-build a wgmma descriptor. The ``gmma_major`` field is a string
    ``"K"`` or ``"MN"`` matching the operand orientation.
    """

    __slots__ = (
        "name", "dtype", "shape", "swizzle",
        "_element_count", "_stage_stride",
        "gmma_layout", "gmma_major",
        "byte_offset",  # offset within dynamic SMEM (0 for static)
    )

    def __init__(
        self,
        name: str,
        dtype: PtxType,
        shape: tuple[int, ...],
        swizzle: str | None,
        gmma_layout: "object | None" = None,
        gmma_major: str | None = None,
        byte_offset: int = 0,
    ) -> None:
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.swizzle = swizzle
        self._element_count = math.prod(shape)
        self._stage_stride = math.prod(shape[1:]) if len(shape) > 1 else 0
        self.gmma_layout = gmma_layout
        self.gmma_major = gmma_major
        self.byte_offset = byte_offset

    def __getitem__(self, idx):
        """Index into this allocation.

        * ``sA[stage]`` — integer stage index, returns a
          :class:`SharedSlice` for passing to TMA / wgmma / stmatrix.
          This is the original stage-indexed API.
        * ``sA[row, col]`` — 2D element access. Emits a
          ``ld.shared.{dtype}`` and returns a fresh ``Reg`` with the
          loaded value. Requires the allocation to be 2D. Row / col can
          be ``Reg`` or Python ``int``.
        """
        if isinstance(idx, tuple) and len(idx) == 2:
            return _smem_load_2d(self, idx[0], idx[1])
        return SharedSlice(self, idx)

    def __setitem__(self, idx, value) -> None:
        """``sA[row, col] = value`` — emit a ``st.shared.{dtype}`` store.

        Only valid when the allocation is 2D. ``value`` must be a ``Reg``
        of a dtype compatible with the allocation's ``dtype``.
        """
        if not (isinstance(idx, tuple) and len(idx) == 2):
            raise TypeError(
                f"SharedAlloc.__setitem__ requires a 2D (row, col) index; "
                f"got {type(idx).__name__}: {idx!r}"
            )
        _smem_store_2d(self, idx[0], idx[1], value)

    def __repr__(self) -> str:
        return f"SharedAlloc({self.name!r}, {self.dtype}, {self.shape})"


class SharedSlice:
    """A stage-indexed slice of a shared allocation."""

    __slots__ = ("alloc", "stage")

    def __init__(self, alloc: SharedAlloc, stage: int) -> None:
        self.alloc = alloc
        self.stage = stage

    @property
    def name(self) -> str:
        """Underlying shared-memory symbol name for this slice."""
        return self.alloc.name

    def __repr__(self) -> str:
        return f"SharedSlice({self.alloc.name!r}[{self.stage}])"


class MbarrierArray:
    """Array of mbarrier objects in shared memory.

    Indexing returns an MbarrierRef for use in ptx.mbarrier.* calls.

    ``byte_offset`` is the byte offset within dynamic SMEM (when
    ``force_dynamic_smem`` is active). Each individual mbarrier is 8
    bytes, so ``MbarrierRef`` for index *i* lives at
    ``byte_offset + i * 8``.
    """

    __slots__ = ("name", "count", "byte_offset")

    def __init__(self, name: str, count: int, byte_offset: int = 0) -> None:
        self.name = name
        self.count = count
        self.byte_offset = byte_offset

    def __getitem__(self, idx: int) -> MbarrierRef:
        if not 0 <= idx < self.count:
            raise IndexError(
                f"Mbarrier index {idx} out of range for {self.name}[{self.count}]"
            )
        return MbarrierRef(self, idx)

    def __repr__(self) -> str:
        return f"MbarrierArray({self.name!r}, {self.count})"


class MbarrierRef:
    """Reference to a single mbarrier object.

    ``byte_offset`` is the byte offset of this specific mbarrier
    within dynamic SMEM: ``array.byte_offset + idx * 8``.  When the
    array was allocated in dynamic SMEM mode (``name == "dyn_smem"``),
    instruction emitters use this offset for addressing.
    """

    __slots__ = ("array", "idx")

    def __init__(self, array: MbarrierArray, idx: int) -> None:
        self.array = array
        self.idx = idx

    @property
    def name(self) -> str:
        """Underlying shared-memory symbol name for this mbarrier array."""
        return self.array.name

    @property
    def byte_offset(self) -> int:
        """Byte offset of this mbarrier within dynamic SMEM."""
        return self.array.byte_offset + self.idx * 8

    def __repr__(self) -> str:
        return f"MbarrierRef({self.array.name!r}[{self.idx}])"


# -- Public API (module-level, use implicit trace context) ------------------

_smem_counter = 0


def base(name: str | None = None):
    """Return a u32 register holding the base address of extern shared memory."""
    from pyptx._trace import get_ctx
    from pyptx.reg import from_ as reg_from
    from pyptx.types import u32

    ctx = get_ctx()
    symbol = name if name is not None else (getattr(ctx, "extern_smem_name", None) or "smem")
    return reg_from(symbol, u32)


def load(dtype: PtxType, address):
    """Emit ``ld.shared.{dtype}`` and return the loaded register."""
    from pyptx._trace import get_ctx
    from pyptx.ir.nodes import AddressOperand, Instruction, RegisterOperand
    from pyptx.ptx import addr as ptx_addr
    from pyptx.reg import scalar as reg_scalar

    ctx = get_ctx()
    out = reg_scalar(dtype)
    addr_op = address if isinstance(address, AddressOperand) else ptx_addr(address)
    ctx.emit(Instruction(
        opcode="ld",
        modifiers=(".shared", dtype.ptx),
        operands=(RegisterOperand(out.name), addr_op),
    ))
    return out


def store(dtype: PtxType, address, value) -> None:
    """Emit ``st.shared.{dtype}``."""
    from pyptx._trace import get_ctx
    from pyptx.ir.nodes import AddressOperand, Instruction, RegisterOperand
    from pyptx.ptx import addr as ptx_addr
    from pyptx.reg import Reg

    if not isinstance(value, Reg):
        raise TypeError(f"smem.store requires a Reg value, got {type(value).__name__}")
    ctx = get_ctx()
    addr_op = address if isinstance(address, AddressOperand) else ptx_addr(address)
    ctx.emit(Instruction(
        opcode="st",
        modifiers=(".shared", dtype.ptx),
        operands=(addr_op, RegisterOperand(value.name)),
    ))


def alloc(
    dtype: PtxType,
    shape: tuple[int, ...] | int,
    swizzle: str | None = None,
    align: int | None = None,
    name: str | None = None,
) -> SharedAlloc:
    """Allocate shared memory.

    Emits: .shared [.align N] .b8 name[bytes];

    Args:
        dtype: Element type (e.g. bf16, f32).
        shape: Shape as tuple (e.g. (STAGES, BM, BK)) or flat int.
        swizzle: Swizzle mode string (e.g. '128B'). Metadata only for now.
        align: Byte alignment. Defaults to 128.
        name: Variable name. Auto-generated if not given.

    Returns:
        SharedAlloc handle for use in ptx.cp.async and ptx.stmatrix calls.
    """
    from pyptx._trace import get_ctx

    global _smem_counter
    ctx = get_ctx()

    if isinstance(shape, int):
        shape = (shape,)

    if name is None:
        name = f"smem_{_smem_counter}"
        _smem_counter += 1

    element_count = math.prod(shape)
    byte_count = element_count * (dtype.bits // 8)

    if align is None:
        align = 128  # default alignment for shared memory

    off = ctx.dyn_smem_offset
    if align > 0 and off % align != 0:
        off = ((off + align - 1) // align) * align
    this_offset = off
    ctx.dyn_smem_offset = off + byte_count

    if ctx.force_dynamic_smem:
        # Dynamic mode: all allocs named dyn_smem, no static VarDecl
        return SharedAlloc("dyn_smem", dtype, shape, swizzle, byte_offset=this_offset)
    else:
        # Static mode: emit named VarDecl
        ctx.var_decls.append(VarDecl(
            state_space=StateSpace.SHARED,
            type=ScalarType.from_ptx(".b8"),
            name=name,
            array_size=byte_count,
            alignment=align,
        ))
        return SharedAlloc(name, dtype, shape, swizzle, byte_offset=this_offset)


def wgmma_tile(
    dtype: PtxType,
    shape: tuple[int, int],
    major: str = "K",
    *,
    align: int | None = None,
    name: str | None = None,
) -> SharedAlloc:
    """Allocate a shared-memory tile in the canonical GMMA layout for
    a wgmma operand.

    The user just says "this is a K-major A of shape (M, K)" and pyptx
    picks the right swizzle/alignment/layout-metadata automatically.
    The returned ``SharedAlloc`` carries a ``gmma_layout`` attribute
    so downstream code (``ptx.wgmma.mma_async``, ``ptx.wgmma.auto_descriptor``)
    can derive the 64-bit descriptor without the user touching LBO,
    SBO, or swizzle mode.

    Args:
        dtype: element type (``bf16``, ``f16``, ``tf32``, ``f32``).
        shape: ``(M, K)`` for an A operand when ``major="K"``, or
            ``(K, N)`` for a B operand when ``major="MN"``.
        major: ``"K"`` (row-major MxK for A / col-major KxN for B) or
            ``"MN"`` (col-major MxK for A / row-major KxN for B).

    Returns:
        A ``SharedAlloc`` with ``.gmma_layout`` set.
    """
    from pyptx.wgmma_layout import Major, pick_gmma_layout

    if major.upper() == "K":
        m_or_n, k = int(shape[0]), int(shape[1])
        major_enum = Major.K
    elif major.upper() == "MN":
        k, m_or_n = int(shape[0]), int(shape[1])
        major_enum = Major.MN
    else:
        raise ValueError(f"major must be 'K' or 'MN', got {major!r}")

    layout = pick_gmma_layout(
        elem_bytes=max(dtype.bits // 8, 1),
        m_or_n=m_or_n,
        k=k,
        major=major_enum,
    )
    alloc = alloc_with_layout(
        dtype,
        shape,
        swizzle=layout.smem_swizzle,
        align=align,
        name=name,
        gmma_layout=layout,
        gmma_major=major.upper(),
    )
    return alloc


def alloc_with_layout(
    dtype: PtxType,
    shape: tuple[int, ...] | int,
    swizzle: str | None = None,
    align: int | None = None,
    name: str | None = None,
    *,
    gmma_layout: "object | None" = None,
    gmma_major: str | None = None,
) -> SharedAlloc:
    """Internal: allocate SMEM and attach GMMA layout metadata.

    Same as ``alloc`` but threads the gmma_layout / gmma_major fields
    through to the returned SharedAlloc. Most users should call
    ``wgmma_tile`` or ``alloc``, not this directly.
    """
    from pyptx._trace import get_ctx

    global _smem_counter
    ctx = get_ctx()

    if isinstance(shape, int):
        shape = (shape,)

    if name is None:
        name = f"smem_{_smem_counter}"
        _smem_counter += 1

    element_count = math.prod(shape)
    byte_count = element_count * (dtype.bits // 8)

    if align is None:
        align = 128

    # Track dynamic SMEM offset (same as alloc())
    off = ctx.dyn_smem_offset
    if align > 0 and off % align != 0:
        off = ((off + align - 1) // align) * align
    this_offset = off
    ctx.dyn_smem_offset = off + byte_count

    if ctx.force_dynamic_smem:
        return SharedAlloc(
            "dyn_smem", dtype, shape, swizzle,
            gmma_layout=gmma_layout, gmma_major=gmma_major,
            byte_offset=this_offset,
        )

    ctx.var_decls.append(VarDecl(
        state_space=StateSpace.SHARED,
        type=ScalarType.from_ptx(".b8"),
        name=name,
        array_size=byte_count,
        alignment=align,
    ))

    return SharedAlloc(
        name, dtype, shape, swizzle,
        gmma_layout=gmma_layout,
        gmma_major=gmma_major,
        byte_offset=this_offset,
    )


_mbar_counter = 0


def mbarrier(count: int, name: str | None = None) -> MbarrierArray:
    """Allocate an array of mbarrier objects in shared memory.

    In static mode (default): emits ``.shared .b64 name[count];``
    In dynamic mode (``force_dynamic_smem``): no VarDecl is emitted;
    the mbarrier lives at ``dyn_smem + byte_offset`` and the name is
    set to ``"dyn_smem"`` so address helpers emit offset-based
    references.

    Args:
        count: Number of mbarrier objects.
        name: Variable name. Auto-generated if not given.

    Returns:
        MbarrierArray handle for use in ptx.mbarrier.* calls.
    """
    from pyptx._trace import get_ctx

    global _mbar_counter
    ctx = get_ctx()

    if name is None:
        name = f"mbar_{_mbar_counter}"
        _mbar_counter += 1

    # Track dynamic SMEM offset for mbarriers (8 bytes each, align 8)
    off = ctx.dyn_smem_offset
    if off % 8 != 0:
        off = ((off + 7) // 8) * 8
    this_offset = off
    ctx.dyn_smem_offset = off + count * 8

    if ctx.force_dynamic_smem:
        # Dynamic mode: no static VarDecl. The MbarrierArray carries
        # the byte_offset so _make_address() can emit
        # [dyn_smem + byte_offset] for each individual mbarrier ref.
        return MbarrierArray("dyn_smem", count, byte_offset=this_offset)

    # Static mode: emit a named VarDecl
    ctx.var_decls.append(VarDecl(
        state_space=StateSpace.SHARED,
        type=ScalarType.from_ptx(".b64"),
        name=name,
        array_size=count,
        alignment=8,
    ))

    # Static shared-memory symbols carry their own base address in PTX,
    # so only the per-element ``idx * 8`` offset should appear in
    # emitted instructions. Threading the cumulative shared-memory byte
    # offset through static mbarriers double-counts the address and
    # corrupts every access after earlier shared allocations.
    return MbarrierArray(name, count, byte_offset=0)


# ---------------------------------------------------------------------------
# GMMA swizzle helpers
# ---------------------------------------------------------------------------
#
# CUTLASS's Swizzle<B, M, S> applies:
#   physical = logical XOR ((logical & yyy_msk) >> S)
# where yyy_msk = ((1 << B) - 1) << (S + M).
#
# For the canonical GMMA/UMMA operand layouts used by CUTLASS on SM90/SM100,
# S=3 and M=4 across the standard 32B/64B/128B swizzles:
#   B32  → Swizzle<1,4,3>: yyy_msk=0x080, shift=3
#   B64  → Swizzle<2,4,3>: yyy_msk=0x180, shift=3
#   B128 → Swizzle<3,4,3>: yyy_msk=0x380, shift=3

_SWIZZLE_PARAMS: dict[str, tuple[int, int]] = {
    "32B":  (0x080, 3),
    "64B":  (0x180, 3),
    "128B": (0x380, 3),
}


def apply_swizzle(logical_offset: "Reg", swizzle: str | None) -> "Reg":
    """Apply GMMA swizzle to a logical byte offset, returning the physical offset.

    ``swizzle`` is ``"32B"``, ``"64B"``, ``"128B"``, or ``None``/``"NONE"``
    (identity).  Emits 3 ALU instructions for non-trivial swizzles.
    """
    if swizzle is None or swizzle.upper() == "NONE":
        return logical_offset

    params = _SWIZZLE_PARAMS.get(swizzle.upper())
    if params is None:
        raise ValueError(f"Unknown swizzle mode {swizzle!r}")

    from pyptx._trace import get_ctx
    from pyptx.ir.nodes import ImmediateOperand, Instruction, RegisterOperand
    from pyptx.reg import scalar as _reg_scalar
    from pyptx.types import u32

    mask, shift = params
    ctx = get_ctx()

    xor_bits = _reg_scalar(u32)
    ctx.emit(Instruction(
        opcode="and", modifiers=(".b32",),
        operands=(
            RegisterOperand(xor_bits.name),
            RegisterOperand(logical_offset.name),
            ImmediateOperand(str(mask)),
        ),
    ))
    ctx.emit(Instruction(
        opcode="shr", modifiers=(".u32",),
        operands=(
            RegisterOperand(xor_bits.name),
            RegisterOperand(xor_bits.name),
            ImmediateOperand(str(shift)),
        ),
    ))
    physical = _reg_scalar(u32)
    ctx.emit(Instruction(
        opcode="xor", modifiers=(".b32",),
        operands=(
            RegisterOperand(physical.name),
            RegisterOperand(logical_offset.name),
            RegisterOperand(xor_bits.name),
        ),
    ))
    return physical


# ---------------------------------------------------------------------------
# 2D element-access helpers: sA[row, col] / sA[row, col] = val
# ---------------------------------------------------------------------------
#
# These back the ``__getitem__((row, col))`` / ``__setitem__((row, col))``
# dispatch on :class:`SharedAlloc`. Every 2D access emits exactly
# three instructions:
#
#   mul.lo.u32  row_x_cols, row, cols
#   add.u32     idx,        row_x_cols, col
#   st.shared.{dtype} [sA + byte_off], value     (for __setitem__)
#
# Or for __getitem__, the final instruction is
#
#   ld.shared.{dtype} %fresh, [sA + byte_off]
#
# returning ``%fresh`` as a Reg. The address is composed by adding
# ``byte_off = idx * elem_bytes`` to the allocation's base symbol in a
# u32 shared-memory register.
#
# This assumes a naive row-major layout and is NOT aware of the wgmma
# swizzle modes — it's the right tool for non-wgmma f32 smem scratch
# like the ``sP`` buffer in Flash Attention's online softmax, not for
# feeding wgmma inputs (those still need ``smem.wgmma_tile`` + the
# swizzled descriptor).


def _dtype_bytes(dtype: PtxType) -> int:
    return max(dtype.bits // 8, 1)


def _2d_byte_offset(alloc: "SharedAlloc", row, col):
    """Build ``row*cols*elem_bytes + col*elem_bytes`` as a u32 Reg.

    Emits at most three instructions (mul, add, shl) depending on how
    much constant folding the inputs allow:

      * ``row`` and ``col`` both Reg → mul.lo.u32 + add.u32 + shl.b32
      * ``row`` Reg, ``col`` int → mul.lo.u32 + add.u32(imm) + shl.b32
      * ``row`` int, ``col`` Reg → mul.lo.u32(imm) + add.u32 + shl.b32

    ``row`` and ``col`` must either be ``Reg``s (u32/s32/b32) or
    Python ints.
    """
    from pyptx._trace import get_ctx
    from pyptx.ir.nodes import ImmediateOperand, Instruction, RegisterOperand
    from pyptx.reg import scalar as _reg_scalar
    from pyptx.reg import Reg
    from pyptx.types import u32

    if len(alloc.shape) != 2:
        raise TypeError(
            f"2D indexing requires a 2D SharedAlloc; got shape {alloc.shape}"
        )
    cols = alloc.shape[1]
    elem_bytes = _dtype_bytes(alloc.dtype)
    ctx = get_ctx()

    # idx = row * cols + col, in elements.
    #
    # ptxas rejects instructions with two immediate operands, so we
    # need to constant-fold the row*cols product when ``row`` is a
    # Python int (``sV[0, col_reg]``-style access). Similarly, when
    # ``col`` is an int and ``row`` is an int, the whole byte offset
    # is a constant and we can materialize it with a single mov.
    def _to_op(val):
        if isinstance(val, Reg):
            return RegisterOperand(val.name)
        if isinstance(val, int):
            return ImmediateOperand(str(val))
        raise TypeError(
            f"2D smem index must be Reg or int; got {type(val).__name__}"
        )

    idx = _reg_scalar(u32)
    if isinstance(row, int) and isinstance(col, int):
        # Fully constant index — one mov.
        ctx.emit(Instruction(
            opcode="mov", modifiers=(".u32",),
            operands=(
                RegisterOperand(idx.name),
                ImmediateOperand(str(row * cols + col)),
            ),
        ))
    elif isinstance(row, int):
        # row*cols folds at trace time; only col varies.
        const_off = row * cols
        if const_off == 0:
            ctx.emit(Instruction(
                opcode="mov", modifiers=(".u32",),
                operands=(
                    RegisterOperand(idx.name),
                    _to_op(col),
                ),
            ))
        else:
            ctx.emit(Instruction(
                opcode="add", modifiers=(".u32",),
                operands=(
                    RegisterOperand(idx.name),
                    _to_op(col),
                    ImmediateOperand(str(const_off)),
                ),
            ))
    else:
        # row is a Reg — emit mul.lo.u32 row, cols_imm.
        row_times_cols = _reg_scalar(u32)
        ctx.emit(Instruction(
            opcode="mul", modifiers=(".lo", ".u32"),
            operands=(
                RegisterOperand(row_times_cols.name),
                RegisterOperand(row.name),
                ImmediateOperand(str(cols)),
            ),
        ))
        ctx.emit(Instruction(
            opcode="add", modifiers=(".u32",),
            operands=(
                RegisterOperand(idx.name),
                RegisterOperand(row_times_cols.name),
                _to_op(col),
            ),
        ))
    # byte_off = idx * elem_bytes (via shift when elem_bytes is a power of 2)
    byte_off = _reg_scalar(u32)
    shift = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}.get(elem_bytes)
    if shift is None:
        # Fall back to mul.lo.u32 — unusual element sizes
        ctx.emit(Instruction(
            opcode="mul", modifiers=(".lo", ".u32"),
            operands=(
                RegisterOperand(byte_off.name),
                RegisterOperand(idx.name),
                ImmediateOperand(str(elem_bytes)),
            ),
        ))
    elif shift == 0:
        return idx   # reuse — byte_off == idx
    else:
        ctx.emit(Instruction(
            opcode="shl", modifiers=(".b32",),
            operands=(
                RegisterOperand(byte_off.name),
                RegisterOperand(idx.name),
                ImmediateOperand(str(shift)),
            ),
        ))
    return byte_off


def _shared_addr(alloc: "SharedAlloc", byte_off) -> "Reg":
    """Compute a u32 shared-memory address: ``&alloc[0] + byte_off``."""
    from pyptx._trace import get_ctx
    from pyptx.ir.nodes import Instruction, RegisterOperand
    from pyptx.reg import scalar as _reg_scalar
    from pyptx.types import u32

    ctx = get_ctx()
    addr = _reg_scalar(u32)
    # mov.b32 addr, <alloc_symbol>  — PTX accepts a shared-mem symbol as
    # an immediate here; ptxas resolves to the symbol's offset.
    ctx.emit(Instruction(
        opcode="mov", modifiers=(".b32",),
        operands=(
            RegisterOperand(addr.name),
            RegisterOperand(alloc.name),  # use as a symbol reference
        ),
    ))
    ctx.emit(Instruction(
        opcode="add", modifiers=(".u32",),
        operands=(
            RegisterOperand(addr.name),
            RegisterOperand(addr.name),
            RegisterOperand(byte_off.name),
        ),
    ))
    return addr


def _smem_load_2d(alloc: "SharedAlloc", row, col) -> "Reg":
    """Emit ``ld.shared.{dtype} %fresh, [addr]`` and return the fresh Reg."""
    from pyptx._trace import get_ctx
    from pyptx.ir.nodes import AddressOperand, Instruction, RegisterOperand
    from pyptx.reg import scalar as _reg_scalar

    ctx = get_ctx()
    byte_off = _2d_byte_offset(alloc, row, col)
    sh_addr = _shared_addr(alloc, byte_off)
    result = _reg_scalar(alloc.dtype)
    ctx.emit(Instruction(
        opcode="ld",
        modifiers=(".shared", alloc.dtype.ptx),
        operands=(
            RegisterOperand(result.name),
            AddressOperand(base=sh_addr.name, offset=None),
        ),
    ))
    return result


def _smem_store_2d(alloc: "SharedAlloc", row, col, value) -> None:
    """Emit ``st.shared.{dtype} [addr], value``."""
    from pyptx._trace import get_ctx
    from pyptx.ir.nodes import AddressOperand, Instruction, RegisterOperand
    from pyptx.reg import Reg

    if not isinstance(value, Reg):
        raise TypeError(
            f"sA[row, col] = value requires ``value`` to be a Reg; "
            f"got {type(value).__name__}"
        )
    ctx = get_ctx()
    byte_off = _2d_byte_offset(alloc, row, col)
    sh_addr = _shared_addr(alloc, byte_off)
    ctx.emit(Instruction(
        opcode="st",
        modifiers=(".shared", alloc.dtype.ptx),
        operands=(
            AddressOperand(base=sh_addr.name, offset=None),
            RegisterOperand(value.name),
        ),
    ))
