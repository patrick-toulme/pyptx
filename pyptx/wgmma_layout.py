"""Canonical GMMA shared-memory layouts for wgmma.mma_async.

wgmma takes its A and B operands as 64-bit descriptors that describe
a specific shared-memory arrangement. The descriptor is derived from
a "canonical layout" — one of four layout families defined by
CUTLASS/Cute (INTERLEAVE / B32 / B64 / B128) whose bit-level geometry
the hardware knows how to read. For a given (dtype, M, K, major) tile,
exactly one canonical layout is natural (the one where the element
row stride equals the layout's core-matrix-tile width). Pick the
wrong one and wgmma reads permuted k indices and you get garbage.

This module is a pure-Python port of the parts of
``cute/atom/mma_traits_sm90_gmma.hpp`` we actually need: given a tile
description it returns the matching:

  * ``layout_type`` — one of {INTERLEAVE, B32, B64, B128}. This goes
    in bits [63:62] of the wgmma descriptor.
  * ``leading_byte_offset`` / ``stride_byte_offset`` — in bytes. The
    wgmma descriptor field stores these values right-shifted by 4
    (i.e. in uint128_t units).
  * ``tma_swizzle`` — a string ("NONE", "32B", "64B", "128B") the
    caller passes to ``smem.alloc(swizzle=...)`` and the matching
    ``Layout.TMA_*B`` in the @kernel spec. The swizzle TMA uses to
    WRITE data into shared memory and the swizzle wgmma uses to READ
    it back must be the SAME — they compose to identity and give the
    logical element order. Any mismatch produces garbage as soon as
    the output depends on the A[k]*B[k] pairing across varying k.

Reference:
  cute/arch/mma_sm90_desc.hpp      — descriptor bit layout
  cute/atom/mma_traits_sm90_gmma.hpp
    - docstring for make_gmma_desc<Major::MN>: layout families
    - docstring for make_gmma_desc<Major::K>:  layout families
    - make_gmma_desc body: stride field extraction from the canonical
      logical_divide result
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class LayoutType(IntEnum):
    """wgmma descriptor layout_type field (bits [63:62])."""

    INTERLEAVE = 0  # no swizzle, 1 uint128_t wide rows
    B128 = 1  # Swizzle<3,4,3>, 8 uint128_t (128-byte) rows
    B64 = 2  # Swizzle<2,4,3>, 4 uint128_t (64-byte)  rows
    B32 = 3  # Swizzle<1,4,3>, 2 uint128_t (32-byte)  rows


class Major(IntEnum):
    """Which operand direction is the "leading" (fastest-varying) axis
    in the original tile layout.

    ``K`` — the K dimension is fast (row-major A of shape (M, K),
    col-major B of shape (K, N)).

    ``MN`` — the M or N dimension is fast (col-major A of shape (M, K),
    row-major B of shape (K, N)).
    """

    K = 0
    MN = 1


@dataclass(frozen=True)
class GmmaLayout:
    """A canonical GMMA shared-memory layout.

    Attributes:
        layout_type: one of LayoutType values; this is what goes in
            bits [63:62] of the wgmma descriptor.
        leading_byte_offset: byte value passed to make_descriptor's
            ``leading_byte_offset=`` kwarg. Stored in bits [29:16]
            after a 4-bit right-shift.
        stride_byte_offset: byte value for ``stride_byte_offset=``.
            Stored in bits [45:32] after the 4-bit right-shift.
        tma_swizzle: the name of the matching TMA swizzle that must be
            used when TMA stores data into the shared buffer this
            descriptor will read. One of ``"NONE"``, ``"32B"``,
            ``"64B"``, ``"128B"``.
        smem_swizzle: the argument value for ``smem.alloc(swizzle=...)``
            that produces a shared memory region aligned for this
            layout. Same names as tma_swizzle (without the "NONE" case
            where swizzle is simply omitted).
    """

    layout_type: LayoutType
    leading_byte_offset: int
    stride_byte_offset: int
    tma_swizzle: str
    smem_swizzle: str | None

    @property
    def swizzle_code(self) -> int:
        """Integer value for ``ptx.wgmma.make_descriptor(swizzle=...)``."""
        return int(self.layout_type)


_LAYOUT_BY_ROW_BYTES: dict[int, LayoutType] = {
    16: LayoutType.INTERLEAVE,  # 1 uint128_t wide
    32: LayoutType.B32,  # 2 uint128_t wide
    64: LayoutType.B64,  # 4 uint128_t wide
    128: LayoutType.B128,  # 8 uint128_t wide
}

_TMA_SWIZZLE_BY_LAYOUT: dict[LayoutType, tuple[str, str | None]] = {
    # (tma_swizzle_layout_name, smem_alloc_swizzle_arg)
    LayoutType.INTERLEAVE: ("NONE", None),
    LayoutType.B32: ("32B", "32B"),
    LayoutType.B64: ("64B", "64B"),
    LayoutType.B128: ("128B", "128B"),
}


def pick_gmma_layout(
    *,
    elem_bytes: int,
    m_or_n: int,
    k: int,
    major: Major,
) -> GmmaLayout:
    """Return the canonical GMMA layout for a (dtype, M/N, K, major) tile.

    Args:
        elem_bytes: size of one matrix element in bytes (2 for bf16/f16,
            4 for tf32/f32).
        m_or_n: the M dimension (for operand A) or N (for operand B).
        k: the K dimension. For wgmma m64nNk16 bf16, always 16.
        major: ``Major.K`` for K-major (row-major A / col-major B) or
            ``Major.MN`` for MN-major (col-major A / row-major B).

    Returns:
        A ``GmmaLayout`` with all the numeric fields the wgmma
        descriptor builder needs. The caller passes
        ``layout.leading_byte_offset``, ``layout.stride_byte_offset``,
        and ``layout.swizzle_code`` to ``ptx.wgmma.make_descriptor``;
        and uses ``layout.tma_swizzle`` in the @kernel Tile's Layout
        and ``layout.smem_swizzle`` in ``smem.alloc``.

    Raises:
        ValueError: if the tile can't be expressed in any of the four
            canonical layouts at all — typically because the row width
            in bytes isn't one of {16, 32, 64, 128}.
    """
    if elem_bytes <= 0:
        raise ValueError(f"elem_bytes must be positive, got {elem_bytes}")
    if m_or_n <= 0 or k <= 0:
        raise ValueError(f"shape must be positive, got ({m_or_n}, {k})")

    if major == Major.K:
        # Row-major A: K is fast. The "row width" that determines the
        # layout family is K * elem_bytes — that's the stride between
        # consecutive M rows of the tile.
        row_bytes = k * elem_bytes
        return _build_k_major(elem_bytes, m_or_n, k, row_bytes)
    else:
        # MN-major B: N is fast. Row width = N * elem_bytes — stride
        # between consecutive K rows.
        row_bytes = m_or_n * elem_bytes
        return _build_mn_major(elem_bytes, m_or_n, k, row_bytes)


def _build_k_major(
    elem_bytes: int, m: int, k: int, row_bytes: int
) -> GmmaLayout:
    """K-major canonical layout for operand A (row-major MxK).

    From cute/atom/mma_traits_sm90_gmma.hpp docstring::

        LayoutType::INTERLEAVE : ((8,m),(T,k)):((1T, SBO),(1, T))
        LayoutType::B32        : ((8,m),(T,k)):((2T, SBO),(1, T))
        LayoutType::B64        : ((8,m),(T,k)):((4T, SBO),(1, T))
        LayoutType::B128       : ((8,m),(T,k)):((8T, SBO),(1, T))

    where T = 16 / elem_bytes (uint128_t width in elements).

    For each layout type the inner M stride (stride<0,0> of the
    canonical form) equals W * T elements = W uint128_t units, where W
    is {1, 2, 4, 8} for {INTERLEAVE, B32, B64, B128}. This is the same
    W as row_bytes / 16, so we can pick the layout directly from
    row_bytes.

    For the descriptor fields CUTLASS's make_gmma_desc assigns::

        desc.stride_byte_offset_ = stride_01  # outer M stride
        desc.leading_byte_offset_ = stride_10 # inner K stride

    stride_10 is always 1 uint128_t = 16 bytes (for K-major, moving 1
    in K is 1 element = ... < 1 uint128_t, so after recast to u128 the
    inner K stride is 1 unit because the recast consumes a full core
    matrix row). stride_01 is 8 * row_bytes (8 M-rows per core matrix).
    """
    if row_bytes not in _LAYOUT_BY_ROW_BYTES and row_bytes <= 128:
        raise ValueError(
            f"pyptx wgmma: K-major tile with K={k}, elem_bytes={elem_bytes} "
            f"has row width {row_bytes} bytes, which is not one of the "
            f"canonical GMMA row widths {list(_LAYOUT_BY_ROW_BYTES)}. Try "
            f"padding K or using a different dtype."
        )
    if m % 8 != 0:
        raise ValueError(
            f"pyptx wgmma: K-major tile requires M divisible by 8 "
            f"(core-matrix row group). Got M={m}."
        )

    if row_bytes > 128:
        layout_type = LayoutType.B128
    else:
        layout_type = _LAYOUT_BY_ROW_BYTES[row_bytes]

    if row_bytes > 128:
        # Multi-stripe case (K wider than one B128 stripe = 128 bytes).
        # The logical K-fast row is decomposed into adjacent 128-byte
        # stripes, each of which stores all M rows for its K-subrange:
        #
        #   sA+0:    stripe 0 (M rows × 128 bytes each)
        #   sA+S:    stripe 1 (M rows × 128 bytes each)
        #   sA+2S:   stripe 2 (if K > 256 bytes)
        #   ...
        #
        # For K-major descriptors the "leading" field corresponds to
        # motion along K. In the multi-stripe case that becomes the
        # distance from one 128-byte stripe to the next.
        stripe_bytes = m * 128
        leading_byte_offset = stripe_bytes
        stride_byte_offset = 8 * 128
    else:
        leading_byte_offset = 16  # always 1 u128 for single-stripe K-major
        stride_byte_offset = 8 * row_bytes  # M stride between 8-row core groups
    tma_sw, smem_sw = _TMA_SWIZZLE_BY_LAYOUT[layout_type]

    return GmmaLayout(
        layout_type=layout_type,
        leading_byte_offset=leading_byte_offset,
        stride_byte_offset=stride_byte_offset,
        tma_swizzle=tma_sw,
        smem_swizzle=smem_sw,
    )


def _build_mn_major(
    elem_bytes: int, n: int, k: int, row_bytes: int
) -> GmmaLayout:
    """MN-major canonical layout for operand B (row-major KxN).

    From cute/atom/mma_traits_sm90_gmma.hpp docstring::

        INTERLEAVE : ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))
        B32        : ((T,2,m),(8,k)):((1,T,LBO),(2T,SBO))
        B64        : ((T,4,m),(8,k)):((1,T,LBO),(4T,SBO))
        B128       : ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))

    For the degenerate m=1 case (N = T elements = 1 uint128_t wide,
    which is N=8 bf16 / N=8 f16 / N=4 tf32 / N=4 f32), the canonical
    form collapses: the outer N dim is size 1 and SBO is trivial. What
    matters is LBO = the stride between 8-K-row slabs, which equals
    8 * row_bytes.

    For wider N the non-INTERLEAVE layouts also apply, but the
    LBO/SBO positions swap — see make_gmma_desc's branch on
    LAYOUT_TYPE. For now we only fully handle N-u128-width = 1 and
    N-u128-width = 2/4/8 (which are the B32/B64/B128 natural cases).
    """
    # For row_bytes > 128 (e.g., N=128 bf16 = 256 bytes), clamp to
    # B128 layout and compute LBO/SBO for the multi-stripe case.
    # CUTLASS does the same — the wgmma instruction handles multi-stripe
    # access internally.
    if row_bytes not in _LAYOUT_BY_ROW_BYTES and row_bytes < 16:
        raise ValueError(
            f"pyptx wgmma: MN-major tile with N={n}, elem_bytes={elem_bytes} "
            f"has row width {row_bytes} bytes, which is too narrow."
        )
    if k % 8 != 0:
        raise ValueError(
            f"pyptx wgmma: MN-major tile requires K divisible by 8 "
            f"(core-matrix K group). Got K={k}."
        )

    if row_bytes > 128:
        layout_type = LayoutType.B128
    else:
        layout_type = _LAYOUT_BY_ROW_BYTES[row_bytes]
    tma_sw, smem_sw = _TMA_SWIZZLE_BY_LAYOUT[layout_type]

    if layout_type == LayoutType.INTERLEAVE:
        # Degenerate N = 1 u128: LBO = stride between 8-K-row slabs.
        leading_byte_offset = 8 * row_bytes
        stride_byte_offset = 16  # trivial, m=1
    elif row_bytes > 128:
        # Multi-stripe case (N wider than one B128 stripe = 128 bytes).
        # Two or more TMA loads produce a STRIPE-MAJOR SMEM layout
        # where each 128-byte stripe is contiguous across all K rows,
        # and consecutive stripes are adjacent:
        #
        #   sB+0:    stripe 0 (K rows × 128 bytes each)
        #   sB+S:    stripe 1 (K rows × 128 bytes each)
        #   sB+2S:   stripe 2 (if N > 256 bytes)
        #   ...
        #
        # The wgmma descriptor encodes:
        #   LBO = stride from one stripe to the next = K * 128 bytes
        #   SBO = stride between 8-K-row groups within a stripe = 8 * 128 = 1024
        stripe_bytes = k * 128   # one stripe: K rows × 128 bytes/row
        leading_byte_offset = stripe_bytes   # LBO: inter-stripe
        stride_byte_offset = 8 * 128         # SBO: 8 K-rows × 128B/row
    else:
        # Single-stripe non-INTERLEAVE (B32/B64/B128 with row_bytes ≤ 128).
        leading_byte_offset = 16
        stride_byte_offset = 8 * row_bytes

    return GmmaLayout(
        layout_type=layout_type,
        leading_byte_offset=leading_byte_offset,
        stride_byte_offset=stride_byte_offset,
        tma_swizzle=tma_sw,
        smem_swizzle=smem_sw,
    )


# ---------------------------------------------------------------------------
# Public helper: give me the layout for "A" or "B" of a GEMM
# ---------------------------------------------------------------------------


def layout_for_a(*, dtype: Any, m: int, k: int) -> GmmaLayout:
    """Canonical GMMA layout for operand A (row-major MxK, K-major).

    Usage::

        la = layout_for_a(dtype=bf16, m=64, k=16)
        # la.layout_type  → LayoutType.B32
        # la.leading_byte_offset → 16
        # la.stride_byte_offset  → 256
        # la.tma_swizzle → "32B"
        # la.smem_swizzle → "32B"
    """
    elem_bytes = _dtype_bytes(dtype)
    return pick_gmma_layout(elem_bytes=elem_bytes, m_or_n=m, k=k, major=Major.K)


def layout_for_b(*, dtype: Any, k: int, n: int) -> GmmaLayout:
    """Canonical GMMA layout for operand B (row-major KxN, MN-major)."""
    elem_bytes = _dtype_bytes(dtype)
    return pick_gmma_layout(elem_bytes=elem_bytes, m_or_n=n, k=k, major=Major.MN)


def _dtype_bytes(dtype: Any) -> int:
    """Extract byte-size from a pyptx PtxType or similar."""
    if hasattr(dtype, "bits"):
        bits = dtype.bits
        return max(bits // 8, 1)
    if isinstance(dtype, int):
        return dtype
    raise TypeError(
        f"pyptx wgmma: cannot determine elem_bytes from {type(dtype).__name__}: {dtype!r}"
    )
