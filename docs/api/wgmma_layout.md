# `pyptx.wgmma_layout`

> This page is generated from source docstrings and public symbols.

Canonical GMMA shared-memory layouts for wgmma.mma_async.

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

## Public API

- [`LayoutType`](#layouttype)
- [`Major`](#major)
- [`GmmaLayout`](#gmmalayout)
- [`pick_gmma_layout`](#pick-gmma-layout)
- [`layout_for_a`](#layout-for-a)
- [`layout_for_b`](#layout-for-b)

<a id="layouttype"></a>

## `LayoutType`

- Kind: `class`

```python
class LayoutType(*values)
```

wgmma descriptor layout_type field (bits [63:62]).

### Members

#### `INTERLEAVE`

- Kind: `attribute`

- Value: `<LayoutType.INTERLEAVE: 0>`

wgmma descriptor layout_type field (bits [63:62]).

#### `B128`

- Kind: `attribute`

- Value: `<LayoutType.B128: 1>`

wgmma descriptor layout_type field (bits [63:62]).

#### `B64`

- Kind: `attribute`

- Value: `<LayoutType.B64: 2>`

wgmma descriptor layout_type field (bits [63:62]).

#### `B32`

- Kind: `attribute`

- Value: `<LayoutType.B32: 3>`

wgmma descriptor layout_type field (bits [63:62]).

<a id="major"></a>

## `Major`

- Kind: `class`

```python
class Major(*values)
```

Which operand direction is the "leading" (fastest-varying) axis
in the original tile layout.

``K`` — the K dimension is fast (row-major A of shape (M, K),
col-major B of shape (K, N)).

``MN`` — the M or N dimension is fast (col-major A of shape (M, K),
row-major B of shape (K, N)).

### Members

#### `K`

- Kind: `attribute`

- Value: `<Major.K: 0>`

Which operand direction is the "leading" (fastest-varying) axis
in the original tile layout.

``K`` — the K dimension is fast (row-major A of shape (M, K),
col-major B of shape (K, N)).

``MN`` — the M or N dimension is fast (col-major A of shape (M, K),
row-major B of shape (K, N)).

#### `MN`

- Kind: `attribute`

- Value: `<Major.MN: 1>`

Which operand direction is the "leading" (fastest-varying) axis
in the original tile layout.

``K`` — the K dimension is fast (row-major A of shape (M, K),
col-major B of shape (K, N)).

``MN`` — the M or N dimension is fast (col-major A of shape (M, K),
row-major B of shape (K, N)).

<a id="gmmalayout"></a>

## `GmmaLayout`

- Kind: `class`

```python
class GmmaLayout(layout_type: 'LayoutType', leading_byte_offset: 'int', stride_byte_offset: 'int', tma_swizzle: 'str', smem_swizzle: 'str | None') -> None
```

A canonical GMMA shared-memory layout.

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

### Members

#### `swizzle_code`

- Kind: `property`

Integer value for ``ptx.wgmma.make_descriptor(swizzle=...)``.

<a id="pick-gmma-layout"></a>

## `pick_gmma_layout`

- Kind: `function`

```python
pick_gmma_layout(*, elem_bytes: 'int', m_or_n: 'int', k: 'int', major: 'Major') -> 'GmmaLayout'
```

Return the canonical GMMA layout for a (dtype, M/N, K, major) tile.

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

<a id="layout-for-a"></a>

## `layout_for_a`

- Kind: `function`

```python
layout_for_a(*, dtype: 'Any', m: 'int', k: 'int') -> 'GmmaLayout'
```

Canonical GMMA layout for operand A (row-major MxK, K-major).

Usage::

    la = layout_for_a(dtype=bf16, m=64, k=16)
    # la.layout_type  → LayoutType.B32
    # la.leading_byte_offset → 16
    # la.stride_byte_offset  → 256
    # la.tma_swizzle → "32B"
    # la.smem_swizzle → "32B"

<a id="layout-for-b"></a>

## `layout_for_b`

- Kind: `function`

```python
layout_for_b(*, dtype: 'Any', k: 'int', n: 'int') -> 'GmmaLayout'
```

Canonical GMMA layout for operand B (row-major KxN, MN-major).
