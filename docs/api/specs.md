# `pyptx.specs`

> This page is generated from source docstrings and public symbols.

Tensor boundary specifications for :func:`pyptx.kernel`.

The :class:`Tile` and :class:`Layout` APIs describe the contract between
the Python runtime world and the traced PTX kernel:

- logical tensor shape
- dtype
- layout
- optional TMA box shape metadata

These specs are used by runtime integrations to:

- resolve symbolic dimensions such as ``"M"`` or ``"K"``
- validate shapes and dtypes at call time
- synthesize TMA descriptors when required

Example:

```python
from pyptx import Tile, Layout, kernel
from pyptx.types import bf16, f32

@kernel(
    in_specs=(
        Tile("M", "K", bf16, Layout.ROW),
        Tile("K", "N", bf16, Layout.COL),
    ),
    out_specs=(Tile("M", "N", f32, Layout.ROW),),
    grid=lambda M, N, K: (M // 128, N // 256),
    block=(128, 1, 1),
    arch="sm_90a",
)
def gemm(A, B, C): ...
```

## Public API

- [`Layout`](#layout)
- [`Dim`](#dim)
- [`Tile`](#tile)
- [`unify_envs`](#unify-envs)

<a id="layout"></a>

## `Layout`

- Kind: `class`

```python
class Layout(*values)
```

Memory layout for a tile.

ROW   — row-major (C order)
COL   — column-major (Fortran order)
TMA_128B — TMA 128-byte swizzle (Hopper)
TMA_64B  — TMA 64-byte swizzle
TMA_32B  — TMA 32-byte swizzle
INTERLEAVED — CUTLASS interleaved layout

### Members

#### `ROW`

- Kind: `attribute`

- Value: `<Layout.ROW: 'row'>`

Memory layout for a tile.

ROW   — row-major (C order)
COL   — column-major (Fortran order)
TMA_128B — TMA 128-byte swizzle (Hopper)
TMA_64B  — TMA 64-byte swizzle
TMA_32B  — TMA 32-byte swizzle
INTERLEAVED — CUTLASS interleaved layout

#### `COL`

- Kind: `attribute`

- Value: `<Layout.COL: 'col'>`

Memory layout for a tile.

ROW   — row-major (C order)
COL   — column-major (Fortran order)
TMA_128B — TMA 128-byte swizzle (Hopper)
TMA_64B  — TMA 64-byte swizzle
TMA_32B  — TMA 32-byte swizzle
INTERLEAVED — CUTLASS interleaved layout

#### `TMA_128B`

- Kind: `attribute`

- Value: `<Layout.TMA_128B: 'tma_128b'>`

Memory layout for a tile.

ROW   — row-major (C order)
COL   — column-major (Fortran order)
TMA_128B — TMA 128-byte swizzle (Hopper)
TMA_64B  — TMA 64-byte swizzle
TMA_32B  — TMA 32-byte swizzle
INTERLEAVED — CUTLASS interleaved layout

#### `TMA_64B`

- Kind: `attribute`

- Value: `<Layout.TMA_64B: 'tma_64b'>`

Memory layout for a tile.

ROW   — row-major (C order)
COL   — column-major (Fortran order)
TMA_128B — TMA 128-byte swizzle (Hopper)
TMA_64B  — TMA 64-byte swizzle
TMA_32B  — TMA 32-byte swizzle
INTERLEAVED — CUTLASS interleaved layout

#### `TMA_32B`

- Kind: `attribute`

- Value: `<Layout.TMA_32B: 'tma_32b'>`

Memory layout for a tile.

ROW   — row-major (C order)
COL   — column-major (Fortran order)
TMA_128B — TMA 128-byte swizzle (Hopper)
TMA_64B  — TMA 64-byte swizzle
TMA_32B  — TMA 32-byte swizzle
INTERLEAVED — CUTLASS interleaved layout

#### `INTERLEAVED`

- Kind: `attribute`

- Value: `<Layout.INTERLEAVED: 'interleaved'>`

Memory layout for a tile.

ROW   — row-major (C order)
COL   — column-major (Fortran order)
TMA_128B — TMA 128-byte swizzle (Hopper)
TMA_64B  — TMA 64-byte swizzle
TMA_32B  — TMA 32-byte swizzle
INTERLEAVED — CUTLASS interleaved layout

<a id="dim"></a>

## `Dim`

- Kind: `namespace`

- Type: `_UnionGenericAlias`

No docstring yet.

### Members

#### `copy_with(params)`

- Kind: `method`

No docstring yet.

<a id="tile"></a>

## `Tile`

- Kind: `class`

```python
class Tile(*dims_and_dtype: 'Dim | PtxType | Layout', dtype: 'PtxType | None' = None, layout: 'Layout' = <Layout.ROW: 'row'>, tma_box: 'tuple[Dim, ...] | None' = None, tma_rank: 'int' = 2, tma_padding: 'bool' = False) -> 'None'
```

A tensor spec: shape + dtype + layout.

Shape dimensions can be integers (concrete) or strings (symbolic variables
that get bound to real integers at call time). For example:

    Tile("M", "K", bf16, Layout.ROW)

describes an MxK bfloat16 row-major tile where M and K are unknown at
decoration time but will be bound when the kernel is called with a
concrete JAX array.

``tma_box`` is the per-TMA-load box shape when this tile is consumed via
``cp.async.bulk.tensor``. Defaults to the full tensor (one load reads
everything). K-loop kernels that use ``Tile.wgmma_a(..., tile_k=...)``
set this implicitly so each TMA load brings in exactly one K slice.

``tma_rank`` selects the descriptor encoding used by the runtime when the
kernel body calls ``tensor.tma_desc()``:

- ``2``: normal rank-2 descriptor
- ``3``: Hopper-style reshaped rank-3 descriptor used by the high-perf
  handwritten GEMM examples

``tma_padding`` only matters for ``tma_rank=3`` and requests the padded
innermost store box used by the Hopper GEMM epilogue.

### Members

#### `layout`

- Kind: `attribute`

- Value: `<Layout.ROW: 'row'>`

Memory layout for a tile.

ROW   — row-major (C order)
COL   — column-major (Fortran order)
TMA_128B — TMA 128-byte swizzle (Hopper)
TMA_64B  — TMA 64-byte swizzle
TMA_32B  — TMA 32-byte swizzle
INTERLEAVED — CUTLASS interleaved layout

#### `tma_box`

- Kind: `attribute`

No docstring yet.

#### `tma_rank`

- Kind: `attribute`

- Value: `2`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

#### `tma_padding`

- Kind: `attribute`

- Value: `False`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

#### `rank`

- Kind: `property`

Number of dimensions in the tile shape.

#### `symbolic_dims`

- Kind: `property`

Return the symbolic dim names in this tile.

#### `resolve_shape(env: 'dict[str, int]') -> 'tuple[int, ...]'`

- Kind: `method`

Resolve symbolic dims using a name -> int environment.

#### `matches(array_shape: 'tuple[int, ...]', array_dtype_name: 'str') -> 'bool'`

- Kind: `method`

Check if a concrete JAX array matches this tile's structure.

Does NOT check symbolic dims — use extract_env for that.

#### `extract_env(array_shape: 'tuple[int, ...]') -> 'dict[str, int]'`

- Kind: `method`

Extract the {symbolic_dim: concrete_int} env from an array shape.

#### `wgmma_a(m: "'Dim'", k: "'Dim'", dtype: 'PtxType', *, tile_m: "'Dim | None'" = None, tile_k: "'Dim | None'" = None) -> "'Tile'"`

- Kind: `classmethod`

Tile for a wgmma A operand (row-major MxK, K-major).

Picks the matching ``Layout.TMA_*B`` automatically based on the
**per-TMA-load K width**, defaulting to the full tensor K. Pass
``tile_k`` explicitly to describe a K-loop kernel that loads a
narrower slice per iteration, and ``tile_m`` to describe a
multi-CTA kernel where each CTA loads a narrower M slice::

    # Whole-K load (one TMA, one wgmma step)
    Tile.wgmma_a(64, 16, bf16)

    # K=16 slices of a K=64 tensor (four TMA loads, four wgmma)
    Tile.wgmma_a(64, 64, bf16, tile_k=16)

    # Multi-CTA: 2048xK tensor, each CTA loads a 64xK slice
    Tile.wgmma_a(2048, 64, bf16, tile_m=64, tile_k=16)

The TMA descriptor's box is ``(tile_m, tile_k)`` — defaulting to
``(M, tile_k)`` when ``tile_m`` is omitted. The user's shared
memory allocation should be sized to match that box, e.g.
``smem.wgmma_tile(bf16, (tile_m, tile_k), major="K")``.

Symbolic dims are supported — if ``k`` (or ``tile_k``) is a
``str``, the layout defaults to ``Layout.ROW`` and the runtime
side is expected to resolve.

#### `wgmma_b(k: "'Dim'", n: "'Dim'", dtype: 'PtxType', *, tile_k: "'Dim | None'" = None, tile_n: "'Dim | None'" = None) -> "'Tile'"`

- Kind: `classmethod`

Tile for a wgmma B operand (row-major KxN, MN-major).

Same idea as :meth:`wgmma_a`. ``tile_n`` slices N (per-CTA col
tile) and ``tile_k`` slices K (per-iteration K slice). The TMA
descriptor's box is ``(tile_k, tile_n)`` — when either is
omitted the full tensor dim is used.

<a id="unify-envs"></a>

## `unify_envs`

- Kind: `function`

```python
unify_envs(envs: 'list[dict[str, int]]') -> 'dict[str, int]'
```

Merge multiple {dim: int} envs; error on conflicts.

If Tile("M", "K") and Tile("K", "N") are both inputs, the K dim must
agree between them — this function catches mismatches.
