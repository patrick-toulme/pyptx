# `pyptx`

> This page is generated from source docstrings and public symbols.

Top-level public API for :mod:`pyptx`.

`pyptx` is a Python DSL for writing PTX kernels directly while keeping
the underlying PTX model explicit.

The package is organized around a small number of public namespaces:

- ``kernel``: the decorator used to trace a Python function into PTX
- ``ptx``: instruction wrappers, control flow helpers, and PTX-specific
  structured sugar
- ``reg``: register allocation and register-level arithmetic helpers
- ``smem``: shared-memory allocation and addressing helpers
- ``Tile`` / ``Layout``: tensor boundary specs used by ``@kernel``
- ``intrinsic``: low-level escape hatch for non-standard instructions

Typical usage:

```python
from pyptx import kernel, reg, smem, ptx, Tile, Layout
from pyptx.types import bf16, f32, u32

@kernel(arch="sm_90a")
def tiny():
    tid = reg.from_(ptx.special.tid.x(), u32)
    value = tid + 1
    ptx.inst.mov.u32(tid, value)
    ptx.ret()
```

Most API reference pages on ``pyptx.dev`` are generated from the
docstrings in this package, so the module docstrings are intended to
serve as the high-level entry point into each namespace.

## Public API

- [`kernel`](#kernel)
- [`reg`](#reg)
- [`smem`](#smem)
- [`ptx`](#ptx)
- [`intrinsic`](#intrinsic)
- [`Tile`](#tile)
- [`Layout`](#layout)
- [`differentiable_kernel`](#differentiable-kernel)

<a id="kernel"></a>

## `kernel`

- Kind: `function`

```python
kernel(fn: 'Callable | None' = None, *, arch: 'str' = 'sm_90a', version: 'tuple[int, int] | None' = None, in_specs: 'Sequence[Tile] | None' = None, out_specs: 'Sequence[Tile] | None' = None, grid: 'Any' = None, block: 'tuple[int, int, int]' = (1, 1, 1), cluster: 'tuple[int, int, int]' = (1, 1, 1), smem: 'int' = 0, raw_params: 'Sequence[tuple[str, str]] | None' = None, extern_smem: 'bool' = False, reqntid: 'tuple[int, ...] | None' = None, raw_directives: 'Sequence[tuple[str, tuple]] | None' = None) -> 'Kernel | Callable[[Callable], Kernel]'
```

Decorator to define a PTX kernel.

Can be used with or without arguments:

    @kernel
    def simple(): ...

    @kernel(arch="sm_100a")
    def blackwell(): ...

    @kernel(
        in_specs=(Tile("M", "K", bf16), Tile("K", "N", bf16)),
        out_specs=(Tile("M", "N", f32),),
        grid=lambda M, N, K: (M // 128, N // 256),
        block=(128, 1, 1),
        cluster=(2, 1, 1),
        arch="sm_90a",
    )
    def gemm(A, B, C, *, BM=128): ...

<a id="reg"></a>

## `reg`

- Kind: `module`

- Target: `pyptx.reg`

Register allocation and register-level DSL sugar.

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

<a id="smem"></a>

## `smem`

- Kind: `module`

- Target: `pyptx.smem`

Shared-memory allocation, addressing, and barrier objects.

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

<a id="ptx"></a>

## `ptx`

- Kind: `module`

- Target: `pyptx.ptx`

PTX instruction namespace.

Every function in this module emits exactly one PTX instruction.
No hidden scheduling, no lowering passes. Ten calls = ten instructions.

Usage (inside a @kernel function):
    from pyptx import ptx

    ptx.wgmma.mma_async(shape=(64,256,16), dtype_d=f32, ...)
    ptx.cp.async.bulk.tensor_2d(dst=sA[0], src=A.tma_desc(), ...)
    ptx.mbarrier.wait(bar[0], phase)
    ptx.raw("tcgen05.mma.cta_group::1 ...;")

    with ptx.if_(is_producer):
        ...
    with ptx.else_():
        ...
    for k in ptx.range_(0, K, BK):
        ...

<a id="intrinsic"></a>

## `intrinsic`

- Kind: `function`

```python
intrinsic(fn: 'F') -> 'F'
```

Mark a function as a PTX intrinsic (named scope of PTX instructions).

The decorator wraps the function so that when it's called inside a
kernel trace, the statements it emits are collected into an
IntrinsicScope IR node named after the function.

The function's return value is preserved — this is purely a scope
annotation layer, not a transformation.

Nesting works: one intrinsic can call another, and both scopes will
show up in the IR.

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

<a id="differentiable-kernel"></a>

## `differentiable_kernel`

- Kind: `function`

```python
differentiable_kernel(forward_kernel, backward_kernel, **kwargs)
```

Wrap forward + backward pyptx kernels for ``torch.autograd``.

See :func:`pyptx.torch_support.differentiable_kernel` for full docs.
