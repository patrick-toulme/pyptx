# `pyptx.smem`

> This page is generated from source docstrings and public symbols.

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

## Public API

- [`SharedAlloc`](#sharedalloc)
- [`SharedSlice`](#sharedslice)
- [`MbarrierArray`](#mbarrierarray)
- [`MbarrierRef`](#mbarrierref)
- [`base`](#base)
- [`load`](#load)
- [`store`](#store)
- [`alloc`](#alloc)
- [`wgmma_tile`](#wgmma-tile)
- [`alloc_with_layout`](#alloc-with-layout)
- [`mbarrier`](#mbarrier)
- [`apply_swizzle`](#apply-swizzle)

<a id="sharedalloc"></a>

## `SharedAlloc`

- Kind: `class`

```python
class SharedAlloc(name: 'str', dtype: 'PtxType', shape: 'tuple[int, ...]', swizzle: 'str | None', gmma_layout: "'object | None'" = None, gmma_major: 'str | None' = None, byte_offset: 'int' = 0) -> 'None'
```

Handle to a shared memory allocation.

Indexing with a stage index returns a SharedSlice representing an offset
into the allocation, suitable for passing to ptx.cp.async instructions.

``gmma_layout`` is set (non-None) when this alloc was produced by
``smem.wgmma_tile`` — it carries the ``GmmaLayout`` needed to
auto-build a wgmma descriptor. The ``gmma_major`` field is a string
``"K"`` or ``"MN"`` matching the operand orientation.

### Members

#### `byte_offset`

- Kind: `attribute`

- Value: `<member 'byte_offset' of 'SharedAlloc' objects>`

No docstring yet.

#### `dtype`

- Kind: `attribute`

- Value: `<member 'dtype' of 'SharedAlloc' objects>`

No docstring yet.

#### `gmma_layout`

- Kind: `attribute`

- Value: `<member 'gmma_layout' of 'SharedAlloc' objects>`

No docstring yet.

#### `gmma_major`

- Kind: `attribute`

- Value: `<member 'gmma_major' of 'SharedAlloc' objects>`

No docstring yet.

#### `name`

- Kind: `attribute`

- Value: `<member 'name' of 'SharedAlloc' objects>`

No docstring yet.

#### `shape`

- Kind: `attribute`

- Value: `<member 'shape' of 'SharedAlloc' objects>`

No docstring yet.

#### `swizzle`

- Kind: `attribute`

- Value: `<member 'swizzle' of 'SharedAlloc' objects>`

No docstring yet.

<a id="sharedslice"></a>

## `SharedSlice`

- Kind: `class`

```python
class SharedSlice(alloc: 'SharedAlloc', stage: 'int') -> 'None'
```

A stage-indexed slice of a shared allocation.

### Members

#### `name`

- Kind: `property`

Underlying shared-memory symbol name for this slice.

#### `alloc`

- Kind: `attribute`

- Value: `<member 'alloc' of 'SharedSlice' objects>`

No docstring yet.

#### `stage`

- Kind: `attribute`

- Value: `<member 'stage' of 'SharedSlice' objects>`

No docstring yet.

<a id="mbarrierarray"></a>

## `MbarrierArray`

- Kind: `class`

```python
class MbarrierArray(name: 'str', count: 'int', byte_offset: 'int' = 0) -> 'None'
```

Array of mbarrier objects in shared memory.

Indexing returns an MbarrierRef for use in ptx.mbarrier.* calls.

``byte_offset`` is the byte offset within dynamic SMEM (when
``force_dynamic_smem`` is active). Each individual mbarrier is 8
bytes, so ``MbarrierRef`` for index *i* lives at
``byte_offset + i * 8``.

### Members

#### `byte_offset`

- Kind: `attribute`

- Value: `<member 'byte_offset' of 'MbarrierArray' objects>`

No docstring yet.

#### `count`

- Kind: `attribute`

- Value: `<member 'count' of 'MbarrierArray' objects>`

No docstring yet.

#### `name`

- Kind: `attribute`

- Value: `<member 'name' of 'MbarrierArray' objects>`

No docstring yet.

<a id="mbarrierref"></a>

## `MbarrierRef`

- Kind: `class`

```python
class MbarrierRef(array: 'MbarrierArray', idx: 'int') -> 'None'
```

Reference to a single mbarrier object.

``byte_offset`` is the byte offset of this specific mbarrier
within dynamic SMEM: ``array.byte_offset + idx * 8``.  When the
array was allocated in dynamic SMEM mode (``name == "dyn_smem"``),
instruction emitters use this offset for addressing.

### Members

#### `name`

- Kind: `property`

Underlying shared-memory symbol name for this mbarrier array.

#### `byte_offset`

- Kind: `property`

Byte offset of this mbarrier within dynamic SMEM.

#### `array`

- Kind: `attribute`

- Value: `<member 'array' of 'MbarrierRef' objects>`

No docstring yet.

#### `idx`

- Kind: `attribute`

- Value: `<member 'idx' of 'MbarrierRef' objects>`

No docstring yet.

<a id="base"></a>

## `base`

- Kind: `function`

```python
base(name: 'str | None' = None)
```

Return a u32 register holding the base address of extern shared memory.

<a id="load"></a>

## `load`

- Kind: `function`

```python
load(dtype: 'PtxType', address)
```

Emit ``ld.shared.{dtype}`` and return the loaded register.

<a id="store"></a>

## `store`

- Kind: `function`

```python
store(dtype: 'PtxType', address, value) -> 'None'
```

Emit ``st.shared.{dtype}``.

<a id="alloc"></a>

## `alloc`

- Kind: `function`

```python
alloc(dtype: 'PtxType', shape: 'tuple[int, ...] | int', swizzle: 'str | None' = None, align: 'int | None' = None, name: 'str | None' = None) -> 'SharedAlloc'
```

Allocate shared memory.

Emits: .shared [.align N] .b8 name[bytes];

Args:
    dtype: Element type (e.g. bf16, f32).
    shape: Shape as tuple (e.g. (STAGES, BM, BK)) or flat int.
    swizzle: Swizzle mode string (e.g. '128B'). Metadata only for now.
    align: Byte alignment. Defaults to 128.
    name: Variable name. Auto-generated if not given.

Returns:
    SharedAlloc handle for use in ptx.cp.async and ptx.stmatrix calls.

<a id="wgmma-tile"></a>

## `wgmma_tile`

- Kind: `function`

```python
wgmma_tile(dtype: 'PtxType', shape: 'tuple[int, int]', major: 'str' = 'K', *, align: 'int | None' = None, name: 'str | None' = None) -> 'SharedAlloc'
```

Allocate a shared-memory tile in the canonical GMMA layout for
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

<a id="alloc-with-layout"></a>

## `alloc_with_layout`

- Kind: `function`

```python
alloc_with_layout(dtype: 'PtxType', shape: 'tuple[int, ...] | int', swizzle: 'str | None' = None, align: 'int | None' = None, name: 'str | None' = None, *, gmma_layout: "'object | None'" = None, gmma_major: 'str | None' = None) -> 'SharedAlloc'
```

Internal: allocate SMEM and attach GMMA layout metadata.

Same as ``alloc`` but threads the gmma_layout / gmma_major fields
through to the returned SharedAlloc. Most users should call
``wgmma_tile`` or ``alloc``, not this directly.

<a id="mbarrier"></a>

## `mbarrier`

- Kind: `function`

```python
mbarrier(count: 'int', name: 'str | None' = None) -> 'MbarrierArray'
```

Allocate an array of mbarrier objects in shared memory.

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

<a id="apply-swizzle"></a>

## `apply_swizzle`

- Kind: `function`

```python
apply_swizzle(logical_offset: "'Reg'", swizzle: 'str | None') -> "'Reg'"
```

Apply GMMA swizzle to a logical byte offset, returning the physical offset.

``swizzle`` is ``"32B"``, ``"64B"``, ``"128B"``, or ``None``/``"NONE"``
(identity).  Emits 3 ALU instructions for non-trivial swizzles.
