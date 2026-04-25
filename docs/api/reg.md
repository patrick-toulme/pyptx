# `pyptx.reg`

> This page is generated from source docstrings and public symbols.

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

## Public API

- [`Reg`](#reg)
- [`PipeRef`](#piperef)
- [`NegPred`](#negpred)
- [`RegArray`](#regarray)
- [`array`](#array)
- [`scalar`](#scalar)
- [`from_`](#from-)
- [`wgmma_frag`](#wgmma-frag)
- [`alloc`](#alloc)
- [`alloc_array`](#alloc-array)

<a id="reg"></a>

## `Reg`

- Kind: `class`

```python
class Reg(name: 'str', dtype: 'PtxType') -> 'None'
```

A symbolic register reference.

Supports arithmetic and comparison operators that emit PTX instructions
and return new Reg values.

### Members

#### `max(other: 'Any') -> "'Reg'"`

- Kind: `method`

Emit an integer ``max`` against ``other`` and return the result.

#### `dtype`

- Kind: `attribute`

- Value: `<member 'dtype' of 'Reg' objects>`

No docstring yet.

#### `name`

- Kind: `attribute`

- Value: `<member 'name' of 'Reg' objects>`

No docstring yet.

<a id="piperef"></a>

## `PipeRef`

- Kind: `class`

```python
class PipeRef(left: 'Reg', right: 'Reg') -> 'None'
```

Pipe operand for setp dual predicate: %p0|%p1.

### Members

#### `left`

- Kind: `attribute`

- Value: `<member 'left' of 'PipeRef' objects>`

No docstring yet.

#### `right`

- Kind: `attribute`

- Value: `<member 'right' of 'PipeRef' objects>`

No docstring yet.

<a id="negpred"></a>

## `NegPred`

- Kind: `class`

```python
class NegPred(reg: 'Reg') -> 'None'
```

Negated predicate: ~p → @!p.

### Members

#### `reg`

- Kind: `attribute`

- Value: `<member 'reg' of 'NegPred' objects>`

No docstring yet.

<a id="regarray"></a>

## `RegArray`

- Kind: `class`

```python
class RegArray(base: 'str', count: 'int', dtype: 'PtxType') -> 'None'
```

Array of registers from .reg .type %name<count>.

Indexing returns Reg objects: acc[0] → Reg('%f0', f32).

### Members

#### `regs() -> 'list[Reg]'`

- Kind: `method`

Materialize the array as a Python list of ``Reg`` objects.

#### `count`

- Kind: `property`

Number of registers in the declared array.

#### `hw_order(*, reverse: 'bool' = False) -> 'list[Reg]'`

- Kind: `method`

Return the register list in declaration or reversed order.

Hopper tensor-core instructions often consume fragments in the
opposite order from the natural ``reg.array`` declaration order.
Naming that choice is clearer than open-coding
``list(reversed(acc))`` at every call site.

#### `base`

- Kind: `property`

Base PTX register name used for this array declaration.

#### `dtype`

- Kind: `property`

Element dtype of each register in the array.

<a id="array"></a>

## `array`

- Kind: `function`

```python
array(dtype: 'PtxType', count: 'int', name: 'str | None' = None) -> 'RegArray'
```

Allocate an array of registers.

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

<a id="scalar"></a>

## `scalar`

- Kind: `function`

```python
scalar(dtype: 'PtxType', init: 'int | float | None' = None, name: 'str | None' = None) -> 'Reg'
```

Allocate a single register, optionally initialized.

Emits: .reg .{dtype} %name;
If init is given, also emits: mov.{dtype} %name, init;

Predicate registers share a single bulk ``.reg .pred %p<N>;``
declaration (grown as needed) so they don't collide with
``_emit_setp``'s pred allocation path — otherwise we'd get both
an individual ``.reg .pred %p1;`` and a bulk
``.reg .pred %p<2>;`` and ptxas rejects the duplicate.

<a id="from-"></a>

## `from_`

- Kind: `function`

```python
from_(src: 'Any', dtype: 'PtxType') -> 'Reg'
```

Allocate ``dtype`` and emit a single ``mov`` from ``src``.

This is the common prologue/helper pattern for special registers and
symbolic operands like ``"smem"``:

    tid = reg.from_(ptx.special.tid.x(), u32)
    smem_base = reg.from_("smem", u32)

<a id="wgmma-frag"></a>

## `wgmma_frag`

- Kind: `function`

```python
wgmma_frag(*, m: 'int', n: 'int', dtype: 'PtxType', name: 'str | None' = None) -> 'RegArray'
```

Allocate an accumulator fragment sized for dense Hopper WGMMA.

For the common dense accumulator shapes, Hopper uses ``m * n / 128``
registers of the accumulator dtype.

<a id="alloc"></a>

## `alloc`

- Kind: `function`

```python
alloc(dtype: 'PtxType') -> 'Reg'
```

Allocate a single register with an auto-assigned index.

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

<a id="alloc-array"></a>

## `alloc_array`

- Kind: `function`

```python
alloc_array(dtype: 'PtxType', count: 'int') -> 'RegArray'
```

Allocate an array of registers with an auto-assigned name.

Unlike reg.array(dtype, count, name=...), you don't pick the name.
Returns a RegArray that can be indexed to get individual Reg refs.

Usage:
    accs = reg.alloc_array(f32, 64)
    ptx.inst.add.f32(accs[0], accs[1], accs[2])
