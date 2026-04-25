# `pyptx.ptx`

> This page is generated from source docstrings and public symbols.

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

## Public API

- [`special`](#special)
- [`sreg`](#sreg)
- [`loc`](#loc)
- [`file_`](#file-)
- [`pragma`](#pragma)
- [`var`](#var)
- [`ctaid_x`](#ctaid-x)
- [`ctaid_y`](#ctaid-y)
- [`global_ptrs`](#global-ptrs)
- [`warp`](#warp)
- [`if_`](#if-)
- [`else_`](#else-)
- [`scope`](#scope)
- [`loop`](#loop)
- [`PipelineState`](#pipelinestate)
- [`pipeline`](#pipeline)
- [`expr`](#expr)
- [`range_`](#range-)
- [`raw`](#raw)
- [`wgmma`](#wgmma)
- [`cp`](#cp)
- [`mbarrier`](#mbarrier)
- [`fence`](#fence)
- [`stmatrix`](#stmatrix)
- [`stmatrix_x4_trans_f32_bf16`](#stmatrix-x4-trans-f32-bf16)
- [`ldmatrix`](#ldmatrix)
- [`bar`](#bar)
- [`NamedBarrier`](#namedbarrier)
- [`named_barrier`](#named-barrier)
- [`setmaxnreg`](#setmaxnreg)
- [`kloop`](#kloop)
- [`selp`](#selp)
- [`tma`](#tma)
- [`inst`](#inst)
- [`pipe`](#pipe)
- [`mov`](#mov)
- [`add`](#add)
- [`ret`](#ret)
- [`bra`](#bra)
- [`label`](#label)
- [`addr`](#addr)
- [`param`](#param)
- [`tcgen05`](#tcgen05)
- [`setmaxnreg_inc`](#setmaxnreg-inc)
- [`setmaxnreg_dec`](#setmaxnreg-dec)
- [`elect_sync`](#elect-sync)
- [`cluster`](#cluster)
- [`cvta`](#cvta)
- [`sub`](#sub)
- [`mul`](#mul)
- [`mad`](#mad)
- [`shl`](#shl)
- [`shr`](#shr)
- [`setp`](#setp)
- [`cvt`](#cvt)
- [`ld`](#ld)
- [`st`](#st)
- [`and_`](#and-)
- [`or_`](#or-)
- [`xor_`](#xor-)
- [`not_`](#not-)
- [`TYPED_WRAPPER_CODEGEN`](#typed-wrapper-codegen)

<a id="special"></a>

## `special`

- Kind: `namespace`

- Type: `_Special`

ptx.special.tid.x(), ptx.special.laneid(), etc.

### Members

#### `tid`

- Kind: `property`

No docstring yet.

#### `ntid`

- Kind: `property`

No docstring yet.

#### `ctaid`

- Kind: `property`

No docstring yet.

#### `nctaid`

- Kind: `property`

No docstring yet.

#### `laneid() -> 'Reg'`

- Kind: `method`

No docstring yet.

#### `warpid() -> 'Reg'`

- Kind: `method`

No docstring yet.

#### `clock() -> 'Reg'`

- Kind: `method`

No docstring yet.

<a id="sreg"></a>

## `sreg`

- Kind: `function`

```python
sreg(name: 'str') -> 'Reg'
```

Reference any PTX special register by name.

Usage:
    ptx.sreg("%cluster_ctarank")
    ptx.sreg("%clusterid.x")
    ptx.sreg("%smid")

For common ones, prefer ptx.special.tid.x() etc.

<a id="loc"></a>

## `loc`

- Kind: `function`

```python
loc(file_idx: 'int', line: 'int', col: 'int' = 0) -> 'None'
```

Emit a .loc debug directive for source attribution.

Usage: ptx.loc(1, 40, 0)  →  .loc 1 40 0

<a id="file-"></a>

## `file_`

- Kind: `function`

```python
file_(file_idx: 'int', filename: 'str') -> 'None'
```

Emit a .file debug directive.

Usage: ptx.file_(1, "kernel.py")  →  .file 1 "kernel.py"

<a id="pragma"></a>

## `pragma`

- Kind: `function`

```python
pragma(value: 'str') -> 'None'
```

Emit a .pragma directive.

Usage: ptx.pragma("nounroll")  →  .pragma "nounroll";

<a id="var"></a>

## `var`

- Kind: `function`

```python
var(state_space: 'str', dtype: 'PtxType', name: 'str', *, size: 'int | None' = None, align: 'int | None' = None, linking: 'str | None' = None) -> 'str'
```

Declare a variable in any state space.

Usage:
    ptx.var("shared", b8, "smem", size=49152, align=128)
    ptx.var("param", b32, "param0")
    ptx.var("global", f32, "output", size=1024, linking="visible")

Returns the variable name (for use with ptx.addr()).

<a id="ctaid-x"></a>

## `ctaid_x`

- Kind: `function`

```python
ctaid_x() -> 'Reg'
```

Convenience alias for ``%ctaid.x``.

<a id="ctaid-y"></a>

## `ctaid_y`

- Kind: `function`

```python
ctaid_y() -> 'Reg'
```

Convenience alias for ``%ctaid.y``.

<a id="global-ptrs"></a>

## `global_ptrs`

- Kind: `function`

```python
global_ptrs(*params: 'Any') -> 'tuple[Reg, ...]'
```

Load kernel parameter pointers into fresh global-space registers.

For each kernel parameter (typically a ``TensorSpec`` passed into the
``@kernel`` function body) this emits the canonical prologue pair::

    ld.param.u64     %rd_n, [param_name];
    cvta.to.global.u64 %rd_n, %rd_n;

and returns a tuple of ``Reg`` objects — one global-space b64
pointer per parameter. Kernels then write::

    px, pw, py = ptx.global_ptrs(X, W, Y)

instead of six lines of boilerplate per invocation. Single
parameter still returns a 1-tuple; call-site unpack with a
trailing comma::

    (px,) = ptx.global_ptrs(X)

<a id="warp"></a>

## `warp`

- Kind: `namespace`

- Type: `_Warp`

``ptx.warp.reduce_sum(val)`` / ``reduce_max(val)`` / ``reduce_min(val)``
— in-place warp-scope reductions.

``width`` is the reduction group size in lanes (default 32 = full
warp). Pass ``width=4`` for the per-row reduction across the
4-lane groups that share a row in the ``wgmma.m64nN`` output
fragment layout — this is what Flash Attention's online softmax
needs to turn its per-thread row_max into a full-row max.

### Members

#### `reduce_sum(val: "'Reg'", *, width: 'int' = 32) -> 'None'`

- Kind: `method`

No docstring yet.

#### `reduce_max(val: "'Reg'", *, width: 'int' = 32) -> 'None'`

- Kind: `method`

No docstring yet.

#### `reduce_min(val: "'Reg'", *, width: 'int' = 32) -> 'None'`

- Kind: `method`

No docstring yet.

<a id="if-"></a>

## `if_`

- Kind: `function`

```python
if_(pred_reg: 'Reg | NegPred') -> 'Generator[None, None, None]'
```

Conditional block. Emits one branch instruction.

Usage:
    with ptx.if_(is_producer):
        ...   # body executes only if pred is true

    # Optional chained else:
    with ptx.if_(p):
        ...
    with ptx.else_():
        ...

<a id="else-"></a>

## `else_`

- Kind: `function`

```python
else_() -> 'Generator[None, None, None]'
```

Else block — must follow an if_() block.

Usage:
    with ptx.if_(pred):
        ...
    with ptx.else_():
        ...

<a id="scope"></a>

## `scope`

- Kind: `function`

```python
scope() -> 'Generator[None, None, None]'
```

Open a PTX ``{ }`` block scope.

Register declarations inside the scope are emitted inline (block-local)
rather than hoisted to the function top. This maps directly to PTX's
nested ``{ ... }`` scoping, where ``.reg`` declarations are local to
the enclosing braces.

Usage::

    with ptx.scope():
        tmp = reg.scalar(b32, name="tmp")
        ptx.inst.mov.b32(tmp, 42)
    # tmp is out of scope here; the name can be reused in another scope

<a id="loop"></a>

## `loop`

- Kind: `function`

```python
loop(label_name: 'str', *, pred: "'Reg | NegPred | None'" = None) -> 'Generator[None, None, None]'
```

Emit a PTX loop: ``label: ... @pred bra label;``

The label is emitted on entry, and a conditional backward branch
is emitted on exit. The body goes inside the ``with`` block.

Usage::

    with ptx.loop("k_loop", pred=p[14]):
        # ... loop body ...
        # at the end, emits: @%p14 bra k_loop;

For unconditional loops (persistent tile loops), omit pred::

    with ptx.loop("tile_loop"):
        # ... body ...
        # emits: bra.uni tile_loop;

<a id="pipelinestate"></a>

## `PipelineState`

- Kind: `class`

```python
class PipelineState(n_stages: 'int', *, cursor: 'Reg | None' = None, phase: 'Reg | None' = None) -> 'None'
```

Loop-carried stage cursor + phase bit for ring-buffered pipelines.

``advance()`` emits the common Hopper pattern:
- compare cursor against ``n_stages``
- flip the phase on wrap
- return the wrapped stage index
- update the loop-carried cursor in place

### Members

#### `advance() -> 'tuple[Reg, Reg]'`

- Kind: `method`

Advance the ring and return ``(stage, phase)``.

#### `cursor`

- Kind: `attribute`

- Value: `<member 'cursor' of 'PipelineState' objects>`

No docstring yet.

#### `n_stages`

- Kind: `attribute`

- Value: `<member 'n_stages' of 'PipelineState' objects>`

No docstring yet.

#### `phase`

- Kind: `attribute`

- Value: `<member 'phase' of 'PipelineState' objects>`

No docstring yet.

<a id="pipeline"></a>

## `pipeline`

- Kind: `function`

```python
pipeline(n_stages: 'int', *, cursor: 'Reg | None' = None, phase: 'Reg | None' = None) -> 'PipelineState'
```

Create a loop-carried pipeline stage/phase helper.

<a id="expr"></a>

## `expr`

- Kind: `function`

```python
expr() -> 'Generator[None, None, None]'
```

Capture a Python expression's PTX instructions into one CompoundExpr.

All ``ptx.inst.*`` calls and Reg operator overloads inside the block
are buffered, then emitted as a single :class:`CompoundExpr` IR node.
Instructions execute in Python evaluation order (which IS the correct
data-dependency order for expressions).

Usage::

    with ptx.expr():
        rd[26] = ((r[192] - 8192) & 0x3FF80) >> 4 | CONST

The PTX output is identical to writing the instructions individually.
The benefit is a compact, readable Python source.

<a id="range-"></a>

## `range_`

- Kind: `function`

```python
range_(start, stop: 'int', step: 'int' = 1) -> 'Generator[Reg, None, None]'
```

Staged loop. Emits PTX branches for the loop structure.

``start`` can be a Python ``int`` or a ``Reg`` (for persistent
kernel scheduling where the loop starts at ``ctaid.x``).

Usage:
    for k in ptx.range_(0, K, BK):
        ...  # k is a Reg holding the loop variable

    # Persistent: start from ctaid.x
    for tile in ptx.range_(cta_id, total_tiles, NUM_SM):
        ...

Emits:
    mov.s32 %rN, start;    (or mov.s32 %rN, %start_reg;)
    $loop:
    setp.ge.s32 %pN, %rN, stop;
    @%pN bra $endloop;
    ... body ...
    add.s32 %rN, %rN, step;
    bra $loop;
    $endloop:

<a id="raw"></a>

## `raw`

- Kind: `function`

```python
raw(text: 'str') -> 'None'
```

Emit a raw PTX instruction string.

Usage: ptx.raw("tcgen05.mma.cta_group::1.kind::f16 ...;")

Parses the text and records the resulting IR instruction(s).

<a id="wgmma"></a>

## `wgmma`

- Kind: `namespace`

- Type: `_Wgmma`

ptx.wgmma.mma_async(...), ptx.wgmma.fence(), etc.

### Members

#### `MASKED_DESC_B128`

- Kind: `attribute`

- Value: `4611686293305360384`

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

#### `mma_async(*, shape: 'tuple[int, int, int]', dtype_d: 'PtxType', dtype_a: 'PtxType', dtype_b: 'PtxType', d: 'RegArray', a: 'Any', b: 'Any', scale_d: "'Reg | bool | int'" = False, scale_a: 'int' = 1, scale_b: 'int' = 1, trans_a: 'int' = 0, trans_b: 'int' = 0, a_k_offset: 'int' = 0, b_k_offset: 'int' = 0, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit ``wgmma.mma_async.sync.aligned.{shape}.{dtype_d}.{dtype_a}.{dtype_b}``.

Operand layout for the dense ``.f32.bf16.bf16`` / ``.f32.f16.f16``
form (from PTX ISA §9.7.14.5)::

    wgmma.mma_async ... d-vec, a-desc, b-desc, scale-d,
                        imm-scale-a, imm-scale-b,
                        imm-trans-a, imm-trans-b;

- ``d``: register vector holding the accumulator.
- ``a``, ``b``: u64 shared-memory descriptor registers (see
  :meth:`make_descriptor`).
- ``scale_d``: ``.pred`` operand. ``False`` (default) means the
  instruction computes ``D = A * B`` (fresh accumulator, ignoring
  whatever was in D). ``True`` means ``D = D + A * B``
  (accumulate into existing D). You may also pass a Reg of
  dtype ``pred`` for runtime selection.
- ``scale_a``, ``scale_b``: ``.s32`` immediates, must be 1 or -1.
  -1 negates the corresponding operand.
- ``trans_a``, ``trans_b``: ``.s32`` immediates, 0 or 1.
  Transpose flag for A / B.

#### `fence(*, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit wgmma.fence.sync.aligned;

#### `commit_group(*, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit wgmma.commit_group.sync.aligned;

#### `wait_group(n: 'int', *, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit wgmma.wait_group.sync.aligned N;

#### `SWIZZLE_NONE`

- Kind: `attribute`

- Value: `0`

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

#### `SWIZZLE_128B`

- Kind: `attribute`

- Value: `1`

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

#### `SWIZZLE_64B`

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

#### `SWIZZLE_32B`

- Kind: `attribute`

- Value: `3`

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

#### `auto_descriptor(smem_base: 'Any', *, dtype: 'Any', shape: 'tuple[int, int]', major: 'str | int') -> "'Reg'"`

- Kind: `method`

Build a wgmma descriptor from a shape + dtype + major hint.

Zero magic-number API: pyptx picks the right canonical GMMA
layout (INTERLEAVE / B32 / B64 / B128) and derives the
leading/stride byte offsets automatically.

Args:
    smem_base: ``SharedAlloc`` / ``SharedSlice`` / ``Reg``
        pointing at the start of the shared memory tile. The
        caller is responsible for allocating the tile with a
        matching ``swizzle=...`` (see the returned layout for
        the right name) and, if the tile is TMA-loaded, for
        using the matching ``Layout.TMA_*B`` in the @kernel
        Tile spec — swizzle on write and swizzle on read must
        be the same or the data comes back permuted.
    dtype: pyptx element type (``bf16``, ``f16``, ``f32``...).
    shape: ``(M, K)`` for A or ``(K, N)`` for B.
    major: ``"K"`` (row-major A / col-major B) or ``"MN"``
        (col-major A / row-major B). Must match the
        ``trans_a`` / ``trans_b`` flags on the subsequent
        ``wgmma.mma_async`` call.

Returns: a fresh ``Reg`` holding the 64-bit descriptor.

#### `make_descriptor(smem_base: 'Any', *, leading_byte_offset: 'int', stride_byte_offset: 'int', swizzle: 'int' = 0, base_offset: 'int' = 0, addr_byte_offset: 'int' = 0) -> "'Reg'"`

- Kind: `method`

Build a wgmma shared-memory descriptor in a fresh u64 register.

wgmma.mma_async takes A and B as 64-bit descriptors that encode:
  bits [13:0]   start_addr      = shared_addr >> 4
  bits [29:16]  leading_offset  = leading_byte_offset >> 4
  bits [45:32]  stride_offset   = stride_byte_offset >> 4
  bits [51:49]  base_offset     (0 unless swizzle requires it)
  bits [63:62]  swizzle mode    (0=none, 1=128B, 2=64B, 3=32B)

``addr_byte_offset`` is added to the smem_base address before
the start_addr field is computed. This is how sub-tile descriptors
work for BK > 16 GEMMs: each of the 4 wgmma calls within a
K-tile references a different 16-column slice of the A and B
allocations by adding ``j * slice_bytes`` to the base.

This helper emits PTX that computes the descriptor at kernel
runtime by taking the shared memory base address (which is
known to ptxas as a relocatable symbol) and OR-ing in the
compile-time-constant leading/stride/base/swizzle fields.

Args:
    smem_base: a ``SharedAlloc`` / ``SharedSlice`` / ``Reg`` holding
        (or naming) the shared memory base the descriptor refers to.
        If a ``SharedAlloc``/``SharedSlice`` is passed, we emit an
        extra ``mov.u64`` to lift the symbolic name into a register.
    leading_byte_offset: constant int — the leading dimension stride
        of the matrix tile in bytes. For a row-major 16x8 bf16
        tile this is 16 (one row).
    stride_byte_offset: constant int — the stride between "core
        matrices" in the tile. For a 16x8 bf16 tile that's split
        into two 8x8 core matrices vertically, this is 128
        (8 rows * 16 bytes/row).
    swizzle: one of ``SWIZZLE_{NONE,128B,64B,32B}``.
    base_offset: swizzle base offset (0-7); 0 for most uses.

Returns: a fresh ``Reg`` of dtype ``u64`` holding the descriptor,
usable directly as ``a=`` / ``b=`` to ``wgmma.mma_async``.

#### `masked_descriptor(smem_addr: 'Any', *, byte_offset: 'int' = 0, mask: 'int' = 262016, const_bits: 'int' = 4611686293305360384) -> "'Reg'"`

- Kind: `method`

Build a descriptor from a computed shared-memory address.

This is the lower-level Hopper GEMM pattern used by handwritten
kernels that derive descriptors from lane/stage-specific shared
addresses:

  tmp  = smem_addr + byte_offset
  bits = tmp & mask
  idx  = bits >> 4
  desc = cvt.u64.u32(idx) | const_bits

<a id="cp"></a>

## `cp`

- Kind: `namespace`

- Type: `_Cp`

No docstring yet.

### Members

#### `async_`

- Kind: `property`

No docstring yet.

#### `async_bulk`

- Kind: `property`

No docstring yet.

<a id="mbarrier"></a>

## `mbarrier`

- Kind: `namespace`

- Type: `_Mbarrier`

ptx.mbarrier — Hopper mbarrier primitives.

Contract for bracket wrapping: ``mbar`` arguments are always
converted to address operands (``[mbar_0]``). State and predicate
output registers are allocated inside the wrapper and returned from
the ``arrive``/``arrive_expect_tx``/``try_wait`` calls so the caller
can use them directly without boilerplate.

### Members

#### `init(mbar: 'Any', count: 'int', *, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit ``mbarrier.init.shared::cta.b64 [mbar], count;``.

#### `inval(mbar: 'Any', *, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit ``mbarrier.inval.shared.b64 [mbar];``.

#### `arrive(mbar: 'Any', *, pred: 'Reg | NegPred | None' = None) -> "'Reg'"`

- Kind: `method`

Emit ``mbarrier.arrive.shared.b64 state, [mbar];``.

Returns the freshly allocated ``b64`` state register so callers
can feed it to a subsequent wait if they need to. Users that
don't care about the token can ignore the return value.

#### `arrive_expect_tx(mbar: 'Any', tx_count: "int | 'Reg'", *, pred: 'Reg | NegPred | None' = None) -> "'Reg'"`

- Kind: `method`

Emit ``mbarrier.arrive.expect_tx.shared::cta.b64 state, [mbar], tx_count;``.

Used by the thread that issues a TMA load: records the expected
transaction-size so the mbarrier knows when the async bulk copy
has fully completed. Returns the state register.

#### `expect_tx(mbar: 'Any', tx_count: "int | 'Reg'", *, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit ``mbarrier.expect_tx.shared.b64 [mbar], tx_count;``.

The standalone form (no arrive, no state register output).

#### `try_wait(mbar: 'Any', phase: "'Reg | int'", *, parity: 'bool' = True, pred: 'Reg | NegPred | None' = None) -> "'Reg'"`

- Kind: `method`

Emit ``mbarrier.try_wait{.parity}.shared.b64 p, [mbar], phase;``.

Returns the freshly allocated ``.pred`` register (``p``) that is
true when the wait completed. The typical use is inside a busy
loop that branches back to the try_wait label until ``p`` is true.

When ``parity=True`` (the default) the instruction is the phase-
bit flavor used with a single-bit phase register; use
``parity=False`` for the token-based form.

#### `wait(mbar: 'Any', phase: "'Reg | int'", *, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit a ``mbarrier.try_wait.parity`` spin loop that blocks the
calling thread until the barrier completes.

Produces roughly::

    wait_loop:
        mbarrier.try_wait.parity.shared.b64 p, [mbar], phase;
        @!p bra wait_loop;

If you need non-blocking behavior, call ``try_wait`` directly and
branch on its return value.

#### `array(base: 'Any', byte_offset: 'int', count: 'int') -> '_BarrierArray'`

- Kind: `method`

Create an indexable barrier array rooted at ``base + byte_offset``.

<a id="fence"></a>

## `fence`

- Kind: `namespace`

- Type: `_Fence`

No docstring yet.

### Members

#### `proxy_async(*, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit fence.proxy.async;

#### `proxy_async_shared_cta(*, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit ``fence.proxy.async.shared::cta;``.

The narrower-scope variant required to make an mbarrier init
(which is a generic shared-memory write) visible to the TMA
async proxy. Without this scope, the async proxy's view of
the mbarrier can lag the thread that init'd it, so
``cp.async.bulk.tensor.*`` with ``.mbarrier::complete_tx::bytes``
silently signals a stale barrier and the corresponding
``mbarrier.try_wait`` never completes.

Canonical Hopper pattern for a TMA-loaded pipeline stage::

    mbarrier.init [bar], count;
    fence.proxy.async.shared::cta;
    mbarrier.arrive.expect_tx [bar], tx_bytes;
    cp.async.bulk.tensor.Nd.shared::cluster.global...;
    mbarrier.try_wait.parity [bar], phase;

#### `proxy_async_generic_acquire_shared_cluster(*, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit the cluster-scope acquire fence used by Mosaic GPU collectives.

PTX spelling::

    fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;

Mosaic inserts this after waiting on a cluster-visible hand-off before
reusing a collective TMA pipeline slot. It is narrower than a generic
fence and pairs with Blackwell cluster-shared async-proxy state.

#### `mbarrier_init(*, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit fence.mbarrier_init.release.cluster;

<a id="stmatrix"></a>

## `stmatrix`

- Kind: `function`

```python
stmatrix(*, smem: 'Any', regs: 'RegArray | list', layout: 'str' = 'x4', trans: 'bool' = False, shape: 'str' = 'm8n8', pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit stmatrix.sync.aligned.{shape}.{count}[.trans].shared.b16.

The ``layout`` kwarg accepts compound forms like "x4.trans" for
backwards compatibility with raw PTX-style strings — they're split
on dots into separate modifiers.

Args:
    smem: destination shared-memory address
    regs: source registers (RegArray or list of Regs)
    layout: either just "x4" / "x2" / "x1", or a compound like
            "x4.trans" which auto-sets ``trans=True``
    trans: whether to emit the .trans modifier (default False)
    shape: tile shape, defaults to "m8n8"
    pred: optional predicate

<a id="stmatrix-x4-trans-f32-bf16"></a>

## `stmatrix_x4_trans_f32_bf16`

- Kind: `function`

```python
stmatrix_x4_trans_f32_bf16(*, frag: "RegArray | list['Reg']", smem_base: 'Any', lane: "'Reg'", row_stride: 'int', tmp_bf16: "list['Reg'] | None" = None, tmp_pack: "list['Reg'] | None" = None) -> 'None'
```

Pack an f32 fragment to bf16 and store it via ``stmatrix.x4.trans``.

<a id="ldmatrix"></a>

## `ldmatrix`

- Kind: `function`

```python
ldmatrix(*, dst: 'RegArray | list[Reg]', src: 'Any', layout: 'str' = 'x4', trans: 'bool' = False, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit ldmatrix.sync.aligned.{layout}[.trans].shared.b16.

<a id="bar"></a>

## `bar`

- Kind: `namespace`

- Type: `_Bar`

No docstring yet.

### Members

#### `sync(n: 'Any' = 0, count: 'Any | None' = None, *, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit bar.sync N;

<a id="namedbarrier"></a>

## `NamedBarrier`

- Kind: `class`

```python
class NamedBarrier(barrier_id: 'Any', count: 'Any | None' = None) -> 'None'
```

Named CTA barrier with an optional participant count.

### Members

#### `sync(*, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit ``bar.sync`` using the stored barrier id and participant count.

#### `barrier_id`

- Kind: `attribute`

- Value: `<member 'barrier_id' of 'NamedBarrier' objects>`

No docstring yet.

#### `count`

- Kind: `attribute`

- Value: `<member 'count' of 'NamedBarrier' objects>`

No docstring yet.

<a id="named-barrier"></a>

## `named_barrier`

- Kind: `function`

```python
named_barrier(barrier_id: 'Any', *, count: 'Any | None' = None) -> 'NamedBarrier'
```

Create a named ``bar.sync`` wrapper.

<a id="setmaxnreg"></a>

## `setmaxnreg`

- Kind: `function`

```python
setmaxnreg(count: 'int', *, inc: 'bool' = True, pred: "'Reg | NegPred | None'" = None) -> 'None'
```

Emit ``setmaxnreg.{inc|dec}.sync.aligned.u32 count;``

Used for warp specialization: consumers increase registers (inc=True),
producers decrease (inc=False).

<a id="kloop"></a>

## `kloop`

- Kind: `function`

```python
kloop(total: "int | 'Reg'", *, unroll: 'int', body: 'Callable[[], None]', loop_label: 'str' = 'kloop') -> 'None'
```

Emit an unrolled counted loop with a peeled tail ladder.

<a id="selp"></a>

## `selp`

- Kind: `function`

```python
selp(dtype: "'PtxType'", dst: "'Reg'", true_val: 'Any', false_val: 'Any', pred_reg: "'Reg'", *, pred: "'Reg | NegPred | None'" = None) -> 'None'
```

Emit ``selp.{type} dst, true_val, false_val, pred;``

Ternary select: ``dst = pred ? true_val : false_val``.

<a id="tma"></a>

## `tma`

- Kind: `namespace`

- Type: `_Tma`

High-level TMA load/store with 3D layout.

Wraps ``cp.async.bulk.tensor.3d`` with the coordinate convention
used by fast.cu: ``{0, row, col/64}`` for the 3D tiled layout.

### Members

#### `load_3d(dst: 'Any', src: 'Any', row: 'Any' = None, col: 'Any' = None, mbar: 'Any' = None, coords: 'tuple[Any, ...] | None' = None, *, pred: "'Reg | NegPred | None'" = None) -> 'None'`

- Kind: `method`

TMA 3D load: ``cp.async.bulk.tensor.3d.shared::cluster.global...``

``col`` is automatically divided by 64 for the 3D coordinate.

#### `load_3d_multicast(dst: 'Any', src: 'Any', row: 'Any' = None, col: 'Any' = None, mbar: 'Any' = None, mask: 'Any' = None, coords: 'tuple[Any, ...] | None' = None, *, issuer: 'int | Reg | NegPred | None' = None, pred: "'Reg | NegPred | None'" = None) -> 'None'`

- Kind: `method`

TMA 3D load with cluster multicast.

#### `store_3d(dst: 'Any', src: 'Any', row: 'Any' = None, col: 'Any' = None, coords: 'tuple[Any, ...] | None' = None, *, pred: "'Reg | NegPred | None'" = None) -> 'None'`

- Kind: `method`

TMA 3D store: ``cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group``

<a id="inst"></a>

## `inst`

- Kind: `namespace`

- Type: `_GenericInst`

Fallback for instructions without a dedicated wrapper.

Usage: ptx.inst.mov.b32(dst, src)
       ptx.inst.add.f32(d, a, b)

<a id="pipe"></a>

## `pipe`

- Kind: `function`

```python
pipe(src: 'Reg') -> '_Pipe'
```

Start an instruction pipeline chain.

Each chained call emits one PTX instruction in order, feeding the
previous result as the first source operand. No instruction
reordering — the PTX is identical to writing the calls separately.

Usage::

    ptx.pipe(r[192]).add.s32(r[193], -8192).and_.b32(r[194], 262016).shr.u32(r[195], 4)

<a id="mov"></a>

## `mov`

- Kind: `function`

```python
mov(dtype: 'PtxType', dst: 'Reg', src: 'Any', *, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit mov.{dtype} dst, src;

<a id="add"></a>

## `add`

- Kind: `function`

```python
add(dtype: 'PtxType', dst: 'Reg', a: 'Any', b: 'Any', *, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit add.{dtype} dst, a, b;

<a id="ret"></a>

## `ret`

- Kind: `function`

```python
ret(*, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit ret;

<a id="bra"></a>

## `bra`

- Kind: `function`

```python
bra(label: 'str', *, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit bra label;

<a id="label"></a>

## `label`

- Kind: `function`

```python
label(name: 'str') -> 'None'
```

Emit a label: label_name:

<a id="addr"></a>

## `addr`

- Kind: `function`

```python
addr(base: 'Any', offset: 'Any' = None) -> 'AddressOperand'
```

Create an address operand: [base], [base+offset].

Accepts anything ``_addr_base_name`` knows about: ``Reg``,
``RegisterOperand``, ``AddressOperand``, ``MbarrierRef``,
``SharedAlloc``, ``SharedSlice``, any ``TensorSpec`` /
``TmaDescriptorHandle`` (duck-typed via ``.name``), or a plain
string.

Usage:
    ptx.addr(rd[0])         → [%rd0]
    ptx.addr(rd[0], 16)     → [%rd0+16]
    ptx.addr("param0")      → [param0]
    ptx.addr(A)             → [A]   # where A is a kernel TensorSpec

<a id="param"></a>

## `param`

- Kind: `function`

```python
param(dtype: 'PtxType', name: 'str', dst: 'Reg | None' = None) -> 'Reg'
```

Load or materialize a kernel parameter and return the destination reg.

For scalar/raw scalar params this emits ``ld.param``.
For raw aggregate params like ``b8.align64.array128 tma_A`` this emits
the existing ``mov``-from-symbol pattern.

<a id="tcgen05"></a>

## `tcgen05`

- Kind: `namespace`

- Type: `_Tcgen05`

ptx.tcgen05.alloc/dealloc/mma/fence/wait/ld/st/cp/shift — Blackwell tensor-core ops.

### Members

#### `BLACKWELL_MASKED_DESC_B128`

- Kind: `attribute`

- Value: `4611756662049538048`

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

#### `alloc(tmem_addr: 'Any', ncols: 'int | Reg', *, cta_group: 'int' = 1, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit tcgen05.alloc.cta_group::N.sync.aligned.shared::cta.b32 [tmem_addr], ncols;

#### `dealloc(taddr: 'Any', ncols: 'int | Reg', *, cta_group: 'int' = 1, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit tcgen05.dealloc.cta_group::N.sync.aligned.b32 taddr, ncols;

#### `relinquish_alloc_permit(*, cta_group: 'int' = 1, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit tcgen05.relinquish_alloc_permit.cta_group::N.sync.aligned;

#### `mma(d_tmem: 'Any', a_desc: 'Any', b_desc: 'Any', idesc: 'Any', *, cta_group: 'int' = 1, kind: 'str' = 'f16', enable_input_d: 'bool | int | None' = True, scale_d: 'Any | None' = None, sparse: 'bool' = False, ashift: 'bool' = False, collector_a: 'str | None' = None, a_is_tmem: 'bool' = False, sparse_metadata: 'Any | None' = None, pred_operand: 'Any | None' = None, scale_c: 'int | None' = None, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit a Blackwell ``tcgen05.mma`` instruction.

The dense F16/BF16/TF32 forms accepted by PTX source use either::

    tcgen05.mma.cta_group::1.kind::f16
        [d_tmem], a_desc, b_desc, idesc, {mask0,mask1,mask2,mask3}, p;

or the runtime-accumulate variant::

    tcgen05.mma.cta_group::1.kind::f16
        [d_tmem], a_desc, b_desc, idesc, {mask0,mask1,mask2,mask3}, p, SCALE_C;

where ``p`` is the runtime accumulate/select-input-D flag,
``SCALE_C`` is a compile-time immediate, and the mask tuple is
typically all zeros for dense CUTLASS/CuTe forms.
Sparse variants insert ``[metadata]`` before ``idesc``.
``scale_d`` here is the accumulate/select-input-D source
(CUTLASS calls it ``scaleC`` / ``accumulate``), not a PTX modifier.

#### `fence_before_thread_sync(*, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit tcgen05.fence::before_thread_sync;

#### `fence_after_thread_sync(*, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit tcgen05.fence::after_thread_sync;

#### `commit(mbar: 'Any', *, cta_group: 'int' = 1, multicast: 'bool' = False, multicast_mask: 'Any | None' = None, space: 'str' = 'cluster', pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit tcgen05.commit.cta_group::N.mbarrier::arrive::one[.multicast::cluster].shared::cluster.b64 [mbar][, mask];

For ``cta_group=2`` the commit must arrive on every participating
CTA's local mbarrier; pass ``multicast=True`` with ``multicast_mask``
set to a u16 bitmask of the peer-CTA ranks to signal.

#### `wait_ld(*, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit tcgen05.wait::ld.sync.aligned;

#### `wait_st(*, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit tcgen05.wait::st.sync.aligned;

#### `ld(dst_regs: 'Any', taddr: 'Any', *, shape: 'str' = '16x128b', count: 'int' = 1, dtype: 'str' = 'b32', pack: 'bool' = False, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit tcgen05.ld.sync.aligned.{shape}.x{count}[.pack::16b] dst, [taddr];

#### `st(taddr: 'Any', src_regs: 'Any', *, shape: 'str' = '16x128b', count: 'int' = 1, dtype: 'str' = 'b32', unpack: 'bool' = False, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit tcgen05.st.sync.aligned.{shape}.x{count}[.unpack::16b] [taddr], src;

#### `cp(taddr: 'Any', src: 'Any', *, cta_group: 'int' = 1, size: 'str' = '128x256b', src_is_addr: 'bool | None' = None, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit tcgen05.cp.cta_group::N.{size} [taddr], [smem];

#### `shift(taddr: 'Any', *, cta_group: 'int' = 1, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit tcgen05.shift.cta_group::N.down [taddr];

#### `make_instr_desc_f16bf16_f32(*, m: 'int' = 128, n: 'int' = 256, ab_dtype: 'str' = 'bf16', a_major: 'str' = 'K', b_major: 'str' = 'K', scale_a: 'int' = 1, scale_b: 'int' = 1, saturate: 'bool' = False, sparse: 'bool' = False, max_shift: 'int' = 0) -> 'int'`

- Kind: `method`

Build the 32-bit Blackwell UMMA instruction descriptor.

Mirrors CUTLASS/CuTe's ``UMMA::make_instr_desc`` for the common
dense F16/BF16 -> F32 path used by the first Blackwell GEMM kernels.
The PTX instruction consumes the upper 32 bits of ``idescE``; this
helper returns that 32-bit descriptor value directly.

#### `descriptor(smem_addr: 'Any', *, byte_offset: 'int' = 0, stride_bytes: 'int', leading_bytes: 'int' = 16, swizzle: 'str' = '128B', version: 'int' = 1, base_offset: 'int' = 0, lbo_mode: 'int' = 0) -> 'Reg'`

- Kind: `method`

Build a Blackwell UMMA shared-memory descriptor.

This mirrors CUTLASS/CuTe's ``UMMA::SmemDescriptor`` encoding.
``stride_bytes`` and ``leading_bytes`` are byte offsets and must be
multiples of 16 because PTX stores them without the low 4 bits.

#### `masked_descriptor(smem_addr: 'Any', *, byte_offset: 'int' = 0, mask: 'int' = 262128, const_bits: 'int' = 4611756662049538048) -> 'Reg'`

- Kind: `method`

Build a Blackwell shared-memory descriptor from a shared address.

This mirrors the CUTLASS SM100 GEMM pattern:

  tmp  = smem_addr + byte_offset
  idx  = (tmp >> 4) & 0x3fff
  desc = cvt.u64.u32(idx) | 0x4000404000010000

Prefer ``ptx.tcgen05.descriptor(...)`` for new code; this helper keeps
the original fixed-B128 GEMM constant for backward compatibility.

<a id="setmaxnreg-inc"></a>

## `setmaxnreg_inc`

- Kind: `function`

```python
setmaxnreg_inc(reg_count: 'int', *, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit setmaxnreg.inc.sync.aligned.u32 N;

<a id="setmaxnreg-dec"></a>

## `setmaxnreg_dec`

- Kind: `function`

```python
setmaxnreg_dec(reg_count: 'int', *, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit setmaxnreg.dec.sync.aligned.u32 N;

<a id="elect-sync"></a>

## `elect_sync`

- Kind: `function`

```python
elect_sync(dst: 'Reg', pred_out: 'Reg', membermask: 'int | Reg') -> 'None'
```

Emit elect.sync dst|pred, membermask;

dst gets the leader lane index, pred_out gets the elected bit.

<a id="cluster"></a>

## `cluster`

- Kind: `namespace`

- Type: `_Cluster`

ptx.cluster.arrive(), ptx.cluster.wait(), ptx.cluster.sync() — barrier.cluster.* helpers.

### Members

#### `arrive(*, aligned: 'bool' = False, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit barrier.cluster.arrive[.aligned];

#### `wait(*, aligned: 'bool' = False, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit barrier.cluster.wait[.aligned];

#### `sync(*, aligned: 'bool' = False, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Emit barrier.cluster.arrive + barrier.cluster.wait.

#### `rank(cta_rank: 'int | Reg') -> 'Reg'`

- Kind: `method`

Return a predicate for ``%cluster_ctarank == cta_rank``.

#### `map_shared_u32(bar_addr: 'Reg', cta_id: 'Reg | int', *, pred: 'Reg | NegPred | None' = None) -> 'Reg'`

- Kind: `method`

Return ``mapa.shared::cluster.u32`` of ``bar_addr`` for ``cta_id``.

#### `arrive_multicast(bar_addr: 'Reg', mask: 'Any', count: "'Reg | int'" = 1, *, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Arrive on the mbarrier at the same SMEM offset on every CTA in a
cluster whose rank bit is set in ``mask`` (u16).

Emits ``mbarrier.arrive.shared::cluster.multicast::cluster.b64 _,
[bar_addr], count, mask`` in a single instruction — one arrive per
target CTA. Replaces the common ``arrive_remote(peer) + arrive()``
pair used for cross-CTA hand-off mbars. The variant isn't in the
generated spec table yet, so we drop to ``raw`` emission.

#### `arrive_remote(bar_addr: 'Reg', cta_id: 'Reg', count: "'Reg | int'" = 1, *, pred: 'Reg | NegPred | None' = None) -> 'None'`

- Kind: `method`

Arrive on a barrier in a remote CTA within the cluster.

Wraps the 3-instruction pattern::

    { mapa.shared::cluster.u32 remAddr, bar_addr, cta_id;
      mbarrier.arrive.shared::cluster.b64 _, [remAddr], count; }

<a id="cvta"></a>

## `cvta`

- Kind: `namespace`

- Type: `_Cvta`

Small helpers for common ``cvta`` conversions.

### Members

#### `param(src: 'Reg', dst: 'Reg | None' = None) -> 'Reg'`

- Kind: `method`

Emit ``cvta.param.u64`` and return the destination register.

#### `to_global(src: 'Reg', dst: 'Reg | None' = None) -> 'Reg'`

- Kind: `method`

Emit ``cvta.to.global.u64`` and return the destination register.

<a id="sub"></a>

## `sub`

- Kind: `function`

```python
sub(dtype: 'PtxType', dst: 'Reg', a: 'Any', b: 'Any', *, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit sub.{dtype} dst, a, b;

<a id="mul"></a>

## `mul`

- Kind: `function`

```python
mul(dtype: 'PtxType', dst: 'Reg', a: 'Any', b: 'Any', *, mode: 'str | None' = None, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit mul[.lo|.hi|.wide].{dtype} dst, a, b;

<a id="mad"></a>

## `mad`

- Kind: `function`

```python
mad(*args, mode: 'str' = 'lo', pred: 'Reg | NegPred | None' = None)
```

Emit ``mad`` in either explicit-dst or expression style.

Explicit-dst form:
    ``ptx.mad(s32, dst, a, b, c)``

Expression form:
    ``dst = ptx.mad(a, b, c)``

<a id="shl"></a>

## `shl`

- Kind: `function`

```python
shl(dtype: 'PtxType', dst: 'Reg', a: 'Any', b: 'Any', *, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit shl.{dtype} dst, a, b;

<a id="shr"></a>

## `shr`

- Kind: `function`

```python
shr(dtype: 'PtxType', dst: 'Reg', a: 'Any', b: 'Any', *, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit shr.{dtype} dst, a, b;

<a id="setp"></a>

## `setp`

- Kind: `function`

```python
setp(cmp_op: 'str', dtype: 'PtxType', pred_out: 'Reg', a: 'Any', b: 'Any', *, pred_negate: 'Reg | NegPred | None' = None) -> 'None'
```

Emit setp.{cmp_op}.{dtype} pred_out, a, b;

<a id="cvt"></a>

## `cvt`

- Kind: `function`

```python
cvt(dst_type: 'PtxType', src_type: 'PtxType', dst: 'Reg', src: 'Any', *, rounding: 'str | None' = None, ftz: 'bool' = False, sat: 'bool' = False, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit cvt[.rnd][.ftz][.sat].{dst_type}.{src_type} dst, src;

<a id="ld"></a>

## `ld`

- Kind: `function`

```python
ld(dtype: 'PtxType', dst: 'Reg', addr: 'Any', *, space: 'str' = 'global', cache: 'str | None' = None, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit ld.{space}[.{cache}].{dtype} dst, [addr];

<a id="st"></a>

## `st`

- Kind: `function`

```python
st(dtype: 'PtxType', addr: 'Any', src: 'Any', *, space: 'str' = 'global', cache: 'str | None' = None, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit st.{space}[.{cache}].{dtype} [addr], src;

<a id="and-"></a>

## `and_`

- Kind: `function`

```python
and_(dtype: 'PtxType', dst: 'Reg', a: 'Any', b: 'Any', *, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit and.{dtype} dst, a, b;

<a id="or-"></a>

## `or_`

- Kind: `function`

```python
or_(dtype: 'PtxType', dst: 'Reg', a: 'Any', b: 'Any', *, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit or.{dtype} dst, a, b;

<a id="xor-"></a>

## `xor_`

- Kind: `function`

```python
xor_(dtype: 'PtxType', dst: 'Reg', a: 'Any', b: 'Any', *, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit xor.{dtype} dst, a, b;

<a id="not-"></a>

## `not_`

- Kind: `function`

```python
not_(dtype: 'PtxType', dst: 'Reg', src: 'Any', *, pred: 'Reg | NegPred | None' = None) -> 'None'
```

Emit not.{dtype} dst, src;

<a id="typed-wrapper-codegen"></a>

## `TYPED_WRAPPER_CODEGEN`

- Kind: `namespace`

- Type: `dict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### Members

#### `get`

- Kind: `attribute`

- Value: `<built-in method get of dict object at 0x7ff0598239c0>`

Return the value for key if key is in the dictionary, else default.

#### `setdefault`

- Kind: `attribute`

- Value: `<built-in method setdefault of dict object at 0x7ff0598239c0>`

Insert key with a value of default if key is not in the dictionary.

Return the value for key if key is in the dictionary, else default.

#### `pop`

- Kind: `attribute`

- Value: `<built-in method pop of dict object at 0x7ff0598239c0>`

D.pop(k[,d]) -> v, remove specified key and return the corresponding value.

If the key is not found, return the default if given; otherwise,
raise a KeyError.

#### `popitem`

- Kind: `attribute`

- Value: `<built-in method popitem of dict object at 0x7ff0598239c0>`

Remove and return a (key, value) pair as a 2-tuple.

Pairs are returned in LIFO (last-in, first-out) order.
Raises KeyError if the dict is empty.

#### `keys`

- Kind: `attribute`

- Value: `<built-in method keys of dict object at 0x7ff0598239c0>`

D.keys() -> a set-like object providing a view on D's keys

#### `items`

- Kind: `attribute`

- Value: `<built-in method items of dict object at 0x7ff0598239c0>`

D.items() -> a set-like object providing a view on D's items

#### `values`

- Kind: `attribute`

- Value: `<built-in method values of dict object at 0x7ff0598239c0>`

D.values() -> an object providing a view on D's values

#### `update`

- Kind: `attribute`

- Value: `<built-in method update of dict object at 0x7ff0598239c0>`

D.update([E, ]**F) -> None.  Update D from mapping/iterable E and F.
If E is present and has a .keys() method, then does:  for k in E.keys(): D[k] = E[k]
If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
In either case, this is followed by: for k in F:  D[k] = F[k]

#### `fromkeys`

- Kind: `attribute`

- Value: `<built-in method fromkeys of type object at 0x7ff05ab31f60>`

Create a new dictionary with keys from iterable and values set to value.

#### `clear`

- Kind: `attribute`

- Value: `<built-in method clear of dict object at 0x7ff0598239c0>`

D.clear() -> None.  Remove all items from D.

#### `copy`

- Kind: `attribute`

- Value: `<built-in method copy of dict object at 0x7ff0598239c0>`

D.copy() -> a shallow copy of D
