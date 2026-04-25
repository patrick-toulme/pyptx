# PTX Namespace

`pyptx.ptx` is the center of the DSL.

If `reg` gives you registers and `smem` gives you shared-memory objects, `ptx` is where the kernel actually becomes PTX.

The namespace covers a few different layers at once:

- direct instruction wrappers
- structured control flow
- Hopper-specific helpers
- convenience wrappers for very common idioms

## Control Flow

Structured control flow helpers let you write branchy PTX without dropping into labels immediately:

```python
tid = reg.from_(ptx.special.tid.x(), u32)

with ptx.if_(tid == 0):
    ptx.inst.mov.u32(tid, 1)

with ptx.loop("k_loop", pred=(tid < 4)):
    tid += 1
```

Important point: these are still PTX control-flow emitters. They are not high-level dataflow constructs.

## Special Registers And Addresses

Examples:

```python
tid = reg.from_(ptx.special.tid.x(), u32)
cta_rank = reg.from_(ptx.sreg("%cluster_ctarank"), u32)
ptr = ptx.param(u64, "ptr")
global_ptr = ptx.cvta.to_global(ptr)
```

This is the common boundary between symbolic kernel parameters, special registers, and ordinary PTX registers.

## Arithmetic Helpers

At the lowest level you can always write instruction wrappers:

```python
ptx.inst.add.s32(dst, a, b)
ptx.inst.mad.lo.s32(dst, a, b, c)
```

For common cases, the DSL also supports expression-style forms:

```python
x = a + b
y = x - 1
z = ptx.mad(a, b, c)
```

The rule of thumb is:

- use expression sugar when the operation is obvious
- drop to `ptx.inst.*` when you need exact modifier control

## Shared Memory And Barriers

`ptx` also holds the barrier and memory-operation wrappers that are too PTX-specific to live in `smem` alone:

```python
full = ptx.mbarrier.array(smem_base, SMEM_FULL, 3)
pipe = ptx.pipeline(3)

stage, phase = pipe.advance()
full.at(stage).wait(phase)
```

This is where a lot of the “handwritten kernel ergonomics” work has gone: the PTX is still explicit, but repetitive patterns are compressed.

## Hopper Features

This is also where the Hopper-specific surface lives:

```python
desc = ptx.wgmma.masked_descriptor(base, byte_offset=-8192, mask=262016)
ptx.tma.load_3d(dst=dst, src=tma_A, coords=(0, row, col), mbar=bar)
ptx.tma.load_3d_multicast(..., issuer=ptx.cluster.rank(0))
ptx.stmatrix_x4_trans_f32_bf16(...)
```

These helpers matter because Hopper kernels are where the PTX is hardest to read and the most bug-prone to write by hand.

## How To Read The API Page

The generated API reference for `pyptx.ptx` is comprehensive, but it is intentionally flat because it follows the source.

When you browse `pyptx.ptx`, think in these groups:

1. control flow: `if_`, `loop`, `kloop`, `scope`
2. addressing and params: `addr`, `param`, `cvta`
3. barriers and pipeline: `mbarrier`, `named_barrier`, `pipeline`
4. matrix/tensor ops: `wgmma`, `tma`, `stmatrix`, `ldmatrix`
5. direct wrappers: `mov`, `add`, `mad`, `cvt`, `ld`, `st`

That grouping is the real mental model even if the generated API page is alphabetical or source-ordered.
