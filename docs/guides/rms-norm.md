# RMS Norm

This page is a walkthrough of `examples/hopper/rms_norm.py` — the
simplest kernel in the repo and the right first read after
[First Kernel](first-kernel.md).

It computes:

```text
Y[b, i] = X[b, i] * W[i] / sqrt(mean(X[b, :]^2) + eps)
```

one row per CTA, `B` CTAs total. Reaches **2.6 TB/s at B=2048, N=8192**
on H100 (88% of HBM3 peak), **3.9×** faster than PyTorch eager.

This kernel is bandwidth-bound, so the interesting work is in the memory
access pattern — not in a tensor-core intrinsic. It's a good stress
test of whether the DSL stays honest when the hot path is `ld.global.v4`
instead of `wgmma.mma_async`.

## What The Kernel Computes Per Row

1. Load the `N`-wide row of `X`, strided across `block` threads.
2. Each thread accumulates its slice's sum-of-squares into a scalar.
3. Warp-level butterfly reduction turns those into one partial per warp.
4. A final warp reduces across partials and broadcasts via SMEM.
5. Compute `rstd = 1/sqrt(mean + eps)`.
6. Reload the per-thread slice, multiply by `rstd * W[i]`, write `Y`.

Steps 1 and 6 are the two bandwidth passes. Everything else is
arithmetic and synchronization.

## Step 1: Pick The Block Size From `N`

`_pick_block(n)` prefers blocks that leave at least 4 `f32` items per
thread **and** where that item count is divisible by 4:

```python
for block in (512, 256, 128, 64, 32):
    ipt = n // block
    if n % block == 0 and ipt >= 4 and ipt % 4 == 0 and block >= 128:
        # ...pick the biggest ipt
```

The "divisible by 4" constraint exists so every memory transaction can
be a `ld.global.v4.f32` — one 16-byte load that feeds four accumulator
FMAs. That's the memory-level parallelism knob: more outstanding v4
loads per thread → more DRAM requests in flight → better HBM
utilization.

## Step 2: Launch Config

```python
@kernel(
    in_specs=(Tile(B, N, f32), Tile(N, f32)),
    out_specs=(Tile(B, N, f32),),
    grid=(B, 1, 1),
    block=(block, 1, 1),
    arch=arch,
)
def rms_norm(X, W, Y):
```

One CTA per batch row. Inside the CTA, `block` threads cooperate on
that row. No cross-CTA communication.

## Step 3: Prologue — Pointers, Row Offset, Per-Warp Bookkeeping

```python
partials = smem.alloc(f32, (num_warps, 1))
stats = smem.alloc(f32, (1, 1))

px, pw, py = ptx.global_ptrs(X, W, Y)

tid = reg.scalar(u32); ptx.inst.mov.u32(tid, ptx.special.tid.x())
row = reg.scalar(u32); ptx.inst.mov.u32(row, ptx.special.ctaid.x())
row_byte_off = row * (N * 4)
px += row_byte_off
py += row_byte_off
lane = tid & (WARP_SIZE - 1)
warp_id = tid >> 5
```

Three things to notice:

- **`ptx.global_ptrs(X, W, Y)`** is one call that unpacks three
  parameter-pointer prologues. Without it you'd write three near-identical
  `ld.param.u64` sequences by hand.
- **`row * (N * 4)`** uses Python operator overloading on `Reg`. The
  DSL emits `mul.wide.u32` under the covers — exactly what you'd
  write by hand.
- `W` has no row offset — it's shared across all rows of the batch.

## Step 4: Pass 1 — Load And Square

With `use_v4` true (the common case), each thread executes
`v4_iters = items_per_thread // 4` vectorized loads:

```python
sum_sq = reg.scalar(f32, init=0.0)
x_vals = reg.array(f32, items_per_thread)

elem_base = tid << 2  # every thread's lane-0 element
for j in range(v4_iters):
    idx = elem_base if j == 0 else elem_base + (j * block * 4)
    ptr = px + idx * 4
    ptx.inst.ld.global_.v4.f32(
        [x_vals[j*4], x_vals[j*4+1], x_vals[j*4+2], x_vals[j*4+3]],
        ptx.addr(ptr),
    )
    for sub in range(4):
        ptx.inst.fma.rn.f32(sum_sq, x_vals[j*4+sub], x_vals[j*4+sub], sum_sq)
```

Two design choices worth reading twice:

1. **`x_vals = reg.array(f32, items_per_thread)` is preserved across
   passes.** The per-thread slice is held in registers through the
   whole kernel, so pass 2 doesn't need to re-load from global. That
   turns a 2-pass-over-HBM algorithm into a 1.5-pass — the weights
   load again, the inputs don't.
2. **`fma.rn.f32(sum_sq, x, x, sum_sq)`** is one PTX instruction doing
   `sum_sq = x*x + sum_sq` rn-rounded. Writing `sum_sq = sum_sq + x*x`
   in the DSL would emit two instructions (mul + add) — not the same.
   When you care about the instruction count, reach for `inst.fma.rn`.

## Step 5: Warp Reduce, Then Block Reduce

```python
ptx.warp.reduce_sum(sum_sq)

with ptx.if_(lane == 0):
    partials[warp_id, 0] = sum_sq
ptx.bar.sync(0)

with ptx.if_(tid == 0):
    block_sum = reg.scalar(f32, init=0.0)
    for i in range(num_warps):
        ptx.inst.add.f32(block_sum, block_sum, partials[i, 0])
    stats[0, 0] = block_sum
ptx.bar.sync(0)

ptx.inst.mov.f32(sum_sq, stats[0, 0])
```

Three hops:

1. **`warp.reduce_sum`** is a butterfly `shfl.bfly.sync` reduction —
   no hand-rolled helper, but the DSL still emits the same five PTX
   instructions you'd write manually.
2. **Per-warp partials to SMEM.** Lane 0 of each warp writes its
   sum. `bar.sync(0)` ensures all warps are past this point before the
   next stage reads.
3. **Thread 0 collapses partials.** Final result parked in
   `stats[0,0]` for broadcast. The second `bar.sync(0)` makes the
   value visible to the whole CTA. Every thread re-reads it.

This is intentionally the two-phase pattern. For small `num_warps`,
the second phase is a trivial linear scan — not worth another warp
butterfly.

## Step 6: Compute `rstd` Once Per Thread

```python
mean_sq = reg.scalar(f32)
inv_n = reg.scalar(f32, init=1.0 / N)
ptx.inst.mul.f32(mean_sq, sum_sq, inv_n)
eps_reg = reg.scalar(f32, init=eps)
ptx.inst.add.f32(mean_sq, mean_sq, eps_reg)
rstd = reg.scalar(f32)
ptx.inst.rsqrt.approx.f32(rstd, mean_sq)
```

`rsqrt.approx.f32` is the single-instruction reciprocal square root.
`N` is baked in at trace time as `1.0 / N` so the kernel doesn't do a
division on the hot path.

## Step 7: Pass 2 — Load `W`, Multiply, Store

```python
for j in range(v4_iters):
    idx = elem_base if j == 0 else elem_base + (j * block * 4)
    off = idx * 4

    w_vals = [reg.scalar(f32) for _ in range(4)]
    ptx.inst.ld.global_.v4.f32(w_vals, ptx.addr(pw + off))

    y_vals = []
    for sub in range(4):
        y_val = reg.scalar(f32)
        ptx.inst.mul.f32(y_val, x_vals[j*4+sub], rstd)
        ptx.inst.mul.f32(y_val, y_val, w_vals[sub])
        y_vals.append(y_val)

    ptx.inst.st.global_.v4.f32(ptx.addr(py + off), y_vals)
```

- `x_vals` is still in registers from pass 1 — no reload.
- `W` is read fresh (it's `N`-wide, not `B*N`, so it's tiny and cached).
- The store is also v4. Every DRAM transaction in this kernel is 16 B.

## Why This Kernel Matters For The DSL

RMS norm is a good DSL test because nothing here benefits from a
tensor-core intrinsic:

- `ld.global.v4.f32` and `st.global.v4.f32` with register lists must
  map cleanly to the PTX form.
- `fma.rn.f32` has to be spellable as one instruction, not lowered
  from Python `*` and `+`.
- `rsqrt.approx.f32`, `reduce_sum`, and `bar.sync(0)` need to be
  first-class, not escape hatches.
- `smem.alloc(f32, (num_warps, 1))` must work for tiny staging, not
  just WGMMA tiles.

If any of those fell back to a lower-level helper, this kernel would
read like a hand-rolled assembler instead of a DSL. That it doesn't is
the point.

## What To Read Next

- [SwiGLU](swiglu.md) — same memory pattern plus a fast-path
  `silu = x * sigmoid(x)` built from `ex2.approx` + `rcp.approx`
- [Hopper GEMM](handwritten-gemm.md) — when the hot path is
  `wgmma.mma_async` instead of `ld.global.v4`
- [PTX Namespace](ptx-namespace.md) — every helper used above, as
  a reference page
