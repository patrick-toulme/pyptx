# SwiGLU

This page walks through `examples/hopper/swiglu.py` — a fused SwiGLU
activation kernel that reaches **2.8 TB/s at M=2048, F=8192 f32** on
H100 (**94% of HBM3 peak**), **1.6×** faster than
`F.silu(gate) * up` eager.

It computes:

```text
h[i, j] = silu(gate[i, j]) * up[i, j]
       = (gate[i, j] * sigmoid(gate[i, j])) * up[i, j]
```

per element. Three tensors in, one out. Bandwidth-bound — the
interesting work is (a) saturating HBM3 with v4 loads, (b) computing
`sigmoid` without a `div.f32` on the hot path.

Read [RMS Norm](rms-norm.md) first if you haven't. SwiGLU uses the
same v4 memory pattern; the new surface here is the math path for
`silu`.

## What Makes SwiGLU Different From A Plain Map

SwiGLU is a **three-input, one-output** elementwise kernel. Compared to
a plain unary map, that changes two things:

- Three DRAM streams to interleave. Two loads (gate, up), one store (out).
- No reductions, no barriers. Each thread is independent, so the kernel
  can afford to have many items per thread to hide latency.

The memory pattern is the same as RMS Norm pass 2 but doubled on the
input side.

## Step 1: Pick The Block Size For v4

```python
for block in (1024, 512, 256, 128, 64, 32):
    if f % (block * 4) == 0 and block >= 128:
        ipt = f // block
        if best is None or ipt > best[1]:
            best = (block, ipt)
```

Same logic as RMS Norm: prefer the config with the **most items per
thread** such that every load can still be a `ld.global.v4.f32`. For
`F=8192`, block=1024 gives `ipt=8` which is 2 v4 loads per thread per
row — good memory-level parallelism.

## Step 2: Launch Config

```python
@kernel(
    in_specs=(Tile(M, F, f32), Tile(M, F, f32)),
    out_specs=(Tile(M, F, f32),),
    grid=(M // rows_per_cta, 1, 1),
    block=(block, 1, 1),
    arch=arch,
)
def fused_silu_mul(gate, up, out):
```

One CTA per row by default. `rows_per_cta > 1` lets a single CTA walk
multiple rows, which is useful when `M` is small enough that launch
overhead dominates (e.g. `M=4`).

## Step 3: Prologue — Three Pointers, One Row Offset

```python
pg, pu, po = ptx.global_ptrs(gate, up, out)

row_base = reg.scalar(u32)
ptx.inst.mov.u32(row_base, ptx.special.ctaid.x())
if rows_per_cta > 1:
    ptx.inst.mul.lo.u32(row_base, row_base, rows_per_cta)
row_byte_off = row_base * (F * 4)
pg += row_byte_off
pu += row_byte_off
po += row_byte_off
```

All three input tensors share the same row layout, so one
`row_byte_off` computed once advances all three pointers. The
`rows_per_cta` branch is resolved at trace time — it's Python `if`,
not `ptx.if_` — so the compiled PTX contains no runtime check.

## Step 4: The Fast `silu` Path

Here's the arithmetic core, one v4 chunk:

```python
g_vals = [reg.scalar(f32) for _ in range(4)]
ptx.inst.ld.global_.v4.f32(g_vals, ptx.addr(pg + off))
u_vals = [reg.scalar(f32) for _ in range(4)]
ptx.inst.ld.global_.v4.f32(u_vals, ptx.addr(pu + off))
out_vals = []
for sub in range(4):
    neg_g = reg.scalar(f32)
    ptx.inst.mul.f32(neg_g, g_vals[sub], neg_log2e)  # -g * log2(e)
    exp_neg = reg.scalar(f32)
    ptx.inst.ex2.approx.f32(exp_neg, neg_g)          # exp(-g)
    denom = reg.scalar(f32)
    ptx.inst.add.f32(denom, one, exp_neg)            # 1 + exp(-g)
    sigm = reg.scalar(f32)
    ptx.inst.rcp.approx.f32(sigm, denom)             # 1 / (1 + exp(-g))
    silu_g = reg.scalar(f32)
    ptx.inst.mul.f32(silu_g, g_vals[sub], sigm)      # g * sigmoid(g)
    out_val = reg.scalar(f32)
    ptx.inst.mul.f32(out_val, silu_g, u_vals[sub])   # silu(g) * u
    out_vals.append(out_val)
ptx.inst.st.global_.v4.f32(ptx.addr(po + off), out_vals)
```

Four places where we're being specific about what we want:

1. **`ex2.approx.f32` instead of `ex.approx.f32`.** Hopper has
   `ex2` (base-2 exponential) as a hardware intrinsic; `ex` (natural)
   would need a multiply-by-`log2(e)` anyway. Fold that constant into
   the input — pre-compute `neg_log2e = -1.4426950408889634` as a
   compile-time register, multiply once.
2. **`rcp.approx.f32` instead of `div.f32`.** `div.f32` is slow and
   typically implemented as `rcp` + `mul` + Newton iteration. We
   don't need the extra precision — approx is good for activations —
   so skip the Newton step.
3. **The whole `silu` is 5 instructions:** mul, ex2, add, rcp, mul.
   Plus one more mul to combine with `up`. That's 6 fp32 instructions
   per element. The v4 load/store pair is 2 instructions per 4
   elements = 0.5 memory instructions per element. Bandwidth bound:
   ~0.083 memory instructions per fp32 op.
4. **Register list arguments.** `g_vals` is a Python list of 4 `Reg`
   objects, passed directly to `ld.global_.v4.f32`. The DSL knows how
   to spell PTX's `{r0, r1, r2, r3}` register-vector syntax.

## Step 5: Non-v4 Fallback

When `items_per_thread` isn't divisible by 4, the kernel has a scalar
fallback that does exactly the same arithmetic per element:

```python
for i in range(items_per_thread):
    idx = reg.scalar(u32)
    ptx.inst.add.u32(idx, tid, i * block)
    off = idx * 4
    # ...single-element ld.global.f32, scalar silu, scalar st.global.f32
```

This path is a ~4× bandwidth regression because each transaction
moves 4 bytes instead of 16. It exists only for shapes that don't
admit v4 — `F` not divisible by `block * 4`. In practice the block
picker keeps this branch cold.

## Step 6: Multi-Row Traversal

```python
for r in range(rows_per_cta):
    process_row(r * F * 4 if rows_per_cta > 1 else 0)
```

`r * F * 4` is the byte offset added on top of each pointer for the
`r`-th row this CTA is handling. The Python `for` unrolls at trace
time — if `rows_per_cta=4`, the emitted kernel has 4 copies of the v4
load/silu/store sequence in a row, no runtime loop counter.

This is a deliberate knob for small `M`. For `M=4` (or similar
debug-sized batches), you don't want to launch 4 CTAs — that's 4×
launch overhead. Setting `rows_per_cta=4` collapses it to 1 CTA doing
all 4 rows, with the added benefit that all 4 row-prologue adds (row
offsets) are bulk-computed together.

## Why This Kernel Matters For The DSL

SwiGLU is the second-simplest kernel in the repo for a reason:

- It's a **no-reduction, no-sync** elementwise path — the cleanest
  possible DSL stress test for memory + math.
- It uses **three different HBM streams** concurrently. If the DSL
  serialized the loads (e.g. by inserting hidden syncs), you'd see
  it in the TB/s number — you don't, so it doesn't.
- `ex2.approx` + `rcp.approx` + `fma` is the standard toolkit for
  building transcendentals at approximate precision. Making all three
  first-class means you can spell `silu`, `gelu`, `tanh`, `softplus`
  without ever reaching for inline asm.
- `ptx.global_ptrs(gate, up, out)` scales linearly to more tensors —
  try writing four or five pointer prologues by hand and the DSL's
  value shows up immediately.

At 2.8 TB/s on H100, this kernel is within 2 percentage points of
memory-bandwidth peak. Everything the DSL does on top of raw PTX costs
effectively zero runtime — that's the claim this kernel validates.

## What To Read Next

- [Layer Norm](../examples/hopper/layer_norm.md) — same shape, same
  v4 pattern, but with a mean pass and a variance pass. Shows the
  two-moment version of RMS Norm.
- [Hopper GEMM](handwritten-gemm.md) — tensor-core hot path instead
  of memory hot path.
- [Grouped GEMM](grouped-gemm.md) — batched multiply, warp-specialized
  K loop.
