# Grouped GEMM

This page walks through `examples/hopper/grouped_gemm.py` — a
uniform-shape batched GEMM that reaches **104 TFLOPS at G=8, M=512,
N=64, K=512** on H100. This is the MoE-scale shape that
`torch.nn.functional.grouped_mm` and Triton's "group GEMM" both target
when every expert has equal capacity.

It computes:

```text
C[g] = A[g] @ B[g]   for g in range(G)
```

with per-group matrices in `bf16`, accumulation in `f32`. Inputs are
presented as flattened 2D views — `A` is `(G*M, K)`, `B` is `(G*K, N)`
— so the kernel can reuse 2D TMA descriptors instead of needing 3D ones.

Read [Hopper GEMM](handwritten-gemm.md) first. The single-problem K-loop
is identical; this guide only covers what's **added** to support G
problems.

## What Changes From Single-Problem GEMM

Three things, in order of how much they affect the body of the kernel:

1. **Grid gains a Z dimension** for the group index:
   ```python
   grid = (N // tile_n, M // BM, G)
   ```
   `ctaid.z` picks which problem this CTA is computing.

2. **Per-CTA pointer offsets** use `ctaid.z * M` for A/C row bases,
   `ctaid.z * K` for B's row base. No change to the TMA descriptor
   itself.

3. **Multi-k WGMMA per K-iter** (`tile_k = 64`): each outer loop
   iteration issues `tile_k // 16 = 4` WGMMAs over the same loaded
   tiles, to amortize TMA + mbarrier overhead.

Everything else — WGMMA shape, epilogue, register fragment layout —
is identical to the single-problem case.

## Step 1: Pick `tile_k` For The Shape

```python
if tile_k is None:
    tile_k = 64 if K % 64 == 0 and K >= 64 else 16
```

WGMMA's `k=16` is fixed, but we can load a bigger chunk of K once and
issue multiple WGMMAs against it:

- `tile_k = 16` → one WGMMA per TMA load (simple, high overhead)
- `tile_k = 64` → four WGMMAs per TMA load (better overhead amortization,
  but requires SMEM space for the wider tile)

For MoE expert shapes where K is typically 512–2048, `tile_k=64` is a
consistent win. The multi-k WGMMA inner loop does the four
tensor-core issues against SMEM offsets of the already-loaded tile.

## Step 2: Pick `tile_n`

```python
for tn in (64, 32, 16, 8):
    if N % tn == 0:
        tile_n = tn
        break
```

MoE output columns are typically small (`N=8`, `N=16`, `N=64`) because
each expert handles a thin slice. The kernel picks the largest `tile_n`
that cleanly divides `N`. WGMMA supports `m64n{8,16,32,64,128,256}k16`
— all of these map to real hardware instructions.

`acc_count = tile_n // 2` because each thread's fragment of the
m64nTILE_N output tile is `TILE_N / 2` `f32` values. For `tile_n=8`
that's 4 registers; for `tile_n=64` that's 32.

## Step 3: Per-CTA Row/Column Bases (Now With Group)

```python
group = reg.scalar(u32)
ptx.inst.mov.u32(group, ptx.special.ctaid.z())

m_row_base = reg.scalar(u32)
ptx.inst.mov.u32(m_row_base, ptx.special.ctaid.y())
ptx.inst.shl.b32(m_row_base, m_row_base, 6)           # * BM=64
group_m_off = reg.scalar(u32)
ptx.inst.mul.lo.u32(group_m_off, group, M)            # group * M
ptx.inst.add.u32(m_row_base, m_row_base, group_m_off) # -> row in A

k_row_base = reg.scalar(u32)
ptx.inst.mul.lo.u32(k_row_base, group, K)             # group * K → row in B

n_col_base = reg.scalar(u32)
ptx.inst.mov.u32(n_col_base, ptx.special.ctaid.x())
shift = {64: 6, 32: 5, 16: 4, 8: 3}[tile_n]
ptx.inst.shl.b32(n_col_base, n_col_base, shift)       # * tile_n
```

Three coordinate systems at once:

- **A** is `(G*M, K)` — row is `group * M + ctaid.y * BM`, col walks K.
- **B** is `(G*K, N)` — row is `group * K + k_off` (changes per K iter),
  col is `ctaid.x * tile_n`.
- **C** is `(G*M, N)` — same row as A's, same col as B's.

The key trick: **the TMA descriptor is still 2D**. We pass it the
flattened 2D shape, and per-CTA row offsets do the group dispatch.
No 3D TMA, no descriptor-per-group, no runtime descriptor patching.

## Step 4: The K Loop

```python
k_off = reg.scalar(u32, init=0)
keep_going = reg.scalar(pred)
ptx.inst.setp.lt.u32(keep_going, k_off, K)
with ptx.loop("k_loop", pred=keep_going):
    b_row = reg.scalar(u32)
    ptx.inst.add.u32(b_row, k_row_base, k_off)

    with ptx.if_(tid == 0):
        ptx.mbarrier.arrive_expect_tx(
            bar[0], BM * tile_k * 2 + tile_k * tile_n * 2,
        )
        ptx.cp.async_.bulk.tensor_2d(
            dst=sA[0], src=A.tma_desc(),
            coord=(k_off, m_row_base), mbar=bar[0],
        )
        ptx.cp.async_.bulk.tensor_2d(
            dst=sB[0], src=B.tma_desc(),
            coord=(n_col_base, b_row), mbar=bar[0],
        )
    ptx.bar.sync(0)
    ptx.mbarrier.wait(bar[0], phase)
    phase ^= 1
```

This is `ptx.loop(...)` with a predicate, which is a runtime loop — not
the Python `for` range unrolling used in `rms_norm.py`. K can be much
bigger than a reasonable unrolled body, so the loop is dynamic with
`phase` toggling between `0` and `1` for the mbarrier's two-phase
wait.

A subtle point: the **TMA tile expectation** accounts for both A and
B: `BM * tile_k * 2 + tile_k * tile_n * 2` bytes. Both TMA transfers
use the same barrier, so `arrive_expect_tx` gets the combined byte
count. The barrier becomes ready only after both async copies finish.

## Step 5: Multi-k WGMMA

```python
ptx.wgmma.fence()
for kk in range(wgmma_k_iters):
    a_off = kk * 32
    b_off = kk * 16 * b_row_bytes
    if kk == 0:
        scale = (k_off != 0)
    else:
        scale = True
    ptx.wgmma.mma_async(
        shape=(64, tile_n, 16),
        dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
        d=acc, a=sA, b=sB,
        scale_d=scale,
        trans_a=0, trans_b=1,
        a_k_offset=a_off, b_k_offset=b_off,
    )
ptx.wgmma.commit_group()
ptx.wgmma.wait_group(0)
```

For `tile_k=64`, this unrolls to four WGMMA issues per K-loop
iteration, each pointing at a different 16-wide slice of the already-
loaded `sA`/`sB` tile:

- `a_off = kk * 32` — 32 bytes between slices in K-major A layout
  (16 bf16 elements = 32 bytes).
- `b_off = kk * 16 * b_row_bytes` — 16 rows per k-slice in MN-major B
  layout, times one row's byte width.

The `scale_d` logic is the same as the single-problem case with one
addition: on kk=0 we scale iff we've done any WGMMA before (k_off != 0),
on kk>0 we always scale — the previous kk already wrote the accumulator
this iteration.

## Step 6: The Epilogue

Identical to the single-problem case structurally, just adjusted for
`tile_n`:

```python
def _grouped_epilogue(C, acc, row_offset, col_offset, N, tile_n):
    # wid = tid / 32, lane = tid % 32
    # frag_row = (wid * 16) + (lane / 4)
    # frag_col = (lane % 4) * 2
    # ...shifted into global coords, then v2 stores per fragment row

    for g in range(tile_n // 8):
        col = frag_col + g * 8
        off_a = (frag_row * N + col) * 4
        ptx.inst.st.global_.v2.f32(ptx.addr(pc + off_a), [acc[g * 4], acc[g * 4 + 1]])
        off_b = (row_b * N + col) * 4
        ptx.inst.st.global_.v2.f32(ptx.addr(pc + off_b), [acc[g * 4 + 2], acc[g * 4 + 3]])
```

Each lane owns a `2x8` strip of the output and scatters it with two
`st.global.v2.f32` per `tile_n // 8` chunk. The `group` offset is
already baked into `m_row_base` (and thus `row_offset`), so the
epilogue itself is group-agnostic.

## When Grouped GEMM Is The Right Call

- **MoE inference** where all experts have equal capacity. Every
  expert processes the same-shape problem; batched dispatch is one
  kernel launch.
- **Attention-head-parallel batched matmul** where every head runs an
  identical shape.
- **Any time you'd otherwise loop `G` kernel launches** for `G`
  identical shapes — this collapses them into one launch with a
  3D grid.

Non-uniform grouped GEMM (each problem has its own `(M_g, N_g, K_g)`)
is a straightforward extension: replace the 3D grid with a persistent
1D grid walking a tile schedule stored in global memory. The per-tile
body is the same.

## Why This Kernel Matters For The DSL

Grouped GEMM is almost entirely the single-problem GEMM with a
`ctaid.z` added to the pointer math. The fact that this works — that
you can extend a kernel without rewriting the WGMMA / TMA / mbarrier
layer — is the DSL property being tested:

- `ptx.cp.async_.bulk.tensor_2d` takes arbitrary runtime coords, so
  per-group row offsets compose.
- `ptx.loop(..., pred=...)` is a runtime loop, lets K grow beyond what
  you'd unroll.
- `ptx.wgmma.mma_async(..., a_k_offset=..., b_k_offset=...)` lets one
  loaded tile feed multiple WGMMA issues without reloading.
- The 3D grid is just `grid=(..., ..., G)` in the decorator. No new
  API.

104 TFLOPS at MoE scale is the kind of perf that usually requires
CUTLASS templates. Here it's 288 lines of Python.

## What To Read Next

- [Hopper GEMM](handwritten-gemm.md) — the single-problem kernel this
  one is built on top of
- [Blackwell GEMM](blackwell-gemm.md) — same GEMM shape on Blackwell
  via `tcgen05.mma` + TMEM
- `examples/hopper/grouped_gemm.py` — the full source, including JAX
  and PyTorch test harnesses
