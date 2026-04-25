# Tiles, Layouts, and TMA

Every kernel decorator opens with something like:

```python
@kernel(
    in_specs=(
        Tile.wgmma_a(M, K, bf16, tile_m=BM, tile_k=tile_k),
        Tile.wgmma_b(K, N, bf16, tile_k=tile_k, tile_n=BN),
    ),
    out_specs=(Tile(M, N, f32, Layout.ROW),),
    grid=(N // BN, M // BM, 1),
    block=(128, 1, 1),
    arch="sm_90a",
)
```

Every argument is load-bearing. This page is a reference for what each
piece controls, when to pick which, and how the `Tile` / `Layout` /
`tma_box` triple composes into a working TMA descriptor + matching
SMEM allocation. Read this before writing a new kernel with different
shapes from the examples.

## The Three Roles A Tile Plays

A `Tile` in an `@kernel` spec does three things at once:

1. **Shape contract.** Tells the framework what shape of JAX/PyTorch
   tensor the kernel accepts (or outputs). Symbolic dims like
   `Tile("M", "K", bf16)` are bound at call time from the actual
   tensor shape.
2. **dtype.** The element type — bf16, f32, b32, etc. Must match the
   runtime array dtype exactly.
3. **TMA descriptor specification.** For input tiles, the `Layout`
   plus `tma_box` determine the TMA descriptor the framework builds
   and hands to the kernel at launch time (via `tensor.tma_desc()`
   inside the kernel body).

The output spec only uses (1) and (2) — outputs are written via plain
`st.global.*`, not TMA (the maintained examples use TMA for inputs
only).

## Layout: What Each Value Means

```python
class Layout(Enum):
    ROW = "row"                 # row-major (C order), no swizzle
    COL = "col"                 # column-major (Fortran order)
    TMA_128B = "tma_128b"       # TMA 128-byte swizzle
    TMA_64B  = "tma_64b"        # TMA 64-byte swizzle
    TMA_32B  = "tma_32b"        # TMA 32-byte swizzle
    INTERLEAVED = "interleaved" # CUTLASS interleaved
```

The three values you'll use 95% of the time:

- **`Layout.ROW`** — the default. Use for outputs. Use for inputs that
  don't feed a tensor-core instruction (e.g. element-wise kernels like
  RMS Norm, SwiGLU). No swizzle, plain row-major DRAM walk.
- **`Layout.TMA_128B`** — the canonical WGMMA/tcgen05 input swizzle.
  Use for inputs that feed `wgmma.mma_async` or `tcgen05.mma`. The
  TMA engine writes data into SMEM in a permuted pattern that WGMMA
  reads back as logical row-major. **Required** for any bf16/f16
  operand with row width ≥ 128 bytes (N ≥ 64 for bf16).
- **`Layout.TMA_64B` / `TMA_32B`** — smaller swizzle variants used
  when the row is narrower than 128 bytes. Usually picked
  automatically by `Tile.wgmma_a` / `Tile.wgmma_b` — you only reach
  for these manually when you're building a kernel with an unusual
  operand shape.

`Layout.COL` and `Layout.INTERLEAVED` exist for completeness; the
maintained examples don't use them.

## The Swizzle Matching Rule

The single most important fact about `Layout.TMA_*B`:

> **The TMA swizzle and the SMEM swizzle must be the same.**

TMA writes into SMEM using one permutation. WGMMA reads from SMEM
using another. These two permutations compose to identity — giving
you logical row-major order back — only if they're the same swizzle
family. Mismatched swizzles produce garbage output as soon as the
result depends on the K-pairing across slices.

Concretely:

```python
# In the @kernel spec:
Tile.wgmma_a(M, K, bf16, tile_m=BM, tile_k=tile_k)   # → Layout.TMA_128B

# In the kernel body:
sA = smem.wgmma_tile(bf16, (BM, tile_k), major="K")   # also 128B
```

`Tile.wgmma_a` picks `Layout.TMA_128B` when the row is ≥ 128 bytes
(which covers most real cases); `smem.wgmma_tile` picks its swizzle
based on the same tile shape. They match by construction. If you
build the TMA and SMEM sides by hand without the `wgmma_*` shortcuts,
**you are responsible for matching them** — they don't check each
other.

## When To Use `Tile.wgmma_a` vs Plain `Tile`

Rule of thumb:

- **`Tile.wgmma_a(M, K, bf16, tile_m=BM, tile_k=tile_k)`**: for any
  input that feeds `wgmma.mma_async` or `tcgen05.mma`. This picks
  the right `Layout.TMA_*B`, sets `tma_box=(tile_m, tile_k)`, and
  the SMEM swizzle you'll allocate via `smem.wgmma_tile(...)` will
  compose correctly. Use `wgmma_b(K, N, bf16, tile_k, tile_n)` for
  the B operand.
- **`Tile(M, N, f32, Layout.ROW)`**: for outputs and for non-MMA
  inputs (element-wise kernels). No TMA, plain row-major access via
  `ld.global.*` / `st.global.*`.
- **`Tile(M, K, bf16, Layout.TMA_128B, tma_box=(BM, BK))`**: the
  explicit form, used by the Blackwell flagship kernel. Same result
  as `Tile.wgmma_a(M, K, bf16, tile_m=BM, tile_k=BK)` but makes the
  layout choice visible. Use when you want to set the layout
  deliberately (e.g. Blackwell, where you may want different swizzle
  patterns than the auto-picked WGMMA one).

## `tma_box`: The Box Shape Per TMA Load

The TMA descriptor built by the framework knows two shapes:

- The **tensor shape** — the full tensor in DRAM (M × K, say).
- The **box shape** — how much one TMA load brings in per issue.

`tma_box=(BM, BK)` means: each `cp.async.bulk.tensor_2d` call with
this descriptor transfers a box of `BM × BK` elements.

```python
@kernel(
    in_specs=(
        Tile(M, K, bf16, Layout.TMA_128B, tma_box=(BM, BK)),
        ...
    ),
)
def kernel(A, B, D):
    sA = smem.alloc(..., (BM, BK))
    # Each TMA load transfers exactly BM * BK * sizeof(bf16) bytes.
    ptx.cp.async_.bulk.tensor_2d(
        dst=sA[0], src=A.tma_desc(),
        coord=(k_off, m_row_base), mbar=...
    )
```

The coordinates `(k_off, m_row_base)` are the **top-left corner** of
the box in the source tensor's coordinate system. The TMA engine then
transfers `(tma_box[0], tma_box[1])` elements starting from that
corner.

Picking the box:

- **Match your SMEM tile.** If your SMEM allocation is `(BM, BK)`,
  your `tma_box` should be `(BM, BK)`. Mismatched sizes either waste
  SMEM or read past the end.
- **Align to 16-byte boundaries.** TMA requires the innermost box
  dim to span a multiple of 16 bytes (`BK * sizeof(dtype) % 16 == 0`).
  For bf16 (2 bytes/elem), that means `BK % 8 == 0`.
- **Smaller isn't always better.** Bigger boxes amortize TMA issue
  overhead and get better DRAM utilization, up to the point where
  SMEM pressure forces smaller stages. The Blackwell GEMM uses
  `(BM=128, BK=64)` which is 16 KB per A tile — comfortable.

When `tma_box` is `None` (the default), the TMA descriptor uses the
full tensor shape — one TMA load brings everything. That's what you
want for kernels where the whole tensor fits in SMEM, rare in
practice beyond toy shapes.

## 2D vs 3D TMA

The repo ships two TMA descriptor ranks:

- **`tma_rank=2`** (default): rank-2 descriptor, called via
  `ptx.cp.async_.bulk.tensor_2d(...)`. Matches the "2D box from a 2D
  tensor" mental model. Used by every maintained example.
- **`tma_rank=3`**: rank-3 descriptor with an explicit minor axis.
  Used by the high-perf Hopper GEMM example
  (`gemm_highperf_hopper.py`). Subtle win: the 3D form lets TMA stage
  wider blocks with less padding overhead in the epilogue.

For grouped GEMM (G problems), you don't need `tma_rank=3` — the
per-group offset math lives in the kernel body (`group * M + ...`)
and the descriptor stays 2D. 3D TMA is a Hopper performance
optimization, not a batching mechanism.

## Symbolic vs Concrete Dimensions

Shape dims can be ints or strings:

```python
Tile("M", "K", bf16, Layout.ROW)   # symbolic, bound at call time
Tile(64, 16, bf16, Layout.ROW)     # concrete, fixed at decoration time
```

Symbolic dims mean one `@kernel` can handle different input sizes
(one traced program, one cubin cache entry per resolved shape). The
pyptx model is **shape-specialized** — each concrete combination of
bound symbolic dims produces its own specialized PTX. A call with
`M=2048, K=8192` and a call with `M=4096, K=8192` each get their own
trace and their own cubin.

The trade-off:

- **Concrete dims** (`Tile(64, 16, bf16)`) generate one cubin, small
  cache, simple debug story.
- **Symbolic dims** (`Tile("M", "K", bf16)`) generate one cubin per
  observed shape, larger cache, more traces over the program's life.

For performance-critical kernels, it's common to make the **tile
sizes concrete** (BM, BN, BK are `int`s baked in) and the **tensor
sizes symbolic** (M, N, K are strings). That's what the `build_gemm`
pattern does: `build_gemm(M=2048, N=8192, K=4096)` concretizes every
shape inside the decorator.

## Output Tiles

Outputs are simpler:

```python
out_specs=(Tile(M, N, f32, Layout.ROW),)
```

No TMA (outputs go via `st.global.*`), so no swizzle, no `tma_box`.
`Layout.ROW` is the only sensible choice unless you have a specific
reason to write column-major (you usually don't — downstream JAX /
PyTorch consumers expect row-major).

The output shape contract still matters: if the kernel returns an
`f32` tensor and you declare `bf16` here, the launch will fail the
shape check.

## Full Example, Piece By Piece

The Hopper GEMM decorator:

```python
@kernel(
    in_specs=(
        Tile.wgmma_a(M, K, bf16, tile_m=64, tile_k=16),
        Tile.wgmma_b(K, N, bf16, tile_k=16, tile_n=64),
    ),
    out_specs=(Tile(M, N, f32),),
    grid=(N // 64, M // 64, 1),
    block=(128, 1, 1),
    arch="sm_90a",
)
```

Reading it:

- **A is m×k bf16**, WGMMA-ready. Each TMA load brings a `64 × 16`
  box. SMEM allocation will be `(64, 16)` with 128B swizzle.
- **B is k×n bf16**, WGMMA-ready. Each TMA load brings a `16 × 64`
  box. Same swizzle matching.
- **D is m×n f32**, plain row-major.
- **Grid** tiles the output: one CTA per `(64, 64)` tile of D.
- **Block** has 128 threads = one Hopper warpgroup (required for
  WGMMA).
- **Arch** selects the ISA — `sm_90a` for Hopper, `sm_100a` for
  Blackwell.

## Checklist For A New Kernel's Tile Specs

Before you write the kernel body, answer each question:

1. **What are the tensor shapes?** Write them as `Tile(..., dtype, layout)`.
   Pick symbolic vs concrete per dim.
2. **Do any inputs feed WGMMA or tcgen05?** → use `Tile.wgmma_a` /
   `Tile.wgmma_b`, and allocate SMEM via `smem.wgmma_tile(..., major=...)`.
   Otherwise `Layout.ROW` + `smem.alloc`.
3. **What's the per-load TMA box?** Match your SMEM tile. Check that
   the innermost dim × dtype size is a multiple of 16 bytes.
4. **Output dtype match?** Kernel writes `f32`, output spec says
   `f32`. No implicit casts.
5. **Grid shape?** How many CTAs cover your output. Usually
   `(N // BN, M // BM, ...)`.
6. **Block shape?** 128 threads for a single warpgroup (WGMMA). 256
   for two warpgroups (rare). 128 for Blackwell 1-SM. Other counts
   for non-MMA kernels.

## What To Read Next

- [Mbarriers and async sync](mbarriers.md) — the companion page that
  explains what `mbar=...` and `mbarrier.wait` actually do.
- [Fragment layouts](fragment-layouts.md) — how WGMMA and `tcgen05.ld`
  scatter the result across lanes in registers.
- [Shared Memory](smem.md) — the SMEM side of the TMA → SMEM →
  WGMMA pipeline.
