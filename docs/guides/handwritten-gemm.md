# Hopper GEMM

This page is a step-by-step walkthrough of the maintained Hopper GEMM
example in `examples/hopper/gemm_highperf_hopper.py`.

!!! note "Looking for Blackwell?"
    The `sm_100a` equivalent is
    `examples/blackwell/gemm_highperf_blackwell.py` — same
    warp-specialized + ring-buffer pattern, but with `tcgen05.mma` +
    TMEM instead of WGMMA, plus a 2-SM `build_gemm_2sm` variant that
    dispatches `tcgen05.mma.cta_group::2` across a 2-CTA cluster. See
    `examples/blackwell/tcgen05_suite.py` for the underlying
    primitives in isolation.

The kernel is deliberately small enough to read in one sitting, but still shows the core Hopper GEMM shape:

- TMA loads into shared memory
- mbarrier-based staging
- WGMMA tensor-core math
- explicit fragment scatter in the epilogue

It is not the largest GEMM this repo has ever contained. It is the current example that best balances performance shape and readability.

## What The Kernel Computes

At a high level, it computes:

```text
C[M, N] = A[M, K] @ B[K, N]
```

with:

- `A` and `B` in `bf16`
- accumulation in `f32`
- output in `f32`

The maintained example specializes each CTA to a `64 x 64` output tile and each tensor-core instruction to `m64n64k16`.

## Step 1: Specialize The Kernel Shape

The outer builder fixes the tile sizes and derives the K-loop trip count:

```python
def build_gemm_m64n64(M: int, N: int, K: int, *, tile_k: int = 16):
    BM, BN = 64, 64
    assert M % BM == 0 and N % BN == 0 and K % tile_k == 0
    n_iters = K // tile_k
```

That tells you almost everything about the kernel:

- one CTA computes one `64 x 64` tile of `C`
- the CTA walks through `K` in chunks of `16`
- each loop iteration does one `wgmma.mma_async.m64n64k16`

The launch config then maps tiles onto the grid:

```python
@kernel(
    in_specs=(
        Tile.wgmma_a(M, K, bf16, tile_m=BM, tile_k=tile_k),
        Tile.wgmma_b(K, N, bf16, tile_k=tile_k, tile_n=BN),
    ),
    out_specs=(Tile(M, N, f32),),
    grid=(N // BN, M // BM, 1),
    block=(128, 1, 1),
    arch="sm_90a",
)
```

Interpret that as:

1. `ctaid.x` chooses the output tile column
2. `ctaid.y` chooses the output tile row
3. the CTA has `128` threads, which is exactly one Hopper warpgroup

That last point matters because WGMMA is warpgroup-scoped.

## Step 2: Allocate Shared-Memory Tiles And Accumulators

At the start of the kernel body, the code allocates the staging buffers and the accumulator fragment:

```python
sA = smem.wgmma_tile(bf16, (BM, tile_k), major="K")
sB = smem.wgmma_tile(bf16, (tile_k, BN), major="MN")
bars = [smem.mbarrier(1) for _ in range(n_iters)]
phases = [reg.scalar(b32, init=0) for _ in range(n_iters)]
acc = reg.array(f32, 32)
```

Each piece has a specific role:

- `sA`: shared-memory tile for the current `A[64, 16]`
- `sB`: shared-memory tile for the current `B[16, 64]`
- `bars[i]`: completion barrier for the `i`th TMA stage
- `acc`: the per-thread slice of the `64 x 64` WGMMA output fragment

The important non-obvious fact is `acc = reg.array(f32, 32)`.

For `m64n64k16`, each thread in the warpgroup owns `32` `f32` accumulator values. That is the register fragment that gets updated every iteration and later scattered to global memory.

## Step 3: Compute Which Output Tile This CTA Owns

Next the kernel computes the tile origins in `M` and `N`:

```python
row_off = reg.scalar(u32)
ptx.inst.mov.u32(row_off, ptx.special.ctaid.y())
ptx.inst.shl.b32(row_off, row_off, 6)  # * 64

col_off = reg.scalar(u32)
ptx.inst.mov.u32(col_off, ptx.special.ctaid.x())
ptx.inst.shl.b32(col_off, col_off, 6)  # * 64
```

Why shift by `6`?

Because `2^6 = 64`, so this is just:

- `row_off = ctaid.y * 64`
- `col_off = ctaid.x * 64`

From here on, every global-memory access in the CTA is relative to that output tile origin.

## Step 4: Initialize The TMA Barriers

The kernel uses one barrier per K-slice:

```python
tid = ptx.special.tid.x()
with ptx.if_(tid == 0):
    for bar in bars:
        ptx.mbarrier.init(bar[0], 1)
    ptx.fence.proxy_async_shared_cta()
```

Only thread `0` performs the initialization.

The mental model is:

1. create the synchronization objects in shared memory
2. fence so the async/TMA path sees them correctly
3. let the whole warpgroup reuse them during the main loop

This example is intentionally simple. It does not do a deep multi-stage producer/consumer pipeline. It keeps one barrier per loop iteration so the control flow is easy to follow.

## Step 5: Walk The K Dimension One Tensor-Core Tile At A Time

The main loop is:

```python
for i in range(n_iters):
    k_off = reg.scalar(u32)
    ptx.inst.mov.u32(k_off, i * tile_k)
    ...
```

Each iteration handles one `K` slice of width `16`.

So if `K = 2048`, then:

- `tile_k = 16`
- `n_iters = 128`
- the CTA performs `128` WGMMA instructions

Every iteration has the same shape:

1. prepare a TMA arrival expectation
2. TMA-load the `A` tile
3. TMA-load the `B` tile
4. wait until those loads are complete
5. issue one `wgmma.mma_async`
6. commit and wait for the tensor-core work

## Step 6: Issue TMA Loads For The Current `A` And `B` Tiles

The TMA path is only issued by thread `0`:

```python
with ptx.if_(tid == 0):
    ptx.mbarrier.arrive_expect_tx(
        bars[i][0], BM * tile_k * 2 + tile_k * BN * 2,
    )
    ptx.cp.async_.bulk.tensor_2d(
        dst=sA[0], src=A.tma_desc(),
        coord=(k_off, row_off), mbar=bars[i][0],
    )
    ptx.cp.async_.bulk.tensor_2d(
        dst=sB[0], src=B.tma_desc(),
        coord=(col_off, k_off), mbar=bars[i][0],
    )
```

Read those coordinates carefully:

- `A` uses `(k_off, row_off)`
- `B` uses `(col_off, k_off)`

That matches the logical slices:

- load `A[row_off:row_off+64, k_off:k_off+16]`
- load `B[k_off:k_off+16, col_off:col_off+64]`

The `arrive_expect_tx(...)` call tells the barrier how many bytes of async traffic to expect before the stage is considered complete.

In other words: the barrier is not “arrived” immediately. It becomes ready once the TMA engine finishes the requested transfers.

## Step 7: Synchronize The Warpgroup Before Using Shared Memory

After issuing the TMA loads, the CTA synchronizes:

```python
ptx.bar.sync(0)
ptx.mbarrier.wait(bars[i][0], phases[i])
```

These two waits serve different purposes:

- `bar.sync(0)`: make sure all threads in the CTA stay in lockstep around the stage
- `mbarrier.wait(...)`: wait for the async copy itself to complete

Only after both are done is it safe to treat `sA` and `sB` as valid WGMMA inputs.

## Step 8: Run The Tensor-Core Multiply-Accumulate

The actual math is compact:

```python
ptx.wgmma.fence()
ptx.wgmma.mma_async(
    shape=(64, 64, 16),
    dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
    d=acc, a=sA, b=sB,
    scale_d=(i != 0), trans_a=0, trans_b=1,
)
ptx.wgmma.commit_group()
ptx.wgmma.wait_group(0)
```

This is the heart of the kernel.

Interpret it as:

```text
acc = A_tile @ B_tile + acc
```

with one important detail:

- `scale_d=(i != 0)` means:
  - on the first iteration, do not scale the old accumulator
  - on later iterations, accumulate into the existing `acc`

That is how the loop turns many `m64n64k16` partial products into one full `64 x 64` GEMM tile.

The `trans_b=1` flag matches how `sB` is laid out for the WGMMA instruction.

## Step 9: Understand The Fragment Layout Before Reading The Epilogue

The epilogue looks intimidating if you do not first accept one fact:

WGMMA does not hand each thread a neat row-major mini-tensor.

Instead, each thread owns a scattered fragment of the `64 x 64` result.

So the epilogue has one job:

1. determine which logical rows/cols this lane owns
2. map the `32` accumulator registers onto those coordinates
3. store them one by one to global memory

The first part is:

```python
tid2 = reg.scalar(u32)
ptx.inst.mov.u32(tid2, ptx.special.tid.x())
wid = tid2 >> 5
lane = tid2 & 31
frag_row = (wid << 4) + (lane >> 2)
frag_col = (lane & 3) << 1
ptx.inst.add.u32(frag_row, frag_row, row_off)
ptx.inst.add.u32(frag_col, frag_col, col_off)
```

Break that down:

- `wid = tid / 32` chooses the warp inside the warpgroup
- `lane = tid % 32` chooses the lane inside that warp
- `frag_row` picks the base row for this lane
- `frag_col` picks the base column pair for this lane

Then `row_off` and `col_off` shift those local fragment coordinates into the global `C` tile owned by this CTA.

## Step 10: Scatter The Accumulator Registers To Global Memory

The final store loop is:

```python
(pc,) = ptx.global_ptrs(C)
for g in range(8):
    for li, (is_b, c_off) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        row = frag_row + 8 if is_b else frag_row
        col = frag_col + g * 8 + c_off
        off = (row * N + col) * 4
        ptx.inst.st.global_.f32(ptx.addr(pc + off), acc[g * 4 + li])
```

This is easier to read if you treat it as two nested decompositions:

- outer loop `g in range(8)`:
  walks across the `N` dimension in groups of `8` columns
- inner loop of `4` values:
  stores two columns for one row and two columns for the row `+8`

That is the fragment layout the earlier WGMMA produced.

So the epilogue is not arbitrary index math. It is a direct unpacking of the hardware fragment layout into row-major `C`.

## The Full Timeline For One CTA

If you want the whole kernel in one mental picture, read it like this:

1. pick the CTA's `64 x 64` output tile from `ctaid.x` and `ctaid.y`
2. allocate shared-memory staging for one `A` tile and one `B` tile
3. initialize one mbarrier per K-slice
4. for each `16`-wide K block:
   - TMA-load `A_tile` and `B_tile`
   - wait for the loads
   - issue `wgmma.mma_async.m64n64k16`
   - accumulate into `acc`
5. map each lane's fragment coordinates
6. scatter the `32` `f32` accumulator values to global `C`

That is the entire kernel.

## Why This Example Matters For The DSL

This example is a good DSL test because it forces the core surfaces to be honest:

- `Tile.wgmma_a(...)` and `Tile.wgmma_b(...)` must encode the right tensor layout contract
- `smem.wgmma_tile(...)` must model real shared-memory staging
- `ptx.cp.async_.bulk.tensor_2d(...)` must be explicit enough for TMA work
- `ptx.wgmma.mma_async(...)` must stay close to the real instruction
- register arithmetic must stay readable without hiding too much

If the DSL cannot make this kernel understandable, it is probably too low-level to be pleasant or too high-level to be trustworthy.

## What To Read Next

- generated page for `examples/hopper/gemm_highperf_hopper.py`
- [PTX Namespace](ptx-namespace.md)
- API docs for `pyptx.ptx`, `pyptx.reg`, and `pyptx.smem`
