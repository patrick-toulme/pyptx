# Blackwell GEMM

This page walks through `examples/blackwell/gemm_highperf_blackwell.py`
— the flagship Blackwell (`sm_100a`) kernel. It closely mirrors the
JAX Pallas `blackwell_matmul_mgpu.py` reference, written in pyptx.

Blackwell introduces a completely different tensor-core path from
Hopper's WGMMA. This guide explains the four pieces you need in mind
before reading the kernel:

1. **`tcgen05.mma`** — Blackwell's async MMA, issued by **one thread**
   per CTA instead of the 128-thread warpgroup WGMMA required.
2. **TMEM (Tensor Memory)** — a new memory space, separate from SMEM
   and registers, that holds the MMA accumulator.
3. **SMEM + instruction descriptors** — compact 64-bit values that
   encode operand layout and MMA shape for tcgen05.
4. **Warp specialization** — the Blackwell idiom of dedicating one
   warp to TMA issue, one warp to MMA dispatch, so producer and
   consumer run independently with mbarrier sync.

Read [Hopper GEMM](handwritten-gemm.md) first if you haven't — this
guide assumes you're familiar with the WGMMA + TMA + mbarrier loop.

## What The Kernel Computes

Standard GEMM: `D[M, N] = A[M, K] @ B[N, K].T`, with A and B in bf16,
D in f32. Tile sizes fixed at `BM=128, BN=256, BK=64`, with
`k_per_instr=16` so each outer K-tile issues 4 tcgen05 MMAs.

The repo also ships a 2-SM cooperative variant (`build_gemm_2sm`) that
dispatches `tcgen05.mma.cta_group::2` across a 2-CTA cluster — two SMs
share one MMA issue. That's a separate kernel; this guide covers the
1-SM version.

## The Four New Primitives

### tcgen05.mma is issued by one thread, not a warpgroup

On Hopper, every warpgroup (128 threads) executes the same `wgmma`
instruction and each thread owns a slice of the fragment. On
Blackwell, **one thread** issues the whole MMA for the CTA:

```python
ptx.tcgen05.mma(tmem_base, desc_a, desc_b, idesc, kind="f16",
                pred_operand=(not is_first))
```

The result accumulates **into TMEM**, not into the issuing thread's
registers. All threads of the CTA will later read their slice of the
accumulator from TMEM in the epilogue.

Because only one thread issues the MMA, warp specialization becomes
natural: dedicate warp 1 lane 0 to MMA issue, free up the other warps
to do other work (TMA, epilogue prep, etc.).

### TMEM is a third memory space

TMEM is separate from registers and SMEM:

- You allocate it explicitly: `ptx.tcgen05.alloc(tmem_slot, 512)`
  reserves 512 columns.
- You read its base via a normal SMEM load:
  `tmem_base = smem.load(b32, ptx.addr(tmem_slot))`.
- You load **from** TMEM into registers with `tcgen05.ld`:
  ```python
  ptx.tcgen05.ld([out[i] for i in range(128)], tmem_addr,
                 shape="32x32b", count=128, dtype="b32")
  ```
- And you must free it at kernel exit: `tcgen05.dealloc(...)` +
  `tcgen05.relinquish_alloc_permit()`.

The lifetime discipline is real — forgetting to dealloc is a kernel
crash on the next launch.

### SMEM and instruction descriptors

Each MMA issue takes three descriptors:

- **SMEM descriptor for A** — 64-bit value encoding A's SMEM base,
  stride, and swizzle pattern.
- **SMEM descriptor for B** — same, for B.
- **Instruction descriptor** — encodes MMA shape, dtype, transpose,
  and other hardware flags. Built once per kernel:
  ```python
  idesc = reg.scalar(b32, init=ptx.tcgen05.make_instr_desc_f16bf16_f32())
  ```

The SMEM descriptors are computed from a constant mask plus the SMEM
pointer:

```python
MMA_DESC_B128 = 0x4000404000010000
desc_a0 = ptx.tcgen05.masked_descriptor(smem_a, const_bits=MMA_DESC_B128)
desc_b0 = ptx.tcgen05.masked_descriptor(smem_b, const_bits=MMA_DESC_B128)
```

Inside a K-tile, successive 16-wide MMA slices advance by a constant
byte offset in the descriptor:

```python
for kk in range(MMAS_PER_KTILE):
    if kk == 0:
        desc_a, desc_b = desc_a0, desc_b0
    else:
        desc_a = reg.scalar(b64); desc_b = reg.scalar(b64)
        ptx.inst.add.s64(desc_a, desc_a0, kk * 2)
        ptx.inst.add.s64(desc_b, desc_b0, kk * 2)
```

That `+ kk * 2` is the canonical "advance by 2 bytes in the descriptor
to skip a 16-wide K-slice" trick — tcgen05 descriptors pack address
bits at specific offsets and this arithmetic walks them directly.

### Warp specialization with mbarriers

The CTA is split into two specialized warps plus two dozen free threads:

- **TMA warp** (warp 0, lane 0): issues `cp.async.bulk.tensor_2d` for A
  and B into a 4-stage SMEM ring buffer, signals arrival on `bar_load`.
- **MMA warp** (warp 1, lane 0): waits on `bar_load`, issues the four
  tcgen05 MMAs per K-tile, signals `bar_consumed` when the SMEM slot is
  free to be refilled.
- **Everybody** waits on `bar_mma` at the end before running the
  epilogue.

The ring buffer lets TMA for K-tile `ki+STAGES-1` overlap MMA for
`ki`. With `STAGES=4` you can have three TMAs in flight while one MMA
is running — enough to hide HBM latency on B200.

## Step 1: SMEM Layout

```python
STAGES = 4
A_STAGE = BM * BK * 2   # 16 KB per A stage
B_STAGE = BN * BK * 2   # 32 KB per B stage

SMEM_A_BASE       = 0
SMEM_B_BASE       = SMEM_A_BASE + STAGES * A_STAGE
SMEM_BAR_LOAD     = SMEM_B_BASE + STAGES * B_STAGE
SMEM_BAR_CONSUMED = SMEM_BAR_LOAD + STAGES * 8
SMEM_BAR_MMA      = SMEM_BAR_CONSUMED + STAGES * 8
SMEM_TMEM_SLOT    = SMEM_BAR_MMA + 8
SMEM_BYTES        = SMEM_TMEM_SLOT + 16
```

That's 192 KB+ of dynamic SMEM — well beyond the 48 KB static limit,
so the kernel uses `extern_smem=True` and the framework calls
`cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM_BYTES)` at
launch.

The SMEM layout is hand-placed: A tiles, then B tiles, then
per-stage barriers, then a scratchpad for the TMEM pointer. Addresses
are resolved via `base = smem.base()` and offset arithmetic; no SMEM
allocator.

## Step 2: Hilbert-Curve Tile Scheduling

The kernel doesn't walk tiles row-major. Instead it groups tiles into
`grid_tile_width` sub-groups along one axis and alternates the inner
direction:

```python
ptx.inst.div.u32(group_id, tile_linear, group_span)
ptx.inst.rem.u32(group_offset, tile_linear, group_span)
# ...forward vs reverse minor index based on group_id parity
is_odd_group = (group_id & 1) != 0
tile_minor = selp(minor_rev, minor_fwd, is_odd_group)
```

This is a **space-filling curve for L2 locality**: tiles close in time
are close in the output, so the same A/B rows are reused while they're
still resident in L2. On large GEMMs it's worth 5–10% over plain
row-major tile scheduling.

The math is verbose but deterministic — `div.u32`, `rem.u32`, `selp.b32`,
all primitives the DSL surfaces directly.

## Step 3: Producer — The TMA Warp

```python
with ptx.if_(is_tma_warp):
    for ki in range(k_iters):
        slot = ki % stages
        smem_a = base + SMEM_A_BASE + slot * A_STAGE
        smem_b = base + smem_b_base + slot * B_STAGE
        mbar_l = bar_load + slot * 8
        mbar_c = bar_consumed + slot * 8

        if ki >= stages:
            # Slot reuse: wait for MMA to finish previous iter.
            consumed_phase = ((ki // stages) - 1) & 1
            with ptx.scope():
                ready = reg.scalar(pred)
                ptx.label(f"cwait_{ki}")
                ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
                    ready, ptx.addr(mbar_c), consumed_phase
                )
                ptx.bra(f"cdone_{ki}", pred=ready)
                ptx.bra(f"cwait_{ki}")
                ptx.label(f"cdone_{ki}")

        ptx.mbarrier.arrive_expect_tx(mbar_l, A_STAGE + B_STAGE)
        ptx.cp.async_.bulk.tensor_2d(
            dst=smem_a, src=A.tma_desc(),
            coord=(ki * BK, m_base), mbar=mbar_l,
        )
        ptx.cp.async_.bulk.tensor_2d(
            dst=smem_b, src=B_T.tma_desc(),
            coord=(ki * BK, n_base), mbar=mbar_l,
        )
```

Three things here:

1. **Slot reuse check.** For `ki >= stages`, the slot we want to
   refill is still being consumed by MMA. The spin-loop with
   `try_wait.parity` is the explicit "wait for consumer" pattern —
   each spin checks the `consumed` barrier's phase bit.
2. **`try_wait.parity` is spelled out manually** (labels + `bra`s)
   because the transpiler sugar for `mbarrier.wait` that lowers to the
   same form isn't used here — we want the exact PTX the Pallas
   reference emits, for bit-identical comparison.
3. **`arrive_expect_tx(..., A_STAGE + B_STAGE)`** tells the barrier
   how many bytes of async traffic to expect. A and B share one
   barrier per stage; the barrier becomes ready after both copies
   complete.

The producer warp never does math. It only moves bytes.

## Step 4: Consumer — The MMA Warp

```python
with ptx.if_(is_mma_warp):
    for ki in range(k_iters):
        slot = ki % stages
        # ...addresses + load_phase = (ki // stages) & 1

        # Wait for TMA to finish this slot.
        with ptx.scope():
            ready = reg.scalar(pred)
            ptx.label(f"lwait_{ki}")
            ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
                ready, ptx.addr(mbar_l), load_phase
            )
            ptx.bra(f"ldone_{ki}", pred=ready)
            ptx.bra(f"lwait_{ki}")
            ptx.label(f"ldone_{ki}")

        desc_a0 = ptx.tcgen05.masked_descriptor(smem_a, const_bits=MMA_DESC_B128)
        desc_b0 = ptx.tcgen05.masked_descriptor(smem_b, const_bits=MMA_DESC_B128)
        for kk in range(MMAS_PER_KTILE):
            if kk == 0:
                desc_a, desc_b = desc_a0, desc_b0
            else:
                desc_a = reg.scalar(b64); desc_b = reg.scalar(b64)
                ptx.inst.add.s64(desc_a, desc_a0, kk * 2)
                ptx.inst.add.s64(desc_b, desc_b0, kk * 2)
            is_first = (ki == 0 and kk == 0)
            ptx.tcgen05.mma(
                tmem_base, desc_a, desc_b, idesc,
                kind="f16", pred_operand=(not is_first),
            )

        # Signal this slot is free.
        ptx.mbarrier.arrive(mbar_c)

    # Commit after all K-tiles.
    ptx.tcgen05.commit(bar_mma, space="cluster")
```

Four things:

1. **`pred_operand=(not is_first)`** is the scale-D equivalent for
   tcgen05. On the very first MMA (`ki=0, kk=0`) we **do not**
   accumulate into prior TMEM state — treat the accumulator as fresh.
   Every subsequent MMA accumulates.
2. **4 MMAs per K-tile** — `MMAS_PER_KTILE = BK // K_PER_INSTR = 4`.
   All four target the same TMEM accumulator; different 16-wide
   K-slices of the same SMEM-resident A/B tiles.
3. **`arrive(mbar_c)`** signals slot reuse after the last MMA of a
   K-tile issues. The producer warp spins on this barrier if it ran
   ahead.
4. **`tcgen05.commit(bar_mma, space="cluster")`** is the MMA-done
   barrier. Only after commit will a `mbarrier.wait` on `bar_mma`
   succeed. Commit happens once after all K-tiles — tcgen05 MMAs are
   async and can pipeline freely until this commit.

## Step 5: Epilogue — TMEM Load + Global Store

```python
with ptx.scope():
    ready = reg.scalar(pred)
    ptx.label("cw")
    ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
        ready, ptx.addr(bar_mma), 0
    )
    ptx.bra("cd", pred=ready)
    ptx.bra("cw")
    ptx.label("cd")

# Thread T reads DP=T (warps 0..3 cover DPs 0..127).
row_base = m_base + tid
# ...compute d_ptr = D + (row * N + n_base) * 4

tmem_row_bits = (tid << 16) & 0x3E00000
tmem_addr = tmem_base + tmem_row_bits

out = reg.array(b32, 128)
for chunk in range(BN // 128):
    chunk_off = chunk * 128
    ptx.tcgen05.ld(
        [out[i] for i in range(128)],
        tmem_addr + chunk_off,
        shape="32x32b", count=128, dtype="b32",
    )
    ptx.tcgen05.wait_ld()

    for vec in range(128 // 4):
        off = (chunk_off + vec * 4) * 4
        ptx.inst.st.global_.v4.b32(
            ptx.addr(d_ptr, off),
            [out[vec * 4], out[vec * 4 + 1], out[vec * 4 + 2], out[vec * 4 + 3]],
        )
```

Epilogue breakdown:

1. **Wait for MMA commit** — all threads spin until `bar_mma` fires.
2. **Compute TMEM address for this thread.** Each thread's slice lives
   at `tmem_base + (tid << 16) & 0x3E00000` — the bit mask packs the
   "data path" index (which of the 128 accumulator rows) into the
   right TMEM address bits.
3. **`tcgen05.ld` with `shape="32x32b", count=128`** pulls 128 32-bit
   values from TMEM into registers. `wait_ld` blocks until the load
   retires.
4. **v4 global stores** scatter 128 registers to global memory, 4 at a
   time. `BN // 128` outer chunks — for `BN=256` that's 2 chunks of 128.

TMEM → registers → HBM is the required path. Blackwell doesn't have
a direct TMEM → HBM store.

## Step 6: Cleanup

```python
with ptx.if_(alloc_warp):
    ptx.tcgen05.dealloc(tmem_base, 512)
    ptx.tcgen05.relinquish_alloc_permit()
ptx.ret()
```

Dealloc must be called by the same warp that called alloc. If you
skip the dealloc or the relinquish, subsequent kernels on the same SM
can fail to allocate TMEM.

## The Full Timeline For One CTA

Put together:

1. Pick the CTA's `128 x 256` output tile from `ctaid.x, ctaid.y` via
   Hilbert scheduling.
2. Allocate 512 columns of TMEM; init `STAGES * 2 + 1` mbarriers.
3. **Producer**: for each K-tile, TMA-load A and B into the ring slot,
   signaling `bar_load` on arrival. Wait for `bar_consumed` before
   reusing a slot.
4. **Consumer**: for each K-tile, wait for `bar_load`, build the MMA
   descriptors, issue 4 `tcgen05.mma` calls that accumulate into
   TMEM, signal `bar_consumed`.
5. After all K-tiles: consumer calls `tcgen05.commit(bar_mma)`.
6. All threads wait on `bar_mma`. Each thread computes its TMEM
   address and does `tcgen05.ld` + v4 global stores.
7. Alloc warp calls `dealloc` + `relinquish_alloc_permit`.

## The 2-SM Variant

`build_gemm_2sm` extends this with `cta_group::2`: **two SMs share one
MMA issue**, cooperating across a 2-CTA cluster. The MMA warp in SM-A
dispatches for the whole cluster; both SMs share a single A tile and
each holds its own half of B.

The upside is fewer MMA issues total and larger effective tile sizes.
The downside is cluster-scoped barriers, more delicate lifetime rules,
and a 5-stage buffer (`STAGES_2SM = 5`) to hide the cross-SM
synchronization. Read `examples/blackwell/gemm_highperf_blackwell.py`
for the full `build_gemm_2sm`.

## Why This Kernel Matters For The DSL

The set of primitives exercised here is the Blackwell-only feature
list that nothing else in the Python DSL space exposes as first-class
calls:

- `ptx.tcgen05.alloc` / `.dealloc` / `.relinquish_alloc_permit`
- `ptx.tcgen05.mma(...)` with `pred_operand` and `cta_group::2` variant
- `ptx.tcgen05.ld(...)` / `.wait_ld` / `.commit`
- `ptx.tcgen05.make_instr_desc_f16bf16_f32()`
- `ptx.tcgen05.masked_descriptor(...)` for SMEM descriptors
- Dynamic SMEM > 48 KB via `smem=N, extern_smem=True` on the kernel
  decorator
- `ptx.mbarrier.try_wait.parity` spin-wait patterns, spelled with
  labels + `bra`

If the DSL forced you to drop to inline PTX for any of these, there'd
be no difference between pyptx and raw-PTX-in-ctypes for Blackwell
kernels. That everything stays inside the DSL is the point.

## What To Read Next

- `examples/blackwell/tcgen05_suite.py` — each of the primitives above
  in isolation, much easier to modify than the full GEMM.
- `examples/blackwell/tcgen05_mma_probe.py` — smallest-possible
  tcgen05.mma kernel. Start here if the GEMM is too much at once.
- [Hopper GEMM](handwritten-gemm.md) — the WGMMA analog for comparison.
- [Grouped GEMM](grouped-gemm.md) — same K-loop structure, Hopper
  path.
