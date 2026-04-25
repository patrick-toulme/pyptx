# Fragment Layouts

Every WGMMA and tcgen05 kernel epilogue starts with bitmath:

```python
wid = tid >> 5
lane = tid & 31
frag_row = (wid << 4) + (lane >> 2)
frag_col = (lane & 3) << 1
```

This guide explains where those shifts and masks come from. Once you
see the pattern, you can write an epilogue for any WGMMA shape, read
tcgen05 output via `tcgen05.ld`, and understand why SMEM swizzle
matters.

## The Core Problem

Tensor-core instructions (WGMMA, tcgen05) don't hand each thread a
contiguous slice of the result. They distribute fragments of the
output tile across **lanes in registers** in a specific pattern that
matches how the hardware computed each element. The epilogue's job
is to decode that pattern — figure out which output rows and columns
this particular lane owns — and scatter those registers to their
global memory destinations.

## WGMMA m64nN k16: The Canonical Layout

Hopper's WGMMA issues produce an `m64 x N` output tile per warpgroup
(128 threads). For `m64n64k16`, the `64 x 64` result is distributed
so that:

- The warpgroup has **4 warps**, each owning 16 logical rows.
- Each warp has **32 lanes**, distributed across 8 rows × 4 column
  groups.
- Each lane owns **two adjacent columns** in each 8-wide column
  group.
- Each lane's registers are laid out as `[col0_row0, col1_row0,
  col0_row8, col1_row8]` per column group — **pairs of rows offset
  by 8**.

That last point is the non-obvious one. Let me draw it out.

## The WGMMA Fragment Picture

For `m64n64k16`, each of the 128 threads gets 32 `f32` accumulator
values. Here's where they sit in the output tile:

```
Warp 0 (lanes 0-31):   output rows  0-15
Warp 1 (lanes 0-31):   output rows 16-31
Warp 2 (lanes 0-31):   output rows 32-47
Warp 3 (lanes 0-31):   output rows 48-63
```

Within a warp, lanes are arranged in an 8×4 grid:

```
                col_group_0  col_group_1  ...  col_group_7
                (cols 0-7)   (cols 8-15)       (cols 56-63)
row_offset 0:   lane 0 (c0,c1)  lane 0 (c8,c9)  ...  lane 0 (c56,c57)
row_offset 0:   lane 1 (c2,c3)  lane 1 (c10,c11) ...
row_offset 0:   lane 2 (c4,c5)  ...
row_offset 0:   lane 3 (c6,c7)  ...
row_offset 1:   lane 4 (c0,c1)  ...
row_offset 1:   lane 5 (c2,c3)  ...
...
row_offset 7:   lane 31 (c6,c7) ...
```

And each lane's 32 registers pack two values per column group:

```
acc[0]  acc[1]  = row+0 col+0, row+0 col+1    (col_group_0)
acc[2]  acc[3]  = row+8 col+0, row+8 col+1    (col_group_0, row offset by 8!)
acc[4]  acc[5]  = row+0 col+8, row+0 col+9    (col_group_1)
acc[6]  acc[7]  = row+8 col+8, row+8 col+9    (col_group_1)
...
acc[30] acc[31] = row+8 col+56, row+8 col+57  (col_group_7)
```

That's **32 = 8 column groups × 2 rows × 2 columns** per lane. Eight
of them are at `frag_row`, eight at `frag_row + 8`.

## Decoding The Bitmath

Given the picture, the formulas are:

```python
wid = tid >> 5                        # which warp (0-3)
lane = tid & 31                       # lane within warp (0-31)
frag_row = (wid << 4) + (lane >> 2)   # wid * 16 + lane / 4
frag_col = (lane & 3) << 1            # (lane % 4) * 2
```

Reading line by line:

- **`wid = tid >> 5`**: tid / 32, which warp this thread is in.
- **`lane = tid & 31`**: tid % 32, which lane within the warp.
- **`frag_row = (wid << 4) + (lane >> 2)`**:
  - `wid << 4` = `wid * 16` = starting row for this warp.
  - `lane >> 2` = `lane / 4` = row offset within the warp (0-7).
  - Sum = this lane's "top" row. The "bottom" row is `frag_row + 8`.
- **`frag_col = (lane & 3) << 1`**:
  - `lane & 3` = `lane % 4` = column group index within the 4-lane
    cluster.
  - `<< 1` = multiply by 2 = pair of columns this lane owns within
    an 8-wide group.

Once `frag_row` and `frag_col` are known, the epilogue writes all 32
registers with a double loop:

```python
for g in range(8):              # 8 column groups (each 8 cols wide)
    for li, (is_b, c_off) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        row = frag_row + 8 if is_b else frag_row
        col = frag_col + g * 8 + c_off
        off = (row * N + col) * 4
        ptx.inst.st.global_.f32(ptx.addr(pc + off), acc[g * 4 + li])
```

The inner iterator is `[(0,0), (0,1), (1,0), (1,1)]`:

- `(0, 0)` → `acc[g*4 + 0]` → row=`frag_row`, col=`frag_col + g*8 + 0`
- `(0, 1)` → `acc[g*4 + 1]` → row=`frag_row`, col=`frag_col + g*8 + 1`
- `(1, 0)` → `acc[g*4 + 2]` → row=`frag_row + 8`, col=`frag_col + g*8 + 0`
- `(1, 1)` → `acc[g*4 + 3]` → row=`frag_row + 8`, col=`frag_col + g*8 + 1`

That's exactly the pack order from the picture above.

## Smaller `tile_n` Values

`Tile.wgmma_b(..., tile_n=N)` scales the same pattern to narrower
output tiles. `tile_n` must be a power of 2 in the range 8–256, and
the number of accumulator registers per lane is `tile_n / 2`:

| `tile_n` | `acc_count` | column groups | epilogue col loop |
|---|---|---|---|
| 8  | 4  | 1 | `g in range(1)` |
| 16 | 8  | 2 | `g in range(2)` |
| 32 | 16 | 4 | `g in range(4)` |
| 64 | 32 | 8 | `g in range(8)` |
| 128 | 64 | 16 | `g in range(16)` |
| 256 | 128 | 32 | `g in range(32)` |

The **per-lane register pattern (2 cols × 2 row-pairs per group)** is
identical across all `tile_n` values. Only the number of column
groups changes. So:

```python
# Grouped GEMM epilogue for arbitrary tile_n:
for g in range(tile_n // 8):
    col = frag_col + g * 8
    off_a = (frag_row * N + col) * 4
    ptx.inst.st.global_.v2.f32(ptx.addr(pc + off_a),
                                [acc[g * 4], acc[g * 4 + 1]])
    off_b = (row_b * N + col) * 4
    ptx.inst.st.global_.v2.f32(ptx.addr(pc + off_b),
                                [acc[g * 4 + 2], acc[g * 4 + 3]])
```

This uses `st.global.v2.f32` to pack the two adjacent-column values
per row into one 8-byte store. Half the stores of the naive scalar
version, same total bytes.

## Blackwell tcgen05: A Different Story

Blackwell's `tcgen05.mma` doesn't scatter the result across lane
registers at all. The accumulator lives in **TMEM**, a separate
memory space. The epilogue has to:

1. Compute this thread's TMEM address.
2. Call `tcgen05.ld` to pull the fragment into registers.
3. Wait for the load to retire.
4. Store the registers to global memory.

The address computation is where the equivalent of `(wid << 4) + ...`
lives for tcgen05. For the flagship Blackwell GEMM:

```python
# Thread T reads "data path" index T — rows 0..127 cover 128 output rows.
row_base = m_base + tid

# TMEM address for this thread's slice.
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
    # ...v4 stores of out[...] to global...
```

Five things to notice:

1. **`tid` maps directly to TMEM row.** Thread `tid` reads the output
   row at `m_base + tid`. Much simpler than WGMMA's
   warp/lane arithmetic.
2. **`(tid << 16) & 0x3E00000`**: bit-packing the data path index
   into the right TMEM address field. The `<< 16` aligns it to the
   DP bits; the mask keeps only the valid 5 bits of DP index (0-127
   actually uses 7 bits, but the mask is 0x3E00000 which is 5 bits
   shifted — the low bit of the DP is held elsewhere in the address).
3. **`shape="32x32b", count=128`**: ask `tcgen05.ld` for 128 32-bit
   values arranged in a 32x32 pattern. Matches the accumulator tile.
4. **`wait_ld`** blocks until the load retires — `tcgen05.ld` is
   async.
5. **Chunking by 128**: for `BN=256`, this loop runs twice. Each
   chunk pulls 128 registers into `out` and then scatters them to
   global memory with `st.global.v4.b32`.

## SMEM Swizzle (B32 / B64 / B128) — Why It Matters

WGMMA doesn't read from SMEM row-major. It reads via a **swizzled
permutation** that matches how the tensor cores expect their
operands laid out. The TMA engine has to **write the SMEM in that
same permutation**, or the read comes out jumbled.

The four canonical swizzle classes:

| Swizzle | Row width | Use case |
|---|---|---|
| INTERLEAVE | 16 bytes (1×uint128) | No swizzle; narrow rows |
| B32 | 32 bytes (2×uint128) | 16-element bf16 rows |
| B64 | 64 bytes (4×uint128) | 32-element bf16 rows |
| B128 | 128 bytes (8×uint128) | ≥64-element bf16 rows |

The rule enforced by `Tile.wgmma_a` / `Tile.wgmma_b` and
`smem.wgmma_tile`:

- Row width in bytes → swizzle class → `Layout.TMA_*B`.
- Same class picked on both the TMA side and the SMEM side.

You'll see this concretely in `pyptx/wgmma_layout.py`:

```python
_LAYOUT_BY_ROW_BYTES = {
    16:  LayoutType.INTERLEAVE,
    32:  LayoutType.B32,
    64:  LayoutType.B64,
    128: LayoutType.B128,
}
```

Row width = `inner_dim * element_bytes`, clamped to 128 at the top.

**Why this matters for fragment layouts:** the fragment picture
above assumes correctly-swizzled SMEM. If your swizzle is wrong,
the WGMMA output is still in the pattern described — but the
underlying computation read jumbled K-slices, so the *values* are
garbage even though the *layout* is as expected. That's the failure
mode: no crash, no warning, just wrong numbers.

## Why Fragment Math Looks So Hand-Coded

Because it is. WGMMA and tcgen05 are hardware-defined instruction
patterns; the per-lane register layout is part of the ISA spec, not
something the compiler decides. If you write the right bitmath, every
lane stores to the right spot. If you don't, it's silent wrong
output.

pyptx's design choice: surface the bitmath as Python, don't hide it.
The alternative (a compiler that auto-generates the epilogue from a
high-level `store_tile(C, acc)` call) would be more ergonomic but
would hide the instruction pattern from the kernel author. If the
author wants a non-standard epilogue — TMA store, partial store,
softmax-in-place, reduction over N — they need to see the fragment
layout directly.

## Checklist For Writing A New Epilogue

1. **Know your WGMMA shape.** `m64nN k16` means 4 warps × 16 rows
   per warp × N columns. The fragment formulas assume this.
2. **Compute `frag_row` and `frag_col`.** Exactly the bitmath above.
   Shift into global coordinates by adding your CTA's `m_base` and
   `n_base`.
3. **Walk column groups.** `for g in range(tile_n // 8)`. Each
   iteration handles 4 registers (2 rows × 2 cols).
4. **Use `st.global.v2` for row-pair stores.** Adjacent columns in
   the same row can coalesce into one 8-byte store.
5. **For `tile_n >= 64`, consider `v4` stores** across row pairs if
   memory alignment allows.

For Blackwell:

1. **Compute TMEM address** from `tid` (or warp/lane) per the ISA
   reference for your `tcgen05.ld` shape.
2. **`tcgen05.ld` + `wait_ld`** to pull into registers.
3. **v4 global stores** from the loaded registers.
4. **Dealloc TMEM** at kernel exit.

## What To Read Next

- [Hopper GEMM](handwritten-gemm.md) — has the canonical
  `m64n64k16` epilogue.
- [Grouped GEMM](grouped-gemm.md) — shows the same pattern
  parameterized over `tile_n`.
- [Blackwell GEMM](blackwell-gemm.md) — the tcgen05 / TMEM
  variant.
- `pyptx/wgmma_layout.py` — source of the swizzle → descriptor
  mapping, if you need to go one layer deeper.
