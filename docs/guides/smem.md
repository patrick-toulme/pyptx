# Shared Memory

The `smem` module handles shared-memory allocation, addressing, barriers, and GMMA layout helpers.

## Allocation

```python
from pyptx import smem
from pyptx.types import bf16, f32

# Plain allocation (row-major, no swizzle)
scratch = smem.alloc(f32, (64, 16))

# GMMA-compatible allocation (swizzled for wgmma)
sA = smem.wgmma_tile(bf16, (64, 16), major="K")
sB = smem.wgmma_tile(bf16, (16, 128), major="MN")
```

`smem.alloc` creates a shared-memory region with optional swizzle metadata.
`smem.wgmma_tile` picks the canonical GMMA layout automatically — the right swizzle mode, alignment, and descriptor metadata for `ptx.wgmma.mma_async`.

## 2D Element Access

Plain allocations support 2D indexing with `sA[row, col]`:

```python
scratch = smem.alloc(f32, (16, 16))

# Load (emits ld.shared.f32)
val = scratch[row, col]

# Store (emits st.shared.f32)
scratch[row, col] = val
```

This assumes row-major layout. It does **not** account for GMMA swizzle modes — use it for scratch buffers, not for wgmma operands.

## Mbarriers

```python
bar = smem.mbarrier(3)  # 3-element mbarrier array

ptx.mbarrier.init(bar[0], 1)
ptx.mbarrier.arrive_expect_tx(bar[0], 2048)
ptx.mbarrier.wait(bar[0], phase)
```

Mbarrier arrays are the synchronization primitive for TMA loads. Each TMA load targets a specific mbarrier; `arrive_expect_tx` tells the barrier how many bytes to expect, and `wait` blocks until all bytes arrive.

## GMMA Swizzle

When chaining wgmma operations (e.g., writing the output of one wgmma to SMEM as the input for the next), the data must be stored in the GMMA-compatible swizzled layout. The `apply_swizzle` function computes physical byte offsets from logical row-major offsets:

```python
from pyptx.smem import apply_swizzle

# Logical byte offset for element P[m][k] in row-major:
# logical = m * K * elem_bytes + k * elem_bytes
logical = reg.scalar(u32)
ptx.inst.mad.lo.u32(logical, row, K * 2, col_byte)

# Apply B32 swizzle (for bf16 with K=16)
physical = apply_swizzle(logical, "32B")

# Store at the swizzled address
addr = reg.scalar(u32)
ptx.inst.add.u32(addr, smem_base, physical)
ptx.inst.st.shared.b16(ptx.addr(addr), value)
```

### Swizzle Modes

| Mode | Row Width | Use Case |
|------|----------|----------|
| `None` | any | No permutation (scratch buffers) |
| `"32B"` | 32 bytes | bf16/f16 with K=16 |
| `"64B"` | 64 bytes | bf16/f16 with K=32, or f32 with K=16 |
| `"128B"` | 128 bytes | bf16/f16 with K=64, or f32 with K=32 |

The swizzle mode is determined by `row_bytes = K × element_bytes`. Use `wgmma_tile` to pick it automatically, or call `apply_swizzle` when storing computed values (like softmax probabilities) to SMEM for the next wgmma.

### Under The Hood

The swizzle follows CUTLASS's `Swizzle<B, M, S>` pattern. For B32 (`Swizzle<1, 4, 3>`):

```
physical = logical XOR ((logical & 0x80) >> 4)
```

This XORs bit 3 with bit 7 of the byte offset, swapping 8-byte blocks within rows where the row group index has bit 2 set (rows 4–7, 12–15, etc. within each 8-row group).

## Dynamic Shared Memory

For kernels needing >48 KB of shared memory, pass `smem=N` to `@kernel`:

```python
@kernel(..., smem=217648)
def gemm(...):
    ...
```

All `smem.alloc` and `smem.mbarrier` calls automatically use offset-based addressing into a single `extern .shared .b8 dyn_smem[]` declaration.
