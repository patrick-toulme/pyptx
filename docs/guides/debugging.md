# Debugging kernels

A kernel that doesn't crash is fine; a kernel that crashes, hangs, or
silently produces garbage is what you actually spend time on. This is the
runbook.

## The three tools you use constantly

```python
# 1. Dump the PTX pyptx emitted — the authoritative artifact.
print(my_kernel.ptx())

# 2. Validate it with ptxas. Fails loudly on structural PTX errors,
#    and `-v` prints register + SMEM usage.
# (pyptx's driver JIT swallows line numbers; ptxas doesn't.)
```

```bash
/usr/local/cuda-12.8/bin/ptxas -arch=sm_90a -v /tmp/k.ptx -o /dev/null
```

```bash
# 3. Localize the failing launch when the error is async.
CUDA_LAUNCH_BLOCKING=1 python my_script.py
```

90% of debugging sessions end once one of these three has pointed at
the right line.

## Common failure modes

### `cuLaunchKernel failed with 700` (illegal memory access)

Almost always a bad address, not a bad value.

**Prime suspects:**

- **Descriptor `addr_byte_offset` wrong for multi-k WGMMA.** For
  K-major A, stepping K by 16 elements = 32 bytes. For MN-major B with
  *single-stripe* layout (row_bytes ≤ 128), stepping K by 16 rows =
  `16 * row_bytes` bytes, **not 32**. Multi-stripe B128 (row_bytes >
  128, e.g. BN=256 bf16) uses +32 per step because the stripe is
  laid out K-major internally. The distinction is in the GMMA
  descriptor math; verify with a targeted test by sweeping
  `b_k_offset = kk * X` for X in `[32, row_bytes, 16*row_bytes]` and
  comparing to a reference matmul.
- **Global-memory offset computed in the wrong dtype.** `row *
  N_BYTES` with a 32-bit `row` and `N_BYTES` that overflows 32 bits
  wraps silently. Use `mul.wide.u32` for the multiplication so the
  result is 64-bit.
- **TMA coord out of range.** `cp.async.bulk.tensor_Nd` with a coord
  that exceeds the descriptor's `tensorSize` reads past the end.

### `CUDA_ERROR_INVALID_PTX` with `Unknown symbol 'smem_N'` or `'mbar_K'`

The kernel allocated > 48 KB of shared memory, so it silently switched
to dynamic SMEM mode — but a few references still use the static
allocation names `smem_0`, `smem_1`, … which no longer exist.

**Fix:** declare the SMEM budget explicitly on `@kernel` so dynamic
mode is on from the start of the trace:

```python
@kernel(arch="sm_90a", smem=MY_KERNEL_SMEM_BYTES, ...)
def my_kernel(...):
    ...
```

When you compute a SMEM byte offset manually, remember that in dynamic
mode `sA.name` resolves to the shared base (`dyn_smem`), not to the
alloc — you need to add `sA.byte_offset`:

```python
sA_base = reg.scalar(u32)
ptx.inst.mov.b32(sA_base, sA.name)
if sA.byte_offset > 0:                      # dynamic mode
    ptx.inst.add.u32(sA_base, sA_base, sA.byte_offset)
```

### NaN or garbage output, no crash

Run correctness against a trivial reference first. If that passes at
small sizes and fails at large ones, the usual causes are:

- **Fragment layout off.** Every WGMMA output has a specific
  `(thread, row, col)` layout; store the wrong fragment index to the
  wrong position and everything downstream is permuted. Dump a single
  fragment element to known positions in an output buffer and
  inspect it in Python before writing the full scatter.
- **Softmax running-state drift.** For flash attention, initializing
  `m_i = -inf` as `0` instead of `-1e30` makes the first `alpha =
  exp2(m_i_old - m_i_new)` produce garbage. The running `l_i` then
  accumulates the wrong scale for every subsequent block.
- **Swizzle on write not matching swizzle on read.** If you write P to
  SMEM with `apply_swizzle(logical, "32B")` and the consuming WGMMA
  descriptor has B128 swizzle, the `k` indices come back permuted.
  Verify that `smem.wgmma_tile(dtype, shape, major)` on the consumer
  side picks the same swizzle your writer applied.

### Launch overhead is ~35 µs instead of ~14 µs

The C++ torch extension fell back to the ctypes path because
`ninja` isn't installed.

```bash
pip install ninja
```

Verify the fast path loaded:

```python
import pyptx.torch_support as ts
assert ts._try_load_cpp_ext() is not None
```

### Kernel hangs

Usually a deadlocked mbarrier.

- **Expected arrive count mismatches actual arrivers.**
  `mbarrier.init(bar, N)` expects N arrives per phase; issue fewer and
  the wait never completes. For warp-specialized kernels, be explicit
  about which warpgroup is arriving, how many times, and count it on
  paper before wiring it up.
- **`wait_group` vs `mbarrier.wait` confusion.** `wgmma.wait_group(0)`
  waits for WGMMA completion; `mbarrier.wait(bar, phase)` waits for a
  TMA transfer. Mixing the two up waits forever.

## Inspecting the generated PTX

The PTX pyptx emits is the authoritative artifact. It's easy to read
once you know what to look for:

```bash
# Count registers used per kernel (high counts can hurt occupancy).
/usr/local/cuda-12.8/bin/ptxas -arch=sm_90a -v /tmp/k.ptx -o /dev/null 2>&1 \
    | grep "Used .* registers"

# Extract the instruction mix — useful for spotting unintended
# scalar loops in what should be vectorized code.
grep -oE '^\s*[a-z]+\.[a-z0-9._]+' /tmp/k.ptx | sort | uniq -c | sort -rn
```

For a transpiled kernel you're porting, `cuobjdump` gets the PTX out
of a compiled `.cubin`:

```bash
cuobjdump -ptx original.cubin > original.ptx
python -m pyptx.codegen original.ptx --sugar > port.py
```

## When in doubt

- Run correctness at the smallest shape that exercises the feature.
  A broken kernel usually fails at any size; a layout bug often only
  fails past some threshold.
- Compare PTX before and after a change: `diff before.ptx after.ptx`.
  A one-line DSL edit should produce a small, local PTX diff. If it
  didn't, you changed something you didn't mean to.
- Reach for `CUDA_LAUNCH_BLOCKING=1` before reaching for a debugger.
  Async errors come from the *previous* launch; blocking pins them.
