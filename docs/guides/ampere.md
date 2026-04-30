# Ampere (A100) and `arch="auto"`

`pyptx` is first-class on Ampere. `mma.sync.aligned.m16n8k{8,16,32}`,
`cp.async`, `ldmatrix`, mbarriers, vector loads, warp shuffles —
everything you need for a real A100 kernel is exposed through the same
DSL surface used for Hopper and Blackwell.

This guide walks through:

1. picking the target arch (`sm_80`, `sm_90a`, `sm_100a`, or `"auto"`),
2. the typed `ptx.mma.sync(...)` wrapper,
3. the typed `ptx.cp.async_.{cg, ca, commit_group, wait_group, wait_all}`
   wrappers,
4. the two A100 GEMM examples shipped with the repo.

## 1. Picking the target arch

Every `@kernel` takes an `arch=` argument. For Ampere you want `sm_80`:

```python
from pyptx import kernel, ptx

@kernel(arch="sm_80", grid=(1, 1, 1), block=(32, 1, 1))
def k():
    ptx.ret()
```

If you don't know which GPU the kernel is going to land on, pass
`arch="auto"`. `pyptx` resolves it once at decorator time by reading the
device's compute capability:

```python
@kernel(arch="auto", grid=(1, 1, 1), block=(32, 1, 1))
def k():
    ptx.ret()
```

| Compute capability | Resolved arch |
| --- | --- |
| 7.5 (T4) | `sm_75` (elementwise only — no `cp.async`/`bf16`/`mbarrier`) |
| 8.0 (A100), 8.6 (RTX 30xx), 8.7 (Jetson AGX Orin), 8.9 (L4 / RTX 40xx, L40) | `sm_80` / `sm_86` / `sm_87` / `sm_89` |
| 9.0 (H100, H200) | `sm_90a` (datacenter — `wgmma`/TMA) |
| 10.0 (B200, GB200) | `sm_100a` (datacenter — `tcgen05`/TMEM) |
| 12.0 (RTX Pro 6000 Blackwell, RTX 50xx) | `sm_120` (workstation — Ampere-class tensor cores, no `tcgen05`/`wgmma`/TMA) |

The `a` suffix is reserved for datacenter Hopper / Blackwell parts that
ship the architecture-accelerated feature sets (`wgmma`, `tcgen05`).
Workstation Blackwell (sm_120) doesn't have those, so plain `sm_120`
is the right target — `sm_120a` is not a valid PTX target on those
cards. The detector handles this automatically.

You can also call the resolver directly:

```python
from pyptx import detect_arch

print(detect_arch())   # e.g. "sm_80" on A100, "sm_90a" on H100,
                       # "sm_120" on RTX Pro 6000 Blackwell
```

`detect_arch()` first asks `torch.cuda.get_device_capability(0)`, then
falls back to `cuda-python`. If neither is available it raises
`RuntimeError` — `arch="auto"` is therefore not the right choice for
build environments without a GPU; pick a concrete `sm_*` instead.

### What runs on what

| Card | arch | Elementwise (RMSNorm/LayerNorm/SwiGLU) | Ampere GEMMs (`cp.async` + bf16) | Hopper kernels (`wgmma`/TMA) | Blackwell kernels (`tcgen05`) |
| --- | --- | --- | --- | --- | --- |
| T4 | sm_75 | ✓ | ✗ (no `cp.async`/bf16) | ✗ | ✗ |
| A100 | sm_80 | ✓ | ✓ | ✗ | ✗ |
| L4 / RTX 40xx | sm_89 | ✓ | ✓ | ✗ | ✗ |
| H100 / H200 | sm_90a | ✓ | ✓ | ✓ | ✗ |
| B200 / GB200 | sm_100a | ✓ | ✓ | ✓ | ✓ |
| RTX Pro 6000 / RTX 50xx | sm_120 | ✓ | ✓ | ✗ | ✗ |

## 2. `ptx.mma.sync(...)` — Ampere tensor-core MMA

The Ampere tensor-core entry point is `mma.sync.aligned`, which `pyptx`
exposes through the typed wrapper `ptx.mma.sync(...)`. It's the
Ampere-equivalent of `ptx.wgmma.mma_async` (Hopper) and
`ptx.tcgen05.mma` (Blackwell):

```python
from pyptx import reg, ptx
from pyptx.types import b32, bf16, f32

a = reg.array(b32, 4)   # 4 packed bf16 register pairs (8 bf16 values)
b = reg.array(b32, 2)   # 2 packed bf16 register pairs (4 bf16 values)
d = reg.array(f32, 4)   # 4 f32 accumulators

ptx.mma.sync(
    shape=(16, 8, 16),                                   # m16n8k16
    dtype_d=f32, dtype_a=bf16, dtype_b=bf16, dtype_c=f32,
    d=[d[0], d[1], d[2], d[3]],
    a=[a[0], a[1], a[2], a[3]],
    b=[b[0], b[1]],
    c=[d[0], d[1], d[2], d[3]],   # accumulate into d in place
    a_layout="row", b_layout="col",                       # default
)
```

Emits exactly:

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    {%fd0, %fd1, %fd2, %fd3},
    {%ba0, %ba1, %ba2, %ba3},
    {%bb0, %bb1},
    {%fd0, %fd1, %fd2, %fd3};
```

Supported dtype combos cover the canonical Ampere set: bf16/bf16→f32,
f16/f16→f32, f16/f16→f16, tf32/tf32→f32, s8/s8→s32. Bad layout strings
(anything that isn't `"row"` or `"col"`) raise `ValueError` at trace
time. The fragment-layout indexing (group_id / t_in_group) is the same
math you'd write by hand against the PTX ISA section on m16n8k16
fragments.

## 3. `ptx.cp.async_` — async global → SMEM staging

Ampere's other tentpole feature is `cp.async`: a non-blocking
global-to-shared-memory copy that overlaps with compute. `pyptx`
exposes the four pieces you need:

```python
from pyptx import ptx

# Issue a 16-byte async copy (cache-global) from global to SMEM.
ptx.cp.async_.cg(ptx.addr(smem_dst), ptx.addr(global_src), 16)

# Or 4/8/16-byte cache-all variant.
ptx.cp.async_.ca(ptx.addr(smem_dst), ptx.addr(global_src), 16)

# Close the in-flight copies into a "group".
ptx.cp.async_.commit_group()

# Wait until at most N groups remain pending.
ptx.cp.async_.wait_group(0)

# Or wait for all in-flight copies.
ptx.cp.async_.wait_all()
```

`commit_group()` / `wait_group(N)` is the standard Ampere prefetch
pipeline: issue stage `s+1` while compute drains stage `s`, then
`wait_group(STAGES - 1)` before reading from stage `s`. The pipeline
drain pattern (last `STAGES` iterations need a different `N`) is
exactly the same as on Hopper — see `examples/ampere/gemm_pipelined.py`
for a worked end-to-end example.

## 4. Examples

### `examples/ampere/gemm.py` — minimal mma.sync GEMM

Single warp per CTA, BM=16, BN=8, BK=16, no SMEM staging — direct
`ld.global.b32` of A and B fragments, one `mma.sync` per K-tile. About
160 lines including docstrings. The clearest reference for the
m16n8k16 fragment-layout math.

### `examples/ampere/gemm_pipelined.py` — first-cut pipelined A100 GEMM

BM=64, BN=64, BK=16, 4 warps/CTA, 2-stage `cp.async` ring buffer,
per-thread `ld.shared.b32` for SMEM→register loads. The
`ptx.bar.sync(0)` between mma and prefetch is essential — without it,
fast warps overwrite SMEM that slow warps are still reading from.
Bit-exact through 4096³, ~64 TFLOPS at 4096³ bf16.

### `examples/ampere/gemm_highperf_ampere.py` — production A100 GEMM

Follows the CUTLASS SM80 + MatmulTutorial v15 design pattern:

- **128×128×32 CTA tile**, 4 warps arranged 2×2 in (M, N), each warp
  owning a 64×64 output sub-tile.
- Per warp per K-iter: **64 `mma.sync.m16n8k16`** (4 M-frags × 8
  N-frags × 2 K-blocks). 256 mma per CTA per K-iter.
- **`ldmatrix.sync.aligned.m8n8.x4.shared.b16`** loads each 16×16 A
  fragment in one warp-collective instruction (4 b32/lane = the full
  m16n8k16 A fragment). Same instruction (without `.trans`) loads two
  16K×8N B fragments at a time — the SMEM `B_T` (N, K) row-major
  layout matches mma's row.col B per-thread layout directly, no
  transpose needed.
- **4-stage `cp.async` ring buffer** (3 in-flight,
  `wait_group(STAGES-2)`). The extra slot vs 3 stages lets one
  `cp.async` overlap with the wait without serializing.
- **Register fragment double-buffering** (CUTLASS / MatmulTutorial v15
  ping-pong): two A and B register banks alternate per K-block.
  At the end of iter `ki`, the next iter's first K-block fragments
  are pre-loaded into bank 0 while bank 1's mma is running — so the
  first mma of iter `ki+1` has zero ldmatrix latency.
- **CUTLASS XOR swizzle** on every SMEM path (`atom_eff = atom ^
  (row & 3)`, equivalently `col_eff = col ^ ((row & 3) << 3)` in bf16
  units). Eliminates the 4-way bank conflict on `ldmatrix.x4` reads
  (8 consecutive M-rows in 64-byte SMEM rows would otherwise touch
  only 2 distinct bank groups). The same formula is applied at
  cp.async writes and ldmatrix reads — they MUST match or data lands
  in the wrong bank.
- **Serpentine N-fragment order** (`n = (mf & 1) ? N-1-nf : nf`) so
  adjacent `mma.sync` calls share one operand register — better
  register cache reuse, fewer ldmatrix→mma stalls.
- **Per-thread SMEM/global offsets hoisted out of the K-loop**. Each
  inner ldmatrix is `addr = stage_base + precomputed_off[mf][kb]`
  (one `add`) instead of 5+ PTX ops per call. With 16 ldmatrix per
  K-iter × 128 K-iters at 4096³, this single change unlocked the
  swizzle's bank-conflict savings (which were drowned out by inline
  XOR arithmetic before hoisting).

**162 TFLOPS at 4096³ bf16** = **73% of cuBLAS** (223 TFLOPS), **2.5×
the simpler `gemm_pipelined.py`**.

References:
- [NVIDIA/cutlass `default_gemm_configuration.h`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/device/default_gemm_configuration.h)
  — SM80 kStages=3, ThreadblockShape 128×128/256, WarpShape 64×64.
- [NVIDIA/cutlass `examples/cute/tutorial/sgemm_sm80.cu`](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu)
  — `SM75_U32x4_LDSM_N`, `Swizzle<3,3,3>`, `SM80_CP_ASYNC_CACHEALWAYS<uint128_t>`.
- [KnowingNothing/MatmulTutorial v15](https://github.com/KnowingNothing/MatmulTutorial/blob/main/examples/matmul/this-sm80/matmul-v15.cu)
  — the worklog this kernel ports techniques from. v15 uses 4 SMEM
  buffers + register frag double-buffer + XOR swizzle + serpentine.
- [Simon Boehm — How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
  — covers the underlying tile-blocking, warp-tiling, and double-buffer
  principles (fp32, no tensor cores).

### `examples/ampere/{rms_norm,layer_norm,swiglu,softmax}.py`

Thin wrappers that build the maintained `examples/hopper/*.py` kernels
with `arch="sm_80"`. Every instruction the Hopper kernels use
(`ld.global.v4.f32`, `fma.rn.f32`, `rsqrt.approx.f32`,
`shfl.sync.bfly.b32`, `mbarrier.*`) is available on sm_80, so the
re-targeting is a pure arch swap.

## A note on TMA, WGMMA, mbarrier multicast, cluster launch

These are Hopper-only. If you target `sm_80` and try to emit
`wgmma.mma_async`, `cp.async.bulk.tensor.*`, `mbarrier.arrive.expect_tx`,
or `cluster.sync`, the spec validator will warn at trace time and the
PTX will fail at `ptxas`. Keep those calls inside arch-conditioned
branches, or split them into `examples/hopper/` and `examples/ampere/`
files (the pattern this repo uses).

## Reproducing the A100 perf numbers

```bash
python benchmarks/bench_ampere_kernels.py            # all kernels
python benchmarks/bench_ampere_kernels.py rms gemm   # subset
```

Numbers in the README perf table are from a single A100 80GB PCIe with
`torch==2.4.1+cu124`. Memory-bound kernels hit ~60–70% of HBM peak at
large sizes. The high-perf GEMM
(`examples/ampere/gemm_highperf_ampere.py`) reaches **162 TFLOPS at
4096³ bf16** — **73% of cuBLAS** (223 TFLOPS), 52% of A100 bf16 peak
(312 TFLOPS), and **2.5×** the simpler `gemm_pipelined.py` baseline.
We haven't spent much time tuning this kernel — the remaining gap is
addressable (persistent / stream-K scheduling, more aggressive
instruction-level overlap, autotuned tile sizes). The current state
shows the full Ampere ISA path end-to-end in editable Python.
