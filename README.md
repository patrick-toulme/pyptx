<p align="center">
  <img src="docs/assets/pyptx-logo.png" alt="pyptx" width="520">
</p>

# pyptx

> Write PTX kernels in Python. Launch them from `jax.jit`, PyTorch, and `torch.compile`.

`pyptx` is a Python DSL for handwritten PTX on NVIDIA Hopper (sm_90a) and
Blackwell (sm_100a).

One call = one instruction. No optimizer, no autotuner, no tensor IR between
the Python function and the PTX it emits.

- explicit registers, predicates, barriers, shared memory
- Hopper: WGMMA, TMA 2D/3D with multicast, mbarriers, cluster launch
- Blackwell: `tcgen05.mma` / `.ld`, TMEM, SMEM descriptors, warp specialization
- callable from JAX, PyTorch eager, and `torch.compile`
- real PTX parser + emitter + **transpiler** — round-trips 218+ real PTX files byte-identical

Docs: [pyptx.dev](https://pyptx.dev) · Examples:
[`examples/hopper/`](examples/hopper),
[`examples/blackwell/`](examples/blackwell) ·
API: [pyptx.dev/api](https://pyptx.dev/api/)

---

## Install

| Command | What you get |
| --- | --- |
| `pip install pyptx` | DSL, parser, emitter, transpiler (no GPU runtime) |
| `pip install 'pyptx[torch]'` | + PyTorch eager and `torch.compile` launch path |
| `pip install 'pyptx[jax]'` | + `jax.jit` launch path via typed FFI |
| `pip install 'pyptx[all]'` | + both PyTorch and JAX |

Tip: `pip install ninja` so the PyTorch C++ extension JIT-builds on first
launch (drops dispatch overhead from ~34 µs to ~14 µs).

## Performance

### Blackwell (B200, bf16)

| Kernel | Shape | pyptx | cuBLAS | best / cuBLAS |
| --- | --- | --- | --- | --- |
| **GEMM** (`tcgen05.mma`, 4-stage pipeline, 1SM) | 8192³ | **1240 TFLOPS** | 1610 | 77% |
| **GEMM** (1SM) | 4096³ | **1194 TFLOPS** | 1532 | 78% |
| **GEMM 2SM** (`cta_group::2`, 5-stage) | 2048³ | **649 TFLOPS** (beats 1SM) | 1006 | 64% |
| **Grouped GEMM** (tcgen05, MoE) | G=4 M=2048 N=256 K=2048 | **401 TFLOPS** | torch ref | **~10.0×** |
| **RMS norm / Layer norm / SwiGLU** | maintained Blackwell ports | benchmarked | torch ref | see kernel suite |

### Hopper (H100 SXM5, bf16 / f32)

| Kernel | Shape | pyptx | vs reference |
| --- | --- | --- | --- |
| **GEMM** (wgmma, warp-specialized) | 8192³ | **815 TFLOPS** | beats cuBLAS ≥ 6K |
| **Grouped GEMM** (bf16→f32) | G=8 M=K=2048 | **104 TFLOPS** | — |
| **RMS norm** (f32) | B=2048 N=8192 | 2.6 TB/s (88% HBM) | **3.9×** torch |
| **Layer norm** (f32) | B=2048 N=8192 | 2.5 TB/s (83% HBM) | **1.5×** `F.layer_norm` |
| **SwiGLU** (f32) | M=2048 F=8192 | 2.8 TB/s (94% HBM) | **1.6×** `F.silu(g)*u` |
| **Softmax** (f32, row-wise) | B=2048 N=8192 | 2.8 TB/s (95% HBM) | **1.16×** `torch.softmax` |
| **Flash attention** (bf16) | M=N=4096, HD=64 | 88 µs | **3.0×** naive torch |

Full benchmark tables + reproduction commands:
[pyptx.dev/performance](https://pyptx.dev/performance/).

PyTorch dispatch tiers:

- **CUDA graph replay**: ~4 µs per launch
- **Turbo eager**: ~14 µs (cached C++ extension)
- **`torch.compile`**: ~14–22 µs (custom_op path)

---

## What it looks like

```python
from pyptx import kernel, reg, smem, ptx, Tile
from pyptx.types import bf16, f32

@kernel(
    in_specs=(Tile("M", "K", bf16), Tile("K", "N", bf16)),
    out_specs=(Tile("M", "N", f32),),
    grid=lambda M, N, K: (N // 64, M // 64),
    block=(128, 1, 1),
    arch="sm_90a",
)
def gemm(A, B, C):
    sA = smem.wgmma_tile(bf16, (64, 16), major="K")
    sB = smem.wgmma_tile(bf16, (16, 64), major="MN")
    acc = reg.array(f32, 32)
    # ... TMA loads + ptx.wgmma.mma_async(...) — each call emits exactly one PTX instruction
```

Every `ptx.*` call is a single PTX instruction. `print(gemm.ptx())` shows
exactly what you wrote.

## One kernel, three runtime paths

The same kernel object works in JAX, PyTorch eager, and `torch.compile`:

```python
# PyTorch eager
out = gemm(a, b)

# torch.compile
out = torch.compile(gemm)(a, b)

# JAX jit (lowers through typed FFI)
out = jax.jit(gemm)(a, b)
```

Under the hood the PTX is JITed through `cuModuleLoadData`, registered
with a ~150-line C++ launch shim, and dispatched from PyTorch via
`torch.library.custom_op` or from JAX via `jax.ffi.ffi_call`.

---

## Transpile existing PTX into pyptx

`pyptx` is also a real PTX-to-Python transpiler. Feed it output from
`nvcc`, Triton, Pallas, or any other source:

```bash
python -m pyptx.codegen kernel.ptx --sugar --name my_kernel > my_kernel.py
```

`--sugar` demangles names, raises spin-loops into `ptx.loop(...)`, collapses
mbarrier-wait blocks, and groups expression chains. Round-trips are
**byte-identical** on 218+ corpus files (CUTLASS, Triton, fast.cu, DeepGEMM,
ThunderKittens, LLVM tests).

The **815 TFLOPS** Hopper GEMM in `examples/hopper/gemm_highperf_hopper.py` is
exactly this workflow applied to
[fast.cu's kernel12](https://github.com/pranjalssh/fast.cu).

---

## Start here

Hopper (sm_90a):

- `examples/hopper/rms_norm.py` — simplest real kernel, v4 loads + warp reduce
- `examples/hopper/grouped_gemm.py` — multi-k WGMMA for MoE shapes
- `examples/hopper/gemm_highperf_hopper.py` — warp-specialized 815 TFLOPS GEMM

Blackwell (sm_100a):

- `examples/blackwell/tcgen05_suite.py` — 13 isolated tcgen05 primitives
  (alloc, MMA, ld, commit/fence, GEMM probes). Run this first on a B200
  to verify the runtime stack.
- `examples/blackwell/gemm_highperf_blackwell.py` — `build_gemm`
  (1SM, 4-stage ring buffer, 1.24 PFLOPS at 8192³ bf16) and
  `build_gemm_2sm` (2SM `cta_group::2` cooperative MMA, 5-stage).
- `examples/blackwell/gemm_experimental_blackwell.py` — persistent and
  Pallas-style experimental GEMM paths, plus the no-TMA tcgen05 debug GEMM.
- `examples/blackwell/grouped_gemm.py` — G-problem MoE grouped GEMM on
  top of the same `tcgen05.mma` mainloop, bit-exact against
  `einsum("gmk,gkn->gmn")` through G=8 M=1024 N=128 K=1024.
- `examples/blackwell/rms_norm.py` / `layer_norm.py` / `swiglu.py` —
  Hopper kernels re-targeted to `sm_100a`.
- `benchmarks/bench_blackwell_gemm.py` — reproduce the 1SM + 2SM +
  cuBLAS table above.
- `benchmarks/bench_blackwell_kernels.py` — Blackwell grouped GEMM,
  RMSNorm, LayerNorm, and SwiGLU benchmark suite.

Docs:

- [Getting Started](https://pyptx.dev/getting-started/)
- [Performance](https://pyptx.dev/performance/)
- [Debugging](https://pyptx.dev/guides/debugging/)
- [vs Triton/CUTLASS/Pallas](https://pyptx.dev/comparison/)

## Status

0.1.0, pre-launch. Scope:

- handwritten PTX DSL with full Hopper ISA (wgmma, TMA 2D/3D, mbarriers, cluster)
- Blackwell `tcgen05` ISA (alloc, `mma.kind::f16/tf32/f8`, `ld`/`st`,
  commit, fence) with instruction-descriptor + SMEM-descriptor helpers
- PTX parser / emitter with 218+ corpus round-trip tests
- PTX → Python transpiler with sugar pass
- JAX runtime integration (typed FFI)
- PyTorch eager + `torch.compile` + CUDA graph replay
- C++ dispatch extension for low-overhead launches
- GMMA/UMMA SMEM swizzle helpers (B32 / B64 / B128, CuTe-compatible `Swizzle<B,4,3>`)
- PyTorch autograd via `differentiable_kernel`

## License

Apache-2.0. See [LICENSE](LICENSE).
