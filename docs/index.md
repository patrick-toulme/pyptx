<p align="center" markdown>
  ![pyptx](assets/pyptx-logo.png){ width="520" }
</p>

# pyptx

**Write PTX kernels in Python. Launch them from JAX, PyTorch, and `torch.compile`.**

=== "PyTorch"

    ```bash
    pip install 'pyptx[torch]'
    ```

=== "JAX"

    ```bash
    pip install 'pyptx[jax]'
    ```

=== "Both"

    ```bash
    pip install 'pyptx[all]'
    ```

=== "DSL only (no GPU runtime)"

    ```bash
    pip install pyptx
    ```

Base `pyptx` has **zero required dependencies** — it ships the DSL,
tracer, parser, emitter, and transpiler. The `[torch]` / `[jax]`
extras pull in the framework you want to launch kernels from.

---

## What is pyptx?

`pyptx` is a Python DSL for handwritten PTX on NVIDIA Hopper (sm_90a) and
Blackwell (sm_100a).

The idea is simple: the Python function **is** the PTX stream. Each DSL call
emits exactly one PTX instruction — no optimizer, no autotuner, no hidden
codegen.

- **One call = one instruction.** `ptx.wgmma.mma_async(...)` emits exactly
  one WGMMA. `ptx.tcgen05.mma(...)` emits exactly one Blackwell UMMA.
- **Hopper.** WGMMA, TMA (2D/3D with multicast), mbarriers, cluster launch,
  `stmatrix` — the instructions Triton/Pallas don't expose.
- **Blackwell.** `tcgen05.mma`, TMEM alloc / ld / st, SMEM and instruction
  descriptors, warp specialization. Same DSL — the only Python library
  that speaks tcgen05 directly.
- **Real runtime integration.** Kernels are callable from `torch.compile`,
  PyTorch eager, and `jax.jit` through a typed FFI.
- **Python all the way down.** ~150 lines of C++ for the launch shim.
  Everything else — DSL, tracer, parser, emitter, transpiler — is pure Python.

---

## A kernel, start to finish

Here's a fused RMS-norm kernel. One CTA per row; threads cooperatively sum
squares with a butterfly-shuffle reduction, then rescale and write back. Every
`ptx.inst.*` call below emits exactly one PTX instruction.

```python
from pyptx import kernel, reg, smem, ptx, Tile
from pyptx.types import f32, u32

@kernel(
    in_specs=(Tile("B", "N", f32), Tile("N", f32)),   # X[B, N], W[N]
    out_specs=(Tile("B", "N", f32),),                 # Y[B, N]
    grid=lambda B, N: (B, 1, 1),
    block=(128, 1, 1),
    arch="sm_90a",
)
def rms_norm(X, W, Y, *, eps: float = 1e-6):
    partials = smem.alloc(f32, (4, 1))                # warp-partial sums
    px, pw, py = ptx.global_ptrs(X, W, Y)             # three param ptrs at once
    tid = reg.scalar(u32); ptx.inst.mov.u32(tid, ptx.special.tid.x())
    row = reg.scalar(u32); ptx.inst.mov.u32(row, ptx.special.ctaid.x())
    N = X.shape[1]
    px += row * (N * 4); py += row * (N * 4)

    # Pass 1: v4 loads, accumulate sum-of-squares per thread.
    sum_sq = reg.scalar(f32, init=0.0)
    x_vals = reg.array(f32, N // 128)
    for j in range(N // 512):
        off = (tid << 4) + j * (128 * 16)             # 4 elems * 4 bytes per thread
        ptx.inst.ld.global_.v4.f32(
            [x_vals[j*4+k] for k in range(4)],
            ptx.addr(px + off),
        )
        for k in range(4):
            ptx.inst.fma.rn.f32(sum_sq, x_vals[j*4+k], x_vals[j*4+k], sum_sq)

    ptx.warp.reduce_sum(sum_sq)                       # canonical shfl.bfly reduce
    # ... block reduce via SMEM, rsqrt, scale by W, v4-store Y ...
    ptx.ret()
```

**2.6 TB/s on H100** — 88% of HBM3 peak, 3.9× faster than the PyTorch
reference. The full kernel is
[`examples/hopper/rms_norm.py`](examples/hopper/rms_norm.md).

Inspect the emitted PTX at any time:

```python
print(rms_norm.ptx())
```

---

## Call it from any runtime

=== "PyTorch eager"

    ```python
    import torch
    from my_kernels import rms_norm

    x = torch.randn(256, 4096, device="cuda")
    w = torch.randn(4096, device="cuda")
    y = rms_norm(x, w)
    ```

=== "torch.compile"

    ```python
    import torch

    @torch.compile
    def fn(x, w):
        return rms_norm(x, w)

    y = fn(x, w)
    ```

=== "JAX jit"

    ```python
    import jax
    import jax.numpy as jnp

    @jax.jit
    def fn(x, w):
        return rms_norm(x, w)

    y = fn(x, w)        # real FFI call through XLA
    ```

The same kernel object goes through all three paths. Torch dispatch uses a
registered `torch.library.custom_op`; JAX uses a typed FFI handler. Launch
overhead is ~14 µs via a C++ extension, ~4 µs under CUDA graph replay.

---

## Performance

### Blackwell (B200, bf16)

| Kernel | Shape | 1SM TFLOPS | 2SM TFLOPS | cuBLAS | best / cuBLAS |
| --- | --- | --- | --- | --- | --- |
| **GEMM** | 2048³ | 645 | **649** | 1006 | 64% |
| **GEMM** | 4096³ | **1194** | 1168 | 1532 | 78% |
| **GEMM** | 8192³ | **1240** | 1046 | 1610 | 77% |
| **Grouped GEMM** (MoE) | G=4 M=2048 N=256 K=2048 | **401** | — | torch ref | **~10.0×** |

2SM uses `tcgen05.mma.cta_group::2` across a 2-CTA cluster with a
cluster-shared mbarrier hand-off; 1SM uses the single-CTA MMA with a
4-stage pipeline and remains the default maintained path.

### Hopper (H100 SXM5)

| Kernel | Shape | pyptx | vs reference |
| --- | ------ | ----- | ------------ |
| **GEMM** (bf16, WGMMA warp-spec) | 8192³ | 815 TFLOPS | beats cuBLAS ≥6K |
| **Grouped GEMM** (bf16→f32) | G=8 M=K=2048 | 104 TFLOPS | — |
| **RMS norm** (f32) | B=2048 N=8192 | 2.6 TB/s (88% HBM) | **3.9×** torch |
| **Layer norm** (f32) | B=2048 N=8192 | 2.5 TB/s (83% HBM) | **1.5×** torch |
| **SwiGLU** (f32) | M=2048 F=8192 | 2.8 TB/s (94% HBM) | **1.6×** torch |
| **Flash attention** (bf16) | M=N=4096, HD=64 | 88 µs | **3.0×** torch naive |

[**Full numbers →**](performance.md){ .md-button }

---

## Also: a real PTX transpiler

`pyptx` ingests PTX and emits runnable pyptx Python that **round-trips
byte-identical**. Feed it output from `nvcc`, Triton, or Pallas:

```bash
python -m pyptx.codegen kernel.ptx --sugar --name my_kernel > my_kernel.py
```

The `--sugar` pass demangles names, raises spin-loops into `ptx.loop(...)`,
collapses mbarrier-wait blocks, and groups expression chains into
`ptx.expr(...)` blocks. The maintained 815 TFLOPS Hopper GEMM in
`examples/hopper/gemm_highperf_hopper.py` is exactly this workflow applied to
[fast.cu's kernel12](https://github.com/pranjalssh/fast.cu).

[**How the transpiler works →**](transpiler.md){ .md-button }

---

## What pyptx is — and isn't

:material-check: **Is:** a place to write real Hopper + Blackwell kernels
with explicit WGMMA / `tcgen05` / TMA / mbarrier / cluster-launch control,
call them from Python, and stay readable.

:material-check: **Is:** a round-trip target for compiled PTX, so existing
kernels can be ported into editable Python.

:material-close: **Isn't:** an autotuner. No search, no heuristics.
Specialize per shape; the DSL gets out of your way.

:material-close: **Isn't:** a tensor compiler. pyptx doesn't have a
high-level IR. If you want "a compiler to target", use Triton or Pallas.

[**Why pyptx vs Triton, CUTLASS, Pallas →**](comparison.md){ .md-button }

---

## Start here

<div class="grid cards" markdown>

- :material-rocket-launch: **[Getting started](getting-started.md)** — mental model + first kernel
- :material-school: **[First kernel guide](guides/first-kernel.md)** — authoring a real kernel
- :material-format-list-bulleted: **[Examples](examples/index.md)** — RMS norm, SwiGLU, GEMM, FA
- :material-api: **[API reference](api/index.md)** — every namespace, every helper
- :material-speedometer: **[Performance](performance.md)** — H100 + B200 benchmarks
- :material-compare: **[vs Triton/CUTLASS/Pallas](comparison.md)** — when to reach for pyptx

</div>
