# pyptx vs other GPU Python tools

Short version:

| Tool | Abstraction layer | User writes | Hopper WGMMA | Blackwell tcgen05 | Transpiles PTX? |
| --- | --- | --- | --- | --- | --- |
| **Triton** | tile / compiler | tile program | auto (via `tl.dot`) | auto (via `tl.dot`) | no |
| **cuTile (NVIDIA)** | tile / compiler | tile program | auto | auto | no |
| **CuTe DSL (NVIDIA)** | CuTe atoms / layouts | atom-level ops | yes (atom) | yes (atom) | no |
| **Pallas (Mosaic-GPU)** | MLIR primitives | primitive calls | yes (`plgpu.wgmma`) | yes (`plgpu.tcgen05_mma`) | no |
| **cuda-python** | driver API | driver calls (PTX is a string) | N/A (no DSL) | N/A | no |
| **Numba CUDA** | Python subset | Python | no | no | no |
| **pyptx** | raw PTX | one PTX instruction per call | yes | yes | yes (byte-identical) |

The unique cell is the last row: pyptx is the only Python tool where *the function body is the PTX instruction stream*. Every other tool either (a) generates PTX for you from a higher-level description, or (b) hands PTX to the driver as an opaque string.

---

## vs Triton

Triton is the standard answer for Python GPU kernels. You write a high-level
tile program — `tl.load`, `tl.dot`, `tl.store`, block pointers — and the
compiler discovers a schedule, inserts sync, picks tensor-core instructions,
and emits PTX.

On Blackwell, `tl.dot` auto-lowers to `tcgen05.mma` with TMEM allocation,
leader-election, and multi-barrier pipelining handled by the compiler. This
is good if you want the compiler to make those decisions. If you want to
control them — say, a custom warp-specialization pattern, a non-standard
wgmma layout, or an exact TMA multicast schedule — the Triton escape hatch
is `inline_asm_elementwise`, which is elementwise-only and doesn't help
with the collective instructions that dominate Hopper/Blackwell kernels.

Pyptx is the other direction. No compiler, no scheduler:

```python
ptx.wgmma.mma_async(shape=(64, 256, 16), ...)   # emits exactly one WGMMA
ptx.tcgen05.mma.async_2sm(...)                   # emits exactly one tcgen05.mma
```

- Reach for **Triton** when you want autotuning and compiler heuristics
  to find a good schedule for you.
- Reach for **pyptx** when you want explicit control over WGMMA patterns,
  warp specialization, cluster launch, or when you're transpiling existing
  PTX into editable Python.

Triton PTX output is a valid pyptx transpile target — `python -m pyptx.codegen kernel.ptx --sugar`
round-trips any Triton-emitted kernel to editable Python.

## vs cuTile (NVIDIA)

cuTile is NVIDIA's own tile-level Python DSL, shipped in CUDA 13.1 (late 2025).
Philosophically similar to Triton: you write tile operations (`ct.load`,
`ct.matmul`, reductions) and the compiler generates PTX — including
tcgen05, TMEM management, leader-election, and multi-barrier pipelining.
It's the highest-level answer in this comparison and targets the same
workflow as Triton but with first-party NVIDIA backing.

Pyptx and cuTile are orthogonal. cuTile is for "I want tile programming
to feel like NumPy and the compiler should figure out the rest." Pyptx is
for "I need to write exactly this PTX instruction sequence."

- Reach for **cuTile** when you want the compiler to handle
  tcgen05 / TMEM / leader election for you.
- Reach for **pyptx** when you want those decisions visible and editable.

## vs CuTe DSL (NVIDIA CUTLASS 4)

[CuTe DSL](https://pypi.org/project/nvidia-cutlass-dsl/) is the Python
interface to CUTLASS's CuTe concepts — layouts, tensors, hardware atoms,
and thread/data hierarchies. It's a low-level programming model that's
philosophically closest to pyptx: both give you direct hardware access
in Python, and both cover Blackwell including `tcgen05.mma` and
`cta_group::2` 2-SM MMA.

The difference is the layer:

- **CuTe DSL** is built on CuTe abstractions. You compose atoms
  (hardware-mapped operations with layout metadata). The DSL enforces
  layout correctness and lets the compiler reason about fragments.
- **pyptx** is one layer lower. Each call emits one PTX instruction,
  no layout abstraction between you and the ISA. `ptx.inst.*` is an
  escape hatch for any PTX instruction — current or future — without
  DSL changes.

That also affects the round-trip story. Pyptx has a PTX parser + emitter
that round-trips byte-identical on a corpus of 218+ real kernels; it
can transpile nvcc/Triton/Pallas/DeepGEMM output directly into editable
Python. CuTe DSL doesn't consume PTX as input.

- Reach for **CuTe DSL** when you want NVIDIA's canonical atom
  abstractions and a production path to CUTLASS kernels.
- Reach for **pyptx** when you want PTX as the notation and the ability
  to read existing PTX kernels as editable Python.

## vs Pallas (Mosaic-GPU)

Pallas is a kernel extension mechanism originally from JAX, callable
from both JAX (via an XLA `CustomCall` wrapper) and PyTorch (via a
direct path that doesn't involve XLA at all). Its GPU backend
(Mosaic-GPU) covers Hopper and Blackwell with genuine user-callable
primitives:
`plgpu.wgmma`, `plgpu.tcgen05_mma`, `plgpu.async_load_tmem` /
`async_store_tmem` / `wait_load_tmem`, TMA primitives, and barrier
support with `orders_tensor_core=True` for tensor-core sync. It's a
serious GPU kernel DSL in its own right.

Pallas and pyptx sit at different layers:

- **Pallas / Mosaic-GPU**: you call DSL primitives. The Mosaic lowering
  handles MLIR → LLVM → PTX. Instruction-level decisions (scheduling,
  register allocation, barrier insertion) belong to the compiler.
  Pallas also exposes `plgpu.inline_mgpu`, which drops to the MLIR
  layer and lets you put inline PTX directly into a kernel — so you're
  not strictly limited to the high-level primitive set.
- **pyptx**: you call PTX instructions directly from Python. No MLIR
  layer, no LLVM, no compiler between you and the emitted PTX text.
  One call = one instruction. `ptx.inst.*` makes any PTX instruction
  callable.

Other practical differences:

- Both Pallas and pyptx are callable from JAX *and* PyTorch — Pallas
  via an XLA `CustomCall` when invoked from `jax.jit`, and a separate
  torch path; pyptx via a typed XLA FFI handler (itself a `CustomCall`
  variant) for JAX, and a ctypes / C++ extension path for PyTorch.
- Under the hood both ultimately load the compiled kernel via
  `cuModuleLoadData` — that's the same driver API in either case.
- Pyptx has a PTX transpiler (PTX → editable Python, byte-identical
  round-trip on 218+ corpus kernels); Pallas doesn't consume PTX as
  input.

- Reach for **Pallas** when you want the Mosaic-GPU abstraction, the
  primitives it offers cover most of what you need, and you're happy
  to drop to `inline_mgpu` for the rest.
- Reach for **pyptx** when you want raw PTX visible at the Python
  call site without an MLIR layer, or you want to read an existing
  PTX kernel as editable Python.

Pyptx kernels and Pallas kernels can coexist in the same JAX program
— pyptx registers its own typed FFI handler so it dispatches
alongside Pallas's `CustomCall` cleanly.

## vs cuda-python / CUDA Python

[cuda-python](https://nvidia.github.io/cuda-python/) is NVIDIA's official
Python binding for the CUDA driver API. It gives you `cuModuleLoadData`,
`cuLaunchKernel`, `cuTensorMapEncodeTiled`, etc. No DSL, no kernel-side
code — PTX goes in as a string you produced some other way.

pyptx *uses* cuda-python under the hood for the driver calls. The value
pyptx adds is everything above the driver: an instruction DSL
(`ptx.inst.*`), a parser and emitter, a trace system, a transpiler,
and JAX/PyTorch runtime bindings.

- Reach for **cuda-python** when you already have PTX (from CUTLASS,
  nvcc, inline asm) and you just need to launch it.
- Reach for **pyptx** when you want to *write* that PTX from Python,
  or read an existing PTX kernel and modify it.

## vs Numba CUDA

Numba CUDA JITs a Python subset to PTX. It's useful for simple kernels
where idiomatic Python is what you want to write.

Numba does not expose WGMMA, TMA, mbarriers, cluster launch, tcgen05,
or TMEM. Hopper-era and Blackwell-era features are unreachable through
the Numba front-end; the escape hatch is inline PTX strings inside a
Numba kernel. At that point you're writing PTX anyway — and pyptx gives
you that as a first-class, typed DSL with a parser, emitter, and
transpiler.

- Reach for **Numba** when you have idiomatic Python to accelerate on
  GPU and you don't need Hopper/Blackwell ISA features.
- Reach for **pyptx** when you need any of those ISA features, or you
  want PTX transpilation.

---

## When pyptx is the wrong answer

- You want one kernel that runs on every NVIDIA GPU generation. Pyptx
  targets Hopper (sm_90a) and Blackwell (sm_100a) specifically.
- You want the compiler to pick the schedule. Triton and cuTile will
  give you more perf per hour of engineering effort on standard
  patterns.
- You want a production GEMM/conv matrix of variants generated for
  you. Use CuTe DSL + CUTLASS, or cuBLAS directly.

## When pyptx is the right answer

- You're writing a Hopper kernel that needs WGMMA, TMA 3D multicast,
  mbarrier phase tracking, or cluster launch explicitly — and you want
  the exact instruction sequence visible in Python.
- You're writing a **Blackwell** kernel that needs `tcgen05.mma`,
  TMEM alloc/ld/store, `cta_group::2` cooperative MMA, or the SMEM +
  instruction descriptors, and you want per-instruction control rather
  than a compiler primitive.
- You're porting an existing hand-tuned PTX kernel (nvcc, Triton,
  CUTLASS, DeepGEMM) into editable Python via the transpiler.
- You're teaching PTX or studying real kernels, and you want Python as
  a notation instead of C++ templates.
- You're experimenting with instruction scheduling, warp
  specialization, or fragment-layout manipulations that a compiler
  would otherwise reorder away.
