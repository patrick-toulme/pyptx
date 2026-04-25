# Getting Started

The fastest way to understand `pyptx` is:

1. read a small kernel
2. inspect the emitted PTX
3. run it through one runtime path

## Mental Model

A `@kernel` function executes in Python at trace time. During that trace:

- `reg.*` allocates PTX registers
- `smem.*` describes shared-memory allocations and addressing
- `ptx.*` emits PTX instructions and structured control flow

The output of that trace is PTX, not Python bytecode and not an intermediate tensor IR.

## Minimal Workflow

```python
from pyptx import kernel, reg, ptx
from pyptx.types import f32, u32

@kernel(arch="sm_90a")
def tiny():
    tid = reg.from_(ptx.special.tid.x(), u32)
    x = tid + 1
    ptx.inst.mov.u32(tid, x)
    ptx.ret()
```

Then inspect the PTX:

```python
print(tiny.ptx())
```

That PTX is the real artifact. The Python function is just the authoring surface.

## A Slightly More Real Example

The pattern looks more interesting once the kernel is called from a framework runtime:

```python
from examples.hopper.rms_norm import build_rms_norm
import torch

k = build_rms_norm(4, 64)

x = torch.randn(4, 64, device="cuda") * 0.3
w = torch.randn(64, device="cuda") * 0.1 + 1.0
out = k(x, w)
```

The same kernel object can also be used with `torch.compile`:

```python
@torch.compile
def compiled(x, w):
    return k(x, w)
```

or JAX:

```python
import jax
import jax.numpy as jnp

@jax.jit
def compiled(x, w):
    return k(x, w)
```

## Mental Checklist

When reading any `pyptx` kernel, ask:

1. what are the tensor boundary specs?
2. what registers are loop-carried state?
3. what is static Python structure, and what is emitted PTX control flow?
4. what runtime path is launching the kernel?

## Hopper vs Blackwell

`pyptx` targets two architectures:

- **Hopper (`sm_90a`)** — WGMMA, TMA 2D/3D with multicast, mbarriers,
  cluster launch. Kernels live in `examples/hopper/`.
- **Blackwell (`sm_100a`)** — `tcgen05.mma` / `.ld`, TMEM, SMEM and
  instruction descriptors, 2-SM cooperative MMA via `cta_group::2`,
  TMA multicast. Kernels live in `examples/blackwell/`.

Pick the target with `arch="sm_90a"` or `arch="sm_100a"` in the
`@kernel` decorator. For a B200, start with
`examples/blackwell/tcgen05_suite.py` — it exercises every Blackwell
primitive (alloc / MMA / ld / commit / fence) in isolation — then
`examples/blackwell/gemm_highperf_blackwell.py` for the 1+ PFLOP 1SM
GEMM.

## What To Read Next

- [First Kernel](guides/first-kernel.md) for the basic authoring pattern
- [JAX Runtime](runtime/jax.md) if you want to call kernels from `jax.jit`
- [Torch Runtime](runtime/torch.md) if you want PyTorch or `torch.compile`
- [Examples](examples/index.md) for real kernels
