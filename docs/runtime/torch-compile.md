# torch.compile

`pyptx` has a `torch.compile`-compatible path.

The current implementation uses `torch.library.custom_op` plus a fake/meta implementation so Dynamo can:

- keep the kernel call in the graph
- infer output shapes during tracing
- avoid breaking on the real CUDA launch path

## What It Looks Like

The authoring style stays the same. The only difference is the runtime wrapper:

```python
import torch

from pyptx import Tile, kernel, ptx, reg
from pyptx.types import f32, u32

WARP_SIZE = 32


def build_rms_norm(B: int, N: int, eps: float = 1e-6):
    items_per_thread = N // WARP_SIZE

    @kernel(
        in_specs=(Tile(B, N, f32), Tile(N, f32)),
        out_specs=(Tile(B, N, f32),),
        grid=(B, 1, 1),
        block=(WARP_SIZE, 1, 1),
        arch="sm_90a",
    )
    def rms_norm(X, W, Y):
        px, pw, py = ptx.global_ptrs(X, W, Y)

        row = reg.from_(ptx.special.ctaid.x(), u32)
        row_byte_off = row * (N * 4)
        px += row_byte_off
        py += row_byte_off

        tid = reg.from_(ptx.special.tid.x(), u32)
        sum_sq = reg.scalar(f32, init=0.0)
        x_vals = reg.array(f32, items_per_thread)

        for i in range(items_per_thread):
            idx = tid + i * WARP_SIZE
            ptr = px + idx * 4
            ptx.inst.ld.global_.f32(x_vals[i], ptx.addr(ptr))
            ptx.inst.fma.rn.f32(sum_sq, x_vals[i], x_vals[i], sum_sq)

        ptx.warp.reduce_sum(sum_sq)

        mean_sq = reg.scalar(f32)
        inv_n = reg.scalar(f32, init=1.0 / N)
        eps_reg = reg.scalar(f32, init=eps)
        rstd = reg.scalar(f32)
        ptx.inst.mul.f32(mean_sq, sum_sq, inv_n)
        ptx.inst.add.f32(mean_sq, mean_sq, eps_reg)
        ptx.inst.rsqrt.approx.f32(rstd, mean_sq)

        for i in range(items_per_thread):
            idx = tid + i * WARP_SIZE
            off = idx * 4
            w_val = reg.scalar(f32)
            y_val = reg.scalar(f32)
            ptx.inst.ld.global_.f32(w_val, ptx.addr(pw + off))
            ptx.inst.mul.f32(y_val, x_vals[i], rstd)
            ptx.inst.mul.f32(y_val, y_val, w_val)
            ptx.inst.st.global_.f32(ptx.addr(py + off), y_val)
        ptx.ret()

    return rms_norm


k = build_rms_norm(4, 64)

@torch.compile
def fn(x, w):
    return k(x, w)


x = torch.randn(4, 64, device="cuda") * 0.3
w = torch.randn(64, device="cuda") * 0.1 + 1.0
out = fn(x, w)
```

That is the intended user experience: the same `pyptx` kernel object works in eager mode and in compiled Torch code.

## Current Model

- eager execution launches through the raw shim path
- compiled execution wraps the same runtime call in a registered custom op
- fake execution returns correctly shaped tensors without touching the GPU

## Why The Fake Implementation Matters

The fake/meta implementation is what lets Dynamo keep the kernel call in the graph without needing to execute the actual CUDA launch during tracing. In other words:

- real execution path: launch the kernel
- compile-time fake path: only produce correctly shaped outputs

That is enough for `torch.compile` to treat the kernel as an opaque runtime op instead of graph-breaking on it.

This keeps the runtime integration relatively small while making compiled Torch workflows possible.

## CUDA Graphs

pyptx kernels also work with `torch.cuda.CUDAGraph` for minimal-overhead replay (~4 µs per call):

```python
k = build_rms_norm(256, 4096)
x = torch.randn(256, 4096, device="cuda")
w = torch.randn(4096, device="cuda")

k(x, w)  # warm up

g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    k(x, w)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g, stream=s):
    out = k(x, w)

g.replay()  # ~4 µs
```

This is the fastest dispatch path and works with all pyptx kernels.

## Scope

What works:

- tracing through `torch.compile`
- opaque custom-op execution
- shape inference via the fake implementation
- CUDA graph capture and replay

What is still follow-up work:

- deeper compiler integration beyond opaque custom ops

The current behavior is covered by the `TestTorchCompile` cases in `tests/test_torch_dispatch.py`.
