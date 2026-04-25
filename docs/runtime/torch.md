# Torch Runtime

`pyptx` supports a direct PyTorch runtime path alongside JAX.

At call time, `Kernel.__call__` detects `torch.Tensor` inputs and dispatches into the Torch integration layer instead of the JAX FFI path.

## What It Looks Like

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


k = build_rms_norm(16, 512)
x = torch.randn(16, 512, device="cuda") * 0.3
w = torch.randn(512, device="cuda") * 0.1 + 1.0
out = k(x, w)
```

That call goes through the Torch runtime path automatically because the inputs are `torch.Tensor` objects.

## Dispatch Tiers

pyptx provides three dispatch tiers for PyTorch, from fastest to slowest:

### CUDA Graphs (4 µs)

Capture a pyptx kernel into a CUDA graph for minimal-overhead replay:

```python
k = build_rms_norm(256, 4096)
x = torch.randn(256, 4096, device="cuda")
w = torch.randn(4096, device="cuda")

# Warm up
k(x, w)

# Capture
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    k(x, w)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g, stream=s):
    out = k(x, w)

# Replay — ~4 µs per call
g.replay()
```

CUDA graphs work with all pyptx kernels. The graph captures the kernel launch and replays it with near-zero CPU overhead.

### Turbo Eager (15–20 µs)

On repeated calls with the same input shapes, pyptx caches the full launch configuration and bypasses all Python dispatch overhead. This happens automatically — no user action required.

```python
k = build_rms_norm(256, 4096)
x = torch.randn(256, 4096, device="cuda")
w = torch.randn(4096, device="cuda")

# First call: full dispatch (~40 µs)
out = k(x, w)

# Subsequent calls: turbo cache hit (~15 µs)
out = k(x, w)
```

### C++ Extension (Optional)

When `ninja` is installed, pyptx JIT-compiles a C++ extension (`pyptx/_shim/torch_ext.cpp`) that eliminates ctypes overhead. This is automatic — the extension is built on first use and cached.

The C++ path reduces per-call overhead from ~40 µs (ctypes) to ~10 µs (C++ pybind11). Combined with the turbo cache, this gives the 15–20 µs eager dispatch.

Install ninja to enable: `pip install ninja`

## torch.compile

pyptx kernels work with `torch.compile` through `torch.library.custom_op`:

```python
k = build_rms_norm(256, 4096)
k_compiled = torch.compile(k)

x = torch.randn(256, 4096, device="cuda")
w = torch.randn(4096, device="cuda")
out = k_compiled(x, w)
```

The kernel appears as an opaque custom op in the compiled graph. A fake/meta implementation handles shape inference during tracing.

## Good First Targets

- `examples/hopper/rms_norm.py` — simplest kernel, shows both JAX and Torch paths
- `examples/hopper/grouped_gemm.py` — batched GEMM with wgmma
- `tests/test_torch_dispatch.py` — dispatch path tests

## Autograd

pyptx supports `torch.autograd` via `differentiable_kernel`. You provide a forward kernel and a backward kernel; pyptx handles the `torch.autograd.Function` wrapper:

```python
from pyptx import differentiable_kernel

fwd = build_my_forward(M, N)
bwd = build_my_backward(M, N)

my_op = differentiable_kernel(
    fwd, bwd,
    save_for_backward=[0, 1],  # save input indices 0 and 1 for backward
)

x = torch.randn(M, N, device="cuda", requires_grad=True)
w = torch.randn(N, device="cuda", requires_grad=True)
out = my_op(x, w)
out.sum().backward()  # calls bwd kernel with saved tensors + grad_output
```

The backward kernel receives `(*saved_tensors, *grad_outputs)` and must return one gradient tensor per input.

## Constraints

- CUDA tensors only (no CPU tensors)
- Contiguous inputs expected
