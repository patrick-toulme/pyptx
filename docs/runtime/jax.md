# JAX Runtime

The JAX path is the original runtime integration for `pyptx`.

At a high level:

1. a `@kernel` traces to PTX
2. PTX is compiled to a cubin / driver-loaded function
3. launch metadata is registered with the shim
4. a `jax.ffi.ffi_call` launches the kernel on XLA's CUDA stream

This path is documented in `pyptx/jax_support.py` and exercised by the GPU runtime tests in `tests/test_gpu_execution.py`.

## What It Looks Like

This is a real `pyptx` kernel, not an imported placeholder:

```python
import jax
import jax.numpy as jnp

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

@jax.jit
def fn(x, w):
    return k(x, w)


x = jnp.ones((16, 512), dtype=jnp.float32)
w = jnp.ones((512,), dtype=jnp.float32)
out = fn(x, w)
```

From the user side, the kernel object behaves like a shape-specialized function. The difference is that the body you wrote was real PTX-oriented code, not a tensor expression that later became a kernel.

## What Happens Under The Hood

At a high level:

1. input shapes bind symbolic dimensions from `Tile(...)`
2. the kernel body traces to PTX
3. PTX is compiled and registered with the launch shim
4. a `jax.ffi.ffi_call` is emitted into the lowered computation
5. the actual launch happens on XLA's CUDA stream at runtime

The important point is that this is still your handwritten PTX kernel. JAX is the runtime boundary, not the codegen layer.

## What JAX Is Good For Here

- integrating kernels into JAX pipelines
- testing end-to-end launch behavior through XLA
- shape-specialized kernels driven by JAX arrays

## Good First Targets

The best JAX examples in this repo to read after this page are:

- `examples/hopper/rms_norm.py`
- `examples/hopper/layer_norm.py`
- Experimental flash attention kernels live under `examples/hopper/experimental/`

## Important Constraint

The JAX path is a runtime integration. It does not change the PTX authoring model. You still write the kernel with `reg`, `smem`, and `ptx`.
