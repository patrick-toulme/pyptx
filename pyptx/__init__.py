"""Top-level public API for :mod:`pyptx`.

`pyptx` is a Python DSL for writing PTX kernels directly while keeping
the underlying PTX model explicit.

The package is organized around a small number of public namespaces:

- ``kernel``: the decorator used to trace a Python function into PTX
- ``ptx``: instruction wrappers, control flow helpers, and PTX-specific
  structured sugar
- ``reg``: register allocation and register-level arithmetic helpers
- ``smem``: shared-memory allocation and addressing helpers
- ``Tile`` / ``Layout``: tensor boundary specs used by ``@kernel``
- ``intrinsic``: low-level escape hatch for non-standard instructions

Typical usage:

```python
from pyptx import kernel, reg, smem, ptx, Tile, Layout
from pyptx.types import bf16, f32, u32

@kernel(arch="sm_90a")
def tiny():
    tid = reg.from_(ptx.special.tid.x(), u32)
    value = tid + 1
    ptx.inst.mov.u32(tid, value)
    ptx.ret()
```

Most API reference pages on ``pyptx.dev`` are generated from the
docstrings in this package, so the module docstrings are intended to
serve as the high-level entry point into each namespace.
"""

__version__ = "0.1.1"

from pyptx.kernel import kernel
from pyptx._intrinsic import intrinsic
from pyptx._arch import detect_arch
from pyptx.specs import Tile, Layout
from pyptx import reg
from pyptx import smem
from pyptx import ptx

def differentiable_kernel(forward_kernel, backward_kernel, **kwargs):
    """Wrap forward + backward pyptx kernels for ``torch.autograd``.

    See :func:`pyptx.torch_support.differentiable_kernel` for full docs.
    """
    from pyptx.torch_support import differentiable_kernel as _dk
    return _dk(forward_kernel, backward_kernel, **kwargs)


__all__ = [
    "kernel", "reg", "smem", "ptx", "intrinsic",
    "Tile", "Layout", "differentiable_kernel", "detect_arch",
]
