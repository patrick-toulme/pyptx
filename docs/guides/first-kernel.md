# First Kernel

The smallest useful `pyptx` workflow is:

1. define a kernel
2. allocate registers
3. emit a few PTX instructions
4. inspect the generated PTX

## Example

```python
from pyptx import kernel, reg, ptx
from pyptx.types import u32

@kernel(arch="sm_90a")
def tiny():
    tid = reg.from_(ptx.special.tid.x(), u32)
    out = tid + 4
    ptx.inst.mov.u32(tid, out)
    ptx.ret()
```

Then inspect it:

```python
print(tiny.ptx())
```

The emitted PTX is small and unsurprising:

```ptx
mov.u32 %r0, %tid.x;
add.s32 %r1, %r0, 4;
mov.u32 %r0, %r1;
ret;
```

## What To Notice

- `reg.from_(...)` is just a convenient way to stage a value into a real PTX register
- `tid + 4` emits arithmetic directly into the trace
- `ptx.ret()` ends the kernel exactly where you write it

This is the core authoring style of `pyptx`: structured Python, explicit PTX model.

## The Three Namespaces

Most kernels are some mix of:

- `reg`: allocate state and use simple register-level arithmetic
- `smem`: allocate shared-memory regions and barriers
- `ptx`: emit instructions, control flow, and low-level GPU operations

If you understand those three namespaces, you understand the shape of the library.

## Next Step

The next thing to read after this page should usually be:

- [PTX Namespace](ptx-namespace.md) to understand the main DSL surface
- [JAX Runtime](../runtime/jax.md) or [Torch Runtime](../runtime/torch.md) to see how kernels are launched
