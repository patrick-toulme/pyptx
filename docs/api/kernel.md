# `pyptx.kernel`

> This page is generated from source docstrings and public symbols.

Kernel tracing, specialization, and runtime dispatch.

The :func:`kernel` decorator is the main entry point for authoring
``pyptx`` kernels. A decorated Python function is traced into PTX and
can then be:

- inspected as PTX text with ``.ptx(...)``
- launched through the JAX runtime path
- launched through the PyTorch eager path
- launched through the ``torch.compile`` custom-op path

Example:

```python
from pyptx import kernel, Tile, Layout
from pyptx.types import bf16, f32

@kernel(
    in_specs=(Tile("M", "K", bf16, Layout.ROW),
              Tile("K", "N", bf16, Layout.COL)),
    out_specs=(Tile("M", "N", f32, Layout.ROW),),
    grid=lambda M, N, K: (M // 128, N // 256),
    block=(128, 1, 1),
    cluster=(2, 1, 1),
    arch="sm_90a",
)
def gemm(A, B, C, *, BM=128, BN=256, BK=64): ...
```

Key concepts:

- Positional parameters correspond to tensor inputs and outputs.
- Keyword-only parameters act as template parameters and are baked into
  the trace.
- ``Tile`` and ``Layout`` describe the tensor boundary contract.
- The kernel body itself emits PTX by calling into ``reg``, ``smem``,
  and ``ptx``.

Practical workflow:

```python
print(gemm.ptx(M=4096, N=4096, K=4096))
```

and then later:

```python
@jax.jit
def fwd(x, w):
    return gemm(x, w)
```

or:

```python
@torch.compile
def fwd(x, w):
    return gemm(x, w)
```

## Public API

- [`TensorSpec`](#tensorspec)
- [`TmaDescriptorHandle`](#tmadescriptorhandle)
- [`Kernel`](#kernel)
- [`kernel`](#kernel)

<a id="tensorspec"></a>

## `TensorSpec`

- Kind: `class`

```python
class TensorSpec(name: 'str', shape: 'tuple[int, ...] | None' = None, dtype: 'Any' = None, layout: 'Layout | None' = None) -> 'None'
```

Placeholder for a tensor argument at trace time.

Carries the parameter name plus (if known) shape and dtype information
derived from the input/output specs. At execution time inside jax.jit
these are bound to real JAX arrays.

Methods like ``tma_desc()`` return symbolic handles that get resolved
to real pointers by the FFI launcher at kernel launch time.

### Members

#### `tma_desc()`

- Kind: `method`

Return a TMA descriptor reference for this tensor.

Used inside a kernel to pass the tensor to a TMA load/store:

    ptx.cp.async_.bulk.tensor_2d(
        dst=sA[0],
        src=A.tma_desc(),
        coord=(x, y),
        mbar=bar[0],
    )

Inside an active kernel trace this function:
  1. Records ``self.name`` on the trace context so the driver
     knows to append a ``.param .u64 <name>_tma_desc`` slot to
     the emitted entry signature and to synthesize a real TMA
     descriptor at compile time.
  2. Emits an ``ld.param.u64`` prologue (once per tensor) that
     loads the descriptor pointer into a fresh register.
  3. Returns that register so it can be used directly as the
     ``src`` of a ``cp.async.bulk.tensor.*`` instruction.

Outside a trace (e.g. in unit tests that probe the TensorSpec
API without entering a kernel), this returns a ``TmaDescriptorHandle``
for backwards compatibility.

#### `dtype`

- Kind: `attribute`

- Value: `<member 'dtype' of 'TensorSpec' objects>`

No docstring yet.

#### `layout`

- Kind: `attribute`

- Value: `<member 'layout' of 'TensorSpec' objects>`

No docstring yet.

#### `name`

- Kind: `attribute`

- Value: `<member 'name' of 'TensorSpec' objects>`

No docstring yet.

#### `shape`

- Kind: `attribute`

- Value: `<member 'shape' of 'TensorSpec' objects>`

No docstring yet.

<a id="tmadescriptorhandle"></a>

## `TmaDescriptorHandle`

- Kind: `class`

```python
class TmaDescriptorHandle(tensor: 'TensorSpec') -> 'None'
```

Symbolic handle for a TMA descriptor.

Carries a reference back to the TensorSpec so the FFI launcher can
build the real cuTensorMap at runtime from the JAX array metadata.
In the emitted PTX it's rendered as the symbolic name (e.g. ``A_desc``).

### Members

#### `name`

- Kind: `attribute`

- Value: `<member 'name' of 'TmaDescriptorHandle' objects>`

No docstring yet.

#### `tensor`

- Kind: `attribute`

- Value: `<member 'tensor' of 'TmaDescriptorHandle' objects>`

No docstring yet.

<a id="kernel"></a>

## `Kernel`

- Kind: `class`

```python
class Kernel(fn: 'Callable', arch: 'str' = 'sm_90a', version: 'tuple[int, int] | None' = None, in_specs: 'Sequence[Tile] | None' = None, out_specs: 'Sequence[Tile] | None' = None, grid: 'Callable[..., tuple[int, int, int]] | tuple[int, int, int] | None' = None, block: 'tuple[int, int, int]' = (1, 1, 1), cluster: 'tuple[int, int, int]' = (1, 1, 1), smem: 'int' = 0, raw_params: 'Sequence[tuple[str, str]] | None' = None, extern_smem: 'bool | str' = False, reqntid: 'tuple[int, ...] | None' = None, raw_directives: 'Sequence[tuple[str, tuple]] | None' = None) -> 'None'
```

A traced PTX kernel. Wraps a Python function that uses ptx.* calls.

### Members

#### `in_specs`

- Kind: `property`

Input tensor specs declared on the kernel.

#### `out_specs`

- Kind: `property`

Output tensor specs declared on the kernel.

#### `grid`

- Kind: `property`

Configured grid tuple or grid resolver callable.

#### `block`

- Kind: `property`

Static CUDA block dimensions for the kernel.

#### `cluster`

- Kind: `property`

CTA cluster dimensions used at launch time.

#### `smem`

- Kind: `property`

Requested dynamic/shared memory size in bytes.

#### `template_params`

- Kind: `property`

Return the declared template parameters and their default values.

Only keyword-only parameters in the function signature count as
template parameters. Positional args are tensor placeholders, not
template parameters.

#### `arch`

- Kind: `property`

Target PTX architecture string, e.g. ``sm_90a``.

#### `ptx(**kwargs: 'Any') -> 'str'`

- Kind: `method`

Trace and emit PTX text. The inspection API.

Pass template kwargs (BM, BN, BK, etc.) and/or shape variables
(M, N, K, ...) to specialize. Defaults from the function signature
fill in any kwargs you don't supply.

Usage:
    print(my_kernel.ptx(M=4096, N=4096, K=4096, BM=128))

#### `module(**kwargs: 'Any') -> 'Module'`

- Kind: `method`

Trace and return the IR Module (for programmatic inspection).

#### `sass(**kwargs: 'Any') -> 'str'`

- Kind: `method`

Compile PTX to cubin and disassemble to SASS via cuobjdump.

This is the "what actually ran on the GPU" view — useful for
performance tuning and understanding how ptxas lowered your PTX.

Requires the CUDA toolkit to be installed (for ptxas + cuobjdump).
Raises RuntimeError with a helpful message if the toolkit is not
available.

Usage:
    print(my_kernel.sass(M=4096, N=4096, K=4096))

<a id="kernel"></a>

## `kernel`

- Kind: `function`

```python
kernel(fn: 'Callable | None' = None, *, arch: 'str' = 'sm_90a', version: 'tuple[int, int] | None' = None, in_specs: 'Sequence[Tile] | None' = None, out_specs: 'Sequence[Tile] | None' = None, grid: 'Any' = None, block: 'tuple[int, int, int]' = (1, 1, 1), cluster: 'tuple[int, int, int]' = (1, 1, 1), smem: 'int' = 0, raw_params: 'Sequence[tuple[str, str]] | None' = None, extern_smem: 'bool' = False, reqntid: 'tuple[int, ...] | None' = None, raw_directives: 'Sequence[tuple[str, tuple]] | None' = None) -> 'Kernel | Callable[[Callable], Kernel]'
```

Decorator to define a PTX kernel.

Can be used with or without arguments:

    @kernel
    def simple(): ...

    @kernel(arch="sm_100a")
    def blackwell(): ...

    @kernel(
        in_specs=(Tile("M", "K", bf16), Tile("K", "N", bf16)),
        out_specs=(Tile("M", "N", f32),),
        grid=lambda M, N, K: (M // 128, N // 256),
        block=(128, 1, 1),
        cluster=(2, 1, 1),
        arch="sm_90a",
    )
    def gemm(A, B, C, *, BM=128): ...
