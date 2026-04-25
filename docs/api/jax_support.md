# `pyptx.jax_support`

> This page is generated from source docstrings and public symbols.

JAX runtime integration for :func:`pyptx.kernel`.

This module owns the JAX/XLA execution path for ``pyptx`` kernels:

1. resolve shapes and template parameters
2. trace the kernel body to PTX
3. compile PTX to a driver-loadable kernel handle
4. register launch metadata with the C++ shim
5. build a ``jax.ffi.ffi_call`` that launches on XLA's CUDA stream

In other words, this module is the bridge between a traced PTX kernel
and an actual ``@jax.jit`` call site.

Important design point:

The C++ shim is intentionally thin. Most of the interesting runtime
logic lives here in Python:

- PTX compilation
- launch metadata registration
- TMA descriptor synthesis
- process-local kernel handle bookkeeping

On machines without the full CUDA/JAX runtime stack, the tracing and
lowering parts still work. That lets codegen and inspection workflows
operate without requiring a live GPU launch environment.

## Public API

- [`CubinRecord`](#cubinrecord)
- [`CubinRegistry`](#cubinregistry)
- [`get_cubin_registry`](#get-cubin-registry)
- [`shim_is_available`](#shim-is-available)
- [`shim_load_error`](#shim-load-error)
- [`compile_ptx_to_cubin`](#compile-ptx-to-cubin)
- [`register_launch_config`](#register-launch-config)
- [`add_scalar_param_to_shim`](#add-scalar-param-to-shim)
- [`synthesize_tma_descriptor`](#synthesize-tma-descriptor)
- [`synthesize_tma_descriptor_3d`](#synthesize-tma-descriptor-3d)
- [`add_tma_spec_to_shim`](#add-tma-spec-to-shim)
- [`set_mock_ffi_callback`](#set-mock-ffi-callback)
- [`ensure_ffi_registered`](#ensure-ffi-registered)
- [`call_kernel_via_ffi`](#call-kernel-via-ffi)

<a id="cubinrecord"></a>

## `CubinRecord`

- Kind: `class`

```python
class CubinRecord(handle: 'int', ptx_source: 'str', kernel_name: 'str', smem_bytes: 'int' = 0, grid: 'tuple[int, int, int]' = (1, 1, 1), block: 'tuple[int, int, int]' = (1, 1, 1), cu_function: 'Optional[int]' = None, module: 'Any' = None, cubin_bytes: 'Optional[bytes]' = None) -> None
```

A compiled kernel + its launch config.

``cu_function`` is the ``CUfunction`` pointer (as an int) returned by
``cuModuleGetFunction``. It's None on laptop builds where cuda-python
isn't installed or the driver isn't available. ``module`` is kept
alive so the function pointer stays valid for the lifetime of the
kernel.

### Members

#### `smem_bytes`

- Kind: `attribute`

- Value: `0`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

#### `grid`

- Kind: `attribute`

- Value: `(1, 1, 1)`

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

#### `block`

- Kind: `attribute`

- Value: `(1, 1, 1)`

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

#### `cu_function`

- Kind: `attribute`

No docstring yet.

#### `module`

- Kind: `attribute`

No docstring yet.

#### `cubin_bytes`

- Kind: `attribute`

No docstring yet.

<a id="cubinregistry"></a>

## `CubinRegistry`

- Kind: `class`

```python
class CubinRegistry() -> 'None'
```

Thread-safe process-local table mapping handle → CubinRecord.

### Members

#### `register(ptx_source: 'str', kernel_name: 'str', cubin_bytes: 'Optional[bytes]' = None, smem_bytes: 'int' = 0, grid: 'tuple[int, int, int]' = (1, 1, 1), block: 'tuple[int, int, int]' = (1, 1, 1), cu_function: 'Optional[int]' = None, module: 'Any' = None) -> 'int'`

- Kind: `method`

Insert a compiled kernel record and return its process-local handle.

#### `get(handle: 'int') -> 'Optional[CubinRecord]'`

- Kind: `method`

Look up a previously registered kernel handle.

#### `clear() -> 'None'`

- Kind: `method`

Drop all registered kernel records.

<a id="get-cubin-registry"></a>

## `get_cubin_registry`

- Kind: `function`

```python
get_cubin_registry() -> 'CubinRegistry'
```

Return the process-local cubin registry singleton.

<a id="shim-is-available"></a>

## `shim_is_available`

- Kind: `function`

```python
shim_is_available() -> 'bool'
```

True if the C++ shim is loaded and ready.

<a id="shim-load-error"></a>

## `shim_load_error`

- Kind: `function`

```python
shim_load_error() -> 'Optional[str]'
```

Return the last shim-load error, or None if the shim loaded fine.

<a id="compile-ptx-to-cubin"></a>

## `compile_ptx_to_cubin`

- Kind: `function`

```python
compile_ptx_to_cubin(ptx_source: 'str', arch: 'str', kernel_name: 'str' = '', dynamic_smem_bytes: 'int' = 0) -> 'Optional[tuple[int, Any]]'
```

Driver-JIT a PTX string into an executable CUfunction.

Returns ``(cu_function_ptr, cu_module)`` on success. The module is
returned so the caller can hold a reference and keep the function
pointer valid for the life of the kernel.

Returns None on laptops or CI machines without cuda-python / a
CUDA driver — the caller may still register PTX metadata for
tracing tests, but any attempt to launch will fail loudly.

The ``kernel_name`` parameter is the PTX entry symbol (e.g.
``"vector_add"``). If empty, we try to extract it from the
``.visible .entry`` line in the PTX source.

<a id="register-launch-config"></a>

## `register_launch_config`

- Kind: `function`

```python
register_launch_config(handle: 'int', cu_function: 'int', grid: 'tuple[int, int, int]', block: 'tuple[int, int, int]', cluster: 'tuple[int, int, int]' = (1, 1, 1), smem_bytes: 'int' = 0) -> 'None'
```

Populate the shim's launch registry with a (handle, cu_fn, ...) entry.

Called once per handle, right after compilation. The shim's FFI
handler will read this entry at kernel-launch time.

<a id="add-scalar-param-to-shim"></a>

## `add_scalar_param_to_shim`

- Kind: `function`

```python
add_scalar_param_to_shim(handle: 'int', *, value_bits: 'int', size_bytes: 'int') -> 'None'
```

Register a scalar raw .param value with the shim's launch config.

<a id="synthesize-tma-descriptor"></a>

## `synthesize_tma_descriptor`

- Kind: `function`

```python
synthesize_tma_descriptor(shape: 'tuple[int, ...]', dtype, layout, box_shape: 'tuple[int, ...] | None' = None, placeholder_ptr: 'int' = 0) -> 'tuple[Any, int, int]'
```

Build a 128-byte CUtensorMap for (shape, dtype, layout).

Returns ``(host_tmap, host_blob_ptr, device_blob_ptr)``:
  - host_tmap is the cuda-python CUtensorMap Python object; keep it
    alive for the lifetime of the kernel.
  - host_blob_ptr is the raw 128-byte struct address inside the
    host_tmap (what cuTensorMapReplaceAddress wants).
  - device_blob_ptr is a freshly-allocated 128-byte device buffer,
    which the shim uploads the patched host blob into at each launch.

box_shape defaults to a sensible tile for the given swizzle/dtype.
placeholder_ptr is the globalAddress stored in the descriptor at
creation time; the shim replaces it on each launch.

<a id="synthesize-tma-descriptor-3d"></a>

## `synthesize_tma_descriptor_3d`

- Kind: `function`

```python
synthesize_tma_descriptor_3d(height: 'int', width: 'int', dtype, box_major: 'int', box_minor: 'int', *, swizzle_128b: 'bool' = True, padding: 'bool' = False, placeholder_ptr: 'int' = 0) -> 'tuple'
```

Build a 3D CUtensorMap matching fast.cu's ``create_tensor_map``.

The 3D layout reshapes a (height, width) row-major matrix into
``(64_elements, height, width/64)`` so TMA can handle tiles wider
than 64 bf16 elements (which exceeds the 128B swizzle line).

Args:
    height: number of rows (M for A, N for B, N for C).
    width: number of columns (K for A, K for B, M for C).
    dtype: element type (e.g. bf16).
    box_major: tile rows to load (BM for A, BN for B).
    box_minor: tile columns to load (BK for A/B, BM/consumers for C).
    swizzle_128b: use 128B swizzle (True for A/B, False for C).
    padding: pad the innermost box dim to 72 (True for C store).
    placeholder_ptr: global address (patched at launch time).

Returns:
    ``(host_tmap, host_blob_ptr, device_blob_ptr)`` — same as
    ``synthesize_tma_descriptor``.

<a id="add-tma-spec-to-shim"></a>

## `add_tma_spec_to_shim`

- Kind: `function`

```python
add_tma_spec_to_shim(handle: 'int', xla_arg_index: 'int', host_blob_ptr: 'int', device_blob_ptr: 'int') -> 'None'
```

Register a TMA spec with the shim's per-handle launch config.

<a id="set-mock-ffi-callback"></a>

## `set_mock_ffi_callback`

- Kind: `function`

```python
set_mock_ffi_callback(callback: 'Optional[Callable]') -> 'None'
```

Install a mock callback (legacy test hook; pre-shim).

<a id="ensure-ffi-registered"></a>

## `ensure_ffi_registered`

- Kind: `function`

```python
ensure_ffi_registered() -> 'bool'
```

Register the pyptx_launch FFI target with JAX, if not already.

Loads the C++ shim, wraps its ``PyptxLaunch`` symbol in a PyCapsule
via ``jax.ffi.pycapsule``, and registers it for the CUDA platform
under the name ``"pyptx_launch"`` with typed FFI (api_version=1).

Returns True if registration succeeded. Returns False (rather than
raising) on laptops without the shim or without JAX — so tracing
tests can still run.

<a id="call-kernel-via-ffi"></a>

## `call_kernel_via_ffi`

- Kind: `function`

```python
call_kernel_via_ffi(*inputs, cubin_handle: 'int', out_specs: 'Sequence[Tile]', out_shape_env: 'dict[str, int]', grid: 'tuple[int, int, int]', block: 'tuple[int, int, int]', cluster: 'tuple[int, int, int]' = (1, 1, 1), smem_bytes: 'int' = 0) -> 'Any'
```

Build a jax.ffi.ffi_call for this kernel invocation.

Uses typed FFI (api_version=1 / custom_call_api_version=4). The only
attribute passed to the handler is ``cubin_handle`` — grid, block,
and smem are already registered in the shim under that handle.

Returns a JAX array (or tuple of arrays) matching ``out_specs``.
