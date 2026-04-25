# `pyptx.torch_support`

> This page is generated from source docstrings and public symbols.

PyTorch runtime integration for :func:`pyptx.kernel`.

This module is the Torch-side counterpart to :mod:`pyptx.jax_support`.
The PTX compilation and launch-record machinery is shared; this module
focuses on the Torch-specific boundary:

1. detect ``torch.Tensor`` inputs
2. collect device pointers and the active CUDA stream
3. allocate output tensors
4. launch through the raw shim entry point
5. expose a ``torch.compile``-compatible custom-op wrapper

The same C++ shim backs both frameworks:

- ``PyptxLaunch`` is used by the JAX/XLA FFI path
- ``pyptx_shim_launch_raw`` is used by the Torch ctypes path

Current scope:

- eager mode works
- ``torch.compile`` works through ``torch.library.custom_op`` plus a
  fake/meta implementation
- inputs are expected to be contiguous CUDA tensors
- backward/autograd via ``differentiable_kernel``

## Public API

- [`is_torch_tensor`](#is-torch-tensor)
- [`any_torch_tensors`](#any-torch-tensors)
- [`call_kernel_via_torch`](#call-kernel-via-torch)
- [`extract_input_shapes`](#extract-input-shapes)
- [`get_or_register_torch_op`](#get-or-register-torch-op)
- [`call_kernel_via_torch_compile`](#call-kernel-via-torch-compile)
- [`differentiable_kernel`](#differentiable-kernel)

<a id="is-torch-tensor"></a>

## `is_torch_tensor`

- Kind: `function`

```python
is_torch_tensor(obj: 'Any') -> 'bool'
```

Return True iff ``obj`` is a PyTorch tensor. False on
non-tensor inputs and on machines where torch isn't installed.

<a id="any-torch-tensors"></a>

## `any_torch_tensors`

- Kind: `function`

```python
any_torch_tensors(inputs: 'Sequence[Any]') -> 'bool'
```

True if ANY of ``inputs`` is a torch.Tensor. Used by the
dispatch logic in ``Kernel.__call__`` to decide between the
JAX path and the PyTorch path.

<a id="call-kernel-via-torch"></a>

## `call_kernel_via_torch`

- Kind: `function`

```python
call_kernel_via_torch(*inputs, cubin_handle: 'int', out_specs: 'Sequence[Tile]', out_shape_env: 'dict[str, int]', grid: 'tuple[int, int, int]', block: 'tuple[int, int, int]', cluster: 'tuple[int, int, int]' = (1, 1, 1), smem_bytes: 'int' = 0) -> 'Any'
```

Launch a pyptx kernel with PyTorch tensor inputs.

The shim's launch config is already registered under ``cubin_handle``
(via ``register_launch_config`` during tracing). This function only:

  1. Allocates output tensors with the right shape / dtype on the
     same CUDA device as the first input.
  2. Builds a ``void**`` array of device pointers in
     inputs-then-outputs order — matching what the shim's FFI
     path builds for JAX.
  3. Calls ``pyptx_shim_launch_raw(handle, stream, ptrs, n)``.
  4. Returns the output tensor(s).

<a id="extract-input-shapes"></a>

## `extract_input_shapes`

- Kind: `function`

```python
extract_input_shapes(inputs) -> 'list[tuple[int, ...]]'
```

Return the concrete shape of each input tensor.

<a id="get-or-register-torch-op"></a>

## `get_or_register_torch_op`

- Kind: `function`

```python
get_or_register_torch_op(cubin_handle: 'int', out_specs: 'Sequence[Tile]', out_shape_env: 'dict[str, int]', grid: 'tuple[int, int, int]', block: 'tuple[int, int, int]', cluster: 'tuple[int, int, int]', smem_bytes: 'int')
```

Return a callable that takes ``(*input_tensors)`` and returns
output tensor(s). The callable is a ``torch.library.custom_op``
that survives ``torch.compile`` / Dynamo tracing.

First call with a given ``cubin_handle`` registers the op;
subsequent calls reuse it.

<a id="call-kernel-via-torch-compile"></a>

## `call_kernel_via_torch_compile`

- Kind: `function`

```python
call_kernel_via_torch_compile(*inputs, cubin_handle: 'int', out_specs: 'Sequence[Tile]', out_shape_env: 'dict[str, int]', grid: 'tuple[int, int, int]', block: 'tuple[int, int, int]', cluster: 'tuple[int, int, int]' = (1, 1, 1), smem_bytes: 'int' = 0) -> 'Any'
```

torch.compile-compatible launch path.

Wraps ``call_kernel_via_torch`` inside a registered
``torch.library.custom_op`` so Dynamo can trace through it.
Returns a single tensor if there's one output, else a tuple.

<a id="differentiable-kernel"></a>

## `differentiable_kernel`

- Kind: `function`

```python
differentiable_kernel(forward_kernel, backward_kernel, *, save_for_backward: 'Sequence[int] | None' = None, num_grad_inputs: 'int | None' = None)
```

Wrap a forward + backward pyptx kernel pair for ``torch.autograd``.

Usage::

    from pyptx.torch_support import differentiable_kernel

    fwd = build_my_forward(M, N)
    bwd = build_my_backward(M, N)

    my_op = differentiable_kernel(
        fwd, bwd,
        save_for_backward=[0, 1],  # save inputs 0 and 1
    )

    # Now supports autograd:
    x = torch.randn(M, N, device="cuda", requires_grad=True)
    w = torch.randn(N, device="cuda", requires_grad=True)
    out = my_op(x, w)
    out.sum().backward()
    print(x.grad, w.grad)

Args:
    forward_kernel: A pyptx ``Kernel`` for the forward pass.
    backward_kernel: A pyptx ``Kernel`` for the backward pass.
        Called with ``(*saved_tensors, *grad_outputs)`` and must
        return one gradient per input.
    save_for_backward: Indices of forward inputs to save for the
        backward pass. Defaults to saving all inputs.
    num_grad_inputs: Number of inputs that need gradients.
        Defaults to the number of forward inputs.
