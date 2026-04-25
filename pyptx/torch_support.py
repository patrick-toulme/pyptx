"""PyTorch runtime integration for :func:`pyptx.kernel`.

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
"""
from __future__ import annotations

import ctypes
import os
from typing import Any, Sequence

# Try to load the C++ Torch extension for lower-overhead dispatch.
_CPP_EXT = None
_CPP_EXT_CHECKED = False

# pyptx internals that are already framework-agnostic
from pyptx.jax_support import (
    _SHIM_LOAD_ERROR,
    _load_shim,
    compile_ptx_to_cubin,
    get_cubin_registry,
    register_launch_config,
    shim_is_available,
    shim_load_error,
    synthesize_tma_descriptor,
    add_tma_spec_to_shim,
)
from pyptx.specs import Tile


# ---------------------------------------------------------------------------
# Torch-side detection (no import at module load; users without torch
# installed shouldn't pay any cost for this module existing).
# ---------------------------------------------------------------------------

_TORCH_TENSOR_TYPE: Any = None


def _get_torch_tensor_type():
    """Lazy-import ``torch.Tensor`` so this module works on machines
    without torch installed."""
    global _TORCH_TENSOR_TYPE
    if _TORCH_TENSOR_TYPE is None:
        try:
            import torch
            _TORCH_TENSOR_TYPE = torch.Tensor
        except ImportError:
            _TORCH_TENSOR_TYPE = False   # sentinel: torch not installed
    return _TORCH_TENSOR_TYPE if _TORCH_TENSOR_TYPE is not False else None


def is_torch_tensor(obj: Any) -> bool:
    """Return True iff ``obj`` is a PyTorch tensor. False on
    non-tensor inputs and on machines where torch isn't installed."""
    t = _get_torch_tensor_type()
    if t is None:
        return False
    return isinstance(obj, t)


def any_torch_tensors(inputs: Sequence[Any]) -> bool:
    """True if ANY of ``inputs`` is a torch.Tensor. Used by the
    dispatch logic in ``Kernel.__call__`` to decide between the
    JAX path and the PyTorch path."""
    return any(is_torch_tensor(x) for x in inputs)


# ---------------------------------------------------------------------------
# Shim loader — attach argtypes for ``pyptx_shim_launch_raw`` on first use
# ---------------------------------------------------------------------------

_LAUNCH_RAW_READY: bool = False


def _ensure_launch_raw_ready() -> Any:
    """Load the shim and attach ctypes signatures for the raw entry
    point. Raises if the shim isn't available."""
    global _LAUNCH_RAW_READY
    shim = _load_shim()
    if shim is None:
        raise RuntimeError(
            f"pyptx: cannot launch PyTorch kernel without the C++ shim: "
            f"{shim_load_error()}"
        )
    if not _LAUNCH_RAW_READY:
        if not hasattr(shim, "pyptx_shim_launch_raw"):
            raise RuntimeError(
                "pyptx: the loaded shim is too old — no pyptx_shim_launch_raw "
                "symbol. Rebuild with pyptx/_shim/build.sh."
            )
        # int pyptx_shim_launch_raw(int64, uint64, void**, size_t, char*)
        shim.pyptx_shim_launch_raw.restype = ctypes.c_int
        shim.pyptx_shim_launch_raw.argtypes = [
            ctypes.c_int64,    # handle
            ctypes.c_uint64,   # stream (CUstream as uintptr_t)
            ctypes.c_void_p,   # pointer to void* array (the buffer pointers)
            ctypes.c_size_t,   # n_buffers
            ctypes.c_char_p,   # err_out (128-byte buffer)
        ]
        _LAUNCH_RAW_READY = True
    return shim


# ---------------------------------------------------------------------------
# C++ extension fast path (optional)
# ---------------------------------------------------------------------------

def _try_load_cpp_ext():
    """Try to JIT-compile the C++ Torch extension for faster dispatch."""
    global _CPP_EXT, _CPP_EXT_CHECKED
    if _CPP_EXT_CHECKED:
        return _CPP_EXT
    _CPP_EXT_CHECKED = True
    try:
        import torch.utils.cpp_extension
        ext_src = os.path.join(os.path.dirname(__file__), "_shim", "torch_ext.cpp")
        if not os.path.exists(ext_src):
            return None
        _CPP_EXT = torch.utils.cpp_extension.load(
            name="pyptx_torch_ext",
            sources=[ext_src],
            verbose=False,
        )
        shim_path = _find_shim_path()
        if shim_path and _CPP_EXT is not None:
            _CPP_EXT.load_shim(shim_path)
    except Exception:
        _CPP_EXT = None
    return _CPP_EXT


def _find_shim_path():
    """Return the path to libpyptx_shim.so, or None."""
    shim_dir = os.path.join(os.path.dirname(__file__), "_shim")
    for name in ("libpyptx_shim.so", "pyptx_shim.so"):
        candidate = os.path.join(shim_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Launch path — extract device ptrs + stream, call the shim
# ---------------------------------------------------------------------------

_CACHED_SHIM = None
_CACHED_ERR_BUF = None
_PTR_ARRAY_TYPES: dict[int, type] = {}


def call_kernel_via_torch(
    *inputs,
    cubin_handle: int,
    out_specs: Sequence[Tile],
    out_shape_env: dict[str, int],
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    cluster: tuple[int, int, int] = (1, 1, 1),
    smem_bytes: int = 0,
) -> Any:
    """Launch a pyptx kernel with PyTorch tensor inputs.

    The shim's launch config is already registered under ``cubin_handle``
    (via ``register_launch_config`` during tracing). This function only:

      1. Allocates output tensors with the right shape / dtype on the
         same CUDA device as the first input.
      2. Builds a ``void**`` array of device pointers in
         inputs-then-outputs order — matching what the shim's FFI
         path builds for JAX.
      3. Calls ``pyptx_shim_launch_raw(handle, stream, ptrs, n)``.
      4. Returns the output tensor(s).
    """
    import torch

    global _CACHED_SHIM, _CACHED_ERR_BUF

    if not inputs:
        raise ValueError("pyptx: PyTorch kernel requires at least one input")

    # --- C++ extension fast path (avoids ctypes overhead) ---
    ext = _try_load_cpp_ext()
    if ext is not None:
        out_shapes = []
        out_dtype_ints = []
        for spec in out_specs:
            out_shapes.append(list(spec.resolve_shape(out_shape_env)))
            td = _ptx_type_to_torch_dtype(spec.dtype)
            out_dtype_ints.append(torch._C._WildcardType(td) if False else
                                 {torch.float16: 5, torch.bfloat16: 15,
                                  torch.float32: 6, torch.float64: 7,
                                  torch.int8: 1, torch.int16: 2,
                                  torch.int32: 3, torch.int64: 4,
                                  torch.uint8: 0, torch.bool: 11}.get(td, 6))
        s = torch.cuda.current_stream(inputs[0].device)
        stream_ptr = int(s.cuda_stream if hasattr(s, 'cuda_stream') else s)
        results = ext.launch_kernel(
            cubin_handle, stream_ptr, list(inputs), out_shapes, out_dtype_ints,
        )
        if len(results) == 1:
            return results[0]
        return tuple(results)

    device = inputs[0].device

    # Allocate outputs from out_specs.
    output_tensors: list[Any] = []
    for spec in out_specs:
        concrete_shape = spec.resolve_shape(out_shape_env)
        torch_dtype = _ptx_type_to_torch_dtype(spec.dtype)
        out = torch.empty(concrete_shape, dtype=torch_dtype, device=device)
        output_tensors.append(out)

    # Build pointer array: inputs then outputs, matching the JAX FFI
    # handler's buffer order so the shim's TMA spec ``xla_arg_index``
    # (same unified index space) still resolves correctly.
    n = len(inputs) + len(output_tensors)
    PtrArrayType = _PTR_ARRAY_TYPES.get(n)
    if PtrArrayType is None:
        PtrArrayType = ctypes.c_void_p * n
        _PTR_ARRAY_TYPES[n] = PtrArrayType
    ptr_array = PtrArrayType()
    for i, t in enumerate(inputs):
        ptr_array[i] = t.data_ptr()
    for i, t in enumerate(output_tensors):
        ptr_array[len(inputs) + i] = t.data_ptr()

    # Get the current CUDA stream for this device.
    s = torch.cuda.current_stream(device)
    stream_u64 = int(s.cuda_stream if hasattr(s, 'cuda_stream') else s)

    # Call the shim (cache shim handle and error buffer).
    if _CACHED_SHIM is None:
        _CACHED_SHIM = _ensure_launch_raw_ready()
        _CACHED_ERR_BUF = ctypes.create_string_buffer(128)
    rc = _CACHED_SHIM.pyptx_shim_launch_raw(
        ctypes.c_int64(cubin_handle),
        ctypes.c_uint64(stream_u64),
        ctypes.cast(ptr_array, ctypes.c_void_p),
        ctypes.c_size_t(n),
        _CACHED_ERR_BUF,
    )
    if rc != 0:
        msg = _CACHED_ERR_BUF.value.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"pyptx: pyptx_shim_launch_raw failed (code {rc}): {msg}"
        )

    if len(output_tensors) == 1:
        return output_tensors[0]
    return tuple(output_tensors)


# ---------------------------------------------------------------------------
# Tensor -> torch dtype mapping (mirror of _ptx_type_to_numpy_dtype)
# ---------------------------------------------------------------------------

def _ptx_type_to_torch_dtype(dtype):
    """Map a pyptx PtxType to a ``torch.dtype``."""
    import torch
    name = dtype.name
    mapping = {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
        "f64": torch.float64,
        "s8": torch.int8,
        "s16": torch.int16,
        "s32": torch.int32,
        "s64": torch.int64,
        "u8": torch.uint8,
        "b8": torch.uint8,
        "b16": torch.uint16 if hasattr(torch, "uint16") else torch.int16,
        "b32": torch.uint32 if hasattr(torch, "uint32") else torch.int32,
        "b64": torch.uint64 if hasattr(torch, "uint64") else torch.int64,
        "pred": torch.bool,
    }
    if name not in mapping:
        raise TypeError(f"Cannot map PtxType {name} to a torch.dtype")
    return mapping[name]


def _torch_dtype_to_ptx_type_name(torch_dtype) -> str:
    """Inverse of ``_ptx_type_to_torch_dtype`` — used to validate that
    an input tensor's dtype matches the spec's PtxType at call time."""
    import torch
    mapping = {
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.float32: "f32",
        torch.float64: "f64",
        torch.int8: "s8",
        torch.int16: "s16",
        torch.int32: "s32",
        torch.int64: "s64",
        torch.uint8: "u8",
        torch.bool: "pred",
    }
    if torch_dtype not in mapping:
        raise TypeError(f"Cannot map torch dtype {torch_dtype} to a pyptx type")
    return mapping[torch_dtype]


def extract_input_shapes(inputs) -> list[tuple[int, ...]]:
    """Return the concrete shape of each input tensor."""
    return [tuple(t.shape) for t in inputs]


# ---------------------------------------------------------------------------
# torch.compile integration via torch.library.custom_op
# ---------------------------------------------------------------------------
#
# Each (kernel, cubin_handle) pair gets a unique custom_op registered the
# first time it's called. The op signature takes a flat list of input
# tensors and returns a flat list of output tensors.
#
# The ``register_fake`` implementation (run during compilation / tracing)
# creates empty tensors with the right shapes — it never touches the GPU.
# The real implementation calls ``call_kernel_via_torch``.
#
# The registration is cached in ``_REGISTERED_OPS`` so each handle is
# only registered once.

_REGISTERED_OPS: dict[int, Any] = {}


def get_or_register_torch_op(
    cubin_handle: int,
    out_specs: Sequence[Tile],
    out_shape_env: dict[str, int],
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    cluster: tuple[int, int, int],
    smem_bytes: int,
):
    """Return a callable that takes ``(*input_tensors)`` and returns
    output tensor(s). The callable is a ``torch.library.custom_op``
    that survives ``torch.compile`` / Dynamo tracing.

    First call with a given ``cubin_handle`` registers the op;
    subsequent calls reuse it.
    """
    if cubin_handle in _REGISTERED_OPS:
        return _REGISTERED_OPS[cubin_handle]

    import torch

    op_name = f"pyptx::kernel_{cubin_handle}"

    # Capture the launch kwargs in the closure.
    launch_kwargs = dict(
        cubin_handle=cubin_handle,
        out_specs=out_specs,
        out_shape_env=out_shape_env,
        grid=grid,
        block=block,
        cluster=cluster,
        smem_bytes=smem_bytes,
    )

    # The real implementation — runs on GPU. Uses an explicit schema
    # string because torch's schema inference can't handle lowercase
    # ``list[torch.Tensor]`` annotations.
    @torch.library.custom_op(
        op_name, mutates_args=(),
        schema="(Tensor[] inputs) -> Tensor[]",
    )
    def pyptx_op(inputs):
        result = call_kernel_via_torch(*inputs, **launch_kwargs)
        if isinstance(result, torch.Tensor):
            return [result]
        return list(result)

    # The fake / meta implementation — runs during torch.compile tracing
    # to infer output shapes without touching the GPU.
    @pyptx_op.register_fake
    def fake(inputs):
        device = inputs[0].device if inputs else torch.device("cuda")
        outs = []
        for spec in out_specs:
            concrete_shape = spec.resolve_shape(out_shape_env)
            torch_dtype = _ptx_type_to_torch_dtype(spec.dtype)
            outs.append(torch.empty(concrete_shape, dtype=torch_dtype, device=device))
        return outs

    _REGISTERED_OPS[cubin_handle] = pyptx_op
    return pyptx_op


def call_kernel_via_torch_compile(
    *inputs,
    cubin_handle: int,
    out_specs: Sequence[Tile],
    out_shape_env: dict[str, int],
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    cluster: tuple[int, int, int] = (1, 1, 1),
    smem_bytes: int = 0,
) -> Any:
    """torch.compile-compatible launch path.

    Wraps ``call_kernel_via_torch`` inside a registered
    ``torch.library.custom_op`` so Dynamo can trace through it.
    Returns a single tensor if there's one output, else a tuple.
    """
    op = get_or_register_torch_op(
        cubin_handle=cubin_handle,
        out_specs=out_specs,
        out_shape_env=out_shape_env,
        grid=grid,
        block=block,
        cluster=cluster,
        smem_bytes=smem_bytes,
    )
    result_list = op(list(inputs))
    if len(result_list) == 1:
        return result_list[0]
    return tuple(result_list)


# ---------------------------------------------------------------------------
# Autograd support — differentiable_kernel
# ---------------------------------------------------------------------------


def differentiable_kernel(
    forward_kernel,
    backward_kernel,
    *,
    save_for_backward: Sequence[int] | None = None,
    num_grad_inputs: int | None = None,
):
    """Wrap a forward + backward pyptx kernel pair for ``torch.autograd``.

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
    """
    import torch

    class _PyptxAutograd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *inputs):
            if save_for_backward is not None:
                ctx.save_for_backward(*(inputs[i] for i in save_for_backward))
                ctx._saved_indices = save_for_backward
            else:
                ctx.save_for_backward(*inputs)
                ctx._saved_indices = list(range(len(inputs)))
            ctx._num_inputs = len(inputs)
            with torch.no_grad():
                return forward_kernel(*inputs)

        @staticmethod
        def backward(ctx, *grad_outputs):
            saved = ctx.saved_tensors
            with torch.no_grad():
                grads = backward_kernel(*saved, *grad_outputs)
            if not isinstance(grads, tuple):
                grads = (grads,)
            n = num_grad_inputs if num_grad_inputs is not None else ctx._num_inputs
            if len(grads) < n:
                grads = grads + (None,) * (n - len(grads))
            return grads[:n]

    def wrapper(*inputs):
        return _PyptxAutograd.apply(*inputs)

    wrapper.__name__ = getattr(forward_kernel, '__name__',
                               getattr(forward_kernel, '_fn', type(forward_kernel)).__name__)
    wrapper.__doc__ = f"Differentiable pyptx kernel ({wrapper.__name__})"
    wrapper.forward_kernel = forward_kernel
    wrapper.backward_kernel = backward_kernel
    return wrapper
