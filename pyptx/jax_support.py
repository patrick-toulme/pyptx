"""JAX runtime integration for :func:`pyptx.kernel`.

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
"""

from __future__ import annotations

import ctypes
import os
import shutil
import site
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from pyptx.specs import Tile


# ---------------------------------------------------------------------------
# Process-local cubin / launch registry
# ---------------------------------------------------------------------------

@dataclass
class CubinRecord:
    """A compiled kernel + its launch config.

    ``cu_function`` is the ``CUfunction`` pointer (as an int) returned by
    ``cuModuleGetFunction``. It's None on laptop builds where cuda-python
    isn't installed or the driver isn't available. ``module`` is kept
    alive so the function pointer stays valid for the lifetime of the
    kernel.
    """

    handle: int
    ptx_source: str
    kernel_name: str
    smem_bytes: int = 0
    grid: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (1, 1, 1)
    cu_function: Optional[int] = None  # CUfunction as intptr
    module: Any = None  # CUmodule (kept alive)
    cubin_bytes: Optional[bytes] = None  # legacy slot; unused by shim path


class CubinRegistry:
    """Thread-safe process-local table mapping handle → CubinRecord."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[int, CubinRecord] = {}
        self._next_handle: int = 1  # 0 reserved for "unregistered"

    def register(
        self,
        ptx_source: str,
        kernel_name: str,
        cubin_bytes: Optional[bytes] = None,
        smem_bytes: int = 0,
        grid: tuple[int, int, int] = (1, 1, 1),
        block: tuple[int, int, int] = (1, 1, 1),
        cu_function: Optional[int] = None,
        module: Any = None,
    ) -> int:
        """Insert a compiled kernel record and return its process-local handle."""
        with self._lock:
            handle = self._next_handle
            self._next_handle += 1
            self._records[handle] = CubinRecord(
                handle=handle,
                ptx_source=ptx_source,
                kernel_name=kernel_name,
                smem_bytes=smem_bytes,
                grid=grid,
                block=block,
                cu_function=cu_function,
                module=module,
                cubin_bytes=cubin_bytes,
            )
            return handle

    def get(self, handle: int) -> Optional[CubinRecord]:
        """Look up a previously registered kernel handle."""
        with self._lock:
            return self._records.get(handle)

    def clear(self) -> None:
        """Drop all registered kernel records."""
        with self._lock:
            self._records.clear()
            self._next_handle = 1

    def __len__(self) -> int:
        return len(self._records)


_GLOBAL_REGISTRY: Optional[CubinRegistry] = None


def get_cubin_registry() -> CubinRegistry:
    """Return the process-local cubin registry singleton."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = CubinRegistry()
    return _GLOBAL_REGISTRY


# ---------------------------------------------------------------------------
# C++ shim loading
# ---------------------------------------------------------------------------
#
# The shim is a small .so produced by pyptx/_shim/build.sh. It exports:
#   - PyptxLaunch                   : the XLA FFI handler symbol
#   - pyptx_shim_register_launch    : add an entry to the launch registry
#   - pyptx_shim_clear_registry     : wipe the registry
#   - pyptx_shim_registry_size      : introspection
#   - pyptx_shim_has_handle         : introspection
#
# We load lazily so `import pyptx` stays clean on laptops where the shim
# hasn't been built.

_SHIM_LOCK = threading.RLock()
_SHIM: Optional[ctypes.CDLL] = None
_SHIM_LOAD_ERROR: Optional[str] = None


def _find_shim_path() -> Optional[str]:
    """Locate libpyptx_shim.so next to this file, auto-building if needed."""
    shim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shim")
    for name in ("libpyptx_shim.so", "pyptx_shim.so"):
        candidate = os.path.join(shim_dir, name)
        if os.path.isfile(candidate):
            return candidate
    try:
        from pyptx._shim.auto_build import try_auto_build
        path = try_auto_build()
        if path is not None:
            return path
    except Exception:
        pass
    return None


def _load_shim() -> Optional[ctypes.CDLL]:
    """Load the C++ launch shim (idempotent).

    Returns the CDLL on success, None on failure (with the reason stored
    in _SHIM_LOAD_ERROR so callers can surface it). Does not raise —
    callers decide whether a missing shim is an error.
    """
    global _SHIM, _SHIM_LOAD_ERROR
    with _SHIM_LOCK:
        if _SHIM is not None:
            return _SHIM
        path = _find_shim_path()
        if path is None:
            _SHIM_LOAD_ERROR = (
                "pyptx launch shim (libpyptx_shim.so) not found. "
                "Run `pyptx/_shim/build.sh` to build it, or install from "
                "a prebuilt wheel."
            )
            return None
        try:
            shim = ctypes.cdll.LoadLibrary(path)
        except OSError as e:
            _SHIM_LOAD_ERROR = f"failed to load {path}: {e}"
            return None

        # Wire up ctypes signatures.
        shim.pyptx_shim_register_launch.restype = None
        shim.pyptx_shim_register_launch.argtypes = [
            ctypes.c_int64,    # handle
            ctypes.c_void_p,   # CUfunction ptr
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,  # grid
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,  # block
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,  # cluster
            ctypes.c_uint32,   # smem_bytes
        ]
        shim.pyptx_shim_clear_registry.restype = None
        shim.pyptx_shim_clear_registry.argtypes = []
        shim.pyptx_shim_registry_size.restype = ctypes.c_size_t
        shim.pyptx_shim_registry_size.argtypes = []
        shim.pyptx_shim_has_handle.restype = ctypes.c_int
        shim.pyptx_shim_has_handle.argtypes = [ctypes.c_int64]
        # PyptxLaunch has a C-function-pointer type; we don't call it
        # directly, we just wrap it in a PyCapsule for JAX.

        _SHIM = shim
        _SHIM_LOAD_ERROR = None
        return _SHIM


def shim_is_available() -> bool:
    """True if the C++ shim is loaded and ready."""
    return _load_shim() is not None


def shim_load_error() -> Optional[str]:
    """Return the last shim-load error, or None if the shim loaded fine."""
    _load_shim()
    return _SHIM_LOAD_ERROR


# ---------------------------------------------------------------------------
# PTX → CUfunction compilation (via cuda-python)
# ---------------------------------------------------------------------------

def _find_ptxas() -> Optional[str]:
    """Locate a usable ``ptxas`` binary."""
    candidates: list[str | None] = []

    # Pip-installed CUDA toolchains can place ptxas under the active
    # environment's site-packages tree instead of PATH.
    site_roots = [
        p for p in site.getsitepackages() + [site.getusersitepackages()]
        if isinstance(p, str)
    ]
    for root in site_roots:
        candidates.extend([
            os.path.join(root, "nvidia", "cuda_nvcc", "bin", "ptxas"),
            os.path.join(root, "nvidia", "cuda_toolkit", "bin", "ptxas"),
        ])

    candidates.extend([
        shutil.which("ptxas"),
        os.path.join(sys.prefix, "bin", "ptxas"),
        "/usr/local/cuda/bin/ptxas",
        "/usr/local/cuda-13.2/bin/ptxas",
        "/usr/local/cuda-13.0/bin/ptxas",
    ])

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _compile_ptx_via_ptxas(
    ptx_source: str,
    arch: str,
    kernel_name: str,
    driver: Any,
) -> tuple[int, Any]:
    """Compile PTX offline with ``ptxas`` and load the resulting cubin."""
    ptxas = _find_ptxas()
    if ptxas is None:
        raise RuntimeError(
            "driver PTX JIT rejected this kernel and ptxas was not found. "
            "Install the CUDA toolkit or add ptxas to PATH."
        )

    with tempfile.TemporaryDirectory(prefix="pyptx_ptxas_") as tmpdir:
        ptx_path = os.path.join(tmpdir, f"{kernel_name}.ptx")
        cubin_path = os.path.join(tmpdir, f"{kernel_name}.cubin")
        with open(ptx_path, "w", encoding="utf-8") as f:
            f.write(ptx_source)

        result = subprocess.run(
            [ptxas, f"-arch={arch}", "-o", cubin_path, ptx_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ptxas failed for {kernel_name} ({arch}):\n{result.stderr}"
            )

        with open(cubin_path, "rb") as f:
            cubin_bytes = f.read()

    err, module = driver.cuModuleLoadData(cubin_bytes)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"cuModuleLoadData(cubin) failed: {err} for kernel {kernel_name}"
        )

    err, fn = driver.cuModuleGetFunction(module, kernel_name.encode())
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"cuModuleGetFunction({kernel_name!r}) failed after ptxas compile: {err}"
        )
    return (int(fn), module)

def compile_ptx_to_cubin(
    ptx_source: str,
    arch: str,
    kernel_name: str = "",
    dynamic_smem_bytes: int = 0,
) -> Optional[tuple[int, Any]]:
    """Driver-JIT a PTX string into an executable CUfunction.

    Returns ``(cu_function_ptr, cu_module)`` on success. The module is
    returned so the caller can hold a reference and keep the function
    pointer valid for the life of the kernel.

    Returns None on laptops or CI machines without cuda-python / a
    CUDA driver — the caller may still register PTX metadata for
    tracing tests, but any attempt to launch will fail loudly.

    The ``kernel_name`` parameter is the PTX entry symbol (e.g.
    ``"vector_add"``). If empty, we try to extract it from the
    ``.visible .entry`` line in the PTX source.
    """
    try:
        from cuda.bindings import driver  # cuda-python >= 12.3
    except ImportError:
        return None

    # Make sure the CUDA driver has been initialized. JAX normally does
    # this the first time it uses a GPU, but we may be called before any
    # JAX op has run.
    try:
        driver.cuInit(0)
    except Exception:
        return None

    try:
        if not kernel_name:
            kernel_name = _extract_entry_name(ptx_source)

        err, module = driver.cuModuleLoadData(ptx_source.encode())
        if err == driver.CUresult.CUDA_ERROR_UNSUPPORTED_PTX_VERSION:
            fn, module = _compile_ptx_via_ptxas(
                ptx_source=ptx_source,
                arch=arch,
                kernel_name=kernel_name,
                driver=driver,
            )
        else:
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"cuModuleLoadData failed: {err}. PTX source:\n{ptx_source[:500]}"
                )

            err, fn = driver.cuModuleGetFunction(module, kernel_name.encode())
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"cuModuleGetFunction({kernel_name!r}) failed: {err}"
                )

        # If the kernel needs dynamic SMEM > 48KB, tell the driver.
        if dynamic_smem_bytes > 0:
            attr = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            err, = driver.cuFuncSetAttribute(fn, attr, dynamic_smem_bytes)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, "
                    f"{dynamic_smem_bytes}) failed: {err}"
                )

        return (int(fn), module)
    except Exception as e:
        # Surface the failure rather than silently swallowing — the user
        # meant to run a kernel and can't.
        raise RuntimeError(f"pyptx: failed to JIT-compile PTX: {e}") from e


def _extract_entry_name(ptx_source: str) -> str:
    """Parse the kernel name out of a ``.visible .entry NAME(...)`` line."""
    for line in ptx_source.splitlines():
        s = line.strip()
        if ".entry" in s:
            # Examples:
            #   .visible .entry vector_add(
            #   .entry my_kernel (
            after = s.split(".entry", 1)[1].lstrip()
            name = ""
            for ch in after:
                if ch.isalnum() or ch in "_$":
                    name += ch
                else:
                    break
            if name:
                return name
    raise ValueError("could not find .entry symbol in PTX source")


def register_launch_config(
    handle: int,
    cu_function: int,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    cluster: tuple[int, int, int] = (1, 1, 1),
    smem_bytes: int = 0,
) -> None:
    """Populate the shim's launch registry with a (handle, cu_fn, ...) entry.

    Called once per handle, right after compilation. The shim's FFI
    handler will read this entry at kernel-launch time.
    """
    shim = _load_shim()
    if shim is None:
        raise RuntimeError(
            f"pyptx: cannot register launch config without the C++ shim: "
            f"{_SHIM_LOAD_ERROR}"
        )
    shim.pyptx_shim_register_launch(
        ctypes.c_int64(handle),
        ctypes.c_void_p(cu_function),
        ctypes.c_uint32(grid[0]),
        ctypes.c_uint32(grid[1]),
        ctypes.c_uint32(grid[2]),
        ctypes.c_uint32(block[0]),
        ctypes.c_uint32(block[1]),
        ctypes.c_uint32(block[2]),
        ctypes.c_uint32(cluster[0]),
        ctypes.c_uint32(cluster[1]),
        ctypes.c_uint32(cluster[2]),
        ctypes.c_uint32(smem_bytes),
    )


def add_scalar_param_to_shim(
    handle: int,
    *,
    value_bits: int,
    size_bytes: int,
) -> None:
    """Register a scalar raw .param value with the shim's launch config."""
    shim = _load_shim()
    if shim is None:
        raise RuntimeError(
            f"pyptx: cannot register scalar raw params without the C++ shim: "
            f"{_SHIM_LOAD_ERROR}"
        )
    if not hasattr(shim, "pyptx_shim_add_scalar_param"):
        raise RuntimeError(
            "pyptx: the loaded shim is too old — it does not export "
            "pyptx_shim_add_scalar_param. Rebuild with pyptx/_shim/build.sh."
        )
    shim.pyptx_shim_add_scalar_param.restype = None
    shim.pyptx_shim_add_scalar_param.argtypes = [
        ctypes.c_int64,   # handle
        ctypes.c_uint64,  # value bits
        ctypes.c_uint32,  # size bytes
    ]
    shim.pyptx_shim_add_scalar_param(
        ctypes.c_int64(handle),
        ctypes.c_uint64(value_bits),
        ctypes.c_uint32(size_bytes),
    )


# ---------------------------------------------------------------------------
# TMA descriptor synthesis
# ---------------------------------------------------------------------------
#
# Hopper's cp.async.bulk.tensor instructions read 128-byte CUtensorMap
# structs ("TMA descriptors") that describe a tiled view over a global
# tensor. Each kernel that uses ``A.tma_desc()`` gets an extra
# ``.param .u64 A_tma_desc`` at the tail of its entry signature, and the
# shim patches a pre-built host-side descriptor with the actual buffer
# pointer on every launch before uploading it to a device-side 128-byte
# slot and passing that slot's address as the param.


@dataclass
class _TmaDescriptorSlot:
    """A compile-time-allocated TMA descriptor slot.

    Carries both the host-side CUtensorMap object (alive for the life of
    the kernel; the shim calls ``cuTensorMapReplaceAddress`` on its raw
    pointer at each launch) and the device-side 128-byte buffer the shim
    uploads to. The XLA arg index identifies which input buffer provides
    the data pointer to patch into the descriptor.
    """

    xla_arg_index: int
    host_tmap: Any      # cuda.bindings.driver.CUtensorMap
    host_blob_ptr: int  # raw pointer returned by host_tmap.getPtr()
    device_blob: int    # CUdeviceptr (int) — 128 bytes of GPU memory
    tensor_name: str    # for diagnostics


def _tma_dtype(dtype) -> Any:
    """Map a pyptx PtxType to CUtensorMapDataType."""
    from cuda.bindings import driver
    name = getattr(dtype, "name", str(dtype))
    table = {
        "bf16": driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        "f16": driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        "f32": driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        "f64": driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
    }
    if name not in table:
        raise ValueError(
            f"pyptx: TMA descriptor synthesis not supported for dtype {name!r}"
        )
    return table[name]


def _tma_swizzle(layout) -> Any:
    """Map a pyptx Layout to CUtensorMapSwizzle."""
    from cuda.bindings import driver
    from pyptx.specs import Layout
    table = {
        Layout.TMA_128B: driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        Layout.TMA_64B: driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
        Layout.TMA_32B: driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    }
    # Row/col-major without swizzle: use NONE; still a valid TMA descriptor.
    if layout in table:
        return table[layout]
    return driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE


def synthesize_tma_descriptor(
    shape: tuple[int, ...],
    dtype,              # PtxType
    layout,             # specs.Layout
    box_shape: tuple[int, ...] | None = None,
    placeholder_ptr: int = 0,
) -> tuple[Any, int, int]:
    """Build a 128-byte CUtensorMap for (shape, dtype, layout).

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
    """
    from cuda.bindings import driver
    from pyptx.specs import Layout

    if len(shape) != 2:
        raise NotImplementedError(
            f"pyptx TMA synthesis only supports rank-2 tensors for now "
            f"(got rank {len(shape)})"
        )

    rows, cols = shape
    elem_bytes = max(dtype.bits // 8, 1)

    # Innermost-first convention: for a row-major (rows, cols) matrix,
    # the innermost (fastest-varying) dimension is "cols".
    u64 = driver.cuuint64_t
    u32 = driver.cuuint32_t
    global_dim = [u64(int(cols)), u64(int(rows))]
    # globalStrides has rank-1 entries. globalStrides[0] is the stride
    # (in bytes) from one element of dim-1 to the next — for row-major
    # 2D that's (cols * elem_bytes).
    global_strides = [u64(int(cols) * elem_bytes)]

    # Default box shape derived from swizzle constraints. For SWIZZLE_128B
    # with bf16 (2B), the innermost box dim must be 64 elements (64*2=128B).
    # Clamp to the global tensor's dims so we never ask TMA to load a box
    # bigger than the underlying tensor — cuTensorMapEncodeTiled rejects
    # that, and even when it accepts, TMA loads silently read garbage.
    if box_shape is None:
        swizzle_bytes = {
            Layout.TMA_128B: 128,
            Layout.TMA_64B: 64,
            Layout.TMA_32B: 32,
        }.get(layout, 128)
        box_cols = max(swizzle_bytes // elem_bytes, 1)
        box_rows = 128
    else:
        assert len(box_shape) == 2
        box_rows, box_cols = box_shape

    box_cols = min(int(box_cols), int(cols))
    box_rows = min(int(box_rows), int(rows))

    box_dim = [u32(int(box_cols)), u32(int(box_rows))]
    elem_strides = [u32(1), u32(1)]

    err, tmap = driver.cuTensorMapEncodeTiled(
        _tma_dtype(dtype),
        2,
        placeholder_ptr,
        global_dim,
        global_strides,
        box_dim,
        elem_strides,
        driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        _tma_swizzle(layout),
        driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"pyptx: cuTensorMapEncodeTiled failed for shape={shape} "
            f"dtype={getattr(dtype, 'name', dtype)} layout={layout}: {err}"
        )

    host_blob_ptr = int(tmap.getPtr())

    # Allocate a 128-byte device buffer to hold the patched descriptor.
    # cuMemAlloc returns 256-byte-aligned memory, which satisfies the
    # TMA descriptor alignment requirement (must be 64-byte aligned).
    err, dev_ptr = driver.cuMemAlloc(128)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"pyptx: cuMemAlloc(128) failed: {err}")

    return tmap, host_blob_ptr, int(dev_ptr)


def synthesize_tma_descriptor_3d(
    height: int,
    width: int,
    dtype,
    box_major: int,
    box_minor: int,
    *,
    swizzle_128b: bool = True,
    padding: bool = False,
    placeholder_ptr: int = 0,
) -> tuple:
    """Build a 3D CUtensorMap matching fast.cu's ``create_tensor_map``.

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
    """
    from cuda.bindings import driver

    elem_bytes = max(dtype.bits // 8, 1)
    assert width % 64 == 0, f"width must be multiple of 64, got {width}"

    u64 = driver.cuuint64_t
    u32 = driver.cuuint32_t

    # 3D shape: (64_innermost, height, width/64)
    global_dim = [u64(64), u64(int(height)), u64(int(width) // 64)]

    # Strides (in bytes): stride[0] = elem_bytes * width, stride[1] = 64 * elem_bytes
    global_strides = [u64(elem_bytes * int(width)), u64(64 * elem_bytes)]

    # Box shape: (72 or 64, box_major, box_minor/64)
    inner_box = 72 if padding else 64
    box_dim = [u32(inner_box), u32(int(box_major)), u32(int(box_minor) // 64)]
    elem_strides = [u32(1), u32(1), u32(1)]

    swizzle = (
        driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
        if swizzle_128b
        else driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
    )

    err, tmap = driver.cuTensorMapEncodeTiled(
        _tma_dtype(dtype),
        3,  # rank = 3
        placeholder_ptr,
        global_dim,
        global_strides,
        box_dim,
        elem_strides,
        driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"pyptx: cuTensorMapEncodeTiled (3D) failed for "
            f"height={height} width={width} box=({box_major},{box_minor}): {err}"
        )

    host_blob_ptr = int(tmap.getPtr())
    err, dev_ptr = driver.cuMemAlloc(128)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"pyptx: cuMemAlloc(128) failed: {err}")

    return tmap, host_blob_ptr, int(dev_ptr)


def add_tma_spec_to_shim(
    handle: int,
    xla_arg_index: int,
    host_blob_ptr: int,
    device_blob_ptr: int,
) -> None:
    """Register a TMA spec with the shim's per-handle launch config."""
    shim = _load_shim()
    if shim is None:
        raise RuntimeError(
            f"pyptx: cannot register TMA spec without the shim: {_SHIM_LOAD_ERROR}"
        )
    if not hasattr(shim, "pyptx_shim_add_tma_spec"):
        raise RuntimeError(
            "pyptx: the loaded shim is too old — it does not export "
            "pyptx_shim_add_tma_spec. Rebuild with pyptx/_shim/build.sh."
        )
    shim.pyptx_shim_add_tma_spec.restype = None
    shim.pyptx_shim_add_tma_spec.argtypes = [
        ctypes.c_int64,     # handle
        ctypes.c_uint32,    # xla_arg_index
        ctypes.c_void_p,    # host_blob pointer
        ctypes.c_uint64,    # device_blob CUdeviceptr
    ]
    shim.pyptx_shim_add_tma_spec(
        ctypes.c_int64(handle),
        ctypes.c_uint32(xla_arg_index),
        ctypes.c_void_p(host_blob_ptr),
        ctypes.c_uint64(device_blob_ptr),
    )


# ---------------------------------------------------------------------------
# FFI target registration
# ---------------------------------------------------------------------------

_FFI_REGISTERED: bool = False
_FFI_LOCK = threading.RLock()
_FFI_MOCK_CALLBACK: Optional[Callable] = None  # legacy test hook


def set_mock_ffi_callback(callback: Optional[Callable]) -> None:
    """Install a mock callback (legacy test hook; pre-shim)."""
    global _FFI_MOCK_CALLBACK
    _FFI_MOCK_CALLBACK = callback


def _pyptx_launch(*args, **kwargs):
    """Legacy Python-callable stub — the real handler is the C++ shim.

    Kept so test_mock_callback still has something to dispatch. On a real
    GPU path, nothing calls into this: the shim's PyptxLaunch symbol is
    what XLA dispatches to via the registered PyCapsule.
    """
    if _FFI_MOCK_CALLBACK is not None:
        return _FFI_MOCK_CALLBACK(*args, **kwargs)
    raise RuntimeError(
        "_pyptx_launch called on a machine without the C++ shim loaded. "
        "This code path only exists for mock tests — real kernel launch "
        "goes through libpyptx_shim.so's PyptxLaunch handler."
    )


def ensure_ffi_registered() -> bool:
    """Register the pyptx_launch FFI target with JAX, if not already.

    Loads the C++ shim, wraps its ``PyptxLaunch`` symbol in a PyCapsule
    via ``jax.ffi.pycapsule``, and registers it for the CUDA platform
    under the name ``"pyptx_launch"`` with typed FFI (api_version=1).

    Returns True if registration succeeded. Returns False (rather than
    raising) on laptops without the shim or without JAX — so tracing
    tests can still run.
    """
    global _FFI_REGISTERED
    with _FFI_LOCK:
        if _FFI_REGISTERED:
            return True
        try:
            import jax
        except ImportError:
            return False

        shim = _load_shim()
        if shim is None:
            # No shim, no launch path. Tracing tests can still run.
            return False

        try:
            capsule = jax.ffi.pycapsule(shim.PyptxLaunch)
            jax.ffi.register_ffi_target(
                "pyptx_launch",
                capsule,
                platform="CUDA",
                api_version=1,
            )
        except Exception as e:
            # Platform not available (CPU-only jaxlib, etc.), or JAX
            # rejected the capsule. Record but don't raise.
            global _SHIM_LOAD_ERROR
            _SHIM_LOAD_ERROR = f"jax.ffi.register_ffi_target failed: {e}"
            return False

        _FFI_REGISTERED = True
        return True


# ---------------------------------------------------------------------------
# Calling a kernel from inside @jax.jit
# ---------------------------------------------------------------------------

def call_kernel_via_ffi(
    *inputs,
    cubin_handle: int,
    out_specs: Sequence[Tile],
    out_shape_env: dict[str, int],
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    cluster: tuple[int, int, int] = (1, 1, 1),
    smem_bytes: int = 0,
) -> Any:
    """Build a jax.ffi.ffi_call for this kernel invocation.

    Uses typed FFI (api_version=1 / custom_call_api_version=4). The only
    attribute passed to the handler is ``cubin_handle`` — grid, block,
    and smem are already registered in the shim under that handle.

    Returns a JAX array (or tuple of arrays) matching ``out_specs``.
    """
    import jax
    import numpy as np
    from jax import ShapeDtypeStruct

    # Build result_shape_dtypes from out_specs + resolved shape env.
    result_shape_dtypes = []
    for spec in out_specs:
        concrete_shape = spec.resolve_shape(out_shape_env)
        numpy_dtype = _ptx_type_to_numpy_dtype(spec.dtype)
        result_shape_dtypes.append(
            ShapeDtypeStruct(concrete_shape, numpy_dtype)
        )

    # Register the FFI target. On laptops without the shim this quietly
    # returns False; we still build the ffi_call because jit().lower()
    # is pure tracing and doesn't fire the handler.
    ensure_ffi_registered()

    call_fn = jax.ffi.ffi_call(
        "pyptx_launch",
        result_shape_dtypes if len(result_shape_dtypes) > 1 else result_shape_dtypes[0],
        has_side_effect=False,
    )

    # The shim's FFI handler only consumes ``cubin_handle`` as an
    # attribute. Everything else (grid, block, smem) is registered
    # alongside the handle via pyptx_shim_register_launch at compile
    # time, so the handler can pick it up from the registry.
    return call_fn(
        *inputs,
        cubin_handle=np.int64(cubin_handle),
    )


def _ptx_type_to_numpy_dtype(dtype):
    """Map pyptx PtxType to numpy/JAX dtype."""
    import jax.numpy as jnp
    name = dtype.name
    mapping = {
        "f16": jnp.float16,
        "bf16": jnp.bfloat16,
        "f32": jnp.float32,
        "f64": jnp.float64,
        "s8": jnp.int8,
        "s16": jnp.int16,
        "s32": jnp.int32,
        "s64": jnp.int64,
        "u8": jnp.uint8,
        "u16": jnp.uint16,
        "u32": jnp.uint32,
        "u64": jnp.uint64,
        "b8": jnp.uint8,
        "b16": jnp.uint16,
        "b32": jnp.uint32,
        "b64": jnp.uint64,
        "pred": jnp.bool_,
    }
    if name not in mapping:
        raise TypeError(f"Cannot map PtxType {name} to numpy dtype")
    return mapping[name]
