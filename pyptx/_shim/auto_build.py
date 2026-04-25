"""Auto-build the pyptx shim on first use.

Called lazily from jax_support._load_shim() when the pre-built .so is
not found. Tries two build modes:

1. With jaxlib headers (full shim: JAX FFI + PyTorch raw path)
2. Without jaxlib (Torch-only: raw path only, -DPYPTX_NO_XLA_FFI)

Either mode produces a working libpyptx_shim.so for PyTorch. Mode 1
also enables the JAX FFI handler.
"""
from __future__ import annotations

import os
import subprocess


def try_auto_build() -> str | None:
    """Attempt to compile libpyptx_shim.so. Returns the path on success."""
    shim_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(shim_dir, "libpyptx_shim.so")

    if os.path.isfile(so_path):
        return so_path

    src = os.path.join(shim_dir, "pyptx_launch.cc")
    if not os.path.isfile(src):
        return None

    # Try with jaxlib headers first (full shim)
    jaxlib_include = _find_jaxlib_include()
    if jaxlib_include is not None:
        if _compile(src, so_path, jaxlib_include=jaxlib_include):
            return so_path

    # Fall back to Torch-only build (no XLA FFI)
    if _compile(src, so_path, no_xla_ffi=True):
        return so_path

    return None


def _find_jaxlib_include() -> str | None:
    try:
        import jaxlib
        p = os.path.join(os.path.dirname(jaxlib.__file__), "include")
        return p if os.path.isdir(p) else None
    except ImportError:
        return None


def _compile(
    src: str,
    out: str,
    *,
    jaxlib_include: str | None = None,
    no_xla_ffi: bool = False,
) -> bool:
    cmd = ["g++", "-std=c++17", "-O2", "-fPIC", "-shared"]
    if jaxlib_include:
        cmd += ["-I", jaxlib_include]
    if no_xla_ffi:
        cmd += ["-DPYPTX_NO_XLA_FFI"]
    cmd += [src, "-o", out, "-ldl"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        return os.path.isfile(out)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
