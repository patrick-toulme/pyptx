"""Hatchling build hook: optionally compile libpyptx_shim.so during wheel build.

Two wheel flavors come out of the same pyproject.toml:

1. **Linux platform wheels** (manylinux_2_28_x86_64 / _aarch64, py3.10-3.13):
   hook compiles the shim with full XLA FFI support (both JAX and Torch
   work out of the box, no runtime rebuild). jaxlib is a build-time dep.
   Built via cibuildwheel on Linux runners.

2. **Pure-Python fallback wheel** (py3-none-any):
   hook skips the compile, wheel stays pure-python. pip picks this up
   on macOS, Windows, or any Linux platform without a matching prebuilt.
   DSL / parser / emitter / transpiler work; @kernel launch raises at
   runtime (no GPU). Built by setting PYPTX_BUILD_SHIM=0.

The XLA FFI v1 ABI is stable, so the .so built against jaxlib >=0.5.0
works for end users on jaxlib >=0.4.20.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class PyptxShimBuildHook(BuildHookInterface):
    PLUGIN_NAME = "pyptx-shim"

    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel":
            return

        # Opt-out: pure-python wheel (py3-none-any). Kernel launch fails
        # at runtime with a clear error; DSL/transpiler still work.
        if os.environ.get("PYPTX_BUILD_SHIM", "1") == "0":
            self.app.display_info("pyptx-shim: PYPTX_BUILD_SHIM=0 — skipping compile, building pure-python wheel")
            return

        # Linux only for prebuilt shim. On macOS/Windows, users should
        # build with PYPTX_BUILD_SHIM=0 to get the pure-python wheel.
        if sys.platform != "linux":
            raise RuntimeError(
                "pyptx prebuilt shim is Linux-only. "
                "Set PYPTX_BUILD_SHIM=0 to build the pure-python wheel instead."
            )

        shim_dir = os.path.join(self.root, "pyptx", "_shim")
        src = os.path.join(shim_dir, "pyptx_launch.cc")
        out = os.path.join(shim_dir, "libpyptx_shim.so")

        cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")
        if cxx is None:
            raise RuntimeError("no C++ compiler found (set $CXX or install g++)")

        # jaxlib ships xla/ffi/api/ffi.h. It's in build-system.requires
        # (Linux only), so the isolated build env has it.
        try:
            import jaxlib  # noqa: F401
            jaxlib_include = os.path.join(
                os.path.dirname(jaxlib.__file__), "include"
            )
            if not os.path.isdir(jaxlib_include):
                raise RuntimeError(
                    f"jaxlib installed but headers missing at {jaxlib_include}"
                )
        except ImportError as e:
            raise RuntimeError(
                "jaxlib required at build time for XLA FFI headers. "
                "It's listed in [build-system].requires for sys_platform=='linux'."
            ) from e

        cmd = [
            cxx, "-std=c++17", "-O2", "-fPIC", "-shared",
            "-I", jaxlib_include,
            src, "-o", out, "-ldl",
        ]
        self.app.display_info(f"pyptx-shim: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        # Register the just-built .so so it's included in the wheel.
        # Done here (not in pyproject.toml) so stale local .so files
        # never leak into pure-python wheels.
        build_data.setdefault("force_include", {})[out] = "pyptx/_shim/libpyptx_shim.so"
