#!/usr/bin/env bash
# Dev-time build script for the pyptx launch shim.
#
# Usage:
#   cd pyptx/_shim
#   ./build.sh
#
# Produces: libpyptx_shim.so in this directory, which pyptx.jax_support
# loads via ctypes.cdll at runtime.
#
# For release, replace with a setuptools Extension + cibuildwheel.
# This script exists so dev can iterate fast without touching packaging.

set -euo pipefail

SHIM_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SHIM_DIR"

# Find jaxlib's include directory (where xla/ffi/api/ffi.h lives).
# We prefer the jaxlib from the active venv so the FFI ABI matches what
# the running JAX process uses at load time.
JAXLIB_INCLUDE=$(python3 -c '
import jaxlib, os, sys
p = os.path.join(os.path.dirname(jaxlib.__file__), "include")
if not os.path.isdir(p):
    sys.exit("jaxlib include dir not found at " + p)
print(p)
')

echo "Using jaxlib include: $JAXLIB_INCLUDE"

g++ -std=c++17 -O2 -fPIC -shared \
    -I "$JAXLIB_INCLUDE" \
    pyptx_launch.cc \
    -o libpyptx_shim.so \
    -ldl

echo "Built: $SHIM_DIR/libpyptx_shim.so"

# Sanity: confirm the handler symbol is present.
nm -D libpyptx_shim.so 2>/dev/null | grep -E 'PyptxLaunch|pyptx_shim_' | sort || true
