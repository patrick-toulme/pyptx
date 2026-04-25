"""C++ launch shim — loaded via ctypes.cdll from pyptx.jax_support._load_shim().

The prebuilt wheel ships libpyptx_shim.so built with -DPYPTX_NO_XLA_FFI
(Torch path only). If jaxlib is present and the shipped .so lacks the XLA
FFI handler, auto_build.try_auto_build() rebuilds with full FFI support.
"""
