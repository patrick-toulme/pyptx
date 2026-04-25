# API Reference

These pages are generated from the `pyptx/` package and module docstrings.

- [`pyptx`](pyptx.md): Top-level public API for :mod:`pyptx`.
- [`pyptx.cache`](cache.md): Disk cache for compiled PTX → cubin.
- [`pyptx.jax_support`](jax_support.md): JAX runtime integration for :func:`pyptx.kernel`.
- [`pyptx.kernel`](kernel.md): Kernel tracing, specialization, and runtime dispatch.
- [`pyptx.ptx`](ptx.md): PTX instruction namespace.
- [`pyptx.reg`](reg.md): Register allocation and register-level DSL sugar.
- [`pyptx.smem`](smem.md): Shared-memory allocation, addressing, and barrier objects.
- [`pyptx.specs`](specs.md): Tensor boundary specifications for :func:`pyptx.kernel`.
- [`pyptx.torch_support`](torch_support.md): PyTorch runtime integration for :func:`pyptx.kernel`.
- [`pyptx.types`](types.md): PTX scalar type descriptors.
- [`pyptx.wgmma_layout`](wgmma_layout.md): Canonical GMMA shared-memory layouts for wgmma.mma_async.
