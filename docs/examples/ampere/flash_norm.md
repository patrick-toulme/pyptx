# Ampere / Flash Norm

[:material-github: View on GitHub](https://github.com/patrick-toulme/pyptx/blob/dev/examples/ampere/flash_norm.py){ .md-button } 
[:material-file-code: `examples/ampere/flash_norm.py`](https://github.com/patrick-toulme/pyptx/blob/dev/examples/ampere/flash_norm.py){ .md-button }

## Overview

Fused FlashNorm for Ampere (sm_80), written in pyptx, callable from JAX
and PyTorch.

Reaches **1.3 TB/s** at B=2048 N=8192 f32 on A100, **1.02x** faster than
the equivalent pyptx rms_norm by eliminating the per-element weight load
and multiply.

``Y[b, i] = X[b, i] / sqrt(mean(X[b, :]^2) + eps)``

Thin arch wrapper around ``examples/hopper/flash_norm.py`` — same kernel,
compiled for ``sm_80``. All instructions (``ld.global.v4.f32``,
``fma.rn.f32``, ``rsqrt.approx.f32``, ``shfl.sync.bfly.b32``,
``bar.sync``) are available since sm_80.

Run ``python examples/ampere/flash_norm.py`` to execute both a ``jax.jit``
path and a PyTorch eager path.

## Source

??? example "Full source"

    ```python
    """Fused FlashNorm for Ampere (sm_80), written in pyptx, callable from JAX
    and PyTorch.

    Reaches **1.3 TB/s** at B=2048 N=8192 f32 on A100, **1.02x** faster than
    the equivalent pyptx rms_norm by eliminating the per-element weight load
    and multiply.

    ``Y[b, i] = X[b, i] / sqrt(mean(X[b, :]^2) + eps)``

    Thin arch wrapper around ``examples/hopper/flash_norm.py`` — same kernel,
    compiled for ``sm_80``. All instructions (``ld.global.v4.f32``,
    ``fma.rn.f32``, ``rsqrt.approx.f32``, ``shfl.sync.bfly.b32``,
    ``bar.sync``) are available since sm_80.

    Run ``python examples/ampere/flash_norm.py`` to execute both a ``jax.jit``
    path and a PyTorch eager path.
    """
    from __future__ import annotations

    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import numpy as np

    try:
        from pyptx.examples.hopper.flash_norm import build_flash_norm as _build_flash_norm
        from pyptx.examples.hopper.flash_norm import flash_norm_ref
    except ImportError:
        from examples.hopper.flash_norm import build_flash_norm as _build_flash_norm
        from examples.hopper.flash_norm import flash_norm_ref


    def build_flash_norm(B: int, N: int, D: int, *, eps: float = 1e-5):
        return _build_flash_norm(B, N, D, eps=eps, arch="sm_80")


    def _run_jax_case(B: int, N: int) -> None:
        _, flash = build_flash_norm(B, N, N)
        np.random.seed(B * 7919 + N)
        x_np = np.random.randn(B, N).astype(np.float32) * 0.3
        x = jnp.asarray(x_np)

        @jax.jit
        def fn(x):
            return flash(x)

        out = np.asarray(fn(x))
        ref = np.asarray(flash_norm_ref(x))
        diff = float(np.abs(out - ref).max())
        ok = bool(np.allclose(out, ref, atol=1e-4, rtol=1e-3))
        status = "OK  " if ok else "FAIL"
        print(f"[JAX  {status}] B={B:4d} N={N:5d}  max_abs={diff:.3e}")


    def _run_torch_case(B: int, N: int) -> None:
        import torch

        _, flash = build_flash_norm(B, N, N)
        np.random.seed(B * 7919 + N)
        x_np = np.random.randn(B, N).astype(np.float32) * 0.3
        x = torch.tensor(x_np, device="cuda")

        out = flash(x)
        torch.cuda.synchronize()
        ms = (x * x).mean(dim=-1, keepdim=True)
        ref = x * torch.rsqrt(ms + 1e-5)
        diff = float((out - ref).abs().max())
        ok = bool(torch.allclose(out, ref, atol=1e-4, rtol=1e-3))
        status = "OK  " if ok else "FAIL"
        print(f"[Torch{status}] B={B:4d} N={N:5d}  max_abs={diff:.3e}")


    def main() -> None:
        _ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()

        for B, N in [(4, 64), (16, 512), (32, 1024), (128, 2048), (256, 4096)]:
            _run_jax_case(B, N)
            _run_torch_case(B, N)


    if __name__ == "__main__":
        main()
    ```
