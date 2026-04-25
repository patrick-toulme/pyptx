# Blackwell / Layer Norm

[:material-github: View on GitHub](https://github.com/patrick-toulme/pyptx/blob/dev/examples/blackwell/layer_norm.py){ .md-button } 
[:material-file-code: `examples/blackwell/layer_norm.py`](https://github.com/patrick-toulme/pyptx/blob/dev/examples/blackwell/layer_norm.py){ .md-button }

## Overview

Blackwell LayerNorm example using the maintained pyptx kernel path.

Run ``python examples/blackwell/layer_norm.py`` to execute both a ``jax.jit``
path and a PyTorch eager path on ``sm_100a``.

## Source

??? example "Full source"

    ```python
    """Blackwell LayerNorm example using the maintained pyptx kernel path.

    Run ``python examples/blackwell/layer_norm.py`` to execute both a ``jax.jit``
    path and a PyTorch eager path on ``sm_100a``.
    """
    from __future__ import annotations

    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import numpy as np

    try:
        from pyptx.examples.hopper.layer_norm import build_layer_norm as _build_layer_norm
        from pyptx.examples.hopper.layer_norm import layer_norm_ref
    except ImportError:
        from examples.hopper.layer_norm import build_layer_norm as _build_layer_norm
        from examples.hopper.layer_norm import layer_norm_ref


    def _pick_rows_per_cta(B: int) -> int:
        if B <= 64:
            return 8
        if B <= 256:
            return 4
        return 1


    def build_layer_norm(B: int, N: int, *, eps: float = 1e-5, rows_per_cta: int | None = None):
        if rows_per_cta is None:
            rows_per_cta = _pick_rows_per_cta(B)
        return _build_layer_norm(B, N, eps=eps, rows_per_cta=rows_per_cta, arch="sm_100a")


    def _run_jax_case(B: int, N: int) -> None:
        k = build_layer_norm(B, N)
        np.random.seed(B * 65537 + N)
        x_np = np.random.randn(B, N).astype(np.float32) * 2.0 - 1.0
        w_np = (np.random.randn(N) * 0.1 + 1.0).astype(np.float32)
        b_np = (np.random.randn(N) * 0.1).astype(np.float32)
        x = jnp.asarray(x_np)
        w = jnp.asarray(w_np)
        b = jnp.asarray(b_np)

        @jax.jit
        def fn(x, w, b):
            return k(x, w, b)

        out = np.asarray(fn(x, w, b))
        ref = np.asarray(layer_norm_ref(x, w, b))
        diff = float(np.abs(out - ref).max())
        ok = bool(np.allclose(out, ref, atol=1e-4, rtol=1e-3))
        status = "OK  " if ok else "FAIL"
        print(f"[JAX  {status}] B={B:4d} N={N:5d}  max_abs={diff:.3e}")


    def _run_torch_case(B: int, N: int) -> None:
        import torch

        k = build_layer_norm(B, N)
        np.random.seed(B * 65537 + N)
        x_np = np.random.randn(B, N).astype(np.float32) * 2.0 - 1.0
        w_np = (np.random.randn(N) * 0.1 + 1.0).astype(np.float32)
        b_np = (np.random.randn(N) * 0.1).astype(np.float32)
        x = torch.tensor(x_np, device="cuda")
        w = torch.tensor(w_np, device="cuda")
        b = torch.tensor(b_np, device="cuda")

        out = k(x, w, b)
        torch.cuda.synchronize()
        ref = torch.nn.functional.layer_norm(x, (N,), w, b)
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
