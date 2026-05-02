# Blackwell / Softmax

[:material-github: View on GitHub](https://github.com/patrick-toulme/pyptx/blob/dev/examples/blackwell/softmax.py){ .md-button } 
[:material-file-code: `examples/blackwell/softmax.py`](https://github.com/patrick-toulme/pyptx/blob/dev/examples/blackwell/softmax.py){ .md-button }

## Overview

Fused row-wise softmax tuned for Blackwell (sm_100a), written in pyptx,
callable from JAX and PyTorch.

Reaches **5.8 TB/s** at B=2048 N=8192 f32 on B200, **2.76x** faster than
``torch.softmax`` eager, which is expected since softmax on torch uses
std::exp which is slower.

Thin arch wrapper around ``examples/hopper/softmax.py`` — same kernel,
compiled for ``sm_100a`` to take advantage of Blackwell-specific PTX
improvements.

Run ``python examples/blackwell/softmax.py`` to execute both a ``jax.jit``
path and a PyTorch eager path.

## Source

??? example "Full source"

    ```python
    """Fused row-wise softmax tuned for Blackwell (sm_100a), written in pyptx,
    callable from JAX and PyTorch.

    Reaches **5.8 TB/s** at B=2048 N=8192 f32 on B200, **2.76x** faster than
    ``torch.softmax`` eager, which is expected since softmax on torch uses
    std::exp which is slower.

    Thin arch wrapper around ``examples/hopper/softmax.py`` — same kernel,
    compiled for ``sm_100a`` to take advantage of Blackwell-specific PTX
    improvements.

    Run ``python examples/blackwell/softmax.py`` to execute both a ``jax.jit``
    path and a PyTorch eager path.
    """
    from __future__ import annotations

    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import numpy as np

    try:
        from pyptx.examples.hopper.softmax import build_softmax as _build_softmax
        from pyptx.examples.hopper.softmax import softmax_ref
    except ImportError:
        from examples.hopper.softmax import build_softmax as _build_softmax
        from examples.hopper.softmax import softmax_ref


    def build_softmax(B: int, N: int):
        return _build_softmax(B, N, arch="sm_100a")


    def _run_jax_case(B: int, N: int) -> None:
        k = build_softmax(B, N)
        np.random.seed(B * 7919 + N)
        x_np = np.random.randn(B, N).astype(np.float32)
        x = jnp.asarray(x_np)

        @jax.jit
        def fn(x):
            return k(x)

        out = np.asarray(fn(x))
        ref = np.asarray(softmax_ref(x))
        diff = float(np.abs(out - ref).max())
        ok = bool(np.allclose(out, ref, atol=1e-4, rtol=1e-3))
        status = "OK  " if ok else "FAIL"
        print(f"[JAX  {status}] B={B:4d} N={N:5d}  max_abs={diff:.3e}")


    def _run_torch_case(B: int, N: int) -> None:
        import torch

        k = build_softmax(B, N)
        np.random.seed(B * 7919 + N)
        x_np = np.random.randn(B, N).astype(np.float32)
        x = torch.tensor(x_np, device="cuda")

        out = k(x)
        torch.cuda.synchronize()
        ref = torch.softmax(x, dim=-1)
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
