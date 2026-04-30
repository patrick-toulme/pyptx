"""Ampere row-wise softmax example using the maintained pyptx kernel path.

Run ``python examples/ampere/softmax.py`` to execute both a ``jax.jit``
path and a PyTorch eager path on ``sm_80`` (A100).

The kernel itself is the maintained ``examples/hopper/softmax.py`` —
v4 loads, two-pass max + sum reductions, ``ex2(fma(x, log2e, -m·log2e))``
fold for the exp. The only arch-specific bit is the ``sm_80`` PTX
target.
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
    return _build_softmax(B, N, arch="sm_80")


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
