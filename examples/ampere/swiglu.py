"""Ampere SwiGLU example using the maintained pyptx kernel path.

Run ``python examples/ampere/swiglu.py`` to execute both a ``jax.jit``
path and a PyTorch eager path on ``sm_80`` (A100).

The kernel itself is the maintained ``examples/hopper/swiglu.py`` — same
v4 vectorized loads/stores, same fast-path silu via ``ex2.approx`` +
``rcp.approx``. The only arch-specific bit is the ``sm_80`` PTX target.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

try:
    from pyptx.examples.hopper.swiglu import build_fused_silu_mul as _build_fused_silu_mul
    from pyptx.examples.hopper.swiglu import fused_silu_mul_ref
except ImportError:
    from examples.hopper.swiglu import build_fused_silu_mul as _build_fused_silu_mul
    from examples.hopper.swiglu import fused_silu_mul_ref


def _pick_rows_per_cta(M: int) -> int:
    if M <= 64:
        return 8
    if M <= 256:
        return 4
    return 1


def build_fused_silu_mul(M: int, F: int, *, rows_per_cta: int | None = None):
    if rows_per_cta is None:
        rows_per_cta = _pick_rows_per_cta(M)
    return _build_fused_silu_mul(M, F, rows_per_cta=rows_per_cta, arch="sm_80")


def _run_jax_case(M: int, F: int) -> None:
    k = build_fused_silu_mul(M, F)
    np.random.seed(M * 65537 + F)
    g = jnp.asarray(np.random.randn(M, F).astype(np.float32) * 0.5)
    u = jnp.asarray(np.random.randn(M, F).astype(np.float32) * 0.5)

    @jax.jit
    def fn(g, u):
        return k(g, u)

    out = np.asarray(fn(g, u))
    ref = np.asarray(fused_silu_mul_ref(g, u))
    diff = float(np.abs(out - ref).max())
    ok = bool(np.allclose(out, ref, atol=1e-4, rtol=1e-3))
    status = "OK  " if ok else "FAIL"
    print(f"[JAX  {status}] M={M:4d} F={F:5d}  max_abs={diff:.3e}")


def _run_torch_case(M: int, F: int) -> None:
    import torch

    k = build_fused_silu_mul(M, F)
    np.random.seed(M * 65537 + F)
    g = torch.tensor(np.random.randn(M, F).astype(np.float32) * 0.5, device="cuda")
    u = torch.tensor(np.random.randn(M, F).astype(np.float32) * 0.5, device="cuda")

    out = k(g, u)
    torch.cuda.synchronize()
    ref = torch.nn.functional.silu(g) * u
    diff = float((out - ref).abs().max())
    ok = bool(torch.allclose(out, ref, atol=1e-4, rtol=1e-3))
    status = "OK  " if ok else "FAIL"
    print(f"[Torch{status}] M={M:4d} F={F:5d}  max_abs={diff:.3e}")


def main() -> None:
    _ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()

    for M, F in [
        (4, 512),
        (16, 512),
        (32, 1024),
        (128, 2048),
        (256, 4096),
        (512, 8192),
    ]:
        _run_jax_case(M, F)
        _run_torch_case(M, F)


if __name__ == "__main__":
    main()
