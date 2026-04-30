"""Ampere RMSNorm example using the maintained pyptx kernel path.

Run ``python examples/ampere/rms_norm.py`` to execute both a ``jax.jit``
path and a PyTorch eager path on ``sm_80`` (A100).

The kernel itself is the maintained ``examples/hopper/rms_norm.py`` —
same v4 vectorized loads, same warp + cross-warp reduction. The only
arch-specific bit is the ``sm_80`` PTX target. All instructions used
(``ld.global.v4.f32``, ``fma.rn.f32``, ``rsqrt.approx.f32``,
``shfl.sync.bfly.b32``, ``mbarrier.*``) are available since sm_80.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

# Wheel-installed path first (pip install pyptx); fall back to repo-root
# layout for `python examples/ampere/rms_norm.py` from a checkout.
try:
    from pyptx.examples.hopper.rms_norm import build_rms_norm as _build_rms_norm
    from pyptx.examples.hopper.rms_norm import rms_norm_ref
except ImportError:
    from examples.hopper.rms_norm import build_rms_norm as _build_rms_norm
    from examples.hopper.rms_norm import rms_norm_ref


def build_rms_norm(B: int, N: int, *, eps: float = 1e-6):
    return _build_rms_norm(B, N, eps=eps, arch="sm_80")


def _run_jax_case(B: int, N: int) -> None:
    k = build_rms_norm(B, N)
    np.random.seed(B * 7919 + N)
    x_np = np.random.randn(B, N).astype(np.float32) * 0.3
    w_np = (np.random.randn(N) * 0.1 + 1.0).astype(np.float32)
    x = jnp.asarray(x_np)
    w = jnp.asarray(w_np)

    @jax.jit
    def fn(x, w):
        return k(x, w)

    out = np.asarray(fn(x, w))
    ref = np.asarray(rms_norm_ref(x, w))
    diff = float(np.abs(out - ref).max())
    ok = bool(np.allclose(out, ref, atol=1e-4, rtol=1e-3))
    status = "OK  " if ok else "FAIL"
    print(f"[JAX  {status}] B={B:4d} N={N:5d}  max_abs={diff:.3e}")


def _run_torch_case(B: int, N: int) -> None:
    import torch

    k = build_rms_norm(B, N)
    np.random.seed(B * 7919 + N)
    x_np = np.random.randn(B, N).astype(np.float32) * 0.3
    w_np = (np.random.randn(N) * 0.1 + 1.0).astype(np.float32)
    x = torch.tensor(x_np, device="cuda")
    w = torch.tensor(w_np, device="cuda")

    out = k(x, w)
    torch.cuda.synchronize()
    ms = (x * x).mean(dim=-1, keepdim=True)
    ref = x * torch.rsqrt(ms + 1e-6) * w
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
