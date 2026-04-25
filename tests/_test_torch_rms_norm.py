"""Smoke test: call examples/hopper/rms_norm.py's kernel from PyTorch eager.

Proves the PyTorch dispatch path works end-to-end:
  torch tensor inputs → Kernel.__call__ → torch_support.call_kernel_via_torch
  → pyptx_shim_launch_raw → cuLaunchKernel on torch's current CUDA stream.
"""
import torch
import numpy as np

import sys
sys.path.insert(0, ".")
from examples.hopper.rms_norm import build_rms_norm

B, N = 4, 64
k = build_rms_norm(B, N)

torch.manual_seed(42)
x = torch.randn(B, N, dtype=torch.float32, device="cuda") * 0.3
w = torch.randn(N, dtype=torch.float32, device="cuda") * 0.1 + 1.0

out = k(x, w)
print("torch kernel returned:", type(out).__name__, out.shape, out.dtype, out.device)

# Reference in torch
eps = 1e-6
mean_sq = (x * x).mean(dim=-1, keepdim=True)
ref = x * torch.rsqrt(mean_sq + eps) * w

diff = float((out - ref).abs().max())
print(f"max_abs = {diff:.3e}")
print(f"torch.allclose(atol=1e-4): {torch.allclose(out, ref, atol=1e-4, rtol=1e-3)}")
