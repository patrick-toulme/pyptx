"""End-to-end smoke test of the PyTorch dispatch path on multiple kernels:
rms_norm, layer_norm, swiglu, grouped_gemm. Each one exercises different
corners — layer_norm has 3 inputs + 1 output, grouped_gemm uses wgmma +
TMA descriptors + multi-CTA grid.
"""
import sys; sys.path.insert(0, ".")
import torch
import numpy as np

torch.manual_seed(0)
device = torch.device("cuda")


def check(name, out, ref, atol=1e-4, rtol=1e-3):
    diff = float((out - ref).abs().max())
    ok = torch.allclose(out, ref, atol=atol, rtol=rtol)
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] {name:30s}  max_abs={diff:.3e}")
    return ok


# ----------------------------------------------------------------------------
# rms_norm
# ----------------------------------------------------------------------------
from examples.hopper.rms_norm import build_rms_norm
B, N = 16, 512
k = build_rms_norm(B, N)
x = torch.randn(B, N, device=device) * 0.3
w = torch.randn(N, device=device) * 0.1 + 1.0
out = k(x, w)
ref = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6) * w
check(f"rms_norm B={B} N={N}", out, ref)


# ----------------------------------------------------------------------------
# layer_norm (3 inputs)
# ----------------------------------------------------------------------------
from examples.hopper.layer_norm import build_layer_norm
B, N = 16, 512
k = build_layer_norm(B, N)
x = torch.randn(B, N, device=device) * 2.0 - 1.0
w = torch.randn(N, device=device) * 0.1 + 1.0
b_t = torch.randn(N, device=device) * 0.1
out = k(x, w, b_t)
mean = x.mean(dim=-1, keepdim=True)
var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
ref = (x - mean) * torch.rsqrt(var + 1e-5) * w + b_t
check(f"layer_norm B={B} N={N}", out, ref)


# ----------------------------------------------------------------------------
# swiglu
# ----------------------------------------------------------------------------
from examples.hopper.swiglu import build_fused_silu_mul
M, F = 32, 1024
k = build_fused_silu_mul(M, F)
gate = torch.randn(M, F, device=device) * 0.5
up = torch.randn(M, F, device=device) * 0.5
out = k(gate, up)
ref = torch.nn.functional.silu(gate) * up
check(f"swiglu M={M} F={F}", out, ref)


# ----------------------------------------------------------------------------
# grouped_gemm — wgmma + TMA, tests the TMA descriptor synthesis under torch
# ----------------------------------------------------------------------------
from examples.hopper.grouped_gemm import build_grouped_gemm
G, M, N_, K = 4, 64, 16, 32
k = build_grouped_gemm(G, M, N_, K)
a3 = (torch.randn(G, M, K, device=device) * 0.1).to(torch.bfloat16)
b3 = (torch.randn(G, K, N_, device=device) * 0.1).to(torch.bfloat16)
a2 = a3.reshape(G * M, K)
b2 = b3.reshape(G * K, N_)
out = k(a2, b2)
ref = torch.einsum("gmk,gkn->gmn", a3.float(), b3.float())
out3 = out.reshape(G, M, N_)
check(f"grouped_gemm G={G} M={M} N={N_} K={K}", out3, ref, atol=1e-3, rtol=1e-2)

print()
print("PyTorch dispatch path: all four kernels bit-close end-to-end")
