"""Smoke tests for the PyTorch runtime dispatch path.

Each test calls one of the committed examples with ``torch.Tensor``
inputs. ``Kernel.__call__`` detects the tensor type and routes to
``pyptx.torch_support.call_kernel_via_torch`` instead of the JAX FFI
path. The same C++ shim backs both: JAX uses ``PyptxLaunch`` (XLA FFI
handler), PyTorch uses ``pyptx_shim_launch_raw`` (ctypes).

Skipped on machines without torch or without a CUDA device.
"""
import math
import pytest

torch = pytest.importorskip("torch")


_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PyTorch dispatch tests need a CUDA device",
)


# Also skip cleanly if pyptx's shim isn't built — e.g., laptop dev.
def _has_shim() -> bool:
    try:
        from pyptx.jax_support import shim_is_available
        return shim_is_available()
    except Exception:
        return False


_SHIM = pytest.mark.skipif(
    not _has_shim(),
    reason="needs libpyptx_shim.so built (pyptx/_shim/build.sh)",
)


@_CUDA
@_SHIM
class TestTorchDispatch:
    """Each test runs one committed ``examples/`` kernel through the
    PyTorch runtime path and compares against a torch-native reference."""

    def test_rms_norm(self):
        from examples.hopper.rms_norm import build_rms_norm
        B, N = 16, 512
        k = build_rms_norm(B, N)
        torch.manual_seed(0)
        x = torch.randn(B, N, device="cuda") * 0.3
        w = torch.randn(N, device="cuda") * 0.1 + 1.0
        out = k(x, w)
        ref = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6) * w
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3), \
            f"max_abs={float((out - ref).abs().max()):g}"

    def test_layer_norm(self):
        from examples.hopper.layer_norm import build_layer_norm
        B, N = 16, 512
        k = build_layer_norm(B, N)
        torch.manual_seed(1)
        x = torch.randn(B, N, device="cuda") * 2.0 - 1.0
        w = torch.randn(N, device="cuda") * 0.1 + 1.0
        b = torch.randn(N, device="cuda") * 0.1
        out = k(x, w, b)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        ref = (x - mean) * torch.rsqrt(var + 1e-5) * w + b
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3), \
            f"max_abs={float((out - ref).abs().max()):g}"

    def test_swiglu(self):
        from examples.hopper.swiglu import build_fused_silu_mul
        M, F = 32, 1024
        k = build_fused_silu_mul(M, F)
        torch.manual_seed(2)
        g = torch.randn(M, F, device="cuda") * 0.5
        u = torch.randn(M, F, device="cuda") * 0.5
        out = k(g, u)
        ref = torch.nn.functional.silu(g) * u
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3), \
            f"max_abs={float((out - ref).abs().max()):g}"

    def test_grouped_gemm_wgmma_tma(self):
        """The hardest path: wgmma + TMA descriptors under torch."""
        from examples.hopper.grouped_gemm import build_grouped_gemm
        G, M, N, K = 4, 64, 16, 32
        k = build_grouped_gemm(G, M, N, K)
        torch.manual_seed(3)
        a3 = (torch.randn(G, M, K, device="cuda") * 0.1).to(torch.bfloat16)
        b3 = (torch.randn(G, K, N, device="cuda") * 0.1).to(torch.bfloat16)
        a2 = a3.reshape(G * M, K)
        b2 = b3.reshape(G * K, N)
        out = k(a2, b2)
        ref = torch.einsum("gmk,gkn->gmn", a3.float(), b3.float())
        out3 = out.reshape(G, M, N)
        assert torch.allclose(out3, ref, atol=1e-3, rtol=1e-2), \
            f"max_abs={float((out3 - ref).abs().max()):g}"

    def test_flash_attention_kloop(self):
        """Full FA2: wgmma + TMA + K-loop + online softmax + frag scatter,
        all running through the torch runtime path."""
        from examples.hopper.experimental.flash_attention_wgmma_kloop import (
            build_flash_attention_kloop, BM, HEAD_DIM,
        )
        N_seq = 64
        k_fn = build_flash_attention_kloop(N_seq)
        torch.manual_seed(4)
        q = (torch.randn(BM, HEAD_DIM, device="cuda") * 0.3).to(torch.bfloat16)
        k = (torch.randn(N_seq, HEAD_DIM, device="cuda") * 0.3).to(torch.bfloat16)
        v = (torch.randn(N_seq, HEAD_DIM, device="cuda") * 0.3).to(torch.bfloat16)
        k_t = k.T.contiguous()
        out = k_fn(q, k_t, v)
        qf, kf, vf = q.float(), k.float(), v.float()
        ref = torch.softmax(qf @ kf.T / math.sqrt(HEAD_DIM), dim=-1) @ vf
        assert torch.allclose(out, ref, atol=2e-2, rtol=1e-2), \
            f"max_abs={float((out - ref).abs().max()):g}"


@_CUDA
@_SHIM
class TestTorchCompile:
    """Verify that @torch.compile can trace through pyptx kernels via
    the registered torch.library.custom_op. The custom_op lets Dynamo
    keep the kernel call in the FX graph without a graph break; the
    ``register_fake`` implementation provides shape inference without
    touching the GPU.

    First-call compilation (PTX → cubin via cuda-python) may produce
    Dynamo warnings about ``cuuint64_t`` which is expected — the
    compilation path contains cuda-python C extension calls that
    Dynamo can't trace. The steady-state graph doesn't break.
    """

    def test_rms_norm_compiled(self):
        from examples.hopper.rms_norm import build_rms_norm
        k = build_rms_norm(4, 64)
        torch.manual_seed(0)
        x = torch.randn(4, 64, device="cuda") * 0.3
        w = torch.randn(64, device="cuda") * 0.1 + 1.0
        ref = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6) * w

        @torch.compile
        def fn(x, w):
            return k(x, w)

        out = fn(x, w)
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3), \
            f"max_abs={float((out - ref).abs().max()):g}"

    def test_grouped_gemm_compiled(self):
        """wgmma + TMA through torch.compile."""
        from examples.hopper.grouped_gemm import build_grouped_gemm
        G, M, N, K = 2, 64, 8, 32
        k = build_grouped_gemm(G, M, N, K)
        torch.manual_seed(1)
        a = (torch.randn(G * M, K, device="cuda") * 0.1).to(torch.bfloat16)
        b = (torch.randn(G * K, N, device="cuda") * 0.1).to(torch.bfloat16)

        @torch.compile
        def fn(a, b):
            return k(a, b)

        out = fn(a, b)
        ref = torch.einsum(
            "gmk,gkn->gmn",
            a.reshape(G, M, K).float(),
            b.reshape(G, K, N).float(),
        )
        assert torch.allclose(out.reshape(G, M, N), ref, atol=1e-3, rtol=1e-2), \
            f"max_abs={float((out.reshape(G, M, N) - ref).abs().max()):g}"

    def test_flash_attention_compiled(self):
        """Full FA2 K-loop through torch.compile."""
        from examples.hopper.experimental.flash_attention_wgmma_kloop import (
            build_flash_attention_kloop, BM, HEAD_DIM,
        )
        k_fn = build_flash_attention_kloop(32)
        torch.manual_seed(2)
        q = (torch.randn(BM, HEAD_DIM, device="cuda") * 0.3).to(torch.bfloat16)
        k = (torch.randn(32, HEAD_DIM, device="cuda") * 0.3).to(torch.bfloat16)
        v = (torch.randn(32, HEAD_DIM, device="cuda") * 0.3).to(torch.bfloat16)
        k_t = k.T.contiguous()

        @torch.compile
        def fn(q, k_t, v):
            return k_fn(q, k_t, v)

        out = fn(q, k_t, v)
        ref = torch.softmax(
            q.float() @ k.float().T / math.sqrt(HEAD_DIM), dim=-1
        ) @ v.float()
        assert torch.allclose(out, ref, atol=2e-2, rtol=1e-2), \
            f"max_abs={float((out - ref).abs().max()):g}"
