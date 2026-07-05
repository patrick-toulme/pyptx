"""Attention forward baselines on Blackwell (B200).

Benchmarks every strong attention implementation available on the box so the
pyptx Blackwell kernel has an honest number to beat:

- torch SDPA, cuDNN backend
- torch SDPA, flash backend (FA2 port inside PyTorch)
- FlashAttention-4 (flash-attn-4, CuTeDSL JIT) if installed

Shapes follow the FA3/FA4 paper convention: head_dim in {64, 128}, H chosen so
H*D = 4096 (model dim), batch*seqlen fixed at 16k tokens, causal and
non-causal. bf16 everywhere. FLOPS = 4*B*H*S^2*D (halved for causal).
"""
from __future__ import annotations

import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from flash_attn.cute import flash_attn_func as fa4_func
except Exception:
    fa4_func = None

TOKENS = 16384
SEQLENS = (1024, 2048, 4096, 8192, 16384)
WARMUP = 20
ITERS = 100


def bench(fn) -> float:
    """Median-of-runs CUDA-event timing, returns microseconds."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / ITERS


def run_case(B: int, H: int, S: int, D: int, causal: bool):
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    # bshd layout for FA4
    q_s = q.transpose(1, 2).contiguous()
    k_s = k.transpose(1, 2).contiguous()
    v_s = v.transpose(1, 2).contiguous()

    flops = 4 * B * H * S * S * D * (0.5 if causal else 1.0)

    results = {}

    def sdpa_cudnn():
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=causal
            )

    def sdpa_flash():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=causal
            )

    for name, fn in (("cudnn", sdpa_cudnn), ("torch-flash", sdpa_flash)):
        try:
            us = bench(fn)
            results[name] = (us, flops / (us * 1e-6) / 1e12)
        except Exception as e:
            results[name] = None

    if fa4_func is not None:
        def fa4():
            return fa4_func(q_s, k_s, v_s, causal=causal)

        try:
            us = bench(fa4)
            results["FA4"] = (us, flops / (us * 1e-6) / 1e12)
        except Exception as e:
            print(f"    FA4 failed: {type(e).__name__}: {e}")
            results["FA4"] = None

    return results


def main() -> None:
    torch.manual_seed(0)
    dev = torch.cuda.get_device_name(0)
    print(f"device: {dev}, torch {torch.__version__}, cudnn {torch.backends.cudnn.version()}")
    names = ["cudnn", "torch-flash", "FA4"]
    for D in (128, 64):
        H = 4096 // D
        for causal in (False, True):
            print(f"\n=== head_dim={D} H={H} causal={causal} (bf16, fwd) ===")
            hdr = f"{'B':>3} {'S':>6}"
            for n in names:
                hdr += f" | {n+' us':>12} {n+' TF':>8}"
            print(hdr)
            for S in SEQLENS:
                B = max(TOKENS // S, 1)
                r = run_case(B, H, S, D, causal)
                line = f"{B:>3} {S:>6}"
                for n in names:
                    if r.get(n):
                        us, tf = r[n]
                        line += f" | {us:>12.1f} {tf:>8.1f}"
                    else:
                        line += f" | {'—':>12} {'—':>8}"
                print(line)


if __name__ == "__main__":
    main()
