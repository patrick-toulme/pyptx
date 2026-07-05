"""Benchmark pyptx Blackwell flash attention vs FA4 / cuDNN on B200."""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "examples/blackwell")
from flash_attention_blackwell import build_flash_attention_blackwell

try:
    from flash_attn.cute import flash_attn_func as fa4_func
except Exception:
    fa4_func = None

WARMUP = 20
ITERS = 100


def bench(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / ITERS


def main() -> None:
    D = 128
    print(f"{'B':>3} {'S':>6} | {'pyptx us':>9} {'pyptx TF':>9} | {'FA4 us':>9} {'FA4 TF':>8} | ratio")
    for B, S in ((16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)):
        H = 32
        k_fn = build_flash_attention_blackwell(S, B * H, D)
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        q2, k2, v2 = (t.reshape(-1, D) for t in (q, k, v))
        q_s, k_s, v_s = (t.transpose(1, 2).contiguous() for t in (q, k, v))
        flops = 4 * B * H * S * S * D

        us_p = bench(lambda: k_fn(q2, k2, v2))
        tf_p = flops / (us_p * 1e-6) / 1e12
        if fa4_func is not None:
            us_f = bench(lambda: fa4_func(q_s, k_s, v_s, causal=False))
            tf_f = flops / (us_f * 1e-6) / 1e12
            print(f"{B:>3} {S:>6} | {us_p:>9.1f} {tf_p:>9.1f} | {us_f:>9.1f} {tf_f:>8.1f} | {us_f/us_p:>5.2f}x")
        else:
            print(f"{B:>3} {S:>6} | {us_p:>9.1f} {tf_p:>9.1f} | {'—':>9} {'—':>8} |")


if __name__ == "__main__":
    main()
