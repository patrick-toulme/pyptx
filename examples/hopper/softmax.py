"""Fused row-wise softmax, written in pyptx, callable from JAX and PyTorch.

Reaches **2.80 TB/s** at B=2048 N=8192 f32 on H100 (95% of HBM3 peak),
**1.16x** faster than ``torch.softmax`` eager. Vectorized via
``ld.global.v4.f32`` / ``st.global.v4.f32`` so every memory transaction
moves 16 bytes. Bandwidth-bound at large shapes; small shapes are
dispatch-bound (~18us floor) until CUDA-graph replay is wired in.

``Y[b, i] = exp(X[b, i] - max(X[b, :])) / sum(exp(X[b, :] - max(X[b, :])))``

One CTA per row. Each thread:

1. Loads ``items_per_thread`` strided values and tracks a per-thread max.
2. Warp + cross-warp reduction (via shared memory) gives the row max.
3. Re-uses the loaded values to compute ``exp(x - row_max)`` (folded as
   ``ex2(fma(x, log2e, -row_max*log2e))``) and a per-thread partial sum.
4. Same warp + cross-warp reduction pattern gives the row sum.
5. ``rcp.approx.f32`` once, then a multiply per element to write the
   normalized result.

Structurally identical to ``examples/hopper/rms_norm.py``; the only
differences are the per-element math (``max`` and ``exp`` instead of
fma + rsqrt) and that we make two reduction passes instead of one.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pyptx import kernel, reg, ptx, smem, Tile
from pyptx.types import f32, u32


WARP_SIZE = 32
BLOCK_CANDIDATES = (512, 256, 128, 64, 32)
LOG2E = 1.4426950408889634


def _pick_block(n: int) -> int:
    for block in BLOCK_CANDIDATES:
        if n % block == 0 and block >= 32:
            return block
    raise AssertionError(
        f"N={n} must be divisible by one of {BLOCK_CANDIDATES}"
    )


def build_softmax(B: int, N: int, *, arch: str = "sm_90a"):
    """Build a row-wise softmax kernel for inputs shaped ``(B, N)``.

    ``N`` must be divisible by the CTA size.
    """
    block = _pick_block(N)
    num_warps = block // WARP_SIZE
    items_per_thread = N // block
    use_v4 = items_per_thread >= 4 and items_per_thread % 4 == 0
    v4_iters = items_per_thread // 4 if use_v4 else 0

    version = (8, 7) if arch.startswith("sm_100") else None
    @kernel(
        in_specs=(Tile(B, N, f32),),
        out_specs=(Tile(B, N, f32),),
        grid=(B, 1, 1),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def softmax(X, Y):
        partials = smem.alloc(f32, (num_warps, 1))
        stats = smem.alloc(f32, (1, 1))

        # --- prologue ------------------------------------------------
        px, py = ptx.global_ptrs(X, Y)

        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        row = reg.scalar(u32)
        ptx.inst.mov.u32(row, ptx.special.ctaid.x())
        row_byte_off = row * (N * 4)
        px += row_byte_off
        py += row_byte_off
        lane = tid & (WARP_SIZE - 1)
        warp_id = tid >> 5

        x_vals = reg.array(f32, items_per_thread)
        row_max = reg.scalar(f32, init=-float("inf"))

        # --- pass 1: load row + per-thread max ------------------------
        if use_v4:
            elem_base = tid << 2
            for j in range(v4_iters):
                idx = elem_base if j == 0 else elem_base + (j * block * 4)
                ptr = px + idx * 4
                ptx.inst.ld.global_.v4.f32(
                    [x_vals[j*4], x_vals[j*4+1], x_vals[j*4+2], x_vals[j*4+3]],
                    ptx.addr(ptr),
                )
                for sub in range(4):
                    ptx.inst.max.f32(row_max, row_max, x_vals[j*4 + sub])
        else:
            for i in range(items_per_thread):
                idx = reg.scalar(u32)
                ptx.inst.add.u32(idx, tid, i * block)
                ptr = px + idx * 4
                ptx.inst.ld.global_.f32(x_vals[i], ptx.addr(ptr))
                ptx.inst.max.f32(row_max, row_max, x_vals[i])

        # --- cross-warp combine: row_max ------------------------------
        ptx.warp.reduce_max(row_max)

        with ptx.if_(lane == 0):
            partials[warp_id, 0] = row_max
        ptx.bar.sync(0)

        with ptx.if_(tid == 0):
            block_max = reg.scalar(f32, init=-float("inf"))
            for i in range(num_warps):
                ptx.inst.max.f32(block_max, block_max, partials[i, 0])
            stats[0, 0] = block_max
        ptx.bar.sync(0)

        ptx.inst.mov.f32(row_max, stats[0, 0])

        # --- pass 2: exp(x - row_max) folded as ex2(fma(x, log2e, -m*log2e))
        neg_log2e = reg.scalar(f32, init=-LOG2E)
        log2e = reg.scalar(f32, init=LOG2E)
        neg_max_l2 = reg.scalar(f32)
        ptx.inst.mul.f32(neg_max_l2, row_max, neg_log2e)

        row_sum = reg.scalar(f32, init=0.0)
        scaled = reg.scalar(f32)
        for i in range(items_per_thread):
            ptx.inst.fma.rn.f32(scaled, x_vals[i], log2e, neg_max_l2)
            ptx.inst.ex2.approx.f32(x_vals[i], scaled)
            ptx.inst.add.f32(row_sum, row_sum, x_vals[i])

        # --- cross-warp combine: row_sum ------------------------------
        ptx.warp.reduce_sum(row_sum)

        with ptx.if_(lane == 0):
            partials[warp_id, 0] = row_sum
        ptx.bar.sync(0)

        with ptx.if_(tid == 0):
            block_sum = reg.scalar(f32, init=0.0)
            for i in range(num_warps):
                ptx.inst.add.f32(block_sum, block_sum, partials[i, 0])
            stats[0, 0] = block_sum
        ptx.bar.sync(0)

        ptx.inst.mov.f32(row_sum, stats[0, 0])

        inv_sum = reg.scalar(f32)
        ptx.inst.rcp.approx.f32(inv_sum, row_sum)

        # --- write normalized output ----------------------------------
        # Store layout MUST match the load layout (v4 vs scalar).
        if use_v4:
            elem_base = tid << 2
            for j in range(v4_iters):
                idx = elem_base if j == 0 else elem_base + (j * block * 4)
                out_ptr = py + idx * 4
                out_vals = []
                for sub in range(4):
                    out_val = reg.scalar(f32)
                    ptx.inst.mul.f32(out_val, x_vals[j*4 + sub], inv_sum)
                    out_vals.append(out_val)
                ptx.inst.st.global_.v4.f32(ptx.addr(out_ptr), out_vals)
        else:
            for i in range(items_per_thread):
                out_val = reg.scalar(f32)
                ptx.inst.mul.f32(out_val, x_vals[i], inv_sum)

                idx = reg.scalar(u32)
                ptx.inst.add.u32(idx, tid, i * block)
                out_ptr = py + idx * 4
                ptx.inst.st.global_.f32(ptx.addr(out_ptr), out_val)

        ptx.ret()

    return softmax


# ---------------------------------------------------------------------------
# JAX reference + test harness
# ---------------------------------------------------------------------------

def softmax_ref(x: jnp.ndarray) -> jnp.ndarray:
    """Plain JAX reference for correctness checking."""
    shifted = x - jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(shifted)
    return e / jnp.sum(e, axis=-1, keepdims=True)


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
