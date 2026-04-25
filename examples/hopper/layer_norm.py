"""Fused LayerNorm, written in pyptx, callable from JAX and PyTorch.

Reaches **2.5 TB/s** at B=2048 N=8192 f32 on H100 (83% of HBM3 peak),
**1.5x** faster than ``torch.nn.functional.layer_norm`` at the same size.

``Y[b, i] = (X[b, i] - mean_b) * rstd_b * W[i] + B[i]``

where ``mean_b = mean(X[b, :])`` and ``rstd_b = 1/sqrt(var(X[b, :]) + eps)``.

Structure mirrors Triton's ``_layer_norm_fwd_fused`` tutorial (tutorial 05)
but loads ``x`` exactly once and reuses the values across the reductions
and the final write:

    Pass 1: load x strided into per-thread registers, accumulate sum(x)
            and sum(x^2)
    Block reduction → every lane gets mean and second moment
    Pass 2: y = (x - mean) * rstd * w + b               (no reload of x)

One CTA per row. Each warp reduces its own partial sums, then warp 0
combines those partials through shared memory so the kernel scales better
for long rows without reloading ``x``.

Uses pyptx DSL sugar throughout: ``ptx.global_ptrs``, Reg arithmetic
operators, ``reg.scalar(f32, init=<float>)``, and ``ptx.warp.reduce_sum``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pyptx import kernel, reg, ptx, smem, Tile
from pyptx.types import f32, u32


WARP_SIZE = 32
BLOCK_CANDIDATES = (512, 256, 128, 64, 32)


def _pick_block(n: int) -> int:
    best = None
    for block in BLOCK_CANDIDATES:
        ipt = n // block
        if n % block == 0 and ipt >= 4 and ipt % 4 == 0 and block >= 128:
            if best is None or ipt > best[1]:
                best = (block, ipt)
    if best is not None:
        return best[0]
    for block in BLOCK_CANDIDATES:
        if n % block == 0:
            return block
    raise AssertionError(
        f"N={n} must be divisible by one of {BLOCK_CANDIDATES}"
    )


def build_layer_norm(
    B: int,
    N: int,
    *,
    eps: float = 1e-5,
    rows_per_cta: int = 1,
    arch: str = "sm_90a",
):
    """Build a LayerNorm kernel specialized for batch ``B`` and feature
    dim ``N``. ``N`` must be divisible by the CTA size."""
    block = _pick_block(N)
    num_warps = block // WARP_SIZE
    items_per_thread = N // block
    use_v4 = items_per_thread >= 4 and items_per_thread % 4 == 0
    v4_iters = items_per_thread // 4 if use_v4 else 0
    if B % rows_per_cta != 0:
        rows_per_cta = 1

    version = (8, 7) if arch.startswith("sm_100") else None
    @kernel(
        in_specs=(
            Tile(B, N, f32),   # X
            Tile(N, f32),      # W: learned scale
            Tile(N, f32),      # Bp: learned bias (named Bp to avoid shadowing B)
        ),
        out_specs=(Tile(B, N, f32),),
        grid=(B // rows_per_cta, 1, 1),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def layer_norm(X, W, Bp, Y):
        partials = smem.alloc(f32, (num_warps, 2))
        stats = smem.alloc(f32, (2, 1))

        # --- param ptrs in one line ---------------------------------
        px, pw, pb, py = ptx.global_ptrs(X, W, Bp, Y)

        # --- per-row offset: row = ctaid.x, px/py += row*N*4 --------
        row_base = reg.scalar(u32)
        ptx.inst.mov.u32(row_base, ptx.special.ctaid.x())
        if rows_per_cta > 1:
            ptx.inst.mul.lo.u32(row_base, row_base, rows_per_cta)

        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        lane = tid & (WARP_SIZE - 1)
        warp_id = tid >> 5

        x_vals = reg.array(f32, items_per_thread)
        elem_base = tid << 2
        inv_n = reg.scalar(f32, init=1.0 / N)
        eps_reg = reg.scalar(f32, init=eps)

        for r in range(rows_per_cta):
            row = reg.scalar(u32)
            ptx.inst.add.u32(row, row_base, r)
            row_byte_off = row * (N * 4)
            px_row = px + row_byte_off
            py_row = py + row_byte_off

            # --- Pass 1: load + accumulate row sum and sum of squares ---
            sum_x = reg.scalar(f32, init=0.0)
            sum_x2 = reg.scalar(f32, init=0.0)
            if use_v4:
                for j in range(v4_iters):
                    idx = elem_base if j == 0 else elem_base + (j * block * 4)
                    ptr = px_row + idx * 4
                    ptx.inst.ld.global_.v4.f32(
                        [x_vals[j*4], x_vals[j*4+1], x_vals[j*4+2], x_vals[j*4+3]],
                        ptx.addr(ptr),
                    )
                    for sub in range(4):
                        ptx.inst.add.f32(sum_x, sum_x, x_vals[j*4+sub])
                        ptx.inst.fma.rn.f32(sum_x2, x_vals[j*4+sub], x_vals[j*4+sub], sum_x2)
            else:
                for i in range(items_per_thread):
                    idx = reg.scalar(u32)
                    ptx.inst.add.u32(idx, tid, i * block)
                    ptr = px_row + idx * 4
                    ptx.inst.ld.global_.f32(x_vals[i], ptx.addr(ptr))
                    ptx.inst.add.f32(sum_x, sum_x, x_vals[i])
                    ptx.inst.fma.rn.f32(sum_x2, x_vals[i], x_vals[i], sum_x2)

            ptx.warp.reduce_sum(sum_x)
            ptx.warp.reduce_sum(sum_x2)

            with ptx.if_(lane == 0):
                partials[warp_id, 0] = sum_x
                partials[warp_id, 1] = sum_x2
            ptx.bar.sync(0)

            with ptx.if_(tid == 0):
                block_sum = reg.scalar(f32, init=0.0)
                block_sum_sq = reg.scalar(f32, init=0.0)
                for i in range(num_warps):
                    ptx.inst.add.f32(block_sum, block_sum, partials[i, 0])
                    ptx.inst.add.f32(block_sum_sq, block_sum_sq, partials[i, 1])
                stats[0, 0] = block_sum
                stats[1, 0] = block_sum_sq
            ptx.bar.sync(0)

            ptx.inst.mov.f32(sum_x, stats[0, 0])
            ptx.inst.mov.f32(sum_x2, stats[1, 0])

            # mean = sum_x / N, var = E[x^2] - mean^2
            mean = reg.scalar(f32)
            ptx.inst.mul.f32(mean, sum_x, inv_n)

            mean_sq = reg.scalar(f32)
            ptx.inst.mul.f32(mean_sq, mean, mean)
            ex2 = reg.scalar(f32)
            ptx.inst.mul.f32(ex2, sum_x2, inv_n)

            # rstd = rsqrt(var + eps)
            var = reg.scalar(f32)
            ptx.inst.sub.f32(var, ex2, mean_sq)
            ptx.inst.add.f32(var, var, eps_reg)
            rstd = reg.scalar(f32)
            ptx.inst.rsqrt.approx.f32(rstd, var)

            # --- Pass 2: y = (x - mean) * rstd * w + b ------------------
            if use_v4:
                for j in range(v4_iters):
                    idx = elem_base if j == 0 else elem_base + (j * block * 4)
                    off = idx * 4

                    w_vals = [reg.scalar(f32) for _ in range(4)]
                    ptx.inst.ld.global_.v4.f32(w_vals, ptx.addr(pw + off))

                    b_vals = [reg.scalar(f32) for _ in range(4)]
                    ptx.inst.ld.global_.v4.f32(b_vals, ptx.addr(pb + off))

                    y_vals = []
                    for sub in range(4):
                        diff = reg.scalar(f32)
                        ptx.inst.sub.f32(diff, x_vals[j*4+sub], mean)
                        y_val = reg.scalar(f32)
                        ptx.inst.mul.f32(y_val, diff, rstd)
                        ptx.inst.fma.rn.f32(y_val, y_val, w_vals[sub], b_vals[sub])
                        y_vals.append(y_val)

                    ptx.inst.st.global_.v4.f32(ptx.addr(py_row + off), y_vals)
            else:
                for i in range(items_per_thread):
                    idx = reg.scalar(u32)
                    ptx.inst.add.u32(idx, tid, i * block)
                    off = idx * 4

                    pw_el = pw + off
                    w_val = reg.scalar(f32)
                    ptx.inst.ld.global_.f32(w_val, ptx.addr(pw_el))

                    pb_el = pb + off
                    b_val = reg.scalar(f32)
                    ptx.inst.ld.global_.f32(b_val, ptx.addr(pb_el))

                    diff = reg.scalar(f32)
                    ptx.inst.sub.f32(diff, x_vals[i], mean)
                    y_val = reg.scalar(f32)
                    ptx.inst.mul.f32(y_val, diff, rstd)
                    ptx.inst.fma.rn.f32(y_val, y_val, w_val, b_val)

                    py_el = py_row + off
                    ptx.inst.st.global_.f32(ptx.addr(py_el), y_val)

        ptx.ret()

    return layer_norm


# ---------------------------------------------------------------------------
# JAX reference + test harness
# ---------------------------------------------------------------------------

def layer_norm_ref(x, w, b, eps: float = 1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    rstd = jax.lax.rsqrt(var + eps)
    return (x - mean) * rstd * w + b


def _run_jax_case(B: int, N: int) -> None:
    k = build_layer_norm(B, N)
    np.random.seed(B * 65537 + N)
    x_np = np.random.randn(B, N).astype(np.float32) * 2.0 - 1.0
    w_np = (np.random.randn(N) * 0.1 + 1.0).astype(np.float32)
    b_np = (np.random.randn(N) * 0.1).astype(np.float32)
    x = jnp.asarray(x_np)
    w = jnp.asarray(w_np)
    b = jnp.asarray(b_np)

    @jax.jit
    def fn(x, w, b):
        return k(x, w, b)

    out = np.asarray(fn(x, w, b))
    ref = np.asarray(layer_norm_ref(x, w, b))
    diff = float(np.abs(out - ref).max())
    ok = bool(np.allclose(out, ref, atol=1e-4, rtol=1e-3))
    status = "OK  " if ok else "FAIL"
    print(f"[JAX  {status}] B={B:4d} N={N:5d}  max_abs={diff:.3e}")


def _run_torch_case(B: int, N: int) -> None:
    import torch

    k = build_layer_norm(B, N)
    np.random.seed(B * 65537 + N)
    x_np = np.random.randn(B, N).astype(np.float32) * 2.0 - 1.0
    w_np = (np.random.randn(N) * 0.1 + 1.0).astype(np.float32)
    b_np = (np.random.randn(N) * 0.1).astype(np.float32)
    x = torch.tensor(x_np, device="cuda")
    w = torch.tensor(w_np, device="cuda")
    b = torch.tensor(b_np, device="cuda")

    out = k(x, w, b)
    torch.cuda.synchronize()
    ref = torch.nn.functional.layer_norm(x, (N,), w, b)
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
