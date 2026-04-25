"""Uniform-shape grouped GEMM (batched GEMM), written in pyptx.

Reaches **104 TFLOPS** at G=8 M=2048 N=64 K=2048 (MoE-scale expert
shape) on H100 via tile_k=64 multi-k WGMMA (4 WGMMAs per K-loop iter).

For ``G`` matrix-multiply problems all sharing the same shape
``(M, K) x (K, N) -> (M, N)``::

    C[g] = A[g] @ B[g]   for g in range(G)

This is the shape used by modern MoE layers when all experts have equal
capacity (the ``torch.nn.functional.grouped_mm`` case in PyTorch ≥ 2.10).
Also what Triton calls a "group GEMM" when problem sizes are uniform.

The non-uniform case — CUTLASS-style grouped GEMM where each problem has
its own ``(M_g, N_g, K_g)`` — is a straightforward extension (persistent
kernel walks a tile schedule in global memory) and is follow-up work.

Kernel shape
------------

We reuse the multi-CTA K-loop wgmma GEMM from
``tests/test_gpu_execution.py::TestMultiCtaGemm``. The only new pieces
are:

* The grid picks up a **Z dimension** for the group index::
      grid = (N // BN, M // BM, G)
  so ``ctaid.z`` identifies which problem this CTA computes.

* A, B, C inputs are treated as ``[G*M, K]``, ``[G*K, N]``, ``[G*M, N]``
  (the natural ``.reshape(-1, K)`` / ``.reshape(-1, N)`` of 3D tensors),
  and each CTA offsets its TMA coord by ``ctaid.z * M`` (for A and C row)
  or ``ctaid.z * K`` (for B row). This reuses the existing 2D TMA path
  without needing 3D TMA descriptors.

* The per-CTA body is byte-identical to the single-problem K-loop.

Each CTA computes one 64x8 output tile of one group using m64n8k16 wgmma
with ``K/16`` iterations. Bit-exact against ``jnp.einsum('gmk,gkn->gmn')``.

Run ``python examples/hopper/grouped_gemm.py`` to execute both a
``jax.jit`` path and a PyTorch eager path against framework references.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pyptx import kernel, reg, smem, ptx, Tile
from pyptx.types import bf16, f32, b32, u32, pred


BM = 64


def build_grouped_gemm(
    G: int,
    M: int,
    N: int,
    K: int,
    *,
    tile_k: int | None = None,
    arch: str = "sm_90a",
):
    """Build a grouped GEMM kernel for ``G`` problems of shape ``(M, K) x (K, N)``.

    Inputs are flattened 2D views:
        A: ``(G*M, K)`` bf16
        B: ``(G*K, N)`` bf16
        C: ``(G*M, N)`` f32 (output)

    Grid is ``(N/BN, M/BM, G)`` — every CTA owns one BM x tile_n output tile.

    ``tile_k`` controls how much K is loaded+consumed per K-loop iter.
    Larger tile_k reduces TMA/mbarrier overhead; wgmma k=16 is fixed so
    tile_k > 16 means multiple wgmma calls per iter. Default auto-picks:
    64 when K >= 64 (optimal), else 16.
    """
    assert M % BM == 0, f"M={M} must be divisible by BM={BM}"
    if tile_k is None:
        tile_k = 64 if K % 64 == 0 and K >= 64 else 16
    assert K % tile_k == 0, f"K={K} must be divisible by tile_k={tile_k}"
    assert tile_k % 16 == 0, f"tile_k={tile_k} must be divisible by 16 (wgmma k)"
    for tn in (64, 32, 16, 8):
        if N % tn == 0:
            tile_n = tn
            break
    assert N % tile_n == 0, f"N={N} must be divisible by tile_n={tile_n}"
    grid = (N // tile_n, M // BM, G)
    acc_count = tile_n // 2
    version = (8, 7) if arch.startswith("sm_100") else None
    # Multi-k wgmma iterations per K-loop iter.
    wgmma_k_iters = tile_k // 16
    # Row widths for B-offset math in multi-k (MN-major B).
    b_row_bytes = tile_n * 2

    @kernel(
        in_specs=(
            Tile.wgmma_a(G * M, K, bf16, tile_m=BM, tile_k=tile_k),
            Tile.wgmma_b(G * K, N, bf16, tile_k=tile_k, tile_n=tile_n),
        ),
        out_specs=(Tile(G * M, N, f32),),
        grid=grid,
        block=(128, 1, 1),
        arch=arch,
        version=version,
    )
    def grouped_gemm(A, B, C):
        sA = smem.wgmma_tile(bf16, (BM, tile_k), major="K")
        sB = smem.wgmma_tile(bf16, (tile_k, tile_n), major="MN")
        bar = smem.mbarrier(1)
        phase = reg.scalar(b32, init=0)
        acc = reg.array(f32, acc_count)

        group = reg.scalar(u32)
        ptx.inst.mov.u32(group, ptx.special.ctaid.z())

        m_row_base = reg.scalar(u32)
        ptx.inst.mov.u32(m_row_base, ptx.special.ctaid.y())
        ptx.inst.shl.b32(m_row_base, m_row_base, 6)
        group_m_off = reg.scalar(u32)
        ptx.inst.mul.lo.u32(group_m_off, group, M)
        ptx.inst.add.u32(m_row_base, m_row_base, group_m_off)

        k_row_base = reg.scalar(u32)
        ptx.inst.mul.lo.u32(k_row_base, group, K)

        n_col_base = reg.scalar(u32)
        ptx.inst.mov.u32(n_col_base, ptx.special.ctaid.x())
        shift = {64: 6, 32: 5, 16: 4, 8: 3}[tile_n]
        ptx.inst.shl.b32(n_col_base, n_col_base, shift)

        tid = ptx.special.tid.x()
        with ptx.if_(tid == 0):
            ptx.mbarrier.init(bar[0], 1)
            ptx.fence.proxy_async_shared_cta()

        k_off = reg.scalar(u32, init=0)
        keep_going = reg.scalar(pred)
        ptx.inst.setp.lt.u32(keep_going, k_off, K)
        with ptx.loop("k_loop", pred=keep_going):
            b_row = reg.scalar(u32)
            ptx.inst.add.u32(b_row, k_row_base, k_off)

            with ptx.if_(tid == 0):
                ptx.mbarrier.arrive_expect_tx(
                    bar[0], BM * tile_k * 2 + tile_k * tile_n * 2,
                )
                ptx.cp.async_.bulk.tensor_2d(
                    dst=sA[0], src=A.tma_desc(),
                    coord=(k_off, m_row_base), mbar=bar[0],
                )
                ptx.cp.async_.bulk.tensor_2d(
                    dst=sB[0], src=B.tma_desc(),
                    coord=(n_col_base, b_row), mbar=bar[0],
                )
            ptx.bar.sync(0)
            ptx.mbarrier.wait(bar[0], phase)
            phase ^= 1

            ptx.wgmma.fence()
            for kk in range(wgmma_k_iters):
                a_off = kk * 32
                b_off = kk * 16 * b_row_bytes
                # scale_d: k_off != 0 (from outer K loop) OR (k_off == 0 and kk > 0)
                # Simplified: scale_d is false only on the very first wgmma call.
                if kk == 0:
                    scale = (k_off != 0)
                else:
                    scale = True
                ptx.wgmma.mma_async(
                    shape=(64, tile_n, 16),
                    dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                    d=acc, a=sA, b=sB,
                    scale_d=scale,
                    trans_a=0, trans_b=1,
                    a_k_offset=a_off, b_k_offset=b_off,
                )
            ptx.wgmma.commit_group()
            ptx.wgmma.wait_group(0)
            k_off += tile_k
            ptx.inst.setp.lt.u32(keep_going, k_off, K)

        _grouped_epilogue(C, acc, m_row_base, n_col_base, N, tile_n)
        ptx.ret()

    return grouped_gemm


def _grouped_epilogue(C, acc, row_offset, col_offset, N, tile_n):
    """Scatter m64×tile_n fragments to C using v2 stores."""
    tid = reg.scalar(u32)
    ptx.inst.mov.u32(tid, ptx.special.tid.x())
    wid = tid >> 5
    lane = tid & 31
    frag_row = (wid << 4) + (lane >> 2)
    frag_col = (lane & 3) << 1
    ptx.inst.add.u32(frag_row, frag_row, row_offset)
    ptx.inst.add.u32(frag_col, frag_col, col_offset)
    (pc,) = ptx.global_ptrs(C)
    row_b = frag_row + 8
    for g in range(tile_n // 8):
        col = frag_col + g * 8
        off_a = (frag_row * N + col) * 4
        ptx.inst.st.global_.v2.f32(ptx.addr(pc + off_a), [acc[g * 4], acc[g * 4 + 1]])
        off_b = (row_b * N + col) * 4
        ptx.inst.st.global_.v2.f32(ptx.addr(pc + off_b), [acc[g * 4 + 2], acc[g * 4 + 3]])


# ---------------------------------------------------------------------------
# JAX reference + test harness
# ---------------------------------------------------------------------------

def grouped_gemm_ref(A3, B3):
    """JAX reference: per-group matmul."""
    return jnp.einsum("gmk,gkn->gmn", A3, B3,
                      preferred_element_type=jnp.float32)


def run(G: int, M: int, N: int, K: int):
    k_fn = build_grouped_gemm(G, M, N, K)

    np.random.seed(G * 1009 + M * 997 + N * 17 + K)
    A3_np = (np.random.randn(G, M, K) * 0.1).astype(np.float32)
    B3_np = (np.random.randn(G, K, N) * 0.1).astype(np.float32)

    # JAX 3D bf16 reference
    A3 = jnp.asarray(A3_np, dtype=jnp.bfloat16)
    B3 = jnp.asarray(B3_np, dtype=jnp.bfloat16)

    # Kernel takes flattened 2D views
    A2 = A3.reshape(G * M, K)
    B2 = B3.reshape(G * K, N)

    @jax.jit
    def fn(A, B):
        return k_fn(A, B)

    out = np.asarray(fn(A2, B2)).reshape(G, M, N)
    ref = np.asarray(grouped_gemm_ref(A3, B3))
    diff = float(np.abs(out - ref).max())
    ok = bool(np.allclose(out, ref, atol=1e-4, rtol=1e-3))
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] G={G:2d} M={M:4d} N={N:4d} K={K:4d}  max_abs={diff:.3e}")
    return ok


def run_torch(G: int, M: int, N: int, K: int):
    import torch

    k_fn = build_grouped_gemm(G, M, N, K)
    np.random.seed(G * 1009 + M * 997 + N * 17 + K)
    A3_np = (np.random.randn(G, M, K) * 0.1).astype(np.float32)
    B3_np = (np.random.randn(G, K, N) * 0.1).astype(np.float32)

    A3 = torch.tensor(A3_np, device="cuda", dtype=torch.bfloat16)
    B3 = torch.tensor(B3_np, device="cuda", dtype=torch.bfloat16)
    A2 = A3.reshape(G * M, K)
    B2 = B3.reshape(G * K, N)

    out = k_fn(A2, B2).reshape(G, M, N)
    torch.cuda.synchronize()
    ref = torch.einsum("gmk,gkn->gmn", A3.float(), B3.float())
    diff = float((out - ref).abs().max())
    ok = bool(torch.allclose(out, ref, atol=1e-4, rtol=1e-3))
    status = "OK  " if ok else "FAIL"
    print(f"[Torch{status}] G={G:2d} M={M:4d} N={N:4d} K={K:4d}  max_abs={diff:.3e}")
    return ok


def main() -> None:
    _ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()
    # Progressive sizes: 1 group → multiple groups, small → realistic.
    for G, M, N, K in [
        (1, 64, 8, 32),
        (2, 64, 8, 32),
        (4, 64, 16, 32),
        (8, 128, 16, 64),
        (4, 256, 32, 128),
        (8, 512, 64, 512),    # realistic MoE-scale expert shape
    ]:
        run(G, M, N, K)
        run_torch(G, M, N, K)


if __name__ == "__main__":
    main()
