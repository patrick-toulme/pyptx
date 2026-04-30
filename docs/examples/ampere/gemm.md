# Ampere / Gemm

[:material-github: View on GitHub](https://github.com/patrick-toulme/pyptx/blob/dev/examples/ampere/gemm.py){ .md-button } 
[:material-file-code: `examples/ampere/gemm.py`](https://github.com/patrick-toulme/pyptx/blob/dev/examples/ampere/gemm.py){ .md-button }

## Overview

Ampere (sm_80) bf16 GEMM via ``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32``.

Bare-bones single-warp design — one CTA computes a 16x8 output tile,
one ``mma.sync`` per K-tile of width 16. Direct global-memory loads,
no SMEM staging, no ``cp.async`` prefetch yet. Intent is to demonstrate
the Ampere tensor-core path end-to-end correctness on A100 — the ergonomics
of using pyptx for arch=sm_80 kernels — not to compete with cuBLAS.

Inputs:
  A:   (M, K) bf16 row-major
  B_T: (N, K) bf16 row-major  (transposed B; mma.sync wants B with K
       contiguous, which row-major (N, K) gives us)
  D:   (M, N) f32 row-major output

Grid:  (N/8, M/16, 1)
Block: (32, 1, 1)  — one warp per CTA

Each lane t in [0, 32) within the warp owns the m16n8k16 fragments per
the standard PTX layout for row.col:

  group_id   = t / 4         # 0..7   — picks 1 of 8 row positions
  t_in_group = t % 4         # 0..3   — picks 1 of 4 col positions

  A fragment (4 b32 regs, each holding 2 packed bf16):
    a[0] = pack(A[group_id,   2*t_in_group + 0 ],
                A[group_id,   2*t_in_group + 1 ])
    a[1] = pack(A[group_id+8, 2*t_in_group + 0 ],
                A[group_id+8, 2*t_in_group + 1 ])
    a[2] = pack(A[group_id,   2*t_in_group + 8 ],
                A[group_id,   2*t_in_group + 9 ])
    a[3] = pack(A[group_id+8, 2*t_in_group + 8 ],
                A[group_id+8, 2*t_in_group + 9 ])

  B fragment (2 b32 regs, each holding 2 packed bf16, B viewed as (N, K)):
    b[0] = pack(B_T[group_id, 2*t_in_group + 0 ],
                B_T[group_id, 2*t_in_group + 1 ])
    b[1] = pack(B_T[group_id, 2*t_in_group + 8 ],
                B_T[group_id, 2*t_in_group + 9 ])

  D fragment (4 f32 regs):
    d[0] = D[group_id,   2*t_in_group + 0]
    d[1] = D[group_id,   2*t_in_group + 1]
    d[2] = D[group_id+8, 2*t_in_group + 0]
    d[3] = D[group_id+8, 2*t_in_group + 1]

## Source

??? example "Full source"

    ```python
    """Ampere (sm_80) bf16 GEMM via ``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32``.

    Bare-bones single-warp design — one CTA computes a 16x8 output tile,
    one ``mma.sync`` per K-tile of width 16. Direct global-memory loads,
    no SMEM staging, no ``cp.async`` prefetch yet. Intent is to demonstrate
    the Ampere tensor-core path end-to-end correctness on A100 — the ergonomics
    of using pyptx for arch=sm_80 kernels — not to compete with cuBLAS.

    Inputs:
      A:   (M, K) bf16 row-major
      B_T: (N, K) bf16 row-major  (transposed B; mma.sync wants B with K
           contiguous, which row-major (N, K) gives us)
      D:   (M, N) f32 row-major output

    Grid:  (N/8, M/16, 1)
    Block: (32, 1, 1)  — one warp per CTA

    Each lane t in [0, 32) within the warp owns the m16n8k16 fragments per
    the standard PTX layout for row.col:

      group_id   = t / 4         # 0..7   — picks 1 of 8 row positions
      t_in_group = t % 4         # 0..3   — picks 1 of 4 col positions

      A fragment (4 b32 regs, each holding 2 packed bf16):
        a[0] = pack(A[group_id,   2*t_in_group + 0 ],
                    A[group_id,   2*t_in_group + 1 ])
        a[1] = pack(A[group_id+8, 2*t_in_group + 0 ],
                    A[group_id+8, 2*t_in_group + 1 ])
        a[2] = pack(A[group_id,   2*t_in_group + 8 ],
                    A[group_id,   2*t_in_group + 9 ])
        a[3] = pack(A[group_id+8, 2*t_in_group + 8 ],
                    A[group_id+8, 2*t_in_group + 9 ])

      B fragment (2 b32 regs, each holding 2 packed bf16, B viewed as (N, K)):
        b[0] = pack(B_T[group_id, 2*t_in_group + 0 ],
                    B_T[group_id, 2*t_in_group + 1 ])
        b[1] = pack(B_T[group_id, 2*t_in_group + 8 ],
                    B_T[group_id, 2*t_in_group + 9 ])

      D fragment (4 f32 regs):
        d[0] = D[group_id,   2*t_in_group + 0]
        d[1] = D[group_id,   2*t_in_group + 1]
        d[2] = D[group_id+8, 2*t_in_group + 0]
        d[3] = D[group_id+8, 2*t_in_group + 1]
    """
    from __future__ import annotations

    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from pyptx import kernel, ptx, reg, Tile
    from pyptx.types import b32, bf16, f32, u32


    BM, BN, BK = 16, 8, 16


    def build_gemm(M: int, N: int, K: int, *, arch: str = "sm_80"):
        """Build a single-warp Ampere bf16 GEMM kernel.

        Requires ``M % 16 == 0``, ``N % 8 == 0``, ``K % 16 == 0``.
        """
        assert M % BM == 0, f"M={M} must be divisible by {BM}"
        assert N % BN == 0, f"N={N} must be divisible by {BN}"
        assert K % BK == 0, f"K={K} must be divisible by {BK}"
        n_iters = K // BK

        @kernel(
            in_specs=(
                Tile(M, K, bf16),
                Tile(N, K, bf16),  # B_T in (N, K) row-major
            ),
            out_specs=(Tile(M, N, f32),),
            grid=(N // BN, M // BM, 1),
            block=(32, 1, 1),
            arch=arch,
        )
        def gemm(A, B_T, D):
            pa, pb, pd = ptx.global_ptrs(A, B_T, D)

            # CTA picks the (m_base, n_base) of the output tile.
            m_base = reg.scalar(u32)
            ptx.inst.mov.u32(m_base, ptx.special.ctaid.y())
            ptx.inst.shl.b32(m_base, m_base, 4)  # * 16

            n_base = reg.scalar(u32)
            ptx.inst.mov.u32(n_base, ptx.special.ctaid.x())
            ptx.inst.shl.b32(n_base, n_base, 3)  # * 8

            # Per-thread fragment indices.
            tid = reg.scalar(u32)
            ptx.inst.mov.u32(tid, ptx.special.tid.x())
            gid = tid >> 2          # group_id   in [0, 8)
            tig = tid & 3           # t_in_group in [0, 4)
            col_lo = tig << 1       # 2*t_in_group  → 0,2,4,6

            # Row indices: group_id and group_id+8, plus m_base.
            row_lo = m_base + gid
            row_hi = row_lo + 8
            # Col index for B's N axis: group_id, plus n_base.
            n_col = n_base + gid

            # Accumulator: 4 f32 regs, zero-initialized via mov.f32 0.0.
            acc = reg.array(f32, 4)
            zero = reg.scalar(f32, init=0.0)
            for i in range(4):
                ptx.inst.mov.f32(acc[i], zero)

            # Pre-allocate fragment storage outside the K loop so we reuse regs.
            a_fr = reg.array(b32, 4)
            b_fr = reg.array(b32, 2)

            for ki in range(n_iters):
                k_base = ki * BK    # element offset in K

                # ---- Load A fragment ----
                # A is (M, K) row-major bf16; element [r, c] at byte offset (r*K + c)*2.
                # Each register holds 2 packed bf16 = 4 bytes, so we load b32.
                for i, (drow, dcol) in enumerate([(0, 0), (8, 0), (0, 8), (8, 8)]):
                    row = row_lo if drow == 0 else row_hi
                    # col element index = k_base + col_lo + dcol
                    col_elem = k_base + col_lo + dcol
                    # byte offset = (row * K + col_elem) * 2
                    elem_idx = row * K + col_elem
                    byte_off = elem_idx * 2
                    ptx.inst.ld.global_.b32(a_fr[i], ptx.addr(pa + byte_off))

                # ---- Load B fragment (B_T = (N, K) row-major, indexed by [n_col, k]) ----
                for i, dcol in enumerate([0, 8]):
                    col_elem = k_base + col_lo + dcol
                    elem_idx = n_col * K + col_elem
                    byte_off = elem_idx * 2
                    ptx.inst.ld.global_.b32(b_fr[i], ptx.addr(pb + byte_off))

                # ---- mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ----
                ptx.mma.sync(
                    shape=(16, 8, 16),
                    dtype_d=f32, dtype_a=bf16, dtype_b=bf16, dtype_c=f32,
                    d=[acc[0], acc[1], acc[2], acc[3]],
                    a=[a_fr[0], a_fr[1], a_fr[2], a_fr[3]],
                    b=[b_fr[0], b_fr[1]],
                    c=[acc[0], acc[1], acc[2], acc[3]],
                    a_layout="row", b_layout="col",
                )

            # ---- Store D fragment to global ----
            # D is (M, N) f32 row-major; element [r, c] at byte offset (r*N + c)*4.
            # Each thread writes 4 f32: (row_lo, col_lo), (row_lo, col_lo+1),
            # (row_hi, col_lo), (row_hi, col_lo+1).
            d_col_base = n_base + col_lo  # global N coord
            for i, (drow, dcol) in enumerate([(0, 0), (0, 1), (8, 0), (8, 1)]):
                row = row_lo if drow == 0 else row_hi
                col_elem = d_col_base + dcol
                elem_idx = row * N + col_elem
                byte_off = elem_idx * 4
                ptx.inst.st.global_.f32(ptx.addr(pd + byte_off), acc[i])

            ptx.ret()

        return gemm


    # ---------------------------------------------------------------------------
    # References + test harness
    # ---------------------------------------------------------------------------

    def gemm_ref(A: jnp.ndarray, B_T: jnp.ndarray) -> jnp.ndarray:
        """JAX reference: D = A @ B_T.T, accumulated in fp32."""
        return jnp.einsum("mk,nk->mn", A.astype(jnp.float32), B_T.astype(jnp.float32))


    def _run_jax_case(M: int, N: int, K: int) -> None:
        k = build_gemm(M, N, K)
        rng = np.random.default_rng(M * 7919 + N * 31 + K)
        A_np = rng.standard_normal((M, K), dtype=np.float32) * 0.1
        BT_np = rng.standard_normal((N, K), dtype=np.float32) * 0.1
        A = jnp.asarray(A_np, dtype=jnp.bfloat16)
        BT = jnp.asarray(BT_np, dtype=jnp.bfloat16)

        @jax.jit
        def fn(A, BT):
            return k(A, BT)

        out = np.asarray(fn(A, BT))
        ref = np.asarray(gemm_ref(A, BT))
        diff = float(np.abs(out - ref).max())
        # bf16 GEMM: tolerate cuBLAS-class drift
        ok = bool(np.allclose(out, ref, atol=1e-2, rtol=1e-2))
        status = "OK  " if ok else "FAIL"
        print(f"[JAX  {status}] M={M:4d} N={N:4d} K={K:4d}  max_abs={diff:.3e}")


    def _run_torch_case(M: int, N: int, K: int) -> None:
        import torch

        k = build_gemm(M, N, K)
        rng = np.random.default_rng(M * 7919 + N * 31 + K)
        A_np = rng.standard_normal((M, K), dtype=np.float32) * 0.1
        BT_np = rng.standard_normal((N, K), dtype=np.float32) * 0.1
        A = torch.tensor(A_np, dtype=torch.bfloat16, device="cuda")
        BT = torch.tensor(BT_np, dtype=torch.bfloat16, device="cuda")

        out = k(A, BT)
        torch.cuda.synchronize()
        ref = (A.float() @ BT.float().T)
        diff = float((out - ref).abs().max())
        ok = bool(torch.allclose(out, ref, atol=1e-2, rtol=1e-2))
        status = "OK  " if ok else "FAIL"
        print(f"[Torch{status}] M={M:4d} N={N:4d} K={K:4d}  max_abs={diff:.3e}")


    def main() -> None:
        _ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()
        for M, N, K in [
            (16, 8, 16),     # single CTA, single K-iter
            (16, 8, 64),     # single CTA, 4 K-iters
            (32, 16, 32),    # 2x2 CTAs
            (64, 32, 64),    # 4x4 CTAs
            (128, 64, 128),  # 8x8 CTAs
        ]:
            _run_jax_case(M, N, K)
            _run_torch_case(M, N, K)


    if __name__ == "__main__":
        main()
    ```
