# Ampere / Gemm Pipelined

[:material-github: View on GitHub](https://github.com/patrick-toulme/pyptx/blob/dev/examples/ampere/gemm_pipelined.py){ .md-button } 
[:material-file-code: `examples/ampere/gemm_pipelined.py`](https://github.com/patrick-toulme/pyptx/blob/dev/examples/ampere/gemm_pipelined.py){ .md-button }

## Overview

A100 (sm_80) bf16 GEMM with cp.async + SMEM ring buffer + mma.sync.

Production-leaning Ampere kernel. Uses every Ampere first-class instruction
pyptx exposes:

    ptx.cp.async_.cg(...)         — async global → SMEM prefetch (16-byte vec)
    ptx.cp.async_.commit_group()  — close pending cp.async into a group
    ptx.cp.async_.wait_group(N)   — wait until <= N groups remain pending
    ptx.mma.sync(shape=(16, 8, 16), ...)        — Ampere tensor-core MMA

For the SMEM → register hand-off this file uses **per-thread `ld.shared.b32`
loads** rather than ``ldmatrix``. The fragment-layout math is identical to
the direct-from-global ``examples/ampere/gemm.py`` — each lane computes its
own m16n8k16 fragment indices and loads a few packed-bf16 pairs. ldmatrix
would be more efficient (single warp-collective instruction, hardware-
optimized bank-conflict handling), but the per-thread path is simpler to
verify and demonstrates the ``cp.async`` + SMEM-staging path on its own.

Block tile:  BM x BN = 64 x 64
K-step:      BK = 16
Warps/CTA:   4    (warp w handles M[w*16 : (w+1)*16])
Per warp:    1 (M) × 8 (N) = 8 ``mma.sync`` calls per K-iter
SMEM stages: 2    (double-buffered ``cp.async`` prefetch)

Inputs:
  A:   (M, K) bf16 row-major
  B_T: (N, K) bf16 row-major
  D:   (M, N) f32 row-major

## Source

??? example "Full source"

    ```python
    """A100 (sm_80) bf16 GEMM with cp.async + SMEM ring buffer + mma.sync.

    Production-leaning Ampere kernel. Uses every Ampere first-class instruction
    pyptx exposes:

        ptx.cp.async_.cg(...)         — async global → SMEM prefetch (16-byte vec)
        ptx.cp.async_.commit_group()  — close pending cp.async into a group
        ptx.cp.async_.wait_group(N)   — wait until <= N groups remain pending
        ptx.mma.sync(shape=(16, 8, 16), ...)        — Ampere tensor-core MMA

    For the SMEM → register hand-off this file uses **per-thread `ld.shared.b32`
    loads** rather than ``ldmatrix``. The fragment-layout math is identical to
    the direct-from-global ``examples/ampere/gemm.py`` — each lane computes its
    own m16n8k16 fragment indices and loads a few packed-bf16 pairs. ldmatrix
    would be more efficient (single warp-collective instruction, hardware-
    optimized bank-conflict handling), but the per-thread path is simpler to
    verify and demonstrates the ``cp.async`` + SMEM-staging path on its own.

    Block tile:  BM x BN = 64 x 64
    K-step:      BK = 16
    Warps/CTA:   4    (warp w handles M[w*16 : (w+1)*16])
    Per warp:    1 (M) × 8 (N) = 8 ``mma.sync`` calls per K-iter
    SMEM stages: 2    (double-buffered ``cp.async`` prefetch)

    Inputs:
      A:   (M, K) bf16 row-major
      B_T: (N, K) bf16 row-major
      D:   (M, N) f32 row-major
    """
    from __future__ import annotations

    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from pyptx import kernel, ptx, reg, smem, Tile
    from pyptx.types import b32, bf16, f32, u32


    BM, BN, BK = 64, 64, 16
    NUM_WARPS = 4
    THREADS = 32 * NUM_WARPS
    WM = BM // NUM_WARPS         # 16
    N_FRAG_N = BN // 8           # 8

    A_STAGE_BYTES = BM * BK * 2  # 2048
    B_STAGE_BYTES = BN * BK * 2  # 2048
    STAGES = 2

    A_SMEM_BASE = 0
    B_SMEM_BASE = STAGES * A_STAGE_BYTES
    SMEM_BYTES = STAGES * (A_STAGE_BYTES + B_STAGE_BYTES)


    def build_gemm_pipelined(M: int, N: int, K: int, *, arch: str = "sm_80"):
        """Build the cp.async + SMEM-staged A100 bf16 GEMM kernel."""
        assert M % BM == 0
        assert N % BN == 0
        assert K % BK == 0
        n_iters = K // BK

        @kernel(
            in_specs=(
                Tile(M, K, bf16),
                Tile(N, K, bf16),
            ),
            out_specs=(Tile(M, N, f32),),
            grid=(N // BN, M // BM, 1),
            block=(THREADS, 1, 1),
            arch=arch,
            smem=SMEM_BYTES,
            extern_smem=True,
        )
        def gemm(A, B_T, D):
            pa, pb, pd = ptx.global_ptrs(A, B_T, D)
            smem_base = smem.base()

            m_base = reg.scalar(u32)
            ptx.inst.mov.u32(m_base, ptx.special.ctaid.y())
            ptx.inst.shl.b32(m_base, m_base, 6)  # * 64
            n_base = reg.scalar(u32)
            ptx.inst.mov.u32(n_base, ptx.special.ctaid.x())
            ptx.inst.shl.b32(n_base, n_base, 6)  # * 64

            tid = reg.scalar(u32)
            ptx.inst.mov.u32(tid, ptx.special.tid.x())
            warp_id = tid >> 5      # 0..3
            lane = tid & 31         # 0..31

            # ----- Per-thread cp.async load layout -----
            # Each thread loads exactly 16 bytes (= 8 bf16) per stage per matrix.
            # 128 threads × 16 bytes = 2048 bytes/stage = matches BM*BK*2 = BN*BK*2.
            # Mapping: thread t loads row t/2, cols (t%2)*8..(t%2)*8+7.
            load_row = tid >> 1                    # 0..63
            col_chunk = tid & 1
            col_start = col_chunk << 3             # 0 or 8
            load_smem_off = (load_row * BK + col_start) * 2  # bytes

            # ----- Accumulator: 8 m16n8 acc tiles × 4 f32 = 32 f32 regs/lane -----
            acc = reg.array(f32, N_FRAG_N * 4)
            zero = reg.scalar(f32, init=0.0)
            for i in range(N_FRAG_N * 4):
                ptx.inst.mov.f32(acc[i], zero)

            def issue_cp_async(s: int, k_idx_reg):
                """Per-thread cp.async for A and B at stage s, K base k_idx_reg."""
                a_smem_dst = smem_base + (A_SMEM_BASE + s * A_STAGE_BYTES) + load_smem_off
                b_smem_dst = smem_base + (B_SMEM_BASE + s * B_STAGE_BYTES) + load_smem_off
                a_global_off = ((m_base + load_row) * K + k_idx_reg + col_start) * 2
                b_global_off = ((n_base + load_row) * K + k_idx_reg + col_start) * 2
                ptx.cp.async_.cg(ptx.addr(a_smem_dst), ptx.addr(pa + a_global_off), 16)
                ptx.cp.async_.cg(ptx.addr(b_smem_dst), ptx.addr(pb + b_global_off), 16)

            # ----- Prologue: prime the pipeline -----
            k_zero = reg.scalar(u32, init=0)
            issue_cp_async(0, k_zero)
            ptx.cp.async_.commit_group()
            if n_iters > 1:
                k_one = reg.scalar(u32, init=BK)
                issue_cp_async(1, k_one)
                ptx.cp.async_.commit_group()

            # ----- m16n8k16 per-lane fragment indices (warp's M slice) -----
            gid = lane >> 2          # 0..7
            tig = lane & 3           # 0..3
            col_lo = tig << 1        # 0,2,4,6 (within each n-frag's 8 cols)
            # Warp's M slice base in SMEM A:
            warp_a_row_base = warp_id << 4   # warp_id * 16 (in SMEM rows)

            # ----- Hoisted fragment registers (reused every K-iter) -----
            # Allocating reg.array inside the K loop creates fresh PTX
            # registers per iter — for K=1024 (64 iters) that's 64*(4+16) =
            # 1280 b32 regs/lane, blowing past spill thresholds in ways that
            # compound subtly. Hoisting keeps the register set bounded.
            a_fr = reg.array(b32, 4)
            b_fr = reg.array(b32, N_FRAG_N * 2)

            # ----- Main K loop -----
            # Pipeline drain: in steady state we wait_group(STAGES-1), keeping
            # one prefetch in flight while the current stage is consumed. In
            # the last STAGES-1 iters no more prefetches are issued, so
            # `pending` decreases each iter and we must lower the threshold —
            # using the steady-state value for the tail would return
            # immediately without actually waiting for the data to land.
            for ki in range(n_iters):
                stage = ki & 1
                tail = max(0, ki - (n_iters - STAGES))
                wait_target = max(0, STAGES - 1 - tail)
                ptx.cp.async_.wait_group(wait_target)
                ptx.bar.sync(0)

                a_stage_base = smem_base + A_SMEM_BASE + stage * A_STAGE_BYTES
                b_stage_base = smem_base + B_SMEM_BASE + stage * B_STAGE_BYTES

                # ---- Load A fragment (4 b32 regs/lane) ----
                for i, (drow, dcol) in enumerate([(0, 0), (8, 0), (0, 8), (8, 8)]):
                    row_in_smem = warp_a_row_base + gid + drow
                    col_pair = col_lo + dcol
                    a_off = (row_in_smem * BK + col_pair) * 2
                    ptx.inst.ld.shared.b32(a_fr[i], ptx.addr(a_stage_base + a_off))

                # ---- Load B fragments (8 m16n8 frags, 2 b32 regs each) ----
                for nf in range(N_FRAG_N):
                    row_in_smem = (nf << 3) + gid
                    for i, dcol in enumerate([0, 8]):
                        col_pair = col_lo + dcol
                        b_off = (row_in_smem * BK + col_pair) * 2
                        ptx.inst.ld.shared.b32(b_fr[nf * 2 + i], ptx.addr(b_stage_base + b_off))

                # ---- Issue 8 mma.sync calls (1 m × 8 n) per warp ----
                for nf in range(N_FRAG_N):
                    ptx.mma.sync(
                        shape=(16, 8, 16),
                        dtype_d=f32, dtype_a=bf16, dtype_b=bf16, dtype_c=f32,
                        d=[acc[nf*4], acc[nf*4+1], acc[nf*4+2], acc[nf*4+3]],
                        a=[a_fr[0], a_fr[1], a_fr[2], a_fr[3]],
                        b=[b_fr[nf*2], b_fr[nf*2+1]],
                        c=[acc[nf*4], acc[nf*4+1], acc[nf*4+2], acc[nf*4+3]],
                        a_layout="row", b_layout="col",
                    )

                # ---- Sync before issuing the next prefetch ----
                # The prefetch writes to the SAME SMEM stage we just read
                # (STAGES=2 ring buffer). Without a bar.sync here, a fast
                # warp could issue its cp.async while a slow warp is still
                # finishing mma — the cp.async writes can land before the
                # slow warp has retired its ld.shared, corrupting in-flight
                # data delivered to mma. The CTA-wide bar.sync forces all
                # warps to complete mma (and the SMEM reads that fed it)
                # before any warp starts overwriting that SMEM stage.
                ptx.bar.sync(0)

                # ---- Prefetch ki+STAGES if there's K left ----
                if ki + STAGES < n_iters:
                    k_next = reg.scalar(u32)
                    ptx.inst.mov.u32(k_next, (ki + STAGES) * BK)
                    next_stage = (ki + STAGES) & 1
                    issue_cp_async(next_stage, k_next)
                    ptx.cp.async_.commit_group()

            # ----- Epilogue: each lane stores its 8 m16n8 acc tiles to global D -----
            # m16n8 fragment per-lane: rows {gid, gid+8}, cols {2*tig, 2*tig+1}
            warp_m_global = m_base + (warp_id << 4)
            row_lo = warp_m_global + gid
            row_hi = row_lo + 8
            for nf in range(N_FRAG_N):
                n_global = n_base + (nf << 3)
                d_col_base = n_global + col_lo
                for i, (drow, dcol) in enumerate([(0, 0), (0, 1), (8, 0), (8, 1)]):
                    row = row_lo if drow == 0 else row_hi
                    col_elem = d_col_base + dcol
                    elem_idx = row * N + col_elem
                    byte_off = elem_idx * 4
                    ptx.inst.st.global_.f32(ptx.addr(pd + byte_off), acc[nf * 4 + i])

            ptx.ret()

        return gemm


    # ---------------------------------------------------------------------------
    # Reference + test harness
    # ---------------------------------------------------------------------------

    def gemm_ref(A: jnp.ndarray, B_T: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("mk,nk->mn", A.astype(jnp.float32), B_T.astype(jnp.float32))


    def _run_jax_case(M: int, N: int, K: int) -> None:
        k = build_gemm_pipelined(M, N, K)
        rng = np.random.default_rng(M * 7919 + N * 31 + K)
        A = jnp.asarray(rng.standard_normal((M, K), dtype=np.float32) * 0.1, dtype=jnp.bfloat16)
        BT = jnp.asarray(rng.standard_normal((N, K), dtype=np.float32) * 0.1, dtype=jnp.bfloat16)

        @jax.jit
        def fn(A, BT):
            return k(A, BT)

        out = np.asarray(fn(A, BT))
        ref = np.asarray(gemm_ref(A, BT))
        diff = float(np.abs(out - ref).max())
        ok = bool(np.allclose(out, ref, atol=1e-2, rtol=1e-2))
        status = "OK  " if ok else "FAIL"
        print(f"[JAX  {status}] M={M:5d} N={N:5d} K={K:5d}  max_abs={diff:.3e}")


    def _run_torch_case(M: int, N: int, K: int) -> None:
        import torch
        k = build_gemm_pipelined(M, N, K)
        rng = np.random.default_rng(M * 7919 + N * 31 + K)
        A = torch.tensor(rng.standard_normal((M, K), dtype=np.float32) * 0.1,
                         dtype=torch.bfloat16, device="cuda")
        BT = torch.tensor(rng.standard_normal((N, K), dtype=np.float32) * 0.1,
                          dtype=torch.bfloat16, device="cuda")
        out = k(A, BT)
        torch.cuda.synchronize()
        ref = (A.float() @ BT.float().T)
        diff = float((out - ref).abs().max())
        ok = bool(torch.allclose(out, ref, atol=1e-2, rtol=1e-2))
        status = "OK  " if ok else "FAIL"
        print(f"[Torch{status}] M={M:5d} N={N:5d} K={K:5d}  max_abs={diff:.3e}")


    def main() -> None:
        _ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()
        for M, N, K in [
            (64, 64, 64),
            (64, 64, 256),
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ]:
            _run_jax_case(M, N, K)
            _run_torch_case(M, N, K)


    if __name__ == "__main__":
        main()
    ```
