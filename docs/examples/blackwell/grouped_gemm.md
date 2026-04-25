# Blackwell / Grouped Gemm

[:material-github: View on GitHub](https://github.com/patrick-toulme/pyptx/blob/dev/examples/blackwell/grouped_gemm.py){ .md-button } 
[:material-file-code: `examples/blackwell/grouped_gemm.py`](https://github.com/patrick-toulme/pyptx/blob/dev/examples/blackwell/grouped_gemm.py){ .md-button }

## Overview

Blackwell grouped GEMM using tcgen05.mma.

Uniform-shape grouped GEMM for ``G`` problems sharing ``(M, K) × (K, N)``:

    C[g] = A[g] @ B[g]   for g in range(G)

Shares the warp-specialized 3-stage-pipeline structure with
``gemm_highperf_blackwell.py``. The only addition is a Z dimension on the
launch grid — ``ctaid.z`` picks which group this CTA computes, and the TMA
coords are offset by ``group * M`` (A / C row offset) or ``group * K``
(B row offset) to index into the flattened ``(G*M, K)`` / ``(G*K, N)`` /
``(G*M, N)`` views.

Requirements:

* ``M`` multiple of ``BM = 128`` (the ``tcgen05.mma.kind::f16`` M).
* ``N`` multiple of 8 and ≤ 256 (tcgen05 N range). When ``N ≤ 256`` we use
  a single ``BN = N`` tile per row and the grid's X extent is 1.
* ``K`` multiple of ``BK = 64``.

For MoE-scale shapes (e.g. ``G=8, M=2048, N=128, K=2048``) this matches
what ``torch.nn.functional.grouped_mm`` is targeting.

## Source

??? example "Full source"

    ```python
    """Blackwell grouped GEMM using tcgen05.mma.

    Uniform-shape grouped GEMM for ``G`` problems sharing ``(M, K) × (K, N)``:

        C[g] = A[g] @ B[g]   for g in range(G)

    Shares the warp-specialized 3-stage-pipeline structure with
    ``gemm_highperf_blackwell.py``. The only addition is a Z dimension on the
    launch grid — ``ctaid.z`` picks which group this CTA computes, and the TMA
    coords are offset by ``group * M`` (A / C row offset) or ``group * K``
    (B row offset) to index into the flattened ``(G*M, K)`` / ``(G*K, N)`` /
    ``(G*M, N)`` views.

    Requirements:

    * ``M`` multiple of ``BM = 128`` (the ``tcgen05.mma.kind::f16`` M).
    * ``N`` multiple of 8 and ≤ 256 (tcgen05 N range). When ``N ≤ 256`` we use
      a single ``BN = N`` tile per row and the grid's X extent is 1.
    * ``K`` multiple of ``BK = 64``.

    For MoE-scale shapes (e.g. ``G=8, M=2048, N=128, K=2048``) this matches
    what ``torch.nn.functional.grouped_mm`` is targeting.
    """
    from __future__ import annotations

    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from pyptx import Tile, kernel, ptx, reg, smem
    from pyptx.specs import Layout
    from pyptx.types import b32, b64, bf16, f32, pred, u32, u64


    BM = 128
    BK = 64
    K_PER_INSTR = 16
    MMAS_PER_KTILE = BK // K_PER_INSTR   # = 4
    STAGES = 3

    MMA_DESC_B128 = 0x4000404000010000
    TMA_WARP_TID = 0
    MMA_WARP_TID = 32


    def _pick_bn(n: int) -> int:
        # tcgen05 accepts N in multiples of 8 between 8 and 256.
        # TMA 128B swizzle needs the innermost box dim to be ≥ 64 bf16 elements
        # (128 bytes), so we require N ≥ 64 and a power-of-two multiple of 64
        # up to 256. For N>256 we tile in N.
        assert n % 64 == 0, f"N={n} must be a multiple of 64 (TMA 128B swizzle)"
        assert n >= 64, f"N={n} must be ≥ 64"
        return min(n, 256)


    def build_grouped_gemm(G: int, M: int, N: int, K: int, *, arch: str = "sm_100a"):
        assert arch.startswith("sm_100"), "Blackwell grouped GEMM only on sm_100*"
        assert M % BM == 0, f"M={M} must be divisible by {BM}"
        assert K % BK == 0, f"K={K} must be divisible by {BK}"
        BN = _pick_bn(N)
        assert N % BN == 0, f"N={N} must be divisible by BN={BN}"

        k_iters = K // BK
        A_STAGE = BM * BK * 2
        B_STAGE = BN * BK * 2

        SMEM_A_BASE       = 0
        SMEM_B_BASE       = SMEM_A_BASE + STAGES * A_STAGE
        SMEM_BAR_LOAD     = SMEM_B_BASE + STAGES * B_STAGE
        SMEM_BAR_CONSUMED = SMEM_BAR_LOAD + STAGES * 8
        SMEM_BAR_MMA      = SMEM_BAR_CONSUMED + STAGES * 8
        SMEM_TMEM_SLOT    = SMEM_BAR_MMA + 8
        SMEM_BYTES        = SMEM_TMEM_SLOT + 16

        n_col_shift = {8: 3, 16: 4, 32: 5, 64: 6, 128: 7, 256: 8}[BN]

        # B is transposed per-group at the caller: source is (G*N, K) K-major so
        # that each TMA tile matches the UMMA b_major="K" SMEM layout (BN rows,
        # BK cols, K fast).
        @kernel(
            in_specs=(
                Tile(G * M, K, bf16, Layout.TMA_128B, tma_box=(BM, BK)),
                Tile(G * N, K, bf16, Layout.TMA_128B, tma_box=(BN, BK)),
            ),
            out_specs=(Tile(G * M, N, f32, Layout.ROW),),
            grid=(N // BN, M // BM, G),
            block=(128, 1, 1),
            arch=arch,
            smem=SMEM_BYTES,
            extern_smem=True,
        )
        def grouped_gemm_bw(A, B_T, C):
            base = smem.base()
            tmem_slot = base + SMEM_TMEM_SLOT
            bar_load = base + SMEM_BAR_LOAD
            bar_consumed = base + SMEM_BAR_CONSUMED
            bar_mma = base + SMEM_BAR_MMA

            tid = reg.scalar(u32); ptx.inst.mov.u32(tid, ptx.special.tid.x())
            alloc_warp = reg.scalar(pred); ptx.inst.setp.lt.u32(alloc_warp, tid, 32)
            is_tma_warp = reg.scalar(pred); ptx.inst.setp.eq.u32(is_tma_warp, tid, TMA_WARP_TID)
            is_mma_warp = reg.scalar(pred); ptx.inst.setp.eq.u32(is_mma_warp, tid, MMA_WARP_TID)

            cta_n = reg.scalar(u32); ptx.inst.mov.u32(cta_n, ptx.special.ctaid.x())
            cta_m = reg.scalar(u32); ptx.inst.mov.u32(cta_m, ptx.special.ctaid.y())
            group = reg.scalar(u32); ptx.inst.mov.u32(group, ptx.special.ctaid.z())

            # Global row/col bases into the flattened views.
            #   A[group*M + cta_m*BM .. +BM, ki*BK .. +BK]
            #   B[group*K + ki*BK    .. +BK, cta_n*BN .. +BN]
            #   C[group*M + cta_m*BM .. +BM, cta_n*BN .. +BN]
            m_base = reg.scalar(u32); ptx.inst.mul.lo.u32(m_base, group, M)
            m_cta  = reg.scalar(u32); ptx.inst.shl.b32(m_cta, cta_m, 7)   # *BM(=128)
            ptx.inst.add.u32(m_base, m_base, m_cta)

            # B_T is (G*N, K). Per-group row base = group*N + cta_n*BN.
            n_row_base = reg.scalar(u32); ptx.inst.mul.lo.u32(n_row_base, group, N)
            n_cta = reg.scalar(u32); ptx.inst.shl.b32(n_cta, cta_n, n_col_shift)
            ptx.inst.add.u32(n_row_base, n_row_base, n_cta)

            # C is (G*M, N). Per-CTA col base in N.
            n_col_base = reg.scalar(u32); ptx.inst.shl.b32(n_col_base, cta_n, n_col_shift)

            # idesc with tcgen05 N set to our tile width.
            idesc = reg.scalar(b32, init=ptx.tcgen05.make_instr_desc_f16bf16_f32(m=BM, n=BN))

            with ptx.if_(tid == 0):
                for s in range(STAGES):
                    ptx.mbarrier.init(bar_load + s * 8, 1)
                    ptx.mbarrier.init(bar_consumed + s * 8, 1)
                ptx.mbarrier.init(bar_mma, 1)
                ptx.fence.proxy_async_shared_cta()
            with ptx.if_(alloc_warp):
                ptx.tcgen05.alloc(tmem_slot, 512)
            ptx.bar.sync(0)

            tmem_base = smem.load(b32, ptx.addr(tmem_slot))

            # ── Producer: warp 0 lane 0 issues TMA loads into the ring buffer.
            with ptx.if_(is_tma_warp):
                for ki in range(k_iters):
                    slot = ki % STAGES
                    smem_a = base + SMEM_A_BASE + slot * A_STAGE
                    smem_b = base + SMEM_B_BASE + slot * B_STAGE
                    mbar_l = bar_load + slot * 8
                    mbar_c = bar_consumed + slot * 8

                    if ki >= STAGES:
                        consumed_phase = ((ki // STAGES) - 1) & 1
                        with ptx.scope():
                            ready = reg.scalar(pred)
                            ptx.label(f"cwait_{ki}")
                            ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
                                ready, ptx.addr(mbar_c), consumed_phase
                            )
                            ptx.bra(f"cdone_{ki}", pred=ready)
                            ptx.bra(f"cwait_{ki}")
                            ptx.label(f"cdone_{ki}")

                    ptx.mbarrier.arrive_expect_tx(mbar_l, A_STAGE + B_STAGE)

                    # A coord: (K-offset, row) = (ki*BK, m_base).
                    ptx.cp.async_.bulk.tensor_2d(
                        dst=smem_a, src=A.tma_desc(),
                        coord=(ki * BK, m_base), mbar=mbar_l,
                    )
                    # B_T coord: (K-offset, N-row) = (ki*BK, n_row_base).
                    ptx.cp.async_.bulk.tensor_2d(
                        dst=smem_b, src=B_T.tma_desc(),
                        coord=(ki * BK, n_row_base), mbar=mbar_l,
                    )

            # ── MMA dispatcher: warp 1 lane 0 issues tcgen05.mma.
            with ptx.if_(is_mma_warp):
                for ki in range(k_iters):
                    slot = ki % STAGES
                    smem_a = base + SMEM_A_BASE + slot * A_STAGE
                    smem_b = base + SMEM_B_BASE + slot * B_STAGE
                    mbar_l = bar_load + slot * 8
                    mbar_c = bar_consumed + slot * 8
                    load_phase = (ki // STAGES) & 1

                    with ptx.scope():
                        ready = reg.scalar(pred)
                        ptx.label(f"lwait_{ki}")
                        ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
                            ready, ptx.addr(mbar_l), load_phase
                        )
                        ptx.bra(f"ldone_{ki}", pred=ready)
                        ptx.bra(f"lwait_{ki}")
                        ptx.label(f"ldone_{ki}")

                    desc_a0 = ptx.tcgen05.masked_descriptor(smem_a, const_bits=MMA_DESC_B128)
                    desc_b0 = ptx.tcgen05.masked_descriptor(smem_b, const_bits=MMA_DESC_B128)
                    for kk in range(MMAS_PER_KTILE):
                        if kk == 0:
                            desc_a, desc_b = desc_a0, desc_b0
                        else:
                            desc_a = reg.scalar(b64); desc_b = reg.scalar(b64)
                            ptx.inst.add.s64(desc_a, desc_a0, kk * 2)
                            ptx.inst.add.s64(desc_b, desc_b0, kk * 2)
                        is_first = (ki == 0 and kk == 0)
                        ptx.tcgen05.mma(
                            tmem_base, desc_a, desc_b, idesc,
                            kind="f16", pred_operand=(not is_first),
                        )

                    ptx.mbarrier.arrive(mbar_c)

                ptx.tcgen05.commit(bar_mma, space="cluster")

            # ── All threads: wait MMA done, run epilogue.
            with ptx.scope():
                ready = reg.scalar(pred)
                ptx.label("cw")
                ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
                    ready, ptx.addr(bar_mma), 0
                )
                ptx.bra("cd", pred=ready)
                ptx.bra("cw")
                ptx.label("cd")

            # Output row of C for thread tid: row = group*M + cta_m*BM + tid.
            row_base = reg.scalar(u32); ptx.inst.add.u32(row_base, m_base, tid)
            (pd,) = ptx.global_ptrs(C)
            row_off = reg.scalar(u64); ptx.inst.mul.wide.u32(row_off, row_base, N)
            tile_col = reg.scalar(u64); ptx.inst.cvt.u64.u32(tile_col, n_col_base)
            d_index = row_off + tile_col
            d_ptr = pd + (d_index << 2)

            tmem_row_bits = (tid << 16) & 0x3E00000
            tmem_addr = tmem_base + tmem_row_bits

            # Stream the epilogue in bounded chunks so BN=256 does not force a
            # 256-register live range across the whole TMEM -> GMEM path.
            if BN <= 128:
                out = reg.array(b32, BN)
                ptx.tcgen05.ld(
                    [out[i] for i in range(BN)],
                    tmem_addr, shape="32x32b", count=BN, dtype="b32",
                )
                ptx.tcgen05.wait_ld()

                if BN % 4 == 0:
                    for vec in range(BN // 4):
                        off = vec * 16
                        ptx.inst.st.global_.v4.b32(
                            ptx.addr(d_ptr, off),
                            [out[vec * 4 + i] for i in range(4)],
                        )
                elif BN % 2 == 0:
                    for vec in range(BN // 2):
                        off = vec * 8
                        ptx.inst.st.global_.v2.b32(
                            ptx.addr(d_ptr, off),
                            [out[vec * 2], out[vec * 2 + 1]],
                        )
                else:
                    for i in range(BN):
                        ptx.inst.st.global_.b32(ptx.addr(d_ptr, i * 4), out[i])
            else:
                out = reg.array(b32, 128)
                for chunk in range(BN // 128):
                    chunk_off = chunk * 128
                    ptx.tcgen05.ld(
                        [out[i] for i in range(128)],
                        tmem_addr + chunk_off,
                        shape="32x32b", count=128, dtype="b32",
                    )
                    ptx.tcgen05.wait_ld()

                    for vec in range(128 // 4):
                        off = (chunk_off + vec * 4) * 4
                        ptx.inst.st.global_.v4.b32(
                            ptx.addr(d_ptr, off),
                            [out[vec * 4 + i] for i in range(4)],
                        )

            with ptx.if_(alloc_warp):
                ptx.tcgen05.dealloc(tmem_base, 512)
                ptx.tcgen05.relinquish_alloc_permit()
            ptx.ret()

        return grouped_gemm_bw


    def run_torch(G: int, M: int, N: int, K: int) -> bool:
        import torch

        k_fn = build_grouped_gemm(G, M, N, K)
        np.random.seed(G * 1009 + M * 997 + N * 17 + K)
        A3_np = (np.random.randn(G, M, K) * 0.1).astype(np.float32)
        B3_np = (np.random.randn(G, K, N) * 0.1).astype(np.float32)
        A3 = torch.tensor(A3_np, device="cuda", dtype=torch.bfloat16)
        B3 = torch.tensor(B3_np, device="cuda", dtype=torch.bfloat16)
        A2 = A3.reshape(G * M, K)
        # Transpose per-group: (G, K, N) -> (G, N, K) -> (G*N, K), K-major.
        B_T2 = B3.transpose(1, 2).contiguous().reshape(G * N, K)
        out = k_fn(A2, B_T2).reshape(G, M, N)
        torch.cuda.synchronize()
        ref = torch.einsum("gmk,gkn->gmn", A3.float(), B3.float())
        diff = float((out - ref).abs().max())
        ok = bool(torch.allclose(out, ref, atol=5e-3, rtol=5e-3))
        status = "OK  " if ok else "FAIL"
        print(f"[Torch{status}] G={G:2d} M={M:4d} N={N:4d} K={K:4d}  max_abs={diff:.3e}")
        return ok


    def run_jax(G: int, M: int, N: int, K: int) -> bool:
        k_fn = build_grouped_gemm(G, M, N, K)
        np.random.seed(G * 1009 + M * 997 + N * 17 + K)
        A3_np = (np.random.randn(G, M, K) * 0.1).astype(np.float32)
        B3_np = (np.random.randn(G, K, N) * 0.1).astype(np.float32)
        # Build the K-major (G*N, K) buffer on the host so JAX/XLA can't insert
        # a non-contiguous copy that breaks the TMA descriptor assumptions.
        B_T2_np = np.ascontiguousarray(np.transpose(B3_np, (0, 2, 1))).reshape(G * N, K)
        A2 = jnp.asarray(A3_np.reshape(G * M, K), dtype=jnp.bfloat16)
        B_T2 = jnp.asarray(B_T2_np, dtype=jnp.bfloat16)
        A3 = jnp.asarray(A3_np, dtype=jnp.bfloat16)
        B3 = jnp.asarray(B3_np, dtype=jnp.bfloat16)

        @jax.jit
        def fn(A, B):
            return k_fn(A, B)

        out = np.asarray(fn(A2, B_T2)).reshape(G, M, N)
        ref = np.asarray(jnp.einsum("gmk,gkn->gmn", A3.astype(jnp.float32), B3.astype(jnp.float32)))
        diff = float(np.abs(out - ref).max())
        ok = bool(np.allclose(out, ref, atol=5e-3, rtol=5e-3))
        status = "OK  " if ok else "FAIL"
        print(f"[JAX  {status}] G={G:2d} M={M:4d} N={N:4d} K={K:4d}  max_abs={diff:.3e}")
        return ok


    def main() -> None:
        print("=== Correctness ===")
        # BM=128 requires M%128==0; TMA 128B swizzle requires N%64==0.
        for G, M, N, K in [
            (1, 128, 64, 64),
            (2, 128, 64, 64),
            (4, 128, 64, 128),
            (8, 128, 64, 256),
            (4, 256, 64, 128),
            (8, 512, 128, 512),
            (8, 1024, 128, 1024),
            (4, 2048, 256, 2048),    # MoE-scale
        ]:
            run_torch(G, M, N, K)
            run_jax(G, M, N, K)


    if __name__ == "__main__":
        main()
    ```
