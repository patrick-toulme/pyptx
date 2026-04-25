# Hopper / Experimental / Flash Attention Parallel

[:material-github: View on GitHub](https://github.com/patrick-toulme/pyptx/blob/dev/examples/hopper/experimental/flash_attention_parallel.py){ .md-button } 
[:material-file-code: `examples/hopper/experimental/flash_attention_parallel.py`](https://github.com/patrick-toulme/pyptx/blob/dev/examples/hopper/experimental/flash_attention_parallel.py){ .md-button }

## Overview

Parallelized FlashAttention-2 forward with wgmma + TMA + K-loop.

Production-shape flash attention that scales across Q tiles. Unlike the
single-CTA tutorial ``flash_attention_wgmma_kloop.py``, this kernel
distributes Q tiles across all SMs (``grid = (M_q // BM, 1, 1)``) so
wall-clock time is ~independent of ``M_q`` up to SM saturation (132 on H100).

Algorithm per CTA: identical online softmax FA2 as the tutorial kernel —
Q @ K^T via wgmma m64n16k16, rowmax/rowsum via butterfly shuffle reduce,
P written back to SMEM with B32 swizzle for the second wgmma, P @ V via
wgmma m64n16k16. Difference: we preload KV for stage+1 while crunching
stage's softmax (1-stage lookahead).

Call shape (fixed for wgmma m64n16k16 on bf16):
    Q   : (M_q, 16)    bf16   — M_q divisible by BM=64
    K_t : (16, N_seq)  bf16   — K transposed once in python
    V   : (N_seq, 16)  bf16
    O   : (M_q, 16)    f32

## Source

??? example "Full source"

    ```python
    """Parallelized FlashAttention-2 forward with wgmma + TMA + K-loop.

    Production-shape flash attention that scales across Q tiles. Unlike the
    single-CTA tutorial ``flash_attention_wgmma_kloop.py``, this kernel
    distributes Q tiles across all SMs (``grid = (M_q // BM, 1, 1)``) so
    wall-clock time is ~independent of ``M_q`` up to SM saturation (132 on H100).

    Algorithm per CTA: identical online softmax FA2 as the tutorial kernel —
    Q @ K^T via wgmma m64n16k16, rowmax/rowsum via butterfly shuffle reduce,
    P written back to SMEM with B32 swizzle for the second wgmma, P @ V via
    wgmma m64n16k16. Difference: we preload KV for stage+1 while crunching
    stage's softmax (1-stage lookahead).

    Call shape (fixed for wgmma m64n16k16 on bf16):
        Q   : (M_q, 16)    bf16   — M_q divisible by BM=64
        K_t : (16, N_seq)  bf16   — K transposed once in python
        V   : (N_seq, 16)  bf16
        O   : (M_q, 16)    f32
    """
    from __future__ import annotations

    import math

    import jax
    import jax.numpy as jnp
    import numpy as np

    from pyptx import kernel, reg, smem, ptx, Tile, Layout
    from pyptx.smem import apply_swizzle
    from pyptx.types import bf16, f32, b16, b32, u32, pred


    BM = 64
    BN = 16
    HEAD_DIM = 16
    LOG2E = 1.4426950408889634

    ROW_A_IDX = (0, 1, 4, 5)
    ROW_B_IDX = (2, 3, 6, 7)


    def build_flash_attention_parallel(M_q: int, N_seq: int,
                                       *, sm_scale: float | None = None):
        """Q-tile-parallel FA kernel. M_q Q rows, N_seq KV rows.

        Each CTA owns one 64-row Q block and iterates all KV.

        M_q divisible by BM=64. N_seq divisible by BN=16.
        """
        assert M_q % BM == 0, f"M_q={M_q} must be divisible by BM={BM}"
        assert N_seq % BN == 0, f"N_seq={N_seq} must be divisible by BN={BN}"
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(HEAD_DIM)
        qk_scale = sm_scale * LOG2E

        @kernel(
            in_specs=(
                Tile.wgmma_a(M_q, HEAD_DIM, bf16, tile_m=BM),
                Tile.wgmma_b(HEAD_DIM, N_seq, bf16, tile_n=BN),
                Tile.wgmma_b(N_seq, HEAD_DIM, bf16, tile_k=BN, tile_n=HEAD_DIM),
            ),
            out_specs=(Tile(M_q, HEAD_DIM, f32),),
            grid=(M_q // BM, 1, 1),
            block=(128, 1, 1),
            arch="sm_90a",
        )
        def flash_attn(Q, K_t, V, O):
            sQ = smem.wgmma_tile(bf16, (BM, HEAD_DIM), major="K")
            # Double-buffered K/V for 1-stage lookahead.
            sK = smem.wgmma_tile(bf16, (HEAD_DIM, BN), major="MN")
            sV = smem.wgmma_tile(bf16, (BN, HEAD_DIM), major="MN")
            sP = smem.alloc(b16, BM * BN)

            bar_q = smem.mbarrier(1)
            bar_k = smem.mbarrier(1)
            phase_q = reg.scalar(b32, init=0)
            phase_k = reg.scalar(b32, init=0)

            # Softmax state + accumulator (double-row fragment layout).
            m_a = reg.scalar(f32, init=-1e30)
            m_b = reg.scalar(f32, init=-1e30)
            l_a = reg.scalar(f32, init=0.0)
            l_b = reg.scalar(f32, init=0.0)
            zero_f = reg.scalar(f32, init=0.0)
            acc_ab = reg.array(f32, 8)
            for i in range(8):
                ptx.inst.mov.f32(acc_ab[i], zero_f)

            qk_scale_reg = reg.scalar(f32, init=qk_scale)

            tid = reg.scalar(u32)
            ptx.inst.mov.u32(tid, ptx.special.tid.x())
            q_row_base = reg.scalar(u32)
            ptx.inst.mov.u32(q_row_base, ptx.special.ctaid.x())
            ptx.inst.shl.b32(q_row_base, q_row_base, 6)  # * 64

            # Init mbars + kick off Q load.
            with ptx.if_(tid == 0):
                ptx.mbarrier.init(bar_q[0], 1)
                ptx.mbarrier.init(bar_k[0], 1)
                ptx.fence.proxy_async_shared_cta()
                ptx.mbarrier.arrive_expect_tx(bar_q[0], BM * HEAD_DIM * 2)
                ptx.cp.async_.bulk.tensor_2d(
                    dst=sQ[0], src=Q.tma_desc(),
                    coord=(0, q_row_base), mbar=bar_q[0],
                )
            ptx.bar.sync(0)
            ptx.mbarrier.wait(bar_q[0], phase_q)

            # Thread coords used throughout.
            warp_id = tid >> 5
            lane = reg.scalar(u32)
            ptx.inst.and_.b32(lane, tid, 31)
            row_a = reg.scalar(u32)
            ptx.inst.shl.b32(row_a, warp_id, 4)
            tmp = reg.scalar(u32)
            ptx.inst.shr.u32(tmp, lane, 2)
            ptx.inst.add.u32(row_a, row_a, tmp)
            row_b = reg.scalar(u32)
            ptx.inst.add.u32(row_b, row_a, 8)
            col_base = reg.scalar(u32)
            ptx.inst.and_.b32(col_base, lane, 3)
            ptx.inst.shl.b32(col_base, col_base, 1)

            # K loop.
            k_col = reg.scalar(u32, init=0)
            keep_going = reg.scalar(pred)
            ptx.inst.setp.lt.u32(keep_going, k_col, N_seq)
            with ptx.loop("kv_loop", pred=keep_going):
                with ptx.if_(tid == 0):
                    ptx.mbarrier.arrive_expect_tx(
                        bar_k[0], HEAD_DIM * BN * 2 + BN * HEAD_DIM * 2,
                    )
                    ptx.cp.async_.bulk.tensor_2d(
                        dst=sK[0], src=K_t.tma_desc(),
                        coord=(k_col, 0), mbar=bar_k[0],
                    )
                    ptx.cp.async_.bulk.tensor_2d(
                        dst=sV[0], src=V.tma_desc(),
                        coord=(0, k_col), mbar=bar_k[0],
                    )
                ptx.bar.sync(0)
                ptx.mbarrier.wait(bar_k[0], phase_k)
                phase_k ^= 1

                qk_frag = reg.array(f32, 8)
                ptx.wgmma.fence()
                ptx.wgmma.mma_async(
                    shape=(64, 16, 16),
                    dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                    d=qk_frag, a=sQ, b=sK,
                    scale_d=False, trans_a=0, trans_b=1,
                )
                ptx.wgmma.commit_group()
                ptx.wgmma.wait_group(0)

                for i in range(8):
                    ptx.inst.mul.f32(qk_frag[i], qk_frag[i], qk_scale_reg)

                row_a_new = reg.scalar(f32)
                ptx.inst.max.f32(row_a_new, qk_frag[ROW_A_IDX[0]], qk_frag[ROW_A_IDX[1]])
                for i in ROW_A_IDX[2:]:
                    ptx.inst.max.f32(row_a_new, row_a_new, qk_frag[i])
                row_b_new = reg.scalar(f32)
                ptx.inst.max.f32(row_b_new, qk_frag[ROW_B_IDX[0]], qk_frag[ROW_B_IDX[1]])
                for i in ROW_B_IDX[2:]:
                    ptx.inst.max.f32(row_b_new, row_b_new, qk_frag[i])
                ptx.warp.reduce_max(row_a_new, width=4)
                ptx.warp.reduce_max(row_b_new, width=4)

                m_a_new = reg.scalar(f32)
                ptx.inst.max.f32(m_a_new, m_a, row_a_new)
                m_b_new = reg.scalar(f32)
                ptx.inst.max.f32(m_b_new, m_b, row_b_new)

                d_a = reg.scalar(f32)
                ptx.inst.sub.f32(d_a, m_a, m_a_new)
                alpha_a = reg.scalar(f32)
                ptx.inst.ex2.approx.f32(alpha_a, d_a)
                d_b = reg.scalar(f32)
                ptx.inst.sub.f32(d_b, m_b, m_b_new)
                alpha_b = reg.scalar(f32)
                ptx.inst.ex2.approx.f32(alpha_b, d_b)

                for i in ROW_A_IDX:
                    diff = reg.scalar(f32)
                    ptx.inst.sub.f32(diff, qk_frag[i], m_a_new)
                    ptx.inst.ex2.approx.f32(qk_frag[i], diff)
                for i in ROW_B_IDX:
                    diff = reg.scalar(f32)
                    ptx.inst.sub.f32(diff, qk_frag[i], m_b_new)
                    ptx.inst.ex2.approx.f32(qk_frag[i], diff)

                ra_sum = reg.scalar(f32)
                ptx.inst.add.f32(ra_sum, qk_frag[ROW_A_IDX[0]], qk_frag[ROW_A_IDX[1]])
                for i in ROW_A_IDX[2:]:
                    ptx.inst.add.f32(ra_sum, ra_sum, qk_frag[i])
                rb_sum = reg.scalar(f32)
                ptx.inst.add.f32(rb_sum, qk_frag[ROW_B_IDX[0]], qk_frag[ROW_B_IDX[1]])
                for i in ROW_B_IDX[2:]:
                    ptx.inst.add.f32(rb_sum, rb_sum, qk_frag[i])
                ptx.warp.reduce_sum(ra_sum, width=4)
                ptx.warp.reduce_sum(rb_sum, width=4)

                ptx.inst.fma.rn.f32(l_a, l_a, alpha_a, ra_sum)
                ptx.inst.fma.rn.f32(l_b, l_b, alpha_b, rb_sum)

                for i in ROW_A_IDX:
                    ptx.inst.mul.f32(acc_ab[i], acc_ab[i], alpha_a)
                for i in ROW_B_IDX:
                    ptx.inst.mul.f32(acc_ab[i], acc_ab[i], alpha_b)

                sP_base = reg.scalar(u32)
                ptx.inst.mov.b32(sP_base, sP.name)
                row_a_byte = reg.scalar(u32)
                ptx.inst.mul.lo.u32(row_a_byte, row_a, BN * 2)
                row_b_byte = reg.scalar(u32)
                ptx.inst.mul.lo.u32(row_b_byte, row_b, BN * 2)
                col_byte = reg.scalar(u32)
                ptx.inst.shl.b32(col_byte, col_base, 1)
                for fi, (is_b, c_off) in enumerate([
                    (False, 0), (False, 2),
                    (True, 0), (True, 2),
                    (False, 16), (False, 18),
                    (True, 16), (True, 18),
                ]):
                    logical = reg.scalar(u32)
                    ptx.inst.add.u32(logical, row_b_byte if is_b else row_a_byte, col_byte)
                    if c_off:
                        ptx.inst.add.u32(logical, logical, c_off)
                    physical = apply_swizzle(logical, "32B")
                    st_addr = reg.scalar(u32)
                    ptx.inst.add.u32(st_addr, sP_base, physical)
                    bf_val = reg.scalar(b16)
                    ptx.inst.cvt.rn.bf16.f32(bf_val, qk_frag[fi])
                    ptx.inst.st.shared.b16(ptx.addr(st_addr), bf_val)
                ptx.bar.sync(0)

                desc_p = ptx.wgmma.auto_descriptor(
                    sP_base, dtype=bf16, shape=(BM, BN), major="K",
                )
                ptx.wgmma.fence()
                ptx.wgmma.mma_async(
                    shape=(64, HEAD_DIM, BN),
                    dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                    d=acc_ab, a=desc_p, b=sV,
                    scale_d=True, trans_a=0, trans_b=1,
                )
                ptx.wgmma.commit_group()
                ptx.wgmma.wait_group(0)
                ptx.bar.sync(0)

                ptx.inst.mov.f32(m_a, m_a_new)
                ptx.inst.mov.f32(m_b, m_b_new)
                k_col += BN
                ptx.inst.setp.lt.u32(keep_going, k_col, N_seq)

            # Finalize: acc /= l, store O at (q_row_base + frag_row).
            inv_a = reg.scalar(f32)
            ptx.inst.rcp.approx.f32(inv_a, l_a)
            inv_b = reg.scalar(f32)
            ptx.inst.rcp.approx.f32(inv_b, l_b)
            for i in ROW_A_IDX:
                ptx.inst.mul.f32(acc_ab[i], acc_ab[i], inv_a)
            for i in ROW_B_IDX:
                ptx.inst.mul.f32(acc_ab[i], acc_ab[i], inv_b)

            (po,) = ptx.global_ptrs(O)
            row_a_global = reg.scalar(u32)
            ptx.inst.add.u32(row_a_global, row_a, q_row_base)
            row_b_global = reg.scalar(u32)
            ptx.inst.add.u32(row_b_global, row_b, q_row_base)
            frag_pos = [
                (row_a_global, 0), (row_a_global, 1),
                (row_b_global, 0), (row_b_global, 1),
                (row_a_global, 8), (row_a_global, 9),
                (row_b_global, 8), (row_b_global, 9),
            ]
            for i, (row, c_off) in enumerate(frag_pos):
                col = reg.scalar(u32)
                ptx.inst.add.u32(col, col_base, c_off)
                off = (row * HEAD_DIM + col) * 4
                ptx.inst.st.global_.f32(ptx.addr(po + off), acc_ab[i])

            ptx.ret()

        return flash_attn


    # ---------------------------------------------------------------------------
    # References + test harness
    # ---------------------------------------------------------------------------

    def attention_ref(q, k, v, sm_scale=None):
        d = q.shape[-1]
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(d)
        s = jnp.matmul(q, k.T) * sm_scale
        p = jax.nn.softmax(s, axis=-1)
        return jnp.matmul(p, v)


    def _run_torch_case(M_q: int, N_seq: int) -> None:
        import torch

        k_fn = build_flash_attention_parallel(M_q, N_seq)
        torch.manual_seed(M_q * 131 + N_seq)
        q = (torch.randn(M_q, HEAD_DIM, device="cuda") * 0.3).to(torch.bfloat16)
        k = (torch.randn(N_seq, HEAD_DIM, device="cuda") * 0.3).to(torch.bfloat16)
        v = (torch.randn(N_seq, HEAD_DIM, device="cuda") * 0.3).to(torch.bfloat16)
        k_t = k.T.contiguous()
        out = k_fn(q, k_t, v)
        torch.cuda.synchronize()
        qf, kf, vf = q.float(), k.float(), v.float()
        ref = torch.softmax(qf @ kf.T / math.sqrt(HEAD_DIM), dim=-1) @ vf
        diff = float((out - ref).abs().max())
        atol = 5e-2 if N_seq <= 64 else 2e-2
        ok = bool(torch.allclose(out, ref, atol=atol, rtol=1e-2))
        status = "OK  " if ok else "FAIL"
        print(f"[Torch{status}] M_q={M_q:5d} N={N_seq:5d}  max_abs={diff:.3e}")


    def _run_jax_case(M_q: int, N_seq: int) -> None:
        k_fn = build_flash_attention_parallel(M_q, N_seq)
        np.random.seed(M_q * 131 + N_seq)
        q_np = (np.random.randn(M_q, HEAD_DIM) * 0.3).astype(np.float32)
        k_np = (np.random.randn(N_seq, HEAD_DIM) * 0.3).astype(np.float32)
        v_np = (np.random.randn(N_seq, HEAD_DIM) * 0.3).astype(np.float32)
        q = jnp.asarray(q_np, dtype=jnp.bfloat16)
        k = jnp.asarray(k_np, dtype=jnp.bfloat16)
        v = jnp.asarray(v_np, dtype=jnp.bfloat16)
        k_t = jnp.asarray(np.ascontiguousarray(np.asarray(k).T), dtype=jnp.bfloat16)

        @jax.jit
        def fn(q, k_t, v):
            return k_fn(q, k_t, v)

        out = np.asarray(fn(q, k_t, v))
        ref = np.asarray(attention_ref(jnp.asarray(q_np), jnp.asarray(k_np), jnp.asarray(v_np)))
        diff = float(np.abs(out - ref).max())
        atol = 5e-2 if N_seq <= 64 else 2e-2
        ok = bool(np.allclose(out, ref, atol=atol, rtol=1e-2))
        status = "OK  " if ok else "FAIL"
        print(f"[JAX  {status}] M_q={M_q:5d} N={N_seq:5d}  max_abs={diff:.3e}")


    def main() -> None:
        _ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()
        for M_q, N_seq in [
            (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024),
            (2048, 2048), (4096, 4096),
        ]:
            _run_jax_case(M_q, N_seq)
            _run_torch_case(M_q, N_seq)


    if __name__ == "__main__":
        main()
    ```
