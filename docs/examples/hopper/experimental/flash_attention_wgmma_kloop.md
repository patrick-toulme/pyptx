# Hopper / Experimental / Flash Attention Wgmma Kloop

[:material-github: View on GitHub](https://github.com/patrick-toulme/pyptx/blob/dev/examples/hopper/experimental/flash_attention_wgmma_kloop.py){ .md-button } 
[:material-file-code: `examples/hopper/experimental/flash_attention_wgmma_kloop.py`](https://github.com/patrick-toulme/pyptx/blob/dev/examples/hopper/experimental/flash_attention_wgmma_kloop.py){ .md-button }

## Overview

FlashAttention-2 forward with wgmma + TMA + online softmax K-loop.

Tutorial single-CTA version: BM=64, BN=16, HEAD_DIM=16. For the
production Q-tile-parallel kernel with head_dim up to 64 and multi-k
WGMMA (2-5x vs naive PyTorch reference), see ``flash_attention_hopper.py``.

This is the real FA shape. Same algorithm as Triton's FA2 tutorial
(``_attn_fwd_inner``): an outer loop over K/V blocks with a running
per-row max ``m_i``, running per-row sum ``l_i``, and an accumulator
``acc`` that gets rescaled each iteration by ``alpha = exp2(m_i_old
- m_i_new)``.

Per iteration::

    1. TMA-load K^T[:, n*BN:(n+1)*BN] into sK
    2. TMA-load V[n*BN:(n+1)*BN, :]   into sV
    3. qk = wgmma(Q, K^T)                            # f32 frag
    4. qk *= qk_scale                                # folded log2(e) in
    5. m_new = max(m_i, row_max(qk))                 # per-row
    6. alpha = exp2(m_i - m_new)
    7. p = exp2(qk - m_new)                          # element-wise
    8. l_new = l_i * alpha + row_sum(p)
    9. acc = acc * alpha                             # rescale accumulator
   10. Write p to sP, bar.sync
   11. acc += P @ V                                  # per-thread matmul
   12. m_i, l_i := m_new, l_new

Finalize::

    acc /= l_i
    store to O

Uses pyptx DSL sugar throughout: ``ptx.global_ptrs``, ``reg.scalar(f32,
init=<float>)``, ``Reg`` arithmetic operators, ``ptx.warp.reduce_*``
with ``width=4`` for the 4-lane-per-row reductions on the wgmma frag,
and ``sP[row, col] = val`` 2D shared-memory indexing.

Run ``python examples/hopper/experimental/flash_attention_wgmma_kloop.py`` to execute
both a ``jax.jit`` path and a PyTorch eager path against framework
references.

Fragment layout (determined empirically; see the single-block variant
commit for the diagnostic)::

    frag[0, 1] = row_a, cols col_base+0, col_base+1
    frag[2, 3] = row_b, cols col_base+0, col_base+1
    frag[4, 5] = row_a, cols col_base+8, col_base+9
    frag[6, 7] = row_b, cols col_base+8, col_base+9

where ``row_a = 16*warp + lane/4``, ``row_b = row_a + 8``, and
``col_base = 2*(lane%4)``.

## Source

??? example "Full source"

    ```python
    """FlashAttention-2 forward with wgmma + TMA + online softmax K-loop.

    Tutorial single-CTA version: BM=64, BN=16, HEAD_DIM=16. For the
    production Q-tile-parallel kernel with head_dim up to 64 and multi-k
    WGMMA (2-5x vs naive PyTorch reference), see ``flash_attention_hopper.py``.

    This is the real FA shape. Same algorithm as Triton's FA2 tutorial
    (``_attn_fwd_inner``): an outer loop over K/V blocks with a running
    per-row max ``m_i``, running per-row sum ``l_i``, and an accumulator
    ``acc`` that gets rescaled each iteration by ``alpha = exp2(m_i_old
    - m_i_new)``.

    Per iteration::

        1. TMA-load K^T[:, n*BN:(n+1)*BN] into sK
        2. TMA-load V[n*BN:(n+1)*BN, :]   into sV
        3. qk = wgmma(Q, K^T)                            # f32 frag
        4. qk *= qk_scale                                # folded log2(e) in
        5. m_new = max(m_i, row_max(qk))                 # per-row
        6. alpha = exp2(m_i - m_new)
        7. p = exp2(qk - m_new)                          # element-wise
        8. l_new = l_i * alpha + row_sum(p)
        9. acc = acc * alpha                             # rescale accumulator
       10. Write p to sP, bar.sync
       11. acc += P @ V                                  # per-thread matmul
       12. m_i, l_i := m_new, l_new

    Finalize::

        acc /= l_i
        store to O

    Uses pyptx DSL sugar throughout: ``ptx.global_ptrs``, ``reg.scalar(f32,
    init=<float>)``, ``Reg`` arithmetic operators, ``ptx.warp.reduce_*``
    with ``width=4`` for the 4-lane-per-row reductions on the wgmma frag,
    and ``sP[row, col] = val`` 2D shared-memory indexing.

    Run ``python examples/hopper/experimental/flash_attention_wgmma_kloop.py`` to execute
    both a ``jax.jit`` path and a PyTorch eager path against framework
    references.

    Fragment layout (determined empirically; see the single-block variant
    commit for the diagnostic)::

        frag[0, 1] = row_a, cols col_base+0, col_base+1
        frag[2, 3] = row_b, cols col_base+0, col_base+1
        frag[4, 5] = row_a, cols col_base+8, col_base+9
        frag[6, 7] = row_b, cols col_base+8, col_base+9

    where ``row_a = 16*warp + lane/4``, ``row_b = row_a + 8``, and
    ``col_base = 2*(lane%4)``.
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

    # Frag index groups (m64n16 f32 output)
    ROW_A_IDX = (0, 1, 4, 5)
    ROW_B_IDX = (2, 3, 6, 7)


    def build_flash_attention_kloop(N_seq: int, *, sm_scale: float | None = None):
        """Full-K-loop FA kernel. ``N_seq`` is the total KV sequence length and
        must be divisible by BN=16. M (= Q sequence length) is fixed at BM=64
        so the output block fits one wgmma.

        Inputs:
            Q   : (64,  16)       bf16   — one Q block
            K_t : (16, N_seq)     bf16   — K transposed once in JAX (head_dim × seq)
            V   : (N_seq, 16)     bf16   — V in natural row-major layout
        Output:
            O   : (64, 16)        f32
        """
        assert N_seq % BN == 0, f"N_seq={N_seq} must be divisible by BN={BN}"
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(HEAD_DIM)
        qk_scale = sm_scale * LOG2E

        @kernel(
            in_specs=(
                Tile.wgmma_a(BM, HEAD_DIM, bf16),                         # Q
                Tile.wgmma_b(HEAD_DIM, N_seq, bf16, tile_n=BN),           # K^T (head_dim, total_seq)
                Tile.wgmma_b(N_seq, HEAD_DIM, bf16, tile_k=BN, tile_n=HEAD_DIM),  # V — wgmma B for P@V
            ),
            out_specs=(Tile(BM, HEAD_DIM, f32),),
            grid=(1, 1, 1),
            block=(128, 1, 1),
            arch="sm_90a",
        )
        def flash_attn_k(Q, K_t, V, O):
            sQ = smem.wgmma_tile(bf16, (BM, HEAD_DIM), major="K")
            sK = smem.wgmma_tile(bf16, (HEAD_DIM, BN), major="MN")   # one K block
            sP = smem.alloc(b16, BM * BN)                              # P as bf16 for wgmma A
            sV = smem.wgmma_tile(bf16, (BN, HEAD_DIM), major="MN")    # V for wgmma B operand

            bar_q = smem.mbarrier(1)
            bar_k = smem.mbarrier(1)
            phase_q = reg.scalar(b32, init=0)
            phase_k = reg.scalar(b32, init=0)

            # --- running per-row softmax state + accumulator ------------
            m_a = reg.scalar(f32, init=-1e30)
            m_b = reg.scalar(f32, init=-1e30)
            l_a = reg.scalar(f32, init=0.0)
            l_b = reg.scalar(f32, init=0.0)
            zero_f = reg.scalar(f32, init=0.0)
            acc_ab = reg.array(f32, 8)   # row_a × 4 cols + row_b × 4 cols
            for i in range(8):
                ptx.inst.mov.f32(acc_ab[i], zero_f)

            qk_scale_reg = reg.scalar(f32, init=qk_scale)

            # Stage tid into a real register (ptxas rejects special regs).
            tid = reg.scalar(u32)
            ptx.inst.mov.u32(tid, ptx.special.tid.x())

            # --- mbarrier init (all bars) + Q load -----------------------
            with ptx.if_(tid == 0):
                ptx.mbarrier.init(bar_q[0], 1)
                ptx.mbarrier.init(bar_k[0], 1)
                ptx.fence.proxy_async_shared_cta()
                ptx.mbarrier.arrive_expect_tx(bar_q[0], BM * HEAD_DIM * 2)
                ptx.cp.async_.bulk.tensor_2d(
                    dst=sQ[0], src=Q.tma_desc(),
                    coord=(0, 0), mbar=bar_q[0],
                )
            ptx.bar.sync(0)
            ptx.mbarrier.wait(bar_q[0], phase_q)

            # --- row_a / row_b / col_base, computed once per thread -----
            warp_id = tid >> 5            # shr.u32
            lane = reg.scalar(u32)
            ptx.inst.and_.b32(lane, tid, 31)
            row_a = reg.scalar(u32)
            ptx.inst.shl.b32(row_a, warp_id, 4)       # warp*16
            tmp = reg.scalar(u32)
            ptx.inst.shr.u32(tmp, lane, 2)
            ptx.inst.add.u32(row_a, row_a, tmp)       # + lane/4
            row_b = reg.scalar(u32)
            ptx.inst.add.u32(row_b, row_a, 8)
            col_base = reg.scalar(u32)
            ptx.inst.and_.b32(col_base, lane, 3)
            ptx.inst.shl.b32(col_base, col_base, 1)   # (lane&3)*2

            # --- K-loop ---------------------------------------------------
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

                # Q @ K^T[block] via wgmma m64n16k16
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

                # Scale qk_frag by qk_scale (folds log2(e) for ex2)
                for i in range(8):
                    ptx.inst.mul.f32(qk_frag[i], qk_frag[i], qk_scale_reg)

                # Row-wise max: within-thread over {0,1,4,5} and {2,3,6,7},
                # then butterfly-shuffle reduce across the 4 lanes that
                # share a row (width=4).
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

                # m_new = max(m_i_old, row_max_new)
                m_a_new = reg.scalar(f32)
                ptx.inst.max.f32(m_a_new, m_a, row_a_new)
                m_b_new = reg.scalar(f32)
                ptx.inst.max.f32(m_b_new, m_b, row_b_new)

                # alpha = exp2(m_i_old - m_new)
                d_a = reg.scalar(f32)
                ptx.inst.sub.f32(d_a, m_a, m_a_new)
                alpha_a = reg.scalar(f32)
                ptx.inst.ex2.approx.f32(alpha_a, d_a)
                d_b = reg.scalar(f32)
                ptx.inst.sub.f32(d_b, m_b, m_b_new)
                alpha_b = reg.scalar(f32)
                ptx.inst.ex2.approx.f32(alpha_b, d_b)

                # p = exp2(qk - m_new)
                for i in ROW_A_IDX:
                    diff = reg.scalar(f32)
                    ptx.inst.sub.f32(diff, qk_frag[i], m_a_new)
                    ptx.inst.ex2.approx.f32(qk_frag[i], diff)
                for i in ROW_B_IDX:
                    diff = reg.scalar(f32)
                    ptx.inst.sub.f32(diff, qk_frag[i], m_b_new)
                    ptx.inst.ex2.approx.f32(qk_frag[i], diff)

                # row_sum of p
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

                # l_new = l_i * alpha + row_sum
                ptx.inst.fma.rn.f32(l_a, l_a, alpha_a, ra_sum)
                ptx.inst.fma.rn.f32(l_b, l_b, alpha_b, rb_sum)

                # acc = acc * alpha (per-row, hardware fragment layout)
                for i in ROW_A_IDX:
                    ptx.inst.mul.f32(acc_ab[i], acc_ab[i], alpha_a)
                for i in ROW_B_IDX:
                    ptx.inst.mul.f32(acc_ab[i], acc_ab[i], alpha_b)

                # --- store p as bf16 to sP with B32 swizzle for wgmma ---
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

                # --- P @ V via wgmma ----------------------------------
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

            # --- finalize: acc /= l_i, store to O -----------------------
            inv_a = reg.scalar(f32)
            ptx.inst.rcp.approx.f32(inv_a, l_a)
            inv_b = reg.scalar(f32)
            ptx.inst.rcp.approx.f32(inv_b, l_b)
            for i in ROW_A_IDX:
                ptx.inst.mul.f32(acc_ab[i], acc_ab[i], inv_a)
            for i in ROW_B_IDX:
                ptx.inst.mul.f32(acc_ab[i], acc_ab[i], inv_b)

            # Store using hardware fragment layout:
            # [0,1]=row_a col+{0,1}, [2,3]=row_b col+{0,1},
            # [4,5]=row_a col+{8,9}, [6,7]=row_b col+{8,9}
            (po,) = ptx.global_ptrs(O)
            frag_pos = [
                (row_a, 0), (row_a, 1), (row_b, 0), (row_b, 1),
                (row_a, 8), (row_a, 9), (row_b, 8), (row_b, 9),
            ]
            for i, (row, c_off) in enumerate(frag_pos):
                col = reg.scalar(u32)
                ptx.inst.add.u32(col, col_base, c_off)
                off = (row * HEAD_DIM + col) * 4
                ptx.inst.st.global_.f32(ptx.addr(po + off), acc_ab[i])

            ptx.ret()

        return flash_attn_k


    # ---------------------------------------------------------------------------
    # JAX reference + test harness
    # ---------------------------------------------------------------------------

    def attention_ref(q, k, v, sm_scale=None):
        d = q.shape[-1]
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(d)
        s = jnp.matmul(q, k.T) * sm_scale
        p = jax.nn.softmax(s, axis=-1)
        return jnp.matmul(p, v)


    def _torch_attention_ref(q, k, v, sm_scale=None):
        import torch

        d = q.shape[-1]
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(d)
        s = torch.matmul(q, k.transpose(-1, -2)) * sm_scale
        p = torch.softmax(s, dim=-1)
        return torch.matmul(p, v)


    def _run_torch_case(N_seq: int) -> None:
        import torch

        k_fn = build_flash_attention_kloop(N_seq)
        np.random.seed(N_seq)
        q_np = (np.random.randn(BM, HEAD_DIM) * 0.3).astype(np.float32)
        k_np = (np.random.randn(N_seq, HEAD_DIM) * 0.3).astype(np.float32)
        v_np = (np.random.randn(N_seq, HEAD_DIM) * 0.3).astype(np.float32)
        q = torch.tensor(q_np, device="cuda", dtype=torch.bfloat16)
        k = torch.tensor(k_np, device="cuda", dtype=torch.bfloat16)
        v = torch.tensor(v_np, device="cuda", dtype=torch.bfloat16)
        k_t = k.transpose(0, 1).contiguous()

        out = k_fn(q, k_t, v)
        torch.cuda.synchronize()
        ref = _torch_attention_ref(q.float(), k.float(), v.float())
        diff = float((out - ref).abs().max())
        atol = 5e-2 if N_seq <= 64 else 2e-2
        ok = bool(torch.allclose(out, ref, atol=atol, rtol=1e-2))
        status = "OK  " if ok else "FAIL"
        print(f"[Torch{status}] N_seq={N_seq:4d}  max_abs={diff:.3e}")


    def main() -> None:
        _ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()

        for N_seq in [16, 32, 64, 128, 256, 512]:
            k_fn = build_flash_attention_kloop(N_seq)
            np.random.seed(N_seq)
            q_np = (np.random.randn(BM, HEAD_DIM) * 0.3).astype(np.float32)
            k_np = (np.random.randn(N_seq, HEAD_DIM) * 0.3).astype(np.float32)
            v_np = (np.random.randn(N_seq, HEAD_DIM) * 0.3).astype(np.float32)
            q = jnp.asarray(q_np, dtype=jnp.bfloat16)
            k = jnp.asarray(k_np, dtype=jnp.bfloat16)
            v = jnp.asarray(v_np, dtype=jnp.bfloat16)

            # Transpose K once (one-time JAX reshape, no GPU copy after cache warm)
            k_t = jnp.asarray(np.ascontiguousarray(np.asarray(k).T), dtype=jnp.bfloat16)

            @jax.jit
            def fn(q, k_t, v):
                return k_fn(q, k_t, v)

            out = np.asarray(fn(q, k_t, v))
            q_f32 = np.asarray(q, dtype=np.float32)
            k_f32 = np.asarray(k, dtype=np.float32)
            v_f32 = np.asarray(v, dtype=np.float32)
            ref = np.asarray(attention_ref(jnp.asarray(q_f32), jnp.asarray(k_f32), jnp.asarray(v_f32)))

            diff = float(np.abs(out - ref).max())
            atol = 5e-2 if N_seq <= 64 else 2e-2
            ok = bool(np.allclose(out, ref, atol=atol, rtol=1e-2))
            status = "OK  " if ok else "FAIL"
            print(f"[JAX  {status}] N_seq={N_seq:4d}  max_abs={diff:.3e}")
            _run_torch_case(N_seq)


    if __name__ == "__main__":
        main()
    ```
