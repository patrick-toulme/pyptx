# Hopper / Gemm Highperf Hopper

[:material-github: View on GitHub](https://github.com/patrick-toulme/pyptx/blob/dev/examples/hopper/gemm_highperf_hopper.py){ .md-button } 
[:material-file-code: `examples/hopper/gemm_highperf_hopper.py`](https://github.com/patrick-toulme/pyptx/blob/dev/examples/hopper/gemm_highperf_hopper.py){ .md-button }

## Overview

Hand-written warp-specialized Hopper GEMM.

Reaches **815 TFLOPS** at 8192x8192 bf16 on H100 SXM5 — beats cuBLAS
(~740 TFLOPS) at matrix sizes >= 6144. This is the maintained
high-performance GEMM example.

Warp-specialized producer/consumer pipeline:

- 3D TMA descriptors for A/B/C
- 2-CTA Hopper clusters with multicast B loads
- 1 producer warpgroup + 2 consumer warpgroups
- m64n256k16 WGMMA (128 f32 regs/thread)
- 3-stage SMEM pipeline (~220 KB dynamic SMEM)
- Hilbert tile schedule in global memory
- TMA store epilogue

Launched through both ``jax.jit`` and the normal PyTorch path.

## Source

??? example "Full source"

    ```python
    """Hand-written warp-specialized Hopper GEMM.

    Reaches **815 TFLOPS** at 8192x8192 bf16 on H100 SXM5 — beats cuBLAS
    (~740 TFLOPS) at matrix sizes >= 6144. This is the maintained
    high-performance GEMM example.

    Warp-specialized producer/consumer pipeline:

    - 3D TMA descriptors for A/B/C
    - 2-CTA Hopper clusters with multicast B loads
    - 1 producer warpgroup + 2 consumer warpgroups
    - m64n256k16 WGMMA (128 f32 regs/thread)
    - 3-stage SMEM pipeline (~220 KB dynamic SMEM)
    - Hilbert tile schedule in global memory
    - TMA store epilogue

    Launched through both ``jax.jit`` and the normal PyTorch path.
    """
    from __future__ import annotations
    import math

    import numpy as np

    from pyptx import kernel, reg, smem, ptx
    from pyptx.specs import Layout, Tile
    from pyptx.types import bf16, b16, b32, f32, u32, s32, pred

    # ── Constants ─────────────────────────────────────────────────────────
    BM, BN, BK = 128, 256, 64
    N_STAGES   = 3
    NUM_SM     = 128
    CLUSTER_M, CLUSTER_N = 2, 1
    CLUSTERS   = CLUSTER_M * CLUSTER_N
    NUM_CLUSTERS = NUM_SM // CLUSTERS
    NUM_CONSUMERS = 2  # wg1 + wg2
    B_WG_M     = BM // NUM_CONSUMERS       # 64: rows per consumer
    B_WG_M_PAD = B_WG_M + 8                # 72: padded for stmatrix
    WGMMA_N    = BN                         # 256
    WGMMA_K    = 16
    SPACE_LEN  = 128
    SCHEDULE_LEN = NUM_CLUSTERS * SPACE_LEN

    # ── SMEM layout (matches SMem<128,256,64,3> in fast.cu) ───────────────
    # A tiles:  BM * BK * 2 * N_STAGES =  128*64*2*3 =  49152 bytes
    # B tiles:  BK * BN * 2 * N_STAGES =  64*256*2*3 =  98304 bytes
    # C buffer: BN * (BM + padding) * 2 = 256*136*2  =  69632 bytes
    # full[3] + empty[3]:  6 * 8         =     48 bytes
    # space[128]:           128 * 4       =    512 bytes
    # Total:                              ~ 217648 bytes
    SMEM_A     = 0
    SMEM_B     = SMEM_A + BM * BK * 2 * N_STAGES        # 49152
    SMEM_C     = SMEM_B + BK * BN * 2 * N_STAGES         # 147456
    SMEM_FULL  = SMEM_C + BN * (BM + (BM // 64) * 8) * 2 # 217088
    SMEM_EMPTY = SMEM_FULL + N_STAGES * 8                 # 217112
    SMEM_SPACE = SMEM_EMPTY + N_STAGES * 8                # 217136
    SMEM_TOTAL = SMEM_SPACE + SPACE_LEN * 4               # 217648

    A_STAGE_BYTES = BM * BK * 2
    B_STAGE_BYTES = BK * BN * 2
    LOAD_TX_BYTES = A_STAGE_BYTES + B_STAGE_BYTES  # 49152


    @kernel(
        arch="sm_90a",
        version=(8, 7),
        in_specs=(
            Tile("M", "K", bf16, Layout.TMA_128B, tma_box=(BM, BK), tma_rank=3),
            Tile("N", "K", bf16, Layout.TMA_128B, tma_box=(BN, BK), tma_rank=3),
            Tile(SCHEDULE_LEN, u32),
        ),
        out_specs=(
            Tile("N", "M", bf16, Layout.ROW, tma_box=(BN, B_WG_M), tma_rank=3, tma_padding=True),
        ),
        grid=(NUM_SM, 1, 1),
        block=(384, 1, 1),
        cluster=(CLUSTERS, 1, 1),
        raw_params=[("u32", "M"), ("u32", "N"), ("u32", "K")],
        extern_smem="smem",
        raw_directives=[
            ("maxntid", (384, 1, 1)),
            ("explicitcluster", ()),
            ("reqnctapercluster", (CLUSTERS, 1, 1)),
        ],
        smem=SMEM_TOTAL,
    )
    def gemm_warp_specialized(A, B, hilbert_schedule, C):
        # ── Load params ───────────────────────────────────────────────────
        K_param = ptx.param(u32, "K")
        tma_A = A.tma_desc()
        tma_B = B.tma_desc()
        tma_C = C.tma_desc()
        (hilbert,) = ptx.global_ptrs(hilbert_schedule)

        # ── Thread/block IDs ──────────────────────────────────────────────
        cluster_rank = reg.scalar(u32)
        ptx.inst.mov.u32(cluster_rank, ptx.sreg("%clusterid.x"))
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        wg_idx = reg.scalar(u32); wg_idx = tid >> 7       # warpgroup 0/1/2
        lane128 = reg.scalar(u32); lane128 = tid & 127     # lane within warpgroup

        # ── Smem base ─────────────────────────────────────────────────────
        smem_base = reg.scalar(u32)
        ptx.inst.mov.u32(smem_base, "smem")
        full = ptx.mbarrier.array(smem_base, SMEM_FULL, N_STAGES)
        empty = ptx.mbarrier.array(smem_base, SMEM_EMPTY, N_STAGES)

        # ── Load Hilbert schedule to SMEM ─────────────────────────────────
        with ptx.if_(tid < SPACE_LEN):
            off = (cluster_rank << 7) + tid
            byte_off = off * 4
            addr = hilbert + byte_off
            val = reg.scalar(u32)
            ptx.inst.ld.global_.u32(val, ptx.addr(addr))
            smem_off = smem_base + (tid << 2)
            ptx.inst.st.shared.u32(ptx.addr(smem_off, SMEM_SPACE), val)

        # ── Compute num_blocks_k ──────────────────────────────────────────
        num_blocks_k = reg.scalar(s32)
        # K / BK (signed division via shift, handling rounding)
        ptx.inst.shr.s32(num_blocks_k, K_param, 6)  # K >> 6 = K / 64

        # ── Mbarrier init (thread 0 only) ─────────────────────────────────
        with ptx.if_(tid == 0):
            full.init_all(1)
            empty.init_all(NUM_CONSUMERS * CLUSTERS)

        # ── Cluster barrier ───────────────────────────────────────────────
        ptx.cluster.sync()

        # ── Cluster CTA rank ──────────────────────────────────────────────
        cta_rank = reg.scalar(u32)
        ptx.inst.mov.u32(cta_rank, ptx.sreg("%cluster_ctarank"))
        rank_m = reg.scalar(u32); rank_m = cta_rank  # CLUSTER_N=1 so rank_m = cta_rank

        # ── Branch: producer (wg0) vs consumer (wg1/wg2) ─────────────────
        ptx.bra("producer_entry", pred=(wg_idx == 0))

        # ══════════════════════════════════════════════════════════════════
        # ██ CONSUMER PATH (wg1 + wg2)
        # ══════════════════════════════════════════════════════════════════
        ptx.setmaxnreg(240, inc=True)
        consumer_id = reg.scalar(u32)
        ptx.inst.sub.u32(consumer_id, wg_idx, 1)  # 0 or 1
        lane_is_cluster = lane128 < CLUSTERS

        # Initial arrive on empty barriers (cross-cluster)
        empty.arrive_remote_all(lane128, 1, pred=lane_is_cluster)

        # ── Persistent tile loop ──────────────────────────────────────────
        # Accumulators: 128 f32 regs for m64n256 wgmma (declared once, reused per tile)
        acc = reg.array(f32, 128)
        acc_frag = acc.hw_order(reverse=True)
        lane_mask = (tid << 5) & 2147479552

        stage_cursor = reg.scalar(u32)
        ptx.inst.mov.u32(stage_cursor, 0)
        phase_full = reg.scalar(u32)
        ptx.inst.mov.u32(phase_full, 0)
        consumer_pipe = ptx.pipeline(N_STAGES, cursor=stage_cursor, phase=phase_full)

        def emit_consumer_quartet(*, first_scale_zero: bool) -> None:
            stage, phase = consumer_pipe.advance()
            full.at(stage).wait(phase)

            ptx.inst.wgmma.fence.sync.aligned()

            a_base = smem_base + (((stage << 13) + lane_mask) << 1)
            b_base = smem_base + (stage << 15)

            a_offsets = (-8192, -8160, -8128, -8096)
            a_masks = (262016, 262112, 262080, 262112)
            b_offsets = (49152, 49184, 49216, 49248)
            b_masks = (262016, 262112, 262080, 262112)
            a_desc = ptx.wgmma.masked_descriptor(
                a_base,
                byte_offset=a_offsets[0],
                mask=a_masks[0],
            )

            for sub_k in range(4):
                b_desc = ptx.wgmma.masked_descriptor(
                    b_base,
                    byte_offset=b_offsets[sub_k],
                    mask=b_masks[sub_k],
                )
                ptx.inst.wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16(
                    acc_frag, a_desc, b_desc,
                    0 if (first_scale_zero and sub_k == 0) else 1,
                    1, 1, 0, 0,
                )

                if sub_k != 3:
                    a_desc = ptx.wgmma.masked_descriptor(
                        a_base,
                        byte_offset=a_offsets[sub_k + 1],
                        mask=a_masks[sub_k + 1],
                    )
            ptx.inst.wgmma.commit_group.sync.aligned()
            ptx.inst.wgmma.wait_group.sync.aligned(0)

            empty.at(stage).arrive_remote(lane128, 1, pred=lane_is_cluster)

        tile_iter = reg.scalar(u32, init=0)
        space_base = smem_base + SMEM_SPACE

        # Epilogue lane/consumer address math is invariant across tiles.
        c_smem_base = smem_base + SMEM_C
        c_consumer_off = consumer_id * (B_WG_M_PAD * BN)
        c_smem_base += c_consumer_off << 1
        rs = [reg.scalar(b16) for _ in range(8)]
        pack = [reg.scalar(b32) for _ in range(4)]
        consumer_bar = ptx.named_barrier(consumer_id + 2, count=128)

        with ptx.loop("consumer_tile_loop"):
            # Load tile index from Hilbert schedule
            space_addr = space_base + (tile_iter << 2)
            tile_packed = reg.scalar(u32)
            ptx.inst.ld.shared.u32(tile_packed, ptx.addr(space_addr))

            # Exit if sentinel (-1)
            ptx.bra("consumer_exit", pred=(tile_packed == -1))

            block_m = tile_packed >> 16
            block_n = tile_packed & 0xFFFF
            # In the fast.cu reference only M is split across the cluster.
            ptx.inst.mad.lo.s32(block_m, block_m, CLUSTER_M, rank_m)

            max_k_quartets = num_blocks_k.max(2)
            remaining_after_first = reg.scalar(s32)
            ptx.inst.add.s32(remaining_after_first, max_k_quartets, -1)

            emit_consumer_quartet(first_scale_zero=True)
            ptx.kloop(
                remaining_after_first,
                unroll=2,
                loop_label="consumer_pair_loop",
                body=lambda: emit_consumer_quartet(first_scale_zero=False),
            )

            # ── Epilogue: f32 → bf16 + stmatrix to SMEM C ────────────────
            ptx.inst.cp.async_.bulk.wait_group(0)
            ptx.stmatrix_x4_trans_f32_bf16(
                frag=acc_frag,
                smem_base=c_smem_base,
                lane=lane128,
                row_stride=B_WG_M_PAD,
                tmp_bf16=rs,
                tmp_pack=pack,
            )

            # Sync within consumer warpgroup before TMA store
            consumer_bar.sync()

            # TMA store C from SMEM to global
            with ptx.if_(lane128 == 0):
                # tensorMapC is built over a physical [N, M] buffer, so the
                # 3D TMA coordinates are (row=N, col=M).
                block_m_bytes = block_m << 7
                store_m = reg.scalar(u32)
                ptx.inst.mad.lo.s32(store_m, consumer_id, B_WG_M, block_m_bytes)
                store_n = block_n << 8
                ptx.tma.store_3d(dst=tma_C, src=c_smem_base, row=store_n, col=store_m)
                ptx.inst.cp.async_.bulk.commit_group()

            # Advance to next tile
            tile_iter += 1

        ptx.label("consumer_exit")
        ptx.ret()

        # ══════════════════════════════════════════════════════════════════
        # ██ PRODUCER PATH (wg0)
        # ══════════════════════════════════════════════════════════════════
        ptx.label("producer_entry")
        ptx.setmaxnreg(24, inc=False)

        with ptx.if_(lane128 == 0):
            p_phase = reg.scalar(u32, init=0)
            p_qidx = reg.scalar(u32, init=0)
            p_tile = reg.scalar(u32, init=0)
            p_space_base = smem_base + SMEM_SPACE
            # Cluster multicast mask: for CLUSTER_M=2, CLUSTER_N=1
            # col_mask = (1 << 0) | (1 << 1) = 3 (both CTAs in cluster)
            col_mask = reg.scalar(b16)
            ptx.inst.mov.u16(col_mask, (1 << CLUSTERS) - 1)

            multicast_issuer = cta_rank == 0
            producer_pipe = ptx.pipeline(N_STAGES, cursor=p_qidx, phase=p_phase)

            def emit_producer_stage(k_coord, a_row_base, b_row_base) -> None:
                stage, phase = producer_pipe.advance()
                empty.at(stage).wait(phase)
                full_slot = full.at(stage)
                full_slot.arrive_expect_tx(LOAD_TX_BYTES)

                a_dst = smem_base + (stage << 14)
                ptx.tma.load_3d(
                    dst=a_dst,
                    src=tma_A,
                    coords=(0, a_row_base, k_coord),
                    mbar=full_slot,
                )

                b_dst = smem_base + SMEM_B + (stage << 15)
                ptx.tma.load_3d_multicast(
                    dst=b_dst,
                    src=tma_B,
                    coords=(0, b_row_base, k_coord),
                    mbar=full_slot,
                    mask=col_mask,
                    issuer=multicast_issuer,
                )

            with ptx.loop("producer_tile_loop"):
                # Load tile index
                p_space = p_space_base + (p_tile << 2)
                p_packed = reg.scalar(u32)
                ptx.inst.ld.shared.u32(p_packed, ptx.addr(p_space))
                ptx.bra("producer_exit", pred=(p_packed == -1))

                tile_m = p_packed >> 16
                a_row_blocks = (tile_m << 1) + rank_m
                a_row = a_row_blocks << 7

                b_row = (p_packed << 8) & 16776960
                p_k = reg.scalar(u32, init=0)

                def emit_producer_k(p_k=p_k, a_row=a_row, b_row=b_row) -> None:
                    emit_producer_stage(p_k, a_row, b_row)
                    p_k += 1

                ptx.kloop(
                    num_blocks_k,
                    unroll=4,
                    loop_label="producer_chunk_loop",
                    body=emit_producer_k,
                )

                p_tile += 1

        ptx.label("producer_exit")
        ptx.ret()


    def _rot(n: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        return x, y


    def _d2xy(n: int, d: int) -> tuple[int, int]:
        x = y = 0
        t = d
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            x, y = _rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return x, y


    def create_hilbert_schedule(m_tiles: int, n_tiles: int) -> np.ndarray:
        """Return the fixed-size per-cluster schedule buffer consumed by the kernel."""
        dim = 1 << ((max(m_tiles, n_tiles) - 1).bit_length())
        space = np.full((NUM_CLUSTERS * SPACE_LEN,), -1, dtype=np.int32)
        total = 0
        pos = [[] for _ in range(NUM_CLUSTERS)]
        core = 0

        for i in range(dim * dim):
            x, y = _d2xy(dim, i)
            if x < m_tiles and y < n_tiles:
                if len(pos[core]) >= SPACE_LEN:
                    raise RuntimeError("Hilbert schedule exceeded SPACE_LEN")
                pos[core].append((x << 16) | y)
                total += 1
                core = (core + 1) % NUM_CLUSTERS

        for i, entries in enumerate(pos):
            base = i * SPACE_LEN
            space[base:base + len(entries)] = np.asarray(entries, dtype=np.int32)

        expected = m_tiles * n_tiles
        if total != expected:
            raise RuntimeError(f"Hilbert schedule mismatch: {total} != {expected}")
        return space


    def gemm_ref(a, b):
        """Reference for the physical output layout used by this kernel."""
        import jax.numpy as jnp

        return (a.astype(jnp.float32) @ b.astype(jnp.float32).T).astype(jnp.bfloat16).T


    def _run_jax_case(size: int) -> None:
        import jax
        import jax.numpy as jnp

        assert size % (BM * CLUSTER_M) == 0
        assert size % BN == 0
        assert size % BK == 0

        rng = np.random.default_rng(size)
        a = jnp.asarray(rng.standard_normal((size, size), dtype=np.float32), dtype=jnp.bfloat16)
        b = jnp.asarray(rng.standard_normal((size, size), dtype=np.float32), dtype=jnp.bfloat16)
        sched = jnp.asarray(
            create_hilbert_schedule(size // (BM * CLUSTER_M), size // BN),
            dtype=jnp.uint32,
        )

        @jax.jit
        def fn(a, b, sched):
            return gemm_warp_specialized(a, b, sched)

        out = fn(a, b, sched).block_until_ready()
        ref = gemm_ref(a, b).block_until_ready()
        diff = float(np.max(np.abs(np.asarray(out, dtype=np.float32) - np.asarray(ref, dtype=np.float32))))
        ok = bool(np.allclose(np.asarray(out, dtype=np.float32), np.asarray(ref, dtype=np.float32), atol=1e-2, rtol=1e-2))
        status = "OK  " if ok else "FAIL"
        print(f"[JAX  {status}] size={size:5d}  max_abs={diff:.3e}")


    def _run_torch_case(size: int) -> None:
        import torch

        assert size % (BM * CLUSTER_M) == 0
        assert size % BN == 0
        assert size % BK == 0

        gen = torch.Generator(device="cuda")
        gen.manual_seed(size)
        a = torch.randn((size, size), device="cuda", dtype=torch.bfloat16, generator=gen)
        b = torch.randn((size, size), device="cuda", dtype=torch.bfloat16, generator=gen)
        sched = torch.tensor(
            create_hilbert_schedule(size // (BM * CLUSTER_M), size // BN),
            device="cuda",
            dtype=torch.int32,
        )

        out = gemm_warp_specialized(a, b, sched)
        torch.cuda.synchronize()
        ref = torch.matmul(a.float(), b.float().T).to(torch.bfloat16).T.contiguous()
        diff = float((out.float() - ref.float()).abs().max().item())
        ok = bool(torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2))
        status = "OK  " if ok else "FAIL"
        print(f"[Torch{status}] size={size:5d}  max_abs={diff:.3e}")


    def main() -> None:
        _run_jax_case(4096)
        _run_torch_case(4096)


    if __name__ == "__main__":
        main()
    ```
