# Blackwell / Gemm Highperf Blackwell

[:material-github: View on GitHub](https://github.com/patrick-toulme/pyptx/blob/dev/examples/blackwell/gemm_highperf_blackwell.py){ .md-button } 
[:material-file-code: `examples/blackwell/gemm_highperf_blackwell.py`](https://github.com/patrick-toulme/pyptx/blob/dev/examples/blackwell/gemm_highperf_blackwell.py){ .md-button }

## Overview

Blackwell tcgen05 GEMM, callable from JAX and PyTorch.

Production ``sm_100a`` GEMM with TMA loads + ``tcgen05.mma`` + ``tcgen05.ld``
epilogue. Closely mirrors the JAX Pallas ``blackwell_matmul_mgpu.py`` reference:

- Warp-specialized: warp 0 lane 0 issues TMA loads; warp 1 lane 0 dispatches
  ``tcgen05.mma``.
- 3-stage SMEM ring buffer so TMA for k-tile ``ki+2`` can overlap MMA for
  ``ki``.
- ``consumed`` barriers let the producer know when the MMA warp has finished
  a slot and the SMEM can be overwritten.
- Single commit after the last K-tile, then TMEM → regs → ``st.global.v4``.

Supports arbitrary ``M``, ``N`` (multiples of BM/BN) and ``K`` (multiple of BK).

## Source

??? example "Full source"

    ```python
    """Blackwell tcgen05 GEMM, callable from JAX and PyTorch.

    Production ``sm_100a`` GEMM with TMA loads + ``tcgen05.mma`` + ``tcgen05.ld``
    epilogue. Closely mirrors the JAX Pallas ``blackwell_matmul_mgpu.py`` reference:

    - Warp-specialized: warp 0 lane 0 issues TMA loads; warp 1 lane 0 dispatches
      ``tcgen05.mma``.
    - 3-stage SMEM ring buffer so TMA for k-tile ``ki+2`` can overlap MMA for
      ``ki``.
    - ``consumed`` barriers let the producer know when the MMA warp has finished
      a slot and the SMEM can be overwritten.
    - Single commit after the last K-tile, then TMEM → regs → ``st.global.v4``.

    Supports arbitrary ``M``, ``N`` (multiples of BM/BN) and ``K`` (multiple of BK).
    """
    from __future__ import annotations

    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from pyptx import Tile, kernel, ptx, reg, smem
    from pyptx.specs import Layout
    from pyptx.types import b32, b64, bf16, f32, pred, u32, u64


    BM = 128
    BN = 256
    BK = 64
    K_PER_INSTR = 16
    MMAS_PER_KTILE = BK // K_PER_INSTR   # = 4
    STAGES = 4                           # 1SM SMEM ring-buffer depth
    STAGES_2SM = 5                       # 2SM ring-buffer depth tuned on B200

    A_STAGE = BM * BK * 2                # 16 KB
    B_STAGE = BN * BK * 2                # 32 KB

    SMEM_A_BASE      = 0
    SMEM_B_BASE      = SMEM_A_BASE + STAGES * A_STAGE
    SMEM_BAR_LOAD    = SMEM_B_BASE + STAGES * B_STAGE
    SMEM_BAR_CONSUMED= SMEM_BAR_LOAD + STAGES * 8
    SMEM_BAR_MMA     = SMEM_BAR_CONSUMED + STAGES * 8
    SMEM_TMEM_SLOT   = SMEM_BAR_MMA + 8
    SMEM_BYTES       = SMEM_TMEM_SLOT + 16

    MMA_DESC_B128 = 0x4000404000010000

    TMA_WARP_TID = 0     # warp 0 lane 0
    MMA_WARP_TID = 32    # warp 1 lane 0
    GRID_MINOR_DIM = "n"
    GRID_TILE_WIDTH = 16
    EPILOGUE_TILE_N = 64


    def _detect_sm_count(default: int = 132) -> int:
        try:
            import torch
            if torch.cuda.is_available():
                return int(torch.cuda.get_device_properties(0).multi_processor_count)
        except Exception:
            pass
        return default


    def build_gemm(
        M: int,
        N: int,
        K: int,
        *,
        arch: str = "sm_100a",
        stages: int = STAGES,
        grid_minor_dim: str = GRID_MINOR_DIM,
        grid_tile_width: int = GRID_TILE_WIDTH,
    ):
        assert arch.startswith("sm_100"), "Blackwell GEMM is only implemented for sm_100*"
        assert M % BM == 0, f"M={M} must be divisible by {BM}"
        assert N % BN == 0, f"N={N} must be divisible by {BN}"
        assert K % BK == 0, f"K={K} must be divisible by {BK}"
        assert stages >= 2, "Blackwell GEMM requires at least two stages"
        assert grid_tile_width >= 1, "grid_tile_width must be at least 1"
        k_iters = K // BK
        tile_m_iters = M // BM
        tile_n_iters = N // BN
        minor_dim = grid_minor_dim.lower()
        assert minor_dim in {"m", "n"}, "grid_minor_dim must be 'm' or 'n'"
        minor_is_n = minor_dim == "n"
        major_iters = tile_m_iters if minor_is_n else tile_n_iters
        minor_iters = tile_n_iters if minor_is_n else tile_m_iters
        group_span = major_iters * grid_tile_width
        has_partial_group = (minor_iters % grid_tile_width) != 0

        smem_b_base = SMEM_A_BASE + stages * A_STAGE
        smem_bar_load = smem_b_base + stages * B_STAGE
        smem_bar_consumed = smem_bar_load + stages * 8
        smem_bar_mma = smem_bar_consumed + stages * 8
        smem_tmem_slot = smem_bar_mma + 8
        smem_bytes = smem_tmem_slot + 16

        @kernel(
            in_specs=(
                Tile(M, K, bf16, Layout.TMA_128B, tma_box=(BM, BK)),
                Tile(N, K, bf16, Layout.TMA_128B, tma_box=(BN, BK)),
            ),
            out_specs=(Tile(M, N, f32, Layout.ROW),),
            grid=(N // BN, M // BM, 1),
            block=(128, 1, 1),
            arch=arch,
            smem=smem_bytes,
            extern_smem=True,
        )
        def blackwell_gemm(A, B_T, D):
            base = smem.base()
            tmem_slot = base + smem_tmem_slot
            bar_load = base + smem_bar_load
            bar_consumed = base + smem_bar_consumed
            bar_mma = base + smem_bar_mma

            tid = reg.scalar(u32); ptx.inst.mov.u32(tid, ptx.special.tid.x())
            alloc_warp = reg.scalar(pred); ptx.inst.setp.lt.u32(alloc_warp, tid, 32)
            is_tma_warp = reg.scalar(pred); ptx.inst.setp.eq.u32(is_tma_warp, tid, TMA_WARP_TID)
            is_mma_warp = reg.scalar(pred); ptx.inst.setp.eq.u32(is_mma_warp, tid, MMA_WARP_TID)

            grid_x = reg.scalar(u32); ptx.inst.mov.u32(grid_x, ptx.special.ctaid.x())
            grid_y = reg.scalar(u32); ptx.inst.mov.u32(grid_y, ptx.special.ctaid.y())
            tile_linear = reg.scalar(u32)
            ptx.inst.mad.lo.u32(tile_linear, grid_y, tile_n_iters, grid_x)

            tile_major = reg.scalar(u32)
            tile_minor = reg.scalar(u32)
            group_id = reg.scalar(u32)
            group_offset = reg.scalar(u32)
            minor_in_group = reg.scalar(u32)
            tile_start = reg.scalar(u32)
            ptx.inst.div.u32(group_id, tile_linear, group_span)
            ptx.inst.rem.u32(group_offset, tile_linear, group_span)
            ptx.inst.mul.lo.u32(tile_start, group_id, grid_tile_width)

            if has_partial_group:
                remaining = reg.scalar(u32)
                group_width = reg.scalar(u32)
                full_group = reg.scalar(pred)
                ptx.inst.sub.u32(remaining, minor_iters, tile_start)
                ptx.inst.setp.ge.u32(full_group, remaining, grid_tile_width)
                ptx.inst.selp.b32(group_width, grid_tile_width, remaining, full_group)
            else:
                group_width = reg.scalar(u32, init=grid_tile_width)

            ptx.inst.div.u32(tile_major, group_offset, group_width)
            ptx.inst.rem.u32(minor_in_group, group_offset, group_width)

            minor_fwd = reg.scalar(u32)
            minor_rev = reg.scalar(u32)
            minor_rev_off = reg.scalar(u32)
            group_width_m1 = reg.scalar(u32)
            odd_bit = reg.scalar(u32)
            is_odd_group = reg.scalar(pred)
            ptx.inst.add.u32(minor_fwd, tile_start, minor_in_group)
            ptx.inst.sub.u32(group_width_m1, group_width, 1)
            ptx.inst.sub.u32(minor_rev_off, group_width_m1, minor_in_group)
            ptx.inst.add.u32(minor_rev, tile_start, minor_rev_off)
            ptx.inst.and_.b32(odd_bit, group_id, 1)
            ptx.inst.setp.ne.u32(is_odd_group, odd_bit, 0)
            ptx.inst.selp.b32(tile_minor, minor_rev, minor_fwd, is_odd_group)

            tile_m = reg.scalar(u32)
            tile_n = reg.scalar(u32)
            if minor_is_n:
                ptx.inst.mov.u32(tile_m, tile_major)
                ptx.inst.mov.u32(tile_n, tile_minor)
            else:
                ptx.inst.mov.u32(tile_m, tile_minor)
                ptx.inst.mov.u32(tile_n, tile_major)

            m_base = tile_m << 7
            n_base = tile_n << 8

            idesc = reg.scalar(b32, init=ptx.tcgen05.make_instr_desc_f16bf16_f32())

            # Init all mbarriers; allocate TMEM.
            with ptx.if_(tid == 0):
                for s in range(stages):
                    ptx.mbarrier.init(bar_load + s * 8, 1)
                    ptx.mbarrier.init(bar_consumed + s * 8, 1)
                ptx.mbarrier.init(bar_mma, 1)
                ptx.fence.proxy_async_shared_cta()
            with ptx.if_(alloc_warp):
                ptx.tcgen05.alloc(tmem_slot, 512)
            ptx.bar.sync(0)

            tmem_base = smem.load(b32, ptx.addr(tmem_slot))

            # ── Producer (warp 0 lane 0): issue TMA loads, ring-buffered ──
            with ptx.if_(is_tma_warp):
                for ki in range(k_iters):
                    slot = ki % stages
                    smem_a = base + SMEM_A_BASE + slot * A_STAGE
                    smem_b = base + smem_b_base + slot * B_STAGE
                    mbar_l = bar_load + slot * 8
                    mbar_c = bar_consumed + slot * 8

                    if ki >= stages:
                        # Slot reuse: wait for MMA to finish previous iter.
                        consumed_phase = ((ki // stages) - 1) & 1
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
                    ptx.cp.async_.bulk.tensor_2d(
                        dst=smem_a, src=A.tma_desc(),
                        coord=(ki * BK, m_base), mbar=mbar_l,
                    )
                    ptx.cp.async_.bulk.tensor_2d(
                        dst=smem_b, src=B_T.tma_desc(),
                        coord=(ki * BK, n_base), mbar=mbar_l,
                    )

            # ── MMA dispatcher (warp 1 lane 0): wait loads, issue MMAs ──
            with ptx.if_(is_mma_warp):
                for ki in range(k_iters):
                    slot = ki % stages
                    smem_a = base + SMEM_A_BASE + slot * A_STAGE
                    smem_b = base + smem_b_base + slot * B_STAGE
                    mbar_l = bar_load + slot * 8
                    mbar_c = bar_consumed + slot * 8
                    load_phase = (ki // stages) & 1

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

                    # Signal this slot is free.
                    ptx.mbarrier.arrive(mbar_c)

                # Commit after all K-tiles.
                ptx.tcgen05.commit(bar_mma, space="cluster")

            # ── All threads: wait MMA done, run epilogue ──
            with ptx.scope():
                ready = reg.scalar(pred)
                ptx.label("cw")
                ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
                    ready, ptx.addr(bar_mma), 0
                )
                ptx.bra("cd", pred=ready)
                ptx.bra("cw")
                ptx.label("cd")

            # Thread T reads DP=T (warps 0..3 cover DPs 0..127).
            row_base = m_base + tid
            (pd,) = ptx.global_ptrs(D)
            row_off = reg.scalar(u64); ptx.inst.mul.wide.u32(row_off, row_base, N)
            tile_col = reg.scalar(u64); ptx.inst.cvt.u64.u32(tile_col, n_base)
            d_index = row_off + tile_col
            d_ptr = pd + (d_index << 2)

            tmem_row_bits = (tid << 16) & 0x3E00000
            tmem_addr = tmem_base + tmem_row_bits

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
                        [out[vec * 4], out[vec * 4 + 1], out[vec * 4 + 2], out[vec * 4 + 3]],
                    )

            with ptx.if_(alloc_warp):
                ptx.tcgen05.dealloc(tmem_base, 512)
                ptx.tcgen05.relinquish_alloc_permit()
            ptx.ret()

        return blackwell_gemm


    def build_gemm_persistent(
        M: int,
        N: int,
        K: int,
        *,
        arch: str = "sm_100a",
        stages: int = STAGES,
        grid_minor_dim: str = "n",
        grid_tile_width: int = 8,
        num_ctas: int | None = None,
    ):
        """Persistent 1SM variant, following the next Pallas scheduling step."""
        assert arch.startswith("sm_100"), "Blackwell GEMM is only implemented for sm_100*"
        assert M % BM == 0, f"M={M} must be divisible by {BM}"
        assert N % BN == 0, f"N={N} must be divisible by {BN}"
        assert K % BK == 0, f"K={K} must be divisible by {BK}"
        assert stages >= 2, "Blackwell GEMM requires at least two stages"
        assert grid_tile_width >= 1, "Persistent grid tile width must be at least 1"
        k_iters = K // BK
        tile_m_iters = M // BM
        tile_n_iters = N // BN
        total_tiles = tile_m_iters * tile_n_iters
        if num_ctas is None:
            num_ctas = min(total_tiles, max(1, _detect_sm_count()))
        else:
            num_ctas = min(total_tiles, max(1, num_ctas))
        max_tile_iters = (total_tiles + num_ctas - 1) // num_ctas
        minor_dim = grid_minor_dim.lower()
        assert minor_dim in {"m", "n"}, "grid_minor_dim must be 'm' or 'n'"
        minor_is_n = minor_dim == "n"
        major_iters = tile_m_iters if minor_is_n else tile_n_iters
        minor_iters = tile_n_iters if minor_is_n else tile_m_iters
        group_span = major_iters * grid_tile_width
        has_partial_group = (minor_iters % grid_tile_width) != 0

        smem_b_base = SMEM_A_BASE + stages * A_STAGE
        smem_bar_load = smem_b_base + stages * B_STAGE
        smem_bar_consumed = smem_bar_load + stages * 8
        smem_bar_mma = smem_bar_consumed + stages * 8
        smem_tmem_slot = smem_bar_mma + 8
        smem_bytes = smem_tmem_slot + 16

        @kernel(
            in_specs=(
                Tile(M, K, bf16, Layout.TMA_128B, tma_box=(BM, BK)),
                Tile(N, K, bf16, Layout.TMA_128B, tma_box=(BN, BK)),
            ),
            out_specs=(Tile(M, N, f32, Layout.ROW),),
            grid=(num_ctas, 1, 1),
            block=(256, 1, 1),
            arch=arch,
            smem=smem_bytes,
            extern_smem=True,
        )
        def blackwell_gemm_persistent(A, B_T, D):
            base = smem.base()
            tmem_slot = base + smem_tmem_slot
            bar_load = base + smem_bar_load
            bar_consumed = base + smem_bar_consumed
            bar_mma = base + smem_bar_mma

            tid = reg.scalar(u32); ptx.inst.mov.u32(tid, ptx.special.tid.x())
            alloc_warp = reg.scalar(pred); ptx.inst.setp.lt.u32(alloc_warp, tid, 32)
            is_tma_warp = reg.scalar(pred); ptx.inst.setp.eq.u32(is_tma_warp, tid, TMA_WARP_TID)
            is_mma_warp = reg.scalar(pred); ptx.inst.setp.eq.u32(is_mma_warp, tid, MMA_WARP_TID)
            cta_x = reg.scalar(u32); ptx.inst.mov.u32(cta_x, ptx.special.ctaid.x())
            idesc = reg.scalar(b32, init=ptx.tcgen05.make_instr_desc_f16bf16_f32())

            with ptx.if_(alloc_warp):
                ptx.tcgen05.alloc(tmem_slot, 512)
            ptx.bar.sync(0)

            tmem_base = smem.load(b32, ptx.addr(tmem_slot))
            (pd,) = ptx.global_ptrs(D)

            for tile_iter in range(max_tile_iters):
                tile_linear = reg.scalar(u32)
                if tile_iter == 0:
                    ptx.inst.mov.u32(tile_linear, cta_x)
                else:
                    ptx.inst.add.u32(tile_linear, cta_x, tile_iter * num_ctas)

                done = reg.scalar(pred)
                ptx.inst.setp.ge.u32(done, tile_linear, total_tiles)
                ptx.bra("persistent_exit", pred=done)

                tile_major = reg.scalar(u32)
                tile_minor = reg.scalar(u32)
                group_id = reg.scalar(u32)
                group_offset = reg.scalar(u32)
                minor_in_group = reg.scalar(u32)
                tile_start = reg.scalar(u32)
                ptx.inst.div.u32(group_id, tile_linear, group_span)
                ptx.inst.rem.u32(group_offset, tile_linear, group_span)
                ptx.inst.mul.lo.u32(tile_start, group_id, grid_tile_width)

                if has_partial_group:
                    remaining = reg.scalar(u32)
                    group_width = reg.scalar(u32)
                    full_group = reg.scalar(pred)
                    ptx.inst.sub.u32(remaining, minor_iters, tile_start)
                    ptx.inst.setp.ge.u32(full_group, remaining, grid_tile_width)
                    ptx.inst.selp.b32(group_width, grid_tile_width, remaining, full_group)
                else:
                    group_width = reg.scalar(u32, init=grid_tile_width)

                ptx.inst.div.u32(tile_major, group_offset, group_width)
                ptx.inst.rem.u32(minor_in_group, group_offset, group_width)

                minor_fwd = reg.scalar(u32)
                minor_rev = reg.scalar(u32)
                minor_rev_off = reg.scalar(u32)
                group_width_m1 = reg.scalar(u32)
                odd_bit = reg.scalar(u32)
                is_odd_group = reg.scalar(pred)
                ptx.inst.add.u32(minor_fwd, tile_start, minor_in_group)
                ptx.inst.sub.u32(group_width_m1, group_width, 1)
                ptx.inst.sub.u32(minor_rev_off, group_width_m1, minor_in_group)
                ptx.inst.add.u32(minor_rev, tile_start, minor_rev_off)
                ptx.inst.and_.b32(odd_bit, group_id, 1)
                ptx.inst.setp.ne.u32(is_odd_group, odd_bit, 0)
                ptx.inst.selp.b32(tile_minor, minor_rev, minor_fwd, is_odd_group)

                tile_m = reg.scalar(u32)
                tile_n = reg.scalar(u32)
                if minor_is_n:
                    ptx.inst.mov.u32(tile_m, tile_major)
                    ptx.inst.mov.u32(tile_n, tile_minor)
                else:
                    ptx.inst.mov.u32(tile_m, tile_minor)
                    ptx.inst.mov.u32(tile_n, tile_major)

                m_base = reg.scalar(u32)
                n_base = reg.scalar(u32)
                ptx.inst.shl.b32(m_base, tile_m, 7)
                ptx.inst.shl.b32(n_base, tile_n, 8)

                with ptx.if_(tid == 0):
                    for s in range(stages):
                        ptx.mbarrier.init(bar_load + s * 8, 1)
                        ptx.mbarrier.init(bar_consumed + s * 8, 1)
                    ptx.mbarrier.init(bar_mma, 1)
                    ptx.fence.proxy_async_shared_cta()
                ptx.bar.sync(0)

                with ptx.if_(is_tma_warp):
                    for ki in range(k_iters):
                        slot = ki % stages
                        slot_uses_per_tile = 1 + ((k_iters - 1 - slot) // stages)
                        slot_use_index = tile_iter * slot_uses_per_tile + (ki // stages)
                        smem_a = base + SMEM_A_BASE + slot * A_STAGE
                        smem_b = base + smem_b_base + slot * B_STAGE
                        mbar_l = bar_load + slot * 8
                        mbar_c = bar_consumed + slot * 8

                        if slot_use_index > 0:
                            consumed_phase = (slot_use_index - 1) & 1
                            with ptx.scope():
                                ready = reg.scalar(pred)
                                ptx.label(f"pcwait_{tile_iter}_{ki}")
                                ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
                                    ready, ptx.addr(mbar_c), consumed_phase
                                )
                                ptx.bra(f"pcdone_{tile_iter}_{ki}", pred=ready)
                                ptx.bra(f"pcwait_{tile_iter}_{ki}")
                                ptx.label(f"pcdone_{tile_iter}_{ki}")
                            ptx.tcgen05.fence_after_thread_sync()

                        ptx.mbarrier.arrive_expect_tx(mbar_l, A_STAGE + B_STAGE)
                        ptx.cp.async_.bulk.tensor_2d(
                            dst=smem_a, src=A.tma_desc(),
                            coord=(ki * BK, m_base), mbar=mbar_l,
                        )
                        ptx.cp.async_.bulk.tensor_2d(
                            dst=smem_b, src=B_T.tma_desc(),
                            coord=(ki * BK, n_base), mbar=mbar_l,
                        )

                with ptx.if_(is_mma_warp):
                    for ki in range(k_iters):
                        slot = ki % stages
                        slot_uses_per_tile = 1 + ((k_iters - 1 - slot) // stages)
                        slot_use_index = tile_iter * slot_uses_per_tile + (ki // stages)
                        smem_a = base + SMEM_A_BASE + slot * A_STAGE
                        smem_b = base + smem_b_base + slot * B_STAGE
                        mbar_l = bar_load + slot * 8
                        mbar_c = bar_consumed + slot * 8
                        load_phase = slot_use_index & 1

                        with ptx.scope():
                            ready = reg.scalar(pred)
                            ptx.label(f"plwait_{tile_iter}_{ki}")
                            ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
                                ready, ptx.addr(mbar_l), load_phase
                            )
                            ptx.bra(f"pldone_{tile_iter}_{ki}", pred=ready)
                            ptx.bra(f"plwait_{tile_iter}_{ki}")
                            ptx.label(f"pldone_{tile_iter}_{ki}")

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

                with ptx.scope():
                    ready = reg.scalar(pred)
                    ptx.label(f"pcw_{tile_iter}")
                    ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
                        ready, ptx.addr(bar_mma), 0
                    )
                    ptx.bra(f"pcd_{tile_iter}", pred=ready)
                    ptx.bra(f"pcw_{tile_iter}")
                    ptx.label(f"pcd_{tile_iter}")

                row_base = reg.scalar(u32)
                ptx.inst.add.u32(row_base, m_base, tid)
                row_off = reg.scalar(u64); ptx.inst.mul.wide.u32(row_off, row_base, N)
                tile_col = reg.scalar(u64); ptx.inst.cvt.u64.u32(tile_col, n_base)
                d_index = row_off + tile_col
                d_ptr = pd + (d_index << 2)

                tmem_row_bits = (tid << 16) & 0x3E00000
                tmem_addr = tmem_base + tmem_row_bits

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
                            [out[vec * 4], out[vec * 4 + 1], out[vec * 4 + 2], out[vec * 4 + 3]],
                        )

                ptx.bar.sync(0)

            ptx.label("persistent_exit")
            with ptx.if_(alloc_warp):
                ptx.tcgen05.dealloc(tmem_base, 512)
                ptx.tcgen05.relinquish_alloc_permit()
            ptx.ret()

        return blackwell_gemm_persistent


    def _build_gemm_persistent_overlap_experimental(
        M: int,
        N: int,
        K: int,
        *,
        arch: str = "sm_100a",
        stages: int = STAGES,
        grid_minor_dim: str = "n",
        grid_tile_width: int = 8,
        num_ctas: int | None = None,
    ):
        """Experimental persistent 1SM kernel with a dedicated store warpgroup.

        This matches the next major Pallas step more closely than ``build_gemm``:
        tile work is scheduled persistently, TMEM accumulators are double-buffered,
        and a second warpgroup drains slot ``i`` while the compute warpgroup fills
        slot ``i ^ 1``.

        The current lowering still assembles but does not launch correctly on B200,
        so it is kept out of the supported path while the stable kernels continue to
        use the simpler epilogue.
        """
        assert arch.startswith("sm_100"), "Blackwell GEMM is only implemented for sm_100*"
        assert M % BM == 0, f"M={M} must be divisible by {BM}"
        assert N % BN == 0, f"N={N} must be divisible by {BN}"
        assert K % BK == 0, f"K={K} must be divisible by {BK}"
        assert stages >= 2, "Blackwell GEMM requires at least two stages"
        assert grid_tile_width >= 1, "Persistent grid tile width must be at least 1"
        k_iters = K // BK
        tile_m_iters = M // BM
        tile_n_iters = N // BN
        total_tiles = tile_m_iters * tile_n_iters
        if num_ctas is None:
            num_ctas = min(total_tiles, max(1, _detect_sm_count()))
        else:
            num_ctas = min(total_tiles, max(1, num_ctas))
        max_tile_iters = (total_tiles + num_ctas - 1) // num_ctas
        minor_dim = grid_minor_dim.lower()
        assert minor_dim in {"m", "n"}, "grid_minor_dim must be 'm' or 'n'"
        minor_is_n = minor_dim == "n"
        major_iters = tile_m_iters if minor_is_n else tile_n_iters
        minor_iters = tile_n_iters if minor_is_n else tile_m_iters
        group_span = major_iters * grid_tile_width
        has_partial_group = (minor_iters % grid_tile_width) != 0
        TMEM_SLOTS = 2
        # tcgen05.alloc is bounded to 512 here. For BN=256 and 2 accumulator slots,
        # the double-buffered TMEM address space is a 128x512 ref, with each slot
        # taking a 256-column slice.
        TMEM_ALLOC = BN * TMEM_SLOTS

        smem_b_base = SMEM_A_BASE + stages * A_STAGE
        smem_bar_load = smem_b_base + stages * B_STAGE
        smem_bar_consumed = smem_bar_load + stages * 8
        smem_bar_mma_done = smem_bar_consumed + stages * 8
        smem_bar_store_done = smem_bar_mma_done + 2 * 8
        smem_tmem_slot = smem_bar_store_done + 2 * 8
        smem_bytes = smem_tmem_slot + 16

        @kernel(
            in_specs=(
                Tile(M, K, bf16, Layout.TMA_128B, tma_box=(BM, BK)),
                Tile(N, K, bf16, Layout.TMA_128B, tma_box=(BN, BK)),
            ),
            out_specs=(Tile(M, N, f32, Layout.ROW),),
            grid=(num_ctas, 1, 1),
            # Pallas launches two warpgroups here: one for TMA/MMA, one for store.
            block=(256, 1, 1),
            arch=arch,
            smem=smem_bytes,
            extern_smem=True,
        )
        def blackwell_gemm_persistent_overlap(A, B_T, D):
            base = smem.base()
            tmem_slot = base + smem_tmem_slot
            bar_load = base + smem_bar_load
            bar_consumed = base + smem_bar_consumed
            bar_mma_done = base + smem_bar_mma_done
            bar_store_done = base + smem_bar_store_done

            tid = reg.scalar(u32); ptx.inst.mov.u32(tid, ptx.special.tid.x())
            alloc_warp = reg.scalar(pred); ptx.inst.setp.lt.u32(alloc_warp, tid, 32)
            is_tma_warp = reg.scalar(pred); ptx.inst.setp.eq.u32(is_tma_warp, tid, TMA_WARP_TID)
            is_mma_warp = reg.scalar(pred); ptx.inst.setp.eq.u32(is_mma_warp, tid, MMA_WARP_TID)
            is_store_wg = reg.scalar(pred); ptx.inst.setp.ge.u32(is_store_wg, tid, 128)
            is_store_leader = reg.scalar(pred); ptx.inst.setp.eq.u32(is_store_leader, tid, 128)
            cta_x = reg.scalar(u32); ptx.inst.mov.u32(cta_x, ptx.special.ctaid.x())
            idesc = reg.scalar(b32, init=ptx.tcgen05.make_instr_desc_f16bf16_f32())

            with ptx.if_(tid == 0):
                for s in range(stages):
                    ptx.mbarrier.init(bar_load + s * 8, 1)
                    ptx.mbarrier.init(bar_consumed + s * 8, 1)
                ptx.mbarrier.init(bar_mma_done + 0 * 8, 1)
                ptx.mbarrier.init(bar_mma_done + 1 * 8, 1)
                ptx.mbarrier.init(bar_store_done + 0 * 8, 1)
                ptx.mbarrier.init(bar_store_done + 1 * 8, 1)
                ptx.fence.proxy_async_shared_cta()
            with ptx.if_(alloc_warp):
                ptx.tcgen05.alloc(tmem_slot, TMEM_ALLOC)
            ptx.bar.sync(0)

            tmem_base = smem.load(b32, ptx.addr(tmem_slot))
            (pd,) = ptx.global_ptrs(D)

            for tile_iter in range(max_tile_iters):
                tile_linear = reg.scalar(u32)
                if tile_iter == 0:
                    ptx.inst.mov.u32(tile_linear, cta_x)
                else:
                    ptx.inst.add.u32(tile_linear, cta_x, tile_iter * num_ctas)

                active = reg.scalar(pred)
                ptx.inst.setp.lt.u32(active, tile_linear, total_tiles)

                with ptx.if_(active):
                    tile_major = reg.scalar(u32)
                    tile_minor = reg.scalar(u32)
                    group_id = reg.scalar(u32)
                    group_offset = reg.scalar(u32)
                    minor_in_group = reg.scalar(u32)
                    tile_start = reg.scalar(u32)
                    ptx.inst.div.u32(group_id, tile_linear, group_span)
                    ptx.inst.rem.u32(group_offset, tile_linear, group_span)
                    ptx.inst.mul.lo.u32(tile_start, group_id, grid_tile_width)

                    if has_partial_group:
                        remaining = reg.scalar(u32)
                        group_width = reg.scalar(u32)
                        full_group = reg.scalar(pred)
                        ptx.inst.sub.u32(remaining, minor_iters, tile_start)
                        ptx.inst.setp.ge.u32(full_group, remaining, grid_tile_width)
                        ptx.inst.selp.b32(group_width, grid_tile_width, remaining, full_group)
                    else:
                        group_width = reg.scalar(u32, init=grid_tile_width)

                    ptx.inst.div.u32(tile_major, group_offset, group_width)
                    ptx.inst.rem.u32(minor_in_group, group_offset, group_width)

                    minor_fwd = reg.scalar(u32)
                    minor_rev = reg.scalar(u32)
                    minor_rev_off = reg.scalar(u32)
                    group_width_m1 = reg.scalar(u32)
                    odd_bit = reg.scalar(u32)
                    is_odd_group = reg.scalar(pred)
                    ptx.inst.add.u32(minor_fwd, tile_start, minor_in_group)
                    ptx.inst.sub.u32(group_width_m1, group_width, 1)
                    ptx.inst.sub.u32(minor_rev_off, group_width_m1, minor_in_group)
                    ptx.inst.add.u32(minor_rev, tile_start, minor_rev_off)
                    ptx.inst.and_.b32(odd_bit, group_id, 1)
                    ptx.inst.setp.ne.u32(is_odd_group, odd_bit, 0)
                    ptx.inst.selp.b32(tile_minor, minor_rev, minor_fwd, is_odd_group)

                    tile_m = reg.scalar(u32)
                    tile_n = reg.scalar(u32)
                    if minor_is_n:
                        ptx.inst.mov.u32(tile_m, tile_major)
                        ptx.inst.mov.u32(tile_n, tile_minor)
                    else:
                        ptx.inst.mov.u32(tile_m, tile_minor)
                        ptx.inst.mov.u32(tile_n, tile_major)

                    m_base = reg.scalar(u32)
                    n_base = reg.scalar(u32)
                    ptx.inst.shl.b32(m_base, tile_m, 7)
                    ptx.inst.shl.b32(n_base, tile_n, 8)

                    acc_slot = tile_iter & 1
                    mma_done_slot = bar_mma_done + acc_slot * 8
                    store_done_slot = bar_store_done + acc_slot * 8
                    slot_tmem_base = tmem_base + acc_slot * BN

                    if tile_iter >= 2:
                        reuse_phase = ((tile_iter // 2) - 1) & 1
                        with ptx.if_(is_mma_warp):
                            with ptx.scope():
                                ready = reg.scalar(pred)
                                ptx.label(f"sdwait_{tile_iter}")
                                ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
                                    ready, ptx.addr(store_done_slot), reuse_phase
                                )
                                ptx.bra(f"sddone_{tile_iter}", pred=ready)
                                ptx.bra(f"sdwait_{tile_iter}")
                                ptx.label(f"sddone_{tile_iter}")
                            ptx.tcgen05.fence_after_thread_sync()

                    with ptx.if_(is_tma_warp):
                        for ki in range(k_iters):
                            slot = ki % stages
                            slot_uses_per_tile = 1 + ((k_iters - 1 - slot) // stages)
                            slot_use_index = tile_iter * slot_uses_per_tile + (ki // stages)
                            smem_a = base + SMEM_A_BASE + slot * A_STAGE
                            smem_b = base + smem_b_base + slot * B_STAGE
                            mbar_l = bar_load + slot * 8
                            mbar_c = bar_consumed + slot * 8

                            if slot_use_index > 0:
                                consumed_phase = (slot_use_index - 1) & 1
                                with ptx.scope():
                                    ready = reg.scalar(pred)
                                    ptx.label(f"ov_cwait_{tile_iter}_{ki}")
                                    ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
                                        ready, ptx.addr(mbar_c), consumed_phase
                                    )
                                    ptx.bra(f"ov_cdone_{tile_iter}_{ki}", pred=ready)
                                    ptx.bra(f"ov_cwait_{tile_iter}_{ki}")
                                    ptx.label(f"ov_cdone_{tile_iter}_{ki}")
                                ptx.tcgen05.fence_after_thread_sync()

                            ptx.mbarrier.arrive_expect_tx(mbar_l, A_STAGE + B_STAGE)
                            ptx.cp.async_.bulk.tensor_2d(
                                dst=smem_a, src=A.tma_desc(),
                                coord=(ki * BK, m_base), mbar=mbar_l,
                            )
                            ptx.cp.async_.bulk.tensor_2d(
                                dst=smem_b, src=B_T.tma_desc(),
                                coord=(ki * BK, n_base), mbar=mbar_l,
                            )

                    with ptx.if_(is_mma_warp):
                        for ki in range(k_iters):
                            slot = ki % stages
                            slot_uses_per_tile = 1 + ((k_iters - 1 - slot) // stages)
                            slot_use_index = tile_iter * slot_uses_per_tile + (ki // stages)
                            smem_a = base + SMEM_A_BASE + slot * A_STAGE
                            smem_b = base + smem_b_base + slot * B_STAGE
                            mbar_l = bar_load + slot * 8
                            mbar_c = bar_consumed + slot * 8
                            load_phase = slot_use_index & 1

                            with ptx.scope():
                                ready = reg.scalar(pred)
                                ptx.label(f"ov_lwait_{tile_iter}_{ki}")
                                ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
                                    ready, ptx.addr(mbar_l), load_phase
                                )
                                ptx.bra(f"ov_ldone_{tile_iter}_{ki}", pred=ready)
                                ptx.bra(f"ov_lwait_{tile_iter}_{ki}")
                                ptx.label(f"ov_ldone_{tile_iter}_{ki}")

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
                                    slot_tmem_base, desc_a, desc_b, idesc,
                                    kind="f16", pred_operand=(not is_first),
                                )

                            # Match Pallas/Mosaic: retire the consumer slot via the
                            # tensor-core commit path, not a plain mbarrier arrive.
                            ptx.tcgen05.commit(mbar_c)

                        ptx.tcgen05.commit(mma_done_slot, space="cluster")

                    with ptx.if_(is_store_wg):
                        store_tid = reg.scalar(u32)
                        ptx.inst.sub.u32(store_tid, tid, 128)
                        store_phase = (tile_iter // 2) & 1
                        with ptx.scope():
                            ready = reg.scalar(pred)
                            ptx.label(f"ov_mwait_{tile_iter}")
                            ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
                                ready, ptx.addr(mma_done_slot), store_phase
                            )
                            ptx.bra(f"ov_mdone_{tile_iter}", pred=ready)
                            ptx.bra(f"ov_mwait_{tile_iter}")
                            ptx.label(f"ov_mdone_{tile_iter}")
                        ptx.tcgen05.fence_after_thread_sync()

                        out = reg.array(b32, EPILOGUE_TILE_N)
                        row_base = reg.scalar(u32)
                        row_off = reg.scalar(u64)
                        tile_col = reg.scalar(u64)
                        tmem_row_bits = reg.scalar(u32)
                        tmem_addr = reg.scalar(b32)
                        ptx.inst.add.u32(row_base, m_base, store_tid)
                        ptx.inst.mul.wide.u32(row_off, row_base, N)
                        ptx.inst.cvt.u64.u32(tile_col, n_base)
                        d_index = row_off + tile_col
                        d_ptr = pd + (d_index << 2)
                        ptx.inst.shl.b32(tmem_row_bits, store_tid, 16)
                        ptx.inst.and_.b32(tmem_row_bits, tmem_row_bits, 0x3E00000)
                        ptx.inst.add.u32(tmem_addr, slot_tmem_base, tmem_row_bits)

                        for chunk in range(BN // EPILOGUE_TILE_N):
                            chunk_off = chunk * EPILOGUE_TILE_N
                            ptx.tcgen05.ld(
                                [out[i] for i in range(EPILOGUE_TILE_N)],
                                tmem_addr + chunk_off,
                                shape="32x32b", count=EPILOGUE_TILE_N, dtype="b32",
                            )
                            ptx.tcgen05.wait_ld()

                            for vec in range(EPILOGUE_TILE_N // 4):
                                off = (chunk_off + vec * 4) * 4
                                ptx.inst.st.global_.v4.b32(
                                    ptx.addr(d_ptr, off),
                                    [out[vec * 4], out[vec * 4 + 1], out[vec * 4 + 2], out[vec * 4 + 3]],
                                )

                        ptx.tcgen05.fence_before_thread_sync()
                        ptx.bar.sync(1, 128, pred=is_store_wg)
                        with ptx.if_(is_store_leader):
                            ptx.mbarrier.arrive(store_done_slot)

            ptx.bar.sync(0)
            with ptx.if_(alloc_warp):
                ptx.tcgen05.dealloc(tmem_base, TMEM_ALLOC)
                ptx.tcgen05.relinquish_alloc_permit()
            ptx.ret()

        return blackwell_gemm_persistent_overlap


    def build_gemm_2sm(
        M: int,
        N: int,
        K: int,
        *,
        arch: str = "sm_100a",
        stages: int = STAGES_2SM,
        grid_minor_dim: str = GRID_MINOR_DIM,
        grid_tile_width: int = GRID_TILE_WIDTH,
    ):
        """2-SM collective UMMA variant — correct, but currently not faster.

        Pairs consecutive ``ctaid.x`` CTAs into a cluster of 2. Both CTAs allocate
        cluster-shared TMEM (``tcgen05.alloc.cta_group::2``). Each CTA loads its
        own ``BM``-row slice of A *and* its own ``BN/2``-col slice of B — per the
        CUTLASS ``SM100_MMA_F16BF16_2x1SM_SS`` traits, the 2SM MMA reads A split
        in the M direction AND B split in the N direction across the two CTAs.

        The lead CTA dispatches ``tcgen05.mma.cta_group::2`` with
        ``idesc.m_dim = 256, idesc.n_dim = BN``, producing a logical 256×BN
        output. Each CTA receives its own 128×BN slice of output in its local
        TMEM. The commit uses ``.multicast::cluster`` with mask ``0b11`` so the
        arrive reaches *both* CTAs' mma_bar.

        **Perf tradeoff**: lead's MMA reads cluster-shared SMEM for A and B, so
        before each iter we ``cluster.sync()`` to ensure the peer CTA's producer
        has landed. That barrier costs ~100ns per K-tile, which at K=8192
        (k_iters=128) overwhelms the MMA throughput gain. Measured on B200:
        1SM 626 TFLOPS vs 2SM 340 TFLOPS at 2048³ bf16. A follow-up can swap
        the per-iter cluster.sync for a cluster-shared mbarrier with count=2 so
        both producers arrive once per slot — that lifts the cap without giving
        up correctness.
        """
        assert arch.startswith("sm_100")
        assert M % (2 * BM) == 0, f"M={M} must be divisible by {2 * BM} for 2SM"
        assert N % BN == 0, f"N={N} must be divisible by {BN}"
        assert K % BK == 0, f"K={K} must be divisible by {BK}"
        assert stages >= 2, "2SM Blackwell GEMM requires at least two stages"
        assert grid_tile_width >= 1, "grid_tile_width must be at least 1"
        k_iters = K // BK
        tile_m_iters = M // (2 * BM)
        tile_n_iters = N // BN
        minor_dim = grid_minor_dim.lower()
        assert minor_dim in {"m", "n"}, "grid_minor_dim must be 'm' or 'n'"
        minor_is_n = minor_dim == "n"
        major_iters = tile_m_iters if minor_is_n else tile_n_iters
        minor_iters = tile_n_iters if minor_is_n else tile_m_iters
        group_span = major_iters * grid_tile_width
        has_partial_group = (minor_iters % grid_tile_width) != 0

        # 2SM splits B in the N direction: each CTA holds BN/2 cols of B_T.
        BN_HALF = BN // 2
        B_STAGE_2SM = BN_HALF * BK * 2          # 16 KB per stage per CTA
        LOAD_TX_BYTES_2SM = 2 * (A_STAGE + B_STAGE_2SM)

        # SMEM bars:
        #   load_bar[slot]   : reserved legacy local-TMA barrier
        #   cluster_bar[slot]: leader-tracked collective TMA completion on CTA 0
        #   consumed_bar[slot]: MMA-retire async arrive via tcgen05.commit.multicast
        #   mma_bar          : final commit arrive after all K-tiles
        SMEM_A_BASE_2SM       = 0
        SMEM_B_BASE_2SM       = SMEM_A_BASE_2SM + stages * A_STAGE
        SMEM_BAR_LOAD_2SM     = SMEM_B_BASE_2SM + stages * B_STAGE_2SM
        SMEM_BAR_CLUSTER_2SM  = SMEM_BAR_LOAD_2SM + stages * 8
        SMEM_BAR_CONSUMED_2SM = SMEM_BAR_CLUSTER_2SM + stages * 8
        SMEM_BAR_MMA_2SM      = SMEM_BAR_CONSUMED_2SM + stages * 8
        SMEM_TMEM_SLOT_2SM    = SMEM_BAR_MMA_2SM + 8
        SMEM_BYTES_2SM        = SMEM_TMEM_SLOT_2SM + 16

        @kernel(
            in_specs=(
                Tile(M, K, bf16, Layout.TMA_128B, tma_box=(BM, BK)),
                Tile(N, K, bf16, Layout.TMA_128B, tma_box=(BN_HALF, BK)),
            ),
            out_specs=(Tile(M, N, f32, Layout.ROW),),
            grid=(M // BM, N // BN, 1),
            cluster=(2, 1, 1),
            block=(128, 1, 1),
            arch=arch,
            smem=SMEM_BYTES_2SM,
            extern_smem=True,
            raw_directives=[
                ("explicitcluster", ()),
                ("reqnctapercluster", (2, 1, 1)),
            ],
        )
        def blackwell_gemm_2sm(A, B_T, D):
            from pyptx.types import b16

            base = smem.base()
            tmem_slot = base + SMEM_TMEM_SLOT_2SM
            bar_load = base + SMEM_BAR_LOAD_2SM
            bar_cluster = base + SMEM_BAR_CLUSTER_2SM
            bar_consumed = base + SMEM_BAR_CONSUMED_2SM
            bar_mma = base + SMEM_BAR_MMA_2SM

            tid = reg.scalar(u32); ptx.inst.mov.u32(tid, ptx.special.tid.x())
            alloc_warp = reg.scalar(pred); ptx.inst.setp.lt.u32(alloc_warp, tid, 32)
            # 3-way warp split so TMA issue can run ahead of the own-TMA wait.
            is_tma_warp = reg.scalar(pred); ptx.inst.setp.eq.u32(is_tma_warp, tid, 0)    # warp 0 lane 0
            is_sig_warp = reg.scalar(pred); ptx.inst.setp.eq.u32(is_sig_warp, tid, 32)   # warp 1 lane 0
            is_mma_warp = reg.scalar(pred); ptx.inst.setp.eq.u32(is_mma_warp, tid, 64)   # warp 2 lane 0

            cta_rank = reg.scalar(u32); ptx.inst.mov.u32(cta_rank, ptx.sreg("%cluster_ctarank"))
            is_lead = reg.scalar(pred); ptx.inst.setp.eq.u32(is_lead, cta_rank, 0)

            grid_x = reg.scalar(u32); ptx.inst.mov.u32(grid_x, ptx.special.ctaid.x())
            grid_y = reg.scalar(u32); ptx.inst.mov.u32(grid_y, ptx.special.ctaid.y())
            cluster_x = reg.scalar(u32)
            ptx.inst.shr.u32(cluster_x, grid_x, 1)
            cluster_linear = reg.scalar(u32)
            ptx.inst.mad.lo.u32(cluster_linear, grid_y, tile_m_iters, cluster_x)

            tile_major = reg.scalar(u32)
            tile_minor = reg.scalar(u32)
            group_id = reg.scalar(u32)
            group_offset = reg.scalar(u32)
            minor_in_group = reg.scalar(u32)
            tile_start = reg.scalar(u32)
            ptx.inst.div.u32(group_id, cluster_linear, group_span)
            ptx.inst.rem.u32(group_offset, cluster_linear, group_span)
            ptx.inst.mul.lo.u32(tile_start, group_id, grid_tile_width)

            if has_partial_group:
                remaining = reg.scalar(u32)
                group_width = reg.scalar(u32)
                full_group = reg.scalar(pred)
                ptx.inst.sub.u32(remaining, minor_iters, tile_start)
                ptx.inst.setp.ge.u32(full_group, remaining, grid_tile_width)
                ptx.inst.selp.b32(group_width, grid_tile_width, remaining, full_group)
            else:
                group_width = reg.scalar(u32, init=grid_tile_width)

            ptx.inst.div.u32(tile_major, group_offset, group_width)
            ptx.inst.rem.u32(minor_in_group, group_offset, group_width)

            minor_fwd = reg.scalar(u32)
            minor_rev = reg.scalar(u32)
            minor_rev_off = reg.scalar(u32)
            group_width_m1 = reg.scalar(u32)
            odd_bit = reg.scalar(u32)
            is_odd_group = reg.scalar(pred)
            ptx.inst.add.u32(minor_fwd, tile_start, minor_in_group)
            ptx.inst.sub.u32(group_width_m1, group_width, 1)
            ptx.inst.sub.u32(minor_rev_off, group_width_m1, minor_in_group)
            ptx.inst.add.u32(minor_rev, tile_start, minor_rev_off)
            ptx.inst.and_.b32(odd_bit, group_id, 1)
            ptx.inst.setp.ne.u32(is_odd_group, odd_bit, 0)
            ptx.inst.selp.b32(tile_minor, minor_rev, minor_fwd, is_odd_group)

            cluster_m = reg.scalar(u32)
            cluster_n = reg.scalar(u32)
            if minor_is_n:
                ptx.inst.mov.u32(cluster_m, tile_major)
                ptx.inst.mov.u32(cluster_n, tile_minor)
            else:
                ptx.inst.mov.u32(cluster_m, tile_minor)
                ptx.inst.mov.u32(cluster_n, tile_major)

            cta_m = reg.scalar(u32)
            ptx.inst.shl.b32(cta_m, cluster_m, 1)
            ptx.inst.add.u32(cta_m, cta_m, cta_rank)
            cta_n = reg.scalar(u32)
            ptx.inst.mov.u32(cta_n, cluster_n)
            m_base = cta_m << 7
            n_base = cta_n << 8

            # Multicast mask for 2-SM tcgen05.commit: both cluster CTAs
            # (rank 0 and 1) receive the retire arrive.
            mma_bar_mask = reg.scalar(b16)
            ptx.inst.mov.b16(mma_bar_mask, 3)

            # 2-SM MMA: idesc encodes M=256.
            idesc = reg.scalar(b32, init=ptx.tcgen05.make_instr_desc_f16bf16_f32(m=256, n=BN))

            with ptx.if_(tid == 0):
                for s in range(stages):
                    ptx.mbarrier.init(bar_load + s * 8, 1)
                    ptx.mbarrier.init(bar_cluster + s * 8, 1)
                    ptx.mbarrier.init(bar_consumed + s * 8, 1)
                ptx.mbarrier.init(bar_mma, 1)
                ptx.fence.proxy_async_shared_cta()
            # Both CTAs' first warp execute tcgen05.alloc.cta_group::2 with the
            # same dst_ptr (CUTE::Allocator2Sm contract).
            with ptx.if_(alloc_warp):
                ptx.tcgen05.alloc(tmem_slot, 512, cta_group=2)
            ptx.cluster.sync()

            tmem_base = smem.load(b32, ptx.addr(tmem_slot))

            # Each CTA loads its BM row slice of A and its BN/2 col slice of B.
            # CTA 0: B_T rows [n_base .. n_base+BN/2]
            # CTA 1: B_T rows [n_base+BN/2 .. n_base+BN]
            cta_n_off = reg.scalar(u32)
            ptx.inst.mul.lo.u32(cta_n_off, cta_rank, BN_HALF)
            b_row_base = reg.scalar(u32); ptx.inst.add.u32(b_row_base, n_base, cta_n_off)

            # ── Producer (warp 0): issue leader-tracked collective TMA loads.
            with ptx.if_(is_tma_warp):
                for ki in range(k_iters):
                    slot = ki % stages
                    smem_a = base + SMEM_A_BASE_2SM + slot * A_STAGE
                    smem_b = base + SMEM_B_BASE_2SM + slot * B_STAGE_2SM
                    mbar_x = bar_cluster + slot * 8
                    mbar_c = bar_consumed + slot * 8

                    if ki >= stages:
                        consumed_phase = ((ki // stages) - 1) & 1
                        with ptx.scope():
                            ready = reg.scalar(pred)
                            ptx.label(f"cwait2_{ki}")
                            ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
                                ready, ptx.addr(mbar_c), consumed_phase
                            )
                            ptx.bra(f"cdone2_{ki}", pred=ready)
                            ptx.bra(f"cwait2_{ki}")
                            ptx.label(f"cdone2_{ki}")
                        # Mosaic inserts a cluster-scope proxy acquire before
                        # reusing a collective pipeline slot.
                        ptx.fence.proxy_async_generic_acquire_shared_cluster()

                    with ptx.if_(is_lead):
                        ptx.mbarrier.arrive_expect_tx(mbar_x, LOAD_TX_BYTES_2SM)
                    mapped_mbar = ptx.cluster.map_shared_u32(mbar_x, 0)
                    ptx.cp.async_.bulk.tensor_2d.shared_cta_global_tile(
                        dst=smem_a,
                        src=A.tma_desc(),
                        coord=(ki * BK, m_base),
                        mbar=mapped_mbar,
                        cta_group=2,
                    )
                    ptx.cp.async_.bulk.tensor_2d.shared_cta_global_tile(
                        dst=smem_b,
                        src=B_T.tma_desc(),
                        coord=(ki * BK, b_row_base),
                        mbar=mapped_mbar,
                        cta_group=2,
                    )

            # ── Signal warp (warp 1): reserved for future dedicated epilogue work.
            with ptx.if_(is_sig_warp):
                pass

            # ── MMA dispatcher (warp 1): wait cluster_bar (both CTAs ready), ──
            # issue the 2SM MMA, and `tcgen05.commit` the consumed barrier so
            # the producer can reuse the slot AFTER the MMA retires — not
            # immediately as a plain `mbarrier.arrive` would.
            with ptx.if_(is_mma_warp):
                for ki in range(k_iters):
                    slot = ki % stages
                    smem_a = base + SMEM_A_BASE_2SM + slot * A_STAGE
                    smem_b = base + SMEM_B_BASE_2SM + slot * B_STAGE_2SM
                    mbar_x = bar_cluster + slot * 8
                    mbar_c = bar_consumed + slot * 8
                    cluster_phase = (ki // stages) & 1

                    with ptx.if_(is_lead):
                        with ptx.scope():
                            ready = reg.scalar(pred)
                            ptx.label(f"xw2_{ki}")
                            ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
                                ready, ptx.addr(mbar_x), cluster_phase
                            )
                            ptx.bra(f"xd2_{ki}", pred=ready)
                            ptx.bra(f"xw2_{ki}")
                            ptx.label(f"xd2_{ki}")
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
                                kind="f16", cta_group=2,
                                pred_operand=(not is_first),
                            )
                        # Multicast commit: consumed_bar[slot] arrives on both
                        # CTAs' local mbars when the K-tile's MMAs retire.
                        ptx.tcgen05.commit(
                            mbar_c,
                            cta_group=2,
                            multicast=True,
                            multicast_mask=mma_bar_mask,
                            space="cluster",
                        )

                with ptx.if_(is_lead):
                    ptx.tcgen05.commit(
                        bar_mma,
                        cta_group=2,
                        multicast=True,
                        multicast_mask=mma_bar_mask,
                        space="cluster",
                    )

            with ptx.scope():
                ready = reg.scalar(pred)
                ptx.label("cw2")
                ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
                    ready, ptx.addr(bar_mma), 0
                )
                ptx.bra("cd2", pred=ready)
                ptx.bra("cw2")
                ptx.label("cd2")

            # Epilogue: each CTA reads its local TMEM (its half of the 256 rows).
            row_base = (cta_m << 7) + tid
            (pd,) = ptx.global_ptrs(D)
            row_off = reg.scalar(u64); ptx.inst.mul.wide.u32(row_off, row_base, N)
            tile_col = reg.scalar(u64); ptx.inst.cvt.u64.u32(tile_col, n_base)
            d_index = row_off + tile_col
            d_ptr = pd + (d_index << 2)

            tmem_row_bits = (tid << 16) & 0x3E00000
            tmem_addr = tmem_base + tmem_row_bits

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
                        [out[vec * 4], out[vec * 4 + 1], out[vec * 4 + 2], out[vec * 4 + 3]],
                    )

            # 2-SM dealloc (both CTAs) + relinquish.
            with ptx.if_(alloc_warp):
                ptx.tcgen05.dealloc(tmem_base, 512, cta_group=2)
                ptx.tcgen05.relinquish_alloc_permit(cta_group=2)
            ptx.ret()

        return blackwell_gemm_2sm
        assert arch.startswith("sm_100")


    def gemm_ref(a, b):
        return jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32))


    def run_torch(M: int, N: int, K: int) -> bool:
        import torch

        k = build_gemm(M, N, K)
        seed = M * 10007 + N * 313 + K
        rng = np.random.default_rng(seed)
        a_np = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
        b_np = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
        a = torch.tensor(a_np, device="cuda", dtype=torch.bfloat16)
        b = torch.tensor(b_np, device="cuda", dtype=torch.bfloat16)
        b_t = b.transpose(0, 1).contiguous()

        out = k(a, b_t)
        torch.cuda.synchronize()
        ref = a.float() @ b.float()
        diff = float((out - ref).abs().max())
        ok = bool(torch.allclose(out, ref, atol=5e-2, rtol=5e-2))
        status = "OK  " if ok else "FAIL"
        print(f"[Torch{status}] M={M:5d} N={N:5d} K={K:5d}  max_abs={diff:.3e}")
        return ok


    def benchmark(M: int, N: int, K: int, iters: int = 20):
        import torch

        k = build_gemm(M, N, K)
        a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
        b_t = b.transpose(0, 1).contiguous()

        for _ in range(5):
            k(a, b_t)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            k(a, b_t)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        flops = 2 * M * N * K
        tflops = flops / (ms * 1e-3) / 1e12
        print(f"[pyptx ] M={M:5d} N={N:5d} K={K:5d}: {ms:7.3f} ms, {tflops:6.1f} TFLOPS")

        for _ in range(5):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        ms_cublas = start.elapsed_time(end) / iters
        tflops_cublas = flops / (ms_cublas * 1e-3) / 1e12
        print(f"[cuBLAS] M={M:5d} N={N:5d} K={K:5d}: {ms_cublas:7.3f} ms, {tflops_cublas:6.1f} TFLOPS")
        return tflops, tflops_cublas


    def run_torch_2sm(M: int, N: int, K: int) -> bool:
        import torch

        k = build_gemm_2sm(M, N, K)
        seed = M * 10007 + N * 313 + K
        rng = np.random.default_rng(seed)
        a_np = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
        b_np = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
        a = torch.tensor(a_np, device="cuda", dtype=torch.bfloat16)
        b = torch.tensor(b_np, device="cuda", dtype=torch.bfloat16)
        b_t = b.transpose(0, 1).contiguous()
        out = k(a, b_t)
        torch.cuda.synchronize()
        ref = a.float() @ b.float()
        diff = float((out - ref).abs().max())
        ok = bool(torch.allclose(out, ref, atol=5e-2, rtol=5e-2))
        status = "OK  " if ok else "FAIL"
        print(f"[Torch 2SM {status}] M={M:5d} N={N:5d} K={K:5d}  max_abs={diff:.3e}")
        return ok


    def main() -> None:
        print("=== 1SM Correctness ===")
        for M, N, K in [
            (128, 256, 64),
            (256, 256, 64),
            (128, 256, 128),
            (256, 512, 128),
            (512, 512, 256),
            (1024, 1024, 512),
        ]:
            run_torch(M, N, K)

        print("\n=== 2SM Correctness ===")
        for M, N, K in [
            (256, 256, 64),
            (256, 256, 128),
            (512, 512, 256),
            (1024, 1024, 512),
        ]:
            run_torch_2sm(M, N, K)

        print("\n=== Benchmark (1SM) ===")
        for M, N, K in [
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
        ]:
            try:
                benchmark(M, N, K)
            except Exception as e:
                print(f"[FAIL] {M}x{N}x{K}: {e}")


    if __name__ == "__main__":
        main()
    ```
