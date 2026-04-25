# Blackwell / Gemm Experimental Blackwell

[:material-github: View on GitHub](https://github.com/patrick-toulme/pyptx/blob/dev/examples/blackwell/gemm_experimental_blackwell.py){ .md-button } 
[:material-file-code: `examples/blackwell/gemm_experimental_blackwell.py`](https://github.com/patrick-toulme/pyptx/blob/dev/examples/blackwell/gemm_experimental_blackwell.py){ .md-button }

## Overview

Experimental Blackwell GEMM variants.

This module keeps the non-maintained Blackwell GEMM paths out of the
production ``gemm_highperf_blackwell.py`` entrypoint:

- ``build_gemm_persistent``: persistent 1SM variant.
- ``build_gemm_pallas_experimental``: Pallas-style dedicated-store warpgroup
  overlap kernel.
- ``build_gemm_no_tma_debug``: no-TMA debug kernel used by the tcgen05
  bring-up probes.

## Source

??? example "Full source"

    ```python
    """Experimental Blackwell GEMM variants.

    This module keeps the non-maintained Blackwell GEMM paths out of the
    production ``gemm_highperf_blackwell.py`` entrypoint:

    - ``build_gemm_persistent``: persistent 1SM variant.
    - ``build_gemm_pallas_experimental``: Pallas-style dedicated-store warpgroup
      overlap kernel.
    - ``build_gemm_no_tma_debug``: no-TMA debug kernel used by the tcgen05
      bring-up probes.
    """
    from __future__ import annotations

    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from pyptx import Tile, kernel, ptx, reg, smem
    from pyptx.smem import apply_swizzle
    from pyptx.specs import Layout
    from pyptx.types import b32, b64, bf16, f32, pred, u32, u64

    try:
        from pyptx.examples.blackwell.gemm_highperf_blackwell import (
            _build_gemm_persistent_overlap_experimental,
            build_gemm_persistent,
        )
    except ImportError:
        from examples.blackwell.gemm_highperf_blackwell import (
            _build_gemm_persistent_overlap_experimental,
            build_gemm_persistent,
        )


    BM = 128
    BN = 256
    BK = 64
    A_BYTES = BM * BK * 2
    B_BYTES = BN * BK * 2
    A_FOOTPRINT = A_BYTES
    B_FOOTPRINT = B_BYTES
    MMA_BAR_OFF = A_FOOTPRINT + B_FOOTPRINT
    TMEM_SLOT_OFF = MMA_BAR_OFF + 16
    SMEM_BYTES = TMEM_SLOT_OFF + 16


    def build_gemm_pallas_experimental(*args, **kwargs):
        return _build_gemm_persistent_overlap_experimental(*args, **kwargs)


    def kmajor_swizzle(row_stride_elems: int) -> str:
        row_bytes = row_stride_elems * 2
        if row_bytes >= 128:
            return "128B"
        if row_bytes >= 64:
            return "64B"
        if row_bytes >= 32:
            return "32B"
        raise ValueError(f"unsupported Blackwell K-major row width: {row_stride_elems} elems")


    def kmajor_swizzled_logical_bytes(row, k_elem, row_stride_elems, mn_extent):
        del mn_extent
        contig_elems = {"32B": 16, "64B": 32, "128B": 64}[kmajor_swizzle(row_stride_elems)]
        row_group = row >> 3
        row_in_group = row & 7
        row_group_bytes = contig_elems * 8 * 2
        return (row_group * row_group_bytes) + ((row_in_group * contig_elems) + k_elem) * 2


    def build_gemm_no_tma_debug(M: int, N: int, K: int, *, arch: str = "sm_100a"):
        assert arch.startswith("sm_100")
        assert M % BM == 0
        assert N % BN == 0
        assert K == BK

        operand_swizzle = kmajor_swizzle(BK)

        @kernel(
            in_specs=(
                Tile(M, K, bf16, Layout.ROW),
                Tile(N, K, bf16, Layout.ROW),
            ),
            out_specs=(Tile(M, N, f32, Layout.ROW),),
            grid=(M // BM, N // BN, 1),
            block=(128, 1, 1),
            arch=arch,
            smem=SMEM_BYTES,
            extern_smem=True,
        )
        def blackwell_gemm_no_tma(A, B_T, D):
            base = smem.base()
            mma_bar = base + MMA_BAR_OFF
            tmem_slot = base + TMEM_SLOT_OFF

            tid = reg.scalar(u32)
            lane = reg.scalar(u32)
            warp = reg.scalar(u32)
            cta_m = reg.scalar(u32)
            cta_n = reg.scalar(u32)
            alloc_warp = reg.scalar(pred)
            epilogue_thread = reg.scalar(pred)
            commit_ready = reg.scalar(pred)
            ptx.inst.mov.u32(tid, ptx.special.tid.x())
            ptx.inst.and_.b32(lane, tid, 31)
            ptx.inst.shr.u32(warp, tid, 5)
            ptx.inst.mov.u32(cta_m, ptx.special.ctaid.x())
            ptx.inst.mov.u32(cta_n, ptx.special.ctaid.y())
            ptx.inst.setp.lt.u32(alloc_warp, tid, 32)
            ptx.inst.setp.lt.u32(epilogue_thread, tid, 128)

            m_base = cta_m << 7
            n_base = cta_n << 8

            with ptx.if_(tid == 0):
                ptx.mbarrier.init(mma_bar, 1)
            with ptx.if_(alloc_warp):
                ptx.tcgen05.alloc(tmem_slot, 512)
            ptx.bar.sync(0)

            tmem_base = smem.load(b32, ptx.addr(tmem_slot))
            idesc = reg.scalar(
                b32,
                init=ptx.tcgen05.make_instr_desc_f16bf16_f32(ab_dtype="bf16"),
            )

            pa, pb, pd = ptx.global_ptrs(A, B_T, D)

            a_words = A_BYTES // 4
            a_base_words = reg.scalar(u64)
            ptx.inst.mul.wide.u32(a_base_words, m_base, BK // 2)
            a_idx = reg.scalar(u32)
            ptx.inst.mov.u32(a_idx, tid)
            a_keep = reg.scalar(pred)
            ptx.inst.setp.lt.u32(a_keep, a_idx, a_words)
            with ptx.loop("copy_a_loop", pred=a_keep):
                word_index = reg.scalar(u64)
                ptx.inst.cvt.u64.u32(word_index, a_idx)
                g_index = a_base_words + word_index
                g_ptr = pa + (g_index << 2)
                word = reg.scalar(b32)
                ptx.inst.ld.global_.b32(word, ptx.addr(g_ptr))
                row = reg.scalar(u32)
                k_word = reg.scalar(u32)
                logical = reg.scalar(u32)
                ptx.inst.shr.u32(row, a_idx, 5)
                ptx.inst.and_.b32(k_word, a_idx, (BK // 2) - 1)
                logical = kmajor_swizzled_logical_bytes(row, k_word << 1, BK, BM)
                physical = apply_swizzle(logical, operand_swizzle)
                ptx.inst.st.shared.b32(ptx.addr(base + physical), word)
                a_idx += 128
                ptx.inst.setp.lt.u32(a_keep, a_idx, a_words)

            b_words = B_BYTES // 4
            b_base_words = reg.scalar(u64)
            ptx.inst.mul.wide.u32(b_base_words, n_base, BK // 2)
            b_idx = reg.scalar(u32)
            ptx.inst.mov.u32(b_idx, tid)
            b_keep = reg.scalar(pred)
            ptx.inst.setp.lt.u32(b_keep, b_idx, b_words)
            with ptx.loop("copy_b_loop", pred=b_keep):
                word_index = reg.scalar(u64)
                ptx.inst.cvt.u64.u32(word_index, b_idx)
                g_index = b_base_words + word_index
                g_ptr = pb + (g_index << 2)
                word = reg.scalar(b32)
                ptx.inst.ld.global_.b32(word, ptx.addr(g_ptr))
                row = reg.scalar(u32)
                k_word = reg.scalar(u32)
                logical = reg.scalar(u32)
                ptx.inst.shr.u32(row, b_idx, 5)
                ptx.inst.and_.b32(k_word, b_idx, (BK // 2) - 1)
                logical = kmajor_swizzled_logical_bytes(row, k_word << 1, BK, BN)
                physical = apply_swizzle(logical, operand_swizzle)
                ptx.inst.st.shared.b32(ptx.addr(base + A_FOOTPRINT + physical), word)
                b_idx += 128
                ptx.inst.setp.lt.u32(b_keep, b_idx, b_words)

            ptx.bar.sync(0)

            desc_a0 = ptx.tcgen05.descriptor(
                base,
                stride_bytes=BK * 16,
                leading_bytes=16,
                swizzle=operand_swizzle,
            )
            desc_b0 = ptx.tcgen05.descriptor(
                base,
                byte_offset=A_FOOTPRINT,
                stride_bytes=BK * 16,
                leading_bytes=16,
                swizzle=operand_swizzle,
            )
            for phase in range(2):
                phase_start = phase * 2
                phase_end = phase_start + 2
                for kk in range(phase_start, phase_end):
                    if kk == 0:
                        desc_a = desc_a0
                        desc_b = desc_b0
                    else:
                        desc_a = reg.scalar(b64)
                        desc_b = reg.scalar(b64)
                        ptx.inst.add.s64(desc_a, desc_a0, kk * 2)
                        ptx.inst.add.s64(desc_b, desc_b0, kk * 2)
                    with ptx.if_(tid == 0):
                        ptx.tcgen05.mma(
                            tmem_base,
                            desc_a,
                            desc_b,
                            idesc,
                            kind="f16",
                            pred_operand=(kk != 0),
                        )
                with ptx.if_(tid == 0):
                    ptx.tcgen05.commit(mma_bar, space="cluster")

                ptx.label(f"commit_wait_{phase}")
                ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
                    commit_ready, ptx.addr(mma_bar), 1
                )
                ptx.bra(f"commit_done_{phase}", pred=commit_ready)
                ptx.bra(f"commit_wait_{phase}")
                ptx.label(f"commit_done_{phase}")

                if phase == 0:
                    with ptx.if_(tid == 0):
                        ptx.mbarrier.init(mma_bar, 1)
                    ptx.bar.sync(0)

            out = reg.array(b32, BN)
            with ptx.if_(epilogue_thread):
                row = reg.scalar(u32)
                ptx.inst.shl.b32(row, warp, 5)
                ptx.inst.add.u32(row, row, lane)
                row_base = (cta_m << 7) + row
                row_off = reg.scalar(u64)
                ptx.inst.mul.wide.u32(row_off, row_base, N)
                tile_col = reg.scalar(u64)
                ptx.inst.cvt.u64.u32(tile_col, n_base)
                d_index = row_off + tile_col
                d_ptr = pd + (d_index << 2)

                tmem_row_bits = (tid << 16) & 0x03E00000
                tmem_addr = tmem_base + tmem_row_bits
                for col in range(BN):
                    ptx.tcgen05.ld(
                        [out[col]],
                        tmem_addr + col,
                        shape="32x32b",
                        count=1,
                        dtype="b32",
                    )
                ptx.tcgen05.wait_ld()
                for vec in range(BN // 4):
                    off = vec * 16
                    ptx.inst.st.global_.v4.b32(
                        ptx.addr(d_ptr, off),
                        [out[vec * 4], out[vec * 4 + 1], out[vec * 4 + 2], out[vec * 4 + 3]],
                    )

            with ptx.if_(alloc_warp):
                ptx.tcgen05.dealloc(tmem_base, 512)
                ptx.tcgen05.relinquish_alloc_permit()
            ptx.ret()

        return blackwell_gemm_no_tma


    __all__ = [
        "build_gemm_no_tma_debug",
        "build_gemm_pallas_experimental",
        "build_gemm_persistent",
    ]
    ```
