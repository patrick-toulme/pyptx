# Blackwell / Tcgen05 Ld Register Probe

[:material-github: View on GitHub](https://github.com/patrick-toulme/pyptx/blob/dev/examples/blackwell/tcgen05_ld_register_probe.py){ .md-button } 
[:material-file-code: `examples/blackwell/tcgen05_ld_register_probe.py`](https://github.com/patrick-toulme/pyptx/blob/dev/examples/blackwell/tcgen05_ld_register_probe.py){ .md-button }

## Overview

Blackwell tcgen05 LD register payload probe.

Run one 128x256x64 UMMA tile, then issue a single tcgen05.ld from TMEM and dump
the raw register payload for the first 64-column subtile. This isolates the
per-thread TMEM load payload from the higher-level GEMM epilogue scatter.

## Source

??? example "Full source"

    ```python
    """Blackwell tcgen05 LD register payload probe.

    Run one 128x256x64 UMMA tile, then issue a single tcgen05.ld from TMEM and dump
    the raw register payload for the first 64-column subtile. This isolates the
    per-thread TMEM load payload from the higher-level GEMM epilogue scatter.
    """
    from __future__ import annotations

    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import torch

    from pyptx import Tile, kernel, ptx, reg, smem
    from pyptx.smem import apply_swizzle
    from pyptx.specs import Layout
    from pyptx.types import b32, b64, bf16, f32, pred, u32, u64


    BM = 128
    BN = 256
    BK = 64
    A_BYTES = BM * BK * 2
    B_BYTES = BN * BK * 2
    A_FOOTPRINT = A_BYTES
    B_FOOTPRINT = BN * BN
    MMA_BAR_OFF = A_FOOTPRINT + B_FOOTPRINT
    TMEM_SLOT_OFF = MMA_BAR_OFF + 16
    SMEM_BYTES = TMEM_SLOT_OFF + 16


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
        contig_elems = {"32B": 16, "64B": 32, "128B": 64}[kmajor_swizzle(row_stride_elems)]
        row_group = row >> 3
        row_in_group = row & 7
        if contig_elems == 64:
            row_group_bytes = mn_extent * 8
        else:
            row_group_bytes = contig_elems * 8 * 2
        return (row_group * row_group_bytes) + ((row_in_group * contig_elems) + k_elem) * 2


    def build(*, load_count: int = 64, arch: str = "sm_100a"):
        if load_count not in (1, 2, 4, 8, 16, 32, 64, 128):
            raise ValueError(f"unsupported load_count={load_count}")
        operand_swizzle = kmajor_swizzle(BK)

        @kernel(
            in_specs=(
                Tile(BM, BK, bf16, Layout.ROW),
                Tile(BN, BK, bf16, Layout.ROW),
            ),
            out_specs=(Tile(BM, load_count, f32, Layout.ROW),),
            grid=(1, 1, 1),
            block=(128, 1, 1),
            arch=arch,
            smem=SMEM_BYTES,
            extern_smem=True,
        )
        def k(A, B_T, O):
            base = smem.base()
            mma_bar = base + MMA_BAR_OFF
            tmem_slot = base + TMEM_SLOT_OFF

            tid = reg.scalar(u32)
            lane = reg.scalar(u32)
            warp = reg.scalar(u32)
            alloc_warp = reg.scalar(pred)
            epilogue_thread = reg.scalar(pred)
            ready = reg.scalar(pred)
            ptx.inst.mov.u32(tid, ptx.special.tid.x())
            ptx.inst.and_.b32(lane, tid, 31)
            ptx.inst.shr.u32(warp, tid, 5)
            ptx.inst.setp.lt.u32(alloc_warp, tid, 32)
            ptx.inst.setp.lt.u32(epilogue_thread, tid, 128)

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

            pa, pb, po = ptx.global_ptrs(A, B_T, O)

            a_words = A_BYTES // 4
            a_idx = reg.scalar(u32)
            ptx.inst.mov.u32(a_idx, tid)
            a_keep = reg.scalar(pred)
            ptx.inst.setp.lt.u32(a_keep, a_idx, a_words)
            with ptx.loop("copy_a_loop", pred=a_keep):
                word_index = reg.scalar(u64)
                ptx.inst.cvt.u64.u32(word_index, a_idx)
                g_ptr = pa + (word_index << 2)
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
            b_idx = reg.scalar(u32)
            ptx.inst.mov.u32(b_idx, tid)
            b_keep = reg.scalar(pred)
            ptx.inst.setp.lt.u32(b_keep, b_idx, b_words)
            with ptx.loop("copy_b_loop", pred=b_keep):
                word_index = reg.scalar(u64)
                ptx.inst.cvt.u64.u32(word_index, b_idx)
                g_ptr = pb + (word_index << 2)
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
                stride_bytes=BM * 8,
                leading_bytes=16,
                swizzle=operand_swizzle,
            )
            desc_b0 = ptx.tcgen05.descriptor(
                base,
                byte_offset=A_FOOTPRINT,
                stride_bytes=BN * 8,
                leading_bytes=16,
                swizzle=operand_swizzle,
            )

            for phase in range(2):
                for kk in range(phase * 2, phase * 2 + 2):
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
                ptx.label(f"wait_{phase}")
                ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(ready, ptx.addr(mma_bar), 1)
                ptx.bra(f"done_{phase}", pred=ready)
                ptx.bra(f"wait_{phase}")
                ptx.label(f"done_{phase}")
                if phase == 0:
                    with ptx.if_(tid == 0):
                        ptx.mbarrier.init(mma_bar, 1)
                    ptx.bar.sync(0)

            out_bits = reg.array(b32, load_count)
            out_vals = reg.array(f32, load_count)
            with ptx.if_(epilogue_thread):
                row = reg.scalar(u32)
                ptx.inst.shl.b32(row, warp, 5)
                ptx.inst.add.u32(row, row, lane)
                row_off = reg.scalar(u64)
                ptx.inst.mul.wide.u32(row_off, row, load_count)
                d_ptr = po + (row_off << 2)

                tmem_addr = tmem_base + ((tid << 16) & 0x600000)
                ptx.tcgen05.ld(
                    [out_bits[i] for i in range(load_count)],
                    tmem_addr,
                    shape="32x32b",
                    count=load_count,
                    dtype="b32",
                )
                ptx.tcgen05.wait_ld()

                for col in range(load_count):
                    ptx.inst.mov.b32(out_vals[col], out_bits[col])
                    ptx.inst.st.global_.f32(ptx.addr(d_ptr, col * 4), out_vals[col])

            with ptx.if_(alloc_warp):
                ptx.tcgen05.dealloc(tmem_base, 512)
                ptx.tcgen05.relinquish_alloc_permit()
            ptx.ret()

        return k


    def run():
        a = torch.zeros((BM, BK), device="cuda", dtype=torch.bfloat16)
        b = torch.zeros((BK, BN), device="cuda", dtype=torch.bfloat16)
        a[:, 0] = 1
        b[0, :] = torch.arange(1, BN + 1, device="cuda", dtype=torch.float32).to(torch.bfloat16)
        for load_count in (16, 32, 64):
            out = build(load_count=load_count)(a, b.t().contiguous())
            torch.cuda.synchronize()
            row0 = out[0].float().cpu()
            nz = torch.nonzero(row0).flatten().tolist()
            print(f"[x{load_count}] row0 nz", nz)
            print(f"[x{load_count}] row0 vals", row0[nz].tolist())


    if __name__ == "__main__":
        run()
    ```
