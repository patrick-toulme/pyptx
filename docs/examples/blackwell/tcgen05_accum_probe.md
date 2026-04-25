# Blackwell / Tcgen05 Accum Probe

[:material-github: View on GitHub](https://github.com/patrick-toulme/pyptx/blob/dev/examples/blackwell/tcgen05_accum_probe.py){ .md-button } 
[:material-file-code: `examples/blackwell/tcgen05_accum_probe.py`](https://github.com/patrick-toulme/pyptx/blob/dev/examples/blackwell/tcgen05_accum_probe.py){ .md-button }

## Overview

Blackwell tcgen05 accumulation probe.

Run the same 128x256x16 UMMA four times against all-ones A/B tiles and read
back the first 32x64 accumulator subtile. Expected value is 64.0f if all four
MMAs accumulate correctly.

## Source

??? example "Full source"

    ```python
    """Blackwell tcgen05 accumulation probe.

    Run the same 128x256x16 UMMA four times against all-ones A/B tiles and read
    back the first 32x64 accumulator subtile. Expected value is 64.0f if all four
    MMAs accumulate correctly.
    """
    from __future__ import annotations

    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import torch

    from pyptx import Tile, kernel, ptx, reg, smem
    from pyptx.smem import apply_swizzle
    from pyptx.specs import Layout
    from pyptx.types import b32, b64, f32, pred, u32


    BM = 128
    BN = 256
    BK_MMA = 16
    A_BYTES = BM * BK_MMA * 2
    B_BYTES = BN * BK_MMA * 2
    BAR_OFF = A_BYTES + B_BYTES
    TMEM_SLOT_OFF = BAR_OFF + 16
    SMEM_BYTES = TMEM_SLOT_OFF + 16
    PACKED_BF16_ONE = 0x3F803F80


    def kmajor_swizzle(row_stride_elems: int) -> str:
        row_bytes = row_stride_elems * 2
        if row_bytes >= 128:
            return "128B"
        if row_bytes >= 64:
            return "64B"
        if row_bytes >= 32:
            return "32B"
        raise ValueError(f"unsupported Blackwell K-major row width: {row_stride_elems} elems")


    def kmajor_swizzled_logical_bytes(row, k_elem, row_stride_elems):
        contig_elems = {"32B": 16, "64B": 32, "128B": 64}[kmajor_swizzle(row_stride_elems)]
        row_group = row >> 3
        row_in_group = row & 7
        return ((row_group * (contig_elems * 8)) + (row_in_group * contig_elems) + k_elem) << 1


    def build(
        *,
        arch: str = "sm_100a",
        advance_descs: bool = True,
        num_mmas: int = 4,
        scale_c: int | None = None,
        split_after: int | None = None,
    ):
        assert 1 <= num_mmas <= 4
        if split_after is not None:
            assert 1 <= split_after < num_mmas
        operand_swizzle = kmajor_swizzle(BK_MMA)

        @kernel(
            in_specs=(Tile(1, 1, f32, Layout.ROW),),
            out_specs=(Tile(32, 64, f32, Layout.ROW),),
            grid=(1, 1, 1),
            block=(128, 1, 1),
            arch=arch,
            smem=SMEM_BYTES,
            extern_smem=True,
        )
        def k(_x, O):
            base = smem.base()
            bar = base + BAR_OFF
            tmem_slot = base + TMEM_SLOT_OFF

            tid = reg.scalar(u32)
            lane = reg.scalar(u32)
            alloc_warp = reg.scalar(pred)
            active_lane = reg.scalar(pred)
            ready = reg.scalar(pred)
            ptx.inst.mov.u32(tid, ptx.special.tid.x())
            ptx.inst.and_.b32(lane, tid, 31)
            ptx.inst.setp.lt.u32(alloc_warp, tid, 32)
            ptx.inst.setp.lt.u32(active_lane, tid, 32)

            with ptx.if_(tid == 0):
                ptx.mbarrier.init(bar, 1)
            with ptx.if_(alloc_warp):
                ptx.tcgen05.alloc(tmem_slot, 512)

            a_idx = reg.scalar(u32)
            ptx.inst.mov.u32(a_idx, tid)
            a_keep = reg.scalar(pred)
            ptx.inst.setp.lt.u32(a_keep, a_idx, A_BYTES // 4)
            with ptx.loop("fill_a_loop", pred=a_keep):
                row = reg.scalar(u32)
                k_word = reg.scalar(u32)
                logical = reg.scalar(u32)
                ptx.inst.shr.u32(row, a_idx, 3)
                ptx.inst.and_.b32(k_word, a_idx, (BK_MMA // 2) - 1)
                logical = kmajor_swizzled_logical_bytes(row, k_word << 1, BK_MMA)
                physical = apply_swizzle(logical, operand_swizzle)
                ptx.inst.st.shared.b32(ptx.addr(base + physical), PACKED_BF16_ONE)
                a_idx += 128
                ptx.inst.setp.lt.u32(a_keep, a_idx, A_BYTES // 4)

            b_idx = reg.scalar(u32)
            ptx.inst.mov.u32(b_idx, tid)
            b_keep = reg.scalar(pred)
            ptx.inst.setp.lt.u32(b_keep, b_idx, B_BYTES // 4)
            with ptx.loop("fill_b_loop", pred=b_keep):
                row = reg.scalar(u32)
                k_word = reg.scalar(u32)
                logical = reg.scalar(u32)
                ptx.inst.shr.u32(row, b_idx, 3)
                ptx.inst.and_.b32(k_word, b_idx, (BK_MMA // 2) - 1)
                logical = kmajor_swizzled_logical_bytes(row, k_word << 1, BK_MMA)
                physical = apply_swizzle(logical, operand_swizzle)
                ptx.inst.st.shared.b32(ptx.addr(base + A_BYTES + physical), PACKED_BF16_ONE)
                b_idx += 128
                ptx.inst.setp.lt.u32(b_keep, b_idx, B_BYTES // 4)

            ptx.bar.sync(0)

            tmem_base = smem.load(b32, ptx.addr(tmem_slot))
            idesc = reg.scalar(b32, init=ptx.tcgen05.make_instr_desc_f16bf16_f32())
            desc_a = ptx.tcgen05.descriptor(
                base,
                stride_bytes=BK_MMA * 16,
                leading_bytes=16,
                swizzle=operand_swizzle,
            )
            desc_b = ptx.tcgen05.descriptor(
                base,
                byte_offset=A_BYTES,
                stride_bytes=BK_MMA * 16,
                leading_bytes=16,
                swizzle=operand_swizzle,
            )

            first_batch = split_after if split_after is not None else num_mmas

            for step in range(first_batch):
                mma_desc_a = desc_a
                mma_desc_b = desc_b
                if advance_descs and step != 0:
                    mma_desc_a = reg.scalar(b64)
                    mma_desc_b = reg.scalar(b64)
                    ptx.inst.add.s64(mma_desc_a, desc_a, step * 2)
                    ptx.inst.add.s64(mma_desc_b, desc_b, step * 2)
                with ptx.if_(tid == 0):
                    ptx.tcgen05.mma(
                        tmem_base,
                        mma_desc_a,
                        mma_desc_b,
                        idesc,
                        kind="f16",
                        pred_operand=(step != 0),
                        scale_c=scale_c,
                    )
            with ptx.if_(tid == 0):
                ptx.tcgen05.commit(bar, space="cluster")

            ptx.label("wait_0")
            ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(ready, ptx.addr(bar), 1)
            ptx.bra("done_0", pred=ready)
            ptx.bra("wait_0")
            ptx.label("done_0")

            if split_after is not None:
                with ptx.if_(tid == 0):
                    ptx.mbarrier.init(bar, 1)
                ptx.bar.sync(0)

                for step in range(split_after, num_mmas):
                    mma_desc_a = desc_a
                    mma_desc_b = desc_b
                    if advance_descs and step != 0:
                        mma_desc_a = reg.scalar(b64)
                        mma_desc_b = reg.scalar(b64)
                        ptx.inst.add.s64(mma_desc_a, desc_a, step * 2)
                        ptx.inst.add.s64(mma_desc_b, desc_b, step * 2)
                    with ptx.if_(tid == 0):
                        ptx.tcgen05.mma(
                            tmem_base,
                            mma_desc_a,
                            mma_desc_b,
                            idesc,
                            kind="f16",
                            pred_operand=True,
                            scale_c=scale_c,
                        )
                with ptx.if_(tid == 0):
                    ptx.tcgen05.commit(bar, space="cluster")

                ptx.label("wait_1")
                ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(ready, ptx.addr(bar), 1)
                ptx.bra("done_1", pred=ready)
                ptx.bra("wait_1")
                ptx.label("done_1")

            out_bits = reg.array(b32, 64)
            out_vals = reg.array(f32, 64)
            tmem_addr = tmem_base + ((lane << 16) & 0x1F0000)
            with ptx.if_(active_lane):
                ptx.tcgen05.ld(
                    [out_bits[i] for i in range(64)],
                    tmem_addr,
                    shape="32x32b",
                    count=64,
                    dtype="b32",
                )
                ptx.tcgen05.wait_ld()

            (po,) = ptx.global_ptrs(O)
            row_off = reg.scalar(u32)
            ptx.inst.mul.lo.u32(row_off, lane, 64)
            base_ptr = po + (row_off << 2)
            with ptx.if_(active_lane):
                for col in range(64):
                    ptx.inst.mov.b32(out_vals[col], out_bits[col])
                    ptx.inst.st.global_.f32(ptx.addr(base_ptr, col * 4), out_vals[col])

            with ptx.if_(alloc_warp):
                ptx.tcgen05.dealloc(tmem_base, 512)
                ptx.tcgen05.relinquish_alloc_permit()
            ptx.ret()

        return k


    def run_torch():
        x = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
        out = build()(x)
        torch.cuda.synchronize()
        print(out[:4, :16].cpu())
        print("max_abs_to_64", float((out - 64.0).abs().max()))


    def run_torch_same_desc():
        x = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
        out = build(advance_descs=False)(x)
        torch.cuda.synchronize()
        print(out[:4, :16].cpu())
        print("same_desc_max_abs_to_64", float((out - 64.0).abs().max()))


    def run_torch_sweep():
        x = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
        for num_mmas in (1, 2, 3, 4):
            out = build(num_mmas=num_mmas, advance_descs=False)(x)
            torch.cuda.synchronize()
            got = float(out[0, 0].item())
            print(f"num_mmas={num_mmas} got={got:.1f}")


    def run_torch_scaled_sweep():
        x = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
        for num_mmas in (1, 2, 3, 4):
            out = build(num_mmas=num_mmas, advance_descs=False, scale_c=1)(x)
            torch.cuda.synchronize()
            got = float(out[0, 0].item())
            print(f"scaled num_mmas={num_mmas} got={got:.1f}")


    def run_torch_split_probe():
        x = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
        out = build(num_mmas=4, advance_descs=False, split_after=2)(x)
        torch.cuda.synchronize()
        print(f"split_after=2 got={float(out[0, 0].item()):.1f}")


    if __name__ == "__main__":
        run_torch()
    ```
