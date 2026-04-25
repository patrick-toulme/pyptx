"""Tests for the higher-level DSL wrappers used by handwritten kernels."""

from pyptx import kernel, reg, ptx, smem
from pyptx.specs import Layout, Tile
from pyptx.types import b32, b16, f32, u32, u64, pred, bf16, s32, b64


def _emit_ptx(fn):
    """Helper: trace a @kernel function and return PTX text."""
    k = kernel(arch="sm_90a")(fn)
    return k.ptx()


def _emit_raw_param_ptx(fn, raw_params, *, extern_smem=False):
    k = kernel(arch="sm_90a", raw_params=raw_params, extern_smem=extern_smem)(fn)
    return k.ptx()


class TestSetmaxnreg:
    def test_inc(self):
        def k():
            ptx.setmaxnreg(240, inc=True)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "setmaxnreg.inc.sync.aligned.u32 240" in ptx_text

    def test_dec(self):
        def k():
            ptx.setmaxnreg(40, inc=False)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "setmaxnreg.dec.sync.aligned.u32 40" in ptx_text


class TestSelp:
    def test_selp_u32(self):
        def k():
            p = reg.scalar(pred)
            dst = reg.scalar(u32)
            ptx.selp(u32, dst, 1, 0, p)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "selp.u32" in ptx_text

    def test_selp_b32(self):
        def k():
            p = reg.scalar(pred)
            a = reg.scalar(b32)
            b = reg.scalar(b32)
            dst = reg.scalar(b32)
            ptx.selp(b32, dst, a, b, p)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "selp.b32" in ptx_text


class TestIntegerHelpers:
    def test_reg_max(self):
        def k():
            x = reg.scalar(u32)
            y = x.max(2)
            ptx.inst.mov.u32(x, y)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "max.u32" in ptx_text

    def test_signed_sub_preserves_dtype(self):
        def k():
            x = reg.scalar(s32, init=5)
            y = x - 1
            ptx.inst.mov.s32(x, y)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "sub.s32" in ptx_text
        assert "mov.s32" in ptx_text

    def test_kloop_unroll_and_tail(self):
        def k():
            total = reg.scalar(u32, init=5)
            acc = reg.scalar(u32, init=0)

            def body(acc=acc):
                acc += 1

            ptx.kloop(total, unroll=4, loop_label="kloop_test", body=body)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "kloop_test:" in ptx_text
        assert "setp.ne.u32" in ptx_text
        assert ptx_text.count("add.u32") >= 5

    def test_mad_expression(self):
        def k():
            a = reg.scalar(u32)
            b = reg.scalar(u32)
            c = reg.scalar(u32)
            dst = ptx.mad(a, b, c)
            ptx.inst.mov.u32(a, dst)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "mad.lo.s32" in ptx_text
        assert "mov.u32" in ptx_text


class TestPredicateExpressions:
    def test_if_accepts_comparison_expr(self):
        def k():
            tid = reg.scalar(u32)
            with ptx.if_(tid == 0):
                ptx.inst.mov.u32(tid, 1)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "setp.eq.u32" in ptx_text

    def test_bra_accepts_comparison_expr(self):
        def k():
            tid = reg.scalar(u32)
            ptx.bra("done", pred=(tid == 0))
            ptx.label("done")
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "setp.eq.u32" in ptx_text
        assert "bra done;" in ptx_text


class TestStmatrix:
    def test_stmatrix_x4_trans(self):
        def k():
            from pyptx import smem
            addr = smem.alloc(bf16, (64, 64))
            acc = reg.array(f32, 4)
            ptx.stmatrix(smem=addr[0], regs=[acc[0], acc[1], acc[2], acc[3]], layout="x4.trans")
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "stmatrix.sync.aligned" in ptx_text

    def test_stmatrix_x4_trans_f32_bf16(self):
        def k():
            base = reg.scalar(u32)
            lane = reg.scalar(u32)
            frag = reg.array(f32, 8)
            tmp_bf16 = [reg.scalar(b16) for _ in range(8)]
            tmp_pack = [reg.scalar(b32) for _ in range(4)]
            ptx.stmatrix_x4_trans_f32_bf16(
                frag=frag,
                smem_base=base,
                lane=lane,
                row_stride=72,
                tmp_bf16=tmp_bf16,
                tmp_pack=tmp_pack,
            )
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "cvt.rn.bf16.f32" in ptx_text
        assert "stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16" in ptx_text


class TestWgmmaWideKMajor:
    def test_auto_descriptor_accepts_wide_k_major_tile(self):
        def k():
            tile = smem.wgmma_tile(bf16, (64, 128), major="K")
            desc = ptx.wgmma.auto_descriptor(
                tile,
                dtype=bf16,
                shape=(64, 128),
                major="K",
            )
            sink = reg.scalar(b64)
            ptx.inst.mov.b64(sink, desc)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert ".shared .align 128 .b8 smem_" in ptx_text
        assert "mov.b64 %rd" in ptx_text


class TestRegArrayHelpers:
    def test_hw_order_reverse(self):
        def k():
            regs = reg.array(f32, 4)
            ordered = regs.hw_order(reverse=True)
            dst = reg.scalar(b32)
            ptx.inst.mov.b32(dst, [ordered[0], ordered[1]])
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "{%f3, %f2}" in ptx_text

    def test_wgmma_frag(self):
        def k():
            frag = reg.wgmma_frag(m=64, n=256, dtype=f32)
            ordered = frag.hw_order(reverse=True)
            ptx.inst.mov.f32(frag[0], ordered[0])
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert ".reg .f32 %f<128>;" in ptx_text


class TestCluster:
    def test_cluster_sync(self):
        def k():
            ptx.cluster.sync()
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "barrier.cluster.arrive" in ptx_text
        assert "barrier.cluster.wait" in ptx_text

    def test_cluster_arrive_remote(self):
        def k():
            bar_addr = reg.scalar(u32)
            cta_id = reg.scalar(u32)
            ptx.cluster.arrive_remote(bar_addr, cta_id, 1)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "mapa.shared::cluster.u32" in ptx_text
        assert "mbarrier.arrive.shared::cluster.b64" in ptx_text
        # Should be inside a { } scope
        assert "{" in ptx_text

    def test_named_barrier_sync(self):
        def k():
            bid = reg.scalar(u32, init=3)
            bar = ptx.named_barrier(bid, count=128)
            bar.sync()
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "bar.sync %r0, 128;" in ptx_text

    def test_cluster_rank_predicate(self):
        def k():
            with ptx.if_(ptx.cluster.rank(0)):
                x = reg.scalar(u32)
                ptx.inst.mov.u32(x, 1)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "%cluster_ctarank" in ptx_text
        assert "setp.eq.u32" in ptx_text


class TestBarrierArray:
    def test_barrier_array_helpers(self):
        def k():
            smem_base = reg.scalar(u32)
            ptx.inst.mov.u32(smem_base, "smem")
            full = ptx.mbarrier.array(smem_base, 64, 3)
            stage = reg.scalar(u32, init=1)
            phase = reg.scalar(u32, init=0)
            full.init_all(2)
            full.at(stage).wait(phase)
            full.arrive_remote_all(stage, pred=(stage < 2))
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "mbarrier.init.shared::cta.b64 [%r0+64], 2;" in ptx_text
        assert "mbarrier.init.shared::cta.b64 [%r0+72], 2;" in ptx_text
        assert "mbarrier.init.shared::cta.b64 [%r0+80], 2;" in ptx_text
        assert "mbarrier.try_wait.parity.shared.b64" in ptx_text
        assert ptx_text.count("mbarrier.arrive.shared::cluster.b64 _, [remAddr32], 1;") == 3

    def test_static_mbarrier_does_not_inherit_prior_smem_offset(self):
        def k():
            smem.alloc(bf16, (128, 64), swizzle="128B")
            bar = smem.mbarrier(1)
            ptx.mbarrier.init(bar[0], 1)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "mbarrier.init.shared::cta.b64 [mbar_" in ptx_text
        assert "+16384" not in ptx_text

    def test_auto_dynamic_smem_retraces_without_orphaned_symbols(self):
        def k():
            smem.alloc(bf16, (256, 128), swizzle="128B")  # 64 KiB
            bar = smem.mbarrier(2)
            ptx.mbarrier.init(bar[0], 1)
            ptx.mbarrier.init(bar[1], 1)
            ptx.mbarrier.arrive_expect_tx(bar[0], 4096)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert ".extern .shared .align 128 .b8 dyn_smem[];" in ptx_text
        assert "mbarrier.init.shared::cta.b64 [mbar_" not in ptx_text
        assert "mbarrier.arrive.expect_tx.shared::cta.b64 %rd" in ptx_text
        assert "smem_0" not in ptx_text
        assert "mbar_0" not in ptx_text


class TestPipelineState:
    def test_pipeline_advance_emits_wrap_logic(self):
        def k():
            cursor = reg.scalar(u32, init=0)
            phase = reg.scalar(u32, init=0)
            pipe = ptx.pipeline(3, cursor=cursor, phase=phase)
            pipe.advance()
            pipe.advance()
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert ptx_text.count("setp.eq.u32") == 2
        assert ptx_text.count("selp.u32") == 4
        assert ptx_text.count("xor.b32") == 2


class TestWgmmaHelpers:
    def test_masked_descriptor(self):
        def k():
            base = reg.scalar(u32, init=1024)
            ptx.wgmma.masked_descriptor(base, byte_offset=-8192, mask=262016)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "add.u32" in ptx_text
        assert "and.b32" in ptx_text
        assert "shr.u32" in ptx_text
        assert "cvt.u64.u32" in ptx_text
        assert "or.b64" in ptx_text


class TestTma3d:
    def test_tma_load_3d(self):
        def k():
            dst = reg.scalar(u32)
            src = reg.scalar(u64)
            mbar = reg.scalar(u32)
            row = reg.scalar(u32)
            col = reg.scalar(u32)
            ptx.tma.load_3d(dst=dst, src=src, row=row, col=col, mbar=mbar)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes" in ptx_text

    def test_tma_load_3d_multicast(self):
        def k():
            dst = reg.scalar(u32)
            src = reg.scalar(u64)
            mbar = reg.scalar(u32)
            row = reg.scalar(u32)
            col = reg.scalar(u32)
            mask = reg.scalar(b16)
            ptx.tma.load_3d_multicast(dst=dst, src=src, row=row, col=col, mbar=mbar, mask=mask)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "multicast::cluster" in ptx_text

    def test_tma_load_3d_multicast_issuer(self):
        def k():
            dst = reg.scalar(u32)
            src = reg.scalar(u64)
            mbar = reg.scalar(u32)
            row = reg.scalar(u32)
            col = reg.scalar(u32)
            mask = reg.scalar(b16)
            ptx.tma.load_3d_multicast(
                dst=dst,
                src=src,
                row=row,
                col=col,
                mbar=mbar,
                mask=mask,
                issuer=ptx.cluster.rank(0),
            )
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "%cluster_ctarank" in ptx_text
        assert "@%p" in ptx_text
        assert "multicast::cluster" in ptx_text

    def test_cvta_helpers(self):
        def k():
            src = reg.scalar(u64)
            dst_param = ptx.cvta.param(src)
            dst_global = ptx.cvta.to_global(src)
            ptx.inst.mov.b64(src, dst_param)
            ptx.inst.mov.b64(src, dst_global)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "cvta.param.u64" in ptx_text
        assert "cvta.to.global.u64" in ptx_text

    def test_tma_store_3d(self):
        def k():
            dst = reg.scalar(u64)
            src = reg.scalar(u32)
            row = reg.scalar(u32)
            col = reg.scalar(u32)
            ptx.tma.store_3d(dst=dst, src=src, row=row, col=col)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group" in ptx_text

    def test_tma_load_3d_int_col(self):
        """Col as integer should be divided by 64 at trace time."""
        def k():
            dst = reg.scalar(u32)
            src = reg.scalar(u64)
            mbar = reg.scalar(u32)
            row = reg.scalar(u32)
            ptx.tma.load_3d(dst=dst, src=src, row=row, col=256, mbar=mbar)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        # 256/64 = 4 should appear as the third coordinate
        assert ", 4}" in ptx_text

    def test_tma_coords_bypass_col_shift(self):
        def k():
            dst = reg.scalar(u32)
            src = reg.scalar(u64)
            mbar = reg.scalar(u32)
            row = reg.scalar(u32)
            col = reg.scalar(u32)
            mask = reg.scalar(b16)
            ptx.tma.load_3d(dst=dst, src=src, coords=(0, row, col), mbar=mbar)
            ptx.tma.load_3d_multicast(dst=dst, src=src, coords=(0, row, col), mbar=mbar, mask=mask)
            ptx.tma.store_3d(dst=src, src=dst, coords=(0, row, col))
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "{0, %r2, %r3}" in ptx_text
        assert "shr.u32" not in ptx_text
        assert "multicast::cluster" in ptx_text
        assert "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes" in ptx_text
        assert "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group" in ptx_text


class TestParamHelpers:
    def test_scalar_param_helper(self):
        def k():
            x = ptx.param(u32, "K")
            ptx.inst.mov.u32(x, x)
            ptx.ret()
        ptx_text = _emit_raw_param_ptx(k, raw_params=[("u32", "K")])
        assert "ld.param.u32" in ptx_text

    def test_array_param_helper_uses_symbol_mov(self):
        def k():
            tma = ptx.param(b64, "tma_A")
            ptx.inst.mov.b64(tma, tma)
            ptx.ret()
        ptx_text = _emit_raw_param_ptx(k, raw_params=[("b8.align64.array128", "tma_A")])
        assert "mov.b64" in ptx_text
        assert "[tma_A]" not in ptx_text


class TestRegConstructors:
    def test_reg_from_symbol(self):
        def k():
            addr = reg.from_("smem", u32)
            ptx.inst.mov.u32(addr, addr)
            ptx.ret()
        ptx_text = _emit_ptx(k)
        assert "mov.u32" in ptx_text
        assert "smem" in ptx_text


class TestSmemHelpers:
    def test_smem_base_load_store(self):
        def k():
            base = smem.base()
            value = reg.scalar(u32)
            smem.store(u32, ptx.addr(base, 64), value)
            loaded = smem.load(u32, ptx.addr(base, 64))
            ptx.inst.mov.u32(value, loaded)
            ptx.ret()
        ptx_text = _emit_raw_param_ptx(k, raw_params=[], extern_smem="smem")
        assert "mov.u32" in ptx_text
        assert "ld.shared.u32" in ptx_text
        assert "st.shared.u32" in ptx_text


class TestCallableRawParams:
    def test_callable_kernel_mixes_tensor_raw_and_tma_params(self):
        @kernel(
            arch="sm_90a",
            in_specs=(Tile("M", "K", bf16, Layout.TMA_128B),),
            out_specs=(Tile("M", "K", bf16, Layout.ROW),),
            raw_params=[("u32", "K")],
        )
        def k(A, Out):
            desc = A.tma_desc()
            kval = ptx.param(u32, "K")
            ptx.inst.mov.b64(desc, desc)
            ptx.inst.mov.u32(kval, kval)
            ptx.ret()

        ptx_text = k.ptx(M=64, K=64)
        assert ".param .u64 A" in ptx_text
        assert ".param .u64 Out" in ptx_text
        assert ".param .u32 K" in ptx_text
        assert ".param .u64 A_tma_desc" in ptx_text
        assert ptx_text.index(".param .u32 K") > ptx_text.index(".param .u64 Out")
        assert ptx_text.index(".param .u64 A_tma_desc") > ptx_text.index(".param .u32 K")

    def test_tile_supports_rank3_tma_metadata(self):
        tile = Tile("M", "K", bf16, Layout.TMA_128B, tma_box=(128, 64), tma_rank=3, tma_padding=True)
        assert tile.tma_box == (128, 64)
        assert tile.tma_rank == 3
        assert tile.tma_padding is True
