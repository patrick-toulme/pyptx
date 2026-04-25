"""GPU runtime tests for the @pyptx.kernel → @jax.jit pipeline.

These tests require:
  - jaxlib with the CUDA plugin (jax.default_backend() == "gpu")
  - cuda-python installed
  - The C++ launch shim built (pyptx/_shim/libpyptx_shim.so)
  - A real NVIDIA GPU accessible to the process

On CI or laptops without any of the above, the whole module is skipped.
These tests prove the full dispatch chain on a real device:

    @kernel body traced → PTX emitted → cuModuleLoadData driver JIT →
    pyptx_shim_register_launch(handle, cu_fn, grid, block, smem) →
    jax.ffi.register_ffi_target(CUDA, typed FFI api_version=1) →
    @jax.jit lowers to stablehlo.custom_call @pyptx_launch →
    C++ shim handler decodes stream+buffers+cubin_handle →
    cuLaunchKernel on XLA's stream →
    correct output.
"""

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pyptx import kernel, reg, smem, ptx, Tile, Layout
from pyptx.types import bf16, f32, b32, u32, u64, b64
from pyptx.jax_support import shim_is_available


_GPU_REQUIRED = pytest.mark.skipif(
    jax.default_backend() != "gpu" or not shim_is_available(),
    reason=(
        "GPU runtime tests require jax[cuda], the C++ launch shim, "
        "and a real NVIDIA device. "
        "Build the shim with pyptx/_shim/build.sh."
    ),
)


@_GPU_REQUIRED
class TestVectorAddExecution:
    """A real vector_add kernel written entirely in the pyptx DSL,
    dispatched through @jax.jit, producing correct output on the GPU."""

    def _build_kernel(self):
        @kernel(
            in_specs=(Tile("N", f32), Tile("N", f32)),
            out_specs=(Tile("N", f32),),
            grid=lambda N: (N // 256, 1, 1),
            block=(256, 1, 1),
            arch="sm_90a",
        )
        def vector_add(a, b, out):
            pa = reg.scalar(b64)
            pb = reg.scalar(b64)
            po = reg.scalar(b64)
            tid = reg.scalar(u32)
            ctaid = reg.scalar(u32)
            gtid = reg.scalar(u32)
            off64 = reg.scalar(u64)
            va = reg.scalar(f32)
            vb = reg.scalar(f32)
            vs = reg.scalar(f32)

            # Load pointer params from .param space
            ptx.inst.ld.param.u64(pa, ptx.addr(a))
            ptx.inst.ld.param.u64(pb, ptx.addr(b))
            ptx.inst.ld.param.u64(po, ptx.addr(out))

            # Generic → global pointers
            ptx.inst.cvta.to.global_.u64(pa, pa)
            ptx.inst.cvta.to.global_.u64(pb, pb)
            ptx.inst.cvta.to.global_.u64(po, po)

            # Global thread id = ctaid.x * 256 + tid.x
            ptx.inst.mov.u32(tid, ptx.special.tid.x())
            ptx.inst.mov.u32(ctaid, ptx.special.ctaid.x())
            ptx.inst.mad.lo.u32(gtid, ctaid, 256, tid)

            # Byte offset = gtid * 4 (f32 = 4 bytes)
            ptx.inst.mul.wide.u32(off64, gtid, 4)
            ptx.inst.add.s64(pa, pa, off64)
            ptx.inst.add.s64(pb, pb, off64)
            ptx.inst.add.s64(po, po, off64)

            # out[tid] = a[tid] + b[tid]
            ptx.inst.ld.global_.f32(va, ptx.addr(pa))
            ptx.inst.ld.global_.f32(vb, ptx.addr(pb))
            ptx.inst.add.f32(vs, va, vb)
            ptx.inst.st.global_.f32(ptx.addr(po), vs)

            ptx.ret()

        return vector_add

    def test_emitted_ptx_is_well_formed(self):
        """The traced PTX should have a proper param list and body."""
        vadd = self._build_kernel()
        text = vadd.ptx(N=1024)
        assert ".visible .entry vector_add(" in text
        assert ".param .u64 a" in text
        assert ".param .u64 b" in text
        assert ".param .u64 out" in text
        assert "ld.param.u64" in text
        assert "cvta.to.global.u64" in text
        assert "mad.lo.u32" in text
        assert "st.global.f32" in text

    def test_vector_add_correctness_1k(self):
        """arange(1024) + ones(1024)*10 → [10, 11, ..., 1033]."""
        vadd = self._build_kernel()

        x = jnp.arange(1024, dtype=jnp.float32)
        y = jnp.ones(1024, dtype=jnp.float32) * 10.0

        @jax.jit
        def fn(a, b):
            return vadd(a, b)

        out = fn(x, y)
        out.block_until_ready()

        expected = x + y
        assert jnp.allclose(out, expected), \
            f"max abs diff: {float(jnp.max(jnp.abs(out - expected)))}"

    def test_vector_add_correctness_different_shape(self):
        """Re-specializing on a larger shape should re-compile and still be correct."""
        vadd = self._build_kernel()

        x = jnp.arange(2048, dtype=jnp.float32)
        y = jnp.arange(2048, dtype=jnp.float32) * 2.0

        @jax.jit
        def fn(a, b):
            return vadd(a, b)

        out = fn(x, y)
        out.block_until_ready()
        assert jnp.allclose(out, x + y)

    def test_back_to_back_launches_use_xla_stream(self):
        """Two sequential kernel launches in a single @jax.jit should see
        each other's output — if the shim leaked to the default stream,
        the second launch could race with the first and produce wrong data."""
        vadd = self._build_kernel()

        x = jnp.arange(1024, dtype=jnp.float32)
        y = jnp.ones(1024, dtype=jnp.float32)

        @jax.jit
        def chain(a, b):
            t = vadd(a, b)       # t = a + b
            u = vadd(t, b)       # u = t + b = a + 2b
            return u

        out = chain(x, y)
        out.block_until_ready()
        expected = x + 2 * y
        assert jnp.allclose(out, expected), \
            f"stream ordering broken? max diff: {float(jnp.max(jnp.abs(out - expected)))}"

    def test_cubin_handle_is_registered_with_shim(self):
        """After lowering, the handle should be in the shim's process-local
        launch registry with the expected grid/block/smem."""
        vadd = self._build_kernel()
        x = jnp.ones(512, dtype=jnp.float32)
        y = jnp.ones(512, dtype=jnp.float32)

        @jax.jit
        def fn(a, b):
            return vadd(a, b)

        _ = fn.lower(x, y)

        from pyptx.jax_support import _load_shim, get_cubin_registry
        shim = _load_shim()
        assert shim is not None, "shim should be loaded on a GPU box"

        records = [
            get_cubin_registry().get(h)
            for h in vadd._cubin_handles.values()
        ]
        # There should be at least one record and every registered handle
        # must be visible to the shim.
        assert records, "no cubin handles were registered"
        for h in vadd._cubin_handles.values():
            assert shim.pyptx_shim_has_handle(h) == 1, \
                f"handle {h} not in shim registry"
        # 512 / 256 = 2 → grid should be (2, 1, 1)
        assert any(r.grid == (2, 1, 1) for r in records)


@_GPU_REQUIRED
class TestTmaCopyExecution:
    """A full TMA copy kernel written entirely in the pyptx DSL, runs
    end-to-end on H100, and produces bitwise-correct output.

    Flow:
        mbarrier.init → fence.proxy.async → mbarrier.arrive.expect_tx →
        cp.async.bulk.tensor.2d (TMA load src → shared) →
        mbarrier.try_wait.parity (spin loop) →
        cp.async.bulk.tensor.2d.store (TMA store shared → dst) →
        cp.async.bulk.commit_group → cp.async.bulk.wait_group 0
    """

    TILE_ROWS = 128
    TILE_COLS = 64

    def _build_kernel(self):
        from pyptx.types import b32
        tile_rows = self.TILE_ROWS
        tile_cols = self.TILE_COLS
        tx_bytes = tile_rows * tile_cols * 2  # bf16 = 2 bytes/elem

        @kernel(
            in_specs=(Tile(tile_rows, tile_cols, bf16, Layout.TMA_128B),),
            out_specs=(Tile(tile_rows, tile_cols, bf16, Layout.TMA_128B),),
            grid=(1, 1, 1),
            block=(1, 1, 1),
            arch="sm_90a",
        )
        def tma_copy(src, dst):
            sA = smem.alloc(bf16, (tile_rows, tile_cols), swizzle="128B")
            bar = smem.mbarrier(1)
            phase = reg.scalar(b32, init=0)

            ptx.mbarrier.init(bar[0], 1)
            ptx.fence.proxy_async()
            ptx.mbarrier.arrive_expect_tx(bar[0], tx_bytes)
            ptx.cp.async_.bulk.tensor_2d(
                dst=sA[0],
                src=src.tma_desc(),
                coord=(0, 0),
                mbar=bar[0],
            )
            ptx.mbarrier.wait(bar[0], phase)
            ptx.cp.async_.bulk.tensor_2d.store(
                dst=dst.tma_desc(),
                src=sA[0],
                coord=(0, 0),
            )
            ptx.cp.async_.bulk.commit_group()
            ptx.cp.async_.bulk.wait_group(0)
            ptx.ret()

        return tma_copy

    def test_emitted_ptx_has_all_the_right_pieces(self):
        tma_copy = self._build_kernel()
        text = tma_copy.ptx()
        # Entry signature: 2 regular + 2 TMA desc params.
        assert ".param .u64 src" in text
        assert ".param .u64 dst" in text
        assert ".param .u64 src_tma_desc" in text
        assert ".param .u64 dst_tma_desc" in text
        # mbarrier dance
        assert "mbarrier.init.shared.b64 [mbar_0], 1" in text
        assert "fence.proxy.async" in text
        assert "mbarrier.arrive.expect_tx.shared.b64" in text
        assert "mbarrier.try_wait.parity.shared.b64" in text
        # TMA load + store
        assert "cp.async.bulk.tensor.2d.shared::cluster.global" in text
        assert "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group" in text
        # Commit/wait fence for the TMA store
        assert "cp.async.bulk.commit_group" in text
        assert "cp.async.bulk.wait_group 0" in text

    def test_tma_copy_is_bitwise_identity(self):
        """End-to-end: pipe an arange tile through the TMA copy kernel
        under @jax.jit on H100 and verify bitwise equality with input."""
        import numpy as np
        tma_copy = self._build_kernel()

        src_np = np.arange(
            self.TILE_ROWS * self.TILE_COLS, dtype=np.float32
        ).reshape(self.TILE_ROWS, self.TILE_COLS)
        x = jnp.asarray(src_np, dtype=jnp.bfloat16)

        @jax.jit
        def fn(src):
            return tma_copy(src)

        out = fn(x)
        out.block_until_ready()
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert jnp.array_equal(out, x), \
            f"TMA copy is not identity: first mismatch at {jnp.where(out != x)}"

    def test_tma_copy_different_input_values(self):
        """Re-run the same kernel with a different input pattern to
        verify the shim really patches the descriptor each launch."""
        import numpy as np
        tma_copy = self._build_kernel()

        a = jnp.asarray(
            np.arange(self.TILE_ROWS * self.TILE_COLS, dtype=np.float32)
            .reshape(self.TILE_ROWS, self.TILE_COLS),
            dtype=jnp.bfloat16,
        )
        b = jnp.asarray(
            (np.arange(self.TILE_ROWS * self.TILE_COLS, dtype=np.float32)
             .reshape(self.TILE_ROWS, self.TILE_COLS)) * -1.0,
            dtype=jnp.bfloat16,
        )

        @jax.jit
        def fn(src):
            return tma_copy(src)

        out_a = fn(a)
        out_a.block_until_ready()
        out_b = fn(b)
        out_b.block_until_ready()
        assert jnp.array_equal(out_a, a)
        assert jnp.array_equal(out_b, b)
        # And their outputs should differ (not reusing stale descriptor)
        assert not jnp.array_equal(out_a, out_b)


@_GPU_REQUIRED
class TestWgmmaBasic:
    """wgmma.mma_async infrastructure: kernel compiles on driver JIT and
    produces correct output for uniform inputs.

    Non-uniform-A correctness is gated on descriptor LBO/SBO tuning —
    see task #50. The infrastructure itself (make_descriptor, the
    8-operand mma_async form, pred scale_d, trans_a/trans_b, and the
    fixed if_/setp plumbing) is solid and landed here.
    """

    def _build_kernel(self):
        from pyptx.types import s32

        @kernel(
            in_specs=(
                # A uses Layout.TMA_32B so TMA pre-swizzles the tile on
                # store-to-shared. wgmma's make_descriptor(swizzle=3)
                # reads the same tile back and the two swizzles cancel,
                # producing the logical data view. Without matching
                # swizzles, A's k and B's k don't line up and you get
                # garbage for any kernel whose output depends on the
                # pairing of A[i,k] and B[k,j].
                Tile(64, 16, bf16, Layout.TMA_32B),
                Tile(16, 8, bf16),
            ),
            out_specs=(Tile(64, 8, f32),),
            grid=(1, 1, 1),
            block=(128, 1, 1),
            arch="sm_90a",
        )
        def wgmma_tile(A, B, C):
            sA = smem.alloc(bf16, (64, 16), swizzle="32B")
            sB = smem.alloc(bf16, (16, 8))
            bar = smem.mbarrier(1)
            phase = reg.scalar(b32, init=0)

            # Thread 0 drives TMA setup
            tid = ptx.special.tid.x()
            is_zero = tid == 0
            with ptx.if_(is_zero):
                ptx.mbarrier.init(bar[0], 1)
                ptx.fence.proxy_async()
                ptx.mbarrier.arrive_expect_tx(bar[0], 2304)
                ptx.cp.async_.bulk.tensor_2d(
                    dst=sA[0], src=A.tma_desc(),
                    coord=(0, 0), mbar=bar[0],
                )
                ptx.cp.async_.bulk.tensor_2d(
                    dst=sB[0], src=B.tma_desc(),
                    coord=(0, 0), mbar=bar[0],
                )
            ptx.bar.sync(0)
            ptx.mbarrier.wait(bar[0], phase)

            # K=16 bf16 row-major A in shared = B32 canonical layout
            # (per CUTLASS cute/atom/mma_traits_sm90_gmma.hpp).
            # A row width = 16 bf16 = 32 bytes = 2 u128, which matches
            # Swizzle<1,4,3> with stride<0,0> = 2T. INTERLEAVE (swizzle=0)
            # would require K=8 (1 u128 wide), not K=16.
            desc_a = ptx.wgmma.make_descriptor(
                sA, leading_byte_offset=16,
                stride_byte_offset=256,
                swizzle=ptx.wgmma.SWIZZLE_32B,
            )
            # B is 16x8 bf16 row-major = N-major (trans_b=1). For a
            # 1-u128-wide N dim (8 bf16 = 16 bytes) with INTERLEAVE
            # swizzle, CUTLASS assigns leading_byte_offset to the
            # stride between 8-K-row slabs = 8 rows * 16 bytes/row
            # = 128 bytes. SBO is degenerate for the N=1-u128 case.
            desc_b = ptx.wgmma.make_descriptor(
                sB, leading_byte_offset=128,
                stride_byte_offset=16,
                swizzle=ptx.wgmma.SWIZZLE_NONE,
            )
            acc = reg.array(f32, 4)
            ptx.wgmma.fence()
            ptx.wgmma.mma_async(
                shape=(64, 8, 16),
                dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                d=acc, a=desc_a, b=desc_b,
                scale_d=False, trans_a=0, trans_b=1,
            )
            ptx.wgmma.commit_group()
            ptx.wgmma.wait_group(0)

            # Epilogue: m64n8 thread layout writeback.
            tid2 = reg.scalar(u32)
            ptx.inst.mov.u32(tid2, ptx.special.tid.x())
            group = reg.scalar(u32)
            lane = reg.scalar(u32)
            ptx.inst.shr.u32(group, tid2, 5)
            ptx.inst.and_.b32(lane, tid2, 31)
            row = reg.scalar(u32)
            col = reg.scalar(u32)
            tmp = reg.scalar(u32)
            ptx.inst.shl.b32(row, group, 4)
            ptx.inst.shr.u32(tmp, lane, 2)
            ptx.inst.add.u32(row, row, tmp)
            ptx.inst.and_.b32(col, lane, 3)
            ptx.inst.shl.b32(col, col, 1)
            pc = reg.scalar(b64)
            ptx.inst.ld.param.u64(pc, ptx.addr(C))
            ptx.inst.cvta.to.global_.u64(pc, pc)
            row_col = reg.scalar(u32)
            ptx.inst.shl.b32(tmp, row, 3)
            ptx.inst.add.u32(row_col, tmp, col)
            off = reg.scalar(u64)
            ptx.inst.mul.wide.u32(off, row_col, 4)
            p0 = reg.scalar(b64)
            ptx.inst.add.s64(p0, pc, off)
            ptx.inst.st.global_.f32(ptx.addr(p0), acc[0])
            ptx.inst.st.global_.f32(ptx.addr(p0, 4), acc[1])
            row8 = reg.scalar(u32)
            ptx.inst.add.u32(row8, row, 8)
            ptx.inst.shl.b32(tmp, row8, 3)
            ptx.inst.add.u32(row_col, tmp, col)
            ptx.inst.mul.wide.u32(off, row_col, 4)
            p1 = reg.scalar(b64)
            ptx.inst.add.s64(p1, pc, off)
            ptx.inst.st.global_.f32(ptx.addr(p1), acc[2])
            ptx.inst.st.global_.f32(ptx.addr(p1, 4), acc[3])
            ptx.ret()

        return wgmma_tile

    def test_wgmma_kernel_compiles_and_runs(self):
        """Just proves the kernel reaches cuLaunchKernel without faulting."""
        wgmma = self._build_kernel()
        A = jnp.ones((64, 16), dtype=jnp.bfloat16)
        B = jnp.ones((16, 8), dtype=jnp.bfloat16)

        @jax.jit
        def fn(A, B):
            return wgmma(A, B)

        out = fn(A, B)
        out.block_until_ready()
        assert out.shape == (64, 8)

    def test_wgmma_ones_times_ones_equals_k(self):
        """A=ones(64,16) @ B=ones(16,8) → every entry is k=16."""
        import numpy as np
        wgmma = self._build_kernel()
        A = jnp.ones((64, 16), dtype=jnp.bfloat16)
        B = jnp.ones((16, 8), dtype=jnp.bfloat16)

        @jax.jit
        def fn(A, B):
            return wgmma(A, B)

        out = np.asarray(fn(A, B))
        assert (out == 16.0).all(), f"expected all 16.0, got unique: {np.unique(out)}"

    def test_wgmma_per_column_b(self):
        """A=ones(64,16) @ B[:,j]=j → C[i,j] = 16*j.

        Exercises trans_b=1 row-major B reading — if the B descriptor
        is wrong we get all-same values across columns.
        """
        import numpy as np
        wgmma = self._build_kernel()
        A = jnp.ones((64, 16), dtype=jnp.bfloat16)
        b_np = np.tile(np.arange(8, dtype=np.float32), (16, 1))
        B = jnp.asarray(b_np, dtype=jnp.bfloat16)

        @jax.jit
        def fn(A, B):
            return wgmma(A, B)

        out = np.asarray(fn(A, B))
        expected_row = np.arange(8, dtype=np.float32) * 16.0
        # Every row should equal [0, 16, 32, 48, 64, 80, 96, 112].
        for i in range(64):
            assert np.allclose(out[i], expected_row), \
                f"row {i}: {out[i]} != {expected_row}"

    def test_wgmma_per_row_a(self):
        """A[i,k] = i (per-row distinct), B = ones → C[i,j] = 16*i.

        The discriminator for A descriptor correctness — uniform-A tests
        mask row-stride bugs but this exposes them. Requires swizzle=3
        (SWIZZLE_32B / B32) in the A descriptor, because K=16 bf16
        row-major has a row width of 32 bytes = 2 uint128_t = 2T, and
        CUTLASS's canonical GMMA K-major B32 layout has exactly that
        stride<0,0>. INTERLEAVE (swizzle=0) would require K=8.
        """
        import numpy as np
        wgmma = self._build_kernel()
        a_np = np.tile(
            np.arange(64, dtype=np.float32).reshape(64, 1), (1, 16)
        )
        A = jnp.asarray(a_np, dtype=jnp.bfloat16)
        B = jnp.ones((16, 8), dtype=jnp.bfloat16)

        @jax.jit
        def fn(A, B):
            return wgmma(A, B)

        out = np.asarray(fn(A, B))
        expected = a_np @ np.ones((16, 8), dtype=np.float32)
        assert np.array_equal(out, expected), \
            f"max diff {np.abs(out - expected).max()}"

    def test_wgmma_per_col_a(self):
        """A[i,k] = k (per-col distinct), B = ones → C[i,j] = 0+1+...+15 = 120."""
        import numpy as np
        wgmma = self._build_kernel()
        a_np = np.tile(np.arange(16, dtype=np.float32), (64, 1))
        A = jnp.asarray(a_np, dtype=jnp.bfloat16)
        B = jnp.ones((16, 8), dtype=jnp.bfloat16)

        @jax.jit
        def fn(A, B):
            return wgmma(A, B)

        out = np.asarray(fn(A, B))
        assert (out == 120.0).all(), f"unique: {np.unique(out)}"

    def test_wgmma_k_squared(self):
        """A[i,k]=k, B[k,j]=k → C[i,j] = sum_k k*k = 0+1+4+9+...+225 = 1240.

        This is the 'both operands vary in k' test case. Without the
        matching TMA/wgmma swizzle pair (Layout.TMA_32B on A + swizzle=3
        on the descriptor), wgmma sees permuted k indices for A but
        unpermuted k for B, so the pairs don't align and the dot
        products are garbage. With matching swizzles they cancel.
        """
        import numpy as np
        wgmma = self._build_kernel()
        a = np.tile(np.arange(16, dtype=np.float32), (64, 1))
        b = np.tile(np.arange(16, dtype=np.float32).reshape(16, 1), (1, 8))
        A = jnp.asarray(a, dtype=jnp.bfloat16)
        B = jnp.asarray(b, dtype=jnp.bfloat16)

        @jax.jit
        def fn(A, B):
            return wgmma(A, B)

        out = np.asarray(fn(A, B))
        # Expected 1240 everywhere. bf16 of 1240 rounds to 1240 exactly.
        expected = 1240.0
        assert np.allclose(out, expected), \
            f"max diff {np.abs(out - expected).max()}, unique={np.unique(out)[:5]}"

    def test_wgmma_auto_descriptor_matches_manual(self):
        """``ptx.wgmma.auto_descriptor`` (which uses the new layout
        picker in pyptx.wgmma_layout) produces a kernel that's
        bit-exactly equivalent to the hand-tuned one this class
        already builds — for m64n8k16 bf16 with K=16 A, the auto-picker
        should pick B32 for A and INTERLEAVE for B, yielding
        (LBO=16, SBO=256, swizzle=3) for A and (LBO=128, SBO=16,
        swizzle=0) for B. We verify by building a kernel that uses
        auto_descriptor instead of hand-coded make_descriptor and
        running the same correctness tests."""
        import numpy as np

        @kernel(
            in_specs=(
                Tile(64, 16, bf16, Layout.TMA_32B),
                Tile(16, 8, bf16),
            ),
            out_specs=(Tile(64, 8, f32),),
            grid=(1, 1, 1),
            block=(128, 1, 1),
            arch="sm_90a",
        )
        def wgmma_auto(A, B, C):
            sA = smem.alloc(bf16, (64, 16), swizzle="32B")
            sB = smem.alloc(bf16, (16, 8))
            bar = smem.mbarrier(1)
            phase = reg.scalar(b32, init=0)
            tid = ptx.special.tid.x()
            with ptx.if_(tid == 0):
                ptx.mbarrier.init(bar[0], 1)
                ptx.fence.proxy_async()
                ptx.mbarrier.arrive_expect_tx(bar[0], 2304)
                ptx.cp.async_.bulk.tensor_2d(
                    dst=sA[0], src=A.tma_desc(),
                    coord=(0, 0), mbar=bar[0],
                )
                ptx.cp.async_.bulk.tensor_2d(
                    dst=sB[0], src=B.tma_desc(),
                    coord=(0, 0), mbar=bar[0],
                )
            ptx.bar.sync(0)
            ptx.mbarrier.wait(bar[0], phase)

            # NEW API: no magic numbers, just shape + dtype + major.
            desc_a = ptx.wgmma.auto_descriptor(
                sA, dtype=bf16, shape=(64, 16), major="K",
            )
            desc_b = ptx.wgmma.auto_descriptor(
                sB, dtype=bf16, shape=(16, 8), major="MN",
            )

            acc = reg.array(f32, 4)
            ptx.wgmma.fence()
            ptx.wgmma.mma_async(
                shape=(64, 8, 16),
                dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                d=acc, a=desc_a, b=desc_b,
                scale_d=False, trans_a=0, trans_b=1,
            )
            ptx.wgmma.commit_group()
            ptx.wgmma.wait_group(0)

            # Same epilogue as the hand-tuned kernel.
            tid2 = reg.scalar(u32)
            ptx.inst.mov.u32(tid2, ptx.special.tid.x())
            group = reg.scalar(u32)
            lane = reg.scalar(u32)
            ptx.inst.shr.u32(group, tid2, 5)
            ptx.inst.and_.b32(lane, tid2, 31)
            row = reg.scalar(u32)
            col = reg.scalar(u32)
            tmp = reg.scalar(u32)
            ptx.inst.shl.b32(row, group, 4)
            ptx.inst.shr.u32(tmp, lane, 2)
            ptx.inst.add.u32(row, row, tmp)
            ptx.inst.and_.b32(col, lane, 3)
            ptx.inst.shl.b32(col, col, 1)
            pc = reg.scalar(b64)
            ptx.inst.ld.param.u64(pc, ptx.addr(C))
            ptx.inst.cvta.to.global_.u64(pc, pc)
            row_col = reg.scalar(u32)
            ptx.inst.shl.b32(tmp, row, 3)
            ptx.inst.add.u32(row_col, tmp, col)
            off = reg.scalar(u64)
            ptx.inst.mul.wide.u32(off, row_col, 4)
            p0 = reg.scalar(b64)
            ptx.inst.add.s64(p0, pc, off)
            ptx.inst.st.global_.f32(ptx.addr(p0), acc[0])
            ptx.inst.st.global_.f32(ptx.addr(p0, 4), acc[1])
            row8 = reg.scalar(u32)
            ptx.inst.add.u32(row8, row, 8)
            ptx.inst.shl.b32(tmp, row8, 3)
            ptx.inst.add.u32(row_col, tmp, col)
            ptx.inst.mul.wide.u32(off, row_col, 4)
            p1 = reg.scalar(b64)
            ptx.inst.add.s64(p1, pc, off)
            ptx.inst.st.global_.f32(ptx.addr(p1), acc[2])
            ptx.inst.st.global_.f32(ptx.addr(p1, 4), acc[3])
            ptx.ret()

        # Run against the same random inputs as the manual gold test
        # and verify bit-exact equality with JAX's bf16 matmul.
        np.random.seed(0)
        a = (np.random.randn(64, 16) * 0.1).astype(np.float32)
        b = (np.random.randn(16, 8) * 0.1).astype(np.float32)
        A = jnp.asarray(a, dtype=jnp.bfloat16)
        B = jnp.asarray(b, dtype=jnp.bfloat16)

        @jax.jit
        def fn(A, B):
            return wgmma_auto(A, B)

        out = np.asarray(fn(A, B))
        ref = np.asarray(
            jax.lax.dot_general(
                A, B, (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
        )
        assert np.array_equal(out, ref), \
            f"auto_descriptor path is not bit-equal: max diff {np.abs(out - ref).max()}"

    def test_wgmma_random_matches_jax_bf16_matmul(self):
        """Random bf16 inputs → wgmma output matches jax.lax.dot_general
        (bf16→f32 accumulate) bit-exactly on an H100.

        The gold standard: given arbitrary bf16 A and B, our wgmma must
        produce the same f32 output as JAX's own bf16 matmul. No
        precision fudge factors.
        """
        import numpy as np
        wgmma = self._build_kernel()
        np.random.seed(0)
        a = (np.random.randn(64, 16) * 0.1).astype(np.float32)
        b = (np.random.randn(16, 8) * 0.1).astype(np.float32)
        A = jnp.asarray(a, dtype=jnp.bfloat16)
        B = jnp.asarray(b, dtype=jnp.bfloat16)

        @jax.jit
        def fn(A, B):
            return wgmma(A, B)

        out = np.asarray(fn(A, B))
        ref = np.asarray(
            jax.lax.dot_general(
                A, B, (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
        )
        assert np.array_equal(out, ref), \
            f"max abs diff {np.abs(out - ref).max()}"


@_GPU_REQUIRED
class TestWgmmaCleanAPI:
    """End-to-end test of the production wgmma API — zero magic numbers.

    Builds a kernel using:
      - Tile.wgmma_a / Tile.wgmma_b for the kernel specs  (auto TMA layout)
      - smem.wgmma_tile for shared memory (auto swizzle + metadata)
      - ptx.wgmma.mma_async with SharedAlloc args directly (auto descriptor)

    Compare output to jax.lax.dot_general bit-exactly. This is the shape
    a viral demo would take: 40 lines of Python, no LBO/SBO, no swizzle
    strings duplicated across three call sites.
    """

    def _build_kernel(self):
        @kernel(
            in_specs=(
                Tile.wgmma_a(64, 16, bf16),  # row-major MxK
                Tile.wgmma_b(16, 8, bf16),   # row-major KxN
            ),
            out_specs=(Tile(64, 8, f32),),
            grid=(1, 1, 1),
            block=(128, 1, 1),
            arch="sm_90a",
        )
        def gemm_clean(A, B, C):
            sA = smem.wgmma_tile(bf16, (64, 16), major="K")
            sB = smem.wgmma_tile(bf16, (16, 8), major="MN")
            bar = smem.mbarrier(1)
            phase = reg.scalar(b32, init=0)

            tid = ptx.special.tid.x()
            with ptx.if_(tid == 0):
                ptx.mbarrier.init(bar[0], 1)
                ptx.fence.proxy_async()
                ptx.mbarrier.arrive_expect_tx(bar[0], 2304)
                ptx.cp.async_.bulk.tensor_2d(
                    dst=sA[0], src=A.tma_desc(),
                    coord=(0, 0), mbar=bar[0],
                )
                ptx.cp.async_.bulk.tensor_2d(
                    dst=sB[0], src=B.tma_desc(),
                    coord=(0, 0), mbar=bar[0],
                )
            ptx.bar.sync(0)
            ptx.mbarrier.wait(bar[0], phase)

            acc = reg.array(f32, 4)
            ptx.wgmma.fence()
            # Pass SharedAllocs straight to mma_async; descriptor is
            # auto-built from their canonical GMMA layout metadata.
            ptx.wgmma.mma_async(
                shape=(64, 8, 16),
                dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                d=acc, a=sA, b=sB,
                scale_d=False, trans_a=0, trans_b=1,
            )
            ptx.wgmma.commit_group()
            ptx.wgmma.wait_group(0)

            # Epilogue (unchanged m64n8 thread layout).
            tid2 = reg.scalar(u32)
            ptx.inst.mov.u32(tid2, ptx.special.tid.x())
            group = reg.scalar(u32); lane = reg.scalar(u32)
            ptx.inst.shr.u32(group, tid2, 5)
            ptx.inst.and_.b32(lane, tid2, 31)
            row = reg.scalar(u32); col = reg.scalar(u32); tmp = reg.scalar(u32)
            ptx.inst.shl.b32(row, group, 4)
            ptx.inst.shr.u32(tmp, lane, 2)
            ptx.inst.add.u32(row, row, tmp)
            ptx.inst.and_.b32(col, lane, 3)
            ptx.inst.shl.b32(col, col, 1)
            pc = reg.scalar(b64)
            ptx.inst.ld.param.u64(pc, ptx.addr(C))
            ptx.inst.cvta.to.global_.u64(pc, pc)
            row_col = reg.scalar(u32)
            ptx.inst.shl.b32(tmp, row, 3)
            ptx.inst.add.u32(row_col, tmp, col)
            off = reg.scalar(u64)
            ptx.inst.mul.wide.u32(off, row_col, 4)
            p0 = reg.scalar(b64)
            ptx.inst.add.s64(p0, pc, off)
            ptx.inst.st.global_.f32(ptx.addr(p0), acc[0])
            ptx.inst.st.global_.f32(ptx.addr(p0, 4), acc[1])
            row8 = reg.scalar(u32)
            ptx.inst.add.u32(row8, row, 8)
            ptx.inst.shl.b32(tmp, row8, 3)
            ptx.inst.add.u32(row_col, tmp, col)
            ptx.inst.mul.wide.u32(off, row_col, 4)
            p1 = reg.scalar(b64)
            ptx.inst.add.s64(p1, pc, off)
            ptx.inst.st.global_.f32(ptx.addr(p1), acc[2])
            ptx.inst.st.global_.f32(ptx.addr(p1, 4), acc[3])
            ptx.ret()

        return gemm_clean

    def test_clean_api_bit_exact_random(self):
        """The clean-API kernel matches jax.lax.dot_general bit-exactly
        on random bf16 inputs — just like the hand-tuned version, but
        written without any LBO/SBO/swizzle magic numbers."""
        import numpy as np
        k = self._build_kernel()

        np.random.seed(0)
        a = (np.random.randn(64, 16) * 0.1).astype(np.float32)
        b = (np.random.randn(16, 8) * 0.1).astype(np.float32)
        A = jnp.asarray(a, dtype=jnp.bfloat16)
        B = jnp.asarray(b, dtype=jnp.bfloat16)

        @jax.jit
        def fn(A, B):
            return k(A, B)

        out = np.asarray(fn(A, B))
        ref = np.asarray(
            jax.lax.dot_general(
                A, B, (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
        )
        assert np.array_equal(out, ref), \
            f"clean API not bit-equal: max diff {np.abs(out - ref).max()}"

    def test_clean_api_k_squared(self):
        import numpy as np
        k = self._build_kernel()

        a = np.tile(np.arange(16, dtype=np.float32), (64, 1))
        b = np.tile(np.arange(16, dtype=np.float32).reshape(16, 1), (1, 8))
        A = jnp.asarray(a, dtype=jnp.bfloat16)
        B = jnp.asarray(b, dtype=jnp.bfloat16)

        @jax.jit
        def fn(A, B):
            return k(A, B)

        out = np.asarray(fn(A, B))
        assert (out == 1240.0).all(), f"unique: {np.unique(out)[:5]}"


@_GPU_REQUIRED
class TestWgmmaKLoop:
    """K-loop wgmma GEMM — prove that slicing K across multiple TMA loads
    + multiple wgmma.mma_async calls accumulates correctly.

    Demonstrates Python-level trace-time unrolling (``for k_slice in
    range(K // tile_k):`` inside the kernel body unrolls at trace time),
    and proves ``Tile.wgmma_a(..., tile_k=...)`` plus the matching
    ``Tile.wgmma_b(..., tile_k=..., tile_n=...)`` plumb the per-TMA box
    shape through to the descriptor correctly.

    The bug this guards against: if the TMA box defaulted to the full
    tensor, a sliced TMA load at coord offset would overrun its SMEM
    destination and stomp the adjacent mbarrier — causing the next
    mbarrier wait to hang forever. See the fix in Tile.wgmma_a/_b +
    kernel.py that passes tma_box through to synthesize_tma_descriptor.
    """

    def _build_two_iter(self):
        """Two-iter K-loop: K=32 split into two K=16 slices.

        Hand-unrolled iter 0 then iter 1, each with its own mbarrier.
        Validates the minimal pipelined pattern.
        """
        @kernel(
            in_specs=(
                Tile.wgmma_a(64, 32, bf16, tile_k=16),
                Tile.wgmma_b(32, 8, bf16, tile_k=16, tile_n=8),
            ),
            out_specs=(Tile(64, 8, f32),),
            grid=(1, 1, 1),
            block=(128, 1, 1),
            arch="sm_90a",
        )
        def gemm(A, B, C):
            sA = smem.wgmma_tile(bf16, (64, 16), major="K")
            sB = smem.wgmma_tile(bf16, (16, 8),  major="MN")
            bar0 = smem.mbarrier(1)
            bar1 = smem.mbarrier(1)
            phase0 = reg.scalar(b32, init=0)
            phase1 = reg.scalar(b32, init=0)
            acc = reg.array(f32, 4)

            tid = ptx.special.tid.x()
            with ptx.if_(tid == 0):
                ptx.mbarrier.init(bar0[0], 1)
                ptx.mbarrier.init(bar1[0], 1)
                ptx.fence.proxy_async_shared_cta()

            # Iter 0: load K=[0,16)
            with ptx.if_(tid == 0):
                ptx.mbarrier.arrive_expect_tx(bar0[0], 64*16*2 + 16*8*2)
                ptx.cp.async_.bulk.tensor_2d(dst=sA[0], src=A.tma_desc(), coord=(0, 0), mbar=bar0[0])
                ptx.cp.async_.bulk.tensor_2d(dst=sB[0], src=B.tma_desc(), coord=(0, 0), mbar=bar0[0])
            ptx.bar.sync(0)
            ptx.mbarrier.wait(bar0[0], phase0)

            ptx.wgmma.fence()
            ptx.wgmma.mma_async(shape=(64, 8, 16), dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                d=acc, a=sA, b=sB, scale_d=False, trans_a=0, trans_b=1)
            ptx.wgmma.commit_group()
            ptx.wgmma.wait_group(0)

            # Iter 1: load K=[16,32), accumulate
            with ptx.if_(tid == 0):
                ptx.mbarrier.arrive_expect_tx(bar1[0], 64*16*2 + 16*8*2)
                ptx.cp.async_.bulk.tensor_2d(dst=sA[0], src=A.tma_desc(), coord=(16, 0), mbar=bar1[0])
                ptx.cp.async_.bulk.tensor_2d(dst=sB[0], src=B.tma_desc(), coord=(0, 16), mbar=bar1[0])
            ptx.bar.sync(0)
            ptx.mbarrier.wait(bar1[0], phase1)

            ptx.wgmma.fence()
            ptx.wgmma.mma_async(shape=(64, 8, 16), dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                d=acc, a=sA, b=sB, scale_d=True, trans_a=0, trans_b=1)
            ptx.wgmma.commit_group()
            ptx.wgmma.wait_group(0)

            _wgmma_m64n8_epilogue(C, acc)
            ptx.ret()

        return gemm

    def _build_unrolled_k_loop(self, K: int, tile_k: int = 16):
        """K-loop using a plain Python ``for`` that unrolls at trace time.

        Each iteration gets a fresh mbarrier. Proves that Python-level
        unrolling is enough DSL sugar for K-loop kernels — no special
        helper needed.
        """
        n_iters = K // tile_k
        assert K % tile_k == 0, f"K={K} must be divisible by tile_k={tile_k}"

        @kernel(
            in_specs=(
                Tile.wgmma_a(64, K, bf16, tile_k=tile_k),
                Tile.wgmma_b(K, 8, bf16, tile_k=tile_k, tile_n=8),
            ),
            out_specs=(Tile(64, 8, f32),),
            grid=(1, 1, 1),
            block=(128, 1, 1),
            arch="sm_90a",
        )
        def gemm(A, B, C):
            sA = smem.wgmma_tile(bf16, (64, tile_k), major="K")
            sB = smem.wgmma_tile(bf16, (tile_k, 8), major="MN")
            # One mbarrier per iteration → each parity-wait uses phase=0.
            bars = [smem.mbarrier(1) for _ in range(n_iters)]
            phases = [reg.scalar(b32, init=0) for _ in range(n_iters)]
            acc = reg.array(f32, 4)

            tid = ptx.special.tid.x()
            with ptx.if_(tid == 0):
                for bar in bars:
                    ptx.mbarrier.init(bar[0], 1)
                ptx.fence.proxy_async_shared_cta()

            # Python for unrolls at trace time — one instance per K slice.
            for i in range(n_iters):
                with ptx.if_(tid == 0):
                    ptx.mbarrier.arrive_expect_tx(bars[i][0], 64*tile_k*2 + tile_k*8*2)
                    ptx.cp.async_.bulk.tensor_2d(
                        dst=sA[0], src=A.tma_desc(),
                        coord=(i * tile_k, 0), mbar=bars[i][0],
                    )
                    ptx.cp.async_.bulk.tensor_2d(
                        dst=sB[0], src=B.tma_desc(),
                        coord=(0, i * tile_k), mbar=bars[i][0],
                    )
                ptx.bar.sync(0)
                ptx.mbarrier.wait(bars[i][0], phases[i])

                ptx.wgmma.fence()
                ptx.wgmma.mma_async(
                    shape=(64, 8, 16),
                    dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                    d=acc, a=sA, b=sB,
                    scale_d=(i != 0),   # accumulate on every iter except the first
                    trans_a=0, trans_b=1,
                )
                ptx.wgmma.commit_group()
                ptx.wgmma.wait_group(0)

            _wgmma_m64n8_epilogue(C, acc)
            ptx.ret()

        return gemm

    def test_two_iter_bit_exact(self):
        """The minimal K=32 → 2 × K=16 case. Bit-exact vs jax.lax.dot_general."""
        import numpy as np
        k = self._build_two_iter()
        np.random.seed(1)
        a = (np.random.randn(64, 32) * 0.1).astype(np.float32)
        b = (np.random.randn(32, 8)  * 0.1).astype(np.float32)
        A = jnp.asarray(a, dtype=jnp.bfloat16)
        B = jnp.asarray(b, dtype=jnp.bfloat16)

        @jax.jit
        def fn(A, B):
            return k(A, B)

        out = np.asarray(fn(A, B))
        ref = np.asarray(jax.lax.dot_general(
            A, B, (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ))
        assert np.array_equal(out, ref), \
            f"K=32 two-iter not bit-equal: max diff {np.abs(out - ref).max()}"

    @pytest.mark.parametrize("K", [16, 32, 64, 128])
    def test_unrolled_k_loop_bit_exact(self, K):
        """Python-level for-loop unrolls at trace time. Proven at
        K=16 (1 iter), 32 (2), 64 (4), 128 (8)."""
        import numpy as np
        k = self._build_unrolled_k_loop(K, tile_k=16)
        np.random.seed(K)
        a = (np.random.randn(64, K) * 0.1).astype(np.float32)
        b = (np.random.randn(K, 8)  * 0.1).astype(np.float32)
        A = jnp.asarray(a, dtype=jnp.bfloat16)
        B = jnp.asarray(b, dtype=jnp.bfloat16)

        @jax.jit
        def fn(A, B):
            return k(A, B)

        out = np.asarray(fn(A, B))
        ref = np.asarray(jax.lax.dot_general(
            A, B, (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ))
        assert np.array_equal(out, ref), \
            f"K={K} unrolled K-loop not bit-equal: max diff {np.abs(out - ref).max()}"


def _wgmma_m64n8_epilogue(C, acc):
    """Shared wgmma m64n8 epilogue: scatter 4 fragment regs to 64x8 f32
    using the standard Hopper thread → (row, col) mapping.

    Extracted so the K-loop tests don't repeat the 30-line dance.
    """
    tid2 = reg.scalar(u32)
    ptx.inst.mov.u32(tid2, ptx.special.tid.x())
    group = reg.scalar(u32); lane = reg.scalar(u32)
    ptx.inst.shr.u32(group, tid2, 5)
    ptx.inst.and_.b32(lane, tid2, 31)
    row = reg.scalar(u32); col = reg.scalar(u32); tmp = reg.scalar(u32)
    ptx.inst.shl.b32(row, group, 4)
    ptx.inst.shr.u32(tmp, lane, 2)
    ptx.inst.add.u32(row, row, tmp)
    ptx.inst.and_.b32(col, lane, 3)
    ptx.inst.shl.b32(col, col, 1)
    pc = reg.scalar(b64)
    ptx.inst.ld.param.u64(pc, ptx.addr(C))
    ptx.inst.cvta.to.global_.u64(pc, pc)
    row_col = reg.scalar(u32)
    ptx.inst.shl.b32(tmp, row, 3)
    ptx.inst.add.u32(row_col, tmp, col)
    off = reg.scalar(u64)
    ptx.inst.mul.wide.u32(off, row_col, 4)
    p0 = reg.scalar(b64)
    ptx.inst.add.s64(p0, pc, off)
    ptx.inst.st.global_.f32(ptx.addr(p0), acc[0])
    ptx.inst.st.global_.f32(ptx.addr(p0, 4), acc[1])
    row8 = reg.scalar(u32)
    ptx.inst.add.u32(row8, row, 8)
    ptx.inst.shl.b32(tmp, row8, 3)
    ptx.inst.add.u32(row_col, tmp, col)
    ptx.inst.mul.wide.u32(off, row_col, 4)
    p1 = reg.scalar(b64)
    ptx.inst.add.s64(p1, pc, off)
    ptx.inst.st.global_.f32(ptx.addr(p1), acc[2])
    ptx.inst.st.global_.f32(ptx.addr(p1, 4), acc[3])


@_GPU_REQUIRED
class TestMultiCtaGemm:
    """Multi-CTA K-loop wgmma GEMM — the full end-to-end demo.

    Each CTA computes a single 64×8 output tile by iterating over K in
    ``tile_k=16`` slices (TMA load → wgmma → accumulate). The grid
    covers ``(N/8) × (M/64)`` CTAs so every output element is written
    exactly once.

    The kernel deliberately stays minimal:
      - no double-buffering (sA, sB reused across K iters)
      - no producer/consumer split (every warp does TMA + wgmma)
      - ``m64n8k16`` wgmma (smallest Hopper wgmma shape)

    It is not a production GEMM — peak throughput is ~25 TFLOPS on an
    H100, roughly 5% of dense bf16 tensor core peak. The point is that
    it's ~150 lines of Python, bit-exact against ``jax.lax.dot_general``
    at sizes from 64×8 all the way to 2048×2048, and runs end-to-end
    inside ``@jax.jit`` via the shim.
    """

    @staticmethod
    def _build(M: int, N: int, K: int, *, tile_k: int = 16):
        BM, BN = 64, 8
        assert M % BM == 0 and N % BN == 0 and K % tile_k == 0
        n_iters = K // tile_k
        grid = (N // BN, M // BM, 1)

        @kernel(
            in_specs=(
                Tile.wgmma_a(M, K, bf16, tile_m=BM, tile_k=tile_k),
                Tile.wgmma_b(K, N, bf16, tile_k=tile_k, tile_n=BN),
            ),
            out_specs=(Tile(M, N, f32),),
            grid=grid,
            block=(128, 1, 1),
            arch="sm_90a",
        )
        def gemm(A, B, C):
            sA = smem.wgmma_tile(bf16, (BM, tile_k), major="K")
            sB = smem.wgmma_tile(bf16, (tile_k, BN), major="MN")
            bars = [smem.mbarrier(1) for _ in range(n_iters)]
            phases = [reg.scalar(b32, init=0) for _ in range(n_iters)]
            acc = reg.array(f32, 4)

            # Per-CTA M/N tile offsets (ctaid.y → M tile, ctaid.x → N tile)
            row_offset = reg.scalar(u32)
            col_offset = reg.scalar(u32)
            ptx.inst.mov.u32(row_offset, ptx.special.ctaid.y())
            ptx.inst.shl.b32(row_offset, row_offset, 6)  # * 64
            ptx.inst.mov.u32(col_offset, ptx.special.ctaid.x())
            ptx.inst.shl.b32(col_offset, col_offset, 3)  # * 8

            tid = ptx.special.tid.x()
            with ptx.if_(tid == 0):
                for bar in bars:
                    ptx.mbarrier.init(bar[0], 1)
                ptx.fence.proxy_async_shared_cta()

            for i in range(n_iters):
                k_off = i * tile_k
                k_off_reg = reg.scalar(u32)
                ptx.inst.mov.u32(k_off_reg, k_off)

                with ptx.if_(tid == 0):
                    ptx.mbarrier.arrive_expect_tx(
                        bars[i][0], BM * tile_k * 2 + tile_k * BN * 2,
                    )
                    ptx.cp.async_.bulk.tensor_2d(
                        dst=sA[0], src=A.tma_desc(),
                        coord=(k_off_reg, row_offset), mbar=bars[i][0],
                    )
                    ptx.cp.async_.bulk.tensor_2d(
                        dst=sB[0], src=B.tma_desc(),
                        coord=(col_offset, k_off_reg), mbar=bars[i][0],
                    )
                ptx.bar.sync(0)
                ptx.mbarrier.wait(bars[i][0], phases[i])

                ptx.wgmma.fence()
                ptx.wgmma.mma_async(
                    shape=(64, 8, 16),
                    dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
                    d=acc, a=sA, b=sB,
                    scale_d=(i != 0),
                    trans_a=0, trans_b=1,
                )
                ptx.wgmma.commit_group()
                ptx.wgmma.wait_group(0)

            _multi_cta_m64n8_epilogue(C, acc, row_offset, col_offset, N)
            ptx.ret()

        return gemm

    def _run(self, M: int, N: int, K: int):
        import numpy as np
        k = self._build(M, N, K)
        np.random.seed(M * 100003 + N * 1009 + K)
        a = (np.random.randn(M, K) * 0.1).astype(np.float32)
        b = (np.random.randn(K, N) * 0.1).astype(np.float32)
        A = jnp.asarray(a, dtype=jnp.bfloat16)
        B = jnp.asarray(b, dtype=jnp.bfloat16)

        @jax.jit
        def fn(A, B):
            return k(A, B)

        out = np.asarray(fn(A, B))
        ref = np.asarray(jax.lax.dot_general(
            A, B, (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ))
        assert np.array_equal(out, ref), (
            f"multi-CTA GEMM {M}x{N}x{K} not bit-equal: "
            f"max diff {np.abs(out - ref).max()}"
        )

    # --- progressive size sweep --------------------------------------
    # Each test adds one dimension of scaling:
    #   - single CTA baseline
    #   - M-dim scaling (multi-row CTAs)
    #   - N-dim scaling (multi-col CTAs)
    #   - both
    #   - K scaling (more K-loop iters)
    #   - full 2kx2k
    def test_64x8_k32(self):                  self._run(64, 8, 32)
    def test_128x8_k32_m_scaling(self):       self._run(128, 8, 32)
    def test_64x16_k32_n_scaling(self):       self._run(64, 16, 32)
    def test_128x16_k32_both(self):           self._run(128, 16, 32)
    def test_256x32_k256_k_scaling(self):     self._run(256, 32, 256)
    def test_512x64_k512(self):               self._run(512, 64, 512)
    def test_1024x128_k1024(self):            self._run(1024, 128, 1024)
    def test_2048x2048_k2048_bit_exact(self):
        """The full 2kx2k × 2k GEMM, bit-exact against jax.lax.dot_general.

        Runs 8192 CTAs × 128 K iterations each. ~25 TFLOPS on H100.
        """
        self._run(2048, 2048, 2048)


def _multi_cta_m64n8_epilogue(C, acc, row_offset, col_offset, N):
    """Scatter m64n8 fragments to C[row_offset + frag_row, col_offset + frag_col]."""
    tid2 = reg.scalar(u32)
    ptx.inst.mov.u32(tid2, ptx.special.tid.x())
    group = reg.scalar(u32); lane = reg.scalar(u32)
    ptx.inst.shr.u32(group, tid2, 5)
    ptx.inst.and_.b32(lane, tid2, 31)
    frag_row = reg.scalar(u32); frag_col = reg.scalar(u32); tmp = reg.scalar(u32)
    ptx.inst.shl.b32(frag_row, group, 4)
    ptx.inst.shr.u32(tmp, lane, 2)
    ptx.inst.add.u32(frag_row, frag_row, tmp)
    ptx.inst.and_.b32(frag_col, lane, 3)
    ptx.inst.shl.b32(frag_col, frag_col, 1)
    ptx.inst.add.u32(frag_row, frag_row, row_offset)
    ptx.inst.add.u32(frag_col, frag_col, col_offset)

    pc = reg.scalar(b64); ptx.inst.ld.param.u64(pc, ptx.addr(C))
    ptx.inst.cvta.to.global_.u64(pc, pc)
    row_col = reg.scalar(u32)
    ptx.inst.mov.u32(row_col, N)
    ptx.inst.mul.lo.u32(row_col, frag_row, row_col)
    ptx.inst.add.u32(row_col, row_col, frag_col)
    off = reg.scalar(u64)
    ptx.inst.mul.wide.u32(off, row_col, 4)
    p0 = reg.scalar(b64)
    ptx.inst.add.s64(p0, pc, off)
    ptx.inst.st.global_.f32(ptx.addr(p0), acc[0])
    ptx.inst.st.global_.f32(ptx.addr(p0, 4), acc[1])

    frag_row8 = reg.scalar(u32)
    ptx.inst.add.u32(frag_row8, frag_row, 8)
    ptx.inst.mov.u32(row_col, N)
    ptx.inst.mul.lo.u32(row_col, frag_row8, row_col)
    ptx.inst.add.u32(row_col, row_col, frag_col)
    ptx.inst.mul.wide.u32(off, row_col, 4)
    p1 = reg.scalar(b64)
    ptx.inst.add.s64(p1, pc, off)
    ptx.inst.st.global_.f32(ptx.addr(p1), acc[2])
    ptx.inst.st.global_.f32(ptx.addr(p1, 4), acc[3])


@_GPU_REQUIRED
class TestTmaPipeline:
    """End-to-end TMA descriptor synthesis pipeline on a real H100.

    These tests don't execute a full correctness kernel (that needs the
    mbarrier init/wait dance which is still a DSL gap) — they exercise
    the pipeline from ``Tile(..., Layout.TMA_128B)`` + ``.tma_desc()``
    through driver JIT through shim-side TMA spec registration.
    """

    def _build_kernel(self):
        @kernel(
            in_specs=(Tile("M", "N", bf16, Layout.TMA_128B),),
            out_specs=(Tile("M", "N", bf16, Layout.ROW),),
            grid=(1, 1, 1),
            block=(128, 1, 1),
            arch="sm_90a",
        )
        def tma_copy(src, dst):
            sA = smem.alloc(bf16, (128, 64), swizzle="128B")
            bar = smem.mbarrier(1)
            ptx.cp.async_.bulk.tensor_2d(
                dst=sA[0],
                src=src.tma_desc(),
                coord=(0, 0),
                mbar=bar[0],
            )
            ptx.ret()
        return tma_copy

    def test_ptx_has_tma_desc_param_and_prologue(self):
        """The emitted PTX has a trailing ``.param .u64 src_tma_desc``
        and an ``ld.param.u64`` prologue that loads it into a register."""
        tma_copy = self._build_kernel()
        text = tma_copy.ptx(M=128, N=64)
        assert ".param .u64 src" in text
        assert ".param .u64 dst" in text
        assert ".param .u64 src_tma_desc" in text
        assert "ld.param.u64" in text
        # TMA instruction uses the loaded register, not the symbolic name.
        assert "cp.async.bulk.tensor.2d" in text
        assert "src_tma_desc" in text
        # Operand shape: [dst], [reg, {coords}], [mbar]
        assert "[%rd" in text  # the loaded descriptor reg feeds the TMA
        assert "{0, 0}" in text

    def test_ptx_compiles_via_driver_jit(self):
        """The TMA kernel PTX is legal Hopper assembly — driver JIT accepts it."""
        tma_copy = self._build_kernel()
        ptx_source = tma_copy.ptx(M=128, N=64)
        from pyptx.jax_support import compile_ptx_to_cubin
        result = compile_ptx_to_cubin(ptx_source, "sm_90a", kernel_name="tma_copy")
        assert result is not None
        cu_function, cu_module = result
        assert cu_function != 0

    def test_lower_synthesizes_tma_descriptor(self):
        """Lowering under @jax.jit runs the full plumbing: compile PTX,
        synthesize a CUtensorMap, register the spec with the shim, and
        keep the host/device TMA slots alive on the kernel object."""
        tma_copy = self._build_kernel()
        x = jnp.zeros((128, 64), dtype=jnp.bfloat16)

        @jax.jit
        def fn(src):
            return tma_copy(src)

        lowered = fn.lower(x)
        assert "pyptx_launch" in lowered.as_text()

        from pyptx.jax_support import _load_shim, get_cubin_registry
        shim = _load_shim()

        # A handle should be registered with grid/block populated.
        handles = list(tma_copy._cubin_handles.values())
        assert len(handles) == 1
        h = handles[0]
        rec = get_cubin_registry().get(h)
        assert rec is not None
        assert rec.cu_function is not None and rec.cu_function != 0
        assert rec.grid == (1, 1, 1)
        assert shim.pyptx_shim_has_handle(h) == 1

        # TMA slot metadata should be populated: one host CUtensorMap,
        # one 128-byte device buffer, both alive on the kernel object.
        slots = getattr(tma_copy, "_tma_slots_by_handle", {}).get(h, [])
        assert len(slots) == 1, f"expected 1 TMA slot, got {len(slots)}"
        host_tmap, host_blob_ptr, device_blob_ptr = slots[0]
        assert host_blob_ptr != 0
        assert device_blob_ptr != 0
        # host_tmap is the cuda-python CUtensorMap object keeping the 128
        # bytes alive — verify we can still read getPtr() through it.
        assert int(host_tmap.getPtr()) == host_blob_ptr

    def test_tma_desc_inside_trace_returns_loaded_register(self):
        """Inside a trace, ``TensorSpec.tma_desc()`` should emit an
        ``ld.param.u64`` prologue and return a Reg, not a handle."""
        from pyptx.kernel import TensorSpec, TmaDescriptorHandle
        from pyptx.reg import Reg

        # Outside a trace: backward-compat handle.
        t = TensorSpec("A", shape=(128, 64), dtype=bf16)
        outside = t.tma_desc()
        assert isinstance(outside, TmaDescriptorHandle)

        # Inside a trace: loaded Reg.
        tma_copy = self._build_kernel()
        # Trace it to populate the cache + side table.
        tma_copy.ptx(M=128, N=64)
        # The tma_tensor_names side table should mention "src".
        tma_names_map = getattr(tma_copy, "_tma_names_by_module_id", {})
        assert any("src" in names for names in tma_names_map.values()), \
            f"expected 'src' in tma names, got {tma_names_map}"
