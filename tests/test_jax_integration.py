"""JAX integration tests for pyptx.

These run on CPU (laptop-only) and verify:
- Tile/Layout primitives
- @kernel decorator with in_specs/out_specs/grid/block/cluster
- Kernel.__call__ inside @jax.jit produces the right HLO custom_call
- Cubin cache, cubin registry
- FFI target registration (without actually launching on GPU)

Tests that require a real GPU (cuLaunchKernelEx) are marked and skipped.
"""

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pyptx import kernel, reg, ptx, Tile, Layout
from pyptx.types import f32, bf16, b32, b64, u32, u64, pred
from pyptx.jax_support import (
    CubinRegistry,
    get_cubin_registry,
    ensure_ffi_registered,
    set_mock_ffi_callback,
)
from pyptx.cache import CubinCache, CacheKey, CacheEntry
from pyptx.specs import unify_envs


# ---------------------------------------------------------------------------
# Tile / Layout primitives
# ---------------------------------------------------------------------------

class TestTile:
    def test_symbolic_shape(self):
        t = Tile("M", "K", bf16, Layout.ROW)
        assert t.rank == 2
        assert t.symbolic_dims == ("M", "K")
        assert t.dtype == bf16
        assert t.layout == Layout.ROW

    def test_concrete_shape(self):
        t = Tile(128, 64, f32, Layout.COL)
        assert t.rank == 2
        assert t.symbolic_dims == ()

    def test_mixed_shape(self):
        t = Tile("M", 64, f32)
        assert t.symbolic_dims == ("M",)

    def test_dtype_required(self):
        with pytest.raises(ValueError, match="dtype"):
            Tile("M", "N")

    def test_extract_env(self):
        t = Tile("M", "K", bf16)
        env = t.extract_env((4096, 2048))
        assert env == {"M": 4096, "K": 2048}

    def test_extract_env_rank_mismatch(self):
        t = Tile("M", "K", bf16)
        with pytest.raises(ValueError, match="rank"):
            t.extract_env((4096,))

    def test_resolve_shape(self):
        t = Tile("M", 64, f32)
        assert t.resolve_shape({"M": 128}) == (128, 64)

    def test_unify_envs(self):
        a = Tile("M", "K", bf16).extract_env((4096, 2048))
        b = Tile("K", "N", bf16).extract_env((2048, 8192))
        merged = unify_envs([a, b])
        assert merged == {"M": 4096, "K": 2048, "N": 8192}

    def test_unify_envs_conflict(self):
        a = Tile("M", "K", bf16).extract_env((4096, 2048))
        b = Tile("K", "N", bf16).extract_env((9999, 8192))
        with pytest.raises(ValueError, match="conflicting"):
            unify_envs([a, b])


# ---------------------------------------------------------------------------
# @kernel with JAX specs
# ---------------------------------------------------------------------------

class TestKernelWithSpecs:
    def test_decorator_accepts_specs(self):
        @kernel(
            in_specs=(Tile("N", f32),),
            out_specs=(Tile("N", f32),),
            grid=lambda N: (N // 256, 1, 1),
            block=(256, 1, 1),
            arch="sm_90a",
        )
        def k(x, out):
            ptx.ret()

        assert k.in_specs == (Tile("N", f32),)
        assert k.out_specs == (Tile("N", f32),)
        assert k.block == (256, 1, 1)
        assert k.cluster == (1, 1, 1)
        assert callable(k.grid)

    def test_kernel_without_specs_still_works(self):
        @kernel(arch="sm_90a")
        def simple():
            ptx.ret()
        # .ptx() should still work
        text = simple.ptx()
        assert ".visible .entry simple()" in text
        # Calling with concrete arrays without in_specs raises
        with pytest.raises(NotImplementedError, match="in_specs"):
            simple()

    def test_resolve_grid_from_lambda(self):
        @kernel(
            in_specs=(Tile("M", "N", f32),),
            out_specs=(Tile("M", "N", f32),),
            grid=lambda M, N: (M // 128, N // 256, 1),
            block=(128, 1, 1),
            arch="sm_90a",
        )
        def k(x, out):
            ptx.ret()

        grid = k._resolve_grid({"M": 4096, "N": 8192})
        assert grid == (32, 32, 1)

    def test_resolve_grid_from_tuple(self):
        @kernel(
            in_specs=(Tile("N", f32),),
            out_specs=(Tile("N", f32),),
            grid=(16, 1, 1),
            arch="sm_90a",
        )
        def k(x, out):
            ptx.ret()

        assert k._resolve_grid({"N": 4096}) == (16, 1, 1)


# ---------------------------------------------------------------------------
# Kernel.__call__ → HLO custom_call lowering
# ---------------------------------------------------------------------------

class TestJaxLowering:
    def test_custom_call_in_hlo(self):
        @kernel(
            in_specs=(Tile("N", f32, Layout.ROW),),
            out_specs=(Tile("N", f32, Layout.ROW),),
            grid=lambda N: (N // 256, 1, 1),
            block=(256, 1, 1),
            arch="sm_90a",
        )
        def identity(x, out):
            r = reg.array(b32, 2)
            ptx.inst.mov.u32(r[0], ptx.special.tid.x())
            ptx.ret()

        x = jnp.ones((1024,), dtype=jnp.float32)

        @jax.jit
        def fn(x):
            return identity(x)

        lowered = fn.lower(x)
        text = lowered.as_text()

        # The HLO should contain our custom_call
        assert "pyptx_launch" in text
        # The only FFI attribute is the cubin handle; grid/block/smem
        # are registered with the C++ shim under that handle at compile
        # time, not passed per-call.
        assert "cubin_handle" in text

    def test_cubin_registered_after_lowering(self):
        registry = get_cubin_registry()
        initial_count = len(registry)

        @kernel(
            in_specs=(Tile("N", f32),),
            out_specs=(Tile("N", f32),),
            grid=lambda N: (N // 64, 1, 1),
            block=(64, 1, 1),
            arch="sm_90a",
        )
        def k(x, out):
            ptx.ret()

        x = jnp.ones((256,), dtype=jnp.float32)

        @jax.jit
        def fn(x):
            return k(x)

        _ = fn.lower(x)
        # A cubin should have been registered during lowering
        assert len(registry) > initial_count

    def test_specialization_cache(self):
        @kernel(
            in_specs=(Tile("N", f32),),
            out_specs=(Tile("N", f32),),
            grid=lambda N: (N // 64, 1, 1),
            block=(64, 1, 1),
            arch="sm_90a",
        )
        def k(x, out):
            ptx.ret()

        x1 = jnp.ones((256,), dtype=jnp.float32)
        x2 = jnp.ones((256,), dtype=jnp.float32)

        @jax.jit
        def fn(x):
            return k(x)

        _ = fn.lower(x1)
        handle_count_after_first = len(k._cubin_handles)

        _ = fn.lower(x2)  # same shape, should hit cache
        assert len(k._cubin_handles) == handle_count_after_first

    def test_different_shapes_produce_different_specializations(self):
        @kernel(
            in_specs=(Tile("N", f32),),
            out_specs=(Tile("N", f32),),
            grid=lambda N: (N // 64, 1, 1),
            block=(64, 1, 1),
            arch="sm_90a",
        )
        def k(x, out):
            ptx.ret()

        x1 = jnp.ones((256,), dtype=jnp.float32)
        x2 = jnp.ones((512,), dtype=jnp.float32)

        @jax.jit
        def fn1(x):
            return k(x)
        @jax.jit
        def fn2(x):
            return k(x)

        _ = fn1.lower(x1)
        _ = fn2.lower(x2)
        # Two distinct specializations
        assert len(k._cubin_handles) == 2


# ---------------------------------------------------------------------------
# Cubin cache
# ---------------------------------------------------------------------------

class TestCubinCache:
    def test_cache_key_hash_stable(self):
        k1 = CacheKey(
            fn_id="test_fn",
            template_kwargs=(("BM", 128), ("BN", 256)),
            input_shapes=(((4096, 4096), "float32"),),
            arch="sm_90a",
        )
        k2 = CacheKey(
            fn_id="test_fn",
            template_kwargs=(("BM", 128), ("BN", 256)),
            input_shapes=(((4096, 4096), "float32"),),
            arch="sm_90a",
        )
        assert k1.hash() == k2.hash()

    def test_cache_put_get(self, tmp_path):
        cache = CubinCache(cache_dir=tmp_path)
        key = CacheKey(
            fn_id="test",
            template_kwargs=(),
            input_shapes=(((16,), "float32"),),
            arch="sm_90a",
        )
        entry = CacheEntry(
            key=key,
            ptx_source=".version 8.5\n.target sm_90a\n",
            cubin_bytes=b"fake_cubin",
        )
        cache.put(entry)
        retrieved = cache.get(key)
        assert retrieved is not None
        assert retrieved.cubin_bytes == b"fake_cubin"
        assert retrieved.ptx_source.startswith(".version")

    def test_cache_disk_persistence(self, tmp_path):
        key = CacheKey(
            fn_id="test",
            template_kwargs=(),
            input_shapes=(((16,), "float32"),),
            arch="sm_90a",
        )
        entry = CacheEntry(
            key=key,
            ptx_source=".version 8.5\n",
            cubin_bytes=b"cubin_bytes_here",
        )

        # First cache: write
        cache1 = CubinCache(cache_dir=tmp_path)
        cache1.put(entry)

        # Second cache: read from disk
        cache2 = CubinCache(cache_dir=tmp_path)
        retrieved = cache2.get(key)
        assert retrieved is not None
        assert retrieved.cubin_bytes == b"cubin_bytes_here"

    def test_cache_miss(self, tmp_path):
        cache = CubinCache(cache_dir=tmp_path)
        key = CacheKey(
            fn_id="nonexistent",
            template_kwargs=(),
            input_shapes=(((1,), "float32"),),
            arch="sm_90a",
        )
        assert cache.get(key) is None


# ---------------------------------------------------------------------------
# Cubin registry
# ---------------------------------------------------------------------------

class TestCubinRegistry:
    def test_register_and_lookup(self):
        reg = CubinRegistry()
        h = reg.register(
            ptx_source=".version 8.5\n",
            kernel_name="my_kernel",
            smem_bytes=128,
        )
        assert h > 0
        record = reg.get(h)
        assert record is not None
        assert record.kernel_name == "my_kernel"
        assert record.smem_bytes == 128

    def test_unique_handles(self):
        reg = CubinRegistry()
        handles = [
            reg.register(ptx_source="", kernel_name=f"k{i}")
            for i in range(5)
        ]
        assert len(set(handles)) == 5

    def test_clear(self):
        reg = CubinRegistry()
        reg.register(ptx_source="", kernel_name="k")
        assert len(reg) == 1
        reg.clear()
        assert len(reg) == 0


# ---------------------------------------------------------------------------
# FFI registration
# ---------------------------------------------------------------------------

class TestFfiRegistration:
    def test_register_ffi_target_idempotent(self):
        result = ensure_ffi_registered()
        # On machines without the shim (no GPU), returns False — that's OK.
        assert isinstance(result, bool)
        assert ensure_ffi_registered() == result

    def test_mock_callback(self):
        calls = []

        def mock(*args, **kwargs):
            calls.append((args, kwargs))
            return None

        set_mock_ffi_callback(mock)
        try:
            from pyptx.jax_support import _pyptx_launch
            _pyptx_launch("a", "b", x=1)
            assert len(calls) == 1
            assert calls[0] == (("a", "b"), {"x": 1})
        finally:
            set_mock_ffi_callback(None)


# ---------------------------------------------------------------------------
# End-to-end (requires real GPU, skipped on CPU)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_gpu_execution_skipped_without_cuda(self):
        """A real kernel launch requires cuda-python + a GPU. This test
        documents the expected failure mode on CPU-only systems."""
        if jax.default_backend() == "gpu":
            pytest.skip("Has GPU, skipping CPU-only behavior check")

        @kernel(
            in_specs=(Tile("N", f32),),
            out_specs=(Tile("N", f32),),
            grid=lambda N: (N // 64, 1, 1),
            block=(64, 1, 1),
            arch="sm_90a",
        )
        def k(x, out):
            ptx.ret()

        x = jnp.ones((256,), dtype=jnp.float32)

        @jax.jit
        def fn(x):
            return k(x)

        # Lowering works on CPU — only execution fails
        lowered = fn.lower(x)
        assert "pyptx_launch" in lowered.as_text()
