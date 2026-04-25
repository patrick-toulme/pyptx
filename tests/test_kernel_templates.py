"""Tests for @kernel template kwargs.

Template kwargs are keyword-only parameters in the function signature that
get baked into the kernel at trace time. They are validated, defaulted from
the signature, and the resulting Module is cached on the (arch, version,
kwargs) tuple.
"""

import pytest

from pyptx import kernel, reg, ptx
from pyptx.kernel import Kernel, TensorSpec
from pyptx.types import b32, f32, u32


# ---------------------------------------------------------------------------
# Basic template kwargs
# ---------------------------------------------------------------------------

class TestBasicTemplateKwargs:
    def test_kwarg_specializes(self):
        """Passing a template kwarg should affect the emitted PTX."""
        @kernel(arch="sm_90a")
        def vec_add(*, N=1024):
            reg.array(f32, N // 32)
            ptx.ret()

        text = vec_add.ptx(N=1024)
        assert ".reg .f32 %f<32>;" in text

    def test_default_used_when_not_overridden(self):
        @kernel(arch="sm_90a")
        def vec_add(*, N=1024):
            reg.array(f32, N // 32)
            ptx.ret()

        text = vec_add.ptx()
        # default N=1024 → 32 registers
        assert ".reg .f32 %f<32>;" in text

    def test_override_changes_ptx(self):
        @kernel(arch="sm_90a")
        def vec_add(*, N=1024):
            reg.array(f32, N // 32)
            ptx.ret()

        text_small = vec_add.ptx(N=1024)
        text_big = vec_add.ptx(N=2048)
        assert ".reg .f32 %f<32>;" in text_small
        assert ".reg .f32 %f<64>;" in text_big
        assert text_small != text_big

    def test_partial_override(self):
        """Overriding one kwarg keeps the rest as defaults."""
        @kernel(arch="sm_90a")
        def fn(*, BM=128, BN=256):
            reg.array(f32, BM // 32)
            reg.array(b32, BN // 64)
            ptx.ret()

        text = fn.ptx(BM=64)
        # BM=64 -> 2 f32 regs, BN default 256 -> 4 b32 regs
        assert ".reg .f32 %f<2>;" in text
        assert ".reg .b32 %r<4>;" in text


# ---------------------------------------------------------------------------
# Multiple template params
# ---------------------------------------------------------------------------

class TestMultipleTemplateParams:
    def test_independent_params(self):
        @kernel(arch="sm_90a")
        def tiled(*, BM=128, BN=256, BK=64, STAGES=3):
            reg.array(f32, BM * BN // 1024)
            reg.array(b32, BK)
            reg.array(u32, STAGES)
            ptx.ret()

        text = tiled.ptx(BM=128, BN=256, BK=64, STAGES=3)
        assert ".reg .f32 %f<32>;" in text   # 128*256/1024
        assert ".reg .b32 %r<64>;" in text   # BK=64
        # STAGES=3 → 3 u32 regs (u32 also uses %r prefix, gets a unique base)
        # Just check there is a 3-count reg decl somewhere
        assert "<3>;" in text


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_unknown_kwarg_raises(self):
        @kernel(arch="sm_90a")
        def fn(*, BM=128):
            ptx.ret()

        with pytest.raises(TypeError, match="no template parameter"):
            fn.ptx(NOPE=64)

    def test_unknown_kwarg_lists_available(self):
        @kernel(arch="sm_90a")
        def fn(*, BM=128, BN=256, BK=64):
            ptx.ret()

        with pytest.raises(TypeError, match="BK") as ei:
            fn.ptx(BAD=1)
        msg = str(ei.value)
        assert "BAD" in msg
        assert "BM" in msg
        assert "BN" in msg
        assert "BK" in msg

    def test_no_template_params_unknown_kwarg(self):
        @kernel(arch="sm_90a")
        def fn():
            ptx.ret()

        with pytest.raises(TypeError, match="no template parameter"):
            fn.ptx(BM=128)

    def test_module_also_validates(self):
        @kernel(arch="sm_90a")
        def fn(*, BM=128):
            ptx.ret()

        with pytest.raises(TypeError, match="no template parameter"):
            fn.module(NOPE=1)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_cache_hit_returns_same_module(self):
        @kernel(arch="sm_90a")
        def fn(*, BM=128):
            reg.array(f32, BM // 32)
            ptx.ret()

        m1 = fn.module(BM=128)
        m2 = fn.module(BM=128)
        assert m1 is m2

    def test_default_and_explicit_match(self):
        """fn.module() and fn.module(BM=128) should hit the same cache entry."""
        @kernel(arch="sm_90a")
        def fn(*, BM=128):
            reg.array(f32, BM // 32)
            ptx.ret()

        m1 = fn.module()
        m2 = fn.module(BM=128)
        assert m1 is m2

    def test_cache_miss_different_kwargs(self):
        @kernel(arch="sm_90a")
        def fn(*, BM=128):
            reg.array(f32, BM // 32)
            ptx.ret()

        m1 = fn.module(BM=128)
        m2 = fn.module(BM=256)
        assert m1 is not m2

    def test_cache_separates_independent_kernels(self):
        """Two distinct kernels must not share a cache."""
        @kernel(arch="sm_90a")
        def fn_a(*, BM=128):
            reg.array(f32, BM // 32)
            ptx.ret()

        @kernel(arch="sm_90a")
        def fn_b(*, BM=128):
            reg.array(f32, BM // 32)
            ptx.ret()

        ma = fn_a.module()
        mb = fn_b.module()
        assert ma is not mb
        # Each kernel should have its own cache.
        assert fn_a._cache is not fn_b._cache

    def test_ptx_uses_cache(self):
        """Subsequent ptx() calls should hit the same Module."""
        @kernel(arch="sm_90a")
        def fn(*, N=64):
            reg.array(f32, N)
            ptx.ret()

        # First call populates the cache, second should re-use the Module.
        text1 = fn.ptx(N=64)
        text2 = fn.ptx(N=64)
        assert text1 == text2
        assert fn.module(N=64) is fn.module(N=64)


# ---------------------------------------------------------------------------
# Positional args / TensorSpec
# ---------------------------------------------------------------------------

class TestPositionalArgs:
    def test_positional_bound_to_tensor_spec(self):
        captured: dict = {}

        @kernel(arch="sm_90a")
        def fn(A, B, C, *, BM=128):
            captured["A"] = A
            captured["B"] = B
            captured["C"] = C
            ptx.ret()

        fn.ptx()
        assert isinstance(captured["A"], TensorSpec)
        assert isinstance(captured["B"], TensorSpec)
        assert isinstance(captured["C"], TensorSpec)
        assert captured["A"].name == "A"
        assert captured["B"].name == "B"
        assert captured["C"].name == "C"

    def test_only_positional_no_template(self):
        captured: dict = {}

        @kernel(arch="sm_90a")
        def fn(A):
            captured["A"] = A
            ptx.ret()

        fn.ptx()
        assert isinstance(captured["A"], TensorSpec)
        assert captured["A"].name == "A"

    def test_tensor_spec_repr(self):
        ts = TensorSpec("foo")
        assert "foo" in repr(ts)


# ---------------------------------------------------------------------------
# template_params property
# ---------------------------------------------------------------------------

class TestTemplateParamsProperty:
    def test_returns_declared_params(self):
        @kernel(arch="sm_90a")
        def fn(*, BM=128, BN=256, BK=64):
            ptx.ret()

        assert fn.template_params == {"BM": 128, "BN": 256, "BK": 64}

    def test_empty_for_kernel_without_template(self):
        @kernel(arch="sm_90a")
        def fn():
            ptx.ret()

        assert fn.template_params == {}

    def test_does_not_include_positional(self):
        @kernel(arch="sm_90a")
        def fn(A, B, *, BM=128):
            ptx.ret()

        assert fn.template_params == {"BM": 128}
        assert "A" not in fn.template_params
        assert "B" not in fn.template_params

    def test_property_is_a_copy(self):
        """Mutating the dict returned by template_params shouldn't affect future calls."""
        @kernel(arch="sm_90a")
        def fn(*, BM=128):
            ptx.ret()

        params = fn.template_params
        params["BM"] = 999
        params["NEW"] = 1
        # Subsequent reads still return the original.
        assert fn.template_params == {"BM": 128}


# ---------------------------------------------------------------------------
# Backwards compatibility: kernels with no template params
# ---------------------------------------------------------------------------

class TestBackwardsCompat:
    def test_kernel_no_template_params(self):
        @kernel(arch="sm_90a")
        def simple():
            r = reg.array(b32, 5)
            ptx.mov(b32, r[0], 42)
            ptx.ret()

        text = simple.ptx()
        assert ".version 8.5" in text
        assert ".target sm_90a" in text
        assert ".visible .entry simple()" in text
        assert "mov.b32" in text
        assert "ret;" in text

    def test_kernel_no_template_params_caches(self):
        @kernel(arch="sm_90a")
        def simple():
            ptx.ret()

        m1 = simple.module()
        m2 = simple.module()
        assert m1 is m2

    def test_kernel_with_only_positional(self):
        @kernel(arch="sm_90a")
        def fn(A, B):
            ptx.ret()

        text = fn.ptx()
        assert "ret;" in text


# ---------------------------------------------------------------------------
# Template kwargs actually affect the emitted PTX
# ---------------------------------------------------------------------------

class TestTemplateAffectsPtx:
    def test_template_specializes_register_count(self):
        @kernel(arch="sm_90a")
        def vec_add(*, N=1024):
            reg.array(f32, N // 32)
            ptx.ret()

        ptx1 = vec_add.ptx(N=1024)
        ptx2 = vec_add.ptx(N=2048)
        assert ".reg .f32 %f<32>;" in ptx1
        assert ".reg .f32 %f<64>;" in ptx2
        assert ptx1 != ptx2

    def test_template_specializes_loop_unroll(self):
        """Plain Python `for` over a template param unrolls at trace time."""
        @kernel(arch="sm_90a")
        def fn(*, UNROLL=2):
            r = reg.array(b32, 10)
            for i in range(UNROLL):
                ptx.mov(b32, r[i], i)
            ptx.ret()

        text2 = fn.ptx(UNROLL=2)
        text4 = fn.ptx(UNROLL=4)
        assert text2.count("mov.b32") == 2
        assert text4.count("mov.b32") == 4

    def test_template_kwargs_with_arch_in_key(self):
        """Two kernels at different archs should not collide in the cache."""
        @kernel(arch="sm_90a")
        def fn_a(*, N=64):
            reg.array(f32, N)
            ptx.ret()

        @kernel(arch="sm_100a")
        def fn_b(*, N=64):
            reg.array(f32, N)
            ptx.ret()

        ta = fn_a.ptx()
        tb = fn_b.ptx()
        assert "sm_90a" in ta
        assert "sm_100a" in tb
        assert ".version 8.5" in ta
        assert ".version 8.8" in tb
