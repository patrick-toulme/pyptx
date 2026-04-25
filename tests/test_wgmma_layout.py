"""Unit tests for the pyptx GMMA layout picker (pure-Python port of
cute::make_gmma_desc's layout branch)."""

import pytest

from pyptx.types import bf16, f16, f32, tf32
from pyptx.wgmma_layout import (
    GmmaLayout,
    LayoutType,
    Major,
    layout_for_a,
    layout_for_b,
    pick_gmma_layout,
)


class TestPickKMajor:
    """K-major = row-major A of shape (M, K). Row width = K * elem_bytes."""

    def test_k8_bf16_is_interleave(self):
        """K=8 bf16 → row width 16 bytes = 1 u128 → INTERLEAVE."""
        la = pick_gmma_layout(elem_bytes=2, m_or_n=64, k=8, major=Major.K)
        assert la.layout_type == LayoutType.INTERLEAVE
        assert la.swizzle_code == 0
        assert la.tma_swizzle == "NONE"
        assert la.smem_swizzle is None
        assert la.leading_byte_offset == 16
        assert la.stride_byte_offset == 8 * 16  # 8 rows * 16 B

    def test_k16_bf16_is_b32(self):
        """K=16 bf16 → row width 32 bytes = 2 u128 → B32.

        This is the case that drove the original bug — INTERLEAVE
        doesn't fit this shape and B32 is the natural choice.
        """
        la = pick_gmma_layout(elem_bytes=2, m_or_n=64, k=16, major=Major.K)
        assert la.layout_type == LayoutType.B32
        assert la.swizzle_code == 3  # B32 enum value
        assert la.tma_swizzle == "32B"
        assert la.smem_swizzle == "32B"
        assert la.leading_byte_offset == 16
        assert la.stride_byte_offset == 8 * 32  # 256 bytes

    def test_k32_bf16_is_b64(self):
        """K=32 bf16 → row width 64 bytes = 4 u128 → B64."""
        la = pick_gmma_layout(elem_bytes=2, m_or_n=64, k=32, major=Major.K)
        assert la.layout_type == LayoutType.B64
        assert la.swizzle_code == 2
        assert la.tma_swizzle == "64B"
        assert la.stride_byte_offset == 8 * 64  # 512 bytes

    def test_k64_bf16_is_b128(self):
        """K=64 bf16 → row width 128 bytes = 8 u128 → B128."""
        la = pick_gmma_layout(elem_bytes=2, m_or_n=64, k=64, major=Major.K)
        assert la.layout_type == LayoutType.B128
        assert la.swizzle_code == 1
        assert la.tma_swizzle == "128B"
        assert la.stride_byte_offset == 8 * 128  # 1024 bytes

    def test_k128_bf16_multistripe_is_b128(self):
        """K=128 bf16 → row width 256 bytes = two B128 stripes."""
        la = pick_gmma_layout(elem_bytes=2, m_or_n=64, k=128, major=Major.K)
        assert la.layout_type == LayoutType.B128
        assert la.swizzle_code == 1
        assert la.tma_swizzle == "128B"
        assert la.smem_swizzle == "128B"
        assert la.leading_byte_offset == 64 * 128  # inter-stripe distance
        assert la.stride_byte_offset == 8 * 128  # stride within a stripe

    def test_k4_f32_is_interleave(self):
        """K=4 f32 (4 bytes/elem) → row width 16 bytes → INTERLEAVE."""
        la = pick_gmma_layout(elem_bytes=4, m_or_n=64, k=4, major=Major.K)
        assert la.layout_type == LayoutType.INTERLEAVE

    def test_rejects_non_canonical_row_width(self):
        """K=12 bf16 → row width 24 bytes → no canonical layout."""
        with pytest.raises(ValueError, match="not one of the canonical"):
            pick_gmma_layout(elem_bytes=2, m_or_n=64, k=12, major=Major.K)

    def test_rejects_non_8_m(self):
        with pytest.raises(ValueError, match="M divisible by 8"):
            pick_gmma_layout(elem_bytes=2, m_or_n=63, k=16, major=Major.K)


class TestPickMNMajor:
    """MN-major = row-major B of shape (K, N). Row width = N * elem_bytes."""

    def test_n8_bf16_is_interleave(self):
        """N=8 bf16 → row width 16 bytes = 1 u128 → INTERLEAVE.

        This is the degenerate-m=1 case — the N direction is exactly
        one core-matrix-tile wide. LBO carries the K-slab stride.
        """
        la = pick_gmma_layout(elem_bytes=2, m_or_n=8, k=16, major=Major.MN)
        assert la.layout_type == LayoutType.INTERLEAVE
        assert la.leading_byte_offset == 8 * 16  # 128 bytes

    def test_n16_bf16_is_b32(self):
        la = pick_gmma_layout(elem_bytes=2, m_or_n=16, k=16, major=Major.MN)
        assert la.layout_type == LayoutType.B32
        assert la.leading_byte_offset == 16
        assert la.stride_byte_offset == 8 * 32

    def test_rejects_non_8_k(self):
        with pytest.raises(ValueError, match="K divisible by 8"):
            pick_gmma_layout(elem_bytes=2, m_or_n=8, k=15, major=Major.MN)


class TestHelpers:
    def test_layout_for_a(self):
        la = layout_for_a(dtype=bf16, m=64, k=16)
        assert la.layout_type == LayoutType.B32
        assert la.stride_byte_offset == 256

    def test_layout_for_b_degenerate(self):
        lb = layout_for_b(dtype=bf16, k=16, n=8)
        assert lb.layout_type == LayoutType.INTERLEAVE
        assert lb.leading_byte_offset == 128

    def test_layout_for_a_f16_k16(self):
        la = layout_for_a(dtype=f16, m=64, k=16)
        # f16 = 2 bytes, same as bf16 → same layout
        assert la.layout_type == LayoutType.B32

    def test_layout_for_a_f32_k8(self):
        la = layout_for_a(dtype=f32, m=64, k=8)
        # f32 = 4 bytes, K=8 → row 32 bytes → B32
        assert la.layout_type == LayoutType.B32

    def test_layout_for_a_bf16_k128(self):
        la = layout_for_a(dtype=bf16, m=64, k=128)
        assert la.layout_type == LayoutType.B128
        assert la.leading_byte_offset == 64 * 128
        assert la.stride_byte_offset == 8 * 128


class TestGmmaLayoutSummary:
    """Sanity check that the picker gives different results for different
    shapes, so a consumer relying on it doesn't accidentally get a fixed
    value."""

    def test_row_widths_map_to_distinct_layouts(self):
        pairs = [
            (2, 8),  # 16 bytes → INTERLEAVE
            (2, 16),  # 32 bytes → B32
            (2, 32),  # 64 bytes → B64
            (2, 64),  # 128 bytes → B128
        ]
        layouts = set()
        for eb, k in pairs:
            la = pick_gmma_layout(elem_bytes=eb, m_or_n=64, k=k, major=Major.K)
            layouts.add(la.layout_type)
        assert layouts == {
            LayoutType.INTERLEAVE,
            LayoutType.B32,
            LayoutType.B64,
            LayoutType.B128,
        }
