"""Emit-only tests for ``ptx.mma.sync(...)`` — Ampere/Volta tensor-core MMA wrapper.

Verifies the wrapper produces the right PTX text for the canonical
m16n8k{8,16,32} dtype combos used on Ampere. No GPU needed.
"""
from __future__ import annotations

import pytest

from pyptx import kernel, ptx, reg, Tile
from pyptx.types import b32, bf16, f16, f32, tf32, u32, s8, s32


def _emit_ptx(kfn) -> str:
    """Trace the kernel with concrete dims and return the emitted PTX text."""
    # Force a trace by calling .ptx() if available, else trigger via shape resolution.
    # The Kernel class exposes ._trace internally; for emit-only tests we just want
    # the PTX string, which we can get by invoking _trace directly.
    from pyptx.emitter import emit
    module = kfn._trace()
    return emit(module)


# ---------------------------------------------------------------------------
# bf16 — the most common Ampere variant
# ---------------------------------------------------------------------------

def test_mma_sync_m16n8k16_bf16_emits_correct_opcode():
    @kernel(arch="sm_80", grid=(1, 1, 1), block=(32, 1, 1))
    def k():
        a = reg.array(b32, 4)
        b = reg.array(b32, 2)
        d = reg.array(f32, 4)
        ptx.mma.sync(
            shape=(16, 8, 16),
            dtype_d=f32, dtype_a=bf16, dtype_b=bf16, dtype_c=f32,
            d=[d[0], d[1], d[2], d[3]],
            a=[a[0], a[1], a[2], a[3]],
            b=[b[0], b[1]],
            c=[d[0], d[1], d[2], d[3]],
        )
        ptx.ret()

    text = _emit_ptx(k)
    assert "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32" in text


def test_mma_sync_default_layout_is_row_col():
    """Ampere mma.sync canonical layout is row.col — wrapper defaults match."""
    @kernel(arch="sm_80", grid=(1, 1, 1), block=(32, 1, 1))
    def k():
        a = reg.array(b32, 4); b = reg.array(b32, 2); d = reg.array(f32, 4)
        ptx.mma.sync(
            shape=(16, 8, 16),
            dtype_d=f32, dtype_a=bf16, dtype_b=bf16, dtype_c=f32,
            d=[d[0], d[1], d[2], d[3]],
            a=[a[0], a[1], a[2], a[3]],
            b=[b[0], b[1]],
            c=[d[0], d[1], d[2], d[3]],
        )
        ptx.ret()
    assert ".row.col." in _emit_ptx(k)


def test_mma_sync_explicit_col_row_layout():
    @kernel(arch="sm_80", grid=(1, 1, 1), block=(32, 1, 1))
    def k():
        a = reg.array(b32, 4); b = reg.array(b32, 2); d = reg.array(f32, 4)
        ptx.mma.sync(
            shape=(16, 8, 16),
            dtype_d=f32, dtype_a=bf16, dtype_b=bf16, dtype_c=f32,
            d=[d[0], d[1], d[2], d[3]],
            a=[a[0], a[1], a[2], a[3]],
            b=[b[0], b[1]],
            c=[d[0], d[1], d[2], d[3]],
            a_layout="col", b_layout="row",
        )
        ptx.ret()
    text = _emit_ptx(k)
    assert ".col.row." in text


# ---------------------------------------------------------------------------
# fp16 — alternative Ampere variant
# ---------------------------------------------------------------------------

def test_mma_sync_m16n8k16_fp16():
    @kernel(arch="sm_80", grid=(1, 1, 1), block=(32, 1, 1))
    def k():
        a = reg.array(b32, 4); b = reg.array(b32, 2); d = reg.array(f32, 4)
        ptx.mma.sync(
            shape=(16, 8, 16),
            dtype_d=f32, dtype_a=f16, dtype_b=f16, dtype_c=f32,
            d=[d[0], d[1], d[2], d[3]],
            a=[a[0], a[1], a[2], a[3]],
            b=[b[0], b[1]],
            c=[d[0], d[1], d[2], d[3]],
        )
        ptx.ret()
    assert ".m16n8k16.row.col.f32.f16.f16.f32" in _emit_ptx(k)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_mma_sync_rejects_bad_layout():
    @kernel(arch="sm_80", grid=(1, 1, 1), block=(32, 1, 1))
    def k():
        a = reg.array(b32, 4); b = reg.array(b32, 2); d = reg.array(f32, 4)
        with pytest.raises(ValueError, match="a_layout"):
            ptx.mma.sync(
                shape=(16, 8, 16),
                dtype_d=f32, dtype_a=bf16, dtype_b=bf16, dtype_c=f32,
                d=[d[0], d[1], d[2], d[3]],
                a=[a[0], a[1], a[2], a[3]],
                b=[b[0], b[1]],
                c=[d[0], d[1], d[2], d[3]],
                a_layout="bogus",
            )
        ptx.ret()
    _emit_ptx(k)


# ---------------------------------------------------------------------------
# cp.async wrappers — added at the same time, exercise the basic emit path
# ---------------------------------------------------------------------------

def test_cp_async_cg_emits_correct_opcode():
    @kernel(arch="sm_80", grid=(1, 1, 1), block=(32, 1, 1))
    def k():
        smem_dst = reg.scalar(u32, init=0)
        global_src = reg.scalar(u32, init=0)
        ptx.cp.async_.cg(ptx.addr(smem_dst), ptx.addr(global_src), 16)
        ptx.cp.async_.commit_group()
        ptx.cp.async_.wait_group(0)
        ptx.cp.async_.wait_all()
        ptx.ret()
    text = _emit_ptx(k)
    assert "cp.async.cg.shared.global" in text
    assert "cp.async.commit_group" in text
    assert "cp.async.wait_group" in text
    assert "cp.async.wait_all" in text
