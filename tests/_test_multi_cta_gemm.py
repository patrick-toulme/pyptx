"""Multi-CTA K-loop wgmma GEMM. Each CTA computes a 64x8 output tile;
grid covers (M/64) × (N/8) CTAs. Per-iteration tile_k=16.

First step toward 2kx2k. Start small (128x16, 256x32) for easy debugging.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

from pyptx import kernel, reg, smem, ptx, Tile
from pyptx.types import bf16, f32, b32, u32, u64, b64

_ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()


def build_gemm(M: int, N: int, K: int, *, tile_k: int = 16):
    BM, BN = 64, 8
    assert M % BM == 0 and N % BN == 0 and K % tile_k == 0, (
        f"M={M} N={N} K={K} not divisible by BM=64, BN=8, tile_k={tile_k}"
    )
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

        # Per-CTA tile offsets into A / B / C.
        #   ctaid.y = M tile index → row offset into A and C
        #   ctaid.x = N tile index → col offset into B and C
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
            # per-iter K offset (Python int, trace-time const)
            k_off = i * tile_k

            # compute "k_off_reg" once since it's used in both A and B TMA coords
            k_off_reg = reg.scalar(u32)
            ptx.inst.mov.u32(k_off_reg, k_off)

            with ptx.if_(tid == 0):
                ptx.mbarrier.arrive_expect_tx(bars[i][0], BM*tile_k*2 + tile_k*BN*2)
                # A coord: (k_off, row_offset)  — col varies with K, row = CTA M tile
                ptx.cp.async_.bulk.tensor_2d(
                    dst=sA[0], src=A.tma_desc(),
                    coord=(k_off_reg, row_offset),
                    mbar=bars[i][0],
                )
                # B coord: (col_offset, k_off)  — col = CTA N tile, row varies with K
                ptx.cp.async_.bulk.tensor_2d(
                    dst=sB[0], src=B.tma_desc(),
                    coord=(col_offset, k_off_reg),
                    mbar=bars[i][0],
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

        # Epilogue: scatter acc to C[row_offset + frag_row, col_offset + frag_col]
        # m64n8 thread layout — same as single-CTA, but offsets include
        # the per-CTA (row_offset, col_offset).
        tid2 = reg.scalar(u32); ptx.inst.mov.u32(tid2, ptx.special.tid.x())
        group = reg.scalar(u32); lane = reg.scalar(u32)
        ptx.inst.shr.u32(group, tid2, 5)
        ptx.inst.and_.b32(lane, tid2, 31)
        frag_row = reg.scalar(u32); frag_col = reg.scalar(u32); tmp = reg.scalar(u32)
        ptx.inst.shl.b32(frag_row, group, 4)
        ptx.inst.shr.u32(tmp, lane, 2)
        ptx.inst.add.u32(frag_row, frag_row, tmp)
        ptx.inst.and_.b32(frag_col, lane, 3)
        ptx.inst.shl.b32(frag_col, frag_col, 1)
        # Add per-CTA offsets.
        ptx.inst.add.u32(frag_row, frag_row, row_offset)
        ptx.inst.add.u32(frag_col, frag_col, col_offset)

        pc = reg.scalar(b64); ptx.inst.ld.param.u64(pc, ptx.addr(C))
        ptx.inst.cvta.to.global_.u64(pc, pc)
        # Address = pc + (frag_row * N + frag_col) * 4
        row_col = reg.scalar(u32)
        ptx.inst.mov.u32(row_col, N)  # N = full output width
        ptx.inst.mul.lo.u32(row_col, frag_row, row_col)
        ptx.inst.add.u32(row_col, row_col, frag_col)
        off = reg.scalar(u64)
        ptx.inst.mul.wide.u32(off, row_col, 4)
        p0 = reg.scalar(b64)
        ptx.inst.add.s64(p0, pc, off)
        ptx.inst.st.global_.f32(ptx.addr(p0), acc[0])
        ptx.inst.st.global_.f32(ptx.addr(p0, 4), acc[1])
        # Row+8 for the second pair of fragments
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
        ptx.ret()

    return gemm


def run_test(M: int, N: int, K: int, tile_k: int = 16, *, atol: float = 0.0):
    k = build_gemm(M, N, K, tile_k=tile_k)
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
        A, B, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32,
    ))
    md = float(np.abs(out - ref).max())
    if atol == 0.0:
        ok = bool(np.array_equal(out, ref))
    else:
        ok = bool(np.abs(out - ref).max() <= atol)
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] M={M:5d} N={N:5d} K={K:5d}  max_diff={md:g}")
    if not ok:
        print(f"         out[0,:4]={out[0,:4]}")
        print(f"         ref[0,:4]={ref[0,:4]}")
        print(f"         out[{M-1},{N-4}:]={out[M-1,N-4:]}")
        print(f"         ref[{M-1},{N-4}:]={ref[M-1,N-4:]}")
    return ok


if __name__ == "__main__":
    # Progressive sizes: 1 CTA → multi-CTA → big
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        sizes = [
            (64, 8, 32),
            (64, 8, 128),
            (128, 16, 32),
            (128, 16, 128),
            (256, 32, 256),
            (512, 64, 512),
            (1024, 128, 1024),
            (2048, 2048, 2048),
        ]
    else:
        sizes = [
            (64, 8, 32),
            (128, 8, 32),
            (64, 16, 32),
            (128, 16, 32),
            (256, 32, 128),
        ]
    for M, N, K in sizes:
        # For K >= 1024, bf16 summation order diverges slightly from jax ref.
        atol = 0.0 if K <= 512 else 1e-3
        run_test(M, N, K, atol=atol)
