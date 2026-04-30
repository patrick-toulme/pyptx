"""High-perf A100 (sm_80) bf16 GEMM — ldmatrix + cp.async + 3-stage SMEM.

Follows the CUTLASS SM80 production design:

- **128 × 128 × 32 CTA tile**, 4 warps/CTA arranged 2×2 in (M, N), each warp
  owning a 64 × 64 output sub-tile.
- Per warp per K-iter: **64 ``mma.sync.m16n8k16``** (4 M-frags × 8 N-frags × 2
  K-blocks). 256 mma per CTA per K-iter.
- **ldmatrix.sync.aligned.m8n8.x4.shared.b16** loads each 16×16 A fragment in
  one warp-collective instruction (4 b32/lane = the full A m16n8k16 fragment).
- **ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16** loads each 16×16 worth
  of B (= 2 m16n8k16 B fragments) — ``.trans`` swaps SMEM-rows (N) with
  fragment-rows (K), matching mma's row.col B layout for SMEM stored as
  ``B_T`` (N, K) row-major.
- **3-stage cp.async ring buffer** (CUTLASS kStages=3 default). Each thread
  issues 16-byte cp.async.cg per matrix per stage; with stages=3 the
  ``wait_group(STAGES-2)`` pattern absorbs warp skew naturally — no
  ``bar.sync`` between mma and prefetch is required.

References:
  - CUTLASS ``include/cutlass/gemm/device/default_gemm_configuration.h``
    (SM80 TensorOp: kStages=3, ThreadblockShape 128×128, WarpShape 64×64,
    InstructionShape 16×8×16).
  - CUTLASS ``examples/cute/tutorial/sgemm_sm80.cu``
    (``SM75_U32x4_LDSM_N`` for A, ``.x4.trans`` for B, K_PIPE_MAX=3,
    ``SM80_CP_ASYNC_CACHEALWAYS<uint128_t>``).

Inputs:
  A:   (M, K) bf16 row-major, ``M % 128 == 0``, ``K % 32 == 0``
  B_T: (N, K) bf16 row-major, ``N % 128 == 0``  (transposed B; mma row.col
       wants K-contiguous on B's leading axis)
  D:   (M, N) f32 row-major output

Grid:  (N // 128, M // 128, 1)
Block: (128, 1, 1)
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from pyptx import kernel, ptx, reg, smem, Tile
from pyptx.types import b32, bf16, f32, u32


BM, BN, BK = 128, 128, 32
NUM_WARPS = 4
THREADS = 32 * NUM_WARPS              # 128
WARPS_M, WARPS_N = 2, 2
WM, WN = BM // WARPS_M, BN // WARPS_N  # 64, 64
N_FRAG_M = WM // 16                    # 4
N_FRAG_N = WN // 8                     # 8
N_FRAG_K = BK // 16                    # 2

STAGES = 4                             # 4 SMEM buffers, 3 in flight (CUTLASS pattern)
A_STAGE_BYTES = BM * BK * 2            # 8192
B_STAGE_BYTES = BN * BK * 2            # 8192
A_SMEM_BASE = 0
B_SMEM_BASE = STAGES * A_STAGE_BYTES
SMEM_BYTES = STAGES * (A_STAGE_BYTES + B_STAGE_BYTES)   # 64 KB (extern_smem opt-in)


def build_gemm_highperf(M: int, N: int, K: int, *, arch: str = "sm_80"):
    """Build the production A100 bf16 GEMM kernel."""
    assert M % BM == 0, f"M={M} must be divisible by {BM}"
    assert N % BN == 0, f"N={N} must be divisible by {BN}"
    assert K % BK == 0, f"K={K} must be divisible by {BK}"
    n_iters = K // BK

    @kernel(
        in_specs=(
            Tile(M, K, bf16),
            Tile(N, K, bf16),
        ),
        out_specs=(Tile(M, N, f32),),
        grid=(N // BN, M // BM, 1),
        block=(THREADS, 1, 1),
        arch=arch,
        smem=SMEM_BYTES,
        extern_smem=True,
    )
    def gemm(A, B_T, D):
        pa, pb, pd = ptx.global_ptrs(A, B_T, D)
        smem_base = smem.base()

        # CTA bases.
        m_base = reg.scalar(u32)
        ptx.inst.mov.u32(m_base, ptx.special.ctaid.y())
        ptx.inst.shl.b32(m_base, m_base, 7)         # * 128
        n_base = reg.scalar(u32)
        ptx.inst.mov.u32(n_base, ptx.special.ctaid.x())
        ptx.inst.shl.b32(n_base, n_base, 7)         # * 128

        # Thread + warp identity.
        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        warp_id = tid >> 5                          # 0..3
        lane = tid & 31                             # 0..31
        warp_m = warp_id >> 1                       # 0..1 (top 2 warps M=0, bottom M=1)
        warp_n = warp_id & 1                        # 0..1
        warp_m_smem = warp_m << 6                   # 0 or 64 (warp's M base in SMEM rows)
        warp_n_smem = warp_n << 6                   # 0 or 64

        # ----- cp.async layout -----
        # Each matrix (A or B) is 128 rows × 32 cols of bf16 = 8 KB per stage.
        # 128 threads × 16 B/thread = 2 KB per pass; 4 passes per stage per matrix.
        # Map 4 threads per row (each thread 8 bf16 = 16 B), 32 rows per pass.
        load_row_in_pass = tid >> 2                  # 0..31 (1 row per 4 threads)
        load_col_chunk = tid & 3                     # 0..3  (which 8-bf16 chunk in K=32)
        load_col_off = load_col_chunk << 3           # 0,8,16,24 bf16
        load_smem_col_bytes = load_col_off * 2       # bytes within a row

        # SMEM XOR swizzle (CUTLASS / MatmulTutorial v9 / v15 pattern).
        # For 64-byte SMEM rows (BK=32 bf16 = 4 atoms × 16 B), swizzle the
        # 4 atom slots per row using 2 bits of the row index. The exact
        # formula matches MatmulTutorial v15 (line 38, 76, etc.):
        #   col_eff = col ^ ((row & 3) << 3)        (col in bf16, 4-row period)
        # which is equivalent to:
        #   atom_eff = atom ^ (row & 3)              (atom in 16 B, 4-row period)
        # Applied identically to cp.async writes, ldmatrix reads, and (if
        # used) any epilogue SMEM round-trip — the formula MUST match on
        # all paths or data lands in the wrong bank.
        # MatmulTutorial v9 reports +13% from this on their fp16 kernel.
        USE_SWIZZLE = True

        def smem_swizzle_byte_off(row, atom_idx, sub_atom_bytes=0):
            if USE_SWIZZLE:
                swizzled_atom = atom_idx ^ (row & 3)
                return row * (BK * 2) + swizzled_atom * 16 + sub_atom_bytes
            return row * (BK * 2) + atom_idx * 16 + sub_atom_bytes

        # ----- Precompute per-thread cp.async offsets (loop-invariant) -----
        # cp.async write SMEM offset and global row-base address are
        # per-thread constants. Hoisting these out of the K-loop turns
        # each cp.async into 1-2 ops (add + store) instead of ~8.
        cp_smem_off = []          # per-pass: SMEM byte offset (with swizzle)
        cp_a_row_base_x_K = []    # per-pass: (m_base + row_off) * K, in elements
        cp_b_row_base_x_K = []    # per-pass: (n_base + row_off) * K, in elements
        for pass_id in range(4):
            row_off = load_row_in_pass + (pass_id << 5)
            cp_smem_off.append(smem_swizzle_byte_off(row_off, load_col_chunk))
            cp_a_row_base_x_K.append((m_base + row_off) * K)
            cp_b_row_base_x_K.append((n_base + row_off) * K)

        def issue_cp_async(stage: int, k_idx_reg):
            """Issue cp.async for both A and B at stage, K = k_idx_reg.

            32 rows per pass × 4 passes = 128 rows of each matrix.
            Each thread writes one 16-byte atom; swizzled SMEM dst for ldmatrix.
            """
            a_stage = smem_base + (A_SMEM_BASE + stage * A_STAGE_BYTES)
            b_stage = smem_base + (B_SMEM_BASE + stage * B_STAGE_BYTES)
            for pass_id in range(4):
                a_smem_dst = a_stage + cp_smem_off[pass_id]
                b_smem_dst = b_stage + cp_smem_off[pass_id]
                a_glob_off = (cp_a_row_base_x_K[pass_id] + k_idx_reg + load_col_off) * 2
                b_glob_off = (cp_b_row_base_x_K[pass_id] + k_idx_reg + load_col_off) * 2
                ptx.cp.async_.cg(ptx.addr(a_smem_dst), ptx.addr(pa + a_glob_off), 16)
                ptx.cp.async_.cg(ptx.addr(b_smem_dst), ptx.addr(pb + b_glob_off), 16)

        # ----- Accumulator: 4 M-frag × 8 N-frag × 4 f32 = 128 f32/lane -----
        ACC_TILES = N_FRAG_M * N_FRAG_N
        acc = reg.array(f32, ACC_TILES * 4)
        zero = reg.scalar(f32, init=0.0)
        for i in range(ACC_TILES * 4):
            ptx.inst.mov.f32(acc[i], zero)

        # ----- Prologue: prime STAGES-1 stages -----
        for s in range(STAGES - 1):
            k_s = reg.scalar(u32, init=s * BK)
            issue_cp_async(s, k_s)
            ptx.cp.async_.commit_group()

        # ----- ldmatrix lane-to-address math -----
        # ldmatrix.m8n8.x4 (no trans) — for A.
        # Loads 4 8x8 b16 matrices arranged 2x2 in (rows, cols):
        #   mat 0: rows 0..7,  cols 0..7   (lanes 0..7  provide row pointers)
        #   mat 1: rows 8..15, cols 0..7   (lanes 8..15 provide row pointers)
        #   mat 2: rows 0..7,  cols 8..15  (lanes 16..23 provide row pointers)
        #   mat 3: rows 8..15, cols 8..15  (lanes 24..31 provide row pointers)
        # After load, lane t holds {mat0[t/4, 2*(t%4)..2*(t%4)+1], ...} which
        # exactly matches the m16n8k16 A fragment per-thread layout.
        #
        # Per-lane address: row = t & 15, col_off_in_frag = (t >> 4) << 3.
        a_ldsm_row = lane & 15                                # 0..15
        a_ldsm_col_in_frag = (lane >> 4) << 3                 # 0 or 8 (bf16)

        # ldmatrix.m8n8.x4 (no .trans) — for B.
        # SMEM stores B_T = (N, K) row-major, so SMEM[N, K] gives the bf16
        # value at B[K, N]. For mma's row.col B fragment, lane t holds
        # B[K=2*(t%4)+{0,1,8,9}, N=t/4] — i.e. K-stride at fixed N. That's
        # exactly what ldmatrix (no trans) produces from B_T SMEM:
        #     reg[i] = mat[i][row=t/4, cols=2*(t%4)..2*(t%4)+1]
        #            = B_T[N0 + t/4, K0 + 2*(t%4)..2*(t%4)+1]
        #            = B[K0 + 2*(t%4)..2*(t%4)+1, N0 + t/4]    (matches mma).
        # The 4 mats cover 16K × 16N (= 2 m16n8k16 N-frags) per call:
        #   mat 0: SMEM rows N0..N0+7,    K=K0..K0+7    (K-low, N-frag 0)
        #   mat 1: SMEM rows N0..N0+7,    K=K0+8..K0+15 (K-high, N-frag 0)
        #   mat 2: SMEM rows N0+8..N0+15, K=K0..K0+7    (K-low, N-frag 1)
        #   mat 3: SMEM rows N0+8..N0+15, K=K0+8..K0+15 (K-high, N-frag 1)
        #
        # Per-lane address: N-row = ((t >> 4) << 3) + (t & 7),
        #                   K-col = ((t >> 3) & 1) << 3.
        b_ldsm_row_in_pair = ((lane >> 4) << 3) + (lane & 7)  # 0..15
        b_ldsm_col_in_block = ((lane >> 3) & 1) << 3          # 0 or 8 (bf16)

        # ----- Double-buffered fragment registers -----
        # Two register banks alternate per K-block within an iter, AND across
        # iter boundaries (CUTLASS-style ping-pong). Bank 0 holds (ki, kb=0),
        # bank 1 holds (ki, kb=1). At end of iter ki we pre-load (ki+1, kb=0)
        # into bank 0 — overlapping with bank-1 mma — so the first mma of
        # ki+1 has zero ldmatrix latency.
        a_fr_buf = [reg.array(b32, N_FRAG_M * 4) for _ in range(2)]
        b_fr_buf = [reg.array(b32, (N_FRAG_N // 2) * 4) for _ in range(2)]

        a_atom_lane_part = lane >> 4                            # 0 or 1
        b_atom_lane_part = (lane >> 3) & 1                      # 0 or 1

        # ----- Precomputed per-lane SMEM byte offsets (loop-invariant) -----
        # Each ldmatrix's address = current_stage_base + precomputed_offset.
        # Hoisting the row*64 + atom*16 (+ swizzle XOR) computation out of
        # the K-loop turns each inner-loop ldmatrix from 5-7 PTX address
        # ops into a single add. With 16 ldmatrix per K-iter × 128 K-iters
        # at K=4096, this saves ~10K PTX ops per warp.
        a_ldsm_off = []                       # [mf][kb] → Reg (byte offset within stage)
        for mf in range(N_FRAG_M):
            row_for_mf = warp_m_smem + (mf << 4) + a_ldsm_row
            row_swizz = (row_for_mf & 3) if USE_SWIZZLE else 0
            row_x_64 = row_for_mf * (BK * 2)
            row_off = []
            for kb_ in range(N_FRAG_K):
                atom = (kb_ << 1) + a_atom_lane_part
                atom_eff = (atom ^ row_swizz) if USE_SWIZZLE else atom
                row_off.append(row_x_64 + atom_eff * 16)
            a_ldsm_off.append(row_off)

        b_ldsm_off = []                       # [np_][kb] → Reg
        for np_ in range(N_FRAG_N // 2):
            row_for_np = warp_n_smem + (np_ << 4) + b_ldsm_row_in_pair
            row_swizz = (row_for_np & 3) if USE_SWIZZLE else 0
            row_x_64 = row_for_np * (BK * 2)
            row_off = []
            for kb_ in range(N_FRAG_K):
                atom = (kb_ << 1) + b_atom_lane_part
                atom_eff = (atom ^ row_swizz) if USE_SWIZZLE else atom
                row_off.append(row_x_64 + atom_eff * 16)
            b_ldsm_off.append(row_off)

        def load_a_kb(stage_base, kb_, dst_buf):
            """Issue 4 ldmatrix.x4 calls into dst_buf for K-block kb_."""
            for mf in range(N_FRAG_M):
                addr = stage_base + a_ldsm_off[mf][kb_]
                base_idx = mf * 4
                ptx.ldmatrix(
                    dst=[dst_buf[base_idx + 0], dst_buf[base_idx + 1],
                         dst_buf[base_idx + 2], dst_buf[base_idx + 3]],
                    src=ptx.addr(addr),
                    layout="m8n8.x4",
                )

        def load_b_kb(stage_base, kb_, dst_buf):
            """Issue 4 ldmatrix.x4 calls into dst_buf for K-block kb_."""
            for np_ in range(N_FRAG_N // 2):
                addr = stage_base + b_ldsm_off[np_][kb_]
                base_idx = np_ * 4
                ptx.ldmatrix(
                    dst=[dst_buf[base_idx + 0], dst_buf[base_idx + 1],
                         dst_buf[base_idx + 2], dst_buf[base_idx + 3]],
                    src=ptx.addr(addr),
                    layout="m8n8.x4",
                )

        def mma_buf(a_buf, b_buf):
            """Issue 32 mma.sync (4 M × 8 N) using one register bank pair.

            Serpentine N-order on odd M-rows: ``n = (mf & 1) ? N-1-nf : nf``.
            Adjacent mmas share one operand (better register reuse, fewer
            ldmatrix-induced stalls). MatmulTutorial v13 reports +7% from this.
            """
            for mf in range(N_FRAG_M):
                a_base = mf * 4
                a_regs = [a_buf[a_base + i] for i in range(4)]
                for nf_iter in range(N_FRAG_N):
                    nf = (N_FRAG_N - 1 - nf_iter) if (mf & 1) else nf_iter
                    np_ = nf >> 1
                    nf_in_pair = nf & 1
                    b_base = np_ * 4
                    b_lo = b_base + (nf_in_pair << 1) + 0
                    b_hi = b_base + (nf_in_pair << 1) + 1
                    acc_base = (mf * N_FRAG_N + nf) * 4
                    ptx.mma.sync(
                        shape=(16, 8, 16),
                        dtype_d=f32, dtype_a=bf16, dtype_b=bf16, dtype_c=f32,
                        d=[acc[acc_base + i] for i in range(4)],
                        a=a_regs,
                        b=[b_buf[b_lo], b_buf[b_hi]],
                        c=[acc[acc_base + i] for i in range(4)],
                        a_layout="row", b_layout="col",
                    )

        # ----- Pre-load (iter 0, kb=0) into bank 0 -----
        # Wait for stage 0 to land first.
        wait0_target = max(0, STAGES - 2)
        ptx.cp.async_.wait_group(wait0_target)
        ptx.bar.sync(0)
        a_stage0 = smem_base + A_SMEM_BASE + 0
        b_stage0 = smem_base + B_SMEM_BASE + 0
        load_a_kb(a_stage0, 0, a_fr_buf[0])
        load_b_kb(b_stage0, 0, b_fr_buf[0])

        # ----- Main K loop with cross-iter register double-buffering -----
        for ki in range(n_iters):
            stage = ki % STAGES
            a_stage_base = smem_base + A_SMEM_BASE + stage * A_STAGE_BYTES
            b_stage_base = smem_base + B_SMEM_BASE + stage * B_STAGE_BYTES

            # bank 0 already holds (ki, kb=0). Pre-load (ki, kb=1) into bank 1.
            load_a_kb(a_stage_base, 1, a_fr_buf[1])
            load_b_kb(b_stage_base, 1, b_fr_buf[1])

            # mma using bank 0 (overlaps with bank-1 ldmatrix above).
            mma_buf(a_fr_buf[0], b_fr_buf[0])

            # Issue prefetch for ki + STAGES - 1 into stage (ki+3) % 4.
            if ki + (STAGES - 1) < n_iters:
                k_next = reg.scalar(u32)
                ptx.inst.mov.u32(k_next, (ki + STAGES - 1) * BK)
                next_prefetch_stage = (ki + STAGES - 1) % STAGES
                issue_cp_async(next_prefetch_stage, k_next)
                ptx.cp.async_.commit_group()

            # Wait for next iter's stage + pre-load (ki+1, kb=0) into bank 0.
            if ki + 1 < n_iters:
                tail = max(0, (ki + 1) - (n_iters - STAGES))
                wait_target = max(0, (STAGES - 2) - tail)
                ptx.cp.async_.wait_group(wait_target)
                ptx.bar.sync(0)
                next_stage = (ki + 1) % STAGES
                a_next = smem_base + A_SMEM_BASE + next_stage * A_STAGE_BYTES
                b_next = smem_base + B_SMEM_BASE + next_stage * B_STAGE_BYTES
                load_a_kb(a_next, 0, a_fr_buf[0])
                load_b_kb(b_next, 0, b_fr_buf[0])

            # mma using bank 1 (overlaps with the bank-0 cross-iter pre-load).
            mma_buf(a_fr_buf[1], b_fr_buf[1])

        # ----- Epilogue: write 4 M × 8 N × 4 f32 acc tiles to global D -----
        # m16n8 fragment per-lane: rows {gid, gid+8} × cols {2*tig, 2*tig+1}.
        gid = lane >> 2                                     # 0..7
        tig = lane & 3                                      # 0..3
        col_lo = tig << 1                                   # 0,2,4,6
        warp_m_global = m_base + warp_m_smem                # CTA M + warp M
        warp_n_global = n_base + warp_n_smem                # CTA N + warp N
        for mf in range(N_FRAG_M):
            m_frag_global = warp_m_global + (mf << 4)
            row_lo = m_frag_global + gid
            row_hi = row_lo + 8
            for nf in range(N_FRAG_N):
                n_frag_global = warp_n_global + (nf << 3)
                d_col_base = n_frag_global + col_lo
                acc_base = (mf * N_FRAG_N + nf) * 4
                for i, (drow, dcol) in enumerate([(0, 0), (0, 1), (8, 0), (8, 1)]):
                    row = row_lo if drow == 0 else row_hi
                    col_elem = d_col_base + dcol
                    elem_idx = row * N + col_elem
                    byte_off = elem_idx * 4
                    ptx.inst.st.global_.f32(ptx.addr(pd + byte_off), acc[acc_base + i])

        ptx.ret()

    return gemm


# ---------------------------------------------------------------------------
# Reference + test harness
# ---------------------------------------------------------------------------

def gemm_ref(A: jnp.ndarray, B_T: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("mk,nk->mn", A.astype(jnp.float32), B_T.astype(jnp.float32))


def _run_jax_case(M: int, N: int, K: int) -> None:
    k = build_gemm_highperf(M, N, K)
    rng = np.random.default_rng(M * 7919 + N * 31 + K)
    A = jnp.asarray(rng.standard_normal((M, K), dtype=np.float32) * 0.1, dtype=jnp.bfloat16)
    BT = jnp.asarray(rng.standard_normal((N, K), dtype=np.float32) * 0.1, dtype=jnp.bfloat16)

    @jax.jit
    def fn(A, BT):
        return k(A, BT)

    out = np.asarray(fn(A, BT))
    ref = np.asarray(gemm_ref(A, BT))
    diff = float(np.abs(out - ref).max())
    ok = bool(np.allclose(out, ref, atol=1e-2, rtol=1e-2))
    status = "OK  " if ok else "FAIL"
    print(f"[JAX  {status}] M={M:5d} N={N:5d} K={K:5d}  max_abs={diff:.3e}")


def _run_torch_case(M: int, N: int, K: int) -> None:
    import torch
    k = build_gemm_highperf(M, N, K)
    rng = np.random.default_rng(M * 7919 + N * 31 + K)
    A = torch.tensor(rng.standard_normal((M, K), dtype=np.float32) * 0.1,
                     dtype=torch.bfloat16, device="cuda")
    BT = torch.tensor(rng.standard_normal((N, K), dtype=np.float32) * 0.1,
                      dtype=torch.bfloat16, device="cuda")
    out = k(A, BT)
    torch.cuda.synchronize()
    ref = (A.float() @ BT.float().T)
    diff = float((out - ref).abs().max())
    ok = bool(torch.allclose(out, ref, atol=1e-2, rtol=1e-2))
    status = "OK  " if ok else "FAIL"
    print(f"[Torch{status}] M={M:5d} N={N:5d} K={K:5d}  max_abs={diff:.3e}")


def main() -> None:
    _ = (jnp.ones((4,), dtype=jnp.float32) + 1).block_until_ready()
    for M, N, K in [
        (128, 128, 32),    # single CTA, single K-iter
        (128, 128, 64),    # single CTA, 2 K-iters
        (128, 128, 256),   # single CTA, 8 K-iters
        (256, 256, 256),   # 2x2 CTAs
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]:
        _run_jax_case(M, N, K)
        _run_torch_case(M, N, K)


if __name__ == "__main__":
    main()
