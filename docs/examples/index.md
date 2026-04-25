# Examples

Kernels split by target architecture:

- **Hopper (`sm_90a`)** in `examples/hopper/` — WGMMA, TMA 2D/3D,
  mbarriers, cluster launch.
- **Blackwell (`sm_100a`)** in `examples/blackwell/` — `tcgen05.mma`,
  TMEM, SMEM + instruction descriptors, 2-SM `cta_group::2`
  cooperative MMA.

Each example is complete and runnable with correctness tests against
JAX and PyTorch references.

## Blackwell production kernels

<div class="grid cards" markdown>

-   :material-matrix: **[Blackwell GEMM](blackwell/gemm_highperf_blackwell.md)**

    ---

    **1168 TFLOPS** at 8192³ bf16 (1SM, 4-stage pipeline).
    **613 TFLOPS** at 2048³ (2SM, beats 1SM at small sizes).

    Warp-specialized TMA + MMA on top of `tcgen05.mma.kind::f16`.
    2SM variant uses `cta_group::2`, a cluster-shared mbarrier
    hand-off, and 6-stage pipelining.

    `~450 lines · build_gemm / build_gemm_2sm`

-   :material-view-grid: **[Blackwell Grouped GEMM](blackwell/grouped_gemm.md)**

    ---

    MoE-scale grouped GEMM on `tcgen05.mma`. Bit-exact through
    G=4 M=2048 N=256 K=2048 vs `einsum("gmk,gkn->gmn")`.

    Same 3-stage warp-specialized mainloop as the 1SM GEMM, grid
    gains a Z dimension for the expert index.

    `~300 lines · tcgen05 · MoE shapes`

-   :material-microscope: **[tcgen05 primitive suite](blackwell/tcgen05_suite.md)**

    ---

    13 isolated tests for every Blackwell primitive:
    `tcgen05.alloc`, `.mma`, `.ld`, commit/fence, split-commit
    accumulation, SMEM-descriptor round-trip, GEMM probes.
    **Run this first on a fresh B200** — if it's 13/13, the
    runtime stack is good.

    `~260 lines · 13 probes · runtime sanity check`

</div>

## Hopper production kernels

<div class="grid cards" markdown>

-   :material-matrix: **[Hopper GEMM](hopper/gemm_highperf_hopper.md)**

    ---

    **815 TFLOPS** at 8192³ bf16 — beats cuBLAS at ≥6K.

    Warp-specialized: 1 producer + 2 consumer WGs. 2-CTA clusters
    with TMA multicast. m64n256k16 WGMMA. 3-stage SMEM pipeline.
    Hilbert tile schedule.

    `~460 lines · dynamic SMEM · cluster launch`

-   :material-view-grid: **[Grouped GEMM](hopper/grouped_gemm.md)**

    ---

    **104 TFLOPS** at G=8 M=K=2048 (MoE expert shape).

    Per-CTA K-loop with `tile_k=64` multi-k WGMMA (4 WGMMAs per
    iter). Grid parallelizes over `(N/tile_n, M/BM, G)`.

    `~255 lines · multi-k wgmma · v2 epilogue`

-   :material-format-letter-case: **[RMS norm](hopper/rms_norm.md)**

    ---

    **2.6 TB/s** at B=2048 N=8192 f32 (88% of HBM3 peak). **3.9×**
    faster than torch eager.

    One CTA per row. v4 loads. Two-pass warp reduce via butterfly
    shuffle.

    `~240 lines · v4 ld/st · warp reduce`

-   :material-format-letter-case: **[Layer norm](hopper/layer_norm.md)**

    ---

    **2.5 TB/s** at B=2048 N=8192 f32 (83% of HBM3 peak). **1.5×**
    faster than `F.layer_norm`.

    Same structure as RMS norm with sum + sum-of-squares accumulated
    together; computes `y = (x - µ) * rstd * w + b` in one pass.

    `~270 lines · v4 ld/st · 2 reductions`

-   :material-function-variant: **[SwiGLU](hopper/swiglu.md)**

    ---

    **2.8 TB/s** at M=2048 F=8192 f32 (94% of HBM3 peak). **1.6×**
    faster than `F.silu(g) * u`.

    Fused `silu(gate) * up`. v4 loads. `ex2.approx` + `rcp.approx`
    for a fast sigmoid.

    `~180 lines · v4 ld/st · bandwidth-bound`

-   :material-eye: **[Flash attention (Hopper)](hopper/experimental/flash_attention_hopper.md)**

    ---

    **3.0×** faster than naive `softmax(QK^T) @ V` on H100.

    Q-tile parallel. WGMMA m64n64k16 for Q@K^T and P@V with multi-k
    iteration (head_dim up to 64). Online softmax with per-row
    running max/sum.

    `~380 lines · multi-k wgmma · online softmax`

</div>

## Tutorial kernels

<div class="grid cards" markdown>

-   :material-school: **[FA-2 tutorial kernel (`flash_attention_wgmma_kloop`)](hopper/experimental/flash_attention_wgmma_kloop.md)**

    ---

    Single-CTA FlashAttention-2 with online softmax, BM=64, BN=16,
    HEAD_DIM=16. Designed to read, not to win benchmarks. Good entry
    point to the Hopper primitives.

</div>
