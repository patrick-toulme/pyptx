# Performance

Hopper numbers are from an H100 SXM5 (80 GB HBM3, ~3 TB/s peak bandwidth,
~989 TFLOPS bf16 tensor core peak). Blackwell numbers are from a B200
(~2250 TFLOPS bf16 tensor core peak, dense). Each kernel is measured with
CUDA events amortized over 20–200 iterations after a short warmup.

Reference timings use PyTorch 2.8 / CUDA 12.8 eager or fused kernels,
whichever is the strongest publicly available comparison point.

Reproduce Hopper benchmarks with:

```bash
python benchmarks/bench_final.py
python benchmarks/bench_hopper_gemm.py --size 8192 --iters 16
```

Reproduce Blackwell benchmarks with:

```bash
python benchmarks/bench_blackwell_gemm.py
python benchmarks/bench_blackwell_kernels.py
python examples/blackwell/tcgen05_suite.py
```

---

## Compute kernels

### Blackwell GEMM (tcgen05, bf16)

Warp-specialized 4-stage pipeline: warp 0 lane 0 issues TMA 2D loads for
A and B into a ring buffer; warp 1 lane 0 waits on load barriers and
dispatches `tcgen05.mma.cta_group::1.kind::f16` (m128n256k16) with a
single-commit-per-CTA-tile handshake. Grid is N-major so consecutive
CTAs walk N (unit-stride) for L2 reuse. TMEM accumulator is read back
in two `.32x32b.x128` loads per warp. See
`examples/blackwell/gemm_highperf_blackwell.py`.

| Shape (M × N × K) | 1SM TFLOPS | 2SM TFLOPS | cuBLAS TFLOPS | best / cuBLAS |
| --- | --- | --- | --- | --- |
| 2048³ | 645 | **649** | 1006 | 64% |
| 4096³ | **1194** | 1168 | 1532 | 78% |
| 4096 × 8192 × 8192 | 1240 | **1268** | 1652 | 77% |
| 8192 × 4096 × 8192 | **1315** | 1230 | 1648 | 80% |
| 8192³ | **1240** | 1046 | 1610 | 77% |

The 2SM variant (``build_gemm_2sm``) uses ``tcgen05.mma.cta_group::2``,
a cluster-shared mbarrier hand-off (count=2, self + peer via
``arrive_remote``), ``tcgen05.commit.multicast::cluster`` for per-slot
consumed signaling, and a 5-stage pipeline. Each CTA holds ``BN/2``
cols of B so the extra stages fit inside the dynamic SMEM budget. The
2SM path is useful on selected shapes, but the 1SM kernel remains the
default maintained path because it is simpler and usually faster once K
gets large. The current remaining gap to cuBLAS is largely in the
cluster/store pipeline, not basic tensor-core bring-up.

### Blackwell grouped GEMM (tcgen05, bf16 -> f32)

Grouped GEMM reuses the same Blackwell ``tcgen05.mma`` mainloop with a
G-problem outer loop for MoE-style shapes.

| G | M | N | K | pyptx TFLOPS | vs torch |
| --- | --- | --- | --- | --- | --- |
| 8 | 512 | 128 | 512 | 41.6 | — |
| 8 | 1024 | 128 | 1024 | 160.1 | — |
| 4 | 2048 | 256 | 2048 | **401.0** | **~10.0×** |

The large-shape Torch and JAX paths are both correct on the maintained
Blackwell grouped GEMM. See ``benchmarks/bench_blackwell_kernels.py`` for
the current kernel suite.

### Hopper GEMM (warp-specialized, bf16)

Hand-written m64n256k16 WGMMA kernel with a producer + 2 consumer warpgroups,
2-CTA clusters with TMA multicast, 3-stage SMEM pipeline, and a Hilbert
tile schedule. Transpiled from `fast.cu`'s kernel12 and sugared in
`examples/hopper/gemm_highperf_hopper.py`.

| Size | pyptx | TFLOPS | vs cuBLAS |
| --- | --- | --- | --- |
| 2048³ | 0.087 ms | 198 | — |
| 4096³ | 0.176 ms | 783 | ~parity |
| 6144³ | 0.580 ms | 802 | **+10%** |
| 8192³ | 1.349 ms | **815** | **+10%** |

Beats cuBLAS at matrix sizes ≥ 6144.

### Grouped GEMM (bf16 → f32, MoE shapes)

Multi-k WGMMA with `tile_k=64` (4 WGMMAs per K-loop iter). Used for
MoE where every expert has equal capacity.

| G | M | N | K | pyptx | TFLOPS |
| --- | --- | --- | --- | --- | --- |
| 8 | 512 | 64 | 512 | 17 µs | 16 |
| 16 | 512 | 64 | 512 | 17 µs | 32 |
| 8 | 1024 | 64 | 1024 | 17 µs | 63 |
| 4 | 2048 | 64 | 2048 | 30 µs | 72 |
| 8 | 2048 | 64 | 2048 | 41 µs | **104** |

2.3× over the naive single-WGMMA-per-iter implementation.

---

## Bandwidth-bound kernels

All three norm/activation kernels hit the **launch-overhead floor of
~14 µs** for small shapes (torch has a ~7 µs floor via direct
`cuLaunchKernel`). At large shapes they saturate HBM3.

### RMS norm (f32)

| B | N | pyptx | torch | pyptx GB/s | torch GB/s | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 1024 | 14 µs | 27 µs | 19 | 10 | **1.9×** |
| 256 | 4096 | 14 µs | 28 µs | 602 | 298 | **2.0×** |
| 1024 | 8192 | 31 µs | 104 µs | 2176 | 644 | **3.4×** |
| 2048 | 8192 | 51 µs | 199 µs | **2647** | 675 | **3.9×** |

### Layer norm (f32)

Compared against `torch.nn.functional.layer_norm` (not eager).

| B | N | pyptx | torch | pyptx GB/s | torch GB/s | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 1024 | 14 µs | 7 µs | 19 | 37 | 0.5× |
| 256 | 4096 | 15 µs | 8 µs | 574 | 1017 | 0.6× |
| 1024 | 8192 | 30 µs | 42 µs | 2214 | 1602 | **1.4×** |
| 2048 | 8192 | 54 µs | 82 µs | **2480** | 1643 | **1.5×** |

Small shapes lose to `F.layer_norm` because of the pyptx Torch dispatch
overhead (~14 µs vs torch's ~7 µs). Large shapes win because the
kernel's memory pattern is closer to peak.

### Fused SwiGLU (f32)

| M | F | pyptx | torch | pyptx GB/s | torch GB/s | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 1024 | 14 µs | 8 µs | 29 | 48 | 0.6× |
| 256 | 4096 | 14 µs | 8 µs | 887 | 1543 | 0.6× |
| 1024 | 8192 | 38 µs | 62 µs | 2665 | 1633 | **1.6×** |
| 2048 | 8192 | 72 µs | 117 µs | **2795** | 1724 | **1.6×** |

2795 GB/s is **94% of H100 HBM3 peak**. The kernel is bandwidth-bound.

---

## Flash attention (bf16)

Q-tile parallel, `BN=64` multi-k WGMMA for both Q@K^T and P@V,
`head_dim` up to 64. Compared against naive PyTorch (f32 matmul +
softmax) and `torch.nn.functional.scaled_dot_product_attention` (fused
FA kernel).

| M | N | HD | pyptx | naive | SDPA | vs naive | vs SDPA |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  512 |  512 | 64 | 19 µs | 43 µs | 15 µs | **2.2×** | 0.73× |
| 2048 | 2048 | 64 | 52 µs | 88 µs | 15 µs | **1.7×** | 0.29× |
| 4096 | 4096 | 64 | 88 µs | 262 µs | 25 µs | **3.0×** | 0.29× |
| 8192 | 4096 | 64 | 88 µs | 470 µs | 40 µs | **5.3×** | 0.46× |

Beats the naive reference by 2-5×. Torch SDPA is faster — it's a
production FA3 kernel with warp specialization and 3-stage pipelining.
Closing the gap is ongoing work.

---

## Launch overhead

Per-launch dispatch cost for a trivial kernel at `shape=(4, 64)`:

| Path | Overhead |
| --- | --- |
| PyTorch eager, pyptx via C++ extension | ~14 µs |
| PyTorch eager, pyptx via ctypes (no C++ ext) | ~34 µs |
| PyTorch native (`torch.add` reference) | ~7 µs |
| CUDA graph replay | ~4 µs |

Install `ninja` so the C++ torch extension builds — it drops launch
overhead from 34 µs to 14 µs by avoiding per-call Python→ctypes
marshalling:

```bash
pip install ninja
```

Inside a CUDA graph, the steady-state cost drops to 4 µs on
repeated-shape calls ("turbo graph replay" mode).

---

## Environment

- H100 SXM5 80 GB, CUDA 12.8, NVIDIA driver 580.126.09
- PyTorch 2.8, JAX 0.10, cuda-python 13.2
- Compiled with `ptxas` at release optimization level
- `--pyptx-shim` C++ extension built via `ninja`
- Torch custom_op fast path (`torch.library.custom_op`)
