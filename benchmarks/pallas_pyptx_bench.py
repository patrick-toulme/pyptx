"""Benchmark: Pallas Mosaic GPU kernel PTX → pyptx round-trip → benchmark.

Pallas on GPU uses the Mosaic GPU backend (default since JAX ~0.5). Setting
MOSAIC_GPU_DUMP_TO causes JAX to write compiled artifacts (PTX, SASS, etc.)
to disk. We grab that PTX, round-trip it through pyptx, and benchmark.

This demonstrates pyptx can ingest kernels from the JAX Pallas ecosystem,
not just hand-written PTX or Triton output.

Usage (on H100):
    python benchmarks/pallas_pyptx_bench.py

Environment variables (set automatically if not present):
    MOSAIC_GPU_DUMP_TO   - directory for compiled PTX artifacts
    MOSAIC_GPU_DUMP_PTX  - also dump PTX to stdout (set to "1")
"""

import os
import sys
import time

# Set PTX dump env vars BEFORE importing JAX so Mosaic GPU picks them up.
DUMP_DIR = os.environ.get("MOSAIC_GPU_DUMP_TO", "/tmp/pallas_ptx")
os.makedirs(DUMP_DIR, exist_ok=True)
os.environ.setdefault("MOSAIC_GPU_DUMP_TO", DUMP_DIR)
os.environ.setdefault("MOSAIC_GPU_DUMP_PTX", "1")
# Triton fallback (for legacy Pallas backend)
os.environ.setdefault("TRITON_KERNEL_DUMP", "1")
os.environ.setdefault("TRITON_DUMP_DIR", DUMP_DIR)

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyptx.parser import parse
from pyptx.emitter import emit
from pyptx.codegen.codegen import ptx_to_python

# ---------------------------------------------------------------------------
# GPU init
# ---------------------------------------------------------------------------
_ = (jnp.ones(4) + 1).block_until_ready()
devices = jax.devices()
print(f"JAX {jax.__version__}, devices: {devices}")

try:
    from jax.experimental import pallas as pl
except ImportError:
    print("ERROR: jax.experimental.pallas not available")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Define and run a Pallas matmul
# ---------------------------------------------------------------------------

M = N = K = 2048
BM = BN = 128

a = jnp.array(np.random.randn(M, K).astype(np.float32), dtype=jnp.bfloat16)
b = jnp.array(np.random.randn(K, N).astype(np.float32), dtype=jnp.bfloat16)


def pallas_matmul_kernel(a_ref, b_ref, c_ref):
    c_ref[...] = jnp.dot(
        a_ref[...].astype(jnp.float32),
        b_ref[...].astype(jnp.float32),
    ).astype(jnp.bfloat16)


def run_pallas_matmul(a, b):
    return pl.pallas_call(
        pallas_matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        grid=(M // BM, N // BN),
        in_specs=[
            pl.BlockSpec((BM, K), lambda i, j: (i, 0)),
            pl.BlockSpec((K, BN), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
    )(a, b)


print("\n--- Running Pallas matmul ---")
try:
    pallas_fn = jax.jit(run_pallas_matmul)
    c_pallas = pallas_fn(a, b)
    c_pallas.block_until_ready()
    ref = jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32)).astype(jnp.bfloat16)
    ref.block_until_ready()
    diff = float(jnp.abs(c_pallas.astype(jnp.float32) - ref.astype(jnp.float32)).max())
    print(f"Pallas matmul diff from cuBLAS: {diff:g}")
except Exception as e:
    print(f"Pallas matmul failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ---------------------------------------------------------------------------
# 2. Find the compiled PTX
# ---------------------------------------------------------------------------

print(f"\n--- Extracting PTX (dump dir: {DUMP_DIR}) ---")


def find_ptx_files(*dirs):
    """Walk directories for .ptx files, return sorted by size (largest first)."""
    results = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(".ptx"):
                    full = os.path.join(root, f)
                    size = os.path.getsize(full)
                    results.append((size, full))
    results.sort(reverse=True)
    return results


# Check the Mosaic GPU dump dir first, then Triton cache as fallback
search_dirs = [DUMP_DIR]
triton_cache = os.path.expanduser("~/.triton/cache")
if os.path.isdir(triton_cache):
    search_dirs.append(triton_cache)

ptx_files = find_ptx_files(*search_dirs)
ptx_source = None

if ptx_files:
    print(f"Found {len(ptx_files)} PTX file(s):")
    for size, path in ptx_files[:5]:
        print(f"  {path}: {size} bytes")

    # Use the largest PTX file (most likely the matmul kernel)
    _, best_path = ptx_files[0]
    with open(best_path) as f:
        ptx_source = f.read()
    print(f"\nUsing: {best_path} ({len(ptx_source)} bytes, "
          f"{len(ptx_source.splitlines())} lines)")
else:
    print("No PTX files found in dump directories.")
    print("Mosaic GPU may embed PTX differently in your JAX version.")
    print(f"Checked: {search_dirs}")


# ---------------------------------------------------------------------------
# 3. Round-trip through pyptx
# ---------------------------------------------------------------------------

if ptx_source is not None:
    print("\n--- pyptx Round-Trip ---")
    try:
        t0 = time.perf_counter()
        module = parse(ptx_source)
        parse_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        emitted = emit(module)
        emit_ms = (time.perf_counter() - t0) * 1000

        roundtrip_ok = (emitted == ptx_source)
        print(f"Parse:      {parse_ms:.1f}ms")
        print(f"Emit:       {emit_ms:.1f}ms")
        print(f"Round-trip: {'BYTE-IDENTICAL' if roundtrip_ok else 'DIFF'}")

        if not roundtrip_ok:
            orig = ptx_source.splitlines()
            emit_lines = emitted.splitlines()
            for i, (a_line, b_line) in enumerate(zip(orig, emit_lines)):
                if a_line != b_line:
                    print(f"  First diff at line {i+1}:")
                    print(f"    orig: {a_line!r}")
                    print(f"    emit: {b_line!r}")
                    break
            if len(orig) != len(emit_lines):
                print(f"  Lines: orig={len(orig)}, emit={len(emit_lines)}")

        # Save round-tripped PTX
        with open("/tmp/pallas_gemm.ptx", "w") as f:
            f.write(ptx_source)
        print("Saved PTX to: /tmp/pallas_gemm.ptx")

    except Exception as e:
        print(f"Round-trip failed: {e}")
        import traceback; traceback.print_exc()

    # Transpile to pyptx DSL
    print("\n--- Transpile to pyptx DSL ---")
    try:
        t0 = time.perf_counter()
        dsl_code = ptx_to_python(ptx_source)
        transpile_ms = (time.perf_counter() - t0) * 1000
        print(f"Transpile:  {transpile_ms:.1f}ms")
        print(f"DSL code:   {len(dsl_code)} bytes, {len(dsl_code.splitlines())} lines")

        dsl_path = "/tmp/pallas_gemm_pyptx.py"
        with open(dsl_path, "w") as f:
            f.write(dsl_code)
        print(f"Saved to:   {dsl_path}")

        # Preview
        print("\n--- DSL Preview (first 25 lines) ---")
        for i, line in enumerate(dsl_code.splitlines()[:25], 1):
            print(f"  {i:3d}  {line}")
        print("  ...")
    except Exception as e:
        print(f"Transpile failed: {e}")
        import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# 4. Benchmark: Pallas vs cuBLAS
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"BENCHMARK: Pallas vs cuBLAS ({M}x{N}x{K} bf16)")
print("=" * 60)

WARMUP = 20
ITERS = 200
flops = 2 * M * N * K

# Pallas
for _ in range(WARMUP):
    pallas_fn(a, b).block_until_ready()
t0 = time.perf_counter()
for _ in range(ITERS):
    pallas_fn(a, b).block_until_ready()
t_pallas = (time.perf_counter() - t0) / ITERS

# cuBLAS (jnp.matmul)
matmul_fn = jax.jit(jnp.matmul)
for _ in range(WARMUP):
    matmul_fn(a, b).block_until_ready()
t0 = time.perf_counter()
for _ in range(ITERS):
    matmul_fn(a, b).block_until_ready()
t_cublas = (time.perf_counter() - t0) / ITERS

print(f"\n{'Method':<25} {'Time (ms)':>10} {'TFLOPS':>10} {'vs cuBLAS':>10}")
print("-" * 60)
print(f"{'cuBLAS (jnp.matmul)':<25} {t_cublas*1000:10.3f} "
      f"{flops/t_cublas/1e12:10.0f} {'1.00x':>10}")
print(f"{'Pallas (Mosaic GPU)':<25} {t_pallas*1000:10.3f} "
      f"{flops/t_pallas/1e12:10.0f} {t_pallas/t_cublas:10.2f}x")

print("\nDONE")
