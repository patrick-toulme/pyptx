#!/bin/bash
# Run all pyptx benchmarks on H100.
#
# Usage:
#   ssh into H100, cd to pyptx repo, then:
#   bash benchmarks/run_all.sh
#
# Prerequisites:
#   pip install -e .
#   pip install triton torch  # (or jax[cuda])

set -e

echo "=========================================="
echo " pyptx benchmark suite"
echo "=========================================="
echo ""

# Ensure TRITON_CACHE_DIR is set so we can find compiled PTX
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_pyptx}"
mkdir -p "$TRITON_CACHE_DIR"

echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo ""

# 1. Triton matmul PTX → pyptx round-trip → benchmark
echo "=========================================="
echo " Triton matmul → pyptx"
echo "=========================================="
python benchmarks/triton_pyptx_bench.py 2>&1 | tee /tmp/bench_triton.log
echo ""

# 2. Pallas matmul PTX → pyptx round-trip → benchmark
echo "=========================================="
echo " Pallas matmul → pyptx"
echo "=========================================="
python benchmarks/pallas_pyptx_bench.py 2>&1 | tee /tmp/bench_pallas.log
echo ""

echo "=========================================="
echo " Results saved to /tmp/bench_*.log"
echo " Transpiled DSL saved to /tmp/*_pyptx.py"
echo "=========================================="
