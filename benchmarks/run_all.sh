#!/usr/bin/env bash
# Vestige v6.2.1 Benchmark Runner: 100% reproducible and honest.
# One command runs data verification, builds native Rust binary, executes baselines and full evaluations.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "================================================================================"
echo "  VESTIGE v6.2.1 REPRODUCIBLE BENCHMARK RUNNER"
echo "================================================================================"

# 1. Fetch & verify public datasets
bash benchmarks/fetch_data.sh

# 2. Build native Rust retrieval benchmark binary
echo "=== Building Rust retrieval_bench binary ==="
cargo build --release -p vestige-core --bin retrieval_bench

# 3. Create results directory
RESULTS_DIR="benchmarks/results/v6.2.1"
mkdir -p "${RESULTS_DIR}"

# 4. Run Python BM25 baseline (rank_bm25)
echo "=== Running Python rank_bm25 baseline ==="
PYTHON_CMD="python"
if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
fi
${PYTHON_CMD} benchmarks/baseline_bm25.py --json="${RESULTS_DIR}/baseline_python_bm25.json"

# 5. Run Rust kernel in baseline mode (all algorithmic extensions disabled, idf_power=1.0)
echo "=== Running Rust kernel in baseline mode ==="
node benchmarks/run_stage1_retrieval.mjs --baseline --json="${RESULTS_DIR}/baseline_rust_kernel.json"

# 6. Run full Vestige v6.2.1 engine (official metric, all 500, held-out, ablations)
echo "=== Running Vestige v6.2.1 full benchmark (with held-out and ablations) ==="
node benchmarks/run_stage1_retrieval.mjs --held-out --ablations --json="${RESULTS_DIR}/vestige_v6.2.1_full.json"

echo "================================================================================"
echo "  ALL BENCHMARKS COMPLETED SUCCESSFULLY"
echo "  Results saved to ${RESULTS_DIR}"
echo "================================================================================"
