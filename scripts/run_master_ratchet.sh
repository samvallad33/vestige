#!/usr/bin/env bash
set -euo pipefail

STEP_NAME="${1:?Usage: bash scripts/run_master_ratchet.sh <step_name>}"
MODE="${2:-core}"

echo "=== [1/4] HARDCODE SCAN (${STEP_NAME}) ==="
bash scripts/hardcode_scan.sh

echo "=== [2/4] CLIPPY & RETRIEVAL UNIT TESTS (${STEP_NAME}) ==="
cargo clippy -p vestige-core --all-targets -- -D warnings
cargo test -p vestige-core --lib retrieval:: --no-fail-fast

echo "=== [3/4] BUILD BENCH-FAST BINARY (${STEP_NAME}) ==="
cargo build --profile bench-fast -p vestige-core --bin retrieval_bench

echo "=== [4/4] SEQUENTIAL BENCHMARK RATCHET (${STEP_NAME}, mode=${MODE}) ==="
# 1. LoCoMo-10 FIRST (verify all 5 categories: Single-Hop, Multi-Hop, Open-Domain, Temporal, Adversarial)
node benchmarks/run_stage1_retrieval.mjs --only=locomo --json="benchmarks/results/${STEP_NAME}_locomo.json"
# 2. Core Suites
node benchmarks/run_recall_suite.mjs --bench=beam --split=100k --json="benchmarks/results/${STEP_NAME}_beam100k.json"
python benchmarks/eval_inmind.py
python benchmarks/eval_statebench.py
python benchmarks/eval_convomem.py --max-per-category 30
python benchmarks/eval_longmemeval_temporal.py

if [ "${MODE}" = "full" ]; then
  node benchmarks/run_stage1_retrieval.mjs --only=lme --json="benchmarks/results/${STEP_NAME}_lme_s.json"
  python benchmarks/eval_ruler_multikey_multivalue.py
  python benchmarks/eval_cora_bench.py
  node benchmarks/run_recall_suite.mjs --bench=hotpotqa
  node benchmarks/run_recall_suite.mjs --bench=memoryagentbench
fi
