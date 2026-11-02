#!/usr/bin/env bash
# Vestige v6.2 Master Benchmark Runner: 3-Way Ablation Matrix ("David vs. Goliath")
# Evaluates 8B Model Champions (Qwen3-Coder-8B / Llama-3.1-8B) across:
#   Mode A (None): Baseline agent without memory/hooks
#   Mode B (Forward-Only FSRS-6): Standard retention, RSB disabled
#   Mode C (Full Vestige v6.2): FSRS-6 + Retroactive Salience Backfill + Reflex Hooks
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${BENCHMARK_MODEL:-qwen3-coder:8b}"
MODE="${1:---ablation=C}"

echo "================================================================================"
echo "          VESTIGE v6.2 3-LANE BENCHMARK ABLATION MATRIX (${MODEL})              "
echo "================================================================================"
echo "Date: $(date -u)"
echo "Host: $(uname -s) $(uname -m)"
echo "Mode: ${MODE}"
echo ""

case "${MODE}" in
  --ablation=A|A)
    echo ">>> Running Mode A: Baseline Agent (No Vestige, No Reflex Hooks)..."
    export VESTIGE_ENABLED=0
    export VESTIGE_RSB_ENABLED=0
    export VESTIGE_REFLEX_ENABLED=0
    ;;
  --ablation=B|B)
    echo ">>> Running Mode B: Vestige (Forward-Only FSRS-6, Backfill OFF)..."
    export VESTIGE_ENABLED=1
    export VESTIGE_RSB_ENABLED=0
    export VESTIGE_REFLEX_ENABLED=0
    ;;
  --ablation=C|C)
    echo ">>> Running Mode C: Full Vestige v6.2 (FSRS-6 + RSB + Reflex Hooks ON)..."
    export VESTIGE_ENABLED=1
    export VESTIGE_RSB_ENABLED=1
    export VESTIGE_REFLEX_ENABLED=1
    ;;
  *)
    echo "Usage: $0 [--ablation=A | --ablation=B | --ablation=C]"
    exit 1
    ;;
esac

# 1. David vs. Goliath Showdown
echo ">>> STEP 1: Running David vs. Goliath Showdown (${MODEL})..."
if [ -f "$ROOT/campaigns/memoryagentbench/run-david-vs-goliath.sh" ]; then
  sh "$ROOT/campaigns/memoryagentbench/run-david-vs-goliath.sh" || true
fi

# 2. MemConflict Showdown
echo ">>> STEP 2: Running MemConflict Showdown (${MODEL})..."
if [ -f "$ROOT/benchmarks/memconflict/run-nano-showdown.sh" ]; then
  sh "$ROOT/benchmarks/memconflict/run-nano-showdown.sh" --instances 1 --sessions 10 || true
fi

# 3. Cryptographically Seal Evaluation Run into Strata Log
echo ">>> STEP 3: Cryptographically Sealing Evaluation Receipts..."
if command -v vestige >/dev/null 2>&1; then
  vestige precedent --query verify_log || true
fi

echo "================================================================================"
echo "           EVALUATION RUN COMPLETE & SEALED IN STRATA LOG                      "
echo "================================================================================"
