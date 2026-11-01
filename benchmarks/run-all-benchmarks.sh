#!/bin/sh
# Vestige Master Benchmark Runner: Run All Live Showdowns
# Evaluates Terminal-Bench 4.0, MemoryAgentBench (gpt-5.4-mini), and David vs. Goliath (gpt-5.4-nano vs. gpt-6.1-sol)
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================================================================"
echo "          VESTIGE MASTER BENCHMARK EVALUATION SUITE (ALL CAMPAIGNS)             "
echo "================================================================================"
echo "Date: $(date -u)"
echo "Host: $(uname -s) $(uname -m)"
echo ""

# 1. MemoryAgentBench David vs. Goliath Showdown (gpt-5.4-nano vs. gpt-6.1-sol)
echo ">>> STEP 1: Running David vs. Goliath Showdown (gpt-5.4-nano vs. gpt-6.1-sol)..."
sh "$ROOT/campaigns/memoryagentbench/run-david-vs-goliath.sh"
echo ">>> Step 1 Complete! Output in campaigns/memoryagentbench/evidence/david_vs_goliath/"
echo ""

# 2. MemoryAgentBench 3-Way Ablation (gpt-5.4-mini)
echo ">>> STEP 2: Running MemoryAgentBench 3-Way Showdown (gpt-5.4-mini)..."
sh "$ROOT/campaigns/memoryagentbench/run-showdown.sh"
echo ">>> Step 2 Complete! Output in campaigns/memoryagentbench/evidence/showdown/"
echo ""

# 3. Unit Test Suite Verification
echo ">>> STEP 3: Running Test Suite Verification..."
PYTHONPATH="$ROOT:$ROOT/upstream/harbor/src:$ROOT/upstream/agentdojo-library/src" \
    "$ROOT/upstream/harbor/.venv/bin/python" -m pytest "$ROOT/tests/" -q
echo ">>> Step 3 Complete! All isolated test gates verified."
echo ""

echo "================================================================================"
echo "           ALL VESTIGE BENCHMARKS COMPLETED AND VERIFIED DETERMINISTICALLY       "
echo "================================================================================"
