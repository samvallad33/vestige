#!/usr/bin/env bash
# Runs frozen params on LoCoMo tune (convs 1-5) and test (convs 6-10) splits.
# Writes benchmarks/results/v6.2.1/locomo_heldout.json with CORE session R@5, exact turn R@5,
# soft turn R@5, and per-category scores for each split.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p benchmarks/results/v6.2.1

echo "================================================================================"
echo "  LOCOMO-10 HELD-OUT EVALUATION (CONVS 1-5 TUNE vs CONVS 6-10 TEST)"
echo "================================================================================"

# 1. Ensure datasets and Rust binary are ready
bash benchmarks/fetch_data.sh
cargo build --release -p vestige-core --bin retrieval_bench

# 2. Run Tune split (conversations 1-5)
echo "=== Running Tune Split (Conversations 1-5) ==="
node benchmarks/run_stage1_retrieval.mjs --only=locomo --split=tune --json=benchmarks/results/v6.2.1/locomo_tune.json

# 3. Run Test split (conversations 6-10, frozen params)
echo "=== Running Test Split (Conversations 6-10) ==="
node benchmarks/run_stage1_retrieval.mjs --only=locomo --split=test --json=benchmarks/results/v6.2.1/locomo_test.json

# 4. Generate combined benchmarks/results/v6.2.1/locomo_heldout.json
node -e '
import { readFileSync, writeFileSync } from "node:fs";
const tune = JSON.parse(readFileSync("benchmarks/results/v6.2.1/locomo_tune.json", "utf8")).locomo;
const test = JSON.parse(readFileSync("benchmarks/results/v6.2.1/locomo_test.json", "utf8")).locomo;

const heldout = {
  description: "LoCoMo-10 Held-Out Evaluation: Current frozen params evaluated on Tune (dialogues 1-5) vs Test (dialogues 6-10)",
  note: "All parameters were frozen before scoring the test split.",
  tune: {
    dialogues: tune.dialogues,
    core: {
      scored_questions: tune.core.n,
      session_r5: tune.core.session_r5,
      turn_exact_r5: tune.core.turn_exact_r5,
      turn_soft_r5: tune.core.turn_soft_r5
    },
    total: {
      scored_questions: tune.total.n,
      session_r5: tune.total.session_r5,
      turn_exact_r5: tune.total.turn_exact_r5,
      turn_soft_r5: tune.total.turn_soft_r5
    },
    categories: tune.categories
  },
  test: {
    dialogues: test.dialogues,
    core: {
      scored_questions: test.core.n,
      session_r5: test.core.session_r5,
      turn_exact_r5: test.core.turn_exact_r5,
      turn_soft_r5: test.core.turn_soft_r5
    },
    total: {
      scored_questions: test.total.n,
      session_r5: test.total.session_r5,
      turn_exact_r5: test.total.turn_exact_r5,
      turn_soft_r5: test.total.turn_soft_r5
    },
    categories: test.categories
  }
};

writeFileSync("benchmarks/results/v6.2.1/locomo_heldout.json", JSON.stringify(heldout, null, 2));
console.log("\nHeld-out results successfully written to benchmarks/results/v6.2.1/locomo_heldout.json");
'
