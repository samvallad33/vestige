#!/usr/bin/env bash
# PM-Bench Showdown Runner: Vestige Intention Memory vs. In-Context Baseline on gpt-5.4-nano
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="/Users/entity002/Documents/Vestige/work/benchmark-campaigns-20261031/upstream/harbor/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "================================================================================"
echo "          PM-BENCH: PROSPECTIVE MEMORY SHOWDOWN (gpt-5.4-nano)                  "
echo "================================================================================"
echo "Repository: $REPO_ROOT"
echo "Runner:     $SCRIPT_DIR/run_nano_showdown.py"
echo "Python:     $PYTHON_BIN"
echo ""

DAYS="${1:---days 2}"

"$PYTHON_BIN" "$SCRIPT_DIR/run_nano_showdown.py" $DAYS
