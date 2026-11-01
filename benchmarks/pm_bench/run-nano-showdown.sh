#!/usr/bin/env bash
# PM-Bench Showdown Runner: Vestige Intention Memory vs. In-Context Baseline on gpt-5.4-nano
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -f "$REPO_ROOT/upstream/harbor/.venv/bin/python" ]; then
        PYTHON_BIN="$REPO_ROOT/upstream/harbor/.venv/bin/python"
    elif [ -f "$REPO_ROOT/.venv/bin/python" ]; then
        PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

echo "================================================================================"
echo "          PM-BENCH: PROSPECTIVE MEMORY SHOWDOWN (gpt-5.4-nano)                  "
echo "================================================================================"
echo "Repository: $REPO_ROOT"
echo "Runner:     $SCRIPT_DIR/run_nano_showdown.py"
echo "Python:     $PYTHON_BIN"
echo ""

if [ "$#" -eq 0 ]; then
    set -- --days 2
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/run_nano_showdown.py" "$@"
