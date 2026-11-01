#!/bin/sh
# Vestige Instant Benchmark Runner
# Usage:
#   export OPENAI_API_KEY="your-api-key"
#   sh benchmarks/run.sh [memform|memconflict|pm_bench|all]
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================================================================"
echo "                   VESTIGE INSTANT BENCHMARK RUNNER                             "
echo "================================================================================"

# Check Python
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

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python 3 not found. Please install Python 3."
    exit 1
fi

# Verify openai package is present
if ! "$PYTHON_BIN" -c "import openai" >/dev/null 2>&1; then
    echo "Notice: 'openai' package not detected in $PYTHON_BIN."
    echo "Installing openai..."
    "$PYTHON_BIN" -m pip install openai -q || {
        echo "Error: Failed to install openai. Run: pip install openai"
        exit 1
    }
fi

# Verify API key
if [ -z "${OPENAI_API_KEY:-}" ]; then
    if [ -f "$REPO_ROOT/benchmarks/.env.openai" ]; then
        export OPENAI_API_KEY="$(cat "$REPO_ROOT/benchmarks/.env.openai" | tr -d '\r\n ')"
    elif [ -f "$HOME/.config/vestige/.env.openai" ]; then
        export OPENAI_API_KEY="$(cat "$HOME/.config/vestige/.env.openai" | tr -d '\r\n ')"
    elif [ -f "$HOME/.env.openai" ]; then
        export OPENAI_API_KEY="$(cat "$HOME/.env.openai" | tr -d '\r\n ')"
    else
        echo "Error: OPENAI_API_KEY is not set."
        echo ""
        echo "Please export your API key before running:"
        echo "  export OPENAI_API_KEY=\"sk-...\""
        echo "  sh benchmarks/run.sh"
        exit 1
    fi
fi

TARGET="${1:-memform}"

case "$TARGET" in
    memform)
        echo ">>> Running Track 1: MemForm Showdown (Formation Poisoning Immunity)..."
        exec sh "$SCRIPT_DIR/memform/run-nano-showdown.sh"
        ;;
    memconflict)
        echo ">>> Running Track 2: MemConflict Showdown (Authority Inversion & Causal DAG)..."
        exec sh "$SCRIPT_DIR/memconflict/run-nano-showdown.sh"
        ;;
    pm_bench|pm)
        echo ">>> Running Track 3: PM-Bench Showdown (Prospective Memory Virtual Week)..."
        exec sh "$SCRIPT_DIR/pm_bench/run-nano-showdown.sh" --days 1
        ;;
    all)
        echo ">>> Running Complete Master Suite across all 3 tracks..."
        echo ""
        echo "[1/3] MemForm Showdown:"
        sh "$SCRIPT_DIR/memform/run-nano-showdown.sh"
        echo ""
        echo "[2/3] MemConflict Showdown:"
        sh "$SCRIPT_DIR/memconflict/run-nano-showdown.sh"
        echo ""
        echo "[3/3] PM-Bench Showdown:"
        sh "$SCRIPT_DIR/pm_bench/run-nano-showdown.sh" --days 1
        echo ""
        echo "================================================================================"
        echo "              ALL SHOWDOWNS COMPLETE. RAW JSON IN benchmarks/                   "
        echo "================================================================================"
        ;;
    *)
        echo "Usage: sh benchmarks/run.sh [memform|memconflict|pm_bench|all]"
        exit 1
        ;;
esac
