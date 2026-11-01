#!/bin/sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -f "$REPO/upstream/harbor/.venv/bin/python" ]; then
        PYTHON_BIN="$REPO/upstream/harbor/.venv/bin/python"
    elif [ -f "$REPO/.venv/bin/python" ]; then
        PYTHON_BIN="$REPO/.venv/bin/python"
    elif [ -f "/Users/entity002/Documents/Vestige/work/benchmark-campaigns-20261031/upstream/harbor/.venv/bin/python" ]; then
        PYTHON_BIN="/Users/entity002/Documents/Vestige/work/benchmark-campaigns-20261031/upstream/harbor/.venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/run_nano_showdown.py" "$@"
