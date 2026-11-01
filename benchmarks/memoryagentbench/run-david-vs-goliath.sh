#!/bin/sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$ROOT:$ROOT/upstream/agentdojo-library/src:$ROOT/upstream/harbor/src"
export PYTHONHASHSEED=0

"$ROOT/upstream/harbor/.venv/bin/python" "$SCRIPT_DIR/run_david_vs_goliath.py" "$@"
