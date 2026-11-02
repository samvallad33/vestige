#!/bin/sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -x "/opt/homebrew/bin/python3.14" ]; then
    VENV="/opt/homebrew/bin/python3.14"
elif [ -f "/Users/entity002/Documents/Vestige/work/benchmark-campaigns-20261031/upstream/harbor/.venv/bin/python" ]; then
    VENV="/Users/entity002/Documents/Vestige/work/benchmark-campaigns-20261031/upstream/harbor/.venv/bin/python"
else
    VENV="python3"
fi

exec "$VENV" "$SCRIPT_DIR/run_nano_showdown.py" "$@"
