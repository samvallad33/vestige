#!/bin/sh
# Shell wrapper for the 3-Way Showdown (bare, rag, vestige) on Terminal-Bench.
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$task_root"
export PYTHONPATH="$task_root:$task_root/upstream/harbor/src:$task_root/upstream/agentdojo-library/src"
exec "$task_root/upstream/harbor/.venv/bin/python" -m campaigns.terminal_bench.run_showdown "$@"
