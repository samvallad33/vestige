#!/bin/sh
# Staging and execution wrapper for GPT-6 Astra across the Terminal-Bench suite.
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$task_root"
export PYTHONPATH="$task_root:$task_root/upstream/harbor/src:$task_root/upstream/agentdojo-library/src"
exec "$task_root/upstream/harbor/.venv/bin/python" -m campaigns.terminal_bench.astra_suite "$@"
