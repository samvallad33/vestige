#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
runs="$repo/runs"
benchmarks="locomo,longmemeval,mpbench,memconflict,pmbench,memoryagentbench,memform,terminal_bench,graphwalks,automationbench,vals_finance_v2,cwe_bench,agentdojo,injecagent"
mode="run"
while (($#)); do
  case "$1" in
    --runs) runs="$2"; shift 2 ;;
    --benchmarks) benchmarks="$2"; shift 2 ;;
    --prepare-only) mode="prepare"; shift ;;
    --report-only) mode="report"; shift ;;
    *) printf 'Unsupported argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
mkdir -p "$runs"

if [[ "$mode" == "report" ]]; then
  python3 -m benchmarks.real_engine.reporting --repo "$repo" --runs "$runs"
  exit 0
fi

# Keep model execution separate from compilation. This entry point never starts
# a local LLM. Python3.13 is needed for the official AutomationBench package.
if [[ ! -x "$repo/.venv/bin/python" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.13 "$repo/.venv"
  else
    python3.13 -m venv "$repo/.venv"
  fi
fi
python="$repo/.venv/bin/python"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$python" -r benchmarks/real_engine/requirements.lock
else
  "$python" -m pip install -r benchmarks/real_engine/requirements.lock
fi
"$python" -m benchmarks.real_engine.reporting --repo "$repo" --runs "$runs" --prepare

# AgentDojo is imported from its exact official checkout, never an unpinned
# released version. All other source/scorer adapters load their pinned files.
if [[ -d "$runs/upstream/agentdojo" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$python" --no-deps -e "$runs/upstream/agentdojo"
  else
    "$python" -m pip install --no-deps -e "$runs/upstream/agentdojo"
  fi
fi

if ! cargo build --release --locked -p vestige-mcp --bin vestige-mcp >"$runs/build.log" 2>&1; then
  tail -n 40 "$runs/build.log" >&2
  exit 1
fi
"$python" -m benchmarks.real_engine.reporting --repo "$repo" --runs "$runs" --write-build-manifest

"$python" -m benchmarks.real_engine.reporting --repo "$repo" --runs "$runs" --freeze-terminal-catalog

"$python" -m benchmarks.real_engine.reporting --repo "$repo" --runs "$runs"
if [[ "$mode" == "prepare" ]]; then
  exit 0
fi

if [[ ! -f "$runs/.env.openai" ]]; then
  printf '%s\n' 'Missing owner-only runs/.env.openai credential file; preparation completed with no API requests.' >&2
  exit 1
fi

# One campaign lock, one event loop, one global API semaphore and spend ledger.
# Every per-benchmark reproduce.sh delegates here; simultaneous invocations are
# refused by run.py rather than creating independent40-request budgets.
"$python" -m benchmarks.real_engine.run --runs "$runs" --benchmarks "$benchmarks"
"$python" -m benchmarks.real_engine.reporting --repo "$repo" --runs "$runs"
