#!/usr/bin/env bash
#
# run-hindsight.sh -- the HINDSIGHT (vectorize-io) arm of the multi-agent flagship.
#
# Runs the SAME fleet on the SAME torture bug, but each agent retrieves through a
# self-hosted Hindsight agent-memory server (hindsight-api) over the SAME seeded
# corpus every arm gets (_rag_load_facts, exported from the SAME Vestige DB, never
# re-authored). Hindsight runs its OWN recommended path: retain() (LLM extraction
# on ingest, forced to ollama) + recall() (semantic + BM25 + local cross-encoder
# rerank). Verbatim extraction so stored text == corpus body. Fully local
# (embedded Postgres, local embedder + reranker). Emits results/hindsight.json.
#
# Nothing is scripted to win. If a required key/service is missing this ERRORS.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TORTURE_REPO:-$HERE/../torture-v3.5}"
OUT="$HERE/results/hindsight.json"
# hindsight-client lives in the arms venv (Python 3.12); prefer it.
if [ -z "${PYTHON:-}" ] && [ -x "$HERE/.venv-arms/bin/python" ]; then
  PY="$HERE/.venv-arms/bin/python"
else
  PY="${PYTHON:-python3}"
fi

if [ -z "${VESTIGE_BIN:-}" ]; then
  if [ -x "$HOME/vestige/target/release/vestige" ]; then
    VESTIGE_BIN="$HOME/vestige/target/release/vestige"
  elif command -v vestige >/dev/null 2>&1; then
    VESTIGE_BIN="$(command -v vestige)"
  else
    VESTIGE_BIN="vestige"
  fi
fi
export VESTIGE_BIN
export VESTIGE_DATA_DIR="${VESTIGE_DATA_DIR:-$REPO/.vestige-demo-db}"
# hindsight-api is started on :8899 (default :8888 is often taken on this box).
# 127.0.0.1, NOT localhost: hindsight-api (uvicorn) binds IPv4 only, while macOS
# resolves "localhost" to ::1 FIRST -- so anything holding the IPv6 wildcard on
# this port intercepts every call. That actually happened (a stray
# `python -m http.server 8899` answered HTTP 501 to every hindsight request while
# the real service was healthy on IPv4). Pinning the literal IPv4 address removes
# the whole class. NOTE: this export OVERRIDES runner.py's default -- keep in sync.
export HINDSIGHT_BASE_URL="${HINDSIGHT_BASE_URL:-http://127.0.0.1:8899}"

PROVIDER="${PROVIDER:-anthropic}"

# Provider-aware key guard. Never fabricate.
if [ "$PROVIDER" = "mock" ]; then
  : # $0 scripted provider for wiring verification -- no key needed.
elif [ "$PROVIDER" = "deepseek" ]; then
  [ -n "${DEEPSEEK_API_KEY:-}" ] || { echo "ERROR: PROVIDER=deepseek but DEEPSEEK_API_KEY unset." >&2; exit 2; }
elif [ "$PROVIDER" = "openai" ]; then
  [ -n "${OPENAI_API_KEY:-}" ] || { echo "ERROR: PROVIDER=openai but OPENAI_API_KEY unset." >&2; exit 2; }
elif [ "$PROVIDER" = "openrouter" ]; then
  [ -n "${OPENROUTER_API_KEY:-}" ] || { echo "ERROR: PROVIDER=openrouter but OPENROUTER_API_KEY unset." >&2; exit 2; }
elif [ "$PROVIDER" = "moonshot" ]; then
  [ -n "${MOONSHOT_API_KEY:-}" ] || { echo "ERROR: PROVIDER=moonshot but MOONSHOT_API_KEY unset." >&2; exit 2; }
elif [ "$PROVIDER" = "anthropic" ]; then
  if [ -z "${ANTHROPIC_API_KEY:-}" ] && ! (command -v ant >/dev/null 2>&1 && ant auth status >/dev/null 2>&1); then
    echo "ERROR: ANTHROPIC_API_KEY unset and no 'ant' auth profile active." >&2; exit 2
  fi
else
  echo "ERROR: unknown PROVIDER '$PROVIDER'. Use anthropic|openai|openrouter|moonshot|deepseek|mock." >&2; exit 2
fi

if ! command -v "$VESTIGE_BIN" >/dev/null 2>&1 && [ ! -x "$VESTIGE_BIN" ]; then
  echo "ERROR: vestige CLI not found at '$VESTIGE_BIN'. Set VESTIGE_BIN." >&2; exit 2
fi
if [ ! -d "$REPO" ]; then
  echo "ERROR: torture repo not found at $REPO (set TORTURE_REPO)." >&2; exit 2
fi

# Service check: the Hindsight server must be up (separate process the client does
# NOT start). Fail loud rather than let every agent's tool call error.
if ! curl -sf "$HINDSIGHT_BASE_URL/version" >/dev/null 2>&1; then
  echo "ERROR: Hindsight server not reachable at $HINDSIGHT_BASE_URL." >&2
  echo "       Start it once (fully local, ollama extraction) -- see" >&2
  echo "       arm-services/hindsight/ or overnight-logs/GLM-MULTIARM-READY.md." >&2
  exit 2
fi
# Ollama must be up (retain() extraction LLM). recall() uses local embedder +
# reranker (no key), but ingest needs the LLM unless provider=none.
if ! curl -sf "http://localhost:11434/api/tags" >/dev/null 2>&1; then
  echo "ERROR: Ollama not reachable at http://localhost:11434 (needed for retain() extraction)." >&2
  echo "       Start ollama and pull a tool-capable model: ollama pull llama3.1:8b" >&2
  exit 2
fi

mkdir -p "$HERE/results"

echo "== HINDSIGHT fleet run (agent-memory arm: retain + recall, hybrid rerank) =="
echo "repo:       $REPO"
echo "provider:   $PROVIDER"
echo "python:     $PY"
echo "fleet size: ${FLEET_SIZE:-3}"
echo "vestige:    $VESTIGE_BIN  (shared corpus: $VESTIGE_DATA_DIR)"
echo "hindsight:  $HINDSIGHT_BASE_URL"
echo "out:        $OUT"
echo

SEED="$REPO/.vestige-seed.sh"
if [ -f "$SEED" ]; then
  echo "seeding shared corpus via repo canonical seed: $SEED"
  VESTIGE_BIN="$VESTIGE_BIN" VESTIGE_DATA_DIR="$VESTIGE_DATA_DIR" bash "$SEED" >/dev/null
  echo "seed complete ($VESTIGE_DATA_DIR)"
  echo
else
  echo "ERROR: repo seed script not found at $SEED." >&2; exit 2
fi

if [ ! -d "$REPO/node_modules" ]; then
  echo "installing repo deps (npm install)..."
  (cd "$REPO" && npm install >/dev/null 2>&1) || { echo "ERROR: npm install failed in $REPO" >&2; exit 2; }
fi
TORTURE_REPO="$REPO" bash "$HERE/reset-repo.sh"
echo

exec "$PY" "$HERE/agent/fleet_runner.py" --mode hindsight --repo "$REPO" --out "$OUT"
