#!/usr/bin/env bash
#
# run-zep.sh -- the ZEP / GRAPHITI arm of the multi-agent flagship demo.
#
# Runs the SAME fleet on the SAME torture bug, but each agent has a temporal-
# knowledge-graph retrieval tool (zep_search) over the SAME seeded corpus
# (_rag_load_facts). Graphiti (Zep's OSS engine) extracts entities + edges on
# ingest (one local ollama LLM call per fact) into a local FalkorDB graph, then
# retrieves via hybrid semantic + BM25 + graph RRF. Fully local, NO ZEP_API_KEY.
# Emits results/zep.json.
#
# Nothing is scripted to win. If a required service is missing this ERRORS.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TORTURE_REPO:-$HERE/../torture-v3.5}"
OUT="$HERE/results/zep.json"
# graphiti-core lives in the arms venv (Python 3.12); prefer it.
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
export ZEP_FALKOR_HOST="${ZEP_FALKOR_HOST:-localhost}"
export ZEP_FALKOR_PORT="${ZEP_FALKOR_PORT:-6379}"
# gemma3:12b builds a clean Graphiti graph; llama3.1:8b (8B) emits malformed
# edges -> empty retrieval. Graphiti needs a capable extraction model.
# qwen2.5-graphiti = qwen2.5:14b + num_ctx 8192 (Modelfile). NOT gemma3:12b:
# Zep's own docs require a model with real structured-output support, gemma3 has
# no native function-calling and produced malformed graphs (garbage predicates,
# orphan edges, dead retrieval). The num_ctx bump is the actual fix -- Ollama
# defaults num_ctx to 2048 and silently truncates the FRONT of long prompts,
# cutting Graphiti's schema+rules off the top so the model never sees them.
# NOTE: this export OVERRIDES runner.py's default, so it must be kept in sync.
export ZEP_LLM_MODEL="${ZEP_LLM_MODEL:-qwen2.5-graphiti}"
export ZEP_EMBED_MODEL="${ZEP_EMBED_MODEL:-nomic-embed-text}"

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

# Service check: FalkorDB must be listening (Graphiti's graph store). Fail loud.
if ! (exec 3<>"/dev/tcp/${ZEP_FALKOR_HOST}/${ZEP_FALKOR_PORT}") 2>/dev/null; then
  echo "ERROR: FalkorDB not reachable at ${ZEP_FALKOR_HOST}:${ZEP_FALKOR_PORT}." >&2
  echo "       Start it: docker run -d --name falkordb-bench -p 6379:6379 -p 3001:3000 falkordb/falkordb:latest" >&2
  exit 2
fi
# Ollama must be up (Graphiti extraction LLM + embeddings).
if ! curl -sf "http://localhost:11434/api/tags" >/dev/null 2>&1; then
  echo "ERROR: Ollama not reachable at http://localhost:11434 (needed for Graphiti extraction + embeddings)." >&2
  echo "       Start ollama and pull: ollama pull llama3.1:8b && ollama pull nomic-embed-text" >&2
  exit 2
fi

mkdir -p "$HERE/results"

echo "== ZEP fleet run (Graphiti temporal knowledge graph, local FalkorDB) =="
echo "repo:       $REPO"
echo "provider:   $PROVIDER"
echo "python:     $PY"
echo "fleet size: ${FLEET_SIZE:-3}"
echo "vestige:    $VESTIGE_BIN  (shared corpus: $VESTIGE_DATA_DIR)"
echo "falkordb:   ${ZEP_FALKOR_HOST}:${ZEP_FALKOR_PORT}   LLM: $ZEP_LLM_MODEL   embed: $ZEP_EMBED_MODEL"
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

exec "$PY" "$HERE/agent/fleet_runner.py" --mode zep --repo "$REPO" --out "$OUT"
