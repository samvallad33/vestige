#!/usr/bin/env bash
#
# run-mem0.sh -- the MEM0 arm of the multi-agent flagship demo.
#
# Runs the SAME fleet of N agents on the SAME torture bug, but each agent has a
# mem0 retrieval tool (mem0_search) over the SAME shared corpus every arm gets
# (runner._rag_load_facts -- the seeded Vestige DB, PRODUCTION OUTAGE excluded).
# mem0 uses its OWN RECOMMENDED path: infer=True LLM fact-extraction on ingest
# (llama3.1:8b) + semantic retrieval (nomic-embed-text) + Chroma on-disk. It is
# the fair "isn't Vestige just a memory layer?" competitor -- same corpus, same
# top-K budget (ARM_TOPK), same 6000-char cap, same output shape as the RAG arm;
# only mem0's own extraction+store+retrieval differ. Emits results/mem0.json.
#
# FULLY LOCAL: no cloud key for the memory layer (ollama does extraction +
# embeddings; Chroma is embedded). The only key needed is for the AGENT MODEL
# (PROVIDER). PROVIDER=mock runs the whole thing at $0 for wiring verification.
#
# Nothing is scripted to win. If a required key/binary is missing this ERRORS.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TORTURE_REPO:-$HERE/../torture-v3.5}"
OUT="$HERE/results/mem0.json"
# mem0 needs the Python 3.12 venv with mem0ai installed (system python3 is 3.14,
# unsupported by the memory-system ecosystem). Prefer the arms venv.
if [ -z "${PYTHON:-}" ] && [ -x "$HERE/.venv-arms/bin/python" ]; then
  PY="$HERE/.venv-arms/bin/python"
else
  PY="${PYTHON:-python3}"
fi

# Resolve the vestige CLI (used by _rag_load_facts to export the shared corpus).
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

# mem0 knobs (defaults match the audited arm; all overridable).
export MEM0_LLM_MODEL="${MEM0_LLM_MODEL:-llama3.1:8b}"
export MEM0_EMBED_MODEL="${MEM0_EMBED_MODEL:-nomic-embed-text:latest}"
export MEM0_TELEMETRY="${MEM0_TELEMETRY:-false}"
export ANONYMIZED_TELEMETRY="${ANONYMIZED_TELEMETRY:-false}"
# Fresh, isolated Chroma store per run so a stale collection can't leak.
# mem0_qdrant, NOT mem0_chroma: the arm was switched to qdrant, which is mem0's
# OWN vendor default and the only one of the two that supports mem0's hybrid BM25
# retrieval (on chroma, mem0 itself warns "Hybrid (BM25) scoring will be
# disabled"). The two backends' on-disk formats are NOT interchangeable, so a
# fresh path also avoids loading a stale chroma dir as if it were qdrant.
# NOTE: this export OVERRIDES runner.py's default -- keep the two in sync.
export MEM0_STORE_DIR="${MEM0_STORE_DIR:-$REPO/.arm_store/mem0_qdrant}"

PROVIDER="${PROVIDER:-anthropic}"

# Provider-aware key guard. Never fabricate.
if [ "$PROVIDER" = "mock" ]; then
  : # $0 scripted provider -- no key needed.
elif [ "$PROVIDER" = "openai" ]; then
  [ -n "${OPENAI_API_KEY:-}" ] || { echo "ERROR: PROVIDER=openai but OPENAI_API_KEY unset." >&2; exit 2; }
elif [ "$PROVIDER" = "openrouter" ]; then
  [ -n "${OPENROUTER_API_KEY:-}" ] || { echo "ERROR: PROVIDER=openrouter but OPENROUTER_API_KEY unset." >&2; exit 2; }
elif [ "$PROVIDER" = "moonshot" ]; then
  [ -n "${MOONSHOT_API_KEY:-}" ] || { echo "ERROR: PROVIDER=moonshot but MOONSHOT_API_KEY unset." >&2; exit 2; }
elif [ "$PROVIDER" = "deepseek" ]; then
  [ -n "${DEEPSEEK_API_KEY:-}" ] || { echo "ERROR: PROVIDER=deepseek but DEEPSEEK_API_KEY unset." >&2; exit 2; }
elif [ "$PROVIDER" = "anthropic" ]; then
  if [ -z "${ANTHROPIC_API_KEY:-}" ] && ! (command -v ant >/dev/null 2>&1 && ant auth status >/dev/null 2>&1); then
    echo "ERROR: ANTHROPIC_API_KEY unset and no 'ant' auth profile active." >&2; exit 2
  fi
else
  echo "ERROR: unknown PROVIDER '$PROVIDER'. Use anthropic|openai|openrouter|moonshot|deepseek|mock." >&2
  exit 2
fi

if ! command -v "$VESTIGE_BIN" >/dev/null 2>&1 && [ ! -x "$VESTIGE_BIN" ]; then
  echo "ERROR: vestige CLI not found at '$VESTIGE_BIN'. Set VESTIGE_BIN." >&2; exit 2
fi
if [ ! -d "$REPO" ]; then
  echo "ERROR: torture repo not found at $REPO (set TORTURE_REPO)." >&2; exit 2
fi

mkdir -p "$HERE/results"

echo "== MEM0 fleet run (mem0ai OSS: infer=True LLM-extraction + semantic retrieval) =="
echo "repo:       $REPO"
echo "provider:   $PROVIDER"
echo "mem0 LLM:   $MEM0_LLM_MODEL   embedder: $MEM0_EMBED_MODEL"
echo "python:     $PY"
echo "fleet size: ${FLEET_SIZE:-3}"
echo "vestige:    $VESTIGE_BIN  (shared corpus: $VESTIGE_DATA_DIR)"
echo "out:        $OUT"
echo

# Seed the shared corpus via the repo's own canonical seed (same for every arm).
SEED="$REPO/.vestige-seed.sh"
if [ -f "$SEED" ]; then
  echo "seeding shared corpus via repo canonical seed: $SEED"
  VESTIGE_BIN="$VESTIGE_BIN" VESTIGE_DATA_DIR="$VESTIGE_DATA_DIR" bash "$SEED" >/dev/null
  echo "seed complete ($VESTIGE_DATA_DIR)"
  echo
else
  echo "ERROR: repo seed script not found at $SEED." >&2; exit 2
fi

# Wipe any stale mem0 store so ingest starts clean each run.
rm -rf "$MEM0_STORE_DIR"

if [ ! -d "$REPO/node_modules" ]; then
  echo "installing repo deps (npm install)..."
  (cd "$REPO" && npm install >/dev/null 2>&1) || { echo "ERROR: npm install failed in $REPO" >&2; exit 2; }
fi
TORTURE_REPO="$REPO" bash "$HERE/reset-repo.sh"
echo

exec "$PY" "$HERE/agent/fleet_runner.py" --mode mem0 --repo "$REPO" --out "$OUT"
