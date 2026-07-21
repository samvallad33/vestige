#!/usr/bin/env bash
#
# run-supermemory.sh -- the SUPERMEMORY arm of the multi-agent flagship demo.
#
# Runs the SAME fleet of N agents on the SAME torture-v2 bug, but this time each
# agent retrieves through a self-hosted supermemory-server (supermemoryai/
# supermemory) over the SAME seeded corpus every arm gets (_rag_load_facts,
# exported from the SAME Vestige DB -- never re-authored). supermemory runs its
# OWN recommended pipeline: local ONNX bge-base-en-v1.5 embeddings + Ollama
# gpt-oss:20b extraction on ingest, then its RECOMMENDED document retrieval,
# POST /v3/search, top-K. Fully local (localhost:6767, .supermemory dir); nothing
# leaves the box. It isolates what a turnkey memory PRODUCT buys versus Vestige's
# causal backfill (run-sync.sh). Emits MEASURED fleet numbers to results/supermemory.json.
#
# Nothing is scripted to win. If a required key/binary/service is missing this ERRORS.
#
# Required env:
#   PROVIDER=anthropic (default) -> ANTHROPIC_API_KEY (or `ant auth login`)
#   PROVIDER=openai              -> OPENAI_API_KEY (GPT-5.6 Sol)
#   PROVIDER=openrouter          -> OPENROUTER_API_KEY (moonshotai/kimi-k2.7-code)
#   VESTIGE_BIN                  -> path to the vestige CLI (auto-resolved below)
#   SUPERMEMORY_API_KEY          -> the key printed by supermemory-server on first boot
# Optional env (see README.md):
#   FLEET_SIZE (default 3), MODEL, COST_PER_MTOK_INPUT/OUTPUT, MAX_ITERATIONS,
#   MAX_TOKENS, TEST_CMD, TORTURE_REPO, VESTIGE_DATA_DIR, PYTHON,
#   SUPERMEMORY_BASE_URL (default http://localhost:6767), SUPERMEMORY_CONTAINER,
#   OLLAMA_URL (used only for the reachability preflight below)
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TORTURE_REPO:-$HERE/../torture-v3.5}"
OUT="$HERE/results/supermemory.json"
# supermemory arm uses urllib (stdlib) -- system python3 is fine, but prefer the
# arms venv so it matches the other local-memory arms' interpreter.
if [ -z "${PYTHON:-}" ] && [ -x "$HERE/.venv-arms/bin/python" ]; then
  PY="$HERE/.venv-arms/bin/python"
else
  PY="${PYTHON:-python3}"
fi

# Resolve the vestige CLI to an absolute path (the seed subshell may not inherit
# an interactive PATH), preferring an explicit VESTIGE_BIN, then the release
# binary, then PATH.
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

# The seeded memory the supermemory arm re-ingests. Reuse the repo's isolated demo
# dir so the fleet retrieves over the same seeded history the demo narrates.
export VESTIGE_DATA_DIR="${VESTIGE_DATA_DIR:-$REPO/.vestige-demo-db}"

# supermemory service config (mirrors the arm's Python defaults).
export SUPERMEMORY_BASE_URL="${SUPERMEMORY_BASE_URL:-http://localhost:6767}"
# DO NOT set a default SUPERMEMORY_CONTAINER here. runner.py:627 defaults it to
# "benchmark-${CORRECT_KID}" so each trial gets its OWN container -- the same
# per-trial isolation every other arm gets. This line previously exported a
# FIXED "benchmark" container, which ALWAYS won (os.environ.get found a value)
# and silently disabled that isolation: supermemory then searched a store
# holding every prior trial's rotation runbook at once, each naming a different
# correct key. Verified in trial-1 transcripts, which returned merged OPS-522
# docs naming k_antares/k_sirius/k_borealis -- three trials' keys in one result.
# That is a handicap no other arm carried. Only export it to pin a container
# deliberately (e.g. debugging a single trial).
if [ -n "${SUPERMEMORY_CONTAINER:-}" ]; then
  export SUPERMEMORY_CONTAINER
fi

PROVIDER="${PROVIDER:-anthropic}"

# Provider-aware key guard. Never fabricate.
if [ "$PROVIDER" = "mock" ]; then
  : # $0 scripted provider for wiring verification -- no key needed.
elif [ "$PROVIDER" = "deepseek" ]; then
  if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "ERROR: PROVIDER=deepseek but DEEPSEEK_API_KEY is not set. Refusing to fabricate numbers." >&2
    echo "       Set DEEPSEEK_API_KEY (DeepSeek direct api.deepseek.com, for deepseek-v4-pro)." >&2
    exit 2
  fi
elif [ "$PROVIDER" = "openai" ]; then
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: PROVIDER=openai but OPENAI_API_KEY is not set." >&2
    echo "       Set OPENAI_API_KEY (for gpt-5.6-sol). Refusing to fabricate numbers." >&2
    exit 2
  fi
elif [ "$PROVIDER" = "openrouter" ]; then
  if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: PROVIDER=openrouter but OPENROUTER_API_KEY is not set. Refusing to fabricate numbers." >&2
    echo "       Set OPENROUTER_API_KEY (for moonshotai/kimi-k2.7-code)." >&2
    exit 2
  fi
elif [ "$PROVIDER" = "moonshot" ]; then
  if [ -z "${MOONSHOT_API_KEY:-}" ]; then
    echo "ERROR: PROVIDER=moonshot but MOONSHOT_API_KEY is not set. Refusing to fabricate numbers." >&2
    echo "       Set MOONSHOT_API_KEY (Moonshot direct api.moonshot.ai/v1, for kimi-k3)." >&2
    exit 2
  fi
elif [ "$PROVIDER" = "anthropic" ]; then
  if [ -z "${ANTHROPIC_API_KEY:-}" ] && ! (command -v ant >/dev/null 2>&1 && ant auth status >/dev/null 2>&1); then
    echo "ERROR: ANTHROPIC_API_KEY is not set and no 'ant' auth profile is active." >&2
    echo "       Set ANTHROPIC_API_KEY or run 'ant auth login'. Refusing to fabricate numbers." >&2
    exit 2
  fi
else
  echo "ERROR: unknown PROVIDER '$PROVIDER'. Use 'anthropic', 'openai', 'openrouter', or 'moonshot'." >&2
  exit 2
fi

if ! command -v "$VESTIGE_BIN" >/dev/null 2>&1 && [ ! -x "$VESTIGE_BIN" ]; then
  echo "ERROR: vestige CLI not found at '$VESTIGE_BIN'. Set VESTIGE_BIN." >&2
  exit 2
fi

if [ ! -d "$REPO" ]; then
  echo "ERROR: torture repo not found at $REPO (set TORTURE_REPO)." >&2
  exit 2
fi

# --- supermemory service preflight (fail loud, never fabricate) --------------
# The arm needs the supermemory-server reachable. When self-hosted locally the
# server AUTO-APPLIES its own key for unauthenticated localhost requests, so a
# blank SUPERMEMORY_API_KEY is fine locally; if a key IS set we send it. Probe
# /v3/documents both ways so either config passes. Also needs Ollama serving the
# extraction LLM. Check both BEFORE burning provider tokens.
# Probe /v3/health (returns 200 when the server is up). Do NOT probe
# /v3/documents?limit=1 -- that path 404s on this server version (documents are
# POSTed, not GET-listed), which made curl -f falsely report "not reachable" and
# skip the whole arm. The real ingest+search endpoints (POST /v3/documents,
# POST /v3/search) are exercised by the arm and verified working.
if ! curl -fsS -o /dev/null --max-time 5 "${SUPERMEMORY_BASE_URL}/v3/health" 2>/dev/null \
   && ! curl -fsS -o /dev/null --max-time 5 "${SUPERMEMORY_BASE_URL}/" 2>/dev/null; then
  echo "ERROR: supermemory-server not reachable at ${SUPERMEMORY_BASE_URL}." >&2
  echo "       Start it (fully local) with: SUPERMEMORY_EMBEDDING_PROVIDER=local \\" >&2
  echo "         SUPERMEMORY_LLM_PROVIDER=openai-compatible OPENAI_BASE_URL=http://localhost:11434/v1 \\" >&2
  echo "         OPENAI_API_KEY=ollama SUPERMEMORY_LLM_MODEL=llama3.1:8b PORT=6767 supermemory-server" >&2
  exit 2
fi
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434/api/embeddings}"
OLLAMA_ROOT="${OLLAMA_URL%/api/*}"
if ! curl -fsS -o /dev/null --max-time 5 "${OLLAMA_ROOT}/api/tags" 2>/dev/null; then
  echo "ERROR: Ollama not reachable at ${OLLAMA_ROOT} (needed for supermemory's extraction LLM)." >&2
  echo "       Start ollama and 'ollama pull llama3.1:8b'. Refusing to fabricate numbers." >&2
  exit 2
fi

mkdir -p "$HERE/results"

echo "== SUPERMEMORY fleet run (self-hosted supermemory-server, /v3/search) =="
echo "repo:        $REPO"
echo "provider:    $PROVIDER"
echo "model:       ${MODEL:-$([ "$PROVIDER" = openai ] && echo gpt-5.6-sol || { [ "$PROVIDER" = openrouter ] && echo moonshotai/kimi-k2.7-code || echo claude-opus-4-8; })}"
echo "fleet size:  ${FLEET_SIZE:-3}"
echo "vestige:     $VESTIGE_BIN  (seed corpus: $VESTIGE_DATA_DIR)"
# MUST use a default expansion: this script runs under `set -u`, and
# SUPERMEMORY_CONTAINER is now DELIBERATELY left unset so runner.py can pick the
# per-trial default (benchmark-${CORRECT_KID}) -- see the note above. Referencing
# it bare here killed the whole arm with "unbound variable" before it wrote any
# result, which the harness then recorded as fleet_verdict=errored.
echo "supermemory: $SUPERMEMORY_BASE_URL  (container: ${SUPERMEMORY_CONTAINER:-benchmark-${CORRECT_KID:-?} (per-trial default)})"
echo "out:         $OUT"
echo

# Seed the retrievable memory with the real project history via the repo's own
# canonical seed (cause + noise + lookalike distractor + failure). The supermemory
# arm re-ingests these EXACT facts (via _rag_load_facts) into its own local store.
SEED="$REPO/.vestige-seed.sh"
if [ -f "$SEED" ]; then
  echo "seeding source corpus via repo canonical seed: $SEED"
  VESTIGE_BIN="$VESTIGE_BIN" VESTIGE_DATA_DIR="$VESTIGE_DATA_DIR" bash "$SEED" >/dev/null
  echo "seed complete ($VESTIGE_DATA_DIR)"
  echo
else
  echo "ERROR: repo seed script not found at $SEED." >&2
  exit 2
fi

# Ensure deps present once, then reset base repo to pristine broken + snapshot.
if [ ! -d "$REPO/node_modules" ]; then
  echo "installing repo deps (npm install)..."
  (cd "$REPO" && npm install >/dev/null 2>&1) || {
    echo "ERROR: npm install failed in $REPO" >&2; exit 2; }
fi
TORTURE_REPO="$REPO" bash "$HERE/reset-repo.sh"
echo

exec "$PY" "$HERE/agent/fleet_runner.py" --mode supermemory --repo "$REPO" --out "$OUT"
