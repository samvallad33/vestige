#!/usr/bin/env bash
#
# run-rag.sh -- the DENSE-RAG arm of the multi-agent flagship demo.
#
# Runs the SAME fleet of N agents on the SAME torture-v2 bug, but this time each
# agent has an INDEPENDENT dense-RAG retrieval tool (rag_search) over the SAME
# seeded Vestige DB. There is NO shared bus and NO coordination -- this is the
# anarchy control PLUS pure top-K cosine retrieval. Each agent embeds its own
# query and retrieves the most similar past memories by similarity alone (no
# causal join, no vestige_log). It isolates what plain semantic retrieval buys
# versus Vestige's causal backfill (run-sync.sh). Emits MEASURED fleet numbers to
# results/rag.json.
#
# Nothing is scripted to win. If a required key/binary is missing this ERRORS.
#
# Required env:
#   PROVIDER=anthropic (default) -> ANTHROPIC_API_KEY (or `ant auth login`)
#   PROVIDER=openai              -> OPENAI_API_KEY (GPT-5.6 Sol)
#   PROVIDER=openrouter          -> OPENROUTER_API_KEY (moonshotai/kimi-k2.7-code)
#   VESTIGE_BIN                  -> path to the vestige CLI (auto-resolved below)
# Optional env (see README.md):
#   FLEET_SIZE (default 3), MODEL, COST_PER_MTOK_INPUT/OUTPUT, MAX_ITERATIONS,
#   MAX_TOKENS, TEST_CMD, TORTURE_REPO, VESTIGE_DATA_DIR, PYTHON
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TORTURE_REPO:-$HERE/../torture-v2}"
OUT="$HERE/results/rag.json"
PY="${PYTHON:-python3}"

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

# The seeded memory DB rag_search retrieves over. Reuse the repo's isolated demo
# dir so the fleet retrieves over the same seeded history the demo narrates.
export VESTIGE_DATA_DIR="${VESTIGE_DATA_DIR:-$REPO/.vestige-demo-db}"

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

mkdir -p "$HERE/results"

echo "== RAG fleet run (dense-RAG arm: independent top-K cosine retrieval) =="
echo "repo:       $REPO"
echo "provider:   $PROVIDER"
echo "model:      ${MODEL:-$([ "$PROVIDER" = openai ] && echo gpt-5.6-sol || { [ "$PROVIDER" = openrouter ] && echo moonshotai/kimi-k2.7-code || echo claude-opus-4-8; })}"
echo "fleet size: ${FLEET_SIZE:-3}"
echo "vestige:    $VESTIGE_BIN  (rag index: $VESTIGE_DATA_DIR)"
echo "out:        $OUT"
echo

# Seed the retrievable memory with the real project history via the repo's own
# canonical seed (cause + noise + lookalike distractor + failure). rag_search
# retrieves over this exact seeded DB by pure cosine similarity.
SEED="$REPO/.vestige-seed.sh"
if [ -f "$SEED" ]; then
  echo "seeding rag index via repo canonical seed: $SEED"
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

exec "$PY" "$HERE/agent/fleet_runner.py" --mode rag --repo "$REPO" --out "$OUT"
