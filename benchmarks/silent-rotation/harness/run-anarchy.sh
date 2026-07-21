#!/usr/bin/env bash
#
# run-anarchy.sh -- PHASE 1 of the multi-agent flagship demo (the CONTROL group).
#
# Unleashes a FLEET of N coding agents (default 3) on the SAME torture-v2 bug
# with NO shared memory. Each works in its own checkout and independently guesses
# a fix DIRECTION for the direction-ambiguous canonical-digest drift. Their edits
# are then integrated into one shared tree with a real merge: divergent
# directions collide and the integrated repo fails. Emits the MEASURED fleet
# tokens/cost/wall-clock/conflict-count/final status to results/anarchy.json.
#
# Nothing is scripted to fail. If the required key is missing this ERRORS.
#
# Required env (by provider):
#   PROVIDER=anthropic (default) -> ANTHROPIC_API_KEY (or an `ant auth login` profile)
#   PROVIDER=openai              -> OPENAI_API_KEY (GPT-5.6 Sol)
# Optional env (see README.md):
#   FLEET_SIZE (default 3), MODEL, COST_PER_MTOK_INPUT/OUTPUT, MAX_ITERATIONS,
#   MAX_TOKENS, TEST_CMD, TORTURE_REPO, PYTHON
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TORTURE_REPO:-$HERE/../torture-v2}"
OUT="$HERE/results/anarchy.json"
PY="${PYTHON:-python3}"
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

if [ ! -d "$REPO" ]; then
  echo "ERROR: torture repo not found at $REPO (set TORTURE_REPO)." >&2
  exit 2
fi

mkdir -p "$HERE/results"

echo "== ANARCHY fleet run (Phase 1: control, NO shared memory) =="
echo "repo:       $REPO"
echo "provider:   $PROVIDER"
echo "model:      ${MODEL:-$([ "$PROVIDER" = openai ] && echo gpt-5.6-sol || { [ "$PROVIDER" = openrouter ] && echo moonshotai/kimi-k2.7-code || echo claude-opus-4-8; })}"
echo "fleet size: ${FLEET_SIZE:-3}"
echo "out:        $OUT"
echo

# Ensure deps are present once (the fleet symlinks node_modules per checkout),
# then reset the base repo to its pristine BROKEN state + refresh the snapshot.
if [ ! -d "$REPO/node_modules" ]; then
  echo "installing repo deps (npm install)..."
  (cd "$REPO" && npm install >/dev/null 2>&1) || {
    echo "ERROR: npm install failed in $REPO" >&2; exit 2; }
fi
TORTURE_REPO="$REPO" bash "$HERE/reset-repo.sh"
echo

exec "$PY" "$HERE/agent/fleet_runner.py" --mode anarchy --repo "$REPO" --out "$OUT"
