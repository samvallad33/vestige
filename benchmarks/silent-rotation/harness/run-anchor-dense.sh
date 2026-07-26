#!/usr/bin/env bash
#
# run-anchor-dense.sh -- ABLATION ARM 3 (PREREGISTRATION.md): event-anchoring
# alone, on the competitor retrieval stack.
#
# No Vestige backfill, no causal edge, no shared bus. Each agent gets ONE tool,
# anchor_dense_search, which takes NO query: it embeds the failure event's own
# text and returns the top-K cosine-similar memories from the SAME shared
# corpus rag_search retrieves over. The only difference from the rag arm is
# that no agent ever types a query, so per-agent query variance is removed
# while everything Vestige-specific is absent.
#
# "anchor-dense is the one I most want to see. If it matches sync, the finding
# belongs to nobody and is larger for it." -- PREREGISTRATION.md
#
# Emits MEASURED fleet numbers to results/anchor-dense.json.
# Nothing is scripted to win. If a required key/binary is missing this ERRORS.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TORTURE_REPO:-$HERE/../torture-v2}"
OUT="$HERE/results/anchor-dense.json"
PY="${PYTHON:-python3}"

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

PROVIDER="${PROVIDER:-anthropic}"
if [ "$PROVIDER" = "mock" ]; then
  :
elif [ "$PROVIDER" = "deepseek" ]; then
  [ -n "${DEEPSEEK_API_KEY:-}" ] || { echo "ERROR: PROVIDER=deepseek but DEEPSEEK_API_KEY is not set. Refusing to fabricate numbers." >&2; exit 2; }
elif [ "$PROVIDER" = "openai" ]; then
  [ -n "${OPENAI_API_KEY:-}" ] || { echo "ERROR: PROVIDER=openai but OPENAI_API_KEY is not set. Refusing to fabricate numbers." >&2; exit 2; }
elif [ "$PROVIDER" = "openrouter" ]; then
  [ -n "${OPENROUTER_API_KEY:-}" ] || { echo "ERROR: PROVIDER=openrouter but OPENROUTER_API_KEY is not set. Refusing to fabricate numbers." >&2; exit 2; }
elif [ "$PROVIDER" = "moonshot" ]; then
  [ -n "${MOONSHOT_API_KEY:-}" ] || { echo "ERROR: PROVIDER=moonshot but MOONSHOT_API_KEY is not set. Refusing to fabricate numbers." >&2; exit 2; }
elif [ "$PROVIDER" = "anthropic" ]; then
  if [ -z "${ANTHROPIC_API_KEY:-}" ] && ! (command -v ant >/dev/null 2>&1 && ant auth status >/dev/null 2>&1); then
    echo "ERROR: ANTHROPIC_API_KEY is not set and no 'ant' auth profile is active. Refusing to fabricate numbers." >&2; exit 2
  fi
else
  echo "ERROR: unknown PROVIDER '$PROVIDER'." >&2; exit 2
fi

if ! command -v "$VESTIGE_BIN" >/dev/null 2>&1 && [ ! -x "$VESTIGE_BIN" ]; then
  echo "ERROR: vestige CLI not found at '$VESTIGE_BIN'. Set VESTIGE_BIN." >&2; exit 2
fi
[ -d "$REPO" ] || { echo "ERROR: torture repo not found at $REPO (set TORTURE_REPO)." >&2; exit 2; }

mkdir -p "$HERE/results"

echo "== ANCHOR-DENSE fleet run (ablation arm 3: event anchor on cosine stack) =="
echo "repo:       $REPO"
echo "provider:   $PROVIDER"
echo "fleet size: ${FLEET_SIZE:-3}"
echo "vestige:    $VESTIGE_BIN  (corpus source: $VESTIGE_DATA_DIR)"
echo "out:        $OUT"
echo

SEED="$REPO/.vestige-seed.sh"
if [ -f "$SEED" ]; then
  echo "seeding corpus via repo canonical seed: $SEED"
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

exec "$PY" "$HERE/agent/fleet_runner.py" --mode anchor-dense --repo "$REPO" --out "$OUT"
