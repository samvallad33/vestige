#!/usr/bin/env bash
#
# run-sync-noedge.sh -- ABLATION ARM 2 (PREREGISTRATION.md): the hand-authored
# causal edge, isolated.
#
# Byte-identical to run-sync.sh at the tool level (event anchor + shared write
# bus). The ONLY change is the seed: this arm seeds from .vestige-seed-noedge.sh,
# in which the cause memory's shared `active_key` tag is STRIPPED, so the causal
# edge backfill traverses does not exist. If this arm still reaches >=80% of
# sync's converged-correct rate, the edge is not carrying the result and that
# claim is deleted (decision table committed in PREREGISTRATION.md).
#
# Emits MEASURED fleet numbers to results/sync-noedge.json.
# Nothing is scripted to win. If a required key/binary is missing this ERRORS.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TORTURE_REPO:-$HERE/../torture-v2}"
OUT="$HERE/results/sync-noedge.json"
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

echo "== SYNC-NOEDGE fleet run (ablation arm 2: causal edge stripped from seed) =="
echo "repo:       $REPO"
echo "provider:   $PROVIDER"
echo "fleet size: ${FLEET_SIZE:-3}"
echo "vestige:    $VESTIGE_BIN  (shared bus: $VESTIGE_DATA_DIR)"
echo "out:        $OUT"
echo

# THE ablation delta: seed from the noedge variant. Everything else identical.
SEED="$REPO/.vestige-seed-noedge.sh"
if [ -f "$SEED" ]; then
  echo "seeding shared bus via NOEDGE seed variant: $SEED"
  VESTIGE_BIN="$VESTIGE_BIN" VESTIGE_DATA_DIR="$VESTIGE_DATA_DIR" bash "$SEED" >/dev/null
  echo "seed complete ($VESTIGE_DATA_DIR) -- cause ingested WITHOUT the shared active_key tag"
  echo
else
  echo "ERROR: noedge seed script not found at $SEED (re-run prepare_trial)." >&2; exit 2
fi

if [ ! -d "$REPO/node_modules" ]; then
  echo "installing repo deps (npm install)..."
  (cd "$REPO" && npm install >/dev/null 2>&1) || { echo "ERROR: npm install failed in $REPO" >&2; exit 2; }
fi
TORTURE_REPO="$REPO" bash "$HERE/reset-repo.sh"
echo

exec "$PY" "$HERE/agent/fleet_runner.py" --mode sync-noedge --repo "$REPO" --out "$OUT"
