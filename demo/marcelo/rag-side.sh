#!/usr/bin/env bash
# Standard vector search baseline (not Vestige). Offline after setup.sh.
set -euo pipefail

die() {
  printf 'stopped: %s\n' "$1" >&2
  exit 1
}

DEMO_HOME="${DEMO_HOME:-$HOME/vestige-demo}"
[[ -d "$DEMO_HOME" ]] || die "run demo/marcelo/setup.sh first"
DEMO_HOME="$(cd "$DEMO_HOME" && pwd)"
[[ -f "$DEMO_HOME/.vestige-demo" ]] || die "run demo/marcelo/setup.sh first"
[[ -f "$DEMO_HOME/ingested-commits.txt" ]] || die "run demo/marcelo/vestige-side.sh first"
[[ -f "$DEMO_HOME/causal-walk-commits.txt" ]] || die "run demo/marcelo/vestige-side.sh first"
[[ -x "$DEMO_HOME/rag-venv/bin/python" ]] || die "run demo/marcelo/setup.sh first"

# The corpus is the commit list the Vestige side wrote. DEMO_HISTORY only
# checks that this checkout holds that much history.
if [[ -n "${DEMO_HISTORY:-}" ]]; then
  case "$DEMO_HISTORY" in
    [1-9]|[1-9][0-9]|[1-9][0-9][0-9]) ;;
    *) die "DEMO_HISTORY must be a positive integer" ;;
  esac
  if [[ "$DEMO_HISTORY" -gt 500 ]]; then
    die "DEMO_HISTORY is above 500"
  fi
  FIX_SHA="b52d48973fe9ddb2e78b663ec48a1a68f7e7802d"
  HAVE="$(git -C "$DEMO_HOME/uv" rev-list --count "$FIX_SHA")"
  if [[ "$HAVE" -lt $((DEMO_HISTORY + 1)) ]]; then
    die "the uv checkout does not hold this much history. Run setup.sh with the same DEMO_HISTORY."
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1
export HF_HOME="$DEMO_HOME/hf-cache"
export HUGGINGFACE_HUB_CACHE="$DEMO_HOME/hf-cache/hub"
export SENTENCE_TRANSFORMERS_HOME="$DEMO_HOME/hf-cache/sentence-transformers"
export GIT_NO_LAZY_FETCH=1
export GIT_TERMINAL_PROMPT=0
export CARGO_NET_OFFLINE=true
export PYTHONUNBUFFERED=1

exec "$DEMO_HOME/rag-venv/bin/python" "$SCRIPT_DIR/rag_baseline/search.py" \
  "$DEMO_HOME/uv" \
  "$DEMO_HOME/ingested-commits.txt" \
  "$DEMO_HOME/failure.txt" \
  "$DEMO_HOME/cause-sha.txt" \
  "$DEMO_HOME/ingested-count.txt" \
  "$DEMO_HOME/causal-walk-commits.txt"
