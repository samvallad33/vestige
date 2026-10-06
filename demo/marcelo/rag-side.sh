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
[[ -x "$DEMO_HOME/rag-venv/bin/python" ]] || die "run demo/marcelo/setup.sh first"

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
  "$DEMO_HOME/ingested-count.txt"
