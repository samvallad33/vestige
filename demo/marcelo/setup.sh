#!/usr/bin/env bash
# One-time preparation for the recording. Clone, build, fetch uv, download the
# baseline model. The live commands stay on local files after this prints ready.
set -euo pipefail

unset GIT_NO_LAZY_FETCH
unset CARGO_NET_OFFLINE

die() {
  printf 'stopped: %s\n' "$1" >&2
  exit 1
}

command -v git >/dev/null 2>&1 || die "git is not on PATH"
command -v cargo >/dev/null 2>&1 || die "cargo is not on PATH"
command -v python3 >/dev/null 2>&1 || die "python3 is not on PATH"

DEMO_HOME="${DEMO_HOME:-$HOME/vestige-demo}"
mkdir -p "$DEMO_HOME"
DEMO_HOME="$(cd "$DEMO_HOME" && pwd)"
case "$DEMO_HOME" in
  /|/usr|/usr/*|/bin|/bin/*|/etc|/etc/*|/System|/System/*)
    die "refusing DEMO_HOME=$DEMO_HOME"
    ;;
esac

VESTIGE="$DEMO_HOME/vestige"
UV="$DEMO_HOME/uv"
VENV="$DEMO_HOME/rag-venv"
EXPECTED="b4bcd52b81402722d6c99b96f461278a171d110d"
FIX_SHA="b52d48973fe9ddb2e78b663ec48a1a68f7e7802d"
FAILURE_SHA="351d602d86c484a39bc537f1eb99866ea2c25fc1"
CAUSE_SHA="d2f58d92991fa08b24596fcc6c6472dc5015d3bc"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export HF_HOME="$DEMO_HOME/hf-cache"
export HUGGINGFACE_HUB_CACHE="$DEMO_HOME/hf-cache/hub"
export SENTENCE_TRANSFORMERS_HOME="$DEMO_HOME/hf-cache/sentence-transformers"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HF_HOME"

printf 'work directory: %s\n' "$DEMO_HOME"

if [[ ! -d "$VESTIGE/.git" ]]; then
  printf 'Cloning Vestige and checking out b4bcd52b81.\n'
  git clone --filter=blob:none https://github.com/samvallad33/vestige.git "$VESTIGE"
fi
HEAD_FULL="$(git -C "$VESTIGE" rev-parse HEAD)"
if [[ "$HEAD_FULL" != "$EXPECTED" ]]; then
  printf 'Checking out b4bcd52b81.\n'
  git -C "$VESTIGE" fetch --filter=blob:none origin "$EXPECTED"
  git -C "$VESTIGE" checkout --detach "$EXPECTED"
  HEAD_FULL="$(git -C "$VESTIGE" rev-parse HEAD)"
fi
[[ "$HEAD_FULL" == "$EXPECTED" ]] || die "checkout is $HEAD_FULL"
printf 'This is pre-release code from PR #445 (commit b4bcd52b81), not the v4.1.1 release.\n'
printf 'HEAD %s\n' "$HEAD_FULL"
if git -C "$VESTIGE" remote get-url origin >/dev/null 2>&1; then
  git -C "$VESTIGE" remote remove origin
fi

if [[ ! -x "$VESTIGE/target/debug/vestige" || ! -x "$VESTIGE/target/debug/vestige-mcp" ]]; then
  printf 'Building the default-feature binaries.\n'
  (cd "$VESTIGE" && cargo build -p vestige-mcp)
else
  printf 'Binaries are already present.\n'
fi
[[ -x "$VESTIGE/target/debug/vestige" && -x "$VESTIGE/target/debug/vestige-mcp" ]] || die "binaries were not produced"

if ! git -C "$UV" cat-file -e "${CAUSE_SHA}^{commit}" >/dev/null 2>&1; then
  printf 'Fetching the uv checkout.\n'
  if [[ ! -d "$UV/.git" ]]; then
    git init "$UV"
  fi
  if ! git -C "$UV" remote get-url origin >/dev/null 2>&1; then
    git -C "$UV" remote add origin https://github.com/astral-sh/uv.git
  fi
  git -C "$UV" fetch --depth=40 origin "$FIX_SHA"
  git -C "$UV" checkout --detach FETCH_HEAD
fi
git -C "$UV" cat-file -e "${FIX_SHA}^{commit}" >/dev/null 2>&1 || die "uv is missing the named revision"
git -C "$UV" cat-file -e "${FAILURE_SHA}^{commit}" >/dev/null 2>&1 || die "uv is missing the observed revision"
git -C "$UV" cat-file -e "${CAUSE_SHA}^{commit}" >/dev/null 2>&1 || die "uv is missing the recorded cause"
UV_HEAD="$(git -C "$UV" rev-parse HEAD)"
printf 'uv HEAD %s\n' "$UV_HEAD"
if git -C "$UV" remote get-url origin >/dev/null 2>&1; then
  git -C "$UV" remote remove origin
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  if ! python3 -m venv "$VENV" >"$DEMO_HOME/venv-create.log" 2>&1; then
    die "python3 -m venv failed. On Debian, install the python3-venv package. On macOS, use a python.org or Homebrew Python."
  fi
fi
"$VENV/bin/python" -c 'import pip' >/dev/null 2>&1 || die "the Python environment has no pip"
printf 'Preparing the Python environment and downloading the baseline model.\n'
if [[ "$(uname -s)" == "Darwin" ]]; then
  "$VENV/bin/python" -m pip install --disable-pip-version-check -r "$SCRIPT_DIR/rag_baseline/requirements.txt"
else
  "$VENV/bin/python" -m pip install --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cpu
  "$VENV/bin/python" -m pip install --disable-pip-version-check -r "$SCRIPT_DIR/rag_baseline/requirements.txt"
fi
env -u TRANSFORMERS_OFFLINE -u HF_DATASETS_OFFLINE HF_HUB_OFFLINE=0 \
  "$VENV/bin/python" -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("all-MiniLM-L6-v2")'

printf '%s\n' "$DEMO_HOME" > "$DEMO_HOME/.vestige-demo"
printf 'ready\n'
