#!/usr/bin/env bash
# check-sandwich-prereqs.sh — Verify host can run the Vestige Cognitive Sandwich.
set -u

ok()   { printf '  \033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn() { printf '  \033[1;33m[WARN]\033[0m %s\n' "$*"; FAIL=1; }
miss() { printf '  \033[1;31m[MISS]\033[0m %s\n' "$*"; FAIL=1; }
info() { printf '  \033[1;36m[INFO]\033[0m %s\n' "$*"; }

load_vestige_sanhedrin_env() {
  [ -f "$1" ] || return 0
  command -v python3 >/dev/null 2>&1 || return 0
  while IFS="$(printf '\t')" read -r key value; do
    case "$key" in
      VESTIGE_SANHEDRIN_ENABLED|VESTIGE_SANHEDRIN_MODEL|VESTIGE_SANHEDRIN_ENDPOINT|VESTIGE_SANHEDRIN_BACKEND|VESTIGE_SANHEDRIN_CLAIM_MODE|VESTIGE_SANHEDRIN_OUTPUT|VESTIGE_SANHEDRIN_PYTHON|VESTIGE_DASHBOARD_PORT)
        export "$key=$value"
        ;;
    esac
  done < <(python3 - "$1" <<'PY'
import shlex
import sys

allowed = {
    "VESTIGE_SANHEDRIN_ENABLED",
    "VESTIGE_SANHEDRIN_MODEL",
    "VESTIGE_SANHEDRIN_ENDPOINT",
    "VESTIGE_SANHEDRIN_BACKEND",
    "VESTIGE_SANHEDRIN_CLAIM_MODE",
    "VESTIGE_SANHEDRIN_OUTPUT",
    "VESTIGE_SANHEDRIN_PYTHON",
    "VESTIGE_DASHBOARD_PORT",
}

try:
    lines = open(sys.argv[1], encoding="utf-8").read().splitlines()
except OSError:
    sys.exit(0)

for raw in lines:
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    try:
        parts = shlex.split(line, posix=True)
    except ValueError:
        continue
    if len(parts) != 1 or "=" not in parts[0]:
        continue
    key, value = parts[0].split("=", 1)
    if key in allowed and "\t" not in value and "\0" not in value:
        print(f"{key}\t{value}")
PY
  )
}

FAIL=0
CHECK_PREFLIGHT=0
CHECK_SANHEDRIN=0
DASHBOARD_PORT="${VESTIGE_DASHBOARD_PORT:-3927}"
SANHEDRIN_ENV="${VESTIGE_SANHEDRIN_ENV:-$HOME/.claude/hooks/vestige-sanhedrin.env}"

for arg in "$@"; do
  case "$arg" in
    --preflight|--enable-preflight) CHECK_PREFLIGHT=1 ;;
    --sanhedrin|--enable-sanhedrin) CHECK_SANHEDRIN=1 ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/check-sandwich-prereqs.sh [--preflight] [--sanhedrin]

Without flags, verifies that the default install has no Vestige hooks wired.
With --preflight, checks the optional UserPromptSubmit hook layer.
With --sanhedrin, checks the optional OpenAI-compatible verifier endpoint.
EOF
      exit 0
      ;;
  esac
done

if [ -f "$SANHEDRIN_ENV" ]; then
  load_vestige_sanhedrin_env "$SANHEDRIN_ENV" || true
fi

SANHEDRIN_ENDPOINT="${VESTIGE_SANHEDRIN_ENDPOINT:-${MLX_ENDPOINT:-}}"
SANHEDRIN_ENDPOINT="${SANHEDRIN_ENDPOINT%/}"
SANHEDRIN_MODELS_URL=""
[ -n "$SANHEDRIN_ENDPOINT" ] && SANHEDRIN_MODELS_URL="${SANHEDRIN_ENDPOINT%/chat/completions}/models"
SANHEDRIN_CLAIM_MODE="${VESTIGE_SANHEDRIN_CLAIM_MODE:-0}"
SANHEDRIN_OUTPUT="${VESTIGE_SANHEDRIN_OUTPUT:-text}"

echo "Vestige Cognitive Sandwich — Prereq Check"
echo

# Platform
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
ok "Platform: $OS_NAME $ARCH_NAME"
if [ "$OS_NAME" != "Darwin" ] || [ "$ARCH_NAME" != "arm64" ]; then
  info "Local MLX launchd is Apple Silicon-only; base hooks and endpoint-backed Sanhedrin can run on x86."
fi

# Python
if command -v python3 >/dev/null; then
  PY="$(python3 -c 'import sys;print(".".join(map(str,sys.version_info[:2])))' 2>/dev/null)"
  case "$PY" in
    3.9|3.1[0-9]|3.[2-9]*) ok "Python $PY" ;;
    *) warn "Python $PY (need 3.9+)" ;;
  esac
else
  miss "python3 not found"
fi

# CLI tools
command -v jq            >/dev/null && ok "jq"            || miss "jq missing — brew install jq"
if [ "$CHECK_PREFLIGHT" -eq 1 ]; then
  command -v claude        >/dev/null && ok "claude CLI"    || miss "claude CLI — install Claude Code"
  command -v vestige-mcp   >/dev/null && ok "vestige-mcp"   || miss "vestige-mcp — cargo install vestige-mcp"

  # Vestige MCP HTTP API
  if curl -fsS -m 2 "http://127.0.0.1:${DASHBOARD_PORT}/api/health" >/dev/null 2>&1; then
    ok "vestige-mcp dashboard responding on :$DASHBOARD_PORT"
  else
    warn "vestige-mcp dashboard not responding on :$DASHBOARD_PORT"
  fi
fi

# Settings hook wiring
if [ "$CHECK_PREFLIGHT" -eq 0 ] && [ "$CHECK_SANHEDRIN" -eq 0 ]; then
  if [ -f "$HOME/.claude/settings.json" ] && \
     jq -e 'any((.hooks.UserPromptSubmit[]?.hooks[]?, .hooks.Stop[]?.hooks[]?); ((.command? // "") | test("synthesis-preflight\\.sh|cwd-state-injector\\.sh|vestige-pulse-daemon\\.sh|preflight-swarm\\.sh|load-all-memory\\.sh|veto-detector\\.sh|sanhedrin\\.sh|synthesis-stop-validator\\.sh|synthesis-gate\\.sh")))' "$HOME/.claude/settings.json" >/dev/null 2>&1; then
    warn "Vestige hooks are still wired; run: install-sandwich.sh --force"
  else
    ok "no Vestige Claude Code hooks wired by default"
  fi
fi

if [ "$CHECK_PREFLIGHT" -eq 1 ]; then
  echo
  echo "Optional Preflight"

  if [ -f "$HOME/.claude/settings.json" ] && \
     jq -e 'any(.hooks.UserPromptSubmit[]?.hooks[]?; ((.command? // "") | contains("synthesis-preflight.sh"))) and any(.hooks.UserPromptSubmit[]?.hooks[]?; ((.command? // "") | contains("preflight-swarm.sh")))' "$HOME/.claude/settings.json" >/dev/null 2>&1; then
    ok "preflight UserPromptSubmit hooks wired"
  else
    warn "preflight hooks not wired — run: install-sandwich.sh --enable-preflight"
  fi

  info "preflight-swarm.sh uses claude -p with Haiku when enabled; default installs do not wire it."
fi

if [ "$CHECK_SANHEDRIN" -eq 1 ]; then
  echo
  echo "Optional Sanhedrin"

  if [ -f "$SANHEDRIN_ENV" ]; then
    ok "Sanhedrin env file present"
  else
    warn "Sanhedrin env file missing — run: install-sandwich.sh --enable-sanhedrin"
  fi
  info "Sanhedrin claim mode: $SANHEDRIN_CLAIM_MODE; output: $SANHEDRIN_OUTPUT"

  if [ "$OS_NAME" = "Darwin" ] && [ "$ARCH_NAME" = "arm64" ]; then
    command -v uv            >/dev/null && ok "uv"            || warn "uv missing — brew install uv"
    command -v mlx_lm.server >/dev/null && ok "mlx-lm"        || warn "mlx-lm — uv tool install mlx-lm"
    command -v hf            >/dev/null && ok "huggingface_hub CLI" || warn "hf — uv tool install 'huggingface_hub[cli]'"

    MODEL="${VESTIGE_SANHEDRIN_MODEL:-${VESTIGE_SANDWICH_MODEL:-}}"
    HF_HOME_DEFAULT="${HF_HOME:-$HOME/.cache/huggingface}"
    ENC_MODEL="models--$(printf '%s' "$MODEL" | sed 's|/|--|g')"
    if [ -z "$MODEL" ]; then
      info "No local MLX model configured; choose any OpenAI-compatible Sanhedrin model or preset."
    elif [ -d "$HF_HOME_DEFAULT/hub/$ENC_MODEL" ]; then
      ok "Model cached: $MODEL"
    else
      info "Model not cached: $MODEL (local MLX path downloads ~19GB)"
    fi

    if [ -f "$HOME/Library/LaunchAgents/com.vestige.mlx-server.plist" ]; then
      ok "launchd plist installed"
    else
      info "launchd plist not installed; endpoint-backed Sanhedrin can still run"
    fi
  else
    info "Skipping MLX/launchd checks on $OS_NAME $ARCH_NAME"
  fi

  if [ -z "$SANHEDRIN_MODELS_URL" ]; then
    warn "Sanhedrin endpoint/model not configured yet; hook will fail open until configured"
  elif curl -fsS -m 2 "$SANHEDRIN_MODELS_URL" >/dev/null 2>&1; then
    ok "Sanhedrin model endpoint responding at $SANHEDRIN_MODELS_URL"
  else
    warn "Sanhedrin endpoint not responding at $SANHEDRIN_MODELS_URL"
  fi

  if [ -f "$HOME/.claude/settings.json" ] && \
     jq -e '.hooks.Stop[]?.hooks[]?.command | contains("sanhedrin.sh")' "$HOME/.claude/settings.json" >/dev/null 2>&1; then
    ok "Sanhedrin Stop hook wired"
  else
    warn "Sanhedrin Stop hook not wired — run: install-sandwich.sh --enable-sanhedrin"
  fi
else
  echo
  info "Sanhedrin is optional and not checked. Use --sanhedrin to verify an enabled endpoint."
fi

echo
if [ $FAIL -eq 0 ]; then
  echo "  Ready. Default install has no Vestige Claude Code hooks wired and makes no automatic model calls."
  exit 0
else
  echo "  Fix the items above, then re-run."
  exit 1
fi
