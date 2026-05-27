#!/bin/bash
# sanhedrin.sh — Stop hook (Post-Cognitive Sanhedrin / Full Agent-Type Guillotine)
#
# Spawns the Executioner subagent (Haiku 4.5, fresh context, Vestige MCP
# tools) to run mcp__vestige__deep_reference 8-stage contradiction analysis
# on the last assistant draft. If any technical claim contradicts a
# high-trust memory, exit 2 with the veto reason — forces Main Claude to
# rewrite.
#
# Runs AFTER veto-detector.sh (fast regex against veto-tagged memories).
# Sanhedrin is the deeper semantic check: it reads the draft as a real
# reasoning agent, extracts claims, runs deep_reference on each.
#
# Architecture:
#   Main Claude finishes draft → Stop hook chain fires →
#   veto-detector.sh (50ms regex, may block) →
#   sanhedrin.sh (2-8s Haiku subagent, may block) →
#   synthesis-stop-validator.sh (existing regex hedge check, may block)
#
# Opt-in: set VESTIGE_SANHEDRIN_ENABLED=1 in parent shell, or install with
# scripts/install-sandwich.sh --enable-sanhedrin.
# Re-entrancy lock: VESTIGE_EXECUTIONER_ACTIVE=1 inside the subagent.
#
# Ship date 2026-04-20.

set -u

load_vestige_sanhedrin_env() {
  [ -f "$1" ] || return 0
  command -v python3 >/dev/null 2>&1 || return 0
  while IFS="$(printf '\t')" read -r key value; do
    case "$key" in
      VESTIGE_SANHEDRIN_ENABLED|VESTIGE_SANHEDRIN_MODEL|VESTIGE_SANHEDRIN_ENDPOINT|VESTIGE_SANHEDRIN_API_KEY|VESTIGE_SANHEDRIN_BACKEND|VESTIGE_SANHEDRIN_CLAIM_MODE|VESTIGE_SANHEDRIN_OUTPUT|VESTIGE_SANHEDRIN_PYTHON|VESTIGE_SANHEDRIN_STATE_DIR|VESTIGE_SANHEDRIN_ALLOW_COMMAND_LEDGER|VESTIGE_SANHEDRIN_ALLOW_LOOSE_LEDGER|VESTIGE_DASHBOARD_PORT)
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
    "VESTIGE_SANHEDRIN_API_KEY",
    "VESTIGE_SANHEDRIN_BACKEND",
    "VESTIGE_SANHEDRIN_CLAIM_MODE",
    "VESTIGE_SANHEDRIN_OUTPUT",
    "VESTIGE_SANHEDRIN_PYTHON",
    "VESTIGE_SANHEDRIN_STATE_DIR",
    "VESTIGE_SANHEDRIN_ALLOW_COMMAND_LEDGER",
    "VESTIGE_SANHEDRIN_ALLOW_LOOSE_LEDGER",
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

# === OPT-IN GATE ===
# Sanhedrin is opt-in and model-agnostic. It never guesses a large verifier
# model; if endpoint/model are unset, the bridge fails open with telemetry.
# The installer writes this env file only for --enable-sanhedrin.
SANHEDRIN_ENV="${VESTIGE_SANHEDRIN_ENV:-$HOME/.claude/hooks/vestige-sanhedrin.env}"
if [ -f "$SANHEDRIN_ENV" ]; then
  load_vestige_sanhedrin_env "$SANHEDRIN_ENV" || exit 0
fi

case "${VESTIGE_SANHEDRIN_ENABLED:-0}" in
  1|true|TRUE|yes|YES|on|ON) ;;
  *) exit 0 ;;
esac

# === RE-ENTRANCY GUARD ===
# The Executioner's own Stop hook will fire when it returns — prevent
# recursive spawns that would fork-bomb the quota.
if [ "${VESTIGE_EXECUTIONER_ACTIVE:-0}" = "1" ]; then
  exit 0
fi

PYTHON_BIN="${VESTIGE_SANHEDRIN_PYTHON:-}"
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 2>/dev/null || printf '')"
fi
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="/usr/bin/python3"
fi
if ! "$PYTHON_BIN" -c 'import sys' >/dev/null 2>&1; then
  exit 0
fi

record_sanhedrin_fail_open() {
  REASON="$1"
  DETAIL="${2:-}"
  "$PYTHON_BIN" - "$REASON" "$DETAIL" <<'PY' 2>/dev/null || true
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

reason = sys.argv[1] if len(sys.argv) > 1 else "unknown"
detail = sys.argv[2] if len(sys.argv) > 2 else ""
state_dir = Path(os.environ.get("VESTIGE_SANHEDRIN_STATE_DIR") or Path.home() / ".vestige" / "sanhedrin")
try:
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "fail-open.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runId": os.environ.get("VESTIGE_SANHEDRIN_RUN_ID"),
            "reason": reason,
            "detail": detail[:500],
            "transcript": os.environ.get("TRANSCRIPT_PATH") or os.environ.get("VESTIGE_SANHEDRIN_TRANSCRIPT"),
        }) + "\n")
except OSError:
    pass
PY
}

# === READ STOP HOOK INPUT ===
INPUT="$(cat)"
TRANSCRIPT_PATH="$(printf '%s' "$INPUT" | "$PYTHON_BIN" -c 'import sys,json;d=json.load(sys.stdin);print(d.get("transcript_path",""))' 2>/dev/null || printf '')"

if [ -z "$TRANSCRIPT_PATH" ] || [ ! -f "$TRANSCRIPT_PATH" ]; then
  exit 0
fi

# === EXTRACT LAST ASSISTANT DRAFT ===
# Read the transcript JSONL, pull the last assistant message text.
export TRANSCRIPT_PATH
DRAFT_SCRIPT="$(mktemp -t vestige-sanhedrin-draft.XXXXXX)"
trap 'rm -f "$DRAFT_SCRIPT"' EXIT

cat > "$DRAFT_SCRIPT" <<'DRAFT_PYEOF'
import json, os, re, sys

transcript = os.environ.get("TRANSCRIPT_PATH", "")
last_assistant = ""

try:
    with open(transcript) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            role = obj.get("role") or obj.get("type", "")
            content = obj.get("message", {}).get("content", obj.get("content", ""))
            text = ""
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text += block.get("text", "") + "\n"
            elif isinstance(content, str):
                text = content
            if role == "assistant":
                last_assistant = text
except Exception:
    sys.exit(0)

# Print nothing if no draft. Short verification claims still need Receipt Lock.
stripped = last_assistant.strip()
if not stripped:
    sys.exit(0)

# Legacy gate: only check drafts that contain technical indicators. Claim mode
# deliberately broadens this to any substantive assistant draft while keeping
# Sanhedrin opt-in through VESTIGE_SANHEDRIN_ENABLED.
claim_mode = os.environ.get("VESTIGE_SANHEDRIN_CLAIM_MODE", "") == "1"
receipt_gate = bool(
    re.search(
        r"\b((all\s+)?(tests?|test suite|build|lint|typecheck|checks?|cargo test|npm test|pnpm test|pytest|vitest|jest|playwright|tsc|clippy)\s+(passed|passes|passing|green|succeeded|succeeds|clean)|(verified|validated|confirmed)\s+(with|by|via))\b",
        stripped,
        re.I,
    )
)
if len(stripped) < 100 and not receipt_gate:
    sys.exit(0)

if not claim_mode:
    has_code = "`" in stripped or "```" in stripped
    has_cmd = any(kw in stripped.lower() for kw in ["install", "run ", "use ", "call ", "invoke", "execute"])
    has_path = "/" in stripped and any(ext in stripped for ext in [".rs", ".ts", ".py", ".sh", ".md", ".json"])

    if not (has_code or has_cmd or has_path or receipt_gate):
        sys.exit(0)

# Truncate to 4000 chars to keep Haiku prompt bounded
if len(stripped) > 4000:
    stripped = stripped[:4000] + "... [truncated]"

print(stripped)
DRAFT_PYEOF

DRAFT="$("$PYTHON_BIN" "$DRAFT_SCRIPT" 2>/dev/null || printf '')"

if [ -z "$DRAFT" ]; then
  exit 0
fi

# === VERIFY local executioner bridge available ===
# 2026-04-25: switched from Haiku 4.5 subagent to an OpenAI-compatible
# local/remote endpoint. On Apple Silicon the optional launchd path starts
# mlx_lm.server; on x86 users can point VESTIGE_SANHEDRIN_ENDPOINT at vLLM,
# Ollama, llama.cpp, or any compatible /v1/chat/completions endpoint.
# Fail-open if the endpoint is unreachable.
BRIDGE="$HOME/.claude/hooks/sanhedrin-local.py"
if [ ! -x "$BRIDGE" ] && [ ! -f "$BRIDGE" ]; then
  exit 0
fi

# === SPAWN LOCAL EXECUTIONER (background with timeout) ===
OUTPUT_FILE="$(mktemp -t vestige-sanhedrin-out.XXXXXX)"
trap 'rm -f "$DRAFT_SCRIPT" "$OUTPUT_FILE"' EXIT
export VESTIGE_SANHEDRIN_TRANSCRIPT="$TRANSCRIPT_PATH"
export VESTIGE_SANHEDRIN_RUN_ID="${VESTIGE_SANHEDRIN_RUN_ID:-$(date +%s)-$$}"
export VESTIGE_EXECUTIONER_ACTIVE=1

(
  printf '%s\n' "$DRAFT" | "$PYTHON_BIN" "$BRIDGE" > "$OUTPUT_FILE" 2>/dev/null
) &

EXEC_PID=$!

# === TIMEOUT GUARD (60 seconds) ===
# Local Qwen3.6-35B-A3B on M5/M3 Max typically returns in 5-15s for the
# single-shot judgment. 60s ceiling preserves the existing settings.json
# Stop hook timeout (70s) and gives headroom for cold model load if
# launchd just restarted. Bridge fail-opens internally if mlx-server is
# unreachable, so timeout-kill here is the secondary safety net.
WAITED=0
while [ "$WAITED" -lt 60 ]; do
  if ! /bin/kill -0 "$EXEC_PID" 2>/dev/null; then
    break
  fi
  sleep 1
  WAITED=$((WAITED + 1))
done
if /bin/kill -0 "$EXEC_PID" 2>/dev/null; then
  /bin/kill "$EXEC_PID" 2>/dev/null
  wait "$EXEC_PID" 2>/dev/null
  record_sanhedrin_fail_open "timeout" "sanhedrin-local.py exceeded 60s"
  exit 0
fi
wait "$EXEC_PID" 2>/dev/null

EXECUTIONER_OUTPUT="$(cat "$OUTPUT_FILE" 2>/dev/null || printf '')"

# === PARSE VERDICT ===
sanhedrin_veto() {
  REASON="$1"
  REASON="$(printf '%s' "$REASON" | "$PYTHON_BIN" -c 'import sys; print(sys.stdin.read().strip())' 2>/dev/null || printf '%s' "$REASON")"

  if printf '%s' "$REASON" | /usr/bin/grep -qi 'Receipt Lock'; then
    cat >&2 <<SANHEDRIN_RECEIPT_MSG
[SANHEDRIN VETO - Receipt Lock rejected draft]

$REASON

You may NOT stop with an unsupported verification claim. Either run the
matching test/build/lint/typecheck command successfully in this session, or
rewrite the response to say the command was not run.

Receipt artifact:
~/.vestige/sanhedrin/latest.html
SANHEDRIN_RECEIPT_MSG
    exit 2
  fi

  cat >&2 <<SANHEDRIN_MSG
[SANHEDRIN VETO - Post-Cognitive Executioner (LOCAL) rejected draft]

$REASON

The Executioner (Sanhedrin endpoint, fresh context, fed Vestige
deep_reference evidence over HTTP) judged your draft and
found a contradiction against a high-trust memory.

You may NOT stop. Rewrite WITHOUT the contradicted claim. Use
mcp__vestige__deep_reference to inspect the cited memory and cite the
correct replacement pattern from its \`recommended\` field.

Bridge script:
~/.claude/hooks/sanhedrin-local.py

Receipt artifact:
~/.vestige/sanhedrin/latest.html
SANHEDRIN_MSG
  exit 2
}

if [ "${VESTIGE_SANHEDRIN_OUTPUT:-}" = "json" ]; then
  JSON_PARSED="$(printf '%s' "$EXECUTIONER_OUTPUT" | "$PYTHON_BIN" -c '
import json
import sys

raw = sys.stdin.read()

def loads_candidate(text):
    text = text.strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except Exception:
        return None
    return value if isinstance(value, dict) else None

obj = loads_candidate(raw)
if obj is None:
    for line in reversed([ln for ln in raw.splitlines() if ln.strip()]):
        obj = loads_candidate(line)
        if obj is not None:
            break
if obj is None:
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        obj = loads_candidate(raw[start:end + 1])

if obj is None:
    sys.exit(1)

decision = obj.get("decision", obj.get("verdict", obj.get("answer", "")))
reason = obj.get("reason", obj.get("message", obj.get("explanation", "")))
if isinstance(decision, bool):
    decision = "yes" if decision else "no"
elif decision is None:
    decision = ""
else:
    decision = str(decision)

if reason is None:
    reason = ""
elif not isinstance(reason, str):
    reason = json.dumps(reason, ensure_ascii=False)

print(decision.strip())
print(reason.strip())
' 2>/dev/null || printf '')"

  if [ -n "$JSON_PARSED" ]; then
    JSON_DECISION="$(printf '%s\n' "$JSON_PARSED" | /usr/bin/sed -n '1p' | "$PYTHON_BIN" -c 'import sys; print(sys.stdin.read().strip().lower())' 2>/dev/null || printf '')"
    JSON_REASON="$(printf '%s\n' "$JSON_PARSED" | /usr/bin/sed '1d')"

    case "$JSON_DECISION" in
      yes|pass|allow|allowed|clean|true)
        exit 0
        ;;
      no|fail|block|blocked|veto|false)
        sanhedrin_veto "$JSON_REASON"
        ;;
    esac
  fi
fi

TRIMMED="$(printf '%s' "$EXECUTIONER_OUTPUT" | /usr/bin/awk 'NF {print; exit}' | /usr/bin/awk '{$1=$1;print}')"

if [ -z "$TRIMMED" ]; then
  record_sanhedrin_fail_open "empty_verdict" "sanhedrin-local.py produced no parseable output"
  exit 0
fi

# "yes" verdict - draft is clean, allow stop
case "$TRIMMED" in
  yes|YES|Yes|yes.|Yes.)
    exit 0
    ;;
esac

# "no - <reason>" or "no: <reason>" verdict - block the stop, force rewrite
# Documented spec is `no - [Sanhedrin Veto] [CLASS]: <reason>` (hyphen-space).
# Legacy `no: <reason>` also accepted for backward compat.
case "$TRIMMED" in
  no\ -*|NO\ -*|No\ -*|no:*|NO:*|No:*)
    case "$TRIMMED" in
      no\ -*|NO\ -*|No\ -*)
        REASON="${TRIMMED#* - }"
        ;;
      *)
        REASON="${TRIMMED#*:}"
        ;;
    esac
    sanhedrin_veto "$REASON"
    ;;
esac

# Unparseable verdict — fail open (do not block on Executioner errors)
record_sanhedrin_fail_open "unparseable_verdict" "$TRIMMED"
exit 0
