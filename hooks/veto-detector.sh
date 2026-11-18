#!/bin/bash
# veto-detector.sh — Stop hook (Hallucination & Deprecated-Pattern Guillotine)
#
# Queries vestige-mcp dashboard API for veto-pattern, deprecated-pattern, and
# suppressed memories, then checks if the last assistant draft contains any of
# their trigger phrases. Honors stop_hook_active so it never infinite-loops.
# Fails open if vestige-mcp is not running or no veto memories exist.

set -u

PY_BIN=""
for _cand in python3 python; do
  if command -v "$_cand" >/dev/null 2>&1 && "$_cand" -I -c "import sys" >/dev/null 2>&1; then
    PY_BIN="$_cand"
    break
  fi
done
CURL_BIN="$(command -v curl 2>/dev/null || true)"
[ -n "$PY_BIN" ] && [ -n "$CURL_BIN" ] || exit 0

INPUT="$(cat)"
PARSED="$(VESTIGE_VETO_IN="$INPUT" "$PY_BIN" -I -c '
import json, os
try:
    d = json.loads(os.environ.get("VESTIGE_VETO_IN", "") or "{}")
except Exception:
    d = {}
if d.get("stop_hook_active") is True:
    print("SKIP\t")
else:
    print("CHECK\t" + str(d.get("transcript_path") or ""))
' 2>/dev/null || printf 'SKIP\t')"

MODE="${PARSED%%	*}"
TRANSCRIPT_PATH="${PARSED#*	}"

if [ "$MODE" = "SKIP" ] || [ -z "$TRANSCRIPT_PATH" ] || [ ! -f "$TRANSCRIPT_PATH" ]; then
  exit 0
fi

PORT="${VESTIGE_DASHBOARD_PORT:-3927}"
if ! "$CURL_BIN" -fsS -m 0.5 "http://127.0.0.1:${PORT}/api/health" > /dev/null 2>&1; then
  exit 0
fi

VETO_V1="$("$CURL_BIN" -fsS -m 1.5 "http://127.0.0.1:${PORT}/api/memories?tag=veto-pattern&limit=50" 2>/dev/null || printf '{}')"
VETO_V2="$("$CURL_BIN" -fsS -m 1.5 "http://127.0.0.1:${PORT}/api/memories?tag=deprecated-pattern&limit=50" 2>/dev/null || printf '{}')"

RESULT="$(TRANSCRIPT_PATH="$TRANSCRIPT_PATH" VETO_V1="$VETO_V1" VETO_V2="$VETO_V2" "$PY_BIN" -I -c '
import json, os, re, sys

transcript = os.environ.get("TRANSCRIPT_PATH", "")
seen_ids = set()
veto_memories = []

for raw_var in ("VETO_V1", "VETO_V2"):
    try:
        vdata = json.loads(os.environ.get(raw_var, "") or "{}")
    except Exception:
        continue
    for m in (vdata.get("memories") or []):
        mid = str(m.get("id") or "")
        if not mid or mid in seen_ids:
            continue
        tags = set(m.get("tags") or [])
        if not (tags & {"veto-pattern", "deprecated-pattern", "suppressed"}):
            continue
        content = m.get("content") or ""
        triggers = re.findall(
            r"(?:VETO_TRIGGER|TRIGGER PHRASE|TRIGGER|FORBIDDEN|DEPRECATED|NEVER):\s*(.+?)(?:\n|$)",
            content,
        )
        for t in triggers:
            t = t.strip().strip("`\"\x27 ")
            if t and len(t) >= 3:
                seen_ids.add(mid)
                veto_memories.append({
                    "id": mid,
                    "trigger": t,
                    "retention": float(m.get("retentionStrength") or 0),
                })
                break

if not veto_memories:
    sys.exit(0)

last_assistant = ""
try:
    with open(transcript, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            role = obj.get("role") or obj.get("type", "")
            content = (obj.get("message") or {}).get("content", obj.get("content", ""))
            text = ""
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text += str(block.get("text") or "") + "\n"
            elif isinstance(content, str):
                text = content
            if role == "assistant" and text.strip():
                last_assistant = text
except Exception:
    sys.exit(0)

if not last_assistant:
    sys.exit(0)

hits = [
    v for v in veto_memories
    if v["retention"] >= 0.5 and re.search(r"(?i)" + re.escape(v["trigger"]), last_assistant)
]
if not hits:
    sys.exit(0)

hits.sort(key=lambda x: x["retention"], reverse=True)
top = hits[0]
print(f"VETO_HIT\t{top[\"id\"]}\t{top[\"trigger\"]}\t{int(top[\"retention\"] * 100)}")
' 2>/dev/null || printf '')"

if [ -z "$RESULT" ]; then
  exit 0
fi

NODE_ID="$(printf '%s' "$RESULT" | cut -f2)"
TRIGGER="$(printf '%s' "$RESULT" | cut -f3)"
RETENTION="$(printf '%s' "$RESULT" | cut -f4)"

cat >&2 <<VETO_MSG
[VESTIGE VETO — deprecated or vetoed pattern detected in draft]

Contradicts suppressed/vetoed memory node ${NODE_ID} (trust ${RETENTION}%).
Trigger phrase detected: "${TRIGGER}"

Rewrite WITHOUT the flagged pattern and verify the replacement via:
  recall(handle="${NODE_ID}")
VETO_MSG

exit 2
