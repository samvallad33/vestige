#!/bin/bash
# vestige-pulse-daemon.sh — UserPromptSubmit hook for recent Vestige insights
#
# Polls /api/changelog for DreamCompleted, BackfillCompleted, or
# ConnectionDiscovered events and emits proper hookSpecificOutput JSON with
# full mem-... handles. Rate-limited to once per 20 minutes per session.
# Fails open silently if vestige-mcp dashboard is offline.

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

STATE_DIR="${VESTIGE_PULSE_STATE_DIR:-${TMPDIR:-/tmp}/vestige-pulse-daemon}"
mkdir -p "$STATE_DIR"
LAST_FIRE_FILE="$STATE_DIR/last_fire"
SESSION_ID_FILE="$STATE_DIR/session_id"

INPUT="$(cat)"
SESSION_ID="$(VESTIGE_PULSE_IN="$INPUT" "$PY_BIN" -I -c '
import json, os
try:
    d = json.loads(os.environ.get("VESTIGE_PULSE_IN", "") or "{}")
except Exception:
    d = {}
print(d.get("session_id") or "")
' 2>/dev/null || printf '')"

MIN_INTERVAL_SEC=1200
NOW=$(date +%s 2>/dev/null || echo 0)

if [ -f "$LAST_FIRE_FILE" ]; then
  LAST_FIRE=$(cat "$LAST_FIRE_FILE" 2>/dev/null || echo 0)
  LAST_SESSION=$(cat "$SESSION_ID_FILE" 2>/dev/null || echo "")
  if [ "$LAST_SESSION" = "$SESSION_ID" ] && [ $((NOW - LAST_FIRE)) -lt $MIN_INTERVAL_SEC ]; then
    exit 0
  fi
fi

PORT="${VESTIGE_DASHBOARD_PORT:-3927}"
if ! "$CURL_BIN" -fsS -m 0.5 "http://127.0.0.1:${PORT}/api/health" > /dev/null 2>&1; then
  exit 0
fi

SINCE_UTC="$("$PY_BIN" -I -c '
from datetime import datetime, timedelta, timezone
print((datetime.now(timezone.utc) - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ"))
' 2>/dev/null || true)"

[ -n "$SINCE_UTC" ] || exit 0

RESULT="$("$CURL_BIN" -fsS -m 2 \
  "http://127.0.0.1:${PORT}/api/changelog?start=${SINCE_UTC}&limit=50" \
  2>/dev/null || printf '')"

[ -n "$RESULT" ] || exit 0

INSIGHT="$(VESTIGE_CHANGELOG_JSON="$RESULT" "$PY_BIN" -I -c '
import json, os, sys

def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0

try:
    data = json.loads(os.environ.get("VESTIGE_CHANGELOG_JSON", ""))
except Exception:
    sys.exit(0)
if not isinstance(data, dict):
    sys.exit(0)

for ev in (data.get("events") or []):
    if not isinstance(ev, dict):
        continue
    etype = ev.get("type", "")
    payload = ev.get("data") if isinstance(ev.get("data"), dict) else ev
    if etype in ("DreamCompleted", "dream", "consolidation"):
        insights = as_int(payload.get("insights_generated") or payload.get("insightsGenerated"))
        if insights > 0:
            stats = payload.get("stats") or {}
            connections = as_int(
                payload.get("connections_persisted")
                or payload.get("connectionsPersisted")
                or payload.get("connections_found")
                or payload.get("connectionsFound")
                or stats.get("connections")
            )
            print(f"DREAM: {insights} insights, {connections} new connections surfaced. Inspect via curate(action=\"dream\") or causal_walk.")
            sys.exit(0)
    elif etype in ("ConnectionDiscovered", "connection"):
        src = str(payload.get("source_id") or payload.get("sourceId") or payload.get("source") or "")
        tgt = str(payload.get("target_id") or payload.get("targetId") or payload.get("target") or "")
        if src and tgt:
            print(f"CONNECTION: New causal edge [{src}] <-> [{tgt}] discovered. Inspect via recall(handle=\"{src}\") or causal_walk(from_handle=\"{tgt}\").")
            sys.exit(0)
    elif etype in ("BackfillCompleted", "backfill"):
        mid = str(payload.get("memory_id") or payload.get("memoryId") or "")
        rid = str(payload.get("receipt_id") or payload.get("receiptId") or "")
        if mid:
            print(f"RSB: Retroactive salience raised {mid} (receipt {rid}). Inspect via recall(handle=\"{mid}\").")
            sys.exit(0)
' 2>/dev/null || printf '')"

if [ -z "$INSIGHT" ]; then
  exit 0
fi

echo "$NOW" > "$LAST_FIRE_FILE"
echo "$SESSION_ID" > "$SESSION_ID_FILE"

VESTIGE_PULSE_INSIGHT="$INSIGHT" "$PY_BIN" -I -c '
import json, os
insight = os.environ.get("VESTIGE_PULSE_INSIGHT", "").strip()
msg = (
    "[VESTIGE PULSE — autonomous insight from the cognitive engine]\n\n"
    f"{insight}\n\n"
    "Surface this naturally if relevant to the current task."
)
print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": msg
    }
}))
'
exit 0
