#!/bin/bash
# synthesis-preflight.sh — UserPromptSubmit hook (v2.2: full 20-char Strata handle injection)
#
# Injects memory content directly via /api/deep_reference so relevant evidence
# is available before Claude drafts a decision-adjacent response.
# Fails open: exits 0 silently if dashboard is unreachable or no relevant hits exist.

set -u

PY_BIN=""
for _cand in python3 python; do
  if command -v "$_cand" >/dev/null 2>&1 && "$_cand" -I -c "import sys" >/dev/null 2>&1; then
    PY_BIN="$_cand"
    break
  fi
done
CURL_BIN="$(command -v curl 2>/dev/null || true)"
GREP_BIN="$(command -v grep 2>/dev/null || true)"
[ -n "$PY_BIN" ] && [ -n "$CURL_BIN" ] && [ -n "$GREP_BIN" ] || exit 0

INPUT="$(cat)"
PROMPT="$(VESTIGE_PRE_IN="$INPUT" "$PY_BIN" -I -c '
import json, os
try:
    d = json.loads(os.environ.get("VESTIGE_PRE_IN", "") or "{}")
except Exception:
    d = {}
print(d.get("prompt") or d.get("user_prompt") or "")
' 2>/dev/null || printf '')"

if [ -z "$PROMPT" ]; then
  exit 0
fi

DECISION_GATE_RE='submit|submission|benchmark|competition|final|ship|launch|deploy|commit|decide|decision|recommend|should i|should we|what should|purchase|buy|invest|architect|architecture|strategy|prep|prioriti|compose|tradeoff|trade-off|config|which|pick|choose|pitch|forecast|target|plan|roadmap|v2\.|v3\.|scale|grow|growth|distrib|brand|position|moat|vs\.|vs\b|instead of'

if ! printf '%s' "$PROMPT" | LC_ALL=C "$GREP_BIN" -iqE "$DECISION_GATE_RE"; then
  exit 0
fi

PORT="${VESTIGE_DASHBOARD_PORT:-3927}"
BASE="http://127.0.0.1:${PORT}"

if ! "$CURL_BIN" -fsS -m 0.5 "${BASE}/api/health" > /dev/null 2>&1; then
  exit 0
fi

DR_BODY="$(VPRE_PROMPT="$PROMPT" "$PY_BIN" -I -c '
import json, os
p = os.environ.get("VPRE_PROMPT", "")[:1500]
print(json.dumps({"query": p, "depth": 15}))
' 2>/dev/null || printf '')"

[ -n "$DR_BODY" ] || exit 0

DR_RESP="$("$CURL_BIN" -fsS -m 5 -X POST "${BASE}/api/deep_reference" \
  -H 'Content-Type: application/json' \
  -d "$DR_BODY" 2>/dev/null || printf '')"

[ -n "$DR_RESP" ] || exit 0

VPRE_DR_RESP="$DR_RESP" "$PY_BIN" -I -c '
import json, os, sys

try:
    d = json.loads(os.environ.get("VPRE_DR_RESP", ""))
except Exception:
    sys.exit(0)
if not isinstance(d, dict):
    sys.exit(0)

confidence = float(d.get("confidence") or 0)
intent = d.get("intent") or ""
reasoning = (d.get("reasoning") or "").strip()
recommended = d.get("recommended") or {}
evidence = d.get("evidence") or []

if not evidence and not recommended:
    sys.exit(0)

out = []
out.append("[VESTIGE PREFLIGHT — deep_reference auto-injected]")
out.append(f"Intent: {intent or \"Synthesis\"} | Confidence: {int(confidence * 100)}%")
out.append("")

if reasoning:
    out.append("REASONING CHAIN (Vestige Strata + FSRS-6):")
    out.append(reasoning[:1200])
    if len(reasoning) > 1200:
        out.append(f"  ...[reasoning truncated, full chain {len(reasoning)} chars]")
    out.append("")

if recommended:
    rec_id = str(recommended.get("memory_id") or recommended.get("id") or "")
    rec_trust = float(recommended.get("trust_score") or 0)
    rec_date = str(recommended.get("date") or "")[:10]
    rec_preview = recommended.get("answer_preview") or recommended.get("preview") or ""
    out.append(f"RECOMMENDED MEMORY [{rec_id}] trust={rec_trust:.2f} date={rec_date}:")
    out.append(rec_preview[:600])
    out.append("")

if evidence:
    out.append(f"TOP {min(len(evidence), 4)} EVIDENCE:")
    for e in evidence[:4]:
        eid = str(e.get("id") or "")
        role = e.get("role", "?")
        trust = float(e.get("trust") or 0)
        date = str(e.get("date") or "")[:10]
        preview = str(e.get("preview") or "").strip()
        out.append(f"  [{eid}] role={role} trust={trust:.2f} date={date}")
        out.append(f"    {preview[:350]}")
    out.append("")

out.append("ENFORCEMENT: Compose these into your response, do NOT just list them.")
out.append("Expand or verify any handle via recall(handle=\"<full-id>\") or causal_walk(from_handle=\"<full-id>\").")
out.append("Required shape: (a) Composing: [memories] - logic. (b) Never-composed: [combos|None].")
out.append("(c) Recommendation: concrete action.")

ctx = "\n".join(out)[:9800]
print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": ctx
    }
}))
'
exit 0
