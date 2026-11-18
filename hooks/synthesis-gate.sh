#!/bin/bash
# synthesis-gate.sh — optional UserPromptSubmit hook
#
# Injects a compact synthesis contract for decision-adjacent prompts.
# Public-safe: portable binaries, zero local paths or private examples.

set -euo pipefail

PY_BIN=""
for _cand in python3 python; do
  if command -v "$_cand" >/dev/null 2>&1 && "$_cand" -I -c "import sys" >/dev/null 2>&1; then
    PY_BIN="$_cand"
    break
  fi
done
GREP_BIN="$(command -v grep 2>/dev/null || true)"
[ -n "$PY_BIN" ] && [ -n "$GREP_BIN" ] || exit 0

INPUT="$(cat)"
PROMPT="$(VESTIGE_GATE_IN="$INPUT" "$PY_BIN" -I -c '
import json, os
try:
    d = json.loads(os.environ.get("VESTIGE_GATE_IN", "") or "{}")
except Exception:
    d = {}
print(d.get("prompt") or d.get("user_prompt") or "")
' 2>/dev/null || printf '')"

DECISION_REGEX='(submit|submission|final|ship|launch|deploy|commit|decide|decision|recommend|should i|should we|what should|purchase|buy|invest|architect|architecture|strategy|prep|prioriti|compose|tradeoff|trade-off|config|which (should|model|approach|one)|pick|choose|benchmark|competition|perform)'

if printf '%s' "$PROMPT" | "$GREP_BIN" -qiE "$DECISION_REGEX"; then
  "$PY_BIN" -I -c '
import json
msg = (
    "[VESTIGE SYNTHESIS GATE]\n\n"
    "This prompt appears decision-adjacent. Ground decisions in exact Vestige handles before answering.\n\n"
    "Reasoning contract:\n"
    "1. Check recall(handle=...) and causal_walk(...) across related decisions and prior failures.\n"
    "2. Convert each useful memory into: fact -> implication -> action.\n"
    "3. Surface contradictions or superseded nodes before recommending (curate superseded beliefs when replaced).\n"
    "4. Do not dump raw memory summaries as the answer. Compose them into a concrete recommendation.\n"
    "5. If no recorded memory changes the answer, say so briefly and proceed from source evidence."
)
print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": msg
    }
}))
'
fi

exit 0
