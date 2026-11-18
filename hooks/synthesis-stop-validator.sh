#!/bin/bash
# synthesis-stop-validator.sh — optional Stop hook
#
# Blocks a narrow failure mode: a response that cites multiple memories but
# stops at summary instead of composing them into a decision.
# Honors stop_hook_active so it never loops.

set -euo pipefail

PY_BIN=""
for _cand in python3 python; do
  if command -v "$_cand" >/dev/null 2>&1 && "$_cand" -I -c "import sys" >/dev/null 2>&1; then
    PY_BIN="$_cand"
    break
  fi
done
[ -n "$PY_BIN" ] || exit 0

INPUT="$(cat)"
PARSED="$(VESTIGE_STOP_IN="$INPUT" "$PY_BIN" -I -c '
import json, os
try:
    d = json.loads(os.environ.get("VESTIGE_STOP_IN", "") or "{}")
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

RESULT="$(TRANSCRIPT_PATH="$TRANSCRIPT_PATH" "$PY_BIN" -I -c '
import json, os, re, sys

transcript = os.environ.get("TRANSCRIPT_PATH", "")
last_user = ""
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
            if role == "user" and text.strip():
                last_user = text
            elif role == "assistant" and text.strip():
                last_assistant = text
except Exception:
    sys.exit(0)

decision_re = re.compile(
    r"(submit|submission|final|ship|launch|deploy|commit|decide|decision|"
    r"recommend|should i|should we|what should|purchase|buy|invest|architect|architecture|"
    r"strategy|prep|prioriti|compose|tradeoff|trade-off|config|which "
    r"(should|model|approach|one)|pick|choose|benchmark|competition|perform)",
    re.IGNORECASE,
)
if not decision_re.search(last_user):
    sys.exit(0)

memory_re = re.compile(
    r"(mem-[0-9a-f]{16}|memory|vestige|recall|causal_walk|curate|retriev|"
    r"saved memor|stored memor|prior memor|fsrs|trust score)",
    re.IGNORECASE,
)
if not memory_re.search(last_assistant):
    sys.exit(0)

summary_patterns = [
    r"mem-[a-f0-9]{16}",
    r"memory\s+[a-f0-9]{4,}",
    r"saved memory",
    r"according to memory",
    r"the memory (says|states|notes|indicates)",
    r"per memory",
    r"memories? (say|says|state|note|indicate)",
]
summary_hits = sum(len(re.findall(pat, last_assistant, re.IGNORECASE)) for pat in summary_patterns)

composition_re = re.compile(
    r"(compos|combin|together|concrete action|recommend(ation)?\s*[:\-]|"
    r"never[- ]composed|the synthesis is|therefore|so the action is)",
    re.IGNORECASE,
)
composition_hits = len(composition_re.findall(last_assistant))

if summary_hits >= 3 and composition_hits == 0:
    print("BLOCK_SUMMARY")
else:
    print("PASS")
' 2>/dev/null || printf 'PASS')"

case "$RESULT" in
  BLOCK_SUMMARY)
    cat >&2 <<'BLOCKMSG'
[STOP BLOCKED — VESTIGE SYNTHESIS VALIDATOR]

The response cites multiple memories but does not compose them into a decision.
Rewrite it so the retrieved evidence becomes:

1. Evidence: the memory facts (with full handle IDs) that matter.
2. Implication: what those facts change.
3. Action: the concrete recommendation.

Do not stop at "Memory A says X, Memory B says Y." Compose the evidence.
BLOCKMSG
    exit 2
    ;;
  *)
    exit 0
    ;;
esac
