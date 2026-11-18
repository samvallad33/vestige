#!/bin/bash
# cwd-state-injector.sh — SessionStart + UserPromptSubmit hook
#
# Injects live git + CI + Operator + rule state as additionalContext on the
# active hook event so agents start every turn grounded in actual repo state.
# Cached in ${TMPDIR:-/tmp}/cwd-state-{hash}.json for 60s. Fails open.

set -u

PY_BIN=""
for _cand in python3 python; do
  if command -v "$_cand" >/dev/null 2>&1 && "$_cand" -I -c "import sys" >/dev/null 2>&1; then
    PY_BIN="$_cand"
    break
  fi
done
GIT_BIN="$(command -v git 2>/dev/null || true)"
[ -n "$PY_BIN" ] && [ -n "$GIT_BIN" ] || exit 0

INPUT="$(cat)"
PARSED="$(VESTIGE_HOOK_INPUT="$INPUT" "$PY_BIN" -I -c '
import json, os
try:
    d = json.loads(os.environ.get("VESTIGE_HOOK_INPUT", "") or "{}")
except Exception:
    d = {}
cwd = d.get("cwd") or ""
ev = d.get("hook_event_name") or "UserPromptSubmit"
if ev not in ("SessionStart", "UserPromptSubmit"):
    ev = "UserPromptSubmit"
print(ev + "\t" + cwd)
' 2>/dev/null || printf 'UserPromptSubmit\t')"

EVENT_NAME="${PARSED%%	*}"
CWD="${PARSED#*	}"
[ -n "$EVENT_NAME" ] || EVENT_NAME="UserPromptSubmit"

if [ -z "$CWD" ] || [ ! -d "$CWD" ]; then
  CWD="$(pwd 2>/dev/null || true)"
fi

cd "$CWD" 2>/dev/null || exit 0
if ! "$GIT_BIN" rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  exit 0
fi

CACHE_KEY="$(VESTIGE_CWD_RAW="$CWD:$EVENT_NAME" "$PY_BIN" -I -c '
import hashlib, os
print(hashlib.sha256(os.environ.get("VESTIGE_CWD_RAW","").encode()).hexdigest()[:16])
' 2>/dev/null || echo "default")"
CACHE_DIR="${TMPDIR:-/tmp}"
CACHE_FILE="${CACHE_DIR%/}/cwd-state-${CACHE_KEY}.json"
if [ -f "$CACHE_FILE" ]; then
  MTIME=$(stat -c %Y "$CACHE_FILE" 2>/dev/null || stat -f %m "$CACHE_FILE" 2>/dev/null || echo 0)
  case "$MTIME" in ''|*[!0-9]*) MTIME=0 ;; esac
  NOW=$(date +%s 2>/dev/null || echo 0)
  AGE=$((NOW - MTIME))
  if [ "$AGE" -ge 0 ] && [ "$AGE" -lt 60 ] && [ -s "$CACHE_FILE" ]; then
    cat "$CACHE_FILE"
    exit 0
  fi
fi

BRANCH="$("$GIT_BIN" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
HEAD_SHA="$("$GIT_BIN" rev-parse --short HEAD 2>/dev/null || true)"
HEAD_SUBJECT="$("$GIT_BIN" log -1 --format='%s' 2>/dev/null | head -c 100 || true)"
HEAD_AUTHOR="$("$GIT_BIN" log -1 --format='%an' 2>/dev/null || true)"
STAGED_COUNT="$("$GIT_BIN" diff --cached --name-only 2>/dev/null | wc -l | awk '{print $1}')"
UNSTAGED_COUNT="$("$GIT_BIN" diff --name-only 2>/dev/null | wc -l | awk '{print $1}')"
UNTRACKED_COUNT="$("$GIT_BIN" ls-files --others --exclude-standard 2>/dev/null | wc -l | awk '{print $1}')"
CONFLICT_COUNT="$("$GIT_BIN" diff --name-only --diff-filter=U 2>/dev/null | wc -l | awk '{print $1}')"
AHEAD_BEHIND="$("$GIT_BIN" rev-list --left-right --count HEAD...@{upstream} 2>/dev/null | awk '{printf "ahead=%s behind=%s", $1, $2}')"
[ -n "$AHEAD_BEHIND" ] || AHEAD_BEHIND="no-upstream"

GIT_DIR="$("$GIT_BIN" rev-parse --git-dir 2>/dev/null || true)"
OP_MODE="normal"
if [ -n "$GIT_DIR" ]; then
  if [ -d "$GIT_DIR/rebase-merge" ] || [ -d "$GIT_DIR/rebase-apply" ]; then
    OP_MODE="rebase"
  elif [ -f "$GIT_DIR/MERGE_HEAD" ]; then
    OP_MODE="merge"
  elif [ -f "$GIT_DIR/BISECT_LOG" ]; then
    OP_MODE="bisect"
  fi
fi

WORKTREE_COUNT="$("$GIT_BIN" worktree list --porcelain 2>/dev/null | grep -c '^worktree ' || echo 1)"

RULE_TAGS="0"
if [ -f ".vestige-rules" ]; then
  RULE_TAGS="$(grep -cve '^[[:space:]]*\(#\|$\)' .vestige-rules 2>/dev/null || echo 0)"
fi

OP_HOME="${OPERATOR_HOME:-$HOME/.operator}"
TODAY_UTC="$(date -u +%Y-%m-%d 2>/dev/null || true)"
TODAY_STOPS="0"
if [ -n "$TODAY_UTC" ] && [ -f "$OP_HOME/receipts/$TODAY_UTC.jsonl" ]; then
  TODAY_STOPS="$(grep -c '"decision":[[:space:]]*"STOP"' "$OP_HOME/receipts/$TODAY_UTC.jsonl" 2>/dev/null || echo 0)"
fi

CI_STATE=""
PR_COUNT=""
ISSUE_COUNT=""
if [ "${VESTIGE_CWD_GH:-0}" = "1" ] && command -v gh > /dev/null 2>&1 && "$GIT_BIN" config --get remote.origin.url > /dev/null 2>&1; then
  GH_PARSED="$(VESTIGE_GH_BRANCH="$BRANCH" "$PY_BIN" -I -c '
import json, os, subprocess
branch = os.environ.get("VESTIGE_GH_BRANCH", "")
def run_gh(args):
    try:
        r = subprocess.run(
            ["gh"] + args,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=2.5,
        )
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None
ci = run_gh(["run", "list", "--branch", branch, "--limit", "1", "--json", "status,conclusion,name"])
prs = run_gh(["pr", "list", "--state", "open", "--limit", "100", "--json", "number"])
iss = run_gh(["issue", "list", "--state", "open", "--limit", "100", "--json", "number"])
ci_str = ""
if isinstance(ci, list) and ci:
    r = ci[0]
    ci_str = "%s:%s:%s" % (r.get("name", "?"), r.get("status", "?"), r.get("conclusion") or "...")
pr_str = str(len(prs)) if isinstance(prs, list) else ""
is_str = str(len(iss)) if isinstance(iss, list) else ""
print(ci_str + "\t" + pr_str + "\t" + is_str)
' 2>/dev/null || printf '\t\t')"
  CI_STATE="${GH_PARSED%%	*}"
  _GH_REST="${GH_PARSED#*	}"
  PR_COUNT="${_GH_REST%%	*}"
  ISSUE_COUNT="${_GH_REST#*	}"
fi

REPO_NAME="$(basename "$CWD")"
CONTEXT_LINES=()
CONTEXT_LINES+=("[CWD STATE — auto-injected, 60s cache]")
CONTEXT_LINES+=("  repo: $REPO_NAME  branch: $BRANCH  HEAD: $HEAD_SHA  mode: $OP_MODE  worktrees: $WORKTREE_COUNT")
if [ -n "$HEAD_SUBJECT" ]; then
  CONTEXT_LINES+=("  last commit: \"$HEAD_SUBJECT\" by $HEAD_AUTHOR")
fi
CONTEXT_LINES+=("  working tree: staged=$STAGED_COUNT unstaged=$UNSTAGED_COUNT untracked=$UNTRACKED_COUNT conflicts=$CONFLICT_COUNT")
if [ "$AHEAD_BEHIND" != "no-upstream" ]; then
  CONTEXT_LINES+=("  vs upstream: $AHEAD_BEHIND")
fi
if [ -n "$CI_STATE" ]; then
  CONTEXT_LINES+=("  last CI run: $CI_STATE")
fi
if [ -n "$PR_COUNT" ] && [ -n "$ISSUE_COUNT" ]; then
  CONTEXT_LINES+=("  open: PRs=$PR_COUNT issues=$ISSUE_COUNT")
fi
CONTEXT_LINES+=("  vestige: rules_tags=$RULE_TAGS operator_stops_today=$TODAY_STOPS")

JSON_OUT="$(VESTIGE_CWD_EVENT="$EVENT_NAME" VESTIGE_CWD_STATE="$(printf '%s\n' "${CONTEXT_LINES[@]}")" "$PY_BIN" -I -c '
import json, os
ev = os.environ.get("VESTIGE_CWD_EVENT", "UserPromptSubmit")
ctx = os.environ.get("VESTIGE_CWD_STATE", "").strip()
print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": ev,
        "additionalContext": ctx
    }
}))
')"

printf '%s' "$JSON_OUT" > "$CACHE_FILE"
printf '%s' "$JSON_OUT"
exit 0
