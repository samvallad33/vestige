#!/usr/bin/env bash
# check-no-private-paths.sh — Fail if a private local filesystem path, personal
# username, or machine name leaks into this public repo.
#
# Background: some planning docs once carried real absolute paths such as
#   /home/<realuser>/prppl/vestige-phase2/...
# plus a developer machine name. Those are metadata leaks (username, folder
# layout, host) — not credentials, but exactly the kind of thing the public-repo
# hygiene rules say to keep out. They were removed; this guard stops them (and
# anything like them) from creeping back in.
#
# It scans only tracked files (git grep) for two classes of leak:
#   1. Known private tokens — exact handles / private worktree names.
#   2. Absolute home paths that reveal a REAL username, while allowlisting the
#      obvious placeholders the docs legitimately use (e.g. /Users/you/...).
#
# If a match is a genuine false positive, add the placeholder to ALLOW_USERS or
# refine the pattern below — do not weaken it to a no-op.
set -u

cd "$(git rev-parse --show-toplevel)" || {
  echo "check-no-private-paths: not inside a git repository" >&2
  exit 2
}

# 1) Exact private tokens that must never appear anywhere in the repo.
BLOCK_TOKENS=(
  'delandtj'
  'prppl'
)

# 2) Placeholder usernames the docs legitimately use in example home paths.
#    A /home/<name>/ or /Users/<name>/ whose <name> is NOT one of these trips.
ALLOW_USERS='you|user|username|name|me|example|runner|youruser|YOU|USER|USERNAME|YOURNAME'

# This guard and its workflow name the tokens/patterns, so exclude them.
EXCLUDES=(
  ':(exclude)scripts/check-no-private-paths.sh'
  ':(exclude).github/workflows/guard-no-private-cloud.yml'
)

violations=0
report=""

# --- 1. exact private tokens ---------------------------------------------------
for tok in "${BLOCK_TOKENS[@]}"; do
  if hits=$(git grep -Ini -F -e "$tok" -- "${EXCLUDES[@]}" 2>/dev/null); then
    if [ -n "$hits" ]; then
      violations=$((violations + 1))
      report+=$'\n'"  ✗ private token: \"$tok\""$'\n'
      report+="$(printf '%s\n' "$hits" | sed 's/^/      /')"$'\n'
    fi
  fi
done

# --- 2. absolute home paths that reveal a real username ------------------------
# Candidate lines contain /home/<x>/ or /Users/<x>/ or C:\Users\<x>\; then drop
# the lines whose username segment is an allowlisted placeholder.
if cand=$(git grep -nIE '(/home/|/Users/|[Cc]:\\Users\\)[A-Za-z0-9_.-]+[/\\]' \
            -- "${EXCLUDES[@]}" 2>/dev/null); then
  leaks=$(printf '%s\n' "$cand" \
            | grep -vE "(/home/|/Users/|[Cc]:\\\\Users\\\\)(${ALLOW_USERS})[/\\]" || true)
  if [ -n "$leaks" ]; then
    violations=$((violations + 1))
    report+=$'\n'"  ✗ private home path (real username?):"$'\n'
    report+="$(printf '%s\n' "$leaks" | sed 's/^/      /')"$'\n'
  fi
fi

if [ "$violations" -ne 0 ]; then
  echo "════════════════════════════════════════════════════════════════════"
  echo "  PRIVATE PATH / USERNAME / MACHINE NAME DETECTED IN PUBLIC REPO"
  echo "════════════════════════════════════════════════════════════════════"
  echo "$report"
  echo "════════════════════════════════════════════════════════════════════"
  echo "  Real local paths, usernames, and host names must not be committed."
  echo "  Use a placeholder (e.g. /Users/you/... or ~/) instead. If a match is"
  echo "  a false positive, extend ALLOW_USERS in scripts/check-no-private-paths.sh."
  echo "════════════════════════════════════════════════════════════════════"
  exit 1
fi

echo "check-no-private-paths: OK — no private paths, usernames, or host names in public repo"
exit 0
