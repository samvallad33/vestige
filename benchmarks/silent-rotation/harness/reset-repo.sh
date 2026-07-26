#!/usr/bin/env bash
#
# Reset the torture-repo source back to its pristine BROKEN state so every run
# starts from the SAME failure. The agent edits source files to fix the bug;
# this undoes those edits between runs.
#
# Mechanism: on first use it snapshots the repo's source (everything except
# node_modules/dist/.git and the vestige demo data) into ./.repo-snapshot.
# Subsequent calls restore from that snapshot. If the repo is a git repo, it
# uses `git checkout` / `git clean` instead.
#
# Env:
#   TORTURE_REPO   path to the repo (default: ../torture-v2, the LIVE torture
#                  repo). The wrappers always pass TORTURE_REPO explicitly; the
#                  default here just avoids pointing a bare invocation at the old
#                  dead ../torture-repo.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TORTURE_REPO:-$HERE/../torture-v2}"

if [ ! -d "$REPO" ]; then
  echo "ERROR: torture repo not found at $REPO (set TORTURE_REPO)." >&2
  exit 2
fi

# The snapshot MUST be keyed to the specific repo, or switching TORTURE_REPO
# would restore one repo's pristine source into a DIFFERENT repo (which silently
# swapped the old auth-service bug into torture-v2 once). Store the snapshot
# INSIDE the repo it belongs to, so each repo carries its own.
REPO_ABS="$(cd "$REPO" && pwd)"
SNAP="$REPO_ABS/.repo-snapshot"

# Prefer git ONLY if the repo's OWN files are actually git-tracked. A repo that
# merely sits INSIDE a parent git working tree (e.g. torture-v3.5 nested under
# flagship-demo's backup git init) will make `rev-parse --git-dir` succeed, but
# `git checkout -- .` then restores NOTHING because the repo's files are
# untracked -- silently leaving prior-run pollution (e.g. a config an earlier
# agent edited) in place. That exact false-positive invalidated a paid run. So
# gate on whether git actually tracks files under $REPO; if 0 tracked files,
# fall through to the snapshot-based reset (the real source of truth).
if command -v git >/dev/null 2>&1 && git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  _tracked="$(git -C "$REPO" ls-files -- . 2>/dev/null | head -1 || true)"
  if [ -n "$_tracked" ]; then
    git -C "$REPO" checkout -- . 2>/dev/null || true
    git -C "$REPO" clean -fd -e node_modules -e .vestige-demo-data 2>/dev/null || true
    echo "reset via git"
    exit 0
  fi
  # else: repo is untracked (only the parent is a git repo) -> use the snapshot.
fi

# Snapshot-based reset for a non-git repo.
RSYNC_EXCLUDES=(--exclude node_modules --exclude dist --exclude .git \
                --exclude .vestige-demo-data --exclude .vestige-demo-db \
                --exclude .repo-snapshot --exclude '*.log')

if [ ! -d "$SNAP" ]; then
  echo "no snapshot yet -- capturing pristine state into $SNAP"
  mkdir -p "$SNAP"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${RSYNC_EXCLUDES[@]}" "$REPO/" "$SNAP/"
  else
    # Fallback without rsync: copy then prune heavy dirs.
    cp -a "$REPO/." "$SNAP/"
    rm -rf "$SNAP/node_modules" "$SNAP/dist" "$SNAP/.git" \
           "$SNAP/.vestige-demo-data" "$SNAP/.vestige-demo-db"
  fi
  echo "snapshot captured"
  exit 0
fi

echo "restoring pristine source from $SNAP"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "${RSYNC_EXCLUDES[@]}" "$SNAP/" "$REPO/"
else
  # Fallback: overwrite tracked files from the snapshot (does not delete new files).
  cp -a "$SNAP/." "$REPO/"
fi
echo "reset via snapshot"
