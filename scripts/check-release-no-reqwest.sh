#!/usr/bin/env bash
# Keep this entrypoint compatible with macOS Bash 3.2 and Git Bash.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
py_bin=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -I -c "import sys" >/dev/null 2>&1; then
    py_bin="$cand"
    break
  fi
done
[ -n "$py_bin" ] || { echo "check-release-no-reqwest: python3/python interpreter not found" >&2; exit 1; }
exec "$py_bin" "$root/scripts/check-release-no-reqwest.py"
