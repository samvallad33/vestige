#!/usr/bin/env bash
# Fail if a Linux ELF requires glibc newer than Ubuntu 22.04 (2.35) or
# libstdc++ newer than GCC 12 / Ubuntu 22.04 (GLIBCXX_3.4.30).
#
# Debian 12 is glibc 2.36 / GLIBCXX_3.4.30, so the 22.04 ceiling is the
# floor that covers both. Used by CI against the linux-gnu release binary.
set -euo pipefail

usage() {
  echo "Usage: $0 <path-to-elf>" >&2
  exit 2
}

[[ $# -eq 1 ]] || usage
BIN=$1
[[ -f "$BIN" ]] || { echo "error: not a file: $BIN" >&2; exit 2; }

if ! command -v objdump >/dev/null 2>&1; then
  echo "error: objdump is required (binutils)" >&2
  exit 2
fi

MAX_GLIBC=2.35
MAX_GLIBCXX=3.4.30

version_gt() {
  # Return 0 if $1 > $2 using sort -V.
  local a=$1 b=$2
  [[ "$a" != "$b" && "$(printf '%s\n%s\n' "$a" "$b" | sort -V | tail -1)" == "$a" ]]
}

dynsyms=$(objdump -T "$BIN" 2>/dev/null || true)

glibc_versions=$(printf '%s\n' "$dynsyms" \
  | grep -oE 'GLIBC_[0-9]+\.[0-9]+(\.[0-9]+)?' \
  | sed 's/^GLIBC_//' \
  | sort -uV)
glibcxx_versions=$(printf '%s\n' "$dynsyms" \
  | grep -oE 'GLIBCXX_[0-9]+\.[0-9]+(\.[0-9]+)?' \
  | sed 's/^GLIBCXX_//' \
  | sort -uV)

failed=0

echo "glibc versions referenced by $(basename "$BIN"):"
if [[ -z "$glibc_versions" ]]; then
  echo "  (none)"
else
  while IFS= read -r ver; do
    echo "  GLIBC_$ver"
    if version_gt "$ver" "$MAX_GLIBC"; then
      echo "error: $BIN requires GLIBC_$ver (Ubuntu 22.04 / Debian 12 ceiling is $MAX_GLIBC)" >&2
      failed=1
    fi
  done <<< "$glibc_versions"
fi

echo "GLIBCXX versions referenced by $(basename "$BIN"):"
if [[ -z "$glibcxx_versions" ]]; then
  echo "  (none)"
else
  while IFS= read -r ver; do
    echo "  GLIBCXX_$ver"
    if version_gt "$ver" "$MAX_GLIBCXX"; then
      echo "error: $BIN requires GLIBCXX_$ver (Ubuntu 22.04 / Debian 12 ceiling is $MAX_GLIBCXX)" >&2
      failed=1
    fi
  done <<< "$glibcxx_versions"
fi

if [[ "$failed" -ne 0 ]]; then
  echo "dynamic NEEDED libraries:" >&2
  objdump -p "$BIN" | awk '/NEEDED/ {print "  "$2}' >&2 || true
  exit 1
fi

echo "ok: $(basename "$BIN") stays within glibc $MAX_GLIBC / GLIBCXX $MAX_GLIBCXX"
