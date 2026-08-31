#!/usr/bin/env bash
# Start vestige-mcp the way an MCP client does: exec the binary, send
# initialize over stdio, require a protocolVersion in the JSON-RPC reply.
# --version alone is not enough; #174 / #175 crashed on process start.
set -euo pipefail

usage() {
  echo "Usage: $0 <path-to-vestige-mcp>" >&2
  exit 2
}

[[ $# -eq 1 ]] || usage
BIN=$1
[[ -x "$BIN" ]] || { echo "error: not executable: $BIN" >&2; exit 2; }

version_out=$("$BIN" --version)
echo "version: $version_out"
printf '%s\n' "$version_out" | grep -Eq 'vestige-mcp[[:space:]]+[0-9]+\.[0-9]+\.[0-9]+' \
  || { echo "error: --version did not print vestige-mcp <semver>: $version_out" >&2; exit 1; }

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
stderr_log="$tmpdir/stderr.log"

init='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"linux-compat-smoke","version":"0"}}}'

set +e
stdout_out=$(
  printf '%s\n' "$init" \
    | "$BIN" --data-dir "$tmpdir" 2>"$stderr_log"
)
status=$?
set -e

echo "stdio initialize exit=$status"
echo "stderr (truncated):"
tail -n 40 "$stderr_log" || true
echo "stdout:"
printf '%s\n' "$stdout_out"

if [[ "$status" -ne 0 ]]; then
  echo "error: vestige-mcp exited $status during initialize" >&2
  exit 1
fi

printf '%s\n' "$stdout_out" | grep -Eq '"protocolVersion"[[:space:]]*:[[:space:]]*"[^"]+"' \
  || { echo "error: initialize reply missing protocolVersion" >&2; exit 1; }
printf '%s\n' "$stdout_out" | grep -Eq '"name"[[:space:]]*:[[:space:]]*"vestige"' \
  || { echo "error: initialize reply missing serverInfo.name=vestige" >&2; exit 1; }

echo "ok: vestige-mcp started and answered initialize"
