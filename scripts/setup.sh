#!/usr/bin/env sh
# Configure the locally installed Vestige memory server for detected MCP clients.
# Usage: curl -fsSL https://raw.githubusercontent.com/samvallad33/vestige/main/scripts/setup.sh | sh
# Flags: curl -fsSL https://raw.githubusercontent.com/samvallad33/vestige/main/scripts/setup.sh | sh -s -- --dry-run
set -eu

REPO="samvallad33/vestige"
INIT_URL="${VESTIGE_INIT_URL:-https://raw.githubusercontent.com/${REPO}/main/packages/vestige-init/bin/init.js}"
INSTALL_DIR="${VESTIGE_INSTALL_DIR:-$HOME/.local/bin}"
INIT_VERSION="${VESTIGE_INIT_VERSION:-7.1.0}"
CACHE_DIR="${VESTIGE_SETUP_CACHE_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/vestige/setup}"

fail() { printf '%s\n' "vestige setup: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "requires $1"; }

need curl
need node

binary="${VESTIGE_MCP_PATH:-$INSTALL_DIR/vestige-mcp}"
[ -x "$binary" ] || fail "vestige-mcp is not installed at $binary; install it first with https://github.com/${REPO}#install"

mkdir -p "$CACHE_DIR"
init_path="$CACHE_DIR/vestige-init.js"
staged_init="$(mktemp "$CACHE_DIR/.vestige-init.XXXXXX")"
cleanup() { rm -f "$staged_init"; }
trap cleanup EXIT HUP INT TERM

printf '%s\n' "Starting the Vestige setup wizard..."
curl --fail --location --silent --show-error "$INIT_URL" --output "$staged_init"
chmod 600 "$staged_init"
mv -f "$staged_init" "$init_path"

if [ "${VESTIGE_SETUP_YES:-}" = "1" ] && [ "$#" -eq 0 ]; then
  set -- --all --yes
fi

if [ "$#" -eq 0 ]; then
  [ -r /dev/tty ] || fail "run this from an interactive terminal, or use --dry-run, --client <name>, or --all --yes"
  VESTIGE_MCP_PATH="$binary" VESTIGE_INIT_VERSION="$INIT_VERSION" node "$init_path" --interactive </dev/tty
else
  VESTIGE_MCP_PATH="$binary" VESTIGE_INIT_VERSION="$INIT_VERSION" node "$init_path" "$@"
fi
