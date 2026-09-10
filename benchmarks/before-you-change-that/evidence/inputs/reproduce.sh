#!/bin/sh
set -eu

RUSTC_BIN="${RUSTC:-@recorded-home@/.cargo/bin/rustc}"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/intention-fixture.XXXXXX")"
trap 'rm -rf "$BUILD_DIR"' EXIT HUP INT TERM

"$RUSTC_BIN" --edition=2021 "$SCRIPT_DIR/public_checks.rs" -o "$BUILD_DIR/public-checks"
"$BUILD_DIR/public-checks" "$BUILD_DIR/state"
