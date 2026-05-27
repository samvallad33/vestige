#!/bin/bash
set -euo pipefail

VERSION="${1:-2.1.23}"
REPO="samvallad33/vestige"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "Building Vestige MCPB v${VERSION}..."

# Create server directory
mkdir -p server

die() {
  echo "error: $*" >&2
  exit 1
}

verify_checksum() {
  local archive="$1"
  local checksum="$2"
  local expected actual
  expected="$(awk '{print tolower($1)}' "$checksum")"
  case "$expected" in
    [0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) ;;
    *) die "invalid checksum file: $checksum" ;;
  esac
  [ "${#expected}" -eq 64 ] || die "invalid checksum length for $archive"
  actual="$(shasum -a 256 "$archive" | awk '{print tolower($1)}')"
  [ "$actual" = "$expected" ] || die "checksum mismatch for $(basename "$archive")"
}

validate_member() {
  local member="${1#./}"
  shift
  case "$member" in
    ""|/*|../*|*/../*|*"/.."|*":"*) die "unsafe archive member: $member" ;;
  esac
  for expected in "$@"; do
    [ "$member" = "$expected" ] && return 0
  done
  die "unexpected archive member: $member"
}

validate_tar() {
  local archive="$1"
  shift
  while IFS= read -r member; do
    [ -n "$member" ] || continue
    validate_member "$member" "$@"
  done < <(tar -tzf "$archive")
}

validate_zip() {
  local archive="$1"
  shift
  while IFS= read -r member; do
    [ -n "$member" ] || continue
    validate_member "$member" "$@"
  done < <(unzip -Z1 "$archive")
}

download_release_asset() {
  local name="$1"
  local archive="$TMPDIR/$name"
  local checksum="$TMPDIR/$name.sha256"
  curl -fsSL "https://github.com/${REPO}/releases/download/v${VERSION}/${name}" -o "$archive"
  curl -fsSL "https://github.com/${REPO}/releases/download/v${VERSION}/${name}.sha256" -o "$checksum"
  verify_checksum "$archive" "$checksum"
  printf '%s\n' "$archive"
}

# Download macOS ARM64
echo "Downloading macOS ARM64 binary..."
ARCHIVE="$(download_release_asset "vestige-mcp-aarch64-apple-darwin.tar.gz")"
validate_tar "$ARCHIVE" vestige-mcp vestige vestige-restore
tar -xzf "$ARCHIVE" -C server
mv server/vestige-mcp server/vestige-mcp-darwin-arm64
mv server/vestige server/vestige-darwin-arm64
rm -f server/vestige-restore

# Download Linux x64
echo "Downloading Linux x64 binary..."
ARCHIVE="$(download_release_asset "vestige-mcp-x86_64-unknown-linux-gnu.tar.gz")"
validate_tar "$ARCHIVE" vestige-mcp vestige vestige-restore
tar -xzf "$ARCHIVE" -C server
mv server/vestige-mcp server/vestige-mcp-linux-x64
mv server/vestige server/vestige-linux-x64
rm -f server/vestige-restore

# Download Windows x64
echo "Downloading Windows x64 binary..."
ARCHIVE="$(download_release_asset "vestige-mcp-x86_64-pc-windows-msvc.zip")"
validate_zip "$ARCHIVE" vestige-mcp.exe vestige.exe vestige-restore.exe
unzip -q "$ARCHIVE" -d server
mv server/vestige-mcp.exe server/vestige-mcp-win32-x64.exe
mv server/vestige.exe server/vestige-win32-x64.exe
rm -f server/vestige-restore.exe

# Make executable
chmod +x server/*

echo "Binaries downloaded. Run 'mcpb pack' to create bundle."
