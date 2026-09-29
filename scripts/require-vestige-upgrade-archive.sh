#!/usr/bin/env bash
# Fail the release build when a target archive does not ship vestige-upgrade.
set -euo pipefail

archive="${1:?usage: require-vestige-upgrade-archive.sh <archive>}"
if [ ! -f "$archive" ]; then
  echo "::error::missing release archive: $archive"
  exit 1
fi

case "$archive" in
  *.zip)
    need="vestige-upgrade.exe"
    members="$(unzip -Z1 "$archive")"
    ;;
  *.tar.gz | *.tgz)
    need="vestige-upgrade"
    members="$(tar -tzf "$archive")"
    ;;
  *)
    echo "::error::unknown archive type: $archive"
    exit 1
    ;;
esac

if ! printf '%s\n' "$members" | sed 's|^\./||' | grep -qx "$need"; then
  echo "::error::$archive does not contain $need"
  printf '%s\n' "$members"
  exit 1
fi

echo "$archive contains $need"
