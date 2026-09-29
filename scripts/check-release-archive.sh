#!/bin/sh
# Fail the release when an archive does not ship vestige-upgrade.
set -eu
archive=${1:?archive path}
case "$archive" in
  *.zip)
    if ! unzip -Z1 "$archive" | grep -qx 'vestige-upgrade.exe'; then
      echo "missing vestige-upgrade.exe in $archive" >&2
      exit 1
    fi
    ;;
  *.tar.gz)
    if ! tar -tzf "$archive" | grep -qx 'vestige-upgrade'; then
      echo "missing vestige-upgrade in $archive" >&2
      exit 1
    fi
    ;;
  *)
    echo "unknown archive type: $archive" >&2
    exit 1
    ;;
esac
