#!/usr/bin/env bash
# Fails when the release GitHub marks "latest" cannot serve the README install
# command.
#
# The README, the release notes and `vestige update` all fetch
#   https://github.com/<repo>/releases/latest/download/vestige-mcp-<target>.<ext>
# so whichever release holds the "latest" flag must carry every platform
# archive and its checksum. On 2026-10-01 a non-product release (an Operator
# Lite tag) took the flag and that URL returned 404 until the flag was moved
# back.
#
# Usage: check-latest-release.sh [owner/repo] [tag]
#   owner/repo  defaults to samvallad33/vestige
#   tag         check this release instead of the one marked latest
#
# Needs the GitHub CLI (`gh`), authenticated or with GH_TOKEN set. Read-only.
set -euo pipefail

repo="${1:-samvallad33/vestige}"
tag="${2:-}"

if [ -n "$tag" ]; then
  endpoint="repos/$repo/releases/tags/$tag"
else
  endpoint="repos/$repo/releases/latest"
fi

# First line: the tag. Remaining lines: asset names.
listing="$(gh api "$endpoint" --jq '.tag_name, (.assets[].name)')"

release_tag=""
assets=()
while IFS= read -r line; do
  if [ -z "$release_tag" ]; then
    release_tag="$line"
  elif [ -n "$line" ]; then
    assets+=("$line")
  fi
done <<<"$listing"

if [ -z "$release_tag" ]; then
  echo "::error::could not read a release from $endpoint"
  exit 1
fi

required=(
  vestige-mcp-aarch64-apple-darwin.tar.gz
  vestige-mcp-x86_64-apple-darwin.tar.gz
  vestige-mcp-x86_64-unknown-linux-gnu.tar.gz
  vestige-mcp-aarch64-unknown-linux-gnu.tar.gz
  vestige-mcp-x86_64-pc-windows-msvc.zip
)

missing=()
for archive in "${required[@]}"; do
  for want in "$archive" "$archive.sha256"; do
    found=0
    # "${assets[@]+...}" keeps bash 3.2 (macOS) quiet when a release has no assets.
    for have in ${assets[@]+"${assets[@]}"}; do
      if [ "$have" = "$want" ]; then
        found=1
        break
      fi
    done
    if [ "$found" -eq 0 ]; then
      missing+=("$want")
    fi
  done
done

status=0

# A product release is tagged vMAJOR.MINOR.PATCH, optionally with a suffix.
if [[ ! "$release_tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.]+)?$ ]]; then
  echo "::error::release '$release_tag' is not a product release tag (vX.Y.Z) but is the one being served"
  status=1
fi

if [ "${#missing[@]}" -gt 0 ]; then
  echo "::error::release '$release_tag' is missing ${#missing[@]} required asset(s):"
  for name in "${missing[@]}"; do
    echo "  - $name"
  done
  status=1
fi

if [ "$status" -ne 0 ]; then
  cat <<REMEDY

The README install URL (releases/latest/download/...) and 'vestige update' are
broken while this holds. To repair:

  gh release edit <newest vX.Y.Z tag> --repo $repo --latest

and publish non-product releases (Operator Lite, benchmarks, launches) with
--latest=false so they never take the flag.
REMEDY
  exit 1
fi

echo "ok: release '$release_tag' carries all ${#required[@]} platform archives and their checksums"
