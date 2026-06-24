#!/usr/bin/env bash
# check-no-private-cloud.sh — Fail if private Vestige Cloud *service* code leaks
# into this public repository.
#
# Vestige Cloud is split:
#   - PUBLIC client  (this repo): a thin HTTP sync backend that only moves
#                    encrypted bytes and presents an opaque bearer token.
#                    Legitimate. Reads VESTIGE_CLOUD_ENDPOINT / _SYNC_KEY /
#                    _ENCRYPTION_KEY env vars on the client side.
#   - PRIVATE service (separate repo, no public remote): the hosted blob
#                    service that owns sync-key -> namespace mapping, per-user
#                    isolation, Lemon Squeezy billing webhooks, and
#                    transactional email. This MUST NEVER be committed here.
#
# This guard scans only tracked files (git grep) for distinctive *service*
# signatures — module headers, billing/provider internals, and server-side
# auth/namespace mapping — chosen so the legitimate public client does NOT
# match. It deliberately does NOT match the VESTIGE_CLOUD_* client env-var
# prefix, which the public client uses legitimately.
set -u

cd "$(git rev-parse --show-toplevel)" || {
  echo "check-no-private-cloud: not inside a git repository" >&2
  exit 2
}

# Distinctive private-service signatures. Each is a fixed string (grep -F via
# -e with --fixed-strings) that appears in the private vestige-cloud service
# and must never appear in this public repo. Keep these specific.
PATTERNS=(
  # Service crate identity / entrypoint
  'name = "vestige-cloud"'
  'Vestige Cloud — hosted managed-sync blob service'
  # Service module headers
  'Sync-key store and authentication'
  'Blob storage for the managed-sync service'
  'Lemon Squeezy webhook handling'
  'Transactional email delivery via Resend'
  # Billing / provider internals (server-only)
  'LEMONSQUEEZY_WEBHOOK_SECRET'
  'lemonsqueezy'
  # Server-side sync-key -> namespace mapping (the authoritative mapping that
  # by design lives ONLY in the hosted service, never the client)
  'sync_keys SET key_hash'
)

# Files this guard itself lives in / references the patterns must be excluded,
# or it would always flag itself. Exclude this script and any allowlist doc.
EXCLUDES=(
  ':(exclude)scripts/check-no-private-cloud.sh'
  ':(exclude).github/workflows/guard-no-private-cloud.yml'
)

violations=0
report=""

for pat in "${PATTERNS[@]}"; do
  # -I skip binary, -n line numbers, -F fixed string, -i case-insensitive.
  if hits=$(git grep -Ini -F -e "$pat" -- "${EXCLUDES[@]}" 2>/dev/null); then
    if [ -n "$hits" ]; then
      violations=$((violations + 1))
      report+=$'\n'"  ✗ private-service marker found: \"$pat\""$'\n'
      report+="$(printf '%s\n' "$hits" | sed 's/^/      /')"$'\n'
    fi
  fi
done

if [ "$violations" -ne 0 ]; then
  echo "════════════════════════════════════════════════════════════════════"
  echo "  PRIVATE CLOUD SERVICE CODE DETECTED IN PUBLIC REPO"
  echo "════════════════════════════════════════════════════════════════════"
  echo "$report"
  echo "════════════════════════════════════════════════════════════════════"
  echo "  The hosted Vestige Cloud service (billing, namespace mapping,"
  echo "  per-user isolation) must live ONLY in the private repo, never here."
  echo "  Remove the file(s) above from this repo. If a match is a false"
  echo "  positive, refine the pattern in scripts/check-no-private-cloud.sh."
  echo "════════════════════════════════════════════════════════════════════"
  exit 1
fi

echo "check-no-private-cloud: OK — no private Vestige Cloud service code in public repo"
exit 0
