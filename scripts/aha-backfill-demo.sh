#!/usr/bin/env bash
# A 90-second, isolated Vestige demo. Choose one pitch-ready scenario:
# timeout (default), migration, or permissions.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
demo_dir="$(mktemp -d -t vestige-backfill-demo.XXXXXX)"
cleanup() { rm -rf "$demo_dir"; }
trap cleanup EXIT
scenario="${1:-timeout}"

case "$scenario" in
  timeout)
    title="Configuration failure"
    cause="Changed API_TIMEOUT from 15 seconds to 2 seconds to speed up cold starts."
    cause_tags="API_TIMEOUT,deploy-env"
    distractor="The billing service returned a 500 Internal Server Error last month."
    distractor_tags="billing-service"
    failure="The auth service crashed with a 500 during a login surge."
    failure_tags="auth-service,API_TIMEOUT,crash"
    ;;
  migration)
    title="Data migration failure"
    cause="Approved CUSTOMER_ID as text for the partner order-import mapping."
    cause_tags="CUSTOMER_ID,data-migration"
    distractor="CRM migration failure: a nightly sync created duplicate customer profiles."
    distractor_tags="crm-import,duplicates"
    failure="ETL bug: a supplier feed assigned historical order lines to two tenants."
    failure_tags="CUSTOMER_ID,import"
    ;;
  permissions)
    title="Stale access failure"
    cause="Set PERMISSION_CACHE_TTL to 30 minutes to reduce authorization lookups."
    cause_tags="PERMISSION_CACHE_TTL,cache"
    distractor="Password-reset incident: a vendor's MFA token was rejected."
    distractor_tags="admin-access,permissions"
    failure="Security bug: a former employee retained billing-export capability after offboarding."
    failure_tags="PERMISSION_CACHE_TTL,offboarding"
    ;;
  *)
    echo "Choose one scenario: timeout, migration, or permissions." >&2
    exit 2
    ;;
esac

echo "Building the local Vestige demo binary..."
cargo build --quiet --manifest-path "$repo_root/Cargo.toml" -p vestige-mcp --bin vestige
vestige="$repo_root/target/debug/vestige"

run() { "$vestige" --data-dir "$demo_dir" "$@"; }

echo
echo "=== $title ==="
echo "1/3  Three facts from an incident timeline"
run ingest "$cause" --tags "$cause_tags" --node-type decision --ago-days 3
run ingest "$distractor" --tags "$distractor_tags" --node-type event --ago-days 20
run ingest "$failure" --tags "$failure_tags" --node-type event

echo
echo "2/3  The ordinary search result looks relevant, but is not the cause"
echo "3/3  Backfill follows the shared incident evidence backward"
run backfill --contrast

echo
echo "Aha: Vestige did not merely retrieve the most similar error. It surfaced an"
echo "earlier decision that shares the incident evidence, then promoted that candidate"
echo "for future retrieval."
