#!/usr/bin/env bash
# hardcode_scan_selftest.sh: prove scripts/hardcode_scan.sh actually rejects a
# hand-typed open-class list (bad fixture -> nonzero), accepts closed-class
# grammar (good fixture -> zero), and that the real tree is clean.
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
if HARDCODE_SCAN_ROOT=scripts/hardcode_scan_fixtures/bad bash scripts/hardcode_scan.sh >/dev/null 2>&1; then
  echo "selftest FAIL: scan accepted the fake open-class list fixture"; fail=1
else
  echo "selftest ok: bad fixture rejected (nonzero exit)"
fi
if HARDCODE_SCAN_ROOT=scripts/hardcode_scan_fixtures/good bash scripts/hardcode_scan.sh >/dev/null 2>&1; then
  echo "selftest ok: closed-class fixture accepted"
else
  echo "selftest FAIL: scan rejected the closed-class fixture"; fail=1
fi
if bash scripts/hardcode_scan.sh; then
  echo "selftest ok: real tree clean"
else
  echo "selftest FAIL: real tree has findings"; fail=1
fi
exit $fail
