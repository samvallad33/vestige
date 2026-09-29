#!/bin/bash
# claim-gate.sh — receipt-bound claim gate (PR 4, fail-closed).
#
# The enforcement client for the deterministic claim checker. It is a thin,
# pure-shell relay: it reads ONE structured claim JSON document on stdin,
# posts it through `vestige claim-check --json -`, and
#
#   - exits 0 (allow)  ONLY on an explicit "allow":true verdict;
#   - exits 2 (BLOCK)  on "allow":false (the deterministic checker denied:
#     no cited TOOL_RESULT receipt matched the claim kind's family table,
#     or a cited receipt failed, or nothing was cited);
#   - exits 2 (BLOCK)  on EVERY unreachable or error condition — vestige
#     missing, the store unavailable, malformed input, unparseable verdict.
#
# Fail-closed by construction (H8): anything the gate cannot verify blocks.
# Opting out means not installing this hook; there is no verdict bypass.
#
# No model calls, no LLM endpoint, no regex over prose (H2/H3): the claim
# kind is a FIELD of the structured JSON the caller posts, and the only
# pattern matching here is exact-token matching on the verdict's own
# structured output. The verdict itself is a signed CLAIM_VERDICT receipt
# any verifier can replay (`rederive_claim_verdicts`).
#
# Wiring: point the host's Stop/claim hook at this script. The claim
# document the caller posts is:
#   {"session": "...", "claim_kind": "tests_pass", "target_handle": "...",
#    "cited_record_ids": [<TOOL_RESULT frame seqs>]}
# where the cited seqs come from `vestige record-tool --phase result`.

set -u

VESTIGE_BIN="${VESTIGE_BIN:-vestige}"
# Optional STRATA log directory override (same default as the CLI:
# <data-dir>/strata). Pointing the gate at a store only changes WHERE the
# receipts live, never whether a claim is allowed; absent/unwritable
# stores still fail closed through the exit-status path below.
STRATA_DIR_ARGS=()
if [ -n "${VESTIGE_STRATA_DIR:-}" ]; then
  STRATA_DIR_ARGS=(--strata-dir "$VESTIGE_STRATA_DIR")
fi

CLAIM_JSON="$(cat)"

# An empty claim is never an allow (fail-closed, H8).
if [ -z "$CLAIM_JSON" ]; then
  echo "[claim-gate] BLOCK: no claim JSON on stdin (fail-closed)" >&2
  exit 2
fi

if ! command -v "$VESTIGE_BIN" >/dev/null 2>&1; then
  echo "[claim-gate] BLOCK: claim gate unavailable (fail-closed): '$VESTIGE_BIN' not found" >&2
  exit 2
fi

VERDICT="$(printf '%s\n' "$CLAIM_JSON" | "$VESTIGE_BIN" claim-check --json - "${STRATA_DIR_ARGS[@]}" 2>/dev/null)"
STATUS=$?

# Any nonzero exit is a block: the CLI's contract is 0 = Allow, nonzero =
# Deny or error, and the hook cannot allow what it could not verify.
if [ "$STATUS" -ne 0 ]; then
  echo "[claim-gate] BLOCK: claim gate unavailable (fail-closed): exit $STATUS" >&2
  if [ -n "$VERDICT" ]; then
    printf '%s\n' "$VERDICT" >&2
  fi
  exit 2
fi

case "$VERDICT" in
  *'"allow":true'*)
    # Explicit allow verdict from the deterministic checker.
    exit 0
    ;;
  *'"allow":false'*)
    echo "[claim-gate] BLOCK: claim denied by the deterministic checker —" \
      "run the matching tool and post its receipt (record-tool), or rewrite" \
      "without the claim." >&2
    printf '%s\n' "$VERDICT" >&2
    exit 2
    ;;
  *)
    echo "[claim-gate] BLOCK: claim gate unavailable (fail-closed): unparseable verdict" >&2
    exit 2
    ;;
esac
