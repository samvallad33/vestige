# Sanhedrin Test-Integrity Delta Receipts

Receipt Lock proves a narrower claim: a verification command actually ran and
succeeded. Test-integrity deltas are an optional companion receipt for the
stronger claim that the tests still mean what the draft says they mean.

This receipt is intentionally mechanical. It is not a broad correctness oracle
and it does not ask a second model to decide whether the implementation is good.
It records whether the verification artifact changed in ways that should
upgrade, downgrade, or send the verification claim to human review.

## Boundary

Keep these claims separate:

1. **Command receipt:** `cargo test`, `npm test`, `pytest`, or another verifier
   command ran after the relevant edit and exited successfully.
2. **Test-integrity delta:** the tests/specs behind that verifier were not
   removed, skipped, weakened, or replaced after implementation in a way that
   makes the green result less admissible.

A run can have a valid command receipt and still receive a downgraded
integrity decision.

## Optional JSON Shape

```json
{
  "schema": "vestige.sanhedrin.test_integrity_delta.v1",
  "id": "tid_<stable hash>",
  "commandReceiptId": "receipt_<stable hash>",
  "verificationClaim": "All tests passed.",
  "specSource": {
    "contextId": "spec_ctx_04",
    "testFiles": [
      {
        "path": "tests/cart.test.ts",
        "hashBeforeImplementation": "sha256:...",
        "hashAfterVerification": "sha256:..."
      }
    ]
  },
  "implementationContext": "impl_ctx_09",
  "verifierContext": "verify_ctx_02",
  "delta": {
    "testFilesChangedAfterImplementation": true,
    "removedOrDisabledTests": [
      {
        "kind": "skip_or_only",
        "path": "tests/cart.test.ts",
        "line": 42
      }
    ],
    "removedAssertions": 2,
    "weakenedExpectations": [
      {
        "path": "tests/cart.test.ts",
        "from": "throws InvalidCouponError",
        "to": "does not throw"
      }
    ],
    "snapshotChurnWithoutSourceChange": false,
    "coverageDelta": -3.8,
    "mocksReplacingRealBoundary": [
      {
        "module": "PaymentGateway",
        "before": "integration-ish fake",
        "after": "empty stub"
      }
    ]
  },
  "freshVerifier": {
    "commandReceiptId": "receipt_<stable hash>",
    "exitCode": 0,
    "checkedAfterLastRelevantEdit": true
  },
  "decision": "downgraded",
  "reason": "tests passed, but the tests were weakened after implementation"
}
```

## Decisions

- `accepted` — a verifier command succeeded after the last relevant edit and no
  integrity downgrade was detected.
- `downgraded` — the command succeeded, but the tests/specs changed in a way
  that makes the verification claim weaker than stated.
- `needs_human_review` — the delta may be legitimate, but a local mechanical
  check cannot safely classify it. Snapshot updates are a common example.

## Minimal Fixture Suite

These cases are small enough to live as fixtures without turning Sanhedrin into
a correctness judge. Machine-readable examples live in
[`docs/fixtures/sanhedrin-test-integrity-deltas/`](fixtures/sanhedrin-test-integrity-deltas/).

| Case | Input pattern | Expected decision | Why |
| --- | --- | --- | --- |
| unchanged-good | implementation changes source; tests unchanged; fresh verifier succeeds | `accepted` | Green tests are supported by a fresh command receipt and unchanged test artifact. |
| skipped-test | implementation adds `.skip`, `.only`, `#[ignore]`, or equivalent before verifier succeeds | `downgraded` | The command ran, but the claim no longer represents the original test obligation. |
| weakened-assertion | expectation is relaxed after implementation, e.g. `throws InvalidCouponError` -> `does not throw` | `downgraded` | The verifier passed against a weaker assertion than the one available before implementation. |
| justified-snapshot | snapshot changes alongside an intentional source/UI change | `needs_human_review` or `accepted` by policy | Snapshot churn can be valid, but the receipt should make the policy decision explicit. |

## Non-goals

- Do not infer whether the implementation is correct in the world.
- Do not require full semantic diffing before Receipt Lock can operate.
- Do not treat staged evidence or a model explanation as equivalent to a fresh
  command receipt.
- Do not block every test edit. The goal is to keep the verification claim
  honest when the test artifact changed after implementation.
