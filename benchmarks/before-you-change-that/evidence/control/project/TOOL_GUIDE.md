# Local task environment

This is an original fictional Rust fixture with local append-only files. Run
`./reproduce.sh`. The deliberately unordered `fixture.json` archive is identical
for every arm, and its timestamps are synthetic story chronology.

Only `src/ledger/reconcile.rs` and `runtime.conf` are application-editable.
Create `DECISION.json` separately. Preserve the archive, `intention.json`, this
guide, `CONTRACT.md`, `decision.schema.json`, `public_checks.rs`,
`observation_adapter.rs`, and `reproduce.sh`. The generic adapter contains no
hidden cases or expected answers; the independent evaluator remains external.

## Context intentions and ordinary history

The same prospective instruction exists in the ordinary archive and in
`intention.json`. Every arm can retrieve the ordinary record. Inspect callable
tools even when resources are empty. If a native Vestige service is connected,
check pending intentions before touching the relevant source with
`context.file` set to `src/ledger/reconcile.rs`. The trigger uses the current
file string against the saved `file_pattern`; an unrelated path should not
match. Keep the intention unfulfilled during the model run. The coordinator
may mark it complete only after independent application tests pass.

When native Recall is available, complete at least one successful Recall audit.
Every Recall call must use the exact scope `hardened-demo` and
`includeCrossScope: false`. Use native UUIDs returned by Recall or other native
responses for later direct-ID `memory` reads; fixture labels are not native
UUIDs. For every connected-service arm, every citation used as accepted
evidence, in the causal path, or in a rejected hypothesis must appear in native
response data as well as the common archive. Other memory services may use
search, list, tags, or metadata. A runbook phrase, matching number, intention
delivery, or receipt is context rather than proof of the current cause.
