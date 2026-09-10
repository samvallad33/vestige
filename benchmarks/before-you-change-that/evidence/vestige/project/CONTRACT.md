# Synthetic reconciliation repair contract

Repair the reconciliation incident while preserving the unrelated export path.
The ledger journal is append-only. A reversal may arrive before its referenced
charge, retries must not duplicate effects, and a stale in-progress marker must
recover by reloading durable files. Accepted state and export bytes must remain
deterministic. There is no network, provider, database, or real payment effect.

You may edit only `src/ledger/reconcile.rs` and `runtime.conf`. Create
`DECISION.json` using `decision.schema.json`. Cite the connected historical
path and explain rejected alternatives. Preserve all other fixture files.
