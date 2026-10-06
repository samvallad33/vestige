# Receipts and replay

> Written for Vestige 4.2.0.

On a Strata log every write is proposed, checked by the gate, and admitted as an effect. The write returns, or can be resolved to, a receipt. `receipt` `get` shows what the write did. `receipt` `replay` re-derives the state from the log and reports any mismatch.

A receipt proves what the log recorded. It does not prove that the content is true, and it does not say what an agent did with the record afterward.

The JSON snippets below are MCP tool arguments: submit each object to the named tool in your MCP client. They use disposable fixture text. Do not put secrets or customer data in a test record. The log is append-only, so a record you save stays on it.

## What a receipt is

A receipt id looks like `eff-` followed by 16 hex digits. It names one admitted effect. You get one in these ways:

- `memory` actions `promote`, `demote` and `edit`, `suppress`, `intention`, and `ghostlink` `weave` return `receiptId`.
- `smart_ingest` returns `nodeId`. Pass that id as `receipt_id`, and `receipt` resolves it to the receipt of the write that created the record.
- `receipt` accepts either an `eff-` id or a memory id.

## Workflow

Save a harmless record. **Tool:** `smart_ingest`

```json
{
  "content": "Fixture only: the demo dashboard listens on port 5199.",
  "node_type": "fact",
  "tags": ["docs-fixture", "receipt-demo"]
}
```

Take `nodeId` from the response, then read the receipt. **Tool:** `receipt`

```json
{
  "action": "get",
  "receipt_id": "<nodeId from smart_ingest>"
}
```

The response has these parts:

- `receipt.receipt_id`: the `eff-` id.
- `receipt.mutations[0].note`: `digest=<blake3 hex> event_seq=<n> effect_seq=<n>`, plus `rating=` for a promote or demote, `rule=edit` for an edit, and `edge=<kind> target=<id>` for a link. The digest is of the admitted payload. The record's text is not in the receipt.
- `attestation.status`: `strata_effect`. `attestation.verification` says whether the log verified (`locallyVerified`, `chainValid`, `gateAllowed`, `payloadDigest`, and how many segments, sealed segments and frames were checked).
- `replayCapsule`: always `null` on a Strata log. See [What is not in 4.x](#what-is-not-in-4x).
- `claimBoundary`: the limit of this receipt. Keep it with any note you export.

Before it looks anything up, `get` verifies the whole log. A damaged log halts there with an error. It never attests a receipt from a log it could not verify.

Now replay it. **Tool:** `receipt`

```json
{
  "action": "replay",
  "receipt_id": "<nodeId or eff- id>"
}
```

Replay is read-only. It rebuilds state from the log, compares it with the recorded digest, and returns `kind: "strata"`, `readOnly: true`, `matched`, a list of `mismatches`, `nodeId`, `receiptId`, `stateDigest`, `replayedDigest` and `frames`. The same call returns the same answer, and it does not write to the log. On a tampered log it fails and does not repair anything.

A record imported by the v3 upgrade has no admitted write on the new log to replay. Replay says so (`imported_from_v3`). The signed migration receipt covers it, and `vestige strata-verify <data-dir>` checks that receipt.

## What the checks establish

`get` and `replay` establish that:

- every segment's frame chain is intact, and every sealed segment's signed trailer verifies;
- the effect cites an Allow decision from the gate;
- the payload digest matches the admitted frame; and
- replaying the log yields the state the receipt names.

They do not establish an external timestamp, that the content is true, or that the log is complete beyond what its chain shows. The signing key, `receipt-signing.key`, sits in the data directory beside `log/`. Whoever can read that key can sign. A Strata receipt is not a per-receipt signed envelope; the signature is on the sealed segment.

## What is not in 4.x

v3 recorded a receipt for each retrieval and could replay it with named evidence withheld. A Strata log does not do that:

- `recall` by handle returns the records and their one-hop neighbors. It does not return a `receiptId`.
- `receipt` `replay` with `withheld_slots` is refused: "withheld_slots do not apply to a Strata log replay". Counterfactual replay of a frozen evidence pack has no equivalent here.
- `receipt` `save_walk` returns `unavailable_in_4_0`: walk receipts are not recorded on a Strata log yet. `causal_walk` returns the path of recorded edges with each answer instead.
- The v3 signing variables `VESTIGE_RECEIPT_SIGNING_KEY_ID` and `VESTIGE_RECEIPT_SIGNING_SEED_PATH` apply to the v3 engine only.

## Privacy, retention and erasure

Receipts live in the Strata log, so they have the log's retention: forever. They name record ids and payload digests, not text. A record you suppress is hidden from every read, so a lookup by its memory id finds nothing, and its bytes stay on the log. Nothing is erased. `purge` and `memory` actions `purge` and `delete` return `unavailable_in_4_0`, and real erasure is planned as crypto-erasure ([#402](https://github.com/samvallad33/vestige/issues/402)).

## Durability and backups

The single-writer lock is an OS lock that the kernel releases when its holder exits, and a full disk is an error to the caller. `vestige backup <new-folder>` seals the log and copies it while your agents run. `maintain` action `backup` writes into `<data-dir>/backups/`. Neither makes a physical power-loss guarantee: survival still depends on the operating system, the filesystem and the storage device honoring completed flush requests. Restore by stopping Vestige and copying the backup's `log/` into the data directory. Details: [STORAGE.md](STORAGE.md).

## Troubleshooting

| What you see | What to do |
| --- | --- |
| `Receipt '<id>' was not found` | Use the id from a write on this same data directory, or a memory id. A suppressed or retired record has no visible receipt. |
| `imported_from_v3: ...` | The record came from the upgrade. Run `vestige strata-verify <data-dir>` instead. |
| An error that mentions `verification failed`, `strata halt` or a hash mismatch | The log failed verification. Do not trust any receipt from it. Restore a backup, and run `vestige strata-verify`. |
| `withheld_slots do not apply to a Strata log replay` | Drop `withheld_slots`. Counterfactual replay is not available on Strata. |

## Related references

- [MCP tool registration and the `receipt` schema](../crates/vestige-mcp/src/server.rs)
- [Receipt tool implementation](../crates/vestige-mcp/src/tools/receipt.rs)
- [Storage behavior](STORAGE.md)
- [Configuration reference](CONFIGURATION.md)
