# Decision receipts and controlled evidence replay

Vestige records a receipt for a retrieval so an investigation can inspect the
memory evidence that actually crossed the retrieval boundary. A controlled
replay can then remove selected evidence from that *frozen final context* and
compare the two evidence packs.

This is investigation and provenance support, not a causality engine. Every
replay result carries this product claim boundary:

> Controlled replay shows how the recorded memory context changes when
> specified evidence is withheld. It does not establish that a memory caused an
> agent answer or any real-world outcome.

Use the workflow below with a configured Vestige MCP server. The JSON snippets
are MCP tool arguments: submit each object to the named tool in your MCP
client. They deliberately use disposable fixture text. Do not put secrets,
customer data, prompts, or incident details into a test receipt unless your
local retention and access controls allow that data to remain in the database.

## What kind of receipt is this?

There are two different receipt types:

- A **retrieval receipt** is created by a retrieval tool. Use `recall` for new
  work. It identifies the returned and suppressed memory ids, records a trust
  floor and decay risk, and may have a frozen replay capsule. Only a retrieval
  receipt with an active frozen capsule can be replayed.
- A **synaptic capture receipt** records a synaptic-capture decision. It is not
  a retrieval receipt and does not become a controlled replay just because it
  is inspectable. Its `claimBoundary` describes its own capture evidence.

`receipt` has exactly two actions:

- `get` reads one persisted receipt, its replay-capsule summary, and its local
  signature state.
- `replay` removes named receipt-local slots from an active retrieval capsule.
  It never reruns search, backfills a removed candidate, expands the graph,
  calls a model, or changes memory state.

## Copyable, sanitized incident workflow

Start in an empty test profile or use a dedicated local fixture database. First
store two harmless facts. `smart_ingest` may decide to create, update, or merge
an item, so use the response as the record of what it did.

**Tool:** `smart_ingest`

```json
{
  "content": "Fixture only: the demo dashboard listens on port 5199.",
  "tags": ["docs-fixture", "receipt-demo"]
}
```

**Tool:** `smart_ingest`

```json
{
  "content": "Fixture only: the demo dashboard health endpoint is /healthz.",
  "tags": ["docs-fixture", "receipt-demo"]
}
```

Run a retrieval with a run id that is meaningful only in this local test. A
successful `recall` response includes `receiptId` and an inline `receipt` when
receipt persistence succeeds. Save the generated `receiptId`; it is different
on every run.

**Tool:** `recall`

```json
{
  "query": "fixture dashboard port",
  "runId": "docs_receipt_fixture"
}
```

If the response reports receipt persistence as unavailable, stop here and use
the troubleshooting section below. A successful retrieval alone is not a
durable receipt claim.

Inspect the stored artifact, replacing the placeholder with the `receiptId`
returned by `recall`.

**Tool:** `receipt`

```json
{
  "action": "get",
  "receipt_id": "<receiptId from recall>"
}
```

For a replayable retrieval, inspect these fields:

- `receipt.retrieved`, `receipt.suppressed`, `trust_floor`, and `decay_risk`
  describe the recorded retrieval receipt.
- `replayCapsule.privacyState` must be `active` and
  `replayCapsule.replayable` must be `true`.
- `replayCapsule.items` is the final, ranked evidence pack. It exposes opaque
  `evidenceSlot` values such as `evidence_1`, plus rank, token estimate, trust
  score, and decay risk. It does not expose the memory text in the replay
  projection.
- `claimBoundary` is the limit for this receipt type. Keep it with any exported
  investigation note.

Withhold one of the listed slots. This call is idempotent for the same source
receipt and slot set; a retry can return the same replay id and receipt id.

**Tool:** `receipt`

```json
{
  "action": "replay",
  "receipt_id": "<receiptId from recall>",
  "withheld_slots": ["evidence_1"]
}
```

Compare `result.baseline` with `result.counterfactual`, then record
`result.replayInfluence` and `result.withheldSlots`. The comparison is a
structural answer to *what changed in this recorded evidence pack when this
slot was absent?* It is not evidence that the slot caused an agent response,
the incident, or a real-world result. The replay creates an audit record but
does not mutate the memory state.

## Signature status and its limit

Signing is optional. Without an operator-provisioned signing key, `receipt`
returns `attestation.status: "legacy_unsigned"`; that is expected and means no
DSSE envelope exists for that receipt.

For a configured signer, `receipt` returns `attestation.status: "signed_v1"`
and the local verification fields, including `signatureValid`,
`canonicalPayload`, `receiptBindingValid`, chain-row checks, and the verified
key id/fingerprint. A locally valid DSSE signature proves that the stored
envelope verifies against the registered local key and binds to the receipt
payload that Vestige checked. It does **not** prove external anchoring, trusted
time, truth, completeness, or non-equivocation.

The signer is intentionally opt-in. The server reads both of these environment
variables only after the operator has provisioned a seed sidecar and registered
the matching active Ed25519 public key:

```text
VESTIGE_RECEIPT_SIGNING_KEY_ID
VESTIGE_RECEIPT_SIGNING_SEED_PATH
```

Current public MCP and CLI surfaces do not expose a signing-key provisioning or
registration command. Do not invent a database write or point either variable
at an arbitrary file. An operator integration must provision the seed and
register the matching key before enabling both variables. A partial
configuration, unreadable sidecar, unregistered key, key mismatch, or inactive
key fails receipt persistence closed rather than quietly emitting an unsigned
receipt.

## Privacy, retention, and deletion

Receipts are local database records. Retrieval receipts include memory ids and
other retrieval metadata, so treat them as incident artifacts and limit who can
read the local database and its backups. Replay output is more restrictive: it
uses receipt-local slots and summary fields rather than raw memory text,
queries, prompts, or model output. That is not a substitute for data
classification or a retention policy.

If a source memory is suppressed or purged, its replay capsule becomes
non-replayable under its privacy state. Do not infer that a replay remains
available after a lifecycle operation.

`memory` action `purge` removes canonical content and embeddings after an
explicit `confirm: true`. The current public purge response is a
`legacy_audited_purge` with `unlearning.verdict: "incomplete"`: it retains only
opaque audit/sync markers and limited metadata. It does **not** establish
complete machine unlearning. Verify and remove copies under your own backups,
exports, sync systems, and incident tooling according to their separate
retention controls.

## Durability and backups

Persistent SQLite stores use WAL. The default `hardened` profile uses SQLite
WAL with `synchronous = FULL`; on macOS it also enables `fullfsync` and
`checkpoint_fullfsync`. Select it explicitly when starting the server if you
want the setting visible in your deployment configuration:

```bash
VESTIGE_SQLITE_DURABILITY=hardened vestige-mcp
```

`balanced` preserves WAL with `synchronous = NORMAL` for operators who
explicitly accept a larger power-loss window in exchange for lower write
latency:

```bash
VESTIGE_SQLITE_DURABILITY=balanced vestige-mcp
```

The `maintain` tool can make a consistent SQLite backup that includes committed
WAL state:

**Tool:** `maintain`

```json
{
  "action": "backup"
}
```

It returns the generated backup path and size. Copy that file using storage
appropriate to its contents. Neither profile nor a successful backup makes a
physical power-loss guarantee: survival still depends on the operating system,
filesystem, controller, storage device, and their honoring completed flush
requests.

## Troubleshooting

| What you see | What to do |
| --- | --- |
| `Receipt persistence is temporarily unavailable` or no `receiptId` | Treat the retrieval as not durably receipted. Retry after resolving the local storage problem; do not construct a receipt id yourself. |
| `Receipt '<id>' was not found` | Use the `receiptId` from the retrieval that ran against this same local database. Receipt ids are not portable handles by themselves. |
| Replay says evidence is unavailable or not replayable | Re-run a new retrieval after checking whether the original evidence was suppressed, purged, or came from a legacy/non-retrieval receipt. |
| Signing was enabled and receipt persistence fails | Check that both signing variables are set, the sidecar is readable and secure, and the matching public key was registered and active by the operator integration. Do not fall back to calling the receipt signed. |

## Related references

- [MCP tool registration and the `receipt` schema](../crates/vestige-mcp/src/server.rs)
- [Receipt tool implementation](../crates/vestige-mcp/src/tools/receipt.rs)
- [Storage behavior](STORAGE.md)
- [Configuration reference](CONFIGURATION.md)
