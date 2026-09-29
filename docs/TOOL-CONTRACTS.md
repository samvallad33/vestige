# Tool contracts and progressive discovery

Vestige advertises eighteen MCP tools. Most are action multiplexers: `recall`
takes exact handles, `memory_status` uses `view`, and tools such as `memory`
and `dedup` use `action`. An agent should choose the action needed for its
current task. There is no requirement to call every tool during every session.

## Discover the installed contract

```json
{"name":"memory_status","arguments":{"view":"tools"}}
```

This returns the same names, descriptions, action enums, defaults and tool-level
annotations as the running server's `tools/list`. Request one complete schema:

```json
{"name":"memory_status","arguments":{"view":"tools","tool":"maintain"}}
```

The inventory includes compiled feature flags. They describe the build, not
runtime readiness: an upstream connector still needs configuration. 4.0 has no
embedding or vector-search features at all — duplicate scanning is exact
content identity, and `recall` is exact-handle only (see
[HANDLES.md](HANDLES.md)). Use health to inspect runtime state.

The MCP annotations cover the whole tool. `receipt` replay persists a durable
replay artifact, so neither mixed tool is read-only. Receipt replay reuses its
durable identity on retry. `suppress` compounds and is not idempotent.
Client-side approval settings and the server's review mode still determine
whether a mutation runs. A successful protocol response can report
`pendingReview=true` and `success=false`; inspect the application result
before claiming a change.

## Investigation before reinforcement

`causal_walk` (the successor to the retired `backfill` tool) investigates a
failure from explicit start points — a failing test, a stack frame, a CI run,
a logged write, a version range — through exact mechanism edges to suspect
change records. With no start point it returns `needs_report`; it never
guesses. The default preview persists nothing; every candidate is a hypothesis
and `causality_verified` stays false.

```json
{"name":"causal_walk","arguments":{"scope":"project","start_points":[{"kind":"failing_test","name":"test_rate_limit"}]}}
```

Review candidates against current evidence before explicitly requesting
`promote=true`: promotion records an `evidence_of` trail edge through the
ordinary typed-edge surface; it does not create a proven causal relationship.

<!-- TODO(comment): the 3.x backfill contract additionally excluded
suppressed and superseded memories from candidates; whether the 4.0
causal_walk candidate pool applies the same exclusion is not verifiable at
this HEAD (walks read recorded walk sources, not the live-node filter). -->

`graph(action="never_composed")` also defaults to `user`. Supply `scope` for a
project or `includeCrossScope=true` to deliberately investigate across projects.
Absence from the local composition history does not establish worldwide
novelty. The response labels `globalNoveltyVerified=false`. Suppressed,
superseded, expired and future memories are excluded from composition candidates.

These scope controls currently apply to `never_composed` only. Other graph
actions retain their existing global behavior and reject these controls rather
than silently ignoring them. Scope is a selection boundary in the local store;
this patch does not establish multi-tenant authorization across every API.

## Reviewed changes and truthful outcomes

- **Dedup:** Applying a strong match without `confirm=true` requires the current
  `auto_apply` policy to permit it. Policy and all affected namespace identities
  are checked inside the mutation transaction. A plan spanning namespaces is
  rejected even with confirmation. Existing same-scope apply/undo remains
  available. Scan/planning scope and stale-plan content checks are separate
  concerns; this gate does not certify the plan's meaning.
- **Smart ingest:** Supply `content` or `items`, never both. Ambiguous input is
  rejected before storage. Batch results retain legacy `success` and add
  `batchOutcome` (`applied`, `no_changes`, `partial`, `failed`) and `atomic=false`.
  Earlier successful items remain committed when a later item fails.
- **Memory edit:** The content update goes through the store write path. 4.0
  has no embedding state to invalidate; there is no `embeddingStatus`, no
  vector journal, and no re-embedding step. The edit is durable when the write
  lands.
- **Maintenance:** The advertised per-action schemas come from their actual
  handlers. Portable export, `since`, confined export filenames, restore merge
  options, GC age filters, dream controls and scoring context are discoverable.
  Unsupported action fields are rejected. Export supports `since`; it does not
  support `start`/`end`. Advertised snake_case GC and scoring arguments are
  honored. Lifecycle, logs and GC are store-wide; dream accepts an explicit namespace `scope`.
- **Suppression:** Default review mode can hold the operation for review.
  Explicit fast-mode suppression still compounds. New operations have an exact
  per-operation journal, a 24-hour reversal window and atomic conflict checks
  for local state and journaled neighbor effects, as detailed below.

## Deterministic intentions

`intention(action="check")` accepts `context.current_time` (or legacy
`currentTime`) as an RFC3339 clock with timezone. Invalid timestamps fail.
Checks use that clock for trigger, deadline and snooze comparisons.
`include_snoozed=true` makes snoozed records visible, but does not let them fire
before `snoozedUntil`, including when overdue.

Context triggers require all supplied constraints. Codebase and topic matching
retain their case-insensitive substring behavior; `file_pattern` is a literal,
case-sensitive path substring, not a glob. Event triggers require an explicit
observed event key; there is no natural-language event-condition evaluator.

```json
{"name":"intention","arguments":{"action":"set","description":"Review the finished build","trigger":{"type":"event","condition":"build_finished"}}}
```

```json
{"name":"intention","arguments":{"action":"check","context":{"current_time":"2026-09-10T12:00:00Z","event":"build_finished"}}}
```

Dates, trigger types, priorities, list statuses and duration bounds validate
before persistence. Parsed absolute time triggers keep their timestamp. These
checks do not run a background scheduler or prove that an agent will supply the
right event context.

## Handle-only recall and bounded evidence

`recall` is exact-handle only. Free text, a malformed handle, or a handle that
resolves to nothing returns the fail-closed `handle_required` payload with no
candidates field — see [HANDLES.md](HANDLES.md) for the grammar and the
payload shape. Response envelopes stay budgeted: the server omits whole
records and reports truncation rather than cutting a record away from its
evidence, and budget accounting uses serialized UTF-8 bytes divided by four,
rounded up — not an exact model tokenizer count. Expand omitted details
through memory ids.

`session_start` is the bounded start-of-session packet: relevant memories,
open intentions, status, predictions, and codebase context under one token
budget. `graph` covers association and contradiction-style inspection over
recorded edges; what a walk returns is always traceable to frames in the log
([RECEIPTS.md](RECEIPTS.md)), never to a ranker.

## Verification and upgrade boundary

The disposable `scripts/test-tool-frontier.py` fixture checks actual stdio
responses, discovery parity, namespace selection, preview effects, review
holds, and receipt replay. It uses a temporary store and no connector
credentials. `scripts/test-context-evidence.py` separately checks the
source-aware context slice. Unit tests cover tool-specific behavior and
transaction invariants.

`causal_walk`'s preview default, stricter argument validation and namespace
defaults are compatibility changes. Update callers that relied on implicit
promotion, cross-project reasoning, or silently ignored arguments.

### Merge plan and undo consistency

Merge and supersede previews include fingerprints of content, source identity,
scope, protection, suppression and temporal state. Apply checks these inside
its write transaction; changed or legacy previews require a new plan. Members
must be distinct and currently active. Confirmation does not bypass these checks.

Merge undo checks the post-apply fingerprints before restoring durable state in
one transaction. Later edits or control changes produce a conflict. Legacy
operations without fingerprints require manual recovery review. This contract
does not claim reversal of suppression cascades or external side effects.

### Suppression reversal

New suppressions journal the local count, timestamp, retrieval strength, retention
strength and stability in the same transaction as the penalty. Reversal restores
the latest active snapshot, including values clipped at the penalty floor. It
requires an unexpired snapshot and unchanged local suppression state. Stacked
reversals restore each earlier timestamp, so a new suppression cannot extend the
reversal window of an old one. Concurrent or later state changes fail without
partial restoration. The response identifies `reversalScope: "local_state_and_journaled_cascades"`
and `journaledCascadeReversal: "atomic"`.

Schema 34 adds the local suppression journal. Schema 35 binds each neighbor
penalty to that suppression operation, recording its before and after state in
the same transaction. Repeated sweeps apply each operation/neighbor effect at
most once. Cascades exclude protected or suppressed neighbors and other scopes.
Reversal checks every journaled neighbor before restoring the seed and neighbors
atomically. A later neighbor change rejects the whole reversal.

Legacy and portable-imported suppressions without journal snapshots require
explicit review; the tool does not invent their prior strengths. Unrecorded
historical cascades and external effects cannot be reconstructed; the response
reports `unrecordedEffectsReversed: false`. Both journals cascade on memory purge.
Before installation, retain a paired database backup for rollback to an older binary.

### Bounded restore

MCP restore reads at most 64 MiB from a regular file. Legacy JSON batches are
limited to 10000 memories, validated in a disposable store, then imported into
the target in one transaction. Empty content rejects the batch before target
writes. Legacy restore copies only memory rows and
reports `atomic: true`; it does not import staging settings or audit journals.
Portable archives continue through their transactional importer. An index-refresh
error after commit requires inspecting the target before retrying. Input limits
bound file bytes and legacy rows, not a strict process-memory or elapsed-time budget.

### Lifecycle, logs, GC and dream pages

`maintain(action="consolidate", phase="lifecycle", batchSize=100, budgetMs=1000)`
previews a transactional ID page. Apply with `dry_run=false`. It updates decay,
emotional promotion and activation, preserving protected/suppressed rows. Limits
are 1–1000 rows and a cooperative 1–10000 ms processing budget; waiting for a lock
or completing an SQL operation can exceed that time. Access-history input caps
at 500 events per memory. `phase="logs"` trims bounded old-log batches and uses
`hasMore` rather than a cursor; it rejects `after` and `budgetMs`.

GC uses bounded transactional pages with eligibility rechecked before deletion.
Protected records survive and a page failure rolls back the whole page, including
purge side effects. Its UUID cursor is a live scan position. After a sweep, start
again without a cursor to reconsider earlier inserts or state changes.

Dream selects one scoped UUID page, excludes suppressed or currently invalid
memories, and caps the input to fit `max_pairs` (default 1225). `memory_count`
accepts 5–500; `max_pairs` accepts 10–124750. Resume using `nextCursor` while
`hasMore` is true. Discovery compares pairs within each page; it does not cover
cross-page pairs. Only processed waking tags at or before the operation's start
are cleared, preserving unprocessed tags and newer tags. Pair/row limits do not
establish a hard inference deadline or dollar cap. Default full consolidation
(`phase="all"`) and background paths retain existing behavior.

The installable `integrations/python` runtime owns the transcript and selected
catalog and implements explicit retained-packet acknowledgment. It refreshes
packets after compaction and serializes OpenAI Responses/Anthropic Messages
requests. It does not install itself into another agent or invoke a model.
