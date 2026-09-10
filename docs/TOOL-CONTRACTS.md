# Tool contracts and progressive discovery

Vestige advertises fourteen MCP tools. Most are action multiplexers: `recall`
uses `mode`, `memory_status` uses `view`, and tools such as `memory` and `dedup`
use `action`. An agent should choose the action needed for its current task.
There is no requirement to call every tool during every session.

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
runtime readiness: a compiled embedder may still be loading, and an upstream
connector still needs configuration. Embedding-backed duplicate scan, memory
merge planning and apply need embeddings plus vector search. Tag maintenance
and tag undo work without those features. Use health to inspect runtime state.

The MCP annotations cover the whole tool. `recall` reason mode records
composition evidence; `receipt` replay persists a durable replay artifact.
Neither mixed tool is read-only. Receipt replay reuses its durable identity on
retry. `suppress` compounds and is not idempotent. Client-side approval settings
and the server's Memory PR review mode still determine whether a mutation runs.
A successful protocol response can report `pendingReview=true` and
`success=false`; inspect the application result before claiming a change.

## Investigation before reinforcement

Backfill defaults to `promote=false`. A preview ranks earlier candidates using
shared entities and chronology without changing their strengths or writing
candidate edges. Its `causes` field is retained for compatibility, but every
candidate is a hypothesis. The response explicitly reports
`causality_verified=false`, scan limits, and whether reinforcement occurred.
Existing edges may be reported during a preview; they are not created by it.

```json
{"name":"backfill","arguments":{"failure_id":"<memory-id>","scope":"project"}}
```

Review candidates against current evidence before explicitly requesting
`promote=true`. Expired or future memories are not reinforced. Suppressed and
superseded memories are excluded from candidates. Promotion records a
`backfill_candidate` edge; it does not create a proven causal relationship.
The tool's default namespace is `user`; both failure and candidate selection
use the requested namespace before scan limits. Automatic failure-ingestion
hooks preserve that namespace and explicitly preview candidates without promotion.
They report candidate count and hypothesis status.

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
- **Memory edit:** The content update marks embeddings pending in the same SQL
  transaction, removing stale profile and legacy vectors and journaling invalidation
  for other processes. The tool reports `embeddingStatus` as `available` or `pending` instead
  of promising that regeneration succeeded. Direct and bulk vector readers exclude
  dirty nodes, and regeneration includes dirty nodes even when an old vector has
  the same model and dimensions. Embedding persistence compares the original content and active profile inside
  its write transaction and rejects a stale computation before storing it.
  Tests cover the persistence/index path with supplied vectors; they do not load a model.
- **Maintenance:** The advertised per-action schemas come from their actual
  handlers. Portable export, `since`, confined export filenames, restore merge
  options, GC age filters, dream controls and scoring context are discoverable.
  Unsupported action fields are rejected. Export supports `since`; it does not
  support `start`/`end`. Advertised snake_case GC and scoring arguments are
  honored. Lifecycle, embeddings, logs and GC are store-wide; dream accepts an explicit namespace `scope`.
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

## Scoped reasoning and bounded evidence

Lookup also budgets the complete response envelope with room for server receipt
metadata. It omits whole cards and reports `evidenceIncomplete` or `truncated`
when the budget cannot carry the evidence. Known dissent groups are omitted
together rather than returning only one side. Lookup and reason share the
serialized-byte budget unit described below; it is not an exact tokenizer count.

Lookup's opt-in `context_packet=true` returns stable evidence cards, sorted by
ID, with changing scores and diagnostics omitted. Source and temporal metadata
remain attached when permitted by the output mask. `temporalState` describes
only the validity interval (`current`, `historical`, `future`, or `unknown`),
not source authority or factual correctness. A complete packet receives a
`packetId` bound to its evidence, store, query/filter arguments and output profile.
Changes to content, selected membership, validity state, or the boundary change
the ID. Incomplete packets have no reusable ID.

Only send `known_packet_id` while the previously returned complete packet still
exists in model context. An exact match returns `notModified=true` and no cards.
After compaction, eviction, a new conversation, or uncertain retention, omit it
to get a full refresh. The server does not know the client's context state.
`examples/python/context_packets.py` demonstrates this host-side handshake and
selection of exact tool schemas for clients that support dynamic catalogs.
Neither capability automatically modifies Codex, activates provider prompt
caching, or demonstrates lower billed tokens. Hash identity is not a signature
or authorization boundary.

`recall(mode="reason")` defaults to the `user` namespace and applies supported
scope, type, validity, source, retention and tag filters to retrieved and
activation-expanded evidence. Cross-namespace reasoning requires
`includeCrossScope=true`. Contradiction inspection also defaults to `user`.
Topic search uses a bounded global candidate pool before namespace filtering;
this prevents foreign evidence from being returned but can reduce recall in a
large mixed-namespace store. Unsupported controls fail with mode-specific errors.

Reason confidence is a heuristic score, not a calibrated truth probability.
A supplied `token_budget` keeps complete evidence groups or omits them instead
of cutting claims from their evidence. Budget accounting uses serialized UTF-8
bytes divided by four, rounded up, for structured content including server
metadata. This is not an exact model tokenizer count or a budget for the full
JSON-RPC envelope and its duplicate text representation. Expand omitted details
through memory IDs. Retrieval exposure records include only returned evidence.

## Verification and upgrade boundary

The disposable `scripts/test-tool-frontier.py` fixture checks actual stdio
responses, discovery parity, namespace selection, preview effects, review
holds, receipt replay, and embedding-state reporting. It uses a temporary store
and no connector credentials. `scripts/test-context-evidence.py` separately
checks the source-aware context slice. Unit tests cover mode-specific behavior
and transaction invariants; a client/model adoption benchmark is still needed
to measure whether agents select tools more effectively.

This candidate adds schema 34 (local suppression journal) and schema 35
(journaled cascade effects). Backfill's
new preview default, stricter argument validation, namespace defaults and
corrected annotations are compatibility changes. Update callers that relied on
implicit promotion, cross-project reasoning, or silently ignored arguments.
The prior code-anchor change on this branch has its own versioned-hash rollback
boundary: read `CODE-CONTEXT-EVIDENCE.md` before installing either change.

### Merge plan and undo consistency

Merge and supersede previews include fingerprints of content, source identity,
scope, protection, suppression and temporal state. Apply checks these inside
its write transaction; changed or legacy previews require a new plan. Members
must be distinct and currently active. Confirmation does not bypass these checks.

Merge undo checks the post-apply fingerprints before restoring durable state in
one transaction. Later edits or control changes produce a conflict. Legacy
operations without fingerprints require manual recovery review. Embedding
regeneration follows commit; failure leaves the embedding pending. This contract
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
writes. Legacy restore copies only memory rows, leaves embeddings pending and
reports `atomic: true`; it does not import staging settings or audit journals.
Portable archives continue through their transactional importer. An index-refresh
error after commit requires inspecting the target before retrying. Input limits
bound file bytes and legacy rows, not a strict process-memory or elapsed-time budget.

### Incremental embedding maintenance

`maintain(action="consolidate", phase="embeddings", batchSize=10)` previews a
page by default. `dry_run=false` processes at most 100 selected memories per
call using an already available active runtime; it does not install a model.
The response exposes selection, success/failure/skip counts, runtime availability,
elapsed time, `hasMore`, and `nextCursor`. Work runs off the async executor thread.
Committed embedding rows are the checkpoint. Resume with `after=nextCursor`, then
start a new sweep without `after` to discover earlier inserts or retry failures.
The cursor is a live scan position, not a snapshot or a hard inference deadline.
Suppressed memories are excluded from selection. Embedding batches cap at 100 rows. The default full consolidation behavior
remains separate and retains its compatibility contract.


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
