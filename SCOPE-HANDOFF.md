# SCOPE-HANDOFF — build/t5-legacy-isolation (legacy-sqlite quarantine)

(Replaces the w1d-absorb handoff, already merged at 18b13c7.)

Branch: `build/t5-legacy-isolation` (off main @ 82f0cf5). Committed locally;
nothing pushed. Mission: every line of SQLite/rusqlite code moves behind a
`legacy-sqlite` cargo feature (default ON), so the wipe is a flag flip. No
SQLite code was deleted — quarantine only. Sibling strata crates
(`crates/strata-gate`, `crates/strata-kernel`, not workspace members) untouched.

## Acceptance gates (both verified live)

- `cargo check --workspace` — SUCCESS, zero errors.
- `cargo check --workspace --no-default-features` — SUCCESS, zero errors.
- Bonus, beyond the gate: `--all-targets` in BOTH modes also passes with
  zero errors (in-file test modules and integration targets are feature-gated;
  see "tests" below).

## The feature

- `vestige-core`: `legacy-sqlite = ["dep:rusqlite"]`, in `default`
  (default = `["legacy-sqlite", "bundled-sqlite", "codebase-git"]`).
  `bundled-sqlite` and `encryption` now imply `legacy-sqlite`. `rusqlite`
  is `optional = true`.
- `vestige-mcp`: `legacy-sqlite = ["vestige-core/legacy-sqlite",
  "vestige-core/bundled-sqlite", "dep:rusqlite"]`, in `default`. The
  previously hardcoded `features = ["bundled-sqlite"]` on the core dep moved
  into this feature; core dep is now `default-features = false` with no
  unconditional features. mcp's own `rusqlite` (backup WAL checkpoint) is
  `optional = true` under the feature.
- `tests/e2e`, `tests/phase_1`: same pattern (`legacy-sqlite` pass-through,
  default ON; e2e's core dep is now `default-features = false`).

## Constructor path

`vestige_core::storage::open_storage(Option<PathBuf>)` is the sanctioned way
to open a store:

- feature ON → `SqliteMemoryStore::new` (identical behavior).
- feature OFF → returns `LegacySqliteDisabled`, message verbatim:
  `"built without legacy-sqlite; STRATA default lands in the next merge"`.
  `LegacySqliteDisabled` (thiserror) is compiled in EVERY build so surfaces
  can print it regardless of feature state (feature unification can leave
  core's feature on while mcp's is off — the mcp shells print the error type
  directly for exactly that reason).

Call sites migrated: mcp `main.rs` (serve), `bin/restore.rs`, `bin/cli.rs`
(via its local `open_storage()`). The `Storage = SqliteMemoryStore` alias and
`Storage::new` itself now exist only in feature builds.

## Everything now behind `legacy-sqlite`

vestige-core — whole files compiled out without the feature (31 files):
- `storage/sqlite/` — all 12 files (mod, actors, admin, connectors, ingest,
  lifecycle, merge, purge, records, search, sync, tests) plus
  `storage/v3_regression_tests.rs` (included via `#[path]` from sqlite/mod.rs).
- `storage/` siblings that talk rusqlite or the store type (15):
  attestation_store, blast, edges, intention_claim, intention_graph_store,
  maintenance_batches, migrations, portable, replay_store, resolver,
  session_queries, synaptic_store, trace_store, unlearning_store,
  walk_receipts.
- `fts.rs` (SQLite FTS5 sanitizers; its re-export in `search/` gated too).
- `codebase/anchor.rs` (persists/verifies anchors through the store).
- `consolidation/dream_compile.rs` (4-phase engine wired to the store).

vestige-core — partially gated files (gated items, file still compiles):
`lib.rs` (fts mod + gated storage/dream_compile re-exports + prelude split),
`storage/mod.rs` (module decls, re-exports, `Storage` alias, open_storage),
`storage/memory_store.rs` (`From<StorageError>` + one test),
`storage/receipt_attestation.rs` (CounterfactualReplay match arms),
`trace/receipt.rs` (`ReceiptEvidence::CounterfactualReplay` variant + arm),
`search/mod.rs` (fts re-export), `advanced/causal_walk.rs`
(`walk_storage`/`persist_evidence_edges` + tests), `advanced/mod.rs`,
`codebase/mod.rs` (anchor mod/re-exports), `connectors/mod.rs`
(`run_sync` + `ConnectorCursor` import + both test mods), `connectors/github.rs`
(tests), `consolidation/mod.rs` (dream_compile mod/re-exports),
`projection.rs` (`select_durable`/`project` + tests).

vestige-mcp — whole modules gated OFF (57 files): `tools/` (45), `dashboard/`
(6), `resources/` (5), `server.rs`, `cognitive.rs`, `autopilot.rs`,
`actor_surface.rs`, `trace_recorder.rs`, `protocol/http.rs`,
`protocol/stdio.rs`. Partially gated: `lib.rs`, `main.rs` (serve body behind
one cfg block; feature-off shell logs the LegacySqliteDisabled error and
exits 1), `bin/cli.rs` (30 store-backed `run_*`/helpers gated; feature-off
dispatch keeps only `update`/`sandwich` working, every store subcommand
fails with the clear error), `bin/restore.rs` (restore loop gated; parse +
`-h/-V` still work), `protocol/mod.rs`, `Cargo.toml`.

tests: `tests/phase_1` — all 4 [[test]] targets `required-features =
["legacy-sqlite"]`, own rusqlite dep optional under the feature.
`tests/e2e` — all 20 [[test]] targets `required-features = ["legacy-sqlite"]`,
lib prelude split. `vestige-mcp` integration tests (4) `required-features`.
In-file test modules gated with `#[cfg(all(test, feature = "legacy-sqlite"))]`
where they build stores.

## portable.rs verification (mission open question)

`storage/portable.rs` DOES read the SQLite database directly for export
(rusqlite Connection usage), so it took the feature — note the flip
consequence: a feature-off build has no portable export/import until STRATA
ships its own archive path. The pure archive TYPES live in the same file and
are gated with it.

## Files that still reach rusqlite with the feature OFF

Zero. Live grep for `rusqlite` outside `storage/sqlite/`, the gated storage
siblings, and `codebase/anchor.rs` returns only: comment text in
`storage/mod.rs`, and mcp files (`bin/cli.rs`, `tools/receipt.rs`,
`tools/forgotten_lesson.rs`, `dashboard/handlers.rs`, `trace_recorder.rs`)
whose every rusqlite reference sits inside `#[cfg(feature = "legacy-sqlite")]`
modules/functions, plus `tests/phase_1/domain_column_migration.rs` (target
excluded via required-features). `vestige-spacetime` never depended on
rusqlite.

## Known noise (expected, do not "fix" blindly)

- Feature-off builds emit dead-code warnings for helpers whose only callers
  are gated (e.g. `causal_walk::commit_sha_of`, `projection::CANDIDATE_LIMIT`,
  cli's `truncate`). They come back when the callers return; silencing them
  crate-wide would hide real rot.
- mcp feature-off binaries are intentional stubs that exit(1) with the
  quarantine message (only `vestige update`/`sandwich` still function in
  `bin/cli.rs`).
- The flip to default-off (post-STRATA-migration) is: remove `legacy-sqlite`
  from `default` in vestige-core, vestige-mcp, tests/e2e, tests/phase_1.
  Nothing else — that was the point.

## Verification commands (re-run before the flip)

```bash
cargo check --workspace
cargo check --workspace --no-default-features
cargo check --workspace --all-targets
cargo check --workspace --no-default-features --all-targets
```

## SqliteMemoryStore methods (`crates/vestige-core/src/storage/edges.rs`)

```rust
impl SqliteMemoryStore {
    /// Rejects link_type outside TYPED_EDGE_VOCABULARY (Err::InvalidEdge).
    /// INSERT OR REPLACE into memory_connections incl. edge_meta/created_by_run.
    /// FKs enforced: both endpoints must exist in knowledge_nodes.
    pub fn save_typed_edge(&self, edge: &TypedEdge) -> Result<()>;

    /// Typed (vocabulary-only) edges incident to node_id by direction.
    /// Legacy connections stay invisible here.
    pub fn edges_for(&self, node_id: &str, dir: EdgeDirection) -> Result<Vec<TypedEdge>>;

    /// Forward BFS (source->target) over the given kinds. Cycle-safe, exact,
    /// depth-bounded; returns reached ids sorted, start excluded; empty
    /// link_types => empty result.
    pub fn edge_reachability(&self, start_id: &str, link_types: &[EdgeKind],
                             max_depth: usize) -> Result<Vec<String>>;

    /// root_id + derived_from closure, sorted. READ-ONLY: deletes/writes
    /// nothing; output is the review-gate input for a retire decision.
    pub fn retire_subgraph(&self, root_id: &str) -> Result<Vec<String>>;

    /// Writes purge_tombstones row; prior_content_hash = SHA-256 of the node's
    /// content if the row still exists (simulated purge), else NULL.
    /// Re-record replaces (one row per id).
    pub fn record_tombstone(&self, purged_id: &str, reason: &str) -> Result<()>;

    pub fn get_purge_tombstone(&self, purged_id: &str) -> Result<Option<PurgeTombstone>>;
}
```

## Notes for downstream agents

- Edge rows share the `memory_connections` PK `(source_id, target_id)` with
  legacy connections; a typed save overwrites a legacy row with the same pair.
- Node ids for files/symbols/spans (`anchored_to`, `touched`) must be
  `knowledge_nodes` rows — decide the node-creation convention before wiring
  the MCP surface.
- Review gating for `supersedes`/`corrects` is policy for the tools layer;
  storage only guarantees vocabulary validation.
- `evidence_of` replaces `backfill_candidate` for NEW writes; nothing
  rewrites existing `backfill_candidate` rows (out of scope here).
- Tests: `cargo test -p vestige-core --lib edges` — 4 tests (typed
  save/reject, diamond+cycle reachability, non-deleting retire, tombstone on
  simulated purge). Migration replay/idempotence guards pass (35 tests).

# SCOPE-HANDOFF — w3d: selftest + forgotten_lesson

Branch `build/w3d-selftest-forgotten` (base `main` @ `5fbe1df`). All additions
are append-only except three marked test-constant updates forced by parity
guards; nothing outside the listed files was touched.

## New exposed APIs

### `vestige-mcp` MCP tools

| Tool | Module | Signature |
|---|---|---|
| `selftest` | `crates/vestige-mcp/src/tools/selftest.rs` | `pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String>` — no arguments |
| `forgotten_lesson` | `crates/vestige-mcp/src/tools/forgotten_lesson.rs` | same shape; args `{failure_id (required), scope?, scan_limit?}` |

`tools::selftest::schema()`, `tools::forgotten_lesson::schema()` — full JSON
schemas; both registered in `tools::compact::full_schema` (so
`memory_status view='tools'` unfolds them).

Pure helpers other agents can reuse:

- `tools::forgotten_lesson::detect_lesson(&KnowledgeNode, &HashSet<String> failure_anchors, DateTime<Utc> failure_at) -> Option<ForgottenLesson>` — one shared detection rule (lesson tag / exact past-tense fix marker + >=1 exact shared anchor + FSRS R < 0.5 at failure time).
- `tools::forgotten_lesson::{lesson_tagged, has_fix_marker}`, `pub const FORGOTTEN_THRESHOLD: f64 = 0.5`.
- `tools::selftest` internals are private; the calibration flow is only the `execute` tool.

### CLI subcommands (`crates/vestige-mcp/src/bin/cli.rs`)

- `vestige selftest` — prints the same payload as the MCP tool.
- `vestige forgotten-lesson <FAILURE_ID> [--scope S] [--json]`.

## Calibration definition (selftest)

`gap_calibration = true` iff the 6th round (cause and failure share NO anchor)
returns `triggered=true` with an EMPTY causes list AND a non-null `gap` whose
`missing_entities` contains the withheld env-shaped anchor `planted_cause_6`.
Rounds 1-5 each isolate one planted cause (5d backdated, own scope) with a
failure sharing exactly one anchor; hit@1 = cause ranked first, hit@3 = top 3.
The live store is only read (`Storage::backup_to` / `VACUUM INTO` snapshot to a
tempdir); all mutation happens on the copy; the temp store is deleted.

## Non-append-only edits (forced by existing parity guards)

`crates/vestige-mcp/src/server.rs` tests only: advertised-tool count 16 -> 18
(two places), read-only hint list now
`["forgotten_lesson", "memory_status", "selftest", "session_start"]`. Catalog
and dispatch arms themselves are append-only additions.

## Test status

- `cargo test -p vestige-mcp --lib selftest` → 2/2 ok.
- `cargo test -p vestige-mcp --lib forgotten` → 5/5 ok.
- Full `--lib` suite: 763 passed, 2 failed — BOTH failures
  (`server::tests::test_recall_lookup_matches_search_shape`,
  `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`)
  reproduce on the clean base commit (verified via stash) and are pre-existing.
- `cargo clippy -p vestige-mcp --lib --bins`: clean.
# SCOPE-HANDOFF — w3d: selftest + forgotten_lesson

Branch `build/w3d-selftest-forgotten` (base `main` @ `5fbe1df`). All additions
are append-only except three marked test-constant updates forced by parity
guards; nothing outside the listed files was touched.

## New exposed APIs

### `vestige-mcp` MCP tools

| Tool | Module | Signature |
|---|---|---|
| `selftest` | `crates/vestige-mcp/src/tools/selftest.rs` | `pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String>` — no arguments |
| `forgotten_lesson` | `crates/vestige-mcp/src/tools/forgotten_lesson.rs` | same shape; args `{failure_id (required), scope?, scan_limit?}` |

`tools::selftest::schema()`, `tools::forgotten_lesson::schema()` — full JSON
schemas; both registered in `tools::compact::full_schema` (so
`memory_status view='tools'` unfolds them).

Pure helpers other agents can reuse:

- `tools::forgotten_lesson::detect_lesson(&KnowledgeNode, &HashSet<String> failure_anchors, DateTime<Utc> failure_at) -> Option<ForgottenLesson>` — one shared detection rule (lesson tag / exact past-tense fix marker + >=1 exact shared anchor + FSRS R < 0.5 at failure time).
- `tools::forgotten_lesson::{lesson_tagged, has_fix_marker}`, `pub const FORGOTTEN_THRESHOLD: f64 = 0.5`.
- `tools::selftest` internals are private; the calibration flow is only the `execute` tool.

### CLI subcommands (`crates/vestige-mcp/src/bin/cli.rs`)

- `vestige selftest` — prints the same payload as the MCP tool.
- `vestige forgotten-lesson <FAILURE_ID> [--scope S] [--json]`.

## Calibration definition (selftest)

`gap_calibration = true` iff the 6th round (cause and failure share NO anchor)
returns `triggered=true` with an EMPTY causes list AND a non-null `gap` whose
`missing_entities` contains the withheld env-shaped anchor `planted_cause_6`.
Rounds 1-5 each isolate one planted cause (5d backdated, own scope) with a
failure sharing exactly one anchor; hit@1 = cause ranked first, hit@3 = top 3.
The live store is only read (`Storage::backup_to` / `VACUUM INTO` snapshot to a
tempdir); all mutation happens on the copy; the temp store is deleted.

## Non-append-only edits (forced by existing parity guards)

`crates/vestige-mcp/src/server.rs` tests only: advertised-tool count 16 -> 18
(two places), read-only hint list now
`["forgotten_lesson", "memory_status", "selftest", "session_start"]`. Catalog
and dispatch arms themselves are append-only additions.

## Test status

- `cargo test -p vestige-mcp --lib selftest` → 2/2 ok.
- `cargo test -p vestige-mcp --lib forgotten` → 5/5 ok.
- Full `--lib` suite: 763 passed, 2 failed — BOTH failures
  (`server::tests::test_recall_lookup_matches_search_shape`,
  `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`)
  reproduce on the clean base commit (verified via stash) and are pre-existing.
- `cargo clippy -p vestige-mcp --lib --bins`: clean.

# SCOPE-HANDOFF — build/w3c-session-sync

Base: main @ 5fbe1df. Scope held: `crates/vestige-mcp/src/tools/session_context.rs`,
`crates/vestige-mcp/src/tools/source_sync.rs`, new
`crates/vestige-core/src/storage/session_queries.rs` (+ the one-line module
wiring and re-export in `crates/vestige-core/src/storage/mod.rs`, required to
compile the new file). Nothing else touched.

## What was built

### 1. `session_queries.rs` (new, on SqliteMemoryStore)

- `open_failures_touching(changed_files: &[String]) -> Vec<OpenFailureTouching{id, content_preview, anchor}>`
  — failure-like nodes (`retroactive_backfill::looks_like_failure`, reuse) that
  are open (not superseded/suppressed, inside validity window) and whose
  recorded files intersect the changed set by EXACT path equality:
  `code_memory_anchors.file_path` first (anchor rendered `path` or
  `path:symbol`), then the `files:` line of a git-commit record (the
  ` (+N more)` suffix stripped). Prefix/suffix near-misses do not hit. Bounded
  chunked node scan (pages of 500, hard cap 10k nodes).
- `last_session_failed_calls(run_id: Option<&str>) -> Vec<FailedToolCall{run_id, tool, at, error_excerpt}>`
  — `agent_traces` rows of the given run, or the latest run by
  `agent_runs.last_at` when `None`, whose serialized `mcp.call` payload
  carries `success: false`. Last 20, chronological. Rows without a `success`
  field are not failed calls.
- `append_mcp_call_outcome(run_id, tool, success, error, at_ms)` — writes an
  `mcp.call`-shaped trace row extended with `success`/`error` (argsHash is the
  opaque constant `"outcome"`). Payload keeps the exact event shape so
  `get_trace` replay still parses it. NOTE: no production caller yet — the
  trace recorder (`trace_recorder.rs`) is out of scope and still writes
  plain `mcp.call` events, so the failed-calls section stays empty until the
  recorder adopts this method. That adoption is the natural follow-up.
- `closed_issue_nodes(source_system, scope)` / `git_commit_nodes(limit)` —
  exact-tag lookups (`state:closed`, `git-commit`) backing the closed_by link.

### 2. `session_context.rs` (session_start) — purely additive

- New optional arg `changed_files: Vec<String>` (schema + serde, camelCase
  alias). Absent = open-failures section skipped. Present but no exact match =
  no section (silence, never an empty header).
- Section `**Open failures touching changed files:**` — up to 8 lines
  `- [id] preview (anchor)`, budget-accounted like every other section.
- Section `**Last session failed calls (run):**` — automatic, latest run, up
  to 8 lines `- tool: error_excerpt`.
- Both sections degrade silently on query error; all existing sections and
  the budget-trim loop are untouched.

### 3. `source_sync.rs` — `closed_by` chain linking (github, local-only)

- Verified what the github connector fetches today: issues + comments only —
  no closing-PR sha, no timeline/cross-referenced events. Per the build
  instruction, remote fetch is SKIPPED; linking is local-only and
  deterministic.
- `commit_closes_issue(content, issue_number)` — GitHub closing keywords
  (close/closes/closed/fix/fixes/fixed/resolve/resolves/resolved), whole-word,
  case-insensitive, and `#<number>` with exact digit boundaries (`#420` never
  satisfies #42; `abc#42`/`##42` rejected), keyword + reference on the SAME
  line.
- `link_closed_by_from_local_commits(storage, scope)` — for each live closed
  github issue in scope, writes `memory_connections` edges
  `link_type="closed_by"`, `source_id` = issue node, `target_id` = commit
  node, via the existing `save_connection` (INSERT OR REPLACE ⇒ idempotent
  re-syncs). Runs after every successful github `run_sync`; count surfaced as
  `closedByLinks` in the tool result and appended to the summary when > 0.
  Redmine is untouched.

## Tests (all green)

- `cargo test -p vestige-core --lib session_queries` — 6/6.
- `cargo test -p vestige-core --lib storage::` — 372/372.
- `cargo test -p vestige-mcp --lib session` — 27/27 (4 new:
  open-failures populate from anchors + `files:` lines, negative no-noise
  near-miss/absent-arg, failed-calls latest-run, silent without outcome rows).
- `cargo test -p vestige-mcp --lib source_sync` — 4/4 (keyword forms,
  near-miss rejects incl. `discloses`/`#420`/cross-line, edge appears +
  idempotent + open-issue/commit negatives, silent without candidates).
- Full `cargo test -p vestige-mcp --lib`: 764 passed, 2 failed —
  `server::tests::test_recall_lookup_matches_search_shape` and
  `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`
  both REPRODUCE ON PRISTINE main @ 5fbe1df (verified via stash); pre-existing,
  not from this branch.

## Follow-ups (out of scope here)

- Recorder adoption of `append_mcp_call_outcome` in `trace_recorder.rs` /
  `server.rs` dispatch so the failed-calls section has a production producer.
- Remote closing-PR lookup if ever wanted: needs a timeline-events fetch in
  the github connector (new API surface, deliberately not added).
# SCOPE-HANDOFF — build/w3c-session-sync

Base: main @ 5fbe1df. Scope held: `crates/vestige-mcp/src/tools/session_context.rs`,
`crates/vestige-mcp/src/tools/source_sync.rs`, new
`crates/vestige-core/src/storage/session_queries.rs` (+ the one-line module
wiring and re-export in `crates/vestige-core/src/storage/mod.rs`, required to
compile the new file). Nothing else touched.

## What was built

### 1. `session_queries.rs` (new, on SqliteMemoryStore)

- `open_failures_touching(changed_files: &[String]) -> Vec<OpenFailureTouching{id, content_preview, anchor}>`
  — failure-like nodes (`retroactive_backfill::looks_like_failure`, reuse) that
  are open (not superseded/suppressed, inside validity window) and whose
  recorded files intersect the changed set by EXACT path equality:
  `code_memory_anchors.file_path` first (anchor rendered `path` or
  `path:symbol`), then the `files:` line of a git-commit record (the
  ` (+N more)` suffix stripped). Prefix/suffix near-misses do not hit. Bounded
  chunked node scan (pages of 500, hard cap 10k nodes).
- `last_session_failed_calls(run_id: Option<&str>) -> Vec<FailedToolCall{run_id, tool, at, error_excerpt}>`
  — `agent_traces` rows of the given run, or the latest run by
  `agent_runs.last_at` when `None`, whose serialized `mcp.call` payload
  carries `success: false`. Last 20, chronological. Rows without a `success`
  field are not failed calls.
- `append_mcp_call_outcome(run_id, tool, success, error, at_ms)` — writes an
  `mcp.call`-shaped trace row extended with `success`/`error` (argsHash is the
  opaque constant `"outcome"`). Payload keeps the exact event shape so
  `get_trace` replay still parses it. NOTE: no production caller yet — the
  trace recorder (`trace_recorder.rs`) is out of scope and still writes
  plain `mcp.call` events, so the failed-calls section stays empty until the
  recorder adopts this method. That adoption is the natural follow-up.
- `closed_issue_nodes(source_system, scope)` / `git_commit_nodes(limit)` —
  exact-tag lookups (`state:closed`, `git-commit`) backing the closed_by link.

### 2. `session_context.rs` (session_start) — purely additive

- New optional arg `changed_files: Vec<String>` (schema + serde, camelCase
  alias). Absent = open-failures section skipped. Present but no exact match =
  no section (silence, never an empty header).
- Section `**Open failures touching changed files:**` — up to 8 lines
  `- [id] preview (anchor)`, budget-accounted like every other section.
- Section `**Last session failed calls (run):**` — automatic, latest run, up
  to 8 lines `- tool: error_excerpt`.
- Both sections degrade silently on query error; all existing sections and
  the budget-trim loop are untouched.

### 3. `source_sync.rs` — `closed_by` chain linking (github, local-only)

- Verified what the github connector fetches today: issues + comments only —
  no closing-PR sha, no timeline/cross-referenced events. Per the build
  instruction, remote fetch is SKIPPED; linking is local-only and
  deterministic.
- `commit_closes_issue(content, issue_number)` — GitHub closing keywords
  (close/closes/closed/fix/fixes/fixed/resolve/resolves/resolved), whole-word,
  case-insensitive, and `#<number>` with exact digit boundaries (`#420` never
  satisfies #42; `abc#42`/`##42` rejected), keyword + reference on the SAME
  line.
- `link_closed_by_from_local_commits(storage, scope)` — for each live closed
  github issue in scope, writes `memory_connections` edges
  `link_type="closed_by"`, `source_id` = issue node, `target_id` = commit
  node, via the existing `save_connection` (INSERT OR REPLACE ⇒ idempotent
  re-syncs). Runs after every successful github `run_sync`; count surfaced as
  `closedByLinks` in the tool result and appended to the summary when > 0.
  Redmine is untouched.

## Tests (all green)

- `cargo test -p vestige-core --lib session_queries` — 6/6.
- `cargo test -p vestige-core --lib storage::` — 372/372.
- `cargo test -p vestige-mcp --lib session` — 27/27 (4 new:
  open-failures populate from anchors + `files:` lines, negative no-noise
  near-miss/absent-arg, failed-calls latest-run, silent without outcome rows).
- `cargo test -p vestige-mcp --lib source_sync` — 4/4 (keyword forms,
  near-miss rejects incl. `discloses`/`#420`/cross-line, edge appears +
  idempotent + open-issue/commit negatives, silent without candidates).
- Full `cargo test -p vestige-mcp --lib`: 764 passed, 2 failed —
  `server::tests::test_recall_lookup_matches_search_shape` and
  `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`
  both REPRODUCE ON PRISTINE main @ 5fbe1df (verified via stash); pre-existing,
  not from this branch.

## Follow-ups (out of scope here)

- Recorder adoption of `append_mcp_call_outcome` in `trace_recorder.rs` /
  `server.rs` dispatch so the failed-calls section has a production producer.
- Remote closing-PR lookup if ever wanted: needs a timeline-events fetch in
  the github connector (new API surface, deliberately not added).

# Scope Handoff — build/w3e-receipts-status

Base: `main @ 5fbe1df`. Branch: `build/w3e-receipts-status`. Not pushed.

## What was built

### 1. Walk receipts (core storage)

New file `crates/vestige-core/src/storage/walk_receipts.rs` (migration **V39**, non-destructive, `walk_receipts` table: `receipt_id PK, digest UNIQUE, canonical_json, engine_version, created_at`).

- `canonical_walk_json(params: &Value) -> Result<String>` — canonicalizes via
  `serde_json_canonicalizer` (RFC 8785 / JCS), the same helper the receipt DSSE
  chain already uses. Sorted keys, no insignificant whitespace, stable number
  formatting; rejects non-object envelopes.
- `Storage::save_walk_receipt(canonical_json: &str, params: &Value) -> Result<WalkReceiptHandle>`
  — re-canonicalizes `params` internally and **rejects** a `canonical_json` that
  does not match (digest always computable from stored bytes). Digest =
  `blake3(canonical bytes)` hex (dep already in core Cargo.toml). `receipt_id =
  "wr_" + digest[..24]` → saving the same envelope is idempotent
  (`reused_existing: true`, single row). `engine_version` =
  `env!("CARGO_PKG_VERSION")` at save time.
- `Storage::get_walk_receipt(receipt_id) -> Result<Option<StoredWalkReceipt>>`.
- `Storage::coverage_snapshot() -> Result<CoverageSnapshot>` — exact SQL
  aggregates for the coverage view (see §3).

### 2. Replay placement (scope adjustment, per task instruction)

`tools::backfill::execute` lives in the MCP layer only, so **replay lives in the
MCP layer** (`crates/vestige-mcp/src/tools/receipt.rs`), not core. The core file
contributes constants only: `WALK_RECEIPT_SCHEMA_V1`,
`WALK_RECEIPT_CLAIM_BOUNDARY` (`trace/receipt.rs`, re-exported at crate root).

`receipt` tool surface (schema `additionalProperties: false`, `deny_unknown_fields`):

- `{action: "save_walk", params: {...}}` → `{receiptId, digest, canonicalParams
  (byte-stable echo), engineVersion, reusedExisting, claimBoundary}`. No clock
  fields → byte-stable output.
- `{action: "replay", receipt_id: "wr_…" [, remove_edge]}` → dispatches on walk
  receipts (ids in `walk_receipts`); non-`wr_` ids keep the counterfactual
  capsule replay untouched. `remove_edge` (memory id or `source->target`)
  filters that candidate from the pool before the ablated run. Replay is always
  a **pure preview**: recorded `promote/git_repo/worked_in/broke_in/why_not`
  are reported as ignored, never applied (no edges written, no reinforcement).
  Runs the reach twice (baseline + ablated) with a candidate assembly mirroring
  `backfill::build_candidates` (supersession-following, commit-tag flag; no git
  version-range re-application), asserts both runs evaluated the same failure,
  and reports `verdictDelta {filterApplied, verdictChanged, baseline, ablated}`
  (ids + 2-dp scores only → deterministic serialization; failure entities are
  sorted because `extract_entities` iterates a hash set).
- Validation: `withheld_slots` xor `remove_edge`; `params` only for
  `save_walk`; `receipt_id` not for `save_walk`.

### 3. memory_status view="coverage"

`{anchorCoveragePct (COUNT(DISTINCT node_id) code_memory_anchors / COUNT(*)
knowledge_nodes, 2-dp, 0.0 on empty store), anchoredNodes, totalNodes,
edgeCountsByType (memory_connections GROUP BY link_type ORDER BY link_type),
indexFreshness {newestGitCommitRecord + AgeDays (MAX(created_at) over
json_each(tags)='git-commit' nodes), newestAgentTraceAt + AgeHours (MAX(at)
agent_traces, millis), stalenessNote (deterministic thresholds: commit >30d,
trace >72h)}}` + claim boundary.

## Canonicalization spec

RFC 8785 (JCS) via `serde_json_canonicalizer::to_vec`; digest = blake3 of the
UTF-8 canonical bytes, hex; id = `wr_` + first 24 hex chars. Same value → same
bytes → same digest → same id, regardless of input key order/whitespace.

## Test status

- `cargo test -p vestige-core --lib walk_receipt` — 3/3 pass
  (canonicalization stability incl. key-order shuffle + idempotent save +
  mismatch rejection; coverage math on seeded store).
- `cargo test -p vestige-mcp --lib tools::receipt` — 9/9 pass (save twice →
  same digest/id; replay byte-identical on unchanged store; remove_edge
  verdict delta incl. `src->tgt` form and no-op edge; promote never applied).
- `cargo test -p vestige-mcp --lib tools::memory_status` — 5/5 pass
  (incl. coverage math: 1/5 anchored = 20.0%, grouped/ordered edge counts).
- `cargo test -p vestige-core --lib migrations` — 36/36 pass (V39 registered).
- `cargo clippy -p vestige-core -p vestige-mcp --lib` — clean.

### Pre-existing failures on the CLEAN base (not touched, out of scope)

Verified by stashing this diff and re-running:

- `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage` —
  the seeded distractor ("500 Internal Server Error…") outranks the env-var
  cause under the current word-tier IDF scorer. The walk replay mirrors the
  live tool's ranking faithfully, which is why the replay tests are
  rank-agnostic (ablate whatever is ranked top).
- `server::tests::test_recall_lookup_matches_search_shape` — asserts parity
  with the removed `search` tool.

## Files changed

`crates/vestige-core/src/storage/walk_receipts.rs` (new),
`storage/migrations.rs` (V39), `storage/mod.rs` (mod + re-exports),
`trace/receipt.rs` + `trace/mod.rs` + `lib.rs` (walk constants + re-exports),
`crates/vestige-mcp/src/tools/receipt.rs`, `tools/memory_status.rs`,
`server.rs` (receipt tool description string only — it enumerates actions).

# Scope handoff — build/w2a-hunk-lines

Base: main @ 5fbe1df. Scope: `crates/vestige-core/src/advanced/git_records.rs` + tests only.

## Out-of-scope file touched (minimal, compile-only)

`crates/vestige-mcp/src/tools/backfill.rs` — test helper `commit_record_content`
(line ~861) constructs `GitCommit` literally. Added the three new fields with
empty values so the crate compiles:

```rust
hunks: vec![],
extra_hunks: 0,
imports: vec![],
```

No logic changed; `tools::backfill` tests pass unchanged.

## What changed in git_records.rs

- `HunkSpan { file, start, len, symbol }` — new-side `@@` spans, `MAX_HUNKS = 200`,
  overflow counted in `GitCommit::extra_hunks`.
- `GitCommit::imports: Vec<(file_in_commit, target_path, resolved)>` —
  `MAX_IMPORTS = 40`, deduped, sorted by (file, target).
- `record_content` new lines:
  - `hunks: file:start+len` comma-joined, item list capped at 400 chars
    (`MAX_HUNK_LINE`), followed by ` (+N more)` when spans were dropped or
    overflowed; emitted only when spans exist.
  - `imports: file->target` comma-joined; unresolved edges render as
    `file->?target` (target kept as written, never guessed).
- Import capture: `use a::b::Item` / `import x.y.z` (multi-segment only, so
  `import os` noise is skipped) / `from x.y import z` / `#include "p"` and
  `#include <p>`. Resolution is exact module-segment matching against the same
  commit's file list and their module dirs (`crate` roots at `src/`; Rust
  `.rs`/`mod.rs`, Python `.py`/`__init__.py`, include verbatim). File hits beat
  module-dir hits; longest path first. No fuzzy matching.
- Hunks/imports for files beyond `MAX_FILES` are dropped (same
  non-attribution rule as symbols), not counted as span overflow.

## Pre-existing test failure (not from this branch)

`tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`
(crates/vestige-mcp) fails on pristine base @ 5fbe1df — verified via stash
with these changes removed: the billing-service distractor outranks the
env-var cause. The test never touches `GitCommit`/`record_content`; the two
commits below base reworked exactly that ranking code. All other
`tools::backfill` tests pass.

## Incident note: shared stash stack race

This repo's worktrees share one `git stash` stack. During verification, a
`git stash pop` in this worktree raced sibling worktrees and pulled in two
foreign stashes. Both were re-stashed onto the shared stack as labeled
entries — find them with `git stash list | grep RESCUE`:

- `RESCUE-A(65a5ea0)` — storage/session_context/source_sync changes, owner:
  build/w2c-resolver worktree.
- `RESCUE-B(ecfcddfb)` — dashboard cinema changes (MemoryCinema, sandbox,
  storm, temporal-emissive.ts), owner: feat/dashboard-live-max worktree.

Original dropped hashes are recoverable from the stash reflog if needed.
Use hash-based `git stash apply <hash>` (resolved by message) in this repo,
never positional `stash@{n}`.
# Scope handoff — build/w2a-hunk-lines

Base: main @ 5fbe1df. Scope: `crates/vestige-core/src/advanced/git_records.rs` + tests only.

## Out-of-scope file touched (minimal, compile-only)

`crates/vestige-mcp/src/tools/backfill.rs` — test helper `commit_record_content`
(line ~861) constructs `GitCommit` literally. Added the three new fields with
empty values so the crate compiles:

```rust
hunks: vec![],
extra_hunks: 0,
imports: vec![],
```

No logic changed; `tools::backfill` tests pass unchanged.

## What changed in git_records.rs

- `HunkSpan { file, start, len, symbol }` — new-side `@@` spans, `MAX_HUNKS = 200`,
  overflow counted in `GitCommit::extra_hunks`.
- `GitCommit::imports: Vec<(file_in_commit, target_path, resolved)>` —
  `MAX_IMPORTS = 40`, deduped, sorted by (file, target).
- `record_content` new lines:
  - `hunks: file:start+len` comma-joined, item list capped at 400 chars
    (`MAX_HUNK_LINE`), followed by ` (+N more)` when spans were dropped or
    overflowed; emitted only when spans exist.
  - `imports: file->target` comma-joined; unresolved edges render as
    `file->?target` (target kept as written, never guessed).
- Import capture: `use a::b::Item` / `import x.y.z` (multi-segment only, so
  `import os` noise is skipped) / `from x.y import z` / `#include "p"` and
  `#include <p>`. Resolution is exact module-segment matching against the same
  commit's file list and their module dirs (`crate` roots at `src/`; Rust
  `.rs`/`mod.rs`, Python `.py`/`__init__.py`, include verbatim). File hits beat
  module-dir hits; longest path first. No fuzzy matching.
- Hunks/imports for files beyond `MAX_FILES` are dropped (same
  non-attribution rule as symbols), not counted as span overflow.

## Pre-existing test failure (not from this branch)

`tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`
(crates/vestige-mcp) fails on pristine base @ 5fbe1df — verified via stash
with these changes removed: the billing-service distractor outranks the
env-var cause. The test never touches `GitCommit`/`record_content`; the two
commits below base reworked exactly that ranking code. All other
`tools::backfill` tests pass.

## Incident note: shared stash stack race

This repo's worktrees share one `git stash` stack. During verification, a
`git stash pop` in this worktree raced sibling worktrees and pulled in two
foreign stashes. Both were re-stashed onto the shared stack as labeled
entries — find them with `git stash list | grep RESCUE`:

- `RESCUE-A(65a5ea0)` — storage/session_context/source_sync changes, owner:
  build/w2c-resolver worktree.
- `RESCUE-B(ecfcddfb)` — dashboard cinema changes (MemoryCinema, sandbox,
  storm, temporal-emissive.ts), owner: feat/dashboard-live-max worktree.

Original dropped hashes are recoverable from the stash reflog if needed.
Use hash-based `git stash apply <hash>` (resolved by message) in this repo,
never positional `stash@{n}`.

# SCOPE-HANDOFF — Handle-Based Recall Resolver (build/w2c-resolver)

Branch: `build/w2c-resolver`, base = `main` @ `5fbe1df`. Not pushed.

## Files (the entire scope)

- `crates/vestige-core/src/storage/resolver.rs` — NEW. The resolver.
- `crates/vestige-core/src/storage/mod.rs` — wiring only (`mod resolver;` + re-exports).
- `crates/vestige-mcp/src/tools/recall.rs` — thin handle-mode extension at entry.
- `SCOPE-HANDOFF.md` — this file.

## Core API (vestige-core)

```rust
// crates/vestige-core/src/storage/resolver.rs
impl SqliteMemoryStore {
    pub fn resolve_handle(&self, query: &str) -> HandleResolution;
}

pub struct HandleResolution {
    pub kind: HandleKind,                     // what the query resolved as
    pub ids: Vec<String>,                     // node/trace ids resolved
    pub exact: bool,                          // true iff equality (not prefix)
    pub candidates: Vec<(String, HandleKind)>,// when prefix-ambiguous, capped at 20
    pub handle_required: Option<String>,      // guidance when nothing matched
}

pub enum HandleKind { Memory, Commit, File, Symbol, Test, Run, ToolCall, Tag, Unknown }

pub const MAX_CANDIDATES: usize = 20;
pub const HANDLE_REQUIRED_DETAIL: &str =
    "recall is handle-based: pass a memory id, commit sha, file, symbol, test, run, or tool-call id";
```

Import paths: `vestige_core::storage::{HandleKind, HandleResolution, MAX_CANDIDATES, HANDLE_REQUIRED_DETAIL}`. `resolve_handle` is a method on `SqliteMemoryStore` (`vestige_core::Storage`). Read-only: no FSRS/edge/graph writes.

## Resolution rules (EXACT or PREFIX only — no fuzzy, no lexical ranking, no FTS)

| # | Kind | Match | Prefix? | Source |
|---|------|-------|---------|--------|
| 1 | Memory | query parses as uuid, equals `knowledge_nodes.id` | no | `knowledge_nodes` |
| 2 | Commit | `commit <sha>` on content line 1 of `git-commit`-tagged nodes; case-insensitive | **yes, >= 7 hex chars**; 4–6 hex chars w/ a digit = ambiguity error; unique prefix resolves with `exact=false`; multiple = capped candidates | `knowledge_nodes` |
| 3 | File / Test | whole-token boundary match over content + tags (coarse LIKE prefilter, exact Rust verify). File if path-shaped (`/` or `.`); Test if test-shaped (`tests/`, `_test`, `test_` prefix) | no | `knowledge_nodes` |
| 4 | Symbol | query normalized camel→snake+lowercase (same `extract_entities` normalization); matched against Code-tier extracted entities only — Word/Path/Version tiers excluded so `pyvenv` can never prefix-match `pyvenv.cfg` | **yes (unique prefix)** | `knowledge_nodes` |
| 5 | Run | equals `agent_runs.run_id` or `agent_traces.run_id` | no | V18 tables |
| 6 | ToolCall | equals `agent_traces.id` (trace event id) | no | V18 tables |
| 7 | Tag | exact string equality against parsed tags JSON (case-sensitive) | no | `knowledge_nodes` |

Unresolved / empty → `kind=Unknown`, `handle_required=Some(HANDLE_REQUIRED_DETAIL)`.

## MCP surface (recall)

New schema property `handle` (string). Semantics in `recall::execute`:

- args WITHOUT a `handle` key → legacy mode dispatch, byte-identical behavior
  (hot-path invariant preserved; existing tests pass unchanged). **This gate
  (`handle_flow` returning None) is the single flip point for making recall
  handle-only by default.**
- args WITH `handle: "<handle>"` → resolve, then:
  - resolved: `{"handle", "kind", "exact", "nodes":[{id,type,content,tags}], "neighbors":[{from,to,link_type,strength,direction,node}]}` — neighbors are one hop over `memory_connections` (both directions, strength-desc, deduped, capped at 20 edges).
  - ambiguous: `{"error":"ambiguous", "detail", "handle", "kind", "candidates":[{id,kind}]}`.
  - nothing: `{"error":"handle_required", "detail", "candidates":[]}` (`detail` carries the resolver's specific too-short-sha message when applicable).
- args WITH `handle: ""` (or non-string) → free-text branch: same
  `handle_required` payload, with candidates mined by running the resolver on
  the whole `query` text and its first 8 identifier-shaped tokens (exact/prefix
  results only, deduped, capped at 20).

## Integrator notes

- Scope-agnostic by design: handles are globally unique, so resolution ignores
  `scope`. Filter client-side if a scoped surface is needed.
- Symbol prefix scans all nodes and runs `extract_entities` per row (no entity
  index yet). Correctness-first; add an entity index before high-QPS use.
- Commit resolution reads shas only from `git-commit`-tagged records in the
  canonical `git_records::record_content` shape.
- Known deviation from the letter of the design: File and Test share one
  boundary-exact step (file before symbol per the order); a test-shaped query
  resolves as kind `Test` even before the symbol step would have matched.
- English hex words without digits (`face`, `added`) are not treated as
  too-short shas; 4–6 char hex WITH a digit is the ambiguity error.

## Tests

- `cargo test -p vestige-core --lib resolver` → 8 passed (uuid exact/prefix
  paths, full/prefix/ambiguous/too-short shas, file exact + NO-fuzzy +
  NO-prefix receipts (`pyvenv` and `pyvenv.c` both fail against `pyvenv.cfg`),
  symbol camel/snake/env normalization + prefix, test names, tags exact,
  run/toolcall exact, free prose unresolved).
- `cargo test -p vestige-mcp --lib tools::recall` → 7 passed (schema, legacy
  default, contradictions, uuid + one-hop neighbors ordering, sha exact +
  ambiguous, handle_required with/without candidates, no-handle legacy path).
- `cargo test -p vestige-core --lib "storage::sqlite"` → 220 passed (no
  regressions in the neighboring suite).
- Pre-existing failure, NOT from this branch (fails on clean `main` @
  `5fbe1df` too): `vestige-mcp server::tests::test_recall_lookup_matches_search_shape`
  — it calls the removed `search` tool and compares against its error text.
  `server.rs` is outside this task's file scope.
# SCOPE-HANDOFF — Handle-Based Recall Resolver (build/w2c-resolver)

Branch: `build/w2c-resolver`, base = `main` @ `5fbe1df`. Not pushed.

## Files (the entire scope)

- `crates/vestige-core/src/storage/resolver.rs` — NEW. The resolver.
- `crates/vestige-core/src/storage/mod.rs` — wiring only (`mod resolver;` + re-exports).
- `crates/vestige-mcp/src/tools/recall.rs` — thin handle-mode extension at entry.
- `SCOPE-HANDOFF.md` — this file.

## Core API (vestige-core)

```rust
// crates/vestige-core/src/storage/resolver.rs
impl SqliteMemoryStore {
    pub fn resolve_handle(&self, query: &str) -> HandleResolution;
}

pub struct HandleResolution {
    pub kind: HandleKind,                     // what the query resolved as
    pub ids: Vec<String>,                     // node/trace ids resolved
    pub exact: bool,                          // true iff equality (not prefix)
    pub candidates: Vec<(String, HandleKind)>,// when prefix-ambiguous, capped at 20
    pub handle_required: Option<String>,      // guidance when nothing matched
}

pub enum HandleKind { Memory, Commit, File, Symbol, Test, Run, ToolCall, Tag, Unknown }

pub const MAX_CANDIDATES: usize = 20;
pub const HANDLE_REQUIRED_DETAIL: &str =
    "recall is handle-based: pass a memory id, commit sha, file, symbol, test, run, or tool-call id";
```

Import paths: `vestige_core::storage::{HandleKind, HandleResolution, MAX_CANDIDATES, HANDLE_REQUIRED_DETAIL}`. `resolve_handle` is a method on `SqliteMemoryStore` (`vestige_core::Storage`). Read-only: no FSRS/edge/graph writes.

## Resolution rules (EXACT or PREFIX only — no fuzzy, no lexical ranking, no FTS)

| # | Kind | Match | Prefix? | Source |
|---|------|-------|---------|--------|
| 1 | Memory | query parses as uuid, equals `knowledge_nodes.id` | no | `knowledge_nodes` |
| 2 | Commit | `commit <sha>` on content line 1 of `git-commit`-tagged nodes; case-insensitive | **yes, >= 7 hex chars**; 4–6 hex chars w/ a digit = ambiguity error; unique prefix resolves with `exact=false`; multiple = capped candidates | `knowledge_nodes` |
| 3 | File / Test | whole-token boundary match over content + tags (coarse LIKE prefilter, exact Rust verify). File if path-shaped (`/` or `.`); Test if test-shaped (`tests/`, `_test`, `test_` prefix) | no | `knowledge_nodes` |
| 4 | Symbol | query normalized camel→snake+lowercase (same `extract_entities` normalization); matched against Code-tier extracted entities only — Word/Path/Version tiers excluded so `pyvenv` can never prefix-match `pyvenv.cfg` | **yes (unique prefix)** | `knowledge_nodes` |
| 5 | Run | equals `agent_runs.run_id` or `agent_traces.run_id` | no | V18 tables |
| 6 | ToolCall | equals `agent_traces.id` (trace event id) | no | V18 tables |
| 7 | Tag | exact string equality against parsed tags JSON (case-sensitive) | no | `knowledge_nodes` |

Unresolved / empty → `kind=Unknown`, `handle_required=Some(HANDLE_REQUIRED_DETAIL)`.

## MCP surface (recall)

New schema property `handle` (string). Semantics in `recall::execute`:

- args WITHOUT a `handle` key → legacy mode dispatch, byte-identical behavior
  (hot-path invariant preserved; existing tests pass unchanged). **This gate
  (`handle_flow` returning None) is the single flip point for making recall
  handle-only by default.**
- args WITH `handle: "<handle>"` → resolve, then:
  - resolved: `{"handle", "kind", "exact", "nodes":[{id,type,content,tags}], "neighbors":[{from,to,link_type,strength,direction,node}]}` — neighbors are one hop over `memory_connections` (both directions, strength-desc, deduped, capped at 20 edges).
  - ambiguous: `{"error":"ambiguous", "detail", "handle", "kind", "candidates":[{id,kind}]}`.
  - nothing: `{"error":"handle_required", "detail", "candidates":[]}` (`detail` carries the resolver's specific too-short-sha message when applicable).
- args WITH `handle: ""` (or non-string) → free-text branch: same
  `handle_required` payload, with candidates mined by running the resolver on
  the whole `query` text and its first 8 identifier-shaped tokens (exact/prefix
  results only, deduped, capped at 20).

## Integrator notes

- Scope-agnostic by design: handles are globally unique, so resolution ignores
  `scope`. Filter client-side if a scoped surface is needed.
- Symbol prefix scans all nodes and runs `extract_entities` per row (no entity
  index yet). Correctness-first; add an entity index before high-QPS use.
- Commit resolution reads shas only from `git-commit`-tagged records in the
  canonical `git_records::record_content` shape.
- Known deviation from the letter of the design: File and Test share one
  boundary-exact step (file before symbol per the order); a test-shaped query
  resolves as kind `Test` even before the symbol step would have matched.
- English hex words without digits (`face`, `added`) are not treated as
  too-short shas; 4–6 char hex WITH a digit is the ambiguity error.

## Tests

- `cargo test -p vestige-core --lib resolver` → 8 passed (uuid exact/prefix
  paths, full/prefix/ambiguous/too-short shas, file exact + NO-fuzzy +
  NO-prefix receipts (`pyvenv` and `pyvenv.c` both fail against `pyvenv.cfg`),
  symbol camel/snake/env normalization + prefix, test names, tags exact,
  run/toolcall exact, free prose unresolved).
- `cargo test -p vestige-mcp --lib tools::recall` → 7 passed (schema, legacy
  default, contradictions, uuid + one-hop neighbors ordering, sha exact +
  ambiguous, handle_required with/without candidates, no-handle legacy path).
- `cargo test -p vestige-core --lib "storage::sqlite"` → 220 passed (no
  regressions in the neighboring suite).
- Pre-existing failure, NOT from this branch (fails on clean `main` @
  `5fbe1df` too): `vestige-mcp server::tests::test_recall_lookup_matches_search_shape`
  — it calls the removed `search` tool and compares against its error text.
  `server.rs` is outside this task's file scope.

# W1E Test Cleanup — Scope Handoff

Branch `build/w1e-tests-cleanup` (base: main @ 5fbe1df). Test hygiene ONLY, per
owner order. Source modules still present in this tree; parallel agents
(w1a-core-kill, w1b-search-collapse, w1c-ingest-flip, w1d-mcp-surface) own the
source deletions.

## Deleted (clearly vector/embedding-only)

- `tests/e2e/src/mocks/mock_embedding.rs` (whole file; no test used
  `MockEmbeddingService` directly — only re-exports in `tests/e2e/src/lib.rs`
  and `tests/e2e/src/mocks/mod.rs`, both cleaned).
- `tests/e2e/tests/extreme/proof_of_superiority.rs`:
  `test_proof_hippocampal_indexing_efficiency` (embedding-compression bench).
- `tests/e2e/tests/cognitive/comparative_benchmarks.rs`:
  `test_two_phase_vs_flat_search`, `test_index_compression_ratio`.
- `crates/vestige-core/src/storage/v3_regression_tests.rs`:
  `v3_delayed_embedding_cannot_resurrect_a_purged_memory` (gated).
- `crates/vestige-core/src/storage/sqlite/tests.rs` (27 tests + 5 helpers):
  runtime vector-gate tests (`vector_search_env_value_parsing`,
  `disabling_vector_search_...`, `test_runtime_vector_gate_env_...` x2),
  embedding identity/regeneration/dirty-embedding tests, embedding profile
  lifecycle tests (`init_embeddings_permits/rejects_...`,
  `reopening_after_qwen_pointer_...`, `embedding_profiles_keep_vectors_isolated`,
  `activation_rejects_ready_...`, `migration_vector_and_node_checkpoint_...`,
  `purge_removes_vectors_from_every_embedding_profile` + `ready_profile_manifest`
  helper), `non_256_active_profile_...`, all peer/vector-journal tests
  (`peer_process_write_...`, `peer_reembedding_...`, `peer_purge_...`,
  `peer_content_edit_...`, `own_writes_...`, `a_journal_pruned_...`,
  `vector_journal_prune_...`), `embedding_maintenance_preview_...`,
  `embedding_write_rejects_stale_...`,
  `purge_and_profile_activation_do_not_deadlock`, and orphaned helpers
  (`with_vector_search_disabled`, `persist_test_vector`, `index_contains`,
  `nearest`, `ingest_plain`).
- `crates/vestige-mcp/tests/e2e_real_binary.rs`: embedding warm-up test,
  `the_real_embedding_runtime_...`, `tag_prefix_filtering_..._hybrid_path`
  (keyword-path sibling kept), `approved_purge_removes_the_stored_embedding`,
  `embeddings_and_semantic_retrieval_survive_a_restart`,
  `corrupt_fts_rebuild_preserves_embeddings`; removed the
  `node_embeddings`-count assertion from the keyword purge-tombstone test;
  trimmed `embeddingsCompiledIn`/`embeddingReady` from the memory_status
  health shape assertion.
- `crates/vestige-mcp/tests/e2e_failure_cases.rs`:
  `conflicting_ingest_during_a_labile_window_...` (vector-gated supersede path
  by its own doc).
- `crates/vestige-mcp/tests/common/mod.rs`: removed `wait_for_embeddings`,
  `ingest_embedded`, `ingest_inner`'s embedding assert, `wait_for_log_notification`,
  `EMBEDDING_TIMEOUT`, `EMBEDDINGS_READY`.

## Adapted (kept green WITHOUT embeddings at runtime)

- `proof_of_superiority.rs` `test_proof_multi_hop_beats_similarity`: local
  mock-vector similarity half removed; now asserts structural results only
  (4-hop membership + path). Comprehensive summary test renumbered to 5
  capabilities (hippocampal compression block removed).
- `e2e_real_binary.rs` `contradictions_...` and `correction_...`: converted
  from #[ignore]d real-embedding-runtime tests to default-suite keyword-path
  tests (their subjects are lexical by construction). Verified green.

## Borderline — left gated / left as-is for the integration pass

All still behind `#[cfg(all(feature = "embeddings", feature = "vector-search"))]`
in `crates/vestige-core/src/storage/sqlite/tests.rs`; their subject code is
itself feature-gated, so decide at integration (dedup/merge tool SURVIVES in
the 15-tool surface — if merge code is un-gated rather than deleted, these need
keyword-path fixtures, not deletion):

- Merge/supersede suite: `test_merge_candidates_threshold_classification`,
  `test_plan_merge_is_preview_only_no_mutation`,
  `concurrent_apply_of_one_plan_applies_it_exactly_once`,
  `test_merge_state_rejects_...`, `test_merge_undo_conflict_...`,
  `test_merge_undo_concurrent_...`, `test_apply_then_undo_merge_...`,
  `test_supersede_invalidates_...`, `test_protect_blocks_merge_away`,
  all `test_auto_dedup_*` (6) + `set_retention`/`with_auto_merge_env` helpers,
  `pinning_auto_merge_in_one_test_...`,
  `test_apply_requires_confirm_...`, `apply_match_obeys_...`,
  `apply_match_can_use_...`, `test_merge_policy_roundtrip_persists`.
- Validity-window suite (fixture needs active embedding profile via
  `storage_with_marker_gate_runtime` + `MarkerEmbedder`):
  `inferred_as_of_validity_...`, `inferred_as_of_must_not_resurrect_...`,
  `explicit_valid_from_on_reinforce_...`, `create_path_still_stamps_...`,
  `older_dated_claim_after_newer_fact_...`.
- Reconsolidation suite (same fixture): `contradiction_during_live_window_...`,
  `approve_verdict_applies_...`, `reject_verdict_leaves_...`,
  `quarantine_verdict_suppresses_...`, `expired_window_auto_closes_...`,
  `pending_reconsolidation_plans_...` + `reconsolidation_candidate` helper.
- Vector fixture helpers still used by the above: `seed_node`, `axis_vector`.
- `comparative_benchmarks.rs` `test_barcode_orthogonality` +
  `test_content_pointer_accuracy`: structural (hash barcodes, content
  pointers) but reference `neuroscience::hippocampal_index`. If w1a deletes
  that module, these go with it. NOTE: the vestige-e2e-tests crate defines no
  features, so a `#[cfg(feature = "embeddings")]` there would be silently
  always-off; they were left running instead.
- `e2e_real_binary.rs` `foreign_key_orphans_are_repaired_...`: uses
  `node_embeddings` as a generic CASCADE-child fixture. If the table is
  dropped, swap the fixture to another CASCADE child (e.g. fsrs_cards).
- `tests/e2e/tests/journeys/ingest_recall_review.rs`
  `test_recall_search_modes`: constructs `SearchMode::Semantic` (w1b scope).
- `tests/e2e/Cargo.toml` still requests `features = ["embeddings",
  "vector-search"]` from vestige-core — drop when the features die.

## Notes for integration

- No e2e scenario exercised `match_context`/`search` tool aliases at this
  base (grep-verified); nothing to flip to removal-error assertions. The
  aliases exist only in source (`crates/vestige-mcp/src/server.rs`). HOWEVER,
  `cargo test -p vestige-mcp --lib` at this base already fails
  `server::tests::test_recall_lookup_matches_search_shape`
  ("recall(mode=lookup) must equal search byte-for-byte") — that IS a
  search-alias-expects-success test living in src tests; w1d should flip it
  to expect the removal error. Second pre-existing failure:
  `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`.
  Both verified pre-existing via `git stash` roundtrip at base 5fbe1df
  (lib build contains none of this branch's changes).
- `crates/vestige-core/src/storage/sqlite/embeddings.rs`:
  `embedding_model_matches_active` / `embedding_vector_for_active_model` are
  now unused (their only callers were deleted tests) — dead-code warning under
  default features; resolves itself when w1a deletes the embeddings module.
- `cargo check --tests --no-default-features` (workspace): FAILS in NON-TEST
  source, pre-existing at base: `crates/vestige-mcp/src/tools/search_unified.rs:1261`
  E0382 "use of moved value: `rerank_candidates`" (into_iter at 1257, reused at
  1261; the no-default-features branch of the reranker). Not fixed here per
  owner order — belongs to w1b/w1d. vestige-core's test build compiles clean
  under no-default-features apart from the dead-code warning above.

## Verification (this tree, default features)

- `cargo test -p vestige-core --lib`: 936 passed, 0 failed, 1 ignored.
- `cargo test -p vestige-e2e-tests --test comparative_benchmarks --test
  proof_of_superiority`: 23 + 4 passed, 0 failed.
- `cargo test -p vestige-mcp --lib`: 756 passed, 2 failed — both failures
  pre-existing at base (see Notes), unaffected by this branch.
- `cargo test -p vestige-mcp --test e2e_real_binary -- contradictions
  correction`: 2 passed (the two keyword-path conversions).
- `cargo check -p vestige-e2e-tests --tests`: clean.
# W1E Test Cleanup — Scope Handoff

Branch `build/w1e-tests-cleanup` (base: main @ 5fbe1df). Test hygiene ONLY, per
owner order. Source modules still present in this tree; parallel agents
(w1a-core-kill, w1b-search-collapse, w1c-ingest-flip, w1d-mcp-surface) own the
source deletions.

## Deleted (clearly vector/embedding-only)

- `tests/e2e/src/mocks/mock_embedding.rs` (whole file; no test used
  `MockEmbeddingService` directly — only re-exports in `tests/e2e/src/lib.rs`
  and `tests/e2e/src/mocks/mod.rs`, both cleaned).
- `tests/e2e/tests/extreme/proof_of_superiority.rs`:
  `test_proof_hippocampal_indexing_efficiency` (embedding-compression bench).
- `tests/e2e/tests/cognitive/comparative_benchmarks.rs`:
  `test_two_phase_vs_flat_search`, `test_index_compression_ratio`.
- `crates/vestige-core/src/storage/v3_regression_tests.rs`:
  `v3_delayed_embedding_cannot_resurrect_a_purged_memory` (gated).
- `crates/vestige-core/src/storage/sqlite/tests.rs` (27 tests + 5 helpers):
  runtime vector-gate tests (`vector_search_env_value_parsing`,
  `disabling_vector_search_...`, `test_runtime_vector_gate_env_...` x2),
  embedding identity/regeneration/dirty-embedding tests, embedding profile
  lifecycle tests (`init_embeddings_permits/rejects_...`,
  `reopening_after_qwen_pointer_...`, `embedding_profiles_keep_vectors_isolated`,
  `activation_rejects_ready_...`, `migration_vector_and_node_checkpoint_...`,
  `purge_removes_vectors_from_every_embedding_profile` + `ready_profile_manifest`
  helper), `non_256_active_profile_...`, all peer/vector-journal tests
  (`peer_process_write_...`, `peer_reembedding_...`, `peer_purge_...`,
  `peer_content_edit_...`, `own_writes_...`, `a_journal_pruned_...`,
  `vector_journal_prune_...`), `embedding_maintenance_preview_...`,
  `embedding_write_rejects_stale_...`,
  `purge_and_profile_activation_do_not_deadlock`, and orphaned helpers
  (`with_vector_search_disabled`, `persist_test_vector`, `index_contains`,
  `nearest`, `ingest_plain`).
- `crates/vestige-mcp/tests/e2e_real_binary.rs`: embedding warm-up test,
  `the_real_embedding_runtime_...`, `tag_prefix_filtering_..._hybrid_path`
  (keyword-path sibling kept), `approved_purge_removes_the_stored_embedding`,
  `embeddings_and_semantic_retrieval_survive_a_restart`,
  `corrupt_fts_rebuild_preserves_embeddings`; removed the
  `node_embeddings`-count assertion from the keyword purge-tombstone test;
  trimmed `embeddingsCompiledIn`/`embeddingReady` from the memory_status
  health shape assertion.
- `crates/vestige-mcp/tests/e2e_failure_cases.rs`:
  `conflicting_ingest_during_a_labile_window_...` (vector-gated supersede path
  by its own doc).
- `crates/vestige-mcp/tests/common/mod.rs`: removed `wait_for_embeddings`,
  `ingest_embedded`, `ingest_inner`'s embedding assert, `wait_for_log_notification`,
  `EMBEDDING_TIMEOUT`, `EMBEDDINGS_READY`.

## Adapted (kept green WITHOUT embeddings at runtime)

- `proof_of_superiority.rs` `test_proof_multi_hop_beats_similarity`: local
  mock-vector similarity half removed; now asserts structural results only
  (4-hop membership + path). Comprehensive summary test renumbered to 5
  capabilities (hippocampal compression block removed).
- `e2e_real_binary.rs` `contradictions_...` and `correction_...`: converted
  from #[ignore]d real-embedding-runtime tests to default-suite keyword-path
  tests (their subjects are lexical by construction). Verified green.

## Borderline — left gated / left as-is for the integration pass

All still behind `#[cfg(all(feature = "embeddings", feature = "vector-search"))]`
in `crates/vestige-core/src/storage/sqlite/tests.rs`; their subject code is
itself feature-gated, so decide at integration (dedup/merge tool SURVIVES in
the 15-tool surface — if merge code is un-gated rather than deleted, these need
keyword-path fixtures, not deletion):

- Merge/supersede suite: `test_merge_candidates_threshold_classification`,
  `test_plan_merge_is_preview_only_no_mutation`,
  `concurrent_apply_of_one_plan_applies_it_exactly_once`,
  `test_merge_state_rejects_...`, `test_merge_undo_conflict_...`,
  `test_merge_undo_concurrent_...`, `test_apply_then_undo_merge_...`,
  `test_supersede_invalidates_...`, `test_protect_blocks_merge_away`,
  all `test_auto_dedup_*` (6) + `set_retention`/`with_auto_merge_env` helpers,
  `pinning_auto_merge_in_one_test_...`,
  `test_apply_requires_confirm_...`, `apply_match_obeys_...`,
  `apply_match_can_use_...`, `test_merge_policy_roundtrip_persists`.
- Validity-window suite (fixture needs active embedding profile via
  `storage_with_marker_gate_runtime` + `MarkerEmbedder`):
  `inferred_as_of_validity_...`, `inferred_as_of_must_not_resurrect_...`,
  `explicit_valid_from_on_reinforce_...`, `create_path_still_stamps_...`,
  `older_dated_claim_after_newer_fact_...`.
- Reconsolidation suite (same fixture): `contradiction_during_live_window_...`,
  `approve_verdict_applies_...`, `reject_verdict_leaves_...`,
  `quarantine_verdict_suppresses_...`, `expired_window_auto_closes_...`,
  `pending_reconsolidation_plans_...` + `reconsolidation_candidate` helper.
- Vector fixture helpers still used by the above: `seed_node`, `axis_vector`.
- `comparative_benchmarks.rs` `test_barcode_orthogonality` +
  `test_content_pointer_accuracy`: structural (hash barcodes, content
  pointers) but reference `neuroscience::hippocampal_index`. If w1a deletes
  that module, these go with it. NOTE: the vestige-e2e-tests crate defines no
  features, so a `#[cfg(feature = "embeddings")]` there would be silently
  always-off; they were left running instead.
- `e2e_real_binary.rs` `foreign_key_orphans_are_repaired_...`: uses
  `node_embeddings` as a generic CASCADE-child fixture. If the table is
  dropped, swap the fixture to another CASCADE child (e.g. fsrs_cards).
- `tests/e2e/tests/journeys/ingest_recall_review.rs`
  `test_recall_search_modes`: constructs `SearchMode::Semantic` (w1b scope).
- `tests/e2e/Cargo.toml` still requests `features = ["embeddings",
  "vector-search"]` from vestige-core — drop when the features die.

## Notes for integration

- No e2e scenario exercised `match_context`/`search` tool aliases at this
  base (grep-verified); nothing to flip to removal-error assertions. The
  aliases exist only in source (`crates/vestige-mcp/src/server.rs`). HOWEVER,
  `cargo test -p vestige-mcp --lib` at this base already fails
  `server::tests::test_recall_lookup_matches_search_shape`
  ("recall(mode=lookup) must equal search byte-for-byte") — that IS a
  search-alias-expects-success test living in src tests; w1d should flip it
  to expect the removal error. Second pre-existing failure:
  `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`.
  Both verified pre-existing via `git stash` roundtrip at base 5fbe1df
  (lib build contains none of this branch's changes).
- `crates/vestige-core/src/storage/sqlite/embeddings.rs`:
  `embedding_model_matches_active` / `embedding_vector_for_active_model` are
  now unused (their only callers were deleted tests) — dead-code warning under
  default features; resolves itself when w1a deletes the embeddings module.
- `cargo check --tests --no-default-features` (workspace): FAILS in NON-TEST
  source, pre-existing at base: `crates/vestige-mcp/src/tools/search_unified.rs:1261`
  E0382 "use of moved value: `rerank_candidates`" (into_iter at 1257, reused at
  1261; the no-default-features branch of the reranker). Not fixed here per
  owner order — belongs to w1b/w1d. vestige-core's test build compiles clean
  under no-default-features apart from the dead-code warning above.

## Verification (this tree, default features)

- `cargo test -p vestige-core --lib`: 936 passed, 0 failed, 1 ignored.
- `cargo test -p vestige-e2e-tests --test comparative_benchmarks --test
  proof_of_superiority`: 23 + 4 passed, 0 failed.
- `cargo test -p vestige-mcp --lib`: 756 passed, 2 failed — both failures
  pre-existing at base (see Notes), unaffected by this branch.
- `cargo test -p vestige-mcp --test e2e_real_binary -- contradictions
  correction`: 2 passed (the two keyword-path conversions).
- `cargo check -p vestige-e2e-tests --tests`: clean.

# SCOPE-HANDOFF — causal_walk (build/w3a-causal-walk)

`causal_walk` REPLACES `backfill` as the advertised flagship. `backfill`
(module `crates/vestige-mcp/src/tools/backfill.rs`, core
`advanced/retroactive_backfill.rs`) is UNMODIFIED and stays dispatchable as a
hidden alias. Nothing else outside the files below was touched.

## Files

| File | Change |
| --- | --- |
| `crates/vestige-core/src/advanced/causal_walk.rs` | NEW — engine + storage assembly + promote + 10 unit tests |
| `crates/vestige-core/src/advanced/mod.rs` | `pub mod causal_walk;` + re-exports |
| `crates/vestige-mcp/src/tools/causal_walk.rs` | NEW — schema + execute + tests |
| `crates/vestige-mcp/src/tools/mod.rs` | `pub mod causal_walk;` |
| `crates/vestige-mcp/src/tools/compact.rs` | full-schema registry: `backfill` → `causal_walk` (registry tracks the ADVERTISED catalog; hidden names must not enter it) |
| `crates/vestige-mcp/src/server.rs` | catalog entry swapped (count stays 16), `causal_walk` dispatch arm added, `backfill` arm marked hidden alias, advertised-surface test updated |
| `crates/vestige-mcp/src/bin/cli.rs` | NEW subcommand block only: `CausalWalk` variant + dispatch arm + `run_causal_walk` handler |
| `SCOPE-HANDOFF.md` | this file |

## Mechanisms (start point → suspects)

1. `failing_test {name}` → records naming the test (test-file records) →
   commit records touching their path-tier entities → each commit's OTHER
   files are the suspects. Hop: `failing_test/co_touch`, hop text
   `<test-file> -> <other files>`.
2. `stack_frame {frame: "file:line"|"file"}` → commit records whose hunk
   spans (files / `file/symbol` symbols) cover the file → the LAST commit
   touching it before the failure anchor is the prime suspect (SZZ-lite;
   earlier touchers land in `rejected`). Hop: `stack_frame/last_toucher`.
3. `ci_run {run_id}` → `agent_traces` run via `storage.get_trace(run_id)` →
   failure channel = `SanhedrinVeto` claim text + `MemoryRetrieve`/`MemoryWrite`/
   veto `evidence_ids` records → their entities → records. Hop:
   `ci_run/failed_calls`.
4. `logged_write {node_id}` → the node → its `get_connections_for_memory`
   edges (neighbours seeded as edge evidence, surviving with 0 shared
   anchors) + records sharing its entities. Hops: `logged_write/edge`,
   `logged_write/shared_anchor`.
5. `version_range {worked_in, broke_in, repo}` → `git rev-list
   worked_in..broke_in` sha set (`git_records::parse_rev_list`) → RESTRICTOR:
   out-of-range commit seeds rejected `outside version range W..B`; in-range
   commits overlapping the anchor pool get hop `version_range/rev_list`.
   Alone (no failure start point) it yields no anchors → needs_report.
   Multiple distinct repos → needs_report (single repo per call).

## Core API (vestige-core::advanced::causal_walk)

```rust
pub const EVIDENCE_LINK_TYPE: &str = "evidence_of";
pub const MAX_HOPS: usize = 2;

pub enum StartPoint { FailingTest{name}, StackFrame{frame}, CiRun{run_id},
                      LoggedWrite{node_id}, VersionRange{worked_in,broke_in,repo} }
// serde: {"kind":"failing_test","name":...} etc. (tag="kind", snake_case)

pub struct CausalWalkRequest { pub scope: String, pub start_points: Vec<StartPoint>,
                               pub lookback_days: i64, pub scan_limit: i32 } // Default: user/30/500
pub struct PathHop { pub via: String, pub hop: String }
pub struct CausalCause { pub id: String, pub sha: Option<String>, pub score: f64,
                         pub path: Vec<PathHop>, pub shared_anchors: Vec<String>,
                         #[serde(skip)] pub evidence_to: Vec<String> }
pub struct NeedsReport { pub missing: Vec<String>, pub required_start_points: Vec<String> }
pub struct WalkRejection { pub id: String, pub reason: String, pub shared_anchors: usize }
pub struct CausalWalkResult { pub causes: Vec<CausalCause>,
                              pub needs_report: Option<NeedsReport>,
                              pub rejected: Vec<WalkRejection> }

pub fn walk_storage(storage: &Storage, req: &CausalWalkRequest) -> Result<CausalWalkResult, String>;
pub fn persist_evidence_edges(storage: &Storage, result: &CausalWalkResult)
    -> Result<Vec<(String,String)>, String>;  // cause -> evidence, link_type "evidence_of"
```

`CausalWalkOptions { lookback_days, max_causes, max_rejections }::walk(starts, records, ctx)`
is the pure engine (records = `WalkRecord`, ctx = runs/edges/rev-lists) if you
need it without a store.

## Output schema (MCP `causal_walk`)

```
{ causes: [{id, sha?, score, path: [{via, hop}], shared_anchors, content_preview}],
  needs_report?: {missing, required_start_points},
  rejected: [{id, reason, shared_anchors, content_preview}],   // top 3
  promote: {edges_persisted, edges: [[cause, evidence]], link_type: "evidence_of"},
  preview, scope, headline, evidence_status: "hypothesis", causality_verified: false }
```

No start point / unanchored handle → `needs_report` (never an error, never
guessed). Promote is the ONLY write path.

## Ranking (ported from run_trail)

score = Σ probabilistic-IDF(anchor) × tier weight (Path 1.0 / Code 0.9 /
Version 0.6 / Word 0.3, df over the in-window candidate pool) + 0.3 recency
term + 0.25 change-record bonus. Sort: score desc → change-record first →
older first. Window: candidate created_at ≤ failure anchor (= newest
start-point-anchored record, else now) and within lookback_days (default 30).

## CLI

`vestige causal-walk --failing-test NAME | --stack-frame F[:L] | --ci-run ID |
--logged-write ID | (--git-repo P --worked-in T --broke-in T)`
`[--lookback-days N] [--no-promote] [--scope S] [--json]`

## Tests

- `cargo test -p vestige-core --lib causal_walk` — 10 tests: co-touch walk,
  stack_frame last-toucher, version_range restriction (real temp git repo),
  no-start refusal, wrong-handle refusal (x2), ci_run trace walk,
  logged_write edge walk, promote vs preview, unresolvable range refusal,
  record-format parsing.
- `cargo test -p vestige-mcp --lib causal` — schema variant test + tokio
  end-to-end (temp storage): needs_report, preview writes nothing,
  promote persists `evidence_of` edges.
- Server surface: advertised count stays 16 (`causal_walk` in, `backfill`
  hidden); full-schema registry test, wire-budget test, and hints test
  updated accordingly.

## Notes for adjacent agents

- Do NOT re-add `backfill` to the advertised catalog or `compact::full_schema`
  registry — `full_schema_registry_matches_the_advertised_catalog` fails if
  catalog and registry diverge.
- Richer edge APIs: keep persisting through `save_connection` with
  `link_type = "evidence_of"` (constant `causal_walk::EVIDENCE_LINK_TYPE`) so
  `memory_connections` stays compatible; `CausalCause::evidence_to` carries
  the per-cause link targets (serde-skipped).
- Supersession is intentionally NOT followed in causal_walk (MVP bound);
  commit records upsert idempotently via `(git, repo#sha)`.
# SCOPE-HANDOFF — causal_walk (build/w3a-causal-walk)

`causal_walk` REPLACES `backfill` as the advertised flagship. `backfill`
(module `crates/vestige-mcp/src/tools/backfill.rs`, core
`advanced/retroactive_backfill.rs`) is UNMODIFIED and stays dispatchable as a
hidden alias. Nothing else outside the files below was touched.

## Files

| File | Change |
| --- | --- |
| `crates/vestige-core/src/advanced/causal_walk.rs` | NEW — engine + storage assembly + promote + 10 unit tests |
| `crates/vestige-core/src/advanced/mod.rs` | `pub mod causal_walk;` + re-exports |
| `crates/vestige-mcp/src/tools/causal_walk.rs` | NEW — schema + execute + tests |
| `crates/vestige-mcp/src/tools/mod.rs` | `pub mod causal_walk;` |
| `crates/vestige-mcp/src/tools/compact.rs` | full-schema registry: `backfill` → `causal_walk` (registry tracks the ADVERTISED catalog; hidden names must not enter it) |
| `crates/vestige-mcp/src/server.rs` | catalog entry swapped (count stays 16), `causal_walk` dispatch arm added, `backfill` arm marked hidden alias, advertised-surface test updated |
| `crates/vestige-mcp/src/bin/cli.rs` | NEW subcommand block only: `CausalWalk` variant + dispatch arm + `run_causal_walk` handler |
| `SCOPE-HANDOFF.md` | this file |

## Mechanisms (start point → suspects)

1. `failing_test {name}` → records naming the test (test-file records) →
   commit records touching their path-tier entities → each commit's OTHER
   files are the suspects. Hop: `failing_test/co_touch`, hop text
   `<test-file> -> <other files>`.
2. `stack_frame {frame: "file:line"|"file"}` → commit records whose hunk
   spans (files / `file/symbol` symbols) cover the file → the LAST commit
   touching it before the failure anchor is the prime suspect (SZZ-lite;
   earlier touchers land in `rejected`). Hop: `stack_frame/last_toucher`.
3. `ci_run {run_id}` → `agent_traces` run via `storage.get_trace(run_id)` →
   failure channel = `SanhedrinVeto` claim text + `MemoryRetrieve`/`MemoryWrite`/
   veto `evidence_ids` records → their entities → records. Hop:
   `ci_run/failed_calls`.
4. `logged_write {node_id}` → the node → its `get_connections_for_memory`
   edges (neighbours seeded as edge evidence, surviving with 0 shared
   anchors) + records sharing its entities. Hops: `logged_write/edge`,
   `logged_write/shared_anchor`.
5. `version_range {worked_in, broke_in, repo}` → `git rev-list
   worked_in..broke_in` sha set (`git_records::parse_rev_list`) → RESTRICTOR:
   out-of-range commit seeds rejected `outside version range W..B`; in-range
   commits overlapping the anchor pool get hop `version_range/rev_list`.
   Alone (no failure start point) it yields no anchors → needs_report.
   Multiple distinct repos → needs_report (single repo per call).

## Core API (vestige-core::advanced::causal_walk)

```rust
pub const EVIDENCE_LINK_TYPE: &str = "evidence_of";
pub const MAX_HOPS: usize = 2;

pub enum StartPoint { FailingTest{name}, StackFrame{frame}, CiRun{run_id},
                      LoggedWrite{node_id}, VersionRange{worked_in,broke_in,repo} }
// serde: {"kind":"failing_test","name":...} etc. (tag="kind", snake_case)

pub struct CausalWalkRequest { pub scope: String, pub start_points: Vec<StartPoint>,
                               pub lookback_days: i64, pub scan_limit: i32 } // Default: user/30/500
pub struct PathHop { pub via: String, pub hop: String }
pub struct CausalCause { pub id: String, pub sha: Option<String>, pub score: f64,
                         pub path: Vec<PathHop>, pub shared_anchors: Vec<String>,
                         #[serde(skip)] pub evidence_to: Vec<String> }
pub struct NeedsReport { pub missing: Vec<String>, pub required_start_points: Vec<String> }
pub struct WalkRejection { pub id: String, pub reason: String, pub shared_anchors: usize }
pub struct CausalWalkResult { pub causes: Vec<CausalCause>,
                              pub needs_report: Option<NeedsReport>,
                              pub rejected: Vec<WalkRejection> }

pub fn walk_storage(storage: &Storage, req: &CausalWalkRequest) -> Result<CausalWalkResult, String>;
pub fn persist_evidence_edges(storage: &Storage, result: &CausalWalkResult)
    -> Result<Vec<(String,String)>, String>;  // cause -> evidence, link_type "evidence_of"
```

`CausalWalkOptions { lookback_days, max_causes, max_rejections }::walk(starts, records, ctx)`
is the pure engine (records = `WalkRecord`, ctx = runs/edges/rev-lists) if you
need it without a store.

## Output schema (MCP `causal_walk`)

```
{ causes: [{id, sha?, score, path: [{via, hop}], shared_anchors, content_preview}],
  needs_report?: {missing, required_start_points},
  rejected: [{id, reason, shared_anchors, content_preview}],   // top 3
  promote: {edges_persisted, edges: [[cause, evidence]], link_type: "evidence_of"},
  preview, scope, headline, evidence_status: "hypothesis", causality_verified: false }
```

No start point / unanchored handle → `needs_report` (never an error, never
guessed). Promote is the ONLY write path.

## Ranking (ported from run_trail)

score = Σ probabilistic-IDF(anchor) × tier weight (Path 1.0 / Code 0.9 /
Version 0.6 / Word 0.3, df over the in-window candidate pool) + 0.3 recency
term + 0.25 change-record bonus. Sort: score desc → change-record first →
older first. Window: candidate created_at ≤ failure anchor (= newest
start-point-anchored record, else now) and within lookback_days (default 30).

## CLI

`vestige causal-walk --failing-test NAME | --stack-frame F[:L] | --ci-run ID |
--logged-write ID | (--git-repo P --worked-in T --broke-in T)`
`[--lookback-days N] [--no-promote] [--scope S] [--json]`

## Tests

- `cargo test -p vestige-core --lib causal_walk` — 10 tests: co-touch walk,
  stack_frame last-toucher, version_range restriction (real temp git repo),
  no-start refusal, wrong-handle refusal (x2), ci_run trace walk,
  logged_write edge walk, promote vs preview, unresolvable range refusal,
  record-format parsing.
- `cargo test -p vestige-mcp --lib causal` — schema variant test + tokio
  end-to-end (temp storage): needs_report, preview writes nothing,
  promote persists `evidence_of` edges.
- Server surface: advertised count stays 16 (`causal_walk` in, `backfill`
  hidden); full-schema registry test, wire-budget test, and hints test
  updated accordingly.

## Notes for adjacent agents

- Do NOT re-add `backfill` to the advertised catalog or `compact::full_schema`
  registry — `full_schema_registry_matches_the_advertised_catalog` fails if
  catalog and registry diverge.
- Richer edge APIs: keep persisting through `save_connection` with
  `link_type = "evidence_of"` (constant `causal_walk::EVIDENCE_LINK_TYPE`) so
  `memory_connections` stays compatible; `CausalCause::evidence_to` carries
  the per-cause link targets (serde-skipped).
- Supersession is intentionally NOT followed in causal_walk (MVP bound);
  commit records upsert idempotently via `(git, repo#sha)`.

# SCOPE HANDOFF — build/w1c-ingest-flip (ingest-flip: no similarity, exact-equality nomination)

Branch: `build/w1c-ingest-flip`, base = `main @ 5fbe1df`. Not pushed.

Owner decision implemented: NO similarity anywhere; dedup/merge nomination =
exact equality only (identical content hash / same declared source key / exact
entity-set equality); edges only from declared sources.

## What changed in this branch's scope

- `crates/vestige-core/src/advanced/prediction_error.rs`
  - `evaluate` / `evaluate_with_intent` dropped the `new_embedding` param.
  - `CandidateMemory.embedding` field deleted.
  - `cosine_similarity` deleted; replaced by `content_similarity` (Dice
    coefficient over lowercased alphanumeric tokens > 2 chars). Exported via
    `advanced::content_similarity`.
  - `NEAR_IDENTICAL_THRESHOLD` recalibrated 0.92 → 0.85 for the Dice scale so
    benign near-paraphrases still Reinforce; contradiction-marker check runs
    before the threshold, so corrections never reinforce.
- `crates/vestige-core/src/advanced/merge_supersede.rs`
  - `W_EMBEDDING` removed. New weights: `W_TAGS = 0.40`, `W_TOKENS = 0.60`
    (sum = 1.0). Comment states nomination is exact-equality-first and scores
    are tie-breakers/review labels only.
  - `MatchSignals.embedding_similarity` field REMOVED (breaking API change).
  - `score_pair` signature is now `score_pair(a_tags, b_tags, a_content, b_content)`.
- `crates/vestige-core/src/advanced/dreams.rs`
  - Both cosine branches deleted; `tag_similarity*0.4 + content_word_similarity*0.6`
    is now the whole computation. Module-level `cosine_similarity` fn deleted.
  - `DreamMemory.embedding` field KEPT but inert (out-of-scope constructors
    still set it — see below).
- `crates/vestige-core/src/storage/sqlite/merge.rs`
  - `merge_candidates`: O(n²) cosine scan deleted (was embeddings+vector-search
    cfg-gated). Now UNGATED and nominates via: (1) SQL group-by on
    `COALESCE(content_hash, content)`; (2) SQL group-by on the source key at
    the same granularity as the store's UNIQUE index
    `(source_system, COALESCE(source_project,''), source_id)`; (3) exact
    equality of non-empty `retroactive_backfill::extract_entities` sets
    (in-memory compare). Classification is advisory — nominated clusters are
    surfaced even when labelled NonMatch (shared source key with diverged
    content is review-worthy; old drop-on-NonMatch was a cosine-era guard).
  - `pair_similarity` deleted (embedding-based); plan_merge / plan_supersede /
    plan_reconsolidation now call the 4-arg `score_pair` directly.
  - New test module `exact_nomination_tests` (6 tests).
- `crates/vestige-core/src/storage/sqlite/ingest.rs` (embedding paths only)
  - Smart ingest: embedding-runtime preflight, `get_document_embedding`, and
    `semantic_search_raw` candidate retrieval REMOVED. Candidate selection is
    keyword-only via `Storage::search` (FTS5/BM25, public API in search.rs —
    not modified). `get_node_embedding` per-candidate fetch removed.
  - KEPT (cfg-gated, deletion belongs to the embeddings agent):
    `generate_embedding_for_node` after raw insert, and vector invalidation +
    regeneration in `update_node_content_unchecked`. These are vector STORAGE
    hygiene, not similarity decisions; removing them broke
    `embedding::lifecycle` and `peer_content_edit_invalidates_vectors` tests
    that the embeddings-deletion agent will remove wholesale.
  - NOTE: the whole `smart_ingest*` chain is still
    `#[cfg(all(feature = "embeddings", feature = "vector-search"))]`-gated. It
    no longer needs embeddings at all — the embeddings agent should un-gate it
    when the feature flags go.
- `crates/vestige-mcp/src/tools/dedup.rs`
  - `find_duplicates` (`execute`): cosine clustering + UnionFind deleted;
    now UNGATED exact-equality grouping (content-hash identity + source key at
    UNIQUE-index granularity). `similarity_threshold` parameter removed from
    the schema (unknown args are ignored by serde, so old callers keep
    working). Output: `similarityToAnchor` → `matchRelation`
    (anchor|content|source_key|transitive); `totalWithEmbeddings` →
    `totalScanned`; `threshold`/`pairsChecked` removed.
  - plan/apply/undo/verdict/tag_*/protect/policy flows untouched.

## Forced out-of-scope compile fixes (minimal, mechanical)

- `crates/vestige-core/src/advanced/mod.rs`: re-export
  `prediction_error::cosine_similarity` → `content_similarity`.
- `crates/vestige-mcp/src/tools/merge.rs` (2 sites): dropped the
  `embeddingSimilarity` output lines referencing the removed
  `MatchSignals.embedding_similarity` field. Everything else in that file is
  untouched; NOTE its `merge_candidates_schema()` may still advertise a
  `similarity_threshold`-style knob cosmetically (not compile-relevant).

## For the embeddings-deletion agent (cross-scope notes)

- After deleting `storage/sqlite/embeddings.rs` and the `embeddings/` module:
  - `vestige_core::cosine_similarity` re-export at `lib.rs:572` must go.
  - `get_document_embedding` is already dead code (warning) — disappears with
    the module.
  - Remaining callers of `storage.get_node_embedding(...)` OUTSIDE my scope:
    `crates/vestige-mcp/src/tools/dream.rs` (~line 88) and
    `crates/vestige-mcp/src/dashboard/handlers.rs` (~line 1863) — they only
    populate the now-inert `DreamMemory.embedding`; delete those calls and the
    field.
  - `lifecycle.rs` / `search.rs` untouched per scope (search.rs `hybrid_search`
    etc. still reference `semantic_search_raw`).
  - Un-gate the `smart_ingest*` chain + merge plan/apply/undo cfg gates when
    the features are dropped (see above).
  - `knowledge_nodes.has_embedding` / `node_embeddings` /
    `embedding_profile_vectors` schema: my files no longer maintain them
    except the cfg-gated blocks noted above.

## Test status (default features)

- `cargo test -p vestige-core`: 971 passed, 0 failed (1 ignored, pre-existing).
- `cargo test -p vestige-mcp`: 759 passed, 2 failed:
  - `server::tests::test_recall_lookup_matches_search_shape`
  - `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`
  Both are verified PRE-EXISTING on base `5fbe1df` (checked in a clean
  worktree); they are fallout from commit 7bc2516's search-tool removal and
  backfill rework, not from this branch.
- All in-scope test modules green: prediction_error (16), merge_supersede,
  dreams, sqlite::merge::exact_nomination_tests (6), mcp tools::dedup (12).
# SCOPE HANDOFF — build/w1c-ingest-flip (ingest-flip: no similarity, exact-equality nomination)

Branch: `build/w1c-ingest-flip`, base = `main @ 5fbe1df`. Not pushed.

Owner decision implemented: NO similarity anywhere; dedup/merge nomination =
exact equality only (identical content hash / same declared source key / exact
entity-set equality); edges only from declared sources.

## What changed in this branch's scope

- `crates/vestige-core/src/advanced/prediction_error.rs`
  - `evaluate` / `evaluate_with_intent` dropped the `new_embedding` param.
  - `CandidateMemory.embedding` field deleted.
  - `cosine_similarity` deleted; replaced by `content_similarity` (Dice
    coefficient over lowercased alphanumeric tokens > 2 chars). Exported via
    `advanced::content_similarity`.
  - `NEAR_IDENTICAL_THRESHOLD` recalibrated 0.92 → 0.85 for the Dice scale so
    benign near-paraphrases still Reinforce; contradiction-marker check runs
    before the threshold, so corrections never reinforce.
- `crates/vestige-core/src/advanced/merge_supersede.rs`
  - `W_EMBEDDING` removed. New weights: `W_TAGS = 0.40`, `W_TOKENS = 0.60`
    (sum = 1.0). Comment states nomination is exact-equality-first and scores
    are tie-breakers/review labels only.
  - `MatchSignals.embedding_similarity` field REMOVED (breaking API change).
  - `score_pair` signature is now `score_pair(a_tags, b_tags, a_content, b_content)`.
- `crates/vestige-core/src/advanced/dreams.rs`
  - Both cosine branches deleted; `tag_similarity*0.4 + content_word_similarity*0.6`
    is now the whole computation. Module-level `cosine_similarity` fn deleted.
  - `DreamMemory.embedding` field KEPT but inert (out-of-scope constructors
    still set it — see below).
- `crates/vestige-core/src/storage/sqlite/merge.rs`
  - `merge_candidates`: O(n²) cosine scan deleted (was embeddings+vector-search
    cfg-gated). Now UNGATED and nominates via: (1) SQL group-by on
    `COALESCE(content_hash, content)`; (2) SQL group-by on the source key at
    the same granularity as the store's UNIQUE index
    `(source_system, COALESCE(source_project,''), source_id)`; (3) exact
    equality of non-empty `retroactive_backfill::extract_entities` sets
    (in-memory compare). Classification is advisory — nominated clusters are
    surfaced even when labelled NonMatch (shared source key with diverged
    content is review-worthy; old drop-on-NonMatch was a cosine-era guard).
  - `pair_similarity` deleted (embedding-based); plan_merge / plan_supersede /
    plan_reconsolidation now call the 4-arg `score_pair` directly.
  - New test module `exact_nomination_tests` (6 tests).
- `crates/vestige-core/src/storage/sqlite/ingest.rs` (embedding paths only)
  - Smart ingest: embedding-runtime preflight, `get_document_embedding`, and
    `semantic_search_raw` candidate retrieval REMOVED. Candidate selection is
    keyword-only via `Storage::search` (FTS5/BM25, public API in search.rs —
    not modified). `get_node_embedding` per-candidate fetch removed.
  - KEPT (cfg-gated, deletion belongs to the embeddings agent):
    `generate_embedding_for_node` after raw insert, and vector invalidation +
    regeneration in `update_node_content_unchecked`. These are vector STORAGE
    hygiene, not similarity decisions; removing them broke
    `embedding::lifecycle` and `peer_content_edit_invalidates_vectors` tests
    that the embeddings-deletion agent will remove wholesale.
  - NOTE: the whole `smart_ingest*` chain is still
    `#[cfg(all(feature = "embeddings", feature = "vector-search"))]`-gated. It
    no longer needs embeddings at all — the embeddings agent should un-gate it
    when the feature flags go.
- `crates/vestige-mcp/src/tools/dedup.rs`
  - `find_duplicates` (`execute`): cosine clustering + UnionFind deleted;
    now UNGATED exact-equality grouping (content-hash identity + source key at
    UNIQUE-index granularity). `similarity_threshold` parameter removed from
    the schema (unknown args are ignored by serde, so old callers keep
    working). Output: `similarityToAnchor` → `matchRelation`
    (anchor|content|source_key|transitive); `totalWithEmbeddings` →
    `totalScanned`; `threshold`/`pairsChecked` removed.
  - plan/apply/undo/verdict/tag_*/protect/policy flows untouched.

## Forced out-of-scope compile fixes (minimal, mechanical)

- `crates/vestige-core/src/advanced/mod.rs`: re-export
  `prediction_error::cosine_similarity` → `content_similarity`.
- `crates/vestige-mcp/src/tools/merge.rs` (2 sites): dropped the
  `embeddingSimilarity` output lines referencing the removed
  `MatchSignals.embedding_similarity` field. Everything else in that file is
  untouched; NOTE its `merge_candidates_schema()` may still advertise a
  `similarity_threshold`-style knob cosmetically (not compile-relevant).

## For the embeddings-deletion agent (cross-scope notes)

- After deleting `storage/sqlite/embeddings.rs` and the `embeddings/` module:
  - `vestige_core::cosine_similarity` re-export at `lib.rs:572` must go.
  - `get_document_embedding` is already dead code (warning) — disappears with
    the module.
  - Remaining callers of `storage.get_node_embedding(...)` OUTSIDE my scope:
    `crates/vestige-mcp/src/tools/dream.rs` (~line 88) and
    `crates/vestige-mcp/src/dashboard/handlers.rs` (~line 1863) — they only
    populate the now-inert `DreamMemory.embedding`; delete those calls and the
    field.
  - `lifecycle.rs` / `search.rs` untouched per scope (search.rs `hybrid_search`
    etc. still reference `semantic_search_raw`).
  - Un-gate the `smart_ingest*` chain + merge plan/apply/undo cfg gates when
    the features are dropped (see above).
  - `knowledge_nodes.has_embedding` / `node_embeddings` /
    `embedding_profile_vectors` schema: my files no longer maintain them
    except the cfg-gated blocks noted above.

## Test status (default features)

- `cargo test -p vestige-core`: 971 passed, 0 failed (1 ignored, pre-existing).
- `cargo test -p vestige-mcp`: 759 passed, 2 failed:
  - `server::tests::test_recall_lookup_matches_search_shape`
  - `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`
  Both are verified PRE-EXISTING on base `5fbe1df` (checked in a clean
  worktree); they are fallout from commit 7bc2516's search-tool removal and
  backfill rework, not from this branch.
- All in-scope test modules green: prediction_error (16), merge_supersede,
  dreams, sqlite::merge::exact_nomination_tests (6), mcp tools::dedup (12).

- The sibling branch's selftest + forgotten_lesson tools (16→18) are NOT in
  this base (ba94298 advertises 16 tools); catalog tests pin 16 here and pass.
  Nothing to keep yet — they arrive with that branch's merge.
- vestige-core still emits 3 clippy warnings (dead alias, doc-comment,
  while-let) — out of scope, untouched.
- Behavioral note: with keyword-only retrieval, four rare word-tier shared
  entities can outscore one code-tier entity in backfill ranking (see the
  backfill.rs test comment). That weighting lives in core.

# SCOPE-HANDOFF — build/t4-verify-harness (determinism/parity gate for the SQLite→STRATA default flip)

Branch: `build/t4-verify-harness` (base = main @ 82f0cf5). Committed locally;
NOT pushed. Adds the verification + differential harness that gates flipping
the default store from SQLite to STRATA.

## What landed

1. **`crates/strata-verify`** (standalone, own `[workspace]` table like its
   siblings): `verify_store(dir) -> VerifyReport { log_tail_ok,
   checkpoint_chain_ok, state_root_matches, gate_verdicts_rederived, gaps:
   Vec<GapKind>, duration_ms, failures }`. Every field is RECOMPUTED from raw
   bytes — `strata_kernel::verify_with_head` plus an independent explicit
   fold pass (per-event hashes, seq monotonicity, segment-versioned replay,
   state roots at every checkpoint, head anchor), gate frame density + head
   anchor, verdict rederivation under the stored `gate.policy`
   (`strata_gate::rederive_verdicts`), and the structural sweep
   (`strata_gate::sweep`). Every `VerifyFailure` Display names the offending
   seq / checkpoint log_seq. `layout.rs` defines the on-disk contract
   (`kernel.log`, `kernel.checkpoints`, `kernel.head`, `gate.log`,
   `gate.policy`, `gate.head`; framing magic `STVFv1\0\0`) the sibling
   `strata-store` crate must emit, with the writer (`write_store`) the
   harness uses until that crate merges.
2. **`vestige strata-verify <dir>`** — surgical block in
   `crates/vestige-mcp/src/bin/cli.rs` (JSON report, exit 1 on failure);
   `strata-verify` path dep added to vestige-mcp. Standalone bin
   `strata-verify` also exists in the crate.
3. **`tests/differential`** (`vestige-differential-tests`, workspace member):
   the cross-engine driver. Takes an op script (JSON `{ingest,edge,review,
   suppress}` list; `Script::from_json`), replays against BOTH engines —
   vestige-core SQLite `Storage` (real ingest / typed edges / `mark_reviewed`
   / `suppress_memory` + the real `FSRSScheduler` core as a pinned-time
   shadow fold) and strata (GateRuntime propose→gate→effect over MemLog +
   kernel v1 fold, then materialized + `verify_store`d). Asserts: same node
   COUNTS, same content digests sorted, same edge sets sorted (SQLite side
   is a store read-back), same suppressed sets, same per-card integer FSRS
   trajectory (review_count, lapse_count; phase on Hard-free scripts). FSRS
   stability/difficulty are compared as Q32.32-quantized streams RUN-OVER-RUN
   per engine (byte-identical snapshots), never cross-engine — the engines
   pin different weight tables by design. Seeded SplitMix64 fuzz ≥ 20 ops.
   Manual runner: `cargo run -p vestige-differential-tests --bin differential
   -- --seed N`.

## Root-workspace wiring (deliberate)

`crates/strata-{kernel,gate,verify}` are in the root `Cargo.toml` `exclude`
list (like fastembed-rs) so the workspace can path-depend on them WITHOUT
absorbing them; they keep their own workspace tables and lockfiles. When the
`crates/strata` integration lands, drop their `[workspace]` tables, remove
the excludes, and move them into `members`.

## Findings the flip decision should read

- `mark_reviewed` derives elapsed from the WALL CLOCK
  (`days_since_review(last_accessed)`), so (a) stored stability is not
  byte-reproducible run-over-run, and (b) under fast test cadence every
  post-first review classifies same-day and the STORE's lapse counter
  under-counts vs any pinned-time reference. The harness records this in
  `extras.store_lapses_total` instead of asserting it away. This is the
  concrete non-determinism the default flip retires.
- Known engine delta: a Hard (rating 2) review at elapsed ≥ 1 day moves a
  card to `Review` in vestige-core but preserves phase in the strata kernel.
  Phase parity is asserted only on Hard-free scripts.
- 1 SEQ = 1 day is the harness's shared time base; `elapsed_seq >= 1` keeps
  both engines off their same-day/short-term paths so lapse counting aligns.

## Test status (all green)

- `cargo test -p strata-verify` (standalone): 7 passed — clean-store verify,
  sealed-segment byte flip (names the seq), checkpoint byte flip (names
  log_seq), forged EFFECT without GATE (re-anchor-consistent; sweep flags
  `orphan_effect` naming the forged seq; rederivation still passes), tampered
  GATE verdict (rederivation names the gate seq), missing artifact, trailing
  event past head checkpoint.
- `cargo test -p vestige-differential-tests`: 5 passed — fixed-script full
  parity (incl. phase), two-runs byte-identical per engine, 3-seed fuzz
  parity + determinism, 6-seed materialize-and-verify-green, script JSON
  round-trip.
- `cargo test --workspace`: ALL PASS (exit 0), including 789-test
  vestige-mcp lib. `cargo clippy` clean for both new crates (0 warnings).
- CLI smoke: `vestige strata-verify <materialized-store>` prints the report
  and exits 0/1 by `report.ok()`.
# SCOPE-HANDOFF — build/w2b-edge-schema (typed edge vocabulary)

# SCOPE-HANDOFF — build/w1d-absorb (vestige-mcp cleanup after core vector removal)

Branch: `build/w1d-absorb` (off main tip ba94298). Scope: `crates/vestige-mcp`
only; vestige-core untouched. Committed locally; nothing pushed.
(Replaces the w2b-edge-schema handoff that was already merged at 45cd598.)

## Status

- `cargo check -p vestige-mcp` — 0 errors, 0 warnings for vestige-mcp
  (1 remaining warning is vestige-core's dead `EmbeddingProfileMigrationRow`
  alias, out of scope).
- `cargo test -p vestige-mcp --lib` — 760 passed, 0 failed.
- `cargo clippy -p vestige-mcp --all-targets` — clean for vestige-mcp
  (remaining warnings are all in vestige-core).

## Error classes fixed (37 lib errors + test-target fallout)

1. `cognitive.rs` — removed `AdaptiveEmbedder` and `Reranker` import/field/init
   (cfg'd reranker arm included); `COGNITIVE_MODULE_COUNT` now 16+10 (+1
   vector-search-gated TemporalSearcher, which still exists in core); test
   updated.
2. `dashboard/handlers.rs` — deleted the entire embedding-profile section
   (6 handlers, request type, snapshot/receipt helpers, ~420 lines) and their
   6 routes in `dashboard/mod.rs`; dream builder now `embedding: None`.
3. `tools/smart_ingest.rs` — removed `ContentType` import + both
   `ContentType::detect` calls + the `warming` response block.
4. `tools/dream.rs` — `embedding: None` (content-word similarity path stays).
5. `resources/memory.rs` — `embeddingServiceReady` stubbed `false`.
6. `src/main.rs` — removed the cfg'd embeddings init block
   (embedding_model_cached / init_embeddings / generate_embeddings) and the
   cross-encoder reranker load block (Reranker::load_cross_encoder); stale
   comments updated.
7. `bin/cli.rs` — deleted the `embeddings` subcommand tree (EmbeddingCommands,
   dispatch, run_embeddings_* helpers, artifact-root/confirmation helpers), the
   `--embedding-from` Recall flag and its profile-attach logic, the
   `LEGACY_NOMIC_PROFILE_ID` preflight SQL, and all `is_embedding_ready` uses;
   backfill contrast mode and `vestige ingest` are keyword-only (cfg'd semantic
   branches removed); run_consolidate note simplified.
8. Deleted `tools/context.rs` (match_context) and `tools/warming.rs`
   (+ mod entries, server.rs dispatch arm, warming response blocks in
   search_unified/smart_ingest, their tests).
9. `server.rs` — removed the `explore_connections` hidden-alias dispatch arm
   (graph tool + graph_unified's internal use of explore stay); removed its
   three trace_recorder entries (is_retrieval_tool + two evidence maps).
10. Remaining `hybrid_search(_filtered)` call sites compile against core's
    keyword-only implementations — no mcp-side change needed.
- `tools/search_unified.rs` — fixed E0382 duplicate `rerank_candidates` move
  (leftover double `reranked_results` binding).
- `tools/cross_reference.rs` — removed the cfg'd reranker block in
  `retrieve_and_rank_candidates`.
- `tools/maintenance.rs` — `embeddingReady: false`; consolidate phase
  "embeddings" arm removed; maintain schema enum/error text updated.
- `Cargo.toml` — declared `embeddings = []` / `vector-search = []` as always-off
  stubs (mirrors core's stub) so legacy cfg sites don't trip `unexpected_cfgs`
  (78 → 0 warnings).

## Test fallout fixed

- Deleted 5 embedding-profile handler tests + test-mod imports in handlers.rs.
- `maintain.rs`: embeddings-phase test rewritten as
  `bounded_phases_preview_by_default_and_rejects_misapplied_controls`
  (lifecycle phase previews by default; "embeddings" phase now rejected).
- `server.rs::test_recall_lookup_matches_search_shape`: now compares
  `recall(default)` vs `recall(mode=lookup)` — the `search` alias was already
  a removal stub at base ba94298, so the old byte-comparison could never pass.
- `tools/backfill.rs` live test: the "shares NO entity" distractor fixture
  actually shared 4 word-tier entities under core's tiered extractor (w1c,
  commit 7bc2516) and outranked the env-var cause (4 word-idf > 1 code-idf).
  Reworded the distractor ("An outage hit the billing system last month") to
  match its documented zero-shared-entity intent; all other assertions
  unchanged.

## Notes / not finished

pub enum EdgeKind { Touched, AnchoredTo, DerivedFrom, Supersedes, Corrects,
                    ClosedBy, ProjectedTo, EvidenceOf }
impl EdgeKind {
    pub const ALL: [EdgeKind; 8];
    pub fn as_str(self) -> &'static str;
    pub fn parse(link_type: &str) -> Option<EdgeKind>;   // None outside vocabulary
}

pub enum EdgeDirection { Outgoing, Incoming, Both }

#[derive(Serialize, Deserialize, Default)]
pub struct EdgeMeta { pub sha: Option<String>, pub span: Option<String>,
                      pub run_id: Option<String> }

pub struct TypedEdge {
    pub source_id: String, pub target_id: String,
    pub link_type: String,          // validated against vocabulary on save
    pub strength: f64,              // typed edges default to 1.0
    pub meta: EdgeMeta,             // -> edge_meta JSON column
    pub created_by_run: Option<String>,
    pub created_at: DateTime<Utc>, pub last_activated: DateTime<Utc>,
    pub activation_count: i32,
}
impl TypedEdge {
    pub fn new(source_id: impl Into<String>, target_id: impl Into<String>,
               link_type: &str) -> Self;
}

pub struct PurgeTombstone { pub purged_id: String, pub purged_at: DateTime<Utc>,
    pub reason: Option<String>, pub prior_content_hash: Option<String> }
```

## SqliteMemoryStore methods (`crates/vestige-core/src/storage/edges.rs`)

```rust
impl SqliteMemoryStore {
    /// Rejects link_type outside TYPED_EDGE_VOCABULARY (Err::InvalidEdge).
    /// INSERT OR REPLACE into memory_connections incl. edge_meta/created_by_run.
    /// FKs enforced: both endpoints must exist in knowledge_nodes.
    pub fn save_typed_edge(&self, edge: &TypedEdge) -> Result<()>;

    /// Typed (vocabulary-only) edges incident to node_id by direction.
    /// Legacy connections stay invisible here.
    pub fn edges_for(&self, node_id: &str, dir: EdgeDirection) -> Result<Vec<TypedEdge>>;

    /// Forward BFS (source->target) over the given kinds. Cycle-safe, exact,
    /// depth-bounded; returns reached ids sorted, start excluded; empty
    /// link_types => empty result.
    pub fn edge_reachability(&self, start_id: &str, link_types: &[EdgeKind],
                             max_depth: usize) -> Result<Vec<String>>;

    /// root_id + derived_from closure, sorted. READ-ONLY: deletes/writes
    /// nothing; output is the review-gate input for a retire decision.
    pub fn retire_subgraph(&self, root_id: &str) -> Result<Vec<String>>;

    /// Writes purge_tombstones row; prior_content_hash = SHA-256 of the node's
    /// content if the row still exists (simulated purge), else NULL.
    /// Re-record replaces (one row per id).
    pub fn record_tombstone(&self, purged_id: &str, reason: &str) -> Result<()>;

    pub fn get_purge_tombstone(&self, purged_id: &str) -> Result<Option<PurgeTombstone>>;
}
```

## Notes for downstream agents

- Edge rows share the `memory_connections` PK `(source_id, target_id)` with
  legacy connections; a typed save overwrites a legacy row with the same pair.
- Node ids for files/symbols/spans (`anchored_to`, `touched`) must be
  `knowledge_nodes` rows — decide the node-creation convention before wiring
  the MCP surface.
- Review gating for `supersedes`/`corrects` is policy for the tools layer;
  storage only guarantees vocabulary validation.
- `evidence_of` replaces `backfill_candidate` for NEW writes; nothing
  rewrites existing `backfill_candidate` rows (out of scope here).
- Tests: `cargo test -p vestige-core --lib edges` — 4 tests (typed
  save/reject, diamond+cycle reachability, non-deleting retire, tombstone on
  simulated purge). Migration replay/idempotence guards pass (35 tests).

# SCOPE-HANDOFF — w3d: selftest + forgotten_lesson

Branch `build/w3d-selftest-forgotten` (base `main` @ `5fbe1df`). All additions
are append-only except three marked test-constant updates forced by parity
guards; nothing outside the listed files was touched.

## New exposed APIs

### `vestige-mcp` MCP tools

| Tool | Module | Signature |
|---|---|---|
| `selftest` | `crates/vestige-mcp/src/tools/selftest.rs` | `pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String>` — no arguments |
| `forgotten_lesson` | `crates/vestige-mcp/src/tools/forgotten_lesson.rs` | same shape; args `{failure_id (required), scope?, scan_limit?}` |

`tools::selftest::schema()`, `tools::forgotten_lesson::schema()` — full JSON
schemas; both registered in `tools::compact::full_schema` (so
`memory_status view='tools'` unfolds them).

Pure helpers other agents can reuse:

- `tools::forgotten_lesson::detect_lesson(&KnowledgeNode, &HashSet<String> failure_anchors, DateTime<Utc> failure_at) -> Option<ForgottenLesson>` — one shared detection rule (lesson tag / exact past-tense fix marker + >=1 exact shared anchor + FSRS R < 0.5 at failure time).
- `tools::forgotten_lesson::{lesson_tagged, has_fix_marker}`, `pub const FORGOTTEN_THRESHOLD: f64 = 0.5`.
- `tools::selftest` internals are private; the calibration flow is only the `execute` tool.

### CLI subcommands (`crates/vestige-mcp/src/bin/cli.rs`)

- `vestige selftest` — prints the same payload as the MCP tool.
- `vestige forgotten-lesson <FAILURE_ID> [--scope S] [--json]`.

## Calibration definition (selftest)

`gap_calibration = true` iff the 6th round (cause and failure share NO anchor)
returns `triggered=true` with an EMPTY causes list AND a non-null `gap` whose
`missing_entities` contains the withheld env-shaped anchor `planted_cause_6`.
Rounds 1-5 each isolate one planted cause (5d backdated, own scope) with a
failure sharing exactly one anchor; hit@1 = cause ranked first, hit@3 = top 3.
The live store is only read (`Storage::backup_to` / `VACUUM INTO` snapshot to a
tempdir); all mutation happens on the copy; the temp store is deleted.

## Non-append-only edits (forced by existing parity guards)

`crates/vestige-mcp/src/server.rs` tests only: advertised-tool count 16 -> 18
(two places), read-only hint list now
`["forgotten_lesson", "memory_status", "selftest", "session_start"]`. Catalog
and dispatch arms themselves are append-only additions.

## Test status

- `cargo test -p vestige-mcp --lib selftest` → 2/2 ok.
- `cargo test -p vestige-mcp --lib forgotten` → 5/5 ok.
- Full `--lib` suite: 763 passed, 2 failed — BOTH failures
  (`server::tests::test_recall_lookup_matches_search_shape`,
  `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`)
  reproduce on the clean base commit (verified via stash) and are pre-existing.
- `cargo clippy -p vestige-mcp --lib --bins`: clean.
# SCOPE-HANDOFF — w3d: selftest + forgotten_lesson

Branch `build/w3d-selftest-forgotten` (base `main` @ `5fbe1df`). All additions
are append-only except three marked test-constant updates forced by parity
guards; nothing outside the listed files was touched.

## New exposed APIs

### `vestige-mcp` MCP tools

| Tool | Module | Signature |
|---|---|---|
| `selftest` | `crates/vestige-mcp/src/tools/selftest.rs` | `pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String>` — no arguments |
| `forgotten_lesson` | `crates/vestige-mcp/src/tools/forgotten_lesson.rs` | same shape; args `{failure_id (required), scope?, scan_limit?}` |

`tools::selftest::schema()`, `tools::forgotten_lesson::schema()` — full JSON
schemas; both registered in `tools::compact::full_schema` (so
`memory_status view='tools'` unfolds them).

Pure helpers other agents can reuse:

- `tools::forgotten_lesson::detect_lesson(&KnowledgeNode, &HashSet<String> failure_anchors, DateTime<Utc> failure_at) -> Option<ForgottenLesson>` — one shared detection rule (lesson tag / exact past-tense fix marker + >=1 exact shared anchor + FSRS R < 0.5 at failure time).
- `tools::forgotten_lesson::{lesson_tagged, has_fix_marker}`, `pub const FORGOTTEN_THRESHOLD: f64 = 0.5`.
- `tools::selftest` internals are private; the calibration flow is only the `execute` tool.

### CLI subcommands (`crates/vestige-mcp/src/bin/cli.rs`)

- `vestige selftest` — prints the same payload as the MCP tool.
- `vestige forgotten-lesson <FAILURE_ID> [--scope S] [--json]`.

## Calibration definition (selftest)

`gap_calibration = true` iff the 6th round (cause and failure share NO anchor)
returns `triggered=true` with an EMPTY causes list AND a non-null `gap` whose
`missing_entities` contains the withheld env-shaped anchor `planted_cause_6`.
Rounds 1-5 each isolate one planted cause (5d backdated, own scope) with a
failure sharing exactly one anchor; hit@1 = cause ranked first, hit@3 = top 3.
The live store is only read (`Storage::backup_to` / `VACUUM INTO` snapshot to a
tempdir); all mutation happens on the copy; the temp store is deleted.

## Non-append-only edits (forced by existing parity guards)

`crates/vestige-mcp/src/server.rs` tests only: advertised-tool count 16 -> 18
(two places), read-only hint list now
`["forgotten_lesson", "memory_status", "selftest", "session_start"]`. Catalog
and dispatch arms themselves are append-only additions.

## Test status

- `cargo test -p vestige-mcp --lib selftest` → 2/2 ok.
- `cargo test -p vestige-mcp --lib forgotten` → 5/5 ok.
- Full `--lib` suite: 763 passed, 2 failed — BOTH failures
  (`server::tests::test_recall_lookup_matches_search_shape`,
  `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`)
  reproduce on the clean base commit (verified via stash) and are pre-existing.
- `cargo clippy -p vestige-mcp --lib --bins`: clean.

# SCOPE-HANDOFF — build/w3c-session-sync

Base: main @ 5fbe1df. Scope held: `crates/vestige-mcp/src/tools/session_context.rs`,
`crates/vestige-mcp/src/tools/source_sync.rs`, new
`crates/vestige-core/src/storage/session_queries.rs` (+ the one-line module
wiring and re-export in `crates/vestige-core/src/storage/mod.rs`, required to
compile the new file). Nothing else touched.

## What was built

### 1. `session_queries.rs` (new, on SqliteMemoryStore)

- `open_failures_touching(changed_files: &[String]) -> Vec<OpenFailureTouching{id, content_preview, anchor}>`
  — failure-like nodes (`retroactive_backfill::looks_like_failure`, reuse) that
  are open (not superseded/suppressed, inside validity window) and whose
  recorded files intersect the changed set by EXACT path equality:
  `code_memory_anchors.file_path` first (anchor rendered `path` or
  `path:symbol`), then the `files:` line of a git-commit record (the
  ` (+N more)` suffix stripped). Prefix/suffix near-misses do not hit. Bounded
  chunked node scan (pages of 500, hard cap 10k nodes).
- `last_session_failed_calls(run_id: Option<&str>) -> Vec<FailedToolCall{run_id, tool, at, error_excerpt}>`
  — `agent_traces` rows of the given run, or the latest run by
  `agent_runs.last_at` when `None`, whose serialized `mcp.call` payload
  carries `success: false`. Last 20, chronological. Rows without a `success`
  field are not failed calls.
- `append_mcp_call_outcome(run_id, tool, success, error, at_ms)` — writes an
  `mcp.call`-shaped trace row extended with `success`/`error` (argsHash is the
  opaque constant `"outcome"`). Payload keeps the exact event shape so
  `get_trace` replay still parses it. NOTE: no production caller yet — the
  trace recorder (`trace_recorder.rs`) is out of scope and still writes
  plain `mcp.call` events, so the failed-calls section stays empty until the
  recorder adopts this method. That adoption is the natural follow-up.
- `closed_issue_nodes(source_system, scope)` / `git_commit_nodes(limit)` —
  exact-tag lookups (`state:closed`, `git-commit`) backing the closed_by link.

### 2. `session_context.rs` (session_start) — purely additive

- New optional arg `changed_files: Vec<String>` (schema + serde, camelCase
  alias). Absent = open-failures section skipped. Present but no exact match =
  no section (silence, never an empty header).
- Section `**Open failures touching changed files:**` — up to 8 lines
  `- [id] preview (anchor)`, budget-accounted like every other section.
- Section `**Last session failed calls (run):**` — automatic, latest run, up
  to 8 lines `- tool: error_excerpt`.
- Both sections degrade silently on query error; all existing sections and
  the budget-trim loop are untouched.

### 3. `source_sync.rs` — `closed_by` chain linking (github, local-only)

- Verified what the github connector fetches today: issues + comments only —
  no closing-PR sha, no timeline/cross-referenced events. Per the build
  instruction, remote fetch is SKIPPED; linking is local-only and
  deterministic.
- `commit_closes_issue(content, issue_number)` — GitHub closing keywords
  (close/closes/closed/fix/fixes/fixed/resolve/resolves/resolved), whole-word,
  case-insensitive, and `#<number>` with exact digit boundaries (`#420` never
  satisfies #42; `abc#42`/`##42` rejected), keyword + reference on the SAME
  line.
- `link_closed_by_from_local_commits(storage, scope)` — for each live closed
  github issue in scope, writes `memory_connections` edges
  `link_type="closed_by"`, `source_id` = issue node, `target_id` = commit
  node, via the existing `save_connection` (INSERT OR REPLACE ⇒ idempotent
  re-syncs). Runs after every successful github `run_sync`; count surfaced as
  `closedByLinks` in the tool result and appended to the summary when > 0.
  Redmine is untouched.

## Tests (all green)

- `cargo test -p vestige-core --lib session_queries` — 6/6.
- `cargo test -p vestige-core --lib storage::` — 372/372.
- `cargo test -p vestige-mcp --lib session` — 27/27 (4 new:
  open-failures populate from anchors + `files:` lines, negative no-noise
  near-miss/absent-arg, failed-calls latest-run, silent without outcome rows).
- `cargo test -p vestige-mcp --lib source_sync` — 4/4 (keyword forms,
  near-miss rejects incl. `discloses`/`#420`/cross-line, edge appears +
  idempotent + open-issue/commit negatives, silent without candidates).
- Full `cargo test -p vestige-mcp --lib`: 764 passed, 2 failed —
  `server::tests::test_recall_lookup_matches_search_shape` and
  `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`
  both REPRODUCE ON PRISTINE main @ 5fbe1df (verified via stash); pre-existing,
  not from this branch.

## Follow-ups (out of scope here)

- Recorder adoption of `append_mcp_call_outcome` in `trace_recorder.rs` /
  `server.rs` dispatch so the failed-calls section has a production producer.
- Remote closing-PR lookup if ever wanted: needs a timeline-events fetch in
  the github connector (new API surface, deliberately not added).
# SCOPE-HANDOFF — build/w3c-session-sync

Base: main @ 5fbe1df. Scope held: `crates/vestige-mcp/src/tools/session_context.rs`,
`crates/vestige-mcp/src/tools/source_sync.rs`, new
`crates/vestige-core/src/storage/session_queries.rs` (+ the one-line module
wiring and re-export in `crates/vestige-core/src/storage/mod.rs`, required to
compile the new file). Nothing else touched.

## What was built

### 1. `session_queries.rs` (new, on SqliteMemoryStore)

- `open_failures_touching(changed_files: &[String]) -> Vec<OpenFailureTouching{id, content_preview, anchor}>`
  — failure-like nodes (`retroactive_backfill::looks_like_failure`, reuse) that
  are open (not superseded/suppressed, inside validity window) and whose
  recorded files intersect the changed set by EXACT path equality:
  `code_memory_anchors.file_path` first (anchor rendered `path` or
  `path:symbol`), then the `files:` line of a git-commit record (the
  ` (+N more)` suffix stripped). Prefix/suffix near-misses do not hit. Bounded
  chunked node scan (pages of 500, hard cap 10k nodes).
- `last_session_failed_calls(run_id: Option<&str>) -> Vec<FailedToolCall{run_id, tool, at, error_excerpt}>`
  — `agent_traces` rows of the given run, or the latest run by
  `agent_runs.last_at` when `None`, whose serialized `mcp.call` payload
  carries `success: false`. Last 20, chronological. Rows without a `success`
  field are not failed calls.
- `append_mcp_call_outcome(run_id, tool, success, error, at_ms)` — writes an
  `mcp.call`-shaped trace row extended with `success`/`error` (argsHash is the
  opaque constant `"outcome"`). Payload keeps the exact event shape so
  `get_trace` replay still parses it. NOTE: no production caller yet — the
  trace recorder (`trace_recorder.rs`) is out of scope and still writes
  plain `mcp.call` events, so the failed-calls section stays empty until the
  recorder adopts this method. That adoption is the natural follow-up.
- `closed_issue_nodes(source_system, scope)` / `git_commit_nodes(limit)` —
  exact-tag lookups (`state:closed`, `git-commit`) backing the closed_by link.

### 2. `session_context.rs` (session_start) — purely additive

- New optional arg `changed_files: Vec<String>` (schema + serde, camelCase
  alias). Absent = open-failures section skipped. Present but no exact match =
  no section (silence, never an empty header).
- Section `**Open failures touching changed files:**` — up to 8 lines
  `- [id] preview (anchor)`, budget-accounted like every other section.
- Section `**Last session failed calls (run):**` — automatic, latest run, up
  to 8 lines `- tool: error_excerpt`.
- Both sections degrade silently on query error; all existing sections and
  the budget-trim loop are untouched.

### 3. `source_sync.rs` — `closed_by` chain linking (github, local-only)

- Verified what the github connector fetches today: issues + comments only —
  no closing-PR sha, no timeline/cross-referenced events. Per the build
  instruction, remote fetch is SKIPPED; linking is local-only and
  deterministic.
- `commit_closes_issue(content, issue_number)` — GitHub closing keywords
  (close/closes/closed/fix/fixes/fixed/resolve/resolves/resolved), whole-word,
  case-insensitive, and `#<number>` with exact digit boundaries (`#420` never
  satisfies #42; `abc#42`/`##42` rejected), keyword + reference on the SAME
  line.
- `link_closed_by_from_local_commits(storage, scope)` — for each live closed
  github issue in scope, writes `memory_connections` edges
  `link_type="closed_by"`, `source_id` = issue node, `target_id` = commit
  node, via the existing `save_connection` (INSERT OR REPLACE ⇒ idempotent
  re-syncs). Runs after every successful github `run_sync`; count surfaced as
  `closedByLinks` in the tool result and appended to the summary when > 0.
  Redmine is untouched.

## Tests (all green)

- `cargo test -p vestige-core --lib session_queries` — 6/6.
- `cargo test -p vestige-core --lib storage::` — 372/372.
- `cargo test -p vestige-mcp --lib session` — 27/27 (4 new:
  open-failures populate from anchors + `files:` lines, negative no-noise
  near-miss/absent-arg, failed-calls latest-run, silent without outcome rows).
- `cargo test -p vestige-mcp --lib source_sync` — 4/4 (keyword forms,
  near-miss rejects incl. `discloses`/`#420`/cross-line, edge appears +
  idempotent + open-issue/commit negatives, silent without candidates).
- Full `cargo test -p vestige-mcp --lib`: 764 passed, 2 failed —
  `server::tests::test_recall_lookup_matches_search_shape` and
  `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`
  both REPRODUCE ON PRISTINE main @ 5fbe1df (verified via stash); pre-existing,
  not from this branch.

## Follow-ups (out of scope here)

- Recorder adoption of `append_mcp_call_outcome` in `trace_recorder.rs` /
  `server.rs` dispatch so the failed-calls section has a production producer.
- Remote closing-PR lookup if ever wanted: needs a timeline-events fetch in
  the github connector (new API surface, deliberately not added).

# Scope Handoff — build/w3e-receipts-status

Base: `main @ 5fbe1df`. Branch: `build/w3e-receipts-status`. Not pushed.

## What was built

### 1. Walk receipts (core storage)

New file `crates/vestige-core/src/storage/walk_receipts.rs` (migration **V39**, non-destructive, `walk_receipts` table: `receipt_id PK, digest UNIQUE, canonical_json, engine_version, created_at`).

- `canonical_walk_json(params: &Value) -> Result<String>` — canonicalizes via
  `serde_json_canonicalizer` (RFC 8785 / JCS), the same helper the receipt DSSE
  chain already uses. Sorted keys, no insignificant whitespace, stable number
  formatting; rejects non-object envelopes.
- `Storage::save_walk_receipt(canonical_json: &str, params: &Value) -> Result<WalkReceiptHandle>`
  — re-canonicalizes `params` internally and **rejects** a `canonical_json` that
  does not match (digest always computable from stored bytes). Digest =
  `blake3(canonical bytes)` hex (dep already in core Cargo.toml). `receipt_id =
  "wr_" + digest[..24]` → saving the same envelope is idempotent
  (`reused_existing: true`, single row). `engine_version` =
  `env!("CARGO_PKG_VERSION")` at save time.
- `Storage::get_walk_receipt(receipt_id) -> Result<Option<StoredWalkReceipt>>`.
- `Storage::coverage_snapshot() -> Result<CoverageSnapshot>` — exact SQL
  aggregates for the coverage view (see §3).

### 2. Replay placement (scope adjustment, per task instruction)

`tools::backfill::execute` lives in the MCP layer only, so **replay lives in the
MCP layer** (`crates/vestige-mcp/src/tools/receipt.rs`), not core. The core file
contributes constants only: `WALK_RECEIPT_SCHEMA_V1`,
`WALK_RECEIPT_CLAIM_BOUNDARY` (`trace/receipt.rs`, re-exported at crate root).

`receipt` tool surface (schema `additionalProperties: false`, `deny_unknown_fields`):

- `{action: "save_walk", params: {...}}` → `{receiptId, digest, canonicalParams
  (byte-stable echo), engineVersion, reusedExisting, claimBoundary}`. No clock
  fields → byte-stable output.
- `{action: "replay", receipt_id: "wr_…" [, remove_edge]}` → dispatches on walk
  receipts (ids in `walk_receipts`); non-`wr_` ids keep the counterfactual
  capsule replay untouched. `remove_edge` (memory id or `source->target`)
  filters that candidate from the pool before the ablated run. Replay is always
  a **pure preview**: recorded `promote/git_repo/worked_in/broke_in/why_not`
  are reported as ignored, never applied (no edges written, no reinforcement).
  Runs the reach twice (baseline + ablated) with a candidate assembly mirroring
  `backfill::build_candidates` (supersession-following, commit-tag flag; no git
  version-range re-application), asserts both runs evaluated the same failure,
  and reports `verdictDelta {filterApplied, verdictChanged, baseline, ablated}`
  (ids + 2-dp scores only → deterministic serialization; failure entities are
  sorted because `extract_entities` iterates a hash set).
- Validation: `withheld_slots` xor `remove_edge`; `params` only for
  `save_walk`; `receipt_id` not for `save_walk`.

### 3. memory_status view="coverage"

`{anchorCoveragePct (COUNT(DISTINCT node_id) code_memory_anchors / COUNT(*)
knowledge_nodes, 2-dp, 0.0 on empty store), anchoredNodes, totalNodes,
edgeCountsByType (memory_connections GROUP BY link_type ORDER BY link_type),
indexFreshness {newestGitCommitRecord + AgeDays (MAX(created_at) over
json_each(tags)='git-commit' nodes), newestAgentTraceAt + AgeHours (MAX(at)
agent_traces, millis), stalenessNote (deterministic thresholds: commit >30d,
trace >72h)}}` + claim boundary.

## Canonicalization spec

RFC 8785 (JCS) via `serde_json_canonicalizer::to_vec`; digest = blake3 of the
UTF-8 canonical bytes, hex; id = `wr_` + first 24 hex chars. Same value → same
bytes → same digest → same id, regardless of input key order/whitespace.

## Test status

- `cargo test -p vestige-core --lib walk_receipt` — 3/3 pass
  (canonicalization stability incl. key-order shuffle + idempotent save +
  mismatch rejection; coverage math on seeded store).
- `cargo test -p vestige-mcp --lib tools::receipt` — 9/9 pass (save twice →
  same digest/id; replay byte-identical on unchanged store; remove_edge
  verdict delta incl. `src->tgt` form and no-op edge; promote never applied).
- `cargo test -p vestige-mcp --lib tools::memory_status` — 5/5 pass
  (incl. coverage math: 1/5 anchored = 20.0%, grouped/ordered edge counts).
- `cargo test -p vestige-core --lib migrations` — 36/36 pass (V39 registered).
- `cargo clippy -p vestige-core -p vestige-mcp --lib` — clean.

### Pre-existing failures on the CLEAN base (not touched, out of scope)

Verified by stashing this diff and re-running:

- `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage` —
  the seeded distractor ("500 Internal Server Error…") outranks the env-var
  cause under the current word-tier IDF scorer. The walk replay mirrors the
  live tool's ranking faithfully, which is why the replay tests are
  rank-agnostic (ablate whatever is ranked top).
- `server::tests::test_recall_lookup_matches_search_shape` — asserts parity
  with the removed `search` tool.

## Files changed

`crates/vestige-core/src/storage/walk_receipts.rs` (new),
`storage/migrations.rs` (V39), `storage/mod.rs` (mod + re-exports),
`trace/receipt.rs` + `trace/mod.rs` + `lib.rs` (walk constants + re-exports),
`crates/vestige-mcp/src/tools/receipt.rs`, `tools/memory_status.rs`,
`server.rs` (receipt tool description string only — it enumerates actions).

# Scope handoff — build/w2a-hunk-lines

Base: main @ 5fbe1df. Scope: `crates/vestige-core/src/advanced/git_records.rs` + tests only.

## Out-of-scope file touched (minimal, compile-only)

`crates/vestige-mcp/src/tools/backfill.rs` — test helper `commit_record_content`
(line ~861) constructs `GitCommit` literally. Added the three new fields with
empty values so the crate compiles:

```rust
hunks: vec![],
extra_hunks: 0,
imports: vec![],
```

No logic changed; `tools::backfill` tests pass unchanged.

## What changed in git_records.rs

- `HunkSpan { file, start, len, symbol }` — new-side `@@` spans, `MAX_HUNKS = 200`,
  overflow counted in `GitCommit::extra_hunks`.
- `GitCommit::imports: Vec<(file_in_commit, target_path, resolved)>` —
  `MAX_IMPORTS = 40`, deduped, sorted by (file, target).
- `record_content` new lines:
  - `hunks: file:start+len` comma-joined, item list capped at 400 chars
    (`MAX_HUNK_LINE`), followed by ` (+N more)` when spans were dropped or
    overflowed; emitted only when spans exist.
  - `imports: file->target` comma-joined; unresolved edges render as
    `file->?target` (target kept as written, never guessed).
- Import capture: `use a::b::Item` / `import x.y.z` (multi-segment only, so
  `import os` noise is skipped) / `from x.y import z` / `#include "p"` and
  `#include <p>`. Resolution is exact module-segment matching against the same
  commit's file list and their module dirs (`crate` roots at `src/`; Rust
  `.rs`/`mod.rs`, Python `.py`/`__init__.py`, include verbatim). File hits beat
  module-dir hits; longest path first. No fuzzy matching.
- Hunks/imports for files beyond `MAX_FILES` are dropped (same
  non-attribution rule as symbols), not counted as span overflow.

## Pre-existing test failure (not from this branch)

`tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`
(crates/vestige-mcp) fails on pristine base @ 5fbe1df — verified via stash
with these changes removed: the billing-service distractor outranks the
env-var cause. The test never touches `GitCommit`/`record_content`; the two
commits below base reworked exactly that ranking code. All other
`tools::backfill` tests pass.

## Incident note: shared stash stack race

This repo's worktrees share one `git stash` stack. During verification, a
`git stash pop` in this worktree raced sibling worktrees and pulled in two
foreign stashes. Both were re-stashed onto the shared stack as labeled
entries — find them with `git stash list | grep RESCUE`:

- `RESCUE-A(65a5ea0)` — storage/session_context/source_sync changes, owner:
  build/w2c-resolver worktree.
- `RESCUE-B(ecfcddfb)` — dashboard cinema changes (MemoryCinema, sandbox,
  storm, temporal-emissive.ts), owner: feat/dashboard-live-max worktree.

Original dropped hashes are recoverable from the stash reflog if needed.
Use hash-based `git stash apply <hash>` (resolved by message) in this repo,
never positional `stash@{n}`.
# Scope handoff — build/w2a-hunk-lines

Base: main @ 5fbe1df. Scope: `crates/vestige-core/src/advanced/git_records.rs` + tests only.

## Out-of-scope file touched (minimal, compile-only)

`crates/vestige-mcp/src/tools/backfill.rs` — test helper `commit_record_content`
(line ~861) constructs `GitCommit` literally. Added the three new fields with
empty values so the crate compiles:

```rust
hunks: vec![],
extra_hunks: 0,
imports: vec![],
```

No logic changed; `tools::backfill` tests pass unchanged.

## What changed in git_records.rs

- `HunkSpan { file, start, len, symbol }` — new-side `@@` spans, `MAX_HUNKS = 200`,
  overflow counted in `GitCommit::extra_hunks`.
- `GitCommit::imports: Vec<(file_in_commit, target_path, resolved)>` —
  `MAX_IMPORTS = 40`, deduped, sorted by (file, target).
- `record_content` new lines:
  - `hunks: file:start+len` comma-joined, item list capped at 400 chars
    (`MAX_HUNK_LINE`), followed by ` (+N more)` when spans were dropped or
    overflowed; emitted only when spans exist.
  - `imports: file->target` comma-joined; unresolved edges render as
    `file->?target` (target kept as written, never guessed).
- Import capture: `use a::b::Item` / `import x.y.z` (multi-segment only, so
  `import os` noise is skipped) / `from x.y import z` / `#include "p"` and
  `#include <p>`. Resolution is exact module-segment matching against the same
  commit's file list and their module dirs (`crate` roots at `src/`; Rust
  `.rs`/`mod.rs`, Python `.py`/`__init__.py`, include verbatim). File hits beat
  module-dir hits; longest path first. No fuzzy matching.
- Hunks/imports for files beyond `MAX_FILES` are dropped (same
  non-attribution rule as symbols), not counted as span overflow.

## Pre-existing test failure (not from this branch)

`tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`
(crates/vestige-mcp) fails on pristine base @ 5fbe1df — verified via stash
with these changes removed: the billing-service distractor outranks the
env-var cause. The test never touches `GitCommit`/`record_content`; the two
commits below base reworked exactly that ranking code. All other
`tools::backfill` tests pass.

## Incident note: shared stash stack race

This repo's worktrees share one `git stash` stack. During verification, a
`git stash pop` in this worktree raced sibling worktrees and pulled in two
foreign stashes. Both were re-stashed onto the shared stack as labeled
entries — find them with `git stash list | grep RESCUE`:

- `RESCUE-A(65a5ea0)` — storage/session_context/source_sync changes, owner:
  build/w2c-resolver worktree.
- `RESCUE-B(ecfcddfb)` — dashboard cinema changes (MemoryCinema, sandbox,
  storm, temporal-emissive.ts), owner: feat/dashboard-live-max worktree.

Original dropped hashes are recoverable from the stash reflog if needed.
Use hash-based `git stash apply <hash>` (resolved by message) in this repo,
never positional `stash@{n}`.

# SCOPE-HANDOFF — Handle-Based Recall Resolver (build/w2c-resolver)

Branch: `build/w2c-resolver`, base = `main` @ `5fbe1df`. Not pushed.

## Files (the entire scope)

- `crates/vestige-core/src/storage/resolver.rs` — NEW. The resolver.
- `crates/vestige-core/src/storage/mod.rs` — wiring only (`mod resolver;` + re-exports).
- `crates/vestige-mcp/src/tools/recall.rs` — thin handle-mode extension at entry.
- `SCOPE-HANDOFF.md` — this file.

## Core API (vestige-core)

```rust
// crates/vestige-core/src/storage/resolver.rs
impl SqliteMemoryStore {
    pub fn resolve_handle(&self, query: &str) -> HandleResolution;
}

pub struct HandleResolution {
    pub kind: HandleKind,                     // what the query resolved as
    pub ids: Vec<String>,                     // node/trace ids resolved
    pub exact: bool,                          // true iff equality (not prefix)
    pub candidates: Vec<(String, HandleKind)>,// when prefix-ambiguous, capped at 20
    pub handle_required: Option<String>,      // guidance when nothing matched
}

pub enum HandleKind { Memory, Commit, File, Symbol, Test, Run, ToolCall, Tag, Unknown }

pub const MAX_CANDIDATES: usize = 20;
pub const HANDLE_REQUIRED_DETAIL: &str =
    "recall is handle-based: pass a memory id, commit sha, file, symbol, test, run, or tool-call id";
```

Import paths: `vestige_core::storage::{HandleKind, HandleResolution, MAX_CANDIDATES, HANDLE_REQUIRED_DETAIL}`. `resolve_handle` is a method on `SqliteMemoryStore` (`vestige_core::Storage`). Read-only: no FSRS/edge/graph writes.

## Resolution rules (EXACT or PREFIX only — no fuzzy, no lexical ranking, no FTS)

| # | Kind | Match | Prefix? | Source |
|---|------|-------|---------|--------|
| 1 | Memory | query parses as uuid, equals `knowledge_nodes.id` | no | `knowledge_nodes` |
| 2 | Commit | `commit <sha>` on content line 1 of `git-commit`-tagged nodes; case-insensitive | **yes, >= 7 hex chars**; 4–6 hex chars w/ a digit = ambiguity error; unique prefix resolves with `exact=false`; multiple = capped candidates | `knowledge_nodes` |
| 3 | File / Test | whole-token boundary match over content + tags (coarse LIKE prefilter, exact Rust verify). File if path-shaped (`/` or `.`); Test if test-shaped (`tests/`, `_test`, `test_` prefix) | no | `knowledge_nodes` |
| 4 | Symbol | query normalized camel→snake+lowercase (same `extract_entities` normalization); matched against Code-tier extracted entities only — Word/Path/Version tiers excluded so `pyvenv` can never prefix-match `pyvenv.cfg` | **yes (unique prefix)** | `knowledge_nodes` |
| 5 | Run | equals `agent_runs.run_id` or `agent_traces.run_id` | no | V18 tables |
| 6 | ToolCall | equals `agent_traces.id` (trace event id) | no | V18 tables |
| 7 | Tag | exact string equality against parsed tags JSON (case-sensitive) | no | `knowledge_nodes` |

Unresolved / empty → `kind=Unknown`, `handle_required=Some(HANDLE_REQUIRED_DETAIL)`.

## MCP surface (recall)

New schema property `handle` (string). Semantics in `recall::execute`:

- args WITHOUT a `handle` key → legacy mode dispatch, byte-identical behavior
  (hot-path invariant preserved; existing tests pass unchanged). **This gate
  (`handle_flow` returning None) is the single flip point for making recall
  handle-only by default.**
- args WITH `handle: "<handle>"` → resolve, then:
  - resolved: `{"handle", "kind", "exact", "nodes":[{id,type,content,tags}], "neighbors":[{from,to,link_type,strength,direction,node}]}` — neighbors are one hop over `memory_connections` (both directions, strength-desc, deduped, capped at 20 edges).
  - ambiguous: `{"error":"ambiguous", "detail", "handle", "kind", "candidates":[{id,kind}]}`.
  - nothing: `{"error":"handle_required", "detail", "candidates":[]}` (`detail` carries the resolver's specific too-short-sha message when applicable).
- args WITH `handle: ""` (or non-string) → free-text branch: same
  `handle_required` payload, with candidates mined by running the resolver on
  the whole `query` text and its first 8 identifier-shaped tokens (exact/prefix
  results only, deduped, capped at 20).

## Integrator notes

- Scope-agnostic by design: handles are globally unique, so resolution ignores
  `scope`. Filter client-side if a scoped surface is needed.
- Symbol prefix scans all nodes and runs `extract_entities` per row (no entity
  index yet). Correctness-first; add an entity index before high-QPS use.
- Commit resolution reads shas only from `git-commit`-tagged records in the
  canonical `git_records::record_content` shape.
- Known deviation from the letter of the design: File and Test share one
  boundary-exact step (file before symbol per the order); a test-shaped query
  resolves as kind `Test` even before the symbol step would have matched.
- English hex words without digits (`face`, `added`) are not treated as
  too-short shas; 4–6 char hex WITH a digit is the ambiguity error.

## Tests

- `cargo test -p vestige-core --lib resolver` → 8 passed (uuid exact/prefix
  paths, full/prefix/ambiguous/too-short shas, file exact + NO-fuzzy +
  NO-prefix receipts (`pyvenv` and `pyvenv.c` both fail against `pyvenv.cfg`),
  symbol camel/snake/env normalization + prefix, test names, tags exact,
  run/toolcall exact, free prose unresolved).
- `cargo test -p vestige-mcp --lib tools::recall` → 7 passed (schema, legacy
  default, contradictions, uuid + one-hop neighbors ordering, sha exact +
  ambiguous, handle_required with/without candidates, no-handle legacy path).
- `cargo test -p vestige-core --lib "storage::sqlite"` → 220 passed (no
  regressions in the neighboring suite).
- Pre-existing failure, NOT from this branch (fails on clean `main` @
  `5fbe1df` too): `vestige-mcp server::tests::test_recall_lookup_matches_search_shape`
  — it calls the removed `search` tool and compares against its error text.
  `server.rs` is outside this task's file scope.
# SCOPE-HANDOFF — Handle-Based Recall Resolver (build/w2c-resolver)

Branch: `build/w2c-resolver`, base = `main` @ `5fbe1df`. Not pushed.

## Files (the entire scope)

- `crates/vestige-core/src/storage/resolver.rs` — NEW. The resolver.
- `crates/vestige-core/src/storage/mod.rs` — wiring only (`mod resolver;` + re-exports).
- `crates/vestige-mcp/src/tools/recall.rs` — thin handle-mode extension at entry.
- `SCOPE-HANDOFF.md` — this file.

## Core API (vestige-core)

```rust
// crates/vestige-core/src/storage/resolver.rs
impl SqliteMemoryStore {
    pub fn resolve_handle(&self, query: &str) -> HandleResolution;
}

pub struct HandleResolution {
    pub kind: HandleKind,                     // what the query resolved as
    pub ids: Vec<String>,                     // node/trace ids resolved
    pub exact: bool,                          // true iff equality (not prefix)
    pub candidates: Vec<(String, HandleKind)>,// when prefix-ambiguous, capped at 20
    pub handle_required: Option<String>,      // guidance when nothing matched
}

pub enum HandleKind { Memory, Commit, File, Symbol, Test, Run, ToolCall, Tag, Unknown }

pub const MAX_CANDIDATES: usize = 20;
pub const HANDLE_REQUIRED_DETAIL: &str =
    "recall is handle-based: pass a memory id, commit sha, file, symbol, test, run, or tool-call id";
```

Import paths: `vestige_core::storage::{HandleKind, HandleResolution, MAX_CANDIDATES, HANDLE_REQUIRED_DETAIL}`. `resolve_handle` is a method on `SqliteMemoryStore` (`vestige_core::Storage`). Read-only: no FSRS/edge/graph writes.

## Resolution rules (EXACT or PREFIX only — no fuzzy, no lexical ranking, no FTS)

| # | Kind | Match | Prefix? | Source |
|---|------|-------|---------|--------|
| 1 | Memory | query parses as uuid, equals `knowledge_nodes.id` | no | `knowledge_nodes` |
| 2 | Commit | `commit <sha>` on content line 1 of `git-commit`-tagged nodes; case-insensitive | **yes, >= 7 hex chars**; 4–6 hex chars w/ a digit = ambiguity error; unique prefix resolves with `exact=false`; multiple = capped candidates | `knowledge_nodes` |
| 3 | File / Test | whole-token boundary match over content + tags (coarse LIKE prefilter, exact Rust verify). File if path-shaped (`/` or `.`); Test if test-shaped (`tests/`, `_test`, `test_` prefix) | no | `knowledge_nodes` |
| 4 | Symbol | query normalized camel→snake+lowercase (same `extract_entities` normalization); matched against Code-tier extracted entities only — Word/Path/Version tiers excluded so `pyvenv` can never prefix-match `pyvenv.cfg` | **yes (unique prefix)** | `knowledge_nodes` |
| 5 | Run | equals `agent_runs.run_id` or `agent_traces.run_id` | no | V18 tables |
| 6 | ToolCall | equals `agent_traces.id` (trace event id) | no | V18 tables |
| 7 | Tag | exact string equality against parsed tags JSON (case-sensitive) | no | `knowledge_nodes` |

Unresolved / empty → `kind=Unknown`, `handle_required=Some(HANDLE_REQUIRED_DETAIL)`.

## MCP surface (recall)

New schema property `handle` (string). Semantics in `recall::execute`:

- args WITHOUT a `handle` key → legacy mode dispatch, byte-identical behavior
  (hot-path invariant preserved; existing tests pass unchanged). **This gate
  (`handle_flow` returning None) is the single flip point for making recall
  handle-only by default.**
- args WITH `handle: "<handle>"` → resolve, then:
  - resolved: `{"handle", "kind", "exact", "nodes":[{id,type,content,tags}], "neighbors":[{from,to,link_type,strength,direction,node}]}` — neighbors are one hop over `memory_connections` (both directions, strength-desc, deduped, capped at 20 edges).
  - ambiguous: `{"error":"ambiguous", "detail", "handle", "kind", "candidates":[{id,kind}]}`.
  - nothing: `{"error":"handle_required", "detail", "candidates":[]}` (`detail` carries the resolver's specific too-short-sha message when applicable).
- args WITH `handle: ""` (or non-string) → free-text branch: same
  `handle_required` payload, with candidates mined by running the resolver on
  the whole `query` text and its first 8 identifier-shaped tokens (exact/prefix
  results only, deduped, capped at 20).

## Integrator notes

- Scope-agnostic by design: handles are globally unique, so resolution ignores
  `scope`. Filter client-side if a scoped surface is needed.
- Symbol prefix scans all nodes and runs `extract_entities` per row (no entity
  index yet). Correctness-first; add an entity index before high-QPS use.
- Commit resolution reads shas only from `git-commit`-tagged records in the
  canonical `git_records::record_content` shape.
- Known deviation from the letter of the design: File and Test share one
  boundary-exact step (file before symbol per the order); a test-shaped query
  resolves as kind `Test` even before the symbol step would have matched.
- English hex words without digits (`face`, `added`) are not treated as
  too-short shas; 4–6 char hex WITH a digit is the ambiguity error.

## Tests

- `cargo test -p vestige-core --lib resolver` → 8 passed (uuid exact/prefix
  paths, full/prefix/ambiguous/too-short shas, file exact + NO-fuzzy +
  NO-prefix receipts (`pyvenv` and `pyvenv.c` both fail against `pyvenv.cfg`),
  symbol camel/snake/env normalization + prefix, test names, tags exact,
  run/toolcall exact, free prose unresolved).
- `cargo test -p vestige-mcp --lib tools::recall` → 7 passed (schema, legacy
  default, contradictions, uuid + one-hop neighbors ordering, sha exact +
  ambiguous, handle_required with/without candidates, no-handle legacy path).
- `cargo test -p vestige-core --lib "storage::sqlite"` → 220 passed (no
  regressions in the neighboring suite).
- Pre-existing failure, NOT from this branch (fails on clean `main` @
  `5fbe1df` too): `vestige-mcp server::tests::test_recall_lookup_matches_search_shape`
  — it calls the removed `search` tool and compares against its error text.
  `server.rs` is outside this task's file scope.

# W1E Test Cleanup — Scope Handoff

Branch `build/w1e-tests-cleanup` (base: main @ 5fbe1df). Test hygiene ONLY, per
owner order. Source modules still present in this tree; parallel agents
(w1a-core-kill, w1b-search-collapse, w1c-ingest-flip, w1d-mcp-surface) own the
source deletions.

## Deleted (clearly vector/embedding-only)

- `tests/e2e/src/mocks/mock_embedding.rs` (whole file; no test used
  `MockEmbeddingService` directly — only re-exports in `tests/e2e/src/lib.rs`
  and `tests/e2e/src/mocks/mod.rs`, both cleaned).
- `tests/e2e/tests/extreme/proof_of_superiority.rs`:
  `test_proof_hippocampal_indexing_efficiency` (embedding-compression bench).
- `tests/e2e/tests/cognitive/comparative_benchmarks.rs`:
  `test_two_phase_vs_flat_search`, `test_index_compression_ratio`.
- `crates/vestige-core/src/storage/v3_regression_tests.rs`:
  `v3_delayed_embedding_cannot_resurrect_a_purged_memory` (gated).
- `crates/vestige-core/src/storage/sqlite/tests.rs` (27 tests + 5 helpers):
  runtime vector-gate tests (`vector_search_env_value_parsing`,
  `disabling_vector_search_...`, `test_runtime_vector_gate_env_...` x2),
  embedding identity/regeneration/dirty-embedding tests, embedding profile
  lifecycle tests (`init_embeddings_permits/rejects_...`,
  `reopening_after_qwen_pointer_...`, `embedding_profiles_keep_vectors_isolated`,
  `activation_rejects_ready_...`, `migration_vector_and_node_checkpoint_...`,
  `purge_removes_vectors_from_every_embedding_profile` + `ready_profile_manifest`
  helper), `non_256_active_profile_...`, all peer/vector-journal tests
  (`peer_process_write_...`, `peer_reembedding_...`, `peer_purge_...`,
  `peer_content_edit_...`, `own_writes_...`, `a_journal_pruned_...`,
  `vector_journal_prune_...`), `embedding_maintenance_preview_...`,
  `embedding_write_rejects_stale_...`,
  `purge_and_profile_activation_do_not_deadlock`, and orphaned helpers
  (`with_vector_search_disabled`, `persist_test_vector`, `index_contains`,
  `nearest`, `ingest_plain`).
- `crates/vestige-mcp/tests/e2e_real_binary.rs`: embedding warm-up test,
  `the_real_embedding_runtime_...`, `tag_prefix_filtering_..._hybrid_path`
  (keyword-path sibling kept), `approved_purge_removes_the_stored_embedding`,
  `embeddings_and_semantic_retrieval_survive_a_restart`,
  `corrupt_fts_rebuild_preserves_embeddings`; removed the
  `node_embeddings`-count assertion from the keyword purge-tombstone test;
  trimmed `embeddingsCompiledIn`/`embeddingReady` from the memory_status
  health shape assertion.
- `crates/vestige-mcp/tests/e2e_failure_cases.rs`:
  `conflicting_ingest_during_a_labile_window_...` (vector-gated supersede path
  by its own doc).
- `crates/vestige-mcp/tests/common/mod.rs`: removed `wait_for_embeddings`,
  `ingest_embedded`, `ingest_inner`'s embedding assert, `wait_for_log_notification`,
  `EMBEDDING_TIMEOUT`, `EMBEDDINGS_READY`.

## Adapted (kept green WITHOUT embeddings at runtime)

- `proof_of_superiority.rs` `test_proof_multi_hop_beats_similarity`: local
  mock-vector similarity half removed; now asserts structural results only
  (4-hop membership + path). Comprehensive summary test renumbered to 5
  capabilities (hippocampal compression block removed).
- `e2e_real_binary.rs` `contradictions_...` and `correction_...`: converted
  from #[ignore]d real-embedding-runtime tests to default-suite keyword-path
  tests (their subjects are lexical by construction). Verified green.

## Borderline — left gated / left as-is for the integration pass

All still behind `#[cfg(all(feature = "embeddings", feature = "vector-search"))]`
in `crates/vestige-core/src/storage/sqlite/tests.rs`; their subject code is
itself feature-gated, so decide at integration (dedup/merge tool SURVIVES in
the 15-tool surface — if merge code is un-gated rather than deleted, these need
keyword-path fixtures, not deletion):

- Merge/supersede suite: `test_merge_candidates_threshold_classification`,
  `test_plan_merge_is_preview_only_no_mutation`,
  `concurrent_apply_of_one_plan_applies_it_exactly_once`,
  `test_merge_state_rejects_...`, `test_merge_undo_conflict_...`,
  `test_merge_undo_concurrent_...`, `test_apply_then_undo_merge_...`,
  `test_supersede_invalidates_...`, `test_protect_blocks_merge_away`,
  all `test_auto_dedup_*` (6) + `set_retention`/`with_auto_merge_env` helpers,
  `pinning_auto_merge_in_one_test_...`,
  `test_apply_requires_confirm_...`, `apply_match_obeys_...`,
  `apply_match_can_use_...`, `test_merge_policy_roundtrip_persists`.
- Validity-window suite (fixture needs active embedding profile via
  `storage_with_marker_gate_runtime` + `MarkerEmbedder`):
  `inferred_as_of_validity_...`, `inferred_as_of_must_not_resurrect_...`,
  `explicit_valid_from_on_reinforce_...`, `create_path_still_stamps_...`,
  `older_dated_claim_after_newer_fact_...`.
- Reconsolidation suite (same fixture): `contradiction_during_live_window_...`,
  `approve_verdict_applies_...`, `reject_verdict_leaves_...`,
  `quarantine_verdict_suppresses_...`, `expired_window_auto_closes_...`,
  `pending_reconsolidation_plans_...` + `reconsolidation_candidate` helper.
- Vector fixture helpers still used by the above: `seed_node`, `axis_vector`.
- `comparative_benchmarks.rs` `test_barcode_orthogonality` +
  `test_content_pointer_accuracy`: structural (hash barcodes, content
  pointers) but reference `neuroscience::hippocampal_index`. If w1a deletes
  that module, these go with it. NOTE: the vestige-e2e-tests crate defines no
  features, so a `#[cfg(feature = "embeddings")]` there would be silently
  always-off; they were left running instead.
- `e2e_real_binary.rs` `foreign_key_orphans_are_repaired_...`: uses
  `node_embeddings` as a generic CASCADE-child fixture. If the table is
  dropped, swap the fixture to another CASCADE child (e.g. fsrs_cards).
- `tests/e2e/tests/journeys/ingest_recall_review.rs`
  `test_recall_search_modes`: constructs `SearchMode::Semantic` (w1b scope).
- `tests/e2e/Cargo.toml` still requests `features = ["embeddings",
  "vector-search"]` from vestige-core — drop when the features die.

## Notes for integration

- No e2e scenario exercised `match_context`/`search` tool aliases at this
  base (grep-verified); nothing to flip to removal-error assertions. The
  aliases exist only in source (`crates/vestige-mcp/src/server.rs`). HOWEVER,
  `cargo test -p vestige-mcp --lib` at this base already fails
  `server::tests::test_recall_lookup_matches_search_shape`
  ("recall(mode=lookup) must equal search byte-for-byte") — that IS a
  search-alias-expects-success test living in src tests; w1d should flip it
  to expect the removal error. Second pre-existing failure:
  `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`.
  Both verified pre-existing via `git stash` roundtrip at base 5fbe1df
  (lib build contains none of this branch's changes).
- `crates/vestige-core/src/storage/sqlite/embeddings.rs`:
  `embedding_model_matches_active` / `embedding_vector_for_active_model` are
  now unused (their only callers were deleted tests) — dead-code warning under
  default features; resolves itself when w1a deletes the embeddings module.
- `cargo check --tests --no-default-features` (workspace): FAILS in NON-TEST
  source, pre-existing at base: `crates/vestige-mcp/src/tools/search_unified.rs:1261`
  E0382 "use of moved value: `rerank_candidates`" (into_iter at 1257, reused at
  1261; the no-default-features branch of the reranker). Not fixed here per
  owner order — belongs to w1b/w1d. vestige-core's test build compiles clean
  under no-default-features apart from the dead-code warning above.

## Verification (this tree, default features)

- `cargo test -p vestige-core --lib`: 936 passed, 0 failed, 1 ignored.
- `cargo test -p vestige-e2e-tests --test comparative_benchmarks --test
  proof_of_superiority`: 23 + 4 passed, 0 failed.
- `cargo test -p vestige-mcp --lib`: 756 passed, 2 failed — both failures
  pre-existing at base (see Notes), unaffected by this branch.
- `cargo test -p vestige-mcp --test e2e_real_binary -- contradictions
  correction`: 2 passed (the two keyword-path conversions).
- `cargo check -p vestige-e2e-tests --tests`: clean.
# W1E Test Cleanup — Scope Handoff

Branch `build/w1e-tests-cleanup` (base: main @ 5fbe1df). Test hygiene ONLY, per
owner order. Source modules still present in this tree; parallel agents
(w1a-core-kill, w1b-search-collapse, w1c-ingest-flip, w1d-mcp-surface) own the
source deletions.

## Deleted (clearly vector/embedding-only)

- `tests/e2e/src/mocks/mock_embedding.rs` (whole file; no test used
  `MockEmbeddingService` directly — only re-exports in `tests/e2e/src/lib.rs`
  and `tests/e2e/src/mocks/mod.rs`, both cleaned).
- `tests/e2e/tests/extreme/proof_of_superiority.rs`:
  `test_proof_hippocampal_indexing_efficiency` (embedding-compression bench).
- `tests/e2e/tests/cognitive/comparative_benchmarks.rs`:
  `test_two_phase_vs_flat_search`, `test_index_compression_ratio`.
- `crates/vestige-core/src/storage/v3_regression_tests.rs`:
  `v3_delayed_embedding_cannot_resurrect_a_purged_memory` (gated).
- `crates/vestige-core/src/storage/sqlite/tests.rs` (27 tests + 5 helpers):
  runtime vector-gate tests (`vector_search_env_value_parsing`,
  `disabling_vector_search_...`, `test_runtime_vector_gate_env_...` x2),
  embedding identity/regeneration/dirty-embedding tests, embedding profile
  lifecycle tests (`init_embeddings_permits/rejects_...`,
  `reopening_after_qwen_pointer_...`, `embedding_profiles_keep_vectors_isolated`,
  `activation_rejects_ready_...`, `migration_vector_and_node_checkpoint_...`,
  `purge_removes_vectors_from_every_embedding_profile` + `ready_profile_manifest`
  helper), `non_256_active_profile_...`, all peer/vector-journal tests
  (`peer_process_write_...`, `peer_reembedding_...`, `peer_purge_...`,
  `peer_content_edit_...`, `own_writes_...`, `a_journal_pruned_...`,
  `vector_journal_prune_...`), `embedding_maintenance_preview_...`,
  `embedding_write_rejects_stale_...`,
  `purge_and_profile_activation_do_not_deadlock`, and orphaned helpers
  (`with_vector_search_disabled`, `persist_test_vector`, `index_contains`,
  `nearest`, `ingest_plain`).
- `crates/vestige-mcp/tests/e2e_real_binary.rs`: embedding warm-up test,
  `the_real_embedding_runtime_...`, `tag_prefix_filtering_..._hybrid_path`
  (keyword-path sibling kept), `approved_purge_removes_the_stored_embedding`,
  `embeddings_and_semantic_retrieval_survive_a_restart`,
  `corrupt_fts_rebuild_preserves_embeddings`; removed the
  `node_embeddings`-count assertion from the keyword purge-tombstone test;
  trimmed `embeddingsCompiledIn`/`embeddingReady` from the memory_status
  health shape assertion.
- `crates/vestige-mcp/tests/e2e_failure_cases.rs`:
  `conflicting_ingest_during_a_labile_window_...` (vector-gated supersede path
  by its own doc).
- `crates/vestige-mcp/tests/common/mod.rs`: removed `wait_for_embeddings`,
  `ingest_embedded`, `ingest_inner`'s embedding assert, `wait_for_log_notification`,
  `EMBEDDING_TIMEOUT`, `EMBEDDINGS_READY`.

## Adapted (kept green WITHOUT embeddings at runtime)

- `proof_of_superiority.rs` `test_proof_multi_hop_beats_similarity`: local
  mock-vector similarity half removed; now asserts structural results only
  (4-hop membership + path). Comprehensive summary test renumbered to 5
  capabilities (hippocampal compression block removed).
- `e2e_real_binary.rs` `contradictions_...` and `correction_...`: converted
  from #[ignore]d real-embedding-runtime tests to default-suite keyword-path
  tests (their subjects are lexical by construction). Verified green.

## Borderline — left gated / left as-is for the integration pass

All still behind `#[cfg(all(feature = "embeddings", feature = "vector-search"))]`
in `crates/vestige-core/src/storage/sqlite/tests.rs`; their subject code is
itself feature-gated, so decide at integration (dedup/merge tool SURVIVES in
the 15-tool surface — if merge code is un-gated rather than deleted, these need
keyword-path fixtures, not deletion):

- Merge/supersede suite: `test_merge_candidates_threshold_classification`,
  `test_plan_merge_is_preview_only_no_mutation`,
  `concurrent_apply_of_one_plan_applies_it_exactly_once`,
  `test_merge_state_rejects_...`, `test_merge_undo_conflict_...`,
  `test_merge_undo_concurrent_...`, `test_apply_then_undo_merge_...`,
  `test_supersede_invalidates_...`, `test_protect_blocks_merge_away`,
  all `test_auto_dedup_*` (6) + `set_retention`/`with_auto_merge_env` helpers,
  `pinning_auto_merge_in_one_test_...`,
  `test_apply_requires_confirm_...`, `apply_match_obeys_...`,
  `apply_match_can_use_...`, `test_merge_policy_roundtrip_persists`.
- Validity-window suite (fixture needs active embedding profile via
  `storage_with_marker_gate_runtime` + `MarkerEmbedder`):
  `inferred_as_of_validity_...`, `inferred_as_of_must_not_resurrect_...`,
  `explicit_valid_from_on_reinforce_...`, `create_path_still_stamps_...`,
  `older_dated_claim_after_newer_fact_...`.
- Reconsolidation suite (same fixture): `contradiction_during_live_window_...`,
  `approve_verdict_applies_...`, `reject_verdict_leaves_...`,
  `quarantine_verdict_suppresses_...`, `expired_window_auto_closes_...`,
  `pending_reconsolidation_plans_...` + `reconsolidation_candidate` helper.
- Vector fixture helpers still used by the above: `seed_node`, `axis_vector`.
- `comparative_benchmarks.rs` `test_barcode_orthogonality` +
  `test_content_pointer_accuracy`: structural (hash barcodes, content
  pointers) but reference `neuroscience::hippocampal_index`. If w1a deletes
  that module, these go with it. NOTE: the vestige-e2e-tests crate defines no
  features, so a `#[cfg(feature = "embeddings")]` there would be silently
  always-off; they were left running instead.
- `e2e_real_binary.rs` `foreign_key_orphans_are_repaired_...`: uses
  `node_embeddings` as a generic CASCADE-child fixture. If the table is
  dropped, swap the fixture to another CASCADE child (e.g. fsrs_cards).
- `tests/e2e/tests/journeys/ingest_recall_review.rs`
  `test_recall_search_modes`: constructs `SearchMode::Semantic` (w1b scope).
- `tests/e2e/Cargo.toml` still requests `features = ["embeddings",
  "vector-search"]` from vestige-core — drop when the features die.

## Notes for integration

- No e2e scenario exercised `match_context`/`search` tool aliases at this
  base (grep-verified); nothing to flip to removal-error assertions. The
  aliases exist only in source (`crates/vestige-mcp/src/server.rs`). HOWEVER,
  `cargo test -p vestige-mcp --lib` at this base already fails
  `server::tests::test_recall_lookup_matches_search_shape`
  ("recall(mode=lookup) must equal search byte-for-byte") — that IS a
  search-alias-expects-success test living in src tests; w1d should flip it
  to expect the removal error. Second pre-existing failure:
  `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`.
  Both verified pre-existing via `git stash` roundtrip at base 5fbe1df
  (lib build contains none of this branch's changes).
- `crates/vestige-core/src/storage/sqlite/embeddings.rs`:
  `embedding_model_matches_active` / `embedding_vector_for_active_model` are
  now unused (their only callers were deleted tests) — dead-code warning under
  default features; resolves itself when w1a deletes the embeddings module.
- `cargo check --tests --no-default-features` (workspace): FAILS in NON-TEST
  source, pre-existing at base: `crates/vestige-mcp/src/tools/search_unified.rs:1261`
  E0382 "use of moved value: `rerank_candidates`" (into_iter at 1257, reused at
  1261; the no-default-features branch of the reranker). Not fixed here per
  owner order — belongs to w1b/w1d. vestige-core's test build compiles clean
  under no-default-features apart from the dead-code warning above.

## Verification (this tree, default features)

- `cargo test -p vestige-core --lib`: 936 passed, 0 failed, 1 ignored.
- `cargo test -p vestige-e2e-tests --test comparative_benchmarks --test
  proof_of_superiority`: 23 + 4 passed, 0 failed.
- `cargo test -p vestige-mcp --lib`: 756 passed, 2 failed — both failures
  pre-existing at base (see Notes), unaffected by this branch.
- `cargo test -p vestige-mcp --test e2e_real_binary -- contradictions
  correction`: 2 passed (the two keyword-path conversions).
- `cargo check -p vestige-e2e-tests --tests`: clean.

# SCOPE-HANDOFF — causal_walk (build/w3a-causal-walk)

`causal_walk` REPLACES `backfill` as the advertised flagship. `backfill`
(module `crates/vestige-mcp/src/tools/backfill.rs`, core
`advanced/retroactive_backfill.rs`) is UNMODIFIED and stays dispatchable as a
hidden alias. Nothing else outside the files below was touched.

## Files

| File | Change |
| --- | --- |
| `crates/vestige-core/src/advanced/causal_walk.rs` | NEW — engine + storage assembly + promote + 10 unit tests |
| `crates/vestige-core/src/advanced/mod.rs` | `pub mod causal_walk;` + re-exports |
| `crates/vestige-mcp/src/tools/causal_walk.rs` | NEW — schema + execute + tests |
| `crates/vestige-mcp/src/tools/mod.rs` | `pub mod causal_walk;` |
| `crates/vestige-mcp/src/tools/compact.rs` | full-schema registry: `backfill` → `causal_walk` (registry tracks the ADVERTISED catalog; hidden names must not enter it) |
| `crates/vestige-mcp/src/server.rs` | catalog entry swapped (count stays 16), `causal_walk` dispatch arm added, `backfill` arm marked hidden alias, advertised-surface test updated |
| `crates/vestige-mcp/src/bin/cli.rs` | NEW subcommand block only: `CausalWalk` variant + dispatch arm + `run_causal_walk` handler |
| `SCOPE-HANDOFF.md` | this file |

## Mechanisms (start point → suspects)

1. `failing_test {name}` → records naming the test (test-file records) →
   commit records touching their path-tier entities → each commit's OTHER
   files are the suspects. Hop: `failing_test/co_touch`, hop text
   `<test-file> -> <other files>`.
2. `stack_frame {frame: "file:line"|"file"}` → commit records whose hunk
   spans (files / `file/symbol` symbols) cover the file → the LAST commit
   touching it before the failure anchor is the prime suspect (SZZ-lite;
   earlier touchers land in `rejected`). Hop: `stack_frame/last_toucher`.
3. `ci_run {run_id}` → `agent_traces` run via `storage.get_trace(run_id)` →
   failure channel = `SanhedrinVeto` claim text + `MemoryRetrieve`/`MemoryWrite`/
   veto `evidence_ids` records → their entities → records. Hop:
   `ci_run/failed_calls`.
4. `logged_write {node_id}` → the node → its `get_connections_for_memory`
   edges (neighbours seeded as edge evidence, surviving with 0 shared
   anchors) + records sharing its entities. Hops: `logged_write/edge`,
   `logged_write/shared_anchor`.
5. `version_range {worked_in, broke_in, repo}` → `git rev-list
   worked_in..broke_in` sha set (`git_records::parse_rev_list`) → RESTRICTOR:
   out-of-range commit seeds rejected `outside version range W..B`; in-range
   commits overlapping the anchor pool get hop `version_range/rev_list`.
   Alone (no failure start point) it yields no anchors → needs_report.
   Multiple distinct repos → needs_report (single repo per call).

## Core API (vestige-core::advanced::causal_walk)

```rust
pub const EVIDENCE_LINK_TYPE: &str = "evidence_of";
pub const MAX_HOPS: usize = 2;

pub enum StartPoint { FailingTest{name}, StackFrame{frame}, CiRun{run_id},
                      LoggedWrite{node_id}, VersionRange{worked_in,broke_in,repo} }
// serde: {"kind":"failing_test","name":...} etc. (tag="kind", snake_case)

pub struct CausalWalkRequest { pub scope: String, pub start_points: Vec<StartPoint>,
                               pub lookback_days: i64, pub scan_limit: i32 } // Default: user/30/500
pub struct PathHop { pub via: String, pub hop: String }
pub struct CausalCause { pub id: String, pub sha: Option<String>, pub score: f64,
                         pub path: Vec<PathHop>, pub shared_anchors: Vec<String>,
                         #[serde(skip)] pub evidence_to: Vec<String> }
pub struct NeedsReport { pub missing: Vec<String>, pub required_start_points: Vec<String> }
pub struct WalkRejection { pub id: String, pub reason: String, pub shared_anchors: usize }
pub struct CausalWalkResult { pub causes: Vec<CausalCause>,
                              pub needs_report: Option<NeedsReport>,
                              pub rejected: Vec<WalkRejection> }

pub fn walk_storage(storage: &Storage, req: &CausalWalkRequest) -> Result<CausalWalkResult, String>;
pub fn persist_evidence_edges(storage: &Storage, result: &CausalWalkResult)
    -> Result<Vec<(String,String)>, String>;  // cause -> evidence, link_type "evidence_of"
```

`CausalWalkOptions { lookback_days, max_causes, max_rejections }::walk(starts, records, ctx)`
is the pure engine (records = `WalkRecord`, ctx = runs/edges/rev-lists) if you
need it without a store.

## Output schema (MCP `causal_walk`)

```
{ causes: [{id, sha?, score, path: [{via, hop}], shared_anchors, content_preview}],
  needs_report?: {missing, required_start_points},
  rejected: [{id, reason, shared_anchors, content_preview}],   // top 3
  promote: {edges_persisted, edges: [[cause, evidence]], link_type: "evidence_of"},
  preview, scope, headline, evidence_status: "hypothesis", causality_verified: false }
```

No start point / unanchored handle → `needs_report` (never an error, never
guessed). Promote is the ONLY write path.

## Ranking (ported from run_trail)

score = Σ probabilistic-IDF(anchor) × tier weight (Path 1.0 / Code 0.9 /
Version 0.6 / Word 0.3, df over the in-window candidate pool) + 0.3 recency
term + 0.25 change-record bonus. Sort: score desc → change-record first →
older first. Window: candidate created_at ≤ failure anchor (= newest
start-point-anchored record, else now) and within lookback_days (default 30).

## CLI

`vestige causal-walk --failing-test NAME | --stack-frame F[:L] | --ci-run ID |
--logged-write ID | (--git-repo P --worked-in T --broke-in T)`
`[--lookback-days N] [--no-promote] [--scope S] [--json]`

## Tests

- `cargo test -p vestige-core --lib causal_walk` — 10 tests: co-touch walk,
  stack_frame last-toucher, version_range restriction (real temp git repo),
  no-start refusal, wrong-handle refusal (x2), ci_run trace walk,
  logged_write edge walk, promote vs preview, unresolvable range refusal,
  record-format parsing.
- `cargo test -p vestige-mcp --lib causal` — schema variant test + tokio
  end-to-end (temp storage): needs_report, preview writes nothing,
  promote persists `evidence_of` edges.
- Server surface: advertised count stays 16 (`causal_walk` in, `backfill`
  hidden); full-schema registry test, wire-budget test, and hints test
  updated accordingly.

## Notes for adjacent agents

- Do NOT re-add `backfill` to the advertised catalog or `compact::full_schema`
  registry — `full_schema_registry_matches_the_advertised_catalog` fails if
  catalog and registry diverge.
- Richer edge APIs: keep persisting through `save_connection` with
  `link_type = "evidence_of"` (constant `causal_walk::EVIDENCE_LINK_TYPE`) so
  `memory_connections` stays compatible; `CausalCause::evidence_to` carries
  the per-cause link targets (serde-skipped).
- Supersession is intentionally NOT followed in causal_walk (MVP bound);
  commit records upsert idempotently via `(git, repo#sha)`.
# SCOPE-HANDOFF — causal_walk (build/w3a-causal-walk)

`causal_walk` REPLACES `backfill` as the advertised flagship. `backfill`
(module `crates/vestige-mcp/src/tools/backfill.rs`, core
`advanced/retroactive_backfill.rs`) is UNMODIFIED and stays dispatchable as a
hidden alias. Nothing else outside the files below was touched.

## Files

| File | Change |
| --- | --- |
| `crates/vestige-core/src/advanced/causal_walk.rs` | NEW — engine + storage assembly + promote + 10 unit tests |
| `crates/vestige-core/src/advanced/mod.rs` | `pub mod causal_walk;` + re-exports |
| `crates/vestige-mcp/src/tools/causal_walk.rs` | NEW — schema + execute + tests |
| `crates/vestige-mcp/src/tools/mod.rs` | `pub mod causal_walk;` |
| `crates/vestige-mcp/src/tools/compact.rs` | full-schema registry: `backfill` → `causal_walk` (registry tracks the ADVERTISED catalog; hidden names must not enter it) |
| `crates/vestige-mcp/src/server.rs` | catalog entry swapped (count stays 16), `causal_walk` dispatch arm added, `backfill` arm marked hidden alias, advertised-surface test updated |
| `crates/vestige-mcp/src/bin/cli.rs` | NEW subcommand block only: `CausalWalk` variant + dispatch arm + `run_causal_walk` handler |
| `SCOPE-HANDOFF.md` | this file |

## Mechanisms (start point → suspects)

1. `failing_test {name}` → records naming the test (test-file records) →
   commit records touching their path-tier entities → each commit's OTHER
   files are the suspects. Hop: `failing_test/co_touch`, hop text
   `<test-file> -> <other files>`.
2. `stack_frame {frame: "file:line"|"file"}` → commit records whose hunk
   spans (files / `file/symbol` symbols) cover the file → the LAST commit
   touching it before the failure anchor is the prime suspect (SZZ-lite;
   earlier touchers land in `rejected`). Hop: `stack_frame/last_toucher`.
3. `ci_run {run_id}` → `agent_traces` run via `storage.get_trace(run_id)` →
   failure channel = `SanhedrinVeto` claim text + `MemoryRetrieve`/`MemoryWrite`/
   veto `evidence_ids` records → their entities → records. Hop:
   `ci_run/failed_calls`.
4. `logged_write {node_id}` → the node → its `get_connections_for_memory`
   edges (neighbours seeded as edge evidence, surviving with 0 shared
   anchors) + records sharing its entities. Hops: `logged_write/edge`,
   `logged_write/shared_anchor`.
5. `version_range {worked_in, broke_in, repo}` → `git rev-list
   worked_in..broke_in` sha set (`git_records::parse_rev_list`) → RESTRICTOR:
   out-of-range commit seeds rejected `outside version range W..B`; in-range
   commits overlapping the anchor pool get hop `version_range/rev_list`.
   Alone (no failure start point) it yields no anchors → needs_report.
   Multiple distinct repos → needs_report (single repo per call).

## Core API (vestige-core::advanced::causal_walk)

```rust
pub const EVIDENCE_LINK_TYPE: &str = "evidence_of";
pub const MAX_HOPS: usize = 2;

pub enum StartPoint { FailingTest{name}, StackFrame{frame}, CiRun{run_id},
                      LoggedWrite{node_id}, VersionRange{worked_in,broke_in,repo} }
// serde: {"kind":"failing_test","name":...} etc. (tag="kind", snake_case)

pub struct CausalWalkRequest { pub scope: String, pub start_points: Vec<StartPoint>,
                               pub lookback_days: i64, pub scan_limit: i32 } // Default: user/30/500
pub struct PathHop { pub via: String, pub hop: String }
pub struct CausalCause { pub id: String, pub sha: Option<String>, pub score: f64,
                         pub path: Vec<PathHop>, pub shared_anchors: Vec<String>,
                         #[serde(skip)] pub evidence_to: Vec<String> }
pub struct NeedsReport { pub missing: Vec<String>, pub required_start_points: Vec<String> }
pub struct WalkRejection { pub id: String, pub reason: String, pub shared_anchors: usize }
pub struct CausalWalkResult { pub causes: Vec<CausalCause>,
                              pub needs_report: Option<NeedsReport>,
                              pub rejected: Vec<WalkRejection> }

pub fn walk_storage(storage: &Storage, req: &CausalWalkRequest) -> Result<CausalWalkResult, String>;
pub fn persist_evidence_edges(storage: &Storage, result: &CausalWalkResult)
    -> Result<Vec<(String,String)>, String>;  // cause -> evidence, link_type "evidence_of"
```

`CausalWalkOptions { lookback_days, max_causes, max_rejections }::walk(starts, records, ctx)`
is the pure engine (records = `WalkRecord`, ctx = runs/edges/rev-lists) if you
need it without a store.

## Output schema (MCP `causal_walk`)

```
{ causes: [{id, sha?, score, path: [{via, hop}], shared_anchors, content_preview}],
  needs_report?: {missing, required_start_points},
  rejected: [{id, reason, shared_anchors, content_preview}],   // top 3
  promote: {edges_persisted, edges: [[cause, evidence]], link_type: "evidence_of"},
  preview, scope, headline, evidence_status: "hypothesis", causality_verified: false }
```

No start point / unanchored handle → `needs_report` (never an error, never
guessed). Promote is the ONLY write path.

## Ranking (ported from run_trail)

score = Σ probabilistic-IDF(anchor) × tier weight (Path 1.0 / Code 0.9 /
Version 0.6 / Word 0.3, df over the in-window candidate pool) + 0.3 recency
term + 0.25 change-record bonus. Sort: score desc → change-record first →
older first. Window: candidate created_at ≤ failure anchor (= newest
start-point-anchored record, else now) and within lookback_days (default 30).

## CLI

`vestige causal-walk --failing-test NAME | --stack-frame F[:L] | --ci-run ID |
--logged-write ID | (--git-repo P --worked-in T --broke-in T)`
`[--lookback-days N] [--no-promote] [--scope S] [--json]`

## Tests

- `cargo test -p vestige-core --lib causal_walk` — 10 tests: co-touch walk,
  stack_frame last-toucher, version_range restriction (real temp git repo),
  no-start refusal, wrong-handle refusal (x2), ci_run trace walk,
  logged_write edge walk, promote vs preview, unresolvable range refusal,
  record-format parsing.
- `cargo test -p vestige-mcp --lib causal` — schema variant test + tokio
  end-to-end (temp storage): needs_report, preview writes nothing,
  promote persists `evidence_of` edges.
- Server surface: advertised count stays 16 (`causal_walk` in, `backfill`
  hidden); full-schema registry test, wire-budget test, and hints test
  updated accordingly.

## Notes for adjacent agents

- Do NOT re-add `backfill` to the advertised catalog or `compact::full_schema`
  registry — `full_schema_registry_matches_the_advertised_catalog` fails if
  catalog and registry diverge.
- Richer edge APIs: keep persisting through `save_connection` with
  `link_type = "evidence_of"` (constant `causal_walk::EVIDENCE_LINK_TYPE`) so
  `memory_connections` stays compatible; `CausalCause::evidence_to` carries
  the per-cause link targets (serde-skipped).
- Supersession is intentionally NOT followed in causal_walk (MVP bound);
  commit records upsert idempotently via `(git, repo#sha)`.

# SCOPE HANDOFF — build/w1c-ingest-flip (ingest-flip: no similarity, exact-equality nomination)

Branch: `build/w1c-ingest-flip`, base = `main @ 5fbe1df`. Not pushed.

Owner decision implemented: NO similarity anywhere; dedup/merge nomination =
exact equality only (identical content hash / same declared source key / exact
entity-set equality); edges only from declared sources.

## What changed in this branch's scope

- `crates/vestige-core/src/advanced/prediction_error.rs`
  - `evaluate` / `evaluate_with_intent` dropped the `new_embedding` param.
  - `CandidateMemory.embedding` field deleted.
  - `cosine_similarity` deleted; replaced by `content_similarity` (Dice
    coefficient over lowercased alphanumeric tokens > 2 chars). Exported via
    `advanced::content_similarity`.
  - `NEAR_IDENTICAL_THRESHOLD` recalibrated 0.92 → 0.85 for the Dice scale so
    benign near-paraphrases still Reinforce; contradiction-marker check runs
    before the threshold, so corrections never reinforce.
- `crates/vestige-core/src/advanced/merge_supersede.rs`
  - `W_EMBEDDING` removed. New weights: `W_TAGS = 0.40`, `W_TOKENS = 0.60`
    (sum = 1.0). Comment states nomination is exact-equality-first and scores
    are tie-breakers/review labels only.
  - `MatchSignals.embedding_similarity` field REMOVED (breaking API change).
  - `score_pair` signature is now `score_pair(a_tags, b_tags, a_content, b_content)`.
- `crates/vestige-core/src/advanced/dreams.rs`
  - Both cosine branches deleted; `tag_similarity*0.4 + content_word_similarity*0.6`
    is now the whole computation. Module-level `cosine_similarity` fn deleted.
  - `DreamMemory.embedding` field KEPT but inert (out-of-scope constructors
    still set it — see below).
- `crates/vestige-core/src/storage/sqlite/merge.rs`
  - `merge_candidates`: O(n²) cosine scan deleted (was embeddings+vector-search
    cfg-gated). Now UNGATED and nominates via: (1) SQL group-by on
    `COALESCE(content_hash, content)`; (2) SQL group-by on the source key at
    the same granularity as the store's UNIQUE index
    `(source_system, COALESCE(source_project,''), source_id)`; (3) exact
    equality of non-empty `retroactive_backfill::extract_entities` sets
    (in-memory compare). Classification is advisory — nominated clusters are
    surfaced even when labelled NonMatch (shared source key with diverged
    content is review-worthy; old drop-on-NonMatch was a cosine-era guard).
  - `pair_similarity` deleted (embedding-based); plan_merge / plan_supersede /
    plan_reconsolidation now call the 4-arg `score_pair` directly.
  - New test module `exact_nomination_tests` (6 tests).
- `crates/vestige-core/src/storage/sqlite/ingest.rs` (embedding paths only)
  - Smart ingest: embedding-runtime preflight, `get_document_embedding`, and
    `semantic_search_raw` candidate retrieval REMOVED. Candidate selection is
    keyword-only via `Storage::search` (FTS5/BM25, public API in search.rs —
    not modified). `get_node_embedding` per-candidate fetch removed.
  - KEPT (cfg-gated, deletion belongs to the embeddings agent):
    `generate_embedding_for_node` after raw insert, and vector invalidation +
    regeneration in `update_node_content_unchecked`. These are vector STORAGE
    hygiene, not similarity decisions; removing them broke
    `embedding::lifecycle` and `peer_content_edit_invalidates_vectors` tests
    that the embeddings-deletion agent will remove wholesale.
  - NOTE: the whole `smart_ingest*` chain is still
    `#[cfg(all(feature = "embeddings", feature = "vector-search"))]`-gated. It
    no longer needs embeddings at all — the embeddings agent should un-gate it
    when the feature flags go.
- `crates/vestige-mcp/src/tools/dedup.rs`
  - `find_duplicates` (`execute`): cosine clustering + UnionFind deleted;
    now UNGATED exact-equality grouping (content-hash identity + source key at
    UNIQUE-index granularity). `similarity_threshold` parameter removed from
    the schema (unknown args are ignored by serde, so old callers keep
    working). Output: `similarityToAnchor` → `matchRelation`
    (anchor|content|source_key|transitive); `totalWithEmbeddings` →
    `totalScanned`; `threshold`/`pairsChecked` removed.
  - plan/apply/undo/verdict/tag_*/protect/policy flows untouched.

## Forced out-of-scope compile fixes (minimal, mechanical)

- `crates/vestige-core/src/advanced/mod.rs`: re-export
  `prediction_error::cosine_similarity` → `content_similarity`.
- `crates/vestige-mcp/src/tools/merge.rs` (2 sites): dropped the
  `embeddingSimilarity` output lines referencing the removed
  `MatchSignals.embedding_similarity` field. Everything else in that file is
  untouched; NOTE its `merge_candidates_schema()` may still advertise a
  `similarity_threshold`-style knob cosmetically (not compile-relevant).

## For the embeddings-deletion agent (cross-scope notes)

- After deleting `storage/sqlite/embeddings.rs` and the `embeddings/` module:
  - `vestige_core::cosine_similarity` re-export at `lib.rs:572` must go.
  - `get_document_embedding` is already dead code (warning) — disappears with
    the module.
  - Remaining callers of `storage.get_node_embedding(...)` OUTSIDE my scope:
    `crates/vestige-mcp/src/tools/dream.rs` (~line 88) and
    `crates/vestige-mcp/src/dashboard/handlers.rs` (~line 1863) — they only
    populate the now-inert `DreamMemory.embedding`; delete those calls and the
    field.
  - `lifecycle.rs` / `search.rs` untouched per scope (search.rs `hybrid_search`
    etc. still reference `semantic_search_raw`).
  - Un-gate the `smart_ingest*` chain + merge plan/apply/undo cfg gates when
    the features are dropped (see above).
  - `knowledge_nodes.has_embedding` / `node_embeddings` /
    `embedding_profile_vectors` schema: my files no longer maintain them
    except the cfg-gated blocks noted above.

## Test status (default features)

- `cargo test -p vestige-core`: 971 passed, 0 failed (1 ignored, pre-existing).
- `cargo test -p vestige-mcp`: 759 passed, 2 failed:
  - `server::tests::test_recall_lookup_matches_search_shape`
  - `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`
  Both are verified PRE-EXISTING on base `5fbe1df` (checked in a clean
  worktree); they are fallout from commit 7bc2516's search-tool removal and
  backfill rework, not from this branch.
- All in-scope test modules green: prediction_error (16), merge_supersede,
  dreams, sqlite::merge::exact_nomination_tests (6), mcp tools::dedup (12).
# SCOPE HANDOFF — build/w1c-ingest-flip (ingest-flip: no similarity, exact-equality nomination)

Branch: `build/w1c-ingest-flip`, base = `main @ 5fbe1df`. Not pushed.

Owner decision implemented: NO similarity anywhere; dedup/merge nomination =
exact equality only (identical content hash / same declared source key / exact
entity-set equality); edges only from declared sources.

## What changed in this branch's scope

- `crates/vestige-core/src/advanced/prediction_error.rs`
  - `evaluate` / `evaluate_with_intent` dropped the `new_embedding` param.
  - `CandidateMemory.embedding` field deleted.
  - `cosine_similarity` deleted; replaced by `content_similarity` (Dice
    coefficient over lowercased alphanumeric tokens > 2 chars). Exported via
    `advanced::content_similarity`.
  - `NEAR_IDENTICAL_THRESHOLD` recalibrated 0.92 → 0.85 for the Dice scale so
    benign near-paraphrases still Reinforce; contradiction-marker check runs
    before the threshold, so corrections never reinforce.
- `crates/vestige-core/src/advanced/merge_supersede.rs`
  - `W_EMBEDDING` removed. New weights: `W_TAGS = 0.40`, `W_TOKENS = 0.60`
    (sum = 1.0). Comment states nomination is exact-equality-first and scores
    are tie-breakers/review labels only.
  - `MatchSignals.embedding_similarity` field REMOVED (breaking API change).
  - `score_pair` signature is now `score_pair(a_tags, b_tags, a_content, b_content)`.
- `crates/vestige-core/src/advanced/dreams.rs`
  - Both cosine branches deleted; `tag_similarity*0.4 + content_word_similarity*0.6`
    is now the whole computation. Module-level `cosine_similarity` fn deleted.
  - `DreamMemory.embedding` field KEPT but inert (out-of-scope constructors
    still set it — see below).
- `crates/vestige-core/src/storage/sqlite/merge.rs`
  - `merge_candidates`: O(n²) cosine scan deleted (was embeddings+vector-search
    cfg-gated). Now UNGATED and nominates via: (1) SQL group-by on
    `COALESCE(content_hash, content)`; (2) SQL group-by on the source key at
    the same granularity as the store's UNIQUE index
    `(source_system, COALESCE(source_project,''), source_id)`; (3) exact
    equality of non-empty `retroactive_backfill::extract_entities` sets
    (in-memory compare). Classification is advisory — nominated clusters are
    surfaced even when labelled NonMatch (shared source key with diverged
    content is review-worthy; old drop-on-NonMatch was a cosine-era guard).
  - `pair_similarity` deleted (embedding-based); plan_merge / plan_supersede /
    plan_reconsolidation now call the 4-arg `score_pair` directly.
  - New test module `exact_nomination_tests` (6 tests).
- `crates/vestige-core/src/storage/sqlite/ingest.rs` (embedding paths only)
  - Smart ingest: embedding-runtime preflight, `get_document_embedding`, and
    `semantic_search_raw` candidate retrieval REMOVED. Candidate selection is
    keyword-only via `Storage::search` (FTS5/BM25, public API in search.rs —
    not modified). `get_node_embedding` per-candidate fetch removed.
  - KEPT (cfg-gated, deletion belongs to the embeddings agent):
    `generate_embedding_for_node` after raw insert, and vector invalidation +
    regeneration in `update_node_content_unchecked`. These are vector STORAGE
    hygiene, not similarity decisions; removing them broke
    `embedding::lifecycle` and `peer_content_edit_invalidates_vectors` tests
    that the embeddings-deletion agent will remove wholesale.
  - NOTE: the whole `smart_ingest*` chain is still
    `#[cfg(all(feature = "embeddings", feature = "vector-search"))]`-gated. It
    no longer needs embeddings at all — the embeddings agent should un-gate it
    when the feature flags go.
- `crates/vestige-mcp/src/tools/dedup.rs`
  - `find_duplicates` (`execute`): cosine clustering + UnionFind deleted;
    now UNGATED exact-equality grouping (content-hash identity + source key at
    UNIQUE-index granularity). `similarity_threshold` parameter removed from
    the schema (unknown args are ignored by serde, so old callers keep
    working). Output: `similarityToAnchor` → `matchRelation`
    (anchor|content|source_key|transitive); `totalWithEmbeddings` →
    `totalScanned`; `threshold`/`pairsChecked` removed.
  - plan/apply/undo/verdict/tag_*/protect/policy flows untouched.

## Forced out-of-scope compile fixes (minimal, mechanical)

- `crates/vestige-core/src/advanced/mod.rs`: re-export
  `prediction_error::cosine_similarity` → `content_similarity`.
- `crates/vestige-mcp/src/tools/merge.rs` (2 sites): dropped the
  `embeddingSimilarity` output lines referencing the removed
  `MatchSignals.embedding_similarity` field. Everything else in that file is
  untouched; NOTE its `merge_candidates_schema()` may still advertise a
  `similarity_threshold`-style knob cosmetically (not compile-relevant).

## For the embeddings-deletion agent (cross-scope notes)

- After deleting `storage/sqlite/embeddings.rs` and the `embeddings/` module:
  - `vestige_core::cosine_similarity` re-export at `lib.rs:572` must go.
  - `get_document_embedding` is already dead code (warning) — disappears with
    the module.
  - Remaining callers of `storage.get_node_embedding(...)` OUTSIDE my scope:
    `crates/vestige-mcp/src/tools/dream.rs` (~line 88) and
    `crates/vestige-mcp/src/dashboard/handlers.rs` (~line 1863) — they only
    populate the now-inert `DreamMemory.embedding`; delete those calls and the
    field.
  - `lifecycle.rs` / `search.rs` untouched per scope (search.rs `hybrid_search`
    etc. still reference `semantic_search_raw`).
  - Un-gate the `smart_ingest*` chain + merge plan/apply/undo cfg gates when
    the features are dropped (see above).
  - `knowledge_nodes.has_embedding` / `node_embeddings` /
    `embedding_profile_vectors` schema: my files no longer maintain them
    except the cfg-gated blocks noted above.

## Test status (default features)

- `cargo test -p vestige-core`: 971 passed, 0 failed (1 ignored, pre-existing).
- `cargo test -p vestige-mcp`: 759 passed, 2 failed:
  - `server::tests::test_recall_lookup_matches_search_shape`
  - `tools::backfill::tests::live_backfill_surfaces_root_cause_through_storage`
  Both are verified PRE-EXISTING on base `5fbe1df` (checked in a clean
  worktree); they are fallout from commit 7bc2516's search-tool removal and
  backfill rework, not from this branch.
- All in-scope test modules green: prediction_error (16), merge_supersede,
  dreams, sqlite::merge::exact_nomination_tests (6), mcp tools::dedup (12).

- The sibling branch's selftest + forgotten_lesson tools (16→18) are NOT in
  this base (ba94298 advertises 16 tools); catalog tests pin 16 here and pass.
  Nothing to keep yet — they arrive with that branch's merge.
- vestige-core still emits 3 clippy warnings (dead alias, doc-comment,
  while-let) — out of scope, untouched.
- Behavioral note: with keyword-only retrieval, four rare word-tier shared
  entities can outscore one code-tier entity in backfill ranking (see the
  backfill.rs test comment). That weighting lives in core.

# SCOPE-HANDOFF — build/t4-verify-harness (determinism/parity gate for the SQLite→STRATA default flip)

Branch: `build/t4-verify-harness` (base = main @ 82f0cf5). Committed locally;
NOT pushed. Adds the verification + differential harness that gates flipping
the default store from SQLite to STRATA.

## What landed

1. **`crates/strata-verify`** (standalone, own `[workspace]` table like its
   siblings): `verify_store(dir) -> VerifyReport { log_tail_ok,
   checkpoint_chain_ok, state_root_matches, gate_verdicts_rederived, gaps:
   Vec<GapKind>, duration_ms, failures }`. Every field is RECOMPUTED from raw
   bytes — `strata_kernel::verify_with_head` plus an independent explicit
   fold pass (per-event hashes, seq monotonicity, segment-versioned replay,
   state roots at every checkpoint, head anchor), gate frame density + head
   anchor, verdict rederivation under the stored `gate.policy`
   (`strata_gate::rederive_verdicts`), and the structural sweep
   (`strata_gate::sweep`). Every `VerifyFailure` Display names the offending
   seq / checkpoint log_seq. `layout.rs` defines the on-disk contract
   (`kernel.log`, `kernel.checkpoints`, `kernel.head`, `gate.log`,
   `gate.policy`, `gate.head`; framing magic `STVFv1\0\0`) the sibling
   `strata-store` crate must emit, with the writer (`write_store`) the
   harness uses until that crate merges.
2. **`vestige strata-verify <dir>`** — surgical block in
   `crates/vestige-mcp/src/bin/cli.rs` (JSON report, exit 1 on failure);
   `strata-verify` path dep added to vestige-mcp. Standalone bin
   `strata-verify` also exists in the crate.
3. **`tests/differential`** (`vestige-differential-tests`, workspace member):
   the cross-engine driver. Takes an op script (JSON `{ingest,edge,review,
   suppress}` list; `Script::from_json`), replays against BOTH engines —
   vestige-core SQLite `Storage` (real ingest / typed edges / `mark_reviewed`
   / `suppress_memory` + the real `FSRSScheduler` core as a pinned-time
   shadow fold) and strata (GateRuntime propose→gate→effect over MemLog +
   kernel v1 fold, then materialized + `verify_store`d). Asserts: same node
   COUNTS, same content digests sorted, same edge sets sorted (SQLite side
   is a store read-back), same suppressed sets, same per-card integer FSRS
   trajectory (review_count, lapse_count; phase on Hard-free scripts). FSRS
   stability/difficulty are compared as Q32.32-quantized streams RUN-OVER-RUN
   per engine (byte-identical snapshots), never cross-engine — the engines
   pin different weight tables by design. Seeded SplitMix64 fuzz ≥ 20 ops.
   Manual runner: `cargo run -p vestige-differential-tests --bin differential
   -- --seed N`.

## Root-workspace wiring (deliberate)

`crates/strata-{kernel,gate,verify}` are in the root `Cargo.toml` `exclude`
list (like fastembed-rs) so the workspace can path-depend on them WITHOUT
absorbing them; they keep their own workspace tables and lockfiles. When the
`crates/strata` integration lands, drop their `[workspace]` tables, remove
the excludes, and move them into `members`.

## Findings the flip decision should read

- `mark_reviewed` derives elapsed from the WALL CLOCK
  (`days_since_review(last_accessed)`), so (a) stored stability is not
  byte-reproducible run-over-run, and (b) under fast test cadence every
  post-first review classifies same-day and the STORE's lapse counter
  under-counts vs any pinned-time reference. The harness records this in
  `extras.store_lapses_total` instead of asserting it away. This is the
  concrete non-determinism the default flip retires.
- Known engine delta: a Hard (rating 2) review at elapsed ≥ 1 day moves a
  card to `Review` in vestige-core but preserves phase in the strata kernel.
  Phase parity is asserted only on Hard-free scripts.
- 1 SEQ = 1 day is the harness's shared time base; `elapsed_seq >= 1` keeps
  both engines off their same-day/short-term paths so lapse counting aligns.

## Test status (all green)

- `cargo test -p strata-verify` (standalone): 7 passed — clean-store verify,
  sealed-segment byte flip (names the seq), checkpoint byte flip (names
  log_seq), forged EFFECT without GATE (re-anchor-consistent; sweep flags
  `orphan_effect` naming the forged seq; rederivation still passes), tampered
  GATE verdict (rederivation names the gate seq), missing artifact, trailing
  event past head checkpoint.
- `cargo test -p vestige-differential-tests`: 5 passed — fixed-script full
  parity (incl. phase), two-runs byte-identical per engine, 3-seed fuzz
  parity + determinism, 6-seed materialize-and-verify-green, script JSON
  round-trip.
- `cargo test --workspace`: ALL PASS (exit 0), including 789-test
  vestige-mcp lib. `cargo clippy` clean for both new crates (0 warnings).
- CLI smoke: `vestige strata-verify <materialized-store>` prints the report
  and exits 0/1 by `report.ok()`.
# SCOPE-HANDOFF — T1: Storage Trait Wall (branch `build/t1-trait-wall`)

(Replaces the w1d-absorb handoff that was already merged at 18b13c7.)

Base: `main @ 82f0cf5`. Status: **complete** — workspace green, suites green,
wall proven by mock test. This is the seam STRATA (T2) implements.

## What changed

- `crates/vestige-core/src/storage/memory_store.rs`
  - `LocalMemoryStore` now carries the full product seam: **26 async methods**
    (phase-1 cognitive surface, unchanged semantics) **+ 183 sync methods**
    (the product surface every tool/server/CLI uses). The sync methods have
    loud default bodies (`Err(StorageError::Init("<name> is not implemented
    by this backend"))`, or `unimplemented!` for non-Result returns), so a
    partial backend compiles and fails loudly rather than silently
    succeeding. NOTE: default bodies on the **async** methods are NOT
    possible with `trait-variant 0.1.3` (its desugaring rejects them);
    implement the async seam explicitly (see the mock for a copy-paste
    Err-stub block).
  - The dyn-compatible `MemoryStore` trait mirrors all 209 methods
    (async boxed-future + sync passthrough); the blanket
    `impl<T: MemoryStoreSend> MemoryStore for T` forwards everything.
  - Two phase-1 async methods were renamed to free their names for the
    product-shaped sync methods: `search` -> `search_records`,
    `get_stats` -> `get_store_stats`.
- `crates/vestige-core/src/storage/sqlite/mod.rs`
  - `impl MemoryStoreSend for SqliteMemoryStore` now forwards all 183 sync
    methods to the inherent SQLite implementations (explicit
    `SqliteMemoryStore::method(self, ..)` form). SQLite is an implementation
    behind the wall, nothing more.
- `crates/vestige-core/src/storage/mod.rs`
  - `pub type Storage = dyn MemoryStore;` (was `= SqliteMemoryStore`). Every
    `Arc<Storage>` / `&Storage` site is now engine-agnostic.
  - `open_storage(path: Option<PathBuf>) -> Result<Arc<dyn MemoryStore>>`
    is the only constructor the MCP layer may call. Path helpers
    `default_db_path()` / `db_path_for_data_dir()` are free functions now.
  - New re-exports: `PurgeReport`.
- `crates/vestige-mcp` — zero `SqliteMemoryStore` references (grep-verified,
  including comments). All constructors go through `vestige_core::open_storage`.
  Core product fns that took `&SqliteMemoryStore` (`run_sync`,
  `run_dream_compile`, `walk_storage`, `persist_evidence_edges`) now sit on
  the trait surface.
- `crates/vestige-mcp/tests/backfill_trait_wall.rs` — **the wall proof**: a
  pure in-memory mock (overrides only the 7 methods backfill touches; async
  seam Err-stubbed) serves `tools::backfill::execute` end-to-end. Passing
  means no SQLite reachability from the tools layer.

## Rust coercion notes for callers (bit us repeatedly)

- `&Arc<dyn Storage>` does NOT coerce to `&dyn Storage` — pass `&*storage`.
- `&Arc<Concrete>` does NOT coerce to `&Arc<dyn Storage>` — coerce the Arc
  first, or pass `&*storage` into `&Storage` params.
- `&Concrete` DOES coerce to `&dyn Storage` (single unsize step).

## The trait surface T2 must implement

Implement `MemoryStoreSend` (you get `LocalMemoryStore` + `MemoryStore` for
free via the blanket impl). Error type: `StorageError` (`storage::Result`).
Signatures mirror `SqliteMemoryStore`'s inherent methods exactly — check
`crates/vestige-core/src/storage/memory_store.rs` (source of truth) for
exact args/returns.

### Sync product seam (183 methods, defaults provided)

actor_policy_snapshot, append_mcp_call_outcome, append_trace_event, apply_failure_feedback, apply_intention_graph, apply_tag_mutation, backup_to, blast_radius, blast_radius_with_link_types, capture_synaptic_event, checkpoint_wal, clear_dream_page_tags, closed_issue_nodes, code_anchors_for_node, code_anchors_for_nodes, commit_intention_check, concrete_search_filtered, count_memories_below_retention, count_memories_since, count_pending_memory_prs, count_suppressed, coverage_snapshot, create_context_ablation_replay, current_code_context_nodes, data_dir, db_path, decide_memory_pr, decide_pending_memory_mutation, delete_node, demote_memory, demote_memory_as_actor, dream_compile_candidates, due_for_review_node_ids, expire_stale_reconsolidation_plans, export_portable_archive, export_portable_archive_to_path, get_active_intentions, get_active_intentions_in_scope, get_agent_run, get_all_connections, get_all_nodes, get_all_nodes_in_scope, get_avg_retention, get_composition_event, get_composition_members, get_composition_neighbors, get_composition_outcomes, get_compositions_for_memory, get_connections_for_memory, get_connector_cursor, get_consolidation_history, get_context_ablation_replay, get_dream_history, get_insights, get_intention, get_intentions_by_status, get_last_consolidation, get_last_dream, get_memory_pr, get_memory_subgraph, get_merge_operation, get_merge_policy, get_most_connected_memory, get_never_composed_candidates, get_never_composed_candidates_in_scope, get_node, get_overdue_intentions, get_receipt, get_receipt_attestation_envelope, get_recent_composition_events, get_recent_composition_events_page, get_recent_connections, get_recent_state_transitions, get_retention_distribution, get_retention_trend, get_retrieval_replay_capsule, get_review_queue, get_state_transitions, get_stats, get_trace, get_walk_receipt, git_commit_nodes, grant_actor_role, hybrid_search, hybrid_search_filtered, hygiene_snapshot, import_portable_archive, import_portable_archive_from_path, ingest, ingest_in_scope, ingest_in_scope_with_secret_policy, ingest_with_secret_policy, intention_memory_snapshot, last_backup_timestamp, last_session_failed_calls, latest_receipt_chain_entry, link_receipt_to_run, list_agent_runs, list_endorsement_events, list_memory_prs, list_merge_operations, list_receipts, list_receipts_for_run, list_reconsolidation_plans, list_tag_operations, load_active_synaptic_tags, lowest_retention_nodes, maintain_gc_batch, maintain_lifecycle_batch, maintain_log_batch, maintenance_memory_page, mark_reviewed, merge_candidates, merge_undo, node_is_in_scope, open_failures_touching, preview_tag_mutation, process_actor_did, process_synaptic_ingest, projection_candidates, promote_memory, promote_memory_as_actor, promote_memory_backfill, prune_agent_traces, purge_node, query_time_range, recall, receipt_attestation_status, reconcile_source_tombstones, record_anchor_verification, record_batch_retrieval, record_code_anchors, record_composition_outcome, record_memory_access, record_reinforce_endorsement, register_receipt_signing_key, registered_receipt_signing_key, release_quarantine, replace_code_anchors, replay_intention_graph, resolve_actor_role, resolve_commit_sha_root, resolve_handle, retire_affected, reverse_suppression, run_consolidation, run_rac1_cascade_sweep, save_composition, save_connection, save_connector_cursor, save_counterfactual_replay_receipt, save_dream_history, save_insight, save_intention, save_memory_pr, save_receipt, save_retrieval_receipt_with_replay_capsule, save_signed_retrieval_receipt_with_replay_capsule_atomic, save_synaptic_tag, save_walk_receipt, schema_introspection, search, set_created_at, set_merge_policy, set_process_actor, set_protected, sidecar_dir, smart_ingest_excluding_in_scope_with_secret_policy_and_labile, snooze_intention, state_distribution, strengthen_connection, superseded_node_ids, supersession_pairs, suppress_memory, sync_portable_archive_cloud, sync_portable_archive_file, tag_vocabulary, undo_tag_mutation, update_intention_status, update_memory_state, update_node_content, upsert_by_source, verify_stored_receipt_attestation

### Async cognitive seam (26 methods, NO defaults — implement explicitly)

add_edge, classify, count, delete, delete_domain, fts_search, get, get_domain, get_due_memories, get_edges, get_neighbors, get_scheduling, get_store_stats, health_check, init, insert, list_domains, register_model, registered_model, remove_edge, search_records, update, update_scheduling, upsert_domain, vacuum, vector_search

## Verification receipts

- `cargo check --workspace --all-targets` — 0 errors.
- `cargo test -p vestige-core -p vestige-mcp` — exit 0 (core green; mcp lib
  789 passed, protocol suites 46 + 35 passed, sized-store 2 passed,
  wall mock test 1 passed).
- `cargo test -p vestige-phase-1-tests` — 19 passed (dyn-trait consumers).
- `grep -rn SqliteMemoryStore crates/vestige-mcp/` — 0 hits.

## Known non-goals / leftovers

- The empty `embeddings` feature still guards a dead
  `storage.init_embeddings()` call in `bin/cli.rs` (dead at default features
  since the vectorless build; unchanged from base).
- `tests/e2e` harness holds `Arc<Storage>` via `open_storage`; its suites
  compile but were not run here (not in scope; they drive server binaries).

# SCOPE-HANDOFF — build/wire-strata (STRATA behind the trait wall)

Branch `build/wire-strata` (base main @ 1c9ee5d, "THE SQLITE WIPE").
Committed locally; nothing pushed. Mission: the STRATA Causal Proof Engine
is the runtime store behind vestige-core's `LocalMemoryStore`/`MemoryStore`
trait wall — the default (wiped, no-legacy-sqlite) build boots on
StrataStore.

## Constructor-hook design (the direction problem, solved)

`vestige-core` must not depend on strata crates, so the engine is INJECTED:

- `vestige_core::storage::STORE_CONSTRUCTOR: Mutex<Option<StoreConstructor>>`
  (unconditional, every build) + `install_store_constructor(fn)` +
  `pub type StoreConstructor = fn(PathBuf) -> Result<Arc<dyn MemoryStore>, String>`.
- The feature-off `open_storage` twin now consults the hook (constructor
  runs with the caller's path; the constructor owns interpreting it) and
  returns `OpenStoreError::{NoBackend, Constructor}` — the old
  always-`LegacySqliteDisabled` behavior is the `NoBackend` arm; the struct
  itself stays compiled in every build.
- `strata_bridge::strata_constructor(path)` resolves the store dir:
  `$VESTIGE_STRATA_DIR` wins; a `vestige.db`-shaped/file-looking path maps
  to `<parent>/strata`; a directory maps to `<dir>/strata`.
- `vestige-mcp/src/strata_boot.rs::open_strata_storage(db_path)` installs
  the hook and opens STRATA; `main.rs` serve() routes on the MCP crate's OWN
  `legacy-sqlite` feature: off (the default) → STRATA, on → SQLite via
  `open_storage` unchanged. `bin/cli.rs` and `bin/restore.rs` install the
  hook at entry under `not(feature = "legacy-sqlite")` so genuinely
  legacy-free links construct a store instead of exiting.

CRITICAL wiring fact: workspace builds link vestige-core with
`legacy-sqlite` unified ON (the strata-migrate path dep requests
`bundled-sqlite`), so core's `open_storage` cfg alone CANNOT decide the
mode — the mcp crate's own feature is authoritative, which is why the mode
flip lives in `main.rs`. Consequence: `cargo check -p vestige-core
--no-default-features` (pure, no unifiers) is broken PRE-EXISTING at this
base (E0432s: `memory_store.rs` imports sqlite-only types ungated); the
workspace gates pass via unification. Untouched here; the trait's type
split is its own wave.

## crates/strata-bridge (new, standalone with empty [workspace])

`StrataBackend` — the `Send + Sync + 'static` handle. StrataStore v1 is
`!Send` (Rc gate-log cache, one writer per dir via `strata.lock`), so the
backend owns a DEDICATED OS THREAD that exclusively holds the store; every
trait call ships one `Send` closure and blocks on a per-call reply channel.
Jobs run under `catch_unwind` (a panicking job never kills the thread; the
caller gets a loud `StorageError::Init("... job panicked")`). Open runs
replay + checkpoint-chain verification fail-closed on the thread; a failed
open makes every later call fail loudly.

Implements `MemoryStoreSend` (blanket impls yield `LocalMemoryStore` +
dyn `MemoryStore`; `Arc<StrataBackend>` coerces to `Arc<Storage>`):

- 17 sync-seam overrides (the strata-store v1 surface):
  ingest, ingest_in_scope, get_node, get_all_nodes_in_scope,
  set_created_at, save_connection, get_connections_for_memory,
  supersession_pairs, get_never_composed_candidates_in_scope, backup_to,
  mark_reviewed, data_dir, db_path (=`<dir>/log`), sidecar_dir,
  process_actor_did (None — no actor table; set_process_actor warns),
  last_backup_timestamp (None — truthful; stops session_start's status
  hint hitting the default `unimplemented!`), resolve_handle (Unknown +
  `HANDLE_REQUIRED_DETAIL` — the type's own negative answer; stops
  recall handle-mode panics).
- 26 async-surface bodies (the trait gives no defaults there):
  truthful: init, health_check, registered_model(None), count,
  get_store_stats (node/edge counts), and the trait-sanctioned empty
  domain answers (list_domains/get_domain/classify); loud
  `MemoryStoreError::Init("... not implemented by the STRATA backend")`
  for the Uuid-keyed Phase-1 row/graph surface (insert/get/update/delete/
  search_records/fts_search/vector_search/scheduling/due/edges/neighbors),
  register_model, upsert/delete_domain, vacuum.
- Every other method (≈166 sync) keeps T1's loud-fail defaults. No stubs.

Coercions (KnowledgeNode↔NodeRecord etc.) documented in the crate docs:
no hidden clock on ingest (created_at 0 unless set_created_at/explicit
validity — SQLite's Utc::now() stamping is deliberately NOT imitated);
strength f64↔milli round-trip; `valid_from == created_at` reads as None;
VALID_FOREVER reads as None; Q32.32 card → stability/difficulty exact
dequantize; retrievability → retrieval/retention strength; storage
strength constant 1.0; sentiment/source/source_envelope dropped.
`ingest` uses `DEFAULT_MEMORY_SCOPE` ("user"). NeverComposed candidates
carry real shared_tags but zero fusion scores (strata computes none).

## Tests & verification (all live this branch)

- `cargo check --workspace --no-default-features` — 0 errors.
- `cargo check --workspace` — 0 errors.
- `cargo check --workspace --no-default-features --all-targets` — 0 errors.
- `cargo check -p vestige-mcp --features legacy-sqlite` — 0 errors.
- `cargo clippy -p strata-bridge` — clean.
- `cargo test -p vestige-mcp --test strata_boot` — 2/2 (no
  required-features; runs in the default wiped mode): boots the real
  McpServer handler (initialize + tools/list) over a StrataBackend,
  ingests one memory over the trait (mem- id, FSRS rep folded), reads it
  back, saves one `derived_from` edge, reads it (milli strength), pair
  leaves never-composed, free-form link_type rejected, unimplemented
  method loud-fails, open failure loud, close/reopen replays the log and
  both node and edge survive; plus hook-install + path-resolution test.
- `cargo test -p vestige-mcp --lib` — 712 passed, 0 failed.
- strata-bridge has NO unit tests of its own: a non-member crate with
  dev-deps cannot `cargo test -p` from the root, and its own workspace
  lacks the unification that makes core compile. Its coverage lives in
  the mcp integration test above (same pattern as this file's earlier
  strata entries: verify from the root workspace only).

## Known boundaries (documented, deliberate)

- The `memory` MCP tool uuid-gates ids, so STRATA's `mem-<hex>` ids cannot
  pass it yet; ingests reach STRATA through `smart_ingest`-adjacent store
  calls or the trait, and `smart_ingest` itself needs
  `smart_ingest_excluding_...` (loud default) — the tool-layer id/pipeline
  flip is the next wave, not storage wiring.
- `suppress`/`blast_radius` reach `retire_affected` (default
  `unimplemented!`) but fail loudly EARLIER at `suppress_memory`/
  `blast_radius` defaults, so the panic is unreachable through tools.
- CLI subcommands against a STRATA store: the store-backed ones loud-fail
  per method (v1 surface); `vestige` CLI in default builds still talks to
  unified-link SQLite unless core is genuinely compiled legacy-free.
