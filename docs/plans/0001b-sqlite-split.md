# Sub-Plan 0001b: Split sqlite.rs into a sqlite/ directory

**Status**: Draft
**Branch**: `feat/storage-trait-phase1` (Phase 1 amendment, PR A)
**Depends on**: `0001a-trait-rewrite.md` (must land first; it carries the
`trait_variant`-generated trait declaration that `trait_impl.rs` matches)
**Related**: `docs/adr/0002-phase-2-execution.md` (D3, D6)

---

## Context

`crates/vestige-core/src/storage/sqlite.rs` is the single SQLite backend file
that Phase 1 inherited from pre-trait Vestige and then appended the
`LocalMemoryStore` trait impl block to. The file is 8713 lines as of
commit 790c0c8 on `feat/storage-trait-phase1`. ADR 0002 D3 decides to split
it into a `sqlite/` directory before Phase 2 lands `postgres/` as a peer.
Reasoning, in one paragraph:

The Postgres backend is going in as a directory of seven small files
(`postgres/{mod,pool,migrations,registry,search,migrate_cli,reembed}.rs`).
If SQLite stays as one 8K-line file alongside that, the repo says "backends
look like one big file or seven small ones, pick a side", which forces
every future maintainer to re-litigate the layout. Splitting now -- as
**pure code motion**, no public-API change, no behavioural change, no
migration -- lets both backends look the same, keeps each surface mappable
in a single editor tab, and shrinks the diffs Phase 2 has to review.
This sub-plan is sized as one focused implementation session.

The split is **private** to `storage/sqlite/`. Cognitive modules continue
to `use crate::storage::SqliteMemoryStore`; the existing re-exports in
`crates/vestige-core/src/storage/mod.rs` keep working without touching
any caller. Tests stay green commit-by-commit.

This sub-plan depends on `0001a-trait-rewrite.md` landing first because
`sqlite/trait_impl.rs` is the file that picks up the new trait_variant
attribute. Doing the split first would force a second rewrite of
`trait_impl.rs` when the trait rewrite arrives. Order matters; this is
the cheap-to-respect ordering.

---

## Target Layout

Final directory after this sub-plan:

```
crates/vestige-core/src/storage/sqlite/
  mod.rs           -- module root: SqliteMemoryStore struct, new(),
                      reader/writer locks, error types, shared helpers,
                      portable-sync-related types, record types
  crud.rs          -- ingest/smart_ingest/get/update/delete/purge/search-by-id
  search.rs        -- fts, semantic, hybrid, time-based queries
  scheduling.rs    -- FSRS state, decay, consolidation, review, promote/demote,
                      suppression, gc, retention, waking tags
  graph.rs         -- memory_connections (edges), subgraph, neighbors
  domain.rs        -- domains/domain_scores column readers, classify stub
                      (Phase 4 will expand this file)
  registry.rs      -- embedding_model table, enforce_model, register_model body
  portable_sync.rs -- portable export/import/sync + merge helpers
  trait_impl.rs    -- impl LocalMemoryStore for SqliteMemoryStore
```

`storage/mod.rs` is unchanged in spirit: it still does `mod sqlite;` and
`pub use sqlite::{...};` -- the only difference is that `sqlite` is now a
directory module instead of a leaf file. The re-export list does not
change.

---

## Current File Map (line numbers from commit 790c0c8)

The current `sqlite.rs` is structurally:

| Region | Lines | Contents |
|--------|-------|----------|
| Header | 1-43 | Imports, feature-gated imports |
| Error types | 45-89 | `StorageError`, `Result`, `SmartIngestResult`, `MergeWrite` |
| Portable sync types | 97-211 | `PortableSyncBackend` trait, `FilePortableSyncBackend` struct, `PortableSyncReport`, `PurgeReport` |
| Constants | 233-273 | `PORTABLE_TABLES`, `PORTABLE_USER_DATA_TABLES`, `PortableMergeState`, env constants |
| Struct decl | 287-301 | `SqliteMemoryStore` struct fields |
| Impl block 1 | 303-3740 | Constructor + bulk of native API |
| Record structs | 3747-3866 | `IntentionRecord`, `InsightRecord`, `ConnectionRecord`, `MemoryStateRecord`, `StateTransitionRecord`, `ConsolidationHistoryRecord`, `DreamHistoryRecord`, `Default for InsightRecord` |
| Impl block 2 | 3868-6133 | Intentions / Insights / Connections / States / History / Backup / Portable / GC / Subgraph |
| Impl block 3 | 6139-6272 | Trait-helper methods (`node_to_record`, `read_domain_columns`, `enforce_model`) |
| Trait impl | 6274-7110 | `impl LocalMemoryStore for SqliteMemoryStore` |
| Tests | 7112-8713 | `#[cfg(test)] mod tests`: native API tests + trait round-trip tests |

---

## Mapping Table

Every public method, every private helper, every struct, every test module
in the current file -- with the destination file in the new layout. Line
ranges cite the current `sqlite.rs` (commit 790c0c8 on
`feat/storage-trait-phase1`, viewed through the
`/home/delandtj/prppl/vestige-phase2` worktree).

### Header and shared types (-> `sqlite/mod.rs`)

| Item | Lines | Destination | Notes |
|------|-------|-------------|-------|
| Module-level `use` imports | 1-43 | `sqlite/mod.rs` | Trimmed per-file in destination; what does not fit in `mod.rs` moves with its consumers |
| `StorageError` enum + `Result` alias | 49-71 | `sqlite/mod.rs` | Re-exported through `pub use` chain; called from every sub-module |
| `SmartIngestResult` struct | 73-89 | `sqlite/mod.rs` | Returned by `crud::smart_ingest`; defined here because other code depends on the type |
| `MergeWrite` enum | 91-95 | `sqlite/portable_sync.rs` | Only used by merge helpers |
| `PortableSyncBackend` trait | 97-109 | `sqlite/portable_sync.rs` | Public trait; re-exported through `mod.rs` |
| `FilePortableSyncBackend` struct + `impl` | 111-194 | `sqlite/portable_sync.rs` | |
| `PortableSyncReport` struct | 196-211 | `sqlite/portable_sync.rs` | |
| `PurgeReport` struct | 213-231 | `sqlite/crud.rs` | Returned by `purge_node` |
| `PORTABLE_TABLES` constant | 237-254 | `sqlite/portable_sync.rs` | |
| `PORTABLE_USER_DATA_TABLES` constant | 256-272 | `sqlite/portable_sync.rs` | |
| `PortableMergeState` struct | 274-277 | `sqlite/portable_sync.rs` | |
| `DATA_DIR_ENV` constant | 279 | `sqlite/mod.rs` | Read by constructor |
| `DATABASE_FILE` constant | 280 | `sqlite/mod.rs` | Read by constructor |
| `SqliteMemoryStore` struct decl | 282-301 | `sqlite/mod.rs` | All fields stay public-to-crate within the directory |

### Constructor and config (-> `sqlite/mod.rs`)

These are foundational; they live in `mod.rs` because every sub-module
calls them or operates on the struct they build.

| Item | Lines | Destination | Notes |
|------|-------|-------------|-------|
| `fn data_dir_from_env` | 304-313 | `sqlite/mod.rs` | private helper |
| `fn expand_tilde` | 314-332 | `sqlite/mod.rs` | private helper |
| `fn prepare_data_dir` | 333-346 | `sqlite/mod.rs` | private helper |
| `pub fn db_path_for_data_dir` | 347-355 | `sqlite/mod.rs` | |
| `pub fn default_db_path` | 356-368 | `sqlite/mod.rs` | |
| `fn configure_connection` | 369-396 | `sqlite/mod.rs` | |
| `pub fn new` | 397-457 | `sqlite/mod.rs` | The constructor |
| `pub fn db_path` | 458-462 | `sqlite/mod.rs` | |
| `pub fn data_dir` | 463-467 | `sqlite/mod.rs` | |
| `pub fn sidecar_dir` | 468-473 | `sqlite/mod.rs` | |
| `fn load_embeddings_into_index` | 474-552 | `sqlite/mod.rs` | Called by `new`; touches vector index |

### CRUD: ingest, get, update, delete, purge (-> `sqlite/crud.rs`)

| Item | Lines | Destination | Notes |
|------|-------|-------------|-------|
| `pub fn ingest` | 553-643 | `sqlite/crud.rs` | |
| `pub fn smart_ingest` | 644-864 | `sqlite/crud.rs` | Calls vector search via `self.semantic_search`; cross-module call resolved by impl block being on the same struct |
| `pub fn get_node_embedding` (vector-search) | 865-887 | `sqlite/crud.rs` | embedding read for one node |
| `pub fn get_all_embeddings` (vector-search) | 888-914 | `sqlite/crud.rs` | |
| `pub fn get_node_embedding` (no vector-search stub) | 915-919 | `sqlite/crud.rs` | feature-gated alternative |
| `pub fn update_node_content` | 920-951 | `sqlite/crud.rs` | |
| `fn generate_embedding_for_node` | 952-999 | `sqlite/crud.rs` | private helper; only called by ingest and update_node_content |
| `pub fn get_node` | 1000-1011 | `sqlite/crud.rs` | |
| `fn parse_timestamp` | 1012-1027 | `sqlite/mod.rs` | **Shared helper**: row_to_node uses it, intention/insight rows also parse timestamps. Bump to `pub(super) fn` |
| `fn row_to_node` | 1028-1119 | `sqlite/mod.rs` | **Shared helper**: crud reads single nodes; search.rs builds node lists from rows; scheduling.rs returns nodes for review queue. Bump to `pub(super) fn`. Static method (no `&self`) so a free function in `mod.rs` is fine |
| `pub fn delete_node` | 1842-1869 | `sqlite/crud.rs` | |
| `pub fn purge_node` | 1870-1987 | `sqlite/crud.rs` | |
| `fn node_exists` | 1988-1996 | `sqlite/crud.rs` | static helper, called only by purge |
| `fn record_sync_tombstone` | 1997-2014 | `sqlite/crud.rs` | static helper, called by delete and purge |
| `pub fn get_all_nodes` | 2268-2291 | `sqlite/crud.rs` | bulk read |
| `pub fn get_nodes_by_type_and_tag` | 2292-2342 | `sqlite/crud.rs` | bulk read |

### Search: fts, semantic, hybrid, temporal (-> `sqlite/search.rs`)

| Item | Lines | Destination | Notes |
|------|-------|-------------|-------|
| `pub fn recall` | 1120-1147 | `sqlite/search.rs` | top-level recall path |
| `fn keyword_search` | 1148-1180 | `sqlite/search.rs` | private |
| `pub fn search` | 2015-2043 | `sqlite/search.rs` | |
| `pub fn search_terms` | 2044-2075 | `sqlite/search.rs` | |
| `pub fn concrete_search_filtered` | 2076-2172 | `sqlite/search.rs` | |
| `fn upsert_concrete_result` | 2173-2197 | `sqlite/search.rs` | static helper |
| `fn normalize_literal_query` | 2198-2210 | `sqlite/search.rs` | static helper |
| `fn escape_like` | 2211-2224 | `sqlite/search.rs` | static helper |
| `fn literal_match_score` | 2225-2248 | `sqlite/search.rs` | static helper |
| `fn node_matches_type_filters` | 2249-2267 | `sqlite/search.rs` | static helper |
| `pub fn is_embedding_ready` (both feature variants) | 2343-2354 | `sqlite/search.rs` | both versions move together |
| `pub fn init_embeddings` (both feature variants) | 2355-2367 | `sqlite/search.rs` | both versions move together |
| `fn get_query_embedding` | 2368-2400 | `sqlite/search.rs` | private; uses `query_cache` field |
| `pub fn semantic_search` | 2401-2434 | `sqlite/search.rs` | |
| `pub fn hybrid_search` (feature on) | 2435-2452 | `sqlite/search.rs` | |
| `pub fn hybrid_search_filtered` (feature on) | 2453-2581 | `sqlite/search.rs` | |
| `pub fn hybrid_search` (feature off) | 2582-2593 | `sqlite/search.rs` | feature-gated stub |
| `pub fn hybrid_search_filtered` (feature off) | 2594-2635 | `sqlite/search.rs` | feature-gated stub |
| `fn keyword_search_with_scores` | 2636-2726 | `sqlite/search.rs` | |
| `fn semantic_search_raw` | 2727-2765 | `sqlite/search.rs` | |
| `pub fn generate_embeddings` | 2766-2819 | `sqlite/search.rs` | populates embeddings post hoc |
| `fn embedding_regeneration_candidates` | 2820-2891 | `sqlite/search.rs` | called by generate_embeddings |
| `pub fn query_at_time` | 2892-2933 | `sqlite/search.rs` | temporal query |
| `pub fn query_time_range` | 2934-3005 | `sqlite/search.rs` | temporal query |
| `fn embedding_model_matches_active` (associated fn) | 5652-5671 | `sqlite/search.rs` | static helper for hybrid_search; promoted to `pub(super)` (test references it) |
| `fn embedding_model_supports_matryoshka` | 5672-5677 | `sqlite/search.rs` | static helper |
| `fn embedding_vector_for_active_model` | 5678-5697 | `sqlite/search.rs` | static helper |
| `fn active_embedding_model_like_pattern` | 5698-5713 | `sqlite/search.rs` | static helper |

### Scheduling: FSRS, decay, consolidation, review, promote/demote, suppression, gc, retention (-> `sqlite/scheduling.rs`)

This is the busiest destination file. The grouping rule is: anything that
touches FSRS scheduling fields (`stability`, `difficulty`, `retrievability`,
`reps`, `lapses`, `retention_strength`, `retrieval_strength`) or the
review/decay/consolidation/gc lifecycle lives here.

| Item | Lines | Destination | Notes |
|------|-------|-------------|-------|
| `pub fn mark_reviewed` | 1181-1275 | `sqlite/scheduling.rs` | FSRS state mutation |
| `pub fn strengthen_on_access` | 1276-1344 | `sqlite/scheduling.rs` | |
| `pub fn strengthen_batch_on_access` | 1345-1357 | `sqlite/scheduling.rs` | |
| `pub fn mark_memory_useful` | 1358-1377 | `sqlite/scheduling.rs` | |
| `fn log_access` | 1378-1393 | `sqlite/scheduling.rs` | private |
| `pub fn promote_memory` | 1394-1425 | `sqlite/scheduling.rs` | |
| `pub fn demote_memory` | 1426-1472 | `sqlite/scheduling.rs` | |
| `pub fn suppress_memory` | 1473-1504 | `sqlite/scheduling.rs` | active forgetting |
| `pub fn reverse_suppression` | 1505-1552 | `sqlite/scheduling.rs` | |
| `pub fn count_suppressed` | 1553-1567 | `sqlite/scheduling.rs` | |
| `pub fn get_recently_suppressed` | 1568-1594 | `sqlite/scheduling.rs` | |
| `pub fn apply_rac1_cascade` | 1595-1641 | `sqlite/scheduling.rs` | active forgetting cascade |
| `pub fn run_rac1_cascade_sweep` | 1642-1657 | `sqlite/scheduling.rs` | |
| `pub fn get_review_queue` | 1658-1681 | `sqlite/scheduling.rs` | |
| `pub fn preview_review` | 1682-1712 | `sqlite/scheduling.rs` | |
| `pub fn get_stats` | 1713-1841 | `sqlite/scheduling.rs` | reports retention/lapses/review counts; lives here for symmetry with the FSRS reporters next door |
| `pub fn apply_decay` | 3006-3095 | `sqlite/scheduling.rs` | core decay loop |
| `fn get_fsrs_w20` | 3096-3119 | `sqlite/scheduling.rs` | |
| `pub fn run_consolidation` | 3120-3407 | `sqlite/scheduling.rs` | |
| `fn auto_dedup_consolidation` | 3408-3538 | `sqlite/scheduling.rs` | called by run_consolidation |
| `fn compute_act_r_activations` | 3539-3605 | `sqlite/scheduling.rs` | called by run_consolidation |
| `fn prune_access_log` | 3606-3620 | `sqlite/scheduling.rs` | called by run_consolidation |
| `fn optimize_w20_if_ready` | 3621-3720 | `sqlite/scheduling.rs` | called by run_consolidation |
| `fn generate_missing_embeddings` | 3721-3740 | `sqlite/scheduling.rs` | called by run_consolidation |
| `pub fn get_state_transitions` | 5714-5748 | `sqlite/scheduling.rs` | audit trail tied to scheduling state |
| `pub fn get_avg_retention` | 5780-5792 | `sqlite/scheduling.rs` | |
| `pub fn get_retention_distribution` | 5794-5825 | `sqlite/scheduling.rs` | |
| `pub fn get_retention_trend` | 5826-5858 | `sqlite/scheduling.rs` | |
| `pub fn save_retention_snapshot` | 5859-5878 | `sqlite/scheduling.rs` | |
| `pub fn count_memories_below_retention` | 5879-5892 | `sqlite/scheduling.rs` | |
| `pub fn gc_below_retention` | 5893-5936 | `sqlite/scheduling.rs` | |
| `pub fn auto_promote_frequent_access` | 5937-5985 | `sqlite/scheduling.rs` | |
| `pub fn set_waking_tag` | 5986-5998 | `sqlite/scheduling.rs` | waking SWR tagging |
| `pub fn clear_waking_tags` | 5999-6011 | `sqlite/scheduling.rs` | |
| `pub fn get_waking_tagged_memories` | 6012-6028 | `sqlite/scheduling.rs` | |
| `pub fn get_recent_state_transitions` | 6105-6132 | `sqlite/scheduling.rs` | |

### Graph: edges (memory_connections), neighbors, subgraph (-> `sqlite/graph.rs`)

| Item | Lines | Destination | Notes |
|------|-------|-------------|-------|
| `pub fn save_connection` | 4180-4202 | `sqlite/graph.rs` | |
| `pub fn get_connections_for_memory` | 4203-4220 | `sqlite/graph.rs` | |
| `pub fn get_all_connections` | 4221-4236 | `sqlite/graph.rs` | |
| `pub fn strengthen_connection` | 4237-4259 | `sqlite/graph.rs` | |
| `pub fn apply_connection_decay` | 4260-4272 | `sqlite/graph.rs` | |
| `pub fn prune_weak_connections` | 4273-4284 | `sqlite/graph.rs` | |
| `fn row_to_connection` | 4285-4305 | `sqlite/graph.rs` | private |
| `pub fn get_most_connected_memory` | 6029-6046 | `sqlite/graph.rs` | |
| `pub fn get_memory_subgraph` | 6048-6103 | `sqlite/graph.rs` | calls `get_connections_for_memory`, `get_node`, `get_all_connections` -- all resolvable through `self` |

### Domain (-> `sqlite/domain.rs`)

Phase 1 keeps domain logic to JSON-column reads + classify stub. Phase 4
expands this file. Keeping the file in the split so Phase 4 has an
obvious place to add to.

| Item | Lines | Destination | Notes |
|------|-------|-------------|-------|
| `fn read_domain_columns` | 6167-6196 | `sqlite/domain.rs` | private helper used by trait `get`. Bump to `pub(super)` |

The trait methods `list_domains` / `get_domain` / `upsert_domain` /
`delete_domain` / `classify` live in `sqlite/trait_impl.rs`; they
delegate to thin helpers that, in Phase 1, are essentially noops or
JSON reads. Phase 4 will move the substance of those methods into
`sqlite/domain.rs` as real functions.

### Registry: embedding_model table (-> `sqlite/registry.rs`)

| Item | Lines | Destination | Notes |
|------|-------|-------------|-------|
| `fn enforce_model` | 6203-6272 | `sqlite/registry.rs` | private helper used by trait `insert` and `update`. Bump to `pub(super)` |

The trait methods `registered_model` and `register_model` themselves
live in `sqlite/trait_impl.rs`. Phase 2's `postgres/registry.rs` will
mirror this layout.

### Intentions, Insights, Memory States, Consolidation History, Dream History, Backup (-> `sqlite/mod.rs`)

These were tacked onto `SqliteMemoryStore` over time as the cognitive
modules needed somewhere to persist their state. They are not part of the
trait surface, they are not naturally grouped with crud/search/scheduling,
and they are each fairly small and self-contained. They live in `mod.rs`
under labelled sections (one big impl block can carry them) rather than
inventing extra files like `intentions.rs` and `insights.rs`. Postgres
will mirror this once Phase 5 picks up the work; for now they have no
peer.

Rationale: every one of these methods writes to a single table, the
bodies are short, and grouping them next to the constructor preserves
"open `mod.rs` to see the whole struct" as the navigation default.

| Item | Lines | Destination | Notes |
|------|-------|-------------|-------|
| `IntentionRecord` struct | 3747-3766 | `sqlite/mod.rs` | re-exported through `storage/mod.rs` |
| `InsightRecord` struct + `Default` | 3767-3797 | `sqlite/mod.rs` | re-exported |
| `ConnectionRecord` struct | 3799-3809 | `sqlite/mod.rs` | re-exported; consumed by `graph.rs` |
| `MemoryStateRecord` struct | 3811-3821 | `sqlite/mod.rs` | |
| `StateTransitionRecord` struct | 3823-3833 | `sqlite/mod.rs` | re-exported |
| `ConsolidationHistoryRecord` struct | 3835-3846 | `sqlite/mod.rs` | |
| `DreamHistoryRecord` struct | 3848-3866 | `sqlite/mod.rs` | re-exported |
| `pub fn save_intention` etc. (intention block) | 3874-4058 | `sqlite/mod.rs` | one impl block, in-section labelled |
| `fn row_to_intention` | 4023-4058 | `sqlite/mod.rs` | private |
| insights block (`save_insight`, `get_insights`, etc.) | 4065-4174 | `sqlite/mod.rs` | |
| `fn row_to_insight` | 4153-4173 | `sqlite/mod.rs` | private |
| memory-state block | 4306-4459 | `sqlite/mod.rs` | |
| `fn row_to_memory_state` | 4431-4459 | `sqlite/mod.rs` | private |
| consolidation-history block | 4465-4540 | `sqlite/mod.rs` | |
| dream-history block | 4546-4638 | `sqlite/mod.rs` | |
| `pub fn count_memories_since` | 4639-4651 | `sqlite/mod.rs` | |
| `fn scan_last_backup_timestamp` | 4652-4682 | `sqlite/mod.rs` | |
| `pub fn last_backup_timestamp` | 4683-4688 | `sqlite/mod.rs` | |
| `pub fn get_last_backup_timestamp` (associated) | 4689-4696 | `sqlite/mod.rs` | |
| `pub fn backup_to` | 5749-5774 | `sqlite/mod.rs` | sqlite VACUUM INTO; called by backup tool |

### Portable export/import/sync (-> `sqlite/portable_sync.rs`)

This is the second-largest destination after `scheduling.rs` and the most
self-contained.

| Item | Lines | Destination | Notes |
|------|-------|-------------|-------|
| `pub fn export_portable_archive` | 4699-4755 | `sqlite/portable_sync.rs` | |
| `pub fn export_portable_archive_to_path` | 4756-4806 | `sqlite/portable_sync.rs` | |
| `pub fn import_portable_archive` | 4807-4978 | `sqlite/portable_sync.rs` | |
| `pub fn import_portable_archive_from_path` | 4979-4996 | `sqlite/portable_sync.rs` | |
| `pub fn sync_portable_archive` (generic over backend) | 4997-5025 | `sqlite/portable_sync.rs` | |
| `pub fn sync_portable_archive_file` | 5026-5030 | `sqlite/portable_sync.rs` | |
| `fn merge_portable_table` | 5031-5073 | `sqlite/portable_sync.rs` | |
| `fn merge_knowledge_nodes` | 5074-5126 | `sqlite/portable_sync.rs` | |
| `fn merge_sync_tombstones` | 5127-5204 | `sqlite/portable_sync.rs` | |
| `fn merge_deletion_tombstones` | 5205-5245 | `sqlite/portable_sync.rs` | |
| `fn merge_keyed_table` | 5246-5281 | `sqlite/portable_sync.rs` | |
| `fn row_references_locally_newer_node` | 5282-5302 | `sqlite/portable_sync.rs` | |
| `fn merge_append_only_table` | 5303-5338 | `sqlite/portable_sync.rs` | |
| `fn parent_rows_exist` | 5339-5370 | `sqlite/portable_sync.rs` | |
| `fn insert_or_replace_row` | 5371-5386 | `sqlite/portable_sync.rs` | |
| `fn merge_key_columns` | 5387-5398 | `sqlite/portable_sync.rs` | |
| `fn upsert_row_with_columns` | 5399-5446 | `sqlite/portable_sync.rs` | |
| `fn insert_row_with_columns` | 5447-5469 | `sqlite/portable_sync.rs` | |
| `fn merge_row_exists` | 5470-5487 | `sqlite/portable_sync.rs` | |
| `fn row_exists_by_values` | 5488-5507 | `sqlite/portable_sync.rs` | |
| `fn row_values_for_columns` | 5508-5528 | `sqlite/portable_sync.rs` | |
| `fn portable_value` | 5529-5540 | `sqlite/portable_sync.rs` | |
| `fn portable_text` | 5541-5551 | `sqlite/portable_sync.rs` | |
| `fn portable_timestamp` | 5552-5559 | `sqlite/portable_sync.rs` | |
| `fn parse_rfc3339_opt` | 5560-5565 | `sqlite/portable_sync.rs` | |
| `fn tombstone_timestamp` | 5566-5580 | `sqlite/portable_sync.rs` | |
| `fn current_schema_version` | 5581-5589 | `sqlite/portable_sync.rs` | static helper |
| `fn ensure_portable_import_target_empty` | 5590-5604 | `sqlite/portable_sync.rs` | static helper |
| `fn table_exists` | 5605-5613 | `sqlite/portable_sync.rs` | static helper |
| `fn table_row_count` | 5614-5618 | `sqlite/portable_sync.rs` | static helper |
| `fn table_columns` | 5619-5630 | `sqlite/portable_sync.rs` | static helper |
| `fn portable_value_from_ref` | 5631-5646 | `sqlite/portable_sync.rs` | static helper |
| `fn quote_ident` | 5647-5651 | `sqlite/portable_sync.rs` | static helper |

### Trait helpers and trait impl (-> `sqlite/trait_impl.rs`)

| Item | Lines | Destination | Notes |
|------|-------|-------------|-------|
| `fn node_to_record` | 6142-6164 | `sqlite/trait_impl.rs` | associated fn used only by trait body; co-locate |
| `impl LocalMemoryStore for SqliteMemoryStore` block | 6274-7110 | `sqlite/trait_impl.rs` | full trait impl; attribute changes from `#[async_trait::async_trait]` to whatever 0001a settles on (`#[trait_variant::make(...)]` is on the trait declaration; the impl block carries no attribute under trait_variant) |

### Tests

The current `#[cfg(test)] mod tests` block at lines 7112-8713 contains
**two** distinct test families:

1. **Native API tests** (7120-8198): unit tests against the legacy
   `pub fn` surface (`test_ingest_and_get`, `test_search`, `test_review`,
   `test_delete`, `test_dream_history_save_and_get_last`,
   `test_portable_archive_exact_round_trip`, `test_keyword_search_*`,
   `test_concrete_search_*`, `test_purge_*`, etc.).
2. **Trait round-trip tests** (8200-8712, after the
   `// ===== Phase 1: LocalMemoryStore trait round-trip tests =====`
   banner): `trait_init_is_idempotent`, `trait_register_model_*`,
   `trait_insert_*`, `trait_get_*`, `trait_update_*`, `trait_delete_*`,
   `trait_fts_search_*`, `trait_hybrid_search_*`,
   `trait_scheduling_*`, `trait_add_edge_*`, `trait_get_edges_*`,
   `trait_remove_edge_*`, `trait_get_neighbors_*`, `trait_list_domains_*`,
   `trait_upsert_*`, `trait_classify_*`, `trait_count_*`,
   `trait_get_stats_*`, `trait_vacuum_*`,
   `trait_insert_refuses_dimension_mismatch`.

See the Test Relocation section below for the resolution.

---

## Visibility Changes

The split moves items into sibling files inside one module. Helpers that
were `fn ...` (i.e. crate-private but file-private under the current
layout, since the file *is* the module) need their visibility lifted
just enough that sibling files can call them. The principle is: smallest
bump that makes the call site compile.

`pub(super)` is sufficient for everything below; nothing needs
`pub(crate)`. The trait `LocalMemoryStore` exposure does not change --
sub-modules call `self.method(...)` on `SqliteMemoryStore`, which
resolves through the impl blocks defined in their own files; visibility
is automatic at impl-block scope.

Items that need a visibility bump (currently private fn, become
`pub(super) fn`):

- `parse_timestamp` (1012): called by `row_to_node` and by intention /
  insight row mappers.
- `row_to_node` (1028): called by `crud.rs`, `search.rs`,
  `scheduling.rs`. Static associated fn.
- `read_domain_columns` (6167): called by `trait_impl.rs`.
- `enforce_model` (6203): called by `trait_impl.rs`.
- `embedding_model_matches_active` (5652): currently called by
  `hybrid_search_filtered`; tests also reference it. Has to remain
  `pub(super) fn` and be `pub` only if the existing test names reach it
  through a re-export. (See Test Relocation.)
- `embedding_model_supports_matryoshka` (5672): private; only callers in
  `search.rs` after the move; stays `fn` (no bump needed).
- `embedding_vector_for_active_model` (5678): same as the matches
  function -- a test references it. Bump to `pub(super)`.
- `active_embedding_model_like_pattern` (5698): private; only used by
  search; stays `fn`.
- `generate_embedding_for_node` (952): currently called by `ingest` and
  `update_node_content`. Both move to `crud.rs`; stays `fn`.
- `get_query_embedding` (2368): only used inside `search.rs`; stays `fn`.
- `keyword_search_with_scores` (2636): only used inside `search.rs`;
  stays `fn`.
- `semantic_search_raw` (2727): only used inside `search.rs`; stays `fn`.
- `embedding_regeneration_candidates` (2820): used by
  `generate_embeddings`; both move to `search.rs`; stays `fn`. The
  existing test (line 7167) references it through `storage.method()`,
  which will continue to work because the test file can move with it.
- `log_access` (1378): private to `scheduling.rs`; stays `fn`.
- All the `auto_dedup_consolidation` / `compute_act_r_activations` /
  `prune_access_log` / `optimize_w20_if_ready` /
  `generate_missing_embeddings` helpers (3408-3740): private to
  `scheduling.rs`; stays `fn`.
- `row_to_intention` / `row_to_insight` / `row_to_memory_state` /
  `row_to_connection`: all stay private in their destination file (only
  one caller each).
- All `merge_*` / `portable_*` / `parse_rfc3339_opt` / `quote_ident`:
  private to `portable_sync.rs`; stays `fn`.
- `node_exists` (1988): private to `crud.rs`; stays `fn`.
- `record_sync_tombstone` (1997): private to `crud.rs`; stays `fn`.
- `get_fsrs_w20` (3096): private to `scheduling.rs`; stays `fn`.

Items already `pub fn` (or `pub(crate) fn`) stay as they are -- no
visibility regression.

Field visibility on `SqliteMemoryStore` itself: currently all fields are
private. The sub-modules access them via `self.field`. Because impl
blocks for `SqliteMemoryStore` are written in sibling files of the same
module, `self.field` reaches private fields without a visibility bump.
**No field visibility changes are required.** Confirm this during the
first motion commit; if Rust disagrees, mark the relevant fields
`pub(super)` and document in the commit message.

---

## Public Re-exports

`crates/vestige-core/src/storage/mod.rs` currently exports:

```rust
mod memory_store;
mod migrations;
mod portable;
mod sqlite;

pub use memory_store::{...};
pub use migrations::MIGRATIONS;
pub use portable::{...};
pub use sqlite::{
    ConnectionRecord, ConsolidationHistoryRecord, DreamHistoryRecord, FilePortableSyncBackend,
    InsightRecord, IntentionRecord, PortableSyncBackend, PortableSyncReport, Result,
    SmartIngestResult, SqliteMemoryStore, StateTransitionRecord, StorageError,
};

pub type Storage = SqliteMemoryStore;
```

After the split, `mod sqlite;` resolves to the new directory module
(`storage/sqlite/mod.rs`). The `pub use sqlite::{...}` block resolves
against the items re-exported by `storage/sqlite/mod.rs`.

`storage/sqlite/mod.rs` therefore needs the same names visible at its
top level. Add at the end of `mod.rs`:

```rust
mod crud;
mod search;
mod scheduling;
mod graph;
mod domain;
mod registry;
mod portable_sync;
mod trait_impl;

pub use portable_sync::{FilePortableSyncBackend, PortableSyncBackend, PortableSyncReport};
// SqliteMemoryStore, StorageError, Result, SmartIngestResult, IntentionRecord,
// InsightRecord, ConnectionRecord, StateTransitionRecord,
// ConsolidationHistoryRecord, DreamHistoryRecord are defined in mod.rs itself,
// so they are already in scope and do not need a re-export.
```

The `crates/vestige-core/src/storage/mod.rs` file does not change. The
`pub type Storage = SqliteMemoryStore;` alias keeps working.

If `cargo build` complains that `storage/mod.rs` cannot resolve a name
in its `pub use sqlite::{...}` block, the fix is to add the missing name
to `sqlite/mod.rs`'s re-export tail; no change to `storage/mod.rs`.

---

## Test Relocation

Two test families, two destinations.

**Native API tests** (current lines 7120-8198) cover the legacy `pub fn`
surface. They live close to their subject:

- Tests that touch the constructor, common helpers, and shared setup
  (`create_test_storage`, `create_test_storage_at`,
  `test_storage_creation`, `test_get_last_backup_timestamp_no_panic`)
  move to `sqlite/mod.rs` in a `#[cfg(test)] mod tests` block.
- `test_ingest_and_get`, `test_delete`,
  `test_purge_scrubs_insight_json_orphans_children_and_writes_tombstone`
  go to `sqlite/crud.rs` as a `#[cfg(test)] mod tests` block.
- `test_search`, `test_keyword_search_with_include_types`,
  `test_keyword_search_with_exclude_types`,
  `test_include_types_takes_precedence_over_exclude`,
  `test_type_filter_with_no_matches_returns_empty`,
  `test_hybrid_search_backward_compat`,
  `test_concrete_search_literal_identifier_lands_first`,
  `test_embedding_model_family_matching`,
  `test_embedding_regeneration_candidates_include_entire_mismatched_corpus`
  go to `sqlite/search.rs`.
- `test_review` goes to `sqlite/scheduling.rs`.
- `test_dream_history_save_and_get_last`, `test_dream_history_empty`,
  `test_count_memories_since` go to `sqlite/mod.rs` (history tables live
  there).
- All `test_portable_*` go to `sqlite/portable_sync.rs`.
- `test_file_portable_sync_round_trips_between_devices` goes to
  `sqlite/portable_sync.rs`.

**Trait round-trip tests** (current lines 8200-8712) test the
`LocalMemoryStore` trait impl. Two viable layouts:

A. Co-locate with the impl in `sqlite/trait_impl.rs` (one big
   `#[cfg(test)] mod trait_tests`).
B. Keep them as a single `tests.rs` file in the sqlite directory.

**Decision: A.** Co-locate. The trait round-trip tests are explicitly
testing the `impl LocalMemoryStore for SqliteMemoryStore` block;
co-location means a reader who edits the trait impl sees its tests in
the same file. Option B would mean two places to look every time a
trait method changes shape. For an 8K-line collapse the tradeoff
favours co-location.

Concretely: `sqlite/trait_impl.rs` ends with a
`#[cfg(test)] mod trait_tests { ... }` block that contains all 30+
`trait_*` tests, plus the shared `make_record`, `rt`, and small helpers
defined inside the current test mod for trait tests (lines 8208-8226).

---

## Commit Sequence

Each commit moves one logical group. After each commit:

```
cargo build -p vestige-core
cargo test  -p vestige-core
cargo clippy -p vestige-core -- -D warnings
```

must pass. Order is chosen so that each move is small, the next move
does not depend on the previous having grown surprising visibility, and
the largest / riskiest move (the trait impl, with the new
trait_variant attribute) lands last.

| # | Commit | What moves | Tests touched |
|---|--------|-----------|----------------|
| 1 | `refactor(sqlite): scaffold sqlite/ directory` | Convert `sqlite.rs` -> `sqlite/mod.rs` verbatim (rename + create empty sibling files `crud.rs`, `search.rs`, `scheduling.rs`, `graph.rs`, `domain.rs`, `registry.rs`, `portable_sync.rs`, `trait_impl.rs` each with `use super::*;`). At this point `mod.rs` declares the new modules but they are empty. | None move. Build proves the rename works. |
| 2 | `refactor(sqlite): split out portable sync` | Move all `merge_*`, `portable_*`, `export_*`, `import_*`, `sync_*` items + `MergeWrite`, `PortableSyncBackend`, `FilePortableSyncBackend`, `PortableSyncReport`, `PortableMergeState`, `PORTABLE_TABLES`, `PORTABLE_USER_DATA_TABLES`, helper statics into `sqlite/portable_sync.rs`. Add `pub use portable_sync::{...}` in `mod.rs` for the public types. | `test_portable_*` and `test_file_portable_sync_round_trips_between_devices` move too. |
| 3 | `refactor(sqlite): split out graph / connections` | Move `save_connection`, `get_connections_for_memory`, `get_all_connections`, `strengthen_connection`, `apply_connection_decay`, `prune_weak_connections`, `row_to_connection`, `get_most_connected_memory`, `get_memory_subgraph` to `sqlite/graph.rs`. | None move (no native graph tests; trait edge tests still in trait_tests). |
| 4 | `refactor(sqlite): split out scheduling / fsrs / consolidation` | Move all items listed in the Scheduling row to `sqlite/scheduling.rs`. | `test_review` moves. |
| 5 | `refactor(sqlite): split out search / fts / semantic / hybrid` | Move all items listed in the Search row to `sqlite/search.rs`. Add `pub(super)` to the four `embedding_model_*` helpers that tests reference. | `test_search`, `test_keyword_search_*`, `test_include_types_*`, `test_type_filter_*`, `test_hybrid_search_*`, `test_concrete_search_*`, `test_embedding_model_family_matching`, `test_embedding_regeneration_candidates_include_entire_mismatched_corpus` move. |
| 6 | `refactor(sqlite): split out crud / ingest / get / update / delete / purge` | Move `ingest`, `smart_ingest`, `get_node`, `update_node_content`, `delete_node`, `purge_node`, `get_all_nodes`, `get_nodes_by_type_and_tag`, `node_exists`, `record_sync_tombstone`, `generate_embedding_for_node`, `get_node_embedding`, `get_all_embeddings`, `PurgeReport` to `sqlite/crud.rs`. Bump `row_to_node` and `parse_timestamp` to `pub(super) fn` in `mod.rs`. | `test_ingest_and_get`, `test_delete`, `test_purge_scrubs_insight_json_orphans_children_and_writes_tombstone` move. |
| 7 | `refactor(sqlite): split out registry helper` | Move `enforce_model` to `sqlite/registry.rs`, bumped to `pub(super)`. | None move. |
| 8 | `refactor(sqlite): split out domain helper` | Move `read_domain_columns` to `sqlite/domain.rs`, bumped to `pub(super)`. | None move. |
| 9 | `refactor(sqlite): split out trait impl + tests` | Move `node_to_record` and the full `impl LocalMemoryStore for SqliteMemoryStore` block to `sqlite/trait_impl.rs`. Move the entire trait round-trip test module (lines 8200-8712, including `make_record` and `rt` helpers) to a `#[cfg(test)] mod trait_tests` block at the bottom of `trait_impl.rs`. This is the commit where the trait_variant attribute (from sub-plan 0001a) is observed: the impl block on `SqliteMemoryStore` uses whatever syntax the rewritten trait expects (no `#[async_trait::async_trait]`). | All `trait_*` tests move. |

Commit 1 is the only commit that creates new files; the rest move
existing code into them. Reviewers can bisect through this list to
find any silent-semantic change.

---

## Verification

Run after every commit. All three must pass before pushing:

```
cargo build -p vestige-core
cargo test  -p vestige-core
cargo clippy -p vestige-core -- -D warnings
```

The Phase 1 amendment branch must also build with the no-default-features
configuration that the release binary uses for the alternative feature
set:

```
cargo build -p vestige-core --no-default-features
cargo test  -p vestige-core --no-default-features
```

Some of the methods being moved (`get_node_embedding`, `is_embedding_ready`,
`init_embeddings`, the feature-on/feature-off `hybrid_search` pair) have
two definitions guarded by feature flags. The split must preserve both
copies in the same destination file with their existing `#[cfg(...)]`
attributes; the no-default-features build confirms.

After the last commit, run the workspace-wide check that Phase 1 promised:

```
cargo build --workspace
cargo test  --workspace
```

This catches downstream consumers (`vestige-mcp`, `vestige`,
`vestige-restore`) that might depend on a specific module path (they
should not -- they import from `crate::storage::SqliteMemoryStore` and
the re-exports in `storage/mod.rs` -- but the workspace build is the
authoritative confirmation).

---

## Acceptance Criteria

1. `crates/vestige-core/src/storage/sqlite.rs` no longer exists. In its
   place is `crates/vestige-core/src/storage/sqlite/` with the eight
   files listed in the Target Layout section, each below 2000 lines.
2. `crates/vestige-core/src/storage/mod.rs` is unchanged (or
   functionally unchanged -- the `pub use sqlite::{...}` block contains
   the same identifiers in the same order).
3. Every cognitive module and binary in the workspace
   (`vestige-core`, `vestige-mcp`, `vestige`, `vestige-restore`)
   compiles with no source edits other than the ones in
   `crates/vestige-core/src/storage/sqlite/`.
4. `cargo build -p vestige-core`,
   `cargo test  -p vestige-core`,
   `cargo clippy -p vestige-core -- -D warnings`,
   `cargo build -p vestige-core --no-default-features`, and
   `cargo test  -p vestige-core --no-default-features` all pass at the
   end of every commit in the sequence (bisectability).
5. `cargo test --workspace` matches the Phase 1 baseline test count
   (758 tests, of which 352 are in `vestige-core`). No new tests are
   added by this sub-plan; no existing test is renamed or deleted.
6. The on-disk SQLite schema is unchanged. A live database created on
   the pre-split build opens cleanly on the post-split build and round
   trips a memory.
7. `git log --follow` works for at least one method in each destination
   file (i.e. `git mv` was used where the line range constitutes most
   of the file content of the destination, otherwise a `git log -p`
   on the new file shows the history before the rename through the
   block-move detection that recent `git log` versions support).
8. No public symbol disappears from `crate::storage::*`. A reviewer can
   verify with:
   ```
   cargo doc -p vestige-core --no-deps
   ```
   before and after the split, and `diff` the generated
   `target/doc/vestige_core/storage/index.html` lists.

---

## Non-Goals (explicit)

- No public API change. The trait surface (`LocalMemoryStore`,
  `MemoryStore`), the legacy `pub fn` surface on `SqliteMemoryStore`,
  the re-exports through `storage/mod.rs`, and the `pub type Storage =
  SqliteMemoryStore;` alias are all preserved.
- No behavioural change. No SQL is rewritten, no FSRS parameter is
  retuned, no embedding model is touched, no migration is added.
- No new tests. Tests move with their subject; no new tests are
  written.
- No clippy fix-ups that pre-date this sub-plan. If `cargo clippy
  -- -D warnings` was passing before the split, it must continue to
  pass; if it was not passing, the failures stay where they are and
  are addressed in a separate commit (out of scope here).
- No removal of the `pub type Storage = SqliteMemoryStore;` BC alias.
  That happens at the end of Phase 4 per ADR 0001.
- No reorganisation of `storage/memory_store.rs`,
  `storage/migrations.rs`, or `storage/portable.rs`. Those files are
  out of scope; the split is private to `storage/sqlite/`.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Silent semantic change introduced by a motion commit | Per-commit `cargo test -p vestige-core` keeps the bisect window to a single commit. Reviewer bisects with `cargo test -p vestige-core` as the witness. |
| Sibling-file `self.field` accesses fail because Rust enforces module visibility on tuple-struct or named fields | Confirmed: associated impl blocks in sibling files of the same `mod sqlite` reach private fields. If the compiler disagrees, bump the affected fields to `pub(super)` in `mod.rs` and note the bump in the commit message. |
| Test-only helpers (`create_test_storage`, `make_record`, `rt`) get duplicated across test modules | Lift them once into a `#[cfg(test)] mod test_support { ... }` sub-module in `sqlite/mod.rs` and re-export with `pub(super) use`. Do this in commit 1 (scaffold), not later. |
| `#[cfg(all(feature = "embeddings", feature = "vector-search"))]` items end up in `mod.rs` where they pollute the shared layer | Audit during commit 5 (search split); items behind both feature flags belong in `search.rs`. The `query_cache` field stays in `mod.rs` because the struct definition is there; the field declaration is feature-gated and that gate moves with the struct as-is. |
| `git log --follow` blame chains break on the moved methods | Use `git mv sqlite.rs sqlite/mod.rs` in commit 1 so commit 1 looks like a rename (`git log --follow` keeps working). Subsequent commits are content moves inside the module; modern `git log --follow -M -C` heuristics still trace the lines. Reviewers who need pristine blame should bisect to before commit 1. |
| Sub-plan 0001a (trait rewrite) has not landed when this work starts | Block: do not start commits 1-9 until 0001a is on the same branch (`feat/storage-trait-phase1`) and tests pass. `trait_impl.rs` lands the new attribute in commit 9; if 0001a is not in, commit 9 fails. |

---

## Self-Contained Brief (for /goal)

A fresh Claude Code session can execute this sub-plan by:

1. Reading this file end to end.
2. Reading `crates/vestige-core/src/storage/sqlite.rs` (the file to be
   split) in full, using line ranges from the Mapping Table to confirm
   the current shape matches the brief.
3. Reading `crates/vestige-core/src/storage/mod.rs` (the re-export
   surface that must continue to work).
4. Reading `crates/vestige-core/src/storage/memory_store.rs` (the
   trait surface that `trait_impl.rs` implements).
5. Confirming sub-plan 0001a has landed on the current branch by
   checking that `memory_store.rs` no longer carries
   `#[async_trait::async_trait]` on the trait declaration.
6. Working through the Commit Sequence in order, running the
   Verification commands after each commit.

The session does not need to read ADR 0002 or the master Phase 2 plan
to do this work. The split is purely mechanical relative to the
mapping table above.
