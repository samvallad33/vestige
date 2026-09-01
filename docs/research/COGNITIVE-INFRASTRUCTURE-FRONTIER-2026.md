# Cognitive Infrastructure Frontier 2026: What Vestige Builds Next

## Methodology

This roadmap comes from a two-stage research pass. Stage one was a 5-agent codebase mapping exercise that walked every module under `crates/vestige-core` and `crates/vestige-mcp`, cross-checked against `apps/dashboard/src`, and classified 38 concrete features across 5 clusters: Neuroscience-Inspired Memory Mechanisms, Advanced Cognition (Backfill/Reconsolidation/Prediction/Speculation/Dreams), MCP Tool Surface Thinness + Dashboard Parity, Multi-Agent Fleet Coordination Surface, and Storage/Sync Layer Maturity For Scale. Each feature was tagged with a maturity level (`dead_code`, `stub`, `thin_backend_no_surface`, `backend_solid_no_ui`, `fully_wired`), grounded in `grep`/`rg` evidence, git log commit counts, and exact file/line citations. Stage two took the 8 highest-leverage findings from stage one and cross-referenced each against 2026 web research on mem0, Zep/Graphiti, Letta, LangMem, Supermemory, and recent arXiv publications (May-July 2026) on multi-agent memory governance, retroactive consolidation, and precision-aware retrieval benchmarks. The result: 8 features get full build plans below, the remaining 30 are logged so nothing the mapping found is silently dropped.

The throughline across all 8 deep-dive features is that Vestige already has the one primitive competitors do not: a causal edge type in its memory graph plus retroactive salience backfill. Almost every recommendation below is about connecting already-built, currently-orphaned code to that causal graph rather than inventing new subsystems.

## Ranked Build Opportunities

| Rank | Feature | Maturity today | Effort |
|---|---|---|---|
| 1 | Codebase-awareness module (git history mining, pattern detection, relationship graph) | `dead_code`, 5,534 LOC built | L |
| 2 | Synaptic Tagging & Capture (retroactive importance rescue) | `thin_backend_no_surface`, 1,620 LOC | S |
| 3 | Hippocampal Indexing (sparse pre-filter + provenance) | `thin_backend_no_surface`, 2,303 LOC | M |
| 4 | Shared write bus / causal-arbitrated fleet writes | does not exist in shipped code | XL |
| 5 | ImportanceTracker causal-importance channel | `stub`, write-only | S |
| 6 | IntentDetector zero-integration intent classification | `stub`, permanent no-op | M |
| 7 | Content-type-routed adaptive embedding | `dead_code` | M |
| 8 | Causal-anchored, receipt-verified compression | `dead_code`, 743 LOC | M |

Ranking logic: rank 1-3 are cases where the hard engineering is already written and sitting idle, and each one strengthens the causal graph that ranks 4-5 depend on. Rank 4 is the biggest strategic payoff (a fleet-memory capability literally no competitor product ships) but it is the only item requiring substantial net-new code, so it is sequenced after the causal graph is richer. Ranks 5-8 are valuable, decoupled improvements that can land in any order once the causal graph work is underway.

---

## 1. Codebase-Awareness Module: Git-Native Fleet Intelligence

### Current State

`crates/vestige-core/src/codebase/` is a fully built, unit-tested, 5,534-line subsystem: `git.rs` (812 LOC, `GitAnalyzer.learn_from_history()`), `patterns.rs` (729 LOC, `PatternDetector` with built-in bug-fix pattern recognition), `relationships.rs` (725 LOC, `RelationshipTracker` co-edit graph), `watcher.rs` (728 LOC, `CodebaseWatcher` filesystem watching), `context.rs` (980 LOC, `ContextCapture`), and `mod.rs` (769 LOC, whose own doc comment calls it "Vestige's KILLER DIFFERENTIATOR"). None of it runs in production. `rg -n "CodebaseMemory" --type rust -l` matches only `codebase/mod.rs` and the `pub use` re-export in `crates/vestige-core/src/lib.rs`. The actual shipped MCP tool named `codebase` is a separate, much thinner implementation, `crates/vestige-mcp/src/tools/codebase_unified.rs` (692 LOC), which imports only `vestige_core::{IngestInput, OutputConfig, Storage}` and never touches `vestige_core::codebase::*`. It supports three actions (`remember_pattern`, `remember_decision`, `get_context`) that store plain `IngestInput` records through the generic memory pipeline. It does not call `GitAnalyzer`, does not run pattern detection, does not build a relationship graph, and does not watch files. `rg -n "GitAnalyzer|PatternDetector|RelationshipTracker|CodebaseWatcher|ContextCapture" crates/vestige-mcp/src/` returns zero matches. No dashboard route references any of the five public types.

### 2026 Landscape

mem0 and Supermemory retrofit generic vector/graph memory for coding use via MCP; neither mines git history or models file co-change. Zep/Graphiti is conversation-centric temporal reasoning (entity/fact evolution over chat history), not commit-centric, so it has no git log mining, no file relationship graph, and no bug-fix pattern detection. Letta's OS-inspired paging model has no codebase awareness at all. The closest competitor is Kage, a git-native memory tool that stores manually-authored markdown lesson files verified against current code at write/recall/diff time, but the notes are human-authored, not algorithmically mined from `git log`, and it has no co-edit graph, no file watcher, and no fleet coordination surface. Cognee and similar code-KG tools build structural knowledge graphs from AST parsing, giving file:line-cited navigation, but the graph is a static structural snapshot, not a temporal one; there is no git-history learning or bug-fix pattern mining. Academic support: HAFixAgent shows git blame and commit-history context measurably improves LLM bug-repair accuracy, and SWE-ContextBench (arXiv 2602.08316, 1,476 tasks across 51 repos) shows agents that reuse prior related-issue/PR context perform significantly better than agents treating each task independently. Neither result has been shipped as persistent, fleet-shared memory by any product.

### Differentiation Angle

Nobody ships a git-evidenced, causally-linked, live-updating codebase relationship graph shared across a coding agent fleet. The gap is not "build git mining," it is "wire git mining, which Vestige already has, into the causal graph, which Vestige already has." Co-edit relationships from `RelationshipTracker` become correlated edges. `PatternDetector`'s bug-fix findings route through the existing retroactive-salience-backfill machinery, so a fix in file A caused by an earlier change in file B produces a real causal edge with commit SHAs attached, not a similarity score. Every surfaced pattern carries a receipt (commit SHAs, co-edit counts, confidence), which differentiates it from Kage's unverifiable hand-written notes and from Zep/mem0's opaque retrieval.

### Concrete Build Plan

1. Persist `CodebaseNode` through `Storage`/SQLite instead of the current in-memory `RwLock<Vec<CodebaseNode>>`, so results survive restarts and are queryable fleet-wide.
2. In `codebase_unified.rs`, replace the thin `remember_pattern`/`remember_decision`/`get_context` handlers with real calls into `GitAnalyzer.learn_from_history()` and `RelationshipTracker`'s co-edit graph.
3. Map co-change relationships onto Vestige's existing causal edge type: co-edits become correlated edges, and `PatternDetector` bug-fix findings flow through the existing retroactive-salience-backfill path (`retroactive_backfill.rs`) instead of a parallel scoring system.
4. Attach a receipt (commit SHAs, co-edit counts, confidence) to every surfaced pattern via the existing audit trail.
5. Wire `CodebaseWatcher`'s file events into per-agent session state in `Storage`, so "who else in the fleet is touching this file right now" becomes a live query, activating the dead watcher as the multi-agent fleet coordination surface it is already filed under.

### Why It Scales For Multi-Agent Fleets

`CodebaseWatcher` and `GitAnalyzer` already run entirely local against the repo's own `.git` and filesystem, so scaling to a concurrent agent fleet means treating Vestige's existing local SQLite `Storage` as the coordination substrate instead of standing up new infrastructure: each agent's session start writes an ephemeral active-file-set row, the co-edit graph is computed once per repo and shared read-only, and pattern/history results are cached and invalidated on new commits rather than recomputed per session. This stays local-first, unlike Zep/mem0's hosted graph infrastructure, while giving every agent a live answer to who else is touching a file and what history says usually happens next. Because it reuses the causal-edge/backfill machinery, this scales as a graph query, not a new subsystem, and receipts on every claim directly address the "contradiction persistence" and "provenance collapse" failure modes named in the Governed Shared Memory paper (arXiv 2606.24535).

### Effort: L

---

## 2. Synaptic Tagging & Capture: Retroactive Importance Rescue

### Current State

`crates/vestige-core/src/neuroscience/synaptic_tagging.rs` (1,620 LOC) implements the biological Synaptic Tagging and Capture window from Frey & Morris 1997: `trigger_prp()` retroactively "captures" temporally-adjacent memories into an `ImportanceCluster` when a high-importance event fires, with a default ~9-hour-back/2-hour-forward window. This runs for real on every ingest: `tag_memory()` and `trigger_prp()` are called from `smart_ingest.rs:586,590` and `trigger_prp()` is called again from `autopilot.rs:257` on `MemoryCreated` events. But the `CaptureResult` from `smart_ingest.rs:590` is bound to `_capture` and immediately discarded, so nothing downstream ever reads which memories got captured or applies a strength boost. Every read method (`get_active_tags`, `get_all_clusters`, `get_clusters_for_memory`, `is_captured`, `has_active_tag`, `stats`, `get_capture_candidates`) has zero callers in `vestige-mcp`. The only tool-shaped code resembling a surface, `trigger_importance`/`find_tagged`/`tagging_stats` in `tools/tagging.rs`, is explicitly marked `(Deprecated)` in its own module doc, is not registered in `tools/list` (`server.rs` comments it as "internal, not in tools/list"), and does not even use the real `SynapticTaggingSystem` state: `trigger_importance` constructs a throwaway `SynapticTaggingSystem::with_config()` per call, built fresh and discarded after one request, and `find_tagged`/`tagging_stats` bypass the module entirely by filtering `storage.get_all_nodes()` on `retention_strength` as a proxy.

### 2026 Landscape

mem0's four-lever consolidation framework (importance, merge, decay, eviction) scores importance once at write time via LLM rating and never revisits the judgment. Zep/Graphiti's bi-temporal graph invalidates facts going forward as new information arrives but has no mechanism to strengthen an old, weak memory's salience retroactively. Letta's agent-driven memory management (explicit `core_memory_append`/archival calls) is entirely write-time and agent-judgment-dependent; if the agent misses the importance of something in the moment, there is no automatic backward-looking rescue. LangMem's consolidation merges related facts and resolves contradictions but does not re-score older memories in light of new evidence. Supermemory's LLM-guided reranking and supersession operate on retrieval and on which fact is current, not on rewriting stored importance retroactively. The Hindsight "Consolidation Problem" survey directly compares mem0/Zep/Letta/LangChain/Hindsight across these four levers and states plainly that no shipped system implements retroactive importance recalibration. The one piece of prior art that names this exact mechanism, "HindsightTag: A Synaptic Tagging-and-Capture Framework for Retroactive Memory Consolidation in LLM Agents," is an independent researcher's preprint still awaiting arXiv cs.AI endorsement, with no available implementation, benchmark, or causal gating.

**Unknown:** HindsightTag has not been independently verified as published or peer-reviewed; treat it as an idea with no shipped artifact, not a competing product.

### Differentiation Angle

Ship "Causally-Gated Importance Capture with Receipts": the only memory system that (1) retroactively promotes a memory's stored strength when a later, unrelated high-importance event fires, not just decays or invalidates forward in time, (2) gates that promotion by causal-graph adjacency in addition to the biological temporal window, so capture stays precision-safe instead of indiscriminately boosting anything that happened to occur nearby in time, and (3) emits a receipt for every capture event so the strength change is auditable, explainable, and revocable. All of this at zero added latency and zero cloud calls, because the `CaptureResult` is already computed on every ingest today and simply thrown away.

### Concrete Build Plan

1. Bind the real `CaptureResult` in `smart_ingest.rs` (stop discarding it as `_capture`) and apply a bounded `retention_strength` boost via the existing `ImportanceCluster` data, respecting any existing suppress/purge state so a capture can never resurrect an explicitly suppressed memory.
2. Add causal gating: intersect the PRP temporal window with Vestige's causal edge type in the memory graph. A memory is only captured if it is temporally in-window AND causally reachable from the triggering event, addressing the precision concern PrecisionMemBench raises about untargeted retrieval noise.
3. Emit a receipt for every capture event (trigger memory ID, captured memory IDs, causal-path justification, strength delta) through the existing receipt/audit system, giving users a "why did this memory's importance just change" answer no competitor offers.
4. Register a real MCP tool (`synaptic_capture_history` or similar) using `get_active_tags`/`get_all_clusters`/`get_clusters_for_memory`/`is_captured`/`get_capture_candidates`, and retire the deprecated `trigger_importance`/`find_tagged`/`tagging_stats` trio in `tools/tagging.rs`, which use throwaway state and bypass the real module.
5. Surface a capture timeline in the dashboard's existing Observatory/causal-graph visualization, reusing rendering infrastructure that already exists.

### Why It Scales For Multi-Agent Fleets

If Agent 3 hits a build failure four hours after Agent 1 made an unremarkable-looking decision, causally-gated capture retroactively promotes Agent 1's decision automatically, before any agent manually greps or diagnoses from scratch. Concurrency is already handled: SQLite write serialization already covers `tag_memory`/`trigger_prp` on every ingest in production, so the new work (strength boost plus receipt write) rides the same transaction path with no new concurrency surface. The real scaling risk is capture noise growing with agent count, since more concurrent agents means more candidate memories inside any given temporal window; this is exactly why causal-graph gating must be mandatory, not optional, at fleet scale. Captured memories remain fully subject to later suppression or purge, and the receipt trail survives that, in line with the provenance/supersession model the Governed Shared Memory paper (MemClaw/ArgusFleet) recommends.

### Effort: S

---

## 3. Hippocampal Indexing: Sparse Pre-Filter With Native Provenance

### Current State

`crates/vestige-core/src/neuroscience/hippocampal_index.rs` (2,303 LOC, the largest single file in the neuroscience cluster) implements a complete two-phase retrieval system inspired by Teyler & Rudy 2007: `MemoryBarcode` unique IDs, an `ImportanceFlags` bitset, a `ContentPointer`/`ContentStore`/`StorageLocation` abstraction over SQLite/vector-store/filesystem/inline content, an in-memory `MemoryIndex` with associative `IndexLinks`, and a `HippocampalIndex` struct exposing `index_memory`, `search_indices`, `recall`, `recall_semantic`, `get_associations`, `add_association`, `retrieve_content`, `migrate_node`, `migrate_batch`, `create_semantic_associations`, `update_importance_flags`, and `prune_weak_links`. The write path is wired: `cog.hippocampal_index.index_memory()` is called from 3 sites (`codebase_unified.rs` twice, `smart_ingest.rs`), with the result discarded via `let _ =`. On the read side, exactly one method has a real caller: `get_associations()` is called from `explore.rs:98-101` and blended into the `graph` unified tool's `associations` action, which the dashboard's `/explore` route already exercises (see item 4 in "Also Worth Noting" below). Every other read/analysis method, `search_indices`, `recall`, `recall_semantic`, `add_association`, `retrieve_content`, `migrate_node`, `migrate_batch`, `create_semantic_associations`, and `update_importance_flags`, has zero callers anywhere in `crates/vestige-mcp/src`. Critically, `search_indices`/`recall`, the two methods that would actually make this a sparse pre-filter ahead of the main recall pipeline, are among the unwired ones: the sparse-index/content-pointer split has never been inserted into the hot retrieval path it was built for.

There is also a second, quieter bug in the same family as the ImportanceTracker no-op described in feature 5 below: the periodic maintenance sweep's "13. Hippocampal Index Maintenance" step (`storage/sqlite.rs:3952-3955`) already calls `prune_weak_links()`, but on a brand-new `HippocampalIndex::new()` constructed inline for that one call, not on `CognitiveEngine`'s persistent instance. Since the fresh instance's link table is always empty, every scheduled prune is a no-op against real data, identical in shape to the `apply_importance_decay()` bug at step 14.

No dashboard file references `HippocampalIndex` or `MemoryBarcode` in code (a UI copy string in `DreamStageReplay.svelte` uses the word "hippocampal" descriptively, not as a reference to the module).

### 2026 Landscape

mem0's two-phase extraction/retrieval touches the full graph+vector store on every recall, with no sparse pre-index or pointer/content separation. Zep/Graphiti always pays graph-traversal cost; there is no cheap index-only tier ahead of the graph, and while it is strong at temporal supersession it is not built for multi-agent provenance or fleet-scale write concurrency. Letta's three-tier core/archival/recall model is the closest architectural analog among competitors, but it is single-agent runtime-centric (agents run inside Letta) with no native multi-agent fleet provenance model, and the tiering targets context-window economy, not query-time precision at a growing shared store. Supermemory is deliberately flat, embeddings plus time-annotation, optimized for single-user token economy, not multi-agent provenance. DecentMem (arXiv 2605.22721) explicitly rejects a centralized shared memory repository, giving each agent its own dual-pool memory instead, citing communication overhead and "diversity collapse" as the reason; this solves a different topology than Vestige's actual deployment (one local-first store shared by a fleet on one machine). MemClaw / Governed Shared Memory (arXiv 2606.24535) treats fleet memory as a distributed-systems problem with scoped retrieval and provenance tracking, reconstructing depth-four derivation chains with correct writer identity at sub-second per-hop latency, but it does so with a full distributed governance service. Supermemory's own reported latency research states two-stage retrieval (fast approximate search first, rerank only top candidates) cuts latency to 40-80ms, and that extraction/maintenance, not retrieval, is the dominant cost in most systems.

### Differentiation Angle

Activate the already-built sparse-index/content-pointer split as a fleet-scale precision pre-filter with native provenance, getting most of MemClaw's provenance/scoping benefit and PrecisionMemBench-style noise-hard-failure precision at a fraction of the architectural cost, because Vestige is local-first and single-node per fleet rather than a distributed multi-tenant service. `MemoryBarcode` is a natural provenance token, `ImportanceFlags` is a policy/salience-scoping bitset, and `ContentPointer`/`ContentStore` already provide the lazy-resolve boundary that keeps expensive resolution (SQLite/vector/filesystem) off the hot path.

### Concrete Build Plan

1. Insert `HippocampalIndex.search_indices()`/`recall()` as an actual pre-filter stage inside the main MCP recall path in `crates/vestige-mcp`, ahead of the hybrid embedding+keyword search: scan the in-memory `MemoryBarcode`/`ImportanceFlags` index first to produce a narrowed `ContentPointer` candidate set, then resolve full content only for that narrowed set via `retrieve_content`.
2. Add a writer/agent-id plus session-id field to `MemoryBarcode` so every indexed memory is natively provenance-tagged at write time, and use it to scope `search_indices`/`recall` by current agent or fleet session by default, directly targeting the "provenance collapse" and "unauthorized leakage" failure modes from the Governed Shared Memory paper, solved as an in-process filter rather than a distributed governance service.
3. Wire `update_importance_flags()` into the retroactive-salience-backfill pathway, so when backfill traces a causal-upstream failure and boosts an earlier memory's salience, that boost lands in the sparse index's `ImportanceFlags` bitset directly. This connects the two most architecturally significant but currently disconnected systems in the codebase: backfill decides what matters, the hippocampal index makes what-matters fast to retrieve.
4. Fix the `prune_weak_links()` wiring bug at `storage/sqlite.rs:3952-3955`: point the maintenance sweep at `CognitiveEngine`'s persistent `HippocampalIndex` instance instead of a throwaway `HippocampalIndex::new()`, so `IndexLinks` accumulated from N concurrent agents' `add_association` calls actually get pruned instead of the sweep silently doing nothing.

### Why It Scales For Multi-Agent Fleets

As agent count N grows, barcode/importance-bitset pre-filtering keeps the in-memory scan cheap and avoids every agent's recall call hitting the disk-backed SQLite/vector store for coarse candidate generation, since only the narrowed `ContentPointer` set triggers a real content fetch. Writer/session-scoped barcodes give natural read isolation between concurrent agents without locking or distributed consensus. A correctly-wired `prune_weak_links` (once the step-13 no-op is fixed) keeps `IndexLinks` from growing unbounded as write volume scales with fleet size, avoiding the "contradiction-persistence" failure mode the multi-agent literature flags. None of this requires new storage, new dependencies, or schema migrations beyond one struct field; the marginal cost to activate it is call-site wiring plus one bug fix, which keeps the change scoped enough to verify with `cargo test`/`clippy`/the dashboard build gates rather than a rewrite.

### Effort: M

---

## 4. Shared Write Bus: Causal-Arbitrated Fleet Write Coordination

### Current State

There is no dedicated, named "shared write bus" or fleet-coordination feature anywhere in `crates/vestige-core` or `crates/vestige-mcp`. `grep` for `agent_id`/`fleet`/`multi-writer`/`conflict-resolution`/`write-lock`/`shared-write` across `crates/vestige-core/src/storage/*.rs` and `crates/vestige-core/src/*.rs` returns zero matches. What exists is generic: `SqliteMemoryStore` (`crates/vestige-core/src/storage/sqlite.rs`, `struct SqliteMemoryStore` at line 304, PRAGMA configuration in `configure_connection()` starting at line 466) applies standard PRAGMAs, WAL journal mode, `synchronous=NORMAL`, `busy_timeout=5000`, which is ordinary SQLite multi-process safety, not something Vestige built specifically for fleets. There is no per-write `agent_id` column, no conflict-resolution logic for concurrent live writes, and no MCP tool for coordinating a fleet. `portable.rs` (171 LOC) is a genuinely separate concept: whole-database archive export/import for offline device-to-device sync with a real "Merge" conflict-handling mode, but this is asynchronous file-based sync between one user's devices, not live coordination between concurrently-running agent processes. The "shared write bus" referenced in `docs/business/OUTREACH-WAVE-02-SILENT-ROTATION.md` and `OUTREACH-WAVE-03-REPLICATION-CRITICS.md` is entirely a benchmark-harness construct on the unmerged `benchmark/silent-rotation` branch: `fleet_runner.py`'s `SharedBus` class shells out to the `vestige` CLI with every agent pointed at the same `--data-dir`, and serializes writes with a plain Python `threading.Lock()` inside the benchmark process, reusing the pre-existing generic `source` string field for attribution rather than a purpose-built mechanism. Both outreach docs state explicitly that "the ablation separating event anchoring, causal traversal, and the shared write bus is pre-registered but unrun."

### 2026 Landscape

mem0's Group Chat flow attributes writes via a message `name` field, filterable at retrieval time, but has no conflict detection or arbitration logic at all. Zep/Graphiti's bi-temporal model handles "this fact replaced that fact over time" for a single writer's evolving knowledge, not live arbitration between two agents' concurrent, competing writes right now. Letta's 2026 Context Repositories feature (git-backed memory, isolated worktrees per subagent, merge back via git's built-in conflict resolution with commit-message versioning) is the closest shipped analog to a fleet write bus, but conflict resolution is generic git text-diff merge with no concept of why a write happened or which of two conflicting claims is more likely true given downstream outcomes. StateFuse (arXiv 2607.05844) rejects last-writer-wins and keeps conflicts as explicit, auditable `ConflictSet` objects via CRDT OpSet merge, but explicitly narrows its claim to surfacing and correction, not resolving which claim is correct. MemTX (arXiv 2607.23929) gates irreversible tool actions behind a validate-and-commit pipeline with an 8-state lifecycle and 5 isolation levels, but adjudicates conflicts by temporal-precedence and source-authority, not causal outcome. Grite (arXiv 2606.19616), a server-less git-native coordination substrate for concurrent coding agents using advisory leases, gets duplicate-work rate to 0.00 and triples goodput, but leases are advisory only and cannot enforce on an uncooperative agent, and conflict handling is convergence-based, not causal-evidence-based. The Governed Shared Memory paper (arXiv 2606.24535) resolves conflicts via temporal supersession (`supersedes_id` links), and its own ArgusFleet evaluation documents a real production ordering bug: a synchronous dedup gate ran before the async contradiction detector, letting real contradictions slip through undetected.

### Differentiation Angle

None of these systems use causal evidence to adjudicate which of two concurrent, contradictory writes from different fleet agents was actually correct. Point Vestige's existing backfill mechanism forward instead of only backward: when two agents write contradictory claims, do not last-writer-wins it (current de facto behavior via SQLite WAL) and do not just flag-and-stop (StateFuse's approach). Asynchronously keep tracing both claims until one accumulates a causal edge to a validated downstream outcome (a passing test, a successful build, a merged PR) and the other accumulates a causal edge to a failure, then arbitrate on that evidence. This turns write coordination from a generic CRDT/transaction problem into a causal-inference problem, which is structurally impossible for mem0/Zep/Letta/StateFuse/MemTX to replicate without first building an equivalent causal-edge primitive into their schema.

### Concrete Build Plan

1. Add a `writer_id`/agent-attribution column via migration to the memories table in `sqlite.rs` (today the only per-write identity is the generic `source` string field reused ad hoc by connectors and the benchmark harness's `fleet-agent-{id}` convention). Promote it to a first-class column, matching the minimum bar the Governed Shared Memory paper sets for trustworthy fleet memory.
2. At write time, run a synchronous conflict-detection check reusing the existing dedup/similarity logic to flag when a new write is a near-duplicate-but-contradictory claim about the same entity/file/decision written by a different `writer_id` within a recent window. Keep this synchronous, deliberately avoiding the exact ordering bug the Governed Shared Memory paper found in production where their sync dedup gate ran before their async contradiction detector.
3. When a conflict is flagged, asynchronously invoke the existing backfill mechanism on both competing memory IDs (extend the `backfill` MCP tool with an `arbitrate(id_a, id_b)` mode) to see which claim develops a causal edge to a later-validated outcome versus a causal edge to a failure. Write the verdict as a new causal edge plus a receipt, and leave both claims in the graph, matching Vestige's existing purge-tombstone discipline of never silently deleting.
4. Surface a fleet-conflict view in the dashboard showing pending vs. arbitrated conflicts. Today there is nothing to show, since SQLite WAL alone has no fleet-aware state to expose.

### Why It Scales For Multi-Agent Fleets

The existing single-writer `Mutex<Connection>` plus WAL plus `busy_timeout=5000` already serializes disk writes correctly for any number of OS processes on one data directory; fleets of 5-50 concurrent coding agents are fine here because individual memory writes are small metadata operations, not the heavy coding work itself. Conflict-detection stays on the synchronous write path (cheap similarity check, already-paid-for dedup logic) so it can never race an async step; the causal arbitration step is the only piece that runs asynchronously, and it must complete or degrade to "pending arbitration" visibly rather than silently. This only targets Vestige's actual scope, one shared local `--data-dir` per team/repo, not a hosted multi-tenant cloud service, which sidesteps the harder unauthorized-leakage/tenant-isolation problem MemClaw had to solve. As fleet size grows, the causal-arbitration backlog is bounded by how many genuinely contradictory concurrent writes occur, not by fleet size directly, since most fleet writes target disjoint files/entities and never trigger the conflict path.

### Effort: XL

---

## 5. ImportanceTracker: A Causally-Confirmed Importance Channel

### Current State

`crates/vestige-core/src/advanced/importance.rs` (502 LOC) tracks per-memory `usage_importance`/`recency_importance`/`connection_importance` with helpful/unhelpful reinforcement, decay, and "neglected memory"/"top by importance" queries. This is a completely different module from the MCP tool literally named `importance_score` (`crates/vestige-mcp/src/tools/importance.rs`), which wraps an unrelated 4-channel dopamine/norepinephrine/acetylcholine/serotonin neuroscience model (`cog.importance_signals`) also used by the dashboard's `/importance` page. `ImportanceTracker.on_retrieved()` is called for real on every promote/demote (`feedback.rs:95,167`) and every search-result marked helpful/unhelpful (`memory_unified.rs:331,388`), so it accumulates a real usage signal. But every read/analysis method is unreachable: `apply_importance_decay()`, `get_top_by_importance()`, `get_neglected_memories()`, `weight_by_importance()`, and `get_all_scores()` are never called on `cog.importance_tracker` anywhere in `vestige-mcp`. The one place `apply_importance_decay()` IS called, the maintenance sweep at `storage/sqlite.rs:3958-3960` (step 14), constructs a brand-new, empty `ImportanceTracker::new()` and calls decay on it, a complete no-op since the fresh instance's `scores` map is empty; it never touches the persistent scores CognitiveEngine actually accumulated. The module is write-only dead weight.

### 2026 Landscape

mem0's three-signal retrieval score (recency times LLM-rated importance times relevance) assigns importance once at write time and never updates it from outcomes; a separate decay layer re-ranks by recency only. Zep/Graphiti is time-aware, not outcome-aware; no signal ties a fact's importance to whether using it actually helped or hurt a downstream task. Letta and LangMem's frontier direction is end-to-end RL-trained "memory models" (Memory-R2/LoGo-GRPO, AgeMem), where store/retrieve/update/discard become RL-optimized tool calls, which is heavyweight and not deployable inside a lightweight local MCP server with no training infrastructure. Supermemory's composite score (temporal decay, access frequency, semantic signals) is still purely associational, with no causal graph to draw an outcome-attribution signal from. The state-of-the-art academic primitive is the Memory Worth (MW) metric, a two-counter online statistic tracking co-occurrence with success/failure outcomes, proposed in "When to Forget: A Memory Governance Primitive" (arXiv 2604.12007). The paper reports MW reaching Spearman rho=0.89 against true utility after 10,000 episodes, but the authors explicitly state MW "measures outcome co-occurrence rather than causal contribution, it is an associational quantity, not a causal one." That is precisely the gap Vestige's causal edge type and backfill mechanism can close.

**Unknown:** the rho=0.89 figure is self-reported by the paper's own evaluation; treat it as the paper's claim, not independently reproduced.

### Differentiation Angle

Vestige uniquely already has the missing ingredient every competitor lacks: retroactive salience backfill plus a causal edge type. Fusing that with the currently-dead `ImportanceTracker` produces a `causal_importance` channel, importance that is causally-confirmed rather than merely correlated-with-a-later-thumbs-up. This is the exact capability the Memory Worth authors name as future work, and no product competitor's architecture supports it at all.

### Concrete Build Plan

1. Fix the wiring bug at `storage/sqlite.rs:3958-3960`: the maintenance sweep must call `apply_importance_decay` on `CognitiveEngine`'s persistent `ImportanceTracker` instance, not a disposable `ImportanceTracker::new()`. This alone makes decay real instead of a no-op. Note that the maintenance sweep has the same bug one step earlier, at step 13 for `HippocampalIndex.prune_weak_links()` (see feature 3 above); fix both in the same pass, since they are the identical class of mistake.
2. Wire the currently-unreachable read methods (`get_top_by_importance`, `get_neglected_memories`, `weight_by_importance`) into an actual surface: expose "neglected memory" resurfacing as a real MCP tool/dashboard panel, and feed `weight_by_importance` as an actual re-rank multiplier in `memory_unified.rs` search, not just a recorded-but-unused score.
3. Extend `on_retrieved()` so that when a search result's helpful/unhelpful reinforcement coincides with a confirmed causal-upstream edge from backfill (this memory was traced as causally upstream of a later failure or success, not merely retrieved alongside it), write a separate `causal_importance` delta with materially higher weight than ordinary correlational reinforcement, and surface the distinction ("causally confirmed" vs. "merely co-retrieved") in recall output and the dashboard's importance view.

### Why It Scales For Multi-Agent Fleets

Extend the per-memory importance map to key by `(memory_id, agent_role)` rather than `memory_id` alone, so a memory can register as critical for a reviewer-agent while being irrelevant to a drafting-agent, entirely inside one local SQLite file. This avoids the communication/coordination overhead DecentMem (arXiv 2605.22721) pays to get per-agent memory diversity via decentralization. Because Vestige keeps a centralized causal graph, cross-agent fair credit assignment (agent A's fix causally upstream of agent B's later success) falls out of the existing backfill machinery for free, the same goal Memory-R2/tree-based credit-assignment research pursues via expensive RL rollouts, achieved here with zero training and no per-agent memory forking.

### Effort: S

---

## 6. IntentDetector: Zero-Integration Intent Classification

### Current State

`crates/vestige-core/src/advanced/intent.rs` (914 LOC) scores a sliding window of recorded `UserAction`s against 5 hand-coded pattern scorers (Debugging/Refactoring/Learning/NewFeature/Maintenance) to produce a `DetectedIntent` with a confidence score. `detect_intent()` is genuinely called at 3 production call sites (`intention_unified.rs:307`, `smart_ingest.rs:194,397`), each auto-tagging a memory or reminder when confidence exceeds 0.5. But `record_action()`, the only way to populate the action history `detect_intent()` reads, is never called anywhere in `vestige-mcp`. `get_recent_actions()` therefore always returns an empty `Vec`, so `detect_intent()` always returns `DetectedIntent::Unknown` with `confidence: 0.0`, and all 3 call sites' `if intent_result.confidence > 0.5` gate is permanently unreachable. The feature looks wired (it is called, its output is checked, there is a real branch) but is provably a permanent no-op in production; the auto-intent-tagging feature described in the code comments has never fired on a single real memory.

### 2026 Landscape

mem0 routes each incoming chat message to a fixed set of intents (create_file, write_code, chat, etc.) via an LLM classification call before acting, an approach that requires an LLM call per turn (cost plus latency) and derives intent from chat messages, not from tool/action telemetry. Zep/Graphiti's bitemporal graph is strong at retroactive correction but has no dedicated action-history-driven task-intent classifier. Letta requires the agent to actively participate in memory management via explicit function calls, the same instrumentation-burden problem that killed Vestige's own `record_action()`, just owned by a different framework instead of solved generically. STITCH (arXiv 2601.10702) indexes every trajectory step with a structured "contextual intent" cue and retrieves history by intent compatibility, beating similarity-only baselines by 35.6% on CAME-Bench, but it assumes a controlled single-agent trajectory the system can instrument directly, built and evaluated as part of an agent framework rather than as an MCP server serving many independent unmodified clients.

### Differentiation Angle

Every competitor's intent-detection approach assumes the memory system owns or sits inside the agent loop, so it can cheaply instrument every step. Vestige does not own the loop; it is an MCP server called by unmodified coding-agent clients, which is exactly why `record_action()` was never wired: nobody controls call sites in Claude Code, Cursor, or Goose to sprinkle in explicit action logging. The fix is to stop requiring instrumentation at all: derive intent for free from the MCP tool-dispatch boundary Vestige already owns, since every `recall`/`smart_ingest`/`codebase`/`graph`/`backfill` call already IS an action. This makes Vestige the only memory system whose intent signal is zero-integration; it activates the moment any MCP client starts calling standard tools, with no SDK, no wrapper, no cooperating agent framework required. Then write detected intent transitions into the causal graph itself, so "why did this fix land" queries can show intent-state history (Debugging to Refactoring to NewFeature) as first-class causal-upstream evidence.

### Concrete Build Plan

1. Move `record_action()` calls out of application logic entirely and into the single MCP tool-dispatch middleware in `vestige-mcp`, the shared boundary every tool handler already passes through. Map each existing tool call to an action type (`recall` to search, `smart_ingest` to edit/commit, `codebase` to file-open/explore, `backfill` to debugging-signal, `graph` to explore, `suppress`/`dedup` to maintenance), so intent detection self-populates from tools that already fire, with zero new call sites added to caller code.
2. Extract lightweight signals (regex/keyword, not an LLM call) from `smart_ingest`'s content field, file paths, error strings, stack-trace shapes, commit-style verbs, to enrich the `UserAction` record beyond just "which tool was called."
3. Replace the binary `confidence > 0.5` gate with storage of raw per-pattern scores on the ingested memory/reminder row, so even low-confidence signal is queryable later instead of being all-or-nothing.
4. Emit a causal-graph edge or node on each detected intent-state transition, so `graph`/`backfill` queries can show intent history as part of a root-cause trace.
5. Key the sliding action window by session/agent identity, already present per MCP connection in the local SQLite schema, so intent state is tracked per-agent, enabling a fleet-level "who is doing what right now" view.

### Why It Scales For Multi-Agent Fleets

Because the fix lives at the MCP dispatch boundary, a single choke point already handling all tool traffic, action recording is O(1) per call and needs no new network hop, staying local-first and adding negligible latency versus a chat-message LLM-classification approach like mem0's. Batching action-history writes to SQLite (WAL mode, single writer) avoids lock contention when many agents call tools concurrently, since action records are small structured rows, not embeddings. Fleet-level views (which agents are currently debugging the same subsystem) are a read-side aggregation over the per-agent windows and can be exposed as a new lightweight query without touching the write path. Because this stays entirely local, no cloud round-trip, no LLM call for classification, it scales to many concurrent agents on one machine at effectively zero marginal cost per agent.

### Effort: M

---

## 7. Content-Type-Routed Adaptive Embedding

### Current State

`crates/vestige-core/src/advanced/adaptive_embedding.rs` (771 LOC) detects content type (code/technical/error-log/structured/NL) via heuristics and picks an "embedding strategy," but then generates a fake embedding with `pseudo_embed()`, a hash-based vector, not a call to the real fastembed pipeline Vestige actually uses elsewhere. `AdaptiveEmbedder` is instantiated as a `CognitiveEngine` field (`cognitive.rs:78,161`) but `.embed()`/`.embed_auto()` are never called anywhere in `vestige-mcp`. `rg -rn "\.adaptive_embedder\b" crates/vestige-mcp/src` matches only the field declaration and instantiation, no method calls. `ContentType::detect()` is called twice in `smart_ingest.rs` (lines 207 and 408) into a variable named `_content_type` with a comment claiming it is "for logging," but no log statement exists; the result is computed and immediately discarded.

### 2026 Landscape

mem0's hybrid vector plus graph plus KV storage routes by content nature (facts vs. entity relationships vs. exact flags), not by embedding model; one embedding space underlies all three. PrecisionMemBench measured mem0 OSS at 0.06 mean retrieval precision with zero active retrieval passes, the target memory buried in unrelated noise. Zep/Graphiti differentiates content by LLM extraction and chunking, not embedding model, and pays for it with 600k+ token graphs per conversation and hours of post-ingest lag before new facts become retrievable. Letta's embedding config is a single global setting on the agent, no content-type routing at all. LangMem models semantic/episodic/procedural memory types but embeds all of them into one 1536-dim vector space via whatever store backend is configured; content type informs which memory TYPE gets extracted, not which embedding model is used. Supermemory is the closest competitor, with AST-aware chunking for code, OCR for images, transcription for video, and markets 95% Recall@15 at roughly 720 tokens on LongMemEval, but the specialization is at the chunking/extraction layer, not the embedding-model layer, and it is a hosted cloud service with no receipt/provenance layer. The Voyage AI voyage-code-3 benchmark reports code-specific embedding models outperform general-purpose text embeddings by 13-17% on code retrieval datasets across a 32-dataset suite; fastembed-rs already ships a local equivalent, `jinaai/jina-embeddings-v2-base-code`, so this gain is available with zero new cloud dependency.

**Unknown:** the 13-17% figure comes from Voyage AI's own ongoing 2026 benchmark comparisons and has not been independently reproduced against Vestige's specific corpus.

### Differentiation Angle

Nobody in the 2026 field routes different content types to genuinely different local embedding models AND makes that routing decision part of an auditable causal-provenance trail. Make `ContentType` a first-class, receipt-visible dimension of retrieval, not just an embedding-model selector but a recorded "why this memory matched" fact tied into the existing causal graph, turning the currently-dead detection logic into an explanation layer competitors do not have.

### Concrete Build Plan

1. Delete `pseudo_embed()`; wire `AdaptiveEmbedder.embed_auto()` into both `smart_ingest.rs` call sites (lines 207 and 408), replacing the discarded `_content_type` with a real routing call into the existing fastembed pipeline.
2. Load a second local ONNX model fastembed-rs already supports, `jinaai/jina-embeddings-v2-base-code`, lazily downloaded exactly like the current default text model. Route `ContentType::Code`/`Technical` to it, keep the default model for NL/structured content. Zero new dependencies.
3. For `ContentType::ErrorLog`, add a normalized stack-frame structural fingerprint (hash-based near-dup key) alongside the embedding, since exact/near-exact dedup of recurring stack traces is a structural-match problem embedding similarity alone misses.
4. Persist `content_type` as a first-class field on the memory record and surface it in receipt/graph metadata, so a causal edge from an ErrorLog memory to the Code memory that fixed it records which embedding strategy fired on each side.
5. Use `content_type` for asymmetric retrieval: an error-log-shaped query should rank Code-type fix memories above other ErrorLog memories (near-duplicate noise).
6. Use `content_type` as a supersession key: a new Code-type memory for the same symbol/file supersedes stale Code memories, addressing the stale-propagation and contradiction-persistence failure modes named in the Governed Shared Memory paper.

### Why It Scales For Multi-Agent Fleets

Split embedding computation into per-content-type batched worker queues so a burst of code-heavy ingestion from one agent does not starve NL/decision embedding for other agents. Use `content_type` as the supersession key from the Governed Shared Memory paper, cutting stale-propagation and contradiction-persistence directly. Give ErrorLog-type memories a faster decay/consolidation cycle than Code or Decision-type memories, feeding into the Backfill/Reconsolidation cluster, since transient stack traces are noise once root-caused while the causal fix and decision behind it are durable. Keep this as a policy layer on one shared local SQLite store rather than a per-agent pool rewrite, fitting Vestige's local-first single-file architecture. All embedding models stay local via fastembed ONNX, so fleet scale does not add per-agent cloud embedding API cost.

### Effort: M

---

## 8. Causal-Anchored, Receipt-Verified Compression

### Current State

`crates/vestige-core/src/advanced/compression.rs` (743 LOC) groups old, semantically-similar memories, extracts "key facts" via naive sentence scoring, and produces a `CompressedMemory` summary with a self-reported "semantic fidelity" score. `MemoryCompressor` is instantiated as a `CognitiveEngine` field (`cognitive.rs:76,159`) but never referenced again. `rg -rn "\.compressor\b" crates/vestige-mcp/src` matches only that field declaration and instantiation. No MCP tool, no maintenance sweep, no dashboard route calls `.compress()`, `.can_compress()`, or `.find_compressible_groups()`. Old memories in a real Vestige store are never actually compressed; the storage/latency win described in the module docstring never happens.

### 2026 Landscape

mem0 claims up to 80% prompt token reduction via flat, importance-scored extraction, with no causal or provenance structure, a root-cause memory and a routine restatement compress under the same logic, and no verifiable fidelity check beyond internal scoring. Zep/Graphiti avoids lossy compression almost entirely by keeping a temporal knowledge graph with validity windows on every fact, scoring 63.8% vs. mem0's 49% on LongMemEval by not compressing away facts, at the cost of a huge footprint (600k+ reported tokens per conversation) and delayed post-ingestion retrieval. Letta's sleep-time compute runs a background thread during idle windows that parses recent logs, compresses insights, and commits to an isolated git branch to avoid lock contention, a good architectural precedent, but with no causal-graph awareness, no contradiction-boundary respect, and no cross-agent provenance tagging. Supermemory explicitly avoids deep graph structure, so nothing stops it from compressing away a root-cause trace along with routine noise. MemClaw (arXiv 2606.24535) is purpose-built for multi-tenant fleet memory governance and reconstructs depth-four derivation chains with correct writer identity, but it is a standalone governance layer bolted onto generic shared memory, not integrated with a causal-failure-tracing graph or a compression subsystem. Two cautionary results: Compress the Context, Keep the Commitments (arXiv 2605.17304) proposes replacing self-reported fidelity scores with measurable ones (Critical Atom Recall, round-trip recoverability); and a 480-evaluation study on coding agents (arXiv 2605.18854) found no condensation strategy significantly improved task quality, while LLM-based condensers increased token costs 24-94%, a caution against generic LLM-summarization compression.

### Differentiation Angle

"Causal-Anchored, Receipt-Verified Compression": never compress away a memory that is a causal-upstream root-cause anchor, protecting the exact traceability retroactive salience backfill depends on. Refuse to merge memories connected by a live contradiction edge instead of averaging them into mush. Replace the self-reported "semantic fidelity" score with an actual verifiable check, `verified_recall`, by re-running Vestige's existing hybrid embedding+keyword search against the compressed text and confirming it still retrieves every folded-in source memory ID. Tag every `CompressedMemory` with contributing agents' provenance so a coding fleet never loses "who found this."

### Concrete Build Plan

1. Add a `maintain(action="compress")` MCP tool path calling the already-implemented `find_compressible_groups()`/`compress()`, shipped together with the differentiators below rather than as naive summarization.
2. Modify grouping logic to exclude from any compressible group: any memory with outgoing causal edges (a root-cause anchor), and any group whose members span a live contradiction edge (defer those to the existing contradiction/dedup/suppress tools).
3. Replace the self-reported fidelity score with `verified_recall`: after producing a `CompressedMemory`, query it back through Vestige's existing hybrid search and confirm each source memory ID is still reachable in top-k. If verification fails, do not collapse the group; leave it uncompressed and emit a receipt explaining why. No new ML dependency, no LLM-judge call.
4. Add agent-provenance tracking to `CompressedMemory` and route every compression event through Vestige's existing receipt/audit mechanism ("N raw memories to 1 compressed memory, verified_recall=X, contributing_agents=[...], receipt_id=Y").
5. Gate the sweep behind an explicit opt-in maintenance flag, consistent with the "heavy background automation stays optional" rule in `vestige/CLAUDE.md`, running it as an idle/background sweep.
6. Surface compression receipts on an existing dashboard audit/receipts route so a compression decision can be inspected and undone, not just trusted.

### Why It Scales For Multi-Agent Fleets

Causal-edge and contradiction-edge exclusions mean a compression sweep can never silently erase the one memory that traces a regression to its root cause, precisely the "provenance collapse" and "contradiction persistence" failure modes the Governed Shared Memory paper names as unsolved for fleet memory, solvable cheaply here because the causal graph and contradiction detection already exist. Per-agent provenance tagging on `CompressedMemory` preserves "which agent found this" across compression, unlike flat-summarization systems that blend memories into a single blob. Because verification reuses Vestige's own already-built embedding+keyword search instead of an LLM-judge-per-pair design (contrast MemRefine, arXiv 2606.13177), the fidelity check adds no inference cost and no LLM-judge bottleneck, scaling to many concurrent agents without a cloud dependency. Running compression as an idle background sweep, mirroring Letta's isolated-branch precedent, avoids write contention with multiple agents hitting the same SQLite store concurrently.

### Effort: M

---

## Also Worth Noting

The remaining 30 findings from the codebase mapping. Each is either fully wired and healthy, has a specific dashboard-parity gap worth a small fix, or is intentionally out of scope for now. Grouped by cluster.

**MCP Tool Surface Thinness + Dashboard Parity**

1. **`context.rs` (Encoding Specificity Principle context-weighted retrieval).** Marked `(Deprecated)` in its own module doc at `context.rs:1`. Present since the v1.0.0 initial commit through 7 commits, most recently a capacity-overflow safety clamp, but never promoted into a unified tool and explicitly demoted out of `tools/list` (`server.rs:915-927`). No dashboard route references `context`/`match_context` anywhere. Note: the *concept* of encoding specificity is separately alive and well through `ContextMatcher` in `search_unified.rs` (see item 10 below); this specific standalone module is the orphaned duplicate.

2. **`tagging.rs` (manual-trigger MCP wrapper, distinct from the researched Synaptic Tagging module).** Also `(Deprecated)` in its own doc comment, same fate as `context.rs`: 5 commits, never consolidated, demoted from `tools/list`. The underlying `SynapticTaggingSystem` it wraps is still live and used automatically inside the real consolidation pipeline (`neuroscience/synaptic_tagging.rs`, `consolidation/phases.rs`, `storage/sqlite.rs`), so only this manual-trigger MCP wrapper is orphaned, not the mechanism itself (see feature 2 above).

3. **Emotional Memory (flashbulb encoding, mood-congruent retrieval).** Only 2 of its methods, `evaluate_content()` and `record_encoding()`, are called anywhere outside its own file, both from `consolidation/phases.rs`. `stability_multiplier()` and `mood_congruence_boost()`, the functions that would feed emotional signal into FSRS decay or search ranking, are dead code across the repo. No dashboard component references `is_flashbulb`/`flashbulb`/`emotional_valence`/`EmotionalMemory` anywhere in `apps/dashboard/src`.

4. **`chains.rs` (`MemoryChainBuilder` / reasoning-path BFS).** Genuinely wired into the `graph` unified MCP tool (actions `chain`/`associations`/`bridges`), not dead. But `apps/dashboard/src/lib/stores/api.ts`'s `explore()` helper is only ever called with `action='associations'` from `routes/(app)/explore/+page.svelte:202`; no dashboard code passes `action='chain'` or `'bridges'`, so the BFS reasoning-path and bridge-discovery logic has zero dashboard surface despite being agent-callable.

5. **`maintain.rs` (unified `maintain` tool: consolidate/dream/gc/importance_score/backup/export/restore).** Split maturity: `consolidate` and `dream` have dashboard surface via `settings/+page.svelte` and `dreams/+page.svelte`. The other 4 of 7 actions, `gc`, `backup`, `export`, `restore`, have zero dashboard route, zero handler, and zero router entry (confirmed by grep for `/api/gc`, `/api/backup`, `/api/export`, `/api/restore`, no matches). A user can only trigger garbage collection, DB backup, export, or restore via an MCP client.

6. **`health.rs` (`memory_health` retention backend for `memory_status view=retention`).** Unreachable from the dashboard: the `/stats` page's retention chart is powered by a completely separate, independently-written handler (`crates/vestige-mcp/src/dashboard/handlers.rs:1829` `retention_distribution`, `:1147` `health_check`) that recomputes distribution buckets from scratch via `storage.get_all_nodes()` rather than calling `storage.get_retention_distribution()`/`health::execute`. The concept is visible in the dashboard; this specific file contributes nothing to it.

7. **Cloud sync (`cloud_sync.rs` + `cloud_crypto.rs`).** `cloud-sync` is a default Cargo feature so it ships in official binaries, but `grep -rn cloud_sync\|VESTIGE_CLOUD` across `apps/dashboard/src` returns zero matches, and no MCP tool registers it. It also depends on a hosted blob service that is private and unpublished per project memory (`~/vestige-launch-private/vestige-cloud`, no code in this public repo). The crypto and conflict-resolution engineering is solid; there is no MCP tool, no dashboard, and the hosted server ships from a different, non-public repo.

8. **Portable archive format (`portable.rs`).** Only 1 commit. Consumed transitively by `cloud_sync.rs` and file sync in `sqlite.rs` (`merge_portable_table`, `sync_portable_archive`). No MCP tool or dashboard route touches it directly; it is intentionally low-level and CLI-only (`vestige sync`/`vestige portable`), not underdeveloped so much as correctly scoped.

9. **`contradictions.rs` (trust-weighted contradiction detection).** Live backend for `recall`'s contradictions mode, but the dashboard's real, working `/contradictions` route (585 LOC, WebGPU `ContradictionArcs` visualization, calling `api.contradictions()`) hits `crates/vestige-mcp/src/dashboard/handlers.rs:2735 list_contradictions`, which re-imports the same cross-reference helpers (`appears_contradictory`, `compute_trust`, `topic_overlap`) and reimplements the pairing loop inline rather than calling `tools::contradictions::execute`. Genuinely live with real data, just duplicated logic.

10. **`graph_unified.rs` (unified `graph` tool: chain/associations/bridges/predict/memory_graph/composed_graph).** The current canonical, most actively maintained graph tool (12 actions, full test coverage), with real dashboard parity (`/graph`, `/explore`, `/observatory` routes), but via a separate HTTP path: `handlers::get_graph` (`crates/vestige-mcp/src/dashboard/handlers.rs:1213`) calls `storage.get_memory_subgraph` directly instead of `tools::graph_unified::execute`. Same core capability, parallel plumbing.

11. **`memory_status.rs` (unified `memory_status` tool: health/retention/timeline/changelog).** Real `/stats` page (439 LOC) and `/timeline` route exist, but through dedicated `dashboard/handlers.rs` functions (`get_stats`, `retention_distribution`, `health_check`, `get_timeline`, `get_changelog`) that reimplement the same math directly against `Storage`, not through this facade.

12. **`importance.rs` (backend for `maintain action=importance_score`, the neuroscience 4-channel tool, distinct from the researched `ImportanceTracker`).** Real interactive scorer at `routes/(app)/importance/+page.svelte` (609 LOC), but it hits `dashboard/handlers.rs:1731 score_importance`, which independently calls `cog.importance_signals.compute_importance` directly plus a no-cognitive-engine fallback heuristic, a parallel implementation rather than a call into this file.

13. **Merge/Supersede control tools (`merge.rs` + `dedup.rs` wrapper).** MCP tool layer is complete and fully wired end-to-end. The gap is dashboard-only and deliberate: `DuplicateCluster.svelte` renders a visibly disabled "Merge unavailable" button with an inline comment stating merge is "NOT wired to a backend yet" and the control is "visibly disabled (never lies)." `dashboard/mod.rs`'s route table exposes only `/api/duplicates` (read-only scan), no `/api/merge*` route. Honest, deliberately-disabled UI, not silently broken, but invisible to dashboard users despite being production-ready via MCP.

14. **Suppress (active forgetting HTTP surface).** Fully wired end-to-end: MCP tool, REST route (`dashboard/mod.rs` POST `/api/memories/{id}/suppress` and `/unsuppress`), dashboard client (`api.ts`), and a live UI call site in `routes/(app)/memories/+page.svelte:344,356`. `dashboard/mod.rs` lines 141-151 explicitly comment this gap was deliberately closed in v2.0.7. Also feeds the Black Box observatory's "suppress" trace lane.

**Neuroscience-Inspired Memory Mechanisms (fully wired, listed for completeness)**

15. **Active Forgetting (top-down inhibitory control).** Fully wired: `ActiveForgettingSystem` is called from `suppress.rs`, `search_unified.rs` (score adjustment), and `dashboard/handlers.rs` (2 sites). The `suppress` MCP tool is one of the 13 top-level v2.3 tools, and the dashboard has a dedicated `ForgettingIndicator.svelte` plus `forgetting-plan.ts` and `rescue-plan.ts` in the Observatory layer.

16. **Context-Dependent Memory (Encoding Specificity Principle, live implementation).** `cognitive.rs:55` holds `context_matcher: ContextMatcher` on `CognitiveEngine`. `search_unified.rs:614-645` builds an `EncodingContext` from the query, computes a context-score boost, and produces temporal/topical/session hints included in the response (`search_unified.rs:860-861`). Backs the top-level `recall` tool.

17. **Multi-Channel Importance Signaling (4-channel novelty/arousal/reward/attention).** The most fully wired mechanism in the cluster end-to-end: `learn_content()` called in `smart_ingest.rs`, `compute_importance()` dispatched from `maintain`, and a dedicated 609-line dashboard route (`routes/(app)/importance/+page.svelte`) rendering live salience.

18. **Memory States (accessibility continuum, Retrieval-Induced Forgetting).** `cog.competition_mgr.run_competition()` called in `search_unified.rs:671`. Dashboard has a dedicated `MemoryStateLegend.svelte` plus state-aware rendering in `Graph3D.svelte`, `graph/nodes.ts`, and `/dreams`.

19. **Neuroscience module aggregator (re-export surface).** Pure plumbing file; no logic of its own to be dead or alive.

20. **Predictive Memory Retrieval (Friston Free Energy / Active Inference).** `record_query`/`record_memory_access` called from `autopilot.rs` and `search_unified.rs` on every `recall`. `predict.rs`/`session_context.rs` call the read methods; the `predict` action is folded into the unified `graph` tool and separately exposed to the dashboard via `api.ts:118`, consumed by the importance page.

21. **Prospective Memory (Einstein & McDaniel intentions).** `check_triggers()` polled every 60s by a background task in `autopilot.rs`. The `intention` MCP tool is one of the 13 top-level v2.3 tools, backed by `intention_unified.rs`, with a dedicated `routes/(app)/intentions/+page.svelte`. Note: persistence for intention CRUD actually goes through `Storage::get_active_intentions()`, a separate storage-layer implementation, rather than `ProspectiveMemory`'s own in-memory store, so the in-memory struct is used mainly for live context-monitoring.

22. **Spreading Activation (Collins & Loftus semantic network).** Used pervasively: `.add_edge()` in `cognitive.rs` and `dream.rs`, `.activate()` in `autopilot.rs`, `cross_reference.rs`, and `search_unified.rs`. Dashboard has a dedicated `ActivationNetwork.svelte` plus `activation-helpers.ts`.

**Advanced Cognition: Backfill, Reconsolidation, Prediction, Speculation, Dreams (fully wired, listed for completeness)**

23. **`cross_project.rs` (`CrossProjectLearner`, cross-codebase pattern transfer).** Correctly fully wired overall via `/api/patterns/cross-project` and `patterns/+page.svelte`, but has an internal logic bug: `detect_applicable()` (used by the `codebase get_context` MCP action) is always called with a near-empty `ProjectContext` (`codebase_unified.rs:341-353`), only `name` is populated; `path`, `languages`, `frameworks`, `file_types`, `dependencies`, and `structure` are all left empty. Since `check_trigger()` only matches on `file_types`/`dependencies`/`structure`, and hard-codes `Some((false, ...))` for `TriggerType::Topic`, the only trigger type any pattern is ever registered with (`cross_project.rs:621`), `detect_applicable()` always returns an empty Vec through the MCP `get_context` path, a separate dead sub-path from the genuinely-working dashboard `/patterns` route.

24. **`dreams.rs` (`MemoryDreamer`/`ConsolidationScheduler`).** Second flagship feature alongside retroactive backfill: `dream.rs::execute()` persists insight/connection/history records and hydrates the activation network; `routes/(app)/dreams/+page.svelte:163` calls `api.dream()`. Genuinely mature end-to-end.

25. **`merge_supersede.rs` (Fellegi-Sunter duplicate scoring).** Wired into the real production merge/dedupe pipeline via multiple call sites in `sqlite.rs`, exposed through the `merge` MCP tool, and surfaced on the dashboard's `/duplicates` page (read side; write side is item 13 above).

26. **`prediction_error.rs` (`PredictionErrorGate`, ingest gating).** Flagship, mature: `Storage::smart_ingest`/`smart_ingest_excluding` (`sqlite.rs:778-851+`) instantiate the gate and call `.evaluate()` on every smart-ingest write, the code path behind Vestige's primary memory-write tool.

27. **`reconsolidation.rs` (`ReconsolidationManager`, Nader's labile-window editing).** Fully wired real behavioral loop: every search result marks the retrieved memory labile (`search_unified.rs:782`), promote/demote feedback applies modifications while labile, and `server.rs` periodically closes expired windows. Dashboard's `MemoryAuditTrail` has a dedicated "reconsolidated" event glyph.

28. **`retroactive_backfill.rs` (Retroactive Salience Backfill).** Vestige's flagship mechanic, wired into the `backfill` MCP tool which persists an explicit `backfill_candidate` evidence edge before any promotion, surfaced on the Observatory's "salience-rescue" demo scene.

29. **`speculative.rs` (`SpeculativeRetriever`, pre-fetch prediction).** Genuinely wired overall (3 of 4 channels live: query-similarity, co-access, temporal), but the file-context channel (`predict_from_files`) is dead in practice: its only data source, `file_memory_map`, is populated exclusively via `record_access(memory_id, file_context, ...)`, and the single production call site (`search_unified.rs:765`) always passes `file_context: None`. No caller reports "this file is open" as real IDE/editor context, so `file_memory_map` stays permanently empty.

**Storage/Sync Layer Maturity For Scale**

30. **Storage/Sync duplicates: `dedup.rs` wrapper.** Covered under item 13 above alongside `merge.rs`; the same "backend complete, dashboard write-path deliberately disabled" story applies to the dedup MCP surface, not a separate gap.

---

## Recommended Build Order

The build order follows the causal graph, not just the priority ranking. Almost every feature above either strengthens the causal graph or consumes its richness; sequence the strengtheners first.

**Phase 1: Strengthen the causal graph (weeks 1-3).**
Start with Codebase-Awareness Module wiring (rank 1). This is the largest already-built asset and the one whose payoff compounds: every co-edit relationship and bug-fix pattern it surfaces becomes a new causal edge that every downstream feature below can consume. Run Synaptic Tagging & Capture (rank 2) in parallel; it is the smallest effort item and its causal-gating requirement forces early clarity on how causal-edge queries should be exposed internally, infrastructure the later phases reuse. Do not sequence these after anything else: everything downstream is more valuable with a richer, receipt-backed causal graph underneath it.

**Phase 2: Build consumers of causal richness (weeks 3-6).**
Hippocampal Indexing (rank 3) comes next because `update_importance_flags()` needs a real source of causal-backfill-driven salience boosts to be worth wiring, which phase 1 now provides. Wire the pre-filter stage first (it has independent latency/precision value even before phase 1 lands, and fixing the `prune_weak_links()` no-op bug is a same-day win alongside it), then wire the backfill-to-`ImportanceFlags` connection once Codebase-Awareness and Synaptic Tagging are emitting richer causal edges. ImportanceTracker's causal-importance channel (rank 5) is a natural follow-on here too: it is a small, mostly-independent fix (unblock the maintenance-sweep no-op bug immediately, it costs nothing and can land day one), but the `causal_importance` differentiator specifically needs backfill-confirmed causal edges to be meaningful, so treat the bug fix as phase 1 and the causal-confirmation logic as phase 2.

**Phase 3: The fleet-scale moat feature (weeks 6-10).**
Shared Write Bus / causal-arbitrated write coordination (rank 4) is the biggest strategic payoff and the only item requiring substantial net-new code. Sequence it last among the causal-graph-dependent work because its core mechanism, tracing two competing claims forward to see which one accumulates a causal edge to a validated outcome, is far more reliable once Codebase-Awareness and Synaptic Tagging have been feeding the causal graph for weeks and Hippocampal Indexing's provenance fields (writer/session id on `MemoryBarcode`) are already in the schema, since the write bus's own `writer_id` migration can reuse that groundwork instead of duplicating it.

**Phase 4: Decoupled wins, any order (parallelizable throughout).**
IntentDetector (rank 6), content-type-routed embedding (rank 7), and causal-anchored compression (rank 8) do not block or get blocked by phases 1-3 in any hard sense, so they can be picked up by a second workstream in parallel with phases 1-3, or slotted into gaps. One soft dependency worth respecting: land compression (rank 8) after Codebase-Awareness and Synaptic Tagging (phase 1) even though it is not required, because compression's causal-edge exclusion rule is only as good as the causal graph it is checking against; running compression before the graph is enriched risks compressing memories that would have been protected as root-cause anchors once phase 1 lands. IntentDetector and adaptive embedding have no such dependency and can start immediately.

**What not to build yet.** Cloud sync (also-worth-noting item 7) requires a hosted service that lives outside this repo; do not invest further engineering here until the private `vestige-cloud` service has a public-facing plan. The Merge/Supersede dashboard button (item 13) is a small, well-scoped follow-up (wire `/api/merge*` REST routes to the already-complete MCP tool) worth picking up opportunistically but does not compete for roadmap priority against the 8 features above.