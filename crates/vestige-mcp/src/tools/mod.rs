//! MCP Tools
//!
//! Tool implementations for the Vestige MCP server.
//!
//! 4.x advertised surface is 16 tools: the v2.2 consolidated 12 plus the
//! controlled `receipt` surface and the flagship `backfill` primitive. The
//! unified facade modules (recall, dedup, memory_status, graph_unified, maintain, plus
//! the earlier *_unified) dispatch on an action/mode/view discriminator and
//! delegate to the granular handler modules below, which stay in the crate as
//! the implementation layer and as hidden back-compat aliases (see the redirect
//! arms in server.rs).

// Wire-budget compaction for tools/list (#212); full schemas stay
// available through memory_status view='tools'.
pub mod compact;

// Active unified tools
mod code_context;
pub mod codebase_unified;
pub mod intention_unified;
pub mod memory_unified;
pub mod search_unified;

// v2.2: Unified retrieval surface — folds search + deep_reference +
// cross_reference + contradictions into one mode-dispatched tool.
// mode=lookup (default) is a zero-overhead pass-through to search_unified.
pub mod recall;
pub mod receipt;
pub mod smart_ingest;
// #57: external-source connectors (GitHub Issues / Redmine retrieval layer).
// The tool is absent from tools/list and dispatch unless this feature is on.
#[cfg(feature = "connectors")]
pub mod source_sync;

// v1.2: Temporal query tools
pub mod changelog;
pub mod timeline;

// v1.2: Maintenance tools
pub mod maintenance;

// v2.2: Unified maintenance surface — folds consolidate + dream + gc +
// importance_score + backup + export + restore into one action-dispatched tool.
pub mod maintain;

// v2.2: Unified status surface — folds system_status + memory_health +
// memory_timeline + memory_changelog into one view-dispatched tool.
pub mod hygiene_stats;
pub mod memory_status;

// One shape for capabilities this build does not have (never zero-success).
pub mod unavailable;

// `codebase action=ingest_repo`: the repo connector (commits -> change records).
pub mod repo_ingest;

// v1.3: Auto-save and dedup tools
pub mod dedup;
pub mod importance;

// v2.1.25: Merge / Supersede controls (Phase 3)
pub mod merge;

// v1.5: Cognitive tools
pub mod dream;
// v2.3: The 4-phase DreamEngine wired to durable state — review-gated
// (every proposed change lands as a Memory PR; no autonomous memory writes).
pub mod dream_compile;
pub mod explore;
pub mod predict;
pub mod restore;

// v1.8: Context Packets
pub mod session_context;

// v1.9: Autonomic tools
pub mod graph;
pub mod health;

// v2.2: Unified graph surface — folds explore_connections + predict +
// memory_graph + composed_graph into one action-dispatched tool.
// 4.0: a hidden alias of `ghostlink`.
pub mod graph_unified;

// 4.0: GhostLink, the advertised composition surface (propose with the
// bridge and divergent lenses, bounty, weave, map, inspect, explore,
// predict, harden) over recorded structure only.
pub mod ghostlink;

// v2.1: Cross-reference (connect the dots)
pub mod composed_graph;
pub mod contradictions;
pub mod cross_reference;
pub(crate) mod lookup_packet;

// v2.0.5: Active Forgetting — Anderson 2025 + Davis Rac1
pub mod suppress;

// Blast Radius — exact downstream reach of a cause/source record
pub mod blast_radius;

// Retroactive Salience Backfill — Cai 2024 Nature (memory with hindsight).
// v3.2: superseded as the advertised flagship by `causal_walk` below; still
// dispatched as a hidden back-compat alias.
pub mod backfill;

// Causal Walk — the successor to backfill: explicit start points (failing
// test, stack frame, CI run, logged write, version range) walked through
// exact mechanism edges to the change records behind a failure.
pub mod causal_walk;

// w3d: planted-cause self-calibration + decayed-lesson detection, both
// built on the backfill surface above.
pub mod forgotten_lesson;
pub mod selftest;

// Internal/backwards-compat tools still dispatched by server.rs for specific
// tool names. Each module below has live callers via string dispatch in
// `server.rs` (match arms on request.name).
//
// The nine legacy siblings here pre-v2.0.8 (checkpoint, codebase, consolidate,
// ingest, intentions, knowledge, recall, search, stats) were removed in the
// post-v2.0.8 dead-code sweep — all nine had zero callers after the
// unification work landed `*_unified` + `maintenance::*` replacements.
pub mod feedback;
pub mod memory_states;
pub mod review;
pub mod tagging;

/// Evidence-aware intention command adapter.
pub mod intention_graph;

pub mod project;

/// Ignored real-repo proof. `cargo test --workspace` does not run it.
#[cfg(test)]
mod causal_git_proof;

/// Catalog gaps for causal_walk and ingest_repo. Passing tests cover the
/// slice landed with the mapping doc. `#[ignore]` tests name the high-priority
/// gaps whose facts are already on disk.
#[cfg(test)]
mod blind_spots;
