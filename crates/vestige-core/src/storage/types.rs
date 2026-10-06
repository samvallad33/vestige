//! Shared storage type definitions — available in EVERY build mode.
//!
//! The `LocalMemoryStore` trait (and the MCP surface above it) names these
//! types in method signatures, so the DEFINITIONS must exist with or without
//! the `legacy-sqlite` feature (dual-mode compilation, strata/fix-00a).
//! Only the type definitions live here; every implementation (inherent impls,
//! rusqlite-backed constructors, parsing helpers) stays in the module that
//! owns it. The `legacy-sqlite` backend modules re-export their former
//! definitions from here so all existing import paths keep resolving.
//!
//! The one backend-coupled edge is [`StorageError::Database`]: it carries the
//! backend error as a `String` so no `rusqlite` type leaks into a
//! feature-free build. The legacy backend provides the typed
//! `From<rusqlite::Error>` conversion (see `sqlite::mod`).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::memory::KnowledgeNode;
use crate::neuroscience::{DecayFunction, SynapticTag};
use crate::storage::receipt_attestation::{
    DisclosureMapping, ReceiptAttestationV1, SignedReceiptAttestation, VerificationReport,
};
use crate::trace::{MemoryPr, Receipt};

// ----------------------------------------------------------------------------
// ERROR
// ----------------------------------------------------------------------------

/// Storage error type
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    /// Database error (stringified backend error; the legacy backend
    /// converts the typed `rusqlite::Error` through `From`).
    #[error("Database error: {0}")]
    Database(String),
    /// Node not found
    #[error("Node not found: {0}")]
    NotFound(String),
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Invalid timestamp
    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(String),
    /// Initialization error
    #[error("Initialization error: {0}")]
    Init(String),
    /// A likely credential was detected before any write side effect.
    #[error(
        "Refused to store probable credential(s): {kinds:?}. Secret bytes were not stored, logged, or returned. Redact the value or use an explicit allow-secrets override only when intentional."
    )]
    SecretDetected { kinds: Vec<String> },
    /// A project namespace must be a short, non-empty identifier.
    #[error("Invalid memory scope: {0}")]
    InvalidScope(String),
    /// A typed-edge write or traversal referenced a link type outside the
    /// owner-approved vocabulary, or otherwise malformed edge input.
    #[error("Invalid typed edge: {0}")]
    InvalidEdge(String),
    /// A profile operation would violate the explicit/reversible embedding
    /// profile contract.
    #[error("Invalid embedding profile: {0}")]
    InvalidEmbeddingProfile(String),
    /// A v3 SQLite store was found where a 4.0 Strata directory is required.
    /// The original file was never opened read-write, migrated in place, or
    /// modified in any way; the caller must migrate it first.
    #[error(
        "v3 SQLite store at {path} (schema version {schema_version}) cannot be opened by 4.0. hint: {hint}"
    )]
    V3StoreNeedsMigration {
        /// Path of the refused SQLite file (display form).
        path: String,
        /// Highest `schema_version` row in the store; 0 when unreadable.
        schema_version: u32,
        /// Actionable next step for the operator.
        hint: String,
    },
}

/// Storage result type
pub type Result<T> = std::result::Result<T, StorageError>;

// ----------------------------------------------------------------------------
// HYGIENE + TAGS
// ----------------------------------------------------------------------------

/// Content-bounded row used to compute full-store hygiene statistics without
/// loading every memory body or issuing per-memory access-log queries.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HygieneNodeSummary {
    pub id: String,
    pub node_type: String,
    pub created_at: DateTime<Utc>,
    pub retention_strength: f64,
    pub tags: Vec<String>,
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_until: Option<DateTime<Utc>>,
    pub superseded: bool,
    pub content_bytes: usize,
    pub content_preview: String,
    /// No access evidence exists AND the memory was created inside the
    /// retained access-log window, so the absence of log rows is meaningful.
    pub never_accessed: bool,
    /// No access evidence exists but the memory predates the retained
    /// access-log window: pruning makes past access unknowable, so this row
    /// must never be claimed as never-accessed.
    pub access_unknown: bool,
}

/// Full hygiene population plus row-corruption findings. Malformed rows are
/// tolerated (mirroring `row_to_node`) and reported instead of aborting the
/// whole stats view, because hand-edited stores are exactly where hygiene
/// tooling is needed most.
#[derive(Debug, Clone)]
pub struct HygieneSnapshot {
    pub nodes: Vec<HygieneNodeSummary>,
    /// Rows whose stored `tags` column is NULL or unparseable JSON; their
    /// tags are treated as empty in `nodes`.
    pub malformed_tag_rows: usize,
    /// Capped id list for the malformed rows (first
    /// [`MAX_MALFORMED_TAG_ROW_IDS`] in id order).
    pub malformed_tag_row_ids: Vec<String>,
    pub malformed_tag_row_ids_truncated: bool,
    /// Rows whose nullable `retention_strength` was NULL and fell back to the
    /// schema default of 1.0.
    pub defaulted_retention_rows: usize,
}

/// Exact tag vocabulary for one scope plus the count of stored tags that were
/// skipped because they exceed the 200-character similarity safety limit.
/// Overlong stored tags degrade gracefully (skip-and-count) instead of
/// disabling suggestions for the whole scope.
#[derive(Debug, Clone)]
pub struct TagVocabulary {
    pub tags: Vec<String>,
    pub skipped_overlong: usize,
}

// ----------------------------------------------------------------------------
// WAL CHECKPOINT
// ----------------------------------------------------------------------------

/// SQLite WAL checkpoint mode exposed for explicit lifecycle operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalCheckpointMode {
    /// Checkpoint as many frames as possible without blocking active readers.
    Passive,
    /// Checkpoint and truncate the WAL after application writes have stopped.
    Truncate,
}

/// Raw `wal_checkpoint` counters reported by SQLite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WalCheckpointStatus {
    pub busy: i64,
    pub log_frames: i64,
    pub checkpointed_frames: i64,
}

// ----------------------------------------------------------------------------
// SMART INGEST
// ----------------------------------------------------------------------------

/// Result of smart ingest with prediction error gating
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmartIngestResult {
    /// Decision made: "create", "update", "supersede", "merge", "reinforce", etc.
    pub decision: String,
    /// The resulting node (new or updated)
    pub node: KnowledgeNode,
    /// ID of superseded memory (if any)
    pub superseded_id: Option<String>,
    /// Similarity to closest existing memory (0.0 - 1.0)
    pub similarity: Option<f32>,
    /// Prediction error (1.0 - similarity)
    pub prediction_error: Option<f32>,
    /// Human-readable explanation of the decision
    pub reason: String,
    /// Previous content when smart ingest mutated an existing memory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_content: Option<String>,
    /// Existing memory id that received merged or appended content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merged_from: Option<String>,
    /// Full updated content after a merge/append/context write.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge_preview: Option<String>,
    /// World-time close stamped onto a newly created dated claim that is
    /// already superseded by a currently-valid fact starting later.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_closed_until: Option<DateTime<Utc>>,
    /// Set when the write path conflicted with a memory inside its labile
    /// window and routed the conflict through a reconsolidation merge plan
    /// instead of mutating immediately (`decision == "reconsolidation_pending"`
    /// for a deferred supersede; a plain `"create"` for a contradiction that
    /// was stored separately and linked to a verdict plan).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reconsolidation_plan_id: Option<String>,
}

// ----------------------------------------------------------------------------
// PORTABLE SYNC + PURGE
// ----------------------------------------------------------------------------

/// Summary of a pull-merge-push sync operation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PortableSyncReport {
    /// Backend label that was synced.
    pub backend: String,
    /// Whether an existing remote archive was pulled before pushing.
    pub pulled: bool,
    /// Merge report from the pull phase, if a remote archive existed.
    pub pull: Option<PortableImportReport>,
    /// Number of tables written to the backend during push.
    pub pushed_tables: usize,
    /// Number of rows written to the backend during push.
    pub pushed_rows: usize,
    /// Portable archive format written during push.
    pub archive_format: String,
}

/// Report returned by an irreversible content purge.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PurgeReport {
    /// Memory ID requested for purge.
    pub memory_id: String,
    /// Whether a live memory row was found and removed.
    pub deleted: bool,
    /// Non-content tombstone timestamp.
    pub deleted_at: DateTime<Utc>,
    /// Number of graph edges removed by foreign-key cascade.
    pub edges_pruned: i64,
    /// Number of insight rows whose source list was rewritten.
    pub insights_rewritten: i64,
    /// Number of insight rows dropped because fewer than two source memories remained.
    pub insights_deleted: i64,
    /// Number of temporal-summary children detached from this parent.
    pub children_orphaned: i64,
    /// This established purge path audits legacy local cleanup only.  It does
    /// not claim the post-V25 lineage coverage required for verified local
    /// machine unlearning.
    pub unlearning_scope: crate::storage::UnlearningScope,
    /// Legacy purge is intentionally never labeled `VerifiedWithinScope`.
    pub unlearning_verdict: crate::storage::UnlearningVerdict,
    /// Fixed boundary shown by MCP callers rather than a free-form guarantee.
    pub unlearning_claim_boundary: &'static str,
    /// `eff-` receipt when a named `purge` retire was admitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt_id: Option<String>,
}

// ----------------------------------------------------------------------------
// FAILURE FEEDBACK
// ----------------------------------------------------------------------------

/// What one post-retrieval failure feedback pass did.
#[derive(Debug, Clone, serde::Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FailureFeedbackReport {
    pub failure_id: String,
    pub window_minutes: i64,
    pub receipts_considered: usize,
    pub memories_demoted: usize,
    pub total_delta: f64,
}

// ----------------------------------------------------------------------------
// PERSISTENCE LAYER: Intentions, Insights, Connections, States
// ----------------------------------------------------------------------------

/// Intention data for persistence (matches the intentions table schema)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IntentionRecord {
    pub id: String,
    pub content: String,
    pub trigger_type: String,
    pub trigger_data: String, // JSON
    pub priority: i32,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub deadline: Option<DateTime<Utc>>,
    pub fulfilled_at: Option<DateTime<Utc>>,
    pub reminder_count: i32,
    pub last_reminded_at: Option<DateTime<Utc>>,
    pub notes: Option<String>,
    pub tags: Vec<String>,
    pub related_memories: Vec<String>,
    pub snoozed_until: Option<DateTime<Utc>>,
    pub source_type: String,
    pub source_data: Option<String>,
    /// Project namespace. `None` (legacy rows) resolves to the `user`
    /// namespace; see `effective_scope`. Prospective surfacing in recall only
    /// ever reads intentions whose effective scope equals the query scope.
    pub scope: Option<String>,
}

impl IntentionRecord {
    /// Normalized namespace: blank or `None` is `user`.
    pub fn effective_scope(&self) -> &str {
        match self.scope.as_deref() {
            Some(scope) if !scope.trim().is_empty() => scope.trim(),
            _ => "user",
        }
    }
}

/// Insight data for persistence (matches the insights table schema)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InsightRecord {
    pub id: String,
    pub insight: String,
    pub source_memories: Vec<String>,
    pub confidence: f64,
    pub novelty_score: f64,
    pub insight_type: String,
    pub generated_at: DateTime<Utc>,
    pub tags: Vec<String>,
    pub feedback: Option<String>,
    pub applied_count: i32,
}

/// Memory connection for activation network
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConnectionRecord {
    pub source_id: String,
    pub target_id: String,
    pub strength: f64,
    pub link_type: String,
    pub created_at: DateTime<Utc>,
    pub last_activated: DateTime<Utc>,
    pub activation_count: i32,
}

/// State transition record for audit trail
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StateTransitionRecord {
    pub id: i64,
    pub memory_id: String,
    pub from_state: String,
    pub to_state: String,
    pub reason_type: String,
    pub reason_data: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// Consolidation history record
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConsolidationHistoryRecord {
    pub id: i64,
    pub completed_at: DateTime<Utc>,
    pub duration_ms: i64,
    pub memories_replayed: i32,
    pub connections_found: i32,
    pub connections_strengthened: i32,
    pub connections_pruned: i32,
    pub insights_generated: i32,
}

/// Dream history record — persists dream metadata for automation triggers
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DreamHistoryRecord {
    pub dreamed_at: DateTime<Utc>,
    pub duration_ms: i64,
    pub memories_replayed: i32,
    pub connections_found: i32,
    pub insights_generated: i32,
    pub memories_strengthened: i32,
    pub memories_compressed: i32,
    // v2.0: 4-Phase dream cycle metrics
    pub phase_nrem1_ms: Option<i64>,
    pub phase_nrem3_ms: Option<i64>,
    pub phase_rem_ms: Option<i64>,
    pub phase_integration_ms: Option<i64>,
    pub summaries_generated: Option<i32>,
    pub emotional_memories_processed: Option<i32>,
    pub creative_connections_found: Option<i32>,
}

// ----------------------------------------------------------------------------
// COMPOSITION GRAPH
// ----------------------------------------------------------------------------

/// Composition event envelope for ComposedGraph.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompositionEventRecord {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub tool: String,
    pub mode: String,
    pub query: Option<String>,
    pub query_hash: Option<String>,
    pub confidence: Option<f64>,
    pub status: Option<String>,
    pub output_preview: Option<String>,
    pub metadata: serde_json::Value,
}

/// Memory participating in a composition event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompositionMemberRecord {
    pub event_id: String,
    pub memory_id: String,
    pub role: String,
    pub rank: i32,
    pub trust: Option<f64>,
    pub score: Option<f64>,
    pub preview: Option<String>,
    pub metadata: serde_json::Value,
}

/// Outcome label attached to a composition event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompositionOutcomeRecord {
    pub id: String,
    pub event_id: String,
    pub outcome_type: String,
    pub labeled_at: DateTime<Utc>,
    pub label_source: String,
    pub confidence_delta: Option<f64>,
    pub notes: Option<String>,
    pub metadata: serde_json::Value,
}

/// Memory most often composed with another memory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompositionNeighborRecord {
    pub memory_id: String,
    pub composed_count: i64,
    pub latest_event_at: DateTime<Utc>,
}

/// Candidate memory pair that shares useful shape but has never been composed.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeverComposedCandidate {
    pub first_id: String,
    pub second_id: String,
    pub score: f64,
    pub novelty_score: f64,
    pub bridge_score: f64,
    pub trust_score: f64,
    pub outcome_score_adjustment: f64,
    pub shared_tags: Vec<String>,
    pub boundary_tags: Vec<String>,
    pub shared_terms: Vec<String>,
    pub prior_outcomes: Vec<String>,
    pub outcome_signal: String,
    pub first_node_type: String,
    pub second_node_type: String,
    pub first_preview: String,
    pub second_preview: String,
    pub reason: String,
    pub composition_question: String,
}

// ----------------------------------------------------------------------------
// ACTOR PROVENANCE
// ----------------------------------------------------------------------------

/// One recorded actor endorsement, bound to the exact content revision it
/// supported. Mirrors a row of `actor_endorsement_events` (migration V38).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct EndorsementEventRecord {
    pub event_id: String,
    pub memory_id: String,
    pub actor_did: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claimed_role: Option<String>,
    pub effective_role: String,
    pub resolved_weight: f64,
    pub resolution_disposition: String,
    pub policy_version: u64,
    /// `support`, `oppose`, or `self_support`.
    pub endorsement_kind: String,
    /// SHA-256 hex of the exact content revision this stance binds to.
    pub revision_digest: String,
    /// The prior this event contributes to Phase B aggregation. Exactly 0.0
    /// for self-support; same-actor retries do not add rows.
    pub independent_prior: f64,
    pub tool: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub receipt_id: Option<String>,
    pub created_at: String,
}

/// Outcome of one actor-attributed mutation: the mutated node, its receipt
/// (persisted in the same transaction), and the endorsement event.
#[derive(Debug, Clone)]
pub struct ActorMutationOutcome {
    /// The node state before the mutation.
    pub before: KnowledgeNode,
    /// The node state after the mutation committed.
    pub node: KnowledgeNode,
    /// The receipt persisted inside the same transaction.
    pub receipt: Receipt,
    /// The endorsement event recorded inside the same transaction.
    pub endorsement: EndorsementEventRecord,
    /// True when this actor had already recorded the same stance on the same
    /// revision: the mutation applies, but no second vote was created.
    pub already_recorded: bool,
}

// ----------------------------------------------------------------------------
// CONNECTOR SYNC (#57) — idempotent external-source ingestion
// ----------------------------------------------------------------------------

/// What `upsert_by_source` did with one external record. Drives the
/// created/updated/unchanged/tombstoned counts a connector reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceUpsertOutcome {
    /// No memory existed for this `(source_system, source_id)` — inserted.
    Created,
    /// A memory existed and the `content_hash` changed — body + envelope updated
    /// and the embedding regenerated.
    Updated,
    /// A memory existed with the same `content_hash` — nothing rewritten except
    /// `synced_at` (so an incremental re-scan is free).
    Unchanged,
}

/// Result of one `upsert_by_source` call.
#[derive(Debug, Clone)]
pub struct SourceUpsertResult {
    pub outcome: SourceUpsertOutcome,
    /// Memory id of the affected node (new or existing).
    pub node_id: String,
}

/// Incremental-sync checkpoint for one `(source_system, scope)`.
#[derive(Debug, Clone, Default)]
pub struct ConnectorCursor {
    pub source_system: String,
    pub scope: String,
    /// High-water mark on the source's update timestamp. `None` on first sync.
    pub cursor_updated_at: Option<DateTime<Utc>>,
    pub last_synced_at: Option<DateTime<Utc>>,
    pub last_full_reconcile_at: Option<DateTime<Utc>>,
    pub records_seen: i64,
}

/// Outcome of a tombstone reconciliation pass.
#[derive(Debug, Clone, Default)]
pub struct ReconcileReport {
    /// Memory ids that were tombstoned (no longer visible upstream).
    pub tombstoned: Vec<String>,
    /// Number of local records considered for this scope.
    pub considered: usize,
}

// ----------------------------------------------------------------------------
// BLAST RADIUS
// ----------------------------------------------------------------------------

/// One record inside a blast report.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BlastAffected {
    /// The affected memory id.
    pub id: String,
    /// How it was reached: "root", "shared_commit:<sha>", or the edge
    /// link_type ("derived_from" | "backfill_candidate" | "evidence_of").
    pub via: String,
    /// BFS depth from the root. Root and shared-sha siblings sit at 0;
    /// direct edge targets at 1; capped at [`BLAST_MAX_DEPTH`].
    pub depth: u32,
}

/// Exact downstream reach of one root memory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlastReport {
    /// The root (cause/source) memory id the traversal started from.
    pub root_id: String,
    /// Root itself (depth 0), shared-sha siblings (depth 0), then edge
    /// descendants by ascending (depth, id). Sorted deterministically.
    pub affected: Vec<BlastAffected>,
    /// `affected.len()`.
    pub total: usize,
}

/// Per-id outcome of a retire pass. Retire NEVER deletes: each id is
/// suppressed through the existing storage mechanism, so the row survives,
/// the suppression is journaled, and the 24h reversal window applies.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RetireOutcome {
    pub id: String,
    /// true when the suppression flip was applied.
    pub suppressed: bool,
    /// Suppression count after the flip (0 when it failed).
    pub suppression_count: i32,
    /// Human-readable failure reason ("not found", storage error, ...).
    pub error: Option<String>,
}

// ----------------------------------------------------------------------------
// HANDLE RESOLUTION
// ----------------------------------------------------------------------------

/// Hard cap on ambiguous-prefix candidates reported per resolution.
pub const MAX_CANDIDATES: usize = 20;

/// The canonical detail string for the `handle_required` error payload.
pub const HANDLE_REQUIRED_DETAIL: &str = "recall is handle-based: pass a memory id, commit sha, file, symbol, test, run, or tool-call id";

/// What kind of handle a query resolved as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HandleKind {
    Memory,
    Commit,
    File,
    Symbol,
    Test,
    Run,
    ToolCall,
    Tag,
    /// No handle rule matched (unresolved query).
    Unknown,
}

impl HandleKind {
    /// Stable lowercase name (matches the serde serialization).
    pub fn as_str(self) -> &'static str {
        match self {
            HandleKind::Memory => "memory",
            HandleKind::Commit => "commit",
            HandleKind::File => "file",
            HandleKind::Symbol => "symbol",
            HandleKind::Test => "test",
            HandleKind::Run => "run",
            HandleKind::ToolCall => "tool_call",
            HandleKind::Tag => "tag",
            HandleKind::Unknown => "unknown",
        }
    }
}

/// One recorded row behind a resolved id.
///
/// The `table` tag names the registry the hit came from. Nothing in a proof
/// is inferred from node content.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "table", rename_all = "snake_case")]
pub enum HandleProof {
    /// Anchor row and the effect that admitted it.
    Anchor {
        /// Memory the anchor belongs to.
        id: String,
        /// Anchor id.
        anchor_id: String,
        /// `eff-` receipt of the admitting effect.
        receipt: String,
    },
    /// Commit record: creating frame plus its effect receipt.
    Commit {
        /// Commit memory id.
        id: String,
        /// Log seq of the creating `UpsertNode` frame.
        frame_seq: u64,
        /// Chain hash of that frame, lowercase hex.
        frame_hash: String,
        /// `eff-` receipt of the creating effect.
        receipt: String,
    },
    /// Run record and the effect that admitted it.
    Run {
        /// Run id.
        id: String,
        /// `eff-` receipt of the admitting effect.
        receipt: String,
    },
    /// Typed edge and the data frame that admitted it.
    Edge {
        /// Source memory id.
        id: String,
        source: String,
        target: String,
        link_type: String,
        meta_sha: Option<String>,
        /// Log seq of the `SaveEdge` frame.
        frame_seq: u64,
    },
    /// Memory id (exact or unique prefix).
    Memory {
        /// Memory id.
        id: String,
        /// `eff-` receipt of the creating effect, when the log has one.
        receipt: Option<String>,
    },
    /// Exact tag carried by a live memory.
    Tag {
        /// Memory id.
        id: String,
    },
}

/// Outcome of resolving one query against the store.
///
/// - `ids` non-empty: resolved. When `exact` is false the match was a unique
///   prefix (only a commit sha can get here).
/// - `ids` empty and `candidates` non-empty: ambiguous — the caller must
///   disambiguate (or supply a longer prefix).
/// - both empty: no handle matched; `handle_required` carries the message the
///   handle-based recall flow reports (including the too-short-sha error).
///
/// `proofs` names the recorded row behind each resolved id.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HandleResolution {
    pub kind: HandleKind,
    pub ids: Vec<String>,
    pub exact: bool,
    /// `(id, kind)` pairs shown when a query is ambiguous. Capped at
    /// [`MAX_CANDIDATES`].
    pub candidates: Vec<(String, HandleKind)>,
    /// Set when nothing resolved: the handle_required guidance (or the
    /// specific ambiguity error for a too-short sha).
    pub handle_required: Option<String>,
    /// Recorded rows behind `ids`. Empty when nothing resolved.
    pub proofs: Vec<HandleProof>,
}

// ----------------------------------------------------------------------------
// SESSION QUERIES
// ----------------------------------------------------------------------------

/// One failure-like memory whose recorded files intersect the changed set.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct OpenFailureTouching {
    /// The failure memory's node id.
    pub id: String,
    /// First line/sentence of its content, capped.
    pub content_preview: String,
    /// Where it matched: `path` or `path:symbol` for a code anchor, the bare
    /// path for a git-commit `files:` line entry. `None` never occurs today
    /// (a row exists only because something matched) but keeps the struct
    /// forward-compatible with anchor-less sources.
    pub anchor: Option<String>,
}

/// One failed tool call (`mcp.call` payload with `success: false`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct FailedToolCall {
    pub run_id: String,
    pub tool: String,
    /// Wall-clock millis, straight from the `agent_traces.at` column.
    pub at: i64,
    /// Capped excerpt of the payload's `error` field (string, or the
    /// `message`/`detail` member of an error object). Empty when the payload
    /// recorded no error text.
    pub error_excerpt: String,
}

/// One closed external-issue node, keyed by its source id (the bare issue
/// number as recorded by the connector).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ClosedIssueNode {
    pub node_id: String,
    pub issue_number: String,
}

/// One locally ingested git-commit record (tag `git-commit`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct GitCommitNode {
    pub node_id: String,
    /// Full record content: `commit <sha> <subject>` header plus
    /// `files:`/`modules:`/`symbols:`/`mentions:` lines.
    pub content: String,
}

// ----------------------------------------------------------------------------
// WALK RECEIPTS
// ----------------------------------------------------------------------------

/// Handle returned when a walk receipt is saved. Deterministic: the same
/// canonical parameters always yield the same pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalkReceiptHandle {
    pub receipt_id: String,
    pub digest: String,
    /// True when the row already existed (idempotent re-save of the same
    /// canonical envelope) rather than being written by this call.
    pub reused_existing: bool,
}

/// One persisted walk receipt row.
#[derive(Debug, Clone)]
pub struct StoredWalkReceipt {
    pub receipt_id: String,
    pub digest: String,
    /// The exact canonical JSON bytes that were digested.
    pub canonical_json: String,
    /// The canonical envelope parsed back into a JSON value.
    pub params: serde_json::Value,
    /// vestige-core version that canonicalized and digested the envelope.
    pub engine_version: String,
    pub created_at: String,
}

/// Store-wide coverage aggregates for `memory_status(view="coverage")`.
#[derive(Debug, Clone, PartialEq)]
pub struct CoverageSnapshot {
    /// Total knowledge nodes in the store.
    pub total_nodes: u64,
    /// Distinct nodes carrying at least one code memory anchor.
    pub anchored_nodes: u64,
    /// `anchored_nodes / total_nodes * 100`, rounded to 2 decimals. 0.0 when
    /// the store is empty (no division by zero, no NaN).
    pub anchor_coverage_pct: f64,
    /// `memory_connections` counts grouped by `link_type`, ordered by type
    /// ascending so serialization is byte-stable.
    pub edge_counts_by_type: Vec<(String, u64)>,
    /// RFC 3339 `created_at` of the newest git-commit-tagged node, if any.
    pub newest_git_commit_record: Option<String>,
    /// Whole days since that record was created (floor of exact days).
    pub newest_git_commit_record_age_days: Option<i64>,
    /// Wall-clock millis of the newest Black Box trace event, if any.
    pub newest_agent_trace_at: Option<i64>,
    /// Hours since that event (rounded to 1 decimal).
    pub newest_agent_trace_age_hours: Option<f64>,
}

// ----------------------------------------------------------------------------
// RETRIEVAL REPLAY
// ----------------------------------------------------------------------------

/// Privacy state of a frozen capsule or replay record.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayPrivacyState {
    Active,
    Redacted,
    Purged,
}

/// Coarse decay signal frozen with one evidence item.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ReplayDecayRisk {
    Low,
    Medium,
    High,
}

/// One item in the exact evidence pack supplied by the retrieval boundary.
///
/// `memory_id` is only an internal dependency locator. It is never copied into
/// replay result JSON. `private_digest` must be a keyed digest, not a public
/// hash of memory content.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalReplayItemDraft {
    pub evidence_slot: String,
    pub memory_id: String,
    pub private_digest: String,
    pub token_estimate: u64,
    pub trust_score: f64,
    pub decay_risk: ReplayDecayRisk,
}

/// Draft frozen alongside one retrieval receipt.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalReplayCapsuleDraft {
    pub source_receipt_id: String,
    pub policy_digest: String,
    pub items: Vec<RetrievalReplayItemDraft>,
    pub created_at: DateTime<Utc>,
}

impl RetrievalReplayCapsuleDraft {
    pub fn new(
        source_receipt_id: impl Into<String>,
        policy_digest: impl Into<String>,
        items: Vec<RetrievalReplayItemDraft>,
    ) -> Self {
        Self {
            source_receipt_id: source_receipt_id.into(),
            policy_digest: policy_digest.into(),
            items,
            created_at: Utc::now(),
        }
    }
}

/// Public capsule projection. Stable memory ids and private item digests never
/// enter this type; non-active capsules expose no item rows at all.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalReplayCapsuleSummary {
    pub capsule_id: String,
    pub source_receipt_id: String,
    pub schema_version: u32,
    pub algorithm_version: String,
    pub selection_boundary: String,
    pub redaction_generation: u64,
    pub privacy_state: ReplayPrivacyState,
    pub replayable: bool,
    pub policy_digest: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_evidence_digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_merkle_root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_token_estimate: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trust_floor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decay_risk: Option<ReplayDecayRisk>,
    pub items: Vec<ReplayEvidenceItemSummary>,
    pub created_at: DateTime<Utc>,
}

/// Result of durable capsule creation.
#[derive(Debug, Clone, PartialEq)]
pub struct DurableRetrievalReplayCapsule {
    pub capsule: RetrievalReplayCapsuleSummary,
    pub reused_existing: bool,
}

/// Exact aggregate of one ordered evidence set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ReplayEvidenceItemSummary {
    pub evidence_slot: String,
    /// One-based rank in the exact final evidence order.
    pub rank: u32,
    pub token_estimate: u64,
    pub trust_score: f64,
    pub decay_risk: ReplayDecayRisk,
}

/// Exact aggregate of one ordered evidence set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ReplayEvidenceSetSummary {
    pub items: Vec<ReplayEvidenceItemSummary>,
    pub ordered_slots: Vec<String>,
    pub item_count: u64,
    pub token_estimate: u64,
    pub trust_floor: f64,
    pub decay_risk: ReplayDecayRisk,
    pub ordered_evidence_digest: String,
    pub merkle_root: String,
}

/// Measured structural difference between baseline and ablated context.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ReplayInfluence {
    pub removed_item_count: u64,
    pub removed_token_estimate: u64,
    pub trust_floor_delta: f64,
    pub decay_risk_changed: bool,
    pub ordered_evidence_digest_changed: bool,
    pub merkle_root_changed: bool,
}

/// Privacy-safe evidence payload for one controlled replay.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CounterfactualReplayResult {
    pub source_receipt_id: String,
    pub schema_version: u32,
    pub algorithm_version: String,
    pub selection_boundary: String,
    pub policy_digest: String,
    pub redaction_generation: u64,
    pub withheld_slots: Vec<String>,
    pub baseline: ReplayEvidenceSetSummary,
    pub counterfactual: ReplayEvidenceSetSummary,
    pub replay_influence: ReplayInfluence,
    /// Replay persisted an audit record but did not mutate cognitive state.
    pub memory_state_was_read_only: bool,
    pub claim_boundary: String,
}

/// Stored counterfactual replay. `result` becomes `None` after privacy
/// invalidation; structural audit linkage may remain.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct StoredCounterfactualReplay {
    pub replay_id: String,
    pub idempotency_key: String,
    pub capsule_id: String,
    pub source_receipt_id: String,
    pub receipt_id: Option<String>,
    pub algorithm_version: String,
    pub redaction_generation: u64,
    pub withheld_slots: Vec<String>,
    pub privacy_state: ReplayPrivacyState,
    pub result: Option<CounterfactualReplayResult>,
    pub created_at: DateTime<Utc>,
}

/// Result of replay creation or an idempotent retry.
#[derive(Debug, Clone, PartialEq)]
pub struct DurableCounterfactualReplay {
    pub replay: StoredCounterfactualReplay,
    pub reused_existing: bool,
}

// ----------------------------------------------------------------------------
// SIGNED RECEIPT ATTESTATION STORAGE
// ----------------------------------------------------------------------------

/// Public state of a receipt at the V24 boundary. Absence of an immutable
/// envelope is deliberately explicit rather than silently treated as valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReceiptAttestationStatus {
    LegacyUnsigned,
    SignedV1,
}

/// One all-or-nothing signed-receipt write.
pub struct SignedReceiptWrite<'a> {
    pub receipt: &'a Receipt,
    pub attestation: &'a ReceiptAttestationV1,
    pub signed: &'a SignedReceiptAttestation,
    pub disclosures: &'a [DisclosureMapping],
    pub run_id: Option<&'a str>,
    pub tool: Option<&'a str>,
    pub query: Option<&'a str>,
}

/// Durable identifiers returned only after the SQLite commit succeeds.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DurableSignedReceipt {
    pub receipt_id: String,
    pub chain_id: String,
    pub sequence: u64,
    pub payload_digest: String,
    pub entry_digest: String,
    pub signing_key_id: String,
    pub signer_key_fingerprint: String,
}

/// Commit result for a signed retrieval receipt and its frozen replay capsule.
#[derive(Debug, Clone, PartialEq)]
pub struct DurableSignedRetrievalReceipt {
    pub receipt: DurableSignedReceipt,
    pub replay_capsule: DurableRetrievalReplayCapsule,
}

/// Locally re-verified stored receipt state. This establishes cryptographic
/// integrity against the local trusted-key registry and current database rows;
/// it is not an independently published checkpoint or trusted timestamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredReceiptAttestationVerification {
    pub report: VerificationReport,
    pub receipt_binding_valid: bool,
}

impl StoredReceiptAttestationVerification {
    pub fn is_valid(&self) -> bool {
        self.report.is_valid() && self.receipt_binding_valid
    }
}

// ----------------------------------------------------------------------------
// SYNAPTIC CAPTURE
// ----------------------------------------------------------------------------

/// Frozen scoring policy supplied by the cognitive engine.
#[derive(Debug, Clone)]
pub struct SynapticCapturePolicy {
    pub backward_hours: f64,
    pub forward_hours: f64,
    pub tag_lifetime_hours: f64,
    pub minimum_tag_strength: f64,
    pub maximum_captures: usize,
    pub decay_function: DecayFunction,
}

/// One durable importance-event evaluation request.
#[derive(Debug, Clone)]
pub struct SynapticCaptureRequest {
    pub trigger_memory_id: String,
    pub event_type: String,
    pub occurred_at: DateTime<Utc>,
    pub strength: f64,
    pub policy: SynapticCapturePolicy,
}

/// Observable result of a committed capture transaction.
#[derive(Debug, Clone)]
pub struct DurableSynapticCapture {
    pub event_id: String,
    pub receipt: Receipt,
    pub captured_count: usize,
    pub reused_existing: bool,
}

/// Privacy-safe numeric snapshot of the importance signal that opened an
/// event. Explanation strings are intentionally excluded.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SynapticSignalSnapshot {
    pub novelty: f64,
    pub arousal: f64,
    pub reward: f64,
    pub attention: f64,
    pub composite: f64,
}

/// A frozen V2 importance event supplied with the tag produced by one ingest.
#[derive(Debug, Clone)]
pub struct SynapticImportanceEvent {
    pub event_type: String,
    pub occurred_at: DateTime<Utc>,
    pub strength: f64,
    pub policy: SynapticCapturePolicy,
    pub signal_snapshot: SynapticSignalSnapshot,
}

/// Atomic event + tag request used by smart_ingest V2.
#[derive(Debug, Clone)]
pub struct SynapticIngestRequest {
    pub memory_id: String,
    pub tag: Option<SynapticTag>,
    pub event: Option<SynapticImportanceEvent>,
}

/// One immutable V2 forward pair receipt.
#[derive(Debug, Clone)]
pub struct DurableSynapticPairReceipt {
    pub event_id: String,
    pub receipt: Receipt,
    pub disposition: crate::trace::SynapticCaptureDisposition,
    pub reused_existing: bool,
}

/// Observable result of one committed V2 ingest transaction.
#[derive(Debug, Clone)]
pub struct SynapticIngestOutcome {
    pub event: Option<DurableSynapticCapture>,
    pub tag_id: Option<String>,
    pub tag_persisted: bool,
    /// Whether the request tag remains active after the committed transaction.
    /// A tag can be persisted then immediately consumed by an already-open
    /// forward event, so callers must not restore it into a live projection
    /// solely because `tag_persisted` is true.
    pub tag_active: bool,
    pub forward_receipts: Vec<DurableSynapticPairReceipt>,
}

// ----------------------------------------------------------------------------
// PORTABLE ARCHIVE
// ----------------------------------------------------------------------------

/// Full exact portable archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PortableArchive {
    /// Stable format marker used for compatibility checks.
    pub archive_format: String,
    /// Vestige version that produced the archive.
    pub vestige_version: String,
    /// SQLite schema version of the source database.
    pub schema_version: u32,
    /// Archive creation timestamp.
    pub exported_at: DateTime<Utc>,
    /// Export mode. v1 only writes "exact".
    pub mode: String,
    /// Dumped storage tables in deterministic import order.
    pub tables: Vec<PortableTable>,
}

impl PortableArchive {
    /// Count all rows across all tables.
    pub fn total_rows(&self) -> usize {
        self.tables.iter().map(|table| table.rows.len()).sum()
    }
}

/// One table in a portable archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PortableTable {
    /// SQLite table name.
    pub name: String,
    /// Column names in row value order.
    pub columns: Vec<String>,
    /// Raw rows. Each row has the same order as `columns`.
    pub rows: Vec<Vec<PortableValue>>,
}

/// SQLite value encoded in JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "camelCase")]
pub enum PortableValue {
    /// SQL NULL.
    Null,
    /// SQL INTEGER.
    Integer(i64),
    /// SQL REAL.
    Real(f64),
    /// SQL TEXT.
    Text(String),
    /// SQL BLOB, hex encoded.
    Blob(String),
}

/// Import behavior for duplicate primary keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortableImportMode {
    /// Reject import if user data already exists, then insert rows exactly.
    EmptyOnly,
    /// Merge archive rows into an existing database.
    ///
    /// This mode is intended for file-backed sync between devices. It applies
    /// tombstones, upserts row-keyed state, and appends audit/history rows.
    Merge,
}

/// Summary of an exact portable import.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PortableImportReport {
    /// Number of imported tables.
    pub tables_imported: usize,
    /// Number of imported rows.
    pub rows_imported: usize,
    /// Number of archive tables skipped because the target schema lacks them.
    pub tables_skipped: usize,
    /// Whether FTS was rebuilt after import.
    pub fts_rebuilt: bool,
    /// Number of rows inserted.
    #[serde(default)]
    pub rows_inserted: usize,
    /// Number of existing rows updated/replaced.
    #[serde(default)]
    pub rows_updated: usize,
    /// Number of rows skipped because local state was newer or unsupported.
    #[serde(default)]
    pub rows_skipped: usize,
    /// Number of local rows deleted by imported tombstones.
    #[serde(default)]
    pub rows_deleted: usize,
    /// Number of merge conflicts resolved by keeping local state.
    #[serde(default)]
    pub conflicts_kept_local: usize,
}

// ----------------------------------------------------------------------------
// BLACK BOX / TRACE ROLL-UP
// ----------------------------------------------------------------------------

/// Side effect applied while atomically deciding a pre-execution mutation PR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendingMemoryMutationEffect {
    /// The reviewer kept the memory unchanged.
    Kept,
    /// The reviewer approved the pending purge/delete.
    Purged,
    /// The reviewer held the memory under active suppression.
    Suppressed,
}

/// Result of deciding a PR created before a destructive mutation.
#[derive(Debug, Clone)]
pub struct PendingMemoryMutationDecision {
    /// Final PR state returned even when an approved purge removed its row.
    pub pr: MemoryPr,
    /// Mutation side effect committed with the decision.
    pub effect: PendingMemoryMutationEffect,
}

/// A roll-up summary of one agent run, for the Black Box run list.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct AgentRunSummary {
    /// The run id.
    pub run_id: String,
    /// The first tool invoked in the run (the run's "entry point").
    pub first_tool: Option<String>,
    /// Total events recorded.
    pub event_count: i64,
    /// Memories retrieved across the run.
    pub retrieved_count: i64,
    /// Memories suppressed across the run.
    pub suppressed_count: i64,
    /// Memory writes across the run.
    pub write_count: i64,
    /// Sanhedrin vetoes across the run.
    pub veto_count: i64,
    /// Millis of the first event.
    pub started_at: i64,
    /// Millis of the most recent event.
    pub last_at: i64,
}

// ----------------------------------------------------------------------------
// RECEIPT ATTESTATION RE-EXPORTS
// ----------------------------------------------------------------------------
//
// Pure-logic attestation types that already live in the ungated
// `receipt_attestation` module, re-exported here so the storage type surface
// has one canonical import site.

pub use crate::storage::receipt_attestation::{ChainEntry, DsseEnvelope, TrustedSigningKey};
