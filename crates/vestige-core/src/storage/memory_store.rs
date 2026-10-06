//! Backend-agnostic memory store trait.
//!
//! This is the single abstraction every cognitive module sits above. It is
//! intentionally flat: one trait, plus the Phase 4 sync product seam
//! (~175 sync methods mirroring the SQLite reference surface).

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

// Phase 4 wall: types referenced by the product-wide sync seam below.
use crate::SchemaIntrospection;
use crate::actor::{ActorPolicySnapshot, RoleResolution};
use crate::advanced::reconsolidation::LabileCandidate;
use crate::advanced::{MergeCandidate, MergeOperation, MergePlan, MergePolicy};
use crate::codebase::anchor::{AnchorStatus, CodeAnchor};
use crate::fsrs::Rating;
use crate::intention_graph::Command;
use crate::memory::{ConsolidationResult, IngestInput, KnowledgeNode, MemoryStats, RecallInput};
use crate::neuroscience::SynapticTag;
use crate::security::SecretPolicy;
use crate::trace::{MemoryPr, MemoryPrAction, MemoryPrStatus, MemoryTraceEvent, Receipt};
// Dual-mode rule (strata/fix-00a): every storage type this trait names is
// defined in the UNGATED `super::types` module, so the trait compiles with
// AND without `legacy-sqlite`. The pure receipt-attestation trio is
// re-exported through `types` as well.
use super::types::{
    ActorMutationOutcome, AgentRunSummary, BlastReport, ChainEntry, ClosedIssueNode,
    CompositionEventRecord, CompositionMemberRecord, CompositionNeighborRecord,
    CompositionOutcomeRecord, ConnectionRecord, ConnectorCursor, ConsolidationHistoryRecord,
    CoverageSnapshot, DreamHistoryRecord, DsseEnvelope, DurableCounterfactualReplay,
    DurableRetrievalReplayCapsule, DurableSignedRetrievalReceipt, DurableSynapticCapture,
    EndorsementEventRecord, FailedToolCall, FailureFeedbackReport, GitCommitNode, HandleResolution,
    HygieneSnapshot, InsightRecord, IntentionRecord, NeverComposedCandidate, OpenFailureTouching,
    PendingMemoryMutationDecision, PortableArchive, PortableImportMode, PortableImportReport,
    PortableSyncReport, PurgeReport, ReceiptAttestationStatus, ReconcileReport,
    Result as StoreResult, RetireOutcome, RetrievalReplayCapsuleDraft,
    RetrievalReplayCapsuleSummary, SignedReceiptWrite, SmartIngestResult, SourceUpsertResult,
    StateTransitionRecord, StorageError, StoredCounterfactualReplay,
    StoredReceiptAttestationVerification, StoredWalkReceipt, SynapticCaptureRequest,
    SynapticIngestOutcome, SynapticIngestRequest, TagVocabulary, TrustedSigningKey,
    WalCheckpointMode, WalCheckpointStatus, WalkReceiptHandle,
};

// ----------------------------------------------------------------------------
// ERROR
// ----------------------------------------------------------------------------

/// Error returned by every `LocalMemoryStore` / `MemoryStore` method.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum MemoryStoreError {
    #[error("not found: {0}")]
    NotFound(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error(
        "embedding model mismatch: store registered {registered_name} (dim {registered_dim}, \
         hash {registered_hash}), embedder is {actual_name} (dim {actual_dim}, hash {actual_hash})"
    )]
    ModelMismatch {
        registered_name: String,
        registered_dim: usize,
        registered_hash: String,
        actual_name: String,
        actual_dim: usize,
        actual_hash: String,
    },

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("initialization error: {0}")]
    Init(String),

    #[error("probable credential rejected: {0}")]
    SecretDetected(String),
}

// Conversion from the legacy SQLite error type (behind `legacy-sqlite`,
// build/t5-legacy-isolation).
#[cfg(feature = "legacy-sqlite")]
impl From<crate::storage::StorageError> for MemoryStoreError {
    fn from(e: crate::storage::StorageError) -> Self {
        use crate::storage::StorageError as S;
        match e {
            S::NotFound(s) => MemoryStoreError::NotFound(s),
            S::Database(e) => MemoryStoreError::Backend(e.to_string()),
            S::Io(e) => MemoryStoreError::Backend(e.to_string()),
            S::InvalidTimestamp(s) => MemoryStoreError::Backend(format!("invalid timestamp: {s}")),
            S::InvalidScope(s) => MemoryStoreError::InvalidInput(s),
            S::InvalidEdge(s) => MemoryStoreError::InvalidInput(format!("invalid typed edge: {s}")),
            S::InvalidEmbeddingProfile(s) => {
                MemoryStoreError::InvalidInput(format!("invalid embedding profile: {s}"))
            }
            S::Init(s) => MemoryStoreError::Init(s),
            S::V3StoreNeedsMigration {
                path,
                schema_version,
                hint,
            } => MemoryStoreError::Init(format!(
                "v3 SQLite store at {path} (schema version {schema_version}) needs migration: {hint}"
            )),
            S::SecretDetected { kinds } => MemoryStoreError::SecretDetected(kinds.join(", ")),
        }
    }
}

pub type MemoryStoreResult<T> = std::result::Result<T, MemoryStoreError>;

// ----------------------------------------------------------------------------
// DATA TYPES
// ----------------------------------------------------------------------------

/// Backend-agnostic memory record.
///
/// Phase 1 intentionally keeps this type independent of `KnowledgeNode` to
/// avoid dragging 30+ legacy fields through the trait surface. The SQLite
/// backend converts between `MemoryRecord` and `KnowledgeNode` at the
/// boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: Uuid,
    /// Empty = unclassified. Populated in Phase 4.
    pub domains: Vec<String>,
    /// Raw similarity per domain centroid. Empty until Phase 4 runs clustering.
    pub domain_scores: HashMap<String, f64>,
    pub content: String,
    pub node_type: String,
    pub tags: Vec<String>,
    pub embedding: Option<Vec<f32>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

/// FSRS-6 scheduling state, one row per memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingState {
    pub memory_id: Uuid,
    pub stability: f64,
    pub difficulty: f64,
    pub retrievability: f64,
    pub last_review: Option<DateTime<Utc>>,
    pub next_review: Option<DateTime<Utc>>,
    pub reps: u32,
    pub lapses: u32,
}

/// Hybrid search request.
#[derive(Debug, Clone, Default)]
pub struct SearchQuery {
    pub domains: Option<Vec<String>>,
    pub text: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub tags: Option<Vec<String>>,
    pub node_types: Option<Vec<String>>,
    pub limit: usize,
    pub min_retrievability: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub record: MemoryRecord,
    pub score: f64,
    pub fts_score: Option<f64>,
    pub vector_score: Option<f64>,
}

/// Edge in the spreading-activation graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEdge {
    pub source_id: Uuid,
    pub target_id: Uuid,
    pub edge_type: String,
    pub weight: f64,
    pub created_at: DateTime<Utc>,
}

/// A topical domain (populated in Phase 4). Phase 1 only needs the type to
/// shape the trait surface; discover/classify are Phase 4 work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Domain {
    pub id: String,
    pub label: String,
    pub centroid: Vec<f32>,
    pub top_terms: Vec<String>,
    pub memory_count: usize,
    pub created_at: DateTime<Utc>,
}

/// Result of classifying one vector against all known domains.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub scores: HashMap<String, f64>,
    pub domains: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StoreStats {
    pub total_memories: usize,
    pub memories_with_embeddings: usize,
    pub total_edges: usize,
    pub total_domains: usize,
    pub registered_model_name: Option<String>,
    pub registered_model_dim: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded { reason: String },
    Unavailable { reason: String },
}

// ----------------------------------------------------------------------------
// EMBEDDING MODEL SIGNATURE
// ----------------------------------------------------------------------------

/// Snapshot of the embedding model that was used to write vectors into the
/// store. Persisted in the `embedding_model` table; compared on every write
/// before the vector is accepted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSignature {
    pub name: String,
    pub dimension: usize,
    /// Lowercase hex-encoded blake3 hash, 64 chars.
    pub hash: String,
}

// ----------------------------------------------------------------------------
// TRAIT
// ----------------------------------------------------------------------------

/// Internal source trait declared with native async-fn-in-trait.
///
/// `#[trait_variant::make(MemoryStoreSend: Send)]` derives a Send-bounded
/// variant whose returned futures are `Send`. In trait_variant 0.1.x the
/// macro emits the blanket `impl<T: MemoryStoreSend> LocalMemoryStore for T`,
/// so backends implement `MemoryStoreSend` (the Send variant) and get
/// `LocalMemoryStore` (the non-Send variant) for free.
///
/// Most callers should reach for the dyn-compatible `MemoryStore` trait
/// declared below, which adapts `MemoryStoreSend` into a boxed-future surface
/// and is the public storage abstraction for cognitive modules and tests
/// that want `Arc<dyn MemoryStore>`.
#[allow(unused_variables)] // stub defaults ignore their arguments
#[trait_variant::make(MemoryStoreSend: Send)]
pub trait LocalMemoryStore: Sync + 'static {
    // --- Lifecycle ---
    async fn init(&self) -> MemoryStoreResult<()>;
    async fn health_check(&self) -> MemoryStoreResult<HealthStatus>;

    // --- Embedding model registry ---
    async fn registered_model(&self) -> MemoryStoreResult<Option<ModelSignature>>;
    async fn register_model(&self, sig: &ModelSignature) -> MemoryStoreResult<()>;

    // --- CRUD ---
    async fn insert(&self, record: &MemoryRecord) -> MemoryStoreResult<Uuid>;
    async fn get(&self, id: Uuid) -> MemoryStoreResult<Option<MemoryRecord>>;
    async fn update(&self, record: &MemoryRecord) -> MemoryStoreResult<()>;
    async fn delete(&self, id: Uuid) -> MemoryStoreResult<()>;

    // --- Search ---
    async fn search_records(&self, query: &SearchQuery) -> MemoryStoreResult<Vec<SearchResult>>;
    async fn fts_search(&self, text: &str, limit: usize) -> MemoryStoreResult<Vec<SearchResult>>;
    async fn vector_search(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> MemoryStoreResult<Vec<SearchResult>>;

    // --- FSRS Scheduling ---
    async fn get_scheduling(&self, memory_id: Uuid) -> MemoryStoreResult<Option<SchedulingState>>;
    async fn update_scheduling(&self, state: &SchedulingState) -> MemoryStoreResult<()>;
    async fn get_due_memories(
        &self,
        before: DateTime<Utc>,
        limit: usize,
    ) -> MemoryStoreResult<Vec<(MemoryRecord, SchedulingState)>>;

    // --- Graph (spreading activation) ---
    async fn add_edge(&self, edge: &MemoryEdge) -> MemoryStoreResult<()>;
    async fn get_edges(
        &self,
        node_id: Uuid,
        edge_type: Option<&str>,
    ) -> MemoryStoreResult<Vec<MemoryEdge>>;
    async fn remove_edge(&self, source: Uuid, target: Uuid) -> MemoryStoreResult<()>;
    async fn get_neighbors(
        &self,
        node_id: Uuid,
        depth: usize,
    ) -> MemoryStoreResult<Vec<(MemoryRecord, f64)>>;

    // --- Domains (Phase 1: stubs return empty; full impl in Phase 4) ---
    async fn list_domains(&self) -> MemoryStoreResult<Vec<Domain>>;
    async fn get_domain(&self, id: &str) -> MemoryStoreResult<Option<Domain>>;
    async fn upsert_domain(&self, domain: &Domain) -> MemoryStoreResult<()>;
    async fn delete_domain(&self, id: &str) -> MemoryStoreResult<()>;
    /// Phase 1: returns `Ok(vec![])` since no centroids exist. Phase 4 wires
    /// the full soft-assignment pass.
    async fn classify(&self, embedding: &[f32]) -> MemoryStoreResult<Vec<(String, f64)>>;

    // --- Bulk / Maintenance ---
    async fn count(&self) -> MemoryStoreResult<usize>;
    async fn get_store_stats(&self) -> MemoryStoreResult<StoreStats>;
    async fn vacuum(&self) -> MemoryStoreResult<()>;

    // ------------------------------------------------------------------------
    // PHASE 4 PRODUCT SEAM (sync)
    //
    // The product-wide surface every tool, server, and CLI entry point uses.
    // These mirror the SQLite reference implementation's inherent methods one
    // for one so the whole product compiles against `Arc<dyn MemoryStore>`.
    // Backends implement (override) what they support; the default bodies
    // fail loudly (StorageError::Init / unimplemented!) so a partial backend
    // can never silently succeed.
    // ------------------------------------------------------------------------

    /// Admit `projected_to` edges for a confirmed projection.
    ///
    /// Returns `(receipt_id, blake3_hex)` of `region`. The file write happens
    /// only after this returns `Ok`. Default backends refuse.
    fn admit_projection(
        &self,
        _memory_ids: &[String],
        _target: &str,
        _region: &[u8],
    ) -> StoreResult<(String, String)> {
        Err(StorageError::Init(
            concat!(
                stringify!(admit_projection),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Snapshot of actor policy state.
    fn actor_policy_snapshot(&self) -> StoreResult<ActorPolicySnapshot> {
        Err(StorageError::Init(
            concat!(
                stringify!(actor_policy_snapshot),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn append_mcp_call_outcome(
        &self,
        _run_id: &str,
        _tool: &str,
        _success: bool,
        _error: Option<&str>,
        _at_ms: i64,
    ) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(append_mcp_call_outcome),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Append one trace event.
    fn append_trace_event(&self, _event: &MemoryTraceEvent) -> StoreResult<i64> {
        Err(StorageError::Init(
            concat!(
                stringify!(append_trace_event),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Apply failure feedback across the lookback window.
    fn apply_failure_feedback(
        &self,
        _failure_id: &str,
        _window: Duration,
    ) -> StoreResult<FailureFeedbackReport> {
        Err(StorageError::Init(
            concat!(
                stringify!(apply_failure_feedback),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Apply one intention-graph command in a scope.
    fn apply_intention_graph(
        &self,
        _scope: &str,
        _command: Command,
        _now: DateTime<Utc>,
    ) -> Result<Value, String> {
        Err(concat!(
            stringify!(apply_intention_graph),
            " is not implemented by this backend"
        )
        .to_string())
    }

    /// See the SQLite reference implementation for semantics.
    fn apply_tag_mutation(
        &self,
        _source_tags: &[String],
        _target_tag: &str,
        _scope: Option<&str>,
        _preview_token: &str,
        _op_type: &str,
        _reason: &str,
    ) -> StoreResult<MergeOperation> {
        Err(StorageError::Init(
            concat!(
                stringify!(apply_tag_mutation),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Write a consistent backup of the whole store.
    fn backup_to(&self, _path: &Path) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(stringify!(backup_to), " is not implemented by this backend").into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn blast_radius(&self, _root_id: &str, _open_only: bool) -> StoreResult<BlastReport> {
        Err(StorageError::Init(
            concat!(
                stringify!(blast_radius),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn blast_radius_with_link_types(
        &self,
        _root_id: &str,
        _open_only: bool,
        _link_types: &[&str],
    ) -> StoreResult<BlastReport> {
        Err(StorageError::Init(
            concat!(
                stringify!(blast_radius_with_link_types),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Evaluate and durably commit one synaptic capture.
    fn capture_synaptic_event(
        &self,
        _request: &SynapticCaptureRequest,
    ) -> StoreResult<DurableSynapticCapture> {
        Err(StorageError::Init(
            concat!(
                stringify!(capture_synaptic_event),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Checkpoint the write-ahead log.
    fn checkpoint_wal(&self, _mode: WalCheckpointMode) -> StoreResult<WalCheckpointStatus> {
        Err(StorageError::Init(
            concat!(
                stringify!(checkpoint_wal),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn clear_dream_page_tags(
        &self,
        _ids: &[String],
        _started_at: DateTime<Utc>,
    ) -> StoreResult<usize> {
        Err(StorageError::Init(
            concat!(
                stringify!(clear_dream_page_tags),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Closed-issue memories per connector.
    fn closed_issue_nodes(
        &self,
        _source_system: &str,
        _scope: &str,
    ) -> StoreResult<Vec<ClosedIssueNode>> {
        Err(StorageError::Init(
            concat!(
                stringify!(closed_issue_nodes),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Anchors recorded for one node.
    fn code_anchors_for_node(&self, _node_id: &str) -> StoreResult<Vec<CodeAnchor>> {
        Err(StorageError::Init(
            concat!(
                stringify!(code_anchors_for_node),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Anchors for a batch of nodes.
    fn code_anchors_for_nodes(
        &self,
        _node_ids: &[String],
    ) -> StoreResult<HashMap<String, Vec<CodeAnchor>>> {
        Err(StorageError::Init(
            concat!(
                stringify!(code_anchors_for_nodes),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Durably commit intention state transitions.
    fn commit_intention_check(
        &self,
        _changes: &[(IntentionRecord, IntentionRecord)],
    ) -> Result<(), String> {
        Err(concat!(
            stringify!(commit_intention_check),
            " is not implemented by this backend"
        )
        .to_string())
    }

    /// See the SQLite reference implementation for semantics.
    fn concrete_search_filtered(
        &self,
        _query: &str,
        _limit: i32,
        _include_types: Option<&[String]>,
        _exclude_types: Option<&[String]>,
    ) -> StoreResult<Vec<crate::memory::SearchResult>> {
        Err(StorageError::Init(
            concat!(
                stringify!(concrete_search_filtered),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Count nodes under a retention threshold.
    fn count_memories_below_retention(&self, _threshold: f64) -> StoreResult<i64> {
        Err(StorageError::Init(
            concat!(
                stringify!(count_memories_below_retention),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Count nodes created since a timestamp.
    fn count_memories_since(&self, _since: DateTime<Utc>) -> StoreResult<i64> {
        Err(StorageError::Init(
            concat!(
                stringify!(count_memories_since),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Count unreviewed memory PRs.
    fn count_pending_memory_prs(&self) -> StoreResult<i64> {
        Err(StorageError::Init(
            concat!(
                stringify!(count_pending_memory_prs),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Number of currently suppressed memories.
    fn count_suppressed(&self) -> StoreResult<usize> {
        Err(StorageError::Init(
            concat!(
                stringify!(count_suppressed),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn coverage_snapshot(&self) -> StoreResult<CoverageSnapshot> {
        Err(StorageError::Init(
            concat!(
                stringify!(coverage_snapshot),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn create_context_ablation_replay(
        &self,
        _source_receipt_id: &str,
        _withheld_slots: &[String],
    ) -> StoreResult<DurableCounterfactualReplay> {
        Err(StorageError::Init(
            concat!(
                stringify!(create_context_ablation_replay),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Re-derive a Strata log from its frames and compare that fold to one receipt.
    /// Read-only. SQLite backends do not implement it.
    fn replay_receipt(&self, _receipt_id: &str) -> StoreResult<Value> {
        Err(StorageError::Init(
            concat!(
                stringify!(replay_receipt),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn current_code_context_nodes(
        &self,
        _node_type: &str,
        _tag: Option<&str>,
        _scope: &str,
        _limit: i32,
    ) -> StoreResult<Vec<KnowledgeNode>> {
        Err(StorageError::Init(
            concat!(
                stringify!(current_code_context_nodes),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Every scope that holds current code advice of `node_type` (optionally
    /// carrying one exact tag), with how many items it holds, ordered by scope
    /// name. Same eligibility as `current_code_context_nodes`, but across
    /// scopes, so a caller can say where advice lives instead of reporting an
    /// empty namespace as an empty store.
    fn current_code_context_scope_counts(
        &self,
        _node_type: &str,
        _tag: Option<&str>,
    ) -> StoreResult<Vec<(String, usize)>> {
        Err(StorageError::Init(
            concat!(
                stringify!(current_code_context_scope_counts),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Root data directory backing this store.
    fn data_dir(&self) -> &Path {
        unimplemented!(concat!(
            stringify!(data_dir),
            " is not implemented by this backend"
        ))
    }

    /// Path of the primary database artifact.
    fn db_path(&self) -> &Path {
        unimplemented!(concat!(
            stringify!(db_path),
            " is not implemented by this backend"
        ))
    }

    /// True when this store is the Strata log.
    ///
    /// The default is false. Callers that only need to tell Strata from
    /// every other backend use this instead of [`Self::db_path`], whose
    /// default panics.
    fn is_strata(&self) -> bool {
        false
    }

    /// See the SQLite reference implementation for semantics.
    fn decide_memory_pr(&self, _id: &str, _action: MemoryPrAction) -> StoreResult<MemoryPr> {
        Err(StorageError::Init(
            concat!(
                stringify!(decide_memory_pr),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn decide_pending_memory_mutation(
        &self,
        _id: &str,
        _action: MemoryPrAction,
    ) -> StoreResult<Option<PendingMemoryMutationDecision>> {
        Err(StorageError::Init(
            concat!(
                stringify!(decide_pending_memory_mutation),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Delete a node; true when a row was removed.
    fn delete_node(&self, _id: &str) -> StoreResult<bool> {
        Err(StorageError::Init(
            concat!(
                stringify!(delete_node),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Demote a memory's retrieval strength.
    fn demote_memory(&self, _id: &str) -> StoreResult<KnowledgeNode> {
        Err(StorageError::Init(
            concat!(
                stringify!(demote_memory),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn demote_memory_as_actor(
        &self,
        _id: &str,
        _claimed_role: Option<&str>,
        _tool: &str,
    ) -> StoreResult<ActorMutationOutcome> {
        Err(StorageError::Init(
            concat!(
                stringify!(demote_memory_as_actor),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn due_for_review_node_ids(&self, _limit: usize) -> StoreResult<Vec<String>> {
        Err(StorageError::Init(
            concat!(
                stringify!(due_for_review_node_ids),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn expire_stale_reconsolidation_plans(&self) -> StoreResult<Vec<String>> {
        Err(StorageError::Init(
            concat!(
                stringify!(expire_stale_reconsolidation_plans),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Export a portable archive to a path.
    fn export_portable_archive_to_path(&self, _path: &Path) -> StoreResult<PortableArchive> {
        Err(StorageError::Init(
            concat!(
                stringify!(export_portable_archive_to_path),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// The node a canonical duplicate resolves to: the id of the first
    /// regular node in `scope` whose canonicalized `content` matches, or
    /// `None` when this backend keeps no canonical index (the write then
    /// proceeds as a fresh create).
    ///
    /// Canonicalization is version-pinned by the Strata store
    /// (`nfc-lower-zwstrip-wscollapse-v1`); backends without it never match.
    fn find_duplicate_by_canonical_hash(
        &self,
        _scope: &str,
        _content: &str,
    ) -> StoreResult<Option<String>> {
        Ok(None)
    }

    /// The recorded first write for `(scope, intent_id)` as
    /// `(node_id, effect_seq, response_digest)`, for idempotent replay.
    /// `None` when no write under this intent landed (or the backend keeps
    /// no intent index).
    fn find_intent_record(
        &self,
        _scope: &str,
        _intent_id: &str,
    ) -> StoreResult<Option<(String, u64, String)>> {
        Ok(None)
    }

    /// All open intentions across scopes.
    fn get_active_intentions(&self) -> StoreResult<Vec<IntentionRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_active_intentions),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Open intentions inside one scope.
    fn get_active_intentions_in_scope(&self, _scope: &str) -> StoreResult<Vec<IntentionRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_active_intentions_in_scope),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// One agent run summary.
    fn get_agent_run(&self, _run_id: &str) -> StoreResult<Option<AgentRunSummary>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_agent_run),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Every edge in the store.
    fn get_all_connections(&self) -> StoreResult<Vec<ConnectionRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_all_connections),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Page through every node in the default scope order.
    fn get_all_nodes(&self, _limit: i32, _offset: i32) -> StoreResult<Vec<KnowledgeNode>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_all_nodes),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Page through every node in a project scope.
    fn get_all_nodes_in_scope(
        &self,
        _scope: &str,
        _limit: i32,
        _offset: i32,
    ) -> StoreResult<Vec<KnowledgeNode>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_all_nodes_in_scope),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_avg_retention(&self) -> StoreResult<f64> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_avg_retention),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_composition_event(&self, _id: &str) -> StoreResult<Option<CompositionEventRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_composition_event),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Members of one composition event.
    fn get_composition_members(
        &self,
        _event_id: &str,
    ) -> StoreResult<Vec<CompositionMemberRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_composition_members),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_composition_neighbors(
        &self,
        _memory_id: &str,
        _limit: i32,
    ) -> StoreResult<Vec<CompositionNeighborRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_composition_neighbors),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_composition_outcomes(
        &self,
        _event_id: &str,
    ) -> StoreResult<Vec<CompositionOutcomeRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_composition_outcomes),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_compositions_for_memory(
        &self,
        _memory_id: &str,
        _limit: i32,
    ) -> StoreResult<Vec<CompositionEventRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_compositions_for_memory),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Edges touching one memory.
    fn get_connections_for_memory(&self, _memory_id: &str) -> StoreResult<Vec<ConnectionRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_connections_for_memory),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_consolidation_history(
        &self,
        _limit: i32,
    ) -> StoreResult<Vec<ConsolidationHistoryRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_consolidation_history),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_context_ablation_replay(
        &self,
        _replay_id: &str,
    ) -> StoreResult<Option<StoredCounterfactualReplay>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_context_ablation_replay),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Read recent dream history rows.
    fn get_dream_history(&self, _limit: i32) -> StoreResult<Vec<DreamHistoryRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_dream_history),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Read recent insight rows.
    fn get_insights(&self, _limit: i32) -> StoreResult<Vec<InsightRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_insights),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Fetch one intention by id.
    fn get_intention(&self, _id: &str) -> StoreResult<Option<IntentionRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_intention),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_intentions_by_status(&self, _status: &str) -> StoreResult<Vec<IntentionRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_intentions_by_status),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Timestamp of the last consolidation.
    fn get_last_consolidation(&self) -> StoreResult<Option<DateTime<Utc>>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_last_consolidation),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Timestamp of the last dream pass.
    fn get_last_dream(&self) -> StoreResult<Option<DateTime<Utc>>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_last_dream),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_memory_pr(&self, _id: &str) -> StoreResult<Option<MemoryPr>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_memory_pr),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_memory_subgraph(
        &self,
        _center_id: &str,
        _depth: u32,
        _max_nodes: usize,
    ) -> StoreResult<(Vec<KnowledgeNode>, Vec<ConnectionRecord>)> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_memory_subgraph),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_merge_operation(&self, _operation_id: &str) -> StoreResult<Option<MergeOperation>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_merge_operation),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Load the active merge/supersession policy.
    fn get_merge_policy(&self) -> StoreResult<MergePolicy> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_merge_policy),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_most_connected_memory(&self) -> StoreResult<Option<String>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_most_connected_memory),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_never_composed_candidates(
        &self,
        _limit: i32,
        _tag_filter: Option<&[String]>,
    ) -> StoreResult<Vec<NeverComposedCandidate>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_never_composed_candidates),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_never_composed_candidates_in_scope(
        &self,
        _limit: i32,
        _tag_filter: Option<&[String]>,
        _scope: Option<&str>,
    ) -> StoreResult<Vec<NeverComposedCandidate>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_never_composed_candidates_in_scope),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Fetch one memory node by id.
    fn get_node(&self, _id: &str) -> StoreResult<Option<KnowledgeNode>> {
        Err(StorageError::Init(
            concat!(stringify!(get_node), " is not implemented by this backend").into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_overdue_intentions(&self) -> StoreResult<Vec<IntentionRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_overdue_intentions),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Fetch one receipt by id.
    fn get_receipt(&self, _receipt_id: &str) -> StoreResult<Option<Receipt>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_receipt),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_receipt_attestation_envelope(
        &self,
        _receipt_id: &str,
    ) -> StoreResult<Option<DsseEnvelope>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_receipt_attestation_envelope),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_recent_composition_events(
        &self,
        _limit: i32,
    ) -> StoreResult<Vec<CompositionEventRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_recent_composition_events),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_recent_composition_events_page(
        &self,
        _limit: i32,
        _offset: i32,
    ) -> StoreResult<Vec<CompositionEventRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_recent_composition_events_page),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_recent_connections(&self, _limit: usize) -> StoreResult<Vec<ConnectionRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_recent_connections),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_recent_state_transitions(&self, _limit: i32) -> StoreResult<Vec<StateTransitionRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_recent_state_transitions),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_retention_distribution(&self) -> StoreResult<Vec<(String, i64)>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_retention_distribution),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Retention trend summary string.
    fn get_retention_trend(&self) -> StoreResult<String> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_retention_trend),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_retrieval_replay_capsule(
        &self,
        _source_receipt_id: &str,
    ) -> StoreResult<Option<RetrievalReplayCapsuleSummary>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_retrieval_replay_capsule),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Nodes scheduled for review.
    fn get_review_queue(&self, _limit: i32) -> StoreResult<Vec<KnowledgeNode>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_review_queue),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_state_transitions(
        &self,
        _memory_id: &str,
        _limit: i32,
    ) -> StoreResult<Vec<StateTransitionRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_state_transitions),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Product store statistics (counts, strength, states).
    fn get_stats(&self) -> StoreResult<MemoryStats> {
        Err(StorageError::Init(
            concat!(stringify!(get_stats), " is not implemented by this backend").into(),
        ))
    }

    /// Trace events for one agent run.
    fn get_trace(&self, _run_id: &str) -> StoreResult<Vec<MemoryTraceEvent>> {
        Err(StorageError::Init(
            concat!(stringify!(get_trace), " is not implemented by this backend").into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn get_walk_receipt(&self, _receipt_id: &str) -> StoreResult<Option<StoredWalkReceipt>> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_walk_receipt),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Recent git commit memory nodes.
    fn git_commit_nodes(&self, _limit: usize) -> StoreResult<Vec<GitCommitNode>> {
        Err(StorageError::Init(
            concat!(
                stringify!(git_commit_nodes),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Grant an actor role; returns grant row id.
    fn grant_actor_role(
        &self,
        _actor_did: &str,
        _role: &str,
        _note: Option<&str>,
    ) -> StoreResult<u64> {
        Err(StorageError::Init(
            concat!(
                stringify!(grant_actor_role),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn hybrid_search(
        &self,
        _query: &str,
        _limit: i32,
        _keyword_weight: f32,
        _semantic_weight: f32,
    ) -> StoreResult<Vec<crate::memory::SearchResult>> {
        Err(StorageError::Init(
            concat!(
                stringify!(hybrid_search),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn hybrid_search_filtered(
        &self,
        _query: &str,
        _limit: i32,
        _keyword_weight: f32,
        _semantic_weight: f32,
        _include_types: Option<&[String]>,
        _exclude_types: Option<&[String]>,
    ) -> StoreResult<Vec<crate::memory::SearchResult>> {
        Err(StorageError::Init(
            concat!(
                stringify!(hybrid_search_filtered),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn hygiene_snapshot(&self, _scope: Option<&str>) -> StoreResult<HygieneSnapshot> {
        Err(StorageError::Init(
            concat!(
                stringify!(hygiene_snapshot),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Import a portable archive in memory.
    fn import_portable_archive(
        &self,
        _archive: &PortableArchive,
        _mode: PortableImportMode,
    ) -> StoreResult<PortableImportReport> {
        Err(StorageError::Init(
            concat!(
                stringify!(import_portable_archive),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Import a portable archive from a path.
    fn import_portable_archive_from_path(
        &self,
        _path: &Path,
        _mode: PortableImportMode,
    ) -> StoreResult<PortableImportReport> {
        Err(StorageError::Init(
            concat!(
                stringify!(import_portable_archive_from_path),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Ingest a memory into the default scope.
    fn ingest(&self, _input: IngestInput) -> StoreResult<KnowledgeNode> {
        Err(StorageError::Init(
            concat!(stringify!(ingest), " is not implemented by this backend").into(),
        ))
    }

    /// Ingest a memory into a named project scope.
    fn ingest_in_scope(&self, _input: IngestInput, _scope: &str) -> StoreResult<KnowledgeNode> {
        Err(StorageError::Init(
            concat!(
                stringify!(ingest_in_scope),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Ingest into a scope with an explicit secret policy.
    fn ingest_in_scope_with_secret_policy(
        &self,
        _input: IngestInput,
        _scope: &str,
        _policy: SecretPolicy,
    ) -> StoreResult<KnowledgeNode> {
        Err(StorageError::Init(
            concat!(
                stringify!(ingest_in_scope_with_secret_policy),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Ingest into the default scope with an explicit secret policy.
    fn ingest_with_secret_policy(
        &self,
        _input: IngestInput,
        _policy: SecretPolicy,
    ) -> StoreResult<KnowledgeNode> {
        Err(StorageError::Init(
            concat!(
                stringify!(ingest_with_secret_policy),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// True only for the Strata log backend. The default is false so a probe
    /// never calls [`Self::db_path`], whose default panics.
    fn is_strata_log(&self) -> bool {
        false
    }

    /// Evidence snapshot of one memory inside intention evaluation.
    fn intention_memory_snapshot(
        &self,
        _scope: &str,
        _memory_id: &str,
        _now: DateTime<Utc>,
    ) -> Result<Value, String> {
        Err(concat!(
            stringify!(intention_memory_snapshot),
            " is not implemented by this backend"
        )
        .to_string())
    }

    /// Timestamp of the newest backup, if any.
    fn last_backup_timestamp(&self) -> Option<DateTime<Utc>> {
        unimplemented!(concat!(
            stringify!(last_backup_timestamp),
            " is not implemented by this backend"
        ))
    }

    /// Failed tool calls for the session funnel.
    fn last_session_failed_calls(&self, _run_id: Option<&str>) -> StoreResult<Vec<FailedToolCall>> {
        Err(StorageError::Init(
            concat!(
                stringify!(last_session_failed_calls),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn latest_receipt_chain_entry(&self) -> StoreResult<Option<ChainEntry>> {
        Err(StorageError::Init(
            concat!(
                stringify!(latest_receipt_chain_entry),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// The latest `eff-` effect-receipt id naming `node_id`, or `None` when
    /// the backend proves no per-node effects. Pure lookup: reads nothing
    /// else and writes nothing.
    fn latest_receipt_id_for_node(&self, _node_id: &str) -> Option<String> {
        None
    }

    /// Attach a receipt to a run; true on new link.
    fn link_receipt_to_run(&self, _receipt_id: &str, _run_id: &str) -> StoreResult<bool> {
        Err(StorageError::Init(
            concat!(
                stringify!(link_receipt_to_run),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Recent agent run summaries.
    fn list_agent_runs(&self, _limit: usize) -> StoreResult<Vec<AgentRunSummary>> {
        Err(StorageError::Init(
            concat!(
                stringify!(list_agent_runs),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Endorsement event feed with filters.
    fn list_endorsement_events(
        &self,
        _memory_id: Option<&str>,
        _actor_did: Option<&str>,
        _limit: usize,
    ) -> StoreResult<Vec<EndorsementEventRecord>> {
        Err(StorageError::Init(
            concat!(
                stringify!(list_endorsement_events),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn list_memory_prs(
        &self,
        _status: Option<MemoryPrStatus>,
        _limit: usize,
    ) -> StoreResult<Vec<MemoryPr>> {
        Err(StorageError::Init(
            concat!(
                stringify!(list_memory_prs),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn list_merge_operations(&self, _limit: usize) -> StoreResult<Vec<MergeOperation>> {
        Err(StorageError::Init(
            concat!(
                stringify!(list_merge_operations),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Recent receipts.
    fn list_receipts(&self, _limit: usize) -> StoreResult<Vec<Receipt>> {
        Err(StorageError::Init(
            concat!(
                stringify!(list_receipts),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Receipts attached to one run.
    fn list_receipts_for_run(&self, _run_id: &str, _limit: usize) -> StoreResult<Vec<Receipt>> {
        Err(StorageError::Init(
            concat!(
                stringify!(list_receipts_for_run),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Pending reconsolidation plans with reasons.
    fn list_reconsolidation_plans(&self, _limit: usize) -> StoreResult<Vec<(MergePlan, String)>> {
        Err(StorageError::Init(
            concat!(
                stringify!(list_reconsolidation_plans),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn list_tag_operations(
        &self,
        _limit: usize,
        _scope: Option<&str>,
    ) -> StoreResult<Vec<MergeOperation>> {
        Err(StorageError::Init(
            concat!(
                stringify!(list_tag_operations),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Active synaptic tags for context assembly.
    fn load_active_synaptic_tags(&self) -> StoreResult<Vec<SynapticTag>> {
        Err(StorageError::Init(
            concat!(
                stringify!(load_active_synaptic_tags),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Weakest nodes by retention strength.
    fn lowest_retention_nodes(&self, _limit: usize) -> StoreResult<Vec<(String, f64)>> {
        Err(StorageError::Init(
            concat!(
                stringify!(lowest_retention_nodes),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Batched GC maintenance with retention and age gates.
    fn maintain_gc_batch(
        &self,
        _limit: usize,
        _after: Option<&str>,
        _budget_ms: u64,
        _dry_run: bool,
        _min_retention: f64,
        _max_age_days: Option<u64>,
    ) -> StoreResult<Value> {
        Err(StorageError::Init(
            concat!(
                stringify!(maintain_gc_batch),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Batched lifecycle maintenance with cursor and budget.
    fn maintain_lifecycle_batch(
        &self,
        _limit: usize,
        _after: Option<&str>,
        _budget_ms: u64,
        _dry_run: bool,
    ) -> StoreResult<Value> {
        Err(StorageError::Init(
            concat!(
                stringify!(maintain_lifecycle_batch),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Batched access-log maintenance.
    fn maintain_log_batch(&self, _limit: usize, _dry_run: bool) -> StoreResult<Value> {
        Err(StorageError::Init(
            concat!(
                stringify!(maintain_log_batch),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn maintenance_memory_page(
        &self,
        _limit: usize,
        _after: Option<&str>,
        _scope: &str,
    ) -> StoreResult<(Vec<crate::KnowledgeNode>, bool)> {
        Err(StorageError::Init(
            concat!(
                stringify!(maintenance_memory_page),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn mark_reviewed(&self, _id: &str, _rating: Rating) -> StoreResult<KnowledgeNode> {
        Err(StorageError::Init(
            concat!(
                stringify!(mark_reviewed),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn merge_candidates(
        &self,
        _policy: MergePolicy,
        _limit: usize,
        _tag_filter: &[String],
    ) -> StoreResult<Vec<MergeCandidate>> {
        Err(StorageError::Init(
            concat!(
                stringify!(merge_candidates),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Undo one merge operation by id.
    fn merge_undo(&self, _op_id: &str) -> StoreResult<MergeOperation> {
        Err(StorageError::Init(
            concat!(
                stringify!(merge_undo),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Gate-space effect seq of the write that last named `node_id`, or
    /// `None` when the backend proves no per-node effects. Companion of
    /// [`Self::latest_receipt_id_for_node`]: the receipt id is
    /// `eff-<effect_seq:016x>`.
    fn node_effect_seq(&self, _node_id: &str) -> Option<u64> {
        None
    }

    /// Check node membership in a project scope.
    fn node_is_in_scope(&self, _id: &str, _scope: &str) -> StoreResult<bool> {
        Err(StorageError::Init(
            concat!(
                stringify!(node_is_in_scope),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Open failures touching a file set.
    fn open_failures_touching(
        &self,
        _changed_files: &[String],
    ) -> StoreResult<Vec<OpenFailureTouching>> {
        Err(StorageError::Init(
            concat!(
                stringify!(open_failures_touching),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn preview_tag_mutation(
        &self,
        _source_tags: &[String],
        _target_tag: &str,
        _scope: Option<&str>,
    ) -> StoreResult<serde_json::Value> {
        Err(StorageError::Init(
            concat!(
                stringify!(preview_tag_mutation),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Read the process actor DID.
    fn process_actor_did(&self) -> Option<String> {
        unimplemented!(concat!(
            stringify!(process_actor_did),
            " is not implemented by this backend"
        ))
    }

    /// Synaptic-tagging ingest pipeline step.
    fn process_synaptic_ingest(
        &self,
        _request: &SynapticIngestRequest,
    ) -> StoreResult<SynapticIngestOutcome> {
        Err(StorageError::Init(
            concat!(
                stringify!(process_synaptic_ingest),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Durable projection candidates in a scope.
    fn projection_candidates(
        &self,
        _scope: &str,
        _min_retention: f64,
        _limit: i32,
    ) -> StoreResult<Vec<KnowledgeNode>> {
        Err(StorageError::Init(
            concat!(
                stringify!(projection_candidates),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Promote a memory's retrieval strength and reps.
    fn promote_memory(&self, _id: &str) -> StoreResult<KnowledgeNode> {
        Err(StorageError::Init(
            concat!(
                stringify!(promote_memory),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn promote_memory_as_actor(
        &self,
        _id: &str,
        _claimed_role: Option<&str>,
        _tool: &str,
    ) -> StoreResult<ActorMutationOutcome> {
        Err(StorageError::Init(
            concat!(
                stringify!(promote_memory_as_actor),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Bounded FSRS promotion used by backfill promotion.
    fn promote_memory_backfill(&self, _id: &str) -> StoreResult<KnowledgeNode> {
        Err(StorageError::Init(
            concat!(
                stringify!(promote_memory_backfill),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Prune stale trace rows; returns rows removed.
    fn prune_agent_traces(&self) -> StoreResult<i64> {
        Err(StorageError::Init(
            concat!(
                stringify!(prune_agent_traces),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn purge_node(&self, _id: &str, _reason: Option<&str>) -> StoreResult<PurgeReport> {
        Err(StorageError::Init(
            concat!(
                stringify!(purge_node),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn query_time_range(
        &self,
        _start: Option<DateTime<Utc>>,
        _end: Option<DateTime<Utc>>,
        _limit: i32,
        _node_type: Option<&str>,
        _tags: Option<&[String]>,
    ) -> StoreResult<Vec<KnowledgeNode>> {
        Err(StorageError::Init(
            concat!(
                stringify!(query_time_range),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Product recall: hybrid retrieval over KnowledgeNodes.
    fn recall(&self, _input: RecallInput) -> StoreResult<Vec<KnowledgeNode>> {
        Err(StorageError::Init(
            concat!(stringify!(recall), " is not implemented by this backend").into(),
        ))
    }

    /// Attestation status for one receipt.
    fn receipt_attestation_status(
        &self,
        _receipt_id: &str,
    ) -> StoreResult<Option<ReceiptAttestationStatus>> {
        Err(StorageError::Init(
            concat!(
                stringify!(receipt_attestation_status),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Record one anchor verification result.
    fn record_anchor_verification(
        &self,
        _anchor_id: &str,
        _status: AnchorStatus,
        _checked_at: DateTime<Utc>,
    ) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(record_anchor_verification),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Record retrieval provenance for a batch of ids.
    fn record_batch_retrieval(&self, _ids: &[&str]) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(record_batch_retrieval),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Persist code anchors for memories.
    fn record_code_anchors(&self, _anchors: &[CodeAnchor]) -> StoreResult<usize> {
        Err(StorageError::Init(
            concat!(
                stringify!(record_code_anchors),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn record_composition_outcome(&self, _outcome: &CompositionOutcomeRecord) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(record_composition_outcome),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Record one access-log row for retention math.
    fn record_memory_access(&self, _memory_id: &str) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(record_memory_access),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Record `(scope, intent_id) -> (node_id, effect_seq, response_digest)`
    /// after an intented write was admitted. First write wins: backends that
    /// keep an intent index never overwrite an existing entry. The default
    /// is a silent no-op so backends without intent tracking still compile
    /// and behave as if every submission were fresh.
    fn record_intent_entry(
        &self,
        _scope: &str,
        _intent_id: &str,
        _node_id: &str,
        _effect_seq: u64,
        _response_digest: &str,
    ) -> StoreResult<()> {
        Ok(())
    }

    /// Endorse a reinforce mutation with provenance.
    fn record_reinforce_endorsement(
        &self,
        _id: &str,
        _claimed_role: Option<&str>,
        _tool: &str,
    ) -> StoreResult<ActorMutationOutcome> {
        Err(StorageError::Init(
            concat!(
                stringify!(record_reinforce_endorsement),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn registered_receipt_signing_key(
        &self,
        _key_id: &str,
    ) -> StoreResult<Option<TrustedSigningKey>> {
        Err(StorageError::Init(
            concat!(
                stringify!(registered_receipt_signing_key),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Free a quarantined memory back into retrieval.
    fn release_quarantine(&self, _id: &str) -> StoreResult<KnowledgeNode> {
        Err(StorageError::Init(
            concat!(
                stringify!(release_quarantine),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn replace_code_anchors(
        &self,
        _node_id: &str,
        _scope: &str,
        _anchors: &[CodeAnchor],
    ) -> StoreResult<usize> {
        Err(StorageError::Init(
            concat!(
                stringify!(replace_code_anchors),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Replay a scope's intention event log into a plan.
    fn replay_intention_graph(&self, _scope: &str) -> Result<Value, String> {
        Err(concat!(
            stringify!(replay_intention_graph),
            " is not implemented by this backend"
        )
        .to_string())
    }

    /// Resolve effective role for the current actor.
    fn resolve_actor_role(
        &self,
        _claimed_role: Option<&str>,
    ) -> StoreResult<(String, RoleResolution)> {
        Err(StorageError::Init(
            concat!(
                stringify!(resolve_actor_role),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn resolve_commit_sha_root(&self, _sha_prefix: &str) -> StoreResult<Option<String>> {
        Err(StorageError::Init(
            concat!(
                stringify!(resolve_commit_sha_root),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Resolve @handle/#tag/^sha memory handles to candidates.
    fn resolve_handle(&self, _query: &str) -> HandleResolution {
        unimplemented!(concat!(
            stringify!(resolve_handle),
            " is not implemented by this backend"
        ))
    }

    /// Retire blast-affected memories and report outcomes.
    fn retire_affected(&self, _ids: &[&str], _reason: &str) -> Vec<RetireOutcome> {
        unimplemented!(concat!(
            stringify!(retire_affected),
            " is not implemented by this backend"
        ))
    }

    /// Undo a suppression inside the labile window.
    fn reverse_suppression(&self, _id: &str, _labile_hours: i64) -> StoreResult<KnowledgeNode> {
        Err(StorageError::Init(
            concat!(
                stringify!(reverse_suppression),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Run one full consolidation pass.
    fn run_consolidation(&self) -> StoreResult<ConsolidationResult> {
        Err(StorageError::Init(
            concat!(
                stringify!(run_consolidation),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn save_composition(
        &self,
        _event: &CompositionEventRecord,
        _members: &[CompositionMemberRecord],
        _outcomes: &[CompositionOutcomeRecord],
    ) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(save_composition),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Persist a typed connection edge between memories.
    fn save_connection(&self, _connection: &ConnectionRecord) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(save_connection),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Persist a counterfactual replay receipt.
    fn save_counterfactual_replay_receipt(
        &self,
        _replay_id: &str,
        _receipt: &Receipt,
        _run_id: Option<&str>,
        _tool: Option<&str>,
    ) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(save_counterfactual_replay_receipt),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Persist one dream history row.
    fn save_dream_history(&self, _record: &DreamHistoryRecord) -> StoreResult<i64> {
        Err(StorageError::Init(
            concat!(
                stringify!(save_dream_history),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Persist one insight row.
    fn save_insight(&self, _insight: &InsightRecord) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(save_insight),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Persist one intention row.
    fn save_intention(&self, _intention: &IntentionRecord) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(save_intention),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Persist a memory PR for review.
    fn save_memory_pr(&self, _pr: &MemoryPr) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(save_memory_pr),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Persist a retrieval/mutation receipt.
    fn save_receipt(
        &self,
        _receipt: &Receipt,
        _run_id: Option<&str>,
        _tool: Option<&str>,
        _query: Option<&str>,
    ) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(save_receipt),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn save_retrieval_receipt_with_replay_capsule(
        &self,
        _receipt: &Receipt,
        _run_id: Option<&str>,
        _tool: Option<&str>,
        _draft: &RetrievalReplayCapsuleDraft,
    ) -> StoreResult<DurableRetrievalReplayCapsule> {
        Err(StorageError::Init(
            concat!(
                stringify!(save_retrieval_receipt_with_replay_capsule),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn save_signed_retrieval_receipt_with_replay_capsule_atomic(
        &self,
        _write: SignedReceiptWrite<'_>,
        _draft: &RetrievalReplayCapsuleDraft,
    ) -> StoreResult<DurableSignedRetrievalReceipt> {
        Err(StorageError::Init(
            concat!(
                stringify!(save_signed_retrieval_receipt_with_replay_capsule_atomic),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Persist a synaptic tag episode; returns its id.
    fn save_synaptic_tag(&self, _tag: &SynapticTag) -> StoreResult<String> {
        Err(StorageError::Init(
            concat!(
                stringify!(save_synaptic_tag),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn save_walk_receipt(
        &self,
        _canonical_json: &str,
        _params: &Value,
    ) -> StoreResult<WalkReceiptHandle> {
        Err(StorageError::Init(
            concat!(
                stringify!(save_walk_receipt),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Backend schema self-description for status views.
    fn schema_introspection(&self) -> StoreResult<SchemaIntrospection> {
        Err(StorageError::Init(
            concat!(
                stringify!(schema_introspection),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// FTS text search returning product nodes.
    fn search(&self, _query: &str, _limit: i32) -> StoreResult<Vec<KnowledgeNode>> {
        Err(StorageError::Init(
            concat!(stringify!(search), " is not implemented by this backend").into(),
        ))
    }

    /// Backdate a node for scenario replay and tests.
    fn set_created_at(&self, _id: &str, _when: DateTime<Utc>) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(set_created_at),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn set_merge_policy(&self, _policy: crate::advanced::MergePolicy) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(set_merge_policy),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Persist the process actor DID.
    fn set_process_actor(&self, _did: &str) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(set_process_actor),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn set_protected(&self, _id: &str, _protected: bool) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(set_protected),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Named sidecar directory under the data dir.
    fn sidecar_dir(&self, _name: &str) -> PathBuf {
        unimplemented!(concat!(
            stringify!(sidecar_dir),
            " is not implemented by this backend"
        ))
    }
    /// Register a trusted receipt signing key; true on new registration.
    fn register_receipt_signing_key(&self, _key: &TrustedSigningKey) -> StoreResult<bool> {
        Err(StorageError::Init(
            concat!(
                stringify!(register_receipt_signing_key),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }
    /// Export the whole store as a portable archive in memory.
    fn export_portable_archive(&self) -> StoreResult<PortableArchive> {
        Err(StorageError::Init(
            concat!(
                stringify!(export_portable_archive),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }
    /// Reconcile connector tombstones against live external ids.
    fn reconcile_source_tombstones(
        &self,
        _source_system: &str,
        _scope: &str,
        _live_ids: &[String],
    ) -> StoreResult<ReconcileReport> {
        Err(StorageError::Init(
            concat!(
                stringify!(reconcile_source_tombstones),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }
    /// Fetch the sync cursor for a connector source and scope.
    fn get_connector_cursor(
        &self,
        _source_system: &str,
        _scope: &str,
    ) -> StoreResult<ConnectorCursor> {
        Err(StorageError::Init(
            concat!(
                stringify!(get_connector_cursor),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }
    /// Persist the sync cursor for a connector source.
    fn save_connector_cursor(&self, _cursor: &ConnectorCursor) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(save_connector_cursor),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }
    /// Dream-compile candidate pairs in a scope.
    fn dream_compile_candidates(
        &self,
        _scope: &str,
        _limit: usize,
    ) -> StoreResult<Vec<KnowledgeNode>> {
        Err(StorageError::Init(
            concat!(
                stringify!(dream_compile_candidates),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }
    /// Strengthen a connection edge; true when an edge was updated.
    fn strengthen_connection(
        &self,
        _source_id: &str,
        _target_id: &str,
        _boost: f64,
    ) -> StoreResult<bool> {
        Err(StorageError::Init(
            concat!(
                stringify!(strengthen_connection),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }
    /// Sweep the suppression cascade (Rac1 window) and retire affected memories.
    fn run_rac1_cascade_sweep(&self) -> StoreResult<(usize, usize)> {
        Err(StorageError::Init(
            concat!(
                stringify!(run_rac1_cascade_sweep),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Prediction-error-gated ingest with dedup shield exclusions and labile candidates.
    fn smart_ingest_excluding_in_scope_with_secret_policy_and_labile(
        &self,
        _input: IngestInput,
        _scope: &str,
        _excluded_node_ids: &[String],
        _policy: SecretPolicy,
        _labile: &[LabileCandidate],
    ) -> StoreResult<SmartIngestResult> {
        Err(StorageError::Init(
            concat!(
                stringify!(smart_ingest_excluding_in_scope_with_secret_policy_and_labile),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn snooze_intention(&self, _id: &str, _until: DateTime<Utc>) -> StoreResult<bool> {
        Err(StorageError::Init(
            concat!(
                stringify!(snooze_intention),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Counts across learning states.
    fn state_distribution(&self) -> StoreResult<(i64, i64, i64, i64)> {
        Err(StorageError::Init(
            concat!(
                stringify!(state_distribution),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Set of all superseded node ids.
    fn superseded_node_ids(&self) -> StoreResult<HashSet<String>> {
        Err(StorageError::Init(
            concat!(
                stringify!(superseded_node_ids),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Superseded-id to replacement-id pairs.
    fn supersession_pairs(&self) -> StoreResult<Vec<(String, String)>> {
        Err(StorageError::Init(
            concat!(
                stringify!(supersession_pairs),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Top-down suppression (SIF/Davis Rac1 semantics).
    fn suppress_memory(&self, _id: &str) -> StoreResult<KnowledgeNode> {
        Err(StorageError::Init(
            concat!(
                stringify!(suppress_memory),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Sync the portable archive against a cloud backend.
    fn sync_portable_archive_cloud(
        &self,
        _endpoint: &str,
        _sync_key: &str,
        _encryption_key: Option<String>,
    ) -> StoreResult<PortableSyncReport> {
        Err(StorageError::Init(
            concat!(
                stringify!(sync_portable_archive_cloud),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Sync the portable archive against a file backend.
    fn sync_portable_archive_file(&self, _path: &Path) -> StoreResult<PortableSyncReport> {
        Err(StorageError::Init(
            concat!(
                stringify!(sync_portable_archive_file),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Tag vocabulary with counts.
    fn tag_vocabulary(&self, _scope: Option<&str>) -> StoreResult<TagVocabulary> {
        Err(StorageError::Init(
            concat!(
                stringify!(tag_vocabulary),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn undo_tag_mutation(&self, _operation_id: &str) -> StoreResult<MergeOperation> {
        Err(StorageError::Init(
            concat!(
                stringify!(undo_tag_mutation),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn update_intention_status(&self, _id: &str, _status: &str) -> StoreResult<bool> {
        Err(StorageError::Init(
            concat!(
                stringify!(update_intention_status),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn update_memory_state(
        &self,
        _memory_id: &str,
        _new_state: &str,
        _reason: &str,
    ) -> StoreResult<bool> {
        Err(StorageError::Init(
            concat!(
                stringify!(update_memory_state),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn update_node_content(&self, _id: &str, _new_content: &str) -> StoreResult<()> {
        Err(StorageError::Init(
            concat!(
                stringify!(update_node_content),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// Connector upsert keyed by external source identity.
    fn upsert_by_source(&self, _input: IngestInput) -> StoreResult<SourceUpsertResult> {
        Err(StorageError::Init(
            concat!(
                stringify!(upsert_by_source),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }

    /// See the SQLite reference implementation for semantics.
    fn verify_stored_receipt_attestation(
        &self,
        _receipt_id: &str,
    ) -> StoreResult<Option<StoredReceiptAttestationVerification>> {
        Err(StorageError::Init(
            concat!(
                stringify!(verify_stored_receipt_attestation),
                " is not implemented by this backend"
            )
            .into(),
        ))
    }
}

// ----------------------------------------------------------------------------
// DYN-COMPATIBLE STORAGE TRAIT
// ----------------------------------------------------------------------------

/// Boxed Send future returning a `MemoryStoreResult<T>`, bound to the lifetime
/// of the borrows captured by the call (typically `&self` plus any reference
/// arguments). Used as the return type of every method on the dyn-compatible
/// `MemoryStore` trait below.
pub type BoxedStoreFuture<'a, T> = Pin<Box<dyn Future<Output = MemoryStoreResult<T>> + Send + 'a>>;

/// Dyn-compatible storage trait.
///
/// `MemoryStoreSend` above is the trait users implement; it uses native
/// async-fn-in-trait return types (RPITIT), which gives zero-allocation
/// static dispatch but is not dyn-safe. This trait wraps every method in
/// `Pin<Box<dyn Future + Send + '_>>` so `Arc<dyn MemoryStore>` works for
/// the cognitive module surface and the Phase 1 integration tests.
///
/// Implementations should not target this trait directly; the blanket
/// `impl<T: MemoryStoreSend> MemoryStore for T` adapts every Send-variant
/// implementation automatically. Each call boxes the returned future
/// exactly once, identical to the cost of the previous design.
pub trait MemoryStore: Send + Sync + 'static {
    fn init<'a>(&'a self) -> BoxedStoreFuture<'a, ()>;
    fn health_check<'a>(&'a self) -> BoxedStoreFuture<'a, HealthStatus>;

    fn registered_model<'a>(&'a self) -> BoxedStoreFuture<'a, Option<ModelSignature>>;
    fn register_model<'a>(&'a self, sig: &'a ModelSignature) -> BoxedStoreFuture<'a, ()>;

    fn insert<'a>(&'a self, record: &'a MemoryRecord) -> BoxedStoreFuture<'a, Uuid>;
    fn get<'a>(&'a self, id: Uuid) -> BoxedStoreFuture<'a, Option<MemoryRecord>>;
    fn update<'a>(&'a self, record: &'a MemoryRecord) -> BoxedStoreFuture<'a, ()>;
    fn delete<'a>(&'a self, id: Uuid) -> BoxedStoreFuture<'a, ()>;

    fn search_records<'a>(
        &'a self,
        query: &'a SearchQuery,
    ) -> BoxedStoreFuture<'a, Vec<SearchResult>>;
    fn fts_search<'a>(
        &'a self,
        text: &'a str,
        limit: usize,
    ) -> BoxedStoreFuture<'a, Vec<SearchResult>>;
    fn vector_search<'a>(
        &'a self,
        embedding: &'a [f32],
        limit: usize,
    ) -> BoxedStoreFuture<'a, Vec<SearchResult>>;

    fn get_scheduling<'a>(
        &'a self,
        memory_id: Uuid,
    ) -> BoxedStoreFuture<'a, Option<SchedulingState>>;
    fn update_scheduling<'a>(&'a self, state: &'a SchedulingState) -> BoxedStoreFuture<'a, ()>;
    fn get_due_memories<'a>(
        &'a self,
        before: DateTime<Utc>,
        limit: usize,
    ) -> BoxedStoreFuture<'a, Vec<(MemoryRecord, SchedulingState)>>;

    fn add_edge<'a>(&'a self, edge: &'a MemoryEdge) -> BoxedStoreFuture<'a, ()>;
    fn get_edges<'a>(
        &'a self,
        node_id: Uuid,
        edge_type: Option<&'a str>,
    ) -> BoxedStoreFuture<'a, Vec<MemoryEdge>>;
    fn remove_edge<'a>(&'a self, source: Uuid, target: Uuid) -> BoxedStoreFuture<'a, ()>;
    fn get_neighbors<'a>(
        &'a self,
        node_id: Uuid,
        depth: usize,
    ) -> BoxedStoreFuture<'a, Vec<(MemoryRecord, f64)>>;

    fn list_domains<'a>(&'a self) -> BoxedStoreFuture<'a, Vec<Domain>>;
    fn get_domain<'a>(&'a self, id: &'a str) -> BoxedStoreFuture<'a, Option<Domain>>;
    fn upsert_domain<'a>(&'a self, domain: &'a Domain) -> BoxedStoreFuture<'a, ()>;
    fn delete_domain<'a>(&'a self, id: &'a str) -> BoxedStoreFuture<'a, ()>;
    fn classify<'a>(&'a self, embedding: &'a [f32]) -> BoxedStoreFuture<'a, Vec<(String, f64)>>;

    fn count<'a>(&'a self) -> BoxedStoreFuture<'a, usize>;
    fn get_store_stats<'a>(&'a self) -> BoxedStoreFuture<'a, StoreStats>;
    fn vacuum<'a>(&'a self) -> BoxedStoreFuture<'a, ()>;

    // --- Phase 4 product seam (sync; dyn-compatible, forwarded from MemoryStoreSend) ---
    fn admit_projection(
        &self,
        memory_ids: &[String],
        target: &str,
        region: &[u8],
    ) -> StoreResult<(String, String)>;
    fn actor_policy_snapshot(&self) -> StoreResult<ActorPolicySnapshot>;
    fn append_mcp_call_outcome(
        &self,
        run_id: &str,
        tool: &str,
        success: bool,
        error: Option<&str>,
        at_ms: i64,
    ) -> StoreResult<()>;
    fn append_trace_event(&self, event: &MemoryTraceEvent) -> StoreResult<i64>;
    fn apply_failure_feedback(
        &self,
        failure_id: &str,
        window: Duration,
    ) -> StoreResult<FailureFeedbackReport>;
    fn apply_intention_graph(
        &self,
        scope: &str,
        command: Command,
        now: DateTime<Utc>,
    ) -> Result<Value, String>;
    fn apply_tag_mutation(
        &self,
        source_tags: &[String],
        target_tag: &str,
        scope: Option<&str>,
        preview_token: &str,
        op_type: &str,
        reason: &str,
    ) -> StoreResult<MergeOperation>;
    fn backup_to(&self, path: &Path) -> StoreResult<()>;
    fn blast_radius(&self, root_id: &str, open_only: bool) -> StoreResult<BlastReport>;
    fn blast_radius_with_link_types(
        &self,
        root_id: &str,
        open_only: bool,
        link_types: &[&str],
    ) -> StoreResult<BlastReport>;
    fn capture_synaptic_event(
        &self,
        request: &SynapticCaptureRequest,
    ) -> StoreResult<DurableSynapticCapture>;
    fn checkpoint_wal(&self, mode: WalCheckpointMode) -> StoreResult<WalCheckpointStatus>;
    fn clear_dream_page_tags(
        &self,
        ids: &[String],
        started_at: DateTime<Utc>,
    ) -> StoreResult<usize>;
    fn closed_issue_nodes(
        &self,
        source_system: &str,
        scope: &str,
    ) -> StoreResult<Vec<ClosedIssueNode>>;
    fn code_anchors_for_node(&self, node_id: &str) -> StoreResult<Vec<CodeAnchor>>;
    fn code_anchors_for_nodes(
        &self,
        node_ids: &[String],
    ) -> StoreResult<HashMap<String, Vec<CodeAnchor>>>;
    fn commit_intention_check(
        &self,
        changes: &[(IntentionRecord, IntentionRecord)],
    ) -> Result<(), String>;
    fn concrete_search_filtered(
        &self,
        query: &str,
        limit: i32,
        include_types: Option<&[String]>,
        exclude_types: Option<&[String]>,
    ) -> StoreResult<Vec<crate::memory::SearchResult>>;
    fn count_memories_below_retention(&self, threshold: f64) -> StoreResult<i64>;
    fn count_memories_since(&self, since: DateTime<Utc>) -> StoreResult<i64>;
    fn count_pending_memory_prs(&self) -> StoreResult<i64>;
    fn count_suppressed(&self) -> StoreResult<usize>;
    fn coverage_snapshot(&self) -> StoreResult<CoverageSnapshot>;
    fn create_context_ablation_replay(
        &self,
        source_receipt_id: &str,
        withheld_slots: &[String],
    ) -> StoreResult<DurableCounterfactualReplay>;
    fn replay_receipt(&self, receipt_id: &str) -> StoreResult<Value>;
    fn current_code_context_nodes(
        &self,
        node_type: &str,
        tag: Option<&str>,
        scope: &str,
        limit: i32,
    ) -> StoreResult<Vec<KnowledgeNode>>;
    fn current_code_context_scope_counts(
        &self,
        node_type: &str,
        tag: Option<&str>,
    ) -> StoreResult<Vec<(String, usize)>>;
    fn data_dir(&self) -> &Path;
    fn db_path(&self) -> &Path;
    /// True when this store is the Strata log. Default backends are not.
    fn is_strata(&self) -> bool;
    fn decide_memory_pr(&self, id: &str, action: MemoryPrAction) -> StoreResult<MemoryPr>;
    fn decide_pending_memory_mutation(
        &self,
        id: &str,
        action: MemoryPrAction,
    ) -> StoreResult<Option<PendingMemoryMutationDecision>>;
    fn delete_node(&self, id: &str) -> StoreResult<bool>;
    fn demote_memory(&self, id: &str) -> StoreResult<KnowledgeNode>;
    fn demote_memory_as_actor(
        &self,
        id: &str,
        claimed_role: Option<&str>,
        tool: &str,
    ) -> StoreResult<ActorMutationOutcome>;
    fn due_for_review_node_ids(&self, limit: usize) -> StoreResult<Vec<String>>;
    fn expire_stale_reconsolidation_plans(&self) -> StoreResult<Vec<String>>;
    fn export_portable_archive_to_path(&self, path: &Path) -> StoreResult<PortableArchive>;
    fn find_duplicate_by_canonical_hash(
        &self,
        scope: &str,
        content: &str,
    ) -> StoreResult<Option<String>>;
    fn find_intent_record(
        &self,
        scope: &str,
        intent_id: &str,
    ) -> StoreResult<Option<(String, u64, String)>>;
    fn get_active_intentions(&self) -> StoreResult<Vec<IntentionRecord>>;
    fn get_active_intentions_in_scope(&self, scope: &str) -> StoreResult<Vec<IntentionRecord>>;
    fn get_agent_run(&self, run_id: &str) -> StoreResult<Option<AgentRunSummary>>;
    fn get_all_connections(&self) -> StoreResult<Vec<ConnectionRecord>>;
    fn get_all_nodes(&self, limit: i32, offset: i32) -> StoreResult<Vec<KnowledgeNode>>;
    fn get_all_nodes_in_scope(
        &self,
        scope: &str,
        limit: i32,
        offset: i32,
    ) -> StoreResult<Vec<KnowledgeNode>>;
    fn get_avg_retention(&self) -> StoreResult<f64>;
    fn get_composition_event(&self, id: &str) -> StoreResult<Option<CompositionEventRecord>>;
    fn get_composition_members(&self, event_id: &str) -> StoreResult<Vec<CompositionMemberRecord>>;
    fn get_composition_neighbors(
        &self,
        memory_id: &str,
        limit: i32,
    ) -> StoreResult<Vec<CompositionNeighborRecord>>;
    fn get_composition_outcomes(
        &self,
        event_id: &str,
    ) -> StoreResult<Vec<CompositionOutcomeRecord>>;
    fn get_compositions_for_memory(
        &self,
        memory_id: &str,
        limit: i32,
    ) -> StoreResult<Vec<CompositionEventRecord>>;
    fn get_connections_for_memory(&self, memory_id: &str) -> StoreResult<Vec<ConnectionRecord>>;
    fn get_consolidation_history(&self, limit: i32)
    -> StoreResult<Vec<ConsolidationHistoryRecord>>;
    fn get_context_ablation_replay(
        &self,
        replay_id: &str,
    ) -> StoreResult<Option<StoredCounterfactualReplay>>;
    fn get_dream_history(&self, limit: i32) -> StoreResult<Vec<DreamHistoryRecord>>;
    fn get_insights(&self, limit: i32) -> StoreResult<Vec<InsightRecord>>;
    fn get_intention(&self, id: &str) -> StoreResult<Option<IntentionRecord>>;
    fn get_intentions_by_status(&self, status: &str) -> StoreResult<Vec<IntentionRecord>>;
    fn get_last_consolidation(&self) -> StoreResult<Option<DateTime<Utc>>>;
    fn get_last_dream(&self) -> StoreResult<Option<DateTime<Utc>>>;
    fn get_memory_pr(&self, id: &str) -> StoreResult<Option<MemoryPr>>;
    fn get_memory_subgraph(
        &self,
        center_id: &str,
        depth: u32,
        max_nodes: usize,
    ) -> StoreResult<(Vec<KnowledgeNode>, Vec<ConnectionRecord>)>;
    fn get_merge_operation(&self, operation_id: &str) -> StoreResult<Option<MergeOperation>>;
    fn get_merge_policy(&self) -> StoreResult<MergePolicy>;
    fn get_most_connected_memory(&self) -> StoreResult<Option<String>>;
    fn get_never_composed_candidates(
        &self,
        limit: i32,
        tag_filter: Option<&[String]>,
    ) -> StoreResult<Vec<NeverComposedCandidate>>;
    fn get_never_composed_candidates_in_scope(
        &self,
        limit: i32,
        tag_filter: Option<&[String]>,
        scope: Option<&str>,
    ) -> StoreResult<Vec<NeverComposedCandidate>>;
    fn get_node(&self, id: &str) -> StoreResult<Option<KnowledgeNode>>;
    fn get_overdue_intentions(&self) -> StoreResult<Vec<IntentionRecord>>;
    fn get_receipt(&self, receipt_id: &str) -> StoreResult<Option<Receipt>>;
    fn get_receipt_attestation_envelope(
        &self,
        receipt_id: &str,
    ) -> StoreResult<Option<DsseEnvelope>>;
    fn get_recent_composition_events(&self, limit: i32)
    -> StoreResult<Vec<CompositionEventRecord>>;
    fn get_recent_composition_events_page(
        &self,
        limit: i32,
        offset: i32,
    ) -> StoreResult<Vec<CompositionEventRecord>>;
    fn get_recent_connections(&self, limit: usize) -> StoreResult<Vec<ConnectionRecord>>;
    fn get_recent_state_transitions(&self, limit: i32) -> StoreResult<Vec<StateTransitionRecord>>;
    fn get_retention_distribution(&self) -> StoreResult<Vec<(String, i64)>>;
    fn get_retention_trend(&self) -> StoreResult<String>;
    fn get_retrieval_replay_capsule(
        &self,
        source_receipt_id: &str,
    ) -> StoreResult<Option<RetrievalReplayCapsuleSummary>>;
    fn get_review_queue(&self, limit: i32) -> StoreResult<Vec<KnowledgeNode>>;
    fn get_state_transitions(
        &self,
        memory_id: &str,
        limit: i32,
    ) -> StoreResult<Vec<StateTransitionRecord>>;
    fn get_stats(&self) -> StoreResult<MemoryStats>;
    fn get_trace(&self, run_id: &str) -> StoreResult<Vec<MemoryTraceEvent>>;
    fn get_walk_receipt(&self, receipt_id: &str) -> StoreResult<Option<StoredWalkReceipt>>;
    fn git_commit_nodes(&self, limit: usize) -> StoreResult<Vec<GitCommitNode>>;
    fn grant_actor_role(&self, actor_did: &str, role: &str, note: Option<&str>)
    -> StoreResult<u64>;
    fn hybrid_search(
        &self,
        query: &str,
        limit: i32,
        keyword_weight: f32,
        semantic_weight: f32,
    ) -> StoreResult<Vec<crate::memory::SearchResult>>;
    fn hybrid_search_filtered(
        &self,
        query: &str,
        limit: i32,
        keyword_weight: f32,
        semantic_weight: f32,
        include_types: Option<&[String]>,
        exclude_types: Option<&[String]>,
    ) -> StoreResult<Vec<crate::memory::SearchResult>>;
    fn hygiene_snapshot(&self, scope: Option<&str>) -> StoreResult<HygieneSnapshot>;
    fn import_portable_archive(
        &self,
        archive: &PortableArchive,
        mode: PortableImportMode,
    ) -> StoreResult<PortableImportReport>;
    fn import_portable_archive_from_path(
        &self,
        path: &Path,
        mode: PortableImportMode,
    ) -> StoreResult<PortableImportReport>;
    fn ingest(&self, input: IngestInput) -> StoreResult<KnowledgeNode>;
    fn ingest_in_scope(&self, input: IngestInput, scope: &str) -> StoreResult<KnowledgeNode>;
    fn ingest_in_scope_with_secret_policy(
        &self,
        input: IngestInput,
        scope: &str,
        policy: SecretPolicy,
    ) -> StoreResult<KnowledgeNode>;
    fn ingest_with_secret_policy(
        &self,
        input: IngestInput,
        policy: SecretPolicy,
    ) -> StoreResult<KnowledgeNode>;
    fn is_strata_log(&self) -> bool;
    fn intention_memory_snapshot(
        &self,
        scope: &str,
        memory_id: &str,
        now: DateTime<Utc>,
    ) -> Result<Value, String>;
    fn last_backup_timestamp(&self) -> Option<DateTime<Utc>>;
    fn last_session_failed_calls(&self, run_id: Option<&str>) -> StoreResult<Vec<FailedToolCall>>;
    fn latest_receipt_chain_entry(&self) -> StoreResult<Option<ChainEntry>>;
    fn latest_receipt_id_for_node(&self, node_id: &str) -> Option<String>;
    fn link_receipt_to_run(&self, receipt_id: &str, run_id: &str) -> StoreResult<bool>;
    fn list_agent_runs(&self, limit: usize) -> StoreResult<Vec<AgentRunSummary>>;
    fn list_endorsement_events(
        &self,
        memory_id: Option<&str>,
        actor_did: Option<&str>,
        limit: usize,
    ) -> StoreResult<Vec<EndorsementEventRecord>>;
    fn list_memory_prs(
        &self,
        status: Option<MemoryPrStatus>,
        limit: usize,
    ) -> StoreResult<Vec<MemoryPr>>;
    fn list_merge_operations(&self, limit: usize) -> StoreResult<Vec<MergeOperation>>;
    fn list_receipts(&self, limit: usize) -> StoreResult<Vec<Receipt>>;
    fn list_receipts_for_run(&self, run_id: &str, limit: usize) -> StoreResult<Vec<Receipt>>;
    fn list_reconsolidation_plans(&self, limit: usize) -> StoreResult<Vec<(MergePlan, String)>>;
    fn list_tag_operations(
        &self,
        limit: usize,
        scope: Option<&str>,
    ) -> StoreResult<Vec<MergeOperation>>;
    fn load_active_synaptic_tags(&self) -> StoreResult<Vec<SynapticTag>>;
    fn lowest_retention_nodes(&self, limit: usize) -> StoreResult<Vec<(String, f64)>>;
    fn maintain_gc_batch(
        &self,
        limit: usize,
        after: Option<&str>,
        budget_ms: u64,
        dry_run: bool,
        min_retention: f64,
        max_age_days: Option<u64>,
    ) -> StoreResult<Value>;
    fn maintain_lifecycle_batch(
        &self,
        limit: usize,
        after: Option<&str>,
        budget_ms: u64,
        dry_run: bool,
    ) -> StoreResult<Value>;
    fn maintain_log_batch(&self, limit: usize, dry_run: bool) -> StoreResult<Value>;
    fn maintenance_memory_page(
        &self,
        limit: usize,
        after: Option<&str>,
        scope: &str,
    ) -> StoreResult<(Vec<crate::KnowledgeNode>, bool)>;
    fn mark_reviewed(&self, id: &str, rating: Rating) -> StoreResult<KnowledgeNode>;
    fn merge_candidates(
        &self,
        policy: MergePolicy,
        limit: usize,
        tag_filter: &[String],
    ) -> StoreResult<Vec<MergeCandidate>>;
    fn merge_undo(&self, op_id: &str) -> StoreResult<MergeOperation>;
    fn node_effect_seq(&self, node_id: &str) -> Option<u64>;
    fn node_is_in_scope(&self, id: &str, scope: &str) -> StoreResult<bool>;
    fn open_failures_touching(
        &self,
        changed_files: &[String],
    ) -> StoreResult<Vec<OpenFailureTouching>>;
    fn preview_tag_mutation(
        &self,
        source_tags: &[String],
        target_tag: &str,
        scope: Option<&str>,
    ) -> StoreResult<serde_json::Value>;
    fn process_actor_did(&self) -> Option<String>;
    fn process_synaptic_ingest(
        &self,
        request: &SynapticIngestRequest,
    ) -> StoreResult<SynapticIngestOutcome>;
    fn projection_candidates(
        &self,
        scope: &str,
        min_retention: f64,
        limit: i32,
    ) -> StoreResult<Vec<KnowledgeNode>>;
    fn promote_memory(&self, id: &str) -> StoreResult<KnowledgeNode>;
    fn promote_memory_as_actor(
        &self,
        id: &str,
        claimed_role: Option<&str>,
        tool: &str,
    ) -> StoreResult<ActorMutationOutcome>;
    fn promote_memory_backfill(&self, id: &str) -> StoreResult<KnowledgeNode>;
    fn prune_agent_traces(&self) -> StoreResult<i64>;
    fn purge_node(&self, id: &str, reason: Option<&str>) -> StoreResult<PurgeReport>;
    fn query_time_range(
        &self,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
        limit: i32,
        node_type: Option<&str>,
        tags: Option<&[String]>,
    ) -> StoreResult<Vec<KnowledgeNode>>;
    fn recall(&self, input: RecallInput) -> StoreResult<Vec<KnowledgeNode>>;
    fn receipt_attestation_status(
        &self,
        receipt_id: &str,
    ) -> StoreResult<Option<ReceiptAttestationStatus>>;
    fn record_anchor_verification(
        &self,
        anchor_id: &str,
        status: AnchorStatus,
        checked_at: DateTime<Utc>,
    ) -> StoreResult<()>;
    fn record_batch_retrieval(&self, ids: &[&str]) -> StoreResult<()>;
    fn record_code_anchors(&self, anchors: &[CodeAnchor]) -> StoreResult<usize>;
    fn record_composition_outcome(&self, outcome: &CompositionOutcomeRecord) -> StoreResult<()>;
    fn record_memory_access(&self, memory_id: &str) -> StoreResult<()>;
    fn record_intent_entry(
        &self,
        scope: &str,
        intent_id: &str,
        node_id: &str,
        effect_seq: u64,
        response_digest: &str,
    ) -> StoreResult<()>;
    fn record_reinforce_endorsement(
        &self,
        id: &str,
        claimed_role: Option<&str>,
        tool: &str,
    ) -> StoreResult<ActorMutationOutcome>;
    fn registered_receipt_signing_key(
        &self,
        key_id: &str,
    ) -> StoreResult<Option<TrustedSigningKey>>;
    fn release_quarantine(&self, id: &str) -> StoreResult<KnowledgeNode>;
    fn replace_code_anchors(
        &self,
        node_id: &str,
        scope: &str,
        anchors: &[CodeAnchor],
    ) -> StoreResult<usize>;
    fn replay_intention_graph(&self, scope: &str) -> Result<Value, String>;
    fn resolve_actor_role(
        &self,
        claimed_role: Option<&str>,
    ) -> StoreResult<(String, RoleResolution)>;
    fn resolve_commit_sha_root(&self, sha_prefix: &str) -> StoreResult<Option<String>>;
    fn resolve_handle(&self, query: &str) -> HandleResolution;
    fn retire_affected(&self, ids: &[&str], reason: &str) -> Vec<RetireOutcome>;
    fn reverse_suppression(&self, id: &str, labile_hours: i64) -> StoreResult<KnowledgeNode>;
    fn run_consolidation(&self) -> StoreResult<ConsolidationResult>;
    fn save_composition(
        &self,
        event: &CompositionEventRecord,
        members: &[CompositionMemberRecord],
        outcomes: &[CompositionOutcomeRecord],
    ) -> StoreResult<()>;
    fn save_connection(&self, connection: &ConnectionRecord) -> StoreResult<()>;
    fn save_counterfactual_replay_receipt(
        &self,
        replay_id: &str,
        receipt: &Receipt,
        run_id: Option<&str>,
        tool: Option<&str>,
    ) -> StoreResult<()>;
    fn save_dream_history(&self, record: &DreamHistoryRecord) -> StoreResult<i64>;
    fn save_insight(&self, insight: &InsightRecord) -> StoreResult<()>;
    fn save_intention(&self, intention: &IntentionRecord) -> StoreResult<()>;
    fn save_memory_pr(&self, pr: &MemoryPr) -> StoreResult<()>;
    fn save_receipt(
        &self,
        receipt: &Receipt,
        run_id: Option<&str>,
        tool: Option<&str>,
        query: Option<&str>,
    ) -> StoreResult<()>;
    fn save_retrieval_receipt_with_replay_capsule(
        &self,
        receipt: &Receipt,
        run_id: Option<&str>,
        tool: Option<&str>,
        draft: &RetrievalReplayCapsuleDraft,
    ) -> StoreResult<DurableRetrievalReplayCapsule>;
    fn save_signed_retrieval_receipt_with_replay_capsule_atomic(
        &self,
        write: SignedReceiptWrite<'_>,
        draft: &RetrievalReplayCapsuleDraft,
    ) -> StoreResult<DurableSignedRetrievalReceipt>;
    fn save_synaptic_tag(&self, tag: &SynapticTag) -> StoreResult<String>;
    fn save_walk_receipt(
        &self,
        canonical_json: &str,
        params: &Value,
    ) -> StoreResult<WalkReceiptHandle>;
    fn schema_introspection(&self) -> StoreResult<SchemaIntrospection>;
    fn search(&self, query: &str, limit: i32) -> StoreResult<Vec<KnowledgeNode>>;
    fn set_created_at(&self, id: &str, when: DateTime<Utc>) -> StoreResult<()>;
    fn set_merge_policy(&self, policy: crate::advanced::MergePolicy) -> StoreResult<()>;
    fn set_process_actor(&self, did: &str) -> StoreResult<()>;
    fn set_protected(&self, id: &str, protected: bool) -> StoreResult<()>;
    fn sidecar_dir(&self, name: &str) -> PathBuf;
    fn register_receipt_signing_key(&self, key: &TrustedSigningKey) -> StoreResult<bool>;
    fn export_portable_archive(&self) -> StoreResult<PortableArchive>;
    fn reconcile_source_tombstones(
        &self,
        source_system: &str,
        scope: &str,
        live_ids: &[String],
    ) -> StoreResult<ReconcileReport>;
    fn get_connector_cursor(
        &self,
        source_system: &str,
        scope: &str,
    ) -> StoreResult<ConnectorCursor>;
    fn save_connector_cursor(&self, cursor: &ConnectorCursor) -> StoreResult<()>;
    fn dream_compile_candidates(
        &self,
        scope: &str,
        limit: usize,
    ) -> StoreResult<Vec<KnowledgeNode>>;
    fn strengthen_connection(
        &self,
        source_id: &str,
        target_id: &str,
        boost: f64,
    ) -> StoreResult<bool>;
    fn run_rac1_cascade_sweep(&self) -> StoreResult<(usize, usize)>;
    fn smart_ingest_excluding_in_scope_with_secret_policy_and_labile(
        &self,
        input: IngestInput,
        scope: &str,
        excluded_node_ids: &[String],
        policy: SecretPolicy,
        labile: &[LabileCandidate],
    ) -> StoreResult<SmartIngestResult>;
    fn snooze_intention(&self, id: &str, until: DateTime<Utc>) -> StoreResult<bool>;
    fn state_distribution(&self) -> StoreResult<(i64, i64, i64, i64)>;
    fn superseded_node_ids(&self) -> StoreResult<HashSet<String>>;
    fn supersession_pairs(&self) -> StoreResult<Vec<(String, String)>>;
    fn suppress_memory(&self, id: &str) -> StoreResult<KnowledgeNode>;
    fn sync_portable_archive_cloud(
        &self,
        endpoint: &str,
        sync_key: &str,
        encryption_key: Option<String>,
    ) -> StoreResult<PortableSyncReport>;
    fn sync_portable_archive_file(&self, path: &Path) -> StoreResult<PortableSyncReport>;
    fn tag_vocabulary(&self, scope: Option<&str>) -> StoreResult<TagVocabulary>;
    fn undo_tag_mutation(&self, operation_id: &str) -> StoreResult<MergeOperation>;
    fn update_intention_status(&self, id: &str, status: &str) -> StoreResult<bool>;
    fn update_memory_state(
        &self,
        memory_id: &str,
        new_state: &str,
        reason: &str,
    ) -> StoreResult<bool>;
    fn update_node_content(&self, id: &str, new_content: &str) -> StoreResult<()>;
    fn upsert_by_source(&self, input: IngestInput) -> StoreResult<SourceUpsertResult>;
    fn verify_stored_receipt_attestation(
        &self,
        receipt_id: &str,
    ) -> StoreResult<Option<StoredReceiptAttestationVerification>>;
}

impl<T> MemoryStore for T
where
    T: MemoryStoreSend,
{
    fn init<'a>(&'a self) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::init(self))
    }
    fn health_check<'a>(&'a self) -> BoxedStoreFuture<'a, HealthStatus> {
        Box::pin(<T as MemoryStoreSend>::health_check(self))
    }

    fn registered_model<'a>(&'a self) -> BoxedStoreFuture<'a, Option<ModelSignature>> {
        Box::pin(<T as MemoryStoreSend>::registered_model(self))
    }
    fn register_model<'a>(&'a self, sig: &'a ModelSignature) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::register_model(self, sig))
    }

    fn insert<'a>(&'a self, record: &'a MemoryRecord) -> BoxedStoreFuture<'a, Uuid> {
        Box::pin(<T as MemoryStoreSend>::insert(self, record))
    }
    fn get<'a>(&'a self, id: Uuid) -> BoxedStoreFuture<'a, Option<MemoryRecord>> {
        Box::pin(<T as MemoryStoreSend>::get(self, id))
    }
    fn update<'a>(&'a self, record: &'a MemoryRecord) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::update(self, record))
    }
    fn delete<'a>(&'a self, id: Uuid) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::delete(self, id))
    }

    fn search_records<'a>(
        &'a self,
        query: &'a SearchQuery,
    ) -> BoxedStoreFuture<'a, Vec<SearchResult>> {
        Box::pin(<T as MemoryStoreSend>::search_records(self, query))
    }
    fn fts_search<'a>(
        &'a self,
        text: &'a str,
        limit: usize,
    ) -> BoxedStoreFuture<'a, Vec<SearchResult>> {
        Box::pin(<T as MemoryStoreSend>::fts_search(self, text, limit))
    }
    fn vector_search<'a>(
        &'a self,
        embedding: &'a [f32],
        limit: usize,
    ) -> BoxedStoreFuture<'a, Vec<SearchResult>> {
        Box::pin(<T as MemoryStoreSend>::vector_search(
            self, embedding, limit,
        ))
    }

    fn get_scheduling<'a>(
        &'a self,
        memory_id: Uuid,
    ) -> BoxedStoreFuture<'a, Option<SchedulingState>> {
        Box::pin(<T as MemoryStoreSend>::get_scheduling(self, memory_id))
    }
    fn update_scheduling<'a>(&'a self, state: &'a SchedulingState) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::update_scheduling(self, state))
    }
    fn get_due_memories<'a>(
        &'a self,
        before: DateTime<Utc>,
        limit: usize,
    ) -> BoxedStoreFuture<'a, Vec<(MemoryRecord, SchedulingState)>> {
        Box::pin(<T as MemoryStoreSend>::get_due_memories(
            self, before, limit,
        ))
    }

    fn add_edge<'a>(&'a self, edge: &'a MemoryEdge) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::add_edge(self, edge))
    }
    fn get_edges<'a>(
        &'a self,
        node_id: Uuid,
        edge_type: Option<&'a str>,
    ) -> BoxedStoreFuture<'a, Vec<MemoryEdge>> {
        Box::pin(<T as MemoryStoreSend>::get_edges(self, node_id, edge_type))
    }
    fn remove_edge<'a>(&'a self, source: Uuid, target: Uuid) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::remove_edge(self, source, target))
    }
    fn get_neighbors<'a>(
        &'a self,
        node_id: Uuid,
        depth: usize,
    ) -> BoxedStoreFuture<'a, Vec<(MemoryRecord, f64)>> {
        Box::pin(<T as MemoryStoreSend>::get_neighbors(self, node_id, depth))
    }

    fn list_domains<'a>(&'a self) -> BoxedStoreFuture<'a, Vec<Domain>> {
        Box::pin(<T as MemoryStoreSend>::list_domains(self))
    }
    fn get_domain<'a>(&'a self, id: &'a str) -> BoxedStoreFuture<'a, Option<Domain>> {
        Box::pin(<T as MemoryStoreSend>::get_domain(self, id))
    }
    fn upsert_domain<'a>(&'a self, domain: &'a Domain) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::upsert_domain(self, domain))
    }
    fn delete_domain<'a>(&'a self, id: &'a str) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::delete_domain(self, id))
    }
    fn classify<'a>(&'a self, embedding: &'a [f32]) -> BoxedStoreFuture<'a, Vec<(String, f64)>> {
        Box::pin(<T as MemoryStoreSend>::classify(self, embedding))
    }

    fn count<'a>(&'a self) -> BoxedStoreFuture<'a, usize> {
        Box::pin(<T as MemoryStoreSend>::count(self))
    }
    fn get_store_stats<'a>(&'a self) -> BoxedStoreFuture<'a, StoreStats> {
        Box::pin(<T as MemoryStoreSend>::get_store_stats(self))
    }
    fn vacuum<'a>(&'a self) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::vacuum(self))
    }

    fn admit_projection(
        &self,
        memory_ids: &[String],
        target: &str,
        region: &[u8],
    ) -> StoreResult<(String, String)> {
        <T as MemoryStoreSend>::admit_projection(self, memory_ids, target, region)
    }
    fn actor_policy_snapshot(&self) -> StoreResult<ActorPolicySnapshot> {
        <T as MemoryStoreSend>::actor_policy_snapshot(self)
    }
    fn append_mcp_call_outcome(
        &self,
        run_id: &str,
        tool: &str,
        success: bool,
        error: Option<&str>,
        at_ms: i64,
    ) -> StoreResult<()> {
        <T as MemoryStoreSend>::append_mcp_call_outcome(self, run_id, tool, success, error, at_ms)
    }
    fn append_trace_event(&self, event: &MemoryTraceEvent) -> StoreResult<i64> {
        <T as MemoryStoreSend>::append_trace_event(self, event)
    }
    fn apply_failure_feedback(
        &self,
        failure_id: &str,
        window: Duration,
    ) -> StoreResult<FailureFeedbackReport> {
        <T as MemoryStoreSend>::apply_failure_feedback(self, failure_id, window)
    }
    fn apply_intention_graph(
        &self,
        scope: &str,
        command: Command,
        now: DateTime<Utc>,
    ) -> Result<Value, String> {
        <T as MemoryStoreSend>::apply_intention_graph(self, scope, command, now)
    }
    fn apply_tag_mutation(
        &self,
        source_tags: &[String],
        target_tag: &str,
        scope: Option<&str>,
        preview_token: &str,
        op_type: &str,
        reason: &str,
    ) -> StoreResult<MergeOperation> {
        <T as MemoryStoreSend>::apply_tag_mutation(
            self,
            source_tags,
            target_tag,
            scope,
            preview_token,
            op_type,
            reason,
        )
    }
    fn backup_to(&self, path: &Path) -> StoreResult<()> {
        <T as MemoryStoreSend>::backup_to(self, path)
    }
    fn blast_radius(&self, root_id: &str, open_only: bool) -> StoreResult<BlastReport> {
        <T as MemoryStoreSend>::blast_radius(self, root_id, open_only)
    }
    fn blast_radius_with_link_types(
        &self,
        root_id: &str,
        open_only: bool,
        link_types: &[&str],
    ) -> StoreResult<BlastReport> {
        <T as MemoryStoreSend>::blast_radius_with_link_types(self, root_id, open_only, link_types)
    }
    fn capture_synaptic_event(
        &self,
        request: &SynapticCaptureRequest,
    ) -> StoreResult<DurableSynapticCapture> {
        <T as MemoryStoreSend>::capture_synaptic_event(self, request)
    }
    fn checkpoint_wal(&self, mode: WalCheckpointMode) -> StoreResult<WalCheckpointStatus> {
        <T as MemoryStoreSend>::checkpoint_wal(self, mode)
    }
    fn clear_dream_page_tags(
        &self,
        ids: &[String],
        started_at: DateTime<Utc>,
    ) -> StoreResult<usize> {
        <T as MemoryStoreSend>::clear_dream_page_tags(self, ids, started_at)
    }
    fn closed_issue_nodes(
        &self,
        source_system: &str,
        scope: &str,
    ) -> StoreResult<Vec<ClosedIssueNode>> {
        <T as MemoryStoreSend>::closed_issue_nodes(self, source_system, scope)
    }
    fn code_anchors_for_node(&self, node_id: &str) -> StoreResult<Vec<CodeAnchor>> {
        <T as MemoryStoreSend>::code_anchors_for_node(self, node_id)
    }
    fn code_anchors_for_nodes(
        &self,
        node_ids: &[String],
    ) -> StoreResult<HashMap<String, Vec<CodeAnchor>>> {
        <T as MemoryStoreSend>::code_anchors_for_nodes(self, node_ids)
    }
    fn commit_intention_check(
        &self,
        changes: &[(IntentionRecord, IntentionRecord)],
    ) -> Result<(), String> {
        <T as MemoryStoreSend>::commit_intention_check(self, changes)
    }
    fn concrete_search_filtered(
        &self,
        query: &str,
        limit: i32,
        include_types: Option<&[String]>,
        exclude_types: Option<&[String]>,
    ) -> StoreResult<Vec<crate::memory::SearchResult>> {
        <T as MemoryStoreSend>::concrete_search_filtered(
            self,
            query,
            limit,
            include_types,
            exclude_types,
        )
    }
    fn count_memories_below_retention(&self, threshold: f64) -> StoreResult<i64> {
        <T as MemoryStoreSend>::count_memories_below_retention(self, threshold)
    }
    fn count_memories_since(&self, since: DateTime<Utc>) -> StoreResult<i64> {
        <T as MemoryStoreSend>::count_memories_since(self, since)
    }
    fn count_pending_memory_prs(&self) -> StoreResult<i64> {
        <T as MemoryStoreSend>::count_pending_memory_prs(self)
    }
    fn count_suppressed(&self) -> StoreResult<usize> {
        <T as MemoryStoreSend>::count_suppressed(self)
    }
    fn coverage_snapshot(&self) -> StoreResult<CoverageSnapshot> {
        <T as MemoryStoreSend>::coverage_snapshot(self)
    }
    fn create_context_ablation_replay(
        &self,
        source_receipt_id: &str,
        withheld_slots: &[String],
    ) -> StoreResult<DurableCounterfactualReplay> {
        <T as MemoryStoreSend>::create_context_ablation_replay(
            self,
            source_receipt_id,
            withheld_slots,
        )
    }
    fn replay_receipt(&self, receipt_id: &str) -> StoreResult<Value> {
        <T as MemoryStoreSend>::replay_receipt(self, receipt_id)
    }
    fn current_code_context_nodes(
        &self,
        node_type: &str,
        tag: Option<&str>,
        scope: &str,
        limit: i32,
    ) -> StoreResult<Vec<KnowledgeNode>> {
        <T as MemoryStoreSend>::current_code_context_nodes(self, node_type, tag, scope, limit)
    }
    fn current_code_context_scope_counts(
        &self,
        node_type: &str,
        tag: Option<&str>,
    ) -> StoreResult<Vec<(String, usize)>> {
        <T as MemoryStoreSend>::current_code_context_scope_counts(self, node_type, tag)
    }
    fn data_dir(&self) -> &Path {
        <T as MemoryStoreSend>::data_dir(self)
    }
    fn db_path(&self) -> &Path {
        <T as MemoryStoreSend>::db_path(self)
    }
    fn is_strata(&self) -> bool {
        <T as MemoryStoreSend>::is_strata(self)
    }
    fn decide_memory_pr(&self, id: &str, action: MemoryPrAction) -> StoreResult<MemoryPr> {
        <T as MemoryStoreSend>::decide_memory_pr(self, id, action)
    }
    fn decide_pending_memory_mutation(
        &self,
        id: &str,
        action: MemoryPrAction,
    ) -> StoreResult<Option<PendingMemoryMutationDecision>> {
        <T as MemoryStoreSend>::decide_pending_memory_mutation(self, id, action)
    }
    fn delete_node(&self, id: &str) -> StoreResult<bool> {
        <T as MemoryStoreSend>::delete_node(self, id)
    }
    fn demote_memory(&self, id: &str) -> StoreResult<KnowledgeNode> {
        <T as MemoryStoreSend>::demote_memory(self, id)
    }
    fn demote_memory_as_actor(
        &self,
        id: &str,
        claimed_role: Option<&str>,
        tool: &str,
    ) -> StoreResult<ActorMutationOutcome> {
        <T as MemoryStoreSend>::demote_memory_as_actor(self, id, claimed_role, tool)
    }
    fn due_for_review_node_ids(&self, limit: usize) -> StoreResult<Vec<String>> {
        <T as MemoryStoreSend>::due_for_review_node_ids(self, limit)
    }
    fn expire_stale_reconsolidation_plans(&self) -> StoreResult<Vec<String>> {
        <T as MemoryStoreSend>::expire_stale_reconsolidation_plans(self)
    }
    fn export_portable_archive_to_path(&self, path: &Path) -> StoreResult<PortableArchive> {
        <T as MemoryStoreSend>::export_portable_archive_to_path(self, path)
    }
    fn find_duplicate_by_canonical_hash(
        &self,
        scope: &str,
        content: &str,
    ) -> StoreResult<Option<String>> {
        <T as MemoryStoreSend>::find_duplicate_by_canonical_hash(self, scope, content)
    }
    fn find_intent_record(
        &self,
        scope: &str,
        intent_id: &str,
    ) -> StoreResult<Option<(String, u64, String)>> {
        <T as MemoryStoreSend>::find_intent_record(self, scope, intent_id)
    }
    fn get_active_intentions(&self) -> StoreResult<Vec<IntentionRecord>> {
        <T as MemoryStoreSend>::get_active_intentions(self)
    }
    fn get_active_intentions_in_scope(&self, scope: &str) -> StoreResult<Vec<IntentionRecord>> {
        <T as MemoryStoreSend>::get_active_intentions_in_scope(self, scope)
    }
    fn get_agent_run(&self, run_id: &str) -> StoreResult<Option<AgentRunSummary>> {
        <T as MemoryStoreSend>::get_agent_run(self, run_id)
    }
    fn get_all_connections(&self) -> StoreResult<Vec<ConnectionRecord>> {
        <T as MemoryStoreSend>::get_all_connections(self)
    }
    fn get_all_nodes(&self, limit: i32, offset: i32) -> StoreResult<Vec<KnowledgeNode>> {
        <T as MemoryStoreSend>::get_all_nodes(self, limit, offset)
    }
    fn get_all_nodes_in_scope(
        &self,
        scope: &str,
        limit: i32,
        offset: i32,
    ) -> StoreResult<Vec<KnowledgeNode>> {
        <T as MemoryStoreSend>::get_all_nodes_in_scope(self, scope, limit, offset)
    }
    fn get_avg_retention(&self) -> StoreResult<f64> {
        <T as MemoryStoreSend>::get_avg_retention(self)
    }
    fn get_composition_event(&self, id: &str) -> StoreResult<Option<CompositionEventRecord>> {
        <T as MemoryStoreSend>::get_composition_event(self, id)
    }
    fn get_composition_members(&self, event_id: &str) -> StoreResult<Vec<CompositionMemberRecord>> {
        <T as MemoryStoreSend>::get_composition_members(self, event_id)
    }
    fn get_composition_neighbors(
        &self,
        memory_id: &str,
        limit: i32,
    ) -> StoreResult<Vec<CompositionNeighborRecord>> {
        <T as MemoryStoreSend>::get_composition_neighbors(self, memory_id, limit)
    }
    fn get_composition_outcomes(
        &self,
        event_id: &str,
    ) -> StoreResult<Vec<CompositionOutcomeRecord>> {
        <T as MemoryStoreSend>::get_composition_outcomes(self, event_id)
    }
    fn get_compositions_for_memory(
        &self,
        memory_id: &str,
        limit: i32,
    ) -> StoreResult<Vec<CompositionEventRecord>> {
        <T as MemoryStoreSend>::get_compositions_for_memory(self, memory_id, limit)
    }
    fn get_connections_for_memory(&self, memory_id: &str) -> StoreResult<Vec<ConnectionRecord>> {
        <T as MemoryStoreSend>::get_connections_for_memory(self, memory_id)
    }
    fn get_consolidation_history(
        &self,
        limit: i32,
    ) -> StoreResult<Vec<ConsolidationHistoryRecord>> {
        <T as MemoryStoreSend>::get_consolidation_history(self, limit)
    }
    fn get_context_ablation_replay(
        &self,
        replay_id: &str,
    ) -> StoreResult<Option<StoredCounterfactualReplay>> {
        <T as MemoryStoreSend>::get_context_ablation_replay(self, replay_id)
    }
    fn get_dream_history(&self, limit: i32) -> StoreResult<Vec<DreamHistoryRecord>> {
        <T as MemoryStoreSend>::get_dream_history(self, limit)
    }
    fn get_insights(&self, limit: i32) -> StoreResult<Vec<InsightRecord>> {
        <T as MemoryStoreSend>::get_insights(self, limit)
    }
    fn get_intention(&self, id: &str) -> StoreResult<Option<IntentionRecord>> {
        <T as MemoryStoreSend>::get_intention(self, id)
    }
    fn get_intentions_by_status(&self, status: &str) -> StoreResult<Vec<IntentionRecord>> {
        <T as MemoryStoreSend>::get_intentions_by_status(self, status)
    }
    fn get_last_consolidation(&self) -> StoreResult<Option<DateTime<Utc>>> {
        <T as MemoryStoreSend>::get_last_consolidation(self)
    }
    fn get_last_dream(&self) -> StoreResult<Option<DateTime<Utc>>> {
        <T as MemoryStoreSend>::get_last_dream(self)
    }
    fn get_memory_pr(&self, id: &str) -> StoreResult<Option<MemoryPr>> {
        <T as MemoryStoreSend>::get_memory_pr(self, id)
    }
    fn get_memory_subgraph(
        &self,
        center_id: &str,
        depth: u32,
        max_nodes: usize,
    ) -> StoreResult<(Vec<KnowledgeNode>, Vec<ConnectionRecord>)> {
        <T as MemoryStoreSend>::get_memory_subgraph(self, center_id, depth, max_nodes)
    }
    fn get_merge_operation(&self, operation_id: &str) -> StoreResult<Option<MergeOperation>> {
        <T as MemoryStoreSend>::get_merge_operation(self, operation_id)
    }
    fn get_merge_policy(&self) -> StoreResult<MergePolicy> {
        <T as MemoryStoreSend>::get_merge_policy(self)
    }
    fn get_most_connected_memory(&self) -> StoreResult<Option<String>> {
        <T as MemoryStoreSend>::get_most_connected_memory(self)
    }
    fn get_never_composed_candidates(
        &self,
        limit: i32,
        tag_filter: Option<&[String]>,
    ) -> StoreResult<Vec<NeverComposedCandidate>> {
        <T as MemoryStoreSend>::get_never_composed_candidates(self, limit, tag_filter)
    }
    fn get_never_composed_candidates_in_scope(
        &self,
        limit: i32,
        tag_filter: Option<&[String]>,
        scope: Option<&str>,
    ) -> StoreResult<Vec<NeverComposedCandidate>> {
        <T as MemoryStoreSend>::get_never_composed_candidates_in_scope(
            self, limit, tag_filter, scope,
        )
    }
    fn get_node(&self, id: &str) -> StoreResult<Option<KnowledgeNode>> {
        <T as MemoryStoreSend>::get_node(self, id)
    }
    fn get_overdue_intentions(&self) -> StoreResult<Vec<IntentionRecord>> {
        <T as MemoryStoreSend>::get_overdue_intentions(self)
    }
    fn get_receipt(&self, receipt_id: &str) -> StoreResult<Option<Receipt>> {
        <T as MemoryStoreSend>::get_receipt(self, receipt_id)
    }
    fn get_receipt_attestation_envelope(
        &self,
        receipt_id: &str,
    ) -> StoreResult<Option<DsseEnvelope>> {
        <T as MemoryStoreSend>::get_receipt_attestation_envelope(self, receipt_id)
    }
    fn get_recent_composition_events(
        &self,
        limit: i32,
    ) -> StoreResult<Vec<CompositionEventRecord>> {
        <T as MemoryStoreSend>::get_recent_composition_events(self, limit)
    }
    fn get_recent_composition_events_page(
        &self,
        limit: i32,
        offset: i32,
    ) -> StoreResult<Vec<CompositionEventRecord>> {
        <T as MemoryStoreSend>::get_recent_composition_events_page(self, limit, offset)
    }
    fn get_recent_connections(&self, limit: usize) -> StoreResult<Vec<ConnectionRecord>> {
        <T as MemoryStoreSend>::get_recent_connections(self, limit)
    }
    fn get_recent_state_transitions(&self, limit: i32) -> StoreResult<Vec<StateTransitionRecord>> {
        <T as MemoryStoreSend>::get_recent_state_transitions(self, limit)
    }
    fn get_retention_distribution(&self) -> StoreResult<Vec<(String, i64)>> {
        <T as MemoryStoreSend>::get_retention_distribution(self)
    }
    fn get_retention_trend(&self) -> StoreResult<String> {
        <T as MemoryStoreSend>::get_retention_trend(self)
    }
    fn get_retrieval_replay_capsule(
        &self,
        source_receipt_id: &str,
    ) -> StoreResult<Option<RetrievalReplayCapsuleSummary>> {
        <T as MemoryStoreSend>::get_retrieval_replay_capsule(self, source_receipt_id)
    }
    fn get_review_queue(&self, limit: i32) -> StoreResult<Vec<KnowledgeNode>> {
        <T as MemoryStoreSend>::get_review_queue(self, limit)
    }
    fn get_state_transitions(
        &self,
        memory_id: &str,
        limit: i32,
    ) -> StoreResult<Vec<StateTransitionRecord>> {
        <T as MemoryStoreSend>::get_state_transitions(self, memory_id, limit)
    }
    fn get_stats(&self) -> StoreResult<MemoryStats> {
        <T as MemoryStoreSend>::get_stats(self)
    }
    fn get_trace(&self, run_id: &str) -> StoreResult<Vec<MemoryTraceEvent>> {
        <T as MemoryStoreSend>::get_trace(self, run_id)
    }
    fn get_walk_receipt(&self, receipt_id: &str) -> StoreResult<Option<StoredWalkReceipt>> {
        <T as MemoryStoreSend>::get_walk_receipt(self, receipt_id)
    }
    fn git_commit_nodes(&self, limit: usize) -> StoreResult<Vec<GitCommitNode>> {
        <T as MemoryStoreSend>::git_commit_nodes(self, limit)
    }
    fn grant_actor_role(
        &self,
        actor_did: &str,
        role: &str,
        note: Option<&str>,
    ) -> StoreResult<u64> {
        <T as MemoryStoreSend>::grant_actor_role(self, actor_did, role, note)
    }
    fn hybrid_search(
        &self,
        query: &str,
        limit: i32,
        keyword_weight: f32,
        semantic_weight: f32,
    ) -> StoreResult<Vec<crate::memory::SearchResult>> {
        <T as MemoryStoreSend>::hybrid_search(self, query, limit, keyword_weight, semantic_weight)
    }
    fn hybrid_search_filtered(
        &self,
        query: &str,
        limit: i32,
        keyword_weight: f32,
        semantic_weight: f32,
        include_types: Option<&[String]>,
        exclude_types: Option<&[String]>,
    ) -> StoreResult<Vec<crate::memory::SearchResult>> {
        <T as MemoryStoreSend>::hybrid_search_filtered(
            self,
            query,
            limit,
            keyword_weight,
            semantic_weight,
            include_types,
            exclude_types,
        )
    }
    fn hygiene_snapshot(&self, scope: Option<&str>) -> StoreResult<HygieneSnapshot> {
        <T as MemoryStoreSend>::hygiene_snapshot(self, scope)
    }
    fn import_portable_archive(
        &self,
        archive: &PortableArchive,
        mode: PortableImportMode,
    ) -> StoreResult<PortableImportReport> {
        <T as MemoryStoreSend>::import_portable_archive(self, archive, mode)
    }
    fn import_portable_archive_from_path(
        &self,
        path: &Path,
        mode: PortableImportMode,
    ) -> StoreResult<PortableImportReport> {
        <T as MemoryStoreSend>::import_portable_archive_from_path(self, path, mode)
    }
    fn ingest(&self, input: IngestInput) -> StoreResult<KnowledgeNode> {
        <T as MemoryStoreSend>::ingest(self, input)
    }
    fn ingest_in_scope(&self, input: IngestInput, scope: &str) -> StoreResult<KnowledgeNode> {
        <T as MemoryStoreSend>::ingest_in_scope(self, input, scope)
    }
    fn ingest_in_scope_with_secret_policy(
        &self,
        input: IngestInput,
        scope: &str,
        policy: SecretPolicy,
    ) -> StoreResult<KnowledgeNode> {
        <T as MemoryStoreSend>::ingest_in_scope_with_secret_policy(self, input, scope, policy)
    }
    fn ingest_with_secret_policy(
        &self,
        input: IngestInput,
        policy: SecretPolicy,
    ) -> StoreResult<KnowledgeNode> {
        <T as MemoryStoreSend>::ingest_with_secret_policy(self, input, policy)
    }
    fn is_strata_log(&self) -> bool {
        <T as MemoryStoreSend>::is_strata_log(self)
    }
    fn intention_memory_snapshot(
        &self,
        scope: &str,
        memory_id: &str,
        now: DateTime<Utc>,
    ) -> Result<Value, String> {
        <T as MemoryStoreSend>::intention_memory_snapshot(self, scope, memory_id, now)
    }
    fn last_backup_timestamp(&self) -> Option<DateTime<Utc>> {
        <T as MemoryStoreSend>::last_backup_timestamp(self)
    }
    fn last_session_failed_calls(&self, run_id: Option<&str>) -> StoreResult<Vec<FailedToolCall>> {
        <T as MemoryStoreSend>::last_session_failed_calls(self, run_id)
    }
    fn latest_receipt_chain_entry(&self) -> StoreResult<Option<ChainEntry>> {
        <T as MemoryStoreSend>::latest_receipt_chain_entry(self)
    }
    fn latest_receipt_id_for_node(&self, node_id: &str) -> Option<String> {
        <T as MemoryStoreSend>::latest_receipt_id_for_node(self, node_id)
    }
    fn link_receipt_to_run(&self, receipt_id: &str, run_id: &str) -> StoreResult<bool> {
        <T as MemoryStoreSend>::link_receipt_to_run(self, receipt_id, run_id)
    }
    fn list_agent_runs(&self, limit: usize) -> StoreResult<Vec<AgentRunSummary>> {
        <T as MemoryStoreSend>::list_agent_runs(self, limit)
    }
    fn list_endorsement_events(
        &self,
        memory_id: Option<&str>,
        actor_did: Option<&str>,
        limit: usize,
    ) -> StoreResult<Vec<EndorsementEventRecord>> {
        <T as MemoryStoreSend>::list_endorsement_events(self, memory_id, actor_did, limit)
    }
    fn list_memory_prs(
        &self,
        status: Option<MemoryPrStatus>,
        limit: usize,
    ) -> StoreResult<Vec<MemoryPr>> {
        <T as MemoryStoreSend>::list_memory_prs(self, status, limit)
    }
    fn list_merge_operations(&self, limit: usize) -> StoreResult<Vec<MergeOperation>> {
        <T as MemoryStoreSend>::list_merge_operations(self, limit)
    }
    fn list_receipts(&self, limit: usize) -> StoreResult<Vec<Receipt>> {
        <T as MemoryStoreSend>::list_receipts(self, limit)
    }
    fn list_receipts_for_run(&self, run_id: &str, limit: usize) -> StoreResult<Vec<Receipt>> {
        <T as MemoryStoreSend>::list_receipts_for_run(self, run_id, limit)
    }
    fn list_reconsolidation_plans(&self, limit: usize) -> StoreResult<Vec<(MergePlan, String)>> {
        <T as MemoryStoreSend>::list_reconsolidation_plans(self, limit)
    }
    fn list_tag_operations(
        &self,
        limit: usize,
        scope: Option<&str>,
    ) -> StoreResult<Vec<MergeOperation>> {
        <T as MemoryStoreSend>::list_tag_operations(self, limit, scope)
    }
    fn load_active_synaptic_tags(&self) -> StoreResult<Vec<SynapticTag>> {
        <T as MemoryStoreSend>::load_active_synaptic_tags(self)
    }
    fn lowest_retention_nodes(&self, limit: usize) -> StoreResult<Vec<(String, f64)>> {
        <T as MemoryStoreSend>::lowest_retention_nodes(self, limit)
    }
    fn maintain_gc_batch(
        &self,
        limit: usize,
        after: Option<&str>,
        budget_ms: u64,
        dry_run: bool,
        min_retention: f64,
        max_age_days: Option<u64>,
    ) -> StoreResult<Value> {
        <T as MemoryStoreSend>::maintain_gc_batch(
            self,
            limit,
            after,
            budget_ms,
            dry_run,
            min_retention,
            max_age_days,
        )
    }
    fn maintain_lifecycle_batch(
        &self,
        limit: usize,
        after: Option<&str>,
        budget_ms: u64,
        dry_run: bool,
    ) -> StoreResult<Value> {
        <T as MemoryStoreSend>::maintain_lifecycle_batch(self, limit, after, budget_ms, dry_run)
    }
    fn maintain_log_batch(&self, limit: usize, dry_run: bool) -> StoreResult<Value> {
        <T as MemoryStoreSend>::maintain_log_batch(self, limit, dry_run)
    }
    fn maintenance_memory_page(
        &self,
        limit: usize,
        after: Option<&str>,
        scope: &str,
    ) -> StoreResult<(Vec<crate::KnowledgeNode>, bool)> {
        <T as MemoryStoreSend>::maintenance_memory_page(self, limit, after, scope)
    }
    fn mark_reviewed(&self, id: &str, rating: Rating) -> StoreResult<KnowledgeNode> {
        <T as MemoryStoreSend>::mark_reviewed(self, id, rating)
    }
    fn merge_candidates(
        &self,
        policy: MergePolicy,
        limit: usize,
        tag_filter: &[String],
    ) -> StoreResult<Vec<MergeCandidate>> {
        <T as MemoryStoreSend>::merge_candidates(self, policy, limit, tag_filter)
    }
    fn merge_undo(&self, op_id: &str) -> StoreResult<MergeOperation> {
        <T as MemoryStoreSend>::merge_undo(self, op_id)
    }
    fn node_effect_seq(&self, node_id: &str) -> Option<u64> {
        <T as MemoryStoreSend>::node_effect_seq(self, node_id)
    }
    fn node_is_in_scope(&self, id: &str, scope: &str) -> StoreResult<bool> {
        <T as MemoryStoreSend>::node_is_in_scope(self, id, scope)
    }
    fn open_failures_touching(
        &self,
        changed_files: &[String],
    ) -> StoreResult<Vec<OpenFailureTouching>> {
        <T as MemoryStoreSend>::open_failures_touching(self, changed_files)
    }
    fn preview_tag_mutation(
        &self,
        source_tags: &[String],
        target_tag: &str,
        scope: Option<&str>,
    ) -> StoreResult<serde_json::Value> {
        <T as MemoryStoreSend>::preview_tag_mutation(self, source_tags, target_tag, scope)
    }
    fn process_actor_did(&self) -> Option<String> {
        <T as MemoryStoreSend>::process_actor_did(self)
    }
    fn process_synaptic_ingest(
        &self,
        request: &SynapticIngestRequest,
    ) -> StoreResult<SynapticIngestOutcome> {
        <T as MemoryStoreSend>::process_synaptic_ingest(self, request)
    }
    fn projection_candidates(
        &self,
        scope: &str,
        min_retention: f64,
        limit: i32,
    ) -> StoreResult<Vec<KnowledgeNode>> {
        <T as MemoryStoreSend>::projection_candidates(self, scope, min_retention, limit)
    }
    fn promote_memory(&self, id: &str) -> StoreResult<KnowledgeNode> {
        <T as MemoryStoreSend>::promote_memory(self, id)
    }
    fn promote_memory_as_actor(
        &self,
        id: &str,
        claimed_role: Option<&str>,
        tool: &str,
    ) -> StoreResult<ActorMutationOutcome> {
        <T as MemoryStoreSend>::promote_memory_as_actor(self, id, claimed_role, tool)
    }
    fn promote_memory_backfill(&self, id: &str) -> StoreResult<KnowledgeNode> {
        <T as MemoryStoreSend>::promote_memory_backfill(self, id)
    }
    fn prune_agent_traces(&self) -> StoreResult<i64> {
        <T as MemoryStoreSend>::prune_agent_traces(self)
    }
    fn purge_node(&self, id: &str, reason: Option<&str>) -> StoreResult<PurgeReport> {
        <T as MemoryStoreSend>::purge_node(self, id, reason)
    }
    fn query_time_range(
        &self,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
        limit: i32,
        node_type: Option<&str>,
        tags: Option<&[String]>,
    ) -> StoreResult<Vec<KnowledgeNode>> {
        <T as MemoryStoreSend>::query_time_range(self, start, end, limit, node_type, tags)
    }
    fn recall(&self, input: RecallInput) -> StoreResult<Vec<KnowledgeNode>> {
        <T as MemoryStoreSend>::recall(self, input)
    }
    fn receipt_attestation_status(
        &self,
        receipt_id: &str,
    ) -> StoreResult<Option<ReceiptAttestationStatus>> {
        <T as MemoryStoreSend>::receipt_attestation_status(self, receipt_id)
    }
    fn record_anchor_verification(
        &self,
        anchor_id: &str,
        status: AnchorStatus,
        checked_at: DateTime<Utc>,
    ) -> StoreResult<()> {
        <T as MemoryStoreSend>::record_anchor_verification(self, anchor_id, status, checked_at)
    }
    fn record_batch_retrieval(&self, ids: &[&str]) -> StoreResult<()> {
        <T as MemoryStoreSend>::record_batch_retrieval(self, ids)
    }
    fn record_code_anchors(&self, anchors: &[CodeAnchor]) -> StoreResult<usize> {
        <T as MemoryStoreSend>::record_code_anchors(self, anchors)
    }
    fn record_composition_outcome(&self, outcome: &CompositionOutcomeRecord) -> StoreResult<()> {
        <T as MemoryStoreSend>::record_composition_outcome(self, outcome)
    }
    fn record_memory_access(&self, memory_id: &str) -> StoreResult<()> {
        <T as MemoryStoreSend>::record_memory_access(self, memory_id)
    }
    fn record_intent_entry(
        &self,
        scope: &str,
        intent_id: &str,
        node_id: &str,
        effect_seq: u64,
        response_digest: &str,
    ) -> StoreResult<()> {
        <T as MemoryStoreSend>::record_intent_entry(
            self,
            scope,
            intent_id,
            node_id,
            effect_seq,
            response_digest,
        )
    }
    fn record_reinforce_endorsement(
        &self,
        id: &str,
        claimed_role: Option<&str>,
        tool: &str,
    ) -> StoreResult<ActorMutationOutcome> {
        <T as MemoryStoreSend>::record_reinforce_endorsement(self, id, claimed_role, tool)
    }
    fn registered_receipt_signing_key(
        &self,
        key_id: &str,
    ) -> StoreResult<Option<TrustedSigningKey>> {
        <T as MemoryStoreSend>::registered_receipt_signing_key(self, key_id)
    }
    fn release_quarantine(&self, id: &str) -> StoreResult<KnowledgeNode> {
        <T as MemoryStoreSend>::release_quarantine(self, id)
    }
    fn replace_code_anchors(
        &self,
        node_id: &str,
        scope: &str,
        anchors: &[CodeAnchor],
    ) -> StoreResult<usize> {
        <T as MemoryStoreSend>::replace_code_anchors(self, node_id, scope, anchors)
    }
    fn replay_intention_graph(&self, scope: &str) -> Result<Value, String> {
        <T as MemoryStoreSend>::replay_intention_graph(self, scope)
    }
    fn resolve_actor_role(
        &self,
        claimed_role: Option<&str>,
    ) -> StoreResult<(String, RoleResolution)> {
        <T as MemoryStoreSend>::resolve_actor_role(self, claimed_role)
    }
    fn resolve_commit_sha_root(&self, sha_prefix: &str) -> StoreResult<Option<String>> {
        <T as MemoryStoreSend>::resolve_commit_sha_root(self, sha_prefix)
    }
    fn resolve_handle(&self, query: &str) -> HandleResolution {
        <T as MemoryStoreSend>::resolve_handle(self, query)
    }
    fn retire_affected(&self, ids: &[&str], reason: &str) -> Vec<RetireOutcome> {
        <T as MemoryStoreSend>::retire_affected(self, ids, reason)
    }
    fn reverse_suppression(&self, id: &str, labile_hours: i64) -> StoreResult<KnowledgeNode> {
        <T as MemoryStoreSend>::reverse_suppression(self, id, labile_hours)
    }
    fn run_consolidation(&self) -> StoreResult<ConsolidationResult> {
        <T as MemoryStoreSend>::run_consolidation(self)
    }
    fn save_composition(
        &self,
        event: &CompositionEventRecord,
        members: &[CompositionMemberRecord],
        outcomes: &[CompositionOutcomeRecord],
    ) -> StoreResult<()> {
        <T as MemoryStoreSend>::save_composition(self, event, members, outcomes)
    }
    fn save_connection(&self, connection: &ConnectionRecord) -> StoreResult<()> {
        <T as MemoryStoreSend>::save_connection(self, connection)
    }
    fn save_counterfactual_replay_receipt(
        &self,
        replay_id: &str,
        receipt: &Receipt,
        run_id: Option<&str>,
        tool: Option<&str>,
    ) -> StoreResult<()> {
        <T as MemoryStoreSend>::save_counterfactual_replay_receipt(
            self, replay_id, receipt, run_id, tool,
        )
    }
    fn save_dream_history(&self, record: &DreamHistoryRecord) -> StoreResult<i64> {
        <T as MemoryStoreSend>::save_dream_history(self, record)
    }
    fn save_insight(&self, insight: &InsightRecord) -> StoreResult<()> {
        <T as MemoryStoreSend>::save_insight(self, insight)
    }
    fn save_intention(&self, intention: &IntentionRecord) -> StoreResult<()> {
        <T as MemoryStoreSend>::save_intention(self, intention)
    }
    fn save_memory_pr(&self, pr: &MemoryPr) -> StoreResult<()> {
        <T as MemoryStoreSend>::save_memory_pr(self, pr)
    }
    fn save_receipt(
        &self,
        receipt: &Receipt,
        run_id: Option<&str>,
        tool: Option<&str>,
        query: Option<&str>,
    ) -> StoreResult<()> {
        <T as MemoryStoreSend>::save_receipt(self, receipt, run_id, tool, query)
    }
    fn save_retrieval_receipt_with_replay_capsule(
        &self,
        receipt: &Receipt,
        run_id: Option<&str>,
        tool: Option<&str>,
        draft: &RetrievalReplayCapsuleDraft,
    ) -> StoreResult<DurableRetrievalReplayCapsule> {
        <T as MemoryStoreSend>::save_retrieval_receipt_with_replay_capsule(
            self, receipt, run_id, tool, draft,
        )
    }
    fn save_signed_retrieval_receipt_with_replay_capsule_atomic(
        &self,
        write: SignedReceiptWrite<'_>,
        draft: &RetrievalReplayCapsuleDraft,
    ) -> StoreResult<DurableSignedRetrievalReceipt> {
        <T as MemoryStoreSend>::save_signed_retrieval_receipt_with_replay_capsule_atomic(
            self, write, draft,
        )
    }
    fn save_synaptic_tag(&self, tag: &SynapticTag) -> StoreResult<String> {
        <T as MemoryStoreSend>::save_synaptic_tag(self, tag)
    }
    fn save_walk_receipt(
        &self,
        canonical_json: &str,
        params: &Value,
    ) -> StoreResult<WalkReceiptHandle> {
        <T as MemoryStoreSend>::save_walk_receipt(self, canonical_json, params)
    }
    fn schema_introspection(&self) -> StoreResult<SchemaIntrospection> {
        <T as MemoryStoreSend>::schema_introspection(self)
    }
    fn search(&self, query: &str, limit: i32) -> StoreResult<Vec<KnowledgeNode>> {
        <T as MemoryStoreSend>::search(self, query, limit)
    }
    fn set_created_at(&self, id: &str, when: DateTime<Utc>) -> StoreResult<()> {
        <T as MemoryStoreSend>::set_created_at(self, id, when)
    }
    fn set_merge_policy(&self, policy: crate::advanced::MergePolicy) -> StoreResult<()> {
        <T as MemoryStoreSend>::set_merge_policy(self, policy)
    }
    fn set_process_actor(&self, did: &str) -> StoreResult<()> {
        <T as MemoryStoreSend>::set_process_actor(self, did)
    }
    fn set_protected(&self, id: &str, protected: bool) -> StoreResult<()> {
        <T as MemoryStoreSend>::set_protected(self, id, protected)
    }
    fn sidecar_dir(&self, name: &str) -> PathBuf {
        <T as MemoryStoreSend>::sidecar_dir(self, name)
    }
    fn register_receipt_signing_key(&self, key: &TrustedSigningKey) -> StoreResult<bool> {
        <T as MemoryStoreSend>::register_receipt_signing_key(self, key)
    }
    fn export_portable_archive(&self) -> StoreResult<PortableArchive> {
        <T as MemoryStoreSend>::export_portable_archive(self)
    }
    fn reconcile_source_tombstones(
        &self,
        source_system: &str,
        scope: &str,
        live_ids: &[String],
    ) -> StoreResult<ReconcileReport> {
        <T as MemoryStoreSend>::reconcile_source_tombstones(self, source_system, scope, live_ids)
    }
    fn get_connector_cursor(
        &self,
        source_system: &str,
        scope: &str,
    ) -> StoreResult<ConnectorCursor> {
        <T as MemoryStoreSend>::get_connector_cursor(self, source_system, scope)
    }
    fn save_connector_cursor(&self, cursor: &ConnectorCursor) -> StoreResult<()> {
        <T as MemoryStoreSend>::save_connector_cursor(self, cursor)
    }
    fn dream_compile_candidates(
        &self,
        scope: &str,
        limit: usize,
    ) -> StoreResult<Vec<KnowledgeNode>> {
        <T as MemoryStoreSend>::dream_compile_candidates(self, scope, limit)
    }
    fn strengthen_connection(
        &self,
        source_id: &str,
        target_id: &str,
        boost: f64,
    ) -> StoreResult<bool> {
        <T as MemoryStoreSend>::strengthen_connection(self, source_id, target_id, boost)
    }
    fn run_rac1_cascade_sweep(&self) -> StoreResult<(usize, usize)> {
        <T as MemoryStoreSend>::run_rac1_cascade_sweep(self)
    }
    fn smart_ingest_excluding_in_scope_with_secret_policy_and_labile(
        &self,
        input: IngestInput,
        scope: &str,
        excluded_node_ids: &[String],
        policy: SecretPolicy,
        labile: &[LabileCandidate],
    ) -> StoreResult<SmartIngestResult> {
        <T as MemoryStoreSend>::smart_ingest_excluding_in_scope_with_secret_policy_and_labile(
            self,
            input,
            scope,
            excluded_node_ids,
            policy,
            labile,
        )
    }
    fn snooze_intention(&self, id: &str, until: DateTime<Utc>) -> StoreResult<bool> {
        <T as MemoryStoreSend>::snooze_intention(self, id, until)
    }
    fn state_distribution(&self) -> StoreResult<(i64, i64, i64, i64)> {
        <T as MemoryStoreSend>::state_distribution(self)
    }
    fn superseded_node_ids(&self) -> StoreResult<HashSet<String>> {
        <T as MemoryStoreSend>::superseded_node_ids(self)
    }
    fn supersession_pairs(&self) -> StoreResult<Vec<(String, String)>> {
        <T as MemoryStoreSend>::supersession_pairs(self)
    }
    fn suppress_memory(&self, id: &str) -> StoreResult<KnowledgeNode> {
        <T as MemoryStoreSend>::suppress_memory(self, id)
    }
    fn sync_portable_archive_cloud(
        &self,
        endpoint: &str,
        sync_key: &str,
        encryption_key: Option<String>,
    ) -> StoreResult<PortableSyncReport> {
        <T as MemoryStoreSend>::sync_portable_archive_cloud(
            self,
            endpoint,
            sync_key,
            encryption_key,
        )
    }
    fn sync_portable_archive_file(&self, path: &Path) -> StoreResult<PortableSyncReport> {
        <T as MemoryStoreSend>::sync_portable_archive_file(self, path)
    }
    fn tag_vocabulary(&self, scope: Option<&str>) -> StoreResult<TagVocabulary> {
        <T as MemoryStoreSend>::tag_vocabulary(self, scope)
    }
    fn undo_tag_mutation(&self, operation_id: &str) -> StoreResult<MergeOperation> {
        <T as MemoryStoreSend>::undo_tag_mutation(self, operation_id)
    }
    fn update_intention_status(&self, id: &str, status: &str) -> StoreResult<bool> {
        <T as MemoryStoreSend>::update_intention_status(self, id, status)
    }
    fn update_memory_state(
        &self,
        memory_id: &str,
        new_state: &str,
        reason: &str,
    ) -> StoreResult<bool> {
        <T as MemoryStoreSend>::update_memory_state(self, memory_id, new_state, reason)
    }
    fn update_node_content(&self, id: &str, new_content: &str) -> StoreResult<()> {
        <T as MemoryStoreSend>::update_node_content(self, id, new_content)
    }
    fn upsert_by_source(&self, input: IngestInput) -> StoreResult<SourceUpsertResult> {
        <T as MemoryStoreSend>::upsert_by_source(self, input)
    }
    fn verify_stored_receipt_attestation(
        &self,
        receipt_id: &str,
    ) -> StoreResult<Option<StoredReceiptAttestationVerification>> {
        <T as MemoryStoreSend>::verify_stored_receipt_attestation(self, receipt_id)
    }
}

// ----------------------------------------------------------------------------
// UNIT TESTS
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "legacy-sqlite")]
    use crate::storage::StorageError;

    #[cfg(feature = "legacy-sqlite")]
    #[test]
    fn memory_store_error_from_storage_error() {
        let se = StorageError::NotFound("abc".to_string());
        let mse = MemoryStoreError::from(se);
        assert!(matches!(mse, MemoryStoreError::NotFound(_)));

        let se2 = StorageError::Init("init failure".to_string());
        let mse2 = MemoryStoreError::from(se2);
        assert!(matches!(mse2, MemoryStoreError::Init(_)));
    }

    #[test]
    fn model_signature_serde_round_trip() {
        let sig = ModelSignature {
            name: "nomic-ai/nomic-embed-text-v1.5".to_string(),
            dimension: 256,
            hash: "a".repeat(64),
        };
        let json = serde_json::to_string(&sig).expect("serialize");
        let sig2: ModelSignature = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(sig, sig2);
    }

    #[test]
    fn memory_record_serde_round_trip() {
        let rec = MemoryRecord {
            id: Uuid::new_v4(),
            domains: vec!["dev".to_string()],
            domain_scores: {
                let mut m = HashMap::new();
                m.insert("dev".to_string(), 0.9);
                m
            },
            content: "hello".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["tag1".to_string()],
            embedding: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: serde_json::json!({}),
        };
        let json = serde_json::to_string(&rec).expect("serialize");
        let rec2: MemoryRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rec.content, rec2.content);
        assert_eq!(rec.domains, rec2.domains);
    }
}
