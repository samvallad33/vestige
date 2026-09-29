//! Storage Module
//!
//! Backend-agnostic memory store abstraction plus SQLite reference impl.
//!
//! The SQLite reference implementation (and every module that talks
//! rusqlite) is quarantined behind the `legacy-sqlite` feature
//! (build/t5-legacy-isolation). Default ON; the flip to default-off lands
//! with the STRATA backend. Builders must open stores through
//! [`open_storage`] so a `legacy-sqlite`-free build fails with the clear
//! [`LegacySqliteDisabled`] error instead of a compile error.

#[cfg(feature = "legacy-sqlite")]
mod attestation_store;
#[cfg(feature = "cloud-sync")]
mod cloud_crypto;
#[cfg(all(feature = "cloud-sync", feature = "legacy-sqlite"))]
mod cloud_sync;
#[cfg(feature = "legacy-sqlite")]
mod edges;
#[cfg(feature = "legacy-sqlite")]
mod blast;
#[cfg(feature = "legacy-sqlite")]
mod intention_claim;
#[cfg(feature = "legacy-sqlite")]
mod intention_graph_store;
mod memory_store;
#[cfg(feature = "legacy-sqlite")]
mod maintenance_batches;
#[cfg(feature = "legacy-sqlite")]
mod migrations;
#[cfg(feature = "legacy-sqlite")]
mod portable;
pub mod receipt_attestation;
#[cfg(feature = "legacy-sqlite")]
mod replay_store;
#[cfg(feature = "legacy-sqlite")]
mod resolver;
#[cfg(feature = "legacy-sqlite")]
mod session_queries;
#[cfg(feature = "legacy-sqlite")]
mod sqlite;
#[cfg(feature = "legacy-sqlite")]
mod synaptic_store;
#[cfg(feature = "legacy-sqlite")]
mod trace_store;
#[cfg(feature = "legacy-sqlite")]
mod walk_receipts;
pub mod unlearning;
#[cfg(feature = "legacy-sqlite")]
mod unlearning_store;

#[cfg(all(feature = "cloud-sync", feature = "legacy-sqlite"))]
pub use cloud_sync::HttpPortableSyncBackend;

#[cfg(feature = "legacy-sqlite")]
pub use attestation_store::{
    DurableSignedReceipt, DurableSignedRetrievalReceipt, ProvisionedReceiptSigningKey,
    ReceiptAttestationStatus, ReceiptSigningKeyTransition, SignedReceiptWrite,
    StoredReceiptAttestationVerification, load_receipt_signing_seed,
    provision_receipt_signing_key_sidecar,
};
#[cfg(feature = "legacy-sqlite")]
pub use blast::{
    BLAST_LINK_TYPES, BLAST_MAX_DEPTH, BLAST_SCAN_NODE_CAP, BlastAffected, BlastReport,
    RetireOutcome, commit_sha_of,
};
pub use memory_store::{
    ClassificationResult, Domain, HealthStatus, LocalMemoryStore, MemoryEdge, MemoryRecord,
    MemoryStore, MemoryStoreError, MemoryStoreResult, MemoryStoreSend, ModelSignature,
    SchedulingState, SearchQuery, SearchResult, StoreStats,
};
#[cfg(feature = "legacy-sqlite")]
pub use edges::{
    EdgeDirection, EdgeKind, EdgeMeta, PurgeTombstone, TYPED_EDGE_VOCABULARY, TypedEdge,
};
#[cfg(feature = "legacy-sqlite")]
pub use migrations::MIGRATIONS;
#[cfg(feature = "legacy-sqlite")]
pub use portable::{
    PORTABLE_ARCHIVE_FORMAT, PortableArchive, PortableImportMode, PortableImportReport,
    PortableTable, PortableValue,
};
#[cfg(feature = "legacy-sqlite")]
pub use resolver::{HANDLE_REQUIRED_DETAIL, HandleKind, HandleResolution, MAX_CANDIDATES};
#[cfg(feature = "legacy-sqlite")]
pub use replay_store::{
    CounterfactualReplayResult, DurableCounterfactualReplay, DurableRetrievalReplayCapsule,
    FrozenReplayItem, REPLAY_ALGORITHM_VERSION, REPLAY_CLAIM_BOUNDARY, REPLAY_SCHEMA_VERSION,
    REPLAY_SELECTION_BOUNDARY, ReplayBuildError, ReplayDecayRisk, ReplayEvidenceItemSummary,
    ReplayEvidenceSetSummary, ReplayInfluence, ReplayInvalidationReason,
    ReplayMaterializationCheck, ReplayPrivacyInvalidation, ReplayPrivacyState,
    RetrievalReplayCapsuleDraft, RetrievalReplayCapsuleSummary, RetrievalReplayItemDraft,
    StoredCounterfactualReplay, ablate_frozen_context, private_evidence_digest,
    replay_evidence_slot, replay_idempotency_key, replay_policy_digest,
};
#[cfg(feature = "legacy-sqlite")]
pub use sqlite::{WalCheckpointMode, WalCheckpointStatus};
#[cfg(feature = "legacy-sqlite")]
pub use sqlite::{
    ACCESS_LOG_RETENTION_DAYS, ActorMutationOutcome, CompositionEventRecord,
    CompositionMemberRecord, CompositionNeighborRecord, CompositionOutcomeRecord,
    ConnectionRecord, ConnectorCursor, ConsolidationHistoryRecord, DEFAULT_MEMORY_SCOPE,
    DreamHistoryRecord, EmbeddingProfileIntegrityManifest,
    EmbeddingProfileMigrationNodeCheckpoint, EmbeddingProfileMigrationRecord,
    EmbeddingProfileVector, EndorsementEventRecord, FilePortableSyncBackend, HygieneNodeSummary,
    HygieneSnapshot, InsightRecord, IntentionRecord, NeverComposedCandidate,
    PortableSyncBackend, PortableSyncReport, PurgeReport, ReconcileReport, Result,
    SmartIngestResult,
    SourceUpsertOutcome, SourceUpsertResult, SqliteMemoryStore, StateTransitionRecord,
    StorageError, TagVocabulary,
};
#[cfg(feature = "legacy-sqlite")]
pub use synaptic_store::{
    DurableSynapticCapture, DurableSynapticPairReceipt, SYNAPTIC_CAPTURE_ALGORITHM_V1,
    SYNAPTIC_CAPTURE_ALGORITHM_V2, SYNAPTIC_CAPTURE_CLAIM_BOUNDARY, SYNAPTIC_CAPTURE_SCHEMA_V1,
    SYNAPTIC_CAPTURE_SCHEMA_V2, SYNAPTIC_CONTEXT_ALGORITHM_V1, SYNAPTIC_CONTEXT_THRESHOLD_V1,
    SynapticCapturePolicy, SynapticCaptureRequest, SynapticImportanceEvent, SynapticIngestOutcome,
    SynapticIngestRequest, SynapticSignalSnapshot,
};
#[cfg(feature = "legacy-sqlite")]
pub use session_queries::{
    ClosedIssueNode, FailedToolCall, GitCommitNode, OpenFailureTouching, FAILED_CALLS_MAX,
};
#[cfg(feature = "legacy-sqlite")]
pub use trace_store::{
    AgentRunSummary, PendingMemoryMutationDecision, PendingMemoryMutationEffect,
};
#[cfg(feature = "legacy-sqlite")]
pub use walk_receipts::{
    CoverageSnapshot, StoredWalkReceipt, WalkReceiptHandle, WALK_RECEIPT_SCHEMA_V1,
    canonical_walk_json,
};
pub use unlearning::{
    AntiResurrectionCommitments, ArtifactKind, ArtifactRef, CheckStatus, Commitment, CommitmentKey,
    CommitmentKind, ErasureLedgerRecord, GuaranteeExclusion, LineageClosure, LineageEdge,
    LineageRelation, PostconditionCheck, PostconditionKind, PostconditionReport, SurfaceAction,
    SurfaceDetailCode, SurfaceResult, UnlearningScope, UnlearningVerdict,
    VERIFIED_LOCAL_UNLEARNING_SCHEMA_V1, VerificationFence, anti_resurrection_commitments,
    commit_lineage_closure, compute_lineage_closure, evaluate_unlearning_verdict,
};
#[cfg(feature = "legacy-sqlite")]
pub use unlearning_store::{
    AntiResurrectionGateStatus, CanaryScanResult, EligibleAuditRecord, ErasureFailureCode,
    ErasureJobStart, ErasureJobStatus, ExactCanary, LocalCanaryTable, StoredErasureJob,
    TombstoneWriteOutcome, UnlearningStore, UnlearningStoreError, UnlearningStoreResult,
    V25_REQUIRED_LOCAL_CANARY_TABLES, V25_UNLEARNING_STORAGE_SCHEMA_EXPECTATION,
    V25_UNLEARNING_STORAGE_SCHEMA_VERSION,
};

/// Backwards-compatibility alias. Retained until Phase 4 completes so every
/// existing `Arc<Storage>` call site keeps compiling. Scheduled for removal
/// once no downstream source file references it.
#[cfg(feature = "legacy-sqlite")]

/// Error returned by [`open_storage`] when the binary was built without the
/// `legacy-sqlite` feature. Exists in every build so callers can name it
/// (and print it) regardless of feature state.
#[derive(Debug, thiserror::Error)]
#[error("built without legacy-sqlite; STRATA default lands in the next merge")]
pub struct LegacySqliteDisabled;

/// Open the process-local store.
///
/// The single sanctioned constructor path (build/t5-legacy-isolation).
/// With `legacy-sqlite` on (current default) this is
/// `SqliteMemoryStore::new`. Without it, the call fails with
/// [`LegacySqliteDisabled`] ("built without legacy-sqlite; STRATA default
/// lands in the next merge") — the STRATA backend, not SQLite, becomes the
/// default in the next merge.
/// Phase 4 storage wall: the product-wide seam.
///
/// Every tool, server, CLI entry point, and cognitive module holds the store
/// through this trait object (`Arc<Storage>`, `&Storage`). `SqliteMemoryStore`
/// is one implementation of it, constructed only via [`open_storage`] (and
/// direct backend construction inside `vestige-core`'s own tests). A second
/// engine implements `MemoryStoreSend` and drops in behind the same alias.
pub type Storage = dyn MemoryStore;

/// Default database artifact path for the SQLite backend.
pub fn default_db_path() -> Result<std::path::PathBuf> {
    SqliteMemoryStore::default_db_path()
}

/// Database artifact path for a given data directory (SQLite backend).
pub fn db_path_for_data_dir(data_dir: std::path::PathBuf) -> Result<std::path::PathBuf> {
    SqliteMemoryStore::db_path_for_data_dir(data_dir)
}

/// Construct the default local backend (SQLite reference implementation)
/// behind the Phase 4 storage trait. This is the only constructor the MCP
/// layer is allowed to call. With `legacy-sqlite` compiled out (the STRATA
/// default), it fails with [`LegacySqliteDisabled`].
#[cfg(feature = "legacy-sqlite")]
pub fn open_storage(path: Option<std::path::PathBuf>) -> Result<std::sync::Arc<dyn MemoryStore>> {
    Ok(std::sync::Arc::new(SqliteMemoryStore::new(path)?))
}

/// Feature-off twin of [`open_storage`]: consults the injectable
/// [`STORE_CONSTRUCTOR`] hook (installed at startup by binary front-ends,
/// e.g. vestige-mcp installing the STRATA backend via strata-bridge) so a
/// `legacy-sqlite`-free build boots on whatever engine the host chose.
/// Without an installed constructor it fails with
/// [`OpenStoreError::NoBackend`] (the old `LegacySqliteDisabled` state).
#[cfg(not(feature = "legacy-sqlite"))]
pub fn open_storage(
    path: Option<std::path::PathBuf>,
) -> std::result::Result<std::sync::Arc<dyn MemoryStore>, OpenStoreError> {
    // fn pointers are Copy: read without consuming so repeated opens in one
    // process (CLI subcommands, tests) keep working.
    let installed = STORE_CONSTRUCTOR
        .lock()
        .map_err(|_| OpenStoreError::NoBackend)?
        .clone();
    match installed {
        Some(constructor) => {
            // The constructor owns path interpretation (e.g. a DB-file path
            // resolved to its `<data dir>/strata` directory).
            let path = path.unwrap_or_else(|| {
                std::env::var_os("VESTIGE_DATA_DIR")
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
            });
            constructor(path).map_err(OpenStoreError::Constructor)
        }
        None => Err(OpenStoreError::NoBackend),
    }
}

/// Error returned by the feature-off [`open_storage`].
#[cfg(not(feature = "legacy-sqlite"))]
#[derive(Debug, thiserror::Error)]
pub enum OpenStoreError {
    /// No backend constructor was installed; this build has no engine.
    #[error("built without legacy-sqlite and no store constructor installed; install one via storage::install_store_constructor (the STRATA backend lives in strata-bridge)")]
    NoBackend,
    /// The installed constructor ran and failed.
    #[error("store constructor failed: {0}")]
    Constructor(String),
}

/// Injectable backend constructor for `legacy-sqlite`-free builds
/// (build/wire-strata). vestige-core must not depend on strata crates, so
/// the binary front-end (vestige-mcp) installs the real engine at startup:
/// `install_store_constructor(strata_bridge::strata_constructor)`. The
/// function receives the path passed to [`open_storage`] (a DB-file-shaped
/// path or data dir) and owns interpreting it for its engine.
pub type StoreConstructor =
    fn(std::path::PathBuf) -> std::result::Result<std::sync::Arc<dyn MemoryStore>, String>;

/// The installed backend constructor, if any. Read through [`open_storage`]
/// (feature-off builds) or [`install_store_constructor`] to set it.
pub static STORE_CONSTRUCTOR: std::sync::Mutex<Option<StoreConstructor>> =
    std::sync::Mutex::new(None);

/// Install the backend constructor used by feature-off [`open_storage`].
/// Installing over an existing constructor replaces it (tests rely on this).
pub fn install_store_constructor(constructor: StoreConstructor) {
    if let Ok(mut slot) = STORE_CONSTRUCTOR.lock() {
        *slot = Some(constructor);
    }
}
