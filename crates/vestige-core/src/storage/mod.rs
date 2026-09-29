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
//!
//! Type definitions shared by the trait surface and every backend live in
//! [`types`] UNGATED, so the workspace compiles with AND without
//! `legacy-sqlite` (dual-mode rule, strata/fix-00a).

#[cfg(feature = "legacy-sqlite")]
mod attestation_store;
#[cfg(feature = "legacy-sqlite")]
mod blast;
// cloud_crypto's only consumer is cloud_sync, which needs the legacy
// engine; ungating it made the whole module dead code in the
// no-embeddings profile.
#[cfg(all(feature = "cloud-sync", feature = "legacy-sqlite"))]
mod cloud_crypto;
#[cfg(all(feature = "cloud-sync", feature = "legacy-sqlite"))]
mod cloud_sync;
#[cfg(feature = "legacy-sqlite")]
mod edges;
#[cfg(feature = "legacy-sqlite")]
mod intention_claim;
#[cfg(feature = "legacy-sqlite")]
mod intention_graph_store;
#[cfg(feature = "legacy-sqlite")]
mod maintenance_batches;
mod memory_store;
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
/// Feature-independent type definitions shared by every backend and the
/// `LocalMemoryStore` trait surface. Available with AND without
/// `legacy-sqlite` — dual-mode compilation (strata/fix-00a).
pub mod types;
pub mod unlearning;
#[cfg(feature = "legacy-sqlite")]
mod unlearning_store;
/// Feature-independent v3 SQLite guard: detects the legacy engine by magic
/// bytes and refuses read-write opens (PR 0a). Available with AND without
/// `legacy-sqlite`.
pub mod v3_guard;
#[cfg(feature = "legacy-sqlite")]
mod walk_receipts;

// Every type the backend-agnostic `LocalMemoryStore` trait names is defined
// in `types` and re-exported here UNGATED so the trait (and its callers)
// compile with or without `legacy-sqlite`.
pub use types::*;

#[cfg(all(feature = "cloud-sync", feature = "legacy-sqlite"))]
pub use cloud_sync::HttpPortableSyncBackend;

#[cfg(feature = "legacy-sqlite")]
pub use attestation_store::{
    ProvisionedReceiptSigningKey, ReceiptSigningKeyTransition, load_receipt_signing_seed,
    provision_receipt_signing_key_sidecar,
};
#[cfg(feature = "legacy-sqlite")]
pub use blast::{BLAST_LINK_TYPES, BLAST_MAX_DEPTH, BLAST_SCAN_NODE_CAP, commit_sha_of};
#[cfg(feature = "legacy-sqlite")]
pub use edges::{
    EdgeDirection, EdgeKind, EdgeMeta, PurgeTombstone, TYPED_EDGE_VOCABULARY, TypedEdge,
};
pub use memory_store::{
    ClassificationResult, Domain, HealthStatus, LocalMemoryStore, MemoryEdge, MemoryRecord,
    MemoryStore, MemoryStoreError, MemoryStoreResult, MemoryStoreSend, ModelSignature,
    SchedulingState, SearchQuery, SearchResult, StoreStats,
};
#[cfg(feature = "legacy-sqlite")]
pub use migrations::MIGRATIONS;
#[cfg(feature = "legacy-sqlite")]
pub use portable::PORTABLE_ARCHIVE_FORMAT;
#[cfg(feature = "legacy-sqlite")]
pub use replay_store::{
    FrozenReplayItem, REPLAY_ALGORITHM_VERSION, REPLAY_CLAIM_BOUNDARY, REPLAY_SCHEMA_VERSION,
    REPLAY_SELECTION_BOUNDARY, ReplayBuildError, ReplayInvalidationReason,
    ReplayMaterializationCheck, ReplayPrivacyInvalidation, ablate_frozen_context,
    private_evidence_digest, replay_evidence_slot, replay_idempotency_key, replay_policy_digest,
};
#[cfg(feature = "legacy-sqlite")]
pub use session_queries::FAILED_CALLS_MAX;
#[cfg(feature = "legacy-sqlite")]
pub use sqlite::{
    ACCESS_LOG_RETENTION_DAYS, DEFAULT_MEMORY_SCOPE, EmbeddingProfileIntegrityManifest,
    EmbeddingProfileMigrationNodeCheckpoint, EmbeddingProfileMigrationRecord,
    EmbeddingProfileVector, FilePortableSyncBackend, PortableSyncBackend, SqliteMemoryStore,
};
#[cfg(feature = "legacy-sqlite")]
pub use synaptic_store::{
    SYNAPTIC_CAPTURE_ALGORITHM_V1, SYNAPTIC_CAPTURE_ALGORITHM_V2, SYNAPTIC_CAPTURE_CLAIM_BOUNDARY,
    SYNAPTIC_CAPTURE_SCHEMA_V1, SYNAPTIC_CAPTURE_SCHEMA_V2, SYNAPTIC_CONTEXT_ALGORITHM_V1,
    SYNAPTIC_CONTEXT_THRESHOLD_V1,
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
#[cfg(feature = "legacy-sqlite")]
pub use walk_receipts::{WALK_RECEIPT_SCHEMA_V1, canonical_walk_json};

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
#[cfg(feature = "legacy-sqlite")]
pub fn default_db_path() -> Result<std::path::PathBuf> {
    SqliteMemoryStore::default_db_path()
}

/// Database artifact path for a given data directory (SQLite backend).
#[cfg(feature = "legacy-sqlite")]
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

/// Feature-off twin of [`open_storage`]: always fails because no legacy
/// backend exists in this build; the STRATA backend constructs directly.
#[cfg(not(feature = "legacy-sqlite"))]
pub fn open_storage(
    _path: Option<std::path::PathBuf>,
) -> std::result::Result<std::sync::Arc<dyn MemoryStore>, LegacySqliteDisabled> {
    Err(LegacySqliteDisabled)
}
