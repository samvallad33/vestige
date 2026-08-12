//! Storage Module
//!
//! Backend-agnostic memory store abstraction plus SQLite reference impl.

mod attestation_store;
#[cfg(feature = "cloud-sync")]
mod cloud_crypto;
#[cfg(feature = "cloud-sync")]
mod cloud_sync;
mod memory_store;
mod migrations;
mod portable;
pub mod receipt_attestation;
mod replay_store;
mod sqlite;
mod synaptic_store;
mod trace_store;
pub mod unlearning;
mod unlearning_store;

#[cfg(feature = "cloud-sync")]
pub use cloud_sync::HttpPortableSyncBackend;

pub use attestation_store::{
    DurableSignedReceipt, DurableSignedRetrievalReceipt, ProvisionedReceiptSigningKey,
    ReceiptAttestationStatus, ReceiptSigningKeyTransition, SignedReceiptWrite,
    StoredReceiptAttestationVerification, load_receipt_signing_seed,
    provision_receipt_signing_key_sidecar,
};
pub use memory_store::{
    ClassificationResult, Domain, HealthStatus, LocalMemoryStore, MemoryEdge, MemoryRecord,
    MemoryStore, MemoryStoreError, MemoryStoreResult, MemoryStoreSend, ModelSignature,
    SchedulingState, SearchQuery, SearchResult, StoreStats,
};
pub use migrations::MIGRATIONS;
pub use portable::{
    PORTABLE_ARCHIVE_FORMAT, PortableArchive, PortableImportMode, PortableImportReport,
    PortableTable, PortableValue,
};
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
pub use sqlite::{
    CompositionEventRecord, CompositionMemberRecord, CompositionNeighborRecord,
    CompositionOutcomeRecord, ConnectionRecord, ConnectorCursor, ConsolidationHistoryRecord,
    DEFAULT_MEMORY_SCOPE, DreamHistoryRecord, EmbeddingProfileIntegrityManifest,
    EmbeddingProfileMigrationNodeCheckpoint, EmbeddingProfileMigrationRecord,
    EmbeddingProfileVector, FilePortableSyncBackend, InsightRecord, IntentionRecord,
    NeverComposedCandidate, PortableSyncBackend, PortableSyncReport, ReconcileReport, Result,
    SmartIngestResult, SourceUpsertOutcome, SourceUpsertResult, SqliteMemoryStore,
    StateTransitionRecord, StorageError,
};
pub use synaptic_store::{
    DurableSynapticCapture, DurableSynapticPairReceipt, SYNAPTIC_CAPTURE_ALGORITHM_V1,
    SYNAPTIC_CAPTURE_ALGORITHM_V2, SYNAPTIC_CAPTURE_CLAIM_BOUNDARY, SYNAPTIC_CAPTURE_SCHEMA_V1,
    SYNAPTIC_CAPTURE_SCHEMA_V2, SYNAPTIC_CONTEXT_ALGORITHM_V1, SYNAPTIC_CONTEXT_THRESHOLD_V1,
    SynapticCapturePolicy, SynapticCaptureRequest, SynapticImportanceEvent, SynapticIngestOutcome,
    SynapticIngestRequest, SynapticSignalSnapshot,
};
pub use trace_store::{
    AgentRunSummary, PendingMemoryMutationDecision, PendingMemoryMutationEffect,
};
pub use unlearning::{
    AntiResurrectionCommitments, ArtifactKind, ArtifactRef, CheckStatus, Commitment, CommitmentKey,
    CommitmentKind, ErasureLedgerRecord, GuaranteeExclusion, LineageClosure, LineageEdge,
    LineageRelation, PostconditionCheck, PostconditionKind, PostconditionReport, SurfaceAction,
    SurfaceDetailCode, SurfaceResult, UnlearningScope, UnlearningVerdict,
    VERIFIED_LOCAL_UNLEARNING_SCHEMA_V1, VerificationFence, anti_resurrection_commitments,
    commit_lineage_closure, compute_lineage_closure, evaluate_unlearning_verdict,
};
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
pub type Storage = SqliteMemoryStore;
