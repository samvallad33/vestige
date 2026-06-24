//! Storage Module
//!
//! Backend-agnostic memory store abstraction plus SQLite reference impl.

#[cfg(feature = "cloud-sync")]
mod cloud_crypto;
#[cfg(feature = "cloud-sync")]
mod cloud_sync;
mod memory_store;
mod migrations;
mod portable;
mod sqlite;
mod trace_store;

#[cfg(feature = "cloud-sync")]
pub use cloud_sync::HttpPortableSyncBackend;

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
pub use sqlite::{
    CompositionEventRecord, CompositionMemberRecord, CompositionNeighborRecord,
    CompositionOutcomeRecord, ConnectionRecord, ConnectorCursor, ConsolidationHistoryRecord,
    DreamHistoryRecord, FilePortableSyncBackend, InsightRecord, IntentionRecord,
    NeverComposedCandidate, PortableSyncBackend, PortableSyncReport, ReconcileReport, Result,
    SmartIngestResult, SourceUpsertOutcome, SourceUpsertResult, SqliteMemoryStore,
    StateTransitionRecord, StorageError,
};
pub use trace_store::AgentRunSummary;

/// Backwards-compatibility alias. Retained until Phase 4 completes so every
/// existing `Arc<Storage>` call site keeps compiling. Scheduled for removal
/// once no downstream source file references it.
pub type Storage = SqliteMemoryStore;
