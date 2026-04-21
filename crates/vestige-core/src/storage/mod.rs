//! Storage Module
//!
//! Backend-agnostic memory store abstraction plus SQLite reference impl.

mod memory_store;
mod migrations;
mod portable;
mod sqlite;

pub use memory_store::{
    ClassificationResult, Domain, HealthStatus, LocalMemoryStore, MemoryEdge, MemoryRecord,
    MemoryStore, MemoryStoreError, MemoryStoreResult, ModelSignature, SchedulingState,
    SearchQuery, SearchResult, StoreStats,
};
pub use migrations::MIGRATIONS;
pub use portable::{
    PORTABLE_ARCHIVE_FORMAT, PortableArchive, PortableImportMode, PortableImportReport,
    PortableTable, PortableValue,
};
pub use sqlite::{
    ConnectionRecord, ConsolidationHistoryRecord, DreamHistoryRecord, FilePortableSyncBackend,
    InsightRecord, IntentionRecord, PortableSyncBackend, PortableSyncReport, Result,
    SmartIngestResult, SqliteMemoryStore, StateTransitionRecord, StorageError,
};

/// Backwards-compatibility alias. Retained until Phase 4 completes so every
/// existing `Arc<Storage>` call site keeps compiling. Scheduled for removal
/// once no downstream source file references it.
pub type Storage = SqliteMemoryStore;
