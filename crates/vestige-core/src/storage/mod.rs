//! Storage Module
//!
//! SQLite-based storage layer with:
//! - FTS5 full-text search with query sanitization
//! - Embedded vector storage
//! - FSRS-6 state management
//! - Temporal memory support

mod migrations;
mod sqlite;

pub use migrations::MIGRATIONS;
pub use sqlite::{
    ConsolidationHistoryRecord, InsightRecord, IntentionRecord, Result, SmartIngestResult, Storage,
    StorageError,
};
