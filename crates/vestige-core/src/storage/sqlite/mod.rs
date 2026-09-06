//! SQLite Storage Implementation
//!
//! Core storage layer with integrated embeddings and vector search.

use chrono::{DateTime, Duration, NaiveDateTime, Utc};
use directories::{BaseDirs, ProjectDirs};
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
use lru::LruCache;
use rusqlite::types::{Type, Value, ValueRef};
use rusqlite::{Connection, OptionalExtension, params, params_from_iter};
use std::collections::{HashMap, HashSet};
use std::io::Write;
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
use std::num::NonZeroUsize;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
use std::sync::{Arc, RwLock};
use uuid::Uuid;

use crate::embedding::{
    ActiveEmbeddingProfile, BuiltinEmbeddingProfile, EmbeddingMigrationState, EmbeddingProfileId,
    EmbeddingProfileManifest, EmbeddingProfileState, ProfileMigrationCheckpoint,
    VerificationStatus,
};
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
use crate::embedding::{EmbeddingRuntimeBackend, ProfiledEmbedder};
use crate::fsrs::{
    DEFAULT_DECAY, FSRSScheduler, FSRSState, LearningState, MAX_STABILITY, Rating,
    retrievability_with_decay,
};
use crate::fts::{sanitize_fts5_or_query, sanitize_fts5_query};
use crate::memory::{
    ConsolidationResult, IngestInput, KnowledgeNode, MatchType, MemoryStats, RecallInput,
    SearchMode, SearchResult,
};
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
use crate::memory::{EmbeddingResult, SimilarityResult};
use crate::security::{SecretFinding, SecretPolicy, scan_secrets};
use crate::storage::portable::{
    PORTABLE_ARCHIVE_FORMAT, PortableArchive, PortableImportMode, PortableImportReport,
    PortableTable, PortableValue, encode_hex,
};

#[cfg(all(test, feature = "embeddings"))]
use crate::embeddings::EMBEDDING_DIMENSIONS;
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
use crate::embeddings::Embedding;
#[cfg(feature = "embeddings")]
use crate::embeddings::EmbeddingService;

#[cfg(feature = "vector-search")]
use crate::search::{VectorIndex, VectorIndexConfig, reciprocal_rank_fusion};

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
use crate::search::hyde;

mod admin;
mod connectors;
mod embeddings;
mod ingest;
mod lifecycle;
mod merge;
mod purge;
mod records;
mod search;
mod store_trait;
mod sync;

pub use connectors::{ConnectorCursor, ReconcileReport, SourceUpsertOutcome, SourceUpsertResult};
pub use records::{
    CompositionEventRecord, CompositionMemberRecord, CompositionNeighborRecord,
    CompositionOutcomeRecord, ConnectionRecord, ConsolidationHistoryRecord, DreamHistoryRecord,
    InsightRecord, IntentionRecord, NeverComposedCandidate, StateTransitionRecord,
};

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Storage error type
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    /// Database error
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),
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
    /// A profile operation would violate the explicit/reversible embedding
    /// profile contract.
    #[error("Invalid embedding profile: {0}")]
    InvalidEmbeddingProfile(String),
}

/// Storage result type
pub type Result<T> = std::result::Result<T, StorageError>;

/// Namespace used by existing, unscoped callers and by rows written before
/// project scopes were exposed. Scoped callers must opt into a different value.
pub const DEFAULT_MEMORY_SCOPE: &str = "user";
const MAX_TAG_MUTATION_MEMORIES: usize = 50_000;
const MAX_TAG_MUTATION_AUDIT_BYTES: usize = 16 * 1024 * 1024;
/// Retention window for `memory_access_log` rows. `prune_access_log` deletes
/// everything older on every consolidation, so any "never accessed" claim is
/// only meaningful for memories created inside this window.
pub const ACCESS_LOG_RETENTION_DAYS: i64 = 90;
/// Cap on the malformed-row id list surfaced by [`HygieneSnapshot`].
const MAX_MALFORMED_TAG_ROW_IDS: usize = 50;

#[cfg(test)]
thread_local! {
    /// Test-only fail point armed by regression tests to prove the tag
    /// UPDATE loop and its audit INSERT share one SQLite transaction: an
    /// injected failure between them must roll back both. Invisible in
    /// release builds.
    static FAIL_TAG_MUTATION_BEFORE_AUDIT: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}
type TagMutationState = (
    std::collections::BTreeMap<String, usize>,
    usize,
    Vec<(String, Vec<String>, Vec<String>)>,
);

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

#[cfg(any(test, all(feature = "embeddings", feature = "vector-search")))]
fn temporal_candidate_is_eligible(
    incoming_from: Option<DateTime<Utc>>,
    incoming_until: Option<DateTime<Utc>>,
    existing_from: Option<DateTime<Utc>>,
    existing_is_current: bool,
    now: DateTime<Utc>,
) -> bool {
    let incoming_is_older = match (incoming_from, existing_from) {
        (Some(incoming), Some(existing)) => incoming < existing,
        _ => false,
    };
    let incoming_is_expired = incoming_until.is_some_and(|until| until < now);
    !incoming_is_older && !(incoming_is_expired && existing_is_current)
}

#[cfg(test)]
mod temporal_candidate_tests {
    use super::temporal_candidate_is_eligible;
    use chrono::{Duration, Utc};

    #[test]
    fn older_dated_summary_cannot_mutate_newer_current_policy() {
        let now = Utc::now();
        assert!(!temporal_candidate_is_eligible(
            Some(now - Duration::days(365)),
            Some(now - Duration::days(180)),
            Some(now - Duration::days(30)),
            true,
            now,
        ));
    }

    #[test]
    fn newer_policy_remains_eligible_to_replace_an_older_fact() {
        let now = Utc::now();
        assert!(temporal_candidate_is_eligible(
            Some(now),
            None,
            Some(now - Duration::days(30)),
            true,
            now,
        ));
    }
}

/// Environment variable selecting the SQLite commit-durability policy.
pub const VESTIGE_SQLITE_DURABILITY_ENV: &str = "VESTIGE_SQLITE_DURABILITY";

/// SQLite durability policy for persistent Vestige databases.
///
/// `Hardened` is the default and acknowledges a commit only after SQLite has
/// used its FULL WAL synchronization path. `Balanced` preserves the historical
/// WAL + NORMAL behavior for operators who explicitly accept the power-loss
/// window in exchange for lower write latency.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SqliteDurabilityProfile {
    /// WAL + FULL, with macOS full-fsync requests enabled.
    #[default]
    Hardened,
    /// WAL + NORMAL, preserving the pre-hardening performance profile.
    Balanced,
}

impl SqliteDurabilityProfile {
    /// Stable lowercase profile name used in status output and configuration.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hardened => "hardened",
            Self::Balanced => "balanced",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "hardened" => Ok(Self::Hardened),
            "balanced" => Ok(Self::Balanced),
            _ => Err(StorageError::Init(format!(
                "Invalid {VESTIGE_SQLITE_DURABILITY_ENV} value '{value}'; expected hardened|balanced"
            ))),
        }
    }

    fn from_env() -> Result<Self> {
        match std::env::var(VESTIGE_SQLITE_DURABILITY_ENV) {
            Ok(value) => Self::parse(&value),
            Err(std::env::VarError::NotPresent) => Ok(Self::default()),
            Err(std::env::VarError::NotUnicode(_)) => Err(StorageError::Init(format!(
                "{VESTIGE_SQLITE_DURABILITY_ENV} must be valid UTF-8 and one of hardened|balanced"
            ))),
        }
    }
}

/// Effective SQLite PRAGMAs read back from one live connection.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SqliteConnectionPragmas {
    pub journal_mode: String,
    pub synchronous: i64,
    pub synchronous_label: String,
    pub fullfsync_enabled: bool,
    pub fullfsync_meaningful_on_this_platform: bool,
    pub checkpoint_fullfsync_enabled: bool,
    pub wal_autocheckpoint_pages: i64,
    pub foreign_keys_enabled: bool,
    pub busy_timeout_ms: i64,
}

/// Result of integrity and V21 receipt-consistency checks at one startup phase.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SqliteIntegrityStatus {
    pub quick_check: String,
    pub foreign_key_violations: u64,
    pub synaptic_checks_applied: bool,
    pub synaptic_consistency_violations: u64,
}

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

/// Verified startup durability and recovery state retained by the store.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SqliteDurabilityStatus {
    pub profile: SqliteDurabilityProfile,
    pub writer: SqliteConnectionPragmas,
    pub reader: SqliteConnectionPragmas,
    pub before_migrations: SqliteIntegrityStatus,
    pub after_migrations: SqliteIntegrityStatus,
    pub startup_checkpoint: WalCheckpointStatus,
    pub commit_acknowledgement: String,
    pub claim_boundary: String,
}

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergeWrite {
    Inserted,
    Updated,
}

/// Backend interface for portable sync storage.
///
/// The first shipped backend is a local file, which works with Dropbox, iCloud,
/// Syncthing, Git, shared volumes, or any other folder sync tool. Remote stores
/// can implement this trait without changing merge semantics.
pub trait PortableSyncBackend {
    /// Human-readable backend label for reports.
    fn label(&self) -> String;
    /// Read the current remote archive. `Ok(None)` means no remote exists yet.
    fn read_archive(&self) -> Result<Option<PortableArchive>>;
    /// Atomically write the merged archive back to the backend when possible.
    fn write_archive(&self, archive: &PortableArchive) -> Result<()>;
}

/// File-backed portable sync backend.
#[derive(Debug, Clone)]
pub struct FilePortableSyncBackend {
    path: PathBuf,
}

impl FilePortableSyncBackend {
    /// Create a file-backed sync backend for a portable archive path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Archive path backing this sync store.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl PortableSyncBackend for FilePortableSyncBackend {
    fn label(&self) -> String {
        format!("file:{}", self.path.display())
    }

    fn read_archive(&self) -> Result<Option<PortableArchive>> {
        if !self.path.exists() {
            return Ok(None);
        }
        let file = std::fs::File::open(&self.path)?;
        let archive: PortableArchive = serde_json::from_reader(file).map_err(|e| {
            StorageError::Init(format!(
                "Failed to parse portable sync archive '{}': {}",
                self.path.display(),
                e
            ))
        })?;
        Ok(Some(archive))
    }

    fn write_archive(&self, archive: &PortableArchive) -> Result<()> {
        let parent = self.path.parent().unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent)?;
        let filename = self
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("vestige-sync.json");
        let temp_path = parent.join(format!(".{}.tmp-{}", filename, Uuid::new_v4()));

        #[cfg(unix)]
        let mut file = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&temp_path)?
        };
        #[cfg(not(unix))]
        let mut file = std::fs::File::create(&temp_path)?;
        if let Err(e) = serde_json::to_writer_pretty(&mut file, archive) {
            let _ = std::fs::remove_file(&temp_path);
            return Err(StorageError::Init(format!(
                "Failed to write portable sync archive '{}': {}",
                self.path.display(),
                e
            )));
        }
        file.flush()?;
        file.sync_all()?;
        drop(file);

        if let Err(rename_err) = std::fs::rename(&temp_path, &self.path) {
            if self.path.exists() {
                std::fs::remove_file(&self.path)?;
                std::fs::rename(&temp_path, &self.path)?;
            } else {
                let _ = std::fs::remove_file(&temp_path);
                return Err(rename_err.into());
            }
        }
        Ok(())
    }
}

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
}

/// Persistent vector row belonging to exactly one embedding profile.
///
/// The bytes are intentionally opaque here: storage must not reinterpret or
/// compare vectors from two profiles. The runtime validates encoding and builds
/// a profile-specific index before it ever performs semantic retrieval.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingProfileVector {
    pub profile_id: String,
    pub node_id: String,
    pub embedding: Vec<u8>,
    pub dimensions: u32,
    pub model: String,
    pub created_at: DateTime<Utc>,
}

/// Integrity evidence persisted beside a profile's vector rows and HNSW
/// sidecar. The runtime owns the meaning of `manifest_json`, while SQLite owns
/// atomic persistence and count bookkeeping.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingProfileIntegrityManifest {
    pub profile_id: String,
    pub manifest_json: serde_json::Value,
    pub manifest_hash: String,
    pub vector_count: u64,
    pub index_member_count: u64,
    pub index_integrity_hash: Option<String>,
    pub updated_at: DateTime<Utc>,
}

/// A durable migration run. Checkpoints live in a sibling table so a crash can
/// resume at memory granularity without ever altering the active profile.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingProfileMigrationRecord {
    pub migration_id: String,
    pub source_profile_id: String,
    pub destination_profile_id: String,
    pub state: String,
    pub total_memories: u64,
    pub completed_memories: u64,
    pub failed_memory_ids: Vec<String>,
    pub last_memory_id: Option<String>,
    pub snapshot_path: Option<String>,
    pub validation_report: Option<serde_json::Value>,
    pub started_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Per-memory migration progress. A failed row is retained rather than hidden
/// behind a misleading completed state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingProfileMigrationNodeCheckpoint {
    pub migration_id: String,
    pub node_id: String,
    pub state: String,
    pub error: Option<String>,
    pub updated_at: DateTime<Utc>,
}

type EmbeddingProfileMigrationRow = (
    String,
    String,
    String,
    i64,
    i64,
    String,
    Option<String>,
    String,
    String,
);

// ============================================================================
// STORAGE
// ============================================================================

const PORTABLE_TABLES: &[&str] = &[
    "knowledge_nodes",
    "node_embeddings",
    "fsrs_cards",
    "memory_states",
    "memory_connections",
    "memory_access_log",
    "state_transitions",
    "intentions",
    "insights",
    "sessions",
    "fsrs_config",
    "consolidation_history",
    "dream_history",
    "retention_snapshots",
    "sync_tombstones",
    "deletion_tombstones",
    "composition_events",
    "composition_members",
    "composition_outcomes",
];

const PORTABLE_USER_DATA_TABLES: &[&str] = &[
    "knowledge_nodes",
    "node_embeddings",
    "fsrs_cards",
    "memory_states",
    "memory_connections",
    "memory_access_log",
    "state_transitions",
    "intentions",
    "insights",
    "sessions",
    "consolidation_history",
    "dream_history",
    "retention_snapshots",
    "sync_tombstones",
    "deletion_tombstones",
    "composition_events",
    "composition_members",
    "composition_outcomes",
];

#[derive(Default)]
struct PortableMergeState {
    locally_newer_nodes: HashSet<String>,
}

/// Effects produced by the shared local/portable deletion coordinator.
///
/// Keeping these counters separate from the public report lets portable sync
/// execute the identical cleanup inside its existing merge transaction.
pub(crate) struct PurgeCleanup {
    edges_pruned: i64,
    insights_rewritten: i64,
    insights_deleted: i64,
    children_orphaned: i64,
}

const DATA_DIR_ENV: &str = "VESTIGE_DATA_DIR";
const DATABASE_FILE: &str = "vestige.db";
#[cfg(feature = "vector-search")]
const VESTIGE_DISABLE_VECTOR_SEARCH: &str = "VESTIGE_DISABLE_VECTOR_SEARCH";

// Test-only override for the runtime vector-search gate, scoped to the
// current thread. Tests run in parallel inside one process, so a test that
// wants the index disabled must not touch the process environment: every
// other test thread building a `Storage` at that moment would silently get
// no index. This cell is what `with_vector_search_disabled` flips instead.
#[cfg(all(test, feature = "vector-search"))]
thread_local! {
    static VECTOR_SEARCH_DISABLED_FOR_TEST: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

// Test-only override for `VESTIGE_AUTO_CONSOLIDATE_MERGE`, scoped to the
// current thread for the same reason as the vector-search override: this
// gate decides whether consolidation hard-deletes near-duplicates, so a
// process-wide flag would reach every consolidation test running at once.
// `Some(None)` pins the variable unset; `Some(Some(v))` pins a value.
#[cfg(all(test, feature = "embeddings", feature = "vector-search"))]
thread_local! {
    static AUTO_CONSOLIDATE_MERGE_FOR_TEST: std::cell::RefCell<Option<Option<String>>> =
        const { std::cell::RefCell::new(None) };
}

/// Whether an environment value asks for vector search to be turned off.
/// Only affirmative values count, so `VESTIGE_DISABLE_VECTOR_SEARCH=0` leaves
/// the index on and reports it as on.
#[cfg(feature = "vector-search")]
fn env_value_disables_vector_search(value: &std::ffi::OsStr) -> bool {
    let value = value.to_ascii_lowercase();
    matches!(
        value.to_str(),
        Some("1" | "true" | "yes" | "on" | "enable" | "enabled")
    )
}
/// Immutable compatibility identity for vectors written before Embedding
/// Profiles existed. It is deliberately explicit: raw-text vectors must never
/// be confused with the corrected Nomic retrieval encoding contract.
pub const LEGACY_EMBEDDING_PROFILE_ID: &str = "nomic-v1.5-legacy-raw-256";

/// Main storage struct with integrated embedding and vector search
///
/// Uses separate reader/writer connections for interior mutability.
/// All methods take `&self` (not `&mut self`), making Storage `Send + Sync`
/// so the MCP layer can use `Arc<Storage>` instead of `Arc<Mutex<Storage>>`.
pub struct SqliteMemoryStore {
    db_path: PathBuf,
    durability_status: SqliteDurabilityStatus,
    // `pub(crate)` so the sibling `trace_store` module (Black Box / Receipts /
    // Memory PRs CRUD) can lock the same writer/reader connections and follow
    // the established store idiom without duplicating connection management.
    pub(crate) writer: Mutex<Connection>,
    pub(crate) reader: Mutex<Connection>,
    scheduler: Mutex<FSRSScheduler>,
    #[cfg(feature = "embeddings")]
    embedding_service: EmbeddingService,
    #[cfg(feature = "vector-search")]
    vector_index: Option<Mutex<VectorIndex>>,
    /// LRU cache for query embeddings to avoid re-embedding repeated queries
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    query_cache: Option<Mutex<LruCache<String, Vec<f32>>>>,
    /// Explicit, process-local runtime for an active optional embedding
    /// profile.  It is never restored from disk: a caller must re-verify and
    /// attach local artifacts in every process before Qwen retrieval can run.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    attached_profile_runtime: RwLock<Option<AttachedProfileRuntime>>,
    /// Cached model signature. `None` until the first embedding is written.
    registered_model: std::sync::RwLock<Option<crate::storage::memory_store::ModelSignature>>,
    /// Where this process's vector index stands relative to the shared
    /// database: the last `PRAGMA data_version` it saw and the last
    /// `vector_journal.seq` it absorbed. See `refresh_vector_index_if_stale`.
    #[cfg(feature = "vector-search")]
    vector_index_watermark: Mutex<VectorIndexWatermark>,
}

/// Where the in-process vector index stands relative to the shared database
/// (#181). See `SqliteMemoryStore::refresh_vector_index_if_stale`.
#[cfg(feature = "vector-search")]
#[derive(Debug, Clone, Copy)]
struct VectorIndexWatermark {
    /// Last `PRAGMA data_version` observed on the reader connection.
    ///
    /// SQLite increments this on a connection whenever ANOTHER connection commits
    /// to the database. It is the cheapest possible cross-process change signal:
    /// no table scan, no file stat. It only says THAT something changed; the
    /// journal says what. `-1` means never read.
    data_version: i64,
    /// Highest `vector_journal.seq` this index has absorbed. Every row past it
    /// is a vector write this process has not seen. `-1` means unknown, which
    /// makes the next refresh reconcile the index against the table instead of
    /// trusting the journal.
    journal_seq: i64,
}

#[cfg(feature = "vector-search")]
impl Default for VectorIndexWatermark {
    fn default() -> Self {
        Self {
            data_version: -1,
            journal_seq: -1,
        }
    }
}

/// What a refresh found in the journal past the watermark.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
enum VectorRefreshPlan {
    /// The journal is intact: apply exactly these per-node changes (`None` is a
    /// removal) and move the watermark to `head`.
    Incremental {
        changes: Vec<(String, Option<Vec<u8>>)>,
        head: i64,
    },
    /// The watermark is unknown or the journal was pruned past it: compare the
    /// index against the table instead.
    Reconcile,
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[derive(Clone)]
struct AttachedProfileRuntime {
    profile_id: EmbeddingProfileId,
    embedder: Arc<ProfiledEmbedder>,
}

/// Row-mapping errors are never dropped silently. Each unreadable row is
/// logged with the operation that skipped it; the operation still completes
/// on the rows it could read.
fn warn_skipped_row<T>(operation: &'static str) -> impl FnMut(rusqlite::Result<T>) -> Option<T> {
    move |row| match row {
        Ok(value) => Some(value),
        Err(error) => {
            tracing::warn!(%error, operation, "Skipping an unreadable row");
            None
        }
    }
}

/// What one post-retrieval failure feedback pass did. See
/// [`SqliteMemoryStore::apply_failure_feedback`].
#[derive(Debug, Clone, serde::Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FailureFeedbackReport {
    pub failure_id: String,
    pub window_minutes: i64,
    pub receipts_considered: usize,
    pub memories_demoted: usize,
    pub total_delta: f64,
}

/// Begin a READ snapshot on the reader connection.
///
/// A DEFERRED transaction on a connection that only reads gives several
/// statements one consistent view of the database (WAL snapshot isolation),
/// which is what a "rows plus the journal position that describes them" read
/// needs. It must never be used on the writer: a DEFERRED transaction that
/// reads and then writes can fail with `SQLITE_BUSY_SNAPSHOT`, and SQLite does
/// not consult the busy handler for that upgrade. Writers go through
/// [`SqliteMemoryStore::begin_write_transaction`], which begins IMMEDIATE. The
/// `write_transaction_policy` lint enforces both halves of that split.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
fn begin_read_snapshot(conn: &Connection) -> Result<rusqlite::Transaction<'_>> {
    Ok(rusqlite::Transaction::new_unchecked(
        conn,
        rusqlite::TransactionBehavior::Deferred,
    )?)
}

/// Truncate `content` to `max` chars on a char boundary, collapsing newlines.
fn preview(content: &str, max: usize) -> String {
    let c = content.replace('\n', " ");
    if c.len() > max {
        format!("{}...", &c[..c.floor_char_boundary(max)])
    } else {
        c
    }
}

impl SqliteMemoryStore {
    /// Begin a WRITE transaction on the writer connection.
    ///
    /// `BEGIN IMMEDIATE` takes the write lock up front, where `busy_timeout`
    /// (5 s) applies, and SQLite then guarantees no `SQLITE_BUSY` until
    /// `COMMIT`; a DEFERRED transaction that reads first could instead fail
    /// with `SQLITE_BUSY_SNAPSHOT` the moment another process committed. On
    /// top of the busy timeout this retries `BUSY`/`LOCKED` three times with
    /// 100/200/400 ms backoff and logs each retry with the calling operation,
    /// so a CLI backup running beside the MCP server shows up in the log
    /// instead of as a failed write.
    pub(super) fn begin_write_transaction<'c>(
        conn: &'c Connection,
        operation: &'static str,
    ) -> Result<rusqlite::Transaction<'c>> {
        const RETRY_DELAYS_MS: [u64; 3] = [100, 200, 400];
        let mut attempt = 0usize;
        loop {
            // `new_unchecked` takes a shared borrow (the writer connection is
            // already exclusive behind its mutex), which lets the retry loop
            // return the transaction without fighting the borrow checker.
            match rusqlite::Transaction::new_unchecked(
                conn,
                rusqlite::TransactionBehavior::Immediate,
            ) {
                Ok(tx) => return Ok(tx),
                Err(rusqlite::Error::SqliteFailure(error, message))
                    if matches!(
                        error.code,
                        rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
                    ) && attempt < RETRY_DELAYS_MS.len() =>
                {
                    let delay_ms = RETRY_DELAYS_MS[attempt];
                    attempt += 1;
                    tracing::warn!(
                        operation,
                        attempt,
                        delay_ms,
                        code = ?error.code,
                        detail = message.as_deref().unwrap_or(""),
                        "SQLite write lock busy; retrying"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
}

#[cfg(test)]
mod tests;
