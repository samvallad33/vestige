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
const VESTIGE_DISABLE_VECTOR_SEARCH: &str = "VESTIGE_DISABLE_VECTOR_SEARCH";
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

impl SqliteMemoryStore {
    #[cfg(feature = "vector-search")]
    fn vector_search_enabled_by_cpu() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let has_required_features = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma");

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let has_required_features = true;

        let disabled_by_env = std::env::var_os(VESTIGE_DISABLE_VECTOR_SEARCH)
            .and_then(|v| {
                let value = v.to_ascii_lowercase();
                if value == "1"
                    || value == "true"
                    || value == "yes"
                    || value == "on"
                    || value == "enable"
                    || value == "enabled"
                {
                    Some(())
                } else {
                    None
                }
            })
            .is_some();

        has_required_features && !disabled_by_env
    }

    #[cfg(feature = "vector-search")]
    fn vector_search_unavailable_reason() -> Option<&'static str> {
        if std::env::var_os(VESTIGE_DISABLE_VECTOR_SEARCH).is_some() {
            return Some("disabled by VESTIGE_DISABLE_VECTOR_SEARCH");
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return Some("unsupported CPU: AVX2 required");
            }
            if !std::arch::is_x86_feature_detected!("fma") {
                return Some("unsupported CPU: FMA required");
            }
        }

        None
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn vector_search_available(&self) -> bool {
        self.vector_index.is_some()
    }

    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    fn vector_search_available(&self) -> bool {
        false
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn regular_ingest_result(
        &self,
        input: IngestInput,
        scope: &str,
        reason: impl Into<String>,
        policy: SecretPolicy,
    ) -> Result<SmartIngestResult> {
        let node = self.ingest_in_scope_with_secret_policy(input, scope, policy)?;
        Ok(SmartIngestResult {
            decision: "create".to_string(),
            node,
            superseded_id: None,
            similarity: None,
            prediction_error: Some(1.0),
            reason: reason.into(),
            previous_content: None,
            merged_from: None,
            merge_preview: None,
            auto_closed_until: None,
        })
    }

    fn data_dir_from_env() -> Option<PathBuf> {
        std::env::var_os(DATA_DIR_ENV).and_then(|value| {
            if value.is_empty() {
                None
            } else {
                Some(PathBuf::from(value))
            }
        })
    }

    fn expand_tilde(path: PathBuf) -> PathBuf {
        let rest = {
            let mut components = path.components();
            match components.next() {
                Some(Component::Normal(first)) if first == "~" => {
                    Some(components.as_path().to_path_buf())
                }
                _ => None,
            }
        };

        match rest {
            Some(rest) => BaseDirs::new()
                .map(|dirs| dirs.home_dir().join(rest))
                .unwrap_or(path),
            None => path,
        }
    }

    fn prepare_data_dir(data_dir: PathBuf) -> Result<PathBuf> {
        let data_dir = Self::expand_tilde(data_dir);
        std::fs::create_dir_all(&data_dir)?;
        // Restrict directory permissions to owner-only on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o700);
            let _ = std::fs::set_permissions(&data_dir, perms);
        }
        Ok(data_dir.join(DATABASE_FILE))
    }

    /// Resolve a Vestige database path from an explicit data directory.
    pub fn db_path_for_data_dir(data_dir: PathBuf) -> Result<PathBuf> {
        Self::prepare_data_dir(data_dir)
    }

    /// Resolve the default Vestige database path.
    ///
    /// `VESTIGE_DATA_DIR` is treated as a directory and wins over the platform
    /// per-user data directory. The database file is always `vestige.db` inside
    /// that directory.
    pub fn default_db_path() -> Result<PathBuf> {
        if let Some(data_dir) = Self::data_dir_from_env() {
            return Self::prepare_data_dir(data_dir);
        }

        let proj_dirs = ProjectDirs::from("com", "vestige", "core").ok_or_else(|| {
            StorageError::Init("Could not determine project directories".to_string())
        })?;

        Self::prepare_data_dir(proj_dirs.data_dir().to_path_buf())
    }

    /// Apply PRAGMAs and optional encryption to a connection.
    fn configure_connection(
        conn: &Connection,
        profile: SqliteDurabilityProfile,
        writer: bool,
    ) -> Result<()> {
        // Apply encryption key if SQLCipher is enabled and key is provided
        #[cfg(feature = "encryption")]
        {
            if let Ok(key) = std::env::var("VESTIGE_ENCRYPTION_KEY") {
                if !key.is_empty() {
                    conn.pragma_update(None, "key", &key)?;
                }
            }
        }

        // WAL is persistent database state, so only the writer requests the
        // transition. Every connection still receives its own synchronous,
        // foreign-key, timeout, and full-fsync settings.
        if writer {
            conn.execute_batch("PRAGMA journal_mode = WAL;")?;
        }

        let durability_pragmas = match profile {
            SqliteDurabilityProfile::Hardened => {
                "PRAGMA synchronous = FULL;
                 PRAGMA fullfsync = ON;
                 PRAGMA checkpoint_fullfsync = ON;"
            }
            SqliteDurabilityProfile::Balanced => {
                "PRAGMA synchronous = NORMAL;
                 PRAGMA fullfsync = OFF;
                 PRAGMA checkpoint_fullfsync = OFF;"
            }
        };
        conn.execute_batch(durability_pragmas)?;
        conn.execute_batch(
            "PRAGMA cache_size = -64000;
             PRAGMA temp_store = MEMORY;
             PRAGMA foreign_keys = ON;
             PRAGMA busy_timeout = 5000;
             PRAGMA mmap_size = 268435456;
             PRAGMA wal_autocheckpoint = 1000;
             PRAGMA journal_size_limit = 67108864;",
        )?;

        Ok(())
    }

    fn read_effective_pragmas(conn: &Connection) -> Result<SqliteConnectionPragmas> {
        let journal_mode: String = conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
        let synchronous: i64 = conn.query_row("PRAGMA synchronous", [], |row| row.get(0))?;
        let fullfsync: i64 = conn.query_row("PRAGMA fullfsync", [], |row| row.get(0))?;
        let checkpoint_fullfsync: i64 =
            conn.query_row("PRAGMA checkpoint_fullfsync", [], |row| row.get(0))?;
        let wal_autocheckpoint_pages: i64 =
            conn.query_row("PRAGMA wal_autocheckpoint", [], |row| row.get(0))?;
        let foreign_keys: i64 = conn.query_row("PRAGMA foreign_keys", [], |row| row.get(0))?;
        let busy_timeout_ms: i64 = conn.query_row("PRAGMA busy_timeout", [], |row| row.get(0))?;
        let synchronous_label = match synchronous {
            0 => "off",
            1 => "normal",
            2 => "full",
            3 => "extra",
            _ => "unknown",
        }
        .to_string();

        Ok(SqliteConnectionPragmas {
            journal_mode: journal_mode.to_ascii_lowercase(),
            synchronous,
            synchronous_label,
            fullfsync_enabled: fullfsync != 0,
            fullfsync_meaningful_on_this_platform: cfg!(target_os = "macos"),
            checkpoint_fullfsync_enabled: checkpoint_fullfsync != 0,
            wal_autocheckpoint_pages,
            foreign_keys_enabled: foreign_keys != 0,
            busy_timeout_ms,
        })
    }

    fn verify_effective_pragmas(
        profile: SqliteDurabilityProfile,
        role: &str,
        pragmas: &SqliteConnectionPragmas,
    ) -> Result<()> {
        if !pragmas.foreign_keys_enabled {
            return Err(StorageError::Init(format!(
                "SQLite {role} connection refused foreign_keys=ON"
            )));
        }
        if profile == SqliteDurabilityProfile::Hardened {
            if pragmas.journal_mode != "wal" {
                return Err(StorageError::Init(format!(
                    "Hardened SQLite startup refused durability downgrade: {role} journal_mode is '{}' instead of WAL",
                    pragmas.journal_mode
                )));
            }
            if pragmas.synchronous != 2 {
                return Err(StorageError::Init(format!(
                    "Hardened SQLite startup refused durability downgrade: {role} synchronous is '{}' instead of FULL",
                    pragmas.synchronous_label
                )));
            }
            #[cfg(target_os = "macos")]
            if !pragmas.fullfsync_enabled || !pragmas.checkpoint_fullfsync_enabled {
                return Err(StorageError::Init(format!(
                    "Hardened SQLite startup refused durability downgrade: {role} fullfsync={} checkpoint_fullfsync={} instead of both enabled",
                    pragmas.fullfsync_enabled, pragmas.checkpoint_fullfsync_enabled
                )));
            }
        }
        Ok(())
    }

    fn table_has_column(conn: &Connection, table: &str, column: &str) -> Result<bool> {
        let exists: i64 = conn.query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM pragma_table_info(?1) WHERE name = ?2
             )",
            params![table, column],
            |row| row.get(0),
        )?;
        Ok(exists != 0)
    }

    fn count_foreign_key_violations(conn: &Connection) -> Result<u64> {
        let mut stmt = conn.prepare("PRAGMA foreign_key_check")?;
        let mut rows = stmt.query([])?;
        let mut count = 0_u64;
        while rows.next()?.is_some() {
            count += 1;
        }
        Ok(count)
    }

    /// Delete orphaned child rows whose own foreign key is declared
    /// `ON DELETE CASCADE`. Such a row is unreachable -- its parent is already
    /// gone and the schema states it should have gone with it -- so removing it
    /// restores the invariant without discarding anything a reader could reach.
    /// Rows whose FK is NOT cascade-declared are deliberately left alone so the
    /// caller still fails loudly on genuine corruption.
    fn repair_cascade_orphans(conn: &Connection) -> Result<u64> {
        // (child_table, rowid) pairs reported by the checker.
        let violations: Vec<(String, Option<i64>)> = {
            let mut stmt = conn.prepare("PRAGMA foreign_key_check")?;
            let mapped = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Option<i64>>(1)?))
            })?;
            let mut v = Vec::new();
            for r in mapped {
                v.push(r?);
            }
            v
        };
        if violations.is_empty() {
            return Ok(0);
        }

        // A table is repairable only if EVERY one of its foreign keys that
        // points at knowledge_nodes is ON DELETE CASCADE.
        let mut cascade_ok: std::collections::HashMap<String, bool> =
            std::collections::HashMap::new();
        let mut repaired = 0_u64;
        // This transaction reads (`PRAGMA foreign_key_list`) before it writes
        // (`DELETE`), which is exactly the shape that returns
        // `SQLITE_BUSY_SNAPSHOT` under DEFERRED without consulting the busy
        // handler. It runs while the store is being opened, so a CLI writing
        // beside the server must not be able to fail the open.
        let tx = Self::begin_write_transaction(conn, "repair_cascade_orphans")?;
        for (table, rowid) in violations {
            let Some(rowid) = rowid else { continue };
            let ok = match cascade_ok.get(&table) {
                Some(v) => *v,
                None => {
                    let mut stmt = tx.prepare(&format!(
                        "PRAGMA foreign_key_list(\"{}\")",
                        table.replace('"', "\"\"")
                    ))?;
                    let rows = stmt.query_map([], |row| {
                        Ok((row.get::<_, String>(2)?, row.get::<_, String>(6)?))
                    })?;
                    let mut all_cascade = false;
                    for r in rows {
                        let (parent, on_delete) = r?;
                        if parent.eq_ignore_ascii_case("knowledge_nodes") {
                            all_cascade = on_delete.eq_ignore_ascii_case("CASCADE");
                            if !all_cascade {
                                break;
                            }
                        }
                    }
                    cascade_ok.insert(table.clone(), all_cascade);
                    all_cascade
                }
            };
            if !ok {
                continue;
            }
            let n = tx.execute(
                &format!(
                    "DELETE FROM \"{}\" WHERE rowid = ?1",
                    table.replace('"', "\"\"")
                ),
                params![rowid],
            )?;
            repaired += n as u64;
        }
        tx.commit()?;
        Ok(repaired)
    }

    /// Run `PRAGMA quick_check` and return its rows.
    fn quick_check_rows(conn: &Connection) -> Result<Vec<String>> {
        let mut out = Vec::new();
        let mut stmt = conn.prepare("PRAGMA quick_check")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Names of every FTS5 virtual table in the schema.
    fn fts5_table_names(conn: &Connection) -> Result<Vec<String>> {
        let mut stmt = conn.prepare(
            "SELECT name FROM sqlite_master \
             WHERE type = 'table' AND sql LIKE '%USING fts5%'",
        )?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Is every failing quick_check row attributable to an FTS5 index we can
    /// rebuild? A single row we cannot attribute means real damage, and the
    /// caller must fail rather than paper over it.
    fn quick_check_failure_is_only_fts5(rows: &[String], fts_tables: &[String]) -> bool {
        if fts_tables.is_empty() {
            return false;
        }
        rows.iter().filter(|r| r.as_str() != "ok").all(|r| {
            let lower = r.to_lowercase();
            lower.contains("fts5") && fts_tables.iter().any(|t| lower.contains(&t.to_lowercase()))
        })
    }

    fn run_integrity_checks(conn: &Connection, phase: &str) -> Result<SqliteIntegrityStatus> {
        let mut quick_rows = Self::quick_check_rows(conn)?;

        // An FTS5 external-content index is DERIVED STATE. `knowledge_fts` is
        // declared `content='knowledge_nodes'`, so every token in it is
        // reconstructible from a table quick_check just verified. Refusing to
        // open the whole store because a rebuildable index is damaged strands
        // the user's memories behind an index we can regenerate in seconds --
        // and that is exactly what happened in the field: a store with 2,926
        // intact memories became unopenable over one corrupt fts5 blob.
        //
        // So: rebuild and re-check. Only if the rebuild fails, or the re-check
        // still fails, is this real corruption worth refusing over. This
        // deliberately mirrors the CASCADE-orphan repair below -- repair derived
        // state, fail loudly on genuine damage.
        //
        // Writer phases only. The runtime reader must never attempt a write.
        let repairable_phase = phase == "pre-migration" || phase == "post-migration";
        let quick_ok =
            |rows: &[String]| rows.len() == 1 && rows.first().map(String::as_str) == Some("ok");
        if !quick_ok(&quick_rows) && repairable_phase {
            let fts_tables = Self::fts5_table_names(conn)?;
            if Self::quick_check_failure_is_only_fts5(&quick_rows, &fts_tables) {
                let detail = quick_rows.join("; ");
                let mut rebuilt = Vec::new();
                for table in &fts_tables {
                    // Identifier is read back from sqlite_master, not user input.
                    let quoted = table.replace('"', "\"\"");
                    match conn.execute_batch(&format!(
                        "INSERT INTO \"{quoted}\"(\"{quoted}\") VALUES('rebuild');"
                    )) {
                        Ok(()) => rebuilt.push(table.clone()),
                        Err(error) => {
                            return Err(StorageError::Init(format!(
                                "SQLite {phase} quick_check failed ({detail}) and rebuilding \
                                 FTS index '{table}' also failed: {error}"
                            )));
                        }
                    }
                }
                quick_rows = Self::quick_check_rows(conn)?;
                if quick_ok(&quick_rows) {
                    tracing::warn!(
                        phase,
                        rebuilt = ?rebuilt,
                        detail,
                        "rebuilt corrupt FTS index from its content table; store opened normally"
                    );
                }
            }
        }

        let quick_check = quick_rows.join("; ");
        if !quick_ok(&quick_rows) {
            return Err(StorageError::Init(format!(
                "SQLite {phase} quick_check failed: {quick_check}"
            )));
        }

        let mut foreign_key_violations = Self::count_foreign_key_violations(conn)?;
        if foreign_key_violations != 0 && phase == "pre-migration" {
            // Deletion residue from older builds (and from any delete path that
            // ran without `PRAGMA foreign_keys = ON`) leaves child rows whose
            // knowledge_nodes parent is already gone. Those rows are unreachable
            // by construction, and their own schema says ON DELETE CASCADE --
            // "if the parent goes, I go". Refusing to open the database over
            // them bricks every store that predates FK enforcement, with no
            // recovery path short of manual SQLite surgery. Repair them here,
            // BEFORE migrations, then re-check. Only CASCADE-declared FKs are
            // repaired; anything else still fails loudly below.
            let repaired = Self::repair_cascade_orphans(conn)?;
            foreign_key_violations = Self::count_foreign_key_violations(conn)?;
            if repaired > 0 {
                tracing::warn!(
                    repaired,
                    remaining = foreign_key_violations,
                    "repaired orphaned child rows left by an earlier delete (ON DELETE CASCADE residue)"
                );
            }
        }
        if foreign_key_violations != 0 {
            return Err(StorageError::Init(format!(
                "SQLite {phase} foreign_key_check found {foreign_key_violations} violation(s)"
            )));
        }

        let synaptic_tables = [
            "synaptic_tags",
            "synaptic_events",
            "synaptic_capture_items",
            "memory_receipts",
        ];
        let mut synaptic_checks_applied = true;
        for table in synaptic_tables {
            if !Self::table_exists(conn, table)? {
                synaptic_checks_applied = false;
                break;
            }
        }

        let synaptic_consistency_violations = if synaptic_checks_applied {
            let missing_receipts: i64 = conn.query_row(
                "SELECT COUNT(*)
                 FROM synaptic_events e
                 LEFT JOIN memory_receipts r ON r.receipt_id = e.receipt_id
                 WHERE e.receipt_id IS NULL OR r.receipt_id IS NULL",
                [],
                |row| row.get(0),
            )?;
            let invalid_event_receipt_predicates =
                if Self::table_has_column(conn, "synaptic_events", "public_event_id")? {
                    conn.query_row(
                        "SELECT COUNT(*)
                     FROM synaptic_events e
                     JOIN memory_receipts r ON r.receipt_id = e.receipt_id
                     WHERE CASE json_extract(r.payload, '$.evidence.predicate.schemaVersion')
                         WHEN 1 THEN
                                json_extract(r.payload, '$.evidence.kind')
                                    IS NOT 'synaptic_capture'
                             OR e.algorithm_version IS NOT 'vestige.synaptic_capture.v1'
                             OR e.public_event_id IS NULL
                             OR json_extract(r.payload, '$.evidence.predicate.algorithmVersion')
                                    IS NOT 'vestige.synaptic_capture.v1'
                             OR json_type(r.payload, '$.evidence.predicate.trigger.eventId')
                                    IS NOT 'text'
                             OR json_extract(r.payload, '$.evidence.predicate.trigger.eventId')
                                    IS NOT e.public_event_id
                         WHEN 2 THEN
                                json_extract(r.payload, '$.evidence.kind')
                                    IS NOT 'synaptic_capture'
                             OR e.algorithm_version IS NOT 'vestige.synaptic_capture.v2'
                             OR e.public_event_id IS NULL
                             OR json_extract(r.payload, '$.evidence.predicate.algorithmVersion')
                                    IS NOT 'vestige.synaptic_capture.v2'
                             OR json_extract(r.payload, '$.evidence.predicate.receiptRole')
                                    IS NOT 'root'
                             OR json_type(
                                    r.payload,
                                    '$.evidence.predicate.parentReceiptId'
                                ) IS NOT NULL
                             OR json_type(r.payload, '$.evidence.predicate.trigger.eventId')
                                    IS NOT 'text'
                             OR json_extract(r.payload, '$.evidence.predicate.trigger.eventId')
                                    IS NOT e.public_event_id
                         ELSE 1
                     END",
                        [],
                        |row| row.get::<_, i64>(0),
                    )?
                } else {
                    0
                };
            let invalid_items: i64 = conn.query_row(
                "SELECT COUNT(*)
                 FROM synaptic_capture_items i
                 LEFT JOIN synaptic_events e ON e.event_id = i.event_id
                 LEFT JOIN synaptic_tags t ON t.tag_id = i.tag_id
                 LEFT JOIN memory_receipts r ON r.receipt_id = i.receipt_id
                 WHERE e.event_id IS NULL OR t.tag_id IS NULL OR r.receipt_id IS NULL
                    OR i.memory_id IS NOT t.memory_id",
                [],
                |row| row.get(0),
            )?;
            // V21 stores one root receipt id on both the event and every item.
            // V22 may store a per-pair child receipt on an item, so the startup
            // invariant becomes predicate-version aware once the V22 columns
            // exist. Preparing this SQL conditionally keeps pre-V22 databases
            // valid during checks that run before pending migrations.
            let invalid_item_receipt_predicates = if Self::table_has_column(
                conn,
                "synaptic_events",
                "public_event_id",
            )? && Self::table_has_column(
                conn,
                "synaptic_capture_items",
                "evaluation_direction",
            )? {
                conn.query_row(
                    "SELECT COUNT(*)
                     FROM synaptic_capture_items i
                     JOIN synaptic_events e ON e.event_id = i.event_id
                     JOIN memory_receipts r ON r.receipt_id = i.receipt_id
                     WHERE CASE json_extract(r.payload, '$.evidence.predicate.schemaVersion')
                         WHEN 1 THEN
                                i.receipt_id <> e.receipt_id
                             OR i.evaluation_direction IS NOT 'backward'
                             OR i.algorithm_version IS NOT 'vestige.synaptic_capture.v1'
                         WHEN 2 THEN
                                json_extract(r.payload, '$.evidence.kind') IS NOT 'synaptic_capture'
                             OR i.algorithm_version IS NOT 'vestige.synaptic_capture.v2'
                             OR i.evaluation_direction NOT IN ('backward', 'forward')
                             OR json_extract(r.payload, '$.evidence.predicate.algorithmVersion')
                                    IS NOT 'vestige.synaptic_capture.v2'
                             OR CASE i.evaluation_direction
                                  WHEN 'backward' THEN
                                         i.receipt_id <> e.receipt_id
                                      OR json_extract(r.payload, '$.evidence.predicate.receiptRole')
                                             IS NOT 'root'
                                      OR json_type(
                                             r.payload,
                                             '$.evidence.predicate.parentReceiptId'
                                         ) IS NOT NULL
                                  WHEN 'forward' THEN
                                         i.receipt_id = e.receipt_id
                                      OR json_extract(r.payload, '$.evidence.predicate.receiptRole')
                                             IS NOT 'pair'
                                      OR json_extract(r.payload, '$.evidence.predicate.parentReceiptId')
                                             IS NOT e.receipt_id
                                  ELSE 1
                                END
                             OR e.public_event_id IS NULL
                             OR json_type(r.payload, '$.evidence.predicate.trigger.eventId')
                                    IS NOT 'text'
                             OR json_extract(r.payload, '$.evidence.predicate.trigger.eventId')
                                    IS NOT e.public_event_id
                             OR json_extract(r.payload, '$.evidence.predicate.evaluationDirection')
                                    IS NOT i.evaluation_direction
                             OR json_array_length(
                                    json_extract(r.payload, '$.evidence.predicate.candidates')
                                ) IS NOT 1
                             OR json_extract(
                                    r.payload,
                                    '$.evidence.predicate.candidates[0].evidenceSlot'
                                ) IS NOT i.evidence_slot
                         ELSE 1
                     END",
                    [],
                    |row| row.get::<_, i64>(0),
                )?
            } else {
                conn.query_row(
                    "SELECT COUNT(*)
                     FROM synaptic_capture_items i
                     JOIN synaptic_events e ON e.event_id = i.event_id
                     WHERE i.receipt_id <> e.receipt_id",
                    [],
                    |row| row.get::<_, i64>(0),
                )?
            };
            let duplicate_active_tags: i64 = conn.query_row(
                "SELECT COUNT(*) FROM (
                     SELECT memory_id
                     FROM synaptic_tags
                     WHERE state = 'active'
                     GROUP BY memory_id
                     HAVING COUNT(*) > 1
                 )",
                [],
                |row| row.get(0),
            )?;
            let invalid_captured_tags: i64 = conn.query_row(
                "SELECT COUNT(*)
                 FROM synaptic_tags t
                 LEFT JOIN synaptic_events e ON e.event_id = t.capture_event_id
                 LEFT JOIN synaptic_capture_items i
                   ON i.event_id = t.capture_event_id
                  AND i.tag_id = t.tag_id
                  AND i.disposition = 'captured'
                 WHERE (t.state = 'captured' AND (
                           t.capture_event_id IS NULL
                        OR t.captured_at_ms IS NULL
                        OR e.event_id IS NULL
                        OR i.tag_id IS NULL
                       ))
                    OR (t.state <> 'captured' AND (
                           t.capture_event_id IS NOT NULL
                        OR t.captured_at_ms IS NOT NULL
                       ))",
                [],
                |row| row.get(0),
            )?;
            (missing_receipts
                + invalid_event_receipt_predicates
                + invalid_items
                + invalid_item_receipt_predicates
                + duplicate_active_tags
                + invalid_captured_tags) as u64
        } else {
            0
        };
        if synaptic_consistency_violations != 0 {
            return Err(StorageError::Init(format!(
                "SQLite {phase} synaptic receipt consistency checks found {synaptic_consistency_violations} violation(s)"
            )));
        }

        Ok(SqliteIntegrityStatus {
            quick_check,
            foreign_key_violations,
            synaptic_checks_applied,
            synaptic_consistency_violations,
        })
    }

    fn checkpoint_connection(
        conn: &Connection,
        mode: WalCheckpointMode,
    ) -> Result<WalCheckpointStatus> {
        let sql = match mode {
            WalCheckpointMode::Passive => "PRAGMA wal_checkpoint(PASSIVE)",
            WalCheckpointMode::Truncate => "PRAGMA wal_checkpoint(TRUNCATE)",
        };
        let (busy, log_frames, checkpointed_frames) =
            conn.query_row(sql, [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?;
        Ok(WalCheckpointStatus {
            busy,
            log_frames,
            checkpointed_frames,
        })
    }

    /// Create new storage instance
    pub fn new(db_path: Option<PathBuf>) -> Result<Self> {
        Self::new_with_durability(db_path, SqliteDurabilityProfile::from_env()?)
    }

    /// Create storage with an explicit durability policy.
    ///
    /// This is primarily useful for controlled benchmarks and embedded callers
    /// that cannot use process environment configuration.
    pub fn new_with_durability(
        db_path: Option<PathBuf>,
        profile: SqliteDurabilityProfile,
    ) -> Result<Self> {
        let path = match db_path {
            Some(p) => p,
            None => Self::default_db_path()?,
        };

        // Open writer connection
        let writer_conn = Connection::open(&path)?;

        // Restrict database file permissions to owner-only on Unix
        #[cfg(unix)]
        if path.exists() {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            let _ = std::fs::set_permissions(&path, perms);
        }

        Self::configure_connection(&writer_conn, profile, true)?;
        let writer_pragmas = Self::read_effective_pragmas(&writer_conn)?;
        Self::verify_effective_pragmas(profile, "writer", &writer_pragmas)?;

        // Opening the database lets SQLite recover a committed WAL. Validate
        // that recovered state before migrations can change the schema.
        let before_migrations = Self::run_integrity_checks(&writer_conn, "pre-migration")?;

        // Apply migrations on writer only
        super::migrations::apply_migrations(&writer_conn)?;
        // Issue #191: heal v1.x raw-768 vectors copied verbatim under the
        // 256-dim legacy profile before any strict dimension check can abort
        // startup. No-op on clean stores.
        super::migrations::repair_legacy_raw_profile_vectors(
            &writer_conn,
            LEGACY_EMBEDDING_PROFILE_ID,
        )?;
        writer_conn.execute_batch("PRAGMA optimize = 0x10002;")?;
        let after_migrations = Self::run_integrity_checks(&writer_conn, "post-migration")?;
        let startup_checkpoint =
            Self::checkpoint_connection(&writer_conn, WalCheckpointMode::Passive)?;

        // Open reader connection to same path
        let reader_conn = Connection::open(&path)?;
        Self::configure_connection(&reader_conn, profile, false)?;
        let reader_pragmas = Self::read_effective_pragmas(&reader_conn)?;
        Self::verify_effective_pragmas(profile, "reader", &reader_pragmas)?;

        let durability_status = SqliteDurabilityStatus {
            profile,
            writer: writer_pragmas,
            reader: reader_pragmas,
            before_migrations,
            after_migrations,
            startup_checkpoint,
            commit_acknowledgement: match profile {
                SqliteDurabilityProfile::Hardened => {
                    "tx.commit() returned after SQLite FULL WAL synchronization"
                }
                SqliteDurabilityProfile::Balanced => {
                    "tx.commit() returned under SQLite NORMAL WAL synchronization"
                }
            }
            .to_string(),
            claim_boundary: "Process-crash tests prove transaction atomicity and recovery at the tested commit boundaries. Power-loss durability still depends on the operating system, filesystem, controller, and storage device honoring completed flush requests; WAL requires local shared-memory and locking semantics."
                .to_string(),
        };

        #[cfg(feature = "embeddings")]
        let embedding_service = EmbeddingService::new();

        #[cfg(feature = "vector-search")]
        let vector_index = if Self::vector_search_enabled_by_cpu() {
            let vector_index = VectorIndex::new()
                .map_err(|e| StorageError::Init(format!("Failed to create vector index: {}", e)))?;
            Some(Mutex::new(vector_index))
        } else {
            tracing::warn!(
                "Vector search disabled: {}",
                Self::vector_search_unavailable_reason().unwrap_or("manual override"),
            );
            None
        };

        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        let query_cache = if vector_index.is_some() {
            Some(Mutex::new(LruCache::new(
                NonZeroUsize::new(100).expect("100 is non-zero"),
            )))
        } else {
            None
        };

        let storage = Self {
            db_path: path,
            durability_status,
            writer: Mutex::new(writer_conn),
            reader: Mutex::new(reader_conn),
            scheduler: Mutex::new(FSRSScheduler::default()),
            #[cfg(feature = "embeddings")]
            embedding_service,
            #[cfg(feature = "vector-search")]
            vector_index,
            #[cfg(feature = "vector-search")]
            vector_index_watermark: Mutex::new(VectorIndexWatermark::default()),
            #[cfg(all(feature = "embeddings", feature = "vector-search"))]
            query_cache,
            #[cfg(all(feature = "embeddings", feature = "vector-search"))]
            attached_profile_runtime: RwLock::new(None),
            registered_model: std::sync::RwLock::new(None),
        };

        // V20 seeds a minimal SQL row so old databases migrate atomically.
        // Replace that bootstrap with the complete, serializable manifest before
        // exposing the store to callers.
        storage.ensure_legacy_embedding_profile_manifest()?;

        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        if storage.vector_index.is_some() {
            storage.load_embeddings_into_index()?;
        }

        Ok(storage)
    }

    /// Absolute path of the SQLite database this storage instance uses.
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Verified durability profile and startup-recovery results.
    pub fn durability_status(&self) -> &SqliteDurabilityStatus {
        &self.durability_status
    }

    /// Run an explicit SQLite WAL checkpoint and return SQLite's raw counters.
    ///
    /// `Passive` is safe for live status/recovery workflows. `Truncate` should
    /// be used only after application writers have stopped (for example, at a
    /// quiesced backup or graceful-shutdown boundary); it is not what makes an
    /// already-acknowledged hardened commit durable.
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

    pub fn checkpoint_wal(&self, mode: WalCheckpointMode) -> Result<WalCheckpointStatus> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        Self::checkpoint_connection(&writer, mode)
    }

    /// Re-run integrity and V21 consistency checks against the live database.
    pub fn verify_integrity(&self) -> Result<SqliteIntegrityStatus> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        Self::run_integrity_checks(&reader, "runtime")
    }

    /// Data directory containing the SQLite database and sidecar folders.
    pub fn data_dir(&self) -> &Path {
        self.db_path.parent().unwrap_or_else(|| Path::new("."))
    }

    /// Sidecar directory for files belonging to this storage instance.
    pub fn sidecar_dir(&self, name: &str) -> PathBuf {
        self.data_dir().join(name)
    }

    /// Return the profile-scoped HNSW sidecar location. The profile ID is
    /// validated before being placed in a path, preventing traversal through a
    /// manifest or CLI argument.
    pub fn embedding_profile_index_dir(&self, profile_id: &EmbeddingProfileId) -> Result<PathBuf> {
        EmbeddingProfileId::new(profile_id.as_str().to_string())
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        Ok(self
            .sidecar_dir("embedding-profiles")
            .join(profile_id.as_str())
            .join("hnsw"))
    }

    /// Load existing embeddings into vector index
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn load_embeddings_into_index(&self) -> Result<()> {
        let Some(index) = self.vector_index.as_ref() else {
            return Ok(());
        };
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let active_profile_id = Self::active_profile_id_from_conn(&reader)?
            .unwrap_or_else(|| LEGACY_EMBEDDING_PROFILE_ID.to_string());
        drop(reader);
        let (rebuilt, journal_seq) = self.build_embedding_profile_index(&active_profile_id)?;
        {
            let mut index = index
                .lock()
                .map_err(|_| StorageError::Init("Vector index lock poisoned".to_string()))?;
            *index = rebuilt;
        }
        self.reset_vector_index_watermark(journal_seq);
        Ok(())
    }

    /// Build an isolated exact-dimension HNSW index without touching the live
    /// index. Activation uses this preflight so an invalid destination can
    /// never become the visible database pointer.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn build_embedding_profile_index(&self, profile_id: &str) -> Result<(VectorIndex, i64)> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        // One read snapshot for the rows AND the journal head, so the watermark
        // handed back describes exactly the rows that went into this index (#181).
        let snapshot = begin_read_snapshot(&reader)?;
        let profile_dimension: usize = snapshot
            .query_row(
                "SELECT embedding_dimension FROM embedding_profiles WHERE profile_id = ?1",
                params![profile_id],
                |row| row.get::<_, i64>(0),
            )?
            .try_into()
            .map_err(|_| {
                StorageError::InvalidEmbeddingProfile(format!(
                    "profile '{}' has an invalid embedding dimension",
                    profile_id
                ))
            })?;
        let mut stmt = snapshot.prepare(
            "SELECT node_id, embedding, model
             FROM embedding_profile_vectors
             WHERE profile_id = ?1",
        )?;

        // Never drop rows silently: a row this rebuild cannot read is a memory
        // that stays invisible to semantic search until it is re-embedded, and
        // the operator must be able to see that in the log.
        let mut unreadable_rows = 0_usize;
        let embeddings: Vec<(String, Vec<u8>, String)> = stmt
            .query_map(params![profile_id], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .filter_map(|r| match r {
                Ok(row) => Some(row),
                Err(error) => {
                    unreadable_rows += 1;
                    tracing::warn!(
                        %error,
                        profile_id,
                        "Skipping an unreadable embedding_profile_vectors row during vector index rebuild"
                    );
                    None
                }
            })
            .collect();
        if unreadable_rows > 0 {
            tracing::warn!(
                unreadable_rows,
                profile_id,
                "Vector index rebuild skipped rows; those memories stay keyword-searchable only until re-embedded"
            );
        }

        drop(stmt);
        let journal_seq: i64 = snapshot.query_row(
            "SELECT COALESCE(MAX(seq), 0) FROM vector_journal",
            [],
            |row| row.get(0),
        )?;
        drop(snapshot);
        drop(reader);

        // An index is a profile-scoped structure. In particular, never
        // Matryoshka-truncate a 1024/native profile into the legacy 256d index.
        let mut index = VectorIndex::with_config(VectorIndexConfig {
            dimensions: profile_dimension,
            ..VectorIndexConfig::default()
        })
        .map_err(|e| {
            StorageError::Init(format!("Failed to rebuild vector index before load: {}", e))
        })?;

        for (node_id, embedding_bytes, _model_name) in embeddings {
            let embedding = Embedding::from_bytes(&embedding_bytes).ok_or_else(|| {
                StorageError::InvalidEmbeddingProfile(format!(
                    "profile '{}' contains unreadable vector '{}'",
                    profile_id, node_id
                ))
            })?;
            if embedding.dimensions != profile_dimension {
                return Err(StorageError::InvalidEmbeddingProfile(format!(
                    "profile '{}' declares {} dimensions but vector '{}' has {}",
                    profile_id, profile_dimension, node_id, embedding.dimensions
                )));
            }
            index.add(&node_id, &embedding.vector).map_err(|error| {
                StorageError::InvalidEmbeddingProfile(format!(
                    "profile '{}' failed to build index for vector '{}': {}",
                    profile_id, node_id, error
                ))
            })?;
        }
        Ok((index, journal_seq))
    }

    fn secret_findings_for_input(input: &IngestInput) -> Vec<SecretFinding> {
        let mut findings = scan_secrets(&input.content);
        let mut scan_field = |value: &str| {
            for finding in scan_secrets(value) {
                if !findings.contains(&finding) {
                    findings.push(finding);
                }
            }
        };

        if let Some(source) = input.source.as_deref() {
            scan_field(source);
        }
        for tag in &input.tags {
            scan_field(tag);
        }
        if let Some(envelope) = input.source_envelope.as_ref() {
            for value in [
                envelope.source_url.as_deref(),
                envelope.source_project.as_deref(),
                envelope.source_type.as_deref(),
                envelope.source_author.as_deref(),
            ]
            .into_iter()
            .flatten()
            {
                scan_field(value);
            }
        }
        findings
    }

    fn enforce_secret_policy_for_input(input: &IngestInput, policy: SecretPolicy) -> Result<()> {
        if policy == SecretPolicy::AllowExplicitly {
            return Ok(());
        }

        let kinds: Vec<String> = Self::secret_findings_for_input(input)
            .into_iter()
            .filter(SecretFinding::blocks_ingestion)
            .map(|finding| finding.kind.as_str().to_string())
            .collect();
        if kinds.is_empty() {
            Ok(())
        } else {
            Err(StorageError::SecretDetected { kinds })
        }
    }

    fn enforce_secret_policy_for_content(content: &str, policy: SecretPolicy) -> Result<()> {
        if policy == SecretPolicy::AllowExplicitly {
            return Ok(());
        }
        let kinds: Vec<String> = scan_secrets(content)
            .into_iter()
            .filter(SecretFinding::blocks_ingestion)
            .map(|finding| finding.kind.as_str().to_string())
            .collect();
        if kinds.is_empty() {
            Ok(())
        } else {
            Err(StorageError::SecretDetected { kinds })
        }
    }

    /// Normalize a caller-provided project namespace before it reaches storage.
    /// Namespaces are identifiers, not user content: blank, oversized, and
    /// control-character values make audit and operator tooling ambiguous.
    fn normalize_scope(scope: &str) -> Result<&str> {
        let normalized = scope.trim();
        if normalized.is_empty()
            || normalized.len() > 200
            || normalized.chars().any(char::is_control)
        {
            return Err(StorageError::InvalidScope(
                "expected a non-empty identifier of at most 200 visible characters".to_string(),
            ));
        }
        Ok(normalized)
    }

    fn enforce_secret_policy_for_record(
        record: &crate::storage::memory_store::MemoryRecord,
        policy: SecretPolicy,
    ) -> Result<()> {
        if policy == SecretPolicy::AllowExplicitly {
            return Ok(());
        }

        // `MemoryStoreSend::insert` persists this selected set of record
        // fields directly. Keep it on the same default-deny policy as
        // `IngestInput`; otherwise a credential-shaped tag or source bypasses
        // the public ingest choke point.
        let mut findings = scan_secrets(&record.content);
        let mut scan_field = |value: &str| {
            for finding in scan_secrets(value) {
                if !findings.contains(&finding) {
                    findings.push(finding);
                }
            }
        };
        scan_field(&record.node_type);
        for tag in &record.tags {
            scan_field(tag);
        }
        for domain in &record.domains {
            scan_field(domain);
        }
        if let Some(source) = record
            .metadata
            .get("source")
            .and_then(|value| value.as_str())
        {
            scan_field(source);
        }

        let kinds: Vec<String> = findings
            .into_iter()
            .filter(SecretFinding::blocks_ingestion)
            .map(|finding| finding.kind.as_str().to_string())
            .collect();
        if kinds.is_empty() {
            Ok(())
        } else {
            Err(StorageError::SecretDetected { kinds })
        }
    }

    fn enforce_secret_policy_for_portable_archive(
        archive: &PortableArchive,
        policy: SecretPolicy,
    ) -> Result<()> {
        if policy == SecretPolicy::AllowExplicitly {
            return Ok(());
        }

        let mut kinds = Vec::new();
        for table in archive
            .tables
            .iter()
            .filter(|table| table.name == "knowledge_nodes")
        {
            for field in [
                "content",
                "source",
                "tags",
                "source_url",
                "source_project",
                "source_type",
                "source_author",
            ] {
                let Some(index) = table.columns.iter().position(|column| column == field) else {
                    continue;
                };
                for row in &table.rows {
                    let Some(PortableValue::Text(value)) = row.get(index) else {
                        continue;
                    };
                    for finding in scan_secrets(value)
                        .into_iter()
                        .filter(SecretFinding::blocks_ingestion)
                    {
                        let kind = finding.kind.as_str().to_string();
                        if !kinds.contains(&kind) {
                            kinds.push(kind);
                        }
                    }
                }
            }
        }

        if kinds.is_empty() {
            Ok(())
        } else {
            Err(StorageError::SecretDetected { kinds })
        }
    }

    /// Ingest a new memory, rejecting likely credentials by default.
    pub fn ingest(&self, input: IngestInput) -> Result<KnowledgeNode> {
        self.ingest_in_scope_with_secret_policy(input, DEFAULT_MEMORY_SCOPE, SecretPolicy::Reject)
    }

    /// Ingest a new memory using an explicit credential-storage policy.
    ///
    /// Callers should use [`SecretPolicy::AllowExplicitly`] only for a direct,
    /// intentional user action. Connector and background writers must retain
    /// the default rejection policy.
    pub fn ingest_with_secret_policy(
        &self,
        input: IngestInput,
        policy: SecretPolicy,
    ) -> Result<KnowledgeNode> {
        self.ingest_in_scope_with_secret_policy(input, DEFAULT_MEMORY_SCOPE, policy)
    }

    /// Ingest a memory into a named project namespace.
    pub fn ingest_in_scope(&self, input: IngestInput, scope: &str) -> Result<KnowledgeNode> {
        self.ingest_in_scope_with_secret_policy(input, scope, SecretPolicy::Reject)
    }

    /// Ingest a memory into a named project namespace with an explicit secret policy.
    pub fn ingest_in_scope_with_secret_policy(
        &self,
        input: IngestInput,
        scope: &str,
        policy: SecretPolicy,
    ) -> Result<KnowledgeNode> {
        Self::enforce_secret_policy_for_input(&input, policy)?;
        self.ingest_unchecked_in_scope(input, Self::normalize_scope(scope)?)
    }

    /// Raw scoped insert after a caller has completed the credential preflight.
    fn ingest_unchecked_in_scope(&self, input: IngestInput, scope: &str) -> Result<KnowledgeNode> {
        let now = Utc::now();
        let id = Uuid::new_v4().to_string();

        let fsrs_state = self
            .scheduler
            .lock()
            .map_err(|_| StorageError::Init("Scheduler lock poisoned".into()))?
            .new_card();

        // Sentiment boost for stability
        let sentiment_boost = if input.sentiment_magnitude > 0.0 {
            1.0 + (input.sentiment_magnitude * 0.5)
        } else {
            1.0
        };

        let tags_json = serde_json::to_string(&input.tags).unwrap_or_else(|_| "[]".to_string());
        let next_review = now + Duration::days(fsrs_state.scheduled_days as i64);
        let valid_from_str = input.valid_from.map(|dt| dt.to_rfc3339());
        let valid_until_str = input.valid_until.map(|dt| dt.to_rfc3339());

        // #57 Source envelope — flatten to nullable column values. A node with
        // no external provenance leaves all nine columns NULL (legacy shape).
        let env = input.source_envelope.clone().unwrap_or_default();
        let env_source_updated_at = env.source_updated_at.map(|dt| dt.to_rfc3339());
        let env_synced_at = env.synced_at.map(|dt| dt.to_rfc3339());

        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            writer.execute(
                "INSERT INTO knowledge_nodes (
                    id, content, node_type, created_at, updated_at, last_accessed,
                    stability, difficulty, reps, lapses, learning_state,
                    storage_strength, retrieval_strength, retention_strength,
                    sentiment_score, sentiment_magnitude, next_review, scheduled_days,
                    source, tags, valid_from, valid_until, has_embedding, embedding_model,
                    domains, domain_scores,
                    scope, source_system, source_id, source_url, source_updated_at,
                    content_hash, synced_at, source_project, source_type, source_author
                ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6,
                    ?7, ?8, ?9, ?10, ?11,
                    ?12, ?13, ?14,
                    ?15, ?16, ?17, ?18,
                    ?19, ?20, ?21, ?22, ?23, ?24,
                    '[]', '{}',
                    ?25, ?26, ?27, ?28, ?29,
                    ?30, ?31, ?32, ?33, ?34
                )",
                params![
                    id,
                    input.content,
                    input.node_type,
                    now.to_rfc3339(),
                    now.to_rfc3339(),
                    now.to_rfc3339(),
                    // Clamp to MAX_STABILITY: the sentiment boost is otherwise
                    // persisted unbounded, letting an emotional memory's stability
                    // exceed the FSRS-6 ceiling every other write path respects.
                    (fsrs_state.stability * sentiment_boost).min(MAX_STABILITY),
                    fsrs_state.difficulty,
                    fsrs_state.reps,
                    fsrs_state.lapses,
                    "new",
                    1.0,
                    1.0,
                    1.0,
                    input.sentiment_score,
                    input.sentiment_magnitude,
                    next_review.to_rfc3339(),
                    fsrs_state.scheduled_days,
                    input.source,
                    tags_json,
                    valid_from_str,
                    valid_until_str,
                    0,
                    Option::<String>::None,
                    scope,
                    env.source_system,
                    env.source_id,
                    env.source_url,
                    env_source_updated_at,
                    env.content_hash,
                    env_synced_at,
                    env.source_project,
                    env.source_type,
                    env.source_author,
                ],
            )?;
        }

        // Generate embedding if available
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        if let Err(e) = self.generate_embedding_for_node(&id, &input.content) {
            tracing::warn!("Failed to generate embedding for {}: {}", id, e);
        }

        self.get_node(&id)?
            .ok_or_else(|| StorageError::NotFound(id))
    }

    /// Smart ingest with Prediction Error Gating
    ///
    /// Uses neuroscience-inspired prediction error to decide whether to:
    /// - Create a new memory (high prediction error)
    /// - Update an existing memory (low prediction error)
    /// - Supersede a demoted/outdated memory (correction)
    ///
    /// This solves the "bad vs good similar memory" problem.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn smart_ingest(&self, input: IngestInput) -> Result<SmartIngestResult> {
        self.smart_ingest_in_scope_with_secret_policy(
            input,
            DEFAULT_MEMORY_SCOPE,
            SecretPolicy::Reject,
        )
    }

    /// Smart-ingest a memory while considering candidates only from the same namespace.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn smart_ingest_in_scope(
        &self,
        input: IngestInput,
        scope: &str,
    ) -> Result<SmartIngestResult> {
        self.smart_ingest_in_scope_with_secret_policy(input, scope, SecretPolicy::Reject)
    }

    /// Smart ingest with an explicit credential-storage policy.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn smart_ingest_with_secret_policy(
        &self,
        input: IngestInput,
        policy: SecretPolicy,
    ) -> Result<SmartIngestResult> {
        self.smart_ingest_in_scope_with_secret_policy(input, DEFAULT_MEMORY_SCOPE, policy)
    }

    /// Smart-ingest a memory into a named project namespace with an explicit secret policy.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn smart_ingest_in_scope_with_secret_policy(
        &self,
        input: IngestInput,
        scope: &str,
        policy: SecretPolicy,
    ) -> Result<SmartIngestResult> {
        self.smart_ingest_excluding_in_scope_with_secret_policy(input, scope, &[], policy)
    }

    /// Smart ingest with caller-provided candidate exclusions.
    ///
    /// Batch callers use this to keep two new items from the same caller-curated
    /// batch from merging into each other while still allowing smart updates
    /// against memories that existed before the batch began.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn smart_ingest_excluding(
        &self,
        input: IngestInput,
        excluded_node_ids: &[String],
    ) -> Result<SmartIngestResult> {
        self.smart_ingest_excluding_in_scope_with_secret_policy(
            input,
            DEFAULT_MEMORY_SCOPE,
            excluded_node_ids,
            SecretPolicy::Reject,
        )
    }

    /// Smart ingest with exclusions and an explicit credential-storage policy.
    /// The credential preflight happens before embedding, candidate selection,
    /// or any possible supersede/demotion side effect.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn smart_ingest_excluding_with_secret_policy(
        &self,
        input: IngestInput,
        excluded_node_ids: &[String],
        policy: SecretPolicy,
    ) -> Result<SmartIngestResult> {
        self.smart_ingest_excluding_in_scope_with_secret_policy(
            input,
            DEFAULT_MEMORY_SCOPE,
            excluded_node_ids,
            policy,
        )
    }

    /// Scoped smart-ingest with candidate exclusions and an explicit secret policy.
    /// Candidate selection is scope-bound before the prediction-error gate runs,
    /// preventing similarly-worded memories in another project from merging.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn smart_ingest_excluding_in_scope_with_secret_policy(
        &self,
        input: IngestInput,
        scope: &str,
        excluded_node_ids: &[String],
        policy: SecretPolicy,
    ) -> Result<SmartIngestResult> {
        use crate::advanced::prediction_error::{
            CandidateMemory, GateDecision, PredictionErrorGate, UpdateType,
        };

        Self::enforce_secret_policy_for_input(&input, policy)?;
        let scope = Self::normalize_scope(scope)?;

        // Generate embedding for new content
        if !self.active_embedding_runtime_ready()? {
            return self.regular_ingest_result(
                input,
                scope,
                "Embeddings not available, falling back to regular ingest",
                policy,
            );
        }

        if !self.vector_search_available() {
            return self.regular_ingest_result(
                input,
                scope,
                "Vector search unavailable, falling back to regular ingest",
                policy,
            );
        }

        // The prediction gate compares a candidate *document* with stored
        // document vectors. Qwen's retrieval profile intentionally uses a
        // different query template, so using get_query_embedding here would
        // silently compare different encoded spaces.
        let new_embedding = self.get_document_embedding(&input.content)?;

        // Find similar memories using semantic search
        let similar = self.semantic_search_raw(&input.content, 10)?;

        // Build candidate memories
        let mut candidates: Vec<CandidateMemory> = Vec::new();
        // The earliest currently-valid similar fact starting AFTER the
        // incoming dated claim. When only such newer facts exclude every
        // candidate, the incoming claim is a stale snapshot whose world time
        // is already known to end where the newer fact begins.
        let mut superseding_valid_from: Option<DateTime<Utc>> = None;
        for (node_id, _similarity) in similar.iter() {
            if excluded_node_ids.iter().any(|id| id == node_id) {
                continue;
            }
            if !self.node_is_in_scope(node_id, scope)? {
                continue;
            }
            if let Some(node) = self.get_node(node_id)? {
                // A historical snapshot must never mutate, reinforce, or demote
                // a fact whose validity starts later. Likewise, an already
                // expired input cannot supersede a currently-valid policy.
                if !temporal_candidate_is_eligible(
                    input.valid_from,
                    input.valid_until,
                    node.valid_from,
                    node.is_currently_valid(),
                    Utc::now(),
                ) {
                    if let (Some(incoming), Some(existing)) = (input.valid_from, node.valid_from)
                        && incoming < existing
                        && node.is_currently_valid()
                        && superseding_valid_from.is_none_or(|earliest| existing < earliest)
                    {
                        superseding_valid_from = Some(existing);
                    }
                    continue;
                }
                // Get embedding for this node
                if let Some(emb) = self.get_node_embedding(node_id)? {
                    // Check if this memory was previously demoted (low retrieval strength)
                    let was_demoted = node.retrieval_strength < 0.3;
                    let was_promoted = node.retrieval_strength > 0.85;

                    candidates.push(CandidateMemory {
                        id: node.id.clone(),
                        content: node.content.clone(),
                        embedding: emb,
                        retrieval_strength: node.retrieval_strength,
                        retention_strength: node.retention_strength,
                        tags: node.tags.clone(),
                        source: node.source.clone(),
                        was_demoted,
                        was_promoted,
                    });
                }
            }
        }

        // Evaluate with prediction error gate
        let mut gate = PredictionErrorGate::new();
        let decision = gate.evaluate(&input.content, &new_embedding, &candidates);

        match decision {
            GateDecision::Create {
                prediction_error,
                related_memory_ids,
                reason,
                ..
            } => {
                // A dated claim (explicit or inferred) that lost every
                // candidate to a currently-valid newer fact is created as a
                // closed historical snapshot, never as an open current fact.
                // Undated content and explicitly bounded claims are untouched.
                // `candidates.is_empty()` enforces the precondition this comment
                // already states. `superseding_valid_from` is recorded while
                // skipping INELIGIBLE candidates, so it can be set even when other
                // nodes were eligible, went into `candidates`, and the gate chose
                // Create on its own merits. Without this check an unrelated newer
                // fact elsewhere in the store stamps a brand-new memory as already
                // expired at creation -- it is born invisible to ordinary recall.
                let auto_closed_until = superseding_valid_from.filter(|_| {
                    candidates.is_empty()
                        && input.valid_from.is_some()
                        && input.valid_until.is_none()
                });
                // Create new memory
                let mut node = self.ingest_in_scope_with_secret_policy(input, scope, policy)?;
                if let Some(closes_at) = auto_closed_until {
                    self.close_node_validity(&node.id, closes_at)?;
                    let id = node.id.clone();
                    node = self.get_node(&id)?.ok_or(StorageError::NotFound(id))?;
                }
                let mut reason = if related_memory_ids.is_empty() {
                    format!("Created new memory: {:?}", reason)
                } else {
                    format!(
                        "Created new memory: {:?}. Semantically similar (not linked): {:?}",
                        reason, related_memory_ids
                    )
                };
                if let Some(closes_at) = auto_closed_until {
                    reason.push_str(&format!(
                        ". Closed validity at {} because a currently-valid newer fact starts then",
                        closes_at.to_rfc3339()
                    ));
                }
                Ok(SmartIngestResult {
                    decision: "create".to_string(),
                    node,
                    superseded_id: None,
                    similarity: None,
                    prediction_error: Some(prediction_error),
                    reason,
                    previous_content: None,
                    merged_from: None,
                    merge_preview: None,
                    auto_closed_until,
                })
            }
            GateDecision::Update {
                target_id,
                similarity,
                update_type,
                prediction_error,
            } => {
                // A prose-inferred "as of" date describes the incoming text
                // only; it may stamp a NEW node but must never rewrite an
                // existing node's window. Only explicit caller validity may.
                let explicit_valid_from = input.valid_from.filter(|_| !input.validity_inferred);
                match update_type {
                    UpdateType::Reinforce => {
                        if explicit_valid_from.is_some() || input.valid_until.is_some() {
                            self.update_node_validity(
                                &target_id,
                                explicit_valid_from,
                                input.valid_until,
                            )?;
                        }
                        // Just strengthen the existing memory
                        self.strengthen_on_access(&target_id)?;
                        let node = self
                            .get_node(&target_id)?
                            .ok_or_else(|| StorageError::NotFound(target_id.clone()))?;
                        Ok(SmartIngestResult {
                            decision: "reinforce".to_string(),
                            node,
                            superseded_id: None,
                            similarity: Some(similarity),
                            prediction_error: Some(prediction_error),
                            reason: "Content nearly identical - reinforced existing memory"
                                .to_string(),
                            previous_content: None,
                            merged_from: None,
                            merge_preview: None,
                            auto_closed_until: None,
                        })
                    }
                    UpdateType::Merge | UpdateType::Append => {
                        // Update the existing memory with merged content
                        let existing = self
                            .get_node(&target_id)?
                            .ok_or_else(|| StorageError::NotFound(target_id.clone()))?;
                        let previous_content = existing.content.clone();

                        let merged_content = format!(
                            "{}\n\n[Updated {}]\n{}",
                            previous_content,
                            chrono::Utc::now().format("%Y-%m-%d"),
                            input.content
                        );

                        self.update_node_content_with_secret_policy(
                            &target_id,
                            &merged_content,
                            policy,
                        )?;
                        if explicit_valid_from.is_some() || input.valid_until.is_some() {
                            self.update_node_validity(
                                &target_id,
                                explicit_valid_from,
                                input.valid_until,
                            )?;
                        }
                        self.strengthen_on_access(&target_id)?;

                        let node = self
                            .get_node(&target_id)?
                            .ok_or_else(|| StorageError::NotFound(target_id.clone()))?;

                        Ok(SmartIngestResult {
                            decision: "update".to_string(),
                            node,
                            superseded_id: None,
                            similarity: Some(similarity),
                            prediction_error: Some(prediction_error),
                            reason: "Merged with existing similar memory".to_string(),
                            previous_content: Some(previous_content),
                            merged_from: Some(target_id),
                            merge_preview: Some(merged_content),
                            auto_closed_until: None,
                        })
                    }
                    UpdateType::Replace => {
                        // Replace content entirely
                        let existing = self
                            .get_node(&target_id)?
                            .ok_or_else(|| StorageError::NotFound(target_id.clone()))?;
                        let previous_content = existing.content;

                        self.update_node_content_with_secret_policy(
                            &target_id,
                            &input.content,
                            policy,
                        )?;
                        if explicit_valid_from.is_some() || input.valid_until.is_some() {
                            self.update_node_validity(
                                &target_id,
                                explicit_valid_from,
                                input.valid_until,
                            )?;
                        }
                        let node = self
                            .get_node(&target_id)?
                            .ok_or_else(|| StorageError::NotFound(target_id.clone()))?;

                        Ok(SmartIngestResult {
                            decision: "replace".to_string(),
                            node,
                            superseded_id: None,
                            similarity: Some(similarity),
                            prediction_error: Some(prediction_error),
                            reason: "Replaced existing memory with new content".to_string(),
                            previous_content: Some(previous_content),
                            merged_from: Some(target_id),
                            merge_preview: Some(input.content),
                            auto_closed_until: None,
                        })
                    }
                    UpdateType::AddContext => {
                        // Add as context without modifying main content
                        let existing = self
                            .get_node(&target_id)?
                            .ok_or_else(|| StorageError::NotFound(target_id.clone()))?;
                        let previous_content = existing.content.clone();

                        let merged_content =
                            format!("{}\n\n---\nContext: {}", previous_content, input.content);

                        self.update_node_content_with_secret_policy(
                            &target_id,
                            &merged_content,
                            policy,
                        )?;
                        if explicit_valid_from.is_some() || input.valid_until.is_some() {
                            self.update_node_validity(
                                &target_id,
                                explicit_valid_from,
                                input.valid_until,
                            )?;
                        }
                        let node = self
                            .get_node(&target_id)?
                            .ok_or_else(|| StorageError::NotFound(target_id.clone()))?;

                        Ok(SmartIngestResult {
                            decision: "add_context".to_string(),
                            node,
                            superseded_id: None,
                            similarity: Some(similarity),
                            prediction_error: Some(prediction_error),
                            reason: "Added new content as context to existing memory".to_string(),
                            previous_content: Some(previous_content),
                            merged_from: Some(target_id),
                            merge_preview: Some(merged_content),
                            auto_closed_until: None,
                        })
                    }
                }
            }
            GateDecision::Supersede {
                old_memory_id,
                similarity,
                supersede_reason,
                prediction_error,
            } => {
                // Close the old fact's world-time interval before demoting it.
                // An explicitly dated replacement takes effect at its declared
                // start; otherwise — including a prose-inferred "as of" date,
                // which must never backdate another node's expiry — the
                // supersession becomes effective now.
                self.close_node_validity(
                    &old_memory_id,
                    input
                        .valid_from
                        .filter(|_| !input.validity_inferred)
                        .unwrap_or_else(Utc::now),
                )?;
                self.demote_memory(&old_memory_id)?;

                // Create the new improved memory
                let node = self.ingest_in_scope_with_secret_policy(input, scope, policy)?;

                Ok(SmartIngestResult {
                    decision: "supersede".to_string(),
                    node,
                    superseded_id: Some(old_memory_id),
                    similarity: Some(similarity),
                    prediction_error: Some(prediction_error),
                    reason: format!("New memory supersedes old: {:?}", supersede_reason),
                    previous_content: None,
                    merged_from: None,
                    merge_preview: None,
                    auto_closed_until: None,
                })
            }
            GateDecision::Merge {
                memory_ids,
                avg_similarity,
                strategy,
            } => {
                // For now, create new and link to existing
                let node = self.ingest_in_scope_with_secret_policy(input, scope, policy)?;

                Ok(SmartIngestResult {
                    decision: "merge".to_string(),
                    node,
                    superseded_id: None,
                    similarity: Some(avg_similarity),
                    prediction_error: Some(1.0 - avg_similarity),
                    reason: format!(
                        "Created new memory linked to {} similar memories ({:?})",
                        memory_ids.len(),
                        strategy
                    ),
                    previous_content: None,
                    merged_from: None,
                    merge_preview: None,
                    auto_closed_until: None,
                })
            }
        }
    }

    /// Get the embedding vector for a node
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn get_node_embedding(&self, node_id: &str) -> Result<Option<Vec<f32>>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let active_profile_id = Self::active_profile_id_from_conn(&reader)?
            .unwrap_or_else(|| LEGACY_EMBEDDING_PROFILE_ID.to_string());
        let mut stmt = reader.prepare(
            "SELECT embedding FROM embedding_profile_vectors
             WHERE profile_id = ?1 AND node_id = ?2",
        )?;

        let embedding_row: Option<Vec<u8>> = stmt
            .query_row(params![&active_profile_id, node_id], |row| row.get(0))
            .optional()?;

        // Direct table writes remain a supported test and migration fixture.
        // Only the legacy profile may consult that compatibility mirror; every
        // non-legacy profile is strictly isolated from it.
        let embedding_row =
            if embedding_row.is_none() && active_profile_id == LEGACY_EMBEDDING_PROFILE_ID {
                reader
                    .query_row(
                        "SELECT embedding FROM node_embeddings WHERE node_id = ?1",
                        params![node_id],
                        |row| row.get(0),
                    )
                    .optional()?
            } else {
                embedding_row
            };

        Ok(embedding_row
            .and_then(|bytes| Embedding::from_bytes(&bytes).map(|embedding| embedding.vector)))
    }

    /// Get all embedding vectors for duplicate detection
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn get_all_embeddings(&self) -> Result<Vec<(String, Vec<f32>)>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let active_profile_id = Self::active_profile_id_from_conn(&reader)?
            .unwrap_or_else(|| LEGACY_EMBEDDING_PROFILE_ID.to_string());
        let mut stmt = reader.prepare(
            "SELECT node_id, embedding FROM embedding_profile_vectors WHERE profile_id = ?1",
        )?;

        let mut unreadable_rows = 0_usize;
        let mut undecodable_rows = 0_usize;
        let mut results: Vec<(String, Vec<f32>)> = stmt
            .query_map(params![&active_profile_id], |row| {
                let node_id: String = row.get(0)?;
                let embedding_bytes: Vec<u8> = row.get(1)?;
                Ok((node_id, embedding_bytes))
            })?
            .filter_map(|r| match r {
                Ok(row) => Some(row),
                Err(error) => {
                    unreadable_rows += 1;
                    tracing::warn!(
                        %error,
                        profile_id = %active_profile_id,
                        "Skipping an unreadable embedding row while loading vectors"
                    );
                    None
                }
            })
            .filter_map(|(id, bytes)| match Embedding::from_bytes(&bytes) {
                Some(embedding) => Some((id, embedding.vector)),
                None => {
                    undecodable_rows += 1;
                    tracing::warn!(
                        node_id = %id,
                        profile_id = %active_profile_id,
                        "Skipping an undecodable embedding blob while loading vectors"
                    );
                    None
                }
            })
            .collect();
        // Keep direct writes to the historic table readable for the legacy
        // profile only. The anti-join avoids duplicate node ids after V20's
        // one-time copy, while non-legacy profiles never see this mirror.
        //
        // This mirror is the branch an unmigrated store actually takes, so it
        // gets the same "never drop a row silently" treatment as the profile
        // table above: a row skipped here is a memory that stays invisible to
        // semantic search, and the operator has to be able to see it.
        if active_profile_id == LEGACY_EMBEDDING_PROFILE_ID {
            drop(stmt);
            let mut legacy_stmt = reader.prepare(
                "SELECT ne.node_id, ne.embedding
                 FROM node_embeddings ne
                 WHERE NOT EXISTS (
                     SELECT 1 FROM embedding_profile_vectors pv
                     WHERE pv.profile_id = ?1 AND pv.node_id = ne.node_id
                 )",
            )?;
            results.extend(
                legacy_stmt
                    .query_map(params![LEGACY_EMBEDDING_PROFILE_ID], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
                    })?
                    .filter_map(|row| match row {
                        Ok(row) => Some(row),
                        Err(error) => {
                            unreadable_rows += 1;
                            tracing::warn!(
                                %error,
                                profile_id = %active_profile_id,
                                "Skipping an unreadable node_embeddings row while loading legacy vectors"
                            );
                            None
                        }
                    })
                    .filter_map(|(id, bytes)| match Embedding::from_bytes(&bytes) {
                        Some(embedding) => Some((id, embedding.vector)),
                        None => {
                            undecodable_rows += 1;
                            tracing::warn!(
                                node_id = %id,
                                profile_id = %active_profile_id,
                                "Skipping an undecodable node_embeddings blob while loading legacy vectors"
                            );
                            None
                        }
                    }),
            );
        }

        // Summarised after the legacy mirror so one line covers both sources.
        if unreadable_rows + undecodable_rows > 0 {
            tracing::warn!(
                unreadable_rows,
                undecodable_rows,
                profile_id = %active_profile_id,
                "Vector load skipped rows; those memories stay keyword-searchable only until re-embedded"
            );
        }

        Ok(results)
    }

    /// Fallback for builds without local embeddings/vector search.
    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    pub fn get_node_embedding(&self, _node_id: &str) -> Result<Option<Vec<f32>>> {
        Ok(None)
    }

    /// Update the content of an existing node, rejecting likely credentials by
    /// default.
    pub fn update_node_content(&self, id: &str, new_content: &str) -> Result<()> {
        self.update_node_content_with_secret_policy(id, new_content, SecretPolicy::Reject)
    }

    /// Update node content using an explicit credential-storage policy.
    pub fn update_node_content_with_secret_policy(
        &self,
        id: &str,
        new_content: &str,
        policy: SecretPolicy,
    ) -> Result<()> {
        Self::enforce_secret_policy_for_content(new_content, policy)?;
        self.update_node_content_unchecked(id, new_content)
    }

    /// Update a node's declared world-time interval without changing its
    /// project namespace or transaction-time history. A `None` bound keeps the
    /// stored column: updating only `valid_from` never clears an existing
    /// `valid_until`, and vice versa.
    pub fn update_node_validity(
        &self,
        id: &str,
        valid_from: Option<DateTime<Utc>>,
        valid_until: Option<DateTime<Utc>>,
    ) -> Result<()> {
        if let (Some(from), Some(until)) = (valid_from, valid_until)
            && until <= from
        {
            return Err(StorageError::InvalidTimestamp(
                "valid_until must be after valid_from".to_string(),
            ));
        }
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        // Validate the EFFECTIVE post-merge window under the writer lock so
        // a partial update cannot invert a window against the stored bound.
        let stored: Option<(Option<String>, Option<String>)> = writer
            .query_row(
                "SELECT valid_from, valid_until FROM knowledge_nodes WHERE id = ?1",
                params![id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if let Some((stored_from, stored_until)) = stored {
            let parse = |value: Option<String>| {
                value.and_then(|value| {
                    DateTime::parse_from_rfc3339(&value)
                        .map(|dt| dt.with_timezone(&Utc))
                        .ok()
                })
            };
            let effective_from = valid_from.or_else(|| parse(stored_from));
            let effective_until = valid_until.or_else(|| parse(stored_until));
            if let (Some(from), Some(until)) = (effective_from, effective_until)
                && until <= from
            {
                return Err(StorageError::InvalidTimestamp(
                    "valid_until must be after valid_from".to_string(),
                ));
            }
        }
        writer.execute(
            "UPDATE knowledge_nodes SET valid_from = COALESCE(?1, valid_from), valid_until = COALESCE(?2, valid_until), updated_at = ?3 WHERE id = ?4",
            params![
                valid_from.map(|value| value.to_rfc3339()),
                valid_until.map(|value| value.to_rfc3339()),
                Utc::now().to_rfc3339(),
                id
            ],
        )?;
        Ok(())
    }

    fn close_node_validity(&self, id: &str, valid_until: DateTime<Utc>) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "UPDATE knowledge_nodes SET valid_until = ?1, updated_at = ?2 WHERE id = ?3",
            params![valid_until.to_rfc3339(), Utc::now().to_rfc3339(), id],
        )?;
        Ok(())
    }

    fn update_node_content_unchecked(&self, id: &str, new_content: &str) -> Result<()> {
        let now = Utc::now();

        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            writer.execute(
                "UPDATE knowledge_nodes SET content = ?1, updated_at = ?2 WHERE id = ?3",
                params![new_content, now.to_rfc3339(), id],
            )?;
        }

        // Regenerate embedding for updated content
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        {
            // Remove old embedding from index
            if let Some(index) = self.vector_index.as_ref()
                && let Ok(mut index) = index.lock()
            {
                let _ = index.remove(id);
            }
            // Generate new embedding. If the embedder isn't ready yet (e.g. the
            // model is still downloading on first run), generate_embedding_for_node
            // is a no-op — which previously left the OLD, now-stale embedding row
            // with has_embedding = 1, so semantic search kept matching the old
            // content and the consolidation regeneration query (which only selects
            // has_embedding = 0 / missing rows / model mismatch) never refreshed
            // it. Flip has_embedding to 0 on the not-ready path so the stale vector
            // is picked up and rebuilt once the embedder comes online.
            if self.active_embedding_runtime_ready().unwrap_or(false) {
                if let Err(e) = self.generate_embedding_for_node(id, new_content) {
                    tracing::warn!("Failed to regenerate embedding for {}: {}", id, e);
                }
            } else if let Ok(writer) = self.writer.lock() {
                let _ = writer.execute(
                    "UPDATE knowledge_nodes SET has_embedding = 0 WHERE id = ?1",
                    params![id],
                );
            }
        }

        Ok(())
    }

    /// Generate embedding for a node
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn generate_embedding_for_node(&self, node_id: &str, content: &str) -> Result<()> {
        if !self.active_embedding_runtime_ready()? {
            return Ok(());
        }

        let active = self.active_embedding_profile()?.ok_or_else(|| {
            StorageError::InvalidEmbeddingProfile("no active embedding profile pointer".to_string())
        })?;
        let manifest = self
            .embedding_profile_manifest(&active.profile_id)?
            .ok_or_else(|| StorageError::NotFound(active.profile_id.to_string()))?;
        let encoded_content = manifest
            .profile
            .encode_document(content)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;

        let (embedding_bytes, embedding_dimensions, model_name, vector) =
            if let Some(embedder) = self.attached_embedder_for(&active.profile_id)? {
                let runtime = tokio::runtime::Runtime::new().map_err(|error| {
                    StorageError::Init(format!("Create local embedding runtime: {error}"))
                })?;
                let vector = runtime
                    .block_on(embedder.embed_document(content))
                    .map_err(|error| StorageError::Init(format!("Embedding failed: {error}")))?;
                let bytes = vector
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                (bytes, vector.len(), active.profile_id.to_string(), vector)
            } else {
                let embedding = self
                    .embedding_service
                    .embed(&encoded_content)
                    .map_err(|e| StorageError::Init(format!("Embedding failed: {e}")))?;
                (
                    embedding.to_bytes(),
                    embedding.dimensions,
                    self.embedding_service.model_name().to_string(),
                    embedding.vector,
                )
            };
        if embedding_dimensions != manifest.profile.embedding_dimension {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "active profile '{}' requires {} dimensions but its runtime produced {}",
                active.profile_id, manifest.profile.embedding_dimension, embedding_dimensions
            )));
        }

        self.persist_node_embedding(
            node_id,
            &embedding_bytes,
            embedding_dimensions,
            &model_name,
            &vector,
            active.profile_id.as_str() == LEGACY_EMBEDDING_PROFILE_ID,
        )
    }

    /// Write one node's vector everywhere semantic search reads it: the
    /// active profile's vector table (and the historic `node_embeddings`
    /// mirror while the legacy profile is active), the node's
    /// `has_embedding` flag, and the in-memory vector index. Shared by the
    /// embedder path and by [`MemoryStoreSend::insert`] for caller-supplied
    /// vectors, so no path can accept an embedding and drop it.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn persist_node_embedding(
        &self,
        node_id: &str,
        embedding_bytes: &[u8],
        embedding_dimensions: usize,
        model_name: &str,
        vector: &[f32],
        mirror_to_legacy_table: bool,
    ) -> Result<()> {
        let now = Utc::now();

        // One transaction for the three rows, with the journal head read inside
        // it. We hold the write lock, so no peer can commit between our INSERT
        // and that read: `journal_head` is the seq the trigger just appended.
        let journal_head: i64 = {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            let tx = Self::begin_write_transaction(&writer, "persist_node_embedding")?;
            if mirror_to_legacy_table {
                tx.execute(
                    "INSERT OR REPLACE INTO node_embeddings (node_id, embedding, dimensions, model, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        node_id,
                        embedding_bytes,
                        embedding_dimensions as i32,
                        model_name,
                        now.to_rfc3339(),
                    ],
                )?;
            }

            let active_profile_id = Self::active_profile_id_from_conn(&tx)?
                .unwrap_or_else(|| LEGACY_EMBEDDING_PROFILE_ID.to_string());
            tx.execute(
                "INSERT OR REPLACE INTO embedding_profile_vectors
                    (profile_id, node_id, embedding, dimensions, model, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    active_profile_id,
                    node_id,
                    embedding_bytes,
                    embedding_dimensions as i32,
                    model_name,
                    now.to_rfc3339(),
                ],
            )?;

            tx.execute(
                "UPDATE knowledge_nodes SET has_embedding = 1, embedding_model = ?2 WHERE id = ?1",
                params![node_id, model_name],
            )?;
            let head: i64 = tx.query_row(
                "SELECT COALESCE(MAX(seq), 0) FROM vector_journal",
                [],
                |row| row.get(0),
            )?;
            tx.commit()?;
            head
        };

        if let Some(index) = self.vector_index.as_ref() {
            let mut index = index
                .lock()
                .map_err(|_| StorageError::Init("Vector index lock poisoned".to_string()))?;
            index
                .add(node_id, vector)
                .map_err(|e| StorageError::Init(format!("Vector index add failed: {}", e)))?;
        }

        // Our own write bumps the reader's data_version exactly like a peer's
        // would, but the vector is already in the index. If ours is the only
        // journal row since the last refresh, absorb it now so the next search
        // does not re-add it. Anything else in between (a peer's row, an unknown
        // watermark) is left for the refresh, which re-adds ours harmlessly.
        if let Ok(mut watermark) = self.vector_index_watermark.lock()
            && watermark.journal_seq + 1 == journal_head
        {
            watermark.journal_seq = journal_head;
        }

        Ok(())
    }

    /// Index a caller-supplied embedding under the active profile, or refuse
    /// it loudly. Used by [`MemoryStoreSend::insert`]: a record that carries
    /// a vector is either searchable when the call returns or the call fails.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn index_supplied_embedding(
        &self,
        node_id: &str,
        vector: &[f32],
        model_name: Option<&str>,
    ) -> crate::storage::memory_store::MemoryStoreResult<()> {
        use crate::storage::memory_store::MemoryStoreError;
        let active = self
            .active_embedding_profile()
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?
            .ok_or_else(|| {
                MemoryStoreError::InvalidInput(
                    "record carries an embedding but the store has no active embedding profile to index it under"
                        .to_string(),
                )
            })?;
        let manifest = self
            .embedding_profile_manifest(&active.profile_id)
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?
            .ok_or_else(|| {
                MemoryStoreError::Backend(format!(
                    "active embedding profile '{}' has no manifest",
                    active.profile_id
                ))
            })?;
        if vector.len() != manifest.profile.embedding_dimension {
            return Err(MemoryStoreError::InvalidInput(format!(
                "embedding length {} != active profile '{}' dimension {}",
                vector.len(),
                active.profile_id,
                manifest.profile.embedding_dimension
            )));
        }
        let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
        let model = model_name
            .map(str::to_string)
            .unwrap_or_else(|| active.profile_id.to_string());
        self.persist_node_embedding(
            node_id,
            &bytes,
            vector.len(),
            &model,
            vector,
            active.profile_id.as_str() == LEGACY_EMBEDDING_PROFILE_ID,
        )
        .map_err(|e| MemoryStoreError::Backend(e.to_string()))
    }

    /// Read the active profile pointer from a caller-held connection. The
    /// pointer has one row and is changed in the same SQLite transaction as
    /// profile status, so readers can never observe a half-switch.
    fn active_profile_id_from_conn(conn: &Connection) -> Result<Option<String>> {
        conn.query_row(
            "SELECT active_profile_id FROM embedding_profile_state WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .optional()
        .map_err(StorageError::from)
    }

    fn profile_state_text(state: EmbeddingProfileState) -> Result<String> {
        serde_json::to_value(state)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| {
                StorageError::InvalidEmbeddingProfile(
                    "profile state must serialize to a string".to_string(),
                )
            })
    }

    fn migration_state_text(state: EmbeddingMigrationState) -> Result<String> {
        serde_json::to_value(state)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| {
                StorageError::InvalidEmbeddingProfile(
                    "migration state must serialize to a string".to_string(),
                )
            })
    }

    fn parse_rfc3339(value: String, field: &str) -> Result<DateTime<Utc>> {
        // V20's SQL bootstrap uses SQLite `datetime('now')` while subsequent
        // Rust writes use RFC3339. Reuse the store's tolerant parser so either
        // durable timestamp representation round-trips through profile APIs.
        Self::parse_timestamp(&value, field).map_err(StorageError::from)
    }

    fn ensure_legacy_embedding_profile_manifest(&self) -> Result<()> {
        // Normal opens must be completely idempotent. In particular, never
        // rewrite a preserved legacy profile to Active after an explicit Qwen
        // activation has moved the durable pointer elsewhere.
        let existing_manifest = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            reader
                .query_row(
                    "SELECT manifest_json FROM embedding_profile_manifests WHERE profile_id = ?1",
                    params![LEGACY_EMBEDDING_PROFILE_ID],
                    |row| row.get::<_, String>(0),
                )
                .optional()?
        };
        if existing_manifest
            .as_deref()
            .is_some_and(|json| serde_json::from_str::<EmbeddingProfileManifest>(json).is_ok())
        {
            return Ok(());
        }

        let mut manifest = EmbeddingProfileManifest::not_installed(
            BuiltinEmbeddingProfile::NomicLegacyRaw256.profile(),
        )
        .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let active_is_legacy = self
            .active_embedding_profile()?
            .is_none_or(|active| active.profile_id.as_str() == LEGACY_EMBEDDING_PROFILE_ID);
        manifest.state = if active_is_legacy {
            EmbeddingProfileState::Active
        } else {
            // This only upgrades V20's bootstrap '{}' placeholder. Existing
            // valid manifests already returned above, so no lifecycle receipt
            // or user choice is ever overwritten on reopen.
            EmbeddingProfileState::Ready
        };
        self.save_embedding_profile_manifest(&manifest)
    }

    /// Persist a full profile contract and its lifecycle receipt. This is a
    /// metadata operation only: saving an Installed manifest never downloads,
    /// migrates, or activates a model.
    pub fn save_embedding_profile_manifest(
        &self,
        manifest: &EmbeddingProfileManifest,
    ) -> Result<()> {
        manifest
            .validate()
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let profile = &manifest.profile;
        let manifest_json = serde_json::to_string(manifest)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let artifact_hashes = serde_json::to_string(&profile.verified_model_artifact_hashes)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let runtime = serde_json::to_string(&manifest.runtime)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let verification = serde_json::to_string(&manifest.verification)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let evaluation = serde_json::to_string(&manifest.evaluation)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let failure = serde_json::to_string(&manifest.failure)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let state = Self::profile_state_text(manifest.state)?;
        let now = Utc::now();
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "save_embedding_profile_manifest")?;
        let existing: Option<(String, i64)> = tx
            .query_row(
                "SELECT pm.manifest_json, pm.vector_count
                 FROM embedding_profile_manifests pm WHERE pm.profile_id = ?1",
                params![profile.profile_id.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if let Some((existing_json, vector_count)) = existing {
            // The V20 bootstrap row is deliberately replaced once with the
            // canonical legacy manifest. After that, a profile ID is an
            // immutable vector-space identity, not a mutable model selector.
            if let Ok(existing_manifest) =
                serde_json::from_str::<EmbeddingProfileManifest>(&existing_json)
                && vector_count > 0
                && existing_manifest.profile != manifest.profile
            {
                return Err(StorageError::InvalidEmbeddingProfile(format!(
                    "profile '{}' already owns {} vectors; changing its encoding contract requires a new profile ID",
                    profile.profile_id, vector_count
                )));
            }
        }
        if manifest.state == EmbeddingProfileState::Active {
            let pointer = Self::active_profile_id_from_conn(&tx)?;
            if pointer.as_deref() != Some(profile.profile_id.as_str()) {
                return Err(StorageError::InvalidEmbeddingProfile(format!(
                    "profile '{}' may become active only through activate_embedding_profile",
                    profile.profile_id
                )));
            }
        }
        tx.execute(
            "INSERT INTO embedding_profiles (
                profile_id, model_id, immutable_model_revision, verified_model_artifact_hashes,
                runtime_backend, embedding_dimension, normalization_method,
                document_encoding_template, query_encoding_template, maximum_token_limit,
                chunking_strategy, status, installed_at, last_verified_at, runtime_metadata,
                verification, evaluation, failure, created_at, updated_at
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20
             ) ON CONFLICT(profile_id) DO UPDATE SET
                model_id = excluded.model_id,
                immutable_model_revision = excluded.immutable_model_revision,
                verified_model_artifact_hashes = excluded.verified_model_artifact_hashes,
                runtime_backend = excluded.runtime_backend,
                embedding_dimension = excluded.embedding_dimension,
                normalization_method = excluded.normalization_method,
                document_encoding_template = excluded.document_encoding_template,
                query_encoding_template = excluded.query_encoding_template,
                maximum_token_limit = excluded.maximum_token_limit,
                chunking_strategy = excluded.chunking_strategy,
                status = excluded.status,
                installed_at = excluded.installed_at,
                last_verified_at = excluded.last_verified_at,
                runtime_metadata = excluded.runtime_metadata,
                verification = excluded.verification,
                evaluation = excluded.evaluation,
                failure = excluded.failure,
                updated_at = excluded.updated_at",
            params![
                profile.profile_id.as_str(),
                profile.model_id,
                profile.immutable_model_revision,
                artifact_hashes,
                serde_json::to_string(&profile.runtime_backend).unwrap_or_default(),
                profile.embedding_dimension as i64,
                serde_json::to_string(&profile.normalization_method).unwrap_or_default(),
                serde_json::to_string(&profile.document_encoding_template).unwrap_or_default(),
                serde_json::to_string(&profile.query_encoding_template).unwrap_or_default(),
                profile.maximum_token_limit as i64,
                serde_json::to_string(&profile.chunking_strategy).unwrap_or_default(),
                state,
                manifest.installed_at.map(|value| value.to_rfc3339()),
                manifest.last_verified_at.map(|value| value.to_rfc3339()),
                runtime,
                verification,
                evaluation,
                failure,
                profile.created_at.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )?;
        tx.execute(
            "INSERT INTO embedding_profile_manifests (
                profile_id, manifest_json, manifest_hash, vector_count, index_member_count, index_integrity_hash, updated_at
             ) VALUES (
                ?1, ?2, ?3,
                (SELECT COUNT(*) FROM embedding_profile_vectors WHERE profile_id = ?1),
                0, NULL, ?4
             ) ON CONFLICT(profile_id) DO UPDATE SET
                manifest_json = excluded.manifest_json,
                manifest_hash = excluded.manifest_hash,
                vector_count = (SELECT COUNT(*) FROM embedding_profile_vectors WHERE profile_id = excluded.profile_id),
                updated_at = excluded.updated_at",
            params![profile.profile_id.as_str(), manifest_json, manifest.manifest_hash(), now.to_rfc3339()],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Read a profile's complete persisted contract and lifecycle state.
    pub fn embedding_profile_manifest(
        &self,
        profile_id: &EmbeddingProfileId,
    ) -> Result<Option<EmbeddingProfileManifest>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let manifest: Option<String> = reader
            .query_row(
                "SELECT manifest_json FROM embedding_profile_manifests WHERE profile_id = ?1",
                params![profile_id.as_str()],
                |row| row.get(0),
            )
            .optional()?;
        manifest
            .map(|json| {
                serde_json::from_str(&json)
                    .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))
            })
            .transpose()
    }

    /// List known profiles without triggering any install/runtime work.
    pub fn list_embedding_profile_manifests(&self) -> Result<Vec<EmbeddingProfileManifest>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut statement = reader
            .prepare("SELECT manifest_json FROM embedding_profile_manifests ORDER BY profile_id")?;
        statement
            .query_map([], |row| row.get::<_, String>(0))?
            .map(|row| {
                let json = row?;
                serde_json::from_str(&json).map_err(|error| {
                    rusqlite::Error::FromSqlConversionFailure(0, Type::Text, Box::new(error))
                })
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(StorageError::from)
    }

    /// Read the active semantic-retrieval profile pointer.
    pub fn active_embedding_profile(&self) -> Result<Option<ActiveEmbeddingProfile>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let row: Option<(String, Option<String>, String)> = reader
            .query_row(
                "SELECT active_profile_id, previous_profile_id, activated_at
                 FROM embedding_profile_state WHERE singleton = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        row.map(|(profile_id, previous_profile_id, activated_at)| {
            Ok(ActiveEmbeddingProfile {
                profile_id: EmbeddingProfileId::new(profile_id)
                    .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?,
                previous_profile_id: previous_profile_id
                    .map(EmbeddingProfileId::new)
                    .transpose()
                    .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?,
                activated_at: Self::parse_rfc3339(activated_at, "profile activation timestamp")?,
            })
        })
        .transpose()
    }

    /// Change only the active-profile pointer after the caller has explicitly
    /// installed, evaluated, migrated, and validated the destination. The
    /// pointer and both status updates are one SQLite transaction; no vector
    /// rows are copied, removed, or re-embedded during activation.
    pub fn activate_embedding_profile(
        &self,
        profile_id: &EmbeddingProfileId,
    ) -> Result<ActiveEmbeddingProfile> {
        let target_state = Self::profile_state_text(EmbeddingProfileState::Ready)?;
        let active_state = Self::profile_state_text(EmbeddingProfileState::Active)?;
        let now = Utc::now();
        // Prebuild outside the write transaction. A malformed vector, mixed
        // dimensions, or unbuildable index fails here while the old pointer is
        // still live. Once the index lock is acquired, searches block until the
        // pointer transaction and in-memory index swap have both completed.
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        let rebuilt_index = if self.vector_index.is_some() {
            Some(self.build_embedding_profile_index(profile_id.as_str())?)
        } else {
            None
        };
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        let mut live_index = match self.vector_index.as_ref() {
            Some(index) => Some(
                index
                    .lock()
                    .map_err(|_| StorageError::Init("Vector index lock poisoned".to_string()))?,
            ),
            None => None,
        };
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "activate_embedding_profile")?;
        let current: Option<String> = tx
            .query_row(
                "SELECT active_profile_id FROM embedding_profile_state WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let stored_state: Option<String> = tx
            .query_row(
                "SELECT status FROM embedding_profiles WHERE profile_id = ?1",
                params![profile_id.as_str()],
                |row| row.get(0),
            )
            .optional()?;
        let Some(stored_state) = stored_state else {
            return Err(StorageError::NotFound(profile_id.to_string()));
        };
        let manifest_json: String = tx
            .query_row(
                "SELECT manifest_json FROM embedding_profile_manifests WHERE profile_id = ?1",
                params![profile_id.as_str()],
                |row| row.get(0),
            )
            .optional()?
            .ok_or_else(|| {
                StorageError::NotFound(format!("embedding profile manifest {profile_id}"))
            })?;
        let manifest: EmbeddingProfileManifest = serde_json::from_str(&manifest_json)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let legacy_rollback = profile_id.as_str() == LEGACY_EMBEDDING_PROFILE_ID
            && current.is_some()
            && current.as_deref() != Some(profile_id.as_str());
        if stored_state != target_state && stored_state != active_state {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "profile '{}' is '{}' and cannot be activated; only a validated ready profile may change live semantic retrieval",
                profile_id, stored_state
            )));
        }
        if current.as_deref() == Some(profile_id.as_str()) {
            tx.commit()?;
            return Ok(ActiveEmbeddingProfile {
                profile_id: profile_id.clone(),
                previous_profile_id: None,
                activated_at: now,
            });
        }
        if !legacy_rollback
            && (manifest.state != EmbeddingProfileState::Ready
                || manifest.verification.status != VerificationStatus::Verified
                || manifest.verification.verified_artifacts.is_empty()
                || manifest
                    .runtime
                    .as_ref()
                    .is_none_or(|runtime| !runtime.local_only)
                || manifest.evaluation.is_none())
        {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "profile '{}' must have a ready, locally verified runtime and completed evaluation before activation",
                profile_id
            )));
        }
        let completed_state = Self::migration_state_text(EmbeddingMigrationState::Completed)?;
        let completed_migration: i64 = tx.query_row(
            "SELECT COUNT(*) FROM embedding_profile_migrations
             WHERE destination_profile_id = ?1 AND state = ?2",
            params![profile_id.as_str(), completed_state],
            |row| row.get(0),
        )?;
        let integrity: Option<(i64, i64, String)> = tx
            .query_row(
                "SELECT vector_count, index_member_count, manifest_hash
                 FROM embedding_profile_manifests WHERE profile_id = ?1",
                params![profile_id.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        let Some((vector_count, index_member_count, manifest_hash)) = integrity else {
            return Err(StorageError::InvalidEmbeddingProfile(
                "missing profile integrity manifest".to_string(),
            ));
        };
        if !legacy_rollback
            && (completed_migration == 0
                || vector_count != index_member_count
                || manifest_hash != manifest.manifest_hash())
        {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "profile '{}' lacks a completed migration with a validated matching index manifest",
                profile_id
            )));
        }
        let wrong_dimension_vectors: i64 = tx.query_row(
            "SELECT COUNT(*) FROM embedding_profile_vectors
             WHERE profile_id = ?1 AND dimensions != ?2",
            params![
                profile_id.as_str(),
                manifest.profile.embedding_dimension as i64
            ],
            |row| row.get(0),
        )?;
        if wrong_dimension_vectors != 0 {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "profile '{}' has {} vectors incompatible with its declared dimension",
                profile_id, wrong_dimension_vectors
            )));
        }
        if let Some(current_id) = &current {
            // A prior active profile remains validated and rollback-ready. It
            // becomes Ready (not Inactive), so any future activation still
            // passes the exact same validation gate.
            tx.execute(
                "UPDATE embedding_profiles SET status = ?1, updated_at = ?2 WHERE profile_id = ?3",
                params![target_state, now.to_rfc3339(), current_id],
            )?;
        }
        tx.execute(
            "UPDATE embedding_profiles SET status = ?1, updated_at = ?2 WHERE profile_id = ?3",
            params![active_state, now.to_rfc3339(), profile_id.as_str()],
        )?;
        tx.execute(
            "INSERT INTO embedding_profile_state (
                singleton, active_profile_id, previous_profile_id, activated_at, updated_at
             ) VALUES (1, ?1, ?2, ?3, ?3)
             ON CONFLICT(singleton) DO UPDATE SET
                active_profile_id = excluded.active_profile_id,
                previous_profile_id = excluded.previous_profile_id,
                activated_at = excluded.activated_at,
                updated_at = excluded.updated_at",
            params![profile_id.as_str(), current, now.to_rfc3339()],
        )?;
        tx.commit()?;
        // The index lock blocks semantic search across the committed-pointer /
        // in-memory-index handoff. The replacement was built and fully checked
        // before the pointer was ever visible.
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        {
            let swapped_journal_seq = if let (Some(live_index), Some((rebuilt_index, journal_seq))) =
                (live_index.as_deref_mut(), rebuilt_index)
            {
                *live_index = rebuilt_index;
                Some(journal_seq)
            } else {
                None
            };
            // Release the index before touching the watermark: the refresh path
            // never holds both locks at once, so neither may this one.
            drop(live_index);
            if let Some(journal_seq) = swapped_journal_seq {
                self.reset_vector_index_watermark(journal_seq);
            }
        }
        Ok(ActiveEmbeddingProfile {
            profile_id: profile_id.clone(),
            previous_profile_id: current
                .map(EmbeddingProfileId::new)
                .transpose()
                .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?,
            activated_at: now,
        })
    }

    /// Instant rollback is exactly another explicit pointer change; the old
    /// profile's isolated vectors and sidecar remain intact.
    pub fn rollback_embedding_profile(
        &self,
        profile_id: &EmbeddingProfileId,
    ) -> Result<ActiveEmbeddingProfile> {
        self.activate_embedding_profile(profile_id)
    }

    /// Store a vector in one profile's private vector space. Dimension and
    /// profile identity are checked at write time, preventing a migration from
    /// accidentally contaminating its destination profile.
    pub fn put_embedding_profile_vector(&self, vector: &EmbeddingProfileVector) -> Result<()> {
        if vector.profile_id.trim().is_empty()
            || vector.node_id.trim().is_empty()
            || vector.dimensions == 0
        {
            return Err(StorageError::InvalidEmbeddingProfile(
                "profile vector requires a profile ID, node ID, and positive dimensions"
                    .to_string(),
            ));
        }
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "put_embedding_profile_vector")?;
        let declared_dimension: Option<i64> = tx
            .query_row(
                "SELECT embedding_dimension FROM embedding_profiles WHERE profile_id = ?1",
                params![&vector.profile_id],
                |row| row.get(0),
            )
            .optional()?;
        let Some(declared_dimension) = declared_dimension else {
            return Err(StorageError::NotFound(vector.profile_id.clone()));
        };
        if declared_dimension != i64::from(vector.dimensions) {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "profile '{}' declares {} dimensions but attempted vector has {}",
                vector.profile_id, declared_dimension, vector.dimensions
            )));
        }
        tx.execute(
            "INSERT INTO embedding_profile_vectors
                (profile_id, node_id, embedding, dimensions, model, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(profile_id, node_id) DO UPDATE SET
                embedding = excluded.embedding, dimensions = excluded.dimensions,
                model = excluded.model, created_at = excluded.created_at",
            params![
                &vector.profile_id,
                &vector.node_id,
                &vector.embedding,
                vector.dimensions as i64,
                &vector.model,
                vector.created_at.to_rfc3339(),
            ],
        )?;
        tx.execute(
            "UPDATE embedding_profile_manifests
             SET vector_count = (SELECT COUNT(*) FROM embedding_profile_vectors WHERE profile_id = ?1),
                 updated_at = ?2
             WHERE profile_id = ?1",
            params![&vector.profile_id, Utc::now().to_rfc3339()],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Read one profile-scoped vector. This never falls back to another profile.
    pub fn embedding_profile_vector(
        &self,
        profile_id: &EmbeddingProfileId,
        node_id: &str,
    ) -> Result<Option<EmbeddingProfileVector>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let row: Option<(Vec<u8>, i64, String, String)> = reader
            .query_row(
                "SELECT embedding, dimensions, model, created_at
                 FROM embedding_profile_vectors WHERE profile_id = ?1 AND node_id = ?2",
                params![profile_id.as_str(), node_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;
        row.map(|(embedding, dimensions, model, created_at)| {
            Ok(EmbeddingProfileVector {
                profile_id: profile_id.to_string(),
                node_id: node_id.to_string(),
                embedding,
                dimensions: dimensions.try_into().map_err(|_| {
                    StorageError::InvalidEmbeddingProfile("negative vector dimensions".to_string())
                })?,
                model,
                created_at: Self::parse_rfc3339(created_at, "profile vector timestamp")?,
            })
        })
        .transpose()
    }

    /// Record the validated vector/index membership for a profile. This is the
    /// final integrity receipt required before `activate_embedding_profile` can
    /// move live semantic retrieval to the profile.
    pub fn save_embedding_profile_integrity_manifest(
        &self,
        integrity: &EmbeddingProfileIntegrityManifest,
    ) -> Result<()> {
        let profile_id = EmbeddingProfileId::new(integrity.profile_id.clone())
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let manifest = self
            .embedding_profile_manifest(&profile_id)?
            .ok_or_else(|| StorageError::NotFound(profile_id.to_string()))?;
        if integrity.manifest_hash != manifest.manifest_hash() {
            return Err(StorageError::InvalidEmbeddingProfile(
                "integrity receipt hash does not match the persisted profile manifest".to_string(),
            ));
        }
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let actual_vector_count: i64 = writer.query_row(
            "SELECT COUNT(*) FROM embedding_profile_vectors WHERE profile_id = ?1",
            params![profile_id.as_str()],
            |row| row.get(0),
        )?;
        if integrity.vector_count != actual_vector_count as u64
            || integrity.index_member_count != integrity.vector_count
        {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "integrity receipt for '{}' does not match stored profile vectors",
                profile_id
            )));
        }
        writer.execute(
            "UPDATE embedding_profile_manifests SET
                manifest_json = manifest_json,
                manifest_hash = ?2,
                vector_count = ?3,
                index_member_count = ?4,
                index_integrity_hash = ?5,
                updated_at = ?6
             WHERE profile_id = ?1",
            params![
                profile_id.as_str(),
                &integrity.manifest_hash,
                integrity.vector_count as i64,
                integrity.index_member_count as i64,
                &integrity.index_integrity_hash,
                integrity.updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Persist (or resume) a migration checkpoint. The active-profile pointer
    /// is deliberately untouched; migration is not activation.
    pub fn save_profile_migration_checkpoint(
        &self,
        checkpoint: &ProfileMigrationCheckpoint,
    ) -> Result<()> {
        if checkpoint.source_profile_id == checkpoint.destination_profile_id {
            return Err(StorageError::InvalidEmbeddingProfile(
                "migration source and destination profiles must differ".to_string(),
            ));
        }
        if checkpoint.completed_memories > checkpoint.total_memories {
            return Err(StorageError::InvalidEmbeddingProfile(
                "migration completed memories cannot exceed total memories".to_string(),
            ));
        }
        let state = Self::migration_state_text(checkpoint.state)?;
        let failed_memory_ids = serde_json::to_string(&checkpoint.failed_memory_ids)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "save_profile_migration_checkpoint")?;
        let profiles: i64 = tx.query_row(
            "SELECT COUNT(*) FROM embedding_profiles WHERE profile_id IN (?1, ?2)",
            params![
                checkpoint.source_profile_id.as_str(),
                checkpoint.destination_profile_id.as_str()
            ],
            |row| row.get(0),
        )?;
        if profiles != 2 {
            return Err(StorageError::NotFound(
                "migration source or destination embedding profile".to_string(),
            ));
        }
        tx.execute(
            "INSERT INTO embedding_profile_migrations (
                migration_id, source_profile_id, destination_profile_id, state,
                total_memories, completed_memories, failed_memory_ids, last_memory_id,
                started_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(migration_id) DO UPDATE SET
                source_profile_id = excluded.source_profile_id,
                destination_profile_id = excluded.destination_profile_id,
                state = excluded.state,
                total_memories = excluded.total_memories,
                completed_memories = excluded.completed_memories,
                failed_memory_ids = excluded.failed_memory_ids,
                last_memory_id = excluded.last_memory_id,
                updated_at = excluded.updated_at",
            params![
                checkpoint.migration_id.to_string(),
                checkpoint.source_profile_id.as_str(),
                checkpoint.destination_profile_id.as_str(),
                state,
                checkpoint.total_memories as i64,
                checkpoint.completed_memories as i64,
                failed_memory_ids,
                checkpoint.last_memory_id.map(|value| value.to_string()),
                checkpoint.started_at.to_rfc3339(),
                checkpoint.updated_at.to_rfc3339(),
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Fetch a resumable migration checkpoint by immutable migration ID.
    pub fn profile_migration_checkpoint(
        &self,
        migration_id: Uuid,
    ) -> Result<Option<ProfileMigrationCheckpoint>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let row: Option<EmbeddingProfileMigrationRow> = reader
            .query_row(
                "SELECT source_profile_id, destination_profile_id, state, total_memories,
                        completed_memories, failed_memory_ids, last_memory_id, started_at, updated_at
                 FROM embedding_profile_migrations WHERE migration_id = ?1",
                params![migration_id.to_string()],
                |row| Ok((
                    row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?,
                    row.get(5)?, row.get(6)?, row.get(7)?, row.get(8)?,
                )),
            )
            .optional()?;
        row.map(
            |(source, destination, state, total, completed, failed, last, started, updated)| {
                Ok(ProfileMigrationCheckpoint {
                    migration_id,
                    source_profile_id: EmbeddingProfileId::new(source).map_err(|error| {
                        StorageError::InvalidEmbeddingProfile(error.to_string())
                    })?,
                    destination_profile_id: EmbeddingProfileId::new(destination).map_err(
                        |error| StorageError::InvalidEmbeddingProfile(error.to_string()),
                    )?,
                    state: serde_json::from_value(serde_json::Value::String(state)).map_err(
                        |error| StorageError::InvalidEmbeddingProfile(error.to_string()),
                    )?,
                    total_memories: total.try_into().map_err(|_| {
                        StorageError::InvalidEmbeddingProfile(
                            "negative migration total".to_string(),
                        )
                    })?,
                    completed_memories: completed.try_into().map_err(|_| {
                        StorageError::InvalidEmbeddingProfile(
                            "negative migration completed count".to_string(),
                        )
                    })?,
                    failed_memory_ids: serde_json::from_str(&failed).map_err(|error| {
                        StorageError::InvalidEmbeddingProfile(error.to_string())
                    })?,
                    last_memory_id: last
                        .map(|value| {
                            Uuid::parse_str(&value).map_err(|error| {
                                StorageError::InvalidEmbeddingProfile(error.to_string())
                            })
                        })
                        .transpose()?,
                    started_at: Self::parse_rfc3339(started, "migration start timestamp")?,
                    updated_at: Self::parse_rfc3339(updated, "migration update timestamp")?,
                })
            },
        )
        .transpose()
    }

    /// Upsert one durable work item for a migration repair/resume queue.
    pub fn save_embedding_profile_migration_node_checkpoint(
        &self,
        checkpoint: &EmbeddingProfileMigrationNodeCheckpoint,
    ) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "INSERT INTO embedding_profile_migration_checkpoints
                (migration_id, node_id, state, error, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(migration_id, node_id) DO UPDATE SET
                state = excluded.state, error = excluded.error, updated_at = excluded.updated_at",
            params![
                &checkpoint.migration_id,
                &checkpoint.node_id,
                &checkpoint.state,
                &checkpoint.error,
                checkpoint.updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Atomically persist one destination-profile vector and its durable
    /// per-node migration checkpoint. A crash therefore leaves either neither
    /// record or both records—never a vector that a resume cursor believes was
    /// not written (or vice versa).
    pub fn put_embedding_profile_vector_with_migration_checkpoint(
        &self,
        vector: &EmbeddingProfileVector,
        checkpoint: &EmbeddingProfileMigrationNodeCheckpoint,
    ) -> Result<()> {
        if vector.node_id != checkpoint.node_id {
            return Err(StorageError::InvalidEmbeddingProfile(
                "migration checkpoint node ID must match its vector node ID".to_string(),
            ));
        }
        if vector.dimensions == 0 {
            return Err(StorageError::InvalidEmbeddingProfile(
                "profile vector dimensions must be positive".to_string(),
            ));
        }
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "put_embedding_profile_vector_with_migration_checkpoint")?;
        let destination_profile: Option<String> = tx
            .query_row(
                "SELECT destination_profile_id FROM embedding_profile_migrations WHERE migration_id = ?1",
                params![&checkpoint.migration_id],
                |row| row.get(0),
            )
            .optional()?;
        if destination_profile.as_deref() != Some(vector.profile_id.as_str()) {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "migration '{}' does not target profile '{}'",
                checkpoint.migration_id, vector.profile_id
            )));
        }
        let declared_dimension: i64 = tx.query_row(
            "SELECT embedding_dimension FROM embedding_profiles WHERE profile_id = ?1",
            params![&vector.profile_id],
            |row| row.get(0),
        )?;
        if declared_dimension != i64::from(vector.dimensions) {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "profile '{}' declares {} dimensions but attempted vector has {}",
                vector.profile_id, declared_dimension, vector.dimensions
            )));
        }
        tx.execute(
            "INSERT INTO embedding_profile_vectors
                (profile_id, node_id, embedding, dimensions, model, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(profile_id, node_id) DO UPDATE SET
                embedding = excluded.embedding, dimensions = excluded.dimensions,
                model = excluded.model, created_at = excluded.created_at",
            params![
                &vector.profile_id,
                &vector.node_id,
                &vector.embedding,
                vector.dimensions as i64,
                &vector.model,
                vector.created_at.to_rfc3339(),
            ],
        )?;
        tx.execute(
            "INSERT INTO embedding_profile_migration_checkpoints
                (migration_id, node_id, state, error, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(migration_id, node_id) DO UPDATE SET
                state = excluded.state, error = excluded.error, updated_at = excluded.updated_at",
            params![
                &checkpoint.migration_id,
                &checkpoint.node_id,
                &checkpoint.state,
                &checkpoint.error,
                checkpoint.updated_at.to_rfc3339(),
            ],
        )?;
        tx.execute(
            "UPDATE embedding_profile_manifests
             SET vector_count = (SELECT COUNT(*) FROM embedding_profile_vectors WHERE profile_id = ?1),
                 updated_at = ?2
             WHERE profile_id = ?1",
            params![&vector.profile_id, Utc::now().to_rfc3339()],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Return the latest resumable migration checkpoint for a destination
    /// profile, ordered deterministically by update time and migration ID.
    pub fn latest_profile_migration_checkpoint_for_destination(
        &self,
        profile_id: &EmbeddingProfileId,
    ) -> Result<Option<ProfileMigrationCheckpoint>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let migration_id: Option<String> = reader
            .query_row(
                "SELECT migration_id FROM embedding_profile_migrations
                 WHERE destination_profile_id = ?1
                 ORDER BY updated_at DESC, migration_id DESC LIMIT 1",
                params![profile_id.as_str()],
                |row| row.get(0),
            )
            .optional()?;
        drop(reader);
        migration_id
            .map(|id| {
                Uuid::parse_str(&id)
                    .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))
                    .and_then(|id| {
                        self.profile_migration_checkpoint(id)?.ok_or_else(|| {
                            StorageError::NotFound(format!("migration checkpoint {id}"))
                        })
                    })
            })
            .transpose()
    }

    /// Persist a migration snapshot receipt without ever storing a private
    /// absolute path. The path is relative to `data_dir()` and the report must
    /// bind both the snapshot bytes and the stable corpus snapshot by SHA-256.
    pub fn save_profile_migration_snapshot_receipt(
        &self,
        migration_id: Uuid,
        relative_snapshot_path: &Path,
        validation_report: &serde_json::Value,
    ) -> Result<()> {
        if relative_snapshot_path.is_absolute()
            || relative_snapshot_path.as_os_str().is_empty()
            || relative_snapshot_path.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(StorageError::InvalidEmbeddingProfile(
                "migration snapshot path must be a non-empty relative path under the Vestige data directory".to_string(),
            ));
        }
        let required_sha256 = ["snapshot_sha256", "corpus_sha256"];
        for key in required_sha256 {
            let valid = validation_report
                .get(key)
                .and_then(serde_json::Value::as_str)
                .is_some_and(|value| {
                    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
                });
            if !valid {
                return Err(StorageError::InvalidEmbeddingProfile(format!(
                    "migration validation report requires a 64-character SHA-256 '{key}'"
                )));
            }
        }
        let path = relative_snapshot_path.to_str().ok_or_else(|| {
            StorageError::InvalidEmbeddingProfile("snapshot path must be UTF-8".to_string())
        })?;
        let report = serde_json::to_string(validation_report)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let changed = writer.execute(
            "UPDATE embedding_profile_migrations
             SET snapshot_path = ?1, validation_report = ?2, updated_at = ?3
             WHERE migration_id = ?4",
            params![
                path,
                report,
                Utc::now().to_rfc3339(),
                migration_id.to_string()
            ],
        )?;
        if changed != 1 {
            return Err(StorageError::NotFound(format!("migration {migration_id}")));
        }
        Ok(())
    }

    /// Get a node by ID
    pub fn get_node(&self, id: &str) -> Result<Option<KnowledgeNode>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare("SELECT * FROM knowledge_nodes WHERE id = ?1")?;

        let node = stmt.query_row(params![id], Self::row_to_node).optional()?;
        Ok(node)
    }

    /// Return whether a node belongs to a namespace. NULL and blank historic
    /// values are treated as `user`, matching V27's compatibility migration.
    pub fn node_is_in_scope(&self, id: &str, scope: &str) -> Result<bool> {
        let scope = Self::normalize_scope(scope)?;
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let present: Option<i32> = reader
            .query_row(
                "SELECT 1 FROM knowledge_nodes
                 WHERE id = ?1
                   AND COALESCE(NULLIF(trim(scope), ''), 'user') = ?2",
                params![id, scope],
                |row| row.get(0),
            )
            .optional()?;
        Ok(present.is_some())
    }

    /// Parse a stored timestamp into a UTC `DateTime`.
    ///
    /// The canonical on-disk format is RFC 3339 (every Rust writer in this
    /// crate uses `DateTime::to_rfc3339()`). However, timestamps can also be
    /// written by external tooling that bypasses this storage layer — most
    /// notably session hooks or manual maintenance that touch the DB with raw
    /// `sqlite3`. SQLite's native `datetime('now')` / `CURRENT_TIMESTAMP`
    /// emit a space-separated, timezone-less `YYYY-MM-DD HH:MM:SS[.fff]`
    /// string that `parse_from_rfc3339` rejects, which would otherwise make
    /// every affected row unreadable.
    ///
    /// We therefore parse RFC 3339 first and fall back to the SQLite-native
    /// format (assumed UTC) so the store stays tolerant of either writer.
    fn parse_timestamp(value: &str, field_name: &str) -> rusqlite::Result<DateTime<Utc>> {
        if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
            return Ok(dt.with_timezone(&Utc));
        }

        // Fallback: SQLite-native "YYYY-MM-DD HH:MM:SS" (with optional
        // fractional seconds), which has no timezone and is assumed UTC.
        if let Ok(naive) = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f") {
            return Ok(naive.and_utc());
        }

        Err(rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Text,
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Invalid {} timestamp '{}': not RFC 3339 or SQLite datetime format",
                    field_name, value
                ),
            )),
        ))
    }

    /// Convert a row to KnowledgeNode
    fn row_to_node(row: &rusqlite::Row) -> rusqlite::Result<KnowledgeNode> {
        let tags_json: String = row.get("tags")?;
        let tags: Vec<String> = match serde_json::from_str(&tags_json) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(raw = %tags_json, "Failed to deserialize tags JSON, using empty: {}", e);
                Vec::new()
            }
        };

        let created_at: String = row.get("created_at")?;
        let updated_at: String = row.get("updated_at")?;
        let last_accessed: String = row.get("last_accessed")?;
        let next_review: Option<String> = row.get("next_review")?;

        let created_at = Self::parse_timestamp(&created_at, "created_at")?;
        let updated_at = Self::parse_timestamp(&updated_at, "updated_at")?;
        let last_accessed = Self::parse_timestamp(&last_accessed, "last_accessed")?;

        let next_review = next_review.and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&Utc))
                .ok()
        });

        let valid_from: Option<String> = row.get("valid_from").ok().flatten();
        let valid_until: Option<String> = row.get("valid_until").ok().flatten();

        let valid_from = valid_from.and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&Utc))
                .ok()
        });

        let valid_until = valid_until.and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&Utc))
                .ok()
        });

        let has_embedding: Option<i32> = row.get("has_embedding").ok();
        let embedding_model: Option<String> = row.get("embedding_model").ok().flatten();

        // v2.0.5 Active Forgetting columns (Migration V10)
        let suppression_count: i32 = row.get("suppression_count").unwrap_or(0);
        let suppressed_at_str: Option<String> = row.get("suppressed_at").ok().flatten();
        let suppressed_at = suppressed_at_str.and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&Utc))
                .ok()
        });

        // #57 Source envelope columns (Migration V17). `.ok().flatten()` is
        // tolerant of pre-V17 databases that lack these columns. Collapse an
        // all-NULL envelope to `None` so legacy nodes serialize unchanged.
        let parse_ts = |s: Option<String>| -> Option<DateTime<Utc>> {
            s.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .map(|dt| dt.with_timezone(&Utc))
                    .ok()
            })
        };
        let envelope = crate::memory::SourceEnvelope {
            source_system: row.get("source_system").ok().flatten(),
            source_id: row.get("source_id").ok().flatten(),
            source_url: row.get("source_url").ok().flatten(),
            source_updated_at: parse_ts(row.get("source_updated_at").ok().flatten()),
            content_hash: row.get("content_hash").ok().flatten(),
            synced_at: parse_ts(row.get("synced_at").ok().flatten()),
            source_project: row.get("source_project").ok().flatten(),
            source_type: row.get("source_type").ok().flatten(),
            source_author: row.get("source_author").ok().flatten(),
        };
        let source_envelope = if envelope.is_empty() {
            None
        } else {
            Some(envelope)
        };

        Ok(KnowledgeNode {
            id: row.get("id")?,
            content: row.get("content")?,
            node_type: row.get("node_type")?,
            created_at,
            updated_at,
            last_accessed,
            stability: row.get("stability")?,
            difficulty: row.get("difficulty")?,
            reps: row.get("reps")?,
            lapses: row.get("lapses")?,
            storage_strength: row.get("storage_strength")?,
            retrieval_strength: row.get("retrieval_strength")?,
            retention_strength: row.get("retention_strength")?,
            sentiment_score: row.get("sentiment_score")?,
            sentiment_magnitude: row.get("sentiment_magnitude")?,
            next_review,
            source: row.get("source")?,
            tags,
            valid_from,
            valid_until,
            has_embedding: has_embedding.map(|v| v == 1),
            embedding_model,
            // v2.0 fields
            utility_score: row.get("utility_score").ok(),
            times_retrieved: row.get("times_retrieved").ok(),
            times_useful: row.get("times_useful").ok(),
            emotional_valence: row.get("emotional_valence").ok(),
            flashbulb: row.get::<_, Option<bool>>("flashbulb").ok().flatten(),
            temporal_level: row
                .get::<_, Option<String>>("temporal_level")
                .ok()
                .flatten(),
            // v2.0.5 Active Forgetting
            suppression_count,
            suppressed_at,
            // #57 Source envelope
            source_envelope,
        })
    }

    /// Recall memories matching a query
    pub fn recall(&self, input: RecallInput) -> Result<Vec<KnowledgeNode>> {
        self.recall_in_scope(input, DEFAULT_MEMORY_SCOPE)
    }

    /// Recall only memories from one namespace. This is intentionally the
    /// safe default for all core recall: callers that need a project must name
    /// it, and cross-project retrieval is an explicit higher-level operation.
    pub fn recall_in_scope(&self, input: RecallInput, scope: &str) -> Result<Vec<KnowledgeNode>> {
        let scope = Self::normalize_scope(scope)?;
        let nodes = match input.search_mode {
            SearchMode::Keyword => {
                self.keyword_search(&input.query, input.limit, input.min_retention)?
            }
            #[cfg(all(feature = "embeddings", feature = "vector-search"))]
            SearchMode::Semantic => {
                if !self.vector_search_available() {
                    self.keyword_search(&input.query, input.limit, input.min_retention)?
                } else {
                    let results = self.semantic_search(&input.query, input.limit, 0.3)?;
                    results.into_iter().map(|r| r.node).collect()
                }
            }
            #[cfg(all(feature = "embeddings", feature = "vector-search"))]
            SearchMode::Hybrid => {
                let results = self.hybrid_search(&input.query, input.limit, 0.3, 0.7)?;
                results.into_iter().map(|r| r.node).collect()
            }
            #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
            _ => self.keyword_search(&input.query, input.limit, input.min_retention)?,
        };

        // Retrieval is evidence that a memory was shown, not evidence that it
        // was correct or useful. Preserve the telemetry without changing its
        // ranking or FSRS state; callers must send explicit positive feedback
        // to reinforce a memory.
        let nodes: Vec<KnowledgeNode> = nodes
            .into_iter()
            .filter_map(|node| match self.node_is_in_scope(&node.id, scope) {
                Ok(true) => Some(Ok(node)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            })
            .collect::<Result<Vec<_>>>()?;

        let ids: Vec<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
        let _ = self.record_batch_retrieval(&ids); // Ignore errors, don't fail recall

        Ok(nodes)
    }

    /// Keyword search with FTS5
    fn keyword_search(
        &self,
        query: &str,
        limit: i32,
        min_retention: f64,
    ) -> Result<Vec<KnowledgeNode>> {
        let sanitized_query = sanitize_fts5_query(query);

        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT n.* FROM knowledge_nodes n
             JOIN knowledge_fts fts ON n.id = fts.id
             WHERE knowledge_fts MATCH ?1
             AND n.retention_strength >= ?2
             ORDER BY n.retention_strength DESC
             LIMIT ?3",
        )?;

        let nodes = stmt.query_map(params![sanitized_query, min_retention, limit], |row| {
            Self::row_to_node(row)
        })?;

        let mut result = Vec::new();
        for node in nodes {
            result.push(node?);
        }
        Ok(result)
    }

    /// Mark a memory as reviewed
    pub fn mark_reviewed(&self, id: &str, rating: Rating) -> Result<KnowledgeNode> {
        let node = self
            .get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))?;

        let learning_state = match node.reps {
            0 => LearningState::New,
            _ if node.lapses > 0 && node.reps == node.lapses => LearningState::Relearning,
            _ => LearningState::Review,
        };

        let current_state = FSRSState {
            difficulty: node.difficulty,
            stability: node.stability,
            state: learning_state,
            reps: node.reps,
            lapses: node.lapses,
            last_review: node.last_accessed,
            scheduled_days: 0,
        };

        let scheduler = self
            .scheduler
            .lock()
            .map_err(|_| StorageError::Init("Scheduler lock poisoned".into()))?;
        let elapsed_days = scheduler.days_since_review(&current_state.last_review);

        let sentiment_boost = if node.sentiment_magnitude > 0.0 {
            Some(node.sentiment_magnitude)
        } else {
            None
        };

        let result = scheduler.review(&current_state, rating, elapsed_days, sentiment_boost);
        drop(scheduler);

        let now = Utc::now();
        let next_review = now + Duration::days(result.interval as i64);

        let new_storage_strength = if rating != Rating::Again {
            node.storage_strength + 0.1
        } else {
            node.storage_strength + 0.3
        };

        let new_retrieval_strength = 1.0;
        let new_retention =
            (new_retrieval_strength * 0.7) + ((new_storage_strength / 10.0).min(1.0) * 0.3);

        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            writer.execute(
                "UPDATE knowledge_nodes SET
                    stability = ?1,
                    difficulty = ?2,
                    reps = ?3,
                    lapses = ?4,
                    learning_state = ?5,
                    storage_strength = ?6,
                    retrieval_strength = ?7,
                    retention_strength = ?8,
                    last_accessed = ?9,
                    updated_at = ?10,
                    next_review = ?11,
                    scheduled_days = ?12
                WHERE id = ?13",
                params![
                    result.state.stability,
                    result.state.difficulty,
                    result.state.reps,
                    result.state.lapses,
                    format!("{:?}", result.state.state).to_lowercase(),
                    new_storage_strength,
                    new_retrieval_strength,
                    new_retention,
                    now.to_rfc3339(),
                    now.to_rfc3339(),
                    next_review.to_rfc3339(),
                    result.interval,
                    id,
                ],
            )?;
        }

        self.get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    /// Reinforce a memory after an intentional confirmation of relevance.
    ///
    /// Ordinary retrieval must use [`Self::record_batch_retrieval`] instead:
    /// being shown in search is not evidence that a memory was correct or
    /// useful. This helper remains for explicit duplicate/reinforcement flows.
    /// It implements the Testing Effect (Roediger & Karpicke 2006) + v1.4.0
    /// content-aware cross-memory reinforcement: semantically similar neighbors
    /// receive a diminished boost proportional to cosine similarity.
    pub fn strengthen_on_access(&self, id: &str) -> Result<()> {
        let now = Utc::now();

        // Primary boost on the accessed node
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            writer.execute(
                "UPDATE knowledge_nodes SET
                    last_accessed = ?1,
                    retrieval_strength = MIN(1.0, retrieval_strength + 0.05),
                    retention_strength = MIN(1.0, retention_strength + 0.02),
                    times_retrieved = COALESCE(times_retrieved, 0) + 1,
                    utility_score = CASE
                        WHEN COALESCE(times_retrieved, 0) + 1 > 0
                        THEN CAST(COALESCE(times_useful, 0) AS REAL) / (COALESCE(times_retrieved, 0) + 1)
                        ELSE 0.0
                    END
                WHERE id = ?2",
                params![now.to_rfc3339(), id],
            )?;
        }

        // This is a deliberate reinforcement, not a passive search hit.
        let _ = self.log_access(id, "reinforce");

        // Content-aware cross-memory reinforcement: boost semantically similar neighbors
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        {
            if let Some(index) = self.vector_index.as_ref()
                && let Ok(Some(embedding)) = self.get_node_embedding(id)
            {
                let index = index
                    .lock()
                    .map_err(|_| StorageError::Init("Vector index lock poisoned".to_string()))?;

                // Query top-6 similar (one will be self, so we get ~5 neighbors)
                let neighbors_result = index.search(&embedding, 6);
                drop(index);

                if let Ok(neighbors) = neighbors_result {
                    let writer = self
                        .writer
                        .lock()
                        .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
                    for (neighbor_id, similarity) in neighbors {
                        if neighbor_id == id || similarity < 0.7 {
                            continue;
                        }
                        // Diminished boost: 0.02 * similarity (max ~0.02)
                        let boost = 0.02 * similarity as f64;
                        let retention_boost = 0.008 * similarity as f64;
                        let _ = writer.execute(
                            "UPDATE knowledge_nodes SET
                                retrieval_strength = MIN(1.0, retrieval_strength + ?1),
                                retention_strength = MIN(1.0, retention_strength + ?2)
                            WHERE id = ?3",
                            params![boost, retention_boost, neighbor_id],
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Batch-strengthen memories after an intentional confirmation of relevance.
    pub fn strengthen_batch_on_access(&self, ids: &[&str]) -> Result<()> {
        for id in ids {
            self.strengthen_on_access(id)?;
            // Also record access in memory_states for audit trail (Bug #1 fix)
            let _ = self.record_memory_access(id);
        }
        Ok(())
    }

    /// Record that a memory was returned to a caller without reinforcing it.
    ///
    /// A search hit is not proof of correctness or usefulness. We retain only
    /// access-log evidence for auditability, leaving node state and every
    /// learning/ranking signal untouched. Call [`Self::promote_memory`] or
    /// [`Self::mark_memory_useful`] only after an explicit positive signal.
    pub fn record_batch_retrieval(&self, ids: &[&str]) -> Result<()> {
        for id in ids {
            self.log_access(id, "retrieval_shown")?;
        }

        Ok(())
    }

    /// Mark a memory as "useful" — called when a retrieved memory is subsequently
    /// referenced in a save or decision (MemRL-inspired utility tracking).
    ///
    /// Increments `times_useful` and recomputes `utility_score = times_useful / times_retrieved`.
    pub fn mark_memory_useful(&self, id: &str) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "UPDATE knowledge_nodes SET
                times_useful = COALESCE(times_useful, 0) + 1,
                utility_score = CASE
                    WHEN COALESCE(times_retrieved, 0) > 0
                    THEN MIN(1.0, CAST(COALESCE(times_useful, 0) + 1 AS REAL) / COALESCE(times_retrieved, 0))
                    ELSE 1.0
                END
            WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// Log a memory interaction for audit and explicit-feedback learning.
    pub(crate) fn log_access(&self, node_id: &str, access_type: &str) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "INSERT INTO memory_access_log (node_id, access_type, accessed_at)
             VALUES (?1, ?2, ?3)",
            params![node_id, access_type, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    /// Promote a memory (thumbs up) - used when a memory led to a good outcome
    /// Significantly boosts retrieval strength so it surfaces more often.
    /// v1.9.0: Also sets waking SWR tag for preferential dream replay.
    pub fn promote_memory(&self, id: &str) -> Result<KnowledgeNode> {
        let now = Utc::now();

        // Explicit positive feedback: boost strength and record that this
        // memory proved useful to the caller.
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            writer.execute(
                "UPDATE knowledge_nodes SET
                    last_accessed = ?1,
                    retrieval_strength = MIN(1.0, retrieval_strength + 0.20),
                    retention_strength = MIN(1.0, retention_strength + 0.10),
                    stability = stability * 1.5,
                    times_useful = COALESCE(times_useful, 0) + 1,
                    utility_score = CASE
                        WHEN COALESCE(times_retrieved, 0) > 0
                        THEN MIN(1.0, CAST(COALESCE(times_useful, 0) + 1 AS REAL) / COALESCE(times_retrieved, 0))
                        ELSE 1.0
                    END
                WHERE id = ?2",
                params![now.to_rfc3339(), id],
            )?;
        }

        let _ = self.log_access(id, "promote");

        // v1.9.0: Set waking SWR tag for preferential dream replay
        let _ = self.set_waking_tag(id);

        self.get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    /// Backfill-specific promote: identical retrieval/retention boost to
    /// `promote_memory`, but the stability multiply is CAPPED at an additive
    /// +365-day ceiling: `MIN(stability * 1.5, stability + 365.0)`. The `1.5`
    /// factor preserves the multiplier `promote_memory` already applied; the
    /// `+365` ceiling is the same additive bound `retroactive_backfill.rs`
    /// uses for its reason string (that module pairs +365 with a 2.5 factor
    /// for display only — this DB write intentionally keeps 1.5 so backfill
    /// promotion strength is unchanged, just bounded). Repeated per-(cause,
    /// failure) backfill promotions therefore cannot inflate stability without
    /// bound. Used by the step-8.5 auto-fire path and the manual `backfill` tool.
    pub fn promote_memory_backfill(&self, id: &str) -> Result<KnowledgeNode> {
        let now = Utc::now();

        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            writer.execute(
                "UPDATE knowledge_nodes SET
                    last_accessed = ?1,
                    retrieval_strength = MIN(1.0, retrieval_strength + 0.20),
                    retention_strength = MIN(1.0, retention_strength + 0.10),
                    stability = MIN(stability * 1.5, stability + 365.0)
                WHERE id = ?2",
                params![now.to_rfc3339(), id],
            )?;
        }

        let _ = self.log_access(id, "promote");
        let _ = self.set_waking_tag(id);

        self.get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    /// Demote a memory (thumbs down) - used when a memory led to a bad outcome
    /// Significantly reduces retrieval strength so better alternatives surface
    /// Does NOT delete - the memory stays for reference but ranks lower
    pub fn demote_memory(&self, id: &str) -> Result<KnowledgeNode> {
        // Strong penalty: -0.3 retrieval, -0.15 retention, halve stability
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            // last_accessed intentionally untouched -- see suppress_memory: a
            // demotion is an inhibition event, not a recall, and apply_decay
            // would otherwise recompute the penalty away.
            writer.execute(
                "UPDATE knowledge_nodes SET
                    retrieval_strength = MAX(0.05, retrieval_strength - 0.30),
                    retention_strength = MAX(0.05, retention_strength - 0.15),
                    stability = stability * 0.5
                WHERE id = ?1",
                params![id],
            )?;
        }

        let _ = self.log_access(id, "demote");

        self.get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    // ========================================================================
    // Active Forgetting (v2.0.5 — Anderson 2025 + Davis Rac1)
    // ========================================================================

    /// Top-down memory suppression (Suppression-Induced Forgetting).
    ///
    /// Distinct from `delete` (which removes the row) and from
    /// `demote_memory` (which is a single thumb-down hit). Each call
    /// compounds: `suppression_count` is incremented, `suppressed_at` is
    /// bumped to now, and FSRS state is dealt a strong blow:
    ///
    /// - `retrieval_strength -= 0.35` (stronger than demote's -0.30)
    /// - `retention_strength -= 0.20`
    /// - `stability *= 0.4`
    ///
    /// Reversible within a 24-hour labile window via
    /// [`Self::reverse_suppression`].
    ///
    /// Reference: Anderson et al. (2025). Brain mechanisms underlying the
    /// inhibitory control of thought. *Nature Reviews Neuroscience*.
    /// DOI: 10.1038/s41583-025-00929-y
    pub fn suppress_memory(&self, id: &str) -> Result<KnowledgeNode> {
        let now = Utc::now();
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            let tx = Self::begin_write_transaction(&writer, "suppress_memory")?;
            let changed = tx.execute(
                // NOTE: last_accessed is deliberately NOT touched here. apply_decay
                // RECOMPUTES retrieval_strength/retention_strength from
                // days_since(last_accessed) rather than decaying the stored value,
                // so stamping "now" would make an inhibited memory look freshly
                // recalled and the next consolidation pass would overwrite this
                // whole penalty -- silently un-suppressing it within hours.
                "UPDATE knowledge_nodes SET
                    suppression_count = COALESCE(suppression_count, 0) + 1,
                    suppressed_at = ?1,
                    retrieval_strength = MAX(0.05, retrieval_strength - 0.35),
                    retention_strength = MAX(0.05, retention_strength - 0.20),
                    stability = stability * 0.4
                WHERE id = ?2",
                params![now.to_rfc3339(), id],
            )?;
            if changed == 0 {
                return Err(StorageError::NotFound(id.to_string()));
            }
            Self::invalidate_replay_evidence_for_memory_in_transaction(
                &tx,
                id,
                crate::storage::ReplayInvalidationReason::Suppressed,
            )?;
            tx.commit()?;
        }

        let _ = self.log_access(id, "suppress");

        self.get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    /// Reverse a previous suppression if within the 24-hour labile window.
    ///
    /// Returns `Err(StorageError::NotFound)` if the memory has never been
    /// suppressed, or `Err(StorageError::Init)` with a "labile window expired"
    /// message if more than `labile_hours` have passed. Matches Nader
    /// reconsolidation semantics on a 24h axis.
    pub fn reverse_suppression(&self, id: &str, labile_hours: i64) -> Result<KnowledgeNode> {
        let node = self
            .get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))?;

        let suppressed_at = node.suppressed_at.ok_or_else(|| {
            StorageError::Init(format!(
                "memory {} has no active suppression to reverse",
                id
            ))
        })?;

        let elapsed = Utc::now() - suppressed_at;
        if elapsed >= chrono::Duration::hours(labile_hours) {
            return Err(StorageError::Init(format!(
                "labile window expired ({}h since suppression; limit {}h)",
                elapsed.num_hours(),
                labile_hours
            )));
        }

        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            // True inverse of suppress_memory (which applies stability * 0.4,
            // retrieval - 0.35, retention - 0.20). Dividing by 0.4 exactly undoes
            // the * 0.4, and adding back the same 0.35 / 0.20 deltas (clamped to
            // 1.0) undoes the subtraction. Previously this used non-inverse deltas
            // (* 1.25, + 0.15, + 0.10), so suppress-then-reverse left stability
            // permanently halved (0.4 * 1.25 = 0.5) while reporting a full undo.
            // Note: where the forward pass hit the MAX(0.05) floor, the exact
            // pre-value is unrecoverable without a snapshot — that clip aside,
            // this restores the pre-suppression FSRS state.
            writer.execute(
                "UPDATE knowledge_nodes SET
                    suppression_count = MAX(0, COALESCE(suppression_count, 0) - 1),
                    suppressed_at = CASE
                        WHEN COALESCE(suppression_count, 0) - 1 <= 0 THEN NULL
                        ELSE suppressed_at
                    END,
                    retrieval_strength = MIN(1.0, retrieval_strength + 0.35),
                    retention_strength = MIN(1.0, retention_strength + 0.20),
                    stability = stability / 0.4
                WHERE id = ?1",
                params![id],
            )?;
        }

        let _ = self.log_access(id, "reverse_suppress");

        self.get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    /// Release a memory from quarantine **unconditionally** (no labile-window
    /// limit), used when a Memory PR is approved.
    ///
    /// Unlike [`Self::reverse_suppression`] (which models a time-bounded "undo"
    /// of an active-forgetting decision and refuses after the labile window),
    /// approving a quarantined risky write is an explicit reviewer decision that
    /// must always restore the memory's retrieval influence — even days later.
    /// Fully clears the suppression (count → 0, `suppressed_at` → NULL) and
    /// restores strengths. A no-op (returns the node) if it isn't suppressed.
    pub fn release_quarantine(&self, id: &str) -> Result<KnowledgeNode> {
        let node = self
            .get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))?;

        if node.suppression_count == 0 && node.suppressed_at.is_none() {
            // Nothing to release — idempotent.
            return Ok(node);
        }

        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            writer.execute(
                "UPDATE knowledge_nodes SET
                    suppression_count = 0,
                    suppressed_at = NULL,
                    retrieval_strength = MIN(1.0, retrieval_strength + 0.15),
                    retention_strength = MIN(1.0, retention_strength + 0.10),
                    stability = stability * 1.25
                WHERE id = ?1",
                params![id],
            )?;
        }

        let _ = self.log_access(id, "release_quarantine");

        self.get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    /// Test-only: backdate a node's `suppressed_at` to simulate a suppression
    /// that happened long ago (e.g. to verify release works past the labile
    /// window). `pub(crate)` so sibling test modules can reach it.
    #[cfg(test)]
    pub(crate) fn set_suppressed_at_for_test(&self, id: &str, when: DateTime<Utc>) {
        if let Ok(writer) = self.writer.lock() {
            let _ = writer.execute(
                "UPDATE knowledge_nodes SET suppressed_at = ?1 WHERE id = ?2",
                params![when.to_rfc3339(), id],
            );
        }
    }

    /// Backdate a node's `created_at`. Intended for tests and demo seeding (e.g.
    /// to simulate a memory formed days ago so Retroactive Salience Backfill can
    /// reach back to it). Cross-crate `pub` so the MCP backfill test + demo
    /// harness can plant a dated cause. Returns Ok(()) on success.
    pub fn set_created_at(&self, id: &str, when: DateTime<Utc>) -> Result<()> {
        if let Ok(writer) = self.writer.lock() {
            writer.execute(
                "UPDATE knowledge_nodes SET created_at = ?1 WHERE id = ?2",
                params![when.to_rfc3339(), id],
            )?;
        }
        Ok(())
    }

    /// Count memories currently in a suppressed state (suppression_count > 0).
    pub fn count_suppressed(&self) -> Result<usize> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let count: i64 = reader.query_row(
            "SELECT COUNT(*) FROM knowledge_nodes WHERE COALESCE(suppression_count, 0) > 0",
            [],
            |row| row.get(0),
        )?;
        Ok(count.max(0) as usize)
    }

    /// Fetch memories suppressed within the last `window_hours` (still within
    /// the labile window). Used by the Rac1 cascade sweep.
    pub fn get_recently_suppressed(&self, window_hours: i64) -> Result<Vec<KnowledgeNode>> {
        let cutoff = (Utc::now() - chrono::Duration::hours(window_hours)).to_rfc3339();
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT * FROM knowledge_nodes
             WHERE suppressed_at IS NOT NULL AND suppressed_at >= ?1",
        )?;
        let rows = stmt.query_map(params![cutoff], Self::row_to_node)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Apply one-hop Rac1 cascade from a single suppressed seed memory:
    /// walk `memory_connections` edges and attenuate neighbor FSRS state
    /// proportional to edge strength.
    ///
    /// Returns the number of neighbors affected.
    ///
    /// Reference: Cervantes-Sandoval & Davis (2020). Rac1 Impairs Forgetting-
    /// Induced Cellular Plasticity in Mushroom Body Output Neurons.
    /// *Front Cell Neurosci*. PMC7477079
    pub fn apply_rac1_cascade(&self, seed_id: &str) -> Result<usize> {
        use crate::neuroscience::active_forgetting::ActiveForgettingSystem;
        let sys = ActiveForgettingSystem::new();

        let edges = self.get_connections_for_memory(seed_id)?;
        if edges.is_empty() {
            return Ok(0);
        }

        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;

        let mut affected = 0usize;
        for edge in edges.iter().take(100) {
            let neighbor_id = if edge.source_id == seed_id {
                &edge.target_id
            } else {
                &edge.source_id
            };

            // Never cascade back into the suppressed seed
            if neighbor_id == seed_id {
                continue;
            }

            let stability_factor = sys.cascade_stability_factor(edge.strength);
            let retrieval_decrement = sys.cascade_retrieval_decrement(edge.strength);

            let rows = writer.execute(
                "UPDATE knowledge_nodes SET
                    stability = MAX(0.1, stability * ?1),
                    retrieval_strength = MAX(0.05, retrieval_strength - ?2)
                 WHERE id = ?3 AND COALESCE(suppression_count, 0) = 0",
                params![stability_factor, retrieval_decrement, neighbor_id],
            )?;
            affected += rows;
        }

        Ok(affected)
    }

    /// Sweep all recently-suppressed memories and apply Rac1 cascade to each.
    /// Intended to run from the background consolidation loop every tick.
    ///
    /// Returns `(seeds_processed, neighbors_affected)`.
    pub fn run_rac1_cascade_sweep(&self) -> Result<(usize, usize)> {
        // 72h keeps the cascade window slightly longer than the 24h labile
        // reversibility window — so suppressions that lock in continue to
        // propagate for 48h after they become irreversible.
        let seeds = self.get_recently_suppressed(72)?;
        let mut total_affected = 0usize;
        for seed in &seeds {
            match self.apply_rac1_cascade(&seed.id) {
                Ok(n) => total_affected += n,
                Err(e) => tracing::warn!("Rac1 cascade failed for {}: {}", seed.id, e),
            }
        }
        Ok((seeds.len(), total_affected))
    }

    /// Get memories due for review
    pub fn get_review_queue(&self, limit: i32) -> Result<Vec<KnowledgeNode>> {
        let now = Utc::now().to_rfc3339();

        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT * FROM knowledge_nodes
             WHERE next_review <= ?1
             ORDER BY next_review ASC
             LIMIT ?2",
        )?;

        let nodes = stmt.query_map(params![now, limit], Self::row_to_node)?;

        let mut result = Vec::new();
        for node in nodes {
            result.push(node?);
        }
        Ok(result)
    }

    /// Preview FSRS review outcomes for all rating options
    pub fn preview_review(&self, id: &str) -> Result<crate::fsrs::PreviewResults> {
        let node = self
            .get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))?;

        let learning_state = match node.reps {
            0 => LearningState::New,
            _ if node.lapses > 0 && node.reps == node.lapses => LearningState::Relearning,
            _ => LearningState::Review,
        };

        let current_state = FSRSState {
            difficulty: node.difficulty,
            stability: node.stability,
            state: learning_state,
            reps: node.reps,
            lapses: node.lapses,
            last_review: node.last_accessed,
            scheduled_days: 0,
        };

        let scheduler = self
            .scheduler
            .lock()
            .map_err(|_| StorageError::Init("Scheduler lock poisoned".into()))?;
        let elapsed_days = scheduler.days_since_review(&current_state.last_review);

        Ok(scheduler.preview_reviews(&current_state, elapsed_days))
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> Result<MemoryStats> {
        let now = Utc::now().to_rfc3339();

        // Resolve the active pointer before taking the shared reader lock.
        // `active_embedding_profile` reads through that same mutex; calling it
        // below after acquiring `reader` would self-deadlock every stats read.
        #[cfg(feature = "embeddings")]
        let active_profile = self.active_embedding_profile()?;

        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;

        let total: i64 =
            reader.query_row("SELECT COUNT(*) FROM knowledge_nodes", [], |row| row.get(0))?;

        let due: i64 = reader.query_row(
            "SELECT COUNT(*) FROM knowledge_nodes WHERE next_review <= ?1",
            params![now],
            |row| row.get(0),
        )?;

        let avg_retention: f64 = reader.query_row(
            "SELECT COALESCE(AVG(retention_strength), 0) FROM knowledge_nodes",
            [],
            |row| row.get(0),
        )?;

        let avg_storage: f64 = reader.query_row(
            "SELECT COALESCE(AVG(storage_strength), 1) FROM knowledge_nodes",
            [],
            |row| row.get(0),
        )?;

        let avg_retrieval: f64 = reader.query_row(
            "SELECT COALESCE(AVG(retrieval_strength), 1) FROM knowledge_nodes",
            [],
            |row| row.get(0),
        )?;

        let oldest: Option<String> = reader
            .query_row("SELECT MIN(created_at) FROM knowledge_nodes", [], |row| {
                row.get(0)
            })
            .ok();

        let newest: Option<String> = reader
            .query_row("SELECT MAX(created_at) FROM knowledge_nodes", [], |row| {
                row.get(0)
            })
            .ok();

        let nodes_with_embeddings: i64 = reader.query_row(
            "SELECT COUNT(*) FROM knowledge_nodes WHERE has_embedding = 1",
            [],
            |row| row.get(0),
        )?;

        let embedding_model: Option<String> = reader
            .query_row(
                "SELECT model
                 FROM node_embeddings
                 GROUP BY model
                 ORDER BY COUNT(*) DESC, model ASC
                 LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;

        #[cfg(feature = "embeddings")]
        let active_embedding_model = active_profile.as_ref().and_then(|active| {
            reader
                .query_row(
                    "SELECT model_id FROM embedding_profiles WHERE profile_id = ?1",
                    params![active.profile_id.as_str()],
                    |row| row.get::<_, String>(0),
                )
                .ok()
        });
        #[cfg(not(feature = "embeddings"))]
        let active_embedding_model: Option<String> = None;

        #[cfg(feature = "embeddings")]
        let (nodes_with_active_embeddings, nodes_with_mismatched_embeddings) = {
            let active_profile_id = active_profile
                .as_ref()
                .map(|active| active.profile_id.as_str());
            let active_model = active_embedding_model.as_deref();
            let active_count: i64 = reader.query_row(
                "SELECT COUNT(*)
                 FROM knowledge_nodes kn
                 WHERE EXISTS (
                       SELECT 1 FROM embedding_profile_vectors epv
                       WHERE epv.node_id = kn.id
                         AND epv.profile_id = ?1
                         AND epv.model = ?2
                         AND epv.dimensions = (
                             SELECT embedding_dimension FROM embedding_profiles
                             WHERE profile_id = ?1
                         )
                   )",
                params![active_profile_id, active_model],
                |row| row.get(0),
            )?;
            let mismatched_count: i64 = reader.query_row(
                "SELECT COUNT(*)
                 FROM knowledge_nodes kn
                 WHERE (kn.has_embedding = 1 OR EXISTS (
                       SELECT 1 FROM embedding_profile_vectors epv WHERE epv.node_id = kn.id
                   ))
                   AND NOT EXISTS (
                       SELECT 1 FROM embedding_profile_vectors epv
                       WHERE epv.node_id = kn.id
                         AND epv.profile_id = ?1
                         AND epv.model = ?2
                         AND epv.dimensions = (
                             SELECT embedding_dimension FROM embedding_profiles
                             WHERE profile_id = ?1
                         )
                   )",
                params![active_profile_id, active_model],
                |row| row.get(0),
            )?;
            (active_count, mismatched_count)
        };
        #[cfg(not(feature = "embeddings"))]
        let (nodes_with_active_embeddings, nodes_with_mismatched_embeddings) =
            (nodes_with_embeddings, 0);

        Ok(MemoryStats {
            total_nodes: total,
            nodes_due_for_review: due,
            average_retention: avg_retention,
            average_storage_strength: avg_storage,
            average_retrieval_strength: avg_retrieval,
            oldest_memory: oldest.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .map(|dt| dt.with_timezone(&Utc))
                    .ok()
            }),
            newest_memory: newest.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .map(|dt| dt.with_timezone(&Utc))
                    .ok()
            }),
            nodes_with_embeddings,
            nodes_with_active_embeddings,
            nodes_with_mismatched_embeddings,
            embedding_model,
            active_embedding_model,
        })
    }

    /// Introspect the live SQLite schema: schema version + per-table row/column
    /// shape + embedding-coverage convenience fields.
    ///
    /// This is the v2.1.24+ replacement for the direct-SQLite reads that
    /// audit scripts and migration guards previously had to perform. The set
    /// of tables walked matches `PORTABLE_USER_DATA_TABLES` — the same
    /// canonical set used by portable export / import — so the surface stays
    /// stable across migrations rather than chasing arbitrary
    /// `sqlite_master` rows.
    ///
    /// Cost: O(N_tables) `COUNT(*)` queries + one PRAGMA per table. Negligible
    /// at the table cardinalities Vestige carries (~15 tables, all indexed).
    /// Safe to call on every MCP `system_status` invocation when the flag is
    /// set; callers wanting to limit cost should leave the flag off (default).
    pub fn schema_introspection(&self) -> Result<crate::SchemaIntrospection> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;

        let schema_version = Self::current_schema_version(&reader)?;

        // schema_version has the row (version PK + applied_at TEXT). Read the
        // applied_at for the current version row; tolerate failure (legacy
        // databases may have skipped the applied_at fill on early upgrades).
        let applied_at_str: Option<String> = reader
            .query_row(
                "SELECT applied_at FROM schema_version WHERE version = ?1",
                params![schema_version as i64],
                |row| row.get(0),
            )
            .optional()?;
        let schema_version_applied_at = applied_at_str.and_then(|s| {
            // The migration scripts use `datetime('now')` which yields
            // SQLite's "YYYY-MM-DD HH:MM:SS" UTC form (NOT RFC3339).
            // Try the SQLite form first, fall back to RFC3339 for any
            // future migrations that switch.
            chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S")
                .map(|naive| naive.and_utc())
                .or_else(|_| DateTime::parse_from_rfc3339(&s).map(|dt| dt.with_timezone(&Utc)))
                .ok()
        });

        let mut tables = Vec::with_capacity(PORTABLE_USER_DATA_TABLES.len());
        for table_name in PORTABLE_USER_DATA_TABLES {
            if Self::table_exists(&reader, table_name)? {
                let rows = Self::table_row_count(&reader, table_name)?;
                let columns = Self::table_columns(&reader, table_name)?;
                tables.push(crate::TableIntrospection {
                    name: (*table_name).to_string(),
                    rows,
                    columns,
                });
            }
        }

        // Convenience: active-profile coverage is the number of nodes with no
        // vector in the currently selected isolated vector space.
        let active_profile_id = Self::active_profile_id_from_conn(&reader)?;
        let embedding_null_count: i64 = reader
            .query_row(
                "SELECT COUNT(*) FROM knowledge_nodes kn
                 WHERE NOT EXISTS (
                     SELECT 1 FROM embedding_profile_vectors epv
                     WHERE epv.node_id = kn.id AND epv.profile_id = ?1
                 )",
                params![active_profile_id],
                |row| row.get(0),
            )
            .unwrap_or(0);

        #[cfg(feature = "embeddings")]
        let active_embedding_model = active_profile_id.as_deref().and_then(|profile_id| {
            reader
                .query_row(
                    "SELECT model_id FROM embedding_profiles WHERE profile_id = ?1",
                    params![profile_id],
                    |row| row.get::<_, String>(0),
                )
                .ok()
        });
        #[cfg(not(feature = "embeddings"))]
        let active_embedding_model: Option<String> = None;

        #[cfg(feature = "embeddings")]
        let active_embedding_dimensions: Option<u32> =
            active_profile_id.as_deref().and_then(|profile_id| {
                reader
                    .query_row(
                        "SELECT embedding_dimension FROM embedding_profiles WHERE profile_id = ?1",
                        params![profile_id],
                        |row| row.get::<_, i64>(0),
                    )
                    .ok()
                    .and_then(|dimension| u32::try_from(dimension).ok())
            });
        #[cfg(not(feature = "embeddings"))]
        let active_embedding_dimensions: Option<u32> = None;

        Ok(crate::SchemaIntrospection {
            schema_version,
            schema_version_applied_at,
            tables,
            embedding_null_count,
            active_embedding_model,
            active_embedding_dimensions,
        })
    }

    /// Delete a node through the same privacy cleanup coordinator as an explicit
    /// purge.  Keeping one deletion path prevents maintenance, dashboard, and
    /// library callers from bypassing replay invalidation or durable-evidence
    /// redaction.
    pub fn delete_node(&self, id: &str) -> Result<bool> {
        Ok(self.purge_node(id, None)?.deleted)
    }

    /// Permanently purge a memory's content and embeddings.
    ///
    /// This is the one local deletion coordinator. It scrubs non-FK references,
    /// invalidates replay evidence, detaches temporal-summary children, and
    /// writes an opaque deletion marker for audit/sync. It remains a legacy
    /// cleanup path and deliberately does not claim verified local unlearning.
    pub fn purge_node(&self, id: &str, reason: Option<&str>) -> Result<PurgeReport> {
        // The reason is logged, never persisted: deletion_tombstones are
        // content-free by contract (an opaque marker, no reason, no tags), so
        // a purged memory leaves nothing recoverable. The local log line is
        // how an operator answers "what deleted this?" without the tombstone
        // ever carrying it.
        if let Some(reason) = reason {
            tracing::info!(memory_id = %id, reason, "purging memory");
        }
        let deleted_at = Utc::now();
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "purge_node")?;
        let cleanup = Self::purge_node_in_transaction(&tx, id, deleted_at, true)?;
        tx.commit()?;
        // Release the writer BEFORE taking the vector-index lock. Every other
        // combined site in this file orders writer -> drop -> index, and
        // `activate_embedding_profile` orders index -> writer, so holding the
        // writer here while waiting on the index is an AB/BA deadlock between a
        // purge and a concurrent profile activation.
        drop(writer);

        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        if cleanup.is_some()
            && let Some(index) = self.vector_index.as_ref()
            && let Ok(mut index) = index.lock()
        {
            let _ = index.remove(id);
        }

        let Some(cleanup) = cleanup else {
            return Ok(PurgeReport {
                memory_id: id.to_string(),
                deleted: false,
                deleted_at,
                edges_pruned: 0,
                insights_rewritten: 0,
                insights_deleted: 0,
                children_orphaned: 0,
                unlearning_scope: crate::storage::UnlearningScope::LegacyAuditedPurge,
                unlearning_verdict: crate::storage::UnlearningVerdict::Incomplete,
                unlearning_claim_boundary: "No purge ran because the requested memory was not found; no unlearning audit or verified-local erasure claim was produced.",
            });
        };

        Ok(PurgeReport {
            memory_id: id.to_string(),
            deleted: true,
            deleted_at,
            edges_pruned: cleanup.edges_pruned,
            insights_rewritten: cleanup.insights_rewritten,
            insights_deleted: cleanup.insights_deleted,
            children_orphaned: cleanup.children_orphaned,
            unlearning_scope: crate::storage::UnlearningScope::LegacyAuditedPurge,
            unlearning_verdict: crate::storage::UnlearningVerdict::Incomplete,
            unlearning_claim_boundary: "Legacy cleanup completed, but this operation has no V25 lineage-completeness proof, full required-surface audit, or anti-resurrection ingress gate. It does not establish complete machine unlearning, erasure of unmanaged copies, media forensics, provider backups, external model weights, or re-ingest prevention.",
        })
    }

    /// Remove a committed purge from the optional in-process vector index.
    pub(crate) fn remove_purged_node_from_vector_index(&self, id: &str) {
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        if let Some(index) = self.vector_index.as_ref()
            && let Ok(mut index) = index.lock()
        {
            let _ = index.remove(id);
        }
        #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
        let _ = id;
    }

    /// Execute the privacy-critical delete work inside a caller-owned SQLite
    /// transaction. Portable merge uses this exact path so a remote deletion
    /// cannot leave local non-FK evidence behind.
    pub(crate) fn purge_node_in_transaction(
        tx: &rusqlite::Transaction<'_>,
        id: &str,
        deleted_at: DateTime<Utc>,
        write_tombstones: bool,
    ) -> Result<Option<PurgeCleanup>> {
        let node = tx
            .prepare("SELECT * FROM knowledge_nodes WHERE id = ?1")?
            .query_row(params![id], Self::row_to_node)
            .optional()?;

        let Some(node) = node else {
            return Ok(None);
        };

        let edges_pruned: i64 = tx.query_row(
            "SELECT COUNT(*) FROM memory_connections WHERE source_id = ?1 OR target_id = ?1",
            params![id],
            |row| row.get(0),
        )?;

        let insight_refs: Vec<(String, String)> = {
            let mut stmt = tx.prepare(
                "SELECT id, source_memories FROM insights WHERE source_memories LIKE ?1",
            )?;
            let pattern = format!("%{}%", id);
            // Purge fails closed. A row this scrub cannot read is a row that
            // still references the memory the caller asked us to erase, and
            // reporting a successful purge over it is the one lie this store
            // must never tell. Propagating rolls the whole purge back.
            stmt.query_map(params![pattern], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };

        let mut insights_rewritten = 0_i64;
        let mut insights_deleted = 0_i64;
        for (insight_id, source_json) in insight_refs {
            let mut sources: Vec<String> = serde_json::from_str(&source_json).unwrap_or_default();
            let before = sources.len();
            sources.retain(|source_id| source_id != id);

            if sources.len() == before {
                continue;
            }

            if sources.len() < 2 {
                insights_deleted +=
                    tx.execute("DELETE FROM insights WHERE id = ?1", params![insight_id])? as i64;
            } else {
                let rewritten = serde_json::to_string(&sources).unwrap_or_else(|_| "[]".into());
                insights_rewritten += tx.execute(
                    "UPDATE insights SET source_memories = ?1 WHERE id = ?2",
                    params![rewritten, insight_id],
                )? as i64;
            }
        }

        let children_orphaned = tx.execute(
            "UPDATE knowledge_nodes SET summary_parent_id = NULL WHERE summary_parent_id = ?1",
            params![id],
        )? as i64;

        // Review records are intentionally not FK-linked to memories, so a
        // normal node delete would retain their subject id, previews, tags, and
        // potentially user-provided rationale. An erasure request takes privacy
        // precedence over that historical review record.
        tx.execute(
            r#"DELETE FROM memory_prs
                WHERE subject_id = ?1
                   OR (?2 <> '' AND instr(title, ?2) > 0)
                   OR instr(diff, ?1) > 0 OR (?2 <> '' AND instr(diff, ?2) > 0)
                   OR instr(signals, ?1) > 0 OR (?2 <> '' AND instr(signals, ?2) > 0)"#,
            params![id, &node.content],
        )?;

        // Composition members intentionally preserve historical memory ids.
        // Once a user requests erasure, retaining the surrounding event can
        // still expose the memory through query/output/metadata fields. Delete
        // the whole affected event (and FK-cascaded members/outcomes) rather
        // than attempting partial JSON surgery.
        tx.execute(
            r#"DELETE FROM composition_events
                WHERE id IN (
                    SELECT event_id FROM composition_members WHERE memory_id = ?1
                )
                   OR (?2 <> '' AND instr(COALESCE(query, ''), ?2) > 0)
                   OR (?2 <> '' AND instr(COALESCE(output_preview, ''), ?2) > 0)
                   OR instr(metadata, ?1) > 0
                   OR (?2 <> '' AND instr(metadata, ?2) > 0)"#,
            params![id, &node.content],
        )?;

        // A purge must erase frozen replay dependency locators and invalidate
        // every derived replay in the same transaction as the memory removal.
        // This also upgrades a previously redacted capsule to `purged`.
        Self::invalidate_replay_evidence_for_memory_in_transaction(
            tx,
            id,
            crate::storage::ReplayInvalidationReason::Purged,
        )?;

        tx.execute(
            "UPDATE composition_members SET preview = NULL WHERE memory_id = ?1",
            params![id],
        )?;

        // Purge overrides historical receipt fidelity: remove the stable id
        // from every persisted receipt payload while retaining its evidence
        // slots, score, disposition, and measured deltas. Public reads also
        // resolve current state, but this closes the raw V21 audit-row copy.
        let receipt_refs: Vec<(String, String)> = {
            let mut stmt = tx
                .prepare("SELECT receipt_id, payload FROM memory_receipts WHERE payload LIKE ?1")?;
            let pattern = format!("%{}%", id);
            // Purge fails closed. A row this scrub cannot read is a row that
            // still references the memory the caller asked us to erase, and
            // reporting a successful purge over it is the one lie this store
            // must never tell. Propagating rolls the whole purge back.
            stmt.query_map(params![pattern], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        for (receipt_id, payload) in receipt_refs {
            let Ok(mut receipt) = serde_json::from_str::<crate::trace::Receipt>(&payload) else {
                // Not structurally redactable. The raw-text sweeps below still
                // strip the content; say so, because the id may survive here.
                tracing::warn!(
                    receipt_id = %receipt_id,
                    memory_id = %id,
                    "purge could not parse a receipt payload for structured redaction; the raw-text sweep still applies"
                );
                continue;
            };
            receipt.redact_memory_id(id, "purged_1");
            let rewritten = serde_json::to_string(&receipt)
                .map_err(|e| StorageError::Init(format!("receipt redact serialize: {e}")))?;
            tx.execute(
                "UPDATE memory_receipts SET payload = ?1 WHERE receipt_id = ?2",
                params![rewritten, receipt_id],
            )?;
        }
        tx.execute(
            "UPDATE memory_receipts SET query = NULL
             WHERE instr(COALESCE(query, ''), ?1) > 0
                OR (?2 <> '' AND instr(COALESCE(query, ''), ?2) > 0)",
            params![id, &node.content],
        )?;

        // Black Box traces are public/exportable evidence too. Rewrite every
        // id-bearing payload and delete any trace containing the target text;
        // a structured redactor cannot safely prove removal of arbitrary text
        // from historical trace JSON.
        let trace_refs: Vec<(String, String)> = {
            let mut stmt =
                tx.prepare("SELECT id, payload FROM agent_traces WHERE payload LIKE ?1")?;
            let pattern = format!("%{}%", id);
            // Purge fails closed. A row this scrub cannot read is a row that
            // still references the memory the caller asked us to erase, and
            // reporting a successful purge over it is the one lie this store
            // must never tell. Propagating rolls the whole purge back.
            stmt.query_map(params![pattern], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        for (trace_id, payload) in trace_refs {
            let Ok(mut event) = serde_json::from_str::<crate::trace::MemoryTraceEvent>(&payload)
            else {
                // Same as receipts: unparseable payloads fall through to the
                // raw-text delete below, but the operator should see it.
                tracing::warn!(
                    trace_id = %trace_id,
                    memory_id = %id,
                    "purge could not parse a trace payload for structured redaction; the raw-text sweep still applies"
                );
                continue;
            };
            event.redact_memory_id(id, "purged_1");
            let rewritten = serde_json::to_string(&event)
                .map_err(|e| StorageError::Init(format!("trace redact serialize: {e}")))?;
            tx.execute(
                "UPDATE agent_traces SET payload = ?1 WHERE id = ?2",
                params![rewritten, trace_id],
            )?;
        }
        tx.execute(
            "DELETE FROM agent_traces WHERE ?1 <> '' AND instr(payload, ?1) > 0",
            params![&node.content],
        )?;

        // A trigger event otherwise preserves the purged stable id outside the
        // knowledge-node FK graph. Capture-item rows cascade with the event;
        // candidate rows cascade through their synaptic tag on node deletion.
        //
        // A captured tag is only valid while it is bound to the capture item
        // and event which prove that state.  Purging the trigger deletes that
        // proof, so retaining `captured` would leave an invalid durable state
        // that prevents a later startup integrity check from succeeding.  An
        // expired tag cannot be recaptured, which preserves the one-promotion
        // lifecycle without claiming evidence that no longer exists.
        tx.execute(
            "UPDATE synaptic_tags
             SET state = 'expired', capture_event_id = NULL, captured_at_ms = NULL
             WHERE capture_event_id IN (
                 SELECT event_id FROM synaptic_events WHERE trigger_memory_id = ?1
             )",
            params![id],
        )?;
        tx.execute(
            "DELETE FROM synaptic_events WHERE trigger_memory_id = ?1",
            params![id],
        )?;

        // V24 deliberately keeps the immutable, identity-free DSSE envelope
        // after erasure, but its private disclosure mapping is deletable. The
        // FK also covers this when the node delete succeeds; doing it
        // explicitly keeps the privacy operation visible and makes a schema
        // regression fail before the canonical row is removed.
        if Self::table_exists(tx, "receipt_disclosures")? {
            tx.execute(
                "DELETE FROM receipt_disclosures WHERE memory_id = ?1",
                params![id],
            )?;
        }

        if write_tombstones {
            // The V13 table predates commitment-only V25 evidence, but it can
            // still be made content-free without a migration: use an opaque
            // stable marker as its primary key, store no caller reason, and
            // retain no tags. `sync_tombstones` uses the same marker and
            // resolves it locally during merge, so portable deletion
            // propagation remains functional.
            let tombstone_marker = Self::opaque_tombstone_marker(id);
            tx.execute(
                "INSERT INTO deletion_tombstones (
                memory_id, deleted_at, reason, node_type, tags,
                edges_pruned, insights_rewritten, insights_deleted, children_orphaned
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(memory_id) DO UPDATE SET
                deleted_at = excluded.deleted_at,
                reason = excluded.reason,
                node_type = excluded.node_type,
                tags = excluded.tags,
                edges_pruned = excluded.edges_pruned,
                insights_rewritten = excluded.insights_rewritten,
                insights_deleted = excluded.insights_deleted,
                children_orphaned = excluded.children_orphaned",
                params![
                    tombstone_marker,
                    deleted_at.to_rfc3339(),
                    Option::<&str>::None,
                    node.node_type,
                    "[]",
                    edges_pruned,
                    insights_rewritten,
                    insights_deleted,
                    children_orphaned,
                ],
            )?;
            Self::record_sync_tombstone(tx, "knowledge_nodes", id)?;
        }
        tx.execute("DELETE FROM knowledge_nodes WHERE id = ?1", params![id])?;

        Ok(Some(PurgeCleanup {
            edges_pruned,
            insights_rewritten,
            insights_deleted,
            children_orphaned,
        }))
    }

    fn node_exists(conn: &Connection, id: &str) -> Result<bool> {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM knowledge_nodes WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    fn record_sync_tombstone(conn: &Connection, table_name: &str, row_id: &str) -> Result<()> {
        let tombstone_row_id = if table_name == "knowledge_nodes" {
            Self::opaque_tombstone_marker(row_id)
        } else {
            row_id.to_string()
        };
        conn.execute(
            "INSERT INTO sync_tombstones (table_name, row_id, deleted_at, reason)
             VALUES (?1, ?2, ?3, NULL)
             ON CONFLICT(table_name, row_id) DO UPDATE SET
                deleted_at = excluded.deleted_at,
                reason = excluded.reason",
            params![table_name, tombstone_row_id, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    /// Deterministic, domain-separated marker for legacy deletion/sync rows.
    /// Knowledge-node UUIDs are not content, but persisting them makes deletion
    /// history linkable to a removed record and exposes them in portable
    /// archives. The marker is enough to match an already-local UUID during
    /// merge without retaining the UUID itself.
    fn opaque_tombstone_marker(memory_id: &str) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vestige.legacy-tombstone-marker.v1\\0");
        hasher.update(memory_id.as_bytes());
        format!("opaque:{}", hasher.finalize().to_hex())
    }

    fn resolve_tombstone_memory_id(
        tx: &rusqlite::Transaction<'_>,
        tombstone_row_id: &str,
    ) -> Result<Option<String>> {
        // Older archives retain raw ids. Keep their import behavior intact
        // while ensuring all newly produced tombstones are opaque.
        if !tombstone_row_id.starts_with("opaque:") {
            return Ok(Some(tombstone_row_id.to_string()));
        }
        let mut statement = tx.prepare("SELECT id FROM knowledge_nodes")?;
        let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
        for row in rows {
            let id = row?;
            if Self::opaque_tombstone_marker(&id) == tombstone_row_id {
                return Ok(Some(id));
            }
        }
        Ok(None)
    }

    /// Search with full-text search
    pub fn search(&self, query: &str, limit: i32) -> Result<Vec<KnowledgeNode>> {
        // OR-of-tokens + BM25 rank: matches rows sharing ANY distinctive token,
        // ranked by lexical relevance. (The old whole-string phrase match required
        // all tokens adjacent and in order, so multi-word queries returned nothing.)
        let Some(sanitized_query) = sanitize_fts5_or_query(query) else {
            return Ok(Vec::new());
        };

        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT n.* FROM knowledge_nodes n
             JOIN knowledge_fts fts ON n.id = fts.id
             WHERE knowledge_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;

        let nodes = stmt.query_map(params![sanitized_query, limit], Self::row_to_node)?;

        let mut result = Vec::new();
        for node in nodes {
            result.push(node?);
        }
        Ok(result)
    }

    /// FTS5 keyword search using individual-term matching (implicit AND).
    ///
    /// Unlike `search()` which uses phrase matching (words must be adjacent),
    /// this returns documents containing ALL query words in any order and position.
    /// This is more useful for free-text queries from external callers.
    pub fn search_terms(&self, query: &str, limit: i32) -> Result<Vec<KnowledgeNode>> {
        use crate::fts::sanitize_fts5_terms;
        let Some(terms) = sanitize_fts5_terms(query) else {
            return Ok(vec![]);
        };

        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT n.* FROM knowledge_nodes n
             JOIN knowledge_fts fts ON n.id = fts.id
             WHERE knowledge_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;

        let nodes = stmt.query_map(params![terms, limit], Self::row_to_node)?;

        let mut result = Vec::new();
        for node in nodes {
            result.push(node?);
        }
        Ok(result)
    }

    /// Concrete keyword/literal search that skips semantic expansion and
    /// cognitive reranking.
    ///
    /// This path is for identifiers, paths, quoted strings, env vars, UUIDs,
    /// and other exact user intent where "close enough" is wrong.
    pub fn concrete_search_filtered(
        &self,
        query: &str,
        limit: i32,
        include_types: Option<&[String]>,
        exclude_types: Option<&[String]>,
    ) -> Result<Vec<SearchResult>> {
        let literal = Self::normalize_literal_query(query);
        if literal.is_empty() {
            return Ok(vec![]);
        }

        let limit = limit.max(1) as usize;
        let fetch_limit = ((limit * 10).min(500)) as i32;
        let mut by_id: HashMap<String, SearchResult> = HashMap::new();

        if let Some(terms) = crate::fts::sanitize_fts5_terms(&literal) {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            let mut stmt = reader.prepare(
                "SELECT n.*, rank AS fts_rank FROM knowledge_nodes n
                 JOIN knowledge_fts fts ON n.id = fts.id
                 WHERE knowledge_fts MATCH ?1
                 ORDER BY rank
                 LIMIT ?2",
            )?;

            let rows = stmt.query_map(params![terms, fetch_limit], |row| {
                let node = Self::row_to_node(row)?;
                let rank = row.get::<_, f64>("fts_rank").unwrap_or(0.0);
                Ok((node, rank))
            })?;

            // Collect first, then NORMALIZE. Raw BM25 magnitude is unbounded,
            // while literal_match_score returns fixed constants in 1.2..=3.0, and
            // both land in the same `combined_score`. Measured on a 202-document
            // corpus: a note that merely CITES a UUID three times scores 27.5,
            // against the fixed 3.0 given to the memory whose id IS that UUID --
            // so on the documented exact-lookup path the thing you asked for was
            // routinely outranked by something that only mentions it, 9x over.
            //
            // Map the FTS leg into 0.0..=1.0, strictly below the 1.2 literal
            // floor. Relative BM25 ordering is preserved among pure keyword hits,
            // but any literal match now outranks any non-literal one.
            let scored_rows: Vec<(KnowledgeNode, f64)> = rows
                .filter_map(warn_skipped_row("concrete_search_filtered"))
                .filter(|(node, _)| {
                    Self::node_matches_type_filters(node, include_types, exclude_types)
                })
                .collect();
            let max_magnitude = scored_rows
                .iter()
                .map(|(_, rank)| (-*rank as f32).max(0.0))
                .fold(0.0_f32, f32::max);
            const FTS_BAND_TOP: f32 = 1.0; // < LITERAL_FLOOR (1.2)
            for (idx, (node, rank)) in scored_rows.into_iter().enumerate() {
                let magnitude = (-rank as f32).max(0.0);
                let base_score = if max_magnitude > 0.0 {
                    (magnitude / max_magnitude) * FTS_BAND_TOP
                } else {
                    // No usable BM25 (e.g. a term present in every row): fall back
                    // to rank order, still inside the band.
                    FTS_BAND_TOP / (idx as f32 + 1.0)
                };
                Self::upsert_concrete_result(&mut by_id, node, base_score, Some(base_score));
            }
        }

        let escaped = Self::escape_like(&literal.to_lowercase());
        let pattern = format!("%{}%", escaped);
        let prefix_pattern = format!("{}%", escaped);
        {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            let mut stmt = reader.prepare(
                "SELECT n.* FROM knowledge_nodes n
                 WHERE lower(n.id) = ?2
                    OR lower(n.content) LIKE ?1 ESCAPE '\\'
                    OR lower(COALESCE(n.source, '')) LIKE ?1 ESCAPE '\\'
                    OR lower(n.tags) LIKE ?1 ESCAPE '\\'
                 ORDER BY
                    CASE
                        WHEN lower(n.id) = ?2 THEN 0
                        WHEN lower(n.content) = ?2 THEN 1
                        WHEN lower(n.content) LIKE ?3 ESCAPE '\\' THEN 2
                        ELSE 3
                    END,
                    n.updated_at DESC
                 LIMIT ?4",
            )?;

            let rows = stmt.query_map(
                params![pattern, literal.to_lowercase(), prefix_pattern, fetch_limit],
                Self::row_to_node,
            )?;

            for row in rows {
                let node = row?;
                if !Self::node_matches_type_filters(&node, include_types, exclude_types) {
                    continue;
                }
                if let Some(score) = Self::literal_match_score(&literal, &node) {
                    Self::upsert_concrete_result(&mut by_id, node, score, Some(score));
                }
            }
        }

        let mut results: Vec<SearchResult> = by_id.into_values().collect();
        results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.node.updated_at.cmp(&a.node.updated_at))
        });
        results.truncate(limit);
        Ok(results)
    }

    fn upsert_concrete_result(
        by_id: &mut HashMap<String, SearchResult>,
        node: KnowledgeNode,
        score: f32,
        keyword_score: Option<f32>,
    ) {
        by_id
            .entry(node.id.clone())
            .and_modify(|existing| {
                existing.combined_score = existing.combined_score.max(score);
                existing.keyword_score = match (existing.keyword_score, keyword_score) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (None, Some(b)) => Some(b),
                    (a, None) => a,
                };
            })
            .or_insert(SearchResult {
                node,
                keyword_score,
                semantic_score: None,
                combined_score: score,
                match_type: MatchType::Keyword,
            });
    }

    fn normalize_literal_query(query: &str) -> String {
        let trimmed = query.trim();
        if trimmed.len() >= 2 {
            let bytes = trimmed.as_bytes();
            let quoted = (bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"')
                || (bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\'');
            if quoted {
                return trimmed[1..trimmed.len() - 1].trim().to_string();
            }
        }
        trimmed.to_string()
    }

    fn escape_like(value: &str) -> String {
        let mut escaped = String::with_capacity(value.len());
        for ch in value.chars() {
            match ch {
                '\\' | '%' | '_' => {
                    escaped.push('\\');
                    escaped.push(ch);
                }
                _ => escaped.push(ch),
            }
        }
        escaped
    }

    fn literal_match_score(query: &str, node: &KnowledgeNode) -> Option<f32> {
        let q = query.to_lowercase();
        let content = node.content.to_lowercase();
        let tags = node.tags.join(" ").to_lowercase();
        let source = node.source.as_deref().unwrap_or("").to_lowercase();
        let id = node.id.to_lowercase();

        if id == q {
            Some(3.0)
        } else if content == q {
            Some(2.5)
        } else if content.starts_with(&q) {
            Some(2.0)
        } else if content.contains(&q) {
            Some(1.6)
        } else if source.contains(&q) {
            Some(1.4)
        } else if tags.contains(&q) {
            Some(1.2)
        } else {
            None
        }
    }

    fn node_matches_type_filters(
        node: &KnowledgeNode,
        include_types: Option<&[String]>,
        exclude_types: Option<&[String]>,
    ) -> bool {
        if let Some(includes) = include_types
            && !includes.is_empty()
        {
            return includes.iter().any(|t| t == &node.node_type);
        }
        if let Some(excludes) = exclude_types
            && !excludes.is_empty()
        {
            return !excludes.iter().any(|t| t == &node.node_type);
        }
        true
    }

    /// Get all nodes (paginated)
    pub fn get_all_nodes(&self, limit: i32, offset: i32) -> Result<Vec<KnowledgeNode>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT * FROM knowledge_nodes
             ORDER BY created_at DESC
             LIMIT ?1 OFFSET ?2",
        )?;

        let nodes = stmt.query_map(params![limit, offset], Self::row_to_node)?;

        let mut result = Vec::new();
        for node in nodes {
            result.push(node?);
        }
        Ok(result)
    }

    /// Read the complete metadata population for hygiene aggregation in one
    /// query. Content is bounded to a short preview; access history is reduced
    /// with `NOT EXISTS` in SQL, avoiding both full-body loads and N+1 reads.
    /// `None` is an explicit all-scopes request. `Some(scope)` uses the same
    /// legacy-compatible normalized predicate as the tag-maintenance scans.
    ///
    /// Access classification is honest about the pruned log: `never_accessed`
    /// requires zero log rows AND zero durable retrieval counters
    /// (`times_retrieved`/`times_useful`) AND creation inside the retained
    /// [`ACCESS_LOG_RETENTION_DAYS`] window. Older rows without durable
    /// evidence are reported as `access_unknown` instead — their pre-prune
    /// access history is unknowable, never provably absent.
    ///
    /// Malformed legacy rows (NULL/unparseable `tags`, NULL
    /// `retention_strength`) are tolerated exactly like `row_to_node` and
    /// surfaced as counts, so one hand-edited row cannot abort the stats view.
    pub fn hygiene_snapshot(&self, scope: Option<&str>) -> Result<HygieneSnapshot> {
        let scope = scope.map(Self::normalize_scope).transpose()?;
        let log_window_start = Utc::now() - Duration::days(ACCESS_LOG_RETENTION_DAYS);
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let sql = if scope.is_some() {
            "SELECT n.id, n.node_type, n.created_at, n.retention_strength, n.tags,
                    n.valid_from, n.valid_until, n.superseded_by IS NOT NULL,
                    length(CAST(n.content AS BLOB)), substr(n.content, 1, 240),
                    (NOT EXISTS (
                        SELECT 1 FROM memory_access_log AS access
                        WHERE access.node_id = n.id
                    ))
                    AND COALESCE(n.times_retrieved, 0) = 0
                    AND COALESCE(n.times_useful, 0) = 0
             FROM knowledge_nodes AS n
             WHERE COALESCE(NULLIF(trim(n.scope), ''), 'user') = ?1
             ORDER BY n.id"
        } else {
            "SELECT n.id, n.node_type, n.created_at, n.retention_strength, n.tags,
                    n.valid_from, n.valid_until, n.superseded_by IS NOT NULL,
                    length(CAST(n.content AS BLOB)), substr(n.content, 1, 240),
                    (NOT EXISTS (
                        SELECT 1 FROM memory_access_log AS access
                        WHERE access.node_id = n.id
                    ))
                    AND COALESCE(n.times_retrieved, 0) = 0
                    AND COALESCE(n.times_useful, 0) = 0
             FROM knowledge_nodes AS n
             ORDER BY n.id"
        };
        let mut stmt = reader.prepare(sql)?;
        let mut rows = match scope {
            Some(scope) => stmt.query(params![scope])?,
            None => stmt.query([])?,
        };
        let mut summaries = Vec::new();
        let mut malformed_tag_rows = 0usize;
        let mut malformed_tag_row_ids = Vec::new();
        let mut malformed_tag_row_ids_truncated = false;
        let mut defaulted_retention_rows = 0usize;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let parsed_tags = match row.get::<_, Option<String>>(4)? {
                Some(tags_raw) => match serde_json::from_str::<Vec<String>>(&tags_raw) {
                    Ok(tags) => Some(tags),
                    Err(error) => {
                        tracing::warn!(
                            memory_id = %id,
                            "hygiene snapshot: unparseable tags JSON, treating as untagged: {error}"
                        );
                        None
                    }
                },
                None => None,
            };
            let tags = parsed_tags.unwrap_or_else(|| {
                malformed_tag_rows += 1;
                if malformed_tag_row_ids.len() < MAX_MALFORMED_TAG_ROW_IDS {
                    malformed_tag_row_ids.push(id.clone());
                } else {
                    malformed_tag_row_ids_truncated = true;
                }
                Vec::new()
            });
            let retention_strength = match row.get::<_, Option<f64>>(3)? {
                Some(value) => value,
                None => {
                    // The column is nullable and hand-edited stores can hold
                    // NULL; report those rows at the schema default of 1.0.
                    defaulted_retention_rows += 1;
                    1.0
                }
            };
            let valid_from = row
                .get::<_, Option<String>>(5)?
                .map(|value| Self::parse_timestamp(&value, "valid_from"))
                .transpose()?;
            let valid_until = row
                .get::<_, Option<String>>(6)?
                .map(|value| Self::parse_timestamp(&value, "valid_until"))
                .transpose()?;
            let created_at = Self::parse_timestamp(&row.get::<_, String>(2)?, "created_at")?;
            let no_access_evidence: bool = row.get(10)?;
            let created_inside_log_window = created_at >= log_window_start;
            summaries.push(HygieneNodeSummary {
                id,
                node_type: row.get(1)?,
                created_at,
                retention_strength,
                tags,
                valid_from,
                valid_until,
                superseded: row.get(7)?,
                content_bytes: row.get::<_, i64>(8)?.max(0) as usize,
                content_preview: row.get(9)?,
                never_accessed: no_access_evidence && created_inside_log_window,
                access_unknown: no_access_evidence && !created_inside_log_window,
            });
        }
        Ok(HygieneSnapshot {
            nodes: summaries,
            malformed_tag_rows,
            malformed_tag_row_ids,
            malformed_tag_row_ids_truncated,
            defaulted_retention_rows,
        })
    }

    /// Return the complete, exact tag vocabulary for one scope (or all scopes
    /// when explicitly requested). This powers non-mutating ingest nudges; it
    /// parses JSON arrays rather than relying on substring SQL matching.
    ///
    /// Stored tags longer than the 200-character similarity safety limit are
    /// skipped and counted instead of erroring the whole scope, mirroring how
    /// overlong INPUT tags and secret-shaped vocabulary tags already degrade
    /// gracefully. The 10,000-tag vocabulary bound stays a hard error and is
    /// evaluated over the remaining (eligible) vocabulary, so skipping
    /// overlong tags can never mask it.
    pub fn tag_vocabulary(&self, scope: Option<&str>) -> Result<TagVocabulary> {
        const MAX_TAG_VOCABULARY: usize = 10_000;
        let scope = scope.map(Self::normalize_scope).transpose()?;
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let overlong_sql = if scope.is_some() {
            "SELECT COUNT(DISTINCT tags.value)
             FROM knowledge_nodes AS node, json_each(node.tags) AS tags
             WHERE COALESCE(NULLIF(trim(node.scope), ''), 'user') = ?1
               AND tags.type = 'text'
               AND length(tags.value) > 200"
        } else {
            "SELECT COUNT(DISTINCT tags.value)
             FROM knowledge_nodes AS node, json_each(node.tags) AS tags
             WHERE tags.type = 'text'
               AND length(tags.value) > 200"
        };
        let skipped_overlong: i64 = match scope {
            Some(scope) => reader.query_row(overlong_sql, params![scope], |row| row.get(0))?,
            None => reader.query_row(overlong_sql, [], |row| row.get(0))?,
        };
        let sql = if scope.is_some() {
            "SELECT DISTINCT tags.value
             FROM knowledge_nodes AS node, json_each(node.tags) AS tags
             WHERE COALESCE(NULLIF(trim(node.scope), ''), 'user') = ?1
               AND tags.type = 'text'
               AND length(tags.value) <= 200
             ORDER BY tags.value
             LIMIT 10001"
        } else {
            "SELECT DISTINCT tags.value
             FROM knowledge_nodes AS node, json_each(node.tags) AS tags
             WHERE tags.type = 'text'
               AND length(tags.value) <= 200
             ORDER BY tags.value
             LIMIT 10001"
        };
        let mut stmt = reader.prepare(sql)?;
        let mut rows = match scope {
            Some(scope) => stmt.query(params![scope])?,
            None => stmt.query([])?,
        };
        let mut vocabulary = Vec::new();
        while let Some(row) = rows.next()? {
            vocabulary.push(row.get(0)?);
        }
        if vocabulary.len() > MAX_TAG_VOCABULARY {
            return Err(StorageError::Init(format!(
                "tag vocabulary exceeds the {MAX_TAG_VOCABULARY}-tag similarity safety limit"
            )));
        }
        Ok(TagVocabulary {
            tags: vocabulary,
            skipped_overlong: skipped_overlong.max(0) as usize,
        })
    }

    /// Get nodes by type and optional tag filter
    ///
    /// This is used for codebase context retrieval where we need to query
    /// by node_type (pattern/decision) and filter by codebase tag.
    pub fn get_nodes_by_type_and_tag(
        &self,
        node_type: &str,
        tag_filter: Option<&str>,
        limit: i32,
    ) -> Result<Vec<KnowledgeNode>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        match tag_filter {
            Some(tag) => {
                // Query with tag filter using JSON LIKE search
                // Tags are stored as JSON array, e.g., '["pattern", "codebase", "codebase:vestige"]'
                let tag_pattern = format!("%\"{}%", tag);
                let mut stmt = reader.prepare(
                    "SELECT * FROM knowledge_nodes
                     WHERE node_type = ?1
                     AND tags LIKE ?2
                     ORDER BY retention_strength DESC, created_at DESC
                     LIMIT ?3",
                )?;
                let rows = stmt.query_map(params![node_type, tag_pattern, limit], |row| {
                    Self::row_to_node(row)
                })?;
                let mut nodes = Vec::new();
                for node in rows.flatten() {
                    nodes.push(node);
                }
                Ok(nodes)
            }
            None => {
                // Query without tag filter
                let mut stmt = reader.prepare(
                    "SELECT * FROM knowledge_nodes
                     WHERE node_type = ?1
                     ORDER BY retention_strength DESC, created_at DESC
                     LIMIT ?2",
                )?;
                let rows = stmt.query_map(params![node_type, limit], Self::row_to_node)?;
                let mut nodes = Vec::new();
                for node in rows.flatten() {
                    nodes.push(node);
                }
                Ok(nodes)
            }
        }
    }

    /// Attach a verified process-local runtime to the currently active profile.
    ///
    /// This deliberately stores no artifact path and never initializes or
    /// downloads a model. The profile contract must exactly match the persisted
    /// active profile, which prevents a runner for one vector space from being
    /// used to query another.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub(crate) fn attach_active_profile_embedder(
        &self,
        profile_id: &EmbeddingProfileId,
        embedder: Arc<ProfiledEmbedder>,
    ) -> Result<()> {
        let active = self.active_embedding_profile()?.ok_or_else(|| {
            StorageError::InvalidEmbeddingProfile("no active embedding profile pointer".to_string())
        })?;
        if active.profile_id != *profile_id {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "cannot attach runtime for '{}' while '{}' is active",
                profile_id, active.profile_id
            )));
        }
        let manifest = self
            .embedding_profile_manifest(profile_id)?
            .ok_or_else(|| StorageError::NotFound(profile_id.to_string()))?;
        if manifest.profile != *embedder.profile()
            || manifest.verification.status != VerificationStatus::Verified
            || manifest
                .runtime
                .as_ref()
                .is_none_or(|runtime| !runtime.local_only)
        {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "profile '{}' does not have a matching verified local runtime contract",
                profile_id
            )));
        }
        let mut attached = self.attached_profile_runtime.write().map_err(|_| {
            StorageError::Init("Attached profile runtime lock poisoned".to_string())
        })?;
        *attached = Some(AttachedProfileRuntime {
            profile_id: profile_id.clone(),
            embedder,
        });
        if let Some(cache) = &self.query_cache {
            cache
                .lock()
                .map_err(|_| StorageError::Init("Query cache lock poisoned".to_string()))?
                .clear();
        }
        Ok(())
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn attached_embedder_for(
        &self,
        profile_id: &EmbeddingProfileId,
    ) -> Result<Option<Arc<ProfiledEmbedder>>> {
        let attached = self.attached_profile_runtime.read().map_err(|_| {
            StorageError::Init("Attached profile runtime lock poisoned".to_string())
        })?;
        Ok(attached.as_ref().and_then(|runtime| {
            (runtime.profile_id == *profile_id).then(|| runtime.embedder.clone())
        }))
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn active_embedding_runtime_ready(&self) -> Result<bool> {
        let active = self.active_embedding_profile()?.ok_or_else(|| {
            StorageError::InvalidEmbeddingProfile("no active embedding profile pointer".to_string())
        })?;
        let manifest = self
            .embedding_profile_manifest(&active.profile_id)?
            .ok_or_else(|| StorageError::NotFound(active.profile_id.to_string()))?;
        if self.attached_embedder_for(&active.profile_id)?.is_some() {
            return Ok(true);
        }
        Ok(
            manifest.profile.runtime_backend != EmbeddingRuntimeBackend::FastembedCandle
                && self.embedding_service.is_ready(),
        )
    }

    /// Check if the active profile has a usable local query runtime. Optional
    /// Qwen profiles return false until a verified runner is explicitly
    /// attached to this process.
    #[cfg(feature = "embeddings")]
    pub fn is_embedding_ready(&self) -> bool {
        #[cfg(feature = "vector-search")]
        {
            self.active_embedding_runtime_ready().unwrap_or(false)
        }
        #[cfg(not(feature = "vector-search"))]
        self.embedding_service.is_ready()
    }

    #[cfg(not(feature = "embeddings"))]
    pub fn is_embedding_ready(&self) -> bool {
        false
    }

    /// Initialize the released Nomic default without widening optional profile
    /// activation into an implicit model-selection path.
    ///
    /// Existing installs have always initialized the active legacy Nomic
    /// runtime from normal CLI/MCP startup. Preserve that contract exactly.
    /// Every non-legacy profile, including all Qwen variants, remains an
    /// explicit artifact-backed workflow and cannot be initialized here.
    #[cfg(feature = "embeddings")]
    pub fn init_embeddings(&self) -> Result<()> {
        let active = self.active_embedding_profile()?.ok_or_else(|| {
            StorageError::InvalidEmbeddingProfile("no active embedding profile pointer".to_string())
        })?;
        if active.profile_id.as_str() != LEGACY_EMBEDDING_PROFILE_ID {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "direct embedding initialization is supported only for the released legacy Nomic profile; '{}' requires the explicit profile workflow",
                active.profile_id
            )));
        }
        self.embedding_service.init().map_err(|error| {
            StorageError::Init(format!("Initialize legacy Nomic embeddings: {error}"))
        })
    }

    #[cfg(not(feature = "embeddings"))]
    pub fn init_embeddings(&self) -> Result<()> {
        Ok(()) // No-op when embeddings feature is disabled
    }

    /// Get query embedding from cache or compute it
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn get_query_embedding(&self, query: &str) -> Result<Vec<f32>> {
        let active = self.active_embedding_profile()?.ok_or_else(|| {
            StorageError::InvalidEmbeddingProfile("no active embedding profile pointer".to_string())
        })?;
        let manifest = self
            .embedding_profile_manifest(&active.profile_id)?
            .ok_or_else(|| StorageError::NotFound(active.profile_id.to_string()))?;
        let cache_key = format!("{}\0{}", active.profile_id, query);
        // Check cache first
        let Some(index_cache) = self.query_cache.as_ref() else {
            return Err(StorageError::Init("Query cache unavailable".to_string()));
        };
        {
            let mut cache = index_cache
                .lock()
                .map_err(|_| StorageError::Init("Query cache lock poisoned".to_string()))?;
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }

        // Never fall back from an active optional profile to the legacy
        // service. Qwen vectors and Nomic vectors are different spaces; a
        // missing explicit attachment is an availability error, not permission
        // to issue a semantically invalid query.
        let vector = if let Some(embedder) = self.attached_embedder_for(&active.profile_id)? {
            let runtime = tokio::runtime::Runtime::new().map_err(|error| {
                StorageError::Init(format!("Create local query runtime: {error}"))
            })?;
            runtime
                .block_on(embedder.embed_query(query))
                .map_err(|error| StorageError::Init(format!("Failed to embed query: {error}")))?
        } else if manifest.profile.runtime_backend == EmbeddingRuntimeBackend::FastembedCandle {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "active profile '{}' requires an explicitly attached verified local runtime; supply its artifact directory for this process",
                active.profile_id
            )));
        } else {
            self.embedding_service
                .embed(
                    &manifest.profile.encode_query(query).map_err(|error| {
                        StorageError::InvalidEmbeddingProfile(error.to_string())
                    })?,
                )
                .map_err(|e| StorageError::Init(format!("Failed to embed query: {e}")))?
                .vector
        };
        if vector.len() != manifest.profile.embedding_dimension {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "active profile '{}' requires {} dimensions but its runtime produced {}",
                active.profile_id,
                manifest.profile.embedding_dimension,
                vector.len()
            )));
        }

        // Store in cache
        {
            let mut cache = index_cache
                .lock()
                .map_err(|_| StorageError::Init("Query cache lock poisoned".to_string()))?;
            cache.put(cache_key, vector.clone());
        }

        Ok(vector)
    }

    /// Compute one document vector for the active profile without populating
    /// the query cache. Document and query templates are distinct parts of a
    /// profile contract, particularly for Qwen retrieval profiles.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn get_document_embedding(&self, content: &str) -> Result<Vec<f32>> {
        let active = self.active_embedding_profile()?.ok_or_else(|| {
            StorageError::InvalidEmbeddingProfile("no active embedding profile pointer".to_string())
        })?;
        let manifest = self
            .embedding_profile_manifest(&active.profile_id)?
            .ok_or_else(|| StorageError::NotFound(active.profile_id.to_string()))?;
        let vector = if let Some(embedder) = self.attached_embedder_for(&active.profile_id)? {
            let runtime = tokio::runtime::Runtime::new().map_err(|error| {
                StorageError::Init(format!("Create local document runtime: {error}"))
            })?;
            runtime
                .block_on(embedder.embed_document(content))
                .map_err(|error| StorageError::Init(format!("Failed to embed document: {error}")))?
        } else if manifest.profile.runtime_backend == EmbeddingRuntimeBackend::FastembedCandle {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "active profile '{}' requires an explicitly attached verified local runtime; supply its artifact directory for this process",
                active.profile_id
            )));
        } else {
            self.embedding_service
                .embed(
                    &manifest.profile.encode_document(content).map_err(|error| {
                        StorageError::InvalidEmbeddingProfile(error.to_string())
                    })?,
                )
                .map_err(|error| StorageError::Init(format!("Failed to embed document: {error}")))?
                .vector
        };
        if vector.len() != manifest.profile.embedding_dimension {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "active profile '{}' requires {} dimensions but its runtime produced {}",
                active.profile_id,
                manifest.profile.embedding_dimension,
                vector.len()
            )));
        }
        Ok(vector)
    }

    /// Semantic search
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn semantic_search(
        &self,
        query: &str,
        limit: i32,
        min_similarity: f32,
    ) -> Result<Vec<SimilarityResult>> {
        let Some(index_lock) = self.vector_index.as_ref() else {
            return Err(StorageError::Init(
                "Vector search unavailable: disabled for this machine".to_string(),
            ));
        };

        if !self.active_embedding_runtime_ready()? {
            return Err(StorageError::Init("Embedding model not ready".to_string()));
        }

        let query_embedding = self.get_query_embedding(query)?;

        let index = index_lock
            .lock()
            .map_err(|_| StorageError::Init("Vector index lock poisoned".to_string()))?;

        let results = index
            .search_with_threshold(&query_embedding, limit as usize, min_similarity)
            .map_err(|e| StorageError::Init(format!("Vector search failed: {}", e)))?;

        let mut similarity_results = Vec::with_capacity(results.len());

        for (node_id, similarity) in results {
            if let Some(node) = self.get_node(&node_id)? {
                similarity_results.push(SimilarityResult { node, similarity });
            }
        }

        Ok(similarity_results)
    }

    /// Hybrid search (delegates to hybrid_search_filtered with no type filters)
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn hybrid_search(
        &self,
        query: &str,
        limit: i32,
        keyword_weight: f32,
        semantic_weight: f32,
    ) -> Result<Vec<SearchResult>> {
        self.hybrid_search_filtered(query, limit, keyword_weight, semantic_weight, None, None)
    }

    /// Hybrid search with optional type filtering pushed into the storage layer.
    ///
    /// When `include_types` is `Some`, only nodes whose `node_type` matches one of
    /// the given strings are returned. When `exclude_types` is `Some`, nodes whose
    /// `node_type` matches are excluded. `include_types` takes precedence over
    /// `exclude_types`. Both are case-sensitive and compared against the stored
    /// `node_type` value.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn hybrid_search_filtered(
        &self,
        query: &str,
        limit: i32,
        keyword_weight: f32,
        semantic_weight: f32,
        include_types: Option<&[String]>,
        exclude_types: Option<&[String]>,
    ) -> Result<Vec<SearchResult>> {
        let has_type_filter = include_types.is_some() || exclude_types.is_some();
        // Over-fetch more aggressively when type filters are active so that
        // after filtering we still have enough candidates to fill `limit`.
        let overfetch_factor = if has_type_filter { 4 } else { 2 };

        let keyword_results = self.keyword_search_with_scores(
            query,
            limit * overfetch_factor,
            include_types,
            exclude_types,
        )?;

        let semantic_results =
            if self.vector_search_available() && self.active_embedding_runtime_ready()? {
                self.semantic_search_raw(query, limit * overfetch_factor)?
            } else {
                vec![]
            };

        // Reciprocal Rank Fusion (k=60) when both lists are present: it is scale-free
        // and rewards a memory that appears in BOTH the keyword and semantic lists —
        // exactly the structurally-similar-different-words paraphrase that linear
        // max-norm fusion buried. Falls back to linear when only one list exists.
        // (keyword_weight/semantic_weight retained in the signature for compatibility;
        // RRF is rank-based so the weights no longer scale the fused score.)
        let _ = (keyword_weight, semantic_weight);
        let combined = if !semantic_results.is_empty() {
            reciprocal_rank_fusion(&keyword_results, &semantic_results, 60.0)
        } else {
            keyword_results.clone()
        };

        let mut results = Vec::with_capacity(limit as usize);

        for (node_id, combined_score) in combined.into_iter() {
            if results.len() >= limit as usize {
                break;
            }
            if let Some(node) = self.get_node(&node_id)? {
                // Apply type filtering for results that came from semantic search
                // (keyword search already filters in SQL, but semantic search cannot)
                if let Some(includes) = include_types {
                    if !includes.iter().any(|t| t == &node.node_type) {
                        continue;
                    }
                } else if let Some(excludes) = exclude_types
                    && excludes.iter().any(|t| t == &node.node_type)
                {
                    continue;
                }
                let keyword_score = keyword_results
                    .iter()
                    .find(|(id, _)| id == &node_id)
                    .map(|(_, s)| *s);
                let semantic_score = semantic_results
                    .iter()
                    .find(|(id, _)| id == &node_id)
                    .map(|(_, s)| *s);

                let match_type = match (keyword_score.is_some(), semantic_score.is_some()) {
                    (true, true) => MatchType::Both,
                    (true, false) => MatchType::Keyword,
                    (false, true) => MatchType::Semantic,
                    (false, false) => MatchType::Keyword,
                };

                // Carry the RRF fused score as the relevance signal, NOT a linear
                // kw*w + sem*w recomputation. RRF is what selected these candidates
                // and rewards both-list agreement; overwriting it with the linear
                // weighted_score made the final ranking diverge from RRF order
                // (a both-list paraphrase could rank below a keyword-only hit).
                // The min-max normalization in the rerank below then operates on
                // RRF scores, so final relevance ordering matches RRF ordering.
                results.push(SearchResult {
                    node,
                    keyword_score,
                    semantic_score,
                    combined_score,
                    match_type,
                });
            }
        }

        // Three-signal reranking (Park et al. Generative Agents 2023)
        // final_score = 0.2*recency + 0.3*importance + 0.5*relevance
        //
        // relevance MUST live in [0,1] for the weights to balance. The raw
        // weighted_score does not: keyword-only results max out at
        // `1.0 * keyword_weight` (0.3 by default), so the strongest match's
        // relevance term was capped at 0.5*0.3 = 0.15 and lost to recency (up to
        // 0.2) or importance (up to 0.3) — a fresh, weakly-relevant node could
        // outrank the best match. Min-max normalize relevance across the result
        // set so the best match scores ~1.0 regardless of the weight scaling.
        let (min_rel, max_rel) = results
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), r| {
                (mn.min(r.combined_score), mx.max(r.combined_score))
            });
        let rel_span = (max_rel - min_rel) as f64;

        let now = Utc::now();
        for result in &mut results {
            let hours_since = (now - result.node.last_accessed).num_seconds() as f64 / 3600.0;
            let recency = 0.995_f64.powf(hours_since.max(0.0));

            // ACT-R activation as importance signal (pre-computed during consolidation)
            let activation: f64 = self
                .reader
                .lock()
                .map(|r| {
                    r.query_row(
                        "SELECT COALESCE(activation, 0.0) FROM knowledge_nodes WHERE id = ?1",
                        params![result.node.id],
                        |row| row.get(0),
                    )
                    .unwrap_or(0.0)
                })
                .unwrap_or(0.0);
            // Normalize ACT-R activation [-2, 5] → [0, 1]
            let importance = ((activation + 2.0) / 7.0).clamp(0.0, 1.0);

            // Min-max normalized relevance in [0,1]. When every result ties
            // (span 0), fall back to 1.0 so relevance still dominates ranking.
            let relevance = if rel_span > f64::EPSILON {
                (result.combined_score - min_rel) as f64 / rel_span
            } else {
                1.0
            };

            let final_score = 0.2 * recency + 0.3 * importance + 0.5 * relevance;
            result.combined_score = final_score as f32;
        }

        results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Keyword-only fallback for builds without local embeddings/vector search.
    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    pub fn hybrid_search(
        &self,
        query: &str,
        limit: i32,
        _keyword_weight: f32,
        _semantic_weight: f32,
    ) -> Result<Vec<SearchResult>> {
        self.hybrid_search_filtered(query, limit, 1.0, 0.0, None, None)
    }

    /// Keyword-only fallback for builds without local embeddings/vector search.
    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    pub fn hybrid_search_filtered(
        &self,
        query: &str,
        limit: i32,
        _keyword_weight: f32,
        _semantic_weight: f32,
        include_types: Option<&[String]>,
        exclude_types: Option<&[String]>,
    ) -> Result<Vec<SearchResult>> {
        let nodes = self.search_terms(query, limit.max(1) * 4)?;
        let mut results = Vec::new();

        for node in nodes {
            if let Some(includes) = include_types {
                if !includes.iter().any(|t| t == &node.node_type) {
                    continue;
                }
            } else if let Some(excludes) = exclude_types
                && excludes.iter().any(|t| t == &node.node_type)
            {
                continue;
            }

            let score = 1.0 / (results.len() as f32 + 1.0);
            results.push(SearchResult {
                node,
                keyword_score: Some(score),
                semantic_score: None,
                combined_score: score,
                match_type: MatchType::Keyword,
            });

            if results.len() >= limit.max(1) as usize {
                break;
            }
        }

        Ok(results)
    }

    /// Keyword search returning scores, with optional type filtering in the SQL query.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn keyword_search_with_scores(
        &self,
        query: &str,
        limit: i32,
        include_types: Option<&[String]>,
        exclude_types: Option<&[String]>,
    ) -> Result<Vec<(String, f32)>> {
        // Use individual-term matching (implicit AND) so multi-word queries find
        // documents where all words appear anywhere, not just as adjacent phrases.
        use crate::fts::sanitize_fts5_terms;
        let Some(terms_query) = sanitize_fts5_terms(query) else {
            return Ok(vec![]);
        };

        // Build the type filter clause and collect parameter values.
        // We use numbered parameters: ?1 = query, ?2 = limit, ?3.. = type strings.
        let mut type_clause = String::new();
        let type_values: Vec<&str>;

        if let Some(includes) = include_types {
            if !includes.is_empty() {
                let placeholders: Vec<String> =
                    (0..includes.len()).map(|i| format!("?{}", i + 3)).collect();
                type_clause = format!(" AND n.node_type IN ({})", placeholders.join(","));
                type_values = includes.iter().map(|s| s.as_str()).collect();
            } else {
                type_values = vec![];
            }
        } else if let Some(excludes) = exclude_types {
            if !excludes.is_empty() {
                let placeholders: Vec<String> =
                    (0..excludes.len()).map(|i| format!("?{}", i + 3)).collect();
                type_clause = format!(" AND n.node_type NOT IN ({})", placeholders.join(","));
                type_values = excludes.iter().map(|s| s.as_str()).collect();
            } else {
                type_values = vec![];
            }
        } else {
            type_values = vec![];
        }

        let sql = format!(
            "SELECT n.id, rank FROM knowledge_nodes n
             JOIN knowledge_fts fts ON n.id = fts.id
             WHERE knowledge_fts MATCH ?1{}
             ORDER BY rank
             LIMIT ?2",
            type_clause
        );

        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(&sql)?;

        // Build the parameter list: [query, limit, ...type_values]
        let mut param_values: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        param_values.push(Box::new(terms_query));
        param_values.push(Box::new(limit));
        for tv in &type_values {
            param_values.push(Box::new(tv.to_string()));
        }
        let params_ref: Vec<&dyn rusqlite::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let results: Vec<(String, f32)> = stmt
            .query_map(params_ref.as_slice(), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)? as f32))
            })?
            .filter_map(warn_skipped_row("keyword_search_with_scores"))
            .map(|(id, rank)| (id, (-rank).max(0.0)))
            .collect();

        if results.is_empty() {
            return Ok(vec![]);
        }

        let max_score = results.iter().map(|(_, s)| *s).fold(0.0_f32, f32::max);
        if max_score > 0.0 {
            Ok(results
                .into_iter()
                .map(|(id, s)| (id, s / max_score))
                .collect())
        } else {
            Ok(results)
        }
    }

    /// Bring the in-process vector index up to date with vectors written by OTHER
    /// processes since this one last looked. Returns the number of index
    /// mutations applied: vectors added, replaced or removed.
    ///
    /// THE BUG THIS FIXES (#181). The HNSW index is process-local: it is built once
    /// at startup from `embedding_profile_vectors` and thereafter only ever appended
    /// to by THIS process's own ingests. A second MCP server writing to the same
    /// SQLite file is therefore invisible to it. In a normal setup, a desktop
    /// client, an editor integration, a CLI and a dashboard all pointed at one store,
    /// every long-lived process is semantically blind to everything its peers have
    /// written since it booted. The consequences are silent: the prediction-error
    /// gate sees no similar candidate and creates a duplicate instead of reinforcing,
    /// and recall returns an incomplete answer with no indication anything is missing.
    /// The FTS5 leg reads SQLite directly and is unaffected, which is exactly why the
    /// failure is partial and hard to notice.
    ///
    /// THE SIGNAL. `PRAGMA data_version` is incremented on a connection whenever a
    /// DIFFERENT connection commits. Reading it is a single pragma with no table
    /// access, so this check is affordable on every query, and when nothing has
    /// changed it costs one integer comparison. It only says THAT something changed.
    ///
    /// WHAT CHANGED comes from `vector_journal` (migration V32). Three triggers append
    /// one row per insert, update or delete of `embedding_profile_vectors`, keyed by
    /// an AUTOINCREMENT `seq` that is allocated inside the writer's transaction: so it
    /// is monotonic in commit order, never reused, and independent of wall clocks.
    /// The index remembers the last `seq` it absorbed and reads exactly the rows past
    /// it. A peer re-embedding an existing node is an upsert row, so the stale vector
    /// is replaced; a peer's purge is a delete row, so the dead vector leaves the
    /// index. The first version of this refresh rescanned every vector row, blob
    /// included, on every external commit, and could not see re-embeddings at all
    /// because it skipped any id the index already held.
    ///
    /// RECONCILE. If the watermark is unknown, or the journal has been pruned past
    /// it, the index is compared against the table instead: one covering scan of
    /// node ids for the active profile, add what is missing, drop what is gone. That
    /// is O(N) over ids only, and it runs in exactly those two cases.
    ///
    /// LOCK DISCIPLINE. This acquires the reader lock, the watermark lock and the
    /// index lock SEQUENTIALLY and never holds two at once. `semantic_search_raw`
    /// holds only the index lock, so no ordering cycle exists and this cannot
    /// deadlock against it.
    ///
    /// FAILS OPEN. A refresh problem must degrade to a possibly-stale index, never
    /// break the query: returning an error here would turn a peer's write into an
    /// outage.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn refresh_vector_index_if_stale(&self) -> usize {
        let Some(index_mutex) = self.vector_index.as_ref() else {
            return 0;
        };

        // --- reader lock: has anyone else committed? ---
        let current_version: i64 = {
            let Ok(reader) = self.reader.lock() else {
                return 0;
            };
            match reader.query_row("PRAGMA data_version", [], |row| row.get(0)) {
                Ok(v) => v,
                Err(_) => return 0,
            }
        };
        // --- watermark lock: compare, and take the journal position ---
        let last_seq = {
            let Ok(mut watermark) = self.vector_index_watermark.lock() else {
                return 0;
            };
            if watermark.data_version == current_version {
                return 0; // nothing has changed since we last looked
            }
            watermark.data_version = current_version;
            watermark.journal_seq
        };

        let Ok(Some(active)) = self.active_embedding_profile() else {
            return 0;
        };
        let profile_id = active.profile_id.as_str();

        // --- reader lock, one snapshot: what changed past the watermark? ---
        let plan = {
            let Ok(reader) = self.reader.lock() else {
                return 0;
            };
            let Ok(snapshot) = begin_read_snapshot(&reader) else {
                return 0;
            };
            match Self::vector_refresh_plan(&snapshot, profile_id, last_seq) {
                Ok(plan) => plan,
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "vector index refresh could not read the journal; the index may be stale until the next query"
                    );
                    return 0;
                }
            }
        };

        let (changes, head) = match plan {
            VectorRefreshPlan::Reconcile => {
                return self.reconcile_vector_index(index_mutex, profile_id);
            }
            VectorRefreshPlan::Incremental { changes, head } => (changes, head),
        };

        // --- index lock: apply exactly what the journal named ---
        let mut applied = 0usize;
        if !changes.is_empty() {
            let Ok(mut index) = index_mutex.lock() else {
                return 0;
            };
            for (node_id, blob) in changes {
                let mutated = match blob {
                    None => matches!(index.remove(&node_id), Ok(true)),
                    Some(blob) => Self::add_journaled_vector(&mut index, &node_id, &blob),
                };
                if mutated {
                    applied += 1;
                }
            }
        }
        // --- watermark lock: current through `head` ---
        if let Ok(mut watermark) = self.vector_index_watermark.lock()
            && watermark.journal_seq < head
        {
            watermark.journal_seq = head;
        }
        if applied > 0 {
            tracing::debug!(
                applied,
                head,
                data_version = current_version,
                "refreshed vector index with memories written by another process"
            );
        }
        applied
    }

    /// Read the journal past `last_seq` for `profile_id` inside `snapshot`, and
    /// decide whether the index can follow it or must reconcile.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn vector_refresh_plan(
        snapshot: &Connection,
        profile_id: &str,
        last_seq: i64,
    ) -> rusqlite::Result<VectorRefreshPlan> {
        if last_seq < 0 {
            return Ok(VectorRefreshPlan::Reconcile);
        }
        let (oldest, head): (Option<i64>, i64) = snapshot.query_row(
            "SELECT MIN(seq), COALESCE(MAX(seq), 0) FROM vector_journal",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        // Pruned past us: rows between our watermark and the oldest survivor are
        // gone, or the journal was emptied after we had already seen rows.
        let pruned_past_us = match oldest {
            Some(oldest) => oldest > last_seq + 1,
            None => last_seq > 0,
        };
        if pruned_past_us {
            return Ok(VectorRefreshPlan::Reconcile);
        }

        // Last op per node wins; the journal is read in seq order.
        let mut latest: HashMap<String, bool> = HashMap::new();
        let mut stmt = snapshot.prepare(
            "SELECT node_id, op FROM vector_journal
             WHERE profile_id = ?1 AND seq > ?2
             ORDER BY seq",
        )?;
        let rows = stmt.query_map(params![profile_id, last_seq], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        for row in rows {
            let (node_id, op) = row?;
            latest.insert(node_id, op == "delete");
        }
        drop(stmt);

        let mut fetch = snapshot.prepare(
            "SELECT embedding FROM embedding_profile_vectors WHERE profile_id = ?1 AND node_id = ?2",
        )?;
        let mut changes = Vec::with_capacity(latest.len());
        for (node_id, deleted) in latest {
            if deleted {
                changes.push((node_id, None));
                continue;
            }
            // Same snapshot as the journal read, so an upsert whose row is
            // nevertheless absent can only be a later delete we also saw.
            let blob: Option<Vec<u8>> = fetch
                .query_row(params![profile_id, &node_id], |row| row.get(0))
                .optional()?;
            changes.push((node_id, blob));
        }
        Ok(VectorRefreshPlan::Incremental { changes, head })
    }

    /// Decode and add one journaled vector. Same decoder the startup builder
    /// uses, so a vector added here is identical to one added by a rebuild. A
    /// vector this index cannot hold (unreadable, wrong dimension) is skipped;
    /// the memory stays keyword-searchable and never fails a query.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn add_journaled_vector(index: &mut VectorIndex, node_id: &str, blob: &[u8]) -> bool {
        let Some(embedding) = Embedding::from_bytes(blob) else {
            tracing::warn!(node_id, "skipping an unreadable vector during index refresh");
            return false;
        };
        if embedding.dimensions != index.dimensions() {
            return false; // another profile's width: not ours to hold
        }
        index.add(node_id, &embedding.vector).is_ok()
    }

    /// Compare the index against `embedding_profile_vectors` for `profile_id`
    /// and fix the difference. Used when the journal cannot be trusted to be
    /// complete: an unknown watermark, or a journal pruned past it. O(N) over
    /// node ids (a covering index scan), fetching only the vectors that are
    /// missing. Returns the number of index mutations.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn reconcile_vector_index(&self, index_mutex: &Mutex<VectorIndex>, profile_id: &str) -> usize {
        // --- reader lock, one snapshot: every id the table holds, and the head ---
        let (present, head): (HashSet<String>, i64) = {
            let Ok(reader) = self.reader.lock() else {
                return 0;
            };
            let Ok(snapshot) = begin_read_snapshot(&reader) else {
                return 0;
            };
            let Ok(mut stmt) = snapshot
                .prepare("SELECT node_id FROM embedding_profile_vectors WHERE profile_id = ?1")
            else {
                return 0;
            };
            let Ok(rows) = stmt.query_map(params![profile_id], |row| row.get::<_, String>(0))
            else {
                return 0;
            };
            let present: HashSet<String> = rows
                .filter_map(warn_skipped_row("reconcile_vector_index"))
                .collect();
            drop(stmt);
            let head: i64 = match snapshot.query_row(
                "SELECT COALESCE(MAX(seq), 0) FROM vector_journal",
                [],
                |row| row.get(0),
            ) {
                Ok(head) => head,
                Err(_) => return 0,
            };
            (present, head)
        };

        // --- index lock: the two differences ---
        let (missing, gone): (Vec<String>, Vec<String>) = {
            let Ok(index) = index_mutex.lock() else {
                return 0;
            };
            let missing = present
                .iter()
                .filter(|id| !index.contains(id))
                .cloned()
                .collect();
            let gone = index
                .keys()
                .filter(|key| !present.contains(*key))
                .map(str::to_string)
                .collect();
            (missing, gone)
        };

        // --- reader lock: fetch only what is missing ---
        let blobs: Vec<(String, Vec<u8>)> = if missing.is_empty() {
            Vec::new()
        } else {
            let Ok(reader) = self.reader.lock() else {
                return 0;
            };
            let Ok(mut stmt) = reader.prepare(
                "SELECT embedding FROM embedding_profile_vectors WHERE profile_id = ?1 AND node_id = ?2",
            ) else {
                return 0;
            };
            missing
                .into_iter()
                .filter_map(|node_id| {
                    stmt.query_row(params![profile_id, &node_id], |row| row.get::<_, Vec<u8>>(0))
                        .optional()
                        .ok()
                        .flatten()
                        .map(|blob| (node_id, blob))
                })
                .collect()
        };

        // --- index lock: apply ---
        let mut applied = 0usize;
        {
            let Ok(mut index) = index_mutex.lock() else {
                return 0;
            };
            for node_id in gone {
                if matches!(index.remove(&node_id), Ok(true)) {
                    applied += 1;
                }
            }
            for (node_id, blob) in blobs {
                if Self::add_journaled_vector(&mut index, &node_id, &blob) {
                    applied += 1;
                }
            }
        }
        // A vector this process wrote between the snapshot and the apply above
        // may have been removed as `gone`; its journal row sits past `head`, so
        // the next refresh puts it back.
        if let Ok(mut watermark) = self.vector_index_watermark.lock()
            && watermark.journal_seq < head
        {
            watermark.journal_seq = head;
        }
        tracing::info!(
            applied,
            head,
            profile_id,
            "reconciled the vector index against the database"
        );
        applied
    }

    /// Point the watermark at a freshly built index: it holds every vector row
    /// as of journal position `journal_seq`, and the next search must look at
    /// the journal once regardless of what the reader's data_version says.
    #[cfg(feature = "vector-search")]
    fn reset_vector_index_watermark(&self, journal_seq: i64) {
        if let Ok(mut watermark) = self.vector_index_watermark.lock() {
            watermark.journal_seq = journal_seq;
            watermark.data_version = -1;
        }
    }

    /// Trim `vector_journal` (#181). A row is needed only until every peer has
    /// absorbed it, and a peer that has been away long enough to miss trimmed
    /// rows reconciles against the table, so this keeps the newest 10,000 rows
    /// plus everything younger than seven days and deletes the rest. Ids only;
    /// there is no content in this table to protect. Returns rows deleted.
    pub(crate) fn prune_vector_journal(&self) -> Result<usize> {
        const KEEP_ROWS: i64 = 10_000;
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let deleted = writer.execute(
            "DELETE FROM vector_journal
             WHERE seq <= (SELECT COALESCE(MAX(seq), 0) FROM vector_journal) - ?1
               AND at < strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-7 days')",
            params![KEEP_ROWS],
        )?;
        Ok(deleted)
    }

    /// Semantic search returning scores
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn semantic_search_raw(&self, query: &str, limit: i32) -> Result<Vec<(String, f32)>> {
        if !self.vector_search_available() {
            return Ok(vec![]);
        }
        if !self.active_embedding_runtime_ready()? {
            return Err(StorageError::InvalidEmbeddingProfile(
                "active embedding profile has no explicitly attached local query runtime"
                    .to_string(),
            ));
        }

        // HyDE query expansion: for conceptual queries, embed expanded variants
        // and use the centroid for broader semantic coverage
        let intent = hyde::classify_intent(query);
        let query_embedding = match intent {
            hyde::QueryIntent::Definition
            | hyde::QueryIntent::HowTo
            | hyde::QueryIntent::Reasoning
            | hyde::QueryIntent::Lookup => {
                let variants = hyde::expand_query(query);
                let embeddings: Vec<Vec<f32>> = variants
                    .iter()
                    .filter_map(|v| self.get_query_embedding(v).ok())
                    .collect();
                if embeddings.len() > 1 {
                    hyde::centroid_embedding(&embeddings)
                } else {
                    self.get_query_embedding(query)?
                }
            }
            _ => self.get_query_embedding(query)?,
        };

        // Pick up anything a peer process wrote since we last searched (#181).
        // Cheap when nothing changed: one PRAGMA and an integer comparison. Runs
        // BEFORE the index lock is taken, and takes its own locks sequentially,
        // so it cannot deadlock against the search below.
        self.refresh_vector_index_if_stale();

        let index = self.vector_index.as_ref().unwrap();
        let index = index
            .lock()
            .map_err(|_| StorageError::Init("Vector index lock poisoned".to_string()))?;

        index
            .search(&query_embedding, limit as usize)
            .map_err(|e| StorageError::Init(format!("Vector search failed: {}", e)))
    }

    /// Generate embeddings for nodes
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn generate_embeddings(
        &self,
        node_ids: Option<&[String]>,
        force: bool,
    ) -> Result<EmbeddingResult> {
        if !self.active_embedding_runtime_ready()? {
            // Generating vectors is never authority to download or initialize
            // a model. Explicit profile installation/runtime preparation must
            // happen first; callers receive an honest empty result meanwhile.
            tracing::debug!("Skipping embedding generation: active runtime is not installed/ready");
            return Ok(EmbeddingResult::default());
        }

        let active = self.active_embedding_profile()?.ok_or_else(|| {
            StorageError::InvalidEmbeddingProfile("no active embedding profile pointer".to_string())
        })?;
        let active_manifest = self
            .embedding_profile_manifest(&active.profile_id)?
            .ok_or_else(|| StorageError::NotFound(active.profile_id.to_string()))?;
        let active_model = active_manifest.profile.model_id.as_str();
        let mut result = EmbeddingResult::default();
        let nodes = self.embedding_regeneration_candidates(
            &active.profile_id,
            active_manifest.profile.embedding_dimension,
            active_model,
            node_ids,
            force,
        )?;

        for (id, content, stored_model) in nodes {
            if !force {
                let stored_model: Option<String> = self
                    .reader
                    .lock()
                    .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?
                    .query_row(
                        "SELECT model FROM embedding_profile_vectors
                         WHERE profile_id = ?1 AND node_id = ?2",
                        params![active.profile_id.as_str(), &id],
                        |row| row.get(0),
                    )
                    .optional()?
                    .or(stored_model);

                if stored_model.as_deref() == Some(active_model) {
                    result.skipped += 1;
                    continue;
                }
            }

            match self.generate_embedding_for_node(&id, &content) {
                Ok(()) => result.successful += 1,
                Err(e) => {
                    result.failed += 1;
                    result.errors.push(format!("{}: {}", id, e));
                }
            }
        }

        Ok(result)
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn embedding_regeneration_candidates(
        &self,
        profile_id: &EmbeddingProfileId,
        profile_dimension: usize,
        profile_model: &str,
        node_ids: Option<&[String]>,
        force: bool,
    ) -> Result<Vec<(String, String, Option<String>)>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;

        if let Some(ids) = node_ids {
            if ids.is_empty() {
                return Ok(Vec::new());
            }

            let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let query = format!(
                "SELECT kn.id, kn.content, epv.model
                 FROM knowledge_nodes kn
                 LEFT JOIN embedding_profile_vectors epv
                   ON epv.node_id = kn.id AND epv.profile_id = ?
                 WHERE kn.id IN ({})",
                placeholders
            );

            let mut stmt = reader.prepare(&query)?;
            let profile = profile_id.as_str();
            let mut params: Vec<&dyn rusqlite::ToSql> = vec![&profile];
            params.extend(ids.iter().map(|id| id as &dyn rusqlite::ToSql));
            let rows = stmt.query_map(params.as_slice(), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })?;
            return Ok(rows.filter_map(warn_skipped_row("embedding_regeneration_candidates")).collect());
        }

        if force {
            let mut stmt = reader.prepare(
                "SELECT kn.id, kn.content, epv.model
                 FROM knowledge_nodes kn
                 LEFT JOIN embedding_profile_vectors epv
                   ON epv.node_id = kn.id AND epv.profile_id = ?1",
            )?;
            let rows = stmt.query_map(params![profile_id.as_str()], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })?;
            return Ok(rows.filter_map(warn_skipped_row("embedding_regeneration_candidates")).collect());
        }

        let mut stmt = reader.prepare(
            "SELECT kn.id, kn.content, epv.model
             FROM knowledge_nodes kn
             LEFT JOIN embedding_profile_vectors epv
               ON epv.node_id = kn.id AND epv.profile_id = ?1
             WHERE epv.node_id IS NULL OR epv.dimensions != ?2 OR epv.model != ?3",
        )?;
        let rows = stmt.query_map(
            params![profile_id.as_str(), profile_dimension as i64, profile_model],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            },
        )?;
        Ok(rows.filter_map(warn_skipped_row("embedding_regeneration_candidates")).collect())
    }

    /// Query memories valid at a specific time
    pub fn query_at_time(
        &self,
        point_in_time: DateTime<Utc>,
        limit: i32,
    ) -> Result<Vec<KnowledgeNode>> {
        let timestamp = point_in_time.to_rfc3339();

        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT * FROM knowledge_nodes
             WHERE (valid_from IS NULL OR valid_from <= ?1)
             AND (valid_until IS NULL OR valid_until >= ?1)
             ORDER BY created_at DESC
             LIMIT ?2",
        )?;

        let nodes = stmt.query_map(params![timestamp, limit], Self::row_to_node)?;

        let mut result = Vec::new();
        for node in nodes {
            result.push(node?);
        }
        Ok(result)
    }

    /// Query memories created/modified in a time range, optionally filtered by
    /// `node_type` and/or `tags`.
    ///
    /// All filters are pushed into the SQL `WHERE` clause so that `LIMIT` is
    /// applied AFTER filtering. If filters were applied in Rust after `LIMIT`,
    /// sparse types/tags could be crowded out by a dominant set within the
    /// limit window — e.g. a query for a rare tag against a corpus where
    /// every day has hundreds of rows with a common tag would return 0
    /// matches after `LIMIT` crowded the rare-tag rows out.
    ///
    /// Tag filtering uses `tags LIKE '%"tag"%'` — an exact-match JSON pattern
    /// that keys off the quote characters around each tag in the stored JSON
    /// array. This avoids the substring-match false positive where `alpha`
    /// would otherwise match `alphabet`.
    pub fn query_time_range(
        &self,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
        limit: i32,
        node_type: Option<&str>,
        tags: Option<&[String]>,
    ) -> Result<Vec<KnowledgeNode>> {
        let start_str = start.map(|dt| dt.to_rfc3339());
        let end_str = end.map(|dt| dt.to_rfc3339());

        let mut conditions: Vec<String> = Vec::new();
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        let mut idx = 1;

        if let Some(ref s) = start_str {
            conditions.push(format!("created_at >= ?{}", idx));
            params.push(Box::new(s.clone()) as Box<dyn rusqlite::ToSql>);
            idx += 1;
        }
        if let Some(ref e) = end_str {
            conditions.push(format!("created_at <= ?{}", idx));
            params.push(Box::new(e.clone()) as Box<dyn rusqlite::ToSql>);
            idx += 1;
        }
        if let Some(nt) = node_type {
            conditions.push(format!("LOWER(node_type) = LOWER(?{})", idx));
            params.push(Box::new(nt.to_string()) as Box<dyn rusqlite::ToSql>);
            idx += 1;
        }
        if let Some(tag_list) = tags.filter(|t| !t.is_empty()) {
            let mut tag_conditions = Vec::new();
            for tag in tag_list {
                tag_conditions.push(format!("tags LIKE ?{}", idx));
                params.push(Box::new(format!("%\"{}\"%", tag)) as Box<dyn rusqlite::ToSql>);
                idx += 1;
            }
            conditions.push(format!("({})", tag_conditions.join(" OR ")));
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let query = format!(
            "SELECT * FROM knowledge_nodes {} ORDER BY created_at DESC LIMIT ?{}",
            where_clause, idx
        );
        params.push(Box::new(limit) as Box<dyn rusqlite::ToSql>);

        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(&query)?;
        let params_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        let nodes = stmt.query_map(params_refs.as_slice(), Self::row_to_node)?;

        let mut result = Vec::new();
        for node in nodes {
            result.push(node?);
        }
        Ok(result)
    }

    /// Apply FSRS-6 decay to all memories using batched pagination to avoid OOM.
    ///
    /// Uses the real FSRS-6 retrievability formula: R = (1 + factor * t / S)^(-w20)
    /// with personalized w20 from fsrs_config table. Sentiment boost extends
    /// effective stability for emotional memories.
    pub fn apply_decay(&self) -> Result<i32> {
        // Read personalized w20 from config (falls back to default 0.1542)
        let w20 = self.get_fsrs_w20().unwrap_or(DEFAULT_DECAY);
        let sleep = crate::SleepConsolidation::new();

        const BATCH_SIZE: i64 = 500;
        let now = Utc::now();
        let mut count = 0i32;
        let mut offset = 0i64;

        loop {
            // Read batch using reader
            let batch: Vec<(String, String, f64, f64, f64, f64)> = {
                let reader = self
                    .reader
                    .lock()
                    .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
                reader
                    .prepare(
                        "SELECT id, last_accessed, storage_strength, retrieval_strength,
                                sentiment_magnitude, stability
                         FROM knowledge_nodes
                         ORDER BY id
                         LIMIT ?1 OFFSET ?2",
                    )?
                    .query_map(params![BATCH_SIZE, offset], |row| {
                        Ok((
                            row.get(0)?,
                            row.get(1)?,
                            row.get(2)?,
                            row.get(3)?,
                            row.get(4)?,
                            row.get(5)?,
                        ))
                    })?
                    .filter_map(warn_skipped_row("apply_decay"))
                    .collect()
            };

            if batch.is_empty() {
                break;
            }

            let batch_len = batch.len() as i64;

            // Write batch using writer transaction
            {
                let writer = self
                    .writer
                    .lock()
                    .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
                let tx = Self::begin_write_transaction(&writer, "apply_decay")?;

                for (id, last_accessed, storage_strength, _, sentiment_mag, stability) in &batch {
                    let last = DateTime::parse_from_rfc3339(last_accessed)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or(now);

                    let days_since = (now - last).num_seconds() as f64 / 86400.0;

                    if days_since > 0.0 {
                        // Sentiment boost: emotional memories decay slower (up to 1.5x stability)
                        let effective_stability = stability * (1.0 + sentiment_mag * 0.5);

                        // Real FSRS-6 retrievability with personalized w20
                        let new_retrieval =
                            retrievability_with_decay(effective_stability, days_since, w20);

                        // Use SleepConsolidation for retention calculation
                        let new_retention =
                            sleep.calculate_retention(*storage_strength, new_retrieval);

                        tx.execute(
                            "UPDATE knowledge_nodes SET retrieval_strength = ?1, retention_strength = ?2 WHERE id = ?3",
                            params![new_retrieval, new_retention, id],
                        )?;

                        count += 1;
                    }
                }

                tx.commit()?;
            }
            offset += batch_len;
        }

        Ok(count)
    }

    /// Read personalized w20 from fsrs_config table
    fn get_fsrs_w20(&self) -> Result<f64> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        reader
            .query_row(
                "SELECT value FROM fsrs_config WHERE key = 'w20'",
                [],
                |row| row.get(0),
            )
            .map_err(|e| StorageError::Init(format!("Failed to read w20: {}", e)))
    }

    /// Run full FSRS-6 consolidation cycle (v1.4.0)
    ///
    /// 7-step automatic consolidation:
    /// 1. Apply FSRS-6 decay with personalized w20
    /// 2. Promote emotional memories (synaptic tagging)
    /// 3. Generate missing embeddings
    /// 4. Auto-dedup: merge similar memories (episodic → semantic)
    /// 5. Compute ACT-R base-level activations from access history
    /// 6. Prune old access log entries (keep 90 days)
    /// 7. Optimize w20 if enough usage data exists
    pub fn run_consolidation(&self) -> Result<ConsolidationResult> {
        let start = std::time::Instant::now();

        // Before decay, remove residual recency supplied only by the legacy
        // passive-search behavior. Otherwise a memory last shown just before
        // the upgrade would incorrectly avoid its first post-upgrade decay.
        let _ = self.repair_legacy_passive_retrieval_state();

        // v1.5.0: Use SleepConsolidation for structured consolidation
        let sleep = crate::SleepConsolidation::new();

        // Repair stability values that escaped the MAX_STABILITY invariant
        // before the sentiment-boost clamp existed (issue #121): a real store
        // was measured carrying five outliers up to 1.4e24 days. Idempotent,
        // and a no-op on healthy stores.
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            let repaired = writer.execute(
                "UPDATE knowledge_nodes SET stability = ?1 WHERE stability > ?1",
                params![crate::fsrs::MAX_STABILITY],
            )?;
            if repaired > 0 {
                tracing::warn!(
                    repaired,
                    "clamped runaway stability values back to MAX_STABILITY"
                );
            }
        }

        // 1. Apply FSRS-6 decay with real formula + personalized w20
        let decay_applied = self.apply_decay()? as i64;

        // 2. Promote emotional memories via SleepConsolidation
        let mut promoted = 0i64;
        {
            let candidates: Vec<(String, f64, f64)> = {
                let reader = self
                    .reader
                    .lock()
                    .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
                reader
                    .prepare(
                        "SELECT id, sentiment_magnitude, storage_strength
                         FROM knowledge_nodes
                         WHERE storage_strength < 10.0",
                    )?
                    .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
                    .filter_map(warn_skipped_row("run_consolidation"))
                    .collect()
            };

            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            for (id, sentiment_mag, storage_strength) in &candidates {
                if sleep.should_promote(*sentiment_mag, *storage_strength) {
                    let boosted = sleep.promotion_boost(*storage_strength);
                    writer.execute(
                        "UPDATE knowledge_nodes SET storage_strength = ?1 WHERE id = ?2",
                        params![boosted, id],
                    )?;
                    promoted += 1;
                }
            }
        }

        // 3. Generate missing and model-mismatched embeddings.
        // This must drain the whole set so embedder upgrades do not strand v1 corpora.
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        let embeddings_generated = self.generate_missing_embeddings()?;
        #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
        let embeddings_generated = 0i64;

        // 4. Auto-dedup: merge similar memories (episodic → semantic consolidation)
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        let duplicates_merged = self.auto_dedup_consolidation().unwrap_or(0);
        #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
        let duplicates_merged = 0i64;

        // 5. Compute ACT-R activations from access history
        let activations_computed = self.compute_act_r_activations().unwrap_or(0);

        // 6. Prune old access log entries (keep 90 days)
        let _ = self.prune_access_log();

        // 6b. Prune the vector journal (#181): ids only, kept long enough for
        // every peer process to absorb them, then trimmed.
        let _ = self.prune_vector_journal();

        // 6.5. Prune old Black Box trace events (keep 30 days by default;
        // VESTIGE_TRACE_RETENTION_DAYS overrides, 0 = keep forever). Best-effort
        // like the access-log sweep: a failure never blocks consolidation.
        let _ = self.prune_agent_traces();

        // 6.6. Fold the WAL back into the main database while the store is
        // quiet. `wal_autocheckpoint` (1000 pages) already runs on commit, but
        // a PASSIVE checkpoint here keeps the .wal from ratcheting upward over
        // long uptimes. Best-effort like the sweeps above.
        match self.checkpoint_wal(WalCheckpointMode::Passive) {
            Ok(status) => tracing::debug!(
                log_frames = status.log_frames,
                checkpointed_frames = status.checkpointed_frames,
                busy = status.busy,
                "WAL checkpoint after consolidation"
            ),
            Err(error) => {
                tracing::warn!(%error, "WAL checkpoint after consolidation failed")
            }
        }

        // 7. Optimize w20 if enough usage data
        let w20_optimized = self.optimize_w20_if_ready().unwrap_or(None);

        // ====================================================================
        // v1.5.0: Extended consolidation steps 8-15
        // ====================================================================

        // 8. Memory Dreams — synthesize insights (sync path)
        let mut _insights_generated = 0i64;
        {
            let dreamer = crate::advanced::dreams::MemoryDreamer::new();
            let recent = self.get_all_nodes(100, 0).unwrap_or_default();
            let dream_memories: Vec<crate::advanced::dreams::DreamMemory> = recent
                .iter()
                .map(|n| crate::advanced::dreams::DreamMemory {
                    id: n.id.clone(),
                    content: n.content.clone(),
                    embedding: None,
                    tags: n.tags.clone(),
                    created_at: n.created_at,
                    access_count: n.reps as u32,
                })
                .collect();
            if dream_memories.len() >= 5 {
                let insights = dreamer.synthesize_insights(&dream_memories);
                _insights_generated = insights.len() as i64;
                for insight in &insights {
                    let record = InsightRecord {
                        id: Uuid::new_v4().to_string(),
                        insight: insight.insight.clone(),
                        source_memories: insight.source_memories.clone(),
                        confidence: insight.confidence,
                        novelty_score: insight.novelty_score,
                        insight_type: format!("{:?}", insight.insight_type),
                        generated_at: Utc::now(),
                        tags: vec![],
                        feedback: None,
                        applied_count: 0,
                    };
                    let _ = self.save_insight(&record);
                }
            }
        }

        // 8.5. Retroactive Salience Backfill — memory with hindsight (auto-fire).
        //
        // The dream pass (step 8) replays memories forward to synthesize insights.
        // This is its backward twin: when a recent memory is a salient FAILURE,
        // reach BACKWARD across history and PROMOTE the quiet earlier memory that
        // caused it — the root cause a semantic search structurally cannot surface
        // because it is causally upstream, not *similar*. Faithful port of the
        // offline ensemble co-reactivation in Zaki/Cai et al. 2024 Nature; the
        // consolidation pass IS the offline window. Bounded on every axis so a
        // noisy day cannot trigger a promotion storm, and idempotent across cycles
        // via a durable causal edge (so the same cause is promoted once per
        // failure, not every cycle).
        //
        // OPT-OUT (backfill-safety, v2.2.1): auto-fire is ON by default — it shipped
        // and was documented in v2.2.0, so we keep the behavior — but is now bounded
        // and disableable. It mutates FSRS scores on the canonical store and can lift
        // a memory across a downstream consolidation floor, so a consumer that reads
        // `stability` as a durability gate can turn it off with
        // VESTIGE_BACKFILL_AUTOFIRE=0 (or false/off/no). The `backfill` MCP tool + CLI
        // remain available for on-demand, operator-driven backfill regardless of the
        // gate. The promote is bounded: both the auto-fire and manual paths call
        // promote_memory_backfill (stability = MIN(stability*1.5, stability+365)) so
        // repeated per-(cause, failure) promotions cannot inflate without bound (the
        // prior comment claimed promote_memory was capped — it was not).
        let backfill_autofire = std::env::var("VESTIGE_BACKFILL_AUTOFIRE")
            .map(|v| {
                let v = v.trim();
                !(v.eq_ignore_ascii_case("false")
                    || v.eq_ignore_ascii_case("off")
                    || v.eq_ignore_ascii_case("no")
                    || v == "0")
            })
            .unwrap_or(true);
        let mut backfilled_causes = 0i64;
        if backfill_autofire {
            use crate::advanced::retroactive_backfill::{
                self as rb, BackfillCandidate, FailureEvent, RetroactiveBackfill,
            };
            const MAX_FAILURES_PER_CYCLE: usize = 5;
            const CANDIDATE_SCAN: i32 = 500;

            let recent = self.get_all_nodes(CANDIDATE_SCAN, 0).unwrap_or_default();
            let failures: Vec<&KnowledgeNode> = recent
                .iter()
                .filter(|n| rb::looks_like_failure(&n.content, &n.tags))
                .take(MAX_FAILURES_PER_CYCLE)
                .collect();

            if !failures.is_empty() {
                let backfill = RetroactiveBackfill::new();
                let mut already_promoted: std::collections::HashSet<(String, String)> =
                    std::collections::HashSet::new();

                for failure_node in failures {
                    let failure = FailureEvent {
                        id: failure_node.id.clone(),
                        content: failure_node.content.clone(),
                        entities: rb::extract_entities(&failure_node.content, &failure_node.tags),
                        tags: failure_node.tags.clone(),
                        prediction_error: 0.9,
                        manual: false,
                    };
                    // candidates = every OTHER memory strictly older than the
                    // failure, EXCLUDING other failures (a root cause is the quiet
                    // upstream change, not an earlier crash).
                    let candidates: Vec<BackfillCandidate> = recent
                        .iter()
                        .filter(|c| c.id != failure_node.id)
                        .filter(|c| !rb::looks_like_failure(&c.content, &c.tags))
                        .filter_map(|c| {
                            let age = (failure_node.created_at - c.created_at).num_seconds() as f64
                                / 86_400.0;
                            if age <= 0.0 {
                                return None;
                            }
                            Some(BackfillCandidate {
                                id: c.id.clone(),
                                content: c.content.clone(),
                                entities: rb::extract_entities(&c.content, &c.tags),
                                age_days_before_failure: age,
                                stability: c.stability,
                                similarity_to_failure: None,
                            })
                        })
                        .collect();

                    let result = backfill.run(&failure, &candidates);
                    if !result.triggered {
                        continue;
                    }
                    for cause in &result.causes {
                        if !already_promoted
                            .insert((cause.memory_id.clone(), failure_node.id.clone()))
                        {
                            continue;
                        }
                        // Cross-cycle idempotency: a durable causal edge is both the
                        // dedup key and a first-class artifact. Write it FIRST, only
                        // promote if it persisted (a failed edge write => retry next
                        // cycle cleanly, never double-inflate).
                        let link_type = crate::memory::EdgeType::Causal.to_string();
                        let already_linked = self
                            .get_connections_for_memory(&cause.memory_id)
                            .map(|conns| {
                                conns.iter().any(|c| {
                                    c.source_id == cause.memory_id
                                        && c.target_id == failure_node.id
                                        && c.link_type == link_type
                                })
                            })
                            .unwrap_or(false);
                        if already_linked {
                            continue;
                        }
                        let conn = ConnectionRecord {
                            source_id: cause.memory_id.clone(),
                            target_id: failure_node.id.clone(),
                            strength: 1.0,
                            link_type,
                            created_at: Utc::now(),
                            last_activated: Utc::now(),
                            activation_count: 0,
                        };
                        if self.save_connection(&conn).is_err() {
                            continue;
                        }
                        if self.promote_memory_backfill(&cause.memory_id).is_ok() {
                            backfilled_causes += 1;
                        }
                    }
                }
                if backfilled_causes > 0 {
                    tracing::info!(
                        backfilled_causes,
                        "Retroactive Salience Backfill: promoted {} root-cause memor{} a semantic search would miss",
                        backfilled_causes,
                        if backfilled_causes == 1 { "y" } else { "ies" }
                    );
                }
            }
        }

        // 9. Memory Compression (old memories → summaries)
        let mut _memories_compressed = 0i64;
        {
            let mut compressor = crate::advanced::compression::MemoryCompressor::new();
            let all_nodes = self.get_all_nodes(500, 0).unwrap_or_default();
            let thirty_days_ago = Utc::now() - Duration::days(30);
            let old_memories: Vec<crate::advanced::compression::MemoryForCompression> = all_nodes
                .iter()
                .filter(|n| n.created_at < thirty_days_ago && n.retention_strength < 0.5)
                .map(|n| crate::advanced::compression::MemoryForCompression {
                    id: n.id.clone(),
                    content: n.content.clone(),
                    tags: n.tags.clone(),
                    created_at: n.created_at,
                    last_accessed: Some(n.last_accessed),
                    embedding: None,
                })
                .collect();
            if old_memories.len() >= 3 {
                let groups = compressor.find_compressible_groups(&old_memories);
                for group_ids in groups.iter().take(5) {
                    // Limit to 5 groups per consolidation
                    let group: Vec<_> = old_memories
                        .iter()
                        .filter(|m| group_ids.contains(&m.id))
                        .cloned()
                        .collect();
                    if let Some(_compressed) = compressor.compress(&group) {
                        _memories_compressed += group.len() as i64;
                    }
                }
            }
        }

        // 10. Memory State Transitions (Active→Dormant→Silent→Unavailable)
        let _state_transitions: i64;
        {
            let service = crate::neuroscience::memory_states::StateUpdateService::new();
            let all_nodes = self.get_all_nodes(500, 0).unwrap_or_default();
            let mut lifecycles: Vec<crate::neuroscience::memory_states::MemoryLifecycle> =
                all_nodes
                    .iter()
                    .map(|n| {
                        let mut lc = crate::neuroscience::memory_states::MemoryLifecycle::new();
                        lc.last_access = n.last_accessed;
                        lc.access_count = n.reps as u32;
                        lc.state = if n.retention_strength > 0.7 {
                            crate::neuroscience::memory_states::MemoryState::Active
                        } else if n.retention_strength > 0.3 {
                            crate::neuroscience::memory_states::MemoryState::Dormant
                        } else if n.retention_strength > 0.1 {
                            crate::neuroscience::memory_states::MemoryState::Silent
                        } else {
                            crate::neuroscience::memory_states::MemoryState::Unavailable
                        };
                        lc
                    })
                    .collect();
            let batch_result = service.batch_update(&mut lifecycles);
            _state_transitions = batch_result.total_transitions as i64;
        }

        // 11. Synaptic Capture Sweep (retroactive importance)
        {
            let mut sts = crate::neuroscience::synaptic_tagging::SynapticTaggingSystem::new();
            let _ = sts.sweep_for_capture(Utc::now());
            sts.decay_tags();
        }

        // 12. Cross-Project Learning (detect universal patterns)
        {
            let learner = crate::advanced::cross_project::CrossProjectLearner::new();
            let _patterns = learner.find_universal_patterns();
        }

        // 13. Hippocampal Index Maintenance
        {
            let index = crate::neuroscience::hippocampal_index::HippocampalIndex::new();
            let _ = index.prune_weak_links();
        }

        // 14. Importance Evolution (decay stale importance)
        {
            let tracker = crate::advanced::importance::ImportanceTracker::new();
            tracker.apply_importance_decay();
        }

        // 15. Connection Graph Maintenance (decay + prune weak connections)
        let _connections_pruned = self.prune_weak_connections(0.05).unwrap_or(0) as i64;

        // 16. FTS5 index optimization — merge segments for faster keyword search
        // 17. Run PRAGMA optimize to refresh query planner statistics
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            let _ = writer
                .execute_batch("INSERT INTO knowledge_fts(knowledge_fts) VALUES('optimize');");
            let _ = writer.execute_batch("PRAGMA optimize;");
        }

        // ====================================================================
        // v1.9.0: Autonomic features (18-20)
        // ====================================================================

        // 18. Auto-promote memories with 3+ accesses in 24h (frequency-dependent potentiation)
        let auto_promoted = self.auto_promote_frequent_access().unwrap_or(0);
        promoted += auto_promoted;

        // 19. Retention Target System — REPORT ONLY. Consolidation never
        // deletes memories.
        //
        // Until v2.6.0 this step hard-deleted every memory below 0.3
        // retention older than 30 days whenever average retention slipped
        // under a target. It looked dormant for months only because decay was
        // broken (the w20 story in fsrs/optimizer.rs); the day decay came
        // back to life it silently destroyed 23 real memories from a live
        // 2,929-memory store in a single cycle — unattended, unrecoverable,
        // invisible in the consolidation output, and with no protected-pin
        // exemption. Forgetting in Vestige means DOWN-RANKING (the
        // accessibility states); destruction is reserved for the explicit,
        // previewable, dry-run-by-default `maintain {action:"gc"}` and
        // `purge` paths. VESTIGE_RETENTION_TARGET no longer gates anything
        // destructive.
        {
            let avg_retention = self.get_avg_retention().unwrap_or(1.0);
            let total = self.get_stats().map(|s| s.total_nodes).unwrap_or(0);
            let below_target = self.count_memories_below_retention(0.3).unwrap_or(0);

            if below_target > 0 {
                tracing::info!(
                    avg_retention,
                    gc_candidates = below_target,
                    "{} memories sit below 0.3 retention; review them with maintain {{action:\"gc\", dry_run:true}} — consolidation deletes nothing",
                    below_target
                );
            }

            // 20. Save retention snapshot for trend tracking. `gc_triggered`
            // is permanently false: the autonomic GC no longer exists.
            let _ = self.save_retention_snapshot(avg_retention, total, below_target, false);
        }

        let duration = start.elapsed().as_millis() as i64;

        // Record consolidation history
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            let _ = writer.execute(
                "INSERT INTO consolidation_history (completed_at, duration_ms, memories_replayed, duplicates_merged, activations_computed, w20_optimized)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    Utc::now().to_rfc3339(),
                    duration,
                    decay_applied,
                    duplicates_merged,
                    activations_computed,
                    w20_optimized,
                ],
            );
        }

        Ok(ConsolidationResult {
            nodes_processed: decay_applied,
            nodes_promoted: promoted,
            nodes_pruned: 0,
            decay_applied,
            duration_ms: duration,
            embeddings_generated,
            duplicates_merged,
            neighbors_reinforced: 0,
            activations_computed,
            w20_optimized,
            backfilled_causes,
        })
    }

    /// Auto-deduplicate similar memories during consolidation (episodic → semantic merge)
    ///
    /// Finds clusters with cosine similarity >= 0.85, keeps the strongest node,
    /// appends unique content from weaker nodes, and deletes duplicates.
    /// Honors the `VESTIGE_AUTO_CONSOLIDATE_MERGE` opt-out (unset → on) and
    /// never merges away or deletes protected (pinned) nodes (#142).
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn auto_dedup_consolidation(&self) -> Result<i64> {
        // OPT-IN (v2.6.0, reversing the #142 opt-out): this pass concat-merges
        // near-duplicate memories and HARD-DELETES the weaker ones with no
        // reflog. Unattended destruction of user memories is opt-IN, never a
        // default: set VESTIGE_AUTO_CONSOLIDATE_MERGE=1 (or true/on/yes) to
        // enable it. Unset or any other/malformed value fails CLOSED — the
        // safe direction for a destructive gate (#142's opt-out parsed the
        // same input as fail-OPEN, so a typo destroyed data). The `dedup` MCP
        // tool remains the on-demand, previewable, reversible path and is
        // unaffected by this gate. Gate here (not the caller) so it stays
        // with the pin filter and self-protects against a future second
        // caller.
        let auto_merge = std::env::var("VESTIGE_AUTO_CONSOLIDATE_MERGE")
            .map(|v| {
                let v = v.trim();
                v.eq_ignore_ascii_case("true")
                    || v.eq_ignore_ascii_case("on")
                    || v.eq_ignore_ascii_case("yes")
                    || v == "1"
            })
            .unwrap_or(false);
        if !auto_merge {
            return Ok(0);
        }

        let all_embeddings = self.get_all_embeddings()?;
        let n = all_embeddings.len();

        if !(2..=2000).contains(&n) {
            return Ok(0);
        }

        // Protected (pinned) memories must never be touched by this unattended,
        // no-audit pass — mirroring the interactive contract that a protected
        // node may only survive a merge, never be absorbed (see `plan_merge`).
        // Fetch the set ONCE here, before the per-cluster reader lock is taken:
        // both `protected_node_ids()` and `is_protected()` take their OWN reader
        // lock, so calling either inside the lock window below would self-deadlock
        // the non-reentrant Mutex. Skipping protected ids at BOTH the outer
        // (anchor) and inner (member) loops guarantees a protected node is never
        // an anchor and never a cluster member — so it can never be the keeper nor
        // land in weak_ids, and is thus never merged into and never deleted. Fails
        // SAFE via `?`: on a poisoned lock the caller's unwrap_or(0) skips the
        // merge this cycle rather than risk absorbing a pin. #142
        let protected = self.protected_node_ids()?;

        // Scope map, fetched ONCE alongside `protected` and for the same reason:
        // the per-cluster reader lock below is non-reentrant, so this cannot be
        // looked up inside the loop. This pass merges content and then HARD
        // DELETES the weak nodes, unattended and with no audit row. Without a
        // scope guard it will happily fuse two different projects' near-identical
        // notes -- e.g. the same convention worded alike but naming different
        // credentials -- and destroy one of them. Memories only ever cluster with
        // memories in their OWN scope.
        let scopes: std::collections::HashMap<String, String> = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            let mut stmt = reader.prepare(
                "SELECT id, COALESCE(NULLIF(TRIM(scope), ''), 'user') FROM knowledge_nodes",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            let mut m = std::collections::HashMap::new();
            for r in rows {
                let (id, sc) = r?;
                m.insert(id, sc);
            }
            m
        };
        let scope_of = |id: &str| -> &str { scopes.get(id).map(String::as_str).unwrap_or("user") };

        const SIMILARITY_THRESHOLD: f32 = 0.85;
        let mut merged_count = 0i64;
        let mut consumed: std::collections::HashSet<String> = std::collections::HashSet::new();

        for i in 0..n {
            if consumed.contains(&all_embeddings[i].0) || protected.contains(&all_embeddings[i].0) {
                continue;
            }

            let mut cluster: Vec<(usize, f32)> = Vec::new();

            let anchor_scope = scope_of(&all_embeddings[i].0);
            for j in (i + 1)..n {
                if consumed.contains(&all_embeddings[j].0)
                    || protected.contains(&all_embeddings[j].0)
                {
                    continue;
                }
                // Never cluster across project scopes: the merge below deletes.
                if scope_of(&all_embeddings[j].0) != anchor_scope {
                    continue;
                }
                let sim = crate::embeddings::cosine_similarity(
                    &all_embeddings[i].1,
                    &all_embeddings[j].1,
                );
                if sim >= SIMILARITY_THRESHOLD {
                    cluster.push((j, sim));
                }
            }

            if cluster.is_empty() {
                continue;
            }

            // Find the strongest node (highest retention_strength)
            let anchor_id = &all_embeddings[i].0;
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            let anchor_retention: f64 = reader
                .query_row(
                    "SELECT retention_strength FROM knowledge_nodes WHERE id = ?1",
                    params![anchor_id],
                    |row| row.get(0),
                )
                .unwrap_or(0.0);

            let mut best_idx = i;
            let mut best_retention = anchor_retention;

            for &(j, _) in &cluster {
                let dup_id = &all_embeddings[j].0;
                let dup_retention: f64 = reader
                    .query_row(
                        "SELECT retention_strength FROM knowledge_nodes WHERE id = ?1",
                        params![dup_id],
                        |row| row.get(0),
                    )
                    .unwrap_or(0.0);
                if dup_retention > best_retention {
                    best_retention = dup_retention;
                    best_idx = j;
                }
            }

            let best_id = all_embeddings[best_idx].0.clone();

            // Get keeper's content
            let keeper_content: String = reader
                .query_row(
                    "SELECT content FROM knowledge_nodes WHERE id = ?1",
                    params![best_id],
                    |row| row.get(0),
                )
                .unwrap_or_default();

            // Collect weak node IDs (all nodes in cluster except the keeper)
            let mut weak_ids: Vec<String> = Vec::new();
            if best_idx != i {
                weak_ids.push(anchor_id.clone());
            }
            for &(j, _) in &cluster {
                if j != best_idx {
                    weak_ids.push(all_embeddings[j].0.clone());
                }
            }

            // Merge unique content from weak nodes
            let mut merged_content = keeper_content.clone();
            for weak_id in &weak_ids {
                let weak_content: String = reader
                    .query_row(
                        "SELECT content FROM knowledge_nodes WHERE id = ?1",
                        params![weak_id],
                        |row| row.get(0),
                    )
                    .unwrap_or_default();

                let weak_trimmed = weak_content.trim();
                if !merged_content.contains(weak_trimmed) && weak_trimmed.len() > 20 {
                    merged_content.push_str("\n\n[MERGED] ");
                    merged_content.push_str(weak_trimmed);
                }
            }

            // Drop reader before taking writer locks in update/delete
            drop(reader);

            // Update keeper with merged content. The update result is the
            // gate for the deletions below: if the keeper never absorbed the
            // weak nodes' content, deleting them destroys it. The previous
            // `let _ =` discarded exactly that failure and deleted anyway.
            let content_preserved = if merged_content != keeper_content {
                self.update_node_content(&best_id, &merged_content).is_ok()
            } else {
                true
            };

            if content_preserved {
                // Delete weak nodes — their content verifiably lives on in
                // the keeper (or was already contained in it).
                for weak_id in &weak_ids {
                    let _ = self.delete_node(weak_id);
                    consumed.insert(weak_id.clone());
                    merged_count += 1;
                }
            } else {
                tracing::warn!(
                    keeper = %best_id,
                    weak = weak_ids.len(),
                    "auto-dedup: keeper content update failed; weak nodes kept (nothing deleted)"
                );
                for weak_id in &weak_ids {
                    consumed.insert(weak_id.clone());
                }
            }

            consumed.insert(best_id);
        }

        Ok(merged_count)
    }

    /// Restore the last meaningful interaction for memories whose most recent
    /// `last_accessed` value came from the old passive-search behavior.
    ///
    /// Pre-2.3.0 `search_hit` rows updated `last_accessed`, which also fed the
    /// recency ranker and FSRS decay. `retrieval_shown` is intentionally not
    /// included: the new event never writes node state. A passive event is
    /// logged immediately after the old update, so we repair only nodes whose
    /// timestamp is no later than their latest legacy hit. An unlogged FSRS
    /// review updates `updated_at`, making it a safe fallback before
    /// `created_at` when no explicit interaction is recorded.
    fn repair_legacy_passive_retrieval_state(&self) -> Result<i64> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let repaired = writer.execute(
            "UPDATE knowledge_nodes AS node
             SET last_accessed = MAX(
                 COALESCE(
                     (
                         SELECT MAX(explicit.accessed_at)
                         FROM memory_access_log AS explicit
                         WHERE explicit.node_id = node.id
                           AND explicit.access_type NOT IN ('search_hit', 'retrieval_shown')
                     ),
                     node.created_at
                 ),
                 node.updated_at
             )
             WHERE EXISTS (
                 SELECT 1
                 FROM memory_access_log AS passive
                 WHERE passive.node_id = node.id
                   AND passive.access_type = 'search_hit'
             )
               AND node.last_accessed <= (
                 SELECT MAX(passive.accessed_at)
                 FROM memory_access_log AS passive
                 WHERE passive.node_id = node.id
                   AND passive.access_type = 'search_hit'
             )",
            [],
        )?;
        Ok(repaired as i64)
    }

    /// Compute ACT-R base-level activation for all nodes from access history.
    /// B_i = ln(Σ t_j^(-d)) where t_j = days since j-th access, d = 0.5
    fn compute_act_r_activations(&self) -> Result<i64> {
        const ACT_R_DECAY: f64 = 0.5;
        let now = Utc::now();

        // This also protects direct callers that compute ACT-R without using
        // the full consolidation cycle.
        self.repair_legacy_passive_retrieval_state()?;

        let node_ids: Vec<String> = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            reader
                .prepare(
                    "SELECT DISTINCT node_id FROM memory_access_log
                     WHERE access_type NOT IN ('search_hit', 'retrieval_shown')",
                )?
                .query_map([], |row| row.get(0))?
                .filter_map(warn_skipped_row("compute_act_r_activations"))
                .collect()
        };

        let mut count = 0i64;
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "compute_act_r_activations")?;

        // Discard residual activation from legacy search-hit rows as well as
        // new retrieval-only telemetry. Otherwise historical passive reads
        // would keep influencing rank after the behavior changes.
        tx.execute(
            "UPDATE knowledge_nodes SET activation = 0.0
             WHERE id NOT IN (
                SELECT DISTINCT node_id FROM memory_access_log
                WHERE access_type NOT IN ('search_hit', 'retrieval_shown')
             )",
            [],
        )?;

        if node_ids.is_empty() {
            tx.commit()?;
            return Ok(0);
        }

        for node_id in &node_ids {
            let timestamps: Vec<String> = tx
                .prepare(
                    "SELECT accessed_at FROM memory_access_log
                     WHERE node_id = ?1 AND access_type NOT IN ('search_hit', 'retrieval_shown')
                     ORDER BY accessed_at DESC
                     LIMIT 500",
                )?
                .query_map(params![node_id], |row| row.get(0))?
                .filter_map(warn_skipped_row("compute_act_r_activations"))
                .collect();

            if timestamps.is_empty() {
                continue;
            }

            let mut sum_decay = 0.0_f64;
            for ts_str in &timestamps {
                let accessed_at = DateTime::parse_from_rfc3339(ts_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or(now);
                let days_since = (now - accessed_at).num_seconds() as f64 / 86400.0;
                let t = days_since.max(0.001);
                sum_decay += t.powf(-ACT_R_DECAY);
            }

            let activation = sum_decay.ln();

            tx.execute(
                "UPDATE knowledge_nodes SET activation = ?1 WHERE id = ?2",
                params![activation, node_id],
            )?;
            count += 1;
        }

        tx.commit()?;
        Ok(count)
    }

    /// Prune old access log entries (keep the last [`ACCESS_LOG_RETENTION_DAYS`]).
    /// `hygiene_snapshot` derives its "never accessed" window from the same
    /// constant; keep the two in lockstep.
    fn prune_access_log(&self) -> Result<i64> {
        let cutoff = (Utc::now() - Duration::days(ACCESS_LOG_RETENTION_DAYS)).to_rfc3339();
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let deleted = writer.execute(
            "DELETE FROM memory_access_log WHERE accessed_at < ?1",
            params![cutoff],
        )? as i64;
        Ok(deleted)
    }

    /// Optimize personalized w20 (forgetting curve decay) if enough access data exists.
    /// Uses FSRSOptimizer golden section search on real retrieval history.
    fn optimize_w20_if_ready(&self) -> Result<Option<f64>> {
        use crate::fsrs::{FSRSOptimizer, ReviewLog};

        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;

        let access_count: i64 = reader
            .query_row(
                "SELECT COUNT(*) FROM memory_access_log
                 WHERE access_type NOT IN ('search_hit', 'retrieval_shown')",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        if access_count < 100 {
            return Ok(None);
        }

        let mut optimizer = FSRSOptimizer::new();

        // Most RECENT window, not the oldest. The previous `ASC LIMIT 1000`
        // trained forever on the earliest era of the log — and because the
        // 90-day log pruning slides that window, the training set drifted
        // under the optimizer's feet, producing fits that swung between
        // 0.0104 and 0.137 on the same store with no behavior change.
        let logs: Vec<(String, String, String)> = reader
            .prepare(
                "SELECT node_id, access_type, accessed_at FROM (
                     SELECT mal.node_id, mal.access_type, mal.accessed_at
                     FROM memory_access_log mal
                     WHERE mal.access_type NOT IN ('search_hit', 'retrieval_shown')
                     ORDER BY mal.accessed_at DESC
                     LIMIT 1000
                 ) ORDER BY accessed_at ASC",
            )?
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .filter_map(warn_skipped_row("optimize_w20_if_ready"))
            .collect();

        for (node_id, access_type, accessed_at) in &logs {
            // Get node state for stability/difficulty
            let node_state: Option<(f64, f64, String)> = reader
                .query_row(
                    "SELECT stability, difficulty, created_at FROM knowledge_nodes WHERE id = ?1",
                    params![node_id],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .ok();

            if let Some((stability, difficulty, created_at)) = node_state {
                let ts = DateTime::parse_from_rfc3339(accessed_at)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                let created = DateTime::parse_from_rfc3339(&created_at)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or(ts);

                // Suppression is the strongest forgetting signal a user can
                // send; feeding it to the optimizer as a SUCCESSFUL recall
                // (the old catch-all) taught the curve that nothing is ever
                // forgotten. A reversed suppression is a correction of that
                // signal, not a recall outcome either way; score it neutral.
                let rating = match access_type.as_str() {
                    "promote" => 4,
                    "search_hit" => 3,
                    "demote" | "suppress" => 1,
                    _ => 3,
                };

                let elapsed = (ts - created).num_seconds() as f64 / 86400.0;

                optimizer.add_review(ReviewLog {
                    timestamp: ts,
                    rating,
                    stability,
                    difficulty,
                    elapsed_days: elapsed.max(0.001),
                });
            }
        }

        drop(reader);

        if !optimizer.has_enough_data() {
            return Ok(None);
        }

        let optimized_w20 = optimizer.optimize_decay();

        // Save to config
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            writer.execute(
                "INSERT OR REPLACE INTO fsrs_config (key, value, updated_at)
                 VALUES ('w20', ?1, ?2)",
                params![optimized_w20, Utc::now().to_rfc3339()],
            )?;
        }

        tracing::info!(
            w20 = optimized_w20,
            "Personalized w20 optimized from access history"
        );

        Ok(Some(optimized_w20))
    }

    /// Generate all missing or active-model-mismatched embeddings.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn generate_missing_embeddings(&self) -> Result<i64> {
        if !self.active_embedding_runtime_ready()? {
            tracing::debug!(
                "Skipping consolidation embedding generation: active profile runtime is unavailable"
            );
            return Ok(0);
        }

        let result = self.generate_embeddings(None, false)?;
        if result.failed > 0 {
            tracing::warn!(
                failed = result.failed,
                "Some embeddings could not be regenerated during consolidation"
            );
        }

        Ok(result.successful)
    }
}

// ============================================================================
// PERSISTENCE LAYER: Intentions, Insights, Connections, States
// ============================================================================

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

impl Default for InsightRecord {
    fn default() -> Self {
        Self {
            id: String::new(),
            insight: String::new(),
            source_memories: Vec::new(),
            confidence: 0.0,
            novelty_score: 0.0,
            insight_type: String::new(),
            generated_at: Utc::now(),
            tags: Vec::new(),
            feedback: None,
            applied_count: 0,
        }
    }
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

/// Memory state record
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryStateRecord {
    pub memory_id: String,
    pub state: String, // 'active', 'dormant', 'silent', 'unavailable'
    pub last_access: DateTime<Utc>,
    pub access_count: i32,
    pub state_entered_at: DateTime<Utc>,
    pub suppression_until: Option<DateTime<Utc>>,
    pub suppressed_by: Vec<String>,
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

impl SqliteMemoryStore {
    // ========================================================================
    // COMPOSEDGRAPH PERSISTENCE
    // ========================================================================

    /// Save a complete composition event with members and optional outcomes in one transaction.
    pub fn save_composition(
        &self,
        event: &CompositionEventRecord,
        members: &[CompositionMemberRecord],
        outcomes: &[CompositionOutcomeRecord],
    ) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "save_composition")?;

        let metadata_json =
            serde_json::to_string(&event.metadata).unwrap_or_else(|_| "{}".to_string());
        tx.execute(
            "INSERT OR REPLACE INTO composition_events (
                id, created_at, tool, mode, query, query_hash, confidence, status,
                output_preview, metadata
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                event.id,
                event.created_at.to_rfc3339(),
                event.tool,
                event.mode,
                event.query,
                event.query_hash,
                event.confidence,
                event.status,
                event.output_preview,
                metadata_json,
            ],
        )?;

        for member in members {
            let mut member = member.clone();
            Self::snapshot_composition_member_tags(&tx, &mut member)?;
            Self::insert_composition_member(&tx, &member)?;
        }
        for outcome in outcomes {
            Self::insert_composition_outcome(&tx, outcome)?;
        }

        tx.commit()?;
        Ok(())
    }

    /// Add one outcome label to an existing composition event.
    pub fn record_composition_outcome(&self, outcome: &CompositionOutcomeRecord) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        Self::insert_composition_outcome(&writer, outcome)
    }

    /// Get one composition event by id.
    pub fn get_composition_event(&self, id: &str) -> Result<Option<CompositionEventRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare("SELECT * FROM composition_events WHERE id = ?1")?;
        stmt.query_row(params![id], Self::row_to_composition_event)
            .optional()
            .map_err(StorageError::from)
    }

    /// Get recent composition events.
    pub fn get_recent_composition_events(&self, limit: i32) -> Result<Vec<CompositionEventRecord>> {
        self.get_recent_composition_events_page(limit, 0)
    }

    /// Get recent composition events with explicit pagination.
    pub fn get_recent_composition_events_page(
        &self,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<CompositionEventRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT * FROM composition_events
             ORDER BY created_at DESC
             LIMIT ?1 OFFSET ?2",
        )?;
        let rows = stmt.query_map(
            params![limit.max(1), offset.max(0)],
            Self::row_to_composition_event,
        )?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Get all members for a composition event.
    pub fn get_composition_members(&self, event_id: &str) -> Result<Vec<CompositionMemberRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT * FROM composition_members
             WHERE event_id = ?1
             ORDER BY rank ASC, role ASC, memory_id ASC",
        )?;
        let rows = stmt.query_map(params![event_id], Self::row_to_composition_member)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Get all outcomes for a composition event.
    pub fn get_composition_outcomes(
        &self,
        event_id: &str,
    ) -> Result<Vec<CompositionOutcomeRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT * FROM composition_outcomes
             WHERE event_id = ?1
             ORDER BY labeled_at DESC",
        )?;
        let rows = stmt.query_map(params![event_id], Self::row_to_composition_outcome)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Get composition events containing a memory id.
    pub fn get_compositions_for_memory(
        &self,
        memory_id: &str,
        limit: i32,
    ) -> Result<Vec<CompositionEventRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT DISTINCT e.*
             FROM composition_events e
             JOIN composition_members m ON m.event_id = e.id
             WHERE m.memory_id = ?1
             ORDER BY e.created_at DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(
            params![memory_id, limit.max(1)],
            Self::row_to_composition_event,
        )?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Return memories most frequently composed with the requested memory.
    pub fn get_composition_neighbors(
        &self,
        memory_id: &str,
        limit: i32,
    ) -> Result<Vec<CompositionNeighborRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "WITH distinct_members AS (
                SELECT DISTINCT event_id, memory_id FROM composition_members
             )
             SELECT other.memory_id, COUNT(DISTINCT other.event_id) AS composed_count, MAX(e.created_at) AS latest_event_at
             FROM distinct_members self
             JOIN distinct_members other
               ON other.event_id = self.event_id AND other.memory_id != self.memory_id
             JOIN composition_events e ON e.id = self.event_id
             WHERE self.memory_id = ?1
             GROUP BY other.memory_id
             ORDER BY composed_count DESC, latest_event_at DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![memory_id, limit.max(1)], |row| {
            Ok(CompositionNeighborRecord {
                memory_id: row.get(0)?,
                composed_count: row.get(1)?,
                latest_event_at: Self::parse_timestamp(
                    &row.get::<_, String>(2)?,
                    "latest_event_at",
                )?,
            })
        })?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Generate ranked memory pairs that share useful tags but have not yet been composed.
    pub fn get_never_composed_candidates(
        &self,
        limit: i32,
        tag_filter: Option<&[String]>,
    ) -> Result<Vec<NeverComposedCandidate>> {
        let nodes = self.composition_candidate_nodes(tag_filter)?;
        let composed_pairs = self.composed_pair_set()?;
        let composition_degrees = self.composition_degree_map()?;
        let outcome_map = self.composition_outcome_map()?;

        // SEMANTIC-BAND GATE (the composition generativity unlock): load embeddings so a pair
        // that shares NO literal tag/word but lives in the "distant-but-relatable" cosine band
        // can still surface as a never-composed insight — exactly the non-obvious combination
        // a keyword/exact-overlap gate (and cosine-NN search) can never return. The band excludes
        // near-duplicates (>= 0.85, those are the same idea) and unrelated noise (< 0.45).
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        let embedding_map: std::collections::HashMap<String, Vec<f32>> = self
            .get_all_embeddings()
            .map(|v| v.into_iter().collect())
            .unwrap_or_default();
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        const COMPOSE_BAND_LO: f32 = 0.45;
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        const COMPOSE_BAND_HI: f32 = 0.85;

        let mut candidates = Vec::new();

        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let a = &nodes[i];
                let b = &nodes[j];
                let pair = Self::pair_key(&a.id, &b.id);
                if composed_pairs.contains(&pair) {
                    continue;
                }

                if let Some(filter) = tag_filter
                    && !filter.is_empty()
                    && !Self::node_pair_matches_tag_filter(a, b, filter)
                {
                    continue;
                }

                let shared_tags = Self::shared_tags(&a.tags, &b.tags);
                let shared_terms = Self::shared_content_terms(&a.content, &b.content, 8);

                // Semantic-band cosine: lets a pair with NO shared surface tokens but a
                // related MEANING through the gate (the generative cross-domain combination).
                #[cfg(all(feature = "embeddings", feature = "vector-search"))]
                let band_cos: Option<f32> =
                    match (embedding_map.get(&a.id), embedding_map.get(&b.id)) {
                        (Some(ea), Some(eb)) => {
                            let c = crate::embeddings::cosine_similarity(ea, eb);
                            if (COMPOSE_BAND_LO..COMPOSE_BAND_HI).contains(&c) {
                                Some(c)
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };
                #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
                let band_cos: Option<f32> = None;

                // Admit the pair if it shares surface signal OR it sits in the semantic band.
                if shared_tags.is_empty() && shared_terms.is_empty() && band_cos.is_none() {
                    continue;
                }

                let boundary_tags = Self::boundary_tags_for_pair(&a.tags, &b.tags);
                let trust_score =
                    ((a.retention_strength + b.retention_strength) / 2.0).clamp(0.0, 1.0);
                let degree_a = composition_degrees.get(&a.id).copied().unwrap_or(0) as f64;
                let degree_b = composition_degrees.get(&b.id).copied().unwrap_or(0) as f64;
                let novelty_score = ((1.0 / (1.0 + degree_a)) + (1.0 / (1.0 + degree_b))) / 2.0;
                let bridge_score = Self::composition_bridge_score(
                    a,
                    b,
                    &shared_tags,
                    &shared_terms,
                    &boundary_tags,
                );
                let anchor_score =
                    (shared_tags.len() as f64 * 0.45) + (shared_terms.len().min(5) as f64 * 0.25);
                // Semantic-band pairs (no surface overlap) get an anchor from cosine so they
                // clear the cutoff: a mid-band 0.45-0.85 meaning-match is a strong compose signal.
                let band_anchor = band_cos
                    .map(|c| 1.0 + (c as f64 - 0.45) * 2.0)
                    .unwrap_or(0.0);
                let prior_outcomes = Self::pair_prior_outcomes(&outcome_map, &a.id, &b.id);
                let outcome_signal = Self::outcome_signal(&prior_outcomes);
                let outcome_score_adjustment = Self::outcome_score_adjustment(&prior_outcomes);
                let score = anchor_score
                    + band_anchor
                    + (bridge_score * 2.0)
                    + (novelty_score * 1.5)
                    + trust_score
                    + outcome_score_adjustment;
                if score < 1.6 {
                    continue;
                }

                let reason = if !boundary_tags.is_empty() {
                    format!(
                        "Untried bridge across {} with {}",
                        boundary_tags.join(", "),
                        Self::anchor_summary(&shared_tags, &shared_terms)
                    )
                } else if a.node_type != b.node_type {
                    format!(
                        "Untried {} -> {} composition with {}",
                        a.node_type,
                        b.node_type,
                        Self::anchor_summary(&shared_tags, &shared_terms)
                    )
                } else {
                    format!(
                        "Never composed despite {}",
                        Self::anchor_summary(&shared_tags, &shared_terms)
                    )
                };
                let composition_question =
                    Self::composition_question(a, b, &shared_tags, &shared_terms, &boundary_tags);
                candidates.push(NeverComposedCandidate {
                    first_id: a.id.clone(),
                    second_id: b.id.clone(),
                    score,
                    novelty_score,
                    bridge_score,
                    trust_score,
                    outcome_score_adjustment,
                    shared_tags,
                    boundary_tags,
                    shared_terms,
                    prior_outcomes,
                    outcome_signal,
                    first_node_type: a.node_type.clone(),
                    second_node_type: b.node_type.clone(),
                    first_preview: preview(&a.content, 160),
                    second_preview: preview(&b.content, 160),
                    reason,
                    composition_question,
                });
            }
        }

        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(limit.max(1) as usize);
        Ok(candidates)
    }

    fn insert_composition_member(
        conn: &Connection,
        member: &CompositionMemberRecord,
    ) -> Result<()> {
        let metadata_json =
            serde_json::to_string(&member.metadata).unwrap_or_else(|_| "{}".to_string());
        conn.execute(
            "INSERT OR REPLACE INTO composition_members (
                event_id, memory_id, role, rank, trust, score, preview, metadata
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                member.event_id,
                member.memory_id,
                member.role,
                member.rank,
                member.trust,
                member.score,
                member.preview,
                metadata_json,
            ],
        )?;
        Ok(())
    }

    fn snapshot_composition_member_tags(
        conn: &Connection,
        member: &mut CompositionMemberRecord,
    ) -> Result<()> {
        if member.metadata.get("tags").is_some() {
            return Ok(());
        }

        let tags_json: Option<String> = conn
            .query_row(
                "SELECT tags FROM knowledge_nodes WHERE id = ?1",
                params![member.memory_id],
                |row| row.get(0),
            )
            .optional()?;
        let Some(tags_json) = tags_json else {
            return Ok(());
        };
        let Ok(tags) = serde_json::from_str::<Vec<String>>(&tags_json) else {
            return Ok(());
        };
        if tags.is_empty() {
            return Ok(());
        }

        if let Some(object) = member.metadata.as_object_mut() {
            object.insert("tags".to_string(), serde_json::json!(tags));
        } else {
            member.metadata = serde_json::json!({ "tags": tags });
        }
        Ok(())
    }

    fn insert_composition_outcome(
        conn: &Connection,
        outcome: &CompositionOutcomeRecord,
    ) -> Result<()> {
        let metadata_json =
            serde_json::to_string(&outcome.metadata).unwrap_or_else(|_| "{}".to_string());
        conn.execute(
            "INSERT OR REPLACE INTO composition_outcomes (
                id, event_id, outcome_type, labeled_at, label_source,
                confidence_delta, notes, metadata
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                outcome.id,
                outcome.event_id,
                outcome.outcome_type,
                outcome.labeled_at.to_rfc3339(),
                outcome.label_source,
                outcome.confidence_delta,
                outcome.notes,
                metadata_json,
            ],
        )?;
        Ok(())
    }

    fn row_to_composition_event(row: &rusqlite::Row) -> rusqlite::Result<CompositionEventRecord> {
        let metadata_json: String = row.get("metadata")?;
        Ok(CompositionEventRecord {
            id: row.get("id")?,
            created_at: Self::parse_timestamp(&row.get::<_, String>("created_at")?, "created_at")?,
            tool: row.get("tool")?,
            mode: row.get("mode")?,
            query: row.get("query").ok().flatten(),
            query_hash: row.get("query_hash").ok().flatten(),
            confidence: row.get("confidence").ok().flatten(),
            status: row.get("status").ok().flatten(),
            output_preview: row.get("output_preview").ok().flatten(),
            metadata: serde_json::from_str(&metadata_json)
                .unwrap_or_else(|_| serde_json::json!({})),
        })
    }

    fn row_to_composition_member(row: &rusqlite::Row) -> rusqlite::Result<CompositionMemberRecord> {
        let metadata_json: String = row.get("metadata")?;
        Ok(CompositionMemberRecord {
            event_id: row.get("event_id")?,
            memory_id: row.get("memory_id")?,
            role: row.get("role")?,
            rank: row.get("rank").unwrap_or(0),
            trust: row.get("trust").ok().flatten(),
            score: row.get("score").ok().flatten(),
            preview: row.get("preview").ok().flatten(),
            metadata: serde_json::from_str(&metadata_json)
                .unwrap_or_else(|_| serde_json::json!({})),
        })
    }

    fn row_to_composition_outcome(
        row: &rusqlite::Row,
    ) -> rusqlite::Result<CompositionOutcomeRecord> {
        let metadata_json: String = row.get("metadata")?;
        Ok(CompositionOutcomeRecord {
            id: row.get("id")?,
            event_id: row.get("event_id")?,
            outcome_type: row.get("outcome_type")?,
            labeled_at: Self::parse_timestamp(&row.get::<_, String>("labeled_at")?, "labeled_at")?,
            label_source: row
                .get("label_source")
                .unwrap_or_else(|_| "tool".to_string()),
            confidence_delta: row.get("confidence_delta").ok().flatten(),
            notes: row.get("notes").ok().flatten(),
            metadata: serde_json::from_str(&metadata_json)
                .unwrap_or_else(|_| serde_json::json!({})),
        })
    }

    fn composition_event_exists(conn: &Connection, id: &str) -> Result<bool> {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM composition_events WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    fn composed_pair_set(&self) -> Result<HashSet<(String, String)>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT event_id, memory_id
             FROM composition_members
             ORDER BY event_id, memory_id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut grouped: HashMap<String, Vec<String>> = HashMap::new();
        for row in rows {
            let (event_id, memory_id) = row?;
            grouped.entry(event_id).or_default().push(memory_id);
        }

        let mut pairs = HashSet::new();
        for ids in grouped.values_mut() {
            ids.sort();
            ids.dedup();
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    pairs.insert(Self::pair_key(&ids[i], &ids[j]));
                }
            }
        }
        Ok(pairs)
    }

    fn pair_key(a: &str, b: &str) -> (String, String) {
        if a <= b {
            (a.to_string(), b.to_string())
        } else {
            (b.to_string(), a.to_string())
        }
    }

    fn shared_tags(a: &[String], b: &[String]) -> Vec<String> {
        let b_set: HashSet<&str> = b.iter().map(String::as_str).collect();
        let mut shared = a
            .iter()
            .filter(|tag| b_set.contains(tag.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        shared.sort();
        shared.dedup();
        shared
    }

    fn node_pair_matches_tag_filter(
        a: &KnowledgeNode,
        b: &KnowledgeNode,
        tag_filter: &[String],
    ) -> bool {
        a.tags.iter().chain(b.tags.iter()).any(|tag| {
            tag_filter
                .iter()
                .any(|wanted| wanted == tag || tag.starts_with(&format!("{wanted}:")))
        })
    }

    fn boundary_tags_for_pair(a: &[String], b: &[String]) -> Vec<String> {
        let mut tags = a
            .iter()
            .chain(b.iter())
            .filter(|tag| Self::is_boundary_tag(tag))
            .cloned()
            .collect::<Vec<_>>();
        tags.sort();
        tags.dedup();
        tags
    }

    fn composition_bridge_score(
        a: &KnowledgeNode,
        b: &KnowledgeNode,
        shared_tags: &[String],
        shared_terms: &[String],
        boundary_tags: &[String],
    ) -> f64 {
        let tag_distance = Self::tag_distance(&a.tags, &b.tags);
        let node_type_bridge = if a.node_type != b.node_type { 1.0 } else { 0.0 };
        let boundary_bridge = (boundary_tags.len() as f64 / 4.0).min(1.0);
        let lexical_anchor = if shared_terms.is_empty() { 0.0 } else { 1.0 };
        let tag_anchor = if shared_tags.is_empty() { 0.0 } else { 1.0 };

        (tag_distance * 0.30
            + node_type_bridge * 0.20
            + boundary_bridge * 0.25
            + lexical_anchor * 0.15
            + tag_anchor * 0.10)
            .clamp(0.0, 1.0)
    }

    fn tag_distance(a: &[String], b: &[String]) -> f64 {
        let a_set = a.iter().map(String::as_str).collect::<HashSet<_>>();
        let b_set = b.iter().map(String::as_str).collect::<HashSet<_>>();
        let union = a_set.union(&b_set).count();
        if union == 0 {
            return 0.0;
        }
        let intersection = a_set.intersection(&b_set).count();
        1.0 - (intersection as f64 / union as f64)
    }

    fn shared_content_terms(a: &str, b: &str, limit: usize) -> Vec<String> {
        let a_terms = Self::content_terms(a);
        let b_terms = Self::content_terms(b);
        let mut shared = a_terms
            .intersection(&b_terms)
            .cloned()
            .collect::<Vec<String>>();
        shared.sort_by(|left, right| {
            Self::term_specificity_score(right)
                .cmp(&Self::term_specificity_score(left))
                .then_with(|| left.cmp(right))
        });
        shared.truncate(limit);
        shared
    }

    fn content_terms(content: &str) -> HashSet<String> {
        const STOPWORDS: &[&str] = &[
            "about", "after", "again", "against", "because", "before", "between", "could", "every",
            "first", "from", "have", "into", "memory", "needs", "should", "their", "there",
            "these", "thing", "through", "using", "where", "which", "while", "would",
        ];
        content
            .to_ascii_lowercase()
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_')
            .filter(|term| term.len() >= 5 && !STOPWORDS.contains(term))
            .map(ToOwned::to_owned)
            .collect()
    }

    fn term_specificity_score(term: &str) -> usize {
        term.len()
            + term.chars().filter(|ch| ch.is_ascii_digit()).count() * 2
            + usize::from(term.contains('-')) * 2
            + usize::from(term.contains('_')) * 2
    }

    fn anchor_summary(shared_tags: &[String], shared_terms: &[String]) -> String {
        if !shared_tags.is_empty() && !shared_terms.is_empty() {
            format!(
                "shared tags ({}) and shared terms ({})",
                shared_tags.join(", "),
                shared_terms
                    .iter()
                    .take(4)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else if !shared_tags.is_empty() {
            format!("shared tags ({})", shared_tags.join(", "))
        } else {
            format!(
                "shared terms ({})",
                shared_terms
                    .iter()
                    .take(4)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }

    fn composition_question(
        a: &KnowledgeNode,
        b: &KnowledgeNode,
        shared_tags: &[String],
        shared_terms: &[String],
        boundary_tags: &[String],
    ) -> String {
        let anchor = if !boundary_tags.is_empty() {
            boundary_tags.join(", ")
        } else if !shared_tags.is_empty() {
            shared_tags.join(", ")
        } else {
            shared_terms
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "What changes if a {} memory and a {} memory are composed through {}?",
            a.node_type, b.node_type, anchor
        )
    }

    fn composition_degree_map(&self) -> Result<HashMap<String, i64>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT memory_id, COUNT(DISTINCT event_id) AS composition_count
             FROM composition_members
             GROUP BY memory_id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;
        let mut result = HashMap::new();
        for row in rows {
            let (memory_id, count) = row?;
            result.insert(memory_id, count);
        }
        Ok(result)
    }

    fn composition_candidate_nodes(
        &self,
        tag_filter: Option<&[String]>,
    ) -> Result<Vec<KnowledgeNode>> {
        const BASE_SCAN_LIMIT: i32 = 750;
        const TAGGED_SCAN_LIMIT: i32 = 1500;

        let mut nodes = self.get_all_nodes(BASE_SCAN_LIMIT, 0)?;
        if let Some(filter) = tag_filter
            && !filter.is_empty()
        {
            let tagged_nodes = self.get_nodes_matching_any_tag_prefix(filter, TAGGED_SCAN_LIMIT)?;
            let mut by_id = HashMap::new();
            for node in nodes.into_iter().chain(tagged_nodes) {
                by_id.entry(node.id.clone()).or_insert(node);
            }
            nodes = by_id.into_values().collect();
            nodes.sort_by(|a, b| {
                b.retention_strength
                    .partial_cmp(&a.retention_strength)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.created_at.cmp(&a.created_at))
            });
        }
        Ok(nodes)
    }

    fn get_nodes_matching_any_tag_prefix(
        &self,
        tag_filter: &[String],
        limit: i32,
    ) -> Result<Vec<KnowledgeNode>> {
        let mut patterns = Vec::new();
        for wanted in tag_filter
            .iter()
            .map(|tag| tag.trim())
            .filter(|tag| !tag.is_empty())
        {
            patterns.push(format!("%\"{}\"%", wanted));
            patterns.push(format!("%\"{}:%", wanted));
        }
        if patterns.is_empty() {
            return Ok(Vec::new());
        }

        let clauses = std::iter::repeat_n("tags LIKE ?", patterns.len())
            .collect::<Vec<_>>()
            .join(" OR ");
        let sql = format!(
            "SELECT * FROM knowledge_nodes
             WHERE {clauses}
             ORDER BY retention_strength DESC, created_at DESC
             LIMIT {}",
            limit.clamp(1, 5000)
        );

        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(patterns.iter()), Self::row_to_node)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    fn composition_outcome_map(&self) -> Result<HashMap<String, HashSet<String>>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT DISTINCT m.memory_id, o.outcome_type
             FROM composition_members m
             JOIN composition_outcomes o ON o.event_id = m.event_id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut result: HashMap<String, HashSet<String>> = HashMap::new();
        for row in rows {
            let (memory_id, outcome) = row?;
            result.entry(memory_id).or_default().insert(outcome);
        }
        Ok(result)
    }

    fn pair_prior_outcomes(
        outcome_map: &HashMap<String, HashSet<String>>,
        first_id: &str,
        second_id: &str,
    ) -> Vec<String> {
        let mut outcomes = outcome_map
            .get(first_id)
            .into_iter()
            .chain(outcome_map.get(second_id))
            .flat_map(|values| values.iter().cloned())
            .collect::<Vec<_>>();
        outcomes.sort();
        outcomes.dedup();
        outcomes
    }

    fn outcome_signal(prior_outcomes: &[String]) -> String {
        if prior_outcomes.is_empty() {
            return "clean".to_string();
        }

        let has_closed = prior_outcomes.iter().any(|outcome| {
            matches!(
                outcome.as_str(),
                "dead_end"
                    | "rejected"
                    | "bad_severity"
                    | "user_demoted"
                    | "closed_by_scope"
                    | "closed_by_false_assumption"
                    | "closed_by_user"
                    | "expired_lane"
            )
        });
        let has_duplicate = prior_outcomes
            .iter()
            .any(|outcome| matches!(outcome.as_str(), "duplicate_risk" | "closed_by_duplicate"));
        let has_success = prior_outcomes.iter().any(|outcome| {
            matches!(
                outcome.as_str(),
                "accepted" | "helpful" | "submitted" | "user_promoted"
            )
        });
        let has_needs_poc = prior_outcomes.iter().any(|outcome| outcome == "needs_poc");

        if (has_closed || has_duplicate) && has_success {
            "mixed_prior_outcomes".to_string()
        } else if has_closed {
            "prior_closed_door".to_string()
        } else if has_duplicate {
            "prior_duplicate_risk".to_string()
        } else if has_success {
            "prior_success".to_string()
        } else if has_needs_poc {
            "prior_needs_poc".to_string()
        } else {
            "prior_outcome".to_string()
        }
    }

    fn outcome_score_adjustment(prior_outcomes: &[String]) -> f64 {
        let mut adjustment: f64 = 0.0;
        for outcome in prior_outcomes {
            adjustment += match outcome.as_str() {
                "accepted" => 0.35,
                "helpful" => 0.25,
                "submitted" => 0.15,
                "user_promoted" => 0.20,
                "needs_poc" => -0.05,
                "duplicate_risk" => -0.35,
                "closed_by_duplicate" => -0.40,
                "dead_end"
                | "rejected"
                | "bad_severity"
                | "closed_by_scope"
                | "closed_by_false_assumption"
                | "closed_by_user"
                | "expired_lane" => -0.45,
                "user_demoted" => -0.20,
                _ => 0.0,
            };
        }
        adjustment.clamp(-0.8, 0.5)
    }

    fn is_boundary_tag(tag: &str) -> bool {
        let lowered = tag.to_ascii_lowercase();
        lowered.starts_with("boundary-")
            || matches!(
                lowered.as_str(),
                "time"
                    | "chain"
                    | "role"
                    | "oracle"
                    | "queue"
                    | "settlement"
                    | "keeper"
                    | "upgrade"
                    | "pause"
                    | "accounting"
                    | "scope"
            )
    }

    // ========================================================================
    // INTENTIONS PERSISTENCE
    // ========================================================================

    /// Save an intention to the database
    pub fn save_intention(&self, intention: &IntentionRecord) -> Result<()> {
        let tags_json = serde_json::to_string(&intention.tags).unwrap_or_else(|_| "[]".to_string());
        let related_json =
            serde_json::to_string(&intention.related_memories).unwrap_or_else(|_| "[]".to_string());

        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "INSERT OR REPLACE INTO intentions (
                id, content, trigger_type, trigger_data, priority, status,
                created_at, deadline, fulfilled_at, reminder_count, last_reminded_at,
                notes, tags, related_memories, snoozed_until, source_type, source_data
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            params![
                intention.id,
                intention.content,
                intention.trigger_type,
                intention.trigger_data,
                intention.priority,
                intention.status,
                intention.created_at.to_rfc3339(),
                intention.deadline.map(|dt| dt.to_rfc3339()),
                intention.fulfilled_at.map(|dt| dt.to_rfc3339()),
                intention.reminder_count,
                intention.last_reminded_at.map(|dt| dt.to_rfc3339()),
                intention.notes,
                tags_json,
                related_json,
                intention.snoozed_until.map(|dt| dt.to_rfc3339()),
                intention.source_type,
                intention.source_data,
            ],
        )?;
        Ok(())
    }

    /// Get an intention by ID
    pub fn get_intention(&self, id: &str) -> Result<Option<IntentionRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare("SELECT * FROM intentions WHERE id = ?1")?;

        stmt.query_row(params![id], Self::row_to_intention)
            .optional()
            .map_err(StorageError::from)
    }

    /// Get all active intentions
    pub fn get_active_intentions(&self) -> Result<Vec<IntentionRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT * FROM intentions WHERE status = 'active' ORDER BY priority DESC, created_at ASC"
        )?;

        let rows = stmt.query_map([], Self::row_to_intention)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Get intentions by status
    pub fn get_intentions_by_status(&self, status: &str) -> Result<Vec<IntentionRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT * FROM intentions WHERE status = ?1 ORDER BY priority DESC, created_at ASC",
        )?;

        let rows = stmt.query_map(params![status], Self::row_to_intention)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Update intention status
    pub fn update_intention_status(&self, id: &str, status: &str) -> Result<bool> {
        let now = Utc::now();
        let fulfilled_at = if status == "fulfilled" {
            Some(now.to_rfc3339())
        } else {
            None
        };

        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let rows = writer.execute(
            "UPDATE intentions SET status = ?1, fulfilled_at = ?2 WHERE id = ?3",
            params![status, fulfilled_at, id],
        )?;
        Ok(rows > 0)
    }

    /// Delete an intention
    pub fn delete_intention(&self, id: &str) -> Result<bool> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let rows = writer.execute("DELETE FROM intentions WHERE id = ?1", params![id])?;
        Ok(rows > 0)
    }

    /// Get overdue intentions
    pub fn get_overdue_intentions(&self) -> Result<Vec<IntentionRecord>> {
        let now = Utc::now().to_rfc3339();
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT * FROM intentions WHERE status = 'active' AND deadline IS NOT NULL AND deadline < ?1 ORDER BY deadline ASC"
        )?;

        let rows = stmt.query_map(params![now], Self::row_to_intention)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Snooze an intention
    pub fn snooze_intention(&self, id: &str, until: DateTime<Utc>) -> Result<bool> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let rows = writer.execute(
            "UPDATE intentions SET status = 'snoozed', snoozed_until = ?1 WHERE id = ?2",
            params![until.to_rfc3339(), id],
        )?;
        Ok(rows > 0)
    }

    fn row_to_intention(row: &rusqlite::Row) -> rusqlite::Result<IntentionRecord> {
        let tags_json: String = row.get("tags")?;
        let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
        let related_json: String = row.get("related_memories")?;
        let related: Vec<String> = serde_json::from_str(&related_json).unwrap_or_default();

        let parse_opt_dt = |s: Option<String>| -> Option<DateTime<Utc>> {
            s.and_then(|v| {
                DateTime::parse_from_rfc3339(&v)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            })
        };

        Ok(IntentionRecord {
            id: row.get("id")?,
            content: row.get("content")?,
            trigger_type: row.get("trigger_type")?,
            trigger_data: row.get("trigger_data")?,
            priority: row.get("priority")?,
            status: row.get("status")?,
            created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>("created_at")?)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            deadline: parse_opt_dt(row.get("deadline").ok().flatten()),
            fulfilled_at: parse_opt_dt(row.get("fulfilled_at").ok().flatten()),
            reminder_count: row.get("reminder_count").unwrap_or(0),
            last_reminded_at: parse_opt_dt(row.get("last_reminded_at").ok().flatten()),
            notes: row.get("notes").ok().flatten(),
            tags,
            related_memories: related,
            snoozed_until: parse_opt_dt(row.get("snoozed_until").ok().flatten()),
            source_type: row.get("source_type").unwrap_or_else(|_| "api".to_string()),
            source_data: row.get("source_data").ok().flatten(),
        })
    }

    // ========================================================================
    // INSIGHTS PERSISTENCE
    // ========================================================================

    /// Save an insight to the database
    pub fn save_insight(&self, insight: &InsightRecord) -> Result<()> {
        let source_json =
            serde_json::to_string(&insight.source_memories).unwrap_or_else(|_| "[]".to_string());
        let tags_json = serde_json::to_string(&insight.tags).unwrap_or_else(|_| "[]".to_string());

        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "INSERT OR REPLACE INTO insights (
                id, insight, source_memories, confidence, novelty_score, insight_type,
                generated_at, tags, feedback, applied_count
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                insight.id,
                insight.insight,
                source_json,
                insight.confidence,
                insight.novelty_score,
                insight.insight_type,
                insight.generated_at.to_rfc3339(),
                tags_json,
                insight.feedback,
                insight.applied_count,
            ],
        )?;
        Ok(())
    }

    /// Get insights with optional limit
    pub fn get_insights(&self, limit: i32) -> Result<Vec<InsightRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt =
            reader.prepare("SELECT * FROM insights ORDER BY generated_at DESC LIMIT ?1")?;

        let rows = stmt.query_map(params![limit], Self::row_to_insight)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Get insights without feedback (pending review)
    pub fn get_pending_insights(&self) -> Result<Vec<InsightRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader
            .prepare("SELECT * FROM insights WHERE feedback IS NULL ORDER BY novelty_score DESC")?;

        let rows = stmt.query_map([], Self::row_to_insight)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Mark insight feedback
    pub fn mark_insight_feedback(&self, id: &str, feedback: &str) -> Result<bool> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let rows = writer.execute(
            "UPDATE insights SET feedback = ?1 WHERE id = ?2",
            params![feedback, id],
        )?;
        Ok(rows > 0)
    }

    /// Clear all insights
    pub fn clear_insights(&self) -> Result<i32> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let count: i32 = writer.query_row("SELECT COUNT(*) FROM insights", [], |row| row.get(0))?;
        writer.execute("DELETE FROM insights", [])?;
        Ok(count)
    }

    fn row_to_insight(row: &rusqlite::Row) -> rusqlite::Result<InsightRecord> {
        let source_json: String = row.get("source_memories")?;
        let source_memories: Vec<String> = serde_json::from_str(&source_json).unwrap_or_default();
        let tags_json: String = row.get("tags")?;
        let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();

        Ok(InsightRecord {
            id: row.get("id")?,
            insight: row.get("insight")?,
            source_memories,
            confidence: row.get("confidence")?,
            novelty_score: row.get("novelty_score")?,
            insight_type: row.get("insight_type")?,
            generated_at: DateTime::parse_from_rfc3339(&row.get::<_, String>("generated_at")?)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            tags,
            feedback: row.get("feedback").ok().flatten(),
            applied_count: row.get("applied_count").unwrap_or(0),
        })
    }

    // ========================================================================
    // MEMORY CONNECTIONS PERSISTENCE (Activation Network)
    // ========================================================================

    /// Save a memory connection
    pub fn save_connection(&self, connection: &ConnectionRecord) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "INSERT OR REPLACE INTO memory_connections (
                source_id, target_id, strength, link_type, created_at, last_activated, activation_count
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                connection.source_id,
                connection.target_id,
                connection.strength,
                connection.link_type,
                connection.created_at.to_rfc3339(),
                connection.last_activated.to_rfc3339(),
                connection.activation_count,
            ],
        )?;
        Ok(())
    }

    /// Get connections for a memory
    pub fn get_connections_for_memory(&self, memory_id: &str) -> Result<Vec<ConnectionRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT * FROM memory_connections WHERE source_id = ?1 OR target_id = ?1 ORDER BY strength DESC"
        )?;

        let rows = stmt.query_map(params![memory_id], Self::row_to_connection)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Get all connections (for building activation network)
    pub fn get_all_connections(&self) -> Result<Vec<ConnectionRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare("SELECT * FROM memory_connections ORDER BY strength DESC")?;

        let rows = stmt.query_map([], Self::row_to_connection)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// The most recently created connections, capped at `limit`. Used by polling
    /// surfaces (e.g. the dashboard changelog) that only need recent activity and
    /// must not load the entire `memory_connections` table on every request.
    pub fn get_recent_connections(&self, limit: usize) -> Result<Vec<ConnectionRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt =
            reader.prepare("SELECT * FROM memory_connections ORDER BY created_at DESC LIMIT ?1")?;
        let rows = stmt.query_map([limit as i64], Self::row_to_connection)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Strengthen a connection
    pub fn strengthen_connection(
        &self,
        source_id: &str,
        target_id: &str,
        boost: f64,
    ) -> Result<bool> {
        let now = Utc::now().to_rfc3339();
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let rows = writer.execute(
            "UPDATE memory_connections SET
                strength = MIN(strength + ?1, 1.0),
                last_activated = ?2,
                activation_count = activation_count + 1
             WHERE source_id = ?3 AND target_id = ?4",
            params![boost, now, source_id, target_id],
        )?;
        Ok(rows > 0)
    }

    /// Apply decay to all connections
    pub fn apply_connection_decay(&self, decay_factor: f64) -> Result<i32> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let rows = writer.execute(
            "UPDATE memory_connections SET strength = strength * ?1",
            params![decay_factor],
        )?;
        Ok(rows as i32)
    }

    /// Prune weak connections below threshold
    pub fn prune_weak_connections(&self, min_strength: f64) -> Result<i32> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let rows = writer.execute(
            "DELETE FROM memory_connections WHERE strength < ?1",
            params![min_strength],
        )?;
        Ok(rows as i32)
    }

    fn row_to_connection(row: &rusqlite::Row) -> rusqlite::Result<ConnectionRecord> {
        Ok(ConnectionRecord {
            source_id: row.get("source_id")?,
            target_id: row.get("target_id")?,
            strength: row.get("strength")?,
            link_type: row.get("link_type")?,
            created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>("created_at")?)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            last_activated: DateTime::parse_from_rfc3339(&row.get::<_, String>("last_activated")?)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            activation_count: row.get("activation_count").unwrap_or(0),
        })
    }

    // ========================================================================
    // MEMORY STATES PERSISTENCE
    // ========================================================================

    /// Save or update memory state
    pub fn save_memory_state(&self, state: &MemoryStateRecord) -> Result<()> {
        let suppressed_json =
            serde_json::to_string(&state.suppressed_by).unwrap_or_else(|_| "[]".to_string());

        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "INSERT OR REPLACE INTO memory_states (
                memory_id, state, last_access, access_count, state_entered_at,
                suppression_until, suppressed_by
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                state.memory_id,
                state.state,
                state.last_access.to_rfc3339(),
                state.access_count,
                state.state_entered_at.to_rfc3339(),
                state.suppression_until.map(|dt| dt.to_rfc3339()),
                suppressed_json,
            ],
        )?;
        Ok(())
    }

    /// Get memory state
    pub fn get_memory_state(&self, memory_id: &str) -> Result<Option<MemoryStateRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare("SELECT * FROM memory_states WHERE memory_id = ?1")?;

        stmt.query_row(params![memory_id], Self::row_to_memory_state)
            .optional()
            .map_err(StorageError::from)
    }

    /// Get memories by state
    pub fn get_memories_by_state(&self, state: &str) -> Result<Vec<String>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare("SELECT memory_id FROM memory_states WHERE state = ?1")?;

        let rows = stmt.query_map(params![state], |row| row.get::<_, String>(0))?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Update memory state
    pub fn update_memory_state(
        &self,
        memory_id: &str,
        new_state: &str,
        reason: &str,
    ) -> Result<bool> {
        let now = Utc::now();

        // Get old state for transition record
        if let Some(old_record) = self.get_memory_state(memory_id)? {
            // Record state transition
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            writer.execute(
                "INSERT INTO state_transitions (memory_id, from_state, to_state, reason_type, timestamp)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![memory_id, old_record.state, new_state, reason, now.to_rfc3339()],
            )?;
        }

        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let rows = writer.execute(
            "UPDATE memory_states SET state = ?1, state_entered_at = ?2 WHERE memory_id = ?3",
            params![new_state, now.to_rfc3339(), memory_id],
        )?;
        Ok(rows > 0)
    }

    /// Record access to memory (updates state)
    pub fn record_memory_access(&self, memory_id: &str) -> Result<()> {
        let now = Utc::now();

        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;

        // Check if state exists (writer can read too)
        let exists: bool = writer.query_row(
            "SELECT EXISTS(SELECT 1 FROM memory_states WHERE memory_id = ?1)",
            params![memory_id],
            |row| row.get(0),
        )?;

        if exists {
            writer.execute(
                "UPDATE memory_states SET
                    last_access = ?1,
                    access_count = access_count + 1,
                    state = 'active',
                    state_entered_at = CASE WHEN state != 'active' THEN ?1 ELSE state_entered_at END
                 WHERE memory_id = ?2",
                params![now.to_rfc3339(), memory_id],
            )?;
        } else {
            writer.execute(
                "INSERT INTO memory_states (memory_id, state, last_access, access_count, state_entered_at)
                 VALUES (?1, 'active', ?2, 1, ?2)",
                params![memory_id, now.to_rfc3339()],
            )?;
        }
        Ok(())
    }

    fn row_to_memory_state(row: &rusqlite::Row) -> rusqlite::Result<MemoryStateRecord> {
        let suppressed_json: String = row.get("suppressed_by")?;
        let suppressed_by: Vec<String> = serde_json::from_str(&suppressed_json).unwrap_or_default();

        let parse_opt_dt = |s: Option<String>| -> Option<DateTime<Utc>> {
            s.and_then(|v| {
                DateTime::parse_from_rfc3339(&v)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            })
        };

        Ok(MemoryStateRecord {
            memory_id: row.get("memory_id")?,
            state: row.get("state")?,
            last_access: DateTime::parse_from_rfc3339(&row.get::<_, String>("last_access")?)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            access_count: row.get("access_count").unwrap_or(1),
            state_entered_at: DateTime::parse_from_rfc3339(
                &row.get::<_, String>("state_entered_at")?,
            )
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
            suppression_until: parse_opt_dt(row.get("suppression_until").ok().flatten()),
            suppressed_by,
        })
    }

    // ========================================================================
    // CONSOLIDATION HISTORY
    // ========================================================================

    /// Save consolidation history record
    pub fn save_consolidation_history(&self, record: &ConsolidationHistoryRecord) -> Result<i64> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "INSERT INTO consolidation_history (
                completed_at, duration_ms, memories_replayed, connections_found,
                connections_strengthened, connections_pruned, insights_generated
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                record.completed_at.to_rfc3339(),
                record.duration_ms,
                record.memories_replayed,
                record.connections_found,
                record.connections_strengthened,
                record.connections_pruned,
                record.insights_generated,
            ],
        )?;
        Ok(writer.last_insert_rowid())
    }

    /// Get last consolidation timestamp
    pub fn get_last_consolidation(&self) -> Result<Option<DateTime<Utc>>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let result: Option<String> = reader
            .query_row(
                "SELECT MAX(completed_at) FROM consolidation_history",
                [],
                |row| row.get(0),
            )
            .ok()
            .flatten();

        Ok(result.and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .ok()
                .map(|dt| dt.with_timezone(&Utc))
        }))
    }

    /// Get consolidation history
    pub fn get_consolidation_history(&self, limit: i32) -> Result<Vec<ConsolidationHistoryRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader
            .prepare("SELECT * FROM consolidation_history ORDER BY completed_at DESC LIMIT ?1")?;

        let rows = stmt.query_map(params![limit], |row| {
            Ok(ConsolidationHistoryRecord {
                id: row.get("id")?,
                completed_at: DateTime::parse_from_rfc3339(&row.get::<_, String>("completed_at")?)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                duration_ms: row.get("duration_ms")?,
                memories_replayed: row.get("memories_replayed").unwrap_or(0),
                connections_found: row.get("connections_found").unwrap_or(0),
                connections_strengthened: row.get("connections_strengthened").unwrap_or(0),
                connections_pruned: row.get("connections_pruned").unwrap_or(0),
                insights_generated: row.get("insights_generated").unwrap_or(0),
            })
        })?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    // ========================================================================
    // DREAM HISTORY PERSISTENCE
    // ========================================================================

    /// Save a dream history record
    pub fn save_dream_history(&self, record: &DreamHistoryRecord) -> Result<i64> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "INSERT INTO dream_history (
                dreamed_at, duration_ms, memories_replayed, connections_found,
                insights_generated, memories_strengthened, memories_compressed,
                phase_nrem1_ms, phase_nrem3_ms, phase_rem_ms, phase_integration_ms,
                summaries_generated, emotional_memories_processed, creative_connections_found
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                record.dreamed_at.to_rfc3339(),
                record.duration_ms,
                record.memories_replayed,
                record.connections_found,
                record.insights_generated,
                record.memories_strengthened,
                record.memories_compressed,
                record.phase_nrem1_ms,
                record.phase_nrem3_ms,
                record.phase_rem_ms,
                record.phase_integration_ms,
                record.summaries_generated,
                record.emotional_memories_processed,
                record.creative_connections_found,
            ],
        )?;
        Ok(writer.last_insert_rowid())
    }

    /// Get last dream timestamp
    pub fn get_last_dream(&self) -> Result<Option<DateTime<Utc>>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let result: Option<String> = reader
            .query_row("SELECT MAX(dreamed_at) FROM dream_history", [], |row| {
                row.get(0)
            })
            .ok()
            .flatten();

        Ok(result.and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .ok()
                .map(|dt| dt.with_timezone(&Utc))
        }))
    }

    /// Get dream history (most recent first)
    pub fn get_dream_history(&self, limit: i32) -> Result<Vec<DreamHistoryRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT dreamed_at, duration_ms, memories_replayed, connections_found,
                    insights_generated, memories_strengthened, memories_compressed,
                    phase_nrem1_ms, phase_nrem3_ms, phase_rem_ms, phase_integration_ms,
                    summaries_generated, emotional_memories_processed, creative_connections_found
             FROM dream_history ORDER BY dreamed_at DESC LIMIT ?1",
        )?;
        let records = stmt
            .query_map(params![limit], |row| {
                let dreamed_at_str: String = row.get(0)?;
                let dreamed_at = DateTime::parse_from_rfc3339(&dreamed_at_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());
                Ok(DreamHistoryRecord {
                    dreamed_at,
                    duration_ms: row.get(1)?,
                    memories_replayed: row.get(2)?,
                    connections_found: row.get(3)?,
                    insights_generated: row.get(4)?,
                    memories_strengthened: row.get(5)?,
                    memories_compressed: row.get(6)?,
                    phase_nrem1_ms: row.get(7)?,
                    phase_nrem3_ms: row.get(8)?,
                    phase_rem_ms: row.get(9)?,
                    phase_integration_ms: row.get(10)?,
                    summaries_generated: row.get(11)?,
                    emotional_memories_processed: row.get(12)?,
                    creative_connections_found: row.get(13)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(records)
    }

    /// Count memories created since a given timestamp
    pub fn count_memories_since(&self, since: DateTime<Utc>) -> Result<i64> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let count: i64 = reader.query_row(
            "SELECT COUNT(*) FROM knowledge_nodes WHERE created_at >= ?1",
            params![since.to_rfc3339()],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    fn scan_last_backup_timestamp(backup_dir: &Path) -> Option<DateTime<Utc>> {
        if !backup_dir.exists() {
            return None;
        }

        let mut latest: Option<DateTime<Utc>> = None;

        if let Ok(entries) = std::fs::read_dir(backup_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                // Parse vestige-YYYYMMDD-HHMMSS.db
                if let Some(ts_part) = name_str
                    .strip_prefix("vestige-")
                    .and_then(|s| s.strip_suffix(".db"))
                    && let Ok(naive) =
                        chrono::NaiveDateTime::parse_from_str(ts_part, "%Y%m%d-%H%M%S")
                {
                    let dt = naive.and_utc();
                    if latest.as_ref().is_none_or(|l| dt > *l) {
                        latest = Some(dt);
                    }
                }
            }
        }

        latest
    }

    /// Get last backup timestamp for this storage instance.
    /// Parses `vestige-YYYYMMDD-HHMMSS.db` filenames.
    pub fn last_backup_timestamp(&self) -> Option<DateTime<Utc>> {
        Self::scan_last_backup_timestamp(&self.sidecar_dir("backups"))
    }

    /// Get last backup timestamp in the default backups directory.
    /// Kept for compatibility with older callers.
    pub fn get_last_backup_timestamp() -> Option<DateTime<Utc>> {
        let backup_dir = Self::default_db_path().ok()?.parent()?.join("backups");
        Self::scan_last_backup_timestamp(&backup_dir)
    }

    /// Export an exact portable archive preserving raw Vestige storage rows.
    ///
    /// Unlike the user-facing JSON export, this preserves IDs, timestamps,
    /// FSRS state, graph edges, suppression state, history tables, and raw
    /// embedding blobs. It is intended for Vestige-to-Vestige device transfer.
    pub fn export_portable_archive(&self) -> Result<PortableArchive> {
        let mut reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let tx = reader.transaction()?;

        let schema_version = Self::current_schema_version(&tx)?;
        let mut tables = Vec::new();

        for table_name in PORTABLE_TABLES {
            if !Self::table_exists(&tx, table_name)? {
                continue;
            }

            let quoted_table = Self::quote_ident(table_name);
            let mut stmt = tx.prepare(&format!("SELECT * FROM {} ORDER BY rowid", quoted_table))?;
            let columns: Vec<String> = stmt
                .column_names()
                .iter()
                .map(|name| (*name).to_string())
                .collect();
            let column_count = columns.len();

            let rows = stmt.query_map([], |row| {
                let mut values = Vec::with_capacity(column_count);
                for idx in 0..column_count {
                    values.push(Self::portable_value_from_ref(row.get_ref(idx)?)?);
                }
                Ok(values)
            })?;

            let mut portable_rows = Vec::new();
            for row in rows {
                portable_rows.push(row?);
            }

            tables.push(PortableTable {
                name: (*table_name).to_string(),
                columns,
                rows: portable_rows,
            });
        }

        let archive = PortableArchive {
            archive_format: PORTABLE_ARCHIVE_FORMAT.to_string(),
            vestige_version: crate::VERSION.to_string(),
            schema_version,
            exported_at: Utc::now(),
            mode: "exact".to_string(),
            tables,
        };
        tx.commit()?;
        Ok(archive)
    }

    /// Write an exact portable archive to a JSON file.
    pub fn export_portable_archive_to_path(
        &self,
        path: &std::path::Path,
    ) -> Result<PortableArchive> {
        let archive = self.export_portable_archive()?;
        let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("vestige-portable.json");
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
        if let Err(e) = serde_json::to_writer_pretty(&mut file, &archive) {
            let _ = std::fs::remove_file(&temp_path);
            return Err(StorageError::Init(format!(
                "Failed to write portable archive: {}",
                e
            )));
        }
        file.flush()?;
        file.sync_all()?;
        drop(file);

        if let Err(rename_err) = std::fs::rename(&temp_path, path) {
            if path.exists() {
                std::fs::remove_file(path)?;
                std::fs::rename(&temp_path, path)?;
            } else {
                let _ = std::fs::remove_file(&temp_path);
                return Err(rename_err.into());
            }
        }
        Ok(archive)
    }

    /// Import an exact portable archive.
    ///
    /// `EmptyOnly` preserves the conservative migration path. `Merge` is used
    /// by portable sync to combine non-empty databases with tombstones and
    /// newer-local conflict handling.
    pub fn import_portable_archive(
        &self,
        archive: &PortableArchive,
        mode: PortableImportMode,
    ) -> Result<PortableImportReport> {
        self.import_portable_archive_with_secret_policy(archive, mode, SecretPolicy::Reject)
    }

    /// Import an exact archive using an explicit credential-storage policy.
    ///
    /// The archive is preflighted before a writer or transaction is opened, so
    /// a rejected archive cannot partially import safe sibling rows.
    pub fn import_portable_archive_with_secret_policy(
        &self,
        archive: &PortableArchive,
        mode: PortableImportMode,
        policy: SecretPolicy,
    ) -> Result<PortableImportReport> {
        if archive.archive_format != PORTABLE_ARCHIVE_FORMAT {
            return Err(StorageError::Init(format!(
                "Unsupported portable archive format '{}'",
                archive.archive_format
            )));
        }
        if archive.mode != "exact" {
            return Err(StorageError::Init(format!(
                "Unsupported portable archive mode '{}'",
                archive.mode
            )));
        }

        Self::enforce_secret_policy_for_portable_archive(archive, policy)?;

        let mut seen_tables = std::collections::HashSet::new();
        let mut tables_by_name = std::collections::HashMap::new();
        for table in &archive.tables {
            if !PORTABLE_TABLES.contains(&table.name.as_str()) {
                return Err(StorageError::Init(format!(
                    "Portable archive contains unsupported table '{}'",
                    table.name
                )));
            }
            if !seen_tables.insert(table.name.as_str()) {
                return Err(StorageError::Init(format!(
                    "Portable archive contains duplicate table '{}'",
                    table.name
                )));
            }
            tables_by_name.insert(table.name.as_str(), table);
        }

        let mut report = PortableImportReport {
            tables_imported: 0,
            rows_imported: 0,
            tables_skipped: 0,
            fts_rebuilt: false,
            rows_inserted: 0,
            rows_updated: 0,
            rows_skipped: 0,
            rows_deleted: 0,
            conflicts_kept_local: 0,
        };

        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;

            let current_schema = Self::current_schema_version(&writer)?;
            if archive.schema_version > current_schema {
                return Err(StorageError::Init(format!(
                    "Archive schema version {} is newer than this Vestige database schema {}",
                    archive.schema_version, current_schema
                )));
            }

            match mode {
                PortableImportMode::EmptyOnly => {
                    Self::ensure_portable_import_target_empty(&writer)?
                }
                PortableImportMode::Merge => {}
            }

            let tx = Self::begin_write_transaction(&writer, "import_portable_archive_with_secret_policy")?;
            let mut merge_state = PortableMergeState::default();

            for table_name in PORTABLE_TABLES {
                let Some(table) = tables_by_name.get(table_name) else {
                    continue;
                };

                if !Self::table_exists(&tx, table_name)? {
                    report.tables_skipped += 1;
                    continue;
                }

                if mode == PortableImportMode::Merge {
                    Self::merge_portable_table(
                        &tx,
                        table_name,
                        table,
                        &mut report,
                        &mut merge_state,
                    )?;
                    report.tables_imported += 1;
                    continue;
                }

                let target_columns = Self::table_columns(&tx, table_name)?;
                let mut insert_columns = Vec::new();
                let mut source_indexes = Vec::new();

                for (idx, column) in table.columns.iter().enumerate() {
                    if target_columns.iter().any(|target| target == column) {
                        insert_columns.push(column.clone());
                        source_indexes.push(idx);
                    }
                }

                if insert_columns.is_empty() {
                    report.tables_skipped += 1;
                    continue;
                }

                let quoted_table = Self::quote_ident(table_name);
                let quoted_columns = insert_columns
                    .iter()
                    .map(|column| Self::quote_ident(column))
                    .collect::<Vec<_>>()
                    .join(", ");
                let placeholders = std::iter::repeat_n("?", insert_columns.len())
                    .collect::<Vec<_>>()
                    .join(", ");
                let verb = if *table_name == "fsrs_config" {
                    "INSERT OR REPLACE"
                } else {
                    "INSERT"
                };
                let sql = format!(
                    "{} INTO {} ({}) VALUES ({})",
                    verb, quoted_table, quoted_columns, placeholders
                );

                for row in &table.rows {
                    if row.len() != table.columns.len() {
                        return Err(StorageError::Init(format!(
                            "Portable archive row in table '{}' has {} values for {} columns",
                            table_name,
                            row.len(),
                            table.columns.len()
                        )));
                    }

                    let values = source_indexes
                        .iter()
                        .map(|idx| row[*idx].to_sql_value())
                        .collect::<std::result::Result<Vec<_>, _>>()
                        .map_err(|e| {
                            StorageError::Init(format!("Invalid portable value: {}", e))
                        })?;
                    tx.execute(&sql, params_from_iter(values))?;
                    report.rows_imported += 1;
                    report.rows_inserted += 1;
                }

                report.tables_imported += 1;
            }

            if Self::table_exists(&tx, "knowledge_fts")? {
                tx.execute(
                    "INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild')",
                    [],
                )?;
                report.fts_rebuilt = true;
            }

            tx.commit()?;
        }

        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        self.load_embeddings_into_index()?;

        Ok(report)
    }

    /// Read and import an exact portable archive JSON file.
    pub fn import_portable_archive_from_path(
        &self,
        path: &std::path::Path,
        mode: PortableImportMode,
    ) -> Result<PortableImportReport> {
        let file = std::fs::File::open(path)?;
        let archive: PortableArchive = serde_json::from_reader(file)
            .map_err(|e| StorageError::Init(format!("Failed to parse portable archive: {}", e)))?;
        self.import_portable_archive(&archive, mode)
    }

    /// Synchronize this database with a pluggable portable archive backend.
    ///
    /// Sync is pull-merge-push:
    /// 1. read remote archive if present,
    /// 2. merge it into the local database using tombstones and conflict rules,
    /// 3. export the merged local database,
    /// 4. write the archive back through the backend.
    pub fn sync_portable_archive<B: PortableSyncBackend>(
        &self,
        backend: &B,
    ) -> Result<PortableSyncReport> {
        let (pulled, pull) = match backend.read_archive()? {
            Some(remote) => (
                true,
                Some(self.import_portable_archive(&remote, PortableImportMode::Merge)?),
            ),
            None => (false, None),
        };

        let archive = self.export_portable_archive()?;
        let pushed_tables = archive.tables.len();
        let pushed_rows = archive.total_rows();
        let archive_format = archive.archive_format.clone();
        backend.write_archive(&archive)?;

        Ok(PortableSyncReport {
            backend: backend.label(),
            pulled,
            pull,
            pushed_tables,
            pushed_rows,
            archive_format,
        })
    }

    /// Synchronize this database with a file-backed portable archive.
    pub fn sync_portable_archive_file(&self, path: &std::path::Path) -> Result<PortableSyncReport> {
        let backend = FilePortableSyncBackend::new(path);
        self.sync_portable_archive(&backend)
    }

    /// Synchronize this database with the hosted Vestige Cloud managed-sync
    /// service. `endpoint` is the base URL (e.g. `https://sync.vestige.dev`) and
    /// `sync_key` is the per-user key issued at purchase. Pull-merge-push is
    /// identical to file sync — only the transport differs.
    ///
    /// When `encryption_key` is `Some`, the archive is encrypted client-side
    /// (XChaCha20-Poly1305) before upload, so the server only stores ciphertext
    /// (zero-knowledge). The passphrase never leaves this process.
    #[cfg(feature = "cloud-sync")]
    pub fn sync_portable_archive_cloud(
        &self,
        endpoint: &str,
        sync_key: &str,
        encryption_key: Option<String>,
    ) -> Result<PortableSyncReport> {
        let backend = super::cloud_sync::HttpPortableSyncBackend::new_with_encryption(
            endpoint,
            sync_key,
            encryption_key,
        )?;
        self.sync_portable_archive(&backend)
    }

    fn merge_portable_table(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        table: &PortableTable,
        report: &mut PortableImportReport,
        state: &mut PortableMergeState,
    ) -> Result<()> {
        match table_name {
            "sync_tombstones" => Self::merge_sync_tombstones(tx, table, report),
            "knowledge_nodes" => Self::merge_knowledge_nodes(tx, table, report, state),
            "memory_access_log"
            | "state_transitions"
            | "consolidation_history"
            | "dream_history"
            | "retention_snapshots" => Self::merge_append_only_table(tx, table_name, table, report),
            "composition_events" | "composition_outcomes" => {
                Self::merge_keyed_table(tx, table_name, table, &["id"], report, state)
            }
            "composition_members" => Self::merge_keyed_table(
                tx,
                table_name,
                table,
                &["event_id", "memory_id", "role"],
                report,
                state,
            ),
            "node_embeddings" => {
                Self::merge_keyed_table(tx, table_name, table, &["node_id"], report, state)
            }
            "fsrs_cards" | "memory_states" => {
                Self::merge_keyed_table(tx, table_name, table, &["memory_id"], report, state)
            }
            "deletion_tombstones" => Self::merge_deletion_tombstones(tx, table, report),
            "memory_connections" => Self::merge_keyed_table(
                tx,
                table_name,
                table,
                &["source_id", "target_id"],
                report,
                state,
            ),
            "intentions" | "insights" | "sessions" => {
                Self::merge_keyed_table(tx, table_name, table, &["id"], report, state)
            }
            "fsrs_config" => {
                Self::merge_keyed_table(tx, table_name, table, &["key"], report, state)
            }
            _ => {
                report.tables_skipped += 1;
                Ok(())
            }
        }
    }

    fn merge_knowledge_nodes(
        tx: &rusqlite::Transaction<'_>,
        table: &PortableTable,
        report: &mut PortableImportReport,
        state: &mut PortableMergeState,
    ) -> Result<()> {
        for row in &table.rows {
            let Some(id) = Self::portable_text(table, row, "id") else {
                report.rows_skipped += 1;
                continue;
            };
            let incoming_updated = Self::portable_timestamp(table, row, "updated_at");

            // An opaque marker represents an explicit purge. Unlike legacy raw
            // tombstones, it is intentionally permanent: no timestamp from a
            // later archive can resurrect the same stable id.
            let rejected_by_opaque_tombstone =
                Self::has_opaque_tombstone(tx, "knowledge_nodes", id)?;
            let rejected_by_legacy_tombstone =
                Self::tombstone_timestamp(tx, "knowledge_nodes", id)?.is_some_and(|deleted_at| {
                    incoming_updated.is_some_and(|updated| deleted_at >= updated)
                });
            if rejected_by_opaque_tombstone || rejected_by_legacy_tombstone {
                report.conflicts_kept_local += 1;
                report.rows_skipped += 1;
                continue;
            }

            let existing_updated: Option<String> = tx
                .query_row(
                    "SELECT updated_at FROM knowledge_nodes WHERE id = ?1",
                    params![id],
                    |row| row.get(0),
                )
                .optional()?;

            if let (Some(existing), Some(incoming)) = (
                existing_updated
                    .as_deref()
                    .and_then(Self::parse_rfc3339_opt),
                incoming_updated,
            ) && existing > incoming
            {
                state.locally_newer_nodes.insert(id.to_string());
                report.conflicts_kept_local += 1;
                report.rows_skipped += 1;
                continue;
            }

            let affected = Self::insert_or_replace_row(tx, "knowledge_nodes", table, row)?;
            report.rows_imported += 1;
            if affected == MergeWrite::Inserted {
                report.rows_inserted += 1;
            } else {
                report.rows_updated += 1;
            }
        }
        Ok(())
    }

    fn merge_sync_tombstones(
        tx: &rusqlite::Transaction<'_>,
        table: &PortableTable,
        report: &mut PortableImportReport,
    ) -> Result<()> {
        for row in &table.rows {
            let Some(table_name) = Self::portable_text(table, row, "table_name") else {
                report.rows_skipped += 1;
                continue;
            };
            let Some(row_id) = Self::portable_text(table, row, "row_id") else {
                report.rows_skipped += 1;
                continue;
            };
            let incoming_deleted_at = Self::portable_timestamp(table, row, "deleted_at");
            let existing_tombstone: Option<String> = tx
                .query_row(
                    "SELECT deleted_at FROM sync_tombstones WHERE table_name = ?1 AND row_id = ?2",
                    params![table_name, row_id],
                    |row| row.get(0),
                )
                .optional()?;
            let existing_deleted_at = existing_tombstone
                .as_ref()
                .and_then(|deleted_at| Self::parse_rfc3339_opt(deleted_at));
            let incoming_wins = match (existing_deleted_at, incoming_deleted_at) {
                (Some(existing), Some(incoming)) => incoming >= existing,
                (Some(_), None) => false,
                (None, _) => true,
            };

            let effective_deleted_at = if incoming_wins {
                let affected = Self::insert_or_replace_row(tx, "sync_tombstones", table, row)?;
                report.rows_imported += 1;
                if affected == MergeWrite::Inserted {
                    report.rows_inserted += 1;
                } else {
                    report.rows_updated += 1;
                }
                incoming_deleted_at
            } else {
                report.rows_skipped += 1;
                existing_deleted_at
            };

            if table_name == "knowledge_nodes" {
                let Some(target_id) = Self::resolve_tombstone_memory_id(tx, row_id)? else {
                    // The target may arrive in a later archive, but this merge
                    // has no raw identifier to delete. The opaque tombstone is
                    // still persisted; a future node merge consults it by
                    // deriving the same marker from the candidate's local id.
                    continue;
                };
                let local_updated: Option<String> = tx
                    .query_row(
                        "SELECT updated_at FROM knowledge_nodes WHERE id = ?1",
                        params![target_id],
                        |row| row.get(0),
                    )
                    .optional()?;
                let should_delete = match (
                    local_updated.as_deref().and_then(Self::parse_rfc3339_opt),
                    effective_deleted_at,
                ) {
                    (Some(local), Some(deleted)) => {
                        row_id.starts_with("opaque:") || deleted >= local
                    }
                    (Some(_), None) => true,
                    (None, _) => false,
                };
                if should_delete {
                    // The remote marker has already been persisted above.
                    // Reuse the local coordinator without rewriting its
                    // timestamp so merge performs the full evidence cleanup
                    // atomically with the portable import.
                    if Self::purge_node_in_transaction(
                        tx,
                        &target_id,
                        effective_deleted_at.unwrap_or_else(Utc::now),
                        false,
                    )?
                    .is_some()
                    {
                        report.rows_deleted += 1;
                    }
                }
            }
        }
        Ok(())
    }

    fn merge_deletion_tombstones(
        tx: &rusqlite::Transaction<'_>,
        table: &PortableTable,
        report: &mut PortableImportReport,
    ) -> Result<()> {
        for row in &table.rows {
            let Some(memory_id) = Self::portable_text(table, row, "memory_id") else {
                report.rows_skipped += 1;
                continue;
            };
            let incoming_deleted_at = Self::portable_timestamp(table, row, "deleted_at");
            let existing_deleted_at: Option<String> = tx
                .query_row(
                    "SELECT deleted_at FROM deletion_tombstones WHERE memory_id = ?1",
                    params![memory_id],
                    |row| row.get(0),
                )
                .optional()?;

            if let (Some(existing), Some(incoming)) = (
                existing_deleted_at
                    .as_deref()
                    .and_then(Self::parse_rfc3339_opt),
                incoming_deleted_at,
            ) && existing > incoming
            {
                report.rows_skipped += 1;
                continue;
            }

            let affected = Self::insert_or_replace_row(tx, "deletion_tombstones", table, row)?;
            report.rows_imported += 1;
            if affected == MergeWrite::Inserted {
                report.rows_inserted += 1;
            } else {
                report.rows_updated += 1;
            }
        }
        Ok(())
    }

    fn merge_keyed_table(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        table: &PortableTable,
        key_columns: &[&str],
        report: &mut PortableImportReport,
        state: &PortableMergeState,
    ) -> Result<()> {
        for row in &table.rows {
            if !Self::parent_rows_exist(tx, table_name, table, row)? {
                report.rows_skipped += 1;
                continue;
            }
            if key_columns
                .iter()
                .any(|column| Self::portable_value(table, row, column).is_none())
            {
                report.rows_skipped += 1;
                continue;
            }
            if Self::row_references_locally_newer_node(table_name, table, row, state) {
                report.conflicts_kept_local += 1;
                report.rows_skipped += 1;
                continue;
            }
            let affected = Self::insert_or_replace_row(tx, table_name, table, row)?;
            report.rows_imported += 1;
            if affected == MergeWrite::Inserted {
                report.rows_inserted += 1;
            } else {
                report.rows_updated += 1;
            }
        }
        Ok(())
    }

    fn row_references_locally_newer_node(
        table_name: &str,
        table: &PortableTable,
        row: &[PortableValue],
        state: &PortableMergeState,
    ) -> bool {
        match table_name {
            "node_embeddings" => Self::portable_text(table, row, "node_id")
                .is_some_and(|id| state.locally_newer_nodes.contains(id)),
            "fsrs_cards" | "memory_states" => Self::portable_text(table, row, "memory_id")
                .is_some_and(|id| state.locally_newer_nodes.contains(id)),
            "memory_connections" => {
                Self::portable_text(table, row, "source_id")
                    .is_some_and(|id| state.locally_newer_nodes.contains(id))
                    || Self::portable_text(table, row, "target_id")
                        .is_some_and(|id| state.locally_newer_nodes.contains(id))
            }
            _ => false,
        }
    }

    fn merge_append_only_table(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        table: &PortableTable,
        report: &mut PortableImportReport,
    ) -> Result<()> {
        for row in &table.rows {
            if !Self::parent_rows_exist(tx, table_name, table, row)? {
                report.rows_skipped += 1;
                continue;
            }

            let insert_columns: Vec<String> = table
                .columns
                .iter()
                .filter(|column| column.as_str() != "id")
                .cloned()
                .collect();
            if insert_columns.is_empty() {
                report.rows_skipped += 1;
                continue;
            }

            let values = Self::row_values_for_columns(table, row, &insert_columns)?;
            if Self::row_exists_by_values(tx, table_name, &insert_columns, &values)? {
                report.rows_skipped += 1;
                continue;
            }

            Self::insert_row_with_columns(tx, table_name, &insert_columns, values)?;
            report.rows_imported += 1;
            report.rows_inserted += 1;
        }
        Ok(())
    }

    fn parent_rows_exist(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        table: &PortableTable,
        row: &[PortableValue],
    ) -> Result<bool> {
        match table_name {
            "node_embeddings" | "memory_access_log" => Self::portable_text(table, row, "node_id")
                .map(|id| Self::node_exists(tx, id))
                .transpose()
                .map(|v| v.unwrap_or(false)),
            "fsrs_cards" | "memory_states" | "state_transitions" => {
                Self::portable_text(table, row, "memory_id")
                    .map(|id| Self::node_exists(tx, id))
                    .transpose()
                    .map(|v| v.unwrap_or(false))
            }
            "memory_connections" => {
                let source_exists = Self::portable_text(table, row, "source_id")
                    .map(|id| Self::node_exists(tx, id))
                    .transpose()?
                    .unwrap_or(false);
                let target_exists = Self::portable_text(table, row, "target_id")
                    .map(|id| Self::node_exists(tx, id))
                    .transpose()?
                    .unwrap_or(false);
                Ok(source_exists && target_exists)
            }
            "composition_members" => {
                let event_exists = Self::portable_text(table, row, "event_id")
                    .map(|id| Self::composition_event_exists(tx, id))
                    .transpose()?
                    .unwrap_or(false);
                Ok(event_exists)
            }
            "composition_outcomes" => {
                let event_exists = Self::portable_text(table, row, "event_id")
                    .map(|id| Self::composition_event_exists(tx, id))
                    .transpose()?
                    .unwrap_or(false);
                Ok(event_exists)
            }
            _ => Ok(true),
        }
    }

    fn insert_or_replace_row(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        table: &PortableTable,
        row: &[PortableValue],
    ) -> Result<MergeWrite> {
        let key_exists = Self::merge_row_exists(tx, table_name, table, row)?;
        let values = Self::row_values_for_columns(table, row, &table.columns)?;
        Self::upsert_row_with_columns(tx, table_name, &table.columns, values)?;
        Ok(if key_exists {
            MergeWrite::Updated
        } else {
            MergeWrite::Inserted
        })
    }

    fn merge_key_columns(table_name: &str) -> &'static [&'static str] {
        match table_name {
            "knowledge_nodes" | "intentions" | "insights" | "sessions" => &["id"],
            "composition_events" | "composition_outcomes" => &["id"],
            "composition_members" => &["event_id", "memory_id", "role"],
            "node_embeddings" => &["node_id"],
            "fsrs_cards" | "memory_states" | "deletion_tombstones" => &["memory_id"],
            "memory_connections" => &["source_id", "target_id"],
            "fsrs_config" => &["key"],
            "sync_tombstones" => &["table_name", "row_id"],
            _ => &[],
        }
    }

    fn upsert_row_with_columns(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        columns: &[String],
        values: Vec<Value>,
    ) -> Result<()> {
        let key_columns = Self::merge_key_columns(table_name);
        if key_columns.is_empty() {
            return Self::insert_row_with_columns(tx, table_name, columns, values);
        }

        let quoted_table = Self::quote_ident(table_name);
        let quoted_columns = columns
            .iter()
            .map(|column| Self::quote_ident(column))
            .collect::<Vec<_>>()
            .join(", ");
        let placeholders = std::iter::repeat_n("?", columns.len())
            .collect::<Vec<_>>()
            .join(", ");
        let conflict_target = key_columns
            .iter()
            .map(|column| Self::quote_ident(column))
            .collect::<Vec<_>>()
            .join(", ");
        let update_columns = columns
            .iter()
            .filter(|column| !key_columns.iter().any(|key| key == &column.as_str()))
            .map(|column| {
                let quoted = Self::quote_ident(column);
                format!("{quoted} = excluded.{quoted}")
            })
            .collect::<Vec<_>>();

        let conflict_action = if update_columns.is_empty() {
            "DO NOTHING".to_string()
        } else {
            format!("DO UPDATE SET {}", update_columns.join(", "))
        };

        let sql = format!(
            "INSERT INTO {} ({}) VALUES ({}) ON CONFLICT({}) {}",
            quoted_table, quoted_columns, placeholders, conflict_target, conflict_action
        );
        tx.execute(&sql, params_from_iter(values))?;
        Ok(())
    }

    fn insert_row_with_columns(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        columns: &[String],
        values: Vec<Value>,
    ) -> Result<()> {
        let quoted_table = Self::quote_ident(table_name);
        let quoted_columns = columns
            .iter()
            .map(|column| Self::quote_ident(column))
            .collect::<Vec<_>>()
            .join(", ");
        let placeholders = std::iter::repeat_n("?", columns.len())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "INSERT OR REPLACE INTO {} ({}) VALUES ({})",
            quoted_table, quoted_columns, placeholders
        );
        tx.execute(&sql, params_from_iter(values))?;
        Ok(())
    }

    fn merge_row_exists(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        table: &PortableTable,
        row: &[PortableValue],
    ) -> Result<bool> {
        let key_columns = Self::merge_key_columns(table_name);
        if key_columns.is_empty() {
            return Ok(false);
        }
        let mut columns = Vec::new();
        for key in key_columns {
            columns.push((*key).to_string());
        }
        let values = Self::row_values_for_columns(table, row, &columns)?;
        Self::row_exists_by_values(tx, table_name, &columns, &values)
    }

    fn row_exists_by_values(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        columns: &[String],
        values: &[Value],
    ) -> Result<bool> {
        let quoted_table = Self::quote_ident(table_name);
        let where_clause = columns
            .iter()
            .map(|column| format!("{} IS ?", Self::quote_ident(column)))
            .collect::<Vec<_>>()
            .join(" AND ");
        let sql = format!(
            "SELECT COUNT(*) FROM {} WHERE {}",
            quoted_table, where_clause
        );
        let count: i64 = tx.query_row(&sql, params_from_iter(values.iter()), |row| row.get(0))?;
        Ok(count > 0)
    }

    fn row_values_for_columns(
        table: &PortableTable,
        row: &[PortableValue],
        columns: &[String],
    ) -> Result<Vec<Value>> {
        columns
            .iter()
            .map(|column| {
                Self::portable_value(table, row, column)
                    .ok_or_else(|| {
                        StorageError::Init(format!(
                            "Portable archive row in table '{}' is missing column '{}'",
                            table.name, column
                        ))
                    })?
                    .to_sql_value()
                    .map_err(|e| StorageError::Init(format!("Invalid portable value: {}", e)))
            })
            .collect()
    }

    fn portable_value<'a>(
        table: &PortableTable,
        row: &'a [PortableValue],
        column: &str,
    ) -> Option<&'a PortableValue> {
        table
            .columns
            .iter()
            .position(|name| name == column)
            .and_then(|idx| row.get(idx))
    }

    fn portable_text<'a>(
        table: &PortableTable,
        row: &'a [PortableValue],
        column: &str,
    ) -> Option<&'a str> {
        match Self::portable_value(table, row, column) {
            Some(PortableValue::Text(value)) => Some(value.as_str()),
            _ => None,
        }
    }

    fn portable_timestamp(
        table: &PortableTable,
        row: &[PortableValue],
        column: &str,
    ) -> Option<DateTime<Utc>> {
        Self::portable_text(table, row, column).and_then(Self::parse_rfc3339_opt)
    }

    fn parse_rfc3339_opt(value: &str) -> Option<DateTime<Utc>> {
        DateTime::parse_from_rfc3339(value)
            .map(|dt| dt.with_timezone(&Utc))
            .ok()
    }

    fn tombstone_timestamp(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        row_id: &str,
    ) -> Result<Option<DateTime<Utc>>> {
        let opaque_marker = if table_name == "knowledge_nodes" {
            Some(Self::opaque_tombstone_marker(row_id))
        } else {
            None
        };
        let deleted_at: Option<String> = tx
            .query_row(
                "SELECT deleted_at FROM sync_tombstones
                 WHERE table_name = ?1 AND (row_id = ?2 OR row_id = ?3)
                 ORDER BY deleted_at DESC LIMIT 1",
                params![table_name, row_id, opaque_marker],
                |row| row.get(0),
            )
            .optional()?;
        Ok(deleted_at.as_deref().and_then(Self::parse_rfc3339_opt))
    }

    fn has_opaque_tombstone(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        row_id: &str,
    ) -> Result<bool> {
        if table_name != "knowledge_nodes" {
            return Ok(false);
        }
        let marker = Self::opaque_tombstone_marker(row_id);
        let exists: Option<i64> = tx
            .query_row(
                "SELECT 1 FROM sync_tombstones WHERE table_name = ?1 AND row_id = ?2",
                params![table_name, marker],
                |row| row.get(0),
            )
            .optional()?;
        Ok(exists.is_some())
    }

    fn current_schema_version(conn: &Connection) -> Result<u32> {
        let version: i64 = conn.query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )?;
        Ok(version as u32)
    }

    fn ensure_portable_import_target_empty(conn: &Connection) -> Result<()> {
        for table_name in PORTABLE_USER_DATA_TABLES {
            if Self::table_exists(conn, table_name)? {
                let count = Self::table_row_count(conn, table_name)?;
                if count > 0 {
                    return Err(StorageError::Init(format!(
                        "Portable import requires an empty target database; table '{}' has {} rows",
                        table_name, count
                    )));
                }
            }
        }
        Ok(())
    }

    fn table_exists(conn: &Connection, table_name: &str) -> Result<bool> {
        let exists: i64 = conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?1",
            params![table_name],
            |row| row.get(0),
        )?;
        Ok(exists > 0)
    }

    fn table_row_count(conn: &Connection, table_name: &str) -> Result<i64> {
        let sql = format!("SELECT COUNT(*) FROM {}", Self::quote_ident(table_name));
        Ok(conn.query_row(&sql, [], |row| row.get(0))?)
    }

    fn table_columns(conn: &Connection, table_name: &str) -> Result<Vec<String>> {
        let sql = format!("PRAGMA table_info({})", Self::quote_ident(table_name));
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;

        let mut columns = Vec::new();
        for row in rows {
            columns.push(row?);
        }
        Ok(columns)
    }

    fn portable_value_from_ref(value: ValueRef<'_>) -> rusqlite::Result<PortableValue> {
        Ok(match value {
            ValueRef::Null => PortableValue::Null,
            ValueRef::Integer(value) => PortableValue::Integer(value),
            ValueRef::Real(value) => PortableValue::Real(value),
            ValueRef::Text(value) => PortableValue::Text(
                std::str::from_utf8(value)
                    .map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(0, Type::Text, Box::new(e))
                    })?
                    .to_string(),
            ),
            ValueRef::Blob(value) => PortableValue::Blob(encode_hex(value)),
        })
    }

    fn quote_ident(identifier: &str) -> String {
        format!("\"{}\"", identifier.replace('"', "\"\""))
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search", test))]
    fn embedding_model_matches_active(stored_model: &str, active_model: &str) -> bool {
        // Profile-aware retrieval never uses model-family matching. This helper
        // remains solely for legacy vector-repair bookkeeping.
        stored_model == active_model
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search", test))]
    fn embedding_vector_for_active_model(
        embedding_bytes: &[u8],
        stored_model: &str,
        active_model: &str,
    ) -> Option<Vec<f32>> {
        if !Self::embedding_model_matches_active(stored_model, active_model) {
            return None;
        }
        Embedding::from_bytes(embedding_bytes).map(|embedding| embedding.vector)
    }

    // ========================================================================
    // STATE TRANSITIONS (Audit Trail)
    // ========================================================================

    /// Get state transitions for a memory
    pub fn get_state_transitions(
        &self,
        memory_id: &str,
        limit: i32,
    ) -> Result<Vec<StateTransitionRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT * FROM state_transitions WHERE memory_id = ?1 ORDER BY timestamp DESC LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![memory_id, limit], |row| {
            Ok(StateTransitionRecord {
                id: row.get("id")?,
                memory_id: row.get("memory_id")?,
                from_state: row.get("from_state")?,
                to_state: row.get("to_state")?,
                reason_type: row.get("reason_type")?,
                reason_data: row.get("reason_data").ok().flatten(),
                timestamp: DateTime::parse_from_rfc3339(&row.get::<_, String>("timestamp")?)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            })
        })?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Create a consistent backup using VACUUM INTO
    pub fn backup_to(&self, path: &std::path::Path) -> Result<()> {
        let path_str = path
            .to_str()
            .ok_or_else(|| StorageError::Init("Invalid backup path encoding".to_string()))?;
        // Validate path: reject control characters (except tab) for defense-in-depth
        if path_str.bytes().any(|b| b < 0x20 && b != b'\t') {
            return Err(StorageError::Init(
                "Backup path contains invalid characters".to_string(),
            ));
        }
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        // VACUUM INTO doesn't support parameterized queries; escape single quotes
        reader.execute_batch(&format!("VACUUM INTO '{}'", path_str.replace('\'', "''")))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(path)?.permissions();
            perms.set_mode(0o600);
            std::fs::set_permissions(path, perms)?;
        }
        Ok(())
    }

    // ========================================================================
    // v1.9.0 AUTONOMIC: Retention Target, Auto-Promote, Waking Tags, Utility
    // ========================================================================

    /// Get average retention across all memories
    pub fn get_avg_retention(&self) -> Result<f64> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let avg: f64 = reader.query_row(
            "SELECT COALESCE(AVG(retention_strength), 0.0) FROM knowledge_nodes",
            [],
            |row| row.get(0),
        )?;
        Ok(avg)
    }

    /// Get retention distribution in buckets (0-20%, 20-40%, 40-60%, 60-80%, 80-100%)
    pub fn get_retention_distribution(&self) -> Result<Vec<(String, i64)>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT
                CASE
                    WHEN retention_strength < 0.2 THEN '0-20%'
                    WHEN retention_strength < 0.4 THEN '20-40%'
                    WHEN retention_strength < 0.6 THEN '40-60%'
                    WHEN retention_strength < 0.8 THEN '60-80%'
                    ELSE '80-100%'
                END as bucket,
                COUNT(*) as count
            FROM knowledge_nodes
            GROUP BY bucket
            ORDER BY bucket",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Get retention trend (improving/declining/stable) from retention snapshots
    pub fn get_retention_trend(&self) -> Result<String> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;

        let snapshots: Vec<f64> = reader
            .prepare(
                "SELECT avg_retention FROM retention_snapshots ORDER BY snapshot_at DESC LIMIT 5",
            )?
            .query_map([], |row| row.get(0))?
            .filter_map(warn_skipped_row("get_retention_trend"))
            .collect();

        if snapshots.len() < 3 {
            return Ok("insufficient_data".to_string());
        }

        // Compare recent vs older snapshots
        let recent_avg = snapshots.iter().take(2).sum::<f64>() / 2.0;
        let older_avg = snapshots.iter().skip(2).sum::<f64>() / (snapshots.len() - 2) as f64;

        let diff = recent_avg - older_avg;
        Ok(if diff > 0.02 {
            "improving".to_string()
        } else if diff < -0.02 {
            "declining".to_string()
        } else {
            "stable".to_string()
        })
    }

    /// Save a retention snapshot (called during consolidation)
    pub fn save_retention_snapshot(
        &self,
        avg_retention: f64,
        total: i64,
        below_target: i64,
        gc_triggered: bool,
    ) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "INSERT INTO retention_snapshots (snapshot_at, avg_retention, total_memories, memories_below_target, gc_triggered)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![Utc::now().to_rfc3339(), avg_retention, total, below_target, gc_triggered],
        )?;
        Ok(())
    }

    /// Count memories below a given retention threshold
    pub fn count_memories_below_retention(&self, threshold: f64) -> Result<i64> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let count: i64 = reader.query_row(
            "SELECT COUNT(*) FROM knowledge_nodes WHERE retention_strength < ?1",
            params![threshold],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Auto-GC memories below threshold (used by retention target system)
    pub fn gc_below_retention(&self, threshold: f64, min_age_days: i64) -> Result<i64> {
        let cutoff = (Utc::now() - Duration::days(min_age_days)).to_rfc3339();

        // Explicitly protected (pinned) memories are never garbage-collected,
        // no matter how far their retention has decayed. A pin is the user
        // saying "keep this"; low retention only says "rarely retrieved", and
        // the second must never override the first. (Until v2.6.0 this query
        // had no such exemption.)
        let protected = self.protected_node_ids()?;

        // Collect IDs first for sync tombstones and vector index cleanup.
        let doomed_ids: Vec<String> = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            let mut stmt = reader.prepare(
                "SELECT id FROM knowledge_nodes WHERE retention_strength < ?1 AND created_at < ?2",
            )?;
            stmt.query_map(params![threshold, cutoff], |row| row.get(0))?
                .filter_map(warn_skipped_row("gc_below_retention"))
                .filter(|id: &String| !protected.contains(id))
                .collect()
        };

        // Do not bulk-delete here. Every deletion must traverse `purge_node`
        // so replay capsules, traces, review records, composition evidence,
        // disclosures, and vector state cannot outlive the canonical node.
        let mut deleted = 0_i64;
        for id in doomed_ids {
            if self.delete_node(&id)? {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    /// Check for auto-promote candidates: memories explicitly promoted 3+ times in 24h.
    pub fn auto_promote_frequent_access(&self) -> Result<i64> {
        let twenty_four_hours_ago = (Utc::now() - Duration::hours(24)).to_rfc3339();
        let now = Utc::now().to_rfc3339();

        // A search hit is not evidence of correctness. Only repeated explicit
        // positive feedback is eligible for this optional extra boost.
        let candidates: Vec<String> = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            let mut stmt = reader.prepare(
                "SELECT node_id, COUNT(*) as access_count
                 FROM memory_access_log
                 WHERE accessed_at >= ?1 AND access_type = 'promote'
                 GROUP BY node_id
                 HAVING access_count >= 3",
            )?;
            stmt.query_map(params![twenty_four_hours_ago], |row| row.get(0))?
                .filter_map(warn_skipped_row("auto_promote_frequent_access"))
                .collect()
        };

        if candidates.is_empty() {
            return Ok(0);
        }

        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let mut promoted = 0i64;
        for id in &candidates {
            let rows = writer.execute(
                "UPDATE knowledge_nodes SET
                    retrieval_strength = MIN(1.0, retrieval_strength + 0.10),
                    retention_strength = MIN(1.0, retention_strength + 0.05),
                    last_accessed = ?1
                WHERE id = ?2 AND retrieval_strength < 0.95",
                params![now, id],
            )?;
            if rows > 0 {
                promoted += 1;
            }
        }

        Ok(promoted)
    }

    /// Set waking tag on a memory (marks it for preferential dream replay)
    pub fn set_waking_tag(&self, memory_id: &str) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "UPDATE knowledge_nodes SET waking_tag = TRUE, waking_tag_at = ?1 WHERE id = ?2",
            params![Utc::now().to_rfc3339(), memory_id],
        )?;
        Ok(())
    }

    /// Clear waking tags (called after dream processes them)
    pub fn clear_waking_tags(&self) -> Result<i64> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let cleared = writer.execute(
            "UPDATE knowledge_nodes SET waking_tag = FALSE, waking_tag_at = NULL WHERE waking_tag = TRUE",
            [],
        )? as i64;
        Ok(cleared)
    }

    /// Get waking-tagged memories for preferential dream replay
    pub fn get_waking_tagged_memories(&self, limit: i32) -> Result<Vec<KnowledgeNode>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT * FROM knowledge_nodes WHERE waking_tag = TRUE ORDER BY waking_tag_at DESC LIMIT ?1"
        )?;
        let nodes = stmt.query_map(params![limit], Self::row_to_node)?;
        let mut result = Vec::new();
        for node in nodes {
            result.push(node?);
        }
        Ok(result)
    }

    /// Get the memory with the most connections (best center node for graph visualization)
    pub fn get_most_connected_memory(&self) -> Result<Option<String>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT id, COUNT(*) as cnt FROM (
                SELECT source_id as id FROM memory_connections
                UNION ALL
                SELECT target_id as id FROM memory_connections
            ) GROUP BY id ORDER BY cnt DESC LIMIT 1",
        )?;
        let result = stmt
            .query_row([], |row| row.get::<_, String>(0))
            .optional()?;
        Ok(result)
    }

    /// Get memories with their connection data for graph visualization
    pub fn get_memory_subgraph(
        &self,
        center_id: &str,
        depth: u32,
        max_nodes: usize,
    ) -> Result<(Vec<KnowledgeNode>, Vec<ConnectionRecord>)> {
        let mut visited_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut frontier = vec![center_id.to_string()];
        visited_ids.insert(center_id.to_string());

        // BFS to discover connected nodes up to depth
        for _ in 0..depth {
            let mut next_frontier = Vec::new();
            for id in &frontier {
                let connections = self.get_connections_for_memory(id)?;
                for conn in &connections {
                    let other_id = if conn.source_id == *id {
                        &conn.target_id
                    } else {
                        &conn.source_id
                    };
                    if visited_ids.insert(other_id.clone()) {
                        next_frontier.push(other_id.clone());
                        if visited_ids.len() >= max_nodes {
                            break;
                        }
                    }
                }
                if visited_ids.len() >= max_nodes {
                    break;
                }
            }
            frontier = next_frontier;
            if frontier.is_empty() || visited_ids.len() >= max_nodes {
                break;
            }
        }

        // Fetch nodes
        let mut nodes = Vec::new();
        for id in &visited_ids {
            if let Some(node) = self.get_node(id)? {
                nodes.push(node);
            }
        }

        // Fetch edges between visited nodes
        let all_connections = self.get_all_connections()?;
        let edges: Vec<ConnectionRecord> = all_connections
            .into_iter()
            .filter(|c| visited_ids.contains(&c.source_id) && visited_ids.contains(&c.target_id))
            .collect();

        Ok((nodes, edges))
    }

    /// Get recent state transitions across all memories (system-wide changelog)
    pub fn get_recent_state_transitions(&self, limit: i32) -> Result<Vec<StateTransitionRecord>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt =
            reader.prepare("SELECT * FROM state_transitions ORDER BY timestamp DESC LIMIT ?1")?;

        let rows = stmt.query_map(params![limit], |row| {
            Ok(StateTransitionRecord {
                id: row.get("id")?,
                memory_id: row.get("memory_id")?,
                from_state: row.get("from_state")?,
                to_state: row.get("to_state")?,
                reason_type: row.get("reason_type")?,
                reason_data: row.get("reason_data").ok().flatten(),
                timestamp: DateTime::parse_from_rfc3339(&row.get::<_, String>("timestamp")?)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            })
        })?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    // ========================================================================
    // Merge / Supersede controls (Phase 3 — v2.1.25)
    //
    // Diff-previewed, confidence-gated, reversible, self-explaining
    // combine/dedupe/supersede on a never-delete (bitemporal) store.
    // Pure scoring/plan/op types live in `advanced::merge_supersede`.
    // ========================================================================

    /// Mark a memory protected (pinned) or unprotected. A protected memory can
    /// never be auto-merged, superseded, or garbage-collected.
    pub fn set_protected(&self, id: &str, protected: bool) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let affected = writer.execute(
            "UPDATE knowledge_nodes SET protected = ?1 WHERE id = ?2",
            params![if protected { 1 } else { 0 }, id],
        )?;
        if affected == 0 {
            return Err(StorageError::NotFound(id.to_string()));
        }
        Ok(())
    }

    /// Is this memory protected (pinned)?
    pub fn is_protected(&self, id: &str) -> Result<bool> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let v: Option<i64> = reader
            .query_row(
                "SELECT protected FROM knowledge_nodes WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .optional()?;
        match v {
            Some(p) => Ok(p != 0),
            None => Err(StorageError::NotFound(id.to_string())),
        }
    }

    /// Read the per-project merge policy (two Fellegi-Sunter thresholds +
    /// auto_apply). Persisted in `fsrs_config` so it survives restarts without a
    /// new table; falls back to defaults (env-overridable) when unset.
    pub fn get_merge_policy(&self) -> Result<crate::advanced::MergePolicy> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let read_key = |key: &str| -> Option<f64> {
            reader
                .query_row(
                    "SELECT value FROM fsrs_config WHERE key = ?1",
                    params![key],
                    |row| row.get::<_, f64>(0),
                )
                .optional()
                .ok()
                .flatten()
        };
        let default = crate::advanced::MergePolicy::default();
        let env_f32 = |name: &str, fallback: f32| -> f32 {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(fallback)
        };
        let match_threshold = read_key("merge_match_threshold")
            .map(|v| v as f32)
            .unwrap_or_else(|| env_f32("VESTIGE_MERGE_MATCH_THRESHOLD", default.match_threshold));
        let possible_threshold = read_key("merge_possible_threshold")
            .map(|v| v as f32)
            .unwrap_or_else(|| {
                env_f32(
                    "VESTIGE_MERGE_POSSIBLE_THRESHOLD",
                    default.possible_threshold,
                )
            });
        let auto_apply = match read_key("merge_auto_apply") {
            Some(v) => v != 0.0,
            None => std::env::var("VESTIGE_MERGE_AUTO_APPLY")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(default.auto_apply),
        };
        Ok(crate::advanced::MergePolicy::new(
            match_threshold,
            possible_threshold,
            auto_apply,
        ))
    }

    /// Persist the per-project merge policy into `fsrs_config`.
    pub fn set_merge_policy(&self, policy: crate::advanced::MergePolicy) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let now = Utc::now().to_rfc3339();
        let put = |key: &str, value: f64| -> Result<()> {
            writer.execute(
                "INSERT OR REPLACE INTO fsrs_config (key, value, updated_at) VALUES (?1, ?2, ?3)",
                params![key, value, now],
            )?;
            Ok(())
        };
        put("merge_match_threshold", policy.match_threshold as f64)?;
        put("merge_possible_threshold", policy.possible_threshold as f64)?;
        put(
            "merge_auto_apply",
            if policy.auto_apply { 1.0 } else { 0.0 },
        )?;
        Ok(())
    }

    /// Surface likely duplicate/overlapping memory clusters with confidence
    /// scores and the signals behind each (Fellegi-Sunter classified).
    ///
    /// Only clusters whose weakest pair scores at or above the policy's
    /// `possible_threshold` are returned. Protected members are flagged so the
    /// caller never auto-merges a pin.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn merge_candidates(
        &self,
        policy: crate::advanced::MergePolicy,
        limit: usize,
        tag_filter: &[String],
    ) -> Result<Vec<crate::advanced::MergeCandidate>> {
        use crate::advanced::{MatchClass, MergeCandidate, score_pair};

        let all_embeddings = self.get_all_embeddings()?;
        if all_embeddings.is_empty() {
            return Ok(vec![]);
        }

        // Load nodes for metadata. Exclude already-superseded nodes — they are
        // historical and must not be re-offered for merge.
        let mut node_map: std::collections::HashMap<String, KnowledgeNode> =
            std::collections::HashMap::new();
        let superseded: std::collections::HashSet<String> = self.superseded_node_ids()?;
        let protected: std::collections::HashSet<String> = self.protected_node_ids()?;

        let mut offset = 0;
        loop {
            let batch = self.get_all_nodes(500, offset)?;
            let n = batch.len();
            for node in batch {
                node_map.insert(node.id.clone(), node);
            }
            if n < 500 {
                break;
            }
            offset += 500;
        }

        // Candidate embeddings, filtered by tag and excluding superseded.
        let items: Vec<(String, Vec<f32>)> = all_embeddings
            .into_iter()
            .filter(|(id, _)| !superseded.contains(id))
            .filter(|(id, _)| {
                if tag_filter.is_empty() {
                    return true;
                }
                node_map
                    .get(id)
                    .map(|n| tag_filter.iter().any(|t| n.tags.contains(t)))
                    .unwrap_or(false)
            })
            .collect();

        let n = items.len();
        if n > 2000 {
            return Err(StorageError::Init(format!(
                "Too many memories to scan ({n} with embeddings). Filter by tags to reduce scope."
            )));
        }

        // Union-find clustering over pairs above the possible threshold.
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], x: usize) -> usize {
            let mut root = x;
            while parent[root] != root {
                root = parent[root];
            }
            let mut cur = x;
            while parent[cur] != root {
                let next = parent[cur];
                parent[cur] = root;
                cur = next;
            }
            root
        }

        // Best pair score per resulting cluster member, for the explanation.
        let mut pair_score: std::collections::HashMap<
            (usize, usize),
            crate::advanced::MatchSignals,
        > = std::collections::HashMap::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = crate::cosine_similarity(&items[i].1, &items[j].1);
                let (a_node, b_node) = (node_map.get(&items[i].0), node_map.get(&items[j].0));
                let signals = score_pair(
                    sim,
                    a_node.map(|n| n.tags.as_slice()).unwrap_or(&[]),
                    b_node.map(|n| n.tags.as_slice()).unwrap_or(&[]),
                    a_node.map(|n| n.content.as_str()).unwrap_or(""),
                    b_node.map(|n| n.content.as_str()).unwrap_or(""),
                );
                if signals.combined_score >= policy.possible_threshold {
                    let ri = find(&mut parent, i);
                    let rj = find(&mut parent, j);
                    if ri != rj {
                        parent[ri] = rj;
                    }
                    pair_score.insert((i, j), signals);
                }
            }
        }

        // Group indices by root.
        let mut clusters: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for i in 0..n {
            let r = find(&mut parent, i);
            clusters.entry(r).or_default().push(i);
        }

        let mut out: Vec<MergeCandidate> = Vec::new();
        for members in clusters.into_values() {
            if members.len() < 2 {
                continue;
            }
            // Cluster confidence = weakest recorded pair (the loosest link).
            let mut min_score = 1.0f32;
            let mut best_signals: Option<crate::advanced::MatchSignals> = None;
            for a in 0..members.len() {
                for b in (a + 1)..members.len() {
                    let key = (members[a].min(members[b]), members[a].max(members[b]));
                    if let Some(sig) = pair_score.get(&key) {
                        if sig.combined_score < min_score {
                            min_score = sig.combined_score;
                        }
                        if best_signals
                            .as_ref()
                            .map(|s| sig.combined_score > s.combined_score)
                            .unwrap_or(true)
                        {
                            best_signals = Some(sig.clone());
                        }
                    }
                }
            }
            let signals = match best_signals {
                Some(s) => s,
                None => continue,
            };

            // Survivor = highest retention member.
            let mut member_ids: Vec<String> =
                members.iter().map(|&idx| items[idx].0.clone()).collect();
            member_ids.sort_by(|a, b| {
                let ra = node_map.get(a).map(|n| n.retention_strength).unwrap_or(0.0);
                let rb = node_map.get(b).map(|n| n.retention_strength).unwrap_or(0.0);
                rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
            });
            let survivor_id = member_ids[0].clone();
            let has_protected_member = member_ids.iter().any(|id| protected.contains(id));
            let previews: Vec<String> = member_ids
                .iter()
                .map(|id| {
                    node_map
                        .get(id)
                        .map(|n| preview(&n.content, 120))
                        .unwrap_or_default()
                })
                .collect();

            let classification = match policy.classify(min_score) {
                MatchClass::NonMatch => continue,
                c => c,
            };

            out.push(MergeCandidate {
                member_ids,
                previews,
                survivor_id,
                confidence: min_score,
                classification,
                signals,
                has_protected_member,
            });
        }

        out.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.truncate(limit);
        Ok(out)
    }

    /// IDs of nodes that have been bitemporally superseded (kept, but invalid).
    pub fn superseded_node_ids(&self) -> Result<std::collections::HashSet<String>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt =
            reader.prepare("SELECT id FROM knowledge_nodes WHERE superseded_by IS NOT NULL")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut set = std::collections::HashSet::new();
        for r in rows {
            set.insert(r?);
        }
        Ok(set)
    }

    /// IDs of protected (pinned) nodes.
    pub fn protected_node_ids(&self) -> Result<std::collections::HashSet<String>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare("SELECT id FROM knowledge_nodes WHERE protected = 1")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut set = std::collections::HashSet::new();
        for r in rows {
            set.insert(r?);
        }
        Ok(set)
    }

    /// Build a previewable MERGE plan (a diff) WITHOUT applying it.
    ///
    /// The survivor is the first id (or highest retention if unspecified). The
    /// plan is persisted to `merge_plans` with status `pending` and returned for
    /// inspection. Nothing about the nodes changes until `apply_plan`.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn plan_merge(
        &self,
        member_ids: &[String],
        survivor_id: Option<&str>,
        policy: crate::advanced::MergePolicy,
    ) -> Result<crate::advanced::MergePlan> {
        use crate::advanced::{
            MatchClass, PlanKind, compose_merged_content, compose_merged_tags, score_pair,
        };

        if member_ids.len() < 2 {
            return Err(StorageError::Init(
                "plan_merge needs at least 2 member ids".into(),
            ));
        }

        let mut nodes: Vec<KnowledgeNode> = Vec::new();
        for id in member_ids {
            let node = self
                .get_node(id)?
                .ok_or_else(|| StorageError::NotFound(id.clone()))?;
            nodes.push(node);
        }

        // Protected nodes can never be absorbed. They may only be the survivor.
        let survivor = match survivor_id {
            Some(s) => s.to_string(),
            None => {
                // highest retention
                nodes
                    .iter()
                    .max_by(|a, b| {
                        a.retention_strength
                            .partial_cmp(&b.retention_strength)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|n| n.id.clone())
                    .unwrap_or_else(|| member_ids[0].clone())
            }
        };
        // The survivor MUST be one of the members. A caller-supplied survivor_id
        // that isn't in member_ids (a typo/mixup through the plan_merge tool)
        // otherwise sails through and panics at the `.find(...).unwrap()` below,
        // taking down the request. Reject it with a clear error instead.
        if !nodes.iter().any(|n| n.id == survivor) {
            return Err(StorageError::Init(format!(
                "survivor_id {survivor} is not among the member_ids being merged"
            )));
        }

        for node in &nodes {
            if node.id != survivor && self.is_protected(&node.id)? {
                return Err(StorageError::Init(format!(
                    "Memory {} is protected and cannot be merged away. Unprotect it first or make it the survivor.",
                    node.id
                )));
            }
        }

        // Order: survivor first, then others.
        nodes.sort_by_key(|n| if n.id == survivor { 0 } else { 1 });

        let members: Vec<(String, String)> = nodes
            .iter()
            .map(|n| (n.id.clone(), n.content.clone()))
            .collect();
        let result_content = compose_merged_content(&members);
        let result_tags =
            compose_merged_tags(&nodes.iter().map(|n| n.tags.clone()).collect::<Vec<_>>());
        let result_source = nodes
            .iter()
            .find(|n| n.id == survivor)
            .and_then(|n| n.source.clone());
        let invalidated_ids: Vec<String> = nodes
            .iter()
            .filter(|n| n.id != survivor)
            .map(|n| n.id.clone())
            .collect();

        // Confidence = weakest pair survivor↔absorbed.
        let survivor_node = nodes.iter().find(|n| n.id == survivor).unwrap();
        let mut min_score = 1.0f32;
        let mut best_signals = score_pair(
            1.0,
            &survivor_node.tags,
            &survivor_node.tags,
            &survivor_node.content,
            &survivor_node.content,
        );
        for node in nodes.iter().filter(|n| n.id != survivor) {
            let sim = self.pair_similarity(&survivor, &node.id)?;
            let sig = score_pair(
                sim,
                &survivor_node.tags,
                &node.tags,
                &survivor_node.content,
                &node.content,
            );
            if sig.combined_score < min_score {
                min_score = sig.combined_score;
                best_signals = sig;
            }
        }
        let classification = policy.classify(min_score);

        let plan = crate::advanced::MergePlan {
            id: uuid::Uuid::new_v4().to_string(),
            kind: PlanKind::Merge,
            survivor_id: survivor.clone(),
            member_ids: member_ids.to_vec(),
            result_content,
            result_tags,
            result_source,
            invalidated_ids,
            confidence: min_score,
            classification,
            signals: best_signals,
            explanation: format!(
                "Merge {} memories into {survivor} ({}). {} memory(ies) will be bitemporally invalidated (kept for audit, marked superseded_by={survivor}).",
                member_ids.len(),
                match classification {
                    MatchClass::Match => "strong duplicate",
                    MatchClass::Possible => "possible duplicate — review advised",
                    MatchClass::NonMatch => "weak match — review strongly advised",
                },
                member_ids.len() - 1
            ),
        };

        self.persist_plan(&plan)?;
        Ok(plan)
    }

    /// Build a previewable SUPERSEDE plan: invalidate `old_id` in favour of
    /// `new_id` (bitemporal, audit-preserving) WITHOUT applying it.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn plan_supersede(
        &self,
        old_id: &str,
        new_id: &str,
        policy: crate::advanced::MergePolicy,
    ) -> Result<crate::advanced::MergePlan> {
        use crate::advanced::{PlanKind, score_pair};

        let old = self
            .get_node(old_id)?
            .ok_or_else(|| StorageError::NotFound(old_id.to_string()))?;
        let new = self
            .get_node(new_id)?
            .ok_or_else(|| StorageError::NotFound(new_id.to_string()))?;

        if self.is_protected(old_id)? {
            return Err(StorageError::Init(format!(
                "Memory {old_id} is protected and cannot be superseded. Unprotect it first."
            )));
        }

        let sim = self.pair_similarity(old_id, new_id)?;
        let signals = score_pair(sim, &old.tags, &new.tags, &old.content, &new.content);
        let classification = policy.classify(signals.combined_score);

        let plan = crate::advanced::MergePlan {
            id: uuid::Uuid::new_v4().to_string(),
            kind: PlanKind::Supersede,
            survivor_id: new_id.to_string(),
            member_ids: vec![old_id.to_string(), new_id.to_string()],
            result_content: new.content.clone(),
            result_tags: new.tags.clone(),
            result_source: new.source.clone(),
            invalidated_ids: vec![old_id.to_string()],
            confidence: signals.combined_score,
            classification,
            signals,
            explanation: format!(
                "Supersede {old_id} with {new_id}. {old_id} is kept and remains queryable for audit, but stamped valid_until=now and superseded_by={new_id} (invalidate, don't delete)."
            ),
        };

        self.persist_plan(&plan)?;
        Ok(plan)
    }

    /// Cosine similarity between two nodes' stored embeddings (0 if missing).
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn pair_similarity(&self, a: &str, b: &str) -> Result<f32> {
        let ea = self.get_node_embedding(a)?;
        let eb = self.get_node_embedding(b)?;
        match (ea, eb) {
            (Some(ea), Some(eb)) => Ok(crate::cosine_similarity(&ea, &eb)),
            _ => Ok(0.0),
        }
    }

    /// Persist a plan row (status pending). Idempotent on plan id.
    fn persist_plan(&self, plan: &crate::advanced::MergePlan) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let payload = serde_json::to_string(plan)
            .map_err(|e| StorageError::Init(format!("plan serialize failed: {e}")))?;
        let member_ids = serde_json::to_string(&plan.member_ids).unwrap_or_else(|_| "[]".into());
        writer.execute(
            "INSERT OR REPLACE INTO merge_plans
                (id, kind, status, created_at, applied_at, survivor_id, member_ids, confidence, classification, payload)
             VALUES (?1, ?2, 'pending', ?3, NULL, ?4, ?5, ?6, ?7, ?8)",
            params![
                plan.id,
                plan.kind.as_str(),
                Utc::now().to_rfc3339(),
                plan.survivor_id,
                member_ids,
                plan.confidence as f64,
                plan.classification.as_str(),
                payload,
            ],
        )?;
        Ok(())
    }

    /// Fetch a stored plan by id.
    pub fn get_plan(&self, plan_id: &str) -> Result<Option<crate::advanced::MergePlan>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let row: Option<(String, String)> = reader
            .query_row(
                "SELECT status, payload FROM merge_plans WHERE id = ?1",
                params![plan_id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?;
        match row {
            Some((_status, payload)) => {
                let plan: crate::advanced::MergePlan = serde_json::from_str(&payload)
                    .map_err(|e| StorageError::Init(format!("plan deserialize failed: {e}")))?;
                Ok(Some(plan))
            }
            None => Ok(None),
        }
    }

    /// Plan status string (pending | applied | cancelled), if the plan exists.
    pub fn plan_status(&self, plan_id: &str) -> Result<Option<String>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let status: Option<String> = reader
            .query_row(
                "SELECT status FROM merge_plans WHERE id = ?1",
                params![plan_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(status)
    }

    /// Preview an exact tag rename/merge without mutating the store.
    ///
    /// Tags are JSON arrays in SQLite, so this intentionally parses every row
    /// instead of using a substring `LIKE` query. That keeps `prixsix` distinct
    /// from `prix-six` and avoids rewriting tags that merely share a prefix.
    pub fn preview_tag_mutation(
        &self,
        source_tags: &[String],
        target_tag: &str,
        scope: Option<&str>,
    ) -> Result<serde_json::Value> {
        let (source_tags, target_tag) = Self::validate_tag_mutation(source_tags, target_tag)?;
        // Secret policy applies to the TARGET (newly persisted) only. A
        // secret-shaped SOURCE tag can legitimately already exist in the store
        // (explicit-allow ingest, pre-scanning clients); matching it adds no
        // new exposure, and rejecting it would make the credential-shaped tag
        // impossible to rename AWAY — backwards for a cleanup tool.
        Self::enforce_secret_policy_for_content(&target_tag, SecretPolicy::Reject)?;
        let scope = scope
            .map(Self::normalize_scope)
            .transpose()?
            .map(str::to_string);
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let (source_counts, target_count, affected) = Self::tag_mutation_state(
            &reader,
            &source_tags,
            &target_tag,
            scope.as_deref(),
            MAX_TAG_MUTATION_MEMORIES,
        )?;
        let preview_token =
            Self::tag_mutation_token(&source_tags, &target_tag, scope.as_deref(), &affected)?;
        let affected_ids: Vec<&String> = affected.iter().map(|(id, _, _)| id).collect();
        let affected_count = affected_ids.len();

        let preview_limit = 200usize;
        Ok(serde_json::json!({
            "sourceTags": source_tags,
            "targetTag": target_tag,
            "scope": scope.clone(),
            "allScopes": scope.is_none(),
            "sourceTagCounts": source_counts,
            "targetTagCount": target_count,
            "affectedMemoryCount": affected_count,
            "affectedMemoryIds": affected_ids.into_iter().take(preview_limit).collect::<Vec<_>>(),
            "affectedMemoryIdsTruncated": affected_count > preview_limit,
            "maximumAffectedMemoriesPerOperation": MAX_TAG_MUTATION_MEMORIES,
            "withinOperationLimit": affected_count <= MAX_TAG_MUTATION_MEMORIES,
            "previewToken": preview_token,
            "requiresConfirmation": true,
        }))
    }

    /// Atomically rename or merge exact tags and append a reversible operation
    /// to the existing memory reflog. Callers must preview and confirmation-gate
    /// this operation before invoking it.
    pub fn apply_tag_mutation(
        &self,
        source_tags: &[String],
        target_tag: &str,
        scope: Option<&str>,
        preview_token: &str,
        op_type: &str,
        reason: &str,
    ) -> Result<crate::advanced::MergeOperation> {
        self.apply_tag_mutation_with_limits(
            source_tags,
            target_tag,
            scope,
            preview_token,
            op_type,
            reason,
            MAX_TAG_MUTATION_MEMORIES,
            MAX_TAG_MUTATION_AUDIT_BYTES,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_tag_mutation_with_limits(
        &self,
        source_tags: &[String],
        target_tag: &str,
        scope: Option<&str>,
        preview_token: &str,
        op_type: &str,
        reason: &str,
        maximum_affected: usize,
        maximum_audit_bytes: usize,
    ) -> Result<crate::advanced::MergeOperation> {
        if !matches!(op_type, "tag_rename" | "tag_merge") {
            return Err(StorageError::Init(format!(
                "invalid tag mutation operation type '{op_type}'"
            )));
        }
        let (source_tags, target_tag) = Self::validate_tag_mutation(source_tags, target_tag)?;
        let scope = scope
            .map(Self::normalize_scope)
            .transpose()?
            .map(str::to_string);
        let reason = Self::validate_tag_mutation_reason(reason)?;
        // As in `preview_tag_mutation`: secret policy guards only the newly
        // persisted TARGET and reason. A secret-shaped SOURCE tag already
        // exists in the store, so matching it to remove it adds no exposure.
        Self::enforce_secret_policy_for_content(&target_tag, SecretPolicy::Reject)?;
        Self::enforce_secret_policy_for_content(&reason, SecretPolicy::Reject)?;
        let now = Utc::now();
        let operation_id = Uuid::new_v4().to_string();
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "apply_tag_mutation_with_limits")?;

        // Recompute the exact preview state while holding the write transaction.
        // A token from an older/different scope, tag set, target, or row state
        // cannot authorize a mutation after preview drift.
        let (_, _, affected) = Self::tag_mutation_state(
            &tx,
            &source_tags,
            &target_tag,
            scope.as_deref(),
            maximum_affected,
        )?;
        let current_token =
            Self::tag_mutation_token(&source_tags, &target_tag, scope.as_deref(), &affected)?;
        if preview_token != current_token {
            return Err(StorageError::Init(
                "tag preview is stale or does not match this scope/source/target; preview again"
                    .into(),
            ));
        }
        if affected.is_empty() {
            return Err(StorageError::NotFound(format!(
                "no memories contain source tag(s): {}",
                source_tags.join(", ")
            )));
        }
        let mut affected_ids = Vec::new();
        let mut previous_tags = serde_json::Map::new();
        let mut applied_tags = serde_json::Map::new();
        for (id, tags, rewritten) in &affected {
            previous_tags.insert(id.clone(), serde_json::json!(tags));
            applied_tags.insert(id.clone(), serde_json::json!(rewritten));
            affected_ids.push(id.clone());
        }

        let undo_payload = serde_json::json!({
            "kind": "tag_mutation",
            "source_tags": source_tags.clone(),
            "target_tag": target_tag.clone(),
            "scope": scope.clone(),
            "all_scopes": scope.is_none(),
            "preview_token": preview_token,
            "previous_tags": previous_tags,
            "applied_tags": applied_tags,
        });
        let undo_payload = undo_payload.to_string();
        if undo_payload.len() > maximum_audit_bytes {
            return Err(StorageError::Init(format!(
                "tag mutation audit payload exceeds the {maximum_audit_bytes}-byte limit; narrow the scope before applying"
            )));
        }

        // Size and plan validation are complete before the first write. The
        // updates and durable audit record still share this one transaction.
        for (id, _, rewritten) in &affected {
            tx.execute(
                "UPDATE knowledge_nodes SET tags = ?1, updated_at = ?2 WHERE id = ?3",
                params![
                    serde_json::to_string(&rewritten).map_err(|error| {
                        StorageError::Init(format!("tag serialization failed: {error}"))
                    })?,
                    now.to_rfc3339(),
                    id,
                ],
            )?;
        }
        // Regression guard for the single-transaction guarantee: an armed test
        // fail point errors out here, after every row UPDATE but before the
        // audit INSERT, and the transaction drop must roll back both.
        #[cfg(test)]
        if FAIL_TAG_MUTATION_BEFORE_AUDIT.with(std::cell::Cell::get) {
            return Err(StorageError::Init(
                "test fail point: injected failure between tag updates and audit insert".into(),
            ));
        }
        tx.execute(
            "INSERT INTO merge_operations
                (id, plan_id, op_type, status, created_at, reverted_at, reverts_op_id,
                 survivor_id, affected_ids, confidence, signals, reason, undo_payload)
             VALUES (?1, NULL, ?2, 'applied', ?3, NULL, NULL, NULL, ?4, NULL, ?5, ?6, ?7)",
            params![
                operation_id,
                op_type,
                now.to_rfc3339(),
                serde_json::to_string(&affected_ids).unwrap_or_else(|_| "[]".into()),
                serde_json::json!({
                    "sourceTags": source_tags,
                    "targetTag": target_tag,
                    "scope": scope.clone(),
                    "allScopes": scope.is_none(),
                    "affectedMemoryCount": affected_ids.len(),
                })
                .to_string(),
                reason,
                undo_payload,
            ],
        )?;
        tx.commit()?;
        drop(writer);

        self.read_operation(&operation_id)?
            .ok_or_else(|| StorageError::Init("tag operation vanished after insert".into()))
    }

    /// Reverse a tag rename/merge from the durable memory reflog.
    pub fn undo_tag_mutation(&self, operation_id: &str) -> Result<crate::advanced::MergeOperation> {
        let operation = self
            .read_operation(operation_id)?
            .ok_or_else(|| StorageError::NotFound(format!("operation {operation_id}")))?;
        if operation.status == "reverted" {
            return Err(StorageError::Init(format!(
                "operation {operation_id} was already reverted"
            )));
        }
        if !matches!(operation.op_type.as_str(), "tag_rename" | "tag_merge") {
            return Err(StorageError::Init(format!(
                "operation {operation_id} is not a tag rename/merge"
            )));
        }

        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "undo_tag_mutation")?;
        let payload: String = tx.query_row(
            "SELECT undo_payload FROM merge_operations WHERE id = ?1",
            params![operation_id],
            |row| row.get(0),
        )?;
        let payload: serde_json::Value = serde_json::from_str(&payload)
            .map_err(|error| StorageError::Init(format!("undo payload parse failed: {error}")))?;
        let previous_tags = payload
            .get("previous_tags")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| StorageError::Init("tag undo payload has no previous_tags".into()))?;
        let applied_tags = payload
            .get("applied_tags")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| StorageError::Init("tag undo payload has no applied_tags".into()))?;
        let now = Utc::now();

        // Refuse to erase later tag edits. Validate every post-state before
        // restoring any row; a conflict or missing memory rolls back the whole
        // transaction and leaves the original operation applied.
        for (id, expected_tags) in applied_tags {
            let current_raw: Option<String> = tx
                .query_row(
                    "SELECT tags FROM knowledge_nodes WHERE id = ?1",
                    params![id],
                    |row| row.get(0),
                )
                .optional()?;
            let current_raw = current_raw.ok_or_else(|| {
                StorageError::NotFound(format!("memory {id} required by tag undo"))
            })?;
            let current: Vec<String> = serde_json::from_str(&current_raw).map_err(|error| {
                StorageError::Init(format!("invalid current tags for memory {id}: {error}"))
            })?;
            let expected: Vec<String> =
                serde_json::from_value(expected_tags.clone()).map_err(|error| {
                    StorageError::Init(format!(
                        "invalid applied tags in undo payload for memory {id}: {error}"
                    ))
                })?;
            if current != expected {
                return Err(StorageError::Init(format!(
                    "tag undo conflict for memory {id}: tags changed after operation; no rows were restored"
                )));
            }
        }

        for (id, previous) in previous_tags {
            let tags: Vec<String> = serde_json::from_value(previous.clone()).map_err(|error| {
                StorageError::Init(format!("invalid previous tags for memory {id}: {error}"))
            })?;
            let changed = tx.execute(
                "UPDATE knowledge_nodes SET tags = ?1, updated_at = ?2 WHERE id = ?3",
                params![
                    serde_json::to_string(&tags).map_err(|error| {
                        StorageError::Init(format!("tag serialization failed: {error}"))
                    })?,
                    now.to_rfc3339(),
                    id,
                ],
            )?;
            if changed != 1 {
                return Err(StorageError::NotFound(format!(
                    "memory {id} required by tag undo"
                )));
            }
        }

        let reverted = tx.execute(
            "UPDATE merge_operations
             SET status = 'reverted', reverted_at = ?1
             WHERE id = ?2 AND status = 'applied'",
            params![now.to_rfc3339(), operation_id],
        )?;
        if reverted != 1 {
            return Err(StorageError::Init(format!(
                "operation {operation_id} could not be marked reverted"
            )));
        }

        let undo_operation_id = Uuid::new_v4().to_string();
        tx.execute(
            "INSERT INTO merge_operations
                (id, plan_id, op_type, status, created_at, reverted_at, reverts_op_id,
                 survivor_id, affected_ids, confidence, signals, reason, undo_payload)
             VALUES (?1, NULL, 'undo', 'applied', ?2, NULL, ?3, NULL, ?4, NULL, NULL, ?5, '{}')",
            params![
                undo_operation_id,
                now.to_rfc3339(),
                operation_id,
                serde_json::to_string(&operation.affected_ids).unwrap_or_else(|_| "[]".into()),
                format!("Reverted {} operation {operation_id}", operation.op_type),
            ],
        )?;
        tx.commit()?;
        drop(writer);

        self.read_operation(&undo_operation_id)?
            .ok_or_else(|| StorageError::Init("tag undo operation vanished after insert".into()))
    }

    fn validate_tag_mutation(
        source_tags: &[String],
        target_tag: &str,
    ) -> Result<(Vec<String>, String)> {
        const MAX_TAG_LENGTH: usize = 200;
        const MAX_SOURCE_TAGS: usize = 50;

        if source_tags.is_empty() || source_tags.len() > MAX_SOURCE_TAGS {
            return Err(StorageError::Init(format!(
                "source_tags must contain 1 to {MAX_SOURCE_TAGS} tags"
            )));
        }

        // Only the TARGET is newly persisted, so only it gets shape rules for
        // new values (trim + length cap). SOURCE tags are exact-match lookup
        // keys for values that already exist in the store: they stay
        // byte-exact (no trim, no length cap) so whitespace-padded or overlong
        // stored tags remain reachable by rename/merge. Empty-after-trim and
        // control characters are still rejected on both sides.
        let target_tag = {
            let tag = target_tag.trim();
            if tag.is_empty() {
                return Err(StorageError::Init("tags cannot be empty".into()));
            }
            if tag.chars().count() > MAX_TAG_LENGTH || tag.chars().any(char::is_control) {
                return Err(StorageError::Init(format!(
                    "invalid tag: expected at most {MAX_TAG_LENGTH} visible characters"
                )));
            }
            tag.to_string()
        };
        let mut unique = std::collections::BTreeSet::new();
        for source in source_tags {
            if source.trim().is_empty() {
                return Err(StorageError::Init("tags cannot be empty".into()));
            }
            if source.chars().any(char::is_control) {
                return Err(StorageError::Init(
                    "invalid source tag: control characters are not allowed".into(),
                ));
            }
            if source == &target_tag {
                return Err(StorageError::Init(
                    "source tags must differ from target_tag".into(),
                ));
            }
            unique.insert(source.clone());
        }
        Ok((unique.into_iter().collect(), target_tag))
    }

    fn validate_tag_mutation_reason(reason: &str) -> Result<String> {
        let reason = reason.trim();
        if reason.is_empty()
            || reason.chars().count() > 1_000
            || reason.chars().any(char::is_control)
        {
            return Err(StorageError::Init(
                "reason must be 1 to 1000 visible characters".into(),
            ));
        }
        Ok(reason.to_string())
    }

    fn tag_mutation_state(
        connection: &Connection,
        source_tags: &[String],
        target_tag: &str,
        scope: Option<&str>,
        maximum_affected: usize,
    ) -> Result<TagMutationState> {
        let mut source_counts: std::collections::BTreeMap<String, usize> =
            source_tags.iter().cloned().map(|tag| (tag, 0)).collect();
        let mut target_count = 0usize;
        let mut affected = Vec::new();

        let sql = if scope.is_some() {
            "SELECT id, tags FROM knowledge_nodes
             WHERE COALESCE(NULLIF(trim(scope), ''), 'user') = ?1
             ORDER BY id"
        } else {
            "SELECT id, tags FROM knowledge_nodes ORDER BY id"
        };
        let mut stmt = connection.prepare(sql)?;
        let mut rows = match scope {
            Some(scope) => stmt.query(params![scope])?,
            None => stmt.query([])?,
        };
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let raw_tags: String = row.get(1)?;
            let tags: Vec<String> = serde_json::from_str(&raw_tags).map_err(|error| {
                StorageError::Init(format!("invalid tags JSON for memory {id}: {error}"))
            })?;
            if tags.iter().any(|tag| tag == target_tag) {
                target_count += 1;
            }
            for source in source_tags {
                if tags.iter().any(|tag| tag == source)
                    && let Some(count) = source_counts.get_mut(source)
                {
                    *count += 1;
                }
            }
            let rewritten = Self::rewrite_tags(&tags, source_tags, target_tag);
            if rewritten != tags {
                affected.push((id, tags, rewritten));
                if affected.len() > maximum_affected {
                    return Err(StorageError::Init(format!(
                        "tag mutation affects more than {maximum_affected} memories; narrow the scope before previewing or applying"
                    )));
                }
            }
        }
        Ok((source_counts, target_count, affected))
    }

    fn tag_mutation_token(
        source_tags: &[String],
        target_tag: &str,
        scope: Option<&str>,
        affected: &[(String, Vec<String>, Vec<String>)],
    ) -> Result<String> {
        let state = serde_json::json!({
            "version": 1,
            "source_tags": source_tags,
            "target_tag": target_tag,
            "scope": scope,
            "all_scopes": scope.is_none(),
            "affected_count": affected.len(),
            "affected": affected.iter().map(|(id, before, _)| {
                serde_json::json!({"id": id, "tags": before})
            }).collect::<Vec<_>>(),
        });
        let encoded = serde_json::to_vec(&state)
            .map_err(|error| StorageError::Init(format!("tag preview encoding failed: {error}")))?;
        Ok(format!("tag-plan-v1:{}", blake3::hash(&encoded).to_hex()))
    }

    fn rewrite_tags(tags: &[String], source_tags: &[String], target_tag: &str) -> Vec<String> {
        let sources: std::collections::HashSet<&str> =
            source_tags.iter().map(String::as_str).collect();
        if !tags.iter().any(|tag| sources.contains(tag.as_str())) {
            return tags.to_vec();
        }
        let mut inserted_target = false;
        let mut rewritten = Vec::with_capacity(tags.len());

        for tag in tags {
            if sources.contains(tag.as_str()) || tag == target_tag {
                if !inserted_target {
                    rewritten.push(target_tag.to_string());
                    inserted_target = true;
                }
            } else {
                rewritten.push(tag.clone());
            }
        }
        rewritten
    }

    /// Execute a previously-generated plan by id. Everything it does is recorded
    /// as a reversible [`MergeOperation`] in `merge_operations`. Returns the
    /// recorded operation id.
    ///
    /// - **merge**: survivor content/tags are rewritten to the merged result;
    ///   each absorbed node is bitemporally invalidated (valid_until=now,
    ///   superseded_by=survivor) and kept queryable.
    /// - **supersede**: old node is bitemporally invalidated in favour of new.
    ///
    /// `auto_apply` must be true in the policy to apply a `Match` plan without an
    /// explicit `confirm`; non-`Match` plans always require `confirm=true`.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn apply_plan(
        &self,
        plan_id: &str,
        confirm: bool,
    ) -> Result<crate::advanced::MergeOperation> {
        use crate::advanced::{MatchClass, PlanKind};

        let plan = self
            .get_plan(plan_id)?
            .ok_or_else(|| StorageError::NotFound(format!("plan {plan_id}")))?;

        match self.plan_status(plan_id)?.as_deref() {
            Some("applied") => {
                return Err(StorageError::Init(format!(
                    "plan {plan_id} was already applied"
                )));
            }
            Some("cancelled") => {
                return Err(StorageError::Init(format!("plan {plan_id} was cancelled")));
            }
            _ => {}
        }

        // Confirmation gate: only auto-applyable Match plans may skip confirm.
        let needs_confirm = !(plan.classification == MatchClass::Match);
        if needs_confirm && !confirm {
            return Err(StorageError::Init(format!(
                "plan {plan_id} is classified '{}' (confidence {:.3}) and requires confirm=true to apply",
                plan.classification.as_str(),
                plan.confidence
            )));
        }

        let now = Utc::now();
        let op_id = uuid::Uuid::new_v4().to_string();

        // The whole apply is ONE IMMEDIATE transaction, and the undo row is
        // written FIRST inside it.
        //
        // The old shape mutated through helpers that each committed on their
        // own, and only afterwards inserted the reflog row. Any failure between
        // the survivor rewrite and that insert left the survivor's content
        // overwritten with NO undo row at all: unrecoverable. SQLITE_BUSY was
        // the realistic trigger, because several MCP server processes share one
        // database file. The plan-status check was not atomic with the
        // mutations either, so two processes could both pass it and both apply.
        //
        // Now: the status re-check, the undo row, the plan transition, the
        // survivor rewrite and the invalidations either all commit or none do,
        // and the status re-check inside the transaction closes the race.
        //
        // The embedding is deliberately NOT regenerated in here. That is model
        // inference behind a tokio runtime, and holding the write lock across
        // it is the same defect as holding a mutex across a model load. The
        // transaction marks the survivor `has_embedding = 0` instead, and the
        // regeneration runs after COMMIT. If it fails, or the process dies
        // first, the node is already flagged and the consolidation cycle's
        // `generate_missing_embeddings` rebuilds it. Stale-but-flagged is
        // recoverable; overwritten-with-no-undo-row is not.
        let mut content_changed = false;
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            let tx = Self::begin_write_transaction(&writer, "apply_plan")?;

            // Re-check status INSIDE the transaction. Two processes can both
            // pass the read above; only one can pass this.
            let status: Option<String> = tx
                .query_row(
                    "SELECT status FROM merge_plans WHERE id = ?1",
                    params![plan_id],
                    |row| row.get(0),
                )
                .optional()?;
            match status.as_deref() {
                Some("applied") => {
                    return Err(StorageError::Init(format!(
                        "plan {plan_id} was already applied"
                    )));
                }
                Some("cancelled") => {
                    return Err(StorageError::Init(format!("plan {plan_id} was cancelled")));
                }
                _ => {}
            }

            // Snapshot everything we need to undo, BEFORE mutating, and from
            // inside the same transaction so the snapshot and the mutation see
            // one consistent state.
            let mut undo = serde_json::Map::new();
            undo.insert("plan_id".into(), serde_json::json!(plan_id));
            undo.insert("kind".into(), serde_json::json!(plan.kind.as_str()));
            undo.insert("survivor_id".into(), serde_json::json!(plan.survivor_id));

            match plan.kind {
                PlanKind::Merge => {
                    let (prev_content, prev_tags_json): (String, String) = tx
                        .query_row(
                            "SELECT content, COALESCE(tags, '[]') FROM knowledge_nodes WHERE id = ?1",
                            params![plan.survivor_id],
                            |row| Ok((row.get(0)?, row.get(1)?)),
                        )
                        .optional()?
                        .ok_or_else(|| StorageError::NotFound(plan.survivor_id.clone()))?;
                    let prev_tags: Vec<String> =
                        serde_json::from_str(&prev_tags_json).unwrap_or_default();
                    undo.insert("survivor_prev_content".into(), serde_json::json!(prev_content));
                    undo.insert("survivor_prev_tags".into(), serde_json::json!(prev_tags));

                    let mut absorbed = Vec::new();
                    for id in &plan.invalidated_ids {
                        let (vu, sb) = Self::read_bitemporal_in_transaction(&tx, id)?;
                        absorbed.push(serde_json::json!({
                            "id": id,
                            "prev_valid_until": vu,
                            "prev_superseded_by": sb,
                        }));
                    }
                    undo.insert("absorbed".into(), serde_json::json!(absorbed));
                    content_changed = prev_content != plan.result_content;
                }
                PlanKind::Supersede => {
                    let old_id = &plan.member_ids[0];
                    let (vu, sb) = Self::read_bitemporal_in_transaction(&tx, old_id)?;
                    undo.insert(
                        "absorbed".into(),
                        serde_json::json!([{
                            "id": old_id,
                            "prev_valid_until": vu,
                            "prev_superseded_by": sb,
                        }]),
                    );
                }
            }

            let affected: Vec<String> = {
                let mut v = vec![plan.survivor_id.clone()];
                v.extend(plan.invalidated_ids.clone());
                v
            };
            let signals = serde_json::to_string(&plan.signals).unwrap_or_else(|_| "{}".into());

            // The undo row goes in FIRST. Nothing below it can leave a mutation
            // without its reversal, because nothing below it commits alone.
            tx.execute(
                "INSERT INTO merge_operations
                    (id, plan_id, op_type, status, created_at, reverted_at, reverts_op_id,
                     survivor_id, affected_ids, confidence, signals, reason, undo_payload)
                 VALUES (?1, ?2, ?3, 'applied', ?4, NULL, NULL, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    op_id,
                    plan_id,
                    plan.kind.as_str(),
                    now.to_rfc3339(),
                    plan.survivor_id,
                    serde_json::to_string(&affected).unwrap_or_else(|_| "[]".into()),
                    plan.confidence as f64,
                    signals,
                    plan.explanation,
                    serde_json::Value::Object(undo).to_string(),
                ],
            )?;
            tx.execute(
                "UPDATE merge_plans SET status = 'applied', applied_at = ?1 WHERE id = ?2",
                params![now.to_rfc3339(), plan_id],
            )?;

            match plan.kind {
                PlanKind::Merge => {
                    let tags_json =
                        serde_json::to_string(&plan.result_tags).unwrap_or_else(|_| "[]".into());
                    tx.execute(
                        "UPDATE knowledge_nodes SET content = ?1, tags = ?2, updated_at = ?3
                         WHERE id = ?4",
                        params![
                            plan.result_content,
                            tags_json,
                            now.to_rfc3339(),
                            plan.survivor_id
                        ],
                    )?;
                    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
                    if content_changed {
                        // Flag for rebuild before COMMIT, so a crash between
                        // here and the regeneration below is self-healing.
                        tx.execute(
                            "UPDATE knowledge_nodes SET has_embedding = 0 WHERE id = ?1",
                            params![plan.survivor_id],
                        )?;
                    }
                    for id in &plan.invalidated_ids {
                        Self::invalidate_node_in_transaction(&tx, id, &plan.survivor_id, now)?;
                    }
                }
                PlanKind::Supersede => {
                    let old_id = &plan.member_ids[0];
                    Self::invalidate_node_in_transaction(&tx, old_id, &plan.survivor_id, now)?;
                }
            }

            tx.commit()?;
        }

        // Committed. Regenerate the survivor's embedding outside the write
        // lock; `has_embedding = 0` is already persisted, so failure here is
        // recoverable by the next consolidation cycle rather than silent.
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        if content_changed {
            if let Some(index) = self.vector_index.as_ref()
                && let Ok(mut index) = index.lock()
            {
                let _ = index.remove(&plan.survivor_id);
            }
            if self.active_embedding_runtime_ready().unwrap_or(false)
                && let Err(e) =
                    self.generate_embedding_for_node(&plan.survivor_id, &plan.result_content)
            {
                tracing::warn!(
                    survivor_id = %plan.survivor_id,
                    error = %e,
                    "apply_plan committed but could not regenerate the survivor embedding; \
                     it stays has_embedding=0 for the next consolidation sweep"
                );
            }
        }

        self.read_operation(&op_id)?
            .ok_or_else(|| StorageError::Init("operation vanished after insert".into()))
    }

    /// Reverse a prior merge/supersede operation by id (the "memory reflog").
    /// Restores survivor content/tags and clears the bitemporal invalidation on
    /// every node the operation touched, then records a compensating `undo` op.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn merge_undo(&self, op_id: &str) -> Result<crate::advanced::MergeOperation> {
        let op = self
            .read_operation(op_id)?
            .ok_or_else(|| StorageError::NotFound(format!("operation {op_id}")))?;
        if matches!(op.op_type.as_str(), "tag_rename" | "tag_merge") {
            return self.undo_tag_mutation(op_id);
        }
        if op.status == "reverted" {
            return Err(StorageError::Init(format!(
                "operation {op_id} was already reverted"
            )));
        }
        if op.op_type == "undo" {
            return Err(StorageError::Init("cannot undo an undo operation".into()));
        }

        let undo: serde_json::Value = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            let payload: String = reader.query_row(
                "SELECT undo_payload FROM merge_operations WHERE id = ?1",
                params![op_id],
                |row| row.get(0),
            )?;
            serde_json::from_str(&payload)
                .map_err(|e| StorageError::Init(format!("undo payload parse failed: {e}")))?
        };

        let kind = undo.get("kind").and_then(|v| v.as_str()).unwrap_or("");
        let survivor_id = undo
            .get("survivor_id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        // Restore survivor content/tags if this was a merge.
        if kind == "merge"
            && let (Some(content), Some(tags)) = (
                undo.get("survivor_prev_content").and_then(|v| v.as_str()),
                undo.get("survivor_prev_tags").and_then(|v| v.as_array()),
            )
        {
            let tags: Vec<String> = tags
                .iter()
                .filter_map(|t| t.as_str().map(|s| s.to_string()))
                .collect();
            self.rewrite_survivor(&survivor_id, content, &tags)?;
        }

        // Clear invalidation on every absorbed node, restoring prior values.
        if let Some(absorbed) = undo.get("absorbed").and_then(|v| v.as_array()) {
            for entry in absorbed {
                let id = entry.get("id").and_then(|v| v.as_str()).unwrap_or_default();
                if id.is_empty() {
                    continue;
                }
                let prev_vu = entry.get("prev_valid_until").and_then(|v| v.as_str());
                let prev_sb = entry.get("prev_superseded_by").and_then(|v| v.as_str());
                self.restore_bitemporal(id, prev_vu, prev_sb)?;
            }
        }

        let now = Utc::now();
        let new_op_id = uuid::Uuid::new_v4().to_string();
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            // Mark original reverted.
            writer.execute(
                "UPDATE merge_operations SET status = 'reverted', reverted_at = ?1 WHERE id = ?2",
                params![now.to_rfc3339(), op_id],
            )?;
            // Re-open the plan so it could be re-applied if desired.
            if let Some(plan_id) = op.plan_id.as_deref() {
                writer.execute(
                    "UPDATE merge_plans SET status = 'pending', applied_at = NULL WHERE id = ?1",
                    params![plan_id],
                )?;
            }
            // Record compensating undo op.
            writer.execute(
                "INSERT INTO merge_operations
                    (id, plan_id, op_type, status, created_at, reverted_at, reverts_op_id,
                     survivor_id, affected_ids, confidence, signals, reason, undo_payload)
                 VALUES (?1, ?2, 'undo', 'applied', ?3, NULL, ?4, ?5, ?6, NULL, NULL, ?7, '{}')",
                params![
                    new_op_id,
                    op.plan_id,
                    now.to_rfc3339(),
                    op_id,
                    survivor_id,
                    serde_json::to_string(&op.affected_ids).unwrap_or_else(|_| "[]".into()),
                    format!("Reverted {} operation {op_id}", op.op_type),
                ],
            )?;
        }

        self.read_operation(&new_op_id)?
            .ok_or_else(|| StorageError::Init("undo operation vanished after insert".into()))
    }

    /// List recent merge/supersede operations (the reflog), newest first.
    pub fn list_merge_operations(
        &self,
        limit: usize,
    ) -> Result<Vec<crate::advanced::MergeOperation>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT id, plan_id, op_type, status, created_at, reverted_at, reverts_op_id,
                    survivor_id, affected_ids, confidence, signals, reason
             FROM merge_operations ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], Self::row_to_operation)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// List tag rename/merge audit operations directly so they cannot be
    /// hidden by a busy merge/supersede reflog. `None` is explicit all-scopes;
    /// a named scope returns operations recorded for that exact scope PLUS
    /// every all-scopes operation, because an all-scopes mutation rewrote this
    /// scope's tags too and must stay visible to an agent auditing it.
    pub fn list_tag_operations(
        &self,
        limit: usize,
        scope: Option<&str>,
    ) -> Result<Vec<crate::advanced::MergeOperation>> {
        let scope = scope.map(Self::normalize_scope).transpose()?;
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let sql = if scope.is_some() {
            "SELECT id, plan_id, op_type, status, created_at, reverted_at, reverts_op_id,
                    survivor_id, affected_ids, confidence, signals, reason
             FROM merge_operations
             WHERE op_type IN ('tag_rename', 'tag_merge')
               AND (json_extract(signals, '$.allScopes') = 1
                    OR json_extract(signals, '$.scope') = ?1)
             ORDER BY created_at DESC, id DESC LIMIT ?2"
        } else {
            "SELECT id, plan_id, op_type, status, created_at, reverted_at, reverts_op_id,
                    survivor_id, affected_ids, confidence, signals, reason
             FROM merge_operations
             WHERE op_type IN ('tag_rename', 'tag_merge')
             ORDER BY created_at DESC, id DESC LIMIT ?1"
        };
        let mut stmt = reader.prepare(sql)?;
        let rows = match scope {
            Some(scope) => stmt.query_map(params![scope, limit as i64], Self::row_to_operation)?,
            None => stmt.query_map(params![limit as i64], Self::row_to_operation)?,
        };
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(StorageError::from)
    }

    /// Read one durable merge/tag operation from the memory reflog.
    pub fn get_merge_operation(
        &self,
        operation_id: &str,
    ) -> Result<Option<crate::advanced::MergeOperation>> {
        self.read_operation(operation_id)
    }

    /// Read a single operation by id.
    fn read_operation(&self, op_id: &str) -> Result<Option<crate::advanced::MergeOperation>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let op = reader
            .query_row(
                "SELECT id, plan_id, op_type, status, created_at, reverted_at, reverts_op_id,
                        survivor_id, affected_ids, confidence, signals, reason
                 FROM merge_operations WHERE id = ?1",
                params![op_id],
                Self::row_to_operation,
            )
            .optional()?;
        Ok(op)
    }

    fn row_to_operation(row: &rusqlite::Row) -> rusqlite::Result<crate::advanced::MergeOperation> {
        let affected: String = row.get("affected_ids")?;
        let affected_ids: Vec<String> = serde_json::from_str(&affected).unwrap_or_default();
        Ok(crate::advanced::MergeOperation {
            id: row.get("id")?,
            plan_id: row.get("plan_id").ok().flatten(),
            op_type: row.get("op_type")?,
            status: row.get("status")?,
            created_at: row.get("created_at")?,
            reverted_at: row.get("reverted_at").ok().flatten(),
            reverts_op_id: row.get("reverts_op_id").ok().flatten(),
            survivor_id: row.get("survivor_id").ok().flatten(),
            affected_ids,
            confidence: row
                .get::<_, Option<f64>>("confidence")
                .ok()
                .flatten()
                .map(|v| v as f32),
            signals: row
                .get::<_, Option<String>>("signals")
                .ok()
                .flatten()
                .and_then(|value| serde_json::from_str(&value).ok()),
            reason: row.get("reason").ok().flatten(),
        })
    }

    /// Read (valid_until, superseded_by) for a node.
    /// Read a node's bitemporal columns off the reader connection. Production
    /// now snapshots inside the apply transaction instead (see
    /// [`Self::read_bitemporal_in_transaction`]); this remains as the assertion
    /// helper the merge/supersede tests read state through.
    #[cfg(test)]
    fn read_bitemporal(&self, id: &str) -> Result<(Option<String>, Option<String>)> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let res = reader
            .query_row(
                "SELECT valid_until, superseded_by FROM knowledge_nodes WHERE id = ?1",
                params![id],
                |row| {
                    Ok((
                        row.get::<_, Option<String>>(0)?,
                        row.get::<_, Option<String>>(1)?,
                    ))
                },
            )
            .optional()?;
        res.ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    /// `read_bitemporal` against an open transaction, so a snapshot and the
    /// mutation it protects observe the same database state.
    fn read_bitemporal_in_transaction(
        tx: &rusqlite::Transaction<'_>,
        id: &str,
    ) -> Result<(Option<String>, Option<String>)> {
        let res = tx
            .query_row(
                "SELECT valid_until, superseded_by FROM knowledge_nodes WHERE id = ?1",
                params![id],
                |row| {
                    Ok((
                        row.get::<_, Option<String>>(0)?,
                        row.get::<_, Option<String>>(1)?,
                    ))
                },
            )
            .optional()?;
        res.ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    /// `invalidate_node` against an open transaction. The helper that takes the
    /// writer lock itself cannot be called from inside a transaction: the lock
    /// is not reentrant, so it would deadlock.
    fn invalidate_node_in_transaction(
        tx: &rusqlite::Transaction<'_>,
        id: &str,
        superseded_by: &str,
        now: DateTime<Utc>,
    ) -> Result<()> {
        tx.execute(
            "UPDATE knowledge_nodes
             SET valid_until = ?1, superseded_by = ?2, updated_at = ?1
             WHERE id = ?3",
            params![now.to_rfc3339(), superseded_by, id],
        )?;
        Ok(())
    }

    /// Restore a node's bitemporal columns (used by undo).
    fn restore_bitemporal(
        &self,
        id: &str,
        valid_until: Option<&str>,
        superseded_by: Option<&str>,
    ) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "UPDATE knowledge_nodes
             SET valid_until = ?1, superseded_by = ?2, updated_at = ?3
             WHERE id = ?4",
            params![valid_until, superseded_by, Utc::now().to_rfc3339(), id],
        )?;
        Ok(())
    }

    /// Rewrite a survivor's content and tags (used by merge apply + undo).
    /// Content rewrite regenerates the embedding via `update_node_content`.
    fn rewrite_survivor(&self, id: &str, content: &str, tags: &[String]) -> Result<()> {
        self.update_node_content(id, content)?;
        let tags_json = serde_json::to_string(tags).unwrap_or_else(|_| "[]".into());
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "UPDATE knowledge_nodes SET tags = ?1, updated_at = ?2 WHERE id = ?3",
            params![tags_json, Utc::now().to_rfc3339(), id],
        )?;
        Ok(())
    }
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

// ============================================================================
// LOCAL MEMORY STORE TRAIT IMPL
// ============================================================================

impl SqliteMemoryStore {
    /// Convert a `KnowledgeNode` (plus optional embedding vector read separately)
    /// into a `MemoryRecord` for the trait surface.
    fn node_to_record(
        node: KnowledgeNode,
        embedding: Option<Vec<f32>>,
    ) -> crate::storage::memory_store::MemoryRecord {
        use crate::storage::memory_store::MemoryRecord;
        let id = uuid::Uuid::parse_str(&node.id).unwrap_or_else(|_| uuid::Uuid::new_v4());
        MemoryRecord {
            id,
            domains: Vec::new(),
            domain_scores: std::collections::HashMap::new(),
            content: node.content,
            node_type: node.node_type,
            tags: node.tags,
            embedding,
            created_at: node.created_at,
            updated_at: node.updated_at,
            metadata: serde_json::json!({
                "source": node.source,
                "stability": node.stability,
                "difficulty": node.difficulty,
                "reps": node.reps,
                "lapses": node.lapses,
                "retention_strength": node.retention_strength,
            }),
        }
    }

    /// Read domains and domain_scores JSON columns for a node by id.
    fn read_domain_columns(
        &self,
        id: &str,
    ) -> (Vec<String>, std::collections::HashMap<String, f64>) {
        let reader = match self.reader.lock() {
            Ok(r) => r,
            Err(_) => return (Vec::new(), std::collections::HashMap::new()),
        };
        let result = reader.query_row(
            "SELECT domains, domain_scores FROM knowledge_nodes WHERE id = ?1",
            rusqlite::params![id],
            |row| {
                let d: Option<String> = row.get(0).ok().flatten();
                let ds: Option<String> = row.get(1).ok().flatten();
                Ok((d, ds))
            },
        );
        match result {
            Ok((d, ds)) => {
                let domains: Vec<String> = d
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default();
                let domain_scores: std::collections::HashMap<String, f64> = ds
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default();
                (domains, domain_scores)
            }
            Err(_) => (Vec::new(), std::collections::HashMap::new()),
        }
    }

    /// Enforce the registered embedding model. Returns `Ok(())` if:
    /// - no vector is being written (`incoming.is_none()`) and nothing is registered
    /// - the incoming signature matches the registered signature
    ///
    /// Auto-registers on the first embedded write.
    fn enforce_model(
        &self,
        incoming: Option<&crate::storage::memory_store::ModelSignature>,
    ) -> crate::storage::memory_store::MemoryStoreResult<()> {
        use crate::storage::memory_store::{MemoryStoreError, ModelSignature};
        let Some(incoming) = incoming else {
            return Ok(());
        };
        // Try from cache first
        {
            let guard = self
                .registered_model
                .read()
                .map_err(|_| MemoryStoreError::Init("registered_model rwlock poisoned".into()))?;
            if let Some(ref reg) = *guard {
                if reg == incoming {
                    return Ok(());
                }
                return Err(MemoryStoreError::ModelMismatch {
                    registered_name: reg.name.clone(),
                    registered_dim: reg.dimension,
                    registered_hash: reg.hash.clone(),
                    actual_name: incoming.name.clone(),
                    actual_dim: incoming.dimension,
                    actual_hash: incoming.hash.clone(),
                });
            }
        }
        // Not registered yet -- auto-register
        let now = Utc::now().to_rfc3339();
        let writer = self
            .writer
            .lock()
            .map_err(|_| MemoryStoreError::Init("Writer lock poisoned".into()))?;
        // Try INSERT OR IGNORE
        writer.execute(
            "INSERT OR IGNORE INTO embedding_model (id, name, dimension, hash, created_at) VALUES (1, ?1, ?2, ?3, ?4)",
            rusqlite::params![incoming.name, incoming.dimension as i64, incoming.hash, now],
        ).map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        // Read back what was stored
        let stored: Option<ModelSignature> = writer
            .query_row(
                "SELECT name, dimension, hash FROM embedding_model WHERE id = 1",
                [],
                |row| {
                    let name: String = row.get(0)?;
                    let dim: i64 = row.get(1)?;
                    let hash: String = row.get(2)?;
                    Ok(ModelSignature {
                        name,
                        dimension: dim as usize,
                        hash,
                    })
                },
            )
            .optional()
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        drop(writer);
        if let Some(stored) = stored {
            if stored != *incoming {
                return Err(MemoryStoreError::ModelMismatch {
                    registered_name: stored.name,
                    registered_dim: stored.dimension,
                    registered_hash: stored.hash,
                    actual_name: incoming.name.clone(),
                    actual_dim: incoming.dimension,
                    actual_hash: incoming.hash.clone(),
                });
            }
            // Populate cache
            let mut guard = self
                .registered_model
                .write()
                .map_err(|_| MemoryStoreError::Init("registered_model rwlock poisoned".into()))?;
            *guard = Some(stored);
        }
        Ok(())
    }
}

impl crate::storage::memory_store::MemoryStoreSend for SqliteMemoryStore {
    async fn init(&self) -> crate::storage::memory_store::MemoryStoreResult<()> {
        // Migrations run in `new`; this is a no-op for the SQLite backend.
        Ok(())
    }

    async fn health_check(
        &self,
    ) -> crate::storage::memory_store::MemoryStoreResult<crate::storage::memory_store::HealthStatus>
    {
        use crate::storage::memory_store::HealthStatus;
        let reader = self.reader.lock().map_err(|_| {
            crate::storage::memory_store::MemoryStoreError::Init("Reader lock poisoned".into())
        })?;
        let ok: rusqlite::Result<i64> = reader.query_row("SELECT 1", [], |row| row.get(0));
        if ok.is_ok() {
            Ok(HealthStatus::Healthy)
        } else {
            Ok(HealthStatus::Degraded {
                reason: "SQLite connectivity check failed".to_string(),
            })
        }
    }

    async fn registered_model(
        &self,
    ) -> crate::storage::memory_store::MemoryStoreResult<
        Option<crate::storage::memory_store::ModelSignature>,
    > {
        use crate::storage::memory_store::MemoryStoreError;
        // Check cache first
        {
            let guard = self
                .registered_model
                .read()
                .map_err(|_| MemoryStoreError::Init("registered_model rwlock poisoned".into()))?;
            if guard.is_some() {
                return Ok(guard.clone());
            }
        }
        // Fall through to DB read
        let reader = self
            .reader
            .lock()
            .map_err(|_| MemoryStoreError::Init("Reader lock poisoned".into()))?;
        let stored: Option<crate::storage::memory_store::ModelSignature> = reader
            .query_row(
                "SELECT name, dimension, hash FROM embedding_model WHERE id = 1",
                [],
                |row| {
                    let name: String = row.get(0)?;
                    let dim: i64 = row.get(1)?;
                    let hash: String = row.get(2)?;
                    Ok(crate::storage::memory_store::ModelSignature {
                        name,
                        dimension: dim as usize,
                        hash,
                    })
                },
            )
            .optional()
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        drop(reader);
        // Populate cache if we read something
        if stored.is_some() {
            let mut guard = self
                .registered_model
                .write()
                .map_err(|_| MemoryStoreError::Init("registered_model rwlock poisoned".into()))?;
            *guard = stored.clone();
        }
        Ok(stored)
    }

    async fn register_model(
        &self,
        sig: &crate::storage::memory_store::ModelSignature,
    ) -> crate::storage::memory_store::MemoryStoreResult<()> {
        self.enforce_model(Some(sig))
    }

    async fn insert(
        &self,
        record: &crate::storage::memory_store::MemoryRecord,
    ) -> crate::storage::memory_store::MemoryStoreResult<uuid::Uuid> {
        use crate::storage::memory_store::{MemoryStoreError, ModelSignature};
        Self::enforce_secret_policy_for_record(record, SecretPolicy::Reject)
            .map_err(MemoryStoreError::from)?;
        // Enforce model registry if embedding is provided
        let mut supplied_model: Option<String> = None;
        if let Some(vec) = &record.embedding {
            // Derive a signature from metadata if present, or use a generic sentinel
            let sig: Option<ModelSignature> = record
                .metadata
                .get("model_name")
                .and_then(|v| v.as_str())
                .zip(
                    record
                        .metadata
                        .get("model_dim")
                        .and_then(|v| v.as_u64())
                        .map(|d| d as usize),
                )
                .zip(record.metadata.get("model_hash").and_then(|v| v.as_str()))
                .map(|((name, dim), hash)| ModelSignature {
                    name: name.to_string(),
                    dimension: dim,
                    hash: hash.to_string(),
                });
            if let Some(ref s) = sig {
                self.enforce_model(Some(s))?;
                if vec.len() != s.dimension {
                    return Err(MemoryStoreError::InvalidInput(format!(
                        "embedding length {} != registered dimension {}",
                        vec.len(),
                        s.dimension
                    )));
                }
                supplied_model = Some(s.name.clone());
            }
        }
        // Insert directly using the record's own id so the caller-supplied UUID is
        // preserved (unlike ingest() which always generates a fresh UUID).
        let id_str = record.id.to_string();
        let now = chrono::Utc::now();
        let tags_json = serde_json::to_string(&record.tags).unwrap_or_else(|_| "[]".to_string());
        let domains_json =
            serde_json::to_string(&record.domains).unwrap_or_else(|_| "[]".to_string());
        let scores_json =
            serde_json::to_string(&record.domain_scores).unwrap_or_else(|_| "{}".to_string());
        let source: Option<String> = record
            .metadata
            .get("source")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| MemoryStoreError::Init("Writer lock poisoned".into()))?;
            writer
                .execute(
                    "INSERT INTO knowledge_nodes (
                    id, content, node_type, created_at, updated_at, last_accessed,
                    stability, difficulty, reps, lapses, learning_state,
                    storage_strength, retrieval_strength, retention_strength,
                    sentiment_score, sentiment_magnitude, next_review, scheduled_days,
                    source, tags, has_embedding, embedding_model,
                    domains, domain_scores
                ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6,
                    1.0, 0.3, 0, 0, 'new',
                    1.0, 1.0, 1.0,
                    0.0, 0.0, ?7, 1,
                    ?8, ?9, 0, NULL,
                    ?10, ?11
                )",
                    rusqlite::params![
                        id_str,
                        record.content,
                        record.node_type,
                        record.created_at.to_rfc3339(),
                        record.updated_at.to_rfc3339(),
                        now.to_rfc3339(),
                        (now + chrono::Duration::days(1)).to_rfc3339(),
                        source,
                        tags_json,
                        domains_json,
                        scores_json,
                    ],
                )
                .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        }
        // A supplied embedding is indexed under the active profile or the
        // insert fails; it is never accepted and silently left unsearchable.
        if let Some(vector) = &record.embedding {
            #[cfg(all(feature = "embeddings", feature = "vector-search"))]
            {
                self.index_supplied_embedding(&id_str, vector, supplied_model.as_deref())?;
            }
            #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
            {
                let _ = (vector, supplied_model);
                return Err(MemoryStoreError::InvalidInput(
                    "record carries an embedding but this build cannot index vectors (embeddings/vector-search features disabled)"
                        .to_string(),
                ));
            }
        }
        Ok(record.id)
    }

    async fn get(
        &self,
        id: uuid::Uuid,
    ) -> crate::storage::memory_store::MemoryStoreResult<
        Option<crate::storage::memory_store::MemoryRecord>,
    > {
        use crate::storage::memory_store::MemoryStoreError;
        let node = self
            .get_node(&id.to_string())
            .map_err(MemoryStoreError::from)?;
        let Some(node) = node else {
            return Ok(None);
        };
        let (domains, domain_scores) = self.read_domain_columns(&id.to_string());
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        let embedding = self.get_node_embedding(&id.to_string()).ok().flatten();
        #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
        let embedding: Option<Vec<f32>> = None;
        let mut rec = Self::node_to_record(node, embedding);
        rec.domains = domains;
        rec.domain_scores = domain_scores;
        Ok(Some(rec))
    }

    async fn update(
        &self,
        record: &crate::storage::memory_store::MemoryRecord,
    ) -> crate::storage::memory_store::MemoryStoreResult<()> {
        use crate::storage::memory_store::MemoryStoreError;
        self.update_node_content(&record.id.to_string(), &record.content)
            .map_err(MemoryStoreError::from)?;
        // Update domains/domain_scores
        let domains_json =
            serde_json::to_string(&record.domains).unwrap_or_else(|_| "[]".to_string());
        let scores_json =
            serde_json::to_string(&record.domain_scores).unwrap_or_else(|_| "{}".to_string());
        let writer = self
            .writer
            .lock()
            .map_err(|_| MemoryStoreError::Init("Writer lock poisoned".into()))?;
        writer
            .execute(
                "UPDATE knowledge_nodes SET domains = ?1, domain_scores = ?2 WHERE id = ?3",
                rusqlite::params![domains_json, scores_json, record.id.to_string()],
            )
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        Ok(())
    }

    async fn delete(&self, id: uuid::Uuid) -> crate::storage::memory_store::MemoryStoreResult<()> {
        use crate::storage::memory_store::MemoryStoreError;
        self.delete_node(&id.to_string())
            .map_err(MemoryStoreError::from)?;
        Ok(())
    }

    async fn search(
        &self,
        query: &crate::storage::memory_store::SearchQuery,
    ) -> crate::storage::memory_store::MemoryStoreResult<
        Vec<crate::storage::memory_store::SearchResult>,
    > {
        use crate::storage::memory_store::{MemoryStoreError, SearchResult};
        // For Phase 1 we delegate to hybrid_search or keyword_search based on what is provided.
        let limit = if query.limit == 0 { 10 } else { query.limit };
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        {
            if let Some(ref text) = query.text {
                let results = self
                    .hybrid_search(text, limit as i32, 0.3, 0.7)
                    .map_err(MemoryStoreError::from)?;
                let out = results
                    .into_iter()
                    .map(|r| {
                        let (domains, domain_scores) = self.read_domain_columns(&r.node.id);
                        let mut rec = Self::node_to_record(r.node, None);
                        rec.domains = domains;
                        rec.domain_scores = domain_scores;
                        SearchResult {
                            score: r.combined_score as f64,
                            fts_score: r.keyword_score.map(|s| s as f64),
                            vector_score: r.semantic_score.map(|s| s as f64),
                            record: rec,
                        }
                    })
                    .collect();
                return Ok(out);
            }
        }
        #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
        {
            if let Some(ref text) = query.text {
                // Use individual-term matching so multi-word queries find documents
                // where all words appear anywhere (not necessarily as a phrase).
                let nodes = self
                    .search_terms(text, limit as i32)
                    .map_err(MemoryStoreError::from)?;
                let out = nodes
                    .into_iter()
                    .map(|node| {
                        let (domains, domain_scores) = self.read_domain_columns(&node.id);
                        let mut rec = Self::node_to_record(node, None);
                        rec.domains = domains;
                        rec.domain_scores = domain_scores;
                        SearchResult {
                            record: rec,
                            score: 1.0,
                            fts_score: Some(1.0),
                            vector_score: None,
                        }
                    })
                    .collect();
                return Ok(out);
            }
        }
        Ok(vec![])
    }

    async fn fts_search(
        &self,
        text: &str,
        limit: usize,
    ) -> crate::storage::memory_store::MemoryStoreResult<
        Vec<crate::storage::memory_store::SearchResult>,
    > {
        use crate::storage::memory_store::{MemoryStoreError, SearchResult};
        // Use individual-term matching so multi-word queries find documents
        // where all words appear anywhere (not necessarily as a phrase).
        let nodes = self
            .search_terms(text, limit as i32)
            .map_err(MemoryStoreError::from)?;
        let out = nodes
            .into_iter()
            .map(|node| {
                let (domains, domain_scores) = self.read_domain_columns(&node.id);
                let mut rec = Self::node_to_record(node, None);
                rec.domains = domains;
                rec.domain_scores = domain_scores;
                SearchResult {
                    record: rec,
                    score: 1.0,
                    fts_score: Some(1.0),
                    vector_score: None,
                }
            })
            .collect();
        Ok(out)
    }

    async fn vector_search(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> crate::storage::memory_store::MemoryStoreResult<
        Vec<crate::storage::memory_store::SearchResult>,
    > {
        use crate::storage::memory_store::{MemoryStoreError, SearchResult};
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        {
            let Some(index) = self.vector_index.as_ref() else {
                return Ok(vec![]);
            };
            let index = index
                .lock()
                .map_err(|_| MemoryStoreError::Init("Vector index lock poisoned".into()))?;
            let raw_results = index
                .search_with_threshold(embedding, limit, 0.0_f32)
                .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
            drop(index);
            let out = raw_results
                .into_iter()
                .filter_map(|(node_id, score)| {
                    let node = self.get_node(&node_id).ok().flatten()?;
                    let (domains, domain_scores) = self.read_domain_columns(&node_id);
                    let mut rec = Self::node_to_record(node, None);
                    rec.domains = domains;
                    rec.domain_scores = domain_scores;
                    Some(SearchResult {
                        record: rec,
                        score: score as f64,
                        fts_score: None,
                        vector_score: Some(score as f64),
                    })
                })
                .collect();
            Ok(out)
        }
        #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
        {
            let _ = (embedding, limit);
            Ok(vec![])
        }
    }

    async fn get_scheduling(
        &self,
        memory_id: uuid::Uuid,
    ) -> crate::storage::memory_store::MemoryStoreResult<
        Option<crate::storage::memory_store::SchedulingState>,
    > {
        use crate::storage::memory_store::{MemoryStoreError, SchedulingState};
        let node = self
            .get_node(&memory_id.to_string())
            .map_err(MemoryStoreError::from)?;
        let Some(node) = node else {
            return Ok(None);
        };
        Ok(Some(SchedulingState {
            memory_id,
            stability: node.stability,
            difficulty: node.difficulty,
            retrievability: node.retention_strength,
            last_review: Some(node.last_accessed),
            next_review: node.next_review,
            reps: node.reps as u32,
            lapses: node.lapses as u32,
        }))
    }

    async fn update_scheduling(
        &self,
        state: &crate::storage::memory_store::SchedulingState,
    ) -> crate::storage::memory_store::MemoryStoreResult<()> {
        use crate::storage::memory_store::MemoryStoreError;
        let writer = self
            .writer
            .lock()
            .map_err(|_| MemoryStoreError::Init("Writer lock poisoned".into()))?;
        let next_review_str = state.next_review.map(|dt| dt.to_rfc3339());
        let last_review_str = state.last_review.map(|dt| dt.to_rfc3339());
        writer
            .execute(
                "UPDATE knowledge_nodes SET stability=?1, difficulty=?2, retention_strength=?3,
                 last_accessed=?4, next_review=?5, reps=?6, lapses=?7
                 WHERE id=?8",
                rusqlite::params![
                    state.stability,
                    state.difficulty,
                    state.retrievability,
                    last_review_str.as_deref().unwrap_or(""),
                    next_review_str,
                    state.reps as i64,
                    state.lapses as i64,
                    state.memory_id.to_string(),
                ],
            )
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        Ok(())
    }

    async fn get_due_memories(
        &self,
        before: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> crate::storage::memory_store::MemoryStoreResult<
        Vec<(
            crate::storage::memory_store::MemoryRecord,
            crate::storage::memory_store::SchedulingState,
        )>,
    > {
        use crate::storage::memory_store::{MemoryStoreError, SchedulingState};
        let reader = self
            .reader
            .lock()
            .map_err(|_| MemoryStoreError::Init("Reader lock poisoned".into()))?;
        let before_str = before.to_rfc3339();
        let mut stmt = reader
            .prepare(
                "SELECT * FROM knowledge_nodes WHERE next_review <= ?1 ORDER BY next_review ASC LIMIT ?2",
            )
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        let nodes: Vec<KnowledgeNode> = stmt
            .query_map(
                rusqlite::params![before_str, limit as i64],
                Self::row_to_node,
            )
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        drop(stmt);
        drop(reader);
        let out = nodes
            .into_iter()
            .map(|node| {
                let id_str = node.id.clone();
                let (domains, domain_scores) = self.read_domain_columns(&id_str);
                let id_uuid =
                    uuid::Uuid::parse_str(&id_str).unwrap_or_else(|_| uuid::Uuid::new_v4());
                let state = SchedulingState {
                    memory_id: id_uuid,
                    stability: node.stability,
                    difficulty: node.difficulty,
                    retrievability: node.retention_strength,
                    last_review: Some(node.last_accessed),
                    next_review: node.next_review,
                    reps: node.reps as u32,
                    lapses: node.lapses as u32,
                };
                let mut rec = Self::node_to_record(node, None);
                rec.domains = domains;
                rec.domain_scores = domain_scores;
                (rec, state)
            })
            .collect();
        Ok(out)
    }

    async fn add_edge(
        &self,
        edge: &crate::storage::memory_store::MemoryEdge,
    ) -> crate::storage::memory_store::MemoryStoreResult<()> {
        use crate::storage::memory_store::MemoryStoreError;
        let conn = ConnectionRecord {
            source_id: edge.source_id.to_string(),
            target_id: edge.target_id.to_string(),
            strength: edge.weight,
            link_type: edge.edge_type.clone(),
            created_at: edge.created_at,
            last_activated: edge.created_at,
            activation_count: 0,
        };
        self.save_connection(&conn).map_err(MemoryStoreError::from)
    }

    async fn get_edges(
        &self,
        node_id: uuid::Uuid,
        edge_type: Option<&str>,
    ) -> crate::storage::memory_store::MemoryStoreResult<
        Vec<crate::storage::memory_store::MemoryEdge>,
    > {
        use crate::storage::memory_store::{MemoryEdge, MemoryStoreError};
        let conns = self
            .get_connections_for_memory(&node_id.to_string())
            .map_err(MemoryStoreError::from)?;
        let edges = conns
            .into_iter()
            .filter(|c| edge_type.is_none_or(|t| c.link_type == t))
            .filter_map(|c| {
                let src = uuid::Uuid::parse_str(&c.source_id).ok()?;
                let tgt = uuid::Uuid::parse_str(&c.target_id).ok()?;
                Some(MemoryEdge {
                    source_id: src,
                    target_id: tgt,
                    edge_type: c.link_type,
                    weight: c.strength,
                    created_at: c.created_at,
                })
            })
            .collect();
        Ok(edges)
    }

    async fn remove_edge(
        &self,
        source: uuid::Uuid,
        target: uuid::Uuid,
    ) -> crate::storage::memory_store::MemoryStoreResult<()> {
        use crate::storage::memory_store::MemoryStoreError;
        let writer = self
            .writer
            .lock()
            .map_err(|_| MemoryStoreError::Init("Writer lock poisoned".into()))?;
        writer
            .execute(
                "DELETE FROM memory_connections WHERE source_id = ?1 AND target_id = ?2",
                rusqlite::params![source.to_string(), target.to_string()],
            )
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        Ok(())
    }

    async fn get_neighbors(
        &self,
        node_id: uuid::Uuid,
        depth: usize,
    ) -> crate::storage::memory_store::MemoryStoreResult<
        Vec<(crate::storage::memory_store::MemoryRecord, f64)>,
    > {
        use crate::storage::memory_store::MemoryStoreError;
        // Depth 0: return just the node itself if it exists.
        if depth == 0 {
            let node = self
                .get_node(&node_id.to_string())
                .map_err(MemoryStoreError::from)?
                .ok_or_else(|| MemoryStoreError::NotFound(node_id.to_string()))?;
            let (domains, domain_scores) = self.read_domain_columns(&node_id.to_string());
            let mut rec = Self::node_to_record(node, None);
            rec.domains = domains;
            rec.domain_scores = domain_scores;
            return Ok(vec![(rec, 1.0)]);
        }
        // BFS up to `depth` levels, capped at 256 nodes.
        const MAX_NODES: usize = 256;
        let mut visited: std::collections::HashMap<uuid::Uuid, f64> =
            std::collections::HashMap::new();
        let mut frontier: Vec<(uuid::Uuid, f64)> = vec![(node_id, 1.0)];
        visited.insert(node_id, 1.0);
        for _ in 0..depth {
            if visited.len() >= MAX_NODES {
                break;
            }
            let mut next_frontier = Vec::new();
            for (current, current_weight) in frontier.iter() {
                let conns = self
                    .get_connections_for_memory(&current.to_string())
                    .unwrap_or_default();
                for conn in conns {
                    let neighbor_id_str = if conn.source_id == current.to_string() {
                        conn.target_id
                    } else {
                        conn.source_id
                    };
                    let Ok(nid) = uuid::Uuid::parse_str(&neighbor_id_str) else {
                        continue;
                    };
                    if let std::collections::hash_map::Entry::Vacant(e) = visited.entry(nid) {
                        let w = current_weight * conn.strength;
                        e.insert(w);
                        next_frontier.push((nid, w));
                        if visited.len() >= MAX_NODES {
                            break;
                        }
                    }
                }
            }
            frontier = next_frontier;
            if frontier.is_empty() {
                break;
            }
        }
        let mut result = Vec::with_capacity(visited.len());
        for (nid, weight) in visited {
            let Some(node) = self.get_node(&nid.to_string()).ok().flatten() else {
                continue;
            };
            let (domains, domain_scores) = self.read_domain_columns(&nid.to_string());
            let mut rec = Self::node_to_record(node, None);
            rec.domains = domains;
            rec.domain_scores = domain_scores;
            result.push((rec, weight));
        }
        Ok(result)
    }

    async fn list_domains(
        &self,
    ) -> crate::storage::memory_store::MemoryStoreResult<Vec<crate::storage::memory_store::Domain>>
    {
        use crate::storage::memory_store::{Domain, MemoryStoreError};
        let reader = self
            .reader
            .lock()
            .map_err(|_| MemoryStoreError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader
            .prepare("SELECT id, label, centroid, top_terms, memory_count, created_at FROM domains ORDER BY created_at ASC")
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let label: String = row.get(1)?;
                let centroid_bytes: Option<Vec<u8>> = row.get(2)?;
                let top_terms_json: String = row.get(3)?;
                let memory_count: i64 = row.get(4)?;
                let created_at_str: String = row.get(5)?;
                Ok((
                    id,
                    label,
                    centroid_bytes,
                    top_terms_json,
                    memory_count,
                    created_at_str,
                ))
            })
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        let mut result = Vec::new();
        for row in rows {
            let (id, label, centroid_bytes, top_terms_json, memory_count, created_at_str) =
                row.map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
            let centroid: Vec<f32> = centroid_bytes
                .map(|b| {
                    b.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                })
                .unwrap_or_default();
            let top_terms: Vec<String> = serde_json::from_str(&top_terms_json).unwrap_or_default();
            let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(|_| Utc::now());
            result.push(Domain {
                id,
                label,
                centroid,
                top_terms,
                memory_count: memory_count as usize,
                created_at,
            });
        }
        Ok(result)
    }

    async fn get_domain(
        &self,
        id: &str,
    ) -> crate::storage::memory_store::MemoryStoreResult<Option<crate::storage::memory_store::Domain>>
    {
        use crate::storage::memory_store::{Domain, MemoryStoreError};
        type DomainRow = (String, String, Option<Vec<u8>>, String, i64, String);
        let reader = self
            .reader
            .lock()
            .map_err(|_| MemoryStoreError::Init("Reader lock poisoned".into()))?;
        let result: Option<DomainRow> = reader
            .query_row(
                "SELECT id, label, centroid, top_terms, memory_count, created_at FROM domains WHERE id = ?1",
                rusqlite::params![id],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .optional()
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        let Some((id, label, centroid_bytes, top_terms_json, memory_count, created_at_str)) =
            result
        else {
            return Ok(None);
        };
        let centroid: Vec<f32> = centroid_bytes
            .map(|b| {
                b.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            })
            .unwrap_or_default();
        let top_terms: Vec<String> = serde_json::from_str(&top_terms_json).unwrap_or_default();
        let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| Utc::now());
        Ok(Some(Domain {
            id,
            label,
            centroid,
            top_terms,
            memory_count: memory_count as usize,
            created_at,
        }))
    }

    async fn upsert_domain(
        &self,
        domain: &crate::storage::memory_store::Domain,
    ) -> crate::storage::memory_store::MemoryStoreResult<()> {
        use crate::storage::memory_store::MemoryStoreError;
        let centroid_bytes: Vec<u8> = domain
            .centroid
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let top_terms_json =
            serde_json::to_string(&domain.top_terms).unwrap_or_else(|_| "[]".to_string());
        let writer = self
            .writer
            .lock()
            .map_err(|_| MemoryStoreError::Init("Writer lock poisoned".into()))?;
        writer
            .execute(
                "INSERT INTO domains (id, label, centroid, top_terms, memory_count, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(id) DO UPDATE SET
                   label = excluded.label,
                   centroid = excluded.centroid,
                   top_terms = excluded.top_terms,
                   memory_count = excluded.memory_count",
                rusqlite::params![
                    domain.id,
                    domain.label,
                    centroid_bytes,
                    top_terms_json,
                    domain.memory_count as i64,
                    domain.created_at.to_rfc3339(),
                ],
            )
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        Ok(())
    }

    async fn delete_domain(&self, id: &str) -> crate::storage::memory_store::MemoryStoreResult<()> {
        use crate::storage::memory_store::MemoryStoreError;
        let writer = self
            .writer
            .lock()
            .map_err(|_| MemoryStoreError::Init("Writer lock poisoned".into()))?;
        writer
            .execute("DELETE FROM domains WHERE id = ?1", rusqlite::params![id])
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        Ok(())
    }

    async fn classify(
        &self,
        _embedding: &[f32],
    ) -> crate::storage::memory_store::MemoryStoreResult<Vec<(String, f64)>> {
        // Phase 1 stub: no centroids yet. Phase 4 wires the full soft-assignment pass.
        Ok(vec![])
    }

    async fn count(&self) -> crate::storage::memory_store::MemoryStoreResult<usize> {
        use crate::storage::memory_store::MemoryStoreError;
        let reader = self
            .reader
            .lock()
            .map_err(|_| MemoryStoreError::Init("Reader lock poisoned".into()))?;
        let n: i64 = reader
            .query_row("SELECT COUNT(*) FROM knowledge_nodes", [], |row| row.get(0))
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        Ok(n as usize)
    }

    async fn get_stats(
        &self,
    ) -> crate::storage::memory_store::MemoryStoreResult<crate::storage::memory_store::StoreStats>
    {
        use crate::storage::memory_store::{MemoryStoreError, StoreStats};
        let reader = self
            .reader
            .lock()
            .map_err(|_| MemoryStoreError::Init("Reader lock poisoned".into()))?;
        let total: i64 = reader
            .query_row("SELECT COUNT(*) FROM knowledge_nodes", [], |row| row.get(0))
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        let with_emb: i64 = reader
            .query_row(
                "SELECT COUNT(*) FROM knowledge_nodes WHERE has_embedding = 1",
                [],
                |row| row.get(0),
            )
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        let total_edges: i64 = reader
            .query_row("SELECT COUNT(*) FROM memory_connections", [], |row| {
                row.get(0)
            })
            .unwrap_or(0);
        let total_domains: i64 = reader
            .query_row("SELECT COUNT(*) FROM domains", [], |row| row.get(0))
            .unwrap_or(0);
        let model_row: Option<(String, i64)> = reader
            .query_row(
                "SELECT name, dimension FROM embedding_model WHERE id = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        let (model_name, model_dim) = model_row
            .map(|(n, d)| (Some(n), Some(d as usize)))
            .unwrap_or((None, None));
        Ok(StoreStats {
            total_memories: total as usize,
            memories_with_embeddings: with_emb as usize,
            total_edges: total_edges as usize,
            total_domains: total_domains as usize,
            registered_model_name: model_name,
            registered_model_dim: model_dim,
        })
    }

    async fn vacuum(&self) -> crate::storage::memory_store::MemoryStoreResult<()> {
        use crate::storage::memory_store::MemoryStoreError;
        let writer = self
            .writer
            .lock()
            .map_err(|_| MemoryStoreError::Init("Writer lock poisoned".into()))?;
        writer
            .execute_batch("VACUUM;")
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
        Ok(())
    }
}

// ============================================================================
// CONNECTOR SYNC (#57) — idempotent external-source ingestion
// ============================================================================

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

impl SqliteMemoryStore {
    /// Idempotently upsert one external-source record, keyed on the envelope's
    /// `(source_system, source_id)` (#57).
    ///
    /// This is the core primitive every connector calls per record. It makes
    /// re-running a sync safe and cheap:
    ///
    /// - **No existing memory** for the key → insert (`Created`).
    /// - **Existing memory, `content_hash` changed** → update content + envelope,
    ///   stamp `updated_at`, regenerate the embedding (`Updated`).
    /// - **Existing memory, `content_hash` unchanged** → touch only `synced_at`
    ///   so the reconcile pass knows the record is still live (`Unchanged`).
    ///
    /// The caller MUST set `source_system`, `source_id`, and `content_hash` on
    /// the input's `source_envelope`; otherwise this falls back to a plain
    /// `ingest` (an un-keyed record can't be deduplicated).
    pub fn upsert_by_source(&self, input: IngestInput) -> Result<SourceUpsertResult> {
        self.upsert_by_source_with_secret_policy(input, SecretPolicy::Reject)
    }

    /// Upsert source content using an explicit credential-storage policy.
    /// Connectors must retain the default reject policy; this escape hatch is
    /// reserved for an explicit, trusted local import.
    pub fn upsert_by_source_with_secret_policy(
        &self,
        input: IngestInput,
        policy: SecretPolicy,
    ) -> Result<SourceUpsertResult> {
        Self::enforce_secret_policy_for_input(&input, policy)?;
        let env = match input.source_envelope.clone() {
            Some(e) if e.has_key() => e,
            // No idempotency key — behave like a normal create.
            _ => {
                let node = self.ingest_with_secret_policy(input, policy)?;
                return Ok(SourceUpsertResult {
                    outcome: SourceUpsertOutcome::Created,
                    node_id: node.id,
                });
            }
        };

        let source_system = env.source_system.clone().unwrap_or_default();
        let source_id = env.source_id.clone().unwrap_or_default();
        // Scope the idempotency key by source_project too: two sources of the
        // same system (e.g. github repos octocat/repoA and octocat/repoB, or two
        // Redmine instances) reuse bare per-project ids ("5"), so keying on
        // (source_system, source_id) alone made repoB's issue #5 overwrite
        // repoA's row in place. The lookup MUST use the exact same
        // COALESCE(source_project, '') semantics as the V19 unique index, which
        // buckets NULL and '' together: a plain `IS ?3` lookup missed a legacy
        // NULL-project row when the envelope carried Some(""), so the fall-through
        // INSERT then hit the UNIQUE constraint on that very bucket.
        let source_project = env.source_project.clone();
        let now = Utc::now();

        // Look up the existing memory for this external record, if any.
        let existing: Option<(String, Option<String>)> = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            reader
                .query_row(
                    "SELECT id, content_hash FROM knowledge_nodes \
                     WHERE source_system = ?1 AND source_id = ?2 \
                       AND COALESCE(source_project, '') = COALESCE(?3, '') LIMIT 1",
                    params![source_system, source_id, source_project],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
                )
                .optional()?
        };

        let Some((node_id, stored_hash)) = existing else {
            // First time we've seen this record — plain insert carries the
            // envelope through the existing ingest path.
            let node = self.ingest_with_secret_policy(input, policy)?;
            return Ok(SourceUpsertResult {
                outcome: SourceUpsertOutcome::Created,
                node_id: node.id,
            });
        };

        let new_hash = env.content_hash.clone();
        let unchanged = match (&stored_hash, &new_hash) {
            // Both present and equal → genuinely unchanged.
            (Some(a), Some(b)) => a == b,
            // Either side missing a hash → be conservative and treat as changed
            // so we never silently skip a real update.
            _ => false,
        };

        let env_source_updated_at = env.source_updated_at.map(|dt| dt.to_rfc3339());
        let synced_at = now.to_rfc3339();

        if unchanged {
            // Cheapest path: only advance liveness + the source cursor field.
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            writer.execute(
                // Un-tombstone fully: a reappearing record clears BOTH bitemporal
                // markers (valid_until AND superseded_by), otherwise it would be
                // resurrected as currently-valid yet still flagged as superseded,
                // which permanently excludes it from merge/consolidation.
                "UPDATE knowledge_nodes \
                 SET synced_at = ?1, source_updated_at = COALESCE(?2, source_updated_at), \
                     source_url = COALESCE(?3, source_url), \
                     valid_until = NULL, superseded_by = NULL \
                 WHERE id = ?4",
                params![synced_at, env_source_updated_at, env.source_url, node_id],
            )?;
            return Ok(SourceUpsertResult {
                outcome: SourceUpsertOutcome::Unchanged,
                node_id,
            });
        }

        // Content changed upstream → update body + full envelope, clear any
        // prior tombstone (`valid_until`), then regenerate the embedding.
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            writer.execute(
                // Clear BOTH bitemporal markers on update (see Unchanged branch).
                "UPDATE knowledge_nodes SET \
                    content = ?1, updated_at = ?2, synced_at = ?3, \
                    content_hash = ?4, source_url = ?5, source_updated_at = ?6, \
                    source_project = ?7, source_type = ?8, source_author = ?9, \
                    valid_until = NULL, superseded_by = NULL \
                 WHERE id = ?10",
                params![
                    input.content,
                    now.to_rfc3339(),
                    synced_at,
                    env.content_hash,
                    env.source_url,
                    env_source_updated_at,
                    env.source_project,
                    env.source_type,
                    env.source_author,
                    node_id,
                ],
            )?;
        }

        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        {
            if let Some(index) = self.vector_index.as_ref()
                && let Ok(mut index) = index.lock()
            {
                let _ = index.remove(&node_id);
            }
            if let Err(e) = self.generate_embedding_for_node(&node_id, &input.content) {
                tracing::warn!("Failed to regenerate embedding for {}: {}", node_id, e);
            }
        }

        Ok(SourceUpsertResult {
            outcome: SourceUpsertOutcome::Updated,
            node_id,
        })
    }

    /// Read the incremental-sync checkpoint for a `(source_system, scope)`.
    /// Returns a zeroed cursor (no high-water mark) if none has been saved yet.
    pub fn get_connector_cursor(
        &self,
        source_system: &str,
        scope: &str,
    ) -> Result<ConnectorCursor> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let row = reader
            .query_row(
                "SELECT cursor_updated_at, last_synced_at, last_full_reconcile_at, records_seen \
                 FROM connector_cursors WHERE source_system = ?1 AND scope = ?2",
                params![source_system, scope],
                |row| {
                    Ok((
                        row.get::<_, Option<String>>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, Option<String>>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                },
            )
            .optional()?;

        let parse = |s: Option<String>| -> Option<DateTime<Utc>> {
            s.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .map(|dt| dt.with_timezone(&Utc))
                    .ok()
            })
        };

        Ok(match row {
            Some((cur, last, recon, seen)) => ConnectorCursor {
                source_system: source_system.to_string(),
                scope: scope.to_string(),
                cursor_updated_at: parse(cur),
                last_synced_at: parse(last),
                last_full_reconcile_at: parse(recon),
                records_seen: seen,
            },
            None => ConnectorCursor {
                source_system: source_system.to_string(),
                scope: scope.to_string(),
                ..Default::default()
            },
        })
    }

    /// Persist the incremental-sync checkpoint for a `(source_system, scope)`.
    pub fn save_connector_cursor(&self, cursor: &ConnectorCursor) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "INSERT INTO connector_cursors \
                (source_system, scope, cursor_updated_at, last_synced_at, \
                 last_full_reconcile_at, records_seen) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
             ON CONFLICT(source_system, scope) DO UPDATE SET \
                cursor_updated_at = excluded.cursor_updated_at, \
                last_synced_at = excluded.last_synced_at, \
                last_full_reconcile_at = excluded.last_full_reconcile_at, \
                records_seen = excluded.records_seen",
            params![
                cursor.source_system,
                cursor.scope,
                cursor.cursor_updated_at.map(|d| d.to_rfc3339()),
                cursor.last_synced_at.map(|d| d.to_rfc3339()),
                cursor.last_full_reconcile_at.map(|d| d.to_rfc3339()),
                cursor.records_seen,
            ],
        )?;
        Ok(())
    }

    /// Reconcile deletions for a scope: tombstone every local memory in
    /// `(source_system, source_project = scope)` whose `source_id` is NOT in the
    /// caller-supplied set of currently-live ids (#57).
    ///
    /// Neither Redmine nor GitHub exposes a deletion feed, so an incremental
    /// `updated_at` sync can never see a delete. The connector therefore
    /// periodically enumerates the full set of live ids and calls this. We
    /// **invalidate, don't purge** (Graphiti-style): the memory keeps its
    /// content for audit but gets `valid_until = now`, so it falls out of
    /// "currently valid" retrieval without losing history. A record that
    /// reappears upstream is un-tombstoned by the next `upsert_by_source`
    /// (which clears `valid_until`).
    pub fn reconcile_source_tombstones(
        &self,
        source_system: &str,
        scope: &str,
        live_ids: &[String],
    ) -> Result<ReconcileReport> {
        let live: std::collections::HashSet<&str> = live_ids.iter().map(|s| s.as_str()).collect();

        // All currently-valid local records for this scope.
        let local: Vec<(String, String)> = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            let mut stmt = reader.prepare(
                "SELECT id, source_id FROM knowledge_nodes \
                 WHERE source_system = ?1 AND source_project = ?2 \
                   AND source_id IS NOT NULL AND valid_until IS NULL",
            )?;
            let rows = stmt.query_map(params![source_system, scope], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            rows.filter_map(warn_skipped_row("reconcile_source_tombstones")).collect()
        };

        let considered = local.len();
        let now = Utc::now().to_rfc3339();
        let mut tombstoned = Vec::new();

        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            for (node_id, source_id) in &local {
                if !live.contains(source_id.as_str()) {
                    writer.execute(
                        "UPDATE knowledge_nodes SET valid_until = ?1 WHERE id = ?2",
                        params![now, node_id],
                    )?;
                    tombstoned.push(node_id.clone());
                }
            }
        }

        Ok(ReconcileReport {
            tombstoned,
            considered,
        })
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advanced::{MatchClass, MergePolicy};
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
    #[cfg(unix)]
    use std::process::{Command, Stdio};
    #[cfg(unix)]
    use std::sync::mpsc;
    use tempfile::tempdir;
    // The public struct was renamed from Storage to SqliteMemoryStore; this
    // alias keeps all existing tests compiling without modification.
    use SqliteMemoryStore as Storage;

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn create_test_storage() -> Storage {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        Storage::new(Some(db_path)).unwrap()
    }

    fn create_test_storage_at(dir: &tempfile::TempDir, name: &str) -> Storage {
        Storage::new(Some(dir.path().join(name))).unwrap()
    }

    // ===================== SQLite durability/recovery ===================

    #[test]
    fn durability_profile_parser_is_explicit_and_fail_closed() {
        assert_eq!(
            SqliteDurabilityProfile::parse("hardened").unwrap(),
            SqliteDurabilityProfile::Hardened
        );
        assert_eq!(
            SqliteDurabilityProfile::parse(" BALANCED ").unwrap(),
            SqliteDurabilityProfile::Balanced
        );
        let error = SqliteDurabilityProfile::parse("normal").unwrap_err();
        assert!(
            error.to_string().contains("expected hardened|balanced"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn hardened_profile_is_verified_before_store_is_returned() {
        let dir = tempdir().unwrap();
        let store = Storage::new_with_durability(
            Some(dir.path().join("hardened.db")),
            SqliteDurabilityProfile::Hardened,
        )
        .unwrap();
        let status = store.durability_status();

        assert_eq!(status.profile, SqliteDurabilityProfile::Hardened);
        assert_eq!(status.writer.journal_mode, "wal");
        assert_eq!(status.writer.synchronous, 2);
        assert_eq!(status.writer.synchronous_label, "full");
        assert!(status.writer.fullfsync_enabled);
        assert!(status.writer.checkpoint_fullfsync_enabled);
        assert_eq!(status.writer.wal_autocheckpoint_pages, 1000);
        assert!(status.writer.foreign_keys_enabled);
        assert_eq!(status.writer.busy_timeout_ms, 5000);
        assert_eq!(status.reader.journal_mode, "wal");
        assert_eq!(status.reader.synchronous, 2);
        assert_eq!(status.before_migrations.quick_check, "ok");
        assert!(!status.before_migrations.synaptic_checks_applied);
        assert_eq!(status.after_migrations.quick_check, "ok");
        assert!(status.after_migrations.synaptic_checks_applied);
        assert_eq!(status.after_migrations.synaptic_consistency_violations, 0);
        assert_eq!(store.verify_integrity().unwrap().quick_check, "ok");
    }

    #[test]
    fn balanced_profile_preserves_normal_sync_only_when_explicit() {
        let dir = tempdir().unwrap();
        let store = Storage::new_with_durability(
            Some(dir.path().join("balanced.db")),
            SqliteDurabilityProfile::Balanced,
        )
        .unwrap();
        let status = store.durability_status();

        assert_eq!(status.profile, SqliteDurabilityProfile::Balanced);
        assert_eq!(status.writer.journal_mode, "wal");
        assert_eq!(status.writer.synchronous, 1);
        assert_eq!(status.writer.synchronous_label, "normal");
        assert!(!status.writer.fullfsync_enabled);
        assert!(!status.writer.checkpoint_fullfsync_enabled);
        assert_eq!(status.reader.synchronous, 1);
    }

    #[test]
    fn explicit_checkpoint_reports_sqlite_counters() {
        let dir = tempdir().unwrap();
        let store = Storage::new_with_durability(
            Some(dir.path().join("checkpoint.db")),
            SqliteDurabilityProfile::Hardened,
        )
        .unwrap();
        store
            .ingest(IngestInput {
                content: "checkpoint one acknowledged write".into(),
                node_type: "fact".into(),
                ..Default::default()
            })
            .unwrap();

        let passive = store.checkpoint_wal(WalCheckpointMode::Passive).unwrap();
        assert_eq!(passive.busy, 0);
        assert!(passive.log_frames >= passive.checkpointed_frames);

        let truncate = store.checkpoint_wal(WalCheckpointMode::Truncate).unwrap();
        assert_eq!(truncate.busy, 0);
    }

    #[test]
    fn backup_to_captures_committed_wal_frames_in_a_consistent_snapshot() {
        let dir = tempdir().unwrap();
        let source_path = dir.path().join("source.db");
        let backup_path = dir.path().join("snapshot.db");
        let store = Storage::new_with_durability(
            Some(source_path.clone()),
            SqliteDurabilityProfile::Hardened,
        )
        .unwrap();
        let node = store
            .ingest(IngestInput {
                content: "backup WAL snapshot sentinel".into(),
                node_type: "fact".into(),
                ..Default::default()
            })
            .unwrap();

        let wal_path = PathBuf::from(format!("{}-wal", source_path.display()));
        assert!(
            std::fs::metadata(&wal_path)
                .map(|metadata| metadata.len() > 0)
                .unwrap_or(false),
            "the source must retain committed WAL frames for this regression"
        );

        store.backup_to(&backup_path).unwrap();
        let backup = Connection::open(&backup_path).unwrap();
        let copied: String = backup
            .query_row(
                "SELECT content FROM knowledge_nodes WHERE id = ?1",
                params![node.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(copied, "backup WAL snapshot sentinel");
    }

    #[test]
    fn startup_rejects_corrupt_database_before_migrations() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("corrupt.db");
        std::fs::write(&path, b"not a sqlite database").unwrap();

        let error = Storage::new_with_durability(Some(path), SqliteDurabilityProfile::Hardened)
            .err()
            .expect("corrupt database must not produce a store");
        assert!(
            error.to_string().contains("file is not a database")
                || error.to_string().contains("malformed"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn startup_rejects_v21_event_without_receipt() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("inconsistent-v21.db");
        {
            let store =
                Storage::new_with_durability(Some(path.clone()), SqliteDurabilityProfile::Hardened)
                    .unwrap();
            store.checkpoint_wal(WalCheckpointMode::Truncate).unwrap();
        }
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute(
                "INSERT INTO synaptic_events
                     (event_id, trigger_memory_id, event_type, occurred_at_ms,
                      window_from_ms, window_to_ms, strength, algorithm_version,
                      receipt_id, recorded_at)
                 VALUES ('broken-event', 'missing-trigger', 'test', 1, 1, 1,
                         1.0, 'test', 'missing-receipt', '1970-01-01T00:00:00Z')",
                [],
            )
            .unwrap();
        }

        let error = Storage::new_with_durability(Some(path), SqliteDurabilityProfile::Hardened)
            .err()
            .expect("inconsistent V21 rows must fail startup");
        assert!(
            error
                .to_string()
                .contains("pre-migration synaptic receipt consistency"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn v22_pair_receipt_bindings_are_version_aware() {
        let dir = tempdir().unwrap();
        let store = Storage::new_with_durability(
            Some(dir.path().join("v22-pair-binding.db")),
            SqliteDurabilityProfile::Hardened,
        )
        .unwrap();
        let memory = store
            .ingest(IngestInput {
                content: "V22 pair binding fixture".into(),
                node_type: "fact".into(),
                ..Default::default()
            })
            .unwrap();
        let root_payload = serde_json::json!({
            "evidence": {
                "kind": "synaptic_capture",
                "predicate": {
                    "schemaVersion": 2,
                    "algorithmVersion": "vestige.synaptic_capture.v2",
                    "receiptRole": "root",
                    "trigger": { "eventId": "public-event" },
                    "candidates": []
                }
            }
        })
        .to_string();
        let child_payload = serde_json::json!({
            "evidence": {
                "kind": "synaptic_capture",
                "predicate": {
                    "schemaVersion": 2,
                    "algorithmVersion": "vestige.synaptic_capture.v2",
                    "receiptRole": "pair",
                    "parentReceiptId": "root-receipt",
                    "evaluationDirection": "forward",
                    "trigger": { "eventId": "public-event" },
                    "candidates": [{ "evidenceSlot": "candidate_1" }]
                }
            }
        })
        .to_string();
        {
            let writer = store.writer.lock().unwrap();
            writer
                .execute(
                    "INSERT INTO memory_receipts(receipt_id, payload, created_at)
                     VALUES ('root-receipt', ?1, '1970-01-01T00:00:00Z')",
                    params![root_payload],
                )
                .unwrap();
            writer
                .execute(
                    "INSERT INTO memory_receipts(receipt_id, payload, created_at)
                     VALUES ('child-receipt', ?1, '1970-01-01T00:00:00Z')",
                    params![child_payload],
                )
                .unwrap();
            writer
                .execute(
                    "INSERT INTO synaptic_events(
                         event_id, trigger_memory_id, event_type, occurred_at_ms,
                         window_from_ms, window_to_ms, strength, algorithm_version,
                         receipt_id, recorded_at, public_event_id, event_state
                     ) VALUES (
                         'private-event', ?1, 'test', 2, 1, 2, 1.0,
                         'vestige.synaptic_capture.v2', 'root-receipt',
                         '1970-01-01T00:00:00Z', 'public-event', 'closed'
                     )",
                    params![memory.id],
                )
                .unwrap();
            writer
                .execute(
                    "INSERT INTO synaptic_tags(
                         tag_id, memory_id, created_at_ms, initial_strength,
                         algorithm_version, state, recorded_at
                     ) VALUES (
                         'tag-1', ?1, 1, 1.0, 'vestige.synaptic_capture.v2',
                         'active', '1970-01-01T00:00:00Z'
                     )",
                    params![memory.id],
                )
                .unwrap();
            writer
                .execute(
                    "INSERT INTO synaptic_capture_items(
                         event_id, tag_id, memory_id, evidence_slot, receipt_id,
                         encoded_at_ms, temporal_distance_hours, capture_probability,
                         tag_strength_at_evaluation, capture_score, disposition,
                         recorded_at, evaluation_direction, algorithm_version
                     ) VALUES (
                         'private-event', 'tag-1', ?1, 'candidate_1', 'child-receipt',
                         1, 0.0, 1.0, 1.0, 1.0, 'below_threshold',
                         '1970-01-01T00:00:00Z', 'forward',
                         'vestige.synaptic_capture.v2'
                     )",
                    params![memory.id],
                )
                .unwrap();
        }

        assert_eq!(
            store
                .verify_integrity()
                .unwrap()
                .synaptic_consistency_violations,
            0
        );

        let invalid_child_payload = serde_json::json!({
            "evidence": {
                "kind": "synaptic_capture",
                "predicate": {
                    "schemaVersion": 2,
                    "algorithmVersion": "vestige.synaptic_capture.v2",
                    "receiptRole": "pair",
                    "parentReceiptId": "wrong-root",
                    "evaluationDirection": "forward",
                    "trigger": { "eventId": "public-event" },
                    "candidates": [{ "evidenceSlot": "candidate_1" }]
                }
            }
        })
        .to_string();
        store
            .writer
            .lock()
            .unwrap()
            .execute(
                "UPDATE memory_receipts SET payload = ?1 WHERE receipt_id = 'child-receipt'",
                params![invalid_child_payload],
            )
            .unwrap();
        let error = store.verify_integrity().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("synaptic receipt consistency checks found 1"),
            "unexpected error: {error}"
        );

        let legacy_child_payload = serde_json::json!({
            "evidence": {
                "kind": "synaptic_capture",
                "predicate": {
                    "schemaVersion": 1,
                    "algorithmVersion": "vestige.synaptic_capture.v1",
                    "trigger": { "eventId": "public-event" },
                    "candidates": [{ "evidenceSlot": "candidate_1" }]
                }
            }
        })
        .to_string();
        store
            .writer
            .lock()
            .unwrap()
            .execute(
                "UPDATE memory_receipts SET payload = ?1 WHERE receipt_id = 'child-receipt'",
                params![legacy_child_payload],
            )
            .unwrap();
        let error = store.verify_integrity().unwrap_err();
        assert!(
            error.to_string().contains("synaptic receipt consistency"),
            "a schema-v1 receipt must not validate a V22 forward item: {error}"
        );

        // SQL `NULL IS NOT NULL` is false, so an explicit non-null/type guard
        // is required or a missing event id on both sides becomes fail-open.
        let missing_event_payload = serde_json::json!({
            "evidence": {
                "kind": "synaptic_capture",
                "predicate": {
                    "schemaVersion": 2,
                    "algorithmVersion": "vestige.synaptic_capture.v2",
                    "receiptRole": "root",
                    "trigger": {},
                    "candidates": []
                }
            }
        })
        .to_string();
        {
            let writer = store.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE synaptic_events SET public_event_id = NULL
                     WHERE event_id = 'private-event'",
                    [],
                )
                .unwrap();
            writer
                .execute(
                    "UPDATE memory_receipts SET payload = ?1
                     WHERE receipt_id = 'root-receipt'",
                    params![missing_event_payload],
                )
                .unwrap();
        }
        let error = store.verify_integrity().unwrap_err();
        assert!(
            error.to_string().contains("synaptic receipt consistency"),
            "missing V22 event ids must fail closed: {error}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn hardened_profile_rejects_missing_fullfsync_readback_on_macos() {
        let mut pragmas = SqliteConnectionPragmas {
            journal_mode: "wal".into(),
            synchronous: 2,
            synchronous_label: "full".into(),
            fullfsync_enabled: true,
            fullfsync_meaningful_on_this_platform: true,
            checkpoint_fullfsync_enabled: true,
            wal_autocheckpoint_pages: 1000,
            foreign_keys_enabled: true,
            busy_timeout_ms: 5000,
        };
        pragmas.fullfsync_enabled = false;
        assert!(
            Storage::verify_effective_pragmas(SqliteDurabilityProfile::Hardened, "test", &pragmas)
                .is_err()
        );
        pragmas.fullfsync_enabled = true;
        pragmas.checkpoint_fullfsync_enabled = false;
        assert!(
            Storage::verify_effective_pragmas(SqliteDurabilityProfile::Hardened, "test", &pragmas)
                .is_err()
        );
    }

    #[test]
    fn hardened_writer_refuses_read_only_non_wal_database() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("readonly-delete.db");
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "PRAGMA journal_mode = DELETE;
                 CREATE TABLE seed(id INTEGER PRIMARY KEY);",
            )
            .unwrap();
        }
        let conn =
            Connection::open_with_flags(&path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();

        let error = Storage::configure_connection(&conn, SqliteDurabilityProfile::Hardened, true)
            .unwrap_err();
        assert!(
            error.to_string().contains("readonly")
                || error.to_string().contains("read-only")
                || error.to_string().contains("attempt to write"),
            "unexpected error: {error}"
        );
    }

    #[cfg(unix)]
    const SQLITE_CRASH_CHILD_SCENARIO: &str = "VESTIGE_SQLITE_CRASH_CHILD_SCENARIO";
    #[cfg(unix)]
    const SQLITE_CRASH_CHILD_PATH: &str = "VESTIGE_SQLITE_CRASH_CHILD_PATH";
    #[cfg(unix)]
    const SQLITE_CRASH_READY: &str = "VESTIGE_SQLITE_CRASH_READY";

    /// Subprocess-only entry point for the process-crash durability harness.
    #[cfg(unix)]
    #[test]
    fn sqlite_crash_child() {
        let Ok(scenario) = std::env::var(SQLITE_CRASH_CHILD_SCENARIO) else {
            return;
        };
        let path = PathBuf::from(
            std::env::var_os(SQLITE_CRASH_CHILD_PATH)
                .expect("crash child requires a database path"),
        );
        let store =
            Storage::new_with_durability(Some(path), SqliteDurabilityProfile::Hardened).unwrap();
        let mut writer = store.writer.lock().unwrap();
        let tx = writer
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .unwrap();
        tx.execute(
            "INSERT INTO durability_probe_transactions(id, value)
             VALUES ('ack-boundary', 'parent')",
            [],
        )
        .unwrap();
        tx.execute(
            "INSERT INTO durability_probe_items(transaction_id, item_index, value)
             VALUES ('ack-boundary', 1, 'first'), ('ack-boundary', 2, 'second')",
            [],
        )
        .unwrap();

        if scenario == "before_commit" {
            println!("{SQLITE_CRASH_READY}=before_commit");
            std::io::stdout().flush().unwrap();
            loop {
                std::thread::park_timeout(std::time::Duration::from_secs(60));
            }
        }

        assert_eq!(scenario, "after_commit");
        tx.commit().unwrap();
        drop(writer);
        println!("{SQLITE_CRASH_READY}=after_commit");
        std::io::stdout().flush().unwrap();
        loop {
            std::thread::park_timeout(std::time::Duration::from_secs(60));
        }
    }

    #[cfg(unix)]
    fn spawn_and_kill_at_commit_boundary(path: &Path, scenario: &str) {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("storage::sqlite::tests::sqlite_crash_child")
            .arg("--nocapture")
            .env(SQLITE_CRASH_CHILD_SCENARIO, scenario)
            .env(SQLITE_CRASH_CHILD_PATH, path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (ready_tx, ready_rx) = mpsc::channel();
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            for line in BufReader::new(stdout)
                .lines()
                .map_while(std::result::Result::ok)
            {
                if line.contains(SQLITE_CRASH_READY) {
                    let _ = ready_tx.send(line);
                    return;
                }
            }
        });

        let marker = ready_rx
            .recv_timeout(std::time::Duration::from_secs(20))
            .unwrap_or_else(|error| {
                let _ = child.kill();
                let stderr = child
                    .stderr
                    .take()
                    .map(|mut stderr| {
                        let mut text = String::new();
                        let _ = std::io::Read::read_to_string(&mut stderr, &mut text);
                        text
                    })
                    .unwrap_or_default();
                panic!("crash child did not reach {scenario}: {error}; stderr={stderr}")
            });
        assert!(marker.contains(scenario), "unexpected marker: {marker}");
        child.kill().unwrap();
        let status = child.wait().unwrap();
        assert!(
            !status.success(),
            "crash child should be killed, not exit cleanly"
        );
    }

    #[cfg(unix)]
    fn prepare_crash_probe(path: &Path) {
        let store = Storage::new_with_durability(
            Some(path.to_path_buf()),
            SqliteDurabilityProfile::Hardened,
        )
        .unwrap();
        store
            .writer
            .lock()
            .unwrap()
            .execute_batch(
                "CREATE TABLE durability_probe_transactions(
                     id TEXT PRIMARY KEY,
                     value TEXT NOT NULL
                 ) STRICT;
                 CREATE TABLE durability_probe_items(
                     transaction_id TEXT NOT NULL,
                     item_index INTEGER NOT NULL,
                     value TEXT NOT NULL,
                     PRIMARY KEY(transaction_id, item_index),
                     FOREIGN KEY(transaction_id)
                         REFERENCES durability_probe_transactions(id)
                         ON DELETE CASCADE
                 ) STRICT;",
            )
            .unwrap();
        store.checkpoint_wal(WalCheckpointMode::Truncate).unwrap();
    }

    #[cfg(unix)]
    fn crash_probe_counts(path: &Path) -> (i64, i64) {
        let store = Storage::new_with_durability(
            Some(path.to_path_buf()),
            SqliteDurabilityProfile::Hardened,
        )
        .unwrap();
        assert_eq!(store.verify_integrity().unwrap().quick_check, "ok");
        let reader = store.reader.lock().unwrap();
        let transactions = reader
            .query_row(
                "SELECT COUNT(*) FROM durability_probe_transactions",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let items = reader
            .query_row("SELECT COUNT(*) FROM durability_probe_items", [], |row| {
                row.get(0)
            })
            .unwrap();
        (transactions, items)
    }

    #[cfg(unix)]
    #[test]
    fn sigkill_before_and_after_commit_respects_atomic_ack_boundary() {
        let dir = tempdir().unwrap();

        let before_path = dir.path().join("before-commit.db");
        prepare_crash_probe(&before_path);
        spawn_and_kill_at_commit_boundary(&before_path, "before_commit");
        assert_eq!(crash_probe_counts(&before_path), (0, 0));

        let after_path = dir.path().join("after-commit.db");
        prepare_crash_probe(&after_path);
        spawn_and_kill_at_commit_boundary(&after_path, "after_commit");
        assert_eq!(crash_probe_counts(&after_path), (1, 2));
    }

    // ===================== Connector sync (#57) =========================

    /// Build an `IngestInput` carrying a source envelope for a GitHub-ish issue.
    fn source_input(id: &str, content: &str, hash: &str) -> IngestInput {
        IngestInput {
            content: content.to_string(),
            node_type: "fact".to_string(),
            source_envelope: Some(crate::memory::SourceEnvelope {
                source_system: Some("github".to_string()),
                source_id: Some(id.to_string()),
                source_url: Some(format!("https://github.com/o/r/issues/{id}")),
                content_hash: Some(hash.to_string()),
                source_project: Some("o/r".to_string()),
                source_type: Some("issue".to_string()),
                source_author: Some("octocat".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn node_count(store: &Storage) -> i64 {
        // Count rows for our test source so embeddings/other tests don't bleed in.
        let reader = store.reader.lock().unwrap();
        reader
            .query_row(
                "SELECT COUNT(*) FROM knowledge_nodes WHERE source_system = 'github'",
                [],
                |r| r.get(0),
            )
            .unwrap()
    }

    // ===================== Secret-ingest policy (#154) ==================

    #[test]
    fn ingest_rejects_probable_github_secret_without_persisting_or_echoing_it() {
        let store = create_test_storage();
        let secret = format!("ghp_{}", "A".repeat(36));

        let err = store
            .ingest(IngestInput {
                content: format!("The temporary token is {secret}"),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap_err();

        assert!(matches!(err, StorageError::SecretDetected { .. }));
        assert!(
            !err.to_string().contains(&secret),
            "rejection must not echo the credential"
        );
        assert_eq!(
            store.get_stats().unwrap().total_nodes,
            0,
            "rejection must happen before creating a node"
        );
    }

    #[test]
    fn update_node_content_rejects_probable_github_secret_without_mutating_node() {
        let store = create_test_storage();
        let node = store
            .ingest(IngestInput {
                content: "safe original content".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let secret = format!("ghp_{}", "A".repeat(36));

        let err = store
            .update_node_content(&node.id, &format!("replacement includes {secret}"))
            .unwrap_err();

        assert!(matches!(err, StorageError::SecretDetected { .. }));
        assert!(
            !err.to_string().contains(&secret),
            "rejection must not echo the credential"
        );
        assert_eq!(
            store.get_node(&node.id).unwrap().unwrap().content,
            "safe original content",
            "rejection must leave existing content intact"
        );
    }

    #[test]
    fn connector_upsert_rejects_probable_github_secret_without_mutating_existing_record() {
        let store = create_test_storage();
        let created = store
            .upsert_by_source(source_input(
                "secret-policy",
                "safe connector body",
                "safe-hash",
            ))
            .unwrap();
        let secret = format!("ghp_{}", "A".repeat(36));

        let err = store
            .upsert_by_source(source_input(
                "secret-policy",
                &format!("updated connector body includes {secret}"),
                "secret-hash",
            ))
            .unwrap_err();

        assert!(matches!(err, StorageError::SecretDetected { .. }));
        assert!(
            !err.to_string().contains(&secret),
            "rejection must not echo the credential"
        );
        let stored = store.get_node(&created.node_id).unwrap().unwrap();
        assert_eq!(stored.content, "safe connector body");
        assert_eq!(
            stored
                .source_envelope
                .as_ref()
                .and_then(|envelope| envelope.content_hash.as_deref()),
            Some("safe-hash"),
            "preflight rejection must happen before connector metadata changes"
        );
    }

    #[test]
    fn portable_import_rejects_secret_archive_atomically_before_safe_rows_import() {
        let source_dir = tempdir().unwrap();
        let target_dir = tempdir().unwrap();
        let source = create_test_storage_at(&source_dir, "source.db");
        source
            .ingest(IngestInput {
                content: "safe portable memory".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let secret = format!("ghp_{}", "A".repeat(36));
        source
            .ingest_with_secret_policy(
                IngestInput {
                    content: format!("intentionally archived credential {secret}"),
                    node_type: "fact".to_string(),
                    ..Default::default()
                },
                SecretPolicy::AllowExplicitly,
            )
            .unwrap();
        let archive = source.export_portable_archive().unwrap();

        let target = create_test_storage_at(&target_dir, "target.db");
        let err = target
            .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
            .unwrap_err();

        assert!(matches!(err, StorageError::SecretDetected { .. }));
        assert!(
            !err.to_string().contains(&secret),
            "rejection must not echo the credential"
        );
        assert_eq!(
            target.get_stats().unwrap().total_nodes,
            0,
            "archive preflight must prevent a partial import of safe sibling rows"
        );
    }

    #[test]
    fn upsert_by_source_is_idempotent_across_reruns() {
        let store = create_test_storage();

        // First sync: a brand-new record → Created.
        let r1 = store
            .upsert_by_source(source_input("1", "Bug: crash on startup", "hash-a"))
            .unwrap();
        assert_eq!(r1.outcome, SourceUpsertOutcome::Created);
        assert_eq!(node_count(&store), 1);

        // Re-sync the SAME record with the SAME hash twice → Unchanged, no dupes.
        for _ in 0..2 {
            let r = store
                .upsert_by_source(source_input("1", "Bug: crash on startup", "hash-a"))
                .unwrap();
            assert_eq!(r.outcome, SourceUpsertOutcome::Unchanged);
            assert_eq!(r.node_id, r1.node_id, "must reuse the same memory id");
        }
        assert_eq!(
            node_count(&store),
            1,
            "idempotent: still exactly one memory"
        );
    }

    #[test]
    fn upsert_by_source_updates_in_place_when_hash_changes() {
        let store = create_test_storage();
        let created = store
            .upsert_by_source(source_input("7", "old body", "hash-old"))
            .unwrap();

        // Upstream edit: content + hash change → Updated, same id, new content.
        let updated = store
            .upsert_by_source(source_input("7", "new edited body", "hash-new"))
            .unwrap();
        assert_eq!(updated.outcome, SourceUpsertOutcome::Updated);
        assert_eq!(updated.node_id, created.node_id);
        assert_eq!(node_count(&store), 1, "update must not duplicate");

        let node = store.get_node(&created.node_id).unwrap().unwrap();
        assert_eq!(node.content, "new edited body");
        let env = node.source_envelope.expect("envelope persisted");
        assert_eq!(env.content_hash.as_deref(), Some("hash-new"));
        assert_eq!(env.source_id.as_deref(), Some("7"));
    }

    #[test]
    fn upsert_by_source_without_key_falls_back_to_create() {
        let store = create_test_storage();
        // Envelope present but missing source_id → not keyed → plain create.
        let input = IngestInput {
            content: "loose note".to_string(),
            node_type: "fact".to_string(),
            source_envelope: Some(crate::memory::SourceEnvelope {
                source_url: Some("https://example.com/x".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let r = store.upsert_by_source(input).unwrap();
        assert_eq!(r.outcome, SourceUpsertOutcome::Created);
    }

    #[test]
    fn connector_cursor_round_trips() {
        let store = create_test_storage();
        // Unknown scope → zeroed cursor.
        let empty = store.get_connector_cursor("github", "o/r").unwrap();
        assert!(empty.cursor_updated_at.is_none());
        assert_eq!(empty.records_seen, 0);

        let ts = Utc::now();
        let cursor = ConnectorCursor {
            source_system: "github".to_string(),
            scope: "o/r".to_string(),
            cursor_updated_at: Some(ts),
            last_synced_at: Some(ts),
            last_full_reconcile_at: None,
            records_seen: 42,
        };
        store.save_connector_cursor(&cursor).unwrap();

        let back = store.get_connector_cursor("github", "o/r").unwrap();
        assert_eq!(back.records_seen, 42);
        assert_eq!(
            back.cursor_updated_at.map(|d| d.to_rfc3339()),
            Some(ts.to_rfc3339())
        );

        // Upsert semantics: saving again replaces, never duplicates.
        let mut c2 = cursor.clone();
        c2.records_seen = 99;
        store.save_connector_cursor(&c2).unwrap();
        assert_eq!(
            store
                .get_connector_cursor("github", "o/r")
                .unwrap()
                .records_seen,
            99
        );
    }

    #[test]
    fn reconcile_tombstones_records_absent_from_live_set() {
        let store = create_test_storage();
        // Three synced issues in scope o/r.
        for id in ["1", "2", "3"] {
            store
                .upsert_by_source(source_input(id, &format!("issue {id}"), &format!("h{id}")))
                .unwrap();
        }

        // Reconcile: only 1 and 3 are still visible upstream → 2 is tombstoned.
        let report = store
            .reconcile_source_tombstones("github", "o/r", &["1".to_string(), "3".to_string()])
            .unwrap();
        assert_eq!(report.considered, 3);
        assert_eq!(report.tombstoned.len(), 1, "exactly issue 2 tombstoned");

        // Issue 2's memory is invalidated (valid_until set) but NOT purged —
        // content retained for audit, just no longer currently-valid.
        let two = {
            let reader = store.reader.lock().unwrap();
            reader
                .query_row(
                    "SELECT id, valid_until FROM knowledge_nodes WHERE source_id = '2'",
                    [],
                    |r| Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?)),
                )
                .unwrap()
        };
        assert!(
            two.1.is_some(),
            "tombstoned record must have valid_until set"
        );
        let node = store.get_node(&two.0).unwrap().unwrap();
        assert!(
            !node.is_currently_valid(),
            "tombstoned node is not valid now"
        );
        assert_eq!(node.content, "issue 2", "content retained for audit");

        // A reappearing record un-tombstones on next upsert (clears valid_until).
        store
            .upsert_by_source(source_input("2", "issue 2", "h2"))
            .unwrap();
        let revived = store.get_node(&two.0).unwrap().unwrap();
        assert!(
            revived.is_currently_valid(),
            "re-synced record is valid again"
        );
    }

    #[test]
    fn upsert_clears_superseded_by_when_record_reappears() {
        // Regression: un-tombstoning must clear BOTH bitemporal markers. A
        // connector node that was superseded/merged (valid_until + superseded_by
        // both set) and then re-observed upstream must come back fully clean,
        // otherwise it is currently-valid yet still flagged superseded and is
        // permanently excluded from merge candidacy.
        let store = create_test_storage();
        let created = store
            .upsert_by_source(source_input("9", "body v1", "h9a"))
            .unwrap();

        // Simulate the node having been superseded (as merge/supersede would).
        {
            let writer = store.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET valid_until = ?1, superseded_by = 'survivor-id' WHERE id = ?2",
                    params![Utc::now().to_rfc3339(), created.node_id],
                )
                .unwrap();
        }
        assert!(
            store
                .superseded_node_ids()
                .unwrap()
                .contains(&created.node_id),
            "precondition: node is superseded"
        );

        // Re-sync with a content change → Updated branch must clear both markers.
        let res = store
            .upsert_by_source(source_input("9", "body v2 edited", "h9b"))
            .unwrap();
        assert_eq!(res.outcome, SourceUpsertOutcome::Updated);
        assert!(
            !store
                .superseded_node_ids()
                .unwrap()
                .contains(&created.node_id),
            "superseded_by must be cleared on re-sync (no bitemporal zombie)"
        );
        let node = store.get_node(&created.node_id).unwrap().unwrap();
        assert!(node.is_currently_valid());

        // Also exercise the Unchanged branch: supersede again, re-sync same hash.
        {
            let writer = store.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET valid_until = ?1, superseded_by = 'survivor-id' WHERE id = ?2",
                    params![Utc::now().to_rfc3339(), created.node_id],
                )
                .unwrap();
        }
        let res2 = store
            .upsert_by_source(source_input("9", "body v2 edited", "h9b"))
            .unwrap();
        assert_eq!(res2.outcome, SourceUpsertOutcome::Unchanged);
        assert!(
            !store
                .superseded_node_ids()
                .unwrap()
                .contains(&created.node_id),
            "Unchanged branch must also clear superseded_by"
        );
    }

    /// Build a `source_input` whose envelope carries an explicit project.
    fn source_input_in_project(
        id: &str,
        content: &str,
        hash: &str,
        project: Option<&str>,
    ) -> IngestInput {
        let mut input = source_input(id, content, hash);
        input.source_envelope.as_mut().unwrap().source_project = project.map(str::to_string);
        input
    }

    #[test]
    fn upsert_by_source_scopes_key_by_project() {
        // V19: two sources of the same system reuse bare per-project ids, so
        // the same (system, id) under DIFFERENT projects must yield two
        // distinct nodes, and re-syncing each must hit its own row (Unchanged).
        let store = create_test_storage();

        let a = store
            .upsert_by_source(source_input_in_project(
                "5",
                "repoA issue 5",
                "hA",
                Some("octocat/repoA"),
            ))
            .unwrap();
        let b = store
            .upsert_by_source(source_input_in_project(
                "5",
                "repoB issue 5",
                "hB",
                Some("octocat/repoB"),
            ))
            .unwrap();
        assert_eq!(a.outcome, SourceUpsertOutcome::Created);
        assert_eq!(b.outcome, SourceUpsertOutcome::Created);
        assert_ne!(a.node_id, b.node_id, "projects must not share a row");
        assert_eq!(node_count(&store), 2);

        // Re-sync both records with unchanged hashes → each resolves to ITS row.
        let ra = store
            .upsert_by_source(source_input_in_project(
                "5",
                "repoA issue 5",
                "hA",
                Some("octocat/repoA"),
            ))
            .unwrap();
        assert_eq!(ra.outcome, SourceUpsertOutcome::Unchanged);
        assert_eq!(ra.node_id, a.node_id);
        let rb = store
            .upsert_by_source(source_input_in_project(
                "5",
                "repoB issue 5",
                "hB",
                Some("octocat/repoB"),
            ))
            .unwrap();
        assert_eq!(rb.outcome, SourceUpsertOutcome::Unchanged);
        assert_eq!(rb.node_id, b.node_id);
        assert_eq!(node_count(&store), 2, "resync must not duplicate");
    }

    #[test]
    fn upsert_by_source_matches_legacy_null_project_row_with_empty_string() {
        // Regression: the V19 unique index buckets NULL and '' together via
        // COALESCE(source_project, ''), but the lookup used `source_project IS
        // ?3`, which treats NULL and '' as distinct. A legacy NULL-project row
        // plus an ''-project envelope for the same (system, id) made the lookup
        // miss, and the fall-through INSERT then hit the UNIQUE constraint.
        let store = create_test_storage();
        let created = store
            .upsert_by_source(source_input("41", "legacy body", "h-legacy"))
            .unwrap();
        assert_eq!(created.outcome, SourceUpsertOutcome::Created);

        // Simulate a pre-V19 legacy row: source_project stored as NULL.
        {
            let writer = store.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET source_project = NULL WHERE id = ?1",
                    params![created.node_id],
                )
                .unwrap();
        }

        // New connector run sends Some("") for the same (system, id). Must
        // UPDATE the legacy row in place — not error on the unique index.
        let res = store
            .upsert_by_source(source_input_in_project(
                "41",
                "legacy body edited",
                "h-new",
                Some(""),
            ))
            .expect("''-project envelope must resolve the NULL-project row, not UNIQUE-fail");
        assert_eq!(res.outcome, SourceUpsertOutcome::Updated);
        assert_eq!(res.node_id, created.node_id, "must reuse the legacy row");
        assert_eq!(node_count(&store), 1, "no duplicate row in the NULL bucket");

        // And the Unchanged path resolves through the same bucket too.
        let res2 = store
            .upsert_by_source(source_input_in_project(
                "41",
                "legacy body edited",
                "h-new",
                Some(""),
            ))
            .unwrap();
        assert_eq!(res2.outcome, SourceUpsertOutcome::Unchanged);
        assert_eq!(res2.node_id, created.node_id);
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn with_vector_search_disabled<T>(f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap();
        let previous = std::env::var_os(VESTIGE_DISABLE_VECTOR_SEARCH);

        // Tests serialize access with ENV_LOCK because process environment
        // mutation is global and unsafe under Rust 2024.
        unsafe {
            std::env::set_var(VESTIGE_DISABLE_VECTOR_SEARCH, "1");
        }

        let result = catch_unwind(AssertUnwindSafe(f));

        unsafe {
            if let Some(value) = previous {
                std::env::set_var(VESTIGE_DISABLE_VECTOR_SEARCH, value);
            } else {
                std::env::remove_var(VESTIGE_DISABLE_VECTOR_SEARCH);
            }
        }

        match result {
            Ok(value) => value,
            Err(payload) => resume_unwind(payload),
        }
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_runtime_vector_gate_env_disables_index_creation() {
        with_vector_search_disabled(|| {
            assert!(!Storage::vector_search_enabled_by_cpu());
            assert_eq!(
                Storage::vector_search_unavailable_reason(),
                Some("disabled by VESTIGE_DISABLE_VECTOR_SEARCH")
            );

            let dir = tempdir().unwrap();
            let storage = create_test_storage_at(&dir, "vector-disabled.db");

            assert!(storage.vector_index.is_none());
            assert!(storage.query_cache.is_none());

            let stats = storage.get_stats().unwrap();
            assert_eq!(stats.total_nodes, 0);

            let schema = storage.schema_introspection().unwrap();
            assert!(schema.schema_version >= 1);
        });
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_runtime_vector_gate_disabled_hybrid_search_uses_keyword_fallback() {
        with_vector_search_disabled(|| {
            let dir = tempdir().unwrap();
            let storage = create_test_storage_at(&dir, "vector-disabled-search.db");

            storage
                .ingest(IngestInput {
                    content: "runtime gate fallback keyword anchor".to_string(),
                    node_type: "fact".to_string(),
                    ..Default::default()
                })
                .unwrap();

            let results = storage
                .hybrid_search("runtime gate fallback keyword", 10, 0.3, 0.7)
                .unwrap();

            assert_eq!(results.len(), 1);
            assert_eq!(results[0].match_type, MatchType::Keyword);
            assert!(results[0].semantic_score.is_none());
            assert!(
                results[0]
                    .node
                    .content
                    .contains("runtime gate fallback keyword anchor")
            );
        });
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_embedding_model_identity_matching() {
        assert!(Storage::embedding_model_matches_active(
            "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-0.6B",
        ));
        assert!(!Storage::embedding_model_matches_active(
            "nomic-embed-text-v1.5",
            "nomic-ai/nomic-embed-text-v1.5",
        ));
        assert!(!Storage::embedding_model_matches_active(
            "nomic-ai/nomic-embed-text-v1.5",
            "Qwen/Qwen3-Embedding-0.6B",
        ));

        let bytes = Embedding::new(vec![1.0; EMBEDDING_DIMENSIONS]).to_bytes();
        assert!(
            Storage::embedding_vector_for_active_model(
                &bytes,
                "nomic-ai/nomic-embed-text-v1.5",
                "Qwen/Qwen3-Embedding-0.6B",
            )
            .is_none()
        );
        assert!(
            Storage::embedding_vector_for_active_model(
                &bytes,
                "Qwen/Qwen3-Embedding-0.6B",
                "Qwen/Qwen3-Embedding-0.6B",
            )
            .is_some()
        );
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_embedding_regeneration_candidates_include_entire_mismatched_corpus() {
        let storage = create_test_storage();
        let stale_model = "all-MiniLM-L6-v2";
        let stale_embedding = Embedding::new(vec![0.0; EMBEDDING_DIMENSIONS]).to_bytes();

        for i in 0..125 {
            let node = storage
                .ingest(IngestInput {
                    content: format!("legacy embedded memory {}", i),
                    node_type: "fact".to_string(),
                    ..Default::default()
                })
                .unwrap();

            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "INSERT OR REPLACE INTO node_embeddings
                     (node_id, embedding, dimensions, model, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    rusqlite::params![
                        &node.id,
                        &stale_embedding,
                        EMBEDDING_DIMENSIONS as i32,
                        stale_model,
                        Utc::now().to_rfc3339()
                    ],
                )
                .unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes
                     SET has_embedding = 1, embedding_model = ?2
                     WHERE id = ?1",
                    rusqlite::params![&node.id, stale_model],
                )
                .unwrap();
            // In a process where the legacy runtime was initialized by an
            // earlier test, ingest also writes a matching V28 profile vector.
            // Remove it so this fixture consistently represents a pre-V28
            // mirror-only stale corpus.
            writer
                .execute(
                    "DELETE FROM embedding_profile_vectors
                     WHERE profile_id = ?1 AND node_id = ?2",
                    rusqlite::params![LEGACY_EMBEDDING_PROFILE_ID, &node.id],
                )
                .unwrap();
        }

        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.nodes_with_mismatched_embeddings, 125);
        assert_eq!(stats.nodes_with_active_embeddings, 0);

        let legacy = EmbeddingProfileId::new(LEGACY_EMBEDDING_PROFILE_ID).unwrap();
        let candidates = storage
            .embedding_regeneration_candidates(
                &legacy,
                EMBEDDING_DIMENSIONS,
                "nomic-ai/nomic-embed-text-v1.5",
                None,
                false,
            )
            .unwrap();
        assert_eq!(candidates.len(), 125);
    }

    #[test]
    fn test_storage_creation() {
        let storage = create_test_storage();
        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.total_nodes, 0);
    }

    #[test]
    fn reopening_after_qwen_pointer_preserves_legacy_manifest_state() {
        let dir = tempdir().unwrap();
        let qwen = BuiltinEmbeddingProfile::QwenBalanced1024
            .profile()
            .profile_id;
        {
            let storage = create_test_storage_at(&dir, "reopen-qwen.db");
            storage
                .save_embedding_profile_manifest(&ready_profile_manifest(
                    BuiltinEmbeddingProfile::QwenBalanced1024,
                ))
                .unwrap();
            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE embedding_profiles SET status = 'ready' WHERE profile_id = ?1",
                    params![LEGACY_EMBEDDING_PROFILE_ID],
                )
                .unwrap();
            writer
                .execute(
                    "UPDATE embedding_profiles SET status = 'active' WHERE profile_id = ?1",
                    params![qwen.as_str()],
                )
                .unwrap();
            writer
                .execute(
                    "UPDATE embedding_profile_state
                     SET active_profile_id = ?1, previous_profile_id = ?2, activated_at = ?3, updated_at = ?3
                     WHERE singleton = 1",
                    params![
                        qwen.as_str(),
                        LEGACY_EMBEDDING_PROFILE_ID,
                        Utc::now().to_rfc3339(),
                    ],
                )
                .unwrap();
        }

        let reopened = create_test_storage_at(&dir, "reopen-qwen.db");
        assert_eq!(
            reopened
                .active_embedding_profile()
                .unwrap()
                .unwrap()
                .profile_id,
            qwen
        );
        let legacy = EmbeddingProfileId::new(LEGACY_EMBEDDING_PROFILE_ID).unwrap();
        assert_eq!(
            reopened
                .embedding_profile_manifest(&legacy)
                .unwrap()
                .unwrap()
                .state,
            EmbeddingProfileState::Active,
            "a valid legacy manifest is preserved exactly; bootstrap must not rewrite it"
        );
    }

    fn ready_profile_manifest(profile: BuiltinEmbeddingProfile) -> EmbeddingProfileManifest {
        let mut manifest = EmbeddingProfileManifest::not_installed(profile.profile()).unwrap();
        manifest.state = EmbeddingProfileState::Ready;
        manifest
    }

    #[test]
    fn embedding_profiles_keep_vectors_isolated() {
        let storage = create_test_storage();
        let legacy = EmbeddingProfileId::new(LEGACY_EMBEDDING_PROFILE_ID).unwrap();
        let qwen = BuiltinEmbeddingProfile::QwenBalanced1024
            .profile()
            .profile_id;
        storage
            .save_embedding_profile_manifest(&ready_profile_manifest(
                BuiltinEmbeddingProfile::QwenBalanced1024,
            ))
            .unwrap();
        let node = storage
            .ingest(IngestInput {
                content: "profile vector isolation".to_string(),
                ..Default::default()
            })
            .unwrap();

        storage
            .put_embedding_profile_vector(&EmbeddingProfileVector {
                profile_id: legacy.to_string(),
                node_id: node.id.clone(),
                embedding: vec![1, 2, 3],
                dimensions: 256,
                model: "legacy".to_string(),
                created_at: Utc::now(),
            })
            .unwrap();
        storage
            .put_embedding_profile_vector(&EmbeddingProfileVector {
                profile_id: qwen.to_string(),
                node_id: node.id.clone(),
                embedding: vec![4, 5, 6],
                dimensions: 1024,
                model: "qwen".to_string(),
                created_at: Utc::now(),
            })
            .unwrap();

        assert_eq!(
            storage
                .embedding_profile_vector(&legacy, &node.id)
                .unwrap()
                .unwrap()
                .embedding,
            vec![1, 2, 3]
        );
        assert_eq!(
            storage
                .embedding_profile_vector(&qwen, &node.id)
                .unwrap()
                .unwrap()
                .embedding,
            vec![4, 5, 6]
        );

        // Neither profile can see the other profile's vector. The source row
        // remains, so a later validated activation never needs a re-embed.
        assert!(
            storage
                .embedding_profile_vector(&legacy, &node.id)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn activation_rejects_ready_profile_without_verified_runtime_and_evaluation() {
        let storage = create_test_storage();
        let manifest = ready_profile_manifest(BuiltinEmbeddingProfile::QwenBalanced1024);
        let profile_id = manifest.profile.profile_id.clone();
        storage.save_embedding_profile_manifest(&manifest).unwrap();

        let error = storage.activate_embedding_profile(&profile_id).unwrap_err();
        assert!(matches!(error, StorageError::InvalidEmbeddingProfile(_)));
        assert_eq!(
            storage
                .active_embedding_profile()
                .unwrap()
                .unwrap()
                .profile_id
                .as_str(),
            LEGACY_EMBEDDING_PROFILE_ID
        );
    }

    #[cfg(feature = "embeddings")]
    #[test]
    fn init_embeddings_permits_the_released_legacy_nomic_profile() {
        let storage = create_test_storage();

        // `init_embeddings` still owns the released Nomic startup path. The
        // actual backend may be unavailable in an offline test environment,
        // but that must surface as backend initialization failure, never as a
        // profile-policy rejection.
        match storage.init_embeddings() {
            Ok(()) | Err(StorageError::Init(_)) => {}
            Err(error) => {
                panic!("legacy Nomic profile must be permitted to use init_embeddings: {error}")
            }
        }
    }

    #[cfg(feature = "embeddings")]
    #[test]
    fn init_embeddings_rejects_an_active_qwen_profile() {
        let storage = create_test_storage();
        let qwen = BuiltinEmbeddingProfile::QwenBalanced1024
            .profile()
            .profile_id;
        storage
            .save_embedding_profile_manifest(&ready_profile_manifest(
                BuiltinEmbeddingProfile::QwenBalanced1024,
            ))
            .unwrap();

        // This intentionally sets only the persisted active pointer. It does
        // not satisfy the activation gate; the point is to verify that the
        // legacy convenience initializer fails closed before it can select or
        // initialize a different vector-space runtime.
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE embedding_profiles SET status = 'ready' WHERE profile_id = ?1",
                params![LEGACY_EMBEDDING_PROFILE_ID],
            )
            .unwrap();
        writer
            .execute(
                "UPDATE embedding_profiles SET status = 'active' WHERE profile_id = ?1",
                params![qwen.as_str()],
            )
            .unwrap();
        writer
            .execute(
                "UPDATE embedding_profile_state
                 SET active_profile_id = ?1, previous_profile_id = ?2,
                     activated_at = ?3, updated_at = ?3
                 WHERE singleton = 1",
                params![
                    qwen.as_str(),
                    LEGACY_EMBEDDING_PROFILE_ID,
                    Utc::now().to_rfc3339(),
                ],
            )
            .unwrap();
        drop(writer);

        let error = storage.init_embeddings().unwrap_err();
        assert!(matches!(error, StorageError::InvalidEmbeddingProfile(_)));
        assert!(error.to_string().contains(qwen.as_str()));
        assert!(error.to_string().contains("explicit profile workflow"));
    }

    #[test]
    fn migration_vector_and_node_checkpoint_commit_together() {
        let storage = create_test_storage();
        let legacy = EmbeddingProfileId::new(LEGACY_EMBEDDING_PROFILE_ID).unwrap();
        let qwen = BuiltinEmbeddingProfile::QwenBalanced1024
            .profile()
            .profile_id;
        storage
            .save_embedding_profile_manifest(&ready_profile_manifest(
                BuiltinEmbeddingProfile::QwenBalanced1024,
            ))
            .unwrap();
        let migration_id = Uuid::new_v4();
        let now = Utc::now();
        storage
            .save_profile_migration_checkpoint(&ProfileMigrationCheckpoint {
                migration_id,
                source_profile_id: legacy,
                destination_profile_id: qwen.clone(),
                state: EmbeddingMigrationState::Running,
                total_memories: 1,
                completed_memories: 0,
                failed_memory_ids: Vec::new(),
                last_memory_id: None,
                started_at: now,
                updated_at: now,
            })
            .unwrap();
        let node = storage
            .ingest(IngestInput {
                content: "atomic migration checkpoint target".to_string(),
                ..Default::default()
            })
            .unwrap();
        storage
            .put_embedding_profile_vector_with_migration_checkpoint(
                &EmbeddingProfileVector {
                    profile_id: qwen.to_string(),
                    node_id: node.id.clone(),
                    embedding: vec![1, 2, 3],
                    dimensions: 1024,
                    model: "qwen-test".to_string(),
                    created_at: now,
                },
                &EmbeddingProfileMigrationNodeCheckpoint {
                    migration_id: migration_id.to_string(),
                    node_id: node.id.clone(),
                    state: "completed".to_string(),
                    error: None,
                    updated_at: now,
                },
            )
            .unwrap();
        assert!(
            storage
                .embedding_profile_vector(&qwen, &node.id)
                .unwrap()
                .is_some()
        );
        let reader = storage.reader.lock().unwrap();
        let checkpoint_rows: i64 = reader
            .query_row(
                "SELECT COUNT(*) FROM embedding_profile_migration_checkpoints
                 WHERE migration_id = ?1 AND node_id = ?2",
                params![migration_id.to_string(), node.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(checkpoint_rows, 1);
    }

    #[test]
    fn purge_removes_vectors_from_every_embedding_profile() {
        let storage = create_test_storage();
        let node = storage
            .ingest(IngestInput {
                content: "profile-wide purge target".to_string(),
                ..Default::default()
            })
            .unwrap();
        let legacy = EmbeddingProfileId::new(LEGACY_EMBEDDING_PROFILE_ID).unwrap();
        let qwen = BuiltinEmbeddingProfile::QwenBalanced1024
            .profile()
            .profile_id;
        storage
            .save_embedding_profile_manifest(&ready_profile_manifest(
                BuiltinEmbeddingProfile::QwenBalanced1024,
            ))
            .unwrap();
        for (profile_id, dimensions) in [(&legacy, 256), (&qwen, 1024)] {
            storage
                .put_embedding_profile_vector(&EmbeddingProfileVector {
                    profile_id: profile_id.to_string(),
                    node_id: node.id.clone(),
                    embedding: vec![7, 8, 9],
                    dimensions,
                    model: "test".to_string(),
                    created_at: Utc::now(),
                })
                .unwrap();
        }

        storage
            .purge_node(&node.id, Some("profile purge test"))
            .unwrap();
        let reader = storage.reader.lock().unwrap();
        let remaining: i64 = reader
            .query_row(
                "SELECT COUNT(*) FROM embedding_profile_vectors WHERE node_id = ?1",
                params![node.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(remaining, 0, "purge must cascade to every profile vector");
    }

    fn ingest_tagged_in_scope(
        storage: &Storage,
        scope: &str,
        content: &str,
        tags: &[&str],
    ) -> KnowledgeNode {
        storage
            .ingest_in_scope(
                IngestInput {
                    content: content.to_string(),
                    node_type: "fact".to_string(),
                    tags: tags.iter().map(|tag| tag.to_string()).collect(),
                    ..Default::default()
                },
                scope,
            )
            .unwrap()
    }

    fn preview_token(preview: &serde_json::Value) -> &str {
        preview["previewToken"].as_str().unwrap()
    }

    #[test]
    fn tag_rename_is_previewed_scoped_exact_atomic_audited_and_reversible() {
        let storage = create_test_storage();
        let user = ingest_tagged_in_scope(
            &storage,
            "user",
            "scoped tag rename fixture",
            &[
                "keep",
                "legacytag156",
                "canonicaltag156",
                "canonicaltag156",
                "tail",
            ],
        );
        let prefix = ingest_tagged_in_scope(
            &storage,
            "user",
            "exact matching fixture",
            &["legacytag156-extra"],
        );
        let project = ingest_tagged_in_scope(
            &storage,
            "project-a",
            "cross scope fixture",
            &["legacytag156"],
        );
        let sources = vec!["legacytag156".to_string()];

        let preview = storage
            .preview_tag_mutation(&sources, "canonicaltag156", Some(" user "))
            .unwrap();
        assert_eq!(preview["affectedMemoryCount"], 1);
        assert_eq!(
            storage.get_node(&user.id).unwrap().unwrap().tags,
            vec![
                "keep",
                "legacytag156",
                "canonicaltag156",
                "canonicaltag156",
                "tail"
            ],
            "preview must not mutate"
        );

        let operation = storage
            .apply_tag_mutation(
                &sources,
                "canonicaltag156",
                Some("user"),
                preview_token(&preview),
                "tag_rename",
                "standardize the issue 156 fixture tag",
            )
            .unwrap();
        assert_eq!(operation.op_type, "tag_rename");
        assert_eq!(operation.affected_ids, vec![user.id.clone()]);
        assert_eq!(
            storage.get_node(&user.id).unwrap().unwrap().tags,
            vec!["keep", "canonicaltag156", "tail"],
            "the union of source and existing target tags is one target at the first affected position"
        );
        assert_eq!(
            storage.get_node(&prefix.id).unwrap().unwrap().tags,
            vec!["legacytag156-extra"],
            "prefix tags must not match"
        );
        assert_eq!(
            storage.get_node(&project.id).unwrap().unwrap().tags,
            vec!["legacytag156"],
            "default scoped maintenance must not cross project boundaries"
        );
        assert!(
            storage
                .keyword_search("canonicaltag156", 10, 0.0)
                .unwrap()
                .iter()
                .any(|node| node.id == user.id),
            "the existing knowledge_nodes update trigger must keep FTS tag search consistent"
        );
        let logged = storage.get_merge_operation(&operation.id).unwrap().unwrap();
        assert_eq!(
            logged.reason.as_deref(),
            Some("standardize the issue 156 fixture tag")
        );

        storage.undo_tag_mutation(&operation.id).unwrap();
        assert_eq!(
            storage.get_node(&user.id).unwrap().unwrap().tags,
            vec![
                "keep",
                "legacytag156",
                "canonicaltag156",
                "canonicaltag156",
                "tail"
            ],
            "undo restores the exact pre-operation array"
        );
    }

    #[test]
    fn tag_rename_all_scopes_is_explicit_and_updates_every_matching_row() {
        let storage = create_test_storage();
        let user = ingest_tagged_in_scope(&storage, "user", "user shared tag", &["shared"]);
        let project =
            ingest_tagged_in_scope(&storage, "project-a", "project shared tag", &["shared"]);
        let sources = vec!["shared".to_string()];

        let scoped = storage
            .preview_tag_mutation(&sources, "canonical", Some("user"))
            .unwrap();
        assert_eq!(scoped["affectedMemoryCount"], 1);
        assert_eq!(scoped["allScopes"], false);

        let preview = storage
            .preview_tag_mutation(&sources, "canonical", None)
            .unwrap();
        assert_eq!(preview["allScopes"], true);
        assert_eq!(preview["affectedMemoryCount"], 2);

        storage
            .apply_tag_mutation(
                &sources,
                "canonical",
                None,
                preview_token(&preview),
                "tag_rename",
                "explicit cross-scope rename",
            )
            .unwrap();
        assert_eq!(
            storage.get_node(&user.id).unwrap().unwrap().tags,
            vec!["canonical"]
        );
        assert_eq!(
            storage.get_node(&project.id).unwrap().unwrap().tags,
            vec!["canonical"]
        );
    }

    #[test]
    fn list_tag_operations_is_not_buried_by_later_merge_rows() {
        let storage = create_test_storage();
        ingest_tagged_in_scope(&storage, "user", "tag burial fixture", &["old"]);
        let sources = vec!["old".to_string()];
        let preview = storage
            .preview_tag_mutation(&sources, "new", Some("user"))
            .unwrap();
        let operation = storage
            .apply_tag_mutation(
                &sources,
                "new",
                Some("user"),
                preview_token(&preview),
                "tag_rename",
                "must remain listed after later merge rows",
            )
            .unwrap();

        {
            let writer = storage.writer.lock().unwrap();
            for index in 0..20 {
                writer
                    .execute(
                        "INSERT INTO merge_operations
                            (id, plan_id, op_type, status, created_at, reverted_at, reverts_op_id,
                             survivor_id, affected_ids, confidence, signals, reason, undo_payload)
                         VALUES (?1, NULL, 'merge', 'applied', ?2, NULL, NULL, NULL, '[]', NULL, NULL, 'later merge', '{}')",
                        params![
                            format!("merge-later-{index:02}"),
                            "2099-01-01T00:00:00+00:00"
                        ],
                    )
                    .unwrap();
            }
        }

        let mixed = storage.list_merge_operations(20).unwrap();
        assert_eq!(mixed.len(), 20);
        assert!(
            mixed
                .iter()
                .all(|operation| operation.op_type != "tag_rename"),
            "the mixed window fills with later merge rows"
        );
        let tags = storage.list_tag_operations(50, None).unwrap();
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].id, operation.id);
        let scoped = storage.list_tag_operations(50, Some("user")).unwrap();
        assert_eq!(scoped.len(), 1);
    }

    #[test]
    fn tag_merge_normalizes_sources_and_rejects_stale_preview_or_undo_conflict() {
        let storage = create_test_storage();
        let node = ingest_tagged_in_scope(
            &storage,
            "user",
            "multi source tag merge",
            &["keep", "beta", "alpha", "target", "target", "tail"],
        );
        // Duplicate sources dedupe; matching is byte-exact (padded variants
        // are separate, reachable keys — see the padded-source test).
        let sources = vec!["beta".to_string(), "alpha".to_string(), "alpha".to_string()];
        let preview = storage
            .preview_tag_mutation(&sources, "target", Some("user"))
            .unwrap();

        {
            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET tags = ?1 WHERE id = ?2",
                    params![
                        serde_json::json!(["keep", "beta", "alpha", "drift"]).to_string(),
                        &node.id
                    ],
                )
                .unwrap();
        }
        let stale_error = storage
            .apply_tag_mutation(
                &sources,
                "target",
                Some("user"),
                preview_token(&preview),
                "tag_merge",
                "merge aliases",
            )
            .unwrap_err();
        assert!(stale_error.to_string().contains("stale"));
        assert!(storage.list_merge_operations(20).unwrap().is_empty());

        let fresh = storage
            .preview_tag_mutation(&sources, "target", Some("user"))
            .unwrap();
        let operation = storage
            .apply_tag_mutation(
                &sources,
                "target",
                Some("user"),
                preview_token(&fresh),
                "tag_merge",
                "merge aliases",
            )
            .unwrap();
        assert_eq!(
            storage.get_node(&node.id).unwrap().unwrap().tags,
            vec!["keep", "target", "drift"]
        );

        {
            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET tags = ?1 WHERE id = ?2",
                    params![
                        serde_json::json!(["keep", "target", "later-edit"]).to_string(),
                        &node.id
                    ],
                )
                .unwrap();
        }
        let conflict = storage.undo_tag_mutation(&operation.id).unwrap_err();
        assert!(conflict.to_string().contains("conflict"));
        assert_eq!(
            storage
                .get_merge_operation(&operation.id)
                .unwrap()
                .unwrap()
                .status,
            "applied",
            "failed undo must not mark the original operation reverted"
        );
    }

    #[test]
    fn tag_mutation_validation_and_malformed_json_fail_without_partial_writes() {
        let storage = create_test_storage();
        let first = ingest_tagged_in_scope(&storage, "user", "first atomic row", &["old"]);
        let second = ingest_tagged_in_scope(&storage, "user", "second atomic row", &["old"]);
        assert!(
            storage
                .preview_tag_mutation(&["same".into()], "same", Some("user"))
                .is_err()
        );
        assert!(
            storage
                .preview_tag_mutation(&["bad\ncontrol".into()], "new", Some("user"))
                .is_err()
        );

        let sources = vec!["old".to_string()];
        let preview = storage
            .preview_tag_mutation(&sources, "new", Some("user"))
            .unwrap();
        {
            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET tags = 'not-json' WHERE id = ?1",
                    params![&second.id],
                )
                .unwrap();
        }
        let error = storage
            .apply_tag_mutation(
                &sources,
                "new",
                Some("user"),
                preview_token(&preview),
                "tag_rename",
                "atomic malformed-json test",
            )
            .unwrap_err();
        assert!(error.to_string().contains("invalid tags JSON"));
        assert_eq!(
            storage.get_node(&first.id).unwrap().unwrap().tags,
            vec!["old"]
        );
        assert!(storage.list_merge_operations(20).unwrap().is_empty());

        let valid_storage = create_test_storage();
        ingest_tagged_in_scope(&valid_storage, "user", "no match", &["other"]);
        let empty = valid_storage
            .preview_tag_mutation(&sources, "new", Some("user"))
            .unwrap();
        assert_eq!(empty["affectedMemoryCount"], 0);
        assert!(
            valid_storage
                .apply_tag_mutation(
                    &sources,
                    "new",
                    Some("user"),
                    preview_token(&empty),
                    "tag_rename",
                    "must reject no-op",
                )
                .is_err()
        );
        assert!(
            valid_storage
                .apply_tag_mutation(
                    &["other".into()],
                    "new",
                    Some("user"),
                    "tag-plan-v1:wrong",
                    "tag_rename",
                    "",
                )
                .is_err(),
            "a nonempty audit reason is mandatory"
        );
    }

    #[test]
    fn tag_mutation_rejects_secret_shaped_persistent_fields_without_side_effects() {
        let storage = create_test_storage();
        let node = ingest_tagged_in_scope(&storage, "user", "secret policy fixture", &["old"]);
        let source = vec!["old".to_string()];
        let credential = format!("ghp_{}", "a".repeat(36));

        let target_error = storage
            .preview_tag_mutation(&source, &credential, Some("user"))
            .unwrap_err()
            .to_string();
        assert!(target_error.contains("probable credential"));
        assert!(!target_error.contains(&credential));

        // SOURCE tags are exact-match lookup keys for values that already
        // exist in the store, so a secret-shaped source is accepted (it must
        // stay renameable AWAY); only newly persisted fields are rejected.
        let source_preview = storage
            .preview_tag_mutation(std::slice::from_ref(&credential), "new", Some("user"))
            .unwrap();
        assert_eq!(source_preview["affectedMemoryCount"], 0);

        let safe_preview = storage
            .preview_tag_mutation(&source, "new", Some("user"))
            .unwrap();
        let reason_error = storage
            .apply_tag_mutation(
                &source,
                "new",
                Some("user"),
                preview_token(&safe_preview),
                "tag_rename",
                &credential,
            )
            .unwrap_err()
            .to_string();
        assert!(reason_error.contains("probable credential"));
        assert!(!reason_error.contains(&credential));
        assert_eq!(
            storage.get_node(&node.id).unwrap().unwrap().tags,
            vec!["old"]
        );
        assert!(storage.list_merge_operations(20).unwrap().is_empty());
    }

    #[test]
    fn tag_mutation_row_and_audit_limits_fail_before_writes() {
        let storage = create_test_storage();
        let nodes: Vec<_> = (0..3)
            .map(|index| {
                ingest_tagged_in_scope(
                    &storage,
                    "user",
                    &format!("row limit fixture {index}"),
                    &["old"],
                )
            })
            .collect();
        let source = vec!["old".to_string()];
        let preview = storage
            .preview_tag_mutation(&source, "new", Some("user"))
            .unwrap();

        let row_limit_error = storage
            .apply_tag_mutation_with_limits(
                &source,
                "new",
                Some("user"),
                preview_token(&preview),
                "tag_rename",
                "row limit fixture",
                2,
                MAX_TAG_MUTATION_AUDIT_BYTES,
            )
            .unwrap_err()
            .to_string();
        assert!(row_limit_error.contains("more than 2 memories"));
        assert!(storage.list_merge_operations(20).unwrap().is_empty());
        for node in &nodes {
            assert_eq!(
                storage.get_node(&node.id).unwrap().unwrap().tags,
                vec!["old"]
            );
        }

        let applied = storage
            .apply_tag_mutation_with_limits(
                &source,
                "new",
                Some("user"),
                preview_token(&preview),
                "tag_rename",
                "exact row limit fixture",
                3,
                MAX_TAG_MUTATION_AUDIT_BYTES,
            )
            .unwrap();
        assert_eq!(applied.affected_ids.len(), 3);

        let audit_storage = create_test_storage();
        let audit_node = ingest_tagged_in_scope(
            &audit_storage,
            "user",
            "audit payload limit fixture",
            &["old"],
        );
        let audit_preview = audit_storage
            .preview_tag_mutation(&source, "new", Some("user"))
            .unwrap();
        let audit_limit_error = audit_storage
            .apply_tag_mutation_with_limits(
                &source,
                "new",
                Some("user"),
                preview_token(&audit_preview),
                "tag_rename",
                "audit payload limit fixture",
                MAX_TAG_MUTATION_MEMORIES,
                1,
            )
            .unwrap_err()
            .to_string();
        assert!(audit_limit_error.contains("1-byte limit"));
        assert_eq!(
            audit_storage
                .get_node(&audit_node.id)
                .unwrap()
                .unwrap()
                .tags,
            vec!["old"]
        );
        assert!(audit_storage.list_merge_operations(20).unwrap().is_empty());
    }

    #[test]
    fn hygiene_snapshot_covers_more_than_five_hundred_without_full_content() {
        let storage = create_test_storage();
        let now = Utc::now().to_rfc3339();
        {
            let mut writer = storage.writer.lock().unwrap();
            let tx = writer.transaction().unwrap();
            for index in 0..501 {
                tx.execute(
                    "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed, tags, scope)
                     VALUES (?1, ?2, 'fact', ?3, ?3, ?3, ?4, 'user')",
                    params![
                        format!("hygiene-{index:04}"),
                        format!("bounded content {index}"),
                        &now,
                        serde_json::json!(["bulk"]).to_string(),
                    ],
                )
                .unwrap();
            }
            tx.commit().unwrap();
        }
        let snapshot = storage.hygiene_snapshot(Some("user")).unwrap();
        assert_eq!(snapshot.nodes.len(), 501);
        assert!(
            snapshot
                .nodes
                .iter()
                .all(|row| row.content_preview.len() <= 240)
        );
        assert!(snapshot.nodes.iter().all(|row| row.never_accessed));
        assert!(snapshot.nodes.iter().all(|row| !row.access_unknown));
        assert_eq!(snapshot.malformed_tag_rows, 0);
        assert_eq!(snapshot.defaulted_retention_rows, 0);
    }

    #[test]
    fn hygiene_snapshot_distinguishes_unknown_pruned_access_from_never_accessed() {
        let storage = create_test_storage();
        let now = Utc::now();
        let fresh = now.to_rfc3339();
        let before_log_window =
            (now - Duration::days(ACCESS_LOG_RETENTION_DAYS + 110)).to_rfc3339();
        {
            let writer = storage.writer.lock().unwrap();
            // Heavily used old memory: its log rows were pruned, but the
            // durable retrieval counter survives on the node row.
            writer
                .execute(
                    "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope, times_retrieved)
                     VALUES ('old-used', 'used before the log window', 'fact',
                             ?1, ?1, ?1, '[]', 'user', 5)",
                    params![&before_log_window],
                )
                .unwrap();
            // Old memory with no evidence either way: pruning makes its
            // history unknowable, so it must not be claimed never-accessed.
            writer
                .execute(
                    "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope)
                     VALUES ('old-unknown', 'predates the log window', 'fact',
                             ?1, ?1, ?1, '[]', 'user')",
                    params![&before_log_window],
                )
                .unwrap();
            // Fresh memory with no accesses: the retained log is complete
            // evidence for its whole lifetime, so never-accessed is provable.
            writer
                .execute(
                    "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope)
                     VALUES ('fresh-never', 'created inside the log window', 'fact',
                             ?1, ?1, ?1, '[]', 'user')",
                    params![&fresh],
                )
                .unwrap();
        }
        let shown = ingest_tagged_in_scope(&storage, "user", "fresh shown row", &[]);
        storage.record_batch_retrieval(&[&shown.id]).unwrap();

        let snapshot = storage.hygiene_snapshot(Some("user")).unwrap();
        let by_id = |id: &str| {
            snapshot
                .nodes
                .iter()
                .find(|node| node.id == id)
                .unwrap_or_else(|| panic!("snapshot row {id}"))
        };
        let old_used = by_id("old-used");
        assert!(
            !old_used.never_accessed && !old_used.access_unknown,
            "a durable retrieval counter proves past access even after log pruning"
        );
        let old_unknown = by_id("old-unknown");
        assert!(
            !old_unknown.never_accessed,
            "a pre-window row without counters must never be claimed never-accessed"
        );
        assert!(old_unknown.access_unknown);
        let fresh_never = by_id("fresh-never");
        assert!(fresh_never.never_accessed && !fresh_never.access_unknown);
        let shown_row = by_id(&shown.id);
        assert!(!shown_row.never_accessed && !shown_row.access_unknown);
    }

    #[test]
    fn hygiene_snapshot_tolerates_malformed_and_null_legacy_rows() {
        let storage = create_test_storage();
        for index in 0..3 {
            ingest_tagged_in_scope(&storage, "user", &format!("clean row {index}"), &["clean"]);
        }
        let now = Utc::now().to_rfc3339();
        {
            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope)
                     VALUES ('bad-json', 'hand-edited tags', 'fact', ?1, ?1, ?1,
                             'not-json', 'user')",
                    params![&now],
                )
                .unwrap();
            writer
                .execute(
                    "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope, retention_strength)
                     VALUES ('null-tags', 'hand-edited null row', 'fact', ?1, ?1, ?1,
                             NULL, 'user', NULL)",
                    params![&now],
                )
                .unwrap();
        }

        let snapshot = storage.hygiene_snapshot(Some("user")).unwrap();
        assert_eq!(
            snapshot.nodes.len(),
            5,
            "aggregates must cover every row, corrupted ones included"
        );
        assert_eq!(snapshot.malformed_tag_rows, 2);
        assert_eq!(
            snapshot.malformed_tag_row_ids,
            vec!["bad-json".to_string(), "null-tags".to_string()]
        );
        assert!(!snapshot.malformed_tag_row_ids_truncated);
        assert_eq!(snapshot.defaulted_retention_rows, 1);
        let null_row = snapshot
            .nodes
            .iter()
            .find(|node| node.id == "null-tags")
            .unwrap();
        assert!(null_row.tags.is_empty());
        assert_eq!(null_row.retention_strength, 1.0);
    }

    #[test]
    fn tag_apply_and_audit_roll_back_together_on_injected_failure() {
        let storage = create_test_storage();
        let first = ingest_tagged_in_scope(&storage, "user", "atomic pair first", &["old"]);
        let second = ingest_tagged_in_scope(&storage, "user", "atomic pair second", &["old"]);
        let sources = vec!["old".to_string()];
        let preview = storage
            .preview_tag_mutation(&sources, "new", Some("user"))
            .unwrap();

        FAIL_TAG_MUTATION_BEFORE_AUDIT.with(|flag| flag.set(true));
        let error = storage
            .apply_tag_mutation(
                &sources,
                "new",
                Some("user"),
                preview_token(&preview),
                "tag_rename",
                "single transaction fail point",
            )
            .unwrap_err();
        FAIL_TAG_MUTATION_BEFORE_AUDIT.with(|flag| flag.set(false));
        assert!(error.to_string().contains("fail point"));
        // The failure fired AFTER every row UPDATE: rollback must restore all
        // tag arrays AND leave no audit row, proving one shared transaction.
        assert_eq!(
            storage.get_node(&first.id).unwrap().unwrap().tags,
            vec!["old"]
        );
        assert_eq!(
            storage.get_node(&second.id).unwrap().unwrap().tags,
            vec!["old"]
        );
        assert!(storage.list_merge_operations(20).unwrap().is_empty());
        assert!(storage.list_tag_operations(20, None).unwrap().is_empty());

        // Disarmed, the same untouched preview applies normally.
        let applied = storage
            .apply_tag_mutation(
                &sources,
                "new",
                Some("user"),
                preview_token(&preview),
                "tag_rename",
                "single transaction fail point disarmed",
            )
            .unwrap();
        assert_eq!(applied.affected_ids.len(), 2);
        assert_eq!(
            storage.get_node(&first.id).unwrap().unwrap().tags,
            vec!["new"]
        );
    }

    #[test]
    fn tag_vocabulary_skips_and_counts_overlong_stored_tags() {
        let storage = create_test_storage();
        ingest_tagged_in_scope(&storage, "user", "normal tag row", &["normal"]);
        let overlong = "x".repeat(201);
        ingest_tagged_in_scope(&storage, "user", "overlong tag row", &[&overlong]);

        let vocabulary = storage.tag_vocabulary(Some("user")).unwrap();
        assert_eq!(vocabulary.tags, vec!["normal".to_string()]);
        assert_eq!(vocabulary.skipped_overlong, 1);
    }

    #[test]
    fn overlong_source_tags_can_be_renamed_away_end_to_end() {
        let storage = create_test_storage();
        let overlong = "y".repeat(250);
        let node =
            ingest_tagged_in_scope(&storage, "user", "overlong rename fixture", &[&overlong]);

        let sources = vec![overlong.clone()];
        let preview = storage
            .preview_tag_mutation(&sources, "short-tag", Some("user"))
            .unwrap();
        assert_eq!(preview["affectedMemoryCount"], 1);
        storage
            .apply_tag_mutation(
                &sources,
                "short-tag",
                Some("user"),
                preview_token(&preview),
                "tag_rename",
                "repair an overlong stored tag",
            )
            .unwrap();
        assert_eq!(
            storage.get_node(&node.id).unwrap().unwrap().tags,
            vec!["short-tag"]
        );
        let vocabulary = storage.tag_vocabulary(Some("user")).unwrap();
        assert_eq!(vocabulary.skipped_overlong, 0, "the overlong tag is gone");
    }

    #[test]
    fn tag_vocabulary_rejects_more_than_ten_thousand_tags() {
        let storage = create_test_storage();
        let tags: Vec<String> = (0..10_001)
            .map(|index| format!("bulk-{index:05}"))
            .collect();
        let now = Utc::now().to_rfc3339();
        {
            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope)
                     VALUES ('vocab-bound', 'vocabulary bound fixture', 'fact', ?1, ?1, ?1,
                             ?2, 'user')",
                    params![&now, serde_json::to_string(&tags).unwrap()],
                )
                .unwrap();
        }
        let error = storage
            .tag_vocabulary(Some("user"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("exceeds the 10000-tag"));
    }

    #[test]
    fn padded_source_tags_are_reachable_byte_exact() {
        let storage = create_test_storage();
        let padded = ingest_tagged_in_scope(&storage, "user", "padded tag fixture", &[" prix-six"]);
        let unpadded = ingest_tagged_in_scope(&storage, "user", "unpadded fixture", &["prix-six"]);

        // The padded stored variant is addressable exactly as stored; the
        // trimmed TARGET merges it into the canonical spelling.
        let sources = vec![" prix-six".to_string()];
        let preview = storage
            .preview_tag_mutation(&sources, "prix-six", Some("user"))
            .unwrap();
        assert_eq!(preview["affectedMemoryCount"], 1);
        storage
            .apply_tag_mutation(
                &sources,
                "prix-six",
                Some("user"),
                preview_token(&preview),
                "tag_rename",
                "collapse a whitespace-padded tag variant",
            )
            .unwrap();
        assert_eq!(
            storage.get_node(&padded.id).unwrap().unwrap().tags,
            vec!["prix-six"]
        );

        // A trimmed source still matches unpadded stored tags as before.
        let trimmed_sources = vec!["prix-six".to_string()];
        let trimmed_preview = storage
            .preview_tag_mutation(&trimmed_sources, "grand-prix", Some("user"))
            .unwrap();
        assert_eq!(trimmed_preview["affectedMemoryCount"], 2);
        storage
            .apply_tag_mutation(
                &trimmed_sources,
                "grand-prix",
                Some("user"),
                preview_token(&trimmed_preview),
                "tag_rename",
                "rename the canonical spelling",
            )
            .unwrap();
        assert_eq!(
            storage.get_node(&unpadded.id).unwrap().unwrap().tags,
            vec!["grand-prix"]
        );
    }

    #[test]
    fn secret_shaped_stored_tags_can_be_renamed_away() {
        let storage = create_test_storage();
        let credential = format!("ghp_{}", "b".repeat(36));
        let node = storage
            .ingest_with_secret_policy(
                IngestInput {
                    content: "explicit-allow credential tag fixture".to_string(),
                    tags: vec![credential.clone(), "keep".to_string()],
                    ..Default::default()
                },
                SecretPolicy::AllowExplicitly,
            )
            .unwrap();

        let sources = vec![credential.clone()];
        let preview = storage
            .preview_tag_mutation(&sources, "rotated-token-reference", Some("user"))
            .unwrap();
        assert_eq!(preview["affectedMemoryCount"], 1);
        storage
            .apply_tag_mutation(
                &sources,
                "rotated-token-reference",
                Some("user"),
                preview_token(&preview),
                "tag_rename",
                "replace a credential-shaped tag with a safe reference",
            )
            .unwrap();
        assert_eq!(
            storage.get_node(&node.id).unwrap().unwrap().tags,
            vec!["rotated-token-reference", "keep"]
        );
    }

    #[test]
    fn scoped_tag_operation_listing_includes_all_scopes_operations() {
        let storage = create_test_storage();
        ingest_tagged_in_scope(&storage, "user", "scoped audit row", &["scoped-old"]);
        ingest_tagged_in_scope(&storage, "user", "shared audit row", &["shared-old"]);
        ingest_tagged_in_scope(&storage, "project-b", "other scope row", &["shared-old"]);

        let scoped_sources = vec!["scoped-old".to_string()];
        let scoped_preview = storage
            .preview_tag_mutation(&scoped_sources, "scoped-new", Some("user"))
            .unwrap();
        let scoped_op = storage
            .apply_tag_mutation(
                &scoped_sources,
                "scoped-new",
                Some("user"),
                preview_token(&scoped_preview),
                "tag_rename",
                "scoped audit fixture",
            )
            .unwrap();

        let shared_sources = vec!["shared-old".to_string()];
        let shared_preview = storage
            .preview_tag_mutation(&shared_sources, "shared-new", None)
            .unwrap();
        let shared_op = storage
            .apply_tag_mutation(
                &shared_sources,
                "shared-new",
                None,
                preview_token(&shared_preview),
                "tag_rename",
                "all-scopes audit fixture",
            )
            .unwrap();

        let user_view = storage.list_tag_operations(50, Some("user")).unwrap();
        let user_ids: Vec<&str> = user_view.iter().map(|op| op.id.as_str()).collect();
        assert!(
            user_ids.contains(&scoped_op.id.as_str()) && user_ids.contains(&shared_op.id.as_str()),
            "a scope's audit must show all-scopes operations that rewrote it"
        );

        let other_view = storage.list_tag_operations(50, Some("project-b")).unwrap();
        assert_eq!(other_view.len(), 1);
        assert_eq!(other_view[0].id, shared_op.id);

        let all_view = storage.list_tag_operations(50, None).unwrap();
        assert_eq!(all_view.len(), 2);
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn non_256_active_profile_builds_matching_dimension_index_without_truncation() {
        let storage = create_test_storage();
        let qwen = BuiltinEmbeddingProfile::QwenBalanced1024
            .profile()
            .profile_id;
        storage
            .save_embedding_profile_manifest(&ready_profile_manifest(
                BuiltinEmbeddingProfile::QwenBalanced1024,
            ))
            .unwrap();
        let node = storage
            .ingest(IngestInput {
                content: "dimension isolation".to_string(),
                ..Default::default()
            })
            .unwrap();
        storage
            .put_embedding_profile_vector(&EmbeddingProfileVector {
                profile_id: qwen.to_string(),
                node_id: node.id,
                embedding: Embedding::new(vec![0.25; 1024]).to_bytes(),
                dimensions: 1024,
                model: "Qwen/Qwen3-Embedding-0.6B".to_string(),
                created_at: Utc::now(),
            })
            .unwrap();
        {
            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE embedding_profile_state SET active_profile_id = ?1 WHERE singleton = 1",
                    params![qwen.as_str()],
                )
                .unwrap();
        }
        storage.load_embeddings_into_index().unwrap();
        assert_eq!(
            storage
                .vector_index
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .dimensions(),
            1024
        );
    }

    #[test]
    fn test_parse_timestamp_accepts_rfc3339_and_sqlite_native() {
        use chrono::TimeZone;

        // Canonical writer: RFC 3339 with fractional seconds + offset.
        let rfc =
            Storage::parse_timestamp("2026-06-12T15:07:59.730+00:00", "last_accessed").unwrap();
        assert_eq!(rfc.to_rfc3339(), "2026-06-12T15:07:59.730+00:00");

        // External writer: SQLite-native `datetime('now')` (space separator,
        // no timezone, no fraction) — must be tolerated, assumed UTC.
        let sqlite = Storage::parse_timestamp("2026-06-12 15:07:59", "last_accessed").unwrap();
        assert_eq!(
            sqlite,
            Utc.with_ymd_and_hms(2026, 6, 12, 15, 7, 59).unwrap()
        );

        // SQLite-native with fractional seconds.
        let sqlite_frac =
            Storage::parse_timestamp("2026-06-12 15:07:59.730", "last_accessed").unwrap();
        assert_eq!(sqlite_frac.timestamp_subsec_millis(), 730);

        // Genuinely malformed input still errors.
        assert!(Storage::parse_timestamp("not-a-timestamp", "last_accessed").is_err());
    }

    #[test]
    fn test_ingest_and_get() {
        let storage = create_test_storage();

        let input = IngestInput {
            content: "Test memory content".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        };

        let node = storage.ingest(input).unwrap();
        assert!(!node.id.is_empty());
        assert_eq!(node.content, "Test memory content");

        let retrieved = storage.get_node(&node.id).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Test memory content");
    }

    #[test]
    fn test_search() {
        let storage = create_test_storage();

        let input = IngestInput {
            content: "The mitochondria is the powerhouse of the cell".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        };

        storage.ingest(input).unwrap();

        let results = storage.search("mitochondria", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("mitochondria"));
    }

    #[test]
    fn passive_recall_records_telemetry_without_reinforcing() {
        let storage = create_test_storage();
        let node = storage
            .ingest(IngestInput {
                content: "current version is 2.0.12 as of 2026-03-04".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let before = storage.get_node(&node.id).unwrap().unwrap();

        {
            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET activation = 0.73 WHERE id = ?1",
                    params![&node.id],
                )
                .unwrap();
        }

        for _ in 0..3 {
            let recalled = storage
                .recall(RecallInput {
                    query: "current version".to_string(),
                    limit: 10,
                    min_retention: 0.0,
                    search_mode: SearchMode::Keyword,
                    valid_at: None,
                })
                .unwrap();
            assert_eq!(recalled.len(), 1);
        }

        let after = storage.get_node(&node.id).unwrap().unwrap();
        assert_eq!(after.retrieval_strength, before.retrieval_strength);
        assert_eq!(after.retention_strength, before.retention_strength);
        assert_eq!(after.stability, before.stability);
        assert_eq!(after.last_accessed, before.last_accessed);
        assert_eq!(after.reps, before.reps);
        assert_eq!(after.next_review, before.next_review);
        assert_eq!(after.times_retrieved, before.times_retrieved);
        assert_eq!(after.times_useful, before.times_useful);
        assert_eq!(after.utility_score, before.utility_score);
        assert_eq!(storage.auto_promote_frequent_access().unwrap(), 0);
        storage.compute_act_r_activations().unwrap();

        let reader = storage.reader.lock().unwrap();
        let retrievals_shown: i64 = reader
            .query_row(
                "SELECT COUNT(*) FROM memory_access_log WHERE node_id = ?1 AND access_type = 'retrieval_shown'",
                params![&node.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(retrievals_shown, 3);
        let activation: f64 = reader
            .query_row(
                "SELECT activation FROM knowledge_nodes WHERE id = ?1",
                params![&node.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(activation, 0.0);
    }

    #[test]
    fn explicit_promotion_marks_a_retrieved_memory_useful() {
        let storage = create_test_storage();
        let node = storage
            .ingest(IngestInput {
                content: "A memory that needs explicit positive feedback".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        storage.demote_memory(&node.id).unwrap();
        let before = storage.get_node(&node.id).unwrap().unwrap();

        storage.record_batch_retrieval(&[&node.id]).unwrap();
        let promoted = storage.promote_memory(&node.id).unwrap();

        assert!(promoted.retrieval_strength > before.retrieval_strength);
        assert!(promoted.retention_strength > before.retention_strength);
        assert_eq!(promoted.times_retrieved.unwrap_or_default(), 0);
        assert_eq!(promoted.times_useful.unwrap_or_default(), 1);
        assert_eq!(promoted.utility_score.unwrap_or_default(), 1.0);
    }

    #[test]
    fn legacy_search_hits_cannot_reinforce_or_reactivate_a_memory() {
        let storage = create_test_storage();
        let node = storage
            .ingest(IngestInput {
                content: "legacy dated current version claim".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();

        {
            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes
                     SET retrieval_strength = 0.50, retention_strength = 0.40, activation = 0.73
                     WHERE id = ?1",
                    params![&node.id],
                )
                .unwrap();
            for _ in 0..3 {
                writer
                    .execute(
                        "INSERT INTO memory_access_log (node_id, access_type, accessed_at)
                         VALUES (?1, 'search_hit', ?2)",
                        params![&node.id, Utc::now().to_rfc3339()],
                    )
                    .unwrap();
            }
        }
        let before = storage.get_node(&node.id).unwrap().unwrap();

        assert_eq!(storage.auto_promote_frequent_access().unwrap(), 0);
        storage.compute_act_r_activations().unwrap();

        let after = storage.get_node(&node.id).unwrap().unwrap();
        assert_eq!(after.retrieval_strength, before.retrieval_strength);
        assert_eq!(after.retention_strength, before.retention_strength);
        assert_eq!(after.last_accessed, before.last_accessed);

        let reader = storage.reader.lock().unwrap();
        let activation: f64 = reader
            .query_row(
                "SELECT activation FROM knowledge_nodes WHERE id = ?1",
                params![&node.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(activation, 0.0);
    }

    #[test]
    fn legacy_search_hits_cannot_preserve_recency_or_delay_decay() {
        let storage = create_test_storage();
        let node = storage
            .ingest(IngestInput {
                content: "a legacy passive search must not preserve freshness".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let created_at = Utc::now() - Duration::days(30);
        let passive_at = Utc::now();

        {
            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET
                        created_at = ?1,
                        updated_at = ?1,
                        last_accessed = ?2,
                        retrieval_strength = 1.0,
                        retention_strength = 1.0
                     WHERE id = ?3",
                    params![created_at.to_rfc3339(), passive_at.to_rfc3339(), &node.id],
                )
                .unwrap();
            writer
                .execute(
                    "INSERT INTO memory_access_log (node_id, access_type, accessed_at)
                     VALUES (?1, 'search_hit', ?2)",
                    params![&node.id, passive_at.to_rfc3339()],
                )
                .unwrap();
        }

        storage.compute_act_r_activations().unwrap();
        let repaired = storage.get_node(&node.id).unwrap().unwrap();
        assert_eq!(repaired.last_accessed, created_at);

        storage.apply_decay().unwrap();
        let decayed = storage.get_node(&node.id).unwrap().unwrap();
        assert!(decayed.retrieval_strength < 1.0);
        assert!(decayed.retention_strength < 1.0);
    }

    #[test]
    fn legacy_recency_repair_preserves_reviewed_memories() {
        let storage = create_test_storage();
        let node = storage
            .ingest(IngestInput {
                content: "a reviewed memory must not be reset by passive logs".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        storage.promote_memory(&node.id).unwrap();
        let reviewed = storage.mark_reviewed(&node.id, Rating::Good).unwrap();

        // New telemetry never changed state, so it must never trigger a
        // legacy-state repair.
        storage.record_batch_retrieval(&[&node.id]).unwrap();
        storage.compute_act_r_activations().unwrap();
        let after_new_telemetry = storage.get_node(&node.id).unwrap().unwrap();
        assert_eq!(after_new_telemetry.last_accessed, reviewed.last_accessed);

        // If an old search-hit row was recorded after a review, restore the
        // review timestamp from updated_at rather than falling back to create.
        {
            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET last_accessed = ?1 WHERE id = ?2",
                    params![Utc::now().to_rfc3339(), &node.id],
                )
                .unwrap();
            writer
                .execute(
                    "INSERT INTO memory_access_log (node_id, access_type, accessed_at)
                     VALUES (?1, 'search_hit', ?2)",
                    params![&node.id, Utc::now().to_rfc3339()],
                )
                .unwrap();
        }
        storage.compute_act_r_activations().unwrap();
        let restored = storage.get_node(&node.id).unwrap().unwrap();
        assert_eq!(restored.last_accessed, reviewed.last_accessed);
    }

    #[test]
    fn test_review() {
        let storage = create_test_storage();

        let input = IngestInput {
            content: "Test review".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        };

        let node = storage.ingest(input).unwrap();
        assert_eq!(node.reps, 0);

        let reviewed = storage.mark_reviewed(&node.id, Rating::Good).unwrap();
        assert_eq!(reviewed.reps, 1);
    }

    #[test]
    fn test_delete() {
        let storage = create_test_storage();

        let input = IngestInput {
            content: "To be deleted".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["sensitive-delete-tag".to_string()],
            ..Default::default()
        };

        let node = storage.ingest(input).unwrap();
        assert!(storage.get_node(&node.id).unwrap().is_some());

        let deleted = storage.delete_node(&node.id).unwrap();
        assert!(deleted);
        assert!(storage.get_node(&node.id).unwrap().is_none());
        let archive = serde_json::to_string(&storage.export_portable_archive().unwrap()).unwrap();
        assert!(!archive.contains(&node.id));
        assert!(!archive.contains("sensitive-delete-tag"));
    }

    /// REGRESSION (v2.6.0 data-safety): consolidation must never delete a
    /// memory. Until this release, an autonomic "retention target" GC inside
    /// run_consolidation hard-deleted everything below 0.3 retention older
    /// than 30 days — dormant only while decay was broken, and it destroyed
    /// 23 real memories from a live store the day decay was fixed. This test
    /// constructs exactly that scenario and asserts nothing dies.
    #[test]
    fn consolidation_never_deletes_low_retention_memories() {
        let storage = create_test_storage();
        let mut ids = Vec::new();
        for i in 0..4 {
            let node = storage
                .ingest(IngestInput {
                    content: format!("Old low-retention memory number {i} that must survive"),
                    node_type: "fact".to_string(),
                    ..Default::default()
                })
                .unwrap();
            ids.push(node.id);
        }
        // Force the doomed profile the old reaper keyed on: retention far
        // below 0.3 and created_at far older than 30 days.
        {
            let writer = storage.writer.lock().unwrap();
            let old = (Utc::now() - Duration::days(120)).to_rfc3339();
            for id in &ids {
                writer
                    .execute(
                        "UPDATE knowledge_nodes
                         SET retention_strength = 0.05, created_at = ?1, last_accessed = ?1
                         WHERE id = ?2",
                        params![old, id],
                    )
                    .unwrap();
            }
        }
        let before: i64 = {
            let reader = storage.reader.lock().unwrap();
            reader
                .query_row("SELECT COUNT(*) FROM knowledge_nodes", [], |r| r.get(0))
                .unwrap()
        };

        storage.run_consolidation().unwrap();

        let after: i64 = {
            let reader = storage.reader.lock().unwrap();
            reader
                .query_row("SELECT COUNT(*) FROM knowledge_nodes", [], |r| r.get(0))
                .unwrap()
        };
        assert_eq!(before, after, "consolidation deleted memories");
        for id in &ids {
            assert!(
                storage.get_node(id).unwrap().is_some(),
                "low-retention memory {id} was reaped by consolidation"
            );
        }
    }

    /// The explicit GC path must never collect a protected (pinned) memory,
    /// no matter how decayed it is. A pin says "keep this"; low retention
    /// only says "rarely retrieved".
    #[test]
    fn gc_spares_protected_memories() {
        let storage = create_test_storage();
        let pinned = storage
            .ingest(IngestInput {
                content: "Pinned but heavily decayed memory".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let doomed = storage
            .ingest(IngestInput {
                content: "Unpinned decayed memory".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        {
            let writer = storage.writer.lock().unwrap();
            let old = (Utc::now() - Duration::days(120)).to_rfc3339();
            for id in [&pinned.id, &doomed.id] {
                writer
                    .execute(
                        "UPDATE knowledge_nodes
                         SET retention_strength = 0.05, created_at = ?1 WHERE id = ?2",
                        params![old, id],
                    )
                    .unwrap();
            }
        }
        storage.set_protected(&pinned.id, true).unwrap();

        let deleted = storage.gc_below_retention(0.3, 30).unwrap();
        assert_eq!(deleted, 1, "only the unpinned memory is collected");
        assert!(storage.get_node(&pinned.id).unwrap().is_some(), "pin survives GC");
        assert!(storage.get_node(&doomed.id).unwrap().is_none());
    }

    #[test]
    fn gc_uses_the_privacy_cleanup_deletion_path() {
        let storage = create_test_storage();
        let node = storage
            .ingest(IngestInput {
                content: "GC deletion privacy target".to_string(),
                node_type: "fact".to_string(),
                tags: vec!["gc-sensitive-tag".to_string()],
                ..Default::default()
            })
            .unwrap();
        {
            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes
                     SET retention_strength = 0.0, created_at = '2000-01-01T00:00:00Z'
                     WHERE id = ?1",
                    params![&node.id],
                )
                .unwrap();
        }

        assert_eq!(storage.gc_below_retention(0.1, 1).unwrap(), 1);
        let archive = serde_json::to_string(&storage.export_portable_archive().unwrap()).unwrap();
        assert!(!archive.contains(&node.id));
        assert!(!archive.contains("gc-sensitive-tag"));
    }

    #[test]
    fn purging_empty_content_does_not_scrub_unrelated_evidence() {
        let storage = create_test_storage();
        let empty = storage
            .ingest(IngestInput {
                content: String::new(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        {
            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "INSERT INTO memory_prs (
                        id, kind, status, title, diff, signals, created_at
                     ) VALUES ('unrelated-review', 'new_fact', 'pending',
                               'keep this review', '{}', '[]', ?1)",
                    params![Utc::now().to_rfc3339()],
                )
                .unwrap();
        }

        assert!(storage.purge_node(&empty.id, None).unwrap().deleted);
        let writer = storage.writer.lock().unwrap();
        let remaining_reviews: i64 = writer
            .query_row(
                "SELECT COUNT(*) FROM memory_prs WHERE id = 'unrelated-review'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(remaining_reviews, 1);
    }

    #[test]
    fn test_composition_save_query_outcome_and_never_composed() {
        let storage = create_test_storage();
        let first = storage
            .ingest(IngestInput {
                content: "Oracle drift can break delayed settlement.".to_string(),
                node_type: "fact".to_string(),
                tags: vec![
                    "protocolgate".to_string(),
                    "boundary-oracle".to_string(),
                    "settlement".to_string(),
                ],
                ..Default::default()
            })
            .unwrap();
        let second = storage
            .ingest(IngestInput {
                content: "Withdrawal queues can settle stale claims.".to_string(),
                node_type: "pattern".to_string(),
                tags: vec![
                    "protocolgate".to_string(),
                    "boundary-queue".to_string(),
                    "settlement".to_string(),
                ],
                ..Default::default()
            })
            .unwrap();
        let third = storage
            .ingest(IngestInput {
                content: "Keeper roles can drift from local validation paths.".to_string(),
                node_type: "pattern".to_string(),
                tags: vec![
                    "protocolgate".to_string(),
                    "boundary-role".to_string(),
                    "settlement".to_string(),
                ],
                ..Default::default()
            })
            .unwrap();

        let before = storage
            .get_never_composed_candidates(10, Some(&["protocolgate".to_string()]))
            .unwrap();
        let first_second_before = before
            .iter()
            .find(|candidate| {
                let pair = Storage::pair_key(&candidate.first_id, &candidate.second_id);
                pair == Storage::pair_key(&first.id, &second.id)
            })
            .expect("uncomposed first/second pair should be ranked before any event");
        assert!(
            first_second_before.bridge_score > 0.0,
            "candidate should expose a bridge score"
        );
        assert!(
            first_second_before.novelty_score > 0.0,
            "candidate should expose a novelty score"
        );
        assert_eq!(
            first_second_before.outcome_signal, "clean",
            "new candidate should start without prior outcome context"
        );
        assert!(
            first_second_before
                .composition_question
                .contains("composed through"),
            "candidate should include a promptable composition question"
        );

        let event = CompositionEventRecord {
            id: "composition-test-1".to_string(),
            created_at: Utc::now(),
            tool: "deep_reference".to_string(),
            mode: "bounty".to_string(),
            query: Some("oracle drift delayed settlement".to_string()),
            query_hash: Some("sha256:test".to_string()),
            confidence: Some(0.87),
            status: Some("resolved".to_string()),
            output_preview: Some("Compose oracle drift with withdrawal queue.".to_string()),
            metadata: serde_json::json!({"workflow": "test"}),
        };
        let members = vec![
            CompositionMemberRecord {
                event_id: event.id.clone(),
                memory_id: first.id.clone(),
                role: "primary".to_string(),
                rank: 0,
                trust: Some(0.8),
                score: Some(0.9),
                preview: Some(preview(&first.content, 120)),
                metadata: serde_json::json!({}),
            },
            CompositionMemberRecord {
                event_id: event.id.clone(),
                memory_id: second.id.clone(),
                role: "supporting".to_string(),
                rank: 1,
                trust: Some(0.7),
                score: Some(0.75),
                preview: Some(preview(&second.content, 120)),
                metadata: serde_json::json!({}),
            },
        ];
        storage.save_composition(&event, &members, &[]).unwrap();

        let outcome = CompositionOutcomeRecord {
            id: "composition-outcome-1".to_string(),
            event_id: event.id.clone(),
            outcome_type: "submitted".to_string(),
            labeled_at: Utc::now(),
            label_source: "test".to_string(),
            confidence_delta: Some(0.1),
            notes: Some("Report submitted".to_string()),
            metadata: serde_json::json!({"severity": "high"}),
        };
        storage.record_composition_outcome(&outcome).unwrap();

        let fetched = storage.get_composition_event(&event.id).unwrap().unwrap();
        assert_eq!(fetched.mode, "bounty");
        assert_eq!(fetched.metadata["workflow"], "test");

        let fetched_members = storage.get_composition_members(&event.id).unwrap();
        assert_eq!(fetched_members.len(), 2);
        assert_eq!(fetched_members[0].role, "primary");

        let fetched_outcomes = storage.get_composition_outcomes(&event.id).unwrap();
        assert_eq!(fetched_outcomes.len(), 1);
        assert_eq!(fetched_outcomes[0].outcome_type, "submitted");

        let for_memory = storage.get_compositions_for_memory(&first.id, 5).unwrap();
        assert_eq!(for_memory.len(), 1);
        assert_eq!(for_memory[0].id, event.id);

        let neighbors = storage.get_composition_neighbors(&first.id, 5).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].memory_id, second.id);

        let after = storage
            .get_never_composed_candidates(10, Some(&["protocolgate".to_string()]))
            .unwrap();
        assert!(
            !after.iter().any(|candidate| {
                let pair = Storage::pair_key(&candidate.first_id, &candidate.second_id);
                pair == Storage::pair_key(&first.id, &second.id)
            }),
            "already-composed first/second pair should be removed"
        );
        assert!(
            after.iter().any(|candidate| {
                let pair = Storage::pair_key(&candidate.first_id, &candidate.second_id);
                pair == Storage::pair_key(&first.id, &third.id)
                    || pair == Storage::pair_key(&second.id, &third.id)
            }),
            "other protocolgate pairs should remain candidates"
        );
    }

    #[test]
    fn test_composition_neighbors_count_distinct_events_not_member_roles() {
        let storage = create_test_storage();
        let first = storage
            .ingest(IngestInput {
                content: "Oracle role appears once in the event.".to_string(),
                node_type: "fact".to_string(),
                tags: vec!["protocolgate".to_string(), "settlement".to_string()],
                ..Default::default()
            })
            .unwrap();
        let second = storage
            .ingest(IngestInput {
                content: "Queue role appears under two evidence roles.".to_string(),
                node_type: "fact".to_string(),
                tags: vec!["protocolgate".to_string(), "settlement".to_string()],
                ..Default::default()
            })
            .unwrap();

        storage
            .save_composition(
                &CompositionEventRecord {
                    id: "multi-role-neighbor-event".to_string(),
                    created_at: Utc::now(),
                    tool: "deep_reference".to_string(),
                    mode: "bounty".to_string(),
                    query: Some("multi role neighbor".to_string()),
                    query_hash: Some("fnv1a64:neighbor".to_string()),
                    confidence: Some(0.7),
                    status: Some("resolved".to_string()),
                    output_preview: None,
                    metadata: serde_json::json!({}),
                },
                &[
                    CompositionMemberRecord {
                        event_id: "multi-role-neighbor-event".to_string(),
                        memory_id: first.id.clone(),
                        role: "primary".to_string(),
                        rank: 0,
                        trust: Some(0.8),
                        score: Some(0.9),
                        preview: None,
                        metadata: serde_json::json!({}),
                    },
                    CompositionMemberRecord {
                        event_id: "multi-role-neighbor-event".to_string(),
                        memory_id: second.id.clone(),
                        role: "supporting".to_string(),
                        rank: 1,
                        trust: Some(0.7),
                        score: Some(0.8),
                        preview: None,
                        metadata: serde_json::json!({}),
                    },
                    CompositionMemberRecord {
                        event_id: "multi-role-neighbor-event".to_string(),
                        memory_id: second.id.clone(),
                        role: "related".to_string(),
                        rank: 2,
                        trust: Some(0.7),
                        score: Some(0.6),
                        preview: None,
                        metadata: serde_json::json!({}),
                    },
                ],
                &[],
            )
            .unwrap();

        let neighbors = storage.get_composition_neighbors(&first.id, 10).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].memory_id, second.id);
        assert_eq!(
            neighbors[0].composed_count, 1,
            "one event with multiple member roles should count as one composition"
        );
    }

    #[test]
    fn test_never_composed_tag_filter_includes_older_tagged_candidates() {
        let storage = create_test_storage();
        let first = storage
            .ingest(IngestInput {
                content: "Older Vestige composition frontier about outcome-shaped recall."
                    .to_string(),
                node_type: "fact".to_string(),
                tags: vec!["project:vestige".to_string(), "composition".to_string()],
                ..Default::default()
            })
            .unwrap();
        let second = storage
            .ingest(IngestInput {
                content: "Older Vestige composition frontier about never-composed recall."
                    .to_string(),
                node_type: "pattern".to_string(),
                tags: vec!["project:vestige".to_string(), "composition".to_string()],
                ..Default::default()
            })
            .unwrap();

        for idx in 0..751 {
            storage
                .ingest(IngestInput {
                    content: format!("Unrelated recent memory {idx} for scan-window pressure."),
                    node_type: "fact".to_string(),
                    tags: vec!["unrelated".to_string()],
                    ..Default::default()
                })
                .unwrap();
        }

        let candidates = storage
            .get_never_composed_candidates(10, Some(&["project".to_string()]))
            .unwrap();
        assert!(
            candidates.iter().any(|candidate| {
                let pair = Storage::pair_key(&candidate.first_id, &candidate.second_id);
                pair == Storage::pair_key(&first.id, &second.id)
            }),
            "tag-filtered frontier should include older namespaced-tag memories outside the base scan window"
        );
    }

    #[test]
    fn test_never_composed_carries_prior_outcome_signal() {
        let storage = create_test_storage();
        let first = storage
            .ingest(IngestInput {
                content: "Oracle drift lane previously looked duplicate-prone.".to_string(),
                node_type: "fact".to_string(),
                tags: vec![
                    "protocolgate".to_string(),
                    "boundary-oracle".to_string(),
                    "settlement".to_string(),
                ],
                ..Default::default()
            })
            .unwrap();
        let second = storage
            .ingest(IngestInput {
                content: "Withdrawal queue lane had weak proof.".to_string(),
                node_type: "fact".to_string(),
                tags: vec![
                    "protocolgate".to_string(),
                    "boundary-queue".to_string(),
                    "settlement".to_string(),
                ],
                ..Default::default()
            })
            .unwrap();
        let third = storage
            .ingest(IngestInput {
                content: "Keeper settlement lane has not been composed with oracle drift."
                    .to_string(),
                node_type: "pattern".to_string(),
                tags: vec![
                    "protocolgate".to_string(),
                    "boundary-role".to_string(),
                    "settlement".to_string(),
                ],
                ..Default::default()
            })
            .unwrap();

        let event = CompositionEventRecord {
            id: "prior-outcome-composition".to_string(),
            created_at: Utc::now(),
            tool: "deep_reference".to_string(),
            mode: "bounty".to_string(),
            query: Some("oracle withdrawal duplicate risk".to_string()),
            query_hash: Some("fnv1a64:prior".to_string()),
            confidence: Some(0.4),
            status: Some("closed".to_string()),
            output_preview: Some("Prior composition was labeled duplicate risk.".to_string()),
            metadata: serde_json::json!({}),
        };
        storage
            .save_composition(
                &event,
                &[
                    CompositionMemberRecord {
                        event_id: event.id.clone(),
                        memory_id: first.id.clone(),
                        role: "primary".to_string(),
                        rank: 0,
                        trust: Some(0.7),
                        score: Some(0.8),
                        preview: None,
                        metadata: serde_json::json!({}),
                    },
                    CompositionMemberRecord {
                        event_id: event.id.clone(),
                        memory_id: second.id.clone(),
                        role: "supporting".to_string(),
                        rank: 1,
                        trust: Some(0.7),
                        score: Some(0.8),
                        preview: None,
                        metadata: serde_json::json!({}),
                    },
                ],
                &[CompositionOutcomeRecord {
                    id: "prior-outcome-label".to_string(),
                    event_id: event.id.clone(),
                    outcome_type: "duplicate_risk".to_string(),
                    labeled_at: Utc::now(),
                    label_source: "test".to_string(),
                    confidence_delta: Some(-0.2),
                    notes: Some("Duplicate family in prior lane.".to_string()),
                    metadata: serde_json::json!({}),
                }],
            )
            .unwrap();

        let candidates = storage
            .get_never_composed_candidates(10, Some(&["protocolgate".to_string()]))
            .unwrap();
        let candidate = candidates
            .iter()
            .find(|candidate| {
                let pair = Storage::pair_key(&candidate.first_id, &candidate.second_id);
                pair == Storage::pair_key(&first.id, &third.id)
            })
            .expect("untried first/third pair should remain a frontier candidate");

        assert!(
            candidate
                .prior_outcomes
                .iter()
                .any(|outcome| outcome == "duplicate_risk"),
            "frontier candidate should expose prior outcome labels from either member"
        );
        assert_eq!(candidate.outcome_signal, "prior_duplicate_risk");
        assert!(
            candidate.outcome_score_adjustment < 0.0,
            "duplicate-risk history should reduce but not hide the untried lane"
        );
    }

    #[test]
    fn test_never_composed_marks_mixed_prior_outcomes() {
        let storage = create_test_storage();
        let successful = storage
            .ingest(IngestInput {
                content: "Accepted release lane linked rollback evidence to install telemetry."
                    .to_string(),
                node_type: "decision".to_string(),
                tags: vec![
                    "project:vestige".to_string(),
                    "release".to_string(),
                    "telemetry".to_string(),
                ],
                ..Default::default()
            })
            .unwrap();
        let closed = storage
            .ingest(IngestInput {
                content: "Closed release lane linked install telemetry to out-of-scope claims."
                    .to_string(),
                node_type: "incident".to_string(),
                tags: vec![
                    "project:vestige".to_string(),
                    "release".to_string(),
                    "telemetry".to_string(),
                ],
                ..Default::default()
            })
            .unwrap();
        let success_helper = storage
            .ingest(IngestInput {
                content: "Helper memory for an accepted release composition.".to_string(),
                node_type: "fact".to_string(),
                tags: vec!["project:vestige".to_string(), "release".to_string()],
                ..Default::default()
            })
            .unwrap();
        let closed_helper = storage
            .ingest(IngestInput {
                content: "Helper memory for a closed release composition.".to_string(),
                node_type: "fact".to_string(),
                tags: vec!["project:vestige".to_string(), "release".to_string()],
                ..Default::default()
            })
            .unwrap();

        storage
            .save_composition(
                &CompositionEventRecord {
                    id: "prior-success-composition".to_string(),
                    created_at: Utc::now(),
                    tool: "deep_reference".to_string(),
                    mode: "release".to_string(),
                    query: Some("accepted release lane".to_string()),
                    query_hash: Some("fnv1a64:success".to_string()),
                    confidence: Some(0.9),
                    status: Some("resolved".to_string()),
                    output_preview: None,
                    metadata: serde_json::json!({}),
                },
                &[
                    CompositionMemberRecord {
                        event_id: "prior-success-composition".to_string(),
                        memory_id: successful.id.clone(),
                        role: "primary".to_string(),
                        rank: 0,
                        trust: Some(0.9),
                        score: Some(0.9),
                        preview: None,
                        metadata: serde_json::json!({}),
                    },
                    CompositionMemberRecord {
                        event_id: "prior-success-composition".to_string(),
                        memory_id: success_helper.id,
                        role: "supporting".to_string(),
                        rank: 1,
                        trust: Some(0.7),
                        score: Some(0.6),
                        preview: None,
                        metadata: serde_json::json!({}),
                    },
                ],
                &[CompositionOutcomeRecord {
                    id: "prior-success-label".to_string(),
                    event_id: "prior-success-composition".to_string(),
                    outcome_type: "accepted".to_string(),
                    labeled_at: Utc::now(),
                    label_source: "test".to_string(),
                    confidence_delta: Some(0.2),
                    notes: None,
                    metadata: serde_json::json!({}),
                }],
            )
            .unwrap();

        storage
            .save_composition(
                &CompositionEventRecord {
                    id: "prior-closed-composition".to_string(),
                    created_at: Utc::now(),
                    tool: "deep_reference".to_string(),
                    mode: "release".to_string(),
                    query: Some("closed release lane".to_string()),
                    query_hash: Some("fnv1a64:closed".to_string()),
                    confidence: Some(0.3),
                    status: Some("closed".to_string()),
                    output_preview: None,
                    metadata: serde_json::json!({}),
                },
                &[
                    CompositionMemberRecord {
                        event_id: "prior-closed-composition".to_string(),
                        memory_id: closed.id.clone(),
                        role: "primary".to_string(),
                        rank: 0,
                        trust: Some(0.8),
                        score: Some(0.7),
                        preview: None,
                        metadata: serde_json::json!({}),
                    },
                    CompositionMemberRecord {
                        event_id: "prior-closed-composition".to_string(),
                        memory_id: closed_helper.id,
                        role: "supporting".to_string(),
                        rank: 1,
                        trust: Some(0.7),
                        score: Some(0.6),
                        preview: None,
                        metadata: serde_json::json!({}),
                    },
                ],
                &[CompositionOutcomeRecord {
                    id: "prior-closed-label".to_string(),
                    event_id: "prior-closed-composition".to_string(),
                    outcome_type: "closed_by_scope".to_string(),
                    labeled_at: Utc::now(),
                    label_source: "test".to_string(),
                    confidence_delta: Some(-0.3),
                    notes: None,
                    metadata: serde_json::json!({}),
                }],
            )
            .unwrap();

        let candidates = storage
            .get_never_composed_candidates(10, Some(&["project".to_string()]))
            .unwrap();
        let candidate = candidates
            .iter()
            .find(|candidate| {
                let pair = Storage::pair_key(&candidate.first_id, &candidate.second_id);
                pair == Storage::pair_key(&successful.id, &closed.id)
            })
            .expect("untried success/closed pair should remain a frontier candidate");

        assert_eq!(candidate.outcome_signal, "mixed_prior_outcomes");
        assert!(
            candidate
                .prior_outcomes
                .iter()
                .any(|outcome| outcome == "accepted")
        );
        assert!(
            candidate
                .prior_outcomes
                .iter()
                .any(|outcome| outcome == "closed_by_scope")
        );
    }

    #[test]
    fn test_never_composed_surfaces_weak_tie_shared_terms_without_shared_tags() {
        let storage = create_test_storage();
        let incident = storage
            .ingest(IngestInput {
                content:
                    "OpenCode handshake stalls when embedding startup blocks stdio negotiation."
                        .to_string(),
                node_type: "incident".to_string(),
                tags: vec!["opencode".to_string(), "startup".to_string()],
                ..Default::default()
            })
            .unwrap();
        let mitigation = storage
            .ingest(IngestInput {
                content: "JetBrains startup should keep embedding backfill behind the handshake."
                    .to_string(),
                node_type: "mitigation".to_string(),
                tags: vec!["jetbrains".to_string(), "background-work".to_string()],
                ..Default::default()
            })
            .unwrap();

        let candidates = storage.get_never_composed_candidates(10, None).unwrap();
        let candidate = candidates
            .iter()
            .find(|candidate| {
                let pair = Storage::pair_key(&candidate.first_id, &candidate.second_id);
                pair == Storage::pair_key(&incident.id, &mitigation.id)
            })
            .expect("shared terms should surface a weak-tie candidate without shared tags");

        assert!(
            candidate.shared_tags.is_empty(),
            "test fixture intentionally has no shared tags"
        );
        assert!(
            candidate
                .shared_terms
                .iter()
                .any(|term| term == "embedding" || term == "startup" || term == "handshake"),
            "shared terms should explain the candidate"
        );
        assert!(
            candidate.bridge_score > 0.5,
            "different tags and node types should create a bridge signal"
        );
    }

    #[test]
    fn test_dream_history_save_and_get_last() {
        let storage = create_test_storage();
        let now = Utc::now();

        let record = DreamHistoryRecord {
            dreamed_at: now,
            duration_ms: 1500,
            memories_replayed: 50,
            connections_found: 12,
            insights_generated: 3,
            memories_strengthened: 8,
            memories_compressed: 2,
            phase_nrem1_ms: None,
            phase_nrem3_ms: None,
            phase_rem_ms: None,
            phase_integration_ms: None,
            summaries_generated: None,
            emotional_memories_processed: None,
            creative_connections_found: None,
        };

        let id = storage.save_dream_history(&record).unwrap();
        assert!(id > 0);

        let last = storage.get_last_dream().unwrap();
        assert!(last.is_some());
        // Timestamps should be within 1 second (RFC3339 round-trip)
        let diff = (last.unwrap() - now).num_seconds().abs();
        assert!(diff <= 1, "Timestamp mismatch: diff={}s", diff);
    }

    #[test]
    fn test_dream_history_empty() {
        let storage = create_test_storage();
        let last = storage.get_last_dream().unwrap();
        assert!(last.is_none());
    }

    #[test]
    fn test_count_memories_since() {
        let storage = create_test_storage();
        let before = Utc::now() - Duration::seconds(10);

        for i in 0..5 {
            storage
                .ingest(IngestInput {
                    content: format!("Count test memory {}", i),
                    node_type: "fact".to_string(),
                    ..Default::default()
                })
                .unwrap();
        }

        let count = storage.count_memories_since(before).unwrap();
        assert_eq!(count, 5);

        let future = Utc::now() + Duration::hours(1);
        let count_future = storage.count_memories_since(future).unwrap();
        assert_eq!(count_future, 0);
    }

    #[test]
    fn test_portable_archive_exact_round_trip() {
        let source_dir = tempdir().unwrap();
        let target_dir = tempdir().unwrap();
        let source = create_test_storage_at(&source_dir, "source.db");

        let first = source
            .ingest(IngestInput {
                content: "Portable archive alpha memory".to_string(),
                node_type: "fact".to_string(),
                tags: vec!["portable".to_string()],
                source: Some("test".to_string()),
                ..Default::default()
            })
            .unwrap();
        let second = source
            .ingest(IngestInput {
                content: "Portable archive beta memory".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        source.mark_reviewed(&first.id, Rating::Good).unwrap();
        source
            .save_connection(&ConnectionRecord {
                source_id: first.id.clone(),
                target_id: second.id.clone(),
                strength: 0.75,
                link_type: "semantic".to_string(),
                created_at: Utc::now(),
                last_activated: Utc::now(),
                activation_count: 1,
            })
            .unwrap();
        source
            .save_composition(
                &CompositionEventRecord {
                    id: "portable-composition-1".to_string(),
                    created_at: Utc::now(),
                    tool: "deep_reference".to_string(),
                    mode: "bounty".to_string(),
                    query: Some("portable composition".to_string()),
                    query_hash: Some("sha256:portable".to_string()),
                    confidence: Some(0.9),
                    status: Some("resolved".to_string()),
                    output_preview: Some("Portable composition event".to_string()),
                    metadata: serde_json::json!({}),
                },
                &[
                    CompositionMemberRecord {
                        event_id: "portable-composition-1".to_string(),
                        memory_id: first.id.clone(),
                        role: "primary".to_string(),
                        rank: 0,
                        trust: Some(0.9),
                        score: Some(1.0),
                        preview: Some("alpha".to_string()),
                        metadata: serde_json::json!({}),
                    },
                    CompositionMemberRecord {
                        event_id: "portable-composition-1".to_string(),
                        memory_id: second.id.clone(),
                        role: "supporting".to_string(),
                        rank: 1,
                        trust: Some(0.8),
                        score: Some(0.8),
                        preview: Some("beta".to_string()),
                        metadata: serde_json::json!({}),
                    },
                ],
                &[CompositionOutcomeRecord {
                    id: "portable-composition-outcome-1".to_string(),
                    event_id: "portable-composition-1".to_string(),
                    outcome_type: "helpful".to_string(),
                    labeled_at: Utc::now(),
                    label_source: "test".to_string(),
                    confidence_delta: None,
                    notes: None,
                    metadata: serde_json::json!({}),
                }],
            )
            .unwrap();

        let archive = source.export_portable_archive().unwrap();
        assert_eq!(archive.archive_format, PORTABLE_ARCHIVE_FORMAT);
        assert!(archive.total_rows() >= 3);
        assert!(
            archive
                .tables
                .iter()
                .any(|table| table.name == "knowledge_nodes" && table.rows.len() == 2)
        );
        for table_name in [
            "composition_events",
            "composition_members",
            "composition_outcomes",
        ] {
            assert!(
                archive.tables.iter().any(|table| table.name == table_name),
                "{table_name} must be included in portable archive"
            );
        }

        let target = create_test_storage_at(&target_dir, "target.db");
        let report = target
            .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
            .unwrap();
        assert!(report.rows_imported >= 3);
        assert!(report.fts_rebuilt);

        let restored = target.get_node(&first.id).unwrap().unwrap();
        assert_eq!(restored.id, first.id);
        assert_eq!(restored.content, first.content);
        assert_eq!(restored.tags, first.tags);
        assert_eq!(restored.reps, 1);

        let connections = target.get_connections_for_memory(&first.id).unwrap();
        assert_eq!(connections.len(), 1);
        assert_eq!(connections[0].target_id, second.id);

        let composition = target
            .get_composition_event("portable-composition-1")
            .unwrap()
            .unwrap();
        assert_eq!(composition.mode, "bounty");
        assert_eq!(
            target
                .get_composition_members("portable-composition-1")
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            target
                .get_composition_outcomes("portable-composition-1")
                .unwrap()
                .len(),
            1
        );

        let results = target.search("alpha", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, first.id);
    }

    #[test]
    fn test_portable_import_rejects_non_empty_target() {
        let source_dir = tempdir().unwrap();
        let target_dir = tempdir().unwrap();
        let source = create_test_storage_at(&source_dir, "source.db");
        source
            .ingest(IngestInput {
                content: "Source memory".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let archive = source.export_portable_archive().unwrap();

        let target = create_test_storage_at(&target_dir, "target.db");
        target
            .ingest(IngestInput {
                content: "Existing target memory".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();

        let err = target
            .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("requires an empty target database")
        );
    }

    #[test]
    fn test_portable_import_rejects_unknown_mode() {
        let source_dir = tempdir().unwrap();
        let target_dir = tempdir().unwrap();
        let source = create_test_storage_at(&source_dir, "source.db");
        source
            .ingest(IngestInput {
                content: "Source memory".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let mut archive = source.export_portable_archive().unwrap();
        archive.mode = "merge".to_string();

        let target = create_test_storage_at(&target_dir, "target.db");
        let err = target
            .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("Unsupported portable archive mode")
        );
    }

    #[test]
    fn test_portable_import_rejects_malformed_table_list() {
        let source_dir = tempdir().unwrap();
        let target_dir = tempdir().unwrap();
        let source = create_test_storage_at(&source_dir, "source.db");
        source
            .ingest(IngestInput {
                content: "Source memory".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();

        let mut duplicate_archive = source.export_portable_archive().unwrap();
        let duplicate_table = duplicate_archive
            .tables
            .iter()
            .find(|table| table.name == "knowledge_nodes")
            .unwrap()
            .clone();
        duplicate_archive.tables.push(duplicate_table);

        let target = create_test_storage_at(&target_dir, "target-duplicate.db");
        let err = target
            .import_portable_archive(&duplicate_archive, PortableImportMode::EmptyOnly)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("Portable archive contains duplicate table")
        );

        let mut unknown_archive = source.export_portable_archive().unwrap();
        unknown_archive.tables.push(PortableTable {
            name: "sqlite_sequence".to_string(),
            columns: vec!["name".to_string(), "seq".to_string()],
            rows: vec![],
        });

        let target = create_test_storage_at(&target_dir, "target-unknown.db");
        let err = target
            .import_portable_archive(&unknown_archive, PortableImportMode::EmptyOnly)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("Portable archive contains unsupported table")
        );
    }

    #[test]
    fn test_portable_merge_import_combines_non_empty_databases() {
        let source_dir = tempdir().unwrap();
        let target_dir = tempdir().unwrap();
        let source = create_test_storage_at(&source_dir, "source.db");
        let target = create_test_storage_at(&target_dir, "target.db");

        let source_node = source
            .ingest(IngestInput {
                content: "Source sync memory".to_string(),
                node_type: "fact".to_string(),
                tags: vec!["sync".to_string()],
                ..Default::default()
            })
            .unwrap();
        let target_node = target
            .ingest(IngestInput {
                content: "Target local memory".to_string(),
                node_type: "fact".to_string(),
                tags: vec!["local".to_string()],
                ..Default::default()
            })
            .unwrap();

        let archive = source.export_portable_archive().unwrap();
        let report = target
            .import_portable_archive(&archive, PortableImportMode::Merge)
            .unwrap();

        assert!(report.rows_inserted > 0);
        assert!(target.get_node(&source_node.id).unwrap().is_some());
        assert!(target.get_node(&target_node.id).unwrap().is_some());
    }

    #[test]
    fn test_portable_merge_import_keeps_newer_local_memory() {
        let source_dir = tempdir().unwrap();
        let target_dir = tempdir().unwrap();
        let source = create_test_storage_at(&source_dir, "source.db");
        let target = create_test_storage_at(&target_dir, "target.db");

        let node = source
            .ingest(IngestInput {
                content: "Original shared memory".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let archive = source.export_portable_archive().unwrap();
        target
            .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
            .unwrap();

        let newer = (Utc::now() + Duration::hours(1)).to_rfc3339();
        {
            let writer = target.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET content = ?1, updated_at = ?2 WHERE id = ?3",
                    params!["Newer local edit", newer, &node.id],
                )
                .unwrap();
        }

        let report = target
            .import_portable_archive(&archive, PortableImportMode::Merge)
            .unwrap();

        assert!(report.conflicts_kept_local >= 1);
        let restored = target.get_node(&node.id).unwrap().unwrap();
        assert_eq!(restored.content, "Newer local edit");
    }

    #[test]
    fn test_portable_merge_import_keeps_children_for_newer_local_memory() {
        let source_dir = tempdir().unwrap();
        let target_dir = tempdir().unwrap();
        let source = create_test_storage_at(&source_dir, "source.db");
        let target = create_test_storage_at(&target_dir, "target.db");

        let node = source
            .ingest(IngestInput {
                content: "Shared parent with child rows".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();

        let source_time = Utc::now().to_rfc3339();
        {
            let writer = source.writer.lock().unwrap();
            writer
                .execute(
                    "INSERT OR REPLACE INTO node_embeddings
                     (node_id, embedding, dimensions, model, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![&node.id, vec![1_u8, 2, 3, 4], 4, "test-model", &source_time],
                )
                .unwrap();
            writer
                .execute(
                    "INSERT OR REPLACE INTO fsrs_cards
                     (memory_id, difficulty, stability, state, reps, lapses)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![&node.id, 3.0_f64, 2.0_f64, "review", 2_i64, 0_i64],
                )
                .unwrap();
            writer
                .execute(
                    "INSERT OR REPLACE INTO memory_states
                     (memory_id, state, last_access, access_count, state_entered_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![&node.id, "active", &source_time, 1_i64, &source_time],
                )
                .unwrap();
        }

        let archive = source.export_portable_archive().unwrap();
        target
            .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
            .unwrap();

        let local_time = (Utc::now() + Duration::hours(1)).to_rfc3339();
        {
            let writer = target.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET content = ?1, updated_at = ?2 WHERE id = ?3",
                    params!["Newer local parent edit", &local_time, &node.id],
                )
                .unwrap();
            writer
                .execute(
                    "INSERT OR REPLACE INTO node_embeddings
                     (node_id, embedding, dimensions, model, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![&node.id, vec![9_u8, 8, 7, 6], 4, "test-model", &local_time],
                )
                .unwrap();
            writer
                .execute(
                    "INSERT OR REPLACE INTO fsrs_cards
                     (memory_id, difficulty, stability, state, reps, lapses)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![&node.id, 9.0_f64, 8.0_f64, "review", 9_i64, 1_i64],
                )
                .unwrap();
            writer
                .execute(
                    "INSERT OR REPLACE INTO memory_states
                     (memory_id, state, last_access, access_count, state_entered_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![&node.id, "silent", &local_time, 42_i64, &local_time],
                )
                .unwrap();
        }

        let report = target
            .import_portable_archive(&archive, PortableImportMode::Merge)
            .unwrap();

        assert!(report.conflicts_kept_local >= 4);
        let restored = target.get_node(&node.id).unwrap().unwrap();
        assert_eq!(restored.content, "Newer local parent edit");

        let reader = target.reader.lock().unwrap();
        let embedding: Vec<u8> = reader
            .query_row(
                "SELECT embedding FROM node_embeddings WHERE node_id = ?1",
                params![&node.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(embedding, vec![9_u8, 8, 7, 6]);

        let difficulty: f64 = reader
            .query_row(
                "SELECT difficulty FROM fsrs_cards WHERE memory_id = ?1",
                params![&node.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(difficulty, 9.0);

        let (state, access_count): (String, i64) = reader
            .query_row(
                "SELECT state, access_count FROM memory_states WHERE memory_id = ?1",
                params![&node.id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(state, "silent");
        assert_eq!(access_count, 42);
    }

    #[test]
    fn test_portable_merge_import_keeps_composition_members_for_newer_local_memory() {
        let source_dir = tempdir().unwrap();
        let target_dir = tempdir().unwrap();
        let source = create_test_storage_at(&source_dir, "source.db");
        let target = create_test_storage_at(&target_dir, "target.db");

        let node = source
            .ingest(IngestInput {
                content: "Shared memory with historical composition".to_string(),
                node_type: "fact".to_string(),
                tags: vec!["protocolgate".to_string()],
                ..Default::default()
            })
            .unwrap();
        source
            .save_composition(
                &CompositionEventRecord {
                    id: "merge-composition-1".to_string(),
                    created_at: Utc::now(),
                    tool: "deep_reference".to_string(),
                    mode: "bounty".to_string(),
                    query: Some("historical composition".to_string()),
                    query_hash: Some("sha256:historical".to_string()),
                    confidence: Some(0.7),
                    status: Some("resolved".to_string()),
                    output_preview: Some("Historical composition survives merge".to_string()),
                    metadata: serde_json::json!({}),
                },
                &[CompositionMemberRecord {
                    event_id: "merge-composition-1".to_string(),
                    memory_id: node.id.clone(),
                    role: "primary".to_string(),
                    rank: 0,
                    trust: Some(0.8),
                    score: Some(0.9),
                    preview: Some("historical".to_string()),
                    metadata: serde_json::json!({}),
                }],
                &[],
            )
            .unwrap();

        let archive = source.export_portable_archive().unwrap();
        target
            .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
            .unwrap();

        let local_time = (Utc::now() + Duration::hours(1)).to_rfc3339();
        {
            let writer = target.writer.lock().unwrap();
            writer
                .execute(
                    "DELETE FROM composition_members WHERE event_id = ?1",
                    params!["merge-composition-1"],
                )
                .unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET content = ?1, updated_at = ?2 WHERE id = ?3",
                    params!["Newer local content", &local_time, &node.id],
                )
                .unwrap();
        }

        target
            .import_portable_archive(&archive, PortableImportMode::Merge)
            .unwrap();

        let restored = target.get_node(&node.id).unwrap().unwrap();
        assert_eq!(restored.content, "Newer local content");
        let members = target
            .get_composition_members("merge-composition-1")
            .unwrap();
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].memory_id, node.id);
    }

    #[test]
    fn test_portable_merge_import_applies_delete_tombstones() {
        let source_dir = tempdir().unwrap();
        let target_dir = tempdir().unwrap();
        let source = create_test_storage_at(&source_dir, "source.db");
        let target = create_test_storage_at(&target_dir, "target.db");

        let node = source
            .ingest(IngestInput {
                content: "Memory deleted on source".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let archive = source.export_portable_archive().unwrap();
        target
            .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
            .unwrap();
        assert!(target.get_node(&node.id).unwrap().is_some());

        source.delete_node(&node.id).unwrap();
        let delete_archive = source.export_portable_archive().unwrap();
        let report = target
            .import_portable_archive(&delete_archive, PortableImportMode::Merge)
            .unwrap();

        assert!(report.rows_deleted >= 1);
        assert!(target.get_node(&node.id).unwrap().is_none());
    }

    #[test]
    fn test_portable_merge_import_preserves_purge_tombstones() {
        let source_dir = tempdir().unwrap();
        let target_dir = tempdir().unwrap();
        let source = create_test_storage_at(&source_dir, "source.db");
        let target = create_test_storage_at(&target_dir, "target.db");

        let node = source
            .ingest(IngestInput {
                content: "Memory purged on source".to_string(),
                node_type: "fact".to_string(),
                tags: vec!["sync".to_string()],
                ..Default::default()
            })
            .unwrap();
        source
            .save_composition(
                &CompositionEventRecord {
                    id: "portable-purge-composition".to_string(),
                    created_at: Utc::now(),
                    tool: "deep_reference".to_string(),
                    mode: "sync".to_string(),
                    query: Some("portable purge preview".to_string()),
                    query_hash: Some("fnv1a64:portable-purge".to_string()),
                    confidence: Some(0.7),
                    status: Some("resolved".to_string()),
                    output_preview: None,
                    metadata: serde_json::json!({}),
                },
                &[CompositionMemberRecord {
                    event_id: "portable-purge-composition".to_string(),
                    memory_id: node.id.clone(),
                    role: "primary".to_string(),
                    rank: 0,
                    trust: Some(0.8),
                    score: Some(0.8),
                    preview: Some("Portable purge composition preview leak".to_string()),
                    metadata: serde_json::json!({}),
                }],
                &[],
            )
            .unwrap();
        let archive = source.export_portable_archive().unwrap();
        target
            .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
            .unwrap();
        assert!(target.get_node(&node.id).unwrap().is_some());
        assert_eq!(
            target
                .get_composition_members("portable-purge-composition")
                .unwrap()[0]
                .preview
                .as_deref(),
            Some("Portable purge composition preview leak")
        );
        {
            let writer = target.writer.lock().unwrap();
            writer
                .execute(
                    "INSERT INTO memory_prs (
                        id, kind, status, title, subject_id, diff, signals, created_at
                     ) VALUES (?1, 'new_fact', 'pending', ?2, ?3, '{}', '[]', ?4)",
                    params![
                        "portable-purge-review",
                        "remote cleanup review",
                        &node.id,
                        Utc::now().to_rfc3339(),
                    ],
                )
                .unwrap();
        }

        source
            .purge_node(&node.id, Some("sync purge test"))
            .unwrap();
        let purge_archive = source.export_portable_archive().unwrap();
        assert!(
            !serde_json::to_string(&purge_archive)
                .unwrap()
                .contains("Portable purge composition preview leak"),
            "source portable archive should not retain purged composition previews"
        );
        let report = target
            .import_portable_archive(&purge_archive, PortableImportMode::Merge)
            .unwrap();

        assert!(report.rows_deleted >= 1);
        assert!(target.get_node(&node.id).unwrap().is_none());
        assert!(
            target
                .get_composition_members("portable-purge-composition")
                .unwrap()
                .is_empty(),
            "portable purge merge should delete composition evidence that references the target"
        );

        let writer = target.writer.lock().unwrap();
        let review_count: i64 = writer
            .query_row(
                "SELECT COUNT(*) FROM memory_prs WHERE id = 'portable-purge-review'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            review_count, 0,
            "portable purge merge must run the full non-FK evidence cleanup"
        );
        let tombstone_count: i64 = writer
            .query_row(
                "SELECT COUNT(*) FROM deletion_tombstones
                 WHERE memory_id = ?1 AND reason IS NULL AND tags = '[]'",
                params![SqliteMemoryStore::opaque_tombstone_marker(&node.id)],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(tombstone_count, 1);
    }

    #[test]
    fn opaque_tombstone_rejects_a_later_node_archive_even_with_newer_timestamp() {
        let source_dir = tempdir().unwrap();
        let target_dir = tempdir().unwrap();
        let source = create_test_storage_at(&source_dir, "source.db");
        let target = create_test_storage_at(&target_dir, "target.db");

        let node = source
            .ingest(IngestInput {
                content: "must not resurrect after opaque tombstone".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let mut later_node_archive = source.export_portable_archive().unwrap();
        let node_table = later_node_archive
            .tables
            .iter_mut()
            .find(|table| table.name == "knowledge_nodes")
            .unwrap();
        let updated_at_index = node_table
            .columns
            .iter()
            .position(|column| column == "updated_at")
            .unwrap();
        match &mut node_table.rows[0][updated_at_index] {
            PortableValue::Text(value) => *value = (Utc::now() + Duration::hours(24)).to_rfc3339(),
            value => panic!("knowledge_nodes.updated_at must be text, got {value:?}"),
        }

        source.purge_node(&node.id, None).unwrap();
        let tombstone_archive = source.export_portable_archive().unwrap();
        target
            .import_portable_archive(&tombstone_archive, PortableImportMode::Merge)
            .unwrap();
        assert!(target.get_node(&node.id).unwrap().is_none());

        let report = target
            .import_portable_archive(&later_node_archive, PortableImportMode::Merge)
            .unwrap();
        assert!(target.get_node(&node.id).unwrap().is_none());
        assert!(report.rows_skipped >= 1);
    }

    #[test]
    fn test_portable_merge_import_purge_wins_over_newer_local_edit() {
        let source_dir = tempdir().unwrap();
        let target_dir = tempdir().unwrap();
        let source = create_test_storage_at(&source_dir, "source.db");
        let target = create_test_storage_at(&target_dir, "target.db");

        let node = source
            .ingest(IngestInput {
                content: "Memory that will be purged on source".to_string(),
                node_type: "fact".to_string(),
                tags: vec!["sync".to_string()],
                ..Default::default()
            })
            .unwrap();
        let archive = source.export_portable_archive().unwrap();
        target
            .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
            .unwrap();

        let newer = (Utc::now() + Duration::hours(1)).to_rfc3339();
        {
            let writer = target.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET content = ?1, updated_at = ?2 WHERE id = ?3",
                    params!["Newer local edit before purge arrives", newer, &node.id],
                )
                .unwrap();
        }

        source
            .purge_node(&node.id, Some("hard purge wins sync conflict"))
            .unwrap();
        let purge_archive = source.export_portable_archive().unwrap();
        let report = target
            .import_portable_archive(&purge_archive, PortableImportMode::Merge)
            .unwrap();

        assert!(report.rows_deleted >= 1);
        assert!(target.get_node(&node.id).unwrap().is_none());
    }

    #[test]
    fn test_file_portable_sync_round_trips_between_devices() {
        let sync_dir = tempdir().unwrap();
        let first_dir = tempdir().unwrap();
        let second_dir = tempdir().unwrap();
        let sync_path = sync_dir.path().join("vestige-sync.json");
        let first = create_test_storage_at(&first_dir, "first.db");
        let second = create_test_storage_at(&second_dir, "second.db");

        let first_node = first
            .ingest(IngestInput {
                content: "First device memory".to_string(),
                node_type: "fact".to_string(),
                tags: vec!["sync".to_string()],
                ..Default::default()
            })
            .unwrap();
        let first_push = first.sync_portable_archive_file(&sync_path).unwrap();
        assert!(!first_push.pulled);
        assert!(sync_path.exists());

        let second_node = second
            .ingest(IngestInput {
                content: "Second device memory".to_string(),
                node_type: "fact".to_string(),
                tags: vec!["sync".to_string()],
                ..Default::default()
            })
            .unwrap();
        let second_sync = second.sync_portable_archive_file(&sync_path).unwrap();
        assert!(second_sync.pulled);
        assert!(second.get_node(&first_node.id).unwrap().is_some());

        let first_sync = first.sync_portable_archive_file(&sync_path).unwrap();
        assert!(first_sync.pulled);
        assert!(first.get_node(&second_node.id).unwrap().is_some());
        assert!(first_sync.pushed_rows >= 2);
    }

    #[test]
    fn test_get_last_backup_timestamp_no_panic() {
        // Static method should not panic even if no backups exist
        let _ = Storage::get_last_backup_timestamp();
    }

    #[test]
    fn test_keyword_search_with_include_types() {
        let storage = create_test_storage();

        // Ingest nodes of different types all containing the word "quantum"
        storage
            .ingest(IngestInput {
                content: "Quantum mechanics is fundamental to physics".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        storage
            .ingest(IngestInput {
                content: "Quantum computing uses qubits for calculation".to_string(),
                node_type: "concept".to_string(),
                ..Default::default()
            })
            .unwrap();
        storage
            .ingest(IngestInput {
                content: "Quantum entanglement was demonstrated in the lab".to_string(),
                node_type: "event".to_string(),
                ..Default::default()
            })
            .unwrap();

        // Search with include_types = ["fact"] — should only return the fact
        let include = vec!["fact".to_string()];
        let results = storage
            .hybrid_search_filtered("quantum", 10, 0.3, 0.7, Some(&include), None)
            .unwrap();

        assert!(!results.is_empty(), "should return at least one result");
        for r in &results {
            assert_eq!(
                r.node.node_type, "fact",
                "include_types=[fact] should only return facts, got: {}",
                r.node.node_type
            );
        }
    }

    #[test]
    fn test_keyword_search_with_exclude_types() {
        let storage = create_test_storage();

        storage
            .ingest(IngestInput {
                content: "Photosynthesis converts sunlight to energy".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        storage
            .ingest(IngestInput {
                content: "Photosynthesis is a complex biochemical process".to_string(),
                node_type: "reflection".to_string(),
                ..Default::default()
            })
            .unwrap();

        // Search with exclude_types = ["reflection"] — should skip the reflection
        let exclude = vec!["reflection".to_string()];
        let results = storage
            .hybrid_search_filtered("photosynthesis", 10, 0.3, 0.7, None, Some(&exclude))
            .unwrap();

        assert!(!results.is_empty(), "should return at least one result");
        for r in &results {
            assert_ne!(
                r.node.node_type, "reflection",
                "exclude_types=[reflection] should not return reflections"
            );
        }
    }

    #[test]
    fn test_include_types_takes_precedence_over_exclude() {
        let storage = create_test_storage();

        storage
            .ingest(IngestInput {
                content: "Gravity holds planets in orbit around stars".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        storage
            .ingest(IngestInput {
                content: "Gravity waves were first detected by LIGO".to_string(),
                node_type: "event".to_string(),
                ..Default::default()
            })
            .unwrap();

        // When both are provided, include_types wins
        let include = vec!["fact".to_string()];
        let exclude = vec!["fact".to_string()];
        let results = storage
            .hybrid_search_filtered("gravity", 10, 0.3, 0.7, Some(&include), Some(&exclude))
            .unwrap();

        // include_types takes precedence — facts should be returned
        assert!(!results.is_empty());
        for r in &results {
            assert_eq!(r.node.node_type, "fact");
        }
    }

    #[test]
    fn test_type_filter_with_no_matches_returns_empty() {
        let storage = create_test_storage();

        storage
            .ingest(IngestInput {
                content: "DNA carries genetic information in cells".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();

        // Search for a type that doesn't exist among matches
        let include = vec!["person".to_string()];
        let results = storage
            .hybrid_search_filtered("DNA", 10, 0.3, 0.7, Some(&include), None)
            .unwrap();

        assert!(
            results.is_empty(),
            "filtering for a non-matching type should return empty results"
        );
    }

    #[test]
    fn test_hybrid_search_backward_compat() {
        // Ensure the original hybrid_search (no type filters) still works
        let storage = create_test_storage();

        storage
            .ingest(IngestInput {
                content: "Neurons transmit electrical signals in the brain".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();

        let results = storage.hybrid_search("neurons", 10, 0.3, 0.7).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].node.content.contains("Neurons"));
    }

    #[test]
    fn test_concrete_search_literal_identifier_lands_first() {
        let storage = create_test_storage();

        storage
            .ingest(IngestInput {
                content: "General OpenAI API setup notes without the exact env var".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let target = storage
            .ingest(IngestInput {
                content: "Set OPENAI_API_KEY before running the release smoke tests".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        storage
            .ingest(IngestInput {
                content: "API keys should be handled carefully in shell profiles".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();

        let results = storage
            .concrete_search_filtered("OPENAI_API_KEY", 10, None, None)
            .unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].node.id, target.id);
        assert_eq!(results[0].match_type, MatchType::Keyword);
        assert!(results[0].semantic_score.is_none());
    }

    /// A memory that merely CITES an identifier must not outrank the memory that
    /// IS it. Raw BM25 magnitude is unbounded while literal_match_score is capped
    /// at 3.0, and both fed the same combined_score: measured on a 202-document
    /// corpus, a note citing a UUID three times scored 27.5 against the exact
    /// match's 3.0. That inverted the documented exact-lookup guarantee.
    /// A corrupt FTS index must NOT strand the user's memories. `knowledge_fts`
    /// is declared `content='knowledge_nodes'`, so it is derived state and is
    /// always reconstructible. This reproduces the field failure: a store with
    /// intact memories became unopenable because one fts5 blob was damaged.
    #[test]
    fn corrupt_fts_index_is_rebuilt_instead_of_bricking_the_store() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("vestige.db");

        // Seed a store and close it cleanly.
        {
            let storage = Storage::new(Some(path.clone())).expect("open");
            for i in 0..5 {
                storage
                    .ingest(IngestInput {
                        content: format!("memory number {i} about deployment"),
                        node_type: "fact".to_string(),
                        ..Default::default()
                    })
                    .expect("ingest");
            }
        }

        // Corrupt the FTS index the way an interrupted rebuild does. The
        // damage is a FIXED byte pattern, not randomblob(): an unseeded random
        // block sometimes wrecks the segment so badly that quick_check itself
        // fails with SQLITE_NOMEM before the heal gate can classify the rows,
        // and the store (correctly) refuses loudly. That path is documented
        // shipped behavior, but it made this test flake in CI; a deterministic
        // pattern exercises the rebuild path every run.
        {
            let conn = Connection::open(&path).expect("raw open");
            let pattern = "A5".repeat(200);
            conn.execute_batch(&format!(
                "UPDATE knowledge_fts_data SET block = x'{pattern}' \
                 WHERE id = (SELECT id FROM knowledge_fts_data WHERE id > 1 LIMIT 1);"
            ))
            .expect("corrupt");
            let corrupt = conn
                .execute_batch(
                    "INSERT INTO knowledge_fts(knowledge_fts) VALUES('integrity-check');",
                )
                .is_err();
            assert!(corrupt, "the fixture must actually corrupt the index");
        }

        // Reopening must succeed by rebuilding, not fail.
        let storage = Storage::new(Some(path.clone()))
            .expect("a corrupt DERIVED index must not prevent opening the store");
        let all = storage.get_all_nodes(100, 0).expect("list nodes");
        assert_eq!(all.len(), 5, "every memory must survive the rebuild");

        // And the rebuilt index must actually be usable again.
        let hits = storage
            .concrete_search_filtered("deployment", 10, None, None)
            .expect("keyword search after rebuild");
        assert!(
            !hits.is_empty(),
            "the rebuilt index must find the seeded memories"
        );
    }

    #[test]
    fn test_concrete_search_exact_match_beats_a_doc_that_only_cites_it() {
        let storage = create_test_storage();

        // Filler so BM25's IDF term is meaningful rather than degenerate.
        for i in 0..40 {
            storage
                .ingest(IngestInput {
                    content: format!("Routine note {i} about deployment pipelines and review"),
                    node_type: "fact".to_string(),
                    ..Default::default()
                })
                .unwrap();
        }

        let needle = "PAYMENTS_REDIS_URL";
        let target = storage
            .ingest(IngestInput {
                content: needle.to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        // Cites the identifier repeatedly -> large BM25 magnitude.
        storage
            .ingest(IngestInput {
                content: format!(
                    "See {needle} for the rollout; {needle} was rotated in review, and \
                     {needle} supersedes the older connection note entirely"
                ),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();

        let results = storage
            .concrete_search_filtered(needle, 10, None, None)
            .unwrap();
        assert!(!results.is_empty(), "exact lookup must return something");
        assert_eq!(
            results[0].node.id, target.id,
            "the memory that IS the identifier must rank first, not the one citing it"
        );
    }

    #[test]
    fn test_purge_scrubs_insight_json_orphans_children_and_writes_tombstone() {
        let storage = create_test_storage();
        let doomed = storage
            .ingest(IngestInput {
                content: "Sensitive purge target memory".to_string(),
                node_type: "fact".to_string(),
                tags: vec!["sensitive".to_string()],
                ..Default::default()
            })
            .unwrap();
        let other_a = storage
            .ingest(IngestInput {
                content: "Other source memory A".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let other_b = storage
            .ingest(IngestInput {
                content: "Other source memory B".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let child = storage
            .ingest(IngestInput {
                content: "Temporal summary child".to_string(),
                node_type: "summary".to_string(),
                ..Default::default()
            })
            .unwrap();

        {
            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "INSERT INTO memory_connections (
                        source_id, target_id, strength, link_type, created_at, last_activated, activation_count
                     ) VALUES (?1, ?2, 0.9, 'semantic', ?3, ?3, 0)",
                    params![doomed.id, other_a.id, Utc::now().to_rfc3339()],
                )
                .unwrap();
            writer
                .execute(
                    "INSERT INTO insights (
                        id, insight, source_memories, confidence, novelty_score, insight_type, generated_at
                     ) VALUES (?1, 'drop me', ?2, 0.9, 0.2, 'synthesis', ?3)",
                    params![
                        Uuid::new_v4().to_string(),
                        serde_json::to_string(&vec![doomed.id.clone(), other_a.id.clone()]).unwrap(),
                        Utc::now().to_rfc3339()
                    ],
                )
                .unwrap();
            writer
                .execute(
                    "INSERT INTO insights (
                        id, insight, source_memories, confidence, novelty_score, insight_type, generated_at
                     ) VALUES (?1, 'rewrite me', ?2, 0.9, 0.2, 'synthesis', ?3)",
                    params![
                        Uuid::new_v4().to_string(),
                        serde_json::to_string(&vec![
                            doomed.id.clone(),
                            other_a.id.clone(),
                            other_b.id.clone()
                        ])
                        .unwrap(),
                        Utc::now().to_rfc3339()
                    ],
                )
                .unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET summary_parent_id = ?1 WHERE id = ?2",
                    params![doomed.id, child.id],
                )
                .unwrap();
            writer
                .execute(
                    "INSERT INTO memory_prs (
                        id, kind, status, title, subject_id, diff, signals, created_at
                     ) VALUES (?1, 'new_fact', 'pending', ?2, ?3, ?4, '[]', ?5)",
                    params![
                        "purge-review-leak",
                        "Sensitive purge target memory review preview",
                        doomed.id,
                        serde_json::json!({
                            "contentPreview": "Sensitive purge target memory"
                        })
                        .to_string(),
                        Utc::now().to_rfc3339(),
                    ],
                )
                .unwrap();
        }

        storage
            .save_composition(
                &CompositionEventRecord {
                    id: "purge-composition-preview-test".to_string(),
                    created_at: Utc::now(),
                    tool: "deep_reference".to_string(),
                    mode: "audit".to_string(),
                    query: Some("purge preview leak".to_string()),
                    query_hash: Some("fnv1a64:purge".to_string()),
                    confidence: Some(0.7),
                    status: Some("resolved".to_string()),
                    output_preview: None,
                    metadata: serde_json::json!({}),
                },
                &[CompositionMemberRecord {
                    event_id: "purge-composition-preview-test".to_string(),
                    memory_id: doomed.id.clone(),
                    role: "primary".to_string(),
                    rank: 0,
                    trust: Some(0.8),
                    score: Some(0.9),
                    preview: Some("Sensitive purge target memory preview leak".to_string()),
                    metadata: serde_json::json!({}),
                }],
                &[],
            )
            .unwrap();

        let report = storage
            .purge_node(&doomed.id, Some("user requested hard purge"))
            .unwrap();
        assert!(report.deleted);
        assert_eq!(report.edges_pruned, 1);
        assert_eq!(report.insights_deleted, 1);
        assert_eq!(report.insights_rewritten, 1);
        assert_eq!(report.children_orphaned, 1);
        assert!(storage.get_node(&doomed.id).unwrap().is_none());

        let writer = storage.writer.lock().unwrap();
        let remaining_refs: Vec<String> = writer
            .prepare("SELECT source_memories FROM insights")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .filter_map(|row| row.ok())
            .collect();
        assert_eq!(remaining_refs.len(), 1);
        assert!(!remaining_refs[0].contains(&doomed.id));

        let child_parent: Option<String> = writer
            .query_row(
                "SELECT summary_parent_id FROM knowledge_nodes WHERE id = ?1",
                params![child.id],
                |row| row.get(0),
            )
            .unwrap();
        assert!(child_parent.is_none());

        let tombstone_count: i64 = writer
            .query_row(
                "SELECT COUNT(*) FROM deletion_tombstones
                 WHERE memory_id = ?1 AND reason IS NULL AND tags = '[]'",
                params![SqliteMemoryStore::opaque_tombstone_marker(&doomed.id)],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(tombstone_count, 1);
        let sync_tombstone_count: i64 = writer
            .query_row(
                "SELECT COUNT(*) FROM sync_tombstones
                 WHERE table_name = 'knowledge_nodes' AND row_id = ?1 AND reason IS NULL",
                params![SqliteMemoryStore::opaque_tombstone_marker(&doomed.id)],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(sync_tombstone_count, 1);

        let members = storage
            .get_composition_members("purge-composition-preview-test")
            .unwrap();
        assert!(
            members.is_empty(),
            "purge should remove composition evidence that references the target"
        );
        let review_count: i64 = writer
            .query_row(
                "SELECT COUNT(*) FROM memory_prs WHERE id = 'purge-review-leak'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            review_count, 0,
            "purge should remove linked review evidence"
        );
        let archive_json =
            serde_json::to_string(&storage.export_portable_archive().unwrap()).unwrap();
        assert!(
            !archive_json.contains("Sensitive purge target memory preview leak"),
            "portable archive should not retain purged memory content through composition previews"
        );
        assert!(
            !archive_json.contains(&doomed.id),
            "portable archive should not retain the purged memory's raw identifier"
        );
        assert!(
            !archive_json.contains("user requested hard purge"),
            "portable archive should not retain caller-controlled purge rationale"
        );

        let has_content_column: i64 = writer
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('deletion_tombstones') WHERE name = 'content'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(has_content_column, 0);
    }

    /// Purge is an erasure guarantee, so a referencing row the scrub cannot
    /// read has to abort the whole purge rather than be skipped. Before this,
    /// the three reference sweeps in `purge_node_in_transaction` used
    /// `filter_map(|row| row.ok())`, so an unreadable row silently kept its
    /// reference to a memory the caller was told had been erased.
    #[test]
    fn purge_fails_closed_when_a_referencing_row_cannot_be_read() {
        let storage = create_test_storage();
        let doomed = storage
            .ingest(IngestInput {
                content: "Purge target with an unreadable referrer".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();

        // `insights` is a non-STRICT table, so a BLOB survives in the TEXT
        // primary key. rusqlite then refuses to read it as a String, which is
        // exactly the "row we cannot read" the sweep used to swallow.
        {
            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "INSERT INTO insights (
                        id, insight, source_memories, confidence, novelty_score, insight_type, generated_at
                     ) VALUES (?1, 'unreadable id', ?2, 0.9, 0.2, 'synthesis', ?3)",
                    params![
                        vec![0xF0_u8, 0x9F, 0x92, 0xA9, 0xFF, 0xFE],
                        serde_json::to_string(&vec![doomed.id.clone()]).unwrap(),
                        Utc::now().to_rfc3339()
                    ],
                )
                .unwrap();
        }

        // The purge must refuse, not report success.
        let error = storage.purge_node(&doomed.id, None).unwrap_err();
        let rendered = error.to_string().to_lowercase();
        assert!(
            rendered.contains("type") || rendered.contains("column") || rendered.contains("convert"),
            "expected a read failure to surface, got: {rendered}"
        );

        // And the transaction rolled back: nothing was half-erased.
        assert!(
            storage.get_node(&doomed.id).unwrap().is_some(),
            "a refused purge must leave the memory intact, not partially scrubbed"
        );
    }

    // ========================================================================
    // Merge / Supersede controls (Phase 3 — v2.1.25)
    //
    // These exercise the full lifecycle without the live embedding model by
    // seeding the `node_embeddings` table directly with the ACTIVE model name,
    // so `get_all_embeddings` / `get_node_embedding` accept them.
    // ========================================================================

    /// Ingest a node and seed it with a controllable embedding under the active
    /// model so similarity is deterministic in tests.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn seed_node(storage: &Storage, content: &str, tags: &[&str], vector: Vec<f32>) -> String {
        let node = storage
            .ingest(IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                tags: tags.iter().map(|t| t.to_string()).collect(),
                ..Default::default()
            })
            .unwrap();
        let bytes = Embedding::new(vector).to_bytes();
        let active = storage.embedding_service.model_name().to_string();
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "INSERT OR REPLACE INTO node_embeddings
                 (node_id, embedding, dimensions, model, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    &node.id,
                    &bytes,
                    EMBEDDING_DIMENSIONS as i32,
                    active,
                    Utc::now().to_rfc3339()
                ],
            )
            .unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET has_embedding = 1 WHERE id = ?1",
                rusqlite::params![&node.id],
            )
            .unwrap();
        node.id
    }

    /// A near-unit vector pointing mostly along `axis`, so two nodes sharing an
    /// axis are highly similar and nodes on different axes are not.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn axis_vector(axis: usize, jitter: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; EMBEDDING_DIMENSIONS];
        v[axis % EMBEDDING_DIMENSIONS] = 1.0;
        v[(axis + 1) % EMBEDDING_DIMENSIONS] = jitter;
        v
    }

    // =========================================================================
    // #181: the in-process vector index follows writes made by peer processes
    // =========================================================================

    /// Write a caller-supplied vector for `node_id` through the production
    /// funnel every embedding write uses, so the row lands in the active
    /// profile's table, the legacy mirror, the `has_embedding` flag and THIS
    /// store's in-memory index exactly as a real ingest would.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn persist_test_vector(storage: &Storage, node_id: &str, vector: &[f32]) {
        let bytes = Embedding::new(vector.to_vec()).to_bytes();
        storage
            .persist_node_embedding(node_id, &bytes, vector.len(), "test-model", vector, true)
            .unwrap();
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn index_contains(storage: &Storage, node_id: &str) -> bool {
        storage
            .vector_index
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .contains(node_id)
    }

    /// Id of the top hit for `vector` in this store's in-memory index.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn nearest(storage: &Storage, vector: &[f32]) -> Option<String> {
        storage
            .vector_index
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .search(vector, 1)
            .unwrap()
            .into_iter()
            .next()
            .map(|(id, _)| id)
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn ingest_plain(storage: &Storage, content: &str) -> String {
        storage
            .ingest(IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap()
            .id
    }

    /// #181: a memory written by a peer process must become semantically
    /// searchable in THIS process without a restart. The pre-refresh assertion
    /// is the negative half: it is exactly what every process saw before the fix.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn peer_process_write_is_visible_to_the_vector_index_without_restart() {
        let dir = tempdir().unwrap();
        let ours = create_test_storage_at(&dir, "shared.db");
        let peer = create_test_storage_at(&dir, "shared.db");

        let id = ingest_plain(&peer, "written by a sibling MCP server process");
        persist_test_vector(&peer, &id, &axis_vector(3, 0.01));

        assert!(
            !index_contains(&ours, &id),
            "a process-local index cannot know about a peer's write until it refreshes"
        );

        assert_eq!(
            ours.refresh_vector_index_if_stale(),
            1,
            "exactly the peer's row is absorbed"
        );
        assert!(index_contains(&ours, &id));
        assert_eq!(nearest(&ours, &axis_vector(3, 0.0)).as_deref(), Some(id.as_str()));

        assert_eq!(
            ours.refresh_vector_index_if_stale(),
            0,
            "nothing new since: one PRAGMA and an early return"
        );
    }

    /// #181: a peer re-embedding an existing node through the UPSERT path (used by
    /// profile repair, which keeps the row's rowid) must replace the stale vector
    /// here. A rowid or contains()-based refresh could never notice this write.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn peer_reembedding_replaces_the_stale_vector_here() {
        let dir = tempdir().unwrap();
        let ours = create_test_storage_at(&dir, "shared.db");
        let peer = create_test_storage_at(&dir, "shared.db");

        let moved = ingest_plain(&peer, "a memory whose vector will be regenerated");
        let decoy = ingest_plain(&peer, "a decoy that stays on the old axis");
        persist_test_vector(&peer, &moved, &axis_vector(3, 0.01));
        persist_test_vector(&peer, &decoy, &axis_vector(3, 0.02));
        assert_eq!(ours.refresh_vector_index_if_stale(), 2);
        assert_eq!(
            nearest(&ours, &axis_vector(3, 0.01)).as_deref(),
            Some(moved.as_str())
        );

        let active = peer
            .active_embedding_profile()
            .unwrap()
            .expect("test stores have an active profile")
            .profile_id
            .to_string();
        peer.put_embedding_profile_vector(&EmbeddingProfileVector {
            profile_id: active,
            node_id: moved.clone(),
            embedding: Embedding::new(axis_vector(9, 0.01)).to_bytes(),
            dimensions: EMBEDDING_DIMENSIONS as u32,
            model: "test-model".to_string(),
            created_at: Utc::now(),
        })
        .unwrap();

        assert_eq!(
            ours.refresh_vector_index_if_stale(),
            1,
            "one upsert row, one replaced vector"
        );
        assert_eq!(
            nearest(&ours, &axis_vector(9, 0.0)).as_deref(),
            Some(moved.as_str())
        );
        assert_eq!(
            nearest(&ours, &axis_vector(3, 0.02)).as_deref(),
            Some(decoy.as_str()),
            "the moved node must no longer win its old axis"
        );
    }

    /// #181: a peer's purge cascades to its vector row, the delete trigger
    /// journals it, and the dead vector leaves this index too.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn peer_purge_removes_the_vector_here() {
        let dir = tempdir().unwrap();
        let ours = create_test_storage_at(&dir, "shared.db");
        let peer = create_test_storage_at(&dir, "shared.db");

        let id = ingest_plain(&peer, "a memory the peer will purge");
        persist_test_vector(&peer, &id, &axis_vector(4, 0.01));
        assert_eq!(ours.refresh_vector_index_if_stale(), 1);
        assert!(index_contains(&ours, &id));

        peer.purge_node(&id, Some("peer purge")).unwrap();

        assert_eq!(
            ours.refresh_vector_index_if_stale(),
            1,
            "one delete row, one removal"
        );
        assert!(!index_contains(&ours, &id));
    }

    /// #181: this process's own writes bump the reader's data_version exactly
    /// like a peer's would, but the vector is already in the index. The journal
    /// head says so, and the refresh re-adds nothing.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn own_writes_are_not_reabsorbed_as_peer_changes() {
        let storage = create_test_storage();
        let id = ingest_plain(&storage, "written by this very process");
        persist_test_vector(&storage, &id, &axis_vector(5, 0.01));
        assert!(index_contains(&storage, &id));
        assert_eq!(storage.refresh_vector_index_if_stale(), 0);
    }

    /// #181: when the journal has been pruned past this process's watermark, the
    /// refresh must not trust it. It reconciles against the table and still
    /// absorbs everything the peers wrote, including rows the journal no longer
    /// names.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn a_journal_pruned_past_the_watermark_reconciles_against_the_table() {
        let dir = tempdir().unwrap();
        let ours = create_test_storage_at(&dir, "shared.db");
        let peer = create_test_storage_at(&dir, "shared.db");

        let first = ingest_plain(&peer, "first peer memory");
        let second = ingest_plain(&peer, "second peer memory");
        persist_test_vector(&peer, &first, &axis_vector(1, 0.01));
        persist_test_vector(&peer, &second, &axis_vector(2, 0.01));

        // A prune that ran before this process caught up: only the newest row
        // survives, so the journal alone would name just one of the two.
        {
            let writer = peer.writer.lock().unwrap();
            writer
                .execute(
                    "DELETE FROM vector_journal WHERE seq < (SELECT MAX(seq) FROM vector_journal)",
                    [],
                )
                .unwrap();
        }

        assert_eq!(
            ours.refresh_vector_index_if_stale(),
            2,
            "reconcile absorbs both peer vectors, not just the journal survivor"
        );
        assert!(index_contains(&ours, &first));
        assert!(index_contains(&ours, &second));
        assert_eq!(
            ours.refresh_vector_index_if_stale(),
            0,
            "the watermark moved to the head, so the next look is incremental and empty"
        );
    }

    /// #181 housekeeping: pruning removes only rows that are both older than
    /// the retention window AND further behind the head than the keep window.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn vector_journal_prune_keeps_recent_rows_and_the_head() {
        let storage = create_test_storage();
        let id = ingest_plain(&storage, "one vector, at least one journal row");
        persist_test_vector(&storage, &id, &axis_vector(6, 0.01));
        assert_eq!(
            storage.prune_vector_journal().unwrap(),
            0,
            "fresh rows are never pruned"
        );
        {
            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE vector_journal SET at = '2000-01-01T00:00:00.000Z'",
                    [],
                )
                .unwrap();
            // Push the head far past the keep window with one synthetic row.
            writer
                .execute(
                    "INSERT INTO vector_journal (seq, profile_id, node_id, op)
                     VALUES (20000, 'synthetic', 'head', 'upsert')",
                    [],
                )
                .unwrap();
        }
        let deleted = storage.prune_vector_journal().unwrap();
        assert!(deleted >= 1, "old rows far behind the head must go");
        let remaining: i64 = storage
            .reader
            .lock()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM vector_journal", [], |row| row.get(0))
            .unwrap();
        assert_eq!(remaining, 1, "only the fresh head row survives");
    }

    // =========================================================================
    // Phase 1 trait-method unit tests
    // =========================================================================
    use crate::storage::memory_store::{
        MemoryEdge, MemoryRecord, MemoryStore, MemoryStoreError, ModelSignature, SchedulingState,
    };

    fn make_record(content: &str) -> MemoryRecord {
        MemoryRecord {
            id: uuid::Uuid::new_v4(),
            domains: vec![],
            domain_scores: Default::default(),
            content: content.to_string(),
            node_type: "fact".to_string(),
            tags: vec!["test".to_string()],
            embedding: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metadata: serde_json::json!({}),
        }
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Runtime::new().unwrap()
    }

    #[test]
    fn trait_init_is_idempotent() {
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            s.init().await.unwrap();
            s.init().await.unwrap();
        });
    }

    #[test]
    fn trait_health_check_reports_healthy_on_fresh_db() {
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            let h = s.health_check().await.unwrap();
            assert!(matches!(
                h,
                crate::storage::memory_store::HealthStatus::Healthy
            ));
        });
    }

    #[test]
    fn trait_register_model_first_write_succeeds() {
        let s = create_test_storage();
        let sig = ModelSignature {
            name: "test-model".to_string(),
            dimension: 256,
            hash: "a".repeat(64),
        };
        let rt = rt();
        rt.block_on(async {
            s.register_model(&sig).await.unwrap();
            let got = s.registered_model().await.unwrap();
            assert_eq!(got, Some(sig));
        });
    }

    #[test]
    fn trait_register_model_mismatched_write_refused() {
        let s = create_test_storage();
        let sig = ModelSignature {
            name: "model-a".to_string(),
            dimension: 256,
            hash: "a".repeat(64),
        };
        let sig2 = ModelSignature {
            name: "model-b".to_string(),
            dimension: 256,
            hash: "b".repeat(64),
        };
        let rt = rt();
        rt.block_on(async {
            s.register_model(&sig).await.unwrap();
            let err = s.register_model(&sig2).await.unwrap_err();
            assert!(matches!(err, MemoryStoreError::ModelMismatch { .. }));
        });
    }

    #[test]
    fn trait_register_model_same_signature_idempotent() {
        let s = create_test_storage();
        let sig = ModelSignature {
            name: "test-model".to_string(),
            dimension: 256,
            hash: "a".repeat(64),
        };
        let rt = rt();
        rt.block_on(async {
            s.register_model(&sig).await.unwrap();
            s.register_model(&sig).await.unwrap(); // second call must not error
        });
    }

    #[test]
    fn trait_insert_returns_uuid() {
        let s = create_test_storage();
        let rec = make_record("test content");
        let expected_id = rec.id;
        let rt = rt();
        rt.block_on(async {
            let got = s.insert(&rec).await.unwrap();
            assert_eq!(got, expected_id);
        });
    }

    #[test]
    fn trait_get_missing_returns_none() {
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            let got = s.get(uuid::Uuid::new_v4()).await.unwrap();
            assert!(got.is_none());
        });
    }

    #[test]
    fn trait_get_after_insert_round_trip() {
        let s = create_test_storage();
        let rec = make_record("round trip content");
        let id = rec.id;
        let rt = rt();
        rt.block_on(async {
            s.insert(&rec).await.unwrap();
            let got = s.get(id).await.unwrap().unwrap();
            assert_eq!(got.content, "round trip content");
            assert_eq!(got.node_type, "fact");
            assert!(got.domains.is_empty());
            assert!(got.domain_scores.is_empty());
        });
    }

    #[test]
    fn trait_update_modifies_content() {
        let s = create_test_storage();
        let rec = make_record("original content");
        let id = rec.id;
        let rt = rt();
        rt.block_on(async {
            s.insert(&rec).await.unwrap();
            let mut updated = s.get(id).await.unwrap().unwrap();
            updated.content = "updated content".to_string();
            s.update(&updated).await.unwrap();
            let got = s.get(id).await.unwrap().unwrap();
            assert_eq!(got.content, "updated content");
        });
    }

    #[test]
    fn trait_delete_removes_record() {
        let s = create_test_storage();
        let rec = make_record("to be deleted");
        let id = rec.id;
        let rt = rt();
        rt.block_on(async {
            s.insert(&rec).await.unwrap();
            s.delete(id).await.unwrap();
            let got = s.get(id).await.unwrap();
            assert!(got.is_none());
        });
    }

    #[test]
    fn trait_fts_search_returns_tokens_match() {
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            let rec = make_record("mitochondria powerhouse cell energy");
            s.insert(&rec).await.unwrap();
            let results = s.fts_search("mitochondria", 10).await.unwrap();
            assert!(!results.is_empty());
        });
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_merge_candidates_threshold_classification() {
        let storage = create_test_storage();
        // Two near-identical (same axis) — should be offered as a candidate.
        let a = seed_node(
            &storage,
            "Use tokio runtime for async Rust services",
            &["rust", "async"],
            axis_vector(3, 0.02),
        );
        let b = seed_node(
            &storage,
            "Use the tokio runtime for async Rust services",
            &["rust", "async"],
            axis_vector(3, 0.01),
        );
        // One unrelated (different axis) — must not join the cluster.
        let _c = seed_node(
            &storage,
            "Prefer postgres for relational data",
            &["db"],
            axis_vector(200, 0.0),
        );

        let policy = MergePolicy::default();
        let candidates = storage.merge_candidates(policy, 20, &[]).unwrap();
        assert_eq!(candidates.len(), 1, "exactly one duplicate cluster");
        let cluster = &candidates[0];
        assert_eq!(cluster.member_ids.len(), 2);
        assert!(cluster.member_ids.contains(&a));
        assert!(cluster.member_ids.contains(&b));
        assert!(
            cluster.confidence >= policy.possible_threshold,
            "confidence above possible threshold"
        );
        assert!(!cluster.has_protected_member);
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_plan_merge_is_preview_only_no_mutation() {
        let storage = create_test_storage();
        let a = seed_node(
            &storage,
            "Fact A about caching",
            &["perf"],
            axis_vector(5, 0.02),
        );
        let b = seed_node(
            &storage,
            "Fact A about caching, expanded",
            &["perf", "cache"],
            axis_vector(5, 0.01),
        );

        let plan = storage
            .plan_merge(&[a.clone(), b.clone()], None, MergePolicy::default())
            .unwrap();

        // Plan diff is populated...
        assert!(plan.result_content.contains("Fact A about caching"));
        assert!(plan.result_tags.contains(&"cache".to_string()));
        assert_eq!(plan.invalidated_ids.len(), 1);

        // ...but NOTHING changed: both nodes still valid, content untouched.
        let na = storage.get_node(&a).unwrap().unwrap();
        let nb = storage.get_node(&b).unwrap().unwrap();
        assert_eq!(na.content, "Fact A about caching");
        assert_eq!(nb.content, "Fact A about caching, expanded");
        let (vu_a, sb_a) = storage.read_bitemporal(&a).unwrap();
        let (vu_b, sb_b) = storage.read_bitemporal(&b).unwrap();
        assert!(vu_a.is_none() && sb_a.is_none());
        assert!(vu_b.is_none() && sb_b.is_none());

        // Plan persisted as pending.
        assert_eq!(
            storage.plan_status(&plan.id).unwrap().as_deref(),
            Some("pending")
        );
    }

    /// #180: `apply_plan` used to mutate through helpers that each committed on
    /// their own and only afterwards insert the undo row, and its plan-status
    /// check was not atomic with those mutations. Two MCP server processes
    /// sharing one database file could both pass the check and both apply, and
    /// a failure between the survivor rewrite and the undo insert left the
    /// survivor overwritten with no way back. The whole apply is one IMMEDIATE
    /// transaction now, so the plan applies exactly once no matter how many
    /// callers race it, and every mutation is covered by an undo row.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn concurrent_apply_of_one_plan_applies_it_exactly_once() {
        let storage = std::sync::Arc::new(create_test_storage());
        let survivor = seed_node(&storage, "Canonical race note", &["r"], axis_vector(5, 0.02));
        let absorbed = seed_node(&storage, "Detail to absorb", &["r", "s"], axis_vector(5, 0.01));

        let plan = storage
            .plan_merge(
                &[survivor.clone(), absorbed.clone()],
                Some(&survivor),
                MergePolicy::default(),
            )
            .unwrap();

        let racers: Vec<_> = (0..2)
            .map(|_| {
                let storage = std::sync::Arc::clone(&storage);
                let plan_id = plan.id.clone();
                std::thread::spawn(move || storage.apply_plan(&plan_id, true).is_ok())
            })
            .collect();
        let wins = racers
            .into_iter()
            .filter_map(|handle| handle.join().ok())
            .filter(|applied| *applied)
            .count();

        assert_eq!(
            wins, 1,
            "exactly one racer may apply the plan; the other must be refused"
        );

        // Exactly one reflog row, so the mutation that happened is reversible
        // and did not happen twice.
        let ops: i64 = {
            let reader = storage.reader.lock().unwrap();
            reader
                .query_row(
                    "SELECT COUNT(*) FROM merge_operations WHERE plan_id = ?1 AND status = 'applied'",
                    params![plan.id],
                    |row| row.get(0),
                )
                .unwrap()
        };
        assert_eq!(ops, 1, "one apply must leave exactly one undo row");

        // And the undo row actually carries what it needs to reverse.
        let payload: String = {
            let reader = storage.reader.lock().unwrap();
            reader
                .query_row(
                    "SELECT undo_payload FROM merge_operations WHERE plan_id = ?1",
                    params![plan.id],
                    |row| row.get(0),
                )
                .unwrap()
        };
        let undo: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert!(
            undo.get("survivor_prev_content").is_some(),
            "undo row must snapshot the survivor's pre-merge content: {undo}"
        );
        assert_eq!(storage.plan_status(&plan.id).unwrap().as_deref(), Some("applied"));
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_apply_then_undo_merge_is_reversible() {
        let storage = create_test_storage();
        let survivor = seed_node(
            &storage,
            "Keep this canonical note",
            &["x"],
            axis_vector(7, 0.02),
        );
        let absorbed = seed_node(
            &storage,
            "Extra detail to fold in",
            &["x", "y"],
            axis_vector(7, 0.01),
        );

        let plan = storage
            .plan_merge(
                &[survivor.clone(), absorbed.clone()],
                Some(&survivor),
                MergePolicy::default(),
            )
            .unwrap();
        let op = storage.apply_plan(&plan.id, true).unwrap();
        assert_eq!(op.op_type, "merge");

        // After apply: survivor content merged, absorbed bitemporally invalidated
        // but STILL QUERYABLE (never deleted).
        let surv = storage.get_node(&survivor).unwrap().unwrap();
        assert!(surv.content.contains("Keep this canonical note"));
        assert!(surv.content.contains("Extra detail to fold in"));
        assert!(surv.tags.contains(&"y".to_string()));

        let (vu, sb) = storage.read_bitemporal(&absorbed).unwrap();
        assert!(vu.is_some(), "absorbed node stamped valid_until");
        assert_eq!(sb.as_deref(), Some(survivor.as_str()));
        // Old node is still fully retrievable for audit.
        assert!(
            storage.get_node(&absorbed).unwrap().is_some(),
            "superseded node remains queryable"
        );
        assert!(storage.superseded_node_ids().unwrap().contains(&absorbed));

        // Undo restores everything.
        let undo = storage.merge_undo(&op.id).unwrap();
        assert_eq!(undo.op_type, "undo");
        let surv_after = storage.get_node(&survivor).unwrap().unwrap();
        assert_eq!(surv_after.content, "Keep this canonical note");
        let (vu2, sb2) = storage.read_bitemporal(&absorbed).unwrap();
        assert!(
            vu2.is_none() && sb2.is_none(),
            "invalidation cleared on undo"
        );
        assert!(!storage.superseded_node_ids().unwrap().contains(&absorbed));

        // The original op is now marked reverted; double-undo is rejected.
        assert!(storage.merge_undo(&op.id).is_err());
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_supersede_invalidates_old_but_keeps_it_queryable() {
        let storage = create_test_storage();
        let old = seed_node(&storage, "LR should be 1e-4", &["ml"], axis_vector(9, 0.02));
        let new = seed_node(
            &storage,
            "Correction: LR should be 3e-4",
            &["ml"],
            axis_vector(9, 0.01),
        );

        let plan = storage
            .plan_supersede(&old, &new, MergePolicy::default())
            .unwrap();
        // Preview did not mutate.
        let (vu0, _) = storage.read_bitemporal(&old).unwrap();
        assert!(vu0.is_none());

        let op = storage.apply_plan(&plan.id, true).unwrap();
        assert_eq!(op.op_type, "supersede");

        let (vu, sb) = storage.read_bitemporal(&old).unwrap();
        assert!(vu.is_some(), "old stamped valid_until");
        assert_eq!(sb.as_deref(), Some(new.as_str()));
        // New node untouched and valid.
        let (vu_new, sb_new) = storage.read_bitemporal(&new).unwrap();
        assert!(vu_new.is_none() && sb_new.is_none());
        // Old still queryable for audit (invalidate, don't delete).
        let old_node = storage.get_node(&old).unwrap().unwrap();
        assert_eq!(old_node.content, "LR should be 1e-4");

        // And reversible.
        storage.merge_undo(&op.id).unwrap();
        let (vu_r, sb_r) = storage.read_bitemporal(&old).unwrap();
        assert!(vu_r.is_none() && sb_r.is_none());
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_protect_blocks_merge_away() {
        let storage = create_test_storage();
        let pinned = seed_node(
            &storage,
            "Load-bearing fact",
            &["pin"],
            axis_vector(11, 0.02),
        );
        let other = seed_node(
            &storage,
            "Load-bearing fact restated",
            &["pin"],
            axis_vector(11, 0.01),
        );
        storage.set_protected(&pinned, true).unwrap();
        assert!(storage.is_protected(&pinned).unwrap());

        // Protected node may not be merged AWAY (survivor=other).
        let err = storage.plan_merge(
            &[other.clone(), pinned.clone()],
            Some(&other),
            MergePolicy::default(),
        );
        assert!(err.is_err(), "merging a protected node away must fail");

        // But it CAN be the survivor.
        let ok = storage.plan_merge(
            &[pinned.clone(), other.clone()],
            Some(&pinned),
            MergePolicy::default(),
        );
        assert!(ok.is_ok(), "protected node can be the survivor");

        // Supersede of a protected node is also blocked.
        assert!(
            storage
                .plan_supersede(&pinned, &other, MergePolicy::default())
                .is_err(),
            "superseding a protected node must fail"
        );

        // merge_candidates flags the protected member.
        let cands = storage
            .merge_candidates(MergePolicy::default(), 20, &[])
            .unwrap();
        assert!(cands.iter().all(|c| c.has_protected_member));
    }

    // ========================================================================
    // Auto-consolidation merge: opt-out gate + protected-pin exclusion (#142)
    //
    // These exercise `auto_dedup_consolidation` directly — the unattended,
    // no-audit pass the 6h background consolidation cycle runs. seed_node/
    // axis_vector give deterministic same-axis clusters (cosine ~1.0 >> the 0.85
    // threshold); set_retention pins down which node wins the keeper tiebreak.
    // ========================================================================

    /// Force a node's retention_strength so the keeper tiebreak in
    /// `auto_dedup_consolidation` is deterministic regardless of insertion order.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn set_retention(storage: &Storage, id: &str, value: f64) {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET retention_strength = ?1 WHERE id = ?2",
                rusqlite::params![value, id],
            )
            .unwrap();
    }

    /// Run `f` with VESTIGE_AUTO_CONSOLIDATE_MERGE pinned to `value`
    /// (None = pinned-unset, i.e. the documented ON default), serialized via
    /// ENV_LOCK and restored afterward (process env is global + unsafe under
    /// Rust 2024). Sibling of `with_vector_search_disabled`.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn with_auto_merge_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
        const KEY: &str = "VESTIGE_AUTO_CONSOLIDATE_MERGE";
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = std::env::var_os(KEY);
        unsafe {
            match value {
                Some(v) => std::env::set_var(KEY, v),
                None => std::env::remove_var(KEY),
            }
        }
        let result = catch_unwind(AssertUnwindSafe(f));
        unsafe {
            if let Some(prev) = previous {
                std::env::set_var(KEY, prev);
            } else {
                std::env::remove_var(KEY);
            }
        }
        match result {
            Ok(value) => value,
            Err(payload) => resume_unwind(payload),
        }
    }

    // --- A. Default (flag unset): NOTHING merges, nothing is deleted ---------
    // v2.6.0 flipped the #142 opt-out into an opt-in: unattended destruction
    // of user memories must be asked for, never inherited.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_auto_dedup_default_off_preserves_near_duplicates() {
        with_auto_merge_env(None, || {
            let storage = create_test_storage();
            let keeper = seed_node(
                &storage,
                "Rate limiting uses a token bucket per client API key",
                &["api"],
                axis_vector(21, 0.02),
            );
            let dup = seed_node(
                &storage,
                "Rate limiting uses a token-bucket algorithm per client API key, refilled steadily",
                &["api"],
                axis_vector(21, 0.01),
            );
            set_retention(&storage, &keeper, 0.9);
            set_retention(&storage, &dup, 0.3);

            let merged = storage.auto_dedup_consolidation().unwrap();
            assert_eq!(merged, 0, "flag unset: the destructive pass must not run");
            assert!(
                storage.get_node(&dup).unwrap().is_some(),
                "near-duplicate survives by default"
            );
            assert!(
                !storage
                    .get_node(&keeper)
                    .unwrap()
                    .unwrap()
                    .content
                    .contains("[MERGED]"),
                "keeper content untouched by default"
            );
        });
        // Explicit opt-in (trimmed, case-insensitive 1/true/on/yes) enables it.
        for value in ["1", "true", "ON", "  Yes  "] {
            with_auto_merge_env(Some(value), || {
                let storage = create_test_storage();
                let keeper = seed_node(
                    &storage,
                    "Rate limiting uses a token bucket per client API key",
                    &["api"],
                    axis_vector(21, 0.02),
                );
                let dup = seed_node(
                    &storage,
                    "Rate limiting uses a token-bucket algorithm per client API key, refilled steadily",
                    &["api"],
                    axis_vector(21, 0.01),
                );
                set_retention(&storage, &keeper, 0.9);
                set_retention(&storage, &dup, 0.3);

                let merged = storage.auto_dedup_consolidation().unwrap();
                assert_eq!(merged, 1, "opted in ({value:?}): weak node folds into keeper");
                assert!(storage.get_node(&dup).unwrap().is_none());
                assert!(
                    storage
                        .get_node(&keeper)
                        .unwrap()
                        .unwrap()
                        .content
                        .contains("[MERGED]"),
                    "keeper carries the folded-in [MERGED] block"
                );
            });
        }
    }

    // --- B. Flag off suppresses the merge (parametrized) ---------------------
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_auto_dedup_env_off_suppresses_merge() {
        // trimmed + case-insensitive false/off/no/0 all disable.
        for value in ["false", "off", "no", "0", "  OFF  ", "False"] {
            with_auto_merge_env(Some(value), || {
                let storage = create_test_storage();
                let a = seed_node(
                    &storage,
                    "Prometheus scrapes its targets every 15 seconds",
                    &["obs"],
                    axis_vector(23, 0.02),
                );
                let b = seed_node(
                    &storage,
                    "Prometheus scrapes its configured targets every 15s by default",
                    &["obs"],
                    axis_vector(23, 0.01),
                );

                let merged = storage.auto_dedup_consolidation().unwrap();
                assert_eq!(merged, 0, "value {value:?} must suppress the merge");
                // Both nodes survive, content byte-identical (no [MERGED] block).
                assert_eq!(
                    storage.get_node(&a).unwrap().unwrap().content,
                    "Prometheus scrapes its targets every 15 seconds"
                );
                assert_eq!(
                    storage.get_node(&b).unwrap().unwrap().content,
                    "Prometheus scrapes its configured targets every 15s by default"
                );
            });
        }
    }

    // --- B (cont). A malformed value fails CLOSED: no destruction on a typo --
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_auto_dedup_env_garbage_fails_closed_and_preserves() {
        with_auto_merge_env(Some("banana"), || {
            let storage = create_test_storage();
            let keeper = seed_node(
                &storage,
                "Cache entries expire after a five minute TTL",
                &["cache"],
                axis_vector(25, 0.02),
            );
            let dup = seed_node(
                &storage,
                "Cache entries expire after a five-minute TTL window by default",
                &["cache"],
                axis_vector(25, 0.01),
            );
            set_retention(&storage, &keeper, 0.9);
            set_retention(&storage, &dup, 0.3);

            let merged = storage.auto_dedup_consolidation().unwrap();
            assert_eq!(merged, 0, "malformed value fails closed for a destructive gate");
            assert!(storage.get_node(&dup).unwrap().is_some(), "nothing deleted on a typo");
        });
    }

    // --- C(a). Protected would-be keeper: untouched; others merge -----------
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_auto_dedup_protected_would_be_keeper_untouched_others_merge() {
        with_auto_merge_env(Some("1"), || {
            let storage = create_test_storage();
            // P has the highest retention, so absent protection it would be the
            // keeper. Protected → skipped entirely; the two unprotected merge alone.
            let pinned = seed_node(
                &storage,
                "Deploys are gated on a green CI run and one approval",
                &["ci"],
                axis_vector(27, 0.02),
            );
            let keeper = seed_node(
                &storage,
                "Deploys are gated on a green CI pipeline plus one reviewer approval",
                &["ci"],
                axis_vector(27, 0.01),
            );
            let member = seed_node(
                &storage,
                "Deploys require a green CI run and at least one approving review",
                &["ci"],
                axis_vector(27, 0.015),
            );
            set_retention(&storage, &pinned, 0.95);
            set_retention(&storage, &keeper, 0.80);
            set_retention(&storage, &member, 0.30);
            storage.set_protected(&pinned, true).unwrap();
            let pinned_content = storage.get_node(&pinned).unwrap().unwrap().content;

            let merged = storage.auto_dedup_consolidation().unwrap();
            assert_eq!(
                merged, 1,
                "the two unprotected near-dups merge among themselves"
            );

            // Protected node byte-for-byte untouched and still protected.
            let p = storage.get_node(&pinned).unwrap().unwrap();
            assert_eq!(p.content, pinned_content, "protected keeper not absorbed");
            assert!(!p.content.contains("[MERGED]"));
            assert!(storage.is_protected(&pinned).unwrap());
            // Unprotected pair merged: `member` gone, `keeper` carries [MERGED].
            assert!(storage.get_node(&member).unwrap().is_none());
            let keeper_node = storage.get_node(&keeper).unwrap().unwrap();
            assert!(keeper_node.content.contains("[MERGED]"));
        });
    }

    // --- C(b) / Regression (#142): protected weak member is never absorbed --
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn auto_dedup_regression_142_protected_weak_member_not_absorbed() {
        with_auto_merge_env(Some("1"), || {
            let storage = create_test_storage();
            // Regression (#142): before the fix this pinned node — the weaker
            // member of the cluster — was silently absorbed into the stronger
            // unprotected keeper and hard-deleted by the unattended pass. The
            // PINNED-CANARY-142 marker makes accidental absorption detectable.
            let pinned = seed_node(
                &storage,
                "Feature flags default to off in production PINNED-CANARY-142",
                &["flags"],
                axis_vector(29, 0.02),
            );
            let keeper = seed_node(
                &storage,
                "Feature flags default to off in the production environment",
                &["flags"],
                axis_vector(29, 0.01),
            );
            let member = seed_node(
                &storage,
                "Feature flags are off by default in production deployments",
                &["flags"],
                axis_vector(29, 0.015),
            );
            // Pinned is the LOWEST-retention member — pre-fix it would land in
            // weak_ids and be deleted + absorbed by the keeper.
            set_retention(&storage, &pinned, 0.10);
            set_retention(&storage, &keeper, 0.80);
            set_retention(&storage, &member, 0.30);
            storage.set_protected(&pinned, true).unwrap();
            let pinned_content = storage.get_node(&pinned).unwrap().unwrap().content;

            let merged = storage.auto_dedup_consolidation().unwrap();
            assert_eq!(merged, 1, "only the two unprotected near-dups merge");

            // Invariant 1: the protected node still exists, byte-identical.
            let p = storage.get_node(&pinned).unwrap();
            assert!(p.is_some(), "protected node must not be deleted");
            assert_eq!(
                p.unwrap().content,
                pinned_content,
                "protected node not absorbed"
            );
            assert!(storage.is_protected(&pinned).unwrap());

            // Invariant 2: the keeper did NOT gain the protected node's content.
            let keeper_node = storage.get_node(&keeper).unwrap().unwrap();
            assert!(
                !keeper_node.content.contains("PINNED-CANARY-142"),
                "keeper must not absorb the protected node's content"
            );
            // The legitimate unprotected pair still merged (member folded in).
            assert!(storage.get_node(&member).unwrap().is_none());
            assert!(keeper_node.content.contains("[MERGED]"));
        });
    }

    // --- C(c). Two protected near-dups: neither merges ----------------------
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_auto_dedup_two_protected_near_dups_neither_merges() {
        with_auto_merge_env(Some("1"), || {
            let storage = create_test_storage();
            let a = seed_node(
                &storage,
                "Backups run nightly and are retained for thirty days",
                &["backup"],
                axis_vector(31, 0.02),
            );
            let b = seed_node(
                &storage,
                "Backups run every night and are kept for thirty days",
                &["backup"],
                axis_vector(31, 0.01),
            );
            storage.set_protected(&a, true).unwrap();
            storage.set_protected(&b, true).unwrap();
            let (ca, cb) = (
                storage.get_node(&a).unwrap().unwrap().content,
                storage.get_node(&b).unwrap().unwrap().content,
            );

            let merged = storage.auto_dedup_consolidation().unwrap();
            assert_eq!(merged, 0, "two protected near-dups: nothing merges");
            assert_eq!(storage.get_node(&a).unwrap().unwrap().content, ca);
            assert_eq!(storage.get_node(&b).unwrap().unwrap().content, cb);
        });
    }

    // --- C(d). Protected + a single unprotected near-dup: no merge ----------
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_auto_dedup_protected_plus_single_unprotected_no_merge() {
        with_auto_merge_env(Some("1"), || {
            let storage = create_test_storage();
            let pinned = seed_node(
                &storage,
                "Secrets are stored in the vault, never in the repo",
                &["sec"],
                axis_vector(33, 0.02),
            );
            let other = seed_node(
                &storage,
                "Secrets live in the vault and are never committed to the repo",
                &["sec"],
                axis_vector(33, 0.01),
            );
            storage.set_protected(&pinned, true).unwrap();
            let (cp, co) = (
                storage.get_node(&pinned).unwrap().unwrap().content,
                storage.get_node(&other).unwrap().unwrap().content,
            );

            let merged = storage.auto_dedup_consolidation().unwrap();
            assert_eq!(merged, 0, "a lone unprotected node cannot form a cluster");
            assert_eq!(storage.get_node(&pinned).unwrap().unwrap().content, cp);
            assert_eq!(storage.get_node(&other).unwrap().unwrap().content, co);
        });
    }

    // --- D. Liveness: protected + two unprotected → the two merge -----------
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_auto_dedup_protected_plus_two_unprotected_liveness() {
        with_auto_merge_env(Some("1"), || {
            let storage = create_test_storage();
            // The pin exclusion must not block a legitimate merge of the others.
            let pinned = seed_node(
                &storage,
                "The API returns ISO-8601 timestamps in UTC",
                &["api"],
                axis_vector(35, 0.02),
            );
            let keeper = seed_node(
                &storage,
                "The API returns ISO 8601 timestamps in UTC by convention",
                &["api"],
                axis_vector(35, 0.01),
            );
            let member = seed_node(
                &storage,
                "All API timestamps are returned as ISO-8601 in the UTC timezone",
                &["api"],
                axis_vector(35, 0.015),
            );
            set_retention(&storage, &pinned, 0.50);
            set_retention(&storage, &keeper, 0.80);
            set_retention(&storage, &member, 0.30);
            storage.set_protected(&pinned, true).unwrap();
            let pinned_content = storage.get_node(&pinned).unwrap().unwrap().content;

            let merged = storage.auto_dedup_consolidation().unwrap();
            assert_eq!(merged, 1, "the two unprotected near-dups still merge");
            assert!(storage.get_node(&member).unwrap().is_none());
            let keeper_node = storage.get_node(&keeper).unwrap().unwrap();
            assert!(keeper_node.content.contains("[MERGED]"));
            // Protected node untouched.
            assert_eq!(
                storage.get_node(&pinned).unwrap().unwrap().content,
                pinned_content
            );
            assert!(storage.is_protected(&pinned).unwrap());
        });
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_apply_requires_confirm_for_low_confidence() {
        let storage = create_test_storage();
        // Tighten thresholds so a moderate pair lands in 'possible' (needs confirm).
        let strict = MergePolicy::new(0.99, 0.5, false);
        storage.set_merge_policy(strict).unwrap();

        let a = seed_node(&storage, "Topic alpha note", &["t"], axis_vector(13, 0.30));
        let b = seed_node(&storage, "Topic alpha aside", &["t"], axis_vector(13, 0.60));
        let plan = storage
            .plan_merge(&[a, b], None, storage.get_merge_policy().unwrap())
            .unwrap();
        assert_ne!(plan.classification, MatchClass::Match);

        // Without confirm => rejected.
        assert!(storage.apply_plan(&plan.id, false).is_err());
        // With confirm => applied.
        assert!(storage.apply_plan(&plan.id, true).is_ok());
        // Re-applying an applied plan => rejected.
        assert!(storage.apply_plan(&plan.id, true).is_err());
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn test_merge_policy_roundtrip_persists() {
        let storage = create_test_storage();
        let p = MergePolicy::new(0.9, 0.6, true);
        storage.set_merge_policy(p).unwrap();
        let got = storage.get_merge_policy().unwrap();
        assert!((got.match_threshold - 0.9).abs() < 1e-6);
        assert!((got.possible_threshold - 0.6).abs() < 1e-6);
        assert!(got.auto_apply);
    }

    #[test]
    fn test_set_protected_unknown_node_errors() {
        let storage = create_test_storage();
        assert!(storage.set_protected("does-not-exist", true).is_err());
    }

    #[test]
    fn trait_hybrid_search_multi_word_via_insert() {
        // Verify that hybrid_search finds records inserted via the trait insert()
        // even when no embedding is present (keyword path via terms matching).
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            let rec = make_record("quantum entanglement superposition physics");
            s.insert(&rec).await.unwrap();
            let results = s.hybrid_search("quantum physics", 10, 0.3, 0.7).unwrap();
            assert!(
                !results.is_empty(),
                "hybrid_search must find record containing 'quantum' and 'physics'"
            );
        });
    }

    #[test]
    fn trait_scheduling_round_trip() {
        let s = create_test_storage();
        let rec = make_record("fsrs scheduling test");
        let id = rec.id;
        let rt = rt();
        rt.block_on(async {
            s.insert(&rec).await.unwrap();
            let state = SchedulingState {
                memory_id: id,
                stability: 5.0,
                difficulty: 0.4,
                retrievability: 0.8,
                last_review: Some(chrono::Utc::now()),
                next_review: Some(chrono::Utc::now() + chrono::Duration::days(7)),
                reps: 3,
                lapses: 1,
            };
            s.update_scheduling(&state).await.unwrap();
            let got = s.get_scheduling(id).await.unwrap().unwrap();
            assert!((got.stability - 5.0).abs() < 0.01);
        });
    }

    #[test]
    fn trait_get_scheduling_missing_returns_none() {
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            let got = s.get_scheduling(uuid::Uuid::new_v4()).await.unwrap();
            assert!(got.is_none());
        });
    }

    #[test]
    fn trait_get_due_memories_returns_in_order() {
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            for i in 0..3usize {
                let rec = make_record(&format!("due memory {i}"));
                let id = rec.id;
                s.insert(&rec).await.unwrap();
                let state = SchedulingState {
                    memory_id: id,
                    stability: 1.0,
                    difficulty: 0.3,
                    retrievability: 0.5,
                    last_review: Some(chrono::Utc::now()),
                    next_review: Some(chrono::Utc::now() - chrono::Duration::days(3 - i as i64)),
                    reps: 1,
                    lapses: 0,
                };
                s.update_scheduling(&state).await.unwrap();
            }
            let due = s.get_due_memories(chrono::Utc::now(), 10).await.unwrap();
            assert_eq!(due.len(), 3);
        });
    }

    #[test]
    fn trait_add_edge_is_idempotent() {
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            let rec_a = make_record("node a");
            let rec_b = make_record("node b");
            let id_a = rec_a.id;
            let id_b = rec_b.id;
            s.insert(&rec_a).await.unwrap();
            s.insert(&rec_b).await.unwrap();
            let edge = MemoryEdge {
                source_id: id_a,
                target_id: id_b,
                edge_type: "semantic".to_string(),
                weight: 0.9,
                created_at: chrono::Utc::now(),
            };
            s.add_edge(&edge).await.unwrap();
            s.add_edge(&edge).await.unwrap(); // idempotent
            let edges = s.get_edges(id_a, None).await.unwrap();
            let filtered: Vec<_> = edges
                .iter()
                .filter(|e| e.source_id == id_a && e.target_id == id_b)
                .collect();
            assert_eq!(filtered.len(), 1, "edge must not be duplicated");
        });
    }

    #[test]
    fn trait_get_edges_filters_by_type() {
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            let rec_a = make_record("filter a");
            let rec_b = make_record("filter b");
            let id_a = rec_a.id;
            let id_b = rec_b.id;
            s.insert(&rec_a).await.unwrap();
            s.insert(&rec_b).await.unwrap();
            let edge = MemoryEdge {
                source_id: id_a,
                target_id: id_b,
                edge_type: "causal".to_string(),
                weight: 0.5,
                created_at: chrono::Utc::now(),
            };
            s.add_edge(&edge).await.unwrap();
            let causal = s.get_edges(id_a, Some("causal")).await.unwrap();
            assert!(!causal.is_empty());
            let semantic = s.get_edges(id_a, Some("semantic")).await.unwrap();
            assert!(semantic.is_empty());
        });
    }

    #[test]
    fn trait_remove_edge_deletes_single() {
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            let rec_a = make_record("rm edge a");
            let rec_b = make_record("rm edge b");
            let id_a = rec_a.id;
            let id_b = rec_b.id;
            s.insert(&rec_a).await.unwrap();
            s.insert(&rec_b).await.unwrap();
            let edge = MemoryEdge {
                source_id: id_a,
                target_id: id_b,
                edge_type: "semantic".to_string(),
                weight: 0.7,
                created_at: chrono::Utc::now(),
            };
            s.add_edge(&edge).await.unwrap();
            s.remove_edge(id_a, id_b).await.unwrap();
            let edges = s.get_edges(id_a, None).await.unwrap();
            assert!(edges.is_empty());
        });
    }

    #[test]
    fn trait_get_neighbors_bfs_depth_zero_returns_self_only() {
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            let rec = make_record("depth zero");
            let id = rec.id;
            s.insert(&rec).await.unwrap();
            let neighbors = s.get_neighbors(id, 0).await.unwrap();
            assert_eq!(neighbors.len(), 1);
            assert_eq!(neighbors[0].0.id, id);
        });
    }

    #[test]
    fn trait_get_neighbors_bfs_depth_two_expands() {
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            let rec_a = make_record("bfs node a");
            let rec_b = make_record("bfs node b");
            let rec_c = make_record("bfs node c");
            let id_a = rec_a.id;
            let id_b = rec_b.id;
            let id_c = rec_c.id;
            s.insert(&rec_a).await.unwrap();
            s.insert(&rec_b).await.unwrap();
            s.insert(&rec_c).await.unwrap();
            s.add_edge(&MemoryEdge {
                source_id: id_a,
                target_id: id_b,
                edge_type: "semantic".to_string(),
                weight: 1.0,
                created_at: chrono::Utc::now(),
            })
            .await
            .unwrap();
            s.add_edge(&MemoryEdge {
                source_id: id_b,
                target_id: id_c,
                edge_type: "semantic".to_string(),
                weight: 1.0,
                created_at: chrono::Utc::now(),
            })
            .await
            .unwrap();
            let neighbors = s.get_neighbors(id_a, 2).await.unwrap();
            let ids: Vec<uuid::Uuid> = neighbors.iter().map(|(r, _)| r.id).collect();
            assert!(ids.contains(&id_a));
            assert!(ids.contains(&id_b));
            assert!(ids.contains(&id_c));
        });
    }

    #[test]
    fn trait_list_domains_empty_in_phase_1() {
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            let domains = s.list_domains().await.unwrap();
            assert!(domains.is_empty());
        });
    }

    #[test]
    fn trait_upsert_then_get_domain_round_trip() {
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            let domain = crate::storage::memory_store::Domain {
                id: "dev".to_string(),
                label: "Development".to_string(),
                centroid: vec![0.1, 0.2, 0.3],
                top_terms: vec!["rust".to_string(), "code".to_string()],
                memory_count: 42,
                created_at: chrono::Utc::now(),
            };
            s.upsert_domain(&domain).await.unwrap();
            let got = s.get_domain("dev").await.unwrap().unwrap();
            assert_eq!(got.id, "dev");
            assert_eq!(got.memory_count, 42);
        });
    }

    #[test]
    fn trait_delete_domain_idempotent() {
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            s.delete_domain("nonexistent").await.unwrap();
            s.delete_domain("nonexistent").await.unwrap();
        });
    }

    #[test]
    fn trait_classify_with_no_domains_returns_empty() {
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            let result = s.classify(&[0.1, 0.2, 0.3]).await.unwrap();
            assert!(result.is_empty());
        });
    }

    #[test]
    fn trait_count_matches_insert_count() {
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            for i in 0..5usize {
                let rec = make_record(&format!("count test {i}"));
                s.insert(&rec).await.unwrap();
            }
            assert_eq!(s.count().await.unwrap(), 5);
        });
    }

    #[test]
    fn trait_insert_rejects_secret_shaped_tags_and_source_without_a_row() {
        let s = create_test_storage();
        let credential = format!("ghp_{}", "A".repeat(36));
        let rt = rt();
        rt.block_on(async {
            let mut tagged = make_record("safe direct trait insert");
            tagged.tags = vec![credential.clone()];
            let err = s.insert(&tagged).await.unwrap_err();
            assert!(matches!(err, MemoryStoreError::SecretDetected(_)));
            assert!(!err.to_string().contains(&credential));
            assert_eq!(s.count().await.unwrap(), 0);

            let mut sourced = make_record("another safe direct trait insert");
            sourced.metadata = serde_json::json!({ "source": credential });
            let err = s.insert(&sourced).await.unwrap_err();
            assert!(matches!(err, MemoryStoreError::SecretDetected(_)));
            assert_eq!(s.count().await.unwrap(), 0);
        });
    }

    #[test]
    fn trait_get_stats_reports_registered_model() {
        let s = create_test_storage();
        let sig = ModelSignature {
            name: "test-model".to_string(),
            dimension: 256,
            hash: "c".repeat(64),
        };
        let rt = rt();
        rt.block_on(async {
            use crate::storage::memory_store::MemoryStore;
            // Cast to &dyn MemoryStore so the async trait method is called
            // instead of the inherent sync get_stats() on SqliteMemoryStore.
            let dyn_s: &dyn MemoryStore = &s;
            dyn_s.register_model(&sig).await.unwrap();
            let stats = dyn_s.get_stats().await.unwrap();
            assert_eq!(stats.registered_model_name, Some("test-model".to_string()));
            assert_eq!(stats.registered_model_dim, Some(256));
        });
    }

    #[test]
    fn trait_vacuum_succeeds() {
        let s = create_test_storage();
        let rt = rt();
        rt.block_on(async {
            s.vacuum().await.unwrap();
        });
    }

    #[test]
    fn trait_insert_refuses_dimension_mismatch() {
        let s = create_test_storage();
        let sig = ModelSignature {
            name: "test-model".to_string(),
            dimension: 256,
            hash: "d".repeat(64),
        };
        let rt = rt();
        rt.block_on(async {
            s.register_model(&sig).await.unwrap();
            // Build a record with wrong dimension (512 instead of 256) and
            // declare the model signature in metadata
            let mut rec = make_record("dimension mismatch");
            rec.embedding = Some(vec![0.0f32; 512]);
            rec.metadata = serde_json::json!({
                "model_name": "test-model",
                "model_dim": 256_u64,
                "model_hash": "d".repeat(64),
            });
            let err = s.insert(&rec).await.unwrap_err();
            assert!(
                matches!(err, MemoryStoreError::InvalidInput(_)),
                "expected InvalidInput, got {:?}",
                err
            );
        });
    }

    // Seed a node's stability directly via the scheduling seam so the +365 cap
    // in promote_memory_backfill is actually exercised (a freshly ingested node
    // has low stability where the *1.5 multiply, not the additive ceiling, wins).
    fn seed_stability(s: &Storage, id: &str, stability: f64) {
        use crate::storage::memory_store::{MemoryStoreSend, SchedulingState};
        rt().block_on(async {
            let state = SchedulingState {
                memory_id: uuid::Uuid::parse_str(id).unwrap(),
                stability,
                difficulty: 0.4,
                retrievability: 0.8,
                last_review: Some(chrono::Utc::now()),
                next_review: Some(chrono::Utc::now() + chrono::Duration::days(7)),
                reps: 3,
                lapses: 0,
            };
            MemoryStoreSend::update_scheduling(s, &state).await.unwrap();
        });
    }

    #[test]
    fn promote_memory_backfill_caps_stability_at_plus_365() {
        // Above the crossover (stability=730) the additive +365 ceiling must win
        // over the *1.5 multiply, so repeated backfill promotions cannot inflate
        // stability without bound. This is the bound issue #103 asked us to apply.
        let s = create_test_storage();
        let node = s
            .ingest(IngestInput {
                content: "high-stability cause memory".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        seed_stability(&s, &node.id, 1000.0);

        let promoted = s.promote_memory_backfill(&node.id).unwrap();
        // 1000 * 1.5 = 1500 (uncapped) vs 1000 + 365 = 1365 (capped). Cap wins.
        assert!(
            (promoted.stability - 1365.0).abs() < 1e-6,
            "expected additive +365 cap (1365.0), got {} (uncapped would be 1500.0)",
            promoted.stability
        );
    }

    #[test]
    fn promote_memory_backfill_uses_multiply_below_crossover() {
        // Below the crossover the *1.5 multiply wins (the cap never binds), so
        // backfill promotion strength is unchanged from the old promote_memory.
        let s = create_test_storage();
        let node = s
            .ingest(IngestInput {
                content: "low-stability cause memory".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        seed_stability(&s, &node.id, 10.0);

        let promoted = s.promote_memory_backfill(&node.id).unwrap();
        // 10 * 1.5 = 15 (multiply) vs 10 + 365 = 375 (cap). Multiply wins.
        assert!(
            (promoted.stability - 15.0).abs() < 1e-6,
            "expected *1.5 multiply (15.0) below crossover, got {}",
            promoted.stability
        );
    }

    #[test]
    fn suppress_then_reverse_restores_fsrs_state() {
        // reverse_suppression must be a TRUE inverse of suppress_memory. Suppress
        // applies stability*0.4, retrieval-0.35, retention-0.20; reverse now undoes
        // exactly that (stability/0.4, retrieval+0.35, retention+0.20). Previously
        // reverse used non-inverse deltas and left stability permanently halved.
        let s = create_test_storage();
        let node = s
            .ingest(IngestInput {
                content: "a memory to suppress then un-suppress".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        // Seed above the 0.05 floor so the forward pass never clips (making the
        // round-trip exactly recoverable).
        seed_stability(&s, &node.id, 20.0);
        let before = s.get_node(&node.id).unwrap().unwrap();

        s.suppress_memory(&node.id).unwrap();
        let suppressed = s.get_node(&node.id).unwrap().unwrap();
        assert!(
            (suppressed.stability - before.stability * 0.4).abs() < 1e-6,
            "suppress must multiply stability by 0.4"
        );

        let reversed = s.reverse_suppression(&node.id, 24).unwrap();
        // stability: 20 * 0.4 / 0.4 = 20 (fully restored, not 0.5x)
        assert!(
            (reversed.stability - before.stability).abs() < 1e-6,
            "reverse must restore stability to {} (got {})",
            before.stability,
            reversed.stability
        );
        assert!(
            (reversed.retrieval_strength - before.retrieval_strength).abs() < 1e-6,
            "reverse must restore retrieval_strength"
        );
        assert!(
            (reversed.retention_strength - before.retention_strength).abs() < 1e-6,
            "reverse must restore retention_strength"
        );
    }

    #[test]
    fn backfill_autofire_gate_defaults_on_and_reads_opt_out() {
        // v2.2.1 opt-out semantics: unset => ON (preserves shipped v2.2.0
        // behavior); explicit 0/false/off/no => OFF; anything else => ON.
        fn parse(v: Option<&str>) -> bool {
            v.map(|v| {
                let v = v.trim();
                !(v.eq_ignore_ascii_case("false")
                    || v.eq_ignore_ascii_case("off")
                    || v.eq_ignore_ascii_case("no")
                    || v == "0")
            })
            .unwrap_or(true)
        }
        assert!(parse(None), "unset must default ON");
        assert!(parse(Some("1")), "1 is ON");
        assert!(parse(Some("true")), "true is ON");
        assert!(parse(Some("anything")), "unrecognized is ON");
        assert!(!parse(Some("0")), "0 is OFF");
        assert!(!parse(Some("false")), "false is OFF");
        assert!(!parse(Some("OFF")), "OFF (case-insensitive) is OFF");
        assert!(!parse(Some(" no ")), "whitespace-padded no is OFF (trim)");
    }

    // =====================================================================
    // Smart-ingest bitemporal validity gates (issue #156)
    // =====================================================================

    /// Marker-keyed test embedder: contents sharing the "alpha" marker embed
    /// to the same axis (cosine similarity 1.0) and everything else lands on
    /// the orthogonal axis, so gate decisions are fully controlled by content.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    struct MarkerEmbedder;

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    impl crate::embedder::EmbedderSend for MarkerEmbedder {
        async fn embed(&self, text: &str) -> crate::embedder::EmbedderResult<Vec<f32>> {
            Ok(if text.contains("alpha") {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            })
        }
        fn model_name(&self) -> &str {
            "marker-gate-test-runner"
        }
        fn dimension(&self) -> usize {
            2
        }
        fn model_hash(&self) -> String {
            "0".repeat(64)
        }
        async fn embed_batch(
            &self,
            texts: &[&str],
        ) -> crate::embedder::EmbedderResult<Vec<Vec<f32>>> {
            let mut result = Vec::with_capacity(texts.len());
            for text in texts {
                result.push(if text.contains("alpha") {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                });
            }
            Ok(result)
        }
    }

    /// Install, promote, activate, and attach a verified 2-dimensional marker
    /// profile so smart-ingest gate decisions run end to end without a model
    /// download. Mirrors the lifecycle install/evaluate/migrate/activate flow;
    /// the Ready promotion is applied directly to the persisted manifest since
    /// Regression: a purge (writer lock, then vector-index lock) racing a
    /// profile activation (index lock, then writer lock) deadlocked the process
    /// before the Sep 2026 hardening pass because `purge_node` kept its writer
    /// guard alive while it waited on the index. Both paths must run to
    /// completion under a watchdog.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn purge_and_profile_activation_do_not_deadlock() {
        use std::sync::mpsc;
        use std::time::Duration;

        let dir = tempfile::tempdir().unwrap();
        let storage = std::sync::Arc::new(storage_with_marker_gate_runtime(&dir));
        let ids: Vec<String> = (0..48)
            .map(|i| {
                storage
                    .ingest(IngestInput {
                        content: format!("alpha purge race memory number {i}"),
                        ..Default::default()
                    })
                    .unwrap()
                    .id
            })
            .collect();
        let active = storage
            .active_embedding_profile()
            .unwrap()
            .expect("marker fixture activates a profile");

        let (done_tx, done_rx) = mpsc::channel::<&'static str>();
        let purger = {
            let storage = std::sync::Arc::clone(&storage);
            let done_tx = done_tx.clone();
            std::thread::spawn(move || {
                for id in ids {
                    storage.purge_node(&id, Some("lock-order race")).unwrap();
                }
                let _ = done_tx.send("purge");
            })
        };
        let activator = {
            let storage = std::sync::Arc::clone(&storage);
            std::thread::spawn(move || {
                for _ in 0..48 {
                    storage
                        .activate_embedding_profile(&active.profile_id)
                        .unwrap();
                }
                let _ = done_tx.send("activate");
            })
        };
        for _ in 0..2 {
            done_rx.recv_timeout(Duration::from_secs(30)).expect(
                "purge vs profile activation deadlocked (writer->index vs index->writer)",
            );
        }
        purger.join().unwrap();
        activator.join().unwrap();
    }

    /// the process-local registry is private to the lifecycle module.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn storage_with_marker_gate_runtime(dir: &tempfile::TempDir) -> Storage {
        use crate::embedding::{
            ChunkingStrategy, EmbeddingDevice, EmbeddingEvaluationSummary, EmbeddingNormalization,
            EmbeddingProfile, EmbeddingProfileLifecycle, EmbeddingRuntimeMetadata,
            EncodingTemplate, ModelArtifactHash, VerifiedLocalArtifact,
        };
        use sha2::{Digest, Sha256};

        let storage = create_test_storage_at(dir, "marker-gate.db");
        let artifact_bytes: &[u8] = b"marker gate test artifact";
        std::fs::write(dir.path().join("runner.bin"), artifact_bytes).unwrap();
        let artifact = ModelArtifactHash::sha256(
            "runner.bin",
            format!("{:x}", Sha256::digest(artifact_bytes)),
        );
        let profile = EmbeddingProfile {
            profile_id: EmbeddingProfileId::new("marker-gate-test-2d").unwrap(),
            display_name: "Marker Gate Test Profile".to_string(),
            model_id: "test/marker-gate".to_string(),
            immutable_model_revision: "immutable-test-revision".to_string(),
            verified_model_artifact_hashes: vec![artifact.clone()],
            runtime_backend: EmbeddingRuntimeBackend::FastembedCandle,
            embedding_dimension: 2,
            normalization_method: EmbeddingNormalization::L2,
            document_encoding_template: EncodingTemplate::Raw,
            query_encoding_template: EncodingTemplate::Raw,
            maximum_token_limit: 64,
            chunking_strategy: ChunkingStrategy::WholeDocument,
            created_at: Utc::now(),
        };
        let artifacts = vec![VerifiedLocalArtifact::from_root(artifact, dir.path()).unwrap()];
        let source = EmbeddingProfileId::new("nomic-v1.5-legacy-raw-256").unwrap();
        let lifecycle = EmbeddingProfileLifecycle::new(&storage);
        let mut manifest = lifecycle
            .install_verified(
                profile.clone(),
                &artifacts,
                EmbeddingRuntimeMetadata {
                    backend: EmbeddingRuntimeBackend::FastembedCandle,
                    device: EmbeddingDevice::Cpu,
                    runtime_version: "test".to_string(),
                    initialized_at: Utc::now(),
                    local_only: true,
                },
                Arc::new(MarkerEmbedder),
            )
            .unwrap();
        manifest.state = EmbeddingProfileState::Ready;
        manifest.evaluation = Some(EmbeddingEvaluationSummary {
            evaluation_id: Uuid::new_v4(),
            compared_against: source.clone(),
            completed_at: Utc::now(),
            corpus_size: 0,
            recall_at_5: None,
            recall_at_10: None,
            ndcg_at_10: None,
            exact_match_preservation: None,
            false_positive_rate: None,
            p50_query_latency_ms: None,
            p95_query_latency_ms: None,
            ingestion_throughput_per_second: None,
            report_hash: "1".repeat(64),
        });
        storage.save_embedding_profile_manifest(&manifest).unwrap();
        lifecycle
            .migrate_registered(&profile.profile_id, &source, None, None)
            .unwrap();
        storage
            .activate_embedding_profile(&profile.profile_id)
            .unwrap();
        lifecycle
            .attach_registered_active_profile(&profile.profile_id)
            .unwrap();
        storage
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn inferred_as_of_validity_never_mutates_an_existing_nodes_window() {
        let dir = tempdir().unwrap();
        let storage = storage_with_marker_gate_runtime(&dir);
        let explicit_from = Utc::now() - Duration::days(60);
        let explicit_until = Utc::now() + Duration::days(300);
        let target = storage
            .ingest(IngestInput {
                content: "alpha deployment policy is retries with backoff".to_string(),
                valid_from: Some(explicit_from),
                valid_until: Some(explicit_until),
                ..Default::default()
            })
            .unwrap();

        // Near-identical content whose only validity is a prose-inferred
        // "as of" date must reinforce WITHOUT touching the target's window.
        let result = storage
            .smart_ingest(IngestInput {
                content: "alpha deployment policy is retries with backoff, as of last review"
                    .to_string(),
                valid_from: Some(Utc::now() - Duration::days(10)),
                validity_inferred: true,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(result.decision, "reinforce");
        assert_eq!(result.node.id, target.id);

        let node = storage.get_node(&target.id).unwrap().unwrap();
        assert_eq!(
            node.valid_from.map(|value| value.to_rfc3339()),
            Some(explicit_from.to_rfc3339()),
            "an inferred date must not move valid_from"
        );
        assert_eq!(
            node.valid_until.map(|value| value.to_rfc3339()),
            Some(explicit_until.to_rfc3339()),
            "an inferred date must not clear or move valid_until"
        );
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn inferred_as_of_must_not_resurrect_an_expired_similar_node() {
        let dir = tempdir().unwrap();
        let storage = storage_with_marker_gate_runtime(&dir);
        let expired_from = Utc::now() - Duration::days(90);
        let expired_until = Utc::now() - Duration::days(7);
        let target = storage
            .ingest(IngestInput {
                content: "alpha deployment policy is retries with backoff".to_string(),
                valid_from: Some(expired_from),
                valid_until: Some(expired_until),
                ..Default::default()
            })
            .unwrap();
        assert!(
            !storage
                .get_node(&target.id)
                .unwrap()
                .unwrap()
                .is_currently_valid(),
            "fixture must start expired"
        );

        // Live #170 failure: inferred validFrom + PEG update used to
        // REPLACE valid_until with NULL and un-expire the similar row.
        let result = storage
            .smart_ingest(IngestInput {
                content: "alpha deployment policy is retries with backoff as of 2026-03-04"
                    .to_string(),
                valid_from: Some(Utc::now() - Duration::days(10)),
                validity_inferred: true,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(result.decision, "reinforce");
        assert_eq!(result.node.id, target.id);

        let node = storage.get_node(&target.id).unwrap().unwrap();
        assert_eq!(
            node.valid_from.map(|value| value.to_rfc3339()),
            Some(expired_from.to_rfc3339()),
            "inferred as-of must not move the expired node's valid_from"
        );
        assert_eq!(
            node.valid_until.map(|value| value.to_rfc3339()),
            Some(expired_until.to_rfc3339()),
            "inferred as-of must not un-expire the similar node"
        );
        assert!(!node.is_currently_valid());
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn explicit_valid_from_on_reinforce_updates_without_clearing_valid_until() {
        let dir = tempdir().unwrap();
        let storage = storage_with_marker_gate_runtime(&dir);
        let explicit_from = Utc::now() - Duration::days(60);
        let explicit_until = Utc::now() + Duration::days(300);
        let target = storage
            .ingest(IngestInput {
                content: "alpha deployment policy is retries with backoff".to_string(),
                valid_from: Some(explicit_from),
                valid_until: Some(explicit_until),
                ..Default::default()
            })
            .unwrap();

        // An explicit caller-supplied valid_from still updates the window, and
        // omitting valid_until must preserve the stored bound (merge, not
        // replace).
        let new_from = Utc::now() - Duration::days(10);
        let result = storage
            .smart_ingest(IngestInput {
                content: "alpha deployment policy is retries with backoff".to_string(),
                valid_from: Some(new_from),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(result.decision, "reinforce");

        let node = storage.get_node(&target.id).unwrap().unwrap();
        assert_eq!(
            node.valid_from.map(|value| value.to_rfc3339()),
            Some(new_from.to_rfc3339()),
            "explicit valid_from must update the window"
        );
        assert_eq!(
            node.valid_until.map(|value| value.to_rfc3339()),
            Some(explicit_until.to_rfc3339()),
            "an explicit valid_from-only update must not NULL valid_until"
        );
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn create_path_still_stamps_inferred_validity_on_the_new_node() {
        let dir = tempdir().unwrap();
        let storage = storage_with_marker_gate_runtime(&dir);
        let inferred_from = Utc::now() - Duration::days(10);
        let result = storage
            .smart_ingest(IngestInput {
                content: "beta service quota was raised".to_string(),
                valid_from: Some(inferred_from),
                validity_inferred: true,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(result.decision, "create");
        assert!(result.auto_closed_until.is_none());
        let node = storage.get_node(&result.node.id).unwrap().unwrap();
        assert_eq!(
            node.valid_from.map(|value| value.to_rfc3339()),
            Some(inferred_from.to_rfc3339()),
            "the issue-156 feature: inferred validity stamps the NEW node"
        );
        assert!(node.valid_until.is_none());
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn older_dated_claim_after_newer_fact_is_created_with_a_closed_window() {
        let dir = tempdir().unwrap();
        let storage = storage_with_marker_gate_runtime(&dir);
        let newer_from = Utc::now() - Duration::days(30);
        storage
            .ingest(IngestInput {
                content: "alpha runtime version is 5.0".to_string(),
                valid_from: Some(newer_from),
                ..Default::default()
            })
            .unwrap();

        // The stale snapshot loses every candidate to the newer current fact,
        // so it is created as a CLOSED historical claim, never an open one.
        let result = storage
            .smart_ingest(IngestInput {
                content: "alpha runtime version is 4.2".to_string(),
                valid_from: Some(Utc::now() - Duration::days(180)),
                validity_inferred: true,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(result.decision, "create");
        assert_eq!(
            result.auto_closed_until.map(|value| value.to_rfc3339()),
            Some(newer_from.to_rfc3339()),
            "the new node closes exactly where the newer fact begins"
        );
        assert!(result.reason.contains("Closed validity at"));
        let node = storage.get_node(&result.node.id).unwrap().unwrap();
        assert_eq!(
            node.valid_until.map(|value| value.to_rfc3339()),
            Some(newer_from.to_rfc3339())
        );
        assert!(!node.is_currently_valid());

        // Reverse order on a fresh store: the newer dated claim arriving
        // second must NOT auto-close anything. That is this half's subject.
        //
        // It previously also asserted `reinforce`, which encoded a defect. The
        // pair here ("version is 4.2" against "version is 5.0") is a mutually
        // exclusive VALUE conflict, the same shape as PostgreSQL 14 -> 16, and
        // the write-path detector could not see it: short numeric tokens were
        // dropped by the substantive-word length filter, so the two texts
        // looked identical and the gate reinforced on similarity alone. The
        // effect was that telling Vestige "the version is now 5.0" discarded
        // that update and made it believe 4.2 MORE strongly. The gate now
        // keeps both claims. See advanced::contradiction.
        let dir = tempdir().unwrap();
        let storage = storage_with_marker_gate_runtime(&dir);
        let target = storage
            .ingest(IngestInput {
                content: "alpha runtime version is 4.2".to_string(),
                valid_from: Some(Utc::now() - Duration::days(180)),
                ..Default::default()
            })
            .unwrap();
        let result = storage
            .smart_ingest(IngestInput {
                content: "alpha runtime version is 5.0".to_string(),
                valid_from: Some(newer_from),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            result.decision, "create",
            "a version change is a value conflict, not a reinforcement"
        );
        assert_ne!(
            result.node.id, target.id,
            "the superseded 4.2 claim must not be overwritten in place"
        );
        assert!(
            result.auto_closed_until.is_none(),
            "a newer dated claim arriving second closes nothing"
        );
        assert_eq!(
            result.node.valid_from.map(|value| value.to_rfc3339()),
            Some(newer_from.to_rfc3339()),
            "the new node carries its own validity"
        );
        // The older claim survives intact, still open, still retrievable.
        let previous = storage.get_node(&target.id).unwrap().unwrap();
        assert!(previous.valid_until.is_none());
        assert!(previous.content.contains("4.2"));
    }

    #[test]
    fn update_node_validity_merges_bounds_and_validates_the_effective_window() {
        let storage = create_test_storage();
        let node = storage
            .ingest(IngestInput {
                content: "validity merge fixture".to_string(),
                ..Default::default()
            })
            .unwrap();
        let from = Utc::now() - Duration::days(10);
        let until = Utc::now() + Duration::days(10);
        storage
            .update_node_validity(&node.id, Some(from), Some(until))
            .unwrap();

        // Updating only valid_from must not clear the stored valid_until.
        let new_from = Utc::now() - Duration::days(5);
        storage
            .update_node_validity(&node.id, Some(new_from), None)
            .unwrap();
        let stored = storage.get_node(&node.id).unwrap().unwrap();
        assert_eq!(
            stored.valid_from.map(|value| value.to_rfc3339()),
            Some(new_from.to_rfc3339())
        );
        assert_eq!(
            stored.valid_until.map(|value| value.to_rfc3339()),
            Some(until.to_rfc3339())
        );

        // Updating only valid_until must not clear the stored valid_from.
        let new_until = Utc::now() + Duration::days(20);
        storage
            .update_node_validity(&node.id, None, Some(new_until))
            .unwrap();
        let stored = storage.get_node(&node.id).unwrap().unwrap();
        assert_eq!(
            stored.valid_from.map(|value| value.to_rfc3339()),
            Some(new_from.to_rfc3339())
        );
        assert_eq!(
            stored.valid_until.map(|value| value.to_rfc3339()),
            Some(new_until.to_rfc3339())
        );

        // The EFFECTIVE post-merge window is validated: a valid_from at or
        // beyond the stored valid_until is rejected and nothing changes.
        let error = storage
            .update_node_validity(&node.id, Some(Utc::now() + Duration::days(30)), None)
            .unwrap_err();
        assert!(matches!(error, StorageError::InvalidTimestamp(_)));
        let stored = storage.get_node(&node.id).unwrap().unwrap();
        assert_eq!(
            stored.valid_from.map(|value| value.to_rfc3339()),
            Some(new_from.to_rfc3339())
        );
        assert_eq!(
            stored.valid_until.map(|value| value.to_rfc3339()),
            Some(new_until.to_rfc3339())
        );

        // Both bounds supplied together are still validated up front.
        let error = storage
            .update_node_validity(&node.id, Some(until), Some(from))
            .unwrap_err();
        assert!(matches!(error, StorageError::InvalidTimestamp(_)));
    }
}

/// Policy lint: every writer transaction in this file must begin IMMEDIATE.
///
/// A DEFERRED transaction that reads before it writes can fail with
/// `SQLITE_BUSY_SNAPSHOT` the moment another process (the CLI next to the MCP
/// server) commits in between, and SQLite does not consult the busy handler
/// for that upgrade. `BEGIN IMMEDIATE` takes the write lock up front, where
/// `busy_timeout` applies, and SQLite then guarantees no `SQLITE_BUSY` until
/// COMMIT. Read-only transactions on the reader connection stay DEFERRED.
#[cfg(test)]
mod write_transaction_policy {
    /// Every storage module that can open a transaction on a writer
    /// connection. The rule is module-wide, not file-wide: the first version
    /// of this lint read `sqlite.rs` alone, and two writers drifted DEFERRED
    /// in the blind spot (`trace_store.rs`'s memory-PR decide path, and this
    /// file's own `unchecked_transaction` in the open-time FK repair).
    const STORAGE_SOURCES: [(&str, &str); 9] = [
        ("sqlite.rs", include_str!("sqlite.rs")),
        ("migrations.rs", include_str!("migrations.rs")),
        ("trace_store.rs", include_str!("trace_store.rs")),
        ("synaptic_store.rs", include_str!("synaptic_store.rs")),
        ("replay_store.rs", include_str!("replay_store.rs")),
        ("attestation_store.rs", include_str!("attestation_store.rs")),
        ("unlearning_store.rs", include_str!("unlearning_store.rs")),
        ("memory_store.rs", include_str!("memory_store.rs")),
        ("portable.rs", include_str!("portable.rs")),
    ];

    /// Modules whose writers must additionally route through the shared
    /// helper, so a BUSY past the busy timeout is retried and logged rather
    /// than surfacing to the caller on the first refusal. Beginning IMMEDIATE
    /// by hand is correct but silent: it takes the write lock up front and
    /// then gives up on the first refusal past the 5 s busy timeout, with
    /// nothing in the log to say a writer lost a race.
    ///
    /// The needle matches the single-line form production used (the guard
    /// receiver and the behaviour call on one line). Test fixtures that
    /// genuinely need to drive a transaction by hand (a rollback or
    /// lock-contention harness on their own connection) build it across lines
    /// and are deliberately not caught. Note this comment cannot spell the
    /// needle out: the lint reads this file, so a literal spelling would flag
    /// itself, which is exactly what it did on the first draft of this text.
    const HELPER_ROUTED: [&str; 5] = [
        "sqlite.rs",
        "trace_store.rs",
        "synaptic_store.rs",
        "replay_store.rs",
        "attestation_store.rs",
    ];

    /// Production transactions propagate with `?`; test fixtures `.unwrap()`
    /// on their own in-memory connections. The `?` suffix is what separates
    /// the two here, and it is the convention the storage layer already uses.
    #[test]
    fn writer_transactions_begin_immediate() {
        // Assembled at runtime so this lint never matches its own source lines.
        let deferred_writer = ["writer.", "transaction()?"].concat();
        let deferred_unchecked = ["unchecked_", "transaction()?"].concat();
        let bypasses_helper = ["writer.", "transaction_with_behavior("].concat();
        let snapshot_on_writer = ["begin_read_", "snapshot(&writer"].concat();

        let mut offenders: Vec<String> = Vec::new();
        for (name, source) in STORAGE_SOURCES {
            for (index, line) in source.lines().enumerate() {
                let number = index + 1;
                if line.contains(&deferred_writer) || line.contains(&deferred_unchecked) {
                    offenders.push(format!(
                        "{name}:{number} opens a DEFERRED writer transaction; a read-then-write \
                         DEFERRED transaction can fail with SQLITE_BUSY_SNAPSHOT and SQLite does \
                         not consult the busy handler for that upgrade"
                    ));
                }
                if HELPER_ROUTED.contains(&name) && line.contains(&bypasses_helper) {
                    offenders.push(format!(
                        "{name}:{number} opens a writer transaction directly; use \
                         SqliteMemoryStore::begin_write_transaction so BUSY retries are logged"
                    ));
                }
                if line.contains(&snapshot_on_writer) {
                    offenders.push(format!(
                        "{name}:{number} opens a DEFERRED read snapshot on the writer connection; \
                         snapshots belong on the reader, writers begin IMMEDIATE"
                    ));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "writer transactions must begin IMMEDIATE:\n{}",
            offenders.join("\n")
        );
    }

    #[test]
    fn the_write_transaction_helper_exists_and_is_shared() {
        let source = include_str!("sqlite.rs");
        assert!(
            source.contains("fn begin_write_transaction"),
            "the write-transaction helper must exist"
        );
        // Sibling storage modules are not descendants of this one, so the
        // helper has to stay at least `pub(super)` for them to reach it.
        assert!(
            source.contains(["pub(super) fn begin_write_", "transaction"].concat().as_str()),
            "the helper must stay visible to sibling storage modules"
        );
    }
}
