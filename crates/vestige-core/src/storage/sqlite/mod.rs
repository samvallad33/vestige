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

impl SqliteMemoryStore {
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

    /// Only local suppression state is reversible; cascade effects are separate.
    fn suppression_state_on(conn: &Connection, id: &str) -> Result<String> {
        let state: (i64, Option<String>, f64, f64, f64) = conn
            .query_row(
                "SELECT COALESCE(suppression_count, 0), suppressed_at,
             retrieval_strength, retention_strength, stability FROM knowledge_nodes WHERE id = ?1",
                params![id],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))?;
        serde_json::to_string(&state).map_err(|error| StorageError::Init(error.to_string()))
    }

    /// Read a bounded namespace before pagination, including historical nodes.
    /// NULL/blank legacy scopes match user; other namespaces cannot crowd out
    /// this namespace's candidates by consuming the LIMIT first.
    pub fn get_all_nodes_in_scope(
        &self,
        scope: &str,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<KnowledgeNode>> {
        let scope = Self::normalize_scope(scope)?;
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT * FROM knowledge_nodes
             WHERE COALESCE(NULLIF(trim(scope), ''), 'user') = ?1
             ORDER BY created_at DESC, id ASC LIMIT ?2 OFFSET ?3",
        )?;
        let rows = stmt.query_map(
            params![scope, limit.clamp(1, 5000), offset.max(0)],
            Self::row_to_node,
        )?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    /// Current code advice, with exact project tags and an explicit namespace.
    /// Eligibility is applied before LIMIT so expired rows cannot crowd out
    /// current advice. Historical reads continue to use the existing APIs.
    pub fn current_code_context_nodes(
        &self,
        node_type: &str,
        tag: Option<&str>,
        scope: &str,
        limit: i32,
    ) -> Result<Vec<KnowledgeNode>> {
        let scope = Self::normalize_scope(scope)?;
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT n.* FROM knowledge_nodes n
             WHERE n.node_type = ?1 AND n.scope = ?2
               AND (?3 IS NULL OR EXISTS (
                   SELECT 1 FROM json_each(n.tags) t WHERE t.type = 'text' AND t.value = ?3))
               AND n.superseded_by IS NULL
               AND (n.valid_from IS NULL OR julianday(n.valid_from) <= julianday(?4))
               AND (n.valid_until IS NULL OR julianday(n.valid_until) > julianday(?4))
             ORDER BY n.retention_strength DESC, n.created_at DESC, n.id ASC LIMIT ?5",
        )?;
        let rows = stmt.query_map(
            params![node_type, scope, tag, Utc::now().to_rfc3339(), limit],
            Self::row_to_node,
        )?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    /// Get nodes by type and optional tag filter
    ///
    /// This is used for codebase context retrieval where we need to query
    /// by node_type (pattern/decision) and filter by codebase tag.
    /// Select current durable memory within the scope before applying the cap.
    pub fn projection_candidates(
        &self,
        scope: &str,
        min_retention: f64,
        limit: i32,
    ) -> Result<Vec<KnowledgeNode>> {
        let scope = Self::normalize_scope(scope)?;
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT * FROM knowledge_nodes
             WHERE COALESCE(NULLIF(trim(scope), ''), 'user') = ?1
               AND retention_strength >= ?2 AND suppression_count = 0 AND superseded_by IS NULL
               AND (valid_from IS NULL OR julianday(valid_from) <= julianday('now'))
               AND (valid_until IS NULL OR julianday(valid_until) > julianday('now'))
               AND (node_type IN ('decision','pattern') OR
                    (node_type IN ('fact','note') AND EXISTS
                     (SELECT 1 FROM json_each(knowledge_nodes.tags) WHERE lower(value) IN ('rule','preference','convention'))))
             ORDER BY CASE node_type WHEN 'decision' THEN 0 WHEN 'pattern' THEN 1 ELSE 2 END,
                      updated_at DESC, id ASC LIMIT ?3")?;
        let rows = stmt.query_map(params![scope, min_retention, limit], Self::row_to_node)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    /// Repair a bounded page of dirty/missing active-profile embeddings.
    /// The cursor is a scan position, not a frozen snapshot; restart from None
    /// after a sweep to discover new or failed rows preceding it.
    pub fn maintain_embedding_batch(
        &self,
        limit: usize,
        after: Option<&str>,
        dry_run: bool,
    ) -> Result<serde_json::Value> {
        if !(1..=100).contains(&limit) {
            return Err(StorageError::Init(
                "embedding batch limit must be 1..100".into(),
            ));
        }
        if after.is_some_and(|id| uuid::Uuid::parse_str(id).is_err()) {
            return Err(StorageError::Init(
                "embedding cursor must be a memory UUID".into(),
            ));
        }
        let started = std::time::Instant::now();
        let ids: Vec<String> = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            let profile = Self::active_profile_id_from_conn(&reader)?
                .unwrap_or_else(|| LEGACY_EMBEDDING_PROFILE_ID.into());
            let mut statement = reader.prepare("SELECT id FROM knowledge_nodes n
                WHERE id > ?1 AND COALESCE(suppression_count, 0) = 0
                AND (COALESCE(has_embedding, 0) = 0 OR NOT EXISTS (
                    SELECT 1 FROM embedding_profile_vectors v WHERE v.node_id = n.id AND v.profile_id = ?2))
                ORDER BY id LIMIT ?3")?;
            statement
                .query_map(
                    params![after.unwrap_or(""), profile, (limit + 1) as i64],
                    |row| row.get(0),
                )?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        let has_more = ids.len() > limit;
        let selected = &ids[..ids.len().min(limit)];
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        let runtime_ready = self.active_embedding_runtime_ready()?;
        #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
        let runtime_ready = false;
        let blocked = !dry_run && !runtime_ready && !selected.is_empty();
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        let result = if dry_run || blocked {
            EmbeddingResult::default()
        } else {
            self.generate_embeddings(Some(selected), false)?
        };
        #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
        let result = crate::memory::EmbeddingResult::default();
        let cursor = if blocked {
            after.map(str::to_string)
        } else {
            selected.last().cloned()
        };
        Ok(serde_json::json!({
            "phase": "embeddings", "dryRun": dry_run, "batchSize": limit,
            "selected": selected.len(), "successful": result.successful, "failed": result.failed,
            "skipped": result.skipped, "runtimeReady": runtime_ready,
            "status": if blocked { "runtime_unavailable" } else if dry_run { "preview" } else { "processed" },
            "hasMore": has_more || blocked, "nextCursor": cursor,
            "durationMs": started.elapsed().as_millis(),
            "checkpoint": "committed embedding rows; restart cursor after a sweep to retry failures or discover earlier inserts",
            "bound": "at most batchSize selected memories; no hard inference deadline"
        }))
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
    /// Scope candidates before either the recent or tag-targeted scan budget.
    /// None is an explicit cross-scope request; callers choose their boundary.
    pub fn get_never_composed_candidates_in_scope(
        &self,
        limit: i32,
        tag_filter: Option<&[String]>,
        scope: Option<&str>,
    ) -> Result<Vec<NeverComposedCandidate>> {
        let scope = scope.map(Self::normalize_scope).transpose()?;
        let nodes = self.composition_candidate_nodes(tag_filter, scope)?;
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

    /// Hash only mutation-relevant state. Access counters and passive decay do
    /// not invalidate a plan; content, source identity and control state do.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn merge_state_on(
        conn: &Connection,
        ids: &[String],
    ) -> Result<std::collections::BTreeMap<String, String>> {
        use sha2::{Digest, Sha256};
        let mut state = std::collections::BTreeMap::new();
        for id in ids {
            let payload: String = conn.query_row(
                "SELECT json_array(content, node_type, COALESCE(tags, '[]'), source,
                    COALESCE(NULLIF(trim(scope), ''), 'user'), protected, suppression_count,
                    valid_from, valid_until, superseded_by, source_system, source_project,
                    source_id, source_url, source_updated_at, content_hash, source_type, source_author)
                 FROM knowledge_nodes WHERE id = ?1", params![id], |row| row.get(0))
                .optional()?.ok_or_else(|| StorageError::NotFound(id.clone()))?;
            state.insert(
                id.clone(),
                format!("{:x}", Sha256::digest(payload.as_bytes())),
            );
        }
        Ok(state)
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn merge_state_snapshot(
        &self,
        ids: &[String],
    ) -> Result<std::collections::BTreeMap<String, String>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        Self::merge_state_on(&reader, ids)
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
                self.index_supplied_embedding(
                    &id_str,
                    vector,
                    supplied_model.as_deref(),
                    &record.content,
                )?;
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
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
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
                    b.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c))
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
                b.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
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

impl SqliteMemoryStore {}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests;

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
    const STORAGE_SOURCES: &[(&str, &str)] = &[
        ("sqlite/mod.rs", include_str!("mod.rs")),
        ("sqlite/admin.rs", include_str!("admin.rs")),
        ("sqlite/connectors.rs", include_str!("connectors.rs")),
        ("sqlite/embeddings.rs", include_str!("embeddings.rs")),
        ("sqlite/ingest.rs", include_str!("ingest.rs")),
        ("sqlite/lifecycle.rs", include_str!("lifecycle.rs")),
        ("sqlite/merge.rs", include_str!("merge.rs")),
        ("sqlite/purge.rs", include_str!("purge.rs")),
        ("sqlite/records.rs", include_str!("records.rs")),
        ("sqlite/search.rs", include_str!("search.rs")),
        ("sqlite/sync.rs", include_str!("sync.rs")),
        ("migrations.rs", include_str!("../migrations.rs")),
        ("trace_store.rs", include_str!("../trace_store.rs")),
        ("synaptic_store.rs", include_str!("../synaptic_store.rs")),
        ("replay_store.rs", include_str!("../replay_store.rs")),
        (
            "attestation_store.rs",
            include_str!("../attestation_store.rs"),
        ),
        (
            "unlearning_store.rs",
            include_str!("../unlearning_store.rs"),
        ),
        ("memory_store.rs", include_str!("../memory_store.rs")),
        ("portable.rs", include_str!("../portable.rs")),
        ("intention_claim.rs", include_str!("../intention_claim.rs")),
        (
            "intention_graph_store.rs",
            include_str!("../intention_graph_store.rs"),
        ),
        (
            "maintenance_batches.rs",
            include_str!("../maintenance_batches.rs"),
        ),
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
        for &(name, source) in STORAGE_SOURCES {
            for (index, line) in source.lines().enumerate() {
                let number = index + 1;
                if line.contains(&deferred_writer) || line.contains(&deferred_unchecked) {
                    offenders.push(format!(
                        "{name}:{number} opens a DEFERRED writer transaction; a read-then-write \
                         DEFERRED transaction can fail with SQLITE_BUSY_SNAPSHOT and SQLite does \
                         not consult the busy handler for that upgrade"
                    ));
                }
                if (HELPER_ROUTED.contains(&name) || name.starts_with("sqlite/"))
                    && line.contains(&bypasses_helper)
                {
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
        let source = include_str!("mod.rs");
        assert!(
            source.contains("fn begin_write_transaction"),
            "the write-transaction helper must exist"
        );
        // Sibling storage modules are not descendants of this one, so the
        // helper has to stay at least `pub(super)` for them to reach it.
        assert!(
            source.contains(
                ["pub(super) fn begin_write_", "transaction"]
                    .concat()
                    .as_str()
            ),
            "the helper must stay visible to sibling storage modules"
        );
    }
}

#[cfg(test)]
#[path = "../v3_regression_tests.rs"]
mod v3_regression_tests;

mod admin;
mod connectors;
mod embeddings;
mod ingest;
mod lifecycle;
mod merge;
mod purge;
mod records;
mod search;
mod sync;
