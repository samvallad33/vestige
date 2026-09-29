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

use crate::fsrs::{
    DEFAULT_DECAY, FSRSScheduler, FSRSState, LearningState, MAX_STABILITY, Rating,
    retrievability_with_decay,
};
use crate::fts::{sanitize_fts5_or_query, sanitize_fts5_query};
use crate::memory::{
    ConsolidationResult, IngestInput, KnowledgeNode, MatchType, MemoryStats, RecallInput,
    SearchMode, SearchResult,
};
use crate::security::{SecretFinding, SecretPolicy, scan_secrets};
use crate::storage::portable::{
    PORTABLE_ARCHIVE_FORMAT, PortableArchive, PortableImportMode, PortableImportReport,
    PortableTable, PortableValue, encode_hex,
};

// Phase 4 wall: types referenced by the MemoryStoreSend forwarding seam below.
use crate::actor::{ActorPolicySnapshot, RoleResolution};
use crate::advanced::reconsolidation::LabileCandidate;
use crate::advanced::{MergeCandidate, MergeOperation, MergePlan, MergePolicy};
use crate::codebase::anchor::{AnchorStatus, CodeAnchor};
use crate::intention_graph::Command;
use crate::neuroscience::SynapticTag;
use crate::storage::attestation_store::ReceiptAttestationStatus;
use crate::storage::attestation_store::{
    DurableSignedRetrievalReceipt, SignedReceiptWrite, StoredReceiptAttestationVerification,
};
use crate::storage::receipt_attestation::{ChainEntry, DsseEnvelope, TrustedSigningKey};
use crate::storage::blast::{BlastReport, RetireOutcome};
use crate::storage::resolver::HandleResolution;
use crate::storage::session_queries::{
    ClosedIssueNode, FailedToolCall, GitCommitNode, OpenFailureTouching,
};
use crate::storage::synaptic_store::{
    DurableSynapticCapture, SynapticCaptureRequest, SynapticIngestOutcome, SynapticIngestRequest,
};
use crate::storage::replay_store::{
    DurableCounterfactualReplay, DurableRetrievalReplayCapsule, RetrievalReplayCapsuleDraft,
    RetrievalReplayCapsuleSummary, StoredCounterfactualReplay,
};
use crate::storage::trace_store::{AgentRunSummary, PendingMemoryMutationDecision};
use crate::storage::walk_receipts::{CoverageSnapshot, StoredWalkReceipt, WalkReceiptHandle};
use crate::trace::{MemoryPr, MemoryPrAction, MemoryPrStatus, MemoryTraceEvent, Receipt};
use crate::SchemaIntrospection;




// ============================================================================
// ERROR TYPES
// ============================================================================

// `StorageError` and `Result` are defined in (and re-exported from) the
// ungated `crate::storage::types` module (dual-mode compilation,
// strata/fix-00a). Every other shared type below likewise.
pub use crate::storage::types::{
    CompositionEventRecord, CompositionMemberRecord, CompositionNeighborRecord,
    CompositionOutcomeRecord, ConnectorCursor, ConsolidationHistoryRecord, DreamHistoryRecord,
    FailureFeedbackReport, HygieneNodeSummary, HygieneSnapshot, InsightRecord, IntentionRecord,
    ConnectionRecord, NeverComposedCandidate, PortableSyncReport, PurgeReport, ReconcileReport,
    Result, SmartIngestResult, SourceUpsertOutcome, SourceUpsertResult, StateTransitionRecord,
    StorageError, TagVocabulary, WalCheckpointMode, WalCheckpointStatus,
};

/// Backend-typed conversion kept beside the SQLite code that produces it:
/// `StorageError::Database` carries a stringified error so the enum itself
/// stays buildable without `rusqlite`.
impl From<rusqlite::Error> for StorageError {
    fn from(e: rusqlite::Error) -> Self {
        StorageError::Database(e.to_string())
    }
}

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

// `HygieneNodeSummary`, `HygieneSnapshot`, and `TagVocabulary` are defined
// in (and re-exported from) `crate::storage::types`.

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

// `WalCheckpointMode` / `WalCheckpointStatus` are defined in (and
// re-exported from) `crate::storage::types`.

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

// `SmartIngestResult` is defined in (and re-exported from)
// `crate::storage::types`.

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

// `PortableSyncReport` and `PurgeReport` are defined in (and re-exported
// from) `crate::storage::types`.

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

#[allow(dead_code)] // read by the embedding_profile migration path; PR 10 sweeps it
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
// w1b search collapse: the VESTIGE_DISABLE_VECTOR_SEARCH kill-switch (env
// const, its affirmative-value parser, and the cfg(test) thread-local
// override) was removed together with the vector-search runtime paths.

// Test-only override for `VESTIGE_AUTO_CONSOLIDATE_MERGE`, scoped to the
// current thread for the same reason the old vector-search override was:
// this gate decides whether consolidation hard-deletes near-duplicates, so a
// process-wide flag would reach every consolidation test running at once.
// `Some(None)` pins the variable unset; `Some(Some(v))` pins a value.
#[cfg(all(test, feature = "embeddings", feature = "vector-search"))]
thread_local! {
    static AUTO_CONSOLIDATE_MERGE_FOR_TEST: std::cell::RefCell<Option<Option<String>>> =
        const { std::cell::RefCell::new(None) };
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
    /// The stable process actor did:key bound to this store at startup
    /// (#252 Phase A). `None` until a caller sets it; node writes stamp it
    /// as `author_actor_did` so self-support can be detected. Every local
    /// write through this store is by this process's actor.
    process_actor_did: std::sync::RwLock<Option<String>>,
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

// `FailureFeedbackReport` is defined in (and re-exported from)
// `crate::storage::types`.

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

}

// ============================================================================
// PERSISTENCE LAYER: Intentions, Insights, Connections, States
// ============================================================================

// `IntentionRecord` is defined in (and re-exported from)
// `crate::storage::types`.

impl IntentionRecord {
    /// Normalized namespace for this intention: blank/None -> "user",
    /// matching the `COALESCE(NULLIF(trim(scope), ''), 'user')` convention
    /// used by the scoped knowledge-node queries.
    pub fn effective_scope(&self) -> &str {
        match self.scope.as_deref() {
            Some(scope) if !scope.trim().is_empty() => scope.trim(),
            _ => "user",
        }
    }
}

// `InsightRecord` is defined in (and re-exported from)
// `crate::storage::types`.

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

// `ConnectionRecord` is defined in (and re-exported from)
// `crate::storage::types`.

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

// `StateTransitionRecord`, `ConsolidationHistoryRecord`, and
// `DreamHistoryRecord` are defined in (and re-exported from)
// `crate::storage::types`.

// `CompositionEventRecord`, `CompositionMemberRecord`,
// `CompositionOutcomeRecord`, `CompositionNeighborRecord`, and
// `NeverComposedCandidate` are defined in (and re-exported from)
// `crate::storage::types`.

impl SqliteMemoryStore {
    /// Adjacency over recorded typed causal edges (both directions treated
    /// as undirected hops for never-composed proximity; admission-relevant
    /// types only — never inferred, never similarity).
    fn typed_edge_adjacency(
        &self,
        link_types: &[&str],
    ) -> Result<std::collections::HashMap<String, Vec<String>>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let placeholders = link_types
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT source_id, target_id FROM memory_connections WHERE link_type IN ({placeholders})"
        );
        let mut stmt = reader.prepare(&sql)?;
        let mut map: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        let rows = stmt.query_map(rusqlite::params_from_iter(link_types.iter()), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        for r in rows {
            let (s, t) = r?;
            map.entry(s.clone()).or_default().push(t.clone());
            map.entry(t).or_default().push(s);
        }
        Ok(map)
    }

    /// For every pool node, hop distances (1..=max) to every other pool node
    /// over the given adjacency. Deterministic, cycle-safe.
    fn hop_distances(
        adjacency: &std::collections::HashMap<String, Vec<String>>,
        pool: &std::collections::HashSet<&str>,
        max: usize,
    ) -> std::collections::HashMap<String, std::collections::HashMap<String, usize>> {
        let mut out = std::collections::HashMap::new();
        for start in pool {
            let mut dist: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            let mut frontier = vec![(*start).to_string()];
            for depth in 1..=max {
                if frontier.is_empty() {
                    break;
                }
                let mut next = Vec::new();
                for node in &frontier {
                    for nb in adjacency.get(node).into_iter().flatten() {
                        if nb != start && !dist.contains_key(nb) {
                            dist.insert(nb.clone(), depth);
                            next.push(nb.clone());
                        }
                    }
                }
                frontier = next;
            }
            // keep only in-pool targets
            dist.retain(|k, _| pool.contains(k.as_str()));
            out.insert((*start).to_string(), dist);
        }
        out
    }

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

        // GhostLink admission (2026-09-28 ruling): pairs come ONLY from
        // recorded causal-edge hops — touched / derived_from / closed_by.
        // No tag admission, no term admission, no similarity of any kind.
        // With no typed edges in scope this returns NOTHING rather than
        // falling back on shared words.
        let typed_adjacency = self.typed_edge_adjacency(&["touched", "derived_from", "closed_by"])?;
        let pool_ids: std::collections::HashSet<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
        // hop-distance map over the pool (BFS depth <= 3, cycle-safe)
        let hop_map = Self::hop_distances(&typed_adjacency, &pool_ids, 3);

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

                // Edge-hop admission only: b must be reachable from a within
                // MAX hops over recorded typed edges. Shared words prove nothing.
                let hops = match hop_map.get(&a.id).and_then(|m| m.get(&b.id)) {
                    Some(h) => *h,
                    None => continue,
                };

                let shared_tags = Self::shared_tags(&a.tags, &b.tags);
                let shared_terms = Self::shared_content_terms(&a.content, &b.content, 8);
                let boundary_tags = Self::boundary_tags_for_pair(&a.tags, &b.tags);
                let trust_score =
                    ((a.retention_strength + b.retention_strength) / 2.0).clamp(0.0, 1.0);
                let degree_a = composition_degrees.get(&a.id).copied().unwrap_or(0) as f64;
                let degree_b = composition_degrees.get(&b.id).copied().unwrap_or(0) as f64;
                let novelty_score = ((1.0 / (1.0 + degree_a)) + (1.0 / (1.0 + degree_b))) / 2.0;
                // proximity from the causal graph replaces the old anchor score:
                // closer hops rank higher, purely structural
                let bridge_score = 1.0 / hops as f64;
                let anchor_score = 1.5 + bridge_score;
                let prior_outcomes = Self::pair_prior_outcomes(&outcome_map, &a.id, &b.id);
                let outcome_signal = Self::outcome_signal(&prior_outcomes);
                let outcome_score_adjustment = Self::outcome_score_adjustment(&prior_outcomes);
                let score = anchor_score
                    + (bridge_score * 2.0)
                    + (novelty_score * 1.5)
                    + trust_score
                    + outcome_score_adjustment;

                let reason = format!(
                    "Connected by {} typed-edge hop{} but never composed",
                    hops,
                    if hops == 1 { "" } else { "s" }
                );
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
                Sha256::digest(payload.as_bytes())
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>(),
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

    async fn search_records(
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
        // w1b search collapse: vector retrieval is removed from this store.
        // The trait signature is kept because MemoryStore requires it; this
        // shell is inert until the exact resolver lands.
        let _ = (embedding, limit);
        Ok(vec![])
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

    async fn get_store_stats(
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

    // ------------------------------------------------------------------
    // Phase 4 product seam: forward every sync trait method to the
    // inherent SQLite implementation one for one. The explicit
    // `SqliteMemoryStore::method(self, ..)` form keeps resolution on the
    // inherent method (not the trait method being defined here).
    // ------------------------------------------------------------------

    fn actor_policy_snapshot(&self) -> Result<ActorPolicySnapshot> {
        SqliteMemoryStore::actor_policy_snapshot(self)
    }
    fn append_mcp_call_outcome(&self, run_id: &str, tool: &str, success: bool, error: Option<&str>, at_ms: i64,) -> Result<()> {
        SqliteMemoryStore::append_mcp_call_outcome(self, run_id, tool, success, error, at_ms)
    }
    fn append_trace_event(&self, event: &MemoryTraceEvent) -> Result<i64> {
        SqliteMemoryStore::append_trace_event(self, event)
    }
    fn apply_failure_feedback(&self, failure_id: &str, window: Duration) -> Result<FailureFeedbackReport> {
        SqliteMemoryStore::apply_failure_feedback(self, failure_id, window)
    }
    fn apply_intention_graph(&self, scope: &str, command: Command, now: DateTime<Utc>) -> std::result::Result<serde_json::Value, String> {
        SqliteMemoryStore::apply_intention_graph(self, scope, command, now)
    }
    fn apply_tag_mutation(&self, source_tags: &[String], target_tag: &str, scope: Option<&str>, preview_token: &str, op_type: &str, reason: &str,) -> Result<MergeOperation> {
        SqliteMemoryStore::apply_tag_mutation(self, source_tags, target_tag, scope, preview_token, op_type, reason)
    }
    fn backup_to(&self, path: &Path) -> Result<()> {
        SqliteMemoryStore::backup_to(self, path)
    }
    fn blast_radius(&self, root_id: &str, open_only: bool) -> Result<BlastReport> {
        SqliteMemoryStore::blast_radius(self, root_id, open_only)
    }
    fn blast_radius_with_link_types(&self, root_id: &str, open_only: bool, link_types: &[&str],) -> Result<BlastReport> {
        SqliteMemoryStore::blast_radius_with_link_types(self, root_id, open_only, link_types)
    }
    fn capture_synaptic_event(&self, request: &SynapticCaptureRequest) -> Result<DurableSynapticCapture> {
        SqliteMemoryStore::capture_synaptic_event(self, request)
    }
    fn checkpoint_wal(&self, mode: WalCheckpointMode) -> Result<WalCheckpointStatus> {
        SqliteMemoryStore::checkpoint_wal(self, mode)
    }
    fn clear_dream_page_tags(&self, ids: &[String], started_at: DateTime<Utc>,) -> Result<usize> {
        SqliteMemoryStore::clear_dream_page_tags(self, ids, started_at)
    }
    fn closed_issue_nodes(&self, source_system: &str, scope: &str) -> Result<Vec<ClosedIssueNode>> {
        SqliteMemoryStore::closed_issue_nodes(self, source_system, scope)
    }
    fn code_anchors_for_node(&self, node_id: &str) -> Result<Vec<CodeAnchor>> {
        SqliteMemoryStore::code_anchors_for_node(self, node_id)
    }
    fn code_anchors_for_nodes(&self, node_ids: &[String]) -> Result<HashMap<String, Vec<CodeAnchor>>> {
        SqliteMemoryStore::code_anchors_for_nodes(self, node_ids)
    }
    fn commit_intention_check(&self, changes: &[(IntentionRecord, IntentionRecord)]) -> std::result::Result<(), String> {
        SqliteMemoryStore::commit_intention_check(self, changes)
    }
    fn concrete_search_filtered(&self, query: &str, limit: i32, include_types: Option<&[String]>, exclude_types: Option<&[String]>,) -> Result<Vec<crate::memory::SearchResult>> {
        SqliteMemoryStore::concrete_search_filtered(self, query, limit, include_types, exclude_types)
    }
    fn count_memories_below_retention(&self, threshold: f64) -> Result<i64> {
        SqliteMemoryStore::count_memories_below_retention(self, threshold)
    }
    fn count_memories_since(&self, since: DateTime<Utc>) -> Result<i64> {
        SqliteMemoryStore::count_memories_since(self, since)
    }
    fn count_pending_memory_prs(&self) -> Result<i64> {
        SqliteMemoryStore::count_pending_memory_prs(self)
    }
    fn count_suppressed(&self) -> Result<usize> {
        SqliteMemoryStore::count_suppressed(self)
    }
    fn coverage_snapshot(&self) -> Result<CoverageSnapshot> {
        SqliteMemoryStore::coverage_snapshot(self)
    }
    fn create_context_ablation_replay(&self, source_receipt_id: &str, withheld_slots: &[String],) -> Result<DurableCounterfactualReplay> {
        SqliteMemoryStore::create_context_ablation_replay(self, source_receipt_id, withheld_slots)
    }
    fn current_code_context_nodes(&self, node_type: &str, tag: Option<&str>, scope: &str, limit: i32) -> Result<Vec<KnowledgeNode>> {
        SqliteMemoryStore::current_code_context_nodes(self, node_type, tag, scope, limit)
    }
    fn data_dir(&self) -> &Path {
        SqliteMemoryStore::data_dir(self)
    }
    fn db_path(&self) -> &Path {
        SqliteMemoryStore::db_path(self)
    }
    fn decide_memory_pr(&self, id: &str, action: MemoryPrAction) -> Result<MemoryPr> {
        SqliteMemoryStore::decide_memory_pr(self, id, action)
    }
    fn decide_pending_memory_mutation(&self, id: &str, action: MemoryPrAction,) -> Result<Option<PendingMemoryMutationDecision>> {
        SqliteMemoryStore::decide_pending_memory_mutation(self, id, action)
    }
    fn delete_node(&self, id: &str) -> Result<bool> {
        SqliteMemoryStore::delete_node(self, id)
    }
    fn demote_memory(&self, id: &str) -> Result<KnowledgeNode> {
        SqliteMemoryStore::demote_memory(self, id)
    }
    fn demote_memory_as_actor(&self, id: &str, claimed_role: Option<&str>, tool: &str,) -> Result<ActorMutationOutcome> {
        SqliteMemoryStore::demote_memory_as_actor(self, id, claimed_role, tool)
    }
    fn due_for_review_node_ids(&self, limit: usize) -> Result<Vec<String>> {
        SqliteMemoryStore::due_for_review_node_ids(self, limit)
    }
    fn expire_stale_reconsolidation_plans(&self) -> Result<Vec<String>> {
        SqliteMemoryStore::expire_stale_reconsolidation_plans(self)
    }
    fn export_portable_archive_to_path(&self, path: &Path) -> Result<PortableArchive> {
        SqliteMemoryStore::export_portable_archive_to_path(self, path)
    }
    fn get_active_intentions(&self) -> Result<Vec<IntentionRecord>> {
        SqliteMemoryStore::get_active_intentions(self)
    }
    fn get_active_intentions_in_scope(&self, scope: &str) -> Result<Vec<IntentionRecord>> {
        SqliteMemoryStore::get_active_intentions_in_scope(self, scope)
    }
    fn get_agent_run(&self, run_id: &str) -> Result<Option<AgentRunSummary>> {
        SqliteMemoryStore::get_agent_run(self, run_id)
    }
    fn get_all_connections(&self) -> Result<Vec<ConnectionRecord>> {
        SqliteMemoryStore::get_all_connections(self)
    }
    fn get_all_nodes(&self, limit: i32, offset: i32) -> Result<Vec<KnowledgeNode>> {
        SqliteMemoryStore::get_all_nodes(self, limit, offset)
    }
    fn get_all_nodes_in_scope(&self, scope: &str, limit: i32, offset: i32) -> Result<Vec<KnowledgeNode>> {
        SqliteMemoryStore::get_all_nodes_in_scope(self, scope, limit, offset)
    }
    fn get_avg_retention(&self) -> Result<f64> {
        SqliteMemoryStore::get_avg_retention(self)
    }
    fn get_composition_event(&self, id: &str) -> Result<Option<CompositionEventRecord>> {
        SqliteMemoryStore::get_composition_event(self, id)
    }
    fn get_composition_members(&self, event_id: &str) -> Result<Vec<CompositionMemberRecord>> {
        SqliteMemoryStore::get_composition_members(self, event_id)
    }
    fn get_composition_neighbors(&self, memory_id: &str, limit: i32,) -> Result<Vec<CompositionNeighborRecord>> {
        SqliteMemoryStore::get_composition_neighbors(self, memory_id, limit)
    }
    fn get_composition_outcomes(&self, event_id: &str,) -> Result<Vec<CompositionOutcomeRecord>> {
        SqliteMemoryStore::get_composition_outcomes(self, event_id)
    }
    fn get_compositions_for_memory(&self, memory_id: &str, limit: i32,) -> Result<Vec<CompositionEventRecord>> {
        SqliteMemoryStore::get_compositions_for_memory(self, memory_id, limit)
    }
    fn get_connections_for_memory(&self, memory_id: &str) -> Result<Vec<ConnectionRecord>> {
        SqliteMemoryStore::get_connections_for_memory(self, memory_id)
    }
    fn get_consolidation_history(&self, limit: i32) -> Result<Vec<ConsolidationHistoryRecord>> {
        SqliteMemoryStore::get_consolidation_history(self, limit)
    }
    fn get_context_ablation_replay(&self, replay_id: &str,) -> Result<Option<StoredCounterfactualReplay>> {
        SqliteMemoryStore::get_context_ablation_replay(self, replay_id)
    }
    fn get_dream_history(&self, limit: i32) -> Result<Vec<DreamHistoryRecord>> {
        SqliteMemoryStore::get_dream_history(self, limit)
    }
    fn get_insights(&self, limit: i32) -> Result<Vec<InsightRecord>> {
        SqliteMemoryStore::get_insights(self, limit)
    }
    fn get_intention(&self, id: &str) -> Result<Option<IntentionRecord>> {
        SqliteMemoryStore::get_intention(self, id)
    }
    fn get_intentions_by_status(&self, status: &str) -> Result<Vec<IntentionRecord>> {
        SqliteMemoryStore::get_intentions_by_status(self, status)
    }
    fn get_last_consolidation(&self) -> Result<Option<DateTime<Utc>>> {
        SqliteMemoryStore::get_last_consolidation(self)
    }
    fn get_last_dream(&self) -> Result<Option<DateTime<Utc>>> {
        SqliteMemoryStore::get_last_dream(self)
    }
    fn get_memory_pr(&self, id: &str) -> Result<Option<MemoryPr>> {
        SqliteMemoryStore::get_memory_pr(self, id)
    }
    fn get_memory_subgraph(&self, center_id: &str, depth: u32, max_nodes: usize,) -> Result<(Vec<KnowledgeNode>, Vec<ConnectionRecord>)> {
        SqliteMemoryStore::get_memory_subgraph(self, center_id, depth, max_nodes)
    }
    fn get_merge_operation(&self, operation_id: &str,) -> Result<Option<MergeOperation>> {
        SqliteMemoryStore::get_merge_operation(self, operation_id)
    }
    fn get_merge_policy(&self) -> Result<MergePolicy> {
        SqliteMemoryStore::get_merge_policy(self)
    }
    fn get_most_connected_memory(&self) -> Result<Option<String>> {
        SqliteMemoryStore::get_most_connected_memory(self)
    }
    fn get_never_composed_candidates(&self, limit: i32, tag_filter: Option<&[String]>,) -> Result<Vec<NeverComposedCandidate>> {
        SqliteMemoryStore::get_never_composed_candidates(self, limit, tag_filter)
    }
    fn get_never_composed_candidates_in_scope(&self, limit: i32, tag_filter: Option<&[String]>, scope: Option<&str>) -> Result<Vec<NeverComposedCandidate>> {
        SqliteMemoryStore::get_never_composed_candidates_in_scope(self, limit, tag_filter, scope)
    }
    fn get_node(&self, id: &str) -> Result<Option<KnowledgeNode>> {
        SqliteMemoryStore::get_node(self, id)
    }
    fn get_overdue_intentions(&self) -> Result<Vec<IntentionRecord>> {
        SqliteMemoryStore::get_overdue_intentions(self)
    }
    fn get_receipt(&self, receipt_id: &str) -> Result<Option<Receipt>> {
        SqliteMemoryStore::get_receipt(self, receipt_id)
    }
    fn get_receipt_attestation_envelope(&self, receipt_id: &str,) -> Result<Option<DsseEnvelope>> {
        SqliteMemoryStore::get_receipt_attestation_envelope(self, receipt_id)
    }
    fn get_recent_composition_events(&self, limit: i32) -> Result<Vec<CompositionEventRecord>> {
        SqliteMemoryStore::get_recent_composition_events(self, limit)
    }
    fn get_recent_composition_events_page(&self, limit: i32, offset: i32,) -> Result<Vec<CompositionEventRecord>> {
        SqliteMemoryStore::get_recent_composition_events_page(self, limit, offset)
    }
    fn get_recent_connections(&self, limit: usize) -> Result<Vec<ConnectionRecord>> {
        SqliteMemoryStore::get_recent_connections(self, limit)
    }
    fn get_recent_state_transitions(&self, limit: i32) -> Result<Vec<StateTransitionRecord>> {
        SqliteMemoryStore::get_recent_state_transitions(self, limit)
    }
    fn get_retention_distribution(&self) -> Result<Vec<(String, i64)>> {
        SqliteMemoryStore::get_retention_distribution(self)
    }
    fn get_retention_trend(&self) -> Result<String> {
        SqliteMemoryStore::get_retention_trend(self)
    }
    fn get_retrieval_replay_capsule(&self, source_receipt_id: &str,) -> Result<Option<RetrievalReplayCapsuleSummary>> {
        SqliteMemoryStore::get_retrieval_replay_capsule(self, source_receipt_id)
    }
    fn get_review_queue(&self, limit: i32) -> Result<Vec<KnowledgeNode>> {
        SqliteMemoryStore::get_review_queue(self, limit)
    }
    fn get_state_transitions(&self, memory_id: &str, limit: i32,) -> Result<Vec<StateTransitionRecord>> {
        SqliteMemoryStore::get_state_transitions(self, memory_id, limit)
    }
    fn get_stats(&self) -> Result<MemoryStats> {
        SqliteMemoryStore::get_stats(self)
    }
    fn get_trace(&self, run_id: &str) -> Result<Vec<MemoryTraceEvent>> {
        SqliteMemoryStore::get_trace(self, run_id)
    }
    fn get_walk_receipt(&self, receipt_id: &str) -> Result<Option<StoredWalkReceipt>> {
        SqliteMemoryStore::get_walk_receipt(self, receipt_id)
    }
    fn git_commit_nodes(&self, limit: usize) -> Result<Vec<GitCommitNode>> {
        SqliteMemoryStore::git_commit_nodes(self, limit)
    }
    fn grant_actor_role(&self, actor_did: &str, role: &str, note: Option<&str>) -> Result<u64> {
        SqliteMemoryStore::grant_actor_role(self, actor_did, role, note)
    }
    fn hybrid_search(&self, query: &str, limit: i32, keyword_weight: f32, semantic_weight: f32,) -> Result<Vec<crate::memory::SearchResult>> {
        SqliteMemoryStore::hybrid_search(self, query, limit, keyword_weight, semantic_weight)
    }
    fn hybrid_search_filtered(&self, query: &str, limit: i32, keyword_weight: f32, semantic_weight: f32, include_types: Option<&[String]>, exclude_types: Option<&[String]>,) -> Result<Vec<crate::memory::SearchResult>> {
        SqliteMemoryStore::hybrid_search_filtered(self, query, limit, keyword_weight, semantic_weight, include_types, exclude_types)
    }
    fn hygiene_snapshot(&self, scope: Option<&str>) -> Result<HygieneSnapshot> {
        SqliteMemoryStore::hygiene_snapshot(self, scope)
    }
    fn import_portable_archive(&self, archive: &PortableArchive, mode: PortableImportMode) -> Result<PortableImportReport> {
        SqliteMemoryStore::import_portable_archive(self, archive, mode)
    }
    fn import_portable_archive_from_path(&self, path: &Path, mode: PortableImportMode) -> Result<PortableImportReport> {
        SqliteMemoryStore::import_portable_archive_from_path(self, path, mode)
    }
    fn ingest(&self, input: IngestInput) -> Result<KnowledgeNode> {
        SqliteMemoryStore::ingest(self, input)
    }
    fn ingest_in_scope(&self, input: IngestInput, scope: &str) -> Result<KnowledgeNode> {
        SqliteMemoryStore::ingest_in_scope(self, input, scope)
    }
    fn ingest_in_scope_with_secret_policy(&self, input: IngestInput, scope: &str, policy: SecretPolicy) -> Result<KnowledgeNode> {
        SqliteMemoryStore::ingest_in_scope_with_secret_policy(self, input, scope, policy)
    }
    fn ingest_with_secret_policy(&self, input: IngestInput, policy: SecretPolicy) -> Result<KnowledgeNode> {
        SqliteMemoryStore::ingest_with_secret_policy(self, input, policy)
    }
    fn intention_memory_snapshot(&self, scope: &str, memory_id: &str, now: DateTime<Utc>) -> std::result::Result<serde_json::Value, String> {
        SqliteMemoryStore::intention_memory_snapshot(self, scope, memory_id, now)
    }
    fn last_backup_timestamp(&self) -> Option<DateTime<Utc>> {
        SqliteMemoryStore::last_backup_timestamp(self)
    }
    fn last_session_failed_calls(&self, run_id: Option<&str>) -> Result<Vec<FailedToolCall>> {
        SqliteMemoryStore::last_session_failed_calls(self, run_id)
    }
    fn latest_receipt_chain_entry(&self) -> Result<Option<ChainEntry>> {
        SqliteMemoryStore::latest_receipt_chain_entry(self)
    }
    fn link_receipt_to_run(&self, receipt_id: &str, run_id: &str) -> Result<bool> {
        SqliteMemoryStore::link_receipt_to_run(self, receipt_id, run_id)
    }
    fn list_agent_runs(&self, limit: usize) -> Result<Vec<AgentRunSummary>> {
        SqliteMemoryStore::list_agent_runs(self, limit)
    }
    fn list_endorsement_events(&self, memory_id: Option<&str>, actor_did: Option<&str>, limit: usize) -> Result<Vec<EndorsementEventRecord>> {
        SqliteMemoryStore::list_endorsement_events(self, memory_id, actor_did, limit)
    }
    fn list_memory_prs(&self, status: Option<MemoryPrStatus>, limit: usize,) -> Result<Vec<MemoryPr>> {
        SqliteMemoryStore::list_memory_prs(self, status, limit)
    }
    fn list_merge_operations(&self, limit: usize,) -> Result<Vec<MergeOperation>> {
        SqliteMemoryStore::list_merge_operations(self, limit)
    }
    fn list_receipts(&self, limit: usize) -> Result<Vec<Receipt>> {
        SqliteMemoryStore::list_receipts(self, limit)
    }
    fn list_receipts_for_run(&self, run_id: &str, limit: usize) -> Result<Vec<Receipt>> {
        SqliteMemoryStore::list_receipts_for_run(self, run_id, limit)
    }
    fn list_reconsolidation_plans(&self, limit: usize) -> Result<Vec<(MergePlan, String)>> {
        SqliteMemoryStore::list_reconsolidation_plans(self, limit)
    }
    fn list_tag_operations(&self, limit: usize, scope: Option<&str>,) -> Result<Vec<MergeOperation>> {
        SqliteMemoryStore::list_tag_operations(self, limit, scope)
    }
    fn load_active_synaptic_tags(&self) -> Result<Vec<SynapticTag>> {
        SqliteMemoryStore::load_active_synaptic_tags(self)
    }
    fn lowest_retention_nodes(&self, limit: usize) -> Result<Vec<(String, f64)>> {
        SqliteMemoryStore::lowest_retention_nodes(self, limit)
    }
    fn maintain_gc_batch(&self, limit: usize, after: Option<&str>, budget_ms: u64, dry_run: bool, min_retention: f64, max_age_days: Option<u64>) -> Result<serde_json::Value> {
        SqliteMemoryStore::maintain_gc_batch(self, limit, after, budget_ms, dry_run, min_retention, max_age_days)
    }
    fn maintain_lifecycle_batch(&self, limit: usize, after: Option<&str>, budget_ms: u64, dry_run: bool) -> Result<serde_json::Value> {
        SqliteMemoryStore::maintain_lifecycle_batch(self, limit, after, budget_ms, dry_run)
    }
    fn maintain_log_batch(&self, limit: usize, dry_run: bool) -> Result<serde_json::Value> {
        SqliteMemoryStore::maintain_log_batch(self, limit, dry_run)
    }
    fn maintenance_memory_page(&self, limit: usize, after: Option<&str>, scope: &str,) -> Result<(Vec<crate::KnowledgeNode>, bool)> {
        SqliteMemoryStore::maintenance_memory_page(self, limit, after, scope)
    }
    fn mark_reviewed(&self, id: &str, rating: Rating) -> Result<KnowledgeNode> {
        SqliteMemoryStore::mark_reviewed(self, id, rating)
    }
    fn merge_candidates(&self, policy: MergePolicy, limit: usize, tag_filter: &[String]) -> Result<Vec<MergeCandidate>> {
        SqliteMemoryStore::merge_candidates(self, policy, limit, tag_filter)
    }
    fn merge_undo(&self, op_id: &str) -> Result<MergeOperation> {
        // The inherent implementation was embedding-gated and is gone with
        // the vector wipe; the delegation below resolved to THIS trait method
        // (infinite recursion, clippy-found). Fail loud instead.
        let _ = op_id;
        Err(StorageError::Init(
            "merge_undo requires the embedding runtime, which 4.0 removed; supersede/correct via admission instead".to_string(),
        ))
    }
    fn node_is_in_scope(&self, id: &str, scope: &str) -> Result<bool> {
        SqliteMemoryStore::node_is_in_scope(self, id, scope)
    }
    fn open_failures_touching(&self, changed_files: &[String]) -> Result<Vec<OpenFailureTouching>> {
        SqliteMemoryStore::open_failures_touching(self, changed_files)
    }
    fn preview_tag_mutation(&self, source_tags: &[String], target_tag: &str, scope: Option<&str>,) -> Result<serde_json::Value> {
        SqliteMemoryStore::preview_tag_mutation(self, source_tags, target_tag, scope)
    }
    fn process_actor_did(&self) -> Option<String> {
        SqliteMemoryStore::process_actor_did(self)
    }
    fn process_synaptic_ingest(&self, request: &SynapticIngestRequest) -> Result<SynapticIngestOutcome> {
        SqliteMemoryStore::process_synaptic_ingest(self, request)
    }
    fn projection_candidates(&self, scope: &str, min_retention: f64, limit: i32) -> Result<Vec<KnowledgeNode>> {
        SqliteMemoryStore::projection_candidates(self, scope, min_retention, limit)
    }
    fn promote_memory(&self, id: &str) -> Result<KnowledgeNode> {
        SqliteMemoryStore::promote_memory(self, id)
    }
    fn promote_memory_as_actor(&self, id: &str, claimed_role: Option<&str>, tool: &str,) -> Result<ActorMutationOutcome> {
        SqliteMemoryStore::promote_memory_as_actor(self, id, claimed_role, tool)
    }
    fn promote_memory_backfill(&self, id: &str) -> Result<KnowledgeNode> {
        SqliteMemoryStore::promote_memory_backfill(self, id)
    }
    fn prune_agent_traces(&self) -> Result<i64> {
        SqliteMemoryStore::prune_agent_traces(self)
    }
    fn purge_node(&self, id: &str, reason: Option<&str>) -> Result<PurgeReport> {
        SqliteMemoryStore::purge_node(self, id, reason)
    }
    fn query_time_range(&self, start: Option<DateTime<Utc>>, end: Option<DateTime<Utc>>, limit: i32, node_type: Option<&str>, tags: Option<&[String]>,) -> Result<Vec<KnowledgeNode>> {
        SqliteMemoryStore::query_time_range(self, start, end, limit, node_type, tags)
    }
    fn recall(&self, input: RecallInput) -> Result<Vec<KnowledgeNode>> {
        SqliteMemoryStore::recall(self, input)
    }
    fn receipt_attestation_status(&self, receipt_id: &str) -> Result<Option<ReceiptAttestationStatus>> {
        SqliteMemoryStore::receipt_attestation_status(self, receipt_id)
    }
    fn record_anchor_verification(&self, anchor_id: &str, status: AnchorStatus, checked_at: DateTime<Utc>) -> Result<()> {
        SqliteMemoryStore::record_anchor_verification(self, anchor_id, status, checked_at)
    }
    fn record_batch_retrieval(&self, ids: &[&str]) -> Result<()> {
        SqliteMemoryStore::record_batch_retrieval(self, ids)
    }
    fn record_code_anchors(&self, anchors: &[CodeAnchor]) -> Result<usize> {
        SqliteMemoryStore::record_code_anchors(self, anchors)
    }
    fn record_composition_outcome(&self, outcome: &CompositionOutcomeRecord) -> Result<()> {
        SqliteMemoryStore::record_composition_outcome(self, outcome)
    }
    fn record_memory_access(&self, memory_id: &str) -> Result<()> {
        SqliteMemoryStore::record_memory_access(self, memory_id)
    }
    fn record_reinforce_endorsement(&self, id: &str, claimed_role: Option<&str>, tool: &str) -> Result<ActorMutationOutcome> {
        SqliteMemoryStore::record_reinforce_endorsement(self, id, claimed_role, tool)
    }
    fn registered_receipt_signing_key(&self, key_id: &str,) -> Result<Option<TrustedSigningKey>> {
        SqliteMemoryStore::registered_receipt_signing_key(self, key_id)
    }
    fn release_quarantine(&self, id: &str) -> Result<KnowledgeNode> {
        SqliteMemoryStore::release_quarantine(self, id)
    }
    fn replace_code_anchors(&self, node_id: &str, scope: &str, anchors: &[CodeAnchor],) -> Result<usize> {
        SqliteMemoryStore::replace_code_anchors(self, node_id, scope, anchors)
    }
    fn replay_intention_graph(&self, scope: &str) -> std::result::Result<serde_json::Value, String> {
        SqliteMemoryStore::replay_intention_graph(self, scope)
    }
    fn resolve_actor_role(&self, claimed_role: Option<&str>) -> Result<(String, RoleResolution)> {
        SqliteMemoryStore::resolve_actor_role(self, claimed_role)
    }
    fn resolve_commit_sha_root(&self, sha_prefix: &str) -> Result<Option<String>> {
        SqliteMemoryStore::resolve_commit_sha_root(self, sha_prefix)
    }
    fn resolve_handle(&self, query: &str) -> HandleResolution {
        SqliteMemoryStore::resolve_handle(self, query)
    }
    fn retire_affected(&self, ids: &[&str], reason: &str) -> Vec<RetireOutcome> {
        SqliteMemoryStore::retire_affected(self, ids, reason)
    }
    fn reverse_suppression(&self, id: &str, labile_hours: i64) -> Result<KnowledgeNode> {
        SqliteMemoryStore::reverse_suppression(self, id, labile_hours)
    }
    fn run_consolidation(&self) -> Result<ConsolidationResult> {
        SqliteMemoryStore::run_consolidation(self)
    }
    fn save_composition(&self, event: &CompositionEventRecord, members: &[CompositionMemberRecord], outcomes: &[CompositionOutcomeRecord],) -> Result<()> {
        SqliteMemoryStore::save_composition(self, event, members, outcomes)
    }
    fn save_connection(&self, connection: &ConnectionRecord) -> Result<()> {
        SqliteMemoryStore::save_connection(self, connection)
    }
    fn save_counterfactual_replay_receipt(&self, replay_id: &str, receipt: &Receipt, run_id: Option<&str>, tool: Option<&str>) -> Result<()> {
        SqliteMemoryStore::save_counterfactual_replay_receipt(self, replay_id, receipt, run_id, tool)
    }
    fn save_dream_history(&self, record: &DreamHistoryRecord) -> Result<i64> {
        SqliteMemoryStore::save_dream_history(self, record)
    }
    fn save_insight(&self, insight: &InsightRecord) -> Result<()> {
        SqliteMemoryStore::save_insight(self, insight)
    }
    fn save_intention(&self, intention: &IntentionRecord) -> Result<()> {
        SqliteMemoryStore::save_intention(self, intention)
    }
    fn save_memory_pr(&self, pr: &MemoryPr) -> Result<()> {
        SqliteMemoryStore::save_memory_pr(self, pr)
    }
    fn save_receipt(&self, receipt: &Receipt, run_id: Option<&str>, tool: Option<&str>, query: Option<&str>) -> Result<()> {
        SqliteMemoryStore::save_receipt(self, receipt, run_id, tool, query)
    }
    fn save_retrieval_receipt_with_replay_capsule(&self, receipt: &Receipt, run_id: Option<&str>, tool: Option<&str>, draft: &RetrievalReplayCapsuleDraft,) -> Result<DurableRetrievalReplayCapsule> {
        SqliteMemoryStore::save_retrieval_receipt_with_replay_capsule(self, receipt, run_id, tool, draft)
    }
    fn save_signed_retrieval_receipt_with_replay_capsule_atomic(&self, write: SignedReceiptWrite<'_>, draft: &RetrievalReplayCapsuleDraft,) -> Result<DurableSignedRetrievalReceipt> {
        SqliteMemoryStore::save_signed_retrieval_receipt_with_replay_capsule_atomic(self, write, draft)
    }
    fn save_synaptic_tag(&self, tag: &SynapticTag) -> Result<String> {
        SqliteMemoryStore::save_synaptic_tag(self, tag)
    }
    fn save_walk_receipt(&self, canonical_json: &str, params: &serde_json::Value,) -> Result<WalkReceiptHandle> {
        SqliteMemoryStore::save_walk_receipt(self, canonical_json, params)
    }
    fn schema_introspection(&self) -> Result<SchemaIntrospection> {
        SqliteMemoryStore::schema_introspection(self)
    }
    fn search(&self, query: &str, limit: i32) -> Result<Vec<KnowledgeNode>> {
        SqliteMemoryStore::search(self, query, limit)
    }
    fn set_created_at(&self, id: &str, when: DateTime<Utc>) -> Result<()> {
        SqliteMemoryStore::set_created_at(self, id, when)
    }
    fn set_merge_policy(&self, policy: crate::advanced::MergePolicy) -> Result<()> {
        SqliteMemoryStore::set_merge_policy(self, policy)
    }
    fn set_process_actor(&self, did: &str) -> Result<()> {
        SqliteMemoryStore::set_process_actor(self, did)
    }
    fn set_protected(&self, id: &str, protected: bool) -> Result<()> {
        SqliteMemoryStore::set_protected(self, id, protected)
    }
    fn sidecar_dir(&self, name: &str) -> PathBuf {
        SqliteMemoryStore::sidecar_dir(self, name)
    }
    fn register_receipt_signing_key(&self, key: &TrustedSigningKey) -> Result<bool> {
        SqliteMemoryStore::register_receipt_signing_key(self, key)
    }
    fn export_portable_archive(&self) -> Result<PortableArchive> {
        SqliteMemoryStore::export_portable_archive(self)
    }
    fn reconcile_source_tombstones(&self, source_system: &str, scope: &str, live_ids: &[String]) -> Result<ReconcileReport> {
        SqliteMemoryStore::reconcile_source_tombstones(self, source_system, scope, live_ids)
    }
    fn get_connector_cursor(&self, source_system: &str, scope: &str) -> Result<ConnectorCursor> {
        SqliteMemoryStore::get_connector_cursor(self, source_system, scope)
    }
    fn save_connector_cursor(&self, cursor: &ConnectorCursor) -> Result<()> {
        SqliteMemoryStore::save_connector_cursor(self, cursor)
    }
    fn dream_compile_candidates(&self, scope: &str, limit: usize) -> Result<Vec<KnowledgeNode>> {
        SqliteMemoryStore::dream_compile_candidates(self, scope, limit)
    }
    fn strengthen_connection(&self, source_id: &str, target_id: &str, boost: f64) -> Result<bool> {
        SqliteMemoryStore::strengthen_connection(self, source_id, target_id, boost)
    }
    fn run_rac1_cascade_sweep(&self) -> Result<(usize, usize)> {
        SqliteMemoryStore::run_rac1_cascade_sweep(self)
    }
    fn smart_ingest_excluding_in_scope_with_secret_policy_and_labile(&self, _input: IngestInput, scope: &str, excluded_node_ids: &[String], policy: SecretPolicy, labile: &[LabileCandidate]) -> Result<SmartIngestResult> {
        // Same embedding-gated wipe as merge_undo: the delegation recursed
        // into itself. The plain (non-excluding) smart_ingest path is live.
        let _ = (scope, excluded_node_ids, policy, labile);
        Err(StorageError::Init(
            "smart_ingest with exclusions required the embedding runtime, which 4.0 removed; use smart_ingest".to_string(),
        ))
    }
    fn snooze_intention(&self, id: &str, until: DateTime<Utc>) -> Result<bool> {
        SqliteMemoryStore::snooze_intention(self, id, until)
    }
    fn state_distribution(&self) -> Result<(i64, i64, i64, i64)> {
        SqliteMemoryStore::state_distribution(self)
    }
    fn superseded_node_ids(&self) -> Result<HashSet<String>> {
        SqliteMemoryStore::superseded_node_ids(self)
    }
    fn supersession_pairs(&self) -> Result<Vec<(String, String)>> {
        SqliteMemoryStore::supersession_pairs(self)
    }
    fn suppress_memory(&self, id: &str) -> Result<KnowledgeNode> {
        SqliteMemoryStore::suppress_memory(self, id)
    }
    fn sync_portable_archive_cloud(&self, endpoint: &str, sync_key: &str, encryption_key: Option<String>) -> Result<PortableSyncReport> {
        SqliteMemoryStore::sync_portable_archive_cloud(self, endpoint, sync_key, encryption_key)
    }
    fn sync_portable_archive_file(&self, path: &Path) -> Result<PortableSyncReport> {
        SqliteMemoryStore::sync_portable_archive_file(self, path)
    }
    fn tag_vocabulary(&self, scope: Option<&str>) -> Result<TagVocabulary> {
        SqliteMemoryStore::tag_vocabulary(self, scope)
    }
    fn undo_tag_mutation(&self, operation_id: &str) -> Result<MergeOperation> {
        SqliteMemoryStore::undo_tag_mutation(self, operation_id)
    }
    fn update_intention_status(&self, id: &str, status: &str) -> Result<bool> {
        SqliteMemoryStore::update_intention_status(self, id, status)
    }
    fn update_memory_state(&self, memory_id: &str, new_state: &str, reason: &str,) -> Result<bool> {
        SqliteMemoryStore::update_memory_state(self, memory_id, new_state, reason)
    }
    fn update_node_content(&self, id: &str, new_content: &str) -> Result<()> {
        SqliteMemoryStore::update_node_content(self, id, new_content)
    }
    fn upsert_by_source(&self, input: IngestInput) -> Result<SourceUpsertResult> {
        SqliteMemoryStore::upsert_by_source(self, input)
    }
    fn verify_stored_receipt_attestation(&self, receipt_id: &str,) -> Result<Option<StoredReceiptAttestationVerification>> {
        SqliteMemoryStore::verify_stored_receipt_attestation(self, receipt_id)
    }
}

// ============================================================================
// CONNECTOR SYNC (#57) — idempotent external-source ingestion
// ============================================================================

// `SourceUpsertOutcome`, `SourceUpsertResult`, `ConnectorCursor`, and
// `ReconcileReport` are defined in (and re-exported from)
// `crate::storage::types`.

impl SqliteMemoryStore {}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
// Reopen-style durability tests: run with the v3-engine harness feature
// (`cargo test -p vestige-core --features v3-engine`); the guard stays
// armed for plain test builds (audit: fresh-install create-then-refuse).
#[cfg(all(test, feature = "v3-engine"))]
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
        ("sqlite/ingest.rs", include_str!("ingest.rs")),
        ("sqlite/lifecycle.rs", include_str!("lifecycle.rs")),
        ("sqlite/merge.rs", include_str!("merge.rs")),
        ("sqlite/purge.rs", include_str!("purge.rs")),
        ("sqlite/records.rs", include_str!("records.rs")),
        ("sqlite/search.rs", include_str!("search.rs")),
        ("sqlite/sync.rs", include_str!("sync.rs")),
        ("edges.rs", include_str!("../edges.rs")),
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
mod actors;
mod connectors;
mod ingest;
mod lifecycle;
mod merge;
mod purge;
mod records;
mod search;
mod sync;

pub use actors::{ActorMutationOutcome, EndorsementEventRecord};
