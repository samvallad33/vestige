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
use uuid::Uuid;

use crate::fsrs::{
    DEFAULT_DECAY, FSRSScheduler, FSRSState, LearningState, Rating, retrievability_with_decay,
};
use crate::fts::sanitize_fts5_query;
use crate::memory::{
    ConsolidationResult, IngestInput, KnowledgeNode, MatchType, MemoryStats, RecallInput,
    SearchMode, SearchResult,
};
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
use crate::memory::{EmbeddingResult, SimilarityResult};
use crate::storage::portable::{
    PORTABLE_ARCHIVE_FORMAT, PortableArchive, PortableImportMode, PortableImportReport,
    PortableTable, PortableValue, encode_hex,
};

#[cfg(feature = "embeddings")]
use crate::embeddings::EmbeddingService;
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
use crate::embeddings::{EMBEDDING_DIMENSIONS, Embedding, matryoshka_truncate};

#[cfg(feature = "vector-search")]
use crate::search::{VectorIndex, linear_combination};

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
}

/// Storage result type
pub type Result<T> = std::result::Result<T, StorageError>;

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
}

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

const DATA_DIR_ENV: &str = "VESTIGE_DATA_DIR";
const DATABASE_FILE: &str = "vestige.db";
const VESTIGE_DISABLE_VECTOR_SEARCH: &str = "VESTIGE_DISABLE_VECTOR_SEARCH";

/// Main storage struct with integrated embedding and vector search
///
/// Uses separate reader/writer connections for interior mutability.
/// All methods take `&self` (not `&mut self`), making Storage `Send + Sync`
/// so the MCP layer can use `Arc<Storage>` instead of `Arc<Mutex<Storage>>`.
pub struct SqliteMemoryStore {
    db_path: PathBuf,
    writer: Mutex<Connection>,
    reader: Mutex<Connection>,
    scheduler: Mutex<FSRSScheduler>,
    #[cfg(feature = "embeddings")]
    embedding_service: EmbeddingService,
    #[cfg(feature = "vector-search")]
    vector_index: Option<Mutex<VectorIndex>>,
    /// LRU cache for query embeddings to avoid re-embedding repeated queries
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    query_cache: Option<Mutex<LruCache<String, Vec<f32>>>>,
    /// Cached model signature. `None` until the first embedding is written.
    registered_model: std::sync::RwLock<Option<crate::storage::memory_store::ModelSignature>>,
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
        reason: impl Into<String>,
    ) -> Result<SmartIngestResult> {
        let node = self.ingest(input)?;
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

    /// Apply PRAGMAs and optional encryption to a connection
    fn configure_connection(conn: &Connection) -> Result<()> {
        // Apply encryption key if SQLCipher is enabled and key is provided
        #[cfg(feature = "encryption")]
        {
            if let Ok(key) = std::env::var("VESTIGE_ENCRYPTION_KEY") {
                if !key.is_empty() {
                    conn.pragma_update(None, "key", &key)?;
                }
            }
        }

        // Configure SQLite for performance
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA cache_size = -64000;
             PRAGMA temp_store = MEMORY;
             PRAGMA foreign_keys = ON;
             PRAGMA busy_timeout = 5000;
             PRAGMA mmap_size = 268435456;
             PRAGMA journal_size_limit = 67108864;
             PRAGMA optimize = 0x10002;",
        )?;

        Ok(())
    }

    /// Create new storage instance
    pub fn new(db_path: Option<PathBuf>) -> Result<Self> {
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

        Self::configure_connection(&writer_conn)?;

        // Apply migrations on writer only
        super::migrations::apply_migrations(&writer_conn)?;

        // Open reader connection to same path
        let reader_conn = Connection::open(&path)?;
        Self::configure_connection(&reader_conn)?;

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
            writer: Mutex::new(writer_conn),
            reader: Mutex::new(reader_conn),
            scheduler: Mutex::new(FSRSScheduler::default()),
            #[cfg(feature = "embeddings")]
            embedding_service,
            #[cfg(feature = "vector-search")]
            vector_index,
            #[cfg(all(feature = "embeddings", feature = "vector-search"))]
            query_cache,
            registered_model: std::sync::RwLock::new(None),
        };

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

    /// Data directory containing the SQLite database and sidecar folders.
    pub fn data_dir(&self) -> &Path {
        self.db_path.parent().unwrap_or_else(|| Path::new("."))
    }

    /// Sidecar directory for files belonging to this storage instance.
    pub fn sidecar_dir(&self, name: &str) -> PathBuf {
        self.data_dir().join(name)
    }

    /// Load existing embeddings into vector index
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn load_embeddings_into_index(&self) -> Result<()> {
        let Some(index) = self.vector_index.as_ref() else {
            return Ok(());
        };

        let mut index = index
            .lock()
            .map_err(|_| StorageError::Init("Vector index lock poisoned".to_string()))?;
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;

        let mut stmt = reader.prepare("SELECT node_id, embedding, model FROM node_embeddings")?;

        let embeddings: Vec<(String, Vec<u8>, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .filter_map(|r| r.ok())
            .collect();

        drop(stmt);
        drop(reader);

        *index = VectorIndex::new().map_err(|e| {
            StorageError::Init(format!("Failed to rebuild vector index before load: {}", e))
        })?;

        let mut load_failures = 0u32;
        let mut skipped_model_mismatches = 0u32;
        let active_model = self.embedding_service.model_name();
        for (node_id, embedding_bytes, model_name) in embeddings {
            if !Self::embedding_model_matches_active(&model_name, active_model) {
                skipped_model_mismatches += 1;
                continue;
            }

            if let Some(embedding) = Embedding::from_bytes(&embedding_bytes) {
                // Handle Matryoshka models explicitly. Do not silently truncate
                // unknown embedding families into the active 256d index.
                let vector = if embedding.dimensions != EMBEDDING_DIMENSIONS {
                    let model_lower = model_name.to_ascii_lowercase();
                    if model_lower.contains("nomic") || model_lower.contains("qwen3") {
                        matryoshka_truncate(embedding.vector)
                    } else {
                        load_failures += 1;
                        tracing::warn!(
                            node_id = %node_id,
                            model = %model_name,
                            dimensions = embedding.dimensions,
                            expected = EMBEDDING_DIMENSIONS,
                            "Skipping embedding with incompatible dimensions"
                        );
                        continue;
                    }
                } else {
                    embedding.vector
                };
                if let Err(e) = index.add(&node_id, &vector) {
                    load_failures += 1;
                    tracing::warn!("Failed to load embedding for {}: {}", node_id, e);
                }
            }
        }
        if load_failures > 0 {
            tracing::error!(
                count = load_failures,
                "Vector index: {} embeddings failed to load",
                load_failures
            );
        }
        if skipped_model_mismatches > 0 {
            tracing::info!(
                count = skipped_model_mismatches,
                active_model = active_model,
                "Vector index skipped embeddings from a different model family; run consolidation to re-embed them"
            );
        }

        Ok(())
    }

    /// Ingest a new memory
    pub fn ingest(&self, input: IngestInput) -> Result<KnowledgeNode> {
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
                    source_system, source_id, source_url, source_updated_at,
                    content_hash, synced_at, source_project, source_type, source_author
                ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6,
                    ?7, ?8, ?9, ?10, ?11,
                    ?12, ?13, ?14,
                    ?15, ?16, ?17, ?18,
                    ?19, ?20, ?21, ?22, ?23, ?24,
                    '[]', '{}',
                    ?25, ?26, ?27, ?28,
                    ?29, ?30, ?31, ?32, ?33
                )",
                params![
                    id,
                    input.content,
                    input.node_type,
                    now.to_rfc3339(),
                    now.to_rfc3339(),
                    now.to_rfc3339(),
                    fsrs_state.stability * sentiment_boost,
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
        self.smart_ingest_excluding(input, &[])
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
        use crate::advanced::prediction_error::{
            CandidateMemory, GateDecision, PredictionErrorGate, UpdateType,
        };

        // Generate embedding for new content
        if !self.embedding_service.is_ready() {
            return self.regular_ingest_result(
                input,
                "Embeddings not available, falling back to regular ingest",
            );
        }

        if !self.vector_search_available() {
            return self.regular_ingest_result(
                input,
                "Vector search unavailable, falling back to regular ingest",
            );
        }

        let new_embedding = self
            .embedding_service
            .embed(&input.content)
            .map_err(|e| StorageError::Init(format!("Embedding failed: {}", e)))?;

        // Find similar memories using semantic search
        let similar = self.semantic_search_raw(&input.content, 10)?;

        // Build candidate memories
        let mut candidates: Vec<CandidateMemory> = Vec::new();
        for (node_id, _similarity) in similar.iter() {
            if excluded_node_ids.iter().any(|id| id == node_id) {
                continue;
            }
            if let Some(node) = self.get_node(node_id)? {
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
        let decision = gate.evaluate(&input.content, &new_embedding.vector, &candidates);

        match decision {
            GateDecision::Create {
                prediction_error,
                related_memory_ids,
                reason,
                ..
            } => {
                // Create new memory
                let node = self.ingest(input)?;
                Ok(SmartIngestResult {
                    decision: "create".to_string(),
                    node,
                    superseded_id: None,
                    similarity: None,
                    prediction_error: Some(prediction_error),
                    reason: if related_memory_ids.is_empty() {
                        format!("Created new memory: {:?}", reason)
                    } else {
                        format!(
                            "Created new memory: {:?}. Semantically similar (not linked): {:?}",
                            reason, related_memory_ids
                        )
                    },
                    previous_content: None,
                    merged_from: None,
                    merge_preview: None,
                })
            }
            GateDecision::Update {
                target_id,
                similarity,
                update_type,
                prediction_error,
            } => {
                match update_type {
                    UpdateType::Reinforce => {
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

                        self.update_node_content(&target_id, &merged_content)?;
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
                        })
                    }
                    UpdateType::Replace => {
                        // Replace content entirely
                        let existing = self
                            .get_node(&target_id)?
                            .ok_or_else(|| StorageError::NotFound(target_id.clone()))?;
                        let previous_content = existing.content;

                        self.update_node_content(&target_id, &input.content)?;
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

                        self.update_node_content(&target_id, &merged_content)?;
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
                // Demote the old memory and create new
                self.demote_memory(&old_memory_id)?;

                // Create the new improved memory
                let node = self.ingest(input)?;

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
                })
            }
            GateDecision::Merge {
                memory_ids,
                avg_similarity,
                strategy,
            } => {
                // For now, create new and link to existing
                let node = self.ingest(input)?;

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
        let mut stmt =
            reader.prepare("SELECT embedding, model FROM node_embeddings WHERE node_id = ?1")?;

        let embedding_row: Option<(Vec<u8>, String)> = stmt
            .query_row(params![node_id], |row| Ok((row.get(0)?, row.get(1)?)))
            .optional()?;

        Ok(embedding_row.and_then(|(bytes, model)| {
            Self::embedding_vector_for_active_model(
                &bytes,
                &model,
                self.embedding_service.model_name(),
            )
        }))
    }

    /// Get all embedding vectors for duplicate detection
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn get_all_embeddings(&self) -> Result<Vec<(String, Vec<f32>)>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare("SELECT node_id, embedding, model FROM node_embeddings")?;
        let active_model = self.embedding_service.model_name();

        let results: Vec<(String, Vec<f32>)> = stmt
            .query_map([], |row| {
                let node_id: String = row.get(0)?;
                let embedding_bytes: Vec<u8> = row.get(1)?;
                let model: String = row.get(2)?;
                Ok((node_id, embedding_bytes, model))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|(id, bytes, model)| {
                Self::embedding_vector_for_active_model(&bytes, &model, active_model)
                    .map(|vector| (id, vector))
            })
            .collect();

        Ok(results)
    }

    /// Fallback for builds without local embeddings/vector search.
    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    pub fn get_node_embedding(&self, _node_id: &str) -> Result<Option<Vec<f32>>> {
        Ok(None)
    }

    /// Update the content of an existing node
    pub fn update_node_content(&self, id: &str, new_content: &str) -> Result<()> {
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
            // Generate new embedding
            if let Err(e) = self.generate_embedding_for_node(id, new_content) {
                tracing::warn!("Failed to regenerate embedding for {}: {}", id, e);
            }
        }

        Ok(())
    }

    /// Generate embedding for a node
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn generate_embedding_for_node(&self, node_id: &str, content: &str) -> Result<()> {
        if !self.embedding_service.is_ready() {
            return Ok(());
        }

        let embedding = self
            .embedding_service
            .embed(content)
            .map_err(|e| StorageError::Init(format!("Embedding failed: {}", e)))?;
        let model_name = self.embedding_service.model_name();

        let now = Utc::now();

        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            writer.execute(
                "INSERT OR REPLACE INTO node_embeddings (node_id, embedding, dimensions, model, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    node_id,
                    embedding.to_bytes(),
                    embedding.dimensions as i32,
                    model_name,
                    now.to_rfc3339(),
                ],
            )?;

            writer.execute(
                "UPDATE knowledge_nodes SET has_embedding = 1, embedding_model = ?2 WHERE id = ?1",
                params![node_id, model_name],
            )?;
        }

        if let Some(index) = self.vector_index.as_ref() {
            let mut index = index
                .lock()
                .map_err(|_| StorageError::Init("Vector index lock poisoned".to_string()))?;
            index
                .add(node_id, &embedding.vector)
                .map_err(|e| StorageError::Init(format!("Vector index add failed: {}", e)))?;
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

        // Auto-strengthen memories on access (Testing Effect - Roediger & Karpicke 2006)
        // This implements "use it or lose it" - accessed memories get stronger
        let ids: Vec<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
        let _ = self.strengthen_batch_on_access(&ids); // Ignore errors, don't fail recall

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

    /// Passively strengthen a memory when it's accessed (recalled/searched).
    /// Implements the Testing Effect (Roediger & Karpicke 2006) + v1.4.0
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

        // Log access for ACT-R activation computation
        let _ = self.log_access(id, "search_hit");

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

    /// Batch strengthen multiple memories on access
    pub fn strengthen_batch_on_access(&self, ids: &[&str]) -> Result<()> {
        for id in ids {
            self.strengthen_on_access(id)?;
            // Also record access in memory_states for audit trail (Bug #1 fix)
            let _ = self.record_memory_access(id);
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

    /// Log a memory access event for ACT-R activation computation
    fn log_access(&self, node_id: &str, access_type: &str) -> Result<()> {
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

        // Strong boost: +0.2 retrieval, +0.1 retention
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
                    stability = stability * 1.5
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

    /// Demote a memory (thumbs down) - used when a memory led to a bad outcome
    /// Significantly reduces retrieval strength so better alternatives surface
    /// Does NOT delete - the memory stays for reference but ranks lower
    pub fn demote_memory(&self, id: &str) -> Result<KnowledgeNode> {
        let now = Utc::now();

        // Strong penalty: -0.3 retrieval, -0.15 retention, halve stability
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            writer.execute(
                "UPDATE knowledge_nodes SET
                    last_accessed = ?1,
                    retrieval_strength = MAX(0.05, retrieval_strength - 0.30),
                    retention_strength = MAX(0.05, retention_strength - 0.15),
                    stability = stability * 0.5
                WHERE id = ?2",
                params![now.to_rfc3339(), id],
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
            writer.execute(
                "UPDATE knowledge_nodes SET
                    last_accessed = ?1,
                    suppression_count = COALESCE(suppression_count, 0) + 1,
                    suppressed_at = ?1,
                    retrieval_strength = MAX(0.05, retrieval_strength - 0.35),
                    retention_strength = MAX(0.05, retention_strength - 0.20),
                    stability = stability * 0.4
                WHERE id = ?2",
                params![now.to_rfc3339(), id],
            )?;
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
            writer.execute(
                "UPDATE knowledge_nodes SET
                    suppression_count = MAX(0, COALESCE(suppression_count, 0) - 1),
                    suppressed_at = CASE
                        WHEN COALESCE(suppression_count, 0) - 1 <= 0 THEN NULL
                        ELSE suppressed_at
                    END,
                    retrieval_strength = MIN(1.0, retrieval_strength + 0.15),
                    retention_strength = MIN(1.0, retention_strength + 0.10),
                    stability = stability * 1.25
                WHERE id = ?1",
                params![id],
            )?;
        }

        let _ = self.log_access(id, "reverse_suppress");

        self.get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
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
        let active_embedding_model = Some(self.embedding_service.model_name().to_string());
        #[cfg(not(feature = "embeddings"))]
        let active_embedding_model = None;

        #[cfg(feature = "embeddings")]
        let (nodes_with_active_embeddings, nodes_with_mismatched_embeddings) = {
            let active_model = active_embedding_model.as_deref().unwrap_or_default();
            let model_pattern = Self::active_embedding_model_like_pattern(active_model);
            let active_count: i64 = reader.query_row(
                "SELECT COUNT(*)
                 FROM knowledge_nodes kn
                 WHERE kn.has_embedding = 1
                   AND EXISTS (
                       SELECT 1 FROM node_embeddings ne
                       WHERE ne.node_id = kn.id
                         AND ne.model LIKE ?1
                   )",
                params![&model_pattern],
                |row| row.get(0),
            )?;
            let mismatched_count: i64 = reader.query_row(
                "SELECT COUNT(*)
                 FROM knowledge_nodes kn
                 WHERE kn.has_embedding = 1
                   AND NOT EXISTS (
                       SELECT 1 FROM node_embeddings ne
                       WHERE ne.node_id = kn.id
                         AND ne.model LIKE ?1
                   )",
                params![&model_pattern],
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

        // Convenience: embedding-coverage NULL count. Defined as the number
        // of knowledge_nodes with NO matching row in node_embeddings. This is
        // distinct from `nodes_with_embeddings` in MemoryStats (which uses
        // the `has_embedding` column flag); we compute the join-based truth
        // here so audit scripts can detect drift between the flag and the
        // actual embeddings table.
        let embedding_null_count: i64 = reader
            .query_row(
                "SELECT COUNT(*) FROM knowledge_nodes kn
                 WHERE NOT EXISTS (
                     SELECT 1 FROM node_embeddings ne WHERE ne.node_id = kn.id
                 )",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        #[cfg(feature = "embeddings")]
        let active_embedding_model = Some(self.embedding_service.model_name().to_string());
        #[cfg(not(feature = "embeddings"))]
        let active_embedding_model: Option<String> = None;

        #[cfg(feature = "embeddings")]
        let active_embedding_dimensions: Option<u32> =
            Some(self.embedding_service.dimensions() as u32);
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

    /// Delete a node
    pub fn delete_node(&self, id: &str) -> Result<bool> {
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = writer.transaction()?;
        if Self::node_exists(&tx, id)? {
            Self::record_sync_tombstone(&tx, "knowledge_nodes", id, "delete_node")?;
        }
        let rows = tx.execute("DELETE FROM knowledge_nodes WHERE id = ?1", params![id])?;
        tx.commit()?;

        // Clean up vector index to prevent stale search results
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        if rows > 0
            && let Some(index) = self.vector_index.as_ref()
            && let Ok(mut index) = index.lock()
        {
            let _ = index.remove(id);
        }

        Ok(rows > 0)
    }

    /// Permanently purge a memory's content and embeddings.
    ///
    /// Unlike `delete_node`, purge also scrubs non-FK JSON references in
    /// `insights.source_memories`, detaches temporal-summary children, and
    /// writes a content-free deletion tombstone for audit/sync.
    pub fn purge_node(&self, id: &str, reason: Option<&str>) -> Result<PurgeReport> {
        let deleted_at = Utc::now();
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = writer.transaction()?;

        let node = tx
            .prepare("SELECT * FROM knowledge_nodes WHERE id = ?1")?
            .query_row(params![id], Self::row_to_node)
            .optional()?;

        let Some(node) = node else {
            return Ok(PurgeReport {
                memory_id: id.to_string(),
                deleted: false,
                deleted_at,
                edges_pruned: 0,
                insights_rewritten: 0,
                insights_deleted: 0,
                children_orphaned: 0,
            });
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
            stmt.query_map(params![pattern], |row| Ok((row.get(0)?, row.get(1)?)))?
                .filter_map(|row| row.ok())
                .collect()
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

        tx.execute(
            "UPDATE composition_members SET preview = NULL WHERE memory_id = ?1",
            params![id],
        )?;

        let tags_json = serde_json::to_string(&node.tags).unwrap_or_else(|_| "[]".to_string());
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
                id,
                deleted_at.to_rfc3339(),
                reason,
                node.node_type,
                tags_json,
                edges_pruned,
                insights_rewritten,
                insights_deleted,
                children_orphaned,
            ],
        )?;

        Self::record_sync_tombstone(&tx, "knowledge_nodes", id, "purge_node")?;
        tx.execute("DELETE FROM knowledge_nodes WHERE id = ?1", params![id])?;
        tx.commit()?;

        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        if let Some(index) = self.vector_index.as_ref()
            && let Ok(mut index) = index.lock()
        {
            let _ = index.remove(id);
        }

        Ok(PurgeReport {
            memory_id: id.to_string(),
            deleted: true,
            deleted_at,
            edges_pruned,
            insights_rewritten,
            insights_deleted,
            children_orphaned,
        })
    }

    fn node_exists(conn: &Connection, id: &str) -> Result<bool> {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM knowledge_nodes WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    fn record_sync_tombstone(
        conn: &Connection,
        table_name: &str,
        row_id: &str,
        reason: &str,
    ) -> Result<()> {
        conn.execute(
            "INSERT INTO sync_tombstones (table_name, row_id, deleted_at, reason)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(table_name, row_id) DO UPDATE SET
                deleted_at = excluded.deleted_at,
                reason = excluded.reason",
            params![table_name, row_id, Utc::now().to_rfc3339(), reason],
        )?;
        Ok(())
    }

    /// Search with full-text search
    pub fn search(&self, query: &str, limit: i32) -> Result<Vec<KnowledgeNode>> {
        let sanitized_query = sanitize_fts5_query(query);

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

            for (idx, row) in rows.enumerate() {
                let (node, rank) = row?;
                if !Self::node_matches_type_filters(&node, include_types, exclude_types) {
                    continue;
                }
                let base_score = (1.0 / (idx as f32 + 1.0)).max((-rank as f32).max(0.0));
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

    /// Check if embedding service is ready
    #[cfg(feature = "embeddings")]
    pub fn is_embedding_ready(&self) -> bool {
        self.embedding_service.is_ready()
    }

    #[cfg(not(feature = "embeddings"))]
    pub fn is_embedding_ready(&self) -> bool {
        false
    }

    /// Initialize the embedding service explicitly
    /// Call this at startup to catch initialization errors early
    #[cfg(feature = "embeddings")]
    pub fn init_embeddings(&self) -> Result<()> {
        self.embedding_service.init().map_err(|e| {
            StorageError::Init(format!("Embedding service initialization failed: {}", e))
        })
    }

    #[cfg(not(feature = "embeddings"))]
    pub fn init_embeddings(&self) -> Result<()> {
        Ok(()) // No-op when embeddings feature is disabled
    }

    /// Get query embedding from cache or compute it
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn get_query_embedding(&self, query: &str) -> Result<Vec<f32>> {
        let cache_key = format!("{}\0{}", self.embedding_service.model_name(), query);
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

        // Not in cache, compute embedding
        let embedding = self
            .embedding_service
            .embed_query(query)
            .map_err(|e| StorageError::Init(format!("Failed to embed query: {}", e)))?;

        // Store in cache
        {
            let mut cache = index_cache
                .lock()
                .map_err(|_| StorageError::Init("Query cache lock poisoned".to_string()))?;
            cache.put(cache_key, embedding.vector.clone());
        }

        Ok(embedding.vector)
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

        if !self.embedding_service.is_ready() {
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
            if self.vector_search_available() && self.embedding_service.is_ready() {
                self.semantic_search_raw(query, limit * overfetch_factor)?
            } else {
                vec![]
            };

        let combined = if !semantic_results.is_empty() {
            linear_combination(
                &keyword_results,
                &semantic_results,
                keyword_weight,
                semantic_weight,
            )
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

                let weighted_score = match (keyword_score, semantic_score) {
                    (Some(kw), Some(sem)) => kw * keyword_weight + sem * semantic_weight,
                    (Some(kw), None) => kw * keyword_weight,
                    (None, Some(sem)) => sem * semantic_weight,
                    (None, None) => combined_score,
                };

                results.push(SearchResult {
                    node,
                    keyword_score,
                    semantic_score,
                    combined_score: weighted_score,
                    match_type,
                });
            }
        }

        // Three-signal reranking (Park et al. Generative Agents 2023)
        // final_score = 0.2*recency + 0.3*importance + 0.5*relevance
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

            let relevance = result.combined_score as f64;

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
            .filter_map(|r| r.ok())
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

    /// Semantic search returning scores
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn semantic_search_raw(&self, query: &str, limit: i32) -> Result<Vec<(String, f32)>> {
        if !self.vector_search_available() {
            return Ok(vec![]);
        }
        if !self.embedding_service.is_ready() {
            return Ok(vec![]);
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
        if !self.embedding_service.is_ready() {
            self.embedding_service.init().map_err(|e| {
                StorageError::Init(format!("Failed to init embedding service: {}", e))
            })?;
        }

        let mut result = EmbeddingResult::default();
        let active_model = self.embedding_service.model_name();
        let nodes = self.embedding_regeneration_candidates(node_ids, force)?;

        for (id, content, stored_model) in nodes {
            if !force {
                let (has_emb, stored_model): (i32, Option<String>) = self
                    .reader
                    .lock()
                    .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?
                    .query_row(
                        "SELECT COALESCE(kn.has_embedding, 0), COALESCE(ne.model, kn.embedding_model)
                         FROM knowledge_nodes kn
                         LEFT JOIN node_embeddings ne ON ne.node_id = kn.id
                         WHERE kn.id = ?1",
                        params![&id],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .unwrap_or((0, stored_model));

                if has_emb == 1
                    && stored_model.as_deref().is_some_and(|model| {
                        Self::embedding_model_matches_active(model, active_model)
                    })
                {
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
                "SELECT kn.id, kn.content, COALESCE(ne.model, kn.embedding_model) AS embedding_model
                 FROM knowledge_nodes kn
                 LEFT JOIN node_embeddings ne ON ne.node_id = kn.id
                 WHERE kn.id IN ({})",
                placeholders
            );

            let mut stmt = reader.prepare(&query)?;
            let params: Vec<&dyn rusqlite::ToSql> =
                ids.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
            let rows = stmt.query_map(params.as_slice(), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })?;
            return Ok(rows.filter_map(|r| r.ok()).collect());
        }

        if force {
            let mut stmt =
                reader.prepare("SELECT id, content, embedding_model FROM knowledge_nodes")?;
            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })?;
            return Ok(rows.filter_map(|r| r.ok()).collect());
        }

        let active_model = self.embedding_service.model_name();
        let model_pattern = Self::active_embedding_model_like_pattern(active_model);
        let mut stmt = reader.prepare(
            "SELECT kn.id, kn.content, COALESCE(ne.model, kn.embedding_model) AS embedding_model
             FROM knowledge_nodes kn
             LEFT JOIN node_embeddings ne ON ne.node_id = kn.id
             WHERE kn.has_embedding = 0
                OR kn.has_embedding IS NULL
                OR ne.node_id IS NULL
                OR COALESCE(ne.model, kn.embedding_model, '') NOT LIKE ?1",
        )?;
        let rows = stmt.query_map(params![model_pattern], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
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
                    .filter_map(|r| r.ok())
                    .collect()
            };

            if batch.is_empty() {
                break;
            }

            let batch_len = batch.len() as i64;

            // Write batch using writer transaction
            {
                let mut writer = self
                    .writer
                    .lock()
                    .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
                let tx = writer.transaction()?;

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

        // v1.5.0: Use SleepConsolidation for structured consolidation
        let sleep = crate::SleepConsolidation::new();

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
                    .filter_map(|r| r.ok())
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

        // 19. Retention Target System — auto-GC if avg retention below target
        let mut gc_triggered = false;
        {
            let retention_target: f64 = std::env::var("VESTIGE_RETENTION_TARGET")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.8);

            let avg_retention = self.get_avg_retention().unwrap_or(1.0);
            let total = self.get_stats().map(|s| s.total_nodes).unwrap_or(0);
            let below_target = self.count_memories_below_retention(0.3).unwrap_or(0);

            if avg_retention < retention_target && below_target > 0 {
                let gc_count = self.gc_below_retention(0.3, 30).unwrap_or(0);
                if gc_count > 0 {
                    gc_triggered = true;
                    tracing::info!(
                        avg_retention = avg_retention,
                        target = retention_target,
                        gc_count = gc_count,
                        "Retention target auto-GC: removed {} low-retention memories",
                        gc_count
                    );
                }
            }

            // 20. Save retention snapshot for trend tracking
            let _ = self.save_retention_snapshot(avg_retention, total, below_target, gc_triggered);
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
        })
    }

    /// Auto-deduplicate similar memories during consolidation (episodic → semantic merge)
    ///
    /// Finds clusters with cosine similarity > 0.85, keeps the strongest node,
    /// appends unique content from weaker nodes, and deletes duplicates.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn auto_dedup_consolidation(&self) -> Result<i64> {
        let all_embeddings = self.get_all_embeddings()?;
        let n = all_embeddings.len();

        if !(2..=2000).contains(&n) {
            return Ok(0);
        }

        const SIMILARITY_THRESHOLD: f32 = 0.85;
        let mut merged_count = 0i64;
        let mut consumed: std::collections::HashSet<String> = std::collections::HashSet::new();

        for i in 0..n {
            if consumed.contains(&all_embeddings[i].0) {
                continue;
            }

            let mut cluster: Vec<(usize, f32)> = Vec::new();

            for j in (i + 1)..n {
                if consumed.contains(&all_embeddings[j].0) {
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

            // Update keeper with merged content
            if merged_content != keeper_content {
                let _ = self.update_node_content(&best_id, &merged_content);
            }

            // Delete weak nodes
            for weak_id in &weak_ids {
                let _ = self.delete_node(weak_id);
                consumed.insert(weak_id.clone());
                merged_count += 1;
            }

            consumed.insert(best_id);
        }

        Ok(merged_count)
    }

    /// Compute ACT-R base-level activation for all nodes from access history.
    /// B_i = ln(Σ t_j^(-d)) where t_j = days since j-th access, d = 0.5
    fn compute_act_r_activations(&self) -> Result<i64> {
        const ACT_R_DECAY: f64 = 0.5;
        let now = Utc::now();

        let node_ids: Vec<String> = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            reader
                .prepare("SELECT DISTINCT node_id FROM memory_access_log")?
                .query_map([], |row| row.get(0))?
                .filter_map(|r| r.ok())
                .collect()
        };

        if node_ids.is_empty() {
            return Ok(0);
        }

        let mut count = 0i64;
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = writer.transaction()?;

        for node_id in &node_ids {
            let timestamps: Vec<String> = tx
                .prepare(
                    "SELECT accessed_at FROM memory_access_log
                     WHERE node_id = ?1
                     ORDER BY accessed_at DESC
                     LIMIT 500",
                )?
                .query_map(params![node_id], |row| row.get(0))?
                .filter_map(|r| r.ok())
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

    /// Prune old access log entries (keep last 90 days)
    fn prune_access_log(&self) -> Result<i64> {
        let cutoff = (Utc::now() - Duration::days(90)).to_rfc3339();
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
            .query_row("SELECT COUNT(*) FROM memory_access_log", [], |row| {
                row.get(0)
            })
            .unwrap_or(0);

        if access_count < 100 {
            return Ok(None);
        }

        let mut optimizer = FSRSOptimizer::new();

        let logs: Vec<(String, String, String)> = reader
            .prepare(
                "SELECT mal.node_id, mal.access_type, mal.accessed_at
                 FROM memory_access_log mal
                 ORDER BY mal.accessed_at ASC
                 LIMIT 1000",
            )?
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .filter_map(|r| r.ok())
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

                let rating = match access_type.as_str() {
                    "promote" => 4,
                    "search_hit" => 3,
                    "demote" => 1,
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
        if !self.embedding_service.is_ready()
            && let Err(e) = self.embedding_service.init()
        {
            tracing::warn!("Could not initialize embedding model: {}", e);
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
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = writer.transaction()?;

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
                if shared_tags.is_empty() && shared_terms.is_empty() {
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
                let prior_outcomes = Self::pair_prior_outcomes(&outcome_map, &a.id, &b.id);
                let outcome_signal = Self::outcome_signal(&prior_outcomes);
                let outcome_score_adjustment = Self::outcome_score_adjustment(&prior_outcomes);
                let score = anchor_score
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
            let mut writer = self
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

            let tx = writer.transaction()?;
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
    #[cfg(feature = "cloud-sync")]
    pub fn sync_portable_archive_cloud(
        &self,
        endpoint: &str,
        sync_key: &str,
    ) -> Result<PortableSyncReport> {
        let backend = super::cloud_sync::HttpPortableSyncBackend::new(endpoint, sync_key)?;
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

            if let Some(deleted_at) = Self::tombstone_timestamp(tx, "knowledge_nodes", id)?
                && incoming_updated.is_some_and(|updated| deleted_at >= updated)
            {
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
            let incoming_reason = Self::portable_text(table, row, "reason").map(ToOwned::to_owned);

            let existing_tombstone: Option<(String, Option<String>)> = tx
                .query_row(
                    "SELECT deleted_at, reason FROM sync_tombstones WHERE table_name = ?1 AND row_id = ?2",
                    params![table_name, row_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            let existing_deleted_at = existing_tombstone
                .as_ref()
                .and_then(|(deleted_at, _)| Self::parse_rfc3339_opt(deleted_at));
            let incoming_wins = match (existing_deleted_at, incoming_deleted_at) {
                (Some(existing), Some(incoming)) => incoming >= existing,
                (Some(_), None) => false,
                (None, _) => true,
            };

            let (effective_deleted_at, effective_reason) = if incoming_wins {
                let affected = Self::insert_or_replace_row(tx, "sync_tombstones", table, row)?;
                report.rows_imported += 1;
                if affected == MergeWrite::Inserted {
                    report.rows_inserted += 1;
                } else {
                    report.rows_updated += 1;
                }
                (incoming_deleted_at, incoming_reason)
            } else {
                report.rows_skipped += 1;
                (
                    existing_deleted_at,
                    existing_tombstone.and_then(|(_, reason)| reason),
                )
            };

            if table_name == "knowledge_nodes" {
                let local_updated: Option<String> = tx
                    .query_row(
                        "SELECT updated_at FROM knowledge_nodes WHERE id = ?1",
                        params![row_id],
                        |row| row.get(0),
                    )
                    .optional()?;
                let should_delete = match (
                    local_updated.as_deref().and_then(Self::parse_rfc3339_opt),
                    effective_deleted_at,
                ) {
                    (Some(local), Some(deleted)) => {
                        effective_reason.as_deref() == Some("purge_node") || deleted >= local
                    }
                    (Some(_), None) => true,
                    (None, _) => false,
                };
                if should_delete {
                    tx.execute(
                        "UPDATE composition_members SET preview = NULL WHERE memory_id = ?1",
                        params![row_id],
                    )?;
                    let deleted =
                        tx.execute("DELETE FROM knowledge_nodes WHERE id = ?1", params![row_id])?;
                    report.rows_deleted += deleted;
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
        let deleted_at: Option<String> = tx
            .query_row(
                "SELECT deleted_at FROM sync_tombstones WHERE table_name = ?1 AND row_id = ?2",
                params![table_name, row_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(deleted_at.as_deref().and_then(Self::parse_rfc3339_opt))
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

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn embedding_model_matches_active(stored_model: &str, active_model: &str) -> bool {
        if stored_model == active_model {
            return true;
        }

        let stored = stored_model.to_ascii_lowercase();
        let active = active_model.to_ascii_lowercase();

        if active.contains("qwen3") {
            return stored.contains("qwen3");
        }

        if active.contains("nomic-embed-text-v1.5") {
            return stored.contains("nomic") && stored.contains("v1.5");
        }

        false
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn embedding_model_supports_matryoshka(model_name: &str) -> bool {
        let model = model_name.to_ascii_lowercase();
        model.contains("nomic") || model.contains("qwen3")
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn embedding_vector_for_active_model(
        embedding_bytes: &[u8],
        stored_model: &str,
        active_model: &str,
    ) -> Option<Vec<f32>> {
        if !Self::embedding_model_matches_active(stored_model, active_model) {
            return None;
        }

        let embedding = Embedding::from_bytes(embedding_bytes)?;
        if embedding.dimensions == EMBEDDING_DIMENSIONS {
            Some(embedding.vector)
        } else if Self::embedding_model_supports_matryoshka(stored_model) {
            Some(matryoshka_truncate(embedding.vector))
        } else {
            None
        }
    }

    #[cfg(feature = "embeddings")]
    fn active_embedding_model_like_pattern(active_model: &str) -> String {
        let active = active_model.to_ascii_lowercase();
        if active.contains("qwen3") {
            "%qwen3%".to_string()
        } else if active.contains("nomic-embed-text-v1.5") {
            "%nomic%v1.5%".to_string()
        } else {
            active_model.to_string()
        }
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
            .filter_map(|r| r.ok())
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
                .filter_map(|r| r.ok())
                .collect()
        };

        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        for id in &doomed_ids {
            Self::record_sync_tombstone(&writer, "knowledge_nodes", id, "gc_below_retention")?;
        }
        let deleted = writer.execute(
            "DELETE FROM knowledge_nodes WHERE retention_strength < ?1 AND created_at < ?2",
            params![threshold, cutoff],
        )? as i64;
        drop(writer);

        // Clean up vector index
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        if deleted > 0
            && let Some(index) = self.vector_index.as_ref()
            && let Ok(mut index) = index.lock()
        {
            for id in &doomed_ids {
                let _ = index.remove(id);
            }
        }

        Ok(deleted)
    }

    /// Check for auto-promote candidates: memories accessed 3+ times in last 24h
    pub fn auto_promote_frequent_access(&self) -> Result<i64> {
        let twenty_four_hours_ago = (Utc::now() - Duration::hours(24)).to_rfc3339();
        let now = Utc::now().to_rfc3339();

        // Find memories with 3+ accesses in last 24h
        let candidates: Vec<String> = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            let mut stmt = reader.prepare(
                "SELECT node_id, COUNT(*) as access_count
                 FROM memory_access_log
                 WHERE accessed_at >= ?1
                 GROUP BY node_id
                 HAVING access_count >= 3",
            )?;
            stmt.query_map(params![twenty_four_hours_ago], |row| row.get(0))?
                .filter_map(|r| r.ok())
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

        // Snapshot everything we need to undo, BEFORE mutating.
        let mut undo = serde_json::Map::new();
        undo.insert("plan_id".into(), serde_json::json!(plan_id));
        undo.insert("kind".into(), serde_json::json!(plan.kind.as_str()));
        undo.insert("survivor_id".into(), serde_json::json!(plan.survivor_id));

        match plan.kind {
            PlanKind::Merge => {
                let survivor = self
                    .get_node(&plan.survivor_id)?
                    .ok_or_else(|| StorageError::NotFound(plan.survivor_id.clone()))?;
                undo.insert(
                    "survivor_prev_content".into(),
                    serde_json::json!(survivor.content),
                );
                undo.insert(
                    "survivor_prev_tags".into(),
                    serde_json::json!(survivor.tags),
                );

                // Capture prior valid_until / superseded_by of each absorbed node.
                let mut absorbed = Vec::new();
                for id in &plan.invalidated_ids {
                    let (vu, sb) = self.read_bitemporal(id)?;
                    absorbed.push(serde_json::json!({
                        "id": id,
                        "prev_valid_until": vu,
                        "prev_superseded_by": sb,
                    }));
                }
                undo.insert("absorbed".into(), serde_json::json!(absorbed));

                // Apply: rewrite survivor, invalidate absorbed.
                self.rewrite_survivor(&plan.survivor_id, &plan.result_content, &plan.result_tags)?;
                for id in &plan.invalidated_ids {
                    self.invalidate_node(id, &plan.survivor_id, now)?;
                }
            }
            PlanKind::Supersede => {
                let old_id = &plan.member_ids[0];
                let (vu, sb) = self.read_bitemporal(old_id)?;
                undo.insert(
                    "absorbed".into(),
                    serde_json::json!([{
                        "id": old_id,
                        "prev_valid_until": vu,
                        "prev_superseded_by": sb,
                    }]),
                );
                self.invalidate_node(old_id, &plan.survivor_id, now)?;
            }
        }

        // Record the reversible operation.
        let affected: Vec<String> = {
            let mut v = vec![plan.survivor_id.clone()];
            v.extend(plan.invalidated_ids.clone());
            v
        };
        let signals = serde_json::to_string(&plan.signals).unwrap_or_else(|_| "{}".into());
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            writer.execute(
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
            writer.execute(
                "UPDATE merge_plans SET status = 'applied', applied_at = ?1 WHERE id = ?2",
                params![now.to_rfc3339(), plan_id],
            )?;
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
                    survivor_id, affected_ids, confidence, reason
             FROM merge_operations ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], Self::row_to_operation)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
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
                        survivor_id, affected_ids, confidence, reason
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
            reason: row.get("reason").ok().flatten(),
        })
    }

    /// Read (valid_until, superseded_by) for a node.
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

    /// Bitemporally invalidate a node: stamp valid_until=now and superseded_by,
    /// keeping the row fully queryable (Graphiti-style invalidate, don't delete).
    fn invalidate_node(&self, id: &str, superseded_by: &str, now: DateTime<Utc>) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
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
        // Enforce model registry if embedding is provided
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
        let env = match input.source_envelope.clone() {
            Some(e) if e.has_key() => e,
            // No idempotency key — behave like a normal create.
            _ => {
                let node = self.ingest(input)?;
                return Ok(SourceUpsertResult {
                    outcome: SourceUpsertOutcome::Created,
                    node_id: node.id,
                });
            }
        };

        let source_system = env.source_system.clone().unwrap_or_default();
        let source_id = env.source_id.clone().unwrap_or_default();
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
                     WHERE source_system = ?1 AND source_id = ?2 LIMIT 1",
                    params![source_system, source_id],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
                )
                .optional()?
        };

        let Some((node_id, stored_hash)) = existing else {
            // First time we've seen this record — plain insert carries the
            // envelope through the existing ingest path.
            let node = self.ingest(input)?;
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
            rows.filter_map(|r| r.ok()).collect()
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
    fn test_embedding_model_family_matching() {
        assert!(Storage::embedding_model_matches_active(
            "nomic-embed-text-v1.5",
            "nomic-ai/nomic-embed-text-v1.5",
        ));
        assert!(Storage::embedding_model_matches_active(
            "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-0.6B",
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
        }

        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.nodes_with_mismatched_embeddings, 125);
        assert_eq!(stats.nodes_with_active_embeddings, 0);

        let candidates = storage
            .embedding_regeneration_candidates(None, false)
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
            ..Default::default()
        };

        let node = storage.ingest(input).unwrap();
        assert!(storage.get_node(&node.id).unwrap().is_some());

        let deleted = storage.delete_node(&node.id).unwrap();
        assert!(deleted);
        assert!(storage.get_node(&node.id).unwrap().is_none());
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
                .unwrap()[0]
                .preview
                .is_none(),
            "portable purge merge should scrub target composition previews"
        );

        let writer = target.writer.lock().unwrap();
        let tombstone_count: i64 = writer
            .query_row(
                "SELECT COUNT(*) FROM deletion_tombstones WHERE memory_id = ?1 AND reason = ?2",
                params![node.id, "sync purge test"],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(tombstone_count, 1);
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
                "SELECT COUNT(*) FROM deletion_tombstones WHERE memory_id = ?1 AND reason = ?2",
                params![doomed.id, "user requested hard purge"],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(tombstone_count, 1);

        let members = storage
            .get_composition_members("purge-composition-preview-test")
            .unwrap();
        assert_eq!(members.len(), 1);
        assert!(
            members[0].preview.is_none(),
            "purge should scrub composition member previews for the purged memory"
        );
        let archive_json =
            serde_json::to_string(&storage.export_portable_archive().unwrap()).unwrap();
        assert!(
            !archive_json.contains("Sensitive purge target memory preview leak"),
            "portable archive should not retain purged memory content through composition previews"
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
}
