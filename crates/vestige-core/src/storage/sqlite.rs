//! SQLite Storage Implementation
//!
//! Core storage layer with integrated embeddings and vector search.

use chrono::{DateTime, Duration, Utc};
use directories::ProjectDirs;
#[cfg(feature = "embeddings")]
use lru::LruCache;
use rusqlite::{Connection, OptionalExtension, params};
#[cfg(feature = "embeddings")]
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Mutex;
use uuid::Uuid;

use crate::fsrs::{
    DEFAULT_DECAY, FSRSScheduler, FSRSState, LearningState, Rating, retrievability_with_decay,
};
use crate::fts::sanitize_fts5_query;
use crate::memory::{
    ConsolidationResult, IngestInput, KnowledgeNode, MemoryStats, RecallInput, SearchMode,
};
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
use crate::memory::{EmbeddingResult, MatchType, SearchResult, SimilarityResult};

#[cfg(feature = "embeddings")]
use crate::embeddings::{EMBEDDING_DIMENSIONS, Embedding, EmbeddingService, matryoshka_truncate};

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
}

// ============================================================================
// STORAGE
// ============================================================================

/// Main storage struct with integrated embedding and vector search
///
/// Uses separate reader/writer connections for interior mutability.
/// All methods take `&self` (not `&mut self`), making Storage `Send + Sync`
/// so the MCP layer can use `Arc<Storage>` instead of `Arc<Mutex<Storage>>`.
pub struct Storage {
    writer: Mutex<Connection>,
    reader: Mutex<Connection>,
    scheduler: Mutex<FSRSScheduler>,
    #[cfg(feature = "embeddings")]
    embedding_service: EmbeddingService,
    #[cfg(feature = "vector-search")]
    vector_index: Mutex<VectorIndex>,
    /// LRU cache for query embeddings to avoid re-embedding repeated queries
    #[cfg(feature = "embeddings")]
    query_cache: Mutex<LruCache<String, Vec<f32>>>,
}

impl Storage {
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
            None => {
                let proj_dirs = ProjectDirs::from("com", "vestige", "core").ok_or_else(|| {
                    StorageError::Init("Could not determine project directories".to_string())
                })?;

                let data_dir = proj_dirs.data_dir();
                std::fs::create_dir_all(data_dir)?;
                // Restrict directory permissions to owner-only on Unix
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let perms = std::fs::Permissions::from_mode(0o700);
                    let _ = std::fs::set_permissions(data_dir, perms);
                }
                data_dir.join("vestige.db")
            }
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
        let vector_index = VectorIndex::new()
            .map_err(|e| StorageError::Init(format!("Failed to create vector index: {}", e)))?;

        // Initialize LRU cache for query embeddings (capacity: 100 queries)
        // SAFETY: 100 is always non-zero, this cannot fail
        #[cfg(feature = "embeddings")]
        let query_cache = Mutex::new(LruCache::new(
            NonZeroUsize::new(100).expect("100 is non-zero"),
        ));

        let storage = Self {
            writer: Mutex::new(writer_conn),
            reader: Mutex::new(reader_conn),
            scheduler: Mutex::new(FSRSScheduler::default()),
            #[cfg(feature = "embeddings")]
            embedding_service,
            #[cfg(feature = "vector-search")]
            vector_index: Mutex::new(vector_index),
            #[cfg(feature = "embeddings")]
            query_cache,
        };

        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        storage.load_embeddings_into_index()?;

        Ok(storage)
    }

    /// Load existing embeddings into vector index
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn load_embeddings_into_index(&self) -> Result<()> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;

        let mut stmt = reader.prepare("SELECT node_id, embedding FROM node_embeddings")?;

        let embeddings: Vec<(String, Vec<u8>)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();

        drop(stmt);
        drop(reader);

        let mut index = self
            .vector_index
            .lock()
            .map_err(|_| StorageError::Init("Vector index lock poisoned".to_string()))?;

        let mut load_failures = 0u32;
        for (node_id, embedding_bytes) in embeddings {
            if let Some(embedding) = Embedding::from_bytes(&embedding_bytes) {
                // Handle Matryoshka migration: old 768-dim → truncate to 256-dim
                let vector = if embedding.dimensions != EMBEDDING_DIMENSIONS {
                    matryoshka_truncate(embedding.vector)
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
                    source, tags, valid_from, valid_until, has_embedding, embedding_model
                ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6,
                    ?7, ?8, ?9, ?10, ?11,
                    ?12, ?13, ?14,
                    ?15, ?16, ?17, ?18,
                    ?19, ?20, ?21, ?22, ?23, ?24
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
        use crate::advanced::prediction_error::{
            CandidateMemory, GateDecision, PredictionErrorGate, UpdateType,
        };

        // Generate embedding for new content
        if !self.embedding_service.is_ready() {
            // Fall back to regular ingest if embeddings not available
            let node = self.ingest(input)?;
            return Ok(SmartIngestResult {
                decision: "create".to_string(),
                node,
                superseded_id: None,
                similarity: None,
                prediction_error: Some(1.0),
                reason: "Embeddings not available, falling back to regular ingest".to_string(),
            });
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
                        })
                    }
                    UpdateType::Merge | UpdateType::Append => {
                        // Update the existing memory with merged content
                        let existing = self
                            .get_node(&target_id)?
                            .ok_or_else(|| StorageError::NotFound(target_id.clone()))?;

                        let merged_content = format!(
                            "{}\n\n[Updated {}]\n{}",
                            existing.content,
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
                        })
                    }
                    UpdateType::Replace => {
                        // Replace content entirely
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
                        })
                    }
                    UpdateType::AddContext => {
                        // Add as context without modifying main content
                        let existing = self
                            .get_node(&target_id)?
                            .ok_or_else(|| StorageError::NotFound(target_id.clone()))?;

                        let merged_content =
                            format!("{}\n\n---\nContext: {}", existing.content, input.content);

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
            reader.prepare("SELECT embedding FROM node_embeddings WHERE node_id = ?1")?;

        let embedding_bytes: Option<Vec<u8>> = stmt
            .query_row(params![node_id], |row| row.get(0))
            .optional()?;

        Ok(embedding_bytes
            .and_then(|bytes| crate::embeddings::Embedding::from_bytes(&bytes).map(|e| e.vector)))
    }

    /// Get all embedding vectors for duplicate detection
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn get_all_embeddings(&self) -> Result<Vec<(String, Vec<f32>)>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare("SELECT node_id, embedding FROM node_embeddings")?;

        let results: Vec<(String, Vec<f32>)> = stmt
            .query_map([], |row| {
                let node_id: String = row.get(0)?;
                let embedding_bytes: Vec<u8> = row.get(1)?;
                Ok((node_id, embedding_bytes))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|(id, bytes)| {
                crate::embeddings::Embedding::from_bytes(&bytes).map(|e| (id, e.vector))
            })
            .collect();

        Ok(results)
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
            if let Ok(mut index) = self.vector_index.lock() {
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
                    EMBEDDING_DIMENSIONS as i32,
                    "nomic-embed-text-v1.5",
                    now.to_rfc3339(),
                ],
            )?;

            writer.execute(
                "UPDATE knowledge_nodes SET has_embedding = 1, embedding_model = 'nomic-embed-text-v1.5' WHERE id = ?1",
                params![node_id],
            )?;
        }

        let mut index = self
            .vector_index
            .lock()
            .map_err(|_| StorageError::Init("Vector index lock poisoned".to_string()))?;
        index
            .add(node_id, &embedding.vector)
            .map_err(|e| StorageError::Init(format!("Vector index add failed: {}", e)))?;

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

    /// Parse RFC3339 timestamp
    fn parse_timestamp(value: &str, field_name: &str) -> rusqlite::Result<DateTime<Utc>> {
        DateTime::parse_from_rfc3339(value)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Invalid {} timestamp '{}': {}", field_name, value, e),
                    )),
                )
            })
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
                let results = self.semantic_search(&input.query, input.limit, 0.3)?;
                results.into_iter().map(|r| r.node).collect()
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
            if let Ok(Some(embedding)) = self.get_node_embedding(id) {
                let index = self
                    .vector_index
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

        let embedding_model: Option<String> = if nodes_with_embeddings > 0 {
            Some("nomic-embed-text-v1.5".to_string())
        } else {
            None
        };

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
            embedding_model,
        })
    }

    /// Delete a node
    pub fn delete_node(&self, id: &str) -> Result<bool> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let rows = writer.execute("DELETE FROM knowledge_nodes WHERE id = ?1", params![id])?;

        // Clean up vector index to prevent stale search results
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        if rows > 0
            && let Ok(mut index) = self.vector_index.lock()
        {
            let _ = index.remove(id);
        }

        Ok(rows > 0)
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
    #[cfg(feature = "embeddings")]
    fn get_query_embedding(&self, query: &str) -> Result<Vec<f32>> {
        // Check cache first
        {
            let mut cache = self
                .query_cache
                .lock()
                .map_err(|_| StorageError::Init("Query cache lock poisoned".to_string()))?;
            if let Some(cached) = cache.get(query) {
                return Ok(cached.clone());
            }
        }

        // Not in cache, compute embedding
        let embedding = self
            .embedding_service
            .embed(query)
            .map_err(|e| StorageError::Init(format!("Failed to embed query: {}", e)))?;

        // Store in cache
        {
            let mut cache = self
                .query_cache
                .lock()
                .map_err(|_| StorageError::Init("Query cache lock poisoned".to_string()))?;
            cache.put(query.to_string(), embedding.vector.clone());
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
        if !self.embedding_service.is_ready() {
            return Err(StorageError::Init("Embedding model not ready".to_string()));
        }

        let query_embedding = self.get_query_embedding(query)?;

        let index = self
            .vector_index
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

        let semantic_results = if self.embedding_service.is_ready() {
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

        let index = self
            .vector_index
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

        let nodes: Vec<(String, String)> = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            if let Some(ids) = node_ids {
                let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                let query = format!(
                    "SELECT id, content FROM knowledge_nodes WHERE id IN ({})",
                    placeholders
                );

                let mut result_nodes = Vec::new();
                {
                    let mut stmt = reader.prepare(&query)?;
                    let params: Vec<&dyn rusqlite::ToSql> =
                        ids.iter().map(|s| s as &dyn rusqlite::ToSql).collect();

                    let rows = stmt.query_map(params.as_slice(), |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })?;

                    for r in rows.flatten() {
                        result_nodes.push(r);
                    }
                }
                result_nodes
            } else if force {
                let mut stmt = reader.prepare("SELECT id, content FROM knowledge_nodes")?;
                let rows = stmt.query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?;
                rows.filter_map(|r| r.ok()).collect()
            } else {
                let mut stmt = reader.prepare(
                    "SELECT id, content FROM knowledge_nodes
                         WHERE has_embedding = 0 OR has_embedding IS NULL",
                )?;
                let rows = stmt.query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?;
                rows.filter_map(|r| r.ok()).collect()
            }
        };

        for (id, content) in nodes {
            if !force {
                let has_emb: i32 = self
                    .reader
                    .lock()
                    .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?
                    .query_row(
                        "SELECT COALESCE(has_embedding, 0) FROM knowledge_nodes WHERE id = ?1",
                        params![id],
                        |row| row.get(0),
                    )
                    .unwrap_or(0);

                if has_emb == 1 {
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

        // 3. Generate missing embeddings
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

    /// Generate missing embeddings
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn generate_missing_embeddings(&self) -> Result<i64> {
        if !self.embedding_service.is_ready()
            && let Err(e) = self.embedding_service.init()
        {
            tracing::warn!("Could not initialize embedding model: {}", e);
            return Ok(0);
        }

        let nodes: Vec<(String, String)> = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            reader
                .prepare(
                    "SELECT id, content FROM knowledge_nodes
                     WHERE has_embedding = 0 OR has_embedding IS NULL
                     LIMIT 100",
                )?
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                .filter_map(|r| r.ok())
                .collect()
        };

        let mut count = 0i64;

        for (id, content) in nodes {
            if let Err(e) = self.generate_embedding_for_node(&id, &content) {
                tracing::warn!("Failed to generate embedding for {}: {}", id, e);
            } else {
                count += 1;
            }
        }

        Ok(count)
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

impl Storage {
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

    /// Get last backup timestamp by scanning the backups directory.
    /// Parses `vestige-YYYYMMDD-HHMMSS.db` filenames.
    pub fn get_last_backup_timestamp() -> Option<DateTime<Utc>> {
        let proj_dirs = directories::ProjectDirs::from("com", "vestige", "core")?;
        let backup_dir = proj_dirs.data_dir().parent()?.join("backups");

        if !backup_dir.exists() {
            return None;
        }

        let mut latest: Option<DateTime<Utc>> = None;

        if let Ok(entries) = std::fs::read_dir(&backup_dir) {
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

        // Collect IDs first for vector index cleanup
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
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
        let deleted = writer.execute(
            "DELETE FROM knowledge_nodes WHERE retention_strength < ?1 AND created_at < ?2",
            params![threshold, cutoff],
        )? as i64;
        drop(writer);

        // Clean up vector index
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        if deleted > 0
            && let Ok(mut index) = self.vector_index.lock()
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
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_storage() -> Storage {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        Storage::new(Some(db_path)).unwrap()
    }

    #[test]
    fn test_storage_creation() {
        let storage = create_test_storage();
        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.total_nodes, 0);
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
}
