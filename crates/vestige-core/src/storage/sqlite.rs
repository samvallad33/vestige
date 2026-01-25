//! SQLite Storage Implementation
//!
//! Core storage layer with integrated embeddings and vector search.

use chrono::{DateTime, Duration, Utc};
use directories::ProjectDirs;
use lru::LruCache;
use rusqlite::{params, Connection, OptionalExtension};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Mutex;
use uuid::Uuid;

use crate::fsrs::{FSRSScheduler, FSRSState, LearningState, Rating};
use crate::memory::{
    ConsolidationResult, EmbeddingResult, IngestInput, KnowledgeNode, MatchType, MemoryStats,
    RecallInput, SearchMode, SearchResult, SimilarityResult,
};
use crate::search::sanitize_fts5_query;

#[cfg(feature = "embeddings")]
use crate::embeddings::{Embedding, EmbeddingService, EMBEDDING_DIMENSIONS};

#[cfg(feature = "vector-search")]
use crate::search::{reciprocal_rank_fusion, VectorIndex};

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
pub struct Storage {
    conn: Connection,
    scheduler: FSRSScheduler,
    #[cfg(feature = "embeddings")]
    embedding_service: EmbeddingService,
    #[cfg(feature = "vector-search")]
    vector_index: Mutex<VectorIndex>,
    /// LRU cache for query embeddings to avoid re-embedding repeated queries
    #[cfg(feature = "embeddings")]
    query_cache: Mutex<LruCache<String, Vec<f32>>>,
}

impl Storage {
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
                data_dir.join("vestige.db")
            }
        };

        let conn = Connection::open(&path)?;

        // Configure SQLite for performance
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA cache_size = -64000;
             PRAGMA temp_store = MEMORY;
             PRAGMA foreign_keys = ON;
             PRAGMA busy_timeout = 5000;",
        )?;

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

        let mut storage = Self {
            conn,
            scheduler: FSRSScheduler::default(),
            #[cfg(feature = "embeddings")]
            embedding_service,
            #[cfg(feature = "vector-search")]
            vector_index: Mutex::new(vector_index),
            #[cfg(feature = "embeddings")]
            query_cache,
        };

        storage.init_schema()?;

        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        storage.load_embeddings_into_index()?;

        Ok(storage)
    }

    /// Initialize database schema
    fn init_schema(&mut self) -> Result<()> {
        // Apply migrations
        super::migrations::apply_migrations(&self.conn)?;
        Ok(())
    }

    /// Load existing embeddings into vector index
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn load_embeddings_into_index(&mut self) -> Result<()> {
        let mut stmt = self
            .conn
            .prepare("SELECT node_id, embedding FROM node_embeddings")?;

        let embeddings: Vec<(String, Vec<u8>)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();

        let mut index = self
            .vector_index
            .lock()
            .map_err(|_| StorageError::Init("Vector index lock poisoned".to_string()))?;

        for (node_id, embedding_bytes) in embeddings {
            if let Some(embedding) = Embedding::from_bytes(&embedding_bytes) {
                if let Err(e) = index.add(&node_id, &embedding.vector) {
                    tracing::warn!("Failed to load embedding for {}: {}", node_id, e);
                }
            }
        }

        Ok(())
    }

    /// Ingest a new memory
    pub fn ingest(&mut self, input: IngestInput) -> Result<KnowledgeNode> {
        let now = Utc::now();
        let id = Uuid::new_v4().to_string();

        let fsrs_state = self.scheduler.new_card();

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

        self.conn.execute(
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
    pub fn smart_ingest(
        &mut self,
        input: IngestInput,
    ) -> Result<SmartIngestResult> {
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
        for (node_id, similarity) in similar.iter() {
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
            GateDecision::Create { prediction_error, related_memory_ids, reason, .. } => {
                // Create new memory
                let node = self.ingest(input)?;
                Ok(SmartIngestResult {
                    decision: "create".to_string(),
                    node,
                    superseded_id: None,
                    similarity: None,
                    prediction_error: Some(prediction_error),
                    reason: format!("Created new memory: {:?}. Related: {:?}", reason, related_memory_ids),
                })
            }
            GateDecision::Update { target_id, similarity, update_type, prediction_error } => {
                match update_type {
                    UpdateType::Reinforce => {
                        // Just strengthen the existing memory
                        self.strengthen_on_access(&target_id)?;
                        let node = self.get_node(&target_id)?
                            .ok_or_else(|| StorageError::NotFound(target_id.clone()))?;
                        Ok(SmartIngestResult {
                            decision: "reinforce".to_string(),
                            node,
                            superseded_id: None,
                            similarity: Some(similarity),
                            prediction_error: Some(prediction_error),
                            reason: "Content nearly identical - reinforced existing memory".to_string(),
                        })
                    }
                    UpdateType::Merge | UpdateType::Append => {
                        // Update the existing memory with merged content
                        let existing = self.get_node(&target_id)?
                            .ok_or_else(|| StorageError::NotFound(target_id.clone()))?;

                        let merged_content = format!(
                            "{}\n\n[Updated {}]\n{}",
                            existing.content,
                            chrono::Utc::now().format("%Y-%m-%d"),
                            input.content
                        );

                        self.update_node_content(&target_id, &merged_content)?;
                        self.strengthen_on_access(&target_id)?;

                        let node = self.get_node(&target_id)?
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
                        let node = self.get_node(&target_id)?
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
                        let existing = self.get_node(&target_id)?
                            .ok_or_else(|| StorageError::NotFound(target_id.clone()))?;

                        let merged_content = format!(
                            "{}\n\n---\nContext: {}",
                            existing.content,
                            input.content
                        );

                        self.update_node_content(&target_id, &merged_content)?;
                        let node = self.get_node(&target_id)?
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
            GateDecision::Supersede { old_memory_id, similarity, supersede_reason, prediction_error } => {
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
            GateDecision::Merge { memory_ids, avg_similarity, strategy } => {
                // For now, create new and link to existing
                let node = self.ingest(input)?;

                Ok(SmartIngestResult {
                    decision: "merge".to_string(),
                    node,
                    superseded_id: None,
                    similarity: Some(avg_similarity),
                    prediction_error: Some(1.0 - avg_similarity),
                    reason: format!("Created new memory linked to {} similar memories ({:?})", memory_ids.len(), strategy),
                })
            }
        }
    }

    /// Get the embedding vector for a node
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn get_node_embedding(&self, node_id: &str) -> Result<Option<Vec<f32>>> {
        let mut stmt = self.conn.prepare(
            "SELECT embedding FROM node_embeddings WHERE node_id = ?1"
        )?;

        let embedding_bytes: Option<Vec<u8>> = stmt
            .query_row(params![node_id], |row| row.get(0))
            .optional()?;

        Ok(embedding_bytes.and_then(|bytes| {
            crate::embeddings::Embedding::from_bytes(&bytes).map(|e| e.vector)
        }))
    }

    /// Update the content of an existing node
    pub fn update_node_content(&mut self, id: &str, new_content: &str) -> Result<()> {
        let now = Utc::now();

        self.conn.execute(
            "UPDATE knowledge_nodes SET content = ?1, updated_at = ?2 WHERE id = ?3",
            params![new_content, now.to_rfc3339(), id],
        )?;

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
    fn generate_embedding_for_node(&mut self, node_id: &str, content: &str) -> Result<()> {
        if !self.embedding_service.is_ready() {
            return Ok(());
        }

        let embedding = self
            .embedding_service
            .embed(content)
            .map_err(|e| StorageError::Init(format!("Embedding failed: {}", e)))?;

        let now = Utc::now();

        self.conn.execute(
            "INSERT OR REPLACE INTO node_embeddings (node_id, embedding, dimensions, model, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                node_id,
                embedding.to_bytes(),
                EMBEDDING_DIMENSIONS as i32,
                "all-MiniLM-L6-v2",
                now.to_rfc3339(),
            ],
        )?;

        self.conn.execute(
            "UPDATE knowledge_nodes SET has_embedding = 1, embedding_model = 'all-MiniLM-L6-v2' WHERE id = ?1",
            params![node_id],
        )?;

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
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM knowledge_nodes WHERE id = ?1")?;

        let node = stmt
            .query_row(params![id], |row| self.row_to_node(row))
            .optional()?;
        Ok(node)
    }

    /// Parse RFC3339 timestamp
    fn parse_timestamp(&self, value: &str, field_name: &str) -> rusqlite::Result<DateTime<Utc>> {
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
    fn row_to_node(&self, row: &rusqlite::Row) -> rusqlite::Result<KnowledgeNode> {
        let tags_json: String = row.get("tags")?;
        let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();

        let created_at: String = row.get("created_at")?;
        let updated_at: String = row.get("updated_at")?;
        let last_accessed: String = row.get("last_accessed")?;
        let next_review: Option<String> = row.get("next_review")?;

        let created_at = self.parse_timestamp(&created_at, "created_at")?;
        let updated_at = self.parse_timestamp(&updated_at, "updated_at")?;
        let last_accessed = self.parse_timestamp(&last_accessed, "last_accessed")?;

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
                let results = self.hybrid_search(&input.query, input.limit, 0.5, 0.5)?;
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

        let mut stmt = self.conn.prepare(
            "SELECT n.* FROM knowledge_nodes n
             JOIN knowledge_fts fts ON n.id = fts.id
             WHERE knowledge_fts MATCH ?1
             AND n.retention_strength >= ?2
             ORDER BY n.retention_strength DESC
             LIMIT ?3",
        )?;

        let nodes = stmt.query_map(params![sanitized_query, min_retention, limit], |row| {
            self.row_to_node(row)
        })?;

        let mut result = Vec::new();
        for node in nodes {
            result.push(node?);
        }
        Ok(result)
    }

    /// Mark a memory as reviewed
    pub fn mark_reviewed(&mut self, id: &str, rating: Rating) -> Result<KnowledgeNode> {
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

        let elapsed_days = self.scheduler.days_since_review(&current_state.last_review);

        let sentiment_boost = if node.sentiment_magnitude > 0.0 {
            Some(node.sentiment_magnitude)
        } else {
            None
        };

        let result = self
            .scheduler
            .review(&current_state, rating, elapsed_days, sentiment_boost);

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

        self.conn.execute(
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

        self.get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    /// Passively strengthen a memory when it's accessed (recalled/searched)
    /// This implements the "use it or lose it" principle - memories that are
    /// accessed get a small boost, those that aren't decay naturally.
    /// Based on Testing Effect (Roediger & Karpicke 2006)
    pub fn strengthen_on_access(&self, id: &str) -> Result<()> {
        let now = Utc::now();

        // Small retrieval strength boost (0.05) on each access
        // This is much smaller than a full review but compounds over time
        self.conn.execute(
            "UPDATE knowledge_nodes SET
                last_accessed = ?1,
                retrieval_strength = MIN(1.0, retrieval_strength + 0.05),
                retention_strength = MIN(1.0, retention_strength + 0.02)
            WHERE id = ?2",
            params![now.to_rfc3339(), id],
        )?;

        Ok(())
    }

    /// Batch strengthen multiple memories on access
    pub fn strengthen_batch_on_access(&self, ids: &[&str]) -> Result<()> {
        for id in ids {
            self.strengthen_on_access(id)?;
        }
        Ok(())
    }

    /// Promote a memory (thumbs up) - used when a memory led to a good outcome
    /// Significantly boosts retrieval strength so it surfaces more often
    pub fn promote_memory(&self, id: &str) -> Result<KnowledgeNode> {
        let now = Utc::now();

        // Strong boost: +0.2 retrieval, +0.1 retention
        self.conn.execute(
            "UPDATE knowledge_nodes SET
                last_accessed = ?1,
                retrieval_strength = MIN(1.0, retrieval_strength + 0.20),
                retention_strength = MIN(1.0, retention_strength + 0.10),
                stability = stability * 1.5
            WHERE id = ?2",
            params![now.to_rfc3339(), id],
        )?;

        self.get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    /// Demote a memory (thumbs down) - used when a memory led to a bad outcome
    /// Significantly reduces retrieval strength so better alternatives surface
    /// Does NOT delete - the memory stays for reference but ranks lower
    pub fn demote_memory(&self, id: &str) -> Result<KnowledgeNode> {
        let now = Utc::now();

        // Strong penalty: -0.3 retrieval, -0.15 retention, halve stability
        self.conn.execute(
            "UPDATE knowledge_nodes SET
                last_accessed = ?1,
                retrieval_strength = MAX(0.05, retrieval_strength - 0.30),
                retention_strength = MAX(0.05, retention_strength - 0.15),
                stability = stability * 0.5
            WHERE id = ?2",
            params![now.to_rfc3339(), id],
        )?;

        self.get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    /// Get memories due for review
    pub fn get_review_queue(&self, limit: i32) -> Result<Vec<KnowledgeNode>> {
        let now = Utc::now().to_rfc3339();

        let mut stmt = self.conn.prepare(
            "SELECT * FROM knowledge_nodes
             WHERE next_review <= ?1
             ORDER BY next_review ASC
             LIMIT ?2",
        )?;

        let nodes = stmt.query_map(params![now, limit], |row| self.row_to_node(row))?;

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

        let elapsed_days = self.scheduler.days_since_review(&current_state.last_review);

        Ok(self.scheduler.preview_reviews(&current_state, elapsed_days))
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> Result<MemoryStats> {
        let now = Utc::now().to_rfc3339();

        let total: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM knowledge_nodes", [], |row| row.get(0))?;

        let due: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM knowledge_nodes WHERE next_review <= ?1",
            params![now],
            |row| row.get(0),
        )?;

        let avg_retention: f64 = self.conn.query_row(
            "SELECT COALESCE(AVG(retention_strength), 0) FROM knowledge_nodes",
            [],
            |row| row.get(0),
        )?;

        let avg_storage: f64 = self.conn.query_row(
            "SELECT COALESCE(AVG(storage_strength), 1) FROM knowledge_nodes",
            [],
            |row| row.get(0),
        )?;

        let avg_retrieval: f64 = self.conn.query_row(
            "SELECT COALESCE(AVG(retrieval_strength), 1) FROM knowledge_nodes",
            [],
            |row| row.get(0),
        )?;

        let oldest: Option<String> = self
            .conn
            .query_row("SELECT MIN(created_at) FROM knowledge_nodes", [], |row| {
                row.get(0)
            })
            .ok();

        let newest: Option<String> = self
            .conn
            .query_row("SELECT MAX(created_at) FROM knowledge_nodes", [], |row| {
                row.get(0)
            })
            .ok();

        let nodes_with_embeddings: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM knowledge_nodes WHERE has_embedding = 1",
            [],
            |row| row.get(0),
        )?;

        let embedding_model: Option<String> = if nodes_with_embeddings > 0 {
            Some("all-MiniLM-L6-v2".to_string())
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
    pub fn delete_node(&mut self, id: &str) -> Result<bool> {
        let rows = self
            .conn
            .execute("DELETE FROM knowledge_nodes WHERE id = ?1", params![id])?;
        Ok(rows > 0)
    }

    /// Search with full-text search
    pub fn search(&self, query: &str, limit: i32) -> Result<Vec<KnowledgeNode>> {
        let sanitized_query = sanitize_fts5_query(query);

        let mut stmt = self.conn.prepare(
            "SELECT n.* FROM knowledge_nodes n
             JOIN knowledge_fts fts ON n.id = fts.id
             WHERE knowledge_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;

        let nodes = stmt.query_map(params![sanitized_query, limit], |row| self.row_to_node(row))?;

        let mut result = Vec::new();
        for node in nodes {
            result.push(node?);
        }
        Ok(result)
    }

    /// Get all nodes (paginated)
    pub fn get_all_nodes(&self, limit: i32, offset: i32) -> Result<Vec<KnowledgeNode>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM knowledge_nodes
             ORDER BY created_at DESC
             LIMIT ?1 OFFSET ?2",
        )?;

        let nodes = stmt.query_map(params![limit, offset], |row| self.row_to_node(row))?;

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
        match tag_filter {
            Some(tag) => {
                // Query with tag filter using JSON LIKE search
                // Tags are stored as JSON array, e.g., '["pattern", "codebase", "codebase:vestige"]'
                let tag_pattern = format!("%\"{}%", tag);
                let mut stmt = self.conn.prepare(
                    "SELECT * FROM knowledge_nodes
                     WHERE node_type = ?1
                     AND tags LIKE ?2
                     ORDER BY retention_strength DESC, created_at DESC
                     LIMIT ?3",
                )?;
                let rows = stmt.query_map(params![node_type, tag_pattern, limit], |row| {
                    self.row_to_node(row)
                })?;
                let mut nodes = Vec::new();
                for row in rows {
                    if let Ok(node) = row {
                        nodes.push(node);
                    }
                }
                Ok(nodes)
            }
            None => {
                // Query without tag filter
                let mut stmt = self.conn.prepare(
                    "SELECT * FROM knowledge_nodes
                     WHERE node_type = ?1
                     ORDER BY retention_strength DESC, created_at DESC
                     LIMIT ?2",
                )?;
                let rows = stmt.query_map(params![node_type, limit], |row| self.row_to_node(row))?;
                let mut nodes = Vec::new();
                for row in rows {
                    if let Ok(node) = row {
                        nodes.push(node);
                    }
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

    /// Get query embedding from cache or compute it
    #[cfg(feature = "embeddings")]
    fn get_query_embedding(&self, query: &str) -> Result<Vec<f32>> {
        // Check cache first
        {
            let mut cache = self.query_cache.lock()
                .map_err(|_| StorageError::Init("Query cache lock poisoned".to_string()))?;
            if let Some(cached) = cache.get(query) {
                return Ok(cached.clone());
            }
        }

        // Not in cache, compute embedding
        let embedding = self.embedding_service.embed(query)
            .map_err(|e| StorageError::Init(format!("Failed to embed query: {}", e)))?;

        // Store in cache
        {
            let mut cache = self.query_cache.lock()
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

    /// Hybrid search
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn hybrid_search(
        &self,
        query: &str,
        limit: i32,
        keyword_weight: f32,
        semantic_weight: f32,
    ) -> Result<Vec<SearchResult>> {
        let keyword_results = self.keyword_search_with_scores(query, limit * 2)?;

        let semantic_results = if self.embedding_service.is_ready() {
            self.semantic_search_raw(query, limit * 2)?
        } else {
            vec![]
        };

        let combined = if !semantic_results.is_empty() {
            reciprocal_rank_fusion(&keyword_results, &semantic_results, 60.0)
        } else {
            keyword_results.clone()
        };

        let mut results = Vec::with_capacity(limit as usize);

        for (node_id, combined_score) in combined.into_iter().take(limit as usize) {
            if let Some(node) = self.get_node(&node_id)? {
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

        Ok(results)
    }

    /// Keyword search returning scores
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn keyword_search_with_scores(&self, query: &str, limit: i32) -> Result<Vec<(String, f32)>> {
        let sanitized_query = sanitize_fts5_query(query);

        let mut stmt = self.conn.prepare(
            "SELECT n.id, rank FROM knowledge_nodes n
             JOIN knowledge_fts fts ON n.id = fts.id
             WHERE knowledge_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;

        let results: Vec<(String, f32)> = stmt
            .query_map(params![sanitized_query, limit], |row| {
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

        let query_embedding = self.get_query_embedding(query)?;

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
        &mut self,
        node_ids: Option<&[String]>,
        force: bool,
    ) -> Result<EmbeddingResult> {
        if !self.embedding_service.is_ready() {
            self.embedding_service.init().map_err(|e| {
                StorageError::Init(format!("Failed to init embedding service: {}", e))
            })?;
        }

        let mut result = EmbeddingResult::default();

        let nodes: Vec<(String, String)> = if let Some(ids) = node_ids {
            let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let query = format!(
                "SELECT id, content FROM knowledge_nodes WHERE id IN ({})",
                placeholders
            );

            let mut result_nodes = Vec::new();
            {
                let mut stmt = self.conn.prepare(&query)?;
                let params: Vec<&dyn rusqlite::ToSql> =
                    ids.iter().map(|s| s as &dyn rusqlite::ToSql).collect();

                let rows = stmt.query_map(params.as_slice(), |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?;

                for row in rows {
                    if let Ok(r) = row {
                        result_nodes.push(r);
                    }
                }
            }
            result_nodes
        } else if force {
            let mut stmt = self
                .conn
                .prepare("SELECT id, content FROM knowledge_nodes")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            rows.filter_map(|r| r.ok()).collect()
        } else {
            let mut stmt = self.conn.prepare(
                "SELECT id, content FROM knowledge_nodes
                     WHERE has_embedding = 0 OR has_embedding IS NULL",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            rows.filter_map(|r| r.ok()).collect()
        };

        for (id, content) in nodes {
            if !force {
                let has_emb: i32 = self
                    .conn
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

        let mut stmt = self.conn.prepare(
            "SELECT * FROM knowledge_nodes
             WHERE (valid_from IS NULL OR valid_from <= ?1)
             AND (valid_until IS NULL OR valid_until >= ?1)
             ORDER BY created_at DESC
             LIMIT ?2",
        )?;

        let nodes = stmt.query_map(params![timestamp, limit], |row| self.row_to_node(row))?;

        let mut result = Vec::new();
        for node in nodes {
            result.push(node?);
        }
        Ok(result)
    }

    /// Query memories created/modified in a time range
    pub fn query_time_range(
        &self,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
        limit: i32,
    ) -> Result<Vec<KnowledgeNode>> {
        let start_str = start.map(|dt| dt.to_rfc3339());
        let end_str = end.map(|dt| dt.to_rfc3339());

        let (query, params): (&str, Vec<Box<dyn rusqlite::ToSql>>) = match (&start_str, &end_str) {
            (Some(s), Some(e)) => (
                "SELECT * FROM knowledge_nodes
                 WHERE created_at >= ?1 AND created_at <= ?2
                 ORDER BY created_at DESC
                 LIMIT ?3",
                vec![
                    Box::new(s.clone()) as Box<dyn rusqlite::ToSql>,
                    Box::new(e.clone()) as Box<dyn rusqlite::ToSql>,
                    Box::new(limit) as Box<dyn rusqlite::ToSql>,
                ],
            ),
            (Some(s), None) => (
                "SELECT * FROM knowledge_nodes
                 WHERE created_at >= ?1
                 ORDER BY created_at DESC
                 LIMIT ?2",
                vec![
                    Box::new(s.clone()) as Box<dyn rusqlite::ToSql>,
                    Box::new(limit) as Box<dyn rusqlite::ToSql>,
                ],
            ),
            (None, Some(e)) => (
                "SELECT * FROM knowledge_nodes
                 WHERE created_at <= ?1
                 ORDER BY created_at DESC
                 LIMIT ?2",
                vec![
                    Box::new(e.clone()) as Box<dyn rusqlite::ToSql>,
                    Box::new(limit) as Box<dyn rusqlite::ToSql>,
                ],
            ),
            (None, None) => (
                "SELECT * FROM knowledge_nodes
                 ORDER BY created_at DESC
                 LIMIT ?1",
                vec![Box::new(limit) as Box<dyn rusqlite::ToSql>],
            ),
        };

        let mut stmt = self.conn.prepare(query)?;
        let params_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        let nodes = stmt.query_map(params_refs.as_slice(), |row| self.row_to_node(row))?;

        let mut result = Vec::new();
        for node in nodes {
            result.push(node?);
        }
        Ok(result)
    }

    /// Apply decay to all memories
    pub fn apply_decay(&mut self) -> Result<i32> {
        const FSRS_DECAY: f64 = 0.5;
        const FSRS_FACTOR: f64 = 9.0;

        let now = Utc::now();

        let mut stmt = self.conn.prepare(
            "SELECT id, last_accessed, storage_strength, retrieval_strength,
                    sentiment_magnitude, stability
             FROM knowledge_nodes",
        )?;

        let nodes: Vec<(String, String, f64, f64, f64, f64)> = stmt
            .query_map([], |row| {
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
            .collect();

        let mut count = 0;

        for (id, last_accessed, storage_strength, _, sentiment_mag, stability) in nodes {
            let last = DateTime::parse_from_rfc3339(&last_accessed)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or(now);

            let days_since = (now - last).num_seconds() as f64 / 86400.0;

            if days_since > 0.0 {
                let effective_stability = stability * (1.0 + sentiment_mag * 0.5);

                let new_retrieval = (1.0 + days_since / (FSRS_FACTOR * effective_stability))
                    .powf(-1.0 / FSRS_DECAY);

                let new_retention =
                    (new_retrieval * 0.7) + ((storage_strength / 10.0).min(1.0) * 0.3);

                self.conn.execute(
                    "UPDATE knowledge_nodes SET
                        retrieval_strength = ?1,
                        retention_strength = ?2
                     WHERE id = ?3",
                    params![new_retrieval, new_retention, id],
                )?;

                count += 1;
            }
        }

        Ok(count)
    }

    /// Run consolidation
    pub fn run_consolidation(&mut self) -> Result<ConsolidationResult> {
        let start = std::time::Instant::now();

        let decay_applied = self.apply_decay()? as i64;

        let promoted = self.conn.execute(
            "UPDATE knowledge_nodes SET
                storage_strength = MIN(storage_strength * 1.5, 10.0)
             WHERE sentiment_magnitude > 0.5
             AND storage_strength < 10",
            [],
        )? as i64;

        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        let embeddings_generated = self.generate_missing_embeddings()? as i64;
        #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
        let embeddings_generated = 0i64;

        let duration = start.elapsed().as_millis() as i64;

        Ok(ConsolidationResult {
            nodes_processed: decay_applied,
            nodes_promoted: promoted,
            nodes_pruned: 0,
            decay_applied,
            duration_ms: duration,
            embeddings_generated,
        })
    }

    /// Generate missing embeddings
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn generate_missing_embeddings(&mut self) -> Result<i64> {
        if !self.embedding_service.is_ready() {
            if let Err(e) = self.embedding_service.init() {
                tracing::warn!("Could not initialize embedding model: {}", e);
                return Ok(0);
            }
        }

        let nodes: Vec<(String, String)> = self
            .conn
            .prepare(
                "SELECT id, content FROM knowledge_nodes
                 WHERE has_embedding = 0 OR has_embedding IS NULL
                 LIMIT 100",
            )?
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();

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
    pub trigger_data: String,  // JSON
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
    pub state: String,  // 'active', 'dormant', 'silent', 'unavailable'
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

impl Storage {
    // ========================================================================
    // INTENTIONS PERSISTENCE
    // ========================================================================

    /// Save an intention to the database
    pub fn save_intention(&mut self, intention: &IntentionRecord) -> Result<()> {
        let tags_json = serde_json::to_string(&intention.tags).unwrap_or_else(|_| "[]".to_string());
        let related_json = serde_json::to_string(&intention.related_memories).unwrap_or_else(|_| "[]".to_string());

        self.conn.execute(
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
        let mut stmt = self.conn.prepare(
            "SELECT * FROM intentions WHERE id = ?1"
        )?;

        stmt.query_row(params![id], |row| self.row_to_intention(row))
            .optional()
            .map_err(StorageError::from)
    }

    /// Get all active intentions
    pub fn get_active_intentions(&self) -> Result<Vec<IntentionRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM intentions WHERE status = 'active' ORDER BY priority DESC, created_at ASC"
        )?;

        let rows = stmt.query_map([], |row| self.row_to_intention(row))?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Get intentions by status
    pub fn get_intentions_by_status(&self, status: &str) -> Result<Vec<IntentionRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM intentions WHERE status = ?1 ORDER BY priority DESC, created_at ASC"
        )?;

        let rows = stmt.query_map(params![status], |row| self.row_to_intention(row))?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Update intention status
    pub fn update_intention_status(&mut self, id: &str, status: &str) -> Result<bool> {
        let now = Utc::now();
        let fulfilled_at = if status == "fulfilled" { Some(now.to_rfc3339()) } else { None };

        let rows = self.conn.execute(
            "UPDATE intentions SET status = ?1, fulfilled_at = ?2 WHERE id = ?3",
            params![status, fulfilled_at, id],
        )?;
        Ok(rows > 0)
    }

    /// Delete an intention
    pub fn delete_intention(&mut self, id: &str) -> Result<bool> {
        let rows = self.conn.execute("DELETE FROM intentions WHERE id = ?1", params![id])?;
        Ok(rows > 0)
    }

    /// Get overdue intentions
    pub fn get_overdue_intentions(&self) -> Result<Vec<IntentionRecord>> {
        let now = Utc::now().to_rfc3339();
        let mut stmt = self.conn.prepare(
            "SELECT * FROM intentions WHERE status = 'active' AND deadline IS NOT NULL AND deadline < ?1 ORDER BY deadline ASC"
        )?;

        let rows = stmt.query_map(params![now], |row| self.row_to_intention(row))?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Snooze an intention
    pub fn snooze_intention(&mut self, id: &str, until: DateTime<Utc>) -> Result<bool> {
        let rows = self.conn.execute(
            "UPDATE intentions SET status = 'snoozed', snoozed_until = ?1 WHERE id = ?2",
            params![until.to_rfc3339(), id],
        )?;
        Ok(rows > 0)
    }

    fn row_to_intention(&self, row: &rusqlite::Row) -> rusqlite::Result<IntentionRecord> {
        let tags_json: String = row.get("tags")?;
        let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
        let related_json: String = row.get("related_memories")?;
        let related: Vec<String> = serde_json::from_str(&related_json).unwrap_or_default();

        let parse_opt_dt = |s: Option<String>| -> Option<DateTime<Utc>> {
            s.and_then(|v| DateTime::parse_from_rfc3339(&v).ok().map(|dt| dt.with_timezone(&Utc)))
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
    pub fn save_insight(&mut self, insight: &InsightRecord) -> Result<()> {
        let source_json = serde_json::to_string(&insight.source_memories).unwrap_or_else(|_| "[]".to_string());
        let tags_json = serde_json::to_string(&insight.tags).unwrap_or_else(|_| "[]".to_string());

        self.conn.execute(
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
        let mut stmt = self.conn.prepare(
            "SELECT * FROM insights ORDER BY generated_at DESC LIMIT ?1"
        )?;

        let rows = stmt.query_map(params![limit], |row| self.row_to_insight(row))?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Get insights without feedback (pending review)
    pub fn get_pending_insights(&self) -> Result<Vec<InsightRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM insights WHERE feedback IS NULL ORDER BY novelty_score DESC"
        )?;

        let rows = stmt.query_map([], |row| self.row_to_insight(row))?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Mark insight feedback
    pub fn mark_insight_feedback(&mut self, id: &str, feedback: &str) -> Result<bool> {
        let rows = self.conn.execute(
            "UPDATE insights SET feedback = ?1 WHERE id = ?2",
            params![feedback, id],
        )?;
        Ok(rows > 0)
    }

    /// Clear all insights
    pub fn clear_insights(&mut self) -> Result<i32> {
        let count: i32 = self.conn.query_row("SELECT COUNT(*) FROM insights", [], |row| row.get(0))?;
        self.conn.execute("DELETE FROM insights", [])?;
        Ok(count)
    }

    fn row_to_insight(&self, row: &rusqlite::Row) -> rusqlite::Result<InsightRecord> {
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
    pub fn save_connection(&mut self, connection: &ConnectionRecord) -> Result<()> {
        self.conn.execute(
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
        let mut stmt = self.conn.prepare(
            "SELECT * FROM memory_connections WHERE source_id = ?1 OR target_id = ?1 ORDER BY strength DESC"
        )?;

        let rows = stmt.query_map(params![memory_id], |row| self.row_to_connection(row))?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Get all connections (for building activation network)
    pub fn get_all_connections(&self) -> Result<Vec<ConnectionRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM memory_connections ORDER BY strength DESC"
        )?;

        let rows = stmt.query_map([], |row| self.row_to_connection(row))?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Strengthen a connection
    pub fn strengthen_connection(&mut self, source_id: &str, target_id: &str, boost: f64) -> Result<bool> {
        let now = Utc::now().to_rfc3339();
        let rows = self.conn.execute(
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
    pub fn apply_connection_decay(&mut self, decay_factor: f64) -> Result<i32> {
        let rows = self.conn.execute(
            "UPDATE memory_connections SET strength = strength * ?1",
            params![decay_factor],
        )?;
        Ok(rows as i32)
    }

    /// Prune weak connections below threshold
    pub fn prune_weak_connections(&mut self, min_strength: f64) -> Result<i32> {
        let rows = self.conn.execute(
            "DELETE FROM memory_connections WHERE strength < ?1",
            params![min_strength],
        )?;
        Ok(rows as i32)
    }

    fn row_to_connection(&self, row: &rusqlite::Row) -> rusqlite::Result<ConnectionRecord> {
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
    pub fn save_memory_state(&mut self, state: &MemoryStateRecord) -> Result<()> {
        let suppressed_json = serde_json::to_string(&state.suppressed_by).unwrap_or_else(|_| "[]".to_string());

        self.conn.execute(
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
        let mut stmt = self.conn.prepare(
            "SELECT * FROM memory_states WHERE memory_id = ?1"
        )?;

        stmt.query_row(params![memory_id], |row| self.row_to_memory_state(row))
            .optional()
            .map_err(StorageError::from)
    }

    /// Get memories by state
    pub fn get_memories_by_state(&self, state: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT memory_id FROM memory_states WHERE state = ?1"
        )?;

        let rows = stmt.query_map(params![state], |row| row.get::<_, String>(0))?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Update memory state
    pub fn update_memory_state(&mut self, memory_id: &str, new_state: &str, reason: &str) -> Result<bool> {
        let now = Utc::now();

        // Get old state for transition record
        if let Some(old_record) = self.get_memory_state(memory_id)? {
            // Record state transition
            self.conn.execute(
                "INSERT INTO state_transitions (memory_id, from_state, to_state, reason_type, timestamp)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![memory_id, old_record.state, new_state, reason, now.to_rfc3339()],
            )?;
        }

        let rows = self.conn.execute(
            "UPDATE memory_states SET state = ?1, state_entered_at = ?2 WHERE memory_id = ?3",
            params![new_state, now.to_rfc3339(), memory_id],
        )?;
        Ok(rows > 0)
    }

    /// Record access to memory (updates state)
    pub fn record_memory_access(&mut self, memory_id: &str) -> Result<()> {
        let now = Utc::now();

        // Check if state exists
        let exists: bool = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM memory_states WHERE memory_id = ?1)",
            params![memory_id],
            |row| row.get(0),
        )?;

        if exists {
            self.conn.execute(
                "UPDATE memory_states SET
                    last_access = ?1,
                    access_count = access_count + 1,
                    state = 'active',
                    state_entered_at = CASE WHEN state != 'active' THEN ?1 ELSE state_entered_at END
                 WHERE memory_id = ?2",
                params![now.to_rfc3339(), memory_id],
            )?;
        } else {
            self.conn.execute(
                "INSERT INTO memory_states (memory_id, state, last_access, access_count, state_entered_at)
                 VALUES (?1, 'active', ?2, 1, ?2)",
                params![memory_id, now.to_rfc3339()],
            )?;
        }
        Ok(())
    }

    fn row_to_memory_state(&self, row: &rusqlite::Row) -> rusqlite::Result<MemoryStateRecord> {
        let suppressed_json: String = row.get("suppressed_by")?;
        let suppressed_by: Vec<String> = serde_json::from_str(&suppressed_json).unwrap_or_default();

        let parse_opt_dt = |s: Option<String>| -> Option<DateTime<Utc>> {
            s.and_then(|v| DateTime::parse_from_rfc3339(&v).ok().map(|dt| dt.with_timezone(&Utc)))
        };

        Ok(MemoryStateRecord {
            memory_id: row.get("memory_id")?,
            state: row.get("state")?,
            last_access: DateTime::parse_from_rfc3339(&row.get::<_, String>("last_access")?)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            access_count: row.get("access_count").unwrap_or(1),
            state_entered_at: DateTime::parse_from_rfc3339(&row.get::<_, String>("state_entered_at")?)
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
    pub fn save_consolidation_history(&mut self, record: &ConsolidationHistoryRecord) -> Result<i64> {
        self.conn.execute(
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
        Ok(self.conn.last_insert_rowid())
    }

    /// Get last consolidation timestamp
    pub fn get_last_consolidation(&self) -> Result<Option<DateTime<Utc>>> {
        let result: Option<String> = self.conn.query_row(
            "SELECT MAX(completed_at) FROM consolidation_history",
            [],
            |row| row.get(0),
        ).ok().flatten();

        Ok(result.and_then(|s| {
            DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))
        }))
    }

    /// Get consolidation history
    pub fn get_consolidation_history(&self, limit: i32) -> Result<Vec<ConsolidationHistoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM consolidation_history ORDER BY completed_at DESC LIMIT ?1"
        )?;

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
    // STATE TRANSITIONS (Audit Trail)
    // ========================================================================

    /// Get state transitions for a memory
    pub fn get_state_transitions(&self, memory_id: &str, limit: i32) -> Result<Vec<StateTransitionRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM state_transitions WHERE memory_id = ?1 ORDER BY timestamp DESC LIMIT ?2"
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
        let mut storage = create_test_storage();

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
        let mut storage = create_test_storage();

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
        let mut storage = create_test_storage();

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
        let mut storage = create_test_storage();

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
}
