//! The `MemoryStoreSend` trait implementation and the record-conversion
//! helpers it relies on.

use super::*;

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
