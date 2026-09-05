//! Read paths: node lookup, keyword, concrete, semantic and hybrid search,
//! recall, and temporal queries.

use super::*;

impl SqliteMemoryStore {
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
    pub(super) fn parse_timestamp(
        value: &str,
        field_name: &str,
    ) -> rusqlite::Result<DateTime<Utc>> {
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
    pub(super) fn row_to_node(row: &rusqlite::Row) -> rusqlite::Result<KnowledgeNode> {
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
    pub(super) fn keyword_search(
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
}
