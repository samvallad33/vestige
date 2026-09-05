//! Persistence records and their queries: intentions, insights, connections,
//! memory states, consolidation and dream history, state transitions, and
//! ComposedGraph events.

use super::*;

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

    pub(super) fn composition_event_exists(conn: &Connection, id: &str) -> Result<bool> {
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

    pub(super) fn pair_key(a: &str, b: &str) -> (String, String) {
        if a <= b {
            (a.to_string(), b.to_string())
        } else {
            (b.to_string(), a.to_string())
        }
    }

    pub(super) fn shared_tags(a: &[String], b: &[String]) -> Vec<String> {
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

    pub(super) fn composition_question(
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

    pub(super) fn outcome_signal(prior_outcomes: &[String]) -> String {
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

    pub(super) fn outcome_score_adjustment(prior_outcomes: &[String]) -> f64 {
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
