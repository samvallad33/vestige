//! Ingest storage operations, extracted from the integrated v3 implementation.

use super::*;

impl SqliteMemoryStore {
    pub(super) fn secret_findings_for_input(input: &IngestInput) -> Vec<SecretFinding> {
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

    pub(super) fn enforce_secret_policy_for_input(
        input: &IngestInput,
        policy: SecretPolicy,
    ) -> Result<()> {
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

    pub(super) fn enforce_secret_policy_for_content(
        content: &str,
        policy: SecretPolicy,
    ) -> Result<()> {
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
    pub(super) fn normalize_scope(scope: &str) -> Result<&str> {
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

    pub(super) fn enforce_secret_policy_for_record(
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

    pub(super) fn enforce_secret_policy_for_portable_archive(
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
    pub(super) fn ingest_unchecked_in_scope(
        &self,
        input: IngestInput,
        scope: &str,
    ) -> Result<KnowledgeNode> {
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

        // #252 Phase A: stamp the process actor as the author. A store with
        // no bound actor leaves the column NULL, which reads as unattributed
        // at 1.0 — legacy nodes default neutral with no backfill.
        let author_actor_did = self.process_actor_did();

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
                    content_hash, synced_at, source_project, source_type, source_author,
                    author_actor_did
                ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6,
                    ?7, ?8, ?9, ?10, ?11,
                    ?12, ?13, ?14,
                    ?15, ?16, ?17, ?18,
                    ?19, ?20, ?21, ?22, ?23, ?24,
                    '[]', '{}',
                    ?25, ?26, ?27, ?28, ?29,
                    ?30, ?31, ?32, ?33, ?34,
                    ?35
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
                    author_actor_did,
                ],
            )?;
        }

        // Store an embedding vector if the (soon-to-be-deleted) embeddings
        // subsystem is built in. This is vector STORAGE, not a similarity
        // decision — smart ingest's gating no longer reads it. Removed
        // together with the embeddings module.

        self.get_node(&id)?.ok_or(StorageError::NotFound(id))
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

    pub(super) fn update_node_content_unchecked(&self, id: &str, new_content: &str) -> Result<()> {
        let now = Utc::now();

        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            let tx = Self::begin_write_transaction(&writer, "update_node_content")?;
            tx.execute(
                "UPDATE knowledge_nodes SET content = ?1, updated_at = ?2, has_embedding = 0 WHERE id = ?3",
                params![new_content, now.to_rfc3339(), id],
            )?;
            // Deleting profile vectors journals invalidation for peer indexes.
            // Every profile encodes the old content, so invalidate all of them.
            // (Vector cache hygiene for the still-present embeddings subsystem;
            // goes away with that module. No similarity decision happens here.)
            tx.commit()?;
        }

        // Regenerate the vector for the edited content (storage only).

        Ok(())
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
    pub(super) fn apply_tag_mutation_with_limits(
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

    pub(super) fn validate_tag_mutation(
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

    pub(super) fn validate_tag_mutation_reason(reason: &str) -> Result<String> {
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

    pub(super) fn tag_mutation_state(
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

    pub(super) fn tag_mutation_token(
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

    pub(super) fn rewrite_tags(
        tags: &[String],
        source_tags: &[String],
        target_tag: &str,
    ) -> Vec<String> {
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
}
