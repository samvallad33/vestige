//! Connector sync (#57): idempotent external-source upserts, sync cursors
//! and tombstone reconciliation.

use super::*;

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
            rows.filter_map(warn_skipped_row("reconcile_source_tombstones"))
                .collect()
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
