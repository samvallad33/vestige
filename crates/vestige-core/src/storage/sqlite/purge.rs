//! Deletion paths: delete, purge with content-free tombstones and embedding
//! cleanup, and retention garbage collection.

use super::*;

impl SqliteMemoryStore {
    /// Delete a node through the same privacy cleanup coordinator as an explicit
    /// purge.  Keeping one deletion path prevents maintenance, dashboard, and
    /// library callers from bypassing replay invalidation or durable-evidence
    /// redaction.
    pub fn delete_node(&self, id: &str) -> Result<bool> {
        Ok(self.purge_node(id, None)?.deleted)
    }

    /// Permanently purge a memory's content and embeddings.
    ///
    /// This is the one local deletion coordinator. It scrubs non-FK references,
    /// invalidates replay evidence, detaches temporal-summary children, and
    /// writes an opaque deletion marker for audit/sync. It remains a legacy
    /// cleanup path and deliberately does not claim verified local unlearning.
    pub fn purge_node(&self, id: &str, reason: Option<&str>) -> Result<PurgeReport> {
        // The reason is logged, never persisted: deletion_tombstones are
        // content-free by contract (an opaque marker, no reason, no tags), so
        // a purged memory leaves nothing recoverable. The local log line is
        // how an operator answers "what deleted this?" without the tombstone
        // ever carrying it.
        if let Some(reason) = reason {
            tracing::info!(memory_id = %id, reason, "purging memory");
        }
        let deleted_at = Utc::now();
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "purge_node")?;
        let cleanup = Self::purge_node_in_transaction(&tx, id, deleted_at, true)?;
        tx.commit()?;
        // Release the writer BEFORE taking the vector-index lock. Every other
        // combined site in this file orders writer -> drop -> index, and
        // `activate_embedding_profile` orders index -> writer, so holding the
        // writer here while waiting on the index is an AB/BA deadlock between a
        // purge and a concurrent profile activation.
        drop(writer);

        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        if cleanup.is_some()
            && let Some(index) = self.vector_index.as_ref()
            && let Ok(mut index) = index.lock()
        {
            let _ = index.remove(id);
        }

        let Some(cleanup) = cleanup else {
            return Ok(PurgeReport {
                memory_id: id.to_string(),
                deleted: false,
                deleted_at,
                edges_pruned: 0,
                insights_rewritten: 0,
                insights_deleted: 0,
                children_orphaned: 0,
                unlearning_scope: crate::storage::UnlearningScope::LegacyAuditedPurge,
                unlearning_verdict: crate::storage::UnlearningVerdict::Incomplete,
                unlearning_claim_boundary: "No purge ran because the requested memory was not found; no unlearning audit or verified-local erasure claim was produced.",
            });
        };

        Ok(PurgeReport {
            memory_id: id.to_string(),
            deleted: true,
            deleted_at,
            edges_pruned: cleanup.edges_pruned,
            insights_rewritten: cleanup.insights_rewritten,
            insights_deleted: cleanup.insights_deleted,
            children_orphaned: cleanup.children_orphaned,
            unlearning_scope: crate::storage::UnlearningScope::LegacyAuditedPurge,
            unlearning_verdict: crate::storage::UnlearningVerdict::Incomplete,
            unlearning_claim_boundary: "Legacy cleanup completed, but this operation has no V25 lineage-completeness proof, full required-surface audit, or anti-resurrection ingress gate. It does not establish complete machine unlearning, erasure of unmanaged copies, media forensics, provider backups, external model weights, or re-ingest prevention.",
        })
    }

    /// Remove a committed purge from the optional in-process vector index.
    pub(crate) fn remove_purged_node_from_vector_index(&self, id: &str) {
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        if let Some(index) = self.vector_index.as_ref()
            && let Ok(mut index) = index.lock()
        {
            let _ = index.remove(id);
        }
        #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
        let _ = id;
    }

    /// Execute the privacy-critical delete work inside a caller-owned SQLite
    /// transaction. Portable merge uses this exact path so a remote deletion
    /// cannot leave local non-FK evidence behind.
    pub(crate) fn purge_node_in_transaction(
        tx: &rusqlite::Transaction<'_>,
        id: &str,
        deleted_at: DateTime<Utc>,
        write_tombstones: bool,
    ) -> Result<Option<PurgeCleanup>> {
        let node = tx
            .prepare("SELECT * FROM knowledge_nodes WHERE id = ?1")?
            .query_row(params![id], Self::row_to_node)
            .optional()?;

        let Some(node) = node else {
            return Ok(None);
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
            // Purge fails closed. A row this scrub cannot read is a row that
            // still references the memory the caller asked us to erase, and
            // reporting a successful purge over it is the one lie this store
            // must never tell. Propagating rolls the whole purge back.
            stmt.query_map(params![pattern], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?
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

        // Review records are intentionally not FK-linked to memories, so a
        // normal node delete would retain their subject id, previews, tags, and
        // potentially user-provided rationale. An erasure request takes privacy
        // precedence over that historical review record.
        tx.execute(
            r#"DELETE FROM memory_prs
                WHERE subject_id = ?1
                   OR (?2 <> '' AND instr(title, ?2) > 0)
                   OR instr(diff, ?1) > 0 OR (?2 <> '' AND instr(diff, ?2) > 0)
                   OR instr(signals, ?1) > 0 OR (?2 <> '' AND instr(signals, ?2) > 0)"#,
            params![id, &node.content],
        )?;

        // Composition members intentionally preserve historical memory ids.
        // Once a user requests erasure, retaining the surrounding event can
        // still expose the memory through query/output/metadata fields. Delete
        // the whole affected event (and FK-cascaded members/outcomes) rather
        // than attempting partial JSON surgery.
        tx.execute(
            r#"DELETE FROM composition_events
                WHERE id IN (
                    SELECT event_id FROM composition_members WHERE memory_id = ?1
                )
                   OR (?2 <> '' AND instr(COALESCE(query, ''), ?2) > 0)
                   OR (?2 <> '' AND instr(COALESCE(output_preview, ''), ?2) > 0)
                   OR instr(metadata, ?1) > 0
                   OR (?2 <> '' AND instr(metadata, ?2) > 0)"#,
            params![id, &node.content],
        )?;

        // A purge must erase frozen replay dependency locators and invalidate
        // every derived replay in the same transaction as the memory removal.
        // This also upgrades a previously redacted capsule to `purged`.
        Self::invalidate_replay_evidence_for_memory_in_transaction(
            tx,
            id,
            crate::storage::ReplayInvalidationReason::Purged,
        )?;

        tx.execute(
            "UPDATE composition_members SET preview = NULL WHERE memory_id = ?1",
            params![id],
        )?;

        // Purge overrides historical receipt fidelity: remove the stable id
        // from every persisted receipt payload while retaining its evidence
        // slots, score, disposition, and measured deltas. Public reads also
        // resolve current state, but this closes the raw V21 audit-row copy.
        let receipt_refs: Vec<(String, String)> = {
            let mut stmt = tx
                .prepare("SELECT receipt_id, payload FROM memory_receipts WHERE payload LIKE ?1")?;
            let pattern = format!("%{}%", id);
            // Purge fails closed. A row this scrub cannot read is a row that
            // still references the memory the caller asked us to erase, and
            // reporting a successful purge over it is the one lie this store
            // must never tell. Propagating rolls the whole purge back.
            stmt.query_map(params![pattern], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        for (receipt_id, payload) in receipt_refs {
            let Ok(mut receipt) = serde_json::from_str::<crate::trace::Receipt>(&payload) else {
                // Not structurally redactable. The raw-text sweeps below still
                // strip the content; say so, because the id may survive here.
                tracing::warn!(
                    receipt_id = %receipt_id,
                    memory_id = %id,
                    "purge could not parse a receipt payload for structured redaction; the raw-text sweep still applies"
                );
                continue;
            };
            receipt.redact_memory_id(id, "purged_1");
            let rewritten = serde_json::to_string(&receipt)
                .map_err(|e| StorageError::Init(format!("receipt redact serialize: {e}")))?;
            tx.execute(
                "UPDATE memory_receipts SET payload = ?1 WHERE receipt_id = ?2",
                params![rewritten, receipt_id],
            )?;
        }
        tx.execute(
            "UPDATE memory_receipts SET query = NULL
             WHERE instr(COALESCE(query, ''), ?1) > 0
                OR (?2 <> '' AND instr(COALESCE(query, ''), ?2) > 0)",
            params![id, &node.content],
        )?;

        // Black Box traces are public/exportable evidence too. Rewrite every
        // id-bearing payload and delete any trace containing the target text;
        // a structured redactor cannot safely prove removal of arbitrary text
        // from historical trace JSON.
        let trace_refs: Vec<(String, String)> = {
            let mut stmt =
                tx.prepare("SELECT id, payload FROM agent_traces WHERE payload LIKE ?1")?;
            let pattern = format!("%{}%", id);
            // Purge fails closed. A row this scrub cannot read is a row that
            // still references the memory the caller asked us to erase, and
            // reporting a successful purge over it is the one lie this store
            // must never tell. Propagating rolls the whole purge back.
            stmt.query_map(params![pattern], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        for (trace_id, payload) in trace_refs {
            let Ok(mut event) = serde_json::from_str::<crate::trace::MemoryTraceEvent>(&payload)
            else {
                // Same as receipts: unparseable payloads fall through to the
                // raw-text delete below, but the operator should see it.
                tracing::warn!(
                    trace_id = %trace_id,
                    memory_id = %id,
                    "purge could not parse a trace payload for structured redaction; the raw-text sweep still applies"
                );
                continue;
            };
            event.redact_memory_id(id, "purged_1");
            let rewritten = serde_json::to_string(&event)
                .map_err(|e| StorageError::Init(format!("trace redact serialize: {e}")))?;
            tx.execute(
                "UPDATE agent_traces SET payload = ?1 WHERE id = ?2",
                params![rewritten, trace_id],
            )?;
        }
        tx.execute(
            "DELETE FROM agent_traces WHERE ?1 <> '' AND instr(payload, ?1) > 0",
            params![&node.content],
        )?;

        // A trigger event otherwise preserves the purged stable id outside the
        // knowledge-node FK graph. Capture-item rows cascade with the event;
        // candidate rows cascade through their synaptic tag on node deletion.
        //
        // A captured tag is only valid while it is bound to the capture item
        // and event which prove that state.  Purging the trigger deletes that
        // proof, so retaining `captured` would leave an invalid durable state
        // that prevents a later startup integrity check from succeeding.  An
        // expired tag cannot be recaptured, which preserves the one-promotion
        // lifecycle without claiming evidence that no longer exists.
        tx.execute(
            "UPDATE synaptic_tags
             SET state = 'expired', capture_event_id = NULL, captured_at_ms = NULL
             WHERE capture_event_id IN (
                 SELECT event_id FROM synaptic_events WHERE trigger_memory_id = ?1
             )",
            params![id],
        )?;
        tx.execute(
            "DELETE FROM synaptic_events WHERE trigger_memory_id = ?1",
            params![id],
        )?;

        // V24 deliberately keeps the immutable, identity-free DSSE envelope
        // after erasure, but its private disclosure mapping is deletable. The
        // FK also covers this when the node delete succeeds; doing it
        // explicitly keeps the privacy operation visible and makes a schema
        // regression fail before the canonical row is removed.
        if Self::table_exists(tx, "receipt_disclosures")? {
            tx.execute(
                "DELETE FROM receipt_disclosures WHERE memory_id = ?1",
                params![id],
            )?;
        }

        if write_tombstones {
            // The V13 table predates commitment-only V25 evidence, but it can
            // still be made content-free without a migration: use an opaque
            // stable marker as its primary key, store no caller reason, and
            // retain no tags. `sync_tombstones` uses the same marker and
            // resolves it locally during merge, so portable deletion
            // propagation remains functional.
            let tombstone_marker = Self::opaque_tombstone_marker(id);
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
                    tombstone_marker,
                    deleted_at.to_rfc3339(),
                    Option::<&str>::None,
                    node.node_type,
                    "[]",
                    edges_pruned,
                    insights_rewritten,
                    insights_deleted,
                    children_orphaned,
                ],
            )?;
            Self::record_sync_tombstone(tx, "knowledge_nodes", id)?;
        }
        tx.execute("DELETE FROM knowledge_nodes WHERE id = ?1", params![id])?;

        Ok(Some(PurgeCleanup {
            edges_pruned,
            insights_rewritten,
            insights_deleted,
            children_orphaned,
        }))
    }

    pub(super) fn node_exists(conn: &Connection, id: &str) -> Result<bool> {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM knowledge_nodes WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    fn record_sync_tombstone(conn: &Connection, table_name: &str, row_id: &str) -> Result<()> {
        let tombstone_row_id = if table_name == "knowledge_nodes" {
            Self::opaque_tombstone_marker(row_id)
        } else {
            row_id.to_string()
        };
        conn.execute(
            "INSERT INTO sync_tombstones (table_name, row_id, deleted_at, reason)
             VALUES (?1, ?2, ?3, NULL)
             ON CONFLICT(table_name, row_id) DO UPDATE SET
                deleted_at = excluded.deleted_at,
                reason = excluded.reason",
            params![table_name, tombstone_row_id, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    /// Deterministic, domain-separated marker for legacy deletion/sync rows.
    /// Knowledge-node UUIDs are not content, but persisting them makes deletion
    /// history linkable to a removed record and exposes them in portable
    /// archives. The marker is enough to match an already-local UUID during
    /// merge without retaining the UUID itself.
    pub(super) fn opaque_tombstone_marker(memory_id: &str) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vestige.legacy-tombstone-marker.v1\\0");
        hasher.update(memory_id.as_bytes());
        format!("opaque:{}", hasher.finalize().to_hex())
    }

    pub(super) fn resolve_tombstone_memory_id(
        tx: &rusqlite::Transaction<'_>,
        tombstone_row_id: &str,
    ) -> Result<Option<String>> {
        // Older archives retain raw ids. Keep their import behavior intact
        // while ensuring all newly produced tombstones are opaque.
        if !tombstone_row_id.starts_with("opaque:") {
            return Ok(Some(tombstone_row_id.to_string()));
        }
        let mut statement = tx.prepare("SELECT id FROM knowledge_nodes")?;
        let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
        for row in rows {
            let id = row?;
            if Self::opaque_tombstone_marker(&id) == tombstone_row_id {
                return Ok(Some(id));
            }
        }
        Ok(None)
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

        // Explicitly protected (pinned) memories are never garbage-collected,
        // no matter how far their retention has decayed. A pin is the user
        // saying "keep this"; low retention only says "rarely retrieved", and
        // the second must never override the first. (Until v2.6.0 this query
        // had no such exemption.)
        let protected = self.protected_node_ids()?;

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
                .filter_map(warn_skipped_row("gc_below_retention"))
                .filter(|id: &String| !protected.contains(id))
                .collect()
        };

        // Do not bulk-delete here. Every deletion must traverse `purge_node`
        // so replay capsules, traces, review records, composition evidence,
        // disclosures, and vector state cannot outlive the canonical node.
        let mut deleted = 0_i64;
        for id in doomed_ids {
            if self.delete_node(&id)? {
                deleted += 1;
            }
        }
        Ok(deleted)
    }
}
