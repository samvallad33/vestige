//! Merge storage operations, extracted from the integrated v3 implementation.

use super::*;

impl SqliteMemoryStore {
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

    /// Surface duplicate/overlapping memory clusters with confidence
    /// scores and the signals behind each (Fellegi-Sunter classified).
    ///
    /// NOMINATION IS EXACT EQUALITY ONLY (owner decision 2026-09-28: no
    /// similarity anywhere in dedup/merge nomination). The former O(n²)
    /// embedding-cosine candidate scan is deleted. A pair of memories is
    /// nominated when any of these exact equalities holds:
    ///
    /// 1. **identical content hash** — the stored envelope `content_hash`
    ///    (SQL group-by on `COALESCE(content_hash, content)`; nodes without a
    ///    recorded hash use their byte-identical content as the identity);
    /// 2. **the same declared source key** `(source_system, source_id)` (SQL
    ///    group-by): the same upstream record ingested twice. Current schemas
    ///    enforce a UNIQUE index on the key, so this nominator mainly
    ///    catches stores written before that constraint existed;
    /// 3. **exactly equal non-empty entity sets** —
    ///    `advanced::retroactive_backfill::extract_entities`, compared as sets
    ///    in memory.
    ///
    /// Tag/token overlap (`advanced::score_pair`) NEVER nominates; it only
    /// orders the nominated clusters and labels them for review. A cluster
    /// nominated by a shared source key but with diverged contents is
    /// intentionally still surfaced (labelled `Possible`/`NonMatch`) instead
    /// of dropped — a repeated declared source is review-worthy on its own,
    /// and the label tells the reviewer how weak the lexical evidence is.
    ///
    /// Protected members are flagged so the caller never auto-merges a pin.
    pub fn merge_candidates(
        &self,
        policy: crate::advanced::MergePolicy,
        limit: usize,
        tag_filter: &[String],
    ) -> Result<Vec<crate::advanced::MergeCandidate>> {
        use crate::advanced::{MergeCandidate, score_pair};
        use std::collections::{BTreeSet, HashMap, HashSet};

        let superseded: HashSet<String> = self.superseded_node_ids()?;
        let protected: HashSet<String> = self.protected_node_ids()?;

        // Load nodes for metadata. Exclude already-superseded nodes — they are
        // historical and must not be re-offered for merge — and apply the
        // caller's tag filter.
        let mut nodes: Vec<KnowledgeNode> = Vec::new();
        let mut offset = 0;
        loop {
            let batch = self.get_all_nodes(500, offset)?;
            let n = batch.len();
            nodes.extend(batch);
            if n < 500 {
                break;
            }
            offset += 500;
        }
        let nodes: Vec<KnowledgeNode> = nodes
            .into_iter()
            .filter(|node| !superseded.contains(&node.id))
            .filter(|node| {
                tag_filter.is_empty() || tag_filter.iter().any(|t| node.tags.contains(t))
            })
            .collect();
        if nodes.len() < 2 {
            return Ok(vec![]);
        }
        let index_of: HashMap<&str, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id.as_str(), i))
            .collect();

        let n = nodes.len();
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
        let union = |parent: &mut Vec<usize>, a: usize, b: usize| {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[ra] = rb;
            }
        };

        // Nominator 1 (SQL group-by): identical content identity — the stored
        // envelope hash when present, else the exact content itself.
        {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            let mut stmt = reader.prepare(
                "SELECT COALESCE(content_hash, content) AS identity_key, id
                 FROM knowledge_nodes
                 WHERE superseded_by IS NULL AND COALESCE(content_hash, content) IS NOT NULL
                 ORDER BY identity_key",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
            for row in rows {
                let (key, id) = row?;
                if let Some(&idx) = index_of.get(id.as_str()) {
                    groups.entry(key).or_default().push(idx);
                }
            }
            for members in groups.into_values() {
                for pair in members.windows(2) {
                    union(&mut parent, pair[0], pair[1]);
                }
            }
        }

        // Nominator 2 (SQL group-by): the same declared source key, at the
        // same granularity the store's own UNIQUE index uses
        // (system, project, id) so two projects' "issue 42" stay separate.
        {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            let mut stmt = reader.prepare(
                "SELECT source_system || ':' || COALESCE(source_project, '') || ':' || source_id AS source_key, id
                 FROM knowledge_nodes
                 WHERE superseded_by IS NULL
                   AND source_system IS NOT NULL AND source_id IS NOT NULL
                 ORDER BY source_key",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
            for row in rows {
                let (key, id) = row?;
                if let Some(&idx) = index_of.get(id.as_str()) {
                    groups.entry(key).or_default().push(idx);
                }
            }
            for members in groups.into_values() {
                for pair in members.windows(2) {
                    union(&mut parent, pair[0], pair[1]);
                }
            }
        }

        // Nominator 3 (in-memory set compare): exactly equal, non-empty
        // extracted-entity sets.
        {
            let mut groups: HashMap<BTreeSet<String>, Vec<usize>> = HashMap::new();
            for (i, node) in nodes.iter().enumerate() {
                let entities: BTreeSet<String> =
                    crate::advanced::retroactive_backfill::extract_entities(
                        &node.content,
                        &node.tags,
                    )
                    .into_iter()
                    .collect();
                if entities.is_empty() {
                    continue;
                }
                groups.entry(entities).or_default().push(i);
            }
            for members in groups.into_values() {
                for pair in members.windows(2) {
                    union(&mut parent, pair[0], pair[1]);
                }
            }
        }

        // Group indices by root.
        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let r = find(&mut parent, i);
            clusters.entry(r).or_default().push(i);
        }

        let mut out: Vec<MergeCandidate> = Vec::new();
        for members in clusters.into_values() {
            if members.len() < 2 {
                continue;
            }
            // Cluster confidence = weakest pairwise lexical score (the loosest
            // link); the best-scoring pair's signals are the explanation.
            let mut min_score = 1.0f32;
            let mut best_signals: Option<crate::advanced::MatchSignals> = None;
            for a in 0..members.len() {
                for b in (a + 1)..members.len() {
                    let (na, nb) = (&nodes[members[a]], &nodes[members[b]]);
                    let sig = score_pair(&na.tags, &nb.tags, &na.content, &nb.content);
                    if sig.combined_score < min_score {
                        min_score = sig.combined_score;
                    }
                    if best_signals
                        .as_ref()
                        .map(|s| sig.combined_score > s.combined_score)
                        .unwrap_or(true)
                    {
                        best_signals = Some(sig);
                    }
                }
            }
            let signals = match best_signals {
                Some(s) => s,
                None => continue,
            };

            // Survivor = highest retention member.
            let mut ranked: Vec<usize> = members.clone();
            ranked.sort_by(|a, b| {
                let ra = nodes[*a].retention_strength;
                let rb = nodes[*b].retention_strength;
                rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
            });
            let member_ids: Vec<String> = ranked.iter().map(|&idx| nodes[idx].id.clone()).collect();
            let survivor_id = member_ids[0].clone();
            let has_protected_member = member_ids.iter().any(|id| protected.contains(id));
            let previews: Vec<String> = ranked
                .iter()
                .map(|&idx| preview(&nodes[idx].content, 120))
                .collect();

            // Advisory label only. Nomination came from exact equality above,
            // so a low lexical score surfaces the cluster for review rather
            // than dropping it (the old cosine scan dropped NonMatch pairs
            // because its nominations were probabilistic; these are not).
            let classification = policy.classify(min_score);

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

    /// (superseded_id, superseding_id) pairs, so a trail can follow the link
    /// to the current belief instead of stopping at the invalidated record.
    pub fn supersession_pairs(&self) -> Result<Vec<(String, String)>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT id, superseded_by FROM knowledge_nodes WHERE superseded_by IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
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

    /// Plan status string (pending | applied | cancelled | rejected |
    /// quarantined | expired), if the plan exists.
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

    /// Record a non-mutating reconsolidation verdict row (reject / quarantine
    /// marker / expiry) in `merge_operations` and set the plan status. These
    /// ops carry an empty undo payload: `merge_undo` refuses anything that is
    /// not `status='applied'`, which is correct — a rejection has nothing to
    /// reverse, and suppression reversal goes through the suppression path's
    /// own 24-hour labile undo, not the merge reflog.
    fn record_reconsolidation_verdict_op(
        &self,
        tx: &rusqlite::Transaction<'_>,
        plan: &crate::advanced::MergePlan,
        status: &str,
        reason: &str,
    ) -> Result<crate::advanced::MergeOperation> {
        let op_id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now();
        let affected = vec![plan.survivor_id.clone()];
        tx.execute(
            "INSERT INTO merge_operations
                (id, plan_id, op_type, status, created_at, reverted_at, reverts_op_id,
                 survivor_id, affected_ids, confidence, signals, reason, undo_payload)
             VALUES (?1, ?2, 'reconsolidation', ?3, ?4, NULL, NULL, ?5, ?6, ?7, NULL, ?8, '{}')",
            params![
                op_id,
                plan.id,
                status,
                now.to_rfc3339(),
                plan.survivor_id,
                serde_json::to_string(&affected).unwrap_or_else(|_| "[]".into()),
                plan.confidence as f64,
                reason,
            ],
        )?;
        tx.execute(
            "UPDATE merge_plans SET status = ?1, applied_at = ?2 WHERE id = ?3",
            params![status, now.to_rfc3339(), plan.id],
        )?;
        Ok(crate::advanced::MergeOperation {
            id: op_id,
            plan_id: Some(plan.id.clone()),
            op_type: "reconsolidation".to_string(),
            status: status.to_string(),
            created_at: now.to_rfc3339(),
            reverted_at: None,
            reverts_op_id: None,
            survivor_id: Some(plan.survivor_id.clone()),
            affected_ids: affected,
            confidence: Some(plan.confidence),
            signals: None,
            reason: Some(reason.to_string()),
        })
    }

    /// Auto-close one expired reconsolidation plan (status `expired`) and
    /// record the close. Used both by the opportunistic sweep and by the
    /// apply-time guard, so an expired plan can never sit pending or be
    /// applied after its labile window closed — no zombie plans.
    pub fn expire_reconsolidation_plan(
        &self,
        plan: &crate::advanced::MergePlan,
    ) -> Result<crate::advanced::MergeOperation> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "expire_reconsolidation_plan")?;
        let op = self.record_reconsolidation_verdict_op(
            &tx,
            plan,
            "expired",
            "Labile window expired without a verdict; reconsolidation plan auto-closed",
        )?;
        tx.commit()?;
        Ok(op)
    }

    /// Close every pending reconsolidation plan whose labile window has
    /// expired. Returns the closed plan ids. Called from the consolidation
    /// cycle and before listing, so verdict surfaces never offer a stale
    /// conflict.
    pub fn expire_stale_reconsolidation_plans(&self) -> Result<Vec<String>> {
        let stale = self.pending_expired_reconsolidation_plans()?;
        let mut closed = Vec::with_capacity(stale.len());
        for plan in stale {
            self.expire_reconsolidation_plan(&plan)?;
            closed.push(plan.id);
        }
        Ok(closed)
    }

    /// Load pending reconsolidation plans past their window.
    fn pending_expired_reconsolidation_plans(&self) -> Result<Vec<crate::advanced::MergePlan>> {
        let cutoff = Utc::now().to_rfc3339();
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT payload FROM merge_plans
             WHERE kind = 'reconsolidation' AND status = 'pending' AND created_at < ?1",
        )?;
        let mut rows = stmt.query(params![cutoff])?;
        let mut stale = Vec::new();
        while let Some(row) = rows.next()? {
            let payload: String = row.get(0)?;
            if let Ok(plan) = serde_json::from_str::<crate::advanced::MergePlan>(&payload)
                && let Some(meta) = &plan.reconsolidation
                && meta.window_expires_at <= Utc::now()
            {
                stale.push(plan);
            }
        }
        Ok(stale)
    }

    /// Pending reconsolidation plans with their verdict deadline, oldest
    /// first. Runs the expiry sweep first, so a caller never sees a plan
    /// whose window already closed.
    pub fn list_reconsolidation_plans(
        &self,
        limit: usize,
    ) -> Result<Vec<(crate::advanced::MergePlan, String)>> {
        self.expire_stale_reconsolidation_plans()?;
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT payload, status FROM merge_plans
             WHERE kind = 'reconsolidation' AND status = 'pending'
             ORDER BY created_at ASC LIMIT ?1",
        )?;
        let mut rows = stmt.query(params![limit as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let payload: String = row.get(0)?;
            let status: String = row.get(1)?;
            if let Ok(plan) = serde_json::from_str::<crate::advanced::MergePlan>(&payload) {
                out.push((plan, status));
            }
        }
        Ok(out)
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
                    survivor_id, affected_ids, confidence, signals, reason
             FROM merge_operations ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], Self::row_to_operation)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Read one durable merge/tag operation from the memory reflog.
    pub fn get_merge_operation(
        &self,
        operation_id: &str,
    ) -> Result<Option<crate::advanced::MergeOperation>> {
        self.read_operation(operation_id)
    }

    /// Read a single operation by id.
    pub(super) fn read_operation(
        &self,
        op_id: &str,
    ) -> Result<Option<crate::advanced::MergeOperation>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let op = reader
            .query_row(
                "SELECT id, plan_id, op_type, status, created_at, reverted_at, reverts_op_id,
                        survivor_id, affected_ids, confidence, signals, reason
                 FROM merge_operations WHERE id = ?1",
                params![op_id],
                Self::row_to_operation,
            )
            .optional()?;
        Ok(op)
    }

    pub(super) fn row_to_operation(
        row: &rusqlite::Row,
    ) -> rusqlite::Result<crate::advanced::MergeOperation> {
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
            signals: row
                .get::<_, Option<String>>("signals")
                .ok()
                .flatten()
                .and_then(|value| serde_json::from_str(&value).ok()),
            reason: row.get("reason").ok().flatten(),
        })
    }
}

#[cfg(test)]
mod exact_nomination_tests {
    //! merge_candidates nominates by EXACT EQUALITY ONLY (owner decision
    //! 2026-09-28): identical content hash / same declared source key /
    //! exactly equal entity sets. Near-identical content must NOT nominate.

    use crate::IngestInput;
    use crate::storage::SqliteMemoryStore as Storage;

    fn store() -> (tempfile::TempDir, Storage) {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (dir, storage)
    }

    fn ingest(storage: &Storage, content: &str, tags: &[&str]) -> String {
        storage
            .ingest(IngestInput {
                content: content.to_string(),
                tags: tags.iter().map(|t| t.to_string()).collect(),
                ..Default::default()
            })
            .unwrap()
            .id
    }

    fn ingest_with_envelope(
        storage: &Storage,
        content: &str,
        envelope: crate::memory::SourceEnvelope,
    ) -> String {
        storage
            .ingest(IngestInput {
                content: content.to_string(),
                source_envelope: Some(envelope),
                ..Default::default()
            })
            .unwrap()
            .id
    }

    fn envelope(
        source_system: Option<&str>,
        source_id: Option<&str>,
        content_hash: Option<&str>,
    ) -> crate::memory::SourceEnvelope {
        crate::memory::SourceEnvelope {
            source_system: source_system.map(String::from),
            source_id: source_id.map(String::from),
            content_hash: content_hash.map(String::from),
            ..Default::default()
        }
    }

    fn candidate_members(storage: &Storage) -> Vec<Vec<String>> {
        storage
            .merge_candidates(crate::advanced::MergePolicy::default(), 20, &[])
            .unwrap()
            .into_iter()
            .map(|c| c.member_ids)
            .collect()
    }

    #[test]
    fn identical_content_nominate() {
        let (_dir, storage) = store();
        let a = ingest(&storage, "Deploy the gateway before Friday", &[]);
        let b = ingest(&storage, "Deploy the gateway before Friday", &[]);
        let _c = ingest(&storage, "An unrelated memory about cooking", &[]);

        let clusters = candidate_members(&storage);
        assert_eq!(clusters.len(), 1, "exactly one exact-duplicate cluster");
        assert!(clusters[0].contains(&a) && clusters[0].contains(&b));
        assert!(!clusters[0].contains(&_c));
    }

    #[test]
    fn identical_content_hash_nominate_across_different_text() {
        let (_dir, storage) = store();
        // Same declared payload hash, different raw text (e.g. two renderings
        // of the same upstream record). The stored hash is the identity.
        let a = ingest_with_envelope(
            &storage,
            "issue 7: timeout on import",
            envelope(None, None, Some("sha256:abc")),
        );
        let b = ingest_with_envelope(
            &storage,
            "issue 7: timeout during import (reformatted)",
            envelope(None, None, Some("sha256:abc")),
        );

        let clusters = candidate_members(&storage);
        assert_eq!(clusters.len(), 1);
        assert!(clusters[0].contains(&a) && clusters[0].contains(&b));
    }

    #[test]
    fn same_source_key_nominate_even_with_diverged_content() {
        let (_dir, storage) = store();
        // Fresh schemas enforce a UNIQUE index on the source key, so a
        // same-key duplicate can only exist in a store written before that
        // constraint (or with it relaxed). Simulate that legacy state by
        // dropping the index for the duration of the test.
        {
            let writer = storage.writer.lock().unwrap();
            writer
                .execute_batch("DROP INDEX idx_nodes_source_key")
                .unwrap();
        }
        let a = ingest_with_envelope(
            &storage,
            "Redmine 42: original description",
            envelope(Some("redmine"), Some("42"), None),
        );
        let b = ingest_with_envelope(
            &storage,
            "Redmine 42: edited description after upstream change",
            envelope(Some("redmine"), Some("42"), None),
        );
        // A different source key must not join.
        let c = ingest_with_envelope(
            &storage,
            "Redmine 42: cross-posted note",
            envelope(Some("jira"), Some("42"), None),
        );

        let clusters = candidate_members(&storage);
        assert_eq!(clusters.len(), 1);
        assert!(clusters[0].contains(&a) && clusters[0].contains(&b));
        assert!(!clusters[0].contains(&c));
    }

    #[test]
    fn near_identical_content_no_longer_nominate() {
        let (_dir, storage) = store();
        // Under the old cosine scan this pair scored ~0.99 and clustered.
        // It shares no content hash, no source key, and its extracted entity
        // sets differ (services vs service), so exact-equality nomination
        // must NOT offer it.
        ingest(&storage, "Use tokio runtime for async Rust services", &[]);
        ingest(
            &storage,
            "Use the tokio runtime for async Rust service",
            &[],
        );

        let clusters = candidate_members(&storage);
        assert!(
            clusters.is_empty(),
            "near-identical content must not be nominated: {clusters:?}"
        );
    }

    #[test]
    fn exact_entity_set_nominate() {
        let (_dir, storage) = store();
        // Two notes over the exact same entity set {alpha, beta, gamma,
        // delta}: different word order and stopword filler (words shorter
        // than 4 letters are not entities).
        ingest(&storage, "alpha beta gamma delta", &[]);
        ingest(&storage, "delta or gamma and beta of alpha", &[]);
        // A third note with a different entity set stays out.
        ingest(&storage, "tokio runtime tuning for the worker pool", &[]);

        let clusters = candidate_members(&storage);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 2);
    }

    #[test]
    fn tag_filter_restricts_nomination() {
        let (_dir, storage) = store();
        ingest(&storage, "Duplicated release note text", &["rust"]);
        ingest(&storage, "Duplicated release note text", &["python"]);

        let clusters = candidate_members(&storage);
        assert_eq!(clusters.len(), 1, "no tag filter: the pair is nominated");

        let filtered = storage
            .merge_candidates(
                crate::advanced::MergePolicy::default(),
                20,
                &["rust".to_string()],
            )
            .unwrap();
        assert!(
            filtered.is_empty(),
            "with one member filtered out, the cluster dissolves"
        );
    }
}
