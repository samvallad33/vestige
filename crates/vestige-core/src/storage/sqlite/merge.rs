//! Merge and supersede controls: protection flags, merge policy, candidate
//! scoring, diff-previewed plans, apply, undo and the bitemporal helpers
//! behind them.

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
        // The survivor MUST be one of the members. A caller-supplied survivor_id
        // that isn't in member_ids (a typo/mixup through the plan_merge tool)
        // otherwise sails through and panics at the `.find(...).unwrap()` below,
        // taking down the request. Reject it with a clear error instead.
        if !nodes.iter().any(|n| n.id == survivor) {
            return Err(StorageError::Init(format!(
                "survivor_id {survivor} is not among the member_ids being merged"
            )));
        }

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
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
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

        // The whole apply is ONE IMMEDIATE transaction, and the undo row is
        // written FIRST inside it.
        //
        // The old shape mutated through helpers that each committed on their
        // own, and only afterwards inserted the reflog row. Any failure between
        // the survivor rewrite and that insert left the survivor's content
        // overwritten with NO undo row at all: unrecoverable. SQLITE_BUSY was
        // the realistic trigger, because several MCP server processes share one
        // database file. The plan-status check was not atomic with the
        // mutations either, so two processes could both pass it and both apply.
        //
        // Now: the status re-check, the undo row, the plan transition, the
        // survivor rewrite and the invalidations either all commit or none do,
        // and the status re-check inside the transaction closes the race.
        //
        // The embedding is deliberately NOT regenerated in here. That is model
        // inference behind a tokio runtime, and holding the write lock across
        // it is the same defect as holding a mutex across a model load. The
        // transaction marks the survivor `has_embedding = 0` instead, and the
        // regeneration runs after COMMIT. If it fails, or the process dies
        // first, the node is already flagged and the consolidation cycle's
        // `generate_missing_embeddings` rebuilds it. Stale-but-flagged is
        // recoverable; overwritten-with-no-undo-row is not.
        let mut content_changed = false;
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            let tx = Self::begin_write_transaction(&writer, "apply_plan")?;

            // Re-check status INSIDE the transaction. Two processes can both
            // pass the read above; only one can pass this.
            let status: Option<String> = tx
                .query_row(
                    "SELECT status FROM merge_plans WHERE id = ?1",
                    params![plan_id],
                    |row| row.get(0),
                )
                .optional()?;
            match status.as_deref() {
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

            // Snapshot everything we need to undo, BEFORE mutating, and from
            // inside the same transaction so the snapshot and the mutation see
            // one consistent state.
            let mut undo = serde_json::Map::new();
            undo.insert("plan_id".into(), serde_json::json!(plan_id));
            undo.insert("kind".into(), serde_json::json!(plan.kind.as_str()));
            undo.insert("survivor_id".into(), serde_json::json!(plan.survivor_id));

            match plan.kind {
                PlanKind::Merge => {
                    let (prev_content, prev_tags_json): (String, String) = tx
                        .query_row(
                            "SELECT content, COALESCE(tags, '[]') FROM knowledge_nodes WHERE id = ?1",
                            params![plan.survivor_id],
                            |row| Ok((row.get(0)?, row.get(1)?)),
                        )
                        .optional()?
                        .ok_or_else(|| StorageError::NotFound(plan.survivor_id.clone()))?;
                    let prev_tags: Vec<String> =
                        serde_json::from_str(&prev_tags_json).unwrap_or_default();
                    undo.insert(
                        "survivor_prev_content".into(),
                        serde_json::json!(prev_content),
                    );
                    undo.insert("survivor_prev_tags".into(), serde_json::json!(prev_tags));

                    let mut absorbed = Vec::new();
                    for id in &plan.invalidated_ids {
                        let (vu, sb) = Self::read_bitemporal_in_transaction(&tx, id)?;
                        absorbed.push(serde_json::json!({
                            "id": id,
                            "prev_valid_until": vu,
                            "prev_superseded_by": sb,
                        }));
                    }
                    undo.insert("absorbed".into(), serde_json::json!(absorbed));
                    content_changed = prev_content != plan.result_content;
                }
                PlanKind::Supersede => {
                    let old_id = &plan.member_ids[0];
                    let (vu, sb) = Self::read_bitemporal_in_transaction(&tx, old_id)?;
                    undo.insert(
                        "absorbed".into(),
                        serde_json::json!([{
                            "id": old_id,
                            "prev_valid_until": vu,
                            "prev_superseded_by": sb,
                        }]),
                    );
                }
            }

            let affected: Vec<String> = {
                let mut v = vec![plan.survivor_id.clone()];
                v.extend(plan.invalidated_ids.clone());
                v
            };
            let signals = serde_json::to_string(&plan.signals).unwrap_or_else(|_| "{}".into());

            // The undo row goes in FIRST. Nothing below it can leave a mutation
            // without its reversal, because nothing below it commits alone.
            tx.execute(
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
            tx.execute(
                "UPDATE merge_plans SET status = 'applied', applied_at = ?1 WHERE id = ?2",
                params![now.to_rfc3339(), plan_id],
            )?;

            match plan.kind {
                PlanKind::Merge => {
                    let tags_json =
                        serde_json::to_string(&plan.result_tags).unwrap_or_else(|_| "[]".into());
                    tx.execute(
                        "UPDATE knowledge_nodes SET content = ?1, tags = ?2, updated_at = ?3
                         WHERE id = ?4",
                        params![
                            plan.result_content,
                            tags_json,
                            now.to_rfc3339(),
                            plan.survivor_id
                        ],
                    )?;
                    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
                    if content_changed {
                        // Flag for rebuild before COMMIT, so a crash between
                        // here and the regeneration below is self-healing.
                        tx.execute(
                            "UPDATE knowledge_nodes SET has_embedding = 0 WHERE id = ?1",
                            params![plan.survivor_id],
                        )?;
                    }
                    for id in &plan.invalidated_ids {
                        Self::invalidate_node_in_transaction(&tx, id, &plan.survivor_id, now)?;
                    }
                }
                PlanKind::Supersede => {
                    let old_id = &plan.member_ids[0];
                    Self::invalidate_node_in_transaction(&tx, old_id, &plan.survivor_id, now)?;
                }
            }

            tx.commit()?;
        }

        // Committed. Regenerate the survivor's embedding outside the write
        // lock; `has_embedding = 0` is already persisted, so failure here is
        // recoverable by the next consolidation cycle rather than silent.
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        if content_changed {
            if let Some(index) = self.vector_index.as_ref()
                && let Ok(mut index) = index.lock()
            {
                let _ = index.remove(&plan.survivor_id);
            }
            if self.active_embedding_runtime_ready().unwrap_or(false)
                && let Err(e) =
                    self.generate_embedding_for_node(&plan.survivor_id, &plan.result_content)
            {
                tracing::warn!(
                    survivor_id = %plan.survivor_id,
                    error = %e,
                    "apply_plan committed but could not regenerate the survivor embedding; \
                     it stays has_embedding=0 for the next consolidation sweep"
                );
            }
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
        if matches!(op.op_type.as_str(), "tag_rename" | "tag_merge") {
            return self.undo_tag_mutation(op_id);
        }
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

    /// Read (valid_until, superseded_by) for a node.
    /// Read a node's bitemporal columns off the reader connection. Production
    /// now snapshots inside the apply transaction instead (see
    /// [`Self::read_bitemporal_in_transaction`]); this remains as the assertion
    /// helper the merge/supersede tests read state through.
    #[cfg(all(test, feature = "embeddings", feature = "vector-search"))]
    pub(super) fn read_bitemporal(&self, id: &str) -> Result<(Option<String>, Option<String>)> {
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

    /// `read_bitemporal` against an open transaction, so a snapshot and the
    /// mutation it protects observe the same database state.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn read_bitemporal_in_transaction(
        tx: &rusqlite::Transaction<'_>,
        id: &str,
    ) -> Result<(Option<String>, Option<String>)> {
        let res = tx
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

    /// `invalidate_node` against an open transaction. The helper that takes the
    /// writer lock itself cannot be called from inside a transaction: the lock
    /// is not reentrant, so it would deadlock.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn invalidate_node_in_transaction(
        tx: &rusqlite::Transaction<'_>,
        id: &str,
        superseded_by: &str,
        now: DateTime<Utc>,
    ) -> Result<()> {
        tx.execute(
            "UPDATE knowledge_nodes
             SET valid_until = ?1, superseded_by = ?2, updated_at = ?1
             WHERE id = ?3",
            params![now.to_rfc3339(), superseded_by, id],
        )?;
        Ok(())
    }

    /// Restore a node's bitemporal columns (used by undo).
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
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
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
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
