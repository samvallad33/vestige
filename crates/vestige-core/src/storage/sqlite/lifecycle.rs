//! Cognitive lifecycle: access strengthening, promotion and demotion,
//! failure feedback, suppression and quarantine, decay, consolidation,
//! auto-dedup and retention statistics.

use super::*;

impl SqliteMemoryStore {
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

    /// Reinforce a memory after an intentional confirmation of relevance.
    ///
    /// Ordinary retrieval must use [`Self::record_batch_retrieval`] instead:
    /// being shown in search is not evidence that a memory was correct or
    /// useful. This helper remains for explicit duplicate/reinforcement flows.
    /// It implements the Testing Effect (Roediger & Karpicke 2006) + v1.4.0
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

        // This is a deliberate reinforcement, not a passive search hit.
        let _ = self.log_access(id, "reinforce");

        // Content-aware cross-memory reinforcement: boost semantically similar neighbors
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        {
            if let Some(index) = self.vector_index.as_ref()
                && let Ok(Some(embedding)) = self.get_node_embedding(id)
            {
                let index = index
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

    /// Batch-strengthen memories after an intentional confirmation of relevance.
    pub fn strengthen_batch_on_access(&self, ids: &[&str]) -> Result<()> {
        for id in ids {
            self.strengthen_on_access(id)?;
            // Also record access in memory_states for audit trail (Bug #1 fix)
            let _ = self.record_memory_access(id);
        }
        Ok(())
    }

    /// Record that a memory was returned to a caller without reinforcing it.
    ///
    /// A search hit is not proof of correctness or usefulness. We retain only
    /// access-log evidence for auditability, leaving node state and every
    /// learning/ranking signal untouched. Call [`Self::promote_memory`] or
    /// [`Self::mark_memory_useful`] only after an explicit positive signal.
    pub fn record_batch_retrieval(&self, ids: &[&str]) -> Result<()> {
        for id in ids {
            self.log_access(id, "retrieval_shown")?;
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

    /// Log a memory interaction for audit and explicit-feedback learning.
    pub(crate) fn log_access(&self, node_id: &str, access_type: &str) -> Result<()> {
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

        // Explicit positive feedback: boost strength and record that this
        // memory proved useful to the caller.
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
                    stability = stability * 1.5,
                    times_useful = COALESCE(times_useful, 0) + 1,
                    utility_score = CASE
                        WHEN COALESCE(times_retrieved, 0) > 0
                        THEN MIN(1.0, CAST(COALESCE(times_useful, 0) + 1 AS REAL) / COALESCE(times_retrieved, 0))
                        ELSE 1.0
                    END
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

    /// Backfill-specific promote: identical retrieval/retention boost to
    /// `promote_memory`, but the stability multiply is CAPPED at an additive
    /// +365-day ceiling: `MIN(stability * 1.5, stability + 365.0)`. The `1.5`
    /// factor preserves the multiplier `promote_memory` already applied; the
    /// `+365` ceiling is the same additive bound `retroactive_backfill.rs`
    /// uses for its reason string (that module pairs +365 with a 2.5 factor
    /// for display only — this DB write intentionally keeps 1.5 so backfill
    /// promotion strength is unchanged, just bounded). Repeated per-(cause,
    /// failure) backfill promotions therefore cannot inflate stability without
    /// bound. Used by the step-8.5 auto-fire path and the manual `backfill` tool.
    pub fn promote_memory_backfill(&self, id: &str) -> Result<KnowledgeNode> {
        let now = Utc::now();

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
                    stability = MIN(stability * 1.5, stability + 365.0)
                WHERE id = ?2",
                params![now.to_rfc3339(), id],
            )?;
        }

        let _ = self.log_access(id, "promote");
        let _ = self.set_waking_tag(id);

        self.get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    /// Demote a memory (thumbs down) - used when a memory led to a bad outcome
    /// Significantly reduces retrieval strength so better alternatives surface
    /// Does NOT delete - the memory stays for reference but ranks lower
    pub fn demote_memory(&self, id: &str) -> Result<KnowledgeNode> {
        // Strong penalty: -0.3 retrieval, -0.15 retention, halve stability
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            // last_accessed intentionally untouched -- see suppress_memory: a
            // demotion is an inhibition event, not a recall, and apply_decay
            // would otherwise recompute the penalty away.
            writer.execute(
                "UPDATE knowledge_nodes SET
                    retrieval_strength = MAX(0.05, retrieval_strength - 0.30),
                    retention_strength = MAX(0.05, retention_strength - 0.15),
                    stability = stability * 0.5
                WHERE id = ?1",
                params![id],
            )?;
        }

        let _ = self.log_access(id, "demote");

        self.get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    // ========================================================================
    // Post-retrieval failure feedback (Heinbockel, Leicht, Wagner, Schwabe 2025)
    // ========================================================================

    /// Lower the accessibility of the memories retrieved shortly before a
    /// failure landed, in proportion to how strongly each was reactivated.
    ///
    /// Heinbockel et al. (eLife 2025, PMID 39878439): noradrenergic arousal
    /// after a memory was cued impaired its later recall, scaled by how strongly
    /// the memory had been reactivated during cueing. The software reading: the
    /// failure is the arousal event, the receipts already say which memories
    /// informed the decisions just before it and in what order, and rank in a
    /// receipt is the reactivation strength (weight 1/(rank+1)). A retrieval is
    /// therefore not a monotone positive signal; this is the opposite-sign term
    /// the FSRS review update lacks, and the first mechanism by which a wrong
    /// memory leaves the top of recall without anyone demoting it by hand.
    ///
    /// Bounded: at most `FAILURE_FEEDBACK_PENALTY` (0.10) of retrieval strength
    /// per memory per failure, never below the 0.05 floor, never touching
    /// stability or content. Scoped: only memories in the failure's scope.
    /// Idempotent per (failure, memory). Recorded in `failure_feedback` and
    /// reversible with [`Self::revert_failure_feedback`]. Retrieval strength
    /// recovers on the next successful access, exactly as after `demote`.
    pub fn apply_failure_feedback(
        &self,
        failure_id: &str,
        window: Duration,
    ) -> Result<FailureFeedbackReport> {
        const FAILURE_FEEDBACK_PENALTY: f64 = 0.10;
        const MAX_RECEIPTS: i64 = 25;

        let (failure_created_at, failure_scope): (String, String) = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            reader
                .query_row(
                    "SELECT created_at, COALESCE(scope, 'user') FROM knowledge_nodes WHERE id = ?1",
                    params![failure_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?
                .ok_or_else(|| StorageError::NotFound(failure_id.to_string()))?
        };
        let until = DateTime::parse_from_rfc3339(&failure_created_at)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());
        let since = until - window;

        // Receipts in the window, newest first. The payload is the serialized
        // Receipt; `retrieved` is best-first, which is the rank we need.
        let receipts: Vec<(String, String)> = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            let mut stmt = reader.prepare(
                "SELECT receipt_id, payload FROM memory_receipts
                 WHERE created_at >= ?1 AND created_at <= ?2
                 ORDER BY created_at DESC LIMIT ?3",
            )?;
            let rows = stmt.query_map(
                params![since.to_rfc3339(), until.to_rfc3339(), MAX_RECEIPTS],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )?;
            rows.filter_map(warn_skipped_row("apply_failure_feedback"))
                .collect()
        };

        // memory_id -> (weight, receipt_id, rank); the strongest reactivation wins.
        let mut weights: HashMap<String, (f64, String, usize)> = HashMap::new();
        for (receipt_id, payload) in &receipts {
            let Ok(receipt) = serde_json::from_str::<crate::trace::Receipt>(payload) else {
                continue;
            };
            for (rank, memory_id) in receipt.retrieved.iter().enumerate() {
                if memory_id == failure_id {
                    continue;
                }
                let weight = 1.0 / (rank as f64 + 1.0);
                let stronger = weights
                    .get(memory_id)
                    .is_none_or(|(existing, _, _)| weight > *existing);
                if stronger {
                    weights.insert(memory_id.clone(), (weight, receipt_id.clone(), rank));
                }
            }
        }

        let mut report = FailureFeedbackReport {
            failure_id: failure_id.to_string(),
            window_minutes: window.num_minutes(),
            receipts_considered: receipts.len(),
            memories_demoted: 0,
            total_delta: 0.0,
        };
        if weights.is_empty() {
            return Ok(report);
        }

        let mut demoted_ids = Vec::new();
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            let tx = Self::begin_write_transaction(&writer, "apply_failure_feedback")?;
            let now = Utc::now().to_rfc3339();
            for (memory_id, (weight, receipt_id, rank)) in weights {
                let already: i64 = tx.query_row(
                    "SELECT COUNT(*) FROM failure_feedback WHERE failure_id = ?1 AND memory_id = ?2",
                    params![failure_id, &memory_id],
                    |row| row.get(0),
                )?;
                if already > 0 {
                    continue;
                }
                // Same scope only, and the memory must still exist.
                let scope: Option<String> = tx
                    .query_row(
                        "SELECT COALESCE(scope, 'user') FROM knowledge_nodes WHERE id = ?1",
                        params![&memory_id],
                        |row| row.get(0),
                    )
                    .optional()?;
                if scope.as_deref() != Some(failure_scope.as_str()) {
                    continue;
                }
                let delta = FAILURE_FEEDBACK_PENALTY * weight;
                tx.execute(
                    "UPDATE knowledge_nodes
                     SET retrieval_strength = MAX(0.05, retrieval_strength - ?1)
                     WHERE id = ?2",
                    params![delta, &memory_id],
                )?;
                tx.execute(
                    "INSERT INTO failure_feedback
                        (failure_id, memory_id, receipt_id, rank, delta, applied_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![failure_id, &memory_id, receipt_id, rank as i64, delta, now],
                )?;
                report.memories_demoted += 1;
                report.total_delta += delta;
                demoted_ids.push(memory_id);
            }
            tx.commit()?;
        }
        for memory_id in demoted_ids {
            let _ = self.log_access(&memory_id, "failure_feedback");
        }
        Ok(report)
    }

    /// Undo every accessibility delta applied for `failure_id` that has not
    /// been reverted yet. Returns how many memories were restored.
    pub fn revert_failure_feedback(&self, failure_id: &str) -> Result<usize> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "revert_failure_feedback")?;
        let rows: Vec<(i64, String, f64)> = {
            let mut stmt = tx.prepare(
                "SELECT id, memory_id, delta FROM failure_feedback
                 WHERE failure_id = ?1 AND reverted_at IS NULL",
            )?;
            let mapped = stmt.query_map(params![failure_id], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?;
            mapped
                .filter_map(warn_skipped_row("revert_failure_feedback"))
                .collect()
        };
        let now = Utc::now().to_rfc3339();
        for (row_id, memory_id, delta) in &rows {
            tx.execute(
                "UPDATE knowledge_nodes
                 SET retrieval_strength = MIN(1.0, retrieval_strength + ?1)
                 WHERE id = ?2",
                params![delta, memory_id],
            )?;
            tx.execute(
                "UPDATE failure_feedback SET reverted_at = ?1 WHERE id = ?2",
                params![now, row_id],
            )?;
        }
        tx.commit()?;
        Ok(rows.len())
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
            let tx = Self::begin_write_transaction(&writer, "suppress_memory")?;
            let changed = tx.execute(
                // NOTE: last_accessed is deliberately NOT touched here. apply_decay
                // RECOMPUTES retrieval_strength/retention_strength from
                // days_since(last_accessed) rather than decaying the stored value,
                // so stamping "now" would make an inhibited memory look freshly
                // recalled and the next consolidation pass would overwrite this
                // whole penalty -- silently un-suppressing it within hours.
                "UPDATE knowledge_nodes SET
                    suppression_count = COALESCE(suppression_count, 0) + 1,
                    suppressed_at = ?1,
                    retrieval_strength = MAX(0.05, retrieval_strength - 0.35),
                    retention_strength = MAX(0.05, retention_strength - 0.20),
                    stability = stability * 0.4
                WHERE id = ?2",
                params![now.to_rfc3339(), id],
            )?;
            if changed == 0 {
                return Err(StorageError::NotFound(id.to_string()));
            }
            Self::invalidate_replay_evidence_for_memory_in_transaction(
                &tx,
                id,
                crate::storage::ReplayInvalidationReason::Suppressed,
            )?;
            tx.commit()?;
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
            // True inverse of suppress_memory (which applies stability * 0.4,
            // retrieval - 0.35, retention - 0.20). Dividing by 0.4 exactly undoes
            // the * 0.4, and adding back the same 0.35 / 0.20 deltas (clamped to
            // 1.0) undoes the subtraction. Previously this used non-inverse deltas
            // (* 1.25, + 0.15, + 0.10), so suppress-then-reverse left stability
            // permanently halved (0.4 * 1.25 = 0.5) while reporting a full undo.
            // Note: where the forward pass hit the MAX(0.05) floor, the exact
            // pre-value is unrecoverable without a snapshot — that clip aside,
            // this restores the pre-suppression FSRS state.
            writer.execute(
                "UPDATE knowledge_nodes SET
                    suppression_count = MAX(0, COALESCE(suppression_count, 0) - 1),
                    suppressed_at = CASE
                        WHEN COALESCE(suppression_count, 0) - 1 <= 0 THEN NULL
                        ELSE suppressed_at
                    END,
                    retrieval_strength = MIN(1.0, retrieval_strength + 0.35),
                    retention_strength = MIN(1.0, retention_strength + 0.20),
                    stability = stability / 0.4
                WHERE id = ?1",
                params![id],
            )?;
        }

        let _ = self.log_access(id, "reverse_suppress");

        self.get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    /// Release a memory from quarantine **unconditionally** (no labile-window
    /// limit), used when a Memory PR is approved.
    ///
    /// Unlike [`Self::reverse_suppression`] (which models a time-bounded "undo"
    /// of an active-forgetting decision and refuses after the labile window),
    /// approving a quarantined risky write is an explicit reviewer decision that
    /// must always restore the memory's retrieval influence — even days later.
    /// Fully clears the suppression (count → 0, `suppressed_at` → NULL) and
    /// restores strengths. A no-op (returns the node) if it isn't suppressed.
    pub fn release_quarantine(&self, id: &str) -> Result<KnowledgeNode> {
        let node = self
            .get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))?;

        if node.suppression_count == 0 && node.suppressed_at.is_none() {
            // Nothing to release — idempotent.
            return Ok(node);
        }

        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            writer.execute(
                "UPDATE knowledge_nodes SET
                    suppression_count = 0,
                    suppressed_at = NULL,
                    retrieval_strength = MIN(1.0, retrieval_strength + 0.15),
                    retention_strength = MIN(1.0, retention_strength + 0.10),
                    stability = stability * 1.25
                WHERE id = ?1",
                params![id],
            )?;
        }

        let _ = self.log_access(id, "release_quarantine");

        self.get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    /// Test-only: backdate a node's `suppressed_at` to simulate a suppression
    /// that happened long ago (e.g. to verify release works past the labile
    /// window). `pub(crate)` so sibling test modules can reach it.
    #[cfg(test)]
    pub(crate) fn set_suppressed_at_for_test(&self, id: &str, when: DateTime<Utc>) {
        if let Ok(writer) = self.writer.lock() {
            let _ = writer.execute(
                "UPDATE knowledge_nodes SET suppressed_at = ?1 WHERE id = ?2",
                params![when.to_rfc3339(), id],
            );
        }
    }

    /// Backdate a node's `created_at`. Intended for tests and demo seeding (e.g.
    /// to simulate a memory formed days ago so Retroactive Salience Backfill can
    /// reach back to it). Cross-crate `pub` so the MCP backfill test + demo
    /// harness can plant a dated cause. Returns Ok(()) on success.
    pub fn set_created_at(&self, id: &str, when: DateTime<Utc>) -> Result<()> {
        if let Ok(writer) = self.writer.lock() {
            writer.execute(
                "UPDATE knowledge_nodes SET created_at = ?1 WHERE id = ?2",
                params![when.to_rfc3339(), id],
            )?;
        }
        Ok(())
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
                    .filter_map(warn_skipped_row("apply_decay"))
                    .collect()
            };

            if batch.is_empty() {
                break;
            }

            let batch_len = batch.len() as i64;

            // Write batch using writer transaction
            {
                let writer = self
                    .writer
                    .lock()
                    .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
                let tx = Self::begin_write_transaction(&writer, "apply_decay")?;

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

        // Before decay, remove residual recency supplied only by the legacy
        // passive-search behavior. Otherwise a memory last shown just before
        // the upgrade would incorrectly avoid its first post-upgrade decay.
        let _ = self.repair_legacy_passive_retrieval_state();

        // v1.5.0: Use SleepConsolidation for structured consolidation
        let sleep = crate::SleepConsolidation::new();

        // Repair stability values that escaped the MAX_STABILITY invariant
        // before the sentiment-boost clamp existed (issue #121): a real store
        // was measured carrying five outliers up to 1.4e24 days. Idempotent,
        // and a no-op on healthy stores.
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            let repaired = writer.execute(
                "UPDATE knowledge_nodes SET stability = ?1 WHERE stability > ?1",
                params![crate::fsrs::MAX_STABILITY],
            )?;
            if repaired > 0 {
                tracing::warn!(
                    repaired,
                    "clamped runaway stability values back to MAX_STABILITY"
                );
            }
        }

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
                    .filter_map(warn_skipped_row("run_consolidation"))
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

        // 3. Generate missing and model-mismatched embeddings.
        // This must drain the whole set so embedder upgrades do not strand v1 corpora.
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

        // 6b. Prune the vector journal (#181): ids only, kept long enough for
        // every peer process to absorb them, then trimmed.
        let _ = self.prune_vector_journal();

        // 6.5. Prune old Black Box trace events (keep 30 days by default;
        // VESTIGE_TRACE_RETENTION_DAYS overrides, 0 = keep forever). Best-effort
        // like the access-log sweep: a failure never blocks consolidation.
        let _ = self.prune_agent_traces();

        // 6.6. Fold the WAL back into the main database while the store is
        // quiet. `wal_autocheckpoint` (1000 pages) already runs on commit, but
        // a PASSIVE checkpoint here keeps the .wal from ratcheting upward over
        // long uptimes. Best-effort like the sweeps above.
        match self.checkpoint_wal(WalCheckpointMode::Passive) {
            Ok(status) => tracing::debug!(
                log_frames = status.log_frames,
                checkpointed_frames = status.checkpointed_frames,
                busy = status.busy,
                "WAL checkpoint after consolidation"
            ),
            Err(error) => {
                tracing::warn!(%error, "WAL checkpoint after consolidation failed")
            }
        }

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

        // 8.5. Retroactive Salience Backfill — memory with hindsight (auto-fire).
        //
        // The dream pass (step 8) replays memories forward to synthesize insights.
        // This is its backward twin: when a recent memory is a salient FAILURE,
        // reach BACKWARD across history and PROMOTE the quiet earlier memory that
        // caused it — the root cause a semantic search structurally cannot surface
        // because it is causally upstream, not *similar*. Faithful port of the
        // offline ensemble co-reactivation in Zaki/Cai et al. 2024 Nature; the
        // consolidation pass IS the offline window. Bounded on every axis so a
        // noisy day cannot trigger a promotion storm, and idempotent across cycles
        // via a durable causal edge (so the same cause is promoted once per
        // failure, not every cycle).
        //
        // OPT-OUT (backfill-safety, v2.2.1): auto-fire is ON by default — it shipped
        // and was documented in v2.2.0, so we keep the behavior — but is now bounded
        // and disableable. It mutates FSRS scores on the canonical store and can lift
        // a memory across a downstream consolidation floor, so a consumer that reads
        // `stability` as a durability gate can turn it off with
        // VESTIGE_BACKFILL_AUTOFIRE=0 (or false/off/no). The `backfill` MCP tool + CLI
        // remain available for on-demand, operator-driven backfill regardless of the
        // gate. The promote is bounded: both the auto-fire and manual paths call
        // promote_memory_backfill (stability = MIN(stability*1.5, stability+365)) so
        // repeated per-(cause, failure) promotions cannot inflate without bound (the
        // prior comment claimed promote_memory was capped — it was not).
        let backfill_autofire = std::env::var("VESTIGE_BACKFILL_AUTOFIRE")
            .map(|v| {
                let v = v.trim();
                !(v.eq_ignore_ascii_case("false")
                    || v.eq_ignore_ascii_case("off")
                    || v.eq_ignore_ascii_case("no")
                    || v == "0")
            })
            .unwrap_or(true);
        let mut backfilled_causes = 0i64;
        if backfill_autofire {
            use crate::advanced::retroactive_backfill::{
                self as rb, BackfillCandidate, FailureEvent, RetroactiveBackfill,
            };
            const MAX_FAILURES_PER_CYCLE: usize = 5;
            const CANDIDATE_SCAN: i32 = 500;

            let recent = self.get_all_nodes(CANDIDATE_SCAN, 0).unwrap_or_default();
            let failures: Vec<&KnowledgeNode> = recent
                .iter()
                .filter(|n| rb::looks_like_failure(&n.content, &n.tags))
                .take(MAX_FAILURES_PER_CYCLE)
                .collect();

            if !failures.is_empty() {
                let backfill = RetroactiveBackfill::new();
                let mut already_promoted: std::collections::HashSet<(String, String)> =
                    std::collections::HashSet::new();

                for failure_node in failures {
                    let failure = FailureEvent {
                        id: failure_node.id.clone(),
                        content: failure_node.content.clone(),
                        entities: rb::extract_entities(&failure_node.content, &failure_node.tags),
                        tags: failure_node.tags.clone(),
                        prediction_error: 0.9,
                        manual: false,
                    };
                    // candidates = every OTHER memory strictly older than the
                    // failure, EXCLUDING other failures (a root cause is the quiet
                    // upstream change, not an earlier crash).
                    let candidates: Vec<BackfillCandidate> = recent
                        .iter()
                        .filter(|c| c.id != failure_node.id)
                        .filter(|c| !rb::looks_like_failure(&c.content, &c.tags))
                        .filter_map(|c| {
                            let age = (failure_node.created_at - c.created_at).num_seconds() as f64
                                / 86_400.0;
                            if age <= 0.0 {
                                return None;
                            }
                            Some(BackfillCandidate {
                                id: c.id.clone(),
                                content: c.content.clone(),
                                entities: rb::extract_entities(&c.content, &c.tags),
                                age_days_before_failure: age,
                                stability: c.stability,
                                similarity_to_failure: None,
                            })
                        })
                        .collect();

                    let result = backfill.run(&failure, &candidates);
                    if !result.triggered {
                        continue;
                    }
                    for cause in &result.causes {
                        if !already_promoted
                            .insert((cause.memory_id.clone(), failure_node.id.clone()))
                        {
                            continue;
                        }
                        // Cross-cycle idempotency: a durable causal edge is both the
                        // dedup key and a first-class artifact. Write it FIRST, only
                        // promote if it persisted (a failed edge write => retry next
                        // cycle cleanly, never double-inflate).
                        let link_type = crate::memory::EdgeType::Causal.to_string();
                        let already_linked = self
                            .get_connections_for_memory(&cause.memory_id)
                            .map(|conns| {
                                conns.iter().any(|c| {
                                    c.source_id == cause.memory_id
                                        && c.target_id == failure_node.id
                                        && c.link_type == link_type
                                })
                            })
                            .unwrap_or(false);
                        if already_linked {
                            continue;
                        }
                        let conn = ConnectionRecord {
                            source_id: cause.memory_id.clone(),
                            target_id: failure_node.id.clone(),
                            strength: 1.0,
                            link_type,
                            created_at: Utc::now(),
                            last_activated: Utc::now(),
                            activation_count: 0,
                        };
                        if self.save_connection(&conn).is_err() {
                            continue;
                        }
                        if self.promote_memory_backfill(&cause.memory_id).is_ok() {
                            backfilled_causes += 1;
                        }
                    }
                }
                if backfilled_causes > 0 {
                    tracing::info!(
                        backfilled_causes,
                        "Retroactive Salience Backfill: promoted {} root-cause memor{} a semantic search would miss",
                        backfilled_causes,
                        if backfilled_causes == 1 { "y" } else { "ies" }
                    );
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

        // 19. Retention Target System — REPORT ONLY. Consolidation never
        // deletes memories.
        //
        // Until v2.6.0 this step hard-deleted every memory below 0.3
        // retention older than 30 days whenever average retention slipped
        // under a target. It looked dormant for months only because decay was
        // broken (the w20 story in fsrs/optimizer.rs); the day decay came
        // back to life it silently destroyed 23 real memories from a live
        // 2,929-memory store in a single cycle — unattended, unrecoverable,
        // invisible in the consolidation output, and with no protected-pin
        // exemption. Forgetting in Vestige means DOWN-RANKING (the
        // accessibility states); destruction is reserved for the explicit,
        // previewable, dry-run-by-default `maintain {action:"gc"}` and
        // `purge` paths. VESTIGE_RETENTION_TARGET no longer gates anything
        // destructive.
        {
            let avg_retention = self.get_avg_retention().unwrap_or(1.0);
            let total = self.get_stats().map(|s| s.total_nodes).unwrap_or(0);
            let below_target = self.count_memories_below_retention(0.3).unwrap_or(0);

            if below_target > 0 {
                tracing::info!(
                    avg_retention,
                    gc_candidates = below_target,
                    "{} memories sit below 0.3 retention; review them with maintain {{action:\"gc\", dry_run:true}} — consolidation deletes nothing",
                    below_target
                );
            }

            // 20. Save retention snapshot for trend tracking. `gc_triggered`
            // is permanently false: the autonomic GC no longer exists.
            let _ = self.save_retention_snapshot(avg_retention, total, below_target, false);
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
            backfilled_causes,
        })
    }

    /// The raw `VESTIGE_AUTO_CONSOLIDATE_MERGE` value, or a test's pinned
    /// value for this thread. Parsing stays in the caller so the fail-closed
    /// rule is read next to the destructive pass it guards.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub(super) fn auto_consolidate_merge_value() -> Option<String> {
        #[cfg(test)]
        if let Some(pinned) = AUTO_CONSOLIDATE_MERGE_FOR_TEST.with(|cell| cell.borrow().clone()) {
            return pinned;
        }
        std::env::var("VESTIGE_AUTO_CONSOLIDATE_MERGE").ok()
    }

    /// Auto-deduplicate similar memories during consolidation (episodic → semantic merge)
    ///
    /// Finds clusters with cosine similarity >= 0.85, keeps the strongest node,
    /// appends unique content from weaker nodes, and deletes duplicates.
    /// Honors the `VESTIGE_AUTO_CONSOLIDATE_MERGE` opt-out (unset → on) and
    /// never merges away or deletes protected (pinned) nodes (#142).
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub(super) fn auto_dedup_consolidation(&self) -> Result<i64> {
        // OPT-IN (v2.6.0, reversing the #142 opt-out): this pass concat-merges
        // near-duplicate memories and HARD-DELETES the weaker ones with no
        // reflog. Unattended destruction of user memories is opt-IN, never a
        // default: set VESTIGE_AUTO_CONSOLIDATE_MERGE=1 (or true/on/yes) to
        // enable it. Unset or any other/malformed value fails CLOSED — the
        // safe direction for a destructive gate (#142's opt-out parsed the
        // same input as fail-OPEN, so a typo destroyed data). The `dedup` MCP
        // tool remains the on-demand, previewable, reversible path and is
        // unaffected by this gate. Gate here (not the caller) so it stays
        // with the pin filter and self-protects against a future second
        // caller.
        let auto_merge = Self::auto_consolidate_merge_value()
            .map(|v| {
                let v = v.trim();
                v.eq_ignore_ascii_case("true")
                    || v.eq_ignore_ascii_case("on")
                    || v.eq_ignore_ascii_case("yes")
                    || v == "1"
            })
            .unwrap_or(false);
        if !auto_merge {
            return Ok(0);
        }

        let all_embeddings = self.get_all_embeddings()?;
        let n = all_embeddings.len();

        if !(2..=2000).contains(&n) {
            return Ok(0);
        }

        // Protected (pinned) memories must never be touched by this unattended,
        // no-audit pass — mirroring the interactive contract that a protected
        // node may only survive a merge, never be absorbed (see `plan_merge`).
        // Fetch the set ONCE here, before the per-cluster reader lock is taken:
        // both `protected_node_ids()` and `is_protected()` take their OWN reader
        // lock, so calling either inside the lock window below would self-deadlock
        // the non-reentrant Mutex. Skipping protected ids at BOTH the outer
        // (anchor) and inner (member) loops guarantees a protected node is never
        // an anchor and never a cluster member — so it can never be the keeper nor
        // land in weak_ids, and is thus never merged into and never deleted. Fails
        // SAFE via `?`: on a poisoned lock the caller's unwrap_or(0) skips the
        // merge this cycle rather than risk absorbing a pin. #142
        let protected = self.protected_node_ids()?;

        // Scope map, fetched ONCE alongside `protected` and for the same reason:
        // the per-cluster reader lock below is non-reentrant, so this cannot be
        // looked up inside the loop. This pass merges content and then HARD
        // DELETES the weak nodes, unattended and with no audit row. Without a
        // scope guard it will happily fuse two different projects' near-identical
        // notes -- e.g. the same convention worded alike but naming different
        // credentials -- and destroy one of them. Memories only ever cluster with
        // memories in their OWN scope.
        let scopes: std::collections::HashMap<String, String> = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            let mut stmt = reader.prepare(
                "SELECT id, COALESCE(NULLIF(TRIM(scope), ''), 'user') FROM knowledge_nodes",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            let mut m = std::collections::HashMap::new();
            for r in rows {
                let (id, sc) = r?;
                m.insert(id, sc);
            }
            m
        };
        let scope_of = |id: &str| -> &str { scopes.get(id).map(String::as_str).unwrap_or("user") };

        const SIMILARITY_THRESHOLD: f32 = 0.85;
        let mut merged_count = 0i64;
        let mut consumed: std::collections::HashSet<String> = std::collections::HashSet::new();

        for i in 0..n {
            if consumed.contains(&all_embeddings[i].0) || protected.contains(&all_embeddings[i].0) {
                continue;
            }

            let mut cluster: Vec<(usize, f32)> = Vec::new();

            let anchor_scope = scope_of(&all_embeddings[i].0);
            for j in (i + 1)..n {
                if consumed.contains(&all_embeddings[j].0)
                    || protected.contains(&all_embeddings[j].0)
                {
                    continue;
                }
                // Never cluster across project scopes: the merge below deletes.
                if scope_of(&all_embeddings[j].0) != anchor_scope {
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

            // Update keeper with merged content. The update result is the
            // gate for the deletions below: if the keeper never absorbed the
            // weak nodes' content, deleting them destroys it. The previous
            // `let _ =` discarded exactly that failure and deleted anyway.
            let content_preserved = if merged_content != keeper_content {
                self.update_node_content(&best_id, &merged_content).is_ok()
            } else {
                true
            };

            if content_preserved {
                // Delete weak nodes — their content verifiably lives on in
                // the keeper (or was already contained in it).
                for weak_id in &weak_ids {
                    let _ = self.delete_node(weak_id);
                    consumed.insert(weak_id.clone());
                    merged_count += 1;
                }
            } else {
                tracing::warn!(
                    keeper = %best_id,
                    weak = weak_ids.len(),
                    "auto-dedup: keeper content update failed; weak nodes kept (nothing deleted)"
                );
                for weak_id in &weak_ids {
                    consumed.insert(weak_id.clone());
                }
            }

            consumed.insert(best_id);
        }

        Ok(merged_count)
    }

    /// Restore the last meaningful interaction for memories whose most recent
    /// `last_accessed` value came from the old passive-search behavior.
    ///
    /// Pre-2.3.0 `search_hit` rows updated `last_accessed`, which also fed the
    /// recency ranker and FSRS decay. `retrieval_shown` is intentionally not
    /// included: the new event never writes node state. A passive event is
    /// logged immediately after the old update, so we repair only nodes whose
    /// timestamp is no later than their latest legacy hit. An unlogged FSRS
    /// review updates `updated_at`, making it a safe fallback before
    /// `created_at` when no explicit interaction is recorded.
    fn repair_legacy_passive_retrieval_state(&self) -> Result<i64> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let repaired = writer.execute(
            "UPDATE knowledge_nodes AS node
             SET last_accessed = MAX(
                 COALESCE(
                     (
                         SELECT MAX(explicit.accessed_at)
                         FROM memory_access_log AS explicit
                         WHERE explicit.node_id = node.id
                           AND explicit.access_type NOT IN ('search_hit', 'retrieval_shown')
                     ),
                     node.created_at
                 ),
                 node.updated_at
             )
             WHERE EXISTS (
                 SELECT 1
                 FROM memory_access_log AS passive
                 WHERE passive.node_id = node.id
                   AND passive.access_type = 'search_hit'
             )
               AND node.last_accessed <= (
                 SELECT MAX(passive.accessed_at)
                 FROM memory_access_log AS passive
                 WHERE passive.node_id = node.id
                   AND passive.access_type = 'search_hit'
             )",
            [],
        )?;
        Ok(repaired as i64)
    }

    /// Compute ACT-R base-level activation for all nodes from access history.
    /// B_i = ln(Σ t_j^(-d)) where t_j = days since j-th access, d = 0.5
    pub(super) fn compute_act_r_activations(&self) -> Result<i64> {
        const ACT_R_DECAY: f64 = 0.5;
        let now = Utc::now();

        // This also protects direct callers that compute ACT-R without using
        // the full consolidation cycle.
        self.repair_legacy_passive_retrieval_state()?;

        let node_ids: Vec<String> = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            reader
                .prepare(
                    "SELECT DISTINCT node_id FROM memory_access_log
                     WHERE access_type NOT IN ('search_hit', 'retrieval_shown')",
                )?
                .query_map([], |row| row.get(0))?
                .filter_map(warn_skipped_row("compute_act_r_activations"))
                .collect()
        };

        let mut count = 0i64;
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "compute_act_r_activations")?;

        // Discard residual activation from legacy search-hit rows as well as
        // new retrieval-only telemetry. Otherwise historical passive reads
        // would keep influencing rank after the behavior changes.
        tx.execute(
            "UPDATE knowledge_nodes SET activation = 0.0
             WHERE id NOT IN (
                SELECT DISTINCT node_id FROM memory_access_log
                WHERE access_type NOT IN ('search_hit', 'retrieval_shown')
             )",
            [],
        )?;

        if node_ids.is_empty() {
            tx.commit()?;
            return Ok(0);
        }

        for node_id in &node_ids {
            let timestamps: Vec<String> = tx
                .prepare(
                    "SELECT accessed_at FROM memory_access_log
                     WHERE node_id = ?1 AND access_type NOT IN ('search_hit', 'retrieval_shown')
                     ORDER BY accessed_at DESC
                     LIMIT 500",
                )?
                .query_map(params![node_id], |row| row.get(0))?
                .filter_map(warn_skipped_row("compute_act_r_activations"))
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

    /// Prune old access log entries (keep the last [`ACCESS_LOG_RETENTION_DAYS`]).
    /// `hygiene_snapshot` derives its "never accessed" window from the same
    /// constant; keep the two in lockstep.
    fn prune_access_log(&self) -> Result<i64> {
        let cutoff = (Utc::now() - Duration::days(ACCESS_LOG_RETENTION_DAYS)).to_rfc3339();
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
            .query_row(
                "SELECT COUNT(*) FROM memory_access_log
                 WHERE access_type NOT IN ('search_hit', 'retrieval_shown')",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        if access_count < 100 {
            return Ok(None);
        }

        let mut optimizer = FSRSOptimizer::new();

        // Most RECENT window, not the oldest. The previous `ASC LIMIT 1000`
        // trained forever on the earliest era of the log — and because the
        // 90-day log pruning slides that window, the training set drifted
        // under the optimizer's feet, producing fits that swung between
        // 0.0104 and 0.137 on the same store with no behavior change.
        let logs: Vec<(String, String, String)> = reader
            .prepare(
                "SELECT node_id, access_type, accessed_at FROM (
                     SELECT mal.node_id, mal.access_type, mal.accessed_at
                     FROM memory_access_log mal
                     WHERE mal.access_type NOT IN ('search_hit', 'retrieval_shown')
                     ORDER BY mal.accessed_at DESC
                     LIMIT 1000
                 ) ORDER BY accessed_at ASC",
            )?
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .filter_map(warn_skipped_row("optimize_w20_if_ready"))
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

                // Suppression is the strongest forgetting signal a user can
                // send; feeding it to the optimizer as a SUCCESSFUL recall
                // (the old catch-all) taught the curve that nothing is ever
                // forgotten. A reversed suppression is a correction of that
                // signal, not a recall outcome either way; score it neutral.
                let rating = match access_type.as_str() {
                    "promote" => 4,
                    "search_hit" => 3,
                    "demote" | "suppress" => 1,
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
            .filter_map(warn_skipped_row("get_retention_trend"))
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

    /// Check for auto-promote candidates: memories explicitly promoted 3+ times in 24h.
    pub fn auto_promote_frequent_access(&self) -> Result<i64> {
        let twenty_four_hours_ago = (Utc::now() - Duration::hours(24)).to_rfc3339();
        let now = Utc::now().to_rfc3339();

        // A search hit is not evidence of correctness. Only repeated explicit
        // positive feedback is eligible for this optional extra boost.
        let candidates: Vec<String> = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            let mut stmt = reader.prepare(
                "SELECT node_id, COUNT(*) as access_count
                 FROM memory_access_log
                 WHERE accessed_at >= ?1 AND access_type = 'promote'
                 GROUP BY node_id
                 HAVING access_count >= 3",
            )?;
            stmt.query_map(params![twenty_four_hours_ago], |row| row.get(0))?
                .filter_map(warn_skipped_row("auto_promote_frequent_access"))
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
}
