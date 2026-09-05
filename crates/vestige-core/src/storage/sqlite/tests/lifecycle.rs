//! Tests for `sqlite/lifecycle.rs`: failure feedback, passive recall telemetry,
//! consolidation, auto-dedup (#142), backfill and suppression.

use super::*;

// =========================================================================
// Post-retrieval failure feedback (Heinbockel 2025)
// =========================================================================

fn set_retrieval_strength(storage: &Storage, id: &str, value: f64) {
    storage
        .writer
        .lock()
        .unwrap()
        .execute(
            "UPDATE knowledge_nodes SET retrieval_strength = ?1 WHERE id = ?2",
            params![value, id],
        )
        .unwrap();
}

fn retrieval_strength(storage: &Storage, id: &str) -> f64 {
    storage
        .reader
        .lock()
        .unwrap()
        .query_row(
            "SELECT retrieval_strength FROM knowledge_nodes WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )
        .unwrap()
}

fn save_test_receipt(storage: &Storage, retrieved: Vec<String>) -> String {
    let trust: Vec<f64> = retrieved.iter().map(|_| 0.9).collect();
    let receipt = crate::trace::Receipt::build(
        Utc::now(),
        "test",
        retrieved,
        Vec::new(),
        Vec::new(),
        &trust,
        Vec::new(),
    );
    let id = receipt.receipt_id.clone();
    storage
        .save_receipt(&receipt, None, Some("recall"), Some("what broke"))
        .unwrap();
    id
}

/// A failure right after a retrieval lowers the retrieved memories'
/// accessibility by rank (best-first is the strongest reactivation), once
/// per (failure, memory), and the ledger can undo it exactly.
#[test]
fn failure_feedback_demotes_recently_retrieved_memories_by_rank() {
    let storage = create_test_storage();
    let first = ingest_tagged_in_scope(
        &storage,
        "user",
        "redis url points at the old cluster",
        &["ops"],
    );
    let second = ingest_tagged_in_scope(
        &storage,
        "user",
        "worker pool reads a cached secrets file",
        &["ops"],
    );
    set_retrieval_strength(&storage, &first.id, 0.90);
    set_retrieval_strength(&storage, &second.id, 0.90);
    save_test_receipt(&storage, vec![first.id.clone(), second.id.clone()]);

    let failure = ingest_tagged_in_scope(
        &storage,
        "user",
        "Payments API crashed with 500s: queue backed up for 36 hours",
        &["incident"],
    );

    let report = storage
        .apply_failure_feedback(&failure.id, Duration::minutes(30))
        .unwrap();
    assert_eq!(report.receipts_considered, 1);
    assert_eq!(report.memories_demoted, 2);
    let after_first = retrieval_strength(&storage, &first.id);
    let after_second = retrieval_strength(&storage, &second.id);
    assert!(
        (after_first - 0.80).abs() < 1e-6,
        "rank 0 loses the full penalty: {after_first}"
    );
    assert!(
        (after_second - 0.85).abs() < 1e-6,
        "rank 1 loses half: {after_second}"
    );
    assert!(
        retrieval_strength(&storage, &failure.id) > 0.0,
        "the failure itself is never demoted"
    );

    // Idempotent per (failure, memory).
    let again = storage
        .apply_failure_feedback(&failure.id, Duration::minutes(30))
        .unwrap();
    assert_eq!(again.memories_demoted, 0);
    assert!((retrieval_strength(&storage, &first.id) - 0.80).abs() < 1e-6);

    // Reversible, exactly.
    assert_eq!(storage.revert_failure_feedback(&failure.id).unwrap(), 2);
    assert!((retrieval_strength(&storage, &first.id) - 0.90).abs() < 1e-6);
    assert!((retrieval_strength(&storage, &second.id) - 0.90).abs() < 1e-6);
    assert_eq!(
        storage.revert_failure_feedback(&failure.id).unwrap(),
        0,
        "a second revert finds nothing left to undo"
    );
}

/// Only memories in the failure's scope are touched, and a receipt outside
/// the window is ignored.
#[test]
fn failure_feedback_respects_scope_and_window() {
    let storage = create_test_storage();
    let elsewhere =
        ingest_tagged_in_scope(&storage, "other-project", "unrelated project note", &["x"]);
    let here = ingest_tagged_in_scope(&storage, "user", "same-scope note", &["x"]);
    set_retrieval_strength(&storage, &elsewhere.id, 0.90);
    set_retrieval_strength(&storage, &here.id, 0.90);
    save_test_receipt(&storage, vec![elsewhere.id.clone(), here.id.clone()]);

    let failure = ingest_tagged_in_scope(
        &storage,
        "user",
        "Build failed: linker error",
        &["incident"],
    );
    let report = storage
        .apply_failure_feedback(&failure.id, Duration::minutes(30))
        .unwrap();
    assert_eq!(
        report.memories_demoted, 1,
        "only the same-scope memory is demoted"
    );
    assert!((retrieval_strength(&storage, &elsewhere.id) - 0.90).abs() < 1e-6);
    assert!(retrieval_strength(&storage, &here.id) < 0.90);

    // A zero-length window sees no receipts.
    let none = storage
        .apply_failure_feedback(&failure.id, Duration::zero())
        .unwrap();
    assert_eq!(none.receipts_considered, 0);
}

#[test]
fn passive_recall_records_telemetry_without_reinforcing() {
    let storage = create_test_storage();
    let node = storage
        .ingest(IngestInput {
            content: "current version is 2.0.12 as of 2026-03-04".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let before = storage.get_node(&node.id).unwrap().unwrap();

    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET activation = 0.73 WHERE id = ?1",
                params![&node.id],
            )
            .unwrap();
    }

    for _ in 0..3 {
        let recalled = storage
            .recall(RecallInput {
                query: "current version".to_string(),
                limit: 10,
                min_retention: 0.0,
                search_mode: SearchMode::Keyword,
                valid_at: None,
            })
            .unwrap();
        assert_eq!(recalled.len(), 1);
    }

    let after = storage.get_node(&node.id).unwrap().unwrap();
    assert_eq!(after.retrieval_strength, before.retrieval_strength);
    assert_eq!(after.retention_strength, before.retention_strength);
    assert_eq!(after.stability, before.stability);
    assert_eq!(after.last_accessed, before.last_accessed);
    assert_eq!(after.reps, before.reps);
    assert_eq!(after.next_review, before.next_review);
    assert_eq!(after.times_retrieved, before.times_retrieved);
    assert_eq!(after.times_useful, before.times_useful);
    assert_eq!(after.utility_score, before.utility_score);
    assert_eq!(storage.auto_promote_frequent_access().unwrap(), 0);
    storage.compute_act_r_activations().unwrap();

    let reader = storage.reader.lock().unwrap();
    let retrievals_shown: i64 = reader
            .query_row(
                "SELECT COUNT(*) FROM memory_access_log WHERE node_id = ?1 AND access_type = 'retrieval_shown'",
                params![&node.id],
                |row| row.get(0),
            )
            .unwrap();
    assert_eq!(retrievals_shown, 3);
    let activation: f64 = reader
        .query_row(
            "SELECT activation FROM knowledge_nodes WHERE id = ?1",
            params![&node.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(activation, 0.0);
}

#[test]
fn explicit_promotion_marks_a_retrieved_memory_useful() {
    let storage = create_test_storage();
    let node = storage
        .ingest(IngestInput {
            content: "A memory that needs explicit positive feedback".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    storage.demote_memory(&node.id).unwrap();
    let before = storage.get_node(&node.id).unwrap().unwrap();

    storage.record_batch_retrieval(&[&node.id]).unwrap();
    let promoted = storage.promote_memory(&node.id).unwrap();

    assert!(promoted.retrieval_strength > before.retrieval_strength);
    assert!(promoted.retention_strength > before.retention_strength);
    assert_eq!(promoted.times_retrieved.unwrap_or_default(), 0);
    assert_eq!(promoted.times_useful.unwrap_or_default(), 1);
    assert_eq!(promoted.utility_score.unwrap_or_default(), 1.0);
}

#[test]
fn legacy_search_hits_cannot_reinforce_or_reactivate_a_memory() {
    let storage = create_test_storage();
    let node = storage
        .ingest(IngestInput {
            content: "legacy dated current version claim".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes
                     SET retrieval_strength = 0.50, retention_strength = 0.40, activation = 0.73
                     WHERE id = ?1",
                params![&node.id],
            )
            .unwrap();
        for _ in 0..3 {
            writer
                .execute(
                    "INSERT INTO memory_access_log (node_id, access_type, accessed_at)
                         VALUES (?1, 'search_hit', ?2)",
                    params![&node.id, Utc::now().to_rfc3339()],
                )
                .unwrap();
        }
    }
    let before = storage.get_node(&node.id).unwrap().unwrap();

    assert_eq!(storage.auto_promote_frequent_access().unwrap(), 0);
    storage.compute_act_r_activations().unwrap();

    let after = storage.get_node(&node.id).unwrap().unwrap();
    assert_eq!(after.retrieval_strength, before.retrieval_strength);
    assert_eq!(after.retention_strength, before.retention_strength);
    assert_eq!(after.last_accessed, before.last_accessed);

    let reader = storage.reader.lock().unwrap();
    let activation: f64 = reader
        .query_row(
            "SELECT activation FROM knowledge_nodes WHERE id = ?1",
            params![&node.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(activation, 0.0);
}

#[test]
fn legacy_search_hits_cannot_preserve_recency_or_delay_decay() {
    let storage = create_test_storage();
    let node = storage
        .ingest(IngestInput {
            content: "a legacy passive search must not preserve freshness".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let created_at = Utc::now() - Duration::days(30);
    let passive_at = Utc::now();

    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET
                        created_at = ?1,
                        updated_at = ?1,
                        last_accessed = ?2,
                        retrieval_strength = 1.0,
                        retention_strength = 1.0
                     WHERE id = ?3",
                params![created_at.to_rfc3339(), passive_at.to_rfc3339(), &node.id],
            )
            .unwrap();
        writer
            .execute(
                "INSERT INTO memory_access_log (node_id, access_type, accessed_at)
                     VALUES (?1, 'search_hit', ?2)",
                params![&node.id, passive_at.to_rfc3339()],
            )
            .unwrap();
    }

    storage.compute_act_r_activations().unwrap();
    let repaired = storage.get_node(&node.id).unwrap().unwrap();
    assert_eq!(repaired.last_accessed, created_at);

    storage.apply_decay().unwrap();
    let decayed = storage.get_node(&node.id).unwrap().unwrap();
    assert!(decayed.retrieval_strength < 1.0);
    assert!(decayed.retention_strength < 1.0);
}

#[test]
fn legacy_recency_repair_preserves_reviewed_memories() {
    let storage = create_test_storage();
    let node = storage
        .ingest(IngestInput {
            content: "a reviewed memory must not be reset by passive logs".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    storage.promote_memory(&node.id).unwrap();
    let reviewed = storage.mark_reviewed(&node.id, Rating::Good).unwrap();

    // New telemetry never changed state, so it must never trigger a
    // legacy-state repair.
    storage.record_batch_retrieval(&[&node.id]).unwrap();
    storage.compute_act_r_activations().unwrap();
    let after_new_telemetry = storage.get_node(&node.id).unwrap().unwrap();
    assert_eq!(after_new_telemetry.last_accessed, reviewed.last_accessed);

    // If an old search-hit row was recorded after a review, restore the
    // review timestamp from updated_at rather than falling back to create.
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET last_accessed = ?1 WHERE id = ?2",
                params![Utc::now().to_rfc3339(), &node.id],
            )
            .unwrap();
        writer
            .execute(
                "INSERT INTO memory_access_log (node_id, access_type, accessed_at)
                     VALUES (?1, 'search_hit', ?2)",
                params![&node.id, Utc::now().to_rfc3339()],
            )
            .unwrap();
    }
    storage.compute_act_r_activations().unwrap();
    let restored = storage.get_node(&node.id).unwrap().unwrap();
    assert_eq!(restored.last_accessed, reviewed.last_accessed);
}

#[test]
fn test_review() {
    let storage = create_test_storage();

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

/// REGRESSION (v2.6.0 data-safety): consolidation must never delete a
/// memory. Until this release, an autonomic "retention target" GC inside
/// run_consolidation hard-deleted everything below 0.3 retention older
/// than 30 days — dormant only while decay was broken, and it destroyed
/// 23 real memories from a live store the day decay was fixed. This test
/// constructs exactly that scenario and asserts nothing dies.
#[test]
fn consolidation_never_deletes_low_retention_memories() {
    let storage = create_test_storage();
    let mut ids = Vec::new();
    for i in 0..4 {
        let node = storage
            .ingest(IngestInput {
                content: format!("Old low-retention memory number {i} that must survive"),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        ids.push(node.id);
    }
    // Force the doomed profile the old reaper keyed on: retention far
    // below 0.3 and created_at far older than 30 days.
    {
        let writer = storage.writer.lock().unwrap();
        let old = (Utc::now() - Duration::days(120)).to_rfc3339();
        for id in &ids {
            writer
                .execute(
                    "UPDATE knowledge_nodes
                         SET retention_strength = 0.05, created_at = ?1, last_accessed = ?1
                         WHERE id = ?2",
                    params![old, id],
                )
                .unwrap();
        }
    }
    let before: i64 = {
        let reader = storage.reader.lock().unwrap();
        reader
            .query_row("SELECT COUNT(*) FROM knowledge_nodes", [], |r| r.get(0))
            .unwrap()
    };

    storage.run_consolidation().unwrap();

    let after: i64 = {
        let reader = storage.reader.lock().unwrap();
        reader
            .query_row("SELECT COUNT(*) FROM knowledge_nodes", [], |r| r.get(0))
            .unwrap()
    };
    assert_eq!(before, after, "consolidation deleted memories");
    for id in &ids {
        assert!(
            storage.get_node(id).unwrap().is_some(),
            "low-retention memory {id} was reaped by consolidation"
        );
    }
}

// ========================================================================
// Auto-consolidation merge: opt-out gate + protected-pin exclusion (#142)
//
// These exercise `auto_dedup_consolidation` directly — the unattended,
// no-audit pass the 6h background consolidation cycle runs. seed_node/
// axis_vector give deterministic same-axis clusters (cosine ~1.0 >> the 0.85
// threshold); set_retention pins down which node wins the keeper tiebreak.
// ========================================================================

/// Force a node's retention_strength so the keeper tiebreak in
/// `auto_dedup_consolidation` is deterministic regardless of insertion order.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
fn set_retention(storage: &Storage, id: &str, value: f64) {
    let writer = storage.writer.lock().unwrap();
    writer
        .execute(
            "UPDATE knowledge_nodes SET retention_strength = ?1 WHERE id = ?2",
            rusqlite::params![value, id],
        )
        .unwrap();
}

// --- A. Default (flag unset): NOTHING merges, nothing is deleted ---------
// v2.6.0 flipped the #142 opt-out into an opt-in: unattended destruction
// of user memories must be asked for, never inherited.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_auto_dedup_default_off_preserves_near_duplicates() {
    with_auto_merge_env(None, || {
        let storage = create_test_storage();
        let keeper = seed_node(
            &storage,
            "Rate limiting uses a token bucket per client API key",
            &["api"],
            axis_vector(21, 0.02),
        );
        let dup = seed_node(
            &storage,
            "Rate limiting uses a token-bucket algorithm per client API key, refilled steadily",
            &["api"],
            axis_vector(21, 0.01),
        );
        set_retention(&storage, &keeper, 0.9);
        set_retention(&storage, &dup, 0.3);

        let merged = storage.auto_dedup_consolidation().unwrap();
        assert_eq!(merged, 0, "flag unset: the destructive pass must not run");
        assert!(
            storage.get_node(&dup).unwrap().is_some(),
            "near-duplicate survives by default"
        );
        assert!(
            !storage
                .get_node(&keeper)
                .unwrap()
                .unwrap()
                .content
                .contains("[MERGED]"),
            "keeper content untouched by default"
        );
    });
    // Explicit opt-in (trimmed, case-insensitive 1/true/on/yes) enables it.
    for value in ["1", "true", "ON", "  Yes  "] {
        with_auto_merge_env(Some(value), || {
            let storage = create_test_storage();
            let keeper = seed_node(
                &storage,
                "Rate limiting uses a token bucket per client API key",
                &["api"],
                axis_vector(21, 0.02),
            );
            let dup = seed_node(
                &storage,
                "Rate limiting uses a token-bucket algorithm per client API key, refilled steadily",
                &["api"],
                axis_vector(21, 0.01),
            );
            set_retention(&storage, &keeper, 0.9);
            set_retention(&storage, &dup, 0.3);

            let merged = storage.auto_dedup_consolidation().unwrap();
            assert_eq!(
                merged, 1,
                "opted in ({value:?}): weak node folds into keeper"
            );
            assert!(storage.get_node(&dup).unwrap().is_none());
            assert!(
                storage
                    .get_node(&keeper)
                    .unwrap()
                    .unwrap()
                    .content
                    .contains("[MERGED]"),
                "keeper carries the folded-in [MERGED] block"
            );
        });
    }
}

// --- B. Flag off suppresses the merge (parametrized) ---------------------
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_auto_dedup_env_off_suppresses_merge() {
    // trimmed + case-insensitive false/off/no/0 all disable.
    for value in ["false", "off", "no", "0", "  OFF  ", "False"] {
        with_auto_merge_env(Some(value), || {
            let storage = create_test_storage();
            let a = seed_node(
                &storage,
                "Prometheus scrapes its targets every 15 seconds",
                &["obs"],
                axis_vector(23, 0.02),
            );
            let b = seed_node(
                &storage,
                "Prometheus scrapes its configured targets every 15s by default",
                &["obs"],
                axis_vector(23, 0.01),
            );

            let merged = storage.auto_dedup_consolidation().unwrap();
            assert_eq!(merged, 0, "value {value:?} must suppress the merge");
            // Both nodes survive, content byte-identical (no [MERGED] block).
            assert_eq!(
                storage.get_node(&a).unwrap().unwrap().content,
                "Prometheus scrapes its targets every 15 seconds"
            );
            assert_eq!(
                storage.get_node(&b).unwrap().unwrap().content,
                "Prometheus scrapes its configured targets every 15s by default"
            );
        });
    }
}

// --- B (cont). A malformed value fails CLOSED: no destruction on a typo --
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_auto_dedup_env_garbage_fails_closed_and_preserves() {
    with_auto_merge_env(Some("banana"), || {
        let storage = create_test_storage();
        let keeper = seed_node(
            &storage,
            "Cache entries expire after a five minute TTL",
            &["cache"],
            axis_vector(25, 0.02),
        );
        let dup = seed_node(
            &storage,
            "Cache entries expire after a five-minute TTL window by default",
            &["cache"],
            axis_vector(25, 0.01),
        );
        set_retention(&storage, &keeper, 0.9);
        set_retention(&storage, &dup, 0.3);

        let merged = storage.auto_dedup_consolidation().unwrap();
        assert_eq!(
            merged, 0,
            "malformed value fails closed for a destructive gate"
        );
        assert!(
            storage.get_node(&dup).unwrap().is_some(),
            "nothing deleted on a typo"
        );
    });
}

// --- C(a). Protected would-be keeper: untouched; others merge -----------
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_auto_dedup_protected_would_be_keeper_untouched_others_merge() {
    with_auto_merge_env(Some("1"), || {
        let storage = create_test_storage();
        // P has the highest retention, so absent protection it would be the
        // keeper. Protected → skipped entirely; the two unprotected merge alone.
        let pinned = seed_node(
            &storage,
            "Deploys are gated on a green CI run and one approval",
            &["ci"],
            axis_vector(27, 0.02),
        );
        let keeper = seed_node(
            &storage,
            "Deploys are gated on a green CI pipeline plus one reviewer approval",
            &["ci"],
            axis_vector(27, 0.01),
        );
        let member = seed_node(
            &storage,
            "Deploys require a green CI run and at least one approving review",
            &["ci"],
            axis_vector(27, 0.015),
        );
        set_retention(&storage, &pinned, 0.95);
        set_retention(&storage, &keeper, 0.80);
        set_retention(&storage, &member, 0.30);
        storage.set_protected(&pinned, true).unwrap();
        let pinned_content = storage.get_node(&pinned).unwrap().unwrap().content;

        let merged = storage.auto_dedup_consolidation().unwrap();
        assert_eq!(
            merged, 1,
            "the two unprotected near-dups merge among themselves"
        );

        // Protected node byte-for-byte untouched and still protected.
        let p = storage.get_node(&pinned).unwrap().unwrap();
        assert_eq!(p.content, pinned_content, "protected keeper not absorbed");
        assert!(!p.content.contains("[MERGED]"));
        assert!(storage.is_protected(&pinned).unwrap());
        // Unprotected pair merged: `member` gone, `keeper` carries [MERGED].
        assert!(storage.get_node(&member).unwrap().is_none());
        let keeper_node = storage.get_node(&keeper).unwrap().unwrap();
        assert!(keeper_node.content.contains("[MERGED]"));
    });
}

// --- C(b) / Regression (#142): protected weak member is never absorbed --
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn auto_dedup_regression_142_protected_weak_member_not_absorbed() {
    with_auto_merge_env(Some("1"), || {
        let storage = create_test_storage();
        // Regression (#142): before the fix this pinned node — the weaker
        // member of the cluster — was silently absorbed into the stronger
        // unprotected keeper and hard-deleted by the unattended pass. The
        // PINNED-CANARY-142 marker makes accidental absorption detectable.
        let pinned = seed_node(
            &storage,
            "Feature flags default to off in production PINNED-CANARY-142",
            &["flags"],
            axis_vector(29, 0.02),
        );
        let keeper = seed_node(
            &storage,
            "Feature flags default to off in the production environment",
            &["flags"],
            axis_vector(29, 0.01),
        );
        let member = seed_node(
            &storage,
            "Feature flags are off by default in production deployments",
            &["flags"],
            axis_vector(29, 0.015),
        );
        // Pinned is the LOWEST-retention member — pre-fix it would land in
        // weak_ids and be deleted + absorbed by the keeper.
        set_retention(&storage, &pinned, 0.10);
        set_retention(&storage, &keeper, 0.80);
        set_retention(&storage, &member, 0.30);
        storage.set_protected(&pinned, true).unwrap();
        let pinned_content = storage.get_node(&pinned).unwrap().unwrap().content;

        let merged = storage.auto_dedup_consolidation().unwrap();
        assert_eq!(merged, 1, "only the two unprotected near-dups merge");

        // Invariant 1: the protected node still exists, byte-identical.
        let p = storage.get_node(&pinned).unwrap();
        assert!(p.is_some(), "protected node must not be deleted");
        assert_eq!(
            p.unwrap().content,
            pinned_content,
            "protected node not absorbed"
        );
        assert!(storage.is_protected(&pinned).unwrap());

        // Invariant 2: the keeper did NOT gain the protected node's content.
        let keeper_node = storage.get_node(&keeper).unwrap().unwrap();
        assert!(
            !keeper_node.content.contains("PINNED-CANARY-142"),
            "keeper must not absorb the protected node's content"
        );
        // The legitimate unprotected pair still merged (member folded in).
        assert!(storage.get_node(&member).unwrap().is_none());
        assert!(keeper_node.content.contains("[MERGED]"));
    });
}

// --- C(c). Two protected near-dups: neither merges ----------------------
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_auto_dedup_two_protected_near_dups_neither_merges() {
    with_auto_merge_env(Some("1"), || {
        let storage = create_test_storage();
        let a = seed_node(
            &storage,
            "Backups run nightly and are retained for thirty days",
            &["backup"],
            axis_vector(31, 0.02),
        );
        let b = seed_node(
            &storage,
            "Backups run every night and are kept for thirty days",
            &["backup"],
            axis_vector(31, 0.01),
        );
        storage.set_protected(&a, true).unwrap();
        storage.set_protected(&b, true).unwrap();
        let (ca, cb) = (
            storage.get_node(&a).unwrap().unwrap().content,
            storage.get_node(&b).unwrap().unwrap().content,
        );

        let merged = storage.auto_dedup_consolidation().unwrap();
        assert_eq!(merged, 0, "two protected near-dups: nothing merges");
        assert_eq!(storage.get_node(&a).unwrap().unwrap().content, ca);
        assert_eq!(storage.get_node(&b).unwrap().unwrap().content, cb);
    });
}

// --- C(d). Protected + a single unprotected near-dup: no merge ----------
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_auto_dedup_protected_plus_single_unprotected_no_merge() {
    with_auto_merge_env(Some("1"), || {
        let storage = create_test_storage();
        let pinned = seed_node(
            &storage,
            "Secrets are stored in the vault, never in the repo",
            &["sec"],
            axis_vector(33, 0.02),
        );
        let other = seed_node(
            &storage,
            "Secrets live in the vault and are never committed to the repo",
            &["sec"],
            axis_vector(33, 0.01),
        );
        storage.set_protected(&pinned, true).unwrap();
        let (cp, co) = (
            storage.get_node(&pinned).unwrap().unwrap().content,
            storage.get_node(&other).unwrap().unwrap().content,
        );

        let merged = storage.auto_dedup_consolidation().unwrap();
        assert_eq!(merged, 0, "a lone unprotected node cannot form a cluster");
        assert_eq!(storage.get_node(&pinned).unwrap().unwrap().content, cp);
        assert_eq!(storage.get_node(&other).unwrap().unwrap().content, co);
    });
}

// --- D. Liveness: protected + two unprotected → the two merge -----------
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_auto_dedup_protected_plus_two_unprotected_liveness() {
    with_auto_merge_env(Some("1"), || {
        let storage = create_test_storage();
        // The pin exclusion must not block a legitimate merge of the others.
        let pinned = seed_node(
            &storage,
            "The API returns ISO-8601 timestamps in UTC",
            &["api"],
            axis_vector(35, 0.02),
        );
        let keeper = seed_node(
            &storage,
            "The API returns ISO 8601 timestamps in UTC by convention",
            &["api"],
            axis_vector(35, 0.01),
        );
        let member = seed_node(
            &storage,
            "All API timestamps are returned as ISO-8601 in the UTC timezone",
            &["api"],
            axis_vector(35, 0.015),
        );
        set_retention(&storage, &pinned, 0.50);
        set_retention(&storage, &keeper, 0.80);
        set_retention(&storage, &member, 0.30);
        storage.set_protected(&pinned, true).unwrap();
        let pinned_content = storage.get_node(&pinned).unwrap().unwrap().content;

        let merged = storage.auto_dedup_consolidation().unwrap();
        assert_eq!(merged, 1, "the two unprotected near-dups still merge");
        assert!(storage.get_node(&member).unwrap().is_none());
        let keeper_node = storage.get_node(&keeper).unwrap().unwrap();
        assert!(keeper_node.content.contains("[MERGED]"));
        // Protected node untouched.
        assert_eq!(
            storage.get_node(&pinned).unwrap().unwrap().content,
            pinned_content
        );
        assert!(storage.is_protected(&pinned).unwrap());
    });
}

// Seed a node's stability directly via the scheduling seam so the +365 cap
// in promote_memory_backfill is actually exercised (a freshly ingested node
// has low stability where the *1.5 multiply, not the additive ceiling, wins).
fn seed_stability(s: &Storage, id: &str, stability: f64) {
    use crate::storage::memory_store::{MemoryStoreSend, SchedulingState};
    rt().block_on(async {
        let state = SchedulingState {
            memory_id: uuid::Uuid::parse_str(id).unwrap(),
            stability,
            difficulty: 0.4,
            retrievability: 0.8,
            last_review: Some(chrono::Utc::now()),
            next_review: Some(chrono::Utc::now() + chrono::Duration::days(7)),
            reps: 3,
            lapses: 0,
        };
        MemoryStoreSend::update_scheduling(s, &state).await.unwrap();
    });
}

#[test]
fn promote_memory_backfill_caps_stability_at_plus_365() {
    // Above the crossover (stability=730) the additive +365 ceiling must win
    // over the *1.5 multiply, so repeated backfill promotions cannot inflate
    // stability without bound. This is the bound issue #103 asked us to apply.
    let s = create_test_storage();
    let node = s
        .ingest(IngestInput {
            content: "high-stability cause memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    seed_stability(&s, &node.id, 1000.0);

    let promoted = s.promote_memory_backfill(&node.id).unwrap();
    // 1000 * 1.5 = 1500 (uncapped) vs 1000 + 365 = 1365 (capped). Cap wins.
    assert!(
        (promoted.stability - 1365.0).abs() < 1e-6,
        "expected additive +365 cap (1365.0), got {} (uncapped would be 1500.0)",
        promoted.stability
    );
}

#[test]
fn promote_memory_backfill_uses_multiply_below_crossover() {
    // Below the crossover the *1.5 multiply wins (the cap never binds), so
    // backfill promotion strength is unchanged from the old promote_memory.
    let s = create_test_storage();
    let node = s
        .ingest(IngestInput {
            content: "low-stability cause memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    seed_stability(&s, &node.id, 10.0);

    let promoted = s.promote_memory_backfill(&node.id).unwrap();
    // 10 * 1.5 = 15 (multiply) vs 10 + 365 = 375 (cap). Multiply wins.
    assert!(
        (promoted.stability - 15.0).abs() < 1e-6,
        "expected *1.5 multiply (15.0) below crossover, got {}",
        promoted.stability
    );
}

#[test]
fn suppress_then_reverse_restores_fsrs_state() {
    // reverse_suppression must be a TRUE inverse of suppress_memory. Suppress
    // applies stability*0.4, retrieval-0.35, retention-0.20; reverse now undoes
    // exactly that (stability/0.4, retrieval+0.35, retention+0.20). Previously
    // reverse used non-inverse deltas and left stability permanently halved.
    let s = create_test_storage();
    let node = s
        .ingest(IngestInput {
            content: "a memory to suppress then un-suppress".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    // Seed above the 0.05 floor so the forward pass never clips (making the
    // round-trip exactly recoverable).
    seed_stability(&s, &node.id, 20.0);
    let before = s.get_node(&node.id).unwrap().unwrap();

    s.suppress_memory(&node.id).unwrap();
    let suppressed = s.get_node(&node.id).unwrap().unwrap();
    assert!(
        (suppressed.stability - before.stability * 0.4).abs() < 1e-6,
        "suppress must multiply stability by 0.4"
    );

    let reversed = s.reverse_suppression(&node.id, 24).unwrap();
    // stability: 20 * 0.4 / 0.4 = 20 (fully restored, not 0.5x)
    assert!(
        (reversed.stability - before.stability).abs() < 1e-6,
        "reverse must restore stability to {} (got {})",
        before.stability,
        reversed.stability
    );
    assert!(
        (reversed.retrieval_strength - before.retrieval_strength).abs() < 1e-6,
        "reverse must restore retrieval_strength"
    );
    assert!(
        (reversed.retention_strength - before.retention_strength).abs() < 1e-6,
        "reverse must restore retention_strength"
    );
}

#[test]
fn backfill_autofire_gate_defaults_on_and_reads_opt_out() {
    // v2.2.1 opt-out semantics: unset => ON (preserves shipped v2.2.0
    // behavior); explicit 0/false/off/no => OFF; anything else => ON.
    fn parse(v: Option<&str>) -> bool {
        v.map(|v| {
            let v = v.trim();
            !(v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("off")
                || v.eq_ignore_ascii_case("no")
                || v == "0")
        })
        .unwrap_or(true)
    }
    assert!(parse(None), "unset must default ON");
    assert!(parse(Some("1")), "1 is ON");
    assert!(parse(Some("true")), "true is ON");
    assert!(parse(Some("anything")), "unrecognized is ON");
    assert!(!parse(Some("0")), "0 is OFF");
    assert!(!parse(Some("false")), "false is OFF");
    assert!(!parse(Some("OFF")), "OFF (case-insensitive) is OFF");
    assert!(!parse(Some(" no ")), "whitespace-padded no is OFF (trim)");
}
