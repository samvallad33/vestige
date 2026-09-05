//! Tests for `sqlite/merge.rs`: merge and supersede plans, apply, undo and
//! protection.

use super::*;

// ========================================================================
// Merge / Supersede controls (Phase 3 — v2.1.25)
//
// These exercise the full lifecycle without the live embedding model by
// seeding the `node_embeddings` table directly with the ACTIVE model name,
// so `get_all_embeddings` / `get_node_embedding` accept them.
// ========================================================================
// =========================================================================
// #181: the in-process vector index follows writes made by peer processes
// =========================================================================

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_merge_candidates_threshold_classification() {
    let storage = create_test_storage();
    // Two near-identical (same axis) — should be offered as a candidate.
    let a = seed_node(
        &storage,
        "Use tokio runtime for async Rust services",
        &["rust", "async"],
        axis_vector(3, 0.02),
    );
    let b = seed_node(
        &storage,
        "Use the tokio runtime for async Rust services",
        &["rust", "async"],
        axis_vector(3, 0.01),
    );
    // One unrelated (different axis) — must not join the cluster.
    let _c = seed_node(
        &storage,
        "Prefer postgres for relational data",
        &["db"],
        axis_vector(200, 0.0),
    );

    let policy = MergePolicy::default();
    let candidates = storage.merge_candidates(policy, 20, &[]).unwrap();
    assert_eq!(candidates.len(), 1, "exactly one duplicate cluster");
    let cluster = &candidates[0];
    assert_eq!(cluster.member_ids.len(), 2);
    assert!(cluster.member_ids.contains(&a));
    assert!(cluster.member_ids.contains(&b));
    assert!(
        cluster.confidence >= policy.possible_threshold,
        "confidence above possible threshold"
    );
    assert!(!cluster.has_protected_member);
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_plan_merge_is_preview_only_no_mutation() {
    let storage = create_test_storage();
    let a = seed_node(
        &storage,
        "Fact A about caching",
        &["perf"],
        axis_vector(5, 0.02),
    );
    let b = seed_node(
        &storage,
        "Fact A about caching, expanded",
        &["perf", "cache"],
        axis_vector(5, 0.01),
    );

    let plan = storage
        .plan_merge(&[a.clone(), b.clone()], None, MergePolicy::default())
        .unwrap();

    // Plan diff is populated...
    assert!(plan.result_content.contains("Fact A about caching"));
    assert!(plan.result_tags.contains(&"cache".to_string()));
    assert_eq!(plan.invalidated_ids.len(), 1);

    // ...but NOTHING changed: both nodes still valid, content untouched.
    let na = storage.get_node(&a).unwrap().unwrap();
    let nb = storage.get_node(&b).unwrap().unwrap();
    assert_eq!(na.content, "Fact A about caching");
    assert_eq!(nb.content, "Fact A about caching, expanded");
    let (vu_a, sb_a) = storage.read_bitemporal(&a).unwrap();
    let (vu_b, sb_b) = storage.read_bitemporal(&b).unwrap();
    assert!(vu_a.is_none() && sb_a.is_none());
    assert!(vu_b.is_none() && sb_b.is_none());

    // Plan persisted as pending.
    assert_eq!(
        storage.plan_status(&plan.id).unwrap().as_deref(),
        Some("pending")
    );
}

/// #180: `apply_plan` used to mutate through helpers that each committed on
/// their own and only afterwards insert the undo row, and its plan-status
/// check was not atomic with those mutations. Two MCP server processes
/// sharing one database file could both pass the check and both apply, and
/// a failure between the survivor rewrite and the undo insert left the
/// survivor overwritten with no way back. The whole apply is one IMMEDIATE
/// transaction now, so the plan applies exactly once no matter how many
/// callers race it, and every mutation is covered by an undo row.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn concurrent_apply_of_one_plan_applies_it_exactly_once() {
    let storage = std::sync::Arc::new(create_test_storage());
    let survivor = seed_node(
        &storage,
        "Canonical race note",
        &["r"],
        axis_vector(5, 0.02),
    );
    let absorbed = seed_node(
        &storage,
        "Detail to absorb",
        &["r", "s"],
        axis_vector(5, 0.01),
    );

    let plan = storage
        .plan_merge(
            &[survivor.clone(), absorbed.clone()],
            Some(&survivor),
            MergePolicy::default(),
        )
        .unwrap();

    let racers: Vec<_> = (0..2)
        .map(|_| {
            let storage = std::sync::Arc::clone(&storage);
            let plan_id = plan.id.clone();
            std::thread::spawn(move || storage.apply_plan(&plan_id, true).is_ok())
        })
        .collect();
    let wins = racers
        .into_iter()
        .filter_map(|handle| handle.join().ok())
        .filter(|applied| *applied)
        .count();

    assert_eq!(
        wins, 1,
        "exactly one racer may apply the plan; the other must be refused"
    );

    // Exactly one reflog row, so the mutation that happened is reversible
    // and did not happen twice.
    let ops: i64 = {
        let reader = storage.reader.lock().unwrap();
        reader
            .query_row(
                "SELECT COUNT(*) FROM merge_operations WHERE plan_id = ?1 AND status = 'applied'",
                params![plan.id],
                |row| row.get(0),
            )
            .unwrap()
    };
    assert_eq!(ops, 1, "one apply must leave exactly one undo row");

    // And the undo row actually carries what it needs to reverse.
    let payload: String = {
        let reader = storage.reader.lock().unwrap();
        reader
            .query_row(
                "SELECT undo_payload FROM merge_operations WHERE plan_id = ?1",
                params![plan.id],
                |row| row.get(0),
            )
            .unwrap()
    };
    let undo: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert!(
        undo.get("survivor_prev_content").is_some(),
        "undo row must snapshot the survivor's pre-merge content: {undo}"
    );
    assert_eq!(
        storage.plan_status(&plan.id).unwrap().as_deref(),
        Some("applied")
    );
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_apply_then_undo_merge_is_reversible() {
    let storage = create_test_storage();
    let survivor = seed_node(
        &storage,
        "Keep this canonical note",
        &["x"],
        axis_vector(7, 0.02),
    );
    let absorbed = seed_node(
        &storage,
        "Extra detail to fold in",
        &["x", "y"],
        axis_vector(7, 0.01),
    );

    let plan = storage
        .plan_merge(
            &[survivor.clone(), absorbed.clone()],
            Some(&survivor),
            MergePolicy::default(),
        )
        .unwrap();
    let op = storage.apply_plan(&plan.id, true).unwrap();
    assert_eq!(op.op_type, "merge");

    // After apply: survivor content merged, absorbed bitemporally invalidated
    // but STILL QUERYABLE (never deleted).
    let surv = storage.get_node(&survivor).unwrap().unwrap();
    assert!(surv.content.contains("Keep this canonical note"));
    assert!(surv.content.contains("Extra detail to fold in"));
    assert!(surv.tags.contains(&"y".to_string()));

    let (vu, sb) = storage.read_bitemporal(&absorbed).unwrap();
    assert!(vu.is_some(), "absorbed node stamped valid_until");
    assert_eq!(sb.as_deref(), Some(survivor.as_str()));
    // Old node is still fully retrievable for audit.
    assert!(
        storage.get_node(&absorbed).unwrap().is_some(),
        "superseded node remains queryable"
    );
    assert!(storage.superseded_node_ids().unwrap().contains(&absorbed));

    // Undo restores everything.
    let undo = storage.merge_undo(&op.id).unwrap();
    assert_eq!(undo.op_type, "undo");
    let surv_after = storage.get_node(&survivor).unwrap().unwrap();
    assert_eq!(surv_after.content, "Keep this canonical note");
    let (vu2, sb2) = storage.read_bitemporal(&absorbed).unwrap();
    assert!(
        vu2.is_none() && sb2.is_none(),
        "invalidation cleared on undo"
    );
    assert!(!storage.superseded_node_ids().unwrap().contains(&absorbed));

    // The original op is now marked reverted; double-undo is rejected.
    assert!(storage.merge_undo(&op.id).is_err());
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_supersede_invalidates_old_but_keeps_it_queryable() {
    let storage = create_test_storage();
    let old = seed_node(&storage, "LR should be 1e-4", &["ml"], axis_vector(9, 0.02));
    let new = seed_node(
        &storage,
        "Correction: LR should be 3e-4",
        &["ml"],
        axis_vector(9, 0.01),
    );

    let plan = storage
        .plan_supersede(&old, &new, MergePolicy::default())
        .unwrap();
    // Preview did not mutate.
    let (vu0, _) = storage.read_bitemporal(&old).unwrap();
    assert!(vu0.is_none());

    let op = storage.apply_plan(&plan.id, true).unwrap();
    assert_eq!(op.op_type, "supersede");

    let (vu, sb) = storage.read_bitemporal(&old).unwrap();
    assert!(vu.is_some(), "old stamped valid_until");
    assert_eq!(sb.as_deref(), Some(new.as_str()));
    // New node untouched and valid.
    let (vu_new, sb_new) = storage.read_bitemporal(&new).unwrap();
    assert!(vu_new.is_none() && sb_new.is_none());
    // Old still queryable for audit (invalidate, don't delete).
    let old_node = storage.get_node(&old).unwrap().unwrap();
    assert_eq!(old_node.content, "LR should be 1e-4");

    // And reversible.
    storage.merge_undo(&op.id).unwrap();
    let (vu_r, sb_r) = storage.read_bitemporal(&old).unwrap();
    assert!(vu_r.is_none() && sb_r.is_none());
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_protect_blocks_merge_away() {
    let storage = create_test_storage();
    let pinned = seed_node(
        &storage,
        "Load-bearing fact",
        &["pin"],
        axis_vector(11, 0.02),
    );
    let other = seed_node(
        &storage,
        "Load-bearing fact restated",
        &["pin"],
        axis_vector(11, 0.01),
    );
    storage.set_protected(&pinned, true).unwrap();
    assert!(storage.is_protected(&pinned).unwrap());

    // Protected node may not be merged AWAY (survivor=other).
    let err = storage.plan_merge(
        &[other.clone(), pinned.clone()],
        Some(&other),
        MergePolicy::default(),
    );
    assert!(err.is_err(), "merging a protected node away must fail");

    // But it CAN be the survivor.
    let ok = storage.plan_merge(
        &[pinned.clone(), other.clone()],
        Some(&pinned),
        MergePolicy::default(),
    );
    assert!(ok.is_ok(), "protected node can be the survivor");

    // Supersede of a protected node is also blocked.
    assert!(
        storage
            .plan_supersede(&pinned, &other, MergePolicy::default())
            .is_err(),
        "superseding a protected node must fail"
    );

    // merge_candidates flags the protected member.
    let cands = storage
        .merge_candidates(MergePolicy::default(), 20, &[])
        .unwrap();
    assert!(cands.iter().all(|c| c.has_protected_member));
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_apply_requires_confirm_for_low_confidence() {
    let storage = create_test_storage();
    // Tighten thresholds so a moderate pair lands in 'possible' (needs confirm).
    let strict = MergePolicy::new(0.99, 0.5, false);
    storage.set_merge_policy(strict).unwrap();

    let a = seed_node(&storage, "Topic alpha note", &["t"], axis_vector(13, 0.30));
    let b = seed_node(&storage, "Topic alpha aside", &["t"], axis_vector(13, 0.60));
    let plan = storage
        .plan_merge(&[a, b], None, storage.get_merge_policy().unwrap())
        .unwrap();
    assert_ne!(plan.classification, MatchClass::Match);

    // Without confirm => rejected.
    assert!(storage.apply_plan(&plan.id, false).is_err());
    // With confirm => applied.
    assert!(storage.apply_plan(&plan.id, true).is_ok());
    // Re-applying an applied plan => rejected.
    assert!(storage.apply_plan(&plan.id, true).is_err());
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_merge_policy_roundtrip_persists() {
    let storage = create_test_storage();
    let p = MergePolicy::new(0.9, 0.6, true);
    storage.set_merge_policy(p).unwrap();
    let got = storage.get_merge_policy().unwrap();
    assert!((got.match_threshold - 0.9).abs() < 1e-6);
    assert!((got.possible_threshold - 0.6).abs() < 1e-6);
    assert!(got.auto_apply);
}

#[test]
fn test_set_protected_unknown_node_errors() {
    let storage = create_test_storage();
    assert!(storage.set_protected("does-not-exist", true).is_err());
}
