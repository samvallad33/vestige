//! Tests for `sqlite/connectors.rs`: source upserts, cursors and reconciliation.

use super::*;

// ===================== Connector sync (#57) =========================
fn node_count(store: &Storage) -> i64 {
    // Count rows for our test source so embeddings/other tests don't bleed in.
    let reader = store.reader.lock().unwrap();
    reader
        .query_row(
            "SELECT COUNT(*) FROM knowledge_nodes WHERE source_system = 'github'",
            [],
            |r| r.get(0),
        )
        .unwrap()
}

#[test]
fn upsert_by_source_is_idempotent_across_reruns() {
    let store = create_test_storage();

    // First sync: a brand-new record → Created.
    let r1 = store
        .upsert_by_source(source_input("1", "Bug: crash on startup", "hash-a"))
        .unwrap();
    assert_eq!(r1.outcome, SourceUpsertOutcome::Created);
    assert_eq!(node_count(&store), 1);

    // Re-sync the SAME record with the SAME hash twice → Unchanged, no dupes.
    for _ in 0..2 {
        let r = store
            .upsert_by_source(source_input("1", "Bug: crash on startup", "hash-a"))
            .unwrap();
        assert_eq!(r.outcome, SourceUpsertOutcome::Unchanged);
        assert_eq!(r.node_id, r1.node_id, "must reuse the same memory id");
    }
    assert_eq!(
        node_count(&store),
        1,
        "idempotent: still exactly one memory"
    );
}

#[test]
fn upsert_by_source_updates_in_place_when_hash_changes() {
    let store = create_test_storage();
    let created = store
        .upsert_by_source(source_input("7", "old body", "hash-old"))
        .unwrap();

    // Upstream edit: content + hash change → Updated, same id, new content.
    let updated = store
        .upsert_by_source(source_input("7", "new edited body", "hash-new"))
        .unwrap();
    assert_eq!(updated.outcome, SourceUpsertOutcome::Updated);
    assert_eq!(updated.node_id, created.node_id);
    assert_eq!(node_count(&store), 1, "update must not duplicate");

    let node = store.get_node(&created.node_id).unwrap().unwrap();
    assert_eq!(node.content, "new edited body");
    let env = node.source_envelope.expect("envelope persisted");
    assert_eq!(env.content_hash.as_deref(), Some("hash-new"));
    assert_eq!(env.source_id.as_deref(), Some("7"));
}

#[test]
fn upsert_by_source_without_key_falls_back_to_create() {
    let store = create_test_storage();
    // Envelope present but missing source_id → not keyed → plain create.
    let input = IngestInput {
        content: "loose note".to_string(),
        node_type: "fact".to_string(),
        source_envelope: Some(crate::memory::SourceEnvelope {
            source_url: Some("https://example.com/x".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let r = store.upsert_by_source(input).unwrap();
    assert_eq!(r.outcome, SourceUpsertOutcome::Created);
}

#[test]
fn connector_cursor_round_trips() {
    let store = create_test_storage();
    // Unknown scope → zeroed cursor.
    let empty = store.get_connector_cursor("github", "o/r").unwrap();
    assert!(empty.cursor_updated_at.is_none());
    assert_eq!(empty.records_seen, 0);

    let ts = Utc::now();
    let cursor = ConnectorCursor {
        source_system: "github".to_string(),
        scope: "o/r".to_string(),
        cursor_updated_at: Some(ts),
        last_synced_at: Some(ts),
        last_full_reconcile_at: None,
        records_seen: 42,
    };
    store.save_connector_cursor(&cursor).unwrap();

    let back = store.get_connector_cursor("github", "o/r").unwrap();
    assert_eq!(back.records_seen, 42);
    assert_eq!(
        back.cursor_updated_at.map(|d| d.to_rfc3339()),
        Some(ts.to_rfc3339())
    );

    // Upsert semantics: saving again replaces, never duplicates.
    let mut c2 = cursor.clone();
    c2.records_seen = 99;
    store.save_connector_cursor(&c2).unwrap();
    assert_eq!(
        store
            .get_connector_cursor("github", "o/r")
            .unwrap()
            .records_seen,
        99
    );
}

#[test]
fn reconcile_tombstones_records_absent_from_live_set() {
    let store = create_test_storage();
    // Three synced issues in scope o/r.
    for id in ["1", "2", "3"] {
        store
            .upsert_by_source(source_input(id, &format!("issue {id}"), &format!("h{id}")))
            .unwrap();
    }

    // Reconcile: only 1 and 3 are still visible upstream → 2 is tombstoned.
    let report = store
        .reconcile_source_tombstones("github", "o/r", &["1".to_string(), "3".to_string()])
        .unwrap();
    assert_eq!(report.considered, 3);
    assert_eq!(report.tombstoned.len(), 1, "exactly issue 2 tombstoned");

    // Issue 2's memory is invalidated (valid_until set) but NOT purged —
    // content retained for audit, just no longer currently-valid.
    let two = {
        let reader = store.reader.lock().unwrap();
        reader
            .query_row(
                "SELECT id, valid_until FROM knowledge_nodes WHERE source_id = '2'",
                [],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?)),
            )
            .unwrap()
    };
    assert!(
        two.1.is_some(),
        "tombstoned record must have valid_until set"
    );
    let node = store.get_node(&two.0).unwrap().unwrap();
    assert!(
        !node.is_currently_valid(),
        "tombstoned node is not valid now"
    );
    assert_eq!(node.content, "issue 2", "content retained for audit");

    // A reappearing record un-tombstones on next upsert (clears valid_until).
    store
        .upsert_by_source(source_input("2", "issue 2", "h2"))
        .unwrap();
    let revived = store.get_node(&two.0).unwrap().unwrap();
    assert!(
        revived.is_currently_valid(),
        "re-synced record is valid again"
    );
}

#[test]
fn upsert_clears_superseded_by_when_record_reappears() {
    // Regression: un-tombstoning must clear BOTH bitemporal markers. A
    // connector node that was superseded/merged (valid_until + superseded_by
    // both set) and then re-observed upstream must come back fully clean,
    // otherwise it is currently-valid yet still flagged superseded and is
    // permanently excluded from merge candidacy.
    let store = create_test_storage();
    let created = store
        .upsert_by_source(source_input("9", "body v1", "h9a"))
        .unwrap();

    // Simulate the node having been superseded (as merge/supersede would).
    {
        let writer = store.writer.lock().unwrap();
        writer
                .execute(
                    "UPDATE knowledge_nodes SET valid_until = ?1, superseded_by = 'survivor-id' WHERE id = ?2",
                    params![Utc::now().to_rfc3339(), created.node_id],
                )
                .unwrap();
    }
    assert!(
        store
            .superseded_node_ids()
            .unwrap()
            .contains(&created.node_id),
        "precondition: node is superseded"
    );

    // Re-sync with a content change → Updated branch must clear both markers.
    let res = store
        .upsert_by_source(source_input("9", "body v2 edited", "h9b"))
        .unwrap();
    assert_eq!(res.outcome, SourceUpsertOutcome::Updated);
    assert!(
        !store
            .superseded_node_ids()
            .unwrap()
            .contains(&created.node_id),
        "superseded_by must be cleared on re-sync (no bitemporal zombie)"
    );
    let node = store.get_node(&created.node_id).unwrap().unwrap();
    assert!(node.is_currently_valid());

    // Also exercise the Unchanged branch: supersede again, re-sync same hash.
    {
        let writer = store.writer.lock().unwrap();
        writer
                .execute(
                    "UPDATE knowledge_nodes SET valid_until = ?1, superseded_by = 'survivor-id' WHERE id = ?2",
                    params![Utc::now().to_rfc3339(), created.node_id],
                )
                .unwrap();
    }
    let res2 = store
        .upsert_by_source(source_input("9", "body v2 edited", "h9b"))
        .unwrap();
    assert_eq!(res2.outcome, SourceUpsertOutcome::Unchanged);
    assert!(
        !store
            .superseded_node_ids()
            .unwrap()
            .contains(&created.node_id),
        "Unchanged branch must also clear superseded_by"
    );
}

/// Build a `source_input` whose envelope carries an explicit project.
fn source_input_in_project(
    id: &str,
    content: &str,
    hash: &str,
    project: Option<&str>,
) -> IngestInput {
    let mut input = source_input(id, content, hash);
    input.source_envelope.as_mut().unwrap().source_project = project.map(str::to_string);
    input
}

#[test]
fn upsert_by_source_scopes_key_by_project() {
    // V19: two sources of the same system reuse bare per-project ids, so
    // the same (system, id) under DIFFERENT projects must yield two
    // distinct nodes, and re-syncing each must hit its own row (Unchanged).
    let store = create_test_storage();

    let a = store
        .upsert_by_source(source_input_in_project(
            "5",
            "repoA issue 5",
            "hA",
            Some("octocat/repoA"),
        ))
        .unwrap();
    let b = store
        .upsert_by_source(source_input_in_project(
            "5",
            "repoB issue 5",
            "hB",
            Some("octocat/repoB"),
        ))
        .unwrap();
    assert_eq!(a.outcome, SourceUpsertOutcome::Created);
    assert_eq!(b.outcome, SourceUpsertOutcome::Created);
    assert_ne!(a.node_id, b.node_id, "projects must not share a row");
    assert_eq!(node_count(&store), 2);

    // Re-sync both records with unchanged hashes → each resolves to ITS row.
    let ra = store
        .upsert_by_source(source_input_in_project(
            "5",
            "repoA issue 5",
            "hA",
            Some("octocat/repoA"),
        ))
        .unwrap();
    assert_eq!(ra.outcome, SourceUpsertOutcome::Unchanged);
    assert_eq!(ra.node_id, a.node_id);
    let rb = store
        .upsert_by_source(source_input_in_project(
            "5",
            "repoB issue 5",
            "hB",
            Some("octocat/repoB"),
        ))
        .unwrap();
    assert_eq!(rb.outcome, SourceUpsertOutcome::Unchanged);
    assert_eq!(rb.node_id, b.node_id);
    assert_eq!(node_count(&store), 2, "resync must not duplicate");
}

#[test]
fn upsert_by_source_matches_legacy_null_project_row_with_empty_string() {
    // Regression: the V19 unique index buckets NULL and '' together via
    // COALESCE(source_project, ''), but the lookup used `source_project IS
    // ?3`, which treats NULL and '' as distinct. A legacy NULL-project row
    // plus an ''-project envelope for the same (system, id) made the lookup
    // miss, and the fall-through INSERT then hit the UNIQUE constraint.
    let store = create_test_storage();
    let created = store
        .upsert_by_source(source_input("41", "legacy body", "h-legacy"))
        .unwrap();
    assert_eq!(created.outcome, SourceUpsertOutcome::Created);

    // Simulate a pre-V19 legacy row: source_project stored as NULL.
    {
        let writer = store.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET source_project = NULL WHERE id = ?1",
                params![created.node_id],
            )
            .unwrap();
    }

    // New connector run sends Some("") for the same (system, id). Must
    // UPDATE the legacy row in place — not error on the unique index.
    let res = store
        .upsert_by_source(source_input_in_project(
            "41",
            "legacy body edited",
            "h-new",
            Some(""),
        ))
        .expect("''-project envelope must resolve the NULL-project row, not UNIQUE-fail");
    assert_eq!(res.outcome, SourceUpsertOutcome::Updated);
    assert_eq!(res.node_id, created.node_id, "must reuse the legacy row");
    assert_eq!(node_count(&store), 1, "no duplicate row in the NULL bucket");

    // And the Unchanged path resolves through the same bucket too.
    let res2 = store
        .upsert_by_source(source_input_in_project(
            "41",
            "legacy body edited",
            "h-new",
            Some(""),
        ))
        .unwrap();
    assert_eq!(res2.outcome, SourceUpsertOutcome::Unchanged);
    assert_eq!(res2.node_id, created.node_id);
}
