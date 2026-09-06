//! Tests for `sqlite/purge.rs`: delete, purge scrubbing and retention GC.

use super::*;

#[test]
fn test_delete() {
    let storage = create_test_storage();

    let input = IngestInput {
        content: "To be deleted".to_string(),
        node_type: "fact".to_string(),
        tags: vec!["sensitive-delete-tag".to_string()],
        ..Default::default()
    };

    let node = storage.ingest(input).unwrap();
    assert!(storage.get_node(&node.id).unwrap().is_some());

    let deleted = storage.delete_node(&node.id).unwrap();
    assert!(deleted);
    assert!(storage.get_node(&node.id).unwrap().is_none());
    let archive = serde_json::to_string(&storage.export_portable_archive().unwrap()).unwrap();
    assert!(!archive.contains(&node.id));
    assert!(!archive.contains("sensitive-delete-tag"));
}

/// The explicit GC path must never collect a protected (pinned) memory,
/// no matter how decayed it is. A pin says "keep this"; low retention
/// only says "rarely retrieved".
#[test]
fn gc_spares_protected_memories() {
    let storage = create_test_storage();
    let pinned = storage
        .ingest(IngestInput {
            content: "Pinned but heavily decayed memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let doomed = storage
        .ingest(IngestInput {
            content: "Unpinned decayed memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    {
        let writer = storage.writer.lock().unwrap();
        let old = (Utc::now() - Duration::days(120)).to_rfc3339();
        for id in [&pinned.id, &doomed.id] {
            writer
                .execute(
                    "UPDATE knowledge_nodes
                         SET retention_strength = 0.05, created_at = ?1 WHERE id = ?2",
                    params![old, id],
                )
                .unwrap();
        }
    }
    storage.set_protected(&pinned.id, true).unwrap();

    let deleted = storage.gc_below_retention(0.3, 30).unwrap();
    assert_eq!(deleted, 1, "only the unpinned memory is collected");
    assert!(
        storage.get_node(&pinned.id).unwrap().is_some(),
        "pin survives GC"
    );
    assert!(storage.get_node(&doomed.id).unwrap().is_none());
}

#[test]
fn gc_uses_the_privacy_cleanup_deletion_path() {
    let storage = create_test_storage();
    let node = storage
        .ingest(IngestInput {
            content: "GC deletion privacy target".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["gc-sensitive-tag".to_string()],
            ..Default::default()
        })
        .unwrap();
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes
                     SET retention_strength = 0.0, created_at = '2000-01-01T00:00:00Z'
                     WHERE id = ?1",
                params![&node.id],
            )
            .unwrap();
    }

    assert_eq!(storage.gc_below_retention(0.1, 1).unwrap(), 1);
    let archive = serde_json::to_string(&storage.export_portable_archive().unwrap()).unwrap();
    assert!(!archive.contains(&node.id));
    assert!(!archive.contains("gc-sensitive-tag"));
}

#[test]
fn purging_empty_content_does_not_scrub_unrelated_evidence() {
    let storage = create_test_storage();
    let empty = storage
        .ingest(IngestInput {
            content: String::new(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "INSERT INTO memory_prs (
                        id, kind, status, title, diff, signals, created_at
                     ) VALUES ('unrelated-review', 'new_fact', 'pending',
                               'keep this review', '{}', '[]', ?1)",
                params![Utc::now().to_rfc3339()],
            )
            .unwrap();
    }

    assert!(storage.purge_node(&empty.id, None).unwrap().deleted);
    let writer = storage.writer.lock().unwrap();
    let remaining_reviews: i64 = writer
        .query_row(
            "SELECT COUNT(*) FROM memory_prs WHERE id = 'unrelated-review'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(remaining_reviews, 1);
}

#[test]
fn test_purge_scrubs_insight_json_orphans_children_and_writes_tombstone() {
    let storage = create_test_storage();
    let doomed = storage
        .ingest(IngestInput {
            content: "Sensitive purge target memory".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["sensitive".to_string()],
            ..Default::default()
        })
        .unwrap();
    let other_a = storage
        .ingest(IngestInput {
            content: "Other source memory A".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let other_b = storage
        .ingest(IngestInput {
            content: "Other source memory B".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let child = storage
        .ingest(IngestInput {
            content: "Temporal summary child".to_string(),
            node_type: "summary".to_string(),
            ..Default::default()
        })
        .unwrap();

    {
        let writer = storage.writer.lock().unwrap();
        writer
                .execute(
                    "INSERT INTO memory_connections (
                        source_id, target_id, strength, link_type, created_at, last_activated, activation_count
                     ) VALUES (?1, ?2, 0.9, 'semantic', ?3, ?3, 0)",
                    params![doomed.id, other_a.id, Utc::now().to_rfc3339()],
                )
                .unwrap();
        writer
                .execute(
                    "INSERT INTO insights (
                        id, insight, source_memories, confidence, novelty_score, insight_type, generated_at
                     ) VALUES (?1, 'drop me', ?2, 0.9, 0.2, 'synthesis', ?3)",
                    params![
                        Uuid::new_v4().to_string(),
                        serde_json::to_string(&vec![doomed.id.clone(), other_a.id.clone()]).unwrap(),
                        Utc::now().to_rfc3339()
                    ],
                )
                .unwrap();
        writer
                .execute(
                    "INSERT INTO insights (
                        id, insight, source_memories, confidence, novelty_score, insight_type, generated_at
                     ) VALUES (?1, 'rewrite me', ?2, 0.9, 0.2, 'synthesis', ?3)",
                    params![
                        Uuid::new_v4().to_string(),
                        serde_json::to_string(&vec![
                            doomed.id.clone(),
                            other_a.id.clone(),
                            other_b.id.clone()
                        ])
                        .unwrap(),
                        Utc::now().to_rfc3339()
                    ],
                )
                .unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET summary_parent_id = ?1 WHERE id = ?2",
                params![doomed.id, child.id],
            )
            .unwrap();
        writer
            .execute(
                "INSERT INTO memory_prs (
                        id, kind, status, title, subject_id, diff, signals, created_at
                     ) VALUES (?1, 'new_fact', 'pending', ?2, ?3, ?4, '[]', ?5)",
                params![
                    "purge-review-leak",
                    "Sensitive purge target memory review preview",
                    doomed.id,
                    serde_json::json!({
                        "contentPreview": "Sensitive purge target memory"
                    })
                    .to_string(),
                    Utc::now().to_rfc3339(),
                ],
            )
            .unwrap();
    }

    storage
        .save_composition(
            &CompositionEventRecord {
                id: "purge-composition-preview-test".to_string(),
                created_at: Utc::now(),
                tool: "deep_reference".to_string(),
                mode: "audit".to_string(),
                query: Some("purge preview leak".to_string()),
                query_hash: Some("fnv1a64:purge".to_string()),
                confidence: Some(0.7),
                status: Some("resolved".to_string()),
                output_preview: None,
                metadata: serde_json::json!({}),
            },
            &[CompositionMemberRecord {
                event_id: "purge-composition-preview-test".to_string(),
                memory_id: doomed.id.clone(),
                role: "primary".to_string(),
                rank: 0,
                trust: Some(0.8),
                score: Some(0.9),
                preview: Some("Sensitive purge target memory preview leak".to_string()),
                metadata: serde_json::json!({}),
            }],
            &[],
        )
        .unwrap();

    let report = storage
        .purge_node(&doomed.id, Some("user requested hard purge"))
        .unwrap();
    assert!(report.deleted);
    assert_eq!(report.edges_pruned, 1);
    assert_eq!(report.insights_deleted, 1);
    assert_eq!(report.insights_rewritten, 1);
    assert_eq!(report.children_orphaned, 1);
    assert!(storage.get_node(&doomed.id).unwrap().is_none());

    let writer = storage.writer.lock().unwrap();
    let remaining_refs: Vec<String> = writer
        .prepare("SELECT source_memories FROM insights")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .filter_map(|row| row.ok())
        .collect();
    assert_eq!(remaining_refs.len(), 1);
    assert!(!remaining_refs[0].contains(&doomed.id));

    let child_parent: Option<String> = writer
        .query_row(
            "SELECT summary_parent_id FROM knowledge_nodes WHERE id = ?1",
            params![child.id],
            |row| row.get(0),
        )
        .unwrap();
    assert!(child_parent.is_none());

    let tombstone_count: i64 = writer
        .query_row(
            "SELECT COUNT(*) FROM deletion_tombstones
                 WHERE memory_id = ?1 AND reason IS NULL AND tags = '[]'",
            params![SqliteMemoryStore::opaque_tombstone_marker(&doomed.id)],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(tombstone_count, 1);
    let sync_tombstone_count: i64 = writer
        .query_row(
            "SELECT COUNT(*) FROM sync_tombstones
                 WHERE table_name = 'knowledge_nodes' AND row_id = ?1 AND reason IS NULL",
            params![SqliteMemoryStore::opaque_tombstone_marker(&doomed.id)],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(sync_tombstone_count, 1);

    let members = storage
        .get_composition_members("purge-composition-preview-test")
        .unwrap();
    assert!(
        members.is_empty(),
        "purge should remove composition evidence that references the target"
    );
    let review_count: i64 = writer
        .query_row(
            "SELECT COUNT(*) FROM memory_prs WHERE id = 'purge-review-leak'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        review_count, 0,
        "purge should remove linked review evidence"
    );
    let archive_json = serde_json::to_string(&storage.export_portable_archive().unwrap()).unwrap();
    assert!(
        !archive_json.contains("Sensitive purge target memory preview leak"),
        "portable archive should not retain purged memory content through composition previews"
    );
    assert!(
        !archive_json.contains(&doomed.id),
        "portable archive should not retain the purged memory's raw identifier"
    );
    assert!(
        !archive_json.contains("user requested hard purge"),
        "portable archive should not retain caller-controlled purge rationale"
    );

    let has_content_column: i64 = writer
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('deletion_tombstones') WHERE name = 'content'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(has_content_column, 0);
}

/// Purge is an erasure guarantee, so a referencing row the scrub cannot
/// read has to abort the whole purge rather than be skipped. Before this,
/// the three reference sweeps in `purge_node_in_transaction` used
/// `filter_map(|row| row.ok())`, so an unreadable row silently kept its
/// reference to a memory the caller was told had been erased.
#[test]
fn purge_fails_closed_when_a_referencing_row_cannot_be_read() {
    let storage = create_test_storage();
    let doomed = storage
        .ingest(IngestInput {
            content: "Purge target with an unreadable referrer".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    // `insights` is a non-STRICT table, so a BLOB survives in the TEXT
    // primary key. rusqlite then refuses to read it as a String, which is
    // exactly the "row we cannot read" the sweep used to swallow.
    {
        let writer = storage.writer.lock().unwrap();
        writer
                .execute(
                    "INSERT INTO insights (
                        id, insight, source_memories, confidence, novelty_score, insight_type, generated_at
                     ) VALUES (?1, 'unreadable id', ?2, 0.9, 0.2, 'synthesis', ?3)",
                    params![
                        vec![0xF0_u8, 0x9F, 0x92, 0xA9, 0xFF, 0xFE],
                        serde_json::to_string(&vec![doomed.id.clone()]).unwrap(),
                        Utc::now().to_rfc3339()
                    ],
                )
                .unwrap();
    }

    // The purge must refuse, not report success.
    let error = storage.purge_node(&doomed.id, None).unwrap_err();
    let rendered = error.to_string().to_lowercase();
    assert!(
        rendered.contains("type") || rendered.contains("column") || rendered.contains("convert"),
        "expected a read failure to surface, got: {rendered}"
    );

    // And the transaction rolled back: nothing was half-erased.
    assert!(
        storage.get_node(&doomed.id).unwrap().is_some(),
        "a refused purge must leave the memory intact, not partially scrubbed"
    );
}
