//! Tests for `sqlite/sync.rs`: portable archive round trips and merge import.

use super::*;

#[test]
fn test_portable_archive_exact_round_trip() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");

    let first = source
        .ingest(IngestInput {
            content: "Portable archive alpha memory".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["portable".to_string()],
            source: Some("test".to_string()),
            ..Default::default()
        })
        .unwrap();
    let second = source
        .ingest(IngestInput {
            content: "Portable archive beta memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    source.mark_reviewed(&first.id, Rating::Good).unwrap();
    source
        .save_connection(&ConnectionRecord {
            source_id: first.id.clone(),
            target_id: second.id.clone(),
            strength: 0.75,
            link_type: "semantic".to_string(),
            created_at: Utc::now(),
            last_activated: Utc::now(),
            activation_count: 1,
        })
        .unwrap();
    source
        .save_composition(
            &CompositionEventRecord {
                id: "portable-composition-1".to_string(),
                created_at: Utc::now(),
                tool: "deep_reference".to_string(),
                mode: "bounty".to_string(),
                query: Some("portable composition".to_string()),
                query_hash: Some("sha256:portable".to_string()),
                confidence: Some(0.9),
                status: Some("resolved".to_string()),
                output_preview: Some("Portable composition event".to_string()),
                metadata: serde_json::json!({}),
            },
            &[
                CompositionMemberRecord {
                    event_id: "portable-composition-1".to_string(),
                    memory_id: first.id.clone(),
                    role: "primary".to_string(),
                    rank: 0,
                    trust: Some(0.9),
                    score: Some(1.0),
                    preview: Some("alpha".to_string()),
                    metadata: serde_json::json!({}),
                },
                CompositionMemberRecord {
                    event_id: "portable-composition-1".to_string(),
                    memory_id: second.id.clone(),
                    role: "supporting".to_string(),
                    rank: 1,
                    trust: Some(0.8),
                    score: Some(0.8),
                    preview: Some("beta".to_string()),
                    metadata: serde_json::json!({}),
                },
            ],
            &[CompositionOutcomeRecord {
                id: "portable-composition-outcome-1".to_string(),
                event_id: "portable-composition-1".to_string(),
                outcome_type: "helpful".to_string(),
                labeled_at: Utc::now(),
                label_source: "test".to_string(),
                confidence_delta: None,
                notes: None,
                metadata: serde_json::json!({}),
            }],
        )
        .unwrap();

    let archive = source.export_portable_archive().unwrap();
    assert_eq!(archive.archive_format, PORTABLE_ARCHIVE_FORMAT);
    assert!(archive.total_rows() >= 3);
    assert!(
        archive
            .tables
            .iter()
            .any(|table| table.name == "knowledge_nodes" && table.rows.len() == 2)
    );
    for table_name in [
        "composition_events",
        "composition_members",
        "composition_outcomes",
    ] {
        assert!(
            archive.tables.iter().any(|table| table.name == table_name),
            "{table_name} must be included in portable archive"
        );
    }

    let target = create_test_storage_at(&target_dir, "target.db");
    let report = target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap();
    assert!(report.rows_imported >= 3);
    assert!(report.fts_rebuilt);

    let restored = target.get_node(&first.id).unwrap().unwrap();
    assert_eq!(restored.id, first.id);
    assert_eq!(restored.content, first.content);
    assert_eq!(restored.tags, first.tags);
    assert_eq!(restored.reps, 1);

    let connections = target.get_connections_for_memory(&first.id).unwrap();
    assert_eq!(connections.len(), 1);
    assert_eq!(connections[0].target_id, second.id);

    let composition = target
        .get_composition_event("portable-composition-1")
        .unwrap()
        .unwrap();
    assert_eq!(composition.mode, "bounty");
    assert_eq!(
        target
            .get_composition_members("portable-composition-1")
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        target
            .get_composition_outcomes("portable-composition-1")
            .unwrap()
            .len(),
        1
    );

    let results = target.search("alpha", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, first.id);
}

#[test]
fn test_portable_import_rejects_non_empty_target() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    source
        .ingest(IngestInput {
            content: "Source memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let archive = source.export_portable_archive().unwrap();

    let target = create_test_storage_at(&target_dir, "target.db");
    target
        .ingest(IngestInput {
            content: "Existing target memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    let err = target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("requires an empty target database")
    );
}

#[test]
fn test_portable_import_rejects_unknown_mode() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    source
        .ingest(IngestInput {
            content: "Source memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let mut archive = source.export_portable_archive().unwrap();
    archive.mode = "merge".to_string();

    let target = create_test_storage_at(&target_dir, "target.db");
    let err = target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("Unsupported portable archive mode")
    );
}

#[test]
fn test_portable_import_rejects_malformed_table_list() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    source
        .ingest(IngestInput {
            content: "Source memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    let mut duplicate_archive = source.export_portable_archive().unwrap();
    let duplicate_table = duplicate_archive
        .tables
        .iter()
        .find(|table| table.name == "knowledge_nodes")
        .unwrap()
        .clone();
    duplicate_archive.tables.push(duplicate_table);

    let target = create_test_storage_at(&target_dir, "target-duplicate.db");
    let err = target
        .import_portable_archive(&duplicate_archive, PortableImportMode::EmptyOnly)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("Portable archive contains duplicate table")
    );

    let mut unknown_archive = source.export_portable_archive().unwrap();
    unknown_archive.tables.push(PortableTable {
        name: "sqlite_sequence".to_string(),
        columns: vec!["name".to_string(), "seq".to_string()],
        rows: vec![],
    });

    let target = create_test_storage_at(&target_dir, "target-unknown.db");
    let err = target
        .import_portable_archive(&unknown_archive, PortableImportMode::EmptyOnly)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("Portable archive contains unsupported table")
    );
}

#[test]
fn test_portable_merge_import_combines_non_empty_databases() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    let target = create_test_storage_at(&target_dir, "target.db");

    let source_node = source
        .ingest(IngestInput {
            content: "Source sync memory".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["sync".to_string()],
            ..Default::default()
        })
        .unwrap();
    let target_node = target
        .ingest(IngestInput {
            content: "Target local memory".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["local".to_string()],
            ..Default::default()
        })
        .unwrap();

    let archive = source.export_portable_archive().unwrap();
    let report = target
        .import_portable_archive(&archive, PortableImportMode::Merge)
        .unwrap();

    assert!(report.rows_inserted > 0);
    assert!(target.get_node(&source_node.id).unwrap().is_some());
    assert!(target.get_node(&target_node.id).unwrap().is_some());
}

#[test]
fn test_portable_merge_import_keeps_newer_local_memory() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    let target = create_test_storage_at(&target_dir, "target.db");

    let node = source
        .ingest(IngestInput {
            content: "Original shared memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let archive = source.export_portable_archive().unwrap();
    target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap();

    let newer = (Utc::now() + Duration::hours(1)).to_rfc3339();
    {
        let writer = target.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET content = ?1, updated_at = ?2 WHERE id = ?3",
                params!["Newer local edit", newer, &node.id],
            )
            .unwrap();
    }

    let report = target
        .import_portable_archive(&archive, PortableImportMode::Merge)
        .unwrap();

    assert!(report.conflicts_kept_local >= 1);
    let restored = target.get_node(&node.id).unwrap().unwrap();
    assert_eq!(restored.content, "Newer local edit");
}

#[test]
fn test_portable_merge_import_keeps_children_for_newer_local_memory() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    let target = create_test_storage_at(&target_dir, "target.db");

    let node = source
        .ingest(IngestInput {
            content: "Shared parent with child rows".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    let source_time = Utc::now().to_rfc3339();
    {
        let writer = source.writer.lock().unwrap();
        writer
            .execute(
                "INSERT OR REPLACE INTO node_embeddings
                     (node_id, embedding, dimensions, model, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                params![&node.id, vec![1_u8, 2, 3, 4], 4, "test-model", &source_time],
            )
            .unwrap();
        writer
            .execute(
                "INSERT OR REPLACE INTO fsrs_cards
                     (memory_id, difficulty, stability, state, reps, lapses)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![&node.id, 3.0_f64, 2.0_f64, "review", 2_i64, 0_i64],
            )
            .unwrap();
        writer
            .execute(
                "INSERT OR REPLACE INTO memory_states
                     (memory_id, state, last_access, access_count, state_entered_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                params![&node.id, "active", &source_time, 1_i64, &source_time],
            )
            .unwrap();
    }

    let archive = source.export_portable_archive().unwrap();
    target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap();

    let local_time = (Utc::now() + Duration::hours(1)).to_rfc3339();
    {
        let writer = target.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET content = ?1, updated_at = ?2 WHERE id = ?3",
                params!["Newer local parent edit", &local_time, &node.id],
            )
            .unwrap();
        writer
            .execute(
                "INSERT OR REPLACE INTO node_embeddings
                     (node_id, embedding, dimensions, model, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                params![&node.id, vec![9_u8, 8, 7, 6], 4, "test-model", &local_time],
            )
            .unwrap();
        writer
            .execute(
                "INSERT OR REPLACE INTO fsrs_cards
                     (memory_id, difficulty, stability, state, reps, lapses)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![&node.id, 9.0_f64, 8.0_f64, "review", 9_i64, 1_i64],
            )
            .unwrap();
        writer
            .execute(
                "INSERT OR REPLACE INTO memory_states
                     (memory_id, state, last_access, access_count, state_entered_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                params![&node.id, "silent", &local_time, 42_i64, &local_time],
            )
            .unwrap();
    }

    let report = target
        .import_portable_archive(&archive, PortableImportMode::Merge)
        .unwrap();

    assert!(report.conflicts_kept_local >= 4);
    let restored = target.get_node(&node.id).unwrap().unwrap();
    assert_eq!(restored.content, "Newer local parent edit");

    let reader = target.reader.lock().unwrap();
    let embedding: Vec<u8> = reader
        .query_row(
            "SELECT embedding FROM node_embeddings WHERE node_id = ?1",
            params![&node.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(embedding, vec![9_u8, 8, 7, 6]);

    let difficulty: f64 = reader
        .query_row(
            "SELECT difficulty FROM fsrs_cards WHERE memory_id = ?1",
            params![&node.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(difficulty, 9.0);

    let (state, access_count): (String, i64) = reader
        .query_row(
            "SELECT state, access_count FROM memory_states WHERE memory_id = ?1",
            params![&node.id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(state, "silent");
    assert_eq!(access_count, 42);
}

#[test]
fn test_portable_merge_import_keeps_composition_members_for_newer_local_memory() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    let target = create_test_storage_at(&target_dir, "target.db");

    let node = source
        .ingest(IngestInput {
            content: "Shared memory with historical composition".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["protocolgate".to_string()],
            ..Default::default()
        })
        .unwrap();
    source
        .save_composition(
            &CompositionEventRecord {
                id: "merge-composition-1".to_string(),
                created_at: Utc::now(),
                tool: "deep_reference".to_string(),
                mode: "bounty".to_string(),
                query: Some("historical composition".to_string()),
                query_hash: Some("sha256:historical".to_string()),
                confidence: Some(0.7),
                status: Some("resolved".to_string()),
                output_preview: Some("Historical composition survives merge".to_string()),
                metadata: serde_json::json!({}),
            },
            &[CompositionMemberRecord {
                event_id: "merge-composition-1".to_string(),
                memory_id: node.id.clone(),
                role: "primary".to_string(),
                rank: 0,
                trust: Some(0.8),
                score: Some(0.9),
                preview: Some("historical".to_string()),
                metadata: serde_json::json!({}),
            }],
            &[],
        )
        .unwrap();

    let archive = source.export_portable_archive().unwrap();
    target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap();

    let local_time = (Utc::now() + Duration::hours(1)).to_rfc3339();
    {
        let writer = target.writer.lock().unwrap();
        writer
            .execute(
                "DELETE FROM composition_members WHERE event_id = ?1",
                params!["merge-composition-1"],
            )
            .unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET content = ?1, updated_at = ?2 WHERE id = ?3",
                params!["Newer local content", &local_time, &node.id],
            )
            .unwrap();
    }

    target
        .import_portable_archive(&archive, PortableImportMode::Merge)
        .unwrap();

    let restored = target.get_node(&node.id).unwrap().unwrap();
    assert_eq!(restored.content, "Newer local content");
    let members = target
        .get_composition_members("merge-composition-1")
        .unwrap();
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].memory_id, node.id);
}

#[test]
fn test_portable_merge_import_applies_delete_tombstones() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    let target = create_test_storage_at(&target_dir, "target.db");

    let node = source
        .ingest(IngestInput {
            content: "Memory deleted on source".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let archive = source.export_portable_archive().unwrap();
    target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap();
    assert!(target.get_node(&node.id).unwrap().is_some());

    source.delete_node(&node.id).unwrap();
    let delete_archive = source.export_portable_archive().unwrap();
    let report = target
        .import_portable_archive(&delete_archive, PortableImportMode::Merge)
        .unwrap();

    assert!(report.rows_deleted >= 1);
    assert!(target.get_node(&node.id).unwrap().is_none());
}

#[test]
fn test_portable_merge_import_preserves_purge_tombstones() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    let target = create_test_storage_at(&target_dir, "target.db");

    let node = source
        .ingest(IngestInput {
            content: "Memory purged on source".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["sync".to_string()],
            ..Default::default()
        })
        .unwrap();
    source
        .save_composition(
            &CompositionEventRecord {
                id: "portable-purge-composition".to_string(),
                created_at: Utc::now(),
                tool: "deep_reference".to_string(),
                mode: "sync".to_string(),
                query: Some("portable purge preview".to_string()),
                query_hash: Some("fnv1a64:portable-purge".to_string()),
                confidence: Some(0.7),
                status: Some("resolved".to_string()),
                output_preview: None,
                metadata: serde_json::json!({}),
            },
            &[CompositionMemberRecord {
                event_id: "portable-purge-composition".to_string(),
                memory_id: node.id.clone(),
                role: "primary".to_string(),
                rank: 0,
                trust: Some(0.8),
                score: Some(0.8),
                preview: Some("Portable purge composition preview leak".to_string()),
                metadata: serde_json::json!({}),
            }],
            &[],
        )
        .unwrap();
    let archive = source.export_portable_archive().unwrap();
    target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap();
    assert!(target.get_node(&node.id).unwrap().is_some());
    assert_eq!(
        target
            .get_composition_members("portable-purge-composition")
            .unwrap()[0]
            .preview
            .as_deref(),
        Some("Portable purge composition preview leak")
    );
    {
        let writer = target.writer.lock().unwrap();
        writer
            .execute(
                "INSERT INTO memory_prs (
                        id, kind, status, title, subject_id, diff, signals, created_at
                     ) VALUES (?1, 'new_fact', 'pending', ?2, ?3, '{}', '[]', ?4)",
                params![
                    "portable-purge-review",
                    "remote cleanup review",
                    &node.id,
                    Utc::now().to_rfc3339(),
                ],
            )
            .unwrap();
    }

    source
        .purge_node(&node.id, Some("sync purge test"))
        .unwrap();
    let purge_archive = source.export_portable_archive().unwrap();
    assert!(
        !serde_json::to_string(&purge_archive)
            .unwrap()
            .contains("Portable purge composition preview leak"),
        "source portable archive should not retain purged composition previews"
    );
    let report = target
        .import_portable_archive(&purge_archive, PortableImportMode::Merge)
        .unwrap();

    assert!(report.rows_deleted >= 1);
    assert!(target.get_node(&node.id).unwrap().is_none());
    assert!(
        target
            .get_composition_members("portable-purge-composition")
            .unwrap()
            .is_empty(),
        "portable purge merge should delete composition evidence that references the target"
    );

    let writer = target.writer.lock().unwrap();
    let review_count: i64 = writer
        .query_row(
            "SELECT COUNT(*) FROM memory_prs WHERE id = 'portable-purge-review'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        review_count, 0,
        "portable purge merge must run the full non-FK evidence cleanup"
    );
    let tombstone_count: i64 = writer
        .query_row(
            "SELECT COUNT(*) FROM deletion_tombstones
                 WHERE memory_id = ?1 AND reason IS NULL AND tags = '[]'",
            params![SqliteMemoryStore::opaque_tombstone_marker(&node.id)],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(tombstone_count, 1);
}

#[test]
fn opaque_tombstone_rejects_a_later_node_archive_even_with_newer_timestamp() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    let target = create_test_storage_at(&target_dir, "target.db");

    let node = source
        .ingest(IngestInput {
            content: "must not resurrect after opaque tombstone".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let mut later_node_archive = source.export_portable_archive().unwrap();
    let node_table = later_node_archive
        .tables
        .iter_mut()
        .find(|table| table.name == "knowledge_nodes")
        .unwrap();
    let updated_at_index = node_table
        .columns
        .iter()
        .position(|column| column == "updated_at")
        .unwrap();
    match &mut node_table.rows[0][updated_at_index] {
        PortableValue::Text(value) => *value = (Utc::now() + Duration::hours(24)).to_rfc3339(),
        value => panic!("knowledge_nodes.updated_at must be text, got {value:?}"),
    }

    source.purge_node(&node.id, None).unwrap();
    let tombstone_archive = source.export_portable_archive().unwrap();
    target
        .import_portable_archive(&tombstone_archive, PortableImportMode::Merge)
        .unwrap();
    assert!(target.get_node(&node.id).unwrap().is_none());

    let report = target
        .import_portable_archive(&later_node_archive, PortableImportMode::Merge)
        .unwrap();
    assert!(target.get_node(&node.id).unwrap().is_none());
    assert!(report.rows_skipped >= 1);
}

#[test]
fn test_portable_merge_import_purge_wins_over_newer_local_edit() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    let target = create_test_storage_at(&target_dir, "target.db");

    let node = source
        .ingest(IngestInput {
            content: "Memory that will be purged on source".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["sync".to_string()],
            ..Default::default()
        })
        .unwrap();
    let archive = source.export_portable_archive().unwrap();
    target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap();

    let newer = (Utc::now() + Duration::hours(1)).to_rfc3339();
    {
        let writer = target.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET content = ?1, updated_at = ?2 WHERE id = ?3",
                params!["Newer local edit before purge arrives", newer, &node.id],
            )
            .unwrap();
    }

    source
        .purge_node(&node.id, Some("hard purge wins sync conflict"))
        .unwrap();
    let purge_archive = source.export_portable_archive().unwrap();
    let report = target
        .import_portable_archive(&purge_archive, PortableImportMode::Merge)
        .unwrap();

    assert!(report.rows_deleted >= 1);
    assert!(target.get_node(&node.id).unwrap().is_none());
}

#[test]
fn test_file_portable_sync_round_trips_between_devices() {
    let sync_dir = tempdir().unwrap();
    let first_dir = tempdir().unwrap();
    let second_dir = tempdir().unwrap();
    let sync_path = sync_dir.path().join("vestige-sync.json");
    let first = create_test_storage_at(&first_dir, "first.db");
    let second = create_test_storage_at(&second_dir, "second.db");

    let first_node = first
        .ingest(IngestInput {
            content: "First device memory".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["sync".to_string()],
            ..Default::default()
        })
        .unwrap();
    let first_push = first.sync_portable_archive_file(&sync_path).unwrap();
    assert!(!first_push.pulled);
    assert!(sync_path.exists());

    let second_node = second
        .ingest(IngestInput {
            content: "Second device memory".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["sync".to_string()],
            ..Default::default()
        })
        .unwrap();
    let second_sync = second.sync_portable_archive_file(&sync_path).unwrap();
    assert!(second_sync.pulled);
    assert!(second.get_node(&first_node.id).unwrap().is_some());

    let first_sync = first.sync_portable_archive_file(&sync_path).unwrap();
    assert!(first_sync.pulled);
    assert!(first.get_node(&second_node.id).unwrap().is_some());
    assert!(first_sync.pushed_rows >= 2);
}

#[test]
fn test_get_last_backup_timestamp_no_panic() {
    // Static method should not panic even if no backups exist
    let _ = Storage::get_last_backup_timestamp();
}
