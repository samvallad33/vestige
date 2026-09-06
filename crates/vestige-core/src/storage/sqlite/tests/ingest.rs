//! Tests for `sqlite/ingest.rs`: the secret-ingest policy, tag mutations and
//! hygiene, and the smart-ingest bitemporal validity gates (#156).

use super::*;

// ===================== Secret-ingest policy (#154) ==================

#[test]
fn ingest_rejects_probable_github_secret_without_persisting_or_echoing_it() {
    let store = create_test_storage();
    let secret = format!("ghp_{}", "A".repeat(36));

    let err = store
        .ingest(IngestInput {
            content: format!("The temporary token is {secret}"),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap_err();

    assert!(matches!(err, StorageError::SecretDetected { .. }));
    assert!(
        !err.to_string().contains(&secret),
        "rejection must not echo the credential"
    );
    assert_eq!(
        store.get_stats().unwrap().total_nodes,
        0,
        "rejection must happen before creating a node"
    );
}

#[test]
fn update_node_content_rejects_probable_github_secret_without_mutating_node() {
    let store = create_test_storage();
    let node = store
        .ingest(IngestInput {
            content: "safe original content".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let secret = format!("ghp_{}", "A".repeat(36));

    let err = store
        .update_node_content(&node.id, &format!("replacement includes {secret}"))
        .unwrap_err();

    assert!(matches!(err, StorageError::SecretDetected { .. }));
    assert!(
        !err.to_string().contains(&secret),
        "rejection must not echo the credential"
    );
    assert_eq!(
        store.get_node(&node.id).unwrap().unwrap().content,
        "safe original content",
        "rejection must leave existing content intact"
    );
}

#[test]
fn connector_upsert_rejects_probable_github_secret_without_mutating_existing_record() {
    let store = create_test_storage();
    let created = store
        .upsert_by_source(source_input(
            "secret-policy",
            "safe connector body",
            "safe-hash",
        ))
        .unwrap();
    let secret = format!("ghp_{}", "A".repeat(36));

    let err = store
        .upsert_by_source(source_input(
            "secret-policy",
            &format!("updated connector body includes {secret}"),
            "secret-hash",
        ))
        .unwrap_err();

    assert!(matches!(err, StorageError::SecretDetected { .. }));
    assert!(
        !err.to_string().contains(&secret),
        "rejection must not echo the credential"
    );
    let stored = store.get_node(&created.node_id).unwrap().unwrap();
    assert_eq!(stored.content, "safe connector body");
    assert_eq!(
        stored
            .source_envelope
            .as_ref()
            .and_then(|envelope| envelope.content_hash.as_deref()),
        Some("safe-hash"),
        "preflight rejection must happen before connector metadata changes"
    );
}

#[test]
fn portable_import_rejects_secret_archive_atomically_before_safe_rows_import() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    source
        .ingest(IngestInput {
            content: "safe portable memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let secret = format!("ghp_{}", "A".repeat(36));
    source
        .ingest_with_secret_policy(
            IngestInput {
                content: format!("intentionally archived credential {secret}"),
                node_type: "fact".to_string(),
                ..Default::default()
            },
            SecretPolicy::AllowExplicitly,
        )
        .unwrap();
    let archive = source.export_portable_archive().unwrap();

    let target = create_test_storage_at(&target_dir, "target.db");
    let err = target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap_err();

    assert!(matches!(err, StorageError::SecretDetected { .. }));
    assert!(
        !err.to_string().contains(&secret),
        "rejection must not echo the credential"
    );
    assert_eq!(
        target.get_stats().unwrap().total_nodes,
        0,
        "archive preflight must prevent a partial import of safe sibling rows"
    );
}

fn preview_token(preview: &serde_json::Value) -> &str {
    preview["previewToken"].as_str().unwrap()
}

#[test]
fn tag_rename_is_previewed_scoped_exact_atomic_audited_and_reversible() {
    let storage = create_test_storage();
    let user = ingest_tagged_in_scope(
        &storage,
        "user",
        "scoped tag rename fixture",
        &[
            "keep",
            "legacytag156",
            "canonicaltag156",
            "canonicaltag156",
            "tail",
        ],
    );
    let prefix = ingest_tagged_in_scope(
        &storage,
        "user",
        "exact matching fixture",
        &["legacytag156-extra"],
    );
    let project = ingest_tagged_in_scope(
        &storage,
        "project-a",
        "cross scope fixture",
        &["legacytag156"],
    );
    let sources = vec!["legacytag156".to_string()];

    let preview = storage
        .preview_tag_mutation(&sources, "canonicaltag156", Some(" user "))
        .unwrap();
    assert_eq!(preview["affectedMemoryCount"], 1);
    assert_eq!(
        storage.get_node(&user.id).unwrap().unwrap().tags,
        vec![
            "keep",
            "legacytag156",
            "canonicaltag156",
            "canonicaltag156",
            "tail"
        ],
        "preview must not mutate"
    );

    let operation = storage
        .apply_tag_mutation(
            &sources,
            "canonicaltag156",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "standardize the issue 156 fixture tag",
        )
        .unwrap();
    assert_eq!(operation.op_type, "tag_rename");
    assert_eq!(operation.affected_ids, vec![user.id.clone()]);
    assert_eq!(
        storage.get_node(&user.id).unwrap().unwrap().tags,
        vec!["keep", "canonicaltag156", "tail"],
        "the union of source and existing target tags is one target at the first affected position"
    );
    assert_eq!(
        storage.get_node(&prefix.id).unwrap().unwrap().tags,
        vec!["legacytag156-extra"],
        "prefix tags must not match"
    );
    assert_eq!(
        storage.get_node(&project.id).unwrap().unwrap().tags,
        vec!["legacytag156"],
        "default scoped maintenance must not cross project boundaries"
    );
    assert!(
        storage
            .keyword_search("canonicaltag156", 10, 0.0)
            .unwrap()
            .iter()
            .any(|node| node.id == user.id),
        "the existing knowledge_nodes update trigger must keep FTS tag search consistent"
    );
    let logged = storage.get_merge_operation(&operation.id).unwrap().unwrap();
    assert_eq!(
        logged.reason.as_deref(),
        Some("standardize the issue 156 fixture tag")
    );

    storage.undo_tag_mutation(&operation.id).unwrap();
    assert_eq!(
        storage.get_node(&user.id).unwrap().unwrap().tags,
        vec![
            "keep",
            "legacytag156",
            "canonicaltag156",
            "canonicaltag156",
            "tail"
        ],
        "undo restores the exact pre-operation array"
    );
}

#[test]
fn tag_rename_all_scopes_is_explicit_and_updates_every_matching_row() {
    let storage = create_test_storage();
    let user = ingest_tagged_in_scope(&storage, "user", "user shared tag", &["shared"]);
    let project = ingest_tagged_in_scope(&storage, "project-a", "project shared tag", &["shared"]);
    let sources = vec!["shared".to_string()];

    let scoped = storage
        .preview_tag_mutation(&sources, "canonical", Some("user"))
        .unwrap();
    assert_eq!(scoped["affectedMemoryCount"], 1);
    assert_eq!(scoped["allScopes"], false);

    let preview = storage
        .preview_tag_mutation(&sources, "canonical", None)
        .unwrap();
    assert_eq!(preview["allScopes"], true);
    assert_eq!(preview["affectedMemoryCount"], 2);

    storage
        .apply_tag_mutation(
            &sources,
            "canonical",
            None,
            preview_token(&preview),
            "tag_rename",
            "explicit cross-scope rename",
        )
        .unwrap();
    assert_eq!(
        storage.get_node(&user.id).unwrap().unwrap().tags,
        vec!["canonical"]
    );
    assert_eq!(
        storage.get_node(&project.id).unwrap().unwrap().tags,
        vec!["canonical"]
    );
}

#[test]
fn list_tag_operations_is_not_buried_by_later_merge_rows() {
    let storage = create_test_storage();
    ingest_tagged_in_scope(&storage, "user", "tag burial fixture", &["old"]);
    let sources = vec!["old".to_string()];
    let preview = storage
        .preview_tag_mutation(&sources, "new", Some("user"))
        .unwrap();
    let operation = storage
        .apply_tag_mutation(
            &sources,
            "new",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "must remain listed after later merge rows",
        )
        .unwrap();

    {
        let writer = storage.writer.lock().unwrap();
        for index in 0..20 {
            writer
                    .execute(
                        "INSERT INTO merge_operations
                            (id, plan_id, op_type, status, created_at, reverted_at, reverts_op_id,
                             survivor_id, affected_ids, confidence, signals, reason, undo_payload)
                         VALUES (?1, NULL, 'merge', 'applied', ?2, NULL, NULL, NULL, '[]', NULL, NULL, 'later merge', '{}')",
                        params![
                            format!("merge-later-{index:02}"),
                            "2099-01-01T00:00:00+00:00"
                        ],
                    )
                    .unwrap();
        }
    }

    let mixed = storage.list_merge_operations(20).unwrap();
    assert_eq!(mixed.len(), 20);
    assert!(
        mixed
            .iter()
            .all(|operation| operation.op_type != "tag_rename"),
        "the mixed window fills with later merge rows"
    );
    let tags = storage.list_tag_operations(50, None).unwrap();
    assert_eq!(tags.len(), 1);
    assert_eq!(tags[0].id, operation.id);
    let scoped = storage.list_tag_operations(50, Some("user")).unwrap();
    assert_eq!(scoped.len(), 1);
}

#[test]
fn tag_merge_normalizes_sources_and_rejects_stale_preview_or_undo_conflict() {
    let storage = create_test_storage();
    let node = ingest_tagged_in_scope(
        &storage,
        "user",
        "multi source tag merge",
        &["keep", "beta", "alpha", "target", "target", "tail"],
    );
    // Duplicate sources dedupe; matching is byte-exact (padded variants
    // are separate, reachable keys — see the padded-source test).
    let sources = vec!["beta".to_string(), "alpha".to_string(), "alpha".to_string()];
    let preview = storage
        .preview_tag_mutation(&sources, "target", Some("user"))
        .unwrap();

    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET tags = ?1 WHERE id = ?2",
                params![
                    serde_json::json!(["keep", "beta", "alpha", "drift"]).to_string(),
                    &node.id
                ],
            )
            .unwrap();
    }
    let stale_error = storage
        .apply_tag_mutation(
            &sources,
            "target",
            Some("user"),
            preview_token(&preview),
            "tag_merge",
            "merge aliases",
        )
        .unwrap_err();
    assert!(stale_error.to_string().contains("stale"));
    assert!(storage.list_merge_operations(20).unwrap().is_empty());

    let fresh = storage
        .preview_tag_mutation(&sources, "target", Some("user"))
        .unwrap();
    let operation = storage
        .apply_tag_mutation(
            &sources,
            "target",
            Some("user"),
            preview_token(&fresh),
            "tag_merge",
            "merge aliases",
        )
        .unwrap();
    assert_eq!(
        storage.get_node(&node.id).unwrap().unwrap().tags,
        vec!["keep", "target", "drift"]
    );

    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET tags = ?1 WHERE id = ?2",
                params![
                    serde_json::json!(["keep", "target", "later-edit"]).to_string(),
                    &node.id
                ],
            )
            .unwrap();
    }
    let conflict = storage.undo_tag_mutation(&operation.id).unwrap_err();
    assert!(conflict.to_string().contains("conflict"));
    assert_eq!(
        storage
            .get_merge_operation(&operation.id)
            .unwrap()
            .unwrap()
            .status,
        "applied",
        "failed undo must not mark the original operation reverted"
    );
}

#[test]
fn tag_mutation_validation_and_malformed_json_fail_without_partial_writes() {
    let storage = create_test_storage();
    let first = ingest_tagged_in_scope(&storage, "user", "first atomic row", &["old"]);
    let second = ingest_tagged_in_scope(&storage, "user", "second atomic row", &["old"]);
    assert!(
        storage
            .preview_tag_mutation(&["same".into()], "same", Some("user"))
            .is_err()
    );
    assert!(
        storage
            .preview_tag_mutation(&["bad\ncontrol".into()], "new", Some("user"))
            .is_err()
    );

    let sources = vec!["old".to_string()];
    let preview = storage
        .preview_tag_mutation(&sources, "new", Some("user"))
        .unwrap();
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET tags = 'not-json' WHERE id = ?1",
                params![&second.id],
            )
            .unwrap();
    }
    let error = storage
        .apply_tag_mutation(
            &sources,
            "new",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "atomic malformed-json test",
        )
        .unwrap_err();
    assert!(error.to_string().contains("invalid tags JSON"));
    assert_eq!(
        storage.get_node(&first.id).unwrap().unwrap().tags,
        vec!["old"]
    );
    assert!(storage.list_merge_operations(20).unwrap().is_empty());

    let valid_storage = create_test_storage();
    ingest_tagged_in_scope(&valid_storage, "user", "no match", &["other"]);
    let empty = valid_storage
        .preview_tag_mutation(&sources, "new", Some("user"))
        .unwrap();
    assert_eq!(empty["affectedMemoryCount"], 0);
    assert!(
        valid_storage
            .apply_tag_mutation(
                &sources,
                "new",
                Some("user"),
                preview_token(&empty),
                "tag_rename",
                "must reject no-op",
            )
            .is_err()
    );
    assert!(
        valid_storage
            .apply_tag_mutation(
                &["other".into()],
                "new",
                Some("user"),
                "tag-plan-v1:wrong",
                "tag_rename",
                "",
            )
            .is_err(),
        "a nonempty audit reason is mandatory"
    );
}

#[test]
fn tag_mutation_rejects_secret_shaped_persistent_fields_without_side_effects() {
    let storage = create_test_storage();
    let node = ingest_tagged_in_scope(&storage, "user", "secret policy fixture", &["old"]);
    let source = vec!["old".to_string()];
    let credential = format!("ghp_{}", "a".repeat(36));

    let target_error = storage
        .preview_tag_mutation(&source, &credential, Some("user"))
        .unwrap_err()
        .to_string();
    assert!(target_error.contains("probable credential"));
    assert!(!target_error.contains(&credential));

    // SOURCE tags are exact-match lookup keys for values that already
    // exist in the store, so a secret-shaped source is accepted (it must
    // stay renameable AWAY); only newly persisted fields are rejected.
    let source_preview = storage
        .preview_tag_mutation(std::slice::from_ref(&credential), "new", Some("user"))
        .unwrap();
    assert_eq!(source_preview["affectedMemoryCount"], 0);

    let safe_preview = storage
        .preview_tag_mutation(&source, "new", Some("user"))
        .unwrap();
    let reason_error = storage
        .apply_tag_mutation(
            &source,
            "new",
            Some("user"),
            preview_token(&safe_preview),
            "tag_rename",
            &credential,
        )
        .unwrap_err()
        .to_string();
    assert!(reason_error.contains("probable credential"));
    assert!(!reason_error.contains(&credential));
    assert_eq!(
        storage.get_node(&node.id).unwrap().unwrap().tags,
        vec!["old"]
    );
    assert!(storage.list_merge_operations(20).unwrap().is_empty());
}

#[test]
fn tag_mutation_row_and_audit_limits_fail_before_writes() {
    let storage = create_test_storage();
    let nodes: Vec<_> = (0..3)
        .map(|index| {
            ingest_tagged_in_scope(
                &storage,
                "user",
                &format!("row limit fixture {index}"),
                &["old"],
            )
        })
        .collect();
    let source = vec!["old".to_string()];
    let preview = storage
        .preview_tag_mutation(&source, "new", Some("user"))
        .unwrap();

    let row_limit_error = storage
        .apply_tag_mutation_with_limits(
            &source,
            "new",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "row limit fixture",
            2,
            MAX_TAG_MUTATION_AUDIT_BYTES,
        )
        .unwrap_err()
        .to_string();
    assert!(row_limit_error.contains("more than 2 memories"));
    assert!(storage.list_merge_operations(20).unwrap().is_empty());
    for node in &nodes {
        assert_eq!(
            storage.get_node(&node.id).unwrap().unwrap().tags,
            vec!["old"]
        );
    }

    let applied = storage
        .apply_tag_mutation_with_limits(
            &source,
            "new",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "exact row limit fixture",
            3,
            MAX_TAG_MUTATION_AUDIT_BYTES,
        )
        .unwrap();
    assert_eq!(applied.affected_ids.len(), 3);

    let audit_storage = create_test_storage();
    let audit_node = ingest_tagged_in_scope(
        &audit_storage,
        "user",
        "audit payload limit fixture",
        &["old"],
    );
    let audit_preview = audit_storage
        .preview_tag_mutation(&source, "new", Some("user"))
        .unwrap();
    let audit_limit_error = audit_storage
        .apply_tag_mutation_with_limits(
            &source,
            "new",
            Some("user"),
            preview_token(&audit_preview),
            "tag_rename",
            "audit payload limit fixture",
            MAX_TAG_MUTATION_MEMORIES,
            1,
        )
        .unwrap_err()
        .to_string();
    assert!(audit_limit_error.contains("1-byte limit"));
    assert_eq!(
        audit_storage
            .get_node(&audit_node.id)
            .unwrap()
            .unwrap()
            .tags,
        vec!["old"]
    );
    assert!(audit_storage.list_merge_operations(20).unwrap().is_empty());
}

#[test]
fn hygiene_snapshot_covers_more_than_five_hundred_without_full_content() {
    let storage = create_test_storage();
    let now = Utc::now().to_rfc3339();
    {
        let mut writer = storage.writer.lock().unwrap();
        let tx = writer.transaction().unwrap();
        for index in 0..501 {
            tx.execute(
                "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed, tags, scope)
                     VALUES (?1, ?2, 'fact', ?3, ?3, ?3, ?4, 'user')",
                params![
                    format!("hygiene-{index:04}"),
                    format!("bounded content {index}"),
                    &now,
                    serde_json::json!(["bulk"]).to_string(),
                ],
            )
            .unwrap();
        }
        tx.commit().unwrap();
    }
    let snapshot = storage.hygiene_snapshot(Some("user")).unwrap();
    assert_eq!(snapshot.nodes.len(), 501);
    assert!(
        snapshot
            .nodes
            .iter()
            .all(|row| row.content_preview.len() <= 240)
    );
    assert!(snapshot.nodes.iter().all(|row| row.never_accessed));
    assert!(snapshot.nodes.iter().all(|row| !row.access_unknown));
    assert_eq!(snapshot.malformed_tag_rows, 0);
    assert_eq!(snapshot.defaulted_retention_rows, 0);
}

#[test]
fn hygiene_snapshot_distinguishes_unknown_pruned_access_from_never_accessed() {
    let storage = create_test_storage();
    let now = Utc::now();
    let fresh = now.to_rfc3339();
    let before_log_window = (now - Duration::days(ACCESS_LOG_RETENTION_DAYS + 110)).to_rfc3339();
    {
        let writer = storage.writer.lock().unwrap();
        // Heavily used old memory: its log rows were pruned, but the
        // durable retrieval counter survives on the node row.
        writer
            .execute(
                "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope, times_retrieved)
                     VALUES ('old-used', 'used before the log window', 'fact',
                             ?1, ?1, ?1, '[]', 'user', 5)",
                params![&before_log_window],
            )
            .unwrap();
        // Old memory with no evidence either way: pruning makes its
        // history unknowable, so it must not be claimed never-accessed.
        writer
            .execute(
                "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope)
                     VALUES ('old-unknown', 'predates the log window', 'fact',
                             ?1, ?1, ?1, '[]', 'user')",
                params![&before_log_window],
            )
            .unwrap();
        // Fresh memory with no accesses: the retained log is complete
        // evidence for its whole lifetime, so never-accessed is provable.
        writer
            .execute(
                "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope)
                     VALUES ('fresh-never', 'created inside the log window', 'fact',
                             ?1, ?1, ?1, '[]', 'user')",
                params![&fresh],
            )
            .unwrap();
    }
    let shown = ingest_tagged_in_scope(&storage, "user", "fresh shown row", &[]);
    storage.record_batch_retrieval(&[&shown.id]).unwrap();

    let snapshot = storage.hygiene_snapshot(Some("user")).unwrap();
    let by_id = |id: &str| {
        snapshot
            .nodes
            .iter()
            .find(|node| node.id == id)
            .unwrap_or_else(|| panic!("snapshot row {id}"))
    };
    let old_used = by_id("old-used");
    assert!(
        !old_used.never_accessed && !old_used.access_unknown,
        "a durable retrieval counter proves past access even after log pruning"
    );
    let old_unknown = by_id("old-unknown");
    assert!(
        !old_unknown.never_accessed,
        "a pre-window row without counters must never be claimed never-accessed"
    );
    assert!(old_unknown.access_unknown);
    let fresh_never = by_id("fresh-never");
    assert!(fresh_never.never_accessed && !fresh_never.access_unknown);
    let shown_row = by_id(&shown.id);
    assert!(!shown_row.never_accessed && !shown_row.access_unknown);
}

#[test]
fn hygiene_snapshot_tolerates_malformed_and_null_legacy_rows() {
    let storage = create_test_storage();
    for index in 0..3 {
        ingest_tagged_in_scope(&storage, "user", &format!("clean row {index}"), &["clean"]);
    }
    let now = Utc::now().to_rfc3339();
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope)
                     VALUES ('bad-json', 'hand-edited tags', 'fact', ?1, ?1, ?1,
                             'not-json', 'user')",
                params![&now],
            )
            .unwrap();
        writer
            .execute(
                "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope, retention_strength)
                     VALUES ('null-tags', 'hand-edited null row', 'fact', ?1, ?1, ?1,
                             NULL, 'user', NULL)",
                params![&now],
            )
            .unwrap();
    }

    let snapshot = storage.hygiene_snapshot(Some("user")).unwrap();
    assert_eq!(
        snapshot.nodes.len(),
        5,
        "aggregates must cover every row, corrupted ones included"
    );
    assert_eq!(snapshot.malformed_tag_rows, 2);
    assert_eq!(
        snapshot.malformed_tag_row_ids,
        vec!["bad-json".to_string(), "null-tags".to_string()]
    );
    assert!(!snapshot.malformed_tag_row_ids_truncated);
    assert_eq!(snapshot.defaulted_retention_rows, 1);
    let null_row = snapshot
        .nodes
        .iter()
        .find(|node| node.id == "null-tags")
        .unwrap();
    assert!(null_row.tags.is_empty());
    assert_eq!(null_row.retention_strength, 1.0);
}

#[test]
fn tag_apply_and_audit_roll_back_together_on_injected_failure() {
    let storage = create_test_storage();
    let first = ingest_tagged_in_scope(&storage, "user", "atomic pair first", &["old"]);
    let second = ingest_tagged_in_scope(&storage, "user", "atomic pair second", &["old"]);
    let sources = vec!["old".to_string()];
    let preview = storage
        .preview_tag_mutation(&sources, "new", Some("user"))
        .unwrap();

    FAIL_TAG_MUTATION_BEFORE_AUDIT.with(|flag| flag.set(true));
    let error = storage
        .apply_tag_mutation(
            &sources,
            "new",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "single transaction fail point",
        )
        .unwrap_err();
    FAIL_TAG_MUTATION_BEFORE_AUDIT.with(|flag| flag.set(false));
    assert!(error.to_string().contains("fail point"));
    // The failure fired AFTER every row UPDATE: rollback must restore all
    // tag arrays AND leave no audit row, proving one shared transaction.
    assert_eq!(
        storage.get_node(&first.id).unwrap().unwrap().tags,
        vec!["old"]
    );
    assert_eq!(
        storage.get_node(&second.id).unwrap().unwrap().tags,
        vec!["old"]
    );
    assert!(storage.list_merge_operations(20).unwrap().is_empty());
    assert!(storage.list_tag_operations(20, None).unwrap().is_empty());

    // Disarmed, the same untouched preview applies normally.
    let applied = storage
        .apply_tag_mutation(
            &sources,
            "new",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "single transaction fail point disarmed",
        )
        .unwrap();
    assert_eq!(applied.affected_ids.len(), 2);
    assert_eq!(
        storage.get_node(&first.id).unwrap().unwrap().tags,
        vec!["new"]
    );
}

#[test]
fn tag_vocabulary_skips_and_counts_overlong_stored_tags() {
    let storage = create_test_storage();
    ingest_tagged_in_scope(&storage, "user", "normal tag row", &["normal"]);
    let overlong = "x".repeat(201);
    ingest_tagged_in_scope(&storage, "user", "overlong tag row", &[&overlong]);

    let vocabulary = storage.tag_vocabulary(Some("user")).unwrap();
    assert_eq!(vocabulary.tags, vec!["normal".to_string()]);
    assert_eq!(vocabulary.skipped_overlong, 1);
}

#[test]
fn overlong_source_tags_can_be_renamed_away_end_to_end() {
    let storage = create_test_storage();
    let overlong = "y".repeat(250);
    let node = ingest_tagged_in_scope(&storage, "user", "overlong rename fixture", &[&overlong]);

    let sources = vec![overlong.clone()];
    let preview = storage
        .preview_tag_mutation(&sources, "short-tag", Some("user"))
        .unwrap();
    assert_eq!(preview["affectedMemoryCount"], 1);
    storage
        .apply_tag_mutation(
            &sources,
            "short-tag",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "repair an overlong stored tag",
        )
        .unwrap();
    assert_eq!(
        storage.get_node(&node.id).unwrap().unwrap().tags,
        vec!["short-tag"]
    );
    let vocabulary = storage.tag_vocabulary(Some("user")).unwrap();
    assert_eq!(vocabulary.skipped_overlong, 0, "the overlong tag is gone");
}

#[test]
fn tag_vocabulary_rejects_more_than_ten_thousand_tags() {
    let storage = create_test_storage();
    let tags: Vec<String> = (0..10_001)
        .map(|index| format!("bulk-{index:05}"))
        .collect();
    let now = Utc::now().to_rfc3339();
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope)
                     VALUES ('vocab-bound', 'vocabulary bound fixture', 'fact', ?1, ?1, ?1,
                             ?2, 'user')",
                params![&now, serde_json::to_string(&tags).unwrap()],
            )
            .unwrap();
    }
    let error = storage
        .tag_vocabulary(Some("user"))
        .unwrap_err()
        .to_string();
    assert!(error.contains("exceeds the 10000-tag"));
}

#[test]
fn padded_source_tags_are_reachable_byte_exact() {
    let storage = create_test_storage();
    let padded = ingest_tagged_in_scope(&storage, "user", "padded tag fixture", &[" prix-six"]);
    let unpadded = ingest_tagged_in_scope(&storage, "user", "unpadded fixture", &["prix-six"]);

    // The padded stored variant is addressable exactly as stored; the
    // trimmed TARGET merges it into the canonical spelling.
    let sources = vec![" prix-six".to_string()];
    let preview = storage
        .preview_tag_mutation(&sources, "prix-six", Some("user"))
        .unwrap();
    assert_eq!(preview["affectedMemoryCount"], 1);
    storage
        .apply_tag_mutation(
            &sources,
            "prix-six",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "collapse a whitespace-padded tag variant",
        )
        .unwrap();
    assert_eq!(
        storage.get_node(&padded.id).unwrap().unwrap().tags,
        vec!["prix-six"]
    );

    // A trimmed source still matches unpadded stored tags as before.
    let trimmed_sources = vec!["prix-six".to_string()];
    let trimmed_preview = storage
        .preview_tag_mutation(&trimmed_sources, "grand-prix", Some("user"))
        .unwrap();
    assert_eq!(trimmed_preview["affectedMemoryCount"], 2);
    storage
        .apply_tag_mutation(
            &trimmed_sources,
            "grand-prix",
            Some("user"),
            preview_token(&trimmed_preview),
            "tag_rename",
            "rename the canonical spelling",
        )
        .unwrap();
    assert_eq!(
        storage.get_node(&unpadded.id).unwrap().unwrap().tags,
        vec!["grand-prix"]
    );
}

#[test]
fn secret_shaped_stored_tags_can_be_renamed_away() {
    let storage = create_test_storage();
    let credential = format!("ghp_{}", "b".repeat(36));
    let node = storage
        .ingest_with_secret_policy(
            IngestInput {
                content: "explicit-allow credential tag fixture".to_string(),
                tags: vec![credential.clone(), "keep".to_string()],
                ..Default::default()
            },
            SecretPolicy::AllowExplicitly,
        )
        .unwrap();

    let sources = vec![credential.clone()];
    let preview = storage
        .preview_tag_mutation(&sources, "rotated-token-reference", Some("user"))
        .unwrap();
    assert_eq!(preview["affectedMemoryCount"], 1);
    storage
        .apply_tag_mutation(
            &sources,
            "rotated-token-reference",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "replace a credential-shaped tag with a safe reference",
        )
        .unwrap();
    assert_eq!(
        storage.get_node(&node.id).unwrap().unwrap().tags,
        vec!["rotated-token-reference", "keep"]
    );
}

#[test]
fn scoped_tag_operation_listing_includes_all_scopes_operations() {
    let storage = create_test_storage();
    ingest_tagged_in_scope(&storage, "user", "scoped audit row", &["scoped-old"]);
    ingest_tagged_in_scope(&storage, "user", "shared audit row", &["shared-old"]);
    ingest_tagged_in_scope(&storage, "project-b", "other scope row", &["shared-old"]);

    let scoped_sources = vec!["scoped-old".to_string()];
    let scoped_preview = storage
        .preview_tag_mutation(&scoped_sources, "scoped-new", Some("user"))
        .unwrap();
    let scoped_op = storage
        .apply_tag_mutation(
            &scoped_sources,
            "scoped-new",
            Some("user"),
            preview_token(&scoped_preview),
            "tag_rename",
            "scoped audit fixture",
        )
        .unwrap();

    let shared_sources = vec!["shared-old".to_string()];
    let shared_preview = storage
        .preview_tag_mutation(&shared_sources, "shared-new", None)
        .unwrap();
    let shared_op = storage
        .apply_tag_mutation(
            &shared_sources,
            "shared-new",
            None,
            preview_token(&shared_preview),
            "tag_rename",
            "all-scopes audit fixture",
        )
        .unwrap();

    let user_view = storage.list_tag_operations(50, Some("user")).unwrap();
    let user_ids: Vec<&str> = user_view.iter().map(|op| op.id.as_str()).collect();
    assert!(
        user_ids.contains(&scoped_op.id.as_str()) && user_ids.contains(&shared_op.id.as_str()),
        "a scope's audit must show all-scopes operations that rewrote it"
    );

    let other_view = storage.list_tag_operations(50, Some("project-b")).unwrap();
    assert_eq!(other_view.len(), 1);
    assert_eq!(other_view[0].id, shared_op.id);

    let all_view = storage.list_tag_operations(50, None).unwrap();
    assert_eq!(all_view.len(), 2);
}

#[test]
fn test_ingest_and_get() {
    let storage = create_test_storage();

    let input = IngestInput {
        content: "Test memory content".to_string(),
        node_type: "fact".to_string(),
        ..Default::default()
    };

    let node = storage.ingest(input).unwrap();
    assert!(!node.id.is_empty());
    assert_eq!(node.content, "Test memory content");

    let retrieved = storage.get_node(&node.id).unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().content, "Test memory content");
}

// =====================================================================
// Smart-ingest bitemporal validity gates (issue #156)
// =====================================================================

/// Marker-keyed test embedder: contents sharing the "alpha" marker embed
/// to the same axis (cosine similarity 1.0) and everything else lands on
/// the orthogonal axis, so gate decisions are fully controlled by content.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
struct MarkerEmbedder;

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
impl crate::embedder::EmbedderSend for MarkerEmbedder {
    async fn embed(&self, text: &str) -> crate::embedder::EmbedderResult<Vec<f32>> {
        Ok(if text.contains("alpha") {
            vec![1.0, 0.0]
        } else {
            vec![0.0, 1.0]
        })
    }
    fn model_name(&self) -> &str {
        "marker-gate-test-runner"
    }
    fn dimension(&self) -> usize {
        2
    }
    fn model_hash(&self) -> String {
        "0".repeat(64)
    }
    async fn embed_batch(&self, texts: &[&str]) -> crate::embedder::EmbedderResult<Vec<Vec<f32>>> {
        let mut result = Vec::with_capacity(texts.len());
        for text in texts {
            result.push(if text.contains("alpha") {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            });
        }
        Ok(result)
    }
}

/// Regression: a purge (writer lock, then vector-index lock) racing a
/// profile activation (index lock, then writer lock) deadlocked the process
/// before the Sep 2026 hardening pass because `purge_node` kept its writer
/// guard alive while it waited on the index. Both paths must run to
/// completion under a watchdog.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn purge_and_profile_activation_do_not_deadlock() {
    use std::sync::mpsc;
    use std::time::Duration;

    let dir = tempfile::tempdir().unwrap();
    let storage = std::sync::Arc::new(storage_with_marker_gate_runtime(&dir));
    let ids: Vec<String> = (0..48)
        .map(|i| {
            storage
                .ingest(IngestInput {
                    content: format!("alpha purge race memory number {i}"),
                    ..Default::default()
                })
                .unwrap()
                .id
        })
        .collect();
    let active = storage
        .active_embedding_profile()
        .unwrap()
        .expect("marker fixture activates a profile");

    let (done_tx, done_rx) = mpsc::channel::<&'static str>();
    let purger = {
        let storage = std::sync::Arc::clone(&storage);
        let done_tx = done_tx.clone();
        std::thread::spawn(move || {
            for id in ids {
                storage.purge_node(&id, Some("lock-order race")).unwrap();
            }
            let _ = done_tx.send("purge");
        })
    };
    let activator = {
        let storage = std::sync::Arc::clone(&storage);
        std::thread::spawn(move || {
            for _ in 0..48 {
                storage
                    .activate_embedding_profile(&active.profile_id)
                    .unwrap();
            }
            let _ = done_tx.send("activate");
        })
    };
    for _ in 0..2 {
        done_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("purge vs profile activation deadlocked (writer->index vs index->writer)");
    }
    purger.join().unwrap();
    activator.join().unwrap();
}

/// Install, promote, activate, and attach a verified 2-dimensional marker
/// profile so smart-ingest gate decisions run end to end without a model
/// download. Mirrors the lifecycle install/evaluate/migrate/activate flow;
/// the Ready promotion is applied directly to the persisted manifest since
/// the process-local registry is private to the lifecycle module.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
fn storage_with_marker_gate_runtime(dir: &tempfile::TempDir) -> Storage {
    use crate::embedding::{
        ChunkingStrategy, EmbeddingDevice, EmbeddingEvaluationSummary, EmbeddingNormalization,
        EmbeddingProfile, EmbeddingProfileLifecycle, EmbeddingRuntimeMetadata, EncodingTemplate,
        ModelArtifactHash, VerifiedLocalArtifact,
    };
    use sha2::{Digest, Sha256};

    let storage = create_test_storage_at(dir, "marker-gate.db");
    let artifact_bytes: &[u8] = b"marker gate test artifact";
    std::fs::write(dir.path().join("runner.bin"), artifact_bytes).unwrap();
    let artifact = ModelArtifactHash::sha256(
        "runner.bin",
        format!("{:x}", Sha256::digest(artifact_bytes)),
    );
    let profile = EmbeddingProfile {
        profile_id: EmbeddingProfileId::new("marker-gate-test-2d").unwrap(),
        display_name: "Marker Gate Test Profile".to_string(),
        model_id: "test/marker-gate".to_string(),
        immutable_model_revision: "immutable-test-revision".to_string(),
        verified_model_artifact_hashes: vec![artifact.clone()],
        runtime_backend: EmbeddingRuntimeBackend::FastembedCandle,
        embedding_dimension: 2,
        normalization_method: EmbeddingNormalization::L2,
        document_encoding_template: EncodingTemplate::Raw,
        query_encoding_template: EncodingTemplate::Raw,
        maximum_token_limit: 64,
        chunking_strategy: ChunkingStrategy::WholeDocument,
        created_at: Utc::now(),
    };
    let artifacts = vec![VerifiedLocalArtifact::from_root(artifact, dir.path()).unwrap()];
    let source = EmbeddingProfileId::new("nomic-v1.5-legacy-raw-256").unwrap();
    let lifecycle = EmbeddingProfileLifecycle::new(&storage);
    let mut manifest = lifecycle
        .install_verified(
            profile.clone(),
            &artifacts,
            EmbeddingRuntimeMetadata {
                backend: EmbeddingRuntimeBackend::FastembedCandle,
                device: EmbeddingDevice::Cpu,
                runtime_version: "test".to_string(),
                initialized_at: Utc::now(),
                local_only: true,
            },
            Arc::new(MarkerEmbedder),
        )
        .unwrap();
    manifest.state = EmbeddingProfileState::Ready;
    manifest.evaluation = Some(EmbeddingEvaluationSummary {
        evaluation_id: Uuid::new_v4(),
        compared_against: source.clone(),
        completed_at: Utc::now(),
        corpus_size: 0,
        recall_at_5: None,
        recall_at_10: None,
        ndcg_at_10: None,
        exact_match_preservation: None,
        false_positive_rate: None,
        p50_query_latency_ms: None,
        p95_query_latency_ms: None,
        ingestion_throughput_per_second: None,
        report_hash: "1".repeat(64),
    });
    storage.save_embedding_profile_manifest(&manifest).unwrap();
    lifecycle
        .migrate_registered(&profile.profile_id, &source, None, None)
        .unwrap();
    storage
        .activate_embedding_profile(&profile.profile_id)
        .unwrap();
    lifecycle
        .attach_registered_active_profile(&profile.profile_id)
        .unwrap();
    storage
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn inferred_as_of_validity_never_mutates_an_existing_nodes_window() {
    let dir = tempdir().unwrap();
    let storage = storage_with_marker_gate_runtime(&dir);
    let explicit_from = Utc::now() - Duration::days(60);
    let explicit_until = Utc::now() + Duration::days(300);
    let target = storage
        .ingest(IngestInput {
            content: "alpha deployment policy is retries with backoff".to_string(),
            valid_from: Some(explicit_from),
            valid_until: Some(explicit_until),
            ..Default::default()
        })
        .unwrap();

    // Near-identical content whose only validity is a prose-inferred
    // "as of" date must reinforce WITHOUT touching the target's window.
    let result = storage
        .smart_ingest(IngestInput {
            content: "alpha deployment policy is retries with backoff, as of last review"
                .to_string(),
            valid_from: Some(Utc::now() - Duration::days(10)),
            validity_inferred: true,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(result.decision, "reinforce");
    assert_eq!(result.node.id, target.id);

    let node = storage.get_node(&target.id).unwrap().unwrap();
    assert_eq!(
        node.valid_from.map(|value| value.to_rfc3339()),
        Some(explicit_from.to_rfc3339()),
        "an inferred date must not move valid_from"
    );
    assert_eq!(
        node.valid_until.map(|value| value.to_rfc3339()),
        Some(explicit_until.to_rfc3339()),
        "an inferred date must not clear or move valid_until"
    );
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn inferred_as_of_must_not_resurrect_an_expired_similar_node() {
    let dir = tempdir().unwrap();
    let storage = storage_with_marker_gate_runtime(&dir);
    let expired_from = Utc::now() - Duration::days(90);
    let expired_until = Utc::now() - Duration::days(7);
    let target = storage
        .ingest(IngestInput {
            content: "alpha deployment policy is retries with backoff".to_string(),
            valid_from: Some(expired_from),
            valid_until: Some(expired_until),
            ..Default::default()
        })
        .unwrap();
    assert!(
        !storage
            .get_node(&target.id)
            .unwrap()
            .unwrap()
            .is_currently_valid(),
        "fixture must start expired"
    );

    // Live #170 failure: inferred validFrom + PEG update used to
    // REPLACE valid_until with NULL and un-expire the similar row.
    let result = storage
        .smart_ingest(IngestInput {
            content: "alpha deployment policy is retries with backoff as of 2026-03-04".to_string(),
            valid_from: Some(Utc::now() - Duration::days(10)),
            validity_inferred: true,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(result.decision, "reinforce");
    assert_eq!(result.node.id, target.id);

    let node = storage.get_node(&target.id).unwrap().unwrap();
    assert_eq!(
        node.valid_from.map(|value| value.to_rfc3339()),
        Some(expired_from.to_rfc3339()),
        "inferred as-of must not move the expired node's valid_from"
    );
    assert_eq!(
        node.valid_until.map(|value| value.to_rfc3339()),
        Some(expired_until.to_rfc3339()),
        "inferred as-of must not un-expire the similar node"
    );
    assert!(!node.is_currently_valid());
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn explicit_valid_from_on_reinforce_updates_without_clearing_valid_until() {
    let dir = tempdir().unwrap();
    let storage = storage_with_marker_gate_runtime(&dir);
    let explicit_from = Utc::now() - Duration::days(60);
    let explicit_until = Utc::now() + Duration::days(300);
    let target = storage
        .ingest(IngestInput {
            content: "alpha deployment policy is retries with backoff".to_string(),
            valid_from: Some(explicit_from),
            valid_until: Some(explicit_until),
            ..Default::default()
        })
        .unwrap();

    // An explicit caller-supplied valid_from still updates the window, and
    // omitting valid_until must preserve the stored bound (merge, not
    // replace).
    let new_from = Utc::now() - Duration::days(10);
    let result = storage
        .smart_ingest(IngestInput {
            content: "alpha deployment policy is retries with backoff".to_string(),
            valid_from: Some(new_from),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(result.decision, "reinforce");

    let node = storage.get_node(&target.id).unwrap().unwrap();
    assert_eq!(
        node.valid_from.map(|value| value.to_rfc3339()),
        Some(new_from.to_rfc3339()),
        "explicit valid_from must update the window"
    );
    assert_eq!(
        node.valid_until.map(|value| value.to_rfc3339()),
        Some(explicit_until.to_rfc3339()),
        "an explicit valid_from-only update must not NULL valid_until"
    );
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn create_path_still_stamps_inferred_validity_on_the_new_node() {
    let dir = tempdir().unwrap();
    let storage = storage_with_marker_gate_runtime(&dir);
    let inferred_from = Utc::now() - Duration::days(10);
    let result = storage
        .smart_ingest(IngestInput {
            content: "beta service quota was raised".to_string(),
            valid_from: Some(inferred_from),
            validity_inferred: true,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(result.decision, "create");
    assert!(result.auto_closed_until.is_none());
    let node = storage.get_node(&result.node.id).unwrap().unwrap();
    assert_eq!(
        node.valid_from.map(|value| value.to_rfc3339()),
        Some(inferred_from.to_rfc3339()),
        "the issue-156 feature: inferred validity stamps the NEW node"
    );
    assert!(node.valid_until.is_none());
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn older_dated_claim_after_newer_fact_is_created_with_a_closed_window() {
    let dir = tempdir().unwrap();
    let storage = storage_with_marker_gate_runtime(&dir);
    let newer_from = Utc::now() - Duration::days(30);
    storage
        .ingest(IngestInput {
            content: "alpha runtime version is 5.0".to_string(),
            valid_from: Some(newer_from),
            ..Default::default()
        })
        .unwrap();

    // The stale snapshot loses every candidate to the newer current fact,
    // so it is created as a CLOSED historical claim, never an open one.
    let result = storage
        .smart_ingest(IngestInput {
            content: "alpha runtime version is 4.2".to_string(),
            valid_from: Some(Utc::now() - Duration::days(180)),
            validity_inferred: true,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(result.decision, "create");
    assert_eq!(
        result.auto_closed_until.map(|value| value.to_rfc3339()),
        Some(newer_from.to_rfc3339()),
        "the new node closes exactly where the newer fact begins"
    );
    assert!(result.reason.contains("Closed validity at"));
    let node = storage.get_node(&result.node.id).unwrap().unwrap();
    assert_eq!(
        node.valid_until.map(|value| value.to_rfc3339()),
        Some(newer_from.to_rfc3339())
    );
    assert!(!node.is_currently_valid());

    // Reverse order on a fresh store: the newer dated claim arriving
    // second must NOT auto-close anything. That is this half's subject.
    //
    // It previously also asserted `reinforce`, which encoded a defect. The
    // pair here ("version is 4.2" against "version is 5.0") is a mutually
    // exclusive VALUE conflict, the same shape as PostgreSQL 14 -> 16, and
    // the write-path detector could not see it: short numeric tokens were
    // dropped by the substantive-word length filter, so the two texts
    // looked identical and the gate reinforced on similarity alone. The
    // effect was that telling Vestige "the version is now 5.0" discarded
    // that update and made it believe 4.2 MORE strongly. The gate now
    // keeps both claims. See advanced::contradiction.
    let dir = tempdir().unwrap();
    let storage = storage_with_marker_gate_runtime(&dir);
    let target = storage
        .ingest(IngestInput {
            content: "alpha runtime version is 4.2".to_string(),
            valid_from: Some(Utc::now() - Duration::days(180)),
            ..Default::default()
        })
        .unwrap();
    let result = storage
        .smart_ingest(IngestInput {
            content: "alpha runtime version is 5.0".to_string(),
            valid_from: Some(newer_from),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(
        result.decision, "create",
        "a version change is a value conflict, not a reinforcement"
    );
    assert_ne!(
        result.node.id, target.id,
        "the superseded 4.2 claim must not be overwritten in place"
    );
    assert!(
        result.auto_closed_until.is_none(),
        "a newer dated claim arriving second closes nothing"
    );
    assert_eq!(
        result.node.valid_from.map(|value| value.to_rfc3339()),
        Some(newer_from.to_rfc3339()),
        "the new node carries its own validity"
    );
    // The older claim survives intact, still open, still retrievable.
    let previous = storage.get_node(&target.id).unwrap().unwrap();
    assert!(previous.valid_until.is_none());
    assert!(previous.content.contains("4.2"));
}

#[test]
fn update_node_validity_merges_bounds_and_validates_the_effective_window() {
    let storage = create_test_storage();
    let node = storage
        .ingest(IngestInput {
            content: "validity merge fixture".to_string(),
            ..Default::default()
        })
        .unwrap();
    let from = Utc::now() - Duration::days(10);
    let until = Utc::now() + Duration::days(10);
    storage
        .update_node_validity(&node.id, Some(from), Some(until))
        .unwrap();

    // Updating only valid_from must not clear the stored valid_until.
    let new_from = Utc::now() - Duration::days(5);
    storage
        .update_node_validity(&node.id, Some(new_from), None)
        .unwrap();
    let stored = storage.get_node(&node.id).unwrap().unwrap();
    assert_eq!(
        stored.valid_from.map(|value| value.to_rfc3339()),
        Some(new_from.to_rfc3339())
    );
    assert_eq!(
        stored.valid_until.map(|value| value.to_rfc3339()),
        Some(until.to_rfc3339())
    );

    // Updating only valid_until must not clear the stored valid_from.
    let new_until = Utc::now() + Duration::days(20);
    storage
        .update_node_validity(&node.id, None, Some(new_until))
        .unwrap();
    let stored = storage.get_node(&node.id).unwrap().unwrap();
    assert_eq!(
        stored.valid_from.map(|value| value.to_rfc3339()),
        Some(new_from.to_rfc3339())
    );
    assert_eq!(
        stored.valid_until.map(|value| value.to_rfc3339()),
        Some(new_until.to_rfc3339())
    );

    // The EFFECTIVE post-merge window is validated: a valid_from at or
    // beyond the stored valid_until is rejected and nothing changes.
    let error = storage
        .update_node_validity(&node.id, Some(Utc::now() + Duration::days(30)), None)
        .unwrap_err();
    assert!(matches!(error, StorageError::InvalidTimestamp(_)));
    let stored = storage.get_node(&node.id).unwrap().unwrap();
    assert_eq!(
        stored.valid_from.map(|value| value.to_rfc3339()),
        Some(new_from.to_rfc3339())
    );
    assert_eq!(
        stored.valid_until.map(|value| value.to_rfc3339()),
        Some(new_until.to_rfc3339())
    );

    // Both bounds supplied together are still validated up front.
    let error = storage
        .update_node_validity(&node.id, Some(until), Some(from))
        .unwrap_err();
    assert!(matches!(error, StorageError::InvalidTimestamp(_)));
}
