//! Tests for `sqlite/embeddings.rs`: the vector-search gate, embedding
//! profiles, migration checkpoints and the peer-process index refresh (#181).

use super::*;

#[cfg(feature = "vector-search")]
#[test]
fn vector_search_env_value_parsing() {
    use std::ffi::OsStr;
    for on in ["1", "true", "TRUE", "yes", "On", "enable", "enabled"] {
        assert!(
            env_value_disables_vector_search(OsStr::new(on)),
            "{on} must disable"
        );
    }
    for off in ["", "0", "false", "no", "off", "disabled", "banana"] {
        assert!(
            !env_value_disables_vector_search(OsStr::new(off)),
            "{off:?} must not disable"
        );
    }
}

/// The regression guard for the test race itself: disabling vector search
/// in one test must be invisible to a storage built on another thread.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn disabling_vector_search_in_one_test_does_not_leak_to_other_threads() {
    with_vector_search_disabled(|| {
        assert!(!Storage::vector_search_enabled_by_cpu());
        let sibling = std::thread::spawn(|| {
            let dir = tempdir().unwrap();
            let storage = create_test_storage_at(&dir, "sibling-thread.db");
            (
                Storage::vector_search_enabled_by_cpu(),
                storage.vector_index.is_some(),
            )
        })
        .join()
        .unwrap();
        assert_eq!(
            sibling,
            (true, true),
            "a sibling thread saw this test's vector-search override"
        );
    });
    assert!(Storage::vector_search_enabled_by_cpu());
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn pinning_auto_merge_in_one_test_does_not_leak_to_other_threads() {
    let real = std::env::var("VESTIGE_AUTO_CONSOLIDATE_MERGE").ok();
    with_auto_merge_env(Some("1"), || {
        assert_eq!(
            Storage::auto_consolidate_merge_value().as_deref(),
            Some("1")
        );
        let sibling = std::thread::spawn(Storage::auto_consolidate_merge_value)
            .join()
            .unwrap();
        assert_eq!(sibling, real, "a sibling thread saw this test's pin");
    });
    assert_eq!(Storage::auto_consolidate_merge_value(), real);
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_runtime_vector_gate_env_disables_index_creation() {
    with_vector_search_disabled(|| {
        assert!(!Storage::vector_search_enabled_by_cpu());
        assert_eq!(
            Storage::vector_search_unavailable_reason(),
            Some("disabled by VESTIGE_DISABLE_VECTOR_SEARCH")
        );

        let dir = tempdir().unwrap();
        let storage = create_test_storage_at(&dir, "vector-disabled.db");

        assert!(storage.vector_index.is_none());
        assert!(storage.query_cache.is_none());

        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.total_nodes, 0);

        let schema = storage.schema_introspection().unwrap();
        assert!(schema.schema_version >= 1);
    });
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_runtime_vector_gate_disabled_hybrid_search_uses_keyword_fallback() {
    with_vector_search_disabled(|| {
        let dir = tempdir().unwrap();
        let storage = create_test_storage_at(&dir, "vector-disabled-search.db");

        storage
            .ingest(IngestInput {
                content: "runtime gate fallback keyword anchor".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();

        let results = storage
            .hybrid_search("runtime gate fallback keyword", 10, 0.3, 0.7)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].match_type, MatchType::Keyword);
        assert!(results[0].semantic_score.is_none());
        assert!(
            results[0]
                .node
                .content
                .contains("runtime gate fallback keyword anchor")
        );
    });
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_embedding_model_identity_matching() {
    assert!(Storage::embedding_model_matches_active(
        "Qwen/Qwen3-Embedding-0.6B",
        "Qwen/Qwen3-Embedding-0.6B",
    ));
    assert!(!Storage::embedding_model_matches_active(
        "nomic-embed-text-v1.5",
        "nomic-ai/nomic-embed-text-v1.5",
    ));
    assert!(!Storage::embedding_model_matches_active(
        "nomic-ai/nomic-embed-text-v1.5",
        "Qwen/Qwen3-Embedding-0.6B",
    ));

    let bytes = Embedding::new(vec![1.0; EMBEDDING_DIMENSIONS]).to_bytes();
    assert!(
        Storage::embedding_vector_for_active_model(
            &bytes,
            "nomic-ai/nomic-embed-text-v1.5",
            "Qwen/Qwen3-Embedding-0.6B",
        )
        .is_none()
    );
    assert!(
        Storage::embedding_vector_for_active_model(
            &bytes,
            "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-0.6B",
        )
        .is_some()
    );
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn test_embedding_regeneration_candidates_include_entire_mismatched_corpus() {
    let storage = create_test_storage();
    let stale_model = "all-MiniLM-L6-v2";
    let stale_embedding = Embedding::new(vec![0.0; EMBEDDING_DIMENSIONS]).to_bytes();

    for i in 0..125 {
        let node = storage
            .ingest(IngestInput {
                content: format!("legacy embedded memory {}", i),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();

        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "INSERT OR REPLACE INTO node_embeddings
                     (node_id, embedding, dimensions, model, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    &node.id,
                    &stale_embedding,
                    EMBEDDING_DIMENSIONS as i32,
                    stale_model,
                    Utc::now().to_rfc3339()
                ],
            )
            .unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes
                     SET has_embedding = 1, embedding_model = ?2
                     WHERE id = ?1",
                rusqlite::params![&node.id, stale_model],
            )
            .unwrap();
        // In a process where the legacy runtime was initialized by an
        // earlier test, ingest also writes a matching V28 profile vector.
        // Remove it so this fixture consistently represents a pre-V28
        // mirror-only stale corpus.
        writer
            .execute(
                "DELETE FROM embedding_profile_vectors
                     WHERE profile_id = ?1 AND node_id = ?2",
                rusqlite::params![LEGACY_EMBEDDING_PROFILE_ID, &node.id],
            )
            .unwrap();
    }

    let stats = storage.get_stats().unwrap();
    assert_eq!(stats.nodes_with_mismatched_embeddings, 125);
    assert_eq!(stats.nodes_with_active_embeddings, 0);

    let legacy = EmbeddingProfileId::new(LEGACY_EMBEDDING_PROFILE_ID).unwrap();
    let candidates = storage
        .embedding_regeneration_candidates(
            &legacy,
            EMBEDDING_DIMENSIONS,
            "nomic-ai/nomic-embed-text-v1.5",
            None,
            false,
        )
        .unwrap();
    assert_eq!(candidates.len(), 125);
}

#[test]
fn test_storage_creation() {
    let storage = create_test_storage();
    let stats = storage.get_stats().unwrap();
    assert_eq!(stats.total_nodes, 0);
}

#[test]
fn reopening_after_qwen_pointer_preserves_legacy_manifest_state() {
    let dir = tempdir().unwrap();
    let qwen = BuiltinEmbeddingProfile::QwenBalanced1024
        .profile()
        .profile_id;
    {
        let storage = create_test_storage_at(&dir, "reopen-qwen.db");
        storage
            .save_embedding_profile_manifest(&ready_profile_manifest(
                BuiltinEmbeddingProfile::QwenBalanced1024,
            ))
            .unwrap();
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE embedding_profiles SET status = 'ready' WHERE profile_id = ?1",
                params![LEGACY_EMBEDDING_PROFILE_ID],
            )
            .unwrap();
        writer
            .execute(
                "UPDATE embedding_profiles SET status = 'active' WHERE profile_id = ?1",
                params![qwen.as_str()],
            )
            .unwrap();
        writer
                .execute(
                    "UPDATE embedding_profile_state
                     SET active_profile_id = ?1, previous_profile_id = ?2, activated_at = ?3, updated_at = ?3
                     WHERE singleton = 1",
                    params![
                        qwen.as_str(),
                        LEGACY_EMBEDDING_PROFILE_ID,
                        Utc::now().to_rfc3339(),
                    ],
                )
                .unwrap();
    }

    let reopened = create_test_storage_at(&dir, "reopen-qwen.db");
    assert_eq!(
        reopened
            .active_embedding_profile()
            .unwrap()
            .unwrap()
            .profile_id,
        qwen
    );
    let legacy = EmbeddingProfileId::new(LEGACY_EMBEDDING_PROFILE_ID).unwrap();
    assert_eq!(
        reopened
            .embedding_profile_manifest(&legacy)
            .unwrap()
            .unwrap()
            .state,
        EmbeddingProfileState::Active,
        "a valid legacy manifest is preserved exactly; bootstrap must not rewrite it"
    );
}

fn ready_profile_manifest(profile: BuiltinEmbeddingProfile) -> EmbeddingProfileManifest {
    let mut manifest = EmbeddingProfileManifest::not_installed(profile.profile()).unwrap();
    manifest.state = EmbeddingProfileState::Ready;
    manifest
}

#[test]
fn embedding_profiles_keep_vectors_isolated() {
    let storage = create_test_storage();
    let legacy = EmbeddingProfileId::new(LEGACY_EMBEDDING_PROFILE_ID).unwrap();
    let qwen = BuiltinEmbeddingProfile::QwenBalanced1024
        .profile()
        .profile_id;
    storage
        .save_embedding_profile_manifest(&ready_profile_manifest(
            BuiltinEmbeddingProfile::QwenBalanced1024,
        ))
        .unwrap();
    let node = storage
        .ingest(IngestInput {
            content: "profile vector isolation".to_string(),
            ..Default::default()
        })
        .unwrap();

    storage
        .put_embedding_profile_vector(&EmbeddingProfileVector {
            profile_id: legacy.to_string(),
            node_id: node.id.clone(),
            embedding: vec![1, 2, 3],
            dimensions: 256,
            model: "legacy".to_string(),
            created_at: Utc::now(),
        })
        .unwrap();
    storage
        .put_embedding_profile_vector(&EmbeddingProfileVector {
            profile_id: qwen.to_string(),
            node_id: node.id.clone(),
            embedding: vec![4, 5, 6],
            dimensions: 1024,
            model: "qwen".to_string(),
            created_at: Utc::now(),
        })
        .unwrap();

    assert_eq!(
        storage
            .embedding_profile_vector(&legacy, &node.id)
            .unwrap()
            .unwrap()
            .embedding,
        vec![1, 2, 3]
    );
    assert_eq!(
        storage
            .embedding_profile_vector(&qwen, &node.id)
            .unwrap()
            .unwrap()
            .embedding,
        vec![4, 5, 6]
    );

    // Neither profile can see the other profile's vector. The source row
    // remains, so a later validated activation never needs a re-embed.
    assert!(
        storage
            .embedding_profile_vector(&legacy, &node.id)
            .unwrap()
            .is_some()
    );
}

#[test]
fn activation_rejects_ready_profile_without_verified_runtime_and_evaluation() {
    let storage = create_test_storage();
    let manifest = ready_profile_manifest(BuiltinEmbeddingProfile::QwenBalanced1024);
    let profile_id = manifest.profile.profile_id.clone();
    storage.save_embedding_profile_manifest(&manifest).unwrap();

    let error = storage.activate_embedding_profile(&profile_id).unwrap_err();
    assert!(matches!(error, StorageError::InvalidEmbeddingProfile(_)));
    assert_eq!(
        storage
            .active_embedding_profile()
            .unwrap()
            .unwrap()
            .profile_id
            .as_str(),
        LEGACY_EMBEDDING_PROFILE_ID
    );
}

#[cfg(feature = "embeddings")]
#[test]
fn init_embeddings_permits_the_released_legacy_nomic_profile() {
    let storage = create_test_storage();

    // `init_embeddings` still owns the released Nomic startup path. The
    // actual backend may be unavailable in an offline test environment,
    // but that must surface as backend initialization failure, never as a
    // profile-policy rejection.
    match storage.init_embeddings() {
        Ok(()) | Err(StorageError::Init(_)) => {}
        Err(error) => {
            panic!("legacy Nomic profile must be permitted to use init_embeddings: {error}")
        }
    }
}

#[cfg(feature = "embeddings")]
#[test]
fn init_embeddings_rejects_an_active_qwen_profile() {
    let storage = create_test_storage();
    let qwen = BuiltinEmbeddingProfile::QwenBalanced1024
        .profile()
        .profile_id;
    storage
        .save_embedding_profile_manifest(&ready_profile_manifest(
            BuiltinEmbeddingProfile::QwenBalanced1024,
        ))
        .unwrap();

    // This intentionally sets only the persisted active pointer. It does
    // not satisfy the activation gate; the point is to verify that the
    // legacy convenience initializer fails closed before it can select or
    // initialize a different vector-space runtime.
    let writer = storage.writer.lock().unwrap();
    writer
        .execute(
            "UPDATE embedding_profiles SET status = 'ready' WHERE profile_id = ?1",
            params![LEGACY_EMBEDDING_PROFILE_ID],
        )
        .unwrap();
    writer
        .execute(
            "UPDATE embedding_profiles SET status = 'active' WHERE profile_id = ?1",
            params![qwen.as_str()],
        )
        .unwrap();
    writer
        .execute(
            "UPDATE embedding_profile_state
                 SET active_profile_id = ?1, previous_profile_id = ?2,
                     activated_at = ?3, updated_at = ?3
                 WHERE singleton = 1",
            params![
                qwen.as_str(),
                LEGACY_EMBEDDING_PROFILE_ID,
                Utc::now().to_rfc3339(),
            ],
        )
        .unwrap();
    drop(writer);

    let error = storage.init_embeddings().unwrap_err();
    assert!(matches!(error, StorageError::InvalidEmbeddingProfile(_)));
    assert!(error.to_string().contains(qwen.as_str()));
    assert!(error.to_string().contains("explicit profile workflow"));
}

#[test]
fn migration_vector_and_node_checkpoint_commit_together() {
    let storage = create_test_storage();
    let legacy = EmbeddingProfileId::new(LEGACY_EMBEDDING_PROFILE_ID).unwrap();
    let qwen = BuiltinEmbeddingProfile::QwenBalanced1024
        .profile()
        .profile_id;
    storage
        .save_embedding_profile_manifest(&ready_profile_manifest(
            BuiltinEmbeddingProfile::QwenBalanced1024,
        ))
        .unwrap();
    let migration_id = Uuid::new_v4();
    let now = Utc::now();
    storage
        .save_profile_migration_checkpoint(&ProfileMigrationCheckpoint {
            migration_id,
            source_profile_id: legacy,
            destination_profile_id: qwen.clone(),
            state: EmbeddingMigrationState::Running,
            total_memories: 1,
            completed_memories: 0,
            failed_memory_ids: Vec::new(),
            last_memory_id: None,
            started_at: now,
            updated_at: now,
        })
        .unwrap();
    let node = storage
        .ingest(IngestInput {
            content: "atomic migration checkpoint target".to_string(),
            ..Default::default()
        })
        .unwrap();
    storage
        .put_embedding_profile_vector_with_migration_checkpoint(
            &EmbeddingProfileVector {
                profile_id: qwen.to_string(),
                node_id: node.id.clone(),
                embedding: vec![1, 2, 3],
                dimensions: 1024,
                model: "qwen-test".to_string(),
                created_at: now,
            },
            &EmbeddingProfileMigrationNodeCheckpoint {
                migration_id: migration_id.to_string(),
                node_id: node.id.clone(),
                state: "completed".to_string(),
                error: None,
                updated_at: now,
            },
        )
        .unwrap();
    assert!(
        storage
            .embedding_profile_vector(&qwen, &node.id)
            .unwrap()
            .is_some()
    );
    let reader = storage.reader.lock().unwrap();
    let checkpoint_rows: i64 = reader
        .query_row(
            "SELECT COUNT(*) FROM embedding_profile_migration_checkpoints
                 WHERE migration_id = ?1 AND node_id = ?2",
            params![migration_id.to_string(), node.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(checkpoint_rows, 1);
}

#[test]
fn purge_removes_vectors_from_every_embedding_profile() {
    let storage = create_test_storage();
    let node = storage
        .ingest(IngestInput {
            content: "profile-wide purge target".to_string(),
            ..Default::default()
        })
        .unwrap();
    let legacy = EmbeddingProfileId::new(LEGACY_EMBEDDING_PROFILE_ID).unwrap();
    let qwen = BuiltinEmbeddingProfile::QwenBalanced1024
        .profile()
        .profile_id;
    storage
        .save_embedding_profile_manifest(&ready_profile_manifest(
            BuiltinEmbeddingProfile::QwenBalanced1024,
        ))
        .unwrap();
    for (profile_id, dimensions) in [(&legacy, 256), (&qwen, 1024)] {
        storage
            .put_embedding_profile_vector(&EmbeddingProfileVector {
                profile_id: profile_id.to_string(),
                node_id: node.id.clone(),
                embedding: vec![7, 8, 9],
                dimensions,
                model: "test".to_string(),
                created_at: Utc::now(),
            })
            .unwrap();
    }

    storage
        .purge_node(&node.id, Some("profile purge test"))
        .unwrap();
    let reader = storage.reader.lock().unwrap();
    let remaining: i64 = reader
        .query_row(
            "SELECT COUNT(*) FROM embedding_profile_vectors WHERE node_id = ?1",
            params![node.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(remaining, 0, "purge must cascade to every profile vector");
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn non_256_active_profile_builds_matching_dimension_index_without_truncation() {
    let storage = create_test_storage();
    let qwen = BuiltinEmbeddingProfile::QwenBalanced1024
        .profile()
        .profile_id;
    storage
        .save_embedding_profile_manifest(&ready_profile_manifest(
            BuiltinEmbeddingProfile::QwenBalanced1024,
        ))
        .unwrap();
    let node = storage
        .ingest(IngestInput {
            content: "dimension isolation".to_string(),
            ..Default::default()
        })
        .unwrap();
    storage
        .put_embedding_profile_vector(&EmbeddingProfileVector {
            profile_id: qwen.to_string(),
            node_id: node.id,
            embedding: Embedding::new(vec![0.25; 1024]).to_bytes(),
            dimensions: 1024,
            model: "Qwen/Qwen3-Embedding-0.6B".to_string(),
            created_at: Utc::now(),
        })
        .unwrap();
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE embedding_profile_state SET active_profile_id = ?1 WHERE singleton = 1",
                params![qwen.as_str()],
            )
            .unwrap();
    }
    storage.load_embeddings_into_index().unwrap();
    assert_eq!(
        storage
            .vector_index
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .dimensions(),
        1024
    );
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
fn ingest_plain(storage: &Storage, content: &str) -> String {
    storage
        .ingest(IngestInput {
            content: content.to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap()
        .id
}

/// #181: a memory written by a peer process must become semantically
/// searchable in THIS process without a restart. The pre-refresh assertion
/// is the negative half: it is exactly what every process saw before the fix.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn peer_process_write_is_visible_to_the_vector_index_without_restart() {
    let dir = tempdir().unwrap();
    let ours = create_test_storage_at(&dir, "shared.db");
    let peer = create_test_storage_at(&dir, "shared.db");

    let id = ingest_plain(&peer, "written by a sibling MCP server process");
    persist_test_vector(&peer, &id, &axis_vector(3, 0.01));

    assert!(
        !index_contains(&ours, &id),
        "a process-local index cannot know about a peer's write until it refreshes"
    );

    assert_eq!(
        ours.refresh_vector_index_if_stale(),
        1,
        "exactly the peer's row is absorbed"
    );
    assert!(index_contains(&ours, &id));
    assert_eq!(
        nearest(&ours, &axis_vector(3, 0.0)).as_deref(),
        Some(id.as_str())
    );

    assert_eq!(
        ours.refresh_vector_index_if_stale(),
        0,
        "nothing new since: one PRAGMA and an early return"
    );
}

/// #181: a peer re-embedding an existing node through the UPSERT path (used by
/// profile repair, which keeps the row's rowid) must replace the stale vector
/// here. A rowid or contains()-based refresh could never notice this write.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn peer_reembedding_replaces_the_stale_vector_here() {
    let dir = tempdir().unwrap();
    let ours = create_test_storage_at(&dir, "shared.db");
    let peer = create_test_storage_at(&dir, "shared.db");

    let moved = ingest_plain(&peer, "a memory whose vector will be regenerated");
    let decoy = ingest_plain(&peer, "a decoy that stays on the old axis");
    persist_test_vector(&peer, &moved, &axis_vector(3, 0.01));
    persist_test_vector(&peer, &decoy, &axis_vector(3, 0.02));
    assert_eq!(ours.refresh_vector_index_if_stale(), 2);
    assert_eq!(
        nearest(&ours, &axis_vector(3, 0.01)).as_deref(),
        Some(moved.as_str())
    );

    let active = peer
        .active_embedding_profile()
        .unwrap()
        .expect("test stores have an active profile")
        .profile_id
        .to_string();
    peer.put_embedding_profile_vector(&EmbeddingProfileVector {
        profile_id: active,
        node_id: moved.clone(),
        embedding: Embedding::new(axis_vector(9, 0.01)).to_bytes(),
        dimensions: EMBEDDING_DIMENSIONS as u32,
        model: "test-model".to_string(),
        created_at: Utc::now(),
    })
    .unwrap();

    assert_eq!(
        ours.refresh_vector_index_if_stale(),
        1,
        "one upsert row, one replaced vector"
    );
    assert_eq!(
        nearest(&ours, &axis_vector(9, 0.0)).as_deref(),
        Some(moved.as_str())
    );
    assert_eq!(
        nearest(&ours, &axis_vector(3, 0.02)).as_deref(),
        Some(decoy.as_str()),
        "the moved node must no longer win its old axis"
    );
}

/// #181: a peer's purge cascades to its vector row, the delete trigger
/// journals it, and the dead vector leaves this index too.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn peer_purge_removes_the_vector_here() {
    let dir = tempdir().unwrap();
    let ours = create_test_storage_at(&dir, "shared.db");
    let peer = create_test_storage_at(&dir, "shared.db");

    let id = ingest_plain(&peer, "a memory the peer will purge");
    persist_test_vector(&peer, &id, &axis_vector(4, 0.01));
    assert_eq!(ours.refresh_vector_index_if_stale(), 1);
    assert!(index_contains(&ours, &id));

    peer.purge_node(&id, Some("peer purge")).unwrap();

    assert_eq!(
        ours.refresh_vector_index_if_stale(),
        1,
        "one delete row, one removal"
    );
    assert!(!index_contains(&ours, &id));
}

/// #181: this process's own writes bump the reader's data_version exactly
/// like a peer's would, but the vector is already in the index. The journal
/// head says so, and the refresh re-adds nothing.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn own_writes_are_not_reabsorbed_as_peer_changes() {
    let storage = create_test_storage();
    let id = ingest_plain(&storage, "written by this very process");
    persist_test_vector(&storage, &id, &axis_vector(5, 0.01));
    assert!(index_contains(&storage, &id));
    assert_eq!(storage.refresh_vector_index_if_stale(), 0);
}

/// #181: when the journal has been pruned past this process's watermark, the
/// refresh must not trust it. It reconciles against the table and still
/// absorbs everything the peers wrote, including rows the journal no longer
/// names.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn a_journal_pruned_past_the_watermark_reconciles_against_the_table() {
    let dir = tempdir().unwrap();
    let ours = create_test_storage_at(&dir, "shared.db");
    let peer = create_test_storage_at(&dir, "shared.db");

    let first = ingest_plain(&peer, "first peer memory");
    let second = ingest_plain(&peer, "second peer memory");
    persist_test_vector(&peer, &first, &axis_vector(1, 0.01));
    persist_test_vector(&peer, &second, &axis_vector(2, 0.01));

    // A prune that ran before this process caught up: only the newest row
    // survives, so the journal alone would name just one of the two.
    {
        let writer = peer.writer.lock().unwrap();
        writer
            .execute(
                "DELETE FROM vector_journal WHERE seq < (SELECT MAX(seq) FROM vector_journal)",
                [],
            )
            .unwrap();
    }

    assert_eq!(
        ours.refresh_vector_index_if_stale(),
        2,
        "reconcile absorbs both peer vectors, not just the journal survivor"
    );
    assert!(index_contains(&ours, &first));
    assert!(index_contains(&ours, &second));
    assert_eq!(
        ours.refresh_vector_index_if_stale(),
        0,
        "the watermark moved to the head, so the next look is incremental and empty"
    );
}

/// #181 housekeeping: pruning removes only rows that are both older than
/// the retention window AND further behind the head than the keep window.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn vector_journal_prune_keeps_recent_rows_and_the_head() {
    let storage = create_test_storage();
    let id = ingest_plain(&storage, "one vector, at least one journal row");
    persist_test_vector(&storage, &id, &axis_vector(6, 0.01));
    assert_eq!(
        storage.prune_vector_journal().unwrap(),
        0,
        "fresh rows are never pruned"
    );
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE vector_journal SET at = '2000-01-01T00:00:00.000Z'",
                [],
            )
            .unwrap();
        // Push the head far past the keep window with one synthetic row.
        writer
            .execute(
                "INSERT INTO vector_journal (seq, profile_id, node_id, op)
                     VALUES (20000, 'synthetic', 'head', 'upsert')",
                [],
            )
            .unwrap();
    }
    let deleted = storage.prune_vector_journal().unwrap();
    assert!(deleted >= 1, "old rows far behind the head must go");
    let remaining: i64 = storage
        .reader
        .lock()
        .unwrap()
        .query_row("SELECT COUNT(*) FROM vector_journal", [], |row| row.get(0))
        .unwrap();
    assert_eq!(remaining, 1, "only the fresh head row survives");
}
