//! Tests for `sqlite/store_trait.rs`: the `MemoryStoreSend` trait surface.

use super::*;

// =========================================================================
// Phase 1 trait-method unit tests
// =========================================================================
fn make_record(content: &str) -> MemoryRecord {
    MemoryRecord {
        id: uuid::Uuid::new_v4(),
        domains: vec![],
        domain_scores: Default::default(),
        content: content.to_string(),
        node_type: "fact".to_string(),
        tags: vec!["test".to_string()],
        embedding: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        metadata: serde_json::json!({}),
    }
}

#[test]
fn trait_init_is_idempotent() {
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        s.init().await.unwrap();
        s.init().await.unwrap();
    });
}

#[test]
fn trait_health_check_reports_healthy_on_fresh_db() {
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        let h = s.health_check().await.unwrap();
        assert!(matches!(
            h,
            crate::storage::memory_store::HealthStatus::Healthy
        ));
    });
}

#[test]
fn trait_register_model_first_write_succeeds() {
    let s = create_test_storage();
    let sig = ModelSignature {
        name: "test-model".to_string(),
        dimension: 256,
        hash: "a".repeat(64),
    };
    let rt = rt();
    rt.block_on(async {
        s.register_model(&sig).await.unwrap();
        let got = s.registered_model().await.unwrap();
        assert_eq!(got, Some(sig));
    });
}

#[test]
fn trait_register_model_mismatched_write_refused() {
    let s = create_test_storage();
    let sig = ModelSignature {
        name: "model-a".to_string(),
        dimension: 256,
        hash: "a".repeat(64),
    };
    let sig2 = ModelSignature {
        name: "model-b".to_string(),
        dimension: 256,
        hash: "b".repeat(64),
    };
    let rt = rt();
    rt.block_on(async {
        s.register_model(&sig).await.unwrap();
        let err = s.register_model(&sig2).await.unwrap_err();
        assert!(matches!(err, MemoryStoreError::ModelMismatch { .. }));
    });
}

#[test]
fn trait_register_model_same_signature_idempotent() {
    let s = create_test_storage();
    let sig = ModelSignature {
        name: "test-model".to_string(),
        dimension: 256,
        hash: "a".repeat(64),
    };
    let rt = rt();
    rt.block_on(async {
        s.register_model(&sig).await.unwrap();
        s.register_model(&sig).await.unwrap(); // second call must not error
    });
}

#[test]
fn trait_insert_returns_uuid() {
    let s = create_test_storage();
    let rec = make_record("test content");
    let expected_id = rec.id;
    let rt = rt();
    rt.block_on(async {
        let got = s.insert(&rec).await.unwrap();
        assert_eq!(got, expected_id);
    });
}

#[test]
fn trait_get_missing_returns_none() {
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        let got = s.get(uuid::Uuid::new_v4()).await.unwrap();
        assert!(got.is_none());
    });
}

#[test]
fn trait_get_after_insert_round_trip() {
    let s = create_test_storage();
    let rec = make_record("round trip content");
    let id = rec.id;
    let rt = rt();
    rt.block_on(async {
        s.insert(&rec).await.unwrap();
        let got = s.get(id).await.unwrap().unwrap();
        assert_eq!(got.content, "round trip content");
        assert_eq!(got.node_type, "fact");
        assert!(got.domains.is_empty());
        assert!(got.domain_scores.is_empty());
    });
}

#[test]
fn trait_update_modifies_content() {
    let s = create_test_storage();
    let rec = make_record("original content");
    let id = rec.id;
    let rt = rt();
    rt.block_on(async {
        s.insert(&rec).await.unwrap();
        let mut updated = s.get(id).await.unwrap().unwrap();
        updated.content = "updated content".to_string();
        s.update(&updated).await.unwrap();
        let got = s.get(id).await.unwrap().unwrap();
        assert_eq!(got.content, "updated content");
    });
}

#[test]
fn trait_delete_removes_record() {
    let s = create_test_storage();
    let rec = make_record("to be deleted");
    let id = rec.id;
    let rt = rt();
    rt.block_on(async {
        s.insert(&rec).await.unwrap();
        s.delete(id).await.unwrap();
        let got = s.get(id).await.unwrap();
        assert!(got.is_none());
    });
}

#[test]
fn trait_fts_search_returns_tokens_match() {
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        let rec = make_record("mitochondria powerhouse cell energy");
        s.insert(&rec).await.unwrap();
        let results = s.fts_search("mitochondria", 10).await.unwrap();
        assert!(!results.is_empty());
    });
}

#[test]
fn trait_hybrid_search_multi_word_via_insert() {
    // Verify that hybrid_search finds records inserted via the trait insert()
    // even when no embedding is present (keyword path via terms matching).
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        let rec = make_record("quantum entanglement superposition physics");
        s.insert(&rec).await.unwrap();
        let results = s.hybrid_search("quantum physics", 10, 0.3, 0.7).unwrap();
        assert!(
            !results.is_empty(),
            "hybrid_search must find record containing 'quantum' and 'physics'"
        );
    });
}

#[test]
fn trait_scheduling_round_trip() {
    let s = create_test_storage();
    let rec = make_record("fsrs scheduling test");
    let id = rec.id;
    let rt = rt();
    rt.block_on(async {
        s.insert(&rec).await.unwrap();
        let state = SchedulingState {
            memory_id: id,
            stability: 5.0,
            difficulty: 0.4,
            retrievability: 0.8,
            last_review: Some(chrono::Utc::now()),
            next_review: Some(chrono::Utc::now() + chrono::Duration::days(7)),
            reps: 3,
            lapses: 1,
        };
        s.update_scheduling(&state).await.unwrap();
        let got = s.get_scheduling(id).await.unwrap().unwrap();
        assert!((got.stability - 5.0).abs() < 0.01);
    });
}

#[test]
fn trait_get_scheduling_missing_returns_none() {
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        let got = s.get_scheduling(uuid::Uuid::new_v4()).await.unwrap();
        assert!(got.is_none());
    });
}

#[test]
fn trait_get_due_memories_returns_in_order() {
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        for i in 0..3usize {
            let rec = make_record(&format!("due memory {i}"));
            let id = rec.id;
            s.insert(&rec).await.unwrap();
            let state = SchedulingState {
                memory_id: id,
                stability: 1.0,
                difficulty: 0.3,
                retrievability: 0.5,
                last_review: Some(chrono::Utc::now()),
                next_review: Some(chrono::Utc::now() - chrono::Duration::days(3 - i as i64)),
                reps: 1,
                lapses: 0,
            };
            s.update_scheduling(&state).await.unwrap();
        }
        let due = s.get_due_memories(chrono::Utc::now(), 10).await.unwrap();
        assert_eq!(due.len(), 3);
    });
}

#[test]
fn trait_add_edge_is_idempotent() {
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        let rec_a = make_record("node a");
        let rec_b = make_record("node b");
        let id_a = rec_a.id;
        let id_b = rec_b.id;
        s.insert(&rec_a).await.unwrap();
        s.insert(&rec_b).await.unwrap();
        let edge = MemoryEdge {
            source_id: id_a,
            target_id: id_b,
            edge_type: "semantic".to_string(),
            weight: 0.9,
            created_at: chrono::Utc::now(),
        };
        s.add_edge(&edge).await.unwrap();
        s.add_edge(&edge).await.unwrap(); // idempotent
        let edges = s.get_edges(id_a, None).await.unwrap();
        let filtered: Vec<_> = edges
            .iter()
            .filter(|e| e.source_id == id_a && e.target_id == id_b)
            .collect();
        assert_eq!(filtered.len(), 1, "edge must not be duplicated");
    });
}

#[test]
fn trait_get_edges_filters_by_type() {
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        let rec_a = make_record("filter a");
        let rec_b = make_record("filter b");
        let id_a = rec_a.id;
        let id_b = rec_b.id;
        s.insert(&rec_a).await.unwrap();
        s.insert(&rec_b).await.unwrap();
        let edge = MemoryEdge {
            source_id: id_a,
            target_id: id_b,
            edge_type: "causal".to_string(),
            weight: 0.5,
            created_at: chrono::Utc::now(),
        };
        s.add_edge(&edge).await.unwrap();
        let causal = s.get_edges(id_a, Some("causal")).await.unwrap();
        assert!(!causal.is_empty());
        let semantic = s.get_edges(id_a, Some("semantic")).await.unwrap();
        assert!(semantic.is_empty());
    });
}

#[test]
fn trait_remove_edge_deletes_single() {
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        let rec_a = make_record("rm edge a");
        let rec_b = make_record("rm edge b");
        let id_a = rec_a.id;
        let id_b = rec_b.id;
        s.insert(&rec_a).await.unwrap();
        s.insert(&rec_b).await.unwrap();
        let edge = MemoryEdge {
            source_id: id_a,
            target_id: id_b,
            edge_type: "semantic".to_string(),
            weight: 0.7,
            created_at: chrono::Utc::now(),
        };
        s.add_edge(&edge).await.unwrap();
        s.remove_edge(id_a, id_b).await.unwrap();
        let edges = s.get_edges(id_a, None).await.unwrap();
        assert!(edges.is_empty());
    });
}

#[test]
fn trait_get_neighbors_bfs_depth_zero_returns_self_only() {
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        let rec = make_record("depth zero");
        let id = rec.id;
        s.insert(&rec).await.unwrap();
        let neighbors = s.get_neighbors(id, 0).await.unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0.id, id);
    });
}

#[test]
fn trait_get_neighbors_bfs_depth_two_expands() {
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        let rec_a = make_record("bfs node a");
        let rec_b = make_record("bfs node b");
        let rec_c = make_record("bfs node c");
        let id_a = rec_a.id;
        let id_b = rec_b.id;
        let id_c = rec_c.id;
        s.insert(&rec_a).await.unwrap();
        s.insert(&rec_b).await.unwrap();
        s.insert(&rec_c).await.unwrap();
        s.add_edge(&MemoryEdge {
            source_id: id_a,
            target_id: id_b,
            edge_type: "semantic".to_string(),
            weight: 1.0,
            created_at: chrono::Utc::now(),
        })
        .await
        .unwrap();
        s.add_edge(&MemoryEdge {
            source_id: id_b,
            target_id: id_c,
            edge_type: "semantic".to_string(),
            weight: 1.0,
            created_at: chrono::Utc::now(),
        })
        .await
        .unwrap();
        let neighbors = s.get_neighbors(id_a, 2).await.unwrap();
        let ids: Vec<uuid::Uuid> = neighbors.iter().map(|(r, _)| r.id).collect();
        assert!(ids.contains(&id_a));
        assert!(ids.contains(&id_b));
        assert!(ids.contains(&id_c));
    });
}

#[test]
fn trait_list_domains_empty_in_phase_1() {
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        let domains = s.list_domains().await.unwrap();
        assert!(domains.is_empty());
    });
}

#[test]
fn trait_upsert_then_get_domain_round_trip() {
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        let domain = crate::storage::memory_store::Domain {
            id: "dev".to_string(),
            label: "Development".to_string(),
            centroid: vec![0.1, 0.2, 0.3],
            top_terms: vec!["rust".to_string(), "code".to_string()],
            memory_count: 42,
            created_at: chrono::Utc::now(),
        };
        s.upsert_domain(&domain).await.unwrap();
        let got = s.get_domain("dev").await.unwrap().unwrap();
        assert_eq!(got.id, "dev");
        assert_eq!(got.memory_count, 42);
    });
}

#[test]
fn trait_delete_domain_idempotent() {
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        s.delete_domain("nonexistent").await.unwrap();
        s.delete_domain("nonexistent").await.unwrap();
    });
}

#[test]
fn trait_classify_with_no_domains_returns_empty() {
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        let result = s.classify(&[0.1, 0.2, 0.3]).await.unwrap();
        assert!(result.is_empty());
    });
}

#[test]
fn trait_count_matches_insert_count() {
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        for i in 0..5usize {
            let rec = make_record(&format!("count test {i}"));
            s.insert(&rec).await.unwrap();
        }
        assert_eq!(s.count().await.unwrap(), 5);
    });
}

#[test]
fn trait_insert_rejects_secret_shaped_tags_and_source_without_a_row() {
    let s = create_test_storage();
    let credential = format!("ghp_{}", "A".repeat(36));
    let rt = rt();
    rt.block_on(async {
        let mut tagged = make_record("safe direct trait insert");
        tagged.tags = vec![credential.clone()];
        let err = s.insert(&tagged).await.unwrap_err();
        assert!(matches!(err, MemoryStoreError::SecretDetected(_)));
        assert!(!err.to_string().contains(&credential));
        assert_eq!(s.count().await.unwrap(), 0);

        let mut sourced = make_record("another safe direct trait insert");
        sourced.metadata = serde_json::json!({ "source": credential });
        let err = s.insert(&sourced).await.unwrap_err();
        assert!(matches!(err, MemoryStoreError::SecretDetected(_)));
        assert_eq!(s.count().await.unwrap(), 0);
    });
}

#[test]
fn trait_get_stats_reports_registered_model() {
    let s = create_test_storage();
    let sig = ModelSignature {
        name: "test-model".to_string(),
        dimension: 256,
        hash: "c".repeat(64),
    };
    let rt = rt();
    rt.block_on(async {
        use crate::storage::memory_store::MemoryStore;
        // Cast to &dyn MemoryStore so the async trait method is called
        // instead of the inherent sync get_stats() on SqliteMemoryStore.
        let dyn_s: &dyn MemoryStore = &s;
        dyn_s.register_model(&sig).await.unwrap();
        let stats = dyn_s.get_stats().await.unwrap();
        assert_eq!(stats.registered_model_name, Some("test-model".to_string()));
        assert_eq!(stats.registered_model_dim, Some(256));
    });
}

#[test]
fn trait_vacuum_succeeds() {
    let s = create_test_storage();
    let rt = rt();
    rt.block_on(async {
        s.vacuum().await.unwrap();
    });
}

#[test]
fn trait_insert_refuses_dimension_mismatch() {
    let s = create_test_storage();
    let sig = ModelSignature {
        name: "test-model".to_string(),
        dimension: 256,
        hash: "d".repeat(64),
    };
    let rt = rt();
    rt.block_on(async {
        s.register_model(&sig).await.unwrap();
        // Build a record with wrong dimension (512 instead of 256) and
        // declare the model signature in metadata
        let mut rec = make_record("dimension mismatch");
        rec.embedding = Some(vec![0.0f32; 512]);
        rec.metadata = serde_json::json!({
            "model_name": "test-model",
            "model_dim": 256_u64,
            "model_hash": "d".repeat(64),
        });
        let err = s.insert(&rec).await.unwrap_err();
        assert!(
            matches!(err, MemoryStoreError::InvalidInput(_)),
            "expected InvalidInput, got {:?}",
            err
        );
    });
}
