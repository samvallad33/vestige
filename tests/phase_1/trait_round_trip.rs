//! Phase 1 integration tests: round-trip of every trait method through SqliteMemoryStore.

use std::sync::Arc;
use chrono::Utc;
use tempfile::tempdir;
use uuid::Uuid;
use vestige_core::storage::{
    MemoryEdge, MemoryRecord, MemoryStore, SearchQuery, SqliteMemoryStore,
};

fn make_store() -> Arc<dyn MemoryStore> {
    let dir = tempdir().unwrap();
    let db = dir.path().join("test.db");
    // keep the dir alive by leaking it -- this is fine for tests
    std::mem::forget(dir);
    let store = SqliteMemoryStore::new(Some(db)).expect("create store");
    Arc::new(store)
}

fn make_record(content: &str) -> MemoryRecord {
    MemoryRecord {
        id: Uuid::new_v4(),
        domains: vec![],
        domain_scores: Default::default(),
        content: content.to_string(),
        node_type: "fact".to_string(),
        tags: vec!["integration".to_string()],
        embedding: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        metadata: serde_json::json!({}),
    }
}

#[tokio::test]
async fn insert_get_update_delete() {
    let store = make_store();
    let rec = make_record("round-trip CRUD test");
    let id = rec.id;

    store.insert(&rec).await.expect("insert");
    let got = store.get(id).await.expect("get").expect("exists");
    assert_eq!(got.content, "round-trip CRUD test");
    assert_eq!(got.node_type, "fact");
    assert!(got.domains.is_empty());
    assert!(got.domain_scores.is_empty());

    let mut updated = got;
    updated.content = "updated content".to_string();
    store.update(&updated).await.expect("update");

    let after_update = store.get(id).await.expect("get after update").expect("exists");
    assert_eq!(after_update.content, "updated content");

    store.delete(id).await.expect("delete");
    let after_delete = store.get(id).await.expect("get after delete");
    assert!(after_delete.is_none());
}

#[tokio::test]
async fn scheduling_upsert_and_due_scan() {
    use vestige_core::storage::SchedulingState;
    let store = make_store();

    for i in 0..3usize {
        let rec = make_record(&format!("sched memory {i}"));
        let id = rec.id;
        store.insert(&rec).await.expect("insert");
        let next_review = Utc::now() - chrono::Duration::days((i as i64) + 1);
        let state = SchedulingState {
            memory_id: id,
            stability: 1.0,
            difficulty: 0.3,
            retrievability: 0.7,
            last_review: Some(Utc::now()),
            next_review: Some(next_review),
            reps: 1,
            lapses: 0,
        };
        store.update_scheduling(&state).await.expect("update scheduling");
    }

    let due = store
        .get_due_memories(Utc::now(), 10)
        .await
        .expect("get_due_memories");
    assert_eq!(due.len(), 3, "all 3 should be due");
}

#[tokio::test]
async fn edge_crud() {
    let store = make_store();
    let rec_a = make_record("edge node A");
    let rec_b = make_record("edge node B");
    let id_a = rec_a.id;
    let id_b = rec_b.id;
    store.insert(&rec_a).await.expect("insert a");
    store.insert(&rec_b).await.expect("insert b");

    let edge = MemoryEdge {
        source_id: id_a,
        target_id: id_b,
        edge_type: "semantic".to_string(),
        weight: 0.85,
        created_at: Utc::now(),
    };
    store.add_edge(&edge).await.expect("add edge");

    let edges = store.get_edges(id_a, None).await.expect("get edges");
    assert!(!edges.is_empty());

    store.remove_edge(id_a, id_b).await.expect("remove edge");
    let after = store.get_edges(id_a, None).await.expect("get edges after");
    assert!(after.is_empty());
}

#[tokio::test]
async fn count_and_stats_track_inserts() {
    let store = make_store();
    for i in 0..10usize {
        let rec = make_record(&format!("stats memory {i}"));
        store.insert(&rec).await.expect("insert");
    }
    assert_eq!(store.count().await.expect("count"), 10);
    let stats = store.get_stats().await.expect("stats");
    assert_eq!(stats.total_memories, 10);
}

#[tokio::test]
async fn vacuum_after_deletes_reclaims() {
    let dir = tempdir().unwrap();
    let db = dir.path().join("vacuum_test.db");
    let store = SqliteMemoryStore::new(Some(db)).expect("create store");
    let store: Arc<dyn MemoryStore> = Arc::new(store);

    let mut ids = Vec::new();
    for i in 0..50usize {
        let rec = make_record(&format!("vacuum memory {i}"));
        let id = store.insert(&rec).await.expect("insert");
        ids.push(id);
    }
    for id in &ids[..40] {
        store.delete(*id).await.expect("delete");
    }
    // vacuum should not error
    store.vacuum().await.expect("vacuum");
}

#[tokio::test]
async fn list_domains_empty_then_upsert_then_delete() {
    use vestige_core::storage::Domain;
    let store = make_store();

    let domains = store.list_domains().await.expect("list empty");
    assert!(domains.is_empty());

    let d = Domain {
        id: "test-domain".to_string(),
        label: "Test Domain".to_string(),
        centroid: vec![0.1f32, 0.2, 0.3],
        top_terms: vec!["term1".to_string()],
        memory_count: 5,
        created_at: Utc::now(),
    };
    store.upsert_domain(&d).await.expect("upsert domain");
    let after = store.list_domains().await.expect("list after upsert");
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].id, "test-domain");

    store.delete_domain("test-domain").await.expect("delete domain");
    let after_delete = store.list_domains().await.expect("list after delete");
    assert!(after_delete.is_empty());
}

#[tokio::test]
async fn classify_with_no_domains_returns_empty() {
    let store = make_store();
    let result = store.classify(&[0.1f32, 0.2, 0.3]).await.expect("classify");
    assert!(result.is_empty());
}

#[tokio::test]
async fn search_hybrid_returns_results() {
    let store = make_store();
    let rec = make_record("quantum entanglement superposition physics");
    store.insert(&rec).await.expect("insert");

    // Verify fts_search works first (sanity check)
    let fts_results = store.fts_search("quantum", 10).await.expect("fts_search");
    assert!(!fts_results.is_empty(), "fts_search must find 'quantum' after insert");

    let query = SearchQuery {
        text: Some("quantum physics".to_string()),
        limit: 10,
        ..Default::default()
    };
    let results = store.search(&query).await.expect("search");
    // FTS results should include our inserted record
    assert!(!results.is_empty(), "search must return results for 'quantum physics'");
    assert!(results[0].score >= 0.0);
}
