//! Phase 1 integration tests: Arc<dyn MemoryStore> moves across tokio::spawn.
//!
//! This verifies that `#[trait_variant::make(MemoryStore: Send)]` actually
//! produces a Send-bound future so Arc<dyn MemoryStore> is movable.

use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;
use chrono::Utc;
use vestige_core::storage::{MemoryRecord, MemoryStore, SqliteMemoryStore};

fn make_store() -> Arc<dyn MemoryStore> {
    let dir = tempdir().unwrap();
    let db = dir.path().join("send_test.db");
    std::mem::forget(dir);
    Arc::new(SqliteMemoryStore::new(Some(db)).expect("create"))
}

fn make_record(content: &str) -> MemoryRecord {
    MemoryRecord {
        id: Uuid::new_v4(),
        domains: vec![],
        domain_scores: Default::default(),
        content: content.to_string(),
        node_type: "fact".to_string(),
        tags: vec![],
        embedding: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        metadata: serde_json::json!({}),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn arc_dyn_memory_store_moves_across_tokio_tasks() {
    let store: Arc<dyn MemoryStore> = make_store();
    let mut handles = Vec::new();
    for t in 0..16usize {
        let store = Arc::clone(&store);
        let handle = tokio::spawn(async move {
            for i in 0..10usize {
                let rec = make_record(&format!("task {t} memory {i}"));
                store.insert(&rec).await.expect("insert in spawned task");
            }
        });
        handles.push(handle);
    }
    for h in handles {
        h.await.expect("task completed without panic");
    }
    let count = store.count().await.expect("count");
    assert_eq!(count, 160, "all 16*10 inserts must be counted");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_readers_one_writer() {
    let store: Arc<dyn MemoryStore> = make_store();
    // Pre-populate with some data so readers have something to find
    for i in 0..10usize {
        let rec = make_record(&format!("concurrent reader memory {i}"));
        store.insert(&rec).await.expect("pre-insert");
    }

    let mut handles = Vec::new();

    // 32 concurrent readers
    for _ in 0..32usize {
        let store = Arc::clone(&store);
        let handle = tokio::spawn(async move {
            let results = store.fts_search("concurrent reader", 5).await;
            // Should not panic even if results vary due to concurrent writes
            results.expect("fts_search in concurrent reader");
        });
        handles.push(handle);
    }

    // 1 writer inserting more records
    {
        let store = Arc::clone(&store);
        let writer_handle = tokio::spawn(async move {
            for i in 0..20usize {
                let rec = make_record(&format!("writer record {i}"));
                store.insert(&rec).await.expect("concurrent insert");
            }
        });
        handles.push(writer_handle);
    }

    for h in handles {
        h.await.expect("no panics");
    }

    // Eventual consistency check: total count should be at least 10 (initial)
    let count = store.count().await.expect("final count");
    assert!(count >= 10, "at least the pre-populated records must persist");
}
