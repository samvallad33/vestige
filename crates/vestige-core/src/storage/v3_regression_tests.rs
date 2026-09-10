//! Release regression cases exercising durable v3 state and failure boundaries.
use super::*;
use std::sync::{Arc, Barrier};

fn node(store: &SqliteMemoryStore, content: &str, scope: &str) -> String {
    store
        .ingest_in_scope(
            IngestInput {
                content: content.into(),
                node_type: "fact".into(),
                ..Default::default()
            },
            scope,
        )
        .unwrap()
        .id
}
fn edge(store: &SqliteMemoryStore, from: &str, to: &str) {
    store
        .save_connection(&ConnectionRecord {
            source_id: from.into(),
            target_id: to.into(),
            strength: 1.0,
            link_type: "semantic".into(),
            created_at: Utc::now(),
            last_activated: Utc::now(),
            activation_count: 0,
        })
        .unwrap();
}
fn state(store: &SqliteMemoryStore, id: &str) -> String {
    SqliteMemoryStore::suppression_state_on(&store.reader.lock().unwrap(), id).unwrap()
}

#[test]
fn v3_cascade_restart_preserves_undo_and_scope_protection() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("restart.db");
    let store = SqliteMemoryStore::new(Some(path.clone())).unwrap();
    let seed = node(&store, "seed", "project");
    let same = node(&store, "same scope", "project");
    let foreign = node(&store, "foreign scope", "other");
    let protected = node(&store, "protected", "project");
    let suppressed = node(&store, "suppressed", "project");
    store
        .writer
        .lock()
        .unwrap()
        .execute(
            "UPDATE knowledge_nodes SET protected=1 WHERE id=?1",
            params![protected],
        )
        .unwrap();
    store.suppress_memory(&suppressed).unwrap();
    let ids = [&same, &foreign, &protected, &suppressed];
    let before: Vec<_> = ids.iter().map(|id| state(&store, id)).collect();
    for id in ids {
        edge(&store, &seed, id);
    }
    store.suppress_memory(&seed).unwrap();
    assert_eq!(store.apply_rac1_cascade(&seed).unwrap(), 1);
    assert_ne!(state(&store, &same), before[0]);
    for (i, id) in ids.iter().enumerate().skip(1) {
        assert_eq!(state(&store, id), before[i]);
    }
    drop(store);
    let reopened = SqliteMemoryStore::new(Some(path)).unwrap();
    assert_eq!(reopened.apply_rac1_cascade(&seed).unwrap(), 0);
    reopened.reverse_suppression(&seed, 24).unwrap();
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(state(&reopened, id), before[i]);
    }
    assert_eq!(
        reopened
            .reader
            .lock()
            .unwrap()
            .query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))
            .unwrap(),
        "ok"
    );
}

#[test]
fn v3_cascade_journal_failure_rolls_back_neighbor_and_retry_applies_once() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteMemoryStore::new(Some(dir.path().join("atomic.db"))).unwrap();
    let seed = node(&store, "atomic seed", "user");
    let neighbor = node(&store, "atomic neighbor", "user");
    edge(&store, &seed, &neighbor);
    store.suppress_memory(&seed).unwrap();
    let before = state(&store, &neighbor);
    store.writer.lock().unwrap().execute_batch("CREATE TEMP TRIGGER fail_cascade BEFORE INSERT ON suppression_cascade_effects BEGIN SELECT RAISE(ABORT,'fixture failure'); END;").unwrap();
    assert!(store.apply_rac1_cascade(&seed).is_err());
    assert_eq!(state(&store, &neighbor), before);
    store
        .writer
        .lock()
        .unwrap()
        .execute_batch("DROP TRIGGER fail_cascade")
        .unwrap();
    assert_eq!(store.apply_rac1_cascade(&seed).unwrap(), 1);
    assert_eq!(store.apply_rac1_cascade(&seed).unwrap(), 0);
}

#[test]
fn v3_concurrent_cascade_writers_share_one_durable_effect() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("concurrent.db");
    let first = SqliteMemoryStore::new(Some(path.clone())).unwrap();
    let second = SqliteMemoryStore::new(Some(path)).unwrap();
    let seed = node(&first, "concurrent seed", "user");
    let neighbor = node(&first, "concurrent neighbor", "user");
    edge(&first, &seed, &neighbor);
    let before = state(&first, &neighbor);
    first.suppress_memory(&seed).unwrap();
    let barrier = Arc::new(Barrier::new(2));
    std::thread::scope(|scope| {
        let a = scope.spawn(|| {
            barrier.wait();
            first.apply_rac1_cascade(&seed).unwrap()
        });
        let b = scope.spawn(|| {
            barrier.wait();
            second.apply_rac1_cascade(&seed).unwrap()
        });
        assert_eq!(a.join().unwrap() + b.join().unwrap(), 1);
    });
    first.reverse_suppression(&seed, 24).unwrap();
    assert_eq!(state(&second, &neighbor), before);
}

#[test]
fn v3_lifecycle_failure_rolls_back_entire_page() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteMemoryStore::new(Some(dir.path().join("lifecycle.db"))).unwrap();
    let mut ids = [
        node(&store, "first lifecycle", "user"),
        node(&store, "second lifecycle", "user"),
    ];
    ids.sort();
    store
        .writer
        .lock()
        .unwrap()
        .execute(
            "UPDATE knowledge_nodes SET retention_strength=0.01,retrieval_strength=0.01",
            [],
        )
        .unwrap();
    let before: Vec<_> = ids.iter().map(|id| state(&store, id)).collect();
    store.writer.lock().unwrap().execute_batch(&format!("CREATE TEMP TRIGGER fail_second BEFORE UPDATE ON knowledge_nodes WHEN OLD.id='{}' BEGIN SELECT RAISE(ABORT,'fixture failure'); END;",ids[1])).unwrap();
    assert!(
        store
            .maintain_lifecycle_batch(10, None, 10000, false)
            .is_err()
    );
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(state(&store, id), before[i]);
    }
}

#[test]
fn v3_gc_cursor_survives_deleting_its_own_page() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteMemoryStore::new(Some(dir.path().join("gc.db"))).unwrap();
    for i in 0..37 {
        node(&store, &format!("GC fixture {i}"), "user");
    }
    store
        .writer
        .lock()
        .unwrap()
        .execute("UPDATE knowledge_nodes SET retention_strength=0", [])
        .unwrap();
    let mut cursor = None;
    let mut deleted = 0;
    for batch in [1, 2, 3, 5, 8, 13, 21] {
        let page = store
            .maintain_gc_batch(batch, cursor.as_deref(), 10000, false, 0.1, None)
            .unwrap();
        deleted += page["deleted"].as_u64().unwrap();
        cursor = page["nextCursor"].as_str().map(str::to_string);
        if page["hasMore"] == false {
            break;
        }
    }
    assert_eq!(deleted, 37);
    assert_eq!(store.get_stats().unwrap().total_nodes, 0);
    assert_eq!(
        store
            .maintain_gc_batch(3, cursor.as_deref(), 10000, false, 0.1, None)
            .unwrap()["deleted"],
        0
    );
}

#[test]
fn v3_purge_erases_neighbor_snapshots_and_keeps_foreign_keys_valid() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteMemoryStore::new(Some(dir.path().join("purge.db"))).unwrap();
    let seed = node(&store, "purge seed", "user");
    let neighbor = node(&store, "purge neighbor", "user");
    edge(&store, &seed, &neighbor);
    store.suppress_memory(&seed).unwrap();
    store.apply_rac1_cascade(&seed).unwrap();
    store.purge_node(&neighbor, Some("test fixture")).unwrap();
    store.reverse_suppression(&seed, 24).unwrap();
    store.purge_node(&seed, Some("test fixture")).unwrap();
    let reader = store.reader.lock().unwrap();
    for table in ["suppression_operations", "suppression_cascade_effects"] {
        assert_eq!(
            reader
                .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
    assert!(
        !reader
            .prepare("PRAGMA foreign_key_check")
            .unwrap()
            .exists([])
            .unwrap()
    );
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
#[test]
fn v3_delayed_embedding_cannot_resurrect_a_purged_memory() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteMemoryStore::new(Some(dir.path().join("embedding-purge.db"))).unwrap();
    let id = node(&store, "delayed embedding", "user");
    let profile = store.active_embedding_profile().unwrap().unwrap();
    let vector = vec![0.0; EMBEDDING_DIMENSIONS];
    let bytes = Embedding::new(vector.clone()).to_bytes();
    store.purge_node(&id, Some("fixture")).unwrap();
    assert!(
        store
            .persist_node_embedding(
                &id,
                &bytes,
                "fixture",
                &vector,
                true,
                ("delayed embedding", profile.profile_id.as_str())
            )
            .is_err()
    );
    let reader = store.reader.lock().unwrap();
    for table in ["node_embeddings", "embedding_profile_vectors"] {
        assert_eq!(
            reader
                .query_row(
                    &format!("SELECT COUNT(*) FROM {table} WHERE node_id=?1"),
                    params![id],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
            0
        );
    }
    assert!(
        !reader
            .prepare("PRAGMA foreign_key_check")
            .unwrap()
            .exists([])
            .unwrap()
    );
}

#[test]
fn v3_upgrade_preserves_legacy_suppression_without_inventing_undo() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("upgrade.db");
    let store = SqliteMemoryStore::new(Some(path.clone())).unwrap();
    let id = node(&store, "legacy suppression fixture", "user");
    store.suppress_memory(&id).unwrap();
    let legacy_state = state(&store, &id);
    store
        .writer
        .lock()
        .unwrap()
        .execute_batch(
            "DROP TABLE suppression_cascade_effects; DROP TABLE suppression_operations;
         UPDATE schema_version SET version=33;",
        )
        .unwrap();
    drop(store);
    let upgraded = SqliteMemoryStore::new(Some(path.clone())).unwrap();
    assert_eq!(state(&upgraded, &id), legacy_state);
    assert!(upgraded.reverse_suppression(&id, 24).is_err());
    assert_eq!(state(&upgraded, &id), legacy_state);
    upgraded.suppress_memory(&id).unwrap();
    upgraded.reverse_suppression(&id, 24).unwrap();
    assert_eq!(state(&upgraded, &id), legacy_state);
    drop(upgraded);
    let reopened = SqliteMemoryStore::new(Some(path)).unwrap();
    assert_eq!(state(&reopened, &id), legacy_state);
    assert!(reopened.reverse_suppression(&id, 24).is_err());
}
