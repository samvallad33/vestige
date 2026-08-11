//! Phase 1 integration tests: domain column migration and schema upgrade.

use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;
use vestige_core::storage::{MemoryRecord, MemoryStore, SqliteMemoryStore};

#[tokio::test]
async fn fresh_db_has_v16_schema() {
    let dir = tempdir().unwrap();
    let db = dir.path().join("fresh.db");
    let _store = SqliteMemoryStore::new(Some(db.clone())).expect("create");
    // Open a raw connection and check pragma
    let conn = rusqlite::Connection::open(&db).expect("open");
    let cols: Vec<String> = {
        let mut stmt = conn.prepare("PRAGMA table_info(knowledge_nodes)").unwrap();
        stmt.query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    };
    assert!(
        cols.contains(&"domains".to_string()),
        "domains column must exist: {:?}",
        cols
    );
    assert!(
        cols.contains(&"domain_scores".to_string()),
        "domain_scores column must exist"
    );
}

#[tokio::test]
async fn v11_db_upgrades_cleanly() {
    use vestige_core::storage::MIGRATIONS;
    let dir = tempdir().unwrap();
    let db = dir.path().join("v11.db");
    // Create DB with V11 migrations only
    {
        let conn = rusqlite::Connection::open(&db).expect("open");
        for m in MIGRATIONS.iter().filter(|m| m.version <= 11) {
            if m.version == 2 {
                conn.execute_batch(
                    "ALTER TABLE knowledge_nodes ADD COLUMN valid_from TEXT;\n\
                     ALTER TABLE knowledge_nodes ADD COLUMN valid_until TEXT;\n\
                     UPDATE schema_version SET version = 2, applied_at = datetime('now');",
                )
                .expect("apply V2 columns before its dependent indexes");
                continue;
            }
            conn.execute_batch(m.up).expect("apply migration");
        }
        // Insert 5 rows under V11 schema
        for i in 0..5usize {
            conn.execute(
                "INSERT INTO knowledge_nodes (id, content, node_type, created_at, updated_at, \
                 last_accessed, stability, difficulty, reps, lapses, learning_state, \
                 storage_strength, retrieval_strength, retention_strength, \
                 next_review, scheduled_days, has_embedding) \
                 VALUES (?1, ?2, 'fact', datetime('now'), datetime('now'), datetime('now'), \
                 1.0, 0.3, 0, 0, 'new', 1.0, 1.0, 1.0, datetime('now'), 1, 0)",
                rusqlite::params![format!("pre-v16-{i}"), format!("content {i}"),],
            )
            .expect("insert pre-v16 row");
        }
    }
    // Upgrade by opening through SqliteMemoryStore (triggers full migration)
    let _store = SqliteMemoryStore::new(Some(db.clone())).expect("open with v16");
    // Check all 5 rows have empty domains/domain_scores
    let conn = rusqlite::Connection::open(&db).expect("open raw");
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM knowledge_nodes WHERE domains='[]' AND domain_scores='{}'",
            [],
            |row| row.get(0),
        )
        .expect("count");
    assert_eq!(
        count, 5,
        "all pre-v16 rows must have empty domains/domain_scores"
    );
}

#[tokio::test]
async fn empty_domains_serialize_as_brackets() {
    let dir = tempdir().unwrap();
    let db = dir.path().join("empty_domains.db");
    let store = SqliteMemoryStore::new(Some(db.clone())).expect("create");
    let rec = MemoryRecord {
        id: Uuid::new_v4(),
        domains: vec![],
        domain_scores: Default::default(),
        content: "test content".to_string(),
        node_type: "fact".to_string(),
        tags: vec![],
        embedding: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        metadata: serde_json::json!({}),
    };
    store.insert(&rec).await.expect("insert");
    // Check raw sqlite value
    let conn = rusqlite::Connection::open(&db).expect("open raw");
    let (domains, domain_scores): (String, String) = conn
        .query_row(
            "SELECT domains, domain_scores FROM knowledge_nodes LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("query");
    assert_eq!(
        domains, "[]",
        "empty domains should store as '[]', not NULL"
    );
    assert_eq!(
        domain_scores, "{}",
        "empty domain_scores should store as '{{}}'"
    );
}

#[tokio::test]
async fn populated_domains_round_trip() {
    let dir = tempdir().unwrap();
    let db = dir.path().join("populated.db");
    let store: Arc<dyn MemoryStore> = Arc::new(SqliteMemoryStore::new(Some(db)).expect("create"));
    let mut rec = MemoryRecord {
        id: Uuid::new_v4(),
        domains: vec!["dev".to_string(), "infra".to_string()],
        domain_scores: {
            let mut m = std::collections::HashMap::new();
            m.insert("dev".to_string(), 0.82);
            m.insert("infra".to_string(), 0.71);
            m
        },
        content: "populated domains test".to_string(),
        node_type: "fact".to_string(),
        tags: vec![],
        embedding: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        metadata: serde_json::json!({}),
    };
    let id = store.insert(&rec).await.expect("insert");
    // Update the domains via update()
    rec.id = id;
    store.update(&rec).await.expect("update with domains");
    // Read back and verify
    let got = store.get(id).await.expect("get").expect("exists");
    let mut expected_domains = got.domains.clone();
    expected_domains.sort();
    assert_eq!(expected_domains, vec!["dev", "infra"]);
    assert!((got.domain_scores["dev"] - 0.82).abs() < 0.001);
    assert!((got.domain_scores["infra"] - 0.71).abs() < 0.001);
}

#[tokio::test]
async fn domains_table_exists() {
    let dir = tempdir().unwrap();
    let db = dir.path().join("domains_table.db");
    let _store = SqliteMemoryStore::new(Some(db.clone())).expect("create");
    let conn = rusqlite::Connection::open(&db).expect("open raw");
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='domains'",
            [],
            |row| row.get(0),
        )
        .expect("query");
    assert_eq!(count, 1, "domains table must exist after V16 migration");
}
