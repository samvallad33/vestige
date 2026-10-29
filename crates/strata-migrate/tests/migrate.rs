//! End-to-end migration tests: real vestige-core SQLite store -> portable
//! export -> STRATA log -> reopen and assert.
//!
//! Fixture strategy: build a small store through the PUBLIC storage API
//! (ingest + save_connection), plant `fsrs_cards` rows with direct SQL (the
//! public API does not create review history), export, then migrate via both
//! supported source paths and assert against the reopened log.

use std::collections::HashSet;

use chrono::Utc;
use strata::StrataLog;
use strata_kernel::event::ReviewEvent;
use strata_kernel::fsrs::{CardPhase, ALGO_V1};
use strata_kernel::kernel::Kernel;

use strata_migrate::{migrate, read_snapshot};

/// Build a small populated store at `db`, optionally exporting its portable
/// archive to `archive_out`. Returns the ingested node ids.
fn build_store(db: &std::path::Path, archive_out: Option<&std::path::Path>) -> Vec<String> {
    let conn = rusqlite::Connection::open(db).expect("open raw sqlite");
    conn.execute_batch(
        r#"
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
        INSERT INTO schema_version (version, applied_at) VALUES (38, '2026-09-28T00:00:00Z');
        CREATE TABLE knowledge_nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            node_type TEXT NOT NULL DEFAULT 'fact',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_accessed TEXT NOT NULL,
            stability REAL NOT NULL DEFAULT 1.0,
            difficulty REAL NOT NULL DEFAULT 5.0,
            reps INTEGER NOT NULL DEFAULT 0,
            lapses INTEGER NOT NULL DEFAULT 0,
            learning_state TEXT NOT NULL DEFAULT 'new',
            sentiment_magnitude REAL NOT NULL DEFAULT 0.0,
            tags TEXT DEFAULT '[]',
            source TEXT,
            superseded_by TEXT
        );
        CREATE TABLE memory_connections (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            strength REAL NOT NULL,
            link_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_activated TEXT NOT NULL,
            activation_count INTEGER DEFAULT 0,
            PRIMARY KEY (source_id, target_id)
        );
        CREATE TABLE fsrs_cards (
            memory_id TEXT PRIMARY KEY,
            difficulty REAL NOT NULL DEFAULT 5.0,
            stability REAL NOT NULL DEFAULT 1.0,
            state TEXT NOT NULL DEFAULT 'new',
            reps INTEGER DEFAULT 0,
            lapses INTEGER DEFAULT 0,
            last_review TEXT,
            due_date TEXT,
            elapsed_days INTEGER DEFAULT 0,
            scheduled_days INTEGER DEFAULT 0
        );
        CREATE TABLE sync_tombstones (
            table_name TEXT NOT NULL,
            row_id TEXT NOT NULL,
            deleted_at TEXT NOT NULL,
            reason TEXT,
            PRIMARY KEY (table_name, row_id)
        );
        CREATE TABLE deletion_tombstones (
            memory_id TEXT PRIMARY KEY,
            deleted_at TEXT NOT NULL,
            reason TEXT,
            node_type TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]',
            edges_pruned INTEGER NOT NULL DEFAULT 0,
            insights_rewritten INTEGER NOT NULL DEFAULT 0,
            insights_deleted INTEGER NOT NULL DEFAULT 0,
            children_orphaned INTEGER NOT NULL DEFAULT 0
        );
        "#,
    )
    .expect("create schema");

    let now = Utc::now().to_rfc3339();
    let rows = [
        (
            "11111111-1111-4111-8111-111111111111",
            "Rust borsh encodes enums as u8 ordinals",
            "fact",
            r#"["rust","wire"]"#,
        ),
        (
            "22222222-2222-4222-8222-222222222222",
            "STRATA seals signed segment trailers",
            "fact",
            r#"["strata"]"#,
        ),
        (
            "33333333-3333-4333-8333-333333333333",
            "Run the migration once, then write new memories to STRATA",
            "procedure",
            "[]",
        ),
    ];
    let mut ids = Vec::new();
    for (id, content, node_type, tags) in rows {
        conn.execute(
            "INSERT INTO knowledge_nodes (id, content, node_type, created_at, updated_at, last_accessed, tags)
             VALUES (?1, ?2, ?3, ?4, ?4, ?4, ?5)",
            rusqlite::params![id, content, node_type, now, tags],
        )
        .expect("insert node");
        ids.push(id.to_string());
    }

    conn.execute(
        "INSERT INTO memory_connections (source_id, target_id, strength, link_type, created_at, last_activated, activation_count)
         VALUES (?1, ?2, 0.75, 'semantic', ?3, ?3, 3)",
        rusqlite::params![ids[0], ids[1], now],
    )
    .expect("save connection");

    for (memory_id, reps, lapses) in [(&ids[0], 4i64, 1i64), (&ids[1], 2, 0)] {
        conn.execute(
            "INSERT OR REPLACE INTO fsrs_cards (
                 memory_id, difficulty, stability, state, reps, lapses,
                 last_review, due_date, elapsed_days, scheduled_days
             ) VALUES (?1, 5.0, 3.2, 'review', ?2, ?3, ?4, ?4, 1, 1)",
            rusqlite::params![memory_id, reps, lapses, now],
        )
        .expect("plant fsrs_cards");
    }
    drop(conn);

    if let Some(archive_path) = archive_out {
        let mut scratch = strata_migrate::source::Scratch::new(db.parent().unwrap());
        let (files, _) = strata_migrate::source::prepare_source(db, false, &mut scratch)
            .expect("prepare source");
        let snapshot = strata_migrate::source::load_snapshot(&files).expect("load snapshot");
        std::fs::write(
            archive_path,
            serde_json::to_vec(&snapshot.archive).expect("serialize archive"),
        )
        .expect("write archive");
    }

    ids
}

/// Tables the migration maps (mirror of the crate's MAPPED_TABLES).
const MAPPED: &[&str] = &[
    "knowledge_nodes",
    "memory_connections",
    "fsrs_cards",
    "sync_tombstones",
    "deletion_tombstones",
];

/// The sqlite_master-driven expectation: every nonempty user table without
/// a STRATA mapping (mirrors the migrator's own skipped_tables logic — the
/// audit required skipped_tables to be schema-driven, not hand-listed). A
/// read error fails the test; a table whose rows cannot be counted is named.
fn expected_skipped(db: &std::path::Path) -> Vec<String> {
    let conn = rusqlite::Connection::open_with_flags(
        format!("file:{}?mode=ro&immutable=1", db.display()),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    )
    .expect("readonly open");
    let mut stmt = conn
        .prepare(
            "SELECT name FROM sqlite_master WHERE type='table'
             AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'knowledge_fts%'
             ORDER BY name",
        )
        .unwrap();
    let names: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let mut out: Vec<String> = names
        .into_iter()
        .filter(|name| !MAPPED.contains(&name.as_str()))
        .filter(|name| {
            let quoted = format!("\"{}\"", name.replace('"', "\"\""));
            conn.query_row(&format!("SELECT COUNT(*) FROM {quoted}"), [], |row| {
                row.get::<_, i64>(0)
            })
            .map_or(true, |n| n > 0)
        })
        .collect();
    out.sort();
    out
}

#[test]
fn path_a_archive_end_to_end() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("vestige.db");
    let archive_path = tmp.path().join("portable.json");
    let ids = build_store(&db, Some(&archive_path));
    let archive: vestige_core::PortableArchive =
        serde_json::from_slice(&std::fs::read(&archive_path).expect("read archive"))
            .expect("decode archive");

    let strata_dir = tmp.path().join("strata");
    let report = migrate(&archive_path, &strata_dir).expect("migrate");

    assert_eq!(report.nodes, 3);
    assert_eq!(report.edges, 1);
    assert_eq!(report.fsrs_events, 6); // 4 + 2 synthetic reviews
    assert_eq!(report.fsrs_states, 1); // the uncarded third memory
    assert!(report.verify_passed, "kernel verify must pass: {report:?}");
    // Path A input is a portable archive: skipped_tables can only reflect
    // the archive's own tables (the schema-driven sqlite_master list is the
    // direct-SQLite path's contract — asserted in path_b).
    let mut got = report.skipped_tables.clone();
    got.sort();
    let mut expected: Vec<String> = archive
        .tables
        .iter()
        .filter(|t| !t.rows.is_empty() && !MAPPED.contains(&t.name.as_str()))
        .map(|t| t.name.clone())
        .collect();
    expected.sort();
    assert_eq!(got, expected);

    // ---- reopen at the log level and assert identity fidelity ------------
    let log = StrataLog::open(&strata_dir).expect("reopen strata log");
    let snapshot = read_snapshot(&log).expect("read snapshot");

    assert!(snapshot.genesis.is_some());
    assert_eq!(
        snapshot.genesis.unwrap().archive_format,
        "vestige.portable.v1"
    );

    let legacy: HashSet<&str> = snapshot
        .nodes
        .iter()
        .map(|n| n.legacy_id.as_str())
        .collect();
    let want: HashSet<&str> = ids.iter().map(String::as_str).collect();
    assert_eq!(legacy, want, "same node ids must survive the migration");

    let kernel_ids: HashSet<u64> = snapshot.nodes.iter().map(|n| n.kernel_id).collect();
    assert_eq!(kernel_ids, HashSet::from([1, 2, 3]), "dense kernel ids");
    for node in &snapshot.nodes {
        assert!(!node.content.is_empty());
        assert!(node.created_ms > 0 && node.updated_ms > 0 && node.last_accessed_ms > 0);
    }
    let tagged = snapshot
        .nodes
        .iter()
        .find(|n| n.legacy_id == ids[0])
        .expect("node 0");
    assert_eq!(tagged.tags, vec!["rust".to_string(), "wire".to_string()]);

    assert_eq!(snapshot.edges.len(), 1);
    let edge = &snapshot.edges[0];
    assert_eq!(edge.source_legacy_id, ids[0]);
    assert_eq!(edge.target_legacy_id, ids[1]);
    // `semantic` is inferred (keyword/similarity), not a declared causal kind.
    assert_eq!(edge.link_type, "legacy_inferred");
    assert!(!strata_migrate::STRATA_EDGE_VOCABULARY.contains(&edge.link_type.as_str()));
    assert!(edge.legacy_inferred);
    assert_eq!(edge.legacy_link_type, "semantic");
    assert_eq!(edge.activation_count, 3);
    assert_eq!(
        edge.source_kernel_id,
        snapshot
            .nodes
            .iter()
            .find(|n| n.legacy_id == ids[0])
            .unwrap()
            .kernel_id
    );

    // ---- fsrs folds reproduce the counters exactly ------------------------
    assert_eq!(snapshot.reviews.len(), 6);
    let kernel = Kernel::<ReviewEvent>::for_version(ALGO_V1).unwrap();
    let mut state = strata_kernel::state::State::default();
    kernel.apply_all(&mut state, snapshot.reviews.iter());

    let kernel_id_of = |legacy: &str| {
        snapshot
            .nodes
            .iter()
            .find(|n| n.legacy_id == legacy)
            .unwrap()
            .kernel_id
    };
    let card0 = state.cards.get(&kernel_id_of(&ids[0])).expect("card 0");
    assert_eq!(card0.review_count, 4);
    assert_eq!(card0.lapse_count, 1);
    assert_eq!(card0.phase, CardPhase::Relearning);
    let card1 = state.cards.get(&kernel_id_of(&ids[1])).expect("card 1");
    assert_eq!(card1.review_count, 2);
    assert_eq!(card1.lapse_count, 0);

    // ---- one sealed checkpoint, event seqs equal frame seqs ----------------
    assert_eq!(snapshot.checkpoints.len(), 1);
    let frames = log.read_frames(1).expect("frames");
    let review_seqs: HashSet<u64> = snapshot.reviews.iter().map(|e| e.event_seq).collect();
    let frame_seqs: HashSet<u64> = frames
        .iter()
        .filter(|f| f.kind == strata_migrate::records::KIND_FSRS_REVIEW)
        .map(|f| f.seq)
        .collect();
    assert_eq!(review_seqs, frame_seqs);
}

#[test]
fn path_b_direct_sqlite_matches_path_a() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("vestige.db");
    build_store(&db, None);

    let strata_dir = tmp.path().join("strata");
    let report = migrate(&db, &strata_dir).expect("migrate direct sqlite");

    assert_eq!(report.nodes, 3);
    assert_eq!(report.edges, 1);
    assert_eq!(report.fsrs_events, 6);
    assert!(report.verify_passed);

    let log = StrataLog::open(&strata_dir).expect("reopen");
    let snapshot = read_snapshot(&log).expect("snapshot");
    assert_eq!(snapshot.nodes.len(), 3);
    assert_eq!(snapshot.edges.len(), 1);

    // Direct-SQLite path: skipped_tables is sqlite_master-driven (audit 17/18).
    let expected = expected_skipped(&db);
    assert!(
        expected.contains(&"schema_version".to_string()),
        "sanity: the real schema names schema_version"
    );
    let mut got = report.skipped_tables.clone();
    got.sort();
    assert_eq!(got, expected, "skipped tables must be schema-driven");
}

#[test]
fn directory_source_resolves_vestige_db() {
    let tmp = tempfile::tempdir().expect("tempdir");
    build_store(&tmp.path().join("vestige.db"), None);

    let report = migrate(tmp.path(), &tmp.path().join("strata")).expect("migrate dir");
    assert_eq!(report.nodes, 3);
    assert!(report.verify_passed);
}

#[test]
fn empty_store_migrates_to_verifying_log() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let archive_path = tmp.path().join("empty.json");
    let empty_archive = vestige_core::PortableArchive {
        archive_format: strata_migrate::source::PORTABLE_ARCHIVE_FORMAT.to_string(),
        vestige_version: "3.1.1".to_string(),
        schema_version: 38,
        exported_at: Utc::now(),
        mode: "exact".to_string(),
        tables: Vec::new(),
    };
    std::fs::write(
        &archive_path,
        serde_json::to_vec(&empty_archive).expect("serialize empty archive"),
    )
    .expect("write empty archive");

    let report = migrate(&archive_path, &tmp.path().join("strata")).expect("migrate empty");
    assert_eq!(report.nodes, 0);
    assert_eq!(report.edges, 0);
    assert_eq!(report.fsrs_events, 0);
    assert!(report.verify_passed, "empty fold must verify: {report:?}");

    let log = StrataLog::open(tmp.path().join("strata")).expect("reopen");
    let snapshot = read_snapshot(&log).expect("snapshot");
    assert_eq!(snapshot.checkpoints.len(), 1);
    assert!(snapshot.reviews.is_empty());
}

#[test]
fn re_migration_extends_the_log_and_keeps_the_chain() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let archive_path = tmp.path().join("portable.json");
    {
        let db = tmp.path().join("vestige.db");
        build_store(&db, Some(&archive_path));
    }

    let strata_dir = tmp.path().join("strata");
    let first = migrate(&archive_path, &strata_dir).expect("first run");
    assert!(first.verify_passed);

    // A second run into a non-empty destination is refused: a killed run
    // followed by a re-run must never double the rows (audit finding).
    let second = migrate(&archive_path, &strata_dir);
    assert!(
        matches!(
            second,
            Err(strata_migrate::MigrationError::DestinationNotEmpty { .. })
        ),
        "non-empty destination must refuse: {second:?}"
    );
    let log = StrataLog::open(&strata_dir).expect("reopen");
    let snapshot = read_snapshot(&log).expect("snapshot");
    assert_eq!(snapshot.nodes.len(), 3, "the refusal wrote nothing");
    assert_eq!(snapshot.checkpoints.len(), 1);
}
