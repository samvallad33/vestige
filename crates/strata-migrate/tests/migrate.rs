//! End-to-end migration tests: real vestige-core SQLite store -> portable
//! export -> STRATA log -> reopen and assert.
//!
//! Fixture strategy: build a small store through the PUBLIC storage API
//! (ingest + save_connection), plant `fsrs_cards` rows with direct SQL (the
//! public API does not create review history), export, then migrate via both
//! supported source paths and assert against the reopened log.

use std::collections::HashSet;

use chrono::Utc;
use strata_kernel::event::ReviewEvent;
use strata_kernel::fsrs::{CardPhase, ALGO_V1};
use strata_kernel::kernel::Kernel;
use vestige_core::{ConnectionRecord, IngestInput, SqliteMemoryStore};

use strata_migrate::records::EdgeRecord;
use strata_migrate::{migrate, read_snapshot};

fn causal_reaches(edges: &[EdgeRecord], start: u64, target: u64) -> bool {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![start];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        for edge in edges {
            if edge.source_kernel_id == id {
                stack.push(edge.target_kernel_id);
            }
        }
    }
    seen.contains(&target)
}

/// Build a small populated store at `db`, optionally exporting its portable
/// archive to `archive_out` while the (single, legitimate) read-write
/// handle is open — a built store carries the SQLite magic, so 4.0 refuses
/// any later read-write open by design. Returns the ingested node ids.
fn build_store(db: &std::path::Path, archive_out: Option<&std::path::Path>) -> Vec<String> {
    let storage = SqliteMemoryStore::new(Some(db.to_path_buf())).expect("open store");

    let inputs = [
        IngestInput {
            content: "Rust borsh encodes enums as u8 ordinals".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["rust".to_string(), "wire".to_string()],
            ..Default::default()
        },
        IngestInput {
            content: "STRATA seals signed segment trailers".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["strata".to_string()],
            ..Default::default()
        },
        IngestInput {
            content: "Run the migration once, then write new memories to STRATA".to_string(),
            node_type: "procedure".to_string(),
            tags: vec![],
            ..Default::default()
        },
    ];
    let mut ids = Vec::new();
    for input in inputs {
        let node = storage.ingest(input).expect("ingest");
        ids.push(node.id.clone());
    }

    let now = Utc::now();
    storage
        .save_connection(&ConnectionRecord {
            source_id: ids[0].clone(),
            target_id: ids[1].clone(),
            strength: 0.75,
            link_type: "semantic".to_string(),
            created_at: now,
            last_activated: now,
            activation_count: 3,
        })
        .expect("save connection");

    // Plant FSRS review history while the store handle is open: a separate
    // raw connection commits the rows, and the export below must see them.
    // (The public API does not create review history.)
    {
        let conn = rusqlite::Connection::open(db).expect("open raw sqlite");
        for (memory_id, reps, lapses) in [(&ids[0], 4i64, 1i64), (&ids[1], 2, 0)] {
            conn.execute(
                "INSERT OR REPLACE INTO fsrs_cards (
                     memory_id, difficulty, stability, state, reps, lapses,
                     last_review, due_date, elapsed_days, scheduled_days
                 ) VALUES (?1, 5.0, 3.2, 'review', ?2, ?3, ?4, ?4, 1, 1)",
                rusqlite::params![memory_id, reps, lapses, now.to_rfc3339(),],
            )
            .expect("plant fsrs_cards");
        }
    }

    if let Some(archive_path) = archive_out {
        storage
            .export_portable_archive_to_path(archive_path)
            .expect("export archive");
    }
    drop(storage);

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
/// audit required skipped_tables to be schema-driven, not hand-listed).
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
        .filter_map(Result::ok)
        .collect();
    let mut out: Vec<String> = names
        .into_iter()
        .filter(|name| !MAPPED.contains(&name.as_str()))
        .filter(|name| {
            let quoted = format!("\"{name}\"");
            conn.query_row(&format!("SELECT COUNT(*) FROM {quoted}"), [], |row| {
                row.get::<_, i64>(0)
            })
            .map(|n| n > 0)
            .unwrap_or(false)
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
    assert_eq!(
        report.edges, 0,
        "the only v3 row is semantic and is not an edge"
    );
    assert_eq!(report.fsrs_events, 6); // 4 + 2 synthetic reviews
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
    let opened = strata_migrate::open_migrated(&strata_dir).expect("reopen strata log");
    let log = &opened.log;
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

    assert!(
        snapshot.edges.is_empty(),
        "semantic must not be a causal edge"
    );
    assert_eq!(snapshot.legacy_links.len(), 1);
    let link = &snapshot.legacy_links[0];
    assert_eq!(link.source_legacy_id, ids[0]);
    assert_eq!(link.target_legacy_id, ids[1]);
    assert_eq!(link.legacy_link_type, "semantic");
    assert_eq!(link.activation_count, 3);
    let src = snapshot
        .nodes
        .iter()
        .find(|n| n.legacy_id == ids[0])
        .unwrap()
        .kernel_id;
    let dst = snapshot
        .nodes
        .iter()
        .find(|n| n.legacy_id == ids[1])
        .unwrap()
        .kernel_id;
    assert_eq!(link.source_kernel_id, src);
    assert!(
        !causal_reaches(&snapshot.edges, src, dst),
        "a causal walk must not traverse the semantic row"
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
    assert_eq!(report.edges, 0);
    assert_eq!(report.fsrs_events, 6);
    assert!(report.verify_passed);

    let opened = strata_migrate::open_migrated(&strata_dir).expect("reopen");
    let log = &opened.log;
    let snapshot = read_snapshot(&log).expect("snapshot");
    assert_eq!(snapshot.nodes.len(), 3);
    assert_eq!(snapshot.legacy_links.len(), 1);
    assert!(snapshot.edges.is_empty());

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
    let db = tmp.path().join("vestige.db");
    let archive_path = tmp.path().join("empty.json");
    {
        let storage = SqliteMemoryStore::new(Some(db)).expect("open");
        storage
            .export_portable_archive_to_path(&archive_path)
            .expect("export");
    }

    let report = migrate(&archive_path, &tmp.path().join("strata")).expect("migrate empty");
    assert_eq!(report.nodes, 0);
    assert_eq!(report.edges, 0);
    assert_eq!(report.fsrs_events, 0);
    assert!(report.verify_passed, "empty fold must verify: {report:?}");

    let opened = strata_migrate::open_migrated(&tmp.path().join("strata")).expect("reopen");
    let log = &opened.log;
    let snapshot = read_snapshot(&log).expect("snapshot");
    assert_eq!(snapshot.checkpoints.len(), 1);
    assert!(snapshot.reviews.is_empty());
}

#[test]
fn partial_destination_rolls_back_and_a_completed_one_is_refused() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let archive_path = tmp.path().join("portable.json");
    {
        let db = tmp.path().join("vestige.db");
        build_store(&db, Some(&archive_path));
    }

    let strata_dir = tmp.path().join("strata");
    std::fs::create_dir_all(&strata_dir).expect("dest");
    let torn = strata_dir.join("torn.seg");
    std::fs::write(&torn, b"killed mid-run").expect("torn segment");

    let partial = migrate(&archive_path, &strata_dir);
    let msg = partial.as_ref().unwrap_err().to_string();
    assert!(
        msg.contains("--rollback"),
        "refusal must name --rollback: {msg}"
    );
    assert!(
        matches!(
            partial,
            Err(strata_migrate::MigrationError::DestinationPartial { .. })
        ),
        "partial destination must refuse: {partial:?}"
    );
    assert!(torn.exists(), "refusal must not delete the partial dest");

    strata_migrate::rollback(&strata_dir).expect("rollback partial");
    assert!(!strata_dir.exists(), "rollback removes the partial dest");

    let first = migrate(&archive_path, &strata_dir).expect("migrate after rollback");
    assert!(first.verify_passed);
    assert_eq!(first.nodes, 3);

    let second = migrate(&archive_path, &strata_dir);
    assert!(
        matches!(
            second,
            Err(strata_migrate::MigrationError::DestinationComplete { .. })
        ),
        "completed destination must refuse: {second:?}"
    );
    let opened = strata_migrate::open_migrated(&strata_dir).expect("reopen");
    let snapshot = read_snapshot(&opened.log).expect("snapshot");
    assert_eq!(snapshot.nodes.len(), 3, "the refusal wrote nothing");
    assert_eq!(snapshot.checkpoints.len(), 1);
    drop(opened);

    let refused = strata_migrate::rollback(&strata_dir);
    assert!(
        matches!(
            refused,
            Err(strata_migrate::MigrationError::RollbackRefused { .. })
        ),
        "rollback must not delete a completed log: {refused:?}"
    );
    let opened = strata_migrate::open_migrated(&strata_dir).expect("still opens");
    assert_eq!(read_snapshot(&opened.log).expect("snapshot").nodes.len(), 3);
}
