//! End-to-end migration tests. The SQLite path uses the committed schema-38
//! fixture (vestige-core's migration SQL). The portable-archive path is a
//! hand-built JSON archive, because this crate must not open SQLite
//! read-write — that would disarm the v3 guard.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use strata_kernel::event::ReviewEvent;
use strata_kernel::fsrs::{CardPhase, ALGO_V1};
use strata_kernel::kernel::Kernel;
use vestige_core::storage::{PortableTable, PortableValue};
use vestige_core::PortableArchive;

use strata_migrate::records::EdgeRecord;
use strata_migrate::{migrate, read_snapshot};

const N0: &str = "11111111-1111-4111-8111-111111111111";
const N1: &str = "22222222-2222-4222-8222-222222222222";
const N2: &str = "33333333-3333-4333-8333-333333333333";

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

fn text(value: &str) -> PortableValue {
    PortableValue::Text(value.to_string())
}

fn table(name: &str, columns: &[&str], rows: Vec<Vec<PortableValue>>) -> PortableTable {
    PortableTable {
        name: name.to_string(),
        columns: columns.iter().map(|c| (*c).to_string()).collect(),
        rows,
    }
}

fn sample_archive() -> PortableArchive {
    PortableArchive {
        archive_format: "vestige.portable.v1".to_string(),
        vestige_version: "test".to_string(),
        schema_version: 38,
        exported_at: DateTime::<Utc>::from_timestamp(0, 0).expect("epoch"),
        mode: "exact".to_string(),
        tables: vec![
            table(
                "knowledge_nodes",
                &[
                    "id",
                    "content",
                    "node_type",
                    "tags",
                    "created_at",
                    "updated_at",
                    "last_accessed",
                ],
                vec![
                    vec![
                        text(N0),
                        text("Rust borsh encodes enums as u8 ordinals"),
                        text("fact"),
                        text(r#"["rust","wire"]"#),
                        text("2026-01-15T10:00:00+00:00"),
                        text("2026-02-20T11:30:00+00:00"),
                        text("2026-03-01T09:15:00+00:00"),
                    ],
                    vec![
                        text(N1),
                        text("STRATA seals signed segment trailers"),
                        text("fact"),
                        text(r#"["strata"]"#),
                        text("2026-01-15T10:00:00+00:00"),
                        text("2026-02-20T11:30:00+00:00"),
                        text("2026-03-01T09:15:00+00:00"),
                    ],
                    vec![
                        text(N2),
                        text("Run the migration once, then write new memories to STRATA"),
                        text("procedure"),
                        text("[]"),
                        text("2026-01-15T10:00:00+00:00"),
                        text("2026-02-20T11:30:00+00:00"),
                        text("2026-03-01T09:15:00+00:00"),
                    ],
                ],
            ),
            table(
                "memory_connections",
                &[
                    "source_id",
                    "target_id",
                    "strength",
                    "link_type",
                    "created_at",
                    "last_activated",
                    "activation_count",
                ],
                vec![vec![
                    text(N0),
                    text(N1),
                    PortableValue::Real(0.75),
                    text("semantic"),
                    text("2026-01-16T08:00:00+00:00"),
                    text("2026-03-02T08:00:00+00:00"),
                    PortableValue::Integer(3),
                ]],
            ),
            table(
                "fsrs_cards",
                &["memory_id", "reps", "lapses"],
                vec![
                    vec![
                        text(N0),
                        PortableValue::Integer(4),
                        PortableValue::Integer(1),
                    ],
                    vec![
                        text(N1),
                        PortableValue::Integer(2),
                        PortableValue::Integer(0),
                    ],
                ],
            ),
        ],
    }
}

fn write_archive(path: &Path, archive: &PortableArchive) {
    std::fs::write(path, serde_json::to_vec(archive).expect("archive json")).expect("write");
}

fn schema38_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v3.1.1-sample.sqlite")
}

/// Tables the migration emits as frames (mirror of the crate's EMITTED_TABLES).
const MAPPED: &[&str] = &[
    "knowledge_nodes",
    "memory_connections",
    "fsrs_cards",
    "sync_tombstones",
    "deletion_tombstones",
    "walk_receipts",
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
    let archive_path = tmp.path().join("portable.json");
    let archive = sample_archive();
    write_archive(&archive_path, &archive);
    let ids = [N0.to_string(), N1.to_string(), N2.to_string()];

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
    std::fs::copy(schema38_fixture(), &db).expect("copy schema 38 fixture");

    let strata_dir = tmp.path().join("strata");
    let report = migrate(&db, &strata_dir).expect("migrate direct sqlite");

    assert_eq!(report.nodes, 4);
    assert_eq!(report.edges, 2);
    assert_eq!(report.fsrs_events, 5);
    assert!(report.verify_passed);

    let opened = strata_migrate::open_migrated(&strata_dir).expect("reopen");
    let log = &opened.log;
    let snapshot = read_snapshot(&log).expect("snapshot");
    assert_eq!(snapshot.nodes.len(), 4);
    assert_eq!(snapshot.legacy_links.len(), 1);
    assert_eq!(snapshot.edges.len(), 2);

    // Direct-SQLite path: skipped_tables is sqlite_master-driven (audit 17/18).
    let expected = expected_skipped(&db);
    assert!(
        expected.contains(&"schema_version".to_string()),
        "sanity: the real schema names schema_version"
    );
    let mut got = report.skipped_tables.clone();
    got.sort();
    assert_eq!(got, expected, "skipped tables must be schema-driven");

    let receipt = snapshot.receipt.expect("receipt");
    let dropped = &receipt.body.dropped_columns;
    assert!(
        dropped.iter().any(|c| c == "schema_version.applied_at"),
        "applied_at is a real dropped column: {dropped:?}"
    );
    assert!(
        !dropped.iter().any(|c| c == "schema_version.version"),
        "schema version is carried"
    );
    assert!(
        !dropped.iter().any(|c| c.starts_with("knowledge_nodes.")),
        "knowledge_nodes columns are carried: {dropped:?}"
    );
    let mut sorted = dropped.clone();
    sorted.sort();
    assert_eq!(sorted, *dropped, "dropped columns are ordered");
    assert!(report.skipped_tables.iter().any(|t| t == "schema_version"));
    assert!(!report.skipped_tables.iter().any(|t| t == "knowledge_nodes"));
}

#[test]
fn directory_source_resolves_vestige_db() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::copy(schema38_fixture(), tmp.path().join("vestige.db")).expect("copy");

    let report = migrate(tmp.path(), &tmp.path().join("strata")).expect("migrate dir");
    assert_eq!(report.nodes, 4);
    assert!(report.verify_passed);
}

#[test]
fn empty_store_migrates_to_verifying_log() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let archive_path = tmp.path().join("empty.json");
    let archive = PortableArchive {
        archive_format: "vestige.portable.v1".to_string(),
        vestige_version: "test".to_string(),
        schema_version: 38,
        exported_at: DateTime::<Utc>::from_timestamp(0, 0).expect("epoch"),
        mode: "exact".to_string(),
        tables: Vec::new(),
    };
    write_archive(&archive_path, &archive);

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
    write_archive(&archive_path, &sample_archive());

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
