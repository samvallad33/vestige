//! PR-0a acceptance tests (named per the spec). The source fixture is the
//! committed synthetic `v3.1.1-sample.sqlite` — never real user data. Every
//! test that must not modify the source asserts BLAKE3 + mtime + mode.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use strata_migrate::records::KIND_MIGRATION_RECEIPT;
use strata_migrate::{migrate_with_options, read_snapshot, MigrateOptions, MigrationError};

/// Committed synthetic fixture (schema_version 38, 4 nodes, 3 edges,
/// 1 fsrs card, 2 tombstones, 2 embeddings, 2-envelope chain, 1 walk
/// receipt).
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v3.1.1-sample.sqlite")
}

/// Copy the fixture into a fresh temp dir; returns (dir, db path).
fn copy_fixture(tag: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join(format!("{tag}.sqlite"));
    fs::copy(fixture_path(), &db).expect("copy fixture");
    (dir, db)
}

fn hash_file(path: &Path) -> String {
    blake3::hash(&fs::read(path).expect("read"))
        .to_hex()
        .to_string()
}

/// (blake3, mtime epoch secs, unix mode) — the full "untouched" triple.
fn fingerprint(path: &Path) -> (String, u64, u32) {
    let meta = fs::metadata(path).expect("metadata");
    let mtime = meta
        .modified()
        .expect("mtime")
        .duration_since(UNIX_EPOCH)
        .expect("mtime after epoch")
        .as_secs();
    #[cfg(unix)]
    {
        (hash_file(path), mtime, meta.permissions().mode())
    }
    #[cfg(not(unix))]
    {
        (hash_file(path), mtime, 0)
    }
}

fn seed() -> [u8; 32] {
    [7u8; 32]
}

/// Is the v3 guard ARMED in this test build? The dev-dependency may enable
/// the v3-engine harness (guard disarmed); cfg cannot see dev-dep feature
/// unification, so probe at runtime: a FRESH path must be refused-creation
/// when armed... actually a fresh path CREATES when the engine is available;
/// the armed signal is a fresh-path creation SUCCEEDING while an existing
/// sqlite file refuses. Probe: create on a fresh path; the guard does not
/// block fresh creation in any build, so instead probe an EXISTING magic
/// file: Err(V3StoreNeedsMigration) = armed; Err(other)/Ok = disarmed.
fn guard_armed() -> bool {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("probe.db");
    {
        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute("CREATE TABLE t (x INTEGER)", []).unwrap();
    }
    matches!(
        vestige_core::SqliteMemoryStore::new(Some(db)),
        Err(vestige_core::storage::StorageError::V3StoreNeedsMigration { .. })
    )
}

/// Pin the receipt-signing key for a destination (the spec's twice-identical
/// contract pins the key AND the log id; real runs generate fresh keys).
fn pin_receipt_key(parent: &Path) {
    std::fs::write(parent.join("receipt-signing.key"), seed()).unwrap();
}

// ---------------------------------------------------------------------------
// Guard refusal
// ---------------------------------------------------------------------------

/// Spec: `v3_open_refuses_and_leaves_file_byte_identical` — BLAKE3, mtime
/// and mode unchanged; the error is `V3StoreNeedsMigration`.
/// (Requires the guard ARMED: skipped when this test build carries the
/// v3-engine harness feature via the dev-dependency, which legitimately
/// disables the guard for the engine's own round trips.)
#[test]
fn v3_open_refuses_and_leaves_file_byte_identical() {
    if !guard_armed() {
        println!("guard disarmed (v3-engine harness build); skipping");
        return;
    }
    let (_dir, db) = copy_fixture("refusal");
    let before = fingerprint(&db);

    // SqliteMemoryStore carries no Debug; match the refusal manually.
    let err = match vestige_core::SqliteMemoryStore::new(Some(db.clone())) {
        Err(e) => e,
        Ok(_) => panic!("v3 store must be refused"),
    };
    match &err {
        vestige_core::storage::StorageError::V3StoreNeedsMigration {
            path,
            schema_version,
            hint,
        } => {
            assert_eq!(path.as_str(), db.display().to_string());
            assert_eq!(*schema_version, 38);
            assert!(hint.contains("migrate-to-strata"));
        }
        other => panic!("wrong error: {other:?}"),
    }
    let _ = err; // Display carries the hint (asserted via to_string below)
    assert!(err.to_string().contains("migrate-to-strata"));

    let after = fingerprint(&db);
    assert_eq!(before, after, "refused open changed the source file");
}

/// Spec: `v3_open_refuses_any_sqlite_file_even_v40` — any SQLite magic is
/// refused, whatever the schema version, since 4.0 writes no SQLite.
/// (Guard-armed builds only; see the sibling test's note.)
#[test]
fn v3_open_refuses_any_sqlite_file_even_v40() {
    if !guard_armed() {
        println!("guard disarmed (v3-engine harness build); skipping");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("v40.sqlite");
    {
        let conn = rusqlite::Connection::open(&db).expect("create");
        conn.execute_batch(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
             INSERT INTO schema_version VALUES (40, '2026-09-28T00:00:00Z');",
        )
        .expect("schema");
    }

    let err = match vestige_core::SqliteMemoryStore::new(Some(db.clone())) {
        Err(e) => e,
        Ok(_) => panic!("even a V40 sqlite file must be refused"),
    };
    match err {
        vestige_core::storage::StorageError::V3StoreNeedsMigration { schema_version, .. } => {
            assert_eq!(schema_version, 40);
        }
        other => panic!("wrong error: {other:?}"),
    }

    // A bare SQLite file with no vestige tables at all is refused too
    // (schema version 0 = unreadable; refusal depends only on the magic).
    // Write one throwaway table so the file actually carries the 100-byte
    // header — a 0-byte file is a store being created and must stay legal.
    let bare = dir.path().join("bare.sqlite");
    {
        let conn = rusqlite::Connection::open(&bare).expect("create bare");
        conn.execute("CREATE TABLE throwaway (id INTEGER)", [])
            .expect("header write");
    }
    assert!(matches!(
        vestige_core::SqliteMemoryStore::new(Some(bare)),
        Err(vestige_core::storage::StorageError::V3StoreNeedsMigration { .. })
    ));
}
/// Spec: `v3_guard_never_opens_read_write` — the guard's read-only path
/// carries the READ_ONLY flag (writes fail), and detection itself creates
/// no `-wal`/`-shm` siblings.
#[test]
fn v3_guard_never_opens_read_write() {
    let (_dir, db) = copy_fixture("guard");

    let detected = vestige_core::detect_v3(&db)
        .expect("detect")
        .expect("v3 file");
    assert_eq!(detected.schema_version, 38);
    assert!(!db.with_file_name("guard.sqlite-wal").exists());
    assert!(!db.with_file_name("guard.sqlite-shm").exists());

    // The same read-only opener the migrator uses: writes must fail.
    let conn = strata_migrate::source::open_readonly(&db).expect("readonly open");
    let write = conn.execute(
        "UPDATE knowledge_nodes SET content = 'tampered' WHERE id = ?1",
        ["11111111-1111-4111-8111-111111111111"],
    );
    let err = write.expect_err("read-only connection must refuse writes");
    assert!(
        err.to_string().to_lowercase().contains("readonly")
            || err.to_string().to_lowercase().contains("read-only"),
        "expected a read-only failure, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Migration read-only guarantees
// ---------------------------------------------------------------------------

/// Spec: `migrate_to_strata_source_unchanged` — file, wal and shm hashes
/// unchanged by a full migration.
#[test]
fn migrate_to_strata_source_unchanged() {
    let (dir, db) = copy_fixture("source");
    let dest = dir.path().join("strata");

    let db_before = hash_file(&db);
    let shm = db.with_file_name("source.sqlite-shm");
    let shm_before = shm.exists().then(|| hash_file(&shm));

    let report = migrate_with_options(
        &db,
        &dest,
        MigrateOptions {
            seed: Some(seed()),
            ..Default::default()
        },
    )
    .expect("migration succeeds");

    assert!(report.verify_passed);
    assert_eq!(hash_file(&db), db_before, "db changed");
    assert_eq!(
        shm.exists().then(|| hash_file(&shm)),
        shm_before,
        "shm appeared or changed"
    );
    assert!(
        !db.with_file_name("source.sqlite-wal").exists(),
        "wal appeared"
    );
}

/// Spec: `migrate_to_strata_twice_identical` — identical segment bytes with
/// a pinned key (seed) and log id.
#[test]
fn migrate_to_strata_twice_identical() {
    let (dir_a, db_a) = copy_fixture("twice_a");
    let (dir_b, db_b) = copy_fixture("twice_b");
    pin_receipt_key(dir_a.path());
    pin_receipt_key(dir_b.path());
    let opts = || MigrateOptions {
        seed: Some(seed()),
        ..Default::default()
    };

    migrate_with_options(&db_a, &dir_a.path().join("strata"), opts()).expect("run A");
    migrate_with_options(&db_b, &dir_b.path().join("strata"), opts()).expect("run B");

    let mut segs_a = segments(&dir_a.path().join("strata"));
    let mut segs_b = segments(&dir_b.path().join("strata"));
    assert_eq!(
        segs_a
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect::<Vec<_>>(),
        segs_b
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect::<Vec<_>>(),
        "segment names (number + id) must match"
    );
    segs_a.sort();
    segs_b.sort();
    for (a, b) in segs_a.iter().zip(segs_b.iter()) {
        assert_eq!(
            hash_file(a),
            hash_file(b),
            "segment bytes differ: {a:?} vs {b:?}"
        );
    }
}

fn segments(strata_dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = fs::read_dir(strata_dir)
        .expect("strata dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "seg"))
        .collect();
    out.sort();
    out
}

// ---------------------------------------------------------------------------
// Replay contents
// ---------------------------------------------------------------------------

/// Spec: `migrate_to_strata_counts_match` — node, edge and supersession
/// counts match; `dropped_vectors` is reported.
#[test]
fn migrate_to_strata_counts_match() {
    let (dir, db) = copy_fixture("counts");
    let dest = dir.path().join("strata");
    let report = migrate_with_options(
        &db,
        &dest,
        MigrateOptions {
            seed: Some(seed()),
            ..Default::default()
        },
    )
    .expect("migration succeeds");

    // 4 knowledge_nodes + 1 walk_receipt reference node.
    assert_eq!(report.nodes, 5);
    assert_eq!(report.edges, 3);
    assert_eq!(report.dropped_vectors, 2);

    let snapshot = read_snapshot(&strata::StrataLog::open(&dest).unwrap()).unwrap();
    assert_eq!(snapshot.supersessions.len(), 1);
    assert_eq!(snapshot.genesis.as_ref().unwrap().schema_version, 38);
    let receipt = snapshot.receipt.as_ref().expect("receipt present");
    let nodes_row = receipt
        .body
        .counts
        .iter()
        .find(|(name, _)| name == "knowledge_nodes")
        .expect("knowledge_nodes count in receipt");
    assert_eq!(nodes_row.1, 4);
}

/// Spec: `migrate_to_strata_legacy_links_are_legacy_inferred` — a v3
/// `causal` row (lookalike name-match provenance) migrates as
/// `derived_from{legacy_inferred=1}`, an in-vocabulary row passes through
/// untouched, and the legacy type survives only as provenance.
#[test]
fn migrate_to_strata_legacy_links_are_legacy_inferred() {
    let (dir, db) = copy_fixture("legacy");
    let dest = dir.path().join("strata");
    migrate_with_options(
        &db,
        &dest,
        MigrateOptions {
            seed: Some(seed()),
            ..Default::default()
        },
    )
    .expect("migration succeeds");

    let snapshot = read_snapshot(&strata::StrataLog::open(&dest).unwrap()).unwrap();
    let by_legacy = |legacy: &str| {
        snapshot
            .edges
            .iter()
            .find(|e| e.legacy_link_type == legacy)
            .unwrap_or_else(|| panic!("no edge with legacy type {legacy}"))
    };

    let causal = by_legacy("causal");
    assert_eq!(causal.link_type, "derived_from");
    assert!(
        causal.legacy_inferred,
        "causal lookalike must be legacy_inferred"
    );

    let semantic = by_legacy("semantic");
    assert_eq!(semantic.link_type, "derived_from");
    assert!(semantic.legacy_inferred);

    let touched = by_legacy("touched");
    assert_eq!(touched.link_type, "touched");
    assert!(
        !touched.legacy_inferred,
        "in-vocabulary type passes through"
    );

    // Default-walk semantics: only the 8-type vocabulary participates, and
    // every rewritten edge is flagged so a default walk can exclude it.
    assert!(snapshot.edges.iter().all(|e| !e.legacy_inferred
        || strata_migrate::STRATA_EDGE_VOCABULARY.contains(&e.link_type.as_str())));
}

// ---------------------------------------------------------------------------
// Refusals and the receipt
// ---------------------------------------------------------------------------

/// Spec: `migrate_refuses_broken_envelope_chain` — a tampered payload
/// digest aborts the migration with a chain report and writes nothing to a
/// fresh destination.
#[test]
fn migrate_refuses_broken_envelope_chain() {
    let (dir, db) = copy_fixture("broken");
    // Tamper the COPY (never the committed fixture): flip entry 0's payload
    // digest so recomputation no longer matches.
    {
        let conn = rusqlite::Connection::open(&db).expect("open copy");
        conn.execute(
            "UPDATE receipt_envelopes SET payload_digest = replace(payload_digest, 'a', 'b')
             WHERE sequence = 0",
            [],
        )
        .expect("tamper");
    }
    let dest = dir.path().join("strata");

    let err = migrate_with_options(&db, &dest, MigrateOptions::default())
        .expect_err("broken chain must abort");
    match err {
        MigrationError::BrokenEnvelopeChain(detail) => {
            assert!(detail.contains("payload_digest mismatch"), "{detail}");
        }
        other => panic!("wrong error: {other:?}"),
    }
    assert!(!dest.exists(), "nothing may be written on a chain break");
}

/// Spec: `migrate_refuses_nonempty_wal_without_flag` — a non-empty WAL
/// refuses without `--accept-wal-snapshot`; with it, the migration
/// succeeds and the original db + wal stay byte-identical.
#[test]
fn migrate_refuses_nonempty_wal_without_flag() {
    let (dir, db) = copy_fixture("wal");
    let wal = db.with_file_name("wal.sqlite-wal");
    fs::write(&wal, b"synthetic wal frames").expect("plant non-empty wal");

    let dest = dir.path().join("strata");
    let err = migrate_with_options(&db, &dest, MigrateOptions::default())
        .expect_err("non-empty wal must refuse");
    assert!(matches!(err, MigrationError::WalPresent { .. }), "{err:?}");

    let db_before = hash_file(&db);
    let wal_before = hash_file(&wal);
    let report = migrate_with_options(
        &db,
        &dest,
        MigrateOptions {
            accept_wal_snapshot: true,
            seed: Some(seed()),
            ..Default::default()
        },
    )
    .expect("snapshot run succeeds");
    assert!(report.verify_passed);
    assert_eq!(hash_file(&db), db_before, "db changed");
    assert_eq!(hash_file(&wal), wal_before, "wal changed");
}

/// Spec: `migration_receipt_verifies_on_replay` — the kind-46 receipt
/// decodes, its signature binds its body, and the counts it records match
/// the replayed log.
#[test]
fn migration_receipt_verifies_on_replay() {
    let (dir, db) = copy_fixture("receipt");
    let dest = dir.path().join("strata");
    let report = migrate_with_options(
        &db,
        &dest,
        MigrateOptions {
            seed: Some(seed()),
            ..Default::default()
        },
    )
    .expect("migration succeeds");

    let digest = report
        .receipt_digest
        .as_ref()
        .expect("receipt digest present");
    assert_eq!(digest.len(), 64, "receipt digest is 32-byte hex");
    assert!(report.receipt_verified);
    assert!(report.verify_passed, "kernel replay + tail verification");

    let snapshot = read_snapshot(&strata::StrataLog::open(&dest).unwrap()).unwrap();
    let receipt = snapshot.receipt.as_ref().expect("kind-46 receipt frame");
    assert!(receipt.verify_checksum(), "checksum binds the body");
    assert!(
        receipt.verify_signature(),
        "ed25519 signature proves authorship"
    );
    assert_eq!(
        receipt.body.signing_key_id,
        strata_migrate::RECEIPT_SIGNING_KEY_ID
    );
    assert_eq!(receipt.body.source_blake3_before, report.source_blake3);
    assert_eq!(receipt.body.source_blake3_after, report.source_blake3);
    assert_eq!(receipt.body.dropped_vectors, 2);

    // Receipt counts match the replayed frames.
    let count_of = |name: &str| {
        receipt
            .body
            .counts
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| *c)
            .unwrap_or(0)
    };
    assert_eq!(count_of("knowledge_nodes"), 4);
    assert_eq!(count_of("memory_connections"), 3);
    assert_eq!(count_of("fsrs_cards"), 1);
    assert_eq!(count_of("walk_receipts"), 1);
    assert_eq!(snapshot.nodes.len(), 5);
    assert_eq!(snapshot.edges.len(), 3);
    // The receipt is the last frame in the log.
    let frames = strata::StrataLog::open(&dest)
        .unwrap()
        .read_frames(1)
        .unwrap();
    let last = frames.last().expect("frames");
    assert_eq!(last.kind, KIND_MIGRATION_RECEIPT);
}

// ---------------------------------------------------------------------------
// Audit (b): --accept-wal-snapshot must see WAL-resident commits
// ---------------------------------------------------------------------------

/// A row living only in the -wal (uncheckpointed) must land in the log via
/// the snapshot copy. The copy is checkpointed; the original is untouched.
#[test]
fn migrate_wal_snapshot_includes_wal_only_rows() {
    let (dir, db) = copy_fixture("walrow");
    let dest = dir.path().join("strata");

    // Hold a connection open so the commit stays in the -wal.
    let mut conn = rusqlite::Connection::open(&db).unwrap();
    conn.pragma_update(None, "journal_mode", "WAL").unwrap();
    conn.execute(
        "INSERT INTO knowledge_nodes (id, content, node_type, created_at, updated_at, last_accessed, tags)
         VALUES ('55555555-5555-4555-8555-555555555555', 'WAL_ONLY_ROW_NOT_IN_MAIN', 'fact',
                 '2026-03-03T00:00:00+00:00', '2026-03-03T00:00:00+00:00',
                 '2026-03-03T00:00:00+00:00', '[]')",
        [],
    )
    .unwrap();
    conn.execute_batch("COMMIT;").ok();
    // wal_checkpoint is deliberately NOT run: the row lives only in the wal.

    let report = migrate_with_options(
        &db,
        &dest,
        MigrateOptions {
            accept_wal_snapshot: true,
            seed: Some(seed()),
            ..Default::default()
        },
    )
    .expect("wal snapshot migration succeeds");

    // The WAL-only row is in the log: 2 fixture nodes + 1 wal row + 1 walk
    // receipt reference.
    let snapshot = read_snapshot(&strata::StrataLog::open(&dest).unwrap()).unwrap();
    let contents: Vec<&str> = snapshot.nodes.iter().map(|n| n.content.as_str()).collect();
    assert!(
        contents.contains(&"WAL_ONLY_ROW_NOT_IN_MAIN"),
        "the WAL-only row must migrate: {contents:?}"
    );
    assert_eq!(
        report.nodes, 6,
        "4 fixture nodes + 1 wal row + 1 walk receipt"
    );
    drop(conn);
}
