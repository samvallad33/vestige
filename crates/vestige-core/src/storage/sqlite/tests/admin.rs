//! Tests for `sqlite/admin.rs`: durability profiles, checkpoints, integrity
//! and crash recovery.

use super::*;

// ===================== SQLite durability/recovery ===================

#[test]
fn durability_profile_parser_is_explicit_and_fail_closed() {
    assert_eq!(
        SqliteDurabilityProfile::parse("hardened").unwrap(),
        SqliteDurabilityProfile::Hardened
    );
    assert_eq!(
        SqliteDurabilityProfile::parse(" BALANCED ").unwrap(),
        SqliteDurabilityProfile::Balanced
    );
    let error = SqliteDurabilityProfile::parse("normal").unwrap_err();
    assert!(
        error.to_string().contains("expected hardened|balanced"),
        "unexpected error: {error}"
    );
}

#[test]
fn hardened_profile_is_verified_before_store_is_returned() {
    let dir = tempdir().unwrap();
    let store = Storage::new_with_durability(
        Some(dir.path().join("hardened.db")),
        SqliteDurabilityProfile::Hardened,
    )
    .unwrap();
    let status = store.durability_status();

    assert_eq!(status.profile, SqliteDurabilityProfile::Hardened);
    assert_eq!(status.writer.journal_mode, "wal");
    assert_eq!(status.writer.synchronous, 2);
    assert_eq!(status.writer.synchronous_label, "full");
    assert!(status.writer.fullfsync_enabled);
    assert!(status.writer.checkpoint_fullfsync_enabled);
    assert_eq!(status.writer.wal_autocheckpoint_pages, 1000);
    assert!(status.writer.foreign_keys_enabled);
    assert_eq!(status.writer.busy_timeout_ms, 5000);
    assert_eq!(status.reader.journal_mode, "wal");
    assert_eq!(status.reader.synchronous, 2);
    assert_eq!(status.before_migrations.quick_check, "ok");
    assert!(!status.before_migrations.synaptic_checks_applied);
    assert_eq!(status.after_migrations.quick_check, "ok");
    assert!(status.after_migrations.synaptic_checks_applied);
    assert_eq!(status.after_migrations.synaptic_consistency_violations, 0);
    assert_eq!(store.verify_integrity().unwrap().quick_check, "ok");
}

#[test]
fn balanced_profile_preserves_normal_sync_only_when_explicit() {
    let dir = tempdir().unwrap();
    let store = Storage::new_with_durability(
        Some(dir.path().join("balanced.db")),
        SqliteDurabilityProfile::Balanced,
    )
    .unwrap();
    let status = store.durability_status();

    assert_eq!(status.profile, SqliteDurabilityProfile::Balanced);
    assert_eq!(status.writer.journal_mode, "wal");
    assert_eq!(status.writer.synchronous, 1);
    assert_eq!(status.writer.synchronous_label, "normal");
    assert!(!status.writer.fullfsync_enabled);
    assert!(!status.writer.checkpoint_fullfsync_enabled);
    assert_eq!(status.reader.synchronous, 1);
}

#[test]
fn explicit_checkpoint_reports_sqlite_counters() {
    let dir = tempdir().unwrap();
    let store = Storage::new_with_durability(
        Some(dir.path().join("checkpoint.db")),
        SqliteDurabilityProfile::Hardened,
    )
    .unwrap();
    store
        .ingest(IngestInput {
            content: "checkpoint one acknowledged write".into(),
            node_type: "fact".into(),
            ..Default::default()
        })
        .unwrap();

    let passive = store.checkpoint_wal(WalCheckpointMode::Passive).unwrap();
    assert_eq!(passive.busy, 0);
    assert!(passive.log_frames >= passive.checkpointed_frames);

    let truncate = store.checkpoint_wal(WalCheckpointMode::Truncate).unwrap();
    assert_eq!(truncate.busy, 0);
}

#[test]
fn backup_to_captures_committed_wal_frames_in_a_consistent_snapshot() {
    let dir = tempdir().unwrap();
    let source_path = dir.path().join("source.db");
    let backup_path = dir.path().join("snapshot.db");
    let store =
        Storage::new_with_durability(Some(source_path.clone()), SqliteDurabilityProfile::Hardened)
            .unwrap();
    let node = store
        .ingest(IngestInput {
            content: "backup WAL snapshot sentinel".into(),
            node_type: "fact".into(),
            ..Default::default()
        })
        .unwrap();

    let wal_path = PathBuf::from(format!("{}-wal", source_path.display()));
    assert!(
        std::fs::metadata(&wal_path)
            .map(|metadata| metadata.len() > 0)
            .unwrap_or(false),
        "the source must retain committed WAL frames for this regression"
    );

    store.backup_to(&backup_path).unwrap();
    let backup = Connection::open(&backup_path).unwrap();
    let copied: String = backup
        .query_row(
            "SELECT content FROM knowledge_nodes WHERE id = ?1",
            params![node.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(copied, "backup WAL snapshot sentinel");
}

#[test]
fn startup_rejects_corrupt_database_before_migrations() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("corrupt.db");
    std::fs::write(&path, b"not a sqlite database").unwrap();

    let error = Storage::new_with_durability(Some(path), SqliteDurabilityProfile::Hardened)
        .err()
        .expect("corrupt database must not produce a store");
    assert!(
        error.to_string().contains("file is not a database")
            || error.to_string().contains("malformed"),
        "unexpected error: {error}"
    );
}

#[test]
fn startup_rejects_v21_event_without_receipt() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("inconsistent-v21.db");
    {
        let store =
            Storage::new_with_durability(Some(path.clone()), SqliteDurabilityProfile::Hardened)
                .unwrap();
        store.checkpoint_wal(WalCheckpointMode::Truncate).unwrap();
    }
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute(
            "INSERT INTO synaptic_events
                     (event_id, trigger_memory_id, event_type, occurred_at_ms,
                      window_from_ms, window_to_ms, strength, algorithm_version,
                      receipt_id, recorded_at)
                 VALUES ('broken-event', 'missing-trigger', 'test', 1, 1, 1,
                         1.0, 'test', 'missing-receipt', '1970-01-01T00:00:00Z')",
            [],
        )
        .unwrap();
    }

    let error = Storage::new_with_durability(Some(path), SqliteDurabilityProfile::Hardened)
        .err()
        .expect("inconsistent V21 rows must fail startup");
    assert!(
        error
            .to_string()
            .contains("pre-migration synaptic receipt consistency"),
        "unexpected error: {error}"
    );
}

#[test]
fn v22_pair_receipt_bindings_are_version_aware() {
    let dir = tempdir().unwrap();
    let store = Storage::new_with_durability(
        Some(dir.path().join("v22-pair-binding.db")),
        SqliteDurabilityProfile::Hardened,
    )
    .unwrap();
    let memory = store
        .ingest(IngestInput {
            content: "V22 pair binding fixture".into(),
            node_type: "fact".into(),
            ..Default::default()
        })
        .unwrap();
    let root_payload = serde_json::json!({
        "evidence": {
            "kind": "synaptic_capture",
            "predicate": {
                "schemaVersion": 2,
                "algorithmVersion": "vestige.synaptic_capture.v2",
                "receiptRole": "root",
                "trigger": { "eventId": "public-event" },
                "candidates": []
            }
        }
    })
    .to_string();
    let child_payload = serde_json::json!({
        "evidence": {
            "kind": "synaptic_capture",
            "predicate": {
                "schemaVersion": 2,
                "algorithmVersion": "vestige.synaptic_capture.v2",
                "receiptRole": "pair",
                "parentReceiptId": "root-receipt",
                "evaluationDirection": "forward",
                "trigger": { "eventId": "public-event" },
                "candidates": [{ "evidenceSlot": "candidate_1" }]
            }
        }
    })
    .to_string();
    {
        let writer = store.writer.lock().unwrap();
        writer
            .execute(
                "INSERT INTO memory_receipts(receipt_id, payload, created_at)
                     VALUES ('root-receipt', ?1, '1970-01-01T00:00:00Z')",
                params![root_payload],
            )
            .unwrap();
        writer
            .execute(
                "INSERT INTO memory_receipts(receipt_id, payload, created_at)
                     VALUES ('child-receipt', ?1, '1970-01-01T00:00:00Z')",
                params![child_payload],
            )
            .unwrap();
        writer
            .execute(
                "INSERT INTO synaptic_events(
                         event_id, trigger_memory_id, event_type, occurred_at_ms,
                         window_from_ms, window_to_ms, strength, algorithm_version,
                         receipt_id, recorded_at, public_event_id, event_state
                     ) VALUES (
                         'private-event', ?1, 'test', 2, 1, 2, 1.0,
                         'vestige.synaptic_capture.v2', 'root-receipt',
                         '1970-01-01T00:00:00Z', 'public-event', 'closed'
                     )",
                params![memory.id],
            )
            .unwrap();
        writer
            .execute(
                "INSERT INTO synaptic_tags(
                         tag_id, memory_id, created_at_ms, initial_strength,
                         algorithm_version, state, recorded_at
                     ) VALUES (
                         'tag-1', ?1, 1, 1.0, 'vestige.synaptic_capture.v2',
                         'active', '1970-01-01T00:00:00Z'
                     )",
                params![memory.id],
            )
            .unwrap();
        writer
            .execute(
                "INSERT INTO synaptic_capture_items(
                         event_id, tag_id, memory_id, evidence_slot, receipt_id,
                         encoded_at_ms, temporal_distance_hours, capture_probability,
                         tag_strength_at_evaluation, capture_score, disposition,
                         recorded_at, evaluation_direction, algorithm_version
                     ) VALUES (
                         'private-event', 'tag-1', ?1, 'candidate_1', 'child-receipt',
                         1, 0.0, 1.0, 1.0, 1.0, 'below_threshold',
                         '1970-01-01T00:00:00Z', 'forward',
                         'vestige.synaptic_capture.v2'
                     )",
                params![memory.id],
            )
            .unwrap();
    }

    assert_eq!(
        store
            .verify_integrity()
            .unwrap()
            .synaptic_consistency_violations,
        0
    );

    let invalid_child_payload = serde_json::json!({
        "evidence": {
            "kind": "synaptic_capture",
            "predicate": {
                "schemaVersion": 2,
                "algorithmVersion": "vestige.synaptic_capture.v2",
                "receiptRole": "pair",
                "parentReceiptId": "wrong-root",
                "evaluationDirection": "forward",
                "trigger": { "eventId": "public-event" },
                "candidates": [{ "evidenceSlot": "candidate_1" }]
            }
        }
    })
    .to_string();
    store
        .writer
        .lock()
        .unwrap()
        .execute(
            "UPDATE memory_receipts SET payload = ?1 WHERE receipt_id = 'child-receipt'",
            params![invalid_child_payload],
        )
        .unwrap();
    let error = store.verify_integrity().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("synaptic receipt consistency checks found 1"),
        "unexpected error: {error}"
    );

    let legacy_child_payload = serde_json::json!({
        "evidence": {
            "kind": "synaptic_capture",
            "predicate": {
                "schemaVersion": 1,
                "algorithmVersion": "vestige.synaptic_capture.v1",
                "trigger": { "eventId": "public-event" },
                "candidates": [{ "evidenceSlot": "candidate_1" }]
            }
        }
    })
    .to_string();
    store
        .writer
        .lock()
        .unwrap()
        .execute(
            "UPDATE memory_receipts SET payload = ?1 WHERE receipt_id = 'child-receipt'",
            params![legacy_child_payload],
        )
        .unwrap();
    let error = store.verify_integrity().unwrap_err();
    assert!(
        error.to_string().contains("synaptic receipt consistency"),
        "a schema-v1 receipt must not validate a V22 forward item: {error}"
    );

    // SQL `NULL IS NOT NULL` is false, so an explicit non-null/type guard
    // is required or a missing event id on both sides becomes fail-open.
    let missing_event_payload = serde_json::json!({
        "evidence": {
            "kind": "synaptic_capture",
            "predicate": {
                "schemaVersion": 2,
                "algorithmVersion": "vestige.synaptic_capture.v2",
                "receiptRole": "root",
                "trigger": {},
                "candidates": []
            }
        }
    })
    .to_string();
    {
        let writer = store.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE synaptic_events SET public_event_id = NULL
                     WHERE event_id = 'private-event'",
                [],
            )
            .unwrap();
        writer
            .execute(
                "UPDATE memory_receipts SET payload = ?1
                     WHERE receipt_id = 'root-receipt'",
                params![missing_event_payload],
            )
            .unwrap();
    }
    let error = store.verify_integrity().unwrap_err();
    assert!(
        error.to_string().contains("synaptic receipt consistency"),
        "missing V22 event ids must fail closed: {error}"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn hardened_profile_rejects_missing_fullfsync_readback_on_macos() {
    let mut pragmas = SqliteConnectionPragmas {
        journal_mode: "wal".into(),
        synchronous: 2,
        synchronous_label: "full".into(),
        fullfsync_enabled: true,
        fullfsync_meaningful_on_this_platform: true,
        checkpoint_fullfsync_enabled: true,
        wal_autocheckpoint_pages: 1000,
        foreign_keys_enabled: true,
        busy_timeout_ms: 5000,
    };
    pragmas.fullfsync_enabled = false;
    assert!(
        Storage::verify_effective_pragmas(SqliteDurabilityProfile::Hardened, "test", &pragmas)
            .is_err()
    );
    pragmas.fullfsync_enabled = true;
    pragmas.checkpoint_fullfsync_enabled = false;
    assert!(
        Storage::verify_effective_pragmas(SqliteDurabilityProfile::Hardened, "test", &pragmas)
            .is_err()
    );
}

#[test]
fn hardened_writer_refuses_read_only_non_wal_database() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("readonly-delete.db");
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "PRAGMA journal_mode = DELETE;
                 CREATE TABLE seed(id INTEGER PRIMARY KEY);",
        )
        .unwrap();
    }
    let conn =
        Connection::open_with_flags(&path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();

    let error =
        Storage::configure_connection(&conn, SqliteDurabilityProfile::Hardened, true).unwrap_err();
    assert!(
        error.to_string().contains("readonly")
            || error.to_string().contains("read-only")
            || error.to_string().contains("attempt to write"),
        "unexpected error: {error}"
    );
}

#[cfg(unix)]
const SQLITE_CRASH_CHILD_SCENARIO: &str = "VESTIGE_SQLITE_CRASH_CHILD_SCENARIO";

#[cfg(unix)]
const SQLITE_CRASH_CHILD_PATH: &str = "VESTIGE_SQLITE_CRASH_CHILD_PATH";

#[cfg(unix)]
const SQLITE_CRASH_READY: &str = "VESTIGE_SQLITE_CRASH_READY";

/// Subprocess-only entry point for the process-crash durability harness.
#[cfg(unix)]
#[test]
fn sqlite_crash_child() {
    let Ok(scenario) = std::env::var(SQLITE_CRASH_CHILD_SCENARIO) else {
        return;
    };
    let path = PathBuf::from(
        std::env::var_os(SQLITE_CRASH_CHILD_PATH).expect("crash child requires a database path"),
    );
    let store =
        Storage::new_with_durability(Some(path), SqliteDurabilityProfile::Hardened).unwrap();
    let mut writer = store.writer.lock().unwrap();
    let tx = writer
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .unwrap();
    tx.execute(
        "INSERT INTO durability_probe_transactions(id, value)
             VALUES ('ack-boundary', 'parent')",
        [],
    )
    .unwrap();
    tx.execute(
        "INSERT INTO durability_probe_items(transaction_id, item_index, value)
             VALUES ('ack-boundary', 1, 'first'), ('ack-boundary', 2, 'second')",
        [],
    )
    .unwrap();

    if scenario == "before_commit" {
        println!("{SQLITE_CRASH_READY}=before_commit");
        std::io::stdout().flush().unwrap();
        loop {
            std::thread::park_timeout(std::time::Duration::from_secs(60));
        }
    }

    assert_eq!(scenario, "after_commit");
    tx.commit().unwrap();
    drop(writer);
    println!("{SQLITE_CRASH_READY}=after_commit");
    std::io::stdout().flush().unwrap();
    loop {
        std::thread::park_timeout(std::time::Duration::from_secs(60));
    }
}

#[cfg(unix)]
fn spawn_and_kill_at_commit_boundary(path: &Path, scenario: &str) {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("storage::sqlite::tests::admin::sqlite_crash_child")
        .arg("--nocapture")
        .env(SQLITE_CRASH_CHILD_SCENARIO, scenario)
        .env(SQLITE_CRASH_CHILD_PATH, path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let (ready_tx, ready_rx) = mpsc::channel();
    std::thread::spawn(move || {
        use std::io::{BufRead, BufReader};
        for line in BufReader::new(stdout)
            .lines()
            .map_while(std::result::Result::ok)
        {
            if line.contains(SQLITE_CRASH_READY) {
                let _ = ready_tx.send(line);
                return;
            }
        }
    });

    let marker = ready_rx
        .recv_timeout(std::time::Duration::from_secs(20))
        .unwrap_or_else(|error| {
            let _ = child.kill();
            let stderr = child
                .stderr
                .take()
                .map(|mut stderr| {
                    let mut text = String::new();
                    let _ = std::io::Read::read_to_string(&mut stderr, &mut text);
                    text
                })
                .unwrap_or_default();
            panic!("crash child did not reach {scenario}: {error}; stderr={stderr}")
        });
    assert!(marker.contains(scenario), "unexpected marker: {marker}");
    child.kill().unwrap();
    let status = child.wait().unwrap();
    assert!(
        !status.success(),
        "crash child should be killed, not exit cleanly"
    );
}

#[cfg(unix)]
fn prepare_crash_probe(path: &Path) {
    let store =
        Storage::new_with_durability(Some(path.to_path_buf()), SqliteDurabilityProfile::Hardened)
            .unwrap();
    store
        .writer
        .lock()
        .unwrap()
        .execute_batch(
            "CREATE TABLE durability_probe_transactions(
                     id TEXT PRIMARY KEY,
                     value TEXT NOT NULL
                 ) STRICT;
                 CREATE TABLE durability_probe_items(
                     transaction_id TEXT NOT NULL,
                     item_index INTEGER NOT NULL,
                     value TEXT NOT NULL,
                     PRIMARY KEY(transaction_id, item_index),
                     FOREIGN KEY(transaction_id)
                         REFERENCES durability_probe_transactions(id)
                         ON DELETE CASCADE
                 ) STRICT;",
        )
        .unwrap();
    store.checkpoint_wal(WalCheckpointMode::Truncate).unwrap();
}

#[cfg(unix)]
fn crash_probe_counts(path: &Path) -> (i64, i64) {
    let store =
        Storage::new_with_durability(Some(path.to_path_buf()), SqliteDurabilityProfile::Hardened)
            .unwrap();
    assert_eq!(store.verify_integrity().unwrap().quick_check, "ok");
    let reader = store.reader.lock().unwrap();
    let transactions = reader
        .query_row(
            "SELECT COUNT(*) FROM durability_probe_transactions",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let items = reader
        .query_row("SELECT COUNT(*) FROM durability_probe_items", [], |row| {
            row.get(0)
        })
        .unwrap();
    (transactions, items)
}

#[cfg(unix)]
#[test]
fn sigkill_before_and_after_commit_respects_atomic_ack_boundary() {
    let dir = tempdir().unwrap();

    let before_path = dir.path().join("before-commit.db");
    prepare_crash_probe(&before_path);
    spawn_and_kill_at_commit_boundary(&before_path, "before_commit");
    assert_eq!(crash_probe_counts(&before_path), (0, 0));

    let after_path = dir.path().join("after-commit.db");
    prepare_crash_probe(&after_path);
    spawn_and_kill_at_commit_boundary(&after_path, "after_commit");
    assert_eq!(crash_probe_counts(&after_path), (1, 2));
}
