use super::*;
#[cfg(unix)]
use std::process::{Command, Stdio};
#[cfg(unix)]
use std::sync::mpsc;
use tempfile::tempdir;
// The public struct was renamed from Storage to SqliteMemoryStore; this
// alias keeps all existing tests compiling without modification.
use SqliteMemoryStore as Storage;

fn create_test_storage() -> Storage {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    Storage::new(Some(db_path)).unwrap()
}

fn create_test_storage_at(dir: &tempfile::TempDir, name: &str) -> Storage {
    Storage::new(Some(dir.path().join(name))).unwrap()
}

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
        .arg("storage::sqlite::tests::sqlite_crash_child")
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

// ===================== Connector sync (#57) =========================

/// Build an `IngestInput` carrying a source envelope for a GitHub-ish issue.
fn source_input(id: &str, content: &str, hash: &str) -> IngestInput {
    IngestInput {
        content: content.to_string(),
        node_type: "fact".to_string(),
        source_envelope: Some(crate::memory::SourceEnvelope {
            source_system: Some("github".to_string()),
            source_id: Some(id.to_string()),
            source_url: Some(format!("https://github.com/o/r/issues/{id}")),
            content_hash: Some(hash.to_string()),
            source_project: Some("o/r".to_string()),
            source_type: Some("issue".to_string()),
            source_author: Some("octocat".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn node_count(store: &Storage) -> i64 {
    // Count rows for our test source so embeddings/other tests don't bleed in.
    let reader = store.reader.lock().unwrap();
    reader
        .query_row(
            "SELECT COUNT(*) FROM knowledge_nodes WHERE source_system = 'github'",
            [],
            |r| r.get(0),
        )
        .unwrap()
}

// ===================== Secret-ingest policy (#154) ==================

#[test]
fn ingest_rejects_probable_github_secret_without_persisting_or_echoing_it() {
    let store = create_test_storage();
    let secret = format!("ghp_{}", "A".repeat(36));

    let err = store
        .ingest(IngestInput {
            content: format!("The temporary token is {secret}"),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap_err();

    assert!(matches!(err, StorageError::SecretDetected { .. }));
    assert!(
        !err.to_string().contains(&secret),
        "rejection must not echo the credential"
    );
    assert_eq!(
        store.get_stats().unwrap().total_nodes,
        0,
        "rejection must happen before creating a node"
    );
}

#[test]
fn update_node_content_rejects_probable_github_secret_without_mutating_node() {
    let store = create_test_storage();
    let node = store
        .ingest(IngestInput {
            content: "safe original content".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let secret = format!("ghp_{}", "A".repeat(36));

    let err = store
        .update_node_content(&node.id, &format!("replacement includes {secret}"))
        .unwrap_err();

    assert!(matches!(err, StorageError::SecretDetected { .. }));
    assert!(
        !err.to_string().contains(&secret),
        "rejection must not echo the credential"
    );
    assert_eq!(
        store.get_node(&node.id).unwrap().unwrap().content,
        "safe original content",
        "rejection must leave existing content intact"
    );
}

#[test]
fn connector_upsert_rejects_probable_github_secret_without_mutating_existing_record() {
    let store = create_test_storage();
    let created = store
        .upsert_by_source(source_input(
            "secret-policy",
            "safe connector body",
            "safe-hash",
        ))
        .unwrap();
    let secret = format!("ghp_{}", "A".repeat(36));

    let err = store
        .upsert_by_source(source_input(
            "secret-policy",
            &format!("updated connector body includes {secret}"),
            "secret-hash",
        ))
        .unwrap_err();

    assert!(matches!(err, StorageError::SecretDetected { .. }));
    assert!(
        !err.to_string().contains(&secret),
        "rejection must not echo the credential"
    );
    let stored = store.get_node(&created.node_id).unwrap().unwrap();
    assert_eq!(stored.content, "safe connector body");
    assert_eq!(
        stored
            .source_envelope
            .as_ref()
            .and_then(|envelope| envelope.content_hash.as_deref()),
        Some("safe-hash"),
        "preflight rejection must happen before connector metadata changes"
    );
}

#[test]
fn portable_import_rejects_secret_archive_atomically_before_safe_rows_import() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    source
        .ingest(IngestInput {
            content: "safe portable memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let secret = format!("ghp_{}", "A".repeat(36));
    source
        .ingest_with_secret_policy(
            IngestInput {
                content: format!("intentionally archived credential {secret}"),
                node_type: "fact".to_string(),
                ..Default::default()
            },
            SecretPolicy::AllowExplicitly,
        )
        .unwrap();
    let archive = source.export_portable_archive().unwrap();

    let target = create_test_storage_at(&target_dir, "target.db");
    let err = target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap_err();

    assert!(matches!(err, StorageError::SecretDetected { .. }));
    assert!(
        !err.to_string().contains(&secret),
        "rejection must not echo the credential"
    );
    assert_eq!(
        target.get_stats().unwrap().total_nodes,
        0,
        "archive preflight must prevent a partial import of safe sibling rows"
    );
}

#[test]
fn upsert_by_source_is_idempotent_across_reruns() {
    let store = create_test_storage();

    // First sync: a brand-new record → Created.
    let r1 = store
        .upsert_by_source(source_input("1", "Bug: crash on startup", "hash-a"))
        .unwrap();
    assert_eq!(r1.outcome, SourceUpsertOutcome::Created);
    assert_eq!(node_count(&store), 1);

    // Re-sync the SAME record with the SAME hash twice → Unchanged, no dupes.
    for _ in 0..2 {
        let r = store
            .upsert_by_source(source_input("1", "Bug: crash on startup", "hash-a"))
            .unwrap();
        assert_eq!(r.outcome, SourceUpsertOutcome::Unchanged);
        assert_eq!(r.node_id, r1.node_id, "must reuse the same memory id");
    }
    assert_eq!(
        node_count(&store),
        1,
        "idempotent: still exactly one memory"
    );
}

#[test]
fn upsert_by_source_updates_in_place_when_hash_changes() {
    let store = create_test_storage();
    let created = store
        .upsert_by_source(source_input("7", "old body", "hash-old"))
        .unwrap();

    // Upstream edit: content + hash change → Updated, same id, new content.
    let updated = store
        .upsert_by_source(source_input("7", "new edited body", "hash-new"))
        .unwrap();
    assert_eq!(updated.outcome, SourceUpsertOutcome::Updated);
    assert_eq!(updated.node_id, created.node_id);
    assert_eq!(node_count(&store), 1, "update must not duplicate");

    let node = store.get_node(&created.node_id).unwrap().unwrap();
    assert_eq!(node.content, "new edited body");
    let env = node.source_envelope.expect("envelope persisted");
    assert_eq!(env.content_hash.as_deref(), Some("hash-new"));
    assert_eq!(env.source_id.as_deref(), Some("7"));
}

#[test]
fn upsert_by_source_without_key_falls_back_to_create() {
    let store = create_test_storage();
    // Envelope present but missing source_id → not keyed → plain create.
    let input = IngestInput {
        content: "loose note".to_string(),
        node_type: "fact".to_string(),
        source_envelope: Some(crate::memory::SourceEnvelope {
            source_url: Some("https://example.com/x".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let r = store.upsert_by_source(input).unwrap();
    assert_eq!(r.outcome, SourceUpsertOutcome::Created);
}

#[test]
fn connector_cursor_round_trips() {
    let store = create_test_storage();
    // Unknown scope → zeroed cursor.
    let empty = store.get_connector_cursor("github", "o/r").unwrap();
    assert!(empty.cursor_updated_at.is_none());
    assert_eq!(empty.records_seen, 0);

    let ts = Utc::now();
    let cursor = ConnectorCursor {
        source_system: "github".to_string(),
        scope: "o/r".to_string(),
        cursor_updated_at: Some(ts),
        last_synced_at: Some(ts),
        last_full_reconcile_at: None,
        records_seen: 42,
    };
    store.save_connector_cursor(&cursor).unwrap();

    let back = store.get_connector_cursor("github", "o/r").unwrap();
    assert_eq!(back.records_seen, 42);
    assert_eq!(
        back.cursor_updated_at.map(|d| d.to_rfc3339()),
        Some(ts.to_rfc3339())
    );

    // Upsert semantics: saving again replaces, never duplicates.
    let mut c2 = cursor.clone();
    c2.records_seen = 99;
    store.save_connector_cursor(&c2).unwrap();
    assert_eq!(
        store
            .get_connector_cursor("github", "o/r")
            .unwrap()
            .records_seen,
        99
    );
}

#[test]
fn reconcile_tombstones_records_absent_from_live_set() {
    let store = create_test_storage();
    // Three synced issues in scope o/r.
    for id in ["1", "2", "3"] {
        store
            .upsert_by_source(source_input(id, &format!("issue {id}"), &format!("h{id}")))
            .unwrap();
    }

    // Reconcile: only 1 and 3 are still visible upstream → 2 is tombstoned.
    let report = store
        .reconcile_source_tombstones("github", "o/r", &["1".to_string(), "3".to_string()])
        .unwrap();
    assert_eq!(report.considered, 3);
    assert_eq!(report.tombstoned.len(), 1, "exactly issue 2 tombstoned");

    // Issue 2's memory is invalidated (valid_until set) but NOT purged —
    // content retained for audit, just no longer currently-valid.
    let two = {
        let reader = store.reader.lock().unwrap();
        reader
            .query_row(
                "SELECT id, valid_until FROM knowledge_nodes WHERE source_id = '2'",
                [],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?)),
            )
            .unwrap()
    };
    assert!(
        two.1.is_some(),
        "tombstoned record must have valid_until set"
    );
    let node = store.get_node(&two.0).unwrap().unwrap();
    assert!(
        !node.is_currently_valid(),
        "tombstoned node is not valid now"
    );
    assert_eq!(node.content, "issue 2", "content retained for audit");

    // A reappearing record un-tombstones on next upsert (clears valid_until).
    store
        .upsert_by_source(source_input("2", "issue 2", "h2"))
        .unwrap();
    let revived = store.get_node(&two.0).unwrap().unwrap();
    assert!(
        revived.is_currently_valid(),
        "re-synced record is valid again"
    );
}

#[test]
fn upsert_clears_superseded_by_when_record_reappears() {
    // Regression: un-tombstoning must clear BOTH bitemporal markers. A
    // connector node that was superseded/merged (valid_until + superseded_by
    // both set) and then re-observed upstream must come back fully clean,
    // otherwise it is currently-valid yet still flagged superseded and is
    // permanently excluded from merge candidacy.
    let store = create_test_storage();
    let created = store
        .upsert_by_source(source_input("9", "body v1", "h9a"))
        .unwrap();

    // Simulate the node having been superseded (as merge/supersede would).
    {
        let writer = store.writer.lock().unwrap();
        writer
                .execute(
                    "UPDATE knowledge_nodes SET valid_until = ?1, superseded_by = 'survivor-id' WHERE id = ?2",
                    params![Utc::now().to_rfc3339(), created.node_id],
                )
                .unwrap();
    }
    assert!(
        store
            .superseded_node_ids()
            .unwrap()
            .contains(&created.node_id),
        "precondition: node is superseded"
    );

    // Re-sync with a content change → Updated branch must clear both markers.
    let res = store
        .upsert_by_source(source_input("9", "body v2 edited", "h9b"))
        .unwrap();
    assert_eq!(res.outcome, SourceUpsertOutcome::Updated);
    assert!(
        !store
            .superseded_node_ids()
            .unwrap()
            .contains(&created.node_id),
        "superseded_by must be cleared on re-sync (no bitemporal zombie)"
    );
    let node = store.get_node(&created.node_id).unwrap().unwrap();
    assert!(node.is_currently_valid());

    // Also exercise the Unchanged branch: supersede again, re-sync same hash.
    {
        let writer = store.writer.lock().unwrap();
        writer
                .execute(
                    "UPDATE knowledge_nodes SET valid_until = ?1, superseded_by = 'survivor-id' WHERE id = ?2",
                    params![Utc::now().to_rfc3339(), created.node_id],
                )
                .unwrap();
    }
    let res2 = store
        .upsert_by_source(source_input("9", "body v2 edited", "h9b"))
        .unwrap();
    assert_eq!(res2.outcome, SourceUpsertOutcome::Unchanged);
    assert!(
        !store
            .superseded_node_ids()
            .unwrap()
            .contains(&created.node_id),
        "Unchanged branch must also clear superseded_by"
    );
}

/// Build a `source_input` whose envelope carries an explicit project.
fn source_input_in_project(
    id: &str,
    content: &str,
    hash: &str,
    project: Option<&str>,
) -> IngestInput {
    let mut input = source_input(id, content, hash);
    input.source_envelope.as_mut().unwrap().source_project = project.map(str::to_string);
    input
}

#[test]
fn upsert_by_source_scopes_key_by_project() {
    // V19: two sources of the same system reuse bare per-project ids, so
    // the same (system, id) under DIFFERENT projects must yield two
    // distinct nodes, and re-syncing each must hit its own row (Unchanged).
    let store = create_test_storage();

    let a = store
        .upsert_by_source(source_input_in_project(
            "5",
            "repoA issue 5",
            "hA",
            Some("octocat/repoA"),
        ))
        .unwrap();
    let b = store
        .upsert_by_source(source_input_in_project(
            "5",
            "repoB issue 5",
            "hB",
            Some("octocat/repoB"),
        ))
        .unwrap();
    assert_eq!(a.outcome, SourceUpsertOutcome::Created);
    assert_eq!(b.outcome, SourceUpsertOutcome::Created);
    assert_ne!(a.node_id, b.node_id, "projects must not share a row");
    assert_eq!(node_count(&store), 2);

    // Re-sync both records with unchanged hashes → each resolves to ITS row.
    let ra = store
        .upsert_by_source(source_input_in_project(
            "5",
            "repoA issue 5",
            "hA",
            Some("octocat/repoA"),
        ))
        .unwrap();
    assert_eq!(ra.outcome, SourceUpsertOutcome::Unchanged);
    assert_eq!(ra.node_id, a.node_id);
    let rb = store
        .upsert_by_source(source_input_in_project(
            "5",
            "repoB issue 5",
            "hB",
            Some("octocat/repoB"),
        ))
        .unwrap();
    assert_eq!(rb.outcome, SourceUpsertOutcome::Unchanged);
    assert_eq!(rb.node_id, b.node_id);
    assert_eq!(node_count(&store), 2, "resync must not duplicate");
}

#[test]
fn upsert_by_source_matches_legacy_null_project_row_with_empty_string() {
    // Regression: the V19 unique index buckets NULL and '' together via
    // COALESCE(source_project, ''), but the lookup used `source_project IS
    // ?3`, which treats NULL and '' as distinct. A legacy NULL-project row
    // plus an ''-project envelope for the same (system, id) made the lookup
    // miss, and the fall-through INSERT then hit the UNIQUE constraint.
    let store = create_test_storage();
    let created = store
        .upsert_by_source(source_input("41", "legacy body", "h-legacy"))
        .unwrap();
    assert_eq!(created.outcome, SourceUpsertOutcome::Created);

    // Simulate a pre-V19 legacy row: source_project stored as NULL.
    {
        let writer = store.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET source_project = NULL WHERE id = ?1",
                params![created.node_id],
            )
            .unwrap();
    }

    // New connector run sends Some("") for the same (system, id). Must
    // UPDATE the legacy row in place — not error on the unique index.
    let res = store
        .upsert_by_source(source_input_in_project(
            "41",
            "legacy body edited",
            "h-new",
            Some(""),
        ))
        .expect("''-project envelope must resolve the NULL-project row, not UNIQUE-fail");
    assert_eq!(res.outcome, SourceUpsertOutcome::Updated);
    assert_eq!(res.node_id, created.node_id, "must reuse the legacy row");
    assert_eq!(node_count(&store), 1, "no duplicate row in the NULL bucket");

    // And the Unchanged path resolves through the same bucket too.
    let res2 = store
        .upsert_by_source(source_input_in_project(
            "41",
            "legacy body edited",
            "h-new",
            Some(""),
        ))
        .unwrap();
    assert_eq!(res2.outcome, SourceUpsertOutcome::Unchanged);
    assert_eq!(res2.node_id, created.node_id);
}

#[test]
fn test_storage_creation() {
    let storage = create_test_storage();
    let stats = storage.get_stats().unwrap();
    assert_eq!(stats.total_nodes, 0);
}

// =========================================================================
// Post-retrieval failure feedback (Heinbockel 2025)
// =========================================================================

fn set_retrieval_strength(storage: &SqliteMemoryStore, id: &str, value: f64) {
    storage
        .writer
        .lock()
        .unwrap()
        .execute(
            "UPDATE knowledge_nodes SET retrieval_strength = ?1 WHERE id = ?2",
            params![value, id],
        )
        .unwrap();
}

fn retrieval_strength(storage: &SqliteMemoryStore, id: &str) -> f64 {
    storage
        .reader
        .lock()
        .unwrap()
        .query_row(
            "SELECT retrieval_strength FROM knowledge_nodes WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )
        .unwrap()
}

fn save_test_receipt(storage: &SqliteMemoryStore, retrieved: Vec<String>) -> String {
    let trust: Vec<f64> = retrieved.iter().map(|_| 0.9).collect();
    let receipt = crate::trace::Receipt::build(
        Utc::now(),
        "test",
        retrieved,
        Vec::new(),
        Vec::new(),
        &trust,
        Vec::new(),
    );
    let id = receipt.receipt_id.clone();
    storage
        .save_receipt(&receipt, None, Some("recall"), Some("what broke"))
        .unwrap();
    id
}

/// A failure right after a retrieval lowers the retrieved memories'
/// accessibility by rank (best-first is the strongest reactivation), once
/// per (failure, memory), and the ledger can undo it exactly.
#[test]
fn failure_feedback_demotes_recently_retrieved_memories_by_rank() {
    let storage = create_test_storage();
    let first = ingest_tagged_in_scope(
        &storage,
        "user",
        "redis url points at the old cluster",
        &["ops"],
    );
    let second = ingest_tagged_in_scope(
        &storage,
        "user",
        "worker pool reads a cached secrets file",
        &["ops"],
    );
    set_retrieval_strength(&storage, &first.id, 0.90);
    set_retrieval_strength(&storage, &second.id, 0.90);
    save_test_receipt(&storage, vec![first.id.clone(), second.id.clone()]);

    let failure = ingest_tagged_in_scope(
        &storage,
        "user",
        "Payments API crashed with 500s: queue backed up for 36 hours",
        &["incident"],
    );

    let report = storage
        .apply_failure_feedback(&failure.id, Duration::minutes(30))
        .unwrap();
    assert_eq!(report.receipts_considered, 1);
    assert_eq!(report.memories_demoted, 2);
    let after_first = retrieval_strength(&storage, &first.id);
    let after_second = retrieval_strength(&storage, &second.id);
    assert!(
        (after_first - 0.80).abs() < 1e-6,
        "rank 0 loses the full penalty: {after_first}"
    );
    assert!(
        (after_second - 0.85).abs() < 1e-6,
        "rank 1 loses half: {after_second}"
    );
    assert!(
        retrieval_strength(&storage, &failure.id) > 0.0,
        "the failure itself is never demoted"
    );

    // Idempotent per (failure, memory).
    let again = storage
        .apply_failure_feedback(&failure.id, Duration::minutes(30))
        .unwrap();
    assert_eq!(again.memories_demoted, 0);
    assert!((retrieval_strength(&storage, &first.id) - 0.80).abs() < 1e-6);

    // Reversible, exactly.
    assert_eq!(storage.revert_failure_feedback(&failure.id).unwrap(), 2);
    assert!((retrieval_strength(&storage, &first.id) - 0.90).abs() < 1e-6);
    assert!((retrieval_strength(&storage, &second.id) - 0.90).abs() < 1e-6);
    assert_eq!(
        storage.revert_failure_feedback(&failure.id).unwrap(),
        0,
        "a second revert finds nothing left to undo"
    );
}

/// Only memories in the failure's scope are touched, and a receipt outside
/// the window is ignored.
#[test]
fn failure_feedback_respects_scope_and_window() {
    let storage = create_test_storage();
    let elsewhere =
        ingest_tagged_in_scope(&storage, "other-project", "unrelated project note", &["x"]);
    let here = ingest_tagged_in_scope(&storage, "user", "same-scope note", &["x"]);
    set_retrieval_strength(&storage, &elsewhere.id, 0.90);
    set_retrieval_strength(&storage, &here.id, 0.90);
    save_test_receipt(&storage, vec![elsewhere.id.clone(), here.id.clone()]);

    let failure = ingest_tagged_in_scope(
        &storage,
        "user",
        "Build failed: linker error",
        &["incident"],
    );
    let report = storage
        .apply_failure_feedback(&failure.id, Duration::minutes(30))
        .unwrap();
    assert_eq!(
        report.memories_demoted, 1,
        "only the same-scope memory is demoted"
    );
    assert!((retrieval_strength(&storage, &elsewhere.id) - 0.90).abs() < 1e-6);
    assert!(retrieval_strength(&storage, &here.id) < 0.90);

    // A zero-length window sees no receipts.
    let none = storage
        .apply_failure_feedback(&failure.id, Duration::zero())
        .unwrap();
    assert_eq!(none.receipts_considered, 0);
}

fn ingest_tagged_in_scope(
    storage: &SqliteMemoryStore,
    scope: &str,
    content: &str,
    tags: &[&str],
) -> KnowledgeNode {
    storage
        .ingest_in_scope(
            IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                tags: tags.iter().map(|tag| tag.to_string()).collect(),
                ..Default::default()
            },
            scope,
        )
        .unwrap()
}

fn preview_token(preview: &serde_json::Value) -> &str {
    preview["previewToken"].as_str().unwrap()
}

#[test]
fn tag_rename_is_previewed_scoped_exact_atomic_audited_and_reversible() {
    let storage = create_test_storage();
    let user = ingest_tagged_in_scope(
        &storage,
        "user",
        "scoped tag rename fixture",
        &[
            "keep",
            "legacytag156",
            "canonicaltag156",
            "canonicaltag156",
            "tail",
        ],
    );
    let prefix = ingest_tagged_in_scope(
        &storage,
        "user",
        "exact matching fixture",
        &["legacytag156-extra"],
    );
    let project = ingest_tagged_in_scope(
        &storage,
        "project-a",
        "cross scope fixture",
        &["legacytag156"],
    );
    let sources = vec!["legacytag156".to_string()];

    let preview = storage
        .preview_tag_mutation(&sources, "canonicaltag156", Some(" user "))
        .unwrap();
    assert_eq!(preview["affectedMemoryCount"], 1);
    assert_eq!(
        storage.get_node(&user.id).unwrap().unwrap().tags,
        vec![
            "keep",
            "legacytag156",
            "canonicaltag156",
            "canonicaltag156",
            "tail"
        ],
        "preview must not mutate"
    );

    let operation = storage
        .apply_tag_mutation(
            &sources,
            "canonicaltag156",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "standardize the issue 156 fixture tag",
        )
        .unwrap();
    assert_eq!(operation.op_type, "tag_rename");
    assert_eq!(operation.affected_ids, vec![user.id.clone()]);
    assert_eq!(
        storage.get_node(&user.id).unwrap().unwrap().tags,
        vec!["keep", "canonicaltag156", "tail"],
        "the union of source and existing target tags is one target at the first affected position"
    );
    assert_eq!(
        storage.get_node(&prefix.id).unwrap().unwrap().tags,
        vec!["legacytag156-extra"],
        "prefix tags must not match"
    );
    assert_eq!(
        storage.get_node(&project.id).unwrap().unwrap().tags,
        vec!["legacytag156"],
        "default scoped maintenance must not cross project boundaries"
    );
    assert!(
        storage
            .keyword_search("canonicaltag156", 10, 0.0)
            .unwrap()
            .iter()
            .any(|node| node.id == user.id),
        "the existing knowledge_nodes update trigger must keep FTS tag search consistent"
    );
    let logged = storage.get_merge_operation(&operation.id).unwrap().unwrap();
    assert_eq!(
        logged.reason.as_deref(),
        Some("standardize the issue 156 fixture tag")
    );

    storage.undo_tag_mutation(&operation.id).unwrap();
    assert_eq!(
        storage.get_node(&user.id).unwrap().unwrap().tags,
        vec![
            "keep",
            "legacytag156",
            "canonicaltag156",
            "canonicaltag156",
            "tail"
        ],
        "undo restores the exact pre-operation array"
    );
}

#[test]
fn tag_rename_all_scopes_is_explicit_and_updates_every_matching_row() {
    let storage = create_test_storage();
    let user = ingest_tagged_in_scope(&storage, "user", "user shared tag", &["shared"]);
    let project = ingest_tagged_in_scope(&storage, "project-a", "project shared tag", &["shared"]);
    let sources = vec!["shared".to_string()];

    let scoped = storage
        .preview_tag_mutation(&sources, "canonical", Some("user"))
        .unwrap();
    assert_eq!(scoped["affectedMemoryCount"], 1);
    assert_eq!(scoped["allScopes"], false);

    let preview = storage
        .preview_tag_mutation(&sources, "canonical", None)
        .unwrap();
    assert_eq!(preview["allScopes"], true);
    assert_eq!(preview["affectedMemoryCount"], 2);

    storage
        .apply_tag_mutation(
            &sources,
            "canonical",
            None,
            preview_token(&preview),
            "tag_rename",
            "explicit cross-scope rename",
        )
        .unwrap();
    assert_eq!(
        storage.get_node(&user.id).unwrap().unwrap().tags,
        vec!["canonical"]
    );
    assert_eq!(
        storage.get_node(&project.id).unwrap().unwrap().tags,
        vec!["canonical"]
    );
}

#[test]
fn list_tag_operations_is_not_buried_by_later_merge_rows() {
    let storage = create_test_storage();
    ingest_tagged_in_scope(&storage, "user", "tag burial fixture", &["old"]);
    let sources = vec!["old".to_string()];
    let preview = storage
        .preview_tag_mutation(&sources, "new", Some("user"))
        .unwrap();
    let operation = storage
        .apply_tag_mutation(
            &sources,
            "new",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "must remain listed after later merge rows",
        )
        .unwrap();

    {
        let writer = storage.writer.lock().unwrap();
        for index in 0..20 {
            writer
                    .execute(
                        "INSERT INTO merge_operations
                            (id, plan_id, op_type, status, created_at, reverted_at, reverts_op_id,
                             survivor_id, affected_ids, confidence, signals, reason, undo_payload)
                         VALUES (?1, NULL, 'merge', 'applied', ?2, NULL, NULL, NULL, '[]', NULL, NULL, 'later merge', '{}')",
                        params![
                            format!("merge-later-{index:02}"),
                            "2099-01-01T00:00:00+00:00"
                        ],
                    )
                    .unwrap();
        }
    }

    let mixed = storage.list_merge_operations(20).unwrap();
    assert_eq!(mixed.len(), 20);
    assert!(
        mixed
            .iter()
            .all(|operation| operation.op_type != "tag_rename"),
        "the mixed window fills with later merge rows"
    );
    let tags = storage.list_tag_operations(50, None).unwrap();
    assert_eq!(tags.len(), 1);
    assert_eq!(tags[0].id, operation.id);
    let scoped = storage.list_tag_operations(50, Some("user")).unwrap();
    assert_eq!(scoped.len(), 1);
}

#[test]
fn tag_merge_normalizes_sources_and_rejects_stale_preview_or_undo_conflict() {
    let storage = create_test_storage();
    let node = ingest_tagged_in_scope(
        &storage,
        "user",
        "multi source tag merge",
        &["keep", "beta", "alpha", "target", "target", "tail"],
    );
    // Duplicate sources dedupe; matching is byte-exact (padded variants
    // are separate, reachable keys — see the padded-source test).
    let sources = vec!["beta".to_string(), "alpha".to_string(), "alpha".to_string()];
    let preview = storage
        .preview_tag_mutation(&sources, "target", Some("user"))
        .unwrap();

    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET tags = ?1 WHERE id = ?2",
                params![
                    serde_json::json!(["keep", "beta", "alpha", "drift"]).to_string(),
                    &node.id
                ],
            )
            .unwrap();
    }
    let stale_error = storage
        .apply_tag_mutation(
            &sources,
            "target",
            Some("user"),
            preview_token(&preview),
            "tag_merge",
            "merge aliases",
        )
        .unwrap_err();
    assert!(stale_error.to_string().contains("stale"));
    assert!(storage.list_merge_operations(20).unwrap().is_empty());

    let fresh = storage
        .preview_tag_mutation(&sources, "target", Some("user"))
        .unwrap();
    let operation = storage
        .apply_tag_mutation(
            &sources,
            "target",
            Some("user"),
            preview_token(&fresh),
            "tag_merge",
            "merge aliases",
        )
        .unwrap();
    assert_eq!(
        storage.get_node(&node.id).unwrap().unwrap().tags,
        vec!["keep", "target", "drift"]
    );

    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET tags = ?1 WHERE id = ?2",
                params![
                    serde_json::json!(["keep", "target", "later-edit"]).to_string(),
                    &node.id
                ],
            )
            .unwrap();
    }
    let conflict = storage.undo_tag_mutation(&operation.id).unwrap_err();
    assert!(conflict.to_string().contains("conflict"));
    assert_eq!(
        storage
            .get_merge_operation(&operation.id)
            .unwrap()
            .unwrap()
            .status,
        "applied",
        "failed undo must not mark the original operation reverted"
    );
}

#[test]
fn tag_mutation_validation_and_malformed_json_fail_without_partial_writes() {
    let storage = create_test_storage();
    let first = ingest_tagged_in_scope(&storage, "user", "first atomic row", &["old"]);
    let second = ingest_tagged_in_scope(&storage, "user", "second atomic row", &["old"]);
    assert!(
        storage
            .preview_tag_mutation(&["same".into()], "same", Some("user"))
            .is_err()
    );
    assert!(
        storage
            .preview_tag_mutation(&["bad\ncontrol".into()], "new", Some("user"))
            .is_err()
    );

    let sources = vec!["old".to_string()];
    let preview = storage
        .preview_tag_mutation(&sources, "new", Some("user"))
        .unwrap();
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET tags = 'not-json' WHERE id = ?1",
                params![&second.id],
            )
            .unwrap();
    }
    let error = storage
        .apply_tag_mutation(
            &sources,
            "new",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "atomic malformed-json test",
        )
        .unwrap_err();
    assert!(error.to_string().contains("invalid tags JSON"));
    assert_eq!(
        storage.get_node(&first.id).unwrap().unwrap().tags,
        vec!["old"]
    );
    assert!(storage.list_merge_operations(20).unwrap().is_empty());

    let valid_storage = create_test_storage();
    ingest_tagged_in_scope(&valid_storage, "user", "no match", &["other"]);
    let empty = valid_storage
        .preview_tag_mutation(&sources, "new", Some("user"))
        .unwrap();
    assert_eq!(empty["affectedMemoryCount"], 0);
    assert!(
        valid_storage
            .apply_tag_mutation(
                &sources,
                "new",
                Some("user"),
                preview_token(&empty),
                "tag_rename",
                "must reject no-op",
            )
            .is_err()
    );
    assert!(
        valid_storage
            .apply_tag_mutation(
                &["other".into()],
                "new",
                Some("user"),
                "tag-plan-v1:wrong",
                "tag_rename",
                "",
            )
            .is_err(),
        "a nonempty audit reason is mandatory"
    );
}

#[test]
fn tag_mutation_rejects_secret_shaped_persistent_fields_without_side_effects() {
    let storage = create_test_storage();
    let node = ingest_tagged_in_scope(&storage, "user", "secret policy fixture", &["old"]);
    let source = vec!["old".to_string()];
    let credential = format!("ghp_{}", "a".repeat(36));

    let target_error = storage
        .preview_tag_mutation(&source, &credential, Some("user"))
        .unwrap_err()
        .to_string();
    assert!(target_error.contains("probable credential"));
    assert!(!target_error.contains(&credential));

    // SOURCE tags are exact-match lookup keys for values that already
    // exist in the store, so a secret-shaped source is accepted (it must
    // stay renameable AWAY); only newly persisted fields are rejected.
    let source_preview = storage
        .preview_tag_mutation(std::slice::from_ref(&credential), "new", Some("user"))
        .unwrap();
    assert_eq!(source_preview["affectedMemoryCount"], 0);

    let safe_preview = storage
        .preview_tag_mutation(&source, "new", Some("user"))
        .unwrap();
    let reason_error = storage
        .apply_tag_mutation(
            &source,
            "new",
            Some("user"),
            preview_token(&safe_preview),
            "tag_rename",
            &credential,
        )
        .unwrap_err()
        .to_string();
    assert!(reason_error.contains("probable credential"));
    assert!(!reason_error.contains(&credential));
    assert_eq!(
        storage.get_node(&node.id).unwrap().unwrap().tags,
        vec!["old"]
    );
    assert!(storage.list_merge_operations(20).unwrap().is_empty());
}

#[test]
fn tag_mutation_row_and_audit_limits_fail_before_writes() {
    let storage = create_test_storage();
    let nodes: Vec<_> = (0..3)
        .map(|index| {
            ingest_tagged_in_scope(
                &storage,
                "user",
                &format!("row limit fixture {index}"),
                &["old"],
            )
        })
        .collect();
    let source = vec!["old".to_string()];
    let preview = storage
        .preview_tag_mutation(&source, "new", Some("user"))
        .unwrap();

    let row_limit_error = storage
        .apply_tag_mutation_with_limits(
            &source,
            "new",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "row limit fixture",
            2,
            MAX_TAG_MUTATION_AUDIT_BYTES,
        )
        .unwrap_err()
        .to_string();
    assert!(row_limit_error.contains("more than 2 memories"));
    assert!(storage.list_merge_operations(20).unwrap().is_empty());
    for node in &nodes {
        assert_eq!(
            storage.get_node(&node.id).unwrap().unwrap().tags,
            vec!["old"]
        );
    }

    let applied = storage
        .apply_tag_mutation_with_limits(
            &source,
            "new",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "exact row limit fixture",
            3,
            MAX_TAG_MUTATION_AUDIT_BYTES,
        )
        .unwrap();
    assert_eq!(applied.affected_ids.len(), 3);

    let audit_storage = create_test_storage();
    let audit_node = ingest_tagged_in_scope(
        &audit_storage,
        "user",
        "audit payload limit fixture",
        &["old"],
    );
    let audit_preview = audit_storage
        .preview_tag_mutation(&source, "new", Some("user"))
        .unwrap();
    let audit_limit_error = audit_storage
        .apply_tag_mutation_with_limits(
            &source,
            "new",
            Some("user"),
            preview_token(&audit_preview),
            "tag_rename",
            "audit payload limit fixture",
            MAX_TAG_MUTATION_MEMORIES,
            1,
        )
        .unwrap_err()
        .to_string();
    assert!(audit_limit_error.contains("1-byte limit"));
    assert_eq!(
        audit_storage
            .get_node(&audit_node.id)
            .unwrap()
            .unwrap()
            .tags,
        vec!["old"]
    );
    assert!(audit_storage.list_merge_operations(20).unwrap().is_empty());
}

#[test]
fn hygiene_snapshot_covers_more_than_five_hundred_without_full_content() {
    let storage = create_test_storage();
    let now = Utc::now().to_rfc3339();
    {
        let mut writer = storage.writer.lock().unwrap();
        let tx = writer.transaction().unwrap();
        for index in 0..501 {
            tx.execute(
                "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed, tags, scope)
                     VALUES (?1, ?2, 'fact', ?3, ?3, ?3, ?4, 'user')",
                params![
                    format!("hygiene-{index:04}"),
                    format!("bounded content {index}"),
                    &now,
                    serde_json::json!(["bulk"]).to_string(),
                ],
            )
            .unwrap();
        }
        tx.commit().unwrap();
    }
    let snapshot = storage.hygiene_snapshot(Some("user")).unwrap();
    assert_eq!(snapshot.nodes.len(), 501);
    assert!(
        snapshot
            .nodes
            .iter()
            .all(|row| row.content_preview.len() <= 240)
    );
    assert!(snapshot.nodes.iter().all(|row| row.never_accessed));
    assert!(snapshot.nodes.iter().all(|row| !row.access_unknown));
    assert_eq!(snapshot.malformed_tag_rows, 0);
    assert_eq!(snapshot.defaulted_retention_rows, 0);
}

#[test]
fn hygiene_snapshot_distinguishes_unknown_pruned_access_from_never_accessed() {
    let storage = create_test_storage();
    let now = Utc::now();
    let fresh = now.to_rfc3339();
    let before_log_window = (now - Duration::days(ACCESS_LOG_RETENTION_DAYS + 110)).to_rfc3339();
    {
        let writer = storage.writer.lock().unwrap();
        // Heavily used old memory: its log rows were pruned, but the
        // durable retrieval counter survives on the node row.
        writer
            .execute(
                "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope, times_retrieved)
                     VALUES ('old-used', 'used before the log window', 'fact',
                             ?1, ?1, ?1, '[]', 'user', 5)",
                params![&before_log_window],
            )
            .unwrap();
        // Old memory with no evidence either way: pruning makes its
        // history unknowable, so it must not be claimed never-accessed.
        writer
            .execute(
                "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope)
                     VALUES ('old-unknown', 'predates the log window', 'fact',
                             ?1, ?1, ?1, '[]', 'user')",
                params![&before_log_window],
            )
            .unwrap();
        // Fresh memory with no accesses: the retained log is complete
        // evidence for its whole lifetime, so never-accessed is provable.
        writer
            .execute(
                "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope)
                     VALUES ('fresh-never', 'created inside the log window', 'fact',
                             ?1, ?1, ?1, '[]', 'user')",
                params![&fresh],
            )
            .unwrap();
    }
    let shown = ingest_tagged_in_scope(&storage, "user", "fresh shown row", &[]);
    storage.record_batch_retrieval(&[&shown.id]).unwrap();

    let snapshot = storage.hygiene_snapshot(Some("user")).unwrap();
    let by_id = |id: &str| {
        snapshot
            .nodes
            .iter()
            .find(|node| node.id == id)
            .unwrap_or_else(|| panic!("snapshot row {id}"))
    };
    let old_used = by_id("old-used");
    assert!(
        !old_used.never_accessed && !old_used.access_unknown,
        "a durable retrieval counter proves past access even after log pruning"
    );
    let old_unknown = by_id("old-unknown");
    assert!(
        !old_unknown.never_accessed,
        "a pre-window row without counters must never be claimed never-accessed"
    );
    assert!(old_unknown.access_unknown);
    let fresh_never = by_id("fresh-never");
    assert!(fresh_never.never_accessed && !fresh_never.access_unknown);
    let shown_row = by_id(&shown.id);
    assert!(!shown_row.never_accessed && !shown_row.access_unknown);
}

#[test]
fn hygiene_snapshot_tolerates_malformed_and_null_legacy_rows() {
    let storage = create_test_storage();
    for index in 0..3 {
        ingest_tagged_in_scope(&storage, "user", &format!("clean row {index}"), &["clean"]);
    }
    let now = Utc::now().to_rfc3339();
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope)
                     VALUES ('bad-json', 'hand-edited tags', 'fact', ?1, ?1, ?1,
                             'not-json', 'user')",
                params![&now],
            )
            .unwrap();
        writer
            .execute(
                "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope, retention_strength)
                     VALUES ('null-tags', 'hand-edited null row', 'fact', ?1, ?1, ?1,
                             NULL, 'user', NULL)",
                params![&now],
            )
            .unwrap();
    }

    let snapshot = storage.hygiene_snapshot(Some("user")).unwrap();
    assert_eq!(
        snapshot.nodes.len(),
        5,
        "aggregates must cover every row, corrupted ones included"
    );
    assert_eq!(snapshot.malformed_tag_rows, 2);
    assert_eq!(
        snapshot.malformed_tag_row_ids,
        vec!["bad-json".to_string(), "null-tags".to_string()]
    );
    assert!(!snapshot.malformed_tag_row_ids_truncated);
    assert_eq!(snapshot.defaulted_retention_rows, 1);
    let null_row = snapshot
        .nodes
        .iter()
        .find(|node| node.id == "null-tags")
        .unwrap();
    assert!(null_row.tags.is_empty());
    assert_eq!(null_row.retention_strength, 1.0);
}

#[test]
fn tag_apply_and_audit_roll_back_together_on_injected_failure() {
    let storage = create_test_storage();
    let first = ingest_tagged_in_scope(&storage, "user", "atomic pair first", &["old"]);
    let second = ingest_tagged_in_scope(&storage, "user", "atomic pair second", &["old"]);
    let sources = vec!["old".to_string()];
    let preview = storage
        .preview_tag_mutation(&sources, "new", Some("user"))
        .unwrap();

    FAIL_TAG_MUTATION_BEFORE_AUDIT.with(|flag| flag.set(true));
    let error = storage
        .apply_tag_mutation(
            &sources,
            "new",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "single transaction fail point",
        )
        .unwrap_err();
    FAIL_TAG_MUTATION_BEFORE_AUDIT.with(|flag| flag.set(false));
    assert!(error.to_string().contains("fail point"));
    // The failure fired AFTER every row UPDATE: rollback must restore all
    // tag arrays AND leave no audit row, proving one shared transaction.
    assert_eq!(
        storage.get_node(&first.id).unwrap().unwrap().tags,
        vec!["old"]
    );
    assert_eq!(
        storage.get_node(&second.id).unwrap().unwrap().tags,
        vec!["old"]
    );
    assert!(storage.list_merge_operations(20).unwrap().is_empty());
    assert!(storage.list_tag_operations(20, None).unwrap().is_empty());

    // Disarmed, the same untouched preview applies normally.
    let applied = storage
        .apply_tag_mutation(
            &sources,
            "new",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "single transaction fail point disarmed",
        )
        .unwrap();
    assert_eq!(applied.affected_ids.len(), 2);
    assert_eq!(
        storage.get_node(&first.id).unwrap().unwrap().tags,
        vec!["new"]
    );
}

#[test]
fn tag_vocabulary_skips_and_counts_overlong_stored_tags() {
    let storage = create_test_storage();
    ingest_tagged_in_scope(&storage, "user", "normal tag row", &["normal"]);
    let overlong = "x".repeat(201);
    ingest_tagged_in_scope(&storage, "user", "overlong tag row", &[&overlong]);

    let vocabulary = storage.tag_vocabulary(Some("user")).unwrap();
    assert_eq!(vocabulary.tags, vec!["normal".to_string()]);
    assert_eq!(vocabulary.skipped_overlong, 1);
}

#[test]
fn overlong_source_tags_can_be_renamed_away_end_to_end() {
    let storage = create_test_storage();
    let overlong = "y".repeat(250);
    let node = ingest_tagged_in_scope(&storage, "user", "overlong rename fixture", &[&overlong]);

    let sources = vec![overlong.clone()];
    let preview = storage
        .preview_tag_mutation(&sources, "short-tag", Some("user"))
        .unwrap();
    assert_eq!(preview["affectedMemoryCount"], 1);
    storage
        .apply_tag_mutation(
            &sources,
            "short-tag",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "repair an overlong stored tag",
        )
        .unwrap();
    assert_eq!(
        storage.get_node(&node.id).unwrap().unwrap().tags,
        vec!["short-tag"]
    );
    let vocabulary = storage.tag_vocabulary(Some("user")).unwrap();
    assert_eq!(vocabulary.skipped_overlong, 0, "the overlong tag is gone");
}

#[test]
fn tag_vocabulary_rejects_more_than_ten_thousand_tags() {
    let storage = create_test_storage();
    let tags: Vec<String> = (0..10_001)
        .map(|index| format!("bulk-{index:05}"))
        .collect();
    let now = Utc::now().to_rfc3339();
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope)
                     VALUES ('vocab-bound', 'vocabulary bound fixture', 'fact', ?1, ?1, ?1,
                             ?2, 'user')",
                params![&now, serde_json::to_string(&tags).unwrap()],
            )
            .unwrap();
    }
    let error = storage
        .tag_vocabulary(Some("user"))
        .unwrap_err()
        .to_string();
    assert!(error.contains("exceeds the 10000-tag"));
}

#[test]
fn padded_source_tags_are_reachable_byte_exact() {
    let storage = create_test_storage();
    let padded = ingest_tagged_in_scope(&storage, "user", "padded tag fixture", &[" prix-six"]);
    let unpadded = ingest_tagged_in_scope(&storage, "user", "unpadded fixture", &["prix-six"]);

    // The padded stored variant is addressable exactly as stored; the
    // trimmed TARGET merges it into the canonical spelling.
    let sources = vec![" prix-six".to_string()];
    let preview = storage
        .preview_tag_mutation(&sources, "prix-six", Some("user"))
        .unwrap();
    assert_eq!(preview["affectedMemoryCount"], 1);
    storage
        .apply_tag_mutation(
            &sources,
            "prix-six",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "collapse a whitespace-padded tag variant",
        )
        .unwrap();
    assert_eq!(
        storage.get_node(&padded.id).unwrap().unwrap().tags,
        vec!["prix-six"]
    );

    // A trimmed source still matches unpadded stored tags as before.
    let trimmed_sources = vec!["prix-six".to_string()];
    let trimmed_preview = storage
        .preview_tag_mutation(&trimmed_sources, "grand-prix", Some("user"))
        .unwrap();
    assert_eq!(trimmed_preview["affectedMemoryCount"], 2);
    storage
        .apply_tag_mutation(
            &trimmed_sources,
            "grand-prix",
            Some("user"),
            preview_token(&trimmed_preview),
            "tag_rename",
            "rename the canonical spelling",
        )
        .unwrap();
    assert_eq!(
        storage.get_node(&unpadded.id).unwrap().unwrap().tags,
        vec!["grand-prix"]
    );
}

#[test]
fn secret_shaped_stored_tags_can_be_renamed_away() {
    let storage = create_test_storage();
    let credential = format!("ghp_{}", "b".repeat(36));
    let node = storage
        .ingest_with_secret_policy(
            IngestInput {
                content: "explicit-allow credential tag fixture".to_string(),
                tags: vec![credential.clone(), "keep".to_string()],
                ..Default::default()
            },
            SecretPolicy::AllowExplicitly,
        )
        .unwrap();

    let sources = vec![credential.clone()];
    let preview = storage
        .preview_tag_mutation(&sources, "rotated-token-reference", Some("user"))
        .unwrap();
    assert_eq!(preview["affectedMemoryCount"], 1);
    storage
        .apply_tag_mutation(
            &sources,
            "rotated-token-reference",
            Some("user"),
            preview_token(&preview),
            "tag_rename",
            "replace a credential-shaped tag with a safe reference",
        )
        .unwrap();
    assert_eq!(
        storage.get_node(&node.id).unwrap().unwrap().tags,
        vec!["rotated-token-reference", "keep"]
    );
}

#[test]
fn scoped_tag_operation_listing_includes_all_scopes_operations() {
    let storage = create_test_storage();
    ingest_tagged_in_scope(&storage, "user", "scoped audit row", &["scoped-old"]);
    ingest_tagged_in_scope(&storage, "user", "shared audit row", &["shared-old"]);
    ingest_tagged_in_scope(&storage, "project-b", "other scope row", &["shared-old"]);

    let scoped_sources = vec!["scoped-old".to_string()];
    let scoped_preview = storage
        .preview_tag_mutation(&scoped_sources, "scoped-new", Some("user"))
        .unwrap();
    let scoped_op = storage
        .apply_tag_mutation(
            &scoped_sources,
            "scoped-new",
            Some("user"),
            preview_token(&scoped_preview),
            "tag_rename",
            "scoped audit fixture",
        )
        .unwrap();

    let shared_sources = vec!["shared-old".to_string()];
    let shared_preview = storage
        .preview_tag_mutation(&shared_sources, "shared-new", None)
        .unwrap();
    let shared_op = storage
        .apply_tag_mutation(
            &shared_sources,
            "shared-new",
            None,
            preview_token(&shared_preview),
            "tag_rename",
            "all-scopes audit fixture",
        )
        .unwrap();

    let user_view = storage.list_tag_operations(50, Some("user")).unwrap();
    let user_ids: Vec<&str> = user_view.iter().map(|op| op.id.as_str()).collect();
    assert!(
        user_ids.contains(&scoped_op.id.as_str()) && user_ids.contains(&shared_op.id.as_str()),
        "a scope's audit must show all-scopes operations that rewrote it"
    );

    let other_view = storage.list_tag_operations(50, Some("project-b")).unwrap();
    assert_eq!(other_view.len(), 1);
    assert_eq!(other_view[0].id, shared_op.id);

    let all_view = storage.list_tag_operations(50, None).unwrap();
    assert_eq!(all_view.len(), 2);
}

#[test]
fn test_parse_timestamp_accepts_rfc3339_and_sqlite_native() {
    use chrono::TimeZone;

    // Canonical writer: RFC 3339 with fractional seconds + offset.
    let rfc = Storage::parse_timestamp("2026-06-12T15:07:59.730+00:00", "last_accessed").unwrap();
    assert_eq!(rfc.to_rfc3339(), "2026-06-12T15:07:59.730+00:00");

    // External writer: SQLite-native `datetime('now')` (space separator,
    // no timezone, no fraction) — must be tolerated, assumed UTC.
    let sqlite = Storage::parse_timestamp("2026-06-12 15:07:59", "last_accessed").unwrap();
    assert_eq!(
        sqlite,
        Utc.with_ymd_and_hms(2026, 6, 12, 15, 7, 59).unwrap()
    );

    // SQLite-native with fractional seconds.
    let sqlite_frac = Storage::parse_timestamp("2026-06-12 15:07:59.730", "last_accessed").unwrap();
    assert_eq!(sqlite_frac.timestamp_subsec_millis(), 730);

    // Genuinely malformed input still errors.
    assert!(Storage::parse_timestamp("not-a-timestamp", "last_accessed").is_err());
}

#[test]
fn test_ingest_and_get() {
    let storage = create_test_storage();

    let input = IngestInput {
        content: "Test memory content".to_string(),
        node_type: "fact".to_string(),
        ..Default::default()
    };

    let node = storage.ingest(input).unwrap();
    assert!(!node.id.is_empty());
    assert_eq!(node.content, "Test memory content");

    let retrieved = storage.get_node(&node.id).unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().content, "Test memory content");
}

#[test]
fn test_search() {
    let storage = create_test_storage();

    let input = IngestInput {
        content: "The mitochondria is the powerhouse of the cell".to_string(),
        node_type: "fact".to_string(),
        ..Default::default()
    };

    storage.ingest(input).unwrap();

    let results = storage.search("mitochondria", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].content.contains("mitochondria"));
}

#[test]
fn passive_recall_records_telemetry_without_reinforcing() {
    let storage = create_test_storage();
    let node = storage
        .ingest(IngestInput {
            content: "current version is 2.0.12 as of 2026-03-04".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let before = storage.get_node(&node.id).unwrap().unwrap();

    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET activation = 0.73 WHERE id = ?1",
                params![&node.id],
            )
            .unwrap();
    }

    for _ in 0..3 {
        let recalled = storage
            .recall(RecallInput {
                query: "current version".to_string(),
                limit: 10,
                min_retention: 0.0,
                search_mode: SearchMode::Keyword,
                valid_at: None,
            })
            .unwrap();
        assert_eq!(recalled.len(), 1);
    }

    let after = storage.get_node(&node.id).unwrap().unwrap();
    assert_eq!(after.retrieval_strength, before.retrieval_strength);
    assert_eq!(after.retention_strength, before.retention_strength);
    assert_eq!(after.stability, before.stability);
    assert_eq!(after.last_accessed, before.last_accessed);
    assert_eq!(after.reps, before.reps);
    assert_eq!(after.next_review, before.next_review);
    assert_eq!(after.times_retrieved, before.times_retrieved);
    assert_eq!(after.times_useful, before.times_useful);
    assert_eq!(after.utility_score, before.utility_score);
    assert_eq!(storage.auto_promote_frequent_access().unwrap(), 0);
    storage.compute_act_r_activations().unwrap();

    let reader = storage.reader.lock().unwrap();
    let retrievals_shown: i64 = reader
            .query_row(
                "SELECT COUNT(*) FROM memory_access_log WHERE node_id = ?1 AND access_type = 'retrieval_shown'",
                params![&node.id],
                |row| row.get(0),
            )
            .unwrap();
    assert_eq!(retrievals_shown, 3);
    let activation: f64 = reader
        .query_row(
            "SELECT activation FROM knowledge_nodes WHERE id = ?1",
            params![&node.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(activation, 0.0);
}

#[test]
fn explicit_promotion_marks_a_retrieved_memory_useful() {
    let storage = create_test_storage();
    let node = storage
        .ingest(IngestInput {
            content: "A memory that needs explicit positive feedback".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    storage.demote_memory(&node.id).unwrap();
    let before = storage.get_node(&node.id).unwrap().unwrap();

    storage.record_batch_retrieval(&[&node.id]).unwrap();
    let promoted = storage.promote_memory(&node.id).unwrap();

    assert!(promoted.retrieval_strength > before.retrieval_strength);
    assert!(promoted.retention_strength > before.retention_strength);
    assert_eq!(promoted.times_retrieved.unwrap_or_default(), 0);
    assert_eq!(promoted.times_useful.unwrap_or_default(), 1);
    assert_eq!(promoted.utility_score.unwrap_or_default(), 1.0);
}

#[test]
fn legacy_search_hits_cannot_reinforce_or_reactivate_a_memory() {
    let storage = create_test_storage();
    let node = storage
        .ingest(IngestInput {
            content: "legacy dated current version claim".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes
                     SET retrieval_strength = 0.50, retention_strength = 0.40, activation = 0.73
                     WHERE id = ?1",
                params![&node.id],
            )
            .unwrap();
        for _ in 0..3 {
            writer
                .execute(
                    "INSERT INTO memory_access_log (node_id, access_type, accessed_at)
                         VALUES (?1, 'search_hit', ?2)",
                    params![&node.id, Utc::now().to_rfc3339()],
                )
                .unwrap();
        }
    }
    let before = storage.get_node(&node.id).unwrap().unwrap();

    assert_eq!(storage.auto_promote_frequent_access().unwrap(), 0);
    storage.compute_act_r_activations().unwrap();

    let after = storage.get_node(&node.id).unwrap().unwrap();
    assert_eq!(after.retrieval_strength, before.retrieval_strength);
    assert_eq!(after.retention_strength, before.retention_strength);
    assert_eq!(after.last_accessed, before.last_accessed);

    let reader = storage.reader.lock().unwrap();
    let activation: f64 = reader
        .query_row(
            "SELECT activation FROM knowledge_nodes WHERE id = ?1",
            params![&node.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(activation, 0.0);
}

#[test]
fn legacy_search_hits_cannot_preserve_recency_or_delay_decay() {
    let storage = create_test_storage();
    let node = storage
        .ingest(IngestInput {
            content: "a legacy passive search must not preserve freshness".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let created_at = Utc::now() - Duration::days(30);
    let passive_at = Utc::now();

    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET
                        created_at = ?1,
                        updated_at = ?1,
                        last_accessed = ?2,
                        retrieval_strength = 1.0,
                        retention_strength = 1.0
                     WHERE id = ?3",
                params![created_at.to_rfc3339(), passive_at.to_rfc3339(), &node.id],
            )
            .unwrap();
        writer
            .execute(
                "INSERT INTO memory_access_log (node_id, access_type, accessed_at)
                     VALUES (?1, 'search_hit', ?2)",
                params![&node.id, passive_at.to_rfc3339()],
            )
            .unwrap();
    }

    storage.compute_act_r_activations().unwrap();
    let repaired = storage.get_node(&node.id).unwrap().unwrap();
    assert_eq!(repaired.last_accessed, created_at);

    storage.apply_decay().unwrap();
    let decayed = storage.get_node(&node.id).unwrap().unwrap();
    assert!(decayed.retrieval_strength < 1.0);
    assert!(decayed.retention_strength < 1.0);
}

#[test]
fn legacy_recency_repair_preserves_reviewed_memories() {
    let storage = create_test_storage();
    let node = storage
        .ingest(IngestInput {
            content: "a reviewed memory must not be reset by passive logs".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    storage.promote_memory(&node.id).unwrap();
    let reviewed = storage.mark_reviewed(&node.id, Rating::Good).unwrap();

    // New telemetry never changed state, so it must never trigger a
    // legacy-state repair.
    storage.record_batch_retrieval(&[&node.id]).unwrap();
    storage.compute_act_r_activations().unwrap();
    let after_new_telemetry = storage.get_node(&node.id).unwrap().unwrap();
    assert_eq!(after_new_telemetry.last_accessed, reviewed.last_accessed);

    // If an old search-hit row was recorded after a review, restore the
    // review timestamp from updated_at rather than falling back to create.
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET last_accessed = ?1 WHERE id = ?2",
                params![Utc::now().to_rfc3339(), &node.id],
            )
            .unwrap();
        writer
            .execute(
                "INSERT INTO memory_access_log (node_id, access_type, accessed_at)
                     VALUES (?1, 'search_hit', ?2)",
                params![&node.id, Utc::now().to_rfc3339()],
            )
            .unwrap();
    }
    storage.compute_act_r_activations().unwrap();
    let restored = storage.get_node(&node.id).unwrap().unwrap();
    assert_eq!(restored.last_accessed, reviewed.last_accessed);
}

#[test]
fn test_review() {
    let storage = create_test_storage();

    let input = IngestInput {
        content: "Test review".to_string(),
        node_type: "fact".to_string(),
        ..Default::default()
    };

    let node = storage.ingest(input).unwrap();
    assert_eq!(node.reps, 0);

    let reviewed = storage.mark_reviewed(&node.id, Rating::Good).unwrap();
    assert_eq!(reviewed.reps, 1);
}

#[test]
fn test_delete() {
    let storage = create_test_storage();

    let input = IngestInput {
        content: "To be deleted".to_string(),
        node_type: "fact".to_string(),
        tags: vec!["sensitive-delete-tag".to_string()],
        ..Default::default()
    };

    let node = storage.ingest(input).unwrap();
    assert!(storage.get_node(&node.id).unwrap().is_some());

    let deleted = storage.delete_node(&node.id).unwrap();
    assert!(deleted);
    assert!(storage.get_node(&node.id).unwrap().is_none());
    let archive = serde_json::to_string(&storage.export_portable_archive().unwrap()).unwrap();
    assert!(!archive.contains(&node.id));
    assert!(!archive.contains("sensitive-delete-tag"));
}

/// REGRESSION (v2.6.0 data-safety): consolidation must never delete a
/// memory. Until this release, an autonomic "retention target" GC inside
/// run_consolidation hard-deleted everything below 0.3 retention older
/// than 30 days — dormant only while decay was broken, and it destroyed
/// 23 real memories from a live store the day decay was fixed. This test
/// constructs exactly that scenario and asserts nothing dies.
#[test]
fn consolidation_never_deletes_low_retention_memories() {
    let storage = create_test_storage();
    let mut ids = Vec::new();
    for i in 0..4 {
        let node = storage
            .ingest(IngestInput {
                content: format!("Old low-retention memory number {i} that must survive"),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        ids.push(node.id);
    }
    // Force the doomed profile the old reaper keyed on: retention far
    // below 0.3 and created_at far older than 30 days.
    {
        let writer = storage.writer.lock().unwrap();
        let old = (Utc::now() - Duration::days(120)).to_rfc3339();
        for id in &ids {
            writer
                .execute(
                    "UPDATE knowledge_nodes
                         SET retention_strength = 0.05, created_at = ?1, last_accessed = ?1
                         WHERE id = ?2",
                    params![old, id],
                )
                .unwrap();
        }
    }
    let before: i64 = {
        let reader = storage.reader.lock().unwrap();
        reader
            .query_row("SELECT COUNT(*) FROM knowledge_nodes", [], |r| r.get(0))
            .unwrap()
    };

    storage.run_consolidation().unwrap();

    let after: i64 = {
        let reader = storage.reader.lock().unwrap();
        reader
            .query_row("SELECT COUNT(*) FROM knowledge_nodes", [], |r| r.get(0))
            .unwrap()
    };
    assert_eq!(before, after, "consolidation deleted memories");
    for id in &ids {
        assert!(
            storage.get_node(id).unwrap().is_some(),
            "low-retention memory {id} was reaped by consolidation"
        );
    }
}

/// The explicit GC path must never collect a protected (pinned) memory,
/// no matter how decayed it is. A pin says "keep this"; low retention
/// only says "rarely retrieved".
#[test]
fn gc_spares_protected_memories() {
    let storage = create_test_storage();
    let pinned = storage
        .ingest(IngestInput {
            content: "Pinned but heavily decayed memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let doomed = storage
        .ingest(IngestInput {
            content: "Unpinned decayed memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    {
        let writer = storage.writer.lock().unwrap();
        let old = (Utc::now() - Duration::days(120)).to_rfc3339();
        for id in [&pinned.id, &doomed.id] {
            writer
                .execute(
                    "UPDATE knowledge_nodes
                         SET retention_strength = 0.05, created_at = ?1 WHERE id = ?2",
                    params![old, id],
                )
                .unwrap();
        }
    }
    storage.set_protected(&pinned.id, true).unwrap();

    let deleted = storage.gc_below_retention(0.3, 30).unwrap();
    assert_eq!(deleted, 1, "only the unpinned memory is collected");
    assert!(
        storage.get_node(&pinned.id).unwrap().is_some(),
        "pin survives GC"
    );
    assert!(storage.get_node(&doomed.id).unwrap().is_none());
}

#[test]
fn gc_uses_the_privacy_cleanup_deletion_path() {
    let storage = create_test_storage();
    let node = storage
        .ingest(IngestInput {
            content: "GC deletion privacy target".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["gc-sensitive-tag".to_string()],
            ..Default::default()
        })
        .unwrap();
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes
                     SET retention_strength = 0.0, created_at = '2000-01-01T00:00:00Z'
                     WHERE id = ?1",
                params![&node.id],
            )
            .unwrap();
    }

    assert_eq!(storage.gc_below_retention(0.1, 1).unwrap(), 1);
    let archive = serde_json::to_string(&storage.export_portable_archive().unwrap()).unwrap();
    assert!(!archive.contains(&node.id));
    assert!(!archive.contains("gc-sensitive-tag"));
}

#[test]
fn purging_empty_content_does_not_scrub_unrelated_evidence() {
    let storage = create_test_storage();
    let empty = storage
        .ingest(IngestInput {
            content: String::new(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "INSERT INTO memory_prs (
                        id, kind, status, title, diff, signals, created_at
                     ) VALUES ('unrelated-review', 'new_fact', 'pending',
                               'keep this review', '{}', '[]', ?1)",
                params![Utc::now().to_rfc3339()],
            )
            .unwrap();
    }

    assert!(storage.purge_node(&empty.id, None).unwrap().deleted);
    let writer = storage.writer.lock().unwrap();
    let remaining_reviews: i64 = writer
        .query_row(
            "SELECT COUNT(*) FROM memory_prs WHERE id = 'unrelated-review'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(remaining_reviews, 1);
}

/// GhostLink contract (2026-09-28): never-composed pairs are admitted ONLY
/// via recorded typed-edge hops.
fn seed_typed_edge(storage: &crate::storage::Storage, a: &str, b: &str) {
    use crate::storage::ConnectionRecord;
    let now = chrono::Utc::now();
    storage
        .save_connection(&ConnectionRecord {
            source_id: a.to_string(),
            target_id: b.to_string(),
            strength: 1.0,
            link_type: "touched".to_string(),
            created_at: now,
            last_activated: now,
            activation_count: 0,
        })
        .unwrap();
}

/// No typed edges + shared tags only => never_composed returns NOTHING.
#[test]
fn never_composed_tags_only_returns_empty() {
    let storage = create_test_storage();
    storage
        .ingest(IngestInput {
            content: "Alpha lane with shared vocabulary only.".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["shared-tag".to_string()],
            ..Default::default()
        })
        .unwrap();
    storage
        .ingest(IngestInput {
            content: "Beta lane with the same shared vocabulary.".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["shared-tag".to_string(), "extra".to_string()],
            ..Default::default()
        })
        .unwrap();
    let out = storage.get_never_composed_candidates(5, None).unwrap();
    assert!(
        out.is_empty(),
        "shared words must never admit a pair: {out:?}"
    );
}

/// A typed-edge hop admits the pair even with zero shared vocabulary.
#[test]
fn never_composed_admits_via_typed_edge_hop() {
    let storage = create_test_storage();
    let a = storage
        .ingest(IngestInput {
            content: "Commit touched the withdrawal queue module.".to_string(),
            node_type: "event".to_string(),
            tags: vec![],
            ..Default::default()
        })
        .unwrap();
    let b = storage
        .ingest(IngestInput {
            content: "Unrelated wording entirely, no shared tokens at all.".to_string(),
            node_type: "fact".to_string(),
            tags: vec![],
            ..Default::default()
        })
        .unwrap();
    seed_typed_edge(&storage, &a.id, &b.id);
    let out = storage.get_never_composed_candidates(5, None).unwrap();
    assert_eq!(out.len(), 1, "edge hop must admit: {out:?}");
    assert!(
        out[0].reason.contains("typed-edge hop"),
        "{}",
        out[0].reason
    );
}

#[test]
fn test_composition_save_query_outcome_and_never_composed() {
    let storage = create_test_storage();
    let first = storage
        .ingest(IngestInput {
            content: "Oracle drift can break delayed settlement.".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["protocolgate".to_string()],
            ..Default::default()
        })
        .unwrap();
    let second = storage
        .ingest(IngestInput {
            content: "Withdrawal queue needs proof of reserve.".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["protocolgate".to_string()],
            ..Default::default()
        })
        .unwrap();

    // GhostLink contract: without a typed edge, shared tags admit nothing.
    assert!(
        storage
            .get_never_composed_candidates(10, None)
            .unwrap()
            .is_empty()
    );

    seed_typed_edge(&storage, &first.id, &second.id);
    let before = storage.get_never_composed_candidates(10, None).unwrap();
    assert_eq!(before.len(), 1, "edge hop must admit: {before:?}");
    assert!(
        { before[0].first_id == first.id && before[0].second_id == second.id } || {
            before[0].first_id == second.id && before[0].second_id == first.id
        }
    );
    assert!(
        before[0].reason.contains("typed-edge hop"),
        "{}",
        before[0].reason
    );

    let event = CompositionEventRecord {
        id: "composition-test-1".to_string(),
        created_at: Utc::now(),
        tool: "deep_reference".to_string(),
        mode: "bounty".to_string(),
        query: Some("oracle drift withdrawal queue".to_string()),
        query_hash: Some("sha256:test".to_string()),
        confidence: Some(0.87),
        status: Some("resolved".to_string()),
        output_preview: Some("Compose oracle drift with withdrawal queue.".to_string()),
        metadata: serde_json::json!({"workflow": "test"}),
    };
    storage
        .save_composition(
            &event,
            &[
                CompositionMemberRecord {
                    event_id: event.id.clone(),
                    memory_id: first.id.clone(),
                    role: "primary".to_string(),
                    rank: 0,
                    trust: Some(0.9),
                    score: Some(0.9),
                    preview: None,
                    metadata: serde_json::json!({}),
                },
                CompositionMemberRecord {
                    event_id: event.id.clone(),
                    memory_id: second.id.clone(),
                    role: "supporting".to_string(),
                    rank: 1,
                    trust: Some(0.7),
                    score: Some(0.6),
                    preview: None,
                    metadata: serde_json::json!({}),
                },
            ],
            &[],
        )
        .unwrap();

    let after = storage.get_never_composed_candidates(10, None).unwrap();
    assert!(
        after.is_empty(),
        "composed pair must leave the never-composed set: {after:?}"
    );
}

#[test]
fn test_composition_neighbors_count_distinct_events_not_member_roles() {
    let storage = create_test_storage();
    let first = storage
        .ingest(IngestInput {
            content: "Oracle role appears once in the event.".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["protocolgate".to_string(), "settlement".to_string()],
            ..Default::default()
        })
        .unwrap();
    let second = storage
        .ingest(IngestInput {
            content: "Queue role appears under two evidence roles.".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["protocolgate".to_string(), "settlement".to_string()],
            ..Default::default()
        })
        .unwrap();

    storage
        .save_composition(
            &CompositionEventRecord {
                id: "multi-role-neighbor-event".to_string(),
                created_at: Utc::now(),
                tool: "deep_reference".to_string(),
                mode: "bounty".to_string(),
                query: Some("multi role neighbor".to_string()),
                query_hash: Some("fnv1a64:neighbor".to_string()),
                confidence: Some(0.7),
                status: Some("resolved".to_string()),
                output_preview: None,
                metadata: serde_json::json!({}),
            },
            &[
                CompositionMemberRecord {
                    event_id: "multi-role-neighbor-event".to_string(),
                    memory_id: first.id.clone(),
                    role: "primary".to_string(),
                    rank: 0,
                    trust: Some(0.8),
                    score: Some(0.9),
                    preview: None,
                    metadata: serde_json::json!({}),
                },
                CompositionMemberRecord {
                    event_id: "multi-role-neighbor-event".to_string(),
                    memory_id: second.id.clone(),
                    role: "supporting".to_string(),
                    rank: 1,
                    trust: Some(0.7),
                    score: Some(0.8),
                    preview: None,
                    metadata: serde_json::json!({}),
                },
                CompositionMemberRecord {
                    event_id: "multi-role-neighbor-event".to_string(),
                    memory_id: second.id.clone(),
                    role: "related".to_string(),
                    rank: 2,
                    trust: Some(0.7),
                    score: Some(0.6),
                    preview: None,
                    metadata: serde_json::json!({}),
                },
            ],
            &[],
        )
        .unwrap();

    let neighbors = storage.get_composition_neighbors(&first.id, 10).unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].memory_id, second.id);
    assert_eq!(
        neighbors[0].composed_count, 1,
        "one event with multiple member roles should count as one composition"
    );
}

#[test]
fn composition_candidates_exclude_inactive_memories() {
    let storage = create_test_storage();
    let mut ids = Vec::new();
    for label in ["active", "suppressed", "superseded", "expired", "future"] {
        ids.push(
            storage
                .ingest(IngestInput {
                    content: format!("composition lifecycle {label}"),
                    node_type: "fact".into(),
                    ..Default::default()
                })
                .unwrap()
                .id,
        );
    }
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET suppression_count = 1 WHERE id = ?1",
                params![ids[1]],
            )
            .unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET superseded_by = ?1 WHERE id = ?2",
                params![ids[0], ids[2]],
            )
            .unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET valid_until = ?1 WHERE id = ?2",
                params![
                    (Utc::now() - chrono::Duration::days(1)).to_rfc3339(),
                    ids[3]
                ],
            )
            .unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET valid_from = ?1 WHERE id = ?2",
                params![
                    (Utc::now() + chrono::Duration::days(1)).to_rfc3339(),
                    ids[4]
                ],
            )
            .unwrap();
    }
    for scope in [None, Some("user")] {
        let candidates = storage.composition_candidate_nodes(None, scope).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].id, ids[0]);
    }
}

#[test]
fn test_never_composed_tag_filter_includes_older_tagged_candidates() {
    let storage = create_test_storage();
    let first = storage
        .ingest(IngestInput {
            content: "Older Vestige composition frontier about outcome-shaped recall.".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["project:vestige".to_string(), "composition".to_string()],
            ..Default::default()
        })
        .unwrap();
    let second = storage
        .ingest(IngestInput {
            content: "Older Vestige composition frontier about never-composed recall.".to_string(),
            node_type: "pattern".to_string(),
            tags: vec!["project:vestige".to_string(), "composition".to_string()],
            ..Default::default()
        })
        .unwrap();

    for idx in 0..751 {
        storage
            .ingest(IngestInput {
                content: format!("Unrelated recent memory {idx} for scan-window pressure."),
                node_type: "fact".to_string(),
                tags: vec!["unrelated".to_string()],
                ..Default::default()
            })
            .unwrap();
    }

    // GhostLink contract: admission needs the causal edge; the tag filter
    // still scopes the candidate pool (both nodes carry project:vestige).
    seed_typed_edge(&storage, &first.id, &second.id);
    let candidates = storage
        .get_never_composed_candidates(10, Some(&["project".to_string()]))
        .unwrap();
    assert!(
        candidates.iter().any(|candidate| {
            let pair = Storage::pair_key(&candidate.first_id, &candidate.second_id);
            pair == Storage::pair_key(&first.id, &second.id)
        }),
        "tag-filtered frontier should include older namespaced-tag memories outside the base scan window"
    );
}

#[test]
fn test_never_composed_carries_prior_outcome_signal() {
    let storage = create_test_storage();
    let first = storage
        .ingest(IngestInput {
            content: "Oracle drift lane previously looked duplicate-prone.".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["protocolgate".to_string()],
            ..Default::default()
        })
        .unwrap();
    let second = storage
        .ingest(IngestInput {
            content: "Withdrawal queue lane had weak proof.".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["protocolgate".to_string()],
            ..Default::default()
        })
        .unwrap();
    seed_typed_edge(&storage, &first.id, &second.id);

    let event = CompositionEventRecord {
        id: "prior-outcome-composition".to_string(),
        created_at: Utc::now(),
        tool: "deep_reference".to_string(),
        mode: "release".to_string(),
        query: Some("accepted release lane".to_string()),
        query_hash: Some("fnv1a64:success".to_string()),
        confidence: Some(0.9),
        status: Some("resolved".to_string()),
        output_preview: None,
        metadata: serde_json::json!({}),
    };
    storage
        .save_composition(
            &event,
            &[
                CompositionMemberRecord {
                    event_id: event.id.clone(),
                    memory_id: first.id.clone(),
                    role: "primary".to_string(),
                    rank: 0,
                    trust: Some(0.9),
                    score: Some(0.9),
                    preview: None,
                    metadata: serde_json::json!({}),
                },
                CompositionMemberRecord {
                    event_id: event.id.clone(),
                    memory_id: second.id.clone(),
                    role: "supporting".to_string(),
                    rank: 1,
                    trust: Some(0.7),
                    score: Some(0.6),
                    preview: None,
                    metadata: serde_json::json!({}),
                },
            ],
            &[CompositionOutcomeRecord {
                id: "prior-outcome-label".to_string(),
                event_id: event.id.clone(),
                outcome_type: "accepted".to_string(),
                labeled_at: Utc::now(),
                label_source: "test".to_string(),
                confidence_delta: Some(0.2),
                notes: None,
                metadata: serde_json::json!({}),
            }],
        )
        .unwrap();

    let candidates = storage.get_never_composed_candidates(10, None).unwrap();
    assert!(
        candidates.is_empty(),
        "composed pair excluded: {candidates:?}"
    );
}

#[test]
fn test_never_composed_marks_mixed_prior_outcomes() {
    let storage = create_test_storage();
    let successful = storage
        .ingest(IngestInput {
            content: "Successful lane fact.".to_string(),
            node_type: "fact".to_string(),
            tags: vec![],
            ..Default::default()
        })
        .unwrap();
    let closed = storage
        .ingest(IngestInput {
            content: "Closed lane fact.".to_string(),
            node_type: "fact".to_string(),
            tags: vec![],
            ..Default::default()
        })
        .unwrap();
    seed_typed_edge(&storage, &successful.id, &closed.id);
    let out = storage.get_never_composed_candidates(10, None).unwrap();
    assert_eq!(out.len(), 1);
    assert!(
        out[0].reason.contains("typed-edge hop"),
        "{}",
        out[0].reason
    );
}

#[test]
fn test_dream_history_save_and_get_last() {
    let storage = create_test_storage();
    let now = Utc::now();

    let record = DreamHistoryRecord {
        dreamed_at: now,
        duration_ms: 1500,
        memories_replayed: 50,
        connections_found: 12,
        insights_generated: 3,
        memories_strengthened: 8,
        memories_compressed: 2,
        phase_nrem1_ms: None,
        phase_nrem3_ms: None,
        phase_rem_ms: None,
        phase_integration_ms: None,
        summaries_generated: None,
        emotional_memories_processed: None,
        creative_connections_found: None,
    };

    let id = storage.save_dream_history(&record).unwrap();
    assert!(id > 0);

    let last = storage.get_last_dream().unwrap();
    assert!(last.is_some());
    // Timestamps should be within 1 second (RFC3339 round-trip)
    let diff = (last.unwrap() - now).num_seconds().abs();
    assert!(diff <= 1, "Timestamp mismatch: diff={}s", diff);
}

#[test]
fn test_dream_history_empty() {
    let storage = create_test_storage();
    let last = storage.get_last_dream().unwrap();
    assert!(last.is_none());
}

#[test]
fn test_count_memories_since() {
    let storage = create_test_storage();
    let before = Utc::now() - Duration::seconds(10);

    for i in 0..5 {
        storage
            .ingest(IngestInput {
                content: format!("Count test memory {}", i),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
    }

    let count = storage.count_memories_since(before).unwrap();
    assert_eq!(count, 5);

    let future = Utc::now() + Duration::hours(1);
    let count_future = storage.count_memories_since(future).unwrap();
    assert_eq!(count_future, 0);
}

#[test]
fn test_portable_archive_exact_round_trip() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");

    let first = source
        .ingest(IngestInput {
            content: "Portable archive alpha memory".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["portable".to_string()],
            source: Some("test".to_string()),
            ..Default::default()
        })
        .unwrap();
    let second = source
        .ingest(IngestInput {
            content: "Portable archive beta memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    source.mark_reviewed(&first.id, Rating::Good).unwrap();
    source
        .save_connection(&ConnectionRecord {
            source_id: first.id.clone(),
            target_id: second.id.clone(),
            strength: 0.75,
            link_type: "semantic".to_string(),
            created_at: Utc::now(),
            last_activated: Utc::now(),
            activation_count: 1,
        })
        .unwrap();
    source
        .save_composition(
            &CompositionEventRecord {
                id: "portable-composition-1".to_string(),
                created_at: Utc::now(),
                tool: "deep_reference".to_string(),
                mode: "bounty".to_string(),
                query: Some("portable composition".to_string()),
                query_hash: Some("sha256:portable".to_string()),
                confidence: Some(0.9),
                status: Some("resolved".to_string()),
                output_preview: Some("Portable composition event".to_string()),
                metadata: serde_json::json!({}),
            },
            &[
                CompositionMemberRecord {
                    event_id: "portable-composition-1".to_string(),
                    memory_id: first.id.clone(),
                    role: "primary".to_string(),
                    rank: 0,
                    trust: Some(0.9),
                    score: Some(1.0),
                    preview: Some("alpha".to_string()),
                    metadata: serde_json::json!({}),
                },
                CompositionMemberRecord {
                    event_id: "portable-composition-1".to_string(),
                    memory_id: second.id.clone(),
                    role: "supporting".to_string(),
                    rank: 1,
                    trust: Some(0.8),
                    score: Some(0.8),
                    preview: Some("beta".to_string()),
                    metadata: serde_json::json!({}),
                },
            ],
            &[CompositionOutcomeRecord {
                id: "portable-composition-outcome-1".to_string(),
                event_id: "portable-composition-1".to_string(),
                outcome_type: "helpful".to_string(),
                labeled_at: Utc::now(),
                label_source: "test".to_string(),
                confidence_delta: None,
                notes: None,
                metadata: serde_json::json!({}),
            }],
        )
        .unwrap();

    let archive = source.export_portable_archive().unwrap();
    assert_eq!(archive.archive_format, PORTABLE_ARCHIVE_FORMAT);
    assert!(archive.total_rows() >= 3);
    assert!(
        archive
            .tables
            .iter()
            .any(|table| table.name == "knowledge_nodes" && table.rows.len() == 2)
    );
    for table_name in [
        "composition_events",
        "composition_members",
        "composition_outcomes",
    ] {
        assert!(
            archive.tables.iter().any(|table| table.name == table_name),
            "{table_name} must be included in portable archive"
        );
    }

    let target = create_test_storage_at(&target_dir, "target.db");
    let report = target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap();
    assert!(report.rows_imported >= 3);
    assert!(report.fts_rebuilt);

    let restored = target.get_node(&first.id).unwrap().unwrap();
    assert_eq!(restored.id, first.id);
    assert_eq!(restored.content, first.content);
    assert_eq!(restored.tags, first.tags);
    assert_eq!(restored.reps, 1);

    let connections = target.get_connections_for_memory(&first.id).unwrap();
    assert_eq!(connections.len(), 1);
    assert_eq!(connections[0].target_id, second.id);

    let composition = target
        .get_composition_event("portable-composition-1")
        .unwrap()
        .unwrap();
    assert_eq!(composition.mode, "bounty");
    assert_eq!(
        target
            .get_composition_members("portable-composition-1")
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        target
            .get_composition_outcomes("portable-composition-1")
            .unwrap()
            .len(),
        1
    );

    let results = target.search("alpha", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, first.id);
}

#[test]
fn test_portable_import_rejects_non_empty_target() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    source
        .ingest(IngestInput {
            content: "Source memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let archive = source.export_portable_archive().unwrap();

    let target = create_test_storage_at(&target_dir, "target.db");
    target
        .ingest(IngestInput {
            content: "Existing target memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    let err = target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("requires an empty target database")
    );
}

#[test]
fn test_portable_import_rejects_unknown_mode() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    source
        .ingest(IngestInput {
            content: "Source memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let mut archive = source.export_portable_archive().unwrap();
    archive.mode = "merge".to_string();

    let target = create_test_storage_at(&target_dir, "target.db");
    let err = target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("Unsupported portable archive mode")
    );
}

#[test]
fn test_portable_import_rejects_malformed_table_list() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    source
        .ingest(IngestInput {
            content: "Source memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    let mut duplicate_archive = source.export_portable_archive().unwrap();
    let duplicate_table = duplicate_archive
        .tables
        .iter()
        .find(|table| table.name == "knowledge_nodes")
        .unwrap()
        .clone();
    duplicate_archive.tables.push(duplicate_table);

    let target = create_test_storage_at(&target_dir, "target-duplicate.db");
    let err = target
        .import_portable_archive(&duplicate_archive, PortableImportMode::EmptyOnly)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("Portable archive contains duplicate table")
    );

    let mut unknown_archive = source.export_portable_archive().unwrap();
    unknown_archive.tables.push(PortableTable {
        name: "sqlite_sequence".to_string(),
        columns: vec!["name".to_string(), "seq".to_string()],
        rows: vec![],
    });

    let target = create_test_storage_at(&target_dir, "target-unknown.db");
    let err = target
        .import_portable_archive(&unknown_archive, PortableImportMode::EmptyOnly)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("Portable archive contains unsupported table")
    );
}

#[test]
fn test_portable_merge_import_combines_non_empty_databases() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    let target = create_test_storage_at(&target_dir, "target.db");

    let source_node = source
        .ingest(IngestInput {
            content: "Source sync memory".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["sync".to_string()],
            ..Default::default()
        })
        .unwrap();
    let target_node = target
        .ingest(IngestInput {
            content: "Target local memory".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["local".to_string()],
            ..Default::default()
        })
        .unwrap();

    let archive = source.export_portable_archive().unwrap();
    let report = target
        .import_portable_archive(&archive, PortableImportMode::Merge)
        .unwrap();

    assert!(report.rows_inserted > 0);
    assert!(target.get_node(&source_node.id).unwrap().is_some());
    assert!(target.get_node(&target_node.id).unwrap().is_some());
}

#[test]
fn test_portable_merge_import_keeps_newer_local_memory() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    let target = create_test_storage_at(&target_dir, "target.db");

    let node = source
        .ingest(IngestInput {
            content: "Original shared memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let archive = source.export_portable_archive().unwrap();
    target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap();

    let newer = (Utc::now() + Duration::hours(1)).to_rfc3339();
    {
        let writer = target.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET content = ?1, updated_at = ?2 WHERE id = ?3",
                params!["Newer local edit", newer, &node.id],
            )
            .unwrap();
    }

    let report = target
        .import_portable_archive(&archive, PortableImportMode::Merge)
        .unwrap();

    assert!(report.conflicts_kept_local >= 1);
    let restored = target.get_node(&node.id).unwrap().unwrap();
    assert_eq!(restored.content, "Newer local edit");
}

#[test]
fn test_portable_merge_import_keeps_children_for_newer_local_memory() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    let target = create_test_storage_at(&target_dir, "target.db");

    let node = source
        .ingest(IngestInput {
            content: "Shared parent with child rows".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    let source_time = Utc::now().to_rfc3339();
    {
        let writer = source.writer.lock().unwrap();
        writer
            .execute(
                "INSERT OR REPLACE INTO node_embeddings
                     (node_id, embedding, dimensions, model, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                params![&node.id, vec![1_u8, 2, 3, 4], 4, "test-model", &source_time],
            )
            .unwrap();
        writer
            .execute(
                "INSERT OR REPLACE INTO fsrs_cards
                     (memory_id, difficulty, stability, state, reps, lapses)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![&node.id, 3.0_f64, 2.0_f64, "review", 2_i64, 0_i64],
            )
            .unwrap();
        writer
            .execute(
                "INSERT OR REPLACE INTO memory_states
                     (memory_id, state, last_access, access_count, state_entered_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                params![&node.id, "active", &source_time, 1_i64, &source_time],
            )
            .unwrap();
    }

    let archive = source.export_portable_archive().unwrap();
    target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap();

    let local_time = (Utc::now() + Duration::hours(1)).to_rfc3339();
    {
        let writer = target.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET content = ?1, updated_at = ?2 WHERE id = ?3",
                params!["Newer local parent edit", &local_time, &node.id],
            )
            .unwrap();
        writer
            .execute(
                "INSERT OR REPLACE INTO node_embeddings
                     (node_id, embedding, dimensions, model, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                params![&node.id, vec![9_u8, 8, 7, 6], 4, "test-model", &local_time],
            )
            .unwrap();
        writer
            .execute(
                "INSERT OR REPLACE INTO fsrs_cards
                     (memory_id, difficulty, stability, state, reps, lapses)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![&node.id, 9.0_f64, 8.0_f64, "review", 9_i64, 1_i64],
            )
            .unwrap();
        writer
            .execute(
                "INSERT OR REPLACE INTO memory_states
                     (memory_id, state, last_access, access_count, state_entered_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                params![&node.id, "silent", &local_time, 42_i64, &local_time],
            )
            .unwrap();
    }

    let report = target
        .import_portable_archive(&archive, PortableImportMode::Merge)
        .unwrap();

    assert!(report.conflicts_kept_local >= 4);
    let restored = target.get_node(&node.id).unwrap().unwrap();
    assert_eq!(restored.content, "Newer local parent edit");

    let reader = target.reader.lock().unwrap();
    let embedding: Vec<u8> = reader
        .query_row(
            "SELECT embedding FROM node_embeddings WHERE node_id = ?1",
            params![&node.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(embedding, vec![9_u8, 8, 7, 6]);

    let difficulty: f64 = reader
        .query_row(
            "SELECT difficulty FROM fsrs_cards WHERE memory_id = ?1",
            params![&node.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(difficulty, 9.0);

    let (state, access_count): (String, i64) = reader
        .query_row(
            "SELECT state, access_count FROM memory_states WHERE memory_id = ?1",
            params![&node.id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(state, "silent");
    assert_eq!(access_count, 42);
}

#[test]
fn test_portable_merge_import_keeps_composition_members_for_newer_local_memory() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    let target = create_test_storage_at(&target_dir, "target.db");

    let node = source
        .ingest(IngestInput {
            content: "Shared memory with historical composition".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["protocolgate".to_string()],
            ..Default::default()
        })
        .unwrap();
    source
        .save_composition(
            &CompositionEventRecord {
                id: "merge-composition-1".to_string(),
                created_at: Utc::now(),
                tool: "deep_reference".to_string(),
                mode: "bounty".to_string(),
                query: Some("historical composition".to_string()),
                query_hash: Some("sha256:historical".to_string()),
                confidence: Some(0.7),
                status: Some("resolved".to_string()),
                output_preview: Some("Historical composition survives merge".to_string()),
                metadata: serde_json::json!({}),
            },
            &[CompositionMemberRecord {
                event_id: "merge-composition-1".to_string(),
                memory_id: node.id.clone(),
                role: "primary".to_string(),
                rank: 0,
                trust: Some(0.8),
                score: Some(0.9),
                preview: Some("historical".to_string()),
                metadata: serde_json::json!({}),
            }],
            &[],
        )
        .unwrap();

    let archive = source.export_portable_archive().unwrap();
    target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap();

    let local_time = (Utc::now() + Duration::hours(1)).to_rfc3339();
    {
        let writer = target.writer.lock().unwrap();
        writer
            .execute(
                "DELETE FROM composition_members WHERE event_id = ?1",
                params!["merge-composition-1"],
            )
            .unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET content = ?1, updated_at = ?2 WHERE id = ?3",
                params!["Newer local content", &local_time, &node.id],
            )
            .unwrap();
    }

    target
        .import_portable_archive(&archive, PortableImportMode::Merge)
        .unwrap();

    let restored = target.get_node(&node.id).unwrap().unwrap();
    assert_eq!(restored.content, "Newer local content");
    let members = target
        .get_composition_members("merge-composition-1")
        .unwrap();
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].memory_id, node.id);
}

#[test]
fn test_portable_merge_import_applies_delete_tombstones() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    let target = create_test_storage_at(&target_dir, "target.db");

    let node = source
        .ingest(IngestInput {
            content: "Memory deleted on source".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let archive = source.export_portable_archive().unwrap();
    target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap();
    assert!(target.get_node(&node.id).unwrap().is_some());

    source.delete_node(&node.id).unwrap();
    let delete_archive = source.export_portable_archive().unwrap();
    let report = target
        .import_portable_archive(&delete_archive, PortableImportMode::Merge)
        .unwrap();

    assert!(report.rows_deleted >= 1);
    assert!(target.get_node(&node.id).unwrap().is_none());
}

#[test]
fn test_portable_merge_import_preserves_purge_tombstones() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    let target = create_test_storage_at(&target_dir, "target.db");

    let node = source
        .ingest(IngestInput {
            content: "Memory purged on source".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["sync".to_string()],
            ..Default::default()
        })
        .unwrap();
    source
        .save_composition(
            &CompositionEventRecord {
                id: "portable-purge-composition".to_string(),
                created_at: Utc::now(),
                tool: "deep_reference".to_string(),
                mode: "sync".to_string(),
                query: Some("portable purge preview".to_string()),
                query_hash: Some("fnv1a64:portable-purge".to_string()),
                confidence: Some(0.7),
                status: Some("resolved".to_string()),
                output_preview: None,
                metadata: serde_json::json!({}),
            },
            &[CompositionMemberRecord {
                event_id: "portable-purge-composition".to_string(),
                memory_id: node.id.clone(),
                role: "primary".to_string(),
                rank: 0,
                trust: Some(0.8),
                score: Some(0.8),
                preview: Some("Portable purge composition preview leak".to_string()),
                metadata: serde_json::json!({}),
            }],
            &[],
        )
        .unwrap();
    let archive = source.export_portable_archive().unwrap();
    target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap();
    assert!(target.get_node(&node.id).unwrap().is_some());
    assert_eq!(
        target
            .get_composition_members("portable-purge-composition")
            .unwrap()[0]
            .preview
            .as_deref(),
        Some("Portable purge composition preview leak")
    );
    {
        let writer = target.writer.lock().unwrap();
        writer
            .execute(
                "INSERT INTO memory_prs (
                        id, kind, status, title, subject_id, diff, signals, created_at
                     ) VALUES (?1, 'new_fact', 'pending', ?2, ?3, '{}', '[]', ?4)",
                params![
                    "portable-purge-review",
                    "remote cleanup review",
                    &node.id,
                    Utc::now().to_rfc3339(),
                ],
            )
            .unwrap();
    }

    source
        .purge_node(&node.id, Some("sync purge test"))
        .unwrap();
    let purge_archive = source.export_portable_archive().unwrap();
    assert!(
        !serde_json::to_string(&purge_archive)
            .unwrap()
            .contains("Portable purge composition preview leak"),
        "source portable archive should not retain purged composition previews"
    );
    let report = target
        .import_portable_archive(&purge_archive, PortableImportMode::Merge)
        .unwrap();

    assert!(report.rows_deleted >= 1);
    assert!(target.get_node(&node.id).unwrap().is_none());
    assert!(
        target
            .get_composition_members("portable-purge-composition")
            .unwrap()
            .is_empty(),
        "portable purge merge should delete composition evidence that references the target"
    );

    let writer = target.writer.lock().unwrap();
    let review_count: i64 = writer
        .query_row(
            "SELECT COUNT(*) FROM memory_prs WHERE id = 'portable-purge-review'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        review_count, 0,
        "portable purge merge must run the full non-FK evidence cleanup"
    );
    let tombstone_count: i64 = writer
        .query_row(
            "SELECT COUNT(*) FROM deletion_tombstones
                 WHERE memory_id = ?1 AND reason IS NULL AND tags = '[]'",
            params![SqliteMemoryStore::opaque_tombstone_marker(&node.id)],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(tombstone_count, 1);
}

#[test]
fn opaque_tombstone_rejects_a_later_node_archive_even_with_newer_timestamp() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    let target = create_test_storage_at(&target_dir, "target.db");

    let node = source
        .ingest(IngestInput {
            content: "must not resurrect after opaque tombstone".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let mut later_node_archive = source.export_portable_archive().unwrap();
    let node_table = later_node_archive
        .tables
        .iter_mut()
        .find(|table| table.name == "knowledge_nodes")
        .unwrap();
    let updated_at_index = node_table
        .columns
        .iter()
        .position(|column| column == "updated_at")
        .unwrap();
    match &mut node_table.rows[0][updated_at_index] {
        PortableValue::Text(value) => *value = (Utc::now() + Duration::hours(24)).to_rfc3339(),
        value => panic!("knowledge_nodes.updated_at must be text, got {value:?}"),
    }

    source.purge_node(&node.id, None).unwrap();
    let tombstone_archive = source.export_portable_archive().unwrap();
    target
        .import_portable_archive(&tombstone_archive, PortableImportMode::Merge)
        .unwrap();
    assert!(target.get_node(&node.id).unwrap().is_none());

    let report = target
        .import_portable_archive(&later_node_archive, PortableImportMode::Merge)
        .unwrap();
    assert!(target.get_node(&node.id).unwrap().is_none());
    assert!(report.rows_skipped >= 1);
}

#[test]
fn test_portable_merge_import_purge_wins_over_newer_local_edit() {
    let source_dir = tempdir().unwrap();
    let target_dir = tempdir().unwrap();
    let source = create_test_storage_at(&source_dir, "source.db");
    let target = create_test_storage_at(&target_dir, "target.db");

    let node = source
        .ingest(IngestInput {
            content: "Memory that will be purged on source".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["sync".to_string()],
            ..Default::default()
        })
        .unwrap();
    let archive = source.export_portable_archive().unwrap();
    target
        .import_portable_archive(&archive, PortableImportMode::EmptyOnly)
        .unwrap();

    let newer = (Utc::now() + Duration::hours(1)).to_rfc3339();
    {
        let writer = target.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET content = ?1, updated_at = ?2 WHERE id = ?3",
                params!["Newer local edit before purge arrives", newer, &node.id],
            )
            .unwrap();
    }

    source
        .purge_node(&node.id, Some("hard purge wins sync conflict"))
        .unwrap();
    let purge_archive = source.export_portable_archive().unwrap();
    let report = target
        .import_portable_archive(&purge_archive, PortableImportMode::Merge)
        .unwrap();

    assert!(report.rows_deleted >= 1);
    assert!(target.get_node(&node.id).unwrap().is_none());
}

#[test]
fn test_file_portable_sync_round_trips_between_devices() {
    let sync_dir = tempdir().unwrap();
    let first_dir = tempdir().unwrap();
    let second_dir = tempdir().unwrap();
    let sync_path = sync_dir.path().join("vestige-sync.json");
    let first = create_test_storage_at(&first_dir, "first.db");
    let second = create_test_storage_at(&second_dir, "second.db");

    let first_node = first
        .ingest(IngestInput {
            content: "First device memory".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["sync".to_string()],
            ..Default::default()
        })
        .unwrap();
    let first_push = first.sync_portable_archive_file(&sync_path).unwrap();
    assert!(!first_push.pulled);
    assert!(sync_path.exists());

    let second_node = second
        .ingest(IngestInput {
            content: "Second device memory".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["sync".to_string()],
            ..Default::default()
        })
        .unwrap();
    let second_sync = second.sync_portable_archive_file(&sync_path).unwrap();
    assert!(second_sync.pulled);
    assert!(second.get_node(&first_node.id).unwrap().is_some());

    let first_sync = first.sync_portable_archive_file(&sync_path).unwrap();
    assert!(first_sync.pulled);
    assert!(first.get_node(&second_node.id).unwrap().is_some());
    assert!(first_sync.pushed_rows >= 2);
}

#[test]
fn test_get_last_backup_timestamp_no_panic() {
    // Static method should not panic even if no backups exist
    let _ = Storage::get_last_backup_timestamp();
}

#[test]
fn test_keyword_search_with_include_types() {
    let storage = create_test_storage();

    // Ingest nodes of different types all containing the word "quantum"
    storage
        .ingest(IngestInput {
            content: "Quantum mechanics is fundamental to physics".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    storage
        .ingest(IngestInput {
            content: "Quantum computing uses qubits for calculation".to_string(),
            node_type: "concept".to_string(),
            ..Default::default()
        })
        .unwrap();
    storage
        .ingest(IngestInput {
            content: "Quantum entanglement was demonstrated in the lab".to_string(),
            node_type: "event".to_string(),
            ..Default::default()
        })
        .unwrap();

    // Search with include_types = ["fact"] — should only return the fact
    let include = vec!["fact".to_string()];
    let results = storage
        .hybrid_search_filtered("quantum", 10, 0.3, 0.7, Some(&include), None)
        .unwrap();

    assert!(!results.is_empty(), "should return at least one result");
    for r in &results {
        assert_eq!(
            r.node.node_type, "fact",
            "include_types=[fact] should only return facts, got: {}",
            r.node.node_type
        );
    }
}

#[test]
fn test_keyword_search_with_exclude_types() {
    let storage = create_test_storage();

    storage
        .ingest(IngestInput {
            content: "Photosynthesis converts sunlight to energy".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    storage
        .ingest(IngestInput {
            content: "Photosynthesis is a complex biochemical process".to_string(),
            node_type: "reflection".to_string(),
            ..Default::default()
        })
        .unwrap();

    // Search with exclude_types = ["reflection"] — should skip the reflection
    let exclude = vec!["reflection".to_string()];
    let results = storage
        .hybrid_search_filtered("photosynthesis", 10, 0.3, 0.7, None, Some(&exclude))
        .unwrap();

    assert!(!results.is_empty(), "should return at least one result");
    for r in &results {
        assert_ne!(
            r.node.node_type, "reflection",
            "exclude_types=[reflection] should not return reflections"
        );
    }
}

#[test]
fn test_include_types_takes_precedence_over_exclude() {
    let storage = create_test_storage();

    storage
        .ingest(IngestInput {
            content: "Gravity holds planets in orbit around stars".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    storage
        .ingest(IngestInput {
            content: "Gravity waves were first detected by LIGO".to_string(),
            node_type: "event".to_string(),
            ..Default::default()
        })
        .unwrap();

    // When both are provided, include_types wins
    let include = vec!["fact".to_string()];
    let exclude = vec!["fact".to_string()];
    let results = storage
        .hybrid_search_filtered("gravity", 10, 0.3, 0.7, Some(&include), Some(&exclude))
        .unwrap();

    // include_types takes precedence — facts should be returned
    assert!(!results.is_empty());
    for r in &results {
        assert_eq!(r.node.node_type, "fact");
    }
}

#[test]
fn test_type_filter_with_no_matches_returns_empty() {
    let storage = create_test_storage();

    storage
        .ingest(IngestInput {
            content: "DNA carries genetic information in cells".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    // Search for a type that doesn't exist among matches
    let include = vec!["person".to_string()];
    let results = storage
        .hybrid_search_filtered("DNA", 10, 0.3, 0.7, Some(&include), None)
        .unwrap();

    assert!(
        results.is_empty(),
        "filtering for a non-matching type should return empty results"
    );
}

#[test]
fn test_hybrid_search_backward_compat() {
    // Ensure the original hybrid_search (no type filters) still works
    let storage = create_test_storage();

    storage
        .ingest(IngestInput {
            content: "Neurons transmit electrical signals in the brain".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    let results = storage.hybrid_search("neurons", 10, 0.3, 0.7).unwrap();
    assert!(!results.is_empty());
    assert!(results[0].node.content.contains("Neurons"));
}

#[test]
fn test_concrete_search_literal_identifier_lands_first() {
    let storage = create_test_storage();

    storage
        .ingest(IngestInput {
            content: "General OpenAI API setup notes without the exact env var".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let target = storage
        .ingest(IngestInput {
            content: "Set OPENAI_API_KEY before running the release smoke tests".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    storage
        .ingest(IngestInput {
            content: "API keys should be handled carefully in shell profiles".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    let results = storage
        .concrete_search_filtered("OPENAI_API_KEY", 10, None, None)
        .unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].node.id, target.id);
    assert_eq!(results[0].match_type, MatchType::Keyword);
    assert!(results[0].semantic_score.is_none());
}

/// A memory that merely CITES an identifier must not outrank the memory that
/// IS it. Raw BM25 magnitude is unbounded while literal_match_score is capped
/// at 3.0, and both fed the same combined_score: measured on a 202-document
/// corpus, a note citing a UUID three times scored 27.5 against the exact
/// match's 3.0. That inverted the documented exact-lookup guarantee.
/// A corrupt FTS index must NOT strand the user's memories. `knowledge_fts`
/// is declared `content='knowledge_nodes'`, so it is derived state and is
/// always reconstructible. This reproduces the field failure: a store with
/// intact memories became unopenable because one fts5 blob was damaged.
#[test]
fn corrupt_fts_index_is_rebuilt_instead_of_bricking_the_store() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("vestige.db");

    // Seed a store and close it cleanly.
    {
        let storage = Storage::new(Some(path.clone())).expect("open");
        for i in 0..5 {
            storage
                .ingest(IngestInput {
                    content: format!("memory number {i} about deployment"),
                    node_type: "fact".to_string(),
                    ..Default::default()
                })
                .expect("ingest");
        }
    }

    // Corrupt the FTS index the way an interrupted rebuild does. The
    // damage is a FIXED byte pattern, not randomblob(): an unseeded random
    // block sometimes wrecks the segment so badly that quick_check itself
    // fails with SQLITE_NOMEM before the heal gate can classify the rows,
    // and the store (correctly) refuses loudly. That path is documented
    // shipped behavior, but it made this test flake in CI; a deterministic
    // pattern exercises the rebuild path every run.
    {
        let conn = Connection::open(&path).expect("raw open");
        let pattern = "A5".repeat(200);
        conn.execute_batch(&format!(
            "UPDATE knowledge_fts_data SET block = x'{pattern}' \
                 WHERE id = (SELECT id FROM knowledge_fts_data WHERE id > 1 LIMIT 1);"
        ))
        .expect("corrupt");
        let corrupt = conn
            .execute_batch("INSERT INTO knowledge_fts(knowledge_fts) VALUES('integrity-check');")
            .is_err();
        assert!(corrupt, "the fixture must actually corrupt the index");
    }

    // Reopening must succeed by rebuilding, not fail.
    let storage = Storage::new(Some(path.clone()))
        .expect("a corrupt DERIVED index must not prevent opening the store");
    let all = storage.get_all_nodes(100, 0).expect("list nodes");
    assert_eq!(all.len(), 5, "every memory must survive the rebuild");

    // And the rebuilt index must actually be usable again.
    let hits = storage
        .concrete_search_filtered("deployment", 10, None, None)
        .expect("keyword search after rebuild");
    assert!(
        !hits.is_empty(),
        "the rebuilt index must find the seeded memories"
    );
}

#[test]
fn test_concrete_search_exact_match_beats_a_doc_that_only_cites_it() {
    let storage = create_test_storage();

    // Filler so BM25's IDF term is meaningful rather than degenerate.
    for i in 0..40 {
        storage
            .ingest(IngestInput {
                content: format!("Routine note {i} about deployment pipelines and review"),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
    }

    let needle = "PAYMENTS_REDIS_URL";
    let target = storage
        .ingest(IngestInput {
            content: needle.to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    // Cites the identifier repeatedly -> large BM25 magnitude.
    storage
        .ingest(IngestInput {
            content: format!(
                "See {needle} for the rollout; {needle} was rotated in review, and \
                     {needle} supersedes the older connection note entirely"
            ),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    let results = storage
        .concrete_search_filtered(needle, 10, None, None)
        .unwrap();
    assert!(!results.is_empty(), "exact lookup must return something");
    assert_eq!(
        results[0].node.id, target.id,
        "the memory that IS the identifier must rank first, not the one citing it"
    );
}

#[test]
fn test_purge_scrubs_insight_json_orphans_children_and_writes_tombstone() {
    let storage = create_test_storage();
    let doomed = storage
        .ingest(IngestInput {
            content: "Sensitive purge target memory".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["sensitive".to_string()],
            ..Default::default()
        })
        .unwrap();
    let other_a = storage
        .ingest(IngestInput {
            content: "Other source memory A".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let other_b = storage
        .ingest(IngestInput {
            content: "Other source memory B".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let child = storage
        .ingest(IngestInput {
            content: "Temporal summary child".to_string(),
            node_type: "summary".to_string(),
            ..Default::default()
        })
        .unwrap();

    {
        let writer = storage.writer.lock().unwrap();
        writer
                .execute(
                    "INSERT INTO memory_connections (
                        source_id, target_id, strength, link_type, created_at, last_activated, activation_count
                     ) VALUES (?1, ?2, 0.9, 'semantic', ?3, ?3, 0)",
                    params![doomed.id, other_a.id, Utc::now().to_rfc3339()],
                )
                .unwrap();
        writer
                .execute(
                    "INSERT INTO insights (
                        id, insight, source_memories, confidence, novelty_score, insight_type, generated_at
                     ) VALUES (?1, 'drop me', ?2, 0.9, 0.2, 'synthesis', ?3)",
                    params![
                        Uuid::new_v4().to_string(),
                        serde_json::to_string(&vec![doomed.id.clone(), other_a.id.clone()]).unwrap(),
                        Utc::now().to_rfc3339()
                    ],
                )
                .unwrap();
        writer
                .execute(
                    "INSERT INTO insights (
                        id, insight, source_memories, confidence, novelty_score, insight_type, generated_at
                     ) VALUES (?1, 'rewrite me', ?2, 0.9, 0.2, 'synthesis', ?3)",
                    params![
                        Uuid::new_v4().to_string(),
                        serde_json::to_string(&vec![
                            doomed.id.clone(),
                            other_a.id.clone(),
                            other_b.id.clone()
                        ])
                        .unwrap(),
                        Utc::now().to_rfc3339()
                    ],
                )
                .unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET summary_parent_id = ?1 WHERE id = ?2",
                params![doomed.id, child.id],
            )
            .unwrap();
        writer
            .execute(
                "INSERT INTO memory_prs (
                        id, kind, status, title, subject_id, diff, signals, created_at
                     ) VALUES (?1, 'new_fact', 'pending', ?2, ?3, ?4, '[]', ?5)",
                params![
                    "purge-review-leak",
                    "Sensitive purge target memory review preview",
                    doomed.id,
                    serde_json::json!({
                        "contentPreview": "Sensitive purge target memory"
                    })
                    .to_string(),
                    Utc::now().to_rfc3339(),
                ],
            )
            .unwrap();
    }

    storage
        .save_composition(
            &CompositionEventRecord {
                id: "purge-composition-preview-test".to_string(),
                created_at: Utc::now(),
                tool: "deep_reference".to_string(),
                mode: "audit".to_string(),
                query: Some("purge preview leak".to_string()),
                query_hash: Some("fnv1a64:purge".to_string()),
                confidence: Some(0.7),
                status: Some("resolved".to_string()),
                output_preview: None,
                metadata: serde_json::json!({}),
            },
            &[CompositionMemberRecord {
                event_id: "purge-composition-preview-test".to_string(),
                memory_id: doomed.id.clone(),
                role: "primary".to_string(),
                rank: 0,
                trust: Some(0.8),
                score: Some(0.9),
                preview: Some("Sensitive purge target memory preview leak".to_string()),
                metadata: serde_json::json!({}),
            }],
            &[],
        )
        .unwrap();

    let report = storage
        .purge_node(&doomed.id, Some("user requested hard purge"))
        .unwrap();
    assert!(report.deleted);
    assert_eq!(report.edges_pruned, 1);
    assert_eq!(report.insights_deleted, 1);
    assert_eq!(report.insights_rewritten, 1);
    assert_eq!(report.children_orphaned, 1);
    assert!(storage.get_node(&doomed.id).unwrap().is_none());

    let writer = storage.writer.lock().unwrap();
    let remaining_refs: Vec<String> = writer
        .prepare("SELECT source_memories FROM insights")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .filter_map(|row| row.ok())
        .collect();
    assert_eq!(remaining_refs.len(), 1);
    assert!(!remaining_refs[0].contains(&doomed.id));

    let child_parent: Option<String> = writer
        .query_row(
            "SELECT summary_parent_id FROM knowledge_nodes WHERE id = ?1",
            params![child.id],
            |row| row.get(0),
        )
        .unwrap();
    assert!(child_parent.is_none());

    let tombstone_count: i64 = writer
        .query_row(
            "SELECT COUNT(*) FROM deletion_tombstones
                 WHERE memory_id = ?1 AND reason IS NULL AND tags = '[]'",
            params![SqliteMemoryStore::opaque_tombstone_marker(&doomed.id)],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(tombstone_count, 1);
    let sync_tombstone_count: i64 = writer
        .query_row(
            "SELECT COUNT(*) FROM sync_tombstones
                 WHERE table_name = 'knowledge_nodes' AND row_id = ?1 AND reason IS NULL",
            params![SqliteMemoryStore::opaque_tombstone_marker(&doomed.id)],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(sync_tombstone_count, 1);

    let members = storage
        .get_composition_members("purge-composition-preview-test")
        .unwrap();
    assert!(
        members.is_empty(),
        "purge should remove composition evidence that references the target"
    );
    let review_count: i64 = writer
        .query_row(
            "SELECT COUNT(*) FROM memory_prs WHERE id = 'purge-review-leak'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        review_count, 0,
        "purge should remove linked review evidence"
    );
    let archive_json = serde_json::to_string(&storage.export_portable_archive().unwrap()).unwrap();
    assert!(
        !archive_json.contains("Sensitive purge target memory preview leak"),
        "portable archive should not retain purged memory content through composition previews"
    );
    assert!(
        !archive_json.contains(&doomed.id),
        "portable archive should not retain the purged memory's raw identifier"
    );
    assert!(
        !archive_json.contains("user requested hard purge"),
        "portable archive should not retain caller-controlled purge rationale"
    );

    let has_content_column: i64 = writer
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('deletion_tombstones') WHERE name = 'content'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(has_content_column, 0);
}

/// Purge is an erasure guarantee, so a referencing row the scrub cannot
/// read has to abort the whole purge rather than be skipped. Before this,
/// the three reference sweeps in `purge_node_in_transaction` used
/// `filter_map(|row| row.ok())`, so an unreadable row silently kept its
/// reference to a memory the caller was told had been erased.
#[test]
fn purge_fails_closed_when_a_referencing_row_cannot_be_read() {
    let storage = create_test_storage();
    let doomed = storage
        .ingest(IngestInput {
            content: "Purge target with an unreadable referrer".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    // `insights` is a non-STRICT table, so a BLOB survives in the TEXT
    // primary key. rusqlite then refuses to read it as a String, which is
    // exactly the "row we cannot read" the sweep used to swallow.
    {
        let writer = storage.writer.lock().unwrap();
        writer
                .execute(
                    "INSERT INTO insights (
                        id, insight, source_memories, confidence, novelty_score, insight_type, generated_at
                     ) VALUES (?1, 'unreadable id', ?2, 0.9, 0.2, 'synthesis', ?3)",
                    params![
                        vec![0xF0_u8, 0x9F, 0x92, 0xA9, 0xFF, 0xFE],
                        serde_json::to_string(&vec![doomed.id.clone()]).unwrap(),
                        Utc::now().to_rfc3339()
                    ],
                )
                .unwrap();
    }

    // The purge must refuse, not report success.
    let error = storage.purge_node(&doomed.id, None).unwrap_err();
    let rendered = error.to_string().to_lowercase();
    assert!(
        rendered.contains("type") || rendered.contains("column") || rendered.contains("convert"),
        "expected a read failure to surface, got: {rendered}"
    );

    // And the transaction rolled back: nothing was half-erased.
    assert!(
        storage.get_node(&doomed.id).unwrap().is_some(),
        "a refused purge must leave the memory intact, not partially scrubbed"
    );
}

// ========================================================================
// Merge / Supersede controls (Phase 3 — v2.1.25)
//
// These exercise the full lifecycle without the live embedding model by
// seeding the `node_embeddings` table directly with the ACTIVE model name,
// so `get_all_embeddings` / `get_node_embedding` accept them.
// ========================================================================

// =========================================================================
// Phase 1 trait-method unit tests
// =========================================================================
use crate::storage::memory_store::{
    MemoryEdge, MemoryRecord, MemoryStore, MemoryStoreError, ModelSignature, SchedulingState,
};

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

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Runtime::new().unwrap()
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

// ========================================================================
// Auto-consolidation merge: opt-out gate + protected-pin exclusion (#142)
//
// These exercise `auto_dedup_consolidation` directly — the unattended,
// no-audit pass the 6h background consolidation cycle runs. seed_node/
// axis_vector give deterministic same-axis clusters (cosine ~1.0 >> the 0.85
// threshold); set_retention pins down which node wins the keeper tiebreak.
// ========================================================================

// --- A. Default (flag unset): NOTHING merges, nothing is deleted ---------
// v2.6.0 flipped the #142 opt-out into an opt-in: unattended destruction
// of user memories must be asked for, never inherited.

// --- B. Flag off suppresses the merge (parametrized) ---------------------

// --- B (cont). A malformed value fails CLOSED: no destruction on a typo --

// --- C(a). Protected would-be keeper: untouched; others merge -----------

// --- C(b) / Regression (#142): protected weak member is never absorbed --

// --- C(c). Two protected near-dups: neither merges ----------------------

// --- C(d). Protected + a single unprotected near-dup: no merge ----------

// --- D. Liveness: protected + two unprotected → the two merge -----------

#[test]
fn test_set_protected_unknown_node_errors() {
    let storage = create_test_storage();
    assert!(storage.set_protected("does-not-exist", true).is_err());
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
        // (get_store_stats) instead of the inherent sync get_stats() on
        // SqliteMemoryStore.
        let dyn_s: &dyn MemoryStore = &s;
        dyn_s.register_model(&sig).await.unwrap();
        let stats = dyn_s.get_store_stats().await.unwrap();
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

// Seed a node's stability directly via the scheduling seam so the +365 cap
// in promote_memory_backfill is actually exercised (a freshly ingested node
// has low stability where the *1.5 multiply, not the additive ceiling, wins).
fn seed_stability(s: &Storage, id: &str, stability: f64) {
    use crate::storage::memory_store::{MemoryStoreSend, SchedulingState};
    rt().block_on(async {
        let state = SchedulingState {
            memory_id: uuid::Uuid::parse_str(id).unwrap(),
            stability,
            difficulty: 0.4,
            retrievability: 0.8,
            last_review: Some(chrono::Utc::now()),
            next_review: Some(chrono::Utc::now() + chrono::Duration::days(7)),
            reps: 3,
            lapses: 0,
        };
        MemoryStoreSend::update_scheduling(s, &state).await.unwrap();
    });
}

#[test]
fn promote_memory_backfill_caps_stability_at_plus_365() {
    // Above the crossover (stability=730) the additive +365 ceiling must win
    // over the *1.5 multiply, so repeated backfill promotions cannot inflate
    // stability without bound. This is the bound issue #103 asked us to apply.
    let s = create_test_storage();
    let node = s
        .ingest(IngestInput {
            content: "high-stability cause memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    seed_stability(&s, &node.id, 1000.0);

    let promoted = s.promote_memory_backfill(&node.id).unwrap();
    // 1000 * 1.5 = 1500 (uncapped) vs 1000 + 365 = 1365 (capped). Cap wins.
    assert!(
        (promoted.stability - 1365.0).abs() < 1e-6,
        "expected additive +365 cap (1365.0), got {} (uncapped would be 1500.0)",
        promoted.stability
    );
}

#[test]
fn promote_memory_backfill_uses_multiply_below_crossover() {
    // Below the crossover the *1.5 multiply wins (the cap never binds), so
    // backfill promotion strength is unchanged from the old promote_memory.
    let s = create_test_storage();
    let node = s
        .ingest(IngestInput {
            content: "low-stability cause memory".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    seed_stability(&s, &node.id, 10.0);

    let promoted = s.promote_memory_backfill(&node.id).unwrap();
    // 10 * 1.5 = 15 (multiply) vs 10 + 365 = 375 (cap). Multiply wins.
    assert!(
        (promoted.stability - 15.0).abs() < 1e-6,
        "expected *1.5 multiply (15.0) below crossover, got {}",
        promoted.stability
    );
}

#[test]
fn suppression_cascade_is_idempotent_and_reverses_with_seed() {
    let store = create_test_storage();
    let seed = store
        .ingest(IngestInput {
            content: "cascade seed".into(),
            node_type: "fact".into(),
            ..Default::default()
        })
        .unwrap();
    let neighbor = store
        .ingest(IngestInput {
            content: "cascade neighbor".into(),
            node_type: "fact".into(),
            ..Default::default()
        })
        .unwrap();
    store
        .save_connection(&ConnectionRecord {
            source_id: seed.id.clone(),
            target_id: neighbor.id.clone(),
            strength: 1.0,
            link_type: "semantic".into(),
            created_at: Utc::now(),
            last_activated: Utc::now(),
            activation_count: 1,
        })
        .unwrap();
    let before =
        SqliteMemoryStore::suppression_state_on(&store.reader.lock().unwrap(), &neighbor.id)
            .unwrap();
    store.suppress_memory(&seed.id).unwrap();
    assert_eq!(store.apply_rac1_cascade(&seed.id).unwrap(), 1);
    assert_eq!(store.apply_rac1_cascade(&seed.id).unwrap(), 0);
    store.reverse_suppression(&seed.id, 24).unwrap();
    assert_eq!(
        SqliteMemoryStore::suppression_state_on(&store.reader.lock().unwrap(), &neighbor.id)
            .unwrap(),
        before
    );
    assert_eq!(store.apply_rac1_cascade(&seed.id).unwrap(), 0);
    store.suppress_memory(&seed.id).unwrap();
    assert_eq!(store.apply_rac1_cascade(&seed.id).unwrap(), 1);
    store
        .writer
        .lock()
        .unwrap()
        .execute(
            "UPDATE knowledge_nodes SET stability=99 WHERE id=?1",
            params![neighbor.id],
        )
        .unwrap();
    assert!(
        store
            .reverse_suppression(&seed.id, 24)
            .unwrap_err()
            .to_string()
            .contains("neighbor")
    );
    assert_eq!(
        store.get_node(&seed.id).unwrap().unwrap().suppression_count,
        1
    );
    assert_eq!(
        store.get_node(&neighbor.id).unwrap().unwrap().stability,
        99.0
    );
}

#[test]
fn suppression_snapshot_restores_floor_and_stack_atomically() {
    let s = create_test_storage();
    let node = s
        .ingest(IngestInput {
            content: "suppression fixture".into(),
            node_type: "fact".into(),
            ..Default::default()
        })
        .unwrap();
    s.writer
        .lock()
        .unwrap()
        .execute(
            "UPDATE knowledge_nodes SET retrieval_strength = 0.06,
            retention_strength = 0.07, stability = 0.123 WHERE id = ?1",
            params![node.id],
        )
        .unwrap();
    let original =
        SqliteMemoryStore::suppression_state_on(&s.reader.lock().unwrap(), &node.id).unwrap();
    s.suppress_memory(&node.id).unwrap();
    let first =
        SqliteMemoryStore::suppression_state_on(&s.reader.lock().unwrap(), &node.id).unwrap();
    s.suppress_memory(&node.id).unwrap();
    let second =
        SqliteMemoryStore::suppression_state_on(&s.reader.lock().unwrap(), &node.id).unwrap();
    s.writer
        .lock()
        .unwrap()
        .execute_batch(
            "CREATE TEMP TRIGGER fail_reverse BEFORE UPDATE ON suppression_operations
            BEGIN SELECT RAISE(ABORT, 'injected journal failure'); END;",
        )
        .unwrap();
    assert!(s.reverse_suppression(&node.id, 24).is_err());
    assert_eq!(
        SqliteMemoryStore::suppression_state_on(&s.reader.lock().unwrap(), &node.id).unwrap(),
        second
    );
    s.writer
        .lock()
        .unwrap()
        .execute_batch("DROP TRIGGER fail_reverse")
        .unwrap();
    s.reverse_suppression(&node.id, 24).unwrap();
    assert_eq!(
        SqliteMemoryStore::suppression_state_on(&s.reader.lock().unwrap(), &node.id).unwrap(),
        first
    );
    s.reverse_suppression(&node.id, 24).unwrap();
    assert_eq!(
        SqliteMemoryStore::suppression_state_on(&s.reader.lock().unwrap(), &node.id).unwrap(),
        original
    );
    assert!(s.reverse_suppression(&node.id, 24).is_err());
    assert!(s.reverse_suppression(&node.id, i64::MAX).is_err());
}

#[test]
fn suppression_snapshot_rejects_conflict_expiry_and_legacy() {
    let s = create_test_storage();
    let node = s
        .ingest(IngestInput {
            content: "suppression conflicts".into(),
            node_type: "fact".into(),
            ..Default::default()
        })
        .unwrap();
    s.suppress_memory(&node.id).unwrap();
    s.writer
        .lock()
        .unwrap()
        .execute(
            "UPDATE knowledge_nodes SET stability = 99 WHERE id = ?1",
            params![node.id],
        )
        .unwrap();
    assert!(
        s.reverse_suppression(&node.id, 24)
            .unwrap_err()
            .to_string()
            .contains("conflicts")
    );
    assert_eq!(s.get_node(&node.id).unwrap().unwrap().stability, 99.0);
    s.writer.lock().unwrap().execute("UPDATE suppression_operations SET created_at = '2000-01-01T00:00:00Z' WHERE node_id = ?1", params![node.id]).unwrap();
    assert!(
        s.reverse_suppression(&node.id, 24)
            .unwrap_err()
            .to_string()
            .contains("expired")
    );
    s.writer
        .lock()
        .unwrap()
        .execute(
            "DELETE FROM suppression_operations WHERE node_id = ?1",
            params![node.id],
        )
        .unwrap();
    assert!(
        s.reverse_suppression(&node.id, 24)
            .unwrap_err()
            .to_string()
            .contains("legacy")
    );
}

#[test]
fn suppress_then_reverse_restores_fsrs_state() {
    // reverse_suppression must be a TRUE inverse of suppress_memory. Suppress
    // applies stability*0.4, retrieval-0.35, retention-0.20; reverse now undoes
    // exactly that (stability/0.4, retrieval+0.35, retention+0.20). Previously
    // reverse used non-inverse deltas and left stability permanently halved.
    let s = create_test_storage();
    let node = s
        .ingest(IngestInput {
            content: "a memory to suppress then un-suppress".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    // Seed above the 0.05 floor so the forward pass never clips (making the
    // round-trip exactly recoverable).
    seed_stability(&s, &node.id, 20.0);
    let before = s.get_node(&node.id).unwrap().unwrap();

    s.suppress_memory(&node.id).unwrap();
    let suppressed = s.get_node(&node.id).unwrap().unwrap();
    assert!(
        (suppressed.stability - before.stability * 0.4).abs() < 1e-6,
        "suppress must multiply stability by 0.4"
    );

    let reversed = s.reverse_suppression(&node.id, 24).unwrap();
    // stability: 20 * 0.4 / 0.4 = 20 (fully restored, not 0.5x)
    assert!(
        (reversed.stability - before.stability).abs() < 1e-6,
        "reverse must restore stability to {} (got {})",
        before.stability,
        reversed.stability
    );
    assert!(
        (reversed.retrieval_strength - before.retrieval_strength).abs() < 1e-6,
        "reverse must restore retrieval_strength"
    );
    assert!(
        (reversed.retention_strength - before.retention_strength).abs() < 1e-6,
        "reverse must restore retention_strength"
    );
}

#[test]
fn backfill_autofire_gate_defaults_on_and_reads_opt_out() {
    // v2.2.1 opt-out semantics: unset => ON (preserves shipped v2.2.0
    // behavior); explicit 0/false/off/no => OFF; anything else => ON.
    fn parse(v: Option<&str>) -> bool {
        v.map(|v| {
            let v = v.trim();
            !(v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("off")
                || v.eq_ignore_ascii_case("no")
                || v == "0")
        })
        .unwrap_or(true)
    }
    assert!(parse(None), "unset must default ON");
    assert!(parse(Some("1")), "1 is ON");
    assert!(parse(Some("true")), "true is ON");
    assert!(parse(Some("anything")), "unrecognized is ON");
    assert!(!parse(Some("0")), "0 is OFF");
    assert!(!parse(Some("false")), "false is OFF");
    assert!(!parse(Some("OFF")), "OFF (case-insensitive) is OFF");
    assert!(!parse(Some(" no ")), "whitespace-padded no is OFF (trim)");
}

// =====================================================================
// Smart-ingest bitemporal validity gates (issue #156)
// =====================================================================

#[test]
fn update_node_validity_merges_bounds_and_validates_the_effective_window() {
    let storage = create_test_storage();
    let node = storage
        .ingest(IngestInput {
            content: "validity merge fixture".to_string(),
            ..Default::default()
        })
        .unwrap();
    let from = Utc::now() - Duration::days(10);
    let until = Utc::now() + Duration::days(10);
    storage
        .update_node_validity(&node.id, Some(from), Some(until))
        .unwrap();

    // Updating only valid_from must not clear the stored valid_until.
    let new_from = Utc::now() - Duration::days(5);
    storage
        .update_node_validity(&node.id, Some(new_from), None)
        .unwrap();
    let stored = storage.get_node(&node.id).unwrap().unwrap();
    assert_eq!(
        stored.valid_from.map(|value| value.to_rfc3339()),
        Some(new_from.to_rfc3339())
    );
    assert_eq!(
        stored.valid_until.map(|value| value.to_rfc3339()),
        Some(until.to_rfc3339())
    );

    // Updating only valid_until must not clear the stored valid_from.
    let new_until = Utc::now() + Duration::days(20);
    storage
        .update_node_validity(&node.id, None, Some(new_until))
        .unwrap();
    let stored = storage.get_node(&node.id).unwrap().unwrap();
    assert_eq!(
        stored.valid_from.map(|value| value.to_rfc3339()),
        Some(new_from.to_rfc3339())
    );
    assert_eq!(
        stored.valid_until.map(|value| value.to_rfc3339()),
        Some(new_until.to_rfc3339())
    );

    // The EFFECTIVE post-merge window is validated: a valid_from at or
    // beyond the stored valid_until is rejected and nothing changes.
    let error = storage
        .update_node_validity(&node.id, Some(Utc::now() + Duration::days(30)), None)
        .unwrap_err();
    assert!(matches!(error, StorageError::InvalidTimestamp(_)));
    let stored = storage.get_node(&node.id).unwrap().unwrap();
    assert_eq!(
        stored.valid_from.map(|value| value.to_rfc3339()),
        Some(new_from.to_rfc3339())
    );
    assert_eq!(
        stored.valid_until.map(|value| value.to_rfc3339()),
        Some(new_until.to_rfc3339())
    );

    // Both bounds supplied together are still validated up front.
    let error = storage
        .update_node_validity(&node.id, Some(until), Some(from))
        .unwrap_err();
    assert!(matches!(error, StorageError::InvalidTimestamp(_)));
}

// ============================================================================
// #226 NARRATIVE EDGES — co-retrieval links with anti-contamination bounds
// ============================================================================

fn narrative_seed(store: &Storage, n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            store
                .ingest(IngestInput {
                    content: format!("narrative fixture {i}: co-retrieval check"),
                    node_type: "fact".to_string(),
                    ..Default::default()
                })
                .unwrap()
                .id
                .to_string()
        })
        .collect()
}

fn narrative_edges(store: &Storage) -> Vec<(String, String, f64, i64)> {
    let reader = store.reader.lock().unwrap();
    let mut stmt = reader
        .prepare(
            "SELECT source_id, target_id, strength, activation_count
             FROM memory_connections WHERE link_type = 'narrative'",
        )
        .unwrap();
    stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, f64>(2)?,
            row.get::<_, i64>(3)?,
        ))
    })
    .unwrap()
    .filter_map(|r| r.ok())
    .collect()
}

#[test]
fn one_retrieval_links_the_top_pairs_only() {
    let store = create_test_storage();
    let ids = narrative_seed(&store, 5);
    let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    store.record_batch_retrieval(&refs).unwrap();
    let edges = narrative_edges(&store);
    assert!(
        edges.len() <= 3,
        "anti-contamination cap: at most 3 pairs per retrieval, got {}",
        edges.len()
    );
    for (_, _, strength, count) in &edges {
        assert!(
            *strength <= 0.2 + 1e-9,
            "first co-retrieval is a hairline, got {strength}"
        );
        assert_eq!(*count, 1);
    }
}

#[test]
fn repeated_co_retrieval_grows_strength_to_a_cap() {
    let store = create_test_storage();
    let ids = narrative_seed(&store, 2);
    let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    for _ in 0..10 {
        store.record_batch_retrieval(&refs).unwrap();
    }
    let edges = narrative_edges(&store);
    assert_eq!(edges.len(), 1);
    let (_, _, strength, count) = &edges[0];
    assert_eq!(*count, 10);
    assert!(
        (*strength - 0.6).abs() < 1e-9,
        "capped at 0.6, got {strength}"
    );
}

#[test]
fn single_memory_retrievals_link_nothing() {
    let store = create_test_storage();
    let ids = narrative_seed(&store, 2);
    store.record_batch_retrieval(&[ids[0].as_str()]).unwrap();
    store.record_batch_retrieval(&[ids[1].as_str()]).unwrap();
    assert!(narrative_edges(&store).is_empty());
}

// ========================================================================
// RECONSOLIDATION LINKAGE — conflict vs a live labile window becomes a
// reviewable merge plan with explicit verdicts (see
// crate::advanced::reconsolidation for the neuroscience).
// ========================================================================

// ===================== Connector tag freshness ==================
// A synced issue's tags are derived (state:*, label:*). The UPDATE path used
// to write only content/envelope columns, so a label or state change upstream
// left the ORIGINAL tags on the node forever — `tag_prefix=state:open` kept
// matching a closed issue. The Unchanged path must also converge tags, or a
// tag-normalization change would only land one content-change late.

#[test]
fn upsert_by_source_updates_tags_when_content_changes() {
    let store = create_test_storage();
    let mut input = source_input("5", "issue five", "h5");
    input.tags = vec!["github".to_string(), "state:open".to_string()];
    store.upsert_by_source(input).unwrap();

    let mut edited = source_input("5", "issue five CLOSED", "h5b");
    edited.tags = vec!["github".to_string(), "state:closed".to_string()];
    let res = store.upsert_by_source(edited).unwrap();
    assert_eq!(res.outcome, SourceUpsertOutcome::Updated);

    let node = store.get_node(&res.node_id).unwrap().unwrap();
    assert_eq!(
        node.tags,
        vec!["github".to_string(), "state:closed".to_string()],
        "tags must track the upstream state change"
    );
}

#[test]
fn due_for_review_node_ids_returns_only_past_due_rows() {
    let storage = create_test_storage();
    let node = storage
        .ingest(crate::IngestInput {
            content: "review fixture scheduling".into(),
            ..Default::default()
        })
        .unwrap();
    // Future-dated review: not due, must not appear.
    let future = (Utc::now() + chrono::Duration::days(30)).to_rfc3339();
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET next_review = ?1 WHERE id = ?2",
                params![&future, &node.id],
            )
            .unwrap();
    }
    assert!(
        storage.due_for_review_node_ids(10).unwrap().is_empty(),
        "a future next_review must not be reported as due"
    );
    // Backdate next_review into the past: now it is due.
    let past = (Utc::now() - chrono::Duration::days(1)).to_rfc3339();
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "UPDATE knowledge_nodes SET next_review = ?1 WHERE id = ?2",
                params![&past, &node.id],
            )
            .unwrap();
    }
    let due = storage.due_for_review_node_ids(10).unwrap();
    assert_eq!(due, vec![node.id]);
    // The limit is honored.
    assert_eq!(storage.due_for_review_node_ids(0).unwrap().len(), 0);
}

#[test]
fn state_distribution_covers_every_row_with_sql_thresholds() {
    let storage = create_test_storage();
    // Fresh rows: full strengths -> active.
    for index in 0..3 {
        storage
            .ingest(crate::IngestInput {
                content: format!("active fixture {index}"),
                ..Default::default()
            })
            .unwrap();
    }
    // NULL retention rows must count as decayed (0), not silently healthy:
    // accessibility = 0.5*0 + 0.3*1 + 0.2*1 = 0.5 -> dormant.
    let now = Utc::now().to_rfc3339();
    {
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "INSERT INTO knowledge_nodes
                        (id, content, node_type, created_at, updated_at, last_accessed,
                         tags, scope, retention_strength)
                     VALUES ('null-ret', 'hand-edited', 'fact', ?1, ?1, ?1, NULL, 'user', NULL)",
                params![&now],
            )
            .unwrap();
    }
    let (active, dormant, silent, unavailable) = storage.state_distribution().unwrap();
    assert_eq!(active, 3, "fresh rows are fully accessible");
    assert_eq!(dormant, 1, "NULL retention lands dormant, not active");
    assert_eq!(silent + unavailable, 0);
    // Every row is accounted for exactly once.
    assert_eq!(active + dormant + silent + unavailable, 4);
}

#[test]
fn retention_distribution_reports_out_of_range_values_explicitly() {
    let storage = create_test_storage();
    let now = Utc::now().to_rfc3339();
    {
        let writer = storage.writer.lock().unwrap();
        for (id, retention) in [
            ("null-row", None::<f64>),
            ("negative-row", Some(-0.5)),
            ("above-row", Some(1.5)),
            ("normal-row", Some(0.5)),
        ] {
            writer
                .execute(
                    "INSERT INTO knowledge_nodes
                            (id, content, node_type, created_at, updated_at, last_accessed,
                             tags, scope, retention_strength)
                         VALUES (?1, 'bucket fixture', 'fact', ?2, ?2, ?2, NULL, 'user', ?3)",
                    rusqlite::params![id, &now, retention],
                )
                .unwrap();
        }
    }
    let distribution = storage.get_retention_distribution().unwrap();
    let bucket_of = |name: &str| {
        distribution
            .iter()
            .find(|(bucket, _)| bucket == name)
            .map(|(_, count)| *count)
            .unwrap_or(0)
    };
    assert_eq!(
        bucket_of("unknown"),
        1,
        "NULL retention gets its own bucket"
    );
    assert_eq!(bucket_of("below0"), 1);
    assert_eq!(bucket_of("above100%"), 1);
    assert_eq!(bucket_of("40-60%"), 1);
    assert_eq!(
        bucket_of("80-100%") + bucket_of("0-20%") + bucket_of("20-40%") + bucket_of("60-80%"),
        0,
        "out-of-range rows must no longer inflate the in-range buckets"
    );
}
