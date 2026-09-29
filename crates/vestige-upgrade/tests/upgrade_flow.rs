//! In-process v3.1.1 upgrades. The importer lives in this crate; the main
//! binaries only spawn the `vestige-upgrade` executable.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use vestige_upgrade::{
    LOG_DIR_NAME, UPGRADE_LOG_NAME, UpgradeOptions, UpgradeStatus, V311_RELEASE, staging_directory,
    upgrade_if_needed, upgrade_with, verify_migrated_log,
};

fn fixture_db() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../strata-migrate/tests/fixtures/v3.1.1-sample.sqlite")
}

fn sha256_file(path: &Path) -> String {
    let mut file = fs::File::open(path).unwrap();
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn plant(dir: &Path) -> PathBuf {
    let db = dir.join("vestige.db");
    fs::copy(fixture_db(), &db).unwrap();
    db
}

fn assert_failure_message(err: &str, log_path: &Path) {
    assert!(err.contains("upgrade failed"), "{err}");
    assert!(err.contains("untouched"), "{err}");
    assert!(err.contains("v3.1.1"), "{err}");
    assert!(err.contains(V311_RELEASE), "{err}");
    assert!(err.contains(&log_path.display().to_string()), "{err}");
    let logged = fs::read_to_string(log_path).unwrap();
    assert!(logged.contains("upgrade failed"), "{logged}");
    assert!(logged.contains(V311_RELEASE), "{logged}");
}

fn snapshot(log_dir: &Path) -> strata_migrate::snapshot::Snapshot {
    let log = strata::StrataLog::open(log_dir).unwrap();
    strata_migrate::read_snapshot(&log).unwrap()
}

fn knowledge_ids(db: &Path) -> Vec<String> {
    let conn = rusqlite::Connection::open_with_flags(
        db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .unwrap();
    let mut stmt = conn
        .prepare("SELECT id FROM knowledge_nodes ORDER BY id")
        .unwrap();
    stmt.query_map([], |row| row.get(0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect()
}

fn assert_fixture_landed(db: &Path, log_dir: &Path) {
    let verified = verify_migrated_log(log_dir).unwrap();
    assert!(verified.ok, "{verified:?}");
    let snap = snapshot(log_dir);
    let ids = knowledge_ids(db);
    assert!(ids.len() >= 4, "fixture lost its memories");
    for id in &ids {
        let node = snap
            .nodes
            .iter()
            .find(|node| &node.legacy_id == id)
            .unwrap_or_else(|| panic!("dropped memory {id}"));
        if id.starts_with("11111111") {
            let stability = node
                .legacy
                .iter()
                .find(|(key, _)| key == "fsrs_cards.stability")
                .map(|(_, value)| value.as_str())
                .unwrap_or("");
            let parsed: f64 = stability.parse().unwrap_or(f64::NAN);
            assert!((parsed - 12.25).abs() < 1e-9, "stability {stability}");
        }
    }
    let causal = snap
        .edges
        .iter()
        .find(|edge| edge.legacy_link_type == "causal")
        .expect("causal link dropped");
    assert!(causal.legacy_inferred);
    assert_eq!(causal.link_type, "derived_from");
    let semantic = snap
        .edges
        .iter()
        .find(|edge| edge.legacy_link_type == "semantic")
        .expect("semantic link dropped");
    assert!(semantic.legacy_inferred);
    assert_eq!(semantic.link_type, "derived_from");
    let touched = snap
        .edges
        .iter()
        .find(|edge| edge.legacy_link_type == "touched")
        .expect("touched link dropped");
    assert!(!touched.legacy_inferred);
    assert_eq!(touched.link_type, "touched");
    assert_v3_source_kept(db, log_dir);
}

fn assert_v3_source_kept(db: &Path, log_dir: &Path) {
    let snap = snapshot(log_dir);
    let expected_ms = strata_migrate::source::timestamp_ms("2026-02-20T11:30:00+00:00").unwrap();
    let ids = knowledge_ids(db);
    assert!(ids.len() >= 4, "fixture lost its memories");
    for id in &ids {
        let node = snap
            .nodes
            .iter()
            .find(|node| &node.legacy_id == id)
            .unwrap_or_else(|| panic!("dropped memory {id}"));
        let source = node
            .source
            .as_ref()
            .unwrap_or_else(|| panic!("{id} dropped its v3 source"));
        assert_eq!(source.system, "fixture", "{id}");
        assert!(source.project.is_empty(), "{id}");
        assert!(source.id.is_empty(), "{id}");
        assert_eq!(node.source_updated_at_ms, Some(expected_ms), "{id}");
        let raw = node
            .legacy
            .iter()
            .find(|(key, _)| key == "knowledge_nodes.source")
            .map(|(_, value)| value.as_str());
        assert_eq!(raw, Some("fixture"), "{id} legacy source column dropped");
    }
    let receipts: Vec<_> = snap
        .nodes
        .iter()
        .filter(|node| node.node_type == "walk_receipt")
        .collect();
    assert!(
        !receipts.is_empty(),
        "fixture walk receipt missing; cannot check the None path"
    );
    for node in receipts {
        assert!(
            node.source.is_none(),
            "walk receipt invented a source: {:?}",
            node.source
        );
        assert!(node.source_updated_at_ms.is_none());
    }
}

fn dir_has_seg(dir: &Path) -> bool {
    fs::read_dir(dir).ok().is_some_and(|entries| {
        entries
            .flatten()
            .any(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("seg"))
    })
}

#[test]
fn v311_upgrade_keeps_source_and_source_updated_at() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    let status = upgrade_if_needed(&db).unwrap();
    let UpgradeStatus::StrataReady { log_dir } = status else {
        panic!("expected an installed strata log");
    };
    assert_v3_source_kept(&db, &log_dir);
}

#[test]
fn happy_path_on_real_v3_fixture_keeps_the_source_hash() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    let before = sha256_file(&db);
    let staging = staging_directory(dir.path());
    fs::create_dir_all(&staging).unwrap();
    fs::write(staging.join("leftover.seg"), b"not a log").unwrap();

    let status = upgrade_if_needed(&db).unwrap();
    let UpgradeStatus::StrataReady { log_dir } = status else {
        panic!("expected an installed strata log");
    };
    assert_eq!(before, sha256_file(&db), "upgrade modified the v3 file");
    assert!(!staging.exists(), "staging survived a successful swap");
    assert!(log_dir.join("strata.key").is_file() || dir_has_seg(&log_dir));
    let backups: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().contains(".v3-backup-"))
        .map(|entry| entry.path())
        .collect();
    assert_eq!(backups.len(), 1, "{backups:?}");
    assert_eq!(
        before,
        sha256_file(&backups[0]),
        "backup is not a byte copy"
    );
    assert_fixture_landed(&db, &log_dir);

    let again = upgrade_if_needed(&db).unwrap();
    assert!(matches!(again, UpgradeStatus::StrataReady { .. }));
    assert_eq!(before, sha256_file(&db));
    let backups_after = fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().contains(".v3-backup-"))
        .count();
    assert_eq!(backups_after, 1, "relaunch took another backup");
}

#[test]
fn import_failure_leaves_the_v3_hash_and_names_v311() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    let file = fs::OpenOptions::new().write(true).open(&db).unwrap();
    file.set_len(4096).unwrap();
    drop(file);
    let before = sha256_file(&db);
    let err = upgrade_if_needed(&db).unwrap_err();
    let text = err.to_string();
    assert_failure_message(&text, &dir.path().join(UPGRADE_LOG_NAME));
    assert!(text.contains("import failed"), "{text}");
    assert_eq!(before, sha256_file(&db));
    assert!(!staging_directory(dir.path()).exists());
    assert!(!dir.path().join(LOG_DIR_NAME).exists());
}

#[test]
fn verify_failure_leaves_the_v3_hash_and_names_v311() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    let before = sha256_file(&db);
    let err = upgrade_with(
        &db,
        UpgradeOptions {
            after_import: Some(Box::new(|staging| {
                let seg = fs::read_dir(staging)?
                    .flatten()
                    .map(|entry| entry.path())
                    .find(|path| path.extension().and_then(|ext| ext.to_str()) == Some("seg"))
                    .ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::NotFound, "no segment to tamper")
                    })?;
                let mut bytes = fs::read(&seg)?;
                let index = bytes.len() / 2;
                bytes[index] ^= 0xff;
                fs::write(&seg, bytes)?;
                Ok(())
            })),
        },
    )
    .unwrap_err();
    let text = err.to_string();
    assert_failure_message(&text, &dir.path().join(UPGRADE_LOG_NAME));
    assert!(
        text.contains("strata-verify failed"),
        "verify injection did not fail closed: {text}"
    );
    assert_eq!(before, sha256_file(&db));
    assert!(!staging_directory(dir.path()).exists());
    assert!(!dir.path().join(LOG_DIR_NAME).exists());
}
