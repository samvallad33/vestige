//! `strata-verify` must not create or modify any file in the directory it
//! checks, and a good import (no kernel.log) must exit 0.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use strata_store::{IngestInput, StrataStore};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../strata-migrate/tests/fixtures/v3.1.1-sample.sqlite"
);

fn snapshot(root: &Path) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rel_dir = dir
            .strip_prefix(root)
            .unwrap_or(Path::new(""))
            .to_string_lossy()
            .replace('\\', "/");
        if dir != root {
            out.insert(format!("{rel_dir}/"), "dir".to_string());
        }
        for entry in std::fs::read_dir(&dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let rel = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            let bytes = std::fs::read(&path).unwrap();
            out.insert(rel, blake3::hash(&bytes).to_hex().to_string());
        }
    }
    out
}

fn build_log(dir: &Path) {
    std::fs::write(dir.parent().unwrap().join("receipt-signing.key"), [9u8; 32]).unwrap();
    strata_migrate::migrate_with_options(
        Path::new(FIXTURE),
        dir,
        strata_migrate::MigrateOptions {
            seed: Some([9u8; 32]),
            ..Default::default()
        },
    )
    .expect("migration for the read-only check");
}

#[test]
fn good_import_is_unchanged_and_exits_zero() {
    let tmp = tempfile::tempdir().unwrap();
    let log_dir = tmp.path().join("strata");
    build_log(&log_dir);
    assert!(!log_dir.join("kernel.log").exists());

    let before = snapshot(&log_dir);
    assert!(before.keys().any(|name| name.ends_with(".seg")));
    assert!(before.contains_key("strata.key"));

    let report = strata_verify::verify_path(&log_dir);
    assert!(
        report.ok,
        "good import must verify without kernel.log: {}",
        report.json
    );
    assert!(
        !report.json.contains("kernel.log"),
        "migrated log must not fall through: {}",
        report.json
    );
    assert_eq!(snapshot(&log_dir), before, "verify_path wrote the log");

    let bin = env!("CARGO_BIN_EXE_strata-verify");
    let output = Command::new(bin)
        .arg(&log_dir)
        .output()
        .expect("spawn strata-verify");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "exit {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status.code()
    );
    assert!(stdout.contains("\"ok\": true"), "{stdout}");
    assert!(stdout.contains("OK"), "{stdout}");
    assert!(
        !stdout.contains("FAILED"),
        "must not fall through to FAILED:\n{stdout}"
    );
    assert!(
        !stderr.contains("kernel.log"),
        "stderr must not mention kernel.log:\n{stderr}"
    );
    assert_eq!(snapshot(&log_dir), before, "the binary wrote the log");
}

#[test]
fn live_store_root_is_unchanged_and_passes() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("store");
    {
        let mut store = StrataStore::open(&dir).expect("open store");
        store
            .ingest(IngestInput {
                content: "read-only verify must not mint a root segment".into(),
                source: None,
                source_updated_at_ms: None,
                node_type: String::new(),
                tags: Vec::new(),
                created_at_ms: Some(1_700_000_000_000),
                valid_from_ms: None,
                valid_until_ms: None,
            })
            .expect("ingest");
        store
            .seal_checkpoint()
            .expect("checkpoint writes store.meta");
    }
    assert!(dir.join("store.meta").is_file());
    assert!(!dir.join("strata.key").exists());
    let before = snapshot(&dir);

    let report = strata_verify::verify_path(&dir);
    assert!(report.ok, "live store must verify: {}", report.json);
    assert_eq!(snapshot(&dir), before, "verify wrote the store root");

    let output = Command::new(env!("CARGO_BIN_EXE_strata-verify"))
        .arg(&dir)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(snapshot(&dir), before, "the binary wrote the store root");

    // The log directory itself is also a check target (no kernel.log).
    let log_dir = dir.join("log");
    let log_before = snapshot(&log_dir);
    let log_report = strata_verify::verify_path(&log_dir);
    assert!(
        log_report.ok,
        "live log dir must verify: {}",
        log_report.json
    );
    assert_eq!(snapshot(&log_dir), log_before);
}

#[test]
fn fresh_store_without_anchor_passes() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("store");
    {
        let mut store = StrataStore::open(&dir).expect("open store");
        store
            .ingest(IngestInput {
                content: "no checkpoint yet".into(),
                source: None,
                source_updated_at_ms: None,
                node_type: String::new(),
                tags: Vec::new(),
                created_at_ms: Some(1_700_000_000_000),
                valid_from_ms: None,
                valid_until_ms: None,
            })
            .expect("ingest");
    }
    assert!(!dir.join("store.meta").exists());
    let before = snapshot(&dir);
    let report = strata_verify::verify_path(&dir);
    assert!(
        report.ok,
        "unsealed live store must verify: {}",
        report.json
    );
    assert_eq!(snapshot(&dir), before, "verify wrote the store root");
}

#[test]
fn foreign_directory_is_not_initialized() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("notes");
    std::fs::create_dir(&dir).unwrap();
    std::fs::write(dir.join("note.txt"), b"leave me alone").unwrap();
    let before = snapshot(&dir);

    let report = strata_verify::verify_path(&dir);
    assert!(!report.ok);
    assert_eq!(snapshot(&dir), before);
    assert!(!dir.join("strata.key").exists());
    let segs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "seg"))
        .collect();
    assert!(segs.is_empty());
}

#[test]
fn wrong_key_fails_without_replacing_it() {
    let tmp = tempfile::tempdir().unwrap();
    let log_dir = tmp.path().join("strata");
    build_log(&log_dir);
    let before_key = std::fs::read(log_dir.join("strata.key")).unwrap();
    std::fs::write(log_dir.join("strata.key"), [3u8; 32]).unwrap();
    let before = snapshot(&log_dir);

    let report = strata_verify::verify_path(&log_dir);
    assert!(!report.ok, "wrong key must fail: {}", report.json);
    assert!(
        report.failures.iter().any(|f| f.contains("signature")),
        "{:?}",
        report.failures
    );
    assert_eq!(snapshot(&log_dir), before);
    assert_eq!(
        std::fs::read(log_dir.join("strata.key")).unwrap(),
        [3u8; 32]
    );
    assert_ne!(before_key, [3u8; 32]);
}
