//! Tamper, gate receipts, determinism, index rebuild, concurrency, crash
//! recovery, and the "links no SQLite" check.
//!
//! Log and store operations go through `strata-driver` (a separate process).
//! `strata-verify` and `nm` / `cargo tree` run against the binaries the
//! workflow built.

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use serde_json::Value;

use super::support::*;

const SEED: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";

fn scratch() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

fn require_ok<'a>(value: &'a Value, what: &str) -> &'a Value {
    if value.get("ok") != Some(&serde_json::Value::Bool(true)) {
        panic!("{what} failed: {value}");
    }
    value
}

fn file_rows(value: &Value) -> Vec<(String, String)> {
    let mut rows: Vec<(String, String)> = value["files"]
        .as_array()
        .unwrap_or_else(|| panic!("no files in {value}"))
        .iter()
        .map(|row| {
            (
                row["name"].as_str().unwrap_or("").to_string(),
                row["blake3"].as_str().unwrap_or("").to_string(),
            )
        })
        .collect();
    rows.sort();
    rows
}

fn seg_snapshot(dir: &Path) -> Vec<(String, String, u64)> {
    seg_files(dir)
        .into_iter()
        .map(|path| {
            let name = path
                .strip_prefix(dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            (name, sha256_file(&path), file_len(&path))
        })
        .collect()
}

fn read_ready(child: &mut std::process::Child) -> Value {
    let stdout = child.stdout.take().expect("hold stdout");
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        let _ = reader.read_line(&mut line);
        let _ = tx.send(line);
    });
    let line = rx
        .recv_timeout(Duration::from_secs(30))
        .unwrap_or_else(|_| panic!("strata-driver hold produced no ready line"));
    serde_json::from_str(line.trim()).unwrap_or_else(|_| {
        panic!("strata-driver hold emitted non-JSON: {line}");
    })
}

fn hold(dir: &Path) -> (std::process::Child, Value) {
    let mut child = Command::new(driver_bin())
        .args(["hold", dir.to_str().unwrap()])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn hold");
    let ready = read_ready(&mut child);
    if ready.get("ready") != Some(&serde_json::Value::Bool(true)) {
        let _ = child.kill();
        panic!("hold did not become ready: {ready}");
    }
    (child, ready)
}

fn log_dir_of(store: &Path) -> PathBuf {
    let nested = store.join("log");
    if nested.is_dir() {
        nested
    } else {
        store.to_path_buf()
    }
}

fn sealed_segment(dir: &Path) -> PathBuf {
    let mut segs = seg_files(dir);
    assert!(
        segs.len() >= 2,
        "a sealed log has a sealed segment and a new active one, found {segs:?}"
    );
    segs.sort();
    segs.remove(0)
}

#[test]
fn tamper_sealed_byte_flip_refuses() {
    let dir = scratch();
    let path = dir.path().join("log");
    require_ok(
        &run_driver(&["write-log", "--seal", path.to_str().unwrap()]),
        "write sealed log",
    );
    let seg = sealed_segment(&path);
    let before_len = file_len(&seg);
    let before_hash = sha256_file(&seg);
    flip_byte(&seg, 80);
    let flipped_hash = sha256_file(&seg);
    assert_ne!(
        before_hash, flipped_hash,
        "flip did not change the sealed segment"
    );
    let opened = run_driver(&["open-log", path.to_str().unwrap()]);
    assert_eq!(
        sha256_file(&seg),
        flipped_hash,
        "open rewrote the sealed segment"
    );
    assert_eq!(
        file_len(&seg),
        before_len,
        "open truncated the sealed segment"
    );
    assert!(
        opened.get("ok") != Some(&serde_json::Value::Bool(true)),
        "byte flip in a sealed segment was accepted: {opened}"
    );
}

#[test]
fn tamper_forged_key_refuses() {
    let dir = scratch();
    let path = dir.path().join("log");
    require_ok(
        &run_driver(&["write-log", "--seal", path.to_str().unwrap()]),
        "write sealed log",
    );
    let key = path.join("strata.key");
    let before_segs = seg_snapshot(&path);
    {
        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&key)
            .unwrap();
        file.write_all(&[0x11; 32]).unwrap();
    }
    let opened = run_driver(&["open-log", path.to_str().unwrap()]);
    assert_eq!(
        seg_snapshot(&path),
        before_segs,
        "forged key rewrote segments"
    );
    assert!(
        opened.get("ok") != Some(&serde_json::Value::Bool(true)),
        "forged strata.key was accepted: {opened}"
    );
}

#[test]
fn tamper_truncated_sealed_segment_refuses() {
    let dir = scratch();
    let path = dir.path().join("log");
    require_ok(
        &run_driver(&["write-log", "--seal", path.to_str().unwrap()]),
        "write sealed log",
    );
    let seg = sealed_segment(&path);
    let file = OpenOptions::new().write(true).open(&seg).unwrap();
    file.set_len(40).unwrap();
    drop(file);
    let opened = run_driver(&["open-log", path.to_str().unwrap()]);
    assert_eq!(
        file_len(&seg),
        40,
        "open rewrote a truncated sealed segment"
    );
    assert!(
        opened.get("ok") != Some(&serde_json::Value::Bool(true)),
        "truncated sealed segment was accepted: {opened}"
    );
}

#[test]
fn tamper_missing_store_meta_refuses() {
    let dir = scratch();
    let path = dir.path().join("store");
    let written = run_driver(&["write-store", path.to_str().unwrap()]);
    require_ok(&written, "write store");
    let meta = path.join("store.meta");
    assert!(meta.is_file(), "seal_checkpoint did not write store.meta");
    let before = seg_snapshot(&path);
    fs::remove_file(&meta).unwrap();
    let opened = run_driver(&["reopen-store", path.to_str().unwrap()]);
    assert_eq!(
        seg_snapshot(&path),
        before,
        "refusing a missing store.meta must not rewrite the log"
    );
    if opened.get("ok") == Some(&serde_json::Value::Bool(true)) {
        missing(
            "a store with checkpoints opened after store.meta was deleted. \
             The anchor file is part of the chain; its absence must be an error \
             (prompt 10), not an unanchored accept.",
        );
    }
    let err = opened["error"].as_str().unwrap_or("");
    assert!(
        err.to_lowercase().contains("store.meta") || err.to_lowercase().contains("anchor"),
        "missing store.meta error does not name the anchor: {opened}"
    );
}

#[test]
fn tamper_missing_head_state_intact_does_not_truncate() {
    let dir = scratch();
    let path = dir.path().join("log");
    let written = run_driver(&["write-log", path.to_str().unwrap()]);
    require_ok(&written, "write log");
    let frames = written["frames"].as_u64().unwrap();
    let before = seg_snapshot(&path);
    fs::remove_file(path.join("head.state")).unwrap();
    let opened = run_driver(&["open-log", path.to_str().unwrap()]);
    require_ok(&opened, "reopen intact log without head.state");
    assert_eq!(opened["frames"].as_u64(), Some(frames));
    assert_eq!(
        seg_snapshot(&path),
        before,
        "deleting head.state truncated an intact segment"
    );
}

#[test]
fn tamper_missing_head_state_and_damaged_segment_refuses_without_truncate() {
    let dir = scratch();
    let path = dir.path().join("log");
    require_ok(
        &run_driver(&["write-log", path.to_str().unwrap()]),
        "write log",
    );
    fs::remove_file(path.join("head.state")).unwrap();
    let segs = seg_files(&path);
    assert_eq!(segs.len(), 1, "expected one active segment");
    flip_byte(&segs[0], 80);
    let damaged = seg_snapshot(&path);
    let opened = run_driver(&["open-log", path.to_str().unwrap()]);
    let after = seg_snapshot(&path);
    if opened.get("ok") == Some(&serde_json::Value::Bool(true)) || after != damaged {
        missing(
            "deleting head.state and damaging the active segment did not refuse. \
             Open treated the watermark as 0 and truncated or recreated the segment \
             (prompt 8). A missing head.state plus a damaged segment must halt and \
             leave the bytes untouched.",
        );
    }
}

#[test]
fn tamper_missing_key_on_unsealed_log_is_not_regenerated() {
    let dir = scratch();
    let path = dir.path().join("log");
    require_ok(
        &run_driver(&["write-log", path.to_str().unwrap()]),
        "write log",
    );
    let key = path.join("strata.key");
    let original = fs::read(&key).unwrap();
    fs::remove_file(&key).unwrap();
    let before = seg_snapshot(&path);
    let opened = run_driver(&["open-log", path.to_str().unwrap()]);
    let regenerated = key.is_file();
    let same_bytes = regenerated && fs::read(&key).ok().as_deref() == Some(original.as_slice());
    if opened.get("ok") == Some(&serde_json::Value::Bool(true)) || (regenerated && !same_bytes) {
        missing(
            "strata.key was missing on an unsealed log and open minted a new one \
             (prompt 9). A missing key must refuse; it must not be regenerated, \
             or the new key can sign a rewritten tail.",
        );
    }
    assert!(
        !regenerated,
        "open rewrote strata.key while refusing: {opened}"
    );
    assert_eq!(
        seg_snapshot(&path),
        before,
        "refusing a missing key rewrote segments"
    );
}

#[test]
fn gate_write_allow_receipt() {
    let dir = scratch();
    let path = dir.path().join("store");
    let written = run_driver(&["write-store", path.to_str().unwrap()]);
    require_ok(&written, "write store");
    let verdicts = written["verdicts"].as_array().cloned().unwrap_or_default();
    if !verdicts.iter().any(|v| v.as_str() == Some("Allow")) {
        missing(&format!(
            "writes produced no Allow gate receipt. verdicts={verdicts:?} body={written}"
        ));
    }
    assert_eq!(written["nodes"].as_u64(), Some(3));
    assert!(written["digest"].as_str().unwrap_or("").len() == 64);
}

#[test]
fn gate_destructive_hold_receipt_does_not_apply() {
    let dir = scratch();
    let path = dir.path().join("store");
    let written = run_driver(&["write-store", path.to_str().unwrap()]);
    require_ok(&written, "write store");
    let supersede = written["supersede"].as_str().unwrap_or("");
    if !supersede.contains("held") {
        missing(&format!(
            "default policy did not Hold the destructive supersede. supersede={supersede}"
        ));
    }
    assert_eq!(
        written["superseded"],
        serde_json::Value::Bool(false),
        "a Held supersede was applied: {written}"
    );
    let verdicts = written["verdicts"].as_array().cloned().unwrap_or_default();
    if !verdicts.iter().any(|v| v.as_str() == Some("Hold")) {
        missing(&format!(
            "destructive action has no Hold receipt. verdicts={verdicts:?}"
        ));
    }
}

#[test]
fn gate_destructive_allow_policy_receipt_applies() {
    let dir = scratch();
    let path = dir.path().join("store");
    let written = run_driver(&["write-store-permissive", path.to_str().unwrap()]);
    require_ok(&written, "permissive write");
    assert_eq!(written["supersede"], "applied", "{written}");
    assert_eq!(written["superseded"], serde_json::Value::Bool(true));
    let verdicts = written["verdicts"].as_array().cloned().unwrap_or_default();
    assert!(
        verdicts.iter().all(|v| v.as_str() == Some("Allow")),
        "permissive policy recorded a non-Allow verdict: {verdicts:?}"
    );
}

#[test]
fn determinism_seeded_log_bytes_identical() {
    let a = scratch();
    let b = scratch();
    let left = run_driver(&[
        "write-log",
        "--seal",
        "--seed",
        SEED,
        a.path().join("log").to_str().unwrap(),
    ]);
    let right = run_driver(&[
        "write-log",
        "--seal",
        "--seed",
        SEED,
        b.path().join("log").to_str().unwrap(),
    ]);
    require_ok(&left, "seeded run a");
    require_ok(&right, "seeded run b");
    assert_eq!(
        file_rows(&left),
        file_rows(&right),
        "two seeded runs were not byte-identical"
    );
}

#[test]
fn determinism_store_digest_identical_across_processes() {
    let a = scratch();
    let b = scratch();
    let left = run_driver(&["write-store", a.path().join("s").to_str().unwrap()]);
    let right = run_driver(&["write-store", b.path().join("s").to_str().unwrap()]);
    require_ok(&left, "store a");
    require_ok(&right, "store b");
    assert_eq!(left["digest"], right["digest"]);
    assert_eq!(left["nodes"], right["nodes"]);
    assert_eq!(left["edges"], right["edges"]);
    let key_a = fs::read(a.path().join("s/log/strata.key")).unwrap();
    let key_b = fs::read(b.path().join("s/log/strata.key")).unwrap();
    assert_ne!(
        key_a, key_b,
        "two fresh stores shared a signing key; keys must come from OS entropy"
    );
}

#[test]
fn determinism_migrate_data_frames_match_and_keys_differ() {
    let work = scratch();
    let src = work.path().join("src");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let before = sha256_file(&src.join("vestige.db"));
    let dest_a = work.path().join("a");
    let dest_b = work.path().join("b");
    let left = run_driver(&["migrate", src.to_str().unwrap(), dest_a.to_str().unwrap()]);
    let right = run_driver(&["migrate", src.to_str().unwrap(), dest_b.to_str().unwrap()]);
    assert_eq!(before, sha256_file(&src.join("vestige.db")));
    require_ok(&left, "migrate a");
    require_ok(&right, "migrate b");
    let dump_a = run_driver(&["dump-migration", dest_a.to_str().unwrap()]);
    let dump_b = run_driver(&["dump-migration", dest_b.to_str().unwrap()]);
    require_ok(&dump_a, "dump a");
    require_ok(&dump_b, "dump b");
    assert_eq!(dump_a["nodes"], dump_b["nodes"], "node payloads diverged");
    assert_eq!(dump_a["edges"], dump_b["edges"], "edge payloads diverged");
    assert_eq!(dump_a["kinds"], dump_b["kinds"], "frame kinds diverged");
    let key_a = fs::read(dest_a.join("strata.key")).unwrap_or_default();
    let key_b = fs::read(dest_b.join("strata.key")).unwrap_or_default();
    assert!(
        !key_a.is_empty() && key_a != key_b,
        "two migrations of one source shared a signing key. The seed must be OS entropy, not the source BLAKE3."
    );
}

#[test]
fn rebuild_digest_matches_after_reopen() {
    let dir = scratch();
    let path = dir.path().join("store");
    let written = run_driver(&["write-store", path.to_str().unwrap()]);
    require_ok(&written, "write store");
    let opened = run_driver(&["reopen-store", path.to_str().unwrap()]);
    require_ok(&opened, "reopen store");
    assert_eq!(written["digest"], opened["digest"]);
    assert_eq!(written["nodes"], opened["nodes"]);
    assert_eq!(written["edges"], opened["edges"]);
    assert_eq!(opened["sweep_gaps"].as_u64(), Some(0));
}

#[test]
fn rebuild_from_segments_key_and_head_only() {
    let dir = scratch();
    let path = dir.path().join("store");
    let written = run_driver(&["write-store", path.to_str().unwrap()]);
    require_ok(&written, "write store");
    let fresh = dir.path().join("rebuilt");
    let fresh_log = fresh.join("log");
    fs::create_dir_all(&fresh_log).unwrap();
    let source_log = log_dir_of(&path);
    for name in ["strata.key", "head.state"] {
        fs::copy(source_log.join(name), fresh_log.join(name)).unwrap();
    }
    for seg in seg_files(&source_log) {
        let dest = fresh_log.join(seg.file_name().unwrap());
        fs::copy(&seg, dest).unwrap();
    }
    fs::copy(path.join("store.meta"), fresh.join("store.meta")).unwrap();
    let opened = run_driver(&["reopen-store", fresh.to_str().unwrap()]);
    require_ok(&opened, "rebuild from segments");
    assert_eq!(
        written["digest"], opened["digest"],
        "index rebuilt from the log did not match"
    );
    assert_eq!(written["nodes"], opened["nodes"]);
    assert_eq!(written["edges"], opened["edges"]);
}

#[test]
fn concurrent_second_writer_is_refused() {
    let dir = scratch();
    let path = dir.path().join("store");
    let (mut child, ready) = hold(&path);
    let before = tree_hashes(&path);
    let second = run_driver(&["write-store", path.to_str().unwrap()]);
    let err = second["error"].as_str().unwrap_or("").to_string();
    if second.get("ok") == Some(&serde_json::Value::Bool(true)) || !err.contains("locked") {
        let _ = child.kill();
        missing(&format!(
            "a second writer was not refused while the first held the directory lock. \
             second={second}"
        ));
    }
    assert_eq!(
        tree_hashes(&path),
        before,
        "refused writer changed the store"
    );
    let _ = child.kill();
    let _ = child.wait();
    let _ = ready;
}

#[test]
fn concurrent_refused_writer_does_not_change_digest() {
    let dir = scratch();
    let path = dir.path().join("store");
    let (mut child, ready) = hold(&path);
    let digest = ready["digest"].clone();
    let _ = run_driver(&["write-store", path.to_str().unwrap()]);
    let _ = child.kill();
    let _ = child.wait();
    let opened = run_driver(&["reopen-store", path.to_str().unwrap()]);
    require_ok(&opened, "reopen after refused writer");
    assert_eq!(opened["digest"], digest);
    assert_eq!(opened["nodes"], ready["nodes"]);
}

#[test]
fn crash_sigkill_after_ack_reopens_same_digest() {
    let dir = scratch();
    let path = dir.path().join("store");
    let (mut child, ready) = hold(&path);
    let digest = ready["digest"].clone();
    child.kill().unwrap();
    let _ = child.wait();
    let opened = run_driver(&["reopen-store", path.to_str().unwrap()]);
    require_ok(&opened, "reopen after SIGKILL");
    assert_eq!(opened["digest"], digest);
    assert_eq!(opened["nodes"], ready["nodes"]);
}

#[test]
fn crash_torn_tail_above_watermark_recovers_acked_frames() {
    let dir = scratch();
    let path = dir.path().join("log");
    let written = run_driver(&["write-log", path.to_str().unwrap()]);
    require_ok(&written, "write log");
    let frames = written["frames"].as_u64().unwrap();
    let segs = seg_files(&path);
    assert_eq!(segs.len(), 1);
    let before_len = file_len(&segs[0]);
    let before_hash = sha256_file(&segs[0]);
    {
        let mut file = OpenOptions::new().append(true).open(&segs[0]).unwrap();
        file.write_all(&[0xA5; 64]).unwrap();
    }
    assert!(file_len(&segs[0]) > before_len);
    let opened = run_driver(&["open-log", path.to_str().unwrap()]);
    require_ok(&opened, "recover torn tail");
    assert_eq!(opened["read_frames"].as_u64(), Some(frames));
    assert_eq!(
        file_len(&segs[0]),
        before_len,
        "torn tail above the watermark was not truncated back to the acked prefix"
    );
    assert_eq!(sha256_file(&segs[0]), before_hash);
}

#[test]
fn shipped_binary_links_no_sqlite() {
    let mcp = product_bin("vestige-mcp");
    let cli = product_bin("vestige");
    let mut hits = Vec::new();
    for bin in [&mcp, &cli] {
        let output = Command::new("nm")
            .arg("-a")
            .arg(bin)
            .output()
            .unwrap_or_else(|e| panic!("HARNESS: nm {}: {e}", bin.display()));
        let text = String::from_utf8_lossy(&output.stdout);
        for needle in ["sqlite3_open", "sqlite3_prepare", "sqlite3_exec"] {
            if text.contains(needle) {
                hits.push(format!("{} contains {needle}", bin.display()));
            }
        }
    }
    let tree = Command::new("cargo")
        .args([
            "tree",
            "-p",
            "vestige-mcp",
            "-e",
            "features",
            "--prefix",
            "none",
            "-i",
            "rusqlite",
        ])
        .current_dir(repo_root())
        .output()
        .unwrap_or_else(|e| panic!("HARNESS: cargo tree: {e}"));
    let tree_text = format!(
        "{}{}",
        String::from_utf8_lossy(&tree.stdout),
        String::from_utf8_lossy(&tree.stderr)
    );
    if tree.status.success() {
        hits.push(format!(
            "cargo tree still reaches rusqlite: {}",
            tree_text.chars().take(500).collect::<String>()
        ));
    }
    if !hits.is_empty() {
        missing(&format!(
            "the shipped binary still links SQLite. Default features include legacy-sqlite \
             (compilation-only today, but the symbols are in the binary). Drop rusqlite from \
             the vestige and vestige-mcp link. {}",
            hits.join("; ")
        ));
    }
}

#[test]
fn verify_untampered_migrated_log_exits_0() {
    let work = scratch();
    let src = work.path().join("src");
    copy_tree(&fixture("probe-v38-ckpt"), &src);
    let dest = work.path().join("strata");
    let migrated = run_driver(&["migrate", src.to_str().unwrap(), dest.to_str().unwrap()]);
    require_ok(&migrated, "migrate for verify");
    let out = run_cmd(
        &product_bin("strata-verify"),
        &[dest.display().to_string()],
        &[],
        &[],
        Duration::from_secs(60),
    );
    let blob = format!("{}{}", out.stdout, out.stderr);
    if out.status != Some(0) || !blob.contains("OK") {
        missing(&format!(
            "strata-verify did not accept an untampered migrated log (exit {:?}). \
             Output: {}",
            out.status,
            blob.chars().take(900).collect::<String>()
        ));
    }
}

#[test]
fn verify_byte_flip_exits_nonzero() {
    let work = scratch();
    let src = work.path().join("src");
    copy_tree(&fixture("probe-v38-ckpt"), &src);
    let dest = work.path().join("strata");
    let migrated = run_driver(&["migrate", src.to_str().unwrap(), dest.to_str().unwrap()]);
    require_ok(&migrated, "migrate for tamper verify");
    let seg = sealed_segment(&dest);
    let before = file_len(&seg);
    flip_byte(&seg, 80);
    let out = run_cmd(
        &product_bin("strata-verify"),
        &[dest.display().to_string()],
        &[],
        &[],
        Duration::from_secs(60),
    );
    assert_eq!(file_len(&seg), before, "verify rewrote the flipped segment");
    if out.status == Some(0) {
        missing(&format!(
            "strata-verify exited 0 after a sealed-segment byte flip. stdout={} stderr={}",
            out.stdout.chars().take(400).collect::<String>(),
            out.stderr.chars().take(400).collect::<String>()
        ));
    }
}
