//! 4.0 refuses a v3 file without opening it when `vestige-upgrade` is absent,
//! and runs a sibling `vestige-upgrade` when one is present.

#![cfg(unix)]

use std::fs;
use std::io::Read;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;

use sha2::{Digest, Sha256};

fn fixture_db() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
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

fn isolated_vestige(dir: &Path) -> std::path::PathBuf {
    let bin = dir.join("vestige");
    fs::copy(env!("CARGO_BIN_EXE_vestige"), &bin).unwrap();
    let mut perms = fs::metadata(&bin).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&bin, perms).unwrap();
    bin
}

fn no_sidecars(db: &Path) {
    for suffix in ["-wal", "-shm"] {
        let mut name = db.as_os_str().to_os_string();
        name.push(suffix);
        let path = std::path::PathBuf::from(name);
        assert!(!path.exists(), "sqlite sidecar created: {}", path.display());
    }
}

#[test]
fn refuse_path_leaves_the_v3_hash_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let bin_dir = tempfile::tempdir().unwrap();
    let vestige = isolated_vestige(bin_dir.path());
    let db = dir.path().join("vestige.db");
    fs::copy(fixture_db(), &db).unwrap();
    let before = sha256_file(&db);
    // Root ignores mode 000. Otherwise a read of the file fails while stat succeeds.
    // `/proc/self` is owned by the effective uid. Root can read a mode-000 file.
    let unreadable = std::fs::metadata("/proc/self")
        .map(|meta| std::os::unix::fs::MetadataExt::uid(&meta) != 0)
        .unwrap_or(false);
    if unreadable {
        let mut perms = fs::metadata(&db).unwrap().permissions();
        perms.set_mode(0o0);
        fs::set_permissions(&db, perms).unwrap();
    }

    let output = Command::new(&vestige)
        .args(["--data-dir", dir.path().to_str().unwrap(), "stats"])
        .env("PATH", "/usr/bin:/bin")
        .env_remove("VESTIGE_DATA_DIR")
        .output()
        .unwrap();

    if unreadable {
        let mut perms = fs::metadata(&db).unwrap().permissions();
        perms.set_mode(0o644);
        fs::set_permissions(&db, perms).unwrap();
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !output.status.success(),
        "missing vestige-upgrade still exited 0\n{stderr}"
    );
    assert!(
        stderr.contains("vestige-upgrade") && stderr.contains("untouched"),
        "{stderr}"
    );
    assert!(
        stderr.contains("https://github.com/samvallad33/vestige/releases/tag/v3.1.1"),
        "{stderr}"
    );
    assert!(stdout.trim().is_empty(), "refusal wrote stdout: {stdout}");
    assert_eq!(before, sha256_file(&db), "refuse path changed vestige.db");
    no_sidecars(&db);
    assert!(
        !dir.path().join("log").exists(),
        "refusal created a strata log"
    );
}

#[test]
fn sibling_upgrade_runs_and_does_not_open_the_v3_file() {
    let dir = tempfile::tempdir().unwrap();
    let bin_dir = tempfile::tempdir().unwrap();
    let vestige = isolated_vestige(bin_dir.path());
    let marker = dir.path().join("upgrade-ran");
    let script = format!("#!/bin/sh\n: > '{}'\nexit 0\n", marker.display());
    let upgrade = bin_dir.path().join("vestige-upgrade");
    fs::write(&upgrade, script).unwrap();
    let mut perms = fs::metadata(&upgrade).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&upgrade, perms).unwrap();

    let db = dir.path().join("vestige.db");
    fs::copy(fixture_db(), &db).unwrap();
    let before = sha256_file(&db);

    let output = Command::new(&vestige)
        .args(["--data-dir", dir.path().to_str().unwrap(), "stats"])
        .env("PATH", "/usr/bin:/bin")
        .env_remove("VESTIGE_DATA_DIR")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        marker.is_file(),
        "sibling vestige-upgrade was not run\n{stderr}"
    );
    assert_eq!(
        before,
        sha256_file(&db),
        "dispatch opened or wrote vestige.db"
    );
    no_sidecars(&db);
    assert!(
        output.status.success(),
        "stats after a successful upgrade exited {:?}\nstdout: {stdout}\nstderr: {stderr}",
        output.status.code()
    );
    assert!(
        stdout.contains("Total Memories"),
        "stats did not read the strata log\n{stdout}"
    );
    assert!(
        !stdout.contains("upgrading to strata"),
        "upgrade progress leaked to stdout: {stdout}"
    );
}
