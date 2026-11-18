//! Import writes a staging directory and renames it onto `--to` only after
//! the pre-checks pass. A SIGKILL mid-import leaves `--to` untouched; the
//! next run deletes the staging directory and finishes one receipt.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use borsh::BorshDeserialize;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/v3.1.1-sample.sqlite"
);

fn staging_of(dest: &Path) -> PathBuf {
    let name = dest.file_name().unwrap();
    let mut staged = name.to_os_string();
    staged.push(strata_migrate::STAGING_SUFFIX);
    dest.with_file_name(staged)
}

fn has_segment(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries
        .filter_map(Result::ok)
        .any(|entry| entry.path().extension().is_some_and(|ext| ext == "seg"))
}

fn count_receipts(dir: &Path) -> usize {
    let mut segs: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "seg"))
        .collect();
    segs.sort();
    let mut receipts = 0usize;
    for path in segs {
        let bytes = std::fs::read(&path).unwrap();
        let mut cursor = bytes.as_slice();
        if strata::SegmentHeader::deserialize_reader(&mut cursor).is_err() {
            continue;
        }
        loop {
            if cursor.is_empty() {
                break;
            }
            let frame = match strata::Frame::deserialize_reader(&mut cursor) {
                Ok(frame) => frame,
                Err(_) => break,
            };
            if frame.kind == strata_migrate::KIND_MIGRATION_RECEIPT {
                receipts += 1;
            }
        }
    }
    receipts
}

fn snapshot(root: &Path) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let rel = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            out.push((rel, std::fs::read(&path).unwrap()));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[test]
fn occupied_destination_is_refused_before_any_write() {
    let tmp = tempfile::tempdir().unwrap();
    let dest = tmp.path().join("to");
    std::fs::create_dir(&dest).unwrap();
    std::fs::write(dest.join("note.txt"), b"leave me").unwrap();

    let err = strata_migrate::migrate(Path::new(FIXTURE), &dest).unwrap_err();
    assert!(
        matches!(
            err,
            strata_migrate::MigrationError::DestinationNotEmpty { .. }
        ),
        "{err:?}"
    );
    assert_eq!(std::fs::read(dest.join("note.txt")).unwrap(), b"leave me");
    assert!(!dest.join("strata.key").exists());
    assert!(!has_segment(&dest));
    assert!(!staging_of(&dest).exists());
}

#[test]
fn sigkill_mid_import_rerun_finishes_one_receipt() {
    if std::env::var_os("STRATA_MIGRATE_SIGKILL_CHILD").is_some() {
        let from = std::env::var("STRATA_MIGRATE_FROM").unwrap();
        let to = std::env::var("STRATA_MIGRATE_TO").unwrap();
        strata_migrate::migrate(Path::new(&from), Path::new(&to)).expect("child import");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let dest = tmp.path().join("to-kill");
    let staging = staging_of(&dest);
    let exe = std::env::current_exe().unwrap();
    let mut child = Command::new(&exe)
        .args(["--exact", "sigkill_mid_import_rerun_finishes_one_receipt"])
        .env("STRATA_MIGRATE_SIGKILL_CHILD", "1")
        .env("STRATA_MIGRATE_SIGKILL_WINDOW", "1")
        .env("STRATA_MIGRATE_FROM", FIXTURE)
        .env("STRATA_MIGRATE_TO", &dest)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn import");

    let deadline = Instant::now() + Duration::from_secs(20);
    while !has_segment(&staging) {
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("staging never received a segment");
        }
        if let Some(status) = child.try_wait().unwrap() {
            panic!("child exited before the kill window: {status}");
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    child.kill().expect("SIGKILL");
    #[cfg_attr(not(unix), allow(unused_variables))]
    let status = child.wait().unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(
            status.signal(),
            Some(9),
            "kill must be SIGKILL, got {status}"
        );
    }
    assert!(!dest.exists(), "SIGKILL must not publish --to");
    assert!(has_segment(&staging), "partial import stays in staging");

    let control = {
        let dir = tmp.path().join("control");
        strata_migrate::migrate(Path::new(FIXTURE), &dir).expect("control import")
    };
    let report = strata_migrate::migrate(Path::new(FIXTURE), &dest).expect("rerun after SIGKILL");
    assert!(report.verify_passed, "{report:?}");
    assert!(!report.idempotent_reuse);
    assert_eq!(report.nodes, control.nodes);
    assert_eq!(report.edges, control.edges);
    assert_eq!(report.fsrs_events, control.fsrs_events);
    assert!(!staging.exists(), "rerun removes the staging directory");
    assert_eq!(count_receipts(&dest), 1);

    let before = snapshot(&dest);
    let again = strata_migrate::migrate(Path::new(FIXTURE), &dest).expect("idempotent rerun");
    assert!(again.idempotent_reuse);
    assert_eq!(
        snapshot(&dest),
        before,
        "idempotent rerun must not write --to"
    );
    assert_eq!(count_receipts(&dest), 1);
}
