//! The scratch copy made for a source with a live WAL lives in the data
//! directory, is owner-only, and never outlives the run.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use strata_migrate::{migrate_with_options, MigrateOptions};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/v3.1.1-sample.sqlite"
);

/// The first test points `TMPDIR` at a probe directory; both tests hold
/// this lock so neither sees the other's environment.
static ENV_LOCK: Mutex<()> = Mutex::new(());

struct TmpdirGuard(Option<std::ffi::OsString>);

impl Drop for TmpdirGuard {
    fn drop(&mut self) {
        // Safety: serialized by ENV_LOCK across all tests in this binary.
        unsafe {
            match self.0.take() {
                Some(value) => std::env::set_var("TMPDIR", value),
                None => std::env::remove_var("TMPDIR"),
            }
        }
    }
}

/// A data directory holding a copy of the fixture with a non-empty WAL.
fn data_dir_with_wal(root: &Path) -> (PathBuf, PathBuf) {
    let data = root.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let db = data.join("vestige.db");
    std::fs::copy(FIXTURE, &db).unwrap();
    std::fs::write(data.join("vestige.db-wal"), b"synthetic wal frames").unwrap();
    (db, data.join("log"))
}

fn find_named(root: &Path, name: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.file_name().is_some_and(|n| n == name) {
                found.push(path.clone());
            }
            if path.is_dir() {
                stack.push(path);
            }
        }
    }
    found
}

/// Permission bits of `path`; owner-only everywhere but unix by definition.
fn mode_of(path: &Path) -> u32 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        0o600
    }
}

#[test]
fn snapshot_copy_is_owner_only_in_the_data_dir_and_removed() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _restore = TmpdirGuard(std::env::var_os("TMPDIR"));
    let root = tempfile::tempdir().unwrap();
    let (db, dest) = data_dir_with_wal(root.path());
    let data = dest.parent().unwrap().to_path_buf();
    let probe = root.path().join("system-tmp");
    std::fs::create_dir(&probe).unwrap();
    // Safety: serialized by ENV_LOCK across all tests in this binary.
    unsafe {
        std::env::set_var("TMPDIR", &probe);
    }

    let seen: Arc<Mutex<Vec<(PathBuf, u32, u32)>>> = Arc::default();
    let seen_in_hook = Arc::clone(&seen);
    let data_in_hook = data.clone();
    let probe_in_hook = probe.clone();
    let in_probe: Arc<Mutex<Vec<PathBuf>>> = Arc::default();
    let in_probe_hook = Arc::clone(&in_probe);
    let report = migrate_with_options(
        &db,
        &dest,
        MigrateOptions {
            accept_wal_snapshot: true,
            before_import: Some(Box::new(move |_staging| {
                *seen_in_hook.lock().unwrap() = find_named(&data_in_hook, "snapshot.db")
                    .into_iter()
                    .map(|copy| {
                        let dir_mode = mode_of(copy.parent().unwrap());
                        let file_mode = mode_of(&copy);
                        (copy, dir_mode, file_mode)
                    })
                    .collect();
                *in_probe_hook.lock().unwrap() = find_named(&probe_in_hook, "snapshot.db");
                Ok(())
            })),
            ..Default::default()
        },
    )
    .expect("snapshot run");
    assert!(report.verify_passed);

    assert!(
        in_probe.lock().unwrap().is_empty(),
        "the copy must not go to the system temp dir"
    );
    let copies = seen.lock().unwrap().clone();
    assert_eq!(copies.len(), 1, "one scratch copy under the data dir");
    let (_, dir_mode, file_mode) = &copies[0];
    assert_eq!(file_mode & 0o077, 0, "copy is owner-only");
    assert_eq!(dir_mode & 0o077, 0, "scratch dir is owner-only");
    assert!(
        find_named(&data, "snapshot.db").is_empty(),
        "the copy is removed when the run ends"
    );
}

#[test]
fn an_interrupted_run_leaves_nothing_after_the_next_run() {
    if std::env::var_os("STRATA_SCRATCH_CHILD").is_some() {
        let from = std::env::var("STRATA_SCRATCH_FROM").unwrap();
        let to = std::env::var("STRATA_SCRATCH_TO").unwrap();
        let _ = migrate_with_options(
            Path::new(&from),
            Path::new(&to),
            MigrateOptions {
                accept_wal_snapshot: true,
                ..Default::default()
            },
        );
        return;
    }

    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let root = tempfile::tempdir().unwrap();
    let (db, dest) = data_dir_with_wal(root.path());
    let tmp_dir = root.path().join("system-tmp");
    std::fs::create_dir(&tmp_dir).unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "an_interrupted_run_leaves_nothing_after_the_next_run",
        ])
        .env("TMPDIR", &tmp_dir)
        .env("STRATA_SCRATCH_CHILD", "1")
        .env("STRATA_MIGRATE_SIGKILL_WINDOW", "1")
        .env("STRATA_SCRATCH_FROM", &db)
        .env("STRATA_SCRATCH_TO", &dest)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn import");
    let deadline = Instant::now() + Duration::from_secs(30);
    // The window seam holds the child after the import, scratch still live.
    while find_named(root.path(), "snapshot.db").is_empty()
        || !dest.with_file_name("log.strata-staging").exists()
    {
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("scratch copy never appeared");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    std::thread::sleep(Duration::from_millis(300));
    child.kill().unwrap();
    child.wait().unwrap();
    assert!(
        !find_named(root.path(), "snapshot.db").is_empty(),
        "a killed run cannot clean up after itself"
    );

    let report = migrate_with_options(
        &db,
        &dest,
        MigrateOptions {
            accept_wal_snapshot: true,
            ..Default::default()
        },
    )
    .expect("rerun");
    assert!(report.verify_passed);
    assert!(
        find_named(root.path(), "snapshot.db").is_empty(),
        "the next run removes the dead run's copy"
    );
}
