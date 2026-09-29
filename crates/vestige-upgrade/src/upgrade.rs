//! First-launch v3 → strata upgrade, owned by the `vestige-upgrade` binary.
//!
//! The v3 file is only ever read. This process holds `upgrade.lock` across
//! the import and renames `log.strata-staging` onto `log/`. The kernel drops
//! that lock on SIGKILL. A rerun reclaims an unpublished staging directory
//! and leaves an already-renamed log in place. The v3 file is never written.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use strata_migrate::MigrateOptions;

use crate::cross_check;

/// Installed strata log. Same relative path `StrataStore` opens.
/// Staging for this destination is `log` plus [`strata_migrate::STAGING_SUFFIX`]:
/// `<data-dir>/log.strata-staging`.
pub const LOG_DIR_NAME: &str = "log";
/// Append-only upgrade record. The failure message names this path.
pub const UPGRADE_LOG_NAME: &str = "upgrade.log";
/// Last v3 release operators can keep running when 4.0 cannot upgrade.
pub const V311_RELEASE: &str = "https://github.com/samvallad33/vestige/releases/tag/v3.1.1";
/// Exclusive lock in the data directory, outside `log/`, held across the
/// staging rename. Not inside the log, so a reader does not treat it as a segment.
const UPGRADE_LOCK_NAME: &str = "upgrade.lock";
/// How long the after-rename kill window stays open. The test SIGKILLs
/// this process while the lock is still held and `log/` is already published.
const AFTER_RENAME_WINDOW: Duration = Duration::from_secs(60);

/// Backup plus staging log, relative to the sqlite family size.
const SPACE_FACTOR: u64 = 3;

/// What the boot path should do after the upgrade attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeStatus {
    /// No v3 SQLite file at the guarded path.
    NoV3,
    /// A strata log is installed (this launch, or a previous one).
    StrataReady { log_dir: PathBuf },
}

/// Upgrade failed. Display text is the process's stderr message.
#[derive(Debug)]
pub struct UpgradeError {
    log_path: PathBuf,
    detail: String,
}

impl std::fmt::Display for UpgradeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "upgrade failed: {detail}\nThe v3 data is untouched.\nYou can keep using v3.1.1: {V311_RELEASE}\nLog: {log}",
            detail = self.detail,
            log = self.log_path.display(),
        )
    }
}

impl std::error::Error for UpgradeError {}

/// Test seam: runs after a real import and before verify.
pub type AfterImport = Box<dyn FnOnce(&Path) -> io::Result<()>>;

/// Production passes [`UpgradeOptions::default`].
#[derive(Default)]
pub struct UpgradeOptions {
    pub after_import: Option<AfterImport>,
}

/// Upgrade `db_path` when it is a v3 store and no strata log is installed yet.
pub fn upgrade_if_needed(db_path: &Path) -> Result<UpgradeStatus, UpgradeError> {
    upgrade_with(db_path, UpgradeOptions::default())
}

/// Same as [`upgrade_if_needed`], with a post-import seam for tests.
pub fn upgrade_with(
    db_path: &Path,
    mut options: UpgradeOptions,
) -> Result<UpgradeStatus, UpgradeError> {
    let data_dir = data_dir_of(db_path);
    if !upgrade_relevant(&data_dir, db_path) {
        return Ok(UpgradeStatus::NoV3);
    }

    let log_dir = data_dir.join(LOG_DIR_NAME);
    let log_path = data_dir.join(UPGRADE_LOG_NAME);
    if let Some(status) = installed_log(&log_dir, &log_path) {
        return Ok(status);
    }

    let detected = match sqlite_header(db_path) {
        Ok(v) => v,
        Err(e) => return Err(fail(&log_path, format!("v3 detection failed: {e}"))),
    };
    let Some(v3) = detected else {
        return Ok(UpgradeStatus::NoV3);
    };

    // Held until the rename below finishes. SIGKILL drops it; the next
    // process takes it and either reclaims staging or sees the published log.
    let upgrade_lock = hold_upgrade_lock(&data_dir)
        .map_err(|err| fail(&log_path, format!("upgrade lock failed: {err}")))?;
    if let Some(status) = installed_log(&log_dir, &log_path) {
        return Ok(status);
    }

    note(
        &log_path,
        &format!(
            "vestige: v3 store detected at {} (schema {}); upgrading to strata",
            v3.path.display(),
            v3.schema_version
        ),
    );

    if let Err(detail) = ensure_space(&data_dir, db_path) {
        return Err(fail(&log_path, detail));
    }

    let log_path_hook = log_path.clone();
    let backup_db = db_path.to_path_buf();
    let backup_dir = data_dir.clone();
    let backup_log = log_path.clone();
    let after_import = options.after_import.take();
    let report = match strata_migrate::migrate_with_options(
        db_path,
        &log_dir,
        MigrateOptions {
            dry_run: false,
            accept_wal_snapshot: true,
            before_import: Some(Box::new(move |_staging| {
                sweep_partial_backups(&backup_dir);
                let backup = backup_sqlite_family(&backup_db)
                    .map_err(|err| format!("backup failed: {err}"))?;
                note(
                    &backup_log,
                    &format!(
                        "vestige: backed up {} -> {}",
                        backup_db.display(),
                        backup.display()
                    ),
                );
                Ok(())
            })),
            before_publish: Some(Box::new(move |staging| {
                note(
                    &log_path_hook,
                    &format!("vestige: importing into {}", staging.display()),
                );
                if let Some(hook) = after_import {
                    hook(staging).map_err(|err| format!("import failed: {err}"))?;
                }
                note(&log_path_hook, "vestige: verifying strata log");
                match cross_check::verify_migrated_log(staging) {
                    Ok(verified) if verified.ok => Ok(()),
                    Ok(verified) => Err(format!(
                        "strata-verify failed: {}",
                        verified.failures.join("; ")
                    )),
                    Err(err) => Err(format!("strata-verify failed: {err}")),
                }
            })),
            defer_publish: true,
            ..MigrateOptions::default()
        },
    ) {
        Ok(report) => report,
        Err(err) => return Err(fail(&log_path, format!("import failed: {err}"))),
    };
    if !report.verify_passed {
        return Err(fail(
            &log_path,
            "import failed: strata-migrate replay verification failed",
        ));
    }
    if !report.idempotent_reuse {
        let staging = staging_directory(&data_dir);
        if !staging.is_dir() {
            return Err(fail(
                &log_path,
                "import failed: staging directory missing before rename",
            ));
        }
        publish_staged(&staging, &log_dir)
            .map_err(|err| fail(&log_path, format!("import failed: {err}")))?;
    }
    if std::env::var_os("VESTIGE_UPGRADE_SIGKILL_AFTER_RENAME").is_some() {
        std::thread::sleep(AFTER_RENAME_WINDOW);
    }
    drop(upgrade_lock);

    note(
        &log_path,
        &format!("vestige: strata log ready at {}", log_dir.display()),
    );
    Ok(UpgradeStatus::StrataReady { log_dir })
}

fn installed_log(log_dir: &Path, log_path: &Path) -> Option<UpgradeStatus> {
    if !strata_log_ready(log_dir) {
        return None;
    }
    note(
        log_path,
        &format!(
            "vestige: strata log already present at {}; skipping v3 upgrade",
            log_dir.display()
        ),
    );
    Some(UpgradeStatus::StrataReady {
        log_dir: log_dir.to_path_buf(),
    })
}

fn upgrade_relevant(data_dir: &Path, db_path: &Path) -> bool {
    db_path.exists() || data_dir.join(LOG_DIR_NAME).exists() || staging_directory(data_dir).exists()
}

/// `<data-dir>/log.strata-staging`. Same path [`strata_migrate`] publishes into
/// before renaming onto `log/`.
pub fn staging_directory(data_dir: &Path) -> PathBuf {
    let mut name = std::ffi::OsString::from(LOG_DIR_NAME);
    name.push(strata_migrate::STAGING_SUFFIX);
    data_dir.join(name)
}

struct Detected {
    path: PathBuf,
    schema_version: u32,
}

/// Exclusive flock on `<data-dir>/upgrade.lock`. Blocks until the holder exits,
/// including a holder that died by SIGKILL (the kernel releases the lock).
fn hold_upgrade_lock(data_dir: &Path) -> io::Result<File> {
    let path = data_dir.join(UPGRADE_LOCK_NAME);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)?;
    file.lock()?;
    Ok(file)
}

/// Publish `log.strata-staging` as `log/`. Same-directory rename is atomic.
/// Called while [`hold_upgrade_lock`] is still held.
fn publish_staged(staging: &Path, dest: &Path) -> io::Result<()> {
    if dest.exists() {
        for entry in fs::read_dir(dest)? {
            let entry = entry?;
            let name = entry.file_name();
            if name.to_string_lossy().starts_with('.') {
                fs::rename(entry.path(), staging.join(&name))?;
            }
        }
        fs::remove_dir_all(dest)?;
    }
    fs::rename(staging, dest)?;
    let _ = fs::remove_file(dest.join(".upgrade.lock"));
    Ok(())
}

/// SQLite magic plus the `schema_version` table, read through an immutable URI.
/// A missing file is `Ok(None)`. A file that exists but cannot be read, or is
/// not SQLite, is an error: the parent already decided it was a v3 candidate.
fn sqlite_header(path: &Path) -> Result<Option<Detected>, String> {
    const MAGIC: &[u8; 16] = b"SQLite format 3\0";
    let mut header = [0u8; 16];
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) if path_is_file(path) => {
            return Err(format!("vestige.db exists but cannot be read: {err}"));
        }
        Err(err) => return Err(err.to_string()),
    };
    let mut filled = 0usize;
    while filled < header.len() {
        match std::io::Read::read(&mut file, &mut header[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(ref err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(err) => return Err(err.to_string()),
        }
    }
    if filled < MAGIC.len() || &header != MAGIC {
        return Err("vestige.db exists but is not a v3 SQLite store".into());
    }
    let schema_version = schema_version_readonly(path).unwrap_or(0);
    Ok(Some(Detected {
        path: path.to_path_buf(),
        schema_version,
    }))
}

fn schema_version_readonly(path: &Path) -> Option<u32> {
    let uri = format!("file:{}?mode=ro&immutable=1", uri_encode_path(path));
    let flags = rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
        | rusqlite::OpenFlags::SQLITE_OPEN_URI
        | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = rusqlite::Connection::open_with_flags(uri, flags).ok()?;
    conn.query_row("SELECT MAX(version) FROM schema_version", [], |row| {
        row.get(0)
    })
    .ok()
}

fn path_is_file(path: &Path) -> bool {
    fs::metadata(path).is_ok_and(|meta| meta.is_file())
}

fn uri_encode_path(path: &Path) -> String {
    let text = path.to_string_lossy();
    let mut out = String::with_capacity(text.len());
    for byte in text.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'/' | b'.' | b'_' | b'-' | b'~' => {
                out.push(*byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn data_dir_of(db_path: &Path) -> PathBuf {
    match db_path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

pub fn strata_log_ready(log_dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(log_dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let name = entry.file_name();
        name == "strata.key" || Path::new(&name).extension().and_then(|e| e.to_str()) == Some("seg")
    })
}

fn sweep_partial_backups(data_dir: &Path) {
    let Ok(entries) = fs::read_dir(data_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.contains(".v3-backup-") && name.ends_with(".partial") {
            let _ = fs::remove_file(entry.path());
        }
    }
}

fn ensure_space(data_dir: &Path, db_path: &Path) -> Result<(), String> {
    let need = sqlite_family_bytes(db_path).saturating_mul(SPACE_FACTOR);
    match free_bytes(data_dir) {
        Some(free) if free < need => Err(format!(
            "not enough free disk space ({free} bytes free, {need} bytes required)"
        )),
        _ => Ok(()),
    }
}

fn sqlite_family_bytes(db_path: &Path) -> u64 {
    sidecar_paths(db_path)
        .iter()
        .map(|path| fs::metadata(path).map(|m| m.len()).unwrap_or(0))
        .fold(0u64, u64::saturating_add)
}

fn sidecar_paths(db_path: &Path) -> Vec<PathBuf> {
    let mut paths = vec![db_path.to_path_buf()];
    for suffix in ["-wal", "-shm"] {
        let mut name = db_path.as_os_str().to_os_string();
        name.push(suffix);
        paths.push(PathBuf::from(name));
    }
    paths
}

fn backup_sqlite_family(db_path: &Path) -> io::Result<PathBuf> {
    let stamp = format!(
        "{}-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0),
        std::process::id()
    );
    let mut primary = None;
    for src in sidecar_paths(db_path) {
        if !src.exists() {
            continue;
        }
        let file_name = src.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "sqlite path has no file name")
        })?;
        let file_name = file_name.to_string_lossy();
        let parent = src.parent().unwrap_or_else(|| Path::new("."));
        let dest = parent.join(format!("{file_name}.v3-backup-{stamp}"));
        let partial = parent.join(format!("{file_name}.v3-backup-{stamp}.partial"));
        if let Err(e) = copy_file_fsync(&src, &partial) {
            let _ = fs::remove_file(&partial);
            return Err(e);
        }
        if let Err(e) = fs::rename(&partial, &dest) {
            let _ = fs::remove_file(&partial);
            return Err(e);
        }
        let _ = fsync_dir(parent);
        if src == db_path {
            primary = Some(dest);
        }
    }
    primary.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "v3 database disappeared before backup",
        )
    })
}

fn copy_file_fsync(src: &Path, dst: &Path) -> io::Result<()> {
    let mut input = File::open(src)?;
    let mut output = OpenOptions::new().write(true).create_new(true).open(dst)?;
    io::copy(&mut input, &mut output)?;
    output.sync_all()?;
    Ok(())
}

fn fsync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

#[cfg(unix)]
fn free_bytes(dir: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c_path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // Safety: `stat` is a valid out-pointer and `c_path` is a NUL-terminated path.
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), stat.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    // Safety: `statvfs` returned 0, so it wrote a complete `statvfs`.
    let stat = unsafe { stat.assume_init() };
    let frsize = stat.f_frsize;
    let avail = stat.f_bavail;
    if frsize == 0 {
        return None;
    }
    Some(avail.saturating_mul(frsize))
}

#[cfg(not(unix))]
fn free_bytes(_dir: &Path) -> Option<u64> {
    None
}

fn note(log_path: &Path, line: &str) {
    eprintln!("{line}");
    let _ = io::stderr().flush();
    append_log(log_path, line);
}

fn append_log(log_path: &Path, line: &str) {
    let _ = (|| -> io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;
        writeln!(file, "{line}")?;
        file.sync_all()?;
        Ok(())
    })();
}

fn fail(log_path: &Path, detail: impl Into<String>) -> UpgradeError {
    let err = UpgradeError {
        log_path: log_path.to_path_buf(),
        detail: detail.into(),
    };
    append_log(log_path, &err.to_string());
    err
}
