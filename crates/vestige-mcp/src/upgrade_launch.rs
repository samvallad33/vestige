//! First launch of the SQLite-free 4.0 binaries.
//!
//! A v3 store is recognized from the SQLite header (a read, never a write).
//! The import runs in `vestige-upgrade`, looked up next to this executable
//! and then on `PATH`. A missing tool is a one-line refusal. This process
//! does not open the database and does not create sidecars.

use std::path::{Path, PathBuf};
use std::process::Command;

use vestige_core::detect_v3;

pub const LOG_DIR_NAME: &str = "log";
pub const UPGRADE_LOG_NAME: &str = "upgrade.log";
const STAGING_SUFFIX: &str = ".strata-staging";

/// How to get the importer. One line, printed when the tool is not installed.
pub const UPGRADE_HINT: &str = "install vestige-upgrade from https://github.com/samvallad33/vestige/releases and place it next to this binary or on PATH";

pub enum UpgradeLaunch {
    /// The probed file is not a v3 store and no partial upgrade is waiting.
    NotV3,
    /// A strata log is already installed, or `vestige-upgrade` just installed one.
    Ready,
}

pub enum LaunchError {
    Missing(String),
    Exited(i32),
}

impl std::fmt::Display for LaunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaunchError::Missing(message) => f.write_str(message),
            LaunchError::Exited(code) => {
                write!(f, "vestige-upgrade exited with status {code}")
            }
        }
    }
}

/// Detect a v3 store and run `vestige-upgrade` when one is present.
pub fn ensure_upgraded(db_path: &Path) -> Result<UpgradeLaunch, LaunchError> {
    let data_dir = data_dir_of(db_path);
    if log_ready(&data_dir.join(LOG_DIR_NAME)) {
        return Ok(UpgradeLaunch::Ready);
    }
    let detected = detect_v3(db_path)
        .map_err(|err| LaunchError::Missing(format!("v3 detection failed: {err}")))?;
    if detected.is_none() && !staging_directory(&data_dir).exists() {
        return Ok(UpgradeLaunch::NotV3);
    }
    let Some(bin) = locate_upgrade() else {
        let shown = detected
            .as_ref()
            .map(|info| info.path.display().to_string())
            .unwrap_or_else(|| db_path.display().to_string());
        return Err(LaunchError::Missing(format!(
            "v3 store at {shown} was not modified; {UPGRADE_HINT}"
        )));
    };
    let status = Command::new(&bin)
        .arg("--db")
        .arg(db_path)
        .status()
        .map_err(|err| LaunchError::Missing(format!("failed to run {}: {err}", bin.display())))?;
    if status.success() {
        Ok(UpgradeLaunch::Ready)
    } else {
        Err(LaunchError::Exited(status.code().unwrap_or(1)))
    }
}

/// Print a missing-tool hint and exit. A failed upgrade already printed its
/// own stderr; this process exits with that status and adds nothing.
pub fn ensure_upgraded_or_exit(db_path: &Path) -> UpgradeLaunch {
    match ensure_upgraded(db_path) {
        Ok(status) => status,
        Err(LaunchError::Missing(message)) => {
            eprintln!("{message}");
            let _ = std::io::Write::flush(&mut std::io::stderr());
            std::process::exit(1);
        }
        Err(LaunchError::Exited(code)) => std::process::exit(code),
    }
}

pub fn staging_directory(data_dir: &Path) -> PathBuf {
    let mut name = std::ffi::OsString::from(LOG_DIR_NAME);
    name.push(STAGING_SUFFIX);
    data_dir.join(name)
}

fn data_dir_of(db_path: &Path) -> PathBuf {
    match db_path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

fn log_ready(log_dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(log_dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let name = entry.file_name();
        name == "strata.key"
            || Path::new(&name).extension().and_then(|ext| ext.to_str()) == Some("seg")
    })
}

fn locate_upgrade() -> Option<PathBuf> {
    let name = upgrade_file_name();
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let sibling = dir.join(name);
        if sibling.is_file() {
            return Some(sibling);
        }
    }
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn upgrade_file_name() -> &'static str {
    if cfg!(windows) {
        "vestige-upgrade.exe"
    } else {
        "vestige-upgrade"
    }
}
