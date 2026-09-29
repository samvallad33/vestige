//! First launch of the SQLite-free 4.0 binaries.
//!
//! A v3 store is recognized only when `vestige.db` exists. Existence is
//! `metadata` (`stat`): this process never opens the file. The import runs
//! in `vestige-upgrade`, looked up next to this executable and then on
//! `PATH`. A missing tool is a one-line refusal. This process does not
//! create sidecars.

use std::path::{Path, PathBuf};
use std::process::Command;

pub const LOG_DIR_NAME: &str = "log";
pub const UPGRADE_LOG_NAME: &str = "upgrade.log";
const STAGING_SUFFIX: &str = ".strata-staging";

/// How to get the importer. One line, printed when the tool is not installed.
pub const UPGRADE_HINT: &str = "install vestige-upgrade from https://github.com/samvallad33/vestige/releases and place it next to this binary or on PATH";

pub enum UpgradeLaunch {
    /// The probed path is not a file and no partial upgrade is waiting.
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

/// Detect a v3 store by file existence and run `vestige-upgrade` when one is present.
pub fn ensure_upgraded(db_path: &Path) -> Result<UpgradeLaunch, LaunchError> {
    let data_dir = data_dir_of(db_path);
    if log_ready(&data_dir.join(LOG_DIR_NAME)) {
        return Ok(UpgradeLaunch::Ready);
    }
    // Existence only. `metadata` is `stat`; it does not open the file, so a
    // mode-000 database and a non-SQLite database are both candidates.
    let present = match std::fs::metadata(db_path) {
        Ok(meta) => meta.is_file(),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => false,
        Err(err) => {
            return Err(LaunchError::Missing(format!(
                "v3 store at {} was not modified ({err}); {UPGRADE_HINT}",
                db_path.display()
            )));
        }
    };
    if !present && !staging_directory(&data_dir).exists() {
        return Ok(UpgradeLaunch::NotV3);
    }
    let Some(bin) = locate_upgrade() else {
        return Err(LaunchError::Missing(format!(
            "v3 store at {} was not modified; {UPGRADE_HINT}",
            db_path.display()
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
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        for name in sibling_names(&exe) {
            let sibling = dir.join(&name);
            if sibling.is_file() {
                return Some(sibling);
            }
        }
    }
    let path_var = std::env::var_os("PATH")?;
    let plain = upgrade_file_name();
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(plain);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Literal `vestige-upgrade`, plus the platform-suffixed name MCPB uses
/// (`vestige-mcp-linux-x64` → `vestige-upgrade-linux-x64`).
fn sibling_names(exe: &Path) -> Vec<String> {
    let mut names = vec![upgrade_file_name().to_string()];
    let Some(file) = exe.file_name().and_then(|name| name.to_str()) else {
        return names;
    };
    let Some(rest) = file.strip_prefix("vestige-mcp") else {
        return names;
    };
    let suffixed = format!("vestige-upgrade{rest}");
    if suffixed != names[0] {
        names.push(suffixed);
    }
    names
}

fn upgrade_file_name() -> &'static str {
    if cfg!(windows) {
        "vestige-upgrade.exe"
    } else {
        "vestige-upgrade"
    }
}
