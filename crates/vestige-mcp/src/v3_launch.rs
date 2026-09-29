//! 4.0 notices a v3 store only when `vestige.db` exists. It never opens that
//! file. The import runs in `vestige-upgrade`.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

/// Last v3 release operators can keep running when 4.0 cannot upgrade.
pub const V311_RELEASE: &str = "https://github.com/samvallad33/vestige/releases/tag/v3.1.1";

/// Why the launch path stopped before opening a Strata log.
#[derive(Debug)]
pub enum LaunchError {
    /// `vestige.db` is present and `vestige-upgrade` is not installed.
    MissingTool { message: String },
    /// `vestige-upgrade` ran and exited non-zero. Its own stderr has the detail.
    UpgradeFailed { status: ExitStatus },
}

impl std::fmt::Display for LaunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaunchError::MissingTool { message } => f.write_str(message),
            LaunchError::UpgradeFailed { status } => {
                write!(f, "vestige-upgrade exited {status}")
            }
        }
    }
}

impl std::error::Error for LaunchError {}

impl LaunchError {
    pub fn code(&self) -> i32 {
        match self {
            LaunchError::MissingTool { .. } => 1,
            LaunchError::UpgradeFailed { status } => status.code().unwrap_or(1),
        }
    }
}

/// True when `path` is an existing file. Uses `metadata` (stat), never `open`.
pub fn db_present(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|meta| meta.is_file())
}

/// Run `vestige-upgrade` when `vestige.db` exists. Leave the file untouched
/// when the tool is missing.
pub fn upgrade_or_refuse(db_path: &Path) -> Result<(), LaunchError> {
    if !db_present(db_path) {
        return Ok(());
    }
    let Some(bin) = find_upgrade() else {
        return Err(LaunchError::MissingTool {
            message: refusal(db_path),
        });
    };
    let data_dir = db_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let status = Command::new(&bin)
        .arg("--data-dir")
        .arg(data_dir)
        .status()
        .map_err(|err| LaunchError::MissingTool {
            message: format!(
                "failed to run {}: {err}\n{}",
                bin.display(),
                refusal(db_path)
            ),
        })?;
    if status.success() {
        Ok(())
    } else {
        Err(LaunchError::UpgradeFailed { status })
    }
}

fn refusal(db_path: &Path) -> String {
    format!(
        "v3 store detected at {} because vestige.db exists.\n\
         vestige-upgrade was not found next to this binary or on PATH.\n\
         Install vestige-upgrade from the release archive (it ships beside vestige), \
         or keep using v3.1.1: {V311_RELEASE}\n\
         The v3 data is untouched.",
        db_path.display()
    )
}

fn upgrade_file_name() -> &'static str {
    if cfg!(windows) {
        "vestige-upgrade.exe"
    } else {
        "vestige-upgrade"
    }
}

/// Sibling of this executable, then each `PATH` directory.
fn find_upgrade() -> Option<PathBuf> {
    let name = upgrade_file_name();
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}
