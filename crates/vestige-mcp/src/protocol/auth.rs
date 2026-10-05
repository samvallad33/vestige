//! Bearer token authentication for the HTTP transport.
//!
//! Token priority:
//! 1. `VESTIGE_AUTH_TOKEN` env var (override)
//! 2. Read from `<data_dir>/auth_token` file
//! 3. Generate `uuid::Uuid::new_v4()`, write to file with 0o600 permissions
//!
//! Security: The token file is created with restricted permissions from the
//! start (via OpenOptionsExt on Unix) to prevent a TOCTOU race where another
//! process could read the token before permissions are set.

use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

use directories::ProjectDirs;
use tracing::{info, warn};

/// Minimum recommended token length when provided via env var.
const MIN_TOKEN_LENGTH: usize = 32;

/// Data directory chosen with `--data-dir` (or `VESTIGE_DATA_DIR`) for this process.
static DATA_DIR_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();

/// Record the data directory selected on the command line so the auth token
/// file lives next to the database instead of in the platform default.
/// Only the first call has an effect.
pub fn set_data_dir(dir: PathBuf) {
    let _ = DATA_DIR_OVERRIDE.set(dir);
}

/// Pick the directory that holds `auth_token`: an explicit override, then a
/// non-empty `VESTIGE_DATA_DIR`, then the platform default.
fn resolve_token_dir(
    override_dir: Option<&PathBuf>,
    env_dir: Option<OsString>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(dir) = override_dir {
        return Ok(dir.clone());
    }
    if let Some(dir) = env_dir.filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(dir));
    }
    let dirs = ProjectDirs::from("com", "vestige", "core")
        .ok_or("could not determine project directories")?;
    Ok(dirs.data_dir().to_path_buf())
}

/// Return the auth token file path inside the Vestige data directory.
pub fn token_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let dir = resolve_token_dir(
        DATA_DIR_OVERRIDE.get(),
        std::env::var_os("VESTIGE_DATA_DIR"),
    )?;
    Ok(dir.join("auth_token"))
}

/// The `VESTIGE_AUTH_TOKEN` override, when set and not blank.
fn env_auth_token() -> Option<String> {
    let token = std::env::var("VESTIGE_AUTH_TOKEN").ok()?;
    let token = token.trim().to_string();
    if token.is_empty() {
        return None;
    }
    if token.len() < MIN_TOKEN_LENGTH {
        warn!(
            "VESTIGE_AUTH_TOKEN is only {} chars (recommended >= {}). \
             Short tokens are vulnerable to brute-force attacks.",
            token.len(),
            MIN_TOKEN_LENGTH
        );
    }
    Some(token)
}

/// The bearer token if one exists: `VESTIGE_AUTH_TOKEN`, else the
/// `auth_token` file. Never creates one. The dashboard checks a script's
/// token with this, so a request can never cause a file to be written.
pub fn read_auth_token() -> Option<String> {
    if let Some(token) = env_auth_token() {
        return Some(token);
    }
    let token = fs::read_to_string(token_path().ok()?).ok()?;
    let token = token.trim().to_string();
    (!token.is_empty()).then_some(token)
}

/// Get (or create) the bearer token used for HTTP transport authentication.
///
/// Priority:
/// 1. `VESTIGE_AUTH_TOKEN` environment variable
/// 2. Existing `auth_token` file in the data directory
/// 3. Newly generated UUID v4, persisted to file
pub fn get_or_create_auth_token() -> Result<String, Box<dyn std::error::Error>> {
    // 1. Env var override
    if let Some(token) = env_auth_token() {
        info!("Using auth token from VESTIGE_AUTH_TOKEN env var");
        return Ok(token);
    }

    let path = token_path()?;

    // 2. Read existing file
    if path.exists() {
        let token = fs::read_to_string(&path)?.trim().to_string();
        if !token.is_empty() {
            info!("Using auth token from {}", path.display());
            return Ok(token);
        }
    }

    // 3. Generate new token and persist
    let token = uuid::Uuid::new_v4().to_string();

    // Ensure parent directory exists with restricted permissions
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;

        // Restrict parent directory permissions on Unix (owner only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(parent, fs::Permissions::from_mode(0o700));
        }
    }

    // Write token file with restricted permissions from the start.
    // On Unix, we use OpenOptionsExt to set mode 0o600 at creation time,
    // avoiding the TOCTOU race of write-then-chmod.
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;

        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600) // Owner read/write only — set at creation, no race window
            .open(&path)?;
        file.write_all(token.as_bytes())?;
        file.sync_all()?;
    }

    // On non-Unix (Windows), fall back to regular write (Windows ACLs are different)
    #[cfg(not(unix))]
    {
        fs::write(&path, &token)?;
    }

    info!("Generated new auth token at {}", path.display());
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_data_dir_wins_over_env_and_default() {
        let dir = PathBuf::from("/tmp/vestige-cli");
        let got = resolve_token_dir(Some(&dir), Some(OsString::from("/tmp/vestige-env"))).unwrap();
        assert_eq!(got, dir);
    }

    #[test]
    fn env_data_dir_is_used_without_an_explicit_dir() {
        let got = resolve_token_dir(None, Some(OsString::from("/tmp/vestige-env"))).unwrap();
        assert_eq!(got, PathBuf::from("/tmp/vestige-env"));
    }

    #[test]
    fn blank_env_data_dir_falls_back_to_platform_default() {
        let blank = resolve_token_dir(None, Some(OsString::new())).unwrap();
        let unset = resolve_token_dir(None, None).unwrap();
        assert_eq!(blank, unset);
    }
}
