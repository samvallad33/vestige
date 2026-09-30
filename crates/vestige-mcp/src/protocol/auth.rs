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

use std::fs;
use std::path::PathBuf;

use directories::ProjectDirs;
use tracing::{info, warn};

/// Minimum recommended token length when provided via env var.
const MIN_TOKEN_LENGTH: usize = 32;

/// Return the auth token file path inside the Vestige data directory.
pub fn token_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let dirs = ProjectDirs::from("com", "vestige", "core")
        .ok_or("could not determine project directories")?;
    Ok(dirs.data_dir().join("auth_token"))
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
