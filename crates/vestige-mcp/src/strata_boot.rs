//! STRATA boot (build/wire-strata): the wiped-mode runtime store.
//!
//! `vestige-core` must not depend on strata crates, so the engine is
//! injected: this module installs `strata_bridge::strata_constructor` into
//! `vestige_core::storage::STORE_CONSTRUCTOR` (the hook the feature-off
//! `open_storage` consults) and opens the STRATA store directly for the MCP
//! server's boot path.
//!
//! Directory resolution: `$VESTIGE_STRATA_DIR` wins; otherwise the store
//! lives in `<data dir>/strata`, where the data dir follows the same
//! convention as the SQLite backend (`--data-dir` / `$VESTIGE_DATA_DIR` /
//! the platform project data dir).
//!
//! NOTE on feature unification: workspace builds link vestige-core with
//! `legacy-sqlite` unified ON (via the strata-migrate path dep), so
//! `vestige_core::open_storage` alone cannot decide the mode — the MCP
//! binary's own `legacy-sqlite` feature is authoritative, and
//! [`open_strata_storage`] is what `main.rs` calls in `not(feature =
//! "legacy-sqlite")` builds.

use std::path::PathBuf;
use std::sync::Arc;

use vestige_core::storage::{Storage, install_store_constructor};

/// Open the STRATA store for the MCP server (wiped mode).
///
/// Installs the constructor hook first so every later `open_storage` call in
/// this process that reaches a legacy-free core also lands on STRATA, then
/// opens the store through the same constructor the hook carries.
///
/// `db_path` uses the SQLite boot convention (`<data dir>/vestige.db`, or
/// `None` for the default data dir); the STRATA directory is resolved from
/// it (see [`strata_bridge::strata_dir_for`]).
pub fn open_strata_storage(db_path: Option<PathBuf>) -> Result<Arc<Storage>, String> {
    install_store_constructor(strata_bridge::strata_constructor);
    let path = db_path.unwrap_or_else(default_db_path);
    strata_bridge::strata_constructor(path)
}

/// Default DB-file-shaped path (data dir + `vestige.db`), mirroring the
/// SQLite backend's default-location convention so both modes agree on
/// where the data dir is.
fn default_db_path() -> PathBuf {
    if let Some(env) = std::env::var_os("VESTIGE_DATA_DIR") {
        if !env.is_empty() {
            return PathBuf::from(env).join("vestige.db");
        }
    }
    directories::ProjectDirs::from("com", "vestige", "core")
        .map(|dirs| dirs.data_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
        .join("vestige.db")
}
