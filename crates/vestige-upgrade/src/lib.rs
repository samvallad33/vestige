//! v3 importer for the `vestige-upgrade` binary.
//!
//! The main 4.0 binaries do not link this crate. They see that `vestige.db`
//! exists and run this binary. The v3 receipt cross-check lives here;
//! `strata-verify` only checks a Strata log.

mod cross_check;
mod upgrade;

pub use cross_check::verify_migrated_log;
pub use upgrade::{
    LOG_DIR_NAME, UPGRADE_LOG_NAME, UpgradeOptions, UpgradeStatus, V311_RELEASE, staging_directory,
    upgrade_if_needed, upgrade_with,
};
