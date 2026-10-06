//! Vestige MCP Server Library
//!
//! Shared modules accessible to all binaries in the crate.
//!
//! Every module that touches the legacy SQLite store (`Storage`) is
//! quarantined behind the `legacy-sqlite` feature (build/t5-legacy-isolation,
//! mirroring vestige-core). Default ON; flips off with the STRATA backend.
//! A `legacy-sqlite`-free build still compiles every binary — they open
//! storage through `vestige_core::open_storage()` and exit with its clear
//! `LegacySqliteDisabled` error at runtime.

pub mod actor_surface;
pub mod attach;
pub mod auto_connect;
pub mod autopilot;
pub mod cognitive;
pub mod dashboard;
pub mod intake;
mod intention_graph_log;
pub mod protocol;
#[cfg(all(test, not(feature = "legacy-sqlite")))]
mod protocol_stdio_store;
pub mod resources;
pub mod server;
pub mod strata_memory;
pub mod tools;
pub mod trace_recorder;
pub mod v3_launch;

/// Whether this binary was compiled with an embedding runtime and a vector
/// index at all. Builds without them (the Android/Termux profile, #145) are
/// valid builds, and every status surface must say "built without embeddings"
/// where it would otherwise look like a runtime that failed to start.
pub const fn embeddings_compiled_in() -> bool {
    cfg!(vestige_embeddings_removed)
}
