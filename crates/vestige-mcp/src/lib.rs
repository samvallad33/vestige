//! Vestige MCP Server Library
//!
//! Shared modules accessible to all binaries in the crate.

pub mod autopilot;
pub mod cognitive;
pub mod dashboard;
pub mod protocol;
pub mod resources;
pub mod server;
pub mod tools;
pub mod trace_recorder;

/// Whether this binary was compiled with an embedding runtime and a vector
/// index at all. Builds without them (the Android/Termux profile, #145) are
/// valid builds, and every status surface must say "built without embeddings"
/// where it would otherwise look like a runtime that failed to start.
pub const fn embeddings_compiled_in() -> bool {
    cfg!(all(feature = "embeddings", feature = "vector-search"))
}
