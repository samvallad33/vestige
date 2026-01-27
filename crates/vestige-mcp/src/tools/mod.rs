//! MCP Tools
//!
//! Tool implementations for the Vestige MCP server.
//!
//! The unified tools (codebase_unified, intention_unified, memory_unified, search_unified)
//! are the primary API. The granular tools below are kept for backwards compatibility
//! but are not exposed in the MCP tool list.

// Active unified tools
pub mod codebase_unified;
pub mod ingest;
pub mod intention_unified;
pub mod memory_unified;
pub mod search_unified;
pub mod smart_ingest;

// Deprecated tools - kept for internal backwards compatibility
// These modules are intentionally unused in the public API
#[allow(dead_code)]
pub mod codebase;
#[allow(dead_code)]
pub mod consolidate;
#[allow(dead_code)]
pub mod context;
#[allow(dead_code)]
pub mod feedback;
#[allow(dead_code)]
pub mod intentions;
#[allow(dead_code)]
pub mod knowledge;
#[allow(dead_code)]
pub mod memory_states;
#[allow(dead_code)]
pub mod recall;
#[allow(dead_code)]
pub mod review;
#[allow(dead_code)]
pub mod search;
#[allow(dead_code)]
pub mod stats;
#[allow(dead_code)]
pub mod tagging;
