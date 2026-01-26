//! MCP Tools
//!
//! Tool implementations for the Vestige MCP server.

pub mod codebase;
pub mod consolidate;
pub mod ingest;
pub mod intentions;
pub mod knowledge;
pub mod recall;
pub mod review;
pub mod search;
pub mod smart_ingest;
pub mod stats;

// Neuroscience-inspired tools
pub mod context;
pub mod memory_states;
pub mod tagging;

// Feedback / preference learning
pub mod feedback;

// Unified tools (consolidate multiple operations into single tools)
pub mod codebase_unified;
pub mod intention_unified;
pub mod memory_unified;
pub mod search_unified;
