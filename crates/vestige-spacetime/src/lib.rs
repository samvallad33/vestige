//! Production-shape specification for Vestige Memory Spacetime Legs 3–7.
//!
//! This crate isolates the new algorithms from Vestige's existing storage and
//! MCP types. Integration adapters map `KnowledgeNode`, Engram Passport state,
//! graph records, and MCP session leases into these projections.

pub mod common;
pub mod leg3_reality_index;
pub mod leg4_reality_delta;
pub mod leg5_claim_lattice;
pub mod leg6_capabilities;
pub mod leg7_autograph;
