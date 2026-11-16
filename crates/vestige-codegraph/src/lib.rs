#![forbid(unsafe_code)]

//! # `vestige-codegraph` (`#768`)
//!
//! AST-native code memory graph combining incremental `tree-sitter` parsing,
//! read-only pure-Rust `gix` history & hunk extraction, and deterministic
//! equivalence congruence (`vestige why path:line`).

pub mod congruence;
pub mod error;
pub mod git;
pub mod graph;
pub mod parse;
pub mod why;

pub use congruence::{CongruenceEngine, CongruenceRule, MergeWitness};
pub use error::CodeGraphError;
pub use git::{hash_email_pii, CommitRecord, GitWalkOptions, LineRange, Touched};
pub use graph::{CodeGraph, SymbolOccurrence};
pub use parse::{CallEdge, IncrementalParser, LanguageKind, ParsedFile, Symbol, SymbolId, SymbolKind};
pub use why::{CommitHop, WhyChain};
