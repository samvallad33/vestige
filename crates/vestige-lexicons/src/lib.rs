//! Bundled, sha256-pinned static lexical resources isolated into a dedicated leaf crate
//! so incremental rebuilds of `vestige-core` and `vestige-mcp` never re-compile or
//! re-embed 35.5 MiB of static dictionary tables.

/// Princeton WordNet 3.1 full 4-POS lexical relations (bundled data files, WordNet license).
#[path = "../../vestige-core/src/wordnet.rs"]
pub mod wordnet;
/// Roget's 1911 Thesaurus of English Words and Phrases (Project Gutenberg #10681, sha256-pinned).
#[path = "../../vestige-core/src/roget.rs"]
pub mod roget;
/// GeoNames geographical gazetteer (CC BY 4.0, sha256-pinned).
#[path = "../../vestige-core/src/geonames.rs"]
pub mod geonames;
/// Packed ConceptNet 5.7.0 2-hop commonsense bridge (sha256-pinned, Issue #817).
#[path = "../../vestige-core/src/conceptnet.rs"]
pub mod conceptnet;
/// Packed Wikidata CC0 1.0 topological containment & composition graph (sha256-pinned).
#[path = "../../vestige-core/src/wikidata.rs"]
pub mod wikidata;
