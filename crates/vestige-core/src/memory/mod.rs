//! Memory module - Core types and data structures
//!
//! Implements the cognitive memory model with:
//! - Knowledge nodes with FSRS-6 scheduling state
//! - Dual-strength model (Bjork & Bjork 1992)
//! - Temporal memory with bi-temporal validity
//! - Semantic embedding metadata

mod node;
mod strength;
mod temporal;

pub use node::{IngestInput, KnowledgeNode, NodeType, RecallInput, SearchMode, SourceEnvelope};
pub use strength::{DualStrength, StrengthDecay};
pub use temporal::{TemporalRange, TemporalValidity};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ============================================================================
// GOD TIER 2026: MEMORY SCOPES (Like Mem0)
// ============================================================================

/// Memory scope - controls persistence and sharing behavior
/// Competes with Mem0's User/Session/Agent model
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "lowercase")]
pub enum MemoryScope {
    /// Per-session memory, cleared on restart (working memory)
    Session,
    /// Per-user memory, persists across sessions (long-term memory)
    #[default]
    User,
    /// Global agent knowledge, shared across all users (world knowledge)
    Agent,
}

impl std::fmt::Display for MemoryScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryScope::Session => write!(f, "session"),
            MemoryScope::User => write!(f, "user"),
            MemoryScope::Agent => write!(f, "agent"),
        }
    }
}

impl std::str::FromStr for MemoryScope {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "session" => Ok(MemoryScope::Session),
            "user" => Ok(MemoryScope::User),
            "agent" => Ok(MemoryScope::Agent),
            _ => Err(format!("Unknown scope: {}", s)),
        }
    }
}

// ============================================================================
// GOD TIER 2026: MEMORY SYSTEMS (Tulving 1972)
// ============================================================================

/// Memory system classification (based on Tulving's memory systems)
/// - Episodic: Events, conversations, specific moments (decays faster)
/// - Semantic: Facts, concepts, generalizations (stable)
/// - Procedural: How-to knowledge (never decays)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "lowercase")]
pub enum MemorySystem {
    /// What happened - events, conversations, specific moments
    /// Decays faster than semantic memories
    Episodic,
    /// What I know - facts, concepts, generalizations
    /// More stable, the default for most knowledge
    #[default]
    Semantic,
    /// How-to knowledge - skills, procedures
    /// Never decays (like riding a bike)
    Procedural,
}

impl std::fmt::Display for MemorySystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemorySystem::Episodic => write!(f, "episodic"),
            MemorySystem::Semantic => write!(f, "semantic"),
            MemorySystem::Procedural => write!(f, "procedural"),
        }
    }
}

impl std::str::FromStr for MemorySystem {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "episodic" => Ok(MemorySystem::Episodic),
            "semantic" => Ok(MemorySystem::Semantic),
            "procedural" => Ok(MemorySystem::Procedural),
            _ => Err(format!("Unknown memory system: {}", s)),
        }
    }
}

// ============================================================================
// GOD TIER 2026: KNOWLEDGE GRAPH EDGES (Like Zep's Graphiti)
// ============================================================================

/// Type of relationship between knowledge nodes
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum EdgeType {
    /// Semantically related (similar meaning/topic)
    Semantic,
    /// Temporal relationship (happened before/after)
    Temporal,
    /// Causal relationship (A caused B)
    Causal,
    /// Derived knowledge (B is derived from A)
    Derived,
    /// Contradiction (A and B conflict)
    Contradiction,
    /// Refinement (B is a more specific version of A)
    Refinement,
    /// Part-of relationship (A is part of B)
    PartOf,
    /// User-defined relationship
    Custom,
}

impl std::fmt::Display for EdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EdgeType::Semantic => write!(f, "semantic"),
            EdgeType::Temporal => write!(f, "temporal"),
            EdgeType::Causal => write!(f, "causal"),
            EdgeType::Derived => write!(f, "derived"),
            EdgeType::Contradiction => write!(f, "contradiction"),
            EdgeType::Refinement => write!(f, "refinement"),
            EdgeType::PartOf => write!(f, "part_of"),
            EdgeType::Custom => write!(f, "custom"),
        }
    }
}

impl std::str::FromStr for EdgeType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "semantic" => Ok(EdgeType::Semantic),
            "temporal" => Ok(EdgeType::Temporal),
            "causal" => Ok(EdgeType::Causal),
            "derived" => Ok(EdgeType::Derived),
            "contradiction" => Ok(EdgeType::Contradiction),
            "refinement" => Ok(EdgeType::Refinement),
            "part_of" | "partof" => Ok(EdgeType::PartOf),
            "custom" => Ok(EdgeType::Custom),
            _ => Err(format!("Unknown edge type: {}", s)),
        }
    }
}

/// A directed edge in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KnowledgeEdge {
    /// Unique edge ID
    pub id: String,
    /// Source node ID
    pub source_id: String,
    /// Target node ID
    pub target_id: String,
    /// Type of relationship
    pub edge_type: EdgeType,
    /// Edge weight (strength of relationship)
    pub weight: f32,
    /// When this relationship started being true
    pub valid_from: Option<DateTime<Utc>>,
    /// When this relationship stopped being true (None = still valid)
    pub valid_until: Option<DateTime<Utc>>,
    /// When the edge was created
    pub created_at: DateTime<Utc>,
    /// Who/what created the edge
    pub created_by: Option<String>,
    /// Confidence in this relationship (0-1)
    pub confidence: f32,
    /// Additional metadata as JSON
    pub metadata: Option<String>,
}

impl KnowledgeEdge {
    /// Create a new knowledge edge
    pub fn new(source_id: String, target_id: String, edge_type: EdgeType) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            source_id,
            target_id,
            edge_type,
            weight: 1.0,
            valid_from: Some(chrono::Utc::now()),
            valid_until: None,
            created_at: chrono::Utc::now(),
            created_by: None,
            confidence: 1.0,
            metadata: None,
        }
    }

    /// Check if the edge is currently valid (within its [valid_from, valid_until)
    /// window). Delegates to the bi-temporal check so a not-yet-active edge
    /// (future valid_from) or an expired one is correctly reported invalid —
    /// the old version only checked valid_until and ignored valid_from.
    pub fn is_valid(&self) -> bool {
        self.was_valid_at(chrono::Utc::now())
    }

    /// Check if the edge was valid at a given time
    pub fn was_valid_at(&self, time: DateTime<Utc>) -> bool {
        let after_start = self.valid_from.is_none_or(|from| time >= from);
        let before_end = self.valid_until.is_none_or(|until| time < until);
        after_start && before_end
    }
}

// ============================================================================
// MEMORY STATISTICS
// ============================================================================

/// Statistics about the memory system
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStats {
    /// Total number of knowledge nodes
    pub total_nodes: i64,
    /// Nodes currently due for review
    pub nodes_due_for_review: i64,
    /// Average retention strength across all nodes
    pub average_retention: f64,
    /// Average storage strength (Bjork model)
    pub average_storage_strength: f64,
    /// Average retrieval strength (Bjork model)
    pub average_retrieval_strength: f64,
    /// Timestamp of the oldest memory
    pub oldest_memory: Option<DateTime<Utc>>,
    /// Timestamp of the newest memory
    pub newest_memory: Option<DateTime<Utc>>,
    /// Number of nodes with semantic embeddings
    pub nodes_with_embeddings: i64,
    /// Number of nodes with embeddings generated by the active embedding model family
    pub nodes_with_active_embeddings: i64,
    /// Number of nodes whose stored embeddings belong to a different model family
    pub nodes_with_mismatched_embeddings: i64,
    /// Embedding model used (if any)
    pub embedding_model: Option<String>,
    /// Embedding model family currently configured for new queries and writes
    pub active_embedding_model: Option<String>,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            total_nodes: 0,
            nodes_due_for_review: 0,
            average_retention: 0.0,
            average_storage_strength: 0.0,
            average_retrieval_strength: 0.0,
            oldest_memory: None,
            newest_memory: None,
            nodes_with_embeddings: 0,
            nodes_with_active_embeddings: 0,
            nodes_with_mismatched_embeddings: 0,
            embedding_model: None,
            active_embedding_model: None,
        }
    }
}

// ============================================================================
// SCHEMA INTROSPECTION (v2.1.24+: surfaces DB shape to MCP consumers)
// ============================================================================

/// A single SQLite table's introspected shape: name, row count, column list.
///
/// Returned as part of `SchemaIntrospection` from `Storage::schema_introspection()`.
/// Consumers needing more depth (e.g. per-column NULL counts) should request
/// targeted methods rather than expecting this struct to grow unboundedly —
/// the row + column shape covered here is the 80% case for audit / migration
/// guard scripts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TableIntrospection {
    /// SQLite table name.
    pub name: String,
    /// Row count.
    pub rows: i64,
    /// Column names in declaration order.
    pub columns: Vec<String>,
}

/// Result of `Storage::schema_introspection()`. Snapshots the schema version,
/// migration timestamp, and a row/column view of every user-data table.
///
/// Motivation: external consumers (audit scripts, migration guards, downstream
/// upgrade scripts) currently must read SQLite directly to learn the schema
/// version and table shape, which couples them to internal layout. This struct
/// gives them a first-class MCP-callable surface. The list of tables walked is
/// intentionally the same canonical set used elsewhere in storage (the user-
/// data tables) so the surface stays stable across migrations.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SchemaIntrospection {
    /// Current schema version (highest applied migration; matches the
    /// `schema_version` table's MAX(version)).
    pub schema_version: u32,
    /// When the current schema version was applied (RFC3339), if known.
    pub schema_version_applied_at: Option<DateTime<Utc>>,
    /// Per-table introspection rows.
    pub tables: Vec<TableIntrospection>,
    /// Total number of nodes whose `embeddings.embedding` is NULL (i.e., have
    /// no embedding row). Convenience field for embedding-coverage audits;
    /// equivalent to (knowledge_nodes.rows − rows in `embeddings` joined to
    /// knowledge_nodes), so consumers don't have to compute it themselves.
    pub embedding_null_count: i64,
    /// Active embedding model name (mirrors `MemoryStats.active_embedding_model`).
    /// Useful when an audit script wants schema_version + active model in one call.
    pub active_embedding_model: Option<String>,
    /// Embedding dimensions for the active model, if known.
    pub active_embedding_dimensions: Option<u32>,
}

// ============================================================================
// CONSOLIDATION RESULT
// ============================================================================

/// Result of a memory consolidation run (sleep-inspired processing)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[derive(Default)]
pub struct ConsolidationResult {
    /// Number of nodes processed
    pub nodes_processed: i64,
    /// Nodes promoted due to high importance/emotion
    pub nodes_promoted: i64,
    /// Nodes pruned due to low retention
    pub nodes_pruned: i64,
    /// Number of nodes with decay applied
    pub decay_applied: i64,
    /// Processing duration in milliseconds
    pub duration_ms: i64,
    /// Number of embeddings generated
    pub embeddings_generated: i64,
    // v1.4.0: FSRS-6 upgrade
    /// Number of duplicate memories merged during episodic→semantic consolidation
    pub duplicates_merged: i64,
    /// Number of neighbor memories reinforced (tracked per-access, not consolidation)
    pub neighbors_reinforced: i64,
    /// Number of ACT-R activations computed from access history
    pub activations_computed: i64,
    /// Personalized w20 if optimized this cycle
    pub w20_optimized: Option<f64>,
}

// ============================================================================
// SEARCH RESULTS
// ============================================================================

/// Enhanced search result with relevance scores
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResult {
    /// The matched knowledge node
    pub node: KnowledgeNode,
    /// Keyword (BM25/FTS5) score if matched
    pub keyword_score: Option<f32>,
    /// Semantic (embedding) similarity if matched
    pub semantic_score: Option<f32>,
    /// Combined score after RRF fusion
    pub combined_score: f32,
    /// How the result was matched
    pub match_type: MatchType,
}

/// How a search result was matched
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum MatchType {
    /// Matched via keyword (BM25/FTS5) search only
    Keyword,
    /// Matched via semantic (embedding) search only
    Semantic,
    /// Matched via both keyword and semantic search
    Both,
}

/// Semantic similarity search result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimilarityResult {
    /// The matched knowledge node
    pub node: KnowledgeNode,
    /// Cosine similarity score (0.0 to 1.0)
    pub similarity: f32,
}

// ============================================================================
// EMBEDDING RESULT
// ============================================================================

/// Result of embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[derive(Default)]
pub struct EmbeddingResult {
    /// Successfully generated embeddings
    pub successful: i64,
    /// Failed embedding generations
    pub failed: i64,
    /// Skipped (already had embeddings)
    pub skipped: i64,
    /// Error messages for failures
    pub errors: Vec<String>,
}
