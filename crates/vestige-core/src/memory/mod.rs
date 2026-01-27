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

pub use node::{IngestInput, KnowledgeNode, NodeType, RecallInput, SearchMode};
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

    /// Check if the edge is currently valid
    pub fn is_valid(&self) -> bool {
        self.valid_until.is_none()
    }

    /// Check if the edge was valid at a given time
    pub fn was_valid_at(&self, time: DateTime<Utc>) -> bool {
        let after_start = self.valid_from.map_or(true, |from| time >= from);
        let before_end = self.valid_until.map_or(true, |until| time < until);
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
    /// Embedding model used (if any)
    pub embedding_model: Option<String>,
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
            embedding_model: None,
        }
    }
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

