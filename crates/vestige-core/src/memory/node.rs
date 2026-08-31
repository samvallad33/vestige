//! Knowledge Node - The fundamental unit of memory
//!
//! Each node represents a discrete piece of knowledge with:
//! - Content and metadata
//! - FSRS-6 scheduling state
//! - Dual-strength retention model
//! - Temporal validity (bi-temporal)
//! - Embedding metadata

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ============================================================================
// NODE TYPES
// ============================================================================

/// Types of knowledge nodes
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum NodeType {
    /// A discrete fact or piece of information
    #[default]
    Fact,
    /// A concept or abstract idea
    Concept,
    /// A procedure or how-to knowledge
    Procedure,
    /// An event or experience
    Event,
    /// A relationship between entities
    Relationship,
    /// A quote or verbatim text
    Quote,
    /// Code or technical snippet
    Code,
    /// A question to be answered
    Question,
    /// User insight or reflection
    Insight,
}

impl NodeType {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            NodeType::Fact => "fact",
            NodeType::Concept => "concept",
            NodeType::Procedure => "procedure",
            NodeType::Event => "event",
            NodeType::Relationship => "relationship",
            NodeType::Quote => "quote",
            NodeType::Code => "code",
            NodeType::Question => "question",
            NodeType::Insight => "insight",
        }
    }

    /// Parse from string name
    pub fn parse_name(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "fact" => NodeType::Fact,
            "concept" => NodeType::Concept,
            "procedure" => NodeType::Procedure,
            "event" => NodeType::Event,
            "relationship" => NodeType::Relationship,
            "quote" => NodeType::Quote,
            "code" => NodeType::Code,
            "question" => NodeType::Question,
            "insight" => NodeType::Insight,
            _ => NodeType::Fact,
        }
    }
}

impl std::fmt::Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// SOURCE ENVELOPE (#57)
// ============================================================================

/// Structured provenance for a memory that mirrors a record in an external
/// system of record (a Redmine issue, a GitHub Issue, a Jira ticket, a support
/// thread, …).
///
/// The product boundary (#57): the external system stays canonical. Vestige
/// **indexes, connects, retrieves, and cites back**; it does not replace the
/// ticket tracker. This envelope carries exactly the fields a connector needs
/// to do that without leaking stale data:
///
/// - `(source_system, source_id)` is the **idempotency key**. Re-running a sync
///   upserts the same logical record instead of duplicating it.
/// - `content_hash` is the **change detector**. If a re-fetched record hashes
///   to the stored value, the upsert is a no-op (only `synced_at` advances),
///   so an incremental re-scan never churns the index or the embedding model.
/// - `source_url` is the **citation**. Search results link back to the
///   canonical record so the agent can follow it for authoritative detail.
/// - `source_updated_at` is the **cursor field** the connector checkpoints on.
///
/// Every field is optional at the type level so partial connectors and manual
/// imports can populate only what they have, but a real connector should always
/// set `source_system`, `source_id`, and `content_hash`.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceEnvelope {
    /// External system this record came from, e.g. `redmine`, `github`, `jira`.
    /// Namespaces `source_id` so two systems can share numeric ids.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_system: Option<String>,
    /// Stable native id in the source system (Redmine issue id, GitHub issue
    /// number/node id, …). Combined with `source_system` it is the upsert key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    /// Canonical URL back to the record so retrieval can cite the source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
    /// When the source record was last updated upstream (the connector cursor
    /// field — Redmine `updated_on`, GitHub `updated_at`). RFC 3339.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_updated_at: Option<DateTime<Utc>>,
    /// Stable hash of the normalized record content. Idempotency / change key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    /// When the connector last observed this record live. Drives tombstone
    /// reconciliation (a record not seen in a full reconcile pass is gone).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synced_at: Option<DateTime<Utc>>,
    /// Project / repo / space the record belongs to (Redmine project, GitHub
    /// `owner/repo`). Used for scoped sync and search filters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_project: Option<String>,
    /// Record type within the source (`issue`, `comment`, `journal`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_type: Option<String>,
    /// Author / reporter of the record in the source system.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_author: Option<String>,
}

impl SourceEnvelope {
    /// True once the two fields a connector needs for idempotent upsert are
    /// present. Manual imports that only set `source_url` are not "keyed".
    pub fn has_key(&self) -> bool {
        self.source_system.is_some() && self.source_id.is_some()
    }

    /// True if every field is unset — used to collapse an all-`None` envelope
    /// back to `None` on the node so legacy rows stay clean.
    pub fn is_empty(&self) -> bool {
        self.source_system.is_none()
            && self.source_id.is_none()
            && self.source_url.is_none()
            && self.source_updated_at.is_none()
            && self.content_hash.is_none()
            && self.synced_at.is_none()
            && self.source_project.is_none()
            && self.source_type.is_none()
            && self.source_author.is_none()
    }
}

// ============================================================================
// KNOWLEDGE NODE
// ============================================================================

/// A knowledge node in the memory graph
///
/// Combines multiple memory science models:
/// - FSRS-6 for optimal review scheduling
/// - Bjork dual-strength for realistic forgetting
/// - Temporal validity for time-sensitive knowledge
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KnowledgeNode {
    /// Unique identifier (UUID v4)
    pub id: String,
    /// The actual content/knowledge
    pub content: String,
    /// Type of knowledge (fact, concept, procedure, etc.)
    pub node_type: String,
    /// When the node was created
    pub created_at: DateTime<Utc>,
    /// When the node was last modified
    pub updated_at: DateTime<Utc>,
    /// When the node was last accessed/reviewed
    pub last_accessed: DateTime<Utc>,

    // ========== FSRS-6 State (21 parameters) ==========
    /// Memory stability (days until 90% forgetting probability)
    pub stability: f64,
    /// Inherent difficulty (1.0 = easy, 10.0 = hard)
    pub difficulty: f64,
    /// Number of successful reviews
    pub reps: i32,
    /// Number of lapses (forgotten after learning)
    pub lapses: i32,

    // ========== Dual-Strength Model (Bjork & Bjork 1992) ==========
    /// Storage strength - accumulated with practice, never decays
    pub storage_strength: f64,
    /// Retrieval strength - current accessibility, decays over time
    pub retrieval_strength: f64,
    /// Combined retention score (0.0 - 1.0)
    pub retention_strength: f64,

    // ========== Emotional Memory ==========
    /// Sentiment polarity (-1.0 to 1.0)
    pub sentiment_score: f64,
    /// Sentiment intensity (0.0 to 1.0) - affects stability
    pub sentiment_magnitude: f64,

    // ========== Scheduling ==========
    /// Next scheduled review date
    pub next_review: Option<DateTime<Utc>>,

    // ========== Provenance ==========
    /// Source of the knowledge (URL, file, conversation, etc.)
    pub source: Option<String>,
    /// Tags for categorization
    pub tags: Vec<String>,

    // ========== Temporal Memory (Bi-temporal) ==========
    /// When this knowledge became valid
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<DateTime<Utc>>,
    /// When this knowledge stops being valid
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<DateTime<Utc>>,

    // ========== Utility Tracking (MemRL v1.9.0) ==========
    /// Utility score = times_useful / times_retrieved (0.0 to 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub utility_score: Option<f64>,
    /// Number of times this memory was retrieved in search
    #[serde(skip_serializing_if = "Option::is_none")]
    pub times_retrieved: Option<i32>,
    /// Number of times this memory was subsequently useful
    #[serde(skip_serializing_if = "Option::is_none")]
    pub times_useful: Option<i32>,

    // ========== Emotional Memory (v2.0.0) ==========
    /// Emotional valence: -1.0 (negative) to 1.0 (positive)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotional_valence: Option<f64>,
    /// Flashbulb memory flag: ultra-high-fidelity encoding
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flashbulb: Option<bool>,

    // ========== Temporal Hierarchy (v2.0.0) ==========
    /// Temporal level for summary nodes: None=leaf, "daily"/"weekly"/"monthly"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temporal_level: Option<String>,

    // ========== Semantic Embedding ==========
    /// Whether this node has an embedding vector
    #[serde(skip_serializing_if = "Option::is_none")]
    pub has_embedding: Option<bool>,
    /// Which model generated the embedding
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,

    // ========== Active Forgetting (v2.0.5, Anderson 2025 + Davis Rac1) ==========
    /// Top-down suppression count — compounds with each `suppress` call
    /// (Suppression-Induced Forgetting, Anderson 2025).
    #[serde(default)]
    pub suppression_count: i32,
    /// Timestamp of the most recent suppression (for 24h labile window).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suppressed_at: Option<DateTime<Utc>>,

    // ========== Source Envelope (#57, external-source connectors) ==========
    /// Structured provenance for memories ingested from an external system
    /// (Redmine, GitHub Issues, Jira, …). `None` for memories created directly
    /// by an agent or the user — the legacy free-form `source` string above
    /// remains the human-readable label; this envelope is the machine-readable,
    /// idempotency- and citation-bearing record. See [`SourceEnvelope`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_envelope: Option<SourceEnvelope>,
}

impl Default for KnowledgeNode {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: String::new(),
            content: String::new(),
            node_type: "fact".to_string(),
            created_at: now,
            updated_at: now,
            last_accessed: now,
            stability: 2.5,
            difficulty: 5.0,
            reps: 0,
            lapses: 0,
            storage_strength: 1.0,
            retrieval_strength: 1.0,
            retention_strength: 1.0,
            sentiment_score: 0.0,
            sentiment_magnitude: 0.0,
            next_review: None,
            source: None,
            tags: vec![],
            valid_from: None,
            valid_until: None,
            utility_score: None,
            times_retrieved: None,
            times_useful: None,
            emotional_valence: None,
            flashbulb: None,
            temporal_level: None,
            has_embedding: None,
            embedding_model: None,
            suppression_count: 0,
            suppressed_at: None,
            source_envelope: None,
        }
    }
}

impl KnowledgeNode {
    /// Create a new knowledge node with the given content
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            ..Default::default()
        }
    }

    /// Check if this node is currently valid (within temporal bounds)
    pub fn is_valid_at(&self, time: DateTime<Utc>) -> bool {
        let after_start = self.valid_from.map(|t| time >= t).unwrap_or(true);
        let before_end = self.valid_until.map(|t| time <= t).unwrap_or(true);
        after_start && before_end
    }

    /// Check if this node is currently valid (now)
    pub fn is_currently_valid(&self) -> bool {
        self.is_valid_at(Utc::now())
    }

    /// Check if this node is due for review
    pub fn is_due(&self) -> bool {
        self.next_review.map(|t| t <= Utc::now()).unwrap_or(true)
    }

    /// Get the parsed node type
    pub fn get_node_type(&self) -> NodeType {
        NodeType::parse_name(&self.node_type)
    }
}

// ============================================================================
// INPUT TYPES
// ============================================================================

/// Input for creating a new memory
///
/// Uses `deny_unknown_fields` to prevent field injection attacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct IngestInput {
    /// The content to memorize
    pub content: String,
    /// Type of knowledge (fact, concept, procedure, etc.)
    pub node_type: String,
    /// Source of the knowledge
    pub source: Option<String>,
    /// Sentiment polarity (-1.0 to 1.0)
    #[serde(default)]
    pub sentiment_score: f64,
    /// Sentiment intensity (0.0 to 1.0)
    #[serde(default)]
    pub sentiment_magnitude: f64,
    /// Tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
    /// When this knowledge becomes valid
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<DateTime<Utc>>,
    /// When this knowledge stops being valid
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<DateTime<Utc>>,
    /// True when `valid_from` was inferred from prose (an "as of YYYY-MM-DD"
    /// phrase) rather than supplied explicitly by the caller. Inferred
    /// validity may stamp a newly created node but must never rewrite an
    /// existing node's window. Internal flag set by smart ingest; never part
    /// of the wire format.
    #[serde(skip)]
    pub validity_inferred: bool,
    /// Structured provenance for connector-ingested records (#57). When set
    /// with a `(source_system, source_id)` key, callers should route through
    /// `upsert_by_source` for idempotent sync rather than plain `ingest`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_envelope: Option<SourceEnvelope>,
}

impl Default for IngestInput {
    fn default() -> Self {
        Self {
            content: String::new(),
            node_type: "fact".to_string(),
            source: None,
            sentiment_score: 0.0,
            sentiment_magnitude: 0.0,
            tags: vec![],
            valid_from: None,
            valid_until: None,
            validity_inferred: false,
            source_envelope: None,
        }
    }
}

/// Search mode for recall queries
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum SearchMode {
    /// Keyword search only (FTS5/BM25)
    Keyword,
    /// Semantic search only (embeddings)
    Semantic,
    /// Hybrid search with RRF fusion (default, best results)
    #[default]
    Hybrid,
}

/// Input for recalling memories
///
/// Uses `deny_unknown_fields` to prevent field injection attacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RecallInput {
    /// Search query
    pub query: String,
    /// Maximum results to return
    pub limit: i32,
    /// Minimum retention strength (0.0 to 1.0)
    #[serde(default)]
    pub min_retention: f64,
    /// Search mode (keyword, semantic, or hybrid)
    #[serde(default)]
    pub search_mode: SearchMode,
    /// Only return results valid at this time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_at: Option<DateTime<Utc>>,
}

impl Default for RecallInput {
    fn default() -> Self {
        Self {
            query: String::new(),
            limit: 10,
            min_retention: 0.0,
            search_mode: SearchMode::Hybrid,
            valid_at: None,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_type_roundtrip() {
        for node_type in [
            NodeType::Fact,
            NodeType::Concept,
            NodeType::Procedure,
            NodeType::Event,
            NodeType::Code,
        ] {
            assert_eq!(NodeType::parse_name(node_type.as_str()), node_type);
        }
    }

    #[test]
    fn test_knowledge_node_default() {
        let node = KnowledgeNode::default();
        assert!(node.id.is_empty());
        assert_eq!(node.node_type, "fact");
        assert!(node.is_due());
        assert!(node.is_currently_valid());
    }

    #[test]
    fn test_temporal_validity() {
        let mut node = KnowledgeNode::default();
        let now = Utc::now();

        // No bounds = always valid
        assert!(node.is_valid_at(now));

        // Set future valid_from = not valid now
        node.valid_from = Some(now + chrono::Duration::days(1));
        assert!(!node.is_valid_at(now));

        // Set past valid_from = valid now
        node.valid_from = Some(now - chrono::Duration::days(1));
        assert!(node.is_valid_at(now));

        // Set past valid_until = not valid now
        node.valid_until = Some(now - chrono::Duration::hours(1));
        assert!(!node.is_valid_at(now));
    }

    #[test]
    fn test_ingest_input_deny_unknown_fields() {
        // Valid input should parse
        let json = r#"{"content": "test", "nodeType": "fact", "tags": []}"#;
        let result: Result<IngestInput, _> = serde_json::from_str(json);
        assert!(result.is_ok());

        // Unknown field should fail (security feature)
        let json_with_unknown =
            r#"{"content": "test", "nodeType": "fact", "tags": [], "malicious_field": "attack"}"#;
        let result: Result<IngestInput, _> = serde_json::from_str(json_with_unknown);
        assert!(result.is_err());
    }
}
