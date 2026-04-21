//! Backend-agnostic memory store trait.
//!
//! This is the single abstraction every cognitive module sits above. It is
//! intentionally flat: one trait, ~25 methods, no sub-traits.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ----------------------------------------------------------------------------
// ERROR
// ----------------------------------------------------------------------------

/// Error returned by every `LocalMemoryStore` / `MemoryStore` method.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum MemoryStoreError {
    #[error("not found: {0}")]
    NotFound(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error(
        "embedding model mismatch: store registered {registered_name} (dim {registered_dim}, \
         hash {registered_hash}), embedder is {actual_name} (dim {actual_dim}, hash {actual_hash})"
    )]
    ModelMismatch {
        registered_name: String,
        registered_dim: usize,
        registered_hash: String,
        actual_name: String,
        actual_dim: usize,
        actual_hash: String,
    },

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("initialization error: {0}")]
    Init(String),
}

impl From<crate::storage::StorageError> for MemoryStoreError {
    fn from(e: crate::storage::StorageError) -> Self {
        use crate::storage::StorageError as S;
        match e {
            S::NotFound(s) => MemoryStoreError::NotFound(s),
            S::Database(e) => MemoryStoreError::Backend(e.to_string()),
            S::Io(e) => MemoryStoreError::Backend(e.to_string()),
            S::InvalidTimestamp(s) => MemoryStoreError::Backend(format!("invalid timestamp: {s}")),
            S::Init(s) => MemoryStoreError::Init(s),
        }
    }
}

pub type MemoryStoreResult<T> = std::result::Result<T, MemoryStoreError>;

// ----------------------------------------------------------------------------
// DATA TYPES
// ----------------------------------------------------------------------------

/// Backend-agnostic memory record.
///
/// Phase 1 intentionally keeps this type independent of `KnowledgeNode` to
/// avoid dragging 30+ legacy fields through the trait surface. The SQLite
/// backend converts between `MemoryRecord` and `KnowledgeNode` at the
/// boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: Uuid,
    /// Empty = unclassified. Populated in Phase 4.
    pub domains: Vec<String>,
    /// Raw similarity per domain centroid. Empty until Phase 4 runs clustering.
    pub domain_scores: HashMap<String, f64>,
    pub content: String,
    pub node_type: String,
    pub tags: Vec<String>,
    pub embedding: Option<Vec<f32>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

/// FSRS-6 scheduling state, one row per memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingState {
    pub memory_id: Uuid,
    pub stability: f64,
    pub difficulty: f64,
    pub retrievability: f64,
    pub last_review: Option<DateTime<Utc>>,
    pub next_review: Option<DateTime<Utc>>,
    pub reps: u32,
    pub lapses: u32,
}

/// Hybrid search request.
#[derive(Debug, Clone, Default)]
pub struct SearchQuery {
    pub domains: Option<Vec<String>>,
    pub text: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub tags: Option<Vec<String>>,
    pub node_types: Option<Vec<String>>,
    pub limit: usize,
    pub min_retrievability: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub record: MemoryRecord,
    pub score: f64,
    pub fts_score: Option<f64>,
    pub vector_score: Option<f64>,
}

/// Edge in the spreading-activation graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEdge {
    pub source_id: Uuid,
    pub target_id: Uuid,
    pub edge_type: String,
    pub weight: f64,
    pub created_at: DateTime<Utc>,
}

/// A topical domain (populated in Phase 4). Phase 1 only needs the type to
/// shape the trait surface; discover/classify are Phase 4 work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Domain {
    pub id: String,
    pub label: String,
    pub centroid: Vec<f32>,
    pub top_terms: Vec<String>,
    pub memory_count: usize,
    pub created_at: DateTime<Utc>,
}

/// Result of classifying one vector against all known domains.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub scores: HashMap<String, f64>,
    pub domains: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StoreStats {
    pub total_memories: usize,
    pub memories_with_embeddings: usize,
    pub total_edges: usize,
    pub total_domains: usize,
    pub registered_model_name: Option<String>,
    pub registered_model_dim: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded { reason: String },
    Unavailable { reason: String },
}

// ----------------------------------------------------------------------------
// EMBEDDING MODEL SIGNATURE
// ----------------------------------------------------------------------------

/// Snapshot of the embedding model that was used to write vectors into the
/// store. Persisted in the `embedding_model` table; compared on every write
/// before the vector is accepted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSignature {
    pub name: String,
    pub dimension: usize,
    /// Lowercase hex-encoded blake3 hash, 64 chars.
    pub hash: String,
}

// ----------------------------------------------------------------------------
// TRAIT
// ----------------------------------------------------------------------------

/// The single storage abstraction.
///
/// `#[async_trait::async_trait]` makes every `async fn` return a
/// `Pin<Box<dyn Future + Send>>`, which is required for `Arc<dyn MemoryStore>`
/// to be movable across `tokio::spawn` boundaries.
///
/// `LocalMemoryStore` is a type alias kept for source compatibility with code
/// that refers to the non-send variant. In Phase 1 both names refer to the same
/// (dyn-compatible, Send-safe) trait.
#[async_trait::async_trait]
pub trait MemoryStore: Send + Sync + 'static {
    // --- Lifecycle ---
    async fn init(&self) -> MemoryStoreResult<()>;
    async fn health_check(&self) -> MemoryStoreResult<HealthStatus>;

    // --- Embedding model registry ---
    async fn registered_model(&self) -> MemoryStoreResult<Option<ModelSignature>>;
    async fn register_model(&self, sig: &ModelSignature) -> MemoryStoreResult<()>;

    // --- CRUD ---
    async fn insert(&self, record: &MemoryRecord) -> MemoryStoreResult<Uuid>;
    async fn get(&self, id: Uuid) -> MemoryStoreResult<Option<MemoryRecord>>;
    async fn update(&self, record: &MemoryRecord) -> MemoryStoreResult<()>;
    async fn delete(&self, id: Uuid) -> MemoryStoreResult<()>;

    // --- Search ---
    async fn search(&self, query: &SearchQuery) -> MemoryStoreResult<Vec<SearchResult>>;
    async fn fts_search(&self, text: &str, limit: usize) -> MemoryStoreResult<Vec<SearchResult>>;
    async fn vector_search(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> MemoryStoreResult<Vec<SearchResult>>;

    // --- FSRS Scheduling ---
    async fn get_scheduling(
        &self,
        memory_id: Uuid,
    ) -> MemoryStoreResult<Option<SchedulingState>>;
    async fn update_scheduling(&self, state: &SchedulingState) -> MemoryStoreResult<()>;
    async fn get_due_memories(
        &self,
        before: DateTime<Utc>,
        limit: usize,
    ) -> MemoryStoreResult<Vec<(MemoryRecord, SchedulingState)>>;

    // --- Graph (spreading activation) ---
    async fn add_edge(&self, edge: &MemoryEdge) -> MemoryStoreResult<()>;
    async fn get_edges(
        &self,
        node_id: Uuid,
        edge_type: Option<&str>,
    ) -> MemoryStoreResult<Vec<MemoryEdge>>;
    async fn remove_edge(&self, source: Uuid, target: Uuid) -> MemoryStoreResult<()>;
    async fn get_neighbors(
        &self,
        node_id: Uuid,
        depth: usize,
    ) -> MemoryStoreResult<Vec<(MemoryRecord, f64)>>;

    // --- Domains (Phase 1: stubs return empty; full impl in Phase 4) ---
    async fn list_domains(&self) -> MemoryStoreResult<Vec<Domain>>;
    async fn get_domain(&self, id: &str) -> MemoryStoreResult<Option<Domain>>;
    async fn upsert_domain(&self, domain: &Domain) -> MemoryStoreResult<()>;
    async fn delete_domain(&self, id: &str) -> MemoryStoreResult<()>;
    /// Phase 1: returns `Ok(vec![])` since no centroids exist. Phase 4 wires
    /// the full soft-assignment pass.
    async fn classify(&self, embedding: &[f32]) -> MemoryStoreResult<Vec<(String, f64)>>;

    // --- Bulk / Maintenance ---
    async fn count(&self) -> MemoryStoreResult<usize>;
    async fn get_stats(&self) -> MemoryStoreResult<StoreStats>;
    async fn vacuum(&self) -> MemoryStoreResult<()>;
}

/// Type alias kept for source compatibility. Both names refer to the same
/// `async_trait`-annotated trait that is dyn-compatible and `Send + Sync`.
pub use MemoryStore as LocalMemoryStore;

// ----------------------------------------------------------------------------
// UNIT TESTS
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::StorageError;

    #[test]
    fn memory_store_error_from_storage_error() {
        let se = StorageError::NotFound("abc".to_string());
        let mse = MemoryStoreError::from(se);
        assert!(matches!(mse, MemoryStoreError::NotFound(_)));

        let se2 = StorageError::Init("init failure".to_string());
        let mse2 = MemoryStoreError::from(se2);
        assert!(matches!(mse2, MemoryStoreError::Init(_)));
    }

    #[test]
    fn model_signature_serde_round_trip() {
        let sig = ModelSignature {
            name: "nomic-ai/nomic-embed-text-v1.5".to_string(),
            dimension: 256,
            hash: "a".repeat(64),
        };
        let json = serde_json::to_string(&sig).expect("serialize");
        let sig2: ModelSignature = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(sig, sig2);
    }

    #[test]
    fn memory_record_serde_round_trip() {
        let rec = MemoryRecord {
            id: Uuid::new_v4(),
            domains: vec!["dev".to_string()],
            domain_scores: {
                let mut m = HashMap::new();
                m.insert("dev".to_string(), 0.9);
                m
            },
            content: "hello".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["tag1".to_string()],
            embedding: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: serde_json::json!({}),
        };
        let json = serde_json::to_string(&rec).expect("serialize");
        let rec2: MemoryRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rec.content, rec2.content);
        assert_eq!(rec.domains, rec2.domains);
    }
}
