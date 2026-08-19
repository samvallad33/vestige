//! Backend-agnostic memory store trait.
//!
//! This is the single abstraction every cognitive module sits above. It is
//! intentionally flat: one trait, ~25 methods, no sub-traits.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

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
            S::InvalidEmbeddingProfile(s) => {
                MemoryStoreError::InvalidInput(format!("invalid embedding profile: {s}"))
            }
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

/// Internal source trait declared with native async-fn-in-trait.
///
/// `#[trait_variant::make(MemoryStoreSend: Send)]` derives a Send-bounded
/// variant whose returned futures are `Send`. In trait_variant 0.1.x the
/// macro emits the blanket `impl<T: MemoryStoreSend> LocalMemoryStore for T`,
/// so backends implement `MemoryStoreSend` (the Send variant) and get
/// `LocalMemoryStore` (the non-Send variant) for free.
///
/// Most callers should reach for the dyn-compatible `MemoryStore` trait
/// declared below, which adapts `MemoryStoreSend` into a boxed-future surface
/// and is the public storage abstraction for cognitive modules and tests
/// that want `Arc<dyn MemoryStore>`.
#[trait_variant::make(MemoryStoreSend: Send)]
pub trait LocalMemoryStore: Sync + 'static {
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
    async fn get_scheduling(&self, memory_id: Uuid) -> MemoryStoreResult<Option<SchedulingState>>;
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

// ----------------------------------------------------------------------------
// DYN-COMPATIBLE STORAGE TRAIT
// ----------------------------------------------------------------------------

/// Boxed Send future returning a `MemoryStoreResult<T>`, bound to the lifetime
/// of the borrows captured by the call (typically `&self` plus any reference
/// arguments). Used as the return type of every method on the dyn-compatible
/// `MemoryStore` trait below.
pub type BoxedStoreFuture<'a, T> = Pin<Box<dyn Future<Output = MemoryStoreResult<T>> + Send + 'a>>;

/// Dyn-compatible storage trait.
///
/// `MemoryStoreSend` above is the trait users implement; it uses native
/// async-fn-in-trait return types (RPITIT), which gives zero-allocation
/// static dispatch but is not dyn-safe. This trait wraps every method in
/// `Pin<Box<dyn Future + Send + '_>>` so `Arc<dyn MemoryStore>` works for
/// the cognitive module surface and the Phase 1 integration tests.
///
/// Implementations should not target this trait directly; the blanket
/// `impl<T: MemoryStoreSend> MemoryStore for T` adapts every Send-variant
/// implementation automatically. Each call boxes the returned future
/// exactly once, identical to the cost of the previous design.
pub trait MemoryStore: Send + Sync + 'static {
    fn init<'a>(&'a self) -> BoxedStoreFuture<'a, ()>;
    fn health_check<'a>(&'a self) -> BoxedStoreFuture<'a, HealthStatus>;

    fn registered_model<'a>(&'a self) -> BoxedStoreFuture<'a, Option<ModelSignature>>;
    fn register_model<'a>(&'a self, sig: &'a ModelSignature) -> BoxedStoreFuture<'a, ()>;

    fn insert<'a>(&'a self, record: &'a MemoryRecord) -> BoxedStoreFuture<'a, Uuid>;
    fn get<'a>(&'a self, id: Uuid) -> BoxedStoreFuture<'a, Option<MemoryRecord>>;
    fn update<'a>(&'a self, record: &'a MemoryRecord) -> BoxedStoreFuture<'a, ()>;
    fn delete<'a>(&'a self, id: Uuid) -> BoxedStoreFuture<'a, ()>;

    fn search<'a>(&'a self, query: &'a SearchQuery) -> BoxedStoreFuture<'a, Vec<SearchResult>>;
    fn fts_search<'a>(
        &'a self,
        text: &'a str,
        limit: usize,
    ) -> BoxedStoreFuture<'a, Vec<SearchResult>>;
    fn vector_search<'a>(
        &'a self,
        embedding: &'a [f32],
        limit: usize,
    ) -> BoxedStoreFuture<'a, Vec<SearchResult>>;

    fn get_scheduling<'a>(
        &'a self,
        memory_id: Uuid,
    ) -> BoxedStoreFuture<'a, Option<SchedulingState>>;
    fn update_scheduling<'a>(&'a self, state: &'a SchedulingState) -> BoxedStoreFuture<'a, ()>;
    fn get_due_memories<'a>(
        &'a self,
        before: DateTime<Utc>,
        limit: usize,
    ) -> BoxedStoreFuture<'a, Vec<(MemoryRecord, SchedulingState)>>;

    fn add_edge<'a>(&'a self, edge: &'a MemoryEdge) -> BoxedStoreFuture<'a, ()>;
    fn get_edges<'a>(
        &'a self,
        node_id: Uuid,
        edge_type: Option<&'a str>,
    ) -> BoxedStoreFuture<'a, Vec<MemoryEdge>>;
    fn remove_edge<'a>(&'a self, source: Uuid, target: Uuid) -> BoxedStoreFuture<'a, ()>;
    fn get_neighbors<'a>(
        &'a self,
        node_id: Uuid,
        depth: usize,
    ) -> BoxedStoreFuture<'a, Vec<(MemoryRecord, f64)>>;

    fn list_domains<'a>(&'a self) -> BoxedStoreFuture<'a, Vec<Domain>>;
    fn get_domain<'a>(&'a self, id: &'a str) -> BoxedStoreFuture<'a, Option<Domain>>;
    fn upsert_domain<'a>(&'a self, domain: &'a Domain) -> BoxedStoreFuture<'a, ()>;
    fn delete_domain<'a>(&'a self, id: &'a str) -> BoxedStoreFuture<'a, ()>;
    fn classify<'a>(&'a self, embedding: &'a [f32]) -> BoxedStoreFuture<'a, Vec<(String, f64)>>;

    fn count<'a>(&'a self) -> BoxedStoreFuture<'a, usize>;
    fn get_stats<'a>(&'a self) -> BoxedStoreFuture<'a, StoreStats>;
    fn vacuum<'a>(&'a self) -> BoxedStoreFuture<'a, ()>;
}

impl<T> MemoryStore for T
where
    T: MemoryStoreSend,
{
    fn init<'a>(&'a self) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::init(self))
    }
    fn health_check<'a>(&'a self) -> BoxedStoreFuture<'a, HealthStatus> {
        Box::pin(<T as MemoryStoreSend>::health_check(self))
    }

    fn registered_model<'a>(&'a self) -> BoxedStoreFuture<'a, Option<ModelSignature>> {
        Box::pin(<T as MemoryStoreSend>::registered_model(self))
    }
    fn register_model<'a>(&'a self, sig: &'a ModelSignature) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::register_model(self, sig))
    }

    fn insert<'a>(&'a self, record: &'a MemoryRecord) -> BoxedStoreFuture<'a, Uuid> {
        Box::pin(<T as MemoryStoreSend>::insert(self, record))
    }
    fn get<'a>(&'a self, id: Uuid) -> BoxedStoreFuture<'a, Option<MemoryRecord>> {
        Box::pin(<T as MemoryStoreSend>::get(self, id))
    }
    fn update<'a>(&'a self, record: &'a MemoryRecord) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::update(self, record))
    }
    fn delete<'a>(&'a self, id: Uuid) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::delete(self, id))
    }

    fn search<'a>(&'a self, query: &'a SearchQuery) -> BoxedStoreFuture<'a, Vec<SearchResult>> {
        Box::pin(<T as MemoryStoreSend>::search(self, query))
    }
    fn fts_search<'a>(
        &'a self,
        text: &'a str,
        limit: usize,
    ) -> BoxedStoreFuture<'a, Vec<SearchResult>> {
        Box::pin(<T as MemoryStoreSend>::fts_search(self, text, limit))
    }
    fn vector_search<'a>(
        &'a self,
        embedding: &'a [f32],
        limit: usize,
    ) -> BoxedStoreFuture<'a, Vec<SearchResult>> {
        Box::pin(<T as MemoryStoreSend>::vector_search(
            self, embedding, limit,
        ))
    }

    fn get_scheduling<'a>(
        &'a self,
        memory_id: Uuid,
    ) -> BoxedStoreFuture<'a, Option<SchedulingState>> {
        Box::pin(<T as MemoryStoreSend>::get_scheduling(self, memory_id))
    }
    fn update_scheduling<'a>(&'a self, state: &'a SchedulingState) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::update_scheduling(self, state))
    }
    fn get_due_memories<'a>(
        &'a self,
        before: DateTime<Utc>,
        limit: usize,
    ) -> BoxedStoreFuture<'a, Vec<(MemoryRecord, SchedulingState)>> {
        Box::pin(<T as MemoryStoreSend>::get_due_memories(
            self, before, limit,
        ))
    }

    fn add_edge<'a>(&'a self, edge: &'a MemoryEdge) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::add_edge(self, edge))
    }
    fn get_edges<'a>(
        &'a self,
        node_id: Uuid,
        edge_type: Option<&'a str>,
    ) -> BoxedStoreFuture<'a, Vec<MemoryEdge>> {
        Box::pin(<T as MemoryStoreSend>::get_edges(self, node_id, edge_type))
    }
    fn remove_edge<'a>(&'a self, source: Uuid, target: Uuid) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::remove_edge(self, source, target))
    }
    fn get_neighbors<'a>(
        &'a self,
        node_id: Uuid,
        depth: usize,
    ) -> BoxedStoreFuture<'a, Vec<(MemoryRecord, f64)>> {
        Box::pin(<T as MemoryStoreSend>::get_neighbors(self, node_id, depth))
    }

    fn list_domains<'a>(&'a self) -> BoxedStoreFuture<'a, Vec<Domain>> {
        Box::pin(<T as MemoryStoreSend>::list_domains(self))
    }
    fn get_domain<'a>(&'a self, id: &'a str) -> BoxedStoreFuture<'a, Option<Domain>> {
        Box::pin(<T as MemoryStoreSend>::get_domain(self, id))
    }
    fn upsert_domain<'a>(&'a self, domain: &'a Domain) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::upsert_domain(self, domain))
    }
    fn delete_domain<'a>(&'a self, id: &'a str) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::delete_domain(self, id))
    }
    fn classify<'a>(&'a self, embedding: &'a [f32]) -> BoxedStoreFuture<'a, Vec<(String, f64)>> {
        Box::pin(<T as MemoryStoreSend>::classify(self, embedding))
    }

    fn count<'a>(&'a self) -> BoxedStoreFuture<'a, usize> {
        Box::pin(<T as MemoryStoreSend>::count(self))
    }
    fn get_stats<'a>(&'a self) -> BoxedStoreFuture<'a, StoreStats> {
        Box::pin(<T as MemoryStoreSend>::get_stats(self))
    }
    fn vacuum<'a>(&'a self) -> BoxedStoreFuture<'a, ()> {
        Box::pin(<T as MemoryStoreSend>::vacuum(self))
    }
}

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
