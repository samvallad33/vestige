//! Semantic Embeddings Module
//!
//! Provides local embedding generation using fastembed (ONNX-based).
//! No external API calls required - 100% local and private.
//!
//! Supports:
//! - Text embedding generation (768-dimensional vectors via nomic-embed-text-v1.5)
//! - Cosine similarity computation
//! - Batch embedding for efficiency
//! - Hybrid multi-model fusion (future)

mod code;
mod hybrid;
mod local;

/// Register the tract backend exactly once before any fastembed session is created.
/// Under `embeddings` or `ort-dynamic` this is a no-op; only active under `tract`.
#[cfg(feature = "tract")]
pub(crate) fn ensure_tract_backend() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        ort::set_api(ort_tract::api());
    });
}

#[cfg(not(feature = "tract"))]
#[inline(always)]
pub(crate) fn ensure_tract_backend() {}

pub(crate) use local::get_cache_dir;
pub use local::{
    BATCH_SIZE, EMBEDDING_DIMENSIONS, Embedding, EmbeddingError, EmbeddingService, MAX_TEXT_LENGTH,
    cosine_similarity, dot_product, euclidean_distance, matryoshka_truncate,
};

pub use code::CodeEmbedding;
pub use hybrid::HybridEmbedding;
