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

// Compatibility facade: callers that historically looked for embedding
// functionality under `embeddings` can adopt profiles without a module-path
// migration. The canonical home is `crate::embedding`.
pub use crate::embedding::{
    ActiveEmbeddingProfile, BuiltinEmbeddingProfile, ChunkingStrategy,
    EMBEDDING_PROFILE_MANIFEST_SCHEMA_VERSION, EmbeddingDevice, EmbeddingEvaluationSummary,
    EmbeddingMigrationState, EmbeddingNormalization, EmbeddingProfile, EmbeddingProfileError,
    EmbeddingProfileFailure, EmbeddingProfileId, EmbeddingProfileManifest, EmbeddingProfileState,
    EmbeddingRuntimeBackend, EmbeddingRuntimeMetadata, EmbeddingVerification, EncodingTemplate,
    ModelArtifactHash, ProfileMigrationCheckpoint, ProfileRuntimeRegistry, ProfiledEmbedder,
    VerificationStatus, VerifiedLocalArtifact, builtin_embedding_profile_by_id,
    builtin_embedding_profiles,
};

#[cfg(feature = "vector-search")]
pub(crate) use local::get_cache_dir;
pub use local::{cache_populated, embedding_model_cached};
pub use local::{
    BATCH_SIZE, EMBEDDING_DIMENSIONS, Embedding, EmbeddingError, EmbeddingService, MAX_TEXT_LENGTH,
    cosine_similarity, dot_product, euclidean_distance, matryoshka_truncate,
};

pub use code::CodeEmbedding;
pub use hybrid::HybridEmbedding;
