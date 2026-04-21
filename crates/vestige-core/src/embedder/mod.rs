//! Text-to-vector encoding trait. Pluggable per-install.

mod fastembed;

pub use fastembed::FastembedEmbedder;

/// Error returned by every `Embedder` method.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum EmbedderError {
    #[error("embedder initialization failed: {0}")]
    Init(String),
    #[error("embedding generation failed: {0}")]
    EmbedFailed(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

pub type EmbedderResult<T> = std::result::Result<T, EmbedderError>;

/// Pluggable embedder. The storage layer NEVER calls fastembed directly;
/// callers compute vectors via this trait and pass them into `MemoryStore`.
///
/// `#[async_trait::async_trait]` makes every `async fn` return a
/// `Pin<Box<dyn Future + Send>>`, which is required for `Box<dyn Embedder>`
/// and `Arc<dyn Embedder>` to be dyn-compatible.
#[async_trait::async_trait]
pub trait LocalEmbedder: Send + Sync + 'static {
    async fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>>;

    fn model_name(&self) -> &str;

    fn dimension(&self) -> usize;

    /// Stable blake3 hash of (model_name || dimension || vestige-core crate version).
    /// Lowercase hex, 64 chars.
    ///
    /// Used by `MemoryStore::register_model` to detect silent model drift
    /// (e.g. a fastembed minor upgrade that changes vector output).
    fn model_hash(&self) -> String;

    async fn embed_batch(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>>;

    /// Returns the `ModelSignature` describing this embedder. Convenience
    /// wrapper over the three accessors above.
    fn signature(&self) -> crate::storage::ModelSignature {
        crate::storage::ModelSignature {
            name: self.model_name().to_string(),
            dimension: self.dimension(),
            hash: self.model_hash(),
        }
    }
}

/// Type alias: `Embedder` is the dyn-compatible, Send+Sync variant.
/// Both names refer to the same `async_trait`-annotated trait.
pub use LocalEmbedder as Embedder;

