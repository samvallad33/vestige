//! Text-to-vector encoding trait. Pluggable per-install.

use std::future::Future;
use std::pin::Pin;

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

/// Boxed Send future returning an `EmbedderResult<T>`, bound to the lifetime
/// of the borrows captured by the call. Used as the return type of every
/// async method on the dyn-compatible `Embedder` trait below.
pub type BoxedEmbedderFuture<'a, T> =
    Pin<Box<dyn Future<Output = EmbedderResult<T>> + Send + 'a>>;

/// Pluggable embedder. The storage layer NEVER calls fastembed directly;
/// callers compute vectors via this trait and pass them into `MemoryStore`.
///
/// `LocalEmbedder` is the source-of-truth trait declared with native
/// async-fn-in-trait. `#[trait_variant::make(EmbedderSend: Send)]` derives
/// a Send-bounded variant that backends actually implement (the
/// trait_variant 0.1.x blanket goes variant -> source). The dyn-compatible
/// public surface is the `Embedder` trait declared below, which wraps every
/// async method in `Pin<Box<dyn Future + Send + '_>>`.
#[trait_variant::make(EmbedderSend: Send)]
pub trait LocalEmbedder: Sync + 'static {
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

/// Dyn-compatible embedder trait.
///
/// `EmbedderSend` above is the trait users implement; it uses native
/// async-fn-in-trait return types (RPITIT), which gives zero-allocation
/// static dispatch but is not dyn-safe. This trait wraps every async
/// method in `Pin<Box<dyn Future + Send + '_>>` so `Box<dyn Embedder>`
/// and `Arc<dyn Embedder>` work for the cognitive module surface and
/// the Phase 1 integration tests.
///
/// Implementations should not target this trait directly; the blanket
/// `impl<T: EmbedderSend> Embedder for T` adapts every Send-variant
/// implementation automatically.
pub trait Embedder: Send + Sync + 'static {
    fn embed<'a>(&'a self, text: &'a str) -> BoxedEmbedderFuture<'a, Vec<f32>>;
    fn embed_batch<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> BoxedEmbedderFuture<'a, Vec<Vec<f32>>>;
    fn model_name(&self) -> &str;
    fn dimension(&self) -> usize;
    fn model_hash(&self) -> String;
    fn signature(&self) -> crate::storage::ModelSignature;
}

impl<T> Embedder for T
where
    T: EmbedderSend,
{
    fn embed<'a>(&'a self, text: &'a str) -> BoxedEmbedderFuture<'a, Vec<f32>> {
        Box::pin(<T as EmbedderSend>::embed(self, text))
    }
    fn embed_batch<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> BoxedEmbedderFuture<'a, Vec<Vec<f32>>> {
        Box::pin(<T as EmbedderSend>::embed_batch(self, texts))
    }
    fn model_name(&self) -> &str {
        <T as EmbedderSend>::model_name(self)
    }
    fn dimension(&self) -> usize {
        <T as EmbedderSend>::dimension(self)
    }
    fn model_hash(&self) -> String {
        <T as EmbedderSend>::model_hash(self)
    }
    fn signature(&self) -> crate::storage::ModelSignature {
        <T as EmbedderSend>::signature(self)
    }
}
