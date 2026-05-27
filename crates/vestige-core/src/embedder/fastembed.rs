//! `FastembedEmbedder` -- adapts the existing `EmbeddingService` to the
//! `LocalEmbedder` trait.

#[cfg(feature = "embeddings")]
use crate::embeddings::{EMBEDDING_DIMENSIONS, EmbeddingService};

use super::{EmbedderError, EmbedderResult, EmbedderSend};

pub struct FastembedEmbedder {
    #[cfg(feature = "embeddings")]
    inner: EmbeddingService,
    cached_hash: std::sync::OnceLock<String>,
}

impl FastembedEmbedder {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "embeddings")]
            inner: EmbeddingService::new(),
            cached_hash: std::sync::OnceLock::new(),
        }
    }

    fn compute_hash(name: &str, dim: usize) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(name.as_bytes());
        hasher.update(&(dim as u64).to_le_bytes());
        // fastembed's ONNX bytes are not directly accessible at runtime; we
        // use `(name, dim, vestige-core CARGO_PKG_VERSION)` as the
        // signature. If fastembed ever changes its output deterministically
        // between minor versions, bumping the crate version triggers a
        // mismatch -- which is exactly the drift we want to detect.
        hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
        hasher.finalize().to_hex().to_string()
    }
}

impl Default for FastembedEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbedderSend for FastembedEmbedder {
    async fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        #[cfg(feature = "embeddings")]
        {
            let emb = self
                .inner
                .embed(text)
                .map_err(|e| EmbedderError::EmbedFailed(e.to_string()))?;
            Ok(emb.vector)
        }
        #[cfg(not(feature = "embeddings"))]
        {
            let _ = text;
            Err(EmbedderError::Init(
                "embeddings feature not enabled".to_string(),
            ))
        }
    }

    fn model_name(&self) -> &str {
        #[cfg(feature = "embeddings")]
        {
            self.inner.model_name()
        }
        #[cfg(not(feature = "embeddings"))]
        {
            "nomic-ai/nomic-embed-text-v1.5"
        }
    }

    fn dimension(&self) -> usize {
        #[cfg(feature = "embeddings")]
        {
            EMBEDDING_DIMENSIONS
        }
        #[cfg(not(feature = "embeddings"))]
        {
            256
        }
    }

    fn model_hash(&self) -> String {
        self.cached_hash
            .get_or_init(|| Self::compute_hash(self.model_name(), self.dimension()))
            .clone()
    }

    async fn embed_batch(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        #[cfg(feature = "embeddings")]
        {
            let embs = self
                .inner
                .embed_batch(texts)
                .map_err(|e| EmbedderError::EmbedFailed(e.to_string()))?;
            Ok(embs.into_iter().map(|e| e.vector).collect())
        }
        #[cfg(not(feature = "embeddings"))]
        {
            let _ = texts;
            Err(EmbedderError::Init(
                "embeddings feature not enabled".to_string(),
            ))
        }
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedder_reports_correct_name() {
        let e = FastembedEmbedder::new();
        assert!(e.model_name().contains("nomic"), "model name should contain 'nomic'");
    }

    #[test]
    fn embedder_reports_256_dimension() {
        let e = FastembedEmbedder::new();
        assert_eq!(e.dimension(), 256);
    }

    #[test]
    fn embedder_hash_is_stable() {
        let e = FastembedEmbedder::new();
        let h1 = e.model_hash();
        let h2 = e.model_hash();
        assert_eq!(h1, h2, "model_hash must be stable across calls");
    }

    #[test]
    fn embedder_hash_includes_crate_version() {
        // Compute what the hash should be given the known inputs
        let name = FastembedEmbedder::new().model_name().to_string();
        let dim = FastembedEmbedder::new().dimension();
        let expected = FastembedEmbedder::compute_hash(&name, dim);
        let got = FastembedEmbedder::new().model_hash();
        assert_eq!(got, expected);
    }

    #[test]
    fn embedder_signature_matches_accessors() {
        let e = FastembedEmbedder::new();
        let sig = e.signature();
        assert_eq!(sig.name, e.model_name());
        assert_eq!(sig.dimension, e.dimension());
        assert_eq!(sig.hash, e.model_hash());
    }

    #[cfg(feature = "embeddings")]
    #[test]
    fn embedder_embed_smoke() {
        let e = FastembedEmbedder::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let vec = rt.block_on(e.embed("hello world")).expect("embed");
        assert_eq!(vec.len(), 256);
    }

    #[cfg(feature = "embeddings")]
    #[test]
    fn embedder_embed_batch_matches_sequential() {
        let e = FastembedEmbedder::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let texts = ["alpha beta", "gamma delta"];
        let batch = rt.block_on(e.embed_batch(texts.as_ref())).expect("batch");
        let seq_a = rt.block_on(e.embed(texts[0])).expect("seq a");
        let seq_b = rt.block_on(e.embed(texts[1])).expect("seq b");
        assert_eq!(batch[0], seq_a);
        assert_eq!(batch[1], seq_b);
    }
}
