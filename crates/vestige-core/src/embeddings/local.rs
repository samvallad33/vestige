//! Local Semantic Embeddings
//!
//! Uses fastembed v5.11 for local inference.
//!
//! ## Models
//!
//! - **Default**: Nomic Embed Text v1.5 (ONNX, 768d → 256d Matryoshka, 8192 context)
//! - **Optional**: Nomic Embed Text v2 MoE (Candle, 475M params, 305M active, 8 experts)
//!   Enable with `nomic-v2` feature flag + `metal` for Apple Silicon acceleration.
//! - **Optional**: Qwen3 Embedding 0.6B (Candle, 1024d → 256d Matryoshka)
//!   Enable with `qwen3-embeddings` and `VESTIGE_EMBEDDING_MODEL=qwen3-0.6b`.

#[cfg(feature = "qwen3-embeddings")]
use candle_core::{DType, Device};
#[cfg(feature = "qwen3-embeddings")]
use fastembed::Qwen3TextEmbedding;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::sync::{Mutex, OnceLock};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Embedding dimensions after Matryoshka truncation
/// Truncated from 768 → 256 for 3x storage savings with only ~2% quality loss
/// (Matryoshka Representation Learning — the first N dims ARE the N-dim representation)
pub const EMBEDDING_DIMENSIONS: usize = 256;

/// Maximum text length for embedding (truncated if longer)
pub const MAX_TEXT_LENGTH: usize = 8192;

/// Batch size for efficient embedding generation
pub const BATCH_SIZE: usize = 32;

// ============================================================================
// GLOBAL MODEL (with Mutex for fastembed v5 API)
// ============================================================================

/// Result type for model initialization
static EMBEDDING_BACKEND_RESULT: OnceLock<Result<Mutex<EmbeddingBackend>, String>> =
    OnceLock::new();

const NOMIC_V15_MODEL_ID: &str = "nomic-ai/nomic-embed-text-v1.5";
#[cfg(feature = "qwen3-embeddings")]
const QWEN3_06B_MODEL_ID: &str = "Qwen/Qwen3-Embedding-0.6B";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbeddingModelSpec {
    NomicV15,
    #[cfg(feature = "qwen3-embeddings")]
    Qwen3Embedding06B,
}

impl EmbeddingModelSpec {
    fn selected() -> Result<Self, String> {
        let requested = std::env::var("VESTIGE_EMBEDDING_MODEL")
            .ok()
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "nomic-v1.5".to_string());

        match requested.as_str() {
            "nomic" | "nomic-v1.5" | "nomic-embed-text-v1.5" | NOMIC_V15_MODEL_ID => {
                Ok(Self::NomicV15)
            }
            "qwen3" | "qwen3-0.6b" | "qwen3-embedding-0.6b" | "qwen/qwen3-embedding-0.6b" => {
                #[cfg(feature = "qwen3-embeddings")]
                {
                    Ok(Self::Qwen3Embedding06B)
                }
                #[cfg(not(feature = "qwen3-embeddings"))]
                {
                    Err(
                        "VESTIGE_EMBEDDING_MODEL requests Qwen3, but vestige-core was not built with the qwen3-embeddings feature"
                            .to_string(),
                    )
                }
            }
            other => Err(format!(
                "Unsupported VESTIGE_EMBEDDING_MODEL '{}'. Expected 'nomic-v1.5' or 'qwen3-0.6b'.",
                other
            )),
        }
    }

    fn model_name(self) -> &'static str {
        match self {
            Self::NomicV15 => NOMIC_V15_MODEL_ID,
            #[cfg(feature = "qwen3-embeddings")]
            Self::Qwen3Embedding06B => QWEN3_06B_MODEL_ID,
        }
    }
}

enum EmbeddingBackend {
    NomicV15(TextEmbedding),
    #[cfg(feature = "qwen3-embeddings")]
    Qwen3Embedding06B(Qwen3TextEmbedding),
}

impl EmbeddingBackend {
    fn model_name(&self) -> &'static str {
        match self {
            Self::NomicV15(_) => NOMIC_V15_MODEL_ID,
            #[cfg(feature = "qwen3-embeddings")]
            Self::Qwen3Embedding06B(_) => QWEN3_06B_MODEL_ID,
        }
    }
}

fn qwen3_format_query(query: &str) -> String {
    format!(
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}"
    )
}

/// Get the default cache directory for fastembed models.
///
/// Resolution order:
/// 1. `FASTEMBED_CACHE_PATH` env var (explicit override)
/// 2. Platform cache dir via `directories::ProjectDirs`
///    - Linux:   `$XDG_CACHE_HOME/vestige/fastembed` (typically `~/.cache/vestige/fastembed`)
///    - macOS:   `~/Library/Caches/vestige/fastembed`
///    - Windows: `%LOCALAPPDATA%\vestige\cache\fastembed`
/// 3. `~/.cache/vestige/fastembed` (home-dir fallback)
/// 4. `.fastembed_cache` relative to CWD (absolute last resort, should never trigger)
pub(crate) fn get_cache_dir() -> std::path::PathBuf {
    if let Ok(path) = std::env::var("FASTEMBED_CACHE_PATH") {
        return std::path::PathBuf::from(path);
    }

    // qualifier="" produces a clean app-name-only path on Linux/Windows;
    // on macOS the qualifier is used for the bundle ID so we keep it empty
    // to get ~/Library/Caches/vestige/fastembed rather than
    // ~/Library/Caches/com.vestige.vestige/fastembed.
    if let Some(proj_dirs) = directories::ProjectDirs::from("", "vestige", "vestige") {
        return proj_dirs.cache_dir().join("fastembed");
    }

    // Fallback to home directory
    if let Some(base_dirs) = directories::BaseDirs::new() {
        return base_dirs.home_dir().join(".cache/vestige/fastembed");
    }

    // Last resort fallback (shouldn't happen in practice)
    std::path::PathBuf::from(".fastembed_cache")
}

/// Initialize the global embedding backend selected by `VESTIGE_EMBEDDING_MODEL`.
fn get_backend() -> Result<std::sync::MutexGuard<'static, EmbeddingBackend>, EmbeddingError> {
    let result = EMBEDDING_BACKEND_RESULT.get_or_init(|| {
        let spec = EmbeddingModelSpec::selected()?;
        // Get cache directory (respects FASTEMBED_CACHE_PATH env var)
        let cache_dir = get_cache_dir();

        // Create cache directory if it doesn't exist
        if let Err(e) = std::fs::create_dir_all(&cache_dir) {
            tracing::warn!("Failed to create cache directory {:?}: {}", cache_dir, e);
        }

        match spec {
            EmbeddingModelSpec::NomicV15 => {
                let options = InitOptions::new(EmbeddingModel::NomicEmbedTextV15)
                    .with_show_download_progress(true)
                    .with_cache_dir(cache_dir);

                TextEmbedding::try_new(options)
                    .map(EmbeddingBackend::NomicV15)
                    .map(Mutex::new)
                    .map_err(|e| {
                        format!(
                            "Failed to initialize {} embedding model: {}. \
                            Ensure ONNX runtime is available and model files can be downloaded.",
                            spec.model_name(),
                            e
                        )
                    })
            }
            #[cfg(feature = "qwen3-embeddings")]
            EmbeddingModelSpec::Qwen3Embedding06B => {
                let device = qwen3_device();
                Qwen3TextEmbedding::from_hf(
                    QWEN3_06B_MODEL_ID,
                    &device,
                    DType::F32,
                    MAX_TEXT_LENGTH,
                )
                .map(EmbeddingBackend::Qwen3Embedding06B)
                .map(Mutex::new)
                .map_err(|e| {
                    format!(
                        "Failed to initialize {} embedding model: {}. \
                        Ensure Hugging Face model files can be downloaded.",
                        spec.model_name(),
                        e
                    )
                })
            }
        }
    });

    match result {
        Ok(backend) => backend
            .lock()
            .map_err(|e| EmbeddingError::ModelInit(format!("Lock poisoned: {}", e))),
        Err(err) => Err(EmbeddingError::ModelInit(err.clone())),
    }
}

#[cfg(feature = "qwen3-embeddings")]
fn qwen3_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            return device;
        }
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            return device;
        }
    }
    Device::Cpu
}

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Embedding error types
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum EmbeddingError {
    /// Failed to initialize the embedding model
    ModelInit(String),
    /// Failed to generate embedding
    EmbeddingFailed(String),
    /// Invalid input (empty, too long, etc.)
    InvalidInput(String),
}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingError::ModelInit(e) => write!(f, "Model initialization failed: {}", e),
            EmbeddingError::EmbeddingFailed(e) => write!(f, "Embedding generation failed: {}", e),
            EmbeddingError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for EmbeddingError {}

// ============================================================================
// EMBEDDING TYPE
// ============================================================================

/// A semantic embedding vector
#[derive(Debug, Clone)]
pub struct Embedding {
    /// The embedding vector
    pub vector: Vec<f32>,
    /// Dimensions of the vector
    pub dimensions: usize,
}

impl Embedding {
    /// Create a new embedding from a vector
    pub fn new(vector: Vec<f32>) -> Self {
        let dimensions = vector.len();
        Self { vector, dimensions }
    }

    /// Compute cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        if self.dimensions != other.dimensions {
            return 0.0;
        }
        cosine_similarity(&self.vector, &other.vector)
    }

    /// Compute Euclidean distance with another embedding
    pub fn euclidean_distance(&self, other: &Embedding) -> f32 {
        if self.dimensions != other.dimensions {
            return f32::MAX;
        }
        euclidean_distance(&self.vector, &other.vector)
    }

    /// Normalize the embedding vector to unit length
    pub fn normalize(&mut self) {
        let norm = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut self.vector {
                *x /= norm;
            }
        }
    }

    /// Check if the embedding is normalized (unit length)
    pub fn is_normalized(&self) -> bool {
        let norm = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        (norm - 1.0).abs() < 0.001
    }

    /// Convert to bytes for storage
    pub fn to_bytes(&self) -> Vec<u8> {
        self.vector.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if !bytes.len().is_multiple_of(4) {
            return None;
        }
        let vector: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Some(Self::new(vector))
    }
}

// ============================================================================
// EMBEDDING SERVICE
// ============================================================================

/// Service for generating and managing embeddings
pub struct EmbeddingService {
    _unused: (),
}

impl Default for EmbeddingService {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingService {
    /// Create a new embedding service
    pub fn new() -> Self {
        Self { _unused: () }
    }

    /// Check if the model is ready
    pub fn is_ready(&self) -> bool {
        match get_backend() {
            Ok(_) => true,
            Err(e) => {
                tracing::warn!("Embedding model not ready: {}", e);
                false
            }
        }
    }

    /// Check if the model is ready and return the error if not
    pub fn check_ready(&self) -> Result<(), EmbeddingError> {
        get_backend().map(|_| ())
    }

    /// Initialize the model (downloads if necessary)
    pub fn init(&self) -> Result<(), EmbeddingError> {
        let _model = get_backend()?; // Ensures model is loaded and returns any init errors
        Ok(())
    }

    /// Get the model name
    pub fn model_name(&self) -> &'static str {
        if let Some(Ok(backend)) = EMBEDDING_BACKEND_RESULT.get()
            && let Ok(backend) = backend.lock()
        {
            return backend.model_name();
        }

        EmbeddingModelSpec::selected()
            .unwrap_or(EmbeddingModelSpec::NomicV15)
            .model_name()
    }

    /// Get the embedding dimensions
    pub fn dimensions(&self) -> usize {
        EMBEDDING_DIMENSIONS
    }

    /// Generate embedding for a single text
    pub fn embed(&self, text: &str) -> Result<Embedding, EmbeddingError> {
        if text.is_empty() {
            return Err(EmbeddingError::InvalidInput(
                "Text cannot be empty".to_string(),
            ));
        }

        let mut backend = get_backend()?;

        // Truncate if too long (char-boundary safe)
        let text = if text.len() > MAX_TEXT_LENGTH {
            let mut end = MAX_TEXT_LENGTH;
            while !text.is_char_boundary(end) && end > 0 {
                end -= 1;
            }
            &text[..end]
        } else {
            text
        };

        let embeddings = match &mut *backend {
            EmbeddingBackend::NomicV15(model) => model
                .embed(vec![text], None)
                .map_err(|e| EmbeddingError::EmbeddingFailed(e.to_string()))?,
            #[cfg(feature = "qwen3-embeddings")]
            EmbeddingBackend::Qwen3Embedding06B(model) => model
                .embed(&[text])
                .map_err(|e| EmbeddingError::EmbeddingFailed(e.to_string()))?,
        };

        if embeddings.is_empty() {
            return Err(EmbeddingError::EmbeddingFailed(
                "No embedding generated".to_string(),
            ));
        }

        Ok(Embedding::new(matryoshka_truncate(embeddings[0].clone())))
    }

    /// Generate an embedding for retrieval queries.
    ///
    /// Qwen3 uses instruction-formatted queries against raw document embeddings;
    /// Nomic remains symmetric and receives the query unchanged.
    pub fn embed_query(&self, query: &str) -> Result<Embedding, EmbeddingError> {
        if self.model_name().to_ascii_lowercase().contains("qwen3") {
            let formatted = qwen3_format_query(query);
            self.embed(&formatted)
        } else {
            self.embed(query)
        }
    }

    /// Generate embeddings for multiple texts (batch processing)
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let mut backend = get_backend()?;
        let mut all_embeddings = Vec::with_capacity(texts.len());

        // Process in batches for efficiency
        for chunk in texts.chunks(BATCH_SIZE) {
            let truncated: Vec<&str> = chunk
                .iter()
                .map(|t| {
                    if t.len() > MAX_TEXT_LENGTH {
                        let mut end = MAX_TEXT_LENGTH;
                        while !t.is_char_boundary(end) && end > 0 {
                            end -= 1;
                        }
                        &t[..end]
                    } else {
                        *t
                    }
                })
                .collect();

            let embeddings = match &mut *backend {
                EmbeddingBackend::NomicV15(model) => model
                    .embed(truncated, None)
                    .map_err(|e| EmbeddingError::EmbeddingFailed(e.to_string()))?,
                #[cfg(feature = "qwen3-embeddings")]
                EmbeddingBackend::Qwen3Embedding06B(model) => model
                    .embed(&truncated)
                    .map_err(|e| EmbeddingError::EmbeddingFailed(e.to_string()))?,
            };

            for emb in embeddings {
                all_embeddings.push(Embedding::new(matryoshka_truncate(emb)));
            }
        }

        Ok(all_embeddings)
    }

    /// Find most similar embeddings to a query
    pub fn find_similar(
        &self,
        query_embedding: &Embedding,
        candidate_embeddings: &[Embedding],
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let mut similarities: Vec<(usize, f32)> = candidate_embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| (i, query_embedding.cosine_similarity(emb)))
            .collect();

        // Sort by similarity (highest first)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        similarities.into_iter().take(top_k).collect()
    }
}

// ============================================================================
// SIMILARITY FUNCTIONS
// ============================================================================

/// Apply Matryoshka truncation: truncate to EMBEDDING_DIMENSIONS and L2-normalize
///
/// Nomic Embed v1.5 supports Matryoshka Representation Learning,
/// meaning the first N dimensions of the 768-dim output ARE a valid
/// N-dimensional embedding with minimal quality loss (~2% on MTEB for 256-dim).
#[inline]
pub fn matryoshka_truncate(mut vector: Vec<f32>) -> Vec<f32> {
    if vector.len() > EMBEDDING_DIMENSIONS {
        vector.truncate(EMBEDDING_DIMENSIONS);
    }
    // L2-normalize the truncated vector
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut vector {
            *x /= norm;
        }
    }
    vector
}

/// Compute cosine similarity between two vectors
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot_product = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot_product += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denominator = (norm_a * norm_b).sqrt();
    if denominator > 0.0 {
        dot_product / denominator
    } else {
        0.0
    }
}

/// Compute Euclidean distance between two vectors
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Compute dot product between two vectors
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_euclidean_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let dist = euclidean_distance(&a, &b);
        assert!(dist.abs() < 0.0001);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_embedding_to_from_bytes() {
        let original = Embedding::new(vec![1.5, 2.5, 3.5, 4.5]);
        let bytes = original.to_bytes();
        let restored = Embedding::from_bytes(&bytes).unwrap();

        assert_eq!(original.vector.len(), restored.vector.len());
        for (a, b) in original.vector.iter().zip(restored.vector.iter()) {
            assert!((a - b).abs() < 0.0001);
        }
    }

    #[test]
    fn test_qwen3_query_format() {
        let formatted = qwen3_format_query("rust memory portability");
        assert!(formatted.starts_with("Instruct:"));
        assert!(formatted.contains("retrieve relevant passages"));
        assert!(formatted.ends_with("Query: rust memory portability"));
    }

    #[test]
    fn test_embedding_normalize() {
        let mut emb = Embedding::new(vec![3.0, 4.0]);
        emb.normalize();

        // Should be unit length
        assert!(emb.is_normalized());

        // Components should be 0.6 and 0.8 (3/5 and 4/5)
        assert!((emb.vector[0] - 0.6).abs() < 0.0001);
        assert!((emb.vector[1] - 0.8).abs() < 0.0001);
    }

    #[test]
    fn test_find_similar() {
        let service = EmbeddingService::new();

        let query = Embedding::new(vec![1.0, 0.0, 0.0]);
        let candidates = vec![
            Embedding::new(vec![1.0, 0.0, 0.0]),  // Most similar
            Embedding::new(vec![0.7, 0.7, 0.0]),  // Somewhat similar
            Embedding::new(vec![0.0, 1.0, 0.0]),  // Orthogonal
            Embedding::new(vec![-1.0, 0.0, 0.0]), // Opposite
        ];

        let results = service.find_similar(&query, &candidates, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // First candidate should be most similar
        assert!((results[0].1 - 1.0).abs() < 0.0001);
    }
}
