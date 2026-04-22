//! Memory Reranking Module
//!
//! ## Two-Stage Retrieval with Cross-Encoder
//!
//! Uses fastembed's Jina Reranker v1 Turbo (38M params) cross-encoder
//! for high-precision reranking:
//! 1. Stage 1: Retrieve top-50 candidates via hybrid search (fast, high recall)
//! 2. Stage 2: Cross-encoder rerank to find best top-10 (slower, high precision)
//!
//! Falls back to BM25-like term overlap scoring when the cross-encoder
//! model is unavailable.

#[cfg(any(feature = "embeddings", feature = "tract"))]
use crate::embeddings::{ensure_tract_backend, get_cache_dir};
#[cfg(any(feature = "embeddings", feature = "tract"))]
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Default number of candidates to retrieve before reranking
pub const DEFAULT_RETRIEVAL_COUNT: usize = 50;

/// Default number of results after reranking
pub const DEFAULT_RERANK_COUNT: usize = 10;

// ============================================================================
// TYPES
// ============================================================================

/// Reranker error types
#[derive(Debug, Clone)]
pub enum RerankerError {
    /// Failed to initialize the reranker model
    ModelInit(String),
    /// Failed to rerank
    RerankFailed(String),
    /// Invalid input
    InvalidInput(String),
}

impl std::fmt::Display for RerankerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RerankerError::ModelInit(e) => write!(f, "Reranker initialization failed: {}", e),
            RerankerError::RerankFailed(e) => write!(f, "Reranking failed: {}", e),
            RerankerError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for RerankerError {}

/// A reranked result with relevance score
#[derive(Debug, Clone)]
pub struct RerankedResult<T> {
    /// The original item
    pub item: T,
    /// Reranking score (higher is more relevant)
    pub score: f32,
    /// Original rank before reranking
    pub original_rank: usize,
}

// ============================================================================
// RERANKER SERVICE
// ============================================================================

/// Configuration for reranking
#[derive(Debug, Clone)]
pub struct RerankerConfig {
    /// Number of candidates to consider for reranking
    pub candidate_count: usize,
    /// Number of results to return after reranking
    pub result_count: usize,
    /// Minimum score threshold (results below this are filtered)
    pub min_score: Option<f32>,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            candidate_count: DEFAULT_RETRIEVAL_COUNT,
            result_count: DEFAULT_RERANK_COUNT,
            min_score: None,
        }
    }
}

/// Service for reranking search results using a cross-encoder model
///
/// When the `embeddings` feature is enabled and `init_cross_encoder()` is called,
/// uses Jina Reranker v1 Turbo for neural cross-encoder scoring.
/// Falls back to BM25-like term overlap when the model is unavailable.
pub struct Reranker {
    config: RerankerConfig,
    #[cfg(any(feature = "embeddings", feature = "tract"))]
    cross_encoder: Option<TextRerank>,
}

impl Default for Reranker {
    fn default() -> Self {
        Self::new(RerankerConfig::default())
    }
}

impl Reranker {
    /// Create a new reranker with the given configuration
    ///
    /// The cross-encoder model is NOT loaded here — call `init_cross_encoder()`
    /// explicitly to load it. This keeps construction fast and test-friendly.
    pub fn new(config: RerankerConfig) -> Self {
        Self {
            config,
            #[cfg(any(feature = "embeddings", feature = "tract"))]
            cross_encoder: None,
        }
    }

    /// Initialize the cross-encoder model (Jina Reranker v1 Turbo, ~150MB)
    ///
    /// Downloads the model on first call. Call this during server startup,
    /// NOT in tests or hot paths.
    #[cfg(any(feature = "embeddings", feature = "tract"))]
    pub fn init_cross_encoder(&mut self) {
        if self.cross_encoder.is_some() {
            return; // Already initialized
        }

        ensure_tract_backend();

        let options = RerankInitOptions::new(RerankerModel::JINARerankerV1TurboEn)
            .with_cache_dir(get_cache_dir())
            .with_show_download_progress(true);

        match TextRerank::try_new(options) {
            Ok(model) => {
                eprintln!("[vestige] Cross-encoder reranker loaded (Jina Reranker v1 Turbo)");
                self.cross_encoder = Some(model);
            }
            Err(e) => {
                eprintln!("[vestige] Cross-encoder unavailable, using BM25 fallback: {e}");
            }
        }
    }

    /// Check if the cross-encoder model is available
    pub fn has_cross_encoder(&self) -> bool {
        #[cfg(any(feature = "embeddings", feature = "tract"))]
        {
            self.cross_encoder.is_some()
        }
        #[cfg(not(any(feature = "embeddings", feature = "tract")))]
        {
            false
        }
    }

    /// Rerank candidates based on relevance to the query
    ///
    /// Uses cross-encoder model when available for neural relevance scoring.
    /// Falls back to BM25-like term overlap scoring otherwise.
    pub fn rerank<T: Clone>(
        &mut self,
        query: &str,
        candidates: Vec<(T, String)>,
        top_k: Option<usize>,
    ) -> Result<Vec<RerankedResult<T>>, RerankerError> {
        if query.is_empty() {
            return Err(RerankerError::InvalidInput(
                "Query cannot be empty".to_string(),
            ));
        }

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        let limit = top_k.unwrap_or(self.config.result_count);

        // Try cross-encoder first
        #[cfg(any(feature = "embeddings", feature = "tract"))]
        if let Some(ref mut model) = self.cross_encoder {
            let documents: Vec<&str> = candidates.iter().map(|(_, text)| text.as_str()).collect();

            if let Ok(rerank_results) = model.rerank(query, &documents, false, None) {
                let mut results: Vec<RerankedResult<T>> = rerank_results
                    .into_iter()
                    .filter_map(|rr| {
                        candidates.get(rr.index).map(|(item, _)| RerankedResult {
                            item: item.clone(),
                            score: rr.score,
                            original_rank: rr.index,
                        })
                    })
                    .collect();

                results.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                if let Some(min_score) = self.config.min_score {
                    results.retain(|r| r.score >= min_score);
                }

                results.truncate(limit);
                return Ok(results);
            }
            // Cross-encoder failed on this call — fall through to BM25 fallback
        }

        // Fallback: BM25-like scoring
        let mut results: Vec<RerankedResult<T>> = candidates
            .into_iter()
            .enumerate()
            .map(|(rank, (item, text))| {
                let score = Self::compute_relevance_score(query, &text);
                RerankedResult {
                    item,
                    score,
                    original_rank: rank,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(min_score) = self.config.min_score {
            results.retain(|r| r.score >= min_score);
        }

        results.truncate(limit);

        Ok(results)
    }

    /// BM25-inspired term overlap scoring (fallback when cross-encoder unavailable)
    fn compute_relevance_score(query: &str, document: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();
        let doc_lower = document.to_lowercase();
        let doc_len = document.len() as f32;

        if doc_len == 0.0 {
            return 0.0;
        }

        let mut score = 0.0;
        let k1 = 1.2_f32;
        let b = 0.75_f32;
        let avg_doc_len = 500.0_f32;

        for term in &query_terms {
            let tf = doc_lower.matches(term).count() as f32;
            if tf > 0.0 {
                let numerator = tf * (k1 + 1.0);
                let denominator = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len));
                score += numerator / denominator;
            }
        }

        if !query_terms.is_empty() {
            score /= query_terms.len() as f32;
        }

        score
    }

    /// Get the current configuration
    pub fn config(&self) -> &RerankerConfig {
        &self.config
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rerank_basic() {
        let mut reranker = Reranker::default();

        let candidates = vec![
            (1, "The quick brown fox".to_string()),
            (2, "A lazy dog sleeps".to_string()),
            (3, "The fox jumps over".to_string()),
        ];

        let results = reranker.rerank("fox", candidates, Some(2)).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].item == 1 || results[0].item == 3);
    }

    #[test]
    fn test_rerank_empty_candidates() {
        let mut reranker = Reranker::default();
        let candidates: Vec<(i32, String)> = vec![];

        let results = reranker.rerank("query", candidates, Some(5)).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_rerank_empty_query() {
        let mut reranker = Reranker::default();
        let candidates = vec![(1, "some text".to_string())];

        let result = reranker.rerank("", candidates, Some(5));
        assert!(result.is_err());
    }

    #[test]
    fn test_min_score_filter() {
        let mut reranker = Reranker::new(RerankerConfig {
            min_score: Some(0.5),
            ..Default::default()
        });

        let candidates = vec![
            (1, "fox fox fox".to_string()),
            (2, "completely unrelated".to_string()),
        ];

        let results = reranker.rerank("fox", candidates, None).unwrap();

        assert!(results.len() <= 2);
        if !results.is_empty() {
            assert!(results[0].score >= 0.5);
        }
    }

    #[test]
    fn test_default_has_no_cross_encoder() {
        let reranker = Reranker::default();
        // Default constructor does NOT load the model — fast and test-friendly
        assert!(!reranker.has_cross_encoder());
    }
}
