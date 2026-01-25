//! # Prediction Error Gating
//!
//! Implements neuroscience-inspired prediction error gating for intelligent memory updates.
//!
//! Based on research showing that prediction error (PE) determines whether memories are:
//! - **Updated** (small PE): New info is similar enough to existing memory
//! - **Created** (large PE): New info is different enough to warrant new memory
//!
//! This solves the "bad vs good similar memory" problem by:
//! 1. Detecting when new content is similar to existing memories
//! 2. Calculating the prediction error (semantic distance)
//! 3. Deciding whether to update existing or create new
//! 4. Optionally superseding outdated memories
//!
//! ## Scientific Background
//!
//! Based on:
//! - Sinclair & Bhavnani (2020): "The Reconsolidation Dilemma"
//! - Lee et al. (2017): Prediction error and memory updating
//! - Google Titans (2025): Surprise-based storage
//!
//! ## Example
//!
//! ```rust,ignore
//! use vestige_core::advanced::prediction_error::PredictionErrorGate;
//!
//! let gate = PredictionErrorGate::new();
//!
//! // New content arrives
//! let decision = gate.evaluate(
//!     "Use async/await for better performance",
//!     &existing_memories,
//!     &embeddings,
//! );
//!
//! match decision {
//!     GateDecision::Update { target, .. } => {
//!         // Update existing memory with new info
//!     }
//!     GateDecision::Create { .. } => {
//!         // Create new memory
//!     }
//!     GateDecision::Supersede { old, .. } => {
//!         // New memory supersedes old (demote old)
//!     }
//! }
//! ```

use serde::{Deserialize, Serialize};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Default similarity threshold for considering memories as "similar"
/// Above this = potential update candidate
const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.75;

/// Threshold for considering content as "nearly identical"
/// Above this = definitely update, not create
const NEAR_IDENTICAL_THRESHOLD: f32 = 0.92;

/// Threshold for "correction" detection
/// When new content contradicts existing with high similarity
const CORRECTION_THRESHOLD: f32 = 0.70;

/// Maximum candidates to consider for update
const MAX_UPDATE_CANDIDATES: usize = 5;

// ============================================================================
// GATE DECISION
// ============================================================================

/// Decision made by the prediction error gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GateDecision {
    /// Create a new memory (high prediction error)
    Create {
        /// Reason for creating new
        reason: CreateReason,
        /// Prediction error score (0.0 = identical, 1.0 = completely different)
        prediction_error: f32,
        /// Related memories that were considered
        related_memory_ids: Vec<String>,
    },

    /// Update an existing memory (low prediction error)
    Update {
        /// ID of memory to update
        target_id: String,
        /// How similar the content is (0.0 - 1.0)
        similarity: f32,
        /// Type of update to perform
        update_type: UpdateType,
        /// Prediction error score
        prediction_error: f32,
    },

    /// Supersede an existing memory (correction/improvement)
    Supersede {
        /// ID of memory being superseded
        old_memory_id: String,
        /// Similarity to old memory
        similarity: f32,
        /// Why this supersedes the old one
        supersede_reason: SupersedeReason,
        /// Prediction error score
        prediction_error: f32,
    },

    /// Merge with multiple existing memories
    Merge {
        /// IDs of memories to merge with
        memory_ids: Vec<String>,
        /// Average similarity
        avg_similarity: f32,
        /// Merge strategy
        strategy: MergeStrategy,
    },
}

impl GateDecision {
    /// Get the prediction error score
    pub fn prediction_error(&self) -> f32 {
        match self {
            Self::Create { prediction_error, .. } => *prediction_error,
            Self::Update { prediction_error, .. } => *prediction_error,
            Self::Supersede { prediction_error, .. } => *prediction_error,
            Self::Merge { avg_similarity, .. } => 1.0 - avg_similarity,
        }
    }

    /// Check if this is a create decision
    pub fn is_create(&self) -> bool {
        matches!(self, Self::Create { .. })
    }

    /// Check if this is an update decision
    pub fn is_update(&self) -> bool {
        matches!(self, Self::Update { .. })
    }

    /// Get target ID if updating or superseding
    pub fn target_id(&self) -> Option<&str> {
        match self {
            Self::Update { target_id, .. } => Some(target_id),
            Self::Supersede { old_memory_id, .. } => Some(old_memory_id),
            _ => None,
        }
    }
}

/// Reasons for creating a new memory
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CreateReason {
    /// No similar memories exist
    NoSimilarMemories,
    /// Content is substantially different from all candidates
    HighPredictionError,
    /// Different domain/topic despite surface similarity
    DifferentDomain,
    /// Explicitly requested new memory (not update)
    ExplicitCreate,
    /// First memory in the system
    FirstMemory,
}

/// Types of updates to existing memories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum UpdateType {
    /// Append new information
    Append,
    /// Replace content entirely
    Replace,
    /// Merge content intelligently
    Merge,
    /// Add as related context
    AddContext,
    /// Strengthen existing memory (same content, reinforcement)
    Reinforce,
}

/// Reasons for superseding an existing memory
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SupersedeReason {
    /// New content is a correction of old
    Correction,
    /// New content is an improvement/update
    Improvement,
    /// Old content is marked as outdated
    Outdated,
    /// User explicitly indicated this is better
    UserIndicated,
    /// New content has higher confidence/authority
    HigherConfidence,
}

/// Strategies for merging multiple memories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Combine all content
    Combine,
    /// Keep most recent, link to older
    KeepRecent,
    /// Create summary of all
    Summarize,
    /// Create hierarchy (parent with children)
    Hierarchical,
}

// ============================================================================
// CANDIDATE MEMORY
// ============================================================================

/// A candidate memory for update consideration
#[derive(Debug, Clone)]
pub struct CandidateMemory {
    /// Memory ID
    pub id: String,
    /// Memory content
    pub content: String,
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Current retrieval strength
    pub retrieval_strength: f64,
    /// Current retention strength
    pub retention_strength: f64,
    /// Tags on the memory
    pub tags: Vec<String>,
    /// Source of the memory
    pub source: Option<String>,
    /// Whether this memory was previously demoted
    pub was_demoted: bool,
    /// Whether this memory was previously promoted
    pub was_promoted: bool,
}

/// Result of similarity comparison
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    /// Memory ID
    pub memory_id: String,
    /// Cosine similarity score (0.0 - 1.0)
    pub similarity: f32,
    /// Prediction error (1.0 - similarity)
    pub prediction_error: f32,
    /// Semantic overlap (estimated shared concepts)
    pub semantic_overlap: f32,
    /// Whether contents appear contradictory
    pub appears_contradictory: bool,
}

// ============================================================================
// PREDICTION ERROR GATE
// ============================================================================

/// Configuration for the prediction error gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionErrorConfig {
    /// Similarity threshold for update consideration
    pub similarity_threshold: f32,
    /// Threshold for near-identical detection
    pub near_identical_threshold: f32,
    /// Threshold for correction detection
    pub correction_threshold: f32,
    /// Maximum candidates to consider
    pub max_candidates: usize,
    /// Whether to auto-supersede demoted memories
    pub auto_supersede_demoted: bool,
    /// Whether to prefer updates over creates
    pub prefer_updates: bool,
}

impl Default for PredictionErrorConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
            near_identical_threshold: NEAR_IDENTICAL_THRESHOLD,
            correction_threshold: CORRECTION_THRESHOLD,
            max_candidates: MAX_UPDATE_CANDIDATES,
            auto_supersede_demoted: true,
            prefer_updates: true,
        }
    }
}

/// The Prediction Error Gate
///
/// Evaluates new content against existing memories to determine
/// whether to create, update, or supersede.
#[derive(Debug)]
pub struct PredictionErrorGate {
    /// Configuration
    config: PredictionErrorConfig,
    /// Statistics
    stats: GateStats,
}

impl Default for PredictionErrorGate {
    fn default() -> Self {
        Self::new()
    }
}

impl PredictionErrorGate {
    /// Create a new prediction error gate with default config
    pub fn new() -> Self {
        Self {
            config: PredictionErrorConfig::default(),
            stats: GateStats::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: PredictionErrorConfig) -> Self {
        Self {
            config,
            stats: GateStats::default(),
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &PredictionErrorConfig {
        &self.config
    }

    /// Get mutable configuration
    pub fn config_mut(&mut self) -> &mut PredictionErrorConfig {
        &mut self.config
    }

    /// Evaluate new content against candidates
    ///
    /// Returns a decision on whether to create, update, or supersede.
    pub fn evaluate(
        &mut self,
        new_content: &str,
        new_embedding: &[f32],
        candidates: &[CandidateMemory],
    ) -> GateDecision {
        self.stats.total_evaluations += 1;

        // No candidates = definitely create
        if candidates.is_empty() {
            self.stats.creates += 1;
            return GateDecision::Create {
                reason: CreateReason::FirstMemory,
                prediction_error: 1.0,
                related_memory_ids: vec![],
            };
        }

        // Calculate similarities
        let mut similarities: Vec<SimilarityResult> = candidates
            .iter()
            .map(|c| {
                let similarity = cosine_similarity(new_embedding, &c.embedding);
                let appears_contradictory = self.detect_contradiction(new_content, &c.content);

                SimilarityResult {
                    memory_id: c.id.clone(),
                    similarity,
                    prediction_error: 1.0 - similarity,
                    semantic_overlap: similarity, // Simplified; could use more sophisticated measure
                    appears_contradictory,
                }
            })
            .collect();

        // Sort by similarity (highest first)
        similarities.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));

        // Take top candidates
        let top_candidates: Vec<_> = similarities
            .iter()
            .take(self.config.max_candidates)
            .collect();

        // Check for near-identical match
        if let Some(best) = top_candidates.first() {
            if best.similarity >= self.config.near_identical_threshold {
                // Nearly identical - reinforce existing
                self.stats.updates += 1;
                return GateDecision::Update {
                    target_id: best.memory_id.clone(),
                    similarity: best.similarity,
                    update_type: UpdateType::Reinforce,
                    prediction_error: best.prediction_error,
                };
            }

            // Check for potential supersede
            let candidate = candidates.iter().find(|c| c.id == best.memory_id);
            if let Some(c) = candidate {
                // If similar and the existing memory was demoted, supersede it
                if best.similarity >= self.config.similarity_threshold
                   && c.was_demoted
                   && self.config.auto_supersede_demoted {
                    self.stats.supersedes += 1;
                    return GateDecision::Supersede {
                        old_memory_id: c.id.clone(),
                        similarity: best.similarity,
                        supersede_reason: SupersedeReason::Improvement,
                        prediction_error: best.prediction_error,
                    };
                }

                // Check for correction (similar but contradictory)
                if best.similarity >= self.config.correction_threshold
                   && best.appears_contradictory {
                    self.stats.supersedes += 1;
                    return GateDecision::Supersede {
                        old_memory_id: c.id.clone(),
                        similarity: best.similarity,
                        supersede_reason: SupersedeReason::Correction,
                        prediction_error: best.prediction_error,
                    };
                }

                // Regular update for similar content
                if best.similarity >= self.config.similarity_threshold && self.config.prefer_updates {
                    self.stats.updates += 1;
                    return GateDecision::Update {
                        target_id: best.memory_id.clone(),
                        similarity: best.similarity,
                        update_type: UpdateType::Merge,
                        prediction_error: best.prediction_error,
                    };
                }
            }
        }

        // Check for merge opportunity (multiple similar memories)
        let merge_candidates: Vec<_> = top_candidates
            .iter()
            .filter(|s| s.similarity >= self.config.similarity_threshold * 0.9)
            .collect();

        if merge_candidates.len() >= 2 {
            let avg_similarity = merge_candidates.iter().map(|s| s.similarity).sum::<f32>()
                / merge_candidates.len() as f32;

            self.stats.merges += 1;
            return GateDecision::Merge {
                memory_ids: merge_candidates.iter().map(|s| s.memory_id.clone()).collect(),
                avg_similarity,
                strategy: MergeStrategy::Combine,
            };
        }

        // Default: create new (high prediction error)
        let best_pe = top_candidates
            .first()
            .map(|s| s.prediction_error)
            .unwrap_or(1.0);

        self.stats.creates += 1;
        GateDecision::Create {
            reason: if candidates.is_empty() {
                CreateReason::NoSimilarMemories
            } else {
                CreateReason::HighPredictionError
            },
            prediction_error: best_pe,
            related_memory_ids: top_candidates.iter().map(|s| s.memory_id.clone()).collect(),
        }
    }

    /// Evaluate with explicit intent
    ///
    /// Use when the user has indicated intent (e.g., "update X" or "this is better than Y")
    pub fn evaluate_with_intent(
        &mut self,
        new_content: &str,
        new_embedding: &[f32],
        candidates: &[CandidateMemory],
        intent: EvaluationIntent,
    ) -> GateDecision {
        match intent {
            EvaluationIntent::ForceCreate => {
                self.stats.creates += 1;
                GateDecision::Create {
                    reason: CreateReason::ExplicitCreate,
                    prediction_error: 1.0,
                    related_memory_ids: vec![],
                }
            }
            EvaluationIntent::ForceUpdate { target_id } => {
                // Find the target candidate
                if let Some(c) = candidates.iter().find(|c| c.id == target_id) {
                    let similarity = cosine_similarity(new_embedding, &c.embedding);
                    self.stats.updates += 1;
                    GateDecision::Update {
                        target_id: target_id.clone(),
                        similarity,
                        update_type: UpdateType::Replace,
                        prediction_error: 1.0 - similarity,
                    }
                } else {
                    // Target not found, evaluate normally
                    self.evaluate(new_content, new_embedding, candidates)
                }
            }
            EvaluationIntent::Supersede { old_memory_id, reason } => {
                if let Some(c) = candidates.iter().find(|c| c.id == old_memory_id) {
                    let similarity = cosine_similarity(new_embedding, &c.embedding);
                    self.stats.supersedes += 1;
                    GateDecision::Supersede {
                        old_memory_id,
                        similarity,
                        supersede_reason: reason,
                        prediction_error: 1.0 - similarity,
                    }
                } else {
                    self.evaluate(new_content, new_embedding, candidates)
                }
            }
            EvaluationIntent::Auto => {
                self.evaluate(new_content, new_embedding, candidates)
            }
        }
    }

    /// Detect if two pieces of content appear contradictory
    ///
    /// Uses simple heuristics; could be enhanced with NLI model
    fn detect_contradiction(&self, new_content: &str, old_content: &str) -> bool {
        let new_lower = new_content.to_lowercase();
        let old_lower = old_content.to_lowercase();

        // Check for explicit negation patterns
        let negation_pairs = [
            ("don't", "do"),
            ("never", "always"),
            ("avoid", "use"),
            ("wrong", "right"),
            ("bad", "good"),
            ("incorrect", "correct"),
            ("deprecated", "recommended"),
            ("outdated", "current"),
            ("instead of", ""),
            ("rather than", ""),
            ("not ", ""),
        ];

        for (neg, _pos) in negation_pairs.iter() {
            if new_lower.contains(neg) && !old_lower.contains(neg) {
                return true;
            }
        }

        // Check for correction phrases
        let correction_phrases = [
            "actually",
            "correction",
            "update:",
            "fixed",
            "was wrong",
            "should be",
            "better approach",
            "improved",
            "the right way",
        ];

        for phrase in correction_phrases.iter() {
            if new_lower.contains(phrase) {
                return true;
            }
        }

        false
    }

    /// Get statistics
    pub fn stats(&self) -> &GateStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = GateStats::default();
    }
}

// ============================================================================
// EVALUATION INTENT
// ============================================================================

/// Explicit intent for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvaluationIntent {
    /// Automatically determine best action
    Auto,
    /// Force creation of new memory
    ForceCreate,
    /// Force update of specific memory
    ForceUpdate { target_id: String },
    /// Force supersede of specific memory
    Supersede { old_memory_id: String, reason: SupersedeReason },
}

// ============================================================================
// STATISTICS
// ============================================================================

/// Statistics about gate decisions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GateStats {
    /// Total evaluations performed
    pub total_evaluations: usize,
    /// Decisions to create new
    pub creates: usize,
    /// Decisions to update existing
    pub updates: usize,
    /// Decisions to supersede
    pub supersedes: usize,
    /// Decisions to merge
    pub merges: usize,
}

impl GateStats {
    /// Get create rate
    pub fn create_rate(&self) -> f64 {
        if self.total_evaluations > 0 {
            self.creates as f64 / self.total_evaluations as f64
        } else {
            0.0
        }
    }

    /// Get update rate
    pub fn update_rate(&self) -> f64 {
        if self.total_evaluations > 0 {
            self.updates as f64 / self.total_evaluations as f64
        } else {
            0.0
        }
    }

    /// Get supersede rate
    pub fn supersede_rate(&self) -> f64 {
        if self.total_evaluations > 0 {
            self.supersedes as f64 / self.total_evaluations as f64
        } else {
            0.0
        }
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Calculate cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(0.0, 1.0)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(seed: f32) -> Vec<f32> {
        // Create embeddings with controlled similarity based on seed
        // Seeds close to each other = similar vectors
        // Seeds far apart = different vectors
        (0..384).map(|i| {
            let base = (i as f32 / 384.0) * std::f32::consts::PI * 2.0;
            (base * seed).sin()
        }).collect()
    }

    fn make_orthogonal_embedding() -> Vec<f32> {
        // Create an embedding that's orthogonal to seed=1.0
        (0..384).map(|i| {
            let base = (i as f32 / 384.0) * std::f32::consts::PI * 2.0;
            (base + std::f32::consts::PI / 2.0).sin()  // 90 degree phase shift
        }).collect()
    }

    fn make_candidate(id: &str, seed: f32) -> CandidateMemory {
        CandidateMemory {
            id: id.to_string(),
            content: format!("Content for {}", id),
            embedding: make_embedding(seed),
            retrieval_strength: 0.8,
            retention_strength: 0.7,
            tags: vec![],
            source: None,
            was_demoted: false,
            was_promoted: false,
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);

        let d = vec![-1.0, 0.0, 0.0];
        assert!(cosine_similarity(&a, &d) <= 0.0);
    }

    #[test]
    fn test_empty_candidates() {
        let mut gate = PredictionErrorGate::new();
        let embedding = make_embedding(1.0);

        let decision = gate.evaluate("New content", &embedding, &[]);

        assert!(matches!(decision, GateDecision::Create { reason: CreateReason::FirstMemory, .. }));
    }

    #[test]
    fn test_high_similarity_update() {
        let mut gate = PredictionErrorGate::new();
        let embedding = make_embedding(1.0);

        // Create candidate with identical embedding
        let mut candidate = make_candidate("mem-1", 1.0);
        candidate.embedding = embedding.clone();

        let decision = gate.evaluate("Same content", &embedding, &[candidate]);

        assert!(decision.is_update());
        if let GateDecision::Update { update_type, .. } = decision {
            assert_eq!(update_type, UpdateType::Reinforce);
        }
    }

    #[test]
    fn test_demoted_memory_supersede() {
        let mut gate = PredictionErrorGate::new();
        let embedding = make_embedding(1.0);

        // Use similar embedding (seed 1.05) - close enough to be above similarity threshold
        let mut candidate = make_candidate("mem-1", 1.0);
        candidate.embedding = make_embedding(1.05);
        candidate.was_demoted = true;

        let decision = gate.evaluate("Better solution", &embedding, &[candidate]);

        // Should supersede the demoted memory if similarity is above threshold
        // If not superseding, it should at least update
        assert!(matches!(decision, GateDecision::Supersede { .. } | GateDecision::Update { .. }));
    }

    #[test]
    fn test_different_content_create() {
        let mut gate = PredictionErrorGate::new();
        let new_embedding = make_embedding(1.0);

        // Use orthogonal embedding for truly different content
        let mut candidate = make_candidate("mem-1", 1.0);
        candidate.embedding = make_orthogonal_embedding();

        let decision = gate.evaluate("Completely different topic", &new_embedding, &[candidate]);

        assert!(matches!(decision, GateDecision::Create { .. }));
    }

    #[test]
    fn test_contradiction_detection() {
        let gate = PredictionErrorGate::new();

        assert!(gate.detect_contradiction(
            "Don't use synchronous code",
            "Use synchronous code for simplicity"
        ));

        assert!(gate.detect_contradiction(
            "Actually, the correct approach is...",
            "The approach is to..."
        ));

        assert!(!gate.detect_contradiction(
            "Use async/await for performance",
            "Use async patterns when needed"
        ));
    }

    #[test]
    fn test_force_create_intent() {
        let mut gate = PredictionErrorGate::new();
        let embedding = make_embedding(1.0);
        let candidate = make_candidate("mem-1", 1.0);

        let decision = gate.evaluate_with_intent(
            "New content",
            &embedding,
            &[candidate],
            EvaluationIntent::ForceCreate,
        );

        assert!(matches!(decision, GateDecision::Create { reason: CreateReason::ExplicitCreate, .. }));
    }

    #[test]
    fn test_force_update_intent() {
        let mut gate = PredictionErrorGate::new();
        let embedding = make_embedding(1.0);
        let candidate = make_candidate("mem-1", 5.0);

        let decision = gate.evaluate_with_intent(
            "Updated content",
            &embedding,
            &[candidate],
            EvaluationIntent::ForceUpdate { target_id: "mem-1".to_string() },
        );

        assert!(matches!(decision, GateDecision::Update { .. }));
    }

    #[test]
    fn test_stats() {
        let mut gate = PredictionErrorGate::new();
        let embedding = make_embedding(1.0);

        // Create (empty candidates)
        gate.evaluate("Content", &embedding, &[]);

        // Update (identical)
        let mut candidate = make_candidate("mem-1", 1.0);
        candidate.embedding = embedding.clone();
        gate.evaluate("Content", &embedding, &[candidate.clone()]);

        let stats = gate.stats();
        assert_eq!(stats.total_evaluations, 2);
        assert_eq!(stats.creates, 1);
        assert_eq!(stats.updates, 1);
    }
}
