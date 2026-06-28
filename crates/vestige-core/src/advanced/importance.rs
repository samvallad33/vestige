//! # Memory Importance Evolution
//!
//! Memories evolve in importance based on actual usage patterns.
//! Unlike static importance scores, this system learns which memories
//! are truly valuable over time.
//!
//! ## Importance Factors
//!
//! - **Base Importance**: Initial importance from content analysis
//! - **Usage Importance**: Derived from how often a memory is retrieved and found helpful
//! - **Recency Importance**: Recent memories get a boost
//! - **Connection Importance**: Well-connected memories are more valuable
//! - **Decay Factor**: Unused memories naturally decay in importance
//!
//! ## Example
//!
//! ```rust,ignore
//! let tracker = ImportanceTracker::new();
//!
//! // Record usage
//! tracker.on_retrieved("mem-123", true);  // Was helpful
//! tracker.on_retrieved("mem-456", false); // Not helpful
//!
//! // Apply daily decay
//! tracker.apply_importance_decay();
//!
//! // Get weighted search results
//! let weighted = tracker.weight_by_importance(results);
//! ```

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Default decay rate per day (5% decay)
const DEFAULT_DECAY_RATE: f64 = 0.95;

/// Minimum importance (never goes to zero)
const MIN_IMPORTANCE: f64 = 0.01;

/// Maximum importance cap
const MAX_IMPORTANCE: f64 = 1.0;

/// Boost factor when memory is helpful
const HELPFUL_BOOST: f64 = 1.15;

/// Penalty factor when memory is retrieved but not helpful
const UNHELPFUL_PENALTY: f64 = 0.95;

/// Importance score components for a memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceScore {
    /// Memory ID
    pub memory_id: String,
    /// Base importance from content analysis (0.0 to 1.0)
    pub base_importance: f64,
    /// Importance derived from actual usage patterns (0.0 to 1.0)
    pub usage_importance: f64,
    /// Recency-based importance boost (0.0 to 1.0)
    pub recency_importance: f64,
    /// Importance from being connected to other memories (0.0 to 1.0)
    pub connection_importance: f64,
    /// Final computed importance score (0.0 to 1.0)
    pub final_score: f64,
    /// Number of times retrieved
    pub retrieval_count: u32,
    /// Number of times found helpful
    pub helpful_count: u32,
    /// Last time this memory was accessed
    pub last_accessed: Option<DateTime<Utc>>,
    /// When this importance was last calculated
    pub calculated_at: DateTime<Utc>,
}

impl ImportanceScore {
    /// Create a new importance score with default values
    pub fn new(memory_id: &str) -> Self {
        Self {
            memory_id: memory_id.to_string(),
            base_importance: 0.5,
            usage_importance: 0.1, // Start low - must prove useful through retrieval
            recency_importance: 0.5,
            connection_importance: 0.0,
            final_score: 0.5,
            retrieval_count: 0,
            helpful_count: 0,
            last_accessed: None,
            calculated_at: Utc::now(),
        }
    }

    /// Calculate the final importance score from all factors
    pub fn calculate_final(&mut self) {
        // Weighted combination of factors
        const BASE_WEIGHT: f64 = 0.2;
        const USAGE_WEIGHT: f64 = 0.4;
        const RECENCY_WEIGHT: f64 = 0.25;
        const CONNECTION_WEIGHT: f64 = 0.15;

        self.final_score = (self.base_importance * BASE_WEIGHT
            + self.usage_importance * USAGE_WEIGHT
            + self.recency_importance * RECENCY_WEIGHT
            + self.connection_importance * CONNECTION_WEIGHT)
            .clamp(MIN_IMPORTANCE, MAX_IMPORTANCE);

        self.calculated_at = Utc::now();
    }

    /// Get the helpfulness ratio (helpful / total)
    pub fn helpfulness_ratio(&self) -> f64 {
        if self.retrieval_count == 0 {
            return 0.5; // Default when no data
        }
        self.helpful_count as f64 / self.retrieval_count as f64
    }
}

/// A usage event for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEvent {
    /// Memory ID that was used
    pub memory_id: String,
    /// Whether the usage was helpful
    pub was_helpful: bool,
    /// Context in which it was used
    pub context: Option<String>,
    /// When this event occurred
    pub timestamp: DateTime<Utc>,
}

/// Configuration for importance decay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceDecayConfig {
    /// Decay rate per day (0.95 = 5% decay)
    pub decay_rate: f64,
    /// Minimum importance (never decays below this)
    pub min_importance: f64,
    /// Maximum importance cap
    pub max_importance: f64,
    /// Days of inactivity before decay starts
    pub grace_period_days: u32,
    /// Recency half-life in days
    pub recency_half_life_days: f64,
}

impl Default for ImportanceDecayConfig {
    fn default() -> Self {
        Self {
            decay_rate: DEFAULT_DECAY_RATE,
            min_importance: MIN_IMPORTANCE,
            max_importance: MAX_IMPORTANCE,
            grace_period_days: 7,
            recency_half_life_days: 14.0,
        }
    }
}

/// Tracks and evolves memory importance over time
pub struct ImportanceTracker {
    /// Importance scores by memory ID
    scores: Arc<RwLock<HashMap<String, ImportanceScore>>>,
    /// Recent usage events for pattern analysis
    recent_events: Arc<RwLock<Vec<UsageEvent>>>,
    /// Configuration
    config: ImportanceDecayConfig,
}

impl ImportanceTracker {
    /// Create a new importance tracker with default config
    pub fn new() -> Self {
        Self::with_config(ImportanceDecayConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ImportanceDecayConfig) -> Self {
        Self {
            scores: Arc::new(RwLock::new(HashMap::new())),
            recent_events: Arc::new(RwLock::new(Vec::new())),
            config,
        }
    }

    /// Update importance when a memory is retrieved
    pub fn on_retrieved(&self, memory_id: &str, was_helpful: bool) {
        self.record_retrieval(memory_id, was_helpful, None);
    }

    /// Push a usage event (with its context already populated) and update the
    /// importance score. Context is set in the SAME critical section as the push
    /// so a concurrent on_retrieved cannot slip an event in between and steal the
    /// context via last_mut() (the previous two-lock approach raced).
    fn record_retrieval(&self, memory_id: &str, was_helpful: bool, context: Option<String>) {
        let now = Utc::now();

        // Record the event
        if let Ok(mut events) = self.recent_events.write() {
            events.push(UsageEvent {
                memory_id: memory_id.to_string(),
                was_helpful,
                context,
                timestamp: now,
            });

            // Keep only recent events (last 30 days)
            let cutoff = now - Duration::days(30);
            events.retain(|e| e.timestamp > cutoff);
        }

        // Update importance score
        if let Ok(mut scores) = self.scores.write() {
            let score = scores
                .entry(memory_id.to_string())
                .or_insert_with(|| ImportanceScore::new(memory_id));

            score.retrieval_count += 1;
            score.last_accessed = Some(now);

            if was_helpful {
                score.helpful_count += 1;
                score.usage_importance =
                    (score.usage_importance * HELPFUL_BOOST).min(self.config.max_importance);
            } else {
                score.usage_importance =
                    (score.usage_importance * UNHELPFUL_PENALTY).max(self.config.min_importance);
            }

            // Update recency importance (always high when just accessed)
            score.recency_importance = 1.0;

            // Recalculate final score
            score.calculate_final();
        }
    }

    /// Update importance with additional context
    pub fn on_retrieved_with_context(&self, memory_id: &str, was_helpful: bool, context: &str) {
        self.record_retrieval(memory_id, was_helpful, Some(context.to_string()));
    }

    /// Apply importance decay to all memories
    pub fn apply_importance_decay(&self) {
        let now = Utc::now();

        if let Ok(mut scores) = self.scores.write() {
            for score in scores.values_mut() {
                // Calculate days since last access
                let days_inactive = score
                    .last_accessed
                    .map(|last| (now - last).num_days() as u32)
                    .unwrap_or(self.config.grace_period_days + 1);

                // Apply decay if past grace period
                if days_inactive > self.config.grace_period_days {
                    let decay_days = days_inactive - self.config.grace_period_days;
                    let decay_factor = self.config.decay_rate.powi(decay_days as i32);

                    score.usage_importance =
                        (score.usage_importance * decay_factor).max(self.config.min_importance);
                }

                // Apply recency decay
                let recency_days = score
                    .last_accessed
                    .map(|last| (now - last).num_days() as f64)
                    .unwrap_or(self.config.recency_half_life_days * 2.0);

                score.recency_importance =
                    0.5_f64.powf(recency_days / self.config.recency_half_life_days);

                // Recalculate final score
                score.calculate_final();
            }
        }
    }

    /// Weight search results by importance
    pub fn weight_by_importance<T: HasMemoryId + Clone>(
        &self,
        results: Vec<T>,
    ) -> Vec<WeightedResult<T>> {
        let scores = self.scores.read().ok();

        results
            .into_iter()
            .map(|result| {
                let importance = scores
                    .as_ref()
                    .and_then(|s| s.get(result.memory_id()))
                    .map(|s| s.final_score)
                    .unwrap_or(0.5);

                WeightedResult { result, importance }
            })
            .collect()
    }

    /// Get importance score for a specific memory
    pub fn get_importance(&self, memory_id: &str) -> Option<ImportanceScore> {
        self.scores
            .read()
            .ok()
            .and_then(|scores| scores.get(memory_id).cloned())
    }

    /// Set base importance for a memory (from content analysis)
    pub fn set_base_importance(&self, memory_id: &str, base_importance: f64) {
        if let Ok(mut scores) = self.scores.write() {
            let score = scores
                .entry(memory_id.to_string())
                .or_insert_with(|| ImportanceScore::new(memory_id));

            score.base_importance =
                base_importance.clamp(self.config.min_importance, self.config.max_importance);
            score.calculate_final();
        }
    }

    /// Set connection importance for a memory (from graph analysis)
    pub fn set_connection_importance(&self, memory_id: &str, connection_importance: f64) {
        if let Ok(mut scores) = self.scores.write() {
            let score = scores
                .entry(memory_id.to_string())
                .or_insert_with(|| ImportanceScore::new(memory_id));

            score.connection_importance =
                connection_importance.clamp(self.config.min_importance, self.config.max_importance);
            score.calculate_final();
        }
    }

    /// Get all importance scores
    pub fn get_all_scores(&self) -> Vec<ImportanceScore> {
        self.scores
            .read()
            .map(|scores| scores.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Get memories sorted by importance
    pub fn get_top_by_importance(&self, limit: usize) -> Vec<ImportanceScore> {
        let mut scores = self.get_all_scores();
        scores.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(limit);
        scores
    }

    /// Get memories that need attention (low importance but high base)
    pub fn get_neglected_memories(&self, limit: usize) -> Vec<ImportanceScore> {
        let mut scores: Vec<_> = self
            .get_all_scores()
            .into_iter()
            .filter(|s| s.base_importance > 0.6 && s.usage_importance < 0.3)
            .collect();

        scores.sort_by(|a, b| {
            let a_neglect = a.base_importance - a.usage_importance;
            let b_neglect = b.base_importance - b.usage_importance;
            b_neglect
                .partial_cmp(&a_neglect)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scores.truncate(limit);
        scores
    }

    /// Clear all importance data (for testing)
    pub fn clear(&self) {
        if let Ok(mut scores) = self.scores.write() {
            scores.clear();
        }
        if let Ok(mut events) = self.recent_events.write() {
            events.clear();
        }
    }
}

impl Default for ImportanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that have a memory ID
pub trait HasMemoryId {
    fn memory_id(&self) -> &str;
}

/// A result weighted by importance
#[derive(Debug, Clone)]
pub struct WeightedResult<T> {
    /// The original result
    pub result: T,
    /// Importance weight (0.0 to 1.0)
    pub importance: f64,
}

impl<T> WeightedResult<T> {
    /// Get combined score (e.g., relevance * importance)
    pub fn combined_score(&self, relevance: f64) -> f64 {
        // Importance adjusts relevance by up to +/- 30%
        relevance * (0.7 + 0.6 * self.importance)
    }
}

/// Simple memory ID wrapper for search results
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f64,
}

impl HasMemoryId for SearchResult {
    fn memory_id(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_importance_score_calculation() {
        let mut score = ImportanceScore::new("test-mem");
        score.base_importance = 0.8;
        score.usage_importance = 0.9;
        score.recency_importance = 1.0;
        score.connection_importance = 0.5;
        score.calculate_final();

        // Should be weighted combination
        assert!(score.final_score > 0.7);
        assert!(score.final_score < 1.0);
    }

    #[test]
    fn test_on_retrieved_helpful() {
        let tracker = ImportanceTracker::new();

        // Default usage_importance starts at 0.1
        // Each helpful retrieval multiplies by HELPFUL_BOOST (1.15)
        tracker.on_retrieved("mem-1", true);
        tracker.on_retrieved("mem-1", true);
        tracker.on_retrieved("mem-1", true);

        let score = tracker.get_importance("mem-1").unwrap();
        assert_eq!(score.retrieval_count, 3);
        assert_eq!(score.helpful_count, 3);
        // 0.1 * 1.15^3 = ~0.152, so should be > initial 0.1
        assert!(
            score.usage_importance > 0.1,
            "Should be boosted from baseline"
        );
    }

    #[test]
    fn test_on_retrieved_unhelpful() {
        let tracker = ImportanceTracker::new();

        tracker.on_retrieved("mem-1", false);
        tracker.on_retrieved("mem-1", false);
        tracker.on_retrieved("mem-1", false);

        let score = tracker.get_importance("mem-1").unwrap();
        assert_eq!(score.retrieval_count, 3);
        assert_eq!(score.helpful_count, 0);
        assert!(score.usage_importance < 0.5); // Should be penalized
    }

    #[test]
    fn test_helpfulness_ratio() {
        let mut score = ImportanceScore::new("test");
        score.retrieval_count = 10;
        score.helpful_count = 7;

        assert!((score.helpfulness_ratio() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_neglected_memories() {
        let tracker = ImportanceTracker::new();

        // Create a "neglected" memory: high base importance, low usage
        tracker.set_base_importance("neglected", 0.9);
        // Don't retrieve it, so usage stays low

        // Create a well-used memory
        tracker.set_base_importance("used", 0.5);
        tracker.on_retrieved("used", true);
        tracker.on_retrieved("used", true);

        let neglected = tracker.get_neglected_memories(10);
        assert!(!neglected.is_empty());
        assert_eq!(neglected[0].memory_id, "neglected");
    }
}
