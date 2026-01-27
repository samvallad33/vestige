//! # Speculative Memory Retrieval
//!
//! Predict what memories the user will need BEFORE they ask.
//! Uses pattern analysis, temporal modeling, and context understanding
//! to pre-warm the cache with likely-needed memories.
//!
//! ## How It Works
//!
//! 1. Analyzes current working context (files open, recent queries, project state)
//! 2. Learns from historical access patterns (what memories were accessed together)
//! 3. Predicts with confidence scores and reasoning
//! 4. Pre-fetches high-confidence predictions into fast cache
//! 5. Records actual usage to improve future predictions
//!
//! ## Example
//!
//! ```rust,ignore
//! let retriever = SpeculativeRetriever::new(storage);
//!
//! // When user opens auth.rs, predict they'll need JWT memories
//! let predictions = retriever.predict_needed(&context);
//!
//! // Pre-warm cache in background
//! retriever.prefetch(&context).await?;
//! ```

use chrono::{DateTime, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Maximum number of access patterns to track
const MAX_PATTERN_HISTORY: usize = 10_000;

/// Maximum predictions to return
const MAX_PREDICTIONS: usize = 20;

/// Minimum confidence threshold for predictions
const MIN_CONFIDENCE: f64 = 0.3;

/// Decay factor for old patterns (per day)
const PATTERN_DECAY_RATE: f64 = 0.95;

/// A predicted memory that the user is likely to need
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedMemory {
    /// The memory ID that's predicted to be needed
    pub memory_id: String,
    /// Content preview for quick reference
    pub content_preview: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Human-readable reasoning for this prediction
    pub reasoning: String,
    /// What triggered this prediction
    pub trigger: PredictionTrigger,
    /// When this prediction was made
    pub predicted_at: DateTime<Utc>,
}

/// What triggered a prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictionTrigger {
    /// Based on file being opened/edited
    FileContext { file_path: String },
    /// Based on co-access patterns
    CoAccessPattern { related_memory_id: String },
    /// Based on time-of-day patterns
    TemporalPattern { typical_time: String },
    /// Based on project context
    ProjectContext { project_name: String },
    /// Based on detected intent
    IntentBased { intent: String },
    /// Based on semantic similarity to recent queries
    SemanticSimilarity { query: String, similarity: f64 },
}

/// Context for making predictions
#[derive(Debug, Clone, Default)]
pub struct PredictionContext {
    /// Currently open files
    pub open_files: Vec<PathBuf>,
    /// Recent file edits
    pub recent_edits: Vec<PathBuf>,
    /// Recent search queries
    pub recent_queries: Vec<String>,
    /// Recently accessed memory IDs
    pub recent_memory_ids: Vec<String>,
    /// Current project path
    pub project_path: Option<PathBuf>,
    /// Current timestamp
    pub timestamp: Option<DateTime<Utc>>,
}

impl PredictionContext {
    /// Create a new prediction context
    pub fn new() -> Self {
        Self {
            timestamp: Some(Utc::now()),
            ..Default::default()
        }
    }

    /// Add an open file to context
    pub fn with_file(mut self, path: PathBuf) -> Self {
        self.open_files.push(path);
        self
    }

    /// Add a recent query to context
    pub fn with_query(mut self, query: String) -> Self {
        self.recent_queries.push(query);
        self
    }

    /// Set the project path
    pub fn with_project(mut self, path: PathBuf) -> Self {
        self.project_path = Some(path);
        self
    }
}

/// A learned co-access pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePattern {
    /// The trigger memory ID
    pub trigger_id: String,
    /// The predicted memory ID
    pub predicted_id: String,
    /// How often this pattern occurred
    pub frequency: u32,
    /// Success rate (was the prediction useful)
    pub success_rate: f64,
    /// Last time this pattern was observed
    pub last_seen: DateTime<Utc>,
    /// Weight after decay applied
    pub weight: f64,
}

/// Speculative memory retriever that predicts needed memories
pub struct SpeculativeRetriever {
    /// Co-access patterns: trigger_id -> Vec<(predicted_id, pattern)>
    co_access_patterns: Arc<RwLock<HashMap<String, Vec<UsagePattern>>>>,
    /// File-to-memory associations
    file_memory_map: Arc<RwLock<HashMap<String, Vec<String>>>>,
    /// Recent access sequence for pattern detection
    access_sequence: Arc<RwLock<VecDeque<AccessEvent>>>,
    /// Pending predictions (for recording outcomes)
    pending_predictions: Arc<RwLock<HashMap<String, PredictedMemory>>>,
    /// Cache of recently predicted memories
    prediction_cache: Arc<RwLock<Vec<PredictedMemory>>>,
}

/// An access event for pattern learning
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AccessEvent {
    memory_id: String,
    file_context: Option<String>,
    query_context: Option<String>,
    timestamp: DateTime<Utc>,
    was_helpful: Option<bool>,
}

impl SpeculativeRetriever {
    /// Create a new speculative retriever
    pub fn new() -> Self {
        Self {
            co_access_patterns: Arc::new(RwLock::new(HashMap::new())),
            file_memory_map: Arc::new(RwLock::new(HashMap::new())),
            access_sequence: Arc::new(RwLock::new(VecDeque::with_capacity(MAX_PATTERN_HISTORY))),
            pending_predictions: Arc::new(RwLock::new(HashMap::new())),
            prediction_cache: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Predict memories that will be needed based on context
    pub fn predict_needed(&self, context: &PredictionContext) -> Vec<PredictedMemory> {
        let mut predictions: Vec<PredictedMemory> = Vec::new();
        let now = context.timestamp.unwrap_or_else(Utc::now);

        // 1. File-based predictions
        predictions.extend(self.predict_from_files(context, now));

        // 2. Co-access pattern predictions
        predictions.extend(self.predict_from_patterns(context, now));

        // 3. Query similarity predictions
        predictions.extend(self.predict_from_queries(context, now));

        // 4. Temporal pattern predictions
        predictions.extend(self.predict_from_time(now));

        // Deduplicate and sort by confidence
        predictions = self.deduplicate_predictions(predictions);
        predictions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        predictions.truncate(MAX_PREDICTIONS);

        // Filter by minimum confidence
        predictions.retain(|p| p.confidence >= MIN_CONFIDENCE);

        // Store for outcome tracking
        self.store_pending_predictions(&predictions);

        predictions
    }

    /// Pre-warm cache with predicted memories
    pub async fn prefetch(&self, context: &PredictionContext) -> Result<usize, SpeculativeError> {
        let predictions = self.predict_needed(context);
        let count = predictions.len();

        // Store predictions in cache for fast access
        if let Ok(mut cache) = self.prediction_cache.write() {
            *cache = predictions;
        }

        Ok(count)
    }

    /// Record what was actually used to improve future predictions
    pub fn record_usage(&self, _predicted: &[String], actually_used: &[String]) {
        // Update pending predictions with outcomes
        if let Ok(mut pending) = self.pending_predictions.write() {
            for id in actually_used {
                if let Some(prediction) = pending.remove(id) {
                    // This was correctly predicted - strengthen pattern
                    self.strengthen_pattern(&prediction.memory_id, 1.0);
                }
            }

            // Weaken patterns for predictions that weren't used
            for (id, _) in pending.drain() {
                self.weaken_pattern(&id, 0.9);
            }
        }

        // Learn new co-access patterns
        self.learn_co_access_patterns(actually_used);
    }

    /// Record a memory access event
    pub fn record_access(
        &self,
        memory_id: &str,
        file_context: Option<&str>,
        query_context: Option<&str>,
        was_helpful: Option<bool>,
    ) {
        let event = AccessEvent {
            memory_id: memory_id.to_string(),
            file_context: file_context.map(String::from),
            query_context: query_context.map(String::from),
            timestamp: Utc::now(),
            was_helpful,
        };

        if let Ok(mut sequence) = self.access_sequence.write() {
            sequence.push_back(event.clone());

            // Trim old events
            while sequence.len() > MAX_PATTERN_HISTORY {
                sequence.pop_front();
            }
        }

        // Update file-memory associations
        if let Some(file) = file_context {
            if let Ok(mut map) = self.file_memory_map.write() {
                map.entry(file.to_string())
                    .or_insert_with(Vec::new)
                    .push(memory_id.to_string());
            }
        }
    }

    /// Get cached predictions
    pub fn get_cached_predictions(&self) -> Vec<PredictedMemory> {
        self.prediction_cache
            .read()
            .map(|cache| cache.clone())
            .unwrap_or_default()
    }

    /// Apply decay to old patterns
    pub fn apply_pattern_decay(&self) {
        if let Ok(mut patterns) = self.co_access_patterns.write() {
            let now = Utc::now();

            for patterns_list in patterns.values_mut() {
                for pattern in patterns_list.iter_mut() {
                    let days_old = (now - pattern.last_seen).num_days() as f64;
                    pattern.weight *= PATTERN_DECAY_RATE.powf(days_old);
                }

                // Remove patterns that are too weak
                patterns_list.retain(|p| p.weight > 0.01);
            }
        }
    }

    // ========================================================================
    // Private prediction methods
    // ========================================================================

    fn predict_from_files(
        &self,
        context: &PredictionContext,
        now: DateTime<Utc>,
    ) -> Vec<PredictedMemory> {
        let mut predictions = Vec::new();

        if let Ok(file_map) = self.file_memory_map.read() {
            for file in &context.open_files {
                let file_str = file.to_string_lossy().to_string();
                if let Some(memory_ids) = file_map.get(&file_str) {
                    for memory_id in memory_ids {
                        predictions.push(PredictedMemory {
                            memory_id: memory_id.clone(),
                            content_preview: String::new(), // Would be filled by storage lookup
                            confidence: 0.7,
                            reasoning: format!(
                                "You're working on {}, and this memory was useful for that file before",
                                file.file_name().unwrap_or_default().to_string_lossy()
                            ),
                            trigger: PredictionTrigger::FileContext {
                                file_path: file_str.clone()
                            },
                            predicted_at: now,
                        });
                    }
                }
            }
        }

        predictions
    }

    fn predict_from_patterns(
        &self,
        context: &PredictionContext,
        now: DateTime<Utc>,
    ) -> Vec<PredictedMemory> {
        let mut predictions = Vec::new();

        if let Ok(patterns) = self.co_access_patterns.read() {
            for recent_id in &context.recent_memory_ids {
                if let Some(related_patterns) = patterns.get(recent_id) {
                    for pattern in related_patterns {
                        let confidence = pattern.weight * pattern.success_rate;
                        if confidence >= MIN_CONFIDENCE {
                            predictions.push(PredictedMemory {
                                memory_id: pattern.predicted_id.clone(),
                                content_preview: String::new(),
                                confidence,
                                reasoning: format!(
                                    "You accessed a related memory, and these are often used together ({}% of the time)",
                                    (pattern.success_rate * 100.0) as u32
                                ),
                                trigger: PredictionTrigger::CoAccessPattern {
                                    related_memory_id: recent_id.clone()
                                },
                                predicted_at: now,
                            });
                        }
                    }
                }
            }
        }

        predictions
    }

    fn predict_from_queries(
        &self,
        context: &PredictionContext,
        now: DateTime<Utc>,
    ) -> Vec<PredictedMemory> {
        // In a full implementation, this would use semantic similarity
        // to find memories similar to recent queries
        let mut predictions = Vec::new();

        if let Ok(sequence) = self.access_sequence.read() {
            for query in &context.recent_queries {
                // Find memories accessed after similar queries
                for event in sequence.iter().rev().take(100) {
                    if let Some(event_query) = &event.query_context {
                        // Simple substring matching (would use embeddings in production)
                        if event_query.to_lowercase().contains(&query.to_lowercase())
                            || query.to_lowercase().contains(&event_query.to_lowercase())
                        {
                            predictions.push(PredictedMemory {
                                memory_id: event.memory_id.clone(),
                                content_preview: String::new(),
                                confidence: 0.6,
                                reasoning: "This memory was helpful when you searched for similar terms before".to_string(),
                                trigger: PredictionTrigger::SemanticSimilarity {
                                    query: query.clone(),
                                    similarity: 0.8,
                                },
                                predicted_at: now,
                            });
                        }
                    }
                }
            }
        }

        predictions
    }

    fn predict_from_time(&self, now: DateTime<Utc>) -> Vec<PredictedMemory> {
        let mut predictions = Vec::new();
        let hour = now.hour();

        if let Ok(sequence) = self.access_sequence.read() {
            // Find memories frequently accessed at this time of day
            let mut time_counts: HashMap<String, u32> = HashMap::new();

            for event in sequence.iter() {
                if (event.timestamp.hour() as i32 - hour as i32).abs() <= 1 {
                    *time_counts.entry(event.memory_id.clone()).or_insert(0) += 1;
                }
            }

            for (memory_id, count) in time_counts {
                if count >= 3 {
                    let confidence = (count as f64 / 10.0).min(0.5);
                    predictions.push(PredictedMemory {
                        memory_id,
                        content_preview: String::new(),
                        confidence,
                        reasoning: format!("You often access this memory around {}:00", hour),
                        trigger: PredictionTrigger::TemporalPattern {
                            typical_time: format!("{}:00", hour),
                        },
                        predicted_at: now,
                    });
                }
            }
        }

        predictions
    }

    fn deduplicate_predictions(&self, predictions: Vec<PredictedMemory>) -> Vec<PredictedMemory> {
        let mut seen: HashMap<String, PredictedMemory> = HashMap::new();

        for pred in predictions {
            seen.entry(pred.memory_id.clone())
                .and_modify(|existing| {
                    // Keep the one with higher confidence
                    if pred.confidence > existing.confidence {
                        *existing = pred.clone();
                    }
                })
                .or_insert(pred);
        }

        seen.into_values().collect()
    }

    fn store_pending_predictions(&self, predictions: &[PredictedMemory]) {
        if let Ok(mut pending) = self.pending_predictions.write() {
            pending.clear();
            for pred in predictions {
                pending.insert(pred.memory_id.clone(), pred.clone());
            }
        }
    }

    fn strengthen_pattern(&self, memory_id: &str, factor: f64) {
        if let Ok(mut patterns) = self.co_access_patterns.write() {
            for patterns_list in patterns.values_mut() {
                for pattern in patterns_list.iter_mut() {
                    if pattern.predicted_id == memory_id {
                        pattern.weight = (pattern.weight * factor).min(1.0);
                        pattern.frequency += 1;
                        pattern.success_rate = (pattern.success_rate * 0.9) + 0.1;
                        pattern.last_seen = Utc::now();
                    }
                }
            }
        }
    }

    fn weaken_pattern(&self, memory_id: &str, factor: f64) {
        if let Ok(mut patterns) = self.co_access_patterns.write() {
            for patterns_list in patterns.values_mut() {
                for pattern in patterns_list.iter_mut() {
                    if pattern.predicted_id == memory_id {
                        pattern.weight *= factor;
                        pattern.success_rate *= 0.95;
                    }
                }
            }
        }
    }

    fn learn_co_access_patterns(&self, memory_ids: &[String]) {
        if memory_ids.len() < 2 {
            return;
        }

        if let Ok(mut patterns) = self.co_access_patterns.write() {
            // Create patterns between each pair of memories
            for i in 0..memory_ids.len() {
                for j in 0..memory_ids.len() {
                    if i != j {
                        let trigger = &memory_ids[i];
                        let predicted = &memory_ids[j];

                        let patterns_list =
                            patterns.entry(trigger.clone()).or_insert_with(Vec::new);

                        if let Some(existing) = patterns_list
                            .iter_mut()
                            .find(|p| p.predicted_id == *predicted)
                        {
                            existing.frequency += 1;
                            existing.weight = (existing.weight + 0.1).min(1.0);
                            existing.last_seen = Utc::now();
                        } else {
                            patterns_list.push(UsagePattern {
                                trigger_id: trigger.clone(),
                                predicted_id: predicted.clone(),
                                frequency: 1,
                                success_rate: 0.5,
                                last_seen: Utc::now(),
                                weight: 0.5,
                            });
                        }
                    }
                }
            }
        }
    }
}

impl Default for SpeculativeRetriever {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during speculative retrieval
#[derive(Debug, thiserror::Error)]
pub enum SpeculativeError {
    /// Failed to access pattern data
    #[error("Pattern access error: {0}")]
    PatternAccess(String),

    /// Failed to prefetch memories
    #[error("Prefetch error: {0}")]
    Prefetch(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_context() {
        let context = PredictionContext::new()
            .with_file(PathBuf::from("/src/auth.rs"))
            .with_query("JWT token".to_string())
            .with_project(PathBuf::from("/my/project"));

        assert_eq!(context.open_files.len(), 1);
        assert_eq!(context.recent_queries.len(), 1);
        assert!(context.project_path.is_some());
    }

    #[test]
    fn test_record_access() {
        let retriever = SpeculativeRetriever::new();

        retriever.record_access(
            "mem-123",
            Some("/src/auth.rs"),
            Some("JWT token"),
            Some(true),
        );

        // Verify file-memory association was recorded
        let map = retriever.file_memory_map.read().unwrap();
        assert!(map.contains_key("/src/auth.rs"));
    }

    #[test]
    fn test_learn_co_access_patterns() {
        let retriever = SpeculativeRetriever::new();

        retriever.learn_co_access_patterns(&[
            "mem-1".to_string(),
            "mem-2".to_string(),
            "mem-3".to_string(),
        ]);

        let patterns = retriever.co_access_patterns.read().unwrap();
        assert!(patterns.contains_key("mem-1"));
        assert!(patterns.contains_key("mem-2"));
    }
}
