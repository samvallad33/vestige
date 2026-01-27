//! # Predictive Memory Retrieval
//!
//! Implementation of Friston's Free Energy Principle for memory systems.
//! The brain predicts rather than passively stores - this module anticipates
//! what the user needs BEFORE they ask.
//!
//! ## Theoretical Foundation
//!
//! Based on the Active Inference framework (Friston, 2010):
//! - The brain is fundamentally a prediction machine
//! - Memory recall is a predictive process, not passive retrieval
//! - Prediction errors signal novelty and drive enhanced encoding
//! - Free energy minimization guides memory optimization
//!
//! ## How It Works
//!
//! 1. **User Modeling**: Build probabilistic model of user interests, patterns, and context
//! 2. **Predictive Caching**: Pre-fetch likely-needed memories into fast cache
//! 3. **Reinforcement Learning**: Learn from prediction accuracy to improve future predictions
//! 4. **Proactive Surfacing**: Show predictions ("You might also need...")
//! 5. **Novelty Detection**: Prediction errors signal important new information
//!
//! ## Example
//!
//! ```rust,ignore
//! use vestige_core::neuroscience::{PredictiveMemory, UserModel};
//!
//! let mut predictor = PredictiveMemory::new();
//!
//! // Update user model based on activity
//! predictor.record_query("authentication", &["jwt", "oauth"]);
//! predictor.record_interest("security", 0.8);
//!
//! // Get predictions for current context
//! let predictions = predictor.predict_needed_memories(&session_context);
//!
//! // Proactively surface relevant memories
//! for prediction in predictions.iter().filter(|p| p.confidence > 0.7) {
//!     println!("You might need: {} ({}% confidence)",
//!         prediction.memory_id,
//!         (prediction.confidence * 100.0) as u32
//!     );
//! }
//! ```

use chrono::{DateTime, Datelike, Duration, Timelike, Utc, Weekday};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use thiserror::Error;

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

/// Maximum size of the prediction cache
const MAX_CACHE_SIZE: usize = 100;

/// Maximum number of queries to track for pattern analysis
const MAX_QUERY_HISTORY: usize = 500;

/// Maximum predictions to return in a single request
const MAX_PREDICTIONS: usize = 20;

/// Minimum confidence threshold for predictions
const DEFAULT_MIN_CONFIDENCE: f64 = 0.3;

/// Learning rate for interest weight updates
const INTEREST_LEARNING_RATE: f64 = 0.1;

/// Decay factor for interest weights (per day)
const INTEREST_DECAY_RATE: f64 = 0.98;

/// Decay factor for prediction outcomes (for exponential smoothing)
#[allow(dead_code)] // Reserved for future prediction accuracy tracking
const PREDICTION_OUTCOME_DECAY: f64 = 0.9;

/// Time window for session context (minutes)
const SESSION_WINDOW_MINUTES: i64 = 60;

/// Number of recent queries to consider for immediate predictions
const RECENT_QUERY_WINDOW: usize = 10;

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Errors that can occur during predictive retrieval operations
#[derive(Debug, Error)]
pub enum PredictiveMemoryError {
    /// Failed to access prediction cache
    #[error("Cache access error: {0}")]
    CacheAccess(String),

    /// Failed to update user model
    #[error("User model update error: {0}")]
    UserModelUpdate(String),

    /// Failed to generate predictions
    #[error("Prediction generation error: {0}")]
    PredictionGeneration(String),

    /// Lock poisoned during concurrent access
    #[error("Lock poisoned: {0}")]
    LockPoisoned(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for predictive memory operations
pub type Result<T> = std::result::Result<T, PredictiveMemoryError>;

// ============================================================================
// CORE TYPES
// ============================================================================

/// A predicted memory that the user is likely to need
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedMemory {
    /// The memory ID predicted to be needed
    pub memory_id: String,
    /// Content preview for quick reference
    pub content_preview: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Human-readable reasoning for this prediction
    pub reasoning: PredictionReason,
    /// When this prediction was made
    pub predicted_at: DateTime<Utc>,
    /// Tags associated with this memory
    pub tags: Vec<String>,
}

/// Reasons why a memory was predicted to be needed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictionReason {
    /// Based on learned user interests
    InterestBased {
        /// Topic that matched
        topic: String,
        /// Interest weight for this topic
        weight: f64,
    },
    /// Based on recent query patterns
    QueryPattern {
        /// Related query that triggered prediction
        related_query: String,
        /// How often this pattern occurred
        frequency: u32,
    },
    /// Based on temporal patterns (time of day, day of week)
    TemporalPattern {
        /// Description of the temporal pattern
        pattern_description: String,
        /// Historical accuracy of this pattern
        historical_accuracy: f64,
    },
    /// Based on current session context
    SessionContext {
        /// What in the session triggered this
        trigger: String,
        /// Semantic similarity to session content
        similarity: f64,
    },
    /// Based on co-access patterns (memories accessed together)
    CoAccess {
        /// The memory that triggered this prediction
        trigger_memory: String,
        /// How often these are accessed together
        co_occurrence_rate: f64,
    },
    /// Prediction based on semantic similarity
    SemanticSimilarity {
        /// Query or content that was semantically similar
        similar_to: String,
        /// Similarity score
        similarity: f64,
    },
}

impl PredictionReason {
    /// Get a human-readable description of the prediction reason
    pub fn description(&self) -> String {
        match self {
            Self::InterestBased { topic, weight } => {
                format!(
                    "Based on your interest in {} ({}% interest weight)",
                    topic,
                    (weight * 100.0) as u32
                )
            }
            Self::QueryPattern {
                related_query,
                frequency,
            } => {
                format!(
                    "You've searched for similar topics {} times (related: \"{}\")",
                    frequency, related_query
                )
            }
            Self::TemporalPattern {
                pattern_description,
                historical_accuracy,
            } => {
                format!(
                    "{} ({}% historical accuracy)",
                    pattern_description,
                    (historical_accuracy * 100.0) as u32
                )
            }
            Self::SessionContext {
                trigger,
                similarity,
            } => {
                format!(
                    "Relevant to your current session: {} ({}% match)",
                    trigger,
                    (similarity * 100.0) as u32
                )
            }
            Self::CoAccess {
                trigger_memory,
                co_occurrence_rate,
            } => {
                format!(
                    "Often accessed with {} ({}% of the time)",
                    trigger_memory,
                    (co_occurrence_rate * 100.0) as u32
                )
            }
            Self::SemanticSimilarity {
                similar_to,
                similarity,
            } => {
                format!(
                    "Semantically similar to \"{}\" ({}% similarity)",
                    similar_to,
                    (similarity * 100.0) as u32
                )
            }
        }
    }
}

/// Outcome of a prediction (for learning)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionOutcome {
    /// The predicted memory ID
    pub memory_id: String,
    /// The prediction confidence
    pub confidence: f64,
    /// Whether the prediction was used/helpful
    pub was_useful: bool,
    /// Time between prediction and actual use (if used)
    pub time_to_use: Option<Duration>,
    /// When this outcome was recorded
    pub recorded_at: DateTime<Utc>,
}

/// A pattern detected in user queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPattern {
    /// The query content
    pub query: String,
    /// Tags associated with this query
    pub tags: Vec<String>,
    /// When this query was made
    pub timestamp: DateTime<Utc>,
    /// Results that were accessed after this query
    pub accessed_results: Vec<String>,
    /// Whether the user found what they were looking for
    pub was_satisfied: Option<bool>,
}

/// Temporal patterns in user behavior
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemporalPatterns {
    /// Hour of day preferences (0-23) -> topic -> weight
    pub hourly_patterns: HashMap<u32, HashMap<String, f64>>,
    /// Day of week preferences -> topic -> weight
    pub daily_patterns: HashMap<String, HashMap<String, f64>>,
    /// Monthly patterns (for seasonal interests)
    pub monthly_patterns: HashMap<u32, HashMap<String, f64>>,
    /// Activity level by hour (for determining engagement periods)
    pub activity_by_hour: [f64; 24],
}

impl TemporalPatterns {
    /// Create new empty temporal patterns
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the most active hour of the day
    pub fn peak_activity_hour(&self) -> u32 {
        self.activity_by_hour
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(10) // Default to 10 AM
    }

    /// Get topics relevant for the current time
    pub fn topics_for_time(&self, time: DateTime<Utc>) -> Vec<(String, f64)> {
        let hour = time.hour();
        let mut topics = Vec::new();

        if let Some(hour_topics) = self.hourly_patterns.get(&hour) {
            for (topic, weight) in hour_topics {
                topics.push((topic.clone(), *weight));
            }
        }

        let weekday = match time.weekday() {
            Weekday::Mon => "monday",
            Weekday::Tue => "tuesday",
            Weekday::Wed => "wednesday",
            Weekday::Thu => "thursday",
            Weekday::Fri => "friday",
            Weekday::Sat => "saturday",
            Weekday::Sun => "sunday",
        };

        if let Some(day_topics) = self.daily_patterns.get(weekday) {
            for (topic, weight) in day_topics {
                // Combine if already exists
                if let Some(existing) = topics.iter_mut().find(|(t, _)| t == topic) {
                    existing.1 = (existing.1 + weight) / 2.0;
                } else {
                    topics.push((topic.clone(), *weight));
                }
            }
        }

        topics.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        topics
    }

    /// Record activity at a specific time
    pub fn record_activity(&mut self, time: DateTime<Utc>, topic: &str, weight: f64) {
        let hour = time.hour();

        // Update hourly activity
        self.activity_by_hour[hour as usize] = self.activity_by_hour[hour as usize] * 0.9 + 0.1;

        // Update hourly topic patterns
        let hour_topics = self.hourly_patterns.entry(hour).or_default();
        let current = hour_topics.entry(topic.to_string()).or_insert(0.0);
        *current = *current * (1.0 - INTEREST_LEARNING_RATE) + weight * INTEREST_LEARNING_RATE;

        // Update daily patterns
        let weekday = match time.weekday() {
            Weekday::Mon => "monday",
            Weekday::Tue => "tuesday",
            Weekday::Wed => "wednesday",
            Weekday::Thu => "thursday",
            Weekday::Fri => "friday",
            Weekday::Sat => "saturday",
            Weekday::Sun => "sunday",
        };

        let day_topics = self.daily_patterns.entry(weekday.to_string()).or_default();
        let current = day_topics.entry(topic.to_string()).or_insert(0.0);
        *current = *current * (1.0 - INTEREST_LEARNING_RATE) + weight * INTEREST_LEARNING_RATE;

        // Update monthly patterns
        let month = time.month();
        let month_topics = self.monthly_patterns.entry(month).or_default();
        let current = month_topics.entry(topic.to_string()).or_insert(0.0);
        *current = *current * (1.0 - INTEREST_LEARNING_RATE) + weight * INTEREST_LEARNING_RATE;
    }
}

/// Current session context for predictions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionContext {
    /// When the session started
    pub started_at: DateTime<Utc>,
    /// Current working topic/focus
    pub current_focus: Option<String>,
    /// Files currently being worked on
    pub active_files: Vec<String>,
    /// Recent memory accesses in this session
    pub accessed_memories: Vec<String>,
    /// Recent queries in this session
    pub recent_queries: Vec<String>,
    /// Detected intent (if any)
    pub detected_intent: Option<String>,
    /// Project context (if any)
    pub project_context: Option<ProjectContext>,
}

impl SessionContext {
    /// Create a new session context
    pub fn new() -> Self {
        Self {
            started_at: Utc::now(),
            ..Default::default()
        }
    }

    /// Get session duration
    pub fn duration(&self) -> Duration {
        Utc::now() - self.started_at
    }

    /// Check if session is still active (within window)
    pub fn is_active(&self) -> bool {
        self.duration() < Duration::minutes(SESSION_WINDOW_MINUTES)
    }

    /// Add a file to active files
    pub fn add_active_file(&mut self, file: String) {
        if !self.active_files.contains(&file) {
            self.active_files.push(file);
        }
    }

    /// Add an accessed memory
    pub fn add_accessed_memory(&mut self, memory_id: String) {
        if !self.accessed_memories.contains(&memory_id) {
            self.accessed_memories.push(memory_id);
        }
    }

    /// Add a recent query
    pub fn add_query(&mut self, query: String) {
        self.recent_queries.push(query);
        // Keep only recent queries
        if self.recent_queries.len() > RECENT_QUERY_WINDOW {
            self.recent_queries.remove(0);
        }
    }
}

/// Project context for predictions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProjectContext {
    /// Project name
    pub name: String,
    /// Project path
    pub path: String,
    /// Detected frameworks/technologies
    pub technologies: Vec<String>,
    /// Primary programming language
    pub primary_language: Option<String>,
}

// ============================================================================
// USER MODEL
// ============================================================================

/// Model of user interests and behavior for prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserModel {
    /// Topic interest weights (topic -> weight 0.0-1.0)
    pub interests: HashMap<String, f64>,
    /// Recent queries for pattern analysis
    pub recent_queries: VecDeque<QueryPattern>,
    /// Temporal patterns in user behavior
    pub temporal_patterns: TemporalPatterns,
    /// Current session context
    pub session_context: SessionContext,
    /// Co-access patterns (memory_id -> Vec<(memory_id, count)>)
    pub co_access_patterns: HashMap<String, Vec<(String, u32)>>,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
    /// Total number of interactions tracked
    pub total_interactions: u64,
}

impl Default for UserModel {
    fn default() -> Self {
        Self {
            interests: HashMap::new(),
            recent_queries: VecDeque::with_capacity(MAX_QUERY_HISTORY),
            temporal_patterns: TemporalPatterns::new(),
            session_context: SessionContext::new(),
            co_access_patterns: HashMap::new(),
            last_updated: Utc::now(),
            total_interactions: 0,
        }
    }
}

impl UserModel {
    /// Create a new user model
    pub fn new() -> Self {
        Self::default()
    }

    /// Update interest weight for a topic
    pub fn update_interest(&mut self, topic: &str, weight: f64) {
        let normalized_topic = topic.to_lowercase();
        let current = self
            .interests
            .entry(normalized_topic.clone())
            .or_insert(0.0);

        // Exponential moving average for smooth updates
        *current = *current * (1.0 - INTEREST_LEARNING_RATE) + weight * INTEREST_LEARNING_RATE;

        // Clamp to valid range
        *current = current.clamp(0.0, 1.0);

        // Update temporal patterns
        self.temporal_patterns
            .record_activity(Utc::now(), &normalized_topic, weight);
        self.last_updated = Utc::now();
        self.total_interactions += 1;
    }

    /// Record a query for pattern analysis
    pub fn record_query(&mut self, query: &str, tags: &[&str]) {
        let pattern = QueryPattern {
            query: query.to_string(),
            tags: tags.iter().map(|s| s.to_string()).collect(),
            timestamp: Utc::now(),
            accessed_results: Vec::new(),
            was_satisfied: None,
        };

        self.recent_queries.push_back(pattern);

        // Maintain capacity
        while self.recent_queries.len() > MAX_QUERY_HISTORY {
            self.recent_queries.pop_front();
        }

        // Update session context
        self.session_context.add_query(query.to_string());

        // Update interests based on query topics
        for tag in tags {
            self.update_interest(tag, 0.5);
        }

        self.last_updated = Utc::now();
    }

    /// Record that a memory was accessed
    pub fn record_memory_access(&mut self, memory_id: &str, tags: &[String]) {
        // Update session
        self.session_context
            .add_accessed_memory(memory_id.to_string());

        // Update interests based on accessed memory tags
        for tag in tags {
            self.update_interest(tag, 0.7);
        }

        // Update co-access patterns
        // Collect IDs first to avoid borrow issues
        let existing_ids: Vec<String> = self
            .session_context
            .accessed_memories
            .iter()
            .filter(|id| *id != memory_id)
            .cloned()
            .collect();

        for existing_id in existing_ids {
            // Bidirectional co-access
            self.record_co_access(&existing_id, memory_id);
            self.record_co_access(memory_id, &existing_id);
        }

        self.last_updated = Utc::now();
    }

    /// Record co-access between two memories
    fn record_co_access(&mut self, from: &str, to: &str) {
        let patterns = self.co_access_patterns.entry(from.to_string()).or_default();

        if let Some(existing) = patterns.iter_mut().find(|(id, _)| id == to) {
            existing.1 += 1;
        } else {
            patterns.push((to.to_string(), 1));
        }

        // Sort by count and keep top patterns
        patterns.sort_by(|a, b| b.1.cmp(&a.1));
        patterns.truncate(50);
    }

    /// Apply decay to interest weights (call periodically)
    pub fn apply_decay(&mut self) {
        for weight in self.interests.values_mut() {
            *weight *= INTEREST_DECAY_RATE;
        }

        // Remove very low weights
        self.interests.retain(|_, w| *w > 0.01);

        self.last_updated = Utc::now();
    }

    /// Get top interests
    pub fn top_interests(&self, limit: usize) -> Vec<(String, f64)> {
        let mut interests: Vec<_> = self
            .interests
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        interests.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        interests.truncate(limit);
        interests
    }

    /// Get co-access candidates for a memory
    pub fn get_co_access_candidates(&self, memory_id: &str) -> Vec<(String, f64)> {
        self.co_access_patterns
            .get(memory_id)
            .map(|patterns| {
                let total: u32 = patterns.iter().map(|(_, c)| c).sum();
                patterns
                    .iter()
                    .map(|(id, count)| (id.clone(), *count as f64 / total as f64))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if session should be reset
    pub fn should_reset_session(&self) -> bool {
        !self.session_context.is_active()
    }

    /// Reset the session context
    pub fn reset_session(&mut self) {
        self.session_context = SessionContext::new();
    }
}

// ============================================================================
// PREDICTION CACHE (LRU-like)
// ============================================================================

/// Simple LRU-like cache for predictions
#[derive(Debug)]
struct PredictionCache {
    /// Cache entries (key -> (predictions, timestamp))
    entries: HashMap<String, (Vec<PredictedMemory>, DateTime<Utc>)>,
    /// Access order for LRU eviction
    access_order: VecDeque<String>,
    /// Maximum cache size
    max_size: usize,
}

impl PredictionCache {
    fn new(max_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            max_size,
        }
    }

    fn get(&mut self, key: &str) -> Option<&Vec<PredictedMemory>> {
        if self.entries.contains_key(key) {
            // Move to front of access order
            self.access_order.retain(|k| k != key);
            self.access_order.push_front(key.to_string());
            self.entries.get(key).map(|(v, _)| v)
        } else {
            None
        }
    }

    fn insert(&mut self, key: String, predictions: Vec<PredictedMemory>) {
        // Evict if necessary
        while self.entries.len() >= self.max_size {
            if let Some(old_key) = self.access_order.pop_back() {
                self.entries.remove(&old_key);
            }
        }

        self.entries.insert(key.clone(), (predictions, Utc::now()));
        self.access_order.push_front(key);
    }

    fn invalidate(&mut self, key: &str) {
        self.entries.remove(key);
        self.access_order.retain(|k| k != key);
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
    }
}

// ============================================================================
// PREDICTIVE MEMORY ENGINE
// ============================================================================

/// Configuration for the predictive memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveMemoryConfig {
    /// Minimum confidence threshold for predictions
    pub min_confidence: f64,
    /// Maximum predictions to return
    pub max_predictions: usize,
    /// Cache size for predictions
    pub cache_size: usize,
    /// Enable temporal pattern learning
    pub enable_temporal_patterns: bool,
    /// Enable co-access pattern learning
    pub enable_co_access_patterns: bool,
    /// Weight for interest-based predictions
    pub interest_weight: f64,
    /// Weight for temporal predictions
    pub temporal_weight: f64,
    /// Weight for co-access predictions
    pub co_access_weight: f64,
    /// Weight for session context predictions
    pub session_weight: f64,
}

impl Default for PredictiveMemoryConfig {
    fn default() -> Self {
        Self {
            min_confidence: DEFAULT_MIN_CONFIDENCE,
            max_predictions: MAX_PREDICTIONS,
            cache_size: MAX_CACHE_SIZE,
            enable_temporal_patterns: true,
            enable_co_access_patterns: true,
            interest_weight: 0.3,
            temporal_weight: 0.2,
            co_access_weight: 0.3,
            session_weight: 0.2,
        }
    }
}

/// The main predictive memory engine
pub struct PredictiveMemory {
    /// User behavior model
    user_model: Arc<RwLock<UserModel>>,
    /// Prediction cache
    prediction_cache: Arc<RwLock<PredictionCache>>,
    /// History of prediction outcomes for learning
    prediction_history: Arc<RwLock<Vec<PredictionOutcome>>>,
    /// Pending predictions awaiting outcome
    pending_predictions: Arc<RwLock<HashMap<String, PredictedMemory>>>,
    /// Configuration
    config: PredictiveMemoryConfig,
    /// Memory metadata cache (memory_id -> (content_preview, tags))
    memory_metadata: Arc<RwLock<HashMap<String, (String, Vec<String>)>>>,
}

impl PredictiveMemory {
    /// Create a new predictive memory engine with default configuration
    pub fn new() -> Self {
        Self::with_config(PredictiveMemoryConfig::default())
    }

    /// Create a new predictive memory engine with custom configuration
    pub fn with_config(config: PredictiveMemoryConfig) -> Self {
        Self {
            user_model: Arc::new(RwLock::new(UserModel::new())),
            prediction_cache: Arc::new(RwLock::new(PredictionCache::new(config.cache_size))),
            prediction_history: Arc::new(RwLock::new(Vec::new())),
            pending_predictions: Arc::new(RwLock::new(HashMap::new())),
            config,
            memory_metadata: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &PredictiveMemoryConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: PredictiveMemoryConfig) {
        self.config = config;
    }

    /// Record a user query
    pub fn record_query(&self, query: &str, tags: &[&str]) -> Result<()> {
        let mut model = self
            .user_model
            .write()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        model.record_query(query, tags);

        // Invalidate cache for changed interests
        let mut cache = self
            .prediction_cache
            .write()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        for tag in tags {
            cache.invalidate(tag);
        }

        Ok(())
    }

    /// Record an interest with weight
    pub fn record_interest(&self, topic: &str, weight: f64) -> Result<()> {
        let mut model = self
            .user_model
            .write()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        model.update_interest(topic, weight);

        // Invalidate related cache entries
        let mut cache = self
            .prediction_cache
            .write()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;
        cache.invalidate(topic);

        Ok(())
    }

    /// Record that a memory was accessed
    pub fn record_memory_access(
        &self,
        memory_id: &str,
        content_preview: &str,
        tags: &[String],
    ) -> Result<()> {
        // Update user model
        let mut model = self
            .user_model
            .write()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        model.record_memory_access(memory_id, tags);

        // Store metadata for future predictions
        let mut metadata = self
            .memory_metadata
            .write()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        metadata.insert(
            memory_id.to_string(),
            (content_preview.to_string(), tags.to_vec()),
        );

        // Check if this was a predicted memory
        self.record_prediction_outcome(memory_id, true)?;

        Ok(())
    }

    /// Update session context
    pub fn update_session_context(
        &self,
        update_fn: impl FnOnce(&mut SessionContext),
    ) -> Result<()> {
        let mut model = self
            .user_model
            .write()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        // Reset session if stale
        if model.should_reset_session() {
            model.reset_session();
        }

        update_fn(&mut model.session_context);

        Ok(())
    }

    /// Predict memories that will be needed based on current context
    pub fn predict_needed_memories(
        &self,
        context: &SessionContext,
    ) -> Result<Vec<PredictedMemory>> {
        let model = self
            .user_model
            .read()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        let now = Utc::now();
        let mut predictions: Vec<PredictedMemory> = Vec::new();

        // 1. Interest-based predictions
        if self.config.interest_weight > 0.0 {
            predictions.extend(self.predict_from_interests(&model, now));
        }

        // 2. Temporal pattern predictions
        if self.config.enable_temporal_patterns && self.config.temporal_weight > 0.0 {
            predictions.extend(self.predict_from_temporal(&model, now));
        }

        // 3. Co-access pattern predictions
        if self.config.enable_co_access_patterns && self.config.co_access_weight > 0.0 {
            predictions.extend(self.predict_from_co_access(&model, context, now));
        }

        // 4. Session context predictions
        if self.config.session_weight > 0.0 {
            predictions.extend(self.predict_from_session(context, now));
        }

        // Deduplicate and combine scores
        predictions = self.merge_predictions(predictions);

        // Filter by minimum confidence
        predictions.retain(|p| p.confidence >= self.config.min_confidence);

        // Sort by confidence
        predictions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate to max
        predictions.truncate(self.config.max_predictions);

        // Store as pending for outcome tracking
        self.store_pending_predictions(&predictions)?;

        Ok(predictions)
    }

    /// Get proactive suggestions ("You might also need...")
    pub fn get_proactive_suggestions(&self, min_confidence: f64) -> Result<Vec<PredictedMemory>> {
        let model = self
            .user_model
            .read()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        let predictions = self.predict_needed_memories(&model.session_context)?;

        Ok(predictions
            .into_iter()
            .filter(|p| p.confidence >= min_confidence)
            .collect())
    }

    /// Pre-fetch likely-needed memories into cache
    pub async fn prefetch(&self, context: &SessionContext) -> Result<usize> {
        let predictions = self.predict_needed_memories(context)?;
        let count = predictions.len();

        // Generate cache key from context
        let cache_key = self.generate_cache_key(context);

        // Store in cache
        let mut cache = self
            .prediction_cache
            .write()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        cache.insert(cache_key, predictions);

        Ok(count)
    }

    /// Get cached predictions for a context
    pub fn get_cached_predictions(
        &self,
        context: &SessionContext,
    ) -> Result<Option<Vec<PredictedMemory>>> {
        let cache_key = self.generate_cache_key(context);

        let mut cache = self
            .prediction_cache
            .write()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        Ok(cache.get(&cache_key).cloned())
    }

    /// Record the outcome of a prediction (for learning)
    pub fn record_prediction_outcome(&self, memory_id: &str, was_useful: bool) -> Result<()> {
        let mut pending = self
            .pending_predictions
            .write()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        if let Some(prediction) = pending.remove(memory_id) {
            let outcome = PredictionOutcome {
                memory_id: memory_id.to_string(),
                confidence: prediction.confidence,
                was_useful,
                time_to_use: Some(Utc::now() - prediction.predicted_at),
                recorded_at: Utc::now(),
            };

            let mut history = self
                .prediction_history
                .write()
                .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

            history.push(outcome);

            // Keep history manageable
            if history.len() > 10_000 {
                history.drain(0..5000);
            }
        }

        Ok(())
    }

    /// Calculate prediction accuracy based on history
    pub fn prediction_accuracy(&self) -> Result<f64> {
        let history = self
            .prediction_history
            .read()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        if history.is_empty() {
            return Ok(0.0);
        }

        let useful_count = history.iter().filter(|o| o.was_useful).count();
        Ok(useful_count as f64 / history.len() as f64)
    }

    /// Apply decay to learned patterns (call periodically, e.g., daily)
    pub fn apply_decay(&self) -> Result<()> {
        let mut model = self
            .user_model
            .write()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        model.apply_decay();

        // Clear old cache entries
        let mut cache = self
            .prediction_cache
            .write()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        cache.clear();

        Ok(())
    }

    /// Get the current user model (read-only)
    pub fn get_user_model(&self) -> Result<UserModel> {
        let model = self
            .user_model
            .read()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        Ok(model.clone())
    }

    /// Get top interests from the user model
    pub fn get_top_interests(&self, limit: usize) -> Result<Vec<(String, f64)>> {
        let model = self
            .user_model
            .read()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        Ok(model.top_interests(limit))
    }

    /// Signal novelty (prediction error) for enhanced encoding
    pub fn signal_novelty(&self, _memory_id: &str, tags: &[String]) -> Result<f64> {
        // Calculate novelty based on how unexpected this is
        let model = self
            .user_model
            .read()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        // Novelty is higher when tags don't match current interests
        let mut interest_match = 0.0;
        for tag in tags {
            if let Some(weight) = model.interests.get(&tag.to_lowercase()) {
                interest_match += weight;
            }
        }

        // Normalize and invert (high match = low novelty)
        let avg_match = if tags.is_empty() {
            0.0
        } else {
            interest_match / tags.len() as f64
        };
        let novelty = 1.0 - avg_match;

        // Novelty signals should boost encoding of this memory
        // The caller can use this to adjust retention strength

        Ok(novelty)
    }

    // ========================================================================
    // Private prediction methods
    // ========================================================================

    fn predict_from_interests(
        &self,
        model: &UserModel,
        now: DateTime<Utc>,
    ) -> Vec<PredictedMemory> {
        let metadata = self.memory_metadata.read().ok();
        let mut predictions = Vec::new();

        if let Some(meta) = metadata {
            let top_interests = model.top_interests(10);

            for (topic, interest_weight) in top_interests {
                // Find memories with matching tags
                for (memory_id, (content_preview, tags)) in meta.iter() {
                    if tags.iter().any(|t| t.to_lowercase() == topic) {
                        let confidence = interest_weight * self.config.interest_weight;

                        predictions.push(PredictedMemory {
                            memory_id: memory_id.clone(),
                            content_preview: content_preview.clone(),
                            confidence,
                            reasoning: PredictionReason::InterestBased {
                                topic: topic.clone(),
                                weight: interest_weight,
                            },
                            predicted_at: now,
                            tags: tags.clone(),
                        });
                    }
                }
            }
        }

        predictions
    }

    fn predict_from_temporal(&self, model: &UserModel, now: DateTime<Utc>) -> Vec<PredictedMemory> {
        let metadata = self.memory_metadata.read().ok();
        let mut predictions = Vec::new();

        let temporal_topics = model.temporal_patterns.topics_for_time(now);

        if let Some(meta) = metadata {
            for (topic, temporal_weight) in temporal_topics {
                for (memory_id, (content_preview, tags)) in meta.iter() {
                    if tags
                        .iter()
                        .any(|t| t.to_lowercase() == topic.to_lowercase())
                    {
                        let confidence = temporal_weight * self.config.temporal_weight;

                        predictions.push(PredictedMemory {
                            memory_id: memory_id.clone(),
                            content_preview: content_preview.clone(),
                            confidence,
                            reasoning: PredictionReason::TemporalPattern {
                                pattern_description: format!(
                                    "You often work on {} at this time",
                                    topic
                                ),
                                historical_accuracy: temporal_weight,
                            },
                            predicted_at: now,
                            tags: tags.clone(),
                        });
                    }
                }
            }
        }

        predictions
    }

    fn predict_from_co_access(
        &self,
        model: &UserModel,
        context: &SessionContext,
        now: DateTime<Utc>,
    ) -> Vec<PredictedMemory> {
        let metadata = self.memory_metadata.read().ok();
        let mut predictions = Vec::new();

        // For each recently accessed memory, find co-access candidates
        for accessed_id in &context.accessed_memories {
            let candidates = model.get_co_access_candidates(accessed_id);

            for (candidate_id, co_occurrence_rate) in candidates {
                // Skip if already accessed
                if context.accessed_memories.contains(&candidate_id) {
                    continue;
                }

                let confidence = co_occurrence_rate * self.config.co_access_weight;

                let (content_preview, tags) = metadata
                    .as_ref()
                    .and_then(|m| m.get(&candidate_id))
                    .cloned()
                    .unwrap_or_default();

                predictions.push(PredictedMemory {
                    memory_id: candidate_id.clone(),
                    content_preview,
                    confidence,
                    reasoning: PredictionReason::CoAccess {
                        trigger_memory: accessed_id.clone(),
                        co_occurrence_rate,
                    },
                    predicted_at: now,
                    tags,
                });
            }
        }

        predictions
    }

    fn predict_from_session(
        &self,
        context: &SessionContext,
        now: DateTime<Utc>,
    ) -> Vec<PredictedMemory> {
        let metadata = self.memory_metadata.read().ok();
        let mut predictions = Vec::new();

        // Use session focus and recent queries to find relevant memories
        if let Some(meta) = metadata {
            // Match against current focus
            if let Some(focus) = &context.current_focus {
                for (memory_id, (content_preview, tags)) in meta.iter() {
                    if tags
                        .iter()
                        .any(|t| t.to_lowercase().contains(&focus.to_lowercase()))
                        || content_preview
                            .to_lowercase()
                            .contains(&focus.to_lowercase())
                    {
                        let confidence = 0.6 * self.config.session_weight;

                        predictions.push(PredictedMemory {
                            memory_id: memory_id.clone(),
                            content_preview: content_preview.clone(),
                            confidence,
                            reasoning: PredictionReason::SessionContext {
                                trigger: format!("Current focus: {}", focus),
                                similarity: 0.6,
                            },
                            predicted_at: now,
                            tags: tags.clone(),
                        });
                    }
                }
            }

            // Match against recent queries
            for query in &context.recent_queries {
                for (memory_id, (content_preview, tags)) in meta.iter() {
                    let query_lower = query.to_lowercase();
                    if tags.iter().any(|t| query_lower.contains(&t.to_lowercase()))
                        || content_preview.to_lowercase().contains(&query_lower)
                    {
                        let confidence = 0.5 * self.config.session_weight;

                        predictions.push(PredictedMemory {
                            memory_id: memory_id.clone(),
                            content_preview: content_preview.clone(),
                            confidence,
                            reasoning: PredictionReason::SessionContext {
                                trigger: format!("Recent query: {}", query),
                                similarity: 0.5,
                            },
                            predicted_at: now,
                            tags: tags.clone(),
                        });
                    }
                }
            }
        }

        predictions
    }

    fn merge_predictions(&self, predictions: Vec<PredictedMemory>) -> Vec<PredictedMemory> {
        let mut merged: HashMap<String, PredictedMemory> = HashMap::new();

        for pred in predictions {
            merged
                .entry(pred.memory_id.clone())
                .and_modify(|existing| {
                    // Combine confidence scores (taking max, with a small boost for multiple signals)
                    existing.confidence = (existing.confidence.max(pred.confidence) * 1.1).min(1.0);
                })
                .or_insert(pred);
        }

        merged.into_values().collect()
    }

    fn store_pending_predictions(&self, predictions: &[PredictedMemory]) -> Result<()> {
        let mut pending = self
            .pending_predictions
            .write()
            .map_err(|e| PredictiveMemoryError::LockPoisoned(e.to_string()))?;

        pending.clear();
        for pred in predictions {
            pending.insert(pred.memory_id.clone(), pred.clone());
        }

        Ok(())
    }

    fn generate_cache_key(&self, context: &SessionContext) -> String {
        let mut key = String::new();

        if let Some(focus) = &context.current_focus {
            key.push_str(focus);
        }

        for query in context.recent_queries.iter().take(3) {
            key.push_str(query);
        }

        // Include time bucket (hourly)
        key.push_str(&format!("_h{}", Utc::now().hour()));

        key
    }
}

impl Default for PredictiveMemory {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_model_creation() {
        let model = UserModel::new();
        assert!(model.interests.is_empty());
        assert!(model.recent_queries.is_empty());
    }

    #[test]
    fn test_interest_update() {
        let mut model = UserModel::new();
        model.update_interest("rust", 0.8);

        assert!(model.interests.contains_key("rust"));
        assert!(model.interests.get("rust").unwrap() > &0.0);
    }

    #[test]
    fn test_query_recording() {
        let mut model = UserModel::new();
        model.record_query("how to use async", &["rust", "async"]);

        assert_eq!(model.recent_queries.len(), 1);
        assert!(model.interests.contains_key("rust"));
        assert!(model.interests.contains_key("async"));
    }

    #[test]
    fn test_temporal_patterns() {
        let mut patterns = TemporalPatterns::new();
        let now = Utc::now();

        patterns.record_activity(now, "coding", 0.9);

        let topics = patterns.topics_for_time(now);
        assert!(!topics.is_empty());
    }

    #[test]
    fn test_session_context() {
        let mut context = SessionContext::new();
        context.add_query("test query".to_string());
        context.add_active_file("/src/main.rs".to_string());

        assert_eq!(context.recent_queries.len(), 1);
        assert_eq!(context.active_files.len(), 1);
        assert!(context.is_active());
    }

    #[test]
    fn test_predictive_memory_creation() {
        let predictor = PredictiveMemory::new();
        assert_eq!(predictor.config.min_confidence, DEFAULT_MIN_CONFIDENCE);
    }

    #[test]
    fn test_record_query() {
        let predictor = PredictiveMemory::new();
        let result = predictor.record_query("authentication", &["security", "jwt"]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_record_interest() {
        let predictor = PredictiveMemory::new();
        let result = predictor.record_interest("machine learning", 0.9);
        assert!(result.is_ok());

        let interests = predictor.get_top_interests(5).unwrap();
        assert!(!interests.is_empty());
    }

    #[test]
    fn test_prediction_reason_description() {
        let reason = PredictionReason::InterestBased {
            topic: "rust".to_string(),
            weight: 0.8,
        };

        let desc = reason.description();
        assert!(desc.contains("rust"));
        assert!(desc.contains("80%"));
    }

    #[test]
    fn test_predict_needed_memories() {
        let predictor = PredictiveMemory::new();

        // Record some activity
        predictor.record_interest("rust", 0.9).unwrap();
        predictor
            .record_query("async programming", &["rust", "async"])
            .unwrap();

        let context = SessionContext::new();
        let predictions = predictor.predict_needed_memories(&context);

        assert!(predictions.is_ok());
    }

    #[test]
    fn test_novelty_signal() {
        let predictor = PredictiveMemory::new();

        // Record interest in Rust multiple times to build up the weight
        // (INTEREST_LEARNING_RATE is 0.1, so we need multiple calls to reach > 0.5)
        for _ in 0..20 {
            predictor.record_interest("rust", 1.0).unwrap();
        }

        // Novel topic should have high novelty
        let novelty = predictor
            .signal_novelty("mem-1", &["python".to_string()])
            .unwrap();
        assert!(novelty > 0.5, "Python should be novel (got {})", novelty);

        // Familiar topic should have lower novelty
        let novelty = predictor
            .signal_novelty("mem-2", &["rust".to_string()])
            .unwrap();
        assert!(novelty < 0.5, "Rust should be familiar (got {})", novelty);
    }

    #[test]
    fn test_prediction_accuracy() {
        let predictor = PredictiveMemory::new();

        // Initially should be 0.0 (no history)
        let accuracy = predictor.prediction_accuracy().unwrap();
        assert_eq!(accuracy, 0.0);
    }
}

// ============================================================================
// BACKWARD COMPATIBILITY ALIASES
// ============================================================================

/// Alias for backward compatibility with existing code
pub type PredictiveRetriever = PredictiveMemory;

/// Alias for backward compatibility with existing code
pub type Prediction = PredictedMemory;

/// Prediction confidence level for backward compatibility
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PredictionConfidence {
    /// Very low confidence (< 0.2)
    VeryLow,
    /// Low confidence (0.2 - 0.4)
    Low,
    /// Medium confidence (0.4 - 0.6)
    Medium,
    /// High confidence (0.6 - 0.8)
    High,
    /// Very high confidence (> 0.8)
    VeryHigh,
}

impl PredictionConfidence {
    /// Create from a confidence score
    pub fn from_score(score: f64) -> Self {
        if score < 0.2 {
            Self::VeryLow
        } else if score < 0.4 {
            Self::Low
        } else if score < 0.6 {
            Self::Medium
        } else if score < 0.8 {
            Self::High
        } else {
            Self::VeryHigh
        }
    }

    /// Get the numeric range for this confidence level
    pub fn range(&self) -> (f64, f64) {
        match self {
            Self::VeryLow => (0.0, 0.2),
            Self::Low => (0.2, 0.4),
            Self::Medium => (0.4, 0.6),
            Self::High => (0.6, 0.8),
            Self::VeryHigh => (0.8, 1.0),
        }
    }
}

/// Sequence-based predictor for temporal access patterns
#[derive(Debug, Default)]
pub struct SequencePredictor {
    /// Recent access sequences
    sequences: Vec<Vec<String>>,
    /// Maximum sequence length
    max_length: usize,
}

impl SequencePredictor {
    /// Create a new sequence predictor
    pub fn new(max_length: usize) -> Self {
        Self {
            sequences: Vec::new(),
            max_length,
        }
    }

    /// Add an access to the sequence
    pub fn add_access(&mut self, memory_id: String) {
        if self.sequences.is_empty() {
            self.sequences.push(Vec::new());
        }

        if let Some(last) = self.sequences.last_mut() {
            last.push(memory_id);
            if last.len() > self.max_length {
                last.remove(0);
            }
        }
    }

    /// Predict next likely accesses
    pub fn predict_next(&self, _current_id: &str) -> Vec<(String, f64)> {
        // Simple implementation - return empty for now
        // Full implementation would use sequence matching
        Vec::new()
    }
}

/// Temporal predictor for time-based patterns
#[derive(Debug, Default)]
pub struct TemporalPredictor {
    /// Patterns by hour of day
    hourly_patterns: TemporalPatterns,
}

impl TemporalPredictor {
    /// Create a new temporal predictor
    pub fn new() -> Self {
        Self {
            hourly_patterns: TemporalPatterns::new(),
        }
    }

    /// Record an access at the current time
    pub fn record_access(&mut self, _memory_id: &str, topics: &[String]) {
        let now = Utc::now();
        for topic in topics {
            self.hourly_patterns.record_activity(now, topic, 0.5);
        }
    }

    /// Predict memories likely to be accessed now
    pub fn predict_for_time(&self, time: DateTime<Utc>) -> Vec<(String, f64)> {
        self.hourly_patterns.topics_for_time(time)
    }
}

/// Contextual predictor for context-based patterns
#[derive(Debug, Default)]
pub struct ContextualPredictor {
    /// Context-memory associations
    context_memories: HashMap<String, Vec<String>>,
}

impl ContextualPredictor {
    /// Create a new contextual predictor
    pub fn new() -> Self {
        Self::default()
    }

    /// Associate a memory with a context
    pub fn add_association(&mut self, context: &str, memory_id: String) {
        self.context_memories
            .entry(context.to_string())
            .or_default()
            .push(memory_id);
    }

    /// Predict memories for a given context
    pub fn predict_for_context(&self, context: &str) -> Vec<String> {
        self.context_memories
            .get(context)
            .cloned()
            .unwrap_or_default()
    }
}

/// Configuration for predictive retrieval (backward compatibility alias)
pub type PredictiveConfig = PredictiveMemoryConfig;
