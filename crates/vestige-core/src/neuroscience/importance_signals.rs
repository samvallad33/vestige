//! # Multi-channel Importance Signaling System
//!
//! Inspired by how neuromodulators in the brain signal different types of importance:
//!
//! - **Dopamine (Novelty & Reward)**: Signals prediction errors and positive outcomes
//! - **Norepinephrine (Arousal)**: Signals emotional intensity and urgency
//! - **Acetylcholine (Attention)**: Signals focus states and active learning
//! - **Serotonin**: Modulates overall system responsiveness
//!
//! ## The Four Importance Channels
//!
//! 1. **NoveltySignal**: Detects prediction errors - when something doesn't match expectations
//! 2. **ArousalSignal**: Detects emotional intensity through sentiment and keywords
//! 3. **RewardSignal**: Tracks which memories lead to positive outcomes
//! 4. **AttentionSignal**: Detects when the user is actively focused/learning
//!
//! ## Why This Matters
//!
//! Different types of content deserve different treatment:
//! - Novel information needs stronger initial encoding
//! - Emotional content naturally sticks better (flashbulb memories)
//! - Rewarding patterns should be reinforced
//! - Focused learning sessions create stronger memories
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! let signals = ImportanceSignals::new();
//!
//! // Analyze content for importance
//! let content = "CRITICAL: Production database migration failed with data loss!";
//! let context = Context::current();
//!
//! let score = signals.compute_importance(content, &context);
//!
//! // Get transparent breakdown
//! println!("Novelty:   {:.2} - {}", score.novelty, score.explain_novelty());
//! println!("Arousal:   {:.2} - {}", score.arousal, score.explain_arousal());
//! println!("Reward:    {:.2} - {}", score.reward, score.explain_reward());
//! println!("Attention: {:.2} - {}", score.attention, score.explain_attention());
//! println!("Composite: {:.2}", score.composite);
//!
//! // Use score for encoding decisions
//! if score.encoding_boost > 1.0 {
//!     println!("Boosting encoding strength by {:.0}%", (score.encoding_boost - 1.0) * 100.0);
//! }
//! ```
//!
//! ## Biological Inspiration
//!
//! In the brain, neuromodulator systems work together:
//!
//! | System | Neuromodulator | Memory Effect |
//! |--------|---------------|---------------|
//! | Novelty | Dopamine (VTA/SNc) | Enhances hippocampal plasticity |
//! | Arousal | Norepinephrine (LC) | Strengthens amygdala-mediated encoding |
//! | Reward | Dopamine (NAcc) | Reinforces successful patterns |
//! | Attention | Acetylcholine (BF) | Gates learning in cortical circuits |
//!
//! This system translates these biological mechanisms into computational signals
//! that determine memory encoding strength, consolidation priority, and retrieval ranking.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

/// Default weight for novelty signal in composite score
const DEFAULT_NOVELTY_WEIGHT: f64 = 0.25;

/// Default weight for arousal signal in composite score
const DEFAULT_AROUSAL_WEIGHT: f64 = 0.30;

/// Default weight for reward signal in composite score
const DEFAULT_REWARD_WEIGHT: f64 = 0.25;

/// Default weight for attention signal in composite score
const DEFAULT_ATTENTION_WEIGHT: f64 = 0.20;

/// Minimum importance score (never drops to zero)
const MIN_IMPORTANCE: f64 = 0.05;

/// Maximum importance score
const MAX_IMPORTANCE: f64 = 1.0;

/// Default novelty threshold for prediction model
const DEFAULT_NOVELTY_THRESHOLD: f64 = 0.3;

/// Maximum patterns to track in prediction model
const MAX_PREDICTION_PATTERNS: usize = 10_000;

/// Decay rate for pattern frequencies
const PATTERN_DECAY_RATE: f64 = 0.99;

/// Maximum outcome history entries
const MAX_OUTCOME_HISTORY: usize = 5_000;

/// Session inactivity timeout for learning mode detection (minutes)
const LEARNING_MODE_TIMEOUT_MINUTES: i64 = 30;

// ============================================================================
// CONTEXT
// ============================================================================

/// Context for importance computation
///
/// Provides environmental information that affects importance scoring.
#[derive(Debug, Clone, Default)]
pub struct Context {
    /// Current session ID
    pub session_id: Option<String>,
    /// Current project or domain
    pub project: Option<String>,
    /// Recent queries made
    pub recent_queries: Vec<String>,
    /// Current time (for temporal patterns)
    pub timestamp: Option<DateTime<Utc>>,
    /// Whether user is in an active learning session
    pub learning_session_active: bool,
    /// Current emotional context (e.g., "stressed", "focused", "casual")
    pub emotional_context: Option<String>,
    /// Tags relevant to current context
    pub context_tags: Vec<String>,
    /// Recent memory IDs accessed
    pub recent_memory_ids: Vec<String>,
}

impl Context {
    /// Create a new context with current timestamp
    pub fn current() -> Self {
        Self {
            timestamp: Some(Utc::now()),
            ..Default::default()
        }
    }

    /// Set the session ID
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set the project context
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Add a recent query
    pub fn with_query(mut self, query: impl Into<String>) -> Self {
        self.recent_queries.push(query.into());
        self
    }

    /// Set learning session status
    pub fn with_learning_session(mut self, active: bool) -> Self {
        self.learning_session_active = active;
        self
    }

    /// Set emotional context
    pub fn with_emotional_context(mut self, context: impl Into<String>) -> Self {
        self.emotional_context = Some(context.into());
        self
    }

    /// Add context tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.context_tags = tags;
        self
    }
}

// ============================================================================
// NOVELTY SIGNAL (Dopamine-like: Prediction Error)
// ============================================================================

/// Novelty signal inspired by dopamine's role in signaling prediction errors.
///
/// In the brain, dopamine neurons fire when outcomes differ from predictions.
/// This "prediction error" signal drives learning and memory formation.
///
/// ## How It Works
///
/// 1. Maintains a simple n-gram based prediction model of content patterns
/// 2. Computes how much new content deviates from learned patterns
/// 3. High deviation = high novelty = stronger encoding signal
///
/// ## Adaptation
///
/// The model continuously learns from content it sees, so the same content
/// becomes less novel over time - just like habituation in biological systems.
#[derive(Debug)]
pub struct NoveltySignal {
    /// The prediction model that learns content patterns
    prediction_model: PredictionModel,
    /// Threshold below which content is considered "expected"
    novelty_threshold: f64,
}

impl Default for NoveltySignal {
    fn default() -> Self {
        Self::new()
    }
}

impl NoveltySignal {
    /// Create a new novelty signal detector
    pub fn new() -> Self {
        Self {
            prediction_model: PredictionModel::new(),
            novelty_threshold: DEFAULT_NOVELTY_THRESHOLD,
        }
    }

    /// Create with custom novelty threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.novelty_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Compute novelty score for content
    ///
    /// Returns a score from 0.0 (completely expected) to 1.0 (completely novel).
    pub fn compute(&self, content: &str, context: &Context) -> f64 {
        let prediction_error = self.prediction_model.compute_prediction_error(content);

        // Context-based adjustment
        let context_modifier = self.compute_context_modifier(content, context);

        // Combine prediction error with context
        let raw_novelty = (prediction_error * 0.7) + (context_modifier * 0.3);

        // Apply threshold - content below threshold gets reduced novelty
        if raw_novelty < self.novelty_threshold {
            raw_novelty * 0.5
        } else {
            raw_novelty
        }
        .clamp(MIN_IMPORTANCE, MAX_IMPORTANCE)
    }

    /// Update the prediction model with new content (learning)
    pub fn update_model(&mut self, content: &str) {
        self.prediction_model.learn(content);
    }

    /// Check if content is considered novel (above threshold)
    pub fn is_novel(&self, content: &str, context: &Context) -> bool {
        self.compute(content, context) > self.novelty_threshold
    }

    /// Get explanation for novelty score
    pub fn explain(&self, content: &str, context: &Context) -> NoveltyExplanation {
        let score = self.compute(content, context);
        let novel_patterns = self.prediction_model.find_novel_patterns(content);
        let familiar_patterns = self.prediction_model.find_familiar_patterns(content);

        NoveltyExplanation {
            score,
            novel_patterns,
            familiar_patterns,
            prediction_confidence: self.prediction_model.pattern_coverage(content),
        }
    }

    fn compute_context_modifier(&self, content: &str, context: &Context) -> f64 {
        let mut modifier: f64 = 0.5; // Neutral starting point

        // New topics are more novel
        if !context.recent_queries.is_empty() {
            let content_lower = content.to_lowercase();
            let query_overlap = context
                .recent_queries
                .iter()
                .filter(|q| content_lower.contains(&q.to_lowercase()))
                .count();

            if query_overlap == 0 {
                modifier += 0.3; // Content unrelated to recent queries = more novel
            }
        }

        // Content in new project context is more novel
        if context.project.is_some() {
            modifier += 0.1;
        }

        modifier.clamp(0.0, 1.0)
    }
}

/// Explanation of novelty score for transparency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoveltyExplanation {
    /// The computed novelty score
    pub score: f64,
    /// Patterns in content that were novel (not seen before)
    pub novel_patterns: Vec<String>,
    /// Patterns in content that were familiar (seen before)
    pub familiar_patterns: Vec<String>,
    /// How much of the content the model can predict
    pub prediction_confidence: f64,
}

impl NoveltyExplanation {
    /// Generate human-readable explanation
    pub fn explain(&self) -> String {
        if self.score > 0.7 {
            format!(
                "Highly novel content ({:.0}% new). Novel patterns: {}",
                self.score * 100.0,
                self.novel_patterns
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else if self.score > 0.4 {
            format!(
                "Moderately novel ({:.0}% new). Mix of familiar and new patterns.",
                self.score * 100.0
            )
        } else {
            format!(
                "Familiar content ({:.0}% expected). Matches known patterns.",
                (1.0 - self.score) * 100.0
            )
        }
    }
}

/// Simple n-gram based prediction model
#[derive(Debug)]
struct PredictionModel {
    /// N-gram frequencies (pattern -> count)
    patterns: Arc<RwLock<HashMap<String, u32>>>,
    /// Total patterns seen
    total_count: Arc<RwLock<u64>>,
    /// N-gram size
    ngram_size: usize,
}

impl PredictionModel {
    fn new() -> Self {
        Self {
            patterns: Arc::new(RwLock::new(HashMap::new())),
            total_count: Arc::new(RwLock::new(0)),
            ngram_size: 3,
        }
    }

    fn learn(&self, content: &str) {
        let ngrams = self.extract_ngrams(content);

        if let Ok(mut patterns) = self.patterns.write()
            && let Ok(mut total) = self.total_count.write()
        {
            for ngram in ngrams {
                *patterns.entry(ngram).or_insert(0) += 1;
                *total += 1;
            }

            // Prune if too large, then reconcile total_count to the surviving
            // sum. The old code left total_count inflated after pruning/decay,
            // which drove computed novelty (count/total) toward zero over time.
            if patterns.len() > MAX_PREDICTION_PATTERNS {
                self.apply_decay(&mut patterns);
                *total = patterns.values().map(|c| *c as u64).sum();
            }
        }
    }

    fn compute_prediction_error(&self, content: &str) -> f64 {
        let ngrams = self.extract_ngrams(content);
        if ngrams.is_empty() {
            return 0.5; // Unknown content = moderate novelty
        }

        let patterns = match self.patterns.read() {
            Ok(p) => p,
            Err(_) => return 0.5,
        };

        let total = match self.total_count.read() {
            Ok(t) => *t,
            Err(_) => return 0.5,
        };

        if total == 0 || patterns.is_empty() {
            return 1.0; // No training data = everything is novel
        }

        // Calculate what fraction of ngrams are "unexpected"
        let mut unexpected_count = 0;
        let mut total_surprise = 0.0;

        for ngram in &ngrams {
            match patterns.get(ngram) {
                Some(&count) => {
                    // Lower frequency = more surprising
                    let probability = count as f64 / total as f64;
                    total_surprise += 1.0 - probability.sqrt();
                }
                None => {
                    // Never seen = maximum surprise
                    unexpected_count += 1;
                    total_surprise += 1.0;
                }
            }
        }

        // Combine unexpected ratio with average surprise
        let unexpected_ratio = unexpected_count as f64 / ngrams.len() as f64;
        let avg_surprise = total_surprise / ngrams.len() as f64;

        (unexpected_ratio * 0.6 + avg_surprise * 0.4).clamp(0.0, 1.0)
    }

    fn pattern_coverage(&self, content: &str) -> f64 {
        let ngrams = self.extract_ngrams(content);
        if ngrams.is_empty() {
            return 0.0;
        }

        let patterns = match self.patterns.read() {
            Ok(p) => p,
            Err(_) => return 0.0,
        };

        let known_count = ngrams
            .iter()
            .filter(|ng| patterns.contains_key(*ng))
            .count();

        known_count as f64 / ngrams.len() as f64
    }

    fn find_novel_patterns(&self, content: &str) -> Vec<String> {
        let ngrams = self.extract_ngrams(content);
        let patterns = match self.patterns.read() {
            Ok(p) => p,
            Err(_) => return vec![],
        };

        ngrams
            .into_iter()
            .filter(|ng| !patterns.contains_key(ng))
            .take(5)
            .collect()
    }

    fn find_familiar_patterns(&self, content: &str) -> Vec<String> {
        let ngrams = self.extract_ngrams(content);
        let patterns = match self.patterns.read() {
            Ok(p) => p,
            Err(_) => return vec![],
        };

        let mut familiar: Vec<_> = ngrams
            .into_iter()
            .filter_map(|ng| patterns.get(&ng).map(|&count| (ng, count)))
            .collect();

        familiar.sort_by_key(|b| std::cmp::Reverse(b.1));
        familiar.into_iter().take(5).map(|(ng, _)| ng).collect()
    }

    fn extract_ngrams(&self, content: &str) -> Vec<String> {
        let lowercased = content.to_lowercase();
        let words: Vec<&str> = lowercased
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();

        if words.len() < self.ngram_size {
            return words.iter().map(|s| s.to_string()).collect();
        }

        words
            .windows(self.ngram_size)
            .map(|w| w.join(" "))
            .collect()
    }

    fn apply_decay(&self, patterns: &mut HashMap<String, u32>) {
        // Remove lowest frequency patterns
        let mut entries: Vec<_> = patterns.iter().map(|(k, v)| (k.clone(), *v)).collect();
        entries.sort_by_key(|a| a.1);

        // Remove bottom 20%
        let remove_count = patterns.len() / 5;
        for (key, _) in entries.into_iter().take(remove_count) {
            patterns.remove(&key);
        }

        // Apply decay to remaining
        for count in patterns.values_mut() {
            *count = ((*count as f64) * PATTERN_DECAY_RATE) as u32;
        }
    }
}

// ============================================================================
// AROUSAL SIGNAL (Norepinephrine-like: Emotional Intensity)
// ============================================================================

/// Arousal signal inspired by norepinephrine's role in emotional processing.
///
/// In the brain, the locus coeruleus releases norepinephrine during emotionally
/// charged events, which strengthens amygdala-mediated memory encoding.
/// This creates "flashbulb memories" - vivid memories of emotionally intense events.
///
/// ## Detection Methods
///
/// 1. **Sentiment Analysis**: Detects emotional polarity and intensity
/// 2. **Intensity Keywords**: Domain-specific vocabulary indicating urgency/importance
/// 3. **Punctuation Patterns**: !!! and ??? indicate emotional emphasis
/// 4. **Capitalization**: ALL CAPS suggests heightened emotional state
#[derive(Debug)]
pub struct ArousalSignal {
    /// Sentiment analyzer for emotional content detection
    sentiment_analyzer: SentimentAnalyzer,
    /// Domain-specific keywords indicating high intensity
    intensity_keywords: HashSet<String>,
}

impl Default for ArousalSignal {
    fn default() -> Self {
        Self::new()
    }
}

impl ArousalSignal {
    /// Create a new arousal signal detector
    pub fn new() -> Self {
        Self {
            sentiment_analyzer: SentimentAnalyzer::new(),
            intensity_keywords: Self::default_intensity_keywords(),
        }
    }

    /// Add custom intensity keywords
    pub fn with_keywords(mut self, keywords: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for kw in keywords {
            self.intensity_keywords.insert(kw.into().to_lowercase());
        }
        self
    }

    /// Compute arousal score for content
    ///
    /// Returns a score from 0.0 (emotionally neutral) to 1.0 (highly arousing).
    pub fn compute(&self, content: &str) -> f64 {
        let sentiment = self.sentiment_analyzer.analyze(content);
        let keyword_score = self.compute_keyword_intensity(content);
        let punctuation_score = self.compute_punctuation_intensity(content);
        let capitalization_score = self.compute_capitalization_intensity(content);

        // Weighted combination
        let raw_arousal = sentiment.magnitude * 0.35
            + keyword_score * 0.30
            + punctuation_score * 0.20
            + capitalization_score * 0.15;

        raw_arousal.clamp(MIN_IMPORTANCE, MAX_IMPORTANCE)
    }

    /// Detect emotional markers in content
    pub fn detect_emotional_markers(&self, content: &str) -> Vec<EmotionalMarker> {
        let mut markers = Vec::new();
        let content_lower = content.to_lowercase();

        // Check for intensity keywords
        for keyword in &self.intensity_keywords {
            if content_lower.contains(keyword) {
                markers.push(EmotionalMarker {
                    marker_type: MarkerType::IntensityKeyword,
                    text: keyword.clone(),
                    intensity: 0.8,
                });
            }
        }

        // Check sentiment words
        let sentiment = self.sentiment_analyzer.analyze(content);
        for word in sentiment.contributing_words {
            markers.push(EmotionalMarker {
                marker_type: if sentiment.polarity >= 0.0 {
                    MarkerType::PositiveSentiment
                } else {
                    MarkerType::NegativeSentiment
                },
                text: word,
                intensity: sentiment.magnitude.abs(),
            });
        }

        // Check punctuation patterns
        if content.contains("!!!") || content.contains("???") {
            markers.push(EmotionalMarker {
                marker_type: MarkerType::PunctuationEmphasis,
                text: "Multiple punctuation".to_string(),
                intensity: 0.7,
            });
        }

        // Check capitalization
        let caps_ratio = self.compute_capitalization_intensity(content);
        if caps_ratio > 0.3 {
            markers.push(EmotionalMarker {
                marker_type: MarkerType::Capitalization,
                text: "Excessive capitalization".to_string(),
                intensity: caps_ratio,
            });
        }

        markers
    }

    /// Get explanation for arousal score
    pub fn explain(&self, content: &str) -> ArousalExplanation {
        let score = self.compute(content);
        let markers = self.detect_emotional_markers(content);
        let sentiment = self.sentiment_analyzer.analyze(content);

        ArousalExplanation {
            score,
            emotional_markers: markers,
            sentiment_polarity: sentiment.polarity,
            sentiment_magnitude: sentiment.magnitude,
        }
    }

    fn default_intensity_keywords() -> HashSet<String> {
        [
            // Urgency
            "urgent",
            "critical",
            "emergency",
            "immediately",
            "asap",
            "now",
            "deadline",
            "priority",
            "important",
            "crucial",
            "vital",
            // Negative intensity
            "error",
            "failed",
            "failure",
            "crash",
            "broken",
            "bug",
            "issue",
            "problem",
            "wrong",
            "bad",
            "terrible",
            "disaster",
            "catastrophe",
            "panic",
            "crisis",
            "alert",
            "warning",
            "danger",
            "risk",
            // Positive intensity
            "amazing",
            "incredible",
            "awesome",
            "excellent",
            "perfect",
            "brilliant",
            "breakthrough",
            "success",
            "solved",
            "fixed",
            "working",
            "victory",
            "achievement",
            "milestone",
            "celebration",
            // Technical urgency
            "production",
            "outage",
            "downtime",
            "security",
            "vulnerability",
            "exploit",
            "breach",
            "data loss",
            "corruption",
            "rollback",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute_keyword_intensity(&self, content: &str) -> f64 {
        let content_lower = content.to_lowercase();
        let word_count = content.split_whitespace().count().max(1) as f64;

        let keyword_count = self
            .intensity_keywords
            .iter()
            .filter(|kw| content_lower.contains(kw.as_str()))
            .count() as f64;

        // Normalize by content length but cap the intensity
        (keyword_count / word_count * 10.0).min(1.0)
    }

    fn compute_punctuation_intensity(&self, content: &str) -> f64 {
        let char_count = content.chars().count().max(1) as f64;

        let exclamation_count = content.matches('!').count() as f64;
        let question_count = content.matches('?').count() as f64;

        // Multiple consecutive punctuation is more intense
        let multi_punct =
            content.matches("!!").count() as f64 * 2.0 + content.matches("??").count() as f64 * 2.0;

        ((exclamation_count + question_count + multi_punct) / char_count * 20.0).min(1.0)
    }

    fn compute_capitalization_intensity(&self, content: &str) -> f64 {
        let letters: Vec<char> = content.chars().filter(|c| c.is_alphabetic()).collect();
        if letters.is_empty() {
            return 0.0;
        }

        let uppercase_count = letters.iter().filter(|c| c.is_uppercase()).count();
        let ratio = uppercase_count as f64 / letters.len() as f64;

        // Normal text has ~5-10% capitals (sentence starts, names)
        // Anything above 30% suggests emphasis
        if ratio > 0.3 {
            ((ratio - 0.3) * 2.0).min(1.0)
        } else {
            0.0
        }
    }
}

/// An emotional marker detected in content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalMarker {
    /// Type of emotional marker
    pub marker_type: MarkerType,
    /// The text that triggered this marker
    pub text: String,
    /// Intensity of this marker (0.0 to 1.0)
    pub intensity: f64,
}

/// Types of emotional markers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MarkerType {
    /// Positive sentiment word
    PositiveSentiment,
    /// Negative sentiment word
    NegativeSentiment,
    /// Intensity/urgency keyword
    IntensityKeyword,
    /// Emphatic punctuation (!!! ???)
    PunctuationEmphasis,
    /// Excessive capitalization
    Capitalization,
}

/// Explanation of arousal score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArousalExplanation {
    /// The computed arousal score
    pub score: f64,
    /// Emotional markers detected
    pub emotional_markers: Vec<EmotionalMarker>,
    /// Overall sentiment polarity (-1.0 to 1.0)
    pub sentiment_polarity: f64,
    /// Sentiment intensity (0.0 to 1.0)
    pub sentiment_magnitude: f64,
}

impl ArousalExplanation {
    /// Generate human-readable explanation
    pub fn explain(&self) -> String {
        let intensity_level = if self.score > 0.7 {
            "Highly emotional"
        } else if self.score > 0.4 {
            "Moderately emotional"
        } else {
            "Emotionally neutral"
        };

        let sentiment_desc = if self.sentiment_polarity > 0.3 {
            "positive"
        } else if self.sentiment_polarity < -0.3 {
            "negative"
        } else {
            "neutral"
        };

        format!(
            "{} content ({:.0}% arousal) with {} sentiment. {} markers detected.",
            intensity_level,
            self.score * 100.0,
            sentiment_desc,
            self.emotional_markers.len()
        )
    }
}

/// Simple keyword-based sentiment analyzer
#[derive(Debug)]
pub struct SentimentAnalyzer {
    /// Positive sentiment words with weights
    positive_words: HashMap<String, f64>,
    /// Negative sentiment words with weights
    negative_words: HashMap<String, f64>,
    /// Negation words that flip sentiment
    negation_words: HashSet<String>,
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SentimentAnalyzer {
    /// Create a new sentiment analyzer with default vocabulary
    pub fn new() -> Self {
        Self {
            positive_words: Self::default_positive_words(),
            negative_words: Self::default_negative_words(),
            negation_words: Self::default_negation_words(),
        }
    }

    /// Analyze sentiment of content
    pub fn analyze(&self, content: &str) -> SentimentResult {
        let lowercased = content.to_lowercase();
        let words: Vec<&str> = lowercased
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .collect();

        let mut positive_score = 0.0;
        let mut negative_score = 0.0;
        let mut contributing_words = Vec::new();
        let mut negated = false;

        for (i, word) in words.iter().enumerate() {
            // Check for negation
            if self.negation_words.contains(*word) {
                negated = true;
                continue;
            }

            // Check positive words
            if let Some(&weight) = self.positive_words.get(*word) {
                if negated {
                    negative_score += weight;
                    negated = false;
                } else {
                    positive_score += weight;
                }
                contributing_words.push(word.to_string());
            }

            // Check negative words
            if let Some(&weight) = self.negative_words.get(*word) {
                if negated {
                    positive_score += weight;
                    negated = false;
                } else {
                    negative_score += weight;
                }
                contributing_words.push(word.to_string());
            }

            // Reset negation after a few words
            if i > 0 && negated {
                negated = false;
            }
        }

        let word_count = words.len().max(1) as f64;
        let total = positive_score + negative_score;

        SentimentResult {
            polarity: if total > 0.0 {
                (positive_score - negative_score) / total
            } else {
                0.0
            },
            magnitude: ((positive_score + negative_score) / word_count * 5.0).min(1.0),
            contributing_words,
        }
    }

    fn default_positive_words() -> HashMap<String, f64> {
        [
            ("good", 0.5),
            ("great", 0.7),
            ("excellent", 0.9),
            ("amazing", 0.9),
            ("wonderful", 0.8),
            ("fantastic", 0.8),
            ("awesome", 0.8),
            ("brilliant", 0.8),
            ("perfect", 0.9),
            ("love", 0.8),
            ("happy", 0.6),
            ("pleased", 0.5),
            ("successful", 0.7),
            ("success", 0.7),
            ("solved", 0.6),
            ("fixed", 0.5),
            ("working", 0.4),
            ("works", 0.4),
            ("better", 0.5),
            ("best", 0.7),
            ("helpful", 0.5),
            ("useful", 0.5),
            ("efficient", 0.5),
            ("effective", 0.5),
            ("impressive", 0.7),
            ("outstanding", 0.8),
            ("superb", 0.8),
            ("remarkable", 0.7),
            ("thanks", 0.5),
            ("thank", 0.5),
            ("appreciate", 0.6),
            ("grateful", 0.6),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect()
    }

    fn default_negative_words() -> HashMap<String, f64> {
        [
            ("bad", 0.5),
            ("terrible", 0.9),
            ("horrible", 0.9),
            ("awful", 0.8),
            ("poor", 0.5),
            ("wrong", 0.5),
            ("error", 0.6),
            ("fail", 0.7),
            ("failed", 0.7),
            ("failure", 0.8),
            ("broken", 0.7),
            ("bug", 0.5),
            ("crash", 0.8),
            ("crashed", 0.8),
            ("problem", 0.5),
            ("issue", 0.4),
            ("hate", 0.8),
            ("angry", 0.7),
            ("frustrated", 0.6),
            ("annoyed", 0.5),
            ("disappointed", 0.6),
            ("confusing", 0.5),
            ("confused", 0.5),
            ("difficult", 0.4),
            ("hard", 0.3),
            ("impossible", 0.7),
            ("slow", 0.4),
            ("ugly", 0.5),
            ("useless", 0.7),
            ("waste", 0.6),
            ("pain", 0.5),
            ("painful", 0.6),
            ("nightmare", 0.8),
            ("disaster", 0.9),
            ("catastrophe", 0.9),
            ("crisis", 0.7),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect()
    }

    fn default_negation_words() -> HashSet<String> {
        [
            "not",
            "no",
            "never",
            "neither",
            "nobody",
            "nothing",
            "nowhere",
            "dont",
            "doesn't",
            "didn't",
            "won't",
            "wouldn't",
            "couldn't",
            "shouldn't",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "cannot",
            "can't",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }
}

/// Result of sentiment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    /// Polarity from -1.0 (negative) to 1.0 (positive)
    pub polarity: f64,
    /// Intensity/magnitude of sentiment (0.0 to 1.0)
    pub magnitude: f64,
    /// Words that contributed to the sentiment
    pub contributing_words: Vec<String>,
}

// ============================================================================
// REWARD SIGNAL (Dopamine-like: Positive Outcomes)
// ============================================================================

/// Reward signal inspired by dopamine's role in reinforcement learning.
///
/// In the brain, dopamine release in the nucleus accumbens reinforces behaviors
/// that lead to positive outcomes. This signal tracks which memories have been
/// associated with successful outcomes and should be prioritized.
///
/// ## How It Works
///
/// 1. Records outcomes when memories are used (helpful, not helpful, etc.)
/// 2. Learns patterns that predict positive outcomes
/// 3. Gives higher importance to memories with track record of success
#[derive(Debug)]
pub struct RewardSignal {
    /// Outcome history: memory_id -> outcomes
    outcome_history: Arc<RwLock<HashMap<String, Outcome>>>,
    /// Learned reward patterns
    reward_patterns: Arc<RwLock<Vec<RewardPattern>>>,
}

impl Default for RewardSignal {
    fn default() -> Self {
        Self::new()
    }
}

impl RewardSignal {
    /// Create a new reward signal tracker
    pub fn new() -> Self {
        Self {
            outcome_history: Arc::new(RwLock::new(HashMap::new())),
            reward_patterns: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Record an outcome for a memory
    pub fn record_outcome(&self, memory_id: &str, outcome_type: OutcomeType) {
        if let Ok(mut history) = self.outcome_history.write() {
            let outcome = history
                .entry(memory_id.to_string())
                .or_insert_with(|| Outcome::new(memory_id));

            outcome.record(outcome_type);

            // Prune old entries if needed
            if history.len() > MAX_OUTCOME_HISTORY {
                self.prune_old_outcomes(&mut history);
            }
        }
    }

    /// Record outcome with context for pattern learning
    pub fn record_outcome_with_context(
        &self,
        memory_id: &str,
        outcome_type: OutcomeType,
        context_tags: &[String],
    ) {
        self.record_outcome(memory_id, outcome_type.clone());

        // Learn pattern from this outcome
        if matches!(
            outcome_type,
            OutcomeType::Helpful | OutcomeType::VeryHelpful
        ) {
            self.learn_pattern(context_tags, 1.0);
        } else if matches!(outcome_type, OutcomeType::NotHelpful | OutcomeType::Harmful) {
            self.learn_pattern(context_tags, -0.5);
        }
    }

    /// Compute reward score for a memory
    pub fn compute(&self, memory_id: &str) -> f64 {
        let history = match self.outcome_history.read() {
            Ok(h) => h,
            Err(_) => return 0.5, // Default neutral
        };

        match history.get(memory_id) {
            Some(outcome) => outcome.reward_score(),
            None => 0.5, // No history = neutral
        }
    }

    /// Compute reward score with context-based prediction
    pub fn compute_with_context(&self, memory_id: &str, context_tags: &[String]) -> f64 {
        let base_score = self.compute(memory_id);
        let pattern_score = self.compute_pattern_score(context_tags);

        // Combine historical performance with pattern prediction
        (base_score * 0.7 + pattern_score * 0.3).clamp(MIN_IMPORTANCE, MAX_IMPORTANCE)
    }

    /// Get explanation for reward score
    pub fn explain(&self, memory_id: &str) -> RewardExplanation {
        let score = self.compute(memory_id);
        let history = self.outcome_history.read().ok();

        let (helpful_count, total_count, last_outcome) = match &history {
            Some(h) => match h.get(memory_id) {
                Some(outcome) => (
                    outcome.helpful_count,
                    outcome.total_count,
                    outcome.last_outcome.clone(),
                ),
                None => (0, 0, None),
            },
            None => (0, 0, None),
        };

        RewardExplanation {
            score,
            helpful_count,
            total_count,
            helpfulness_ratio: if total_count > 0 {
                helpful_count as f64 / total_count as f64
            } else {
                0.5
            },
            last_outcome,
        }
    }

    /// Get top performing memories
    pub fn get_top_performers(&self, limit: usize) -> Vec<(String, f64)> {
        let history = match self.outcome_history.read() {
            Ok(h) => h,
            Err(_) => return vec![],
        };

        let mut scores: Vec<_> = history
            .iter()
            .map(|(id, outcome)| (id.clone(), outcome.reward_score()))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(limit);
        scores
    }

    fn learn_pattern(&self, tags: &[String], reward: f64) {
        if let Ok(mut patterns) = self.reward_patterns.write() {
            // Check for existing pattern
            for pattern in patterns.iter_mut() {
                if pattern.matches(tags) {
                    pattern.update(reward);
                    return;
                }
            }

            // Create new pattern
            patterns.push(RewardPattern::new(tags, reward));

            // Limit pattern count
            if patterns.len() > 1000 {
                patterns.sort_by(|a, b| {
                    b.strength
                        .partial_cmp(&a.strength)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                patterns.truncate(500);
            }
        }
    }

    fn compute_pattern_score(&self, tags: &[String]) -> f64 {
        let patterns = match self.reward_patterns.read() {
            Ok(p) => p,
            Err(_) => return 0.5,
        };

        let matching: Vec<_> = patterns.iter().filter(|p| p.matches(tags)).collect();

        if matching.is_empty() {
            return 0.5;
        }

        let total_strength: f64 = matching.iter().map(|p| p.strength.abs()).sum();
        if total_strength == 0.0 {
            return 0.5;
        }

        let weighted_sum: f64 = matching
            .iter()
            .map(|p| p.strength * (0.5 + p.strength.signum() * 0.5))
            .sum();

        (weighted_sum / total_strength).clamp(0.0, 1.0)
    }

    fn prune_old_outcomes(&self, history: &mut HashMap<String, Outcome>) {
        // Remove outcomes with lowest scores and oldest access
        let mut entries: Vec<_> = history
            .iter()
            .map(|(k, v)| (k.clone(), v.reward_score(), v.last_accessed))
            .collect();

        entries.sort_by(|a, b| {
            // Sort by score, then by recency
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.2.cmp(&a.2))
        });

        // Keep top entries
        let keep_count = MAX_OUTCOME_HISTORY * 4 / 5;
        let remove: HashSet<_> = entries
            .into_iter()
            .skip(keep_count)
            .map(|(id, _, _)| id)
            .collect();

        history.retain(|k, _| !remove.contains(k));
    }
}

/// Outcome tracking for a single memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    /// Memory ID
    pub memory_id: String,
    /// Total number of times this memory was used
    pub total_count: u32,
    /// Number of times marked as helpful
    pub helpful_count: u32,
    /// Number of times marked as very helpful
    pub very_helpful_count: u32,
    /// Number of times marked as not helpful
    pub not_helpful_count: u32,
    /// Number of times marked as harmful
    pub harmful_count: u32,
    /// Last outcome type
    pub last_outcome: Option<OutcomeType>,
    /// Last time this memory was accessed
    pub last_accessed: DateTime<Utc>,
    /// When tracking started
    pub created_at: DateTime<Utc>,
}

impl Outcome {
    fn new(memory_id: &str) -> Self {
        let now = Utc::now();
        Self {
            memory_id: memory_id.to_string(),
            total_count: 0,
            helpful_count: 0,
            very_helpful_count: 0,
            not_helpful_count: 0,
            harmful_count: 0,
            last_outcome: None,
            last_accessed: now,
            created_at: now,
        }
    }

    fn record(&mut self, outcome: OutcomeType) {
        self.total_count += 1;
        self.last_accessed = Utc::now();

        match &outcome {
            OutcomeType::Helpful => self.helpful_count += 1,
            OutcomeType::VeryHelpful => {
                self.helpful_count += 1;
                self.very_helpful_count += 1;
            }
            OutcomeType::NotHelpful => self.not_helpful_count += 1,
            OutcomeType::Harmful => self.harmful_count += 1,
            OutcomeType::Neutral => {}
        }

        self.last_outcome = Some(outcome);
    }

    fn reward_score(&self) -> f64 {
        if self.total_count == 0 {
            return 0.5;
        }

        // Weighted scoring
        let positive = self.helpful_count as f64 + self.very_helpful_count as f64 * 0.5;
        let negative = self.not_helpful_count as f64 + self.harmful_count as f64 * 2.0;

        let ratio = positive / (positive + negative + 1.0);

        // Apply confidence based on sample size
        let confidence = 1.0 - (1.0 / (self.total_count as f64 + 1.0));

        // Blend with neutral based on confidence
        (0.5 * (1.0 - confidence) + ratio * confidence).clamp(MIN_IMPORTANCE, MAX_IMPORTANCE)
    }
}

/// Types of outcomes that can be recorded
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OutcomeType {
    /// Memory was very helpful
    VeryHelpful,
    /// Memory was helpful
    Helpful,
    /// Memory was neutral/neither helpful nor harmful
    Neutral,
    /// Memory was not helpful
    NotHelpful,
    /// Memory was harmful/misleading
    Harmful,
}

/// Learned pattern that predicts reward
#[derive(Debug, Clone)]
struct RewardPattern {
    /// Tags that define this pattern
    tags: HashSet<String>,
    /// Strength of this pattern (-1.0 to 1.0)
    strength: f64,
    /// Number of times this pattern was observed
    observations: u32,
}

impl RewardPattern {
    fn new(tags: &[String], initial_reward: f64) -> Self {
        Self {
            tags: tags.iter().cloned().collect(),
            strength: initial_reward.clamp(-1.0, 1.0),
            observations: 1,
        }
    }

    fn matches(&self, tags: &[String]) -> bool {
        let tag_set: HashSet<_> = tags.iter().cloned().collect();
        let overlap = self.tags.intersection(&tag_set).count();
        overlap >= self.tags.len().min(tag_set.len()).max(1) / 2
    }

    fn update(&mut self, reward: f64) {
        self.observations += 1;
        // Exponential moving average
        let alpha = 2.0 / (self.observations as f64 + 1.0);
        self.strength = self.strength * (1.0 - alpha) + reward * alpha;
    }
}

/// Explanation of reward score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardExplanation {
    /// The computed reward score
    pub score: f64,
    /// Number of times marked helpful
    pub helpful_count: u32,
    /// Total number of uses
    pub total_count: u32,
    /// Ratio of helpful to total
    pub helpfulness_ratio: f64,
    /// Most recent outcome
    pub last_outcome: Option<OutcomeType>,
}

impl RewardExplanation {
    /// Generate human-readable explanation
    pub fn explain(&self) -> String {
        if self.total_count == 0 {
            "No usage history yet. Default neutral score.".to_string()
        } else {
            format!(
                "Helpful {}/{} times ({:.0}%). Score: {:.2}",
                self.helpful_count,
                self.total_count,
                self.helpfulness_ratio * 100.0,
                self.score
            )
        }
    }
}

// ============================================================================
// ATTENTION SIGNAL (Acetylcholine-like: Focus & Learning Mode)
// ============================================================================

/// Attention signal inspired by acetylcholine's role in attention and learning.
///
/// In the brain, the basal forebrain releases acetylcholine during focused
/// attention and active learning, which gates plasticity in cortical circuits.
/// This signal detects when the user is in an active learning/focused state.
///
/// ## Detection Methods
///
/// 1. **Access Patterns**: Frequent, focused access suggests active learning
/// 2. **Session Analysis**: Sustained engagement indicates focused state
/// 3. **Query Patterns**: Exploratory queries suggest learning mode
#[derive(Debug)]
pub struct AttentionSignal {
    /// Focus detector for analyzing access patterns
    focus_detector: FocusDetector,
    /// Whether learning mode is currently active
    learning_mode_active: Arc<RwLock<bool>>,
    /// Recent sessions for learning mode detection
    sessions: Arc<RwLock<VecDeque<Session>>>,
}

impl Default for AttentionSignal {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionSignal {
    /// Create a new attention signal detector
    pub fn new() -> Self {
        Self {
            focus_detector: FocusDetector::new(),
            learning_mode_active: Arc::new(RwLock::new(false)),
            sessions: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
        }
    }

    /// Compute attention score from access pattern
    pub fn compute(&self, access_pattern: &AccessPattern) -> f64 {
        let focus_score = self.focus_detector.compute_focus(access_pattern);
        let learning_mode = self.is_learning_mode();

        // Boost score if in learning mode
        let base_score = focus_score;
        let learning_boost = if learning_mode { 0.2 } else { 0.0 };

        (base_score + learning_boost).clamp(MIN_IMPORTANCE, MAX_IMPORTANCE)
    }

    /// Record a session activity
    pub fn record_session_activity(&self, session: Session) {
        if let Ok(mut sessions) = self.sessions.write() {
            sessions.push_back(session);

            // Keep only recent sessions
            while sessions.len() > 100 {
                sessions.pop_front();
            }
        }

        // Update learning mode based on sessions
        self.update_learning_mode();
    }

    /// Check if user is in learning mode
    pub fn is_learning_mode(&self) -> bool {
        self.learning_mode_active
            .read()
            .map(|m| *m)
            .unwrap_or(false)
    }

    /// Detect learning mode from a session
    pub fn detect_learning_mode(&self, session: &Session) -> bool {
        // High query frequency suggests active exploration
        let high_query_rate = session.query_count as f64 / session.duration_minutes.max(1.0) > 2.0;

        // Diverse access patterns suggest learning
        let diverse_access = session.unique_memories_accessed > 5;

        // Low edit ratio (more reading than writing) suggests learning
        let reading_mode = (session.edit_count as f64 / session.query_count.max(1) as f64) < 0.3;

        // Long session duration suggests engagement
        let sustained = session.duration_minutes > 15.0;

        (high_query_rate as u8 + diverse_access as u8 + reading_mode as u8 + sustained as u8) >= 2
    }

    /// Get explanation for attention score
    pub fn explain(&self, access_pattern: &AccessPattern) -> AttentionExplanation {
        let score = self.compute(access_pattern);
        let learning_mode = self.is_learning_mode();
        let focus_metrics = self.focus_detector.get_focus_metrics(access_pattern);

        AttentionExplanation {
            score,
            learning_mode_active: learning_mode,
            access_frequency: focus_metrics.access_frequency,
            session_depth: focus_metrics.session_depth,
            query_diversity: focus_metrics.query_diversity,
        }
    }

    /// Set learning mode manually (for external triggers)
    pub fn set_learning_mode(&self, active: bool) {
        if let Ok(mut mode) = self.learning_mode_active.write() {
            *mode = active;
        }
    }

    fn update_learning_mode(&self) {
        let sessions = match self.sessions.read() {
            Ok(s) => s,
            Err(_) => return,
        };

        let now = Utc::now();
        let cutoff = now - Duration::minutes(LEARNING_MODE_TIMEOUT_MINUTES);

        // Check recent sessions for learning indicators
        let recent_sessions: Vec<_> = sessions.iter().filter(|s| s.start_time > cutoff).collect();

        let learning_sessions = recent_sessions
            .iter()
            .filter(|s| self.detect_learning_mode(s))
            .count();

        let is_learning = !recent_sessions.is_empty()
            && learning_sessions as f64 / recent_sessions.len() as f64 > 0.5;

        if let Ok(mut mode) = self.learning_mode_active.write() {
            *mode = is_learning;
        }
    }
}

/// Access pattern data for attention analysis
#[derive(Debug, Clone, Default)]
pub struct AccessPattern {
    /// Memory IDs accessed in this pattern
    pub memory_ids: Vec<String>,
    /// Time between accesses (seconds)
    pub inter_access_times: Vec<f64>,
    /// Queries made
    pub queries: Vec<String>,
    /// Total duration of this access pattern (seconds)
    pub duration_seconds: f64,
    /// Whether accesses were sequential (related) or random
    pub sequential_access: bool,
    /// Number of repeat accesses to same memories
    pub repeat_access_count: u32,
}

impl AccessPattern {
    /// Create a new access pattern
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an access event
    pub fn add_access(&mut self, memory_id: impl Into<String>, time_since_last: f64) {
        self.memory_ids.push(memory_id.into());
        if time_since_last > 0.0 {
            self.inter_access_times.push(time_since_last);
        }
    }

    /// Add a query
    pub fn add_query(&mut self, query: impl Into<String>) {
        self.queries.push(query.into());
    }

    /// Get unique memory count
    pub fn unique_memories(&self) -> usize {
        let set: HashSet<_> = self.memory_ids.iter().collect();
        set.len()
    }

    /// Get average inter-access time
    pub fn avg_inter_access_time(&self) -> f64 {
        if self.inter_access_times.is_empty() {
            return 0.0;
        }
        self.inter_access_times.iter().sum::<f64>() / self.inter_access_times.len() as f64
    }
}

/// Session data for learning mode detection
#[derive(Debug, Clone)]
pub struct Session {
    /// Session ID
    pub session_id: String,
    /// When session started
    pub start_time: DateTime<Utc>,
    /// Duration in minutes
    pub duration_minutes: f64,
    /// Number of queries made
    pub query_count: u32,
    /// Number of edits/actions made
    pub edit_count: u32,
    /// Number of unique memories accessed
    pub unique_memories_accessed: u32,
    /// Whether session includes documentation viewing
    pub viewed_docs: bool,
    /// Query topics (for diversity analysis)
    pub query_topics: Vec<String>,
}

impl Session {
    /// Create a new session
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            start_time: Utc::now(),
            duration_minutes: 0.0,
            query_count: 0,
            edit_count: 0,
            unique_memories_accessed: 0,
            viewed_docs: false,
            query_topics: Vec::new(),
        }
    }
}

/// Focus detector for analyzing attention patterns
#[derive(Debug)]
struct FocusDetector {
    /// Baseline for "normal" inter-access time (seconds)
    baseline_inter_access: f64,
    /// Baseline for session depth (for future depth-weighted focus scoring)
    #[allow(dead_code)]
    baseline_session_depth: f64,
}

impl Default for FocusDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl FocusDetector {
    fn new() -> Self {
        Self {
            baseline_inter_access: 60.0, // 1 minute
            baseline_session_depth: 5.0,
        }
    }

    fn compute_focus(&self, pattern: &AccessPattern) -> f64 {
        let metrics = self.get_focus_metrics(pattern);

        // Combine metrics with weights
        let frequency_score = (metrics.access_frequency * 0.5).min(1.0);
        let depth_score = (metrics.session_depth / 10.0).min(1.0);
        let diversity_score = metrics.query_diversity;

        (frequency_score * 0.4 + depth_score * 0.35 + diversity_score * 0.25)
            .clamp(MIN_IMPORTANCE, MAX_IMPORTANCE)
    }

    fn get_focus_metrics(&self, pattern: &AccessPattern) -> FocusMetrics {
        let avg_time = pattern.avg_inter_access_time();

        // Access frequency: faster access = more focused
        let access_frequency = if avg_time > 0.0 {
            (self.baseline_inter_access / avg_time).min(2.0)
        } else {
            1.0
        };

        // Session depth: more unique memories = deeper exploration
        let session_depth = pattern.unique_memories() as f64;

        // Query diversity: varied queries suggest active exploration
        let unique_query_words: HashSet<String> = pattern
            .queries
            .iter()
            .flat_map(|q| {
                q.to_lowercase()
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            })
            .filter(|w| w.len() > 2)
            .collect();

        let query_diversity = (unique_query_words.len() as f64 / 20.0).min(1.0);

        FocusMetrics {
            access_frequency,
            session_depth,
            query_diversity,
        }
    }
}

/// Focus metrics for transparency
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FocusMetrics {
    access_frequency: f64,
    session_depth: f64,
    query_diversity: f64,
}

/// Explanation of attention score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionExplanation {
    /// The computed attention score
    pub score: f64,
    /// Whether learning mode is active
    pub learning_mode_active: bool,
    /// Normalized access frequency
    pub access_frequency: f64,
    /// Depth of session exploration
    pub session_depth: f64,
    /// Diversity of queries
    pub query_diversity: f64,
}

impl AttentionExplanation {
    /// Generate human-readable explanation
    pub fn explain(&self) -> String {
        let focus_level = if self.score > 0.7 {
            "Highly focused"
        } else if self.score > 0.4 {
            "Moderately focused"
        } else {
            "Low focus"
        };

        let learning_str = if self.learning_mode_active {
            " (Learning mode active)"
        } else {
            ""
        };

        format!(
            "{}{} - Score: {:.2}. Access freq: {:.1}x, Depth: {:.0}, Diversity: {:.0}%",
            focus_level,
            learning_str,
            self.score,
            self.access_frequency,
            self.session_depth,
            self.query_diversity * 100.0
        )
    }
}

// ============================================================================
// COMPOSITE IMPORTANCE SIGNALS
// ============================================================================

/// Multi-dimensional importance scoring inspired by neuromodulator systems.
///
/// Combines four independent importance signals into a composite score:
/// - Novelty (Dopamine): How surprising/unexpected is this content?
/// - Arousal (Norepinephrine): How emotionally intense is this content?
/// - Reward (Dopamine): How often has this content been helpful?
/// - Attention (Acetylcholine): Is the user actively focused/learning?
///
/// Each signal contributes to the final importance score with configurable weights.
#[derive(Debug)]
pub struct ImportanceSignals {
    /// Novelty signal (dopamine-like: prediction error, surprise)
    pub novelty: NoveltySignal,
    /// Arousal signal (norepinephrine-like: emotional intensity)
    pub arousal: ArousalSignal,
    /// Reward signal (dopamine-like: positive outcomes)
    pub reward: RewardSignal,
    /// Attention signal (acetylcholine-like: focus, learning mode)
    pub attention: AttentionSignal,
    /// Weights for composite calculation
    weights: CompositeWeights,
}

impl Default for ImportanceSignals {
    fn default() -> Self {
        Self::new()
    }
}

impl ImportanceSignals {
    /// Create a new importance signals system
    pub fn new() -> Self {
        Self {
            novelty: NoveltySignal::new(),
            arousal: ArousalSignal::new(),
            reward: RewardSignal::new(),
            attention: AttentionSignal::new(),
            weights: CompositeWeights::default(),
        }
    }

    /// Create with custom weights
    pub fn with_weights(mut self, weights: CompositeWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Compute composite importance for content
    pub fn compute_importance(&self, content: &str, context: &Context) -> ImportanceScore {
        let novelty = self.novelty.compute(content, context);
        let arousal = self.arousal.compute(content);

        // For reward and attention, we need additional context
        let reward = context
            .recent_memory_ids
            .first()
            .map(|id| self.reward.compute(id))
            .unwrap_or(0.5);

        let access_pattern = AccessPattern::default();
        let attention = self.attention.compute(&access_pattern);

        self.compute_composite(novelty, arousal, reward, attention, content, context)
    }

    /// Compute composite importance with explicit values
    pub fn compute_importance_explicit(
        &self,
        content: &str,
        context: &Context,
        memory_id: Option<&str>,
        access_pattern: Option<&AccessPattern>,
    ) -> ImportanceScore {
        let novelty = self.novelty.compute(content, context);
        let arousal = self.arousal.compute(content);

        let reward = memory_id.map(|id| self.reward.compute(id)).unwrap_or(0.5);

        let attention = access_pattern
            .map(|p| self.attention.compute(p))
            .unwrap_or(0.5);

        self.compute_composite(novelty, arousal, reward, attention, content, context)
    }

    /// Update novelty model (learning)
    pub fn learn_content(&mut self, content: &str) {
        self.novelty.update_model(content);
    }

    /// Record outcome for reward learning
    pub fn record_outcome(&self, memory_id: &str, outcome: OutcomeType) {
        self.reward.record_outcome(memory_id, outcome);
    }

    /// Record session for attention tracking
    pub fn record_session(&self, session: Session) {
        self.attention.record_session_activity(session);
    }

    /// Get current weights
    pub fn weights(&self) -> &CompositeWeights {
        &self.weights
    }

    /// Set weights
    pub fn set_weights(&mut self, weights: CompositeWeights) {
        self.weights = weights;
    }

    fn compute_composite(
        &self,
        novelty: f64,
        arousal: f64,
        reward: f64,
        attention: f64,
        content: &str,
        context: &Context,
    ) -> ImportanceScore {
        // Weighted composite
        let composite = novelty * self.weights.novelty
            + arousal * self.weights.arousal
            + reward * self.weights.reward
            + attention * self.weights.attention;

        // Encoding boost: high importance = stronger encoding
        let encoding_boost = 1.0 + (composite - 0.5) * 0.6; // 0.7 to 1.3

        // Consolidation priority based on score
        let consolidation_priority = if composite > 0.8 {
            ConsolidationPriority::Critical
        } else if composite > 0.6 {
            ConsolidationPriority::High
        } else if composite > 0.4 {
            ConsolidationPriority::Normal
        } else {
            ConsolidationPriority::Low
        };

        // Build explanations
        let novelty_explanation = self.novelty.explain(content, context);
        let arousal_explanation = self.arousal.explain(content);
        let reward_explanation = context
            .recent_memory_ids
            .first()
            .map(|id| self.reward.explain(id))
            .unwrap_or(RewardExplanation {
                score: 0.5,
                helpful_count: 0,
                total_count: 0,
                helpfulness_ratio: 0.5,
                last_outcome: None,
            });
        let attention_explanation = self.attention.explain(&AccessPattern::default());

        ImportanceScore {
            composite: composite.clamp(MIN_IMPORTANCE, MAX_IMPORTANCE),
            novelty,
            arousal,
            reward,
            attention,
            encoding_boost,
            consolidation_priority,
            weights_used: self.weights.clone(),
            novelty_explanation: Some(novelty_explanation),
            arousal_explanation: Some(arousal_explanation),
            reward_explanation: Some(reward_explanation),
            attention_explanation: Some(attention_explanation),
            computed_at: Utc::now(),
        }
    }
}

/// Weights for composite importance calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeWeights {
    /// Weight for novelty signal
    pub novelty: f64,
    /// Weight for arousal signal
    pub arousal: f64,
    /// Weight for reward signal
    pub reward: f64,
    /// Weight for attention signal
    pub attention: f64,
}

impl Default for CompositeWeights {
    fn default() -> Self {
        Self {
            novelty: DEFAULT_NOVELTY_WEIGHT,
            arousal: DEFAULT_AROUSAL_WEIGHT,
            reward: DEFAULT_REWARD_WEIGHT,
            attention: DEFAULT_ATTENTION_WEIGHT,
        }
    }
}

impl CompositeWeights {
    /// Create with custom weights (will be normalized)
    pub fn new(novelty: f64, arousal: f64, reward: f64, attention: f64) -> Self {
        let total = novelty + arousal + reward + attention;
        if total == 0.0 {
            return Self::default();
        }

        Self {
            novelty: novelty / total,
            arousal: arousal / total,
            reward: reward / total,
            attention: attention / total,
        }
    }

    /// Validate that weights sum to approximately 1.0
    pub fn is_valid(&self) -> bool {
        let sum = self.novelty + self.arousal + self.reward + self.attention;
        (sum - 1.0).abs() < 0.01
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let total = self.novelty + self.arousal + self.reward + self.attention;
        if total > 0.0 {
            self.novelty /= total;
            self.arousal /= total;
            self.reward /= total;
            self.attention /= total;
        }
    }
}

/// Composite importance score with full breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceScore {
    /// Final composite importance score (0.0 to 1.0)
    pub composite: f64,
    /// Novelty component score
    pub novelty: f64,
    /// Arousal component score
    pub arousal: f64,
    /// Reward component score
    pub reward: f64,
    /// Attention component score
    pub attention: f64,
    /// How much to boost encoding strength (typically 0.7 to 1.3)
    pub encoding_boost: f64,
    /// Priority for memory consolidation
    pub consolidation_priority: ConsolidationPriority,
    /// Weights used in calculation
    pub weights_used: CompositeWeights,
    /// Detailed novelty explanation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub novelty_explanation: Option<NoveltyExplanation>,
    /// Detailed arousal explanation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arousal_explanation: Option<ArousalExplanation>,
    /// Detailed reward explanation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward_explanation: Option<RewardExplanation>,
    /// Detailed attention explanation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention_explanation: Option<AttentionExplanation>,
    /// When this score was computed
    pub computed_at: DateTime<Utc>,
}

impl ImportanceScore {
    /// Get human-readable summary of the score
    pub fn summary(&self) -> String {
        format!(
            "Importance: {:.2} (N:{:.0}% A:{:.0}% R:{:.0}% At:{:.0}%) - {} priority",
            self.composite,
            self.novelty * 100.0,
            self.arousal * 100.0,
            self.reward * 100.0,
            self.attention * 100.0,
            match self.consolidation_priority {
                ConsolidationPriority::Critical => "CRITICAL",
                ConsolidationPriority::High => "High",
                ConsolidationPriority::Normal => "Normal",
                ConsolidationPriority::Low => "Low",
            }
        )
    }

    /// Get explanation for why this content is important
    pub fn explain(&self) -> String {
        let mut parts = Vec::new();

        if let Some(ref novelty) = self.novelty_explanation {
            parts.push(format!("Novelty: {}", novelty.explain()));
        }

        if let Some(ref arousal) = self.arousal_explanation {
            parts.push(format!("Arousal: {}", arousal.explain()));
        }

        if let Some(ref reward) = self.reward_explanation {
            parts.push(format!("Reward: {}", reward.explain()));
        }

        if let Some(ref attention) = self.attention_explanation {
            parts.push(format!("Attention: {}", attention.explain()));
        }

        parts.join("\n")
    }

    /// Get the dominant signal (highest contributor)
    pub fn dominant_signal(&self) -> &'static str {
        let weighted = [
            (self.novelty * self.weights_used.novelty, "Novelty"),
            (self.arousal * self.weights_used.arousal, "Arousal"),
            (self.reward * self.weights_used.reward, "Reward"),
            (self.attention * self.weights_used.attention, "Attention"),
        ];

        weighted
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|x| x.1)
            .unwrap_or("Unknown")
    }
}

/// Priority levels for memory consolidation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConsolidationPriority {
    /// Low priority - process last, may be pruned
    Low,
    /// Normal priority - standard processing
    Normal,
    /// High priority - process early, preserve longer
    High,
    /// Critical priority - process immediately, never prune
    Critical,
}

impl ConsolidationPriority {
    /// Get decay rate modifier (lower = slower decay)
    pub fn decay_modifier(&self) -> f64 {
        match self {
            ConsolidationPriority::Critical => 0.5, // 50% slower decay
            ConsolidationPriority::High => 0.75,    // 25% slower decay
            ConsolidationPriority::Normal => 1.0,   // Normal decay
            ConsolidationPriority::Low => 1.25,     // 25% faster decay
        }
    }

    /// Get retrieval boost
    pub fn retrieval_boost(&self) -> f64 {
        match self {
            ConsolidationPriority::Critical => 1.3,
            ConsolidationPriority::High => 1.15,
            ConsolidationPriority::Normal => 1.0,
            ConsolidationPriority::Low => 0.9,
        }
    }
}

// ============================================================================
// INTEGRATION HELPERS
// ============================================================================

/// Configuration for importance-aware encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceEncodingConfig {
    /// Minimum importance for enhanced encoding
    pub enhanced_encoding_threshold: f64,
    /// Maximum encoding boost factor
    pub max_encoding_boost: f64,
    /// Whether to use importance for initial stability
    pub importance_affects_stability: bool,
    /// Base stability modifier per importance point
    pub stability_modifier_per_importance: f64,
}

impl Default for ImportanceEncodingConfig {
    fn default() -> Self {
        Self {
            enhanced_encoding_threshold: 0.6,
            max_encoding_boost: 1.5,
            importance_affects_stability: true,
            stability_modifier_per_importance: 0.5,
        }
    }
}

/// Configuration for importance-aware consolidation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceConsolidationConfig {
    /// Process high-importance memories first
    pub prioritize_high_importance: bool,
    /// Minimum importance to avoid pruning
    pub pruning_protection_threshold: f64,
    /// Boost replay frequency for high-importance memories
    pub replay_importance_scaling: bool,
}

impl Default for ImportanceConsolidationConfig {
    fn default() -> Self {
        Self {
            prioritize_high_importance: true,
            pruning_protection_threshold: 0.7,
            replay_importance_scaling: true,
        }
    }
}

/// Configuration for importance-aware retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceRetrievalConfig {
    /// Weight of importance in ranking (0.0 to 1.0)
    pub importance_ranking_weight: f64,
    /// Boost retrieval score based on consolidation priority
    pub apply_priority_boost: bool,
    /// Include importance breakdown in results
    pub include_importance_breakdown: bool,
}

impl Default for ImportanceRetrievalConfig {
    fn default() -> Self {
        Self {
            importance_ranking_weight: 0.2,
            apply_priority_boost: true,
            include_importance_breakdown: true,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_novelty_signal_basic() {
        let mut novelty = NoveltySignal::new();
        let context = Context::current();

        // First time seeing content should be novel
        let score1 = novelty.compute("The quick brown fox jumps over the lazy dog", &context);
        assert!(score1 > 0.5, "New content should be novel");

        // Learn the pattern
        novelty.update_model("The quick brown fox jumps over the lazy dog");
        novelty.update_model("The quick brown fox jumps over the lazy dog");
        novelty.update_model("The quick brown fox jumps over the lazy dog");

        // Same content should be less novel
        let score2 = novelty.compute("The quick brown fox jumps over the lazy dog", &context);
        assert!(score2 < score1, "Repeated content should be less novel");
    }

    #[test]
    fn test_arousal_signal_emotional_content() {
        let arousal = ArousalSignal::new();

        // Neutral content
        let neutral_score = arousal.compute("The meeting is scheduled for tomorrow.");

        // Highly emotional content
        let emotional_score =
            arousal.compute("CRITICAL ERROR!!! Production database is DOWN! Data loss imminent!");

        assert!(
            emotional_score > neutral_score,
            "Emotional content should have higher arousal"
        );
        assert!(
            emotional_score > 0.6,
            "Highly emotional content should score high"
        );
    }

    #[test]
    fn test_arousal_signal_markers() {
        let arousal = ArousalSignal::new();
        let markers = arousal.detect_emotional_markers("URGENT: Critical failure!!!");

        assert!(!markers.is_empty(), "Should detect emotional markers");

        let has_keyword = markers
            .iter()
            .any(|m| m.marker_type == MarkerType::IntensityKeyword);
        assert!(has_keyword, "Should detect intensity keyword");
    }

    #[test]
    fn test_reward_signal_tracking() {
        let reward = RewardSignal::new();

        // Record positive outcomes
        reward.record_outcome("mem-1", OutcomeType::Helpful);
        reward.record_outcome("mem-1", OutcomeType::VeryHelpful);
        reward.record_outcome("mem-1", OutcomeType::Helpful);

        let score = reward.compute("mem-1");
        assert!(
            score > 0.5,
            "Memory with positive outcomes should score high"
        );

        // Record negative outcomes for different memory
        reward.record_outcome("mem-2", OutcomeType::NotHelpful);
        reward.record_outcome("mem-2", OutcomeType::NotHelpful);

        let neg_score = reward.compute("mem-2");
        assert!(
            neg_score < 0.5,
            "Memory with negative outcomes should score low"
        );
    }

    #[test]
    fn test_attention_signal_learning_mode() {
        let attention = AttentionSignal::new();

        // Create a learning-like session
        let learning_session = Session {
            session_id: "s1".to_string(),
            start_time: Utc::now(),
            duration_minutes: 45.0,
            query_count: 20,
            edit_count: 2,
            unique_memories_accessed: 15,
            viewed_docs: true,
            query_topics: vec!["rust".to_string(), "async".to_string(), "tokio".to_string()],
        };

        assert!(
            attention.detect_learning_mode(&learning_session),
            "Should detect learning mode"
        );
    }

    #[test]
    fn test_composite_importance() {
        let signals = ImportanceSignals::new();
        let context = Context::current()
            .with_project("test-project")
            .with_learning_session(true);

        // Test with emotional, novel content
        let score = signals.compute_importance(
            "BREAKTHROUGH: Solved the critical performance issue that was blocking release!!!",
            &context,
        );

        assert!(score.composite > 0.5, "Important content should score high");
        assert!(
            score.arousal > 0.5,
            "Emotional content should have high arousal"
        );

        // Check encoding boost
        assert!(
            score.encoding_boost >= 1.0,
            "High importance should boost encoding"
        );
    }

    #[test]
    fn test_importance_score_explanation() {
        let signals = ImportanceSignals::new();
        let context = Context::current();

        let score = signals.compute_importance("Critical error in production system!", &context);

        let explanation = score.explain();
        assert!(!explanation.is_empty(), "Should provide explanation");

        let summary = score.summary();
        assert!(
            summary.contains("Importance"),
            "Summary should contain score"
        );
    }

    #[test]
    fn test_composite_weights() {
        let weights = CompositeWeights::new(1.0, 2.0, 1.0, 1.0);

        assert!(weights.is_valid(), "Weights should sum to 1.0");
        assert!(
            weights.arousal > weights.novelty,
            "Arousal should have higher weight"
        );
    }

    #[test]
    fn test_consolidation_priority() {
        assert!(ConsolidationPriority::Critical > ConsolidationPriority::High);
        assert!(ConsolidationPriority::High > ConsolidationPriority::Normal);
        assert!(ConsolidationPriority::Normal > ConsolidationPriority::Low);

        assert!(ConsolidationPriority::Critical.decay_modifier() < 1.0);
        assert!(ConsolidationPriority::Low.decay_modifier() > 1.0);
    }

    #[test]
    fn test_sentiment_analyzer() {
        let analyzer = SentimentAnalyzer::new();

        let positive = analyzer.analyze("This is amazing and wonderful!");
        assert!(positive.polarity > 0.0, "Should detect positive sentiment");

        let negative = analyzer.analyze("This is terrible and broken.");
        assert!(negative.polarity < 0.0, "Should detect negative sentiment");

        let negated = analyzer.analyze("This is not bad at all.");
        // Negation should flip sentiment
        assert!(negated.polarity >= 0.0, "Negation should flip sentiment");
    }

    #[test]
    fn test_context_builder() {
        let context = Context::current()
            .with_session("session-123")
            .with_project("vestige")
            .with_query("importance signals")
            .with_learning_session(true)
            .with_emotional_context("focused")
            .with_tags(vec!["rust".to_string(), "memory".to_string()]);

        assert_eq!(context.session_id, Some("session-123".to_string()));
        assert_eq!(context.project, Some("vestige".to_string()));
        assert_eq!(context.recent_queries.len(), 1);
        assert!(context.learning_session_active);
    }

    #[test]
    fn test_access_pattern() {
        let mut pattern = AccessPattern::new();

        pattern.add_access("mem-1", 0.0);
        pattern.add_access("mem-2", 5.0);
        pattern.add_access("mem-1", 3.0);
        pattern.add_query("search query");

        assert_eq!(pattern.unique_memories(), 2);
        assert_eq!(pattern.avg_inter_access_time(), 4.0);
    }
}
