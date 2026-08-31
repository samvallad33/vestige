//! FSRS-6 Scheduler
//!
//! High-level scheduler that manages review state and produces
//! optimal scheduling decisions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::algorithm::{
    DEFAULT_RETENTION, FSRS6_WEIGHTS, MAX_STABILITY, apply_sentiment_boost, fuzz_interval,
    initial_difficulty_with_weights, initial_stability_with_weights, next_difficulty_with_weights,
    next_forget_stability_with_weights, next_interval_with_decay,
    next_recall_stability_with_weights, retrievability_with_decay, same_day_stability_with_weights,
};

// ============================================================================
// TYPES
// ============================================================================

/// Review ratings (1-4 scale)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rating {
    /// Complete failure to recall
    Again = 1,
    /// Recalled with significant difficulty
    Hard = 2,
    /// Recalled with some effort
    Good = 3,
    /// Instant, effortless recall
    Easy = 4,
}

impl Rating {
    /// Convert to i32
    pub fn as_i32(&self) -> i32 {
        *self as i32
    }

    /// Create from i32
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            1 => Some(Rating::Again),
            2 => Some(Rating::Hard),
            3 => Some(Rating::Good),
            4 => Some(Rating::Easy),
            _ => None,
        }
    }

    /// Get 0-indexed position (for accessing weights array)
    pub fn as_index(&self) -> usize {
        (*self as usize) - 1
    }
}

/// Learning states in the FSRS state machine
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LearningState {
    /// Never reviewed
    #[default]
    New,
    /// In initial learning phase
    Learning,
    /// Graduated to review phase
    Review,
    /// Failed review, relearning
    Relearning,
}

/// FSRS-6 card state
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FSRSState {
    /// Memory difficulty (1.0 to 10.0)
    pub difficulty: f64,
    /// Memory stability in days
    pub stability: f64,
    /// Current learning state
    pub state: LearningState,
    /// Number of successful reviews
    pub reps: i32,
    /// Number of lapses
    pub lapses: i32,
    /// Last review timestamp
    pub last_review: DateTime<Utc>,
    /// Days until next review
    pub scheduled_days: i32,
}

impl Default for FSRSState {
    fn default() -> Self {
        Self {
            difficulty: super::algorithm::initial_difficulty(Rating::Good),
            stability: super::algorithm::initial_stability(Rating::Good),
            state: LearningState::New,
            reps: 0,
            lapses: 0,
            last_review: Utc::now(),
            scheduled_days: 0,
        }
    }
}

/// Result of a review operation
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReviewResult {
    /// Updated state after review
    pub state: FSRSState,
    /// Current retrievability before review
    pub retrievability: f64,
    /// Scheduled interval in days
    pub interval: i32,
    /// Whether this was a lapse (forgotten after learning)
    pub is_lapse: bool,
}

/// Preview results for all grades
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewResults {
    /// Result if rated Again
    pub again: ReviewResult,
    /// Result if rated Hard
    pub hard: ReviewResult,
    /// Result if rated Good
    pub good: ReviewResult,
    /// Result if rated Easy
    pub easy: ReviewResult,
}

/// User-personalizable FSRS parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FSRSParameters {
    /// FSRS-6 weights (21 parameters)
    pub weights: [f64; 21],
    /// Target retention rate (default 0.9)
    pub desired_retention: f64,
    /// Maximum interval in days
    pub max_interval: i32,
    /// Enable interval fuzzing
    pub enable_fuzz: bool,
}

impl Default for FSRSParameters {
    fn default() -> Self {
        Self {
            weights: FSRS6_WEIGHTS,
            desired_retention: DEFAULT_RETENTION,
            max_interval: MAX_STABILITY as i32,
            enable_fuzz: true,
        }
    }
}

// ============================================================================
// SCHEDULER
// ============================================================================

/// FSRS-6 Scheduler
///
/// Manages spaced repetition scheduling using the FSRS-6 algorithm.
pub struct FSRSScheduler {
    params: FSRSParameters,
    enable_sentiment_boost: bool,
    max_sentiment_boost: f64,
}

impl Default for FSRSScheduler {
    fn default() -> Self {
        Self {
            params: FSRSParameters::default(),
            enable_sentiment_boost: true,
            max_sentiment_boost: 2.0,
        }
    }
}

impl FSRSScheduler {
    /// Create a new scheduler with custom parameters
    pub fn new(params: FSRSParameters) -> Self {
        Self {
            params,
            enable_sentiment_boost: true,
            max_sentiment_boost: 2.0,
        }
    }

    /// Configure sentiment boost settings
    pub fn with_sentiment_boost(mut self, enable: bool, max_boost: f64) -> Self {
        self.enable_sentiment_boost = enable;
        self.max_sentiment_boost = max_boost;
        self
    }

    /// Create a new card in the initial state
    pub fn new_card(&self) -> FSRSState {
        FSRSState::default()
    }

    /// Process a review and return the updated state
    ///
    /// # Arguments
    /// * `state` - Current card state
    /// * `grade` - User's rating of the review
    /// * `elapsed_days` - Days since last review
    /// * `sentiment_boost` - Optional sentiment intensity for emotional memories
    pub fn review(
        &self,
        state: &FSRSState,
        grade: Rating,
        elapsed_days: f64,
        sentiment_boost: Option<f64>,
    ) -> ReviewResult {
        let w20 = self.params.weights[20];

        let r = if state.state == LearningState::New {
            1.0
        } else {
            retrievability_with_decay(state.stability, elapsed_days.max(0.0), w20)
        };

        // Check if this is a same-day review (less than 1 day elapsed)
        let is_same_day = elapsed_days < 1.0 && state.state != LearningState::New;

        let (mut new_state, is_lapse) = if state.state == LearningState::New {
            (self.handle_first_review(state, grade), false)
        } else if is_same_day {
            (self.handle_same_day_review(state, grade), false)
        } else if grade == Rating::Again {
            let is_lapse =
                state.state == LearningState::Review || state.state == LearningState::Relearning;
            (self.handle_lapse(state, r), is_lapse)
        } else {
            (self.handle_recall(state, grade, r), false)
        };

        // Apply sentiment boost. Clamped: the boost multiplies the PERSISTED
        // stability on every review, and unclamped compounding was measured
        // driving real memories to stability 1.4e24 days (issue #121) —
        // beyond any horizon the forgetting curve can express.
        if self.enable_sentiment_boost
            && let Some(sentiment) = sentiment_boost
            && sentiment > 0.0
        {
            new_state.stability =
                apply_sentiment_boost(new_state.stability, sentiment, self.max_sentiment_boost)
                    .min(MAX_STABILITY);
        }

        let mut interval =
            next_interval_with_decay(new_state.stability, self.params.desired_retention, w20)
                .min(self.params.max_interval);

        // Apply fuzzing
        if self.params.enable_fuzz && interval > 2 {
            let seed = state.last_review.timestamp() as u64;
            interval = fuzz_interval(interval, seed);
        }

        new_state.scheduled_days = interval;
        new_state.last_review = Utc::now();

        ReviewResult {
            state: new_state,
            retrievability: r,
            interval,
            is_lapse,
        }
    }

    fn handle_first_review(&self, state: &FSRSState, grade: Rating) -> FSRSState {
        let weights = &self.params.weights;
        let d = initial_difficulty_with_weights(grade, weights);
        let s = initial_stability_with_weights(grade, weights);

        let new_state = match grade {
            Rating::Again | Rating::Hard => LearningState::Learning,
            _ => LearningState::Review,
        };

        FSRSState {
            difficulty: d,
            stability: s,
            state: new_state,
            reps: 1,
            lapses: if grade == Rating::Again { 1 } else { 0 },
            last_review: state.last_review,
            scheduled_days: state.scheduled_days,
        }
    }

    fn handle_same_day_review(&self, state: &FSRSState, grade: Rating) -> FSRSState {
        let weights = &self.params.weights;
        let new_s = same_day_stability_with_weights(state.stability, grade, weights);
        let new_d = next_difficulty_with_weights(state.difficulty, grade, weights);

        FSRSState {
            difficulty: new_d,
            stability: new_s,
            state: state.state,
            reps: state.reps + 1,
            lapses: state.lapses,
            last_review: state.last_review,
            scheduled_days: state.scheduled_days,
        }
    }

    fn handle_lapse(&self, state: &FSRSState, r: f64) -> FSRSState {
        let weights = &self.params.weights;
        let new_s =
            next_forget_stability_with_weights(state.difficulty, state.stability, r, weights);
        let new_d = next_difficulty_with_weights(state.difficulty, Rating::Again, weights);

        FSRSState {
            difficulty: new_d,
            stability: new_s,
            state: LearningState::Relearning,
            reps: state.reps + 1,
            lapses: state.lapses + 1,
            last_review: state.last_review,
            scheduled_days: state.scheduled_days,
        }
    }

    fn handle_recall(&self, state: &FSRSState, grade: Rating, r: f64) -> FSRSState {
        let weights = &self.params.weights;
        let new_s = next_recall_stability_with_weights(
            state.stability,
            state.difficulty,
            r,
            grade,
            weights,
        );
        let new_d = next_difficulty_with_weights(state.difficulty, grade, weights);

        FSRSState {
            difficulty: new_d,
            stability: new_s,
            state: LearningState::Review,
            reps: state.reps + 1,
            lapses: state.lapses,
            last_review: state.last_review,
            scheduled_days: state.scheduled_days,
        }
    }

    /// Preview what would happen for each rating
    pub fn preview_reviews(&self, state: &FSRSState, elapsed_days: f64) -> PreviewResults {
        PreviewResults {
            again: self.review(state, Rating::Again, elapsed_days, None),
            hard: self.review(state, Rating::Hard, elapsed_days, None),
            good: self.review(state, Rating::Good, elapsed_days, None),
            easy: self.review(state, Rating::Easy, elapsed_days, None),
        }
    }

    /// Calculate days since last review
    pub fn days_since_review(&self, last_review: &DateTime<Utc>) -> f64 {
        let now = Utc::now();
        let diff = now.signed_duration_since(*last_review);
        (diff.num_seconds() as f64 / 86400.0).max(0.0)
    }

    /// Get the personalized forgetting curve decay parameter
    pub fn get_decay(&self) -> f64 {
        self.params.weights[20]
    }

    /// Update weights for personalization (after training on user data)
    pub fn set_weights(&mut self, weights: [f64; 21]) {
        self.params.weights = weights;
    }

    /// Get current parameters
    pub fn params(&self) -> &FSRSParameters {
        &self.params
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_first_review() {
        let scheduler = FSRSScheduler::default();
        let card = scheduler.new_card();

        let result = scheduler.review(&card, Rating::Good, 0.0, None);

        assert_eq!(result.state.reps, 1);
        assert_eq!(result.state.lapses, 0);
        assert_eq!(result.state.state, LearningState::Review);
        assert!(result.interval > 0);
    }

    #[test]
    fn test_scheduler_lapse_tracking() {
        let scheduler = FSRSScheduler::default();
        let mut card = scheduler.new_card();

        let result = scheduler.review(&card, Rating::Good, 0.0, None);
        card = result.state;
        assert_eq!(card.lapses, 0);

        let result = scheduler.review(&card, Rating::Again, 1.0, None);
        assert!(result.is_lapse);
        assert_eq!(result.state.lapses, 1);
        assert_eq!(result.state.state, LearningState::Relearning);
    }

    #[test]
    fn test_scheduler_same_day_review() {
        let scheduler = FSRSScheduler::default();
        let mut card = scheduler.new_card();

        // First review
        let result = scheduler.review(&card, Rating::Good, 0.0, None);
        card = result.state;
        let initial_stability = card.stability;

        // Same-day review (0.5 days later)
        let result = scheduler.review(&card, Rating::Good, 0.5, None);

        // Should use same-day formula, not regular recall
        assert!(result.state.stability != initial_stability);
    }

    #[test]
    fn test_custom_parameters() {
        let params = FSRSParameters {
            desired_retention: 0.85,
            enable_fuzz: false,
            ..FSRSParameters::default()
        };

        let scheduler = FSRSScheduler::new(params);
        let card = scheduler.new_card();
        let result = scheduler.review(&card, Rating::Good, 0.0, None);

        // Lower retention = longer intervals
        let default_scheduler = FSRSScheduler::default();
        let default_result = default_scheduler.review(&card, Rating::Good, 0.0, None);

        assert!(result.interval > default_result.interval);
    }

    #[test]
    fn test_rating_conversion() {
        assert_eq!(Rating::Again.as_i32(), 1);
        assert_eq!(Rating::Hard.as_i32(), 2);
        assert_eq!(Rating::Good.as_i32(), 3);
        assert_eq!(Rating::Easy.as_i32(), 4);

        assert_eq!(Rating::from_i32(1), Some(Rating::Again));
        assert_eq!(Rating::from_i32(5), None);
    }

    #[test]
    fn test_preview_reviews() {
        let scheduler = FSRSScheduler::default();
        let card = scheduler.new_card();

        let preview = scheduler.preview_reviews(&card, 0.0);

        // Again should have shortest interval
        assert!(preview.again.interval < preview.good.interval);
        // Easy should have longest interval
        assert!(preview.easy.interval > preview.good.interval);
    }
}
