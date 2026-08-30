//! FSRS-6 Parameter Optimizer
//!
//! Personalizes FSRS parameters based on user review history.
//! Uses gradient-free optimization to minimize prediction error.

use super::algorithm::{FSRS6_WEIGHTS, retrievability_with_decay};
use chrono::{DateTime, Utc};

/// Lowest decay exponent the optimizer may fit. Roughly half the FSRS-6
/// default of 0.1542; below this the forgetting curve is flat enough that a
/// year of neglect barely moves retention, which disables every downstream
/// consumer of decay (accessibility states, hygiene stats, forgetting).
pub const MIN_DECAY_BOUND: f64 = 0.08;

/// Failed-recall events (rating 1) required before a decay fit is trusted.
pub const MIN_FORGETTING_EVIDENCE: usize = 5;

// ============================================================================
// REVIEW LOG
// ============================================================================

/// A single review event for optimization
#[derive(Debug, Clone)]
pub struct ReviewLog {
    /// Review timestamp
    pub timestamp: DateTime<Utc>,
    /// Rating given (1-4)
    pub rating: i32,
    /// Stability at time of review
    pub stability: f64,
    /// Difficulty at time of review
    pub difficulty: f64,
    /// Days since last review
    pub elapsed_days: f64,
}

// ============================================================================
// OPTIMIZER
// ============================================================================

/// FSRS parameter optimizer
///
/// Personalizes the 21 FSRS-6 parameters based on user review history.
/// Uses the RMSE (Root Mean Square Error) of retrievability predictions
/// as the loss function.
pub struct FSRSOptimizer {
    /// Current weights being optimized
    weights: [f64; 21],
    /// Review history for training
    reviews: Vec<ReviewLog>,
    /// Minimum reviews required for optimization
    min_reviews: usize,
}

impl Default for FSRSOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl FSRSOptimizer {
    /// Create a new optimizer with default weights
    pub fn new() -> Self {
        Self {
            weights: FSRS6_WEIGHTS,
            reviews: Vec::new(),
            min_reviews: 100,
        }
    }

    /// Add a review to the training history
    pub fn add_review(&mut self, review: ReviewLog) {
        self.reviews.push(review);
    }

    /// Add multiple reviews
    pub fn add_reviews(&mut self, reviews: impl IntoIterator<Item = ReviewLog>) {
        self.reviews.extend(reviews);
    }

    /// Get current weights
    pub fn weights(&self) -> &[f64; 21] {
        &self.weights
    }

    /// Check if enough reviews for optimization
    pub fn has_enough_data(&self) -> bool {
        self.reviews.len() >= self.min_reviews
    }

    /// Does the history contain enough FORGETTING evidence to fit a decay
    /// curve at all?
    ///
    /// The loss treats rating 1 as "forgot" and everything else as
    /// "remembered". A history with no failures makes "nothing is ever
    /// forgotten" the perfect fit, and the golden-section search rides w20
    /// straight into its lower bound. That is not a hypothetical: an agent
    /// memory store's access log is success-dominated by construction, and a
    /// real 2,929-memory store was measured running with w20 = 0.0104 for a
    /// month — flat enough that 217 days of neglect moved retention by less
    /// than 0.09, leaving the Silent and Unavailable accessibility states
    /// unreachable. Success-only data is insufficient evidence, not evidence
    /// of immortal memory.
    pub fn has_forgetting_evidence(&self) -> bool {
        self.reviews.iter().filter(|r| r.rating == 1).count() >= MIN_FORGETTING_EVIDENCE
    }

    /// Get the number of reviews in history
    pub fn review_count(&self) -> usize {
        self.reviews.len()
    }

    /// Calculate RMSE loss for current weights
    pub fn calculate_loss(&self) -> f64 {
        if self.reviews.is_empty() {
            return 0.0;
        }

        let w20 = self.weights[20];
        let mut sum_squared_error = 0.0;

        for review in &self.reviews {
            // Calculate predicted retrievability
            let predicted_r = retrievability_with_decay(review.stability, review.elapsed_days, w20);

            // Convert rating to binary outcome (Again = 0, others = 1)
            let actual = if review.rating == 1 { 0.0 } else { 1.0 };

            let error = predicted_r - actual;
            sum_squared_error += error * error;
        }

        (sum_squared_error / self.reviews.len() as f64).sqrt()
    }

    /// Optimize the forgetting curve decay parameter (w20)
    ///
    /// This is the most personalizable parameter in FSRS-6.
    /// Uses golden section search for 1D optimization.
    pub fn optimize_decay(&mut self) -> f64 {
        if !self.has_enough_data() {
            return self.weights[20];
        }
        if !self.has_forgetting_evidence() {
            // Refuse to fit rather than fit a degenerate curve; the caller
            // keeps (or is restored to) the current default.
            return self.weights[20];
        }

        // Lower bound is decay-meaningful, roughly half the FSRS-6 default
        // (0.1542). The previous bound of 0.01 was a numeric convenience the
        // degenerate fit slammed into; no human forgetting curve is that flat.
        let (mut a, mut b) = (MIN_DECAY_BOUND, 1.0);
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;

        let mut x1 = b - (b - a) / phi;
        let mut x2 = a + (b - a) / phi;

        let mut f1 = self.loss_at_decay(x1);
        let mut f2 = self.loss_at_decay(x2);

        // Golden section iterations
        for _ in 0..50 {
            if f1 < f2 {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = b - (b - a) / phi;
                f1 = self.loss_at_decay(x1);
            } else {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = a + (b - a) / phi;
                f2 = self.loss_at_decay(x2);
            }

            if (b - a).abs() < 0.001 {
                break;
            }
        }

        let optimal_decay = (a + b) / 2.0;
        self.weights[20] = optimal_decay;
        optimal_decay
    }

    /// Calculate loss at a specific decay value
    fn loss_at_decay(&self, decay: f64) -> f64 {
        if self.reviews.is_empty() {
            return 0.0;
        }

        let mut sum_squared_error = 0.0;

        for review in &self.reviews {
            let predicted_r =
                retrievability_with_decay(review.stability, review.elapsed_days, decay);

            let actual = if review.rating == 1 { 0.0 } else { 1.0 };
            let error = predicted_r - actual;
            sum_squared_error += error * error;
        }

        (sum_squared_error / self.reviews.len() as f64).sqrt()
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.weights = FSRS6_WEIGHTS;
        self.reviews.clear();
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn create_test_reviews(count: usize) -> Vec<ReviewLog> {
        let now = Utc::now();
        (0..count)
            .map(|i| ReviewLog {
                timestamp: now - Duration::days(i as i64),
                rating: if i % 5 == 0 { 1 } else { 3 },
                stability: 5.0 + (i as f64 * 0.1),
                difficulty: 5.0,
                elapsed_days: 1.0 + (i as f64 * 0.5),
            })
            .collect()
    }

    #[test]
    fn test_optimizer_creation() {
        let optimizer = FSRSOptimizer::new();
        assert_eq!(optimizer.weights().len(), 21);
        assert!(!optimizer.has_enough_data());
    }

    #[test]
    fn test_add_reviews() {
        let mut optimizer = FSRSOptimizer::new();
        let reviews = create_test_reviews(50);

        optimizer.add_reviews(reviews);
        assert_eq!(optimizer.review_count(), 50);
        assert!(!optimizer.has_enough_data()); // Need 100
    }

    #[test]
    fn test_calculate_loss() {
        let mut optimizer = FSRSOptimizer::new();
        let reviews = create_test_reviews(100);
        optimizer.add_reviews(reviews);

        let loss = optimizer.calculate_loss();
        assert!(loss >= 0.0);
        assert!(loss <= 1.0);
    }

    #[test]
    fn test_optimize_decay() {
        let mut optimizer = FSRSOptimizer::new();
        let reviews = create_test_reviews(200);
        optimizer.add_reviews(reviews);

        let original_decay = optimizer.weights()[20];
        let optimized_decay = optimizer.optimize_decay();

        // Decay should be a reasonable value
        assert!(optimized_decay > 0.01);
        assert!(optimized_decay < 1.0);

        // Optimization should have changed the value
        assert_ne!(original_decay, optimized_decay);
    }

    /// REGRESSION (v2.6.0): a success-only history must not produce a fit.
    /// The measured failure: a real store's access log fed the optimizer
    /// success-dominated data, the fit collapsed into the old 0.01 lower
    /// bound, and store-wide decay silently stopped for a month.
    #[test]
    fn success_only_history_refuses_to_fit() {
        let mut optimizer = FSRSOptimizer::new();
        let now = Utc::now();
        optimizer.add_reviews((0..200).map(|i| ReviewLog {
            timestamp: now - Duration::days(i as i64),
            rating: 3, // remembered, every single time
            stability: 5.0,
            difficulty: 5.0,
            elapsed_days: 1.0 + (i as f64 * 0.5),
        }));
        assert!(optimizer.has_enough_data());
        assert!(!optimizer.has_forgetting_evidence());

        let default_decay = optimizer.weights()[20];
        let result = optimizer.optimize_decay();
        assert_eq!(
            result, default_decay,
            "with no forgetting evidence the fit must be refused, not degenerate"
        );
    }

    /// Even with forgetting evidence, the fit may never go below the
    /// decay-meaningful floor.
    #[test]
    fn fit_respects_the_decay_floor() {
        let mut optimizer = FSRSOptimizer::new();
        let reviews = create_test_reviews(200);
        optimizer.add_reviews(reviews);
        assert!(optimizer.has_forgetting_evidence());

        let optimized = optimizer.optimize_decay();
        assert!(
            optimized >= MIN_DECAY_BOUND,
            "fit {optimized} fell below the floor {MIN_DECAY_BOUND}"
        );
        assert!(optimized < 1.0);
    }

    #[test]
    fn test_reset() {
        let mut optimizer = FSRSOptimizer::new();
        let reviews = create_test_reviews(100);
        optimizer.add_reviews(reviews);

        optimizer.reset();
        assert_eq!(optimizer.review_count(), 0);
        assert_eq!(optimizer.weights()[20], FSRS6_WEIGHTS[20]);
    }
}
