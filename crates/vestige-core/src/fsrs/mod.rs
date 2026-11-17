//! FSRS-6 (Free Spaced Repetition Scheduler) Module
//!
//! The state-of-the-art spaced repetition algorithm (2025-2026).
//! 20-30% more efficient than SM-2 (Anki's original algorithm).
//!
//! Reference: https://github.com/open-spaced-repetition/fsrs4anki
//!
//! ## Key improvements in FSRS-6 over FSRS-5:
//! - 21 parameters (vs 19) with personalizable forgetting curve decay (w20)
//! - Same-day review handling with S^(-w19) term
//! - Better short-term memory modeling
//!
//! ## Core Formulas:
//! - Retrievability: R = (1 + FACTOR * t / S)^(-w20) where FACTOR = 0.9^(-1/w20) - 1
//! - Interval: t = S/FACTOR * (R^(1/w20) - 1)

mod algorithm;
mod scheduler;

pub use algorithm::{
    DEFAULT_DECAY,
    DEFAULT_RETENTION,
    // Constants
    FSRS6_WEIGHTS,
    MAX_DIFFICULTY,
    MAX_STABILITY,
    MIN_DIFFICULTY,
    MIN_STABILITY,
    apply_exposure_dividend,
    apply_sentiment_boost,
    apply_supersession_decay_floor,
    fuzz_interval,
    initial_difficulty,
    initial_difficulty_with_weights,
    initial_stability,
    initial_stability_with_weights,
    lock_arousal_from_severity,
    modulate_stability_by_effort,
    next_difficulty,
    next_difficulty_with_weights,
    next_forget_stability,
    next_forget_stability_with_weights,
    next_interval,
    next_interval_with_decay,
    next_recall_stability,
    next_recall_stability_with_weights,
    // Core functions
    retrievability,
    retrievability_with_decay,
    same_day_stability,
    same_day_stability_with_weights,
};

pub use scheduler::{
    FSRSParameters, FSRSScheduler, FSRSState, LearningState, PreviewResults, Rating, ReviewResult,
};
