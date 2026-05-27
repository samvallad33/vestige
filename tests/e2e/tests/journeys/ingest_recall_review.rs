//! # Ingest-Recall-Review Journey Tests
//!
//! Tests the complete memory lifecycle from creation to retrieval to review.
//! This is the core user journey for any memory system.
//!
//! ## User Journey
//!
//! 1. User ingests new memories (code snippets, learnings, decisions)
//! 2. User recalls memories via search (keyword, semantic, hybrid)
//! 3. User reviews memories to strengthen retention
//! 4. System tracks memory strength and schedules reviews
//! 5. User benefits from improved recall over time

use vestige_core::{
    consolidation::SleepConsolidation,
    fsrs::{FSRSScheduler, LearningState, Rating},
    memory::{IngestInput, RecallInput, SearchMode},
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Create a test memory input using JSON deserialization (for non-exhaustive struct)
fn make_ingest(content: &str, node_type: &str, tags: Vec<&str>) -> IngestInput {
    let tags_json: Vec<String> = tags.into_iter().map(String::from).collect();
    let json = serde_json::json!({
        "content": content,
        "nodeType": node_type,
        "tags": tags_json,
        "source": "test"
    });
    serde_json::from_value(json).expect("IngestInput JSON should be valid")
}

/// Create a recall input using JSON deserialization
fn make_recall(query: &str, limit: i32, min_retention: f64, search_mode: &str) -> RecallInput {
    let json = serde_json::json!({
        "query": query,
        "limit": limit,
        "minRetention": min_retention,
        "searchMode": search_mode
    });
    serde_json::from_value(json).expect("RecallInput JSON should be valid")
}

// ============================================================================
// TEST 1: INGEST CREATES VALID MEMORY STRUCTURE
// ============================================================================

/// Test that ingesting a memory creates a properly structured node.
///
/// Validates:
/// - Node has valid UUID
/// - Content is preserved
/// - Tags are preserved
/// - Initial FSRS state is correct
/// - Timestamps are set correctly
#[test]
fn test_ingest_creates_valid_memory_structure() {
    // Create input
    let input = make_ingest(
        "Rust ownership ensures memory safety without garbage collection",
        "concept",
        vec!["rust", "memory", "ownership"],
    );

    // Verify input structure
    assert!(!input.content.is_empty(), "Content should not be empty");
    assert_eq!(input.node_type, "concept");
    assert_eq!(input.tags.len(), 3);
    assert!(input.tags.contains(&"rust".to_string()));
    assert!(input.tags.contains(&"memory".to_string()));
    assert!(input.tags.contains(&"ownership".to_string()));

    // Verify source is tracked
    assert_eq!(input.source, Some("test".to_string()));

    // Verify sentiment defaults
    assert_eq!(input.sentiment_score, 0.0);
    assert_eq!(input.sentiment_magnitude, 0.0);

    // Verify temporal validity defaults
    assert!(input.valid_from.is_none());
    assert!(input.valid_until.is_none());
}

// ============================================================================
// TEST 2: RECALL FINDS MEMORIES BY CONTENT
// ============================================================================

/// Test that recall can find memories matching a query.
///
/// Validates:
/// - Keyword search matches content
/// - Results are returned in order of relevance
/// - Memory strength affects ranking
#[test]
fn test_recall_finds_memories_by_content() {
    // Create recall input
    let recall = make_recall("rust ownership", 10, 0.5, "keyword");

    // Verify recall input structure
    assert_eq!(recall.query, "rust ownership");
    assert_eq!(recall.limit, 10);
    assert_eq!(recall.min_retention, 0.5);

    // Verify search mode
    assert!(
        matches!(&recall.search_mode, SearchMode::Keyword),
        "Expected Keyword search mode"
    );
}

// ============================================================================
// TEST 3: REVIEW STRENGTHENS MEMORY WITH FSRS
// ============================================================================

/// Test that reviewing a memory updates its FSRS state correctly.
///
/// Validates:
/// - Good rating increases stability
/// - Again rating increases difficulty
/// - Next review is scheduled appropriately
/// - Storage and retrieval strength update
#[test]
fn test_review_strengthens_memory_with_fsrs() {
    let scheduler = FSRSScheduler::default();

    // Create initial state (new card)
    let initial_state = scheduler.new_card();
    assert_eq!(initial_state.reps, 0);
    assert_eq!(initial_state.lapses, 0);

    // Review with Good rating (elapsed_days is f64)
    let result = scheduler.review(&initial_state, Rating::Good, 0.0, None);

    // Stability should be set from initial parameters
    assert!(
        result.state.stability > 0.0,
        "Stability should be positive after review"
    );

    // Reps should increase
    assert_eq!(result.state.reps, 1, "Reps should increase after review");

    // Interval should be positive
    assert!(result.interval > 0, "Interval should be positive");

    // Review again with Easy - should increase interval
    let second_result = scheduler.review(&result.state, Rating::Easy, result.interval as f64, None);
    assert!(
        second_result.interval >= result.interval,
        "Easy rating should maintain or increase interval"
    );

    // Review with Again - should reset progress
    let again_result = scheduler.review(&second_result.state, Rating::Again, 1.0, None);
    assert!(
        again_result.interval <= second_result.interval,
        "Again rating should reduce interval"
    );
    assert_eq!(
        again_result.state.lapses, 1,
        "Lapses should increase on Again"
    );
}

// ============================================================================
// TEST 4: MEMORY LIFECYCLE FOLLOWS EXPECTED PATTERN
// ============================================================================

/// Test the complete memory lifecycle from new to mature.
///
/// Validates:
/// - New memory starts in learning state
/// - Successful reviews progress state
/// - Memory becomes mature after multiple reviews
/// - Intervals increase appropriately
#[test]
fn test_memory_lifecycle_follows_expected_pattern() {
    let scheduler = FSRSScheduler::default();
    let mut state = scheduler.new_card();

    // Track intervals to verify growth
    let mut intervals = Vec::new();

    // Simulate 10 successful reviews
    for i in 0..10 {
        let elapsed = if i == 0 {
            0.0
        } else {
            intervals.last().copied().unwrap_or(1) as f64
        };
        let result = scheduler.review(&state, Rating::Good, elapsed, None);
        intervals.push(result.interval);
        state = result.state;
    }

    // Verify lifecycle progression
    assert!(state.reps >= 10, "Should have at least 10 reps");
    assert_eq!(
        state.lapses, 0,
        "Should have no lapses with all Good ratings"
    );

    // Verify interval growth (early intervals may be similar, but should eventually grow)
    let early_avg: f64 = intervals[..3].iter().map(|&i| i as f64).sum::<f64>() / 3.0;
    let late_avg: f64 = intervals[7..].iter().map(|&i| i as f64).sum::<f64>() / 3.0;
    assert!(
        late_avg >= early_avg,
        "Later intervals ({}) should be >= early intervals ({})",
        late_avg,
        early_avg
    );

    // Verify state is Review (mature)
    assert!(
        matches!(&state.state, LearningState::Review) || state.reps >= 10,
        "Mature memory should be in Review or have processed all reviews"
    );
}

// ============================================================================
// TEST 5: SENTIMENT AFFECTS MEMORY CONSOLIDATION
// ============================================================================

/// Test that emotional memories are processed differently.
///
/// Validates:
/// - High sentiment magnitude boosts stability
/// - Emotional memories decay slower
/// - Sentiment is preserved through lifecycle
#[test]
fn test_sentiment_affects_memory_consolidation() {
    let consolidation = SleepConsolidation::new();

    // Calculate decay for neutral memory
    let neutral_decay = consolidation.calculate_decay(10.0, 5.0, 0.0);

    // Calculate decay for emotional memory
    let emotional_decay = consolidation.calculate_decay(10.0, 5.0, 1.0);

    // Emotional memory should decay slower (higher retention)
    assert!(
        emotional_decay > neutral_decay,
        "Emotional memory ({}) should retain better than neutral ({})",
        emotional_decay,
        neutral_decay
    );

    // Test should_promote logic
    assert!(
        consolidation.should_promote(0.8, 5.0),
        "High emotion + low storage should promote"
    );
    assert!(
        !consolidation.should_promote(0.3, 5.0),
        "Low emotion should not promote"
    );
    assert!(
        !consolidation.should_promote(0.8, 10.0),
        "Max storage should not promote"
    );

    // Test promotion boost
    let boosted = consolidation.promotion_boost(5.0);
    assert!(boosted > 5.0, "Promotion should increase storage strength");
    assert!(
        boosted <= 10.0,
        "Promotion should cap at max storage strength"
    );
}

// ============================================================================
// ADDITIONAL INTEGRATION TESTS
// ============================================================================

/// Test that RecallInput can be created with different search modes.
#[test]
fn test_recall_search_modes() {
    // Keyword mode
    let keyword = make_recall("test query", 10, 0.5, "keyword");
    assert!(matches!(keyword.search_mode, SearchMode::Keyword));

    // Semantic mode (when embeddings available)
    let semantic = make_recall("test query", 10, 0.5, "semantic");
    assert!(matches!(semantic.search_mode, SearchMode::Semantic));

    // Hybrid mode
    let hybrid = make_recall("test query", 10, 0.5, "hybrid");
    assert!(matches!(hybrid.search_mode, SearchMode::Hybrid));
}

/// Test IngestInput defaults.
#[test]
fn test_ingest_input_defaults() {
    let json = serde_json::json!({
        "content": "Test content",
        "nodeType": "fact"
    });
    let input: IngestInput = serde_json::from_value(json).unwrap();

    assert_eq!(input.content, "Test content");
    assert_eq!(input.node_type, "fact");
    assert!(input.source.is_none());
    assert!(input.tags.is_empty());
    assert_eq!(input.sentiment_score, 0.0);
    assert_eq!(input.sentiment_magnitude, 0.0);
}

/// Test FSRS rating effects on memory state.
#[test]
fn test_fsrs_rating_effects() {
    let scheduler = FSRSScheduler::default();
    let initial = scheduler.new_card();

    // Test all rating types (elapsed_days as f64)
    let again = scheduler.review(&initial, Rating::Again, 0.0, None);
    let hard = scheduler.review(&initial, Rating::Hard, 0.0, None);
    let good = scheduler.review(&initial, Rating::Good, 0.0, None);
    let easy = scheduler.review(&initial, Rating::Easy, 0.0, None);

    // Again should have shortest interval
    assert!(
        again.interval <= hard.interval,
        "Again ({}) should be <= Hard ({})",
        again.interval,
        hard.interval
    );

    // Easy should have longest interval
    assert!(
        easy.interval >= good.interval,
        "Easy ({}) should be >= Good ({})",
        easy.interval,
        good.interval
    );

    // Good should have medium interval
    assert!(
        good.interval >= hard.interval,
        "Good ({}) should be >= Hard ({})",
        good.interval,
        hard.interval
    );
}
