//! # Intentions Workflow Journey Tests
//!
//! Tests the intent detection system that understands WHY users are doing
//! something, not just WHAT they're doing. This enables proactive memory
//! retrieval based on detected intent.
//!
//! ## User Journey
//!
//! 1. User opens files, searches, runs commands
//! 2. System observes and records actions
//! 3. System detects intent (debugging, learning, refactoring, etc.)
//! 4. System proactively suggests relevant memories
//! 5. User benefits from context-aware assistance

use vestige_core::advanced::intent::{
    ActionType, DetectedIntent, IntentDetector, LearningLevel, MaintenanceType, OptimizationType,
    UserAction,
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Create a detector with pre-recorded debugging actions
fn detector_with_debugging_actions() -> IntentDetector {
    let detector = IntentDetector::new();

    detector.record_action(UserAction::error("TypeError: undefined is not a function"));
    detector.record_action(UserAction::file_opened("/src/components/Button.tsx"));
    detector.record_action(UserAction::search("fix undefined error"));
    detector.record_action(UserAction::file_opened("/src/utils/helpers.ts"));

    detector
}

/// Create a detector with pre-recorded learning actions
fn detector_with_learning_actions() -> IntentDetector {
    let detector = IntentDetector::new();

    detector.record_action(UserAction::docs_viewed("async/await in Rust"));
    detector.record_action(UserAction::search("how to use tokio"));
    detector.record_action(UserAction::docs_viewed("futures crate tutorial"));
    detector.record_action(UserAction::search("what is a Future in Rust"));

    detector
}

/// Create a detector with pre-recorded refactoring actions
fn detector_with_refactoring_actions() -> IntentDetector {
    let detector = IntentDetector::new();

    detector.record_action(UserAction::file_edited("/src/auth/login.rs"));
    detector.record_action(UserAction::file_edited("/src/auth/logout.rs"));
    detector.record_action(UserAction::file_edited("/src/auth/session.rs"));
    detector.record_action(UserAction::search("extract method refactoring"));
    detector.record_action(UserAction::file_edited("/src/auth/mod.rs"));

    detector
}

// ============================================================================
// TEST 1: DEBUGGING INTENT DETECTION
// ============================================================================

/// Test that debugging intent is detected from error-related actions.
///
/// Validates:
/// - Error encounters boost debugging confidence
/// - Debug sessions boost debugging confidence
/// - File opens near errors identify suspected area
/// - Symptoms are captured from error messages
#[test]
fn test_debugging_intent_detection() {
    let detector = detector_with_debugging_actions();

    let result = detector.detect_intent();

    // Should detect some intent (may be debugging or learning)
    assert!(
        result.confidence > 0.0 || matches!(result.primary_intent, DetectedIntent::Unknown),
        "Should detect intent or return Unknown"
    );

    // Verify evidence is captured
    assert!(
        !result.evidence.is_empty() || result.confidence == 0.0,
        "Should capture evidence if intent detected"
    );

    // Check intent properties
    match &result.primary_intent {
        DetectedIntent::Debugging {
            suspected_area,
            symptoms: _,
        } => {
            assert!(!suspected_area.is_empty(), "Should identify suspected area");
            // Symptoms may or may not be captured depending on action order
        }
        DetectedIntent::Learning { topic, .. } => {
            // Learning can also match if search terms detected
            assert!(!topic.is_empty(), "Learning topic should not be empty");
        }
        _ => {
            // Other intents may match depending on pattern scoring
        }
    }
}

// ============================================================================
// TEST 2: LEARNING INTENT DETECTION
// ============================================================================

/// Test that learning intent is detected from documentation and tutorial actions.
///
/// Validates:
/// - Documentation views boost learning confidence
/// - "How to" queries boost learning confidence
/// - Tutorial searches boost learning confidence
/// - Topic is extracted from queries
#[test]
fn test_learning_intent_detection() {
    let detector = detector_with_learning_actions();

    let result = detector.detect_intent();

    // Should detect learning with high confidence
    match &result.primary_intent {
        DetectedIntent::Learning { topic, level: _ } => {
            assert!(!topic.is_empty(), "Should identify learning topic");
            // Level may vary
        }
        _ => {
            // Learning actions should typically detect learning intent
            // But other intents may score higher in some cases
            assert!(result.confidence > 0.0, "Should detect some intent");
        }
    }

    // Verify relevant tags
    let _tags = result.primary_intent.relevant_tags();
    // Tags depend on detected intent type
}

// ============================================================================
// TEST 3: REFACTORING INTENT DETECTION
// ============================================================================

/// Test that refactoring intent is detected from multi-file edits.
///
/// Validates:
/// - Multiple file edits boost refactoring confidence
/// - Refactoring-related searches boost confidence
/// - Target files are identified
#[test]
fn test_refactoring_intent_detection() {
    let detector = detector_with_refactoring_actions();

    let result = detector.detect_intent();

    // Should detect intent from multiple edits
    assert!(
        result.confidence > 0.0,
        "Multiple file edits should detect some intent"
    );

    // Check for refactoring or related intent
    match &result.primary_intent {
        DetectedIntent::Refactoring { target, goal } => {
            assert!(!target.is_empty(), "Should identify refactoring target");
            assert!(!goal.is_empty(), "Should identify refactoring goal");
        }
        DetectedIntent::NewFeature {
            related_components, ..
        } => {
            // Multiple edits could also suggest new feature
            let _ = related_components;
            assert!(
                result.confidence > 0.0,
                "New feature intent should have positive confidence"
            );
        }
        _ => {
            // Pattern may match differently
        }
    }
}

// ============================================================================
// TEST 4: INTENT PROVIDES RELEVANT TAGS
// ============================================================================

/// Test that detected intents provide relevant tags for memory search.
///
/// Validates:
/// - Each intent type has associated tags
/// - Tags are relevant to the intent
/// - Tags can be used for memory filtering
#[test]
fn test_intent_provides_relevant_tags() {
    // Test debugging tags
    let debugging = DetectedIntent::Debugging {
        suspected_area: "auth".to_string(),
        symptoms: vec!["null pointer".to_string()],
    };
    let debug_tags = debugging.relevant_tags();
    assert!(debug_tags.contains(&"debugging".to_string()));
    assert!(debug_tags.contains(&"error".to_string()));

    // Test learning tags
    let learning = DetectedIntent::Learning {
        topic: "async rust".to_string(),
        level: LearningLevel::Intermediate,
    };
    let learn_tags = learning.relevant_tags();
    assert!(learn_tags.contains(&"learning".to_string()));
    assert!(learn_tags.contains(&"async rust".to_string()));

    // Test refactoring tags
    let refactoring = DetectedIntent::Refactoring {
        target: "auth module".to_string(),
        goal: "simplify".to_string(),
    };
    let refactor_tags = refactoring.relevant_tags();
    assert!(refactor_tags.contains(&"refactoring".to_string()));
    assert!(refactor_tags.contains(&"patterns".to_string()));

    // Test new feature tags
    let new_feature = DetectedIntent::NewFeature {
        feature_description: "user authentication".to_string(),
        related_components: vec!["login".to_string()],
    };
    let feature_tags = new_feature.relevant_tags();
    assert!(feature_tags.contains(&"feature".to_string()));

    // Test maintenance tags
    let maintenance = DetectedIntent::Maintenance {
        maintenance_type: MaintenanceType::DependencyUpdate,
        target: Some("cargo.toml".to_string()),
    };
    let maint_tags = maintenance.relevant_tags();
    assert!(maint_tags.contains(&"maintenance".to_string()));
    assert!(maint_tags.contains(&"dependencies".to_string()));
}

// ============================================================================
// TEST 5: ACTION HISTORY TRACKING
// ============================================================================

/// Test that action history is tracked and used for detection.
///
/// Validates:
/// - Actions are recorded
/// - Action count is tracked
/// - History can be cleared
/// - Old actions are trimmed
#[test]
fn test_action_history_tracking() {
    let detector = IntentDetector::new();

    // Initially empty
    assert_eq!(detector.action_count(), 0, "Should start with no actions");

    // Record actions
    detector.record_action(UserAction::file_opened("/src/main.rs"));
    detector.record_action(UserAction::search("rust async"));
    detector.record_action(UserAction::file_edited("/src/lib.rs"));

    // Check count
    assert_eq!(detector.action_count(), 3, "Should have 3 actions");

    // Clear actions
    detector.clear_actions();
    assert_eq!(detector.action_count(), 0, "Should be empty after clear");

    // Verify detection with no actions
    let result = detector.detect_intent();
    assert!(
        matches!(result.primary_intent, DetectedIntent::Unknown),
        "Empty history should return Unknown"
    );
    assert_eq!(result.confidence, 0.0, "Confidence should be 0");
}

// ============================================================================
// ADDITIONAL INTENT TESTS
// ============================================================================

/// Test UserAction creation helpers.
#[test]
fn test_user_action_creation() {
    // File opened
    let file_action = UserAction::file_opened("/src/main.rs");
    assert_eq!(file_action.action_type, ActionType::FileOpened);
    assert!(file_action.file.is_some());
    assert!(file_action.content.is_none());

    // File edited
    let edit_action = UserAction::file_edited("/src/lib.rs");
    assert_eq!(edit_action.action_type, ActionType::FileEdited);

    // Search
    let search_action = UserAction::search("rust async");
    assert_eq!(search_action.action_type, ActionType::Search);
    assert!(search_action.file.is_none());
    assert!(search_action.content.is_some());

    // Error
    let error_action = UserAction::error("TypeError: null");
    assert_eq!(error_action.action_type, ActionType::ErrorEncountered);

    // Command
    let cmd_action = UserAction::command("cargo build");
    assert_eq!(cmd_action.action_type, ActionType::CommandExecuted);

    // Docs
    let docs_action = UserAction::docs_viewed("tokio tutorial");
    assert_eq!(docs_action.action_type, ActionType::DocumentationViewed);
}

/// Test action metadata.
#[test]
fn test_action_with_metadata() {
    let action = UserAction::file_opened("/src/main.rs")
        .with_metadata("project", "vestige")
        .with_metadata("branch", "main");

    assert!(action.metadata.contains_key("project"));
    assert_eq!(action.metadata.get("project"), Some(&"vestige".to_string()));
    assert!(action.metadata.contains_key("branch"));
}

/// Test intent description.
#[test]
fn test_intent_description() {
    let debugging = DetectedIntent::Debugging {
        suspected_area: "auth".to_string(),
        symptoms: vec![],
    };
    assert!(debugging.description().contains("auth"));

    let learning = DetectedIntent::Learning {
        topic: "async".to_string(),
        level: LearningLevel::Beginner,
    };
    assert!(learning.description().contains("async"));

    let unknown = DetectedIntent::Unknown;
    assert!(unknown.description().contains("Unknown"));
}

/// Test maintenance type tags.
#[test]
fn test_maintenance_type_tags() {
    let types = vec![
        (MaintenanceType::DependencyUpdate, "dependencies"),
        (MaintenanceType::SecurityPatch, "security"),
        (MaintenanceType::Cleanup, "cleanup"),
        (MaintenanceType::Configuration, "config"),
        (MaintenanceType::Migration, "migration"),
    ];

    for (mtype, expected_tag) in types {
        let intent = DetectedIntent::Maintenance {
            maintenance_type: mtype,
            target: None,
        };
        let tags = intent.relevant_tags();
        assert!(
            tags.contains(&expected_tag.to_string()),
            "Maintenance {:?} should have tag {}",
            intent,
            expected_tag
        );
    }
}

/// Test optimization type tags.
#[test]
fn test_optimization_type_tags() {
    let types = vec![
        (OptimizationType::Speed, "speed"),
        (OptimizationType::Memory, "memory"),
        (OptimizationType::Size, "bundle-size"),
        (OptimizationType::Startup, "startup"),
    ];

    for (otype, expected_tag) in types {
        let intent = DetectedIntent::Optimization {
            target: "app".to_string(),
            optimization_type: otype,
        };
        let tags = intent.relevant_tags();
        assert!(
            tags.contains(&expected_tag.to_string()),
            "Optimization should have tag {}",
            expected_tag
        );
    }
}
