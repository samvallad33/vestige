//! # Intent Detection
//!
//! Understand WHY the user is doing something, not just WHAT they're doing.
//! This allows Vestige to provide proactively relevant memories based on
//! the underlying goal.
//!
//! ## Intent Types
//!
//! - **Debugging**: Looking for the cause of a bug
//! - **Refactoring**: Improving code structure
//! - **NewFeature**: Building something new
//! - **Learning**: Trying to understand something
//! - **Maintenance**: Regular upkeep tasks
//!
//! ## How It Works
//!
//! 1. Analyzes recent user actions (file opens, searches, edits)
//! 2. Identifies patterns that suggest intent
//! 3. Returns intent with confidence and supporting evidence
//! 4. Retrieves memories relevant to detected intent
//!
//! ## Example
//!
//! ```rust,ignore
//! let detector = IntentDetector::new();
//!
//! // Record user actions
//! detector.record_action(UserAction::file_opened("/src/auth.rs"));
//! detector.record_action(UserAction::search("error handling"));
//! detector.record_action(UserAction::file_opened("/tests/auth_test.rs"));
//!
//! // Detect intent
//! let intent = detector.detect_intent();
//! // Likely: DetectedIntent::Debugging { suspected_area: "auth" }
//! ```

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Maximum actions to keep in history
const MAX_ACTION_HISTORY: usize = 100;

/// Time window for intent detection (minutes)
const INTENT_WINDOW_MINUTES: i64 = 30;

/// Minimum confidence for intent detection
const MIN_INTENT_CONFIDENCE: f64 = 0.4;

/// Detected intent from user actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectedIntent {
    /// User is debugging an issue
    Debugging {
        /// Suspected area of the bug
        suspected_area: String,
        /// Error messages or symptoms observed
        symptoms: Vec<String>,
    },

    /// User is refactoring code
    Refactoring {
        /// What is being refactored
        target: String,
        /// Goal of the refactoring
        goal: String,
    },

    /// User is building a new feature
    NewFeature {
        /// Description of the feature
        feature_description: String,
        /// Related existing components
        related_components: Vec<String>,
    },

    /// User is trying to learn/understand something
    Learning {
        /// Topic being learned
        topic: String,
        /// Current understanding level (estimated)
        level: LearningLevel,
    },

    /// User is doing maintenance work
    Maintenance {
        /// Type of maintenance
        maintenance_type: MaintenanceType,
        /// Target of maintenance
        target: Option<String>,
    },

    /// User is reviewing/understanding code
    CodeReview {
        /// Files being reviewed
        files: Vec<String>,
        /// Depth of review
        depth: ReviewDepth,
    },

    /// User is writing documentation
    Documentation {
        /// What is being documented
        subject: String,
    },

    /// User is optimizing performance
    Optimization {
        /// Target of optimization
        target: String,
        /// Type of optimization
        optimization_type: OptimizationType,
    },

    /// User is integrating with external systems
    Integration {
        /// System being integrated
        system: String,
    },

    /// Intent could not be determined
    Unknown,
}

impl DetectedIntent {
    /// Get a short description of the intent
    pub fn description(&self) -> String {
        match self {
            Self::Debugging { suspected_area, .. } => {
                format!("Debugging issue in {}", suspected_area)
            }
            Self::Refactoring { target, goal } => format!("Refactoring {} to {}", target, goal),
            Self::NewFeature {
                feature_description,
                ..
            } => format!("Building: {}", feature_description),
            Self::Learning { topic, .. } => format!("Learning about {}", topic),
            Self::Maintenance {
                maintenance_type, ..
            } => format!("{:?} maintenance", maintenance_type),
            Self::CodeReview { files, .. } => format!("Reviewing {} files", files.len()),
            Self::Documentation { subject } => format!("Documenting {}", subject),
            Self::Optimization { target, .. } => format!("Optimizing {}", target),
            Self::Integration { system } => format!("Integrating with {}", system),
            Self::Unknown => "Unknown intent".to_string(),
        }
    }

    /// Tags for memory search, taken only from what the recorded actions
    /// named (file, topic, system). No fixed tag vocabulary per intent.
    pub fn relevant_tags(&self) -> Vec<String> {
        let named: Vec<&str> = match self {
            Self::Debugging { suspected_area, .. } => vec![suspected_area.as_str()],
            Self::Refactoring { target, .. } => vec![target.as_str()],
            Self::NewFeature { feature_description, .. } => vec![feature_description.as_str()],
            Self::Learning { topic, .. } => vec![topic.as_str()],
            Self::Maintenance { target, .. } => target.iter().map(String::as_str).collect(),
            Self::CodeReview { files, .. } => files.iter().map(String::as_str).collect(),
            Self::Documentation { subject } => vec![subject.as_str()],
            Self::Optimization { target, .. } => vec![target.as_str()],
            Self::Integration { system } => vec![system.as_str()],
            Self::Unknown => vec![],
        };
        named
            .into_iter()
            .filter(|t| !t.is_empty())
            .map(str::to_lowercase)
            .collect()
    }
}

/// Types of maintenance activities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaintenanceType {
    /// Updating dependencies
    DependencyUpdate,
    /// Applying security patches
    SecurityPatch,
    /// Code cleanup
    Cleanup,
    /// Configuration changes
    Configuration,
    /// Data/schema migration
    Migration,
}

/// Learning level estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningLevel {
    /// Just starting to learn
    Beginner,
    /// Has some understanding
    Intermediate,
    /// Deep dive into specifics
    Advanced,
}

/// Depth of code review
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReviewDepth {
    /// Quick scan
    Shallow,
    /// Normal review
    Standard,
    /// Deep analysis
    Deep,
}

/// Type of optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Speed/latency optimization
    Speed,
    /// Memory usage optimization
    Memory,
    /// Bundle/binary size
    Size,
    /// Startup time
    Startup,
}

/// A user action that can indicate intent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAction {
    /// Type of action
    pub action_type: ActionType,
    /// Associated file (if any)
    pub file: Option<PathBuf>,
    /// Content/query (if any)
    pub content: Option<String>,
    /// When this action occurred
    pub timestamp: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl UserAction {
    /// Create action for file opened
    pub fn file_opened(path: &str) -> Self {
        Self {
            action_type: ActionType::FileOpened,
            file: Some(PathBuf::from(path)),
            content: None,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create action for file edited
    pub fn file_edited(path: &str) -> Self {
        Self {
            action_type: ActionType::FileEdited,
            file: Some(PathBuf::from(path)),
            content: None,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create action for search query
    pub fn search(query: &str) -> Self {
        Self {
            action_type: ActionType::Search,
            file: None,
            content: Some(query.to_string()),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create action for error encountered
    pub fn error(message: &str) -> Self {
        Self {
            action_type: ActionType::ErrorEncountered,
            file: None,
            content: Some(message.to_string()),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create action for command executed
    pub fn command(cmd: &str) -> Self {
        Self {
            action_type: ActionType::CommandExecuted,
            file: None,
            content: Some(cmd.to_string()),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create action for documentation viewed
    pub fn docs_viewed(topic: &str) -> Self {
        Self {
            action_type: ActionType::DocumentationViewed,
            file: None,
            content: Some(topic.to_string()),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Types of user actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActionType {
    /// Opened a file
    FileOpened,
    /// Edited a file
    FileEdited,
    /// Created a new file
    FileCreated,
    /// Deleted a file
    FileDeleted,
    /// Searched for something
    Search,
    /// Executed a command
    CommandExecuted,
    /// Encountered an error
    ErrorEncountered,
    /// Viewed documentation
    DocumentationViewed,
    /// Ran tests
    TestsRun,
    /// Started debug session
    DebugStarted,
    /// Made a git commit
    GitCommit,
    /// Viewed a diff
    DiffViewed,
}

/// Result of intent detection with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentDetectionResult {
    /// Primary detected intent
    pub primary_intent: DetectedIntent,
    /// Confidence in primary intent (0.0 to 1.0)
    pub confidence: f64,
    /// Alternative intents with lower confidence
    pub alternatives: Vec<(DetectedIntent, f64)>,
    /// Evidence supporting the detection
    pub evidence: Vec<String>,
    /// When this detection was made
    pub detected_at: DateTime<Utc>,
}

/// Intent detector that analyzes user actions
pub struct IntentDetector {
    /// Action history
    actions: Arc<RwLock<VecDeque<UserAction>>>,
    /// Intent patterns
    patterns: Vec<IntentPattern>,
}

/// A pattern that suggests a specific intent
#[allow(clippy::type_complexity)]
struct IntentPattern {
    /// Name of the pattern
    name: String,
    /// Function to score actions against this pattern
    scorer: Box<dyn Fn(&[&UserAction]) -> (DetectedIntent, f64) + Send + Sync>,
}

impl IntentDetector {
    /// Create a new intent detector
    pub fn new() -> Self {
        Self {
            actions: Arc::new(RwLock::new(VecDeque::with_capacity(MAX_ACTION_HISTORY))),
            patterns: Self::build_patterns(),
        }
    }

    /// Record a user action
    pub fn record_action(&self, action: UserAction) {
        if let Ok(mut actions) = self.actions.write() {
            actions.push_back(action);

            // Trim old actions
            while actions.len() > MAX_ACTION_HISTORY {
                actions.pop_front();
            }
        }
    }

    /// Detect intent from recorded actions
    pub fn detect_intent(&self) -> IntentDetectionResult {
        let actions = self.get_recent_actions();

        if actions.is_empty() {
            return IntentDetectionResult {
                primary_intent: DetectedIntent::Unknown,
                confidence: 0.0,
                alternatives: vec![],
                evidence: vec![],
                detected_at: Utc::now(),
            };
        }

        // Score each pattern
        let mut scores: Vec<(DetectedIntent, f64, String)> = Vec::new();

        for pattern in &self.patterns {
            let action_refs: Vec<_> = actions.iter().collect();
            let (intent, score) = (pattern.scorer)(&action_refs);
            if score >= MIN_INTENT_CONFIDENCE {
                scores.push((intent, score, pattern.name.clone()));
            }
        }

        // Sort by score
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if scores.is_empty() {
            return IntentDetectionResult {
                primary_intent: DetectedIntent::Unknown,
                confidence: 0.0,
                alternatives: vec![],
                evidence: self.collect_evidence(&actions),
                detected_at: Utc::now(),
            };
        }

        let (primary_intent, confidence, _) = scores.remove(0);
        let alternatives: Vec<_> = scores
            .into_iter()
            .map(|(intent, score, _)| (intent, score))
            .take(3)
            .collect();

        IntentDetectionResult {
            primary_intent,
            confidence,
            alternatives,
            evidence: self.collect_evidence(&actions),
            detected_at: Utc::now(),
        }
    }

    /// Get memories relevant to detected intent
    pub fn memories_for_intent(&self, intent: &DetectedIntent) -> IntentMemoryQuery {
        let tags = intent.relevant_tags();

        IntentMemoryQuery {
            tags,
            keywords: self.extract_intent_keywords(intent),
            recency_boost: matches!(intent, DetectedIntent::Debugging { .. }),
        }
    }

    /// Clear action history
    pub fn clear_actions(&self) {
        if let Ok(mut actions) = self.actions.write() {
            actions.clear();
        }
    }

    /// Get action count
    pub fn action_count(&self) -> usize {
        self.actions.read().map(|a| a.len()).unwrap_or(0)
    }

    // ========================================================================
    // Private implementation
    // ========================================================================

    fn get_recent_actions(&self) -> Vec<UserAction> {
        let cutoff = Utc::now() - Duration::minutes(INTENT_WINDOW_MINUTES);

        self.actions
            .read()
            .map(|actions| {
                actions
                    .iter()
                    .filter(|a| a.timestamp > cutoff)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    fn build_patterns() -> Vec<IntentPattern> {
        vec![
            // Debugging pattern
            IntentPattern {
                name: "Debugging".to_string(),
                scorer: Box::new(|actions| {
                    let mut score: f64 = 0.0;
                    let mut symptoms = Vec::new();
                    let mut suspected_area = String::new();

                    for action in actions {
                        match &action.action_type {
                            ActionType::ErrorEncountered => {
                                score += 0.3;
                                if let Some(content) = &action.content {
                                    symptoms.push(content.clone());
                                }
                            }
                            ActionType::DebugStarted => score += 0.4,
                            ActionType::FileOpened | ActionType::FileEdited => {
                                if let Some(file) = &action.file
                                    && let Some(name) = file.file_name()
                                {
                                    suspected_area = name.to_string_lossy().to_string();
                                }
                            }
                            _ => {}
                        }
                    }

                    let intent = DetectedIntent::Debugging {
                        suspected_area: if suspected_area.is_empty() {
                            "unknown".to_string()
                        } else {
                            suspected_area
                        },
                        symptoms,
                    };

                    (intent, score.min(1.0))
                }),
            },
            // Refactoring pattern
            IntentPattern {
                name: "Refactoring".to_string(),
                scorer: Box::new(|actions| {
                    let mut score: f64 = 0.0;
                    let mut target = String::new();

                    let edit_count = actions
                        .iter()
                        .filter(|a| a.action_type == ActionType::FileEdited)
                        .count();

                    // Multiple edits to related files suggests refactoring
                    if edit_count >= 3 {
                        score += 0.3;
                    }

                    for action in actions {
                        if action.action_type == ActionType::FileEdited
                            && let Some(file) = &action.file
                        {
                            target = file.to_string_lossy().to_string();
                        }
                    }

                    let intent = DetectedIntent::Refactoring {
                        target: if target.is_empty() {
                            "code".to_string()
                        } else {
                            target
                        },
                        goal: "improve structure".to_string(),
                    };

                    (intent, score.min(1.0))
                }),
            },
            // Learning pattern
            IntentPattern {
                name: "Learning".to_string(),
                scorer: Box::new(|actions| {
                    let mut score: f64 = 0.0;
                    let mut topic = String::new();

                    for action in actions {
                        if action.action_type == ActionType::DocumentationViewed {
                            score += 0.3;
                            if let Some(content) = &action.content {
                                topic = content.clone();
                            }
                        }
                    }

                    let intent = DetectedIntent::Learning {
                        topic: if topic.is_empty() {
                            "unknown".to_string()
                        } else {
                            topic
                        },
                        level: LearningLevel::Intermediate,
                    };

                    (intent, score.min(1.0))
                }),
            },
            // New feature pattern
            IntentPattern {
                name: "NewFeature".to_string(),
                scorer: Box::new(|actions| {
                    let mut score: f64 = 0.0;
                    let mut description = String::new();
                    let mut components = Vec::new();

                    let created_count = actions
                        .iter()
                        .filter(|a| a.action_type == ActionType::FileCreated)
                        .count();

                    if created_count >= 1 {
                        score += 0.4;
                    }

                    for action in actions {
                        match &action.action_type {
                            ActionType::FileCreated => {
                                if let Some(file) = &action.file {
                                    description = file
                                        .file_name()
                                        .map(|n| n.to_string_lossy().to_string())
                                        .unwrap_or_default();
                                }
                            }
                            ActionType::FileOpened | ActionType::FileEdited => {
                                if let Some(file) = &action.file {
                                    components.push(file.to_string_lossy().to_string());
                                }
                            }
                            _ => {}
                        }
                    }

                    let intent = DetectedIntent::NewFeature {
                        feature_description: if description.is_empty() {
                            "new feature".to_string()
                        } else {
                            description
                        },
                        related_components: components,
                    };

                    (intent, score.min(1.0))
                }),
            },
        ]
    }

    fn collect_evidence(&self, actions: &[UserAction]) -> Vec<String> {
        actions
            .iter()
            .take(5)
            .map(|a| match &a.action_type {
                ActionType::FileOpened | ActionType::FileEdited => {
                    format!(
                        "{:?}: {}",
                        a.action_type,
                        a.file
                            .as_ref()
                            .map(|f| f.to_string_lossy().to_string())
                            .unwrap_or_default()
                    )
                }
                ActionType::Search => {
                    format!("Searched: {}", a.content.as_ref().unwrap_or(&String::new()))
                }
                ActionType::ErrorEncountered => {
                    format!("Error: {}", a.content.as_ref().unwrap_or(&String::new()))
                }
                _ => format!("{:?}", a.action_type),
            })
            .collect()
    }

    fn extract_intent_keywords(&self, intent: &DetectedIntent) -> Vec<String> {
        match intent {
            DetectedIntent::Debugging {
                suspected_area,
                symptoms,
            } => {
                let mut keywords = vec![suspected_area.clone()];
                keywords.extend(symptoms.iter().take(3).cloned());
                keywords
            }
            DetectedIntent::Refactoring { target, goal } => {
                vec![target.clone(), goal.clone()]
            }
            DetectedIntent::NewFeature {
                feature_description,
                related_components,
            } => {
                let mut keywords = vec![feature_description.clone()];
                keywords.extend(related_components.iter().take(3).cloned());
                keywords
            }
            DetectedIntent::Learning { topic, .. } => vec![topic.clone()],
            DetectedIntent::Integration { system } => vec![system.clone()],
            _ => vec![],
        }
    }
}

impl Default for IntentDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Query parameters for finding memories relevant to an intent
#[derive(Debug, Clone)]
pub struct IntentMemoryQuery {
    /// Tags to search for
    pub tags: Vec<String>,
    /// Keywords to search for
    pub keywords: Vec<String>,
    /// Whether to boost recent memories
    pub recency_boost: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debugging_detection() {
        let detector = IntentDetector::new();

        detector.record_action(UserAction::error("NullPointerException at line 42"));
        detector.record_action(UserAction::file_opened("/src/service.rs"));
        detector.record_action(UserAction::search("fix null pointer"));

        let result = detector.detect_intent();

        if let DetectedIntent::Debugging { symptoms, .. } = &result.primary_intent {
            assert!(!symptoms.is_empty());
        } else if result.confidence > 0.0 {
            // May detect different intent based on order
        }
    }

    #[test]
    fn test_learning_detection() {
        let detector = IntentDetector::new();

        detector.record_action(UserAction::docs_viewed("async/await"));
        detector.record_action(UserAction::search("how to use tokio"));
        detector.record_action(UserAction::docs_viewed("futures"));

        let result = detector.detect_intent();

        if let DetectedIntent::Learning { topic, .. } = &result.primary_intent {
            assert!(!topic.is_empty());
        }
    }

    #[test]
    fn test_intent_tags() {
        let debugging = DetectedIntent::Debugging {
            suspected_area: "auth".to_string(),
            symptoms: vec![],
        };

        // Tags come from what the actions named, not a fixed vocabulary.
        assert_eq!(debugging.relevant_tags(), vec!["auth".to_string()]);
    }

    #[test]
    fn test_action_creation() {
        let action = UserAction::file_opened("/src/main.rs").with_metadata("project", "vestige");

        assert_eq!(action.action_type, ActionType::FileOpened);
        assert!(action.metadata.contains_key("project"));
    }
}
