//! Codebase-specific memory types for Vestige
//!
//! This module defines the specialized node types that make Vestige's codebase memory
//! unique and powerful. These types capture the contextual knowledge that developers
//! accumulate but traditionally lose - architectural decisions, bug fixes, coding
//! patterns, and file relationships.
//!
//! This is Vestige's KILLER DIFFERENTIATOR. No other AI memory system understands
//! codebases at this level.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ============================================================================
// CODEBASE NODE - The Core Memory Type
// ============================================================================

/// Types of memories specific to codebases.
///
/// Each variant captures a different kind of knowledge that developers accumulate
/// but typically lose over time or when context-switching between projects.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CodebaseNode {
    /// "We use X pattern because Y"
    ///
    /// Captures architectural decisions with their rationale. This is critical
    /// for maintaining consistency and understanding why the codebase evolved
    /// the way it did.
    ArchitecturalDecision(ArchitecturalDecision),

    /// "This bug was caused by X, fixed by Y"
    ///
    /// Records bug fixes with root cause analysis. Invaluable for preventing
    /// regression and understanding historical issues.
    BugFix(BugFix),

    /// "Use this pattern for X"
    ///
    /// Codifies recurring patterns with examples and guidance on when to use them.
    CodePattern(CodePattern),

    /// "These files always change together"
    ///
    /// Tracks file relationships discovered through git history analysis or
    /// explicit user teaching.
    FileRelationship(FileRelationship),

    /// "User prefers X over Y"
    ///
    /// Captures coding preferences and style decisions for consistent suggestions.
    CodingPreference(CodingPreference),

    /// "This function does X and is called by Y"
    ///
    /// Stores knowledge about specific code entities - functions, types, modules.
    CodeEntity(CodeEntity),

    /// "The current task is implementing X"
    ///
    /// Tracks ongoing work context for continuity across sessions.
    WorkContext(WorkContext),
}

impl CodebaseNode {
    /// Get the unique identifier for this node
    pub fn id(&self) -> &str {
        match self {
            Self::ArchitecturalDecision(n) => &n.id,
            Self::BugFix(n) => &n.id,
            Self::CodePattern(n) => &n.id,
            Self::FileRelationship(n) => &n.id,
            Self::CodingPreference(n) => &n.id,
            Self::CodeEntity(n) => &n.id,
            Self::WorkContext(n) => &n.id,
        }
    }

    /// Get the node type as a string
    pub fn node_type(&self) -> &'static str {
        match self {
            Self::ArchitecturalDecision(_) => "architectural_decision",
            Self::BugFix(_) => "bug_fix",
            Self::CodePattern(_) => "code_pattern",
            Self::FileRelationship(_) => "file_relationship",
            Self::CodingPreference(_) => "coding_preference",
            Self::CodeEntity(_) => "code_entity",
            Self::WorkContext(_) => "work_context",
        }
    }

    /// Get the creation timestamp
    pub fn created_at(&self) -> DateTime<Utc> {
        match self {
            Self::ArchitecturalDecision(n) => n.created_at,
            Self::BugFix(n) => n.created_at,
            Self::CodePattern(n) => n.created_at,
            Self::FileRelationship(n) => n.created_at,
            Self::CodingPreference(n) => n.created_at,
            Self::CodeEntity(n) => n.created_at,
            Self::WorkContext(n) => n.created_at,
        }
    }

    /// Get all file paths associated with this node
    pub fn associated_files(&self) -> Vec<&PathBuf> {
        match self {
            Self::ArchitecturalDecision(n) => n.files_affected.iter().collect(),
            Self::BugFix(n) => n.files_changed.iter().collect(),
            Self::CodePattern(n) => n.example_files.iter().collect(),
            Self::FileRelationship(n) => n.files.iter().collect(),
            Self::CodingPreference(_) => vec![],
            Self::CodeEntity(n) => n.file_path.as_ref().map(|p| vec![p]).unwrap_or_default(),
            Self::WorkContext(n) => n.active_files.iter().collect(),
        }
    }

    /// Convert to a searchable text representation
    pub fn to_searchable_text(&self) -> String {
        match self {
            Self::ArchitecturalDecision(n) => {
                format!(
                    "Architectural Decision: {} - Rationale: {} - Context: {}",
                    n.decision,
                    n.rationale,
                    n.context.as_deref().unwrap_or("")
                )
            }
            Self::BugFix(n) => {
                format!(
                    "Bug Fix: {} - Root Cause: {} - Solution: {}",
                    n.symptom, n.root_cause, n.solution
                )
            }
            Self::CodePattern(n) => {
                format!(
                    "Code Pattern: {} - {} - When to use: {}",
                    n.name, n.description, n.when_to_use
                )
            }
            Self::FileRelationship(n) => {
                format!(
                    "File Relationship: {:?} - Type: {:?} - {}",
                    n.files, n.relationship_type, n.description
                )
            }
            Self::CodingPreference(n) => {
                format!(
                    "Coding Preference ({}): {} vs {:?}",
                    n.context, n.preference, n.counter_preference
                )
            }
            Self::CodeEntity(n) => {
                format!(
                    "Code Entity: {} ({:?}) - {}",
                    n.name, n.entity_type, n.description
                )
            }
            Self::WorkContext(n) => {
                format!(
                    "Work Context: {} - {} - Active files: {:?}",
                    n.task_description,
                    n.status.as_str(),
                    n.active_files
                )
            }
        }
    }
}

// ============================================================================
// ARCHITECTURAL DECISION
// ============================================================================

/// Records an architectural decision with its rationale.
///
/// Example:
/// - Decision: "Use Event Sourcing for order management"
/// - Rationale: "Need complete audit trail and ability to replay state"
/// - Files: ["src/orders/events.rs", "src/orders/aggregate.rs"]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArchitecturalDecision {
    pub id: String,
    /// The decision that was made
    pub decision: String,
    /// Why this decision was made
    pub rationale: String,
    /// Files affected by this decision
    pub files_affected: Vec<PathBuf>,
    /// Git commit SHA where this was implemented (if applicable)
    pub commit_sha: Option<String>,
    /// When this decision was recorded
    pub created_at: DateTime<Utc>,
    /// When this decision was last updated
    pub updated_at: Option<DateTime<Utc>>,
    /// Additional context or notes
    pub context: Option<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Status of the decision
    pub status: DecisionStatus,
    /// Alternatives that were considered
    pub alternatives_considered: Vec<String>,
}

/// Status of an architectural decision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum DecisionStatus {
    /// Decision is proposed but not yet implemented
    Proposed,
    /// Decision is accepted and being implemented
    #[default]
    Accepted,
    /// Decision has been superseded by another
    Superseded,
    /// Decision was rejected
    Deprecated,
}


// ============================================================================
// BUG FIX
// ============================================================================

/// Records a bug fix with root cause analysis.
///
/// This is invaluable for:
/// - Preventing regressions
/// - Understanding why certain code exists
/// - Training junior developers on common pitfalls
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BugFix {
    pub id: String,
    /// What symptoms was the bug causing?
    pub symptom: String,
    /// What was the actual root cause?
    pub root_cause: String,
    /// How was it fixed?
    pub solution: String,
    /// Files that were changed to fix the bug
    pub files_changed: Vec<PathBuf>,
    /// Git commit SHA of the fix
    pub commit_sha: String,
    /// When the fix was recorded
    pub created_at: DateTime<Utc>,
    /// Link to issue tracker (if applicable)
    pub issue_link: Option<String>,
    /// Severity of the bug
    pub severity: BugSeverity,
    /// How the bug was discovered
    pub discovered_by: Option<String>,
    /// Prevention measures (what would have caught this earlier)
    pub prevention_notes: Option<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Severity level of a bug
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum BugSeverity {
    Critical,
    High,
    #[default]
    Medium,
    Low,
    Trivial,
}


// ============================================================================
// CODE PATTERN
// ============================================================================

/// Records a reusable code pattern with examples and guidance.
///
/// Patterns can be:
/// - Discovered automatically from git history
/// - Taught explicitly by the user
/// - Extracted from documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodePattern {
    pub id: String,
    /// Name of the pattern (e.g., "Repository Pattern", "Error Handling")
    pub name: String,
    /// Detailed description of the pattern
    pub description: String,
    /// Example code showing the pattern
    pub example_code: String,
    /// Files containing examples of this pattern
    pub example_files: Vec<PathBuf>,
    /// When should this pattern be used?
    pub when_to_use: String,
    /// When should this pattern NOT be used?
    pub when_not_to_use: Option<String>,
    /// Language this pattern applies to
    pub language: Option<String>,
    /// When this pattern was recorded
    pub created_at: DateTime<Utc>,
    /// How many times this pattern has been applied
    pub usage_count: u32,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Related patterns
    pub related_patterns: Vec<String>,
}

// ============================================================================
// FILE RELATIONSHIP
// ============================================================================

/// Tracks relationships between files in the codebase.
///
/// Relationships can be:
/// - Discovered from imports/dependencies
/// - Detected from git co-change patterns
/// - Explicitly taught by the user
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileRelationship {
    pub id: String,
    /// The files involved in this relationship
    pub files: Vec<PathBuf>,
    /// Type of relationship
    pub relationship_type: RelationType,
    /// Strength of the relationship (0.0 - 1.0)
    /// For co-change relationships, this is the frequency they change together
    pub strength: f64,
    /// Human-readable description
    pub description: String,
    /// When this relationship was first detected
    pub created_at: DateTime<Utc>,
    /// When this relationship was last confirmed
    pub last_confirmed: Option<DateTime<Utc>>,
    /// How this relationship was discovered
    pub source: RelationshipSource,
    /// Number of times this relationship has been observed
    pub observation_count: u32,
}

/// Types of relationships between files
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    /// A imports/depends on B
    ImportsDependency,
    /// A tests implementation in B
    TestsImplementation,
    /// A configures service B
    ConfiguresService,
    /// Files are in the same domain/feature area
    SharedDomain,
    /// Files frequently change together in commits
    FrequentCochange,
    /// A extends/implements B
    ExtendsImplements,
    /// A is the interface, B is the implementation
    InterfaceImplementation,
    /// A and B are related through documentation
    DocumentationReference,
}

/// How a relationship was discovered
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationshipSource {
    /// Detected from git history co-change analysis
    GitCochange,
    /// Detected from import/dependency analysis
    ImportAnalysis,
    /// Detected from AST analysis
    AstAnalysis,
    /// Explicitly taught by user
    UserDefined,
    /// Inferred from file naming conventions
    NamingConvention,
}

// ============================================================================
// CODING PREFERENCE
// ============================================================================

/// Records a user's coding preferences for consistent suggestions.
///
/// Examples:
/// - "For error handling, prefer Result over panic"
/// - "For naming, use snake_case for functions"
/// - "For async, prefer tokio over async-std"
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodingPreference {
    pub id: String,
    /// Context where this preference applies (e.g., "error handling", "naming")
    pub context: String,
    /// The preferred approach
    pub preference: String,
    /// What NOT to do (optional)
    pub counter_preference: Option<String>,
    /// Examples showing the preference in action
    pub examples: Vec<String>,
    /// Confidence in this preference (0.0 - 1.0)
    /// Higher confidence = more consistently applied
    pub confidence: f64,
    /// When this preference was recorded
    pub created_at: DateTime<Utc>,
    /// Language this applies to (None = all languages)
    pub language: Option<String>,
    /// How this preference was learned
    pub source: PreferenceSource,
    /// Number of times this preference has been observed
    pub observation_count: u32,
}

/// How a preference was learned
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreferenceSource {
    /// Explicitly stated by user
    UserStated,
    /// Inferred from code review feedback
    CodeReview,
    /// Detected from coding patterns in history
    PatternDetection,
    /// From project configuration (e.g., rustfmt.toml)
    ProjectConfig,
}

// ============================================================================
// CODE ENTITY
// ============================================================================

/// Knowledge about a specific code entity (function, type, module, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodeEntity {
    pub id: String,
    /// Name of the entity
    pub name: String,
    /// Type of entity
    pub entity_type: EntityType,
    /// Description of what this entity does
    pub description: String,
    /// File where this entity is defined
    pub file_path: Option<PathBuf>,
    /// Line number where entity starts
    pub line_number: Option<u32>,
    /// Entities that this one depends on
    pub dependencies: Vec<String>,
    /// Entities that depend on this one
    pub dependents: Vec<String>,
    /// When this was recorded
    pub created_at: DateTime<Utc>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Usage notes or gotchas
    pub notes: Option<String>,
}

/// Type of code entity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    Function,
    Method,
    Struct,
    Enum,
    Trait,
    Interface,
    Class,
    Module,
    Constant,
    Variable,
    Type,
}

// ============================================================================
// WORK CONTEXT
// ============================================================================

/// Tracks the current work context for continuity across sessions.
///
/// This allows Vestige to remember:
/// - What task the user was working on
/// - What files were being edited
/// - What the next steps were
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkContext {
    pub id: String,
    /// Description of the current task
    pub task_description: String,
    /// Files currently being worked on
    pub active_files: Vec<PathBuf>,
    /// Current git branch
    pub branch: Option<String>,
    /// Status of the work
    pub status: WorkStatus,
    /// Next steps that were planned
    pub next_steps: Vec<String>,
    /// Blockers or issues encountered
    pub blockers: Vec<String>,
    /// When this context was created
    pub created_at: DateTime<Utc>,
    /// When this context was last updated
    pub updated_at: DateTime<Utc>,
    /// Related issue/ticket IDs
    pub related_issues: Vec<String>,
    /// Notes about the work
    pub notes: Option<String>,
}

/// Status of work in progress
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkStatus {
    /// Actively being worked on
    InProgress,
    /// Paused, will resume later
    Paused,
    /// Completed
    Completed,
    /// Blocked by something
    Blocked,
    /// Abandoned
    Abandoned,
}

impl WorkStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::Paused => "paused",
            Self::Completed => "completed",
            Self::Blocked => "blocked",
            Self::Abandoned => "abandoned",
        }
    }
}

// ============================================================================
// BUILDER HELPERS
// ============================================================================

impl ArchitecturalDecision {
    pub fn new(id: String, decision: String, rationale: String) -> Self {
        Self {
            id,
            decision,
            rationale,
            files_affected: vec![],
            commit_sha: None,
            created_at: Utc::now(),
            updated_at: None,
            context: None,
            tags: vec![],
            status: DecisionStatus::default(),
            alternatives_considered: vec![],
        }
    }

    pub fn with_files(mut self, files: Vec<PathBuf>) -> Self {
        self.files_affected = files;
        self
    }

    pub fn with_commit(mut self, sha: String) -> Self {
        self.commit_sha = Some(sha);
        self
    }

    pub fn with_context(mut self, context: String) -> Self {
        self.context = Some(context);
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

impl BugFix {
    pub fn new(
        id: String,
        symptom: String,
        root_cause: String,
        solution: String,
        commit_sha: String,
    ) -> Self {
        Self {
            id,
            symptom,
            root_cause,
            solution,
            files_changed: vec![],
            commit_sha,
            created_at: Utc::now(),
            issue_link: None,
            severity: BugSeverity::default(),
            discovered_by: None,
            prevention_notes: None,
            tags: vec![],
        }
    }

    pub fn with_files(mut self, files: Vec<PathBuf>) -> Self {
        self.files_changed = files;
        self
    }

    pub fn with_severity(mut self, severity: BugSeverity) -> Self {
        self.severity = severity;
        self
    }

    pub fn with_issue(mut self, link: String) -> Self {
        self.issue_link = Some(link);
        self
    }
}

impl CodePattern {
    pub fn new(id: String, name: String, description: String, when_to_use: String) -> Self {
        Self {
            id,
            name,
            description,
            example_code: String::new(),
            example_files: vec![],
            when_to_use,
            when_not_to_use: None,
            language: None,
            created_at: Utc::now(),
            usage_count: 0,
            tags: vec![],
            related_patterns: vec![],
        }
    }

    pub fn with_example(mut self, code: String, files: Vec<PathBuf>) -> Self {
        self.example_code = code;
        self.example_files = files;
        self
    }

    pub fn with_language(mut self, language: String) -> Self {
        self.language = Some(language);
        self
    }
}

impl FileRelationship {
    pub fn new(
        id: String,
        files: Vec<PathBuf>,
        relationship_type: RelationType,
        description: String,
    ) -> Self {
        Self {
            id,
            files,
            relationship_type,
            strength: 0.5,
            description,
            created_at: Utc::now(),
            last_confirmed: None,
            source: RelationshipSource::UserDefined,
            observation_count: 1,
        }
    }

    pub fn from_git_cochange(id: String, files: Vec<PathBuf>, strength: f64, count: u32) -> Self {
        Self {
            id,
            files: files.clone(),
            relationship_type: RelationType::FrequentCochange,
            strength,
            description: format!(
                "Files frequently change together ({} co-occurrences)",
                count
            ),
            created_at: Utc::now(),
            last_confirmed: Some(Utc::now()),
            source: RelationshipSource::GitCochange,
            observation_count: count,
        }
    }
}

impl CodingPreference {
    pub fn new(id: String, context: String, preference: String) -> Self {
        Self {
            id,
            context,
            preference,
            counter_preference: None,
            examples: vec![],
            confidence: 0.5,
            created_at: Utc::now(),
            language: None,
            source: PreferenceSource::UserStated,
            observation_count: 1,
        }
    }

    pub fn with_counter(mut self, counter: String) -> Self {
        self.counter_preference = Some(counter);
        self
    }

    pub fn with_examples(mut self, examples: Vec<String>) -> Self {
        self.examples = examples;
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architectural_decision_builder() {
        let decision = ArchitecturalDecision::new(
            "adr-001".to_string(),
            "Use Event Sourcing".to_string(),
            "Need complete audit trail".to_string(),
        )
        .with_files(vec![PathBuf::from("src/events.rs")])
        .with_tags(vec!["architecture".to_string()]);

        assert_eq!(decision.id, "adr-001");
        assert!(!decision.files_affected.is_empty());
        assert!(!decision.tags.is_empty());
    }

    #[test]
    fn test_codebase_node_id() {
        let decision = ArchitecturalDecision::new(
            "test-id".to_string(),
            "Test".to_string(),
            "Test".to_string(),
        );
        let node = CodebaseNode::ArchitecturalDecision(decision);
        assert_eq!(node.id(), "test-id");
        assert_eq!(node.node_type(), "architectural_decision");
    }

    #[test]
    fn test_file_relationship_from_git() {
        let rel = FileRelationship::from_git_cochange(
            "rel-001".to_string(),
            vec![PathBuf::from("src/a.rs"), PathBuf::from("src/b.rs")],
            0.8,
            15,
        );

        assert_eq!(rel.relationship_type, RelationType::FrequentCochange);
        assert_eq!(rel.source, RelationshipSource::GitCochange);
        assert_eq!(rel.strength, 0.8);
        assert_eq!(rel.observation_count, 15);
    }

    #[test]
    fn test_searchable_text() {
        let pattern = CodePattern::new(
            "pat-001".to_string(),
            "Repository Pattern".to_string(),
            "Abstract data access".to_string(),
            "When you need to decouple domain logic from data access".to_string(),
        );
        let node = CodebaseNode::CodePattern(pattern);
        let text = node.to_searchable_text();

        assert!(text.contains("Repository Pattern"));
        assert!(text.contains("Abstract data access"));
    }
}
