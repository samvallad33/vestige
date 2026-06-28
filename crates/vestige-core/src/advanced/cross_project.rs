//! # Cross-Project Learning
//!
//! Learn patterns that apply across ALL projects. Vestige doesn't just remember
//! project-specific knowledge - it identifies universal patterns that make you
//! more effective everywhere.
//!
//! ## Pattern Types
//!
//! - **Code Patterns**: Error handling, async patterns, testing strategies
//! - **Architecture Patterns**: Project structures, module organization
//! - **Process Patterns**: Debug workflows, refactoring approaches
//! - **Domain Patterns**: Industry-specific knowledge that transfers
//!
//! ## How It Works
//!
//! 1. **Pattern Extraction**: Analyzes memories across projects for commonalities
//! 2. **Success Tracking**: Monitors which patterns led to successful outcomes
//! 3. **Applicability Detection**: Recognizes when current context matches a pattern
//! 4. **Suggestion Generation**: Provides actionable suggestions based on patterns
//!
//! ## Example
//!
//! ```rust,ignore
//! let learner = CrossProjectLearner::new();
//!
//! // Find patterns that worked across multiple projects
//! let patterns = learner.find_universal_patterns();
//!
//! // Apply to a new project
//! let suggestions = learner.apply_to_project(Path::new("/new/project"));
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Minimum projects a pattern must appear in to be considered universal
const MIN_PROJECTS_FOR_UNIVERSAL: usize = 2;

/// Minimum success rate for pattern recommendations
const MIN_SUCCESS_RATE: f64 = 0.6;

/// A universal pattern found across multiple projects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalPattern {
    /// Unique pattern ID
    pub id: String,
    /// The pattern itself
    pub pattern: CodePattern,
    /// Projects where this pattern was observed
    pub projects_seen_in: Vec<String>,
    /// Success rate (how often it helped)
    pub success_rate: f64,
    /// Description of when this pattern is applicable
    pub applicability: String,
    /// Confidence in this pattern (based on evidence)
    pub confidence: f64,
    /// When this pattern was first observed
    pub first_seen: DateTime<Utc>,
    /// When this pattern was last observed
    pub last_seen: DateTime<Utc>,
    /// How many times this pattern was applied
    pub application_count: u32,
}

/// A code pattern that can be learned and applied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodePattern {
    /// Pattern name/identifier
    pub name: String,
    /// Pattern category
    pub category: PatternCategory,
    /// Description of the pattern
    pub description: String,
    /// Example code or usage
    pub example: Option<String>,
    /// Conditions that suggest this pattern applies
    pub triggers: Vec<PatternTrigger>,
    /// What the pattern helps with
    pub benefits: Vec<String>,
    /// Potential drawbacks or considerations
    pub considerations: Vec<String>,
}

/// Categories of patterns
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PatternCategory {
    /// Error handling patterns
    ErrorHandling,
    /// Async/concurrent code patterns
    AsyncConcurrency,
    /// Testing strategies
    Testing,
    /// Code organization/architecture
    Architecture,
    /// Performance optimization
    Performance,
    /// Security practices
    Security,
    /// Debugging approaches
    Debugging,
    /// Refactoring techniques
    Refactoring,
    /// Documentation practices
    Documentation,
    /// Build/tooling patterns
    Tooling,
    /// Custom category
    Custom(String),
}

/// Conditions that trigger pattern applicability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternTrigger {
    /// Type of trigger
    pub trigger_type: TriggerType,
    /// Value/pattern to match
    pub value: String,
    /// Confidence that this trigger indicates pattern applies
    pub confidence: f64,
}

/// Types of triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerType {
    /// File name or extension
    FileName,
    /// Code construct or keyword
    CodeConstruct,
    /// Error message pattern
    ErrorMessage,
    /// Directory structure
    DirectoryStructure,
    /// Dependency/import
    Dependency,
    /// Intent detected
    Intent,
    /// Topic being discussed
    Topic,
}

/// Knowledge that might apply to current context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplicableKnowledge {
    /// The pattern that might apply
    pub pattern: UniversalPattern,
    /// Why we think it applies
    pub match_reason: String,
    /// Confidence that it applies here
    pub applicability_confidence: f64,
    /// Specific suggestions for applying it
    pub suggestions: Vec<String>,
    /// Memories that support this application
    pub supporting_memories: Vec<String>,
}

/// A suggestion for applying patterns to a project
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    /// What we suggest
    pub suggestion: String,
    /// Pattern this is based on
    pub based_on: String,
    /// Confidence level
    pub confidence: f64,
    /// Supporting evidence (memory IDs)
    pub evidence: Vec<String>,
    /// Priority (higher = more important)
    pub priority: u32,
}

/// Context about the current project
#[derive(Debug, Clone, Default)]
pub struct ProjectContext {
    /// Project root path
    pub path: Option<PathBuf>,
    /// Project name
    pub name: Option<String>,
    /// Languages used
    pub languages: Vec<String>,
    /// Frameworks detected
    pub frameworks: Vec<String>,
    /// File types present
    pub file_types: HashSet<String>,
    /// Dependencies
    pub dependencies: Vec<String>,
    /// Project structure (key directories)
    pub structure: Vec<String>,
}

impl ProjectContext {
    /// Create context from a project path (would scan project in production)
    pub fn from_path(path: &Path) -> Self {
        Self {
            path: Some(path.to_path_buf()),
            name: path.file_name().map(|n| n.to_string_lossy().to_string()),
            ..Default::default()
        }
    }

    /// Add detected language
    pub fn with_language(mut self, lang: &str) -> Self {
        self.languages.push(lang.to_string());
        self
    }

    /// Add detected framework
    pub fn with_framework(mut self, framework: &str) -> Self {
        self.frameworks.push(framework.to_string());
        self
    }
}

/// Project memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectMemory {
    memory_id: String,
    project_name: String,
    category: Option<PatternCategory>,
    was_helpful: Option<bool>,
    timestamp: DateTime<Utc>,
}

/// Cross-project learning engine
pub struct CrossProjectLearner {
    /// Patterns discovered
    patterns: Arc<RwLock<HashMap<String, UniversalPattern>>>,
    /// Project-memory associations
    project_memories: Arc<RwLock<Vec<ProjectMemory>>>,
    /// Pattern application outcomes
    outcomes: Arc<RwLock<Vec<PatternOutcome>>>,
}

/// Outcome of applying a pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PatternOutcome {
    pattern_id: String,
    project_name: String,
    was_successful: bool,
    timestamp: DateTime<Utc>,
}

impl CrossProjectLearner {
    /// Create a new cross-project learner
    pub fn new() -> Self {
        Self {
            patterns: Arc::new(RwLock::new(HashMap::new())),
            project_memories: Arc::new(RwLock::new(Vec::new())),
            outcomes: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Find patterns that appear in multiple projects
    pub fn find_universal_patterns(&self) -> Vec<UniversalPattern> {
        let patterns = self
            .patterns
            .read()
            .map(|p| p.values().cloned().collect::<Vec<_>>())
            .unwrap_or_default();

        patterns
            .into_iter()
            .filter(|p| {
                p.projects_seen_in.len() >= MIN_PROJECTS_FOR_UNIVERSAL
                    && p.success_rate >= MIN_SUCCESS_RATE
            })
            .collect()
    }

    /// Apply learned patterns to a new project
    pub fn apply_to_project(&self, project: &Path) -> Vec<Suggestion> {
        let context = ProjectContext::from_path(project);
        self.generate_suggestions(&context)
    }

    /// Apply with full context
    pub fn apply_to_context(&self, context: &ProjectContext) -> Vec<Suggestion> {
        self.generate_suggestions(context)
    }

    /// Detect when current situation matches cross-project knowledge
    pub fn detect_applicable(&self, context: &ProjectContext) -> Vec<ApplicableKnowledge> {
        let mut applicable = Vec::new();

        let patterns = self
            .patterns
            .read()
            .map(|p| p.values().cloned().collect::<Vec<_>>())
            .unwrap_or_default();

        for pattern in patterns {
            if let Some(knowledge) = self.check_pattern_applicability(&pattern, context) {
                applicable.push(knowledge);
            }
        }

        // Sort by applicability confidence (handle NaN safely)
        applicable.sort_by(|a, b| {
            b.applicability_confidence
                .partial_cmp(&a.applicability_confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        applicable
    }

    /// Record that a memory was associated with a project
    pub fn record_project_memory(
        &self,
        memory_id: &str,
        project_name: &str,
        category: Option<PatternCategory>,
    ) {
        if let Ok(mut memories) = self.project_memories.write() {
            memories.push(ProjectMemory {
                memory_id: memory_id.to_string(),
                project_name: project_name.to_string(),
                category,
                was_helpful: None,
                timestamp: Utc::now(),
            });
        }
    }

    /// Record outcome of applying a pattern
    pub fn record_pattern_outcome(
        &self,
        pattern_id: &str,
        project_name: &str,
        was_successful: bool,
    ) {
        // Record outcome
        if let Ok(mut outcomes) = self.outcomes.write() {
            outcomes.push(PatternOutcome {
                pattern_id: pattern_id.to_string(),
                project_name: project_name.to_string(),
                was_successful,
                timestamp: Utc::now(),
            });
        }

        // Update pattern success rate
        self.update_pattern_success_rate(pattern_id);
    }

    /// Add or update a pattern
    pub fn add_pattern(&self, pattern: UniversalPattern) {
        if let Ok(mut patterns) = self.patterns.write() {
            patterns.insert(pattern.id.clone(), pattern);
        }
    }

    /// Learn patterns from existing memories
    pub fn learn_from_memories(&self, memories: &[MemoryForLearning]) {
        // Group memories by category
        let mut by_category: HashMap<PatternCategory, Vec<&MemoryForLearning>> = HashMap::new();

        for memory in memories {
            if let Some(cat) = &memory.category {
                by_category.entry(cat.clone()).or_default().push(memory);
            }
        }

        // Find patterns within each category
        for (category, cat_memories) in by_category {
            self.extract_patterns_from_category(category, &cat_memories);
        }
    }

    /// Get all discovered patterns
    pub fn get_all_patterns(&self) -> Vec<UniversalPattern> {
        self.patterns
            .read()
            .map(|p| p.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Get patterns by category
    pub fn get_patterns_by_category(&self, category: &PatternCategory) -> Vec<UniversalPattern> {
        self.patterns
            .read()
            .map(|p| {
                p.values()
                    .filter(|pat| &pat.pattern.category == category)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    // ========================================================================
    // Private implementation
    // ========================================================================

    fn generate_suggestions(&self, context: &ProjectContext) -> Vec<Suggestion> {
        let mut suggestions = Vec::new();

        let patterns = self
            .patterns
            .read()
            .map(|p| p.values().cloned().collect::<Vec<_>>())
            .unwrap_or_default();

        for pattern in patterns {
            if let Some(applicable) = self.check_pattern_applicability(&pattern, context) {
                for (i, suggestion_text) in applicable.suggestions.iter().enumerate() {
                    suggestions.push(Suggestion {
                        suggestion: suggestion_text.clone(),
                        based_on: pattern.pattern.name.clone(),
                        confidence: applicable.applicability_confidence,
                        evidence: applicable.supporting_memories.clone(),
                        // saturating_sub: when confidence is low (small base) and
                        // the suggestion index i exceeds it, a plain `-` underflows
                        // — panicking in debug, wrapping to a huge value in release
                        // (which corrupts the Reverse-priority sort). Saturate to 0.
                        priority: ((10.0 * applicable.applicability_confidence) as u32)
                            .saturating_sub(i as u32),
                    });
                }
            }
        }

        suggestions.sort_by_key(|b| std::cmp::Reverse(b.priority));
        suggestions
    }

    fn check_pattern_applicability(
        &self,
        pattern: &UniversalPattern,
        context: &ProjectContext,
    ) -> Option<ApplicableKnowledge> {
        let mut match_scores: Vec<f64> = Vec::new();
        let mut match_reasons: Vec<String> = Vec::new();

        // Check each trigger
        for trigger in &pattern.pattern.triggers {
            if let Some((matches, reason)) = self.check_trigger(trigger, context)
                && matches
            {
                match_scores.push(trigger.confidence);
                match_reasons.push(reason);
            }
        }

        if match_scores.is_empty() {
            return None;
        }

        // Calculate overall confidence
        let avg_confidence = match_scores.iter().sum::<f64>() / match_scores.len() as f64;

        // Boost confidence based on pattern's track record
        let adjusted_confidence = avg_confidence * pattern.success_rate * pattern.confidence;

        if adjusted_confidence < 0.3 {
            return None;
        }

        // Generate suggestions based on pattern
        let suggestions = self.generate_pattern_suggestions(pattern, context);

        Some(ApplicableKnowledge {
            pattern: pattern.clone(),
            match_reason: match_reasons.join("; "),
            applicability_confidence: adjusted_confidence,
            suggestions,
            supporting_memories: Vec::new(), // Would be filled from storage
        })
    }

    fn check_trigger(
        &self,
        trigger: &PatternTrigger,
        context: &ProjectContext,
    ) -> Option<(bool, String)> {
        match &trigger.trigger_type {
            TriggerType::FileName => {
                let matches = context
                    .file_types
                    .iter()
                    .any(|ft| ft.contains(&trigger.value));
                Some((matches, format!("Found {} files", trigger.value)))
            }
            TriggerType::Dependency => {
                let matches = context
                    .dependencies
                    .iter()
                    .any(|d| d.to_lowercase().contains(&trigger.value.to_lowercase()));
                Some((matches, format!("Uses {}", trigger.value)))
            }
            TriggerType::CodeConstruct => {
                // Would need actual code analysis
                Some((false, String::new()))
            }
            TriggerType::DirectoryStructure => {
                let matches = context.structure.iter().any(|d| d.contains(&trigger.value));
                Some((matches, format!("Has {} directory", trigger.value)))
            }
            TriggerType::Topic | TriggerType::Intent | TriggerType::ErrorMessage => {
                // These would be checked against current conversation/context
                Some((false, String::new()))
            }
        }
    }

    fn generate_pattern_suggestions(
        &self,
        pattern: &UniversalPattern,
        _context: &ProjectContext,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Base suggestion from pattern description
        suggestions.push(format!(
            "Consider using: {} - {}",
            pattern.pattern.name, pattern.pattern.description
        ));

        // Add benefit-based suggestions
        for benefit in &pattern.pattern.benefits {
            suggestions.push(format!("This can help with: {}", benefit));
        }

        // Add example if available
        if let Some(example) = &pattern.pattern.example {
            suggestions.push(format!("Example: {}", example));
        }

        suggestions
    }

    fn update_pattern_success_rate(&self, pattern_id: &str) {
        let (success_count, total_count) = {
            let Some(outcomes) = self.outcomes.read().ok() else {
                return;
            };

            let relevant: Vec<_> = outcomes
                .iter()
                .filter(|o| o.pattern_id == pattern_id)
                .collect();

            let success = relevant.iter().filter(|o| o.was_successful).count();
            (success, relevant.len())
        };

        if total_count == 0 {
            return;
        }

        let success_rate = success_count as f64 / total_count as f64;

        if let Ok(mut patterns) = self.patterns.write()
            && let Some(pattern) = patterns.get_mut(pattern_id)
        {
            pattern.success_rate = success_rate;
            pattern.application_count = total_count as u32;
        }
    }

    fn extract_patterns_from_category(
        &self,
        category: PatternCategory,
        memories: &[&MemoryForLearning],
    ) {
        // Group by project
        let mut by_project: HashMap<&str, Vec<&MemoryForLearning>> = HashMap::new();
        for memory in memories {
            by_project
                .entry(&memory.project_name)
                .or_default()
                .push(memory);
        }

        // Find common themes across projects
        if by_project.len() < MIN_PROJECTS_FOR_UNIVERSAL {
            return;
        }

        // Simple pattern: look for common keywords in content
        let mut keyword_projects: HashMap<String, HashSet<&str>> = HashMap::new();

        for (project, project_memories) in &by_project {
            for memory in project_memories {
                for word in memory.content.split_whitespace() {
                    let clean = word
                        .trim_matches(|c: char| !c.is_alphanumeric())
                        .to_lowercase();
                    if clean.len() > 5 {
                        keyword_projects.entry(clean).or_default().insert(project);
                    }
                }
            }
        }

        // Keywords appearing in multiple projects might indicate patterns
        for (keyword, projects) in keyword_projects {
            if projects.len() >= MIN_PROJECTS_FOR_UNIVERSAL {
                // Create a potential pattern (simplified)
                let pattern_id = format!("auto-{}-{}", category_to_string(&category), keyword);

                if let Ok(mut patterns) = self.patterns.write()
                    && !patterns.contains_key(&pattern_id)
                {
                    patterns.insert(
                        pattern_id.clone(),
                        UniversalPattern {
                            id: pattern_id,
                            pattern: CodePattern {
                                name: format!("{} pattern", keyword),
                                category: category.clone(),
                                description: format!(
                                    "Pattern involving '{}' observed in {} projects",
                                    keyword,
                                    projects.len()
                                ),
                                example: None,
                                triggers: vec![PatternTrigger {
                                    trigger_type: TriggerType::Topic,
                                    value: keyword.clone(),
                                    confidence: 0.5,
                                }],
                                benefits: vec![],
                                considerations: vec![],
                            },
                            projects_seen_in: projects.iter().map(|s| s.to_string()).collect(),
                            success_rate: 0.5, // Default until validated
                            applicability: format!("When working with {}", keyword),
                            confidence: 0.5,
                            first_seen: Utc::now(),
                            last_seen: Utc::now(),
                            application_count: 0,
                        },
                    );
                }
            }
        }
    }
}

impl Default for CrossProjectLearner {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory input for learning
#[derive(Debug, Clone)]
pub struct MemoryForLearning {
    /// Memory ID
    pub id: String,
    /// Memory content
    pub content: String,
    /// Project name
    pub project_name: String,
    /// Category
    pub category: Option<PatternCategory>,
}

fn category_to_string(cat: &PatternCategory) -> &'static str {
    match cat {
        PatternCategory::ErrorHandling => "error-handling",
        PatternCategory::AsyncConcurrency => "async",
        PatternCategory::Testing => "testing",
        PatternCategory::Architecture => "architecture",
        PatternCategory::Performance => "performance",
        PatternCategory::Security => "security",
        PatternCategory::Debugging => "debugging",
        PatternCategory::Refactoring => "refactoring",
        PatternCategory::Documentation => "docs",
        PatternCategory::Tooling => "tooling",
        PatternCategory::Custom(_) => "custom",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_context() {
        let context = ProjectContext::from_path(Path::new("/my/project"))
            .with_language("rust")
            .with_framework("tokio");

        assert_eq!(context.name, Some("project".to_string()));
        assert!(context.languages.contains(&"rust".to_string()));
        assert!(context.frameworks.contains(&"tokio".to_string()));
    }

    #[test]
    fn test_record_pattern_outcome() {
        let learner = CrossProjectLearner::new();

        // Add a pattern
        learner.add_pattern(UniversalPattern {
            id: "test-pattern".to_string(),
            pattern: CodePattern {
                name: "Test".to_string(),
                category: PatternCategory::Testing,
                description: "Test pattern".to_string(),
                example: None,
                triggers: vec![],
                benefits: vec![],
                considerations: vec![],
            },
            projects_seen_in: vec!["proj1".to_string(), "proj2".to_string()],
            success_rate: 0.5,
            applicability: "Testing".to_string(),
            confidence: 0.5,
            first_seen: Utc::now(),
            last_seen: Utc::now(),
            application_count: 0,
        });

        // Record successes
        learner.record_pattern_outcome("test-pattern", "proj3", true);
        learner.record_pattern_outcome("test-pattern", "proj4", true);
        learner.record_pattern_outcome("test-pattern", "proj5", false);

        // Check updated success rate
        let patterns = learner.get_all_patterns();
        let pattern = patterns.iter().find(|p| p.id == "test-pattern").unwrap();
        assert!((pattern.success_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_find_universal_patterns() {
        let learner = CrossProjectLearner::new();

        // Pattern in only one project (not universal)
        learner.add_pattern(UniversalPattern {
            id: "local".to_string(),
            pattern: CodePattern {
                name: "Local".to_string(),
                category: PatternCategory::Testing,
                description: "Local only".to_string(),
                example: None,
                triggers: vec![],
                benefits: vec![],
                considerations: vec![],
            },
            projects_seen_in: vec!["proj1".to_string()],
            success_rate: 0.8,
            applicability: "".to_string(),
            confidence: 0.5,
            first_seen: Utc::now(),
            last_seen: Utc::now(),
            application_count: 0,
        });

        // Pattern in multiple projects (universal)
        learner.add_pattern(UniversalPattern {
            id: "universal".to_string(),
            pattern: CodePattern {
                name: "Universal".to_string(),
                category: PatternCategory::ErrorHandling,
                description: "Universal pattern".to_string(),
                example: None,
                triggers: vec![],
                benefits: vec![],
                considerations: vec![],
            },
            projects_seen_in: vec![
                "proj1".to_string(),
                "proj2".to_string(),
                "proj3".to_string(),
            ],
            success_rate: 0.9,
            applicability: "".to_string(),
            confidence: 0.7,
            first_seen: Utc::now(),
            last_seen: Utc::now(),
            application_count: 5,
        });

        let universal = learner.find_universal_patterns();
        assert_eq!(universal.len(), 1);
        assert_eq!(universal[0].id, "universal");
    }
}
