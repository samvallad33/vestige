//! Pattern detection and storage for codebase memory
//!
//! This module handles:
//! - Learning new patterns from user teaching
//! - Detecting known patterns in code
//! - Suggesting relevant patterns based on context
//!
//! Patterns are the reusable pieces of knowledge that make Vestige smarter
//! over time. As the user teaches patterns, Vestige becomes more helpful
//! for that specific codebase.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::{Deserialize, Serialize};

use super::context::WorkingContext;
use super::types::CodePattern;

// ============================================================================
// ERRORS
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum PatternError {
    #[error("Pattern not found: {0}")]
    NotFound(String),
    #[error("Invalid pattern: {0}")]
    Invalid(String),
    #[error("Storage error: {0}")]
    Storage(String),
}

pub type Result<T> = std::result::Result<T, PatternError>;

// ============================================================================
// PATTERN MATCH
// ============================================================================

/// A detected pattern match in code
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PatternMatch {
    /// The pattern that was matched
    pub pattern: CodePattern,
    /// Confidence of the match (0.0 - 1.0)
    pub confidence: f64,
    /// Location in the code where pattern was detected
    pub location: Option<PatternLocation>,
    /// Suggestions based on this pattern match
    pub suggestions: Vec<String>,
}

/// Location where a pattern was detected
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PatternLocation {
    /// File where pattern was found
    pub file: PathBuf,
    /// Starting line (1-indexed)
    pub start_line: u32,
    /// Ending line (1-indexed)
    pub end_line: u32,
    /// Code snippet that matched
    pub snippet: String,
}

// ============================================================================
// PATTERN SUGGESTION
// ============================================================================

/// A suggested pattern based on context
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PatternSuggestion {
    /// The suggested pattern
    pub pattern: CodePattern,
    /// Why this pattern is being suggested
    pub reason: String,
    /// Relevance score (0.0 - 1.0)
    pub relevance: f64,
    /// Example of how to apply this pattern
    pub example: Option<String>,
}

// ============================================================================
// PATTERN DETECTOR
// ============================================================================

/// Detects and manages code patterns
pub struct PatternDetector {
    /// Stored patterns indexed by ID
    patterns: HashMap<String, CodePattern>,
    /// Patterns indexed by language for faster lookup
    patterns_by_language: HashMap<String, Vec<String>>,
    /// Pattern keywords for text matching
    pattern_keywords: HashMap<String, Vec<String>>,
}

impl PatternDetector {
    /// Create a new pattern detector
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            patterns_by_language: HashMap::new(),
            pattern_keywords: HashMap::new(),
        }
    }

    /// Learn a new pattern from user teaching
    pub fn learn_pattern(&mut self, pattern: CodePattern) -> Result<String> {
        // Validate the pattern
        if pattern.name.is_empty() {
            return Err(PatternError::Invalid(
                "Pattern name cannot be empty".to_string(),
            ));
        }
        if pattern.description.is_empty() {
            return Err(PatternError::Invalid(
                "Pattern description cannot be empty".to_string(),
            ));
        }

        let id = pattern.id.clone();

        // Index by language
        if let Some(ref language) = pattern.language {
            self.patterns_by_language
                .entry(language.to_lowercase())
                .or_default()
                .push(id.clone());
        }

        // Extract keywords for matching
        let keywords = self.extract_keywords(&pattern);
        self.pattern_keywords.insert(id.clone(), keywords);

        // Store the pattern
        self.patterns.insert(id.clone(), pattern);

        Ok(id)
    }

    /// Extract keywords from a pattern for matching
    fn extract_keywords(&self, pattern: &CodePattern) -> Vec<String> {
        let mut keywords = Vec::new();

        // Words from name
        keywords.extend(
            pattern
                .name
                .to_lowercase()
                .split_whitespace()
                .filter(|w| w.len() > 2)
                .map(|s| s.to_string()),
        );

        // Words from description
        keywords.extend(
            pattern
                .description
                .to_lowercase()
                .split_whitespace()
                .filter(|w| w.len() > 3)
                .map(|s| s.to_string()),
        );

        // Tags
        keywords.extend(pattern.tags.iter().map(|t| t.to_lowercase()));

        // Deduplicate
        keywords.sort();
        keywords.dedup();

        keywords
    }

    /// Get a pattern by ID
    pub fn get_pattern(&self, id: &str) -> Option<&CodePattern> {
        self.patterns.get(id)
    }

    /// Get all patterns
    pub fn get_all_patterns(&self) -> Vec<&CodePattern> {
        self.patterns.values().collect()
    }

    /// Get patterns for a specific language
    pub fn get_patterns_for_language(&self, language: &str) -> Vec<&CodePattern> {
        let language_lower = language.to_lowercase();

        self.patterns_by_language
            .get(&language_lower)
            .map(|ids| ids.iter().filter_map(|id| self.patterns.get(id)).collect())
            .unwrap_or_default()
    }

    /// Detect if current code matches known patterns
    pub fn detect_patterns(&self, code: &str, language: &str) -> Result<Vec<PatternMatch>> {
        let mut matches = Vec::new();
        let code_lower = code.to_lowercase();

        // Get relevant patterns for this language
        let relevant_patterns: Vec<_> = self
            .get_patterns_for_language(language)
            .into_iter()
            .chain(self.get_patterns_for_language("*"))
            .collect();

        for pattern in relevant_patterns {
            if let Some(confidence) = self.calculate_match_confidence(code, &code_lower, pattern)
                && confidence >= 0.3
            {
                matches.push(PatternMatch {
                    pattern: pattern.clone(),
                    confidence,
                    location: None, // Would need line-level analysis
                    suggestions: self.generate_suggestions(pattern, code),
                });
            }
        }

        // Sort by confidence
        matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(matches)
    }

    /// Calculate confidence that code matches a pattern
    fn calculate_match_confidence(
        &self,
        _code: &str,
        code_lower: &str,
        pattern: &CodePattern,
    ) -> Option<f64> {
        let keywords = self.pattern_keywords.get(&pattern.id)?;

        if keywords.is_empty() {
            return None;
        }

        // Count keyword matches
        let matches: usize = keywords
            .iter()
            .filter(|kw| code_lower.contains(kw.as_str()))
            .count();

        if matches == 0 {
            return None;
        }

        // Calculate confidence based on keyword match ratio
        let confidence = matches as f64 / keywords.len() as f64;

        // Boost confidence if example code matches
        let boost = if !pattern.example_code.is_empty()
            && code_lower.contains(&pattern.example_code.to_lowercase())
        {
            0.3
        } else {
            0.0
        };

        Some((confidence + boost).min(1.0))
    }

    /// Generate suggestions based on a matched pattern
    fn generate_suggestions(&self, pattern: &CodePattern, _code: &str) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Add the when_to_use guidance
        suggestions.push(format!("Consider: {}", pattern.when_to_use));

        // Add when_not_to_use if present
        if let Some(ref when_not) = pattern.when_not_to_use {
            suggestions.push(format!("Note: {}", when_not));
        }

        suggestions
    }

    /// Suggest patterns based on current context
    pub fn suggest_patterns(&self, context: &WorkingContext) -> Result<Vec<PatternSuggestion>> {
        let mut suggestions = Vec::new();

        // Get the language for the current context
        let language = match &context.project_type {
            super::context::ProjectType::Rust => "rust",
            super::context::ProjectType::TypeScript => "typescript",
            super::context::ProjectType::JavaScript => "javascript",
            super::context::ProjectType::Python => "python",
            super::context::ProjectType::Go => "go",
            super::context::ProjectType::Java => "java",
            super::context::ProjectType::Kotlin => "kotlin",
            super::context::ProjectType::Swift => "swift",
            super::context::ProjectType::CSharp => "csharp",
            super::context::ProjectType::Cpp => "cpp",
            super::context::ProjectType::Ruby => "ruby",
            super::context::ProjectType::Php => "php",
            super::context::ProjectType::Mixed(_) => "*",
            super::context::ProjectType::Unknown => "*",
        };

        // Get patterns for this language
        let language_patterns = self.get_patterns_for_language(language);

        // Score patterns based on context relevance
        for pattern in language_patterns {
            let relevance = self.calculate_context_relevance(pattern, context);

            if relevance >= 0.2 {
                let reason = self.generate_suggestion_reason(pattern, context);

                suggestions.push(PatternSuggestion {
                    pattern: pattern.clone(),
                    reason,
                    relevance,
                    example: if !pattern.example_code.is_empty() {
                        Some(pattern.example_code.clone())
                    } else {
                        None
                    },
                });
            }
        }

        // Sort by relevance
        suggestions.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(suggestions)
    }

    /// Calculate how relevant a pattern is to the current context
    fn calculate_context_relevance(&self, pattern: &CodePattern, context: &WorkingContext) -> f64 {
        let mut score = 0.0;

        // Check if pattern files overlap with active files
        if let Some(ref active) = context.active_file {
            for example_file in &pattern.example_files {
                if self.paths_related(active, example_file) {
                    score += 0.3;
                    break;
                }
            }
        }

        // Check framework relevance
        for framework in &context.frameworks {
            let framework_name = framework.name().to_lowercase();
            if pattern
                .tags
                .iter()
                .any(|t| t.to_lowercase() == framework_name)
                || pattern.description.to_lowercase().contains(&framework_name)
            {
                score += 0.2;
            }
        }

        // Check recent usage
        if pattern.usage_count > 0 {
            score += (pattern.usage_count as f64 / 100.0).min(0.3);
        }

        score.min(1.0)
    }

    /// Check if two paths are related (same directory, similar names, etc.)
    fn paths_related(&self, a: &Path, b: &Path) -> bool {
        // Same parent directory
        if a.parent() == b.parent() {
            return true;
        }

        // Similar file names
        if let (Some(a_stem), Some(b_stem)) = (a.file_stem(), b.file_stem()) {
            let a_str = a_stem.to_string_lossy().to_lowercase();
            let b_str = b_stem.to_string_lossy().to_lowercase();

            if a_str.contains(&b_str) || b_str.contains(&a_str) {
                return true;
            }
        }

        false
    }

    /// Generate a reason for suggesting a pattern
    fn generate_suggestion_reason(
        &self,
        pattern: &CodePattern,
        context: &WorkingContext,
    ) -> String {
        let mut reasons = Vec::new();

        // Language match
        if let Some(ref lang) = pattern.language {
            reasons.push(format!("Relevant for {} code", lang));
        }

        // Framework match
        for framework in &context.frameworks {
            let framework_name = framework.name();
            if pattern
                .tags
                .iter()
                .any(|t| t.eq_ignore_ascii_case(framework_name))
                || pattern
                    .description
                    .to_lowercase()
                    .contains(&framework_name.to_lowercase())
            {
                reasons.push(format!("Used with {}", framework_name));
            }
        }

        // Usage count
        if pattern.usage_count > 5 {
            reasons.push(format!("Commonly used ({} times)", pattern.usage_count));
        }

        if reasons.is_empty() {
            "May be applicable in this context".to_string()
        } else {
            reasons.join("; ")
        }
    }

    /// Update pattern usage count
    pub fn record_pattern_usage(&mut self, pattern_id: &str) -> Result<()> {
        if let Some(pattern) = self.patterns.get_mut(pattern_id) {
            pattern.usage_count += 1;
            Ok(())
        } else {
            Err(PatternError::NotFound(pattern_id.to_string()))
        }
    }

    /// Delete a pattern
    pub fn delete_pattern(&mut self, pattern_id: &str) -> Result<()> {
        if self.patterns.remove(pattern_id).is_some() {
            // Clean up indexes
            for ids in self.patterns_by_language.values_mut() {
                ids.retain(|id| id != pattern_id);
            }
            self.pattern_keywords.remove(pattern_id);
            Ok(())
        } else {
            Err(PatternError::NotFound(pattern_id.to_string()))
        }
    }

    /// Search patterns by query
    pub fn search_patterns(&self, query: &str) -> Vec<&CodePattern> {
        let query_lower = query.to_lowercase();
        let query_words: Vec<_> = query_lower.split_whitespace().collect();

        let mut scored: Vec<_> = self
            .patterns
            .values()
            .filter_map(|pattern| {
                let name_match = pattern.name.to_lowercase().contains(&query_lower);
                let desc_match = pattern.description.to_lowercase().contains(&query_lower);
                let tag_match = pattern
                    .tags
                    .iter()
                    .any(|t| t.to_lowercase().contains(&query_lower));

                // Count word matches
                let keywords = self.pattern_keywords.get(&pattern.id)?;
                let word_matches = query_words
                    .iter()
                    .filter(|w| keywords.iter().any(|kw| kw.contains(*w)))
                    .count();

                let score = if name_match {
                    1.0
                } else if tag_match {
                    0.8
                } else if desc_match {
                    0.6
                } else if word_matches > 0 {
                    0.4 * (word_matches as f64 / query_words.len() as f64)
                } else {
                    return None;
                };

                Some((pattern, score))
            })
            .collect();

        // Sort by score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().map(|(p, _)| p).collect()
    }

    /// Load patterns from storage (to be implemented with actual storage)
    pub fn load_patterns(&mut self, patterns: Vec<CodePattern>) -> Result<()> {
        for pattern in patterns {
            self.learn_pattern(pattern)?;
        }
        Ok(())
    }

    /// Export all patterns for storage
    pub fn export_patterns(&self) -> Vec<CodePattern> {
        self.patterns.values().cloned().collect()
    }
}

impl Default for PatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// BUILT-IN PATTERNS
// ============================================================================

/// Create built-in patterns for common coding patterns
pub fn create_builtin_patterns() -> Vec<CodePattern> {
    vec![
        // Rust Error Handling Pattern
        CodePattern {
            id: "builtin-rust-error-handling".to_string(),
            name: "Rust Error Handling with thiserror".to_string(),
            description: "Use thiserror for defining custom error types with derive macros"
                .to_string(),
            example_code: r#"
#[derive(Debug, thiserror::Error)]
pub enum MyError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
}

pub type Result<T> = std::result::Result<T, MyError>;
"#
            .to_string(),
            example_files: vec![],
            when_to_use: "When defining domain-specific error types in Rust".to_string(),
            when_not_to_use: Some("For simple one-off errors, anyhow might be simpler".to_string()),
            language: Some("rust".to_string()),
            created_at: Utc::now(),
            usage_count: 0,
            tags: vec!["error-handling".to_string(), "rust".to_string()],
            related_patterns: vec!["builtin-rust-result".to_string()],
        },
        // TypeScript React Component Pattern
        CodePattern {
            id: "builtin-react-functional".to_string(),
            name: "React Functional Component".to_string(),
            description: "Modern React functional component with TypeScript".to_string(),
            example_code: r#"
interface Props {
    title: string;
    onClick?: () => void;
}

export function MyComponent({ title, onClick }: Props) {
    return (
        <div onClick={onClick}>
            <h1>{title}</h1>
        </div>
    );
}
"#
            .to_string(),
            example_files: vec![],
            when_to_use: "For all new React components".to_string(),
            when_not_to_use: Some("Class components are rarely needed in modern React".to_string()),
            language: Some("typescript".to_string()),
            created_at: Utc::now(),
            usage_count: 0,
            tags: vec![
                "react".to_string(),
                "typescript".to_string(),
                "component".to_string(),
            ],
            related_patterns: vec![],
        },
        // Repository Pattern
        CodePattern {
            id: "builtin-repository-pattern".to_string(),
            name: "Repository Pattern".to_string(),
            description: "Abstract data access behind a repository interface".to_string(),
            example_code: r#"
pub trait UserRepository {
    fn find_by_id(&self, id: &str) -> Result<Option<User>>;
    fn save(&self, user: &User) -> Result<()>;
    fn delete(&self, id: &str) -> Result<()>;
}

pub struct SqliteUserRepository {
    conn: Connection,
}

impl UserRepository for SqliteUserRepository {
    // Implementation...
}
"#
            .to_string(),
            example_files: vec![],
            when_to_use: "When you need to decouple domain logic from data access".to_string(),
            when_not_to_use: Some("For simple CRUD with no complex domain logic".to_string()),
            language: Some("rust".to_string()),
            created_at: Utc::now(),
            usage_count: 0,
            tags: vec!["architecture".to_string(), "data-access".to_string()],
            related_patterns: vec![],
        },
    ]
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_pattern() -> CodePattern {
        CodePattern {
            id: "test-pattern-1".to_string(),
            name: "Test Pattern".to_string(),
            description: "A test pattern for unit testing".to_string(),
            example_code: "let x = test_function();".to_string(),
            example_files: vec![PathBuf::from("src/test.rs")],
            when_to_use: "When testing".to_string(),
            when_not_to_use: None,
            language: Some("rust".to_string()),
            created_at: Utc::now(),
            usage_count: 0,
            tags: vec!["test".to_string()],
            related_patterns: vec![],
        }
    }

    #[test]
    fn test_learn_pattern() {
        let mut detector = PatternDetector::new();
        let pattern = create_test_pattern();

        let result = detector.learn_pattern(pattern.clone());
        assert!(result.is_ok());

        let stored = detector.get_pattern("test-pattern-1");
        assert!(stored.is_some());
        assert_eq!(stored.unwrap().name, "Test Pattern");
    }

    #[test]
    fn test_detect_patterns() {
        let mut detector = PatternDetector::new();
        let pattern = create_test_pattern();
        detector.learn_pattern(pattern).unwrap();

        let code = "fn main() { let x = test_function(); }";
        let matches = detector.detect_patterns(code, "rust").unwrap();

        assert!(!matches.is_empty());
    }

    #[test]
    fn test_get_patterns_for_language() {
        let mut detector = PatternDetector::new();
        let pattern = create_test_pattern();
        detector.learn_pattern(pattern).unwrap();

        let rust_patterns = detector.get_patterns_for_language("rust");
        assert_eq!(rust_patterns.len(), 1);

        let ts_patterns = detector.get_patterns_for_language("typescript");
        assert!(ts_patterns.is_empty());
    }

    #[test]
    fn test_search_patterns() {
        let mut detector = PatternDetector::new();
        let pattern = create_test_pattern();
        detector.learn_pattern(pattern).unwrap();

        let results = detector.search_patterns("test");
        assert_eq!(results.len(), 1);

        let results = detector.search_patterns("unknown");
        assert!(results.is_empty());
    }

    #[test]
    fn test_delete_pattern() {
        let mut detector = PatternDetector::new();
        let pattern = create_test_pattern();
        detector.learn_pattern(pattern).unwrap();

        assert!(detector.get_pattern("test-pattern-1").is_some());

        detector.delete_pattern("test-pattern-1").unwrap();

        assert!(detector.get_pattern("test-pattern-1").is_none());
    }

    #[test]
    fn test_builtin_patterns() {
        let patterns = create_builtin_patterns();
        assert!(!patterns.is_empty());

        // Check that each pattern has required fields
        for pattern in patterns {
            assert!(!pattern.id.is_empty());
            assert!(!pattern.name.is_empty());
            assert!(!pattern.description.is_empty());
            assert!(!pattern.when_to_use.is_empty());
        }
    }
}
