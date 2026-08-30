//! Codebase Memory Module - Vestige's KILLER DIFFERENTIATOR
//!
//! This module makes Vestige unique in the AI memory market. No other tool
//! understands codebases at this level - remembering architectural decisions,
//! bug fixes, patterns, file relationships, and developer preferences.
//!
//! # Overview
//!
//! The Codebase Memory Module provides:
//!
//! - **Git History Analysis**: Automatically learns from your codebase's history
//!   - Extracts bug fix patterns from commit messages
//!   - Discovers file co-change patterns (files that always change together)
//!   - Understands the evolution of the codebase
//!
//! - **Context Capture**: Knows what you're working on
//!   - Current branch and uncommitted changes
//!   - Project type and frameworks
//!   - Active files and editing context
//!
//! - **Pattern Detection**: Learns and applies coding patterns
//!   - User-taught patterns
//!   - Auto-detected patterns from code
//!   - Context-aware pattern suggestions
//!
//! - **Relationship Tracking**: Understands file relationships
//!   - Import/dependency relationships
//!   - Test-implementation pairs
//!   - Co-edit patterns
//!
//! - **File Watching**: Continuous learning from developer behavior
//!   - Tracks files edited together
//!   - Updates relationship strengths
//!   - Triggers pattern detection
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use vestige_core::codebase::CodebaseMemory;
//! use std::path::PathBuf;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create codebase memory for a project
//! let memory = CodebaseMemory::new(PathBuf::from("/path/to/project"))?;
//!
//! // Learn from git history
//! let analysis = memory.learn_from_history().await?;
//! println!("Found {} bug fixes", analysis.bug_fixes_found);
//! println!("Found {} file relationships", analysis.relationships_found);
//!
//! // Get current context
//! let context = memory.get_context()?;
//! println!("Working on branch: {}", context.git.as_ref().map(|g| &g.current_branch).unwrap_or(&"unknown".to_string()));
//!
//! // Remember an architectural decision
//! memory.remember_decision(
//!     "Use Event Sourcing for order management",
//!     "Need complete audit trail and ability to replay state",
//!     vec![PathBuf::from("src/orders/events.rs")],
//! )?;
//!
//! // Query codebase memories
//! let results = memory.query("error handling", None)?;
//! for node in results {
//!     println!("Found: {}", node.to_searchable_text());
//! }
//! # Ok(())
//! # }
//! ```

pub mod anchor;
pub mod context;
pub mod git;
pub mod patterns;
pub mod relationships;
pub mod types;
pub mod watcher;

// Re-export main types
pub use anchor::{
    AnchorDraft, AnchorStatus, AnchorVerification, CodeAnchor, MAX_ANCHORED_FILE_BYTES,
    MAX_SPAN_LINES, capture_anchor, find_symbol_definition, hash_span, verify_anchor,
};
pub use context::{ContextCapture, FileContext, Framework, ProjectType, WorkingContext};
pub use git::{CommitInfo, GitAnalyzer, GitContext, HistoryAnalysis};
pub use patterns::{PatternDetector, PatternMatch, PatternSuggestion};
pub use relationships::{
    GraphEdge, GraphMetadata, GraphNode, RelatedFile, RelationshipGraph, RelationshipTracker,
};
pub use types::{
    ArchitecturalDecision, BugFix, BugSeverity, CodeEntity, CodePattern, CodebaseNode,
    CodingPreference, DecisionStatus, EntityType, FileRelationship, PreferenceSource, RelationType,
    RelationshipSource, WorkContext, WorkStatus,
};
pub use watcher::{CodebaseWatcher, FileEvent, FileEventKind, WatcherConfig};

use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use uuid::Uuid;

// ============================================================================
// ERRORS
// ============================================================================

/// Unified error type for codebase memory operations
#[derive(Debug, thiserror::Error)]
pub enum CodebaseError {
    #[error("Git error: {0}")]
    Git(#[from] git::GitError),
    #[error("Context error: {0}")]
    Context(#[from] context::ContextError),
    #[error("Pattern error: {0}")]
    Pattern(#[from] patterns::PatternError),
    #[error("Relationship error: {0}")]
    Relationship(#[from] relationships::RelationshipError),
    #[error("Watcher error: {0}")]
    Watcher(#[from] watcher::WatcherError),
    #[error("Storage error: {0}")]
    Storage(String),
    #[error("Not found: {0}")]
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, CodebaseError>;

// ============================================================================
// LEARNING RESULT
// ============================================================================

/// Result of learning from git history
#[derive(Debug)]
pub struct LearningResult {
    /// Bug fixes extracted
    pub bug_fixes_found: usize,
    /// File relationships discovered
    pub relationships_found: usize,
    /// Patterns detected
    pub patterns_detected: usize,
    /// Time range analyzed
    pub analyzed_since: Option<DateTime<Utc>>,
    /// Commits analyzed
    pub commits_analyzed: usize,
    /// Duration of analysis
    pub duration_ms: u64,
}

// ============================================================================
// CODEBASE MEMORY
// ============================================================================

/// Main codebase memory interface
///
/// This is the primary entry point for all codebase memory operations.
/// It coordinates between git analysis, context capture, pattern detection,
/// and relationship tracking.
pub struct CodebaseMemory {
    /// Repository path
    repo_path: PathBuf,
    /// Git analyzer
    pub git: GitAnalyzer,
    /// Context capture
    pub context: ContextCapture,
    /// Pattern detector
    patterns: Arc<RwLock<PatternDetector>>,
    /// Relationship tracker
    relationships: Arc<RwLock<RelationshipTracker>>,
    /// File watcher (optional)
    watcher: Option<Arc<RwLock<CodebaseWatcher>>>,
    /// Stored codebase nodes
    nodes: Arc<RwLock<Vec<CodebaseNode>>>,
}

impl CodebaseMemory {
    /// Create a new CodebaseMemory for a repository
    pub fn new(repo_path: PathBuf) -> Result<Self> {
        let git = GitAnalyzer::new(repo_path.clone())?;
        let context = ContextCapture::new(repo_path.clone())?;
        let patterns = Arc::new(RwLock::new(PatternDetector::new()));
        let relationships = Arc::new(RwLock::new(RelationshipTracker::new()));

        // Load built-in patterns
        {
            let mut detector = patterns.blocking_write();
            for pattern in patterns::create_builtin_patterns() {
                let _ = detector.learn_pattern(pattern);
            }
        }

        Ok(Self {
            repo_path,
            git,
            context,
            patterns,
            relationships,
            watcher: None,
            nodes: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Create with file watching enabled
    pub fn with_watcher(repo_path: PathBuf) -> Result<Self> {
        let mut memory = Self::new(repo_path)?;

        let watcher = CodebaseWatcher::new(
            Arc::clone(&memory.relationships),
            Arc::clone(&memory.patterns),
        );
        memory.watcher = Some(Arc::new(RwLock::new(watcher)));

        Ok(memory)
    }

    // ========================================================================
    // DECISION MANAGEMENT
    // ========================================================================

    /// Remember an architectural decision
    pub fn remember_decision(
        &self,
        decision: &str,
        rationale: &str,
        files_affected: Vec<PathBuf>,
    ) -> Result<String> {
        let id = format!("adr-{}", Uuid::new_v4());

        let node = CodebaseNode::ArchitecturalDecision(ArchitecturalDecision {
            id: id.clone(),
            decision: decision.to_string(),
            rationale: rationale.to_string(),
            files_affected,
            commit_sha: self.git.get_current_context().ok().map(|c| c.head_commit),
            created_at: Utc::now(),
            updated_at: None,
            context: None,
            tags: vec![],
            status: DecisionStatus::Accepted,
            alternatives_considered: vec![],
        });

        self.nodes.blocking_write().push(node);
        Ok(id)
    }

    /// Remember an architectural decision with full details
    pub fn remember_decision_full(&self, decision: ArchitecturalDecision) -> Result<String> {
        let id = decision.id.clone();
        self.nodes
            .blocking_write()
            .push(CodebaseNode::ArchitecturalDecision(decision));
        Ok(id)
    }

    // ========================================================================
    // BUG FIX MANAGEMENT
    // ========================================================================

    /// Remember a bug fix
    pub fn remember_bug_fix(&self, fix: BugFix) -> Result<String> {
        let id = fix.id.clone();
        self.nodes.blocking_write().push(CodebaseNode::BugFix(fix));
        Ok(id)
    }

    /// Remember a bug fix with minimal details
    pub fn remember_bug_fix_simple(
        &self,
        symptom: &str,
        root_cause: &str,
        solution: &str,
        files_changed: Vec<PathBuf>,
    ) -> Result<String> {
        let id = format!("bug-{}", Uuid::new_v4());
        let commit_sha = self
            .git
            .get_current_context()
            .map(|c| c.head_commit)
            .unwrap_or_default();

        let fix = BugFix::new(
            id.clone(),
            symptom.to_string(),
            root_cause.to_string(),
            solution.to_string(),
            commit_sha,
        )
        .with_files(files_changed);

        self.remember_bug_fix(fix)?;
        Ok(id)
    }

    // ========================================================================
    // PATTERN MANAGEMENT
    // ========================================================================

    /// Remember a coding pattern
    pub fn remember_pattern(&self, pattern: CodePattern) -> Result<String> {
        let id = pattern.id.clone();
        self.patterns.blocking_write().learn_pattern(pattern)?;
        Ok(id)
    }

    /// Get pattern suggestions for current context
    pub async fn get_pattern_suggestions(&self) -> Result<Vec<PatternSuggestion>> {
        let context = self.get_context()?;
        let detector = self.patterns.read().await;
        Ok(detector.suggest_patterns(&context)?)
    }

    /// Detect patterns in code
    pub async fn detect_patterns_in_code(
        &self,
        code: &str,
        language: &str,
    ) -> Result<Vec<PatternMatch>> {
        let detector = self.patterns.read().await;
        Ok(detector.detect_patterns(code, language)?)
    }

    // ========================================================================
    // PREFERENCE MANAGEMENT
    // ========================================================================

    /// Remember a coding preference
    pub fn remember_preference(&self, preference: CodingPreference) -> Result<String> {
        let id = preference.id.clone();
        self.nodes
            .blocking_write()
            .push(CodebaseNode::CodingPreference(preference));
        Ok(id)
    }

    /// Remember a simple preference
    pub fn remember_preference_simple(
        &self,
        context: &str,
        preference: &str,
        counter_preference: Option<&str>,
    ) -> Result<String> {
        let id = format!("pref-{}", Uuid::new_v4());

        let pref = CodingPreference::new(id.clone(), context.to_string(), preference.to_string())
            .with_confidence(0.8);

        let pref = if let Some(counter) = counter_preference {
            pref.with_counter(counter.to_string())
        } else {
            pref
        };

        self.remember_preference(pref)?;
        Ok(id)
    }

    // ========================================================================
    // RELATIONSHIP MANAGEMENT
    // ========================================================================

    /// Get files related to a given file
    pub async fn get_related_files(&self, file: &std::path::Path) -> Result<Vec<RelatedFile>> {
        let tracker = self.relationships.read().await;
        Ok(tracker.get_related_files(file)?)
    }

    /// Record that files were edited together
    pub async fn record_coedit(&self, files: &[PathBuf]) -> Result<()> {
        let mut tracker = self.relationships.write().await;
        Ok(tracker.record_coedit(files)?)
    }

    /// Build a relationship graph for visualization
    pub async fn build_relationship_graph(&self) -> Result<RelationshipGraph> {
        let tracker = self.relationships.read().await;
        Ok(tracker.build_graph()?)
    }

    // ========================================================================
    // CONTEXT
    // ========================================================================

    /// Get the current working context
    pub fn get_context(&self) -> Result<WorkingContext> {
        Ok(self.context.capture()?)
    }

    /// Get context for a specific file
    pub fn get_file_context(&self, path: &std::path::Path) -> Result<FileContext> {
        Ok(self.context.context_for_file(path)?)
    }

    /// Set active files for context tracking
    pub fn set_active_files(&mut self, files: Vec<PathBuf>) {
        self.context.set_active_files(files);
    }

    // ========================================================================
    // QUERY
    // ========================================================================

    /// Query codebase memories
    pub fn query(
        &self,
        query: &str,
        context: Option<&WorkingContext>,
    ) -> Result<Vec<CodebaseNode>> {
        let query_lower = query.to_lowercase();
        let nodes = self.nodes.blocking_read();

        let mut results: Vec<_> = nodes
            .iter()
            .filter(|node| {
                let text = node.to_searchable_text().to_lowercase();
                text.contains(&query_lower)
            })
            .cloned()
            .collect();

        // Boost results relevant to current context
        if let Some(ctx) = context {
            results.sort_by(|a, b| {
                let a_relevance = self.calculate_context_relevance(a, ctx);
                let b_relevance = self.calculate_context_relevance(b, ctx);
                b_relevance
                    .partial_cmp(&a_relevance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        Ok(results)
    }

    /// Calculate how relevant a node is to the current context
    fn calculate_context_relevance(&self, node: &CodebaseNode, context: &WorkingContext) -> f64 {
        let mut relevance = 0.0;

        // Check file overlap
        let node_files = node.associated_files();
        if let Some(ref active) = context.active_file {
            for file in &node_files {
                if *file == active {
                    relevance += 1.0;
                } else if file.parent() == active.parent() {
                    relevance += 0.5;
                }
            }
        }

        // Check framework relevance
        for framework in &context.frameworks {
            let text = node.to_searchable_text().to_lowercase();
            if text.contains(&framework.name().to_lowercase()) {
                relevance += 0.3;
            }
        }

        relevance
    }

    /// Get memories relevant to current context
    pub fn get_relevant(&self, context: &WorkingContext) -> Result<Vec<CodebaseNode>> {
        let nodes = self.nodes.blocking_read();

        let mut scored: Vec<_> = nodes
            .iter()
            .map(|node| {
                let relevance = self.calculate_context_relevance(node, context);
                (node.clone(), relevance)
            })
            .filter(|(_, relevance)| *relevance > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored.into_iter().map(|(node, _)| node).collect())
    }

    /// Get a node by ID
    pub fn get_node(&self, id: &str) -> Result<Option<CodebaseNode>> {
        let nodes = self.nodes.blocking_read();
        Ok(nodes.iter().find(|n| n.id() == id).cloned())
    }

    /// Get all nodes of a specific type
    pub fn get_nodes_by_type(&self, node_type: &str) -> Result<Vec<CodebaseNode>> {
        let nodes = self.nodes.blocking_read();
        Ok(nodes
            .iter()
            .filter(|n| n.node_type() == node_type)
            .cloned()
            .collect())
    }

    // ========================================================================
    // LEARNING
    // ========================================================================

    /// Learn from git history
    pub async fn learn_from_history(&self) -> Result<LearningResult> {
        let start = std::time::Instant::now();

        // Analyze history
        let analysis = self.git.analyze_history(None)?;

        // Store bug fixes
        let mut nodes = self.nodes.write().await;
        for fix in &analysis.bug_fixes {
            nodes.push(CodebaseNode::BugFix(fix.clone()));
        }

        // Store file relationships
        let mut tracker = self.relationships.write().await;
        for rel in &analysis.file_relationships {
            let _ = tracker.add_relationship(rel.clone());
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(LearningResult {
            bug_fixes_found: analysis.bug_fixes.len(),
            relationships_found: analysis.file_relationships.len(),
            patterns_detected: 0, // Could be extended
            analyzed_since: analysis.analyzed_since,
            commits_analyzed: analysis.commit_count,
            duration_ms,
        })
    }

    /// Learn from git history since a specific time
    pub async fn learn_from_history_since(&self, since: DateTime<Utc>) -> Result<LearningResult> {
        let start = std::time::Instant::now();

        let analysis = self.git.analyze_history(Some(since))?;

        let mut nodes = self.nodes.write().await;
        for fix in &analysis.bug_fixes {
            nodes.push(CodebaseNode::BugFix(fix.clone()));
        }

        let mut tracker = self.relationships.write().await;
        for rel in &analysis.file_relationships {
            let _ = tracker.add_relationship(rel.clone());
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(LearningResult {
            bug_fixes_found: analysis.bug_fixes.len(),
            relationships_found: analysis.file_relationships.len(),
            patterns_detected: 0,
            analyzed_since: Some(since),
            commits_analyzed: analysis.commit_count,
            duration_ms,
        })
    }

    // ========================================================================
    // FILE WATCHING
    // ========================================================================

    /// Start watching the repository for changes
    pub async fn start_watching(&self) -> Result<()> {
        if let Some(ref watcher) = self.watcher {
            let mut w = watcher.write().await;
            w.watch(&self.repo_path).await?;
        }
        Ok(())
    }

    /// Stop watching the repository
    pub async fn stop_watching(&self) -> Result<()> {
        if let Some(ref watcher) = self.watcher {
            let mut w = watcher.write().await;
            w.stop().await?;
        }
        Ok(())
    }

    // ========================================================================
    // SERIALIZATION
    // ========================================================================

    /// Export all nodes for storage
    pub fn export_nodes(&self) -> Vec<CodebaseNode> {
        self.nodes.blocking_read().clone()
    }

    /// Import nodes from storage
    pub fn import_nodes(&self, nodes: Vec<CodebaseNode>) {
        let mut current = self.nodes.blocking_write();
        current.extend(nodes);
    }

    /// Export patterns for storage
    pub fn export_patterns(&self) -> Vec<CodePattern> {
        self.patterns.blocking_read().export_patterns()
    }

    /// Import patterns from storage
    pub fn import_patterns(&self, patterns: Vec<CodePattern>) -> Result<()> {
        let mut detector = self.patterns.blocking_write();
        detector.load_patterns(patterns)?;
        Ok(())
    }

    /// Export relationships for storage
    pub fn export_relationships(&self) -> Vec<FileRelationship> {
        self.relationships.blocking_read().export_relationships()
    }

    /// Import relationships from storage
    pub fn import_relationships(&self, relationships: Vec<FileRelationship>) -> Result<()> {
        let mut tracker = self.relationships.blocking_write();
        tracker.load_relationships(relationships)?;
        Ok(())
    }

    // ========================================================================
    // STATS
    // ========================================================================

    /// Get statistics about codebase memory
    pub fn get_stats(&self) -> CodebaseStats {
        let nodes = self.nodes.blocking_read();
        let patterns = self.patterns.blocking_read();
        let relationships = self.relationships.blocking_read();

        CodebaseStats {
            total_nodes: nodes.len(),
            architectural_decisions: nodes
                .iter()
                .filter(|n| matches!(n, CodebaseNode::ArchitecturalDecision(_)))
                .count(),
            bug_fixes: nodes
                .iter()
                .filter(|n| matches!(n, CodebaseNode::BugFix(_)))
                .count(),
            patterns: patterns.get_all_patterns().len(),
            preferences: nodes
                .iter()
                .filter(|n| matches!(n, CodebaseNode::CodingPreference(_)))
                .count(),
            file_relationships: relationships.get_all_relationships().len(),
        }
    }
}

/// Statistics about codebase memory
#[derive(Debug, Clone)]
pub struct CodebaseStats {
    pub total_nodes: usize,
    pub architectural_decisions: usize,
    pub bug_fixes: usize,
    pub patterns: usize,
    pub preferences: usize,
    pub file_relationships: usize,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_repo() -> TempDir {
        let dir = TempDir::new().unwrap();

        // Initialize git repo
        git2::Repository::init(dir.path()).unwrap();

        // Create Cargo.toml
        std::fs::write(
            dir.path().join("Cargo.toml"),
            r#"
[package]
name = "test-project"
version = "0.1.0"
"#,
        )
        .unwrap();

        // Create src directory
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "fn main() {}").unwrap();

        dir
    }

    #[test]
    fn test_codebase_memory_creation() {
        let dir = create_test_repo();
        let memory = CodebaseMemory::new(dir.path().to_path_buf());
        assert!(memory.is_ok());
    }

    #[test]
    fn test_remember_decision() {
        let dir = create_test_repo();
        let memory = CodebaseMemory::new(dir.path().to_path_buf()).unwrap();

        let id = memory
            .remember_decision(
                "Use Event Sourcing",
                "Need audit trail",
                vec![PathBuf::from("src/events.rs")],
            )
            .unwrap();

        assert!(id.starts_with("adr-"));

        let node = memory.get_node(&id).unwrap();
        assert!(node.is_some());
    }

    #[test]
    fn test_remember_bug_fix() {
        let dir = create_test_repo();
        let memory = CodebaseMemory::new(dir.path().to_path_buf()).unwrap();

        let id = memory
            .remember_bug_fix_simple(
                "App crashes on startup",
                "Null pointer in config loading",
                "Added null check",
                vec![PathBuf::from("src/config.rs")],
            )
            .unwrap();

        assert!(id.starts_with("bug-"));
    }

    #[test]
    fn test_query() {
        let dir = create_test_repo();
        let memory = CodebaseMemory::new(dir.path().to_path_buf()).unwrap();

        memory
            .remember_decision("Use async/await for IO", "Better performance", vec![])
            .unwrap();

        memory
            .remember_decision("Use channels for communication", "Thread safety", vec![])
            .unwrap();

        let results = memory.query("async", None).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_get_context() {
        let dir = create_test_repo();
        let memory = CodebaseMemory::new(dir.path().to_path_buf()).unwrap();

        let context = memory.get_context().unwrap();
        assert_eq!(context.project_type, ProjectType::Rust);
    }

    #[test]
    fn test_stats() {
        let dir = create_test_repo();
        let memory = CodebaseMemory::new(dir.path().to_path_buf()).unwrap();

        memory.remember_decision("Test", "Test", vec![]).unwrap();

        let stats = memory.get_stats();
        assert_eq!(stats.architectural_decisions, 1);
        assert!(stats.patterns > 0); // Built-in patterns
    }
}
