//! Git history analysis for extracting codebase knowledge
//!
//! This module analyzes git history to automatically extract:
//! - File co-change patterns (files that frequently change together)
//! - Bug fix patterns (from commit messages matching conventional formats)
//! - Current git context (branch, uncommitted changes, recent history)
//!
//! This is a key differentiator for Vestige - learning from the codebase's history
//! without requiring explicit user input.

use chrono::{DateTime, TimeZone, Utc};
use git2::{Commit, Repository, Sort};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::types::{BugFix, BugSeverity, FileRelationship, RelationType, RelationshipSource};

// ============================================================================
// ERRORS
// ============================================================================

/// Errors that can occur during git analysis
#[derive(Debug, thiserror::Error)]
pub enum GitError {
    #[error("Git repository error: {0}")]
    Repository(#[from] git2::Error),
    #[error("Repository not found at: {0}")]
    NotFound(PathBuf),
    #[error("Invalid path: {0}")]
    InvalidPath(String),
    #[error("No commits found")]
    NoCommits,
}

pub type Result<T> = std::result::Result<T, GitError>;

// ============================================================================
// GIT CONTEXT
// ============================================================================

/// Current git context for a repository
#[derive(Debug, Clone)]
pub struct GitContext {
    /// Root path of the repository
    pub repo_root: PathBuf,
    /// Current branch name
    pub current_branch: String,
    /// HEAD commit SHA
    pub head_commit: String,
    /// Files with uncommitted changes (unstaged)
    pub uncommitted_changes: Vec<PathBuf>,
    /// Files staged for commit
    pub staged_changes: Vec<PathBuf>,
    /// Recent commits
    pub recent_commits: Vec<CommitInfo>,
    /// Whether the repository has any commits
    pub has_commits: bool,
    /// Whether there are untracked files
    pub has_untracked: bool,
}

/// Information about a git commit
#[derive(Debug, Clone)]
pub struct CommitInfo {
    /// Commit SHA (short)
    pub sha: String,
    /// Full commit SHA
    pub full_sha: String,
    /// Commit message (first line)
    pub message: String,
    /// Full commit message
    pub full_message: String,
    /// Author name
    pub author: String,
    /// Author email
    pub author_email: String,
    /// Commit timestamp
    pub timestamp: DateTime<Utc>,
    /// Files changed in this commit
    pub files_changed: Vec<PathBuf>,
    /// Is this a merge commit?
    pub is_merge: bool,
}

// ============================================================================
// GIT ANALYZER
// ============================================================================

/// Analyzes git history to extract knowledge
pub struct GitAnalyzer {
    repo_path: PathBuf,
}

impl GitAnalyzer {
    /// Create a new GitAnalyzer for the given repository path
    pub fn new(repo_path: PathBuf) -> Result<Self> {
        // Verify the repository exists
        let _ = Repository::open(&repo_path)?;
        Ok(Self { repo_path })
    }

    /// Open the repository
    fn open_repo(&self) -> Result<Repository> {
        Repository::open(&self.repo_path).map_err(GitError::from)
    }

    /// Get the current git context
    pub fn get_current_context(&self) -> Result<GitContext> {
        let repo = self.open_repo()?;

        // Get repository root
        let repo_root = repo
            .workdir()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| self.repo_path.clone());

        // Get current branch
        let current_branch = self.get_current_branch(&repo)?;

        // Get HEAD commit
        let (head_commit, has_commits) = match repo.head() {
            Ok(head) => match head.peel_to_commit() {
                Ok(commit) => (commit.id().to_string()[..8].to_string(), true),
                Err(_) => (String::new(), false),
            },
            Err(_) => (String::new(), false),
        };

        // Get status
        let statuses = repo.statuses(None)?;
        let mut uncommitted_changes = Vec::new();
        let mut staged_changes = Vec::new();
        let mut has_untracked = false;

        for entry in statuses.iter() {
            let path = entry.path().map(PathBuf::from).unwrap_or_default();

            let status = entry.status();

            if status.is_wt_new() {
                has_untracked = true;
            }
            if status.is_wt_modified() || status.is_wt_deleted() || status.is_wt_renamed() {
                uncommitted_changes.push(path.clone());
            }
            if status.is_index_new()
                || status.is_index_modified()
                || status.is_index_deleted()
                || status.is_index_renamed()
            {
                staged_changes.push(path);
            }
        }

        // Get recent commits
        let recent_commits = if has_commits {
            self.get_recent_commits(&repo, 10)?
        } else {
            vec![]
        };

        Ok(GitContext {
            repo_root,
            current_branch,
            head_commit,
            uncommitted_changes,
            staged_changes,
            recent_commits,
            has_commits,
            has_untracked,
        })
    }

    /// Get the current branch name
    fn get_current_branch(&self, repo: &Repository) -> Result<String> {
        match repo.head() {
            Ok(head) => {
                if head.is_branch() {
                    Ok(head
                        .shorthand()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "unknown".to_string()))
                } else {
                    // Detached HEAD
                    Ok(head
                        .target()
                        .map(|oid| oid.to_string()[..8].to_string())
                        .unwrap_or_else(|| "HEAD".to_string()))
                }
            }
            Err(_) => Ok("main".to_string()), // New repo with no commits
        }
    }

    /// Get recent commits
    fn get_recent_commits(&self, repo: &Repository, limit: usize) -> Result<Vec<CommitInfo>> {
        let mut revwalk = repo.revwalk()?;
        revwalk.push_head()?;
        revwalk.set_sorting(Sort::TIME)?;

        let mut commits = Vec::new();

        for oid in revwalk.take(limit) {
            let oid = oid?;
            let commit = repo.find_commit(oid)?;
            let commit_info = self.commit_to_info(&commit, repo)?;
            commits.push(commit_info);
        }

        Ok(commits)
    }

    /// Convert a git2::Commit to CommitInfo
    fn commit_to_info(&self, commit: &Commit, repo: &Repository) -> Result<CommitInfo> {
        let full_sha = commit.id().to_string();
        let sha = full_sha[..8].to_string();

        let message = commit
            .message()
            .map(|m| m.lines().next().unwrap_or("").to_string())
            .unwrap_or_default();

        let full_message = commit.message().map(|m| m.to_string()).unwrap_or_default();

        let author = commit.author();
        let author_name = author.name().unwrap_or("Unknown").to_string();
        let author_email = author.email().unwrap_or("").to_string();

        let timestamp = Utc
            .timestamp_opt(commit.time().seconds(), 0)
            .single()
            .unwrap_or_else(Utc::now);

        // Get files changed
        let files_changed = self.get_commit_files(commit, repo)?;

        let is_merge = commit.parent_count() > 1;

        Ok(CommitInfo {
            sha,
            full_sha,
            message,
            full_message,
            author: author_name,
            author_email,
            timestamp,
            files_changed,
            is_merge,
        })
    }

    /// Get files changed in a commit
    fn get_commit_files(&self, commit: &Commit, repo: &Repository) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        if commit.parent_count() == 0 {
            // Initial commit - diff against empty tree
            let tree = commit.tree()?;
            let diff = repo.diff_tree_to_tree(None, Some(&tree), None)?;
            for delta in diff.deltas() {
                if let Some(path) = delta.new_file().path() {
                    files.push(path.to_path_buf());
                }
            }
        } else {
            // Normal commit - diff against first parent
            let parent = commit.parent(0)?;
            let parent_tree = parent.tree()?;
            let tree = commit.tree()?;

            let diff = repo.diff_tree_to_tree(Some(&parent_tree), Some(&tree), None)?;

            for delta in diff.deltas() {
                if let Some(path) = delta.new_file().path() {
                    files.push(path.to_path_buf());
                }
                if let Some(path) = delta.old_file().path() {
                    if !files.contains(&path.to_path_buf()) {
                        files.push(path.to_path_buf());
                    }
                }
            }
        }

        Ok(files)
    }

    /// Find files that frequently change together
    ///
    /// This analyzes git history to find pairs of files that are often modified
    /// in the same commit. This can reveal:
    /// - Test files and their implementations
    /// - Related components
    /// - Configuration files and code they configure
    pub fn find_cochange_patterns(
        &self,
        since: Option<DateTime<Utc>>,
        min_cooccurrence: f64,
    ) -> Result<Vec<FileRelationship>> {
        let repo = self.open_repo()?;

        // Track how often each pair of files changes together
        let mut cochange_counts: HashMap<(PathBuf, PathBuf), u32> = HashMap::new();
        let mut file_change_counts: HashMap<PathBuf, u32> = HashMap::new();
        let mut total_commits = 0u32;

        let mut revwalk = repo.revwalk()?;
        revwalk.push_head()?;
        revwalk.set_sorting(Sort::TIME)?;

        for oid in revwalk {
            let oid = oid?;
            let commit = repo.find_commit(oid)?;

            // Check if commit is after 'since' timestamp
            if let Some(since_time) = since {
                let commit_time = Utc
                    .timestamp_opt(commit.time().seconds(), 0)
                    .single()
                    .unwrap_or_else(Utc::now);

                if commit_time < since_time {
                    continue;
                }
            }

            // Skip merge commits
            if commit.parent_count() > 1 {
                continue;
            }

            let files = self.get_commit_files(&commit, &repo)?;

            // Filter to relevant file types
            let relevant_files: Vec<_> = files
                .into_iter()
                .filter(|f| self.is_relevant_file(f))
                .collect();

            if relevant_files.len() < 2 || relevant_files.len() > 50 {
                // Skip commits with too few or too many files
                continue;
            }

            total_commits += 1;

            // Count individual file changes
            for file in &relevant_files {
                *file_change_counts.entry(file.clone()).or_insert(0) += 1;
            }

            // Count co-occurrences for all pairs
            for i in 0..relevant_files.len() {
                for j in (i + 1)..relevant_files.len() {
                    let (a, b) = if relevant_files[i] < relevant_files[j] {
                        (relevant_files[i].clone(), relevant_files[j].clone())
                    } else {
                        (relevant_files[j].clone(), relevant_files[i].clone())
                    };
                    *cochange_counts.entry((a, b)).or_insert(0) += 1;
                }
            }
        }

        if total_commits == 0 {
            return Ok(vec![]);
        }

        // Convert to relationships, filtering by minimum co-occurrence
        let mut relationships = Vec::new();
        let mut id_counter = 0u32;

        for ((file_a, file_b), count) in cochange_counts {
            if count < 2 {
                continue; // Need at least 2 co-occurrences
            }

            // Calculate strength as Jaccard coefficient
            // strength = count(A&B) / (count(A) + count(B) - count(A&B))
            let count_a = file_change_counts.get(&file_a).copied().unwrap_or(0);
            let count_b = file_change_counts.get(&file_b).copied().unwrap_or(0);

            let union = count_a + count_b - count;
            let strength = if union > 0 {
                count as f64 / union as f64
            } else {
                0.0
            };

            if strength >= min_cooccurrence {
                id_counter += 1;
                relationships.push(FileRelationship {
                    id: format!("cochange-{}", id_counter),
                    files: vec![file_a, file_b],
                    relationship_type: RelationType::FrequentCochange,
                    strength,
                    description: format!(
                        "Changed together in {} of {} commits ({:.0}% co-occurrence)",
                        count,
                        total_commits,
                        strength * 100.0
                    ),
                    created_at: Utc::now(),
                    last_confirmed: Some(Utc::now()),
                    source: RelationshipSource::GitCochange,
                    observation_count: count,
                });
            }
        }

        // Sort by strength
        relationships.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));

        Ok(relationships)
    }

    /// Check if a file is relevant for analysis
    fn is_relevant_file(&self, path: &Path) -> bool {
        // Skip common non-source files
        let path_str = path.to_string_lossy();

        // Skip lock files, generated files, etc.
        if path_str.contains("Cargo.lock")
            || path_str.contains("package-lock.json")
            || path_str.contains("yarn.lock")
            || path_str.contains("pnpm-lock.yaml")
            || path_str.contains(".min.")
            || path_str.contains(".map")
            || path_str.contains("node_modules")
            || path_str.contains("target/")
            || path_str.contains("dist/")
            || path_str.contains("build/")
            || path_str.contains(".git/")
        {
            return false;
        }

        // Include source files
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy().to_lowercase();
            matches!(
                ext.as_str(),
                "rs" | "ts"
                    | "tsx"
                    | "js"
                    | "jsx"
                    | "py"
                    | "go"
                    | "java"
                    | "kt"
                    | "swift"
                    | "c"
                    | "cpp"
                    | "h"
                    | "hpp"
                    | "toml"
                    | "yaml"
                    | "yml"
                    | "json"
                    | "md"
                    | "sql"
            )
        } else {
            false
        }
    }

    /// Extract bug fixes from commit messages
    ///
    /// Looks for conventional commit messages like:
    /// - "fix: description"
    /// - "fix(scope): description"
    /// - "bugfix: description"
    /// - Messages containing "fixes #123"
    pub fn extract_bug_fixes(&self, since: Option<DateTime<Utc>>) -> Result<Vec<BugFix>> {
        let repo = self.open_repo()?;
        let mut bug_fixes = Vec::new();

        let mut revwalk = repo.revwalk()?;
        revwalk.push_head()?;
        revwalk.set_sorting(Sort::TIME)?;

        let mut id_counter = 0u32;

        for oid in revwalk {
            let oid = oid?;
            let commit = repo.find_commit(oid)?;

            // Check timestamp
            let commit_time = Utc
                .timestamp_opt(commit.time().seconds(), 0)
                .single()
                .unwrap_or_else(Utc::now);

            if let Some(since_time) = since {
                if commit_time < since_time {
                    continue;
                }
            }

            let message = commit.message().map(|m| m.to_string()).unwrap_or_default();

            // Check if this looks like a bug fix commit
            if let Some(bug_fix) =
                self.parse_bug_fix_commit(&message, &commit, &repo, &mut id_counter)?
            {
                bug_fixes.push(bug_fix);
            }
        }

        Ok(bug_fixes)
    }

    /// Parse a commit message to extract bug fix information
    fn parse_bug_fix_commit(
        &self,
        message: &str,
        commit: &Commit,
        repo: &Repository,
        counter: &mut u32,
    ) -> Result<Option<BugFix>> {
        let message_lower = message.to_lowercase();

        // Check for conventional commit fix patterns
        let is_fix = message_lower.starts_with("fix:")
            || message_lower.starts_with("fix(")
            || message_lower.starts_with("bugfix:")
            || message_lower.starts_with("bugfix(")
            || message_lower.starts_with("hotfix:")
            || message_lower.starts_with("hotfix(")
            || message_lower.contains("fixes #")
            || message_lower.contains("closes #")
            || message_lower.contains("resolves #");

        if !is_fix {
            return Ok(None);
        }

        *counter += 1;

        // Extract the description (first line, removing the prefix)
        let first_line = message.lines().next().unwrap_or("");
        let symptom = if let Some(colon_byte_pos) = first_line.find(':') {
            // Convert byte position to char position for safe slicing
            let colon_char_pos = first_line[..colon_byte_pos].chars().count();
            first_line.chars().skip(colon_char_pos + 1).collect::<String>().trim().to_string()
        } else {
            first_line.to_string()
        };

        // Try to extract root cause and solution from multi-line messages
        let mut root_cause = String::new();
        let mut solution = String::new();
        let mut issue_link = None;

        for line in message.lines().skip(1) {
            let line_lower = line.to_lowercase().trim().to_string();

            if line_lower.starts_with("cause:")
                || line_lower.starts_with("root cause:")
                || line_lower.starts_with("problem:")
            {
                root_cause = line
                    .split_once(':')
                    .map(|(_, v)| v.trim().to_string())
                    .unwrap_or_default();
            } else if line_lower.starts_with("solution:")
                || line_lower.starts_with("fix:")
                || line_lower.starts_with("fixed by:")
            {
                solution = line
                    .split_once(':')
                    .map(|(_, v)| v.trim().to_string())
                    .unwrap_or_default();
            } else if line_lower.contains("fixes #")
                || line_lower.contains("closes #")
                || line_lower.contains("resolves #")
            {
                // Extract issue number (using char-aware iteration)
                if let Some(hash_byte_pos) = line.find('#') {
                    // Convert byte position to char position for safe slicing
                    let hash_char_pos = line[..hash_byte_pos].chars().count();
                    let issue_num: String = line
                        .chars()
                        .skip(hash_char_pos + 1)
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    if !issue_num.is_empty() {
                        issue_link = Some(format!("#{}", issue_num));
                    }
                }
            }
        }

        // If no explicit root cause/solution, use the commit message
        if root_cause.is_empty() {
            root_cause = "See commit for details".to_string();
        }
        if solution.is_empty() {
            solution = symptom.clone();
        }

        // Determine severity from keywords
        let severity = if message_lower.contains("critical")
            || message_lower.contains("security")
            || message_lower.contains("crash")
        {
            BugSeverity::Critical
        } else if message_lower.contains("hotfix") || message_lower.contains("urgent") {
            BugSeverity::High
        } else if message_lower.contains("minor") || message_lower.contains("typo") {
            BugSeverity::Low
        } else {
            BugSeverity::Medium
        };

        let files_changed = self.get_commit_files(commit, repo)?;

        let bug_fix = BugFix {
            id: format!("bug-{}", counter),
            symptom,
            root_cause,
            solution,
            files_changed,
            commit_sha: commit.id().to_string(),
            created_at: Utc
                .timestamp_opt(commit.time().seconds(), 0)
                .single()
                .unwrap_or_else(Utc::now),
            issue_link,
            severity,
            discovered_by: commit.author().name().map(|s| s.to_string()),
            prevention_notes: None,
            tags: vec!["auto-detected".to_string()],
        };

        Ok(Some(bug_fix))
    }

    /// Analyze the full git history and return discovered knowledge
    pub fn analyze_history(&self, since: Option<DateTime<Utc>>) -> Result<HistoryAnalysis> {
        // Extract bug fixes
        let bug_fixes = self.extract_bug_fixes(since)?;

        // Find co-change patterns
        let file_relationships = self.find_cochange_patterns(since, 0.3)?;

        // Get recent activity summary
        let recent_commits = {
            let repo = self.open_repo()?;
            self.get_recent_commits(&repo, 50)?
        };

        // Calculate activity stats
        let mut author_counts: HashMap<String, u32> = HashMap::new();
        let mut file_counts: HashMap<PathBuf, u32> = HashMap::new();

        for commit in &recent_commits {
            *author_counts.entry(commit.author.clone()).or_insert(0) += 1;
            for file in &commit.files_changed {
                *file_counts.entry(file.clone()).or_insert(0) += 1;
            }
        }

        // Top contributors
        let mut top_contributors: Vec<_> = author_counts.into_iter().collect();
        top_contributors.sort_by(|a, b| b.1.cmp(&a.1));

        // Hot files (most frequently changed)
        let mut hot_files: Vec<_> = file_counts.into_iter().collect();
        hot_files.sort_by(|a, b| b.1.cmp(&a.1));

        Ok(HistoryAnalysis {
            bug_fixes,
            file_relationships,
            commit_count: recent_commits.len(),
            top_contributors: top_contributors.into_iter().take(5).collect(),
            hot_files: hot_files.into_iter().take(10).collect(),
            analyzed_since: since,
        })
    }

    /// Get files changed since a specific commit
    pub fn get_files_changed_since(&self, commit_sha: &str) -> Result<Vec<PathBuf>> {
        let repo = self.open_repo()?;

        let target_oid = repo.revparse_single(commit_sha)?.id();
        let head_commit = repo.head()?.peel_to_commit()?;
        let target_commit = repo.find_commit(target_oid)?;

        let head_tree = head_commit.tree()?;
        let target_tree = target_commit.tree()?;

        let diff = repo.diff_tree_to_tree(Some(&target_tree), Some(&head_tree), None)?;

        let mut files = Vec::new();
        for delta in diff.deltas() {
            if let Some(path) = delta.new_file().path() {
                files.push(path.to_path_buf());
            }
        }

        Ok(files)
    }

    /// Get blame information for a file
    pub fn get_file_blame(&self, file_path: &Path, line: u32) -> Result<Option<CommitInfo>> {
        let repo = self.open_repo()?;

        let blame = repo.blame_file(file_path, None)?;

        if let Some(hunk) = blame.get_line(line as usize) {
            let commit_id = hunk.final_commit_id();
            if let Ok(commit) = repo.find_commit(commit_id) {
                return Ok(Some(self.commit_to_info(&commit, &repo)?));
            }
        }

        Ok(None)
    }
}

// ============================================================================
// HISTORY ANALYSIS RESULT
// ============================================================================

/// Result of analyzing git history
#[derive(Debug)]
pub struct HistoryAnalysis {
    /// Bug fixes extracted from commits
    pub bug_fixes: Vec<BugFix>,
    /// File relationships discovered from co-change patterns
    pub file_relationships: Vec<FileRelationship>,
    /// Total commits analyzed
    pub commit_count: usize,
    /// Top contributors (author, commit count)
    pub top_contributors: Vec<(String, u32)>,
    /// Most frequently changed files (path, change count)
    pub hot_files: Vec<(PathBuf, u32)>,
    /// Time period analyzed from
    pub analyzed_since: Option<DateTime<Utc>>,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_repo() -> (TempDir, Repository) {
        let dir = TempDir::new().unwrap();
        let repo = Repository::init(dir.path()).unwrap();

        // Configure signature
        let sig = git2::Signature::now("Test User", "test@example.com").unwrap();

        // Create initial commit
        {
            let tree_id = {
                let mut index = repo.index().unwrap();
                index.write_tree().unwrap()
            };
            let tree = repo.find_tree(tree_id).unwrap();
            repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[])
                .unwrap();
        }

        (dir, repo)
    }

    #[test]
    fn test_git_analyzer_creation() {
        let (dir, _repo) = create_test_repo();
        let analyzer = GitAnalyzer::new(dir.path().to_path_buf());
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_get_current_context() {
        let (dir, _repo) = create_test_repo();
        let analyzer = GitAnalyzer::new(dir.path().to_path_buf()).unwrap();

        let context = analyzer.get_current_context().unwrap();
        assert!(context.has_commits);
        assert!(!context.head_commit.is_empty());
    }

    #[test]
    fn test_is_relevant_file() {
        let analyzer = GitAnalyzer {
            repo_path: PathBuf::from("."),
        };

        assert!(analyzer.is_relevant_file(Path::new("src/main.rs")));
        assert!(analyzer.is_relevant_file(Path::new("lib/utils.ts")));
        assert!(!analyzer.is_relevant_file(Path::new("Cargo.lock")));
        assert!(!analyzer.is_relevant_file(Path::new("node_modules/foo.js")));
        assert!(!analyzer.is_relevant_file(Path::new("target/debug/main")));
    }
}
