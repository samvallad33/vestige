//! Data types shared by the git analyzer and its no-git stand-in.
//!
//! Kept apart from `git.rs` so a build without the `codebase-git` feature (no
//! libgit2, no OpenSSL, no libssh2) still has every type the rest of the crate
//! and the MCP tools name. Only the analysis behind them goes missing there.

use chrono::{DateTime, Utc};
use std::path::PathBuf;

use super::types::{BugFix, FileRelationship};


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
