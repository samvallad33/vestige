//! The `git` module for builds without the `codebase-git` feature.
//!
//! libgit2 drags OpenSSL and libssh2 into every build, and on some targets
//! (Android/Termux, #145) that is the difference between a binary and no
//! binary. This stand-in keeps the public surface of `git.rs` so every caller
//! compiles unchanged, and answers each history question with
//! [`GitError::Unavailable`] instead of pretending to know. Callers that already
//! treat git as optional (`ContextCapture`) swallow that error; callers that need
//! history get an honest reason they can show the user.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};

pub use super::git_types::{CommitInfo, GitContext, HistoryAnalysis};
use super::types::{BugFix, FileRelationship};

/// Errors that can occur during git analysis.
#[derive(Debug, thiserror::Error)]
pub enum GitError {
    #[error("git history is not available in this build: {0}")]
    Unavailable(&'static str),
    #[error("Repository not found at: {0}")]
    NotFound(PathBuf),
    #[error("Invalid path: {0}")]
    InvalidPath(String),
    #[error("No commits found")]
    NoCommits,
}

pub type Result<T> = std::result::Result<T, GitError>;

const UNAVAILABLE: &str = "compiled without the codebase-git feature";

/// Stand-in for the libgit2-backed analyzer.
pub struct GitAnalyzer {
    repo_path: PathBuf,
}

impl GitAnalyzer {
    /// Succeeds for any path: the rest of codebase memory (context capture,
    /// patterns, relationships, anchors) does not need git to work, so a
    /// missing analyzer must not take the whole codebase tool down with it.
    pub fn new(repo_path: PathBuf) -> Result<Self> {
        Ok(Self { repo_path })
    }

    /// The path this analyzer was created for.
    pub fn repo_path(&self) -> &Path {
        &self.repo_path
    }

    /// Current branch, HEAD and working-tree state: unavailable in this build.
    pub fn get_current_context(&self) -> Result<GitContext> {
        Err(GitError::Unavailable(UNAVAILABLE))
    }

    /// Co-change patterns from history: unavailable in this build.
    pub fn find_cochange_patterns(
        &self,
        _since: Option<DateTime<Utc>>,
        _min_cooccurrence: f64,
    ) -> Result<Vec<FileRelationship>> {
        Err(GitError::Unavailable(UNAVAILABLE))
    }

    /// Bug fixes from commit messages: unavailable in this build.
    pub fn extract_bug_fixes(&self, _since: Option<DateTime<Utc>>) -> Result<Vec<BugFix>> {
        Err(GitError::Unavailable(UNAVAILABLE))
    }

    /// Full history analysis: unavailable in this build.
    pub fn analyze_history(&self, _since: Option<DateTime<Utc>>) -> Result<HistoryAnalysis> {
        Err(GitError::Unavailable(UNAVAILABLE))
    }

    /// Files changed since a commit: unavailable in this build.
    pub fn get_files_changed_since(&self, _commit_sha: &str) -> Result<Vec<PathBuf>> {
        Err(GitError::Unavailable(UNAVAILABLE))
    }

    /// Blame for one line: unavailable in this build.
    pub fn get_file_blame(&self, _file_path: &Path, _line: u32) -> Result<Option<CommitInfo>> {
        Err(GitError::Unavailable(UNAVAILABLE))
    }
}
