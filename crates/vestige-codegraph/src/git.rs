#![forbid(unsafe_code)]

//! Pure-Rust read-only `gix` repository walker and deterministic line-hunk emitter.
//!
//! Never writes to the repository. Author emails are protected behind `hash_pii`.

use std::collections::BTreeSet;
use std::ops::ControlFlow;
use std::path::Path;

use gix::object::tree::diff::Change;
use serde::{Deserialize, Serialize};

use crate::error::CodeGraphError;

/// Controls read-only git commit walking and PII hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GitWalkOptions {
    pub hash_pii: bool,
    pub max_commits: Option<usize>,
}

/// 1-based inclusive line range `[start_line, end_line]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LineRange {
    pub start_line: u32,
    pub end_line: u32,
}

impl LineRange {
    #[inline]
    pub fn contains_line(self, line: u32) -> bool {
        line >= self.start_line && line <= self.end_line
    }

    #[inline]
    pub fn overlaps(self, other: (u32, u32)) -> bool {
        self.start_line <= other.1 && other.0 <= self.end_line
    }
}

/// A line-level diff hunk touched by a commit.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Touched {
    pub commit_sha: String,
    pub path: String,
    pub old_path: Option<String>,
    pub old_range: Option<LineRange>,
    pub new_range: Option<LineRange>,
    pub is_rename: bool,
}

/// Snapshot of a changed source file at a specific commit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitFileSnapshot {
    pub path: String,
    pub old_path: Option<String>,
    pub content: Option<String>,
}

/// Read-only metadata and diff hunks extracted from one git commit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitRecord {
    pub sha: String,
    pub parent_shas: Vec<String>,
    pub author_name: String,
    pub author_email: String,
    pub timestamp_s: i64,
    pub message: String,
    pub touched: Vec<Touched>,
    pub snapshots: Vec<CommitFileSnapshot>,
}

/// Walks commit ancestry from `HEAD` (oldest-to-newest replay order) using pure-Rust `gix`.
pub fn walk_repository_history(
    repo_path: &Path,
    opts: GitWalkOptions,
) -> Result<Vec<CommitRecord>, CodeGraphError> {
    let repo = gix::open(repo_path).map_err(|e| CodeGraphError::Git(e.to_string()))?;
    let head_id = match repo.head_id() {
        Ok(id) => id,
        Err(_) => return Ok(Vec::new()),
    };

    let walk = repo
        .rev_walk([head_id])
        .all()
        .map_err(|e| CodeGraphError::Git(e.to_string()))?;

    let mut commit_ids = Vec::new();
    for info in walk {
        let info = info.map_err(|e| CodeGraphError::Git(e.to_string()))?;
        commit_ids.push(info.id);
        if let Some(limit) = opts.max_commits {
            if commit_ids.len() >= limit {
                break;
            }
        }
    }

    // Reverse to chronological (ancestor -> HEAD) order for deterministic replay.
    commit_ids.reverse();

    let mut records = Vec::with_capacity(commit_ids.len());
    let mut visited = BTreeSet::new();

    for oid in commit_ids {
        if !visited.insert(oid) {
            continue;
        }
        let commit = repo
            .find_object(oid)
            .map_err(|e| CodeGraphError::Git(e.to_string()))?
            .try_into_commit()
            .map_err(|e| CodeGraphError::Git(e.to_string()))?;

        let sha = oid.to_hex().to_string();
        let parent_shas: Vec<String> = commit
            .parent_ids()
            .map(|pid| pid.to_hex().to_string())
            .collect();

        let author = commit
            .author()
            .map_err(|e| CodeGraphError::Git(e.to_string()))?;
        let author_name = author.name.to_string();
        let raw_email = author.email.to_string();
        let author_email = if opts.hash_pii {
            hash_email_pii(&raw_email)
        } else {
            raw_email
        };
        let timestamp_s = author
            .time()
            .map(|t| t.seconds)
            .unwrap_or(0);
        let message = commit
            .message_raw()
            .map(|m| m.to_string())
            .unwrap_or_default();

        let current_tree = commit
            .tree()
            .map_err(|e| CodeGraphError::Git(e.to_string()))?;

        let parent_tree = match commit.parent_ids().next() {
            Some(pid) => repo
                .find_object(pid)
                .map_err(|e| CodeGraphError::Git(e.to_string()))?
                .try_into_commit()
                .map_err(|e| CodeGraphError::Git(e.to_string()))?
                .tree()
                .map_err(|e| CodeGraphError::Git(e.to_string()))?,
            None => repo.empty_tree(),
        };

        let mut changes = Vec::new();
        let mut platform = parent_tree
            .changes()
            .map_err(|e| CodeGraphError::Git(e.to_string()))?;
        platform.options(|diff_opts| {
            diff_opts.track_path();
            diff_opts.track_rewrites(Some(Default::default()));
        });

        platform
            .for_each_to_obtain_tree(&current_tree, |change| {
                match change {
                    Change::Addition {
                        location,
                        entry_mode,
                        id,
                        ..
                    } if entry_mode.is_no_tree() => {
                        changes.push(RawTreeChange {
                            path: String::from_utf8_lossy(location.as_ref()).into_owned(),
                            old_path: None,
                            old_id: None,
                            new_id: Some(id.detach()),
                            is_rename: false,
                        });
                    }
                    Change::Deletion {
                        location,
                        entry_mode,
                        id,
                        ..
                    } if entry_mode.is_no_tree() => {
                        changes.push(RawTreeChange {
                            path: String::from_utf8_lossy(location.as_ref()).into_owned(),
                            old_path: Some(String::from_utf8_lossy(location.as_ref()).into_owned()),
                            old_id: Some(id.detach()),
                            new_id: None,
                            is_rename: false,
                        });
                    }
                    Change::Modification {
                        location,
                        previous_id,
                        entry_mode,
                        id,
                        ..
                    } if entry_mode.is_no_tree() => {
                        changes.push(RawTreeChange {
                            path: String::from_utf8_lossy(location.as_ref()).into_owned(),
                            old_path: Some(String::from_utf8_lossy(location.as_ref()).into_owned()),
                            old_id: Some(previous_id.detach()),
                            new_id: Some(id.detach()),
                            is_rename: false,
                        });
                    }
                    Change::Rewrite {
                        source_location,
                        source_id,
                        entry_mode,
                        id,
                        location,
                        ..
                    } if entry_mode.is_no_tree() => {
                        changes.push(RawTreeChange {
                            path: String::from_utf8_lossy(location.as_ref()).into_owned(),
                            old_path: Some(
                                String::from_utf8_lossy(source_location.as_ref()).into_owned(),
                            ),
                            old_id: Some(source_id.detach()),
                            new_id: Some(id.detach()),
                            is_rename: true,
                        });
                    }
                    _ => {}
                }
                Ok::<_, CodeGraphError>(ControlFlow::Continue(()))
            })
            .map_err(|e| CodeGraphError::Git(e.to_string()))?;

        changes.sort_by(|a, b| a.path.cmp(&b.path).then_with(|| a.old_path.cmp(&b.old_path)));

        let mut touched = Vec::new();
        let mut snapshots = Vec::new();

        for ch in changes {
            let old_text = match ch.old_id {
                Some(old_oid) => read_blob_utf8(&repo, old_oid)?,
                None => None,
            };
            let new_text = match ch.new_id {
                Some(new_oid) => read_blob_utf8(&repo, new_oid)?,
                None => None,
            };

            let hunks = compute_line_hunks(
                old_text.as_deref().unwrap_or(""),
                new_text.as_deref().unwrap_or(""),
                ch.old_id.is_some(),
                ch.new_id.is_some(),
            );

            if hunks.is_empty() && ch.is_rename {
                let total_lines = new_text.as_deref().map(count_lines).unwrap_or(1).max(1);
                touched.push(Touched {
                    commit_sha: sha.clone(),
                    path: ch.path.clone(),
                    old_path: ch.old_path.clone(),
                    old_range: Some(LineRange {
                        start_line: 1,
                        end_line: total_lines,
                    }),
                    new_range: Some(LineRange {
                        start_line: 1,
                        end_line: total_lines,
                    }),
                    is_rename: true,
                });
            } else {
                for (old_range, new_range) in hunks {
                    touched.push(Touched {
                        commit_sha: sha.clone(),
                        path: ch.path.clone(),
                        old_path: ch.old_path.clone(),
                        old_range,
                        new_range,
                        is_rename: ch.is_rename,
                    });
                }
            }

            snapshots.push(CommitFileSnapshot {
                path: ch.path,
                old_path: ch.old_path,
                content: new_text,
            });
        }

        records.push(CommitRecord {
            sha,
            parent_shas,
            author_name,
            author_email,
            timestamp_s,
            message,
            touched,
            snapshots,
        });
    }

    Ok(records)
}

struct RawTreeChange {
    path: String,
    old_path: Option<String>,
    old_id: Option<gix::ObjectId>,
    new_id: Option<gix::ObjectId>,
    is_rename: bool,
}

fn read_blob_utf8(
    repo: &gix::Repository,
    oid: gix::ObjectId,
) -> Result<Option<String>, CodeGraphError> {
    let blob = repo
        .find_blob(oid)
        .map_err(|e| CodeGraphError::Git(e.to_string()))?;
    Ok(std::str::from_utf8(&blob.data).ok().map(ToOwned::to_owned))
}

/// Computes BLAKE3 digest of an author email when `hash_pii` is enabled.
pub fn hash_email_pii(email: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vestige.codegraph.pii.v1\0");
    hasher.update(email.trim().to_ascii_lowercase().as_bytes());
    let digest = hasher.finalize();
    format!("blake3:{}", digest.to_hex())
}

fn count_lines(text: &str) -> u32 {
    if text.is_empty() {
        0
    } else {
        text.lines().count() as u32
    }
}

/// Deterministic LCS-based line hunk extractor producing 1-based inclusive `(old_range, new_range)`.
pub fn compute_line_hunks(
    old_text: &str,
    new_text: &str,
    has_old: bool,
    has_new: bool,
) -> Vec<(Option<LineRange>, Option<LineRange>)> {
    let old_lines: Vec<&str> = if has_old {
        old_text.lines().collect()
    } else {
        Vec::new()
    };
    let new_lines: Vec<&str> = if has_new {
        new_text.lines().collect()
    } else {
        Vec::new()
    };

    if !has_old && has_new && !new_lines.is_empty() {
        return vec![(
            None,
            Some(LineRange {
                start_line: 1,
                end_line: new_lines.len() as u32,
            }),
        )];
    }
    if has_old && !has_new && !old_lines.is_empty() {
        return vec![(
            Some(LineRange {
                start_line: 1,
                end_line: old_lines.len() as u32,
            }),
            None,
        )];
    }

    let m = old_lines.len();
    let n = new_lines.len();
    if m == 0 && n == 0 {
        return Vec::new();
    }

    // Trim matching prefix/suffix first to keep LCS DP tiny on real commits.
    let mut prefix = 0usize;
    while prefix < m && prefix < n && old_lines[prefix] == new_lines[prefix] {
        prefix += 1;
    }
    let mut suffix = 0usize;
    while suffix < (m - prefix)
        && suffix < (n - prefix)
        && old_lines[m - 1 - suffix] == new_lines[n - 1 - suffix]
    {
        suffix += 1;
    }

    let old_mid = &old_lines[prefix..(m - suffix)];
    let new_mid = &new_lines[prefix..(n - suffix)];
    if old_mid.is_empty() && new_mid.is_empty() {
        return Vec::new();
    }

    // Fallback to single bounding hunk if mid diff is huge to guarantee fast bounded time.
    if old_mid.len().saturating_mul(new_mid.len()) > 250_000 {
        let old_r = if old_mid.is_empty() {
            None
        } else {
            Some(LineRange {
                start_line: prefix as u32 + 1,
                end_line: (m - suffix) as u32,
            })
        };
        let new_r = if new_mid.is_empty() {
            None
        } else {
            Some(LineRange {
                start_line: prefix as u32 + 1,
                end_line: (n - suffix) as u32,
            })
        };
        return vec![(old_r, new_r)];
    }

    let mm = old_mid.len();
    let nn = new_mid.len();
    let mut dp = vec![vec![0usize; nn + 1]; mm + 1];
    for i in (0..mm).rev() {
        for j in (0..nn).rev() {
            dp[i][j] = if old_mid[i] == new_mid[j] {
                1 + dp[i + 1][j + 1]
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }

    let mut hunks = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;
    let mut old_hunk_start: Option<u32> = None;
    let mut old_hunk_end: u32 = 0;
    let mut new_hunk_start: Option<u32> = None;
    let mut new_hunk_end: u32 = 0;

    let flush = |hunks: &mut Vec<(Option<LineRange>, Option<LineRange>)>,
                 o_s: &mut Option<u32>,
                 o_e: u32,
                 n_s: &mut Option<u32>,
                 n_e: u32| {
        let o_r = o_s.take().map(|start| LineRange {
            start_line: start,
            end_line: o_e,
        });
        let n_r = n_s.take().map(|start| LineRange {
            start_line: start,
            end_line: n_e,
        });
        if o_r.is_some() || n_r.is_some() {
            hunks.push((o_r, n_r));
        }
    };

    while i < mm || j < nn {
        if i < mm && j < nn && old_mid[i] == new_mid[j] {
            flush(
                &mut hunks,
                &mut old_hunk_start,
                old_hunk_end,
                &mut new_hunk_start,
                new_hunk_end,
            );
            i += 1;
            j += 1;
        } else if i < mm && (j == nn || dp[i + 1][j] >= dp[i][j + 1]) {
            let line_no = (prefix + i) as u32 + 1;
            if old_hunk_start.is_none() {
                old_hunk_start = Some(line_no);
            }
            old_hunk_end = line_no;
            i += 1;
        } else {
            let line_no = (prefix + j) as u32 + 1;
            if new_hunk_start.is_none() {
                new_hunk_start = Some(line_no);
            }
            new_hunk_end = line_no;
            j += 1;
        }
    }

    flush(
        &mut hunks,
        &mut old_hunk_start,
        old_hunk_end,
        &mut new_hunk_start,
        new_hunk_end,
    );

    hunks
}
