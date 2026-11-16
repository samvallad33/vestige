#![forbid(unsafe_code)]

//! Deterministic `vestige why path:line` provenance resolver.
//!
//! Traces a 1-based line back to its enclosing symbol, commits, hunks, and
//! congruence rules. Returns `unresolved_reason` for untracked or unrecorded lines;
//! never guesses.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::congruence::CongruenceRule;
use crate::git::Touched;
use crate::graph::CodeGraph;
use crate::parse::Symbol;

/// A single commit hop in a `WhyChain` trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitHop {
    pub sha: String,
    pub hunk: Touched,
    pub rule: Option<CongruenceRule>,
}

/// Complete deterministic trace returned by `CodeGraph::why(path, line)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhyChain {
    pub path: String,
    pub line: u32,
    pub symbol: Option<Symbol>,
    pub commits: Vec<CommitHop>,
    pub unresolved_reason: Option<String>,
}

impl CodeGraph {
    /// Traces `(path, line)` (1-based) through commits, hunks, and congruence classes.
    pub fn why(&self, path: &str, line: u32) -> WhyChain {
        if line == 0 {
            return WhyChain {
                path: path.to_owned(),
                line,
                symbol: None,
                commits: Vec::new(),
                unresolved_reason: Some("line numbers are 1-based; line 0 is invalid".to_owned()),
            };
        }

        let symbol = self.symbol_at_line(path, line).cloned();
        let mut hops_by_commit: BTreeMap<usize, CommitHop> = BTreeMap::new();

        if let Some(ref sym) = symbol {
            let eq_class = self.congruence.equivalence_class(sym.id);
            for member_id in eq_class {
                if let Some(occurrences) = self.symbol_occurrences.get(&member_id) {
                    for occ in occurrences {
                        let commit = &self.commits[occ.commit_index];
                        // Only a hunk that overlaps this symbol's lines. A later edit
                        // elsewhere in the same file must not attach to an unchanged symbol.
                        let matching_hunk = commit
                            .touched
                            .iter()
                            .find(|t| {
                                t.path == occ.symbol.path
                                    && t.new_range
                                        .is_some_and(|r| r.overlaps(occ.symbol.line_range))
                            })
                            .cloned();

                        if let Some(hunk) = matching_hunk {
                            let rule = occ
                                .rule_applied
                                .clone()
                                .or_else(|| self.congruence.rule_for_symbol(occ.symbol.id));
                            hops_by_commit
                                .entry(occ.commit_index)
                                .or_insert(CommitHop {
                                    sha: occ.commit_sha.clone(),
                                    hunk,
                                    rule,
                                });
                        }
                    }
                }
            }
        } else if let Some(path_hunks) = self.hunks_by_path.get(path) {
            for (commit_idx, hunk) in path_hunks {
                if hunk.new_range.is_some_and(|r| r.contains_line(line)) {
                    hops_by_commit.entry(*commit_idx).or_insert(CommitHop {
                        sha: hunk.commit_sha.clone(),
                        hunk: hunk.clone(),
                        rule: None,
                    });
                }
            }
        }

        if hops_by_commit.is_empty() {
            let reason = if !self.hunks_by_path.contains_key(path)
                && !self.head_symbols_by_path.contains_key(path)
            {
                format!("path '{path}' is untracked in indexed git history")
            } else {
                format!("no recorded commit hunk or symbol history found for {path}:{line}")
            };

            return WhyChain {
                path: path.to_owned(),
                line,
                symbol,
                commits: Vec::new(),
                unresolved_reason: Some(reason),
            };
        }

        // Order newest-to-oldest deterministically by reverse `commit_index`.
        let commits: Vec<CommitHop> = hops_by_commit.into_values().rev().collect();
        WhyChain {
            path: path.to_owned(),
            line,
            symbol,
            commits,
            unresolved_reason: None,
        }
    }

    fn symbol_at_line(&self, path: &str, line: u32) -> Option<&Symbol> {
        let syms = self.head_symbols_by_path.get(path)?;
        // Choose the narrowest enclosing symbol deterministically.
        syms.iter()
            .filter(|s| line >= s.line_range.0 && line <= s.line_range.1)
            .min_by_key(|s| (s.line_range.1 - s.line_range.0, s.id))
    }
}
