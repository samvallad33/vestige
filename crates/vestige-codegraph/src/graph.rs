#![forbid(unsafe_code)]

//! Deterministic `CodeGraph` store indexing git commits, AST symbols, call edges,
//! line-level hunks, and congruence classes with a canonical BLAKE3 `digest()`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::congruence::{CongruenceEngine, CongruenceRule};
use crate::error::CodeGraphError;
use crate::git::{walk_repository_history, CommitRecord, GitWalkOptions, Touched};
use crate::parse::{CallEdge, IncrementalParser, Symbol, SymbolId};

/// A symbol occurrence recorded at a specific commit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolOccurrence {
    pub commit_index: usize,
    pub commit_sha: String,
    pub symbol: Symbol,
    pub rule_applied: Option<CongruenceRule>,
}

/// Deterministic AST + Git provenance graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeGraph {
    pub commits: Vec<CommitRecord>,
    pub head_symbols_by_path: BTreeMap<String, Vec<Symbol>>,
    pub symbol_occurrences: BTreeMap<SymbolId, Vec<SymbolOccurrence>>,
    pub hunks_by_path: BTreeMap<String, Vec<(usize, Touched)>>,
    pub call_edges: Vec<CallEdge>,
    pub congruence: CongruenceEngine,
    digest: [u8; 32],
}

impl CodeGraph {
    /// Indexes a git repository read-only via `gix`, `tree-sitter`, and `CongruenceEngine`.
    pub fn index_repo(repo_path: &Path, opts: GitWalkOptions) -> Result<Self, CodeGraphError> {
        let commits = walk_repository_history(repo_path, opts)?;
        let mut parser = IncrementalParser::new();
        let mut head_symbols_by_path: BTreeMap<String, Vec<Symbol>> = BTreeMap::new();
        let mut symbol_occurrences: BTreeMap<SymbolId, Vec<SymbolOccurrence>> = BTreeMap::new();
        let mut hunks_by_path: BTreeMap<String, Vec<(usize, Touched)>> = BTreeMap::new();
        let mut all_call_edges: Vec<CallEdge> = Vec::new();
        let mut congruence = CongruenceEngine::new();

        for (commit_index, commit) in commits.iter().enumerate() {
            for hunk in &commit.touched {
                hunks_by_path
                    .entry(hunk.path.clone())
                    .or_default()
                    .push((commit_index, hunk.clone()));
            }

            for snap in &commit.snapshots {
                if let Some(old_path) = &snap.old_path {
                    if old_path != &snap.path {
                        head_symbols_by_path.remove(old_path);
                    }
                }

                let Some(content) = &snap.content else {
                    head_symbols_by_path.remove(&snap.path);
                    continue;
                };

                let parsed = parser.parse_file(&snap.path, content, None)?;
                let rename_old = snap
                    .old_path
                    .as_deref()
                    .filter(|old_p| *old_p != snap.path.as_str());

                for sym in &parsed.symbols {
                    let witness = congruence.register_symbol(sym, rename_old);
                    symbol_occurrences
                        .entry(sym.id)
                        .or_default()
                        .push(SymbolOccurrence {
                            commit_index,
                            commit_sha: commit.sha.clone(),
                            symbol: sym.clone(),
                            rule_applied: witness.map(|w| w.rule),
                        });
                }

                all_call_edges.extend(parsed.call_edges);
                head_symbols_by_path.insert(snap.path.clone(), parsed.symbols);
            }
        }

        all_call_edges.sort();
        all_call_edges.dedup();

        let mut graph = Self {
            commits,
            head_symbols_by_path,
            symbol_occurrences,
            hunks_by_path,
            call_edges: all_call_edges,
            congruence,
            digest: [0u8; 32],
        };
        graph.digest = compute_graph_digest(&graph);
        Ok(graph)
    }

    #[inline]
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

fn compute_graph_digest(graph: &CodeGraph) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vestige.codegraph.v1\0");
    hasher.update(&(graph.commits.len() as u64).to_le_bytes());

    for commit in &graph.commits {
        hasher.update(commit.sha.as_bytes());
        hasher.update(b"\0");
        hasher.update(commit.author_email.as_bytes());
        hasher.update(b"\0");
        for hunk in &commit.touched {
            hasher.update(hunk.path.as_bytes());
            hasher.update(b"\0");
            if let Some(r) = hunk.new_range {
                hasher.update(&r.start_line.to_le_bytes());
                hasher.update(&r.end_line.to_le_bytes());
            }
        }
    }

    for (path, symbols) in &graph.head_symbols_by_path {
        hasher.update(path.as_bytes());
        hasher.update(b"\0");
        for sym in symbols {
            hasher.update(&sym.id);
            hasher.update(&sym.shape_digest);
        }
    }

    for witness in graph.congruence.witnesses() {
        hasher.update(&witness.from_id);
        hasher.update(&witness.to_id);
    }

    *hasher.finalize().as_bytes()
}
