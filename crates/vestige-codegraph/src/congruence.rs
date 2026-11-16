#![forbid(unsafe_code)]

//! Self-contained deterministic Union-Find + Rewrite-Rule Congruence Engine.
//!
//! Unifies symbol IDs across renames and file moves ONLY when exact deterministic rules fire:
//! - `ExactShapeMatch`: same `SymbolKind` and identical normalized `shape_digest`
//! - `GitRenameAndBodyMatch`: explicit git rename (`old_path -> new_path`) with identical `shape_digest`
//!
//! Zero similarity scores, zero floating-point heuristics, zero fuzzy matching.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::parse::{Symbol, SymbolId, SymbolKind};

/// Exact rule explaining why two symbol IDs belong to the same congruence class.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "rule", rename_all = "snake_case")]
pub enum CongruenceRule {
    ExactShapeMatch {
        kind: SymbolKind,
        shape_digest: [u8; 32],
    },
    GitRenameAndBodyMatch {
        old_path: String,
        new_path: String,
        shape_digest: [u8; 32],
    },
}

/// Recorded witness edge produced when a rewrite rule merges two symbols.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MergeWitness {
    pub from_id: SymbolId,
    pub to_id: SymbolId,
    pub rule: CongruenceRule,
}

/// Deterministic Union-Find congruence closure engine.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CongruenceEngine {
    parent: BTreeMap<SymbolId, SymbolId>,
    by_kind_shape: BTreeMap<(SymbolKind, [u8; 32]), Vec<SymbolId>>,
    by_path_shape: BTreeMap<(String, SymbolKind, [u8; 32]), SymbolId>,
    witnesses: Vec<MergeWitness>,
}

impl CongruenceEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `symbol` and fires applicable deterministic rewrite rules against known symbols
    /// and optional `git_rename_old_path`.
    pub fn register_symbol(
        &mut self,
        symbol: &Symbol,
        git_rename_old_path: Option<&str>,
    ) -> Option<MergeWitness> {
        self.parent.entry(symbol.id).or_insert(symbol.id);

        let mut recorded_witness: Option<MergeWitness> = None;

        // Rule 1: GitRenameAndBodyMatch if file was explicitly moved/renamed and body shape matches.
        if let Some(old_path) = git_rename_old_path {
            if let Some(&prev_id) =
                self.by_path_shape
                    .get(&(old_path.to_owned(), symbol.kind, symbol.shape_digest))
            {
                let rule = CongruenceRule::GitRenameAndBodyMatch {
                    old_path: old_path.to_owned(),
                    new_path: symbol.path.clone(),
                    shape_digest: symbol.shape_digest,
                };
                if let Some(w) = self.union(prev_id, symbol.id, rule) {
                    recorded_witness = Some(w);
                }
            }
        }

        // Rule 2: ExactShapeMatch across renames/refactors with identical normalized AST shape.
        // Copy the peer out before `union` so the map borrow ends. `entry()` held
        // `&mut self` across the `self.union` call and failed E0499.
        let shape_key = (symbol.kind, symbol.shape_digest);
        let canonical_peer = self
            .by_kind_shape
            .get(&shape_key)
            .and_then(|ids| ids.first().copied());
        if let Some(canonical_peer) = canonical_peer {
            if canonical_peer != symbol.id && recorded_witness.is_none() {
                let rule = CongruenceRule::ExactShapeMatch {
                    kind: symbol.kind,
                    shape_digest: symbol.shape_digest,
                };
                if let Some(w) = self.union(canonical_peer, symbol.id, rule) {
                    recorded_witness = Some(w);
                }
            }
        }
        let existing_ids = self.by_kind_shape.entry(shape_key).or_default();
        if !existing_ids.contains(&symbol.id) {
            existing_ids.push(symbol.id);
        }

        self.by_path_shape.insert(
            (symbol.path.clone(), symbol.kind, symbol.shape_digest),
            symbol.id,
        );

        recorded_witness
    }

    pub fn find(&self, mut id: SymbolId) -> SymbolId {
        while let Some(&p) = self.parent.get(&id) {
            if p == id {
                return id;
            }
            id = p;
        }
        id
    }

    /// Returns all symbol IDs unified into the same equivalence class as `id`, sorted deterministically.
    pub fn equivalence_class(&self, id: SymbolId) -> BTreeSet<SymbolId> {
        let root = self.find(id);
        let mut out = BTreeSet::new();
        out.insert(id);
        for &sym_id in self.parent.keys() {
            if self.find(sym_id) == root {
                out.insert(sym_id);
            }
        }
        out
    }

    /// Returns the explicit rewrite witness rule connecting two symbols if directly merged.
    pub fn rule_for_symbol(&self, id: SymbolId) -> Option<CongruenceRule> {
        self.witnesses
            .iter()
            .rev()
            .find(|w| w.to_id == id || w.from_id == id)
            .map(|w| w.rule.clone())
    }

    pub fn witnesses(&self) -> &[MergeWitness] {
        &self.witnesses
    }

    fn union(&mut self, a: SymbolId, b: SymbolId, rule: CongruenceRule) -> Option<MergeWitness> {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return None;
        }
        // Deterministically choose the lexicographically smaller root ID as representative.
        let (rep, child) = if ra < rb { (ra, rb) } else { (rb, ra) };
        self.parent.insert(child, rep);
        let witness = MergeWitness {
            from_id: a,
            to_id: b,
            rule,
        };
        self.witnesses.push(witness.clone());
        Some(witness)
    }
}
