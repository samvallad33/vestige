//! # Merge / Supersede Controls (Phase 3)
//!
//! Diff-previewed, confidence-gated, reversible, self-explaining combine /
//! dedupe / supersede operations on a never-delete (bitemporal) store.
//!
//! This module holds the **pure** logic: candidate scoring, two-threshold
//! classification, and the plan / operation data model. The actual persistence
//! (writing plans, applying them, recording the reversible operation log, and
//! bitemporally invalidating superseded nodes) lives in
//! [`crate::storage`]. Keeping the math here makes it unit-testable without a
//! database.
//!
//! ## Design north star
//!
//! Every combine/dedupe/supersede operation is:
//!
//! - **diff-previewed** — `plan_merge` / `plan_supersede` produce a [`MergePlan`]
//!   you can inspect before anything mutates,
//! - **confidence-gated** — a Fellegi-Sunter two-threshold score classifies each
//!   candidate as match / possible-match / non-match,
//! - **reversible** — every applied plan records a [`MergeOperation`] with an
//!   undo payload (the "git reflog for your agent's memory"),
//! - **self-explaining** — each candidate carries the [`MatchSignals`] that
//!   explain *why* the memories combined,
//! - **opt-in, never silent** — the default is preview/review, never auto-mutate,
//! - **audit-preserving** — superseding stamps `valid_until` and keeps the old
//!   node queryable (Graphiti-style "invalidate, don't delete").
//!
//! ## Why Fellegi-Sunter
//!
//! Pure hashing under-merges (misses paraphrases); aggressive LLM merging
//! over-merges and destroys the audit trail. Fellegi-Sunter record linkage uses
//! **two** thresholds to carve the score space into three zones, so the
//! borderline "possible match" cases are surfaced for review instead of being
//! force-decided. We reuse the embedding cosine similarity already in the store
//! plus cheap lexical signals (tag overlap, token Jaccard) as the match weight.

use serde::{Deserialize, Serialize};

// ============================================================================
// CONSTANTS — the two Fellegi-Sunter thresholds
// ============================================================================

/// Above this combined score → automatic-eligible "match".
pub const DEFAULT_MATCH_THRESHOLD: f32 = 0.86;

/// Between the two thresholds → "possible match", surfaced for review.
/// Below this → "non-match" (never offered).
pub const DEFAULT_POSSIBLE_THRESHOLD: f32 = 0.72;

/// Weight of embedding cosine similarity in the combined score.
const W_EMBEDDING: f32 = 0.70;
/// Weight of tag overlap (Jaccard) in the combined score.
const W_TAGS: f32 = 0.15;
/// Weight of content token overlap (Jaccard) in the combined score.
const W_TOKENS: f32 = 0.15;

// ============================================================================
// CLASSIFICATION
// ============================================================================

/// Fellegi-Sunter three-way classification of a candidate pair/cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchClass {
    /// Score ≥ match threshold — strong duplicate, auto-merge eligible.
    Match,
    /// Between thresholds — surfaced for human/agent review, never auto-applied.
    Possible,
    /// Below the possible threshold — not offered as a candidate.
    NonMatch,
}

impl MatchClass {
    /// String label used in tool output and the `classification` column.
    pub fn as_str(&self) -> &'static str {
        match self {
            MatchClass::Match => "match",
            MatchClass::Possible => "possible",
            MatchClass::NonMatch => "non_match",
        }
    }
}

/// Per-merge-policy thresholds. Wired to `vestige.toml` when present, else the
/// defaults above. `auto_apply` gates whether `Match`-class candidates may be
/// applied without an explicit preview step (default: false — never silent).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MergePolicy {
    /// Score ≥ this → `Match`.
    pub match_threshold: f32,
    /// Score in `[possible_threshold, match_threshold)` → `Possible`.
    pub possible_threshold: f32,
    /// If true, `Match`-class candidates may be auto-applied. Default false:
    /// the product promise is review/preview, not silent mutation.
    pub auto_apply: bool,
}

impl Default for MergePolicy {
    fn default() -> Self {
        Self {
            match_threshold: DEFAULT_MATCH_THRESHOLD,
            possible_threshold: DEFAULT_POSSIBLE_THRESHOLD,
            auto_apply: false,
        }
    }
}

impl MergePolicy {
    /// Build a policy, clamping thresholds into `[0,1]` and ensuring
    /// `possible_threshold <= match_threshold`.
    pub fn new(match_threshold: f32, possible_threshold: f32, auto_apply: bool) -> Self {
        let match_threshold = match_threshold.clamp(0.0, 1.0);
        let possible_threshold = possible_threshold.clamp(0.0, match_threshold);
        Self {
            match_threshold,
            possible_threshold,
            auto_apply,
        }
    }

    /// Classify a combined match score.
    pub fn classify(&self, score: f32) -> MatchClass {
        if score >= self.match_threshold {
            MatchClass::Match
        } else if score >= self.possible_threshold {
            MatchClass::Possible
        } else {
            MatchClass::NonMatch
        }
    }
}

// ============================================================================
// SIGNALS — the self-explaining "why did these combine?"
// ============================================================================

/// The individual signals behind a candidate's score. Surfaced verbatim so a
/// user can see *why* two memories were judged duplicates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchSignals {
    /// Cosine similarity of the two embeddings (0–1).
    pub embedding_similarity: f32,
    /// Jaccard overlap of the two tag sets (0–1).
    pub tag_overlap: f32,
    /// Jaccard overlap of content tokens (0–1).
    pub token_overlap: f32,
    /// Combined weighted score that was classified.
    pub combined_score: f32,
}

/// Compute the combined match score and its signal breakdown for a pair.
pub fn score_pair(
    embedding_similarity: f32,
    a_tags: &[String],
    b_tags: &[String],
    a_content: &str,
    b_content: &str,
) -> MatchSignals {
    let tag_overlap = jaccard(&tag_set(a_tags), &tag_set(b_tags));
    let token_overlap = jaccard(&token_set(a_content), &token_set(b_content));
    let combined_score = (W_EMBEDDING * embedding_similarity.clamp(0.0, 1.0)
        + W_TAGS * tag_overlap
        + W_TOKENS * token_overlap)
        .clamp(0.0, 1.0);
    MatchSignals {
        embedding_similarity: embedding_similarity.clamp(0.0, 1.0),
        tag_overlap,
        token_overlap,
        combined_score,
    }
}

fn tag_set(tags: &[String]) -> std::collections::HashSet<String> {
    tags.iter().map(|t| t.to_lowercase()).collect()
}

fn token_set(content: &str) -> std::collections::HashSet<String> {
    content
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 2)
        .map(|t| t.to_lowercase())
        .collect()
}

fn jaccard(a: &std::collections::HashSet<String>, b: &std::collections::HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 { 0.0 } else { inter / union }
}

// ============================================================================
// CANDIDATE
// ============================================================================

/// A surfaced merge candidate: a cluster of likely-duplicate memories with the
/// signals and classification that justify offering it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeCandidate {
    /// Node ids in the cluster. The first is the suggested survivor (highest
    /// retention).
    pub member_ids: Vec<String>,
    /// Short content previews, parallel to `member_ids`.
    pub previews: Vec<String>,
    /// Suggested survivor id (kept after a merge).
    pub survivor_id: String,
    /// Combined match score for the cluster (min pairwise within the cluster —
    /// the weakest link, so a cluster is only as confident as its loosest pair).
    pub confidence: f32,
    /// Three-way classification under the active policy.
    pub classification: MatchClass,
    /// Signals for the survivor↔closest-member pair (the explanation).
    pub signals: MatchSignals,
    /// True if any member is protected (pinned) — blocks auto-merge.
    pub has_protected_member: bool,
}

// ============================================================================
// PLAN — the previewable diff
// ============================================================================

/// What kind of plan this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanKind {
    /// Combine N memories into one survivor.
    Merge,
    /// Invalidate A in favour of B (bitemporal, audit-preserving).
    Supersede,
}

impl PlanKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            PlanKind::Merge => "merge",
            PlanKind::Supersede => "supersede",
        }
    }
}

/// A previewable plan: exactly what *would* change, without changing anything.
/// Persisted to `merge_plans`; consumed by `apply_plan` via its `id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergePlan {
    /// Plan id (UUID).
    pub id: String,
    /// merge | supersede.
    pub kind: PlanKind,
    /// Node kept after the operation.
    pub survivor_id: String,
    /// All node ids involved.
    pub member_ids: Vec<String>,
    /// Resulting content of the survivor after applying.
    pub result_content: String,
    /// Resulting tag set of the survivor after applying.
    pub result_tags: Vec<String>,
    /// Resulting provenance / source string after applying.
    pub result_source: Option<String>,
    /// For supersede: ids that get bitemporally invalidated (their
    /// `valid_until` stamped, kept queryable). For merge: the absorbed ids.
    pub invalidated_ids: Vec<String>,
    /// Match confidence (0–1) for the plan.
    pub confidence: f32,
    /// Three-way classification.
    pub classification: MatchClass,
    /// Signals explaining the plan.
    pub signals: MatchSignals,
    /// Human-readable explanation of what this plan does.
    pub explanation: String,
}

// ============================================================================
// OPERATION LOG — the reversible "memory reflog"
// ============================================================================

/// A recorded, reversible operation. One row in `merge_operations`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeOperation {
    /// Operation id (UUID).
    pub id: String,
    /// Plan id this came from (if any).
    pub plan_id: Option<String>,
    /// merge | supersede | undo.
    pub op_type: String,
    /// applied | reverted.
    pub status: String,
    /// When recorded (RFC3339).
    pub created_at: String,
    /// When reverted (RFC3339), if reverted.
    pub reverted_at: Option<String>,
    /// For undo ops: the op id being reversed.
    pub reverts_op_id: Option<String>,
    /// Survivor node id.
    pub survivor_id: Option<String>,
    /// Node ids touched by the op.
    pub affected_ids: Vec<String>,
    /// Match confidence.
    pub confidence: Option<f32>,
    /// Human-readable reason.
    pub reason: Option<String>,
}

// ============================================================================
// MERGE COMPOSITION — pure helpers used by the storage apply path
// ============================================================================

/// Compose merged content from an ordered list of (id, content) members.
/// Survivor content leads; each absorbed member is appended with provenance so
/// nothing is silently dropped (anti-pattern: Mem0 #4896 double-store /
/// contradiction loss).
pub fn compose_merged_content(members: &[(String, String)]) -> String {
    if members.is_empty() {
        return String::new();
    }
    let mut out = members[0].1.trim().to_string();
    for (id, content) in &members[1..] {
        let c = content.trim();
        if c.is_empty() || out.contains(c) {
            continue;
        }
        out.push_str("\n\n[merged from ");
        out.push_str(id);
        out.push_str("]\n");
        out.push_str(c);
    }
    out
}

/// Union the tag sets of all members, preserving first-seen order.
pub fn compose_merged_tags(member_tags: &[Vec<String>]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for tags in member_tags {
        for t in tags {
            if seen.insert(t.to_lowercase()) {
                out.push(t.clone());
            }
        }
    }
    out
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_three_zones() {
        let policy = MergePolicy::default();
        assert_eq!(policy.classify(0.95), MatchClass::Match);
        assert_eq!(policy.classify(0.80), MatchClass::Possible);
        assert_eq!(policy.classify(0.50), MatchClass::NonMatch);
        // boundaries are inclusive at the lower edge of each higher zone
        assert_eq!(policy.classify(DEFAULT_MATCH_THRESHOLD), MatchClass::Match);
        assert_eq!(
            policy.classify(DEFAULT_POSSIBLE_THRESHOLD),
            MatchClass::Possible
        );
    }

    #[test]
    fn policy_clamps_and_orders() {
        // possible above match gets clamped down to match
        let p = MergePolicy::new(0.8, 0.95, true);
        assert!(p.possible_threshold <= p.match_threshold);
        // out-of-range clamps to [0,1]
        let p2 = MergePolicy::new(2.0, -1.0, false);
        assert_eq!(p2.match_threshold, 1.0);
        assert_eq!(p2.possible_threshold, 0.0);
    }

    #[test]
    fn score_pair_combines_signals() {
        let s = score_pair(
            1.0,
            &["rust".into(), "async".into()],
            &["rust".into(), "async".into()],
            "use tokio for async rust",
            "use tokio for async rust",
        );
        assert!((s.embedding_similarity - 1.0).abs() < 1e-6);
        assert!((s.tag_overlap - 1.0).abs() < 1e-6);
        assert!(s.token_overlap > 0.9);
        assert!(s.combined_score > 0.95);
    }

    #[test]
    fn score_pair_disjoint_is_low() {
        let s = score_pair(
            0.1,
            &["a".into()],
            &["b".into()],
            "completely different topic alpha",
            "totally unrelated subject beta",
        );
        assert!(s.combined_score < 0.3);
        assert_eq!(MergePolicy::default().classify(s.combined_score), MatchClass::NonMatch);
    }

    #[test]
    fn jaccard_basics() {
        let a: std::collections::HashSet<String> = ["x".into(), "y".into()].into_iter().collect();
        let b: std::collections::HashSet<String> = ["y".into(), "z".into()].into_iter().collect();
        assert!((jaccard(&a, &b) - (1.0 / 3.0)).abs() < 1e-6);
        let empty: std::collections::HashSet<String> = Default::default();
        assert_eq!(jaccard(&empty, &empty), 0.0);
    }

    #[test]
    fn compose_merged_content_dedups_and_attributes() {
        let members = vec![
            ("a".into(), "Keep this.".into()),
            ("b".into(), "Extra detail.".into()),
            ("c".into(), "Keep this.".into()), // duplicate of survivor → skipped
        ];
        let merged = compose_merged_content(&members);
        assert!(merged.starts_with("Keep this."));
        assert!(merged.contains("[merged from b]"));
        assert!(merged.contains("Extra detail."));
        // duplicate content not appended twice
        assert_eq!(merged.matches("Keep this.").count(), 1);
    }

    #[test]
    fn compose_merged_tags_unions_in_order() {
        let tags = vec![
            vec!["rust".into(), "async".into()],
            vec!["async".into(), "tokio".into()],
        ];
        let merged = compose_merged_tags(&tags);
        assert_eq!(merged, vec!["rust", "async", "tokio"]);
    }

    #[test]
    fn match_class_labels() {
        assert_eq!(MatchClass::Match.as_str(), "match");
        assert_eq!(MatchClass::Possible.as_str(), "possible");
        assert_eq!(MatchClass::NonMatch.as_str(), "non_match");
    }
}
