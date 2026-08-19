//! # Retroactive Salience Backfill
//!
//! Memory with hindsight. When a salient *failure* event lands (a bug, crash,
//! regression — the "aversive event"), this reaches **backward in time** and
//! promotes the quiet earlier memory that secretly caused it — the one a pure
//! semantic search will never surface because it isn't *similar* to the failure,
//! only *causally upstream* of it.
//!
//! ## Scientific basis
//!
//! Faithful port of Zaki, Cai et al. (2024), *Nature* 637:145-155, "Offline
//! ensemble co-reactivation links memories across days." Key findings ported:
//!
//! - A **neutral** memory formed earlier is retroactively promoted to important
//!   only when a **salient** event later co-reactivates the two ensembles
//!   offline. (Here: the dream/consolidation pass is the offline window.)
//! - **The asymmetry is backward-only**: "fear links retrospectively, but not
//!   prospectively." A failure promotes the *past* cause, never a future memory.
//!   This is also exactly correct for software: a root cause is always upstream
//!   in time. The biological directionality earns its keep, it is not decorative.
//! - Linking flows along the **overlap ensemble** — memories that share entities
//!   (same file, env var, service, symbol). That shared-entity edge is the join
//!   key the backward scan follows; semantic similarity is deliberately NOT the
//!   ranking signal (that is the whole point — RAG already covers similarity).
//!
//! Honesty note for callers: this is scoped to *failure → backward causal
//! backfill*, not a universal "all salience flows backward" law. The Cai paper
//! is an aversive→neutral paradigm; we mirror that scope intentionally.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ============================================================================
// CONSTANTS
// ============================================================================

/// A memory must be at least this surprising (prediction error, 0..1) to count
/// as a salient "aversive event" that can trigger a backfill. Mirrors the gate's
/// own surprise scale. Manual triggers bypass this.
pub const DEFAULT_SALIENCE_THRESHOLD: f32 = 0.55;

/// How far back in time the backward reach scans, in days. The Cai paradigm
/// linked across ~2 days; software causes can be older, so we default wider.
pub const DEFAULT_LOOKBACK_DAYS: i64 = 30;

/// A candidate must share at least this many entities with the failure to be
/// considered causally upstream (1 shared file/env-var/service is enough).
pub const MIN_SHARED_ENTITIES: usize = 1;

/// Words that mark a memory as a failure/"aversive" event when auto-detecting.
/// Lowercased substring match against content + tags.
pub const FAILURE_MARKERS: &[&str] = &[
    "error",
    "bug",
    "crash",
    "crashed",
    "regression",
    "broke",
    "broken",
    "failure",
    "failed",
    "panic",
    "exception",
    "fault",
    "outage",
    "incident",
    // NOTE: bare "500" was removed — it matched benign content like "$500",
    // "500 users", or "line 500" and wrongly flagged a quiet CAUSE memory as a
    // failure, excluding it from the backward reach. The specific HTTP error
    // codes 502/503/504 below stay; a genuine "HTTP 500" is still caught by
    // "error"/"failed"/"exception" in any real incident note.
    "timeout",
    "deadlock",
    "leak",
    "corrupt",
    "stack overflow",
    // performance/degradation failures (an agent should backfill from these too)
    "spiked",
    "latency",
    "degraded",
    "slow",
    "hang",
    "hung",
    "throttled",
    "oom",
    "502",
    "503",
    "504",
    "rejected",
    "denied",
    "flaky",
    // real-incident vocabulary (CauseBench found these missing — postmortems often
    // describe failures without the classic crash words above)
    "pinned",
    "saturated",
    "saturation",
    "stalled",
    "exhausted",
    "exhaustion",
    "overload",
    "overloaded",
    "backlog",
    "fell behind",
    "lag",
    "lagging",
    "unavailable",
    "down",
    "dropped",
    "reset",
    "refused",
    "stampede",
    "thrashing",
    "starved",
    "starvation",
    "expired",
    "expiry",
    "overflow",
];

/// How strongly to promote the backfilled cause: multiply its stability by this
/// (capped). A real boost so the cause stops decaying and surfaces in future
/// recalls — without overwriting the FSRS history.
pub const PROMOTION_STABILITY_FACTOR: f64 = 2.5;

// ============================================================================
// INPUT TYPES
// ============================================================================

/// The minimal view of a memory the backfill needs. Built from a KnowledgeNode
/// by the caller (keeps this module storage-agnostic + trivially testable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfillCandidate {
    pub id: String,
    pub content: String,
    /// Entities this memory mentions: files, env vars, services, symbols.
    pub entities: Vec<String>,
    /// Age in days relative to the failure event (older = larger). Negative or
    /// zero means it is NOT in the past relative to the failure → excluded.
    pub age_days_before_failure: f64,
    /// Current FSRS stability (we promote by boosting this).
    pub stability: f64,
    /// Optional cosine similarity to the failure, ONLY used to demonstrate that
    /// the cause ranks LOW on similarity (the thing RAG misses). Not a ranker.
    pub similarity_to_failure: Option<f32>,
}

/// The salient failure event that triggers the backward reach.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureEvent {
    pub id: String,
    pub content: String,
    pub entities: Vec<String>,
    /// The failure's tags — failure markers can live in a tag, so salience
    /// detection must see them too.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Prediction error / surprise of this event (0..1).
    pub prediction_error: f32,
    /// True if a caller explicitly marked this salient (manual override path).
    pub manual: bool,
}

/// Pull shared-entity join keys from content + tags (single source of truth used
/// by the MCP tool, CLI, and offline pass so they never diverge).
pub fn extract_entities(content: &str, tags: &[String]) -> Vec<String> {
    use std::collections::HashSet;
    let mut set: HashSet<String> = tags.iter().map(|t| t.to_lowercase()).collect();
    for raw in content
        .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.' || c == '/' || c == '-'))
    {
        let tok = raw.trim_matches(|c: char| c == '.' || c == '/' || c == '-');
        if tok.len() < 3 {
            continue;
        }
        let is_env = tok.len() >= 3
            && tok
                .chars()
                .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
            && tok.chars().any(|c| c.is_ascii_uppercase());
        let is_path = (tok.contains('/') || tok.contains('.'))
            && tok.chars().any(|c| c.is_ascii_alphabetic());
        if is_env || is_path {
            set.insert(tok.to_lowercase());
        }
    }
    set.into_iter().collect()
}

/// Whole-word marker match: `marker` must appear bounded by non-alphanumeric
/// chars, not embedded in a larger identifier. This is the difference between
/// matching "timeout" in "a request timeout" (a real failure) and NOT matching it
/// inside the config var `API_TIMEOUT` (a perfectly ordinary env var). Plain
/// substring over-fires: "timeout" hits "API_TIMEOUT", "leak" hits "leaky", "500"
/// hits "$500" — which wrongly flags a quiet CAUSE as a failure and excludes it
/// from the backward reach.
fn contains_marker_word(hay: &str, marker: &str) -> bool {
    let mut from = 0usize;
    while let Some(pos) = hay[from..].find(marker) {
        let start = from + pos;
        let end = start + marker.len();
        // Inspect the actual char before/after the match, not a raw byte cast to
        // char: for a multibyte UTF-8 boundary the raw byte is a continuation
        // byte (0x80-0xBF), which `as char` misreads as a non-alphanumeric and
        // wrongly passes the word-boundary check. char iteration is boundary-safe.
        let before_ok = hay[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_alphanumeric() || c == '_'));
        let after_ok = hay[end..]
            .chars()
            .next()
            .is_none_or(|c| !(c.is_alphanumeric() || c == '_'));
        if before_ok && after_ok {
            return true;
        }
        from = start + 1;
    }
    false
}

/// Does this content/tags pair read like a failure? Whole-word marker match over
/// content + tags (see [`contains_marker_word`] for why whole-word, not substring).
/// Shared by every caller so failure detection never drifts.
pub fn looks_like_failure(content: &str, tags: &[String]) -> bool {
    let hay = content.to_lowercase();
    if FAILURE_MARKERS
        .iter()
        .any(|m| contains_marker_word(&hay, m))
    {
        return true;
    }
    tags.iter().any(|t| {
        let tl = t.to_lowercase();
        FAILURE_MARKERS.iter().any(|m| contains_marker_word(&tl, m))
    })
}

impl FailureEvent {
    /// Auto-detection: is this memory a salient "aversive event"? True when it
    /// is sufficiently surprising AND carries a failure marker — or when a caller
    /// manually flagged it. (The "both" trigger: auto-detect + manual override.)
    pub fn is_salient(&self, salience_threshold: f32) -> bool {
        if self.manual {
            return true;
        }
        if self.prediction_error < salience_threshold {
            return false;
        }
        looks_like_failure(&self.content, &self.tags)
    }
}

// ============================================================================
// OUTPUT TYPES
// ============================================================================

/// One promoted memory: a quiet earlier cause the failure reached back to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BackfilledCause {
    pub memory_id: String,
    /// The entities it shares with the failure (the causal join).
    pub shared_entities: Vec<String>,
    /// Days before the failure this memory was formed.
    pub age_days: f64,
    /// Backfill score (higher = stronger candidate cause).
    pub score: f64,
    /// New stability after promotion (= old * factor, capped).
    pub promoted_stability: f64,
    /// Its similarity rank position among candidates by similarity (1 = most
    /// similar). A high number here is the proof: the cause is NOT what a
    /// similarity search would have surfaced.
    pub similarity_rank: Option<usize>,
    /// Human-readable why.
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfillResult {
    pub triggered: bool,
    pub failure_id: String,
    pub causes: Vec<BackfilledCause>,
    pub scanned: usize,
}

// ============================================================================
// THE BACKFILL
// ============================================================================

#[derive(Debug, Clone)]
pub struct RetroactiveBackfill {
    pub salience_threshold: f32,
    pub lookback_days: i64,
    pub min_shared_entities: usize,
    pub max_causes: usize,
}

impl Default for RetroactiveBackfill {
    fn default() -> Self {
        Self {
            salience_threshold: DEFAULT_SALIENCE_THRESHOLD,
            lookback_days: DEFAULT_LOOKBACK_DAYS,
            min_shared_entities: MIN_SHARED_ENTITIES,
            max_causes: 3,
        }
    }
}

impl RetroactiveBackfill {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run the backward reach. Given a (possibly salient) failure and the pool of
    /// earlier candidate memories, return which past memories to promote and why.
    ///
    /// Backward-only by construction: candidates with `age_days_before_failure`
    /// <= 0 (i.e. concurrent or future) are never considered.
    pub fn run(&self, failure: &FailureEvent, candidates: &[BackfillCandidate]) -> BackfillResult {
        if !failure.is_salient(self.salience_threshold) {
            return BackfillResult {
                triggered: false,
                failure_id: failure.id.clone(),
                causes: vec![],
                scanned: 0,
            };
        }

        let failure_entities: HashSet<&str> = failure.entities.iter().map(|s| s.as_str()).collect();

        // similarity ranking (only to PROVE the cause ranks low on similarity)
        let mut by_sim: Vec<(&str, f32)> = candidates
            .iter()
            .filter_map(|c| c.similarity_to_failure.map(|s| (c.id.as_str(), s)))
            .collect();
        by_sim.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let sim_rank = |id: &str| -> Option<usize> {
            by_sim.iter().position(|(cid, _)| *cid == id).map(|p| p + 1)
        };

        let mut scored: Vec<BackfilledCause> = candidates
            .iter()
            // backward-only: must be strictly in the past, within lookback
            .filter(|c| {
                c.age_days_before_failure > 0.0
                    && c.age_days_before_failure <= self.lookback_days as f64
            })
            .filter_map(|c| {
                let shared: Vec<String> = c
                    .entities
                    .iter()
                    .filter(|e| failure_entities.contains(e.as_str()))
                    .cloned()
                    .collect();
                if shared.len() < self.min_shared_entities {
                    return None;
                }
                let score = self.score(c, shared.len());
                let promoted = (c.stability * PROMOTION_STABILITY_FACTOR).min(c.stability + 365.0);
                let rank = sim_rank(&c.id);
                let reason = format!(
                    "Reached back {:.1}d to a quiet memory sharing {} entit{} ({}) with the failure; \
                     it ranked {} on similarity, so semantic search would have missed it.",
                    c.age_days_before_failure,
                    shared.len(),
                    if shared.len() == 1 { "y" } else { "ies" },
                    shared.join(", "),
                    rank.map(|r| format!("#{r}")).unwrap_or_else(|| "untracked".into()),
                );
                Some(BackfilledCause {
                    memory_id: c.id.clone(),
                    shared_entities: shared,
                    age_days: c.age_days_before_failure,
                    score,
                    promoted_stability: promoted,
                    similarity_rank: rank,
                    reason,
                })
            })
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(self.max_causes);

        BackfillResult {
            triggered: true,
            failure_id: failure.id.clone(),
            causes: scored,
            scanned: candidates.len(),
        }
    }

    /// Score a candidate cause. More shared entities = stronger causal join.
    /// Recency among the past matters a little (a change yesterday is a more
    /// likely cause than one a month ago) but is deliberately a *weak* term so
    /// genuinely old causes still surface — the opposite of recency-only ranking.
    /// LOW similarity is rewarded slightly: a cause that is dissimilar to the
    /// failure is exactly the one RAG cannot find, so it is the most valuable
    /// to backfill.
    fn score(&self, c: &BackfillCandidate, shared: usize) -> f64 {
        let entity_term = shared as f64; // dominant signal
        // gentle recency-in-the-past: 1.0 at the failure, fading with age
        let recency_term =
            0.3 * (1.0 / (1.0 + c.age_days_before_failure / self.lookback_days as f64));
        // dissimilarity bonus: the less similar, the more "RAG would miss it"
        let dissim_term = c
            .similarity_to_failure
            .map(|s| 0.5 * (1.0 - s as f64).max(0.0))
            .unwrap_or(0.0);
        entity_term + recency_term + dissim_term
    }
}

// ============================================================================
// TESTS — the receipt: plant a cause, inject a failure, assert backfill finds
// the cause that a similarity search ranks near the bottom.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn failure() -> FailureEvent {
        FailureEvent {
            id: "fail-wed".into(),
            content: "Service crashed: 500 Internal Server Error on the auth endpoint".into(),
            entities: vec!["auth-service".into(), "API_TIMEOUT".into()],
            tags: vec![],
            prediction_error: 0.9,
            manual: false,
        }
    }

    /// The headline scenario: a quiet env-var change days ago caused a crash now.
    /// Semantic search ranks it LAST (it's not similar to "crash"); backfill
    /// promotes it because it shares the API_TIMEOUT entity, backward in time.
    #[test]
    fn backfill_surfaces_the_cause_rag_misses() {
        let candidates = vec![
            // the actual cause: a quiet config note from 3 days ago. Low similarity.
            BackfillCandidate {
                id: "cause-mon".into(),
                content: "Set API_TIMEOUT=2 in the deploy env to speed up cold starts".into(),
                entities: vec!["API_TIMEOUT".into(), "deploy-env".into()],
                age_days_before_failure: 3.0,
                stability: 5.0,
                similarity_to_failure: Some(0.11), // dissimilar — RAG would miss it
            },
            // a noisy distractor: semantically similar to the crash, but NOT causal
            // (shares no entity with the failure).
            BackfillCandidate {
                id: "noise-similar".into(),
                content: "Another 500 error happened in the billing service last month".into(),
                entities: vec!["billing-service".into()],
                age_days_before_failure: 20.0,
                stability: 3.0,
                similarity_to_failure: Some(0.82), // similar — RAG WOULD surface this
            },
            // a future memory — must never be backfilled (backward-only).
            BackfillCandidate {
                id: "future".into(),
                content: "Plan to add API_TIMEOUT retries next sprint".into(),
                entities: vec!["API_TIMEOUT".into()],
                age_days_before_failure: -1.0,
                stability: 2.0,
                similarity_to_failure: Some(0.4),
            },
        ];

        let result = RetroactiveBackfill::new().run(&failure(), &candidates);

        assert!(
            result.triggered,
            "high-PE failure with markers must trigger"
        );
        assert!(!result.causes.is_empty(), "must surface at least one cause");

        let top = &result.causes[0];
        // the promoted memory is the real cause, not the similar distractor
        assert_eq!(
            top.memory_id, "cause-mon",
            "must promote the causal env-var note"
        );
        assert!(top.shared_entities.contains(&"API_TIMEOUT".to_string()));
        // and it is provably NOT what similarity search would have surfaced:
        assert!(
            top.similarity_rank.unwrap() > 1,
            "the cause must rank below the similar distractor on similarity (that's the point)"
        );
        // backward-only: the future memory is never promoted
        assert!(
            !result.causes.iter().any(|c| c.memory_id == "future"),
            "backward-only: a future memory must never be backfilled"
        );
        // it gets a real stability boost (stops decaying, will surface next time)
        assert!(
            top.promoted_stability > 5.0,
            "the cause must be promoted (boosted stability)"
        );
    }

    #[test]
    fn non_salient_event_does_not_trigger() {
        let calm = FailureEvent {
            id: "calm".into(),
            content: "Refactored the logging format for readability".into(),
            entities: vec!["logger".into()],
            tags: vec![],
            prediction_error: 0.2, // low surprise
            manual: false,
        };
        let result = RetroactiveBackfill::new().run(&calm, &[]);
        assert!(
            !result.triggered,
            "a calm, low-surprise note must not fire a backfill"
        );
    }

    #[test]
    fn manual_override_triggers_without_markers() {
        // No failure word, low PE — but the caller explicitly marked it salient.
        let manual = FailureEvent {
            id: "manual".into(),
            content: "Latency crept up on the checkout path".into(),
            entities: vec!["checkout".into()],
            tags: vec![],
            prediction_error: 0.1,
            manual: true,
        };
        let candidates = vec![BackfillCandidate {
            id: "cause".into(),
            content: "Disabled the checkout cache while debugging".into(),
            entities: vec!["checkout".into()],
            age_days_before_failure: 2.0,
            stability: 4.0,
            similarity_to_failure: Some(0.3),
        }];
        let result = RetroactiveBackfill::new().run(&manual, &candidates);
        assert!(
            result.triggered,
            "manual override must trigger regardless of markers/PE"
        );
        assert_eq!(result.causes[0].memory_id, "cause");
    }

    #[test]
    fn requires_a_shared_entity_no_spurious_links() {
        // A salient failure but the only past memory shares NO entity — we must
        // NOT invent a causal link (avoids the A-B,B-C spurious-edge failure mode).
        let candidates = vec![BackfillCandidate {
            id: "unrelated".into(),
            content: "Updated the README badges".into(),
            entities: vec!["README".into()],
            age_days_before_failure: 1.0,
            stability: 4.0,
            similarity_to_failure: Some(0.05),
        }];
        let result = RetroactiveBackfill::new().run(&failure(), &candidates);
        assert!(result.triggered);
        assert!(
            result.causes.is_empty(),
            "no shared entity => no backfill (don't fabricate a cause)"
        );
    }
}
