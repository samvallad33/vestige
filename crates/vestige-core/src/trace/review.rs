//! # Memory PRs — review changes to an agent's brain like code
//!
//! Ordinary context auto-commits and always leaves a receipt. But a *risky*
//! write — one where the agent is rewriting its own brain — opens a reviewable
//! [`MemoryPr`] instead. [`classify_write`] is the immune system: given a
//! [`WriteContext`] and a [`ReviewMode`], it returns the [`RiskClass`] and the
//! [`RiskSignal`]s that explain, in plain language, *why* a write needs review.
//!
//! ## The three modes (one-click in the dashboard)
//!
//! | Mode | Behaviour |
//! |------|-----------|
//! | [`ReviewMode::Fast`] | Never gate. Every write auto-commits. (Demos, trusted solo flows.) |
//! | [`ReviewMode::RiskGated`] | **Default.** Auto-commit ordinary writes; open a PR for risky ones. |
//! | [`ReviewMode::Paranoid`] | Gate *every* write. Nothing enters the brain without approval. |
//!
//! ## What counts as "risky" (the taxonomy)
//!
//! A write is risky when any of these hold:
//! - it **contradicts a high-trust memory**,
//! - it **supersedes / forgets / merges / protects** existing memory,
//! - it touches **identity, user preference, workflow, or project positioning**,
//! - it asserts a **permission / auth / security / money / bounty / legal-ish** fact,
//! - it is a **dream consolidation** proposal,
//! - it **resurrects a decayed** memory (below the retention threshold),
//! - it is part of a **low-confidence batch import**,
//! - it is an **external connector write without strong provenance**.
//!
//! Each rule maps to a [`RiskSignal`] so the resulting Memory PR is fully
//! self-explaining.

use serde::{Deserialize, Serialize};

use super::WriteSource;

/// A memory is "high trust" at or above this FSRS retrievability/trust score.
/// Contradicting something this trusted is always worth a review.
pub const HIGH_TRUST_FLOOR: f64 = 0.7;

/// Writes below this confidence are treated as low-confidence (e.g. a bulk
/// import where the model wasn't sure).
pub const LOW_CONFIDENCE_FLOOR: f64 = 0.5;

// ============================================================================
// REVIEW MODE
// ============================================================================

/// How aggressively the agent's brain gates incoming writes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReviewMode {
    /// Never gate — every write auto-commits.
    Fast,
    /// Default: auto-commit ordinary writes, open a PR for risky ones.
    #[default]
    RiskGated,
    /// Gate every write — nothing enters the brain without approval.
    Paranoid,
}

impl ReviewMode {
    /// Stable string label, also the wire form.
    pub fn as_str(&self) -> &'static str {
        match self {
            ReviewMode::Fast => "fast",
            ReviewMode::RiskGated => "risk_gated",
            ReviewMode::Paranoid => "paranoid",
        }
    }

    /// Parse from a label (case-insensitive, tolerant of `-`/`_`). Falls back to
    /// the default [`ReviewMode::RiskGated`] on anything unrecognised.
    pub fn from_label(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "fast" => ReviewMode::Fast,
            "paranoid" => ReviewMode::Paranoid,
            _ => ReviewMode::RiskGated,
        }
    }
}

// ============================================================================
// RISK CLASSIFICATION
// ============================================================================

/// The outcome of [`classify_write`]: does this write auto-commit or open a PR?
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RiskClass {
    /// Ordinary context — auto-commit (a receipt is still generated).
    AutoCommit,
    /// Risky — open a [`MemoryPr`] for review.
    Review,
}

impl RiskClass {
    /// Stable string label.
    pub fn as_str(&self) -> &'static str {
        match self {
            RiskClass::AutoCommit => "auto_commit",
            RiskClass::Review => "review",
        }
    }

    /// Whether this write should be held for review.
    pub fn needs_review(&self) -> bool {
        matches!(self, RiskClass::Review)
    }
}

/// A single, self-explaining reason a write was flagged for review. The
/// `code` is stable for filtering/telemetry; the `detail` is human prose for
/// the PR card.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RiskSignal {
    /// Stable machine code, e.g. `"contradicts_high_trust"`.
    pub code: String,
    /// Plain-language explanation shown on the Memory PR.
    pub detail: String,
}

impl RiskSignal {
    fn new(code: &str, detail: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            detail: detail.into(),
        }
    }
}

/// Everything [`classify_write`] needs to decide whether a write is risky.
///
/// All fields default to the "ordinary, safe" interpretation so callers only
/// set the signals that actually apply to their write.
#[derive(Debug, Clone, Default)]
pub struct WriteContext {
    /// Who is performing the write.
    pub source: Option<WriteSource>,
    /// The node type being written, e.g. `"fact"`, `"preference"`, `"identity"`.
    pub node_type: String,
    /// The content (or a representative slice) — scanned for sensitive topics.
    pub content: String,
    /// Tags attached to the write — also scanned for sensitive topics.
    pub tags: Vec<String>,
    /// The write contradicts an existing memory whose trust is this high.
    /// `None` if there is no contradiction.
    pub contradicts_trust: Option<f64>,
    /// This write supersedes / replaces an existing memory.
    pub supersedes: bool,
    /// This write forgets / suppresses an existing memory.
    pub forgets: bool,
    /// This write merges existing memories.
    pub merges: bool,
    /// This write protects / pins a memory.
    pub protects: bool,
    /// This write resurrects a memory that had decayed below retention.
    pub resurrects_decayed: bool,
    /// Confidence of the write (0..1). `None` means "not a batch / unknown".
    pub confidence: Option<f64>,
    /// This write is one of many in a bulk import.
    pub batch_import: bool,
    /// For connector writes: whether the source envelope carries strong
    /// provenance (a verified `source_system` + `source_id` + URL).
    pub strong_provenance: bool,
}

/// Sensitive topic substrings. A write whose content/tags/type mention any of
/// these is treated as touching identity / preference / security / money /
/// legal / workflow / positioning and is routed to review.
const SENSITIVE_TOPICS: &[(&str, &str)] = &[
    // identity & preference
    ("identity", "identity fact"),
    ("preference", "user preference"),
    ("workflow", "workflow rule"),
    ("positioning", "project positioning"),
    ("persona", "agent persona"),
    // permission / auth / security
    ("permission", "tool permission"),
    ("auth", "authentication / authorization"),
    ("token", "credential / token"),
    ("secret", "secret material"),
    ("password", "credential"),
    ("api key", "credential / API key"),
    ("security", "security-relevant fact"),
    ("vuln", "security vulnerability"),
    ("vulnerability", "security vulnerability"),
    ("credential", "credential material"),
    ("credentials", "credential material"),
    ("api key", "credential / API key"),
    ("apikey", "credential / API key"),
    // money / bounty / legal
    ("money", "financial fact"),
    ("payment", "financial fact"),
    ("invoice", "financial fact"),
    ("bounty", "bounty / payout"),
    ("salary", "financial fact"),
    ("license", "legal / license fact"),
    ("legal", "legal-relevant fact"),
    ("contract", "legal / contract fact"),
];

/// Node types that are intrinsically sensitive regardless of content.
const SENSITIVE_NODE_TYPES: &[&str] = &[
    "identity",
    "preference",
    "user_preference",
    "credential",
    "permission",
    "security",
    "constitution",
];

/// Classify a write into auto-commit vs. review, with the signals explaining the
/// decision.
///
/// This is the immune system. It is pure and deterministic, so the dashboard's
/// "explain this PR" view and the agent's `Ask Agent Why` action see exactly the
/// same reasoning the gate used.
pub fn classify_write(ctx: &WriteContext, mode: ReviewMode) -> (RiskClass, Vec<RiskSignal>) {
    // Mode shortcuts.
    match mode {
        // Fast never gates — but we still collect signals so the receipt/PR
        // record can note what *would* have been flagged.
        ReviewMode::Fast => return (RiskClass::AutoCommit, Vec::new()),
        ReviewMode::Paranoid => {
            let mut signals = collect_signals(ctx);
            if signals.is_empty() {
                signals.push(RiskSignal::new(
                    "paranoid_mode",
                    "Paranoid mode: every write is reviewed before entering memory.",
                ));
            }
            return (RiskClass::Review, signals);
        }
        ReviewMode::RiskGated => {}
    }

    let signals = collect_signals(ctx);
    if signals.is_empty() {
        (RiskClass::AutoCommit, signals)
    } else {
        (RiskClass::Review, signals)
    }
}

/// Gather every risk signal that applies to a write, independent of mode.
fn collect_signals(ctx: &WriteContext) -> Vec<RiskSignal> {
    let mut signals = Vec::new();

    // 1. Contradiction against a high-trust memory.
    if let Some(trust) = ctx.contradicts_trust
        && trust >= HIGH_TRUST_FLOOR
    {
        signals.push(RiskSignal::new(
            "contradicts_high_trust",
            format!(
                "Contradicts an existing high-trust memory (trust {:.2} ≥ {:.2}).",
                trust, HIGH_TRUST_FLOOR
            ),
        ));
    }

    // 2. Structural rewrites of existing memory.
    if ctx.supersedes {
        signals.push(RiskSignal::new(
            "supersedes_memory",
            "Supersedes / replaces an existing memory.",
        ));
    }
    if ctx.forgets {
        signals.push(RiskSignal::new(
            "forgets_memory",
            "Forgets / suppresses an existing memory.",
        ));
    }
    if ctx.merges {
        signals.push(RiskSignal::new(
            "merges_memory",
            "Merges existing memories into one.",
        ));
    }
    if ctx.protects {
        signals.push(RiskSignal::new(
            "protects_memory",
            "Protects / pins a memory against decay and forgetting.",
        ));
    }

    // 3. Sensitive node types & topics (identity / preference / workflow /
    //    positioning / permission / auth / security / money / legal).
    let node_type_lc = ctx.node_type.to_ascii_lowercase();
    if SENSITIVE_NODE_TYPES.contains(&node_type_lc.as_str()) {
        signals.push(RiskSignal::new(
            "sensitive_node_type",
            format!("Writes a sensitive node type: `{}`.", node_type_lc),
        ));
    }
    if let Some(topic) = first_sensitive_topic(&ctx.content, &ctx.tags) {
        signals.push(RiskSignal::new(
            "sensitive_topic",
            format!("Touches a sensitive topic: {topic}."),
        ));
    }

    // 4. Dream consolidation proposals.
    if matches!(ctx.source, Some(WriteSource::Dream)) {
        signals.push(RiskSignal::new(
            "dream_consolidation",
            "Proposed by dream consolidation — a machine-generated change to memory.",
        ));
    }

    // 5. Decay-below-threshold resurrection.
    if ctx.resurrects_decayed {
        signals.push(RiskSignal::new(
            "resurrects_decayed",
            "Resurrects a memory that had decayed below the retention threshold.",
        ));
    }

    // 6. Low-confidence batch imports.
    if ctx.batch_import {
        if let Some(conf) = ctx.confidence {
            if conf < LOW_CONFIDENCE_FLOOR {
                signals.push(RiskSignal::new(
                    "low_confidence_batch",
                    format!(
                        "Low-confidence batch import (confidence {:.2} < {:.2}).",
                        conf, LOW_CONFIDENCE_FLOOR
                    ),
                ));
            }
        } else {
            signals.push(RiskSignal::new(
                "unscored_batch",
                "Batch import with no confidence score.",
            ));
        }
    }

    // 7. External connector writes without strong provenance.
    if matches!(ctx.source, Some(WriteSource::Connector)) && !ctx.strong_provenance {
        signals.push(RiskSignal::new(
            "weak_provenance_connector",
            "External connector write without strong provenance (unverified source envelope).",
        ));
    }

    signals
}

/// Return the human label of the first sensitive topic found in content/tags.
///
/// B6: matches on WORD BOUNDARIES, not substrings — so "tokenizer" no longer
/// trips "token", "author" no longer trips "auth", "secretary" no longer trips
/// "secret". Multi-word needles (e.g. "api key") match a consecutive run of
/// words. The text is lowercased and split on any non-alphanumeric char.
fn first_sensitive_topic(content: &str, tags: &[String]) -> Option<&'static str> {
    // Tokenize content + tags into lowercased alphanumeric words.
    let mut words: Vec<String> = Vec::new();
    let mut push_words = |s: &str| {
        for w in s
            .to_ascii_lowercase()
            .split(|c: char| !c.is_ascii_alphanumeric())
        {
            if !w.is_empty() {
                words.push(w.to_string());
            }
        }
    };
    push_words(content);
    for t in tags {
        push_words(t);
    }

    SENSITIVE_TOPICS
        .iter()
        .find(|(needle, _)| matches_word_sequence(&words, needle))
        .map(|(_, label)| *label)
}

/// Whether `needle` (one or more space-separated words) appears as a consecutive
/// whole-word run in `words`.
fn matches_word_sequence(words: &[String], needle: &str) -> bool {
    let needle_words: Vec<&str> = needle.split_whitespace().collect();
    if needle_words.is_empty() {
        return false;
    }
    if needle_words.len() == 1 {
        return words.iter().any(|w| w == needle_words[0]);
    }
    words
        .windows(needle_words.len())
        .any(|win| win.iter().zip(&needle_words).all(|(w, n)| w == n))
}

// ============================================================================
// MEMORY PR DATA MODEL
// ============================================================================

/// What kind of change a Memory PR represents.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPrKind {
    /// A brand-new fact entering the brain.
    NewFact,
    /// An existing fact being strengthened / reinforced.
    StrengthenedFact,
    /// A contradiction was detected against existing memory.
    ContradictionDetected,
    /// A memory being superseded by a newer one.
    MemorySuperseded,
    /// A new edge added to the knowledge graph.
    EdgeAdded,
    /// A node decayed below the retention threshold.
    NodeDecayed,
    /// Dream consolidation proposed a merge / insight.
    DreamConsolidation,
}

impl MemoryPrKind {
    /// Stable string label.
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryPrKind::NewFact => "new_fact",
            MemoryPrKind::StrengthenedFact => "strengthened_fact",
            MemoryPrKind::ContradictionDetected => "contradiction_detected",
            MemoryPrKind::MemorySuperseded => "memory_superseded",
            MemoryPrKind::EdgeAdded => "edge_added",
            MemoryPrKind::NodeDecayed => "node_decayed",
            MemoryPrKind::DreamConsolidation => "dream_consolidation",
        }
    }

    /// Parse from a label; `None` if unrecognised.
    pub fn from_label(s: &str) -> Option<Self> {
        Some(match s {
            "new_fact" => MemoryPrKind::NewFact,
            "strengthened_fact" => MemoryPrKind::StrengthenedFact,
            "contradiction_detected" => MemoryPrKind::ContradictionDetected,
            "memory_superseded" => MemoryPrKind::MemorySuperseded,
            "edge_added" => MemoryPrKind::EdgeAdded,
            "node_decayed" => MemoryPrKind::NodeDecayed,
            "dream_consolidation" => MemoryPrKind::DreamConsolidation,
            _ => return None,
        })
    }
}

/// The review status of a Memory PR.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPrStatus {
    /// Awaiting a decision.
    #[default]
    Pending,
    /// Promoted into long-term memory as-is.
    Promoted,
    /// Merged into an existing memory.
    Merged,
    /// Superseded an existing memory.
    Superseded,
    /// Quarantined — held in the firewall, not used for retrieval.
    Quarantined,
    /// Forgotten — rejected and suppressed.
    Forgotten,
}

impl MemoryPrStatus {
    /// Stable string label.
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryPrStatus::Pending => "pending",
            MemoryPrStatus::Promoted => "promoted",
            MemoryPrStatus::Merged => "merged",
            MemoryPrStatus::Superseded => "superseded",
            MemoryPrStatus::Quarantined => "quarantined",
            MemoryPrStatus::Forgotten => "forgotten",
        }
    }
}

/// The actions a reviewer can take on a Memory PR (the buttons in the diff UI).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPrAction {
    /// Accept the change as-is.
    Promote,
    /// Fold it into an existing memory.
    Merge,
    /// Use it to supersede an existing memory.
    Supersede,
    /// Hold it in the firewall.
    Quarantine,
    /// Reject and suppress it.
    Forget,
    /// Ask the agent to explain the change (returns the risk signals).
    AskAgentWhy,
}

impl MemoryPrAction {
    /// Parse from a URL/path label; `None` if unrecognised.
    pub fn from_label(s: &str) -> Option<Self> {
        Some(match s {
            "promote" => MemoryPrAction::Promote,
            "merge" => MemoryPrAction::Merge,
            "supersede" => MemoryPrAction::Supersede,
            "quarantine" => MemoryPrAction::Quarantine,
            "forget" => MemoryPrAction::Forget,
            "ask_agent_why" | "ask-agent-why" | "why" => MemoryPrAction::AskAgentWhy,
            _ => return None,
        })
    }

    /// The status this action moves the PR into (`None` for `AskAgentWhy`, which
    /// is read-only).
    pub fn resulting_status(&self) -> Option<MemoryPrStatus> {
        Some(match self {
            MemoryPrAction::Promote => MemoryPrStatus::Promoted,
            MemoryPrAction::Merge => MemoryPrStatus::Merged,
            MemoryPrAction::Supersede => MemoryPrStatus::Superseded,
            MemoryPrAction::Quarantine => MemoryPrStatus::Quarantined,
            MemoryPrAction::Forget => MemoryPrStatus::Forgotten,
            MemoryPrAction::AskAgentWhy => return None,
        })
    }

    /// Whether deciding the PR with this action should **release** the subject
    /// memory from quarantine (reverse the suppression that gate_writes applied).
    ///
    /// A risky write is committed-then-suppressed; approving it must restore its
    /// retrieval influence, otherwise the UI says "promoted" while the memory
    /// stays held out — the bug this guards against. Accept actions release;
    /// `Quarantine` keeps it held; `Forget` rejects it (stays suppressed);
    /// `AskAgentWhy` is read-only.
    pub fn releases_memory(&self) -> bool {
        matches!(
            self,
            MemoryPrAction::Promote | MemoryPrAction::Merge | MemoryPrAction::Supersede
        )
    }
}

/// A reviewable change to the agent's brain — the persisted Memory PR record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryPr {
    /// UUID.
    pub id: String,
    /// What kind of change this is.
    pub kind: MemoryPrKind,
    /// Current review status.
    pub status: MemoryPrStatus,
    /// Short human title for the PR list.
    pub title: String,
    /// The proposed change as a structured diff (before/after, ids, payload).
    pub diff: serde_json::Value,
    /// The self-explaining risk signals that opened this PR.
    pub signals: Vec<RiskSignal>,
    /// The memory id this PR concerns, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject_id: Option<String>,
    /// The run that produced this change, linking the PR back to the black box.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// RFC3339 creation time.
    pub created_at: String,
    /// RFC3339 decision time, once decided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decided_at: Option<String>,
    /// The action that resolved this PR, once decided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision: Option<MemoryPrAction>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ordinary() -> WriteContext {
        WriteContext {
            source: Some(WriteSource::Agent),
            node_type: "fact".into(),
            content: "The build uses cargo and pnpm.".into(),
            tags: vec!["build".into()],
            ..Default::default()
        }
    }

    #[test]
    fn ordinary_write_auto_commits_in_risk_gated() {
        let (class, signals) = classify_write(&ordinary(), ReviewMode::RiskGated);
        assert_eq!(class, RiskClass::AutoCommit);
        assert!(signals.is_empty());
    }

    #[test]
    fn fast_mode_never_gates_even_risky_writes() {
        let mut ctx = ordinary();
        ctx.supersedes = true;
        ctx.contradicts_trust = Some(0.95);
        let (class, _) = classify_write(&ctx, ReviewMode::Fast);
        assert_eq!(class, RiskClass::AutoCommit);
    }

    #[test]
    fn paranoid_mode_gates_even_ordinary_writes() {
        let (class, signals) = classify_write(&ordinary(), ReviewMode::Paranoid);
        assert_eq!(class, RiskClass::Review);
        assert_eq!(signals[0].code, "paranoid_mode");
    }

    #[test]
    fn contradiction_against_high_trust_is_risky() {
        let mut ctx = ordinary();
        ctx.contradicts_trust = Some(0.82);
        let (class, signals) = classify_write(&ctx, ReviewMode::RiskGated);
        assert_eq!(class, RiskClass::Review);
        assert!(signals.iter().any(|s| s.code == "contradicts_high_trust"));
    }

    #[test]
    fn contradiction_against_low_trust_is_fine() {
        let mut ctx = ordinary();
        ctx.contradicts_trust = Some(0.3);
        let (class, _) = classify_write(&ctx, ReviewMode::RiskGated);
        assert_eq!(class, RiskClass::AutoCommit);
    }

    #[test]
    fn supersede_forget_merge_protect_all_gate() {
        for set in [
            |c: &mut WriteContext| c.supersedes = true,
            |c: &mut WriteContext| c.forgets = true,
            |c: &mut WriteContext| c.merges = true,
            |c: &mut WriteContext| c.protects = true,
        ] {
            let mut ctx = ordinary();
            set(&mut ctx);
            let (class, _) = classify_write(&ctx, ReviewMode::RiskGated);
            assert_eq!(class, RiskClass::Review);
        }
    }

    #[test]
    fn sensitive_topics_gate() {
        for topic in [
            "remember my auth token is xyz",
            "Sam's salary is confidential",
            "the bounty payout terms",
            "user preference: dark mode",
            "this is a security vulnerability",
        ] {
            let mut ctx = ordinary();
            ctx.content = topic.into();
            let (class, signals) = classify_write(&ctx, ReviewMode::RiskGated);
            assert_eq!(class, RiskClass::Review, "should gate: {topic}");
            assert!(signals.iter().any(|s| s.code == "sensitive_topic"));
        }
    }

    #[test]
    fn sensitive_topic_word_boundary_no_false_positives_b6() {
        // B6: these ordinary technical writes must NOT gate — they only CONTAIN
        // a sensitive substring, they don't USE the sensitive word.
        // These each only CONTAIN a sensitive substring; the word-boundary fix
        // means they no longer gate. (Note: bare "license"/"contract"/"legal"
        // ARE kept as gating words — a license/contract fact is legitimately
        // legal-relevant — so they're intentionally not in this benign set.)
        for benign in [
            "The tokenizer converts input strings to embeddings.",
            "The author of this module is documented in the header.",
            "The secretary pattern coordinates the worker pool.",
            "Contraction of the array happens during compaction.",
            "The authority record links to the canonical node.",
            "The authentication-free endpoint is for health checks.", // "authentication" != "auth"
        ] {
            let mut ctx = ordinary();
            ctx.content = benign.into();
            ctx.node_type = "fact".into();
            ctx.tags = vec![];
            let (class, _) = classify_write(&ctx, ReviewMode::RiskGated);
            assert_eq!(
                class,
                RiskClass::AutoCommit,
                "must NOT gate ordinary write: {benign}"
            );
        }
    }

    #[test]
    fn sensitive_topic_word_boundary_still_catches_real_b6() {
        // The real sensitive phrasings must still gate.
        for risky in [
            "store the auth token for the deploy",
            "this is a security vulnerability in the parser",
            "the api key for the service",
            "remember the user preference for dark mode",
            "the bounty payout is configured",
        ] {
            let mut ctx = ordinary();
            ctx.content = risky.into();
            ctx.node_type = "fact".into();
            ctx.tags = vec![];
            let (class, signals) = classify_write(&ctx, ReviewMode::RiskGated);
            assert_eq!(class, RiskClass::Review, "must gate: {risky}");
            assert!(signals.iter().any(|s| s.code == "sensitive_topic"));
        }
    }

    #[test]
    fn sensitive_node_type_gates() {
        let mut ctx = ordinary();
        ctx.node_type = "identity".into();
        let (class, signals) = classify_write(&ctx, ReviewMode::RiskGated);
        assert_eq!(class, RiskClass::Review);
        assert!(signals.iter().any(|s| s.code == "sensitive_node_type"));
    }

    #[test]
    fn dream_consolidation_gates() {
        let mut ctx = ordinary();
        ctx.source = Some(WriteSource::Dream);
        let (class, signals) = classify_write(&ctx, ReviewMode::RiskGated);
        assert_eq!(class, RiskClass::Review);
        assert!(signals.iter().any(|s| s.code == "dream_consolidation"));
    }

    #[test]
    fn decayed_resurrection_gates() {
        let mut ctx = ordinary();
        ctx.resurrects_decayed = true;
        let (class, _) = classify_write(&ctx, ReviewMode::RiskGated);
        assert_eq!(class, RiskClass::Review);
    }

    #[test]
    fn low_confidence_batch_gates_but_confident_batch_does_not() {
        let mut low = ordinary();
        low.batch_import = true;
        low.confidence = Some(0.3);
        assert_eq!(
            classify_write(&low, ReviewMode::RiskGated).0,
            RiskClass::Review
        );

        let mut high = ordinary();
        high.batch_import = true;
        high.confidence = Some(0.9);
        assert_eq!(
            classify_write(&high, ReviewMode::RiskGated).0,
            RiskClass::AutoCommit
        );
    }

    #[test]
    fn weak_provenance_connector_gates_strong_does_not() {
        let mut weak = ordinary();
        weak.source = Some(WriteSource::Connector);
        weak.strong_provenance = false;
        assert_eq!(
            classify_write(&weak, ReviewMode::RiskGated).0,
            RiskClass::Review
        );

        let mut strong = ordinary();
        strong.source = Some(WriteSource::Connector);
        strong.strong_provenance = true;
        assert_eq!(
            classify_write(&strong, ReviewMode::RiskGated).0,
            RiskClass::AutoCommit
        );
    }

    #[test]
    fn mode_label_roundtrip() {
        assert_eq!(ReviewMode::from_label("FAST"), ReviewMode::Fast);
        assert_eq!(ReviewMode::from_label("risk-gated"), ReviewMode::RiskGated);
        assert_eq!(ReviewMode::from_label("paranoid"), ReviewMode::Paranoid);
        assert_eq!(ReviewMode::from_label("garbage"), ReviewMode::RiskGated);
    }

    #[test]
    fn action_resulting_status() {
        assert_eq!(
            MemoryPrAction::Promote.resulting_status(),
            Some(MemoryPrStatus::Promoted)
        );
        assert_eq!(MemoryPrAction::AskAgentWhy.resulting_status(), None);
    }

    #[test]
    fn only_accept_actions_release_the_memory() {
        // B1: accepting a risky write must release it from quarantine.
        assert!(MemoryPrAction::Promote.releases_memory());
        assert!(MemoryPrAction::Merge.releases_memory());
        assert!(MemoryPrAction::Supersede.releases_memory());
        // Rejecting / holding keeps it suppressed.
        assert!(!MemoryPrAction::Forget.releases_memory());
        assert!(!MemoryPrAction::Quarantine.releases_memory());
        assert!(!MemoryPrAction::AskAgentWhy.releases_memory());
    }
}
