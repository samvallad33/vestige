//! Store-level data types: the node registry record, ingest input, the typed
//! connection record, and the owner-approved 8-type edge vocabulary (same
//! vocabulary as `vestige-core`'s V39 edges, re-declared so this crate stays
//! standalone).
//!
//! Determinism rules: persisted records carry integers and strings only — no
//! floats (edge strength is milli-units), no wall-clock reads, no random ids.

use borsh::{BorshDeserialize, BorshSerialize};

/// `valid_until_ms` sentinel: the record never stops being valid.
pub const VALID_FOREVER_MS: i64 = i64::MAX;

/// The owner-approved typed-edge vocabulary, in canonical order (mirrors
/// `vestige-core` `storage::edges::TYPED_EDGE_VOCABULARY`).
pub const TYPED_EDGE_VOCABULARY: &[&str] = &[
    "touched",
    "anchored_to",
    "derived_from",
    "supersedes",
    "corrects",
    "closed_by",
    "projected_to",
    "evidence_of",
];

/// A typed edge kind from the owner-approved vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, BorshSerialize, BorshDeserialize)]
pub enum EdgeKind {
    /// commit -> file/symbol
    Touched,
    /// record <-> file/symbol/span
    AnchoredTo,
    /// record -> record, declared only (never inferred)
    DerivedFrom,
    /// record -> record, review-gated invalidation trail
    Supersedes,
    /// record -> record, review-gated correction trail
    Corrects,
    /// issue -> PR -> commit -> check
    ClosedBy,
    /// projection trail
    ProjectedTo,
    /// walk-trail evidence edge
    EvidenceOf,
}

impl EdgeKind {
    /// Every vocabulary member, in [`TYPED_EDGE_VOCABULARY`] order.
    pub const ALL: [EdgeKind; 8] = [
        EdgeKind::Touched,
        EdgeKind::AnchoredTo,
        EdgeKind::DerivedFrom,
        EdgeKind::Supersedes,
        EdgeKind::Corrects,
        EdgeKind::ClosedBy,
        EdgeKind::ProjectedTo,
        EdgeKind::EvidenceOf,
    ];

    /// The canonical `link_type` string.
    pub fn as_str(self) -> &'static str {
        match self {
            EdgeKind::Touched => "touched",
            EdgeKind::AnchoredTo => "anchored_to",
            EdgeKind::DerivedFrom => "derived_from",
            EdgeKind::Supersedes => "supersedes",
            EdgeKind::Corrects => "corrects",
            EdgeKind::ClosedBy => "closed_by",
            EdgeKind::ProjectedTo => "projected_to",
            EdgeKind::EvidenceOf => "evidence_of",
        }
    }

    /// Parse a `link_type` against the vocabulary; `None` for anything else
    /// (including legacy free-form types like `semantic`).
    pub fn parse(link_type: &str) -> Option<EdgeKind> {
        TYPED_EDGE_VOCABULARY
            .iter()
            .position(|candidate| *candidate == link_type)
            .map(|index| EdgeKind::ALL[index])
    }
}

impl std::fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which side of a node to list edges for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    /// Edges where the node is the `source_id`.
    Outgoing,
    /// Edges where the node is the `target_id`.
    Incoming,
    /// Both orientations.
    Both,
}

/// One memory in the registry. All fields are integers/strings: the derived
/// map digest is computed over exactly this shape, so it must stay canonical.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct NodeRecord {
    /// Stable id derived from the log head at ingest (`mem-<seq-hex>`).
    pub id: String,
    /// strata-kernel FSRS algo version governing this record's review fold
    /// (`strata_kernel::ALGO_V2` for v1 stores).
    pub kernel_id: u32,
    /// Scope namespace ("" = default scope); the scope filter key for
    /// `get_all_nodes_in_scope`.
    pub scope: String,
    /// The memorized content.
    pub content: String,
    /// Categorization tags (sorted + deduplicated at ingest for determinism).
    pub tags: Vec<String>,
    /// Knowledge type ("fact", "concept", ...); "" is normalized to "fact".
    pub node_type: String,
    /// Caller-supplied creation time in unix milliseconds (default 0 — this
    /// store never reads a wall clock).
    pub created_at_ms: i64,
    /// Bitemporal validity start (ms). Defaults to `created_at_ms`.
    pub valid_from_ms: i64,
    /// Bitemporal validity end (ms). Defaults to [`VALID_FOREVER_MS`].
    pub valid_until_ms: i64,
    /// Set when a later record superseded this one: the superseder's id.
    pub superseded_by: Option<String>,
}

impl NodeRecord {
    /// Is this record live (not superseded)?
    pub fn is_live(&self) -> bool {
        self.superseded_by.is_none()
    }

    /// Append-only undo of a create: the compensating `UpsertNode` points
    /// `superseded_by` at the record's own id. Reads hide that tombstone.
    /// A supersession by a different id is not an undo.
    pub fn is_undo_tombstone(&self) -> bool {
        self.superseded_by.as_deref() == Some(self.id.as_str())
    }
}

/// Input for creating a new memory (store-local mirror of the vestige-core
/// ingest input; float sentiment fields are dropped — no floats in persisted
/// state).
#[derive(Debug, Clone, Default, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct IngestInput {
    /// The content to memorize.
    pub content: String,
    /// Knowledge type; empty string defaults to "fact".
    pub node_type: String,
    /// Tags (sorted + deduplicated on ingest).
    pub tags: Vec<String>,
    /// Creation time (unix ms). `None` = 0 (deterministic default).
    pub created_at_ms: Option<i64>,
    /// Validity start override (unix ms). `None` = `created_at_ms`.
    pub valid_from_ms: Option<i64>,
    /// Validity end override (unix ms). `None` = [`VALID_FOREVER_MS`].
    pub valid_until_ms: Option<i64>,
}

/// One typed edge (mirror of the vestige-core connection record; strength is
/// integer milli-units and timestamps are ms integers).
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct ConnectionRecord {
    /// Source node id (must exist as a node).
    pub source_id: String,
    /// Target node id, or an external file/symbol anchor id (targets may
    /// dangle by design — `anchored_to` points at non-memory artifacts).
    pub target_id: String,
    /// Edge strength in milli-units (1000 = 1.0). Non-negative.
    pub strength_milli: i64,
    /// One of [`TYPED_EDGE_VOCABULARY`] (validated at write time).
    pub link_type: String,
    /// Commit/artifact sha the edge was declared against, if any.
    pub meta_sha: Option<String>,
    /// Creation time (unix ms).
    pub created_at_ms: i64,
    /// How often this edge was traversed/activated.
    pub activation_count: i64,
}

impl Default for ConnectionRecord {
    fn default() -> Self {
        Self {
            source_id: String::new(),
            target_id: String::new(),
            strength_milli: 1000,
            link_type: EdgeKind::DerivedFrom.as_str().to_string(),
            meta_sha: None,
            created_at_ms: 0,
            activation_count: 0,
        }
    }
}

impl ConnectionRecord {
    /// Edge strength as a float (derived, never persisted).
    pub fn strength(&self) -> f64 {
        self.strength_milli as f64 / 1000.0
    }
}

/// Whole-word failure markers, ported from vestige-core's
/// `advanced::retroactive_backfill::FAILURE_MARKERS` (same list, same
/// semantics: bare "500" and bare "pinned" stay removed).
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
    "timeout",
    "deadlock",
    "leak",
    "corrupt",
    "stack overflow",
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

/// Whole-word (not substring) marker match, boundary-safe over UTF-8.
fn contains_marker_word(hay: &str, marker: &str) -> bool {
    let mut from = 0usize;
    while let Some(pos) = hay[from..].find(marker) {
        let start = from + pos;
        let end = start + marker.len();
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

/// Does this content/tags pair read like a failure? Compatible with
/// vestige-core's `looks_like_failure` so failure detection does not drift
/// between the SQLite store and this store.
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
