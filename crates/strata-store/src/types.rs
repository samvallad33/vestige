//! Store-level data types: the node registry record, ingest input, the typed
//! connection record, and the owner-approved 9-type edge vocabulary (same
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
    "reopened",
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
    /// later failure on the same exact handle after a closes edge
    Reopened,
}

impl EdgeKind {
    /// Every vocabulary member, in [`TYPED_EDGE_VOCABULARY`] order.
    pub const ALL: [EdgeKind; 9] = [
        EdgeKind::Touched,
        EdgeKind::AnchoredTo,
        EdgeKind::DerivedFrom,
        EdgeKind::Supersedes,
        EdgeKind::Corrects,
        EdgeKind::ClosedBy,
        EdgeKind::ProjectedTo,
        EdgeKind::EvidenceOf,
        EdgeKind::Reopened,
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
            EdgeKind::Reopened => "reopened",
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
    /// Source provenance key, when the memory came from a connector or a v3 row.
    pub source: Option<SourceKey>,
    /// Source row's last-updated timestamp (unix ms).
    pub source_updated_at_ms: Option<i64>,
}

impl NodeRecord {
    /// Is this record live (not superseded)?
    pub fn is_live(&self) -> bool {
        self.superseded_by.is_none()
    }
}

/// Provenance key for re-derivation: `(source_system, source_project,
/// source_id)`. Re-deriving the same key with a later `source_updated_at`
/// is a sanctioned supersede path — same source, same fact slot.
///
/// A v3 row that only has the free-form `source` text (no
/// `source_system` / `source_id`) is stored as `system = <that text>` with
/// empty `project` and `id`. `None` means the row had no source.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct SourceKey {
    /// Connector/system name (exact bytes; never case-folded).
    pub system: String,
    /// Project namespace within the system.
    pub project: String,
    /// The source row id.
    pub id: String,
}

/// Input for creating a new memory (store-local mirror of the vestige-core
/// ingest input; float sentiment fields are dropped — no floats in persisted
/// state).
#[derive(Debug, Clone, Default, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct IngestInput {
    /// The content to memorize.
    pub content: String,
    /// Source provenance key, when the memory came from a connector or a v3 row.
    pub source: Option<SourceKey>,
    /// Source row's last-updated timestamp (unix ms) for re-derivation.
    pub source_updated_at_ms: Option<i64>,
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

/// One prospective intention. Timestamps are unix milliseconds supplied by
/// the caller; this store does not read a clock. Trigger text is stored
/// verbatim — matching stays in the tool, and it is exact or structural.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct IntentionRecord {
    /// Caller-supplied id (the tool's uuid). The registry key.
    pub id: String,
    /// What to remember to do.
    pub content: String,
    /// `time`, `context`, `event`, `activity`, `recurring`, `compound`, or `manual`.
    pub trigger_type: String,
    /// Canonical trigger JSON produced by the tool.
    pub trigger_data: String,
    /// 1=low, 2=normal, 3=high, 4=critical.
    pub priority: i32,
    /// `active`, `fulfilled`, `cancelled`, or `snoozed`.
    pub status: String,
    /// Creation time (unix ms).
    pub created_at_ms: i64,
    /// Optional deadline (unix ms).
    pub deadline_ms: Option<i64>,
    /// Set when status becomes `fulfilled`.
    pub fulfilled_at_ms: Option<i64>,
    /// How many times a check has delivered this intention.
    pub reminder_count: i32,
    /// Last delivery time (unix ms).
    pub last_reminded_at_ms: Option<i64>,
    /// Free-form notes. Unused by `set`.
    pub notes: Option<String>,
    /// Tags, in caller order.
    pub tags: Vec<String>,
    /// Related memory ids, in caller order.
    pub related_memories: Vec<String>,
    /// Snooze deadline (unix ms).
    pub snoozed_until_ms: Option<i64>,
    /// `mcp` or `nlp`.
    pub source_type: String,
    /// Optional source payload.
    pub source_data: Option<String>,
    /// Project namespace. Blank resolves to `user` on read.
    pub scope: Option<String>,
}

impl IntentionRecord {
    /// The columns a check compares before it claims a delivery.
    ///
    /// Same predicate as the SQLite `UPDATE ... WHERE` in
    /// `commit_intention_check`: identity, notes, tags, and scope are not
    /// part of the claim.
    pub fn same_claim(&self, other: &Self) -> bool {
        self.trigger_type == other.trigger_type
            && self.trigger_data == other.trigger_data
            && self.status == other.status
            && self.reminder_count == other.reminder_count
            && self.last_reminded_at_ms == other.last_reminded_at_ms
            && self.snoozed_until_ms == other.snoozed_until_ms
            && self.content == other.content
            && self.priority == other.priority
            && self.deadline_ms == other.deadline_ms
            && self.fulfilled_at_ms == other.fulfilled_at_ms
    }
}

/// One code anchor: a pointer from a memory into source, plus the
/// fingerprint a later check compares against the working tree.
///
/// Mirror of vestige-core's `CodeAnchor` with integer times (unix ms). The
/// store never hashes source and never reads a clock: every field is the
/// caller's. `last_status` is the persisted verdict string (`verified`,
/// `moved`, `drifted`, `missing`, `unverifiable`); `None` means never checked.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct AnchorRecord {
    /// Anchor id (the registry key). Unique across nodes.
    pub id: String,
    /// The memory this anchor belongs to.
    pub node_id: String,
    /// Repository-relative (or absolute) path of the anchored file.
    pub file_path: String,
    /// Symbol the memory is about, if any.
    pub symbol: Option<String>,
    /// Display kind of the symbol (`fn`, `class`, ...).
    pub symbol_kind: Option<String>,
    /// Capture-time first line of the span (1-based). A reporting hint.
    pub start_line: Option<u32>,
    /// Capture-time last line of the span (1-based, inclusive).
    pub end_line: Option<u32>,
    /// Lines the hash covers. `None` means the anchor cannot be checked.
    pub span_lines: Option<u32>,
    /// Versioned content fingerprint. `None` means the anchor cannot be checked.
    pub content_hash: Option<String>,
    /// Capture time (unix ms).
    pub captured_at_ms: i64,
    /// Time of the latest recorded verification (unix ms).
    pub last_verified_at_ms: Option<i64>,
    /// Latest recorded verdict.
    pub last_status: Option<String>,
}

/// Does this memory carry a recorded failure label?
///
/// Failure is recorded structure, not a vocabulary guess: the writer tags the
/// memory `failure` (case-insensitive, or a namespaced `failure:<detail>` tag).
/// Content words are never scanned, so a quiet cause that happens to say
/// "timeout" or "error" is not misread as the failure itself. Callers that hold
/// a full node also honour `node_type == "failure"` and recorded
/// `corrects`/`supersedes`/`closes` links.
pub fn looks_like_failure(_content: &str, tags: &[String]) -> bool {
    tags.iter().any(|t| {
        let t = t.trim();
        t.eq_ignore_ascii_case("failure")
            || t.get(..8).is_some_and(|p| p.eq_ignore_ascii_case("failure:"))
    })
}

