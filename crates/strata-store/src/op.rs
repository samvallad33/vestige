//! The store's data-frame payload: one borsh enum covering every mutation the
//! store can land. Each admitted write appends exactly one `STORE_WRITE`
//! frame whose payload is `borsh(StoreOp)` and whose blake3 digest equals the
//! admitting `EFFECT.payload_digest`.

use crate::types::{AnchorRecord, ConnectionRecord, IntentionRecord, NodeRecord, RunRecord};
use borsh::{BorshDeserialize, BorshSerialize};

/// Frame kind: a store data frame (payload = `borsh(StoreOp)`).
///
/// Kinds 1..=7 belong to strata-gate records; store frames start at 32 to
/// leave room for further gate duties.
pub const KIND_STORE_WRITE: u8 = 32;

/// Frame kind: a sealed FSRS checkpoint (payload =
/// `borsh(strata_kernel::Checkpoint)`).
pub const KIND_STORE_CHECKPOINT: u8 = 33;

/// One admitted mutation. The FSRS review event for an ingest is folded
/// deterministically from the frame seq (see `store::StrataStore`), so it
/// carries no explicit event here; explicit reviews use [`StoreOp::ReviewNode`].
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub enum StoreOp {
    /// Create or rewrite a node record (ingest and `set_created_at` path).
    /// A brand-new id folds one ingest `ReviewEvent` into the kernel state.
    UpsertNode {
        /// The full record; `id` is the registry key.
        record: NodeRecord,
    },
    /// Append one typed edge and update the forward/reverse indexes.
    SaveEdge {
        /// The edge; `link_type` was vocabulary-validated at write time.
        edge: ConnectionRecord,
    },
    /// Mark `id` as superseded by `superseded_by` (gate action RETIRE).
    SupersedeNode {
        /// The node being retired.
        id: String,
        /// The superseding node.
        superseded_by: String,
    },
    /// Fold an explicit FSRS review for a card.
    ///
    /// Wire, after the `u8` discriminant `3`: `card_id: u64`, `rating: u8`,
    /// then `borsh(Option<i64>)` for `reviewed_at_ms`. The option tag is
    /// always present. A payload that ends after `rating` does not decode.
    /// The `i64` sits in the `STORE_WRITE` payload, so the existing
    /// `payload_blake3` and segment signature already cover it.
    ReviewNode {
        /// Card id (the blake3-derived u64 handle of the node id).
        card_id: u64,
        /// Rating 1..=4 (clamped by the kernel fold if outside).
        rating: u8,
        /// Unix epoch milliseconds of the review. `None` when the source
        /// had no last-review time. Still encoded (option tag `0`).
        reviewed_at_ms: Option<i64>,
    },
    /// Insert or replace intention rows. One admitted effect covers the
    /// whole batch so a check claim is all-or-nothing. Intentions are not
    /// memory cards: applying this op does not fold a review.
    UpsertIntentions {
        /// Records keyed by `id`.
        records: Vec<IntentionRecord>,
    },
    // Variants below were appended for code anchors. Borsh discriminants are
    // positional: never reorder, only append, or old logs stop replaying.
    /// Insert or replace code anchors by anchor id (discriminant `5`). One
    /// admitted effect covers the batch. Anchors are not memory cards:
    /// applying this op folds no review.
    RecordAnchors {
        /// Anchors keyed by `id`; each names its own `node_id`.
        anchors: Vec<AnchorRecord>,
    },
    /// Drop every anchor of `node_id`, then insert `anchors` (discriminant
    /// `6`). The memory itself is not touched.
    ReplaceAnchors {
        /// The memory whose anchors are replaced.
        node_id: String,
        /// The replacement set; every row names `node_id`.
        anchors: Vec<AnchorRecord>,
    },
    /// Cache one anchor's latest verification verdict (discriminant `7`).
    /// Informational: readers re-verify against the live tree.
    RecordAnchorVerdict {
        /// The anchor checked.
        anchor_id: String,
        /// Verdict string (`verified`, `moved`, `drifted`, `missing`,
        /// `unverifiable`).
        status: String,
        /// Check time (unix ms), supplied by the caller.
        checked_at_ms: i64,
    },
    /// Insert or replace run records by `run_id` (discriminant `8`).
    ///
    /// One admitted effect covers the batch. Runs are not memory cards:
    /// applying this op folds no review. Appended after every earlier
    /// variant so logs written before runs existed still decode.
    RecordRuns {
        /// Records keyed by `run_id`.
        runs: Vec<RunRecord>,
    },
}
