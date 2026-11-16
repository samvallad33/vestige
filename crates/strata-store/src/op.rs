//! The store's data-frame payload: one borsh enum covering every mutation the
//! store can land. Each admitted write appends exactly one `STORE_WRITE`
//! frame whose payload is `borsh(StoreOp)` and whose blake3 digest equals the
//! admitting `EFFECT.payload_digest`.

use crate::types::{AnchorRecord, ConnectionRecord, IntentionRecord, NodeRecord};
use crate::BackfillAdjustment;
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
    /// Append several typed edges at once (discriminant `8`). One admitted
    /// effect covers the batch, so the edges one write adds land together or
    /// not at all, behind one gate decision and one data frame. Applied in
    /// order, exactly as that many [`StoreOp::SaveEdge`] would be.
    ///
    /// A build older than this variant does not decode the frame: it counts
    /// it as an orphan write and does not see these edges.
    SaveEdges {
        /// The edges; each `link_type` was vocabulary-validated at write time.
        edges: Vec<ConnectionRecord>,
    },
    /// Exact, bounded stability change with its entity evidence (discriminant
    /// `9`). Appended to preserve every pre-existing log's wire encoding.
    AdjustSalience {
        /// Request with the actual bounded after-score.
        adjustment: BackfillAdjustment,
        /// Exact before-score, compared again during replay.
        old_stability_q: i64,
        /// Latest card event at admission; guards conflicting score changes.
        prior_card_event_seq: u64,
        /// Original effect receipt when this is an exact compensating undo.
        undo_of: Option<String>,
    },
    /// Admit a remote writer. Discriminant 10, appended so earlier logs keep
    /// their tags. The gate allows this only under the named mesh-grant rule.
    GrantWriter {
        /// did:key of the writer's log verifying key.
        writer_did: String,
        /// Pinned 32-byte ed25519 verifying key. Not a seed.
        verifying_key: [u8; 32],
        /// blake3 of the mesh policy this grant pins.
        policy_hash: [u8; 32],
        /// Scope names. Empty means the whole store.
        scopes: Vec<String>,
        /// Writer signature binding the log key to the transport endpoint id.
        endpoint_binding: Vec<u8>,
    },
    /// One contiguous admitted range of a granted writer's chain.
    MergeChain {
        /// Writer whose frames `from_seq..=to_seq` were admitted.
        writer_did: String,
        /// First newly admitted frame seq.
        from_seq: u64,
        /// Last newly admitted frame seq.
        to_seq: u64,
        /// Borsh of the signed head the range was checked against.
        signed_head: Vec<u8>,
        /// blake3 of the concatenated consistency-proof nodes.
        consistency_proof_hash: [u8; 32],
        /// blake3 of the re-derived verdict list.
        rederive_hash: [u8; 32],
        /// Mesh digest after this range was folded.
        mesh_digest_after: [u8; 32],
        /// Claim pairs this range left in conflict.
        conflicts: Vec<ClaimRef>,
    },
    /// A claim-slot contradiction. Both claims stay; this receipt names them.
    ConflictRecord {
        /// Subject the two claims share.
        subject_id: String,
        /// Field the two claims share.
        field_id: String,
        /// One side. Order is the lower frame hash first.
        left: ClaimRef,
        /// The other side.
        right: ClaimRef,
    },
    /// A value asserted for one claim slot.
    AssertClaim {
        /// Subject handle the slot belongs to.
        subject_id: String,
        /// Field inside the subject.
        field_id: String,
        /// Canonical bytes of the asserted value.
        value: Vec<u8>,
        /// Validity start, unix milliseconds.
        valid_from_ms: i64,
        /// Validity end, unix milliseconds.
        valid_until_ms: i64,
    },
    /// Two signed heads for one seq under one writer key.
    Equivocation {
        /// Writer that signed both heads.
        writer_did: String,
        /// First signed head, borsh.
        head_a: Vec<u8>,
        /// Second signed head, borsh.
        head_b: Vec<u8>,
    },
    /// The peer's mesh policy hash is not the local one.
    PolicyMismatch {
        /// Local policy hash.
        local_hash: [u8; 32],
        /// Peer's policy hash.
        peer_hash: [u8; 32],
    },
    /// A scheduling hint named a handle the segment bytes do not contain.
    IndexMismatch {
        /// Sender that offered the hint.
        sender_did: String,
        /// Segment the hint named.
        segment: u64,
        /// Handle the hint named.
        handle: String,
    },
    /// Local proof that a memory's data key was destroyed.
    TombstoneReceipt {
        /// Memory whose key was destroyed.
        memory_id: String,
        /// Writer whose purge tombstone was folded, when this is a peer purge.
        writer_did: String,
        /// Peer's frame seq of the purge, or 0 for a local purge.
        peer_seq: u64,
    },
    /// A whole peer range was refused. Nothing in the range was folded.
    DenyRange {
        /// Writer whose range was refused.
        writer_did: String,
        /// Each frame the re-derivation or the sweep rejected.
        failures: Vec<FrameFail>,
    },
    /// Genesis of a writer chain forked off another device's key.
    ForkGenesis {
        /// Writer this chain was derived from.
        derived_from_did: String,
        /// That writer's seq at the fork point.
        derived_from_seq: u64,
        /// That writer's frame hash at the fork point.
        derived_from_frame: [u8; 32],
    },
    /// A purge of one memory. Folding a peer's tombstone destroys the local key.
    PurgeTombstone {
        /// Memory being purged.
        memory_id: String,
    },
}

/// One side of a claim, named so two devices can cite the same frame.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct ClaimRef {
    /// Writer that recorded the claim.
    pub writer_did: String,
    /// Data-frame seq of the claim.
    pub seq: u64,
    /// Chain hash of that frame.
    pub frame_hash: [u8; 32],
}

/// One frame a range denial names.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct FrameFail {
    /// Writer of the failing frame.
    pub writer_did: String,
    /// Log seq of the failing frame.
    pub seq: u64,
}

/// Ops that carry another writer's evidence and must not themselves produce
/// a further merge receipt when a batch contains only these.
pub fn is_mesh_receipt(op: &StoreOp) -> bool {
    matches!(
        op,
        StoreOp::GrantWriter { .. }
            | StoreOp::MergeChain { .. }
            | StoreOp::ConflictRecord { .. }
            | StoreOp::Equivocation { .. }
            | StoreOp::PolicyMismatch { .. }
            | StoreOp::IndexMismatch { .. }
            | StoreOp::TombstoneReceipt { .. }
            | StoreOp::DenyRange { .. }
            | StoreOp::ForkGenesis { .. }
    )
}
