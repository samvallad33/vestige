//! The store's data-frame payload: one borsh enum covering every mutation the
//! store can land. Each admitted write appends exactly one `STORE_WRITE`
//! frame whose payload is `borsh(StoreOp)` and whose blake3 digest equals the
//! admitting `EFFECT.payload_digest`.

use crate::types::{ConnectionRecord, NodeRecord};
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
    ReviewNode {
        /// Card id (the blake3-derived u64 handle of the node id).
        card_id: u64,
        /// Rating 1..=4 (clamped by the kernel fold if outside).
        rating: u8,
    },
    /// One sanctioned source re-derivation transaction (PR 2): land the new
    /// node, the lineage edge (supersedes/corrects), and the retire mark in
    /// a single admitted batch. The gate admits it as a WRITE because the
    /// source-key contract (same key, strictly later source_updated_at) is
    /// the sanctioned supersede path; a plain RETIRE still holds.
    ReplaceBySource {
        /// The new node record.
        new_record: crate::types::NodeRecord,
        /// The old node id.
        old_id: String,
        /// The lineage edge (new -> old).
        edge: crate::types::ConnectionRecord,
    },
}
