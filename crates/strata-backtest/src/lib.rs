//! Offline backtest harness over a signed Strata log.
//!
//! The live binaries are not linked. [`strata_store::StrataStore::as_of`]
//! folds a bounded prefix without appending. This crate opens stores only as
//! copies, ranks with a [`Mechanism`], and writes a canonical JSON manifest
//! that pins the preregistration hash and the signed head of every cut.
//!
//! Gain, Need, and the Q layer are the model under test. They are not a
//! claim about what the live fold already does.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod canon;
pub mod cuts;
pub mod events;
pub mod manifest;
pub mod mechanism;
pub mod open;
pub mod protocol;
pub mod rng;

pub use events::{
    Body, DreamWindow, Rec, candidates, card_handle, events_from_admitted, in_dream, rating4_count,
    suffix_review_nodes, suffix_subjects, typed_edge_count,
};
pub use manifest::{CutHead, harness_manifest};
pub use mechanism::{
    IdOrder, Mechanism, Prefix, Proof, Query, Subject, firewall_accepts, project_prefix,
    require_firewall,
};
pub use open::{CopiedStore, copy_tree, open_copy};
pub use protocol::{TABLE_ID, outcome_sign, review_sign};
