//! # strata-store — the Causal Proof Engine's memory surface
//!
//! The vestige tool surface implemented over the proof stack: every
//! mutation gate-admitted, every reopen bit-identical.
//!
//! The SQLite wipe's core data layer: an implementation of the vestige memory
//! surface whose ONLY source of truth is a [`strata::StrataLog`] — an
//! append-only, hash-chained, signed segment log. Everything the store answers
//! from (node registry, typed-edge indexes, FSRS fold) is a derived index
//! rebuilt by replay on open.
//!
//! ## Write path (every mutation)
//!
//! ```text
//! build StoreOp -> PROPOSE -> compute_inputs -> GATE (policy verdict)
//!              -> EFFECT (admitted: verdict Allow + chain checks)
//!              -> STORE_WRITE data frame (payload = borsh(StoreOp))
//!              -> apply to in-memory derived maps
//! ```
//!
//! A mutation whose gate verdict is not `Allow` never appends an `EFFECT` and
//! never writes a data frame: denied/hold writes leave at most a `PROPOSE` and
//! a rejecting `GATE` in the log and change nothing. On replay, a `STORE_WRITE`
//! frame is applied ONLY when an admitted `EFFECT` frame earlier in the log
//! carries `payload_digest == blake3(payload)` — unadmitted data frames are
//! counted as orphans and ignored.
//!
//! ## Determinism
//!
//! * No clocks: `created_at_ms` is caller-supplied data (default 0), never a
//!   hidden `SystemTime::now()` read. Time-as-seq is the ordering authority.
//! * No floats in persisted state: edge strength is `strength_milli` (i64);
//!   FSRS stability/difficulty live quantized inside the kernel's Q32.32
//!   `CardState`. Retrievability is derived on read and never stored.
//! * Node ids are derived from the log head (`mem-<next_seq>`), so replay
//!   reconstructs identical ids.
//!
//! ## Frame kinds
//!
//! `1..=7` are the strata-gate record kinds (PROPOSE/GATE/EFFECT/...).
//! `32` = `STORE_WRITE` (payload `borsh(StoreOp)`), `33` = `STORE_CHECKPOINT`
//! (payload `borsh(strata_kernel::Checkpoint)`).
//!
//! ## v1 scope (documented deviations in SCOPE-HANDOFF.md)
//!
//! Single-threaded single-writer (`!Send` gate-log cache); reads append
//! nothing (reads-as-writes is a later wave); one FSRS kernel version
//! (`ALGO_V2`) per store, recorded per record as `kernel_id`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod admission;
pub mod claim_slots;
mod error;
mod gate_log;
pub mod kinds;
mod op;
mod store;
pub mod types;

#[cfg(test)]
mod tests;

pub use error::StoreError;
pub use gate_log::StrataEventLog;
pub use op::{StoreOp, KIND_STORE_CHECKPOINT, KIND_STORE_WRITE};
pub use store::{default_policy, StrataStore};
pub use types::{
    looks_like_failure, ConnectionRecord, EdgeDirection, EdgeKind, IngestInput, NodeRecord,
    TYPED_EDGE_VOCABULARY, VALID_FOREVER_MS,
};
