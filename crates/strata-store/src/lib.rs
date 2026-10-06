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
//! * Ordering stays sequence-based. `created_at_ms` is caller-supplied
//!   (default 0). An explicit [`StoreOp::ReviewNode`] always carries
//!   `reviewed_at_ms` (`borsh` `Option`, tag present). Live admits use the
//!   admission clock. Retrievability reads that clock; `None` uses seq distance.
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
//! An upgraded store also replays the v3 importer's frames: nodes and edges
//! (sharing `32`/`33`), `0x24` supersessions, and FSRS state. `0x22`
//! `FSRS_REVIEW` folds a v3 `fsrs_cards` rating series and `0x27`
//! `FSRS_STATE` folds a v3 card carried from `knowledge_nodes`, both onto
//! the node's card handle and into the checkpointed fold.
//!
//! ## v1 scope
//!
//! Single-writer (`Send` via the gate-log mutex); reads append
//! nothing (reads-as-writes is a later wave); one FSRS kernel version
//! (`ALGO_V2`) per store, recorded per record as `kernel_id`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod anchor;
pub mod canonical;
mod card;
mod error;
mod gate_log;
mod ghostlink;
mod op;
mod store;
mod types;

#[cfg(test)]
mod ghostlink_tests;
#[cfg(test)]
mod import_tests;
#[cfg(test)]
mod tests;

pub use canonical::{
    canonical_hash, canonical_hash_hex, canonicalize, intent_digest, CANONICAL_PIPELINE_VERSION,
    DUPLICATE_SOURCE,
};
pub use error::StoreError;
pub use gate_log::StrataEventLog;
pub use ghostlink::{
    composition_pair, parse_weave_source, weave_source, BridgeCandidate, BridgeReport,
    CompositionRecord, DivergentEval, DivergentPage, DivergentSummary, GhostSnapshot, Lane,
    PathStep, PathVia, PoolFilter, ADMITTING_KINDS, BEYOND_RADIUS, BRIDGE_MAX_HOPS,
    COMPOSITION_NODE_TYPE, DIVERGENT_RADIUS, GHOSTLINK_TAG, LEGACY_INFERRED, LENS_TAG_PREFIX,
    MEASURED_MEMBER_CAP, OUTCOME_TAG_PREFIX, WEAVE_SOURCE_PREFIX, WEAVE_TAG,
};
pub use op::{StoreOp, KIND_STORE_CHECKPOINT, KIND_STORE_WRITE};
pub use store::{
    default_policy, effect_receipt_id, permissive_policy, retire_rule_id, AdmissionContext,
    EffectAction, EffectProof, NodeWrite, RecordedOrigin, Refold, RetireReceipt, StrataStore,
    SupersedeHop, RULE_EDIT, RULE_INTENTIONS, RULE_PURGE, RULE_SUPPRESS,
};
pub use types::{
    looks_like_failure, AnchorRecord, ConnectionRecord, EdgeDirection, EdgeKind, IngestInput,
    IntentionRecord, NodeRecord, SourceKey, TYPED_EDGE_VOCABULARY, VALID_FOREVER_MS,
};
