//! The folded state the kernel maintains and hashes.

use borsh::{BorshDeserialize, BorshSerialize};
use std::collections::BTreeMap;

use crate::fsrs::CardState;
use crate::lint_state_type;

/// Kernel state: the deterministic fold of every applied event.
///
/// `applied_seq` is the seq of the last applied event (`0` = none applied
/// yet); it is bookkeeping for callers and is NOT part of
/// [`crate::checkpoint::state_root`], which is defined over the card map
/// only (per contract).
///
/// `cards` is a `BTreeMap`, so borsh serialization is always in ascending
/// `card_id` order — that ordering IS the canonical form.
#[derive(Debug, Clone, Default, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct State {
    /// Seq of the last applied event (0 = nothing applied).
    pub applied_seq: u64,
    /// Per-card folded state, canonically ordered by ascending `card_id`.
    pub cards: BTreeMap<u64, CardState>,
    /// Per-card short-term stability `S_short` (Q32.32 days) for `ALGO_V4` (FSRS-7).
    /// Always empty under `ALGO_V1`..=`ALGO_V3` so prior state roots stay bit-identical.
    pub short_stability_q: BTreeMap<u64, i64>,
}

lint_state_type!(State {
    applied_seq: u64,
    cards: BTreeMap<u64, CardState>,
    short_stability_q: BTreeMap<u64, i64>,
});
