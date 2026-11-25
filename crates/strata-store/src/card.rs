//! The store's card fold: reviews plus imported v3 card states.
//!
//! Admitted `ReviewNode` ops, ingest reviews, and the importer's
//! `FSRS_REVIEW` frames fold as kernel [`ReviewEvent`]s. An upgraded v3
//! memory also arrives as an `FSRS_STATE` frame that carries the card
//! itself: its stability, difficulty, and counters are not reproducible
//! from any rating series. [`CardEvent`] puts both on one fold, so a store
//! checkpoint's state root covers imported cards and the kernel verifier
//! ([`strata_kernel::verify::verify_with_head`]) replays them like any other
//! event. Reviews fold under the pinned kernel of the checkpoint's version;
//! an import sets the card once and is identical under every version.

use borsh::{BorshDeserialize, BorshSerialize};
use strata_kernel::canonical::to_q32_32;
use strata_kernel::event::{ReviewEvent, StrataEvent};
use strata_kernel::fsrs::{
    fold_v1, fold_v2, fold_v3, fold_v4, CardPhase, CardState, ALGO_V1, ALGO_V2, ALGO_V3, ALGO_V4,
    D_MAX, D_MIN, S_MAX, S_MIN,
};
use strata_kernel::kernel::{KernelTableEntry, VersionedFold};
use strata_kernel::state::State;

/// An imported card: the state the v3 store scheduled from, keyed by the
/// store's card handle and stamped with the importing frame's seq.
#[derive(Debug, Clone, Copy, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub(crate) struct ImportedCard {
    /// Store card handle (`handle_of(v3 memory id)`).
    pub card_id: u64,
    /// Seq of the `FSRS_STATE` frame. Becomes the card's `last_seq`.
    pub event_seq: u64,
    /// Stability, Q32.32 days. Clamped into the kernel's stability bounds.
    pub stability_q: i64,
    /// Difficulty, Q32.32. Clamped into `[1, 10]`.
    pub difficulty_q: i64,
    /// Reviews v3 recorded.
    pub review_count: u32,
    /// Lapses v3 recorded.
    pub lapse_count: u32,
    /// v3 learning phase.
    pub phase: CardPhase,
}

/// One event on the store's card fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub(crate) enum CardEvent {
    /// A review, folded by the version-pinned FSRS kernel.
    Review(ReviewEvent),
    /// An imported card. Sets the card when it has none; a card that
    /// already exists is left as it is.
    Import(ImportedCard),
    /// Exact stability change. It is not a review: counters, review clock,
    /// phase and the card's last review seq stay unchanged.
    AdjustStability {
        card_id: u64,
        event_seq: u64,
        stability_q: i64,
    },
}

impl CardEvent {
    pub(crate) fn card_id(&self) -> u64 {
        match self {
            Self::Review(event) => event.card_id,
            Self::Import(card) => card.card_id,
            Self::AdjustStability { card_id, .. } => *card_id,
        }
    }
}

impl StrataEvent for CardEvent {
    fn seq(&self) -> u64 {
        match self {
            CardEvent::Review(event) => event.event_seq,
            CardEvent::Import(card) => card.event_seq,
            CardEvent::AdjustStability { event_seq, .. } => *event_seq,
        }
    }
}

fn adjust(state: &mut State, card_id: u64, event_seq: u64, stability_q: i64) {
    state.applied_seq = state.applied_seq.max(event_seq);
    if let Some(card) = state.cards.get_mut(&card_id) {
        card.stability_q = stability_q;
    }
}

fn import(state: &mut State, card: &ImportedCard) {
    state.applied_seq = state.applied_seq.max(card.event_seq);
    state
        .cards
        .entry(card.card_id)
        .or_insert_with(|| CardState {
            stability_q: card.stability_q.clamp(to_q32_32(S_MIN), to_q32_32(S_MAX)),
            difficulty_q: card.difficulty_q.clamp(to_q32_32(D_MIN), to_q32_32(D_MAX)),
            last_seq: card.event_seq,
            review_count: card.review_count,
            lapse_count: card.lapse_count.min(card.review_count),
            phase: card.phase,
        });
}

fn fold_card_v1(state: &mut State, event: &CardEvent) {
    match event {
        CardEvent::Review(review) => fold_v1(state, review),
        CardEvent::Import(card) => import(state, card),
        CardEvent::AdjustStability {
            card_id,
            event_seq,
            stability_q,
        } => adjust(state, *card_id, *event_seq, *stability_q),
    }
}

fn fold_card_v2(state: &mut State, event: &CardEvent) {
    match event {
        CardEvent::Review(review) => fold_v2(state, review),
        CardEvent::Import(card) => import(state, card),
        CardEvent::AdjustStability {
            card_id,
            event_seq,
            stability_q,
        } => adjust(state, *card_id, *event_seq, *stability_q),
    }
}

fn fold_card_v3(state: &mut State, event: &CardEvent) {
    match event {
        CardEvent::Review(review) => fold_v3(state, review),
        CardEvent::Import(card) => import(state, card),
        CardEvent::AdjustStability {
            card_id,
            event_seq,
            stability_q,
        } => adjust(state, *card_id, *event_seq, *stability_q),
    }
}

fn fold_card_v4(state: &mut State, event: &CardEvent) {
    match event {
        CardEvent::Review(review) => fold_v4(state, review),
        CardEvent::Import(card) => import(state, card),
        CardEvent::AdjustStability {
            card_id,
            event_seq,
            stability_q,
        } => adjust(state, *card_id, *event_seq, *stability_q),
    }
}

impl VersionedFold for CardEvent {
    fn kernel_table() -> &'static [KernelTableEntry<Self>] {
        &[
            (ALGO_V1, fold_card_v1 as fn(&mut State, &CardEvent)),
            (ALGO_V2, fold_card_v2 as fn(&mut State, &CardEvent)),
            (ALGO_V3, fold_card_v3 as fn(&mut State, &CardEvent)),
            (ALGO_V4, fold_card_v4 as fn(&mut State, &CardEvent)),
        ]
    }
}
