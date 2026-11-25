//! Checkpoints: the hash-chained anchors between the event log and the
//! folded state.

use borsh::{BorshDeserialize, BorshSerialize};

use crate::state::State;

/// Magic prefix binding the checkpoint format (`"STRATAv1"`).
///
/// Bumping the format (field layout changes) means a new magic, not an edit.
pub const MAGIC: [u8; 8] = *b"STRATAv1";

/// A checkpoint anchors the state fold at a log position.
///
/// Canonical borsh layout (fixed order): `magic: [u8;8]`,
/// `algo_version: u32`, `log_seq: u64`, `prev_checkpoint: [u8;32]`,
/// `state_root: [u8;32]`.
///
/// * `log_seq` — every event with `seq <= log_seq` has been folded into
///   `state_root`.
/// * `prev_checkpoint` — `checkpoint_hash` of the preceding checkpoint;
///   `[0; 32]` for the genesis checkpoint.
/// * `state_root` — see [`state_root`].
///
/// `checkpoint_hash(c) = blake3(borsh(c))` ([`Checkpoint::hash`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct Checkpoint {
    /// Format magic, must equal [`MAGIC`].
    pub magic: [u8; 8],
    /// Algorithm version used to fold events up to this checkpoint.
    pub algo_version: u32,
    /// All events with `seq <= log_seq` are reflected in `state_root`.
    pub log_seq: u64,
    /// `checkpoint_hash` of the previous checkpoint; all zeros for genesis.
    pub prev_checkpoint: [u8; 32],
    /// Root of the folded state at `log_seq`.
    pub state_root: [u8; 32],
}

/// Root hash of a folded [`State`].
///
/// ```text
/// state_root(state) = blake3( concat_{card_id ASCENDING} blake3(borsh((card_id, CardState))) )
/// ```
///
/// `State.cards` is a `BTreeMap`, so iteration IS the required ascending
/// order. Each card contributes exactly one 32-byte subhash, so applying one
/// more event to one card is an incremental append of one subhash — this is
/// the "per-card subhash = incremental apply" property. The root of an empty
/// state is `blake3("")`.
pub fn state_root(state: &State) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for (card_id, card) in &state.cards {
        let entry = borsh::to_vec(&(*card_id, card))
            .expect("borsh serialize of canonical state is infallible");
        hasher.update(blake3::hash(&entry).as_bytes());
    }
    // Commit FSRS-7 (`ALGO_V4`) short-term stabilities when present.
    // Empty under `ALGO_V1`..=`ALGO_V3`, preserving all historical state roots bit-exact.
    if !state.short_stability_q.is_empty() {
        for (card_id, s_short_q) in &state.short_stability_q {
            let entry = borsh::to_vec(&(b"FSRS7_SS", *card_id, *s_short_q))
                .expect("borsh serialize of FSRS-7 short stability is infallible");
            hasher.update(blake3::hash(&entry).as_bytes());
        }
    }
    *hasher.finalize().as_bytes()
}

/// Hash of a checkpoint: `blake3(borsh(checkpoint))`.
pub fn checkpoint_hash(checkpoint: &Checkpoint) -> [u8; 32] {
    let bytes = borsh::to_vec(checkpoint).expect("borsh serialize of Checkpoint is infallible");
    *blake3::hash(&bytes).as_bytes()
}

impl Checkpoint {
    /// The genesis checkpoint of an empty log under `algo_version`:
    /// `log_seq = 0`, `prev = [0;32]`, root of the empty state.
    pub fn genesis(algo_version: u32) -> Self {
        Checkpoint {
            magic: MAGIC,
            algo_version,
            log_seq: 0,
            prev_checkpoint: [0; 32],
            state_root: state_root(&State::default()),
        }
    }

    /// Seal a checkpoint over `state` at `log_seq`, chaining onto `prev`
    /// (the hash of the previous checkpoint).
    pub fn seal(algo_version: u32, log_seq: u64, prev: [u8; 32], state: &State) -> Self {
        Checkpoint {
            magic: MAGIC,
            algo_version,
            log_seq,
            prev_checkpoint: prev,
            state_root: state_root(state),
        }
    }

    /// `checkpoint_hash(self)`.
    pub fn hash(&self) -> [u8; 32] {
        checkpoint_hash(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{ReviewEvent, StrataEvent};
    use crate::fsrs::{fold_v1, V1_WEIGHTS_MILLI};
    use crate::state::State;

    #[test]
    fn empty_state_root_is_blake3_of_nothing() {
        assert_eq!(state_root(&State::default()), *blake3::hash(b"").as_bytes());
    }

    #[test]
    fn genesis_matches_spec() {
        let g = Checkpoint::genesis(1);
        assert_eq!(g.magic, MAGIC);
        assert_eq!(g.log_seq, 0);
        assert_eq!(g.prev_checkpoint, [0u8; 32]);
        assert_eq!(g.state_root, state_root(&State::default()));
    }

    #[test]
    fn subhash_is_incremental_per_card() {
        let mut s1 = State::default();
        fold_v1(
            &mut s1,
            &ReviewEvent {
                card_id: 1,
                rating: 3,
                event_seq: 1,
            },
        );
        fold_v1(
            &mut s1,
            &ReviewEvent {
                card_id: 5,
                rating: 4,
                event_seq: 2,
            },
        );

        // Same cards, different order of application -> same root (order
        // independence comes from ascending-id hashing, not apply order).
        let mut s2 = State::default();
        fold_v1(
            &mut s2,
            &ReviewEvent {
                card_id: 5,
                rating: 4,
                event_seq: 2,
            },
        );
        fold_v1(
            &mut s2,
            &ReviewEvent {
                card_id: 1,
                rating: 3,
                event_seq: 1,
            },
        );
        assert_eq!(state_root(&s1), state_root(&s2));

        // Manual concat-of-subhashes matches state_root.
        let mut h = blake3::Hasher::new();
        for id in [1u64, 5u64] {
            let e = borsh::to_vec(&(id, s1.cards[&id].clone())).unwrap();
            h.update(blake3::hash(&e).as_bytes());
        }
        assert_eq!(state_root(&s1), *h.finalize().as_bytes());
    }

    #[test]
    fn weights_are_referenced_not_duplicated() {
        // Guard: v1 table stays 21 slots (compile-time length is already
        // enforced by the type; sanity-check values are milli-range).
        assert_eq!(V1_WEIGHTS_MILLI.len(), 21);
        assert!(V1_WEIGHTS_MILLI.iter().all(|w| *w < 100_000));
    }

    #[test]
    fn checkpoint_hash_is_borsh_blake3() {
        let c = Checkpoint::genesis(1);
        let expect = *blake3::hash(&borsh::to_vec(&c).unwrap()).as_bytes();
        assert_eq!(c.hash(), expect);
    }

    #[test]
    fn seq_trait_agrees_with_field() {
        let e = ReviewEvent {
            card_id: 9,
            rating: 2,
            event_seq: 42,
        };
        assert_eq!(StrataEvent::seq(&e), e.event_seq);
    }
}
