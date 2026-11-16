//! #399 fsrs-kernel properties.
//!
//! The fold is a pure function of `(algorithm, event sequence)`. Stability
//! and difficulty stay inside the quantized `[S_MIN, S_MAX]` and
//! `[D_MIN, D_MAX]` bounds. Retrievability stays in `[0, 1]`, is `1` at the
//! card's last seq, and is monotonically non-increasing as seq grows.
//! `verify` accepts an honest checkpoint chain and rejects a skipped-event
//! mutant, a one-byte `state_root` edit, a broken `prev_checkpoint`, and a
//! wrong head anchor.
//!
//! The generator is an in-file xorshift. `STRATA_PROP_CASES` raises the
//! pinned count. No dev-dependency: this lane adds test files only.

use strata_kernel::canonical::to_q32_32;
use strata_kernel::checkpoint::{checkpoint_hash, state_root, Checkpoint};
use strata_kernel::event::ReviewEvent;
use strata_kernel::fsrs::{FsrsFold, ALGO_V1, ALGO_V2, D_MAX, D_MIN, S_MAX, S_MIN};
use strata_kernel::kernel::kernel_for;
use strata_kernel::state::State;
use strata_kernel::verify::{verify, verify_with_head, VerifyError};

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn cases() -> u64 {
    std::env::var("STRATA_PROP_CASES")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .filter(|n: &u64| *n > 0)
        .unwrap_or(32)
}

fn event(card: u64, rating: u8, seq: u64) -> ReviewEvent {
    ReviewEvent {
        card_id: card,
        rating,
        event_seq: seq,
    }
}

fn fold_all(version: u32, events: &[ReviewEvent]) -> State {
    let apply = kernel_for::<ReviewEvent>(version).expect("pinned version");
    let mut state = State::default();
    for ev in events {
        apply(&mut state, ev);
    }
    state
}

fn hashed(events: &[ReviewEvent]) -> Vec<(u64, [u8; 32], ReviewEvent)> {
    events
        .iter()
        .map(|ev| {
            let bytes = borsh::to_vec(ev).expect("event borsh");
            (*blake3::hash(&bytes).as_bytes(), ev)
        })
        .map(|(hash, ev)| (ev.event_seq, hash, *ev))
        .collect()
}

fn gen_events(seed: u64, n: usize) -> Vec<ReviewEvent> {
    let mut rng = Rng::new(seed);
    let mut seq = 0u64;
    let mut events = Vec::with_capacity(n);
    for _ in 0..n {
        let card = rng.next_u64() % 4;
        let rating = (rng.next_u64() % 6) as u8;
        seq = seq.saturating_add(1 + (rng.next_u64() % 7));
        events.push(event(card, rating, seq));
    }
    events
}

fn assert_bounds(state: &State, version: u32) {
    let s_min = to_q32_32(S_MIN);
    let s_max = to_q32_32(S_MAX);
    let d_min = to_q32_32(D_MIN);
    let d_max = to_q32_32(D_MAX);
    for card in state.cards.values() {
        assert!(
            (s_min..=s_max).contains(&card.stability_q),
            "stability {} outside [{s_min}, {s_max}]",
            card.stability_q
        );
        assert!(
            (d_min..=d_max).contains(&card.difficulty_q),
            "difficulty {} outside [{d_min}, {d_max}]",
            card.difficulty_q
        );
        let at_review = FsrsFold::retrievability(card, card.last_seq, version).expect("R");
        assert_eq!(at_review, 1.0, "R at last_seq");
        let mut prev = at_review;
        let mut cursor = card.last_seq;
        for step in [1u64, 2, 5, 30, 365, 10_000] {
            cursor = cursor.saturating_add(step);
            let score = FsrsFold::retrievability(card, cursor, version).expect("R");
            assert!(
                (0.0..=1.0).contains(&score) && !score.is_nan(),
                "R {score} at seq {cursor}"
            );
            assert!(
                score <= prev + 1e-12,
                "R rose from {prev} to {score} as seq grew"
            );
            prev = score;
        }
    }
}

fn seal_chain(version: u32, events: &[ReviewEvent]) -> Vec<Checkpoint> {
    let apply = kernel_for::<ReviewEvent>(version).expect("version");
    let mut state = State::default();
    let mut checkpoints = Vec::new();
    let mut prev = [0u8; 32];
    for (i, ev) in events.iter().enumerate() {
        apply(&mut state, ev);
        if (i + 1) % 5 == 0 || i + 1 == events.len() {
            let checkpoint = Checkpoint::seal(version, ev.event_seq, prev, &state);
            prev = checkpoint_hash(&checkpoint);
            checkpoints.push(checkpoint);
        }
    }
    checkpoints
}

#[test]
fn prop_fold_is_deterministic_and_bounded() {
    for version in [ALGO_V1, ALGO_V2] {
        for seed in 1..=cases() {
            let n = 4 + (seed as usize % 24);
            let events = gen_events(seed ^ u64::from(version) << 32, n);
            let once = fold_all(version, &events);
            let twice = fold_all(version, &events);
            assert_eq!(
                borsh::to_vec(&once).expect("borsh"),
                borsh::to_vec(&twice).expect("borsh"),
                "version {version} seed {seed}"
            );
            assert_eq!(state_root(&once), state_root(&twice));
            let bytes = borsh::to_vec(&once).expect("borsh");
            let round: State = borsh::from_slice(&bytes).expect("borsh");
            assert_eq!(once, round);
            assert_eq!(state_root(&once), state_root(&round));
            assert_bounds(&once, version);
        }
    }
}

#[test]
fn prop_v1_and_v2_diverge_on_an_easy_review() {
    let events = [event(1, 4, 1)];
    assert_ne!(
        state_root(&fold_all(ALGO_V1, &events)),
        state_root(&fold_all(ALGO_V2, &events)),
        "a version alias would hide a weight change"
    );
}

#[test]
fn prop_same_seq_update_stays_in_bounds() {
    for version in [ALGO_V1, ALGO_V2] {
        let apply = kernel_for::<ReviewEvent>(version).expect("version");
        let mut state = State::default();
        for rating in [0u8, 1, 2, 3, 4, 5, 255] {
            apply(&mut state, &event(7, rating, 10));
            assert_bounds(&state, version);
        }
    }
}

#[test]
fn prop_retrievability_inverse_stays_in_range() {
    for version in [ALGO_V1, ALGO_V2] {
        for target in [
            f64::NAN,
            f64::NEG_INFINITY,
            -1.0,
            0.0,
            0.05,
            0.5,
            0.999,
            1.0,
            2.0,
        ] {
            for days in [0u64, 1, 7, 30, 365, 10_000] {
                let stability =
                    FsrsFold::stability_for_retrievability(target, days, version).expect("S");
                assert!(
                    (S_MIN..=S_MAX).contains(&stability) && !stability.is_nan(),
                    "S {stability} for target {target} days {days} version {version}"
                );
            }
        }
    }
}

#[test]
fn prop_unknown_version_is_refused() {
    assert!(strata_kernel::fsrs::weights_for(0).is_err());
    assert!(strata_kernel::fsrs::weights_for(99).is_err());
    assert!(kernel_for::<ReviewEvent>(99).is_err());
    let state = fold_all(ALGO_V2, &[event(1, 3, 1)]);
    let card = state.cards.values().next().expect("card");
    assert!(FsrsFold::retrievability(card, card.last_seq, 99).is_err());
}

#[test]
fn prop_verify_accepts_honest_chains_and_rejects_one_byte() {
    for version in [ALGO_V1, ALGO_V2] {
        for seed in 1..=cases().min(16) {
            let events = gen_events(seed.wrapping_mul(0x9E37) ^ u64::from(version), 12);
            let checkpoints = seal_chain(version, &events);
            assert!(checkpoints.len() >= 2, "need a chain");
            let records = hashed(&events);
            verify(&checkpoints, records.iter().cloned())
                .unwrap_or_else(|err| panic!("honest chain version {version} seed {seed}: {err}"));
            let head = checkpoint_hash(checkpoints.last().expect("head"));
            verify_with_head(&checkpoints, Some(head), records.iter().cloned())
                .unwrap_or_else(|err| panic!("anchored head: {err}"));

            let mut bad_head = head;
            bad_head[0] ^= 0xff;
            assert!(
                matches!(
                    verify_with_head(&checkpoints, Some(bad_head), records.iter().cloned()),
                    Err(VerifyError::CheckpointHashMismatch { .. })
                ),
                "one-byte head anchor must be refused"
            );

            let mut broken = checkpoints.clone();
            broken[1].prev_checkpoint[3] ^= 0xff;
            assert!(
                matches!(
                    verify(&broken, records.iter().cloned()),
                    Err(VerifyError::ChainBreak { .. })
                ),
                "one-byte prev_checkpoint must break the chain"
            );

            let mut root = checkpoints[0];
            root.state_root[7] ^= 0xff;
            let prefix: Vec<_> = records
                .iter()
                .cloned()
                .filter(|record| record.0 <= root.log_seq)
                .collect();
            assert!(
                matches!(
                    verify(std::slice::from_ref(&root), prefix.into_iter()),
                    Err(VerifyError::StateRootMismatch { .. })
                ),
                "one-byte state_root must fail replay"
            );
        }
    }
}

/// Mutant: the checkpoint was sealed after skipping the middle event.
/// `verify` replays the real sequence and must reject that root.
#[test]
fn mutant_skipped_review_is_killed() {
    let events = gen_events(0x0F5E5, 9);
    let skip = events.len() / 2;
    let apply = kernel_for::<ReviewEvent>(ALGO_V2).expect("v2");
    let mut mutant = State::default();
    for (i, ev) in events.iter().enumerate() {
        if i == skip {
            continue;
        }
        apply(&mut mutant, ev);
    }
    let checkpoint = Checkpoint::seal(
        ALGO_V2,
        events.last().expect("last").event_seq,
        [0; 32],
        &mutant,
    );
    let err = verify(&[checkpoint], hashed(&events).into_iter()).expect_err("mutant must fail");
    assert!(
        matches!(err, VerifyError::StateRootMismatch { .. }),
        "skipped review was not killed: {err}"
    );
    let honest = Checkpoint::seal(
        ALGO_V2,
        events.last().expect("last").event_seq,
        [0; 32],
        &fold_all(ALGO_V2, &events),
    );
    verify(&[honest], hashed(&events).into_iter()).expect("honest fold verifies");
}
