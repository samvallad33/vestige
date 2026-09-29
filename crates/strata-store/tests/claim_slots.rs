//! PR-9 acceptance tests: claim-slot conflicts on exact (subject, field).
//!
//! Contradiction means exactly the same slot with a different canonical
//! value at an overlapping `as_of`; word-level lookalikes ("always use v2"
//! vs "never use v2") never conflict because no claims are involved.
//!
//! `as_of` is caller-supplied (H6): a claim is live at `as_of` only when
//! its CLAIM frame is already in the log (`frame_seq <= as_of.seq`) AND its
//! validity has started (`valid_from.seq <= as_of.seq`). Recordings below
//! use `now(&store)` (the log head) as the decision point; validity
//! boundaries are exercised with explicit seqs above the fixture's frame
//! seqs.

use borsh::BorshDeserialize;
use strata_store::claim_slots::{
    cnhk, encode_numeric, encode_version, field_table, parse_version, values_conflict, ClaimInput,
    ClaimOutcomeKind, ClaimRecord, ConflictReceipt, CLAIM_FIELDS_PARAMS_ID, FIELD_KIND_EXCLUSIVE,
    FIELD_KIND_NUMERIC, FIELD_KIND_SET, FIELD_KIND_VERSION,
};
use strata_store::kinds::{AsOf, KIND_CLAIM, KIND_CONFLICT_RECEIPT};
use strata_store::StrataStore;

fn as_of(seq: u64) -> AsOf {
    AsOf {
        seq,
        valid_time_us: 0,
    }
}

/// The caller-supplied decision point "right now": the log head.
fn now(store: &StrataStore) -> AsOf {
    as_of(store.log().head().next_seq)
}

fn input(content: &str) -> strata_store::IngestInput {
    strata_store::IngestInput {
        content: content.to_string(),
        ..Default::default()
    }
}

/// Install the test's field table. One PARAMS set per params_id: a later
/// append for `claim-fields/1` replaces the whole set, so every field a
/// test uses goes into this one combined set.
fn set_fields(store: &mut StrataStore, knobs: Vec<(String, i64)>) {
    store
        .append_params(CLAIM_FIELDS_PARAMS_ID, knobs)
        .expect("field table params");
}

fn field_kind(id: i64, kind: i64) -> (String, i64) {
    (format!("field.{id}.kind"), kind)
}

fn numeric_field(id: i64, tolerance: i64) -> Vec<(String, i64)> {
    vec![
        field_kind(id, FIELD_KIND_NUMERIC),
        (format!("field.{id}.tolerance"), tolerance),
    ]
}

fn claim(subject: &str, field_id: i64, value: &[u8], valid_from_seq: u64) -> ClaimInput {
    ClaimInput {
        subject_handle: subject.to_string(),
        field_id,
        value_bytes: value.to_vec(),
        valid_from: as_of(valid_from_seq),
        source_receipt: [7u8; 32],
    }
}

fn frames_of_kind(store: &StrataStore, kind: u8) -> Vec<(u64, Vec<u8>)> {
    store
        .log()
        .read_frames(1)
        .unwrap()
        .into_iter()
        .filter(|f| f.kind == kind)
        .map(|f| (f.seq, f.payload))
        .collect()
}

/// Spec: slot_conflict_same_subject_field_differing_value — two live claims
/// on the same exact (subject, field) with differing values conflict; the
/// later claim is Held and a CONFLICT_RECEIPT lands.
#[test]
fn slot_conflict_same_subject_field_differing_value() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    set_fields(&mut store, vec![field_kind(1, FIELD_KIND_EXCLUSIVE)]);

    let a = store
        .record_claim(claim("svc", 1, b"postgres", 1), now(&store))
        .unwrap();
    assert_eq!(a.outcome, ClaimOutcomeKind::Recorded);
    assert!(a.conflicts.is_empty());

    let b = store
        .record_claim(claim("svc", 1, b"mysql", 5), now(&store))
        .unwrap();
    assert_eq!(b.outcome, ClaimOutcomeKind::Held);
    assert_eq!(b.conflicts.len(), 1);
    let (receipt_seq, a_seq, b_seq) = b.conflicts[0];
    assert_eq!(a_seq, a.claim_seq);
    assert_eq!(b_seq, b.claim_seq);

    // The receipt echoes the contested slot and cites both claim seqs.
    let receipts = frames_of_kind(&store, KIND_CONFLICT_RECEIPT);
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].0, receipt_seq);
    let decoded = ConflictReceipt::try_from_slice(&receipts[0].1).unwrap();
    assert_eq!(decoded.subject, "svc");
    assert_eq!(decoded.field_id, 1);
    assert_eq!(decoded.a_seq, a.claim_seq);
    assert_eq!(decoded.b_seq, b.claim_seq);

    // The derived view agrees at the current decision point.
    let pairs = store.conflicts_at(now(&store)).unwrap();
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].a_seq, a.claim_seq);
    assert_eq!(pairs[0].b_seq, b.claim_seq);
}

/// Spec: slot_no_conflict_different_field — the same subject on two
/// different fields never conflicts, even with opposite values.
#[test]
fn slot_no_conflict_different_field() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    set_fields(
        &mut store,
        vec![
            field_kind(1, FIELD_KIND_EXCLUSIVE),
            field_kind(2, FIELD_KIND_EXCLUSIVE),
        ],
    );

    let a = store
        .record_claim(claim("svc", 1, b"postgres", 1), now(&store))
        .unwrap();
    let b = store
        .record_claim(claim("svc", 2, b"mysql", 2), now(&store))
        .unwrap();
    assert_eq!(a.outcome, ClaimOutcomeKind::Recorded);
    assert_eq!(b.outcome, ClaimOutcomeKind::Recorded);
    assert!(store.conflicts_at(now(&store)).unwrap().is_empty());
    assert!(frames_of_kind(&store, KIND_CONFLICT_RECEIPT).is_empty());
    // Both claims are live, ordered by (subject, field, seq).
    let live = store.live_claims_at(now(&store)).unwrap();
    assert_eq!(live.len(), 2);
    assert_eq!((live[0].claim.field_id, live[1].claim.field_id), (1, 2));
}

/// Spec: slot_no_conflict_non_overlapping_validity — a claim whose validity
/// has not started at the queried `as_of` never conflicts at that `as_of`;
/// once both are valid the slot does conflict. This fixture's CLAIM frames
/// all land below seq 49 (two PARAMS appends + two gate-admitted claims),
/// so the 49/50 boundary isolates the validity dimension.
#[test]
fn slot_no_conflict_non_overlapping_validity() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    set_fields(&mut store, vec![field_kind(1, FIELD_KIND_EXCLUSIVE)]);

    let a = store
        .record_claim(claim("svc", 1, b"postgres", 2), now(&store))
        .unwrap();
    let b = store
        .record_claim(claim("svc", 1, b"mysql", 50), now(&store))
        .unwrap();
    assert_eq!(a.outcome, ClaimOutcomeKind::Recorded);
    assert_eq!(b.outcome, ClaimOutcomeKind::Held);

    // Before seq 50 the later claim's validity has not started: its slot
    // cannot conflict yet.
    assert!(store.conflicts_at(as_of(49)).unwrap().is_empty());
    let live = store.live_claims_at(as_of(49)).unwrap();
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].seq, a.claim_seq);

    // From seq 50 both claims are valid: exactly one conflict pair.
    let live = store.live_claims_at(as_of(50)).unwrap();
    assert_eq!(live.len(), 2);
    let pairs = store.conflicts_at(as_of(50)).unwrap();
    assert_eq!(pairs.len(), 1);
    assert_eq!((pairs[0].a_seq, pairs[0].b_seq), (a.claim_seq, b.claim_seq));
}

/// Spec: slot_integer_tolerance — numeric values within the integer
/// tolerance (in the field's unit) are the same value; beyond it they
/// conflict.
#[test]
fn slot_integer_tolerance() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    set_fields(&mut store, numeric_field(3, 5));

    let a = store
        .record_claim(claim("gw", 3, &encode_numeric(100), 1), now(&store))
        .unwrap();
    // |100 - 104| = 4 <= 5: same value, no conflict.
    let b = store
        .record_claim(claim("gw", 3, &encode_numeric(104), 2), now(&store))
        .unwrap();
    assert_eq!(a.outcome, ClaimOutcomeKind::Recorded);
    assert_eq!(b.outcome, ClaimOutcomeKind::Recorded);
    assert!(store.conflicts_at(now(&store)).unwrap().is_empty());
    assert!(frames_of_kind(&store, KIND_CONFLICT_RECEIPT).is_empty());

    // |100 - 106| = 6 > 5: conflict with a, but not with b (|104 - 106| = 2).
    let c = store
        .record_claim(claim("gw", 3, &encode_numeric(106), 3), now(&store))
        .unwrap();
    assert_eq!(c.outcome, ClaimOutcomeKind::Held);
    assert_eq!(c.conflicts.len(), 1);
    assert_eq!(c.conflicts[0].1, a.claim_seq);

    let pairs = store.conflicts_at(now(&store)).unwrap();
    assert_eq!(pairs.len(), 1);
    assert_eq!((pairs[0].a_seq, pairs[0].b_seq), (a.claim_seq, c.claim_seq));

    // Non-canonical numeric bytes are refused (fail closed).
    let err = store
        .record_claim(claim("gw", 3, b"100", 4), now(&store))
        .unwrap_err();
    assert!(err.to_string().contains("canonical borsh i64"));
}

/// Spec: slot_values_exact_bytes_no_casefold — value bytes are exact bytes;
/// "Prod" != "prod" (the field type decides, the store never folds).
#[test]
fn slot_values_exact_bytes_no_casefold() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    set_fields(&mut store, vec![field_kind(4, FIELD_KIND_EXCLUSIVE)]);

    let a = store
        .record_claim(claim("env", 4, b"Prod", 1), now(&store))
        .unwrap();
    // Case differs: different bytes, so a conflict (no case folding).
    let b = store
        .record_claim(claim("env", 4, b"prod", 2), now(&store))
        .unwrap();
    assert_eq!(b.outcome, ClaimOutcomeKind::Held);
    assert_eq!(b.conflicts.len(), 1);
    assert_eq!(b.conflicts[0].1, a.claim_seq);

    // Re-asserting "Prod" still conflicts with the live held "prod"
    // (pairwise, claims are never deleted): the slot stays contested until
    // an explicit corrects/supersedes resolves it.
    let c = store
        .record_claim(claim("env", 4, b"Prod", 3), now(&store))
        .unwrap();
    assert_eq!(c.outcome, ClaimOutcomeKind::Held);
    assert_eq!(c.conflicts.len(), 1);
    assert_eq!(c.conflicts[0].1, b.claim_seq);

    let pairs = store.conflicts_at(now(&store)).unwrap();
    assert_eq!(pairs.len(), 2);
    assert_eq!((pairs[0].a_seq, pairs[0].b_seq), (a.claim_seq, b.claim_seq));
    assert_eq!((pairs[1].a_seq, pairs[1].b_seq), (b.claim_seq, c.claim_seq));

    // Unit check of the comparison itself.
    let sem = field_table(&store)[&4];
    assert!(values_conflict(&sem, b"Prod", b"prod"));
    assert!(!values_conflict(&sem, b"Prod", b"Prod"));
}

/// Spec: negation_text_is_not_conflict — two texts that merely disagree in
/// wording ("always use v2" / "never use v2") with NO claims recorded
/// produce no conflict and no receipt. Word-level contradiction logic is
/// gone; only exact slots conflict.
#[test]
fn negation_text_is_not_conflict() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    store
        .ingest_admitted(input("always use v2"), "", None, as_of(1))
        .unwrap();
    store
        .ingest_admitted(input("never use v2"), "", None, as_of(2))
        .unwrap();

    // No claims exist, so no slot can conflict at any as_of.
    assert!(store.conflicts_at(as_of(u64::MAX - 1)).unwrap().is_empty());
    assert!(store
        .live_claims_at(as_of(u64::MAX - 1))
        .unwrap()
        .is_empty());

    let frames = store.log().read_frames(1).unwrap();
    assert!(
        !frames.iter().any(|f| f.kind == KIND_CONFLICT_RECEIPT),
        "no conflict receipt may exist for text lookalikes"
    );
    assert!(!frames.iter().any(|f| f.kind == KIND_CLAIM));

    // The two lookalike texts stayed two unmerged nodes.
    assert_eq!(store.node_count(), 2);
}

/// Spec: conflict_receipt_replays_bit_identical — close and reopen the
/// store: the CLAIM and CONFLICT_RECEIPT frame bytes are identical and the
/// derived conflict view replays identically.
#[test]
fn conflict_receipt_replays_bit_identical() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("claims");
    let (before_frames, before_pairs, before_digest) = {
        let mut store = StrataStore::open(&path).unwrap();
        set_fields(&mut store, vec![field_kind(1, FIELD_KIND_EXCLUSIVE)]);
        store
            .record_claim(claim("svc", 1, b"postgres", 1), now(&store))
            .unwrap();
        store
            .record_claim(claim("svc", 1, b"mysql", 5), now(&store))
            .unwrap();
        let frames: Vec<(u64, u8, Vec<u8>)> = store
            .log()
            .read_frames(1)
            .unwrap()
            .into_iter()
            .filter(|f| f.kind == KIND_CLAIM || f.kind == KIND_CONFLICT_RECEIPT)
            .map(|f| (f.seq, f.kind, f.payload))
            .collect();
        let pairs = store.conflicts_at(as_of(u64::MAX - 1)).unwrap();
        (frames, pairs, store.state_digest())
    };
    assert_eq!(before_frames.len(), 3, "two claims + one conflict receipt");

    // Reopen: pure replay, no re-append.
    let store = StrataStore::open(&path).unwrap();
    let after_frames: Vec<(u64, u8, Vec<u8>)> = store
        .log()
        .read_frames(1)
        .unwrap()
        .into_iter()
        .filter(|f| f.kind == KIND_CLAIM || f.kind == KIND_CONFLICT_RECEIPT)
        .map(|f| (f.seq, f.kind, f.payload))
        .collect();
    assert_eq!(
        before_frames, after_frames,
        "receipt bytes replay bit-identical"
    );
    assert_eq!(
        before_pairs,
        store.conflicts_at(as_of(u64::MAX - 1)).unwrap()
    );
    assert_eq!(before_digest, store.state_digest());

    // The conflict receipt survives reopen with its cited seqs intact.
    let receipts = frames_of_kind(&store, KIND_CONFLICT_RECEIPT);
    let decoded = ConflictReceipt::try_from_slice(&receipts[0].1).unwrap();
    assert_eq!(
        (decoded.a_seq, decoded.b_seq),
        (before_frames[0].0, before_frames[1].0)
    );
}

/// Spec: cnhk_proof_verifies — a claim whose validity started strictly
/// after the decision's as_of yields a verifiable could-not-have-known
/// proof.
#[test]
fn cnhk_proof_verifies() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    set_fields(&mut store, vec![field_kind(1, FIELD_KIND_EXCLUSIVE)]);

    let outcome = store
        .record_claim(
            claim(
                "svc",
                1,
                &encode_version(&parse_version("v2.1").unwrap()),
                100,
            ),
            now(&store),
        )
        .unwrap();
    assert_eq!(outcome.outcome, ClaimOutcomeKind::Recorded);

    let (claim_seq, record) = frames_of_kind(&store, KIND_CLAIM)
        .into_iter()
        .map(|(seq, payload)| (seq, ClaimRecord::try_from_slice(&payload).unwrap()))
        .next()
        .expect("the recorded claim frame");
    assert_eq!(claim_seq, outcome.claim_seq);
    assert_eq!(record.valid_from.seq, 100);
    assert_eq!(record.value_bytes, encode_version(&[2, 1]));

    let decision_ref = [9u8; 32];
    let proof = cnhk(&record, claim_seq, 50, decision_ref).expect("claim postdates the decision");
    assert_eq!(proof.claim_seq, claim_seq);
    assert_eq!(proof.decision_as_of_seq, 50);
    assert_eq!(proof.decision_ref, decision_ref);
    assert_eq!(proof.claim_ref, {
        let bytes = borsh::to_vec(&record).unwrap();
        *blake3::hash(&bytes).as_bytes()
    });
    assert!(proof.checksum_valid());
    assert!(record.valid_from.seq > proof.decision_as_of_seq);

    // Any tampering breaks the checksum.
    let mut bad = proof.clone();
    bad.claim_seq += 1;
    assert!(!bad.checksum_valid());
}

/// Spec: cnhk_rejects_when_claim_predates_decision — a claim whose validity
/// started at or before the decision's as_of is knowable and must be
/// rejected (fail closed, including the boundary).
#[test]
fn cnhk_rejects_when_claim_predates_decision() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    set_fields(&mut store, vec![field_kind(1, FIELD_KIND_EXCLUSIVE)]);

    let outcome = store
        .record_claim(claim("svc", 1, b"postgres", 3), now(&store))
        .unwrap();
    let (_, record) = frames_of_kind(&store, KIND_CLAIM)
        .into_iter()
        .map(|(seq, payload)| (seq, ClaimRecord::try_from_slice(&payload).unwrap()))
        .next()
        .expect("the recorded claim frame");
    let claim_seq = outcome.claim_seq;

    let err = cnhk(&record, claim_seq, 7, [1u8; 32]).unwrap_err();
    assert!(err.contains("predates"), "got: {err}");
    // Boundary: valid_from == decision as_of means the claim was knowable.
    let err = cnhk(&record, claim_seq, 3, [1u8; 32]).unwrap_err();
    assert!(err.contains("predates"), "got: {err}");
}

/// The full field-table round trip: every ported semantics survives the
/// PARAMS encoding (Set and Version included) and drives comparisons.
#[test]
fn field_table_round_trip_all_semantics() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    store
        .append_params(
            CLAIM_FIELDS_PARAMS_ID,
            vec![
                ("field.1.kind".into(), FIELD_KIND_EXCLUSIVE),
                ("field.2.kind".into(), FIELD_KIND_SET),
                ("field.3.kind".into(), FIELD_KIND_NUMERIC),
                ("field.3.tolerance".into(), 5),
                ("field.4.kind".into(), FIELD_KIND_VERSION),
            ],
        )
        .unwrap();
    let table = field_table(&store);
    assert_eq!(
        table[&1],
        strata_store::claim_slots::FieldSemantics::Exclusive
    );
    assert_eq!(table[&2], strata_store::claim_slots::FieldSemantics::Set);
    assert_eq!(
        table[&3],
        strata_store::claim_slots::FieldSemantics::Numeric { tolerance: 5 }
    );
    assert_eq!(
        table[&4],
        strata_store::claim_slots::FieldSemantics::Version
    );

    // Set fields are additive: differing members never conflict.
    let a = store
        .record_claim(claim("svc", 2, b"primary", 1), now(&store))
        .unwrap();
    let b = store
        .record_claim(claim("svc", 2, b"replica", 2), now(&store))
        .unwrap();
    assert_eq!(a.outcome, ClaimOutcomeKind::Recorded);
    assert_eq!(b.outcome, ClaimOutcomeKind::Recorded);

    // Version fields compare canonical components: 2.10 != 2.9 numerically
    // (10 > 9) even though the prose sorts the other way.
    let v1 = store
        .record_claim(claim("svc", 4, &encode_version(&[2, 9]), 1), now(&store))
        .unwrap();
    let v2 = store
        .record_claim(claim("svc", 4, &encode_version(&[2, 10]), 2), now(&store))
        .unwrap();
    assert_eq!(v2.outcome, ClaimOutcomeKind::Held);
    assert_eq!(v2.conflicts[0].1, v1.claim_seq);
}
