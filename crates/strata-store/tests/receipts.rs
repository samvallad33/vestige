//! PR-0c named tests: gate-admitted receipts and PARAMS hashing.

use strata_store::kinds::{AsOf, ReceiptHeader};
use strata_store::StrataStore;

fn header(params_hash: [u8; 32], seq: u64) -> ReceiptHeader {
    ReceiptHeader {
        params_hash,
        as_of: AsOf {
            seq,
            valid_time_us: 0,
        },
    }
}

/// Spec: receipt_append_is_gate_admitted — every receipt frame has an
/// admitting EFFECT, proven by reopen + rederive_verdicts covering its
/// propose, and the frame lands only on Allow.
#[test]
fn receipt_append_is_gate_admitted() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("receipts");
    {
        let mut store = StrataStore::open(&path).unwrap();
        let params_hash = store
            .append_params("guard/1", vec![("k".into(), 1)])
            .unwrap();
        let payload =
            borsh::to_vec(&(header(params_hash, 1), "test receipt body".to_string())).unwrap();
        let (effect_seq, frame_seq) = store
            .append_receipt(strata_store::kinds::KIND_SESSION_MARK, &payload, Vec::new())
            .expect("receipt append is admitted");
        assert!(effect_seq > 0 && frame_seq > effect_seq);
    }

    // Reopen: the receipt frame replays and the gate verdict re-derives.
    let store = StrataStore::open(&path).unwrap();
    let verdicts = store.rederive_verdicts().unwrap();
    assert!(
        verdicts
            .iter()
            .any(|(_, v)| matches!(v, strata_gate::Verdict::Allow)),
        "the receipt's propose must re-derive to Allow"
    );
    let frames = store.log().read_frames(1).unwrap();
    assert!(
        frames
            .iter()
            .any(|f| f.kind == strata_store::kinds::KIND_SESSION_MARK),
        "the receipt frame is durable"
    );
}

/// Spec: params_hash_changes_when_params_change — a changed knob appends a
/// new PARAMS set whose hash differs; the hash is stable when nothing
/// changes.
#[test]
fn params_hash_changes_when_params_change() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    let h1 = store
        .append_params("walk/1", vec![("k".into(), 3)])
        .unwrap();
    let h1_again = store
        .append_params("other/1", vec![("x".into(), 9)])
        .unwrap();
    // A different params_id does not move the walk/1 set's own hash, and the
    // active-set hash is deterministic across identical appends.
    let active1 = store.current_params_hash();
    let h1_replay = {
        let dir2 = tempfile::tempdir().unwrap();
        let mut store2 = StrataStore::open(dir2.path()).unwrap();
        let a = store2
            .append_params("walk/1", vec![("k".into(), 3)])
            .unwrap();
        let _ = store2
            .append_params("other/1", vec![("x".into(), 9)])
            .unwrap();
        assert_eq!(a, h1);
        assert_eq!(store2.current_params_hash(), active1);
        a
    };
    let _ = h1_again;
    let _ = h1_replay;

    // Changing a knob changes the active hash.
    store
        .append_params("walk/1", vec![("k".into(), 4)])
        .unwrap();
    let active2 = store.current_params_hash();
    assert_ne!(
        active1, active2,
        "a changed knob changes the active params hash"
    );
}

/// Receipts bind the active params at append time (header round-trip).
#[test]
fn receipt_header_binds_params_and_as_of() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    let params_hash = store
        .append_params("bind/1", vec![("floor".into(), 70)])
        .unwrap();
    let payload = borsh::to_vec(&(header(params_hash, 7), 42u64)).unwrap();
    store
        .append_receipt(strata_store::kinds::KIND_JOB, &payload, Vec::new())
        .unwrap();
    let frames = store.log().read_frames(1).unwrap();
    let job = frames
        .iter()
        .find(|f| f.kind == strata_store::kinds::KIND_JOB)
        .expect("job receipt frame");
    use borsh::BorshDeserialize;
    let (decoded, _): (ReceiptHeader, u64) =
        borsh::BorshDeserialize::try_from_slice(&job.payload).unwrap();
    assert_eq!(decoded.params_hash, params_hash);
    assert_eq!(decoded.as_of.seq, 7);
}
