//! PR 4 named acceptance tests: receipt-bound claims, fail-closed.

use borsh::BorshDeserialize;
use strata_gate::{DutyKind, GapDetail};
use strata_store::claims::{
    rule, verify_claim, ClaimRecord, ClaimVerdictRecord, ToolResultRecord, ADVISORY_VETO_KIND,
    OUTCOME_PASS,
};
use strata_store::kinds::{AsOf, ReceiptHeader, KIND_CLAIM, KIND_CLAIM_VERDICT};
use strata_store::StrataStore;

const SESSION: &str = "session-a";

fn asof(seq: u64) -> AsOf {
    AsOf {
        seq,
        valid_time_us: 0,
    }
}

fn family() -> Vec<(String, String)> {
    vec![
        ("tests_pass".to_string(), "cargo".to_string()),
        ("lint_ok".to_string(), "cargo".to_string()),
    ]
}

/// Post one passing/failing TOOL_RESULT and return its frame seq (the id a
/// claim cites).
fn post_result(
    store: &mut StrataStore,
    session: &str,
    tool: &str,
    exit_status: i32,
    outcomes: &[(&str, &str)],
) -> u64 {
    let record = ToolResultRecord {
        header: ReceiptHeader {
            params_hash: [0; 32],
            as_of: asof(store.log().head().next_seq),
        },
        session: session.to_string(),
        call_id: format!("call-{}", store.log().head().next_seq),
        tool: tool.to_string(),
        target_handle: "repo://x".to_string(),
        exit_status,
        test_outcomes: outcomes
            .iter()
            .map(|(test, outcome)| (test.to_string(), outcome.to_string()))
            .collect(),
    };
    store
        .record_tool_result(&record)
        .expect("tool result lands")
}

/// Build a CLAIM addressed at the current head.
fn claim(store: &StrataStore, kind: &str, cited: &[u64]) -> ClaimRecord {
    ClaimRecord {
        header: ReceiptHeader {
            params_hash: [0; 32],
            as_of: asof(store.log().head().next_seq),
        },
        session: SESSION.to_string(),
        claim_kind: kind.to_string(),
        target_handle: "repo://x".to_string(),
        cited_record_ids: cited.to_vec(),
    }
}

fn verdict_frames(store: &StrataStore) -> Vec<(u64, ClaimVerdictRecord)> {
    store.read_claim_verdicts().expect("verdicts read")
}

/// Spec: claim_with_matching_pass_result_allows.
#[test]
fn claim_with_matching_pass_result_allows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();
    let outcome = {
        let mut store = StrataStore::open(path).unwrap();
        let receipt_seq = post_result(&mut store, SESSION, "cargo", 0, &[("t1", OUTCOME_PASS)]);
        let claim = claim(&store, "tests_pass", &[receipt_seq]);
        store.check_claim(&claim, &family()).unwrap()
    };

    assert!(outcome.verdict.allow, "a matching pass result allows");
    assert_eq!(outcome.verdict.rule_ids, vec![rule::PASS_EXIT_ZERO]);
    assert!(outcome.verdict.gap.is_none(), "no GAP for an allowed claim");

    // The verdict frame is durable and decodes to the same decision.
    let reopened = StrataStore::open(path).unwrap();
    let verdicts = verdict_frames(&reopened);
    assert_eq!(verdicts.len(), 1, "exactly one verdict receipt");
    assert!(verdicts[0].1.allow);
    assert_eq!(verdicts[0].1.claim_frame_seq, outcome.claim_frame_seq);
}

/// Spec: claim_citing_failed_result_denies.
#[test]
fn claim_citing_failed_result_denies() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    let receipt_seq = post_result(
        &mut store,
        SESSION,
        "cargo",
        101,
        &[("t1", "fail"), ("t2", OUTCOME_PASS)],
    );
    let claim = claim(&store, "tests_pass", &[receipt_seq]);
    let outcome = store.check_claim(&claim, &family()).unwrap();

    assert!(!outcome.verdict.allow, "a failed cited result denies");
    assert_eq!(outcome.verdict.rule_ids, vec![rule::FAILED_RESULT]);
    assert!(
        outcome.verdict.gap.is_none(),
        "the citation existed: no GAP"
    );
}

/// Spec: claim_citing_other_session_denies.
#[test]
fn claim_citing_other_session_denies() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    let receipt_seq = post_result(&mut store, "session-b", "cargo", 0, &[("t1", OUTCOME_PASS)]);
    let claim = claim(&store, "tests_pass", &[receipt_seq]);
    let outcome = store.check_claim(&claim, &family()).unwrap();

    assert!(
        !outcome.verdict.allow,
        "another session's receipt never backs a claim"
    );
    assert_eq!(outcome.verdict.rule_ids, vec![rule::CROSS_SESSION]);
}

/// Spec: claim_citing_future_record_denies.
#[test]
fn claim_citing_future_record_denies() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    let receipt_seq = post_result(&mut store, SESSION, "cargo", 0, &[("t1", OUTCOME_PASS)]);
    assert!(receipt_seq > 1, "precondition: the result has a real seq");

    // The claim asserts a decision point BELOW the cited result's seq.
    let mut claim = claim(&store, "tests_pass", &[receipt_seq]);
    claim.header.as_of = asof(receipt_seq - 1);
    let outcome = store.check_claim(&claim, &family()).unwrap();

    assert!(
        !outcome.verdict.allow,
        "a citation at-or-before rule is enforced"
    );
    assert_eq!(outcome.verdict.rule_ids, vec![rule::CITATION_FUTURE]);
}

/// Spec: claim_without_citation_writes_gap_and_denies.
#[test]
fn claim_without_citation_writes_gap_and_denies() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();
    let outcome = {
        let mut store = StrataStore::open(path).unwrap();
        let claim = claim(&store, "tests_pass", &[]);
        store.check_claim(&claim, &family()).unwrap()
    };

    assert!(!outcome.verdict.allow, "an uncited claim denies");
    assert_eq!(outcome.verdict.rule_ids, vec![rule::NO_CITATION]);
    let gap = outcome.verdict.gap.expect("missing citation writes a GAP");
    assert_eq!(gap.duty, DutyKind::ReadNoReceipt);
    match gap.detail {
        GapDetail::ReadNoReceipt {
            reader_seq,
            dangling_id,
        } => {
            assert_eq!(reader_seq, outcome.claim_frame_seq);
            assert_eq!(dangling_id, 0, "cited nothing");
        }
        other => panic!("unexpected gap detail: {other:?}"),
    }

    // The GAP is part of the durable signed verdict receipt.
    let reopened = StrataStore::open(path).unwrap();
    let verdicts = verdict_frames(&reopened);
    assert_eq!(verdicts.len(), 1);
    assert!(verdicts[0].1.gap.is_some(), "the GAP is replayable");
    assert!(!verdicts[0].1.allow);
}

/// Spec: claim_verdict_rederives_bit_identical — close, reopen, replay the
/// log alone, and every stored verdict re-derives to identical borsh bytes.
#[test]
fn claim_verdict_rederives_bit_identical() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();
    {
        let mut store = StrataStore::open(path).unwrap();
        let pass = post_result(&mut store, SESSION, "cargo", 0, &[("t1", OUTCOME_PASS)]);
        let fail = post_result(&mut store, SESSION, "cargo", 1, &[("t1", "fail")]);
        let allow = claim(&store, "tests_pass", &[pass]);
        store.check_claim(&allow, &family()).unwrap();
        let deny = claim(&store, "tests_pass", &[fail]);
        store.check_claim(&deny, &family()).unwrap();
    }

    let reopened = StrataStore::open(path).unwrap();
    let rederived = reopened.rederive_claim_verdicts(&family()).unwrap();
    assert_eq!(rederived.len(), 2, "both verdicts re-derive");
    assert!(
        rederived.iter().all(|v| v.bit_identical),
        "every re-derived verdict is bit-identical to its stored receipt"
    );
    assert!(
        rederived.iter().any(|v| v.stored.allow) && rederived.iter().any(|v| !v.stored.allow),
        "both an allow and a deny verdict were exercised"
    );

    // The family table is bound into the verdict: a different table no
    // longer re-derives the stored receipts (never silently re-decides).
    let other_table = vec![("tests_pass".to_string(), "npm".to_string())];
    let mismatched = reopened.rederive_claim_verdicts(&other_table).unwrap();
    assert!(
        mismatched.iter().any(|v| !v.bit_identical),
        "a different family table is detectable, not silently accepted"
    );
}

/// Spec: claim_kind_is_field_not_prose — the checker only ever sees the
/// structured `claim_kind` field (prose has no input channel and no CLAIM
/// record means no verdict, no allow), and a lookalike tool name never
/// satisfies a family entry.
#[test]
fn claim_kind_is_field_not_prose() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    let receipt_seq = post_result(&mut store, SESSION, "cargo", 0, &[("t1", OUTCOME_PASS)]);

    // (a) No CLAIM record exists — nothing in the log produced a verdict,
    // no matter what prose a draft contained.
    assert!(
        verdict_frames(&store).is_empty(),
        "no CLAIM record means no verdict and no allow"
    );

    // (b) Prose past in as a claim_kind field matches nothing: the table is
    // exact strings, so "all tests pass ✅" denies (fail-closed), and the
    // pure checker's inputs are typed records — prose cannot reach it.
    let table = family();
    let mut prose_claim = claim(&store, "tests_pass", &[receipt_seq]);
    prose_claim.claim_kind = "all tests pass \u{2705}".to_string();
    let results = store.read_tool_results().unwrap();
    let v = verify_claim(&prose_claim, &results, &table);
    assert!(!v.allow && v.rule_ids == vec![rule::UNKNOWN_CLAIM_KIND]);

    // (c) A TOOL_RESULT whose tool name merely resembles a family entry
    // (`cargo-test-helper` vs `cargo`) does not satisfy tests_pass.
    let lookalike = post_result(
        &mut store,
        SESSION,
        "cargo-test-helper",
        0,
        &[("t1", OUTCOME_PASS)],
    );
    let claim = claim(&store, "tests_pass", &[lookalike]);
    let outcome = store.check_claim(&claim, &family()).unwrap();
    assert!(!outcome.verdict.allow);
    assert_eq!(outcome.verdict.rule_ids, vec![rule::FAMILY_MISMATCH]);
}

/// Spec: advisory_veto_never_changes_verdict — a model-appended advisory
/// veto annotation sits outside the decision path; verdicts before and
/// after it are identical, and it never produces a verdict of its own.
#[test]
fn advisory_veto_never_changes_verdict() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();
    {
        let mut store = StrataStore::open(path).unwrap();
        let receipt_seq = post_result(&mut store, SESSION, "cargo", 0, &[("t1", OUTCOME_PASS)]);

        // Both checks are the SAME claim at the SAME decision point: only
        // the advisory veto lands in between.
        let before = claim(&store, "tests_pass", &[receipt_seq]);
        let decision_point = before.header.as_of;
        let before_outcome = store.check_claim(&before, &family()).unwrap();
        assert!(before_outcome.verdict.allow);

        // The model appends its advisory veto OUTSIDE the decision path.
        let veto_seq = store
            .record_advisory_veto(SESSION, "repo://x", decision_point)
            .unwrap();
        assert!(veto_seq > before_outcome.claim_frame_seq);

        // The same claim, checked after the veto: identical decision.
        let mut after = claim(&store, "tests_pass", &[receipt_seq]);
        after.header.as_of = decision_point;
        let after_outcome = store.check_claim(&after, &family()).unwrap();
        assert_eq!(before_outcome.verdict.allow, after_outcome.verdict.allow);
        assert_eq!(
            before_outcome.verdict.rule_ids, after_outcome.verdict.rule_ids,
            "the advisory veto changed no rule outcome"
        );
        assert_eq!(
            before_outcome.verdict.claim_digest, after_outcome.verdict.claim_digest,
            "identical claims digest identically across the veto"
        );
    }

    // Replay: only the two real checks have verdicts; the annotation has
    // none, and every stored verdict still re-derives bit-identically.
    let reopened = StrataStore::open(path).unwrap();
    let verdicts = verdict_frames(&reopened);
    assert_eq!(verdicts.len(), 2, "the advisory veto produced no verdict");
    let rederived = reopened.rederive_claim_verdicts(&family()).unwrap();
    assert!(rederived.iter().all(|v| v.bit_identical));

    // Structural guarantee: an advisory-veto record is a CLAIM frame whose
    // kind field is the dedicated annotation kind, not a verdict input.
    let advisory: Vec<_> = reopened
        .log()
        .read_frames(1)
        .unwrap()
        .into_iter()
        .filter(|f| f.kind == KIND_CLAIM)
        .filter_map(|f| {
            <ClaimRecord as BorshDeserialize>::try_from_slice(&f.payload)
                .ok()
                .map(|c| (f.seq, c))
        })
        .collect();
    let annotation = advisory
        .iter()
        .find(|(_, c)| c.claim_kind == ADVISORY_VETO_KIND)
        .expect("the advisory veto annotation is a typed CLAIM frame");
    assert!(annotation.1.cited_record_ids.is_empty());
    assert_eq!(
        reopened
            .log()
            .read_frames(1)
            .unwrap()
            .iter()
            .filter(|f| f.kind == KIND_CLAIM_VERDICT)
            .count(),
        2,
        "the annotation is never checked, so it can never allow anything"
    );
    // The pure checker denies the annotation kind outright if it is ever
    // passed in (fail-closed, H8).
    let results = reopened.read_tool_results().unwrap();
    let veto_check = verify_claim(&annotation.1, &results, &family());
    assert!(!veto_check.allow);
    assert_eq!(veto_check.rule_ids, vec![rule::UNKNOWN_CLAIM_KIND]);
}
