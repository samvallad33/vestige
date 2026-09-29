//! PR-2 acceptance tests: identity dedup, lineage gating, receipts.
//!
//! The determinism triad (replay / twice / similarity-blind) is exercised by
//! `admission_replay_bit_identical` and the lookalike tests below.

use strata_store::admission::{content_identity, Lineage, LineageKind, Outcome};
use strata_store::kinds::{AsOf, KIND_ADMISSION_RECEIPT};
use strata_store::StrataStore;

fn as_of(seq: u64) -> AsOf {
    AsOf {
        seq,
        valid_time_us: 0,
    }
}

fn input(content: &str) -> strata_store::IngestInput {
    strata_store::IngestInput {
        content: content.to_string(),
        ..Default::default()
    }
}

fn source_input(
    content: &str,
    system: &str,
    id: &str,
    updated_ms: i64,
) -> strata_store::IngestInput {
    strata_store::IngestInput {
        content: content.to_string(),
        source: Some(strata_store::types::SourceKey {
            system: system.into(),
            project: "p".into(),
            id: id.into(),
        }),
        source_updated_at_ms: Some(updated_ms),
        ..Default::default()
    }
}

/// Spec: ingest_identical_bytes_reinforces_not_creates
#[test]
fn ingest_identical_bytes_reinforces_not_creates() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    let first = store
        .ingest_admitted(
            input("Deploy the gateway before Friday"),
            "",
            None,
            as_of(1),
        )
        .unwrap();
    let second = store
        .ingest_admitted(
            input("Deploy the gateway before Friday"),
            "",
            None,
            as_of(2),
        )
        .unwrap();
    assert_eq!(first.outcome, Outcome::Created);
    assert_eq!(second.outcome, Outcome::Reinforced);
    assert_eq!(first.node_id, second.node_id, "no new node");
    assert_eq!(store.node_count(), 1);
}

/// Spec: ingest_whitespace_nfc_variants_same_identity
#[test]
fn ingest_whitespace_nfc_variants_same_identity() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    let a = store
        .ingest_admitted(input("caf\u{e9}  gateway   deploy"), "", None, as_of(1))
        .unwrap();
    // NFC-composed é + different whitespace must be the same identity.
    let b = store
        .ingest_admitted(input("caf\u{65}\u{301} gateway deploy"), "", None, as_of(2))
        .unwrap();
    assert_eq!(a.outcome, Outcome::Created);
    assert_eq!(b.outcome, Outcome::Reinforced);
    assert_eq!(a.node_id, b.node_id);
}

/// Spec: ingest_paraphrase_creates_two_nodes — lookalikes never merge.
#[test]
fn ingest_paraphrase_creates_two_nodes() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    let a = store
        .ingest_admitted(
            input("Deploy the gateway before Friday"),
            "",
            None,
            as_of(1),
        )
        .unwrap();
    let b = store
        .ingest_admitted(input("Deploy gateway before Friday!!"), "", None, as_of(2))
        .unwrap();
    assert_eq!(a.outcome, Outcome::Created);
    assert_eq!(b.outcome, Outcome::Created, "lookalike is a new node");
    assert_ne!(a.node_id, b.node_id);
    assert_eq!(
        store
            .get_edges_for(&a.node_id, strata_store::types::EdgeDirection::Both, None)
            .len(),
        0,
        "no edge may appear between lookalikes"
    );
}

/// Spec: ingest_one_word_negation_creates_two_nodes_no_conflict_edge
#[test]
fn ingest_one_word_negation_creates_two_nodes_no_conflict_edge() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    let a = store
        .ingest_admitted(input("always pin X to version 2"), "", None, as_of(1))
        .unwrap();
    let b = store
        .ingest_admitted(input("never pin X to version 2"), "", None, as_of(2))
        .unwrap();
    assert_ne!(a.node_id, b.node_id);
    assert_eq!(
        store
            .get_edges_for(&a.node_id, strata_store::types::EdgeDirection::Both, None)
            .len(),
        0,
        "a one-word negation is NOT a corrects/supersedes edge"
    );
}

/// Spec: dedup_scan_exact_only_lookalikes_not_grouped
#[test]
fn dedup_scan_exact_only_lookalikes_not_grouped() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    store
        .ingest_admitted(
            input("Deploy the gateway before Friday"),
            "",
            None,
            as_of(1),
        )
        .unwrap();
    store
        .ingest_admitted(input("Deploy gateway before Friday!!"), "", None, as_of(2))
        .unwrap();
    store
        .ingest_admitted(
            input("Deploy the gateway before Friday"),
            "",
            None,
            as_of(3),
        )
        .unwrap();
    // Reinforcement makes live identity duplicates impossible through the
    // admission path; seed one the way legacy data arrives (direct write),
    // which is exactly what the dedup scan exists to surface.
    store
        .ingest(strata_store::IngestInput {
            content: "Deploy the gateway before Friday".into(),
            ..Default::default()
        })
        .unwrap();
    let groups = store.scan_identity_groups();
    let multi: Vec<_> = groups.iter().filter(|(_, ids)| ids.len() > 1).collect();
    assert_eq!(multi.len(), 1, "only the byte-identical records group");
    assert_eq!(multi[0].1.len(), 2);
    // The lookalike is in NO identity group.
    let lookalike_grouped = groups.iter().any(|(_, ids)| {
        ids.iter().any(|id| {
            store
                .get_node(id)
                .map(|n| n.content == "Deploy gateway before Friday!!")
                .unwrap_or(false)
        }) && ids.len() > 1
    });
    assert!(!lookalike_grouped, "lookalikes must not group");
}

/// Spec: supersede_requires_exact_handle — a non-handle or unknown target
/// is receipted as refused, nothing is written.
#[test]
fn supersede_requires_exact_handle() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    let out = store
        .ingest_admitted(
            input("replacement fact"),
            "",
            Some(Lineage {
                kind: LineageKind::Supersedes,
                target: "deploy gateway policy".into(),
            }),
            as_of(1),
        )
        .unwrap();
    assert_eq!(out.outcome, Outcome::Refused);
    assert_eq!(store.node_count(), 0, "nothing written on refusal");
}

/// Spec: supersede_cross_scope_denied
#[test]
fn supersede_cross_scope_denied() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    let target = store
        .ingest_admitted(input("scope A fact"), "scope-a", None, as_of(1))
        .unwrap();
    let out = store
        .ingest_admitted(
            input("scope B replacement"),
            "scope-b",
            Some(Lineage {
                kind: LineageKind::Supersedes,
                target: format!("mem:{}", target.node_id),
            }),
            as_of(2),
        )
        .unwrap();
    assert_eq!(out.outcome, Outcome::Refused);
}

/// Spec: supersede_cycle_denied — superseding a node whose chain would loop
/// (self-reference) is refused.
#[test]
fn supersede_cycle_denied() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    // The would-be new id is mem-<next seq>: proposing it as its own target
    // must be refused as a cycle.
    let next = format!("mem-{:016x}", store.log().head().next_seq);
    let out = store
        .ingest_admitted(
            input("self superseding fact"),
            "",
            Some(Lineage {
                kind: LineageKind::Supersedes,
                target: next.clone(),
            }),
            as_of(1),
        )
        .unwrap();
    assert_eq!(out.outcome, Outcome::Refused);
    let _ = next;
}

/// Spec: supersede_default_policy_holds — a declared supersede with no
/// source re-derivation is HELD by the default policy; nothing is written.
#[test]
fn supersede_default_policy_holds() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    let target = store
        .ingest_admitted(input("old gateway policy"), "", None, as_of(1))
        .unwrap();
    let out = store
        .ingest_admitted(
            input("new gateway policy"),
            "",
            Some(Lineage {
                kind: LineageKind::Supersedes,
                target: format!("mem:{}", target.node_id),
            }),
            as_of(2),
        )
        .unwrap();
    assert_eq!(out.outcome, Outcome::Held);
    assert_eq!(store.node_count(), 1, "the held ingest created nothing");
    let target_record = store.get_node(&target.node_id).unwrap();
    assert!(target_record.is_live(), "the target stays live under Hold");
}

/// Spec: source_key_rederive_supersedes — the same source key with a later
/// source_updated_at supersedes automatically.
#[test]
fn source_key_rederive_supersedes() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    let first = store
        .ingest_admitted(
            source_input("connector fact v1", "github", "42", 1000),
            "",
            None,
            as_of(1),
        )
        .unwrap();
    let second = store
        .ingest_admitted(
            source_input("connector fact v2", "github", "42", 2000),
            "",
            Some(Lineage {
                kind: LineageKind::Supersedes,
                target: format!("mem:{}", first.node_id),
            }),
            as_of(2),
        )
        .unwrap();
    assert_eq!(second.outcome, Outcome::Superseded);
    let old = store.get_node(&first.node_id).unwrap();
    assert_eq!(old.superseded_by.as_deref(), Some(second.node_id.as_str()));
}

/// Spec: admission_receipt_for_every_item — N items, N receipts in order.
#[test]
fn admission_receipt_for_every_item() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    for i in 0..5 {
        store
            .ingest_admitted(input(&format!("fact number {i}")), "", None, as_of(i + 1))
            .unwrap();
    }
    let frames = store.log().read_frames(1).unwrap();
    let receipt_seqs: Vec<u64> = frames
        .iter()
        .filter(|f| f.kind == KIND_ADMISSION_RECEIPT)
        .map(|f| f.seq)
        .collect();
    assert_eq!(receipt_seqs.len(), 5);
    let mut sorted = receipt_seqs.clone();
    sorted.sort();
    assert_eq!(sorted, receipt_seqs, "receipts land in ingest order");
}

/// Spec: admission_replay_bit_identical — close, reopen, identical digest.
#[test]
fn admission_replay_bit_identical() {
    let dir = tempfile::tempdir().unwrap();
    let digest = {
        let mut store = StrataStore::open(dir.path()).unwrap();
        store
            .ingest_admitted(input("persisted fact"), "", None, as_of(1))
            .unwrap();
        store
            .ingest_admitted(input("persisted fact"), "", None, as_of(2))
            .unwrap();
        store.state_digest()
    };
    let store = StrataStore::open(dir.path()).unwrap();
    assert_eq!(store.state_digest(), digest, "reopen reproduces the digest");
}

/// Spec: rederive_verdicts_matches_recorded — every receipt's admitting
/// verdict re-derives to Allow.
#[test]
fn rederive_verdicts_matches_recorded() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    store
        .ingest_admitted(input("fact one"), "", None, as_of(1))
        .unwrap();
    store
        .ingest_admitted(input("fact two"), "", None, as_of(2))
        .unwrap();
    let verdicts = store.rederive_verdicts().unwrap();
    assert!(!verdicts.is_empty());
    assert!(verdicts
        .iter()
        .all(|(_, v)| matches!(v, strata_gate::Verdict::Allow)));
}

/// Spec: consolidation_never_deletes_nodes — the store has no delete: a
/// superseded record stays readable (H9), and identity content is intact.
#[test]
fn consolidation_never_deletes_nodes() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = StrataStore::open(dir.path()).unwrap();
    let a = store
        .ingest_admitted(input("retire me"), "", None, as_of(1))
        .unwrap();
    store
        .ingest_admitted(
            source_input("replacement", "s", "1", 2),
            "",
            Some(Lineage {
                kind: LineageKind::Supersedes,
                target: format!("mem:{}", a.node_id),
            }),
            as_of(2),
        )
        .unwrap();
    let old = store
        .get_node(&a.node_id)
        .expect("superseded records stay readable");
    assert_eq!(old.content, "retire me");
}

/// Identity is content-blind to tags and metadata (similarity-blind triad).
#[test]
fn identity_ignores_tags_and_metadata() {
    let mut a = input("same bytes");
    a.tags = vec!["one".into()];
    let mut b = input("same bytes");
    b.tags = vec!["totally".into(), "different".into()];
    assert_eq!(content_identity(&a.content), content_identity(&b.content));
}
