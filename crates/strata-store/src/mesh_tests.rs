//! Merge-fold acceptance tests. The falsifier is
//! `offline_writers_fold_to_one_mesh_digest`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use strata_gate::record::{GateEvent, RecordKind};

use crate::op::FrameFail;
use crate::store::mesh::{
    canonical_handle, check_peer_range, fingerprint, mesh_grant_context, writer_did,
};
use crate::store::{
    default_policy, effect_receipt_id, handle_of, permissive_policy, AdmissionContext, EffectAction,
};
use crate::types::{ConnectionRecord, IngestInput};
use crate::{StoreError, StrataStore};

fn temp_dir(name: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "strata-mesh-{}-{n}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn input(content: &str) -> IngestInput {
    IngestInput {
        content: content.to_string(),
        source: None,
        source_updated_at_ms: None,
        node_type: String::new(),
        tags: Vec::new(),
        created_at_ms: Some(1_700_000_000_000),
        valid_from_ms: None,
        valid_until_ms: None,
    }
}

fn exchange(left: &mut StrataStore, right: &mut StrataStore) {
    let left_dir = temp_dir("exchange-left");
    let right_dir = temp_dir("exchange-right");
    left.export_chain(&left_dir).unwrap();
    right.export_chain(&right_dir).unwrap();
    assert!(!left_dir.join("strata.key").exists());
    assert!(!right_dir.join("strata.key").exists());
    left.import_chain(&right_dir, &mesh_grant_context()).unwrap();
    right.import_chain(&left_dir, &mesh_grant_context()).unwrap();
    let _ = std::fs::remove_dir_all(&left_dir);
    let _ = std::fs::remove_dir_all(&right_dir);
}

#[test]
fn offline_writers_fold_to_one_mesh_digest() {
    let dir_a = temp_dir("fold-a");
    let dir_b = temp_dir("fold-b");
    let mut a = StrataStore::open(&dir_a).unwrap();
    let mut b = StrataStore::open(&dir_b).unwrap();

    let a_own = a.ingest(input("alpha keeps its own fact")).unwrap();
    a.review(&a_own, 3).unwrap();
    let b_own = b.ingest(input("beta keeps its own fact")).unwrap();
    b.review(&b_own, 3).unwrap();
    let a_same = a.ingest(input("the shared sentence")).unwrap();
    let b_same = b.ingest(input("the shared sentence")).unwrap();
    a.review(&a_same, 4).unwrap();
    b.review(&b_same, 2).unwrap();
    a.assert_claim("slot", "color", b"red", 0, i64::MAX).unwrap();
    b.assert_claim("slot", "color", b"blue", 0, i64::MAX).unwrap();

    assert_ne!(
        a.canonical_handle(&a_own),
        b.canonical_handle(&b_own),
        "writer-qualified handles differ even when the local mint collides"
    );
    assert_ne!(a.writer_did(), b.writer_did());

    let export_a = temp_dir("export-a");
    let export_b = temp_dir("export-b");
    a.export_chain(&export_a).unwrap();
    b.export_chain(&export_b).unwrap();
    assert!(!export_a.join("strata.key").exists());
    assert!(export_a.join("writer.pub").is_file());

    let into_a = a.import_chain(&export_b, &mesh_grant_context()).unwrap();
    let into_b = b.import_chain(&export_a, &mesh_grant_context()).unwrap();
    assert!(!into_a.receipt_only && !into_b.receipt_only);
    let proof_a = a
        .effect_by_seq(into_a.receipt_seq.unwrap())
        .unwrap()
        .expect("merge receipt");
    let proof_b = b
        .effect_by_seq(into_b.receipt_seq.unwrap())
        .unwrap()
        .expect("merge receipt");
    assert_eq!(proof_a.action, EffectAction::Merge);
    assert_eq!(proof_b.action, EffectAction::Merge);
    assert_eq!(
        effect_receipt_id(proof_a.effect_seq),
        format!("eff-{:016x}", proof_a.effect_seq)
    );

    assert_eq!(a.mesh_digest(), b.mesh_digest());
    assert_eq!(a.mesh_projection().conflicts, b.mesh_projection().conflicts);
    assert_eq!(a.mesh_projection().conflicts.len(), 1);
    assert_eq!(a.mesh_conflict_receipts().len(), 1);
    assert_eq!(b.mesh_conflict_receipts().len(), 1);
    let conflict = &a.mesh_conflict_receipts()[0];
    assert_ne!(conflict.left.writer_did, conflict.right.writer_did);
    assert_ne!(conflict.left.frame_hash, conflict.right.frame_hash);

    let projection = a.mesh_projection();
    assert_eq!(projection.reinforcements.len(), 1);
    let shared = projection
        .nodes
        .keys()
        .filter(|id| projection.nodes.get(*id) == projection.nodes.get(&projection.reinforcements[0].kept))
        .count();
    assert!(shared >= 1);
    let kept = projection.reinforcements[0].kept.clone();
    let reviews = *projection.review_counts.get(&handle_of(&kept)).unwrap_or(&0);
    assert!(reviews >= 4, "both ingests and both reviews land on the kept card, got {reviews}");

    let again = a.import_chain(&export_b, &mesh_grant_context()).unwrap();
    assert!(again.receipt_only);
    assert!(again.receipt_seq.is_none());
    assert_eq!(a.mesh_digest(), into_a.mesh_digest);

    assert_eq!(a.resolve_bare_id(&a_own).unwrap(), a_own);
    assert_eq!(
        a.resolve_bare_id(&effect_receipt_id(proof_a.effect_seq)).unwrap(),
        effect_receipt_id(proof_a.effect_seq)
    );

    let _ = (dir_a, dir_b, export_a, export_b, a_same, b_same);
}

#[test]
fn ungranted_chain_does_not_fold_and_mcp_cannot_grant() {
    let dir = temp_dir("no-grant");
    let mut store = StrataStore::open(&dir).unwrap();
    let before = store.mesh_digest();
    let err = store
        .grant_writer([7u8; 32], Vec::new(), Vec::new(), &AdmissionContext::default())
        .unwrap_err();
    assert!(matches!(err, StoreError::Held { .. }));
    assert!(store.mesh_grants().is_empty());
    assert_eq!(store.mesh_digest(), before);

    let peer_dir = temp_dir("no-grant-peer");
    let mut peer = StrataStore::open(&peer_dir).unwrap();
    peer.ingest(input("peer fact")).unwrap();
    let export = temp_dir("no-grant-export");
    peer.export_chain(&export).unwrap();
    let err = store
        .import_chain(&export, &AdmissionContext::default())
        .unwrap_err();
    assert!(matches!(err, StoreError::Held { .. }));
    assert_eq!(store.mesh_digest(), before);
    assert!(store.mesh_grants().is_empty());
}

#[test]
fn policy_mismatch_is_a_receipt_and_does_not_fold() {
    let dir = temp_dir("policy-local");
    let mut local = StrataStore::open(&dir).unwrap();
    let before = local.mesh_digest();
    let peer_dir = temp_dir("policy-peer");
    let mut peer = StrataStore::open_with_policy(&peer_dir, permissive_policy()).unwrap();
    peer.ingest(input("other policy")).unwrap();
    let export = temp_dir("policy-export");
    peer.export_chain(&export).unwrap();
    let report = local.import_chain(&export, &mesh_grant_context()).unwrap();
    assert!(report.policy_mismatch);
    assert_eq!(local.mesh_policy_mismatches().len(), 1);
    assert_eq!(local.mesh_digest(), before);
    let proof = local.effect_by_seq(report.receipt_seq.unwrap()).unwrap().unwrap();
    assert_eq!(proof.action, EffectAction::PolicyMismatch);
}

#[test]
fn equivocating_writer_is_quarantined_with_both_heads() {
    let dir_a = temp_dir("equiv-a");
    let dir_b = temp_dir("equiv-b");
    let mut a = StrataStore::open(&dir_a).unwrap();
    let mut b = StrataStore::open(&dir_b).unwrap();
    a.ingest(input("original chain")).unwrap();
    let export = temp_dir("equiv-export");
    a.export_chain(&export).unwrap();
    b.import_chain(&export, &mesh_grant_context()).unwrap();

    let fork_parent = temp_dir("equiv-fork");
    let log_dir = fork_parent.join("log");
    std::fs::create_dir_all(&log_dir).unwrap();
    std::fs::copy(dir_a.join("log/strata.key"), log_dir.join("strata.key")).unwrap();
    let mut fork = StrataStore::open(&fork_parent).unwrap();
    assert_eq!(fork.writer_did(), a.writer_did());
    fork.ingest(input("a different history under the same key")).unwrap();
    let fork_export = temp_dir("equiv-fork-export");
    fork.export_chain(&fork_export).unwrap();
    let report = b.import_chain(&fork_export, &mesh_grant_context()).unwrap();
    assert!(report.equivocation);
    assert!(b.mesh_quarantined(&a.writer_did()));
    let proof = b.effect_by_seq(report.receipt_seq.unwrap()).unwrap().unwrap();
    assert_eq!(proof.action, EffectAction::Equivocation);
}

#[test]
fn purge_tombstone_destroys_the_local_data_key() {
    let dir_a = temp_dir("purge-a");
    let dir_b = temp_dir("purge-b");
    let mut a = StrataStore::open(&dir_a).unwrap();
    let mut b = StrataStore::open(&dir_b).unwrap();
    let id = a.ingest(input("secret fact")).unwrap();
    a.bind_data_key(&id).unwrap();
    b.bind_data_key(&id).unwrap();
    assert!(a.data_key_present(&id));
    assert!(b.data_key_present(&id));
    exchange(&mut a, &mut b);
    a.purge_local(&id).unwrap();
    assert!(!a.data_key_present(&id));
    assert!(a.mesh_tombstones().contains(&id));
    let export = temp_dir("purge-export");
    a.export_chain(&export).unwrap();
    b.import_chain(&export, &mesh_grant_context()).unwrap();
    assert!(!b.data_key_present(&id));
    assert!(b.mesh_tombstones().contains(&id));
}

#[test]
fn ambiguous_bare_id_lists_qualified_candidates() {
    let dir_a = temp_dir("amb-a");
    let dir_b = temp_dir("amb-b");
    let dir_c = temp_dir("amb-c");
    let mut a = StrataStore::open(&dir_a).unwrap();
    let mut b = StrataStore::open(&dir_b).unwrap();
    let mut c = StrataStore::open(&dir_c).unwrap();
    let id_a = a.ingest(input("left memory")).unwrap();
    let id_b = b.ingest(input("right memory")).unwrap();
    assert_eq!(id_a, id_b, "the local mint still uses the next seq");
    let export_a = temp_dir("amb-export-a");
    let export_b = temp_dir("amb-export-b");
    a.export_chain(&export_a).unwrap();
    b.export_chain(&export_b).unwrap();
    c.import_chain(&export_a, &mesh_grant_context()).unwrap();
    c.import_chain(&export_b, &mesh_grant_context()).unwrap();
    let err = c.resolve_bare_id(&id_a).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("ambiguous"), "{message}");
    assert!(message.contains(&canonical_handle(&a.writer_did(), &id_a)), "{message}");
    assert!(message.contains(&canonical_handle(&b.writer_did(), &id_b)), "{message}");
    assert_eq!(a.resolve_bare_id(&id_a).unwrap(), id_a);
}

#[test]
fn conflict_stays_open_until_a_recorded_corrects() {
    let dir_a = temp_dir("close-a");
    let dir_b = temp_dir("close-b");
    let mut a = StrataStore::open(&dir_a).unwrap();
    let mut b = StrataStore::open(&dir_b).unwrap();
    let id = a.ingest(input("subject carrier")).unwrap();
    b.ingest(input("other carrier")).unwrap();
    a.assert_claim("slot", "color", b"red", 0, i64::MAX).unwrap();
    b.assert_claim("slot", "color", b"blue", 0, i64::MAX).unwrap();
    exchange(&mut a, &mut b);
    assert_eq!(a.mesh_projection().conflicts.len(), 1);
    a.save_connection(&ConnectionRecord {
        source_id: id,
        target_id: "slot".into(),
        link_type: "corrects".into(),
        ..ConnectionRecord::default()
    })
    .unwrap();
    let export = temp_dir("close-export");
    a.export_chain(&export).unwrap();
    b.import_chain(&export, &mesh_grant_context()).unwrap();
    let export_b = temp_dir("close-export-b");
    b.export_chain(&export_b).unwrap();
    a.import_chain(&export_b, &mesh_grant_context()).unwrap();
    assert!(a.mesh_projection().conflicts.is_empty());
    assert!(b.mesh_projection().conflicts.is_empty());
    assert_eq!(a.mesh_digest(), b.mesh_digest());
}

#[test]
fn import_order_and_interleaving_reach_one_digest() {
    let mut writers = Vec::new();
    let mut exports = Vec::new();
    for name in ["w0", "w1", "w2"] {
        let dir = temp_dir(name);
        let mut store = StrataStore::open(&dir).unwrap();
        store.ingest(input(&format!("fact from {name}"))).unwrap();
        store.ingest(input("shared across writers")).unwrap();
        let export = temp_dir(&format!("{name}-export"));
        store.export_chain(&export).unwrap();
        writers.push(store);
        exports.push(export);
    }
    let orders = [vec![0, 1, 2], vec![2, 0, 1], vec![1, 2, 0]];
    let mut digests = Vec::new();
    for (index, order) in orders.iter().enumerate() {
        let dir = temp_dir(&format!("order-{index}"));
        let mut store = StrataStore::open(&dir).unwrap();
        for slot in order {
            let report = store
                .import_chain(&exports[*slot], &mesh_grant_context())
                .unwrap();
            assert!(report.receipt_seq.is_some());
        }
        for export in &exports {
            let again = store.import_chain(export, &mesh_grant_context()).unwrap();
            assert!(again.receipt_only);
        }
        digests.push(store.mesh_digest());
    }
    assert_eq!(digests[0], digests[1]);
    assert_eq!(digests[1], digests[2]);

    let dir_a = temp_dir("live-a");
    let dir_b = temp_dir("live-b");
    let mut a = StrataStore::open(&dir_a).unwrap();
    let mut b = StrataStore::open(&dir_b).unwrap();
    a.ingest(input("round one a")).unwrap();
    b.ingest(input("round one b")).unwrap();
    exchange(&mut a, &mut b);
    assert_eq!(a.mesh_digest(), b.mesh_digest());
    a.ingest(input("round two a")).unwrap();
    assert_ne!(a.mesh_digest(), b.mesh_digest());
    exchange(&mut a, &mut b);
    assert_eq!(a.mesh_digest(), b.mesh_digest());
    // The importer's merge receipt is not in the snapshot the peer just
    // exported. One more exchange carries that receipt and the frontiers meet.
    exchange(&mut a, &mut b);
    assert_eq!(a.mesh_digest(), b.mesh_digest());
    assert_eq!(a.mesh_frontier(), b.mesh_frontier());
    let _ = writers;
}

#[test]
fn fork_identity_starts_a_new_chain_at_the_fork_point() {
    let dir = temp_dir("fork");
    let mut store = StrataStore::open(&dir).unwrap();
    store.ingest(input("before the fork")).unwrap();
    let old = store.writer_did();
    let new_did = store.fork_identity().unwrap();
    assert_ne!(old, new_did);
    drop(store);
    let reopened = StrataStore::open(&dir).unwrap();
    assert_eq!(reopened.writer_did(), new_did);
    let forks = reopened.mesh_forks();
    assert_eq!(forks.len(), 1);
    assert_eq!(forks[0].0, old);
    assert!(reopened.mesh_grants().contains(&old));
    assert!(!dir.join("mesh/chains").join(fingerprint(&reopened.pinned_writer(&old).unwrap().verifying_key)).join("strata.key").exists());
}

#[test]
fn range_gap_is_denied_as_a_whole() {
    let events = vec![
        GateEvent {
            seq: 0,
            kind: RecordKind::Propose,
            payload: vec![1, 2, 3],
        },
        GateEvent {
            seq: 2,
            kind: RecordKind::Gate,
            payload: vec![4],
        },
    ];
    let failures = check_peer_range(&events, &default_policy(), "did:key:ztest", &Default::default());
    assert!(!failures.is_empty());
    let dir = temp_dir("deny");
    let mut store = StrataStore::open(&dir).unwrap();
    let before = store.mesh_digest();
    let seq = store
        .deny_range("did:key:ztest", failures.clone())
        .unwrap();
    let proof = store.effect_by_seq(seq).unwrap().unwrap();
    assert_eq!(proof.action, EffectAction::DenyRange);
    assert_eq!(store.mesh_denials()[0].1, failures);
    assert_eq!(store.mesh_digest(), before, "a denial does not fold the range");
    let _ = FrameFail {
        writer_did: "did:key:ztest".into(),
        seq: 2,
    };
}

#[test]
fn peerless_mesh_digest_is_stable_across_reopen() {
    let dir = temp_dir("reopen-mesh");
    let digest = {
        let mut store = StrataStore::open(&dir).unwrap();
        store.ingest(input("local only")).unwrap();
        let digest = store.mesh_digest();
        let state = store.state_digest();
        drop(store);
        let reopened = StrataStore::open(&dir).unwrap();
        assert_eq!(reopened.mesh_digest(), digest);
        assert_eq!(reopened.state_digest(), state);
        digest
    };
    assert_ne!(digest, [0u8; 32]);
}

#[test]
fn writer_did_matches_the_core_algorithm() {
    let key = [9u8; 32];
    let did = writer_did(&key);
    assert!(did.starts_with("did:key:z"));
    assert_eq!(did, writer_did(&key));
}

#[test]
fn replica_refuses_a_source_that_contains_a_signing_seed() {
    let dir = temp_dir("seed-src");
    let mut store = StrataStore::open(&dir).unwrap();
    store.ingest(input("x")).unwrap();
    let export = temp_dir("seed-export");
    store.export_chain(&export).unwrap();
    std::fs::write(export.join("strata.key"), [3u8; 32]).unwrap();
    let peer_dir = temp_dir("seed-peer");
    let mut peer = StrataStore::open(&peer_dir).unwrap();
    let err = peer.import_chain(&export, &mesh_grant_context()).unwrap_err();
    assert!(err.to_string().contains("strata.key"), "{err}");
    assert_eq!(std::fs::read(export.join("strata.key")).unwrap(), [3u8; 32]);
}
