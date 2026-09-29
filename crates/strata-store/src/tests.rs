//! In-crate tests. Each test owns a unique temp directory (the log enforces
//! a single writer per directory via `strata.lock`).

use std::collections::BTreeSet;
use std::path::PathBuf;

use borsh::BorshDeserialize;
use strata_gate::policy::{ANY_KIND, WILDCARD_PREFIX};
use strata_gate::record::{action_kind, EffectRecord, GateRecord, Propose, RecordKind, Verdict};
use strata_gate::{Policy, Rule};

use crate::op::{KIND_STORE_CHECKPOINT, KIND_STORE_WRITE};
use crate::store::handle_of;
use crate::types::{ConnectionRecord, EdgeDirection, EdgeKind, IngestInput};
use crate::{
    default_policy, effect_receipt_id, looks_like_failure, retire_rule_id, AdmissionContext,
    RetireReceipt, StoreError, StrataStore, RULE_EDIT, RULE_INTENTIONS, RULE_PURGE, RULE_SUPPRESS,
};

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("strata-store-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Allow-everything policy (used where a test needs RETIRE actions to land).
fn permissive_policy() -> Policy {
    Policy {
        rules: vec![Rule {
            match_kind: ANY_KIND,
            match_params_hash_prefix: WILDCARD_PREFIX,
            max_blast_radius: u32::MAX,
            forbid_forgotten_lessons: false,
            require_human: false,
            verdict: Verdict::Allow,
        }],
    }
}

fn input(content: &str, tags: &[&str]) -> IngestInput {
    IngestInput {
        content: content.to_string(),
        node_type: String::new(),
        tags: tags.iter().map(|t| t.to_string()).collect(),
        created_at_ms: Some(1_700_000_000_000),
        valid_from_ms: None,
        valid_until_ms: None,
    }
}

#[test]
fn ingest_read_roundtrip() {
    let dir = temp_dir("roundtrip");
    let mut store = StrataStore::open(&dir).expect("open");

    let a = store
        .ingest_in_scope(
            input("postgres deadlocks under concurrent vacuum", &["db", "ops"]),
            "proj",
        )
        .expect("ingest a");
    let b = store
        .ingest(input("the fix was row-level advisory locks", &[]))
        .expect("ingest b");

    let node = store.get_node(&a).expect("node a");
    assert_eq!(node.content, "postgres deadlocks under concurrent vacuum");
    assert_eq!(node.scope, "proj");
    assert_eq!(node.tags, vec!["db".to_string(), "ops".to_string()]);
    assert_eq!(node.node_type, "fact");
    assert_eq!(node.created_at_ms, 1_700_000_000_000);
    assert_eq!(node.valid_from_ms, node.created_at_ms);
    assert_eq!(node.valid_until_ms, crate::VALID_FOREVER_MS);
    assert!(node.superseded_by.is_none());

    let scoped = store.get_all_nodes_in_scope("proj");
    assert_eq!(scoped.len(), 1);
    assert_eq!(scoped[0].id, a);
    // Default scope keeps its own node.
    assert_eq!(store.get_all_nodes_in_scope("").len(), 1);
    assert!(store.get_node(&b).is_some());
    assert!(store.get_node("mem-does-not-exist").is_none());

    // FSRS: the ingest folded one review per card; retrievability is derived.
    assert_eq!(store.review_event_count(), 2);
    let r = store
        .retrievability(&a)
        .expect("retrievability")
        .expect("card exists");
    assert!((0.0..=1.0).contains(&r), "retrievability in (0,1]: {r}");
    let card = store.card_state(&a).expect("card");
    assert_eq!(card.review_count, 1);
    assert_eq!(card.last_seq, 4); // first ingest's data frame lands at log seq 4

    store
        .verify_checkpoint_chain()
        .expect("chain ok with no checkpoints");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn edge_write_and_reverse_query() {
    let dir = temp_dir("edges");
    let mut store = StrataStore::open(&dir).expect("open");
    let a = store.ingest(input("node a", &[])).expect("a");
    let b = store.ingest(input("node b", &[])).expect("b");

    let edge = ConnectionRecord {
        source_id: a.clone(),
        target_id: b.clone(),
        strength_milli: 750,
        link_type: EdgeKind::DerivedFrom.as_str().to_string(),
        meta_sha: Some("abc123".to_string()),
        created_at_ms: 42,
        activation_count: 0,
    };
    store.save_connection(&edge).expect("save edge");

    // Forward query from the source.
    let out = store.get_edges_for(&a, EdgeDirection::Outgoing, Some(EdgeKind::DerivedFrom));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].target_id, b);
    assert_eq!(out[0].strength_milli, 750);
    assert_eq!(out[0].strength(), 0.75);

    // Reverse query: the target sees the edge as incoming.
    let inc = store.get_edges_for(&b, EdgeDirection::Incoming, None);
    assert_eq!(inc.len(), 1);
    assert_eq!(inc[0].source_id, a);

    // get_connections_for_memory spans both directions.
    assert_eq!(store.get_connections_for_memory(&a).len(), 1);
    assert_eq!(store.get_connections_for_memory(&b).len(), 1);

    // Anchored edge to a non-memory target is allowed and reverse-indexed.
    store
        .save_connection(&ConnectionRecord {
            source_id: b.clone(),
            target_id: "src/main.rs".to_string(),
            ..ConnectionRecord::default()
        })
        .expect("anchor edge");
    assert_eq!(
        store
            .get_edges_for("src/main.rs", EdgeDirection::Incoming, None)
            .len(),
        1
    );

    // Non-vocabulary link types are rejected before any frame is written.
    let head_before = store.log().head().frames_total;
    let bad = ConnectionRecord {
        link_type: "semantic".into(),
        source_id: a.clone(),
        target_id: b,
        ..ConnectionRecord::default()
    };
    assert!(matches!(
        store.save_connection(&bad),
        Err(StoreError::InvalidInput(_))
    ));
    assert_eq!(store.log().head().frames_total, head_before);

    // Kind filtering works across the vocabulary.
    assert_eq!(
        store
            .get_edges_for(&a, EdgeDirection::Both, Some(EdgeKind::Touched))
            .len(),
        0
    );
    assert_eq!(store.edge_count(), 2);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn reopen_rebuilds_bit_identically() {
    let dir = temp_dir("reopen");
    let digest1;
    let pairs1;
    let never1;
    {
        let mut store = StrataStore::open_with_policy(&dir, permissive_policy()).expect("open");
        let a = store
            .ingest_in_scope(input("fact one", &["x"]), "proj")
            .expect("a");
        let b = store
            .ingest_in_scope(input("fact two", &["y"]), "proj")
            .expect("b");
        let c = store
            .ingest_in_scope(input("superseded fact", &[]), "proj")
            .expect("c");
        let d = store
            .ingest_in_scope(input("unlinked fact", &[]), "proj")
            .expect("d");
        assert!(store.get_node(&d).expect("d").is_live());

        store.set_created_at(&a, 123).expect("set_created_at");
        store
            .save_connection(&ConnectionRecord {
                source_id: a.clone(),
                target_id: b.clone(),
                link_type: EdgeKind::EvidenceOf.as_str().to_string(),
                ..ConnectionRecord::default()
            })
            .expect("edge");
        store.review(&a, 2).expect("review a");
        store.review(&b, 4).expect("review b");
        store.seal_checkpoint().expect("seal 1");

        store.review(&a, 3).expect("review a again");
        store
            .supersede(&c, &a)
            .expect("supersede lands under permissive policy");
        store.seal_checkpoint().expect("seal 2");

        assert_eq!(store.supersession_pairs(), vec![(c.clone(), a.clone())]);
        pairs1 = store.supersession_pairs();
        never1 = store.get_never_composed("proj", 10);
        assert!(!never1.is_empty(), "d is uncomposed with a and b");
        digest1 = store.state_digest();
    }
    // open -> close -> open rebuilds identical maps from the log alone.
    let store2 = StrataStore::open_with_policy(&dir, permissive_policy()).expect("reopen");
    assert_eq!(
        store2.state_digest(),
        digest1,
        "derived maps differ after reopen"
    );
    assert_eq!(store2.node_count(), 3);
    assert_eq!(store2.edge_count(), 1);
    assert_eq!(store2.review_event_count(), 7); // 4 ingest folds + 3 explicit reviews
    assert_eq!(store2.supersession_pairs(), pairs1);
    assert_eq!(store2.get_never_composed("proj", 10), never1);
    assert_eq!(store2.checkpoints().len(), 2);
    store2
        .verify_checkpoint_chain()
        .expect("checkpoint chain verifies");
    assert!(store2.sweep().is_empty(), "sweep must be clean");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn deny_policy_blocks_effect() {
    let dir = temp_dir("deny");
    {
        let mut store = StrataStore::open_with_policy(&dir, Policy::default()).expect("open");
        let err = store
            .ingest(input("should never land", &[]))
            .expect_err("deny-all policy");
        assert!(matches!(err, StoreError::Denied { .. }));

        // Nothing landed in the derived maps...
        assert_eq!(store.node_count(), 0);
        // ...and the log holds exactly PROPOSE + GATE(deny), no EFFECT, no data.
        let frames = store.log().read_frames(1).expect("frames");
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].kind, RecordKind::Propose.to_u8());
        assert_eq!(frames[1].kind, RecordKind::Gate.to_u8());
        let gate = strata_gate::record::GateRecord::try_from_slice(&frames[1].payload).unwrap();
        assert_eq!(gate.verdict, Verdict::Deny);
    }
    // Reopen under the default policy: still empty, no orphans, clean sweep.
    let store2 = StrataStore::open(&dir).expect("reopen");
    assert_eq!(store2.node_count(), 0);
    assert_eq!(store2.orphan_write_count(), 0);
    assert!(store2.sweep().is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn default_policy_holds_retire_permissive_lands_it() {
    let dir = temp_dir("retire");
    let (old, new) = {
        let mut store = StrataStore::open(&dir).expect("open with default policy");
        let old = store.ingest(input("old fact", &[])).expect("old");
        let new = store.ingest(input("new fact", &[])).expect("new");

        let err = store
            .supersede(&old, &new)
            .expect_err("default policy holds RETIRE");
        assert!(matches!(err, StoreError::Held { .. }));
        assert!(store.get_node(&old).expect("old").superseded_by.is_none());
        assert!(store.supersession_pairs().is_empty());
        (old, new)
    };
    // The writer lock releases on drop; a review-gated policy lands the
    // supersession on reopen.
    let mut gated = StrataStore::open_with_policy(&dir, permissive_policy()).expect("reopen gated");
    gated.supersede(&old, &new).expect("supersede lands");
    assert_eq!(gated.supersession_pairs(), vec![(old.clone(), new.clone())]);
    assert_eq!(
        gated.get_all_nodes_in_scope("").len(),
        1,
        "superseded node leaves the live set"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn checkpoint_chain_verifies_and_tamper_detected() {
    let dir = temp_dir("checkpoints");
    {
        let mut store = StrataStore::open(&dir).expect("open");
        let a = store.ingest(input("fact a", &[])).expect("a");
        store.review(&a, 3).expect("review");
        let cp1 = store.seal_checkpoint().expect("seal 1");
        store.review(&a, 1).expect("hard review");
        let cp2 = store.seal_checkpoint().expect("seal 2");

        // Chain shape: v2 links back to v1, roots commit the folded state.
        assert_eq!(
            cp2.prev_checkpoint,
            strata_kernel::checkpoint::checkpoint_hash(&cp1)
        );
        assert_eq!(cp1.prev_checkpoint, [0u8; 32]);
        assert!(cp2.log_seq > cp1.log_seq);
        store.verify_checkpoint_chain().expect("chain verifies");
    }
    // Tamper the external anchor: the successor-less head checkpoint's hash
    // no longer matches, so open refuses the store.
    {
        let meta = dir.join("store.meta");
        let mut bytes = std::fs::read(&meta).expect("meta");
        let len = bytes.len();
        bytes[len - 1] ^= 0xff; // head_log_seq's last byte
        std::fs::write(&meta, &bytes).expect("rewrite meta");
        // Borsh still decodes (same lengths, different value) -> wrong anchor.
        let result = StrataStore::open(&dir);
        assert!(
            matches!(result, Err(StoreError::Verify(_))),
            "tampered anchor must fail open"
        );
    }
    // Flipping a hash byte instead of the seq byte also trips verification.
    {
        let meta = dir.join("store.meta");
        let mut bytes = std::fs::read(&meta).expect("meta");
        bytes[8] ^= 0xff; // inside head_checkpoint_hash
        std::fs::write(&meta, &bytes).expect("rewrite meta");
        assert!(matches!(
            StrataStore::open(&dir),
            Err(StoreError::Verify(_))
        ));
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn backup_roundtrip_opens_and_matches() {
    let dir = temp_dir("backup-src");
    let dest = temp_dir("backup-dest");
    let digest1;
    {
        let mut store = StrataStore::open(&dir).expect("open");
        let a = store.ingest(input("backed up fact", &[])).expect("a");
        store.review(&a, 4).expect("review");
        store.seal_checkpoint().expect("seal");
        store.backup_to(&dest).expect("backup");

        // The live store keeps working after the backup sealed its segment.
        store
            .ingest(input("post-backup fact", &[]))
            .expect("post-backup ingest");
        digest1 = store.state_digest();
    }
    // The backup reopens as a full store: same log prefix, same derived state
    // as at backup time (pre "post-backup fact").
    let backup = StrataStore::open(&dest).expect("open backup");
    assert_eq!(backup.node_count(), 1);
    assert_eq!(backup.checkpoints().len(), 1);
    backup
        .verify_checkpoint_chain()
        .expect("backup chain verifies");

    // And the ORIGINAL still holds the extra mutation.
    let reopened = StrataStore::open(&dir).expect("reopen original");
    assert_eq!(reopened.node_count(), 2);
    assert_eq!(reopened.state_digest(), digest1);
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&dest).ok();
}

#[test]
fn every_store_write_has_an_admitting_effect() {
    let dir = temp_dir("admission");
    let mut store = StrataStore::open_with_policy(&dir, permissive_policy()).expect("open");
    let a = store.ingest(input("a", &[])).expect("a");
    let b = store.ingest(input("b", &[])).expect("b");
    store.review(&a, 3).expect("review");
    store
        .save_connection(&ConnectionRecord {
            source_id: a.clone(),
            target_id: b.clone(),
            ..ConnectionRecord::default()
        })
        .expect("edge");
    store.supersede(&b, &a).expect("supersede");
    store.seal_checkpoint().expect("seal");

    let frames = store.log().read_frames(1).expect("frames");
    let mut admitted: BTreeSet<[u8; 32]> = BTreeSet::new();
    let mut data_frames = 0usize;
    let mut checkpoint_frames = 0usize;
    for f in &frames {
        if f.kind == RecordKind::Effect.to_u8() {
            let e = strata_gate::record::EffectRecord::try_from_slice(&f.payload).unwrap();
            admitted.insert(e.payload_digest);
        } else if f.kind == KIND_STORE_WRITE {
            data_frames += 1;
            let digest = *blake3::hash(&f.payload).as_bytes();
            assert!(
                admitted.contains(&digest),
                "data frame at seq {} unadmitted",
                f.seq
            );
        } else if f.kind == KIND_STORE_CHECKPOINT {
            checkpoint_frames += 1;
        }
    }
    assert_eq!(data_frames, 5); // 2 upserts + 1 review + 1 edge + 1 supersede
    assert_eq!(checkpoint_frames, 1);

    // Re-deriving every verdict under the pinned policy matches the log.
    let verdicts = store.rederive_verdicts().expect("rederive");
    assert!(verdicts.iter().all(|(_, v)| *v == Verdict::Allow));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn reads_append_nothing() {
    let dir = temp_dir("reads");
    let mut store = StrataStore::open(&dir).expect("open");
    let a = store.ingest(input("read me", &[])).expect("a");
    let head_before = store.log().head().frames_total;

    store.get_node(&a);
    store.get_all_nodes_in_scope("");
    store.get_connections_for_memory(&a);
    store.get_never_composed("", 5);
    store.supersession_pairs();
    store.retrievability(&a).expect("retrievability");
    store.is_failure_memory(&a);
    store.state_digest();

    assert_eq!(
        store.log().head().frames_total,
        head_before,
        "reads must not append"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn failure_marker_port_matches_vestige_semantics() {
    // Same canary cases vestige-core tests for.
    assert!(looks_like_failure("connection pool saturated at 100%", &[]));
    assert!(!looks_like_failure(
        "pinned rails to 5.2 for the tz workaround",
        &[]
    ));
    assert!(looks_like_failure(
        "deploy went fine",
        &["incident-postmortem".to_string()]
    ));
    assert!(!looks_like_failure("the migration added a column", &[]));
    // Whole-word, not substring.
    assert!(!looks_like_failure(
        "slowly but surely the index warmed",
        &[]
    ));
    assert!(looks_like_failure("the replica lag grew unbounded", &[]));

    let dir = temp_dir("failure");
    let mut store = StrataStore::open(&dir).expect("open");
    let f = store
        .ingest(input("api returned 502 after the release", &[]))
        .expect("f");
    let q = store
        .ingest(input("a quiet architectural note", &[]))
        .expect("q");
    assert_eq!(store.is_failure_memory(&f), Some(true));
    assert_eq!(store.is_failure_memory(&q), Some(false));
    assert_eq!(store.is_failure_memory("nope"), None);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn node_ids_are_log_derived_and_handle_is_stable() {
    let dir = temp_dir("ids");
    let mut store = StrataStore::open(&dir).expect("open");
    let a = store.ingest(input("first", &[])).expect("a");
    let b = store.ingest(input("second", &[])).expect("b");
    // Each ingest advances the head by exactly 4 frames (P/G/E + data).
    assert_ne!(a, b);
    assert!(a.starts_with("mem-"));
    // handle_of is a pure function of the id (used as the FSRS card id).
    assert_eq!(handle_of(&a), handle_of(&a));
    assert_ne!(handle_of(&a), handle_of(&b));
    assert_eq!(store.card_state(&a).map(|c| c.review_count), Some(1));
    std::fs::remove_dir_all(&dir).ok();
}

fn ctx(rule: Option<&str>, confirm: bool) -> AdmissionContext {
    AdmissionContext {
        rule_id: rule.map(str::to_string),
        confirm,
    }
}

fn assert_held(err: StoreError) {
    assert!(matches!(err, StoreError::Held { .. }), "{err}");
}

fn effect_frames(store: &StrataStore) -> Vec<(u64, EffectRecord)> {
    let mut out = Vec::new();
    let mut gseq = 0u64;
    for frame in store.log().read_frames(1).expect("frames") {
        let Some(kind) = RecordKind::from_u8(frame.kind) else {
            continue;
        };
        if kind == RecordKind::Effect {
            let effect = EffectRecord::try_from_slice(&frame.payload).expect("effect");
            out.push((gseq, effect));
        }
        gseq += 1;
    }
    out
}

fn propose_at(store: &StrataStore, propose_seq: u64) -> Propose {
    let mut gseq = 0u64;
    for frame in store.log().read_frames(1).expect("frames") {
        let Some(kind) = RecordKind::from_u8(frame.kind) else {
            continue;
        };
        if kind == RecordKind::Propose && gseq == propose_seq {
            return Propose::try_from_slice(&frame.payload).expect("propose");
        }
        gseq += 1;
    }
    panic!("no propose at gate seq {propose_seq}");
}

/// The admitting EffectRecord's proposal carries the rule id, and the
/// receipt id is `eff-` plus that effect seq.
fn assert_rule_receipt(store: &StrataStore, receipt: &RetireReceipt, rule: &'static str) {
    assert_eq!(receipt.rule_id, Some(rule));
    assert_eq!(receipt.receipt_id, effect_receipt_id(receipt.effect_seq));
    assert!(receipt.receipt_id.starts_with("eff-"));
    let effect = effect_frames(store)
        .into_iter()
        .find(|(seq, _)| *seq == receipt.effect_seq)
        .map(|(_, effect)| effect)
        .expect("EffectRecord");
    let propose = propose_at(store, effect.propose_seq);
    assert_eq!(propose.action_kind, action_kind::RETIRE);
    assert_eq!(propose.action_hash, effect.action_hash);
    assert_eq!(retire_rule_id(&propose.params_hash), Some(rule));
    let digest_landed = store
        .log()
        .read_frames(1)
        .expect("frames")
        .into_iter()
        .any(|frame| {
            frame.kind == KIND_STORE_WRITE
                && *blake3::hash(&frame.payload).as_bytes() == effect.payload_digest
        });
    assert!(digest_landed, "STORE_WRITE cited by the effect");
}

fn stored_gate_verdicts(store: &StrataStore) -> Vec<(u64, Verdict)> {
    let mut out = Vec::new();
    let mut gseq = 0u64;
    for frame in store.log().read_frames(1).expect("frames") {
        let Some(kind) = RecordKind::from_u8(frame.kind) else {
            continue;
        };
        if kind == RecordKind::Gate {
            let gate = GateRecord::try_from_slice(&frame.payload).expect("gate");
            out.push((gseq, gate.verdict));
        }
        gseq += 1;
    }
    out
}

fn pair(name: &str) -> (PathBuf, StrataStore, String, String) {
    let dir = temp_dir(name);
    let mut store = StrataStore::open(&dir).expect("open");
    let old = store.ingest(input("predecessor", &[])).expect("old");
    let successor = store.ingest(input("successor", &[])).expect("new");
    (dir, store, old, successor)
}

fn expect_allow(
    dir: &PathBuf,
    store: StrataStore,
    receipt: RetireReceipt,
    rule: &'static str,
    old: &str,
) {
    assert_rule_receipt(&store, &receipt, rule);
    assert!(store.get_node(old).expect("old").superseded_by.is_some());
    assert_eq!(
        store.rederive_verdicts().expect("rederive"),
        stored_gate_verdicts(&store)
    );
    assert!(store.sweep().is_empty());
    let seq = receipt.effect_seq;
    let receipt_id = receipt.receipt_id.clone();
    drop(store);
    let reopened = StrataStore::open(dir).expect("reopen");
    let again = reopened.retire_receipt(seq).expect("replayed receipt");
    assert_eq!(again.rule_id, Some(rule));
    assert_eq!(again.receipt_id, receipt_id);
    assert_rule_receipt(&reopened, &again, rule);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn named_retire_rules_precede_the_catchall_hold() {
    let policy = default_policy();
    assert_eq!(policy.rules.len(), 6);
    let ids = [RULE_EDIT, RULE_SUPPRESS, RULE_INTENTIONS, RULE_PURGE];
    let mut prefixes = Vec::new();
    for (rule, id) in policy.rules.iter().take(4).zip(ids) {
        assert_eq!(rule.match_kind, action_kind::RETIRE);
        assert_eq!(rule.verdict, Verdict::Allow);
        assert_ne!(rule.match_params_hash_prefix, WILDCARD_PREFIX);
        let mut hash = [0u8; 32];
        hash[..8].copy_from_slice(&rule.match_params_hash_prefix);
        assert_eq!(retire_rule_id(&hash), Some(id));
        prefixes.push(rule.match_params_hash_prefix);
    }
    prefixes.sort();
    prefixes.dedup();
    assert_eq!(prefixes.len(), 4, "the four rule prefixes must be distinct");
    assert_eq!(policy.rules[4].match_kind, action_kind::RETIRE);
    assert_eq!(policy.rules[4].verdict, Verdict::Hold);
    assert_eq!(policy.rules[4].match_params_hash_prefix, WILDCARD_PREFIX);
    assert_eq!(policy.rules[5].match_kind, ANY_KIND);
    assert_eq!(policy.rules[5].verdict, Verdict::Allow);
}

#[test]
fn edit_allows_only_with_rule_id_and_successor_in_the_call() {
    let dir = temp_dir("edit-allow");
    let mut store = StrataStore::open(&dir).expect("open");
    let old = store.ingest(input("edit", &["suppress"])).expect("old");
    store.begin_tool_call();
    let successor = store.ingest(input("successor", &[])).expect("successor");
    let receipt = store
        .retire(&old, &successor, &ctx(Some(RULE_EDIT), false))
        .expect("edit allows");
    expect_allow(&dir, store, receipt, RULE_EDIT, &old);
}

#[test]
fn edit_held_without_rule_id() {
    let (dir, mut store, old, successor) = pair("edit-no-id");
    store.begin_tool_call();
    let err = store
        .retire(&old, &successor, &ctx(None, false))
        .expect_err("no rule id");
    assert_held(err);
    assert!(store.get_node(&old).unwrap().superseded_by.is_none());
    drop(store);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn edit_held_with_unknown_rule_id() {
    let (dir, mut store, old, successor) = pair("edit-unknown");
    store.begin_tool_call();
    let err = store
        .retire(&old, &successor, &ctx(Some("edited"), true))
        .expect_err("unknown id");
    assert_held(err);
    drop(store);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn edit_held_when_successor_was_not_admitted_in_this_call() {
    let (dir, mut store, old, successor) = pair("edit-stale");
    let err = store
        .retire(&old, &successor, &ctx(Some(RULE_EDIT), false))
        .expect_err("successor predates the call");
    assert_held(err);
    store.begin_tool_call();
    store.end_tool_call();
    let err = store
        .retire(&old, &successor, &ctx(Some(RULE_EDIT), false))
        .expect_err("call closed");
    assert_held(err);
    assert!(store.get_node(&old).unwrap().superseded_by.is_none());
    drop(store);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn suppress_allows_with_rule_id() {
    let (dir, mut store, old, successor) = pair("suppress-allow");
    let receipt = store
        .retire(&old, &successor, &ctx(Some(RULE_SUPPRESS), false))
        .expect("suppress allows");
    expect_allow(&dir, store, receipt, RULE_SUPPRESS, &old);
}

#[test]
fn suppress_held_without_rule_id() {
    let (dir, mut store, old, successor) = pair("suppress-no-id");
    let err = store
        .retire(&old, &successor, &ctx(None, true))
        .expect_err("no rule id");
    assert_held(err);
    drop(store);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn suppress_held_with_unknown_rule_id() {
    let (dir, mut store, old, successor) = pair("suppress-unknown");
    let err = store
        .retire(&old, &successor, &ctx(Some("suppressed"), false))
        .expect_err("unknown id");
    assert_held(err);
    drop(store);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn intentions_allows_with_rule_id() {
    let (dir, mut store, old, successor) = pair("intentions-allow");
    let receipt = store
        .retire(&old, &successor, &ctx(Some(RULE_INTENTIONS), false))
        .expect("intentions allows");
    expect_allow(&dir, store, receipt, RULE_INTENTIONS, &old);
}

#[test]
fn intentions_held_without_rule_id() {
    let (dir, mut store, old, successor) = pair("intentions-no-id");
    let err = store
        .retire(&old, &successor, &AdmissionContext::default())
        .expect_err("no rule id");
    assert_held(err);
    drop(store);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn intentions_held_with_unknown_rule_id() {
    let (dir, mut store, old, successor) = pair("intentions-unknown");
    let err = store
        .retire(&old, &successor, &ctx(Some("intention"), false))
        .expect_err("unknown id");
    assert_held(err);
    drop(store);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn purge_allows_with_rule_id_and_confirm() {
    let (dir, mut store, old, successor) = pair("purge-allow");
    let receipt = store
        .retire(&old, &successor, &ctx(Some(RULE_PURGE), true))
        .expect("purge allows");
    expect_allow(&dir, store, receipt, RULE_PURGE, &old);
}

#[test]
fn purge_held_without_rule_id() {
    let (dir, mut store, old, successor) = pair("purge-no-id");
    let err = store
        .retire(&old, &successor, &ctx(None, true))
        .expect_err("confirm without a rule id");
    assert_held(err);
    drop(store);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn purge_held_with_unknown_rule_id() {
    let (dir, mut store, old, successor) = pair("purge-unknown");
    let err = store
        .retire(&old, &successor, &ctx(Some("purged"), true))
        .expect_err("unknown id");
    assert_held(err);
    drop(store);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn purge_held_without_confirm() {
    let (dir, mut store, old, successor) = pair("purge-noconfirm");
    let err = store
        .retire(&old, &successor, &ctx(Some(RULE_PURGE), false))
        .expect_err("purge without confirm");
    assert_held(err);
    assert!(store.get_node(&old).unwrap().superseded_by.is_none());
    drop(store);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn retire_without_rule_id_stays_held() {
    let dir = temp_dir("no-rule");
    let mut store = StrataStore::open(&dir).expect("open");
    let old = store
        .ingest(input("edit suppress intentions purge", &["purge", "edit"]))
        .expect("old");
    let successor = store
        .ingest(input("suppress", &["intentions"]))
        .expect("successor");
    let before = effect_frames(&store).len();
    let err = store.supersede(&old, &successor).expect_err("supersede");
    assert_held(err);
    let err = store
        .retire(&old, &successor, &AdmissionContext::default())
        .expect_err("empty context");
    assert_held(err);
    assert_eq!(effect_frames(&store).len(), before);
    assert_eq!(
        store.rederive_verdicts().expect("rederive"),
        stored_gate_verdicts(&store)
    );
    assert!(store.sweep().is_empty());
    assert!(store.get_node(&old).unwrap().superseded_by.is_none());
    assert!(store.supersession_pairs().is_empty());
    drop(store);
    std::fs::remove_dir_all(&dir).ok();
}
