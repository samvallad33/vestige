//! In-crate tests. Each test owns a unique temp directory (the log enforces
//! a single writer per directory via `strata.lock`).

use std::collections::BTreeSet;
use std::path::PathBuf;

use borsh::BorshDeserialize;
use strata::payload_blake3;
use strata_gate::policy::{ANY_KIND, WILDCARD_PREFIX};
use strata_gate::record::{action_kind, EffectRecord, GateRecord, Propose, RecordKind, Verdict};
use strata_gate::{Policy, Rule};
use strata_kernel::fsrs::{FsrsFold, ALGO_V2};

use crate::op::{StoreOp, KIND_STORE_CHECKPOINT, KIND_STORE_WRITE};
use crate::store::{
    classify_checkpoint_payload, classify_write_payload, decode_exact, handle_of, migration_edge,
    migration_node, CheckpointPayload, WritePayload,
};
use crate::types::{ConnectionRecord, EdgeDirection, EdgeKind, IngestInput};
use crate::{
    default_policy, effect_receipt_id, looks_like_failure, retire_rule_id, AdmissionContext,
    EffectAction, RetireReceipt, StoreError, StrataStore, RULE_EDIT, RULE_INTENTIONS, RULE_PURGE,
    RULE_SUPPRESS,
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
        source: None,
        source_updated_at_ms: None,
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
    assert!(node.source.is_none());
    assert!(node.source_updated_at_ms.is_none());

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
fn ingest_keeps_source_and_replays_it() {
    let dir = temp_dir("source");
    let mut store = StrataStore::open(&dir).expect("open");
    let updated_ms = 1_771_585_800_000;
    let id = store
        .ingest(IngestInput {
            content: "imported from v3".into(),
            source: Some(crate::SourceKey {
                system: "fixture".into(),
                project: String::new(),
                id: String::new(),
            }),
            source_updated_at_ms: Some(updated_ms),
            ..input("ignored", &[])
        })
        .expect("ingest");
    let node = store.get_node(&id).expect("node");
    let source = node.source.expect("source dropped");
    assert_eq!(source.system, "fixture");
    assert!(source.project.is_empty());
    assert!(source.id.is_empty());
    assert_eq!(node.source_updated_at_ms, Some(updated_ms));
    drop(store);

    let reopened = StrataStore::open(&dir).expect("reopen");
    let again = reopened.get_node(&id).expect("replayed");
    assert_eq!(
        again.source.as_ref().map(|key| key.system.as_str()),
        Some("fixture")
    );
    assert_eq!(again.source_updated_at_ms, Some(updated_ms));
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
fn missing_store_meta_on_populated_store_fails_open() {
    let dir = temp_dir("missing-meta");
    {
        let mut store = StrataStore::open(&dir).expect("open");
        let id = store.ingest(input("populated fact", &[])).expect("ingest");
        store.review(&id, 3).expect("review");
        store.seal_checkpoint().expect("seal writes store.meta");
        assert!(dir.join("store.meta").is_file());
        assert_eq!(store.node_count(), 1);
    }
    std::fs::remove_file(dir.join("store.meta")).expect("delete store.meta");
    let opened = StrataStore::open(&dir);
    let msg = match opened {
        Ok(_) => panic!("open must fail"),
        Err(err) => err.to_string(),
    };
    assert!(
        msg.contains("store.meta is missing") && msg.contains("log has frames"),
        "{msg}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Every directory and file of a backup is owner-only on unix, whatever the
/// process umask is.
#[cfg(unix)]
#[test]
fn backup_is_owner_only() {
    use std::os::unix::fs::PermissionsExt;

    fn walk(path: &std::path::Path, bad: &mut Vec<String>) {
        let meta = std::fs::metadata(path).expect("stat");
        let mode = meta.permissions().mode() & 0o777;
        let want = if meta.is_dir() { 0o700 } else { 0o600 };
        if mode != want {
            bad.push(format!("{} is {mode:o}, want {want:o}", path.display()));
        }
        if meta.is_dir() {
            for entry in std::fs::read_dir(path).expect("read_dir") {
                walk(&entry.expect("entry").path(), bad);
            }
        }
    }

    let dir = temp_dir("backup-perm-src");
    let parent = temp_dir("backup-perm-parent");
    let dest = parent.join("nested").join("copy");
    {
        let mut store = StrataStore::open(&dir).expect("open");
        store.ingest(input("private fact", &[])).expect("ingest");
        store.seal_checkpoint().expect("seal");
        store.backup_to(&dest).expect("backup");
    }
    let mut bad = Vec::new();
    walk(&dest, &mut bad);
    assert!(
        std::fs::metadata(dest.join("log")).is_ok(),
        "backup has a log dir"
    );
    assert!(bad.is_empty(), "backup is not owner-only: {bad:?}");
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&parent).ok();
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
fn replay_loads_imported_nodes_and_keeps_edge_kinds() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../strata-migrate/tests/fixtures/v3.1.1-sample.sqlite");
    let dir = temp_dir("import");
    let log = dir.join("log");
    strata_migrate::migrate(&fixture, &log).expect("migrate");
    let store = StrataStore::open(&dir).expect("replay imported log");
    let ids: BTreeSet<_> = store.nodes().into_iter().map(|node| node.id).collect();
    assert!(ids.contains("11111111-1111-4111-8111-111111111111"));
    assert!(ids.contains("22222222-2222-4222-8222-222222222222"));
    assert!(ids.contains("33333333-3333-4333-8333-333333333333"));
    let edges = store.edges();
    assert_eq!(edges.len(), 3);
    assert!(edges.iter().any(|edge| edge.link_type == "touched"));
    assert!(edges.iter().any(|edge| {
        edge.link_type == "legacy_inferred" && EdgeKind::parse(&edge.link_type).is_none()
    }));
    assert!(edges.iter().all(|edge| edge.link_type != "derived_from"));
    let touched = store.get_edges_for(
        "11111111-1111-4111-8111-111111111111",
        EdgeDirection::Outgoing,
        Some(EdgeKind::Touched),
    );
    assert_eq!(touched.len(), 1);
    assert_eq!(touched[0].target_id, "33333333-3333-4333-8333-333333333333");
    std::fs::remove_dir_all(&dir).ok();
}

fn log_frames(dir: &std::path::Path) -> Vec<strata::FrameRecord> {
    let log = strata::StrataLog::open(dir.join("log")).expect("reopen log");
    log.read_frames(1).expect("read frames")
}

/// Frames the real store writes, and frames migrated from the v3.1.1 fixture.
/// Replay calls `classify_*`. Crafted payloads can still satisfy both decoders.
#[test]
fn replay_store_and_migration_frames_do_not_cross_classify() {
    let dir = temp_dir("kinds");
    let mut store = StrataStore::open_with_policy(&dir, permissive_policy()).expect("open");
    let alpha = store.ingest(input("alpha record", &["t"])).expect("alpha");
    let beta = store.ingest(input("beta record", &[])).expect("beta");
    store
        .save_connection(&ConnectionRecord {
            source_id: alpha.clone(),
            target_id: beta.clone(),
            strength_milli: 1000,
            link_type: EdgeKind::Touched.as_str().to_string(),
            meta_sha: None,
            created_at_ms: 7,
            activation_count: 1,
        })
        .expect("edge");
    store.review(&alpha, 3).expect("review");
    store.supersede(&alpha, &beta).expect("supersede");
    let sealed = store.seal_checkpoint().expect("checkpoint");
    let reviews = store.review_event_count();
    drop(store);

    let frames = log_frames(&dir);
    let mut saw_upsert = false;
    let mut saw_edge = false;
    let mut saw_supersede = false;
    let mut saw_review = false;
    let mut store_checkpoints = 0usize;
    for frame in &frames {
        if frame.kind == KIND_STORE_WRITE {
            let op = decode_exact::<StoreOp>(&frame.payload).expect("store write is a StoreOp");
            assert!(
                decode_exact::<strata_migrate::NodeRecord>(&frame.payload).is_none(),
                "StoreOp payload also round-trips as a migration node"
            );
            assert!(matches!(
                classify_write_payload(&frame.payload),
                WritePayload::StoreOp(_)
            ));
            match op {
                StoreOp::UpsertNode { .. } => saw_upsert = true,
                StoreOp::SaveEdge { .. } => saw_edge = true,
                StoreOp::SupersedeNode { .. } => saw_supersede = true,
                StoreOp::ReviewNode { .. } => saw_review = true,
                // This fixture writes no intention batch. The arm keeps the
                // match exhaustive after UpsertIntentions landed.
                StoreOp::UpsertIntentions { .. } => {}
                // Nor any code anchors.
                StoreOp::RecordAnchors { .. }
                | StoreOp::ReplaceAnchors { .. }
                | StoreOp::RecordAnchorVerdict { .. } => {}
            }
        } else if frame.kind == KIND_STORE_CHECKPOINT {
            store_checkpoints += 1;
            let cp = decode_exact::<strata_kernel::checkpoint::Checkpoint>(&frame.payload)
                .expect("checkpoint frame");
            assert_eq!(cp.magic, strata_kernel::checkpoint::MAGIC);
            assert!(
                decode_exact::<strata_migrate::EdgeRecord>(&frame.payload).is_none(),
                "checkpoint payload also round-trips as a migration edge"
            );
            assert!(matches!(
                classify_checkpoint_payload(&frame.payload),
                CheckpointPayload::Checkpoint(_)
            ));
        }
    }
    assert!(saw_upsert && saw_edge && saw_supersede && saw_review);
    assert_eq!(store_checkpoints, 1);

    let reopened = StrataStore::open(&dir).expect("replay store log");
    assert_eq!(
        reopened.get_node(&alpha).expect("alpha").content,
        "alpha record"
    );
    assert_eq!(
        reopened
            .get_node(&alpha)
            .expect("alpha")
            .superseded_by
            .as_deref(),
        Some(beta.as_str())
    );
    assert_eq!(reopened.nodes().len(), 2);
    assert_eq!(reopened.edges().len(), 1);
    assert_eq!(reopened.edges()[0].link_type, "touched");
    assert_eq!(reopened.checkpoints(), &[sealed]);
    assert_eq!(reopened.review_event_count(), reviews);
    assert_eq!(reopened.orphan_write_count(), 0);
    drop(reopened);
    std::fs::remove_dir_all(&dir).ok();

    let imported = temp_dir("kinds-import");
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../strata-migrate/tests/fixtures/v3.1.1-sample.sqlite");
    strata_migrate::migrate(&fixture, &imported.join("log")).expect("migrate");
    let frames = log_frames(&imported);
    let mut migration_nodes = 0usize;
    let mut migration_edges = 0usize;
    for frame in &frames {
        if frame.kind == KIND_STORE_WRITE {
            migration_nodes += 1;
            assert!(
                decode_exact::<strata_migrate::NodeRecord>(&frame.payload).is_some(),
                "migration node did not round-trip"
            );
            assert!(
                decode_exact::<StoreOp>(&frame.payload).is_none(),
                "migration node also round-trips as a StoreOp"
            );
            assert!(matches!(
                classify_write_payload(&frame.payload),
                WritePayload::ImportedNode(_)
            ));
        } else if frame.kind == KIND_STORE_CHECKPOINT {
            migration_edges += 1;
            assert!(
                decode_exact::<strata_migrate::EdgeRecord>(&frame.payload).is_some(),
                "migration edge did not round-trip"
            );
            let as_checkpoint =
                decode_exact::<strata_kernel::checkpoint::Checkpoint>(&frame.payload);
            assert!(
                as_checkpoint
                    .as_ref()
                    .is_none_or(|cp| cp.magic != strata_kernel::checkpoint::MAGIC),
                "migration edge also round-trips as a store checkpoint"
            );
            assert!(matches!(
                classify_checkpoint_payload(&frame.payload),
                CheckpointPayload::ImportedEdge(_)
            ));
        }
    }
    assert!(migration_nodes > 0 && migration_edges > 0);
    let store = StrataStore::open(&imported).expect("replay migration");
    assert_eq!(store.nodes().len(), migration_nodes);
    assert_eq!(store.edges().len(), migration_edges);
    assert_eq!(store.checkpoints().len(), 0);
    assert_eq!(store.orphan_write_count(), 0);
    assert!(store
        .get_node("11111111-1111-4111-8111-111111111111")
        .is_some());
    assert!(store.edges().iter().any(|edge| edge.link_type == "touched"));
    assert!(store
        .edges()
        .iter()
        .any(|edge| edge.link_type == "legacy_inferred"));
    std::fs::remove_dir_all(&imported).ok();
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

/// Classification keys off this constant. A bump has to update the guard.
#[test]
fn migration_record_version_is_1() {
    assert_eq!(strata_migrate::RECORD_VERSION, 1);
}

fn crafted_version_256_node() -> Vec<u8> {
    borsh::to_vec(&strata_migrate::NodeRecord {
        record_version: 256,
        legacy_id: String::new(),
        kernel_id: 0,
        content: String::new(),
        node_type: "\0".to_string(),
        tags: Vec::new(),
        created_ms: 0,
        updated_ms: 0,
        last_accessed_ms: 0,
        legacy: Vec::new(),
        source: None,
        source_updated_at_ms: None,
    })
    .expect("encode")
}

/// Sized so the bytes also decode exactly as a migration `NodeRecord`
/// (record_version 257), including its two trailing `Option` fields. If
/// `NodeRecord` gains fields, re-derive the lengths so this stays ambiguous.
fn crafted_version_257_edge_bytes() -> Vec<u8> {
    borsh::to_vec(&StoreOp::SaveEdge {
        edge: ConnectionRecord {
            source_id: "\0".to_string(),
            target_id: "\0".repeat(9),
            strength_milli: 0,
            link_type: "\0".repeat(8),
            meta_sha: None,
            created_at_ms: 0,
            activation_count: 0,
        },
    })
    .expect("encode")
}

fn crafted_version_0x5453_edge() -> Vec<u8> {
    let magic = strata_kernel::checkpoint::MAGIC;
    borsh::to_vec(&strata_migrate::EdgeRecord {
        record_version: u16::from_le_bytes([magic[0], magic[1]]),
        source_kernel_id: u64::from_le_bytes([
            magic[2], magic[3], magic[4], magic[5], magic[6], magic[7], 0, 0,
        ]),
        target_kernel_id: 0,
        source_legacy_id: "12345678901234567".to_string(),
        target_legacy_id: String::new(),
        link_type: String::new(),
        legacy_inferred: false,
        legacy_link_type: String::new(),
        strength_q32: 0,
        created_ms: 0,
        last_activated_ms: 0,
        activation_count: 0,
        legacy: Vec::new(),
    })
    .expect("encode")
}

fn append_payload(dir: &std::path::Path, kind: u8, payload: &[u8]) {
    let log = strata::StrataLog::open(dir.join("log")).expect("log");
    log.append(kind, payload).expect("append");
}

/// The three cross-decoding payloads are not migration records, so replay
/// does not load them as imported nodes or edges.
#[test]
fn crafted_payloads_are_not_loaded_as_migration_records() {
    let node_256 = crafted_version_256_node();
    let decoded_256 =
        decode_exact::<strata_migrate::NodeRecord>(&node_256).expect("256 decodes as a node");
    assert_eq!(decoded_256.record_version, 256);
    assert!(migration_node(&node_256).is_none());
    assert!(matches!(
        classify_write_payload(&node_256),
        WritePayload::StoreOp(_)
    ));

    let edge_257 = crafted_version_257_edge_bytes();
    let decoded_257 =
        decode_exact::<strata_migrate::NodeRecord>(&edge_257).expect("257 decodes as a node");
    assert_eq!(decoded_257.record_version, 257);
    assert!(migration_node(&edge_257).is_none());
    assert!(matches!(
        classify_write_payload(&edge_257),
        WritePayload::StoreOp(_)
    ));

    let edge_5453 = crafted_version_0x5453_edge();
    let decoded_5453 =
        decode_exact::<strata_migrate::EdgeRecord>(&edge_5453).expect("0x5453 decodes as an edge");
    assert_eq!(decoded_5453.record_version, 0x5453);
    assert!(migration_edge(&edge_5453).is_none());
    assert!(matches!(
        classify_checkpoint_payload(&edge_5453),
        CheckpointPayload::Checkpoint(_)
    ));

    let nodes = temp_dir("crafted-nodes");
    append_payload(&nodes, KIND_STORE_WRITE, &node_256);
    append_payload(&nodes, KIND_STORE_WRITE, &edge_257);
    let store = StrataStore::open(&nodes).expect("replay non-v1 nodes");
    assert!(store.nodes().is_empty());
    assert!(store.edges().is_empty());
    assert_eq!(store.orphan_write_count(), 2);
    drop(store);
    std::fs::remove_dir_all(&nodes).ok();

    let edges = temp_dir("crafted-edge");
    append_payload(&edges, KIND_STORE_CHECKPOINT, &edge_5453);
    match StrataStore::open(&edges) {
        Ok(store) => {
            assert!(store.edges().is_empty());
            assert!(store.nodes().is_empty());
        }
        Err(StoreError::Verify(_)) => {}
        Err(err) => panic!("unexpected open error: {err}"),
    }
    std::fs::remove_dir_all(&edges).ok();
}

fn review_payload(card_id: u64, rating: u8, reviewed_at_ms: Option<i64>) -> Vec<u8> {
    let mut bytes = vec![3u8];
    bytes.extend_from_slice(&card_id.to_le_bytes());
    bytes.push(rating);
    // borsh Option: tag 0 = None, tag 1 + i64 = Some. Always present.
    match reviewed_at_ms {
        None => bytes.push(0),
        Some(ms) => {
            bytes.push(1);
            bytes.extend_from_slice(&ms.to_le_bytes());
        }
    }
    bytes
}

#[test]
fn review_node_round_trip() {
    let stamped = StoreOp::ReviewNode {
        card_id: 0x1122_3344_5566_7788,
        rating: 4,
        reviewed_at_ms: Some(1_700_000_000_000),
    };
    let bytes = borsh::to_vec(&stamped).expect("encode");
    assert_eq!(
        bytes,
        review_payload(0x1122_3344_5566_7788, 4, Some(1_700_000_000_000))
    );
    assert_eq!(StoreOp::try_from_slice(&bytes).unwrap(), stamped);
    assert!(bytes.ends_with(&1_700_000_000_000i64.to_le_bytes()));

    let unset = StoreOp::ReviewNode {
        card_id: 7,
        rating: 1,
        reviewed_at_ms: None,
    };
    let bytes = borsh::to_vec(&unset).unwrap();
    assert_eq!(bytes, review_payload(7, 1, None));
    assert_eq!(bytes.last(), Some(&0));
    assert_eq!(StoreOp::try_from_slice(&bytes).unwrap(), unset);

    // Discriminant + card_id + rating, and nothing else, is not a frame.
    let short = &bytes[..10];
    assert!(StoreOp::try_from_slice(short).is_err());
}

#[test]
fn retrievability_uses_review_time_not_import_seq() {
    let dir = temp_dir("review-time");
    let mut store = StrataStore::open(&dir).expect("open");
    let id = store
        .ingest(input("reviewed years ago", &[]))
        .expect("ingest");
    let reviewed_at = 1_577_836_800_000i64; // 2020-01-01T00:00:00Z
    let as_of = reviewed_at + 400 * 86_400_000;
    store
        .review_at(&id, 3, Some(reviewed_at))
        .expect("review at past clock");

    let card = store.card_state(&id).expect("card");
    let head = store.log().head().last_acked_seq;
    let got = store
        .retrievability_at(&id, as_of)
        .expect("r")
        .expect("card");
    let from_review =
        FsrsFold::retrievability_at_review(&card, Some(reviewed_at), as_of, head, ALGO_V2)
            .expect("formula");
    let from_import =
        FsrsFold::retrievability_at_review(&card, None, as_of, head, ALGO_V2).expect("seq");
    assert_eq!(got, from_review);
    assert!(
        (head - card.last_seq) < 20,
        "import distance is a handful of frames, not 400 days"
    );
    assert!(
        got < from_import,
        "review clock decays; import seq does not: {got} vs {from_import}"
    );
    assert_eq!(store.reviewed_at_ms(&id), Some(reviewed_at));

    let frames = store.log().read_frames(1).expect("frames");
    let frame = frames
        .iter()
        .rev()
        .find(|frame| {
            matches!(
                StoreOp::try_from_slice(&frame.payload),
                Ok(StoreOp::ReviewNode {
                    reviewed_at_ms: Some(_),
                    ..
                })
            )
        })
        .expect("review frame");
    assert_eq!(
        frame.payload_blake3,
        payload_blake3(frame.kind, &frame.payload)
    );
    assert!(frame.payload.ends_with(&reviewed_at.to_le_bytes()));

    drop(store);
    let reopened = StrataStore::open(&dir).expect("reopen");
    assert_eq!(reopened.reviewed_at_ms(&id), Some(reviewed_at));
    assert_eq!(
        reopened.retrievability_at(&id, as_of).unwrap().unwrap(),
        got
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn retention_of_unreviewed_node_ignores_unrelated_writes() {
    let dir = temp_dir("retention-clock");
    let mut store = StrataStore::open(&dir).expect("open");
    let id = store
        .ingest(input("a decision written at the ingest clock", &[]))
        .expect("ingest");
    let created = store.get_node(&id).expect("node").created_at_ms;
    let day = 86_400_000i64;

    // Right after the write nothing has decayed.
    let fresh = store.retrievability_at(&id, created).unwrap().unwrap();
    assert_eq!(fresh, 1.0);

    // Unrelated writes advance the log, not the clock.
    for n in 0..40 {
        store
            .ingest(input(&format!("unrelated note {n}"), &[]))
            .expect("ingest other");
    }
    let after_writes = store.retrievability_at(&id, created).unwrap().unwrap();
    assert_eq!(
        after_writes, 1.0,
        "log writes must not decay retention: {after_writes}"
    );

    // Elapsed time does decay it, by the derived formula over whole days.
    let card = store.card_state(&id).expect("card");
    let later = created + 30 * day;
    let got = store.retrievability_at(&id, later).unwrap().unwrap();
    let expected = FsrsFold::retrievability(&card, card.last_seq + 30, ALGO_V2).expect("formula");
    assert_eq!(got, expected);
    assert!(got < 1.0, "thirty days must decay: {got}");
    let much_later = store
        .retrievability_at(&id, created + 90 * day)
        .unwrap()
        .unwrap();
    assert!(much_later < got, "monotone in elapsed time");

    // Reopen derives the same value from the log alone.
    drop(store);
    let reopened = StrataStore::open(&dir).expect("reopen");
    assert_eq!(
        reopened.retrievability_at(&id, later).unwrap().unwrap(),
        got
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn review_without_clock_replays_as_unset() {
    let dir = temp_dir("review-unset");
    let mut store = StrataStore::open(&dir).expect("open");
    let id = store.ingest(input("no clock", &[])).expect("ingest");
    store
        .review_at(&id, 2, None)
        .expect("review without a clock");
    let frames = store.log().read_frames(1).expect("frames");
    let frame = frames
        .iter()
        .rev()
        .find(|frame| frame.kind == KIND_STORE_WRITE && frame.payload.first() == Some(&3))
        .expect("review frame");
    assert_eq!(
        frame.payload,
        review_payload(handle_of(&id), 2, None).as_slice()
    );
    assert!(store.reviewed_at_ms(&id).is_none());
    drop(store);
    let reopened = StrataStore::open(&dir).expect("reopen");
    assert!(reopened.reviewed_at_ms(&id).is_none());
    std::fs::remove_dir_all(&dir).ok();
}

fn intention(id: &str, content: &str) -> crate::IntentionRecord {
    crate::IntentionRecord {
        id: id.to_string(),
        content: content.to_string(),
        trigger_type: "time".into(),
        trigger_data: r#"{"type":"time","at":"2020-01-01T00:00:00Z"}"#.into(),
        priority: 2,
        status: "active".into(),
        created_at_ms: 1_700_000_000_000,
        deadline_ms: None,
        fulfilled_at_ms: None,
        reminder_count: 0,
        last_reminded_at_ms: None,
        notes: None,
        tags: vec!["fixture".into()],
        related_memories: Vec::new(),
        snoozed_until_ms: None,
        source_type: "mcp".into(),
        source_data: None,
        scope: Some("user".into()),
    }
}

#[test]
fn intention_upsert_replays_and_rejects_an_empty_id() {
    let dir = temp_dir("intention");
    let mut store = StrataStore::open(&dir).expect("open");
    let effect = store
        .upsert_intentions(vec![intention("int-1", "Synthetic reminder")])
        .expect("admit");
    assert!(effect > 0);
    assert_eq!(store.origin_seq("int-1"), Some(effect));
    let err = store
        .upsert_intentions(vec![intention("", "no id")])
        .expect_err("empty id");
    assert!(err.to_string().contains("id must not be empty"), "{err}");
    let digest = store.state_digest();
    drop(store);

    let reopened = StrataStore::open(&dir).expect("reopen");
    let row = reopened.get_intention("int-1").expect("replayed");
    assert_eq!(row.content, "Synthetic reminder");
    assert_eq!(row.trigger_type, "time");
    assert_eq!(reopened.state_digest(), digest);
    assert_eq!(reopened.intentions().len(), 1);
    std::fs::remove_dir_all(&dir).ok();
}

/// The upgrade admits into `<dir>/log.strata-staging`, then renames it onto
/// `<dir>/log`. The plain open after the rename replays the same state and
/// proves the same per-row receipts.
#[test]
fn staged_log_admits_then_replays_after_rename() {
    let dir = temp_dir("staged-log");
    let staged = dir.join("log.strata-staging");
    let mut store =
        StrataStore::open_log_with_policy(&dir, &staged, default_policy()).expect("open staged");
    let effect = store
        .upsert_intentions(vec![intention("int-s1", "one"), intention("int-s2", "two")])
        .expect("admit");
    let digest = store.state_digest();
    drop(store);
    assert!(
        !dir.join("log").exists(),
        "a staged open must not create <dir>/log"
    );

    std::fs::rename(&staged, dir.join("log")).expect("publish");
    let reopened = StrataStore::open(&dir).expect("reopen published");
    assert_eq!(reopened.state_digest(), digest);
    assert_eq!(reopened.intentions().len(), 2);
    let proved: Vec<_> = reopened
        .prove_effects()
        .expect("prove")
        .into_iter()
        .filter(|proof| proof.action == EffectAction::Intention)
        .map(|proof| (proof.node_id, proof.effect_seq))
        .collect();
    assert_eq!(
        proved,
        vec![
            ("int-s1".to_string(), effect),
            ("int-s2".to_string(), effect)
        ]
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn upsert_intentions_op_round_trips_through_borsh_and_replay() {
    let records = vec![intention("int-a", "one"), intention("int-b", "two")];
    let op = StoreOp::UpsertIntentions {
        records: records.clone(),
    };
    let bytes = borsh::to_vec(&op).expect("encode");
    let decoded = StoreOp::try_from_slice(&bytes).expect("decode");
    assert_eq!(decoded, op);

    let dir = temp_dir("intention-roundtrip");
    let mut store = StrataStore::open(&dir).expect("open");
    let effect = store.upsert_intentions(records).expect("admit");
    let digest = store.state_digest();
    drop(store);

    let reopened = StrataStore::open(&dir).expect("reopen");
    assert_eq!(reopened.origin_seq("int-a"), Some(effect));
    assert_eq!(reopened.origin_seq("int-b"), Some(effect));
    assert_eq!(reopened.get_intention("int-a").expect("a").content, "one");
    assert_eq!(reopened.get_intention("int-b").expect("b").content, "two");
    assert_eq!(reopened.state_digest(), digest);
    let writes: Vec<_> = reopened
        .log()
        .read_frames(1)
        .expect("frames")
        .into_iter()
        .filter(|frame| frame.kind == KIND_STORE_WRITE)
        .collect();
    assert_eq!(writes.len(), 1, "one op, one data frame");
    assert_eq!(
        StoreOp::try_from_slice(&writes[0].payload).expect("payload"),
        op
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn intention_batch_is_one_admitted_write_and_a_bad_batch_writes_nothing() {
    let dir = temp_dir("intention-atomic");
    let mut store = StrataStore::open(&dir).expect("open");
    let before = store.log().head().frames_total;
    let effect = store
        .upsert_intentions(vec![intention("a", "one"), intention("b", "two")])
        .expect("admit");
    let after = store.log().head().frames_total;
    assert_eq!(after - before, 4, "propose, gate, effect, one data frame");
    assert_eq!(store.origin_seq("a"), Some(effect));
    assert_eq!(store.origin_seq("b"), Some(effect));
    let writes = store
        .log()
        .read_frames(1)
        .expect("frames")
        .into_iter()
        .filter(|frame| frame.kind == KIND_STORE_WRITE)
        .count();
    assert_eq!(writes, 1);

    let head = store.log().head().frames_total;
    let digest = store.state_digest();
    let err = store
        .upsert_intentions(vec![intention("a", "changed"), intention("a", "again")])
        .expect_err("duplicate id");
    assert!(err.to_string().contains("duplicate"), "{err}");
    assert_eq!(store.log().head().frames_total, head);
    assert_eq!(store.state_digest(), digest);
    assert_eq!(store.get_intention("a").expect("a").content, "one");
    assert_eq!(store.get_intention("b").expect("b").content, "two");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn intention_batch_proves_one_effect_per_row_across_reopen() {
    let dir = temp_dir("intention-proof");
    let mut store = StrataStore::open(&dir).expect("open");
    let effect = store
        .upsert_intentions(vec![intention("a", "one"), intention("b", "two")])
        .expect("admit");
    drop(store);

    let reopened = StrataStore::open(&dir).expect("reopen");
    let proof = reopened
        .effect_by_seq(effect)
        .expect("prove")
        .expect("intention effect is proved");
    assert_eq!(proof.action, EffectAction::Intention);
    assert_eq!(proof.node_id, "a");
    for id in ["a", "b"] {
        let latest = reopened
            .latest_effect(id)
            .expect("prove")
            .expect("row cites the batch effect");
        assert_eq!(latest.effect_seq, effect);
        assert_eq!(latest.payload_digest, proof.payload_digest);
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn recorded_origin_reads_the_creating_frame_and_supersede_chain() {
    let dir = temp_dir("origin");
    let mut store = StrataStore::open_with_policy(&dir, permissive_policy()).expect("open");
    let old = store
        .ingest_in_scope(input("older fact", &["seed"]), "user")
        .expect("old");
    let new = store
        .ingest_in_scope(input("newer fact", &[]), "user")
        .expect("new");
    store.supersede(&old, &new).expect("supersede");
    store
        .save_connection(&ConnectionRecord {
            source_id: new.clone(),
            target_id: old.clone(),
            link_type: EdgeKind::Supersedes.as_str().to_string(),
            ..ConnectionRecord::default()
        })
        .expect("edge");

    let origin = store.recorded_origin(&old).expect("read").expect("origin");
    assert_eq!(origin.record.id, old);
    assert_eq!(origin.record.content, "older fact");
    assert_eq!(origin.record.tags, vec!["seed".to_string()]);
    assert_eq!(origin.record.created_at_ms, 1_700_000_000_000);
    assert_eq!(origin.frame_kind, KIND_STORE_WRITE);
    assert_eq!(origin.effect_frame_seq.unwrap() + 1, origin.frame_seq);
    assert_eq!(origin.supersede_chain.len(), 2);
    assert_eq!(origin.supersede_chain[0].id, old);
    assert_eq!(origin.supersede_chain[0].superseded_by, new);
    assert_eq!(origin.supersede_chain[0].recorded_as, "SupersedeNode");
    assert_eq!(origin.supersede_chain[1].recorded_as, "supersedes");
    assert_eq!(origin.supersede_chain[1].id, old);
    assert_eq!(origin.supersede_chain[1].superseded_by, new);
    assert!(store
        .recorded_origin("mem-missing")
        .expect("read")
        .is_none());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn review_card_matches_independent_fsrs6_fold() {
    let dir = temp_dir("fsrs-fold");
    let mut store = StrataStore::open(&dir).expect("open");
    let id = store.ingest(input("fsrs fixture", &[])).expect("ingest");
    store.review(&id, 4).expect("easy");
    store.review(&id, 1).expect("again");
    let mut state = strata_kernel::state::State::default();
    let kernel = strata_kernel::kernel::Kernel::<strata_kernel::event::ReviewEvent>::for_version(
        strata_kernel::fsrs::ALGO_V2,
    )
    .expect("v2 kernel");
    for event in store.review_events() {
        kernel.apply(&mut state, &event);
    }
    let card = store.card_state(&id).expect("card");
    assert_eq!(card, state.cards[&handle_of(&id)]);
    assert_eq!(card.review_count, 3);
    assert_eq!(card.lapse_count, 1);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn edit_returns_successor_and_receipt_naming_edit() {
    let dir = temp_dir("edit-bytes");
    let mut store = StrataStore::open(&dir).expect("open");
    let id = store
        .ingest(input("original strata edit bytes", &["kept"]))
        .expect("ingest");
    store.review(&id, 4).expect("easy");
    store.review(&id, 1).expect("again");
    let before = store.card_state(&id).expect("card");
    let old_reviews = store.review_events().len();
    let (successor, receipt) = store
        .edit(
            &id,
            "replacement strata edit bytes",
            &AdmissionContext {
                rule_id: Some(RULE_EDIT.to_string()),
                confirm: false,
            },
        )
        .expect("edit");
    assert_ne!(successor, id);
    assert_eq!(receipt.rule_id, Some(RULE_EDIT));
    assert_eq!(receipt.receipt_id, effect_receipt_id(receipt.effect_seq));
    assert_eq!(
        store.get_node(&id).expect("old").superseded_by.as_deref(),
        Some(successor.as_str())
    );
    assert_eq!(
        store.get_node(&id).expect("old").content,
        "original strata edit bytes"
    );
    let live = store.get_all_nodes_in_scope("");
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].id, successor);
    assert_eq!(live[0].content, "replacement strata edit bytes");
    assert_eq!(live[0].tags, vec!["kept".to_string()]);
    assert_eq!(store.card_state(&id).expect("old card"), before);
    let successor_card = store.card_state(&successor).expect("successor card");
    assert_eq!(successor_card.review_count, 1);
    assert_eq!(successor_card.lapse_count, 0);
    assert_eq!(store.review_events().len(), old_reviews + 1);
    let proof = store
        .effect_by_seq(receipt.effect_seq)
        .expect("prove")
        .expect("edit effect");
    assert_eq!(proof.action, EffectAction::Edit);
    assert_eq!(proof.node_id, successor);
    drop(store);
    let mut blob = Vec::new();
    for entry in std::fs::read_dir(dir.join("log")).expect("log dir") {
        let path = entry.expect("entry").path();
        if path.is_file() {
            blob.extend(std::fs::read(&path).expect("read"));
        }
    }
    assert!(blob
        .windows(b"original strata edit bytes".len())
        .any(|w| w == b"original strata edit bytes"));
    assert!(blob
        .windows(b"replacement strata edit bytes".len())
        .any(|w| w == b"replacement strata edit bytes"));
    let store = StrataStore::open(&dir).expect("reopen");
    let again = store
        .retire_receipt(receipt.effect_seq)
        .expect("replayed receipt");
    assert_eq!(again.rule_id, Some(RULE_EDIT));
    assert_eq!(again.receipt_id, receipt.receipt_id);
    assert_eq!(
        store.get_node(&id).expect("old").content,
        "original strata edit bytes"
    );
    assert_eq!(store.card_state(&id).expect("old card"), before);
    assert_eq!(
        store
            .card_state(&successor)
            .expect("successor")
            .review_count,
        1
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn held_edit_without_rule_id_leaves_no_live_duplicate() {
    let dir = temp_dir("edit-held-dup");
    let mut store = StrataStore::open(&dir).expect("open");
    let id = store.ingest(input("stays live", &[])).expect("ingest");
    let live = store.get_all_nodes_in_scope("").len();
    let total = store.node_count();
    let err = store
        .edit(&id, "would duplicate", &AdmissionContext::default())
        .expect_err("no rule id");
    assert!(matches!(err, StoreError::Held { .. }), "{err}");
    assert_eq!(store.node_count(), total);
    assert_eq!(store.get_all_nodes_in_scope("").len(), live);
    assert!(store.supersession_pairs().is_empty());
    assert_eq!(store.get_node(&id).expect("old").content, "stays live");
    assert!(store.get_node(&id).expect("old").superseded_by.is_none());
    let err = store
        .edit(
            &id,
            "still no",
            &AdmissionContext {
                rule_id: Some("edited".into()),
                confirm: true,
            },
        )
        .expect_err("unknown rule");
    assert!(matches!(err, StoreError::Held { .. }), "{err}");
    assert_eq!(store.node_count(), total);
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
fn empty_policy_denies_review_without_changing_the_card() {
    let dir = temp_dir("deny-review");
    let id = {
        let mut store = StrataStore::open(&dir).expect("open");
        store.ingest(input("held out", &[])).expect("ingest")
    };
    let mut store =
        StrataStore::open_with_policy(&dir, Policy { rules: Vec::new() }).expect("reopen denied");
    let before = store.card_state(&id).expect("card");
    let err = store.review(&id, 4).expect_err("deny");
    assert!(matches!(err, StoreError::Denied { .. }), "{err}");
    assert_eq!(store.card_state(&id).expect("card"), before);
    assert_eq!(store.get_node(&id).expect("node").content, "held out");

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

#[test]
fn undo_appends_a_compensating_upsert_and_hides_the_create() {
    let dir = temp_dir("undo");
    let marker = "undo-marker-content";
    let frame_seq = {
        let mut store = StrataStore::open(&dir).expect("open");
        let id = store.ingest(input(marker, &["kept"])).expect("ingest");
        let other = store.ingest(input("stays visible", &[])).expect("other");
        let writes = store.node_writes();
        let create = writes
            .iter()
            .find(|write| write.record.id == id)
            .expect("create write");
        assert_eq!(create.op_type, "write");
        assert_eq!(create.status, "applied");
        let before = store.log().head().next_seq;
        let frames_before = store.log().read_frames(1).expect("frames").len();
        let undo_seq = store.undo_node_write(create.frame_seq).expect("undo");
        assert!(undo_seq > create.frame_seq);
        assert!(store.log().head().next_seq > before);
        let frames = store.log().read_frames(1).expect("frames");
        assert!(
            frames.len() > frames_before,
            "undo appends; it does not truncate"
        );
        assert!(
            frames.iter().any(|frame| {
                frame.kind == KIND_STORE_WRITE
                    && frame
                        .payload
                        .windows(marker.len())
                        .any(|window| window == marker.as_bytes())
            }),
            "the original content stays in the log"
        );
        assert!(!store.get_node(&id).expect("tombstone").is_live());
        assert!(store.get_node(&other).expect("other").is_live());
        assert_eq!(store.node_count(), 1);
        let again = store.undo_node_write(create.frame_seq);
        assert!(again.is_err(), "the create is no longer the tip");
        let undo_write = store
            .node_writes()
            .into_iter()
            .find(|write| write.frame_seq == undo_seq)
            .expect("compensating write");
        assert_eq!(undo_write.op_type, "undo");
        let undo_again = store.undo_node_write(undo_seq);
        assert!(undo_again.is_err(), "cannot undo an undo");
        store.state_digest()
    };
    let reopened = StrataStore::open(&dir).expect("reopen");
    assert_eq!(reopened.state_digest(), frame_seq);
    assert_eq!(reopened.node_count(), 1);
    assert!(reopened
        .nodes()
        .iter()
        .any(|record| record.content == marker && !record.is_live()));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn undo_restores_the_previous_upsert_without_rewriting_it() {
    let dir = temp_dir("undo-edit");
    let mut store = StrataStore::open(&dir).expect("open");
    let id = store.ingest(input("original body", &[])).expect("ingest");
    store.set_created_at(&id, 50).expect("edit");
    let writes = store.node_writes();
    let edit = writes
        .iter()
        .rev()
        .find(|write| write.record.id == id && write.op_type == "write")
        .expect("edit");
    assert_eq!(edit.record.created_at_ms, 50);
    let create_seq = writes
        .iter()
        .find(|write| write.record.id == id)
        .expect("create")
        .frame_seq;
    assert!(store.undo_node_write(create_seq).is_err());
    store.undo_node_write(edit.frame_seq).expect("undo edit");
    let restored = store.get_node(&id).expect("restored");
    assert!(restored.is_live());
    assert_eq!(restored.created_at_ms, 1_700_000_000_000);
    assert_eq!(restored.content, "original body");
    std::fs::remove_dir_all(&dir).ok();
}

fn edit_ctx() -> AdmissionContext {
    AdmissionContext {
        rule_id: Some(RULE_EDIT.to_string()),
        confirm: false,
    }
}

#[test]
fn undoing_an_edit_restores_the_previous_version_and_retires_only_the_edit() {
    let dir = temp_dir("undo-edit-chain");
    let (old_id, new_id, digest) = {
        let mut store = StrataStore::open(&dir).expect("open");
        let old_id = store.ingest(input("version one", &[])).expect("ingest");
        let (new_id, _) = store
            .edit(&old_id, "version two", &edit_ctx())
            .expect("edit");
        assert!(!store.get_node(&old_id).expect("old").is_live());
        let successor_write = store
            .node_writes()
            .into_iter()
            .find(|write| write.record.id == new_id)
            .expect("successor write");
        let undo_seq = store
            .undo_node_write(successor_write.frame_seq)
            .expect("undo edit");
        let old = store.get_node(&old_id).expect("old node");
        assert!(old.is_live(), "the previous version is live again");
        assert_eq!(old.content, "version one");
        assert!(old.superseded_by.is_none());
        assert!(
            !store.get_node(&new_id).expect("new node").is_live(),
            "the edit itself is retired"
        );
        assert_eq!(store.node_count(), 1);
        let undo_write = store
            .node_writes()
            .into_iter()
            .find(|write| write.frame_seq == undo_seq)
            .expect("undo write");
        assert_eq!(undo_write.op_type, "undo");
        assert!(
            store.undo_node_write(undo_seq).is_err(),
            "an undo is not itself undone"
        );
        let restore_write = store
            .node_writes()
            .into_iter()
            .rev()
            .find(|write| write.record.id == old_id)
            .expect("restore write");
        assert_eq!(restore_write.op_type, "undo");
        assert!(store.undo_node_write(restore_write.frame_seq).is_err());
        (old_id, new_id, store.state_digest())
    };
    let reopened = StrataStore::open(&dir).expect("reopen");
    assert_eq!(reopened.state_digest(), digest);
    assert!(reopened.get_node(&old_id).expect("old").is_live());
    assert!(!reopened.get_node(&new_id).expect("new").is_live());
    assert_eq!(reopened.node_count(), 1);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn undoing_an_edit_moves_code_anchors_back_to_the_restored_version() {
    let dir = temp_dir("undo-edit-anchors");
    let mut store = StrataStore::open(&dir).expect("open");
    let old_id = store.ingest(input("anchored one", &[])).expect("ingest");
    let anchor = anchor("anchor-undo-1", &old_id, "src/lib.rs", 10);
    store
        .record_anchors(vec![anchor.clone()])
        .expect("anchor the memory");
    let (new_id, _) = store
        .edit(&old_id, "anchored two", &edit_ctx())
        .expect("edit");
    store
        .record_anchors(vec![crate::AnchorRecord {
            node_id: new_id.clone(),
            ..anchor.clone()
        }])
        .expect("move the anchor with the edit");
    let successor_write = store
        .node_writes()
        .into_iter()
        .find(|write| write.record.id == new_id)
        .expect("successor write");
    store
        .undo_node_write(successor_write.frame_seq)
        .expect("undo edit");
    let rows = store.anchors_for(&old_id);
    assert_eq!(rows.len(), 1, "the anchor follows the restored memory");
    assert_eq!(rows[0].id, anchor.id);
    assert!(store.anchors_for(&new_id).is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

fn imported_node(id: &str, legacy: &[(&str, &str)]) -> Vec<u8> {
    borsh::to_vec(&strata_migrate::NodeRecord {
        record_version: strata_migrate::RECORD_VERSION,
        legacy_id: id.to_string(),
        kernel_id: 0,
        content: format!("imported {id}"),
        node_type: "fact".to_string(),
        tags: Vec::new(),
        created_ms: 1_700_000_000_000,
        updated_ms: 1_700_000_000_000,
        last_accessed_ms: 1_700_000_000_000,
        legacy: legacy
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect(),
        source: None,
        source_updated_at_ms: None,
    })
    .expect("encode")
}

/// A real v3 store has project scopes, suppressed memories and supersession
/// links. After the upgrade each keeps its meaning, and reviewing an imported
/// memory still proves (a review naming an imported card used to fail every
/// later proof pass).
#[test]
fn imported_scope_suppression_and_supersession_survive_replay() {
    let dir = temp_dir("imported-state");
    let project = "11111111-1111-4111-8111-aaaaaaaaaaaa";
    let suppressed = "22222222-2222-4222-8222-aaaaaaaaaaaa";
    let old = "33333333-3333-4333-8333-aaaaaaaaaaaa";
    let new = "44444444-4444-4444-8444-aaaaaaaaaaaa";
    append_payload(
        &dir,
        KIND_STORE_WRITE,
        &imported_node(
            project,
            &[
                ("knowledge_nodes.scope", "biohub"),
                ("knowledge_nodes.suppression_count", "0"),
            ],
        ),
    );
    append_payload(
        &dir,
        KIND_STORE_WRITE,
        &imported_node(
            suppressed,
            &[
                ("knowledge_nodes.scope", "user"),
                ("knowledge_nodes.suppression_count", "2"),
            ],
        ),
    );
    append_payload(&dir, KIND_STORE_WRITE, &imported_node(old, &[]));
    append_payload(&dir, KIND_STORE_WRITE, &imported_node(new, &[]));
    append_payload(
        &dir,
        strata_migrate::records::KIND_SUPERSESSION,
        &borsh::to_vec(&strata_migrate::SupersessionRecord {
            record_version: strata_migrate::RECORD_VERSION,
            superseded_legacy_id: old.to_string(),
            superseded_by_legacy_id: new.to_string(),
            superseded_kernel_id: 0,
            superseded_by_kernel_id: 0,
        })
        .expect("encode"),
    );

    let mut store = StrataStore::open(&dir).expect("replay imported state");
    let kept = store.get_node(project).expect("project node");
    assert_eq!(kept.scope, "biohub");
    assert!(kept.is_live());
    assert!(!store.get_node(suppressed).expect("suppressed").is_live());
    assert_eq!(
        store.get_node(old).expect("old").superseded_by.as_deref(),
        Some(new)
    );
    assert!(store.get_node(new).expect("new").is_live());
    assert_eq!(store.get_node(old).expect("old").scope, "user");

    store.review(project, 3).expect("review an imported memory");
    let proofs = store.prove_effects().expect("proofs still derive");
    assert!(proofs.iter().any(|proof| proof.node_id == project));
    drop(store);
    let reopened = StrataStore::open(&dir).expect("reopen");
    assert!(reopened.prove_effects().is_ok());
    assert!(!reopened.get_node(suppressed).expect("suppressed").is_live());
    std::fs::remove_dir_all(&dir).ok();
}

// ----------------------------------------------------------------------
// Code anchors
// ----------------------------------------------------------------------

fn anchor(id: &str, node_id: &str, file_path: &str, start_line: u32) -> crate::AnchorRecord {
    crate::AnchorRecord {
        id: id.to_string(),
        node_id: node_id.to_string(),
        file_path: file_path.to_string(),
        symbol: Some("load_config".into()),
        symbol_kind: Some("fn".into()),
        start_line: Some(start_line),
        end_line: Some(start_line + 3),
        span_lines: Some(4),
        content_hash: Some("v2:0123456789abcdef0123456789abcdef".into()),
        captured_at_ms: 1_700_000_000_000,
        last_verified_at_ms: None,
        last_status: None,
    }
}

fn anchor_ids(rows: &[crate::AnchorRecord]) -> Vec<&str> {
    rows.iter().map(|row| row.id.as_str()).collect()
}

#[test]
fn anchor_ops_append_discriminants_after_every_existing_op() {
    // Borsh discriminants are positional. The five pre-anchor ops keep 0..=4,
    // so every log written before anchors existed decodes to the same ops.
    let existing: [(StoreOp, u8); 2] = [
        (
            StoreOp::SupersedeNode {
                id: "a".into(),
                superseded_by: "b".into(),
            },
            2,
        ),
        (
            StoreOp::UpsertIntentions {
                records: vec![intention("i", "x")],
            },
            4,
        ),
    ];
    for (op, tag) in existing {
        assert_eq!(borsh::to_vec(&op).expect("encode")[0], tag);
    }
    let appended = [
        (
            StoreOp::RecordAnchors {
                anchors: vec![anchor("anchor-1", "mem-1", "src/a.rs", 1)],
            },
            5u8,
        ),
        (
            StoreOp::ReplaceAnchors {
                node_id: "mem-1".into(),
                anchors: vec![anchor("anchor-2", "mem-1", "src/a.rs", 1)],
            },
            6,
        ),
        (
            StoreOp::RecordAnchorVerdict {
                anchor_id: "anchor-1".into(),
                status: "drifted".into(),
                checked_at_ms: 1_700_000_000_001,
            },
            7,
        ),
    ];
    for (op, tag) in appended {
        let bytes = borsh::to_vec(&op).expect("encode");
        assert_eq!(bytes[0], tag);
        assert_eq!(decode_exact::<StoreOp>(&bytes), Some(op));
        assert!(matches!(
            classify_write_payload(&bytes),
            WritePayload::StoreOp(_)
        ));
    }
}

#[test]
fn a_log_written_before_anchors_replays_with_no_anchors() {
    let dir = temp_dir("anchor-old-log");
    let mut store = StrataStore::open(&dir).expect("open");
    let a = store.ingest(input("pre-anchor memory", &["t"])).expect("a");
    let b = store
        .ingest(input("second pre-anchor memory", &[]))
        .expect("b");
    store
        .save_connection(&ConnectionRecord {
            source_id: a.clone(),
            target_id: b.clone(),
            ..ConnectionRecord::default()
        })
        .expect("edge");
    store.review(&a, 4).expect("review");
    store
        .upsert_intentions(vec![intention("int-old", "old reminder")])
        .expect("intention");
    let digest = store.state_digest();
    drop(store);

    // Every data frame is one of the five ops that predate anchors.
    for frame in log_frames(&dir) {
        if frame.kind == KIND_STORE_WRITE {
            assert!(
                frame.payload[0] <= 4,
                "pre-anchor op tag {}",
                frame.payload[0]
            );
        }
    }
    let reopened = StrataStore::open(&dir).expect("replay the old log");
    assert_eq!(reopened.state_digest(), digest);
    assert_eq!(reopened.orphan_write_count(), 0);
    assert_eq!(reopened.node_count(), 2);
    assert_eq!(reopened.edge_count(), 1);
    assert!(reopened.get_intention("int-old").is_some());
    assert!(reopened.anchors_for(&a).is_empty());
    assert!(reopened.prove_effects().is_ok());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn anchors_record_through_one_admitted_write_and_replay() {
    let dir = temp_dir("anchor-record");
    let mut store = StrataStore::open(&dir).expect("open");
    let node = store.ingest(input("pattern memory", &[])).expect("node");
    let before = store.log().head().frames_total;
    let effect = store
        .record_anchors(vec![
            anchor("anchor-b", &node, "src/z.rs", 10),
            anchor("anchor-a", &node, "src/a.rs", 40),
            anchor("anchor-c", &node, "src/a.rs", 2),
        ])
        .expect("admit");
    assert!(effect > 0);
    assert_eq!(
        store.log().head().frames_total - before,
        4,
        "propose, gate, effect, one data frame"
    );
    // SQLite order: file path, then start line.
    let rows = store.anchors_for(&node);
    assert_eq!(anchor_ids(&rows), vec!["anchor-c", "anchor-a", "anchor-b"]);
    assert_eq!(rows[0], anchor("anchor-c", &node, "src/a.rs", 2));
    // Anchors are not memories: no card, no origin, no node.
    assert_eq!(store.review_event_count(), 1);
    assert_eq!(store.origin_seq("anchor-a"), None);
    assert_eq!(store.node_count(), 1);

    // Insert-or-replace by id, like the SQLite `INSERT OR REPLACE`.
    let mut moved = anchor("anchor-a", &node, "src/a.rs", 41);
    moved.symbol = Some("renamed".into());
    store
        .record_anchors(vec![moved.clone()])
        .expect("replace row");
    assert_eq!(store.anchor("anchor-a"), Some(moved.clone()));
    assert_eq!(store.anchors_for(&node).len(), 3);
    let digest = store.state_digest();
    drop(store);

    let reopened = StrataStore::open(&dir).expect("reopen");
    assert_eq!(reopened.state_digest(), digest);
    assert_eq!(
        anchor_ids(&reopened.anchors_for(&node)),
        vec!["anchor-c", "anchor-a", "anchor-b"]
    );
    assert_eq!(reopened.anchor("anchor-a"), Some(moved));
    assert_eq!(reopened.orphan_write_count(), 0);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_bad_anchor_batch_writes_nothing() {
    let dir = temp_dir("anchor-refuse");
    let mut store = StrataStore::open(&dir).expect("open");
    let node = store.ingest(input("pattern memory", &[])).expect("node");
    let head = store.log().head().frames_total;
    let digest = store.state_digest();

    let empty = store.record_anchors(Vec::new()).expect_err("empty batch");
    assert!(empty.to_string().contains("empty"), "{empty}");
    let dup = store
        .record_anchors(vec![
            anchor("same", &node, "src/a.rs", 1),
            anchor("same", &node, "src/b.rs", 1),
        ])
        .expect_err("duplicate id");
    assert!(dup.to_string().contains("duplicate"), "{dup}");
    let unknown = store
        .record_anchors(vec![anchor("x", "mem-ffffffffffffffff", "src/a.rs", 1)])
        .expect_err("unknown node");
    assert!(matches!(unknown, StoreError::NotFound(_)), "{unknown}");
    let blank = store
        .record_anchors(vec![anchor("y", &node, "", 1)])
        .expect_err("blank path");
    assert!(blank.to_string().contains("must not be empty"), "{blank}");
    let replace_blank = store
        .replace_anchors(&node, Vec::new())
        .expect_err("empty replacement");
    assert!(
        replace_blank.to_string().contains("empty"),
        "{replace_blank}"
    );

    assert_eq!(store.log().head().frames_total, head);
    assert_eq!(store.state_digest(), digest);
    assert!(store.anchors_for(&node).is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn replace_anchors_swaps_one_nodes_rows_and_keeps_the_memory() {
    let dir = temp_dir("anchor-replace");
    let mut store = StrataStore::open(&dir).expect("open");
    let node = store.ingest(input("pattern memory", &[])).expect("node");
    let other = store.ingest(input("other memory", &[])).expect("other");
    store
        .record_anchors(vec![
            anchor("old-1", &node, "src/a.rs", 1),
            anchor("old-2", &node, "src/b.rs", 1),
            anchor("keep", &other, "src/c.rs", 1),
        ])
        .expect("record");
    store
        .record_anchor_verdict("old-1", "drifted", 1_700_000_000_500)
        .expect("verdict");
    let node_before = store.get_node(&node).expect("node");
    let reviews = store.review_event_count();

    let head = store.log().head().frames_total;
    let wrong = store
        .replace_anchors(&node, vec![anchor("new-1", &other, "src/a.rs", 1)])
        .expect_err("row names another node");
    assert!(wrong.to_string().contains("must name"), "{wrong}");
    assert_eq!(store.log().head().frames_total, head);

    let fresh = anchor("new-1", &node, "src/a.rs", 3);
    store
        .replace_anchors(&node, vec![fresh.clone()])
        .expect("replace");
    assert_eq!(store.anchors_for(&node), vec![fresh.clone()]);
    assert_eq!(store.anchor("old-1"), None);
    assert_eq!(store.anchor("old-2"), None);
    // The other memory's anchors and the memory itself are untouched.
    assert_eq!(anchor_ids(&store.anchors_for(&other)), vec!["keep"]);
    assert_eq!(store.get_node(&node), Some(node_before.clone()));
    assert_eq!(store.review_event_count(), reviews);
    // A verdict for a replaced anchor is a no-op now.
    let head = store.log().head().frames_total;
    assert_eq!(
        store
            .record_anchor_verdict("old-1", "verified", 1_700_000_000_600)
            .expect("no-op"),
        None
    );
    assert_eq!(store.log().head().frames_total, head);
    let digest = store.state_digest();
    drop(store);

    let reopened = StrataStore::open(&dir).expect("reopen");
    assert_eq!(reopened.state_digest(), digest);
    assert_eq!(reopened.anchors_for(&node), vec![fresh]);
    assert_eq!(anchor_ids(&reopened.anchors_for(&other)), vec!["keep"]);
    assert_eq!(reopened.get_node(&node), Some(node_before));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn anchor_verdicts_update_the_row_and_the_latest_wins_across_reopen() {
    let dir = temp_dir("anchor-verdict");
    let mut store = StrataStore::open(&dir).expect("open");
    let node = store.ingest(input("pattern memory", &[])).expect("node");
    store
        .record_anchors(vec![
            anchor("a1", &node, "src/a.rs", 1),
            anchor("a2", &node, "src/b.rs", 1),
        ])
        .expect("record");

    let head = store.log().head().frames_total;
    let first = store
        .record_anchor_verdict("a1", "verified", 1_700_000_001_000)
        .expect("admit")
        .expect("known anchor");
    assert_eq!(store.log().head().frames_total - head, 4);
    let second = store
        .record_anchor_verdict("a1", "drifted", 1_700_000_002_000)
        .expect("admit")
        .expect("known anchor");
    assert!(second > first);
    let row = store.anchor("a1").expect("a1");
    assert_eq!(row.last_status.as_deref(), Some("drifted"));
    assert_eq!(row.last_verified_at_ms, Some(1_700_000_002_000));
    // The capture fields are untouched by a verdict.
    assert_eq!(
        row.content_hash,
        anchor("a1", &node, "src/a.rs", 1).content_hash
    );
    assert_eq!(store.anchor("a2").expect("a2").last_status, None);

    let head = store.log().head().frames_total;
    assert_eq!(
        store
            .record_anchor_verdict("anchor-unknown", "verified", 1)
            .expect("unknown id is a no-op"),
        None
    );
    let blank = store
        .record_anchor_verdict("a1", "", 1)
        .expect_err("blank verdict");
    assert!(blank.to_string().contains("empty"), "{blank}");
    assert_eq!(store.log().head().frames_total, head);
    let digest = store.state_digest();
    drop(store);

    let reopened = StrataStore::open(&dir).expect("reopen");
    assert_eq!(reopened.state_digest(), digest);
    let row = reopened.anchor("a1").expect("a1");
    assert_eq!(row.last_status.as_deref(), Some("drifted"));
    assert_eq!(row.last_verified_at_ms, Some(1_700_000_002_000));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn anchors_of_a_retired_node_are_not_returned_or_written() {
    let dir = temp_dir("anchor-retired");
    let mut store = StrataStore::open(&dir).expect("open");
    let doomed = store.ingest(input("retired pattern", &[])).expect("doomed");
    let kept = store.ingest(input("kept pattern", &[])).expect("kept");
    store
        .record_anchors(vec![
            anchor("doomed-1", &doomed, "src/a.rs", 1),
            anchor("kept-1", &kept, "src/a.rs", 1),
        ])
        .expect("record");
    store
        .retire(
            &doomed,
            &kept,
            &AdmissionContext {
                rule_id: Some(RULE_SUPPRESS.to_string()),
                confirm: false,
            },
        )
        .expect("retire");

    assert!(store.anchors_for(&doomed).is_empty());
    assert_eq!(store.anchor("doomed-1"), None);
    assert_eq!(anchor_ids(&store.anchors_for(&kept)), vec!["kept-1"]);

    let head = store.log().head().frames_total;
    let record = store
        .record_anchors(vec![anchor("doomed-2", &doomed, "src/b.rs", 1)])
        .expect_err("retired node");
    assert!(record.to_string().contains("retired"), "{record}");
    let replace = store
        .replace_anchors(&doomed, vec![anchor("doomed-3", &doomed, "src/b.rs", 1)])
        .expect_err("retired node");
    assert!(replace.to_string().contains("retired"), "{replace}");
    assert_eq!(
        store
            .record_anchor_verdict("doomed-1", "verified", 1)
            .expect("no-op"),
        None
    );
    assert_eq!(store.log().head().frames_total, head);
    drop(store);

    let reopened = StrataStore::open(&dir).expect("reopen");
    assert!(reopened.anchors_for(&doomed).is_empty());
    assert_eq!(reopened.anchor("doomed-1"), None);
    assert_eq!(anchor_ids(&reopened.anchors_for(&kept)), vec!["kept-1"]);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn edge_effects_prove_and_the_memory_receipt_keeps_proving() {
    let dir = temp_dir("edge-proof");
    let mut store = StrataStore::open(&dir).expect("open");
    let source = store.ingest(input("source memory", &[])).expect("source");
    let target = store.ingest(input("target memory", &[])).expect("target");
    let created = store
        .latest_effect(&source)
        .expect("prove")
        .expect("create effect");
    let saved = store
        .save_connection(&ConnectionRecord {
            source_id: source.clone(),
            target_id: target.clone(),
            link_type: EdgeKind::DerivedFrom.as_str().to_string(),
            strength_milli: 1000,
            ..ConnectionRecord::default()
        })
        .expect("edge");
    drop(store);

    let reopened = StrataStore::open(&dir).expect("reopen");
    let proof = reopened
        .effect_by_seq(saved)
        .expect("prove")
        .expect("an admitted edge proves");
    assert_eq!(proof.action, EffectAction::Edge);
    assert_eq!(proof.node_id, source);
    assert_eq!(
        proof.edge,
        Some((target.clone(), EdgeKind::DerivedFrom.as_str().to_string()))
    );
    // The memory's own receipt is still its create; edges never shadow it.
    assert_eq!(
        reopened.latest_effect(&source).expect("prove"),
        Some(created)
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn anchor_effects_prove_and_the_memory_receipt_keeps_proving() {
    let dir = temp_dir("anchor-proof");
    let mut store = StrataStore::open(&dir).expect("open");
    let node = store.ingest(input("pattern memory", &[])).expect("node");
    let created = store
        .latest_effect(&node)
        .expect("prove")
        .expect("create effect");
    assert_eq!(created.action, EffectAction::Create);
    let recorded = store
        .record_anchors(vec![
            anchor("p1", &node, "src/a.rs", 1),
            anchor("p2", &node, "src/b.rs", 1),
        ])
        .expect("record");
    let checked = store
        .record_anchor_verdict("p1", "verified", 1_700_000_003_000)
        .expect("admit")
        .expect("known");
    let replaced = store
        .replace_anchors(&node, vec![anchor("p3", &node, "src/a.rs", 5)])
        .expect("replace");
    drop(store);

    let reopened = StrataStore::open(&dir).expect("reopen");
    let proofs = reopened.prove_effects().expect("every effect still proves");
    // The memory's own receipt is still its create; anchors never shadow it.
    assert_eq!(reopened.latest_effect(&node).expect("prove"), Some(created));
    let batch: Vec<_> = proofs
        .iter()
        .filter(|proof| proof.effect_seq == recorded)
        .collect();
    assert_eq!(batch.len(), 2, "one proof per anchor row");
    assert!(batch
        .iter()
        .all(|proof| proof.action == EffectAction::Anchor && proof.rating.is_none()));
    assert_eq!(
        batch.iter().map(|p| p.node_id.as_str()).collect::<Vec<_>>(),
        vec!["p1", "p2"]
    );
    let verdict = reopened
        .effect_by_seq(checked)
        .expect("prove")
        .expect("verdict effect");
    assert_eq!(verdict.action, EffectAction::AnchorVerdict);
    assert_eq!(verdict.node_id, "p1");
    let swap = reopened
        .effect_by_seq(replaced)
        .expect("prove")
        .expect("replace effect");
    assert_eq!(swap.action, EffectAction::Anchor);
    assert_eq!(swap.node_id, "p3");
    // A fresh fold of the log lands on the live digest.
    assert_eq!(
        reopened.refold().expect("refold").state_digest,
        reopened.state_digest()
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_unadmitted_anchor_frame_is_an_orphan() {
    let dir = temp_dir("anchor-orphan");
    let mut store = StrataStore::open(&dir).expect("open");
    let node = store.ingest(input("pattern memory", &[])).expect("node");
    drop(store);
    // A data frame with no admitting EFFECT: replay must ignore it.
    let forged = borsh::to_vec(&StoreOp::RecordAnchors {
        anchors: vec![anchor("forged", &node, "src/a.rs", 1)],
    })
    .expect("encode");
    append_payload(&dir, KIND_STORE_WRITE, &forged);

    let reopened = StrataStore::open(&dir).expect("replay");
    assert_eq!(reopened.orphan_write_count(), 1);
    assert!(reopened.anchors_for(&node).is_empty());
    assert_eq!(reopened.anchor("forged"), None);
    std::fs::remove_dir_all(&dir).ok();
}

/// Remove every segment file under the store's log directory. A lookup that
/// still answers afterwards was served from the in-memory effect index, not
/// from a fresh read of the log.
fn remove_log_segments(dir: &std::path::Path) {
    for entry in std::fs::read_dir(dir.join("log")).expect("log dir") {
        let path = entry.expect("entry").path();
        if path.extension().is_some_and(|ext| ext == "seg") {
            std::fs::remove_file(&path).expect("remove segment");
        }
    }
}

#[test]
fn effect_lookups_do_not_rescan_the_log_after_writes() {
    let dir = temp_dir("effect-index-live");
    let mut store = StrataStore::open(&dir).expect("open");
    let mut ids = Vec::new();
    for n in 0..6 {
        ids.push(
            store
                .ingest(input(&format!("indexed memory {n}"), &[]))
                .expect("ingest"),
        );
    }
    store.review(&ids[0], 4).expect("review");
    let (successor, receipt) = store
        .edit(&ids[1], "indexed memory one, edited", &edit_ctx())
        .expect("edit");
    let expected = store.prove_effects().expect("full scan proves");
    assert!(expected.len() >= 8);

    remove_log_segments(&dir);

    for proof in &expected {
        let by_seq = store
            .effect_by_seq(proof.effect_seq)
            .expect("lookup must not read the log")
            .expect("indexed effect");
        let first = expected
            .iter()
            .find(|candidate| candidate.effect_seq == proof.effect_seq)
            .expect("first proof for seq");
        assert_eq!(&by_seq, first);
    }
    let latest = store
        .latest_effect(&successor)
        .expect("lookup must not read the log")
        .expect("successor has an effect");
    assert_eq!(latest.effect_seq, receipt.effect_seq);
    assert_eq!(latest.action, EffectAction::Edit);
    let reviewed = store
        .latest_effect(&ids[0])
        .expect("lookup must not read the log")
        .expect("reviewed node has an effect");
    assert_eq!(reviewed.action, EffectAction::Review);
    assert_eq!(reviewed.rating, Some(4));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn effect_index_rebuilt_on_open_matches_the_full_scan() {
    let dir = temp_dir("effect-index-reopen");
    let mut store = StrataStore::open(&dir).expect("open");
    let a = store.ingest(input("reopen memory a", &[])).expect("a");
    let b = store.ingest(input("reopen memory b", &[])).expect("b");
    store.review(&a, 3).expect("review");
    store
        .save_connection(&ConnectionRecord {
            source_id: a.clone(),
            target_id: b.clone(),
            strength_milli: 500,
            link_type: "derived_from".into(),
            meta_sha: None,
            created_at_ms: 1,
            activation_count: 0,
        })
        .expect("edge");
    store
        .upsert_intentions(vec![intention("i1", "one"), intention("i2", "two")])
        .expect("intentions");
    store
        .edit(&b, "reopen memory b, edited", &edit_ctx())
        .expect("edit");
    drop(store);

    let reopened = StrataStore::open(&dir).expect("reopen");
    let expected = reopened.prove_effects().expect("full scan proves");
    remove_log_segments(&dir);

    for proof in &expected {
        let by_seq = reopened
            .effect_by_seq(proof.effect_seq)
            .expect("indexed")
            .expect("present");
        let first = expected
            .iter()
            .find(|candidate| candidate.effect_seq == proof.effect_seq)
            .expect("first proof for seq");
        assert_eq!(&by_seq, first);
        if proof.action != EffectAction::Edge {
            let want = expected
                .iter()
                .filter(|candidate| {
                    candidate.node_id == proof.node_id && candidate.action != EffectAction::Edge
                })
                .max_by_key(|candidate| candidate.effect_seq)
                .expect("latest");
            assert_eq!(
                reopened.latest_effect(&proof.node_id).expect("indexed"),
                Some(want.clone())
            );
        }
    }
    assert_eq!(reopened.effect_by_seq(u64::MAX).expect("indexed"), None);
    assert_eq!(
        reopened.latest_effect("no-such-node").expect("indexed"),
        None
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Segment files of a store's log, in numeric order.
fn segment_files(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut segs: Vec<PathBuf> = std::fs::read_dir(dir.join("log"))
        .expect("read log dir")
        .map(|entry| entry.expect("entry").path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("seg"))
        .collect();
    segs.sort();
    segs
}

/// Flip one payload byte of the first frame in `seg`.
fn flip_first_frame(seg: &std::path::Path) {
    let mut bytes = std::fs::read(seg).expect("read segment");
    let (_frame, used) = strata::parse_frame(&bytes[strata::HEADER_WIRE_SIZE..]).expect("frame");
    assert!(used > 8);
    bytes[strata::HEADER_WIRE_SIZE + 8] ^= 0xff;
    std::fs::write(seg, &bytes).expect("write segment");
}

#[test]
fn backup_of_a_log_with_a_damaged_sealed_segment_fails() {
    let dir = temp_dir("backup-sealed-damage");
    let first = temp_dir("backup-sealed-damage-first");
    let second = temp_dir("backup-sealed-damage-second");
    std::fs::remove_dir_all(&second).ok();
    let mut store = StrataStore::open(&dir).expect("open");
    store.ingest(input("first fact", &[])).expect("first");
    store
        .backup_to(&first)
        .expect("healthy backup seals segment 0");
    store.ingest(input("second fact", &[])).expect("second");

    flip_first_frame(&segment_files(&dir)[0]);
    let err = store
        .backup_to(&second)
        .expect_err("a backup must not copy a log that fails verification");
    assert!(matches!(err, StoreError::Log(_)), "got {err}");
    assert!(
        !second.join("log").exists(),
        "a refused backup must not leave a partial copy"
    );
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&first).ok();
}

#[test]
fn backup_of_a_log_with_a_damaged_active_segment_fails() {
    let dir = temp_dir("backup-active-damage");
    let dest = temp_dir("backup-active-damage-dest");
    std::fs::remove_dir_all(&dest).ok();
    let mut store = StrataStore::open(&dir).expect("open");
    store.ingest(input("first fact", &[])).expect("first");
    store.ingest(input("second fact", &[])).expect("second");

    flip_first_frame(&segment_files(&dir)[0]);
    let before: Vec<Vec<u8>> = segment_files(&dir)
        .iter()
        .map(|seg| std::fs::read(seg).expect("read"))
        .collect();
    store
        .backup_to(&dest)
        .expect_err("a backup must not seal and copy a damaged log");
    let after: Vec<Vec<u8>> = segment_files(&dir)
        .iter()
        .map(|seg| std::fs::read(seg).expect("read"))
        .collect();
    assert_eq!(before, after, "a refused backup must not seal the segment");
    assert!(!dest.join("log").exists());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_store_that_found_damage_in_acked_history_refuses_writes() {
    let dir = temp_dir("damage-stops-writes");
    let mut store = StrataStore::open(&dir).expect("open");
    store.ingest(input("first fact", &[])).expect("first");
    store.ingest(input("second fact", &[])).expect("second");

    flip_first_frame(&segment_files(&dir)[0]);
    assert!(
        store.refold().is_err(),
        "a refold over damaged history must fail"
    );
    let err = store
        .ingest(input("written after the damage was found", &[]))
        .expect_err("a store that found acked damage must stop accepting writes");
    assert!(matches!(err, StoreError::Log(_)), "got {err}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn proving_effects_over_a_damaged_sealed_segment_fails_instead_of_hiding_later_ones() {
    let dir = temp_dir("prove-sealed-damage");
    let dest = temp_dir("prove-sealed-damage-dest");
    let mut store = StrataStore::open(&dir).expect("open");
    store.ingest(input("first fact", &[])).expect("first");
    store.backup_to(&dest).expect("seal segment 0");
    let second = store.ingest(input("second fact", &[])).expect("second");

    flip_first_frame(&segment_files(&dir)[0]);
    // Lookups are served from the effect index, proved when each effect was
    // admitted, so damage found later never hides the second fact's effect.
    // The damage itself is reported by verification, which `receipt get`
    // runs before it attests anything.
    assert!(
        store
            .latest_effect(&second)
            .expect("an index lookup reads no segment")
            .is_some(),
        "the second fact's effect must not be hidden"
    );
    store
        .log()
        .verify_log()
        .expect_err("a damaged log must not verify");
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&dest).ok();
}

/// Every file under `root`, as (path relative to `root`, bytes), sorted.
fn tree_files(root: &std::path::Path) -> Vec<(String, Vec<u8>)> {
    fn walk(root: &std::path::Path, dir: &std::path::Path, out: &mut Vec<(String, Vec<u8>)>) {
        for entry in std::fs::read_dir(dir).expect("read_dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                let rel = path.strip_prefix(root).expect("under root");
                out.push((
                    rel.to_string_lossy().into_owned(),
                    std::fs::read(&path).expect("read"),
                ));
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

/// Names in `dir` that a backup staged and did not clean up.
fn staging_leftovers(dir: &std::path::Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .expect("read_dir")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|name| name.contains(".partial"))
        .collect()
}

#[test]
fn backup_to_a_new_directory_holds_the_same_segments_and_verifies() {
    let dir = temp_dir("backup-new-src");
    let parent = temp_dir("backup-new-parent");
    let dest = parent.join("copy");
    let mut store = StrataStore::open(&dir).expect("open");
    store.ingest(input("first fact", &[])).expect("first");
    store.seal_checkpoint().expect("seal");
    store.backup_to(&dest).expect("backup");

    // Byte for byte the live log (minus its lock) plus the anchor file, and
    // nothing else: no staging directory beside or inside the backup.
    let mut want: Vec<(String, Vec<u8>)> = tree_files(&dir.join("log"))
        .into_iter()
        .filter(|(name, _)| name != "strata.lock")
        .map(|(name, bytes)| (format!("log/{name}"), bytes))
        .collect();
    want.push((
        "store.meta".to_string(),
        std::fs::read(dir.join("store.meta")).expect("anchor"),
    ));
    want.sort();
    let got: Vec<(String, Vec<u8>)> = tree_files(&dest)
        .into_iter()
        .map(|(name, bytes)| (name.replace('\\', "/"), bytes))
        .collect();
    assert_eq!(got, want);
    assert_eq!(segment_files(&dest).len(), segment_files(&dir).len());
    assert!(staging_leftovers(&parent).is_empty());

    let backup = StrataStore::open(&dest).expect("the backup opens");
    assert_eq!(backup.node_count(), 1);
    backup.verify_checkpoint_chain().expect("chain verifies");
    backup.log().verify_log().expect("log verifies");
    drop(backup);
    // The live store keeps writing after the backup.
    store.ingest(input("second fact", &[])).expect("second");
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&parent).ok();
}

#[test]
fn a_backup_whose_copy_fails_midway_leaves_nothing_under_the_final_name() {
    use crate::store::BACKUP_COPIES_BEFORE_FAILURE;

    let dir = temp_dir("backup-fail-src");
    let parent = temp_dir("backup-fail-parent");
    let mut store = StrataStore::open(&dir).expect("open");
    store.ingest(input("first fact", &[])).expect("first");
    store.seal_checkpoint().expect("seal");

    // A destination that does not exist yet: it must still not exist.
    let dest = parent.join("copy");
    BACKUP_COPIES_BEFORE_FAILURE.set(Some(2));
    let err = store.backup_to(&dest);
    BACKUP_COPIES_BEFORE_FAILURE.set(None);
    let err = err.expect_err("the third file copy fails");
    assert!(matches!(err, StoreError::Io(_)), "got {err}");
    assert!(!dest.exists(), "a failed backup left {}", dest.display());
    assert!(staging_leftovers(&parent).is_empty());

    // An empty directory that already exists: it must still be empty.
    let empty = parent.join("empty");
    std::fs::create_dir(&empty).expect("mkdir");
    BACKUP_COPIES_BEFORE_FAILURE.set(Some(2));
    let err = store.backup_to(&empty);
    BACKUP_COPIES_BEFORE_FAILURE.set(None);
    err.expect_err("the third file copy fails");
    assert_eq!(std::fs::read_dir(&empty).expect("read_dir").count(), 0);

    // The live store is unharmed and the next backup succeeds.
    store.ingest(input("second fact", &[])).expect("second");
    store.backup_to(&dest).expect("backup once copies work");
    let backup = StrataStore::open(&dest).expect("the backup opens");
    assert_eq!(backup.node_count(), 2);
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&parent).ok();
}

#[test]
fn a_second_backup_into_the_same_directory_replaces_the_first() {
    use crate::store::BACKUP_COPIES_BEFORE_FAILURE;

    let dir = temp_dir("backup-again-src");
    let dest = temp_dir("backup-again-dest");
    let mut store = StrataStore::open(&dir).expect("open");
    store.ingest(input("first fact", &[])).expect("first");
    store.seal_checkpoint().expect("seal");
    store.backup_to(&dest).expect("first backup");
    // A file the caller keeps beside log/ stays where it is.
    std::fs::write(dest.join("receipt-signing.key"), b"kept").expect("write");
    let first = tree_files(&dest);

    // A second backup that fails while copying leaves the first as it was.
    store.ingest(input("second fact", &[])).expect("second");
    store.seal_checkpoint().expect("seal");
    BACKUP_COPIES_BEFORE_FAILURE.set(Some(1));
    let err = store.backup_to(&dest);
    BACKUP_COPIES_BEFORE_FAILURE.set(None);
    err.expect_err("the second file copy fails");
    assert_eq!(tree_files(&dest), first);

    store.backup_to(&dest).expect("second backup");
    assert!(staging_leftovers(&dest).is_empty());
    assert_eq!(
        std::fs::read(dest.join("receipt-signing.key")).expect("read"),
        b"kept"
    );
    for (name, bytes) in tree_files(&dir.join("log")) {
        if name != "strata.lock" {
            let copied = std::fs::read(dest.join("log").join(&name)).expect("copied");
            assert_eq!(copied, bytes, "{name}");
        }
    }
    let backup = StrataStore::open(&dest).expect("the backup opens");
    assert_eq!(backup.node_count(), 2);
    backup.verify_checkpoint_chain().expect("chain verifies");
    backup.log().verify_log().expect("log verifies");
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&dest).ok();
}
