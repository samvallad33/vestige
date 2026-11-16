//! #399 store properties over the public [`strata_store::StrataStore`] API.
//!
//! Invariant map (the standalone `docs/INVARIANTS.md` is outside this lane):
//!
//! | Invariant | Test | Recorded structure |
//! |---|---|---|
//! | replay | `prop_replay_matches_refold` | `state_digest`, `refold` |
//! | admission | `prop_every_store_write_is_admitted` | EFFECT `payload_digest`, `sweep`, `rederive_verdicts` |
//! | reads-append-nothing | `prop_reads_and_refusals_append_nothing` | `log().head().frames_total` |
//! | log-chain | `prop_replay_matches_refold` (seq + `head.state`) | `read_frames`, `read_head_state` |
//! | checkpoint-chain | `prop_checkpoint_chain_accepts_honest_and_refuses_one_byte` | `verify_checkpoint_chain`, `store.meta`, checkpoint frame |
//! | edge-vocabulary | `prop_edge_vocabulary_is_closed` | `EdgeKind::parse`, `edges()` |
//! | supersession | `prop_supersession_chains_end_live_in_scope` (ignored, red) | `supersession_pairs`, node scope |
//! | identity | `prop_exact_duplicate_reinforces` (ignored, red) | canonical hash, node ids |
//!
//! Claim-gate (#301), claim-slots (#302) and capsule (#395) have no public
//! API on this tree, so those suites are not pretended. The two red
//! properties are the #399 falsifiers. Their minimal seeds are committed as
//! `regression_seed_*` tests that stay green while the bug is present.
//! `STRATA_PROP_CASES` raises the pinned count. No `proptest` dev-dependency:
//! this lane adds test files only.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use borsh::BorshDeserialize;
use strata::read_head_state;
use strata_gate::record::{EffectRecord, GateRecord, RecordKind, Verdict};
use strata_store::{
    canonical_hash, default_policy, permissive_policy, AdmissionContext, AnchorRecord,
    ConnectionRecord, EdgeDirection, EdgeKind, IngestInput, IntentionRecord, StoreError, StoreOp,
    StrataStore, KIND_STORE_WRITE, RULE_SUPPRESS, TYPED_EDGE_VOCABULARY,
};

fn cases() -> u64 {
    std::env::var("STRATA_PROP_CASES")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .filter(|n: &u64| *n > 0)
        .unwrap_or(4)
}

struct Tmp {
    path: PathBuf,
}

impl Tmp {
    fn new(tag: &str) -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "strata-store-prop-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("temp dir");
        Self { path }
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn input(content: &str) -> IngestInput {
    IngestInput {
        content: content.to_string(),
        created_at_ms: Some(1_700_000_000_000),
        ..IngestInput::default()
    }
}

fn open(dir: &Path, permissive: bool) -> StrataStore {
    let policy = if permissive {
        permissive_policy()
    } else {
        default_policy()
    };
    StrataStore::open_with_policy(dir, policy).expect("open")
}

fn link(store: &mut StrataStore, from: &str, to: &str, permissive: bool) -> Result<(), StoreError> {
    if permissive {
        store.supersede(from, to)
    } else {
        store
            .retire(
                from,
                to,
                &AdmissionContext {
                    rule_id: Some(RULE_SUPPRESS.to_string()),
                    confirm: false,
                },
            )
            .map(|_| ())
    }
}

fn intention(id: &str, content: &str) -> IntentionRecord {
    IntentionRecord {
        id: id.to_string(),
        content: content.to_string(),
        trigger_type: "manual".to_string(),
        trigger_data: "{}".to_string(),
        priority: 2,
        status: "active".to_string(),
        created_at_ms: 0,
        deadline_ms: None,
        fulfilled_at_ms: None,
        reminder_count: 0,
        last_reminded_at_ms: None,
        notes: None,
        tags: Vec::new(),
        related_memories: Vec::new(),
        snoozed_until_ms: None,
        source_type: "mcp".to_string(),
        source_data: None,
        scope: Some("proj".to_string()),
    }
}

fn anchor(id: &str, node_id: &str) -> AnchorRecord {
    AnchorRecord {
        id: id.to_string(),
        node_id: node_id.to_string(),
        file_path: "src/lib.rs".to_string(),
        symbol: Some("open".to_string()),
        symbol_kind: Some("fn".to_string()),
        start_line: Some(1),
        end_line: Some(4),
        span_lines: Some(4),
        content_hash: Some("abc".to_string()),
        captured_at_ms: 0,
        last_verified_at_ms: None,
        last_status: None,
    }
}

fn frames_total(store: &StrataStore) -> u64 {
    store.log().head().frames_total
}

fn assert_unchanged(store: &StrataStore, before: u64, what: &str) {
    assert_eq!(frames_total(store), before, "{what} appended a frame");
}

fn violation_labels(store: &StrataStore) -> Vec<String> {
    let nodes = store.nodes();
    let by_id: BTreeMap<&str, _> = nodes.iter().map(|node| (node.id.as_str(), node)).collect();
    let mut labels = Vec::new();
    for node in &nodes {
        if node.superseded_by.is_none() {
            continue;
        }
        let mut cur = node.id.as_str();
        let mut seen = BTreeSet::new();
        let origin = node.scope.as_str();
        loop {
            if !seen.insert(cur) {
                labels.push(format!("cycle {}", node.id));
                break;
            }
            let Some(record) = by_id.get(cur) else {
                labels.push(format!("dead-end {} missing {cur}", node.id));
                break;
            };
            if record.scope != origin {
                labels.push(format!(
                    "cross-scope {} ({origin} -> {})",
                    node.id, record.scope
                ));
            }
            match record.superseded_by.as_deref() {
                None => break,
                Some(next) => cur = next,
            }
        }
    }
    labels.sort();
    labels.dedup();
    labels
}

fn known_supersession_bug(label: &str) -> bool {
    label.starts_with("cycle ") || label.starts_with("cross-scope ")
}

fn assert_store_log_chain(store: &StrataStore, dir: &Path) {
    let frames = store.log().read_frames(1).expect("frames");
    let mut prev = 0u64;
    for (i, frame) in frames.iter().enumerate() {
        assert!(frame.seq > prev, "seq did not increase at {i}");
        assert_eq!(frame.seq, i as u64 + 1, "seq gap at {i}");
        prev = frame.seq;
        assert_eq!(
            frame.payload_blake3,
            strata::payload_blake3(frame.kind, &frame.payload)
        );
    }
    let watermark = read_head_state(&dir.join("log")).expect("head.state");
    if frames.is_empty() {
        assert!(watermark.is_none() || watermark == Some(0));
    } else {
        assert_eq!(watermark, Some(frames.last().expect("last").seq));
        assert_eq!(
            store.log().head().last_acked_seq,
            frames.last().expect("last").seq
        );
    }
}

fn assert_admitted(store: &StrataStore) {
    let frames = store.log().read_frames(1).expect("frames");
    let mut admitted = BTreeSet::new();
    for frame in &frames {
        if frame.kind == RecordKind::Effect.to_u8() {
            let effect = EffectRecord::try_from_slice(&frame.payload).expect("effect");
            admitted.insert(effect.payload_digest);
        }
    }
    assert!(
        writes_are_admitted(&frames, &admitted),
        "a STORE_WRITE has no admitting effect"
    );
    for edge in store.edges() {
        assert!(
            EdgeKind::parse(&edge.link_type).is_some(),
            "persisted link_type {} is outside the vocabulary",
            edge.link_type
        );
        assert!(TYPED_EDGE_VOCABULARY.contains(&edge.link_type.as_str()));
    }
    assert!(store.sweep().is_empty(), "sweep {:?}", store.sweep());
    let derived = store.rederive_verdicts().expect("rederive");
    assert_eq!(derived, recorded_verdicts(&frames));
}

fn writes_are_admitted(frames: &[strata::FrameRecord], admitted: &BTreeSet<[u8; 32]>) -> bool {
    let mut seen = BTreeSet::new();
    for frame in frames {
        if frame.kind == RecordKind::Effect.to_u8() {
            if let Ok(effect) = EffectRecord::try_from_slice(&frame.payload) {
                seen.insert(effect.payload_digest);
            }
        } else if frame.kind == KIND_STORE_WRITE {
            let digest = *blake3::hash(&frame.payload).as_bytes();
            if !seen.contains(&digest) || !admitted.contains(&digest) {
                return false;
            }
            if StoreOp::try_from_slice(&frame.payload).is_err() {
                return false;
            }
        }
    }
    true
}

/// Gate-space seqs are dense over kinds 1..=7 only. Store frames sit in the
/// log between them and are invisible to `rederive_verdicts`, so the seq in
/// that result is the index of the gate frame among gate frames, not the log seq.
fn recorded_verdicts(frames: &[strata::FrameRecord]) -> Vec<(u64, Verdict)> {
    let mut gate_seq = 0u64;
    let mut recorded = Vec::new();
    for frame in frames {
        if RecordKind::from_u8(frame.kind).is_none() {
            continue;
        }
        if frame.kind == RecordKind::Gate.to_u8() {
            let gate = GateRecord::try_from_slice(&frame.payload).expect("gate");
            recorded.push((gate_seq, gate.verdict));
        }
        gate_seq += 1;
    }
    recorded
}

fn live_duplicate_groups(store: &StrataStore) -> Vec<(String, Vec<String>)> {
    let mut groups: BTreeMap<(String, [u8; 32]), Vec<String>> = BTreeMap::new();
    for node in store.nodes() {
        if !node.is_live() {
            continue;
        }
        groups
            .entry((node.scope.clone(), canonical_hash(&node.content)))
            .or_default()
            .push(node.id);
    }
    groups
        .into_iter()
        .filter(|(_, ids)| ids.len() > 1)
        .map(|((scope, _), ids)| (scope, ids))
        .collect()
}

fn read_surface(store: &StrataStore) {
    let head = frames_total(store);
    let digest = store.state_digest();
    let _ = store.nodes();
    let _ = store.edges();
    let _ = store.node_count();
    let _ = store.edge_count();
    let _ = store.supersession_pairs();
    let _ = store.intentions();
    let _ = store.checkpoints();
    let _ = store.origins();
    let _ = store.orphan_write_count();
    let _ = store.review_event_count();
    let _ = store.head_clock_ms();
    let _ = store.policy();
    let _ = store.sweep();
    let _ = store.rederive_verdicts();
    let _ = store.verify_checkpoint_chain();
    let _ = store.log().head();
    let _ = store.log().read_frames(1);
    let _ = store.log().verify_tail();
    let _ = store.log().verify_log();
    let _ = store.get_all_nodes_in_scope("proj");
    let _ = store.get_all_nodes_in_scope("");
    let _ = store.get_never_composed("proj", 8);
    let _ = store.find_intent("proj", "missing");
    for node in store.nodes() {
        let _ = store.get_node(&node.id);
        let _ = store.get_connections_for_memory(&node.id);
        let _ = store.get_edges_for(&node.id, EdgeDirection::Both, None);
        let _ = store.get_edges_for(&node.id, EdgeDirection::Outgoing, Some(EdgeKind::Touched));
        let _ = store.retrievability(&node.id);
        let _ = store.retrievability_at(&node.id, 1_800_000_000_000);
        let _ = store.is_failure_memory(&node.id);
        let _ = store.card_state(&node.id);
        let _ = store.reviewed_at_ms(&node.id);
        let _ = store.anchors_for(&node.id);
        let _ = store.get_intention("nope");
        let _ = store.find_node_by_canonical_hash(&node.scope, &canonical_hash(&node.content));
        let _ = store.retire_receipt(1);
    }
    let folded = store.refold().expect("refold");
    assert_eq!(folded.state_digest, digest, "refold digest");
    assert!(
        folded.gate_mismatches.is_empty(),
        "{:?}",
        folded.gate_mismatches
    );
    assert!(folded.gaps.is_empty(), "{:?}", folded.gaps);
    assert_eq!(store.state_digest(), digest);
    assert_eq!(frames_total(store), head, "a read appended a frame");
}

fn exercise(permissive: bool, seed: u64) {
    let tmp = Tmp::new("green");
    let mut store = open(&tmp.path, permissive);
    let policy = if permissive { "permissive" } else { "default" };
    let a = store
        .ingest_in_scope(input(&format!("alpha-{seed}")), "proj")
        .expect("a");
    let b = store
        .ingest_in_scope(input(&format!("beta-{seed}")), "proj")
        .expect("b");
    let _other = store
        .ingest_in_scope(input(&format!("other-{seed}")), "other")
        .expect("other scope");

    let kind = EdgeKind::ALL[(seed as usize) % EdgeKind::ALL.len()];
    let before = frames_total(&store);
    store
        .save_connection(&ConnectionRecord {
            source_id: a.clone(),
            target_id: b.clone(),
            link_type: kind.as_str().to_string(),
            ..ConnectionRecord::default()
        })
        .expect("vocab edge");
    assert!(frames_total(&store) > before);

    let before = frames_total(&store);
    let refused = store.save_connection(&ConnectionRecord {
        source_id: a.clone(),
        target_id: b.clone(),
        link_type: format!("semantic-{seed}"),
        ..ConnectionRecord::default()
    });
    assert!(matches!(refused, Err(StoreError::InvalidInput(_))));
    assert_unchanged(&store, before, "non-vocabulary edge");

    let before = frames_total(&store);
    let negative = store.save_connection(&ConnectionRecord {
        source_id: a.clone(),
        target_id: "src/main.rs".to_string(),
        link_type: EdgeKind::AnchoredTo.as_str().to_string(),
        strength_milli: -1,
        ..ConnectionRecord::default()
    });
    assert!(matches!(negative, Err(StoreError::InvalidInput(_))));
    assert_unchanged(&store, before, "negative strength");

    let before = frames_total(&store);
    assert!(matches!(
        store.ingest(input("   ")),
        Err(StoreError::InvalidInput(_))
    ));
    assert_unchanged(&store, before, "blank content");

    let before = frames_total(&store);
    assert!(matches!(
        store.upsert_intentions(Vec::new()),
        Err(StoreError::InvalidInput(_))
    ));
    assert_unchanged(&store, before, "empty intention batch");

    let rating = 1 + (seed % 4) as u8;
    store
        .review_at(&a, rating, Some(1_700_000_000_000 + seed as i64))
        .expect("review");
    store.review_at(&b, 4, None).expect("review without clock");
    let before = frames_total(&store);
    assert!(matches!(
        store.review(&a, 0),
        Err(StoreError::InvalidInput(_))
    ));
    assert!(matches!(
        store.review(&a, 9),
        Err(StoreError::InvalidInput(_))
    ));
    assert!(matches!(
        store.review("mem-missing", 3),
        Err(StoreError::NotFound(_))
    ));
    assert_unchanged(&store, before, "bad review");

    store
        .upsert_intentions(vec![intention(
            &format!("intent-{seed}"),
            "ship the property",
        )])
        .expect("intention");
    store
        .record_anchors(vec![anchor(&format!("anc-{seed}"), &a)])
        .expect("anchor");

    if !permissive {
        let before_pairs = store.supersession_pairs();
        let before = frames_total(&store);
        let held = store.supersede(&a, &b).expect_err("bare supersede is held");
        assert!(matches!(held, StoreError::Held { .. }), "{policy}: {held}");
        // Hold writes PROPOSE + GATE and must not write the data frame's effect
        // as a landed supersession. frames_total moves; pairs do not.
        assert!(
            frames_total(&store) > before,
            "hold records the gate decision"
        );
        assert_eq!(store.supersession_pairs(), before_pairs);
        assert!(store.get_node(&a).expect("a").is_live());
    }

    link(&mut store, &a, &b, permissive).unwrap_or_else(|err| panic!("{policy} retire: {err}"));
    assert_eq!(store.supersession_pairs(), vec![(a.clone(), b.clone())]);
    assert!(violation_labels(&store).is_empty(), "{policy}");

    let before = frames_total(&store);
    let again = link(&mut store, &a, &b, permissive);
    assert!(
        matches!(again, Err(StoreError::InvalidInput(_))),
        "{again:?}"
    );
    assert_unchanged(&store, before, "second supersede");
    let before = frames_total(&store);
    let itself = link(&mut store, &b, &b, permissive);
    assert!(
        matches!(itself, Err(StoreError::InvalidInput(_))),
        "{itself:?}"
    );
    assert_unchanged(&store, before, "self supersede");
    let before = frames_total(&store);
    let anchored = store.record_anchors(vec![anchor("anc-retired", &a)]);
    assert!(matches!(anchored, Err(StoreError::InvalidInput(_))));
    assert_unchanged(&store, before, "anchor on a retired node");

    store.seal_checkpoint().expect("seal");
    store.verify_checkpoint_chain().expect("chain");
    assert_store_log_chain(&store, &tmp.path);
    assert_admitted(&store);
    assert!(live_duplicate_groups(&store).is_empty());
    let digest = store.state_digest();
    read_surface(&store);

    if seed % 2 == 0 {
        let dest = tmp.path.join("backup");
        store.backup_to(&dest).expect("backup");
        assert_eq!(store.state_digest(), digest, "backup changed live state");
        let backed = StrataStore::open(&dest).expect("open backup");
        assert_eq!(backed.state_digest(), digest);
        backed.verify_checkpoint_chain().expect("backup chain");
    }

    drop(store);
    let reopened = open(&tmp.path, permissive);
    assert_eq!(reopened.state_digest(), digest, "{policy} seed {seed}");
    let folded = reopened.refold().expect("refold");
    assert_eq!(folded.state_digest, digest);
    assert!(folded.gate_mismatches.is_empty());
    assert!(folded.gaps.is_empty());
    assert_admitted(&reopened);
    assert!(violation_labels(&reopened).is_empty());
    assert_store_log_chain(&reopened, &tmp.path);
}

#[test]
fn prop_replay_admission_reads_edges_checkpoints_and_one_way_supersession() {
    for seed in 1..=cases() {
        for permissive in [false, true] {
            exercise(permissive, seed);
        }
    }
}

#[test]
fn prop_checkpoint_chain_accepts_honest_and_refuses_one_byte() {
    let tmp = Tmp::new("checkpoint");
    let digest = {
        let mut store = open(&tmp.path, false);
        let id = store.ingest(input("checkpointed fact")).expect("ingest");
        store.review_at(&id, 3, Some(10)).expect("review");
        store.seal_checkpoint().expect("seal 1");
        store.review_at(&id, 1, Some(20)).expect("review");
        store.seal_checkpoint().expect("seal 2");
        store.verify_checkpoint_chain().expect("honest chain");
        let digest = store.state_digest();
        drop(store);
        let again = open(&tmp.path, false);
        assert_eq!(again.state_digest(), digest);
        again.verify_checkpoint_chain().expect("reopen chain");
        digest
    };

    let meta_path = tmp.path.join("store.meta");
    let meta = std::fs::read(&meta_path).expect("store.meta");
    let mut flipped = meta.clone();
    let meta_index = flipped.len() - 1;
    flipped[meta_index] ^= 0xff;
    std::fs::write(&meta_path, &flipped).expect("flip meta");
    assert!(
        StrataStore::open(&tmp.path).is_err(),
        "one-byte store.meta edit must refuse open"
    );
    std::fs::write(&meta_path, &meta).expect("restore meta");
    assert_eq!(open(&tmp.path, false).state_digest(), digest);

    let seg = only_segment(&tmp.path.join("log"));
    let original = std::fs::read(&seg).expect("segment");
    let mut corrupted = original.clone();
    let offset = checkpoint_payload_byte(&original);
    corrupted[offset] ^= 0xff;
    std::fs::write(&seg, &corrupted).expect("flip checkpoint frame");
    match StrataStore::open(&tmp.path) {
        Err(StoreError::Log(strata::StrataError::Halt(_))) | Err(StoreError::Verify(_)) => {}
        Err(err) => panic!("one-byte checkpoint frame must refuse open, got {err}"),
        Ok(_) => panic!("one-byte checkpoint frame must refuse open"),
    }
    assert_eq!(
        std::fs::metadata(&seg).expect("stat").len(),
        original.len() as u64,
        "refusing a checkpoint edit must not truncate the segment"
    );
}

fn only_segment(dir: &Path) -> PathBuf {
    let mut segs = Vec::new();
    for entry in std::fs::read_dir(dir).expect("log dir") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("seg") {
            segs.push(path);
        }
    }
    segs.sort();
    assert_eq!(segs.len(), 1, "one segment");
    segs.remove(0)
}

fn checkpoint_payload_byte(bytes: &[u8]) -> usize {
    let mut off = strata::HEADER_WIRE_SIZE;
    while off < bytes.len() {
        let (frame, used) = strata::parse_frame(&bytes[off..]).expect("frame");
        if frame.kind == strata_store::KIND_STORE_CHECKPOINT && !frame.payload.is_empty() {
            return off + 5;
        }
        off += used;
    }
    panic!("no checkpoint frame");
}

#[test]
fn prop_edge_vocabulary_is_closed() {
    let tmp = Tmp::new("edges");
    let mut store = open(&tmp.path, false);
    let source = store.ingest(input("edge source")).expect("ingest");
    for kind in EdgeKind::ALL {
        let before = frames_total(&store);
        store
            .save_connection(&ConnectionRecord {
                source_id: source.clone(),
                target_id: format!("target-{}", kind.as_str()),
                link_type: kind.as_str().to_string(),
                ..ConnectionRecord::default()
            })
            .unwrap_or_else(|err| panic!("{}: {err}", kind.as_str()));
        assert!(frames_total(&store) > before);
        assert_eq!(EdgeKind::parse(kind.as_str()), Some(kind));
    }
    for foreign in ["semantic", "related", "caused_by", "touches", "Touched", ""] {
        let before = frames_total(&store);
        let err = store.save_connection(&ConnectionRecord {
            source_id: source.clone(),
            target_id: "x".to_string(),
            link_type: foreign.to_string(),
            ..ConnectionRecord::default()
        });
        assert!(
            matches!(err, Err(StoreError::InvalidInput(_))),
            "{foreign}: {err:?}"
        );
        assert_unchanged(&store, before, foreign);
        assert!(EdgeKind::parse(foreign).is_none());
    }
    assert_eq!(store.edges().len(), EdgeKind::ALL.len());
}

#[test]
fn prop_distinct_content_and_distinct_scopes_do_not_merge() {
    let tmp = Tmp::new("distinct");
    let mut store = open(&tmp.path, false);
    let left = store.ingest_in_scope(input("left"), "proj").expect("left");
    let right = store
        .ingest_in_scope(input("right"), "proj")
        .expect("right");
    let other = store
        .ingest_in_scope(input("left"), "other")
        .expect("other");
    assert_ne!(left, right);
    assert_ne!(left, other);
    assert_eq!(store.node_count(), 3);
    assert!(live_duplicate_groups(&store).is_empty());
    assert_ne!(canonical_hash("left"), canonical_hash("right"));
    assert_eq!(canonical_hash("left"), canonical_hash(" left\n"));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Ingest(&'static str, &'static str),
    Supersede(&'static str, &'static str, &'static str, &'static str),
}

fn labels_of(steps: &[Step], permissive: bool) -> Vec<String> {
    let tmp = Tmp::new("steps");
    let mut store = open(&tmp.path, permissive);
    let mut ids: BTreeMap<(&str, &str), String> = BTreeMap::new();
    for step in steps {
        match *step {
            Step::Ingest(content, scope) => {
                let id = store
                    .ingest_in_scope(input(content), scope)
                    .expect("ingest step");
                ids.insert((scope, content), id);
            }
            Step::Supersede(from_scope, from, to_scope, to) => {
                let from_id = ids.get(&(from_scope, from)).cloned();
                let to_id = ids.get(&(to_scope, to)).cloned();
                if let (Some(from_id), Some(to_id)) = (from_id, to_id) {
                    let _ = link(&mut store, &from_id, &to_id, permissive);
                }
            }
        }
    }
    violation_labels(&store)
}

fn shrink(steps: &[Step], permissive: bool, still_bad: impl Fn(&[String]) -> bool) -> Vec<Step> {
    let mut current = steps.to_vec();
    assert!(
        still_bad(&labels_of(&current, permissive)),
        "seed was not a failing history: {current:?}"
    );
    let mut index = 0;
    while index < current.len() {
        let mut trial = current.clone();
        trial.remove(index);
        if still_bad(&labels_of(&trial, permissive)) {
            current = trial;
        } else {
            index += 1;
        }
    }
    current
}

fn cycle_steps() -> Vec<Step> {
    vec![
        Step::Ingest("noise", "s"),
        Step::Ingest("alpha", "s"),
        Step::Ingest("extra", "other"),
        Step::Ingest("beta", "s"),
        Step::Supersede("s", "alpha", "s", "beta"),
        Step::Supersede("s", "beta", "s", "alpha"),
    ]
}

fn minimal_cycle() -> Vec<Step> {
    vec![
        Step::Ingest("alpha", "s"),
        Step::Ingest("beta", "s"),
        Step::Supersede("s", "alpha", "s", "beta"),
        Step::Supersede("s", "beta", "s", "alpha"),
    ]
}

fn cross_steps() -> Vec<Step> {
    vec![
        Step::Ingest("noise", "s0"),
        Step::Ingest("alpha", "s0"),
        Step::Ingest("beta", "s1"),
        Step::Supersede("s0", "alpha", "s1", "beta"),
    ]
}

fn minimal_cross() -> Vec<Step> {
    vec![
        Step::Ingest("alpha", "s0"),
        Step::Ingest("beta", "s1"),
        Step::Supersede("s0", "alpha", "s1", "beta"),
    ]
}

fn both_supersedes_ok(steps: &[Step], permissive: bool) -> bool {
    let tmp = Tmp::new("ok");
    let mut store = open(&tmp.path, permissive);
    let mut ids: BTreeMap<(&str, &str), String> = BTreeMap::new();
    let mut oks = Vec::new();
    for step in steps {
        match *step {
            Step::Ingest(content, scope) => {
                let id = store
                    .ingest_in_scope(input(content), scope)
                    .expect("ingest");
                ids.insert((scope, content), id);
            }
            Step::Supersede(from_scope, from, to_scope, to) => {
                let from_id = ids.get(&(from_scope, from)).cloned().expect("from");
                let to_id = ids.get(&(to_scope, to)).cloned().expect("to");
                oks.push(link(&mut store, &from_id, &to_id, permissive).is_ok());
            }
        }
    }
    oks.len() == 2 && oks.iter().all(|ok| *ok) || oks.len() == 1 && oks[0]
}

#[test]
fn regression_seed_supersession_cycle_and_cross_scope() {
    let shrunk = shrink(&cycle_steps(), true, |labels| {
        labels.iter().any(|label| label.starts_with("cycle "))
    });
    assert_eq!(
        shrunk,
        minimal_cycle(),
        "cycle falsifier did not shrink to ingest a; ingest b; supersede(a,b); supersede(b,a)"
    );
    let cross = shrink(&cross_steps(), true, |labels| {
        labels.iter().any(|label| label.starts_with("cross-scope "))
    });
    assert_eq!(cross, minimal_cross());

    for permissive in [true, false] {
        assert!(
            both_supersedes_ok(&minimal_cycle(), permissive),
            "cycle calls were refused under permissive={permissive}"
        );
        let labels = labels_of(&minimal_cycle(), permissive);
        assert!(
            labels.iter().any(|label| label.starts_with("cycle ")),
            "permissive={permissive} cycle labels {labels:?}"
        );
        assert!(
            both_supersedes_ok(&minimal_cross(), permissive),
            "cross-scope supersede was refused under permissive={permissive}"
        );
        let labels = labels_of(&minimal_cross(), permissive);
        assert!(
            labels.iter().any(|label| label.starts_with("cross-scope ")),
            "permissive={permissive} cross-scope labels {labels:?}"
        );
        assert!(labels.iter().all(|label| known_supersession_bug(label)));
    }
}

#[test]
fn prop_random_supersession_bugs_stay_in_the_known_classes() {
    let mut rng = Rng::new(0x399);
    for seed in 1..=cases() {
        for permissive in [false, true] {
            let mut steps = Vec::new();
            let contents = ["c0", "c1", "c2", "c3"];
            let scopes = ["s0", "s1"];
            for n in 0..6 {
                if rng.gen_range(2) == 0 {
                    steps.push(Step::Ingest(
                        contents[(seed as usize + n) % contents.len()],
                        scopes[n % scopes.len()],
                    ));
                } else {
                    let from = contents[rng.gen_range(contents.len())];
                    let to = contents[rng.gen_range(contents.len())];
                    steps.push(Step::Supersede(
                        scopes[rng.gen_range(scopes.len())],
                        from,
                        scopes[rng.gen_range(scopes.len())],
                        to,
                    ));
                }
            }
            let labels = labels_of(&steps, permissive);
            for label in &labels {
                assert!(
                    known_supersession_bug(label),
                    "seed {seed} permissive={permissive} new bug {label} after {steps:?}"
                );
            }
        }
    }
}

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
    fn gen_range(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

#[test]
fn regression_seed_exact_duplicate_creates_two_nodes() {
    assert_eq!(canonical_hash("alpha"), canonical_hash(" alpha\n"));
    for (first, second) in [("alpha", "alpha"), ("alpha", " alpha\n")] {
        let tmp = Tmp::new("dup");
        let mut store = open(&tmp.path, false);
        let id1 = store.ingest_in_scope(input(first), "proj").expect("first");
        let id2 = store
            .ingest_in_scope(input(second), "proj")
            .expect("second");
        let groups = live_duplicate_groups(&store);
        assert_ne!(id1, id2, "minimal repro {first:?} / {second:?} merged");
        assert_eq!(store.node_count(), 2, "{first:?} / {second:?}");
        assert_eq!(groups.len(), 1, "{groups:?}");
        assert_eq!(groups[0].1.len(), 2);
        let upserts = store
            .log()
            .read_frames(1)
            .expect("frames")
            .into_iter()
            .filter(|frame| frame.kind == KIND_STORE_WRITE)
            .filter(|frame| {
                matches!(
                    StoreOp::try_from_slice(&frame.payload),
                    Ok(StoreOp::UpsertNode { .. })
                )
            })
            .count();
        assert_eq!(upserts, 2, "two admission data frames for one text");
    }
}

#[test]
#[ignore = "known bug #399: ingest a; ingest b; supersede(a,b); supersede(b,a) under permissive_policy and under default+suppress both return Ok and supersession_pairs holds both directions. A supersede whose successor is in another scope also lands. Goes green with the #300 port."]
fn prop_supersession_chains_end_live_in_scope() {
    for permissive in [false, true] {
        let labels = labels_of(&minimal_cycle(), permissive);
        assert!(
            labels.is_empty(),
            "permissive={permissive} cycle still admitted: {labels:?} steps={:?}",
            minimal_cycle()
        );
        let labels = labels_of(&minimal_cross(), permissive);
        assert!(
            labels.is_empty(),
            "permissive={permissive} cross-scope still admitted: {labels:?} steps={:?}",
            minimal_cross()
        );
    }
}

#[test]
#[ignore = "known bug #399: ingesting the same bytes twice in one scope returns two mem- ids. Canonical equivalents (alpha / ' alpha\\n') do too. Goes green when ingest reinforces onto one node and one admission."]
fn prop_exact_duplicate_reinforces() {
    let tmp = Tmp::new("reinforce");
    let mut store = open(&tmp.path, false);
    let id1 = store
        .ingest_in_scope(input("alpha"), "proj")
        .expect("first");
    let id2 = store
        .ingest_in_scope(input("alpha"), "proj")
        .expect("second");
    assert_eq!(
        id1, id2,
        "exact duplicate must reinforce, not allocate a second node"
    );
    assert_eq!(store.node_count(), 1);
    assert!(live_duplicate_groups(&store).is_empty());
    let id3 = store
        .ingest_in_scope(input(" alpha\n"), "proj")
        .expect("canonical");
    assert_eq!(id1, id3, "canonical duplicate must reinforce");
    assert_eq!(store.node_count(), 1);
}

#[test]
fn mutant_replay_skipping_edge_after_supersede_is_killed() {
    let tmp = Tmp::new("mutant-edge");
    let mut store = open(&tmp.path, true);
    let old = store.ingest(input("old")).expect("old");
    let new = store.ingest(input("new")).expect("new");
    store.supersede(&old, &new).expect("supersede");
    store
        .save_connection(&ConnectionRecord {
            source_id: new,
            target_id: "src/lib.rs".to_string(),
            link_type: EdgeKind::AnchoredTo.as_str().to_string(),
            ..ConnectionRecord::default()
        })
        .expect("edge after supersede");
    let frames = store.log().read_frames(1).expect("frames");
    assert!(
        log_edges_covered(&frames, &store.edges()),
        "honest replay dropped an edge"
    );
    let mut skipped = store.edges();
    let removed = skipped.pop().expect("edge");
    assert!(
        !log_edges_covered(&frames, &skipped),
        "checker accepted a replay that skipped {removed:?}"
    );
}

fn log_edges_covered(frames: &[strata::FrameRecord], edges: &[ConnectionRecord]) -> bool {
    let mut expected = Vec::new();
    for frame in frames {
        if frame.kind != KIND_STORE_WRITE {
            continue;
        }
        match StoreOp::try_from_slice(&frame.payload) {
            Ok(StoreOp::SaveEdge { edge }) => expected.push(edge),
            Ok(StoreOp::SaveEdges { edges }) => expected.extend(edges),
            _ => {}
        }
    }
    expected.iter().all(|edge| edges.contains(edge))
}

#[test]
fn mutant_unadmitted_store_write_is_killed() {
    let tmp = Tmp::new("mutant-admit");
    let mut store = open(&tmp.path, false);
    store.ingest(input("admitted")).expect("ingest");
    let frames = store.log().read_frames(1).expect("frames");
    let mut admitted = BTreeSet::new();
    for frame in &frames {
        if frame.kind == RecordKind::Effect.to_u8() {
            let effect = EffectRecord::try_from_slice(&frame.payload).expect("effect");
            admitted.insert(effect.payload_digest);
        }
    }
    assert!(writes_are_admitted(&frames, &admitted));
    let dropped = admitted.iter().next().copied().expect("effect");
    admitted.remove(&dropped);
    assert!(
        !writes_are_admitted(&frames, &admitted),
        "checker accepted a STORE_WRITE whose admitting digest was dropped"
    );
}
