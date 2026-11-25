//! GhostLink lens tests: bridge admission over recorded typed edges only,
//! divergent eligibility / Path_min / typed divergence, the spread sampler,
//! cursor paging, and the legacy-edge monotonicity guarantee.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::ghostlink::{
    weave_source, Lane, PathVia, PoolFilter, BEYOND_RADIUS, COMPOSITION_NODE_TYPE, GHOSTLINK_TAG,
    LEGACY_INFERRED, WEAVE_TAG,
};
use crate::op::{KIND_STORE_CHECKPOINT, KIND_STORE_WRITE};
use crate::types::{ConnectionRecord, IngestInput, SourceKey};
use crate::{AdmissionContext, StrataStore, RULE_SUPPRESS};

const MONTH_MS: i64 = 30 * 86_400_000;
const BASE_MS: i64 = 1_780_000_000_000;

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "strata-store-ghostlink-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn node(store: &mut StrataStore, content: &str, tags: &[&str], node_type: &str) -> String {
    node_at(store, content, tags, node_type, "user", BASE_MS)
}

fn node_at(
    store: &mut StrataStore,
    content: &str,
    tags: &[&str],
    node_type: &str,
    scope: &str,
    created_at_ms: i64,
) -> String {
    store
        .ingest_in_scope(
            IngestInput {
                content: content.to_string(),
                node_type: node_type.to_string(),
                tags: tags.iter().map(|tag| tag.to_string()).collect(),
                created_at_ms: Some(created_at_ms),
                ..IngestInput::default()
            },
            scope,
        )
        .expect("ingest")
}

fn edge(store: &mut StrataStore, source: &str, target: &str, kind: &str) {
    store
        .save_connection(&ConnectionRecord {
            source_id: source.to_string(),
            target_id: target.to_string(),
            link_type: kind.to_string(),
            ..ConnectionRecord::default()
        })
        .expect("edge");
}

/// A composition record woven the way the MCP weave writes it.
fn weave(store: &mut StrataStore, a: &str, b: &str, outcome: &str) -> String {
    let record = store
        .ingest_in_scope(
            IngestInput {
                content: format!("GhostLink composition of {a} and {b}: outcome {outcome}"),
                source: Some(SourceKey {
                    system: weave_source(a, b),
                    project: String::new(),
                    id: String::new(),
                }),
                node_type: COMPOSITION_NODE_TYPE.to_string(),
                tags: vec![
                    GHOSTLINK_TAG.to_string(),
                    WEAVE_TAG.to_string(),
                    format!("outcome:{outcome}"),
                    "lens:bridge".to_string(),
                ],
                created_at_ms: Some(BASE_MS),
                ..IngestInput::default()
            },
            "user",
        )
        .expect("record");
    edge(store, &record, a, "derived_from");
    edge(store, &record, b, "derived_from");
    record
}

/// Append imported v3 edges (the migration frame shape) to a closed log.
fn append_imported_edges(dir: &Path, edges: &[(String, String, &str)]) {
    let log = strata::StrataLog::open(dir.join("log")).expect("log");
    let frames = edges
        .iter()
        .map(|(source, target, kind)| {
            let payload = borsh::to_vec(&strata_migrate::EdgeRecord {
                record_version: strata_migrate::RECORD_VERSION,
                source_kernel_id: 0,
                target_kernel_id: 0,
                source_legacy_id: source.clone(),
                target_legacy_id: target.clone(),
                link_type: kind.to_string(),
                legacy_inferred: *kind == LEGACY_INFERRED,
                legacy_link_type: "semantic".to_string(),
                strength_q32: 0,
                created_ms: 0,
                last_activated_ms: 0,
                activation_count: 0,
                legacy: Vec::new(),
            })
            .expect("encode edge");
            (KIND_STORE_CHECKPOINT, payload)
        })
        .collect();
    log.append_batch(frames).expect("append");
}

fn user() -> PoolFilter {
    PoolFilter {
        scope: Some("user".into()),
        tags: Vec::new(),
    }
}

fn pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

// ----------------------------------------------------------------------
// Bridge lens
// ----------------------------------------------------------------------

#[test]
fn bridge_is_empty_without_admitting_edges_even_with_shared_tags_and_content() {
    let dir = temp_dir("bridge-empty");
    let mut store = StrataStore::open(&dir).expect("open");
    let a = node(
        &mut store,
        "the cache warms on deploy",
        &["deploy", "cache"],
        "fact",
    );
    let b = node(
        &mut store,
        "the cache warms on deploy",
        &["deploy", "cache"],
        "fact",
    );
    let c = node(
        &mut store,
        "the cache warms on deploy",
        &["deploy"],
        "decision",
    );
    assert!(store.ghost_snapshot(user()).bridge().candidates.is_empty());

    // Non-admitting kinds: anchored_to, evidence_of, legacy_inferred.
    edge(&mut store, &a, &b, "anchored_to");
    edge(&mut store, &b, &c, "evidence_of");
    drop(store);
    append_imported_edges(&dir, &[(a.clone(), c.clone(), LEGACY_INFERRED)]);
    let store = StrataStore::open(&dir).expect("reopen");
    let report = store.ghost_snapshot(user()).bridge();
    assert!(report.candidates.is_empty(), "{report:?}");
    assert_eq!(report.pool_nodes_with_admitting_edges, 0);
    assert_eq!(report.pool_size, 3);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn bridge_admits_one_to_three_hops_and_refuses_four() {
    let dir = temp_dir("bridge-hops");
    let mut store = StrataStore::open(&dir).expect("open");
    let ids: Vec<String> = (0..5)
        .map(|at| node(&mut store, &format!("step {at}"), &[], "event"))
        .collect();
    edge(&mut store, &ids[0], &ids[1], "derived_from");
    edge(&mut store, &ids[2], &ids[1], "touched");
    edge(&mut store, &ids[2], &ids[3], "closed_by");
    edge(&mut store, &ids[3], &ids[4], "derived_from");
    let report = store.ghost_snapshot(user()).bridge();
    let found: Vec<((String, String), u32)> = report
        .candidates
        .iter()
        .map(|c| ((c.first_id.clone(), c.second_id.clone()), c.hops))
        .collect();
    assert!(found.contains(&(pair(&ids[0], &ids[1]), 1)), "{found:?}");
    assert!(found.contains(&(pair(&ids[0], &ids[3]), 3)), "{found:?}");
    assert!(
        !found.iter().any(|(p, _)| *p == pair(&ids[0], &ids[4])),
        "4 hops must not be admitted: {found:?}"
    );

    // The proof path walks recorded edges in order, flagging reversals.
    let three = report
        .candidates
        .iter()
        .find(|c| (c.first_id.clone(), c.second_id.clone()) == pair(&ids[0], &ids[3]))
        .expect("3-hop candidate");
    let steps: Vec<(&str, &str, &str, bool)> = three
        .path
        .iter()
        .map(|s| (s.from.as_str(), s.kind.as_str(), s.to.as_str(), s.reversed))
        .collect();
    assert_eq!(
        steps,
        vec![
            (ids[0].as_str(), "derived_from", ids[1].as_str(), false),
            (ids[1].as_str(), "touched", ids[2].as_str(), true),
            (ids[2].as_str(), "closed_by", ids[3].as_str(), false),
        ]
    );

    // Deterministic: a second snapshot and a reopen give the same report.
    assert_eq!(store.ghost_snapshot(user()).bridge(), report);
    drop(store);
    let store = StrataStore::open(&dir).expect("reopen");
    assert_eq!(store.ghost_snapshot(user()).bridge(), report);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn expired_and_future_memories_are_not_candidates() {
    let dir = temp_dir("bridge-validity");
    let mut store = StrataStore::open(&dir).expect("open");
    let windowed =
        |store: &mut StrataStore, content: &str, from: Option<i64>, until: Option<i64>| {
            store
                .ingest_in_scope(
                    IngestInput {
                        content: content.to_string(),
                        node_type: "fact".to_string(),
                        created_at_ms: Some(BASE_MS),
                        valid_from_ms: from,
                        valid_until_ms: until,
                        ..IngestInput::default()
                    },
                    "user",
                )
                .expect("ingest")
        };
    let hub = node_at(&mut store, "hub", &[], "fact", "user", BASE_MS);
    let live = node_at(&mut store, "live", &[], "fact", "user", BASE_MS);
    let expired = windowed(&mut store, "expired", None, Some(BASE_MS - MONTH_MS));
    let future = windowed(&mut store, "future", Some(BASE_MS + 12 * MONTH_MS), None);
    // The log's own clock moves past the expiry and stays before the start.
    node_at(
        &mut store,
        "latest",
        &[],
        "fact",
        "user",
        BASE_MS + MONTH_MS,
    );
    for id in [&live, &expired, &future] {
        edge(&mut store, id, &hub, "derived_from");
    }

    let snapshot = store.ghost_snapshot(user());
    let pool = snapshot.pool_ids();
    assert!(
        pool.contains(&live.as_str()) && pool.contains(&hub.as_str()),
        "{pool:?}"
    );
    assert!(
        !pool.contains(&expired.as_str()),
        "expired is not a candidate: {pool:?}"
    );
    assert!(
        !pool.contains(&future.as_str()),
        "future is not a candidate: {pool:?}"
    );
    let report = snapshot.bridge();
    let named: Vec<&str> = report
        .candidates
        .iter()
        .flat_map(|c| [c.first_id.as_str(), c.second_id.as_str()])
        .collect();
    assert!(named.contains(&live.as_str()), "{named:?}");
    assert!(
        !named.contains(&expired.as_str()) && !named.contains(&future.as_str()),
        "{named:?}"
    );
    let page = snapshot.divergent_page(None, 20).expect("page");
    let paired: Vec<&str> = page
        .measured
        .iter()
        .chain(page.juxtaposition.iter())
        .flat_map(|c| [c.first_id.as_str(), c.second_id.as_str()])
        .collect();
    assert!(
        !paired.contains(&expired.as_str()) && !paired.contains(&future.as_str()),
        "{paired:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_scoped_proposal_never_walks_or_names_another_scope() {
    let dir = temp_dir("scope-walk");
    let mut store = StrataStore::open(&dir).expect("open");
    let a = node_at(&mut store, "a", &[], "fact", "user", BASE_MS);
    let b = node_at(&mut store, "b", &[], "fact", "user", BASE_MS);
    let other = node_at(&mut store, "other", &[], "fact", "project-x", BASE_MS);
    edge(&mut store, &a, &other, "derived_from");
    edge(&mut store, &b, &other, "derived_from");
    // An artifact id (no memory record) stays walkable in any scope.
    let c = node_at(&mut store, "c", &[], "fact", "user", BASE_MS);
    let d = node_at(&mut store, "d", &[], "fact", "user", BASE_MS);
    edge(&mut store, &c, "file:src/lib.rs", "touched");
    edge(&mut store, &d, "file:src/lib.rs", "touched");

    let snapshot = store.ghost_snapshot(user());
    let report = snapshot.bridge();
    let text = format!("{report:?}");
    assert!(
        !text.contains(&other),
        "bridge named another scope's memory: {text}"
    );
    assert!(
        !report
            .candidates
            .iter()
            .any(|c| (c.first_id.clone(), c.second_id.clone()) == pair(&a, &b)),
        "(a, b) is joined only through another scope: {text}"
    );
    assert!(
        report
            .candidates
            .iter()
            .any(|cand| (cand.first_id.clone(), cand.second_id.clone()) == pair(&c, &d)),
        "an artifact hop stays admitted: {text}"
    );
    let page = format!("{:?}", snapshot.divergent_page(None, 20).expect("page"));
    assert!(
        !page.contains(&other),
        "divergent named another scope's memory: {page}"
    );

    // Across scopes (no scope filter) the same structure admits (a, b).
    let everywhere = store
        .ghost_snapshot(PoolFilter {
            scope: None,
            tags: Vec::new(),
        })
        .bridge();
    assert!(everywhere
        .candidates
        .iter()
        .any(|c| (c.first_id.clone(), c.second_id.clone()) == pair(&a, &b)));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn bridge_top_keeps_exactly_the_best_k_of_the_full_ranking() {
    let dir = temp_dir("bridge-top");
    let mut store = StrataStore::open(&dir).expect("open");
    let ids: Vec<String> = (0..9)
        .map(|at| node(&mut store, &format!("n{at}"), &[], "fact"))
        .collect();
    for at in 0..8 {
        edge(&mut store, &ids[at], &ids[at + 1], "derived_from");
    }
    edge(&mut store, &ids[0], "file:src/hub.rs", "touched");
    edge(&mut store, &ids[5], "file:src/hub.rs", "touched");
    let snapshot = store.ghost_snapshot(user());
    // A deterministic score with ties, so the id tie-break is exercised.
    let score = |a: &str, b: &str, hops: u32| {
        f64::from(hops % 2) + f64::from((a.len() + b.len()) as u32 % 3)
    };
    let full = snapshot.bridge();
    let mut ranked: Vec<(f64, String, String)> = full
        .candidates
        .iter()
        .map(|c| {
            (
                score(&c.first_id, &c.second_id, c.hops),
                c.first_id.clone(),
                c.second_id.clone(),
            )
        })
        .collect();
    ranked.sort_by(|x, y| {
        y.0.total_cmp(&x.0)
            .then_with(|| (&x.1, &x.2).cmp(&(&y.1, &y.2)))
    });
    for k in [0, 1, 3, 7, ranked.len(), ranked.len() + 5] {
        let (top, admitted) = snapshot.bridge_top(k, score);
        assert_eq!(admitted, full.candidates.len(), "k={k}");
        let got: Vec<(String, String)> = top
            .candidates
            .iter()
            .map(|c| (c.first_id.clone(), c.second_id.clone()))
            .collect();
        let want: Vec<(String, String)> = ranked
            .iter()
            .take(k)
            .map(|(_, a, b)| (a.clone(), b.clone()))
            .collect();
        assert_eq!(got, want, "k={k}");
        // Each kept pair carries the same proof the full walk found.
        for c in &top.candidates {
            let same = full
                .candidates
                .iter()
                .find(|f| f.first_id == c.first_id && f.second_id == c.second_id)
                .expect("kept pair is admitted");
            assert_eq!(c.path, same.path);
            assert_eq!(c.hops, same.hops);
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn bridge_excludes_woven_pairs_and_records_leave_the_pool() {
    let dir = temp_dir("bridge-woven");
    let mut store = StrataStore::open(&dir).expect("open");
    let a = node(&mut store, "alpha", &[], "fact");
    let b = node(&mut store, "beta", &[], "fact");
    let c = node(&mut store, "gamma", &[], "fact");
    edge(&mut store, &a, &b, "touched");
    edge(&mut store, &b, &c, "touched");
    let before = store.ghost_snapshot(user()).bridge();
    assert!(before
        .candidates
        .iter()
        .any(|x| (x.first_id.clone(), x.second_id.clone()) == pair(&a, &b)));

    let record = weave(&mut store, &a, &b, "helpful");
    let snapshot = store.ghost_snapshot(user());
    assert!(snapshot.is_woven(&a, &b));
    assert_eq!(snapshot.weave_degree(&a), 1);
    assert_eq!(snapshot.prior_outcomes(&a, &c), vec!["helpful".to_string()]);
    assert!(
        !snapshot.in_pool(&record),
        "composition records are never pool members"
    );
    let after = snapshot.bridge();
    assert!(!after
        .candidates
        .iter()
        .any(|x| (x.first_id.clone(), x.second_id.clone()) == pair(&a, &b)));
    assert!(after.woven_pairs_excluded >= 1);
    assert!(!after
        .candidates
        .iter()
        .any(|x| x.first_id == record || x.second_id == record));

    // A suppressed record no longer weaves: the pair comes back.
    store
        .retire(
            &record,
            &c,
            &AdmissionContext {
                rule_id: Some(RULE_SUPPRESS.to_string()),
                confirm: false,
                human_confirmed: false,
            },
        )
        .expect("suppress record");
    let snapshot = store.ghost_snapshot(user());
    assert!(!snapshot.is_woven(&a, &b));
    assert!(snapshot
        .bridge()
        .candidates
        .iter()
        .any(|x| (x.first_id.clone(), x.second_id.clone()) == pair(&a, &b)));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn bridge_tag_filter_is_exact_identity_on_both_members() {
    let dir = temp_dir("bridge-tags");
    let mut store = StrataStore::open(&dir).expect("open");
    let a = node(&mut store, "a", &["ops"], "fact");
    let b = node(&mut store, "b", &["ops"], "fact");
    let c = node(&mut store, "c", &["ops:db"], "fact");
    edge(&mut store, &a, &b, "derived_from");
    edge(&mut store, &a, &c, "derived_from");
    let filter = PoolFilter {
        scope: Some("user".into()),
        tags: vec!["ops".into()],
    };
    let found: Vec<(String, String)> = store
        .ghost_snapshot(filter)
        .bridge()
        .candidates
        .into_iter()
        .map(|c| (c.first_id, c.second_id))
        .collect();
    assert_eq!(found, vec![pair(&a, &b)], "'ops:db' is not the tag 'ops'");
    std::fs::remove_dir_all(&dir).ok();
}

// ----------------------------------------------------------------------
// Divergent lens
// ----------------------------------------------------------------------

#[test]
fn divergent_excludes_any_edge_kind_and_woven_pairs() {
    let dir = temp_dir("div-exclude");
    let mut store = StrataStore::open(&dir).expect("open");
    let ids: Vec<String> = (0..6)
        .map(|at| node(&mut store, &format!("n{at}"), &[], "fact"))
        .collect();
    edge(&mut store, &ids[0], &ids[1], "anchored_to");
    edge(&mut store, &ids[2], &ids[3], "evidence_of");
    weave(&mut store, &ids[4], &ids[5], "dead_end");
    drop(store);
    append_imported_edges(&dir, &[(ids[1].clone(), ids[2].clone(), LEGACY_INFERRED)]);
    let store = StrataStore::open(&dir).expect("reopen");
    let snapshot = store.ghost_snapshot(user());
    for (a, b) in [(0, 1), (2, 3), (1, 2), (4, 5)] {
        assert!(
            snapshot.divergent_eval(&ids[a], &ids[b]).is_none(),
            "pair {a},{b} must be ineligible"
        );
    }
    let eval = snapshot.divergent_eval(&ids[0], &ids[5]).expect("eligible");
    assert_eq!(eval.lane, Lane::Juxtaposition);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn legacy_path_shortens_path_min_and_is_named() {
    let dir = temp_dir("div-legacy-path");
    let (a, b, x, y, z);
    {
        let mut store = StrataStore::open(&dir).expect("open");
        a = node(&mut store, "a", &[], "fact");
        b = node(&mut store, "b", &[], "fact");
        x = node(&mut store, "x", &[], "fact");
        y = node(&mut store, "y", &[], "fact");
        z = node(&mut store, "z", &[], "fact");
        // Typed path a - x - y - b (3 hops).
        edge(&mut store, &a, &x, "derived_from");
        edge(&mut store, &x, &y, "touched");
        edge(&mut store, &y, &b, "closed_by");
        let snapshot = store.ghost_snapshot(user());
        let typed = snapshot.divergent_eval(&a, &b).expect("eligible");
        assert_eq!(typed.path_min, Some(3));
        assert_eq!(typed.path_via, PathVia::Typed);
        assert_eq!(typed.path.len(), 3);
        assert!(typed.path.iter().all(|step| step.kind != LEGACY_INFERRED));
    }
    // A legacy shortcut a - z - b makes the union path 2 hops.
    append_imported_edges(
        &dir,
        &[
            (a.clone(), z.clone(), LEGACY_INFERRED),
            (z.clone(), b.clone(), LEGACY_INFERRED),
        ],
    );
    let store = StrataStore::open(&dir).expect("reopen");
    let snapshot = store.ghost_snapshot(user());
    let eval = snapshot.divergent_eval(&a, &b).expect("still eligible");
    assert_eq!(eval.path_min, Some(2));
    assert_eq!(eval.path_via, PathVia::LegacyInferred);
    assert!(eval.path.iter().any(|step| step.kind == LEGACY_INFERRED));
    assert_eq!(
        eval.path.first().map(|s| s.from.clone()),
        Some(eval.first_id.clone())
    );
    assert_eq!(
        eval.path.last().map(|s| s.to.clone()),
        Some(eval.second_id.clone())
    );
    // Measured: both have a typed profile, zero typed overlap -> 2 * 1.0.
    assert_eq!(eval.lane, Lane::Measured);
    assert_eq!(eval.score, Some(2.0));
    let _ = (x, y);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn typed_overlap_ignores_legacy_neighbors() {
    let dir = temp_dir("div-overlap");
    let (a, b, c, d, shared_legacy, shared_typed, t1, t2);
    {
        let mut store = StrataStore::open(&dir).expect("open");
        a = node(&mut store, "a", &[], "fact");
        b = node(&mut store, "b", &[], "fact");
        c = node(&mut store, "c", &[], "fact");
        d = node(&mut store, "d", &[], "fact");
        shared_legacy = node(&mut store, "s", &[], "fact");
        shared_typed = node(&mut store, "t", &[], "fact");
        t1 = node(&mut store, "t1", &[], "fact");
        t2 = node(&mut store, "t2", &[], "fact");
        edge(&mut store, &a, &t1, "touched");
        edge(&mut store, &b, &t2, "touched");
        // c and d share a typed neighbor and each has one more.
        edge(&mut store, &c, &shared_typed, "touched");
        edge(&mut store, &d, &shared_typed, "touched");
        edge(&mut store, &c, &t1, "derived_from");
        edge(&mut store, &d, &t2, "derived_from");
    }
    append_imported_edges(
        &dir,
        &[
            (a.clone(), shared_legacy.clone(), LEGACY_INFERRED),
            (b.clone(), shared_legacy.clone(), LEGACY_INFERRED),
        ],
    );
    let store = StrataStore::open(&dir).expect("reopen");
    let snapshot = store.ghost_snapshot(user());
    let legacy_shared = snapshot.divergent_eval(&a, &b).expect("eligible");
    assert_eq!(legacy_shared.shared_typed_neighbors, 0);
    assert_eq!(legacy_shared.divergence, Some(1.0));
    assert_eq!(legacy_shared.path_min, Some(2));
    assert_eq!(legacy_shared.path_via, PathVia::LegacyInferred);
    assert_eq!(legacy_shared.score, Some(2.0));

    let typed_shared = snapshot.divergent_eval(&c, &d).expect("eligible");
    assert_eq!(typed_shared.typed_neighbor_counts, [2, 2]);
    assert_eq!(typed_shared.shared_typed_neighbors, 1);
    assert_eq!(typed_shared.divergence, Some(0.5));
    assert_eq!(typed_shared.path_min, Some(2));
    assert_eq!(typed_shared.path_via, PathVia::Typed);
    assert_eq!(typed_shared.score, Some(1.0));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn unmeasured_pairs_have_no_score() {
    let dir = temp_dir("div-unmeasured");
    let mut store = StrataStore::open(&dir).expect("open");
    let a = node(&mut store, "a", &[], "fact");
    let b = node(&mut store, "b", &[], "fact");
    let t = node(&mut store, "t", &[], "fact");
    edge(&mut store, &a, &t, "touched");
    let snapshot = store.ghost_snapshot(user());
    let eval = snapshot.divergent_eval(&a, &b).expect("eligible");
    assert_eq!(eval.lane, Lane::Juxtaposition);
    assert_eq!(eval.score, None);
    assert_eq!(eval.divergence, None);
    assert_eq!(eval.path_min, None);
    assert_eq!(eval.path_via, PathVia::None);
    assert_eq!(eval.typed_neighbor_counts, [1, 0]);
    std::fs::remove_dir_all(&dir).ok();
}

/// A cold pool: no edges at all, several scopes, types and months.
fn cold_store(dir: &Path, count: usize, content: impl Fn(usize) -> String) -> Vec<String> {
    let mut store = StrataStore::open(dir).expect("open");
    let types = ["fact", "decision", "event"];
    (0..count)
        .map(|at| {
            node_at(
                &mut store,
                &content(at),
                &["same-tag"],
                types[at % types.len()],
                "user",
                BASE_MS + (at as i64 % 4) * MONTH_MS,
            )
        })
        .collect()
}

#[test]
fn juxtaposition_pages_are_deterministic_unique_and_never_repeat() {
    let dir = temp_dir("div-pages");
    let ids = cold_store(&dir, 24, |at| format!("cold memory {at}"));
    let store = StrataStore::open(&dir).expect("open");
    let snapshot = store.ghost_snapshot(user());
    let first = snapshot.divergent_page(None, 5).expect("page");
    assert_eq!(first.measured.len(), 0);
    assert_eq!(first.juxtaposition.len(), 5);
    assert_eq!(first.summary.nodes_with_typed_profile, 0);
    assert_eq!(first.summary.isolated_nodes, 24);
    assert_eq!(first.summary.eligible_pairs, 24 * 23 / 2);
    assert_eq!(first.summary.juxtaposition_eligible_pairs, 24 * 23 / 2);
    // Two calls, and a fresh snapshot, are identical.
    assert_eq!(snapshot.divergent_page(None, 5).expect("again"), first);
    assert_eq!(
        store
            .ghost_snapshot(user())
            .divergent_page(None, 5)
            .expect("fresh"),
        first
    );
    // The page spans buckets: more than one node type among its members.
    let nodes: HashSet<String> = first
        .juxtaposition
        .iter()
        .flat_map(|e| [e.first_id.clone(), e.second_id.clone()])
        .map(|id| store.get_node(&id).expect("node").node_type)
        .collect();
    assert!(
        nodes.len() > 1,
        "page should alternate node types: {nodes:?}"
    );

    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut cursor: Option<String> = None;
    let mut pages = 0;
    loop {
        let page = snapshot.divergent_page(cursor.as_deref(), 5).expect("page");
        let mut members = HashSet::new();
        for eval in &page.juxtaposition {
            assert_eq!(eval.lane, Lane::Juxtaposition);
            assert!(eval.score.is_none());
            assert!(
                members.insert(eval.first_id.clone()),
                "member twice on a page"
            );
            assert!(
                members.insert(eval.second_id.clone()),
                "member twice on a page"
            );
            assert!(
                seen.insert((eval.first_id.clone(), eval.second_id.clone())),
                "pair repeated across pages"
            );
        }
        pages += 1;
        match page.next_cursor {
            Some(next) => cursor = Some(next),
            None => break,
        }
        assert!(pages < 1_000, "paging must terminate");
    }
    assert!(seen.len() > 24, "paging should reach well past one page");
    assert!(seen.iter().all(|(a, b)| ids.contains(a) && ids.contains(b)));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_cursor_goes_stale_when_the_log_moves() {
    let dir = temp_dir("div-stale");
    cold_store(&dir, 6, |at| format!("m{at}"));
    let mut store = StrataStore::open(&dir).expect("open");
    let next = store
        .ghost_snapshot(user())
        .divergent_page(None, 1)
        .expect("page")
        .next_cursor
        .expect("more pages");
    node(&mut store, "a later write", &[], "fact");
    let err = store
        .ghost_snapshot(user())
        .divergent_page(Some(&next), 1)
        .expect_err("stale");
    assert!(err.to_string().contains("stale cursor"), "{err}");
    let other_scope = PoolFilter {
        scope: Some("elsewhere".into()),
        tags: Vec::new(),
    };
    let fresh = store
        .ghost_snapshot(user())
        .divergent_page(None, 1)
        .expect("page")
        .next_cursor
        .expect("more");
    assert!(store
        .ghost_snapshot(other_scope)
        .divergent_page(Some(&fresh), 1)
        .is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_crafted_cursor_is_refused_never_a_panic() {
    let dir = temp_dir("div-crafted");
    cold_store(&dir, 6, |at| format!("m{at}"));
    let store = StrataStore::open(&dir).expect("open");
    let snapshot = store.ghost_snapshot(user());
    let next = snapshot
        .divergent_page(None, 1)
        .expect("page")
        .next_cursor
        .expect("more pages");
    // gl1.<head>.<filter>.<measured_offset>.<s>.<p>.<i>
    let parts: Vec<&str> = next.split('.').collect();
    assert_eq!(parts.len(), 7, "{next}");
    let craft = |offset: &str, s: &str, p: &str, i: &str| {
        format!(
            "{}.{}.{}.{offset}.{s}.{p}.{i}",
            parts[0], parts[1], parts[2]
        )
    };
    let max = usize::MAX.to_string();
    for cursor in [
        // A measured offset past every pair the pool can hold.
        craft(&max, parts[4], parts[5], parts[6]),
        // A slot index past the pool: i + s wraps.
        craft(parts[3], "1", "1", &max),
        // (block + 1) * s wraps.
        craft(parts[3], "2", "0", &max),
        // A step no schedule round has.
        craft(parts[3], &max, "0", "0"),
        // Parity is 0 or 1.
        craft(parts[3], "1", "2", "0"),
    ] {
        let err = snapshot
            .divergent_page(Some(&cursor), 3)
            .expect_err(&cursor);
        assert!(
            err.to_string().contains("not a GhostLink divergent cursor"),
            "{cursor}: {err}"
        );
    }
    // The real cursor still pages.
    assert!(snapshot.divergent_page(Some(&next), 3).is_ok());
    std::fs::remove_dir_all(&dir).ok();
}

/// Page until the cursor ends, checking no pair repeats; every pair seen.
fn page_all(
    snapshot: &crate::ghostlink::GhostSnapshot<'_>,
    limit: usize,
) -> HashSet<(String, String)> {
    let mut seen = HashSet::new();
    let mut cursor: Option<String> = None;
    for _ in 0..200 {
        let page = snapshot
            .divergent_page(cursor.as_deref(), limit)
            .expect("page");
        for c in page.measured.iter().chain(page.juxtaposition.iter()) {
            assert!(
                seen.insert((c.first_id.clone(), c.second_id.clone())),
                "pair repeated across pages: {c:?}"
            );
        }
        match page.next_cursor {
            Some(next) => cursor = Some(next),
            None => return seen,
        }
    }
    panic!("paging never ended");
}

#[test]
fn every_eligible_pair_is_reachable_past_the_measured_cap() {
    let dir = temp_dir("div-cap");
    let mut store = StrataStore::open(&dir).expect("open");
    let ids: Vec<String> = (0..6)
        .map(|at| node(&mut store, &format!("m{at}"), &[], "fact"))
        .collect();
    // Four members gain a typed profile through their own file artifact;
    // no two members are linked, so every one of the 15 pairs is eligible.
    for (at, id) in ids.iter().take(4).enumerate() {
        edge(&mut store, id, &format!("file:src/f{at}.rs"), "touched");
    }
    let snapshot = store.ghost_snapshot(user()).with_measured_cap(2);
    let summary = snapshot.divergent_page(None, 2).expect("page").summary;
    assert_eq!(summary.eligible_pairs, 15, "{summary:?}");
    // The measured lane ranks the pairs inside its window only; the rest,
    // typed pairs beyond the cap included, belong to the sampler.
    assert_eq!(summary.measured_eligible_pairs, 1, "{summary:?}");
    assert_eq!(summary.juxtaposition_eligible_pairs, 14, "{summary:?}");
    assert_eq!(summary.measured_members_beyond_cap, 2, "{summary:?}");
    // One pair per page passes nothing over, so paging reaches every
    // eligible pair: pairs of typed members beyond the cap included.
    assert_eq!(
        page_all(&snapshot, 1).len(),
        15,
        "every eligible pair is reachable"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn paging_ends_when_no_juxtaposition_remains() {
    let dir = temp_dir("div-all-typed");
    let mut store = StrataStore::open(&dir).expect("open");
    let ids: Vec<String> = (0..5)
        .map(|at| node(&mut store, &format!("t{at}"), &[], "fact"))
        .collect();
    for (at, id) in ids.iter().enumerate() {
        edge(&mut store, id, &format!("file:src/t{at}.rs"), "touched");
    }
    let snapshot = store.ghost_snapshot(user());
    // Every member is typed: the measured lane owns all 10 pairs, and the
    // last page of that lane ends the cursor instead of empty pages.
    let first = snapshot.divergent_page(None, 10).expect("page");
    assert_eq!(first.measured.len(), 10, "{first:?}");
    assert_eq!(first.next_cursor, None, "{first:?}");
    assert_eq!(page_all(&snapshot, 3).len(), 10);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_reciprocal_link_counts_once() {
    let dir = temp_dir("div-reciprocal");
    let mut store = StrataStore::open(&dir).expect("open");
    let a = node(&mut store, "a", &[], "fact");
    let b = node(&mut store, "b", &[], "fact");
    node(&mut store, "c", &[], "fact");
    edge(&mut store, &a, &b, "derived_from");
    edge(&mut store, &b, &a, "derived_from");
    let summary = store
        .ghost_snapshot(user())
        .divergent_page(None, 3)
        .expect("page")
        .summary;
    assert_eq!(summary.typed_edges_in_pool, 1, "{summary:?}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn identical_content_and_tags_give_no_advantage() {
    let same = temp_dir("div-same-content");
    let differ = temp_dir("div-differ-content");
    cold_store(&same, 12, |_| {
        "identical words and identical tags".to_string()
    });
    cold_store(&differ, 12, |at| {
        format!("wholly different text number {at}")
    });
    let same_store = StrataStore::open(&same).expect("open");
    let differ_store = StrataStore::open(&differ).expect("open");
    let a = same_store
        .ghost_snapshot(user())
        .divergent_page(None, 6)
        .expect("page");
    let b = differ_store
        .ghost_snapshot(user())
        .divergent_page(None, 6)
        .expect("page");
    // Same ids (same log layout), same pairs, same proofs: text is inert.
    assert_eq!(a.juxtaposition, b.juxtaposition);
    assert!(same_store
        .ghost_snapshot(user())
        .bridge()
        .candidates
        .is_empty());
    std::fs::remove_dir_all(&same).ok();
    std::fs::remove_dir_all(&differ).ok();
}

/// Build the same typed fixture twice; the second log also carries legacy
/// edges. Node ids match because the native frames are identical.
fn monotonic_fixture(dir: &Path, legacy: bool) -> Vec<String> {
    let ids;
    {
        let mut store = StrataStore::open(dir).expect("open");
        ids = (0..10)
            .map(|at| node(&mut store, &format!("mono {at}"), &[], "fact"))
            .collect::<Vec<_>>();
        edge(&mut store, &ids[0], &ids[1], "touched");
        edge(&mut store, &ids[1], &ids[2], "derived_from");
        edge(&mut store, &ids[3], &ids[1], "closed_by");
        edge(&mut store, &ids[4], &ids[5], "touched");
        edge(&mut store, &ids[6], &ids[5], "derived_from");
        edge(&mut store, &ids[7], &ids[8], "anchored_to");
    }
    if legacy {
        let links = [(0, 4), (2, 6), (3, 9), (8, 9), (0, 7), (5, 2)];
        let edges: Vec<(String, String, &str)> = links
            .iter()
            .map(|&(a, b)| (ids[a].clone(), ids[b].clone(), LEGACY_INFERRED))
            .collect();
        append_imported_edges(dir, &edges);
    }
    ids
}

#[test]
fn legacy_edges_only_dampen_divergent_eligibility_and_scores() {
    let without = temp_dir("mono-without");
    let with = temp_dir("mono-with");
    let ids = monotonic_fixture(&without, false);
    assert_eq!(monotonic_fixture(&with, true), ids);
    let clean = StrataStore::open(&without).expect("open");
    let legacy = StrataStore::open(&with).expect("open");
    let clean_view = clean.ghost_snapshot(user());
    let legacy_view = legacy.ghost_snapshot(user());
    let (mut removed, mut lowered) = (0, 0);
    for (at, a) in ids.iter().enumerate() {
        for b in &ids[at + 1..] {
            let base = clean_view.divergent_eval(a, b);
            let damped = legacy_view.divergent_eval(a, b);
            match (&base, &damped) {
                (None, Some(_)) => panic!("legacy edges added candidate {a},{b}"),
                (Some(_), None) => removed += 1,
                (Some(x), Some(y)) => {
                    assert_eq!(x.lane, y.lane, "lanes come from typed profiles only");
                    assert_eq!(x.divergence, y.divergence, "divergence is typed only");
                    if let (Some(sx), Some(sy)) = (x.score, y.score) {
                        assert!(sy <= sx, "legacy raised {a},{b}: {sx} -> {sy}");
                        if sy < sx {
                            lowered += 1;
                        }
                    }
                    let (dx, dy) = (
                        x.path_min.unwrap_or(BEYOND_RADIUS),
                        y.path_min.unwrap_or(BEYOND_RADIUS),
                    );
                    assert!(dy <= dx, "legacy lengthened Path_min for {a},{b}");
                }
                (None, None) => {}
            }
        }
    }
    assert_eq!(removed, 6, "each legacy link removes exactly its own pair");
    assert!(lowered > 0, "the fixture must exercise a dampened score");
    let summary = legacy_view.divergent_page(None, 3).expect("page").summary;
    assert_eq!(summary.legacy_edges_in_pool, 6);
    assert_eq!(summary.legacy_only_pairs_in_pool, 6);
    std::fs::remove_dir_all(&without).ok();
    std::fs::remove_dir_all(&with).ok();
}

#[test]
fn measured_lane_comes_first_best_score_first() {
    let dir = temp_dir("div-measured");
    let mut store = StrataStore::open(&dir).expect("open");
    let ids: Vec<String> = (0..8)
        .map(|at| node(&mut store, &format!("m{at}"), &[], "fact"))
        .collect();
    // Two typed islands plus two cold memories.
    edge(&mut store, &ids[0], &ids[1], "touched");
    edge(&mut store, &ids[2], &ids[1], "touched");
    edge(&mut store, &ids[3], &ids[4], "derived_from");
    let page = store
        .ghost_snapshot(user())
        .divergent_page(None, 50)
        .expect("page");
    assert!(!page.measured.is_empty());
    let scores: Vec<f64> = page
        .measured
        .iter()
        .map(|e| e.score.expect("scored"))
        .collect();
    assert!(scores.windows(2).all(|w| w[0] >= w[1]), "{scores:?}");
    // Across islands nothing connects: beyond the radius scores 7.
    assert_eq!(scores[0], f64::from(BEYOND_RADIUS));
    // 0 and 2 share neighbor 1: d = 2, overlap 1 -> score 0, still listed last.
    let last = page.measured.last().expect("last");
    assert_eq!(
        (last.first_id.clone(), last.second_id.clone()),
        pair(&ids[0], &ids[2])
    );
    assert_eq!(last.score, Some(0.0));
    assert!(page
        .juxtaposition
        .iter()
        .all(|e| e.lane == Lane::Juxtaposition));
    assert!(page
        .measured
        .iter()
        .all(|e| e.typed_neighbor_counts.iter().all(|&n| n > 0)));
    std::fs::remove_dir_all(&dir).ok();
}

/// Owner-scale shape: ~9,000 memories, ~13,500 recorded edges (almost all
/// legacy_inferred), ~3,900 isolated, one ~3,650-node component and ~350
/// small ones. Written as v3 import frames, the way the owner store holds it.
fn owner_scale_store(dir: &Path) {
    let log = strata::StrataLog::open(dir.join("log")).expect("log");
    let id = |at: usize| format!("00000000-0000-4000-8000-{at:012}");
    let mut frames = Vec::with_capacity(23_000);
    for at in 0..9_000usize {
        let payload = borsh::to_vec(&strata_migrate::NodeRecord {
            record_version: strata_migrate::RECORD_VERSION,
            legacy_id: id(at),
            kernel_id: at as u64 + 1,
            content: format!("owner-scale memory {at}"),
            node_type: ["fact", "decision", "event", "pattern"][at % 4].to_string(),
            tags: Vec::new(),
            created_ms: BASE_MS - (at as i64 % 18) * MONTH_MS,
            updated_ms: BASE_MS,
            last_accessed_ms: BASE_MS,
            legacy: Vec::new(),
            source: None,
            source_updated_at_ms: None,
        })
        .expect("node");
        frames.push((KIND_STORE_WRITE, payload));
    }
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = move |bound: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % bound as u64) as usize
    };
    let mut edges: Vec<(usize, usize, &str)> = Vec::new();
    // Giant component: 3,650 nodes from 3,900.
    let giant = 3_900..7_550usize;
    for at in giant.clone().skip(1) {
        edges.push((at, 3_900 + next(at - 3_900), LEGACY_INFERRED));
    }
    while edges.len() < 12_400 {
        let a = 3_900 + next(3_650);
        let b = 3_900 + next(3_650);
        if a != b {
            edges.push((a, b, LEGACY_INFERRED));
        }
    }
    // ~360 small components of four, legacy links only (the owner store
    // has no touched / derived_from / closed_by edge at all).
    let mut at = 7_550;
    while at + 4 <= 9_000 {
        edges.push((at, at + 1, LEGACY_INFERRED));
        edges.push((at + 1, at + 2, LEGACY_INFERRED));
        edges.push((at + 2, at + 3, LEGACY_INFERRED));
        at += 4;
    }
    // Three typed edges: a six-member measured lane of 12 pairs.
    edges.push((100, 200, "derived_from"));
    edges.push((4_000, 4_001, "touched"));
    edges.push((7_600, 300, "closed_by"));
    for (a, b, kind) in &edges {
        let payload = borsh::to_vec(&strata_migrate::EdgeRecord {
            record_version: strata_migrate::RECORD_VERSION,
            source_kernel_id: *a as u64 + 1,
            target_kernel_id: *b as u64 + 1,
            source_legacy_id: id(*a),
            target_legacy_id: id(*b),
            link_type: kind.to_string(),
            legacy_inferred: *kind == LEGACY_INFERRED,
            legacy_link_type: "semantic".into(),
            strength_q32: 0,
            created_ms: 0,
            last_activated_ms: 0,
            activation_count: 0,
            legacy: Vec::new(),
        })
        .expect("edge");
        frames.push((KIND_STORE_CHECKPOINT, payload));
    }
    for chunk in frames.chunks(4_096) {
        log.append_batch(chunk.to_vec()).expect("append");
    }
}

/// One hub memory with `spokes` memories each `touched` to it, written as
/// imported frames (fast): every spoke pair is two hops apart.
fn hub_store(dir: &Path, spokes: usize) {
    let log = strata::StrataLog::open(dir.join("log")).expect("log");
    let id = |at: usize| format!("00000000-0000-4000-8000-{at:012}");
    let mut frames = Vec::with_capacity(2 * spokes + 1);
    for at in 0..=spokes {
        let payload = borsh::to_vec(&strata_migrate::NodeRecord {
            record_version: strata_migrate::RECORD_VERSION,
            legacy_id: id(at),
            kernel_id: at as u64 + 1,
            content: format!("hub-scale memory {at}"),
            node_type: "fact".to_string(),
            tags: Vec::new(),
            created_ms: BASE_MS,
            updated_ms: BASE_MS,
            last_accessed_ms: BASE_MS,
            legacy: Vec::new(),
            source: None,
            source_updated_at_ms: None,
        })
        .expect("node");
        frames.push((KIND_STORE_WRITE, payload));
    }
    for at in 1..=spokes {
        let payload = borsh::to_vec(&strata_migrate::EdgeRecord {
            record_version: strata_migrate::RECORD_VERSION,
            source_kernel_id: at as u64 + 1,
            target_kernel_id: 1,
            source_legacy_id: id(at),
            target_legacy_id: id(0),
            link_type: "touched".to_string(),
            legacy_inferred: false,
            legacy_link_type: "touched".into(),
            strength_q32: 0,
            created_ms: 0,
            last_activated_ms: 0,
            activation_count: 0,
            legacy: Vec::new(),
        })
        .expect("edge");
        frames.push((KIND_STORE_CHECKPOINT, payload));
    }
    for chunk in frames.chunks(4_096) {
        log.append_batch(chunk.to_vec()).expect("append");
    }
}

#[test]
fn a_hub_admits_millions_of_pairs_but_holds_only_the_page() {
    let dir = temp_dir("bridge-hub");
    let spokes = 2_000usize;
    hub_store(&dir, spokes);
    let store = StrataStore::open(&dir).expect("open");
    let snapshot = store.ghost_snapshot(user());
    let started = Instant::now();
    let (report, admitted) = snapshot.bridge_top(10, |_, _, hops| 1.0 / f64::from(hops));
    let elapsed = started.elapsed();
    eprintln!("hub-scale: {admitted} admitted pairs, top 10 in {elapsed:?}");
    // Hub-spoke pairs are one hop, spoke-spoke pairs two.
    assert_eq!(admitted, spokes + spokes * (spokes - 1) / 2);
    assert_eq!(report.candidates.len(), 10);
    assert!(
        report.candidates.iter().all(|c| c.hops == 1),
        "one-hop pairs rank first"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn owner_scale_divergent_and_bridge_answer_under_two_seconds() {
    let dir = temp_dir("owner-scale");
    owner_scale_store(&dir);
    let store = StrataStore::open(&dir).expect("open");
    assert_eq!(store.node_count(), 9_000);
    assert!(store.edge_count() >= 13_000);

    let started = Instant::now();
    let snapshot = store.ghost_snapshot(user());
    let page = snapshot.divergent_page(None, 20).expect("page");
    let second = snapshot
        .divergent_page(page.next_cursor.as_deref(), 20)
        .expect("page 2");
    let bridge = snapshot.bridge();
    let elapsed = started.elapsed();
    eprintln!(
        "owner-scale: divergent x2 + bridge in {elapsed:?}; measured {} jux {} bridge {}",
        page.measured.len(),
        page.juxtaposition.len(),
        bridge.candidates.len()
    );
    assert_eq!(page.summary.nodes_with_typed_profile, 6);
    assert_eq!(page.measured.len(), 12);
    assert_eq!(page.juxtaposition.len(), 8);
    assert_eq!(second.measured.len(), 0);
    assert_eq!(second.juxtaposition.len(), 20);
    assert!(
        elapsed.as_secs_f64() < 2.0,
        "GhostLink must answer an owner-scale store under 2 s; took {elapsed:?}"
    );
    assert!(page.summary.isolated_nodes >= 3_800);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn outcome_conductance_reinforces_proven_paths_damps_dead_ends_and_replays_bit_exact() {
    use crate::ghostlink::{
        apply_outcome_conductance_q16, series_resistance_from_conductance_q16,
        CONDUCTANCE_Q16_ONE, OUTCOME_MULTIPLIER_MAX_Q16, OUTCOME_MULTIPLIER_MIN_Q16,
    };

    let dir = temp_dir("outcome-conductance-854");
    let (a, b, c, d, rec_good, rec_bad, head_seq_before_reopen);
    {
        let mut store = StrataStore::open(&dir).expect("open");
        a = node(&mut store, "root auth session token drift", &["auth"], "fact");
        b = node(&mut store, "token ttl mismatch fix", &["auth"], "fact");
        c = node(&mut store, "false caching hypothesis", &["cache"], "fact");
        d = node(&mut store, "unrelated live node", &["misc"], "fact");
        edge(&mut store, &a, &b, "touched");
        edge(&mut store, &a, &c, "touched");
        edge(&mut store, &c, &d, "derived_from");

        // Before any outcomes, conductance multiplier is strictly identity (1.0 in Q16.16).
        let snap0 = store.ghost_snapshot(user());
        assert_eq!(snap0.outcome_conductance_q16(&a, &b), CONDUCTANCE_Q16_ONE);
        assert!(snap0.outcome_conductance_edges().is_empty());

        // Reinforce (a, b) with accepted + causal_walk_proved, and damp (c, d) with dead_end.
        rec_good = weave(&mut store, &a, &b, "accepted");
        let _rec_proved = weave(&mut store, &a, &b, "causal_walk_proved");
        rec_bad = weave(&mut store, &c, &d, "closed_by_false_assumption");
        let _rec_dead = weave(&mut store, &c, &d, "dead_end");

        let snap = store.ghost_snapshot(user());
        head_seq_before_reopen = snap.head_seq();

        let good = snap.outcome_conductance_for_pair(&a, &b);
        let bad = snap.outcome_conductance_for_pair(&c, &d);

        // Reinforcing outcomes increase conductance (> Q16_ONE) and lower series resistance.
        assert!(good.multiplier_q16 > CONDUCTANCE_Q16_ONE);
        assert!(good.multiplier_q16 <= OUTCOME_MULTIPLIER_MAX_Q16);
        assert!(good.relative_resistance_q16 < CONDUCTANCE_Q16_ONE);
        assert!(good.receipt_ids.contains(&rec_good));
        assert!(good.outcomes.contains(&"accepted".to_string()));
        assert!(good.outcomes.contains(&"causal_walk_proved".to_string()));

        // Inhibitory outcomes lower conductance (< Q16_ONE, clamped at MIN) and raise resistance
        // without deleting any Strata nodes or bytes.
        assert_eq!(bad.multiplier_q16, OUTCOME_MULTIPLIER_MIN_Q16);
        assert!(bad.relative_resistance_q16 > CONDUCTANCE_Q16_ONE * 4);
        assert!(bad.receipt_ids.contains(&rec_bad));
        assert!(store.node_map().contains_key(&c));
        assert!(store.node_map().contains_key(&d));

        // Base conductance routing flips decisively toward the proven wire over the damped wire.
        let base_k = CONDUCTANCE_Q16_ONE;
        let k_good = apply_outcome_conductance_q16(base_k, good.multiplier_q16);
        let k_bad = apply_outcome_conductance_q16(base_k, bad.multiplier_q16);
        assert!(k_good > k_bad * 5);
        assert!(
            series_resistance_from_conductance_q16(k_good)
                < series_resistance_from_conductance_q16(k_bad)
        );

        // `outcome_conductance_edges` returns deterministic canonical edges carrying receipts.
        let edges_first = snap.outcome_conductance_edges();
        assert!(!edges_first.is_empty());
        assert!(edges_first.iter().all(|e| e.head_seq == head_seq_before_reopen));
    }

    // Reopen the append-only Strata log and verify bit-exact replay at identical `head_seq`.
    let reopened = StrataStore::open(&dir).expect("reopen");
    let snap_reopened = reopened.ghost_snapshot(user());
    assert_eq!(snap_reopened.head_seq(), head_seq_before_reopen);
    let good_replayed = snap_reopened.outcome_conductance_for_pair(&b, &a);
    let bad_replayed = snap_reopened.outcome_conductance_for_pair(&d, &c);
    assert!(good_replayed.receipt_ids.contains(&rec_good));
    assert!(bad_replayed.receipt_ids.contains(&rec_bad));
    assert_eq!(bad_replayed.multiplier_q16, OUTCOME_MULTIPLIER_MIN_Q16);
    assert_eq!(
        snap_reopened.outcome_conductance_edges().len(),
        reopened.ghost_snapshot(user()).outcome_conductance_edges().len()
    );
    assert_eq!(
        snap_reopened.outcome_conductance_edges(),
        reopened.ghost_snapshot(user()).outcome_conductance_edges()
    );

    std::fs::remove_dir_all(&dir).ok();
}
