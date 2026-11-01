//! GhostLink ranking tests: proof strength (edge-kind weights, hub damping,
//! the strongest shortest path as the proof), the per-memory page cap, and
//! the rotating measured window that measures every typed member.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::ghostlink::{
    GhostSnapshot, PoolFilter, BEYOND_RADIUS, W_CLOSED_BY, W_DERIVED_FROM, W_TOUCHED_ARTIFACT,
    W_TOUCHED_ASSOCIATION,
};
use crate::types::{ConnectionRecord, IngestInput};
use crate::StrataStore;

const BASE_MS: i64 = 1_780_000_000_000;

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "strata-store-ghostlink-rank-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn node_in(store: &mut StrataStore, content: &str, scope: &str) -> String {
    store
        .ingest_in_scope(
            IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                created_at_ms: Some(BASE_MS),
                ..IngestInput::default()
            },
            scope,
        )
        .expect("ingest")
}

fn node(store: &mut StrataStore, content: &str) -> String {
    node_in(store, content, "user")
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

fn user() -> PoolFilter {
    PoolFilter {
        scope: Some("user".into()),
        tags: Vec::new(),
    }
}

fn key(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

/// Every admitted pair with the proof strength `bridge_ranked` scored it by.
fn strengths(snapshot: &GhostSnapshot<'_>) -> HashMap<(String, String), f64> {
    let mut out = HashMap::new();
    snapshot.bridge_ranked(usize::MAX, usize::MAX, |a, b, _, strength| {
        out.insert(key(a, b), strength);
        strength
    });
    out
}

fn members_once(pairs: &[(String, String)]) -> bool {
    let mut seen = HashSet::new();
    pairs
        .iter()
        .all(|(a, b)| seen.insert(a.clone()) && seen.insert(b.clone()))
}

/// A hub memory auto-connected to six spokes (touched between memories),
/// plus a causal chain `a -derived_from-> x -closed_by-> b`.
struct StarAndChain {
    hub: String,
    spokes: Vec<String>,
    a: String,
    x: String,
    b: String,
}

fn star_and_chain(store: &mut StrataStore) -> StarAndChain {
    let hub = node(store, "hub");
    let spokes: Vec<String> = (0..6)
        .map(|at| node(store, &format!("spoke {at}")))
        .collect();
    for spoke in &spokes {
        edge(store, spoke, &hub, "touched");
    }
    let a = node(store, "chain a");
    let x = node(store, "chain x");
    let b = node(store, "chain b");
    edge(store, &a, &x, "derived_from");
    edge(store, &x, &b, "closed_by");
    StarAndChain {
        hub,
        spokes,
        a,
        x,
        b,
    }
}

// ----------------------------------------------------------------------
// Proof strength
// ----------------------------------------------------------------------

#[test]
fn a_two_hop_causal_path_outranks_a_one_hop_auto_connect_star() {
    let dir = temp_dir("causal-beats-star");
    let mut store = StrataStore::open(&dir).expect("open");
    let fx = star_and_chain(&mut store);
    let snapshot = store.ghost_snapshot(user());
    let strength = strengths(&snapshot);
    let causal = strength[&key(&fx.a, &fx.b)];
    // Both chain ends are leaves and x is a pure pass-through: 1 * 1 / 2.
    assert!(
        (causal - W_DERIVED_FROM * W_CLOSED_BY / 2.0).abs() < 1e-12,
        "{causal}"
    );
    for spoke in &fx.spokes {
        let star = strength[&key(&fx.hub, spoke)];
        assert!(
            star < causal,
            "hub-spoke {star} must rank below causal {causal}"
        );
        // The spoke is a leaf; the hub's five other spokes damp it.
        let want = W_TOUCHED_ASSOCIATION * (1.0 / (2.0f64 + 5.0).log2()).sqrt();
        assert!((star - want).abs() < 1e-12, "{star} vs {want}");
    }
    // Spoke-spoke pairs cross the hub: two association steps and the hub's
    // fan-out make them the weakest of all.
    let spoke_pair = strength[&key(&fx.spokes[0], &fx.spokes[1])];
    assert!(spoke_pair < strength[&key(&fx.hub, &fx.spokes[0])]);

    // Ranked by strength, the causal pair comes before every star pair.
    let (report, _) = snapshot.bridge_ranked(usize::MAX, usize::MAX, |_, _, _, s| s);
    let position = |a: &str, b: &str| {
        report
            .candidates
            .iter()
            .position(|c| (c.first_id.clone(), c.second_id.clone()) == key(a, b))
            .expect("admitted")
    };
    for spoke in &fx.spokes {
        assert!(position(&fx.a, &fx.b) < position(&fx.hub, spoke));
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn scores_spread_instead_of_tying_at_a_ceiling() {
    let dir = temp_dir("spread");
    let mut store = StrataStore::open(&dir).expect("open");
    star_and_chain(&mut store);
    let snapshot = store.ghost_snapshot(user());
    let mut distinct: Vec<f64> = strengths(&snapshot).into_values().collect();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    // Four kinds of pair in the fixture: a causal step (a-x, x-b), the
    // causal chain (a-b), hub-spoke and spoke-spoke, each its own strength.
    assert_eq!(distinct.len(), 4, "{distinct:?}");
    assert!(
        distinct.iter().all(|s| *s > 0.0 && *s <= 1.0),
        "{distinct:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_proof_is_the_strongest_shortest_path_and_explains_the_strength() {
    let dir = temp_dir("strongest-proof");
    let mut store = StrataStore::open(&dir).expect("open");
    let a = node(&mut store, "a");
    let b = node(&mut store, "b");
    // Two 2-hop routes from a to b. The weak one runs through a memory with
    // the lower id, so the plain first-discoverer walk takes it.
    let weak = node(&mut store, "weak middle");
    let strong = node(&mut store, "strong middle");
    edge(&mut store, &a, &weak, "touched");
    edge(&mut store, &b, &weak, "touched");
    edge(&mut store, &a, &strong, "derived_from");
    edge(&mut store, &strong, &b, "closed_by");
    let snapshot = store.ghost_snapshot(user());

    let plain = snapshot.bridge();
    let plain_ab = plain
        .candidates
        .iter()
        .find(|c| (c.first_id.clone(), c.second_id.clone()) == key(&a, &b))
        .expect("a-b admitted");
    assert_eq!(
        plain_ab.path[0].to, weak,
        "first discoverer: {:?}",
        plain_ab.path
    );

    let mut scored = HashMap::new();
    let (ranked, admitted) = snapshot.bridge_ranked(usize::MAX, usize::MAX, |x, y, _, s| {
        scored.insert(key(x, y), s);
        s
    });
    assert_eq!(admitted, plain.candidates.len(), "admission is unchanged");
    let ab = ranked
        .candidates
        .iter()
        .find(|c| (c.first_id.clone(), c.second_id.clone()) == key(&a, &b))
        .expect("a-b admitted");
    assert_eq!(ab.hops, 2);
    assert_eq!(ab.path[0].to, strong, "strongest proof: {:?}", ab.path);
    let kinds: Vec<&str> = ab.path.iter().map(|step| step.kind.as_str()).collect();
    assert_eq!(kinds, ["derived_from", "closed_by"]);
    // The strength the ranking used is exactly the strength of the proof.
    for c in &ranked.candidates {
        let recomputed = snapshot.path_evidence(&c.path);
        assert_eq!(
            recomputed.to_bits(),
            scored[&key(&c.first_id, &c.second_id)].to_bits(),
            "{c:?}"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_shared_file_ranks_between_a_causal_step_and_an_auto_connect_link() {
    let dir = temp_dir("artifact");
    let mut store = StrataStore::open(&dir).expect("open");
    let a = node(&mut store, "a");
    let b = node(&mut store, "b");
    edge(&mut store, &a, "file:src/shared.rs", "touched");
    edge(&mut store, &b, "file:src/shared.rs", "touched");
    let c = node(&mut store, "c");
    let d = node(&mut store, "d");
    edge(&mut store, &c, &d, "touched");
    let e = node(&mut store, "e");
    let f = node(&mut store, "f");
    let m = node(&mut store, "m");
    edge(&mut store, &e, &m, "derived_from");
    edge(&mut store, &m, &f, "derived_from");
    let snapshot = store.ghost_snapshot(user());
    let strength = strengths(&snapshot);
    let file = strength[&key(&a, &b)];
    assert!((file - W_TOUCHED_ARTIFACT * W_TOUCHED_ARTIFACT / 2.0).abs() < 1e-12);
    let association = strength[&key(&c, &d)];
    assert!((association - W_TOUCHED_ASSOCIATION).abs() < 1e-12);
    let causal = strength[&key(&e, &f)];
    assert!(
        causal > file && file > association,
        "{causal} {file} {association}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Append imported v3 `legacy_inferred` edges (the migration frame shape)
/// to a closed log.
fn append_legacy_edges(dir: &std::path::Path, edges: &[(&str, &str)]) {
    let log = strata::StrataLog::open(dir.join("log")).expect("log");
    let frames = edges
        .iter()
        .map(|(source, target)| {
            let payload = borsh::to_vec(&strata_migrate::EdgeRecord {
                record_version: strata_migrate::RECORD_VERSION,
                source_kernel_id: 0,
                target_kernel_id: 0,
                source_legacy_id: source.to_string(),
                target_legacy_id: target.to_string(),
                link_type: crate::ghostlink::LEGACY_INFERRED.to_string(),
                legacy_inferred: true,
                legacy_link_type: "semantic".to_string(),
                strength_q32: 0,
                created_ms: 0,
                last_activated_ms: 0,
                activation_count: 0,
                legacy: Vec::new(),
            })
            .expect("encode edge");
            (crate::op::KIND_STORE_CHECKPOINT, payload)
        })
        .collect();
    log.append_batch(frames).expect("append");
}

#[test]
fn legacy_edges_never_touch_bridge_strength() {
    let dir = temp_dir("legacy-strength");
    let (fx, before) = {
        let mut store = StrataStore::open(&dir).expect("open");
        let fx = star_and_chain(&mut store);
        let before = strengths(&store.ghost_snapshot(user()));
        (fx, before)
    };
    append_legacy_edges(
        &dir,
        &[
            (&fx.a, &fx.hub),
            (&fx.b, &fx.spokes[0]),
            (&fx.x, &fx.spokes[1]),
        ],
    );
    let store = StrataStore::open(&dir).expect("reopen");
    assert!(
        store.ghost_snapshot(user()).linked(&fx.a, &fx.hub),
        "the legacy edge is in the graph"
    );
    let after = strengths(&store.ghost_snapshot(user()));
    assert_eq!(before, after);
    std::fs::remove_dir_all(&dir).ok();
}

// ----------------------------------------------------------------------
// Per-memory cap
// ----------------------------------------------------------------------

#[test]
fn one_memory_cannot_fill_a_bridge_page() {
    let dir = temp_dir("bridge-cap");
    let mut store = StrataStore::open(&dir).expect("open");
    // One hub joined to eight spokes by strong causal edges: uncapped, the
    // hub takes the whole page.
    let hub = node(&mut store, "hub");
    let spokes: Vec<String> = (0..8)
        .map(|at| node(&mut store, &format!("s{at}")))
        .collect();
    for spoke in &spokes {
        edge(&mut store, spoke, &hub, "derived_from");
    }
    let snapshot = store.ghost_snapshot(user());
    let pairs = |report: &crate::ghostlink::BridgeReport| -> Vec<(String, String)> {
        report
            .candidates
            .iter()
            .map(|c| (c.first_id.clone(), c.second_id.clone()))
            .collect()
    };
    let (uncapped, _) = snapshot.bridge_ranked(4, usize::MAX, |_, _, _, s| s);
    assert!(
        pairs(&uncapped).iter().all(|(a, b)| a == &hub || b == &hub),
        "the hub monopolizes the uncapped page: {:?}",
        pairs(&uncapped)
    );
    let (capped, admitted) = snapshot.bridge_ranked(4, 1, |_, _, _, s| s);
    let got = pairs(&capped);
    assert_eq!(got.len(), 4);
    assert!(members_once(&got), "each memory at most once: {got:?}");
    assert_eq!(
        got.iter().filter(|(a, b)| a == &hub || b == &hub).count(),
        1,
        "{got:?}"
    );
    // Cap 2 lets the hub in twice, never three times.
    let (two, _) = snapshot.bridge_ranked(6, 2, |_, _, _, s| s);
    let mut count: HashMap<String, usize> = HashMap::new();
    for (a, b) in pairs(&two) {
        *count.entry(a).or_default() += 1;
        *count.entry(b).or_default() += 1;
    }
    assert!(count.values().all(|&n| n <= 2), "{count:?}");
    assert_eq!(count[&hub], 2);
    // A page the cap would leave short is filled from pairs over the cap.
    let (fill, _) = snapshot.bridge_ranked(admitted + 10, 1, |_, _, _, s| s);
    assert_eq!(
        fill.candidates.len(),
        admitted,
        "the cap never hides a pair"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Cap-greedy with fill over the full ranking, computed the obvious way.
fn brute_capped(ranked: &[(f64, String, String)], k: usize, cap: usize) -> Vec<(String, String)> {
    let mut used: HashMap<&str, usize> = HashMap::new();
    let (mut picks, mut over) = (Vec::new(), Vec::new());
    for (_, a, b) in ranked {
        if picks.len() >= k {
            break;
        }
        let ok = |id: &str| used.get(id).copied().unwrap_or(0) < cap;
        if ok(a) && ok(b) {
            *used.entry(a).or_default() += 1;
            *used.entry(b).or_default() += 1;
            picks.push((a.clone(), b.clone()));
        } else {
            over.push((a.clone(), b.clone()));
        }
    }
    let room = k - picks.len();
    picks.extend(over.into_iter().take(room));
    picks
}

#[test]
fn the_capped_page_is_exactly_the_cap_greedy_page_over_every_pair() {
    let dir = temp_dir("bridge-cap-exact");
    let mut store = StrataStore::open(&dir).expect("open");
    let ids: Vec<String> = (0..14)
        .map(|at| node(&mut store, &format!("n{at}")))
        .collect();
    // A deterministic tangle of every edge kind, hubs included.
    let kinds = ["touched", "derived_from", "closed_by"];
    for at in 0..14 {
        for step in [1usize, 3, 5] {
            let to = (at * 7 + step) % 14;
            if to != at && (at + step) % 3 != 0 {
                edge(&mut store, &ids[at], &ids[to], kinds[(at + step) % 3]);
            }
        }
    }
    edge(&mut store, &ids[2], "file:src/core.rs", "touched");
    edge(&mut store, &ids[9], "file:src/core.rs", "touched");
    let snapshot = store.ghost_snapshot(user());
    let score = |_: &str, _: &str, _: u32, strength: f64| strength;
    let (full, admitted) = snapshot.bridge_ranked(usize::MAX, usize::MAX, score);
    assert!(admitted > 40, "the fixture must be dense: {admitted}");
    let mut ranked: Vec<(f64, String, String)> = Vec::new();
    snapshot.bridge_ranked(usize::MAX, usize::MAX, |a, b, hops, s| {
        let value = score(a, b, hops, s);
        ranked.push((value, a.to_string(), b.to_string()));
        value
    });
    ranked.sort_by(|x, y| {
        y.0.total_cmp(&x.0)
            .then_with(|| (&x.1, &x.2).cmp(&(&y.1, &y.2)))
    });
    let full_pairs: Vec<(String, String)> = full
        .candidates
        .iter()
        .map(|c| (c.first_id.clone(), c.second_id.clone()))
        .collect();
    let want_full: Vec<(String, String)> = ranked
        .iter()
        .map(|(_, a, b)| (a.clone(), b.clone()))
        .collect();
    assert_eq!(full_pairs, want_full);
    for cap in [1usize, 2, 3] {
        for k in [1usize, 2, 5, 9, 20, admitted + 3] {
            let (top, _) = snapshot.bridge_ranked(k, cap, score);
            let got: Vec<(String, String)> = top
                .candidates
                .iter()
                .map(|c| (c.first_id.clone(), c.second_id.clone()))
                .collect();
            assert_eq!(got, brute_capped(&ranked, k, cap), "k={k} cap={cap}");
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

// ----------------------------------------------------------------------
// Divergent: rotating window, cap, tie-break
// ----------------------------------------------------------------------

/// `n` memories, each typed through its own file, so no two are linked and
/// every pair is eligible and measured.
fn typed_islands(store: &mut StrataStore, n: usize) -> Vec<String> {
    (0..n)
        .map(|at| {
            let id = node(store, &format!("island {at}"));
            edge(store, &id, &format!("file:src/island{at}.rs"), "touched");
            id
        })
        .collect()
}

type IdPair = (String, String);

/// Page a capped or rotating ranking to its end: every pair seen, and each
/// page's measured pairs.
fn page_all_with(
    snapshot: &GhostSnapshot<'_>,
    limit: usize,
    cap: usize,
) -> (Vec<IdPair>, Vec<Vec<IdPair>>) {
    let mut all = Vec::new();
    let mut pages = Vec::new();
    let mut seen = HashSet::new();
    let mut cursor: Option<String> = None;
    for _ in 0..2_000 {
        let page = snapshot
            .divergent_page_with(cursor.as_deref(), limit, cap)
            .expect("page");
        let measured: Vec<(String, String)> = page
            .measured
            .iter()
            .map(|e| (e.first_id.clone(), e.second_id.clone()))
            .collect();
        for c in page.measured.iter().chain(page.juxtaposition.iter()) {
            let pair = (c.first_id.clone(), c.second_id.clone());
            assert!(
                seen.insert(pair.clone()),
                "pair repeated across pages: {pair:?}"
            );
            all.push(pair);
        }
        pages.push(measured);
        match page.next_cursor {
            Some(next) => {
                assert!(next.starts_with("gl2."), "{next}");
                cursor = Some(next);
            }
            None => return (all, pages),
        }
    }
    panic!("paging never ended");
}

#[test]
fn a_rotating_window_measures_every_typed_member() {
    let dir = temp_dir("rotate-all");
    let mut store = StrataStore::open(&dir).expect("open");
    let ids = typed_islands(&mut store, 9);
    let cold = node(&mut store, "cold");
    let snapshot = store
        .ghost_snapshot(user())
        .with_measured_cap(4)
        .with_rotating_window();
    let first = snapshot
        .divergent_page_with(None, 3, usize::MAX)
        .expect("page");
    let s = &first.summary;
    assert_eq!(s.measured_members_beyond_cap, 0, "{s:?}");
    assert_eq!(
        s.measured_eligible_pairs, 36,
        "every typed pair is measured: {s:?}"
    );
    assert_eq!(
        s.juxtaposition_eligible_pairs, 9,
        "only the cold member's pairs: {s:?}"
    );
    // 9 members in blocks of 2 -> 5 blocks -> 15 block pairs.
    assert_eq!(s.measured_windows, 15, "{s:?}");

    let (all, pages) = page_all_with(&snapshot, 3, usize::MAX);
    // The measured lane is exhaustive; the cold member's juxtapositions come
    // from the sampler, which places each member once per page.
    assert_eq!(pages.iter().map(Vec::len).sum::<usize>(), 36);
    assert!(all.len() > 36 && all.len() <= 45, "{}", all.len());
    let measured: HashSet<(String, String)> = all
        .iter()
        .filter(|(a, b)| a != &cold && b != &cold)
        .cloned()
        .collect();
    for (at, a) in ids.iter().enumerate() {
        for b in &ids[at + 1..] {
            assert!(
                measured.contains(&key(a, b)),
                "typed pair {a},{b} never measured"
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_rotation_is_fixed_per_head_and_moves_with_the_log() {
    let dir = temp_dir("rotate-seed");
    let mut store = StrataStore::open(&dir).expect("open");
    typed_islands(&mut store, 12);
    let first_page = |store: &StrataStore| -> Vec<(String, String)> {
        let snapshot = store
            .ghost_snapshot(user())
            .with_measured_cap(4)
            .with_rotating_window();
        snapshot
            .divergent_page_with(None, 4, usize::MAX)
            .expect("page")
            .measured
            .iter()
            .map(|e| (e.first_id.clone(), e.second_id.clone()))
            .collect()
    };
    let one = first_page(&store);
    assert_eq!(one, first_page(&store), "same head, same page");
    // A write elsewhere moves the head without touching this pool.
    let mut moved = Vec::new();
    for at in 0..6 {
        node_in(&mut store, &format!("elsewhere {at}"), "elsewhere");
        moved.push(first_page(&store));
    }
    assert!(
        moved.iter().any(|page| page != &one),
        "the window must rotate as the log grows: {one:?} vs {moved:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_capped_divergent_page_holds_each_memory_once_and_still_reaches_every_pair() {
    let dir = temp_dir("div-cap");
    let mut store = StrataStore::open(&dir).expect("open");
    typed_islands(&mut store, 7);
    node(&mut store, "cold one");
    node(&mut store, "cold two");
    for rotating in [false, true] {
        let mut snapshot = store.ghost_snapshot(user()).with_measured_cap(4);
        if rotating {
            snapshot = snapshot.with_rotating_window();
        }
        for limit in [1usize, 2, 3, 5] {
            let (all, pages) = page_all_with(&snapshot, limit, 1);
            for page in &pages {
                assert!(members_once(page), "limit {limit}: {page:?}");
            }
            let eligible = snapshot
                .divergent_page_with(None, 1, 1)
                .expect("page")
                .summary
                .eligible_pairs;
            if !rotating {
                // The plain window leaves pairs past it to the lossy sampler.
                assert!(all.len() as u64 <= eligible);
                continue;
            }
            let measured = pages.iter().map(Vec::len).sum::<usize>();
            assert_eq!(
                measured, 21,
                "every typed pair measured once, limit {limit}"
            );
            assert!(all.len() as u64 <= eligible);
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_cursor_of_one_ranking_is_refused_by_the_other() {
    let dir = temp_dir("div-cursor-mode");
    let mut store = StrataStore::open(&dir).expect("open");
    typed_islands(&mut store, 5);
    let snapshot = store.ghost_snapshot(user());
    let plain = snapshot
        .divergent_page(None, 1)
        .expect("page")
        .next_cursor
        .expect("more");
    let capped = snapshot
        .divergent_page_with(None, 1, 1)
        .expect("page")
        .next_cursor
        .expect("more");
    assert!(plain.starts_with("gl1.") && capped.starts_with("gl2."));
    let err = snapshot
        .divergent_page_with(Some(&plain), 1, 1)
        .expect_err("gl1 in a capped ranking");
    assert!(err.to_string().contains("different ranking"), "{err}");
    let err = snapshot
        .divergent_page(Some(&capped), 1)
        .expect_err("gl2 in the plain ranking");
    assert!(err.to_string().contains("different ranking"), "{err}");
    // A crafted window past the last one is refused, never a panic.
    let mut parts: Vec<String> = capped.split('.').map(str::to_string).collect();
    parts[3] = usize::MAX.to_string();
    let err = snapshot
        .divergent_page_with(Some(&parts.join(".")), 1, 1)
        .expect_err("crafted");
    assert!(
        err.to_string().contains("not a GhostLink divergent cursor"),
        "{err}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn equal_divergent_scores_go_to_the_better_measured_pair() {
    let dir = temp_dir("div-support");
    let mut store = StrataStore::open(&dir).expect("open");
    // Thin members first, so id order alone would list them first.
    let thin: Vec<String> = typed_islands(&mut store, 3);
    let rich: Vec<String> = (0..2)
        .map(|at| {
            let id = node(&mut store, &format!("rich {at}"));
            for file in 0..4 {
                edge(
                    &mut store,
                    &id,
                    &format!("file:src/rich{at}-{file}.rs"),
                    "touched",
                );
            }
            id
        })
        .collect();
    let page = store
        .ghost_snapshot(user())
        .divergent_page(None, 20)
        .expect("page");
    let scores: Vec<f64> = page
        .measured
        .iter()
        .map(|e| e.score.expect("scored"))
        .collect();
    assert!(
        scores.iter().all(|s| *s == f64::from(BEYOND_RADIUS)),
        "every pair is beyond the radius: {scores:?}"
    );
    let top = &page.measured[0];
    assert_eq!(
        (top.first_id.clone(), top.second_id.clone()),
        key(&rich[0], &rich[1])
    );
    assert_eq!(top.typed_neighbor_counts, [4, 4]);
    assert!(thin.len() == 3);
    std::fs::remove_dir_all(&dir).ok();
}
