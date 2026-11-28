//! Unified Incremental Strata GhostCurrent Circuit Index.
//!
//! Single retrieval spine powering `recall(mode="lookup")`, `recall(mode="reason")`,
//! and Reflex's live `memory_packet` hook — eliminating the 3-way split-brain and
//! per-query `O(N)` full-store rebuilds.
//!
//! Pure deterministic math, zero vectors, zero LLMs:
//! - Incremental `ZSet` view deltas (`strata_causal_math::wave16_algebra::ZSet`)
//! - Hardware-accelerated integer posting scoring & sorted intersection (`strata_simd`)
//! - Absorbing Kirchhoff GhostCurrent conductance solver with GhostLink hub damping
//! - AC Temporal Phase & Belnap Polarity interference (`#853`, `strata_episodic`)
//! - Self-Plasticizing Hebbian Retroactive Edge Plasticity (`#854`) from signed receipts
//! - Open-Circuit Back-EMF epistemic abstention & missing-premise proof (`#855`, `strata_causal_math`)
//! - Rissanen MDL marginal bit-gain packet sizing (`mdl_setup_turn_bits`)

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use chrono::{DateTime, Utc};
use rapidhash::HashMapExt;
use rapidhash::fast::RapidHashMap as HashMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use strata_causal_math::wave15_ceiling::prove_premise_coverage;
use strata_causal_math::wave16_algebra::{ZSet, mdl_setup_turn_bits};
use strata_episodic::wave8_physics::{PacketAdmission, destructive_interference_admit};
use strata_simd::{active_simd_isa, dot_i32_postings, intersect_sorted_u32};
use vestige_core::{KnowledgeNode, Storage};

const Q16_ONE: i64 = 65_536;
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "with", "that", "this", "from", "what", "when", "where", "which",
    "who", "how", "why", "are", "was", "were", "have", "has", "had", "been", "will", "would",
    "could", "should", "can", "did", "does", "not", "but", "into", "about", "over", "under",
    "after", "before", "between", "our", "your", "their", "its", "all", "any", "some", "each",
    "more", "most", "other", "than", "then", "also", "just", "only", "very", "now", "here",
];

/// Proof receipt attached to every hit returned by the unified circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitProof {
    pub current_q16: i64,
    pub voltage_q16: i64,
    pub hebbian_boost_q16: i64,
    pub mdl_gain_bits: i64,
    pub polarity_phase: String,
    pub path_witness: Vec<String>,
    pub labile_drifted: bool,
}

/// One ranked node returned by the GhostCurrent circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitHit {
    pub id: String,
    pub score: f32,
    pub proof: CircuitProof,
}

/// Explicit Belnap opposing-polarity conflict surfaced by AC interference (`#853`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitConflict {
    pub source_id: String,
    pub target_id: String,
    pub kind: String,
}

/// Epistemic boundary diagnostic emitted by Open-Circuit Back-EMF (`#855`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicDiagnostic {
    pub status: String,
    pub back_emf_q16: i64,
    pub covered_premises: Vec<String>,
    pub missing_premises: Vec<String>,
    pub simd_isa: String,
}

/// Full result of querying the unified Strata GhostCurrent circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitResponse {
    pub hits: Vec<CircuitHit>,
    pub conflicts: Vec<CircuitConflict>,
    pub epistemic: EpistemicDiagnostic,
    pub total_indexed_nodes: usize,
    pub zset_delta_count: usize,
    pub engine: String,
}

#[derive(Clone)]
struct IndexedNode {
    id: String,
    idx: u32,
    node: KnowledgeNode,
    tokens: BTreeMap<String, i32>,
    retired: bool,
    labile: bool,
    hebbian_delta_q: i64,
}

#[derive(Clone)]
struct CircuitEdge {
    to: u32,
    kind: String,
    conductance_q16: i64,
    opposing_polarity: bool,
}

#[derive(Default)]
struct StoreCircuitCache {
    fast_fp: (usize, u64, u64, u64, u64),
    fingerprint: (usize, usize, usize, usize, u64),
    head_seq: Option<u64>,
    last_labile_hash: u64,
    last_delta_seq: u64,
    delta_seq: u64,
    nodes: Vec<IndexedNode>,
    id_to_idx: HashMap<String, u32>,
    postings: HashMap<String, Vec<u32>>,
    adj: Vec<Vec<CircuitEdge>>,
    scope_ids: HashMap<String, HashSet<String>>,
    receipts: Vec<Value>,
    scores: BTreeMap<String, i128>,
    links_by_id: std::collections::HashMap<String, Vec<vestige_core::retrieval::TurnLink>>,
    entity_tags_lower: Vec<Vec<(String, String, Option<String>)>>,
    zset: ZSet<String>,
    generation: u64,
}

fn cache_registry() -> &'static Mutex<HashMap<PathBuf, StoreCircuitCache>> {
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, StoreCircuitCache>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Fast O(1) structural fingerprint of `storage` without cloning any nodes/edges/receipts.
pub(crate) fn fast_store_stamp(storage: &Storage) -> (usize, u64, u64) {
    if let Some((node_count, seq, edge_count)) = crate::strata_memory::read_store(storage, |s| {
        (
            s.total_node_count(),
            s.log().head().last_acked_seq,
            s.edge_count() as u64,
        )
    }) {
        return (node_count, seq, edge_count);
    }
    let path = storage.db_path();
    let (mut files, mut bytes) = (0u64, 0u64);
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.is_file() {
            files = 1;
            bytes = meta.len();
        } else if let Ok(rd) = std::fs::read_dir(path) {
            for entry in rd.flatten() {
                if let Ok(m) = entry.metadata()
                    && m.is_file()
                {
                    files = files.saturating_add(1);
                    bytes = bytes.saturating_add(m.len());
                }
            }
        }
    }
    (0, files, bytes)
}

/// Notify the incremental `ZSet` cache that a node or edge changed so the next
/// query folds the delta immediately without waiting for count drift.
pub fn mark_zset_delta(storage: &Storage, key: &str, weight: i64) {
    let dir = storage.data_dir().to_path_buf();
    if let Ok(mut reg) = cache_registry().lock() {
        let entry = reg.entry(dir).or_default();
        entry.zset.apply_delta(key.to_string(), weight);
        // Bump generation salt to trigger exact sync on next read.
        entry.delta_seq = entry.delta_seq.wrapping_add(1);
        entry.generation = entry.generation.wrapping_add(1);
        entry.fingerprint.4 = entry.fingerprint.4.wrapping_add(1);
        entry.fast_fp.4 = entry.generation;
    }
}

/// Deterministic tokenizer shared across ingest, slots, and circuit postings.
pub fn tokenize_query_terms(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for raw in text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != ':') {
        let tok = raw.trim_matches(|c: char| c == '-' || c == ':').to_ascii_lowercase();
        if tok.len() < 2 || STOPWORDS.contains(&tok.as_str()) {
            continue;
        }
        if seen.insert(tok.clone()) {
            out.push(tok);
        }
    }
    out
}

fn tokenize_counts(node: &KnowledgeNode) -> BTreeMap<String, i32> {
    let mut counts: BTreeMap<String, i32> = BTreeMap::new();
    let combined = format!("{} {}", node.content, node.tags.join(" "));
    for raw in combined.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != ':') {
        let tok = raw.trim_matches(|c: char| c == '-' || c == ':').to_ascii_lowercase();
        if tok.len() < 2 || STOPWORDS.contains(&tok.as_str()) {
            continue;
        }
        let entry = counts.entry(tok.clone()).or_insert(0);
        *entry = entry.saturating_add(1);
        // Also index typed tag leaf (e.g. `file:src/auth.rs` -> `src/auth.rs`, `auth.rs`)
        if let Some((_, leaf)) = tok.split_once(':')
            && leaf.len() >= 2
        {
            let leaf_entry = counts.entry(leaf.to_string()).or_insert(0);
            *leaf_entry = leaf_entry.saturating_add(2);
        }
    }
    counts
}

fn edge_base_conductance(kind: &str) -> (i64, bool) {
    match kind {
        "supersedes" => (Q16_ONE * 3 / 2, false),
        "closes" | "closed_by" => (Q16_ONE * 5 / 4, false),
        "derived_from" | "evidence_of" => (Q16_ONE, false),
        "synaptic_capture" => (Q16_ONE * 7 / 8, false),
        "backfill" => (Q16_ONE * 6 / 5, false),
        "corrects" | "contradicts" => (Q16_ONE * 3 / 4, true),
        "slot" | "tag" | "calls" | "touched" => (Q16_ONE * 3 / 5, false),
        _ => (Q16_ONE / 2, false),
    }
}

fn sync_cache<'a>(
    cache: &'a mut StoreCircuitCache,
    storage: &Storage,
    labile_ids: &HashSet<String>,
    scope: Option<&str>,
) -> &'a StoreCircuitCache {
    let mut labile_hash: u64 = labile_ids.len() as u64;
    for id in labile_ids {
        for b in id.as_bytes() {
            labile_hash = labile_hash.wrapping_mul(131).wrapping_add(*b as u64);
        }
    }
    let live_head = crate::strata_memory::head_seq(storage);
    let (fast_nodes, fast_files, fast_bytes) = fast_store_stamp(storage);
    let fast_fp = (
        fast_nodes,
        fast_files,
        fast_bytes,
        labile_hash ^ (cache.zset.weights.len() as u64),
        cache.generation,
    );
    if ((live_head.is_some()
        && cache.head_seq == live_head
        && cache.last_labile_hash == labile_hash
        && cache.last_delta_seq == cache.delta_seq)
        || (fast_nodes > 0 && cache.fast_fp == fast_fp))
        && !cache.nodes.is_empty()
    {
        if let Some(sc) = scope
            && !cache.scope_ids.contains_key(sc)
        {
            let ids: HashSet<String> = storage
                .get_all_nodes_in_scope(sc, i32::MAX, 0)
                .unwrap_or_default()
                .into_iter()
                .map(|n| n.id)
                .collect();
            cache.scope_ids.insert(sc.to_string(), ids);
        }
        return cache;
    }

    let all_nodes = storage.get_all_nodes(i32::MAX, 0).unwrap_or_default();
    let all_edges = storage.get_all_connections().unwrap_or_default();
    let receipts = storage.salience_receipts().unwrap_or_default();
    let retired = storage.superseded_node_ids().unwrap_or_default();

    let fp = (
        all_nodes.len(),
        all_edges.len(),
        receipts.len(),
        retired.len(),
        labile_hash ^ cache.delta_seq,
    );
    if cache.fingerprint == fp && !cache.nodes.is_empty() {
        cache.head_seq = live_head;
        cache.last_labile_hash = labile_hash;
        cache.last_delta_seq = cache.delta_seq;
        cache.fast_fp = fast_fp;
        if let Some(sc) = scope
            && !cache.scope_ids.contains_key(sc)
        {
            let ids: HashSet<String> = storage
                .get_all_nodes_in_scope(sc, i32::MAX, 0)
                .unwrap_or_default()
                .into_iter()
                .map(|n| n.id)
                .collect();
            cache.scope_ids.insert(sc.to_string(), ids);
        }
        return cache;
    }
    cache.fast_fp = fast_fp;

    // Recompute incremental ZSet view state and build sorted postings + conductance graph.
    cache.nodes.clear();
    cache.id_to_idx.clear();
    cache.postings.clear();
    cache.scope_ids.clear();
    cache.links_by_id.clear();
    cache.entity_tags_lower.clear();
    cache.zset.weights.clear();
    cache.scores = crate::tools::memory_packet::scores_from_receipts(&receipts);

    // Hebbian retroactive edge/node plasticity (#854) from signed salience receipts.
    let mut hebbian_node_q: HashMap<String, i64> = HashMap::new();
    let mut hebbian_pair_q: HashMap<(String, String), i64> = HashMap::new();
    let undone: HashSet<String> = receipts
        .iter()
        .filter_map(|r| {
            r.get("undo_of")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        })
        .collect();
    for r in &receipts {
        if r.get("undo_of").is_some_and(|v| !v.is_null()) {
            continue;
        }
        if r.get("receipt_id")
            .and_then(Value::as_str)
            .is_some_and(|id| undone.contains(id))
        {
            continue;
        }
        let mem = r.get("memory_id").and_then(Value::as_str).unwrap_or("");
        let trig = r.get("trigger_id").and_then(Value::as_str).unwrap_or("");
        let delta = match (
            r.get("new_stability_q").and_then(Value::as_i64),
            r.get("old_stability_q").and_then(Value::as_i64),
        ) {
            (Some(n), Some(o)) => n - o,
            _ => 0,
        };
        if !mem.is_empty() {
            *hebbian_node_q.entry(mem.to_string()).or_default() += delta;
            if !trig.is_empty() && trig != mem {
                *hebbian_pair_q
                    .entry((trig.to_string(), mem.to_string()))
                    .or_default() += delta;
            }
        }
    }

    let mut sorted_nodes = all_nodes;
    sorted_nodes.sort_by(|a, b| a.created_at.cmp(&b.created_at).then_with(|| a.id.cmp(&b.id)));

    let mut tag_groups: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for (i, node) in sorted_nodes.into_iter().enumerate() {
        let idx = i as u32;
        let tokens = tokenize_counts(&node);
        for tok in tokens.keys() {
            cache.postings.entry(tok.clone()).or_default().push(idx);
        }
        for tag in &node.tags {
            if tag.contains(':') {
                tag_groups.entry(tag.clone()).or_default().push(idx);
            }
        }
        let is_retired = retired.contains(&node.id);
        let is_labile = labile_ids.contains(&node.id);
        let heb = hebbian_node_q.get(&node.id).copied().unwrap_or(0);
        cache.zset.apply_delta(node.id.clone(), 1);
        cache.id_to_idx.insert(node.id.clone(), idx);
        let mut ent_tags = Vec::new();
        for tag in &node.tags {
            if ["person:", "place:", "food:", "file:", "service:", "symbol:"]
                .iter()
                .any(|prefix| tag.starts_with(prefix))
            {
                let low = tag.to_lowercase();
                let leaf = if low.starts_with("symbol:") {
                    low.rsplit_once(':').map(|(_, l)| l.to_string())
                } else {
                    None
                };
                ent_tags.push((tag.clone(), low, leaf));
            }
        }
        cache.entity_tags_lower.push(ent_tags);
        cache.nodes.push(IndexedNode {
            id: node.id.clone(),
            idx,
            node,
            tokens,
            retired: is_retired,
            labile: is_labile,
            hebbian_delta_q: heb,
        });
    }

    let n = cache.nodes.len();
    cache.adj = vec![Vec::new(); n];

    // Build raw adjacency first so we can apply GhostLink hub-degree damping 1/sqrt(deg(u)*deg(v)).
    let mut raw_edges: Vec<(u32, u32, String, i64, bool)> = Vec::new();
    for e in &all_edges {
        if !e.source_id.is_empty() && !e.target_id.is_empty() && e.source_id != e.target_id {
            cache
                .links_by_id
                .entry(e.source_id.clone())
                .or_default()
                .push(vestige_core::retrieval::TurnLink {
                    kind: e.link_type.clone(),
                    to: e.target_id.clone(),
                    degree: 2,
                });
            cache
                .links_by_id
                .entry(e.target_id.clone())
                .or_default()
                .push(vestige_core::retrieval::TurnLink {
                    kind: e.link_type.clone(),
                    to: e.source_id.clone(),
                    degree: 2,
                });
        }
        let (Some(&u), Some(&v)) = (
            cache.id_to_idx.get(&e.source_id),
            cache.id_to_idx.get(&e.target_id),
        ) else {
            continue;
        };
        if u == v {
            continue;
        }
        let (base_g, opposing) = edge_base_conductance(&e.link_type);
        let strength_q = ((e.strength.clamp(0.1, 2.0)) * (Q16_ONE as f64)).round() as i64;
        let heb_pair = hebbian_pair_q
            .get(&(e.source_id.clone(), e.target_id.clone()))
            .or_else(|| hebbian_pair_q.get(&(e.target_id.clone(), e.source_id.clone())))
            .copied()
            .unwrap_or(0);
        // Positive Hebbian reinforcement widens wire conductance; negative pinches it.
        let heb_mult = (Q16_ONE + (heb_pair.clamp(-Q16_ONE / 2, Q16_ONE) / 2)).max(Q16_ONE / 4);
        let g = (base_g * strength_q / Q16_ONE) * heb_mult / Q16_ONE;
        raw_edges.push((u, v, e.link_type.clone(), g.max(1024), opposing));
    }
    for ((trig, mem), delta) in &hebbian_pair_q {
        if let (Some(&u), Some(&v)) = (cache.id_to_idx.get(trig), cache.id_to_idx.get(mem))
            && u != v
        {
            let boost = (Q16_ONE + (*delta).clamp(-Q16_ONE / 2, Q16_ONE)).max(Q16_ONE / 4);
            raw_edges.push((u, v, "backfill".to_string(), boost, false));
        }
    }
    for (tag, members) in tag_groups {
        if members.len() < 2 || members.len() > 32 {
            continue;
        }
        let kind = if tag.starts_with("slot:") {
            "slot"
        } else {
            "tag"
        };
        let (base_g, _) = edge_base_conductance(kind);
        let hub = members[0];
        for &m in members.iter().skip(1) {
            raw_edges.push((hub, m, kind.to_string(), base_g, false));
        }
    }

    let mut degree = vec![1usize; n];
    for &(u, v, _, _, _) in &raw_edges {
        degree[u as usize] += 1;
        degree[v as usize] += 1;
    }
    for (u, v, kind, g, opposing) in raw_edges {
        let deg_prod = ((degree[u as usize] * degree[v as usize]) as f64).sqrt().max(1.0);
        let damped_g = ((g as f64) / deg_prod).round().max(512.0) as i64;
        cache.adj[u as usize].push(CircuitEdge {
            to: v,
            kind: kind.clone(),
            conductance_q16: damped_g,
            opposing_polarity: opposing,
        });
        cache.adj[v as usize].push(CircuitEdge {
            to: u,
            kind,
            conductance_q16: damped_g,
            opposing_polarity: opposing,
        });
    }

    cache.fingerprint = fp;
    cache.head_seq = live_head;
    cache.last_labile_hash = labile_hash;
    cache.last_delta_seq = cache.delta_seq;
    cache.receipts = receipts;
    if let Some(sc) = scope {
        let ids: HashSet<String> = storage
            .get_all_nodes_in_scope(sc, i32::MAX, 0)
            .unwrap_or_default()
            .into_iter()
            .map(|n| n.id)
            .collect();
        cache.scope_ids.insert(sc.to_string(), ids);
    }
    cache
}

/// Prewarm the cached GhostCurrent circuit and store snapshot on daemon startup.
pub fn prewarm(storage: &Storage, scope: &str) {
    let dir = storage.data_dir().to_path_buf();
    let empty_labile = HashSet::new();
    let mut reg = cache_registry().lock().unwrap_or_else(|e| e.into_inner());
    let cache = reg.entry(dir).or_default();
    let _ = sync_cache(cache, storage, &empty_labile, Some(scope));
}

/// Cached snapshot returned cleanly for `memory_packet::packet` when Strata log is live.
pub struct CachedPacketSnapshot {
    pub nodes: Vec<KnowledgeNode>,
    pub scores: BTreeMap<String, i128>,
    pub receipts: Vec<Value>,
    pub entities: BTreeSet<String>,
}

fn word_phrase_match(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    haystack.match_indices(needle).any(|(at, _)| {
        let word = |c: char| c.is_alphanumeric() || c == '_';
        haystack[..at].chars().next_back().is_none_or(|c| !word(c))
            && haystack[at + needle.len()..]
                .chars()
                .next()
                .is_none_or(|c| !word(c))
    })
}

/// Read the cached live scoped nodes, scores, receipts, and matched entity tags for
/// `memory_packet::packet` in O(1) when `head_seq` has not advanced.
pub fn cached_packet_snapshot(
    storage: &Storage,
    scope: &str,
    labile_ids: &HashSet<String>,
    prompt_lower: &str,
    now: DateTime<Utc>,
) -> CachedPacketSnapshot {
    let dir = storage.data_dir().to_path_buf();
    let mut reg = cache_registry().lock().unwrap_or_else(|e| e.into_inner());
    let cache = reg.entry(dir).or_default();
    let cache = sync_cache(cache, storage, labile_ids, Some(scope));
    let scope_set = cache.scope_ids.get(scope);
    let mut nodes = Vec::new();
    let mut entities = BTreeSet::new();
    for (idx, nd) in cache.nodes.iter().enumerate() {
        let in_scope = scope_set.is_some_and(|set| set.contains(&nd.id));
        if !in_scope
            || nd.retired
            || nd.node.valid_from.is_some_and(|t| t > now)
            || nd.node.valid_until.is_some_and(|t| t <= now)
        {
            continue;
        }
        if !prompt_lower.is_empty()
            && let Some(ent_tags) = cache.entity_tags_lower.get(idx)
        {
            for (orig, low, leaf_opt) in ent_tags {
                if entities.contains(orig.as_str()) {
                    continue;
                }
                let matched = word_phrase_match(prompt_lower, low)
                    || low
                        .split_once(':')
                        .is_some_and(|(_, surf)| word_phrase_match(prompt_lower, surf))
                    || leaf_opt.as_deref().is_some_and(|leaf| {
                        !leaf.is_empty()
                            && (word_phrase_match(prompt_lower, leaf)
                                || (leaf.contains('~')
                                    && word_phrase_match(prompt_lower, &leaf.replace('~', " "))))
                    });
                if matched {
                    entities.insert(orig.clone());
                }
            }
        }
        nodes.push(nd.node.clone());
    }
    CachedPacketSnapshot {
        nodes,
        scores: cache.scores.clone(),
        receipts: cache.receipts.clone(),
        entities,
    }
}

/// Query options for the unified GhostCurrent circuit.
pub struct CircuitQuery<'a> {
    pub query: &'a str,
    pub scope: Option<&'a str>,
    pub include_superseded: bool,
    pub valid_at: Option<DateTime<Utc>>,
    pub top_k: usize,
    pub labile_ids: &'a HashSet<String>,
}

/// Execute the ONE unified Strata GhostCurrent + GhostLink + Stara retrieval circuit.
pub fn query_circuit(storage: &Storage, req: &CircuitQuery<'_>) -> CircuitResponse {
    let terms = tokenize_query_terms(req.query);
    let simd = active_simd_isa().to_string();
    if req.query.trim().is_empty() {
        return CircuitResponse {
            hits: Vec::new(),
            conflicts: Vec::new(),
            epistemic: EpistemicDiagnostic {
                status: "empty_query".to_string(),
                back_emf_q16: Q16_ONE,
                covered_premises: Vec::new(),
                missing_premises: Vec::new(),
                simd_isa: simd,
            },
            total_indexed_nodes: 0,
            zset_delta_count: 0,
            engine: "vestige-ghostcurrent-circuit-v1".to_string(),
        };
    }

    let dir = storage.data_dir().to_path_buf();
    let mut reg = cache_registry().lock().unwrap_or_else(|e| e.into_inner());
    let cache = reg.entry(dir).or_default();
    let cache = sync_cache(cache, storage, req.labile_ids, req.scope);
    let n = cache.nodes.len();
    if n == 0 {
        return CircuitResponse {
            hits: Vec::new(),
            conflicts: Vec::new(),
            epistemic: EpistemicDiagnostic {
                status: "abstain_empty_store".to_string(),
                back_emf_q16: Q16_ONE,
                covered_premises: Vec::new(),
                missing_premises: terms,
                simd_isa: simd,
            },
            total_indexed_nodes: 0,
            zset_delta_count: cache.zset.weights.len(),
            engine: "vestige-ghostcurrent-circuit-v1".to_string(),
        };
    }

    // Step 1: Scope & temporal admissibility mask (O(1) cached scope membership lookup).
    let now = req.valid_at.unwrap_or_else(Utc::now);
    let scope_set = req.scope.and_then(|sc| cache.scope_ids.get(sc));
    let mut allowed = vec![false; n];
    for node in &cache.nodes {
        let in_scope = match scope_set {
            Some(set) => set.contains(&node.id),
            None => true,
        };
        if !in_scope {
            continue;
        }
        if !req.include_superseded && node.retired {
            continue;
        }
        if let Some(check_ts) = req.valid_at
            && (node.node.valid_from.is_some_and(|t| t > check_ts)
                || node.node.valid_until.is_some_and(|t| t <= check_ts))
        {
            continue;
        }
        allowed[node.idx as usize] = true;
    }

    // Step 2: Hardware-accelerated SIMD posting scoring (`strata_simd::dot_i32_postings`
    // + `intersect_sorted_u32`) to inject current sources J_src into matching nodes.
    let mut idf_weights: Vec<i32> = Vec::with_capacity(terms.len());
    let mut term_postings: Vec<&[u32]> = Vec::with_capacity(terms.len());
    let mut matched_terms_global: BTreeSet<String> = BTreeSet::new();
    let mut candidate_set: BTreeSet<u32> = BTreeSet::new();

    for t in &terms {
        if let Some(list) = cache.postings.get(t) {
            let df = list.len().max(1) as f64;
            let idf = (((n as f64 + 1.0) / df).ln() + 1.0) * 1024.0;
            idf_weights.push(idf.round().clamp(256.0, 16_384.0) as i32);
            term_postings.push(list.as_slice());
            if list.iter().any(|&idx| allowed[idx as usize]) {
                matched_terms_global.insert(t.clone());
            }
            for &idx in list {
                candidate_set.insert(idx);
            }
        } else {
            idf_weights.push(1024);
            term_postings.push(&[]);
        }
    }

    // Fast SIMD sorted intersection bonus when 2+ rare query terms co-occur in the same node.
    let mut conjunctive_bonus = vec![0i64; n];
    if term_postings.len() >= 2 {
        let mut non_empty: Vec<&[u32]> = term_postings
            .iter()
            .copied()
            .filter(|p| !p.is_empty())
            .collect();
        non_empty.sort_by_key(|p| p.len());
        if non_empty.len() >= 2 {
            let inter = intersect_sorted_u32(non_empty[0], non_empty[1]);
            for idx in inter {
                conjunctive_bonus[idx as usize] = Q16_ONE / 2;
            }
        }
    }

    let mut source_current = vec![0i64; n];
    let mut tf_vec = vec![0i32; terms.len()];
    for &idx in &candidate_set {
        let node = &cache.nodes[idx as usize];
        for (ti, t) in terms.iter().enumerate() {
            tf_vec[ti] = node.tokens.get(t).copied().unwrap_or(0).min(8);
        }
        let dot = dot_i32_postings(&idf_weights, &tf_vec);
        let boost = conjunctive_bonus[idx as usize];
        source_current[idx as usize] = (dot * 8).saturating_add(boost);
    }

    // Step 3: Blend with Stara (`vestige_core::retrieval::rank`) lexical/temporal anchors
    // over a bounded top candidate neighborhood (full set when small, top candidates +
    // 1-hop neighbors when store > 250 nodes) so latency stays sub-10ms on 10K+ stores.
    let stara_pool: Vec<KnowledgeNode> = if n <= 250 {
        cache
            .nodes
            .iter()
            .filter(|nd| allowed[nd.idx as usize])
            .map(|nd| nd.node.clone())
            .collect()
    } else {
        let mut ranked_cands: Vec<(usize, i64)> = candidate_set
            .iter()
            .filter(|&&idx| allowed[idx as usize])
            .map(|&idx| (idx as usize, source_current[idx as usize]))
            .collect();
        ranked_cands.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked_cands.truncate(96);
        let mut pool_idx: BTreeSet<usize> = BTreeSet::new();
        for &(i, _) in &ranked_cands {
            pool_idx.insert(i);
            for e in cache.adj[i].iter().take(4) {
                let nbr = e.to as usize;
                if allowed[nbr] {
                    pool_idx.insert(nbr);
                }
                if pool_idx.len() >= 160 {
                    break;
                }
            }
            if pool_idx.len() >= 160 {
                break;
            }
        }
        pool_idx
            .into_iter()
            .map(|i| cache.nodes[i].node.clone())
            .collect()
    };
    if !stara_pool.is_empty() {
        let stara_hits = crate::tools::recall::rank_loaded_nodes_preloaded(
            &stara_pool,
            Some(&cache.links_by_id),
            &cache.receipts,
            None,
            req.query,
            req.top_k.saturating_mul(4).max(24),
        );
        for (rk, (id, sc)) in stara_hits.into_iter().enumerate() {
            if let Some(&idx) = cache.id_to_idx.get(&id) {
                let rank_q = ((Q16_ONE * 2) / (1 + rk as i64)).max(4_096);
                let sc_q = ((sc.clamp(0.0, 4.0) as f64) * (Q16_ONE as f64) * 0.25).round() as i64;
                source_current[idx as usize] = source_current[idx as usize]
                    .saturating_add(rank_q)
                    .saturating_add(sc_q);
            }
        }
    }

    // Step 4: Absorbing Kirchhoff GhostCurrent Solver (`(L + diag(g_sink)) v = J_src`).
    // Retired/superseded/labile nodes have shunted sink admittance (`g_sink` low/zero) so
    // electrical current passes *through* historical ancestors into their live successors!
    let mut voltage = vec![0i64; n];
    let mut sink_admittance = vec![0i64; n];
    let mut parent_witness: Vec<Option<(u32, String)>> = vec![None; n];

    for node in &cache.nodes {
        let i = node.idx as usize;
        let currently_valid = node.node.valid_from.is_none_or(|t| t <= now)
            && node.node.valid_until.is_none_or(|t| t > now);
        if !allowed[i] || node.retired || !currently_valid {
            // Historical relay node: zero sink absorption (passes current forward to live leaf!).
            sink_admittance[i] = 0;
        } else if node.labile {
            // Anchor Watch marked this memory drifted/labile: heavy shunt until re-verified!
            sink_admittance[i] = Q16_ONE / 8;
        } else {
            let stab_q = ((node.node.retention_strength.clamp(0.25, 1.0)) * (Q16_ONE as f64))
                .round() as i64;
            let heb_q = (node.hebbian_delta_q.clamp(-Q16_ONE / 4, Q16_ONE / 2)) / 2;
            sink_admittance[i] = (stab_q + heb_q).max(Q16_ONE / 4);
        }
    }

    // 10 Gauss-Seidel relaxation sweeps (provably converges on strictly diagonally dominant M-matrix).
    for _sweep in 0..10 {
        for i in 0..n {
            let mut num = source_current[i];
            let mut diag = sink_admittance[i].max(Q16_ONE / 16);
            let mut best_edge_flow = 0i64;
            let mut best_edge: Option<(u32, String)> = None;

            for e in &cache.adj[i] {
                let nbr_v = voltage[e.to as usize];
                let flow = (e.conductance_q16 * nbr_v) / Q16_ONE;
                if flow > best_edge_flow {
                    best_edge_flow = flow;
                    best_edge = Some((e.to, e.kind.clone()));
                }
                num = num.saturating_add(flow);
                diag = diag.saturating_add(e.conductance_q16);
            }
            let next_v = if diag > 0 {
                (num.saturating_mul(Q16_ONE)) / diag
            } else {
                0
            };
            voltage[i] = next_v.max(0);
            if best_edge_flow > 0 && parent_witness[i].is_none() {
                parent_witness[i] = best_edge;
            }
        }
    }

    // Step 5: Compute absorbed sink current J_sink(i) = g_sink(i) * V(i) + AC Conflict Detection (#853).
    let mut scored: Vec<(usize, i64)> = Vec::new();
    for i in 0..n {
        if !allowed[i] || sink_admittance[i] <= 0 || voltage[i] <= 0 {
            continue;
        }
        let j_sink = (sink_admittance[i].saturating_mul(voltage[i])) / Q16_ONE;
        if j_sink > 0 {
            scored.push((i, j_sink));
        }
    }
    scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| cache.nodes[a.0].id.cmp(&cache.nodes[b.0].id)));

    // Belnap destructive interference check (#853) over top candidate neighborhood.
    let top_indices: HashSet<usize> = scored.iter().take(req.top_k * 2).map(|&(i, _)| i).collect();
    let mut conflicts = Vec::new();
    let mut seen_conflicts: HashSet<(String, String)> = HashSet::new();
    let mut polarity_claims: Vec<(String, bool)> = Vec::new();
    for &i in &top_indices {
        let src = &cache.nodes[i].id;
        polarity_claims.push((src.clone(), true));
        for e in &cache.adj[i] {
            if e.opposing_polarity && top_indices.contains(&(e.to as usize)) {
                let dst = &cache.nodes[e.to as usize].id;
                let pair = if src < dst {
                    (src.clone(), dst.clone())
                } else {
                    (dst.clone(), src.clone())
                };
                if seen_conflicts.insert(pair.clone()) {
                    conflicts.push(CircuitConflict {
                        source_id: pair.0,
                        target_id: pair.1,
                        kind: e.kind.clone(),
                    });
                    polarity_claims.push((dst.clone(), false));
                }
            }
        }
    }
    let has_destructive_conflict = matches!(
        destructive_interference_admit(&polarity_claims),
        Some(PacketAdmission::HoldConflict { .. })
    );

    // Step 6: Rissanen MDL packet sizing + Open-Circuit Back-EMF Epistemic Diagnostic (#855).
    let max_j = scored.first().map(|&(_, j)| j).unwrap_or(0).max(1);
    let mut hits = Vec::new();
    for (rank, (i, j_sink)) in scored.into_iter().take(req.top_k).enumerate() {
        let node = &cache.nodes[i];
        let mut witness = vec![node.id.clone()];
        if let Some((nbr, kind)) = &parent_witness[i] {
            witness.push(format!("{kind}:{}", cache.nodes[*nbr as usize].id));
        }
        // Marginal MDL bit gain using `mdl_setup_turn_bits` (smaller/negative = higher compression gain).
        let overlap_terms = terms
            .iter()
            .filter(|t| node.tokens.contains_key(*t))
            .count()
            .max(1);
        let mdl_bits = -mdl_setup_turn_bits(
            node.tokens.len().max(4),
            overlap_terms.saturating_mul(2),
            12,
        ) as i64;
        // Stop admitting pure noise tails once current drops below 2% of peak after slot 3.
        if rank >= 3 && j_sink * 50 < max_j && overlap_terms == 0 {
            break;
        }
        let phase = if conflicts
            .iter()
            .any(|c| c.source_id == node.id || c.target_id == node.id)
        {
            "ac_conflict_held"
        } else if node.labile {
            "labile_drifted_shunt"
        } else {
            "coherent_live_leaf"
        };
        let norm_score = (j_sink as f32) / (max_j as f32);
        hits.push(CircuitHit {
            id: node.id.clone(),
            score: norm_score,
            proof: CircuitProof {
                current_q16: j_sink,
                voltage_q16: voltage[i],
                hebbian_boost_q16: node.hebbian_delta_q,
                mdl_gain_bits: mdl_bits,
                polarity_phase: phase.to_string(),
                path_witness: witness,
                labile_drifted: node.labile,
            },
        });
    }

    let premise_proof = prove_premise_coverage(&terms, &matched_terms_global);
    let back_emf_q16 = if terms.is_empty() {
        0
    } else {
        let missing_ratio = (premise_proof.missing_premises.len() as i64) * Q16_ONE
            / (terms.len() as i64);
        if hits.is_empty() {
            Q16_ONE
        } else {
            missing_ratio
        }
    };
    let status = if hits.is_empty() {
        "abstain_open_circuit"
    } else if has_destructive_conflict {
        "conflict_detected_hold"
    } else if !premise_proof.all_covered {
        "partial_open_circuit"
    } else {
        "grounded"
    };

    CircuitResponse {
        hits,
        conflicts,
        epistemic: EpistemicDiagnostic {
            status: status.to_string(),
            back_emf_q16,
            covered_premises: matched_terms_global.into_iter().collect(),
            missing_premises: premise_proof.missing_premises,
            simd_isa: simd,
        },
        total_indexed_nodes: n,
        zset_delta_count: cache.zset.weights.len(),
        engine: "vestige-ghostcurrent-circuit-v1".to_string(),
    }
}

/// Convert a [`CircuitResponse`] diagnostic block into structured JSON for MCP responses.
pub fn epistemic_json(resp: &CircuitResponse) -> Value {
    json!({
        "engine": resp.engine,
        "status": resp.epistemic.status,
        "backEmfQ16": resp.epistemic.back_emf_q16,
        "coveredPremises": resp.epistemic.covered_premises,
        "missingPremises": resp.epistemic.missing_premises,
        "conflicts": resp.conflicts,
        "indexedNodes": resp.total_indexed_nodes,
        "zsetDeltas": resp.zset_delta_count,
        "simdIsa": resp.epistemic.simd_isa,
    })
}
