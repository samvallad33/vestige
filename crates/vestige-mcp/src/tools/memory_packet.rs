//! Read-only packets for local lifecycle hooks.
//!
//! A prompt is ranked by exact-token BM25 over stored memory text. An empty
//! prompt (session open, no words yet) still returns the highest-salience
//! records. Full record text is packed into about 3_000 tokens. A prompt that
//! names a recorded person does not pull in a different person's records.
use chrono::{Datelike, Days, NaiveDate, Weekday};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use vestige_core::{KnowledgeNode, RerankProvider, Storage};

use super::lexical::{self, HOOK_CHAR_BUDGET};

/// Stores with at most this many live nodes run the full cached `RetrievalEngine`
/// directly; larger stores use the O(1) inverted posting index to pre-filter a
/// sharp deterministic working set before running the exact frozen v7.1 engine.
const FULL_ENGINE_NODE_CAP: usize = 256;
/// Maximum candidate nodes passed into the frozen v7.1 engine for large stores.
const PREFILTER_CANDIDATE_CAP: usize = 160;
/// Cap individual turn text fed into the ranking engine for large stores so
/// multi-KB auto-commit diff dumps never stall hook latency. Returned packets
/// still carry full stored `node.content`.
const LARGE_STORE_TURN_CHAR_CAP: usize = 1_600;

/// Pre-lowercased identity tag record for O(1) prompt matching without per-query allocations.
struct PrecomputedEntityTag {
    raw: String,
    lower: String,
    surface_lower: String,
    leaf_lower: Option<String>,
    leaf_spaced_lower: Option<String>,
}

/// Pre-lowercased `person:` tag record for O(1) prompt matching.
struct PrecomputedPersonTag {
    raw: String,
    lower: String,
    name_lower: String,
}

#[derive(Clone)]
struct PacketSnapshot {
    nodes: Arc<Vec<KnowledgeNode>>,
    pos_by_id: Arc<HashMap<String, usize>>,
    scores: Arc<BTreeMap<String, i128>>,
    receipts: Arc<Vec<Value>>,
    links_by_source: Arc<HashMap<String, Vec<vestige_core::retrieval::TurnLink>>>,
    entity_tags: Arc<Vec<PrecomputedEntityTag>>,
    person_tags: Arc<Vec<PrecomputedPersonTag>>,
    nodes_by_entity: Arc<HashMap<String, Vec<usize>>>,
    by_identity_tag: Arc<BTreeMap<String, Vec<usize>>>,
    postings: Arc<HashMap<String, Vec<(u32, u16)>>>,
    doc_len: Arc<Vec<u32>>,
    avg_dl: f32,
    standing_idx: Arc<Vec<usize>>,
    full_retrieval: Option<Arc<vestige_core::retrieval::RetrievalEngine>>,
}

struct CachedPacketSnapshot {
    stamp: (usize, u64, u64),
    snapshot: PacketSnapshot,
}

fn packet_cache() -> &'static Mutex<HashMap<(PathBuf, String), CachedPacketSnapshot>> {
    static CACHE: OnceLock<Mutex<HashMap<(PathBuf, String), CachedPacketSnapshot>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn is_standing_node(node: &KnowledgeNode) -> bool {
    matches!(node.node_type.as_str(), "decision" | "correction" | "rule")
        || node
            .tags
            .iter()
            .any(|t| t == "pinned" || t == "owner-directive" || t.starts_with("forbid:"))
}

fn truncate_turn_text(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    text.chars().take(max_chars).collect()
}

fn tokenize_for_prefilter(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            for lc in ch.to_lowercase() {
                cur.push(lc);
            }
        } else if !cur.is_empty() {
            if cur.len() >= 2 {
                out.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        }
    }
    if cur.len() >= 2 {
        out.push(cur);
    }
    out
}

fn build_retrieval_engine_subset(
    nodes: &[KnowledgeNode],
    indices: &[usize],
    links_by_source: &HashMap<String, Vec<vestige_core::retrieval::TurnLink>>,
    char_cap: Option<usize>,
) -> vestige_core::retrieval::RetrievalEngine {
    let mut by_day: BTreeMap<String, (i64, Vec<vestige_core::retrieval::Turn>)> = BTreeMap::new();
    for &idx in indices {
        let Some(node) = nodes.get(idx) else {
            continue;
        };
        let day = node.created_at.format("%Y/%m/%d").to_string();
        let ts = node.created_at.timestamp();
        let role = if is_standing_node(node) {
            "user".to_string()
        } else {
            "assistant".to_string()
        };
        let base_text = match char_cap {
            Some(cap) => truncate_turn_text(&node.content, cap),
            None => node.content.clone(),
        };
        let content = if node.tags.is_empty() {
            base_text
        } else {
            format!("{} [{}]", base_text, node.tags.join(" "))
        };
        let entry = by_day.entry(day).or_insert_with(|| (ts, Vec::new()));
        if ts > entry.0 {
            entry.0 = ts;
        }
        entry.1.push(vestige_core::retrieval::Turn {
            id: node.id.clone(),
            role,
            content,
            links: links_by_source.get(&node.id).cloned().unwrap_or_default(),
            stability: node.stability as f32,
        });
    }
    let sessions: Vec<vestige_core::retrieval::Session> = by_day
        .into_iter()
        .map(|(day, (timestamp, turns))| vestige_core::retrieval::Session {
            id: format!("day:{day}"),
            date: day,
            timestamp,
            turns,
        })
        .collect();
    vestige_core::retrieval::RetrievalEngine::new(&sessions, vestige_core::retrieval::Params::default())
}

fn select_prefilter_candidates(
    snapshot: &PacketSnapshot,
    prompt: &str,
    entities: &BTreeSet<String>,
    index_hits: &BTreeSet<String>,
    has_day_hits: bool,
) -> Vec<usize> {
    let n = snapshot.nodes.len();
    let mut bm25: HashMap<usize, f32> = HashMap::new();
    let q_terms: HashSet<String> = tokenize_for_prefilter(prompt).into_iter().collect();
    let n_f = (n.max(1)) as f32;
    let avg_dl = snapshot.avg_dl.max(1.0);
    for term in &q_terms {
        let Some(post) = snapshot.postings.get(term) else {
            continue;
        };
        let df = post.len() as f32;
        let idf = ((n_f - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.01);
        for &(doc_u32, tf_u16) in post {
            let idx = doc_u32 as usize;
            let tf = tf_u16 as f32;
            let dl = snapshot.doc_len.get(idx).copied().unwrap_or(1) as f32;
            let score = idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (dl / avg_dl)));
            *bm25.entry(idx).or_insert(0.0) += score;
        }
    }

    let mut ranked_bm25: Vec<(usize, f32)> = bm25.into_iter().collect();
    ranked_bm25.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut chosen: Vec<usize> = Vec::with_capacity(PREFILTER_CANDIDATE_CAP);
    let mut seen: HashSet<usize> = HashSet::with_capacity(PREFILTER_CANDIDATE_CAP * 2);
    let mut push = |idx: usize| -> bool {
        if idx < n && chosen.len() < PREFILTER_CANDIDATE_CAP && seen.insert(idx) {
            chosen.push(idx);
        }
        chosen.len() >= PREFILTER_CANDIDATE_CAP
    };

    // 1. Exact entity tag matches (highest precision).
    for tag in entities {
        if let Some(idxs) = snapshot.nodes_by_entity.get(tag) {
            for &idx in idxs.iter().take(24) {
                push(idx);
            }
        }
    }
    // 2. Top exact-token BM25 matches.
    for (idx, _) in ranked_bm25.into_iter().take(96) {
        push(idx);
    }
    // 3. Date/avoid index-joined memories (bounded scan).
    if !index_hits.is_empty() {
        for (idx, node) in snapshot.nodes.iter().enumerate() {
            if joins_index(node, index_hits, has_day_hits) && push(idx) {
                break;
            }
        }
    }
    // 4. Pinned / standing rules (most recent first; `nodes` is newest-first from `page_nodes`).
    for &idx in snapshot.standing_idx.iter().take(24) {
        push(idx);
    }
    // 5. Most recent live memories for temporal grounding.
    for idx in 0..n.min(24) {
        push(idx);
    }
    chosen
}

fn fast_bridges(
    nodes: &[KnowledgeNode],
    by_tag: &BTreeMap<String, Vec<usize>>,
    seed_indexes: &[usize],
    limit: usize,
) -> Vec<lexical::TagBridge> {
    if limit == 0 || seed_indexes.is_empty() {
        return Vec::new();
    }
    const IDENTITY_PREFIXES: &[&str] = &["person:", "place:", "food:", "file:", "service:", "symbol:"];
    let is_id_tag = |tag: &str| IDENTITY_PREFIXES.iter().any(|p| tag.starts_with(p));
    let mut start: BTreeSet<&str> = BTreeSet::new();
    for &idx in seed_indexes {
        if let Some(node) = nodes.get(idx) {
            for tag in &node.tags {
                if is_id_tag(tag) {
                    start.insert(tag.as_str());
                }
            }
        }
    }
    if start.is_empty() {
        return Vec::new();
    }
    let seed_set: HashSet<usize> = seed_indexes.iter().copied().collect();
    let mut direct: Vec<(usize, &str)> = Vec::new();
    let mut direct_seen: HashSet<usize> = HashSet::new();
    for &tag in &start {
        if let Some(list) = by_tag.get(tag) {
            for &idx in list {
                if !seed_set.contains(&idx) && direct_seen.insert(idx) {
                    direct.push((idx, tag));
                    if direct.len() >= 64 {
                        break;
                    }
                }
            }
        }
    }
    let mut found: Vec<lexical::TagBridge> = Vec::new();
    let mut found_seen: HashSet<usize> = HashSet::new();
    for (middle, shared) in direct {
        let Some(mid_node) = nodes.get(middle) else {
            continue;
        };
        for onward in &mid_node.tags {
            if !is_id_tag(onward) || start.contains(onward.as_str()) {
                continue;
            }
            if let Some(fars) = by_tag.get(onward.as_str()) {
                for &far in fars {
                    if far == middle || seed_set.contains(&far) || !found_seen.insert(far) {
                        continue;
                    }
                    if nodes[far]
                        .tags
                        .iter()
                        .any(|t| start.contains(t.as_str()))
                    {
                        continue;
                    }
                    found.push(lexical::TagBridge {
                        index: far,
                        path: [
                            shared.to_string(),
                            mid_node.id.clone(),
                            onward.clone(),
                        ],
                    });
                    if found.len() >= limit {
                        return found;
                    }
                }
            }
        }
    }
    found
}

fn load_cached_snapshot(storage: &Storage, scope: &str) -> Result<PacketSnapshot, String> {
    let stamp = crate::circuit_index::fast_store_stamp(storage);
    let key = (storage.db_path().to_path_buf(), scope.to_string());
    if stamp.0 > 0
        && let Ok(guard) = packet_cache().lock()
        && let Some(hit) = guard.get(&key)
        && hit.stamp == stamp
    {
        return Ok(hit.snapshot.clone());
    }
    let raw_nodes = storage
        .get_all_nodes_in_scope(scope, i32::MAX, 0)
        .map_err(|e| e.to_string())?;
    let retired = storage.superseded_node_ids().map_err(|e| e.to_string())?;
    let live_nodes: Vec<KnowledgeNode> = raw_nodes
        .into_iter()
        .filter(|n| !retired.contains(&n.id))
        .collect();
    let receipts = Arc::new(storage.salience_receipts().map_err(|e| e.to_string())?);
    let score_map = scores_from_receipts(receipts.as_ref());

    let mut links_by_source: HashMap<String, Vec<vestige_core::retrieval::TurnLink>> =
        HashMap::new();
    for edge in storage.get_all_connections().unwrap_or_default() {
        links_by_source
            .entry(edge.source_id.clone())
            .or_default()
            .push(vestige_core::retrieval::TurnLink {
                kind: edge.link_type,
                to: edge.target_id,
                degree: 0,
            });
    }

    let mut pos_by_id: HashMap<String, usize> = HashMap::with_capacity(live_nodes.len());
    let mut entity_seen: HashSet<String> = HashSet::new();
    let mut entity_tags: Vec<PrecomputedEntityTag> = Vec::new();
    let mut person_seen: HashSet<String> = HashSet::new();
    let mut person_tags: Vec<PrecomputedPersonTag> = Vec::new();
    let mut nodes_by_entity: HashMap<String, Vec<usize>> = HashMap::new();
    let mut by_identity_tag: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut postings: HashMap<String, Vec<(u32, u16)>> = HashMap::new();
    let mut doc_len: Vec<u32> = Vec::with_capacity(live_nodes.len());
    let mut total_tokens: u64 = 0;
    let mut standing_idx: Vec<usize> = Vec::new();
    const IDENTITY_PREFIXES: &[&str] = &["person:", "place:", "food:", "file:", "service:", "symbol:"];

    for (idx, node) in live_nodes.iter().enumerate() {
        pos_by_id.insert(node.id.clone(), idx);
        if is_standing_node(node) {
            standing_idx.push(idx);
        }
        for tag in &node.tags {
            if IDENTITY_PREFIXES.iter().any(|p| tag.starts_with(p)) {
                nodes_by_entity.entry(tag.clone()).or_default().push(idx);
                by_identity_tag.entry(tag.clone()).or_default().push(idx);
                if entity_seen.insert(tag.clone()) {
                    let lower = tag.to_lowercase();
                    let surface_lower = lower
                        .split_once(':')
                        .map(|(_, s)| s.to_string())
                        .unwrap_or_default();
                    let (leaf_lower, leaf_spaced_lower) = if lower.starts_with("symbol:")
                        && let Some((_, leaf)) = lower.rsplit_once(':')
                        && !leaf.is_empty()
                    {
                        let spaced = leaf.contains('~').then(|| leaf.replace('~', " "));
                        (Some(leaf.to_string()), spaced)
                    } else {
                        (None, None)
                    };
                    entity_tags.push(PrecomputedEntityTag {
                        raw: tag.clone(),
                        lower,
                        surface_lower,
                        leaf_lower,
                        leaf_spaced_lower,
                    });
                }
                if let Some(name) = tag.strip_prefix("person:")
                    && !name.is_empty()
                    && person_seen.insert(tag.clone())
                {
                    person_tags.push(PrecomputedPersonTag {
                        raw: tag.clone(),
                        lower: tag.to_lowercase(),
                        name_lower: name.to_lowercase(),
                    });
                }
            }
        }
        // Build compact inverted postings over node content (capped at 4 KiB per doc) + tags.
        let mut tf: HashMap<String, u16> = HashMap::new();
        let slice = if node.content.len() > 4096 {
            let mut end = 4096;
            while end > 0 && !node.content.is_char_boundary(end) {
                end -= 1;
            }
            &node.content[..end]
        } else {
            node.content.as_str()
        };
        let mut dl = 0u32;
        for tok in tokenize_for_prefilter(slice) {
            dl = dl.saturating_add(1);
            let e = tf.entry(tok).or_insert(0);
            *e = e.saturating_add(1);
        }
        for tag in &node.tags {
            for tok in tokenize_for_prefilter(tag) {
                dl = dl.saturating_add(1);
                let e = tf.entry(tok).or_insert(0);
                *e = e.saturating_add(2);
            }
        }
        doc_len.push(dl);
        total_tokens = total_tokens.saturating_add(u64::from(dl));
        for (tok, count) in tf {
            postings.entry(tok).or_default().push((idx as u32, count));
        }
    }
    let avg_dl = if live_nodes.is_empty() {
        1.0
    } else {
        (total_tokens as f32 / live_nodes.len() as f32).max(1.0)
    };

    let full_retrieval = if live_nodes.len() <= FULL_ENGINE_NODE_CAP {
        let all_idx: Vec<usize> = (0..live_nodes.len()).collect();
        Some(Arc::new(build_retrieval_engine_subset(
            &live_nodes,
            &all_idx,
            &links_by_source,
            None,
        )))
    } else {
        None
    };

    let snapshot = PacketSnapshot {
        nodes: Arc::new(live_nodes),
        pos_by_id: Arc::new(pos_by_id),
        scores: Arc::new(score_map),
        receipts,
        links_by_source: Arc::new(links_by_source),
        entity_tags: Arc::new(entity_tags),
        person_tags: Arc::new(person_tags),
        nodes_by_entity: Arc::new(nodes_by_entity),
        by_identity_tag: Arc::new(by_identity_tag),
        postings: Arc::new(postings),
        doc_len: Arc::new(doc_len),
        avg_dl,
        standing_idx: Arc::new(standing_idx),
        full_retrieval,
    };
    if stamp.0 > 0
        && let Ok(mut guard) = packet_cache().lock()
    {
        if guard.len() >= 16 {
            guard.clear();
        }
        guard.insert(
            key,
            CachedPacketSnapshot {
                stamp,
                snapshot: snapshot.clone(),
            },
        );
    }
    Ok(snapshot)
}

pub(crate) fn scores_from_receipts(receipts: &[Value]) -> BTreeMap<String, i128> {
    let mut scores = BTreeMap::new();
    for row in receipts {
        if let (Some(id), Some(old), Some(new)) = (
            row["memory_id"].as_str(),
            row["old_stability_q"].as_i64(),
            row["new_stability_q"].as_i64(),
        ) {
            *scores.entry(id.to_owned()).or_default() += i128::from(new) - i128::from(old);
        }
    }
    scores
}

pub(crate) fn scores(storage: &Storage) -> BTreeMap<String, i128> {
    let receipts = storage.salience_receipts().unwrap_or_default();
    scores_from_receipts(&receipts)
}

pub(crate) fn compare(
    a: &KnowledgeNode,
    b: &KnowledgeNode,
    scores: &BTreeMap<String, i128>,
) -> std::cmp::Ordering {
    scores
        .get(&b.id)
        .copied()
        .unwrap_or_default()
        .cmp(&scores.get(&a.id).copied().unwrap_or_default())
        .then_with(|| b.stability.total_cmp(&a.stability))
        .then_with(|| b.created_at.cmp(&a.created_at))
        .then_with(|| b.id.cmp(&a.id))
}

fn phrase_lower(text_lower: &str, surface_lower: &str) -> bool {
    if surface_lower.is_empty() {
        return false;
    }
    text_lower.match_indices(surface_lower).any(|(at, _)| {
        let word = |c: char| c.is_alphanumeric() || c == '_';
        text_lower[..at].chars().next_back().is_none_or(|c| !word(c))
            && text_lower[at + surface_lower.len()..]
                .chars()
                .next()
                .is_none_or(|c| !word(c))
    })
}

/// salience-pro 0.3.7 `thing_words`: lower case, each plural read as its
/// singular, so "Hot Tubs" and "hot tub" meet as `hot-tub`.
fn thing_words(raw: &str) -> Vec<String> {
    raw.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| {
            let w = w.to_lowercase();
            let n = w.chars().count();
            if n > 4
                && let Some(stem) = w.strip_suffix("ies")
            {
                format!("{stem}y")
            } else if n > 4
                && let Some(stem) = w.strip_suffix("oes")
            {
                format!("{stem}o")
            } else if n > 3
                && !w.ends_with("ss")
                && !w.ends_with("us")
                && let Some(stem) = w.strip_suffix('s')
            {
                stem.to_string()
            } else {
                w
            }
        })
        .collect()
}

/// Every `avoid:` key a prompt phrase could meet: runs of one to three words.
fn thing_keys(raw: &str) -> BTreeSet<String> {
    let words = thing_words(raw);
    let mut keys = BTreeSet::new();
    for len in 1..=3 {
        for run in words.windows(len) {
            let key = run.join("-");
            if key.chars().count() >= 3 {
                keys.insert(key);
            }
        }
    }
    keys
}

const MONTHS: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

/// Parse an exact `YYYY-MM-DD` or `YYYY/MM/DD` token into a [`NaiveDate`].
fn parse_iso_or_slash_date(word: &str) -> Option<NaiveDate> {
    let sep = if word.contains('-') { '-' } else { '/' };
    let parts: Vec<&str> = word.split(sep).collect();
    let [y, m, d] = parts.as_slice() else {
        return None;
    };
    if y.len() != 4
        || m.len() != 2
        || d.len() != 2
        || ![y, m, d]
            .iter()
            .all(|s| s.bytes().all(|b| b.is_ascii_digit()))
    {
        return None;
    }
    let year = y.parse::<i32>().ok()?;
    let month = m.parse::<u32>().ok()?;
    let day = d.parse::<u32>().ok()?;
    NaiveDate::from_ymd_opt(year, month, day)
}

fn insert_date_keys(tags: &mut BTreeSet<String>, date: NaiveDate) {
    let y = date.year();
    let m = date.month();
    let d = date.day();
    tags.insert(format!("day:{y:04}-{m:02}-{d:02}"));
    tags.insert(format!("date:{y:04}-{m:02}-{d:02}"));
    tags.insert(format!("date:{y:04}/{m:02}/{d:02}"));
    tags.insert(format!("yearly:{m:02}-{d:02}"));
}

fn word_number(token: &str) -> Option<u64> {
    if let Ok(n) = token.parse::<u64>() {
        return (1..=365).contains(&n).then_some(n);
    }
    match token {
        "a" | "an" | "one" => Some(1),
        "couple" | "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        _ => None,
    }
}

fn parse_weekday(token: &str) -> Option<Weekday> {
    match token {
        "monday" => Some(Weekday::Mon),
        "tuesday" => Some(Weekday::Tue),
        "wednesday" => Some(Weekday::Wed),
        "thursday" => Some(Weekday::Thu),
        "friday" => Some(Weekday::Fri),
        "saturday" => Some(Weekday::Sat),
        "sunday" => Some(Weekday::Sun),
        _ => None,
    }
}

/// Fast exact ISO/slash date extractor (skips `thing_keys` 1..3-gram allocation).
fn content_matches_day_hit(content: &str, hits: &BTreeSet<String>) -> bool {
    for raw in content.split(|c: char| !(c.is_ascii_digit() || c == '-' || c == '/')) {
        if raw.len() == 10
            && let Some(date) = parse_iso_or_slash_date(raw)
        {
            let key = format!("day:{:04}-{:02}-{:02}", date.year(), date.month(), date.day());
            if hits.contains(&key) {
                return true;
            }
        }
    }
    false
}

/// A memory joins the prompt's index tags through its stored tags or, for a
/// memory saved before index tags existed, through an exact ISO/slash day its
/// own words or `date:` tags carry. Avoid and yearly keys only come from
/// stored tags: a word in a memory is not a rule against it, and a one-off
/// date does not recur.
fn joins_index(node: &KnowledgeNode, hits: &BTreeSet<String>, has_day_hits: bool) -> bool {
    if hits.is_empty() {
        return false;
    }
    let tag_hit = node.tags.iter().any(|tag| {
        hits.contains(tag)
            || (has_day_hits
                && (tag.starts_with("date:") || tag.starts_with("day:"))
                && content_matches_day_hit(tag, hits))
    });
    tag_hit || (has_day_hits && content_matches_day_hit(&node.content, hits))
}

/// Index-joined memories reserved ahead of lexical hits (salience-pro MAX_FILED_CANDIDATES).
const MAX_RESERVED: usize = 4;

/// Index tags a prompt meets exactly: `avoid:<thing>`, `day:YYYY-MM-DD`,
/// `date:YYYY-MM-DD` / `date:YYYY/MM/DD`, `yearly:MM-DD`, and relative date
/// windows anchored to the prompt's date (or today). Deterministic; no model
/// reads the prompt.
pub(crate) fn index_tags(prompt: &str) -> BTreeSet<String> {
    let mut tags: BTreeSet<String> = thing_keys(prompt)
        .into_iter()
        .map(|key| format!("avoid:{key}"))
        .collect();
    let lower = prompt.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !(c.is_alphanumeric() || c == '-' || c == '/'))
        .filter(|w| !w.is_empty())
        .collect();
    let mut explicit_dates = Vec::new();
    for (i, word) in words.iter().enumerate() {
        if let Some(date) = parse_iso_or_slash_date(word) {
            insert_date_keys(&mut tags, date);
            explicit_dates.push(date);
        }
        if let Some(month) = MONTHS.iter().position(|name| name == word) {
            let day = words.get(i + 1).map(|w| {
                w.trim_end_matches(|c: char| c.is_alphabetic())
                    .parse::<u32>()
                    .unwrap_or(0)
            });
            if let Some(day @ 1..=31) = day {
                tags.insert(format!("yearly:{:02}-{day:02}", month + 1));
            }
        }
    }

    // Do not expand relative offsets when the question is asking "how many days/weeks/months ago"
    let asks_how_long_ago = words.windows(4).any(|w| {
        w[0] == "how"
            && matches!(w[1], "many" | "long")
            && matches!(
                w[2],
                "day" | "days" | "week" | "weeks" | "month" | "months" | "year" | "years"
            )
            && matches!(w[3], "ago" | "passed" | "have" | "had")
    });
    if !asks_how_long_ago {
        let anchor = explicit_dates
            .first()
            .copied()
            .unwrap_or_else(|| chrono::Utc::now().date_naive());
        for (i, word) in words.iter().enumerate() {
            if *word == "yesterday"
                && let Some(d) = anchor.checked_sub_days(Days::new(1))
            {
                tags.insert(format!("day:{}", d.format("%Y-%m-%d")));
            }
            if *word == "ago" && i >= 2 {
                let unit = words[i - 1];
                let raw_n = if i >= 4 && words[i - 3] == "couple" && words[i - 2] == "of" {
                    Some(2)
                } else {
                    word_number(words[i - 2])
                };
                if let Some(n) = raw_n {
                    let center_tol = if unit.starts_with("day") {
                        Some((n, 1u64))
                    } else if unit.starts_with("week") {
                        Some((7 * n, 3u64))
                    } else if unit.starts_with("month") {
                        Some((30 * n, 5u64))
                    } else {
                        None
                    };
                    if let Some((days_back, tol)) = center_tol
                        && let Some(center) = anchor.checked_sub_days(Days::new(days_back))
                    {
                        for delta in 0..=tol {
                            if let Some(d) = center.checked_sub_days(Days::new(delta)) {
                                tags.insert(format!("day:{}", d.format("%Y-%m-%d")));
                            }
                            if let Some(d) = center.checked_add_days(Days::new(delta)) {
                                tags.insert(format!("day:{}", d.format("%Y-%m-%d")));
                            }
                        }
                    }
                }
            }
            if matches!(*word, "last" | "past" | "previous")
                && let Some(&next) = words.get(i + 1)
            {
                if let Some(target_wd) = parse_weekday(next) {
                    let cur = anchor.weekday().num_days_from_monday();
                    let tgt = target_wd.num_days_from_monday();
                    let diff = (cur + 7 - tgt) % 7;
                    let diff = if diff == 0 { 7 } else { diff };
                    if let Some(center) = anchor.checked_sub_days(Days::new(u64::from(diff))) {
                        for delta in 0..=1u64 {
                            if let Some(d) = center.checked_sub_days(Days::new(delta)) {
                                tags.insert(format!("day:{}", d.format("%Y-%m-%d")));
                            }
                            if let Some(d) = center.checked_add_days(Days::new(delta)) {
                                tags.insert(format!("day:{}", d.format("%Y-%m-%d")));
                            }
                        }
                    }
                } else if next == "weekend" {
                    for back in 1..=7u64 {
                        if let Some(d) = anchor.checked_sub_days(Days::new(back))
                            && matches!(d.weekday(), Weekday::Sat | Weekday::Sun)
                        {
                            tags.insert(format!("day:{}", d.format("%Y-%m-%d")));
                        }
                    }
                }
            }
        }
    }
    tags
}

pub(crate) fn packet(storage: &Storage, scope: &str, args: &Value) -> Result<Value, String> {
    let prompt = args["prompt"].as_str().unwrap_or_default();
    if prompt.len() > 64 * 1024 {
        return Err("hook prompt exceeds 64 KiB".into());
    }
    // Fast-path empty prompts (session open): zero retrieval build, zero full-store clone.
    if prompt.trim().is_empty() {
        let count = crate::strata_memory::read_store(storage, |s| {
            s.node_map()
                .values()
                .filter(|r| r.is_live() && r.scope == scope)
                .count()
        })
        .unwrap_or_else(|| {
            storage
                .get_all_nodes_in_scope(scope, i32::MAX, 0)
                .map(|v| v.len())
                .unwrap_or(0)
        });
        return Ok(json!({
            "memory_count": count,
            "entities": [],
            "memories": [],
            "bridges": [],
            "backfill_receipts": [],
            "backfill_autofire": super::backfill::autofire_enabled(),
            "selection": "none_empty_prompt",
            "token_budget": lexical::HOOK_TOKEN_BUDGET,
            "ranking_signal": "none",
            "circuit": Value::Null,
        }));
    }
    let snapshot = load_cached_snapshot(storage, scope)?;
    let now = chrono::Utc::now();
    let nodes: &[KnowledgeNode] = snapshot.nodes.as_ref();
    let is_valid_now = |n: &KnowledgeNode| {
        n.valid_from.is_none_or(|t| t <= now) && n.valid_until.is_none_or(|t| t > now)
    };
    let count = nodes.iter().filter(|n| is_valid_now(n)).count();
    let scores = snapshot.scores.as_ref();
    let tie = |node: &KnowledgeNode| scores.get(&node.id).copied().unwrap_or_default();
    let requested = args["limit"].as_u64().unwrap_or(8).clamp(1, 24) as usize;
    let prompt_lower = prompt.to_lowercase();
    let named: BTreeSet<String> = snapshot
        .person_tags
        .iter()
        .filter(|p| {
            phrase_lower(&prompt_lower, &p.lower) || phrase_lower(&prompt_lower, &p.name_lower)
        })
        .map(|p| p.raw.clone())
        .collect();
    let entities: BTreeSet<String> = snapshot
        .entity_tags
        .iter()
        .filter(|e| {
            phrase_lower(&prompt_lower, &e.lower)
                || phrase_lower(&prompt_lower, &e.surface_lower)
                || e.leaf_lower
                    .as_deref()
                    .is_some_and(|leaf| phrase_lower(&prompt_lower, leaf))
                || e.leaf_spaced_lower
                    .as_deref()
                    .is_some_and(|spaced| phrase_lower(&prompt_lower, spaced))
        })
        .map(|e| e.raw.clone())
        .collect();

    let mut index_hits = index_tags(prompt);
    index_hits.extend(
        named
            .iter()
            .filter_map(|tag| tag.strip_prefix("person:"))
            .map(|name| format!("facts-of:{name}")),
    );
    let has_day_hits = index_hits
        .iter()
        .any(|t| t.starts_with("day:") || t.starts_with("date:"));

    let tr_pool = requested.saturating_mul(6).max(32);
    let today = now.format("%Y/%m/%d").to_string();
    let query = vestige_core::retrieval::Query {
        text: prompt.to_string(),
        date: today.clone(),
    };
    let (rank_res, turns_indexed) = if let Some(full) = snapshot.full_retrieval.as_ref() {
        (full.query(&query, tr_pool), full.turns_indexed())
    } else {
        let candidate_indices =
            select_prefilter_candidates(&snapshot, prompt, &entities, &index_hits, has_day_hits);
        let subset_engine = build_retrieval_engine_subset(
            nodes,
            &candidate_indices,
            snapshot.links_by_source.as_ref(),
            Some(LARGE_STORE_TURN_CHAR_CAP),
        );
        let turns = subset_engine.turns_indexed();
        (subset_engine.query(&query, tr_pool), turns)
    };
    let pos_by_id = snapshot.pos_by_id.as_ref();
    let reranker = vestige_core::active_rerank_provider();
    let mut rerank_latency_us = 0u64;
    let mut rerank_used_reasoning = false;
    let ranked_turns: Vec<(String, f64)> = if reranker.provider_id()
        == vestige_core::IdentityRerankProvider.provider_id()
        || rank_res.turns.len() <= 1
    {
        rank_res
            .turns
            .iter()
            .map(|t| (t.id.clone(), f64::from(t.score)))
            .collect()
    } else {
        let candidates: Vec<vestige_core::RerankCandidate> = rank_res
            .turns
            .iter()
            .enumerate()
            .filter_map(|(rank, item)| {
                let &idx = pos_by_id.get(item.id.as_str())?;
                let node = &nodes[idx];
                if !is_valid_now(node) {
                    return None;
                }
                Some(vestige_core::RerankCandidate {
                    id: item.id.clone(),
                    text: node.content.clone(),
                    session_id: None,
                    timestamp: Some(node.created_at.to_rfc3339()),
                    base_score: f64::from(item.score),
                    base_rank: rank,
                })
            })
            .collect();
        let req = vestige_core::RerankRequest {
            query: prompt.to_string(),
            query_date: Some(today),
            candidates,
            top_k: tr_pool,
            lock_slot0: true,
        };
        let mut resp = reranker.rerank(&req);
        vestige_core::enforce_slot0_lock(&req, &mut resp);
        rerank_latency_us = resp.latency_us;
        rerank_used_reasoning = resp.used_reasoning;
        resp.ranked
            .into_iter()
            .map(|r| (r.id, r.calibrated_score))
            .collect()
    };
    let circuit_diag = json!({
        "engine": vestige_core::retrieval::ENGINE,
        "turns_indexed": turns_indexed,
        "premise_coverage": rank_res.premise_coverage,
        "standing_rules_count": rank_res.standing_rules.len(),
        "rerank_provider": reranker.provider_id(),
        "rerank_latency_us": rerank_latency_us,
        "rerank_used_reasoning": rerank_used_reasoning,
    });
    let standing_rule_indexes: HashSet<usize> = rank_res
        .standing_rules
        .iter()
        .filter_map(|item| pos_by_id.get(item.id.as_str()).copied())
        .filter(|&idx| is_valid_now(&nodes[idx]))
        .collect();
    let mut hits: Vec<lexical::LexicalHit> = Vec::new();
    let mut seen_idx: HashSet<usize> = HashSet::new();
    for (id, score) in rank_res
        .standing_rules
        .iter()
        .map(|item| (item.id.as_str(), f64::from(item.score)))
        .chain(ranked_turns.iter().map(|(id, score)| (id.as_str(), *score)))
    {
        let Some(&index) = pos_by_id.get(id) else {
            continue;
        };
        if !is_valid_now(&nodes[index]) || !seen_idx.insert(index) {
            continue;
        }
        let score_micro = (score * 1_000_000.0).round() as i64;
        if score_micro > 0 || standing_rule_indexes.contains(&index) {
            hits.push(lexical::LexicalHit {
                index,
                score_micro: score_micro.max(1),
            });
        }
    }
    hits.sort_by(|a, b| {
        b.score_micro
            .cmp(&a.score_micro)
            .then_with(|| tie(&nodes[b.index]).cmp(&tie(&nodes[a.index])))
            .then_with(|| nodes[b.index].id.cmp(&nodes[a.index].id))
    });
    let score_map: HashMap<&str, i64> = hits
        .iter()
        .map(|h| (nodes[h.index].id.as_str(), h.score_micro))
        .collect();
    let score_of = |id: &str| score_map.get(id).copied().unwrap_or(0);
    let mut indexes: Vec<usize> = hits
        .iter()
        .map(|hit| hit.index)
        .filter(|index| lexical::allows_person(&nodes[*index], &named))
        .collect();
    let mut seen_out: HashSet<usize> = indexes.iter().copied().collect();
    // Exact entity matches via O(1) `nodes_by_entity` map.
    for tag in &entities {
        if let Some(idxs) = snapshot.nodes_by_entity.get(tag) {
            for &index in idxs {
                if seen_out.insert(index)
                    && is_valid_now(&nodes[index])
                    && lexical::allows_person(&nodes[index], &named)
                {
                    indexes.push(index);
                }
            }
        }
    }
    // Date/avoid/facts-of index-joined memories.
    if !index_hits.is_empty() {
        for (index, node) in nodes.iter().enumerate() {
            if !seen_out.contains(&index)
                && is_valid_now(node)
                && joins_index(node, &index_hits, has_day_hits)
                && lexical::allows_person(node, &named)
            {
                seen_out.insert(index);
                indexes.push(index);
            }
        }
    }
    let top_score = hits.iter().map(|hit| hit.score_micro).max().unwrap_or(0);
    let on_target = |index: &usize| {
        let node = &nodes[*index];
        joins_index(node, &index_hits, has_day_hits)
            || node.tags.iter().any(|tag| entities.contains(tag))
            || (top_score > 0 && score_of(&node.id) * 2 >= top_score)
    };
    let raised = |index: &usize| tie(&nodes[*index]) > 0 && on_target(index);
    let lowered = |index: &usize| tie(&nodes[*index]) < 0;
    let mut reserved: HashSet<usize> = HashSet::new();
    let mut joined: Vec<usize> = indexes
        .iter()
        .copied()
        .filter(|index| joins_index(&nodes[*index], &index_hits, has_day_hits))
        .collect();
    let guarded = |index: &usize| {
        nodes[*index].tags.iter().any(|tag| {
            ["allergy:", "health:", "condition:", "medication:", "avoid:"]
                .iter()
                .any(|prefix| tag.starts_with(prefix))
        })
    };
    joined.sort_by(|a, b| {
        guarded(b)
            .cmp(&guarded(a))
            .then_with(|| compare(&nodes[*a], &nodes[*b], scores))
    });
    reserved.extend(joined.into_iter().take(MAX_RESERVED));
    reserved.extend(standing_rule_indexes.iter().copied());
    indexes.sort_by(|a, b| {
        raised(b)
            .cmp(&raised(a))
            .then_with(|| reserved.contains(b).cmp(&reserved.contains(a)))
            .then_with(|| lowered(a).cmp(&lowered(b)))
            .then_with(|| score_of(&nodes[*b].id).cmp(&score_of(&nodes[*a].id)))
            .then_with(|| compare(&nodes[*a], &nodes[*b], scores))
    });
    let packed_hits: Vec<lexical::LexicalHit> = indexes
        .into_iter()
        .map(|index| lexical::LexicalHit {
            index,
            score_micro: 1,
        })
        .collect();
    let selected = lexical::pack(nodes, &packed_hits, HOOK_CHAR_BUDGET, requested);
    let selection = if !entities.is_empty() {
        "exact_recorded_entities"
    } else if hits.is_empty() {
        "top_salience"
    } else {
        "stara_total_recall_v2"
    };
    let ranking = vestige_core::retrieval::ENGINE;
    let ids: BTreeSet<&str> = selected.iter().map(|n| n.id.as_str()).collect();
    let seed_indexes: Vec<usize> = selected
        .iter()
        .filter_map(|node| pos_by_id.get(&node.id).copied())
        .collect();
    let bridge_rows = fast_bridges(nodes, snapshot.by_identity_tag.as_ref(), &seed_indexes, 4);
    let bridges: Vec<Value> = bridge_rows
        .iter()
        .map(|bridge| {
            let node = &nodes[bridge.index];
            json!({
                "id": node.id,
                "content": node.content,
                "tags": node.tags,
                "path": bridge.path,
            })
        })
        .collect();
    let receipts: Vec<Value> = snapshot
        .receipts
        .iter()
        .rev()
        .filter(|r| r["memory_id"].as_str().is_some_and(|id| ids.contains(id)))
        .take(8)
        .map(|receipt| {
            json!({
                "receipt_id": receipt["receipt_id"],
                "memory_id": receipt["memory_id"],
                "trigger_id": receipt["trigger_id"],
                "reason": receipt["reason"],
                "direction": raise_direction(receipt),
                "old_score": receipt["old_score"],
                "new_score": receipt["new_score"],
                "old_stability_q": receipt["old_stability_q"],
                "new_stability_q": receipt["new_stability_q"],
            })
        })
        .collect();
    Ok(json!({
        "memory_count": count,
        "entities": entities,
        "memories": selected.iter().map(|n| json!({
            "id": n.id,
            "content": n.content,
            "tags": n.tags,
            "stability": n.stability,
            "salience_delta_q": scores.get(&n.id).copied().unwrap_or_default().to_string(),
        })).collect::<Vec<_>>(),
        "bridges": bridges,
        "backfill_receipts": receipts,
        "backfill_autofire": super::backfill::autofire_enabled(),
        "selection": selection,
        "token_budget": lexical::HOOK_TOKEN_BUDGET,
        "ranking_signal": ranking,
        "circuit": circuit_diag,
    }))
}

fn raise_direction(receipt: &Value) -> &'static str {
    let raised = match (
        receipt["new_stability_q"].as_i64(),
        receipt["old_stability_q"].as_i64(),
    ) {
        (Some(new), Some(old)) => new > old,
        _ => receipt["new_score"].as_f64() > receipt["old_score"].as_f64(),
    };
    if raised { "raise" } else { "lower" }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn an_untagged_memory_joins_by_the_dates_in_its_words() {
        let mut node = KnowledgeNode::default();
        node.content = "Dentist on 2026-11-03".into();
        assert!(joins_index(&node, &index_tags("what is on 2026-11-03"), true));
        assert!(!joins_index(&node, &index_tags("plans for November 3"), false));
        node.content = "hot tubs are fun".into();
        assert!(!joins_index(&node, &index_tags("book a hot tub"), false));
    }

    #[test]
    fn index_tags_join_things_and_dates_exactly() {
        let tags = index_tags("Can we try the Hot Tubs on March 3rd or 2026-11-03?");
        assert!(tags.contains("avoid:hot-tub"));
        assert!(tags.contains("avoid:hot"));
        assert!(tags.contains("yearly:03-03"));
        assert!(tags.contains("day:2026-11-03"));
        assert!(tags.contains("yearly:11-03"));
        assert!(!index_tags("strawberries").contains("avoid:strawberrie"));
        assert!(index_tags("strawberries").contains("avoid:strawberry"));
        assert!(!index_tags("march").iter().any(|t| t.starts_with("yearly:")));
    }

    #[test]
    fn entity_surfaces_use_case_insensitive_whole_boundaries() {
        let entity_matches_prompt_lower = |prompt_lower: &str, tag: &str| -> bool {
            let tag_lower = tag.to_lowercase();
            if phrase_lower(prompt_lower, &tag_lower)
                || tag_lower
                    .split_once(':')
                    .is_some_and(|(_, surface)| phrase_lower(prompt_lower, surface))
            {
                return true;
            }
            if tag_lower.starts_with("symbol:")
                && let Some((_, leaf)) = tag_lower.rsplit_once(':')
                && !leaf.is_empty()
            {
                if phrase_lower(prompt_lower, leaf) {
                    return true;
                }
                if leaf.contains('~') && phrase_lower(prompt_lower, &leaf.replace('~', " ")) {
                    return true;
                }
            }
            false
        };
        let phrase = |text: &str, surface: &str| phrase_lower(&text.to_lowercase(), &surface.to_lowercase());
        assert!(phrase("Where should MAYA go?", "Maya"));
        assert!(!phrase("Mayakovsky", "Maya"));
        assert!(phrase("try Bangkok Garden?", "Bangkok Garden"));
        assert!(!phrase("Bangkok Gardens", "Bangkok Garden"));
        assert!(entity_matches_prompt_lower(
            &"What recipe did I try?".to_lowercase(),
            "symbol:demo:recipe"
        ));
        assert!(entity_matches_prompt_lower(
            &"How is Tokyo transit?".to_lowercase(),
            "symbol:demo:tokyo~transit"
        ));
        assert!(!entity_matches_prompt_lower(
            &"What recipe did I try?".to_lowercase(),
            "symbol:demo:garden"
        ));
    }

    #[test]
    fn slash_dates_date_tags_and_relative_offsets_join_index() {
        let mut node = KnowledgeNode::default();
        node.tags = vec!["date:2023/03/15 (Wed) 11:56".into()];
        let hits = index_tags("Current Date: 2023/03/25. What appliance did I buy 10 days ago?");
        assert!(hits.contains("day:2023-03-15"), "{hits:?}");
        assert!(joins_index(&node, &hits, true));
    }

    #[test]
    fn large_store_prefilter_recalls_exact_entity_and_short_circuits_empty_prompt_fast() {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let noise_body = "commit diff noise ".repeat(140); // ~2.5 KB per noise node
        for i in 0..310 {
            let input = vestige_core::IngestInput {
                content: format!("Noise commit #{i}: {noise_body}"),
                tags: vec![format!("commit:{i:08x}")],
                ..vestige_core::IngestInput::default()
            };
            storage.ingest_in_scope(input, "user").unwrap();
        }
        let pinned = vestige_core::IngestInput {
            content: "Standing rule: always use native Rust vestige-hook for PreToolUse.".into(),
            tags: vec!["pinned".into(), "owner-rule".into()],
            ..vestige_core::IngestInput::default()
        };
        storage.ingest_in_scope(pinned, "user").unwrap();

        let target = vestige_core::IngestInput {
            content: "Maya has a severe walnut allergy and prefers espresso in Kyoto.".into(),
            tags: vec!["person:Maya".into(), "place:Kyoto".into()],
            ..vestige_core::IngestInput::default()
        };
        let target_id = storage.ingest_in_scope(target, "user").unwrap().id;

        // 1. Empty prompt short-circuits immediately (<250 ms even on debug builds).
        let t0 = std::time::Instant::now();
        let empty = packet(storage.as_ref(), "user", &json!({ "prompt": "", "limit": 5 })).unwrap();
        assert!(empty["memories"].as_array().is_some());
        assert!(t0.elapsed().as_millis() < 250, "empty prompt too slow: {:?}", t0.elapsed());

        // 2. Warm entity query on >300-node noisy store recalls Maya's exact node via fast prefilter!
        let query_args = json!({ "prompt": "What allergy does Maya have in Kyoto?", "limit": 5 });
        let _warm = packet(storage.as_ref(), "user", &query_args).unwrap();
        let t1 = std::time::Instant::now();
        let res = packet(storage.as_ref(), "user", &query_args).unwrap();
        let elapsed_ms = t1.elapsed().as_millis();
        let ids: Vec<&str> = res["memories"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m["id"].as_str())
            .collect();
        assert!(ids.contains(&target_id.as_str()), "expected Maya node in {ids:?}");
        assert!(elapsed_ms < 2_500, "large-store warm memory_packet exceeded budget: {elapsed_ms}ms");
    }
}
