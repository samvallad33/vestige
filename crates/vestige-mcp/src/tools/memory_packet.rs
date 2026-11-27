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
use vestige_core::{KnowledgeNode, Storage};

use super::lexical::{self, HOOK_CHAR_BUDGET};

#[derive(Clone)]
struct PacketSnapshot {
    nodes: Arc<Vec<KnowledgeNode>>,
    retired: Arc<HashSet<String>>,
    scores: Arc<BTreeMap<String, i128>>,
    receipts: Arc<Vec<Value>>,
    retrieval: Arc<vestige_core::retrieval::RetrievalEngine>,
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

fn build_retrieval_engine(
    storage: &Storage,
    nodes: &[KnowledgeNode],
    retired: &HashSet<String>,
) -> vestige_core::retrieval::RetrievalEngine {
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
    let mut by_day: BTreeMap<String, (i64, Vec<vestige_core::retrieval::Turn>)> = BTreeMap::new();
    for node in nodes {
        if retired.contains(&node.id) {
            continue;
        }
        let day = node.created_at.format("%Y/%m/%d").to_string();
        let ts = node.created_at.timestamp();
        let role = if matches!(node.node_type.as_str(), "decision" | "correction")
            || node
                .tags
                .iter()
                .any(|t| t == "pinned" || t == "owner-directive" || t.starts_with("forbid:"))
        {
            "user".to_string()
        } else {
            "assistant".to_string()
        };
        let content = if node.tags.is_empty() {
            node.content.clone()
        } else {
            format!("{} [{}]", node.content, node.tags.join(" "))
        };
        let entry = by_day.entry(day).or_insert_with(|| (ts, Vec::new()));
        if ts > entry.0 {
            entry.0 = ts;
        }
        entry.1.push(vestige_core::retrieval::Turn {
            id: node.id.clone(),
            role,
            content,
            links: links_by_source.remove(&node.id).unwrap_or_default(),
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
    let nodes = Arc::new(
        storage
            .get_all_nodes_in_scope(scope, i32::MAX, 0)
            .map_err(|e| e.to_string())?,
    );
    let retired = Arc::new(storage.superseded_node_ids().map_err(|e| e.to_string())?);
    let receipts = Arc::new(storage.salience_receipts().map_err(|e| e.to_string())?);
    let score_map = scores_from_receipts(receipts.as_ref());
    let retrieval = Arc::new(build_retrieval_engine(
        storage,
        nodes.as_ref(),
        retired.as_ref(),
    ));
    let snapshot = PacketSnapshot {
        nodes,
        retired,
        scores: Arc::new(score_map),
        receipts,
        retrieval,
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

/// Whether a stored identity tag (`person:`, `place:`, `food:`, `file:`,
/// `service:`, `symbol:`) is named in `prompt_lower` (pre-lowercased prompt).
/// Hierarchical `symbol:<ns>:<leaf>` tags also match on their leaf token.
fn entity_matches_prompt_lower(prompt_lower: &str, tag: &str) -> bool {
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
        if leaf.contains('~') {
            let spaced = leaf.replace('~', " ");
            if phrase_lower(prompt_lower, &spaced) {
                return true;
            }
        }
    }
    false
}

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
    let snapshot = load_cached_snapshot(storage, scope)?;
    let (all_nodes, retired, scores_arc, all_receipts, retrieval) = (
        snapshot.nodes,
        snapshot.retired,
        snapshot.scores,
        snapshot.receipts,
        snapshot.retrieval,
    );
    let now = chrono::Utc::now();
    let nodes: Vec<KnowledgeNode> = all_nodes
        .iter()
        .filter(|n| {
            !retired.contains(&n.id)
                && n.valid_from.is_none_or(|t| t <= now)
                && n.valid_until.is_none_or(|t| t > now)
        })
        .cloned()
        .collect();
    let count = nodes.len();
    let scores = scores_arc.as_ref();
    let tie = |node: &KnowledgeNode| scores.get(&node.id).copied().unwrap_or_default();
    let requested = args["limit"].as_u64().unwrap_or(8).clamp(1, 24) as usize;
    let prompt_lower = prompt.to_lowercase();
    let named = lexical::named_people(&nodes, prompt);
    let entities: BTreeSet<String> = nodes
        .iter()
        .flat_map(|n| n.tags.iter())
        .filter(|tag| {
            ["person:", "place:", "food:", "file:", "service:", "symbol:"]
                .iter()
                .any(|prefix| tag.starts_with(prefix))
        })
        .filter(|tag| entity_matches_prompt_lower(&prompt_lower, tag))
        .cloned()
        .collect();
    let mut circuit_diag = Value::Null;
    let (selected, selection, ranking): (Vec<&KnowledgeNode>, &str, &str) =
        if prompt.trim().is_empty() {
            // No prompt, no target: never inject unrelated salience.
            (Vec::new(), "none_empty_prompt", "none")
        } else {
            let tr_pool = requested.saturating_mul(6).max(32);
            let today = now.format("%Y/%m/%d").to_string();
            let rank_res = retrieval.query(
                &vestige_core::retrieval::Query {
                    text: prompt.to_string(),
                    date: today,
                },
                tr_pool,
            );
            circuit_diag = json!({
                "engine": vestige_core::retrieval::ENGINE,
                "turns_indexed": retrieval.turns_indexed(),
                "premise_coverage": rank_res.premise_coverage,
                "standing_rules_count": rank_res.standing_rules.len(),
            });
            let pos_by_id: HashMap<&str, usize> = nodes
                .iter()
                .enumerate()
                .map(|(i, n)| (n.id.as_str(), i))
                .collect();
            let standing_rule_indexes: HashSet<usize> = rank_res
                .standing_rules
                .iter()
                .filter_map(|item| pos_by_id.get(item.id.as_str()).copied())
                .collect();
            let mut hits: Vec<lexical::LexicalHit> = Vec::new();
            let mut seen_idx: HashSet<usize> = HashSet::new();
            for item in rank_res.standing_rules.iter().chain(rank_res.turns.iter()) {
                let Some(&index) = pos_by_id.get(item.id.as_str()) else {
                    continue;
                };
                if !seen_idx.insert(index) {
                    continue;
                }
                let score_micro = ((item.score as f64) * 1_000_000.0).round() as i64;
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
            // A recorded identity the prompt names still qualifies when the
            // memory's other words do not repeat the prompt.
            // A person card (salience-pro facts-of:): every lasting fact about
            // each person the prompt names is index-joined as well.
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
            let seen_idx: HashSet<usize> = indexes.iter().copied().collect();
            for (index, node) in nodes.iter().enumerate() {
                if !seen_idx.contains(&index)
                    && (node.tags.iter().any(|tag| entities.contains(tag))
                        || joins_index(node, &index_hits, has_day_hits))
                    && lexical::allows_person(node, &named)
                {
                    indexes.push(index);
                }
            }
            // Retroactive Salience Backfill comes first only when on-target:
            // a memory a backfill receipt raised outranks lexical score when
            // it also joins the prompt by an entity or date/avoid index tag,
            // or its own score is at least half the top hit.
            let top_score = hits.iter().map(|hit| hit.score_micro).max().unwrap_or(0);
            let on_target = |index: &usize| {
                let node = &nodes[*index];
                joins_index(node, &index_hits, has_day_hits)
                    || node.tags.iter().any(|tag| entities.contains(tag))
                    || (top_score > 0 && score_of(&node.id) * 2 >= top_score)
            };
            let raised = |index: &usize| tie(&nodes[*index]) > 0 && on_target(index);
            // A memory whose salience receipt lowered it (e.g. a superseded residence via
            // `supersedes` edge) yields to un-demoted peers before lexical tie-breaking.
            let lowered = |index: &usize| tie(&nodes[*index]) < 0;
            // salience-pro 0.3.7 candidate reservation: a memory joined to the
            // prompt by a date or an avoid index tag is reserved a place right
            // after the raised ones, at most MAX_RESERVED, so lexical scores on
            // other words cannot crowd out the date or the thing ruled out.
            let mut reserved: HashSet<usize> = HashSet::new();
            let mut joined: Vec<usize> = indexes
                .iter()
                .copied()
                .filter(|index| joins_index(&nodes[*index], &index_hits, has_day_hits))
                .collect();
            // Safety first: an allergy, health, a condition or a medicine
            // before a preference.
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
            // Reserve the retrieval engine's own stratified `standing_rules` channel.
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
            let packed = lexical::pack(&nodes, &packed_hits, HOOK_CHAR_BUDGET, requested);
            let selection = if !entities.is_empty() {
                "exact_recorded_entities"
            } else if hits.is_empty() {
                "top_salience"
            } else {
                "stara_total_recall_v2"
            };
            (packed, selection, vestige_core::retrieval::ENGINE)
        };
    let ids: BTreeSet<&str> = selected.iter().map(|n| n.id.as_str()).collect();
    let seed_indexes: Vec<usize> = selected
        .iter()
        .filter_map(|node| nodes.iter().position(|candidate| candidate.id == node.id))
        .collect();
    let bridge_rows = lexical::bridges(&nodes, &seed_indexes, 4);
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
    let receipts: Vec<Value> = all_receipts
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
}
