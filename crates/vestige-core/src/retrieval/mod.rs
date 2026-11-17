//! Stara Total-Recall Engine: zero-LLM, zero-vector, deterministic lexical retrieval.
//!
//! Every score is reproducible arithmetic over the text the caller supplied:
//! BM25F with role-segmented fields, IDF power law, distinct-term coverage,
//! ordered span proximity, deterministic suffix stemming, a fixed hypernym
//! lattice, a preference polarity axis, centering carry-forward across turns,
//! deictic relative-date compilation, max-passage window pooling, a
//! hierarchical session-to-turn funnel, question/answer dyad coupling,
//! bounded Adamic-Adar bridge closure and causal recency supersession.
//!
//! Leak guard: every input struct is `deny_unknown_fields`. A turn carries only
//! `id`, `role`, `content`; a session only `id`, `date`, `timestamp`, `turns`;
//! a query only `text` and `date`. Benchmark labels such as `has_answer`,
//! `answer`, `question_type` or `category` are rejected at the parse boundary,
//! so they can never influence a score.

#![allow(clippy::all)]


mod ac3;
mod actr;
mod bm25l_dfr;
mod bridge;
mod coref;
mod dph;
mod focus;
mod ghostlink_dgs;
mod hobbs;
mod index;
mod locke_ledger;
mod psl_ilp;
mod qld;
mod rank;
mod rbp;
mod rocchio;
mod rst;
mod scoring;
mod seating;
mod supersession;
mod temporal_allen;
mod tokens;
mod viterbi;
#[cfg(test)]
mod tests;

// Flatten every submodule back into `retrieval::` so the public API (and every
// crate-internal path) is identical to the former single-file module.
use self::bridge::*;
pub use self::index::*;
use self::rank::*;
pub use self::scoring::*;
use self::seating::*;
use self::supersession::*;
pub use self::tokens::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

// Step 10B: fixed-key SipHash instead of per-process RandomState.
type DetHasher = std::hash::BuildHasherDefault<std::collections::hash_map::DefaultHasher>;
type DetHashMap<K, V> = HashMap<K, V, DetHasher>;
type DetHashSet<K> = HashSet<K, DetHasher>;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::OnceLock;

// =========================================================================
// INPUT / OUTPUT CONTRACT
// =========================================================================

/// Exact edge. `to` is a turn id. The ranker follows it after exact-token
/// selection and does not read either side's words.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct TurnLink {
    pub kind: String,
    pub to: String,
    /// Turns that share this identity. `0` means one pair.
    #[serde(default)]
    pub degree: u32,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Turn {
    #[serde(default)]
    pub id: String,
    /// "user", "assistant", or a speaker name (LoCoMo-style dialogues).
    #[serde(default)]
    pub role: String,
    pub content: String,
    /// Recorded edges. Empty on raw benchmark text. Reply, order, and session
    /// links are written from position inside the ranker.
    #[serde(default)]
    pub links: Vec<TurnLink>,
    /// FSRS stability in days. Zero means unset. Used only to break an exact tie.
    #[serde(default)]
    pub stability: f32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Session {
    pub id: String,
    /// Free-form session date ("2023/05/20 (Sat) 02:21", "1:56 pm on 8 May, 2023").
    #[serde(default)]
    pub date: String,
    /// Unix seconds. Used when `date` is empty or unparseable.
    #[serde(default)]
    pub timestamp: i64,
    pub turns: Vec<Turn>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Query {
    pub text: String,
    #[serde(default)]
    pub date: String,
}

fn default_top_k() -> usize {
    10
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankRequest {
    pub sessions: Vec<Session>,
    pub queries: Vec<Query>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub params: Params,
}

/// Every weapon is individually switchable so ablations are honest and reproducible.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Params {
    pub k1: f32,
    pub b_user: f32,
    pub b_assistant: f32,
    /// Length normalisation on assistant turns when the query attributes content to "you".
    pub b_assistant_attributed: f32,
    pub stem_weight: f32,
    /// Weight of a WordNet expansion term relative to an original query term (1.0).
    /// Strictly below 1.0; expansion only runs behind the zero-overlap gate.
    pub hypernym_weight: f32,
    pub idf_power: f32,
    pub coverage_floor: f32,
    pub proximity_bonus: f32,
    pub proximity_span: usize,
    pub window: usize,
    /// Share of the max-passage window score in the session score.
    pub window_mix: f32,
    pub assistant_attribution_weight: f32,
    pub user_attribution_weight: f32,
    pub preference_boost: f32,
    pub preference_axis_weight: f32,
    pub recency_boost: f32,
    pub recency_band: f32,
    pub recency_depth: usize,
    pub carry: [f32; 3],
    pub carry_answer: f32,
    pub carry_plain_factor: f32,
    pub answer_boost: f32,
    pub funnel: f32,
    pub dyad: f32,
    pub dyad_partner: f32,
    pub bridge_depth: usize,
    pub date_weight: f32,
    pub enable_stemming: bool,
    pub enable_proximity: bool,
    pub enable_coverage: bool,
    pub enable_roles: bool,
    pub enable_preference: bool,
    pub enable_hypernyms: bool,
    pub enable_recency: bool,
    pub enable_window: bool,
    pub enable_carry: bool,
    pub enable_deictic: bool,
    pub enable_funnel: bool,
    pub enable_dyad: bool,
    pub enable_bridge: bool,
    pub enable_scene: bool,
    pub enable_sniper: bool,
    pub enable_ppr: bool,
    pub enable_submodular: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b_user: 0.75,
            b_assistant: 0.75,
            b_assistant_attributed: 0.25,
            stem_weight: 0.70,
            hypernym_weight: 0.5,
            idf_power: 1.5,
            coverage_floor: 0.35,
            proximity_bonus: 0.35,
            proximity_span: 5,
            window: 3,
            window_mix: 0.65,
            assistant_attribution_weight: 1.6,
            user_attribution_weight: 1.5,
            preference_boost: 2.8,
            preference_axis_weight: 1.0,
            recency_boost: 1.30,
            recency_band: 0.70,
            recency_depth: 5,
            carry: [0.65, 0.40, 0.20],
            carry_answer: 0.85,
            carry_plain_factor: 0.5,
            answer_boost: 1.15,
            funnel: 0.65,
            dyad: 0.35,
            dyad_partner: 0.90,
            bridge_depth: 15,
            date_weight: 0.30,
            enable_stemming: true,
            enable_proximity: true,
            enable_coverage: true,
            enable_roles: true,
            enable_preference: true,
            enable_hypernyms: true,
            enable_recency: true,
            enable_window: true,
            enable_carry: true,
            enable_deictic: true,
            enable_funnel: true,
            enable_dyad: true,
            enable_bridge: true,
            enable_scene: true,
            enable_sniper: true,
            enable_ppr: true,
            enable_submodular: true,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct QueryIntent {
    pub preference: bool,
    pub current_state: bool,
    pub assistant_attribution: bool,
    pub user_attribution: bool,
    pub multi_hop: bool,
    pub inference: bool,
    pub temporal: bool,
    pub sniper_when: bool,
    pub sniper_how_many: bool,
    pub sniper_where: bool,
    pub sniper_why: bool,
    pub is_event_sequence: bool,
    /// The query asks for a task whose answer should obey stored imperative rules,
    /// even when those rules share no content words with the question.
    pub task_under_rules: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RankedItem {
    pub id: String,
    pub session_id: String,
    pub score: f32,
    /// Exact query words that matched this item (the proof of the rank).
    pub matched: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RankResult {
    pub intent: QueryIntent,
    pub sessions: Vec<RankedItem>,
    pub turns: Vec<RankedItem>,
    /// Stratified governing channel: standing imperative policies that apply to the query
    /// without consuming episodic `sessions` or `turns` top-k slots.
    pub standing_rules: Vec<RankedItem>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RankResponse {
    pub engine: &'static str,
    pub sessions_indexed: usize,
    pub turns_indexed: usize,
    pub results: Vec<RankResult>,
}

pub const ENGINE: &str = "stara-total-recall/2 (bm25f+idf^p+coverage+proximity+stem+hypernym+pref-axis+carry+deictic+window+scene+sniper+ppr+submodular+funnel+dyad+recency)";


// =========================================================================
// REPLAY DUMP (off unless VESTIGE_REPLAY_DUMP is set; never changes ranking)
// =========================================================================

fn replay_dump_path() -> Option<String> {
    match std::env::var("VESTIGE_REPLAY_DUMP") {
        Ok(p) if !p.is_empty() => Some(p),
        _ => None,
    }
}

/// Snapshot only. No DetHashMap/DetHashSet: those would advance Rust's hasher
/// seed and change later queries in the same RankRequest.
struct ReplaySnap {
    query_text: String,
    query_date: String,
    k: usize,
    is_event_sequence: bool,
    temporal: bool,
    preference: bool,
    assistant_attribution: bool,
    seated: Vec<usize>,
    final_session_indices: Vec<usize>,
    t_scene: Vec<f32>,
    // Dump-only copies of scores the seat already computed. Not read by ranking.
    t_own: Vec<f32>,
    t_turn_raw: Vec<f32>,
    t_scene_sess: Vec<f32>,
    s_score: Vec<f32>,
    bridge_boost: Vec<f32>,
}

thread_local! {
    static REPLAY_SNAPS: std::cell::RefCell<Vec<ReplaySnap>> = std::cell::RefCell::new(Vec::new());
}

fn maybe_dump_replay(
    _ix: &Index,
    q: &Query,
    k: usize,
    it: &QueryIntent,
    t_scene: &[f32],
    seated: &[usize],
    final_session_indices: &[usize],
    t_own: &[f32],
    t_turn_raw: &[f32],
    t_scene_sess: &[f32],
    s_score: &[f32],
    bridge_boost: &[f32],
) {
    if replay_dump_path().is_none() {
        return;
    }
    REPLAY_SNAPS.with(|slot| {
        slot.borrow_mut().push(ReplaySnap {
            query_text: q.text.clone(),
            query_date: q.date.clone(),
            k,
            is_event_sequence: it.is_event_sequence,
            temporal: it.temporal,
            preference: it.preference,
            assistant_attribution: it.assistant_attribution,
            seated: seated.to_vec(),
            final_session_indices: final_session_indices.to_vec(),
            t_scene: t_scene.to_vec(),
            t_own: t_own.to_vec(),
            t_turn_raw: t_turn_raw.to_vec(),
            t_scene_sess: t_scene_sess.to_vec(),
            s_score: s_score.to_vec(),
            bridge_boost: bridge_boost.to_vec(),
        });
    });
}

fn flush_replay_dumps(ix: &Index) {
    let Some(path) = replay_dump_path() else {
        return;
    };
    let snaps = REPLAY_SNAPS.with(|slot| std::mem::take(&mut *slot.borrow_mut()));
    if snaps.is_empty() {
        return;
    }
    let mut file = match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(f) => std::io::BufWriter::new(f),
        Err(e) => {
            eprintln!("VESTIGE_REPLAY_DUMP open failed: {e}");
            return;
        }
    };
    for snap in snaps {
        write_one_replay_dump(ix, &snap, &mut file);
    }
    let _ = file.flush();
}

/// Dump-only. ASCII alphanumeric split, same character geometry as the later
/// `query_tokens` helper. Not wired into ranking.
fn dump_query_tokens(query: &str) -> Vec<String> {
    query
        .to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

/// Dump-only information content from the conversation's own counts.
/// `((n+1)/(df+1)).ln()`. Not wired into ranking.
fn dump_token_ic(n: usize, df: usize) -> f64 {
    ((n as f64 + 1.0) / (df as f64 + 1.0)).ln()
}

/// Last non-space character is ASCII '?'. Dump-only. Not wired into ranking.
fn dump_ends_with_question(body: &str) -> bool {
    body.as_bytes()
        .iter()
        .rev()
        .find(|&&b| b != b' ' && b != b'\t' && b != b'\n' && b != b'\r')
        .copied()
        == Some(b'?')
}

/// Direct same-session reply: the next turn in this session whose stored
/// `answers` edge points at `ti`. None when that edge is absent. Dump-only.
fn dump_response_turn_of(ix: &Index, ti: usize) -> Option<usize> {
    let t = ix.turns.get(ti)?;
    let members = ix.session_turns.get(t.session)?;
    let pos = members.iter().position(|&m| m == ti)?;
    let next = *members.get(pos + 1)?;
    if ix.turns.get(next).and_then(|n| n.answers) == Some(ti) {
        Some(next)
    } else {
        None
    }
}

/// `IC_novel(t \ Q)`: sum of dump_token_ic over the turn's distinct raw_tokens
/// that are not in the query token set, using the engine tokenizer and
/// `t_own_stats` df. Tokens the engine never interned are skipped. Dump-only.
fn dump_turn_ic_novel(ix: &Index, ti: usize, qset: &DetHashSet<String>, n_docs: usize) -> f64 {
    let body = ix.turns.get(ti).map(|t| t.body.as_str()).unwrap_or("");
    let mut seen: DetHashSet<String> = DetHashSet::default();
    let mut sum = 0.0;
    for w in raw_tokens(body) {
        if !seen.insert(w.clone()) || qset.contains(&w) {
            continue;
        }
        let Some(id) = ix.vocab.get(&w) else {
            continue;
        };
        let df = ix.t_own_stats.df.get(&id).copied().unwrap_or(0) as usize;
        sum += dump_token_ic(n_docs, df);
    }
    sum
}

fn dump_edge_ids(ix: &Index, targets: impl IntoIterator<Item = usize>) -> Vec<String> {
    targets
        .into_iter()
        .filter_map(|tgt| ix.turns.get(tgt).map(|t| t.id.clone()))
        .collect()
}

/// Dump-only copy of `pool_structural_degree` with an optional antecedent arm.
/// Not wired into ranking. `include_rsb_antecedent = false` is `degree_none`.
fn dump_pool_structural_degree(ix: &Index, pool: &[usize], include_rsb_antecedent: bool) -> Vec<i64> {
    let mut at = vec![None; ix.turns.len()];
    for (i, &ti) in pool.iter().enumerate() {
        if let Some(slot) = at.get_mut(ti) {
            *slot = Some(i);
        }
    }
    let mut degree = vec![0i64; pool.len()];
    for (src, t) in ix.turns.iter().enumerate() {
        let mut hit = |target: usize| {
            if target == src {
                return;
            }
            if let Some(Some(slot)) = at.get(target) {
                degree[*slot] += 1;
            }
        };
        for &target in &t.policy_backlinks {
            hit(target);
        }
        for &target in &t.supersedes_turns {
            hit(target);
        }
        for &(_, target) in &t.exact_edges {
            hit(target);
        }
        if include_rsb_antecedent {
            if let Some(target) = rsb_antecedent(ix, src) {
                hit(target);
            }
        }
    }
    degree
}

fn write_one_replay_dump(ix: &Index, snap: &ReplaySnap, file: &mut std::io::BufWriter<std::fs::File>) {
    let n_docs = ix.turns.len();
    let turn_ts: Vec<i64> = ix.turns.iter().map(|t| ix.session_ts[t.session]).collect();
    let mut pool: Vec<usize> = snap.seated.clone();
    for ti in order(&snap.t_scene, &turn_ts) {
        if pool.len() >= 50 {
            break;
        }
        if snap.t_scene.get(ti).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        if !pool.contains(&ti) {
            pool.push(ti);
        }
    }
    let mut pre_rank: DetHashMap<usize, usize> = DetHashMap::with_capacity_and_hasher(pool.len(), Default::default());
    for (i, &ti) in pool.iter().enumerate() {
        pre_rank.insert(ti, i);
    }
    let mut final_rank: DetHashMap<usize, usize> = DetHashMap::default();
    for (i, &ti) in snap.seated.iter().enumerate() {
        final_rank.insert(ti, i);
    }
    let deg_closed = dump_pool_structural_degree(ix, &pool, true);
    let deg_none = dump_pool_structural_degree(ix, &pool, false);
    let mut degree_closed: DetHashMap<usize, i64> = DetHashMap::default();
    let mut degree_none: DetHashMap<usize, i64> = DetHashMap::default();
    for (i, &ti) in pool.iter().enumerate() {
        degree_closed.insert(ti, deg_closed[i]);
        degree_none.insert(ti, deg_none[i]);
    }

    let qset: DetHashSet<String> = raw_tokens(&snap.query_text).into_iter().collect();
    let mut qtoks: Vec<Value> = Vec::new();
    let mut seen_tok: DetHashSet<String> = DetHashSet::default();
    for tok in dump_query_tokens(&snap.query_text) {
        if !seen_tok.insert(tok.clone()) {
            continue;
        }
        let mut df = 0usize;
        let mut ids: Vec<u32> = Vec::new();
        if let Some(id) = ix.vocab.get(&tok) {
            ids.push(id);
            df = ix.t_own_stats.df.get(&id).copied().unwrap_or(0) as usize;
        }
        let stemmed = stem(&tok);
        if stemmed != tok {
            if let Some(id) = ix.vocab.get(&stemmed) {
                ids.push(id);
                let d2 = ix.t_own_stats.df.get(&id).copied().unwrap_or(0) as usize;
                if d2 > df {
                    df = d2;
                }
            }
        }
        let ic = dump_token_ic(n_docs, df);
        qtoks.push(json!({
            "tok": tok,
            "df": df,
            "ic": ic,
            "ids": ids,
        }));
    }

    let qtok_present = |ti: usize| -> Vec<bool> {
        let body = ix.turns.get(ti).map(|t| t.body.as_str()).unwrap_or("");
        let body_toks: DetHashSet<String> = dump_query_tokens(body).into_iter().collect();
        let content = ix.turns.get(ti).map(|t| &t.content);
        let mut present: Vec<bool> = Vec::with_capacity(qtoks.len());
        for tok in qtoks.iter() {
            let t = tok.get("tok").and_then(|v| v.as_str()).unwrap_or("");
            let mut hit = body_toks.contains(t);
            if !hit {
                if let Some(arr) = tok.get("ids").and_then(|v| v.as_array()) {
                    if let Some(content) = content {
                        for idv in arr {
                            if let Some(id) = idv.as_u64() {
                                if content.contains(&(id as u32)) {
                                    hit = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            present.push(hit);
        }
        present
    };

    let emit_turn = |ti: usize, rank_pre: Option<usize>| -> Value {
        let t = &ix.turns[ti];
        let cause = rsb_cause(ix, ti);
        let cause_id = cause.map(|c| ix.turns[c].id.clone());
        let cause_rank = cause.and_then(|c| pre_rank.get(&c).copied());
        let mut symptom_ids: Vec<String> = Vec::new();
        for &s in &snap.seated {
            if rsb_cause(ix, s) == Some(ti) {
                symptom_ids.push(ix.turns[s].id.clone());
            }
        }
        let sess = t.session;
        let session_order = turn_pos(ix, ti).unwrap_or(0);
        let parent_id = t.answers.and_then(|p| ix.turns.get(p).map(|x| x.id.clone()));
        json!({
            "id": t.id,
            "idx": ti,
            "rank_pre": rank_pre,
            "rank": final_rank.get(&ti).copied(),
            "t_scene": snap.t_scene.get(ti).copied().unwrap_or(0.0),
            "t_scene_bits": snap.t_scene.get(ti).copied().unwrap_or(0.0).to_bits(),
            "t_own": snap.t_own.get(ti).copied().unwrap_or(0.0),
            "t_turn_raw": snap.t_turn_raw.get(ti).copied().unwrap_or(0.0),
            "t_scene_sess": snap.t_scene_sess.get(ti).copied().unwrap_or(0.0),
            "s_score": snap.s_score.get(sess).copied().unwrap_or(0.0),
            "bridge_boost": snap.bridge_boost.get(sess).copied().unwrap_or(0.0),
            "stability": t.stability,
            "stability_bits": t.stability.to_bits(),
            "session": sess,
            "session_id": ix.session_ids.get(sess).cloned().unwrap_or_default(),
            "session_order": session_order,
            "role": t.speaker,
            "ends_with_question": dump_ends_with_question(&t.body),
            "is_question": t.is_question,
            "turn_ic_novel": dump_turn_ic_novel(ix, ti, &qset, n_docs),
            "ts": ix.session_ts.get(sess).copied().unwrap_or(0),
            "superseded": turn_is_superseded(ix, ti),
            "response_turn_id": dump_response_turn_of(ix, ti).map(|r| ix.turns[r].id.clone()),
            "parent_user_turn_id": parent_id,
            "rsb_seated": final_rank.contains_key(&ti),
            "rsb_cause_id": cause_id,
            "rsb_cause_rank": cause_rank,
            "rsb_symptom_ids": symptom_ids,
            "exact_edges": dump_edge_ids(ix, t.exact_edges.iter().map(|&(_, tgt)| tgt)),
            "policy_backlinks": dump_edge_ids(ix, t.policy_backlinks.iter().copied()),
            "degree_closed": degree_closed.get(&ti).copied().unwrap_or(0),
            "degree_none": degree_none.get(&ti).copied().unwrap_or(0),
            "qtok_present": qtok_present(ti),
        })
    };

    let mut cands: Vec<Value> = Vec::with_capacity(pool.len());
    for (rank, &ti) in pool.iter().enumerate() {
        cands.push(emit_turn(ti, Some(rank)));
    }

    // Sessions in official turn top-10 or official session top-10: every turn.
    let mut sess_keep: DetHashSet<usize> = DetHashSet::default();
    for &ti in snap.seated.iter().take(10) {
        sess_keep.insert(ix.turns[ti].session);
    }
    for &si in snap.final_session_indices.iter().take(10) {
        sess_keep.insert(si);
    }
    let mut sess_keys: Vec<usize> = sess_keep.into_iter().collect();
    sess_keys.sort_unstable();
    let mut session_turns: Vec<Value> = Vec::new();
    for sess in sess_keys {
        let members = match ix.session_turns.get(sess) {
            Some(m) => m,
            None => continue,
        };
        for &ti in members {
            session_turns.push(emit_turn(ti, pre_rank.get(&ti).copied()));
        }
    }

    // Dump-only compact scores for every turn and session. Bit layout of `flags`:
    // 0 question, 1 digit form, 2 clock form, 3 midcap, 4 numeric, 5 temporal,
    // 6 anaphoric, 7 short reply, 8 answers a question, 9 backchannel.
    let score_rows: Vec<Value> = (0..ix.turns.len())
        .map(|ti| {
            let t = &ix.turns[ti];
            let mut flags = 0u16;
            if t.is_question {
                flags |= 1;
            }
            if t.form_digit {
                flags |= 1 << 1;
            }
            if t.form_clock {
                flags |= 1 << 2;
            }
            if t.form_midcap {
                flags |= 1 << 3;
            }
            if t.has_numeric {
                flags |= 1 << 4;
            }
            if t.has_temporal {
                flags |= 1 << 5;
            }
            if t.anaphoric {
                flags |= 1 << 6;
            }
            if t.is_short_reply {
                flags |= 1 << 7;
            }
            if t.answers_question {
                flags |= 1 << 8;
            }
            if t.is_backchannel {
                flags |= 1 << 9;
            }
            json!([
                t.id,
                ix.session_ids.get(t.session).cloned().unwrap_or_default(),
                turn_pos(ix, ti).unwrap_or(0),
                snap.t_scene.get(ti).copied().unwrap_or(0.0).to_bits(),
                snap.t_own.get(ti).copied().unwrap_or(0.0).to_bits(),
                snap.t_turn_raw.get(ti).copied().unwrap_or(0.0).to_bits(),
                flags,
            ])
        })
        .collect();
    let session_rows: Vec<Value> = (0..ix.session_ids.len())
        .map(|si| {
            json!([
                ix.session_ids[si],
                snap.s_score.get(si).copied().unwrap_or(0.0).to_bits(),
                snap.bridge_boost.get(si).copied().unwrap_or(0.0).to_bits(),
            ])
        })
        .collect();

    let query_ts = parse_date(&snap.query_date);
    let slots = wh_slots(&snap.query_text);
    let rec = json!({
        "query_text": snap.query_text,
        "query_date": snap.query_date,
        "query_ts": query_ts,
        "first_session_id": ix.session_ids.first().cloned().unwrap_or_default(),
        "n_sessions": ix.session_ids.len(),
        "n_turns": ix.turns.len(),
        "k": snap.k,
        "wh_slots": slots,
        "wh_class": format!("WH_SLOTS_{slots}"),
        "is_event_sequence": snap.is_event_sequence,
        "temporal": snap.temporal,
        "preference": snap.preference,
        "assistant_attribution": snap.assistant_attribution,
        "pool_size": pool.len(),
        "pre_ids": pool.iter().map(|&ti| ix.turns[ti].id.clone()).collect::<Vec<_>>(),
        "seated_ids": snap.seated.iter().map(|&ti| ix.turns[ti].id.clone()).collect::<Vec<_>>(),
        "final_ids": snap.seated.iter().map(|&ti| ix.turns[ti].id.clone()).collect::<Vec<_>>(),
        "final_session_ids": snap.final_session_indices.iter().map(|&si| ix.session_ids[si].clone()).collect::<Vec<_>>(),
        "head_id": snap.seated.first().map(|&ti| ix.turns[ti].id.clone()),
        "query_tokens": qtoks,
        "candidates": cands,
        "session_turns": session_turns,
        // Dump-only. Every indexed turn and session, so a gold absent from
        // `pre_ids` can be staged. Ranking never reads this object.
        "score_rows": score_rows,
        "session_rows": session_rows,
    });
    let _ = writeln!(file, "{rec}");
}


pub fn rank(req: &RankRequest) -> RankResponse {

    let ix = build_index(&req.sessions, &req.params);
    let results = req.queries.iter().map(|q| rank_one(&ix, q, &req.params, req.top_k)).collect();
    flush_replay_dumps(&ix);
    RankResponse {
        engine: ENGINE,
        sessions_indexed: ix.session_ids.len(),
        turns_indexed: ix.turns.len(),
        results,
    }
}
