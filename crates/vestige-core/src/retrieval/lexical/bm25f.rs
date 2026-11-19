//! Core BM25F scoring, ordering, standing-rule ranking, and speaker/echo predicates.

use crate::retrieval::*;


// =========================================================================
// SCORING
// =========================================================================

pub(in crate::retrieval) struct Fields {
    pub(in crate::retrieval) w: [f32; 3],
    pub(in crate::retrieval) b: [f32; 3],
}

/// 3-Field BM25F (`user`, `assistant`, `blip_caption`) with IDF power law, IDF-weighted
/// distinct-term coverage, and separate `(any_mask, exact_mask)` so 0.45 hypernym hits
/// never poison submodular coverage masks (`covered_mask` / `covered_query_terms`).
pub(in crate::retrieval) struct PreparedPlan {
    /// `Some(idf)` when `qt.id` exists in `st.df`, else `None`.
    pub(in crate::retrieval) term_idf: Vec<Option<f32>>,
    pub(in crate::retrieval) g_idf: Vec<f32>,
    pub(in crate::retrieval) g_total: f32,
}

pub(in crate::retrieval) fn prepare_plan(plan: &QueryPlan, st: &Stats, p: &Params) -> PreparedPlan {
    let ng = plan.groups.len();
    let mut g_idf = vec![0.0f32; ng];
    let mut term_idf = Vec::with_capacity(plan.terms.len());
    for qt in &plan.terms {
        if let Some(&df) = st.df.get(&qt.id) {
            let df = df as f32;
            let idf = ((st.n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0).powf(p.idf_power);
            if let Some(g) = qt.group {
                g_idf[g] = g_idf[g].max(idf * qt.weight);
            }
            term_idf.push(Some(idf));
        } else {
            term_idf.push(None);
        }
    }
    let g_total: f32 = g_idf.iter().sum();
    PreparedPlan { term_idf, g_idf, g_total }
}

pub(in crate::retrieval) fn bm25f_prepared(
    bag: &Bag,
    plan: &QueryPlan,
    prep: &PreparedPlan,
    st: &Stats,
    p: &Params,
    fl: &Fields,
) -> (f32, u64, u64) {
    let mut score = 0.0f32;
    let mut mask = 0u64;
    let mut exact_mask = 0u64;
    let ng = plan.groups.len();
    let mut g_hit_stack = [false; 64];
    let mut g_hit_heap = if ng > 64 { vec![false; ng] } else { Vec::new() };
    let g_hit: &mut [bool] = if ng <= 64 { &mut g_hit_stack[..ng] } else { &mut g_hit_heap };

    for (qt, &maybe_idf) in plan.terms.iter().zip(prep.term_idf.iter()) {
        let Some(idf) = maybe_idf else { continue };
        let mut tfp = 0.0f32;
        for f in 0..3 {
            let tf = bag.tf[f].get(&qt.id).copied().unwrap_or(0.0);
            if tf <= 0.0 {
                continue;
            }
            let norm = 1.0 - fl.b[f] + fl.b[f] * (bag.len[f].max(1.0) / st.avg[f].max(1.0));
            tfp += fl.w[f] * tf / norm.max(0.05);
        }
        if tfp > 0.0 {
            score += qt.weight * idf * tfp * (p.k1 + 1.0) / (tfp + p.k1);
            if let Some(g) = qt.group {
                g_hit[g] = true;
                if g < 64 {
                    mask |= 1 << g;
                    if qt.exact_hit {
                        exact_mask |= 1 << g;
                    }
                }
            }
        }
    }
    if p.enable_coverage && score > 0.0 && prep.g_total > 0.0 {
        let hit: f32 = prep.g_idf.iter().zip(g_hit.iter()).filter(|(_, h)| **h).map(|(i, _)| *i).sum();
        let cov = hit / prep.g_total;
        score *= p.coverage_floor + (1.0 - p.coverage_floor) * cov * cov;
    }
    (score, mask, exact_mask)
}

#[allow(dead_code)]
pub(in crate::retrieval) fn bm25f(bag: &Bag, plan: &QueryPlan, st: &Stats, p: &Params, fl: &Fields) -> (f32, u64, u64) {
    let prep = prepare_plan(plan, st, p);
    bm25f_prepared(bag, plan, &prep, st, p, fl)
}

/// True when two distinct query words land within `span` content tokens of each other.
pub(in crate::retrieval) fn proximity_hit<'a, I: Iterator<Item = &'a (u32, u32)>>(seq: I, plan: &QueryPlan, span: usize) -> bool {
    let mut last: DetHashMap<usize, usize> = DetHashMap::default();
    for (pos, (e, s)) in seq.enumerate() {
        let g = plan.group_of.get(e).or_else(|| plan.group_of.get(s));
        if let Some(&g) = g {
            if last.iter().any(|(&h, &lp)| h != g && pos - lp <= span) {
                return true;
            }
            last.insert(g, pos);
        }
    }
    false
}

pub(in crate::retrieval) fn order(scores: &[f32], ts: &[i64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(ts[b].cmp(&ts[a]))
            .then(a.cmp(&b))
    });
    idx
}

pub(in crate::retrieval) fn jaccard(a: &DetHashSet<u32>, b: &DetHashSet<u32>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = (a.len() + b.len()) as f32 - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

pub(in crate::retrieval) fn raise_floor(scores: &mut [f32], lexical_max: f32, frac: f32, keep: impl Fn(usize) -> bool) -> Vec<bool> {
    let floor = if lexical_max > 0.0 { frac * lexical_max } else { 1.0 };
    let mut lifted = vec![false; scores.len()];
    for i in 0..scores.len() {
        if keep(i) && scores[i] < floor {
            scores[i] = floor;
            lifted[i] = true;
        }
    }
    lifted
}

pub(in crate::retrieval) fn named_partner(a: &str, b: &str) -> bool {
    let role = |s: &str| s.is_empty() || s == "user" || s == "assistant";
    !role(a) && !role(b) && a != b
}

/// Stratified Governing Channel (Satchel SCR 2026; Laird SOAR 2012; Forgy Rete 1982):
/// Ranks active standing imperative policies into `RankResult.standing_rules` without
/// consuming or distorting episodic `sessions` / `turns` slots.
pub(in crate::retrieval) fn rank_standing_rules(
    ix: &Index,
    plan: &QueryPlan,
    eff_it: &QueryIntent,
    episodic_user_quote: bool,
    t_mask: &[u64],
    top_k: usize,
) -> Vec<RankedItem> {
    if ix.standing_rules.is_empty() || episodic_user_quote {
        return Vec::new();
    }
    let it = eff_it;
    // Multi-evidence recall keeps every standing policy on the stratified channel,
    // including zero-overlap rules. The episodic floor stays off for these intents,
    // so the channel can fire without taking one of the five episodic slots.
    let multi_evidence = it.multi_hop || it.sniper_how_many || it.temporal || it.is_event_sequence;
    let mut latest_by_key: DetHashMap<&str, (usize, i32, i32, i64)> = DetHashMap::default();
    for (ri, r) in ix.standing_rules.iter().enumerate() {
        let mut scope_hits = r.scope.iter().filter(|id| plan.focal_ids.contains(id)).count() as i32;
        if scope_hits == 0 && crate::retrieval::seating::c2_policy_inflects_query(ix, r.turn, plan) {
            scope_hits = 1;
        }
        let asst_hits = r.assistant_nouns.iter().filter(|id| plan.focal_ids.contains(id)).count() as i32;
        let spec = 5 * scope_hits + asst_hits;
        let fires = if r.has_explicit_scope {
            spec > 0 || it.task_under_rules || multi_evidence
        } else {
            it.task_under_rules || spec >= 2 || asst_hits >= 2 || multi_evidence
        };
        if !fires {
            continue;
        }
        let scope_neg_len = -(r.scope.len() as i32);
        let cand = (ri, spec, scope_neg_len, r.ts);
        match latest_by_key.get(r.policy_key.as_str()) {
            Some(&cur) if (cur.3, cur.1, cur.2) >= (cand.3, cand.1, cand.2) => {}
            _ => {
                latest_by_key.insert(r.policy_key.as_str(), cand);
            }
        }
    }
    let max_spec = latest_by_key.values().map(|c| c.1).max().unwrap_or(0);
    let mut cands: Vec<(usize, i32, i32, i64)> = latest_by_key
        .into_values()
        .filter(|c| if max_spec > 0 { c.1 > 0 } else { true })
        .collect();
    cands.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then(b.2.cmp(&a.2))
            .then(b.3.cmp(&a.3))
            .then(a.0.cmp(&b.0))
    });
    cands
        .into_iter()
        .take(top_k.max(1))
        .map(|(ri, spec, _, _)| {
            let r = &ix.standing_rules[ri];
            RankedItem {
                id: ix.turns[r.turn].id.clone(),
                session_id: ix.session_ids[r.session].clone(),
                score: 1.0 + spec.max(0) as f32,
                matched: matched_words(plan, t_mask[r.turn]),
            }
        })
        .collect()
}

/// Rule-condition gate: does this standing rule's own "when I ask about X"
/// condition match the query? Exact/stem id overlap with the query focal ids,
/// a shared stem, or one WordNet relation (synonym / direct hypernym / hyponym)
/// between a condition word and a query content word. Unconditional rules never match.
pub(in crate::retrieval) fn rule_condition_matches(r: &StandingRule, plan: &QueryPlan, q_toks: &[String]) -> bool {
    // Two or more rare elicitation-context hits (rule scope enriched from the
    // turns around the rule) are as specific as a condition match.
    if r.scope.iter().filter(|id| plan.focal_ids.contains(id)).count() >= 2 {
        return true;
    }
    if r.condition_words.is_empty() {
        return false;
    }
    if r.condition.iter().any(|id| plan.focal_ids.contains(id)) {
        return true;
    }
    let q_content: Vec<&str> = q_toks.iter().map(String::as_str).filter(|w| is_content(w)).collect();
    // Derivational/inflectional suffix equivalence ("implement" ~ "implementation").
    fn derives(a: &str, b: &str) -> bool {
        let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
        if short.len() < 5 {
            return false;
        }
        long.strip_prefix(short).is_some_and(|rest| {
            // hardcode-scan: closed-class (English inflectional/derivational suffixes)
            matches!(
                rest,
                "" | "s" | "es" | "ed" | "ing" | "er" | "ers" | "al" | "ally" | "ly" | "ment" | "ments"
                    | "tion" | "tions" | "ation" | "ations" | "ion" | "ions" | "ism" | "ist" | "ists"
            )
        })
    }
    // One WordNet hop, or up to two hypernym hops from the query word; multiword
    // lemmas ("philosophical_doctrine") count when one of their parts is the condition word.
    // Same lexeme family: derivational suffix, or a long shared root
    // ("philosophy" ~ "philosophical": >=7 shared leading chars, <=3 chars short of the shorter word).
    let morph_eq = |a: &str, b: &str| {
        if a == b || derives(a, b) {
            return true;
        }
        let shared = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
        shared >= 7 && shared + 3 >= a.len().min(b.len())
    };
    let lemma_hits = |lemmas: &[&str], w: &str| {
        lemmas.iter().any(|l| morph_eq(l, w) || (l.contains('_') && l.split('_').any(|p| morph_eq(p, w))))
    };
    r.condition_words.iter().any(|cw| {
        let cst = stem(cw);
        let c_lemmas = crate::wordnet::expansion_lemmas(cw);
        q_content.iter().any(|&qw| {
            qw == cw
                || (cst.len() >= 4 && stem(qw) == cst)
                || morph_eq(qw, cw)
                || lemma_hits(&c_lemmas, qw)
                || lemma_hits(&crate::wordnet::expansion_lemmas(qw), cw)
                || lemma_hits(&crate::wordnet::upward_hypernyms(qw, 2), cw)
        })
    })
}

pub(in crate::retrieval) fn policy_spec_winner_floor(lexical_max: f32) -> f32 {
    if lexical_max > 0.0 {
        0.715 * lexical_max
    } else {
        0.715 / 0.68
    }
}

pub(in crate::retrieval) fn exact_query_group_ids(plan: &QueryPlan) -> DetHashSet<u32> {
    plan.terms
        .iter()
        .filter(|t| t.exact_hit && t.group.is_some())
        .map(|t| t.id)
        .collect()
}

pub(in crate::retrieval) fn session_speaker_names(ix: &Index, members: &[usize]) -> DetHashSet<String> {
    let mut names = DetHashSet::default();
    for &m in members {
        let sp = ix.turns[m].speaker.as_str();
        if sp.is_empty() || sp == "user" || sp == "assistant" {
            continue;
        }
        for w in raw_tokens(sp) {
            names.insert(w);
        }
    }
    names
}

pub(in crate::retrieval) fn is_speaker_term(ix: &Index, id: u32, names: &DetHashSet<String>) -> bool {
    let Some(name) = ix.vocab.names.get(id as usize) else { return false };
    let bare = name
        .strip_prefix("\u{1}ent:")
        .or_else(|| name.strip_prefix("\u{1}stem:"))
        .unwrap_or(name.as_str());
    if bare.contains('_') {
        return bare.split('_').any(|p| is_speaker_alias(p, names));
    }
    is_speaker_alias(bare, names)
}

pub(in crate::retrieval) fn clause_hits_query(ids: &DetHashSet<u32>, g: &DetHashSet<u32>, ix: &Index, names: &DetHashSet<String>) -> bool {
    ids.iter().any(|id| g.contains(id) && !is_speaker_term(ix, *id, names))
}

pub(in crate::retrieval) fn turn_echoes_query(ix: &Index, t: &TurnIx, g: &DetHashSet<u32>, names: &DetHashSet<String>) -> bool {
    if !t.is_pure_info_question {
        return false;
    }
    if clause_hits_query(&t.pre_question_ids, g, ix, names) {
        return false;
    }
    t.pre_question_ids.is_empty() || clause_hits_query(&t.question_clause_ids, g, ix, names)
}

pub(in crate::retrieval) fn carry_mode_is_parity() -> bool {
    static PARITY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PARITY.get_or_init(|| {
        std::env::var("VESTIGE_CARRY_MODE").map(|v| v == "parity").unwrap_or(false)
    })
}

// ---- moved verbatim from former `src/rungs.rs` (was `include!()`d into retrieval.rs) ----
// crates/vestige-core/src/rungs.rs
// Included inside `crates/vestige-core/src/retrieval.rs` via `include!("rungs.rs");`
// Active structural ranking refinements (Rungs 1, 2, 4 + Step 3A Panini karaka / clean slots / AdjPartnerDiv).

#[inline]
pub(in crate::retrieval) fn is_first_person_pronoun_cap(trimmed: &str) -> bool {
    let norm: String = trimmed.chars().map(|c| if c == '’' { '\'' } else { c }).collect();
    matches!(norm.as_str(), "I" | "I'm" | "I've" | "I'll" | "I'd")
}

/// Counts non-sentence-initial capitalized proper nouns, digits, or quoted titles at ingest time
/// so `--no-dump` (where `TurnIx.body` is empty) receives the exact same slot count as `--dump`.
pub(in crate::retrieval) fn count_slot_entities_mode(prose: &str, exclude_first_person: bool) -> u16 {
    let mut slot_entities = 0u16;
    let mut at_sent_start = true;
    for raw_tok in prose.split_whitespace() {
        let trimmed = raw_tok.trim_matches(|c: char| !c.is_alphanumeric() && c != '"' && c != '\'');
        if trimmed.is_empty() {
            continue;
        }
        let has_digit = trimmed.chars().any(|c| c.is_ascii_digit());
        let starts_quote = raw_tok.starts_with('"') || raw_tok.starts_with('\'');
        let alpha_count = trimmed.chars().filter(|c| c.is_alphabetic()).count();
        let first_alpha = trimmed.chars().find(|c| c.is_alphabetic());
        let cap_ok = !at_sent_start && first_alpha.is_some_and(|c| c.is_uppercase());
        let is_mid_cap = if exclude_first_person {
            cap_ok && alpha_count >= 2 && !is_first_person_pronoun_cap(trimmed)
        } else {
            cap_ok
        };
        if has_digit || starts_quote || is_mid_cap {
            slot_entities = slot_entities.saturating_add(1);
        }
        at_sent_start = raw_tok.ends_with('.') || raw_tok.ends_with('!') || raw_tok.ends_with('?');
    }
    slot_entities
}
