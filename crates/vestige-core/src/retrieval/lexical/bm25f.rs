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
    q_text: &str,
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
    let multi_evidence = it.multi_hop || it.sniper_how_many || it.temporal || it.is_event_sequence;

    // Precompute query words (combining `plan.groups` AND raw query content tokens so high-DF
    // conversation-domain nouns stripped by `corpus_frame` still activate their conditioned
    // standing rules!) + human-knowledge lexicon (WordNet 3.1 + Roget + ConceptNet) once per query.
    let mut q_words_owned: Vec<String> = plan
        .groups
        .iter()
        .filter(|w| !w.starts_with('\u{1}') && is_content(w))
        .cloned()
        .collect();
    for rw in raw_tokens(q_text) {
        if is_content(&rw) && !q_words_owned.iter().any(|existing| existing == &rw) {
            q_words_owned.push(rw);
        }
    }
    let q_words: Vec<&str> = q_words_owned.iter().map(String::as_str).collect();
    let mut q_lex_words: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for &qw in &q_words {
        for (lem, _) in crate::wordnet::full_lexicon_neighbors(qw).into_iter().take(24) {
            q_lex_words.insert(lem.to_string());
            q_lex_words.insert(stem(lem));
        }
        for (lem, _) in crate::roget::conceptual_neighbors(qw, 32) {
            q_lex_words.insert(lem.to_string());
            q_lex_words.insert(stem(lem));
        }
        for (lem, _) in crate::conceptnet::two_hop_neighbors(qw, 10) {
            q_lex_words.insert(lem.to_string());
            q_lex_words.insert(stem(lem));
        }
        if ix.turns.len() >= 24 {
            for cont in crate::wikidata::upward_containers(qw, 1).into_iter().take(4) {
                for part in cont.split_ascii_whitespace() {
                    if part.len() >= 3 && is_content(part) {
                        q_lex_words.insert(part.to_string());
                        q_lex_words.insert(stem(part));
                    }
                }
            }
        }
    }
    let morph_word_hit = |rw: &str| -> bool {
        let rst = stem(rw);
        q_words.iter().any(|&qw| {
            qw == rw
                || (rst.len() >= 4 && stem(qw) == rst)
                || damerau_levenshtein_le_1(rw, qw)
                || (rw.len() >= 4 && qw.len() >= 4 && (rw.starts_with(qw) || qw.starts_with(rw)) && rw.len().abs_diff(qw.len()) <= 2)
                || (rw.len() >= 5
                    && qw.len() >= 5
                    && rw.as_bytes()[..5] == qw.as_bytes()[..5]
                    && rw.len().abs_diff(qw.len()) <= 6)
        })
    };
    let lex_word_hit = |rw: &str| -> bool {
        q_lex_words.contains(rw) || {
            let rst = stem(rw);
            rst.len() >= 4 && q_lex_words.contains(&rst)
        }
    };
    // Closed-class temporal WH / calendar-date query trigger ("When is/was...", "date of..."):
    // asks for a calendar date/time, so standing rules carrying >= 2 closed-class
    // calendar-unit tokens ("month", "day", "year") govern its formatting.
    let q_is_temporal_when = it.sniper_when
        || it.temporal
        || (!multi_evidence && q_words.iter().any(|&w| w == "date" || w == "dates"));

    let mut latest_by_key: DetHashMap<(&str, &[String]), (usize, i32, i32, i64, bool, i32)> = DetHashMap::default();
    for (ri, r) in ix.standing_rules.iter().enumerate() {
        let mut scope_hits = r.scope.iter().filter(|id| plan.focal_ids.contains(id)).count() as i32;
        if scope_hits == 0 && crate::retrieval::seating::c2_policy_inflects_query(ix, r.turn, plan) {
            scope_hits = 1;
        }
        let cond_exact_hits = r.condition.iter().filter(|id| plan.focal_ids.contains(id)).count() as i32;
        let cond_morph_hits = r
            .condition_words
            .iter()
            .filter(|cw| morph_word_hit(cw))
            .count() as i32;
        let cond_lex_hits = r
            .condition_words
            .iter()
            .filter(|cw| {
                !morph_word_hit(cw)
                    && (lex_word_hit(cw)
                        || (ix.turns.len() >= 24
                            && q_words.iter().any(|&qw| {
                                crate::wordnet::heyting_implication_ic_q16(qw, cw) > 0
                                    || crate::wordnet::heyting_implication_ic_q16(cw, qw) > 0
                            })))
            })
            .count() as i32;
        let cond_total = r.condition_words.len() as i32;
        let cond_matched = (cond_morph_hits + cond_lex_hits).max(cond_exact_hits).min(cond_total);
        let all_cond_matched = cond_total > 0 && cond_matched >= cond_total;
        let unmatched_cond = (cond_total - cond_matched).max(0);
        let cal_unit_hits: i32 = if q_is_temporal_when {
            ix.turns
                .get(r.turn)
                .map(|tr| {
                    let body_l = tr.body_for_headers.to_ascii_lowercase();
                    // hardcode-scan: closed-class (ISO calendar unit words & date placeholders)
                    let mut cnt = 0i32;
                    for u in ["month", "day", "year"] {
                        if body_l.contains(u) {
                            cnt += 1;
                        }
                    }
                    let iso_fmt = ((body_l.contains("mm/") || body_l.contains("/mm") || body_l.contains("mm-"))
                        && (body_l.contains("dd") || body_l.contains("yyyy")))
                        || body_l.contains("exact date")
                        || body_l.contains("format date")
                        || body_l.contains("date format");
                    if cnt >= 2 || iso_fmt { 1 } else { 0 }
                })
                .unwrap_or(0)
        } else {
            0
        };
        let own_lex_hits = ix
            .turns
            .get(r.turn)
            .map(|tr| {
                tr.content
                    .iter()
                    .filter_map(|&id| ix.vocab.names.get(id as usize).map(String::as_str))
                    .filter(|w| !w.starts_with('\u{1}') && is_content(w) && (morph_word_hit(w) || lex_word_hit(w)))
                    .count()
                    .min(3) as i32
            })
            .unwrap_or(0);
        let action_morph_hits = ix
            .turns
            .get(r.turn)
            .map(|tr| {
                tr.content
                    .iter()
                    .filter_map(|&id| ix.vocab.names.get(id as usize).map(String::as_str))
                    .filter(|w| {
                        !w.starts_with('\u{1}')
                            && is_content(w)
                            && !r.condition_words.iter().any(|cw| cw == w)
                            && morph_word_hit(w)
                    })
                    .count() as i32
            })
            .unwrap_or(0);
        let asst_hits = r.assistant_nouns.iter().filter(|id| plan.focal_ids.contains(id)).count() as i32;
        let cond_best = cond_exact_hits.max(cond_morph_hits);
        let eff_scope = if !r.condition_words.is_empty()
            && cond_best == 0
            && cond_lex_hits == 0
            && action_morph_hits == 0
        {
            scope_hits.min(1)
        } else {
            scope_hits.min(3)
        };
        // Condition-clause specificity strictly outranks broad session scope hits and weak lexical bridges.
        let spec = 16 * cond_best
            + 10 * action_morph_hits.min(2)
            + 14 * cal_unit_hits
            + 7 * cond_lex_hits
            + if all_cond_matched && cond_total >= 2 { 10 } else { 0 }
            + 2 * own_lex_hits
            + 2 * eff_scope
            + asst_hits.min(2);
        let fires = if r.has_explicit_scope {
            spec > 0 || it.task_under_rules || multi_evidence
        } else {
            it.task_under_rules || spec >= 2 || asst_hits >= 2 || multi_evidence
        };
        if !fires {
            continue;
        }
        let scope_neg_len = -(r.scope.len() as i32);
        let cand = (ri, spec, scope_neg_len, r.ts, all_cond_matched, unmatched_cond);
        let r_key = (r.policy_key.as_str(), r.condition_words.as_slice());
        match latest_by_key.get(&r_key) {
            Some(&cur) if (cur.3, cur.1, cur.2) >= (cand.3, cand.1, cand.2) => {}
            _ => {
                latest_by_key.insert(r_key, cand);
            }
        }
    }
    let any_full_cond = latest_by_key.values().any(|c| c.4 && c.1 > 0);
    let max_spec = latest_by_key.values().map(|c| c.1).max().unwrap_or(0);
    let mut cands: Vec<(usize, i32, i32, i64)> = latest_by_key
        .into_values()
        .filter(|c| if max_spec > 0 { c.1 > 0 } else { true })
        .map(|(ri, mut spec, scope_neg_len, ts, all_cond, unmatched)| {
            if any_full_cond && !all_cond && unmatched > 0 {
                spec = (spec - 5 * unmatched).max(1);
            } else if unmatched > 0 {
                spec = (spec - 3 * unmatched).max(1);
            }
            (ri, spec, scope_neg_len, ts)
        })
        .collect();
    cands.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then(b.3.cmp(&a.3))
            .then(b.0.cmp(&a.0))
            .then(b.2.cmp(&a.2))
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

/// Closed-class universal spelling-typo check (restricted Damerau-Levenshtein <= 1 for len >= 5).
/// Catches single adjacent transposition (`desing` <-> `design`) or single insertion/deletion
/// (`editting` <-> `editing`) without any word list.
pub(in crate::retrieval) fn damerau_levenshtein_le_1(a: &str, b: &str) -> bool {
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    if ab.len() < 5 || bb.len() < 5 || ab.len().abs_diff(bb.len()) > 1 {
        return false;
    }
    if ab == bb {
        return true;
    }
    if ab.len() == bb.len() {
        let mut diffs = [0usize; 3];
        let mut dc = 0usize;
        for i in 0..ab.len() {
            if ab[i] != bb[i] {
                if dc >= 2 {
                    return false;
                }
                diffs[dc] = i;
                dc += 1;
            }
        }
        return dc == 1
            || (dc == 2
                && diffs[0] + 1 == diffs[1]
                && ab[diffs[0]] == bb[diffs[1]]
                && ab[diffs[1]] == bb[diffs[0]]);
    }
    let (short, long) = if ab.len() < bb.len() { (ab, bb) } else { (bb, ab) };
    let (mut i, mut j, mut skipped) = (0usize, 0usize, false);
    while i < short.len() && j < long.len() {
        if short[i] == long[j] {
            i += 1;
            j += 1;
        } else if !skipped {
            skipped = true;
            j += 1;
        } else {
            return false;
        }
    }
    true
}

/// Does a single rule condition word `cw` match a query/context content word `qw`
/// via exact equality, stem, derivational morphology, edit-distance <= 1, WordNet 1-hop /
/// 2-hop hypernym, or ConceptNet 2-hop?
pub(in crate::retrieval) fn rule_condition_word_matches(cw: &str, qw: &str) -> bool {
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
    let morph_eq = |a: &str, b: &str| {
        if a == b || derives(a, b) || damerau_levenshtein_le_1(a, b) {
            return true;
        }
        let shared = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
        shared >= 7 && shared + 3 >= a.len().min(b.len())
    };
    let lemma_hits = |lemmas: &[&str], w: &str| {
        lemmas
            .iter()
            .any(|l| morph_eq(l, w) || (l.contains('_') && l.split('_').any(|p| morph_eq(p, w))))
    };
    let cst = stem(cw);
    if qw == cw || (cst.len() >= 4 && stem(qw) == cst) || morph_eq(qw, cw) {
        return true;
    }
    thread_local! {
        static COND_PAIR_CACHE: std::cell::RefCell<DetHashMap<(String, String), bool>> =
            std::cell::RefCell::new(DetHashMap::default());
    }
    let key = (cw.to_string(), qw.to_string());
    if let Some(hit) = COND_PAIR_CACHE.with(|c| c.borrow().get(&key).copied()) {
        return hit;
    }
    let c_lemmas = crate::wordnet::expansion_lemmas(cw);
    let matched = lemma_hits(&c_lemmas, qw)
        || lemma_hits(&crate::wordnet::expansion_lemmas(qw), cw)
        || lemma_hits(&crate::wordnet::upward_hypernyms(qw, 2), cw)
        || lemma_hits(&crate::wordnet::upward_hypernyms(cw, 2), qw)
        || ((1..=4).contains(&crate::wordnet::senses(cw).len())
            && lemma_hits(&crate::wordnet::class_to_instances(cw, 1), qw))
        || lemma_hits(&crate::wordnet::gloss_and_compound_concepts(cw), qw)
        || ((1..=5).contains(&crate::wordnet::senses(cw).len())
            && crate::wordnet::gloss_tokens_of(cw)
                .into_iter()
                .take(16)
                .any(|gt| gt.len() >= 4 && morph_eq(gt, qw)))
        || ((1..=5).contains(&crate::wordnet::senses(qw).len())
            && crate::wordnet::gloss_tokens_of(qw)
                .into_iter()
                .take(16)
                .any(|gt| gt.len() >= 4 && morph_eq(gt, cw)))
        || crate::wordnet::full_lexicon_neighbors(qw)
            .into_iter()
            .take(24)
            .any(|(lem, _)| morph_eq(lem, cw))
        || crate::roget::conceptual_neighbors(qw, 24)
            .into_iter()
            .any(|(lem, _)| morph_eq(lem, cw))
        || crate::conceptnet::two_hop_neighbors(qw, 10)
            .iter()
            .any(|&(cn, _)| morph_eq(cn, cw))
        || crate::conceptnet::two_hop_neighbors(cw, 10)
            .iter()
            .any(|&(cn, _)| morph_eq(cn, qw));
    COND_PAIR_CACHE.with(|c| {
        let mut m = c.borrow_mut();
        if m.len() >= 65_536 {
            m.clear();
        }
        m.insert(key, matched);
    });
    matched
}

/// Rule-condition gate: does this standing rule's own "when I ask about X"
/// condition match the query? Exact/stem id overlap with the query focal ids,
/// a shared stem, edit-distance <= 1, WordNet/ConceptNet relation, or strict high-IC
/// rule-turn binding (`>= 2` matched rare terms or single ultra-rare `df <= 3`, IC >= 5 bits).
/// Unconditional rules never match here.
pub(in crate::retrieval) fn rule_condition_matches(
    ix: &Index,
    r: &StandingRule,
    plan: &QueryPlan,
    q_toks: &[String],
) -> bool {
    use crate::retrieval::fusion::aspect_coverage_q16::{aspect_ic_q16, Q16_ONE};
    use crate::retrieval::seating::standing_constraint_q16::score_standing_constraint_q16;
    use std::collections::{BTreeMap, BTreeSet};

    // Two or more rare elicitation-context hits (rule scope enriched from the
    // turns around the rule) are as specific as a condition match.
    if r.scope.iter().filter(|id| plan.focal_ids.contains(id)).count() >= 2 {
        return true;
    }

    // Strict Domain F high-IC binding on the rule turn's OWN content (including `<alpha><digit>`
    // splits like `HTML5 -> html`), NEVER ambient surrounding-turn scope:
    // requires either >= 2 matched rare query tokens (`df <= max(3, ns/8)`) OR total shared IC >= 5.0 bits
    // (ultra-rare exact entity like `html` in a 100-session chat) so generic words never hijack.
    let ns = (ix.session_turns.len() as u32).max(1);
    let hi_df_cap = (ns / 8).clamp(2, 8);
    let mut q_hi_ic: BTreeMap<u32, u64> = BTreeMap::new();
    for qt in &plan.terms {
        if !qt.exact_hit {
            continue;
        }
        let df = ix.s_stats.df.get(&qt.id).copied().unwrap_or(0);
        if df > 0 && df <= hi_df_cap {
            q_hi_ic.insert(qt.id, aspect_ic_q16(df, ns));
        }
    }
    if !q_hi_ic.is_empty() {
        let mut rule_own_ents: BTreeSet<u32> = BTreeSet::new();
        if let Some(tr) = ix.turns.get(r.turn) {
            for &cid in &tr.content {
                rule_own_ents.insert(cid);
            }
        }
        let matched_rare_cnt = q_hi_ic.keys().filter(|id| rule_own_ents.contains(id)).count();
        let is_active = !turn_is_superseded(ix, r.turn);
        let min_ic = if matched_rare_cnt >= 2 { Q16_ONE * 3 } else { Q16_ONE * 5 };
        if score_standing_constraint_q16(true, is_active, &rule_own_ents, &q_hi_ic, min_ic) > 0 {
            return true;
        }
    }

    if r.condition_words.is_empty() {
        return false;
    }
    if r.condition.iter().any(|id| plan.focal_ids.contains(id)) {
        return true;
    }
    let q_content: Vec<&str> = q_toks.iter().map(String::as_str).filter(|w| is_content(w)).collect();
    let allow_onto_cond = ix.turns.len() >= 24;
    r.condition_words.iter().any(|cw| {
        q_content.iter().any(|&qw| {
            rule_condition_word_matches(cw, qw)
                || (cw.is_ascii()
                    && qw.is_ascii()
                    && cw.len() >= 5
                    && qw.len() >= 5
                    && cw.as_bytes()[..5] == qw.as_bytes()[..5]
                    && cw.len().abs_diff(qw.len()) <= 6)
                || (allow_onto_cond
                    && (crate::wordnet::heyting_implication_ic_q16(qw, cw) >= 72
                        || crate::wikidata::upward_containers(qw, 1)
                            .into_iter()
                            .any(|cont| cont.split_ascii_whitespace().any(|p| p == cw))))
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
