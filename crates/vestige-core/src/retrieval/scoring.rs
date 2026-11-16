//! Exact lexical lane: query planning, BM25F, scene/window pooling helpers, rungs and score-shaping steps.
//! Split out of the former single-file `retrieval.rs` with no logic change.

use super::*;

// =========================================================================
// QUERY ANALYSIS
// =========================================================================

#[derive(Clone)]
pub(super) struct QTerm {
    pub(super) id: u32,
    pub(super) weight: f32,
    pub(super) group: Option<usize>,
    pub(super) exact_hit: bool,
}

pub(super) struct QueryPlan {
    pub(super) intent: QueryIntent,
    pub(super) terms: Vec<QTerm>,
    pub(super) groups: Vec<String>,
    /// term id -> group, for proximity scanning
    pub(super) group_of: DetHashMap<u32, usize>,
    pub(super) focal_ids: DetHashSet<u32>,
}

/// Normalize punctuation (preserving internal apostrophes in contractions like `i'm` / `don't`)
/// into single-space-padded lowercase tokens so word-boundary cue matching never misfires on
/// substrings ("Army" != "my", "server" != "serve", "Miami's" != "i'", "claim" != "aim")
/// and never strips literal "$0".
pub(super) fn word_padded(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 4);
    out.push(' ');
    let chars: Vec<char> = text.chars().collect();
    let mut prev_space = true;
    for (idx, &ch) in chars.iter().enumerate() {
        let lower = ch.to_ascii_lowercase();
        let is_apos = (lower == '\'' || lower == '’')
            && idx > 0
            && idx + 1 < chars.len()
            && chars[idx - 1].is_alphanumeric()
            && chars[idx + 1].is_alphanumeric();
        // Glue a currency marker to the amount so "$0" stays the literal token "$0".
        let is_currency = ch == '$' && idx + 1 < chars.len() && chars[idx + 1].is_ascii_digit();
        if lower.is_alphanumeric() || is_currency {
            out.push(if is_currency { '$' } else { lower });
            prev_space = false;
        } else if is_apos {
            out.push('\'');
            prev_space = false;
        } else if !prev_space {
            out.push(' ');
            prev_space = true;
        }
    }
    if !prev_space {
        out.push(' ');
    }
    out
}

pub fn classify(text: &str) -> QueryIntent {
    let raw = raw_tokens(text);
    let slots = wh_slots(text);
    let dates = inline_date_tokens(text);
    let year_count = raw
        .iter()
        .filter(|w| w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()))
        .count();
    let digit_groups = raw.iter().filter(|w| w.chars().any(|c| c.is_ascii_digit())).count();
    let temporal = (slots & WH_TIME) != 0
        || !dates.is_empty()
        || year_count > 0
        || raw.iter().any(|w| token_is_temporal_word(w));
    let content_groups = raw.iter().filter(|w| is_content(w)).collect::<DetHashSet<_>>().len();
    QueryIntent {
        // Phrase cues are gone. plan_query may still set this from a typed-slot
        // value that the corpus actually stored.
        preference: false,
        current_state: false,
        assistant_attribution: false,
        user_attribution: false,
        multi_hop: content_groups >= 3 && (digit_groups >= 2 || slots.count_ones() >= 2),
        inference: false,
        temporal,
        sniper_when: (slots & WH_TIME) != 0 || year_count > 0,
        sniper_how_many: (slots & WH_QTY) != 0,
        sniper_where: (slots & WH_PLACE) != 0,
        sniper_why: (slots & WH_CAUSE) != 0,
        is_event_sequence: year_count >= 2 || dates.len() >= 2,
        task_under_rules: false,
    }
}

pub(super) fn plan_query(ix: &Index, q: &Query, p: &Params) -> QueryPlan {
    let mut intent = classify(&q.text);
    let mut terms = Vec::new();
    let mut groups: Vec<String> = Vec::new();
    let mut group_of = DetHashMap::default();
    let mut focal_ids: DetHashSet<u32> = DetHashSet::default();

    let raw_q = raw_tokens(&q.text);
    let episodic_past_recall = wh_slots(&q.text) != 0
        || intent.sniper_how_many
        || intent.multi_hop
        || intent.temporal;
    if !episodic_past_recall
        && raw_q.iter().any(|w| ix.vocab.get(w).is_some_and(|id| ix.constraint_terms.contains(&id)))
    {
        intent.preference = true;
    }
    let inline_dates = if p.enable_deictic { inline_date_tokens(&q.text) } else { Vec::new() };
    let q_ts = if p.enable_deictic { parse_date(&q.date) } else { 0 };
    let deictic_items = if p.enable_deictic && q_ts != 0 {
        deictic_weighted_tokens(&raw_q, q_ts)
    } else {
        Vec::new()
    };
    let rel_spec = if p.enable_deictic && q_ts != 0 {
        parse_relative_time_specs_from_raw(&raw_q, q_ts)
    } else {
        RelTimeSpec {
            points: Vec::new(),
            windows: Vec::new(),
            scaffold_positions: vec![false; raw_q.len()],
        }
    };
    let rel_prior_active = !rel_spec.points.is_empty() || !rel_spec.windows.is_empty();

    let n_docs = ix.t_own_stats.n as usize;
    let frame_token = |w: &str| {
        ix.vocab.get(w).is_some_and(|id| {
            corpus_frame(n_docs, ix.t_own_stats.df.get(&id).copied().unwrap_or(0))
        })
    };
    let domain_groups = raw_q
        .iter()
        .enumerate()
        .filter(|&(idx, w)| {
            is_content(w)
                && !frame_token(w)
                && !(rel_prior_active && rel_spec.scaffold_positions.get(idx).copied().unwrap_or(false))
        })
        .map(|(_, w)| w)
        .collect::<DetHashSet<_>>()
        .len();
    let can_demote_framing = domain_groups >= 1;
    let zero_rel_scaffolding = rel_prior_active && domain_groups >= 1;

    let nick_bind = ix.nicks.bindings_in(&q.text);
    let mut label_group: DetHashMap<String, usize> = DetHashMap::default();
    let mut seen_toks = DetHashSet::default();
    let mut hyp_added: DetHashSet<u32> = DetHashSet::default();
    let mut expandable: Vec<(String, usize)> = Vec::new();
    let corpus_df = |id: u32| ix.t_own_stats.df.get(&id).copied().unwrap_or(0);
    for (idx, w) in raw_q.iter().enumerate() {
        if !is_content(w) {
            continue;
        }
        if zero_rel_scaffolding && rel_spec.scaffold_positions.get(idx).copied().unwrap_or(false) {
            continue;
        }
        if !seen_toks.insert(w.clone()) {
            continue;
        }
        let is_frame = can_demote_framing && frame_token(w);
        if is_frame {
            if let Some(id) = ix.vocab.get(w) {
                terms.push(QTerm { id, weight: 0.30, group: None, exact_hit: false });
            }
            if p.enable_stemming {
                let st = stem(w);
                if let Some(id) = ix.vocab.get(&stem_term(&st)) {
                    terms.push(QTerm { id, weight: 0.20, group: None, exact_hit: false });
                }
            }
            continue;
        }

        let label = nick_bind.get(w).cloned().unwrap_or_else(|| w.clone());
        let g = if let Some(&g) = label_group.get(&label) {
            g
        } else {
            let g = groups.len();
            groups.push(label.clone());
            label_group.insert(label, g);
            g
        };
        if let Some(id) = ix.vocab.get(w) {
            // Full name owns the person group. A vocative nick ("Mel") that also
            // expands to "melanie" must not score 1.0 on top of that expansion.
            let weight = if nick_bind.get(w).is_some_and(|full| ix.vocab.get(full).is_some()) {
                0.35
            } else {
                1.0
            };
            terms.push(QTerm { id, weight, group: Some(g), exact_hit: true });
            group_of.insert(id, g);
            focal_ids.insert(id);
        }
        if let Some(full) = nick_bind.get(w) {
            if let Some(id) = ix.vocab.get(full) {
                terms.push(QTerm { id, weight: 1.0, group: Some(g), exact_hit: true });
                group_of.insert(id, g);
                focal_ids.insert(id);
            }
            if p.enable_stemming {
                let fst = stem(full);
                if let Some(id) = ix.vocab.get(&stem_term(&fst)) {
                    terms.push(QTerm { id, weight: p.stem_weight, group: Some(g), exact_hit: true });
                    group_of.insert(id, g);
                    focal_ids.insert(id);
                }
            }
        }
        let st = stem(w);
        if p.enable_stemming {
            if let Some(id) = ix.vocab.get(&stem_term(&st)) {
                terms.push(QTerm { id, weight: p.stem_weight, group: Some(g), exact_hit: true });
                group_of.insert(id, g);
                focal_ids.insert(id);
            }
        }
        step9_push_typo_fallback(ix, w, &st, nick_bind.get(w).is_some(), g, &mut terms, &mut group_of);
        expandable.push((w.clone(), g));
    }

    // WordNet query expansion (Voorhees 1994; Fang 2008; Gong 2006), three controls:
    // (1) it runs only when the query's own terms have zero lexical overlap with
    //     the indexed corpus (Ounis-style pre-retrieval gate, `zero_overlap.rs`);
    // (2) an expansion term weighs `hypernym_weight` (0.5) of an original term;
    // (3) only expansion lemmas whose corpus document frequency is > 0 are kept.
    // Nothing is added to documents; a query with any overlap is untouched.
    if p.enable_hypernyms && crate::zero_overlap::ounis_zero_overlap_gate(terms.iter().map(|t| corpus_df(t.id))) {
        for (w, g) in &expandable {
            for lemma in crate::wordnet::expansion_lemmas(w) {
                let surface = ix.vocab.get(lemma);
                let stemmed = if p.enable_stemming { ix.vocab.get(&stem_term(&stem(lemma))) } else { None };
                for id in [surface, stemmed].into_iter().flatten() {
                    if corpus_df(id) > 0 && hyp_added.insert(id) {
                        terms.push(QTerm { id, weight: p.hypernym_weight, group: Some(*g), exact_hit: false });
                        group_of.insert(id, *g);
                    }
                }
            }
        }
    }

    // CamelCase & snake/hyphen reverse query expansion (`localStorage` -> `local`, `storage` at 0.55).
    for raw_word in q.text.split_whitespace() {
        let word = raw_word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-');
        if word.is_empty() {
            continue;
        }
        for (part, _) in surface_aux_tokens(word) {
            if seen_toks.contains(&part) {
                continue;
            }
            if let Some(id) = ix.vocab.get(&part) {
                terms.push(QTerm { id, weight: 0.55, group: None, exact_hit: false });
            }
        }
    }

    // Capitalized proper-noun entity phrases present in index (e.g. "Local H", "For Against").
    let mut seen_ents = DetHashSet::default();
    for ent in proper_entity_tokens(&q.text) {
        if !seen_ents.insert(ent.clone()) {
            continue;
        }
        if let Some(id) = ix.vocab.get(&ent) {
            let parts_body = ent.strip_prefix("\u{1}ent:").unwrap_or("");
            let any_erased = parts_body.split('_').any(|part| !groups.iter().any(|g| g == part));
            if any_erased {
                let g = groups.len();
                groups.push(ent);
                terms.push(QTerm { id, weight: 1.65, group: Some(g), exact_hit: true });
                group_of.insert(id, g);
            } else {
                terms.push(QTerm { id, weight: 0.75, group: None, exact_hit: false });
            }
        }
    }

    // Exact technical handles in the query (`Url`, `Email`, `CommitSha`, `IssueRef`, `Version`, `FilePath`).
    for h_str in extract_exact_technical_handles(&q.text) {
        if let Some(id) = ix.vocab.get(&h_str) {
            terms.push(QTerm { id, weight: 1.5, group: None, exact_hit: true });
            focal_ids.insert(id);
        }
    }

    if p.enable_deictic {
        let mut date_weights: std::collections::BTreeMap<String, f32> = std::collections::BTreeMap::new();
        for d in inline_dates {
            let w = if d.starts_with("__DATE__") {
                2.0
            } else if d.starts_with("__MD__") {
                1.4
            } else if d.starts_with("__YM__") {
                1.2
            } else if d.starts_with("__SY__") {
                1.1
            } else if d.starts_with("__MONTH__")
                || d.starts_with("__EARLY__")
                || d.starts_with("__MID__")
                || d.starts_with("__LATE__")
            {
                1.0
            } else if d.starts_with("__YEAR__") || d.starts_with("__SEASON__") {
                0.85
            } else {
                p.date_weight
            };
            let entry = date_weights.entry(d).or_insert(0.0);
            *entry = (*entry).max(w);
        }
        for (d, scale) in deictic_items {
            let base = if d.starts_with("__DATE__") {
                1.9
            } else if d.starts_with("__MD__") {
                1.1
            } else if d.starts_with("__YM__") || d.starts_with("__MONTH__") {
                0.85
            } else if d.starts_with("__YEAR__") || d.starts_with("__SEASON__") || d.starts_with("__SY__") {
                0.55
            } else {
                p.date_weight
            };
            let w = base * scale;
            let entry = date_weights.entry(d).or_insert(0.0);
            *entry = (*entry).max(w);
        }
        for (d, w) in date_weights {
            if let Some(id) = ix.vocab.get(&d) {
                terms.push(QTerm { id, weight: w, group: None, exact_hit: false });
            }
        }
    }

    QueryPlan { intent, terms, groups, group_of, focal_ids }
}

// =========================================================================
// SCORING
// =========================================================================

pub(super) struct Fields {
    pub(super) w: [f32; 3],
    pub(super) b: [f32; 3],
}

/// 3-Field BM25F (`user`, `assistant`, `blip_caption`) with IDF power law, IDF-weighted
/// distinct-term coverage, and separate `(any_mask, exact_mask)` so 0.45 hypernym hits
/// never poison submodular coverage masks (`covered_mask` / `covered_query_terms`).
pub(super) struct PreparedPlan {
    /// `Some(idf)` when `qt.id` exists in `st.df`, else `None`.
    pub(super) term_idf: Vec<Option<f32>>,
    pub(super) g_idf: Vec<f32>,
    pub(super) g_total: f32,
}

pub(super) fn prepare_plan(plan: &QueryPlan, st: &Stats, p: &Params) -> PreparedPlan {
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

pub(super) fn bm25f_prepared(
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
pub(super) fn bm25f(bag: &Bag, plan: &QueryPlan, st: &Stats, p: &Params, fl: &Fields) -> (f32, u64, u64) {
    let prep = prepare_plan(plan, st, p);
    bm25f_prepared(bag, plan, &prep, st, p, fl)
}

/// True when two distinct query words land within `span` content tokens of each other.
pub(super) fn proximity_hit<'a, I: Iterator<Item = &'a (u32, u32)>>(seq: I, plan: &QueryPlan, span: usize) -> bool {
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

pub(super) fn matched_words(plan: &QueryPlan, mask: u64) -> Vec<String> {
    plan.groups
        .iter()
        .enumerate()
        .filter(|(g, _)| *g < 64 && mask & (1 << g) != 0)
        .map(|(_, w)| w.clone())
        .collect()
}

pub(super) fn order(scores: &[f32], ts: &[i64]) -> Vec<usize> {
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

pub(super) fn jaccard(a: &DetHashSet<u32>, b: &DetHashSet<u32>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = (a.len() + b.len()) as f32 - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

pub(super) fn raise_floor(scores: &mut [f32], lexical_max: f32, frac: f32, keep: impl Fn(usize) -> bool) -> Vec<bool> {
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

pub(super) fn named_partner(a: &str, b: &str) -> bool {
    let role = |s: &str| s.is_empty() || s == "user" || s == "assistant";
    !role(a) && !role(b) && a != b
}

/// Stratified Governing Channel (Satchel SCR 2026; Laird SOAR 2012; Forgy Rete 1982):
/// Ranks active standing imperative policies into `RankResult.standing_rules` without
/// consuming or distorting episodic `sessions` / `turns` slots.
pub(super) fn rank_standing_rules(
    ix: &Index,
    plan: &QueryPlan,
    episodic_user_quote: bool,
    t_mask: &[u64],
    top_k: usize,
) -> Vec<RankedItem> {
    if ix.standing_rules.is_empty() || episodic_user_quote {
        return Vec::new();
    }
    let it = &plan.intent;
    // Multi-evidence recall keeps every standing policy on the stratified channel,
    // including zero-overlap rules. The episodic floor stays off for these intents,
    // so the channel can fire without taking one of the five episodic slots.
    let multi_evidence = it.multi_hop || it.sniper_how_many || it.temporal || it.is_event_sequence;
    let mut latest_by_key: DetHashMap<&str, (usize, i32, i32, i64)> = DetHashMap::default();
    for (ri, r) in ix.standing_rules.iter().enumerate() {
        let scope_hits = r.scope.iter().filter(|id| plan.focal_ids.contains(id)).count() as i32;
        let asst_hits = r.assistant_nouns.iter().filter(|id| plan.focal_ids.contains(id)).count() as i32;
        let spec = 5 * scope_hits + asst_hits;
        let fires = if r.has_explicit_scope {
            spec > 0 || it.task_under_rules || multi_evidence
        } else {
            it.task_under_rules || spec >= 3 || multi_evidence
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

pub(super) fn policy_spec_winner_floor(lexical_max: f32) -> f32 {
    if lexical_max > 0.0 {
        0.715 * lexical_max
    } else {
        0.715 / 0.68
    }
}

pub(super) fn exact_query_group_ids(plan: &QueryPlan) -> DetHashSet<u32> {
    plan.terms
        .iter()
        .filter(|t| t.exact_hit && t.group.is_some())
        .map(|t| t.id)
        .collect()
}

pub(super) fn session_speaker_names(ix: &Index, members: &[usize]) -> DetHashSet<String> {
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

pub(super) fn is_speaker_term(ix: &Index, id: u32, names: &DetHashSet<String>) -> bool {
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

pub(super) fn clause_hits_query(ids: &DetHashSet<u32>, g: &DetHashSet<u32>, ix: &Index, names: &DetHashSet<String>) -> bool {
    ids.iter().any(|id| g.contains(id) && !is_speaker_term(ix, *id, names))
}

pub(super) fn turn_echoes_query(ix: &Index, t: &TurnIx, g: &DetHashSet<u32>, names: &DetHashSet<String>) -> bool {
    if !t.is_pure_info_question {
        return false;
    }
    if clause_hits_query(&t.pre_question_ids, g, ix, names) {
        return false;
    }
    t.pre_question_ids.is_empty() || clause_hits_query(&t.question_clause_ids, g, ix, names)
}

pub(super) fn carry_mode_is_parity() -> bool {
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
pub(super) fn is_first_person_pronoun_cap(trimmed: &str) -> bool {
    let norm: String = trimmed.chars().map(|c| if c == '’' { '\'' } else { c }).collect();
    matches!(norm.as_str(), "I" | "I'm" | "I've" | "I'll" | "I'd")
}

/// Counts non-sentence-initial capitalized proper nouns, digits, or quoted titles at ingest time
/// so `--no-dump` (where `TurnIx.body` is empty) receives the exact same slot count as `--dump`.
pub(super) fn count_slot_entities_mode(prose: &str, exclude_first_person: bool) -> u16 {
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

/// Bitmask of query terms literally present in turn `ti`'s own bag.
#[inline]
pub(super) fn rung_own_qmask(ix: &Index, plan: &QueryPlan, ti: usize) -> u64 {
    let mut m = 0u64;
    let feats = &ix.turns[ti].feats;
    for (idx, qt) in plan.terms.iter().take(64).enumerate() {
        if feats.iter().any(|f| f.id == qt.id) {
            m |= 1u64 << idx;
        }
    }
    m
}

/// Novel information content (sum of IDF of non-query content features) + clean proper-noun/numeric slot density.
pub(super) fn turn_novel_slot_profile(
    ix: &Index,
    plan: &QueryPlan,
    ti: usize,
) -> (f32, usize, bool) {
    let q_ids: HashSet<u32> = plan.terms.iter().map(|qt| qt.id).collect();
    let n_docs = ix.t_stats.n.max(1.0);
    let mut novel_ic = 0.0f32;
    let mut seen_novel = HashSet::new();

    for f in &ix.turns[ti].feats {
        if !f.counts_len || q_ids.contains(&f.id) || !seen_novel.insert(f.id) {
            continue;
        }
        let df = *ix.t_stats.df.get(&f.id).unwrap_or(&1) as f32;
        let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
        novel_ic += idf;
    }

    let slot_entities = ix.turns[ti].slot_entities as usize;
    let has_specific_slot = slot_entities >= 1 || novel_ic >= 14.0;
    (novel_ic, slot_entities, has_specific_slot)
}

/// Rung 1: Declarative Answer-Slot Specificity vs. Question-Echo Penalty.
/// Runs on factual WH-queries (`!pref_on && !is_event_sequence`):
/// - Penalizes turns with `is_question == true` that merely echo query nouns without introducing a slot filler.
/// - Boosts declarative user turns (`!is_question`) that introduce novel entities/numbers/titles absent from Q.
pub(super) fn apply_rung1_declarative_slot_specificity(
    ix: &Index,
    plan: &QueryPlan,
    pref_on: bool,
    is_event_sequence: bool,
    t_turn_raw: &mut [f32],
) {
    if pref_on || is_event_sequence {
        return;
    }
    for (ti, t) in ix.turns.iter().enumerate() {
        if t_turn_raw[ti] <= 0.0 {
            continue;
        }
        let (novel_ic, slot_entities, has_specific_slot) = turn_novel_slot_profile(ix, plan, ti);
        if t.is_question && !has_specific_slot {
            t_turn_raw[ti] *= 0.78;
        } else if !t.is_question && t.speaker == "user" {
            let entity_boost = if slot_entities >= 2 {
                1.14
            } else if slot_entities >= 1 {
                1.07
            } else {
                1.0
            };
            let ic_boost = if novel_ic >= 16.0 {
                1.12
            } else if novel_ic >= 10.0 {
                1.05
            } else {
                1.0
            };
            t_turn_raw[ti] *= entity_boost * ic_boost;
        }
    }
}

/// Step 3A (#2 Panini karaka): Closed-class WH target shape -> required structural slot match.
/// Zero open-domain word lists: uses `wh_slots(q_text)` + ingest structural form/entity traits.
pub(super) fn apply_panini_karaka_slot_shape(
    ix: &Index,
    q_text: &str,
    it: &QueryIntent,
    pref_on: bool,
    t_turn_raw: &mut [f32],
) {
    if pref_on || it.is_event_sequence {
        return;
    }
    let slots = wh_slots(q_text);
    let effective_slots = if it.multi_hop || it.temporal {
        slots & !WH_QTY
    } else {
        slots
    };
    if effective_slots == 0 {
        return;
    }
    let matches_karaka = |t: &TurnIx| -> bool {
        if effective_slots & WH_TIME != 0 && !t.has_temporal && !t.form_clock {
            return false;
        }
        if effective_slots & WH_QTY != 0 && !t.has_numeric && !t.form_digit {
            return false;
        }
        if effective_slots & WH_PERSON != 0
            && !t.form_midcap
            && !t.form_handle
            && t.slot_entities_clean == 0
        {
            return false;
        }
        if effective_slots & WH_PLACE != 0
            && !t.has_spatial
            && !t.form_midcap
            && !t.form_handle
            && t.slot_entities_clean == 0
        {
            return false;
        }
        if effective_slots & WH_CAUSE != 0 && !t.has_causal {
            return false;
        }
        true
    };
    let any_sat = ix
        .turns
        .iter()
        .enumerate()
        .any(|(ti, t)| t_turn_raw[ti] > 0.0 && !t.is_question && matches_karaka(t));
    if !any_sat {
        return;
    }
    for (ti, t) in ix.turns.iter().enumerate() {
        if t_turn_raw[ti] <= 0.0 {
            continue;
        }
        if matches_karaka(t) {
            if !t.is_question {
                t_turn_raw[ti] *= 1.12;
            }
        } else if t.is_question {
            t_turn_raw[ti] *= 0.88;
        }
    }
}

/// Bit 0: Deprioritize redundant assistant mirror turn (`ti` = assistant, adjacent partner `|ti - prev| == 1`
/// already selected in same session) inside greedy top-5 on contended multi-session queries (`max_per_session_top5 < 3`)
/// when query is not asking what the assistant said.
#[inline]
pub(super) fn adj_partner_diversity_factor(
    ix: &Index,
    ti: usize,
    t_sess: usize,
    selected_turns: &[usize],
    assistant_attribution: bool,
    short_reply: bool,
    max_per_session_top5: usize,
) -> f32 {
    if selected_turns.len() < 5
        && max_per_session_top5 < 3
        && !assistant_attribution
        && !short_reply
        && ix.turns[ti].field == 1
        && selected_turns
            .iter()
            .any(|&prev| ix.turns[prev].session == t_sess && (prev as isize - ti as isize).abs() == 1)
    {
        0.25
    } else {
        1.0
    }
}

/// Rung 2: Novelty-Gated `session_opener_lift` Guard.
/// Suppresses promoting a `turn_pos == 0` opener turn `cand` over `top` when `cand` is a question
/// or has strictly lower `novel_ic` + lower query-cue coverage than a later declarative turn in the same session.
pub(super) fn rung2_opener_lift_allowed(
    ix: &Index,
    plan: &QueryPlan,
    top: usize,
    cand: usize,
) -> bool {
    let t_cand = &ix.turns[cand];
    let t_top = &ix.turns[top];
    if t_cand.is_question && !t_top.is_question {
        return false;
    }
    let cand_mask = rung_own_qmask(ix, plan, cand);
    let top_mask = rung_own_qmask(ix, plan, top);
    if top_mask.count_ones() > cand_mask.count_ones() {
        return false;
    }
    let (cand_ic, _, _) = turn_novel_slot_profile(ix, plan, cand);
    let (top_ic, _, _) = turn_novel_slot_profile(ix, plan, top);
    // Do not let a low-information generic opener overwrite a high-information answer turn
    cand_ic + 2.0 >= top_ic
}

/// Rung 4: Hub-Damped RM3 Pseudo-Relevance Feedback on `s_score` (Multi-Session All@5).
/// Takes the top-2 anchored sessions on multi-group queries (`plan.terms.len() >= 3`),
/// extracts top-4 hub-damped non-query tokens shared across them (`2 <= df <= N/6`),
/// and adds a bounded `0.20 * rm3_score` over top-20 session candidates so sibling sessions jump from #6..#10 into #2..#5.
pub(super) fn apply_rung4_hub_damped_rm3(
    ix: &Index,
    plan: &QueryPlan,
    s_score: &mut [f32],
) {
    if plan.terms.len() < 3 || s_score.len() < 6 {
        return;
    }
    let mut ranked: Vec<usize> = (0..s_score.len()).collect();
    ranked.sort_by(|&a, &b| {
        s_score[b]
            .partial_cmp(&s_score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    if s_score[ranked[0]] <= 0.0 || s_score[ranked[1]] <= 0.0 {
        return;
    }
    let q_ids: HashSet<u32> = plan.terms.iter().map(|qt| qt.id).collect();
    let n_docs = ix.t_stats.n.max(1.0);
    let max_df = (n_docs / 6.0).max(4.0);
    let session_term_weights = |si: usize| -> HashMap<u32, f32> {
        let mut map = HashMap::new();
        for &ti in &ix.session_turns[si] {
            for f in &ix.turns[ti].feats {
                if !f.counts_len || q_ids.contains(&f.id) {
                    continue;
                }
                let df = *ix.t_stats.df.get(&f.id).unwrap_or(&0) as f32;
                if df >= 2.0 && df <= max_df {
                    let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln();
                    // Hub damping: penalize high-df hub words quadratically in log space
                    let hub_damp = 1.0 / (1.0 + df).ln().powi(2).max(0.25);
                    *map.entry(f.id).or_insert(0.0) += idf * hub_damp;
                }
            }
        }
        map
    };
    let w0 = session_term_weights(ranked[0]);
    let w1 = session_term_weights(ranked[1]);
    let mut shared_rm3: Vec<(u32, f32)> = w0
        .iter()
        .filter_map(|(&fid, &v0)| w1.get(&fid).map(|&v1| (fid, (v0 * v1).sqrt())))
        .collect();
    shared_rm3.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    shared_rm3.truncate(4);
    let norm_rm3: f32 = shared_rm3.iter().map(|x| x.1).sum();
    if norm_rm3 <= 0.0 {
        return;
    }
    let rm3_weights: HashMap<u32, f32> = shared_rm3
        .into_iter()
        .map(|(fid, w)| (fid, w / norm_rm3))
        .collect();
    let anchor_scale = s_score[ranked[0]];
    // Score only pool@20 sessions (excluding top-2 seeds so Slot 0/1 never distort)
    for &si in ranked.iter().skip(2).take(18) {
        if s_score[si] <= 0.0 {
            continue;
        }
        let mut hit_mass = 0.0f32;
        let mut seen = HashSet::new();
        for &ti in &ix.session_turns[si] {
            for f in &ix.turns[ti].feats {
                if seen.insert(f.id) {
                    if let Some(&rw) = rm3_weights.get(&f.id) {
                        hit_mass += rw;
                    }
                }
            }
        }
        if hit_mass > 0.0 {
            s_score[si] += 0.20 * anchor_scale * hit_mass;
        }
    }
}

/// Step 5 (Move 3): Continuous HeidelTime Relative-Time Anchor + Scaffolding Span Detector.
pub(super) struct RelTimeSpec {
    /// Point anchors: `(target_ts, sigma_days)`
    pub(super) points: Vec<(i64, f32)>,
    /// Bounded span windows: `(start_ts, end_ts)`
    pub(super) windows: Vec<(i64, i64)>,
    /// Per-token positions in `raw_tokens(&q.text)` that belong strictly to parsed relative-time scaffolding
    /// so BM25F can zero-weight pure temporal scaffolding words when domain words remain (`81507db6` guard).
    pub(super) scaffold_positions: Vec<bool>,
}

pub(super) fn parse_relative_time_specs_from_raw(raw: &[String], query_ts: i64) -> RelTimeSpec {
    let mut spec = RelTimeSpec {
        points: Vec::new(),
        windows: Vec::new(),
        scaffold_positions: vec![false; raw.len()],
    };
    if query_ts == 0 {
        return spec;
    }
    const DAY: i64 = 86_400;
    for (i, w) in raw.iter().enumerate() {
        let prev = if i > 0 { raw[i - 1].as_str() } else { "" };
        let next = raw.get(i + 1).map(String::as_str).unwrap_or("");
        match w.as_str() {
            "yesterday" | "tonight" => {
                if w == "yesterday" {
                    spec.points.push((query_ts - DAY, 1.5));
                    spec.scaffold_positions[i] = true;
                }
            }
            "night" if prev == "last" => {
                spec.points.push((query_ts - DAY, 1.5));
                spec.scaffold_positions[i - 1] = true;
                spec.scaffold_positions[i] = true;
            }
            "ago" | "earlier" => {
                if matches!(prev, "days" | "weeks" | "months" | "years" | "day" | "week" | "month" | "year") {
                    let has_of = i >= 3 && raw[i - 2] == "of";
                    let q_pos = if has_of {
                        Some(i - 3)
                    } else if i >= 2 {
                        Some(i - 2)
                    } else {
                        None
                    };
                    let q = q_pos.map(|p| raw[p].as_str()).unwrap_or("");
                    let n = if q == "few" {
                        3
                    } else if q == "several" {
                        4
                    } else {
                        parse_qty_word(q)
                    };
                    if n > 0 {
                        let (unit_d, sigma) = match prev.trim_end_matches('s') {
                            "day" => (1i64, (1.5f32).max(n as f32 * 0.25)),
                            "week" => (7, (3.5f32).max(n as f32 * 1.5)),
                            "month" => (30, (7.0f32).max(n as f32 * 3.0)),
                            _ => (365, 25.0),
                        };
                        spec.points.push((query_ts - n * unit_d * DAY, sigma));
                        spec.scaffold_positions[i] = true;
                        spec.scaffold_positions[i - 1] = true;
                        if let Some(qp) = q_pos {
                            spec.scaffold_positions[qp] = true;
                            if qp > 0 && raw[qp - 1] == "a" {
                                spec.scaffold_positions[qp - 1] = true;
                            }
                        }
                    }
                }
            }
            "week" if matches!(prev, "last" | "past" | "previous") && !matches!(next, "ago" | "earlier" | "later") => {
                spec.windows.push((query_ts - 11 * DAY, query_ts));
                spec.scaffold_positions[i - 1] = true;
                spec.scaffold_positions[i] = true;
            }
            "weekend" if matches!(prev, "last" | "past" | "previous") => {
                spec.windows.push((query_ts - 10 * DAY, query_ts));
                spec.scaffold_positions[i - 1] = true;
            }
            "month" if matches!(prev, "last" | "past" | "previous") && !matches!(next, "ago" | "earlier" | "later") => {
                spec.windows.push((query_ts - 32 * DAY, query_ts));
                spec.scaffold_positions[i - 1] = true;
                spec.scaffold_positions[i] = true;
            }
            "year" if matches!(prev, "last" | "past" | "previous") && !matches!(next, "ago" | "earlier" | "later") => {
                spec.windows.push((query_ts - 366 * DAY, query_ts));
                spec.scaffold_positions[i - 1] = true;
                spec.scaffold_positions[i] = true;
            }
            "days" | "weeks" | "months" | "years" if !matches!(next, "ago" | "earlier" | "later") => {
                let has_of = i >= 3 && raw[i - 2] == "of";
                let q_pos = if has_of { i - 3 } else if i >= 2 { i - 2 } else { 0 };
                let prep_pos = if has_of { i.checked_sub(4) } else { i.checked_sub(2) };
                // Strictly deictic lookbacks ("past"/"last"); never treat durations ("in two weeks") as anchored to query_ts.
                let mut matched_prep = prep_pos.filter(|&pp| matches!(raw[pp].as_str(), "past" | "last"));
                // Also support "in the past few weeks" / "past a few days"
                if matched_prep.is_none() && !has_of && i >= 3 && matches!(raw[i - 3].as_str(), "past" | "last") && raw[i - 2] == "a" {
                    matched_prep = Some(i - 3);
                }
                if let Some(pp) = matched_prep {
                    let q_tok = if has_of { raw[q_pos].as_str() } else { prev };
                    let n = if matches!(q_tok, "few" | "several") {
                        6
                    } else if q_tok == "couple" {
                        3
                    } else {
                        parse_qty_word(q_tok)
                    };
                    if n > 0 && n <= 120 {
                        let unit = match w.trim_end_matches('s') {
                            "day" => 1i64,
                            "week" => 7,
                            "month" => 31,
                            _ => 366,
                        };
                        spec.windows.push((query_ts - n * unit * DAY - DAY, query_ts));
                        spec.scaffold_positions[i] = true;
                        spec.scaffold_positions[pp] = true;
                        if has_of {
                            spec.scaffold_positions[q_pos] = true;
                        } else if i > 0 {
                            spec.scaffold_positions[i - 1] = true;
                        }
                    }
                }
            }
            _ => {
                if matches!(prev, "last" | "past" | "previous") && WEEKDAYS.contains(&w.as_str()) {
                    use chrono::Datelike;
                    if let Some(dt) = chrono::DateTime::from_timestamp(query_ts, 0) {
                        let today = dt.weekday().num_days_from_monday() as i64;
                        let target = WEEKDAYS.iter().position(|d| d == w).unwrap_or(0) as i64;
                        let raw_back = (today - target + 7) % 7;
                        // Modulo 0 -> 7 fix: "last Friday" on a Friday is 7 days ago, never yesterday.
                        let back = if raw_back == 0 { 7 } else { raw_back };
                        spec.points.push((query_ts - back * DAY, 2.0));
                        spec.scaffold_positions[i - 1] = true;
                    }
                }
            }
        }
    }
    spec
}

pub(super) fn apply_move3_heideltime_anchor(
    ix: &Index,
    q: &Query,
    multi_hop: bool,
    s_score: &mut [f32],
    t_turn_raw: &mut [f32],
    t_raw: &mut [f32],
    t_own: &mut [f32],
) {
    let query_ts = parse_date(&q.date);
    if query_ts == 0 {
        return;
    }
    let raw = raw_tokens(&q.text);
    let spec = parse_relative_time_specs_from_raw(&raw, query_ts);
    if spec.points.is_empty() && spec.windows.is_empty() {
        return;
    }
    let out_floor_s = if multi_hop { 0.96 } else { 0.88 };
    let out_floor_t = if multi_hop { 0.98 } else { 0.92 };
    let sess_mult = |ts: i64| -> (f32, f32, bool) {
        if ts <= 0 {
            return (1.0, 1.0, false);
        }
        let mut sm = 1.0f32;
        let mut tm = 1.0f32;
        let mut hit = false;
        if !spec.windows.is_empty() {
            let inside = spec
                .windows
                .iter()
                .any(|&(a, b)| ts >= a - 43_200 && ts <= b + 43_200);
            if inside {
                sm *= 1.24;
                tm *= 1.18;
                hit = true;
            } else {
                sm *= out_floor_s;
                tm *= out_floor_t;
            }
        }
        if !spec.points.is_empty() {
            let mut best_g = 0.0f32;
            for &(target_ts, sigma) in &spec.points {
                let dt_days = (ts - target_ts).abs() as f32 / 86_400.0;
                let g = (-0.5 * (dt_days / sigma) * (dt_days / sigma)).exp();
                if g > best_g {
                    best_g = g;
                }
            }
            if best_g >= 0.35 {
                hit = true;
            }
            sm *= out_floor_s + (1.25 - out_floor_s) * best_g;
            tm *= out_floor_t + (1.15 - out_floor_t) * best_g;
        }
        (sm, tm, hit)
    };
    let any_anchor_hit = s_score
        .iter()
        .enumerate()
        .any(|(si, &sc)| sc > 0.0 && sess_mult(ix.session_ts[si]).2);
    if !any_anchor_hit {
        return;
    }
    for (si, sc) in s_score.iter_mut().enumerate() {
        if *sc > 0.0 {
            let (sm, _, _) = sess_mult(ix.session_ts[si]);
            *sc *= sm;
        }
    }
    for ti in 0..ix.turns.len() {
        if t_turn_raw[ti] <= 0.0 && t_raw[ti] <= 0.0 && t_own[ti] <= 0.0 {
            continue;
        }
        let (_, tm, _) = sess_mult(ix.session_ts[ix.turns[ti].session]);
        t_turn_raw[ti] *= tm;
        t_raw[ti] *= tm;
        t_own[ti] *= tm;
    }
}

// ---- moved verbatim from former `src/steps7_10.rs` (was `include!()`d into retrieval.rs) ----
// Computer 2 lane: Steps 7-10 helpers (included from retrieval.rs right after rungs.rs).
// Ingest-time extraction only reads `prose_body`; ranking helpers never read `TurnIx.body`
// (empty under `--no-dump`). Closed-class grammar only; no domain word lists.

/// Sam's 7B rare-modifier idf threshold.
pub(super) const STEP7_RARE_IDF: f32 = 2.2;

/// BM25 idf over turn-own stats, `((n - df + 0.5) / (df + 0.5) + 1).ln().max(0)`.
pub(super) fn step7_idf(stats: &Stats, term: u32) -> f32 {
    let df = stats.df.get(&term).copied().unwrap_or(0) as f32;
    ((stats.n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0)
}

/// Lowercased query words written capitalized somewhere other than sentence start.
pub(super) fn step7_query_proper_surfaces(q_text: &str) -> DetHashSet<String> {
    let mut out = DetHashSet::default();
    let mut sentence_start = true;
    for raw_word in q_text.split_whitespace() {
        let word = raw_word.trim_matches(|c: char| !c.is_alphanumeric());
        let word = word.strip_suffix("'s").or_else(|| word.strip_suffix("’s")).unwrap_or(word);
        if !sentence_start && word.chars().next().is_some_and(|c| c.is_uppercase()) {
            out.insert(word.to_lowercase());
        }
        sentence_start = raw_word.ends_with(['.', '!', '?']);
    }
    out
}

/// Step 7B (Lex specialis) at HOOK B: when the query carries a rare modifier group, sessions
/// that contain it damp their turns that lack it, so the specific rule outranks the general.
pub(super) fn apply_step7_lex_specialis(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    pref_on: bool,
    it: &QueryIntent,
    t_own: &[f32],
    t_scene: &mut [f32],
) {
    if pref_on || it.is_event_sequence || plan.groups.len() < 3 {
        return;
    }
    // v6: the "special law" is a named referent: only proper-name query groups compete.
    let proper = step7_query_proper_surfaces(q_text);
    let mut top_mod: Option<(usize, f32)> = None;
    for (g_idx, g_surf) in plan.groups.iter().enumerate() {
        let gl = g_surf.to_lowercase();
        if gl.len() < 3 || stopwords().contains(gl.as_str()) {
            continue;
        }
        let surf = gl.strip_prefix('\u{1}').and_then(|r| r.strip_prefix("ent:")).unwrap_or(&gl);
        if !surf.split('_').all(|part| proper.contains(part)) {
            continue;
        }
        let max_g_idf = plan
            .terms
            .iter()
            .filter(|qt| qt.group == Some(g_idx))
            .map(|qt| step7_idf(&ix.t_own_stats, qt.id))
            .fold(0.0_f32, f32::max);
        if max_g_idf >= STEP7_RARE_IDF && top_mod.is_none_or(|(_, best)| max_g_idf > best) {
            top_mod = Some((g_idx, max_g_idf));
        }
    }
    let Some((mod_g, _)) = top_mod else {
        return;
    };
    let mod_term_ids: DetHashSet<u32> =
        plan.terms.iter().filter(|qt| qt.group == Some(mod_g)).map(|qt| qt.id).collect();
    let turn_has_mod: Vec<bool> = ix
        .turns
        .iter()
        .map(|t| {
            t.feats.iter().any(|f| f.counts_len && mod_term_ids.contains(&f.id))
                || t.content.iter().any(|id| mod_term_ids.contains(id))
                // v2: a turn spoken by the modifier (named speaker) carries it by identity.
                || raw_tokens(&t.speaker)
                    .iter()
                    .any(|w| ix.vocab.get(w).is_some_and(|id| mod_term_ids.contains(&id)))
        })
        .collect();
    // v8: precedence, not damping. Inside a session that states the special rule, a general
    // turn may not outrank the best special turn; general turns already below it are untouched.
    for sess_turns in &ix.session_turns {
        let best_special = sess_turns
            .iter()
            .filter(|&&ti| !ix.turns[ti].is_question && turn_has_mod[ti] && t_own[ti] > 0.0)
            .map(|&ti| t_scene[ti])
            .fold(0.0_f32, f32::max);
        if best_special <= 0.0 {
            continue;
        }
        for &ti in sess_turns {
            if !turn_has_mod[ti] && t_scene[ti] > best_special {
                t_scene[ti] = (0.82 * t_scene[ti]).max(best_special);
            }
        }
    }
}

/// Step 8 anchor set A(q): query terms with turn-own idf >= 1.5, plus focal ids.
pub(super) fn step8_anchor_set(ix: &Index, plan: &QueryPlan) -> DetHashSet<u32> {
    let mut a: DetHashSet<u32> = plan
        .terms
        .iter()
        .filter(|qt| step7_idf(&ix.t_own_stats, qt.id) >= 1.5)
        .map(|qt| qt.id)
        .collect();
    a.extend(plan.focal_ids.iter().copied());
    a
}

/// T(t) restricted to the anchor set: content ids and length-counted feature ids.
pub(super) fn step8_anchor_hits(t: &TurnIx, anchors: &DetHashSet<u32>) -> std::collections::BTreeSet<u32> {
    let mut out: std::collections::BTreeSet<u32> = t.content.iter().copied().filter(|id| anchors.contains(id)).collect();
    out.extend(t.feats.iter().filter(|f| f.counts_len && anchors.contains(&f.id)).map(|f| f.id));
    out
}

/// Step 8A (Kelal u-ferat, NOVEL) at HOOK A: a later same-speaker particular that refines an
/// earlier general statement on a shared anchor inherits part of the general's rank.
pub(super) fn apply_step8_kelal_u_ferat(ix: &Index, plan: &QueryPlan, t_turn_raw: &mut [f32]) {
    let anchors = step8_anchor_set(ix, plan);
    if anchors.is_empty() {
        return;
    }
    let base = t_turn_raw.to_vec();
    for st in &ix.session_turns {
        if st.len() < 2 {
            continue;
        }
        for pos_j in 1..st.len() {
            let tj = st[pos_j];
            let turn_j = &ix.turns[tj];
            // v2: refinement of one's own statement is a first-person (field 0) act; assistant
            // expansions are elaborations, not a speaker refining their own general claim.
            if turn_j.field != 0 || turn_j.is_question || base[tj] <= 0.0 {
                continue;
            }
            let hits_j = step8_anchor_hits(turn_j, &anchors);
            if hits_j.is_empty() {
                continue;
            }
            if !(turn_j.slot_entities_clean >= 1 || turn_j.form_digit || turn_j.form_clock || turn_j.has_numeric) {
                continue;
            }
            let mut best: Option<(f32, usize)> = None;
            // v3: the general is the speaker's own immediately preceding turn (the particular
            // refines what that speaker just said), not any earlier turn in a 6-turn window.
            let prev_own = st[pos_j.saturating_sub(6)..pos_j]
                .iter()
                .rev()
                .copied()
                .find(|&ti| ix.turns[ti].speaker == turn_j.speaker);
            let general: Vec<usize> = prev_own.into_iter().collect();
            for &ti in &general {
                let turn_i = &ix.turns[ti];
                if turn_i.is_question || base[ti] <= base[tj] {
                    continue;
                }
                let hits_i = step8_anchor_hits(turn_i, &anchors);
                if hits_i.is_disjoint(&hits_j) {
                    continue;
                }
                let introduces = turn_j.entities.iter().any(|e| !turn_i.entities.contains(e))
                    || (turn_j.form_digit && !turn_i.form_digit)
                    || turn_j.slot_entities_clean > turn_i.slot_entities_clean;
                if !introduces {
                    continue;
                }
                let excess = base[ti] - base[tj];
                if best.is_none_or(|(b, _)| excess > b) {
                    best = Some((excess, ti));
                }
            }
            if let Some((excess, ti)) = best {
                let src = &ix.turns[ti];
                let beta = if src.slot_entities_clean == 0 && !src.form_digit { 0.55 } else { 0.35 };
                t_turn_raw[tj] += beta * excess;
            }
        }
    }
}

/// Step 8B (Homeric ring composition, NOVEL) at HOOK A after 8A: when a speaker's anchored
/// turns in a session open and close on the same stem, interior novel particulars get a lift
/// proportional to the opener.
pub(super) fn apply_step8_homeric_ring(ix: &Index, plan: &QueryPlan, t_turn_raw: &mut [f32]) {
    let anchors = step8_anchor_set(ix, plan);
    if anchors.is_empty() {
        return;
    }
    let base = t_turn_raw.to_vec();
    for st in &ix.session_turns {
        // v2: the ring is a first-person (field 0) narrative frame.
        for field in 0..1usize {
            let ring: Vec<usize> = st
                .iter()
                .copied()
                .filter(|&ti| {
                    let t = &ix.turns[ti];
                    t.field == field && base[ti] > 0.0 && !step8_anchor_hits(t, &anchors).is_empty()
                })
                .collect();
            if ring.len() < 3 {
                continue;
            }
            let a = ring[0];
            let k = ring[ring.len() - 1];
            let stems_a: std::collections::BTreeSet<u32> = ix.turns[a].seq.iter().map(|&(_, s)| s).collect();
            let closes = ix.turns[k].seq.iter().any(|&(_, s)| stems_a.contains(&s));
            let interior = &ring[1..ring.len() - 1];
            if !closes || !interior.iter().any(|&c| !ix.turns[c].is_question) {
                continue;
            }
            let ta = &ix.turns[a];
            for &c in interior {
                let tc = &ix.turns[c];
                if tc.is_question {
                    continue;
                }
                let novel = tc.content.iter().filter(|id| !ta.content.contains(id)).count();
                let novel_ratio = novel as f32 / tc.content.len().max(1) as f32;
                if novel_ratio >= 0.50
                    && (tc.slot_entities_clean > ta.slot_entities_clean || (tc.form_digit && !ta.form_digit))
                {
                    // v3: the center may rise toward the frame, never above the opener.
                    let lifted = t_turn_raw[c] + 0.18 * base[a] * novel_ratio;
                    t_turn_raw[c] = lifted.min(base[a].max(t_turn_raw[c]));
                }
            }
        }
    }
}

/// Step 9A: byte-level Damerau-Levenshtein distance exactly 1 (one edit or one adjacent swap).
pub(crate) fn damerau_lev_1(a: &[u8], b: &[u8]) -> bool {
    let la = a.len();
    let lb = b.len();
    if la.abs_diff(lb) > 1 {
        return false;
    }
    if la == lb {
        let mut diffs = Vec::with_capacity(3);
        for i in 0..la {
            if a[i] != b[i] {
                diffs.push(i);
                if diffs.len() > 2 {
                    return false;
                }
            }
        }
        return match diffs.len() {
            1 => true,
            2 => {
                let (i, j) = (diffs[0], diffs[1]);
                j == i + 1 && a[i] == b[j] && a[j] == b[i]
            }
            _ => false,
        };
    }
    let (short, long) = if la < lb { (a, b) } else { (b, a) };
    let mut i = 0;
    let mut j = 0;
    let mut skipped = false;
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

/// Step 9A: the unique lowercase vocab word within Damerau-Levenshtein 1 of a zero-DF query
/// word (len >= 6, same first byte). Ambiguous (2+ candidates) returns None.
pub(crate) fn find_unique_typo_fallback(vocab_names: &[String], w: &str) -> Option<u32> {
    let wb = w.as_bytes();
    if wb.len() < 6 || !wb.iter().all(|b| b.is_ascii_lowercase()) {
        return None;
    }
    let first = wb[0];
    let mut found: Option<u32> = None;
    for (id, cand) in vocab_names.iter().enumerate() {
        let cb = cand.as_bytes();
        if cb.is_empty() || cb[0] != first || cb.starts_with(b"\x01") {
            continue;
        }
        if !cb.iter().all(|b| b.is_ascii_lowercase()) {
            continue;
        }
        // v2: only the Damerau adjacent-swap edit. Single substitutions/insertions between
        // real words collide constantly ("brought"/"bought", "likely"/"lively").
        if cb.len() == wb.len() && damerau_lev_1(wb, cb) && wb.iter().zip(cb).filter(|(x, y)| x != y).count() == 2 {
            if found.is_some() {
                return None;
            }
            found = Some(id as u32);
        }
    }
    found
}

/// Step 9B ingest: number of lines that open with a list marker (`1.` `2)` `- ` `* ` `•`).
pub(super) fn step9_list_item_count(prose_body: &str) -> u8 {
    let mut n: u32 = 0;
    for line in prose_body.lines() {
        let l = line.trim_start();
        let digits = l.bytes().take_while(|b| b.is_ascii_digit()).count();
        let numbered = digits > 0 && digits <= 3 && matches!(l.as_bytes().get(digits), Some(b'.') | Some(b')'));
        if numbered || l.starts_with("- ") || l.starts_with("* ") || l.starts_with('•') {
            n += 1;
        }
    }
    n.min(u8::MAX as u32) as u8
}

/// Step 9B ingest: the first game move number at a line start ("14. Nf3", "14... e5", "3. O-O").
pub(super) fn step9_first_move_num(prose_body: &str) -> u16 {
    for line in prose_body.lines() {
        let l = line.trim_start();
        let digits = l.bytes().take_while(|b| b.is_ascii_digit()).count();
        if digits == 0 || digits > 3 {
            continue;
        }
        let rest = &l[digits..];
        let Some(rest) = rest.strip_prefix("...").or_else(|| rest.strip_prefix('.')) else {
            continue;
        };
        let mv = rest.trim_start().as_bytes();
        let is_move = match mv {
            [b'O', b'-', b'O', ..] => true,
            [p, f, r, ..] if b"KQRBN".contains(p) => {
                (f.is_ascii_lowercase() && (b'a'..=b'h').contains(f)) || *f == b'x' || (b'1'..=b'8').contains(r)
            }
            [f, r, ..] => (b'a'..=b'h').contains(f) && ((b'1'..=b'8').contains(r) || *r == b'x'),
            _ => false,
        };
        if is_move {
            if let Ok(n) = l[..digits].parse::<u16>() {
                return n;
            }
        }
    }
    0
}

/// Closed-class ordinals 1..10 for Step 9B.
pub(super) fn step9_ordinal(w: &str) -> Option<u8> {
    let k = match w {
        "first" | "1st" => 1,
        "second" | "2nd" => 2,
        "third" | "3rd" => 3,
        "fourth" | "4th" => 4,
        "fifth" | "5th" => 5,
        "6th" => 6,
        "7th" => 7,
        "8th" => 8,
        "9th" => 9,
        "10th" => 10,
        _ => return None,
    };
    Some(k)
}

/// Step 9B (Method of loci) at HOOK A: "the <ordinal>" item queries lift list turns long
/// enough to hold that slot; "after/on move N" lifts the turn that opens move N+1 (or N).
pub(super) fn apply_step9_method_of_loci(ix: &Index, q_text: &str, t_turn_raw: &mut [f32]) {
    let toks = raw_tokens(q_text);
    let mut ord_k: Option<u8> = None;
    let mut move_n: Option<u16> = None;
    for (i, w) in toks.iter().enumerate() {
        if i > 0 && toks[i - 1] == "the" {
            if let Some(k) = step9_ordinal(w) {
                let next = toks.get(i + 1).map(String::as_str);
                let temporal = matches!(next, Some("time" | "day" | "week" | "month" | "year" | "session"));
                if !temporal && ord_k.is_none() {
                    ord_k = Some(k);
                }
            }
        }
        if (w == "after" || w == "on") && toks.get(i + 1).is_some_and(|m| m == "move") {
            if let Some(n) = toks.get(i + 2).and_then(|n| n.parse::<u16>().ok()) {
                move_n = Some(n);
            }
        }
    }
    if let Some(k) = ord_k {
        for (ti, t) in ix.turns.iter().enumerate() {
            if !t.is_question && t_turn_raw[ti] > 0.0 && t.list_item_count >= k {
                t_turn_raw[ti] *= 1.15;
            }
        }
    }
    if let Some(n) = move_n {
        for (ti, t) in ix.turns.iter().enumerate() {
            if t_turn_raw[ti] > 0.0 && t.first_move_num > 0 && (t.first_move_num == n + 1 || t.first_move_num == n) {
                t_turn_raw[ti] *= 1.35;
            }
        }
    }
}

/// Step 9A wiring in plan_query: a zero-DF query word (no exact, stem, or nick hit) maps to
/// its unique Damerau-Levenshtein-1 vocab neighbour (0.75) and that neighbour's stem (0.65).
pub(super) fn step9_push_typo_fallback(
    ix: &Index,
    w: &str,
    st: &str,
    has_nick: bool,
    g: usize,
    terms: &mut Vec<QTerm>,
    group_of: &mut DetHashMap<u32, usize>,
) {
    if has_nick || ix.vocab.get(w).is_some() || ix.vocab.get(&stem_term(st)).is_some() {
        return;
    }
    let Some(typo_id) = find_unique_typo_fallback(&ix.vocab.names, w) else {
        return;
    };
    terms.push(QTerm { id: typo_id, weight: 0.75, group: Some(g), exact_hit: false });
    group_of.insert(typo_id, g);
    if let Some(cand_str) = ix.vocab.names.get(typo_id as usize) {
        if let Some(cs_id) = ix.vocab.get(&stem_term(&stem(cand_str))) {
            terms.push(QTerm { id: cs_id, weight: 0.65, group: Some(g), exact_hit: false });
            group_of.insert(cs_id, g);
        }
    }
}

// ---- moved verbatim from former `src/steps11_20.rs` (was `include!()`d into retrieval.rs) ----

pub(super) fn step13_build_query_word_set(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
) -> (String, std::collections::BTreeSet<String>) {
    let q_lower = q_text.to_ascii_lowercase();
    let mut query_words = std::collections::BTreeSet::new();
    for tok in raw_tokens(q_text) {
        let low = tok.to_ascii_lowercase();
        if !low.is_empty() {
            query_words.insert(low);
        }
    }
    for qt in &plan.terms {
        if let Some(name) = ix.vocab.names.get(qt.id as usize) {
            query_words.insert(name.to_ascii_lowercase());
        }
    }
    (q_lower, query_words)
}

pub(super) fn step13_is_rare_bridge_entity_candidate(
    ix: &Index,
    eid: u32,
    q_lower: &str,
    query_words: &std::collections::BTreeSet<String>,
) -> bool {
    if is_technical_handle(&ix.vocab, eid) {
        return false;
    }
    let Some(name) = ix.vocab.names.get(eid as usize) else {
        return false;
    };
    let Some(surface) = name.strip_prefix("\u{1}ent:") else {
        return false;
    };
    let low = surface.to_ascii_lowercase();
    if low.len() < 4 || query_words.contains(&low) {
        return false;
    }
    if MONTHS.contains(&low.as_str())
        || WEEKDAYS.contains(&low.as_str())
        || STARTER_STOP.contains(&low.as_str())
    {
        return false;
    }
    let spaced = low.replace('_', " ");
    if q_lower.contains(&spaced) {
        return false;
    }
    let parts: Vec<&str> = low.split('_').filter(|p| !p.is_empty()).collect();
    !(parts.is_empty()
        || parts.iter().all(|&p| {
            query_words.contains(p)
                || MONTHS.contains(&p)
                || WEEKDAYS.contains(&p)
                || STARTER_STOP.contains(&p)
                || stopwords().contains(p)
        }))
}

/// Step 13 (`#42 Khipu Two-Hop Rare-Entity Turn Bridge` on `t_scene`):
/// Extracts non-query rare proper entities (`\u{1}ent:*`, `2 <= exact_df <= 8`, `idf >= 3.2`)
/// from strong substantive hop-1 turns (`top-3`, `t_own >= 0.40 * top_t_own`) and applies a bounded
/// proportional score lift (`<= +28%` of `base_sc`, capped at `0.80 * src_scene`) to
/// non-question, non-backchannel, non-superseded hop-2 turns in *other* sessions that already
/// match at least one exact query group and sit below the top-rank ceiling.
pub(super) fn apply_step13_khipu_turn_bridge(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    pref_on: bool,
    it: &QueryIntent,
    t_own: &[f32],
    t_exact_mask: &[u64],
    t_scene: &mut [f32],
) {
    use std::collections::{BTreeMap, BTreeSet};

    let gl = plan.groups.len().min(64);
    if pref_on
        || it.is_event_sequence
        || it.assistant_attribution
        || gl == 0
        || t_own.is_empty()
        || t_scene.len() != t_own.len()
        || t_exact_mask.len() != t_own.len()
    {
        return;
    }
    let group_mask = if gl >= 64 { u64::MAX } else { (1u64 << gl) - 1 };

    let (q_lower, query_words) = step13_build_query_word_set(ix, q_text, plan);

    let mut hop1: Vec<(f32, usize, usize)> = Vec::new();
    for (ti, &own) in t_own.iter().enumerate() {
        if own <= 0.0 || (t_exact_mask[ti] & group_mask) == 0 {
            continue;
        }
        let t = &ix.turns[ti];
        if t.field != 0 || t.is_question || t.is_backchannel || turn_is_superseded(ix, ti) {
            continue;
        }
        hop1.push((own, t.session, ti));
    }
    if hop1.is_empty() {
        return;
    }
    hop1.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    let top_own = hop1[0].0;
    let top_session = hop1[0].1;
    hop1.retain(|&(own, _, _)| own >= 0.40 * top_own);
    hop1.truncate(3);

    let mut candidate_eids: BTreeSet<u32> = BTreeSet::new();
    for &(_own, _sess, ti) in &hop1 {
        for &eid in &ix.turns[ti].entities {
            if step13_is_rare_bridge_entity_candidate(ix, eid, &q_lower, &query_words) {
                candidate_eids.insert(eid);
            }
        }
    }
    if candidate_eids.is_empty() {
        return;
    }

    let mut exact_df: BTreeMap<u32, u32> = BTreeMap::new();
    for &eid in &candidate_eids {
        exact_df.insert(eid, 0);
    }
    for t in &ix.turns {
        for &eid in &t.entities {
            if let Some(cnt) = exact_df.get_mut(&eid) {
                *cnt += 1;
            }
        }
    }

    let n_docs = ix.t_own_stats.n.max(ix.turns.len() as f32).max(1.0);
    let mut rare_bridges: BTreeMap<u32, (f32, f32, usize)> = BTreeMap::new();
    for (&eid, &df) in &exact_df {
        if !(2..=8).contains(&df) {
            continue;
        }
        let idf = ((n_docs - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln();
        if idf < 3.2 {
            continue;
        }
        let mut best_src: Option<(f32, usize, usize)> = None;
        for &(_own, sess, ti) in &hop1 {
            if ix.turns[ti].entities.contains(&eid) {
                let sc = t_scene[ti].max(t_own[ti]);
                if sc <= 0.0 {
                    continue;
                }
                let replace = match best_src {
                    None => true,
                    Some((bsc, bsess, bti)) => {
                        sc > bsc
                            || (sc == bsc && (sess < bsess || (sess == bsess && ti < bti)))
                    }
                };
                if replace {
                    best_src = Some((sc, sess, ti));
                }
            }
        }
        if let Some((best_src_scene, best_src_sess, _)) = best_src {
            rare_bridges.insert(eid, (idf, best_src_scene, best_src_sess));
        }
    }
    if rare_bridges.is_empty() {
        return;
    }

    let base_scene = t_scene.to_vec();
    for (tj, sc_mut) in t_scene.iter_mut().enumerate() {
        let base_sc = base_scene[tj];
        if base_sc <= 0.0 || t_own[tj] <= 0.0 || (t_exact_mask[tj] & group_mask) == 0 {
            continue;
        }
        let tj_turn = &ix.turns[tj];
        if tj_turn.session == top_session
            || tj_turn.field != 0
            || tj_turn.is_question
            || tj_turn.is_backchannel
            || turn_is_superseded(ix, tj)
        {
            continue;
        }
        let mut best_cap = 0.0f32;
        let mut hit_count = 0usize;
        let mut idf_sum = 0.0f32;
        for (&eid, &(idf, src_scene, best_src_sess)) in &rare_bridges {
            if tj_turn.session == best_src_sess {
                continue;
            }
            if tj_turn.entities.contains(&eid) {
                hit_count += 1;
                idf_sum += idf;
                if src_scene > best_cap {
                    best_cap = src_scene;
                }
            }
        }
        if hit_count > 0 && best_cap > 0.0 && idf_sum > 0.0 {
            let ceiling = 0.80 * best_cap;
            if base_sc < ceiling {
                let mult_boost = base_sc * (0.05 * idf_sum).min(0.28);
                let additive_cap = 0.22 * best_cap;
                let boost = mult_boost.min(additive_cap);
                *sc_mut = (base_sc + boost).min(ceiling);
            }
        }
    }
}

/// Role frames used to be cut out by a phrase-trigger list. Typed slots and
/// proper-name entities already enter the index; this channel stays empty.
pub(super) fn step14_extract_role_frames(_prose_body: &str) -> (Vec<String>, Vec<String>) {
    (Vec::new(), Vec::new())
}

pub(super) fn step14_is_rec_request(_q_text: &str) -> bool {
    false
}

/// Factual question shape from the closed-class WH slots already computed
/// for the dump. No extra phrase list.
pub(super) fn step14_is_factual_wh_query(q_text: &str) -> bool {
    wh_slots(q_text) != 0
}

/// Step 14 (`Bipartite Preference Role-Object Boost`):
/// When `pref_on && !it.assistant_attribution` (and not a factual-Wh or multi-evidence query),
/// boosts substantive `field == 0` user preference/constraint/role-object declarations when
/// declared objects intersect query content terms (plus leading-session structural constraints),
/// and gently damps `field == 1` assistant replies on explicit recommendation requests so user
/// preferences win slot #1 without eroding rank-10 recall on multi-evidence benchmarks.
pub(super) fn apply_step14_bipartite_pref_boost(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    pref_on: bool,
    it: &QueryIntent,
    t_own: &[f32],
    t_scene: &mut [f32],
) {
    use std::collections::{BTreeMap, BTreeSet};

    let multi_evidence_intent =
        it.multi_hop || it.sniper_how_many || it.temporal || it.is_event_sequence;
    if !pref_on
        || it.assistant_attribution
        || multi_evidence_intent
        || step14_is_factual_wh_query(q_text)
        || ix.turns.is_empty()
        || t_scene.len() != ix.turns.len()
        || t_own.len() != ix.turns.len()
    {
        return;
    }
    let is_rec = step14_is_rec_request(q_text);

    let mut q_content_ids: BTreeSet<u32> = BTreeSet::new();
    for qt in &plan.terms {
        q_content_ids.insert(qt.id);
    }
    for &fid in &plan.focal_ids {
        q_content_ids.insert(fid);
    }

    let mut global_max = 0.0f32;
    let ns = ix.session_turns.len();
    let mut sess_max = vec![0.0f32; ns];
    let mut sess_role_df: Vec<BTreeMap<u32, usize>> = vec![BTreeMap::new(); ns];

    for (ti, &sc) in t_scene.iter().enumerate() {
        if sc > global_max {
            global_max = sc;
        }
        let t = &ix.turns[ti];
        let si = t.session;
        if si < ns {
            if sc > sess_max[si] {
                sess_max[si] = sc;
            }
            let has_decl = t.is_personal_constraint || !t.role_objs.is_empty();
            let pure_info = t.is_pure_info_question && !has_decl;
            if t.field == 0 && !pure_info && !t.is_backchannel && has_decl {
                for &rid in &t.role_objs {
                    *sess_role_df[si].entry(rid).or_insert(0) += 1;
                }
            }
        }
    }
    if global_max <= 0.0 {
        return;
    }

    let hit_floor = 0.40 * global_max;
    for (ti, sc_mut) in t_scene.iter_mut().enumerate() {
        let base_sc = *sc_mut;
        if base_sc <= 0.0 {
            continue;
        }
        let t = &ix.turns[ti];
        let si = t.session;
        if si >= ns || sess_max[si] < hit_floor || t.is_backchannel || turn_is_superseded(ix, ti) {
            continue;
        }

        if t.field == 1 {
            // Preference questions ask about the user's own preferences/tools/constraints (`field == 0`),
            // never assistant completions. Gently damping noisy assistant echoes down to `hit_floor`
            // keeps user declarations ahead on both single-session and multi-session benchmarks.
            let factor = if is_rec { 0.72 } else { 0.76 };
            *sc_mut = base_sc * factor;
            continue;
        }
        if t.field != 0 || t_own[ti] <= 0.0 {
            continue;
        }

        let q_role_hits = t
            .role_objs
            .iter()
            .filter(|id| q_content_ids.contains(id))
            .count();
        let sess_bipartite_shared = t
            .role_objs
            .iter()
            .filter(|rid| sess_role_df[si].get(rid).copied().unwrap_or(0) >= 2)
            .count();
        if q_role_hits > 0 {
            let mult = (1.48 + 0.08 * ((q_role_hits.min(3) - 1) as f32)).min(1.64);
            *sc_mut = base_sc * mult;
        } else if sess_max[si] >= 0.999 * global_max
            && base_sc >= 0.65 * global_max
            && t.is_personal_constraint
        {
            let mult = if sess_bipartite_shared > 0 { 1.24 } else { 1.18 };
            *sc_mut = base_sc * mult;
        }
    }
}

// ---- moved verbatim from former `src/steps11_20.rs` (was `include!()`d into retrieval.rs) ----

pub(super) fn step17_is_rare_bridge_term(
    ix: &Index,
    tid: u32,
    q_lower: &str,
    query_words: &std::collections::BTreeSet<String>,
    query_stems: &std::collections::BTreeSet<String>,
) -> Option<bool> {
    if is_technical_handle(&ix.vocab, tid) {
        return None;
    }
    let name = ix.vocab.names.get(tid as usize)?;
    if let Some(surface) = name.strip_prefix("\u{1}ent:") {
        if step13_is_rare_bridge_entity_candidate(ix, tid, q_lower, query_words) {
            return Some(true);
        }
        let _ = surface;
        return None;
    }
    if name.starts_with('\u{1}') || name.starts_with("stem:") {
        return None;
    }
    let low = name.to_ascii_lowercase();
    if low.len() < 6 || !low.bytes().all(|b| b.is_ascii_lowercase()) {
        return None;
    }
    if query_words.contains(&low)
        || MONTHS.contains(&low.as_str())
        || WEEKDAYS.contains(&low.as_str())
        || STARTER_STOP.contains(&low.as_str())
        || stopwords().contains(low.as_str())
    {
        return None;
    }
    let st = stem(&low);
    if query_stems.contains(&st) {
        return None;
    }
    Some(false)
}

/// Step 17 (`Isnad Multi-Hop Chain` across up to 3 distinct sessions on `t_scene`):
/// Transfers bounded decaying rank along chains of rare non-query shared entities/concepts across
/// up to 3 distinct sessions (`S1 -> b1 -> S2 -> b2 -> S3`), decaying at each hop (`hop3 < hop2 < hop1`),
/// strictly to substantive non-question `field == 0` turns with `>= 1` exact query-group hit and
/// non-trivial base relevance (`base_sc >= 0.35 * top_src1`).
pub(super) fn apply_step17_isnad_multihop_chain(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    pref_on: bool,
    it: &QueryIntent,
    t_own: &[f32],
    t_exact_mask: &[u64],
    t_scene: &mut [f32],
) {
    use std::collections::{BTreeMap, BTreeSet};

    let gl = plan.groups.len().min(64);
    if pref_on
        || it.is_event_sequence
        || it.assistant_attribution
        || gl == 0
        || t_own.is_empty()
        || t_scene.len() != t_own.len()
        || t_exact_mask.len() != t_own.len()
    {
        return;
    }
    let group_mask = if gl >= 64 { u64::MAX } else { (1u64 << gl) - 1 };
    let (q_lower, query_words) = step13_build_query_word_set(ix, q_text, plan);
    let mut query_stems: BTreeSet<String> = BTreeSet::new();
    for w in &query_words {
        query_stems.insert(stem(w));
    }

    let mut hop1: Vec<(f32, usize, usize)> = Vec::new();
    for (ti, &sc) in t_scene.iter().enumerate() {
        if sc <= 0.0 || t_own[ti] <= 0.0 || (t_exact_mask[ti] & group_mask) == 0 {
            continue;
        }
        let t = &ix.turns[ti];
        if t.field != 0 || t.is_question || t.is_backchannel || turn_is_superseded(ix, ti) {
            continue;
        }
        hop1.push((sc.max(t_own[ti]), t.session, ti));
    }
    if hop1.is_empty() {
        return;
    }
    hop1.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    let top_src1 = hop1[0].0;
    let top_session = hop1[0].1;
    hop1.retain(|&(sc, _, _)| sc >= 0.45 * top_src1);
    hop1.truncate(4);

    // Collect candidate rare bridge IDs (entities + rare lowercase content tokens) from hop1 turns.
    let mut hop1_cands: BTreeMap<u32, bool> = BTreeMap::new();
    for &(_sc, _sess, ti) in &hop1 {
        let t = &ix.turns[ti];
        for &id in t.entities.iter().chain(t.content.iter()) {
            if !hop1_cands.contains_key(&id) {
                if let Some(is_ent) =
                    step17_is_rare_bridge_term(ix, id, &q_lower, &query_words, &query_stems)
                {
                    hop1_cands.insert(id, is_ent);
                }
            }
        }
    }
    if hop1_cands.is_empty() {
        return;
    }

    let mut hop1_df: BTreeMap<u32, u32> = BTreeMap::new();
    for &id in hop1_cands.keys() {
        hop1_df.insert(id, 0);
    }
    for t in &ix.turns {
        for &eid in &t.entities {
            if let Some(cnt) = hop1_df.get_mut(&eid) {
                *cnt += 1;
            }
        }
        for &cid in &t.content {
            if let Some(cnt) = hop1_df.get_mut(&cid) {
                *cnt += 1;
            }
        }
    }

    let n_docs = ix.t_own_stats.n.max(ix.turns.len() as f32).max(1.0);
    // id -> (is_ent, idf, src1_sc, src1_sess, df)
    let mut e1_edges: BTreeMap<u32, (bool, f32, f32, usize, u32)> = BTreeMap::new();
    for (&id, &df) in &hop1_df {
        let is_ent = hop1_cands[&id];
        let max_df = if is_ent { 8 } else { 5 };
        let min_idf = if is_ent { 3.2 } else { 3.6 };
        if !(2..=max_df).contains(&df) {
            continue;
        }
        let idf = ((n_docs - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln();
        if idf < min_idf {
            continue;
        }
        let mut best_src: Option<(f32, usize, usize)> = None;
        for &(_sc_own, sess, ti) in &hop1 {
            let t = &ix.turns[ti];
            let has_id = if is_ent {
                t.entities.contains(&id)
            } else {
                t.content.contains(&id)
            };
            if has_id {
                let sc = t_scene[ti].max(t_own[ti]);
                let replace = match best_src {
                    None => true,
                    Some((bsc, bsess, bti)) => {
                        sc > bsc || (sc == bsc && (sess < bsess || (sess == bsess && ti < bti)))
                    }
                };
                if replace {
                    best_src = Some((sc, sess, ti));
                }
            }
        }
        if let Some((src_sc, src_sess, _)) = best_src {
            e1_edges.insert(id, (is_ent, idf, src_sc, src_sess, df));
        }
    }
    if e1_edges.is_empty() {
        return;
    }

    // Snapshot base_scene before applying decaying Hop-2 and Hop-3 transfers.
    let base_scene = t_scene.to_vec();
    let min_target_sc = 0.35 * top_src1;
    // Collect Hop-2 nodes (S2 != S1) for Hop-3 chaining + apply Hop-2 lift for non-ent rare bridges.
    // (hop2_carry, s2, t2, parent_s1)
    let mut hop2_nodes: Vec<(f32, usize, usize, usize)> = Vec::new();
    for (t2, turn2) in ix.turns.iter().enumerate() {
        if turn2.field != 0 || turn2.is_question || turn2.is_backchannel || turn_is_superseded(ix, t2) {
            continue;
        }
        let s2 = turn2.session;
        if s2 == top_session {
            continue;
        }
        let base_sc = base_scene[t2];
        let has_q_hit = base_sc >= min_target_sc
            && t_own[t2] > 0.0
            && (t_exact_mask[t2] & group_mask) != 0;

        let mut best_carry: Option<(f32, usize)> = None;
        let mut content_idf_sum = 0.0f32;
        let mut best_content_src = 0.0f32;
        for &id in turn2.entities.iter().chain(turn2.content.iter()) {
            if let Some(&(is_ent, idf, src1_sc, s1, df)) = e1_edges.get(&id) {
                if s2 == s1 {
                    continue;
                }
                if !is_ent && df <= 5 && has_q_hit {
                    content_idf_sum += idf;
                    if src1_sc > best_content_src {
                        best_content_src = src1_sc;
                    }
                }
                if !has_q_hit && !(is_ent && df <= 4 && idf >= 3.2) {
                    continue;
                }
                let carry = 0.82 * src1_sc;
                if best_carry.is_none_or(|(c, prev_s1)| carry > c || (carry == c && s1 < prev_s1)) {
                    best_carry = Some((carry, s1));
                }
            }
        }
        if has_q_hit && content_idf_sum > 0.0 && best_content_src > 0.0 {
            let hop2_ceiling = 0.82 * best_content_src;
            if base_sc < hop2_ceiling {
                let boost = (base_sc * (0.04 * content_idf_sum).min(0.18))
                    .min(0.16 * best_content_src);
                t_scene[t2] = (base_sc + boost).min(hop2_ceiling);
            }
        }
        if let Some((carry, s1)) = best_carry {
            hop2_nodes.push((carry, s2, t2, s1));
        }
    }
    if hop2_nodes.is_empty() {
        return;
    }
    hop2_nodes.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    hop2_nodes.truncate(10);

    // Extract Hop-2 -> Hop-3 rare non-query bridge terms distinct from e1_edges.
    let mut e2_cands: BTreeMap<u32, bool> = BTreeMap::new();
    for &(_carry, _s2, t2, _s1) in &hop2_nodes {
        let turn2 = &ix.turns[t2];
        for &id in turn2.entities.iter().chain(turn2.content.iter()) {
            if !e1_edges.contains_key(&id) && !e2_cands.contains_key(&id) {
                if let Some(is_ent) =
                    step17_is_rare_bridge_term(ix, id, &q_lower, &query_words, &query_stems)
                {
                    e2_cands.insert(id, is_ent);
                }
            }
        }
    }
    if e2_cands.is_empty() {
        return;
    }
    let mut e2_df: BTreeMap<u32, u32> = BTreeMap::new();
    for &id in e2_cands.keys() {
        e2_df.insert(id, 0);
    }
    for t in &ix.turns {
        for &eid in &t.entities {
            if let Some(cnt) = e2_df.get_mut(&eid) {
                *cnt += 1;
            }
        }
        for &cid in &t.content {
            if let Some(cnt) = e2_df.get_mut(&cid) {
                *cnt += 1;
            }
        }
    }

    // id -> (is_ent, idf, best_hop2_carry, s2, s1)
    let mut e2_edges: BTreeMap<u32, (bool, f32, f32, usize, usize)> = BTreeMap::new();
    for (&id, &df) in &e2_df {
        let is_ent = e2_cands[&id];
        let max_df = if is_ent { 8 } else { 5 };
        let min_idf = if is_ent { 3.2 } else { 3.6 };
        if !(2..=max_df).contains(&df) {
            continue;
        }
        let idf = ((n_docs - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln();
        if idf < min_idf {
            continue;
        }
        let mut best_src: Option<(f32, usize, usize, usize)> = None;
        for &(carry, s2, t2, s1) in &hop2_nodes {
            let turn2 = &ix.turns[t2];
            let has_id = if is_ent {
                turn2.entities.contains(&id)
            } else {
                turn2.content.contains(&id)
            };
            if has_id {
                let replace = match best_src {
                    None => true,
                    Some((bc, bs2, bt2, _)) => {
                        carry > bc || (carry == bc && (s2 < bs2 || (s2 == bs2 && t2 < bt2)))
                    }
                };
                if replace {
                    best_src = Some((carry, s2, t2, s1));
                }
            }
        }
        if let Some((carry, s2, _t2, s1)) = best_src {
            e2_edges.insert(id, (is_ent, idf, carry, s2, s1));
        }
    }
    if e2_edges.is_empty() {
        return;
    }

    for (t3, sc_mut) in t_scene.iter_mut().enumerate() {
        let base_sc = base_scene[t3];
        if base_sc < min_target_sc || t_own[t3] <= 0.0 || (t_exact_mask[t3] & group_mask) == 0 {
            continue;
        }
        let turn3 = &ix.turns[t3];
        let s3 = turn3.session;
        if s3 == top_session
            || turn3.field != 0
            || turn3.is_question
            || turn3.is_backchannel
            || turn_is_superseded(ix, t3)
        {
            continue;
        }
        let mut best_hop2_carry = 0.0f32;
        let mut idf_sum = 0.0f32;
        let mut hit_count = 0usize;
        for &id in turn3.entities.iter().chain(turn3.content.iter()) {
            if let Some(&(_is_ent, idf, carry, s2, s1)) = e2_edges.get(&id) {
                if s3 == s1 || s3 == s2 {
                    continue;
                }
                hit_count += 1;
                idf_sum += idf;
                if carry > best_hop2_carry {
                    best_hop2_carry = carry;
                }
            }
        }
        if hit_count > 0 && best_hop2_carry > 0.0 && idf_sum > 0.0 {
            // Hop 3 is strictly bounded below Hop 2 (`0.88 * 0.82 * src1 ~= 0.72 * src1 < 0.82 * src1`).
            let hop3_ceiling = 0.88 * best_hop2_carry;
            if *sc_mut < hop3_ceiling {
                let boost = (base_sc * (0.03 * idf_sum).min(0.14))
                    .min(0.12 * best_hop2_carry);
                *sc_mut = (*sc_mut + boost).min(hop3_ceiling);
            }
        }
    }
}
