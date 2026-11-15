// crates/vestige-core/src/rungs.rs
// Included inside `crates/vestige-core/src/retrieval.rs` via `include!("rungs.rs");`
// Active structural ranking refinements (Rungs 1, 2, 4 + Step 3A Panini karaka / clean slots / AdjPartnerDiv).

#[inline]
fn is_first_person_pronoun_cap(trimmed: &str) -> bool {
    let norm: String = trimmed.chars().map(|c| if c == '’' { '\'' } else { c }).collect();
    matches!(norm.as_str(), "I" | "I'm" | "I've" | "I'll" | "I'd")
}

/// Counts non-sentence-initial capitalized proper nouns, digits, or quoted titles at ingest time
/// so `--no-dump` (where `TurnIx.body` is empty) receives the exact same slot count as `--dump`.
fn count_slot_entities_mode(prose: &str, exclude_first_person: bool) -> u16 {
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
fn rung_own_qmask(ix: &Index, plan: &QueryPlan, ti: usize) -> u64 {
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
fn turn_novel_slot_profile(
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
fn apply_rung1_declarative_slot_specificity(
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
fn apply_panini_karaka_slot_shape(
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
fn adj_partner_diversity_factor(
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
fn rung2_opener_lift_allowed(
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
fn apply_rung4_hub_damped_rm3(
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
