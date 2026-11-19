//! Precision rungs (1..4) and Move-3 HeidelTime relative-date anchoring.

use crate::retrieval::*;


/// Bitmask of query terms literally present in turn `ti`'s own bag.
#[inline]
pub(in crate::retrieval) fn rung_own_qmask(ix: &Index, plan: &QueryPlan, ti: usize) -> u64 {
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
pub(in crate::retrieval) fn turn_novel_slot_profile(
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
pub(in crate::retrieval) fn apply_rung1_declarative_slot_specificity(
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
pub(in crate::retrieval) fn apply_panini_karaka_slot_shape(
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
pub(in crate::retrieval) fn adj_partner_diversity_factor(
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
pub(in crate::retrieval) fn rung2_opener_lift_allowed(
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
pub(in crate::retrieval) fn apply_rung4_hub_damped_rm3(
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
pub(in crate::retrieval) struct RelTimeSpec {
    /// Point anchors: `(target_ts, sigma_days)`
    pub(in crate::retrieval) points: Vec<(i64, f32)>,
    /// Bounded span windows: `(start_ts, end_ts)`
    pub(in crate::retrieval) windows: Vec<(i64, i64)>,
    /// Per-token positions in `raw_tokens(&q.text)` that belong strictly to parsed relative-time scaffolding
    /// so BM25F can zero-weight pure temporal scaffolding words when domain words remain (`81507db6` guard).
    pub(in crate::retrieval) scaffold_positions: Vec<bool>,
}

pub(in crate::retrieval) fn parse_relative_time_specs_from_raw(raw: &[String], query_ts: i64) -> RelTimeSpec {
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

pub(in crate::retrieval) fn apply_move3_heideltime_anchor(
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
