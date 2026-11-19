//! Seating pass: `step15_comparative`.

use crate::retrieval::*;


// ---- moved verbatim from former `src/steps11_20.rs` (was `include!()`d into retrieval.rs) ----

/// Closed-class check: returns `true` when `q_text` asks a comparative event-ordering,
/// interval, or difference question between two events.
pub(in crate::retrieval) fn step15_is_comparative_query(q_text: &str, it: &QueryIntent) -> bool {
    if !(it.temporal || it.is_event_sequence) {
        return false;
    }
    let slots = wh_slots(q_text);
    let digits = raw_tokens(q_text)
        .into_iter()
        .filter(|w| w.chars().any(|c| c.is_ascii_digit()))
        .count();
    (slots & (WH_TIME | WH_QTY)) != 0 || digits >= 2 || it.is_event_sequence
}

/// Split a comparative event query along closed-class coordinators (`before or after`, `or`,
/// `between ... and`, or interval `before/after ... did I`) into two raw event clauses.
pub(in crate::retrieval) fn step15_split_comparative_clauses(q_text: &str) -> Option<(String, String)> {
    let ql = q_text.to_ascii_lowercase();
    let trim_clause = |s: &str| -> String {
        s.trim_matches(|c: char| !c.is_ascii_alphanumeric())
            .trim()
            .to_string()
    };

    if let Some(idx) = ql.find(" before or after ") {
        let a = &ql[..idx];
        let b = &ql[idx + " before or after ".len()..];
        let a_trim = a.rfind(',').map(|c| &a[c + 1..]).unwrap_or(a);
        return Some((trim_clause(a_trim), trim_clause(b)));
    }

    // Scan explicit `" or "` coordinators, skipping idiomatic non-event hedges.
    let mut search = 0usize;
    let mut chosen_or: Option<usize> = None;
    while let Some(rel) = ql[search..].find(" or ") {
        let pos = search + rel;
        let before = &ql[..pos];
        let after = &ql[pos + 4..];
        let idiom_skip = before.ends_with(" more")
            || before.ends_with(" yes")
            || before.ends_with(" before")
            || after.starts_with("less ")
            || after.starts_with("no ")
            || after.starts_with("not ");
        if !idiom_skip {
            chosen_or = Some(pos);
        }
        search = pos + 4;
    }
    if let Some(pos) = chosen_or {
        let left_full = &ql[..pos];
        let left = if let Some(comma) = left_full.rfind(',') {
            &left_full[comma + 1..]
        } else if let Some(bt) = left_full.rfind(" between ") {
            &left_full[bt + " between ".len()..]
        } else {
            left_full
        };
        let right_full = &ql[pos + 4..];
        let right_end = right_full
            .find(['?', '.', '!'])
            .unwrap_or(right_full.len());
        let right = right_full[..right_end]
            .rfind(',')
            .filter(|&c| {
                let tail = &right_full[c + 1..right_end];
                tail.contains("earliest") || tail.contains("latest") || tail.contains("in ")
            })
            .map(|c| &right_full[..c])
            .unwrap_or(&right_full[..right_end]);
        return Some((trim_clause(left), trim_clause(right)));
    }

    if let Some(bt) = ql.find(" between ") {
        let rest = &ql[bt + " between ".len()..];
        let end = rest.find(['?', '.', '!']).unwrap_or(rest.len());
        let body = &rest[..end];
        let mut and_positions = Vec::new();
        let mut s = 0usize;
        while let Some(rel) = body[s..].find(" and ") {
            let p = s + rel;
            and_positions.push(p);
            s = p + 5;
        }
        if !and_positions.is_empty() {
            let split_p = and_positions[and_positions.len() / 2];
            let left = &body[..split_p];
            let right = &body[split_p + 5..];
            return Some((trim_clause(left), trim_clause(right)));
        }
    }

    // Only split on `" before "` / `" after "` for interval/time comparisons (not single-event
    // `"Where did X move after Y?"`).
    let interval_frame = (wh_slots(q_text) & (WH_TIME | WH_QTY)) != 0;
    if interval_frame {
        for prep in [" before ", " after "] {
            if let Some(p) = ql.find(prep) {
                let left_raw = &ql[..p];
                let right_full = &ql[p + prep.len()..];
                let r_trim = right_full.trim_start();
                if r_trim.starts_with("today")
                    || r_trim.starts_with("that")
                    || r_trim.starts_with("now")
                {
                    continue;
                }
                const INV_AUX: &[&str] = &[
                    " did i ", " did we ", " do i ", " does ", " was i ", " were we ", " have i ",
                    " had i ",
                ];
                let mut inv_hit: Option<(usize, usize)> = None;
                for aux in INV_AUX {
                    if let Some(ai) = right_full.find(aux) {
                        if inv_hit.is_none_or(|(prev, _)| ai < prev) {
                            inv_hit = Some((ai, aux.len()));
                        }
                    }
                }
                if let Some((ai, alen)) = inv_hit {
                    let ev_a = &right_full[..ai];
                    let ev_b = &right_full[ai + alen..];
                    return Some((trim_clause(ev_a), trim_clause(ev_b)));
                }
                return Some((trim_clause(left_raw), trim_clause(right_full)));
            }
        }
    }

    None
}

/// Resolve exclusive bitmasks `(disc_a, disc_b)` over `plan.groups` (bits `0..64`) for the two
/// compared events. Returns `None` unless both exclusive sides contain >= 1 query group.
pub(in crate::retrieval) fn step15_comparative_event_masks(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    it: &QueryIntent,
) -> Option<(u64, u64)> {
    if it.is_event_sequence
        || it.assistant_attribution
        || plan.intent.preference
        || !step15_is_comparative_query(q_text, it)
        || plan.groups.len() < 2
    {
        return None;
    }
    let (clause_a, clause_b) = step15_split_comparative_clauses(q_text)?;
    let clause_to_mask = |clause: &str| -> u64 {
        let mut mask = 0u64;
        for w in raw_tokens(clause) {
            if let Some(g) = plan
                .groups
                .iter()
                .position(|grp| grp == &w)
                .or_else(|| {
                    ix.vocab
                        .get(&w)
                        .and_then(|id| plan.group_of.get(&id).copied())
                })
            {
                if g < 64 {
                    mask |= 1u64 << g;
                }
            }
        }
        mask
    };
    let mask_a = clause_to_mask(&clause_a);
    let mask_b = clause_to_mask(&clause_b);
    let shared = mask_a & mask_b;
    let disc_a = mask_a & !shared;
    let disc_b = mask_b & !shared;
    if disc_a == 0 || disc_b == 0 {
        return None;
    }
    Some((disc_a, disc_b))
}

/// Compute `(best_pop_a, max_a, best_pop_b, max_b)` using each side's highest exclusive bit-count
/// coverage so partial 1-word cross-mentions in the opposite session do not mask a missing side.
pub(in crate::retrieval) fn step15_best_side_coverage(
    ix: &Index,
    t_own: &[f32],
    t_exact_mask: &[u64],
    t_scene: &[f32],
    disc_a: u64,
    disc_b: u64,
) -> (u32, f32, u32, f32) {
    let mut best_pop_a = 0u32;
    let mut best_pop_b = 0u32;
    for (ti, &sc) in t_scene.iter().enumerate() {
        if sc <= 0.0
            || t_own[ti] <= 0.0
            || ix.turns[ti].is_backchannel
            || turn_is_superseded(ix, ti)
        {
            continue;
        }
        let m = t_exact_mask[ti];
        best_pop_a = best_pop_a.max((m & disc_a).count_ones());
        best_pop_b = best_pop_b.max((m & disc_b).count_ones());
    }
    let mut max_a = 0.0f32;
    let mut max_b = 0.0f32;
    for (ti, &sc) in t_scene.iter().enumerate() {
        if sc <= 0.0
            || t_own[ti] <= 0.0
            || ix.turns[ti].is_backchannel
            || turn_is_superseded(ix, ti)
        {
            continue;
        }
        let m = t_exact_mask[ti];
        if best_pop_a > 0 && (m & disc_a).count_ones() >= best_pop_a && sc > max_a {
            max_a = sc;
        }
        if best_pop_b > 0 && (m & disc_b).count_ones() >= best_pop_b && sc > max_b {
            max_b = sc;
        }
    }
    (best_pop_a, max_a, best_pop_b, max_b)
}

/// Step 15 (`Comparative Event-Ordering Dual-Slot Balance`, scoring stage):
/// Balances Side A vs Side B using full exclusive-clause bit coverage (`best_pop_a`, `best_pop_b`)
/// down to `0.10 * strong`, promoting at most one best turn per weaker-side session toward
/// `0.90 * strong` (preserving rank #1).
pub(in crate::retrieval) fn apply_step15_comparative_dual_slot_scores(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    it: &QueryIntent,
    t_own: &[f32],
    t_exact_mask: &[u64],
    t_scene: &mut [f32],
) {
    if t_scene.len() != ix.turns.len()
        || t_own.len() != ix.turns.len()
        || t_exact_mask.len() != ix.turns.len()
    {
        return;
    }
    let Some((disc_a, disc_b)) = step15_comparative_event_masks(ix, q_text, plan, it) else {
        return;
    };
    let (best_pop_a, max_a, best_pop_b, max_b) =
        step15_best_side_coverage(ix, t_own, t_exact_mask, t_scene, disc_a, disc_b);
    if max_a <= 0.0 || max_b <= 0.0 {
        return;
    }

    let strong = max_a.max(max_b);
    let weak = max_a.min(max_b);
    if weak >= 0.58 * strong || weak < 0.10 * strong {
        return;
    }

    let (weak_mask, weak_req_pop) = if max_a < max_b {
        (disc_a, best_pop_a)
    } else {
        (disc_b, best_pop_b)
    };
    let ns = ix.session_turns.len();
    let mut sess_best: Vec<Option<(usize, u32, bool, f32)>> = vec![None; ns];
    for (ti, &sc) in t_scene.iter().enumerate() {
        if sc < 0.70 * weak
            || t_own[ti] <= 0.0
            || ix.turns[ti].is_backchannel
            || turn_is_superseded(ix, ti)
        {
            continue;
        }
        let m = t_exact_mask[ti];
        let w_hits = (m & weak_mask).count_ones();
        if w_hits < weak_req_pop {
            continue;
        }
        let si = ix.turns[ti].session;
        if si >= ns {
            continue;
        }
        let is_user = ix.turns[ti].field == 0;
        let better = match sess_best[si] {
            None => true,
            Some((_, prev_hits, prev_user, prev_sc)) => {
                (w_hits, is_user) > (prev_hits, prev_user)
                    || ((w_hits, is_user) == (prev_hits, prev_user) && sc > prev_sc)
            }
        };
        if better {
            sess_best[si] = Some((ti, w_hits, is_user, sc));
        }
    }

    let mut chosen: Vec<(usize, u32, bool, f32)> = sess_best.into_iter().flatten().collect();
    chosen.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then(b.2.cmp(&a.2))
            .then(b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.0.cmp(&b.0))
    });
    let cap = 0.78 * strong;
    let mult = (cap / weak).min(4.0);
    for (ti, _, _, base_sc) in chosen.into_iter().take(1) {
        let lifted = (base_sc * mult).min(cap);
        if lifted > t_scene[ti] {
            t_scene[ti] = lifted;
        }
    }
}

/// Step 15 (`Comparative Event-Ordering Dual-Slot Seat`, post-Q4 top-5 safeguard):
/// Ensures `final_turn_indices[..5]` contains best-bit coverage for BOTH Side A and Side B
/// (not merely 1 shared keyword inside a rival session), replacing a duplicate-session slot
/// inside `1..5` (never touching slot `0` or floored anchors) and giving the seated turn enough
/// score headroom to survive Kemeny reordering.
pub(in crate::retrieval) fn apply_step15_comparative_dual_slot_seat(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    it: &QueryIntent,
    t_own: &[f32],
    t_exact_mask: &[u64],
    floored: &dyn Fn(usize) -> bool,
    k: usize,
    t_scene: &mut [f32],
    final_turn_indices: &mut Vec<usize>,
) {
    if final_turn_indices.len() < 5 || k < 5 {
        return;
    }
    let Some((disc_a, disc_b)) = step15_comparative_event_masks(ix, q_text, plan, it) else {
        return;
    };
    let (best_pop_a, _, best_pop_b, _) =
        step15_best_side_coverage(ix, t_own, t_exact_mask, t_scene, disc_a, disc_b);
    if best_pop_a == 0 || best_pop_b == 0 {
        return;
    }

    let req_a = best_pop_a.clamp(1, 2);
    let req_b = best_pop_b.clamp(1, 2);
    let top5_sessions: std::collections::BTreeSet<usize> = final_turn_indices[..5]
        .iter()
        .map(|&ti| ix.turns[ti].session)
        .collect();
    let has_a = final_turn_indices[..5]
        .iter()
        .any(|&ti| (t_exact_mask[ti] & disc_a).count_ones() >= req_a);
    let has_b = final_turn_indices[..5]
        .iter()
        .any(|&ti| (t_exact_mask[ti] & disc_b).count_ones() >= req_b);
    if has_a && has_b && top5_sessions.len() >= 2 {
        return;
    }

    let (missing_mask, req_missing, present_mask, req_present) = if !has_a {
        (disc_a, req_a, disc_b, req_b)
    } else if !has_b {
        (disc_b, req_b, disc_a, req_a)
    } else {
        // Both masks appear in a single session that occupies all 5 slots: find best other-session turn.
        (disc_a | disc_b, 1, disc_a | disc_b, 1)
    };
    let need_new_session = top5_sessions.len() < 2;
    let head_sc = t_scene[final_turn_indices[0]];
    if head_sc <= 0.0 {
        return;
    }
    let min_cand_sc = 0.10 * head_sc;

    let mut best_cand: Option<(usize, u32, bool, f32)> = None;
    for (ti, &sc) in t_scene.iter().enumerate() {
        if sc < min_cand_sc
            || t_own[ti] <= 0.0
            || ix.turns[ti].is_backchannel
            || turn_is_superseded(ix, ti)
            || final_turn_indices[..5].contains(&ti)
            || (need_new_session && top5_sessions.contains(&ix.turns[ti].session))
        {
            continue;
        }
        let hits = (t_exact_mask[ti] & missing_mask).count_ones();
        if hits < req_missing {
            continue;
        }
        let is_user = ix.turns[ti].field == 0;
        let better = match best_cand {
            None => true,
            Some((prev_ti, prev_hits, prev_user, prev_sc)) => {
                (hits, is_user) > (prev_hits, prev_user)
                    || ((hits, is_user) == (prev_hits, prev_user)
                        && (sc > prev_sc || (sc == prev_sc && ti < prev_ti)))
            }
        };
        if better {
            best_cand = Some((ti, hits, is_user, sc));
        }
    }
    let Some((cand_ti, _, _, _)) = best_cand else {
        return;
    };

    let victim_slot = (1..5).rev().find(|&slot| {
        let vti = final_turn_indices[slot];
        if floored(vti) {
            return false;
        }
        let vs = ix.turns[vti].session;
        let sess_cnt = final_turn_indices[..5]
            .iter()
            .filter(|&&t| ix.turns[t].session == vs)
            .count();
        let present_elsewhere = final_turn_indices[..5]
            .iter()
            .enumerate()
            .any(|(idx, &t)| idx != slot && (t_exact_mask[t] & present_mask).count_ones() >= req_present);
        sess_cnt >= 2 && present_elsewhere
    });
    let Some(slot) = victim_slot else {
        return;
    };

    let target_sc = 0.80 * head_sc;
    if t_scene[cand_ti] < target_sc {
        t_scene[cand_ti] = target_sc;
    }
    if let Some(existing) = final_turn_indices.iter().position(|&t| t == cand_ti) {
        final_turn_indices.remove(existing);
    }
    let removed = final_turn_indices.remove(slot);
    final_turn_indices.insert(slot, cand_ti);
    let demote_pos = 5.min(final_turn_indices.len());
    final_turn_indices.insert(demote_pos, removed);
    if final_turn_indices.len() > k {
        final_turn_indices.truncate(k);
    }
}
