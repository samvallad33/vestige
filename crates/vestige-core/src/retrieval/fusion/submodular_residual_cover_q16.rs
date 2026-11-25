//! Breakthrough #6: Submodular Facility-Location, `#855` Open-Circuit Residual Peeling,
//! Multi-Session Instance Cover & Maximum-Entropy Chronological Spine Stratifier (`Q16.16`).
//!
//! Pure deterministic `Q16.16` integer fusion kernel targeting the 4 remaining multi-evidence
//! recall ceilings with zero vectors, zero LLMs, and strict zero-regression pass-through on
//! solved single-hop queries:
//!
//! 1. **Mode A — `#855` Open-Circuit Residual Peeling & Bridge-Entity Co-Seating**:
//!    Targets `LongMemEval multi-session All@5` (`77.4%`), `LoCoMo Cat 1 Multi-Hop` (`72.0%`),
//!    and `HotpotQA Sent All@5` (`62.8%` vs `Sent All@10 = 79.4%`). Locks Slot 0, subtracts its
//!    covered query mask, extracts Slot 0's highest-IDF non-query entities as deterministic 2-hop
//!    bridge seeds, incorporates `#855` `GhostCurrentResult` frontier & witness spine telemetry,
//!    and greedily seats Slots `1..k` by submodular marginal gain across distinct sessions.
//! 2. **Mode B — Multi-Session Aggregation / Counting Instance Cover**:
//!    Targets `LongMemEval multi-session / temporal-reasoning All@5` (`avg 2.59 / 2.20` gold
//!    sessions, `99.25% Any@5` -> `All@5` starved by duplicate turns from Session #1). On
//!    closed-class counting/aggregation queries (`"how many"`, `"how much total"`, `"all the"`,
//!    `"every time"`), enforces strict 1-slot-per-distinct-session coverage over matching turns.
//! 3. **Mode C — Maximum-Entropy Chronological Spine Stratifier**:
//!    Targets `BEAM event_ordering` (`0.0% All@5`, `20% R@1`) and `BEAM summarization`
//!    (`0.0% All@5`, `38.9% Turn R@5`). Partitions the entity's qualified session span `[0..M-1]`
//!    into up to `k` evenly spaced chronological quantile strata (`genesis + milestones + tail`),
//!    enforcing strict 1-turn-per-session diversity across the full timeline.

use crate::retrieval::context::RankCtx;
use crate::retrieval::fusion::aspect_coverage_q16::aspect_ic_q16;
use crate::retrieval::graph::ghostcurrent_mdl_q16::GhostCurrentResult;
use crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo;
use crate::retrieval::{
    DetHashSet, Index, QueryIntent, eff_summary_request, stopwords, turn_is_superseded,
};

/// Fixed-point unit `1.0` in `Q16.16`.
pub(in crate::retrieval) const Q16_ONE: i64 = 65_536;

/// Submodular residual cover regime selected purely from closed-class query cues & `#855` state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) enum ResidualCoverRegime {
    /// Timeline, ordering, or full-span summarization -> evenly spaced chronological quantile spine.
    ChronologicalStratifiedSpine,
    /// Counting / total / exhaustive aggregation -> strict distinct-session instance cover.
    MultiSessionInstanceCover,
    /// Partial query coverage or `#855` open-circuit deficit -> residual facet + bridge-entity peeling.
    OpenCircuitBridgePeeling,
    /// Solved single-hop head -> strict no-op pass-through (0.00% regression guarantee).
    PassThrough,
}

/// Read-only inputs for [`apply_submodular_residual_cover_slices`].
pub(in crate::retrieval) struct SubmodularCoverInput<'a> {
    pub(in crate::retrieval) ix: &'a Index,
    pub(in crate::retrieval) query: &'a str,
    pub(in crate::retrieval) eff_it: &'a QueryIntent,
    pub(in crate::retrieval) ghostcurrent: &'a GhostCurrentResult,
    pub(in crate::retrieval) s_score: &'a [f32],
    pub(in crate::retrieval) t_scene: &'a [f32],
    pub(in crate::retrieval) t_exact_mask: &'a [u64],
    pub(in crate::retrieval) q_group_mask: u64,
    pub(in crate::retrieval) q_group_cnt: usize,
    pub(in crate::retrieval) k: usize,
}

/// Classify the submodular residual cover regime using strictly closed-class query structure
/// and `#855` Open-Circuit Back-EMF telemetry.
/// hardcode-scan: closed-class (English quantifiers, sequence cues, and aggregation interrogatives only).
pub(in crate::retrieval) fn detect_cover_regime(
    query: &str,
    eff_it: &QueryIntent,
    ghostcurrent: &GhostCurrentResult,
    q_group_cnt: usize,
    top0_exact_mask: u64,
    q_group_mask: u64,
) -> ResidualCoverRegime {
    let ql = query.trim().to_ascii_lowercase();

    // 1. Chronological ordering or multi-era summarization spine.
    if eff_it.is_event_sequence
        || eff_summary_request(query)
        || ql.contains("chronological")
        || ql.contains("what order")
        || ql.contains("which order")
        || ql.contains("order in which")
        || ql.contains("sequence of")
        || ql.contains("timeline of")
    {
        return ResidualCoverRegime::ChronologicalStratifiedSpine;
    }

    // 2. Multi-session counting / aggregation / exhaustive enumeration.
    if eff_it.sniper_how_many
        || ql.starts_with("how many ")
        || ql.contains(" how many ")
        || ql.contains("in total")
        || ql.contains("altogether")
        || ql.contains("every time ")
        || ql.starts_with("list all ")
        || ql.contains("total number of")
    {
        return ResidualCoverRegime::MultiSessionInstanceCover;
    }

    // 3. Open-Circuit Back-EMF (#855) with genuine uncovered query-group deficit at Slot 0.
    // Never fire on generic `partial_cov` alone (single-hop queries routinely omit 1 non-focal term).
    let head0_cov = (top0_exact_mask & q_group_mask).count_ones() as usize;
    let partial_cov = q_group_cnt >= 2 && head0_cov < q_group_cnt;
    if partial_cov && (ghostcurrent.open_circuit || !ghostcurrent.missing_query_groups.is_empty()) {
        return ResidualCoverRegime::OpenCircuitBridgePeeling;
    }

    ResidualCoverRegime::PassThrough
}

/// Extract up to `max_bridges` rarest non-query entity/content token IDs from turn `h0`
/// to serve as deterministic 2-hop multi-hop bridge keys.
fn extract_slot0_bridge_ids(
    ix: &Index,
    h0: usize,
    q_token_ids: &DetHashSet<u32>,
    max_bridges: usize,
) -> Vec<(u32, i64)> {
    let Some(tr) = ix.turns.get(h0) else {
        return Vec::new();
    };
    let ns_total = ix.session_turns.len().max(1);
    let sw = stopwords();

    let mut ranked: Vec<(bool, u32, u32, i64)> = Vec::new();
    for &cid in &tr.content {
        if q_token_ids.contains(&cid) {
            continue;
        }
        let Some(name) = ix.vocab.names.get(cid as usize) else {
            continue;
        };
        if name.len() < 3 || sw.contains(name.as_str()) {
            continue;
        }
        let df = ix.s_stats.df.get(&cid).copied().unwrap_or(1);
        // Reject ultra-common hubs (> 40% of sessions) when corpus has >= 4 sessions.
        if ns_total >= 4 && (df as usize).saturating_mul(5) > ns_total.saturating_mul(2) {
            continue;
        }
        let is_ent = tr.entities.contains(&cid);
        let ic = aspect_ic_q16(df, ns_total as u32).max(16_384) as i64;
        ranked.push((is_ent, df, cid, ic));
    }

    // Prefer named proper entities first, then lowest session DF (rarest bridge), deterministic ID tiebreak.
    ranked.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    ranked
        .into_iter()
        .take(max_bridges)
        .map(|(_, _, cid, ic)| (cid, ic))
        .collect()
}

/// Mode A: `#855` Open-Circuit Residual Peeling + Bridge-Entity Submodular Marginal-Gain Seating.
fn run_open_circuit_bridge_peeling(
    inp: &SubmodularCoverInput<'_>,
    final_turn_indices: &mut Vec<usize>,
    final_session_indices: &mut Vec<usize>,
) {
    let Some(&h0) = final_turn_indices.first() else {
        return;
    };
    let h0_sess = inp.ix.turns[h0].session;
    let mut covered_mask = inp.t_exact_mask.get(h0).copied().unwrap_or(0) & inp.q_group_mask;
    let mut covered_sessions: DetHashSet<usize> = DetHashSet::default();
    covered_sessions.insert(h0_sess);

    // Collect query token IDs so bridge extraction picks strictly non-query entities from Slot 0.
    let mut q_tok_ids: DetHashSet<u32> = DetHashSet::default();
    for tok in crate::retrieval::raw_tokens(inp.query) {
        if let Some(id) = inp.ix.vocab.get(&tok) {
            q_tok_ids.insert(id);
        }
        if let Some(id) = inp.ix.vocab.get(&crate::retrieval::stem(&tok)) {
            q_tok_ids.insert(id);
        }
    }
    let bridge_ids = extract_slot0_bridge_ids(inp.ix, h0, &q_tok_ids, 4);

    // Build bounded candidate pool: current Top-32 turns + `#855` witness spine + frontier + bridge postings.
    let mut pool: Vec<usize> = Vec::new();
    let mut seen_pool: DetHashSet<usize> = DetHashSet::default();
    seen_pool.insert(h0);
    for &ti in final_turn_indices.iter().take(32) {
        if seen_pool.insert(ti) {
            pool.push(ti);
        }
    }
    for &ti in &inp.ghostcurrent.witness_spine {
        if ti < inp.ix.turns.len() && seen_pool.insert(ti) {
            pool.push(ti);
        }
    }
    if let Some(ft) = inp.ghostcurrent.frontier_turn_idx
        && ft < inp.ix.turns.len()
        && seen_pool.insert(ft)
    {
        pool.push(ft);
    }
    for &(bid, _) in &bridge_ids {
        if let Some(post) = inp.ix.t_postings.get(bid as usize) {
            for &ti_u32 in post.iter().take(24) {
                let ti = ti_u32 as usize;
                if ti < inp.ix.turns.len()
                    && inp.ix.turns[ti].session != h0_sess
                    && seen_pool.insert(ti)
                {
                    pool.push(ti);
                }
            }
        }
    }

    let spine_set: DetHashSet<usize> = inp.ghostcurrent.witness_spine.iter().copied().collect();
    // Preserve Slot 0 and any distinct-session exact-matching turns already seated in Slots 1..3;
    // only replace redundant same-session clones or Slot 4 with at most 1 genuine `newly_covered > 0`
    // residual/bridge turn so calibrated Top-4 multi-session evidence is never evicted.
    let target_k = inp.k.min(5).min(final_turn_indices.len().max(1));
    let mut selected: Vec<usize> = vec![h0];
    let preserve_cap = target_k.saturating_sub(1).min(3);
    // Inspect strictly Slots 1..=2 so distinct-session incumbents stay locked while same-session
    // clones or empty slots yield cleanly to a residual `newly_covered > 0` bridge turn.
    for &ti in final_turn_indices.iter().skip(1).take(2) {
        if selected.len() >= preserve_cap {
            break;
        }
        let tr = &inp.ix.turns[ti];
        let t_m = inp.t_exact_mask.get(ti).copied().unwrap_or(0) & inp.q_group_mask;
        if t_m != 0
            && !covered_sessions.contains(&tr.session)
            && !tr.is_backchannel
            && !turn_is_superseded(inp.ix, ti)
            && !is_discourse_closure_or_meta_echo(inp.ix, ti, inp.query)
        {
            selected.push(ti);
            covered_mask |= t_m;
            covered_sessions.insert(tr.session);
        }
    }
    let preserved_len = selected.len();

    // Greedy submodular marginal gain for at most 1 residual slot (strictly requiring newly_covered > 0).
    let max_total = (preserved_len + 1).min(target_k);
    while selected.len() < max_total {
        let uncovered_mask = inp.q_group_mask & !covered_mask;
        if uncovered_mask == 0 {
            break;
        }
        let mut best_pick: Option<(usize, i64)> = None;

        for &ti in &pool {
            if selected.contains(&ti) {
                continue;
            }
            let tr = &inp.ix.turns[ti];
            if tr.is_backchannel
                || turn_is_superseded(inp.ix, ti)
                || is_discourse_closure_or_meta_echo(inp.ix, ti, inp.query)
            {
                continue;
            }

            let t_mask = inp.t_exact_mask.get(ti).copied().unwrap_or(0);
            let newly_covered = (t_mask & uncovered_mask).count_ones() as i64;
            if newly_covered == 0 {
                continue;
            }
            let is_new_sess = !covered_sessions.contains(&tr.session);

            // Bridge resonance with Slot 0's highest-IDF non-query entities across sessions.
            let mut bridge_q16: i64 = 0;
            if is_new_sess && tr.session != h0_sess {
                for &(bid, ic) in &bridge_ids {
                    if tr.content.contains(&bid) || tr.entities.contains(&bid) {
                        bridge_q16 = bridge_q16.saturating_add((ic * 5) >> 2);
                    }
                }
            }
            let bridge_capped = bridge_q16.min(Q16_ONE.saturating_mul(2));

            let spine_bonus = if spine_set.contains(&ti)
                || inp.ghostcurrent.frontier_turn_idx == Some(ti)
            {
                Q16_ONE
            } else {
                0
            };

            let base_rel = ((inp.t_scene.get(ti).copied().unwrap_or(0.0).max(0.0) * 16_384.0) as i64)
                .min(Q16_ONE);

            let sess_delta = if is_new_sess { Q16_ONE >> 1 } else { 0 };

            let gain_q16 = newly_covered
                .saturating_mul(Q16_ONE.saturating_mul(2))
                .saturating_add(bridge_capped)
                .saturating_add(spine_bonus)
                .saturating_add(base_rel)
                .saturating_add(sess_delta);

            let better = match best_pick {
                None => true,
                Some((prev_ti, prev_g)) => {
                    gain_q16 > prev_g || (gain_q16 == prev_g && ti < prev_ti)
                }
            };
            if better {
                best_pick = Some((ti, gain_q16));
            }
        }

        let Some((win_ti, win_gain)) = best_pick else {
            break;
        };
        if win_gain <= 0 {
            break;
        }
        selected.push(win_ti);
        covered_mask |= inp.t_exact_mask.get(win_ti).copied().unwrap_or(0) & inp.q_group_mask;
        covered_sessions.insert(inp.ix.turns[win_ti].session);
    }

    if selected.len() <= preserved_len {
        return;
    }

    // Splice `selected` to the front of `final_turn_indices`.
    let orig_tlen = final_turn_indices.len();
    let sel_set: DetHashSet<usize> = selected.iter().copied().collect();
    let mut merged = selected.clone();
    for &ti in final_turn_indices.iter() {
        if !sel_set.contains(&ti) {
            merged.push(ti);
        }
    }
    merged.truncate(orig_tlen.max(inp.k));
    *final_turn_indices = merged;

    // Only sync the newly peeled residual turn's session at `preserved_len` (Slot 1 when Slots 1..
    // were clones/empty, or Slot 3/4 when Slots 0..3 already held distinct matching sessions),
    // never evicting calibrated Top-`preserved_len` sessions.
    if let Some(&new_ti) = selected.last() {
        let new_si = inp.ix.turns[new_ti].session;
        let orig_slen = final_session_indices.len();
        let target_s_slot = preserved_len.min(inp.k.saturating_sub(1));
        if orig_slen > target_s_slot
            && !final_session_indices
                .iter()
                .take(target_s_slot)
                .any(|&s| s == new_si)
        {
            if let Some(pos) = final_session_indices.iter().position(|&s| s == new_si) {
                if pos > target_s_slot {
                    final_session_indices.remove(pos);
                    final_session_indices.insert(target_s_slot, new_si);
                }
            } else {
                final_session_indices.insert(target_s_slot, new_si);
                final_session_indices.truncate(orig_slen);
            }
        }
    }
}

/// Mode B: Multi-Session Instance Cover — guarantees up to `k` distinct matching sessions land
/// inside Top-5 turns on counting/aggregation queries while preserving globally-ranked
/// `final_session_indices` untouched.
fn run_multi_session_instance_cover(
    inp: &SubmodularCoverInput<'_>,
    final_turn_indices: &mut Vec<usize>,
    final_session_indices: &mut Vec<usize>,
) {
    let Some(&h0) = final_turn_indices.first() else {
        return;
    };
    let target_k = inp.k.min(5);
    let mut selected: Vec<usize> = vec![h0];
    let mut used_sessions: DetHashSet<usize> = DetHashSet::default();
    used_sessions.insert(inp.ix.turns[h0].session);

    // Pass 1: align turns with the globally-optimized Top-`target_k` distinct sessions first.
    for &si in final_session_indices.iter().take(target_k) {
        if selected.len() >= target_k {
            break;
        }
        if used_sessions.contains(&si) {
            continue;
        }
        if let Some(&ti) = final_turn_indices.iter().find(|&&ti| {
            let tr = &inp.ix.turns[ti];
            tr.session == si
                && !tr.is_backchannel
                && !turn_is_superseded(inp.ix, ti)
                && ((inp.t_exact_mask.get(ti).copied().unwrap_or(0) & inp.q_group_mask) != 0
                    || inp.t_scene.get(ti).copied().unwrap_or(0.0) > 0.0)
        }) {
            selected.push(ti);
            used_sessions.insert(si);
        } else if inp.s_score.get(si).copied().unwrap_or(0.0) > 0.0
            && let Some(best_ti) = best_substantive_turn_in_session(inp, si)
        {
            selected.push(best_ti);
            used_sessions.insert(si);
        }
    }

    // Pass 2: fill any remaining distinct-session turn slots from `final_turn_indices`.
    for &ti in final_turn_indices.iter().skip(1) {
        if selected.len() >= target_k {
            break;
        }
        let tr = &inp.ix.turns[ti];
        if used_sessions.contains(&tr.session) || tr.is_backchannel || turn_is_superseded(inp.ix, ti)
        {
            continue;
        }
        let has_cov = (inp.t_exact_mask.get(ti).copied().unwrap_or(0) & inp.q_group_mask) != 0
            || inp.t_scene.get(ti).copied().unwrap_or(0.0) > 0.0;
        if has_cov {
            selected.push(ti);
            used_sessions.insert(tr.session);
        }
    }

    if selected.len() <= 1 {
        return;
    }

    let orig_tlen = final_turn_indices.len();
    let sel_set: DetHashSet<usize> = selected.iter().copied().collect();
    let mut merged = selected.clone();
    for &ti in final_turn_indices.iter() {
        if !sel_set.contains(&ti) {
            merged.push(ti);
        }
    }
    merged.truncate(orig_tlen.max(inp.k));
    *final_turn_indices = merged;
}

/// Mode C: Maximum-Entropy Chronological Spine Stratifier — partitions all qualified matching
/// sessions into `k` evenly spaced chronological quantiles `[genesis, mid..., tail]` with
/// strict 1-turn-per-session diversity.
fn run_chronological_stratified_spine(
    inp: &SubmodularCoverInput<'_>,
    final_turn_indices: &mut Vec<usize>,
    final_session_indices: &mut Vec<usize>,
) {
    let target_k = inp.k.min(5);
    if target_k < 2 || inp.ix.session_turns.len() < 2 {
        return;
    }

    // Collect qualified sessions (top 24 by lexical/graph score with positive signal).
    let mut qualified_sessions: Vec<(usize, f32)> = final_session_indices
        .iter()
        .copied()
        .filter(|&si| {
            let sc = inp.s_score.get(si).copied().unwrap_or(0.0);
            sc > 0.0
                || inp
                    .ix
                    .session_turns
                    .get(si)
                    .is_some_and(|ts| ts.iter().any(|&ti| (inp.t_exact_mask.get(ti).copied().unwrap_or(0) & inp.q_group_mask) != 0))
        })
        .map(|si| (si, inp.s_score.get(si).copied().unwrap_or(0.0)))
        .collect();

    if qualified_sessions.len() < 2 {
        return;
    }

    // Retain sessions within 25% of top score (capped at 16) so irrelevant noise sessions don't dilute the spine.
    let top_sc = qualified_sessions
        .iter()
        .map(|x| x.1)
        .fold(0.0f32, f32::max);
    let floor_sc = top_sc * 0.25;
    qualified_sessions.retain(|&(_, sc)| sc >= floor_sc || top_sc <= 0.0);
    qualified_sessions.truncate(16);

    if qualified_sessions.len() < 2 {
        return;
    }

    // Sort qualified sessions chronologically by `(session_ts, si)`.
    qualified_sessions.sort_by(|a, b| {
        let ta = inp.ix.session_ts.get(a.0).copied().unwrap_or(0);
        let tb = inp.ix.session_ts.get(b.0).copied().unwrap_or(0);
        ta.cmp(&tb).then_with(|| a.0.cmp(&b.0))
    });

    let m = qualified_sessions.len();
    let pick_n = target_k.min(m);
    let mut chosen_sessions: Vec<usize> = Vec::with_capacity(pick_n);
    let mut chosen_turns: Vec<usize> = Vec::with_capacity(pick_n);

    for j in 0..pick_n {
        let pos = if pick_n == 1 {
            0
        } else {
            (j * (m - 1)) / (pick_n - 1)
        };
        let si = qualified_sessions[pos].0;
        if chosen_sessions.contains(&si) {
            continue;
        }
        if let Some(best_ti) = best_substantive_turn_in_session(inp, si) {
            chosen_sessions.push(si);
            chosen_turns.push(best_ti);
        }
    }

    if chosen_turns.len() < 2 {
        return;
    }

    let sel_t: DetHashSet<usize> = chosen_turns.iter().copied().collect();
    let mut merged_t = chosen_turns.clone();
    for &ti in final_turn_indices.iter() {
        if !sel_t.contains(&ti) {
            merged_t.push(ti);
        }
    }
    *final_turn_indices = merged_t;

    let sel_s: DetHashSet<usize> = chosen_sessions.iter().copied().collect();
    let mut merged_s = chosen_sessions;
    for &si in final_session_indices.iter() {
        if !sel_s.contains(&si) {
            merged_s.push(si);
        }
    }
    *final_session_indices = merged_s;
}

/// Pick the strongest non-backchannel, non-superseded substantive turn inside session `si`.
fn best_substantive_turn_in_session(inp: &SubmodularCoverInput<'_>, si: usize) -> Option<usize> {
    let turn_list = inp.ix.session_turns.get(si)?;
    let mut best: Option<(usize, i64)> = None;
    for &ti in turn_list {
        let tr = &inp.ix.turns[ti];
        if tr.is_backchannel
            || turn_is_superseded(inp.ix, ti)
            || is_discourse_closure_or_meta_echo(inp.ix, ti, inp.query)
        {
            continue;
        }
        let cov = (inp.t_exact_mask.get(ti).copied().unwrap_or(0) & inp.q_group_mask).count_ones()
            as i64;
        let rel = (inp.t_scene.get(ti).copied().unwrap_or(0.0).max(0.0) * 16_384.0) as i64;
        let user_bonus = if tr.field == 0 { 16_384 } else { 0 };
        let sc = cov
            .saturating_mul(Q16_ONE)
            .saturating_add(rel)
            .saturating_add(user_bonus);
        let better = match best {
            None => true,
            Some((prev_ti, prev_sc)) => sc > prev_sc || (sc == prev_sc && ti < prev_ti),
        };
        if better {
            best = Some((ti, sc));
        }
    }
    best.map(|(ti, _)| ti)
}

/// Stateless slice entrypoint for [`apply_submodular_residual_cover`].
pub(in crate::retrieval) fn apply_submodular_residual_cover_slices(
    inp: &SubmodularCoverInput<'_>,
    final_turn_indices: &mut Vec<usize>,
    final_session_indices: &mut Vec<usize>,
) -> ResidualCoverRegime {
    let Some(&h0) = final_turn_indices.first() else {
        return ResidualCoverRegime::PassThrough;
    };
    let top0_exact = inp.t_exact_mask.get(h0).copied().unwrap_or(0);
    let regime = detect_cover_regime(
        inp.query,
        inp.eff_it,
        inp.ghostcurrent,
        inp.q_group_cnt,
        top0_exact,
        inp.q_group_mask,
    );
    match regime {
        ResidualCoverRegime::ChronologicalStratifiedSpine => {
            run_chronological_stratified_spine(inp, final_turn_indices, final_session_indices);
        }
        ResidualCoverRegime::MultiSessionInstanceCover => {
            run_multi_session_instance_cover(inp, final_turn_indices, final_session_indices);
        }
        ResidualCoverRegime::OpenCircuitBridgePeeling => {
            run_open_circuit_bridge_peeling(inp, final_turn_indices, final_session_indices);
        }
        ResidualCoverRegime::PassThrough => {}
    }
    regime
}

/// Single-line pipeline entrypoint for `RankCtx` (Stage 12).
pub(in crate::retrieval) fn apply_submodular_residual_cover(
    ctx: &mut RankCtx<'_>,
) -> ResidualCoverRegime {
    // Protect dedicated downstream/upstream domain engines:
    // - STATE-Bench mutable tool-receipt pipelines (`has_tool_calls`)
    // - Preference / recommendation queries (`preference`, `is_recommendation_request_query`)
    // - Standing-rule tasks (`task_under_rules`)
    // - Dedicated event-ordering & summarization stacks (`is_event_sequence`, `eff_summary_request`)
    if ctx.ix.has_tool_calls
        || ctx.eff_it.preference
        || ctx.eff_it.task_under_rules
        || ctx.eff_it.is_event_sequence
        || eff_summary_request(&ctx.q.text)
        || crate::retrieval::seating::revealed_preference_cone_q16::is_recommendation_request_query(
            &ctx.q.text,
            ctx.pref_on,
        )
    {
        return ResidualCoverRegime::PassThrough;
    }
    let inp = SubmodularCoverInput {
        ix: ctx.ix,
        query: &ctx.q.text,
        eff_it: &ctx.eff_it,
        ghostcurrent: &ctx.ghostcurrent,
        s_score: &ctx.s_score,
        t_scene: &ctx.t_scene,
        t_exact_mask: &ctx.t_exact_mask,
        q_group_mask: ctx.q_group_mask,
        q_group_cnt: ctx.q_group_cnt,
        k: ctx.k,
    };
    apply_submodular_residual_cover_slices(
        &inp,
        &mut ctx.final_turn_indices,
        &mut ctx.final_session_indices,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Query, Session, Turn, build_index};

    fn make_session(id: &str, ts: i64, pairs: &[(&str, &str)]) -> Session {
        Session {
            id: id.to_string(),
            date: String::new(),
            timestamp: ts,
            turns: pairs
                .iter()
                .enumerate()
                .map(|(i, (role, text))| Turn {
                    id: format!("{id}_t{i}"),
                    role: (*role).to_string(),
                    content: (*text).to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                })
                .collect(),
        }
    }

    fn intent_for(ix: &Index, q_text: &str, p: &Params) -> QueryIntent {
        let q = Query {
            text: q_text.to_string(),
            date: String::new(),
        };
        crate::retrieval::plan_query(ix, &q, p).intent
    }

    #[test]
    fn open_circuit_peeling_evicts_session1_clone_and_seats_hop2_bridge_turn() {
        let sessions = vec![
            make_session(
                "s_hop1",
                100,
                &[
                    ("user", "Elena founded Helios Labs in Zurich."),
                    ("user", "Elena mentioned Helios Labs again during lunch."),
                ],
            ),
            make_session(
                "s_hop2",
                200,
                &[("user", "Helios Labs acquired the Quantum Turbine patent in Geneva.")],
            ),
            make_session(
                "s_noise",
                300,
                &[("user", "We bought apples and bread at the market.")],
            ),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = "Which patent did Elena's company acquire?";
        let eff_it = intent_for(&ix, q, &p);

        let gc = GhostCurrentResult {
            open_circuit: true,
            missing_query_groups: vec!["patent".to_string()],
            ..Default::default()
        };
        let inp = SubmodularCoverInput {
            ix: &ix,
            query: q,
            eff_it: &eff_it,
            ghostcurrent: &gc,
            s_score: &[2.0, 1.0, 0.1],
            t_scene: &[2.0, 1.9, 1.2, 0.1],
            t_exact_mask: &[0b01, 0b01, 0b10, 0b00],
            q_group_mask: 0b11,
            q_group_cnt: 2,
            k: 3,
        };
        let mut turns = vec![0, 1, 3, 2];
        let mut sess = vec![0, 2, 1];
        let regime = apply_submodular_residual_cover_slices(&inp, &mut turns, &mut sess);

        assert_eq!(regime, ResidualCoverRegime::OpenCircuitBridgePeeling);
        assert_eq!(turns[0], 0, "Slot 0 must remain locked");
        assert_eq!(
            turns[1], 2,
            "Hop-2 bridge + residual facet turn 2 must beat Session-1 clone turn 1 at Slot 1, got {turns:?}"
        );
        assert_eq!(sess[1], 1, "Hop-2 session 1 must sync into Slot 1, got {sess:?}");
    }

    #[test]
    fn multi_session_counting_forces_distinct_sessions_for_all_at_5() {
        let sessions = vec![
            make_session(
                "s1",
                100,
                &[
                    ("user", "I visited Tokyo in January for a conference."),
                    ("user", "The Tokyo trip was great and we ate ramen."),
                    ("user", "Another note about the Tokyo trip."),
                ],
            ),
            make_session(
                "s2",
                200,
                &[("user", "I visited Lisbon in February for a workshop.")],
            ),
            make_session(
                "s3",
                300,
                &[("user", "I visited Prague in March for a summit.")],
            ),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = "How many cities did I visit this year?";
        let eff_it = intent_for(&ix, q, &p);
        let gc = GhostCurrentResult::default();
        let inp = SubmodularCoverInput {
            ix: &ix,
            query: q,
            eff_it: &eff_it,
            ghostcurrent: &gc,
            s_score: &[3.0, 2.0, 1.5],
            t_scene: &[3.0, 2.8, 2.6, 2.0, 1.5],
            t_exact_mask: &[0b1, 0b1, 0b1, 0b1, 0b1],
            q_group_mask: 0b1,
            q_group_cnt: 1,
            k: 3,
        };
        let mut turns = vec![0, 1, 2, 3, 4];
        let mut sess = vec![0, 1, 2];
        let regime = apply_submodular_residual_cover_slices(&inp, &mut turns, &mut sess);

        assert_eq!(regime, ResidualCoverRegime::MultiSessionInstanceCover);
        assert_eq!(
            &turns[..3],
            &[0, 3, 4],
            "Counting query must seat 1 turn from each of the 3 distinct sessions, got {turns:?}"
        );
    }

    #[test]
    fn chronological_stratifier_spreads_five_milestones_across_timeline_without_session_clumping() {
        let sessions = vec![
            make_session("s0", 100, &[("user", "Started building the prototype rocket.")]),
            make_session("s1", 200, &[("user", "Tested the prototype rocket nozzle.")]),
            make_session("s2", 300, &[("user", "Fixed the prototype rocket telemetry.")]),
            make_session("s3", 400, &[("user", "Completed static fire of prototype rocket.")]),
            make_session("s4", 500, &[("user", "Launched the prototype rocket successfully.")]),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = "Put the prototype rocket milestones in chronological order.";
        let eff_it = intent_for(&ix, q, &p);
        let gc = GhostCurrentResult::default();
        let inp = SubmodularCoverInput {
            ix: &ix,
            query: q,
            eff_it: &eff_it,
            ghostcurrent: &gc,
            s_score: &[1.0, 1.1, 1.2, 1.3, 2.5],
            t_scene: &[1.0, 1.1, 1.2, 1.3, 2.5],
            t_exact_mask: &[0b11, 0b11, 0b11, 0b11, 0b11],
            q_group_mask: 0b11,
            q_group_cnt: 2,
            k: 5,
        };
        let mut turns = vec![4, 3, 2, 1, 0];
        let mut sess = vec![4, 3, 2, 1, 0];
        let regime = apply_submodular_residual_cover_slices(&inp, &mut turns, &mut sess);

        assert_eq!(regime, ResidualCoverRegime::ChronologicalStratifiedSpine);
        assert_eq!(turns, vec![0, 1, 2, 3, 4], "Chronological spine must order earliest genesis -> terminal");
        assert_eq!(sess, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn solved_single_hop_query_is_strict_passthrough_zero_regression() {
        let sessions = vec![
            make_session("s0", 100, &[("user", "My dog's name is Atlas.")]),
            make_session("s1", 200, &[("user", "We walked in the park.")]),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = "What is my dog's name?";
        let eff_it = intent_for(&ix, q, &p);
        let gc = GhostCurrentResult::default();
        let inp = SubmodularCoverInput {
            ix: &ix,
            query: q,
            eff_it: &eff_it,
            ghostcurrent: &gc,
            s_score: &[2.0, 0.2],
            t_scene: &[2.0, 0.2],
            t_exact_mask: &[0b11, 0b00],
            q_group_mask: 0b11,
            q_group_cnt: 2,
            k: 2,
        };
        let mut turns = vec![0, 1];
        let mut sess = vec![0, 1];
        let regime = apply_submodular_residual_cover_slices(&inp, &mut turns, &mut sess);

        assert_eq!(regime, ResidualCoverRegime::PassThrough);
        assert_eq!(turns, vec![0, 1]);
        assert_eq!(sess, vec![0, 1]);
    }
}
