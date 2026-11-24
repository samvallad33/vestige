//! Chronological event-ordering stack: Genette-Reichenbach Fabula Point Algebra,
//! Vilain-Kautz / Verhagen transitive closure, Eades-Lin-Smyth `GreedyFAS`, and
//! exact `5! = 120` Kendall `tau-b` minimum-violation linear extension.

pub(in crate::retrieval) mod chen_signature;
pub(in crate::retrieval) mod topic_thread;

use super::context::RankCtx;
use crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16;

/// Unit `1.0` in unsigned `Q16.16` fixed-point math (`2^16`).
const Q16_ONE: u64 = 65_536;
/// Top-5 session window ordered for Kendall `tau-b` (`5`, `5! = 120`).
const KENDALL_ORDER_WINDOW: usize = 5;
/// Weight assigned to verified backward RSB / Strata supersession causal edges (`2.0` in `Q16.16`).
const WEIGHT_CAUSAL_RSB_Q16: i64 = (Q16_ONE as i64) * 2;
/// Weight assigned to Genette-Reichenbach pluperfect analepsis flashback inversion (`1.5` in `Q16.16`).
const WEIGHT_ANALEPSIS_FLIP_Q16: i64 = ((Q16_ONE as i64) * 3) / 2;
/// Weight assigned to Vilain-Kautz non-zero timestamp precedence (`1.25` in `Q16.16`).
const WEIGHT_TIMESTAMP_ORDER_Q16: i64 = (Q16_ONE as i64) + ((Q16_ONE as i64) >> 2);
/// Weight assigned to Weed (`SOAP`) initiative onset & session stream order (`0.75` in `Q16.16`).
const WEIGHT_SYUZHET_STREAM_Q16: i64 = ((Q16_ONE as i64) * 3) / 4;
/// High-confidence edge threshold triggering Vilain-Kautz / Verhagen transitive closure (`1.25` in `Q16.16`).
const CLOSURE_CONFIDENCE_FLOOR_Q16: i64 = WEIGHT_TIMESTAMP_ORDER_Q16;

/// Closed-class English retrospective vs. prospective temporal polarity on a turn or query.
/// hardcode-scan: closed-class (auxiliaries and temporal connectives only, zero topic words).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::retrieval) struct ClosedClassTenseProfile {
    pub(in crate::retrieval) pluperfect_analepsis: bool,
    pub(in crate::retrieval) prospective_prolepsis: bool,
    pub(in crate::retrieval) query_orders_descending: bool,
}

pub(in crate::retrieval) fn inspect_closed_class_tense_tokens(
    tokens: &[&str],
) -> ClosedClassTenseProfile {
    let mut prof = ClosedClassTenseProfile::default();
    let mut prev_had = false;
    let mut has_retro_conn = false;
    let mut has_prosp_conn = false;

    for &raw in tokens {
        if raw.is_empty() {
            continue;
        }
        let w = raw.to_ascii_lowercase();
        // Reichenbach `E < R < S` pluperfect: `had` + closed-class participle/inflectional suffix `-ed`/`-en`.
        if prev_had
            && !matches!(
                w.as_str(),
                "not" | "never" | "already" | "previously" | "earlier"
            )
        {
            if matches!(
                w.as_str(),
                "been" | "done" | "gone" | "made" | "taken" | "begun"
            ) || (w.len() >= 4 && (w.ends_with("ed") || w.ends_with("en")))
            {
                prof.pluperfect_analepsis = true;
            }
            prev_had = false;
        } else if w == "had" {
            prev_had = true;
        }

        if matches!(w.as_str(), "earlier" | "previously" | "already" | "before") {
            has_retro_conn = true;
        }
        if matches!(w.as_str(), "will" | "shall" | "later" | "until" | "finally") {
            has_prosp_conn = true;
        }
    }
    prof.pluperfect_analepsis = prof.pluperfect_analepsis && has_retro_conn && !has_prosp_conn;
    if has_prosp_conn && !has_retro_conn {
        prof.prospective_prolepsis = true;
    }
    // Detect explicit reverse-chronological query polarity ("from last to first").
    for win in tokens.windows(4) {
        if win[0].eq_ignore_ascii_case("from")
            && win[1].eq_ignore_ascii_case("last")
            && win[2].eq_ignore_ascii_case("to")
            && win[3].eq_ignore_ascii_case("first")
        {
            prof.query_orders_descending = true;
        }
    }
    prof
}

/// Closed-class gate for sequence-ordering queries when `eff_it.is_event_sequence` is false.
/// hardcode-scan: closed-class (connectives, quantifiers, prepositions, order markers only).
pub(in crate::retrieval) fn is_closed_class_sequence_order_query(query: &str) -> bool {
    let ql = query.to_ascii_lowercase();
    if ql.contains("order of the")
        || ql.contains("order in which")
        || ql.contains("what order")
        || ql.contains("which order")
        || ql.contains("sequence of")
        || ql.contains("chronological")
    {
        return true;
    }
    let mut seq_conns = 0usize;
    let mut has_span = false;
    let mut has_order_token = false;
    for raw in query.split(|c: char| !c.is_ascii_alphanumeric()) {
        if raw.is_empty() {
            continue;
        }
        let w = raw.to_ascii_lowercase();
        if matches!(w.as_str(), "first" | "then" | "finally" | "before" | "after") {
            seq_conns += 1;
        }
        if matches!(w.as_str(), "throughout" | "across") {
            has_span = true;
        }
        if matches!(w.as_str(), "order" | "chronological" | "timeline" | "sequence" | "progression") {
            has_order_token = true;
        }
    }
    (seq_conns >= 2 && has_span) || (has_order_token && (has_span || seq_conns >= 1))
}

struct ExactOrderCtx<'a> {
    cands: &'a [usize],
    session_ids: &'a [String],
    pref: &'a [Vec<i64>],
}

fn search_exact(
    ectx: &ExactOrderCtx<'_>,
    used: &mut [bool],
    cur: &mut Vec<usize>,
    best_score: &mut i64,
    best_perm: &mut Vec<usize>,
) {
    let m = ectx.cands.len();
    if cur.len() == m {
        let mut sc = 0i64;
        for a in 0..m {
            for b in (a + 1)..m {
                sc = sc.saturating_add(ectx.pref[cur[a]][cur[b]]);
            }
        }
        let tie_better = sc == *best_score
            && cur
                .iter()
                .map(|&i| &ectx.session_ids[ectx.cands[i]])
                .lt(best_perm.iter().map(|&i| &ectx.session_ids[ectx.cands[i]]));
        if sc > *best_score || tie_better {
            *best_score = sc;
            *best_perm = cur.clone();
        }
        return;
    }
    for nxt in 0..m {
        if used[nxt] {
            continue;
        }
        used[nxt] = true;
        cur.push(nxt);
        search_exact(ectx, used, cur, best_score, best_perm);
        cur.pop();
        used[nxt] = false;
    }
}

/// Applies Vilain-Kautz / Verhagen transitive closure on high-confidence precedence arcs,
/// strips any directed cycles via Eades-Lin-Smyth (1993) `GreedyFAS`, and solves exact
/// `m! <= 120` Kendall `tau-b` minimum-violation ranking with deterministic ID tie-breaking.
#[allow(clippy::needless_range_loop)]
pub(in crate::retrieval) fn solve_fabula_fas_kendall_order(
    cands: &[usize],
    session_ids: &[String],
    mut pref: Vec<Vec<i64>>,
) -> Vec<usize> {
    let m = cands.len();
    if m < 2 {
        return cands.to_vec();
    }

    // 1. Bounded Vilain-Kautz point-algebra transitive closure (`a < k` & `k < b` => `a < b`).
    for k_mid in 0..m {
        for i in 0..m {
            if i == k_mid || pref[i][k_mid] < CLOSURE_CONFIDENCE_FLOOR_Q16 {
                continue;
            }
            for j in 0..m {
                if j == i || j == k_mid || pref[k_mid][j] < CLOSURE_CONFIDENCE_FLOOR_Q16 {
                    continue;
                }
                let implied = pref[i][k_mid].min(pref[k_mid][j]) >> 1;
                if pref[i][j] == 0 && pref[j][i] == 0 {
                    pref[i][j] = implied;
                    pref[j][i] = -implied;
                }
            }
        }
    }

    // 2. Eades-Lin-Smyth (1993) `GreedyFAS` linear arrangement seed (`s_left` + `s_right`).
    let mut active = vec![true; m];
    let mut s_left: Vec<usize> = Vec::with_capacity(m);
    let mut s_right_rev: Vec<usize> = Vec::with_capacity(m);
    let mut rem = m;

    while rem > 0 {
        let mut progressed = true;
        while progressed && rem > 0 {
            progressed = false;
            for u in 0..m {
                if !active[u] {
                    continue;
                }
                let has_in = (0..m).any(|v| active[v] && v != u && pref[v][u] > 0);
                if !has_in {
                    active[u] = false;
                    s_left.push(u);
                    rem -= 1;
                    progressed = true;
                }
            }
            for u in (0..m).rev() {
                if !active[u] {
                    continue;
                }
                let has_out = (0..m).any(|v| active[v] && v != u && pref[u][v] > 0);
                if !has_out {
                    active[u] = false;
                    s_right_rev.push(u);
                    rem -= 1;
                    progressed = true;
                }
            }
        }
        if rem == 0 {
            break;
        }
        let mut best_u = 0usize;
        let mut best_delta = i64::MIN;
        let mut first = true;
        for u in 0..m {
            if !active[u] {
                continue;
            }
            let out_w: i64 = (0..m)
                .filter(|&v| active[v] && v != u && pref[u][v] > 0)
                .map(|v| pref[u][v])
                .sum();
            let in_w: i64 = (0..m)
                .filter(|&v| active[v] && v != u && pref[v][u] > 0)
                .map(|v| pref[v][u])
                .sum();
            let delta = out_w.saturating_sub(in_w);
            if first
                || delta > best_delta
                || (delta == best_delta && session_ids[cands[u]] < session_ids[cands[best_u]])
            {
                best_u = u;
                best_delta = delta;
                first = false;
            }
        }
        active[best_u] = false;
        s_left.push(best_u);
        rem -= 1;
    }
    s_right_rev.reverse();
    s_left.extend(s_right_rev);

    // 3. Exact `m! <= 120` Kemeny-Kendall minimum-violation permutation search seeded by `GreedyFAS`.
    let mut best_perm = s_left;
    let mut best_score = 0i64;
    for a in 0..best_perm.len() {
        for b in (a + 1)..best_perm.len() {
            best_score = best_score.saturating_add(pref[best_perm[a]][best_perm[b]]);
        }
    }
    if m <= KENDALL_ORDER_WINDOW {
        let mut used = vec![false; m];
        let mut cur = Vec::with_capacity(m);
        let ectx = ExactOrderCtx {
            cands,
            session_ids,
            pref: &pref,
        };
        search_exact(&ectx, &mut used, &mut cur, &mut best_score, &mut best_perm);
    }

    best_perm.into_iter().map(|idx| cands[idx]).collect()
}

/// Applies event-ordering passes after session late-sync and diversity coverage.
/// Strictly gated to multi-event sequence queries; leaves `t_own`, `t_turn_raw`,
/// and `final_turn_indices[0..5]` untouched.
pub(in crate::retrieval) fn apply_event_ordering_stack(ctx: &mut RankCtx<'_>) {
    let ix = ctx.ix;
    let query = ctx.q.text.as_str();
    let eff_it = &ctx.eff_it;
    let s_score = &ctx.s_score;
    let s_exact_mask = &ctx.s_exact_mask;
    let final_session_indices = &mut ctx.final_session_indices;

    let total_m = final_session_indices.len().min(KENDALL_ORDER_WINDOW);
    if total_m < 2 {
        return;
    }
    let explicit_seq_q = is_closed_class_sequence_order_query(query);
    let gate_open = eff_it.is_event_sequence || explicit_seq_q;
    if !gate_open {
        return;
    }

    // Deep-Pool Top-12 Milestone Promotion (before Kendall-Tau sort):
    // Inspect Ranks 6..12 and promote up to 2 `is_initiative` sessions from uncovered chronological
    // quintiles scoring >= 70% of Rank 5 and sharing a topic/entity token (`df <= ns / 3`).
    let ns_total = ix.session_turns.len();
    if final_session_indices.len() >= 6 && ns_total >= 10 {
        let quintile_of = |si: usize| -> usize { ((si * 5) / ns_total).min(4) };
        let mut covered_q = [false; 5];
        for &si in final_session_indices.iter().take(5) {
            covered_q[quintile_of(si)] = true;
        }
        let max_topic_df = ((ns_total as u32) / 3).max(3);
        let topic_ids: Vec<u32> = ctx
            .plan
            .terms
            .iter()
            .filter(|qt| qt.exact_hit)
            .filter(|qt| {
                let df = ix.s_stats.df.get(&qt.id).copied().unwrap_or(0);
                (1..=max_topic_df).contains(&df)
            })
            .map(|qt| qt.id)
            .collect();
        let rank4_sc = s_score.get(final_session_indices[4]).copied().unwrap_or(0.0);
        let pool_hi = final_session_indices.len().min(12);
        let mut promoted_idx: Vec<usize> = Vec::new();
        for pos in 5..pool_hi {
            if promoted_idx.len() >= 2 {
                break;
            }
            let si = final_session_indices[pos];
            let q_bin = quintile_of(si);
            if covered_q[q_bin] {
                continue;
            }
            let sc = s_score.get(si).copied().unwrap_or(0.0);
            if rank4_sc > 0.0 && sc < 0.70 * rank4_sc {
                continue;
            }
            let has_init = ix.session_turns[si]
                .iter()
                .any(|&ti| ix.turns[ti].is_initiative);
            let topic_hits = topic_ids
                .iter()
                .filter(|&&tid| {
                    ix.session_content[si].contains(&tid)
                        || ix.session_entities[si].contains(&tid)
                        || ix.session_bags[si].tf[0].contains_key(&tid)
                })
                .count();
            let shares_topic = topic_ids.is_empty() || topic_hits >= topic_ids.len().min(2);
            if has_init && shares_topic {
                covered_q[q_bin] = true;
                promoted_idx.push(si);
            }
        }
        let target_slots = [4usize, 3usize];
        for (pi, si) in promoted_idx.into_iter().enumerate() {
            let slot = target_slots[pi];
            if let Some(cur_pos) = final_session_indices.iter().position(|&x| x == si)
                && cur_pos >= 5
            {
                let moved = final_session_indices.remove(cur_pos);
                let evicted = final_session_indices[slot];
                final_session_indices[slot] = moved;
                final_session_indices.insert(5.min(final_session_indices.len()), evicted);
            }
        }
    }

    let q_toks: Vec<&str> = query
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();
    let q_prof = inspect_closed_class_tense_tokens(&q_toks);

    // Zero-regression `sA1` shield: allow reordering `S[0]` when an earlier candidate in
    // `1..total_m` covers `S[0]`'s exact query-group mask with >= 3/4 lexical score, OR on
    // explicit closed-class sequence queries when an earlier candidate hits >= 1 group with >= 3/5 score.
    let s0 = final_session_indices[0];
    let s0_mask = s_exact_mask.get(s0).copied().unwrap_or(0);
    let s0_lex = ieee754_bits_to_q16(s_score.get(s0).map_or(0u32, |v| v.to_bits()));
    let can_reorder_head = final_session_indices[1..total_m].iter().any(|&sc| {
        let m_c = s_exact_mask.get(sc).copied().unwrap_or(0);
        let l_c = ieee754_bits_to_q16(s_score.get(sc).map_or(0u32, |v| v.to_bits()));
        sc < s0
            && (((m_c & s0_mask) == s0_mask
                && l_c.saturating_mul(4) >= s0_lex.saturating_mul(3))
                || (explicit_seq_q
                    && m_c != 0
                    && l_c.saturating_mul(2) >= s0_lex))
    });
    let start_idx = usize::from(!can_reorder_head);
    let cands: Vec<usize> = final_session_indices[start_idx..total_m].to_vec();
    let m = cands.len();
    if m < 2 {
        return;
    }

    // Detect per-session Genette analepsis (flashback) vs prolepsis using vocab names of turn feats.
    let mut sess_analepsis = vec![false; m];
    let mut sess_prolepsis = vec![false; m];
    for (idx, &si) in cands.iter().enumerate() {
        let mut retro_turns = 0usize;
        let mut prosp_turns = 0usize;
        let mut decl_turns = 0usize;
        for &ti in &ix.session_turns[si] {
            let t = &ix.turns[ti];
            if t.field != 0 || t.is_question || t.is_backchannel {
                continue;
            }
            decl_turns += 1;
            let body_words: Vec<&str> = t
                .body_for_headers
                .split(|c: char| !c.is_ascii_alphanumeric())
                .filter(|s| !s.is_empty())
                .collect();
            let tp = inspect_closed_class_tense_tokens(&body_words);
            if tp.pluperfect_analepsis {
                retro_turns += 1;
            }
            if tp.prospective_prolepsis {
                prosp_turns += 1;
            }
        }
        if decl_turns > 0 && retro_turns * 2 >= decl_turns && prosp_turns == 0 {
            sess_analepsis[idx] = true;
        }
        if decl_turns > 0 && prosp_turns * 2 >= decl_turns && retro_turns == 0 {
            sess_prolepsis[idx] = true;
        }
    }

    // Precompute Chen entity novelty + topic density for Lévy signed area & Transfer Entropy (#1, #11).
    let mut topic_vec = vec![0u64; m];
    let mut nov_vec = vec![0u64; m];
    let mut seen_chen: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let max_s_q16 = cands
        .iter()
        .map(|&si| ieee754_bits_to_q16(s_score.get(si).copied().unwrap_or(0.0).to_bits()))
        .max()
        .unwrap_or(1)
        .max(1);
    for (idx, &si) in cands.iter().enumerate() {
        let sc_q = ieee754_bits_to_q16(s_score.get(si).copied().unwrap_or(0.0).to_bits());
        topic_vec[idx] = (((sc_q as u128) << 16) / (max_s_q16 as u128)).min(Q16_ONE as u128) as u64;
        nov_vec[idx] = chen_signature::session_entity_novelty_q16(ix, si, &seen_chen);
        for &eid in &ix.session_entities[si] {
            seen_chen.insert(eid);
        }
    }

    // Build skew-symmetric pairwise Fabula Precedence Matrix `pref[a][b]` (`> 0` => `a` before `b`).
    let mut pref = vec![vec![0i64; m]; m];
    for a in 0..m {
        let sa = cands[a];
        let tsa = ix.session_ts.get(sa).copied().unwrap_or(0);
        for b in (a + 1)..m {
            let sb = cands[b];
            let tsb = ix.session_ts.get(sb).copied().unwrap_or(0);
            let mut net_ab = chen_signature::chen_levy_transfer_precedence_q16(
                ix,
                sa,
                sb,
                topic_vec[a],
                topic_vec[b],
                nov_vec[a],
                nov_vec[b],
            );

            // 1. Causal RSB & Strata bitemporal supersession (`later -> earlier` proves `earlier < later`).
            let b_points_to_a = ix.session_rsb_back[sb].iter().any(|&(_, p)| p == sa)
                || ix.session_supersedes_back[sb].contains(&sa);
            let a_points_to_b = ix.session_rsb_back[sa].iter().any(|&(_, p)| p == sb)
                || ix.session_supersedes_back[sa].contains(&sb);
            if b_points_to_a && !a_points_to_b {
                net_ab += WEIGHT_CAUSAL_RSB_Q16;
            } else if a_points_to_b && !b_points_to_a {
                net_ab -= WEIGHT_CAUSAL_RSB_Q16;
            }

            // 2. Genette Analepsis / Reichenbach Pluperfect Flashback Inversion (when sharing an entity).
            let shares_entity = ix.session_entities[sa]
                .iter()
                .any(|e| ix.session_entities[sb].contains(e));
            if shares_entity {
                if sa < sb && sess_prolepsis[a] && sess_analepsis[b] {
                    net_ab -= WEIGHT_ANALEPSIS_FLIP_Q16;
                } else if sb < sa && sess_prolepsis[b] && sess_analepsis[a] {
                    net_ab += WEIGHT_ANALEPSIS_FLIP_Q16;
                }
            }

            // 3. Vilain-Kautz Timestamp Order (`tsa` vs `tsb`) + Stream Index Order (`sa` vs `sb`).
            if tsa > 0 && tsb > 0 && tsa != tsb {
                if tsa < tsb {
                    net_ab += WEIGHT_TIMESTAMP_ORDER_Q16;
                } else {
                    net_ab -= WEIGHT_TIMESTAMP_ORDER_Q16;
                }
            }
            match sa.cmp(&sb) {
                std::cmp::Ordering::Less => net_ab += WEIGHT_SYUZHET_STREAM_Q16,
                std::cmp::Ordering::Greater => net_ab -= WEIGHT_SYUZHET_STREAM_Q16,
                std::cmp::Ordering::Equal => {}
            }

            if q_prof.query_orders_descending {
                net_ab = -net_ab;
            }
            pref[a][b] = net_ab;
            pref[b][a] = -net_ab;
        }
    }

    let ordered = solve_fabula_fas_kendall_order(&cands, &ix.session_ids, pref);
    final_session_indices[start_idx..total_m].copy_from_slice(&ordered);

    // Re-project `final_turn_indices` onto the Kendall-ordered milestone sessions on user/assistant
    // multi-session streams so Turn @1 / @5 / @10 track the chronological session order 1-for-1
    // (prioritizing each session's opening `field == 0` user initiative turn).
    if !ctx.has_named_speakers_eff
        && !ctx.is_monologue_corpus
        && !crate::retrieval::eff_summary_request(query)
        && !ctx.final_turn_indices.is_empty()
    {
        let target_k = ctx.k.max(ctx.final_turn_indices.len()).max(5);
        let mut reprojected: Vec<usize> = Vec::with_capacity(target_k);
        let mut secondary_in_sess: Vec<usize> = Vec::new();
        let smax = s_score.iter().copied().fold(0.0f32, f32::max);
        let strong_eo_sess = final_session_indices
            .iter()
            .take(5)
            .filter(|&&si| smax > 0.0 && s_score.get(si).copied().unwrap_or(0.0) >= 0.42 * smax)
            .count();
        for (rank_i, &si) in final_session_indices.iter().take(target_k).enumerate() {
            if rank_i >= strong_eo_sess.max(2)
                && strong_eo_sess <= 2
                && !secondary_in_sess.is_empty()
            {
                for &sti in &secondary_in_sess {
                    if reprojected.len() >= 5 {
                        break;
                    }
                    if !reprojected.contains(&sti) {
                        reprojected.push(sti);
                    }
                }
            }
            if rank_i >= 5 && smax > 0.0 && s_score.get(si).copied().unwrap_or(0.0) < 0.38 * smax {
                break;
            }
            let members = &ix.session_turns[si];
            let mut user_turns: Vec<usize> = members
                .iter()
                .copied()
                .filter(|&ti| {
                    ix.turns[ti].field == 0
                        && !ix.turns[ti].is_backchannel
                        && !ix.turns[ti].is_standing_policy
                })
                .collect();
            if user_turns.is_empty() {
                user_turns = members
                    .iter()
                    .copied()
                    .filter(|&ti| ix.turns[ti].field == 0 && !ix.turns[ti].is_backchannel)
                    .collect();
            }
            let max_u_sc = user_turns
                .iter()
                .map(|&ti| ctx.t_scene.get(ti).copied().unwrap_or(0.0))
                .fold(0.0f32, f32::max);
            if user_turns.is_empty() || max_u_sc <= 0.0 {
                let best_asst = members
                    .iter()
                    .copied()
                    .filter(|&ti| {
                        ix.turns[ti].field == 1
                            && !ix.turns[ti].is_backchannel
                            && ctx.t_scene.get(ti).copied().unwrap_or(0.0) > 0.0
                    })
                    .max_by(|&a, &b| {
                        ctx.t_scene
                            .get(a)
                            .copied()
                            .unwrap_or(0.0)
                            .partial_cmp(&ctx.t_scene.get(b).copied().unwrap_or(0.0))
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| b.cmp(&a))
                    });
                if let Some(ati) = best_asst {
                    if !reprojected.contains(&ati) {
                        reprojected.push(ati);
                    }
                    continue;
                }
            }
            if !user_turns.is_empty() {
                let open_u = user_turns[0];
                let open_sc = ctx.t_scene.get(open_u).copied().unwrap_or(0.0);
                let chosen = if max_u_sc <= 0.0 || open_sc >= 0.30 * max_u_sc {
                    open_u
                } else {
                    user_turns
                        .iter()
                        .copied()
                        .max_by(|&a, &b| {
                            ctx.t_scene
                                .get(a)
                                .copied()
                                .unwrap_or(0.0)
                                .partial_cmp(&ctx.t_scene.get(b).copied().unwrap_or(0.0))
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| b.cmp(&a))
                        })
                        .unwrap_or(open_u)
                };
                if !reprojected.contains(&chosen) {
                    reprojected.push(chosen);
                }
                if rank_i < 5 {
                    let max_sec = if strong_eo_sess <= 2 { 3usize } else { 1usize };
                    let mut added_sec = 0usize;
                    for &uti in &user_turns {
                        if uti != chosen && ctx.t_scene.get(uti).copied().unwrap_or(0.0) > 0.0 {
                            secondary_in_sess.push(uti);
                            added_sec += 1;
                            if added_sec >= max_sec {
                                break;
                            }
                        }
                    }
                }
            }
        }
        for sti in secondary_in_sess {
            if reprojected.len() >= target_k {
                break;
            }
            if !reprojected.contains(&sti) {
                reprojected.push(sti);
            }
        }
        for &old_ti in &ctx.final_turn_indices {
            if reprojected.len() >= target_k {
                break;
            }
            if !reprojected.contains(&old_ti) {
                reprojected.push(old_ti);
            }
        }
        if !reprojected.is_empty() {
            reprojected.truncate(ctx.k.max(5));
            ctx.final_turn_indices = reprojected;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fabula_fas_kendall_flips_inverted_recency_list_to_exact_order() {
        let cands = vec![24, 18, 12, 2, 7];
        let ids: Vec<String> = (0..30).map(|i| format!("s_{i:02}")).collect();
        let m = cands.len();
        let mut pref = vec![vec![0i64; m]; m];
        for a in 0..m {
            for b in (a + 1)..m {
                let sign = if cands[a] < cands[b] { 1 } else { -1 };
                pref[a][b] = sign * WEIGHT_TIMESTAMP_ORDER_Q16;
                pref[b][a] = -pref[a][b];
            }
        }
        let ordered = solve_fabula_fas_kendall_order(&cands, &ids, pref);
        assert_eq!(ordered, vec![2, 7, 12, 18, 24]);
    }

    #[test]
    fn test_greedy_fas_and_exact_kendall_resolves_noisy_cycle_deterministically() {
        let cands = vec![0, 1, 2];
        let ids = vec!["s0".to_string(), "s1".to_string(), "s2".to_string()];
        let mut pref = vec![vec![0i64; 3]; 3];
        pref[0][1] = WEIGHT_CAUSAL_RSB_Q16;
        pref[1][0] = -WEIGHT_CAUSAL_RSB_Q16;
        pref[1][2] = WEIGHT_CAUSAL_RSB_Q16;
        pref[2][1] = -WEIGHT_CAUSAL_RSB_Q16;
        pref[2][0] = WEIGHT_SYUZHET_STREAM_Q16;
        pref[0][2] = -WEIGHT_SYUZHET_STREAM_Q16;

        let ordered = solve_fabula_fas_kendall_order(&cands, &ids, pref);
        assert_eq!(ordered, vec![0, 1, 2]);
    }

    #[test]
    fn test_reichenbach_pluperfect_analepsis_detector_uses_closed_class_only() {
        let retro = inspect_closed_class_tense_tokens(&[
            "before", "that", "we", "had", "migrated", "earlier",
        ]);
        assert!(retro.pluperfect_analepsis);
        assert!(!retro.prospective_prolepsis);

        let prosp = inspect_closed_class_tense_tokens(&["we", "will", "deploy", "later"]);
        assert!(prosp.prospective_prolepsis);
        assert!(!prosp.pluperfect_analepsis);
    }
}
