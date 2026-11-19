//! Chronological event-ordering stack: Genette-Reichenbach Fabula Point Algebra,
//! Vilain-Kautz / Verhagen transitive closure, Eades-Lin-Smyth `GreedyFAS`, and
//! exact `5! = 120` Kendall `tau-b` minimum-violation linear extension.

use super::context::RankCtx;
use crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16;
use crate::retrieval::*;

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
/// hardcode-scan: closed-class (connectives, quantifiers, prepositions, interrogatives only).
pub(in crate::retrieval) fn is_closed_class_sequence_order_query(query: &str) -> bool {
    let mut seq_conns = 0usize;
    let mut has_span = false;
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
    }
    seq_conns >= 2 && has_span
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
    let gate_open = eff_it.is_event_sequence || is_closed_class_sequence_order_query(query);
    if !gate_open {
        return;
    }

    let q_toks: Vec<&str> = query
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();
    let q_prof = inspect_closed_class_tense_tokens(&q_toks);

    // Zero-regression `sA1` shield: allow reordering `S[0]` only when an earlier candidate in
    // `1..total_m` covers `S[0]`'s exact query-group mask AND carries >= 75% (`3/4`) of `S[0]`'s
    // lexical score; otherwise anchor `S[0]` at Rank 1 and order Positions `2..=5`.
    let s0 = final_session_indices[0];
    let s0_mask = s_exact_mask.get(s0).copied().unwrap_or(0);
    let s0_lex = ieee754_bits_to_q16(s_score.get(s0).map_or(0u32, |v| v.to_bits()));
    let can_reorder_head = final_session_indices[1..total_m].iter().any(|&sc| {
        let m_c = s_exact_mask.get(sc).copied().unwrap_or(0);
        let l_c = ieee754_bits_to_q16(s_score.get(sc).map_or(0u32, |v| v.to_bits()));
        sc < s0
            && (m_c & s0_mask) == s0_mask
            && l_c.saturating_mul(4) >= s0_lex.saturating_mul(3)
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
            let feat_words: Vec<&str> = t
                .feats
                .iter()
                .filter_map(|feat| ix.vocab.names.get(feat.id as usize).map(String::as_str))
                .collect();
            let tp = inspect_closed_class_tense_tokens(&feat_words);
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

    // Build skew-symmetric pairwise Fabula Precedence Matrix `pref[a][b]` (`> 0` => `a` before `b`).
    let mut pref = vec![vec![0i64; m]; m];
    for a in 0..m {
        let sa = cands[a];
        let tsa = ix.session_ts.get(sa).copied().unwrap_or(0);
        for b in (a + 1)..m {
            let sb = cands[b];
            let tsb = ix.session_ts.get(sb).copied().unwrap_or(0);
            let mut net_ab = 0i64;

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
