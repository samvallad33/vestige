//! Issues #801, #806, #807: Generalized k-Hop (k=3..6) Integer Viterbi Chain Decoding
//! anchored in GhostLink Divergent Graph Synthesis (Path_min(G_bar) * (1 - Ochiai)),
//! Mackworth AC-3 information-gain arc consistency pruning (#806), and
//! negation / contradiction supersession invalidation (#807).
//!
//! Pure integer / Q16.16 fixed-point dynamic programming (zero floats, zero
//! open-class word lists). Decodes the highest-probability multi-hop chain across
//! 2 to 6 turns/sessions (`t_0 -> t_1 -> ... -> t_{h}`), pruning exponential
//! branching via strict query-bitmask information gain and unvisited-session bounds.

use std::collections::BTreeMap;

use super::ghostlink_dgs::{integer_sqrt, ochiai_divergence_q16, Q16};
use super::index::Index;
use super::is_technical_handle;
use super::scoring::QueryPlan;
use super::supersession::turn_is_superseded;

/// Maximum chain node length (`k = 6` turns / 5 transitions).
pub(super) const MAX_CHAIN_LEN: usize = 6;

/// Damping factor alpha = 0.85 in Q16.16 fixed-point (`55705 / 65536`).
pub(super) const ALPHA_Q16: u64 = 55_705;

/// Wave-1 node cap (matching 2-hop baseline) and extended wave-2..3 cap (`96`).
const WAVE1_NODE_CAP: usize = 64;
const MAX_GRAPH_NODES: usize = 96;

/// Closed-class negation markers (`< 8` literals, zero open-class words) for #807 polarity guard.
const NEG_POLARITY_TOKENS: [&str; 6] = ["not", "no", "never", "dont", "didnt", "isnt"];

/// Decoded multi-hop chain returned by generalized k-hop integer Viterbi.
pub(super) struct ViterbiChain {
    /// Ordered sequence of turn indices `(t_0, ..., t_{m-1})` (`2 <= m <= 6`).
    pub(super) path: Vec<usize>,
    /// Total Q16.16 log/potential path score.
    pub(super) score_q16: u64,
    /// Union of query-group bitmasks covered across the decoded chain.
    pub(super) union_mask: u64,
}

#[inline]
fn turn_has_neg_polarity(ix: &Index, ti: usize) -> bool {
    for tok in NEG_POLARITY_TOKENS {
        if let Some(id) = ix.vocab.get(tok) {
            if ix.turns[ti].content.contains(&id) {
                return true;
            }
        }
    }
    false
}

#[inline]
fn shares_entity_or_role_obj(ix: &Index, u_ti: usize, v_ti: usize) -> bool {
    let tu = &ix.turns[u_ti];
    let tv = &ix.turns[v_ti];
    if tu.entities.iter().any(|e| tv.entities.contains(e)) {
        return true;
    }
    tu.role_objs.iter().any(|o| tv.role_objs.contains(o))
}

/// #807 Contradiction & Negation Invalidation Guard:
/// Reject transitions between turns linked by causal supersession or opposing
/// closed-class negation polarity over the same entity/constraint subject.
#[inline]
pub(super) fn transition_contradicts(ix: &Index, u_ti: usize, v_ti: usize) -> bool {
    let tu = &ix.turns[u_ti];
    let tv = &ix.turns[v_ti];
    if tu.supersedes_turns.contains(&v_ti) || tv.supersedes_turns.contains(&u_ti) {
        return true;
    }
    if ix.session_supersedes_back[tu.session].contains(&tv.session)
        || ix.session_supersedes_back[tv.session].contains(&tu.session)
    {
        return true;
    }
    if tu.speaker == tv.speaker
        && (turn_has_neg_polarity(ix, u_ti) ^ turn_has_neg_polarity(ix, v_ti))
        && shares_entity_or_role_obj(ix, u_ti, v_ti)
    {
        return true;
    }
    false
}

/// Run generalized k-hop (`k = 2..=6`) Integer Viterbi over candidate nodes seeded from
/// `seed_turns`, preserving the exact Wave-1 pool order in `nodes[..64]` and extending
/// multi-hop bridges into `nodes[64..96]`.
pub(super) fn decode_viterbi_chain(
    ix: &Index,
    plan: &QueryPlan,
    seed_turns: &[usize],
    t_scene_q16: &[u64],
    t_mask: &[u64],
) -> Option<ViterbiChain> {
    let nt = ix.turns.len();
    if nt <= 5 || seed_turns.is_empty() {
        return None;
    }

    let t_max_q16 = t_scene_q16[seed_turns[0]];
    if t_max_q16 == 0 {
        return None;
    }

    let max_df = ((nt as u64 * 12) / 100).max(4) as u32;
    let mut nodes: Vec<usize> = Vec::with_capacity(MAX_GRAPH_NODES);
    for &s in seed_turns.iter().take(12) {
        if t_scene_q16[s] > 0 && !turn_is_superseded(ix, s) && !nodes.contains(&s) {
            nodes.push(s);
        }
    }
    if nodes.is_empty() {
        return None;
    }

    // Wave 1: preserve exact BTreeMap<u32, u64> carrier order up to WAVE1_NODE_CAP (64).
    let mut carrier_posts: BTreeMap<u32, u64> = BTreeMap::new();
    for &s in nodes.iter().take(6) {
        for &e in &ix.turns[s].entities {
            if plan.group_of.contains_key(&e) || is_technical_handle(&ix.vocab, e) {
                continue;
            }
            if let Some(&df) = ix.t_stats.df.get(&e) {
                if df >= 2 && df <= max_df {
                    carrier_posts.entry(e).or_insert(df as u64);
                }
            }
        }
    }

    let mut wave1_start = nodes.len();
    for (&e, _) in &carrier_posts {
        if let Some(post) = ix.t_postings.get(e as usize) {
            for &ti_u32 in post {
                let ti = ti_u32 as usize;
                if ti < nt
                    && !ix.turns[ti].is_backchannel
                    && !turn_is_superseded(ix, ti)
                    && !nodes.contains(&ti)
                {
                    nodes.push(ti);
                    if nodes.len() >= WAVE1_NODE_CAP {
                        break;
                    }
                }
            }
        }
        if nodes.len() >= WAVE1_NODE_CAP {
            break;
        }
    }

    // Wave 2..3: append distant multi-hop bridge turns into slots `64..96`.
    for _wave in 0..2 {
        if nodes.len() >= MAX_GRAPH_NODES || wave1_start >= nodes.len() {
            break;
        }
        let wave_end = nodes.len();
        let mut ext_carriers: BTreeMap<u32, u64> = BTreeMap::new();
        for &s in &nodes[wave1_start..wave_end] {
            for &e in &ix.turns[s].entities {
                if plan.group_of.contains_key(&e) || is_technical_handle(&ix.vocab, e) {
                    continue;
                }
                if let Some(&df) = ix.t_stats.df.get(&e) {
                    if df >= 2 && df <= max_df {
                        ext_carriers.entry(e).or_insert(df as u64);
                    }
                }
            }
        }
        wave1_start = wave_end;
        for (&e, _) in &ext_carriers {
            if let Some(post) = ix.t_postings.get(e as usize) {
                for &ti_u32 in post {
                    let ti = ti_u32 as usize;
                    if ti < nt
                        && !ix.turns[ti].is_backchannel
                        && !turn_is_superseded(ix, ti)
                        && !nodes.contains(&ti)
                    {
                        nodes.push(ti);
                        if nodes.len() >= MAX_GRAPH_NODES {
                            break;
                        }
                    }
                }
            }
            if nodes.len() >= MAX_GRAPH_NODES {
                break;
            }
        }
    }

    let n = nodes.len();
    if n < 2 {
        return None;
    }

    // Emission potential in Q16.16
    let mut emit_q16 = vec![0u64; n];
    for (idx, &ti) in nodes.iter().enumerate() {
        let base = ((t_scene_q16[ti] as u128 * Q16 as u128) / (t_max_q16.max(1) as u128))
            .min(Q16 as u128) as u64;
        let cov = (t_mask[ti].count_ones() as u64) * (Q16 / 8);
        emit_q16[idx] = base.saturating_add(cov);
    }

    // Directed transition matrix in Q16.16 with #807 contradiction invalidation
    let mut edges = vec![vec![0u64; n]; n];
    for (ui, &u_ti) in nodes.iter().enumerate() {
        let tu = &ix.turns[u_ti];
        if tu.entities.is_empty() || tu.is_backchannel {
            continue;
        }
        for (vi, &v_ti) in nodes.iter().enumerate() {
            if ui == vi {
                continue;
            }
            let tv = &ix.turns[v_ti];
            if tv.entities.is_empty() || tv.is_backchannel || transition_contradicts(ix, u_ti, v_ti)
            {
                continue;
            }
            edges[ui][vi] = viterbi_edge_q16(ix, plan, tu, tv, u_ti, v_ti, t_mask, max_df);
        }
    }

    // DP tables across hops `step in 0..MAX_CHAIN_LEN`
    let mut dp = vec![vec![0u64; n]; MAX_CHAIN_LEN];
    let mut prev = vec![vec![usize::MAX; n]; MAX_CHAIN_LEN];
    let mut mask_acc = vec![vec![0u64; n]; MAX_CHAIN_LEN];
    let mut sess_mask = vec![vec![0u64; n]; MAX_CHAIN_LEN];

    for (ui, &u_ti) in nodes.iter().enumerate() {
        if emit_q16[ui] > 0 {
            dp[0][ui] = emit_q16[ui];
            mask_acc[0][ui] = t_mask[u_ti];
            sess_mask[0][ui] = 1u64 << (ix.turns[u_ti].session & 63);
        }
    }

    let mut best_score = 0u64;
    let mut best_path: Vec<usize> = Vec::new();
    let mut best_mask = 0u64;

    // Step 1 (2-node chain `u -> v`)
    for vi in 0..n {
        let v_ti = nodes[vi];
        for ui in 0..n {
            if ui == vi || emit_q16[ui] == 0 {
                continue;
            }
            let trans = edges[ui][vi];
            if trans == 0 {
                continue;
            }
            let cand = emit_q16[ui]
                .saturating_add(trans)
                .saturating_add(emit_q16[vi] >> 1);
            if cand > dp[1][vi] {
                dp[1][vi] = cand;
                prev[1][vi] = ui;
                mask_acc[1][vi] = t_mask[nodes[ui]] | t_mask[v_ti];
                sess_mask[1][vi] = (1u64 << (ix.turns[nodes[ui]].session & 63))
                    | (1u64 << (ix.turns[v_ti].session & 63));
            }
        }
        if dp[1][vi] > best_score && prev[1][vi] != usize::MAX {
            let u_ti = nodes[prev[1][vi]];
            let v_ti = nodes[vi];
            let um = mask_acc[1][vi];
            if ix.turns[u_ti].session != ix.turns[v_ti].session
                || um.count_ones() > t_mask[u_ti].count_ones()
            {
                best_score = dp[1][vi];
                best_path = vec![u_ti, v_ti];
                best_mask = um;
            }
        }
    }

    // Generalized k-hop extension (`step = 2..MAX_CHAIN_LEN`, lengths 3..=6)
    let mut alpha_pow_q16 = Q16;
    for step in 2..MAX_CHAIN_LEN {
        alpha_pow_q16 = ((alpha_pow_q16 as u128 * ALPHA_Q16 as u128) >> 16) as u64;

        for (vi, &v_ti) in nodes.iter().enumerate() {
            let tv_sess = ix.turns[v_ti].session;
            let tv_sess_bit = 1u64 << (tv_sess & 63);
            let tv_mask = t_mask[v_ti];

            let mut best_vi_score = 0u64;
            let mut best_ui = usize::MAX;
            let mut best_vi_mask = 0u64;
            let mut best_vi_sess = 0u64;

            for ui in 0..n {
                if dp[step - 1][ui] == 0 || prev[step - 1][ui] == usize::MAX {
                    continue;
                }
                let trans = edges[ui][vi];
                if trans == 0 {
                    continue;
                }

                // #806 Strict Mackworth AC-3 Information-Gain Pruning on k >= 3 hops:
                // Every hop past step 1 MUST strictly cover at least one NEW query-group bit
                // AND bridge into an unvisited session.
                let info_gain = (tv_mask & !mask_acc[step - 1][ui]) != 0;
                let new_sess = (sess_mask[step - 1][ui] & tv_sess_bit) == 0;
                if step == 2 {
                    // Preserve 3rd-hop semantics when crossing out of u's session with positive transition
                    let u_root = nodes[prev[1][ui]];
                    if !info_gain && !(tv_sess != ix.turns[u_root].session && new_sess && tv_mask != 0) {
                        continue;
                    }
                } else if !info_gain || !new_sess {
                    continue;
                }

                // Cycle check along backpointer chain
                let mut cur = ui;
                let mut s = step - 1;
                let mut cycle = cur == vi;
                while !cycle && s > 0 {
                    cur = prev[s][cur];
                    if cur == usize::MAX || cur == vi {
                        cycle = true;
                        break;
                    }
                    s -= 1;
                }
                if cycle {
                    continue;
                }

                let step_trans = if step == 2 {
                    trans >> 1
                } else {
                    ((trans as u128 * alpha_pow_q16 as u128) >> 17) as u64
                };
                let step_emit = if step == 2 {
                    emit_q16[vi] >> 2
                } else {
                    ((emit_q16[vi] as u128 * alpha_pow_q16 as u128) >> 18) as u64
                };
                let cand = dp[step - 1][ui]
                    .saturating_add(step_trans)
                    .saturating_add(step_emit);

                if cand > best_vi_score {
                    best_vi_score = cand;
                    best_ui = ui;
                    best_vi_mask = mask_acc[step - 1][ui] | tv_mask;
                    best_vi_sess = sess_mask[step - 1][ui] | tv_sess_bit;
                }
            }

            if best_ui != usize::MAX {
                dp[step][vi] = best_vi_score;
                prev[step][vi] = best_ui;
                mask_acc[step][vi] = best_vi_mask;
                sess_mask[step][vi] = best_vi_sess;

                let strictly_more_groups = best_vi_mask.count_ones() > best_mask.count_ones();
                let same_or_better_step2 = step == 2
                    && best_vi_mask.count_ones() >= best_mask.count_ones()
                    && best_vi_score > best_score;

                if strictly_more_groups || same_or_better_step2 {
                    best_score = best_vi_score;
                    best_mask = best_vi_mask;
                    let mut rev_path = Vec::with_capacity(step + 1);
                    let mut cur = vi;
                    let mut s = step;
                    loop {
                        rev_path.push(nodes[cur]);
                        if s == 0 {
                            break;
                        }
                        cur = prev[s][cur];
                        if cur == usize::MAX {
                            break;
                        }
                        s -= 1;
                    }
                    rev_path.reverse();
                    best_path = rev_path;
                }
            }
        }
    }

    if best_path.len() < 2 || best_score == 0 {
        None
    } else {
        Some(ViterbiChain {
            path: best_path,
            score_q16: best_score,
            union_mask: best_mask,
        })
    }
}

/// Coupled Session Cluster Expansion (`BridgeSessions = Sessions(u) ∩ Sessions(w)`):
/// Finds declarative intermediate bridge turns in unseated sessions `sb` that share
/// rare carrier entities with BOTH top seed sessions `u` and `w` (pure Q16.16 integer).
pub(super) fn find_coupled_session_bridge_turns(
    ix: &Index,
    plan: &QueryPlan,
    seed_turns: &[usize],
    t_mask: &[u64],
) -> Vec<usize> {
    let ns = ix.session_turns.len();
    let nt = ix.turns.len();
    if ns < 3 || nt <= 5 || seed_turns.len() < 2 {
        return Vec::new();
    }

    let mut head_sessions: Vec<usize> = Vec::with_capacity(5);
    for &ti in seed_turns.iter().take(5) {
        let s = ix.turns[ti].session;
        if !head_sessions.contains(&s) {
            head_sessions.push(s);
        }
    }
    if head_sessions.len() < 2 {
        return Vec::new();
    }
    let u_sess = head_sessions[0];
    let w_sess = head_sessions[1];
    let max_df = ((nt as u64 * 10) / 100).max(4) as u32;

    let collect_carriers = |sess: usize| -> Vec<(u32, u64)> {
        let mut out = Vec::new();
        for &e in &ix.session_entities[sess] {
            if plan.group_of.contains_key(&e) || is_technical_handle(&ix.vocab, e) {
                continue;
            }
            if let Some(&df) = ix.t_stats.df.get(&e) {
                if df >= 2 && df <= max_df {
                    out.push((e, Q16 / (df as u64)));
                }
            }
        }
        out
    };

    let u_carriers = collect_carriers(u_sess);
    let w_carriers = collect_carriers(w_sess);
    if u_carriers.is_empty() || w_carriers.is_empty() {
        return Vec::new();
    }

    let mut covered_mask = 0u64;
    for &ti in seed_turns.iter().take(5) {
        covered_mask |= t_mask[ti];
    }

    let mut best: Option<(usize, u64)> = None;
    for si in 0..ns {
        if head_sessions.contains(&si) {
            continue;
        }
        let u_cond: u64 = u_carriers
            .iter()
            .filter(|(e, _)| ix.session_entities[si].contains(e))
            .map(|(_, c)| *c)
            .sum();
        if u_cond == 0 {
            continue;
        }
        let w_cond: u64 = w_carriers
            .iter()
            .filter(|(e, _)| ix.session_entities[si].contains(e))
            .map(|(_, c)| *c)
            .sum();
        if w_cond == 0 {
            continue;
        }
        let sess_couple_q16 = u_cond.min(w_cond).saturating_add((u_cond + w_cond) >> 2);
        for &ti in &ix.session_turns[si] {
            let t = &ix.turns[ti];
            if t.is_question
                || t.is_backchannel
                || turn_is_superseded(ix, ti)
                || seed_turns.contains(&ti)
            {
                continue;
            }
            let hits_u = u_carriers.iter().any(|(e, _)| t.entities.contains(e));
            let hits_w = w_carriers.iter().any(|(e, _)| t.entities.contains(e));
            let info_bits = (t_mask[ti] & !covered_mask).count_ones() as u64;
            let turn_mult = if hits_u && hits_w { 3 } else if hits_u || hits_w { 2 } else { 1 };
            let score_q16 = sess_couple_q16
                .saturating_mul(turn_mult)
                .saturating_add(info_bits * (Q16 / 4));
            if best.map_or(true, |(_, bs)| score_q16 > bs) {
                best = Some((ti, score_q16));
            }
        }
    }

    best.map(|(ti, _)| vec![ti]).unwrap_or_default()
}

fn viterbi_edge_q16(
    ix: &Index,
    plan: &QueryPlan,
    tu: &super::TurnIx,
    tv: &super::TurnIx,
    u_ti: usize,
    v_ti: usize,
    t_mask: &[u64],
    max_df: u32,
) -> u64 {
    let mut best_trans: u64 = 0;
    for &c in &tu.entities {
        if plan.group_of.contains_key(&c) || is_technical_handle(&ix.vocab, c) {
            continue;
        }
        if !tv.entities.contains(&c) {
            continue;
        }
        let df_c = ix.t_stats.df.get(&c).copied().unwrap_or(0);
        if df_c < 2 || df_c > max_df {
            continue;
        }
        let cond_q16 = Q16 / (df_c as u64);
        let post_c = ix.t_postings.get(c as usize).map(|v| v.as_slice()).unwrap_or(&[]);
        let mut max_div_q16 = Q16 / 2;
        for &e_div in &tv.entities {
            if e_div == c || tu.entities.contains(&e_div) {
                continue;
            }
            let post_div = ix.t_postings.get(e_div as usize).map(|v| v.as_slice()).unwrap_or(&[]);
            let d = ochiai_divergence_q16(post_c, post_div);
            if d > max_div_q16 {
                max_div_q16 = d;
            }
        }
        let mut edge = (cond_q16.saturating_mul(max_div_q16)) >> 16;
        if tu.session != tv.session {
            edge = edge.saturating_mul(3) >> 1;
        }
        let new_groups = (t_mask[v_ti] & !t_mask[u_ti]).count_ones() as u64;
        if new_groups > 0 {
            edge = edge.saturating_add(new_groups * (Q16 / 8));
        }
        let len_d = integer_sqrt((tv.content.len() as u64) + 6).max(2);
        let norm_edge = edge / len_d;
        if norm_edge > best_trans {
            best_trans = norm_edge;
        }
    }
    best_trans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viterbi_constants_are_integer_q16() {
        assert_eq!(Q16, 65_536);
        assert_eq!(ALPHA_Q16, 55_705);
        assert_eq!(MAX_CHAIN_LEN, 6);
        assert_eq!(integer_sqrt(9), 3);
    }
}
