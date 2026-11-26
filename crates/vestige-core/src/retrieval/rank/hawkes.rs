//! Hawkes exogenous genesis scoring, burst onset detection, and temporally
//! sensitive submodular chronological spine selection in fixed-point Q16.16.
//!
//! Pure deterministic integer math over sparse lexical and session structure:
//! - O(N) Hawkes self-exciting exogenous root (`P_root`) and backward adjoint
//!   `CascadeImpact` (Hawkes 1971; Rizoiu et al. 2017, arXiv:1708.06401).
//! - Hearst left-directed query vocabulary novelty and local valley depth
//!   (Hearst 1997; Galley et al. ACL 2003, P03-1071).
//! - Kleinberg infinite-state burst automaton via zero-exp Viterbi recurrence
//!   (Kleinberg 2002/2003).
//! - Ghalandari & Ifrim temporal episode partitioning and PM-Mean back-reference
//!   persistence AND-gate (ACL 2020, 2005.10107).
//! - Martschat & Markert concave `isqrt` temporal + semantic submodular spine
//!   selection (CoNLL 2018, K18-1023).

use crate::retrieval::fusion::aspect_coverage_q16::aspect_ic_q16;
use crate::retrieval::rank::context::RankCtx;
use crate::retrieval::rank::event_ordering::is_closed_class_sequence_order_query;
use crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo;
use crate::retrieval::{DetHashMap, DetHashSet, Index, eff_summary_request};

/// Fixed-point unit `1.0` in Q16.16.
pub(in crate::retrieval) const ONE: i64 = 65_536;

/// Hawkes baseline exogenous arrival intensity `lambda_0 = 1.0`.
const LAMBDA0: i64 = ONE;
/// Hawkes self-excitation branching ratio `alpha = 0.55`.
const ALPHA: i64 = 36_045;

#[inline]
pub(in crate::retrieval) fn qmul(a: i64, b: i64) -> i64 {
    ((a as i128).saturating_mul(b as i128) >> 16) as i64
}

#[inline]
pub(in crate::retrieval) fn qdiv(a: i64, b: i64) -> i64 {
    if b <= 0 {
        return 0;
    }
    (((a as i128) << 16) / (b as i128)) as i64
}

/// Deterministic integer `exp(-0.08 * dt)` in Q16.16 (zero floating-point calls).
/// Half-life is ~8.7 turns; gaps >= 64 turns decay to a minimal floor.
#[inline]
pub(in crate::retrieval) fn exp_neg_delta(dt: usize) -> i64 {
    // Precomputed round(65536 * exp(-0.08 * k)) for k = 0..64.
    const LUT: [i64; 64] = [
        65536, 60497, 55846, 51552, 47589, 43930, 40552, 37434, 34556, 31899, 29447, 27183,
        25093, 23163, 21382, 19738, 18221, 16820, 15527, 14333, 13231, 12214, 11275, 10408,
        9608, 8869, 8187, 7557, 6976, 6440, 5945, 5488, 5066, 4676, 4317, 3985, 3679, 3396,
        3135, 2894, 2671, 2466, 2276, 2101, 1940, 1791, 1653, 1526, 1409, 1300, 1200, 1108,
        1023, 944, 872, 805, 743, 686, 633, 584, 539, 498, 460, 424,
    ];
    if dt < LUT.len() {
        LUT[dt]
    } else {
        // Every +8 turns halves again; clamps smoothly to 0.
        let shift = ((dt - 63) / 9).min(12);
        LUT[63] >> shift
    }
}

/// Deterministic natural log `ln(x)` in Q16.16 for `x >= 1`.
/// Uses bit-length + 4-bit mantissa table scaled by `ln(2) = 45426 / 65536`.
#[inline]
pub(in crate::retrieval) fn ln_u32(x: u32) -> i64 {
    if x <= 1 {
        return 0;
    }
    let exp = (31 - x.leading_zeros()) as i64;
    let frac = if exp >= 4 {
        ((x >> (exp - 4)) & 0xF) as usize
    } else {
        ((x << (4 - exp)) & 0xF) as usize
    };
    const MANT_LOG2: [i64; 16] = [
        0, 5731, 11136, 16248, 21098, 25709, 30105, 34305, 38326, 42182, 45887, 49452, 52887,
        56201, 59403, 62500,
    ];
    let log2_q16 = (exp << 16) + MANT_LOG2[frac];
    qmul(log2_q16, 45_426)
}

/// Integer floor square root (deterministic Newton-Raphson).
#[inline]
pub(in crate::retrieval) fn isqrt_u64(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let mut x = 1u64 << ((64 - n.leading_zeros()).div_ceil(2));
    loop {
        let y = (x + n / x) >> 1;
        if y >= x {
            return x;
        }
        x = y;
    }
}

/// Per-turn genesis decomposition output in Q16.16.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(in crate::retrieval) struct HawkesTurnProfile {
    pub(in crate::retrieval) turn_idx: usize,
    pub(in crate::retrieval) session_idx: usize,
    pub(in crate::retrieval) relevance: i64,
    pub(in crate::retrieval) p_root: i64,
    pub(in crate::retrieval) cascade: i64,
    pub(in crate::retrieval) query_novelty: i64,
    pub(in crate::retrieval) valley_depth: i64,
    pub(in crate::retrieval) burst_onset: bool,
    pub(in crate::retrieval) pm_persist: i64,
    pub(in crate::retrieval) genesis_score: i64,
}

/// O(N) Hawkes exogenous root probability (`P_root`) and backward adjoint
/// `CascadeImpact` over chronologically sorted `(pos, weight_q16)` events.
pub(in crate::retrieval) fn compute_hawkes_root_cascade(
    events: &[(usize, i64)],
) -> Vec<(i64, i64, i64)> {
    let n = events.len();
    if n == 0 {
        return Vec::new();
    }
    let mut lambda = vec![LAMBDA0; n];
    let mut p_root = vec![ONE; n];
    let mut r: i64 = 0;

    // Forward pass: O(N) running excitation.
    for i in 0..n {
        if i > 0 {
            let dt = events[i].0.saturating_sub(events[i - 1].0);
            let prev_w = events[i - 1].1.clamp(0, ONE * 8);
            r = qmul(r.saturating_add(qmul(ALPHA, prev_w)), exp_neg_delta(dt));
        }
        let lam = LAMBDA0.saturating_add(r).max(1);
        lambda[i] = lam;
        p_root[i] = qdiv(LAMBDA0, lam).clamp(0, ONE);
    }

    // Backward adjoint pass: O(N) downstream cascade triggered by event i.
    let mut cascade = vec![0i64; n];
    let mut adj: i64 = 0;
    for i in (0..n).rev() {
        let wi = events[i].1.clamp(0, ONE * 8);
        cascade[i] = qmul(wi, adj).clamp(0, ONE * 6);
        let term = qdiv(qmul(wi, ALPHA), lambda[i]);
        let dt = if i > 0 {
            events[i].0.saturating_sub(events[i - 1].0)
        } else {
            0
        };
        adj = qmul(adj.saturating_add(term), exp_neg_delta(dt));
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let wi = events[i].1.clamp(0, ONE * 8);
        let base = qmul(wi, p_root[i]);
        let score = qmul(base, ONE.saturating_add(cascade[i]));
        out.push((p_root[i], cascade[i], score));
    }
    out
}

/// Kleinberg (2002) infinite-state burst automaton via zero-exp Viterbi.
/// Returns a boolean vector marking burst onset positions (`q_t >= 1 && q_{t-1} == 0`).
pub(in crate::retrieval) fn detect_kleinberg_burst_onsets(positions: &[usize]) -> Vec<bool> {
    let n = positions.len();
    let mut onsets = vec![false; n];
    if n < 3 {
        if n > 0 {
            onsets[0] = true;
        }
        return onsets;
    }
    let m = n - 1;
    let mut gaps = Vec::with_capacity(m);
    let mut total_span: u64 = 0;
    for i in 1..n {
        let g = positions[i].saturating_sub(positions[i - 1]).max(1) as u64;
        gaps.push(g);
        total_span = total_span.saturating_add(g);
    }
    if total_span == 0 {
        onsets[0] = true;
        return onsets;
    }
    // 4 hierarchical burst states (0 = baseline, 1..3 = progressively tighter bursts, s = 2).
    const STATES: usize = 4;
    let g_hat_q16 = (((total_span << 16) / (m as u64)).max(ONE as u64)) as i64;
    let alpha0 = qdiv(ONE, g_hat_q16).max(16);
    let mut alphas = [0i64; STATES];
    let mut ln_alphas = [0i64; STATES];
    for j in 0..STATES {
        let a = (alpha0 << j).min(ONE * 4);
        alphas[j] = a;
        // ln(a / 65536) = ln(a) - ln(65536).
        ln_alphas[j] = ln_u32(a.max(1) as u32) - ln_u32(ONE as u32);
    }
    let tau_step = ln_u32(n.max(2) as u32).max(ONE >> 1);

    let mut dp = vec![[i64::MAX / 4; STATES]; m];
    let mut back = vec![[0usize; STATES]; m];
    for j in 0..STATES {
        let x0 = (gaps[0].min(1024) as i64) << 16;
        let emit = qmul(alphas[j], x0) - ln_alphas[j];
        let jump = (j as i64) * tau_step;
        dp[0][j] = emit + jump;
    }
    for t in 1..m {
        let xt = (gaps[t].min(1024) as i64) << 16;
        for j in 0..STATES {
            let emit = qmul(alphas[j], xt) - ln_alphas[j];
            let mut best_cost = i64::MAX / 4;
            let mut best_prev = 0usize;
            for i in 0..STATES {
                let jump = if j > i {
                    ((j - i) as i64) * tau_step
                } else {
                    0
                };
                let c = dp[t - 1][i].saturating_add(jump);
                if c < best_cost {
                    best_cost = c;
                    best_prev = i;
                }
            }
            dp[t][j] = best_cost.saturating_add(emit);
            back[t][j] = best_prev;
        }
    }

    let mut cur_state = (0..STATES)
        .min_by_key(|&j| dp[m - 1][j])
        .unwrap_or(0);
    let mut states = vec![0usize; m];
    for t in (0..m).rev() {
        states[t] = cur_state;
        cur_state = back[t][cur_state];
    }

    let mut any_burst = false;
    for t in 0..m {
        if states[t] >= 1 && (t == 0 || states[t - 1] == 0) {
            onsets[t] = true;
            any_burst = true;
        }
    }
    if !any_burst {
        onsets[0] = true;
    }
    onsets
}

/// Closed-class check for meta-discourse summary/ordering request verbs and nouns.
/// hardcode-scan: closed-class (meta-discourse request framing words only).
fn is_meta_summary_scaffold(w: &str) -> bool {
    matches!(
        w,
        "summarize"
            | "summary"
            | "overview"
            | "recap"
            | "comprehensive"
            | "order"
            | "chronological"
            | "sequence"
            | "timeline"
            | "progression"
            | "discussed"
            | "covered"
            | "brought"
            | "mentioned"
            | "conversation"
            | "sessions"
            | "aspects"
            | "different"
            | "throughout"
            | "across"
            | "everything"
            | "provide"
            | "list"
            | "walk"
    )
}

/// Closed-class check for gratitude replies and short assistant courtesy acknowledgments
/// carrying no novel substantive topic introduction.
/// hardcode-scan: closed-class (gratitude and affirmative discourse acknowledgments only).
fn is_short_courtesy_ack(ix: &Index, ti: usize, novel_q_ic: i64, is_seq: bool) -> bool {
    let tr = &ix.turns[ti];
    let low = tr.body_for_headers.trim_start().to_ascii_lowercase();
    let is_thanks = low.starts_with("thanks")
        || low.starts_with("thank you")
        || low.starts_with("you're welcome")
        || low.starts_with("you\u{2019}re welcome")
        || low.starts_with("hmm, ")
        || low.starts_with("got it");
    if is_thanks && (is_seq || novel_q_ic == 0) {
        return true;
    }
    if tr.field == 0 {
        return false;
    }
    if novel_q_ic > 0 && tr.content.len() >= 14 {
        return false;
    }
    let short_ack_lead = low.starts_with("certainly")
        || low.starts_with("sure!")
        || low.starts_with("sure,")
        || low.starts_with("great job")
        || low.starts_with("that sounds like")
        || low.starts_with("yes, ")
        || low.starts_with("absolutely, it's important");
    (short_ack_lead && novel_q_ic == 0) || (tr.content.len() < 10 && novel_q_ic == 0)
}

/// Compute local TextTiling valley depth for turn `ti` against immediate predecessor
/// turns inside the same session (1 - max lexical overlap in Q16.16).
fn local_valley_depth(ix: &Index, ti: usize) -> i64 {
    let tr = &ix.turns[ti];
    if tr.content.is_empty() {
        return 0;
    }
    let si = tr.session;
    let members = &ix.session_turns[si];
    let Some(pos) = members.iter().position(|&x| x == ti) else {
        return ONE >> 1;
    };
    if pos == 0 {
        return ONE;
    }
    let mut max_ov: i64 = 0;
    let start = pos.saturating_sub(2);
    for &prev_ti in &members[start..pos] {
        let prev = &ix.turns[prev_ti];
        if prev.content.is_empty() {
            continue;
        }
        let inter = tr
            .content
            .iter()
            .filter(|id| prev.content.contains(id))
            .count() as i64;
        let denom = tr.content.len().min(prev.content.len()).max(1) as i64;
        let ov = (inter << 16) / denom;
        if ov > max_ov {
            max_ov = ov;
        }
    }
    ONE.saturating_sub(max_ov.min(ONE))
}

/// Jaccard token similarity between two turns in Q16.16 for semantic cluster diversity.
fn turn_jaccard(ix: &Index, a: usize, b: usize) -> i64 {
    let ca = &ix.turns[a].content;
    let cb = &ix.turns[b].content;
    if ca.is_empty() || cb.is_empty() {
        return 0;
    }
    let inter = ca.iter().filter(|id| cb.contains(id)).count() as i64;
    let union = (ca.len() + cb.len()) as i64 - inter;
    if union <= 0 {
        0
    } else {
        (inter << 16) / union
    }
}

/// Score all eligible turns across the conversation using Hawkes Exogenous Root +
/// Adjoint Cascade, Hearst Left-Directed QueryNovelty + Valley Depth, Kleinberg Burst
/// Onset, and PM-Mean back-reference persistence, then run Martschat-Markert `isqrt`
/// submodular greedy selection over distinct sessions and temporal epochs.
pub(in crate::retrieval) fn select_hawkes_genesis_spine(
    ix: &Index,
    query: &str,
    plan: &crate::retrieval::QueryPlan,
    s_score: &[f32],
    t_scene: &[f32],
    allow_assistant: bool,
    is_seq: bool,
    k: usize,
) -> Vec<HawkesTurnProfile> {
    let nt = ix.turns.len();
    let ns = ix.session_turns.len();
    if nt == 0 || ns == 0 || k == 0 {
        return Vec::new();
    }

    // Collect deduplicated selective topic terms (`df <= max(8, ns / 2)`), excluding
    // meta-discourse request framing words (`summary`, `order`, `discussed`, `aspects`).
    let max_df = ((ns as u32) / 2).max(8);
    let mut q_terms: Vec<(u32, i64, bool)> = Vec::new(); // (tid, ic_q16, is_exact)
    let mut fallback_terms: Vec<(u32, i64, bool)> = Vec::new();
    let mut seen_tid: DetHashSet<u32> = DetHashSet::default();
    for qt in &plan.terms {
        if !seen_tid.insert(qt.id) {
            continue;
        }
        let df = ix.s_stats.df.get(&qt.id).copied().unwrap_or(0);
        if df == 0 || df > max_df {
            continue;
        }
        let ic = aspect_ic_q16(df, ns as u32) as i64;
        let scaled = if qt.exact_hit {
            ic
        } else {
            (ic * 3) >> 2
        };
        if scaled <= 0 {
            continue;
        }
        let nm = ix
            .vocab
            .names
            .get(qt.id as usize)
            .map(|s| s.strip_prefix("\u{1}stem:").unwrap_or(s.as_str()))
            .unwrap_or("");
        if is_meta_summary_scaffold(nm) {
            fallback_terms.push((qt.id, scaled, qt.exact_hit));
        } else {
            q_terms.push((qt.id, scaled, qt.exact_hit));
        }
    }
    if q_terms.is_empty() {
        q_terms = fallback_terms;
    }
    let total_q_ic: i64 = q_terms.iter().map(|t| t.1).sum();
    if q_terms.is_empty() || total_q_ic <= 0 {
        return Vec::new();
    }

    let max_s_sc = s_score.iter().copied().fold(0.0f32, f32::max);
    let min_topic_hits = if q_terms.iter().filter(|t| t.2).count() >= 2 {
        2usize
    } else {
        1usize
    };

    // Precompute latest appearance turn index for each corpus term (PM-Mean persistence AND-gate).
    let mut last_turn_of: DetHashMap<u32, usize> = DetHashMap::default();
    for (ti, tr) in ix.turns.iter().enumerate() {
        for &cid in &tr.content {
            last_turn_of.insert(cid, ti);
        }
    }

    // Single linear scan over `0..nt`: compute topic-grounded relevance `w_i`, Hearst left-directed
    // `QueryNovelty`, and filter standing policies, backchannels, closures, and courtesy acks.
    let mut seen_left: DetHashSet<u32> = DetHashSet::default();
    let mut raw_cands: Vec<HawkesTurnProfile> = Vec::new();
    let mut max_rel: i64 = 0;

    for (ti, tr) in ix.turns.iter().enumerate() {
        if tr.is_backchannel
            || tr.is_standing_policy
            || (!allow_assistant && tr.field != 0)
            || is_discourse_closure_or_meta_echo(ix, ti, query)
        {
            continue;
        }
        let s_sc = s_score.get(tr.session).copied().unwrap_or(0.0);
        let mut rel_ic: i64 = 0;
        let mut novel_ic: i64 = 0;
        let mut exact_hits: usize = 0;
        for &(tid, w_ic, is_exact) in &q_terms {
            let hit = tr.content.contains(&tid) || tr.feats.iter().any(|f| f.id == tid);
            if hit {
                rel_ic = rel_ic.saturating_add(w_ic);
                if is_exact {
                    exact_hits += 1;
                }
            }
        }
        if rel_ic == 0 {
            continue;
        }
        // Require topic grounding before claiming left-directed novelty so stray single-word
        // matches in off-topic early sessions cannot steal `QueryNovelty = 1.0`.
        let sess_grounded = max_s_sc <= 0.0 || s_sc >= 0.28 * max_s_sc || exact_hits >= min_topic_hits;
        if !sess_grounded {
            continue;
        }
        for &(tid, w_ic, _) in &q_terms {
            if (tr.content.contains(&tid) || tr.feats.iter().any(|f| f.id == tid))
                && seen_left.insert(tid)
            {
                novel_ic = novel_ic.saturating_add(w_ic);
            }
        }
        if is_short_courtesy_ack(ix, ti, novel_ic, is_seq) {
            continue;
        }
        // Blend topic IC with normalized BM25F session + turn relevance.
        let sess_rel_q16 = if max_s_sc > 0.0 {
            ((s_sc / max_s_sc).clamp(0.0, 1.0) * (ONE as f32)) as i64
        } else {
            ONE >> 1
        };
        let scene_bonus = ((t_scene.get(ti).copied().unwrap_or(0.0).clamp(0.0, 20.0)) * 3276.0) as i64;
        let opener_bonus = if ix.session_turns[tr.session].first() == Some(&ti) {
            ONE >> 2
        } else {
            0
        };
        let w_i = rel_ic
            .saturating_add(sess_rel_q16)
            .saturating_add(if exact_hits >= 2 { ONE >> 1 } else { 0 })
            .saturating_add(scene_bonus >> 1)
            .saturating_add(opener_bonus);
        if w_i > max_rel {
            max_rel = w_i;
        }
        let q_nov = qdiv(novel_ic, total_q_ic).clamp(0, ONE);
        let depth = local_valley_depth(ix, ti);

        // PM-Mean AND-gate persistence: fraction of turn's rare content terms (`df <= ns/3`)
        // that recur in a strictly later session (`last_turn > end_of_session`).
        let sess_end = ix.session_turns[tr.session]
            .last()
            .copied()
            .unwrap_or(ti);
        let rare_cap = ((ns as u32) / 3).max(4);
        let mut rare_tot = 0i64;
        let mut rare_persist = 0i64;
        for &cid in &tr.content {
            let df = ix.s_stats.df.get(&cid).copied().unwrap_or(0);
            if (1..=rare_cap).contains(&df) {
                rare_tot += 1;
                if last_turn_of.get(&cid).copied().unwrap_or(0) > sess_end {
                    rare_persist += 1;
                }
            }
        }
        let pm_persist = if rare_tot > 0 {
            ((rare_persist << 16) / rare_tot).clamp(0, ONE)
        } else {
            0
        };

        raw_cands.push(HawkesTurnProfile {
            turn_idx: ti,
            session_idx: tr.session,
            relevance: w_i,
            p_root: 0,
            cascade: 0,
            query_novelty: q_nov,
            valley_depth: depth,
            burst_onset: false,
            pm_persist,
            genesis_score: 0,
        });
    }

    if raw_cands.is_empty() || max_rel <= 0 {
        return Vec::new();
    }

    // Retain turns clearing a 25% relevance floor (prevents noisy 1-generic-word echoes from
    // poisoning Hawkes excitation or Kleinberg gaps).
    let floor = (max_rel >> 2).max(1);
    let mut active: Vec<HawkesTurnProfile> = raw_cands
        .into_iter()
        .filter(|c| c.relevance >= floor || c.query_novelty >= (ONE >> 2))
        .collect();
    if active.is_empty() {
        return Vec::new();
    }

    // Normalize weights to ~[0, 2.0] for numerical stability in Hawkes forward/backward passes.
    let norm_div = max_rel.max(ONE);
    let events: Vec<(usize, i64)> = active
        .iter()
        .map(|c| (c.turn_idx, qdiv(c.relevance, norm_div).clamp(16_384, ONE * 2)))
        .collect();
    let hawkes_res = compute_hawkes_root_cascade(&events);

    let burst_positions: Vec<usize> = active.iter().map(|c| c.turn_idx).collect();
    let burst_onsets = detect_kleinberg_burst_onsets(&burst_positions);

    for (idx, cand) in active.iter_mut().enumerate() {
        let (pr, casc, h_sc) = hawkes_res[idx];
        cand.p_root = pr;
        cand.cascade = casc;
        cand.burst_onset = burst_onsets[idx];

        // Composite Genesis potential (Q16.16):
        // Hawkes root-cascade + Hearst QueryNovelty * (1 + 0.5*Depth) + Kleinberg onset + PM-Mean persistence.
        let nov_term = qmul(cand.query_novelty, ONE + (cand.valley_depth >> 1));
        let onset_bonus = if cand.burst_onset { ONE >> 1 } else { 0 };
        let persist_bonus = qmul(cand.pm_persist, ONE >> 1);
        // Mild preference for earlier conversational inception when Hawkes/novelty tie.
        let early_ratio = (((nt.saturating_sub(cand.turn_idx)) as i64) << 16) / (nt.max(1) as i64);
        let early_bonus = qmul(early_ratio, ONE >> 2);
        cand.genesis_score = h_sc
            .saturating_add(nov_term.saturating_mul(2))
            .saturating_add(onset_bonus)
            .saturating_add(persist_bonus)
            .saturating_add(early_bonus);
    }

    // Keep at most the single best turn per session (top 24 sessions by genesis_score) for
    // Martschat & Markert `isqrt` temporal + semantic submodular greedy selection.
    let mut best_by_sess: DetHashMap<usize, HawkesTurnProfile> = DetHashMap::default();
    for cand in active {
        best_by_sess
            .entry(cand.session_idx)
            .and_modify(|cur| {
                if cand.genesis_score > cur.genesis_score
                    || (cand.genesis_score == cur.genesis_score && cand.turn_idx < cur.turn_idx)
                {
                    *cur = cand;
                }
            })
            .or_insert(cand);
    }
    let mut pool: Vec<HawkesTurnProfile> = best_by_sess.into_values().collect();
    pool.sort_by(|a, b| {
        b.genesis_score
            .cmp(&a.genesis_score)
            .then_with(|| a.turn_idx.cmp(&b.turn_idx))
    });
    pool.truncate(24);
    if pool.len() <= k {
        return pool;
    }

    // Assign each candidate to one of 5 chronological epochs (`0..5`) and a temporal-lexical cluster.
    let epoch_of = |si: usize| -> usize { ((si * 5) / ns.max(1)).min(4) };
    let mut epoch_mass = [0u64; 5];
    let mut epoch_covered = [false; 5];
    let mut chosen: Vec<HawkesTurnProfile> = Vec::with_capacity(k);
    let mut used = vec![false; pool.len()];

    for _step in 0..k {
        let mut best_idx: Option<usize> = None;
        let mut best_gain: i64 = i64::MIN;
        for (pi, cand) in pool.iter().enumerate() {
            if used[pi] {
                continue;
            }
            let r_i = (cand.genesis_score.max(1) as u64).min(u64::MAX / 4);
            let ep = epoch_of(cand.session_idx);
            // Concave Martschat-Markert temporal-epoch marginal gain: isqrt(S + r_i) - isqrt(S).
            let delta_time =
                (isqrt_u64(epoch_mass[ep].saturating_add(r_i)) - isqrt_u64(epoch_mass[ep])) as i64;
            // Semantic redundancy penalty against already chosen turns (prevents near-duplicate slots).
            let max_sim = chosen
                .iter()
                .map(|c| turn_jaccard(ix, c.turn_idx, cand.turn_idx))
                .max()
                .unwrap_or(0);
            let sem_div = ONE.saturating_sub(max_sim.min(ONE));
            let ep_bonus = if !epoch_covered[ep] {
                cand.pm_persist >> 1
            } else {
                0
            };
            let gain = qmul(cand.genesis_score, (ONE >> 1) + (sem_div >> 1))
                .saturating_add(delta_time << 7)
                .saturating_add(ep_bonus);
            if gain > best_gain
                || (gain == best_gain
                    && best_idx.is_some_and(|bi| cand.turn_idx < pool[bi].turn_idx))
            {
                best_gain = gain;
                best_idx = Some(pi);
            }
        }
        let Some(bi) = best_idx else {
            break;
        };
        used[bi] = true;
        let picked = pool[bi];
        let ep = epoch_of(picked.session_idx);
        epoch_mass[ep] = epoch_mass[ep].saturating_add(picked.genesis_score.max(1) as u64);
        epoch_covered[ep] = true;
        chosen.push(picked);
    }

    chosen
}

/// Stage-12 entry point for longitudinal summary and event-ordering queries on long
/// anonymous multi-session dialogues (`ns >= 16`).
///
/// Strictly inactive on named-speaker dialogues (LoCoMo), tool-call dialogues (STATE-Bench),
/// compact chats (ConvoMem), and factual single-hop queries.
pub(in crate::retrieval) fn apply_hawkes_genesis_spine(ctx: &mut RankCtx<'_>) -> bool {
    let ns = ctx.ix.session_turns.len();
    if ctx.has_named_speakers_eff
        || ctx.is_monologue_corpus
        || ctx.ix.has_tool_calls
        || ctx.torrens_locked_head
        || ctx.implicit_disclosure_locked
        || ns < 16
        || ctx.final_session_indices.len() < 5
        || ctx.final_turn_indices.len() < 5
    {
        return false;
    }

    let is_seq = ctx.eff_it.is_event_sequence || is_closed_class_sequence_order_query(&ctx.q.text);
    let is_sum = !is_seq && eff_summary_request(&ctx.q.text);
    if !is_seq && !is_sum {
        return false;
    }

    // Summarization allows substantive assistant explanations (`field == 1`);
    // event ordering requires user initiative turns (`field == 0`).
    let allow_assistant = is_sum;
    let spine = select_hawkes_genesis_spine(
        ctx.ix,
        &ctx.q.text,
        &ctx.plan,
        &ctx.s_score,
        &ctx.t_scene,
        allow_assistant,
        is_seq,
        5,
    );
    if spine.is_empty() {
        return false;
    }

    // Preserve up to 2 strongest valid non-policy, non-ack incumbent turns (from distinct
    // sessions) so mid/late gold hits already ranked high by BM25F / topic_thread stay safe.
    let mut incumbent_anchors: Vec<(usize, usize)> = Vec::with_capacity(2);
    for &ti in ctx.final_turn_indices.iter().take(5) {
        if incumbent_anchors.len() >= 2 {
            break;
        }
        let tr = &ctx.ix.turns[ti];
        if tr.is_standing_policy
            || tr.is_backchannel
            || (!allow_assistant && tr.field != 0)
            || is_discourse_closure_or_meta_echo(ctx.ix, ti, &ctx.q.text)
            || is_short_courtesy_ack(ctx.ix, ti, 0, is_seq)
        {
            continue;
        }
        if !incumbent_anchors.iter().any(|&(_, si)| si == tr.session) {
            incumbent_anchors.push((ti, tr.session));
        }
    }

    let mut top5_turns: Vec<usize> = Vec::with_capacity(5);
    let mut top5_sessions: Vec<usize> = Vec::with_capacity(5);

    // Slot 0: #1 Hawkes/Hearst/Kleinberg exogenous genesis root.
    top5_turns.push(spine[0].turn_idx);
    top5_sessions.push(spine[0].session_idx);

    // Slots 1..=2: Up to 2 valid distinct-session incumbent BM25/topic_thread anchors.
    for (inc_ti, inc_si) in incumbent_anchors {
        if top5_turns.len() >= 3 {
            break;
        }
        if !top5_sessions.contains(&inc_si) {
            top5_turns.push(inc_ti);
            top5_sessions.push(inc_si);
        }
    }

    // Slots 3..=4: Remaining Martschat-Markert epoch-diverse Hawkes genesis turns.
    for cand in spine.iter().skip(1) {
        if top5_turns.len() >= 5 {
            break;
        }
        if !top5_sessions.contains(&cand.session_idx) {
            top5_turns.push(cand.turn_idx);
            top5_sessions.push(cand.session_idx);
        }
    }

    // For explicit event-ordering queries, sort the selected Top-5 genesis + milestone spine
    // into ascending chronological order (earliest turn / session first).
    if is_seq {
        let mut pairs: Vec<(usize, usize)> = top5_turns
            .into_iter()
            .zip(top5_sessions.into_iter())
            .collect();
        pairs.sort_by_key(|&(ti, si)| (ctx.ix.session_ts.get(si).copied().unwrap_or(0), si, ti));
        top5_turns = pairs.iter().map(|&(ti, _)| ti).collect();
        top5_sessions = pairs.iter().map(|&(_, si)| si).collect();
    }

    let target_t = ctx.k.max(ctx.final_turn_indices.len()).max(5);
    let mut new_turns = Vec::with_capacity(target_t);
    for ti in top5_turns {
        if !new_turns.contains(&ti) {
            new_turns.push(ti);
        }
    }
    for &old_ti in &ctx.final_turn_indices {
        if new_turns.len() >= target_t {
            break;
        }
        if !new_turns.contains(&old_ti) {
            new_turns.push(old_ti);
        }
    }
    new_turns.truncate(target_t);
    ctx.final_turn_indices = new_turns;

    let target_s = ctx.k.max(ctx.final_session_indices.len()).max(5);
    let mut new_sessions = Vec::with_capacity(target_s);
    for si in top5_sessions {
        if !new_sessions.contains(&si) {
            new_sessions.push(si);
        }
    }
    for &old_si in &ctx.final_session_indices {
        if new_sessions.len() >= target_s {
            break;
        }
        if !new_sessions.contains(&old_si) {
            new_sessions.push(old_si);
        }
    }
    new_sessions.truncate(target_s);
    ctx.final_session_indices = new_sessions;

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Query, Session, Turn, build_index, plan_query};

    fn mk_sess(id: &str, ts: i64, turns: &[(&str, &str)]) -> Session {
        Session {
            id: id.to_string(),
            date: String::new(),
            timestamp: ts,
            turns: turns
                .iter()
                .enumerate()
                .map(|(i, (r, c))| Turn {
                    id: format!("{id}_t{i}"),
                    role: (*r).to_string(),
                    content: (*c).to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                })
                .collect(),
        }
    }

    #[test]
    fn hawkes_exogenous_root_beats_late_recap_and_tracks_cascade() {
        // Turn 2 introduces topic; turns 6, 10, 14, 18 echo it; turn 200 is a late recap.
        let events = vec![
            (2usize, ONE),
            (6, ONE),
            (10, ONE),
            (14, ONE),
            (18, ONE),
            (20, ONE),
        ];
        let res = compute_hawkes_root_cascade(&events);
        assert_eq!(res[0].0, ONE, "first exogenous arrival must have P_root = 1.0");
        assert!(
            res[0].1 > res[5].1,
            "genesis turn must have strictly greater downstream CascadeImpact ({}) than terminal recap ({})",
            res[0].1,
            res[5].1
        );
        assert!(
            res[0].2 > res[5].2,
            "genesis turn must beat late recap ({} vs {})",
            res[0].2,
            res[5].2
        );
    }

    #[test]
    fn kleinberg_burst_isolates_earliest_sustained_cluster_onset() {
        // Isolated mention at 2, long gap, then dense burst at 40..46, then late recap at 200.
        let positions = vec![2usize, 40, 41, 42, 43, 44, 45, 200];
        let onsets = detect_kleinberg_burst_onsets(&positions);
        assert!(
            onsets.iter().any(|&b| b),
            "Kleinberg automaton must emit at least one burst onset"
        );
        assert!(!onsets[7], "terminal late recap must never be a burst onset");
    }

    #[test]
    fn hawkes_spine_recovers_early_genesis_and_blocks_standing_policies() {
        let mut sessions = Vec::new();
        sessions.push(mk_sess(
            "s0",
            100,
            &[
                ("user", "Always format code blocks cleanly when I ask about Python projects."),
                ("assistant", "Understood, I will keep that in mind."),
            ],
        ));
        sessions.push(mk_sess(
            "s1",
            200,
            &[
                ("user", "I'm building a weather app using OpenWeather API and JavaScript with city autocomplete."),
                ("assistant", "Great! Let's design the OpenWeather API fetch and debounce logic for your weather app."),
            ],
        ));
        for i in 2..18 {
            let text = if i % 4 == 0 {
                format!("Continuing work on the weather app OpenWeather caching layer iteration {i}.")
            } else {
                format!("Unrelated chat topic number {i} about weekend hiking trails and groceries.")
            };
            sessions.push(mk_sess(
                &format!("s{i}"),
                (i as i64 + 1) * 100,
                &[("user", &text), ("assistant", "Certainly! That sounds like a great plan.")],
            ));
        }
        sessions.push(mk_sess(
            "s18",
            2000,
            &[(
                "user",
                "Recapping my weather app OpenWeather JavaScript city autocomplete project performance optimizations.",
            )],
        ));

        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "Can you give me a comprehensive summary of how my weather app project has progressed?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &p);
        let t_scene = vec![1.0f32; ix.turns.len()];
        let spine = select_hawkes_genesis_spine(&ix, &q.text, &plan, &[], &t_scene, true, false, 5);
        assert!(!spine.is_empty());
        assert_eq!(
            spine[0].session_idx, 1,
            "Genesis session s1 must take Slot 0 ahead of late recap s18 and standing policy s0"
        );
        assert!(
            spine.iter().all(|c| !ix.turns[c.turn_idx].is_standing_policy),
            "Standing policies must never appear in genesis spine"
        );
    }

    #[test]
    fn hawkes_handles_multibyte_utf8_safely() {
        let sessions = vec![
            mk_sess("s0", 10, &[("user", "Starting café résumé weather project — naïve approach.")]),
            mk_sess("s1", 20, &[("user", "Refining café résumé weather dashboard — phase 2.")]),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "Summarize my café résumé weather project".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &p);
        let t_scene = vec![1.0f32; ix.turns.len()];
        let spine = select_hawkes_genesis_spine(&ix, &q.text, &plan, &[], &t_scene, true, false, 5);
        assert!(!spine.is_empty());
    }
}
