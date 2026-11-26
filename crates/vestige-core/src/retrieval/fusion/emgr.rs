//! EMGR speaker-coupling, QUEST DPBF Group Steiner with FlowReader bridge kernel,
//! PM-2 Sainte-Laguë seat apportionment, xQuAD aspect diversification, and fast
//! Cholesky greedy MAP Determinantal Point Process (DPP) fusion.
//!
//! All scoring, graph routing, and matrix factorization operate strictly in
//! deterministic fixed-point integer arithmetic (`Q16.16` with `Q32.32` Cholesky
//! accumulators). Zero vectors, zero embeddings, zero LLMs, and deterministic
//! tie-breaking throughout.
//!
//! References:
//! - Sun (2026), *Entity-Memory Graph Retrieval* (arXiv:2608.27925, Eqs. 2-5)
//! - Huang et al. (2026), *ContextFact* (arXiv:2611.29155)
//! - Lu et al. (SIGIR 2019), *QUEST* (arXiv:1908.00469, DPBF Group Steiner)
//! - Mehrish & Vascon (2026), *FlowReader* (arXiv:2606.07235, bridge kernel)
//! - Dang & Croft (SIGIR 2012), *PM-2* Sainte-Laguë proportional apportionment
//! - Santos, Macdonald & Ounis (WWW 2010), *xQuAD* aspect diversification
//! - Chen, Zhang & Zhou (NeurIPS 2018), *Fast Greedy MAP Inference for DPP*
//!   (arXiv:1709.05135, Algorithm 1)

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use crate::retrieval::context::RankCtx;
use crate::retrieval::fusion::aspect_coverage_q16::{aspect_ic_q16, isqrt_q16};
use crate::retrieval::fusion::kemeny_mrf::{ieee754_bits_to_q16, log2_u64_q16};
use crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo;
use crate::retrieval::seating::revealed_preference_cone_q16::is_recommendation_request_query;
use crate::retrieval::{
    Index, QueryIntent, QueryPlan, c2_query_mentions_speaker, c2_speaker_conjoined,
    eff_summary_request, raw_tokens, stopwords, turn_is_superseded,
};

/// Unit `1.0` in signed `Q16.16` fixed-point arithmetic (`2^16`).
pub(in crate::retrieval) const ONE: i64 = 65_536;

/// EMGR entity-score blend weight (`alpha = 0.30` in `Q16.16`).
const ALPHA_ENTITY_Q16: i64 = 19_661;
/// Complementary BM25 blend weight (`1 - alpha = 0.70` in `Q16.16`).
const ONE_MINUS_ALPHA_Q16: i64 = ONE - ALPHA_ENTITY_Q16;
/// Minimum normalized entity relevance threshold (`tau = 0.50` in `Q16.16`).
const TAU_ENTITY_Q16: i64 = ONE >> 1;
/// xQuAD diversification trade-off (`lambda = 0.50` in `Q16.16`).
const XQUAD_LAMBDA_Q16: i64 = ONE >> 1;
/// Maximum candidate turns considered by the Cholesky MAP DPP kernel.
const DPP_MAX_CANDS: usize = 40;
/// Minimum Cholesky pivot `d_j` in `Q16.16` (`~0.001`) before early stop.
const DPP_EPS_Q16: i64 = 64;
/// Maximum candidate sessions expanded into the local turn-entity graph.
const MAX_CAND_SESSIONS: usize = 10;
/// Maximum turn nodes in the local QUEST DPBF graph.
const STEINER_MAX_TURNS: usize = 160;
/// Maximum shared entity nodes in the local QUEST DPBF graph.
const STEINER_MAX_ENTS: usize = 128;
/// Maximum terminal aspect groups (`2^5 = 32` bitmask states).
const STEINER_MAX_GROUPS: usize = 5;
/// Exact bitmask state table width (`1 << STEINER_MAX_GROUPS`).
const STEINER_MASK_SLOTS: usize = 1 << STEINER_MAX_GROUPS;
/// Hard bound on DPBF priority-queue pops for sub-millisecond latency.
const STEINER_MAX_POPS: usize = 2_048;
/// Chronological `t <-> t+1` edge cost (`0.25` in `Q16.16`).
const STEINER_CHRONO_COST_Q16: i64 = 0x04000;
/// Question-to-answer adjacency-pair edge cost (`0.125` in `Q16.16`).
const STEINER_ADJ_PAIR_COST_Q16: i64 = 0x02000;
/// Cross-session entity bridge base cost (`0.50` in `Q16.16`).
const STEINER_BRIDGE_BASE_Q16: i64 = 0x08000;
/// Fallback session-hub edge cost (`1.50` in `Q16.16`).
const STEINER_SESSION_HUB_COST_Q16: i64 = ONE + 0x08000;
/// Declarative Steiner bridge turn boost (`+0.25` in `Q16.16`).
const STEINER_BOOST_Q16: i64 = ONE >> 2;
/// `log2(e)` in `Q16.16` (`round(1.44269504 * 65536)`).
const LOG2_E_Q16: i64 = 94_548;

// hardcode-scan: closed-class (counting and multi-session enumeration interrogatives/quantifiers)
const COUNTING_MARKERS: &[&str] = &[
    "how many",
    "different",
    "across",
    "total",
    "all",
    "both",
    "list",
    "altogether",
    "every time",
];

// =========================================================================
// FIXED-POINT Q16.16 ARITHMETIC HELPERS
// =========================================================================

#[inline]
fn mul_q16(a: i64, b: i64) -> i64 {
    let prod = (a as i128).saturating_mul(b as i128) >> 16;
    prod.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

#[inline]
fn div_q16(num: i64, den: i64) -> i64 {
    if den <= 0 {
        return 0;
    }
    let scaled = (num as i128) << 16;
    (scaled / (den as i128)).clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

#[inline]
fn f32_to_q16(v: f32) -> i64 {
    if v <= 0.0 || !v.is_finite() {
        return 0;
    }
    ieee754_bits_to_q16(v.to_bits()).min(i64::MAX as u64) as i64
}

#[inline]
fn ilog2_q16(x: u64) -> i64 {
    log2_u64_q16(x).min(i64::MAX as u64) as i64
}

/// Exact integer floor square root `floor(sqrt(x))` over `u128`.
#[inline]
fn isqrt_u128(x: u128) -> u64 {
    if x == 0 {
        return 0;
    }
    let mut lo: u128 = 0;
    let mut hi: u128 = 1u128 << 64;
    while lo < hi {
        let mid = lo + ((hi - lo) >> 1) + 1;
        if mid.saturating_mul(mid) <= x {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo.min(u64::MAX as u128) as u64
}

/// Pure-integer fixed-point `exp(x)` in `Q16.16` for `x >= 0` (`e^x = 2^(x * log2(e))`).
#[inline]
fn exp_q16(x_q16: i64) -> i64 {
    if x_q16 <= 0 {
        return ONE;
    }
    let y = mul_q16(x_q16.min(8 * ONE), LOG2_E_Q16);
    let int_part = ((y >> 16) as u32).min(10);
    let f = y & (ONE - 1);
    // 4th-order Taylor polynomial of 2^f = exp(f * ln(2)) on f in [0, 1):
    // coefficients in Q16.16: ln(2)=45426, ln(2)^2/2=15743, ln(2)^3/6=3638, ln(2)^4/24=630.
    let poly = 630;
    let poly = 3_638 + mul_q16(f, poly);
    let poly = 15_743 + mul_q16(f, poly);
    let poly = 45_426 + mul_q16(f, poly);
    let pow2_frac = ONE + mul_q16(f, poly);
    pow2_frac.saturating_mul(1i64 << int_part)
}

/// FlowReader moderate-overlap bridge kernel: `p_uv = c_uv * sqrt(1 - c_uv^2)` in `Q16.16`.
/// Peaks at `c_uv = 1 / sqrt(2)` and vanishes for unrelated (`0`) or clone (`1`) pairs.
#[inline]
fn flowreader_bridge_kernel_q16(c_uv_q16: i64) -> i64 {
    let c = c_uv_q16.clamp(0, ONE);
    if c == 0 || c >= ONE {
        return 0;
    }
    let c_sq = mul_q16(c, c);
    let rem = (ONE - c_sq).max(0) as u64;
    let root = isqrt_q16(rem) as i64;
    mul_q16(c, root)
}

/// Closed-class counting / multi-session enumeration detector.
fn is_counting_query(query: &str, eff_it: &QueryIntent) -> bool {
    if eff_it.sniper_how_many {
        return true;
    }
    let ql = query.trim().to_ascii_lowercase();
    let padded = format!(" {ql} ");
    for &marker in COUNTING_MARKERS {
        match marker {
            "all" => {
                if padded.contains(" all ") && !padded.contains(" at all ") {
                    return true;
                }
            }
            "both" => {
                if padded.contains(" both ") {
                    return true;
                }
            }
            "list" => {
                if ql.starts_with("list ")
                    || padded.contains(" list all ")
                    || padded.contains(" list the ")
                    || padded.contains(" list my ")
                    || padded.contains(" list every ")
                {
                    return true;
                }
            }
            _ => {
                if ql.contains(marker) {
                    return true;
                }
            }
        }
    }
    false
}

/// Collect vocab IDs (plain token and `\u{1}ent:` form) for a speaker name string.
fn speaker_vocab_ids(ix: &Index, speaker: &str) -> Vec<u32> {
    let mut ids = Vec::new();
    for tok in raw_tokens(&speaker.to_ascii_lowercase()) {
        if let Some(id) = ix.vocab.get(&tok)
            && !ids.contains(&id)
        {
            ids.push(id);
        }
        let ent_key = format!("\u{1}ent:{tok}");
        if let Some(eid) = ix.vocab.get(&ent_key)
            && !ids.contains(&eid)
        {
            ids.push(eid);
        }
    }
    ids
}

#[inline]
fn is_named_speaker(speaker: &str) -> bool {
    !speaker.is_empty()
        && !speaker.eq_ignore_ascii_case("user")
        && !speaker.eq_ignore_ascii_case("assistant")
}

// =========================================================================
// 1. EMGR SPEAKER-GATED ATTRIBUTION & 1-HOP CHRONOLOGICAL COUPLING
// =========================================================================

/// Compute EMGR speaker-gated attribution, adjacency-pair transfer, 1-hop chronological
/// coupling, and `0.30 / 0.70` BM25 blend over all turns in `cand_sessions`.
///
/// Returns `(turn_idx, score_q16)` sorted descending by score with deterministic
/// tie-breaking by ascending `turn_idx`.
pub(in crate::retrieval) fn compute_emgr_turn_scores(
    ix: &Index,
    query: &str,
    plan: &QueryPlan,
    t_scene: &[f32],
    cand_sessions: &[usize],
) -> Vec<(usize, i64)> {
    if cand_sessions.is_empty() || ix.turns.is_empty() {
        return Vec::new();
    }

    // Deduplicate candidate sessions deterministically while preserving input order.
    let mut seen_sessions: BTreeSet<usize> = BTreeSet::new();
    let mut ordered_sessions: Vec<usize> = Vec::new();
    for &si in cand_sessions {
        if si < ix.session_turns.len() && seen_sessions.insert(si) {
            ordered_sessions.push(si);
        }
    }

    let mut cand_turns: Vec<usize> = Vec::new();
    let mut turn_to_local: BTreeMap<usize, usize> = BTreeMap::new();
    for &si in &ordered_sessions {
        for &ti in &ix.session_turns[si] {
            if ti < ix.turns.len() && !turn_to_local.contains_key(&ti) {
                let local = cand_turns.len();
                turn_to_local.insert(ti, local);
                cand_turns.push(ti);
            }
        }
    }
    if cand_turns.is_empty() {
        return Vec::new();
    }

    // Discover named speakers present in candidate sessions and identify queried speakers.
    let mut all_named_speakers: BTreeSet<String> = BTreeSet::new();
    let mut speaker_turn_counts: BTreeMap<String, u64> = BTreeMap::new();
    for &ti in &cand_turns {
        let sp = ix.turns[ti].speaker.trim();
        if is_named_speaker(sp) {
            let low = sp.to_ascii_lowercase();
            all_named_speakers.insert(low.clone());
            *speaker_turn_counts.entry(low).or_insert(0) += 1;
        }
    }

    let mut queried_speakers: BTreeSet<String> = BTreeSet::new();
    let mut conjoined_speaker = false;
    for sp in &all_named_speakers {
        if c2_query_mentions_speaker(query, sp) {
            queried_speakers.insert(sp.clone());
            if c2_speaker_conjoined(query, sp) {
                conjoined_speaker = true;
            }
        }
    }
    let single_queried_speaker: Option<String> =
        if queried_speakers.len() == 1 && !conjoined_speaker {
            queried_speakers.iter().next().cloned()
        } else {
            None
        };

    // Collect vocab IDs for all named speakers so vocative address does not spoof content.
    let mut all_speaker_term_ids: BTreeSet<u32> = BTreeSet::new();
    let mut speaker_id_map: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for sp in &all_named_speakers {
        let ids = speaker_vocab_ids(ix, sp);
        for &id in &ids {
            all_speaker_term_ids.insert(id);
        }
        speaker_id_map.insert(sp.clone(), ids);
    }

    // Build matched query entity set Q_E+ with normalized match rho(e, u) in [TAU, ONE].
    let max_qt_w = plan
        .terms
        .iter()
        .filter(|qt| !plan.excluded_ids.contains(&qt.id))
        .map(|qt| f32_to_q16(qt.weight))
        .max()
        .unwrap_or(ONE)
        .max(1);

    let mut q_entity_rho: BTreeMap<u32, i64> = BTreeMap::new();
    for qt in &plan.terms {
        if plan.excluded_ids.contains(&qt.id) {
            continue;
        }
        let raw_w = f32_to_q16(qt.weight);
        let norm_w = div_q16(raw_w, max_qt_w).clamp(0, ONE);
        let rho = if qt.exact_hit {
            norm_w.max(TAU_ENTITY_Q16)
        } else if norm_w >= TAU_ENTITY_Q16 {
            norm_w
        } else {
            continue;
        };
        let entry = q_entity_rho.entry(qt.id).or_insert(0);
        if rho > *entry {
            *entry = rho;
        }
    }
    for &reid in &plan.rare_entity_ids {
        if !plan.excluded_ids.contains(&reid) {
            q_entity_rho.insert(reid, ONE);
        }
    }
    for sp in &queried_speakers {
        if let Some(ids) = speaker_id_map.get(sp) {
            for &sid in ids {
                q_entity_rho.insert(sid, ONE);
            }
        }
    }

    // Eq. 2: Degree-discounted entity score h(e, q) = (sum rho(e, u)) / (|Q_E+| * log2(1 + deg(e))).
    let q_e_plus_len = (q_entity_rho.len().max(1) as i64) * ONE;
    let mut h_entity_q16: BTreeMap<u32, i64> = BTreeMap::new();
    for (&eid, &rho) in &q_entity_rho {
        let deg = ix.t_own_stats.df.get(&eid).copied().unwrap_or(1).max(1) as u64;
        let log_deg = ilog2_q16(1 + deg).max(ONE);
        let denom = mul_q16(q_e_plus_len, log_deg).max(1);
        let h_val = div_q16(rho, denom);
        h_entity_q16.insert(eid, h_val);
    }

    // Fallback speaker score for queried speakers whose label is absent from ix.vocab.
    let mut h_speaker_fallback: BTreeMap<String, i64> = BTreeMap::new();
    for sp in &queried_speakers {
        let sp_deg = speaker_turn_counts.get(sp).copied().unwrap_or(1).max(1);
        let log_deg = ilog2_q16(1 + sp_deg).max(ONE);
        let denom = mul_q16(q_e_plus_len, log_deg).max(1);
        h_speaker_fallback.insert(sp.clone(), div_q16(ONE, denom));
    }

    // Eq. 3: Speaker-only reaction dampening:
    //   s_e0(m) = if c_m > 0 { max(c_m, w_m) } else { w_m >> 2 }
    let n_cand = cand_turns.len();
    let mut s_e0: Vec<i64> = vec![0; n_cand];
    let mut c_score: Vec<i64> = vec![0; n_cand];
    let mut is_reactive_turn: Vec<bool> = vec![false; n_cand];

    for (local, &ti) in cand_turns.iter().enumerate() {
        let tr = &ix.turns[ti];
        let sp_low = tr.speaker.trim().to_ascii_lowercase();
        let reactive = tr.is_question || tr.is_backchannel || tr.content.len() <= 3;
        is_reactive_turn[local] = reactive;

        // Speaker attribution score w_m.
        let mut w_m: i64 = 0;
        if let Some(ids) = speaker_id_map.get(&sp_low) {
            for &sid in ids {
                if let Some(&hv) = h_entity_q16.get(&sid) {
                    w_m = w_m.max(hv);
                }
            }
        }
        if w_m == 0
            && let Some(&fb) = h_speaker_fallback.get(&sp_low)
        {
            w_m = fb;
        }

        // Content entity score c_m (excluding speaker-name tokens and excluded query terms).
        let mut c_m: i64 = 0;
        for &eid in tr.entities.iter().chain(tr.content.iter()).chain(tr.coref_entities.iter()) {
            if all_speaker_term_ids.contains(&eid) || plan.excluded_ids.contains(&eid) {
                continue;
            }
            if let Some(&hv) = h_entity_q16.get(&eid) {
                c_m = c_m.max(hv);
            }
        }
        c_score[local] = c_m;

        // Zero-content turns take an immediate 4x dampening (`w_m >> 2`).
        s_e0[local] = if c_m > 0 { c_m.max(w_m) } else { w_m >> 2 };
    }

    // Adjacency-Pair Transfer:
    // When m is a reactive turn by Speaker(m) != QueriedSpeaker(q), transfer its attribution
    // mass to adjacent t +/- 1 owned by QueriedSpeaker(q) within the same session.
    let mut s_e0_adj = s_e0.clone();
    let mut transferred_from = vec![false; n_cand];
    if let Some(ref q_sp) = single_queried_speaker {
        for &si in &ordered_sessions {
            let sess_members = &ix.session_turns[si];
            for (pos, &ti) in sess_members.iter().enumerate() {
                let Some(&local) = turn_to_local.get(&ti) else {
                    continue;
                };
                if !is_reactive_turn[local] {
                    continue;
                }
                let tr = &ix.turns[ti];
                let sp_low = tr.speaker.trim().to_ascii_lowercase();
                if !is_named_speaker(&sp_low) || sp_low == *q_sp {
                    continue;
                }
                let mass = s_e0[local].max(c_score[local]);
                if mass <= 0 {
                    continue;
                }
                // Prefer t + 1 (direct reply to a question) over t - 1 (preceding statement).
                let mut target_local: Option<usize> = None;
                for &nbr_pos in &[pos.checked_add(1), pos.checked_sub(1)] {
                    let Some(np) = nbr_pos else {
                        continue;
                    };
                    let Some(&nbr_ti) = sess_members.get(np) else {
                        continue;
                    };
                    let nbr_tr = &ix.turns[nbr_ti];
                    if nbr_tr.speaker.trim().eq_ignore_ascii_case(q_sp)
                        && !nbr_tr.is_backchannel
                        && !nbr_tr.is_question
                        && let Some(&nl) = turn_to_local.get(&nbr_ti)
                    {
                        target_local = Some(nl);
                        break;
                    }
                }
                if let Some(tl) = target_local {
                    if mass > s_e0_adj[tl] {
                        s_e0_adj[tl] = mass;
                    }
                    // Dampen the partner reaction/prompt turn after transferring its mass.
                    s_e0_adj[local] >>= 2;
                    transferred_from[local] = true;
                }
            }
        }
    }

    // Eq. 4: 1-Hop Chronological Coupling within the same session:
    //   s_e(m) = max(s_e0(m), max_{m' in {t-1, t+1}} (s_e0(m') >> 1))
    // Pure backchannel turns, transferred partner prompts, and zero-content partner
    // reactions do not receive 1-hop lift from an adjacent hit.
    let mut s_e = s_e0_adj.clone();
    let mut zero_content_partner_reaction = vec![false; n_cand];
    for (local, &ti) in cand_turns.iter().enumerate() {
        let sp_low = ix.turns[ti].speaker.trim().to_ascii_lowercase();
        if c_score[local] == 0
            && is_reactive_turn[local]
            && single_queried_speaker
                .as_deref()
                .is_some_and(|q_sp| is_named_speaker(&sp_low) && sp_low != q_sp)
        {
            zero_content_partner_reaction[local] = true;
        }
    }
    for &si in &ordered_sessions {
        let sess_members = &ix.session_turns[si];
        for (pos, &ti) in sess_members.iter().enumerate() {
            let Some(&local) = turn_to_local.get(&ti) else {
                continue;
            };
            if ix.turns[ti].is_backchannel
                || transferred_from[local]
                || zero_content_partner_reaction[local]
            {
                continue;
            }
            let mut coupled = s_e0_adj[local];
            if pos > 0
                && let Some(&prev_local) = turn_to_local.get(&sess_members[pos - 1])
            {
                coupled = coupled.max(s_e0_adj[prev_local] >> 1);
            }
            if pos + 1 < sess_members.len()
                && let Some(&next_local) = turn_to_local.get(&sess_members[pos + 1])
            {
                coupled = coupled.max(s_e0_adj[next_local] >> 1);
            }
            s_e[local] = coupled;
        }
    }

    // Normalize both channels to [0, ONE] and apply Eq. 5 blend:
    //   score(m) = 0.30 * s_e_norm(m) + 0.70 * bm25_norm(m)
    let max_se = s_e.iter().copied().max().unwrap_or(0);
    let bm25_raw: Vec<i64> = cand_turns
        .iter()
        .map(|&ti| f32_to_q16(t_scene.get(ti).copied().unwrap_or(0.0)))
        .collect();
    let max_bm25 = bm25_raw.iter().copied().max().unwrap_or(0);

    let mut out: Vec<(usize, i64)> = Vec::with_capacity(n_cand);
    for (local, &ti) in cand_turns.iter().enumerate() {
        let se_norm = if max_se > 0 {
            div_q16(s_e[local], max_se).clamp(0, ONE)
        } else {
            0
        };
        let mut bm25_norm = if max_bm25 > 0 {
            div_q16(bm25_raw[local], max_bm25).clamp(0, ONE)
        } else {
            0
        };
        // Dampen BM25 echo on zero-content partner reaction turns and transferred prompts.
        if transferred_from[local]
            || zero_content_partner_reaction[local]
            || (ix.turns[ti].is_backchannel && c_score[local] == 0)
        {
            bm25_norm >>= 2;
        }
        let score = mul_q16(ALPHA_ENTITY_Q16, se_norm)
            .saturating_add(mul_q16(ONE_MINUS_ALPHA_Q16, bm25_norm));
        out.push((ti, score));
    }

    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

// =========================================================================
// 2. PM-2 SAINTE-LAGUË APPORTIONMENT & xQuAD ASPECT DIVERSIFICATION
// =========================================================================

/// PM-2 Sainte-Laguë proportional seat apportionment (`SIGIR 2012`) across session clusters.
///
/// Each session cluster `i` holds vote weight `v_i = max_{t in i} score_q16(t)` and awarded
/// seats `s_i` (`0, 1, 2, ...`). At every slot, selects the session maximizing odd-divisor
/// quotient `(v_i << 16) / (2 * s_i + 1)` (`1, 3, 5, 7...`), guaranteeing Slot 0 goes to the
/// highest-vote session before dividing its priority by 3 so distinct sessions fill Slots 1..k.
pub(in crate::retrieval) fn select_pm2_sainte_lague(
    session_turns: &[(usize, Vec<(usize, i64)>)],
    k: usize,
) -> Vec<usize> {
    if session_turns.is_empty() || k == 0 {
        return Vec::new();
    }

    struct ClusterState {
        session_id: usize,
        vote_q16: i64,
        seats: u64,
        cursor: usize,
        turns: Vec<(usize, i64)>,
    }

    let mut clusters: Vec<ClusterState> = Vec::with_capacity(session_turns.len());
    for (sid, raw_turns) in session_turns {
        if raw_turns.is_empty() {
            continue;
        }
        let mut sorted = raw_turns.clone();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        // Deduplicate by turn_idx keeping highest score.
        let mut dedup: Vec<(usize, i64)> = Vec::with_capacity(sorted.len());
        let mut seen: BTreeSet<usize> = BTreeSet::new();
        for (ti, sc) in sorted {
            if seen.insert(ti) {
                dedup.push((ti, sc));
            }
        }
        if let Some(&(_, max_sc)) = dedup.first() {
            clusters.push(ClusterState {
                session_id: *sid,
                vote_q16: max_sc.max(0),
                seats: 0,
                cursor: 0,
                turns: dedup,
            });
        }
    }
    if clusters.is_empty() {
        return Vec::new();
    }

    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut used_turns: BTreeSet<usize> = BTreeSet::new();

    for _ in 0..k {
        let mut best_cluster: Option<(usize, i128, i64, u64, usize)> = None;
        for (idx, cl) in clusters.iter_mut().enumerate() {
            while cl.cursor < cl.turns.len() && used_turns.contains(&cl.turns[cl.cursor].0) {
                cl.cursor += 1;
            }
            if cl.cursor >= cl.turns.len() {
                continue;
            }
            let divisor = (2 * cl.seats + 1) as i128;
            let quotient = ((cl.vote_q16 as i128) << 16) / divisor;
            let next_turn_score = cl.turns[cl.cursor].1;
            let better = match best_cluster {
                None => true,
                Some((_, b_q, b_sc, b_seats, b_sid)) => {
                    quotient > b_q
                        || (quotient == b_q && next_turn_score > b_sc)
                        || (quotient == b_q
                            && next_turn_score == b_sc
                            && cl.seats < b_seats)
                        || (quotient == b_q
                            && next_turn_score == b_sc
                            && cl.seats == b_seats
                            && cl.session_id < b_sid)
                }
            };
            if better {
                best_cluster = Some((idx, quotient, next_turn_score, cl.seats, cl.session_id));
            }
        }
        let Some((best_idx, _, _, _, _)) = best_cluster else {
            break;
        };
        let cl = &mut clusters[best_idx];
        let picked_ti = cl.turns[cl.cursor].0;
        cl.cursor += 1;
        cl.seats += 1;
        used_turns.insert(picked_ti);
        selected.push(picked_ti);
    }

    selected
}

/// xQuAD (`WWW 2010`) greedy multi-aspect turn selector in `Q16.16`.
///
/// Preserves Slot 0 (`prod_uncovered == ONE`, or explicit `lock_slot0`), then maximizes
/// `(1 - lambda) * rel[d] + lambda * sum_a P(a|q) * p(d|a) * prod_{j in S}(1 - p(j|a))`
/// so Slots `1..k` cover uncovered multi-hop aspects.
fn select_xquad_turns_q16(
    ix: &Index,
    plan: &QueryPlan,
    cands: &[(usize, i64)],
    k: usize,
    lock_slot0: Option<usize>,
) -> Vec<usize> {
    if cands.is_empty() || k == 0 {
        return Vec::new();
    }
    let pool: Vec<(usize, i64)> = cands.iter().copied().take(DPP_MAX_CANDS).collect();
    let n_turns = (ix.turns.len() as u32).max(1);

    // Group non-excluded query terms by aspect group `g`.
    let mut group_exact: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    let mut group_stem: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    let mut group_ic: BTreeMap<usize, u64> = BTreeMap::new();
    for qt in &plan.terms {
        if plan.excluded_ids.contains(&qt.id) {
            continue;
        }
        let Some(g) = qt.group else {
            continue;
        };
        let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0);
        let ic = aspect_ic_q16(df, n_turns).max(ONE as u64 / 4);
        let entry = group_ic.entry(g).or_insert(0);
        if ic > *entry {
            *entry = ic;
        }
        if qt.exact_hit {
            group_exact.entry(g).or_default().push(qt.id);
        } else {
            group_stem.entry(g).or_default().push(qt.id);
        }
    }

    let aspects: Vec<usize> = group_ic.keys().copied().collect();
    if aspects.len() < 2 {
        return pool.iter().take(k).map(|&(ti, _)| ti).collect();
    }

    // Normalize P(a|q) proportional to aspect IC so sum_a P(a|q) == ONE.
    let ic_sum: i64 = aspects
        .iter()
        .map(|g| group_ic[g].min(i64::MAX as u64) as i64)
        .sum::<i64>()
        .max(1);
    let p_a_q: Vec<i64> = aspects
        .iter()
        .map(|g| div_q16(group_ic[g] as i64, ic_sum).clamp(1, ONE))
        .collect();

    let m = pool.len();
    let max_rel = pool.iter().map(|&(_, sc)| sc).max().unwrap_or(1).max(1);
    let rel_norm: Vec<i64> = pool
        .iter()
        .map(|&(_, sc)| div_q16(sc.max(0), max_rel).clamp(0, ONE))
        .collect();

    // Per-turn aspect coverage p(d|a) in {0, ONE/2, ONE}.
    let num_a = aspects.len();
    let mut p_d_a: Vec<Vec<i64>> = vec![vec![0; num_a]; m];
    for (idx, &(ti, _)) in pool.iter().enumerate() {
        let tr = &ix.turns[ti];
        for (ai, g) in aspects.iter().enumerate() {
            let exact_hit = group_exact.get(g).is_some_and(|ids| {
                ids.iter()
                    .any(|id| tr.content.contains(id) || tr.entities.contains(id))
            });
            if exact_hit {
                p_d_a[idx][ai] = ONE;
            } else {
                let stem_hit = group_stem.get(g).is_some_and(|ids| {
                    ids.iter().any(|id| {
                        tr.content.contains(id)
                            || tr.entities.contains(id)
                            || tr.coref_entities.contains(id)
                    })
                });
                if stem_hit {
                    p_d_a[idx][ai] = ONE >> 1;
                }
            }
        }
    }

    let mut prod_uncovered: Vec<i64> = vec![ONE; num_a];
    let mut used = vec![false; m];
    let mut selected: Vec<usize> = Vec::with_capacity(k.min(m));

    for step in 0..k.min(m) {
        let best_idx = if step == 0
            && let Some(lock_ti) = lock_slot0
            && let Some(pos) = pool.iter().position(|&(ti, _)| ti == lock_ti)
        {
            Some(pos)
        } else {
            let step_lambda = if step == 0 { 0 } else { XQUAD_LAMBDA_Q16 };
            let step_rel_w = ONE - step_lambda;
            let mut raw_divs = vec![0i64; m];
            let mut max_div: i64 = 0;
            for idx in 0..m {
                if used[idx] {
                    continue;
                }
                let mut dg: i64 = 0;
                for ai in 0..num_a {
                    let cov = p_d_a[idx][ai];
                    if cov == 0 || prod_uncovered[ai] == 0 {
                        continue;
                    }
                    let term = mul_q16(mul_q16(p_a_q[ai], cov), prod_uncovered[ai]);
                    dg = dg.saturating_add(term);
                }
                raw_divs[idx] = dg;
                if dg > max_div {
                    max_div = dg;
                }
            }
            let mut best: Option<(usize, i64, i64, usize)> = None;
            for idx in 0..m {
                if used[idx] {
                    continue;
                }
                let div_norm = if max_div > 0 {
                    div_q16(raw_divs[idx], max_div).clamp(0, ONE)
                } else {
                    0
                };
                let sc = mul_q16(step_rel_w, rel_norm[idx])
                    .saturating_add(mul_q16(step_lambda, div_norm));
                let ti = pool[idx].0;
                let better = match best {
                    None => true,
                    Some((_, b_sc, b_rel, b_ti)) => {
                        sc > b_sc
                            || (sc == b_sc && rel_norm[idx] > b_rel)
                            || (sc == b_sc && rel_norm[idx] == b_rel && ti < b_ti)
                    }
                };
                if better {
                    best = Some((idx, sc, rel_norm[idx], ti));
                }
            }
            best.map(|(idx, _, _, _)| idx)
        };

        let Some(chosen) = best_idx else {
            break;
        };
        used[chosen] = true;
        selected.push(pool[chosen].0);
        for ai in 0..num_a {
            let keep = ONE - p_d_a[chosen][ai];
            prod_uncovered[ai] = mul_q16(prod_uncovered[ai], keep);
        }
    }

    selected
}

// =========================================================================
// 3. FLOWREADER BRIDGE KERNEL + EXACT BITMASK QUEST DPBF GROUP STEINER TREE
// =========================================================================

#[derive(Clone, Copy)]
enum SteinerParent {
    Empty,
    Terminal,
    Grow { prev_node: u16 },
    Merge { left_mask: u8, right_mask: u8 },
}

/// Extract minimum-cost connected cross-session hop turns using exact bitmask QUEST DPBF
/// Group Steiner Tree (`l <= 5` terminal groups, bounded `<= 2048` pops) enhanced with the
/// FlowReader moderate-overlap bridge kernel `p_uv = c_uv * sqrt(1 - c_uv^2)`.
fn extract_quest_steiner_turns(
    ix: &Index,
    plan: &QueryPlan,
    emgr_scores: &[(usize, i64)],
    cand_sessions: &[usize],
) -> Vec<usize> {
    if emgr_scores.is_empty() || cand_sessions.is_empty() {
        return Vec::new();
    }

    // 1. Select up to STEINER_MAX_TURNS declarative turns across Top-10 candidate sessions.
    let score_map: BTreeMap<usize, i64> = emgr_scores.iter().copied().collect();
    let mut sess_list: Vec<usize> = Vec::new();
    let mut sess_set: BTreeSet<usize> = BTreeSet::new();
    for &si in cand_sessions.iter().take(MAX_CAND_SESSIONS) {
        if si < ix.session_turns.len() && sess_set.insert(si) {
            sess_list.push(si);
        }
    }
    if sess_list.is_empty() {
        return Vec::new();
    }

    let mut scored_turns: Vec<(usize, i64)> = Vec::new();
    for &si in &sess_list {
        for &ti in &ix.session_turns[si] {
            let tr = &ix.turns[ti];
            if tr.is_backchannel || turn_is_superseded(ix, ti) {
                continue;
            }
            let sc = score_map.get(&ti).copied().unwrap_or(0);
            scored_turns.push((ti, sc));
        }
    }
    scored_turns.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored_turns.truncate(STEINER_MAX_TURNS);
    if scored_turns.len() < 2 {
        return Vec::new();
    }

    let n_turns_local = scored_turns.len();
    let turn_ids: Vec<usize> = scored_turns.iter().map(|&(ti, _)| ti).collect();
    let mut turn_to_node: BTreeMap<usize, usize> = BTreeMap::new();
    for (idx, &ti) in turn_ids.iter().enumerate() {
        turn_to_node.insert(ti, idx);
    }

    // 2. Build per-turn bridge entity/rare-content sets and select shared Entity nodes V_E.
    let n_corpus_turns = (ix.turns.len() as u32).max(1);
    let max_bridge_df = (n_corpus_turns / 3).max(2);
    let stop = stopwords();

    let mut q_term_ids: BTreeSet<u32> = BTreeSet::new();
    for qt in &plan.terms {
        if !plan.excluded_ids.contains(&qt.id) {
            q_term_ids.insert(qt.id);
        }
    }

    let mut turn_ent_sets: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); n_turns_local];
    let mut ent_postings: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (local, &ti) in turn_ids.iter().enumerate() {
        let tr = &ix.turns[ti];
        for &eid in tr
            .entities
            .iter()
            .chain(tr.coref_entities.iter())
            .chain(tr.content.iter())
        {
            if plan.excluded_ids.contains(&eid) {
                continue;
            }
            let is_q = q_term_ids.contains(&eid);
            let is_ent = tr.entities.contains(&eid) || tr.coref_entities.contains(&eid);
            let df = ix.t_own_stats.df.get(&eid).copied().unwrap_or(1);
            if !is_q && !is_ent && df > max_bridge_df {
                continue;
            }
            if let Some(name) = ix.vocab.names.get(eid as usize) {
                let clean = name.strip_prefix("\u{1}ent:").unwrap_or(name.as_str());
                if clean.len() < 2 || stop.contains(clean) {
                    continue;
                }
            }
            if turn_ent_sets[local].insert(eid) {
                ent_postings.entry(eid).or_default().push(local);
            }
        }
    }

    // Rank shared entities (degree >= 2 or query-matched) by IC and cap at STEINER_MAX_ENTS.
    let mut cand_ents: Vec<(u32, u64)> = ent_postings
        .iter()
        .filter(|&(eid, posts)| posts.len() >= 2 || q_term_ids.contains(eid))
        .map(|(&eid, _)| {
            let df = ix.t_own_stats.df.get(&eid).copied().unwrap_or(1);
            (eid, aspect_ic_q16(df, n_corpus_turns))
        })
        .collect();
    cand_ents.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    cand_ents.truncate(STEINER_MAX_ENTS);

    let ent_offset = n_turns_local;
    let n_ents = cand_ents.len();
    let sess_offset = ent_offset + n_ents;
    let n_sess = sess_list.len();
    let n_total = sess_offset + n_sess;

    let mut ent_to_node: BTreeMap<u32, usize> = BTreeMap::new();
    for (idx, &(eid, _)) in cand_ents.iter().enumerate() {
        ent_to_node.insert(eid, ent_offset + idx);
    }
    let mut sess_to_node: BTreeMap<usize, usize> = BTreeMap::new();
    for (idx, &si) in sess_list.iter().enumerate() {
        sess_to_node.insert(si, sess_offset + idx);
    }

    // 3. Build undirected positive-cost adjacency lists.
    let mut adj: Vec<Vec<(usize, i64)>> = vec![Vec::new(); n_total];
    let mut add_edge = |u: usize, v: usize, cost: i64| {
        if u == v {
            return;
        }
        let c = cost.max(1);
        adj[u].push((v, c));
        adj[v].push((u, c));
    };

    // 3a. Turn <-> Entity edges (`1 / (word_dist + 1)` proximity cost + degree damping).
    for &(eid, _) in &cand_ents {
        let e_node = ent_to_node[&eid];
        let Some(posts) = ent_postings.get(&eid) else {
            continue;
        };
        let local_deg = posts.len() as u64;
        let deg_damp = mul_q16(STEINER_BRIDGE_BASE_Q16, ilog2_q16(1 + local_deg).max(ONE))
            .clamp(STEINER_CHRONO_COST_Q16, 2 * ONE);
        for &local_t in posts {
            let ti = turn_ids[local_t];
            let tr = &ix.turns[ti];
            let mut min_dist: i64 = 8;
            if q_term_ids.contains(&eid) {
                min_dist = 0;
            } else {
                let mut e_pos: Option<usize> = None;
                let mut q_pos: Option<usize> = None;
                for (pos, &(ex_id, st_id)) in tr.seq.iter().enumerate() {
                    if ex_id == eid || st_id == eid {
                        e_pos = Some(pos);
                    }
                    if q_term_ids.contains(&ex_id) || q_term_ids.contains(&st_id) {
                        q_pos = Some(pos);
                    }
                    if let (Some(ep), Some(qp)) = (e_pos, q_pos) {
                        min_dist = min_dist.min(ep.abs_diff(qp) as i64);
                    }
                }
            }
            let prox = ONE / (min_dist + 1);
            let prox_cost = (ONE - prox.min(ONE - 4_096)).max(4_096);
            let edge_cost = (prox_cost + (deg_damp >> 1)).max(4_096);
            add_edge(local_t, e_node, edge_cost);
        }
    }

    // 3b. Chronological t <-> t+1 edges within the same session.
    for &si in &sess_list {
        let members = &ix.session_turns[si];
        for pair in members.windows(2) {
            let (t_a, t_b) = (pair[0], pair[1]);
            if let (Some(&na), Some(&nb)) = (turn_to_node.get(&t_a), turn_to_node.get(&t_b)) {
                let is_qa = ix.turns[t_a].is_question && !ix.turns[t_b].is_question;
                let cost = if is_qa {
                    STEINER_ADJ_PAIR_COST_Q16
                } else {
                    STEINER_CHRONO_COST_Q16
                };
                add_edge(na, nb, cost);
            }
        }
    }

    // 3c. Cross-session Turn <-> Turn bridge edges weighted by FlowReader moderate-overlap kernel.
    let mut bridge_pairs: BTreeSet<(usize, usize)> = BTreeSet::new();
    for &(eid, _) in &cand_ents {
        let Some(posts) = ent_postings.get(&eid) else {
            continue;
        };
        if posts.len() < 2 || posts.len() > 12 {
            continue;
        }
        for i in 0..posts.len() {
            let u = posts[i];
            let su = ix.turns[turn_ids[u]].session;
            for &v in &posts[(i + 1)..] {
                let sv = ix.turns[turn_ids[v]].session;
                if su == sv {
                    continue;
                }
                let pair = if u < v { (u, v) } else { (v, u) };
                if !bridge_pairs.insert(pair) {
                    continue;
                }
                let set_u = &turn_ent_sets[u];
                let set_v = &turn_ent_sets[v];
                let inter = set_u.intersection(set_v).count() as i64;
                let union = (set_u.len() + set_v.len()).saturating_sub(inter as usize) as i64;
                if inter > 0 && union > 0 {
                    let c_uv = div_q16(inter, union).clamp(0, ONE);
                    let p_uv = flowreader_bridge_kernel_q16(c_uv);
                    if p_uv > 0 {
                        let cost = (ONE - p_uv + STEINER_CHRONO_COST_Q16).max(STEINER_CHRONO_COST_Q16);
                        add_edge(u, v, cost);
                    }
                }
            }
        }
    }

    // 3d. Turn <-> Session hub fallback edges.
    for (local, &ti) in turn_ids.iter().enumerate() {
        if let Some(&s_node) = sess_to_node.get(&ix.turns[ti].session) {
            add_edge(local, s_node, STEINER_SESSION_HUB_COST_Q16);
        }
    }

    // Deduplicate & sort adjacency lists for deterministic relaxation order.
    for edges in &mut adj {
        edges.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        edges.dedup_by(|a, b| a.0 == b.0);
    }

    // 4. Partition query aspects into l <= 5 terminal groups T_0..T_{l-1}.
    let mut group_terms: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    let mut group_ic: BTreeMap<usize, u64> = BTreeMap::new();
    for qt in &plan.terms {
        if plan.excluded_ids.contains(&qt.id) {
            continue;
        }
        let Some(g) = qt.group else {
            continue;
        };
        group_terms.entry(g).or_default().push(qt.id);
        let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(1);
        let ic = aspect_ic_q16(df, n_corpus_turns);
        let entry = group_ic.entry(g).or_insert(0);
        if ic > *entry {
            *entry = ic;
        }
    }

    let mut ranked_groups: Vec<(usize, u64)> = group_ic.into_iter().collect();
    ranked_groups.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut terminal_groups: Vec<Vec<usize>> = Vec::new();
    for (g, _) in ranked_groups {
        if terminal_groups.len() >= STEINER_MAX_GROUPS {
            break;
        }
        let Some(tids) = group_terms.get(&g) else {
            continue;
        };
        let mut nodes: BTreeSet<usize> = BTreeSet::new();
        for (local, &ti) in turn_ids.iter().enumerate() {
            let tr = &ix.turns[ti];
            if tids.iter().any(|id| {
                tr.content.contains(id)
                    || tr.entities.contains(id)
                    || tr.coref_entities.contains(id)
            }) {
                nodes.insert(local);
            }
        }
        if !nodes.is_empty() {
            terminal_groups.push(nodes.into_iter().collect());
        }
    }
    if terminal_groups.len() < 2 {
        return Vec::new();
    }

    // 5. Run exact bitmask DPBF Group Steiner (`BinaryHeap`, bounded <= STEINER_MAX_POPS).
    let l = terminal_groups.len();
    let full_mask: u8 = (1u8 << l) - 1;
    let inf: i64 = i64::MAX / 4;
    let mut best_cost: Vec<[i64; STEINER_MASK_SLOTS]> = vec![[inf; STEINER_MASK_SLOTS]; n_total];
    let mut parent: Vec<[SteinerParent; STEINER_MASK_SLOTS]> =
        vec![[SteinerParent::Empty; STEINER_MASK_SLOTS]; n_total];
    let mut heap: BinaryHeap<Reverse<(i64, u16, u8)>> = BinaryHeap::new();

    for (gi, group_nodes) in terminal_groups.iter().enumerate() {
        let mask: u8 = 1u8 << gi;
        for &v in group_nodes {
            if best_cost[v][mask as usize] > 0 {
                best_cost[v][mask as usize] = 0;
                parent[v][mask as usize] = SteinerParent::Terminal;
                heap.push(Reverse((0, v as u16, mask)));
            }
        }
    }

    let mut pops: usize = 0;
    let mut solution_roots: Vec<usize> = Vec::new();
    let mut best_full_cost: i64 = inf;

    while pops < STEINER_MAX_POPS {
        let Some(Reverse((cost, v_u16, mask))) = heap.pop() else {
            break;
        };
        let v = v_u16 as usize;
        if cost > best_cost[v][mask as usize] {
            continue;
        }
        pops += 1;

        if mask == full_mask {
            if best_full_cost == inf {
                best_full_cost = cost;
            }
            if cost > best_full_cost {
                break;
            }
            if !solution_roots.contains(&v) {
                solution_roots.push(v);
                if solution_roots.len() >= 2 {
                    break;
                }
            }
            continue;
        }

        // Grow along incident edges.
        for &(u, e_cost) in &adj[v] {
            let nc = cost.saturating_add(e_cost);
            if nc < best_cost[u][mask as usize] {
                best_cost[u][mask as usize] = nc;
                parent[u][mask as usize] = SteinerParent::Grow { prev_node: v_u16 };
                heap.push(Reverse((nc, u as u16, mask)));
            }
        }

        // Merge with disjoint masks rooted at the same vertex v.
        for m2 in 1..=full_mask {
            if (mask & m2) != 0 {
                continue;
            }
            let c2 = best_cost[v][m2 as usize];
            if c2 >= inf {
                continue;
            }
            let merged = mask | m2;
            let nc = cost.saturating_add(c2);
            if nc < best_cost[v][merged as usize] {
                best_cost[v][merged as usize] = nc;
                parent[v][merged as usize] = SteinerParent::Merge {
                    left_mask: mask,
                    right_mask: m2,
                };
                heap.push(Reverse((nc, v_u16, merged)));
            }
        }
    }

    if solution_roots.is_empty() {
        return Vec::new();
    }

    // 6. Iteratively unwind parent pointers to extract V* intersect V_T.
    let mut extracted_turns: BTreeSet<usize> = BTreeSet::new();
    for root in solution_roots {
        let mut stack: Vec<(usize, u8)> = vec![(root, full_mask)];
        let mut visited_states: BTreeSet<(usize, u8)> = BTreeSet::new();
        while let Some((node, mask)) = stack.pop() {
            if !visited_states.insert((node, mask)) {
                continue;
            }
            if node < n_turns_local {
                extracted_turns.insert(turn_ids[node]);
            }
            match parent[node][mask as usize] {
                SteinerParent::Empty | SteinerParent::Terminal => {}
                SteinerParent::Grow { prev_node } => {
                    stack.push((prev_node as usize, mask));
                }
                SteinerParent::Merge {
                    left_mask,
                    right_mask,
                } => {
                    stack.push((node, left_mask));
                    stack.push((node, right_mask));
                }
            }
        }
    }

    let mut out: Vec<usize> = extracted_turns.into_iter().collect();
    out.sort_by(|&a, &b| {
        let sa = score_map.get(&a).copied().unwrap_or(0);
        let sb = score_map.get(&b).copied().unwrap_or(0);
        sb.cmp(&sa).then_with(|| a.cmp(&b))
    });
    out
}

// =========================================================================
// 4. O(N^2 M) FAST CHOLESKY GREEDY MAP DETERMINANTAL POINT PROCESS (DPP)
// =========================================================================

/// Build L2-normalized sparse feature vector `phi_i` in `Q16.16` over
/// `[IC-weighted content IDs, entity IDs, session_id tag]` (zero embeddings).
fn build_turn_sparse_phi_q16(ix: &Index, ti: usize) -> Vec<(u64, i64)> {
    let tr = &ix.turns[ti];
    let n_turns = (ix.turns.len() as u32).max(1);
    let stop = stopwords();
    let mut raw: BTreeMap<u64, i64> = BTreeMap::new();

    for &cid in &tr.content {
        if let Some(name) = ix.vocab.names.get(cid as usize)
            && stop.contains(name.as_str())
        {
            continue;
        }
        let df = ix.t_own_stats.df.get(&cid).copied().unwrap_or(1);
        let ic = (aspect_ic_q16(df, n_turns) as i64).max(ONE >> 2);
        raw.insert(cid as u64, ic);
    }
    for &eid in tr.entities.iter().chain(tr.coref_entities.iter()) {
        let df = ix.t_own_stats.df.get(&eid).copied().unwrap_or(1);
        let ic = (aspect_ic_q16(df, n_turns) as i64).max(ONE >> 1);
        let ent_w = ic.saturating_add(ic >> 1); // 1.5x entity emphasis
        raw.insert((1u64 << 32) | (eid as u64), ent_w);
    }

    // Compute content+entity squared L2 norm in Q32.32, then set the session tag
    // weight so same-session identity contributes an exact length-invariant 50%
    // of ||phi||_2^2 (eliminating duplicate same-session reaction turns while
    // keeping cross-session evidence orthogonal).
    let mut content_norm_sq: u128 = 0;
    for &w in raw.values() {
        let wu = w.max(0) as u128;
        content_norm_sq = content_norm_sq.saturating_add(wu.saturating_mul(wu));
    }
    let sess_w = if content_norm_sq > 0 {
        (isqrt_u128(content_norm_sq) as i64).max(ONE)
    } else {
        ONE
    };
    raw.insert((2u64 << 32) | (tr.session as u64), sess_w);

    let total_norm_sq = content_norm_sq.saturating_add((sess_w as u128).saturating_mul(sess_w as u128));
    let norm = (isqrt_u128(total_norm_sq) as i64).max(1);

    let mut phi: Vec<(u64, i64)> = Vec::with_capacity(raw.len());
    for (feat_key, w) in raw {
        let unit_w = div_q16(w, norm);
        if unit_w > 0 {
            phi.push((feat_key, unit_w));
        }
    }
    phi
}

/// Sparse dot product `<phi_a, phi_b>` in `Q16.16` over sorted `(key, weight_q16)` slices.
#[inline]
fn sparse_dot_q16(a: &[(u64, i64)], b: &[(u64, i64)]) -> i64 {
    let (mut i, mut j) = (0usize, 0usize);
    let mut acc: i128 = 0;
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                acc = acc.saturating_add((a[i].1 as i128).saturating_mul(b[j].1 as i128));
                i += 1;
                j += 1;
            }
        }
    }
    ((acc >> 16).clamp(0, ONE as i128)) as i64
}

/// Fast `O(N^2 M)` Cholesky greedy MAP Determinantal Point Process selector
/// (`NeurIPS 2018 arXiv:1709.05135`, Algorithm 1) over an exact PSD sparse lexical,
/// entity, and session Gram kernel `L_{ij} = r_i * <phi_i, phi_j> * r_j`.
///
/// Uses `Q32.32` (`i128`) Cholesky accumulators with integer `isqrt_u128` for zero
/// floating-point drift and deterministic tie-breaking.
pub(in crate::retrieval) fn select_cholesky_dpp(
    ix: &Index,
    cands: &[(usize, i64)],
    k: usize,
) -> Vec<usize> {
    if cands.is_empty() || k == 0 {
        return Vec::new();
    }

    // Deduplicate candidate turns keeping maximum score, then take Top-M (M <= 40).
    let mut best_by_turn: BTreeMap<usize, i64> = BTreeMap::new();
    for &(ti, sc) in cands {
        if ti >= ix.turns.len() {
            continue;
        }
        let entry = best_by_turn.entry(ti).or_insert(i64::MIN);
        if sc > *entry {
            *entry = sc;
        }
    }
    let mut pool: Vec<(usize, i64)> = best_by_turn.into_iter().collect();
    pool.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pool.truncate(DPP_MAX_CANDS);

    let m = pool.len();
    let target_k = k.min(m);
    if target_k == 0 {
        return Vec::new();
    }

    // Build L2-normalized sparse feature vectors phi_i and quality scores r_i = exp(s_norm_i).
    let max_sc = pool.iter().map(|&(_, sc)| sc).max().unwrap_or(1).max(1);
    let phis: Vec<Vec<(u64, i64)>> = pool
        .iter()
        .map(|&(ti, _)| build_turn_sparse_phi_q16(ix, ti))
        .collect();
    let r_q16: Vec<i64> = pool
        .iter()
        .map(|&(_, sc)| {
            let s_norm = div_q16(sc.max(0), max_sc).clamp(0, ONE);
            exp_q16(s_norm)
        })
        .collect();

    // Construct symmetric PSD L-ensemble matrix L_{ij} in Q32.32 (`i128`).
    let mut l_q32: Vec<Vec<i128>> = vec![vec![0i128; m]; m];
    for i in 0..m {
        let ri = r_q16[i] as i128;
        let r_sq_q16 = (ri.saturating_mul(ri)) >> 16;
        l_q32[i][i] = r_sq_q16.saturating_mul(ONE as i128);
        for j in (i + 1)..m {
            let dot = sparse_dot_q16(&phis[i], &phis[j]) as i128;
            let rr_q16 = (ri.saturating_mul(r_q16[j] as i128)) >> 16;
            let val = rr_q16.saturating_mul(dot);
            l_q32[i][j] = val;
            l_q32[j][i] = val;
        }
    }

    // Chen et al. Algorithm 1: incremental Cholesky factor c[i] (Q16.16) and residual d2[i] (Q32.32).
    let mut d2: Vec<i128> = (0..m).map(|i| l_q32[i][i]).collect();
    let mut cholesky_c: Vec<Vec<i64>> = vec![Vec::with_capacity(target_k); m];
    let mut chosen = vec![false; m];
    let mut selected_indices: Vec<usize> = Vec::with_capacity(target_k);

    let argmax_remaining = |d2_slice: &[i128], chosen_slice: &[bool]| -> Option<usize> {
        let mut best: Option<(usize, i128, i64, usize)> = None;
        for i in 0..m {
            if chosen_slice[i] {
                continue;
            }
            let val = d2_slice[i];
            let (ti, sc) = pool[i];
            let better = match best {
                None => true,
                Some((_, b_val, b_sc, b_ti)) => {
                    val > b_val
                        || (val == b_val && sc > b_sc)
                        || (val == b_val && sc == b_sc && ti < b_ti)
                }
            };
            if better {
                best = Some((i, val, sc, ti));
            }
        }
        best.map(|(idx, _, _, _)| idx)
    };

    let Some(mut j) = argmax_remaining(&d2, &chosen) else {
        return Vec::new();
    };
    chosen[j] = true;
    selected_indices.push(j);

    for _step in 1..target_k {
        let d2_j = d2[j].max(0) as u128;
        let dj = isqrt_u128(d2_j) as i64;
        if dj < DPP_EPS_Q16 {
            break;
        }
        let dj_i128 = dj as i128;
        for i in 0..m {
            if chosen[i] {
                continue;
            }
            let mut dot_c: i128 = 0;
            for (&a, &b) in cholesky_c[j].iter().zip(cholesky_c[i].iter()) {
                dot_c = dot_c.saturating_add((a as i128).saturating_mul(b as i128));
            }
            let e_i = ((l_q32[j][i].saturating_sub(dot_c)) / dj_i128)
                .clamp(i64::MIN as i128, i64::MAX as i128) as i64;
            cholesky_c[i].push(e_i);
            let e_sq = (e_i as i128).saturating_mul(e_i as i128);
            d2[i] = d2[i].saturating_sub(e_sq).max(0);
        }
        let Some(next_j) = argmax_remaining(&d2, &chosen) else {
            break;
        };
        chosen[next_j] = true;
        selected_indices.push(next_j);
        j = next_j;
    }

    // If Cholesky terminated early due to numerical rank exhaustion, backfill remaining slots
    // deterministically by base score.
    if selected_indices.len() < target_k {
        for (idx, &is_chosen) in chosen.iter().enumerate() {
            if !is_chosen {
                selected_indices.push(idx);
                if selected_indices.len() >= target_k {
                    break;
                }
            }
        }
    }

    selected_indices.into_iter().map(|idx| pool[idx].0).collect()
}

// =========================================================================
// 5. PIPELINE ENTRY POINT FOR RankCtx
// =========================================================================

/// Identify the query's lowest-DF exact non-stopword term ID (`star_id`) for Slot-0 safety.
fn lowest_df_star_term(ix: &Index, plan: &QueryPlan) -> Option<u32> {
    let stop = stopwords();
    let mut best: Option<(u32, u32)> = None;
    for qt in &plan.terms {
        if !qt.exact_hit || plan.excluded_ids.contains(&qt.id) {
            continue;
        }
        if let Some(name) = ix.vocab.names.get(qt.id as usize) {
            let clean = name.strip_prefix("\u{1}ent:").unwrap_or(name.as_str());
            if clean.len() < 2 || stop.contains(clean) {
                continue;
            }
        }
        let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0);
        if df == 0 {
            continue;
        }
        match best {
            None => best = Some((df, qt.id)),
            Some((b_df, b_id)) if df < b_df || (df == b_df && qt.id < b_id) => {
                best = Some((df, qt.id));
            }
            _ => {}
        }
    }
    best.map(|(_, id)| id)
}

#[inline]
fn turn_has_term(ix: &Index, ti: usize, term_id: u32) -> bool {
    let tr = &ix.turns[ti];
    tr.content.contains(&term_id)
        || tr.entities.contains(&term_id)
        || tr.coref_entities.contains(&term_id)
}

#[inline]
fn is_declarative_evidence(ix: &Index, ti: usize, query: &str) -> bool {
    let tr = &ix.turns[ti];
    !tr.is_question
        && !tr.is_backchannel
        && !turn_is_superseded(ix, ti)
        && !is_discourse_closure_or_meta_echo(ix, ti, query)
}

/// Splice a prioritized Top-`top_n` turn list into `final_turn_indices`, preserving
/// original list length and tail candidates without duplicates.
fn splice_top_turns(final_turn_indices: &mut Vec<usize>, prioritized: &[usize], top_n: usize) {
    if prioritized.is_empty() || final_turn_indices.is_empty() || top_n == 0 {
        return;
    }
    let orig_len = final_turn_indices.len();
    let mut next: Vec<usize> = Vec::with_capacity(orig_len);
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    for &ti in prioritized.iter().take(top_n) {
        if seen.insert(ti) {
            next.push(ti);
        }
    }
    for &ti in final_turn_indices.iter() {
        if next.len() >= orig_len {
            break;
        }
        if seen.insert(ti) {
            next.push(ti);
        }
    }
    *final_turn_indices = next;
}

/// Ensure sessions represented by Slots `1..top_n` appear inside Top-`top_n` of
/// `final_session_indices` without ever altering Slot 0 (`final_session_indices[0]`).
fn sync_tail_sessions_preserving_slot0(
    ix: &Index,
    final_turn_indices: &[usize],
    final_session_indices: &mut Vec<usize>,
    top_n: usize,
) {
    if final_session_indices.len() < 2 || top_n < 2 {
        return;
    }
    let orig_len = final_session_indices.len();
    let head_sess = final_session_indices[0];
    let mut next: Vec<usize> = Vec::with_capacity(orig_len);
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    next.push(head_sess);
    seen.insert(head_sess);

    for &ti in final_turn_indices.iter().take(top_n) {
        if next.len() >= top_n {
            break;
        }
        let si = ix.turns[ti].session;
        if seen.insert(si) {
            next.push(si);
        }
    }
    for &si in final_session_indices.iter() {
        if next.len() >= orig_len {
            break;
        }
        if seen.insert(si) {
            next.push(si);
        }
    }
    *final_session_indices = next;
}

/// Stage 12 pipeline entry point applying EMGR speaker-coupled attribution, QUEST DPBF
/// Group Steiner / xQuAD multi-hop bridging, PM-2 Sainte-Laguë seat apportionment, and
/// Cholesky MAP DPP diversification over `RankCtx`.
///
/// Strictly skips specialized domain engines (`has_tool_calls`, `is_monologue_corpus`,
/// `is_event_sequence`, `eff_summary_request`, `implicit_disclosure_locked`,
/// `torrens_locked_head`, `task_under_rules`, `preference`) and preserves Slot 0 on
/// solved declarative single-hop hits (`star_bit` & `R@1` safety).
pub(in crate::retrieval) fn apply_emgr_steiner_dpp_fusion(ctx: &mut RankCtx<'_>) {
    if ctx.ix.has_tool_calls
        || ctx.is_monologue_corpus
        || ctx.eff_it.is_event_sequence
        || eff_summary_request(&ctx.q.text)
        || ctx.implicit_disclosure_locked
        || ctx.torrens_locked_head
        || ctx.eff_it.task_under_rules
        || ctx.eff_it.preference
        || is_recommendation_request_query(&ctx.q.text, ctx.pref_on)
        || ctx.ns < 3
        || ctx.nt < 12
        || ctx.final_turn_indices.len() < 2
        || ctx.final_session_indices.is_empty()
        || ctx.k < 2
    {
        return;
    }
    let q_raw_od = raw_tokens(&ctx.q.text);
    let q_refs_od: Vec<&str> = q_raw_od.iter().map(String::as_str).collect();
    if ctx.has_named_speakers_eff
        && crate::retrieval::lexical::speaker_translation_q16::is_open_domain_inferential_shape(
            &q_refs_od,
        )
    {
        return;
    }

    let top_n = ctx.k.min(5).min(ctx.final_turn_indices.len());
    let h0 = ctx.final_turn_indices[0];
    let star_id = lowest_df_star_term(ctx.ix, &ctx.plan);
    let h0_declarative = is_declarative_evidence(ctx.ix, h0, &ctx.q.text);
    let h0_has_star = star_id.is_none_or(|sid| turn_has_term(ctx.ix, h0, sid));
    let h0_full_cov = ctx.q_group_cnt > 0
        && (ctx.t_exact_mask.get(h0).copied().unwrap_or(0) & ctx.q_group_mask).count_ones() as usize
            >= ctx.q_group_cnt;
    let counting = is_counting_query(&ctx.q.text, &ctx.eff_it);
    let multi_hop = ctx.eff_it.multi_hop
        || (ctx.q_group_cnt >= 2
            && (ctx.ghostcurrent.open_circuit
                || !ctx.ghostcurrent.missing_query_groups.is_empty()));

    // Collect Top-10 candidate sessions from session head and Top-10 turn seats.
    let mut cand_sessions: Vec<usize> = Vec::with_capacity(MAX_CAND_SESSIONS);
    let mut seen_s: BTreeSet<usize> = BTreeSet::new();
    for &si in ctx.final_session_indices.iter().take(MAX_CAND_SESSIONS) {
        if seen_s.insert(si) {
            cand_sessions.push(si);
        }
    }
    for &ti in ctx.final_turn_indices.iter().take(MAX_CAND_SESSIONS) {
        if cand_sessions.len() >= MAX_CAND_SESSIONS {
            break;
        }
        let si = ctx.ix.turns[ti].session;
        if seen_s.insert(si) {
            cand_sessions.push(si);
        }
    }
    if cand_sessions.is_empty() {
        return;
    }

    if ctx.has_named_speakers_eff {
        // Named-speaker dialogue path:
        // EMGR speaker-gated attribution + t +/- 1 coupling + QUEST DPBF / xQuAD + Cholesky DPP.
        let mut emgr_scored = compute_emgr_turn_scores(
            ctx.ix,
            &ctx.q.text,
            &ctx.plan,
            &ctx.t_scene,
            &cand_sessions,
        );
        emgr_scored.retain(|&(ti, sc)| {
            sc > 0
                && !turn_is_superseded(ctx.ix, ti)
                && !is_discourse_closure_or_meta_echo(ctx.ix, ti, &ctx.q.text)
        });
        if emgr_scored.len() < 2 {
            return;
        }

        // Always preserve a declarative Slot 0 (protects R@1, star_bit, and upstream open-domain locks).
        let lock_slot0 = if h0_declarative { Some(h0) } else { None };

        let mut selected = if multi_hop || (ctx.q_group_cnt >= 2 && !h0_full_cov) {
            let steiner_turns =
                extract_quest_steiner_turns(ctx.ix, &ctx.plan, &emgr_scored, &cand_sessions);
            if !steiner_turns.is_empty() {
                let steiner_set: BTreeSet<usize> = steiner_turns.into_iter().collect();
                for (ti, sc) in &mut emgr_scored {
                    if steiner_set.contains(ti) && is_declarative_evidence(ctx.ix, *ti, &ctx.q.text)
                    {
                        *sc = sc.saturating_add(STEINER_BOOST_Q16);
                    }
                }
                emgr_scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            }
            let xquad_pool =
                select_xquad_turns_q16(ctx.ix, &ctx.plan, &emgr_scored, top_n.saturating_mul(2), lock_slot0);
            let score_lookup: BTreeMap<usize, i64> = emgr_scored.iter().copied().collect();
            let dpp_in: Vec<(usize, i64)> = xquad_pool
                .into_iter()
                .map(|ti| (ti, score_lookup.get(&ti).copied().unwrap_or(0)))
                .collect();
            select_cholesky_dpp(ctx.ix, &dpp_in, top_n)
        } else if counting {
            let dpp_filtered = select_cholesky_dpp(ctx.ix, &emgr_scored, top_n.saturating_mul(3));
            let score_lookup: BTreeMap<usize, i64> = emgr_scored.iter().copied().collect();
            let mut by_sess: BTreeMap<usize, Vec<(usize, i64)>> = BTreeMap::new();
            for ti in dpp_filtered {
                let si = ctx.ix.turns[ti].session;
                by_sess
                    .entry(si)
                    .or_default()
                    .push((ti, score_lookup.get(&ti).copied().unwrap_or(0)));
            }
            let sess_vec: Vec<(usize, Vec<(usize, i64)>)> = by_sess.into_iter().collect();
            select_pm2_sainte_lague(&sess_vec, top_n)
        } else {
            select_cholesky_dpp(ctx.ix, &emgr_scored, top_n)
        };

        // Enforce star_bit & declarative Slot 0 invariant:
        // Never allow a turn missing `star_id` or a question/backchannel turn to displace a
        // declarative Top-10 turn carrying `star_id` at Slot 0.
        if let Some(locked_ti) = lock_slot0 {
            selected.retain(|&ti| ti != locked_ti);
            selected.insert(0, locked_ti);
            selected.truncate(top_n);
        } else if let Some(&new_head) = selected.first() {
            let new_head_decl = is_declarative_evidence(ctx.ix, new_head, &ctx.q.text);
            let new_head_star = star_id.is_none_or(|sid| turn_has_term(ctx.ix, new_head, sid));
            if (!new_head_decl || !new_head_star) && h0_declarative && h0_has_star {
                selected.retain(|&ti| ti != h0);
                selected.insert(0, h0);
                selected.truncate(top_n);
            }
        }

        splice_top_turns(&mut ctx.final_turn_indices, &selected, top_n);
        if multi_hop || counting {
            sync_tail_sessions_preserving_slot0(
                ctx.ix,
                &ctx.final_turn_indices,
                &mut ctx.final_session_indices,
                top_n,
            );
        }
    } else if counting || multi_hop {
        // Anonymous multi-session counting / multi-hop path:
        // Cholesky MAP DPP deduplication + PM-2 Sainte-Laguë odd-divisor session spread,
        // strictly preserving declarative Slot 0 for R@1 safety.
        let mut emgr_scored = compute_emgr_turn_scores(
            ctx.ix,
            &ctx.q.text,
            &ctx.plan,
            &ctx.t_scene,
            &cand_sessions,
        );
        emgr_scored.retain(|&(ti, sc)| {
            sc > 0
                && !turn_is_superseded(ctx.ix, ti)
                && !is_discourse_closure_or_meta_echo(ctx.ix, ti, &ctx.q.text)
        });
        if emgr_scored.len() < 2 {
            return;
        }

        if multi_hop && !counting {
            let steiner_turns =
                extract_quest_steiner_turns(ctx.ix, &ctx.plan, &emgr_scored, &cand_sessions);
            if !steiner_turns.is_empty() {
                let steiner_set: BTreeSet<usize> = steiner_turns.into_iter().collect();
                for (ti, sc) in &mut emgr_scored {
                    if steiner_set.contains(ti) && is_declarative_evidence(ctx.ix, *ti, &ctx.q.text)
                    {
                        *sc = sc.saturating_add(STEINER_BOOST_Q16);
                    }
                }
                emgr_scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            }
        }

        // First deduplicate near-clone turns via Cholesky DPP, then apportion distinct
        // session seats via PM-2 Sainte-Laguë odd divisors (1, 3, 5, ...).
        let dpp_pool = select_cholesky_dpp(ctx.ix, &emgr_scored, top_n.saturating_mul(4));
        let score_lookup: BTreeMap<usize, i64> = emgr_scored.iter().copied().collect();
        let mut by_sess: BTreeMap<usize, Vec<(usize, i64)>> = BTreeMap::new();
        for ti in dpp_pool {
            if !is_declarative_evidence(ctx.ix, ti, &ctx.q.text) {
                continue;
            }
            let si = ctx.ix.turns[ti].session;
            by_sess
                .entry(si)
                .or_default()
                .push((ti, score_lookup.get(&ti).copied().unwrap_or(0)));
        }
        if by_sess.len() < 2 {
            return;
        }

        let sess_vec: Vec<(usize, Vec<(usize, i64)>)> = by_sess.into_iter().collect();
        let mut selected = select_pm2_sainte_lague(&sess_vec, top_n);
        if selected.is_empty() {
            return;
        }

        // Preserve Slot 0 on declarative heads so R@1 never regresses on multi-session queries.
        if h0_declarative {
            selected.retain(|&ti| ti != h0);
            selected.insert(0, h0);
            selected.truncate(top_n);
        }

        splice_top_turns(&mut ctx.final_turn_indices, &selected, top_n);
        sync_tail_sessions_preserving_slot0(
            ctx.ix,
            &ctx.final_turn_indices,
            &mut ctx.final_session_indices,
            top_n,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Query, Session, Turn, build_index, plan_query};

    fn mk_session(id: &str, ts: i64, pairs: &[(&str, &str)]) -> Session {
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

    #[test]
    fn emgr_reaction_dampening_and_adjacency_coupling_beats_partner_reaction() {
        let sessions = vec![mk_session(
            "s0",
            1_000,
            &[
                ("Caroline", "Melanie, what did you paint at the lake?"),
                ("Melanie", "A watercolor sunset over the water."),
                ("Caroline", "Wow Melanie, so cool!"),
            ],
        )];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "What did Melanie paint at the lake?".to_string(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &p);
        // Simulate raw BM25 giving the partner's prompt question & reaction strong lexical overlap.
        let t_scene = vec![1.0f32, 0.72f32, 0.65f32];
        let scores = compute_emgr_turn_scores(&ix, &q.text, &plan, &t_scene, &[0]);

        assert_eq!(scores.len(), 3);
        let score_of = |ti: usize| scores.iter().find(|&&(t, _)| t == ti).unwrap().1;
        assert!(
            score_of(1) > score_of(0),
            "Melanie's declarative reply (turn 1) must beat Caroline's question prompt (turn 0) via adjacency transfer: {scores:?}"
        );
        assert!(
            score_of(1) > score_of(2),
            "Melanie's declarative reply (turn 1) must beat Caroline's zero-content reaction (turn 2): {scores:?}"
        );
    }

    #[test]
    fn pm2_sainte_lague_spreads_five_slots_across_five_distinct_sessions_preserving_slot0() {
        let session_turns: Vec<(usize, Vec<(usize, i64)>)> = vec![
            (0, vec![(10, 100_000), (11, 95_000), (12, 90_000)]),
            (1, vec![(20, 60_000)]),
            (2, vec![(30, 55_000)]),
            (3, vec![(40, 50_000)]),
            (4, vec![(50, 45_000)]),
        ];
        let picked = select_pm2_sainte_lague(&session_turns, 5);
        assert_eq!(
            picked,
            vec![10, 20, 30, 40, 50],
            "Slot 0 must keep Session 0's #1 turn, then odd divisor 3 must spread Slots 1..4 across Sessions 1..4"
        );
    }

    #[test]
    fn quest_dpbf_group_steiner_connects_two_hops_via_shared_bridge_entity() {
        let sessions = vec![
            mk_session("s_hop1", 100, &[("user", "Elena founded Helios in Zurich.")]),
            mk_session(
                "s_hop2",
                200,
                &[("user", "Helios acquired the Quantum patent in Geneva.")],
            ),
            mk_session(
                "s_noise",
                300,
                &[("user", "We bought apples and bread at the market.")],
            ),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "Which patent did Elena's company acquire?".to_string(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &p);
        let emgr_scores = vec![(0, ONE), (1, (ONE * 4) / 5), (2, ONE / 10)];
        let steiner = extract_quest_steiner_turns(&ix, &plan, &emgr_scores, &[0, 1, 2]);

        assert!(
            steiner.contains(&0) && steiner.contains(&1),
            "QUEST DPBF must connect Hop 1 (turn 0) and Hop 2 (turn 1) through bridge entity 'helios': {steiner:?}"
        );
        assert!(
            !steiner.contains(&2),
            "Unrelated distractor turn 2 must not be selected by the minimum-cost Steiner tree: {steiner:?}"
        );
    }

    #[test]
    fn cholesky_dpp_suppresses_same_session_duplicate_for_orthogonal_evidence() {
        let sessions = vec![
            mk_session(
                "s0",
                100,
                &[
                    ("user", "I visited Tokyo for the robotics conference and ate ramen."),
                    ("user", "The Tokyo robotics conference ramen dinner was memorable."),
                ],
            ),
            mk_session(
                "s1",
                200,
                &[("user", "I attended a ceramics workshop in Lisbon in autumn.")],
            ),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        // Turn 1 is a near-duplicate of Turn 0 in Session 0 with high score; Turn 2 is orthogonal in Session 1.
        let cands = vec![(0, 100_000), (1, 95_000), (2, 70_000)];
        let picked = select_cholesky_dpp(&ix, &cands, 2);

        assert_eq!(
            picked,
            vec![0, 2],
            "Cholesky MAP DPP must preserve Slot 0 (turn 0) and prefer orthogonal Session 1 (turn 2) over same-session duplicate (turn 1)"
        );
    }

    #[test]
    fn xquad_preserves_slot0_and_covers_uncovered_second_hop_aspect() {
        let sessions = vec![
            mk_session(
                "s0",
                100,
                &[
                    ("user", "Marcus built the observatory telescope lens."),
                    ("user", "Marcus polished the observatory telescope mirror."),
                ],
            ),
            mk_session(
                "s1",
                200,
                &[("user", "The comet perihelion catalog was archived in Oslo.")],
            ),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "Where was Marcus's comet catalog archived?".to_string(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &p);
        let cands = vec![(0, 100_000), (1, 90_000), (2, 65_000)];
        let picked = select_xquad_turns_q16(&ix, &plan, &cands, 2, Some(0));

        assert_eq!(
            picked,
            vec![0, 2],
            "xQuAD must lock Slot 0 (turn 0) and cover uncovered aspect (turn 2) at Slot 1"
        );
    }

    #[test]
    fn emgr_steiner_dpp_handles_multibyte_utf8_safely() {
        let sessions = vec![
            mk_session(
                "s_utf8_0",
                100,
                &[
                    ("Zoë", "André brought café résumé notes from 東京 🚀."),
                    ("André", "Oui Zoë, très naïve idée!"),
                ],
            ),
            mk_session(
                "s_utf8_1",
                200,
                &[("Zoë", "The 東京 exhibition featured ceramic sculptures and café art.")],
            ),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "What café exhibition did Zoë mention from 東京?".to_string(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &p);
        let t_scene = vec![1.2f32, 0.4f32, 1.0f32];

        let emgr = compute_emgr_turn_scores(&ix, &q.text, &plan, &t_scene, &[0, 1]);
        assert_eq!(emgr.len(), 3);

        let steiner = extract_quest_steiner_turns(&ix, &plan, &emgr, &[0, 1]);
        assert!(steiner.iter().all(|&ti| ti < ix.turns.len()));

        let dpp = select_cholesky_dpp(&ix, &emgr, 2);
        assert_eq!(dpp.len(), 2);

        let pm2 = select_pm2_sainte_lague(&[(0, vec![emgr[0]]), (1, vec![emgr[1]])], 2);
        assert_eq!(pm2.len(), 2);
    }
}
