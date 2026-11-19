//! Zero-float `Q16.16` integer session consensus voter combining:
//! - Shtok, Kurland & Carmel (ACM TOIS 2012) Normalized Query Commitment (`NQC`) dispersion gate,
//! - Metzler & Croft (SIGIR 2005) Sequential Dependence Model (`#1` ordered + `#uw8` unordered cliques),
//! - Lv & Zhai (SIGIR 2009) Positional Language Model (`PLM`) triangular proximity kernel,
//! - Tao & Zhai (SIGIR 2007) IC-weighted Minimum-Cover Span (`MinCover`),
//! - Dwork, Kumar, Naor & Sivakumar (WWW 2001) exact Kemeny-Young rank aggregation (`5! = 120`).

use crate::retrieval::*;

/// Unit `1.0` in unsigned `Q16.16` fixed-point representation (`2^16`).
const Q16_ONE: u64 = 65_536;
/// Fractional shift width (`16` bits) for `Q16.16` fixed-point arithmetic.
const Q16_SHIFT: u32 = 16;
/// IEEE Std 754-2019 binary32 exponent bias (`127`).
const IEEE754_F32_EXP_BIAS: i32 = 127;
/// IEEE Std 754-2019 binary32 fraction bits (`23`) minus `Q16.16` fraction bits (`16`).
const IEEE754_MANTISSA_TO_Q16_SHIFT: i32 = 7;
/// IEEE Std 754-2019 binary32 total significand width (`23` explicit + `1` hidden bit).
const IEEE754_SIGNIFICAND_BITS: u32 = 24;
/// IEEE Std 754-2019 binary32 hidden leading significand bit (`1 << 23`).
const IEEE754_HIDDEN_BIT: u32 = 0x0080_0000;
/// IEEE Std 754-2019 binary32 fraction mask (`(1 << 23) - 1`).
const IEEE754_FRAC_MASK: u32 = 0x007F_FFFF;
/// IEEE Std 754-2019 binary32 8-bit exponent mask (`0xFF`).
const IEEE754_EXP_MASK: u32 = 0xFF;
/// IEEE Std 754-2019 binary32 sign bit shift (`31`).
const IEEE754_SIGN_SHIFT: u32 = 31;
/// IEEE Std 754-2019 binary32 exponent bit shift (`23`).
const IEEE754_EXP_SHIFT: u32 = 23;
/// Maximum left shift (`64 - 24 = 40`) before a 24-bit significand overflows `u64`.
const Q16_SATURATE_LEFT_SHIFT: i32 = 40;
/// Laplace document-frequency smoothing constant (`+1`, Robertson & Spärck Jones 1976).
const LAPLACE_DOC_SMOOTH_ONE: u64 = 1;
/// Bit width (`64`) of the query-group coverage bitmask (`u64`).
const MAX_QUERY_GROUP_BITS: usize = 64;
/// Top-4 fractional bit shift (`16 - 4 = 12`) into `LOG2_MANTISSA_LUT_Q16`.
const LOG2_LUT_INDEX_SHIFT: u32 = 12;
/// Remainder mask (`(1 << 12) - 1`) for linear interpolation between `log2` LUT knots.
const LOG2_LUT_REM_MASK: u64 = 0x0FFF;
/// Exact `round(65536 * log2(1 + m/16))` knots for `m in 0..=16` (Shannon 1948; max err 44 Q16 units).
const LOG2_MANTISSA_LUT_Q16: [u64; 17] = [
    0, 5_732, 11_136, 16_248, 21_098, 25_711, 30_109, 34_312,
    38_336, 42_196, 45_904, 49_472, 52_911, 56_229, 59_434, 62_534, 65_536,
];

/// Top-K session horizon (`5`) for exact `5! = 120` Kemeny-Young consensus (Dwork et al. WWW 2001).
const KEMENY_CANDIDATE_POOL_CAP: usize = 5;
/// Minimum session pool length (`2`) required for pairwise rank aggregation.
const MIN_SESSIONS_FOR_CONSENSUS: usize = 2;
/// Shtok et al. (ACM TOIS 2012) squared NQC dispersion threshold (`0.30 * 65536 = 19661`, `CV >= 0.55`).
const NQC_HIGH_DISPERSION_SQ_Q16: u64 = 19_661;
/// Top-1 over Top-2 dominance numerator (`4`, i.e. `4:3 = 1.33x` crisp peak guard for Single-Hop/Adversarial).
const NQC_DOMINANCE_NUM: u64 = 4;
/// Top-1 over Top-2 dominance denominator (`3`, paired with `NQC_DOMINANCE_NUM`).
const NQC_DOMINANCE_DEN: u64 = 3;

/// Metzler & Croft (SIGIR 2005) maximum content-token gap (`2`) for ordered adjacent cliques (`#1`).
const MRF_ORDERED_GAP_MAX: usize = 2;
/// Metzler & Croft (SIGIR 2005 §4.2) unordered window width (`8` tokens, `#uw8` optimal setting).
const MRF_UNORDERED_WINDOW_TOKENS: usize = 8;
/// Lv & Zhai (SIGIR 2009) triangular positional kernel radius (`16` content tokens).
const PLM_KERNEL_RADIUS_TOKENS: u64 = 16;
/// Turn-boundary virtual token stride (`4` tokens) preserving adjacency-pair proximity across speaker turns.
const PLM_TURN_BOUNDARY_GAP_TOKENS: usize = 4;

/// Voter 1 weight (`30`): whole-session + window lexical sight (Metzler & Croft 2005 unigram backbone).
const VOTER_WEIGHT_SESSION_SIGHT: u32 = 30;
/// Voter 2 weight (`25`): MRF (`#1` + `#uw8`) + PLM + Tao-Zhai `MinCover` span density.
const VOTER_WEIGHT_MRF_PLM_SPAN: u32 = 25;
/// Voter 3 weight (`20`): Fang-Zhai `TDC` + Raaijmakers-Shiffrin `SAM` exact query-IC coverage.
const VOTER_WEIGHT_EXACT_IC_COV: u32 = 20;
/// Voter 4 weight (`15`): Two-turn support (`top1 + top2` scene Q16), defeating 1-turn `C3`/`C6` decoys.
const VOTER_WEIGHT_TWO_TURN_SUPPORT: u32 = 15;
/// Voter 5 weight (`10`): Seated top-5 turn multiplicity from `final_turn_indices[0..5]`.
const VOTER_WEIGHT_TOP5_TURN_SEATS: u32 = 10;
/// Pairwise significance numerator (`17`, i.e. `17:16 = +6.25%` Weber-fraction anti-jitter margin).
const KEMENY_MARGIN_SIG_NUM: u64 = 17;
/// Pairwise significance denominator (`16`, paired with `KEMENY_MARGIN_SIG_NUM`).
const KEMENY_MARGIN_SIG_DEN: u64 = 16;

/// Pure integer IEEE-754 binary32 bit-pattern to `Q16.16` conversion (zero FP instructions).
#[inline]
pub(in crate::retrieval) fn ieee754_bits_to_q16(bits: u32) -> u64 {
    let sign = bits >> IEEE754_SIGN_SHIFT;
    let exp_raw = (bits >> IEEE754_EXP_SHIFT) & IEEE754_EXP_MASK;
    let frac = bits & IEEE754_FRAC_MASK;
    if sign != 0 || exp_raw == 0 {
        return 0;
    }
    if exp_raw == IEEE754_EXP_MASK {
        return if frac == 0 { u64::MAX } else { 0 };
    }
    let mant24 = (frac | IEEE754_HIDDEN_BIT) as u64;
    let shift = (exp_raw as i32) - (IEEE754_F32_EXP_BIAS + IEEE754_MANTISSA_TO_Q16_SHIFT);
    if shift >= 0 {
        if shift >= Q16_SATURATE_LEFT_SHIFT {
            u64::MAX
        } else {
            mant24 << (shift as u32)
        }
    } else {
        let r = (-shift) as u32;
        if r >= IEEE754_SIGNIFICAND_BITS {
            0
        } else {
            mant24 >> r
        }
    }
}

/// Hardware `leading_zeros` + 16-segment piecewise linear `log2(x)` in `Q16.16` integers.
#[inline]
pub(in crate::retrieval) fn log2_u64_q16(x: u64) -> u64 {
    if x <= 1 {
        return 0;
    }
    let msb = (u64::BITS - 1) - x.leading_zeros();
    let int_part = (msb as u64) << Q16_SHIFT;
    let frac16 = if msb >= Q16_SHIFT {
        (x >> (msb - Q16_SHIFT)) & (Q16_ONE - 1)
    } else {
        (x << (Q16_SHIFT - msb)) & (Q16_ONE - 1)
    };
    let idx = (frac16 >> LOG2_LUT_INDEX_SHIFT) as usize;
    let rem = frac16 & LOG2_LUT_REM_MASK;
    let lo = LOG2_MANTISSA_LUT_Q16[idx];
    let hi = LOG2_MANTISSA_LUT_Q16[idx + 1];
    int_part + lo + (((hi - lo) * rem) >> LOG2_LUT_INDEX_SHIFT)
}

/// Smoothed corpus Shannon self-information `log2((N + 1) / (df + 1))` in `Q16.16`.
#[inline]
pub(in crate::retrieval) fn shannon_ic_q16(n_docs: u32, df: u32) -> u64 {
    let num = (n_docs as u64).saturating_add(LAPLACE_DOC_SMOOTH_ONE);
    let den = (df as u64).saturating_add(LAPLACE_DOC_SMOOTH_ONE);
    if num <= den {
        return 0;
    }
    log2_u64_q16(num).saturating_sub(log2_u64_q16(den))
}

/// Precomputes maximal `Q16.16` corpus self-information per query group `0..groups.len().min(64)`.
fn query_group_ic_q16(ix: &Index, plan: &QueryPlan) -> Vec<u64> {
    let g_cnt = plan.groups.len().min(MAX_QUERY_GROUP_BITS);
    let mut g_ic = vec![Q16_ONE; g_cnt];
    let n_turns = (ix.turns.len() as u32).max(1);
    for qt in &plan.terms {
        if let Some(g) = qt.group
            && g < g_cnt
        {
            let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0);
            let ic = shannon_ic_q16(n_turns, df);
            if ic > g_ic[g] {
                g_ic[g] = ic;
            }
        }
    }
    g_ic
}

/// MRF/PLM/MinCover score and exact query-IC coverage for each session in `pool`.
pub(in crate::retrieval) fn pool_mrf_and_exact_ic_q16(
    ix: &Index,
    plan: &QueryPlan,
    pool: &[usize],
    s_exact_mask: &[u64],
) -> (Vec<u64>, Vec<u64>) {
    let g_ic = query_group_ic_q16(ix, plan);
    let g_cnt = g_ic.len();
    let g_mask_limit = if g_cnt == MAX_QUERY_GROUP_BITS {
        u64::MAX
    } else if g_cnt == 0 {
        0
    } else {
        (1u64 << g_cnt) - 1
    };
    let mut mrf = Vec::with_capacity(pool.len());
    let mut exact_ic = Vec::with_capacity(pool.len());
    let mut seq_buf: Vec<(usize, usize)> = Vec::new();
    for &si in pool {
        let mut em = s_exact_mask.get(si).copied().unwrap_or(0) & g_mask_limit;
        let hit_groups = em.count_ones() as u64;
        let mut ic_sum: u64 = 0;
        while em != 0 {
            let g = em.trailing_zeros() as usize;
            ic_sum = ic_sum.saturating_add(g_ic.get(g).copied().unwrap_or(Q16_ONE));
            em &= em - 1;
        }
        exact_ic.push(ic_sum.saturating_add(if hit_groups >= 2 {
            (ic_sum * (hit_groups - 1)) >> 2
        } else {
            0
        }));
        seq_buf.clear();
        let mut cursor = 0usize;
        if let Some(members) = ix.session_turns.get(si) {
            for &ti in members {
                if let Some(turn) = ix.turns.get(ti) {
                    for &(exact_id, stem_id) in &turn.seq {
                        if let Some(&g) = plan
                            .group_of
                            .get(&exact_id)
                            .or_else(|| plan.group_of.get(&stem_id))
                            && g < g_cnt
                        {
                            seq_buf.push((cursor, g));
                        }
                        cursor += 1;
                    }
                }
                cursor += PLM_TURN_BOUNDARY_GAP_TOKENS;
            }
        }
        mrf.push(score_session_mrf_plm_mincover_q16(&seq_buf, &g_ic));
    }
    (mrf, exact_ic)
}

/// Scores a session's ordered `#1` cliques, `#uw8` cliques, `PLM` kernel, and `MinCover` span in `Q16.16`.
pub(in crate::retrieval) fn score_session_mrf_plm_mincover_q16(
    seq_groups: &[(usize, usize)],
    group_ic_q16: &[u64],
) -> u64 {
    if seq_groups.len() < 2 || group_ic_q16.len() < 2 {
        return 0;
    }
    let mut ordered_q16: u64 = 0;
    let mut unordered_q16: u64 = 0;
    for i in 0..seq_groups.len() {
        let (pos_i, g_i) = seq_groups[i];
        let ic_i = group_ic_q16.get(g_i).copied().unwrap_or(Q16_ONE);
        let mut seen_right: u64 = 1u64 << g_i;
        for &(pos_j, g_j) in &seq_groups[(i + 1)..] {
            let dist = pos_j.saturating_sub(pos_i);
            if dist == 0 {
                continue;
            }
            if dist > MRF_UNORDERED_WINDOW_TOKENS {
                break;
            }
            if (seen_right & (1u64 << g_j)) != 0 {
                continue;
            }
            seen_right |= 1u64 << g_j;
            let ic_j = group_ic_q16.get(g_j).copied().unwrap_or(Q16_ONE);
            let pair_ic = (ic_i.saturating_add(ic_j)) >> 1;
            if g_j == g_i + 1 && dist <= MRF_ORDERED_GAP_MAX {
                ordered_q16 = ordered_q16.saturating_add(pair_ic);
            }
            let kernel_num = (MRF_UNORDERED_WINDOW_TOKENS + 1 - dist) as u64;
            let kernel_den = (MRF_UNORDERED_WINDOW_TOKENS + 1) as u64;
            unordered_q16 = unordered_q16.saturating_add((pair_ic * kernel_num) / kernel_den);
        }
    }

    // Tao & Zhai (2007) + Lv & Zhai (2009) Minimum-Cover Span over distinct query groups:
    let mut counts = [0u32; MAX_QUERY_GROUP_BITS];
    let mut active_mask: u64 = 0;
    let mut best_cov_bits: u32 = 0;
    let mut best_span_score_q16: u64 = 0;
    let mut left = 0usize;
    for right in 0..seq_groups.len() {
        let (_, g_r) = seq_groups[right];
        if counts[g_r] == 0 {
            active_mask |= 1u64 << g_r;
        }
        counts[g_r] += 1;
        while left < right {
            let (_, g_l) = seq_groups[left];
            if counts[g_l] > 1 {
                counts[g_l] -= 1;
                left += 1;
            } else {
                break;
            }
        }
        let cov_bits = active_mask.count_ones();
        if cov_bits >= 2 {
            let span_len = (seq_groups[right].0.saturating_sub(seq_groups[left].0) + 1) as u64;
            let slack = span_len.saturating_sub(cov_bits as u64);
            let mut cov_ic_sum: u64 = 0;
            let mut m = active_mask;
            while m != 0 {
                let g = m.trailing_zeros() as usize;
                cov_ic_sum = cov_ic_sum.saturating_add(group_ic_q16.get(g).copied().unwrap_or(Q16_ONE));
                m &= m - 1;
            }
            let plm_span_q16 = (cov_ic_sum.saturating_mul(PLM_KERNEL_RADIUS_TOKENS))
                / PLM_KERNEL_RADIUS_TOKENS.saturating_add(slack);
            if cov_bits > best_cov_bits
                || (cov_bits == best_cov_bits && plm_span_q16 > best_span_score_q16)
            {
                best_cov_bits = cov_bits;
                best_span_score_q16 = plm_span_q16;
            }
        }
    }

    ordered_q16
        .saturating_add(unordered_q16)
        .saturating_add(best_span_score_q16)
}

/// Shtok et al. (2012) integer squared-NQC confidence lock: returns `true` when `#1` has a crisp
/// high-dispersion peak AND dominates `#2` by `>= 4:3` with maximal exact query-IC coverage.
pub(in crate::retrieval) fn nqc_locks_incumbent(sight_q16: &[u64], exact_ic_q16: &[u64]) -> bool {
    if sight_q16.len() < MIN_SESSIONS_FOR_CONSENSUS || sight_q16[0] == 0 {
        return false;
    }
    let max_other_ic = exact_ic_q16[1..].iter().copied().max().unwrap_or(0);
    if exact_ic_q16[0] < max_other_ic {
        return false;
    }
    if sight_q16[0].saturating_mul(NQC_DOMINANCE_DEN)
        < sight_q16[1].saturating_mul(NQC_DOMINANCE_NUM)
    {
        return false;
    }
    let m = sight_q16.len() as u64;
    let sum: u128 = sight_q16.iter().map(|&v| v as u128).sum();
    let mu: u128 = sum / (m as u128);
    if mu == 0 {
        return false;
    }
    let mut sq_dev_sum: u128 = 0;
    for &v in sight_q16 {
        let diff = (v as u128).abs_diff(mu);
        sq_dev_sum = sq_dev_sum.saturating_add(diff.saturating_mul(diff));
    }
    let mean_var = sq_dev_sum / (m as u128);
    let denom = mu.saturating_mul(mu);
    let nqc_sq_q16 = ((mean_var << Q16_SHIFT) / denom.max(1)) as u64;
    nqc_sq_q16 >= NQC_HIGH_DISPERSION_SQ_Q16
}

fn permute_kemeny(
    depth: usize,
    m: usize,
    cur: &mut [usize; KEMENY_CANDIDATE_POOL_CAP],
    margin: &[[u32; KEMENY_CANDIDATE_POOL_CAP]; KEMENY_CANDIDATE_POOL_CAP],
    cand_sessions: &[usize],
    best_score: &mut u32,
    best_perm: &mut [usize; KEMENY_CANDIDATE_POOL_CAP],
) {
    if depth == m {
        let mut sc = 0u32;
        for i in 0..m {
            for j in (i + 1)..m {
                sc += margin[cur[i]][cur[j]];
            }
        }
        let better = if sc != *best_score {
            sc > *best_score
        } else {
            let cur_is_id = (0..m).all(|k| cur[k] == k);
            let best_is_id = (0..m).all(|k| best_perm[k] == k);
            if cur_is_id != best_is_id {
                cur_is_id
            } else {
                (0..m)
                    .find_map(|k| {
                        let sa = cand_sessions[cur[k]];
                        let sb = cand_sessions[best_perm[k]];
                        if sa != sb { Some(sa < sb) } else { None }
                    })
                    .unwrap_or(false)
            }
        };
        if better {
            *best_score = sc;
            *best_perm = *cur;
        }
        return;
    }
    for idx in depth..m {
        cur.swap(depth, idx);
        permute_kemeny(depth + 1, m, cur, margin, cand_sessions, best_score, best_perm);
        cur.swap(depth, idx);
    }
}

/// Exact `m!`-permutation Kemeny-Young solver (`m <= 5`) with deterministic identity/ID tie-break.
pub(in crate::retrieval) fn solve_kemeny_order(
    cand_sessions: &[usize],
    voters: &[[u64; KEMENY_CANDIDATE_POOL_CAP]; 5],
    weights: &[u32; 5],
    discrete_voter_mask: u8,
) -> Vec<usize> {
    let m = cand_sessions.len();
    let mut margin = [[0u32; KEMENY_CANDIDATE_POOL_CAP]; KEMENY_CANDIDATE_POOL_CAP];
    for a in 0..m {
        for b in 0..m {
            if a == b {
                continue;
            }
            let mut w_sum = 0u32;
            for v in 0..5 {
                let va = voters[v][a];
                let vb = voters[v][b];
                let is_discrete = (discrete_voter_mask & (1u8 << v)) != 0;
                let wins = if is_discrete {
                    va > vb
                } else {
                    va > vb
                        && va.saturating_mul(KEMENY_MARGIN_SIG_DEN)
                            >= vb.saturating_mul(KEMENY_MARGIN_SIG_NUM)
                };
                if wins {
                    w_sum += weights[v];
                }
            }
            margin[a][b] = w_sum;
        }
    }

    let mut cur = [0usize, 1, 2, 3, 4];
    let mut best_perm = cur;
    let mut best_score: u32 = 0;
    for i in 0..m {
        for j in (i + 1)..m {
            best_score += margin[cur[i]][cur[j]];
        }
    }
    permute_kemeny(
        0,
        m,
        &mut cur,
        &margin,
        cand_sessions,
        &mut best_score,
        &mut best_perm,
    );

    // Zero-regression head guard: never dethrone `cand[0]` unless the new head wins direct
    // pairwise majority AND matches or exceeds `cand[0]` on exact query-IC coverage (Voter 2 idx).
    let w0 = best_perm[0];
    if w0 != 0 && (margin[w0][0] <= margin[0][w0] || voters[2][w0] < voters[2][0]) {
        return cand_sessions.to_vec();
    }
    (0..m).map(|k| cand_sessions[best_perm[k]]).collect()
}

#[allow(dead_code)]
pub(in crate::retrieval) fn apply_nqc_gated_mrf_kemeny_top5(
    ix: &Index,
    plan: &QueryPlan,
    final_turn_indices: &[usize],
    final_session_indices: &mut Vec<usize>,
    t_scene: &[f32],
    s_score: &[f32],
    bridge_boost: &[f32],
    s_exact_mask: &[u64],
) {
    let m = final_session_indices.len().min(KEMENY_CANDIDATE_POOL_CAP);
    if m < MIN_SESSIONS_FOR_CONSENSUS {
        return;
    }
    let g_ic = query_group_ic_q16(ix, plan);
    let g_cnt = g_ic.len();
    let g_mask_limit = if g_cnt == MAX_QUERY_GROUP_BITS {
        u64::MAX
    } else {
        (1u64 << g_cnt) - 1
    };

    let cands: Vec<usize> = final_session_indices[..m].to_vec();
    let mut v_sight = [0u64; KEMENY_CANDIDATE_POOL_CAP];
    let mut v_mrf = [0u64; KEMENY_CANDIDATE_POOL_CAP];
    let mut v_ic_cov = [0u64; KEMENY_CANDIDATE_POOL_CAP];
    let mut v_two_turn = [0u64; KEMENY_CANDIDATE_POOL_CAP];
    let mut v_seats = [0u64; KEMENY_CANDIDATE_POOL_CAP];

    let mut seq_buf: Vec<(usize, usize)> = Vec::new();
    for (idx, &si) in cands.iter().enumerate() {
        let s_q16 = s_score
            .get(si)
            .map_or(0, |v| ieee754_bits_to_q16(v.to_bits()));
        let b_q16 = bridge_boost
            .get(si)
            .map_or(0, |v| ieee754_bits_to_q16(v.to_bits()));
        v_sight[idx] = s_q16.saturating_add(b_q16);

        let mut em = s_exact_mask.get(si).copied().unwrap_or(0) & g_mask_limit;
        let hit_groups = em.count_ones() as u64;
        let mut ic_sum: u64 = 0;
        while em != 0 {
            let g = em.trailing_zeros() as usize;
            ic_sum = ic_sum.saturating_add(g_ic[g]);
            em &= em - 1;
        }
        // Raaijmakers & Shiffrin (1981) SAM superlinear cue-intersection bonus for multi-group hits:
        v_ic_cov[idx] = ic_sum.saturating_add(if hit_groups >= 2 {
            (ic_sum * (hit_groups - 1)) >> 2
        } else {
            0
        });

        seq_buf.clear();
        let mut cursor = 0usize;
        let mut top1_t: u64 = 0;
        let mut top2_t: u64 = 0;
        if let Some(members) = ix.session_turns.get(si) {
            for &ti in members {
                let t_q16 = t_scene
                    .get(ti)
                    .map_or(0, |v| ieee754_bits_to_q16(v.to_bits()));
                if t_q16 >= top1_t {
                    top2_t = top1_t;
                    top1_t = t_q16;
                } else if t_q16 > top2_t {
                    top2_t = t_q16;
                }
                for &(exact_id, stem_id) in &ix.turns[ti].seq {
                    if let Some(&g) = plan
                        .group_of
                        .get(&exact_id)
                        .or_else(|| plan.group_of.get(&stem_id))
                        && g < g_cnt
                    {
                        seq_buf.push((cursor, g));
                    }
                    cursor += 1;
                }
                cursor += PLM_TURN_BOUNDARY_GAP_TOKENS;
            }
        }
        v_mrf[idx] = score_session_mrf_plm_mincover_q16(&seq_buf, &g_ic);
        v_two_turn[idx] = top1_t.saturating_add(top2_t);
        let seat_cnt = final_turn_indices
            .iter()
            .take(KEMENY_CANDIDATE_POOL_CAP)
            .filter(|&&ti| ix.turns[ti].session == si)
            .count() as u64;
        v_seats[idx] = seat_cnt << Q16_SHIFT;
    }

    if nqc_locks_incumbent(&v_sight[..m], &v_ic_cov[..m]) {
        return;
    }

    let voters = [v_sight, v_mrf, v_ic_cov, v_two_turn, v_seats];
    let weights = [
        VOTER_WEIGHT_SESSION_SIGHT,
        VOTER_WEIGHT_MRF_PLM_SPAN,
        VOTER_WEIGHT_EXACT_IC_COV,
        VOTER_WEIGHT_TWO_TURN_SUPPORT,
        VOTER_WEIGHT_TOP5_TURN_SEATS,
    ];
    // Voters 2 (exact IC sum) and 4 (top-5 turn seat count) compare exact discrete quantities.
    let discrete_mask: u8 = (1u8 << 2) | (1u8 << 4);
    let reordered = solve_kemeny_order(&cands, &voters, &weights, discrete_mask);
    final_session_indices[..m].copy_from_slice(&reordered);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ieee754_and_log2_q16_are_bit_exact_and_monotonic() {
        assert_eq!(ieee754_bits_to_q16(0.0f32.to_bits()), 0);
        assert_eq!(ieee754_bits_to_q16((-2.5f32).to_bits()), 0);
        assert_eq!(ieee754_bits_to_q16(f32::NAN.to_bits()), 0);
        assert_eq!(ieee754_bits_to_q16(1.0f32.to_bits()), Q16_ONE);
        assert_eq!(ieee754_bits_to_q16(2.5f32.to_bits()), 163_840);
        assert_eq!(log2_u64_q16(1), 0);
        assert_eq!(log2_u64_q16(2), Q16_ONE);
        assert_eq!(log2_u64_q16(8), 3 * Q16_ONE);
        assert!(shannon_ic_q16(100, 1) > shannon_ic_q16(100, 20));
    }

    #[test]
    fn mrf_plm_mincover_ranks_tight_ordered_phrase_above_scattered_decoy() {
        let g_ic = vec![3 * Q16_ONE, 4 * Q16_ONE, 3 * Q16_ONE];
        let tight_phrase = vec![(2, 0), (3, 1), (4, 2)];
        let scattered_decoy = vec![(2, 2), (25, 0), (60, 1)];
        let sc_tight = score_session_mrf_plm_mincover_q16(&tight_phrase, &g_ic);
        let sc_decoy = score_session_mrf_plm_mincover_q16(&scattered_decoy, &g_ic);
        assert!(sc_tight > sc_decoy * 2);
    }

    #[test]
    fn nqc_locks_sharp_single_hop_winner_and_kemeny_promotes_ambiguous_consensus() {
        let sharp_sight = [10 * Q16_ONE, 4 * Q16_ONE, 2 * Q16_ONE];
        let sharp_ic = [6 * Q16_ONE, 3 * Q16_ONE, 3 * Q16_ONE];
        assert!(nqc_locks_incumbent(&sharp_sight, &sharp_ic));

        let flat_sight = [10 * Q16_ONE, 9 * Q16_ONE, 8 * Q16_ONE, 0, 0];
        assert!(!nqc_locks_incumbent(&flat_sight[..3], &[4 * Q16_ONE, 6 * Q16_ONE, 3 * Q16_ONE]));

        let cands = vec![10usize, 22usize, 35usize];
        let voters = [
            [10 * Q16_ONE, 9 * Q16_ONE, 7 * Q16_ONE, 0, 0],
            [Q16_ONE, 8 * Q16_ONE, Q16_ONE, 0, 0],
            [4 * Q16_ONE, 7 * Q16_ONE, 3 * Q16_ONE, 0, 0],
            [5 * Q16_ONE, 9 * Q16_ONE, 4 * Q16_ONE, 0, 0],
            [Q16_ONE, 2 * Q16_ONE, 0, 0, 0],
        ];
        let weights = [
            VOTER_WEIGHT_SESSION_SIGHT,
            VOTER_WEIGHT_MRF_PLM_SPAN,
            VOTER_WEIGHT_EXACT_IC_COV,
            VOTER_WEIGHT_TWO_TURN_SUPPORT,
            VOTER_WEIGHT_TOP5_TURN_SEATS,
        ];
        let out = solve_kemeny_order(&cands, &voters, &weights, (1 << 2) | (1 << 4));
        assert_eq!(out[0], 22);
    }
}
