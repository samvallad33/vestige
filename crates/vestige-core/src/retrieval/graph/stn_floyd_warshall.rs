//! Tier C Engine #4 — Dechter Simple Temporal Network (STN) Metric Floyd-Warshall.
//!
//! Compiles integer day constraints `L_ij <= T_j - T_i <= U_ij` over `M <= 24` turns
//! plus virtual origin `0`, executes exact `25^3` Floyd-Warshall (`< 4 us`), prunes
//! negative-cycle contradictions, and co-seats metric endpoint pairs on interval queries.

use crate::retrieval::index::Index;
use crate::retrieval::query::tokens::raw_tokens;
use crate::retrieval::QueryIntent;

const MAX_STN_NODES: usize = 24;
const SECS_PER_DAY: i64 = 86_400;
const STN_INF: i64 = i64::MAX / 8;

#[derive(Clone, Debug)]
pub(in crate::retrieval) struct StnSolution {
    pub(in crate::retrieval) consistent: Vec<bool>,
    pub(in crate::retrieval) endpoint_pair: Option<(usize, usize, i64)>,
}

/// hardcode-scan: closed-class (cardinal number words only).
fn parse_metric_qty(tok: &str) -> Option<i64> {
    if let Ok(v) = tok.parse::<i64>() {
        return (1..=3650).contains(&v).then_some(v);
    }
    match tok {
        "a" | "an" | "one" => Some(1),
        "two" | "couple" => Some(2),
        "three" | "few" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        "fortnight" => Some(14),
        _ => None,
    }
}

/// hardcode-scan: closed-class (temporal units and deictic direction prepositions only).
pub(in crate::retrieval) fn extract_relative_day_offset(tokens: &[String]) -> Option<i64> {
    for (i, tok) in tokens.iter().enumerate() {
        let w = tok.as_str();
        if w == "yesterday" {
            return Some(-1);
        }
        if w == "tomorrow" {
            return Some(1);
        }
        if w == "day" && i >= 1 && tokens[i - 1] == "next" {
            return Some(1);
        }
        if w == "day" && i + 1 < tokens.len() && tokens[i + 1] == "before" {
            return Some(-1);
        }
        let unit_days: Option<i64> = match w {
            "day" | "days" => Some(1),
            "week" | "weeks" => Some(7),
            "month" | "months" => Some(30),
            "year" | "years" => Some(365),
            _ => None,
        };
        if let Some(ud) = unit_days
            && i >= 1
            && i + 1 < tokens.len()
            && let Some(qty) = parse_metric_qty(&tokens[i - 1])
        {
            let dir = match tokens[i + 1].as_str() {
                "before" | "ago" | "earlier" | "prior" => Some(-1i64),
                "after" | "later" | "following" => Some(1i64),
                _ => None,
            };
            if let Some(sign) = dir {
                return Some(sign * qty * ud);
            }
        }
    }
    None
}

/// hardcode-scan: closed-class (metric temporal WH-interrogatives and interval prepositions only).
pub(in crate::retrieval) fn is_metric_interval_query(q_text: &str) -> bool {
    let ql = q_text.to_ascii_lowercase();
    let asks_span = ql.starts_with("how many days")
        || ql.starts_with("how many weeks")
        || ql.starts_with("how many months")
        || ql.starts_with("how many years")
        || ql.starts_with("how long ");
    asks_span
        && (ql.contains(" between ")
            || ql.contains(" before ")
            || ql.contains(" after ")
            || ql.contains(" passed ")
            || ql.contains(" elapsed "))
}

pub(in crate::retrieval) fn solve_stn_floyd_warshall(
    ix: &Index,
    q_text: &str,
    cand_turns: &[usize],
    t_exact_mask: &[u64],
) -> StnSolution {
    let m = cand_turns.len().min(MAX_STN_NODES);
    let n = m + 1;
    let mut dist = vec![vec![STN_INF; n]; n];
    for (i, row) in dist.iter_mut().enumerate() {
        row[i] = 0;
    }

    let add_bound = |d: &mut [Vec<i64>], u: usize, v: usize, low: i64, high: i64| {
        if high < d[u][v] {
            d[u][v] = high;
        }
        let neg_low = low.saturating_neg();
        if neg_low < d[v][u] {
            d[v][u] = neg_low;
        }
    };

    let ns = ix.session_ts.len();
    let mut eff_ts = vec![0i64; ns];
    let mut running_ts = 0i64;
    for si in 0..ns {
        if ix.session_ts[si] > 0 {
            running_ts = ix.session_ts[si];
        }
        eff_ts[si] = running_ts;
    }

    let mut node_has_abs = vec![false; m];
    for (i, &ti) in cand_turns.iter().take(m).enumerate() {
        let tr = &ix.turns[ti];
        let v = i + 1;
        if let Some(&(iv_s, iv_e)) = tr.resolved_timex.first() {
            let lo_d = iv_s.div_euclid(SECS_PER_DAY);
            let hi_d = iv_e.div_euclid(SECS_PER_DAY).max(lo_d);
            add_bound(&mut dist, 0, v, lo_d, hi_d);
            node_has_abs[i] = true;
        } else {
            let sts = eff_ts.get(tr.session).copied().unwrap_or(0);
            if sts > 0 {
                let day = sts.div_euclid(SECS_PER_DAY);
                let off_d = extract_relative_day_offset(&raw_tokens(&tr.body_for_headers))
                    .unwrap_or(0);
                let tol = if off_d == 0 {
                    0
                } else if off_d.abs() >= 30 {
                    3
                } else {
                    1
                };
                add_bound(&mut dist, 0, v, day + off_d - tol, day + off_d + tol);
                node_has_abs[i] = true;
            }
        }
    }

    let offsets: Vec<Option<i64>> = cand_turns
        .iter()
        .take(m)
        .map(|&ti| extract_relative_day_offset(&raw_tokens(&ix.turns[ti].body_for_headers)))
        .collect();
    for i in 0..m {
        let Some(off_d) = offsets[i] else { continue };
        if node_has_abs[i] {
            continue;
        }
        let ti = cand_turns[i];
        let tr_i = &ix.turns[ti];
        // Anchor only to the single closest preceding same-session dated turn so
        // unrelated cross-session entity hubs never create spurious negative cycles.
        if let Some((j, _)) = cand_turns
            .iter()
            .take(m)
            .enumerate()
            .filter(|&(j, &tj)| {
                j != i
                    && node_has_abs[j]
                    && ix.turns[tj].session == tr_i.session
                    && ti.abs_diff(tj) <= 2
            })
            .min_by_key(|&(_, &tj)| ti.abs_diff(tj))
        {
            let tol = if off_d.abs() >= 30 { 3 } else { 1 };
            add_bound(&mut dist, j + 1, i + 1, off_d - tol, off_d + tol);
        }
    }

    for k in 0..n {
        for i in 0..n {
            let dik = dist[i][k];
            if dik >= STN_INF {
                continue;
            }
            for j in 0..n {
                let dkj = dist[k][j];
                if dkj >= STN_INF {
                    continue;
                }
                let cand = dik.saturating_add(dkj);
                if cand < dist[i][j] {
                    dist[i][j] = cand;
                }
            }
        }
    }

    let consistent: Vec<bool> = (0..m).map(|i| dist[i + 1][i + 1] >= 0).collect();

    let mut endpoint_pair: Option<(usize, usize, i64)> = None;
    if is_metric_interval_query(q_text) && m >= 2 {
        let mut best_key: Option<(u32, i64, usize, usize, i64)> = None;
        for a in 0..m {
            if !consistent[a] {
                continue;
            }
            let ta = cand_turns[a];
            let ma = t_exact_mask.get(ta).copied().unwrap_or(0);
            if ma == 0 || ix.turns[ta].is_question || ix.turns[ta].is_backchannel {
                continue;
            }
            for b in (a + 1)..m {
                if !consistent[b] {
                    continue;
                }
                let tb = cand_turns[b];
                if ix.turns[ta].session == ix.turns[tb].session
                    || ix.turns[tb].is_question
                    || ix.turns[tb].is_backchannel
                {
                    continue;
                }
                let mb = t_exact_mask.get(tb).copied().unwrap_or(0);
                if (ma & !mb) == 0 || (mb & !ma) == 0 {
                    continue;
                }
                let u_ab = dist[a + 1][b + 1];
                let u_ba = dist[b + 1][a + 1];
                if u_ab >= STN_INF / 2 || u_ba >= STN_INF / 2 {
                    continue;
                }
                let width = u_ab.saturating_add(u_ba).max(0);
                if width > 7 {
                    continue;
                }
                let span_days = u_ab.abs().max(u_ba.abs());
                let key = ((ma | mb).count_ones(), -width, a, b, span_days);
                if best_key.is_none_or(|bk| (key.0, key.1) > (bk.0, bk.1)) {
                    best_key = Some(key);
                }
            }
        }
        if let Some((_, _, a, b, span_days)) = best_key {
            endpoint_pair = Some((cand_turns[a], cand_turns[b], span_days));
        }
    }

    StnSolution { consistent, endpoint_pair }
}

pub(in crate::retrieval) fn apply_stn_temporal_seating(
    ix: &Index,
    q_text: &str,
    eff_it: &QueryIntent,
    final_turn_indices: &mut Vec<usize>,
    final_session_indices: &mut Vec<usize>,
    t_exact_mask: &[u64],
    k: usize,
) {
    if !(eff_it.temporal || eff_it.sniper_how_many) || final_turn_indices.len() < 2 {
        return;
    }
    let window = final_turn_indices.len().min(MAX_STN_NODES);
    let snap: Vec<usize> = final_turn_indices[..window].to_vec();
    let sol = solve_stn_floyd_warshall(ix, q_text, &snap, t_exact_mask);

    if sol.consistent.iter().any(|&ok| !ok) {
        let mut good: Vec<usize> = Vec::with_capacity(final_turn_indices.len());
        let mut bad: Vec<usize> = Vec::new();
        for (i, &ti) in snap.iter().enumerate() {
            if sol.consistent[i] {
                good.push(ti);
            } else {
                bad.push(ti);
            }
        }
        good.extend_from_slice(&final_turn_indices[window..]);
        good.extend(bad);
        good.truncate(k.max(5));
        *final_turn_indices = good;
    }

    if let Some((ti_a, ti_b, _)) = sol.endpoint_pair {
        // Never evict Slot 0 (`head0`); co-seat the complementary metric endpoint(s) into Slots 1..2.
        let head0 = final_turn_indices[0];
        let targets: Vec<(usize, usize)> = if head0 == ti_a {
            vec![(1usize, ti_b)]
        } else if head0 == ti_b {
            vec![(1usize, ti_a)]
        } else {
            vec![(1usize, ti_a), (2usize, ti_b)]
        };
        for &(slot, target_ti) in &targets {
            if let Some(pos) = final_turn_indices.iter().position(|&x| x == target_ti) {
                final_turn_indices.remove(pos);
            }
            final_turn_indices.insert(slot.min(final_turn_indices.len()), target_ti);
        }
        final_turn_indices.truncate(k.max(5));
        let head_sess = final_session_indices.first().copied();
        for &(slot, target_ti) in &targets {
            let sess = ix.turns[target_ti].session;
            if Some(sess) == head_sess {
                continue;
            }
            if let Some(spos) = final_session_indices.iter().position(|&s| s == sess) {
                final_session_indices.remove(spos);
            }
            final_session_indices.insert(slot.min(final_session_indices.len()), sess);
        }
        final_session_indices.truncate(k.max(5));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Session, Turn};

    #[test]
    fn stn_extracts_relative_day_offsets_and_metric_queries() {
        assert_eq!(extract_relative_day_offset(&raw_tokens("went yesterday")), Some(-1));
        assert_eq!(extract_relative_day_offset(&raw_tokens("three weeks ago")), Some(-21));
        assert_eq!(extract_relative_day_offset(&raw_tokens("2 days later")), Some(2));
        assert!(is_metric_interval_query(
            "How many days passed between Alice's trip and Bob's concert?"
        ));
        assert!(!is_metric_interval_query("When did Alice visit Paris?"));
    }

    #[test]
    fn stn_floyd_warshall_co_seats_complementary_metric_endpoints() {
        let sessions = vec![
            Session {
                id: "s1".into(),
                date: "2023-05-01".into(),
                timestamp: 0,
                turns: vec![Turn {
                    id: "t1".into(),
                    role: "Alice".into(),
                    content: "I finished the marathon today.".into(),
                    ..Default::default()
                }],
            },
            Session {
                id: "s2".into(),
                date: "2023-05-15".into(),
                timestamp: 0,
                turns: vec![Turn {
                    id: "t2".into(),
                    role: "Alice".into(),
                    content: "I attended the symphony concert.".into(),
                    ..Default::default()
                }],
            },
        ];
        let ix = crate::retrieval::index::build_index(&sessions, &Params::default());
        let masks = vec![0b01u64, 0b10u64];
        let sol = solve_stn_floyd_warshall(
            &ix,
            "How many days passed between the marathon and the symphony concert?",
            &[0, 1],
            &masks,
        );
        assert_eq!(sol.consistent, vec![true, true]);
        assert_eq!(sol.endpoint_pair, Some((0, 1, 14)));
    }
}
