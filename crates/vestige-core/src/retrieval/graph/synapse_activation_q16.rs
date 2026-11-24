//! SYNAPSE (Feb 2026) Spreading Activation + Lateral Inhibition over Turn-Entity Neighborhoods.
//!
//! Operates strictly in integer Q16.16 fixed-point arithmetic over a small set of candidate
//! declarative turns (`<= 24`), reinforcing multi-facet supported turns and suppressing
//! isolated high-degree generic hub turns without touching embeddings or `t_own`.

use crate::retrieval::{DetHashMap, DetHashSet, Index};

const Q16_ONE: u64 = 65_536;

/// Propagate activation over a candidate turn subgraph (`cand_turns` = `(turn_idx, base_score_q16)`)
/// using 2 integer steps of spreading activation with lateral inhibition (`deg > 6` hub damping)
/// and habitual-aspect amplification. Returns `(turn_idx, activated_score_q16)` sorted descending.
pub(in crate::retrieval) fn synapse_activate_speaker_turns_q16(
    ix: &Index,
    cand_turns: &[(usize, u64)],
) -> Vec<(usize, u64)> {
    let n = cand_turns.len().min(24);
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![cand_turns[0]];
    }

    let mut term_post: DetHashMap<u32, Vec<usize>> = DetHashMap::default();
    let max_df = (ix.turns.len() as u32 / 12).clamp(6, 20);
    for (i, &(ti, _)) in cand_turns.iter().take(n).enumerate() {
        for &cid in &ix.turns[ti].content {
            let df = ix.t_own_stats.df.get(&cid).copied().unwrap_or(0);
            if df >= 1 && df <= max_df {
                term_post.entry(cid).or_default().push(i);
            }
        }
    }

    let mut adj: Vec<Vec<(usize, u64)>> = vec![Vec::new(); n];
    let mut edge_seen: DetHashSet<(usize, usize)> = DetHashSet::default();
    for (ci, members) in term_post {
        if members.len() < 2 || members.len() > 8 {
            continue;
        }
        let df = ix.t_own_stats.df.get(&ci).copied().unwrap_or(1).max(1) as u64;
        let w = Q16_ONE / df;
        for a_idx in 0..members.len() {
            for b_idx in (a_idx + 1)..members.len() {
                let u = members[a_idx];
                let v = members[b_idx];
                let key = if u < v { (u, v) } else { (v, u) };
                if edge_seen.insert(key) {
                    adj[u].push((v, w));
                    adj[v].push((u, w));
                }
            }
        }
    }
    for i in 0..n {
        let ti = cand_turns[i].0;
        for j in (i + 1)..n {
            let tj = cand_turns[j].0;
            if ix.turns[ti].session == ix.turns[tj].session && ti.abs_diff(tj) <= 2 {
                let key = (i, j);
                if edge_seen.insert(key) {
                    adj[i].push((j, Q16_ONE >> 1));
                    adj[j].push((i, Q16_ONE >> 1));
                }
            }
        }
    }

    let mut act: Vec<u64> = cand_turns.iter().take(n).map(|&(_, s)| s.max(1)).collect();
    for _step in 0..2 {
        let mut next = vec![0u64; n];
        for i in 0..n {
            let deg = adj[i].len();
            // Lateral inhibition: suppress isolated/promiscuous hubs with high fanout or zero peers.
            let self_keep = if deg > 6 {
                (act[i] * 3) / 4
            } else {
                act[i]
            };
            let mut in_flow: u64 = 0;
            let mut norm_sum: u64 = 0;
            for &(nbr, w) in &adj[i] {
                in_flow = in_flow.saturating_add(((act[nbr] as u128 * w as u128) >> 16) as u64);
                norm_sum = norm_sum.saturating_add(w);
            }
            let spread = if norm_sum > 0 {
                ((in_flow as u128 * Q16_ONE as u128) / (norm_sum.max(Q16_ONE) as u128)) as u64
            } else {
                0
            };
            let mut val = self_keep.saturating_add(spread >> 1);
            if ix.turns[cand_turns[i].0].is_habitual {
                val = val.saturating_add(val / 6);
            } else if deg >= 2 && deg <= 5 {
                val = val.saturating_add(val / 10);
            }
            next[i] = val;
        }
        act = next;
    }

    let mut out: Vec<(usize, u64)> = (0..n).map(|i| (cand_turns[i].0, act[i])).collect();
    out.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Session, Turn, build_index};

    #[test]
    fn synapse_boosts_cohesive_supported_turns_above_isolated_decoy() {
        let sessions = vec![
            Session {
                id: "s0".into(),
                date: String::new(),
                timestamp: 1000,
                turns: vec![
                    Turn {
                        id: "t0".into(),
                        role: "Melanie".into(),
                        content: "I always love bach classical symphony concerts".into(),
                        ..Default::default()
                    },
                    Turn {
                        id: "t1".into(),
                        role: "Melanie".into(),
                        content: "Mozart and bach symphonies are relaxing".into(),
                        ..Default::default()
                    },
                ],
            },
            Session {
                id: "s1".into(),
                date: String::new(),
                timestamp: 2000,
                turns: vec![Turn {
                    id: "t2".into(),
                    role: "Melanie".into(),
                    content: "Pop music summer sounds at the festival".into(),
                    ..Default::default()
                }],
            },
        ];
        let ix = build_index(&sessions, &Params::default());
        // t2 starts equal to t0, but t0 is habitual and reinforced by t1 via shared bach/symphony.
        let ranked = synapse_activate_speaker_turns_q16(&ix, &[(0, Q16_ONE), (1, Q16_ONE / 2), (2, Q16_ONE)]);
        assert_eq!(ranked[0].0, 0, "SYNAPSE must elevate reinforced habitual turn t0 above isolated t2");
    }
}
