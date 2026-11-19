//! Centering Theory Discourse Coherence (`Q16.16`, Grosz, Joshi & Weinstein 1995).
//!
//! Tracks Forward-Looking Centers `Cf(U_n)`, Preferred Center `Cp(U_n)`, and Backward-Looking
//! Center `Cb(U_n)` across consecutive turns (carrying `Cb` across `t.anaphoric` pronouns),
//! scoring `CONTINUE` (`1.0`), `RETAIN` (`0.5`), and `SMOOTH-SHIFT` (`0.25`) transitions.
//! Zero floats, zero open-class wordlists.

use crate::retrieval::index::Index;
use crate::retrieval::QueryPlan;
use std::collections::BTreeSet;

const Q16_ONE: u64 = 65_536;
const Q16_SHIFT: u32 = 16;
const W_CONTINUE_Q16: u64 = Q16_ONE;
const W_RETAIN_Q16: u64 = Q16_ONE >> 1;
const W_SMOOTH_SHIFT_Q16: u64 = Q16_ONE >> 2;
const CENTERING_COEFF_Q16: u64 = 16_384; // 0.25

pub(in crate::retrieval) fn score_centering_transitions_q16(
    turns_cf: &[(Vec<u32>, bool)],
    focal_ents: &BTreeSet<u32>,
) -> u64 {
    let mut prev_cf: Vec<u32> = Vec::new();
    let mut prev_cb: Option<u32> = None;
    let mut total_q16 = 0u64;

    for (cf, is_anaphoric) in turns_cf {
        let mut cur_cb = prev_cf.iter().copied().find(|e| cf.contains(e));
        if cur_cb.is_none() && *is_anaphoric {
            cur_cb = prev_cb.or_else(|| prev_cf.first().copied());
        }
        let cur_cp = cf.first().copied().or(cur_cb);

        if let Some(cb) = cur_cb {
            if focal_ents.contains(&cb) {
                let same_cb = prev_cb == Some(cb);
                let is_cp = cur_cp == Some(cb);
                if same_cb && is_cp {
                    total_q16 = total_q16.saturating_add(W_CONTINUE_Q16);
                } else if same_cb {
                    total_q16 = total_q16.saturating_add(W_RETAIN_Q16);
                } else if is_cp {
                    total_q16 = total_q16.saturating_add(W_SMOOTH_SHIFT_Q16);
                }
            }
        }
        if !cf.is_empty() {
            prev_cf = cf.clone();
        }
        prev_cb = cur_cb;
    }
    total_q16
}

pub(in crate::retrieval) fn compute_centering_coherence_lifts_q16(
    ix: &Index,
    plan: &QueryPlan,
    top0: usize,
    smax_q16: u64,
) -> Vec<u64> {
    let ns = ix.session_turns.len();
    let mut out = vec![0u64; ns];
    if ns <= 5 || top0 >= ns || smax_q16 == 0 {
        return out;
    }
    let speakers: BTreeSet<String> = ix.turns.iter().filter(|t| !t.speaker.is_empty())
        .map(|t| t.speaker.to_lowercase()).collect();
    let focal_ents: BTreeSet<u32> = plan
        .terms
        .iter()
        .filter(|qt| qt.exact_hit && qt.group.is_some())
        .filter(|qt| ix.vocab.names.get(qt.id as usize).is_some_and(|name|
            !speakers.contains(name.trim_start_matches("\u{1}ent:"))))
        .map(|qt| qt.id)
        .collect();
    if focal_ents.is_empty() {
        return out;
    }

    let mut sess_sc = vec![0u64; ns];
    for (si, members) in ix.session_turns.iter().enumerate() {
        if si == top0 || members.len() < 2 {
            continue;
        }
        let mut turns_cf: Vec<(Vec<u32>, bool)> = Vec::with_capacity(members.len());
        for &ti in members {
            let t = &ix.turns[ti];
            if t.is_backchannel {
                continue;
            }
            let mut cf: Vec<u32> = Vec::new();
            for &(ex_id, _) in &t.seq {
                if (focal_ents.contains(&ex_id) || t.entities.contains(&ex_id))
                    && !cf.contains(&ex_id)
                {
                    cf.push(ex_id);
                }
            }
            turns_cf.push((cf, t.anaphoric));
        }
        if turns_cf.len() >= 2 {
            sess_sc[si] = score_centering_transitions_q16(&turns_cf, &focal_ents);
        }
    }

    let max_c = sess_sc.iter().copied().max().unwrap_or(0);
    if max_c == 0 {
        return out;
    }
    for (si, &sc) in sess_sc.iter().enumerate() {
        if si != top0 && sc > 0 {
            let norm_q16 = (sc.saturating_mul(Q16_ONE)) / max_c;
            out[si] = (((smax_q16.saturating_mul(CENTERING_COEFF_Q16)) >> Q16_SHIFT)
                .saturating_mul(norm_q16))
                >> Q16_SHIFT;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_centering_continue_beats_passing_mention() {
        let focal: BTreeSet<u32> = [5u32].into_iter().collect();
        let cont = vec![(vec![5, 2], false), (vec![], true), (vec![5, 3], false)];
        let pass = vec![(vec![9, 5], false), (vec![8, 7], false)];
        assert!(
            score_centering_transitions_q16(&cont, &focal)
                > score_centering_transitions_q16(&pass, &focal)
        );
    }
}

#[cfg(test)]
mod speaker_tests {
    use super::*;
    use crate::retrieval::{tests::{ix_of, s, turn}, *};
    #[test]
    fn speaker_only_query_has_no_centering_lift() {
        let sessions: Vec<_> = (0..8).map(|i| s(&i.to_string(), vec![
            turn("Alice", "Alice paints murals."), turn("Bob", "Alice paints again.")])).collect();
        let ix = ix_of(&sessions);
        let plan = crate::retrieval::query::plan::plan_query(&ix, &Query { text: "Alice".into(), date: String::new() }, &Params::default());
        assert!(compute_centering_coherence_lifts_q16(&ix, &plan, 0, 65536).iter().all(|v| *v == 0));
    }
}
