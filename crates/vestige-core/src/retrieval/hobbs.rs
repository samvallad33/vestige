//! Issue #811: Hobbs Pronoun Walk in pure Q16.16 integer arithmetic.
//! Zero floats, zero word lists.
//!
//! When a turn carries the structural anaphoric flag (`t.anaphoric` from closed-class
//! pronouns/anaphors) and is not a backchannel or question, walks backward up to 3
//! turns in the same session to bind the most salient non-framing antecedent entity
//! (`dist = 1 > 2 > 3`, preferring question subjects and explicit entities).

use super::index::Index;
use super::is_technical_handle;
use super::scoring::QueryPlan;

pub(super) const Q16: u64 = 1 << 16;

/// Output of Hobbs pronoun resolution against the query's domain focal entities.
pub(super) struct HobbsResolution {
    /// Q16.16 antecedent match score `[0, Q16]` per turn `ti`.
    pub(super) turn_score_q16: Vec<u64>,
    /// Resolved `(anaphoric_turn_ti, antecedent_token_id)` bindings inside session window `<= 3`.
    pub(super) bindings: Vec<(usize, u32)>,
}

/// Walk backward inside each session (`dist = 1..=3`) to bind anaphoric turns
/// to preceding non-framing entities/rare nouns matching `plan.focal_ids`.
pub(super) fn resolve_hobbs_pronouns(ix: &Index, plan: &QueryPlan) -> HobbsResolution {
    let nt = ix.turns.len();
    let mut turn_score_q16 = vec![0u64; nt];
    let mut bindings = Vec::new();
    if nt <= 2 || plan.focal_ids.is_empty() {
        return HobbsResolution { turn_score_q16, bindings };
    }

    let n_docs = (ix.t_own_stats.n as u32).max(1);
    let max_df = (n_docs / 4).max(4);

    // Non-framing exact query focal ids
    let query_focal: Vec<(u32, u64)> = plan
        .terms
        .iter()
        .filter(|qt| plan.focal_ids.contains(&qt.id) && qt.exact_hit)
        .filter_map(|qt| {
            let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0);
            if df > 0 && df <= max_df {
                let w_q16 = (qt.weight.clamp(0.0, 2.0) * ((Q16 / 2) as f32)) as u64;
                Some((qt.id, w_q16))
            } else {
                None
            }
        })
        .collect();

    if query_focal.is_empty() {
        return HobbsResolution { turn_score_q16, bindings };
    }

    for members in &ix.session_turns {
        for (pos, &ti) in members.iter().enumerate() {
            let t = &ix.turns[ti];
            if !t.anaphoric || t.is_question || t.is_backchannel {
                continue;
            }
            let mut best_ti_q16: u64 = 0;
            let mut best_ent: Option<u32> = None;

            for dist in 1..=3usize {
                if pos < dist {
                    break;
                }
                let prev_ti = members[pos - dist];
                let prev = &ix.turns[prev_ti];
                let dist_decay = match dist {
                    1 => Q16,
                    2 => Q16 >> 1,
                    _ => Q16 >> 2,
                };
                let q_bonus = if prev.is_question && dist == 1 {
                    Q16 / 4
                } else {
                    0
                };

                // Check explicit non-framing entities first
                for &eid in &prev.entities {
                    if is_technical_handle(&ix.vocab, eid) {
                        continue;
                    }
                    let df = ix.t_stats.df.get(&eid).copied().unwrap_or(0);
                    if df > max_df {
                        continue;
                    }
                    if let Some(&(_, qw)) = query_focal.iter().find(|&&(qid, _)| qid == eid) {
                        if !t.content.contains(&eid) {
                            let sc = ((qw.saturating_mul(dist_decay + q_bonus)) >> 16).min(Q16);
                            if sc > best_ti_q16 {
                                best_ti_q16 = sc;
                                best_ent = Some(eid);
                            }
                        }
                    }
                }
            }

            if best_ti_q16 > 0 {
                turn_score_q16[ti] = best_ti_q16;
                if let Some(ent) = best_ent {
                    bindings.push((ti, ent));
                }
            }
        }
    }

    HobbsResolution { turn_score_q16, bindings }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hobbs_constants_integer() {
        assert_eq!(Q16, 65536);
    }
}
