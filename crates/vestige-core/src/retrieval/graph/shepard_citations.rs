//! Shepard's Citations Signed Legal Citator (`Q16.16`, Frank Shepard 1873).
//!
//! Classifies backward RSB citation edges (`citing_session -> cited_session`) into three
//! formal legal treatments using closed-class grammar and Strata supersession:
//! - `Followed` (`+1.0`): non-superseded affirmative citation (boosts cited precedent),
//! - `Distinguished` (`+0.5`): scoped conditional exception (`unless/except/whereas`),
//! - `Overruled` (`0.0` on `current_state`, `+0.5` on historical queries): superseded authority.
//! Zero floats, zero open-class wordlists.

use crate::retrieval::index::Index;
use crate::retrieval::QueryIntent;

const Q16_ONE: u64 = 65_536;
const Q16_SHIFT: u32 = 16;
const SHEPARD_COEFF_Q16: u64 = 14_418; // 0.22

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::retrieval) enum ShepardTreatment {
    Followed,
    Distinguished,
    Overruled,
}

#[inline]
pub(in crate::retrieval) fn shepard_treatment_weight_q16(
    treatment: ShepardTreatment,
    deg: u32,
    current_state: bool,
) -> u64 {
    let base = match treatment {
        ShepardTreatment::Followed => Q16_ONE,
        ShepardTreatment::Distinguished => Q16_ONE >> 1,
        ShepardTreatment::Overruled => {
            if current_state {
                0
            } else {
                Q16_ONE >> 1
            }
        }
    };
    base / (deg.max(2) as u64)
}

pub(in crate::retrieval) fn compute_shepard_citator_lifts_q16(
    ix: &Index,
    it: &QueryIntent,
    s_ord_base: &[usize],
    smax_q16: u64,
) -> Vec<u64> {
    let ns = ix.session_turns.len();
    let mut out = vec![0u64; ns];
    let Some(&top0) = s_ord_base.first() else {
        return out;
    };
    if ns <= 5 || top0 >= ns || smax_q16 == 0 {
        return out;
    }

    // hardcode-scan: closed-class (conditional/distinguishing subordinators only)
    let is_distinguishing_conj =
        |w: &str| matches!(w, "unless" | "except" | "whereas" | "although" | "otherwise");

    let mut raw_auth = vec![0u64; ns];
    for &citing in s_ord_base.iter().take(4) {
        if citing >= ns {
            continue;
        }
        let has_distinguish = ix.session_turns[citing].iter().any(|&ti| {
            ix.turns[ti].feats.iter().any(|f| {
                ix.vocab
                    .names
                    .get(f.id as usize)
                    .is_some_and(|w| is_distinguishing_conj(w.as_str()))
            })
        });
        for &(deg, cited) in &ix.session_rsb_back[citing] {
            if cited >= ns || cited == top0 {
                continue;
            }
            let treatment = if ix.session_supersedes_back[citing].contains(&cited) {
                ShepardTreatment::Overruled
            } else if has_distinguish {
                ShepardTreatment::Distinguished
            } else {
                ShepardTreatment::Followed
            };
            let w = shepard_treatment_weight_q16(treatment, deg, it.current_state);
            raw_auth[cited] = raw_auth[cited].saturating_add(w);
        }
    }

    let max_a = raw_auth.iter().copied().max().unwrap_or(0);
    if max_a == 0 {
        return out;
    }
    for (si, &a) in raw_auth.iter().enumerate() {
        if si != top0 && a > 0 {
            let norm_q16 = (a.saturating_mul(Q16_ONE)) / max_a;
            out[si] = (((smax_q16.saturating_mul(SHEPARD_COEFF_Q16)) >> Q16_SHIFT)
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
    fn test_shepard_followed_beats_overruled_on_current_state() {
        let fol = shepard_treatment_weight_q16(ShepardTreatment::Followed, 2, true);
        let dist = shepard_treatment_weight_q16(ShepardTreatment::Distinguished, 2, true);
        let ovr = shepard_treatment_weight_q16(ShepardTreatment::Overruled, 2, true);
        assert!(fol > dist && dist > ovr && ovr == 0);
    }
}
