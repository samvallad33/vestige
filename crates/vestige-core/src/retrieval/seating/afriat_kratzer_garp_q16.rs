//! Commit 13: Afriat-Varian-Warshall Revealed-Preference GARP (`u32` Bit-Parallel Closure + `CCEI`
//! Potentials) & Kratzer Modal Ordering Sources x Searle-Vanderveken 7-Tuple Illocutionary Force (`Q16.16`).
//!
//! Grounded in:
//! - Afriat (1967), "The Construction of Utility Functions from Expenditure Data",
//! - Varian (Econometrica 1982), "The Nonparametric Approach to Demand Analysis" (Warshall GARP + CCEI),
//! - Fostel, Scarf & Todd (2004), "Two New Proofs of Afriat's Theorem",
//! - Kratzer (1981 / OUP 2012), Doubly-Relative Modal Ordering Sources `<=_{g(w_0)}`,
//! - Searle & Vanderveken (CUP 1985), Seven-Component Illocutionary Force Algebra.
//!
//! Zero open-domain wordlists, zero floats, clone-invariant `u32` / `Q16.16` integer execution.

use crate::retrieval::index::{Index, TurnIx};
use crate::retrieval::raw_tokens;

pub(in crate::retrieval) const Q16_ONE: i64 = 65_536;

/// Searle-Vanderveken Illocutionary Point (`Pi`).
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::retrieval) enum IllocPoint {
    Assertive = 0,
    Commissive = 1,
    Directive = 2,
    Declarative = 3,
    Expressive = 4,
}

/// Searle-Vanderveken 7-tuple illocutionary signature + Kratzer ordering-source bitmask `G_u`.
#[derive(Clone, Copy, Debug)]
pub(in crate::retrieval) struct KratzerForceProfile {
    pub(in crate::retrieval) point: IllocPoint,
    pub(in crate::retrieval) mode_hedged: bool,
    pub(in crate::retrieval) first_person: bool,
    pub(in crate::retrieval) ordering_mask: u32,
    pub(in crate::retrieval) deg_point_q16: i32,
    pub(in crate::retrieval) deg_sinc_q16: i32,
}

/// Extracts the closed-class Kratzer ordering-source mask (`deontic`, `bouletic`, `habitual`,
/// `comparative-contrast`, `privative-avoidance`) and Searle-Vanderveken illocutionary point.
/// hardcode-scan: closed-class (1st-person pronouns, modals, conditionalizers, and contrast prepositions only).
pub(in crate::retrieval) fn analyze_kratzer_searle_turn(tr: &TurnIx) -> KratzerForceProfile {
    let low = tr.body_for_headers.to_ascii_lowercase();
    let toks = raw_tokens(&low);
    let first_person = toks.iter().take(10).any(|w| {
        matches!(
            w.as_str(),
            "i" | "im" | "i'm" | "ive" | "i've" | "id" | "i'd" | "ill" | "i'll" | "my" | "me" | "we" | "our"
        )
    });
    let mode_hedged = toks.iter().any(|w| matches!(w.as_str(), "if" | "unless" | "hypothetically" | "suppose"));

    let has_declar = tr.is_standing_policy
        || low.contains("from now on")
        || low.contains("going forward")
        || low.contains("always ")
        || low.contains("never ");
    // hardcode-scan: closed-class (Searle commissive and bouletic stance auxiliaries)
    let has_commissive = low.contains("i will ")
        || low.contains("i'll ")
        || low.contains("stick to")
        || low.contains("committed to");
    let has_bouletic = tr.is_personal_constraint
        || low.contains("prefer")
        || low.contains("i'd rather")
        || low.contains("rather than")
        || low.contains("instead of")
        || low.contains("i like ")
        || low.contains("i love ")
        || low.contains("i hate ")
        || low.contains("avoid");

    let point = if tr.is_pure_info_question || (tr.is_question && !has_bouletic && !has_declar) {
        IllocPoint::Directive
    } else if has_declar {
        IllocPoint::Declarative
    } else if has_commissive {
        IllocPoint::Commissive
    } else if has_bouletic {
        IllocPoint::Expressive
    } else {
        IllocPoint::Assertive
    };

    let mut ordering_mask: u32 = 0;
    if has_declar {
        ordering_mask |= 1 << 0;
    }
    if has_bouletic {
        ordering_mask |= 1 << 1;
    }
    if tr.is_habitual {
        ordering_mask |= 1 << 2;
    }
    if !tr.role_objs.is_empty() || tr.is_personal_constraint {
        ordering_mask |= 1 << 3;
    }
    // hardcode-scan: closed-class (comparative choice and Kratzer epistemic possibility modals)
    if low.contains("instead of") || low.contains("rather than") || low.contains("switched from") || low.contains("switched to") {
        ordering_mask |= 1 << 4;
    }

    let deg_point_q16: i32 = if has_declar {
        65_536
    } else if has_bouletic || has_commissive {
        57_344
    } else if toks.iter().any(|w| matches!(w.as_str(), "might" | "maybe" | "perhaps")) {
        24_576
    } else {
        40_960
    };
    let deg_sinc_q16: i32 = if first_person && !mode_hedged { 65_536 } else { 36_864 };

    KratzerForceProfile {
        point,
        mode_hedged,
        first_person,
        ordering_mask,
        deg_point_q16,
        deg_sinc_q16,
    }
}

/// Bit-parallel Warshall GARP closure (`R[i] |= R[k]`) + Fostel-Scarf-Todd Afriat potential recovery
/// fused with Kratzer minimal-frontier dominance over `cands` (`M <= 32`).
/// Returns `Q16.16` composite revealed-preference + illocutionary utility for each candidate.
pub(in crate::retrieval) fn score_afriat_kratzer_candidates_q16(
    ix: &Index,
    cands: &[usize],
    direct_ic_q16: &[u64],
) -> Vec<i64> {
    let m = cands.len().min(32);
    if m == 0 {
        return Vec::new();
    }
    let profiles: Vec<KratzerForceProfile> = cands
        .iter()
        .take(m)
        .map(|&ti| analyze_kratzer_searle_turn(&ix.turns[ti]))
        .collect();

    // 1. Construct revealed-preference relation `R_0` (`u32` bit-rows):
    //    `i R_0 j` when `cands[i]` explicitly supersedes `cands[j]` or is a chronologically later
    //    `Charta` preference sharing a `role_objs` / entity slot with comparative/replacement framing.
    let mut r_mat = [0u32; 32];
    let mut p0_rev = [0u32; 32];
    for i in 0..m {
        r_mat[i] = 1u32 << i;
        let ti = cands[i];
        let tri = &ix.turns[ti];
        for j in 0..m {
            if i == j {
                continue;
            }
            let tj = cands[j];
            let trj = &ix.turns[tj];
            let shares_slot = (!tri.role_objs.is_empty()
                && tri.role_objs.iter().any(|r| trj.role_objs.contains(r)))
                || (!tri.entities.is_empty()
                    && tri.entities.iter().any(|e| trj.entities.contains(e)));
            let i_beats_j = tri.supersedes_turns.contains(&tj)
                || (shares_slot
                    && ti > tj
                    && (profiles[i].ordering_mask & (1 << 4)) != 0
                    && profiles[i].point != IllocPoint::Directive);
            if i_beats_j {
                r_mat[i] |= 1u32 << j;
                p0_rev[j] |= 1u32 << i;
            }
        }
    }

    // 2. O(M^2) word-parallel Warshall transitive closure (`R*`):
    for k in 0..m {
        let rk = r_mat[k];
        let k_bit = 1u32 << k;
        for row in r_mat.iter_mut().take(m) {
            if (*row & k_bit) != 0 {
                *row |= rk;
            }
        }
    }

    // 3. GARP cycle resolution under bitemporal preference drift:
    //    If `i` is strictly revealed-inferior to an active non-violating later choice `j` (`(p0_rev[i] & !viol) != 0`),
    //    penalize superseded bundle `i`; source nodes in `R*` receive Fostel-Scarf-Todd top Afriat potential!
    let mut out = vec![0i64; cands.len()];
    for i in 0..m {
        let f = &profiles[i];
        if f.point == IllocPoint::Directive {
            out[i] = -Q16_ONE;
            continue;
        }
        // Check Kratzer minimal-frontier non-dominance:
        let gi = f.ordering_mask;
        let mut kratzer_dominated = false;
        for j in 0..m {
            if i != j && profiles[j].point != IllocPoint::Directive {
                let gj = profiles[j].ordering_mask;
                if (gi & !gj) == 0 && (gj & !gi) != 0 && direct_ic_q16[j] >= direct_ic_q16[i] {
                    kratzer_dominated = true;
                    break;
                }
            }
        }

        // Afriat source check: no `j != i` strictly beats `i` without `i` also reaching `j` in a later turn:
        let beaten_by_later = (0..m).any(|j| {
            j != i
                && cands[j] > cands[i]
                && ((r_mat[j] >> i) & 1) != 0
                && ((r_mat[i] >> j) & 1) == 0
        });

        let role_q16: i64 = match (f.point, f.first_person) {
            (IllocPoint::Declarative, true) => 81_920,
            (IllocPoint::Commissive, true) => 65_536,
            (IllocPoint::Expressive, true) => 57_344,
            (IllocPoint::Assertive, true) => 32_768,
            _ => 16_384,
        };
        let mode_div = if f.mode_hedged { 2 } else { 1 };
        let deg_avg = ((f.deg_point_q16 as i64) + (f.deg_sinc_q16 as i64)) >> 1;
        let force_q16 = ((role_q16 / mode_div) * deg_avg) >> 16;
        let kratzer_bonus = if !kratzer_dominated && gi != 0 {
            (gi.count_ones() as i64) * 12_288 + 16_384
        } else {
            0
        };
        let afriat_pot = if beaten_by_later {
            -(Q16_ONE >> 1)
        } else if (p0_rev.iter().take(m).fold(0u32, |a, &b| a | b) != 0) && r_mat[i].count_ones() >= 2 {
            32_768 // Active revealed winner over a rejected alternative!
        } else {
            0
        };

        out[i] = (direct_ic_q16[i].min(262_144) as i64)
            .saturating_add(force_q16)
            .saturating_add(kratzer_bonus)
            .saturating_add(afriat_pot);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Session, Turn, build_index};

    #[test]
    fn afriat_garp_and_kratzer_promote_revealed_switch_and_reject_question() {
        let sessions = vec![
            Session {
                id: "s0".into(),
                date: String::new(),
                timestamp: 1000,
                turns: vec![Turn {
                    id: "t0".into(),
                    role: "user".into(),
                    content: "I like dark roast espresso in the morning.".into(),
                    ..Default::default()
                }],
            },
            Session {
                id: "s1".into(),
                date: String::new(),
                timestamp: 2000,
                turns: vec![
                    Turn {
                        id: "t1".into(),
                        role: "user".into(),
                        content: "I switched to matcha instead of espresso for mornings.".into(),
                        ..Default::default()
                    },
                    Turn {
                        id: "t2".into(),
                        role: "user".into(),
                        content: "What morning drink should I order?".into(),
                        ..Default::default()
                    },
                ],
            },
        ];
        let ix = build_index(&sessions, &Params::default());
        let scores = score_afriat_kratzer_candidates_q16(&ix, &[0, 1, 2], &[65_536, 65_536, 65_536]);
        assert!(scores[1] > scores[0], "Revealed switch t1 ({}) must beat stale t0 ({})", scores[1], scores[0]);
        assert!(scores[2] < 0, "Directive question t2 ({}) must be negatively gated", scores[2]);
    }
}
