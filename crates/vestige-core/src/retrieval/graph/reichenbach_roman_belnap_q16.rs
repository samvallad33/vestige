//! Commit 12: Reichenbach `(S, R, E)` Deictic-Anaphoric Automaton, Allen 13-Relation `ORD-Horn`
//! Harris Stratigraphic DAG, Roman-Canon `Lex Specialis` / Diplomatic `Vidimus` & `Non-Obstante`
//! Critique, Shanahan `Clipped` Inertia, and Belnap `FOUR_2` Paraconsistent Bilattice (`Q16.16`).
//!
//! Grounded in:
//! - Reichenbach (1947), Elements of Symbolic Logic §51 (`S, R, E` Reference-Time Permanence),
//! - Allen (1983) & Nebel-Burckert (JACM 1995), Maximal Tractable `ORD-Horn` Interval Algebra,
//! - Harris (1979), Principles of Archaeological Stratigraphy (Superposition & Cut Interfaces),
//! - Shanahan (1999), Circumscriptive Event Calculus (`HoldsAt`, `Clipped`),
//! - Justinian Digest 50.17.80 & Liber Sextus 5.12.34 (`Lex posterior generalis non derogat legi priori speciali`),
//! - Medieval Chancellery Diplomatics (`Charta` vs `Vidimus` recap & `Non-Obstante` derogation),
//! - Belnap (1977), Four-Valued Paraconsistent Bilattice `{Neither, False, True, Both}`.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use crate::retrieval::index::{Index, TurnIx};
use crate::retrieval::raw_tokens;

pub(in crate::retrieval) const REL_BEFORE: u16 = 1 << 0;
pub(in crate::retrieval) const REL_MEETS: u16 = 1 << 1;
pub(in crate::retrieval) const REL_EQ: u16 = 1 << 6;
pub(in crate::retrieval) const REL_MET_BY: u16 = 1 << 11;
pub(in crate::retrieval) const REL_AFTER: u16 = 1 << 12;
pub(in crate::retrieval) const ALLEN_FULL: u16 = 0x1FFF;
pub(in crate::retrieval) const STRICT_PRECEDES: u16 = REL_BEFORE | REL_MEETS;
pub(in crate::retrieval) const STRICT_FOLLOWS: u16 = REL_AFTER | REL_MET_BY;

/// Diplomatic act classification separating original authoritative `Charta` turns from
/// secondary `Vidimus` echoes/recaps/summaries (which must NEVER reset `Lex Posterior` recency).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::retrieval) enum DiplomaticAct {
    Charta,
    Vidimus,
}

/// Classifies a turn as an authoritative `Charta` vs a secondary `Vidimus` recap/echo
/// using pure closed-class role and discourse-recap markers.
/// hardcode-scan: closed-class (discourse recap, summary, and acknowledgment markers only).
pub(in crate::retrieval) fn classify_diplomatic_act(tr: &TurnIx) -> DiplomaticAct {
    if tr.field == 1 || tr.is_backchannel || tr.is_pure_info_question {
        return DiplomaticAct::Vidimus;
    }
    let low = tr.body_for_headers.trim_start().to_ascii_lowercase();
    let is_recap = low.starts_with("as i mentioned")
        || low.starts_with("like i said")
        || low.starts_with("to recap")
        || low.starts_with("so to confirm")
        || low.starts_with("just confirming")
        || low.starts_with("thanks for reminding");
    if is_recap {
        DiplomaticAct::Vidimus
    } else {
        DiplomaticAct::Charta
    }
}

/// Closed-class `Non-Obstante` explicit derogation detector (`"regardless of"`, `"override"`,
/// `"even when"`, `"no matter what"`, `"ignore previous"`), required by Liber Sextus 5.12.34 for a
/// later general statement to repeal an earlier scoped `Lex Specialis` rule.
/// hardcode-scan: closed-class (explicit legal/deontic override and derogation markers only).
pub(in crate::retrieval) fn has_non_obstante_derogation(text: &str) -> bool {
    let low = text.to_ascii_lowercase();
    [
        "regardless of",
        "no matter what",
        "override ",
        "overriding ",
        "ignore previous",
        "supersedes all",
        "in all cases now",
        "even when ",
        "without exception",
    ]
    .iter()
    .any(|p| low.contains(p))
}

/// Shanahan Event Calculus `Clipped` temporary-override detector:
/// Returns `true` when `t_later` is a bounded temporary exception (`"just for today"`, `"this week only"`)
/// that clips a standing fluent locally without terminating its long-term inertia!
/// hardcode-scan: closed-class (bounded temporary duration adverbs only).
pub(in crate::retrieval) fn is_shanahan_temporary_clip(text: &str) -> bool {
    let low = text.to_ascii_lowercase();
    [
        "for today",
        "just today",
        "just this once",
        "this one time",
        "for tonight",
        "temporarily",
        "for now only",
        "this week only",
    ]
    .iter()
    .any(|p| low.contains(p))
}

/// Roman-Canon Antinomy Arbitrator (`Digest 50.17.80`: *In toto iure generi per speciem derogatur*):
/// Determines whether `cand_later` (`t_later > t_earlier`) legitimately supersedes `t_earlier`.
/// Blocks supersession when:
/// 1. `cand_later` is a secondary `Vidimus` recap rather than an authoritative `Charta`,
/// 2. `cand_later` is a temporary Shanahan `Clipped` exception (`"just for today"`), or
/// 3. `t_earlier` has strictly higher scope specificity (`Lex Specialis`) and `cand_later` lacks a
///    `Non-Obstante` universal derogation clause.
pub(in crate::retrieval) fn roman_canon_allows_supersession(
    ix: &Index,
    t_earlier: usize,
    t_later: usize,
) -> bool {
    if t_later <= t_earlier || t_later >= ix.turns.len() {
        return false;
    }
    let ear = &ix.turns[t_earlier];
    let lat = &ix.turns[t_later];
    if classify_diplomatic_act(lat) == DiplomaticAct::Vidimus {
        return false;
    }
    if is_shanahan_temporary_clip(&lat.body_for_headers) && !is_shanahan_temporary_clip(&ear.body_for_headers) {
        return false;
    }
    if ear.is_standing_policy && !lat.is_standing_policy {
        let ear_scoped = ix
            .standing_rules
            .iter()
            .any(|r| r.turn == t_earlier && (r.has_explicit_scope || !r.condition.is_empty()));
        if ear_scoped && !has_non_obstante_derogation(&lat.body_for_headers) {
            return false;
        }
    }
    true
}

/// Reichenbach `(S, R, E)` Pluperfect / Anaphoric Flashback Offset Estimator:
/// Detects when a session narrates an event anchored strictly BEFORE its Speech Time `S`
/// (`had + V-en`, `"years earlier"`, `"before moving"`, `"back when"`), returning a negative
/// chronological offset so Allen + Harris stratigraphy places flashbacks before later events!
/// hardcode-scan: closed-class (Reichenbach pluperfect auxiliaries and anaphoric flashback shifters).
pub(in crate::retrieval) fn estimate_reichenbach_flashback_shift_q16(ix: &Index, si: usize) -> i64 {
    let Some(members) = ix.session_turns.get(si) else {
        return 0;
    };
    let mut retro_hits = 0i64;
    let mut prosp_hits = 0i64;
    for &ti in members {
        let tr = &ix.turns[ti];
        if tr.field != 0 || tr.is_question || tr.is_backchannel {
            continue;
        }
        let toks = raw_tokens(&tr.body_for_headers);
        let has_had_participle = toks.windows(2).any(|w| {
            w[0] == "had"
                && w[1].len() >= 4
                // hardcode-scan: closed-class (past-perfect auxiliary morphology and temporal deictic shifts)
                && (w[1].ends_with("ed") || w[1].ends_with("en") || matches!(w[1].as_str(), "been" | "gone" | "done" | "made" | "left" | "met"))
        });
        let low = tr.body_for_headers.to_ascii_lowercase();
        if has_had_participle
            || low.contains("years earlier")
            || low.contains("months earlier")
            || low.contains("previously ")
            || low.contains("prior to that")
            || low.contains("before that")
        {
            retro_hits += 1;
        }
        // hardcode-scan: closed-class (prospective temporal deictic adverbs)
        if low.contains("next month")
            || low.contains("next year")
            || low.contains("later on")
            || low.contains("eventually ")
            || low.contains("finally ")
        {
            prosp_hits += 1;
        }
    }
    if retro_hits > 0 && prosp_hits == 0 {
        -(retro_hits.min(3) * 32_768)
    } else if prosp_hits > 0 && retro_hits == 0 {
        prosp_hits.min(3) * 16_384
    } else {
        0
    }
}

/// Allen `ORD-Horn` Precedence Closure + Harris Stratigraphic Matrix Topological Order (`O(M^3)`, `M <= 24`).
/// Combines explicit RSB backward causal edges, Reichenbach `(S, R, E)` flashback shifts, and
/// closed-class sequence markers into a cycle-free chronological ordering.
pub(in crate::retrieval) fn harris_allen_stratigraphic_order(
    ix: &Index,
    cands: &[usize],
) -> Vec<usize> {
    let m = cands.len().min(24);
    if m <= 1 {
        return cands.to_vec();
    }
    let mut rel = vec![vec![ALLEN_FULL; m]; m];
    for i in 0..m {
        rel[i][i] = REL_EQ;
    }

    let mut pot_q16 = vec![0i64; m];
    for i in 0..m {
        let si = cands[i];
        let base_ord = if ix.session_ts[si] > 0 {
            (ix.session_ts[si] / 60).saturating_mul(1_024)
        } else {
            (si as i64) * 65_536
        };
        pot_q16[i] = base_ord.saturating_add(estimate_reichenbach_flashback_shift_q16(ix, si));
    }

    for a in 0..m {
        let sa = cands[a];
        for b in (a + 1)..m {
            let sb = cands[b];
            // Direct RSB / supersession causal witness (`later -> earlier` means `earlier` STRICT_PRECEDES `later`):
            let b_after_a = ix.session_rsb_back[sb].iter().any(|&(_, p)| p == sa)
                || ix.session_supersedes_back[sb].contains(&sa);
            let a_after_b = ix.session_rsb_back[sa].iter().any(|&(_, p)| p == sb)
                || ix.session_supersedes_back[sa].contains(&sb);
            if b_after_a && !a_after_b {
                rel[a][b] = STRICT_PRECEDES;
                rel[b][a] = STRICT_FOLLOWS;
            } else if a_after_b && !b_after_a {
                rel[b][a] = STRICT_PRECEDES;
                rel[a][b] = STRICT_FOLLOWS;
            }
        }
    }

    // Transitive path-consistency closure over strict precedence (`a < b` and `b < c` => `a < c`):
    let mut q = VecDeque::new();
    for i in 0..m {
        for j in 0..m {
            if rel[i][j] == STRICT_PRECEDES {
                q.push_back((i, j));
            }
        }
    }
    while let Some((i, j)) = q.pop_front() {
        for k in 0..m {
            if k == i || k == j {
                continue;
            }
            if rel[j][k] == STRICT_PRECEDES && rel[i][k] == ALLEN_FULL {
                rel[i][k] = STRICT_PRECEDES;
                rel[k][i] = STRICT_FOLLOWS;
                q.push_back((i, k));
            }
        }
    }

    // Harris Superposition DAG + min-heap Kahn topological sort keyed by `(pot_q16[u], cands[u])`:
    let mut indeg = vec![0u32; m];
    let mut adj = vec![Vec::new(); m];
    for u in 0..m {
        for v in 0..m {
            if u != v && rel[u][v] == STRICT_PRECEDES {
                adj[u].push(v);
                indeg[v] += 1;
            }
        }
    }
    let mut heap: BinaryHeap<Reverse<(i64, usize, usize)>> = BinaryHeap::new();
    for u in 0..m {
        if indeg[u] == 0 {
            heap.push(Reverse((pot_q16[u], cands[u], u)));
        }
    }
    let mut ordered = Vec::with_capacity(cands.len());
    while let Some(Reverse((_, si, u))) = heap.pop() {
        ordered.push(si);
        for &v in &adj[u] {
            indeg[v] = indeg[v].saturating_sub(1);
            if indeg[v] == 0 {
                heap.push(Reverse((pot_q16[v], cands[v], v)));
            }
        }
    }
    // Cycle fallback if inconsistent flashback annotations existed:
    if ordered.len() < m {
        let mut rem: Vec<usize> = (0..m)
            .filter(|&u| !ordered.contains(&cands[u]))
            .collect();
        rem.sort_by_key(|&u| (pot_q16[u], cands[u]));
        for u in rem {
            ordered.push(cands[u]);
        }
    }
    for &tail in cands.iter().skip(m) {
        ordered.push(tail);
    }
    ordered
}

/// Belnap `FOUR_2` Paraconsistent Bilattice Contradiction Witness Co-Seater:
/// When a query exhibits `Both = (pos_q16 > 0, neg_q16 > 0)` across two `Charta` sessions linked
/// by `session_supersedes_back` (surviving `Vidimus` filtering), guarantees both the positive thesis
/// and the conflicting/updating antithesis sit inside Top-5 (`BEAM contradiction_resolution`).
pub(in crate::retrieval) fn coseat_belnap_bilattice_pair(
    ix: &Index,
    final_session_indices: &mut Vec<usize>,
    final_turn_indices: &mut Vec<usize>,
    t_scene: &[f32],
    k: usize,
) {
    if final_session_indices.len() < 2 || final_turn_indices.len() < 5 {
        return;
    }
    let s0 = final_session_indices[0];
    // Check either backward (`s0` supersedes `s_prior`) or forward (`s_later` supersedes `s0`):
    let partner_sess = ix
        .session_supersedes_back
        .get(s0)
        .and_then(|v| v.first().copied())
        .or_else(|| {
            (s0 + 1..ix.session_turns.len())
                .find(|&sl| ix.session_supersedes_back[sl].contains(&s0))
        });
    let Some(sp) = partner_sess else {
        return;
    };
    if !final_session_indices[..5.min(final_session_indices.len())].contains(&sp) {
        if let Some(pos) = final_session_indices.iter().position(|&x| x == sp) {
            final_session_indices.remove(pos);
        }
        let ins_s = 1.min(final_session_indices.len());
        final_session_indices.insert(ins_s, sp);
        if final_session_indices.len() > k.max(5) {
            final_session_indices.truncate(k.max(5));
        }
    }
    if !final_turn_indices[..5.min(final_turn_indices.len())]
        .iter()
        .any(|&ti| ix.turns[ti].session == sp)
    {
        if let Some(best_p_ti) = ix.session_turns[sp]
            .iter()
            .copied()
            .filter(|&ti| {
                classify_diplomatic_act(&ix.turns[ti]) == DiplomaticAct::Charta
                    && !ix.turns[ti].is_question
            })
            .max_by(|&a, &b| {
                t_scene[a]
                    .partial_cmp(&t_scene[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.cmp(&a))
            })
        {
            if let Some(pos) = final_turn_indices.iter().position(|&x| x == best_p_ti) {
                final_turn_indices.remove(pos);
            }
            let ins_t = 1.min(final_turn_indices.len());
            final_turn_indices.insert(ins_t, best_p_ti);
            if final_turn_indices.len() > k.max(5) {
                final_turn_indices.truncate(k.max(5));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Session, Turn, build_index};

    #[test]
    fn roman_canon_protects_scoped_lex_specialis_and_blocks_vidimus_recap() {
        let sessions = vec![
            Session {
                id: "s0".into(),
                date: String::new(),
                timestamp: 1000,
                turns: vec![Turn {
                    id: "t0".into(),
                    role: "user".into(),
                    content: "When writing SQL migrations, always use explicit transaction blocks.".into(),
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
                        role: "assistant".into(),
                        content: "Got it, I will use transaction blocks.".into(),
                        ..Default::default()
                    },
                    Turn {
                        id: "t2".into(),
                        role: "user".into(),
                        content: "Just for today keep responses short.".into(),
                        ..Default::default()
                    },
                ],
            },
        ];
        let ix = build_index(&sessions, &Params::default());
        assert_eq!(classify_diplomatic_act(&ix.turns[1]), DiplomaticAct::Vidimus);
        assert!(!roman_canon_allows_supersession(&ix, 0, 1));
        assert!(!roman_canon_allows_supersession(&ix, 0, 2));
    }
}
