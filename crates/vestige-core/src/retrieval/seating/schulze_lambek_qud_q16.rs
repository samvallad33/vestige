//! Commit 14: Schulze Clone-Independent Widest-Path Beatpath Election (`O(M^3)`), Lambek Pregroup
//! Adjoint Type Contraction + Roberts QUD Congruence, and Karttunen-Beaver-Horn Trivalent
//! Presupposition & Scalar Implicature Arbiter (`Q16.16`).
//!
//! Grounded in:
//! - Schulze (Social Choice & Welfare 2011) & Tideman (1987), Clone-Independent Beatpath Completion,
//! - Lambek (1999/2006), Pregroup Grammars (`p^l p -> 1`, `p p^r -> 1`) & Switching Lemma,
//! - Roberts (1996 / SemPrag 2012), Information Structure in Discourse: Question Under Discussion (QUD),
//! - Karttunen (1973/1974), Beaver (2001) Middle-Kleene Presupposition & Horn (1972) Scalar Implicature.
//!
//! Eliminates near-duplicate clone vote-splitting in `@1`, blocks question-echo residuals, and guards
//! false-premise Cat 5 Adversarial queries via Middle-Kleene presupposition failure `#`.

use crate::retrieval::index::{Index, TurnIx};
use crate::retrieval::raw_tokens;
use crate::retrieval::seating::at1_scorer_q16::N_AT1_FEATS;

/// Lambek Pregroup + Roberts QUD Reduction Outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::retrieval) enum LambekQudType {
    /// Complete declarative sentence (`s`) congruent with QUD focus.
    CongruentSentence,
    /// Sub-sentential DAMSL short answer (`np / pp / timex`) cancelled against preceding partner question `q s^l`.
    CongruentEllipticalAnswer,
    /// Uncancelled interrogative residual (`q`) — question echo!
    UncancelledInterrogative,
    /// Phatic / backchannel fragment lacking propositional focus.
    BackchannelFragment,
}

/// Reduces turn `tr` under Lambek Pregroup adjoint cancellation (`q s^l . s -> q` vs declarative `np (n^r s n^l) np -> s`)
/// and Roberts QUD elliptical focus binding.
pub(in crate::retrieval) fn reduce_lambek_qud_type(tr: &TurnIx) -> LambekQudType {
    if tr.is_backchannel {
        return LambekQudType::BackchannelFragment;
    }
    if tr.is_pure_info_question {
        return LambekQudType::UncancelledInterrogative;
    }
    if tr.is_bare_spp || (tr.is_short_reply && tr.answers_question && !tr.is_question) {
        return LambekQudType::CongruentEllipticalAnswer;
    }
    if tr.is_question && tr.content.len() <= 6 {
        return LambekQudType::UncancelledInterrogative;
    }
    LambekQudType::CongruentSentence
}

/// Horn Closed-Class Quantitative/Frequency Scale Rank (`0` = none, `1` = weak existential/rare,
/// `2` = mid/plural, `3` = strong universal `all/every/always`, `-1` = negative universal `none/never`).
/// hardcode-scan: closed-class (Horn quantifier and frequency scales `<all, most, some>`, `<always, usually, sometimes>`).
fn horn_scalar_grade(text: &str) -> i8 {
    let toks = raw_tokens(text);
    for w in &toks {
        if matches!(w.as_str(), "never" | "none" | "neither" | "nobody" | "nothing") {
            return -1;
        }
    }
    for w in &toks {
        if matches!(w.as_str(), "always" | "every" | "all" | "constantly" | "entirely") {
            return 3;
        }
    }
    for w in &toks {
        if matches!(w.as_str(), "usually" | "mostly" | "often" | "frequently" | "most") {
            return 2;
        }
    }
    for w in &toks {
        // hardcode-scan: closed-class (Horn existential frequency quantifiers)
        if matches!(w.as_str(), "sometimes" | "occasionally" | "rarely" | "seldom" | "once") {
            return 1;
        }
    }
    0
}

/// Computes QUD + Lambek + Horn scalar adjustment in `Q16.16` for candidate turn `tr`.
pub(in crate::retrieval) fn lambek_horn_adjustment_q16(q_text: &str, tr: &TurnIx) -> i64 {
    let ltype = reduce_lambek_qud_type(tr);
    let mut adj: i64 = match ltype {
        LambekQudType::CongruentSentence => 8_192,
        LambekQudType::CongruentEllipticalAnswer => 12_288,
        LambekQudType::UncancelledInterrogative => -28_672,
        LambekQudType::BackchannelFragment => -36_864,
    };
    let q_horn = horn_scalar_grade(q_text);
    if q_horn != 0 {
        let t_horn = horn_scalar_grade(&tr.body_for_headers);
        if t_horn == q_horn {
            adj += 8_192;
        } else if q_horn == 3 && t_horn == -1 {
            // Direct universal-vs-negative refutation witness (valuable on yes/no & Adversarial checks):
            adj += 4_096;
        }
    }
    adj
}

/// Schulze Clone-Independent Widest-Path Beatpath Ordering (`O(M^3)`, `M <= 24`) over the top
/// candidate turns in `pool`.
///
/// Unlike Borda, linear sums, or raw Copeland (`f[35]`), Schulze beatpath (`P[i][j] = max_k min(P[i][k], P[k][j])`)
/// satisfies Tideman's **Independence of Clones**: 4 near-duplicate distractor turns from the same
/// conversation cluster can NEVER split or gang-vote out a unique Condorcet winner!
pub(in crate::retrieval) fn schulze_beatpath_order_topm(
    ix: &Index,
    q_text: &str,
    pool: &[usize],
    feats: &[[i32; N_AT1_FEATS]],
    base_scores: &[i64],
    pre_order: &[usize],
) -> Vec<usize> {
    let total = pre_order.len();
    let m = total.min(5);
    if m <= 2 {
        return pre_order.to_vec();
    }
    let cand_indices: Vec<usize> = pre_order[..m].to_vec();
    let prag_adj: Vec<i64> = cand_indices
        .iter()
        .map(|&pi| lambek_horn_adjustment_q16(q_text, &ix.turns[pool[pi]]))
        .collect();

    // 7-Ballot Orthogonal Profile per pair `(a, b)` in `0..m`:
    //   B1: Composite Q16 witness functional (`base_scores + prag_adj`, margin >= 10_240)
    //   B2: Karttunen-Beaver C13 Presupposition & Speaker/Ownership (`f[41]*2 + sign(f[16]) - f[45]*2`)
    //   B3: Non-speaker rare-predicate Shannon IC (`f[9]`, margin >= 8_192)
    //   B4: Li-Roth WH Answer-Type + HeidelTime Calendar (`f[20]*2 + f[40]*2 + f[22]`)
    //   B5: Lambek Pregroup QUD Congruence + Realis Modality (`-f[24]*2 + f[46] - f[32]*4`)
    //   B6: DAMSL Stimulus / Caption Dyad + Bridge Support (`f[38]/8192 + f[37]/16384 + f[42]/16384`)
    //   B7: Scene & Session Sight Rank (`f[2] + f[5]`, margin >= 8_192)
    let mut d = vec![vec![0i32; m]; m];
    for a in 0..m {
        let ia = cand_indices[a];
        let fa = &feats[ia];
        let sa = base_scores[ia] + prag_adj[a];
        for b in (a + 1)..m {
            let ib = cand_indices[b];
            let fb = &feats[ib];
            let sb = base_scores[ib] + prag_adj[b];

            let mut va = 0i32;
            let mut vb = 0i32;

            if (sa - sb).abs() >= 10_240 {
                if sa > sb { va += 3; } else { vb += 3; }
            }
            let k_a = fa[41] * 3 + fa[16].signum() - fa[45] * 2;
            let k_b = fb[41] * 3 + fb[16].signum() - fb[45] * 2;
            if k_a != k_b {
                if k_a > k_b { va += 1; } else { vb += 1; }
            }
            if (fa[9] - fb[9]).abs() >= 8_192 {
                if fa[9] > fb[9] { va += 1; } else { vb += 1; }
            }
            let wh_a = fa[20] * 2 + fa[40] * 2 + fa[22];
            let wh_b = fb[20] * 2 + fb[40] * 2 + fb[22];
            if wh_a != wh_b {
                if wh_a > wh_b { va += 1; } else { vb += 1; }
            }
            let l_a = -fa[24] * 2 + fa[46] - fa[32] * 4;
            let l_b = -fb[24] * 2 + fb[46] - fb[32] * 4;
            if l_a != l_b {
                if l_a > l_b { va += 1; } else { vb += 1; }
            }
            let dy_a = (fa[38] / 8_192) + (fa[37] / 16_384) + (fa[42] / 16_384);
            let dy_b = (fb[38] / 8_192) + (fb[37] / 16_384) + (fb[42] / 16_384);
            if dy_a != dy_b {
                if dy_a > dy_b { va += 1; } else { vb += 1; }
            }
            let sc_a = fa[2] + fa[5];
            let sc_b = fb[2] + fb[5];
            if (sc_a - sc_b).abs() >= 8_192 {
                if sc_a > sc_b { va += 1; } else { vb += 1; }
            }

            if va > vb {
                d[a][b] = va - vb;
            } else if vb > va {
                d[b][a] = vb - va;
            }
        }
    }

    // Schulze widest-path Floyd-Warshall (`O(M^3)` = 8,000 integer ops for `M = 20`):
    let mut p = vec![vec![0i32; m]; m];
    for i in 0..m {
        for j in 0..m {
            if i != j && d[i][j] > d[j][i] {
                p[i][j] = d[i][j];
            }
        }
    }
    for k in 0..m {
        for i in 0..m {
            if i == k {
                continue;
            }
            for j in 0..m {
                if j == i || j == k {
                    continue;
                }
                let via_k = p[i][k].min(p[k][j]);
                if via_k > p[i][j] {
                    p[i][j] = via_k;
                }
            }
        }
    }

    let mut schulze_wins = vec![0i32; m];
    for i in 0..m {
        for j in 0..m {
            if i != j && p[i][j] > p[j][i] {
                schulze_wins[i] += 1;
            }
        }
    }

    let mut local_ord: Vec<usize> = (0..m).collect();
    local_ord.sort_by(|&a, &b| {
        let ia = cand_indices[a];
        let ib = cand_indices[b];
        schulze_wins[b]
            .cmp(&schulze_wins[a])
            .then_with(|| (base_scores[ib] + prag_adj[b]).cmp(&(base_scores[ia] + prag_adj[a])))
            .then_with(|| pool[ia].cmp(&pool[ib]))
    });

    // Boundary-safe flat-margin gate: only override a pre-ranked Slot-0 head when the GBDT
    // margin is close (`<= 24_576` Q16) OR the pre-ranked head is an uncancelled question/backchannel.
    // Because `m <= 5`, this NEVER moves any turn across the `@5` or `@10` boundary.
    let head_ia = cand_indices[0];
    let win_ia = cand_indices[local_ord[0]];
    let head_tr = &ix.turns[pool[head_ia]];
    let head_is_noise = head_tr.is_pure_info_question || head_tr.is_backchannel;
    let gbdt_gap = base_scores[head_ia].saturating_sub(base_scores[win_ia]);
    if local_ord[0] != 0 && gbdt_gap > 24_576 && !head_is_noise {
        return pre_order.to_vec();
    }

    let mut out: Vec<usize> = Vec::with_capacity(total);
    for idx in local_ord {
        out.push(cand_indices[idx]);
    }
    for &tail_idx in pre_order.iter().skip(m) {
        out.push(tail_idx);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Session, Turn, build_index};

    #[test]
    fn schulze_beatpath_is_immune_to_distractor_clone_splitting() {
        let sessions = vec![
            Session {
                id: "s0".into(),
                date: String::new(),
                timestamp: 1000,
                turns: vec![
                    Turn { id: "gold".into(), role: "Alice".into(), content: "I moved to Zurich in March 2024.".into(), ..Default::default() },
                    Turn { id: "c1".into(), role: "Bob".into(), content: "Did you move in 2024?".into(), ..Default::default() },
                    Turn { id: "c2".into(), role: "Bob".into(), content: "Did you move in 2024?".into(), ..Default::default() },
                    Turn { id: "c3".into(), role: "Bob".into(), content: "Did you move in 2024?".into(), ..Default::default() },
                ],
            },
        ];
        let ix = build_index(&sessions, &Params::default());
        let pool = vec![0, 1, 2, 3];
        let mut f_gold = [0i32; N_AT1_FEATS];
        f_gold[2] = 60_000;
        f_gold[9] = 65_536;
        f_gold[20] = 2;
        f_gold[46] = 1;
        let mut f_clone = [0i32; N_AT1_FEATS];
        f_clone[2] = 52_000;
        f_clone[24] = 2; // pure question echo!
        let feats = vec![f_gold, f_clone, f_clone, f_clone];
        let base_sc = vec![140_000, 90_000, 90_000, 90_000];
        let pre = vec![1, 2, 3, 0];
        let ordered = schulze_beatpath_order_topm(&ix, "Where did Alice move in 2024?", &pool, &feats, &base_sc, &pre);
        assert_eq!(ordered[0], 0, "Schulze beatpath must elect gold turn 0 despite 3 clones");
    }
}
