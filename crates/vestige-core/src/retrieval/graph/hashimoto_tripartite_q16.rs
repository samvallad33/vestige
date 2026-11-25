//! Pigeonhole Set-Cover Multi-Hop Detection & Closed-Class Plural/Collective Quantifier Gate
//! (`Zero LLM, Zero Vectors`).

use crate::retrieval::query::intent::QueryPlan;

/// Pigeonhole Set-Cover Theorem for corpus-split Multi-Hop queries:
/// If `union_cov >= 2` distinct non-speaker query groups are attested in the corpus
/// yet no single session covers all of them (`max_sess_cov < union_cov`), then by
/// the Pigeonhole Principle answering the query strictly requires >= 2 sessions.
/// Conversely, Single-Hop queries satisfied inside one session have `max_sess_cov == union_cov`.
pub(in crate::retrieval) fn pigeonhole_corpus_split_mh(
    plan: &QueryPlan,
    s_exact_mask: &[u64],
    excluded_speakers: &[String],
) -> bool {
    let mut non_sp_gmask = 0u64;
    for (gi, g) in plan.groups.iter().take(64).enumerate() {
        if !excluded_speakers.iter().any(|sp| sp.eq_ignore_ascii_case(g)) {
            non_sp_gmask |= 1u64 << gi;
        }
    }
    if non_sp_gmask == 0 {
        return false;
    }
    let union_cov = s_exact_mask
        .iter()
        .fold(0u64, |acc, &m| acc | (m & non_sp_gmask))
        .count_ones();
    let max_sess_cov = s_exact_mask
        .iter()
        .map(|&m| (m & non_sp_gmask).count_ones())
        .max()
        .unwrap_or(0);
    union_cov >= 2 && max_sess_cov < union_cov
}

/// Closed-class plural WH (`what`/`which` + `>=4`-char plural word ending in `s`, excluding
/// closed-class auxiliaries/determiners) OR closed-class collective quantifiers
/// (`both`/`common`/`shared`/`together`). Zero open-class nouns.
pub(in crate::retrieval) fn closed_class_plural_or_collective_mh(q_toks_eff: &[String]) -> bool {
    // hardcode-scan: closed-class (WH interrogatives + closed-class auxiliaries/determiners)
    let plural_wh = q_toks_eff.windows(2).any(|p| {
        (p[0] == "what" || p[0] == "which")
            && p[1].len() >= 4
            && p[1].ends_with('s')
            && !p[1].ends_with("ss")
            && !p[1].ends_with("us")
            && !p[1].ends_with("is")
            && !matches!(
                p[1].as_str(),
                "was" | "does" | "has" | "this" | "is" | "kind" | "sort" | "else" | "most"
            )
    });
    // hardcode-scan: closed-class (collective/conjunctive quantifiers)
    let collective = q_toks_eff
        .iter()
        .any(|w| matches!(w.as_str(), "both" | "common" | "shared" | "together"));
    plural_wh || collective
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Query, Session, Turn, build_index, plan_query, raw_tokens};

    fn mk_turn(id: &str, role: &str, content: &str) -> Turn {
        Turn {
            id: id.into(),
            role: role.into(),
            content: content.into(),
            links: Vec::new(),
            stability: 0.0,
        }
    }

    #[test]
    fn pigeonhole_set_cover_distinguishes_split_multihop_from_single_hop() {
        let sessions = vec![
            Session {
                id: "D1".into(),
                date: "2023-05-01".into(),
                timestamp: 0,
                turns: vec![mk_turn(
                    "D1:1",
                    "Alice",
                    "I joined the robotics club with Mira.",
                )],
            },
            Session {
                id: "D2".into(),
                date: "2023-05-10".into(),
                timestamp: 0,
                turns: vec![mk_turn(
                    "D2:1",
                    "Alice",
                    "Mira won the regional trophy in Seattle.",
                )],
            },
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "Where did Alice's robotics partner win a trophy?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &p);
        let speakers = vec!["alice".to_string()];

        let non_sp_bits: Vec<usize> = plan
            .groups
            .iter()
            .enumerate()
            .filter(|(_, g)| !g.eq_ignore_ascii_case("alice"))
            .map(|(i, _)| i)
            .collect();
        assert!(non_sp_bits.len() >= 2, "groups={:?}", plan.groups);
        let b0 = 1u64 << non_sp_bits[0];
        let b1 = 1u64 << non_sp_bits[1];

        let split_masks = vec![b0, b1];
        assert!(pigeonhole_corpus_split_mh(&plan, &split_masks, &speakers));

        let single_masks = vec![b0 | b1, b0];
        assert!(!pigeonhole_corpus_split_mh(&plan, &single_masks, &speakers));
    }

    #[test]
    fn closed_class_plural_wh_and_collective_quantifiers_match_cleanly() {
        assert!(closed_class_plural_or_collective_mh(&raw_tokens(
            "What books has Alice read?"
        )));
        assert!(closed_class_plural_or_collective_mh(&raw_tokens(
            "Which places did both Alice and Bob visit?"
        )));
        assert!(closed_class_plural_or_collective_mh(&raw_tokens(
            "What do Alice and Bob have in common?"
        )));
        assert!(!closed_class_plural_or_collective_mh(&raw_tokens(
            "What was Alice doing yesterday?"
        )));
        assert!(!closed_class_plural_or_collective_mh(&raw_tokens(
            "What does Alice like?"
        )));
        assert!(!closed_class_plural_or_collective_mh(&raw_tokens(
            "What kind of dog does Alice own?"
        )));
    }
}
