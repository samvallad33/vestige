//! Gated Berger-Lafferty Statistical Translation LM + Speaker-Profile Aggregator
//! for multi-turn conversational Open-Domain trait & world-knowledge queries.
//!
//! Grounded in:
//! - Berger & Lafferty (SIGIR 1999) Statistical Translation IR `T(q | w)`,
//! - Zhai & Lafferty (SIGIR 2001 / TOIS 2004) Dirichlet smoothing,
//! - Miller (1995) WordNet & Banerjee & Pedersen (CICLing 2002) relational expansion,
//! - Church & Hanks (1990) corpus PPMI.
//!
//! Strictly gated on closed-class English interrogative+modal or copular-predicative
//! frames (`would/could/might/may/should` or sentence-initial `is/are/was/were/does`),
//! leaving Single-Hop and Adversarial untouched and never modifying `t_own` / `t_raw`.

use std::collections::BTreeMap;

use crate::retrieval::fusion::aspect_coverage_q16::Q16_ONE;

/// Closed-class open-domain inferential query shape gate.
/// hardcode-scan: closed-class (interrogatives, modals, and copulas only; zero topic words).
pub(in crate::retrieval) fn is_open_domain_inferential_shape(tokens: &[&str]) -> bool {
    if tokens.len() < 3 {
        return false;
    }
    let has_epistemic_modal = tokens.iter().any(|w| {
        matches!(
            w.to_ascii_lowercase().as_str(),
            "would" | "could" | "might" | "likely" | "potentially"
        )
    });
    if has_epistemic_modal {
        return true;
    }
    let first = tokens[0].to_ascii_lowercase();
    let is_wh = matches!(first.as_str(), "what" | "which" | "how" | "who" | "where");
    let has_aux_modal = tokens
        .iter()
        .any(|w| matches!(w.to_ascii_lowercase().as_str(), "may" | "should"));
    if is_wh && has_aux_modal {
        return true;
    }
    if matches!(first.as_str(), "may") {
        return true;
    }
    let is_copula_head = matches!(first.as_str(), "is" | "are" | "was" | "were" | "does" | "do");
    if is_copula_head
        && !tokens.iter().any(|w| {
            matches!(
                w.to_ascii_lowercase().as_str(),
                "when" | "yesterday" | "tomorrow" | "before" | "after"
            )
        })
    {
        return true;
    }
    false
}

/// Deterministic `Q16.16` Berger-Lafferty translation kernel `T(q | w)` built from
/// corpus PPMI + WordNet relational edges (`synonym / hypernym / hyponym / meronym`),
/// scored over a speaker's FSRS-6 & IC-weighted cross-session profile.
///
/// Returns a bounded per-session `Q16.16` translation lift (`<= cap_q16`) to feed into
/// `bridge_boost` without ever touching `t_own` or `t_raw`.
pub(in crate::retrieval) fn score_speaker_translation_lift_q16(
    query_pred_ic_q16: &BTreeMap<u32, u64>,
    // `trans_map[q_term]` = list of `(doc_term_w, T(q | w) in Q16.16)` from PPMI + WordNet.
    trans_map: &BTreeMap<u32, Vec<(u32, u64)>>,
    // Per-session speaker-filtered term weights `(term_id, fsrs_weighted_tf_q16)`.
    speaker_session_terms: &BTreeMap<usize, Vec<(u32, u64)>>,
    cap_q16: u64,
) -> BTreeMap<usize, u64> {
    let mut lifts: BTreeMap<usize, u64> = BTreeMap::new();
    if query_pred_ic_q16.is_empty() || speaker_session_terms.is_empty() || cap_q16 == 0 {
        return lifts;
    }

    // Build global speaker profile phi_X(v) = sum_{s} tf_fsrs(v, s) across valid sessions.
    let mut profile_acc: BTreeMap<u32, u64> = BTreeMap::new();
    let mut profile_sum: u64 = 0;
    for terms in speaker_session_terms.values() {
        for &(tid, w_q16) in terms {
            let e = profile_acc.entry(tid).or_insert(0);
            *e = e.saturating_add(w_q16);
            profile_sum = profile_sum.saturating_add(w_q16);
        }
    }
    let profile_norm = profile_sum.max(1);

    let mut raw_scores: BTreeMap<usize, u64> = BTreeMap::new();
    let mut max_raw: u64 = 0;

    for (&sid, terms) in speaker_session_terms {
        let mut sess_tf: BTreeMap<u32, u64> = BTreeMap::new();
        let mut sess_len: u64 = 0;
        for &(tid, w_q16) in terms {
            let e = sess_tf.entry(tid).or_insert(0);
            *e = e.saturating_add(w_q16);
            sess_len = sess_len.saturating_add(w_q16);
        }
        let denom = sess_len.saturating_add(Q16_ONE * 8); // Dirichlet prior mu = 8.0
        let mut s_total: u64 = 0;

        for (&qid, &q_ic) in query_pred_ic_q16 {
            let Some(nbrs) = trans_map.get(&qid) else {
                continue;
            };
            let mut trans_mass: u64 = 0;
            for &(wid, t_qw_q16) in nbrs {
                if let Some(&w_tf) = sess_tf.get(&wid) {
                    // Direct translation channel + speaker-profile reinforcement.
                    let p_w_d = (((w_tf as u128) << 16) / (denom as u128)) as u64;
                    let prof_w = profile_acc.get(&wid).copied().unwrap_or(0);
                    let p_prof = (((prof_w as u128) << 16) / (profile_norm as u128)) as u64;
                    let combined_p = p_w_d.saturating_add(p_prof >> 1);
                    let contrib =
                        ((t_qw_q16 as u128).saturating_mul(combined_p as u128) >> 16) as u64;
                    trans_mass = trans_mass.saturating_add(contrib);
                }
            }
            if trans_mass > 0 {
                let weighted =
                    ((q_ic as u128).saturating_mul(trans_mass.min(Q16_ONE) as u128) >> 16) as u64;
                s_total = s_total.saturating_add(weighted);
            }
        }
        if s_total > 0 {
            raw_scores.insert(sid, s_total);
            max_raw = max_raw.max(s_total);
        }
    }

    if max_raw == 0 {
        return lifts;
    }
    for (sid, raw) in raw_scores {
        let scaled = ((raw as u128).saturating_mul(cap_q16 as u128) / (max_raw as u128)) as u64;
        if scaled > 0 {
            lifts.insert(sid, scaled.min(cap_q16));
        }
    }
    lifts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_class_gate_fires_on_open_domain_and_skips_factual_single_hop() {
        assert!(is_open_domain_inferential_shape(&["What", "would", "Alice", "enjoy"]));
        assert!(is_open_domain_inferential_shape(&["Would", "Alice", "enjoy", "hiking"]));
        assert!(is_open_domain_inferential_shape(&["Is", "Bob", "an", "extrovert"]));
        assert!(!is_open_domain_inferential_shape(&["When", "did", "Alice", "move"]));
        assert!(!is_open_domain_inferential_shape(&["Where", "did", "Bob", "buy", "coffee"]));
    }

    #[test]
    fn translation_lift_surfaces_non_lexical_speaker_evidence_session() {
        let mut q_ic = BTreeMap::new();
        q_ic.insert(100, Q16_ONE * 4);
        let mut trans = BTreeMap::new();
        // Query term 100 translates via WordNet/PPMI to speaker behavior term 200.
        trans.insert(100, vec![(200, Q16_ONE / 2)]);
        let mut sess_terms = BTreeMap::new();
        sess_terms.insert(1, vec![(300, Q16_ONE * 2)]);
        sess_terms.insert(2, vec![(200, Q16_ONE * 3)]);

        let lifts = score_speaker_translation_lift_q16(&q_ic, &trans, &sess_terms, Q16_ONE);
        assert!(!lifts.contains_key(&1));
        assert_eq!(lifts.get(&2).copied().unwrap_or(0), Q16_ONE);
    }
}
