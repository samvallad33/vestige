//! Corpus-Induced Query-Frame Entropy (#1) & Structural Slot-Witness Signatures (#10).
//!
//! Pure closed-class grammar + corpus document-frequency statistics (no open-class word lists).

use crate::retrieval::{Index, QueryPlan, raw_tokens};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::retrieval) enum SlotWitnessKind {
    None,
    /// Query asks what tools/items/approaches the user is using ("what ... using", "which ... did i").
    EntityListOrTech,
    /// Query asks how the assistant recommended/suggested handling something ("how did you ...").
    AssistantRecommendation,
    /// Query asks for a numeric/temporal quantity ("how many", "how much", "what time/date").
    NumericOrTemporal,
}

/// Pure corpus-IDF contrast weights per query group (`plan.groups`).
/// When a query mixes rare high-surprisal anchors (`max_idf`) with broad low-IDF meta-discourse
/// scaffolds (`idf < 0.48 * max_idf`), down-weight the broad terms to `[0.60, 0.80]` and gently
/// sharpen the rarest anchors up to `1.18`.
pub(in crate::retrieval) fn compute_query_surprisal_weights(ix: &Index, plan: &QueryPlan) -> Vec<f32> {
    let ng = plan.groups.len();
    let mut g_weights = vec![1.0f32; ng];
    if ng < 4 || ix.t_own_stats.n <= 8.0 {
        return g_weights;
    }

    let n_docs = ix.t_own_stats.n;
    let mut g_idf = vec![0.0f32; ng];
    let mut g_df_ratio = vec![0.0f32; ng];
    for qt in &plan.terms {
        if !qt.exact_hit {
            continue;
        }
        if let Some(g) = qt.group
            && g < ng
        {
            let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0) as f32;
            let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
            if idf > g_idf[g] {
                g_idf[g] = idf;
                g_df_ratio[g] = df / n_docs;
            }
        }
    }

    let max_idf = g_idf.iter().copied().fold(0.0f32, f32::max);
    if max_idf <= 1.2 {
        return g_weights;
    }

    // Only steepen when there is a genuine surprisal gap between rare anchors and broad words.
    let has_broad = g_idf.iter().any(|&v| v > 0.0 && v < 0.52 * max_idf);
    if !has_broad {
        return g_weights;
    }

    for g in 0..ng {
        let idf = g_idf[g];
        if idf <= 0.0 {
            continue;
        }
        let rel = idf / max_idf;
        if rel < 0.45 && g_df_ratio[g] >= 0.06 {
            // Broad meta-discourse scaffold: damp proportionally, floored at 0.62.
            g_weights[g] = (0.58 + 0.42 * (rel / 0.45)).clamp(0.62, 0.88);
        } else if rel >= 0.84 {
            // High-surprisal anchor: mild contrast sharpen.
            g_weights[g] = (1.0 + 0.16 * ((rel - 0.84) / 0.16)).clamp(1.0, 1.16);
        }
    }
    g_weights
}

/// Detect closed-class structural answer-slot expectation from the query text.
pub(in crate::retrieval) fn query_slot_witness_kind(q_text: &str) -> SlotWitnessKind {
    let q_low = q_text.trim().to_ascii_lowercase();
    let toks = raw_tokens(&q_low);
    if q_low.starts_with("how many ")
        || q_low.starts_with("how much ")
        || q_low.contains(" how many ")
        || q_low.contains(" how much ")
    {
        return SlotWitnessKind::NumericOrTemporal;
    }
    if (q_low.starts_with("how did you ") || q_low.contains(" did you "))
        && toks.iter().any(|t| matches!(t.as_str(), "suggest" | "recommend" | "advise" | "propose"))
    {
        return SlotWitnessKind::AssistantRecommendation;
    }
    if (q_low.starts_with("what ") || q_low.starts_with("which "))
        && toks.iter().any(|t| matches!(t.as_str(), "using" | "used" | "use"))
    {
        return SlotWitnessKind::EntityListOrTech;
    }
    SlotWitnessKind::None
}

/// Compute a gentle Q16 structural witness multiplier (`>= Q16_ONE`) when a candidate turn
/// matches the query's requested syntactic answer slot.
pub(in crate::retrieval) fn slot_witness_bonus_q16(
    ix: &Index,
    ti: usize,
    kind: SlotWitnessKind,
) -> u64 {
    const Q16_ONE: u64 = 65_536;
    if kind == SlotWitnessKind::None {
        return Q16_ONE;
    }
    let t = &ix.turns[ti];
    if t.is_backchannel || t.is_standing_policy {
        return Q16_ONE;
    }
    match kind {
        SlotWitnessKind::None => Q16_ONE,
        SlotWitnessKind::NumericOrTemporal => {
            if t.has_numeric && !t.is_question {
                Q16_ONE + (Q16_ONE / 10)
            } else {
                Q16_ONE
            }
        }
        SlotWitnessKind::AssistantRecommendation => {
            let has_enum = t.body_for_headers.contains("\n1.")
                || t.body_for_headers.contains("\n- ")
                || t.body_for_headers.contains("1. ");
            if t.field == 1 && !t.is_question && (has_enum || t.content.len() >= 18) {
                Q16_ONE + (Q16_ONE / 9)
            } else {
                Q16_ONE
            }
        }
        SlotWitnessKind::EntityListOrTech => {
            // Structural witness: version numbers, dots/slashes/underscores, or >= 2 clean entities.
            let body = &t.body_for_headers;
            let has_tech_shape = body.bytes().any(|b| b == b'/' || b == b'_' || b == b'+')
                || (t.has_numeric && body.as_bytes().windows(2).any(|w| w[0] == b'v' && w[1].is_ascii_digit()));
            if t.field == 0 && !t.is_question && (t.slot_entities_clean >= 2 || has_tech_shape) {
                Q16_ONE + (Q16_ONE / 10)
            } else {
                Q16_ONE
            }
        }
    }
}
