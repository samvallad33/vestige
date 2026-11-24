//! Closed-Class Constraint & Directive Density Scorer (#3).
//!
//! Uses strictly closed-class deontic modals, negation/polarity markers, and preference
//! connectives to score turns carrying user preferences or standing instructions.

use crate::retrieval::raw_tokens;

const Q16_ONE: u64 = 65_536;

/// Returns `true` when the query uses closed-class preference/advice/step-request framing.
pub(in crate::retrieval) fn is_advice_or_preference_query(q_text: &str) -> bool {
    let q_low = q_text.trim().to_ascii_lowercase();
    let toks = raw_tokens(&q_low);
    q_low.starts_with("how can i ")
        || q_low.starts_with("how should ")
        || q_low.starts_with("could you ")
        || q_low.starts_with("can you help ")
        || q_low.contains(" would you suggest")
        || q_low.contains("i'd like to ")
        || q_low.contains("i would like to ")
        || toks.iter().any(|t| {
            matches!(
                t.as_str(),
                "prefer" | "preferred" | "preference" | "preferences" | "suggest" | "recommend"
            )
        })
}

/// Score the closed-class deontic/volitional constraint density of a turn in Q16 (`0..`).
pub(in crate::retrieval) fn deontic_directive_density_q16(text: &str) -> u64 {
    let low = text.to_ascii_lowercase();
    let toks = raw_tokens(&low);
    if toks.is_empty() {
        return 0;
    }
    let mut hits: u64 = 0;
    for t in &toks {
        if matches!(
            t.as_str(),
            "prefer"
                | "always"
                | "never"
                | "avoid"
                | "must"
                | "should"
                | "instead"
                | "rather"
                | "need"
                | "want"
                | "unless"
                | "only"
                | "without"
                | "dont"
                | "cant"
                | "cannot"
                | "wont"
                | "shouldnt"
        ) {
            hits += 1;
        }
    }
    if low.contains("i'd like") || low.contains("i would like") || low.contains("make sure") || low.contains("keep it") {
        hits += 2;
    }
    if hits == 0 {
        return 0;
    }
    // Saturating density per 24 tokens so concise constraints beat bloated boilerplate.
    let norm = (toks.len() as u64).max(8).min(48);
    hits.saturating_mul(Q16_ONE) / norm
}
