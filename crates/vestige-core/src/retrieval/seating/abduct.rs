//! Weighted Abduction, Peyrard KL-Importance, Bipartite Coherence & Sperber-Wilson LZ77 NCD
//! Pragmatic Intra-Session Reranker (`Q16.16`).
//!
//! Pure deterministic integer `Q16.16` engine (`ONE = 65_536`) with zero vectors, zero
//! embeddings, zero LLMs, and zero external compression crates:
//!
//! 1. **Hobbs Weighted Abduction + `COLING 2012 C12-1079` `IP` / `CU` Guards**:
//!    Bounded 2-hop meet-in-the-middle backward chaining across bundled Princeton WordNet 3.1,
//!    ConceptNet 5.7, Roget 1911, and Wikidata CC0. Enforces Conditional Unification (`CU`)
//!    blocking on high-DF generic predicates without specific modifier support, awards
//!    Incompatible Property (`IP`) sibling/antonym constraint-revision proofs, and applies
//!    asymmetric WordNet Heyting hypernym / derivational-frame implication.
//! 2. **Peyrard Unifying KL-Divergence Importance (`ACL 2019 P19-1101`) + `SALMA 2026`**:
//!    Evaluates `-KL(P_T || P_{D/K})` in `Q16.16` over both content unigrams and adjacent
//!    content modifier-head bigrams `(m, h)` with exact integer `log2_sum_exp` normalization,
//!    driving zero-information conversational fillers strongly negative.
//! 3. **Guinaudeau & Strube Bipartite Entity-Turn Coherence (`ACL 2013 P13-1010`) + RST**:
//!    Session-local Centering role-weighted (`S=3, O=2, X=1`) distance-decayed one-mode
//!    entity centrality multiplied by closed-class RST factors (`0.5x` Ono epistemic-hedge
//!    satellite penalty vs `1.5x` Marcu priority/causal nucleus promotion, `SIGDIAL 2010 W10-4327`).
//! 4. **Computable Sperber-Wilson Relevance Ratio (`2026.salma-1.5`) + Pure-Rust 32KB LZ77 NCD
//!    (`Findings of ACL 2023`)**:
//!    Symmetric Normalized Compression Similarity over canonical lemma + top-synset byte
//!    streams combined with abductive gain and mean surprisal, divided by effort
//!    `(eps + lambda_len * log2(1 + |T|) + lambda_stop * stop_ratio)`.
//! 5. **Strict Scoped Seating (`final_session_indices[0..5]`, `ns <= 24`)**:
//!    Protects `implicit_disclosure_locked`, `torrens_locked_head`, standing policies, and
//!    declarative `star_bit` Slot-0 factual winners while elevating high-relevance abductive
//!    and constraint-bearing turns ahead of zero-IC fillers in Slots `0..5`.

use std::collections::{BTreeMap, BTreeSet};

use crate::conceptnet::reciprocal_two_hop_neighbors;
use crate::retrieval::context::RankCtx;
use crate::retrieval::fusion::aspect_coverage_q16::aspect_ic_q16;
use crate::retrieval::index::normalize::derivational_base_stems;
use crate::retrieval::seating::gricean_constraint_turn_q16::is_implicit_constraint_planning_query;
use crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo;
use crate::retrieval::seating::revealed_preference_cone_q16::is_factual_past_lookup_query;
use crate::retrieval::{
    DetHashMap, DetHashSet, Index, QueryIntent, QueryPlan, c2_named_session, eff_summary_request,
    raw_tokens, stem, stopwords, turn_is_superseded,
};
use crate::roget::conceptual_neighbors;
use crate::wikidata::upward_containers;
use crate::wordnet::{
    full_lexicon_neighbors, heyting_implication_ic_q16, lcs_sibling_resonance,
    strict_co_hyponym_siblings, upward_hypernyms,
};

/// Signed fixed-point unit `1.0` in `Q16.16` (`2^16`).
pub(in crate::retrieval) const ONE: i64 = 65_536;
const Q16_SHIFT: u32 = 16;

/// Exact `round(65536 * log2(1 + m / 16))` knots for `m in 0..=16`.
const LOG2_MANTISSA_LUT_Q16: [i64; 17] = [
    0, 5_732, 11_136, 16_248, 21_098, 25_711, 30_109, 34_312, 38_336, 42_196, 45_904, 49_472,
    52_911, 56_229, 59_434, 62_534, 65_536,
];

/// Exact `round(65536 * 2^(-m / 16))` knots for `m in 0..=16`.
const EXP2_NEG_LUT_Q16: [u64; 17] = [
    65_536, 62_757, 60_097, 57_549, 55_109, 52_773, 50_535, 48_393, 46_341, 44_376, 42_495, 40_693,
    38_968, 37_316, 35_734, 34_219, 32_768,
];

// Hobbs Weighted Abduction edge backchain cost factors (Q16.16).
const W_SYN_DERIV_Q16: i64 = 0x0000_5999; // ~0.35
const W_WN_CORE_Q16: i64 = 0x0000_9999; // 0.60
const W_CN_DIRECT_Q16: i64 = 0x0000_B333; // 0.70
const W_RELATED_Q16: i64 = 0x0000_CCCD; // 0.80
const ALPHA_DEG_Q16: i64 = 0x0000_147B; // ~0.08 hub degree penalty
const GAMMA_SIB_Q16: i64 = 0x0000_8000; // 0.50 C12-1079 IP sibling/antonym bonus
const HEYTING_SPEC_FACTOR_Q16: i64 = 0x0000_6666; // 0.40 specific=>general implication cost

// Peyrard KL-Divergence Importance weights (Q16.16).
const PEYRARD_ALPHA_Q16: i64 = ONE; // 1.00 session concentration
const PEYRARD_BETA_Q16: i64 = 0x0001_4000; // 1.25 background informativeness
const PEYRARD_PROTECT_FLOOR_Q16: i64 = -4 * ONE; // Slot-0 protection threshold

// Centering syntactic role weights (P13-1010) and RST multipliers (W10-4327).
const ROLE_SUBJECT_Q16: i64 = 3 * ONE;
const ROLE_OBJECT_Q16: i64 = 2 * ONE;
const ROLE_OTHER_Q16: i64 = ONE;
const RST_ONO_HEDGE_Q16: i64 = ONE / 2; // 0.5x
const RST_NEUTRAL_Q16: i64 = ONE; // 1.0x
const RST_MARCU_NUCLEUS_Q16: i64 = (3 * ONE) / 2; // 1.5x

// Sperber-Wilson Relevance Ratio weights (2026.salma-1.5 + ACL 2023 NCD).
const LAMBDA_H_Q16: i64 = 0x0000_6000; // 0.375 mean surprisal weight
const LAMBDA_NCD_Q16: i64 = 2 * ONE; // 2.00 LZ77 NCD weight
const EFFORT_EPS_Q16: i64 = ONE; // 1.00 base cognitive effort
const LAMBDA_LEN_Q16: i64 = 0x0000_6000; // 0.375 log2(1 + |T|) length effort
const LAMBDA_STOP_Q16: i64 = 2 * ONE; // 2.00 filler/stopword effort

// Final fusion weights (Q16.16).
const FUSE_W_SW_Q16: i64 = ONE; // 1.00 Sperber-Wilson relevance
const FUSE_W_PEYRARD_Q16: i64 = 0x0000_A000; // 0.625 Peyrard KL + SALMA
const FUSE_W_COH_Q16: i64 = 0x0000_6000; // 0.375 Bipartite Coherence * RST
const FUSE_W_ABD_Q16: i64 = 0x0000_C000; // 0.75 Direct Abductive + IP proof

// Pure-Rust 32KB-window hash-chained LZ77 constants.
const LZ77_WINDOW: usize = 32_768;
const LZ77_HASH_BITS: usize = 12;
const LZ77_HASH_SIZE: usize = 1 << LZ77_HASH_BITS;
const LZ77_HASH_MASK: usize = LZ77_HASH_SIZE - 1;
const LZ77_MIN_MATCH: usize = 3;
const LZ77_MAX_MATCH: usize = 258;
const LZ77_MAX_CHAIN: usize = 16;
const LZ77_SENTINEL: u32 = u32::MAX;
const MAX_CANON_STREAM_BYTES: usize = 512;

/// Per-turn abductive & pragmatic diagnostic score record in `Q16.16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) struct AbductiveTurnScore {
    pub(in crate::retrieval) turn_idx: usize,
    pub(in crate::retrieval) session_idx: usize,
    pub(in crate::retrieval) abd_gain_q16: i64,
    pub(in crate::retrieval) peyrard_q16: i64,
    pub(in crate::retrieval) coherence_rst_q16: i64,
    pub(in crate::retrieval) ncs_q16: i64,
    pub(in crate::retrieval) sw_rel_q16: i64,
    pub(in crate::retrieval) total_q16: i64,
}

#[inline]
fn q16_mul(a: i64, b: i64) -> i64 {
    let prod = (a as i128).saturating_mul(b as i128) >> Q16_SHIFT;
    prod.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

#[inline]
fn q16_div(num: i64, den: i64) -> i64 {
    if den == 0 {
        return 0;
    }
    let scaled = (num as i128) << Q16_SHIFT;
    (scaled / (den as i128)).clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

/// Deterministic piecewise-linear `log2(x)` in signed `Q16.16` (zero floats).
#[inline]
pub(in crate::retrieval) fn log2_q16(x: u32) -> i64 {
    if x <= 1 {
        return 0;
    }
    let msb = (u32::BITS - 1) - x.leading_zeros();
    let int_part = (msb as i64) << Q16_SHIFT;
    let frac16 = if msb >= Q16_SHIFT {
        ((x as u64) >> (msb - Q16_SHIFT)) & 0xFFFF
    } else {
        ((x as u64) << (Q16_SHIFT - msb)) & 0xFFFF
    };
    let idx = (frac16 >> 12) as usize;
    let rem = (frac16 & 0x0FFF) as i64;
    let lo = LOG2_MANTISSA_LUT_Q16[idx];
    let hi = LOG2_MANTISSA_LUT_Q16[idx + 1];
    int_part + lo + (((hi - lo) * rem) >> 12)
}

#[inline]
fn log2_u64_q16_signed(x: u64) -> i64 {
    if x <= 1 {
        return 0;
    }
    let msb = (u64::BITS - 1) - x.leading_zeros();
    let int_part = (msb as i64) << Q16_SHIFT;
    let frac16 = if msb >= Q16_SHIFT {
        (x >> (msb - Q16_SHIFT)) & 0xFFFF
    } else {
        (x << (Q16_SHIFT - msb)) & 0xFFFF
    };
    let idx = (frac16 >> 12) as usize;
    let rem = (frac16 & 0x0FFF) as i64;
    let lo = LOG2_MANTISSA_LUT_Q16[idx];
    let hi = LOG2_MANTISSA_LUT_Q16[idx + 1];
    int_part + lo + (((hi - lo) * rem) >> 12)
}

/// Exact integer `2^(delta_q16 / 2^16)` scaled by `2^16` for `delta_q16 <= 0`.
#[inline]
fn exp2_neg_q16(delta_q16: i64) -> u64 {
    if delta_q16 >= 0 {
        return ONE as u64;
    }
    let neg = (-delta_q16) as u64;
    let int_shift = (neg >> Q16_SHIFT) as u32;
    if int_shift >= 16 {
        return 0;
    }
    let frac16 = neg & 0xFFFF;
    let idx = (frac16 >> 12) as usize;
    let rem = frac16 & 0x0FFF;
    let lo = EXP2_NEG_LUT_Q16[idx];
    let hi = EXP2_NEG_LUT_Q16[idx + 1];
    let base = lo.saturating_sub(((lo - hi) * rem) >> 12);
    base >> int_shift
}

/// Closed-class English wrapper/light-verb/quantifier predicate for filtering non-specific hubs.
fn is_closed_wrapper(w: &str) -> bool {
    if w.len() < 3 || stopwords().contains(w) {
        return true;
    }
    // hardcode-scan: closed-class (light verbs, modals, quantifiers, temporal units, and discourse request wrappers)
    const CLOSED_WRAPPERS: &[&str] = &[
        "some", "any", "many", "much", "more", "most", "very", "just", "really", "actually",
        "maybe", "perhaps", "almost", "quite", "rather", "still", "already", "always", "never",
        "often", "sometimes", "usually", "lately", "recently", "currently", "finally", "soon",
        "today", "tonight", "yesterday", "tomorrow", "good", "great", "nice", "best", "better",
        "little", "short", "long", "first", "last", "next", "other", "another", "sure", "want",
        "wanted", "wants", "need", "needed", "needs", "like", "liked", "feel", "feeling", "think",
        "thinking", "thought", "know", "knew", "known", "guess", "suppose", "seem", "seems",
        "look", "looking", "going", "coming", "getting", "make", "made", "making", "take", "took",
        "taking", "give", "gave", "giving", "help", "tell", "told", "say", "said", "ask", "asked",
        "try", "trying", "keep", "put", "pull", "find", "filter", "search", "recommend",
        "recommendations", "suggest", "suggestions", "ideas", "advice", "tips", "options", "thing",
        "things", "way", "ways", "time", "times", "day", "days", "week", "weeks", "month",
        "months", "year", "years", "theme", "point", "kind", "type", "sort", "bit", "lot", "part",
        "case", "stuff", "something", "anything", "everything", "nothing", "someone", "anyone",
        "everyone", "here", "there", "now", "then", "well", "too", "back", "even", "around",
    ];
    CLOSED_WRAPPERS.contains(&w)
}

/// Closed-class imperative action, filter, decision, and advisory frame detector.
/// Triggers cleanly without requiring a trailing `?`, while excluding factual past lookups.
pub(in crate::retrieval) fn is_imperative_or_advisory_frame(query: &str) -> bool {
    if is_factual_past_lookup_query(query) || eff_summary_request(query) {
        return false;
    }
    let ql = query.trim().to_ascii_lowercase();
    if ql.is_empty() {
        return false;
    }
    // hardcode-scan: closed-class (English imperative opening verbs and filter/decision directives)
    const IMPERATIVE_OPENERS: &[&str] = &[
        "pull up ",
        "filter ",
        "find ",
        "search ",
        "look up ",
        "recommend ",
        "suggest ",
        "plan ",
        "draft ",
        "prepare ",
        "outline ",
        "put together ",
        "help me ",
        "show me ",
        "give me ",
        "compare ",
        "evaluate ",
        "prioritize ",
        "schedule ",
        "create ",
        "write ",
        "build ",
        "design ",
        "pick ",
        "choose ",
        "select ",
        "organize ",
        "map out ",
        "narrow down ",
        "shortlist ",
    ];
    if IMPERATIVE_OPENERS.iter().any(|&p| ql.starts_with(p)) {
        return true;
    }
    // hardcode-scan: closed-class (English modal advisory, planning, and decision request frames)
    const ADVISORY_CUES: &[&str] = &[
        "what should",
        "how should",
        "when should",
        "where should",
        "should i ",
        "should we ",
        "would it be",
        "could you help",
        "can you help",
        "can you recommend",
        "can you suggest",
        "help me ",
        "any ideas",
        "any suggestions",
        "any recommendations",
        "any advice",
        "any tips",
        "best way",
        "best time",
        "optimal time",
        "what's a good",
        "what is a good",
        "pros and cons",
        "what approach",
        "ways to ",
        "looking to ",
        "trying to decide",
        "pull up ",
        "filter for ",
        "recommend",
        "suggest",
    ];
    ADVISORY_CUES.iter().any(|&c| ql.contains(c))
}

/// Closed-class RST salience multiplier in `Q16.16` (`SIGDIAL 2010 W10-4327`):
/// - `0.5x` (`32_768`) for Ono epistemic-hedge / concession / phatic satellite turns
/// - `1.5x` (`98_304`) for Marcu priority / causal / evaluative nucleus turns
/// - `1.0x` (`65_536`) neutral (or `1.25x` when a substantive nucleus overrides a mild hedge).
pub(in crate::retrieval) fn rst_nucleus_multiplier_q16(text: &str) -> i64 {
    let low = text.trim().to_ascii_lowercase();
    if low.is_empty() {
        return RST_ONO_HEDGE_Q16;
    }
    // hardcode-scan: closed-class (Ono epistemic hedges, concessions, and phatic satellite markers)
    const ONO_HEDGE_SATELLITES: &[&str] = &[
        "i'm not sure",
        "i\u{2019}m not sure",
        "i am not sure",
        "not really sure",
        "not sure",
        "i guess",
        "maybe",
        "perhaps",
        "hard to say",
        "who knows",
        "i suppose",
        "that's true, though",
        "that\u{2019}s true, though",
        "fair enough",
        "good point",
        "makes sense",
        "sounds good",
        "sounds great",
        "will do",
        "got it",
    ];
    // hardcode-scan: closed-class (Marcu priority, causal, evaluative, and deontic nucleus markers)
    const MARCU_NUCLEUS_MARKERS: &[&str] = &[
        "top priority",
        "number one",
        "most important",
        "main priority",
        "key lesson",
        "learned that",
        "learned from",
        "emphasized",
        "crucial",
        "essential",
        "non-negotiable",
        "dealbreaker",
        "must have",
        "because ",
        "the reason",
        "since i ",
        "as a result",
        "therefore",
        "strictly ",
        "never trust",
    ];

    let has_hedge = ONO_HEDGE_SATELLITES.iter().any(|&m| low.contains(m));
    let has_nucleus = MARCU_NUCLEUS_MARKERS.iter().any(|&m| low.contains(m));
    match (has_nucleus, has_hedge) {
        (true, false) => RST_MARCU_NUCLEUS_Q16,
        (false, true) => RST_ONO_HEDGE_Q16,
        (true, true) => {
            let word_cnt = low.split_whitespace().count();
            let starts_hedge = ONO_HEDGE_SATELLITES.iter().any(|&m| low.starts_with(m));
            if starts_hedge && word_cnt <= 12 {
                RST_ONO_HEDGE_Q16
            } else {
                (5 * ONE) / 4
            }
        }
        (false, false) => RST_NEUTRAL_Q16,
    }
}

/// Extract normalized content tokens, splitting hyphenated compounds (`"4-bedroom"` -> `"bedroom"`,
/// `"non-transparent"` -> `"transparent"`) without mutating UTF-8 char boundaries.
fn extract_substantive_atoms(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut push_atom = |w: &str| {
        if w.len() < 3 || is_closed_wrapper(w) {
            return;
        }
        if !w.chars().any(char::is_alphabetic) {
            return;
        }
        if !out.iter().any(|x| x == w) {
            out.push(w.to_string());
        }
    };
    // Preserve hyphenated sub-constituents alongside standard tokenizer output.
    for raw_ws in text.split_whitespace() {
        let low = raw_ws.to_ascii_lowercase();
        if low.contains('-') {
            for part in low.split('-') {
                let clean = part.trim_matches(|c: char| !c.is_alphabetic());
                if clean.len() >= 3 {
                    push_atom(clean);
                }
            }
        }
    }
    for tok in raw_tokens(text) {
        push_atom(&tok);
    }
    out
}

/// Extract ordered unigrams and adjacent content modifier-head bigrams (`SALMA 2026`)
/// plus stopword/wrapper token count for Peyrard KL and Sperber-Wilson effort.
struct TurnLexUnits {
    /// `(unit_key, count)` for unigrams, synthetic `__STOP__`, and adjacent content bigrams.
    units: BTreeMap<String, u32>,
    total_units: u32,
    raw_len: u32,
    stop_or_wrapper_len: u32,
    content_tokens: Vec<String>,
}

const STOP_BUCKET_KEY: &str = "\u{1}__stop__";

fn extract_turn_lex_units(text: &str) -> TurnLexUnits {
    let toks = raw_tokens(text);
    let raw_len = toks.len() as u32;
    let mut stop_or_wrapper_len: u32 = 0;
    let mut units: BTreeMap<String, u32> = BTreeMap::new();
    let mut total_units: u32 = 0;
    let mut content_tokens: Vec<String> = Vec::new();
    let mut prev_content: Option<String> = None;

    for tok in toks {
        let is_wrap = is_closed_wrapper(&tok) || !tok.chars().any(char::is_alphabetic);
        if is_wrap {
            stop_or_wrapper_len += 1;
            *units.entry(STOP_BUCKET_KEY.to_string()).or_insert(0) += 1;
            total_units += 1;
            // Allow single short function/hyphen breaks only on non-clause tokens; break bigram chain.
            prev_content = None;
        } else {
            *units.entry(tok.clone()).or_insert(0) += 1;
            total_units += 1;
            if let Some(ref prev) = prev_content {
                // SALMA 2026 modifier-head bigram preservation `(m, h)` (weighted x2 so
                // low-frequency attribute-bound modifiers are not washed out by unigrams).
                let mut bg = String::with_capacity(prev.len() + 1 + tok.len());
                bg.push_str(prev);
                bg.push('\u{2}');
                bg.push_str(&tok);
                *units.entry(bg).or_insert(0) += 2;
                total_units += 2;
            }
            content_tokens.push(tok.clone());
            prev_content = Some(tok);
        }
    }

    // Also include hyphenated compound head tokens (`"4-bedroom"` -> `"bedroom"`, `"non-transparent"` -> `"transparent"`).
    for raw_ws in text.split_whitespace() {
        let low = raw_ws.to_ascii_lowercase();
        if low.contains('-') {
            let parts: Vec<&str> = low
                .split('-')
                .map(|p| p.trim_matches(|c: char| !c.is_alphabetic()))
                .filter(|p| p.len() >= 3 && !is_closed_wrapper(p))
                .collect();
            for &p in &parts {
                if !content_tokens.iter().any(|c| c == p) {
                    content_tokens.push(p.to_string());
                    *units.entry(p.to_string()).or_insert(0) += 1;
                    total_units += 1;
                }
            }
            if parts.len() >= 2 {
                let mut bg = String::with_capacity(parts[0].len() + 1 + parts[1].len());
                bg.push_str(parts[0]);
                bg.push('\u{2}');
                bg.push_str(parts[1]);
                *units.entry(bg).or_insert(0) += 2;
                total_units += 2;
            }
        }
    }

    TurnLexUnits {
        units,
        total_units,
        raw_len,
        stop_or_wrapper_len,
        content_tokens,
    }
}

// =========================================================================
// ALGORITHM 4A: PURE-RUST 32KB-WINDOW HASH-CHAINED LZ77 & SYMMETRIC NCD
// =========================================================================

#[inline]
fn lz77_hash3(bytes: &[u8], pos: usize) -> usize {
    let v = (bytes[pos] as u32)
        | ((bytes[pos + 1] as u32) << 8)
        | ((bytes[pos + 2] as u32) << 16);
    ((v.wrapping_mul(0x1E35_A7BD)) >> (32 - LZ77_HASH_BITS)) as usize & LZ77_HASH_MASK
}

/// Deterministic pure-Rust 32KB-window hash-chained LZ77 compressed bit-length estimator.
/// Zero external crates, zero unsafe code, `< 1 ms` per Top-5 session.
pub(in crate::retrieval) fn lz77_len(bytes: &[u8]) -> usize {
    let n = bytes.len();
    if n == 0 {
        return 0;
    }
    if n < LZ77_MIN_MATCH {
        return n * 8;
    }
    let mut head = vec![LZ77_SENTINEL; LZ77_HASH_SIZE];
    let mut prev = vec![LZ77_SENTINEL; n];
    let mut pos: usize = 0;
    let mut bits: usize = 0;

    while pos < n {
        if pos + LZ77_MIN_MATCH <= n {
            let h = lz77_hash3(bytes, pos);
            let mut cand = head[h];
            let mut best_len: usize = 0;
            let mut steps: usize = 0;
            let max_here = LZ77_MAX_MATCH.min(n - pos);

            while cand != LZ77_SENTINEL && steps < LZ77_MAX_CHAIN {
                let c = cand as usize;
                if pos.saturating_sub(c) > LZ77_WINDOW {
                    break;
                }
                if best_len < LZ77_MIN_MATCH || bytes[c + best_len] == bytes[pos + best_len] {
                    let mut m = 0usize;
                    while m < max_here && bytes[c + m] == bytes[pos + m] {
                        m += 1;
                    }
                    if m > best_len {
                        best_len = m;
                        if best_len >= max_here {
                            break;
                        }
                    }
                }
                cand = prev[c];
                steps += 1;
            }

            if best_len >= LZ77_MIN_MATCH {
                bits += 16; // 16-bit DEFLATE-style (len, dist) match token
                let end_ins = (pos + best_len).min(n.saturating_sub(LZ77_MIN_MATCH - 1));
                for p in pos..end_ins {
                    let hp = lz77_hash3(bytes, p);
                    prev[p] = head[hp];
                    head[hp] = p as u32;
                }
                pos += best_len;
            } else {
                prev[pos] = head[h];
                head[h] = pos as u32;
                bits += 8; // 8-bit literal byte
                pos += 1;
            }
        } else {
            bits += 8;
            pos += 1;
        }
    }
    bits
}

fn lz77_len_concat(a: &[u8], b: &[u8]) -> usize {
    let mut buf = Vec::with_capacity(a.len() + 1 + b.len());
    buf.extend_from_slice(a);
    buf.push(b' ');
    buf.extend_from_slice(b);
    lz77_len(&buf)
}

/// Symmetric Normalized Compression Similarity `NCS(a, b) = 1 - NCD(a, b)` in `Q16.16`
/// (`Jiang et al., Findings of ACL 2023`). Guaranteed `NCS(a, b) == NCS(b, a)` in `[0, ONE]`.
pub(in crate::retrieval) fn ncd_similarity_q16(a: &[u8], b: &[u8]) -> i64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let ca = lz77_len(a) as i64;
    let cb = lz77_len(b) as i64;
    let cab = lz77_len_concat(a, b) as i64;
    let cba = lz77_len_concat(b, a) as i64;
    let c_sym = (cab + cba) / 2;
    let denom = ca.max(cb).max(1);
    let num = (c_sym - ca.min(cb)).max(0);
    let ncd_q16 = q16_div(num, denom).clamp(0, ONE);
    (ONE - ncd_q16).max(0)
}

/// Build a compact deterministic canonical lemma + top-synset byte stream (`<= 512` bytes)
/// for LZ77 NCD matching over WordNet + ConceptNet + Roget.
fn build_canonical_synset_stream(atoms: &[String]) -> Vec<u8> {
    let mut bag: BTreeSet<String> = BTreeSet::new();
    for atom in atoms.iter().take(12) {
        bag.insert(atom.clone());
        let st = stem(atom);
        if st.len() >= 3 && !is_closed_wrapper(&st) {
            bag.insert(st);
        }
        for &lem in upward_hypernyms(atom, 1).iter().take(4) {
            if !is_closed_wrapper(lem) {
                bag.insert(lem.to_string());
            }
        }
        for &(lem, w) in full_lexicon_neighbors(atom).iter().take(6) {
            if w >= 85 && !is_closed_wrapper(lem) {
                bag.insert(lem.to_string());
            }
        }
        for &(nbr, w) in reciprocal_two_hop_neighbors(atom, 4).iter() {
            if w >= 90 && !is_closed_wrapper(nbr) {
                bag.insert(nbr.to_string());
            }
        }
        for &(nbr, w) in conceptual_neighbors(atom, 4).iter() {
            if w >= 80 && !is_closed_wrapper(nbr) {
                bag.insert(nbr.to_string());
            }
        }
    }
    let mut out = Vec::with_capacity(MAX_CANON_STREAM_BYTES);
    for w in bag {
        let wb = w.as_bytes();
        if !out.is_empty() {
            if out.len() + 1 + wb.len() > MAX_CANON_STREAM_BYTES {
                break;
            }
            out.push(b' ');
        } else if wb.len() > MAX_CANON_STREAM_BYTES {
            break;
        }
        out.extend_from_slice(wb);
    }
    out
}

// =========================================================================
// ALGORITHM 1: HOBBS WEIGHTED ABDUCTION + C12-1079 IP SIBLINGS & CU BLOCK
// =========================================================================

struct QueryAtomSpec {
    word: String,
    stem_word: String,
    ic_q16: i64,
    cost0_q16: i64,
    cu_generic: bool,
    /// Reachable 1-hop concepts `u -> step_factor_q16` in `(0, ONE)`.
    fwd_factors: BTreeMap<String, i64>,
    /// Mutually exclusive WordNet co-hyponym siblings / antonyms (`C12-1079` `IP`).
    ip_siblings: BTreeSet<String>,
}

#[inline]
fn degree_penalized_factor(base_w_q16: i64, degree: usize) -> i64 {
    let deg_pen = ONE + q16_mul(ALPHA_DEG_Q16, log2_q16(degree.max(1) as u32));
    q16_mul(base_w_q16, deg_pen).clamp(ONE / 4, (15 * ONE) / 10)
}

fn word_ic_q16(ix: &Index, word: &str) -> i64 {
    let n_turns = (ix.turns.len() as u32).max(2);
    if let Some(id) = ix.vocab.get(word) {
        let df = ix.t_own_stats.df.get(&id).copied().unwrap_or(0);
        return (aspect_ic_q16(df, n_turns) as i64).clamp(ONE / 2, 8 * ONE);
    }
    let st = stem(word);
    if let Some(id) = ix.vocab.get(&crate::retrieval::stem_term(&st)) {
        let df = ix.t_own_stats.df.get(&id).copied().unwrap_or(0);
        return (aspect_ic_q16(df, n_turns) as i64).clamp(ONE / 2, 8 * ONE);
    }
    // Unseen/OOV term has high intrinsic specificity.
    4 * ONE
}

/// `C12-1079` Conditional Unification (`CU`) generic-predicate detector:
/// returns `true` when `word` is a closed wrapper OR has high session DF (`> max(3, ns/2)`)
/// and low corpus IC (`< 2 * ONE`), blocking spurious overmerged proofs without specific support.
fn is_cu_generic_predicate(ix: &Index, ns: usize, word: &str, ic_q16: i64) -> bool {
    if is_closed_wrapper(word) {
        return true;
    }
    let s_df_limit = (ns / 2).max(3) as u32;
    if let Some(id) = ix.vocab.get(word) {
        let s_df = ix.s_stats.df.get(&id).copied().unwrap_or(0);
        if s_df > s_df_limit && ic_q16 < 2 * ONE {
            return true;
        }
    }
    let st = stem(word);
    if let Some(id) = ix.vocab.get(&crate::retrieval::stem_term(&st)) {
        let s_df = ix.s_stats.df.get(&id).copied().unwrap_or(0);
        if s_df > s_df_limit && ic_q16 < 2 * ONE {
            return true;
        }
    }
    false
}

/// Expand a word 1 hop over bundled WordNet + ConceptNet + Roget + Wikidata, returning
/// `concept -> degree-penalized backchain factor` in `Q16.16`.
fn expand_one_hop_factors(word: &str) -> BTreeMap<String, i64> {
    let mut map: BTreeMap<String, i64> = BTreeMap::new();
    let mut insert_min = |k: &str, fac: i64| {
        if k.len() < 3 || k == word || is_closed_wrapper(k) {
            return;
        }
        let entry = map.entry(k.to_string()).or_insert(i64::MAX);
        if fac < *entry {
            *entry = fac;
        }
    };

    for deriv in derivational_base_stems(word) {
        insert_min(&deriv, W_SYN_DERIV_Q16);
    }

    let wn_nbrs = full_lexicon_neighbors(word);
    let wn_deg = wn_nbrs.len().max(1);
    for &(lem, w) in wn_nbrs.iter().take(20) {
        let base_w = if w >= 95 {
            W_SYN_DERIV_Q16
        } else if w >= 72 {
            W_WN_CORE_Q16
        } else {
            W_RELATED_Q16
        };
        insert_min(lem, degree_penalized_factor(base_w, wn_deg));
    }

    let up_hyp = upward_hypernyms(word, 2);
    let up_deg = up_hyp.len().max(1);
    for &lem in up_hyp.iter().take(12) {
        insert_min(lem, degree_penalized_factor(W_WN_CORE_Q16, up_deg));
    }

    let cn_nbrs = reciprocal_two_hop_neighbors(word, 16);
    let cn_deg = cn_nbrs.len().max(1);
    for &(nbr, w) in &cn_nbrs {
        if w < 70 {
            continue;
        }
        let base_w = if w >= 100 {
            W_CN_DIRECT_Q16
        } else {
            W_RELATED_Q16
        };
        insert_min(nbr, degree_penalized_factor(base_w, cn_deg));
    }

    let rg_nbrs = conceptual_neighbors(word, 20);
    let rg_deg = rg_nbrs.len().max(1);
    for &(nbr, w) in &rg_nbrs {
        if w < 58 {
            continue;
        }
        let base_w = if w >= 80 {
            W_CN_DIRECT_Q16
        } else {
            W_RELATED_Q16
        };
        insert_min(nbr, degree_penalized_factor(base_w, rg_deg));
    }

    let wd_up = upward_containers(word, 2);
    let wd_deg = wd_up.len().max(1);
    for &cont in wd_up.iter().take(6) {
        insert_min(cont, degree_penalized_factor(W_WN_CORE_Q16, wd_deg));
    }

    map
}

fn build_query_abduct_specs(ix: &Index, ns: usize, query: &str) -> Vec<QueryAtomSpec> {
    let atoms = extract_substantive_atoms(query);
    let mut specs = Vec::with_capacity(atoms.len().min(12));
    for word in atoms.into_iter().take(12) {
        let ic_q16 = word_ic_q16(ix, &word);
        let cost0_q16 = ic_q16.clamp(ONE, 8 * ONE);
        let cu_generic = is_cu_generic_predicate(ix, ns, &word, ic_q16);
        let fwd_factors = expand_one_hop_factors(&word);

        let mut ip_siblings: BTreeSet<String> = BTreeSet::new();
        for &sib in strict_co_hyponym_siblings(&word).iter().take(24) {
            if !is_closed_wrapper(sib) {
                ip_siblings.insert(sib.to_string());
            }
        }
        for &(sib, w) in lcs_sibling_resonance(&word, 16).iter() {
            if w >= 66 && !is_closed_wrapper(sib) {
                ip_siblings.insert(sib.to_string());
            }
        }

        let stem_word = stem(&word);
        specs.push(QueryAtomSpec {
            word,
            stem_word,
            ic_q16,
            cost0_q16,
            cu_generic,
            fwd_factors,
            ip_siblings,
        });
    }
    specs
}

/// Compute Hobbs Weighted Abduction gain + `C12-1079` `IP` sibling constraint-revision proofs
/// + asymmetric Heyting hypernym/DR implication in `Q16.16`, reusing a per-word reverse cache.
fn score_abductive_gain_q16(
    ix: &Index,
    ns: usize,
    q_specs: &[QueryAtomSpec],
    turn_atoms: &[String],
    rev_cache: &mut DetHashMap<String, BTreeMap<String, i64>>,
) -> i64 {
    if q_specs.is_empty() || turn_atoms.is_empty() {
        return 0;
    }

    // Precompute per-turn atom metadata and check whether turn carries any specific support
    // (required to unblock `CU`-generic predicates per C12-1079 Sec. 3.3).
    let mut t_meta: Vec<(&str, String, i64, bool)> = Vec::with_capacity(turn_atoms.len().min(28));
    for tw in turn_atoms.iter().take(28) {
        let ic = word_ic_q16(ix, tw);
        let generic = is_cu_generic_predicate(ix, ns, tw, ic);
        t_meta.push((tw.as_str(), stem(tw), ic, generic));
    }

    let has_specific_support = q_specs.iter().any(|qs| {
        !qs.cu_generic
            && t_meta.iter().any(|&(tw, ref ts, _, t_gen)| {
                !t_gen
                    && (tw == qs.word
                        || *ts == qs.stem_word
                        || qs.fwd_factors.contains_key(tw)
                        || qs.ip_siblings.contains(tw))
            })
    });

    // Ensure 1-hop reverse expansions for non-generic turn atoms are populated in cache.
    for &(tw, _, _, t_gen) in &t_meta {
        if !t_gen && !rev_cache.contains_key(tw) {
            rev_cache.insert(tw.to_string(), expand_one_hop_factors(tw));
        }
    }

    let mut total_gain: i64 = 0;

    for qs in q_specs {
        let c0 = qs.cost0_q16;
        let mut best_cost = c0;
        let mut best_sib_bonus: i64 = 0;

        for &(tw, ref ts, t_ic, t_gen) in &t_meta {
            // C12-1079 Conditional Unification (CU) guard: generic predicates cannot unify
            // without specific modifier/entity support.
            if (qs.cu_generic || t_gen) && !has_specific_support {
                continue;
            }

            // 1. Exact or stem match: zero proof cost (full c0 gain).
            if tw == qs.word || (ts.len() >= 4 && *ts == qs.stem_word) {
                best_cost = 0;
                continue;
            }

            // 2. Asymmetric Heyting specific=>general hypernym implication (`cello => instrument`).
            if !t_gen && !qs.cu_generic {
                let heyt = heyting_implication_ic_q16(tw, &qs.word);
                if heyt >= 72 {
                    let h_cost = q16_mul(c0, HEYTING_SPEC_FACTOR_Q16);
                    if h_cost < best_cost {
                        best_cost = h_cost;
                    }
                }
            }

            // 3. Direct 1-hop backchain proof `q -> t`.
            if let Some(&fac) = qs.fwd_factors.get(tw) {
                let p_cost = q16_mul(c0, fac);
                if p_cost < best_cost {
                    best_cost = p_cost;
                }
            }

            // 4. Bounded 2-hop meet-in-the-middle proof `q -> m <- t`.
            if !t_gen
                && !qs.fwd_factors.is_empty()
                && let Some(t_rev) = rev_cache.get(tw)
            {
                for (mid, &f_tm) in t_rev {
                    if let Some(&f_qm) = qs.fwd_factors.get(mid) {
                        // Combined multiplicative factor with mild 2-hop attenuation (+0.15).
                        let comb = q16_mul(f_qm, f_tm).saturating_add(0x0000_2666);
                        if comb < ONE {
                            let p_cost = q16_mul(c0, comb);
                            if p_cost < best_cost {
                                best_cost = p_cost;
                            }
                        }
                    }
                }
            }

            // 5. C12-1079 Incompatible Property (IP) WordNet sibling/antonym constraint proof.
            if !t_gen && qs.ic_q16 >= 2 * ONE && qs.ip_siblings.contains(tw) {
                let bonus = q16_mul(GAMMA_SIB_Q16, qs.ic_q16.min(t_ic));
                if bonus > best_sib_bonus {
                    best_sib_bonus = bonus;
                }
            }
        }

        total_gain = total_gain
            .saturating_add((c0 - best_cost).max(0))
            .saturating_add(best_sib_bonus);
    }

    total_gain.min(24 * ONE)
}

// =========================================================================
// ALGORITHM 2: PEYRARD UNIFYING KL-IMPORTANCE + SALMA MODIFIER-HEAD BIGRAMS
// =========================================================================

struct SessionPeyrardModel {
    /// Unit key -> `log2(P_{D/K}(w)) = alpha*log2(P_D) - beta*log2(P_K) - log2(C_D)` in `Q16.16`.
    log_p_dk_q16: BTreeMap<String, i64>,
    /// Fallback log probability for unseen units.
    fallback_log_p_dk_q16: i64,
}

fn build_session_peyrard_model(
    ix: &Index,
    session_units: &[TurnLexUnits],
) -> SessionPeyrardModel {
    let mut sess_counts: BTreeMap<String, u32> = BTreeMap::new();
    let mut total_d: u32 = 0;

    for tu in session_units {
        for (unit, &cnt) in &tu.units {
            *sess_counts.entry(unit.clone()).or_insert(0) += cnt;
            total_d = total_d.saturating_add(cnt);
        }
    }
    total_d = total_d.max(1);
    let log2_total_d = log2_q16(total_d);
    let n_turns = (ix.turns.len() as u32).max(4);
    let log2_k_total = log2_q16(n_turns.saturating_mul(16).max(64));

    // Compute unnormalized log importance `z_w = alpha*log2(P_D(w)) - beta*log2(P_K(w))`.
    let mut raw_z: BTreeMap<String, i64> = BTreeMap::new();
    let mut z_max: i64 = i64::MIN;

    for (unit, &c_d) in &sess_counts {
        let log_pd = log2_q16(c_d.max(1)) - log2_total_d;
        let log_pk = if unit == STOP_BUCKET_KEY {
            // Stopwords/fillers occupy ~50% of background English probability mass.
            -ONE
        } else if let Some((m, h)) = unit.split_once('\u{2}') {
            // SALMA 2026 modifier-head bigram background rate: bounded by min(df(m), df(h)) / 2.
            let df_m = ix
                .vocab
                .get(m)
                .and_then(|id| ix.t_own_stats.df.get(&id).copied())
                .unwrap_or(1);
            let df_h = ix
                .vocab
                .get(h)
                .and_then(|id| ix.t_own_stats.df.get(&id).copied())
                .unwrap_or(1);
            let est_bg = (df_m.min(df_h) / 2).max(1);
            log2_q16(est_bg) - log2_k_total
        } else {
            let df = ix
                .vocab
                .get(unit.as_str())
                .and_then(|id| ix.t_own_stats.df.get(&id).copied())
                .unwrap_or(1);
            log2_q16(df.max(1)) - log2_k_total
        };

        let z = q16_mul(PEYRARD_ALPHA_Q16, log_pd) - q16_mul(PEYRARD_BETA_Q16, log_pk);
        if z > z_max {
            z_max = z;
        }
        raw_z.insert(unit.clone(), z);
    }

    if raw_z.is_empty() {
        return SessionPeyrardModel {
            log_p_dk_q16: BTreeMap::new(),
            fallback_log_p_dk_q16: -8 * ONE,
        };
    }

    // Exact integer log2(C_D) = z_max + log2(sum_u 2^(z_u - z_max)) where exp2_neg_q16 is 2^16-scaled.
    let mut sum_exp: u64 = 0;
    for &z in raw_z.values() {
        sum_exp = sum_exp.saturating_add(exp2_neg_q16(z - z_max));
    }
    let log2_c = z_max + log2_u64_q16_signed(sum_exp.max(1)) - (16 * ONE);

    let mut log_p_dk_q16: BTreeMap<String, i64> = BTreeMap::new();
    let mut min_log_p: i64 = 0;
    for (unit, z) in raw_z {
        let lp = (z - log2_c).min(0);
        if lp < min_log_p {
            min_log_p = lp;
        }
        log_p_dk_q16.insert(unit, lp);
    }

    SessionPeyrardModel {
        log_p_dk_q16,
        fallback_log_p_dk_q16: min_log_p.saturating_sub(ONE).max(-16 * ONE),
    }
}

/// Peyrard (ACL 2019 P19-1101) Eq. (5) negative KL divergence `-KL(P_T || P_{D/K})` in `Q16.16`
/// augmented with `SALMA 2026` adjacent modifier-head bigrams.
fn peyrard_kl_salience_q16(tu: &TurnLexUnits, model: &SessionPeyrardModel) -> i64 {
    if tu.total_units == 0 || tu.units.is_empty() {
        return -10 * ONE;
    }
    let total_t = tu.total_units;
    let log2_t = log2_q16(total_t);
    let mut theta: i64 = 0;

    for (unit, &c_t) in &tu.units {
        let p_t_q16 = q16_div(c_t as i64, total_t as i64);
        let log_pt = log2_q16(c_t.max(1)) - log2_t;
        let log_pdk = model
            .log_p_dk_q16
            .get(unit)
            .copied()
            .unwrap_or(model.fallback_log_p_dk_q16);
        let diff = log_pdk - log_pt;
        theta = theta.saturating_add(q16_mul(p_t_q16, diff));
    }

    theta.clamp(-16 * ONE, 4 * ONE)
}

// =========================================================================
// ALGORITHM 3: BIPARTITE ENTITY-TURN COHERENCE + RST ONO/MARCU FILTER
// =========================================================================

/// Build Centering role-weighted `(entity_id, role_weight_q16)` lists (`S=3, O=2, X=1`)
/// for every turn in a session (`Guinaudeau & Strube, ACL 2013 P13-1010`).
fn build_session_entity_roles(ix: &Index, s_turns: &[usize]) -> Vec<Vec<(u32, i64)>> {
    let mut out = Vec::with_capacity(s_turns.len());
    for &ti in s_turns {
        let tr = &ix.turns[ti];
        let mut role_map: BTreeMap<u32, i64> = BTreeMap::new();
        let seq_len = tr.seq.len();
        let subj_cut = (seq_len / 3).max(2);
        let obj_cut = ((seq_len * 2) / 3).max(subj_cut + 1);

        for (pos, &(exact_id, _stem_id)) in tr.seq.iter().enumerate() {
            if let Some(name) = ix.vocab.names.get(exact_id as usize)
                && is_closed_wrapper(name)
            {
                continue;
            }
            let is_proper = tr.entities.contains(&exact_id);
            let w = if is_proper || pos < subj_cut {
                ROLE_SUBJECT_Q16
            } else if pos < obj_cut {
                ROLE_OBJECT_Q16
            } else {
                ROLE_OTHER_Q16
            };
            let slot = role_map.entry(exact_id).or_insert(0);
            if w > *slot {
                *slot = w;
            }
        }
        // Ensure capitalized proper entities not in `seq` also carry Subject salience.
        for &eid in &tr.entities {
            let slot = role_map.entry(eid).or_insert(0);
            if ROLE_SUBJECT_Q16 > *slot {
                *slot = ROLE_SUBJECT_Q16;
            }
        }
        out.push(role_map.into_iter().collect());
    }
    out
}

/// Distance-decayed one-mode bipartite projection centrality multiplied by RST factor (`Q16.16`).
fn coherence_rst_score_q16(
    local_idx: usize,
    turns_entities: &[Vec<(u32, i64)>],
    rst_mult_q16: i64,
) -> i64 {
    let ti = &turns_entities[local_idx];
    if ti.is_empty() {
        return 0;
    }
    let mut out_deg: i64 = 0;
    for (k_idx, tk) in turns_entities.iter().enumerate() {
        if k_idx == local_idx || tk.is_empty() {
            continue;
        }
        let dist = local_idx.abs_diff(k_idx) as i64;
        if dist > 16 {
            continue;
        }
        let mut shared_w: i64 = 0;
        let (mut i, mut j) = (0usize, 0usize);
        while i < ti.len() && j < tk.len() {
            match ti[i].0.cmp(&tk[j].0) {
                std::cmp::Ordering::Equal => {
                    shared_w = shared_w.saturating_add(q16_mul(ti[i].1, tk[j].1));
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        out_deg = out_deg.saturating_add(shared_w / dist.max(1));
    }

    // Normalize high raw sums (`S*S = 9 * ONE`) into a well-scaled `[0, 8 * ONE]` range.
    let norm_deg = (out_deg / 3).min(8 * ONE);
    let eff_deg = if norm_deg == 0 && rst_mult_q16 > RST_NEUTRAL_Q16 {
        // Standalone Marcu nucleus turn with concrete entities in a minimal 1-2 turn session.
        ONE / 2
    } else {
        norm_deg
    };
    q16_mul(eff_deg, rst_mult_q16)
}

// =========================================================================
// ALGORITHM 4B: COMPUTABLE SPERBER-WILSON RELEVANCE RATIO
// =========================================================================

fn mean_content_surprisal_q16(ix: &Index, content_tokens: &[String]) -> i64 {
    if content_tokens.is_empty() {
        return 0;
    }
    let mut sum_ic: i64 = 0;
    let mut cnt: i64 = 0;
    for tok in content_tokens.iter().take(32) {
        sum_ic = sum_ic.saturating_add(word_ic_q16(ix, tok));
        cnt += 1;
    }
    if cnt == 0 {
        0
    } else {
        sum_ic / cnt
    }
}

fn sperber_wilson_relevance_q16(
    abd_gain_q16: i64,
    mean_surprisal_q16: i64,
    ncs_q16: i64,
    len_tokens: u32,
    stop_ratio_q16: i64,
) -> i64 {
    let effect = abd_gain_q16
        .saturating_add(q16_mul(LAMBDA_H_Q16, mean_surprisal_q16))
        .saturating_add(q16_mul(LAMBDA_NCD_Q16, ncs_q16))
        .max(0);
    let effort = EFFORT_EPS_Q16
        .saturating_add(q16_mul(LAMBDA_LEN_Q16, log2_q16(1 + len_tokens)))
        .saturating_add(q16_mul(LAMBDA_STOP_Q16, stop_ratio_q16.clamp(0, ONE)))
        .max(ONE / 4);
    q16_div(effect, effort).clamp(0, 24 * ONE)
}

// =========================================================================
// ALGORITHM 5: SCOPED TOP-5 SESSION PRAGMATIC SEATING ENTRY POINT
// =========================================================================

/// Compute the bitmask (`1 << g`) of the query's lowest-DF exact non-stopword, non-wrapper
/// content group (`star_bit`) so declarative factual winners are never demoted.
fn compute_star_bit(ix: &Index, plan: &QueryPlan) -> u64 {
    let mut best: Option<(u32, usize)> = None;
    for qt in &plan.terms {
        if !qt.exact_hit {
            continue;
        }
        let Some(g) = qt.group else {
            continue;
        };
        if g >= 64 {
            continue;
        }
        if let Some(name) = ix.vocab.names.get(qt.id as usize)
            && is_closed_wrapper(name)
        {
            continue;
        }
        let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0);
        if df == 0 {
            continue;
        }
        if best.is_none_or(|(b_df, b_g)| df < b_df || (df == b_df && g < b_g)) {
            best = Some((df, g));
        }
    }
    best.map_or(0, |(_, g)| 1u64 << g)
}

struct AbductSeatView<'a> {
    ix: &'a Index,
    query: &'a str,
    plan: &'a QueryPlan,
    eff_it: &'a QueryIntent,
    pref_on: bool,
    k: usize,
    ns: usize,
    is_monologue_corpus: bool,
    torrens_locked_head: bool,
    implicit_disclosure_locked: bool,
    q_group_cnt: usize,
    q_group_mask: u64,
    t_exact_mask: &'a [u64],
    final_session_indices: &'a [usize],
}

fn apply_abductive_pragmatic_seating_view(
    view: &AbductSeatView<'_>,
    final_turn_indices: &mut Vec<usize>,
) {
    if final_turn_indices.len() < 2
        || view.final_session_indices.is_empty()
        || view.k < 2
        || view.ix.has_tool_calls
        || view.is_monologue_corpus
        || view.implicit_disclosure_locked
        || view.torrens_locked_head
        || view.ns > 24
        || c2_named_session(view.ix, final_turn_indices[0])
    {
        return;
    }

    if view.eff_it.assistant_attribution
        || view.eff_it.is_event_sequence
        || view.eff_it.sniper_how_many
        || eff_summary_request(view.query)
        || is_factual_past_lookup_query(view.query)
    {
        return;
    }

    let h0 = final_turn_indices[0];
    let h0_exact = view.t_exact_mask.get(h0).copied().unwrap_or(0) & view.q_group_mask;
    let low_exact_overlap =
        view.q_group_cnt > 0 && (h0_exact.count_ones() as usize) * 2 < view.q_group_cnt;
    let is_imp_adv = is_imperative_or_advisory_frame(view.query)
        || view.pref_on
        || is_implicit_constraint_planning_query(view.query, view.eff_it, view.pref_on);

    if !is_imp_adv && !low_exact_overlap {
        return;
    }

    let top_sessions: &[usize] =
        &view.final_session_indices[..view.final_session_indices.len().min(5)];
    let q_specs = build_query_abduct_specs(view.ix, view.ns, view.query);
    let q_atoms: Vec<String> = q_specs.iter().map(|s| s.word.clone()).collect();
    let q_stream = build_canonical_synset_stream(&q_atoms);

    let mut rev_cache: DetHashMap<String, BTreeMap<String, i64>> = DetHashMap::default();
    let mut score_by_turn: DetHashMap<usize, AbductiveTurnScore> = DetHashMap::default();
    let mut per_session_best: Vec<AbductiveTurnScore> = Vec::with_capacity(top_sessions.len() + 1);
    let mut s0_second_best: Option<AbductiveTurnScore> = None;

    for (rank_pos, &si) in top_sessions.iter().enumerate() {
        let Some(s_turns) = view.ix.session_turns.get(si) else {
            continue;
        };
        if s_turns.is_empty() {
            continue;
        }
        let session_units: Vec<TurnLexUnits> = s_turns
            .iter()
            .map(|&ti| extract_turn_lex_units(&view.ix.turns[ti].body_for_headers))
            .collect();
        let peyrard_model = build_session_peyrard_model(view.ix, &session_units);
        let entity_roles = build_session_entity_roles(view.ix, s_turns);

        // Mild session-rank prior so top-ranked sessions win close ties.
        let sess_prior_q16 = match rank_pos {
            0 => ONE / 2,
            1 => ONE / 4,
            2 => ONE / 8,
            _ => 0,
        };

        let mut sess_cands: Vec<AbductiveTurnScore> = Vec::new();

        for (local_idx, &ti) in s_turns.iter().enumerate() {
            let tr = &view.ix.turns[ti];
            let tu = &session_units[local_idx];
            let peyrard_q16 = peyrard_kl_salience_q16(tu, &peyrard_model);

            // Only declarative, non-backchannel, non-superseded user turns may be promoted.
            if tr.field != 0
                || tr.is_backchannel
                || tr.is_question
                || turn_is_superseded(view.ix, ti)
            {
                score_by_turn.insert(
                    ti,
                    AbductiveTurnScore {
                        turn_idx: ti,
                        session_idx: si,
                        abd_gain_q16: 0,
                        peyrard_q16,
                        coherence_rst_q16: 0,
                        ncs_q16: 0,
                        sw_rel_q16: 0,
                        total_q16: peyrard_q16.saturating_sub(2 * ONE),
                    },
                );
                continue;
            }

            let t_atoms = extract_substantive_atoms(&tr.body_for_headers);
            let abd_gain_q16 =
                score_abductive_gain_q16(view.ix, view.ns, &q_specs, &t_atoms, &mut rev_cache);
            let rst_mult_q16 = rst_nucleus_multiplier_q16(&tr.body_for_headers);
            let coherence_rst_q16 = coherence_rst_score_q16(local_idx, &entity_roles, rst_mult_q16);
            let mean_surp_q16 = mean_content_surprisal_q16(view.ix, &tu.content_tokens);
            let stop_ratio_q16 = if tu.raw_len == 0 {
                ONE
            } else {
                q16_div(tu.stop_or_wrapper_len as i64, tu.raw_len as i64)
            };
            let t_stream = build_canonical_synset_stream(&t_atoms);
            let ncs_q16 = ncd_similarity_q16(&q_stream, &t_stream);
            let sw_rel_q16 = sperber_wilson_relevance_q16(
                abd_gain_q16,
                mean_surp_q16,
                ncs_q16,
                tu.raw_len,
                stop_ratio_q16,
            );

            let closure_pen = if is_discourse_closure_or_meta_echo(view.ix, ti, "") {
                3 * ONE
            } else {
                0
            };

            let total_q16 = q16_mul(FUSE_W_SW_Q16, sw_rel_q16)
                .saturating_add(q16_mul(FUSE_W_PEYRARD_Q16, peyrard_q16))
                .saturating_add(q16_mul(FUSE_W_COH_Q16, coherence_rst_q16))
                .saturating_add(q16_mul(FUSE_W_ABD_Q16, abd_gain_q16))
                .saturating_add(sess_prior_q16)
                .saturating_sub(closure_pen);

            let rec = AbductiveTurnScore {
                turn_idx: ti,
                session_idx: si,
                abd_gain_q16,
                peyrard_q16,
                coherence_rst_q16,
                ncs_q16,
                sw_rel_q16,
                total_q16,
            };
            score_by_turn.insert(ti, rec);

            // Require genuine abductive/NCD or dense salient coherence evidence before nominating.
            let has_evidence = rec.abd_gain_q16 >= (ONE / 2)
                || rec.ncs_q16 >= (ONE / 4)
                || (rec.coherence_rst_q16 >= (ONE / 2)
                    && rec.peyrard_q16 >= PEYRARD_PROTECT_FLOOR_Q16);
            if has_evidence && !is_discourse_closure_or_meta_echo(view.ix, ti, "") {
                sess_cands.push(rec);
            }
        }

        sess_cands.sort_by(|a, b| {
            b.total_q16
                .cmp(&a.total_q16)
                .then_with(|| b.sw_rel_q16.cmp(&a.sw_rel_q16))
                .then_with(|| b.abd_gain_q16.cmp(&a.abd_gain_q16))
                .then_with(|| b.peyrard_q16.cmp(&a.peyrard_q16))
                .then_with(|| a.turn_idx.cmp(&b.turn_idx))
        });

        if let Some(&first) = sess_cands.first() {
            per_session_best.push(first);
            if rank_pos == 0
                && let Some(&second) = sess_cands.get(1)
                && second.abd_gain_q16 >= ONE
            {
                s0_second_best = Some(second);
            }
        }
    }

    if per_session_best.is_empty() {
        return;
    }
    if let Some(sec) = s0_second_best {
        per_session_best.push(sec);
    }
    per_session_best.sort_by(|a, b| {
        b.total_q16
            .cmp(&a.total_q16)
            .then_with(|| b.sw_rel_q16.cmp(&a.sw_rel_q16))
            .then_with(|| b.abd_gain_q16.cmp(&a.abd_gain_q16))
            .then_with(|| b.peyrard_q16.cmp(&a.peyrard_q16))
            .then_with(|| a.turn_idx.cmp(&b.turn_idx))
    });

    // Protect Slot 0 if it already carries a standing policy OR a declarative `star_bit` factual
    // match with solid Peyrard salience (never demote a clean factual winner!).
    let star_bit = compute_star_bit(view.ix, view.plan);
    let h0_tr = &view.ix.turns[h0];
    let h0_has_star =
        star_bit != 0 && (view.t_exact_mask.get(h0).copied().unwrap_or(0) & star_bit) != 0;
    let h0_is_decl = h0_tr.field == 0
        && !h0_tr.is_question
        && !h0_tr.is_backchannel
        && !turn_is_superseded(view.ix, h0)
        && !is_discourse_closure_or_meta_echo(view.ix, h0, "");
    let h0_score = score_by_turn.get(&h0).copied();
    let h0_peyrard_ok = h0_score.is_none_or(|s| s.peyrard_q16 >= PEYRARD_PROTECT_FLOOR_Q16);
    let protect_slot0 = h0_tr.is_standing_policy || (h0_has_star && h0_is_decl && h0_peyrard_ok);

    // Track declarative Top-10 turns carrying `star_bit` so no non-star candidate ever jumps ahead
    // of a declarative `star_bit` winner on non-advisory queries.
    let top10_star_turns: DetHashSet<usize> = if star_bit == 0 {
        DetHashSet::default()
    } else {
        final_turn_indices
            .iter()
            .take(10)
            .copied()
            .filter(|&ti| {
                let tr = &view.ix.turns[ti];
                tr.field == 0
                    && !tr.is_question
                    && !tr.is_backchannel
                    && !turn_is_superseded(view.ix, ti)
                    && (view.t_exact_mask.get(ti).copied().unwrap_or(0) & star_bit) != 0
            })
            .collect()
    };

    let orig_tlen = final_turn_indices.len();
    let top_cap = 5usize.min(view.k).min(orig_tlen);
    let s0_top = top_sessions[0];
    let mut next_slot = if protect_slot0 { 1usize } else { 0usize };
    let mut seated: DetHashSet<usize> = DetHashSet::default();
    let mut seated_per_sess: DetHashMap<usize, usize> = DetHashMap::default();
    if protect_slot0 {
        seated.insert(h0);
        *seated_per_sess.entry(h0_tr.session).or_insert(0) += 1;
    }

    for cand in per_session_best {
        if next_slot >= top_cap {
            break;
        }
        if !seated.insert(cand.turn_idx) {
            continue;
        }
        let sess_cap = if cand.session_idx == s0_top { 2 } else { 1 };
        let used_in_sess = seated_per_sess.get(&cand.session_idx).copied().unwrap_or(0);
        if used_in_sess >= sess_cap {
            continue;
        }
        let cand_has_star = star_bit != 0
            && (view.t_exact_mask.get(cand.turn_idx).copied().unwrap_or(0) & star_bit) != 0;
        if !is_imp_adv && !top10_star_turns.is_empty() && !cand_has_star {
            continue;
        }

        // Check incumbent at `next_slot`: only replace if incumbent is a weaker turn / filler /
        // question / assistant echo, OR candidate strictly dominates on pragmatic total score.
        let incumbent = final_turn_indices[next_slot];
        if incumbent == cand.turn_idx {
            *seated_per_sess.entry(cand.session_idx).or_insert(0) += 1;
            next_slot += 1;
            continue;
        }
        if top10_star_turns.contains(&incumbent) && !cand_has_star {
            next_slot += 1;
            continue;
        }
        let inc_tr = &view.ix.turns[incumbent];
        let inc_is_weak = inc_tr.field != 0
            || inc_tr.is_question
            || inc_tr.is_backchannel
            || turn_is_superseded(view.ix, incumbent)
            || is_discourse_closure_or_meta_echo(view.ix, incumbent, "")
            || score_by_turn
                .get(&incumbent)
                .is_none_or(|s| cand.total_q16 > s.total_q16.saturating_add(ONE / 4));

        if !inc_is_weak {
            next_slot += 1;
            continue;
        }

        if let Some(pos) = final_turn_indices.iter().position(|&t| t == cand.turn_idx) {
            if pos < next_slot {
                continue;
            }
            final_turn_indices.remove(pos);
        }
        let ins = next_slot.min(final_turn_indices.len());
        final_turn_indices.insert(ins, cand.turn_idx);
        *seated_per_sess.entry(cand.session_idx).or_insert(0) += 1;
        next_slot += 1;
    }

    final_turn_indices.truncate(orig_tlen);
}

/// Public Stage-12 intra-session abductive & pragmatic turn seating pass (`Q16.16`).
/// Strictly scoped inside `ctx.final_session_indices[0..5]` on compact multi-session dialogues
/// (`ns <= 24`), protecting `implicit_disclosure_locked`, `torrens_locked_head`, and `star_bit`
/// factual Slot-0 winners.
pub(in crate::retrieval) fn apply_abductive_pragmatic_seating(ctx: &mut RankCtx<'_>) {
    let view = AbductSeatView {
        ix: ctx.ix,
        query: &ctx.q.text,
        plan: &ctx.plan,
        eff_it: &ctx.eff_it,
        pref_on: ctx.pref_on,
        k: ctx.k,
        ns: ctx.ns,
        is_monologue_corpus: ctx.is_monologue_corpus,
        torrens_locked_head: ctx.torrens_locked_head,
        implicit_disclosure_locked: ctx.implicit_disclosure_locked,
        q_group_cnt: ctx.q_group_cnt,
        q_group_mask: ctx.q_group_mask,
        t_exact_mask: &ctx.t_exact_mask,
        final_session_indices: &ctx.final_session_indices,
    };
    apply_abductive_pragmatic_seating_view(&view, &mut ctx.final_turn_indices);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Query, Session, Turn, build_index, plan_query};

    fn mk_turn(id: &str, role: &str, content: &str) -> Turn {
        Turn {
            id: id.to_string(),
            role: role.to_string(),
            content: content.to_string(),
            links: Vec::new(),
            stability: 0.0,
        }
    }

    fn mk_session(id: &str, ts: i64, turns: Vec<Turn>) -> Session {
        Session {
            id: id.to_string(),
            date: String::new(),
            timestamp: ts,
            turns,
        }
    }

    #[test]
    fn peyrard_kl_with_salma_modifier_head_beats_zero_ic_filler() {
        let sessions = vec![
            mk_session(
                "s0",
                1_700_000_000,
                vec![
                    mk_turn(
                        "t0",
                        "user",
                        "Adding notes for the Garcia family. Their top priority is a separate guest house or permitted ADU.",
                    ),
                    mk_turn(
                        "t1",
                        "assistant",
                        "Noted the separate guest house and permitted ADU requirement for the Garcia family.",
                    ),
                    mk_turn(
                        "t2",
                        "user",
                        "I'm not sure. This has been a theme for a long time.",
                    ),
                    mk_turn(
                        "t3",
                        "assistant",
                        "Let me know when you want to revisit that.",
                    ),
                ],
            ),
            mk_session(
                "s1",
                1_700_086_400,
                vec![
                    mk_turn("t4", "user", "We reviewed quarterly tax brackets and depreciation."),
                    mk_turn("t5", "assistant", "Happy to organize the tax documents."),
                ],
            ),
        ];
        let ix = build_index(&sessions, &Params::default());
        let s0_turns = &ix.session_turns[0];
        let units: Vec<TurnLexUnits> = s0_turns
            .iter()
            .map(|&ti| extract_turn_lex_units(&ix.turns[ti].body_for_headers))
            .collect();
        let model = build_session_peyrard_model(&ix, &units);

        let salient = peyrard_kl_salience_q16(&units[0], &model);
        let filler = peyrard_kl_salience_q16(&units[2], &model);

        assert!(
            filler < -2 * ONE,
            "expected zero-IC filler turn to score strongly negative under Peyrard KL, got {filler}"
        );
        assert!(
            salient > filler + ONE,
            "expected SALMA modifier-head turn ({salient}) to beat zero-IC filler ({filler}) decisively"
        );
    }

    #[test]
    fn lz77_ncd_is_symmetric_and_separates_shared_synsets_from_disjoint() {
        let rep = b"bakery pastry sugar cream pie bakery pastry sugar cream pie";
        assert!(
            lz77_len(rep) < rep.len() * 8,
            "expected LZ77 to compress repeated substring"
        );
        assert_eq!(lz77_len(b""), 0);

        let q_atoms = extract_substantive_atoms("Recommend a weekend baking recipe");
        let rel_atoms =
            extract_substantive_atoms("The neighborhood bakery makes an incredible sugar cream pie");
        let dis_atoms =
            extract_substantive_atoms("Calibrating the orbital spectrometer telescope mirror");

        let q_stream = build_canonical_synset_stream(&q_atoms);
        let rel_stream = build_canonical_synset_stream(&rel_atoms);
        let dis_stream = build_canonical_synset_stream(&dis_atoms);

        let sim_ab = ncd_similarity_q16(&q_stream, &rel_stream);
        let sim_ba = ncd_similarity_q16(&rel_stream, &q_stream);
        assert_eq!(sim_ab, sim_ba, "NCS must be strictly symmetric");

        let sim_dis = ncd_similarity_q16(&q_stream, &dis_stream);
        let sim_self = ncd_similarity_q16(&q_stream, &q_stream);
        assert!(
            sim_self > (ONE * 3) / 4,
            "expected high self-NCS on identical stream, got {sim_self}"
        );
        assert!(
            sim_ab > sim_dis,
            "expected shared synset stream NCS ({sim_ab}) > disjoint stream NCS ({sim_dis})"
        );
    }

    #[test]
    fn bipartite_coherence_and_rst_penalize_ono_hedges_and_boost_marcu_nuclei() {
        assert_eq!(
            rst_nucleus_multiplier_q16("I'm not sure, maybe I guess it has been a theme."),
            RST_ONO_HEDGE_Q16
        );
        assert_eq!(
            rst_nucleus_multiplier_q16(
                "Their top priority is a permitted ADU because they emphasized privacy."
            ),
            RST_MARCU_NUCLEUS_Q16
        );
        assert_eq!(
            rst_nucleus_multiplier_q16("We toured three properties near the river."),
            RST_NEUTRAL_Q16
        );

        let sessions = vec![mk_session(
            "s0",
            1_700_000_000,
            vec![
                mk_turn(
                    "t0",
                    "user",
                    "Garcia family notes: their top priority is a separate guest house because they emphasized privacy.",
                ),
                mk_turn(
                    "t1",
                    "assistant",
                    "Got the Garcia family separate guest house priority recorded.",
                ),
                mk_turn(
                    "t2",
                    "user",
                    "I'm not sure. Maybe this has been a theme for a long time.",
                ),
            ],
        )];
        let ix = build_index(&sessions, &Params::default());
        let s0 = &ix.session_turns[0];
        let roles = build_session_entity_roles(&ix, s0);

        let nucleus_coh = coherence_rst_score_q16(
            0,
            &roles,
            rst_nucleus_multiplier_q16(&ix.turns[s0[0]].body_for_headers),
        );
        let hedge_coh = coherence_rst_score_q16(
            2,
            &roles,
            rst_nucleus_multiplier_q16(&ix.turns[s0[2]].body_for_headers),
        );

        assert!(
            nucleus_coh > hedge_coh + ONE,
            "expected entity-linked Marcu nucleus ({nucleus_coh}) >> isolated Ono hedge ({hedge_coh})"
        );
    }

    #[test]
    fn hobbs_abduction_ip_sibling_constraint_and_cu_generic_block() {
        let sessions = vec![
            mk_session(
                "s0",
                1_700_000_000,
                vec![
                    mk_turn("t0", "user", "My sister plays cello in a chamber quartet."),
                    mk_turn("t1", "user", "We moved the meeting to Tuesday afternoon."),
                    mk_turn(
                        "t2",
                        "user",
                        "I'm not sure. This has been a theme for a long time.",
                    ),
                ],
            ),
            mk_session(
                "s1",
                1_700_086_400,
                vec![mk_turn(
                    "t3",
                    "user",
                    "Let's discuss the general theme over a long time.",
                )],
            ),
            mk_session(
                "s2",
                1_700_172_800,
                vec![mk_turn(
                    "t4",
                    "user",
                    "Another general theme came up over time.",
                )],
            ),
            mk_session(
                "s3",
                1_700_259_200,
                vec![mk_turn(
                    "t5",
                    "user",
                    "That theme repeats every single time.",
                )],
            ),
        ];
        let ix = build_index(&sessions, &Params::default());
        let ns = sessions.len();
        let mut cache: DetHashMap<String, BTreeMap<String, i64>> = DetHashMap::default();

        // 1. C12-1079 IP sibling/antonym constraint proof (`brother` <-> `sister`, `wednesday` <-> `tuesday`).
        let q_sib = build_query_abduct_specs(&ix, ns, "Check my brother schedule for Wednesday");
        let t_sib_atoms = extract_substantive_atoms("My sister moved the meeting to Tuesday");
        let t_filler_atoms =
            extract_substantive_atoms("I'm not sure. This has been a theme for a long time.");
        let gain_sib = score_abductive_gain_q16(&ix, ns, &q_sib, &t_sib_atoms, &mut cache);
        let gain_filler = score_abductive_gain_q16(&ix, ns, &q_sib, &t_filler_atoms, &mut cache);
        assert!(
            gain_sib > 0,
            "expected positive IP sibling/antonym abductive proof gain, got {gain_sib}"
        );
        assert_eq!(
            gain_filler, 0,
            "expected CU block to suppress zero-support filler unification, got {gain_filler}"
        );

        // 2. DF-based C12-1079 CU block: high-DF generic predicate across every session cannot
        //    anchor unification without specific modifier/entity support.
        let cu_sessions: Vec<Session> = (0..5)
            .map(|i| {
                mk_session(
                    &format!("cs{i}"),
                    1_700_000_000 + (i as i64) * 86_400,
                    vec![mk_turn(
                        &format!("ct{i}"),
                        "user",
                        "We reviewed the project status briefly.",
                    )],
                )
            })
            .collect();
        let cu_ix = build_index(&cu_sessions, &Params::default());
        let cu_ns = cu_sessions.len();
        assert!(
            is_cu_generic_predicate(&cu_ix, cu_ns, "project", word_ic_q16(&cu_ix, "project")),
            "expected high-DF cross-session noun 'project' to trigger CU generic block"
        );
        let q_cu = build_query_abduct_specs(&cu_ix, cu_ns, "Evaluate the project runway");
        let t_cu_only = extract_substantive_atoms("We logged the project status briefly.");
        let t_cu_spec = extract_substantive_atoms("The project runway requires disclosure.");
        let gain_cu_blocked = score_abductive_gain_q16(&cu_ix, cu_ns, &q_cu, &t_cu_only, &mut cache);
        let gain_cu_supported =
            score_abductive_gain_q16(&cu_ix, cu_ns, &q_cu, &t_cu_spec, &mut cache);
        assert_eq!(
            gain_cu_blocked, 0,
            "expected CU block to zero out unsupported high-DF predicate match, got {gain_cu_blocked}"
        );
        assert!(
            gain_cu_supported > 0,
            "expected specific modifier support to unblock CU unification, got {gain_cu_supported}"
        );

        // 3. Asymmetric Heyting specific=>general hypernym implication (`cello => instrument`).
        let q_inst = build_query_abduct_specs(&ix, ns, "Recommend maintenance for a musical instrument");
        let t_cello = extract_substantive_atoms("My sister plays cello in a chamber quartet.");
        let gain_cello = score_abductive_gain_q16(&ix, ns, &q_inst, &t_cello, &mut cache);
        assert!(
            gain_cello > 0,
            "expected positive Heyting/abductive proof for cello => instrument, got {gain_cello}"
        );
    }

    #[test]
    fn multibyte_utf8_safety_and_implicit_disclosure_and_star_bit_guards() {
        let sessions = vec![
            mk_session(
                "s0",
                1_700_000_000,
                vec![
                    mk_turn(
                        "t0",
                        "user",
                        "Café résumé notes for 東京 — I’m not sure. Maybe this has been a theme for a long time.",
                    ),
                    mk_turn(
                        "t1",
                        "user",
                        "Their top priority at the café in 東京 is a separate cottage or permitted ADU — because they emphasized privacy 🎉.",
                    ),
                    mk_turn(
                        "t2",
                        "user",
                        "We toured a 4-bedroom villa near the café in 東京 with cedar siding.",
                    ),
                ],
            ),
            mk_session(
                "s1",
                1_700_086_400,
                vec![mk_turn(
                    "t3",
                    "user",
                    "Reviewed unrelated insurance paperwork.",
                )],
            ),
        ];
        let params = Params::default();
        let ix = build_index(&sessions, &params);
        let q = Query {
            text: "Filter for 4-bedroom homes near the café in 東京".to_string(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &params);
        let q_group_cnt = plan.groups.len().min(64);
        let q_group_mask = if q_group_cnt == 64 {
            u64::MAX
        } else if q_group_cnt == 0 {
            0
        } else {
            (1u64 << q_group_cnt) - 1
        };

        // Mark turn 2 (`t2`) as matching `bedroom` / `villa` exact groups so `star_bit` is tested.
        let mut t_exact_mask = vec![0u64; ix.turns.len()];
        for (ti, tr) in ix.turns.iter().enumerate() {
            let mut m = 0u64;
            for qt in &plan.terms {
                if qt.exact_hit
                    && let Some(g) = qt.group
                    && g < 64
                    && tr.content.contains(&qt.id)
                {
                    m |= 1u64 << g;
                }
            }
            t_exact_mask[ti] = m;
        }

        // 1. `implicit_disclosure_locked = true` must be a strict no-op (zero mutation).
        let locked_view = AbductSeatView {
            ix: &ix,
            query: &q.text,
            plan: &plan,
            eff_it: &plan.intent,
            pref_on: false,
            k: 5,
            ns: sessions.len(),
            is_monologue_corpus: false,
            torrens_locked_head: false,
            implicit_disclosure_locked: true,
            q_group_cnt,
            q_group_mask,
            t_exact_mask: &t_exact_mask,
            final_session_indices: &[0, 1],
        };
        let mut locked_turns = vec![0, 3, 2, 1];
        apply_abductive_pragmatic_seating_view(&locked_view, &mut locked_turns);
        assert_eq!(
            locked_turns,
            vec![0, 3, 2, 1],
            "implicit_disclosure_locked must preserve exact turn ordering"
        );

        // 2. Unlocked UTF-8 run: zero panic, and filler turn 0 is replaced by salient turn 1 or 2.
        let unlocked_view = AbductSeatView {
            implicit_disclosure_locked: false,
            ..locked_view
        };
        let mut turns = vec![0, 3, 2, 1];
        apply_abductive_pragmatic_seating_view(&unlocked_view, &mut turns);
        assert_ne!(
            turns[0], 0,
            "expected zero-IC UTF-8 filler at Slot 0 to be replaced by salient constraint turn: {turns:?}"
        );

        // 3. When Slot 0 is already the declarative `star_bit` factual winner (`t2`), it stays locked at Slot 0.
        let mut star_turns = vec![2, 0, 3, 1];
        apply_abductive_pragmatic_seating_view(&unlocked_view, &mut star_turns);
        assert_eq!(
            star_turns[0], 2,
            "expected declarative star_bit factual Slot-0 winner to stay protected: {star_turns:?}"
        );
    }

    #[test]
    fn imperative_or_advisory_frame_triggers_without_question_mark_and_rejects_past_factual() {
        assert!(is_imperative_or_advisory_frame(
            "Pull up 4-bedroom listings for the Garcia family."
        ));
        assert!(is_imperative_or_advisory_frame(
            "Filter options for an autumn weekend getaway"
        ));
        assert!(is_imperative_or_advisory_frame(
            "Recommend a recipe for weekend baking"
        ));
        assert!(is_imperative_or_advisory_frame(
            "What should we emphasize in the financial update?"
        ));
        assert!(!is_imperative_or_advisory_frame(
            "When did I visit Denver?"
        ));
        assert!(!is_imperative_or_advisory_frame(
            "How many bedrooms did I mention?"
        ));
    }
}
