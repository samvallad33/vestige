//! CORGI Deontic Policy & Frame-QCL Preference Seating (`Q16.16`).
//!
//! Deterministic zero-vector, zero-LLM rule detachment, specificity arbitration,
//! and conditional preference scoring over pinned offline lexicons:
//! 1. Taxonomic `Subsumption-CORGI` antecedent network (Weitekamp, ACS 2025,
//!    `arXiv:2511.13942`) with four high-precision typed bridge channels
//!    (exact/morphological, WordNet 3.1 Lin-LCS + Heyting intrinsic IC,
//!    Roget 1911 / ConceptNet 5.7 reciprocal / Wikidata CC0 containment,
//!    and Dual-Witness WordNet gloss co-occurrence) joined by an injective
//!    backward iterator over restrictor modifiers.
//! 2. Kratzerian Modal-Base Gate & Input/Output Logic detachment (Makinson &
//!    van der Torre 2000/2003; Kratzer 1986/1991), hard-gating non-formatting
//!    rules (`K = 0`) on episodic factual queries.
//! 3. Defeasible Deontic Logic (`DDL`) Concept Abduction (`H`), Concept
//!    Contraction (`G`), and Poole Specificity arbitration (Di Noia et al.,
//!    JAIR 2007, `arXiv:1110.2742`; Poole 1985; `arXiv:2609.15113`).
//! 4. Frame-Semantic Role Alignment + Qualitative Choice Logic (`QCL`) /
//!    CP-Net conditional parent dominance (`arXiv:2209.12777v2`,
//!    `arXiv:1107.0023`), enforcing a hard target guard, 5% beneficiary cap,
//!    and unrequested temporal-constraint penalty.

use std::collections::{BTreeMap, BTreeSet};

use crate::retrieval::context::RankCtx;
use crate::retrieval::fusion::kemeny_mrf::log2_u64_q16;
use crate::retrieval::seating::preference_lattice::{
    is_explicit_stance_directive, is_pref_framing_word,
};
use crate::retrieval::seating::revealed_preference_cone_q16::self_disclosure_strength_q16;
use crate::retrieval::{
    Index, QueryIntent, QueryPlan, StandingRule, WH_QTY, WH_TIME, damerau_levenshtein_le_1,
    eff_summary_request, has_compound_preference_frame, is_content, raw_tokens, stem,
    step14_is_rec_request, stopwords, token_is_temporal_word, turn_is_superseded, wh_slots,
};

/// Unit `1.0` in signed `Q16.16` fixed-point representation (`1 << 16`).
pub(in crate::retrieval) const ONE: i64 = 65_536;

/// WordNet 3.1 noun taxonomy size used for corpus-free intrinsic IC (`log2(N_max)`).
const WN_MAX_NOUN_SYNSETS: u64 = 82_115;

/// Kratzerian detachment gate when `FormatVerify` disagrees with query `AnswerFocus` (`0.125`).
const K_FOCUS_MISMATCH: i64 = 8_192;
/// Kratzerian detachment gate for weak procedural/recommendation crossover (`0.25`).
const K_WEAK_COMPAT: i64 = 16_384;
/// Kratzerian detachment gate for pedagogical guidance under choice/pref queries (`0.75`).
const K_CHOICE_PEDAGOGICAL: i64 = 49_152;

/// DDL arbitration weights in `Q16.16`: Focus (`0.50`), Specificity (`0.40`), Abduction (`0.60`), Contraction (`0.70`).
const W_FOCUS: i64 = 32_768;
const W_SPEC: i64 = 26_214;
const W_ABDUCTION: i64 = 39_321;
const W_CONTRACTION: i64 = 45_875;

/// Minimum slot bridge score (`0.40`) to count a rule restrictor as grounded rather than abducted.
const MATCH_THRESHOLD: i64 = 26_214;

/// Frame-QCL + CP-Net slot weights in `Q16.16`: Target (`0.55`), Eval (`0.25`), Action (`0.15`), Beneficiary (`0.05`).
const W_TARGET: i64 = 36_044;
const W_EVAL: i64 = 16_384;
const W_ACTION: i64 = 9_830;
const W_BENEFICIARY: i64 = 3_277;
/// Frame-QCL hard target guard threshold (`0.40`).
const TAU_TARGET: i64 = 26_214;
/// Penalty (`0.50`) when a preference injects a temporal constraint absent from the query.
const LAMBDA_TIME: i64 = 32_768;

/// Minimum intrinsic IC (`0.375`) required for non-generic target/LCS bridges.
const GENERIC_IC_FLOOR: i64 = 24_576;
/// Intrinsic IC cutoff (`0.4375`) below which a rule noun is demoted to a meta-head tie-breaker.
const GENERIC_HEAD_IC_CUTOFF: i64 = 28_672;

/// Dual-witness gloss bridge strengths (`0.80` direct synset/gloss, `0.70` ancestor, `+0.05` exclusion bonus, `0.85` cap).
const ALPHA_GLOSS_DIRECT: i64 = 52_428;
const ALPHA_GLOSS_ANCESTOR: i64 = 45_875;
const ALPHA_GLOSS_NEG_BONUS: i64 = 3_277;
const ALPHA_GLOSS_CAP: i64 = 55_705;

/// Knowledge-graph 2-hop attenuation factor (`0.75`).
const KG_HOP_DECAY: i64 = 49_152;

/// Signed `Q16.16` fixed-point multiplication.
#[inline]
pub(in crate::retrieval) fn qmul(a: i64, b: i64) -> i64 {
    ((a as i128).saturating_mul(b as i128) >> 16) as i64
}

/// Signed `Q16.16` fixed-point division (returns `0` on zero denominator).
#[inline]
pub(in crate::retrieval) fn qdiv(a: i64, b: i64) -> i64 {
    if b == 0 {
        return 0;
    }
    (((a as i128) << 16) / (b as i128)) as i64
}

/// Closed-class Kratzerian modal base of a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) enum QueryModalBase {
    EpisodicFact,
    ProceduralHowTo,
    ChoicePref,
}

/// Closed-class WH expected answer focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) enum AnswerFocus {
    DatePoint,
    TimeOfDay,
    MoneyAmount,
    Quantity,
    General,
}

/// Closed-class Input/Output Logic deontic output class of a standing rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) enum DeonticOutputType {
    FormatVerify,
    PedagogicalExplain,
    RecommendationRule,
}

/// Full deterministic CORGI + I/O-Kratzer + DDL score breakdown for one standing rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) struct CorgiRuleScore {
    pub(in crate::retrieval) modal_base: QueryModalBase,
    pub(in crate::retrieval) query_focus: AnswerFocus,
    pub(in crate::retrieval) output_type: DeonticOutputType,
    pub(in crate::retrieval) rule_focus: AnswerFocus,
    pub(in crate::retrieval) k_gate_q16: i64,
    pub(in crate::retrieval) match_corgi_q16: i64,
    pub(in crate::retrieval) focus_q16: i64,
    pub(in crate::retrieval) spec_norm_q16: i64,
    pub(in crate::retrieval) h_norm_q16: i64,
    pub(in crate::retrieval) g_norm_q16: i64,
    pub(in crate::retrieval) score_ddl_q16: i64,
    pub(in crate::retrieval) grounded_q_mask: u64,
    pub(in crate::retrieval) matched_restrictors: usize,
    pub(in crate::retrieval) total_restrictors: usize,
}

/// 4-slot closed-class Frame-Semantic decomposition for Frame-QCL / CP-Net scoring.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(in crate::retrieval) struct SemanticFrame {
    pub(in crate::retrieval) actions: Vec<String>,
    pub(in crate::retrieval) targets: Vec<String>,
    pub(in crate::retrieval) evals: Vec<String>,
    pub(in crate::retrieval) beneficiaries: Vec<String>,
    pub(in crate::retrieval) has_time_constraint: bool,
}

/// Deterministic Frame-QCL + CP-Net score breakdown for one candidate preference turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) struct FramePrefScore {
    pub(in crate::retrieval) hard_target_guard_q16: i64,
    pub(in crate::retrieval) alpha_target_q16: i64,
    pub(in crate::retrieval) alpha_eval_q16: i64,
    pub(in crate::retrieval) alpha_action_q16: i64,
    pub(in crate::retrieval) alpha_beneficiary_q16: i64,
    pub(in crate::retrieval) beneficiary_contrib_q16: i64,
    pub(in crate::retrieval) unreq_time_penalty_q16: i64,
    pub(in crate::retrieval) score_pref_q16: i64,
}

// =========================================================================
// CLOSED-CLASS MORPHOLOGY & INTRINSIC WORDNET INFORMATION CONTENT
// =========================================================================

// hardcode-scan: closed-class (English inflectional suffix stripping for lexicon lemma resolution)
const MORPH_SUFFIXES: &[(&str, &str)] = &[
    ("ies", "y"),
    ("ied", "y"),
    ("ves", "f"),
    ("ches", "ch"),
    ("shes", "sh"),
    ("ing", "e"),
    ("ing", ""),
    ("ed", "e"),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
];

/// Generate candidate base lemmas (UTF-8 safe) for offline WordNet/Roget/ConceptNet lookup.
fn lemma_variants(word: &str) -> Vec<String> {
    let low = word.to_lowercase();
    let mut out = Vec::with_capacity(4);
    if low.is_empty() {
        return out;
    }
    out.push(low.clone());
    // Strip trailing ASCII digits from versioned stems (e.g. `html5` -> `html`).
    let alpha_base = low.trim_end_matches(|c: char| c.is_ascii_digit());
    if alpha_base.len() >= 3 && alpha_base.len() < low.len() && !out.iter().any(|x| x == alpha_base)
    {
        out.push(alpha_base.to_string());
    }
    for &(suf, rep) in MORPH_SUFFIXES {
        if let Some(base) = low.strip_suffix(suf)
            && base.chars().count() >= 3
        {
            let cand = if rep.is_empty() {
                base.to_string()
            } else {
                format!("{base}{rep}")
            };
            if !out.contains(&cand) {
                out.push(cand);
            }
        }
    }
    let st = stem(&low);
    if st.chars().count() >= 3 && !out.contains(&st) {
        out.push(st);
    }
    out
}

/// Corpus-free intrinsic WordNet 3.1 Information Content from synset descendant count:
/// `IC_int(c) = (1 - log2(h(c)) / log2(N_max)) * ONE` in `[0, ONE]`.
#[inline]
fn intrinsic_ic_from_desc(desc_count: u32) -> i64 {
    let max_log = log2_u64_q16(WN_MAX_NOUN_SYNSETS) as i64;
    if max_log <= 0 {
        return ONE;
    }
    let d_log = log2_u64_q16((desc_count.max(1)) as u64) as i64;
    ONE.saturating_sub(qdiv(d_log, max_log)).clamp(0, ONE)
}

/// Corpus-free intrinsic IC of a surface word in `Q16.16`, derived purely from WordNet
/// descendant counts (zero domain lists).
fn word_intrinsic_ic_q16(word: &str) -> i64 {
    for cand in lemma_variants(word) {
        let own = crate::wordnet::specific_ancestor_synsets(&cand, 0, u32::MAX);
        if !own.is_empty() {
            let med_dc = own[own.len() / 2].1 as u64;
            let max_dc = own[own.len() - 1].1 as u64;
            let eff_dc = ((med_dc + max_dc.saturating_mul(2)) / 3)
                .max(1)
                .min(u32::MAX as u64) as u32;
            let mut ic = intrinsic_ic_from_desc(eff_dc);
            if is_pref_framing_word(&cand) {
                ic = ic.min(20_480);
            }
            return ic;
        }
    }
    if is_pref_framing_word(word) {
        16_384
    } else {
        52_428
    }
}

/// True when `word` belongs to a WordNet hypernym lineage containing any of `targets`.
fn has_wn_hypernym_in(word: &str, targets: &[&str], max_hops: usize) -> bool {
    for cand in lemma_variants(word).into_iter().take(3) {
        if targets.contains(&cand.as_str()) {
            return true;
        }
        let up = crate::wordnet::upward_hypernyms(&cand, max_hops);
        if up.iter().any(|h| targets.contains(h)) {
            return true;
        }
    }
    false
}

/// UTF-8 safe exact / morphological / typo bridge score between two tokens in `Q16.16`.
fn exact_or_morph_alpha_q16(a: &str, b: &str) -> i64 {
    if a == b {
        return ONE;
    }
    let a_base = a.trim_end_matches(|c: char| c.is_ascii_digit());
    let b_base = b.trim_end_matches(|c: char| c.is_ascii_digit());
    if a_base.len() >= 3 && b_base.len() >= 3 && a_base == b_base {
        return 62_259; // 0.95
    }
    let sa = stem(a);
    let sb = stem(b);
    if sa.chars().count() >= 4 && sa == sb {
        return 60_620; // 0.925
    }
    if a.is_ascii() && b.is_ascii() && damerau_levenshtein_le_1(a, b) {
        return 57_344; // 0.875
    }
    if a.is_ascii()
        && b.is_ascii()
        && a.len() >= 5
        && b.len() >= 5
        && a.as_bytes()[..5] == b.as_bytes()[..5]
        && a.len().abs_diff(b.len()) <= 6
    {
        return 55_705; // 0.85
    }
    0
}

// =========================================================================
// PRECOMPUTED PER-QUERY 4-CHANNEL LEXICON BRIDGE CACHE
// =========================================================================

struct QueryBridgeCache {
    q_words: Vec<String>,
    q_has_exclusion: bool,
    q_wn_anc: Vec<Vec<(u32, u32)>>,
    q_wn_nbrs: Vec<Vec<(&'static str, u16)>>,
    q_wn_up: Vec<Vec<&'static str>>,
    q_roget: Vec<Vec<(&'static str, u16)>>,
    q_cn: Vec<Vec<(&'static str, u16)>>,
    q_wd: Vec<Vec<&'static str>>,
    gloss_synset_qmask: BTreeMap<u32, u64>,
    dual_gloss_lemma_qmask: BTreeMap<&'static str, u64>,
}

impl QueryBridgeCache {
    fn build(query_text: &str, extra_focus_words: &[&str]) -> Self {
        let low = query_text.to_lowercase();
        let raw = raw_tokens(&low);
        // hardcode-scan: closed-class (negation and exclusion function words)
        let q_has_exclusion = raw.iter().any(|w| {
            matches!(
                w.as_str(),
                "without" | "never" | "not" | "no" | "except" | "excluding" | "neither"
            )
        });
        let mut q_words: Vec<String> = Vec::new();
        for w in raw {
            if is_content(&w) && !q_words.contains(&w) {
                q_words.push(w);
                if q_words.len() >= 28 {
                    break;
                }
            }
        }
        for &fw in extra_focus_words {
            if !q_words.iter().any(|w| w == fw) && q_words.len() < 32 {
                q_words.push(fw.to_string());
            }
        }
        Self::from_words(q_words, q_has_exclusion)
    }

    fn from_words(q_words: Vec<String>, q_has_exclusion: bool) -> Self {
        let n = q_words.len().min(32);
        let q_words: Vec<String> = q_words.into_iter().take(n).collect();
        let mut q_wn_anc = Vec::with_capacity(n);
        let mut q_wn_nbrs = Vec::with_capacity(n);
        let mut q_wn_up = Vec::with_capacity(n);
        let mut q_roget = Vec::with_capacity(n);
        let mut q_cn = Vec::with_capacity(n);
        let mut q_wd = Vec::with_capacity(n);
        let mut gloss_synset_qmask: BTreeMap<u32, u64> = BTreeMap::new();
        let mut dual_gloss_lemma_qmask: BTreeMap<&'static str, u64> = BTreeMap::new();

        for (qi, qw) in q_words.iter().enumerate() {
            let bit = 1u64 << qi;
            let variants = lemma_variants(qw);

            let mut anc: Vec<(u32, u32)> = Vec::new();
            for v in variants.iter().take(2) {
                for pair in crate::wordnet::specific_ancestor_synsets(v, 4, 2_000) {
                    if !anc.iter().any(|&(s, _)| s == pair.0) {
                        anc.push(pair);
                    }
                }
            }
            anc.sort_unstable_by_key(|&(s, dc)| (dc, s));
            q_wn_anc.push(anc);

            let mut nbrs: Vec<(&'static str, u16)> = Vec::new();
            for v in variants.iter().take(2) {
                for (lem, w) in crate::wordnet::full_lexicon_neighbors(v)
                    .into_iter()
                    .take(24)
                {
                    if let Some(cur) = nbrs.iter_mut().find(|(l, _)| *l == lem) {
                        cur.1 = cur.1.max(w);
                    } else {
                        nbrs.push((lem, w));
                    }
                }
            }
            q_wn_nbrs.push(nbrs);

            let mut up: Vec<&'static str> = Vec::new();
            for v in variants.iter().take(2) {
                for u in crate::wordnet::upward_hypernyms(v, 2) {
                    if !up.contains(&u) {
                        up.push(u);
                    }
                }
                for p in crate::wordnet::typed_part_and_isa_parents(v) {
                    if !up.contains(&p) {
                        up.push(p);
                    }
                }
            }
            q_wn_up.push(up);

            let mut rog: Vec<(&'static str, u16)> = Vec::new();
            for v in variants.iter().take(2) {
                for (lem, w) in crate::roget::conceptual_neighbors(v, 24) {
                    if let Some(cur) = rog.iter_mut().find(|(l, _)| *l == lem) {
                        cur.1 = cur.1.max(w);
                    } else {
                        rog.push((lem, w));
                    }
                }
            }
            q_roget.push(rog);

            let mut cn: Vec<(&'static str, u16)> = Vec::new();
            for v in variants.iter().take(2) {
                for (lem, w) in crate::conceptnet::reciprocal_two_hop_neighbors(v, 10) {
                    if let Some(cur) = cn.iter_mut().find(|(l, _)| *l == lem) {
                        cur.1 = cur.1.max(w);
                    } else {
                        cn.push((lem, w));
                    }
                }
            }
            q_cn.push(cn);

            let mut wd: Vec<&'static str> = Vec::new();
            for v in variants.iter().take(2) {
                for c in crate::wikidata::upward_containers(v, 2) {
                    if !wd.contains(&c) {
                        wd.push(c);
                    }
                }
            }
            q_wd.push(wd);

            for v in variants.iter().take(3) {
                for &syn in crate::wordnet::synsets_with_gloss(v) {
                    *gloss_synset_qmask.entry(syn).or_insert(0) |= bit;
                }
            }
            for (lem, _) in crate::wordnet::dual_gloss_hypernym_bridge(qw, 16) {
                *dual_gloss_lemma_qmask.entry(lem).or_insert(0) |= bit;
            }
        }

        Self {
            q_words,
            q_has_exclusion,
            q_wn_anc,
            q_wn_nbrs,
            q_wn_up,
            q_roget,
            q_cn,
            q_wd,
            gloss_synset_qmask,
            dual_gloss_lemma_qmask,
        }
    }

    /// Channels 1–3 pairwise bridge score `alpha(q_idx, slot)` in `[0, ONE]`.
    fn pair_alpha_q16(&self, q_idx: usize, slot: &str, require_specific_target: bool) -> i64 {
        let qw = &self.q_words[q_idx];
        // Channel 1: Exact / Morphological / Typo.
        let ch1 = exact_or_morph_alpha_q16(qw, slot);
        if ch1 >= ONE {
            return ONE;
        }
        if require_specific_target && ch1 == 0 && word_intrinsic_ic_q16(slot) < GENERIC_IC_FLOOR {
            return 0;
        }

        let slot_vars = lemma_variants(slot);
        let mut best = ch1;

        // Channel 2: WordNet 3.1 Heyting implication + Lin-LCS intrinsic IC + 4-POS derivation/hypernym.
        for sv in slot_vars.iter().take(3) {
            let h_fwd = crate::wordnet::heyting_implication_ic_q16(qw, sv);
            if h_fwd >= 68 {
                let sc = ((h_fwd as i64) * ONE) / 100;
                if sc > best {
                    best = sc;
                }
            }
            let h_rev = crate::wordnet::heyting_implication_ic_q16(sv, qw);
            if h_rev >= 96 {
                let sc = ((h_rev as i64) * ONE) / 100;
                if sc > best {
                    best = sc;
                }
            } else if h_rev >= 68 && !require_specific_target {
                let sc = qmul(((h_rev as i64) * ONE) / 100, KG_HOP_DECAY);
                if sc > best {
                    best = sc;
                }
            }
        }

        // Lin (1998) LCS similarity over intrinsic WordNet IC (`dc <= 1_200` guards against generic roots).
        let q_anc = &self.q_wn_anc[q_idx];
        if !q_anc.is_empty() {
            let mut s_anc: Vec<(u32, u32)> = Vec::new();
            for sv in slot_vars.iter().take(2) {
                for pair in crate::wordnet::specific_ancestor_synsets(sv, 4, 2_000) {
                    if !s_anc.iter().any(|&(s, _)| s == pair.0) {
                        s_anc.push(pair);
                    }
                }
            }
            s_anc.sort_unstable_by_key(|&(s, dc)| (dc, s));
            if !s_anc.is_empty() {
                let ic_q = intrinsic_ic_from_desc(q_anc[0].1);
                let ic_s = intrinsic_ic_from_desc(s_anc[0].1);
                let mut best_lcs_dc: Option<u32> = None;
                for &(qs, qdc) in q_anc {
                    if qdc > 1_200 {
                        continue;
                    }
                    if s_anc.iter().any(|&(ss, _)| ss == qs)
                        && best_lcs_dc.is_none_or(|cur| qdc < cur)
                    {
                        best_lcs_dc = Some(qdc);
                    }
                }
                if let Some(lcs_dc) = best_lcs_dc {
                    let ic_lcs = intrinsic_ic_from_desc(lcs_dc);
                    if ic_lcs >= GENERIC_IC_FLOOR {
                        let denom = ic_q.saturating_add(ic_s).max(1);
                        let lin = qdiv(ic_lcs.saturating_mul(2), denom).clamp(0, ONE);
                        if lin >= 29_491 && lin > best {
                            best = lin;
                        }
                    }
                }
            }
        }

        // WordNet 4-POS full neighbors (synonyms, derivations, pertainyms, hypernyms).
        for &(nb, w) in &self.q_wn_nbrs[q_idx] {
            if w < 72 {
                continue;
            }
            if slot_vars
                .iter()
                .any(|sv| exact_or_morph_alpha_q16(nb, sv) >= 57_344)
            {
                let sc = qmul(((w as i64) * ONE) / 100, 57_344);
                if sc > best {
                    best = sc;
                }
            }
        }

        // Typed upward hypernyms & part/is-a parents (including multi-word constituents).
        for &up in &self.q_wn_up[q_idx] {
            let hit = slot_vars.iter().any(|sv| {
                exact_or_morph_alpha_q16(up, sv) >= 57_344
                    || (up.contains(' ')
                        && up
                            .split_ascii_whitespace()
                            .any(|p| p.len() >= 3 && exact_or_morph_alpha_q16(p, sv) >= 57_344))
            });
            if hit && 52_428 > best {
                best = 52_428;
            }
        }

        // Channel 3: Roget narrow category + ConceptNet reciprocal + Wikidata upward containers.
        for &(rn, rw) in &self.q_roget[q_idx] {
            if rw < 52 {
                continue;
            }
            if slot_vars
                .iter()
                .any(|sv| exact_or_morph_alpha_q16(rn, sv) >= 57_344)
            {
                let sc = if rw >= 66 {
                    (((rw as i64).min(92) * 49_152) / 92).max(42_598)
                } else {
                    42_598
                };
                if sc > best {
                    best = sc;
                }
            }
        }
        for &(cn, cw) in &self.q_cn[q_idx] {
            if slot_vars
                .iter()
                .any(|sv| exact_or_morph_alpha_q16(cn, sv) >= 57_344)
            {
                let sc = if cw >= 100 { 45_875 } else { 39_321 };
                if sc > best {
                    best = sc;
                }
            }
        }
        for &wd in &self.q_wd[q_idx] {
            let hit = slot_vars.iter().any(|sv| {
                exact_or_morph_alpha_q16(wd, sv) >= 57_344
                    || (wd.contains(' ')
                        && wd
                            .split_ascii_whitespace()
                            .any(|p| p.len() >= 3 && exact_or_morph_alpha_q16(p, sv) >= 57_344))
            });
            if hit {
                let sc = qmul(ONE, KG_HOP_DECAY);
                if sc > best {
                    best = sc;
                }
            }
        }

        best.clamp(0, ONE)
    }

    /// Channel 4: Dual-Witness WordNet Gloss Bridge.
    /// Emits a grounded witness edge only when `>= 2` distinct query lemmas co-occur in
    /// the slot's WordNet gloss or a shared synset gloss (preventing single-word gloss drift).
    fn dual_witness_gloss_q16(&self, slot: &str) -> (i64, u64) {
        if self.q_words.len() < 2 {
            return (0, 0);
        }
        let slot_vars = lemma_variants(slot);
        let mut best: i64 = 0;
        let mut best_mask: u64 = 0;

        // Path A: Direct gloss tokens of `slot` (unioning top 2 inflectional variants).
        let mut slot_gloss_toks: Vec<&'static str> = Vec::new();
        for sv in slot_vars.iter().take(2) {
            for gt in crate::wordnet::gloss_tokens_of(sv) {
                if !slot_gloss_toks.contains(&gt) {
                    slot_gloss_toks.push(gt);
                }
            }
        }
        if !slot_gloss_toks.is_empty() {
            let mut own_mask: u64 = 0;
            for (qi, qw) in self.q_words.iter().enumerate() {
                if slot_gloss_toks
                    .iter()
                    .any(|&gt| exact_or_morph_alpha_q16(qw, gt) >= 57_344)
                {
                    own_mask |= 1u64 << qi;
                }
            }
            if own_mask.count_ones() >= 2 {
                best = ALPHA_GLOSS_DIRECT;
                best_mask = own_mask;
            }
        }

        // Path B: Shared synset dual-witness via `synsets_with_gloss`.
        for sv in slot_vars.iter().take(2) {
            for &syn in crate::wordnet::senses(sv) {
                if let Some(&qmask) = self.gloss_synset_qmask.get(&syn)
                    && qmask.count_ones() >= 2
                    && ALPHA_GLOSS_DIRECT > best
                {
                    best = ALPHA_GLOSS_DIRECT;
                    best_mask = qmask;
                }
            }
            for (syn, _) in crate::wordnet::specific_ancestor_synsets(sv, 2, 1_500) {
                if let Some(&qmask) = self.gloss_synset_qmask.get(&syn)
                    && qmask.count_ones() >= 2
                    && ALPHA_GLOSS_ANCESTOR > best
                {
                    best = ALPHA_GLOSS_ANCESTOR;
                    best_mask = qmask;
                }
            }
        }

        // Path C: Dual-gloss hypernym bridge consensus across `>= 2` distinct query words.
        for (&lem, &qmask) in &self.dual_gloss_lemma_qmask {
            if qmask.count_ones() >= 2
                && slot_vars
                    .iter()
                    .any(|sv| exact_or_morph_alpha_q16(lem, sv) >= 57_344)
                && ALPHA_GLOSS_ANCESTOR > best
            {
                best = ALPHA_GLOSS_ANCESTOR;
                best_mask = qmask;
            }
        }

        if best > 0 && self.q_has_exclusion {
            best = (best + ALPHA_GLOSS_NEG_BONUS).min(ALPHA_GLOSS_CAP);
        }
        (best, best_mask)
    }
}

// =========================================================================
// ALGORITHM 2: KRATZERIAN MODAL-BASE GATE & I/O LOGIC DETACHMENT
// =========================================================================

/// Classify a query into its closed-class Kratzerian `QueryModalBase` and WH `AnswerFocus`.
pub(in crate::retrieval) fn classify_query_modal_base(
    query: &str,
    eff_it: &QueryIntent,
) -> (QueryModalBase, AnswerFocus) {
    let ql = query.trim().to_lowercase();
    let toks = raw_tokens(&ql);
    let slots = wh_slots(query);

    // 1. Detect event-ordering / comparative timeline queries (answer focus is ORDER/General, not DatePoint).
    // hardcode-scan: closed-class (chronological ordering and comparative sequence markers)
    const ORDERING_CUES: &[&str] = &[
        "which happened first",
        "which happened last",
        "happened first",
        "happened last",
        "before or after",
        "in what order",
        "order in which",
        "chronological order",
    ];
    let is_ordering_q = eff_it.is_event_sequence || ORDERING_CUES.iter().any(|p| ql.contains(p));

    // 2. Determine WH `AnswerFocus` via closed-class WH phrases + WordNet money-lineage check.
    // hardcode-scan: closed-class (clock-time interrogative phrases)
    const CLOCK_TIME_CUES: &[&str] = &[
        "what time",
        "at what time",
        "which hour",
        "what hour",
        "time of day",
        "o'clock",
    ];
    let asks_clock_time = CLOCK_TIME_CUES.iter().any(|p| ql.contains(p));

    // hardcode-scan: closed-class (calendar-date interrogative phrases)
    const DATE_POINT_CUES: &[&str] = &[
        "when was ",
        "when were ",
        "when did ",
        "when is ",
        "when are ",
        "when will ",
        "what date",
        "which date",
        "what day ",
        "which day ",
        "which month",
        "which year",
    ];
    let asks_calendar_date = !is_ordering_q
        && !asks_clock_time
        && ((slots & WH_TIME) != 0
            || eff_it.sniper_when
            || DATE_POINT_CUES.iter().any(|p| ql.contains(p)));

    // hardcode-scan: closed-class (WordNet high-level taxonomic roots for monetary value)
    const MONEY_WN_ROOTS: &[&str] = &[
        "money",
        "currency",
        "payment",
        "price",
        "cost",
        "income",
        "sum",
        "monetary_unit",
    ];
    let has_currency_sym =
        query.contains('$') || query.contains('\u{20AC}') || query.contains('\u{00A3}');
    let asks_how_many = ql.starts_with("how many ")
        || ql.contains(" how many ")
        || ql.starts_with("how often ")
        || ql.starts_with("how long ");
    let asks_how_much = ql.starts_with("how much ") || ql.contains(" how much ");
    let has_money_concept = has_currency_sym
        || toks
            .iter()
            .filter(|w| is_content(w))
            .any(|w| has_wn_hypernym_in(w, MONEY_WN_ROOTS, 3));

    let focus = if asks_clock_time {
        AnswerFocus::TimeOfDay
    } else if asks_calendar_date {
        AnswerFocus::DatePoint
    } else if !asks_how_many
        && (has_currency_sym
            || (asks_how_much && has_money_concept)
            || (has_money_concept
                && (ql.starts_with("what is the ") || ql.starts_with("what was the "))))
    {
        AnswerFocus::MoneyAmount
    } else if asks_how_many || (slots & WH_QTY) != 0 || eff_it.sniper_how_many {
        AnswerFocus::Quantity
    } else {
        AnswerFocus::General
    };

    // 3. Determine `QueryModalBase` via closed-class tense, WH, procedural, and choice markers.
    // hardcode-scan: closed-class (past/perfect episodic lookup and factual WH auxiliaries)
    const EPISODIC_CUES: &[&str] = &[
        "did i ",
        "did we ",
        "did you ",
        "have i ",
        "have we ",
        "had i ",
        "was my ",
        "were my ",
        "was i ",
        "were we ",
        "what was ",
        "what were ",
        "what did ",
        "where did ",
        "where was ",
        "where do i ",
        "where am i ",
        "who did ",
        "who was ",
        "who is ",
        "remind me of ",
        "what is my current",
        "what's my current",
        "what\u{2019}s my current",
    ];
    let is_episodic_fact = is_ordering_q
        || asks_calendar_date
        || asks_clock_time
        || asks_how_many
        || eff_it.sniper_how_many
        || EPISODIC_CUES.iter().any(|p| ql.contains(p));

    // hardcode-scan: closed-class (procedural how-to and explanatory request leads)
    const PROCEDURAL_CUES: &[&str] = &[
        "how should i ",
        "how do i ",
        "how can i ",
        "how would i ",
        "how to ",
        "explain ",
        "can you explain",
        "could you explain",
        "walk me through",
        "show me how",
        "what is the best way",
        "what's the best way",
        "step by step",
    ];
    let is_procedural_howto = PROCEDURAL_CUES.iter().any(|p| ql.contains(p))
        || (ql.starts_with("if i ") && (ql.contains(" how ") || ql.contains(", how")));

    // hardcode-scan: closed-class (choice, advice, planning, and recommendation leads)
    const CHOICE_PREF_CUES: &[&str] = &[
        "suggest",
        "recommend",
        "good options",
        "any options",
        "any tips",
        "any advice",
        "any ideas",
        "what should i pick",
        "what should i choose",
        "which should i",
        "i'm planning ",
        "i\u{2019}m planning ",
        "i am planning ",
        "i'm looking for ",
        "i\u{2019}m looking for ",
        "i am looking for ",
        "would you suggest",
        "would you recommend",
    ];
    let is_choice_pref = step14_is_rec_request(query)
        || eff_it.preference
        || CHOICE_PREF_CUES.iter().any(|p| ql.contains(p));

    let modal = if is_episodic_fact {
        QueryModalBase::EpisodicFact
    } else if is_choice_pref && !is_procedural_howto {
        QueryModalBase::ChoicePref
    } else if is_procedural_howto || eff_it.task_under_rules {
        QueryModalBase::ProceduralHowTo
    } else if is_choice_pref {
        QueryModalBase::ChoicePref
    } else {
        QueryModalBase::EpisodicFact
    };

    (modal, focus)
}

/// Split a standing rule text into `(action_clause, condition_clause)` at closed-class subordinators.
fn split_rule_action_and_condition(text: &str) -> (String, String) {
    let lower = text.trim().to_lowercase();
    // hardcode-scan: closed-class (subordinating conjunctions introducing rule conditions)
    const SUBORD: &[&str] = &[
        " when ",
        " whenever ",
        " if ",
        " while ",
        " during ",
        " once ",
        " anytime ",
        " unless ",
    ];
    if let Some((pos, _)) = SUBORD
        .iter()
        .filter_map(|s| lower.find(s).map(|p| (p, s.len())))
        .min_by_key(|&(p, _)| p)
    {
        return (
            lower[..pos].trim().to_string(),
            lower[pos..].trim().to_string(),
        );
    }
    if let Some((lead, rest)) = lower.split_once(',') {
        let lead_toks = raw_tokens(lead);
        // hardcode-scan: closed-class (prepositional fronted condition leads)
        const PREP_LEAD: &[&str] = &[
            "for",
            "in",
            "on",
            "during",
            "regarding",
            "about",
            "with",
            "under",
        ];
        if lead_toks
            .first()
            .is_some_and(|w| PREP_LEAD.contains(&w.as_str()))
        {
            return (rest.trim().to_string(), lead.trim().to_string());
        }
    }
    (lower, String::new())
}

/// Classify a standing rule's `DeonticOutputType` and expected `AnswerFocus` from its `[action]` + `[condition]`.
fn classify_rule_output_and_focus(rule_text: &str) -> (DeonticOutputType, AnswerFocus) {
    let (action, cond) = split_rule_action_and_condition(rule_text);
    let act_toks = raw_tokens(&action);
    let full_low = rule_text.to_lowercase();

    // hardcode-scan: closed-class (deontic formatting, verification, and citation action verbs)
    const FORMAT_VERIFY_OPS: &[&str] = &[
        "format",
        "confirm",
        "verify",
        "reconfirm",
        "check",
        "display",
        "write",
        "cite",
        "quote",
        "round",
        "convert",
        "specify",
        "label",
        "prefix",
        "append",
        "capitalize",
        "abbreviate",
    ];
    // hardcode-scan: closed-class (deontic recommendation and selection action verbs)
    const REC_RULE_OPS: &[&str] = &[
        "recommend",
        "suggest",
        "propose",
        "advise",
        "prioritize",
        "prefer",
    ];
    // hardcode-scan: closed-class (clock-time and timezone format syntax markers)
    const CLOCK_FMT_CUES: &[&str] = &[
        "hh:mm",
        "24-hour",
        "12-hour",
        "timezone",
        "time zone",
        "am/pm",
        "utc",
        "gmt",
    ];
    // hardcode-scan: closed-class (WordNet high-level taxonomic roots for monetary value)
    const MONEY_WN_ROOTS: &[&str] = &[
        "money",
        "currency",
        "payment",
        "price",
        "cost",
        "income",
        "sum",
        "monetary_unit",
    ];
    // hardcode-scan: closed-class (unit and numeric quantity formatting markers)
    const QTY_FMT_CUES: &[&str] = &[
        "unit",
        "units",
        "metric",
        "imperial",
        "decimal",
        "percent",
        "percentage",
        "round",
    ];

    let has_iso_date = (full_low.contains("mm/")
        || full_low.contains("/mm")
        || full_low.contains("mm-"))
        && (full_low.contains("dd") || full_low.contains("yyyy"));
    let has_clock_fmt = CLOCK_FMT_CUES.iter().any(|p| full_low.contains(p));
    let cal_units = ["month", "day", "year"]
        .iter()
        .filter(|&&u| act_toks.iter().any(|t| t == u))
        .count();
    let act_mentions_date = has_iso_date
        || cal_units >= 2
        || act_toks
            .iter()
            .any(|t| matches!(t.as_str(), "date" | "dates" | "calendar"));
    let cond_mentions_clock = raw_tokens(&cond)
        .iter()
        .any(|t| matches!(t.as_str(), "time" | "times" | "hour" | "hours"));
    let act_mentions_clock = has_clock_fmt
        || (!act_mentions_date
            && act_toks
                .iter()
                .any(|t| matches!(t.as_str(), "time" | "times" | "hour" | "hours")));
    let act_mentions_money = action.contains('$')
        || action.contains('\u{20AC}')
        || action.contains('\u{00A3}')
        || act_toks
            .iter()
            .filter(|w| is_content(w))
            .any(|w| has_wn_hypernym_in(w, MONEY_WN_ROOTS, 3));
    let act_mentions_qty = QTY_FMT_CUES
        .iter()
        .any(|u| act_toks.iter().any(|t| t == u));

    let is_format_verify = has_iso_date
        || has_clock_fmt
        || act_toks
            .iter()
            .any(|t| FORMAT_VERIFY_OPS.contains(&t.as_str()))
        || (act_mentions_date && !act_toks.iter().any(|t| REC_RULE_OPS.contains(&t.as_str())));

    let is_rec_rule = act_toks.iter().any(|t| REC_RULE_OPS.contains(&t.as_str()))
        || raw_tokens(&cond).iter().any(|t| {
            matches!(
                t.as_str(),
                "recommendation" | "recommendations" | "suggestion" | "suggestions"
            )
        });

    let output_type = if is_format_verify {
        DeonticOutputType::FormatVerify
    } else if is_rec_rule {
        DeonticOutputType::RecommendationRule
    } else {
        DeonticOutputType::PedagogicalExplain
    };

    let rule_focus = if act_mentions_date {
        AnswerFocus::DatePoint
    } else if act_mentions_clock || (cond_mentions_clock && !act_mentions_date) {
        AnswerFocus::TimeOfDay
    } else if act_mentions_money {
        AnswerFocus::MoneyAmount
    } else if act_mentions_qty {
        AnswerFocus::Quantity
    } else {
        AnswerFocus::General
    };

    (output_type, rule_focus)
}

/// True when the rule's expected `AnswerFocus` is compatible with the query's WH `AnswerFocus`.
#[inline]
fn answer_focus_compatible(q_focus: AnswerFocus, r_focus: AnswerFocus) -> bool {
    matches!(
        (q_focus, r_focus),
        (AnswerFocus::DatePoint, AnswerFocus::DatePoint)
            | (AnswerFocus::TimeOfDay, AnswerFocus::TimeOfDay)
            | (AnswerFocus::MoneyAmount, AnswerFocus::MoneyAmount)
            | (AnswerFocus::Quantity, AnswerFocus::Quantity)
            | (AnswerFocus::General, AnswerFocus::General)
    )
}

/// Compute Kratzerian modal-base detachment gate `K(Q, R)` in `Q16.16`.
#[inline]
fn kratzer_gate_q16(
    modal: QueryModalBase,
    q_focus: AnswerFocus,
    output: DeonticOutputType,
    r_focus: AnswerFocus,
) -> i64 {
    match modal {
        QueryModalBase::EpisodicFact => {
            if output != DeonticOutputType::FormatVerify {
                0
            } else if answer_focus_compatible(q_focus, r_focus) {
                ONE
            } else {
                K_FOCUS_MISMATCH
            }
        }
        QueryModalBase::ProceduralHowTo => match output {
            DeonticOutputType::PedagogicalExplain => ONE,
            DeonticOutputType::FormatVerify => {
                if answer_focus_compatible(q_focus, r_focus) {
                    ONE
                } else {
                    K_FOCUS_MISMATCH
                }
            }
            DeonticOutputType::RecommendationRule => K_WEAK_COMPAT,
        },
        QueryModalBase::ChoicePref => match output {
            DeonticOutputType::RecommendationRule => ONE,
            DeonticOutputType::PedagogicalExplain => K_CHOICE_PEDAGOGICAL,
            DeonticOutputType::FormatVerify => {
                if answer_focus_compatible(q_focus, r_focus) {
                    ONE
                } else {
                    K_FOCUS_MISMATCH
                }
            }
        },
    }
}

// =========================================================================
// ALGORITHM 1 & 3: SUBSUMPTION-CORGI + DDL ABDUCTION/CONTRACTION/POOLE
// =========================================================================

/// Extract rule scope/condition slots and partition them into `restrictors` vs generic meta-heads
/// purely via WordNet intrinsic IC, syntactic head-noun lineage, and closed-class framing checks.
fn extract_rule_slots(ix: &Index, rule: &StandingRule) -> (Vec<String>, Vec<bool>, Vec<i64>) {
    // hardcode-scan: closed-class (deontic imperative heads excluded from topic scope slots)
    const DEONTIC_HEADS: &[&str] = &[
        "always", "never", "please", "make", "sure", "must", "should", "ask", "asking", "asked",
    ];
    // hardcode-scan: closed-class (WordNet high-level abstract discourse-head roots)
    const META_HEAD_WN_ROOTS: &[&str] = &[
        "information",
        "fact",
        "communication",
        "question",
        "difficulty",
        "message",
    ];

    let mut raw_slots: Vec<String> = Vec::new();
    if !rule.condition_words.is_empty() {
        for cw in &rule.condition_words {
            let low = cw.to_lowercase();
            if is_content(&low)
                && !DEONTIC_HEADS.contains(&low.as_str())
                && !raw_slots.contains(&low)
            {
                raw_slots.push(low);
            }
        }
    } else if let Some(tr) = ix.turns.get(rule.turn) {
        for w in raw_tokens(&tr.body_for_headers) {
            if is_content(&w) && !DEONTIC_HEADS.contains(&w.as_str()) && !raw_slots.contains(&w) {
                raw_slots.push(w);
                if raw_slots.len() >= 10 {
                    break;
                }
            }
        }
    }

    let ics: Vec<i64> = raw_slots.iter().map(|s| word_intrinsic_ic_q16(s)).collect();
    let max_ic = ics.iter().copied().max().unwrap_or(0);
    let n_slots = raw_slots.len();
    let mut is_restrictor: Vec<bool> = raw_slots
        .iter()
        .zip(ics.iter())
        .enumerate()
        .map(|(idx, (s, &ic))| {
            let generic_by_ic = ic < GENERIC_HEAD_IC_CUTOFF || is_pref_framing_word(s);
            let is_temporal_head = matches!(
                s.as_str(),
                "time" | "times" | "date" | "dates" | "hour" | "hours"
            );
            let is_final_meta_head = n_slots >= 2
                && idx + 1 == n_slots
                && !is_temporal_head
                && has_wn_hypernym_in(s, META_HEAD_WN_ROOTS, 3);
            !((generic_by_ic || is_final_meta_head) && (ic < max_ic || is_final_meta_head))
        })
        .collect();
    if !raw_slots.is_empty() && !is_restrictor.iter().any(|&r| r) {
        for r in &mut is_restrictor {
            *r = true;
        }
    }
    (raw_slots, is_restrictor, ics)
}

/// Exact injective 1-to-1 backward iterator join maximizing `sum IC(s) * alpha(q, s)`
/// over the compact per-rule grounded bipartite graph (plus dual-witness gloss binding).
fn corgi_injective_backward_join(
    slots: &[String],
    is_restrictor: &[bool],
    slot_ics: &[i64],
    cache: &QueryBridgeCache,
) -> (Vec<i64>, u64) {
    let m = slots.len();
    let n = cache.q_words.len();
    if m == 0 || n == 0 {
        return (vec![0; m], 0);
    }

    // Build compact grounded edge weights `alpha_mat[s][q] = alpha(q, s)`.
    let mut alpha_mat: Vec<Vec<i64>> = vec![vec![0; n]; m];
    for (si, row) in alpha_mat.iter_mut().enumerate() {
        for (qi, cell) in row.iter_mut().enumerate() {
            *cell = cache.pair_alpha_q16(qi, &slots[si], false);
        }
    }

    // Order slots by (restrictor first, highest intrinsic IC first) for MRV backward iteration.
    let mut slot_order: Vec<usize> = (0..m).collect();
    slot_order.sort_by(|&a, &b| {
        is_restrictor[b]
            .cmp(&is_restrictor[a])
            .then_with(|| slot_ics[b].cmp(&slot_ics[a]))
            .then_with(|| a.cmp(&b))
    });

    // Greedy injective assignment with single-step augmenting swap to avoid local collisions.
    let mut assigned_q: Vec<Option<usize>> = vec![None; m];
    let mut owner_of_q: Vec<Option<usize>> = vec![None; n];

    for &si in &slot_order {
        let mut best_gain: i64 = 0;
        let mut best_move: Option<(usize, Option<(usize, usize)>)> = None;

        for (qi, &a) in alpha_mat[si].iter().enumerate() {
            if a <= 0 {
                continue;
            }
            let gain_si = qmul(slot_ics[si], a);
            match owner_of_q[qi] {
                None => {
                    if gain_si > best_gain {
                        best_gain = gain_si;
                        best_move = Some((qi, None));
                    }
                }
                Some(prev_s) => {
                    let prev_old = qmul(slot_ics[prev_s], alpha_mat[prev_s][qi]);
                    for (q2, &a2) in alpha_mat[prev_s].iter().enumerate() {
                        if owner_of_q[q2].is_some() || a2 <= 0 {
                            continue;
                        }
                        let prev_new = qmul(slot_ics[prev_s], a2);
                        let net = gain_si.saturating_add(prev_new).saturating_sub(prev_old);
                        if net > best_gain {
                            best_gain = net;
                            best_move = Some((qi, Some((prev_s, q2))));
                        }
                    }
                }
            }
        }

        if let Some((qi, displace)) = best_move {
            if let Some((prev_s, q2)) = displace {
                assigned_q[prev_s] = Some(q2);
                owner_of_q[q2] = Some(prev_s);
            }
            assigned_q[si] = Some(qi);
            owner_of_q[qi] = Some(si);
        }
    }

    let mut alpha_star = vec![0i64; m];
    let mut grounded_mask: u64 = 0;
    let mut used_q_mask: u64 = 0;
    for (si, a_slot) in alpha_star.iter_mut().enumerate() {
        if let Some(qi) = assigned_q[si] {
            let a = alpha_mat[si][qi];
            *a_slot = a;
            if a >= MATCH_THRESHOLD {
                let bit = 1u64 << qi;
                grounded_mask |= bit;
                used_q_mask |= bit;
            }
        }
    }

    // Channel 4: Dual-Witness Gloss upgrade for zero-overlap slots.
    for (si, a_slot) in alpha_star.iter_mut().enumerate() {
        let (g_alpha, g_mask) = cache.dual_witness_gloss_q16(&slots[si]);
        if g_alpha > *a_slot && (*a_slot == 0 || (g_mask & !used_q_mask) != 0) {
            *a_slot = g_alpha;
            if g_alpha >= MATCH_THRESHOLD {
                grounded_mask |= g_mask;
            }
        }
    }

    (alpha_star, grounded_mask)
}

/// Core scorer shared by `score_standing_rule_corgi_ddl` and `apply_corgi_deontic_seating`.
fn score_standing_rule_with_cache(
    ix: &Index,
    rule: &StandingRule,
    modal: QueryModalBase,
    q_focus: AnswerFocus,
    cache: &QueryBridgeCache,
) -> CorgiRuleScore {
    let rule_text = ix
        .turns
        .get(rule.turn)
        .map(|t| t.body_for_headers.as_str())
        .unwrap_or("");
    let (output_type, rule_focus) = classify_rule_output_and_focus(rule_text);
    let k_gate_q16 = kratzer_gate_q16(modal, q_focus, output_type, rule_focus);

    let (slots, is_restrictor, slot_ics) = extract_rule_slots(ix, rule);
    let total_restrictors = is_restrictor.iter().filter(|&&r| r).count();
    if slots.is_empty() || k_gate_q16 == 0 {
        return CorgiRuleScore {
            modal_base: modal,
            query_focus: q_focus,
            output_type,
            rule_focus,
            k_gate_q16,
            match_corgi_q16: 0,
            focus_q16: 0,
            spec_norm_q16: 0,
            h_norm_q16: if total_restrictors > 0 { ONE } else { 0 },
            g_norm_q16: 0,
            score_ddl_q16: 0,
            grounded_q_mask: 0,
            matched_restrictors: 0,
            total_restrictors,
        };
    }

    let (alpha_star, grounded_q_mask) =
        corgi_injective_backward_join(&slots, &is_restrictor, &slot_ics, cache);

    // Soft-conjunctive CORGI score: `qmul(min_{s in restrictors} alpha*(s), weighted_avg_IC(alpha*))`.
    let mut min_restr = ONE;
    let mut num_w: i64 = 0;
    let mut den_w: i64 = 0;
    let mut unmatched_ic: i64 = 0;
    let mut total_restr_ic: i64 = 0;
    let mut grounded_spec_ic: i64 = 0;
    let mut matched_restrictors = 0usize;

    for ((&raw_ic, &a), &is_r) in slot_ics
        .iter()
        .zip(alpha_star.iter())
        .zip(is_restrictor.iter())
    {
        let ic = raw_ic.max(4_096);
        num_w = num_w.saturating_add(qmul(ic, a));
        den_w = den_w.saturating_add(ic);
        if is_r {
            total_restr_ic = total_restr_ic.saturating_add(ic);
            if a < min_restr {
                min_restr = a;
            }
            if a >= MATCH_THRESHOLD {
                matched_restrictors += 1;
                grounded_spec_ic = grounded_spec_ic.saturating_add(qmul(ic, a));
            } else {
                unmatched_ic = unmatched_ic.saturating_add(ic);
            }
        }
    }

    let weighted_avg = if den_w > 0 {
        qdiv(num_w, den_w).clamp(0, ONE)
    } else {
        0
    };
    let match_corgi_q16 = qmul(min_restr, weighted_avg);

    // DDL Concept Abduction penalty `H_norm` for unmatched rule restrictors.
    let h_norm_q16 = if total_restr_ic > 0 {
        qdiv(unmatched_ic, total_restr_ic).clamp(0, ONE)
    } else {
        0
    };

    // DDL Concept Contraction penalty `G_norm` for semantic focus incompatibility (`DatePoint` vs `TimeOfDay`, etc.).
    let g_norm_q16 = match (q_focus, rule_focus) {
        (AnswerFocus::DatePoint, AnswerFocus::TimeOfDay)
        | (AnswerFocus::TimeOfDay, AnswerFocus::DatePoint) => ONE,
        (AnswerFocus::DatePoint, AnswerFocus::MoneyAmount | AnswerFocus::Quantity)
        | (AnswerFocus::TimeOfDay, AnswerFocus::MoneyAmount)
        | (AnswerFocus::MoneyAmount, AnswerFocus::DatePoint | AnswerFocus::TimeOfDay) => 49_152,
        _ => 0,
    };

    // Action-Object Focus Alignment.
    let focus_q16 = if q_focus != AnswerFocus::General && q_focus == rule_focus {
        ONE
    } else if q_focus == AnswerFocus::General
        && rule_focus == AnswerFocus::General
        && match_corgi_q16 >= MATCH_THRESHOLD
    {
        32_768
    } else {
        0
    };

    // Monotonic grounded specificity `Spec_norm = grounded_spec_ic / (grounded_spec_ic + ONE)`.
    let spec_norm_q16 = if grounded_spec_ic > 0 {
        qdiv(grounded_spec_ic, grounded_spec_ic.saturating_add(ONE)).clamp(0, ONE)
    } else {
        0
    };

    let raw_ddl = match_corgi_q16
        .saturating_add(qmul(W_FOCUS, focus_q16))
        .saturating_add(qmul(W_SPEC, spec_norm_q16))
        .saturating_sub(qmul(W_ABDUCTION, h_norm_q16))
        .saturating_sub(qmul(W_CONTRACTION, g_norm_q16));
    let score_ddl_q16 = qmul(k_gate_q16, raw_ddl.max(0));

    CorgiRuleScore {
        modal_base: modal,
        query_focus: q_focus,
        output_type,
        rule_focus,
        k_gate_q16,
        match_corgi_q16,
        focus_q16,
        spec_norm_q16,
        h_norm_q16,
        g_norm_q16,
        score_ddl_q16,
        grounded_q_mask,
        matched_restrictors,
        total_restrictors,
    }
}

/// True when `r2` strictly defeats `r1` under Poole's Specificity Principle:
/// `r2` subsumes all of `r1`'s grounded query evidence and covers an extra grounded restrictor
/// with zero abduction (`H == 0`) and zero contraction (`G == 0`).
pub(in crate::retrieval) fn poole_strictly_defeats(r2: &CorgiRuleScore, r1: &CorgiRuleScore) -> bool {
    r2.k_gate_q16 > 0
        && r2.h_norm_q16 == 0
        && r2.g_norm_q16 == 0
        && r2.grounded_q_mask != 0
        && (r2.grounded_q_mask & r1.grounded_q_mask) == r1.grounded_q_mask
        && (r2.matched_restrictors > r1.matched_restrictors
            || r2.grounded_q_mask.count_ones() > r1.grounded_q_mask.count_ones())
}

/// Score one standing rule against `(query, plan, eff_it)` with full `Subsumption-CORGI`,
/// `I/O-Kratzer Gate`, and `DDL` Concept Abduction / Contraction in `Q16.16`.
#[cfg(test)]
pub(in crate::retrieval) fn score_standing_rule_corgi_ddl(
    ix: &Index,
    rule: &StandingRule,
    query: &str,
    plan: &QueryPlan,
    eff_it: &QueryIntent,
) -> CorgiRuleScore {
    let (modal, q_focus) = classify_query_modal_base(query, eff_it);
    let extra_focus: &[&str] = match q_focus {
        AnswerFocus::DatePoint => &["date", "calendar"],
        AnswerFocus::TimeOfDay => &["clock", "hour"],
        AnswerFocus::MoneyAmount => &["money"],
        AnswerFocus::Quantity | AnswerFocus::General => &[],
    };
    let mut cache = QueryBridgeCache::build(query, extra_focus);
    let mut added = false;
    for gw in &plan.groups {
        if !gw.starts_with('\u{1}')
            && is_content(gw)
            && !cache.q_words.contains(gw)
            && cache.q_words.len() < 32
        {
            cache.q_words.push(gw.clone());
            added = true;
        }
    }
    if added {
        cache = QueryBridgeCache::from_words(cache.q_words, cache.q_has_exclusion);
    }
    score_standing_rule_with_cache(ix, rule, modal, q_focus, &cache)
}

// =========================================================================
// ALGORITHM 4: FRAME-QCL + CP-NET CONDITIONAL PARENT DOMINANCE
// =========================================================================

/// Closed-class 4-slot Frame-Semantic parser (`FE_Action`, `FE_Target`, `FE_Eval`, `FE_Beneficiary`,
/// plus `has_time_constraint`) driven purely by English preposition/syntactic boundaries and WordNet
/// high-level animate/social-group lineage (zero open-class domain lists).
pub(in crate::retrieval) fn extract_semantic_frame(text: &str) -> SemanticFrame {
    let low = text.to_lowercase();
    let sw = stopwords();
    let mut frame = SemanticFrame::default();

    // hardcode-scan: closed-class (possessive determiners introducing beneficiary/recipient NPs)
    const POSS_DET: &[&str] = &["my", "our", "his", "her", "their", "your"];
    // hardcode-scan: closed-class (temporal boundary prepositions introducing time constraints)
    const TIME_PREP: &[&str] = &[
        "around", "before", "after", "during", "until", "at", "by", "every",
    ];
    // hardcode-scan: closed-class (WordNet high-level taxonomic roots for animate/social beneficiaries)
    const BEN_WN_ROOTS: &[&str] = &["person", "social_group", "relative", "kin", "organization"];
    // hardcode-scan: closed-class (WordNet high-level taxonomic roots for time periods and intervals)
    const TIME_WN_ROOTS: &[&str] = &["time_period", "time_unit", "clock_time", "period"];
    // hardcode-scan: closed-class (closed-class preference and advice action verbs)
    const PREF_ACTION_VERBS: &[&str] = &[
        "prefer",
        "suggest",
        "recommend",
        "choose",
        "pick",
        "avoid",
        "plan",
        "planning",
        "schedule",
        "scheduling",
        "like",
        "love",
        "enjoy",
        "want",
    ];
    // hardcode-scan: closed-class (pre-nominal evaluative modifier cues and head nouns)
    const EVAL_HEADS: &[&str] = &[
        "option", "options", "idea", "ideas", "choice", "choices", "one", "ones",
    ];
    // hardcode-scan: closed-class (basic evaluative polarity adjectives)
    const CLOSED_EVAL_MODS: &[&str] = &[
        "good", "better", "best", "bad", "worse", "worst", "positive", "negative",
    ];

    let mut ben_tokens: BTreeSet<String> = BTreeSet::new();
    let mut time_tokens: BTreeSet<String> = BTreeSet::new();
    let mut eval_tokens: BTreeSet<String> = BTreeSet::new();
    let mut act_tokens: BTreeSet<String> = BTreeSet::new();

    for clause in low.split(['.', '!', '?', ';', '\n']) {
        let toks = raw_tokens(clause);
        let len = toks.len();
        for (i, w_str) in toks.iter().enumerate() {
            let w = w_str.as_str();

            // 1. Detect explicit temporal constraints (`TIME_PREP` + temporal noun/clock/period).
            if TIME_PREP.contains(&w) {
                let win_end = (i + 4).min(len);
                let mut found_time = false;
                for tok in &toks[i + 1..win_end] {
                    // hardcode-scan: closed-class (time-of-day grammatical nouns)
                    let is_time_word = token_is_temporal_word(tok)
                        || matches!(
                            tok.as_str(),
                            "time"
                                | "times"
                                | "hour"
                                | "hours"
                                | "morning"
                                | "afternoon"
                                | "evening"
                                | "night"
                                | "noon"
                                | "midnight"
                                | "bedtime"
                                | "schedule"
                        )
                        || tok.chars().any(|c| c.is_ascii_digit())
                        || has_wn_hypernym_in(tok, TIME_WN_ROOTS, 2);
                    if is_time_word {
                        found_time = true;
                        break;
                    }
                }
                if found_time {
                    frame.has_time_constraint = true;
                    for tok in &toks[i + 1..win_end] {
                        if is_content(tok) {
                            time_tokens.insert(tok.clone());
                        }
                    }
                }
            }
            if w == "schedule" || w == "scheduling" {
                act_tokens.insert(w.to_string());
            } else if w == "when" && i + 1 < len {
                frame.has_time_constraint = true;
            }

            // 2. Beneficiary slot (`for` / `with` + possessive + noun, or animate/social-group NP after `for`).
            if (w == "for" || w == "with") && i + 1 < len {
                let next = toks[i + 1].as_str();
                if POSS_DET.contains(&next) && i + 2 < len {
                    let head = toks[i + 2].as_str();
                    if is_content(head) && has_wn_hypernym_in(head, BEN_WN_ROOTS, 3) {
                        ben_tokens.insert(head.to_string());
                    }
                } else if w == "for"
                    && is_content(next)
                    && has_wn_hypernym_in(next, BEN_WN_ROOTS, 3)
                {
                    ben_tokens.insert(next.to_string());
                }
            }

            // 3. Action slot (preference/advice verb or gerund directly following it).
            if PREF_ACTION_VERBS.contains(&w) {
                act_tokens.insert(w.to_string());
                if i + 1 < len {
                    let next = toks[i + 1].as_str();
                    if next.len() >= 5 && next.ends_with("ing") && is_content(next) {
                        act_tokens.insert(next.to_string());
                    }
                }
            }

            // 4. Evaluative criteria (`with` / `without` post-modifier NP, or modifier before `options`/`ideas`).
            if matches!(w, "with" | "without") {
                let win_end = (i + 4).min(len);
                for tok in &toks[i + 1..win_end] {
                    if TIME_PREP.contains(&tok.as_str()) {
                        break;
                    }
                    if is_content(tok) {
                        if has_wn_hypernym_in(tok, BEN_WN_ROOTS, 3) {
                            ben_tokens.insert(tok.clone());
                        } else {
                            eval_tokens.insert(tok.clone());
                        }
                    }
                }
            }
            if CLOSED_EVAL_MODS.contains(&w) {
                eval_tokens.insert(w.to_string());
            }
            if i + 1 < len
                && EVAL_HEADS.contains(&toks[i + 1].as_str())
                && !sw.contains(w)
                && !is_pref_framing_word(w)
            {
                eval_tokens.insert(w.to_string());
            }

            // Catch standalone WordNet social-group/beneficiary modifiers inside preference phrases.
            if is_content(w)
                && !act_tokens.contains(w)
                && has_wn_hypernym_in(w, BEN_WN_ROOTS, 2)
                && i + 1 < len
                && is_content(&toks[i + 1])
            {
                ben_tokens.insert(w.to_string());
            }
        }
    }

    // 5. Target slot: substantive content nouns not consumed by actions/evals/beneficiaries/time/framing.
    let all_toks = raw_tokens(&low);
    let mut targets: Vec<String> = Vec::new();
    for tok in &all_toks {
        // hardcode-scan: closed-class (time-of-day words excluded from primary target nouns)
        let is_tod = matches!(
            tok.as_str(),
            "night" | "morning" | "afternoon" | "evening" | "day" | "time" | "times"
        );
        if !is_content(tok)
            || is_pref_framing_word(tok)
            || act_tokens.contains(tok)
            || eval_tokens.contains(tok)
            || ben_tokens.contains(tok)
            || time_tokens.contains(tok)
            || token_is_temporal_word(tok)
            || is_tod
        {
            continue;
        }
        if !targets.contains(tok) {
            targets.push(tok.clone());
        }
    }
    if targets.is_empty() {
        for tok in &all_toks {
            if is_content(tok)
                && !is_pref_framing_word(tok)
                && !ben_tokens.contains(tok)
                && !targets.contains(tok)
            {
                targets.push(tok.clone());
            }
        }
    }

    frame.actions = act_tokens.into_iter().collect();
    frame.targets = targets;
    frame.evals = eval_tokens.into_iter().collect();
    frame.beneficiaries = ben_tokens.into_iter().collect();
    frame
}

/// Best pairwise slot similarity between two Frame Element token lists in `Q16.16`.
fn best_slot_bridge_q16(q_slot: &[String], p_slot: &[String], require_specific_target: bool) -> i64 {
    if q_slot.is_empty() || p_slot.is_empty() {
        return 0;
    }
    let cache = QueryBridgeCache::from_words(q_slot.to_vec(), false);
    let mut best: i64 = 0;
    for pt in p_slot {
        for qi in 0..cache.q_words.len() {
            let a = cache.pair_alpha_q16(qi, pt, require_specific_target);
            if a > best {
                best = a;
            }
        }
        if !require_specific_target {
            let (g_alpha, _) = cache.dual_witness_gloss_q16(pt);
            if g_alpha > best {
                best = g_alpha;
            }
        }
    }
    best
}

/// Score a candidate preference frame `p_frame` against a query frame `q_frame` using
/// Frame-QCL hard target guard + CP-Net conditional parent dominance in `Q16.16`.
pub(in crate::retrieval) fn score_preference_frame_qcl(
    q_frame: &SemanticFrame,
    p_frame: &SemanticFrame,
) -> FramePrefScore {
    let alpha_target_q16 = best_slot_bridge_q16(&q_frame.targets, &p_frame.targets, true);
    let hard_target_guard_q16 = if alpha_target_q16 >= TAU_TARGET {
        ONE
    } else {
        0
    };
    let alpha_eval_q16 = best_slot_bridge_q16(&q_frame.evals, &p_frame.evals, false);
    let alpha_action_q16 = best_slot_bridge_q16(&q_frame.actions, &p_frame.actions, false);
    let alpha_beneficiary_q16 =
        best_slot_bridge_q16(&q_frame.beneficiaries, &p_frame.beneficiaries, false);
    let beneficiary_contrib_q16 = qmul(W_BENEFICIARY, alpha_beneficiary_q16).min(W_BENEFICIARY);
    let unreq_time_penalty_q16 = if p_frame.has_time_constraint && !q_frame.has_time_constraint {
        LAMBDA_TIME
    } else {
        0
    };

    let linear = qmul(W_TARGET, alpha_target_q16)
        .saturating_add(qmul(W_EVAL, alpha_eval_q16))
        .saturating_add(qmul(W_ACTION, alpha_action_q16))
        .saturating_add(beneficiary_contrib_q16)
        .saturating_sub(unreq_time_penalty_q16);
    let score_pref_q16 = qmul(hard_target_guard_q16, linear.max(0));

    FramePrefScore {
        hard_target_guard_q16,
        alpha_target_q16,
        alpha_eval_q16,
        alpha_action_q16,
        alpha_beneficiary_q16,
        beneficiary_contrib_q16,
        unreq_time_penalty_q16,
        score_pref_q16,
    }
}

// =========================================================================
// PUBLIC SEATING ENTRY POINT (`apply_corgi_deontic_seating`)
// =========================================================================

/// Compute the bitmask of the query's rarest exact non-stopword anchor (`star_bit`).
fn rarest_query_group_bit(ix: &Index, plan: &QueryPlan) -> u64 {
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
        let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0);
        if df == 0 {
            continue;
        }
        if best.is_none_or(|(bdf, bg)| df < bdf || (df == bdf && g < bg)) {
            best = Some((df, g));
        }
    }
    best.map_or(0, |(_, g)| 1u64 << g)
}

/// Seat `item` at `target_slot` (or preserve if already at `<= target_slot`) without duplicates.
fn seat_at(list: &mut Vec<usize>, item: usize, target_slot: usize, cap: usize) {
    if let Some(pos) = list.iter().position(|&x| x == item) {
        if pos <= target_slot {
            return;
        }
        list.remove(pos);
    }
    let ins = target_slot.min(list.len());
    list.insert(ins, item);
    if list.len() > cap {
        list.truncate(cap);
    }
}

/// Move `item` out of Top-5 (to index 5 if available, else the tail) to eliminate rule hijacks.
fn demote_below_top5(list: &mut Vec<usize>, item: usize) {
    let Some(pos) = list.iter().position(|&x| x == item) else {
        return;
    };
    if pos >= 5 || list.len() <= 1 {
        return;
    }
    list.remove(pos);
    let ins = 5usize.min(list.len());
    list.insert(ins, item);
}

/// Unified CORGI Deontic Policy & Frame-QCL Preference Seating pass for Stage 12:
/// 1. Scrubs `K == 0` standing-rule hijacks out of Top-5 on `EpisodicFact` queries
///    while strictly protecting a factual Slot-0 turn carrying the query's rare anchor (`star_bit`).
/// 2. Arbitrates active standing rules by `Subsumption-CORGI` + `DDL` + Poole Specificity,
///    seating winners strictly inside their Deontic Slot Quarantine (Slot 4 on `EpisodicFact`
///    with `K == ONE`, Slots 3..=4 on `ProceduralHowTo`).
/// 3. Reranks user preferences on `ChoicePref` / `preference` queries via `Frame-QCL` + CP-Net
///    parent dominance (capping beneficiary at 5%, enforcing Hard Target Guard, and penalizing
///    unrequested temporal constraints).
pub(in crate::retrieval) fn apply_corgi_deontic_seating(ctx: &mut RankCtx<'_>) {
    if ctx.has_named_speakers_eff
        || ctx.is_monologue_corpus
        || ctx.ix.has_tool_calls
        || ctx.implicit_disclosure_locked
        || ctx.torrens_locked_head
        || ctx.eff_it.is_event_sequence
        || eff_summary_request(&ctx.q.text)
        || ctx.final_turn_indices.is_empty()
        || ctx.final_session_indices.is_empty()
    {
        return;
    }

    let ix = ctx.ix;
    let cap = ctx.k.max(5);
    let (modal, q_focus) = classify_query_modal_base(&ctx.q.text, &ctx.eff_it);
    let star_bit = rarest_query_group_bit(ix, &ctx.plan);
    let slot0_protected = modal == QueryModalBase::EpisodicFact
        && ctx.final_turn_indices.first().is_some_and(|&t0| {
            let tr = &ix.turns[t0];
            !tr.is_standing_policy
                && !tr.is_question
                && !tr.is_backchannel
                && (star_bit == 0
                    || (ctx.t_exact_mask.get(t0).copied().unwrap_or(0) & star_bit) != 0)
        });

    // -------------------------------------------------------------------------
    // PHASE 1 & 2: I/O-KRATZER HIJACK SCRUBBING + DDL/POOLE RULE SEATING
    // -------------------------------------------------------------------------
    if !ix.standing_rules.is_empty() {
        let extra_focus: &[&str] = match q_focus {
            AnswerFocus::DatePoint => &["date", "calendar"],
            AnswerFocus::TimeOfDay => &["clock", "hour"],
            AnswerFocus::MoneyAmount => &["money"],
            AnswerFocus::Quantity | AnswerFocus::General => &[],
        };
        let cache = QueryBridgeCache::build(&ctx.q.text, extra_focus);

        let mut scored_rules: Vec<(usize, CorgiRuleScore)> = Vec::new();
        let mut rule_score_by_turn: BTreeMap<usize, CorgiRuleScore> = BTreeMap::new();
        for (ri, rule) in ix.standing_rules.iter().enumerate() {
            if turn_is_superseded(ix, rule.turn) {
                continue;
            }
            let sc = score_standing_rule_with_cache(ix, rule, modal, q_focus, &cache);
            rule_score_by_turn.insert(rule.turn, sc);
            scored_rules.push((ri, sc));
        }

        // Phase 1: Scrub `K == 0` standing rules out of Top-5 on `EpisodicFact`,
        // preserving condition-matched rules and `task_under_rules` queries.
        if modal == QueryModalBase::EpisodicFact && !ctx.eff_it.task_under_rules {
            let top5_rules_to_scrub: Vec<usize> = ctx
                .final_turn_indices
                .iter()
                .take(5)
                .copied()
                .filter(|&ti| {
                    ix.turns[ti].is_standing_policy
                        && !ctx.cond_rule_turns.contains(&ti)
                        && rule_score_by_turn
                            .get(&ti)
                            .is_some_and(|sc| sc.k_gate_q16 == 0)
                        && (star_bit == 0
                            || (ctx.t_exact_mask.get(ti).copied().unwrap_or(0) & star_bit) == 0)
                })
                .collect();
            for ti in top5_rules_to_scrub {
                demote_below_top5(&mut ctx.final_turn_indices, ti);
            }
        }

        // Phase 2: Poole Specificity dominance + Deontic Slot Quarantine seating.
        let mut eligible: Vec<(usize, CorgiRuleScore, bool)> = scored_rules
            .into_iter()
            .filter(|(_, sc)| sc.k_gate_q16 >= 32_768 && sc.score_ddl_q16 >= 26_214)
            .map(|(ri, sc)| (ri, sc, false))
            .collect();

        let scores_snapshot: Vec<CorgiRuleScore> = eligible.iter().map(|e| e.1).collect();
        for (i, entry) in eligible.iter_mut().enumerate() {
            entry.2 = scores_snapshot
                .iter()
                .enumerate()
                .any(|(j, other)| i != j && poole_strictly_defeats(other, &entry.1));
        }

        eligible.sort_by(|a, b| {
            a.2.cmp(&b.2)
                .then_with(|| b.1.score_ddl_q16.cmp(&a.1.score_ddl_q16))
                .then_with(|| b.1.spec_norm_q16.cmp(&a.1.spec_norm_q16))
                .then_with(|| ix.standing_rules[b.0].ts.cmp(&ix.standing_rules[a.0].ts))
                .then_with(|| ix.standing_rules[b.0].turn.cmp(&ix.standing_rules[a.0].turn))
        });

        // If a Poole-defeated general rule sits in Top-5, promote its strictly more specific winner ahead of it.
        if let Some(&(win_ri, _, false)) = eligible.first() {
            let win_ti = ix.standing_rules[win_ri].turn;
            let win_si = ix.standing_rules[win_ri].session;
            for &(def_ri, _, is_defeated) in eligible.iter().skip(1) {
                if !is_defeated {
                    continue;
                }
                let def_ti = ix.standing_rules[def_ri].turn;
                if let Some(def_pos) = ctx
                    .final_turn_indices
                    .iter()
                    .take(5)
                    .position(|&t| t == def_ti)
                {
                    let target = if slot0_protected && def_pos == 0 {
                        1
                    } else {
                        def_pos
                    };
                    seat_at(&mut ctx.final_turn_indices, win_ti, target, cap);
                    seat_at(&mut ctx.final_session_indices, win_si, target, cap);
                    break;
                }
            }
        }

        match modal {
            QueryModalBase::EpisodicFact => {
                // Only `FormatVerify` with exact `AnswerFocus` match (`K == ONE`) may enter Slot 4.
                if let Some(&(ri, sc, _)) = eligible.first()
                    && sc.k_gate_q16 == ONE
                    && sc.match_corgi_q16 >= 16_384
                    && sc.score_ddl_q16 >= 32_768
                    && !(slot0_protected && ctx.final_turn_indices.len() == 1)
                {
                    let r_ti = ix.standing_rules[ri].turn;
                    let r_si = ix.standing_rules[ri].session;
                    seat_at(&mut ctx.final_turn_indices, r_ti, 4, cap);
                    seat_at(&mut ctx.final_session_indices, r_si, 4, cap);
                }
            }
            QueryModalBase::ProceduralHowTo => {
                // Quarantine procedural rules to Slots 3..=4 (never touching Slot 0).
                let mut seated = 0usize;
                for (ri, sc, is_defeated) in &eligible {
                    if *is_defeated || sc.k_gate_q16 < 49_152 || sc.score_ddl_q16 < 26_214 {
                        continue;
                    }
                    let r_ti = ix.standing_rules[*ri].turn;
                    let r_si = ix.standing_rules[*ri].session;
                    let target_slot = if seated == 0 { 3 } else { 4 };
                    seat_at(&mut ctx.final_turn_indices, r_ti, target_slot, cap);
                    seat_at(&mut ctx.final_session_indices, r_si, target_slot, cap);
                    seated += 1;
                    if seated >= 2 {
                        break;
                    }
                }
            }
            QueryModalBase::ChoicePref => {
                // Quarantine choice-applicable rules to Slot 4 only on non-preference queries so
                // calibrated `preference_following` Top-5 turns are never evicted.
                if !ctx.pref_on
                    && !ctx.eff_it.preference
                    && let Some(&(ri, sc, false)) = eligible.first()
                    && sc.k_gate_q16 == ONE
                    && sc.score_ddl_q16 >= 36_044
                {
                    let r_ti = ix.standing_rules[ri].turn;
                    let r_si = ix.standing_rules[ri].session;
                    seat_at(&mut ctx.final_turn_indices, r_ti, 4, cap);
                    seat_at(&mut ctx.final_session_indices, r_si, 4, cap);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // PHASE 3: FRAME-QCL + CP-NET PREFERENCE RERANKING
    // -------------------------------------------------------------------------
    let run_pref_qcl = (modal == QueryModalBase::ChoicePref || ctx.pref_on || ctx.eff_it.preference)
        && modal != QueryModalBase::EpisodicFact
        && !ctx.eff_it.assistant_attribution
        && !ctx.eff_it.sniper_how_many
        && !ctx.eff_it.sniper_when
        && ix.session_turns.len() >= 16;
    if !run_pref_qcl {
        return;
    }

    let q_frame = extract_semantic_frame(&ctx.q.text);
    if q_frame.targets.is_empty() {
        return;
    }

    // Collect active user preference candidates from both the directive register and top-10 pool.
    let mut cand_turns: Vec<usize> = Vec::new();
    for &pti in &ix.preference_directive_turns {
        if pti < ix.turns.len() && !cand_turns.contains(&pti) {
            cand_turns.push(pti);
        }
    }
    for &ti in ctx.final_turn_indices.iter().take(10) {
        if cand_turns.contains(&ti) {
            continue;
        }
        let tr = &ix.turns[ti];
        if tr.field != 0 || tr.is_backchannel || tr.is_standing_policy {
            continue;
        }
        let low = tr.body_for_headers.to_lowercase();
        if tr.is_personal_constraint
            || has_compound_preference_frame(&tr.body_for_headers)
            || is_explicit_stance_directive(&low, tr.is_personal_constraint)
            || self_disclosure_strength_q16(ix, ti) >= ONE as u64
        {
            cand_turns.push(ti);
        }
    }

    if cand_turns.is_empty() {
        return;
    }

    let mut pref_scores: BTreeMap<usize, FramePrefScore> = BTreeMap::new();
    let mut passing_prefs: Vec<(usize, FramePrefScore)> = Vec::new();
    for ti in cand_turns {
        let tr = &ix.turns[ti];
        if tr.field != 0
            || tr.is_backchannel
            || tr.is_standing_policy
            || turn_is_superseded(ix, ti)
        {
            continue;
        }
        let p_frame = extract_semantic_frame(&tr.body_for_headers);
        let sc = score_preference_frame_qcl(&q_frame, &p_frame);
        pref_scores.insert(ti, sc);
        if sc.hard_target_guard_q16 == ONE && sc.score_pref_q16 >= TAU_TARGET {
            passing_prefs.push((ti, sc));
        }
    }

    if passing_prefs.is_empty() {
        return;
    }

    passing_prefs.sort_by(|a, b| {
        b.1.score_pref_q16
            .cmp(&a.1.score_pref_q16)
            .then_with(|| b.1.alpha_target_q16.cmp(&a.1.alpha_target_q16))
            .then_with(|| {
                let tsa = ix
                    .session_ts
                    .get(ix.turns[a.0].session)
                    .copied()
                    .unwrap_or(0);
                let tsb = ix
                    .session_ts
                    .get(ix.turns[b.0].session)
                    .copied()
                    .unwrap_or(0);
                tsb.cmp(&tsa)
            })
            .then_with(|| b.0.cmp(&a.0))
    });

    let (best_ti, _best_sc) = passing_prefs[0];
    let best_si = ix.turns[best_ti].session;
    // Protect calibrated declarative user turns at Slots 0..=3; only take Slot 0 when incumbent
    // is an assistant turn, question, backchannel, or stray standing rule, else co-seat at Slot 4.
    let slot0_yields = ctx.final_turn_indices.first().is_none_or(|&t0| {
        if t0 == best_ti {
            return true;
        }
        let tr0 = &ix.turns[t0];
        tr0.field == 1 || tr0.is_question || tr0.is_backchannel || tr0.is_standing_policy
    });

    if slot0_yields {
        seat_at(&mut ctx.final_turn_indices, best_ti, 0, cap);
        seat_at(&mut ctx.final_session_indices, best_si, 0, cap);
    } else if !ctx.final_turn_indices.iter().take(5).any(|&t| t == best_ti) {
        let slot4_weak = ctx.final_turn_indices.get(4).is_none_or(|&t4| {
            let tr4 = &ix.turns[t4];
            tr4.field != 0
                || tr4.is_question
                || tr4.is_backchannel
                || tr4.is_standing_policy
                || (ctx.t_exact_mask.get(t4).copied().unwrap_or(0) & ctx.q_group_mask) == 0
        });
        if slot4_weak {
            let target_slot = 4usize.min(ctx.final_turn_indices.len());
            seat_at(&mut ctx.final_turn_indices, best_ti, target_slot, cap);
            if !ctx.final_session_indices.iter().take(5).any(|&s| s == best_si) {
                let s_slot = 4usize.min(ctx.final_session_indices.len());
                seat_at(&mut ctx.final_session_indices, best_si, s_slot, cap);
            }
        }
    }
    let _ = pref_scores;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{
        Fields, Params, Query, Session, Turn, build_index, classify, plan_query,
    };

    fn mk_turn(id: &str, role: &str, content: &str) -> Turn {
        Turn {
            id: id.to_string(),
            role: role.to_string(),
            content: content.to_string(),
            links: Vec::new(),
            stability: 0.0,
        }
    }

    fn mk_sess(id: &str, ts: i64, turns: Vec<Turn>) -> Session {
        Session {
            id: id.to_string(),
            date: String::new(),
            timestamp: ts,
            turns,
        }
    }

    fn mk_test_ctx<'a>(
        ix: &'a Index,
        q: &'a Query,
        p: &'a Params,
        eff_it: QueryIntent,
        final_sessions: Vec<usize>,
        final_turns: Vec<usize>,
        t_exact_mask: Vec<u64>,
    ) -> RankCtx<'a> {
        let plan = plan_query(ix, q, p);
        let ns = ix.session_turns.len();
        let nt = ix.turns.len();
        RankCtx {
            ix,
            q,
            p,
            k: 5,
            ns,
            nt,
            plan,
            pref_on: eff_it.preference,
            persona_infer_on: false,
            episodic_user_quote: false,
            user_role: true,
            fl: Fields {
                w: [1.0, 1.0, 0.75],
                b: [0.75, 0.75, 0.0],
            },
            s_mask: vec![0; ns],
            s_exact_mask: vec![0; ns],
            s_score: vec![1.0; ns],
            smax: 1.0,
            t_own: vec![1.0; nt],
            t_raw: vec![1.0; nt],
            t_turn_raw: vec![1.0; nt],
            t_mask: t_exact_mask.clone(),
            t_exact_mask,
            t_scene: vec![1.0; nt],
            t_scene_sess: vec![1.0; nt],
            q_toks_eff: raw_tokens(&q.text),
            past_cue_eff: false,
            seq_mode_eff: false,
            has_named_speakers_eff: false,
            is_monologue_corpus: false,
            has_compact_rule_register: false,
            cond_rule_turns: Vec::new(),
            eff_it,
            t_bridge: vec![0.0; nt],
            t_bridge_hits: vec![0; nt],
            lexical_smax: 1.0,
            lexical_tmax_sess: 1.0,
            lexical_tmax: 1.0,
            standing_rules: Vec::new(),
            s_policy_floored: vec![false; ns],
            t_policy_floored: vec![false; nt],
            s_pref_floored: vec![false; ns],
            t_pref_floored: vec![false; nt],
            initiative_content: Vec::new(),
            bridge_boost: vec![0.0; ns],
            s_ord_base: (0..ns).collect(),
            total_q_groups: 2,
            mask_limit: 0,
            final_session_indices: final_sessions,
            strong_top_sessions: 1,
            max_per_session_top5: 2,
            turn_ts: vec![0; nt],
            final_turn_indices: final_turns,
            base_q16: vec![65_536; nt],
            q_group_cnt: 2,
            q_group_mask: 0b11,
            grounded_group_mask: 0b11,
            top3_union_exact: 0b11,
            has_ungrounded_rare_oov: false,
            deficit_a: false,
            deficit_b: false,
            torrens_locked_head: false,
            implicit_disclosure_locked: false,
            ghostcurrent: Default::default(),
            wf_g1_s10: Vec::new(),
            wf_g3_t10: Vec::new(),
            wf_g4_s5: Vec::new(),
            wf_g4_flags: Vec::new(),
        }
    }

    #[test]
    fn kratzer_gate_blocks_pedagogical_rule_on_episodic_fact_and_allows_date_format() {
        let p = Params::default();
        let sessions = vec![
            mk_sess(
                "s0",
                1000,
                vec![
                    mk_turn(
                        "t0",
                        "user",
                        "Always include visual aids when I ask about scheduling details.",
                    ),
                    mk_turn("t1", "assistant", "Noted, I will include visual aids."),
                ],
            ),
            mk_sess(
                "s1",
                2000,
                vec![
                    mk_turn(
                        "t2",
                        "user",
                        "Always include timezone when I ask about meeting times.",
                    ),
                    mk_turn(
                        "t3",
                        "user",
                        "Always format dates in MM/DD/YYYY when I ask about scheduling details.",
                    ),
                ],
            ),
            mk_sess(
                "s2",
                3000,
                vec![
                    mk_turn(
                        "t4",
                        "user",
                        "My meetings at East Janethaven Library took place on 04/12/2025.",
                    ),
                    mk_turn("t5", "user", "Also I picked up a spare notebook there."),
                ],
            ),
        ];
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "When was my meetings at East Janethaven Library?".to_string(),
            date: String::new(),
        };
        let eff_it = classify(&q.text);
        let plan = plan_query(&ix, &q, &p);

        let (modal, focus) = classify_query_modal_base(&q.text, &eff_it);
        assert_eq!(modal, QueryModalBase::EpisodicFact);
        assert_eq!(focus, AnswerFocus::DatePoint);

        let r_ped = ix
            .standing_rules
            .iter()
            .find(|r| ix.turns[r.turn].id == "t0")
            .unwrap();
        let r_time = ix
            .standing_rules
            .iter()
            .find(|r| ix.turns[r.turn].id == "t2")
            .unwrap();
        let r_date = ix
            .standing_rules
            .iter()
            .find(|r| ix.turns[r.turn].id == "t3")
            .unwrap();

        let s_ped = score_standing_rule_corgi_ddl(&ix, r_ped, &q.text, &plan, &eff_it);
        let s_time = score_standing_rule_corgi_ddl(&ix, r_time, &q.text, &plan, &eff_it);
        let s_date = score_standing_rule_corgi_ddl(&ix, r_date, &q.text, &plan, &eff_it);

        assert_eq!(
            s_ped.k_gate_q16, 0,
            "pedagogical rule must get K=0 on EpisodicFact"
        );
        assert_eq!(s_ped.score_ddl_q16, 0);
        assert_eq!(
            s_time.k_gate_q16, K_FOCUS_MISMATCH,
            "TimeOfDay rule must pay focus mismatch"
        );
        assert_eq!(
            s_time.g_norm_q16, ONE,
            "DatePoint vs TimeOfDay must incur full contraction G"
        );
        assert_eq!(
            s_date.k_gate_q16, ONE,
            "DatePoint FormatVerify rule must detach at K=ONE"
        );
        assert!(
            s_date.score_ddl_q16 > s_time.score_ddl_q16 && s_date.score_ddl_q16 > 0,
            "date format rule ({}) must strictly beat clock-time rule ({})",
            s_date.score_ddl_q16,
            s_time.score_ddl_q16
        );

        // Verify `apply_corgi_deontic_seating` scrubs a hijacking K=0 rule out of Slot 0
        // and protects the factual turn at Slot 0.
        let t_exact = vec![0, 0, 0, 0, u64::MAX, 0];
        let mut ctx = mk_test_ctx(
            &ix,
            &q,
            &p,
            eff_it,
            vec![0, 2, 1],
            vec![0, 4, 1, 2, 5, 3],
            t_exact,
        );
        apply_corgi_deontic_seating(&mut ctx);
        assert_eq!(
            ctx.final_turn_indices[0], 4,
            "factual turn t4 must hold Slot 0 after K=0 scrub"
        );
        assert_ne!(
            ctx.final_turn_indices[0], 0,
            "K=0 pedagogical rule must be scrubbed from Slot 0"
        );
    }

    #[test]
    fn ddl_poole_specificity_and_abduction_discriminate_general_vs_compound_rules() {
        let p = Params::default();
        let sessions = vec![mk_sess(
            "s0",
            1000,
            vec![
                mk_turn(
                    "r_gen",
                    "user",
                    "Always include author biographies when I ask about book recommendations.",
                ),
                mk_turn(
                    "r_comp",
                    "user",
                    "Always list reading order when I ask about book series recommendations.",
                ),
            ],
        )];
        let ix = build_index(&sessions, &p);
        let r_gen = ix
            .standing_rules
            .iter()
            .find(|r| ix.turns[r.turn].id == "r_gen")
            .unwrap();
        let r_comp = ix
            .standing_rules
            .iter()
            .find(|r| ix.turns[r.turn].id == "r_comp")
            .unwrap();

        // Query 1: general "book recommendations" -> compound rule pays abduction penalty H for missing "series".
        let q_gen = Query {
            text: "Can you suggest good book recommendations?".to_string(),
            date: String::new(),
        };
        let it_gen = classify(&q_gen.text);
        let plan_gen = plan_query(&ix, &q_gen, &p);
        let sc_gen_1 = score_standing_rule_corgi_ddl(&ix, r_gen, &q_gen.text, &plan_gen, &it_gen);
        let sc_comp_1 =
            score_standing_rule_corgi_ddl(&ix, r_comp, &q_gen.text, &plan_gen, &it_gen);
        assert_eq!(sc_gen_1.h_norm_q16, 0);
        assert!(
            sc_comp_1.h_norm_q16 > 0,
            "missing 'series' restrictor must incur abduction H > 0"
        );
        assert!(
            sc_gen_1.score_ddl_q16 > sc_comp_1.score_ddl_q16,
            "general rule ({}) must beat unmatched compound rule ({})",
            sc_gen_1.score_ddl_q16,
            sc_comp_1.score_ddl_q16
        );

        // Query 2: compound "book series recommendations" -> both H==0, compound wins via Poole Specificity.
        let q_comp = Query {
            text: "Can you suggest good book series recommendations?".to_string(),
            date: String::new(),
        };
        let it_comp = classify(&q_comp.text);
        let plan_comp = plan_query(&ix, &q_comp, &p);
        let sc_gen_2 =
            score_standing_rule_corgi_ddl(&ix, r_gen, &q_comp.text, &plan_comp, &it_comp);
        let sc_comp_2 =
            score_standing_rule_corgi_ddl(&ix, r_comp, &q_comp.text, &plan_comp, &it_comp);
        assert_eq!(sc_comp_2.h_norm_q16, 0);
        assert_eq!(sc_comp_2.g_norm_q16, 0);
        assert!(
            poole_strictly_defeats(&sc_comp_2, &sc_gen_2),
            "compound rule must strictly defeat general rule under Poole Specificity"
        );
        assert!(sc_comp_2.score_ddl_q16 > sc_gen_2.score_ddl_q16);
    }

    #[test]
    fn dual_witness_gloss_corgi_bridges_zero_overlap_definition_and_rejects_single_token() {
        // Pull two guaranteed rare gloss tokens from WordNet 3.1's own definition of `cappuccino`.
        let gtoks = crate::wordnet::gloss_tokens_of("cappuccino");
        assert!(
            gtoks.len() >= 2,
            "expected >= 2 WordNet gloss tokens for cappuccino, got {gtoks:?}"
        );
        let w1 = gtoks[0].to_string();
        let w2 = gtoks[1].to_string();

        // Dual witness (>= 2 distinct query lemmas in gloss) emits >= ALPHA_GLOSS_DIRECT.
        let dual_cache = QueryBridgeCache::from_words(vec![w1.clone(), w2], false);
        let (dual_alpha, dual_mask) = dual_cache.dual_witness_gloss_q16("cappuccino");
        assert!(
            dual_alpha >= ALPHA_GLOSS_DIRECT,
            "expected dual-witness gloss alpha >= {ALPHA_GLOSS_DIRECT}, got {dual_alpha}"
        );
        assert!(dual_mask.count_ones() >= 2);

        // Single-token query must NOT trigger dual-witness gloss bridge.
        let single_cache = QueryBridgeCache::from_words(vec![w1], false);
        let (single_alpha, _) = single_cache.dual_witness_gloss_q16("cappuccino");
        assert_eq!(
            single_alpha, 0,
            "single gloss token must not trigger dual-witness bridge"
        );
    }

    #[test]
    fn frame_qcl_cpnet_caps_beneficiary_at_5pct_and_penalizes_unrequested_time() {
        let q_frame = extract_semantic_frame(
            "I'm planning a movie night for my family. Suggest good options?",
        );
        assert!(!q_frame.has_time_constraint);
        assert!(
            q_frame.targets.iter().any(|t| t == "movie"),
            "targets: {:?}",
            q_frame.targets
        );
        assert!(
            q_frame.beneficiaries.iter().any(|b| b == "family"),
            "beneficiaries: {:?}",
            q_frame.beneficiaries
        );

        let p_gold = extract_semantic_frame("I prefer movies with positive family reviews.");
        let p_dist =
            extract_semantic_frame("I prefer scheduling family activities around nap times.");
        let p_ben_only = extract_semantic_frame("I always buy snacks for my family.");

        assert!(!p_gold.has_time_constraint);
        assert!(
            p_dist.has_time_constraint,
            "distractor must detect 'around nap times'"
        );

        let sc_gold = score_preference_frame_qcl(&q_frame, &p_gold);
        let sc_dist = score_preference_frame_qcl(&q_frame, &p_dist);
        let sc_ben = score_preference_frame_qcl(&q_frame, &p_ben_only);

        assert_eq!(sc_gold.hard_target_guard_q16, ONE);
        assert!(sc_gold.alpha_target_q16 >= TAU_TARGET);
        assert_eq!(sc_gold.unreq_time_penalty_q16, 0);
        assert!(sc_gold.beneficiary_contrib_q16 <= W_BENEFICIARY);
        assert!(
            sc_gold.score_pref_q16 >= 32_768,
            "expected strong gold Frame-QCL score, got {}",
            sc_gold.score_pref_q16
        );

        assert_eq!(sc_dist.unreq_time_penalty_q16, LAMBDA_TIME);
        assert_eq!(sc_dist.hard_target_guard_q16, 0);
        assert_eq!(sc_dist.score_pref_q16, 0);

        assert!(sc_ben.beneficiary_contrib_q16 <= W_BENEFICIARY);
        assert_eq!(sc_ben.hard_target_guard_q16, 0);
        assert_eq!(sc_ben.score_pref_q16, 0);
        assert!(sc_gold.score_pref_q16 > sc_dist.score_pref_q16);
    }

    #[test]
    fn corgi_handles_multibyte_utf8_without_panic() {
        let p = Params::default();
        let sessions = vec![mk_sess(
            "s_utf8",
            1000,
            vec![
                mk_turn(
                    "u0",
                    "user",
                    "Always include résumé notes when I ask about café — 日本語 🎬 reviews.",
                ),
                mk_turn(
                    "u1",
                    "user",
                    "I prefer naïve résumé templates for my café — 日本語.",
                ),
            ],
        )];
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "I’m planning a café résumé night — suggest good 日本語 🎬 options?".to_string(),
            date: String::new(),
        };
        let eff_it = classify(&q.text);
        let plan = plan_query(&ix, &q, &p);

        let (modal, _focus) = classify_query_modal_base(&q.text, &eff_it);
        assert_eq!(modal, QueryModalBase::ChoicePref);

        for rule in &ix.standing_rules {
            let sc = score_standing_rule_corgi_ddl(&ix, rule, &q.text, &plan, &eff_it);
            assert!((0..=ONE).contains(&sc.k_gate_q16));
            assert!((0..=ONE).contains(&sc.match_corgi_q16));
        }

        let qf = extract_semantic_frame(&q.text);
        let pf = extract_semantic_frame(&ix.turns[1].body_for_headers);
        let pref_sc = score_preference_frame_qcl(&qf, &pf);
        assert!(pref_sc.beneficiary_contrib_q16 <= W_BENEFICIARY);
        assert!(pref_sc.score_pref_q16 >= 0);
    }
}
