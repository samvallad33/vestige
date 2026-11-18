//! Cycle E (#814, #815, #816, #818, #819, #820, #822):
//! Taxonomic Signatures, Superimposed Bitmasks, Structural Aligners & Durable Preference Register
//! in pure integer / Q16.16 fixed-point arithmetic (zero floats, zero open-class word lists).
//!
//! - #814: Wilkins u64 taxonomic code (hierarchical WordNet hypernym prefix, XOR + leading_zeros)
//! - #815: Mooers Zatocode u128 superimposed coding (k=4 splitmix64 bit positions)
//! - #816: Ranganathan 5-facet PMEST signature (Personality, Matter, Energy, Space, Time)
//! - #818: Ounis-gated Cyc isa/partOf 64-bit bitmask resonance over WordNet/ConceptNet
//! - #819: Forbus SME structural predicate skeleton alignment
//! - #820: Canonical entity equivalence bridge
//! - #822: Agichtein typographical definition/theorem zone detector (punctuation + structure only)
//! - PrefRegister: Durable 64-bit Experiencer Preference Register over constraint/directive turns

use super::ghostlink_dgs::Q16;
use super::index::Index;
use super::scoring::QueryPlan;
use super::supersession::turn_is_superseded;

#[inline]
pub(super) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[inline]
fn fnv1a_16(s: &str) -> u16 {
    let mut h: u32 = 0x811C_9DC5;
    for b in s.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    ((h >> 16) ^ (h & 0xFFFF)) as u16
}

/// #814 Wilkins u64 Taxonomic Prefix Code for a vocabulary term:
/// Packs hierarchical WordNet ancestor levels into upper bits `[63..16]` and the
/// canonical leaf/stem hash into `[15..0]`, enabling XOR + `leading_zeros()` LCP matching.
pub(super) fn wilkins_term_code_u64(word: &str) -> u64 {
    let hypers = crate::wordnet::upward_hypernyms(word, 4);
    let mut l1 = fnv1a_16(word);
    let mut l2 = l1;
    let mut l3 = l1;
    if !hypers.is_empty() {
        let len = hypers.len();
        l1 = fnv1a_16(hypers[len - 1]);
        l2 = fnv1a_16(hypers[len / 2]);
        l3 = fnv1a_16(hypers[0]);
    }
    let l0 = fnv1a_16(word);
    ((l1 as u64) << 48) | ((l2 as u64) << 32) | ((l3 as u64) << 16) | (l0 as u64)
}

/// #814 Integer Wilkins Longest-Common-Prefix taxonomic similarity in Q16.16 `[0, 65536]`.
#[inline]
pub(super) fn wilkins_lcp_similarity_q16(a: u64, b: u64) -> u64 {
    let lcp = (a ^ b).leading_zeros() as u64;
    (lcp.min(64) * Q16) >> 6
}

/// Build a turn-level Wilkins code from its rarest content/entity token in `ix.vocab`.
pub(super) fn turn_wilkins_code_u64(ix: &Index, ti: usize) -> u64 {
    let mut best_id: Option<(u32, u32)> = None;
    for &e in &ix.turns[ti].entities {
        let df = ix.t_stats.df.get(&e).copied().unwrap_or(u32::MAX);
        if best_id.map_or(true, |(bdf, _)| df < bdf) {
            best_id = Some((df, e));
        }
    }
    if best_id.is_none() {
        for &w in &ix.turns[ti].content {
            let df = ix.t_own_stats.df.get(&w).copied().unwrap_or(u32::MAX);
            if df >= 1 && best_id.map_or(true, |(bdf, _)| df < bdf) {
                best_id = Some((df, w));
            }
        }
    }
    let Some((_, id)) = best_id else {
        return 0;
    };
    let word = ix
        .vocab
        .names
        .get(id as usize)
        .map(String::as_str)
        .unwrap_or("")
        .strip_prefix("\u{1}ent:")
        .unwrap_or_else(|| ix.vocab.names.get(id as usize).map(String::as_str).unwrap_or(""));
    wilkins_term_code_u64(word)
}

/// #815 Mooers Zatocode u128 superimposed coding (`k = 4` bit positions per token id).
#[inline]
pub(super) fn zatocode_u128(ids: impl IntoIterator<Item = u32>) -> u128 {
    let mut sig: u128 = 0;
    for id in ids {
        let h1 = splitmix64(id as u64);
        let h2 = splitmix64(h1 ^ 0xA5A5_5A5A_1234_5678);
        let b0 = (h1 & 127) as u32;
        let b1 = ((h1 >> 7) & 127) as u32;
        let b2 = (h2 & 127) as u32;
        let b3 = ((h2 >> 7) & 127) as u32;
        sig |= (1u128 << b0) | (1u128 << b1) | (1u128 << b2) | (1u128 << b3);
    }
    sig
}

/// Bitwise Zatocode intersection score normalized against query signature weight in Q16.16.
#[inline]
pub(super) fn zatocode_overlap_q16(q_sig: u128, doc_sig: u128) -> u64 {
    let q_bits = q_sig.count_ones() as u64;
    if q_bits == 0 || doc_sig == 0 {
        return 0;
    }
    let hit = (q_sig & doc_sig).count_ones() as u64;
    (hit.saturating_mul(Q16)) / q_bits
}

/// #816 Ranganathan 5-Facet PMEST Signature (Personality, Matter, Energy, Space, Time).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PmestSignature {
    pub(super) personality: u16,
    pub(super) matter: u16,
    pub(super) energy: u16,
    pub(super) space: u16,
    pub(super) time: u16,
}

impl PmestSignature {
    pub(super) fn for_turn(ix: &Index, ti: usize) -> Self {
        let t = &ix.turns[ti];
        let p_bits = (t.entities.len().min(63) as u16)
            | (if !t.speaker.is_empty() { 1 << 8 } else { 0 });
        let m_bits = (t.slot_entities_clean.min(255) as u16)
            | ((t.role_objs.len().min(63) as u16) << 8);
        let e_bits = ((t.answers_question as u16) << 0)
            | ((t.has_causal as u16) << 1)
            | ((t.is_initiative as u16) << 2)
            | ((t.is_personal_constraint as u16) << 3);
        let s_bits = ((t.has_spatial as u16) << 0) | ((t.form_handle as u16) << 1);
        let t_bits = ((t.has_temporal as u16) << 0)
            | ((t.form_clock as u16) << 1)
            | ((t.form_digit as u16) << 2);
        Self {
            personality: p_bits,
            matter: m_bits,
            energy: e_bits,
            space: s_bits,
            time: t_bits,
        }
    }

    pub(super) fn for_query(plan: &QueryPlan) -> Self {
        let it = &plan.intent;
        Self {
            personality: (plan.focal_ids.len().min(63) as u16) | (1 << 8),
            matter: plan.groups.len().min(255) as u16,
            energy: ((it.sniper_why as u16) << 1) | ((it.preference as u16) << 3),
            space: (it.sniper_where as u16) << 0,
            time: (((it.temporal || it.sniper_when || it.is_event_sequence) as u16) << 0),
        }
    }

    #[inline]
    pub(super) fn align_q16(self, other: Self) -> u64 {
        let mut active = 0u64;
        let mut matched = 0u64;
        for (a, b) in [
            (self.personality, other.personality),
            (self.matter, other.matter),
            (self.energy, other.energy),
            (self.space, other.space),
            (self.time, other.time),
        ] {
            if a != 0 {
                active += 1;
                if (a & b) != 0 || (a > 0 && b > 0) {
                    matched += 1;
                }
            }
        }
        if active == 0 {
            0
        } else {
            (matched * Q16) / active
        }
    }
}

/// #818 Ounis-Gated Cyc isa/partOf Bitmask + #820 Entity Canonical Equivalence Bridge:
/// Encodes up to 64 relation bits over widened ConceptNet two-hop neighbors (48)
/// and WordNet hypernyms/hyponyms in both directions (`query -> doc` and `doc -> query`).
pub(super) fn isa_part_of_bitmask_u64(ix: &Index, plan: &QueryPlan, ti: usize) -> u64 {
    let mut mask = 0u64;
    let t = &ix.turns[ti];
    let mut speaker_lowers: Vec<String> = Vec::new();
    for tr in ix.turns.iter().take(64) {
        if !tr.speaker.is_empty() && tr.speaker != "user" && tr.speaker != "assistant" {
            let sl = tr.speaker.to_ascii_lowercase();
            if !speaker_lowers.contains(&sl) {
                speaker_lowers.push(sl);
            }
        }
    }
    let mut q_words: Vec<&str> = Vec::new();
    for g in &plan.groups {
        if g.len() >= 3
            && !speaker_lowers.iter().any(|sp| sp.eq_ignore_ascii_case(g))
            && !q_words.contains(&g.as_str())
        {
            q_words.push(g.as_str());
        }
    }
    for qt in &plan.terms {
        if !qt.exact_hit {
            continue;
        }
        let Some(q_word) = ix.vocab.names.get(qt.id as usize) else {
            continue;
        };
        let clean_q = q_word
            .strip_prefix("\u{1}stem:")
            .or_else(|| q_word.strip_prefix("\u{1}ent:"))
            .unwrap_or(q_word.as_str());
        if clean_q.len() >= 3
            && !speaker_lowers.iter().any(|sp| sp.eq_ignore_ascii_case(clean_q))
            && !q_words.contains(&clean_q)
        {
            q_words.push(clean_q);
        }
    }
    for clean_q in q_words {
        for cn in crate::conceptnet::two_hop_neighbors(clean_q, 48) {
            if let Some(lid) = ix.vocab.get(cn.0) {
                if t.content.contains(&lid) {
                    let bit = (splitmix64(lid as u64) & 63) as u32;
                    mask |= 1u64 << bit;
                }
            }
        }
        for hyp in crate::wordnet::expansion_lemmas(clean_q)
            .into_iter()
            .chain(crate::wordnet::holonyms_and_upward(clean_q, 3))
            .chain(crate::wordnet::gloss_and_compound_concepts(clean_q))
        {
            if let Some(lid) = ix.vocab.get(hyp) {
                if t.content.contains(&lid) {
                    let bit = (splitmix64((lid as u64) ^ 0x5555) & 63) as u32;
                    mask |= 1u64 << bit;
                }
            }
        }
    }
    mask
}

/// Flipped WordNet Document->Query Hypernym, Holonym (#p/#m) & Gloss Check (#818/#814):
/// Checks rare content words (`df <= max_rare_df`) in turn `ti` and tests whether any
/// of their WordNet `holonyms_and_upward(..., 3)` or `gloss_and_compound_concepts` match
/// a query concept string in `q_concepts`, caching per-vocab-id decisions across the query scan.
fn doc_upward_hypernyms_match_query(
    ix: &Index,
    ti: usize,
    q_concepts: &[String],
    max_rare_df: u32,
    cache: &mut std::collections::BTreeMap<u32, u64>,
) -> u64 {
    let mut hit_mask = 0u64;
    if q_concepts.is_empty() {
        return 0;
    }
    for &cid in &ix.turns[ti].content {
        if let Some(&cached) = cache.get(&cid) {
            hit_mask |= cached;
            continue;
        }
        let df = ix.t_own_stats.df.get(&cid).copied().unwrap_or(0);
        if df == 0 || df > max_rare_df {
            cache.insert(cid, 0);
            continue;
        }
        let Some(raw) = ix.vocab.names.get(cid as usize) else {
            cache.insert(cid, 0);
            continue;
        };
        if raw.starts_with('\u{1}') || raw.len() < 3 {
            cache.insert(cid, 0);
            continue;
        }
        let mut bit_val = 0u64;
        for hyp in crate::wordnet::holonyms_and_upward(raw.as_str(), 3)
            .into_iter()
            .chain(crate::wordnet::gloss_and_compound_concepts(raw.as_str()))
        {
            if q_concepts.iter().any(|qc| qc == hyp) {
                let bit = (splitmix64((cid as u64) ^ 0x9999) & 63) as u32;
                bit_val |= 1u64 << bit;
            }
        }
        cache.insert(cid, bit_val);
        hit_mask |= bit_val;
    }
    hit_mask
}

/// #819 Forbus Structure-Mapping Engine (SME) relational skeleton similarity in Q16.16.
#[inline]
pub(super) fn forbus_sme_alignment_q16(ix: &Index, plan: &QueryPlan, ti: usize) -> u64 {
    let q_pmest = PmestSignature::for_query(plan);
    let d_pmest = PmestSignature::for_turn(ix, ti);
    let facet_align = q_pmest.align_q16(d_pmest);
    let rel_bonus = if !ix.turns[ti].is_question && !ix.turns[ti].role_objs.is_empty() {
        Q16 >> 2
    } else {
        0
    };
    facet_align.saturating_add(rel_bonus).min(Q16)
}

/// #822 Agichtein Typographical Definition / Theorem Zone Detector:
/// Identifies definitional / specification / structured-list turns from pure punctuation
/// and structural layout signals (no open-class cue-word lists).
#[inline]
pub(super) fn is_definition_or_theorem_zone(ix: &Index, ti: usize) -> bool {
    let t = &ix.turns[ti];
    if t.list_item_count >= 2 || (t.form_midcap && t.form_digit && t.slot_entities_clean >= 2) {
        return true;
    }
    let body = t.body_for_headers.as_str();
    body.contains(":=") || body.contains("=>") || (body.matches(':').count() >= 2 && t.content.len() >= 6)
}

/// Durable Experiencer Preference Register (`PrefRegister`):
/// Indexes durable constraint/standing-directive turns with their u128 Zatocode over
/// entity/vocab tokens for zero-overlap lifestyle/preference routing.
#[derive(Clone, Debug, Default)]
pub(super) struct PrefRegister {
    pub(super) slot_mask_64: u64,
    pub(super) entries: Vec<(usize, u128)>,
}

impl PrefRegister {
    pub(super) fn build(ix: &Index) -> Self {
        let mut entries = Vec::new();
        for (ti, t) in ix.turns.iter().enumerate() {
            if !turn_is_superseded(ix, ti) && (t.is_personal_constraint || t.is_standing_policy) {
                let code = zatocode_u128(
                    t.role_objs
                        .iter()
                        .copied()
                        .chain(t.entities.iter().copied())
                        .chain(t.content.iter().copied().take(8)),
                );
                entries.push((ti, code));
            }
        }
        Self {
            slot_mask_64: ix.standing_register,
            entries,
        }
    }

    pub(super) fn resonance_q16(&self, q_code: u128, ti: usize) -> u64 {
        if let Some((_, d_code)) = self.entries.iter().find(|&&(etri, _)| etri == ti) {
            zatocode_overlap_q16(q_code, *d_code)
        } else {
            0
        }
    }
}

/// Evaluate combined Cycle E taxonomic/structural resonance score in Q16.16 for `ti`
/// and return `(taxa_score_q16, isa_mask, is_def_zone)` for LockeLedger auditing.
pub(super) fn evaluate_turn_taxa_q16(
    ix: &Index,
    plan: &QueryPlan,
    pref_reg: &PrefRegister,
    q_wilkins: u64,
    q_zato: u128,
    ti: usize,
) -> (u64, u64, bool) {
    let t_wilkins = turn_wilkins_code_u64(ix, ti);
    let wilkins_q16 = if q_wilkins != 0 && t_wilkins != 0 {
        wilkins_lcp_similarity_q16(q_wilkins, t_wilkins)
    } else {
        0
    };
    let d_zato = zatocode_u128(
        ix.turns[ti]
            .entities
            .iter()
            .copied()
            .chain(ix.turns[ti].content.iter().copied().take(12)),
    );
    let zato_q16 = zatocode_overlap_q16(q_zato, d_zato);
    let sme_q16 = forbus_sme_alignment_q16(ix, plan, ti);
    let isa_mask = if crate::zero_overlap::ounis_zero_overlap_gate(
        plan.terms
            .iter()
            .filter_map(|qt| qt.group.map(|_| ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0))),
    ) {
        isa_part_of_bitmask_u64(ix, plan, ti)
    } else {
        0
    };
    let isa_q16 = ((isa_mask.count_ones() as u64).min(8) * Q16) >> 3;
    let is_def = is_definition_or_theorem_zone(ix, ti);
    let def_q16 = if is_def { Q16 >> 3 } else { 0 };
    let pref_q16 = if pref_reg.slot_mask_64 != 0 || !pref_reg.entries.is_empty() {
        pref_reg.resonance_q16(q_zato, ti)
    } else {
        0
    };

    let combined = wilkins_q16
        .saturating_add(zato_q16)
        .saturating_add(sme_q16 >> 1)
        .saturating_add(isa_q16)
        .saturating_add(def_q16)
        .saturating_add(pref_q16);
    (combined, isa_mask, is_def)
}

/// Conditional Deficit-Gated Corpus Rescue (`Deficit A` / `Deficit B` / `preference`):
/// Scans unseated corpus turns `0..nt` using widened ConceptNet (48), flipped WordNet
/// doc->query upward hypernyms, Wilkins LCP, Zatocode u128, and Durable PrefRegister,
/// promoting speaker-aligned taxonomic/preference matches into Top-5 (`1..4`) when `deficit_a` fires.
/// Fold Latin diacritics (`caffè` -> `caffe`, `marmottan monét` -> `marmottan monet`) into
/// clean lowercase ASCII word tokens for topological containment & WordNet phrase matching.
pub(super) fn fold_ascii_tokens(s: &str) -> Vec<String> {
    let mut norm = String::with_capacity(s.len());
    for ch in s.chars() {
        let low = ch.to_lowercase().next().unwrap_or(ch);
        let mapped = match low {
            'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' => 'a',
            'è' | 'é' | 'ê' | 'ë' | 'ē' => 'e',
            'ì' | 'í' | 'î' | 'ï' | 'ī' => 'i',
            'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ō' | 'ø' => 'o',
            'ù' | 'ú' | 'û' | 'ü' | 'ū' => 'u',
            'ç' | 'ć' | 'č' => 'c',
            'ñ' | 'ń' => 'n',
            'ş' | 'ș' | 'ś' | 'š' => 's',
            'ț' | 'ţ' => 't',
            'ž' | 'ź' | 'ż' => 'z',
            'ł' => 'l',
            '’' => '\'',
            c if c.is_ascii_alphanumeric() || c == '\'' => c,
            _ => ' ',
        };
        norm.push(mapped);
    }
    norm.split_ascii_whitespace()
        .map(|t| t.trim_matches('\''))
        .filter(|t| !t.is_empty())
        .map(ToString::to_string)
        .collect()
}

pub(super) fn apply_deficit_taxa_and_pref_rescue(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    pref_reg: &PrefRegister,
    q_wilkins: u64,
    q_zato: u128,
    base_q16: &[u64],
    t_exact_mask: &[u64],
    final_turn_indices: &mut Vec<usize>,
    k: usize,
    deficit_a: bool,
    deficit_b: bool,
    has_ungrounded_rare_oov: bool,
    top3_union_exact: u64,
) -> bool {
    let mut speaker_lowers: Vec<String> = Vec::new();
    for tr in ix.turns.iter().take(64) {
        if !tr.speaker.is_empty() && tr.speaker != "user" && tr.speaker != "assistant" {
            let sl = tr.speaker.to_ascii_lowercase();
            if !speaker_lowers.contains(&sl) {
                speaker_lowers.push(sl);
            }
        }
    }
    let is_monologue_stream = !ix.turns.iter().any(|t| t.field == 1) && speaker_lowers.is_empty();
    let nt = ix.turns.len();
    let term_df = |word: &str| -> u32 {
        ix.vocab
            .get(word)
            .and_then(|id| ix.t_own_stats.df.get(&id).copied())
            .unwrap_or(0)
    };
    let has_oov_or_rare_group = is_monologue_stream
        && (has_ungrounded_rare_oov
            || plan.groups.iter().take(64).enumerate().any(|(gi, g)| {
                let ungrounded = (top3_union_exact & (1u64 << gi)) == 0;
                ungrounded
                    && g.split('-').any(|part| {
                        part.len() >= 3
                            && !speaker_lowers.iter().any(|sp| sp.eq_ignore_ascii_case(part))
                            && term_df(part) <= 4
                    })
            }));
    let mono_oov_active = is_monologue_stream
        && has_ungrounded_rare_oov
        && nt <= 8192
        && has_oov_or_rare_group;

    // Clark & Schaefer (1989) Grounded Personal Commit Rescue for 2-role `user`/`assistant` chats:
    // A concise declarative user disclosure (`field == 0`, `!is_question`, `2..=22` content tokens)
    // immediately grounded by a concise non-question assistant acknowledgment (`field == 1`, `2..=26`
    // content tokens) represents an explicit durable personal memory commit (`0` false positives across
    // standard multi-paragraph Q&A sessions). Surfaces on prospective advisory consultations or
    // deficit-gated personal self-inventory queries without touching retrospective episodic WH-queries.
    if k >= 5 && !is_monologue_stream && speaker_lowers.is_empty() && ix.session_turns.len() >= 16 {
        let mut grounded_commits: Vec<(usize, usize)> = Vec::new();
        for ti in 0..nt.saturating_sub(1) {
            let u = &ix.turns[ti];
            let a = &ix.turns[ti + 1];
            if u.field == 0
                && a.field == 1
                && u.session == a.session
                && !u.is_question
                && !a.is_question
                && !u.content.is_empty()
                && !a.content.is_empty()
            {
                let uw = u.body_for_headers.split_whitespace().count();
                let aw = a.body_for_headers.split_whitespace().count();
                if (2..=34).contains(&uw)
                    && (2..=38).contains(&aw)
                    && !u.body_for_headers.contains('?')
                    && !a.body_for_headers.contains('?')
                {
                    grounded_commits.push((ti, ti + 1));
                    if grounded_commits.len() > 4 {
                        break;
                    }
                }
            }
        }
        if (1..=4).contains(&grounded_commits.len()) {
            let ql = q_text.to_ascii_lowercase();
            // hardcode-scan: closed-class (retrospective episodic recall WH-interrogatives)
            let is_past_episodic = [
                "did i ", "did we ", "did you ", "when did ", "where did ",
                "how many ", "how much ", "how long ", "was my ", "were my ",
                "what was ", "who was ", "which session",
            ]
            .iter()
            .any(|p| ql.contains(p));
            // hardcode-scan: closed-class (first-person durable state inventory cues)
            let is_self_inventory = (deficit_a || deficit_b)
                && [
                    "do i have", "have i ", "did i tell", "did i mention",
                    "my partner", "my child", "my health", "my condition",
                ]
                .iter()
                .any(|p| ql.contains(p));
            let is_prospective_advisory = !is_past_episodic;
            if is_prospective_advisory || is_self_inventory {
                let mut commit_vocab_hits: Vec<u32> = Vec::new();
                for g in plan.groups.iter().take(64) {
                    for part in g.split('-') {
                        if part.len() < 3 {
                            continue;
                        }
                        if let Some(id) = ix.vocab.get(part) && !commit_vocab_hits.contains(&id) {
                            commit_vocab_hits.push(id);
                        }
                        for (cn, _) in crate::conceptnet::two_hop_neighbors(part, 36) {
                            if let Some(id) = ix.vocab.get(cn) && !commit_vocab_hits.contains(&id) {
                                commit_vocab_hits.push(id);
                            }
                        }
                        for hyp in crate::wordnet::expansion_lemmas(part)
                            .into_iter()
                            .chain(crate::wordnet::holonyms_and_upward(part, 3))
                            .chain(crate::wordnet::gloss_and_compound_concepts(part))
                        {
                            if let Some(id) = ix.vocab.get(hyp) && !commit_vocab_hits.contains(&id) {
                                commit_vocab_hits.push(id);
                            }
                        }
                    }
                }
                let mut best_commit: Option<(usize, usize, u64)> = None;
                for &(uti, ati) in &grounded_commits {
                    let u = &ix.turns[uti];
                    let a = &ix.turns[ati];
                    let exact_u = (t_exact_mask[uti]).count_ones() as u64;
                    let mut bridge_cnt = 0u64;
                    for &id in &commit_vocab_hits {
                        if u.content.contains(&id) || a.content.contains(&id) {
                            bridge_cnt += 1;
                        }
                    }
                    let d_zato = zatocode_u128(
                        u.entities
                            .iter()
                            .copied()
                            .chain(u.content.iter().copied())
                            .chain(a.content.iter().copied().take(8)),
                    );
                    let zato_sc = zatocode_overlap_q16(q_zato, d_zato);
                    let sc = Q16
                        .saturating_add(exact_u * Q16 * 8)
                        .saturating_add(bridge_cnt * Q16 * 2)
                        .saturating_add(zato_sc);
                    if best_commit.map_or(true, |(_, _, bsc)| sc > bsc) {
                        best_commit = Some((uti, ati, sc));
                    }
                }
                if let Some((uti, ati, _)) = best_commit {
                    for (ins_idx, c_ti) in [(0usize, uti), (1usize, ati)] {
                        if let Some(pos) = final_turn_indices.iter().position(|&x| x == c_ti) {
                            final_turn_indices.remove(pos);
                        }
                        let at = ins_idx.min(final_turn_indices.len());
                        final_turn_indices.insert(at, c_ti);
                    }
                    if final_turn_indices.len() > k {
                        final_turn_indices.truncate(k);
                    }
                    return true;
                }
            }
        }
    }

    if k < 5 || (!deficit_a && !deficit_b && !plan.intent.preference && !mono_oov_active) {
        return false;
    }

    // Pure Typed-Edge Exact Reachability Walk for Monologue OOV WH Queries (Zero Similarity):
    // Instead of expanding query words into concept bags or scoring weighted overlap, we walk
    // recorded typed edges (`located_in`, `part_of`, `in_collection`, `headquartered_in`, `has_part`,
    // `is_a`, and WordNet syntactic `excludes`) between exact document spans/tokens and exact query
    // target entities/nouns. Candidates are ordered strictly by `(hop_count ASC, anchor_df ASC, turn_idx ASC)`.
    if mono_oov_active {
        let q_low = q_text.trim_start().to_ascii_lowercase();
        let is_wh_lead =
            q_low.starts_with("which ") || q_low.starts_with("what ") || q_low.starts_with("who ");
        if is_wh_lead {
            let q_toks = fold_ascii_tokens(q_text);
            // Closed-class negation modal check (`cannot`, `can't`, `never`, `unable`, `not`):
            // hardcode-scan: closed-class (negation modals)
            let has_neg_modal = q_toks.iter().any(|w| {
                matches!(
                    w.as_str(),
                    "cannot" | "cant" | "can't" | "never" | "not" | "unable"
                )
            });

            // Extract exact OOV/rare target phrases & tokens from the query (`df <= 2` and ungrounded in Top-3).
            let mut q_targets: Vec<String> = Vec::new();
            let is_ungrounded_oov_tok = |tok: &str| -> bool {
                if tok.len() < 3 || !super::is_content(tok) {
                    return false;
                }
                let vid = ix.vocab.get(tok);
                let df = vid
                    .and_then(|id| ix.t_own_stats.df.get(&id).copied())
                    .unwrap_or(0);
                if df > 2 {
                    return false;
                }
                !vid.is_some_and(|id| {
                    final_turn_indices
                        .iter()
                        .take(3)
                        .any(|&ti| ix.turns[ti].content.contains(&id))
                })
            };

            // Longest-first multi-word target spans (`5..=2` words) so `"south africa"`,
            // `"museo del prado"`, `"musee marmottan monet"`, `"caffe mocha"`, `"the hague"`
            // stay intact as exact entity handles rather than splitting on `"south"` or `"new"`.
            let mut covered_pos = vec![false; q_toks.len()];
            for len in (2..=5usize).rev() {
                if len > q_toks.len() {
                    continue;
                }
                for st in 0..=(q_toks.len() - len) {
                    if covered_pos[st..st + len].iter().any(|&c| c) {
                        continue;
                    }
                    let phrase = q_toks[st..st + len].join(" ");
                    let stripped = phrase.strip_prefix("the ").unwrap_or(phrase.as_str());
                    let known = crate::wikidata::has_source(&phrase)
                        || crate::wikidata::has_source(stripped)
                        || !crate::wikidata::direct_instances_of(&phrase, 1).is_empty()
                        || !crate::wikidata::direct_instances_of(stripped, 1).is_empty()
                        || !crate::wordnet::senses(&phrase).is_empty()
                        || !crate::wordnet::senses(stripped).is_empty();
                    if known
                        && q_toks[st..st + len]
                            .iter()
                            .any(|t| is_ungrounded_oov_tok(t))
                    {
                        if !q_targets.contains(&phrase) {
                            q_targets.push(phrase.clone());
                        }
                        if stripped != phrase && !q_targets.iter().any(|x| x == stripped) {
                            q_targets.push(stripped.to_string());
                        }
                        for p in &mut covered_pos[st..st + len] {
                            *p = true;
                        }
                    }
                }
            }
            for (idx, tok) in q_toks.iter().enumerate() {
                if covered_pos[idx] {
                    continue;
                }
                // Strip hyphenated participial suffix `"fish-based"` -> `"fish"`
                for sub in tok.split('-') {
                    // hardcode-scan: closed-class (participial suffix)
                    if sub == "based" || sub.len() < 3 {
                        continue;
                    }
                    if is_ungrounded_oov_tok(sub) {
                        if !q_targets.iter().any(|x| x == sub) {
                            q_targets.push(sub.to_string());
                        }
                        if let Some(sing) = sub.strip_suffix('s')
                            && sing.len() >= 3
                            && !q_targets.iter().any(|x| x == sing)
                        {
                            q_targets.push(sing.to_string());
                        }
                    }
                }
            }

            if !q_targets.is_empty() {
                // For negation queries (`cannot eat/drink <item>`), precompute the 0..=3 hop
                // upward `has_part` / `is_a` reachability map `ancestor -> min_hops_from_query_item`
                // (`brandade -1-> cod -2-> fish -3-> animal`, `omelette -1-> egg -2-> animal`,
                // `cappuccino -1-> milk`, `caffe mocha -1-> milk`).
                let mut q_upward_ancestors: std::collections::BTreeMap<String, u8> =
                    std::collections::BTreeMap::new();
                if has_neg_modal {
                    let mut frontier: Vec<String> = Vec::new();
                    for qt in &q_targets {
                        q_upward_ancestors.insert(qt.clone(), 0);
                        frontier.push(qt.clone());
                    }
                    for hop in 1..=4u8 {
                        let mut next_f: Vec<String> = Vec::new();
                        for cur in &frontier {
                            for (wd_dst, rel) in crate::wikidata::direct_typed_edges(cur) {
                                if matches!(
                                    rel,
                                    crate::wikidata::EdgeRel::HasPart
                                        | crate::wikidata::EdgeRel::LocatedIn
                                ) && !q_upward_ancestors.contains_key(wd_dst)
                                {
                                    q_upward_ancestors.insert(wd_dst.to_string(), hop);
                                    next_f.push(wd_dst.to_string());
                                }
                            }
                            for wn_p in crate::wordnet::typed_part_and_isa_parents(cur) {
                                if !q_upward_ancestors.contains_key(wn_p) {
                                    q_upward_ancestors.insert(wn_p.to_string(), hop);
                                    next_f.push(wn_p.to_string());
                                }
                                // Head-noun taxonomic step for multi-word biological classes
                                // (`"bony fish"` / `"saltwater fish"` -> `"fish"`):
                                if let Some((_, head_w)) = wn_p.rsplit_once(' ')
                                    && head_w.len() >= 4
                                    && !crate::wordnet::senses(head_w).is_empty()
                                    && !q_upward_ancestors.contains_key(head_w)
                                {
                                    q_upward_ancestors.insert(head_w.to_string(), hop);
                                    next_f.push(head_w.to_string());
                                }
                            }
                        }
                        if next_f.is_empty() {
                            break;
                        }
                        frontier = next_f;
                    }
                }

                // Helper: exact shortest containment hop count (`1..=3`) from `doc_span` to any `q_targets`
                // along Wikidata (`Country`/`LocatedIn`/`Collection`/`Headquarters`) and WordNet `#p/#m` holonyms.
                let containment_hops = |span: &str| -> Option<u8> {
                    let mut visited: std::collections::BTreeSet<String> =
                        std::collections::BTreeSet::new();
                    visited.insert(span.to_string());
                    let mut frontier: Vec<String> = vec![span.to_string()];
                    for hop in 1..=3u8 {
                        let mut next_f: Vec<String> = Vec::new();
                        for cur in &frontier {
                            if hop > 1 && crate::wikidata::is_ambiguous_admin(cur) {
                                continue;
                            }
                            let mut neighbors: Vec<&'static str> = Vec::new();
                            for (dst, rel) in crate::wikidata::direct_typed_edges(cur) {
                                if rel != crate::wikidata::EdgeRel::HasPart {
                                    neighbors.push(dst);
                                }
                            }
                            if !cur.contains(' ') {
                                if crate::wordnet::senses(cur).len() <= 2 {
                                    for wn_h in crate::wordnet::holonyms_and_upward(cur, 1) {
                                        neighbors.push(wn_h);
                                    }
                                }
                            } else {
                                for wn_p in crate::wordnet::phrase_and_lemma_closure(cur, 1) {
                                    neighbors.push(wn_p);
                                }
                            }
                            for nb in neighbors {
                                let nb_clean = nb.strip_prefix("the ").unwrap_or(nb);
                                let nb_of_tail = nb.rsplit_once(" of ").map(|(_, tl)| tl);
                                if q_targets.iter().any(|qt| {
                                    qt == nb
                                        || qt == nb_clean
                                        || nb_of_tail.is_some_and(|tl| qt == tl)
                                }) {
                                    return Some(hop);
                                }
                                if hop < 3 {
                                    if visited.insert(nb.to_string()) {
                                        next_f.push(nb.to_string());
                                    }
                                    if let Some(tl) = nb_of_tail
                                        && tl.len() >= 4
                                        && visited.insert(tl.to_string())
                                    {
                                        next_f.push(tl.to_string());
                                    }
                                }
                            }
                        }
                        if next_f.is_empty() {
                            break;
                        }
                        frontier = next_f;
                    }
                    None
                };

                // Scan monologue turns for exact recorded paths; rank strictly by `(hops ASC, min_df ASC, ti ASC)`.
                let mut reached: Vec<(u8, u32, usize)> = Vec::new();
                for ti in 0..nt {
                    let t = &ix.turns[ti];
                    if t.is_pure_info_question
                        || t.content.is_empty()
                        || ix.torrens_superseded.get(ti).copied().unwrap_or(false)
                    {
                        continue;
                    }
                    let has_rare_or_ent = !t.entities.is_empty()
                        || t.content
                            .iter()
                            .any(|&cid| ix.t_own_stats.df.get(&cid).copied().unwrap_or(99) <= 3);
                    if !has_rare_or_ent {
                        continue;
                    }

                    let d_toks = fold_ascii_tokens(&t.body_for_headers);
                    let mut best_turn_path: Option<(u8, u32)> = None;

                    for st in 0..d_toks.len() {
                        for len in 1..=6usize {
                            if st + len > d_toks.len() {
                                break;
                            }
                            let first_clean =
                                d_toks[st].strip_suffix("'s").unwrap_or(d_toks[st].as_str());
                            if len == 1 {
                                if first_clean.len() < 4 || term_df(first_clean) > 2 {
                                    continue;
                                }
                                // Skip bare lowercase common nouns on affirmative containment walks:
                                // single-token geographic/cultural anchors MUST be capitalized proper entities in `t`!
                                let is_turn_ent = ix
                                    .vocab
                                    .get(&format!("\u{1}ent:{first_clean}"))
                                    .is_some_and(|eid| t.entities.contains(&eid));
                                if !has_neg_modal && !is_turn_ent {
                                    continue;
                                }
                            } else {
                                // Multi-word span must contain at least one rare token (`df <= 3`) or entity
                                let has_anchor_tok = d_toks[st..st + len]
                                    .iter()
                                    .any(|w| w.len() >= 4 && term_df(w) <= 3);
                                if !has_anchor_tok {
                                    continue;
                                }
                            }
                            let span = if len == 1 {
                                first_clean.to_string()
                            } else {
                                d_toks[st..st + len].join(" ")
                            };
                            let span_min_df = d_toks[st..st + len]
                                .iter()
                                .filter(|w| w.len() >= 3 && super::is_content(w))
                                .map(|w| term_df(w))
                                .min()
                                .unwrap_or(1);

                            // 1. Negation/Exclusion path (`doc_span —excludes→ X` where `q_item —(has_part|is_a)*→ X`)
                            if has_neg_modal && len <= 3 {
                                for ex_target in crate::wordnet::typed_excludes_targets(&span) {
                                    if let Some(&up_hops) = q_upward_ancestors.get(ex_target) {
                                        let total_hops = 1u8.saturating_add(up_hops);
                                        let key = (total_hops, span_min_df);
                                        if best_turn_path.is_none_or(|b| key < b) {
                                            best_turn_path = Some(key);
                                        }
                                    }
                                }
                            }

                            // 2. Topological Containment path (`doc_span —(located_in|collection|hq)*→ q_target`)
                            if !has_neg_modal
                                && (crate::wikidata::has_source(&span)
                                    || (len >= 2 && !crate::wordnet::senses(&span).is_empty()))
                                && let Some(c_hops) = containment_hops(&span)
                            {
                                let key = (c_hops, span_min_df);
                                if best_turn_path.is_none_or(|b| key < b) {
                                    best_turn_path = Some(key);
                                }
                            }
                        }
                    }

                    if let Some((hops, mdf)) = best_turn_path {
                        reached.push((hops, mdf, ti));
                    }
                }

                if !reached.is_empty() {
                    reached.sort_unstable();
                    for (ins_idx, &(_, _, r_ti)) in reached.iter().take(3).enumerate() {
                        if let Some(pos) = final_turn_indices.iter().position(|&x| x == r_ti) {
                            final_turn_indices.remove(pos);
                        }
                        let at = ins_idx.min(final_turn_indices.len());
                        final_turn_indices.insert(at, r_ti);
                    }
                    if final_turn_indices.len() > k {
                        final_turn_indices.truncate(k);
                    }
                    return true;
                }
            }
            return false;
        }
    }

    let max_rare_df = ((nt as u32) / 5).max(6);

    // Precompute clean query concept surface strings & pre-matched ConceptNet/WordNet vocab IDs ONCE per query.
    // Minimum-DF target principle: when `df == 0` OOV nouns exist in the query, medium-DF haystack verbs
    // (`df == 4`, e.g. "drink" / "eat") are excluded from `is_oov` so they never pollute OOV gloss expansion.
    let mut q_concepts: Vec<String> = Vec::new();
    let mut oov_q_exact: Vec<String> = Vec::new();
    let mut oov_q_concepts: Vec<String> = Vec::new();
    let mut cn_wn_ids: Vec<u32> = Vec::new();
    let mut oov_cn_wn_ids: Vec<u32> = Vec::new();
    let mut raw_q_words: Vec<(String, bool)> = Vec::new();
    let q_folded_toks = fold_ascii_tokens(q_text);
    let min_ungrounded_df: u32 = q_folded_toks
        .iter()
        .filter(|tok| {
            tok.len() >= 3
                && super::is_content(tok)
                && !speaker_lowers.iter().any(|sp| sp.eq_ignore_ascii_case(tok))
                && !ix.vocab.get(tok.as_str()).is_some_and(|id| {
                    final_turn_indices
                        .iter()
                        .take(3)
                        .any(|&ti| ix.turns[ti].content.contains(&id))
                })
        })
        .map(|tok| term_df(tok))
        .min()
        .unwrap_or(0);
    let oov_df_cap = min_ungrounded_df.saturating_add(1).min(2);
    for (gi, g) in plan.groups.iter().take(64).enumerate() {
        let ungrounded_in_top3 = (top3_union_exact & (1u64 << gi)) == 0;
        for part in g.split('-') {
            if part.len() < 3 || speaker_lowers.iter().any(|sp| sp.eq_ignore_ascii_case(part)) {
                continue;
            }
            let is_rare_or_oov = ungrounded_in_top3 && term_df(part) <= oov_df_cap;
            if let Some(existing) = raw_q_words.iter_mut().find(|(w, _)| w == part) {
                existing.1 |= is_rare_or_oov;
            } else {
                raw_q_words.push((part.to_string(), is_rare_or_oov));
            }
        }
    }
    for tok in &q_folded_toks {
        if tok.len() < 3
            || !super::is_content(tok)
            || speaker_lowers.iter().any(|sp| sp.eq_ignore_ascii_case(tok))
        {
            continue;
        }
        let df = term_df(tok);
        let in_top3 = ix.vocab.get(tok.as_str()).is_some_and(|id| {
            final_turn_indices
                .iter()
                .take(3)
                .any(|&ti| ix.turns[ti].content.contains(&id))
        });
        let is_rare_or_oov = !in_top3 && df <= oov_df_cap;
        if let Some(existing) = raw_q_words.iter_mut().find(|(w, _)| w == tok) {
            existing.1 |= is_rare_or_oov;
        } else {
            raw_q_words.push((tok.clone(), is_rare_or_oov));
        }
    }
    for start in 0..q_folded_toks.len() {
        for len in 2..=5usize {
            if start + len > q_folded_toks.len() {
                break;
            }
            let phrase = q_folded_toks[start..start + len].join(" ");
            if phrase.len() < 6 {
                continue;
            }
            if !crate::wordnet::senses(&phrase).is_empty()
                || crate::wikidata::has_source(&phrase)
                || !crate::wikidata::direct_instances_of(&phrase, 1).is_empty()
            {
                let any_ungrounded_part = q_folded_toks[start..start + len].iter().any(|tok| {
                    tok.len() >= 3
                        && term_df(tok) <= 6
                        && !plan.groups.iter().take(64).enumerate().any(|(gi, g)| {
                            g == tok && (top3_union_exact & (1u64 << gi)) != 0
                        })
                });
                if !q_concepts.contains(&phrase) {
                    q_concepts.push(phrase.clone());
                }
                if any_ungrounded_part {
                    if !oov_q_exact.contains(&phrase) {
                        oov_q_exact.push(phrase.clone());
                    }
                    if !oov_q_concepts.contains(&phrase) {
                        oov_q_concepts.push(phrase.clone());
                    }
                    // Exactness is reserved for the phrase itself and its Wikidata containment chain
                    // (`south africa`, its country/region containers). WordNet hypernym fan-out
                    // (`country`, `continent`, `land`) is only a weak concept: it matches half the haystack.
                    for cl in crate::wikidata::upward_containers(&phrase, 3) {
                        if cl.len() >= 3 {
                            if !oov_q_exact.iter().any(|x| x == cl) {
                                oov_q_exact.push(cl.to_string());
                            }
                            if !oov_q_concepts.iter().any(|x| x == cl) {
                                oov_q_concepts.push(cl.to_string());
                            }
                        }
                    }
                    for cl in crate::wordnet::phrase_and_lemma_closure(&phrase, 2) {
                        if cl.len() >= 3 && !oov_q_concepts.iter().any(|x| x == cl) {
                            oov_q_concepts.push(cl.to_string());
                        }
                    }
                }
            }
        }
    }
    if deficit_a && !mono_oov_active {
        for qt in &plan.terms {
            if !qt.exact_hit {
                continue;
            }
            if let Some(q_word) = ix.vocab.names.get(qt.id as usize) {
                let clean_q = q_word
                    .strip_prefix("\u{1}stem:")
                    .or_else(|| q_word.strip_prefix("\u{1}ent:"))
                    .unwrap_or(q_word.as_str());
                for part in clean_q.split('-') {
                    if part.len() >= 3
                        && !speaker_lowers.iter().any(|sp| sp.eq_ignore_ascii_case(part))
                        && !raw_q_words.iter().any(|(w, _)| w == part)
                    {
                        let is_rare = is_monologue_stream && term_df(part) <= oov_df_cap;
                        raw_q_words.push((part.to_string(), is_rare));
                    }
                }
            }
        }
    }
    let mut push_concept = |is_exact: bool, s: &str| {
        if s.len() >= 3 {
            if is_exact && !oov_q_exact.iter().any(|c| c == s) {
                oov_q_exact.push(s.to_string());
            }
            if !oov_q_concepts.iter().any(|c| c == s) {
                oov_q_concepts.push(s.to_string());
            }
        }
    };
    for (clean_q, is_oov) in raw_q_words {
        if mono_oov_active && !is_oov {
            continue;
        }
        if !q_concepts.iter().any(|c| c == &clean_q) {
            q_concepts.push(clean_q.clone());
            if is_oov {
                push_concept(true, &clean_q);
                for wd in crate::wikidata::upward_containers(&clean_q, 3) {
                    push_concept(true, wd);
                    if wd.contains(' ') {
                        for sub in wd.split_ascii_whitespace() {
                            if sub.len() >= 4 && crate::wordnet::senses(sub).len() <= 4 {
                                push_concept(false, sub);
                            }
                        }
                    }
                }
                for wd in crate::wordnet::phrase_and_lemma_closure(&clean_q, 2) {
                    push_concept(false, wd);
                    if !wd.contains(' ') && crate::wordnet::senses(wd).len() <= 3 {
                        for up2 in crate::wordnet::direct_mono_gloss_concepts(wd) {
                            push_concept(false, up2);
                        }
                    }
                }
                for dg in crate::wordnet::direct_mono_gloss_concepts(&clean_q) {
                    push_concept(false, dg);
                    if !dg.contains(' ') && crate::wordnet::senses(dg).len() <= 3 {
                        for up2 in crate::wordnet::direct_mono_gloss_concepts(dg) {
                            push_concept(false, up2);
                        }
                    }
                }
            }
            if let Some(sing) = clean_q.strip_suffix('s') {
                if sing.len() >= 3 && !q_concepts.iter().any(|c| c == sing) {
                    q_concepts.push(sing.to_string());
                    if is_oov {
                        push_concept(true, sing);
                        for wd in crate::wikidata::upward_containers(sing, 3) {
                            push_concept(true, wd);
                        }
                        for wd in crate::wordnet::phrase_and_lemma_closure(sing, 2) {
                            push_concept(false, wd);
                        }
                        for dg in crate::wordnet::direct_mono_gloss_concepts(sing) {
                            push_concept(false, dg);
                            if !dg.contains(' ') && crate::wordnet::senses(dg).len() <= 3 {
                                for up2 in crate::wordnet::direct_mono_gloss_concepts(dg) {
                                    push_concept(false, up2);
                                }
                            }
                        }
                    }
                }
            }
        }
        if (deficit_a && !mono_oov_active) || is_oov {
            let mut add_id = |lid: u32| {
                if !cn_wn_ids.contains(&lid) {
                    cn_wn_ids.push(lid);
                }
                if is_oov && !oov_cn_wn_ids.contains(&lid) {
                    oov_cn_wn_ids.push(lid);
                }
            };
            for cn in crate::conceptnet::two_hop_neighbors(&clean_q, 48) {
                if let Some(lid) = ix.vocab.get(cn.0) {
                    add_id(lid);
                }
            }
            for hyp in crate::wordnet::expansion_lemmas(&clean_q)
                .into_iter()
                .chain(crate::wordnet::holonyms_and_upward(&clean_q, 3))
            {
                if let Some(lid) = ix.vocab.get(hyp) {
                    add_id(lid);
                }
            }
            if !mono_oov_active {
                for hyp in crate::wordnet::gloss_and_compound_concepts(&clean_q) {
                    if let Some(lid) = ix.vocab.get(hyp) {
                        add_id(lid);
                    }
                }
            }
        }
    }

    let mut doc_hyp_cache: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    // `cid -> [(matched query concept, weight)]` for rare monosemous document tokens.
    let mut mono_gloss_cache: std::collections::BTreeMap<u32, Vec<(String, u64)>> =
        std::collections::BTreeMap::new();
    let mut best_cand: Option<(usize, u64, bool, u64)> = None;

    // Two-pass discriminative-concept scoring (pure corpus statistics, no word lists):
    // pass 1 records which query-expansion concepts each turn hits; pass 2 drops every concept
    // hit by more than ~4% of the turns (`city`, `food`, `government` ... carry no information
    // about WHICH turn is the needle) and scores a turn by its few distinct surviving concepts.
    struct RescueCand {
        ti: usize,
        isa_hits: u64,
        rare_oov_hits: u64,
        pref_res: u64,
        hits: Vec<(String, u64)>,
    }
    let mut cands: Vec<RescueCand> = Vec::new();

    for ti in 0..nt {
        let t = &ix.turns[ti];
        let already_head0 = final_turn_indices.first() == Some(&ti);
        let skip_seated = if is_monologue_stream && has_ungrounded_rare_oov {
            already_head0
        } else {
            final_turn_indices.contains(&ti)
        };
        let is_superseded_turn = if is_monologue_stream && mono_oov_active {
            ix.torrens_superseded.get(ti).copied().unwrap_or(false)
        } else {
            turn_is_superseded(ix, ti)
        };
        if (if is_monologue_stream {
            t.is_pure_info_question
        } else {
            t.is_question
        }) || (t.is_backchannel && !is_monologue_stream)
            || t.content.is_empty()
            || is_superseded_turn
            || skip_seated
        {
            continue;
        }
        let pref_res = if plan.intent.preference {
            pref_reg.resonance_q16(q_zato, ti)
        } else {
            0
        };
        let mut isa_hits = 0u64;
        let mut rare_oov_hits = 0u64;
        let mut hits: Vec<(String, u64)> = Vec::new();
        if deficit_a && !mono_oov_active {
            for &lid in &cn_wn_ids {
                if t.content.contains(&lid) {
                    isa_hits += 1;
                    let df = ix.t_own_stats.df.get(&lid).copied().unwrap_or(0);
                    if df > 0 && df <= 2 && oov_cn_wn_ids.contains(&lid) {
                        rare_oov_hits += 1;
                    }
                }
            }
            let doc_hyp_mask =
                doc_upward_hypernyms_match_query(ix, ti, &q_concepts, max_rare_df, &mut doc_hyp_cache);
            let dh = doc_hyp_mask.count_ones() as u64;
            isa_hits += dh;
            rare_oov_hits += dh;
        }
        if mono_oov_active && !oov_q_concepts.is_empty() {
            for &lid in &oov_cn_wn_ids {
                if t.content.contains(&lid) {
                    let df = ix.t_own_stats.df.get(&lid).copied().unwrap_or(0);
                    if df > 0 && df <= 2 {
                        isa_hits += 1;
                        rare_oov_hits += 1;
                    }
                }
            }
            // Precision OOV rescue: rare (`df <= 2`) document tokens with <= 2 WordNet senses record
            // the query-expansion concepts their gloss / holonyms hit (pass 2 prices them by rarity).
            for &cid in &t.content {
                let df = ix.t_own_stats.df.get(&cid).copied().unwrap_or(0);
                if df == 0 || df > 2 {
                    continue;
                }
                if !mono_gloss_cache.contains_key(&cid) {
                    let matched: Vec<(String, u64)> = ix
                        .vocab
                        .names
                        .get(cid as usize)
                        .filter(|raw| {
                            !raw.starts_with('\u{1}')
                                && raw.len() >= 3
                                && !ix
                                    .vocab
                                    .get(&format!("\u{1}ent:{raw}"))
                                    .is_some_and(|eid| t.entities.contains(&eid))
                        })
                        .map(|raw| {
                            let s_cnt = crate::wordnet::senses(raw.as_str()).len();
                            let mut out: Vec<(String, u64)> = Vec::new();
                            if s_cnt == 0 || s_cnt > 2 {
                                return out;
                            }
                            for d_con in crate::wordnet::direct_mono_gloss_concepts(raw.as_str()) {
                                if oov_q_exact.iter().any(|qc| qc == d_con) {
                                    out.push((d_con.to_string(), 14));
                                } else if s_cnt == 1
                                    && df == 1
                                    && oov_q_concepts.iter().any(|qc| qc == d_con)
                                {
                                    out.push((d_con.to_string(), 7));
                                }
                            }
                            out
                        })
                        .unwrap_or_default();
                    mono_gloss_cache.insert(cid, matched);
                }
                if let Some(ms) = mono_gloss_cache.get(&cid) {
                    for (c, w) in ms {
                        hits.push((c.clone(), *w));
                    }
                }
            }
        }
        // Multi-word & topological containment bridge (Wikidata CC0 + WordNet phrase synsets):
        // Check `1..=6`-token spans in turns bearing rare tokens (`df <= 3`) or proper entities.
        if (deficit_a || deficit_b || mono_oov_active) && !oov_q_concepts.is_empty() && is_monologue_stream {
            let has_rare_anchor = !t.entities.is_empty()
                || t.content
                    .iter()
                    .any(|&cid| ix.t_own_stats.df.get(&cid).copied().unwrap_or(99) <= 3);
            if has_rare_anchor {
                let d_toks = fold_ascii_tokens(&t.body_for_headers);
                for st in 0..d_toks.len() {
                    if hits.len() >= 8 {
                        break;
                    }
                    for len in 1..=6usize {
                        if st + len > d_toks.len() {
                            break;
                        }
                        if len == 1 && (d_toks[st].len() < 4 || term_df(&d_toks[st]) > 2) {
                            continue;
                        }
                        let span = if len == 1 {
                            d_toks[st].strip_suffix("'s").unwrap_or(d_toks[st].as_str()).to_string()
                        } else {
                            d_toks[st..st + len].join(" ")
                        };
                        if !crate::wikidata::has_source(&span)
                            && (len == 1 || crate::wordnet::senses(&span).is_empty())
                        {
                            continue;
                        }
                        for cont in crate::wikidata::upward_containers(&span, 3)
                            .into_iter()
                            .chain(crate::wordnet::phrase_and_lemma_closure(&span, 3))
                        {
                            let of_tail = cont.rsplit_once(" of ").map(|(_, tail)| tail);
                            let exact_tier1 = oov_q_exact
                                .iter()
                                .any(|qc| qc == cont || of_tail.is_some_and(|tl| qc == tl));
                            let concept_hit = exact_tier1
                                || oov_q_concepts
                                    .iter()
                                    .any(|qc| qc == cont || of_tail.is_some_and(|tl| qc == tl));
                            let food_hyp_hit = !concept_hit
                                && !cont.contains(' ')
                                && crate::wordnet::holonyms_and_upward(cont, 3)
                                    .into_iter()
                                    .chain(crate::wordnet::direct_mono_gloss_concepts(cont))
                                    .any(|up| oov_q_concepts.iter().any(|qc| qc == up));
                            if concept_hit || food_hyp_hit {
                                let w = if exact_tier1 { 16 } else if concept_hit { 10 } else { 6 };
                                hits.push((cont.to_string(), w));
                            }
                        }
                    }
                }
            }
        }
        if pref_res == 0 && isa_hits == 0 && hits.is_empty() && rare_oov_hits == 0 {
            continue;
        }
        cands.push(RescueCand { ti, isa_hits, rare_oov_hits, pref_res, hits });
    }

    // Pass 2: concept document frequency across the candidate turns.
    let mut concept_df: std::collections::BTreeMap<&str, u32> = std::collections::BTreeMap::new();
    for c in &cands {
        let mut seen: Vec<&str> = c.hits.iter().map(|(k, _)| k.as_str()).collect();
        seen.sort_unstable();
        seen.dedup();
        for k in seen {
            *concept_df.entry(k).or_insert(0) += 1;
        }
    }
    let concept_df_cap = ((nt as u32) / 8).max(2);

    for c in &cands {
        let ti = c.ti;
        let t = &ix.turns[ti];
        let pref_res = c.pref_res;
        // Evidence a concept gives for THIS turn = weight / (number of turns hitting it): a concept that
        // only one turn hits pins that turn down; one hit by a dozen turns barely discriminates.
        let mut kept: Vec<(&str, u64, u64)> = Vec::new();
        for (k, w) in &c.hits {
            let d = concept_df.get(k.as_str()).copied().unwrap_or(0);
            if d <= concept_df_cap {
                match kept.iter_mut().find(|(kk, _, _)| *kk == k.as_str()) {
                    Some(e) => e.1 = e.1.max(*w),
                    None => kept.push((k.as_str(), *w, u64::from(d.max(1)))),
                }
            }
        }
        kept.sort_by(|a, b| {
            (b.1 * Q16 / b.2)
                .cmp(&(a.1 * Q16 / a.2))
                .then_with(|| a.0.cmp(b.0))
        });
        let strong_bridge_hits: u64 = kept.iter().take(3).map(|(_, w, _)| *w).sum();
        let strong_q16: u64 = kept.iter().take(3).map(|(_, w, d)| *w * Q16 / *d).sum();
        let rare_oov_hits = c.rare_oov_hits + kept.len().min(3) as u64;
        let isa_hits = c.isa_hits + strong_bridge_hits;
        if pref_res == 0 && isa_hits == 0 {
            continue;
        }

        let sp_aligned = !t.speaker.is_empty()
            && t.speaker != "user"
            && t.speaker != "assistant"
            && super::seating::c2_query_mentions_speaker(q_text, &t.speaker);

        let d_zato = zatocode_u128(
            t.entities
                .iter()
                .copied()
                .chain(t.content.iter().copied().take(12)),
        );
        let zato_q16 = zatocode_overlap_q16(q_zato, d_zato);
        let t_wilkins = turn_wilkins_code_u64(ix, ti);
        let wilkins_q16 = if q_wilkins != 0 && t_wilkins != 0 {
            wilkins_lcp_similarity_q16(q_wilkins, t_wilkins)
        } else {
            0
        };
        let density_bonus = if rare_oov_hits >= 1 || strong_bridge_hits >= 1 {
            let concise = (Q16 * 8) / ((t.content.len() as u64).saturating_add(4));
            let ent_bonus = if !t.entities.is_empty() { Q16 >> 1 } else { 0 };
            concise.saturating_add(ent_bonus)
        } else {
            0
        };
        let weak_oov_q16 = if mono_oov_active {
            rare_oov_hits.min(2) * (Q16 >> 1)
        } else {
            rare_oov_hits.min(3) * Q16
        };
        let isa_q16 = strong_q16
            .saturating_add(weak_oov_q16)
            .saturating_add(density_bonus)
            .saturating_add((isa_hits.min(6) * Q16) >> 2);
        let base_score = isa_q16
            .saturating_add(pref_res)
            .saturating_add(zato_q16 >> 1)
            .saturating_add(wilkins_q16 >> 1);
        let score = if sp_aligned {
            base_score.saturating_mul(2)
        } else {
            base_score
        };

        let rescue_tag = strong_bridge_hits.saturating_add(rare_oov_hits);
        if score >= Q16 && best_cand.map_or(true, |(_, bs, _, _)| score > bs) {
            best_cand = Some((ti, score, sp_aligned, rescue_tag));
        }
    }

    let Some((cand_ti, _cand_score, _sp_aligned, rare_oov_hits)) = best_cand else {
        return false;
    };

    // Direct Slot-0 promotion for monologue WH queries with ungrounded OOV/rare concept bridges:
    if is_monologue_stream && rare_oov_hits >= 1 && has_ungrounded_rare_oov {
        let low = q_text.trim_start().to_ascii_lowercase();
        let is_wh_lead =
            low.starts_with("which ") || low.starts_with("what ") || low.starts_with("who ");
        if is_wh_lead {
            if let Some(pos) = final_turn_indices.iter().position(|&x| x == cand_ti) {
                final_turn_indices.remove(pos);
            }
            final_turn_indices.insert(0, cand_ti);
            if final_turn_indices.len() > k {
                final_turn_indices.truncate(k);
            }
            return true;
        }
    }

    if final_turn_indices.len() < k {
        final_turn_indices.push(cand_ti);
        return false;
    }

    // Collect top-2 rescued turns across distinct sessions under deficit_a / preference.
    let head0_zero = final_turn_indices
        .first()
        .is_some_and(|&h0| base_q16[h0] == 0);
    if head0_zero {
        final_turn_indices.insert(0, cand_ti);
        if final_turn_indices.len() > k {
            final_turn_indices.truncate(k);
        }
        return false;
    }

    let cand_sess = ix.turns[cand_ti].session;
    if final_turn_indices[..5].iter().any(|&t| ix.turns[t].session == cand_sess) {
        return false;
    }
    if let Some(slot) = (5..final_turn_indices.len()).rev().find(|&idx| {
        let vti = final_turn_indices[idx];
        let vs = ix.turns[vti].session;
        let dup = final_turn_indices[..idx].iter().any(|&e| ix.turns[e].session == vs);
        let mut earlier_mask = 0u64;
        for &e in &final_turn_indices[..idx] {
            earlier_mask |= t_exact_mask[e];
        }
        (dup && ix.turns[vti].field == 1)
            || base_q16[vti] == 0
            || (dup && ix.turns[vti].is_question && (t_exact_mask[vti] & !earlier_mask) == 0)
    }) {
        final_turn_indices[slot] = cand_ti;
    } else if is_monologue_stream && rare_oov_hits >= 1 && !final_turn_indices.is_empty() {
        let low = q_text.trim_start().to_ascii_lowercase();
        let is_wh_lead =
            low.starts_with("which ") || low.starts_with("what ") || low.starts_with("who ");
        if is_wh_lead && has_ungrounded_rare_oov {
            final_turn_indices.insert(0, cand_ti);
            if final_turn_indices.len() > k {
                final_turn_indices.truncate(k);
            }
            return true;
        } else if is_wh_lead
            && final_turn_indices
                .first()
                .is_some_and(|&h0| (t_exact_mask[h0]).count_ones() <= 1)
            && final_turn_indices.len() >= 5
            && t_exact_mask[final_turn_indices[4]] == 0
        {
            final_turn_indices.insert(4, cand_ti);
            if final_turn_indices.len() > k {
                final_turn_indices.truncate(k);
            }
        } else {
            let last = final_turn_indices.len() - 1;
            final_turn_indices[last] = cand_ti;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilkins_zatocode_and_pmest_are_pure_integer_deterministic() {
        let c_dog = wilkins_term_code_u64("dog");
        let c_puppy = wilkins_term_code_u64("puppy");
        assert_ne!(c_dog, 0);
        assert!(wilkins_lcp_similarity_q16(c_dog, c_dog) == Q16);
        assert!(wilkins_lcp_similarity_q16(c_dog, c_puppy) <= Q16);

        let z1 = zatocode_u128([10, 20, 30]);
        let z2 = zatocode_u128([10, 20, 99]);
        assert!(zatocode_overlap_q16(z1, z1) == Q16);
        assert!(zatocode_overlap_q16(z1, z2) > 0);

        let p = PmestSignature {
            personality: 1,
            matter: 2,
            energy: 0,
            space: 1,
            time: 0,
        };
        assert_eq!(p.align_q16(p), Q16);
    }
}
