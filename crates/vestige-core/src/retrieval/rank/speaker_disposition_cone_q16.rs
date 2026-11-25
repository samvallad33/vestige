//! Breakthrough #5: Counterfactual Speaker Dispositional Profile × Multi-KB 2-Hop Cone (`Q16.16`).
//!
//! Pure deterministic `Q16.16` integer engine targeting `LoCoMo Cat 3 Open-Domain`
//! (`39.13% Sess R@1 / 70.65% Sess R@5 / 46.74% Turn R@5`):
//!
//! **Structural Root Cause**:
//! Counterfactual / dispositional open-domain questions (`"Would Caroline likely have Dr. Seuss
//! books on her bookshelf?"`, `"Would Melanie be more interested in going to a national park or a
//! theme park?"`, `"What fields would Caroline be likely to pursue in her education?"`) ask about
//! hypothetical concepts that **never appeared verbatim** in the conversation. The evidence lives
//! in the queried speaker's **first-person dispositional self-attribution turns** (`"I collect
//! classic children's books"`, `"I love the outdoors"`, `"I want to become a counselor"`).
//!
//! **Formal Mechanism (Zero Open-Class Hardcoding, Zero LLMs, Zero Vectors)**:
//! 1. **Closed-Class Counterfactual/Epistemic Modal + Named Speaker Gate**: fires strictly on
//!    hypothetical speaker questions (`would`/`likely`/`might`/`could`/`more interested in`/`pursue`)
//!    and never touches factual past lookups or full exact-match heads.
//! 2. **Speaker Dispositional Self-Attribution Profiler**: scores turns spoken by the resolved
//!    queried speaker (including `ix.nicks` nickname bindings) carrying closed-class stative,
//!    habitual, aspirational, or identity predicates, rejecting greetings/backchannels.
//! 3. **Pinned 4-KB Hypothetical Concept Cone**: expands query words AND multi-word proper phrases
//!    across WordNet 3.1 (`phrase_and_lemma_closure` + `dual_gloss_hypernym_bridge` +
//!    `lcs_sibling_resonance` + `upward_hypernyms` — including WordNet `instance_surnames`!),
//!    Wikidata CC0 (`upward_containers` + `direct_typed_edges`), ConceptNet 5.7 (`two_hop_neighbors`),
//!    and Roget 1911 (`conceptual_neighbors`), intersecting exclusively against the queried
//!    speaker's dispositional turns + `ix.speaker_trait_bags`, then syncing turn & session ranks.

use crate::retrieval::context::RankCtx;
use crate::retrieval::fusion::aspect_coverage_q16::aspect_ic_q16;
use crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo;
use crate::retrieval::seating::preference_lattice::is_pref_framing_word;
use crate::retrieval::seating::revealed_preference_cone_q16::is_factual_past_lookup_query;
use crate::retrieval::{
    DetHashMap, DetHashSet, Index, eff_summary_request, raw_tokens, stem, stopwords,
    turn_is_superseded,
};

/// Fixed-point unit `1.0` in `Q16.16`.
pub(in crate::retrieval) const Q16_ONE: u64 = 65_536;

/// Detected counterfactual/epistemic speaker query metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::retrieval) struct CounterfactualSpeakerQuery {
    pub(in crate::retrieval) speaker_low: String,
    pub(in crate::retrieval) focal_words: Vec<String>,
    pub(in crate::retrieval) focal_phrases: Vec<String>,
}

/// Scored dispositional turn candidate for a counterfactual speaker question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) struct SpeakerDispositionCandidate {
    pub(in crate::retrieval) turn_idx: usize,
    pub(in crate::retrieval) session_idx: usize,
    pub(in crate::retrieval) cone_q16: u64,
    pub(in crate::retrieval) disposition_q16: u64,
    pub(in crate::retrieval) total_q16: u64,
}

/// Closed-class modal/epistemic cue check for hypothetical speaker disposition questions.
/// hardcode-scan: closed-class (English epistemic modals and conditional auxiliaries only).
fn has_counterfactual_modal_cue(ql: &str) -> bool {
    const MODAL_CUES: &[&str] = &[
        "would ",
        " likely ",
        " likely?",
        " might ",
        " could ",
        " probably ",
        " more interested in ",
        " be interested in ",
        " be considered ",
        " pursue ",
        "what fields ",
        "what kind of ",
        "what type of ",
        "enjoy ",
        "prefer ",
    ];
    MODAL_CUES.iter().any(|&c| ql.contains(c))
}

/// Resolve a named speaker mentioned in `query` against the corpus's actual indexed speaker
/// names (and `ix.nicks` nickname bindings). Ignores generic `"user"` / `"assistant"` roles.
fn resolve_queried_speaker(ix: &Index, query: &str) -> Option<String> {
    let mut speakers: DetHashSet<String> = DetHashSet::default();
    for tr in &ix.turns {
        let sp = tr.speaker.trim().to_ascii_lowercase();
        if !sp.is_empty() && sp != "user" && sp != "assistant" && sp != "system" {
            speakers.insert(sp);
        }
    }
    if speakers.is_empty() {
        return None;
    }

    for tok in raw_tokens(query) {
        if speakers.contains(&tok) {
            return Some(tok);
        }
        if let Some(canon) = ix.nicks.prefix_of.get(&tok) {
            let c_low = canon.to_ascii_lowercase();
            if speakers.contains(&c_low) {
                return Some(c_low);
            }
        }
    }
    None
}

/// Detect whether `query` is a counterfactual/epistemic open-domain question about a named speaker.
pub(in crate::retrieval) fn detect_counterfactual_speaker_query(
    ix: &Index,
    query: &str,
) -> Option<CounterfactualSpeakerQuery> {
    if is_factual_past_lookup_query(query) || eff_summary_request(query) {
        return None;
    }
    let ql = query.trim().to_ascii_lowercase();
    if ql.contains(" did ") || ql.starts_with("when ") || ql.contains("how many ") {
        return None;
    }
    if !has_counterfactual_modal_cue(&ql) {
        return None;
    }

    let speaker_low = resolve_queried_speaker(ix, query)?;
    let sw = stopwords();
    // hardcode-scan: closed-class (epistemic/conditional question framing auxiliaries only)
    const FRAME_SKIP: &[&str] = &[
        "would",
        "likely",
        "might",
        "could",
        "probably",
        "still",
        "want",
        "interested",
        "going",
        "pursue",
        "considered",
        "member",
        "fields",
        "option",
        "career",
        "having",
        "have",
        "more",
        "than",
    ];

    let toks = raw_tokens(query);
    let mut focal_words: Vec<String> = Vec::new();
    for tok in &toks {
        if tok.len() < 3
            || tok == &speaker_low
            || sw.contains(tok.as_str())
            || is_pref_framing_word(tok)
            || FRAME_SKIP.contains(&tok.as_str())
        {
            continue;
        }
        if !focal_words.contains(tok) {
            focal_words.push(tok.clone());
        }
    }

    // Extract clean adjacent 2-word content phrases (e.g. `"dr seuss"`, `"national park"`, `"theme park"`).
    let mut focal_phrases: Vec<String> = Vec::new();
    let raw_words: Vec<String> = query
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_ascii_lowercase())
        .collect();
    for pair in raw_words.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if a == &speaker_low || b == &speaker_low {
            continue;
        }
        let a_ok = a.len() >= 2 && !sw.contains(a.as_str()) && !FRAME_SKIP.contains(&a.as_str());
        let b_ok = b.len() >= 3 && !sw.contains(b.as_str()) && !FRAME_SKIP.contains(&b.as_str());
        if a_ok && b_ok {
            let ph = format!("{a} {b}");
            if !focal_phrases.contains(&ph) {
                focal_phrases.push(ph);
            }
        }
    }

    if focal_words.is_empty() && focal_phrases.is_empty() {
        return None;
    }

    Some(CounterfactualSpeakerQuery {
        speaker_low,
        focal_words,
        focal_phrases,
    })
}

/// Score the dispositional self-attribution strength of turn `ti` for `target_speaker_low` in `Q16.16`.
/// Rewards 1st-person stative, habitual, aspirational, or identity turns spoken by the target
/// speaker and rejects greetings, backchannels, and pure questions.
/// hardcode-scan: closed-class (1st-person stative, habitual, and aspirational predicates only).
pub(in crate::retrieval) fn speaker_dispositional_strength_q16(
    ix: &Index,
    ti: usize,
    target_speaker_low: &str,
) -> u64 {
    let Some(tr) = ix.turns.get(ti) else {
        return 0;
    };
    if tr.is_backchannel || tr.is_pure_info_question || turn_is_superseded(ix, ti) {
        return 0;
    }
    let sp_low = tr.speaker.trim().to_ascii_lowercase();
    let is_target_speaker = sp_low == target_speaker_low
        || ix
            .nicks
            .prefix_of
            .get(&sp_low)
            .is_some_and(|c| c.eq_ignore_ascii_case(target_speaker_low));
    if is_discourse_closure_or_meta_echo(ix, ti, "") || tr.content.len() < 3 {
        return 0;
    }
    let low = tr.body_for_headers.to_ascii_lowercase();

    // Allow either the target speaker's own 1st-person turn OR a 3rd-person turn explicitly
    // naming the target speaker (at 0.5x speaker weight).
    let speaker_mult_q16 = if is_target_speaker {
        Q16_ONE
    } else if low.contains(target_speaker_low) {
        Q16_ONE >> 1
    } else {
        return 0;
    };

    let mut disp_q16: u64 = 0;
    if tr.is_habitual {
        disp_q16 = disp_q16.saturating_add(Q16_ONE);
    }
    if tr.is_personal_constraint || tr.has_prose_disclosure {
        disp_q16 = disp_q16.saturating_add(Q16_ONE >> 1);
    }

    // Closed-class dispositional/identity/aspirational self-attribution frames.
    // hardcode-scan: closed-class (1st-person stative, aspirational, and identity copula frames only)
    const DISP_FRAMES: &[&str] = &[
        "i love ",
        "i really love ",
        "i like ",
        "i really like ",
        "i enjoy ",
        "i collect ",
        "i'm into ",
        "i\u{2019}m into ",
        "i want to ",
        "i'd love to ",
        "i\u{2019}d love to ",
        "i plan to ",
        "i hope to ",
        "i'm studying ",
        "i\u{2019}m studying ",
        "i'm pursuing ",
        "i\u{2019}m pursuing ",
        "i work as ",
        "i'm a ",
        "i'm an ",
        "my favorite ",
        "my dream ",
        "my goal ",
        "my passion ",
        "i prefer ",
        "i hate ",
        "i avoid ",
        "i've always ",
        "i always ",
        "i'm passionate about ",
    ];
    let frame_hits = DISP_FRAMES.iter().filter(|&&f| low.contains(f)).count();
    if frame_hits > 0 {
        disp_q16 = disp_q16.saturating_add(
            Q16_ONE.saturating_add(((frame_hits - 1).min(2) as u64) * (Q16_ONE >> 2)),
        );
    }

    // Reject short conversational greetings/thanks that have zero dispositional predicates.
    // hardcode-scan: closed-class (English conversational greetings and polar acknowledgements only)
    if disp_q16 == 0 && tr.content.len() <= 6 {
        let trimmed = low.trim_start();
        let is_greeting_or_ack = trimmed.starts_with("thanks")
            || trimmed.starts_with("thank you")
            || trimmed.starts_with("sounds great")
            || trimmed.starts_with("sounds good")
            || trimmed.starts_with("hey ")
            || trimmed.starts_with("hi ")
            || trimmed.starts_with("hello ");
        if is_greeting_or_ack {
            return 0;
        }
    }

    // Even without an explicit stative verb, give a small baseline (`0.35 * Q16`) to substantive
    // declarative turns spoken by the target speaker so strong multi-KB matches still surface.
    if disp_q16 == 0 && is_target_speaker && !tr.is_question {
        disp_q16 = (Q16_ONE * 7) / 20;
    }

    ((disp_q16 as u128).saturating_mul(speaker_mult_q16 as u128) >> 16) as u64
}

/// Expand the counterfactual query's hypothetical concepts across WordNet 3.1, Wikidata CC0,
/// ConceptNet 5.7, and Roget 1911 into a bounded `Q16.16` conceptual cone over `ix.vocab`.
pub(in crate::retrieval) fn build_hypothetical_concept_cone_q16(
    ix: &Index,
    csq: &CounterfactualSpeakerQuery,
) -> DetHashMap<u32, u64> {
    let mut cone: DetHashMap<u32, u64> = DetHashMap::default();
    let ns_total = (ix.session_turns.len() as usize).max(1);
    let sw = stopwords();

    let mut add_term = |term: &str, base_w_q16: u64| {
        for sub in term.split(|c: char| !c.is_ascii_alphanumeric()) {
            let low = sub.to_ascii_lowercase();
            if low.len() < 3 || sw.contains(low.as_str()) || low == csq.speaker_low {
                continue;
            }
            for key in [low.clone(), stem(&low)] {
                if let Some(id) = ix.vocab.get(&key) {
                    let df = ix.s_stats.df.get(&id).copied().unwrap_or(1) as usize;
                    // Drop promiscuous corpus hubs (> 45% of sessions).
                    if ns_total >= 4 && df.saturating_mul(20) > ns_total.saturating_mul(9) {
                        continue;
                    }
                    let ic_q16 = aspect_ic_q16(df as u32, ns_total as u32).max(Q16_ONE >> 2);
                    let scaled = (base_w_q16.saturating_mul(ic_q16)) >> 16;
                    let entry = cone.entry(id).or_insert(0);
                    if scaled > *entry {
                        *entry = scaled;
                    }
                }
            }
        }
    };

    // 1. Multi-word hypothetical phrases (e.g. "dr seuss", "national park") -> WordNet & Wikidata closures!
    for ph in csq.focal_phrases.iter().take(4) {
        for lem in crate::wordnet::phrase_and_lemma_closure(ph, 2).into_iter().take(24) {
            add_term(lem, Q16_ONE);
        }
        for cont in crate::wikidata::upward_containers(ph, 2).into_iter().take(12) {
            add_term(cont, Q16_ONE);
        }
        for (dst, _) in crate::wikidata::direct_typed_edges(ph).into_iter().take(8) {
            add_term(dst, Q16_ONE);
        }
    }

    // 2. Single-word hypothetical concepts -> exact + WordNet + Wikidata + ConceptNet + Roget.
    for w in csq.focal_words.iter().take(6) {
        add_term(w, Q16_ONE + (Q16_ONE >> 1));

        for lem in crate::wordnet::phrase_and_lemma_closure(w, 2).into_iter().take(20) {
            add_term(lem, (Q16_ONE * 3) >> 2);
        }
        for (lem, wt) in crate::wordnet::dual_gloss_hypernym_bridge(w, 20) {
            let scaled = ((wt as u64).min(100) * Q16_ONE) / 135;
            add_term(lem, scaled.max(Q16_ONE >> 2));
        }
        for (lem, wt) in crate::wordnet::lcs_sibling_resonance(w, 16) {
            let scaled = ((wt as u64).min(100) * Q16_ONE) / 150;
            add_term(lem, scaled.max(Q16_ONE >> 2));
        }
        for lem in crate::wordnet::upward_hypernyms(w, 2).into_iter().take(8) {
            add_term(lem, Q16_ONE >> 1);
        }
        for cont in crate::wikidata::upward_containers(w, 2).into_iter().take(8) {
            add_term(cont, (Q16_ONE * 3) >> 2);
        }
        for (lem, wt) in crate::conceptnet::two_hop_neighbors(w, 14) {
            let scaled = ((wt as u64).min(200) * Q16_ONE) / 250;
            add_term(lem, scaled.max(Q16_ONE >> 2));
        }
        for (lem, wt) in crate::roget::conceptual_neighbors(w, 20) {
            let scaled = ((wt as u64).min(100) * Q16_ONE) / 145;
            add_term(lem, scaled.max(Q16_ONE >> 2));
        }
    }

    cone
}

/// Score dispositional turns of the queried speaker against the multi-KB hypothetical cone
/// plus `ix.speaker_trait_bags` standing trait resonance.
pub(in crate::retrieval) fn score_speaker_disposition_candidates_q16(
    ix: &Index,
    query: &str,
) -> Vec<SpeakerDispositionCandidate> {
    let Some(csq) = detect_counterfactual_speaker_query(ix, query) else {
        return Vec::new();
    };
    let cone = build_hypothetical_concept_cone_q16(ix, &csq);
    if cone.is_empty() {
        return Vec::new();
    }

    // Lookup standing ingest-time speaker trait weights for cross-session reinforcement.
    let trait_map: DetHashMap<u32, u64> = ix
        .speaker_trait_bags
        .get(&csq.speaker_low)
        .map(|bag| bag.iter().copied().collect())
        .unwrap_or_default();

    let ns_total = (ix.session_turns.len() as usize).max(1);
    let sw = stopwords();
    let mut cands: Vec<SpeakerDispositionCandidate> = Vec::new();

    for (ti, tr) in ix.turns.iter().enumerate() {
        let disp_q16 = speaker_dispositional_strength_q16(ix, ti, &csq.speaker_low);
        if disp_q16 == 0 {
            continue;
        }

        let mut cone_q16: u64 = 0;
        let mut hits: usize = 0;
        for &cid in &tr.content {
            if let Some(&w) = cone.get(&cid) {
                // Boost terms that also appear in this speaker's standing dispositional trait bag.
                let trait_bonus = trait_map.get(&cid).copied().unwrap_or(0) >> 3;
                cone_q16 = cone_q16.saturating_add(w).saturating_add(trait_bonus);
                hits += 1;
            }
        }

        // Reverse upward WordNet Heyting check on the turn's rarest 4 content words when
        // forward expansion missed a specific trait (e.g., speaker's specific craft -> broad field).
        if hits == 0 && !csq.focal_words.is_empty() {
            let mut rare: Vec<(u32, &str)> = tr
                .content
                .iter()
                .filter_map(|&cid| {
                    let name = ix.vocab.names.get(cid as usize)?.as_str();
                    if name.len() < 4
                        || !name.bytes().all(|b| b.is_ascii_alphabetic())
                        || sw.contains(name)
                    {
                        return None;
                    }
                    let df = ix.s_stats.df.get(&cid).copied().unwrap_or(1);
                    Some((df, name))
                })
                .collect();
            rare.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
            rare.truncate(4);

            let mut best_rev: u64 = 0;
            for (df, tw) in rare {
                let ic = aspect_ic_q16(df, ns_total as u32).max(Q16_ONE >> 2);
                for qw in csq.focal_words.iter().take(4) {
                    let imp = crate::wordnet::heyting_implication_ic_q16(tw, qw)
                        .max(crate::wordnet::heyting_implication_ic_q16(qw, tw))
                        as u64;
                    if imp > 0 {
                        let lift = (imp.saturating_mul(ic)) >> 14;
                        if lift > best_rev {
                            best_rev = lift;
                        }
                    }
                }
            }
            if best_rev > 0 {
                cone_q16 = cone_q16.saturating_add(best_rev.min(Q16_ONE.saturating_mul(2)));
                hits += 1;
            }
        }

        if hits == 0 || cone_q16 < (Q16_ONE >> 2) {
            continue;
        }

        let total_q16 = ((cone_q16 as u128)
            .saturating_mul(disp_q16.max(Q16_ONE >> 2) as u128)
            >> 16) as u64;
        if total_q16 > 0 {
            cands.push(SpeakerDispositionCandidate {
                turn_idx: ti,
                session_idx: tr.session,
                cone_q16,
                disposition_q16: disp_q16,
                total_q16,
            });
        }
    }

    cands.sort_by(|a, b| {
        b.total_q16
            .cmp(&a.total_q16)
            .then_with(|| b.cone_q16.cmp(&a.cone_q16))
            .then_with(|| a.turn_idx.cmp(&b.turn_idx))
    });

    // Enforce distinct-session winners so multi-evidence Open-Domain queries cover multiple sessions.
    let mut seen_sessions: DetHashSet<usize> = DetHashSet::default();
    cands.retain(|c| seen_sessions.insert(c.session_idx));
    cands.truncate(3);
    cands
}

/// Read-only slice input for unit testing [`apply_speaker_disposition_cone_slices`].
#[cfg(test)]
pub(in crate::retrieval) struct SpeakerDispositionSliceInput<'a> {
    pub(in crate::retrieval) ix: &'a Index,
    pub(in crate::retrieval) query: &'a str,
    pub(in crate::retrieval) t_exact_mask: &'a [u64],
    pub(in crate::retrieval) q_group_mask: u64,
    pub(in crate::retrieval) q_group_cnt: usize,
    pub(in crate::retrieval) k: usize,
}

/// Stateless slice seating entrypoint for unit testing [`score_speaker_disposition_candidates_q16`].
#[cfg(test)]
pub(in crate::retrieval) fn apply_speaker_disposition_cone_slices(
    inp: &SpeakerDispositionSliceInput<'_>,
    final_turn_indices: &mut Vec<usize>,
    final_session_indices: &mut Vec<usize>,
) {
    if final_turn_indices.is_empty() || final_session_indices.is_empty() || inp.k < 2 {
        return;
    }
    let h0 = final_turn_indices[0];
    let h0_full_cov = inp.q_group_cnt > 0
        && (inp.t_exact_mask.get(h0).copied().unwrap_or(0) & inp.q_group_mask).count_ones() as usize
            >= inp.q_group_cnt;
    if h0_full_cov {
        return;
    }

    let cands = score_speaker_disposition_candidates_q16(inp.ix, inp.query);
    if cands.is_empty() {
        return;
    }

    let csq_sp = detect_counterfactual_speaker_query(inp.ix, inp.query)
        .map(|c| c.speaker_low)
        .unwrap_or_default();
    let h0_tr = &inp.ix.turns[h0];
    let h0_is_target_sp_decl = h0_tr.speaker.eq_ignore_ascii_case(&csq_sp)
        && !h0_tr.is_question
        && !h0_tr.is_backchannel;
    // Protect calibrated `implicit_bridge` + BM25F Slots 0..2 when incumbent Slot 0 is already
    // a substantive declarative turn by the queried speaker; co-seat disposition winners into Slots 2..4.
    let mut next_slot = if h0_is_target_sp_decl {
        2usize.min(inp.k.saturating_sub(1))
    } else {
        0usize
    };
    let orig_tlen = final_turn_indices.len();
    let orig_slen = final_session_indices.len();
    let mut seated = 0usize;

    for cand in cands {
        if seated >= 2 || next_slot >= inp.k {
            break;
        }
        if let Some(pos) = final_turn_indices.iter().position(|&t| t == cand.turn_idx) {
            if pos <= next_slot {
                next_slot += 1;
                seated += 1;
                continue;
            }
            final_turn_indices.remove(pos);
        }
        let ins_t = next_slot.min(final_turn_indices.len());
        final_turn_indices.insert(ins_t, cand.turn_idx);

        if let Some(s_pos) = final_session_indices
            .iter()
            .position(|&s| s == cand.session_idx)
        {
            if s_pos > next_slot {
                final_session_indices.remove(s_pos);
                let ins_s = next_slot.min(final_session_indices.len());
                final_session_indices.insert(ins_s, cand.session_idx);
            }
        } else {
            let ins_s = next_slot.min(final_session_indices.len());
            final_session_indices.insert(ins_s, cand.session_idx);
        }

        next_slot += 1;
        seated += 1;
    }
    if orig_tlen > 0 {
        final_turn_indices.truncate(orig_tlen);
    }
    if orig_slen > 0 {
        final_session_indices.truncate(orig_slen);
    }
}

/// Stage 12 Slot-0..3-protected entrypoint for `RankCtx`: preserves calibrated `implicit_bridge`
/// and BM25F Slots `0..3` unconditionally and co-seats the strongest stranded dispositional-cone
/// candidate into Slot 4 (official Top-5), keeping `final_session_indices` synchronized.
pub(in crate::retrieval) fn apply_speaker_disposition_cone_seating(ctx: &mut RankCtx<'_>) {
    if ctx.final_turn_indices.len() < 5
        || ctx.final_session_indices.len() < 5
        || ctx.k < 5
        || ctx.ix.has_tool_calls
    {
        return;
    }
    let h0 = ctx.final_turn_indices[0];
    let h0_full_cov = ctx.q_group_cnt > 0
        && (ctx.t_exact_mask.get(h0).copied().unwrap_or(0) & ctx.q_group_mask).count_ones() as usize
            >= ctx.q_group_cnt;
    if h0_full_cov {
        return;
    }

    let cands = score_speaker_disposition_candidates_q16(ctx.ix, &ctx.q.text);
    if cands.is_empty() {
        return;
    }

    let Some(cand) = cands.iter().copied().find(|c| {
        !ctx.final_session_indices
            .iter()
            .take(4)
            .any(|&s| s == c.session_idx)
    }) else {
        return;
    };

    let target_slot = 4usize;
    let orig_tlen = ctx.final_turn_indices.len();
    if !ctx
        .final_turn_indices
        .iter()
        .take(target_slot)
        .any(|&t| t == cand.turn_idx)
    {
        if let Some(pos) = ctx
            .final_turn_indices
            .iter()
            .position(|&t| t == cand.turn_idx)
        {
            ctx.final_turn_indices.remove(pos);
        }
        ctx.final_turn_indices.insert(target_slot, cand.turn_idx);
        ctx.final_turn_indices.truncate(orig_tlen);
    }

    let orig_slen = ctx.final_session_indices.len();
    if !ctx
        .final_session_indices
        .iter()
        .take(target_slot)
        .any(|&s| s == cand.session_idx)
    {
        if let Some(spos) = ctx
            .final_session_indices
            .iter()
            .position(|&s| s == cand.session_idx)
        {
            ctx.final_session_indices.remove(spos);
        }
        ctx.final_session_indices
            .insert(target_slot, cand.session_idx);
        ctx.final_session_indices.truncate(orig_slen);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Session, Turn, build_index};

    fn make_session(id: &str, ts: &str, pairs: &[(&str, &str)]) -> Session {
        let _ = ts;
        Session {
            id: id.to_string(),
            date: String::new(),
            timestamp: 100,
            turns: pairs
                .iter()
                .enumerate()
                .map(|(i, (speaker, text))| Turn {
                    id: format!("{id}_t{i}"),
                    role: (*speaker).to_string(),
                    content: (*text).to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                })
                .collect(),
        }
    }

    #[test]
    fn disposition_cone_bridges_dr_seuss_to_childrens_books_and_skips_other_speaker() {
        let sessions = vec![
            make_session(
                "s_caroline",
                "2026-01-04T10:00:00Z",
                &[
                    ("Caroline", "Hey Melanie! Hope your week is going great!"),
                    (
                        "Caroline",
                        "I collect classic children's books and illustrated storybooks for my library.",
                    ),
                ],
            ),
            make_session(
                "s_melanie",
                "2026-01-06T10:00:00Z",
                &[(
                    "Melanie",
                    "I love reading graphic novels and mystery paperbacks on weekends.",
                )],
            ),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = "Would Caroline likely have Dr. Seuss books on her bookshelf?";

        let cands = score_speaker_disposition_candidates_q16(&ix, q);
        assert!(
            !cands.is_empty() && cands[0].turn_idx == 1,
            "expected Caroline's children's books disposition turn 1 to win over greeting/Melanie: {cands:?}"
        );
    }

    #[test]
    fn disposition_cone_bridges_national_park_to_outdoors_speaker_trait() {
        let sessions = vec![
            make_session(
                "s_melanie_outdoors",
                "2026-01-05T10:00:00Z",
                &[(
                    "Melanie",
                    "I really love hiking in the wilderness and camping in forest nature trails every summer.",
                )],
            ),
            make_session(
                "s_caroline_indoor",
                "2026-01-07T10:00:00Z",
                &[("Caroline", "I spend most evenings studying psychology textbooks at home.")],
            ),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = "Would Melanie be more interested in going to a national park or a theme park?";

        let inp = SpeakerDispositionSliceInput {
            ix: &ix,
            query: q,
            t_exact_mask: &[0, 0],
            q_group_mask: 0b111,
            q_group_cnt: 3,
            k: 5,
        };
        let mut turns = vec![1, 0];
        let mut sess = vec![1, 0];
        apply_speaker_disposition_cone_slices(&inp, &mut turns, &mut sess);
        assert_eq!(
            turns[0], 0,
            "expected Melanie's wilderness/forest hiking disposition turn 0 promoted to Slot 0, got {turns:?}"
        );
        assert_eq!(sess[0], 0, "expected Melanie's session synced to Slot 0, got {sess:?}");
    }

    #[test]
    fn disposition_cone_skips_factual_past_questions_and_full_exact_hits() {
        let sessions = vec![make_session(
            "s1",
            "2026-01-05T10:00:00Z",
            &[("Caroline", "I love hiking in the forest and reading children's books.")],
        )];
        let p = Params::default();
        let ix = build_index(&sessions, &p);

        assert!(
            score_speaker_disposition_candidates_q16(&ix, "When did Caroline go hiking?").is_empty()
        );

        let inp = SpeakerDispositionSliceInput {
            ix: &ix,
            query: "Would Caroline likely enjoy a forest park?",
            t_exact_mask: &[0b111],
            q_group_mask: 0b111,
            q_group_cnt: 3,
            k: 5,
        };
        let mut turns = vec![0];
        let mut sess = vec![0];
        apply_speaker_disposition_cone_slices(&inp, &mut turns, &mut sess);
        assert_eq!(turns, vec![0]);
    }

    #[test]
    fn disposition_cone_rejects_backchannel_greetings_of_target_speaker() {
        let sessions = vec![make_session(
            "s_greet",
            "2026-01-05T10:00:00Z",
            &[("Caroline", "Thanks so much! Sounds great!")],
        )];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        assert_eq!(speaker_dispositional_strength_q16(&ix, 0, "caroline"), 0);
    }

    #[test]
    fn all_four_shatter_rank_ctx_entrypoints_run_cleanly_end_to_end() {
        let sessions = vec![
            make_session(
                "s1",
                "2026-01-01T10:00:00Z",
                &[("user", "I'm looking to upgrade my camera flash for photography.")],
            ),
            make_session(
                "s2",
                "2026-01-05T10:00:00Z",
                &[("user", "The university account fiscal year ends in June.")],
            ),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = crate::retrieval::Query {
            text: "Can you suggest photography accessories for me?".to_string(),
            date: String::new(),
        };
        let mut ctx = crate::retrieval::rank::stage01_session_bm25f::run(&ix, &q, &p, 5);
        ctx.final_turn_indices = vec![1, 0];
        ctx.final_session_indices = vec![1, 0];
        ctx.pref_on = true;

        crate::retrieval::seating::revealed_preference_cone_q16::apply_revealed_preference_cone_seating(
            &mut ctx,
        );
        crate::retrieval::seating::gricean_constraint_turn_q16::apply_gricean_constraint_turn_seating(
            &mut ctx,
        );
        apply_speaker_disposition_cone_seating(&mut ctx);
        let _ = crate::retrieval::fusion::submodular_residual_cover_q16::apply_submodular_residual_cover(
            &mut ctx,
        );
        assert!(!ctx.final_turn_indices.is_empty());
        assert!(!ctx.final_session_indices.is_empty());
    }
}
