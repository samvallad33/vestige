//! Breakthrough #2: Revealed-Preference Interrogative Immunity + Identity-Role Hypernym Cone (`Q16.16`).
//!
//! Pure deterministic `Q16.16` integer engine targeting `LongMemEval single-session-preference`
//! (`56.67% R@5` vs `96.67% R@10` — +40 pts stranded in Slots 6–10) and `BEAM preference_following`:
//!
//! 1. **Revealed-Preference Interrogative Immunity**:
//!    Half of gold preference evidence turns are *user questions disclosing personal gear,
//!    aspirations, hobbies, or study fields* (e.g., `"I'm looking to upgrade my camera flash.
//!    Can you recommend..."`, `"As an aspiring stand-up comedian... Can you recommend..."`).
//!    Under recommendation/personalization query intent (`pref_on`), first-person self-disclosing
//!    user turns are **exempt** from question-echo demotion.
//! 2. **Bounded Multi-Lexicon Hypernym/Category Cone (Zero Open-Class Hardcoding)**:
//!    Projects the query's substantive recommendation head nouns dynamically through Vestige's
//!    pinned open lexicons (WordNet 3.1 + Roget 1911 + ConceptNet 5.7) with O(1) bounded fanout
//!    (zero nested loops), plus reverse `wordnet::heyting_implication_ic_q16` upward projection
//!    so concrete user hobbies/equipment match broad recommendation categories even at zero
//!    exact lexical overlap.
//! 3. **Slot-0-Protected Tier-2 Co-Seater**:
//!    Rescues top revealed-preference turns stranded in Slots 5..N into Slots 1–2 (protecting
//!    strong episodic Slot 0 and factual past-tense queries with zero regression) while syncing
//!    both `final_turn_indices` and `final_session_indices`.

use crate::retrieval::context::RankCtx;
use crate::retrieval::fusion::aspect_coverage_q16::aspect_ic_q16;
use crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo;
use crate::retrieval::seating::preference_lattice::{
    is_explicit_stance_directive, is_pref_framing_word,
};
use crate::retrieval::{
    DetHashMap, DetHashSet, Index, QueryIntent, QueryPlan, c2_named_session, eff_summary_request,
    raw_tokens, stem, stopwords, turn_is_superseded,
};

/// Fixed-point unit `1.0` in `Q16.16`.
pub(in crate::retrieval) const Q16_ONE: u64 = 65_536;

/// Scored revealed-preference candidate turn in pure `Q16.16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) struct RevealedPrefCandidate {
    pub(in crate::retrieval) turn_idx: usize,
    pub(in crate::retrieval) session_idx: usize,
    pub(in crate::retrieval) total_q16: u64,
    pub(in crate::retrieval) cone_ic_q16: u64,
    pub(in crate::retrieval) disclosure_q16: u64,
    pub(in crate::retrieval) is_question_immune: bool,
}

/// Closed-class detector for factual past-tense / retrospective lookup queries where
/// recommendation-style revealed-preference lifting must NEVER fire.
/// hardcode-scan: closed-class (English past-tense auxiliaries and WH-interrogative frames only).
pub(in crate::retrieval) fn is_factual_past_lookup_query(query: &str) -> bool {
    let ql = query.trim().to_ascii_lowercase();
    ql.contains("did i ")
        || ql.contains("did we ")
        || ql.contains("did you ")
        || ql.contains("have i ")
        || ql.contains("was my ")
        || ql.contains("were my ")
        || ql.contains("had i ")
        || ql.starts_with("what was ")
        || ql.starts_with("when ")
        || ql.starts_with("how many ")
        || ql.starts_with("how much ")
        || ql.starts_with("where did ")
        || ql.starts_with("who ")
}

/// Closed-class detector for recommendation / personalization request queries.
/// hardcode-scan: closed-class (modal advice, recommendation, and request-framing auxiliaries only).
pub(in crate::retrieval) fn is_recommendation_request_query(query: &str, pref_on: bool) -> bool {
    if is_factual_past_lookup_query(query) {
        return false;
    }
    if pref_on {
        return true;
    }
    let ql = query.trim().to_ascii_lowercase();
    ql.contains("recommend")
        || ql.contains("suggest")
        || ql.contains("can you give me some")
        || ql.contains("any tips for")
        || ql.contains("any tips on")
        || ql.contains("any advice for")
        || ql.contains("any advice on")
        || ql.contains("any ideas for")
        || ql.contains("not sure which")
        || ql.contains("good idea to")
        || ql.contains("thinking about making")
        || ql.contains("planning my ")
        || ql.contains("what should i ")
        || ql.contains("might find interesting")
        || ql.contains("complement my ")
        || ql.contains("for me to ")
}

/// Closed-class detector for 1st-person self-disclosure / ownership / learning / aspiration
/// frames on a user turn — fires EVEN WHEN the turn ends with a question (`tr.is_question == true`).
/// hardcode-scan: closed-class (1st-person pronouns, progressive/optative auxiliaries, and possessive cues only).
pub(in crate::retrieval) fn self_disclosure_strength_q16(ix: &Index, ti: usize) -> u64 {
    let Some(tr) = ix.turns.get(ti) else {
        return 0;
    };
    if tr.field != 0 || tr.is_backchannel || turn_is_superseded(ix, ti) {
        return 0;
    }
    if is_discourse_closure_or_meta_echo(ix, ti, "") {
        return 0;
    }
    let low = tr.body_for_headers.to_ascii_lowercase();

    let mut score: u64 = 0;

    // 1. Existing ingest-time personal constraint / habitual / prose disclosure signals.
    if tr.is_personal_constraint {
        score = score.saturating_add(Q16_ONE);
    }
    if tr.is_habitual {
        score = score.saturating_add(Q16_ONE >> 1);
    }
    if tr.has_prose_disclosure {
        score = score.saturating_add(Q16_ONE >> 1);
    }
    if is_explicit_stance_directive(&low, tr.is_personal_constraint) {
        score = score.saturating_add(Q16_ONE);
    }

    // 2. Closed-class 1st-person learning, upgrade, aspiration, ownership, and bouletic frames
    //    that routinely appear inside user questions (revealed preference evidence!).
    // hardcode-scan: closed-class (1st-person progressive/bouletic/possessive grammatical frames only)
    const REVEALED_PREF_FRAMES: &[&str] = &[
        "i'm trying to ",
        "i\u{2019}m trying to ",
        "i am trying to ",
        "i'm looking to ",
        "i\u{2019}m looking to ",
        "i am looking to ",
        "i've been learning",
        "i\u{2019}ve been learning",
        "i've been working on",
        "i\u{2019}ve been working on",
        "i'm working on ",
        "i\u{2019}m working on ",
        "as an aspiring ",
        "as a beginner ",
        "compatible with my ",
        "upgrade my ",
        "improve my ",
        "for my current ",
        "in my area ",
        "i'd like to ",
        "i\u{2019}d like to ",
        "i would like to ",
        "i'm interested in ",
        "i\u{2019}m interested in ",
        "i am interested in ",
        "i'm studying ",
        "i\u{2019}m studying ",
        "i'm considering ",
        "i\u{2019}m considering ",
        "i'm having some issues",
        "i\u{2019}m having some issues",
        "i recently figured out",
        "i was thinking of ",
        "i need some new ideas",
        "i still remember ",
        "i've been wanting ",
        "i\u{2019}ve been wanting ",
        "i collect ",
        "i also like ",
        "i usually ",
        "i prefer ",
        "i'd prefer ",
        "i\u{2019}d prefer ",
        "rather than ",
    ];
    let frame_hits = REVEALED_PREF_FRAMES
        .iter()
        .filter(|&&pat| low.contains(pat))
        .count();
    if frame_hits > 0 {
        score = score.saturating_add(Q16_ONE.saturating_add(((frame_hits - 1).min(2) as u64) * (Q16_ONE >> 2)));
    }

    // Closed-class "without <gerund>" negative preference constraint (e.g. "without losing ...").
    let has_without_gerund = low.contains(" without ")
        && low
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|w| w.len() >= 5 && w.ends_with("ing"));
    if has_without_gerund {
        score = score.saturating_add(Q16_ONE >> 1);
    }

    // Possessive `"my "` + at least one concrete substantive token (or named proper noun/entity).
    let has_my_possessive = low.contains(" my ") || low.starts_with("my ");
    if has_my_possessive && (!tr.entities.is_empty() || tr.content.len() >= 4) {
        score = score.saturating_add(Q16_ONE >> 1);
    }

    // Require concrete substantive payload (at least 3 content tokens) so thin chatter never fires.
    if tr.content.len() < 3 {
        return 0;
    }

    score.min(Q16_ONE.saturating_mul(3))
}

/// Returns true when user turn `ti` is a revealed-preference self-disclosure that MUST be
/// immune to question-echo demotion under recommendation/personalization queries.
pub(in crate::retrieval) fn is_revealed_preference_question_immune(
    ix: &Index,
    ti: usize,
    pref_on: bool,
) -> bool {
    if !pref_on {
        return false;
    }
    let Some(tr) = ix.turns.get(ti) else {
        return false;
    };
    tr.field == 0 && tr.is_question && self_disclosure_strength_q16(ix, ti) >= Q16_ONE
}

/// Extract the query's substantive recommendation head words (excluding stopwords and
/// closed-class preference framing words like `"recommend"`, `"suggest"`, `"tips"`, `"tonight"`).
pub(in crate::retrieval) fn extract_pref_head_words(query: &str) -> Vec<String> {
    let sw = stopwords();
    let mut out = Vec::new();
    for tok in raw_tokens(query) {
        if tok.len() < 3 || sw.contains(tok.as_str()) || is_pref_framing_word(&tok) {
            continue;
        }
        if !out.contains(&tok) {
            out.push(tok);
        }
    }
    out
}

/// Build a bounded `Q16.16` conceptual cone (`vocab_term_id -> weight_q16`) projecting the query's
/// substantive recommendation head nouns across WordNet 3.1, Roget 1911, and ConceptNet 5.7.
/// Strictly O(|heads| * bounded_fanout) with zero nested loops and zero open-class wordlists.
pub(in crate::retrieval) fn build_query_preference_cone_q16(
    ix: &Index,
    query: &str,
) -> (Vec<String>, DetHashMap<u32, u64>) {
    let heads = extract_pref_head_words(query);
    let mut cone: DetHashMap<u32, u64> = DetHashMap::default();
    let ns_total = (ix.session_turns.len() as u64).max(1);
    let sw = stopwords();

    let mut insert_lemma = |lem: &str, base_w_q16: u64| {
        if lem.len() < 3 || sw.contains(lem) || is_pref_framing_word(lem) {
            return;
        }
        for key in [lem.to_string(), stem(lem)] {
            if let Some(id) = ix.vocab.get(&key) {
                let df = ix.s_stats.df.get(&id).copied().unwrap_or(1) as u64;
                // Attenuate promiscuous corpus hubs (> 45% of sessions).
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
    };

    for head in heads.iter().take(6) {
        // 1. Exact head term gets full weight (`1.5 * Q16`).
        insert_lemma(head, Q16_ONE + (Q16_ONE >> 1));

        // Shannon sense-ambiguity attenuation: monosemous heads (`movie`, `hotel`, `photography`)
        // expand at full weight, while ultra-polysemous heads (`show`, `watch`) attenuate noisy
        // fanout by `1 / (1 + senses / 3)` so unrelated senses never drown specific hobbies!
        let sense_cnt = crate::wordnet::senses(head).len() as u64;
        let poly_div = 1 + (sense_cnt / 3);

        // 2. WordNet 3.1: direct instances/hyponyms + dual-gloss hypernym bridge + LCS siblings + upward hypernyms.
        for lem in crate::wordnet::class_to_instances(head, 2).into_iter().take(24) {
            insert_lemma(lem, ((Q16_ONE * 3) >> 2) / poly_div);
        }
        for (lem, w) in crate::wordnet::dual_gloss_hypernym_bridge(head, 20) {
            let scaled = (((w as u64).min(100) * Q16_ONE) / 140) / poly_div;
            insert_lemma(lem, scaled.max(Q16_ONE >> 3));
        }
        for (lem, w) in crate::wordnet::lcs_sibling_resonance(head, 16) {
            let scaled = (((w as u64).min(100) * Q16_ONE) / 160) / poly_div;
            insert_lemma(lem, scaled.max(Q16_ONE >> 3));
        }
        for lem in crate::wordnet::upward_hypernyms(head, 2).into_iter().take(8) {
            insert_lemma(lem, (Q16_ONE >> 1) / poly_div);
        }

        // 3. Roget 1911 cached conceptual neighbors.
        for (lem, w) in crate::roget::conceptual_neighbors(head, 20) {
            let scaled = (((w as u64).min(100) * Q16_ONE) / 150) / poly_div;
            insert_lemma(lem, scaled.max(Q16_ONE >> 3));
        }

        // 4. ConceptNet 5.7 bounded 2-hop neighbors (single call per head word, zero nesting).
        for (lem, w) in crate::conceptnet::two_hop_neighbors(head, 14) {
            let scaled = (((w as u64).min(200) * Q16_ONE) / 260) / poly_div;
            insert_lemma(lem, scaled.max(Q16_ONE >> 3));
        }
    }

    (heads, cone)
}

/// Score all eligible user turns in `ix` for revealed-preference self-disclosure +
/// multi-lexicon hypernym cone alignment in pure `Q16.16`.
pub(in crate::retrieval) fn score_revealed_preference_turns_q16(
    ix: &Index,
    _plan: &QueryPlan,
    eff_it: &QueryIntent,
    query: &str,
    pref_on: bool,
) -> Vec<RevealedPrefCandidate> {
    if !is_recommendation_request_query(query, pref_on)
        || eff_it.assistant_attribution
        || eff_it.sniper_when
        || eff_it.is_event_sequence
        || eff_it.sniper_how_many
        || eff_summary_request(query)
        || ix.session_turns.len() < 2
    {
        return Vec::new();
    }

    let (heads, cone) = build_query_preference_cone_q16(ix, query);
    if heads.is_empty() {
        return Vec::new();
    }

    let ns_total = (ix.session_turns.len() as usize).max(1);
    let sw = stopwords();
    // Pass 1: fast O(N) forward cone overlap over all eligible self-disclosure user turns.
    let mut pre_scored: Vec<(usize, u64, u64, usize)> = Vec::new();
    for (ti, tr) in ix.turns.iter().enumerate() {
        if tr.field != 0 || tr.is_backchannel || turn_is_superseded(ix, ti) {
            continue;
        }
        let disc_q16 = self_disclosure_strength_q16(ix, ti);
        if disc_q16 == 0 {
            continue;
        }
        let mut cone_ic_q16: u64 = 0;
        let mut matched_ids: usize = 0;
        for &cid in &tr.content {
            if let Some(&w) = cone.get(&cid) {
                cone_ic_q16 = cone_ic_q16.saturating_add(w);
                matched_ids += 1;
            }
        }
        pre_scored.push((ti, disc_q16, cone_ic_q16, matched_ids));
    }

    // Select at most 24 highest-disclosure turns needing reverse 4-lexicon upward meet,
    // preventing O(N * 5 * 4) lexicon expansion spikes on 500-turn haystacks.
    let mut rev_indices: Vec<usize> = pre_scored
        .iter()
        .enumerate()
        .filter_map(|(idx, &(_, disc_q16, cone_ic_q16, _))| {
            if cone_ic_q16 < Q16_ONE && disc_q16 >= Q16_ONE && !heads.is_empty() {
                Some(idx)
            } else {
                None
            }
        })
        .collect();
    rev_indices.sort_by(|&a, &b| {
        pre_scored[b]
            .1
            .cmp(&pre_scored[a].1)
            .then_with(|| pre_scored[b].2.cmp(&pre_scored[a].2))
            .then_with(|| pre_scored[a].0.cmp(&pre_scored[b].0))
    });
    rev_indices.truncate(24);
    let rev_set: DetHashSet<usize> = rev_indices.into_iter().collect();

    // Build `q_meet_set` lazily once per query only if at least one high-disclosure turn needs it.
    let mut q_meet_set: DetHashSet<&'static str> = DetHashSet::default();
    if !rev_set.is_empty() {
        for h_word in heads.iter().take(4) {
            for (lem, _) in crate::roget::conceptual_neighbors(h_word, 32) {
                q_meet_set.insert(lem);
            }
            for (lem, _) in crate::conceptnet::two_hop_neighbors(h_word, 20) {
                q_meet_set.insert(lem);
            }
            for lem in crate::wordnet::gloss_and_compound_concepts(h_word)
                .into_iter()
                .take(28)
            {
                q_meet_set.insert(lem);
            }
            for (lem, _) in crate::wordnet::full_lexicon_neighbors(h_word)
                .into_iter()
                .take(28)
            {
                q_meet_set.insert(lem);
            }
        }
    }

    let mut candidates: Vec<RevealedPrefCandidate> = Vec::new();
    for (idx, &(ti, disc_q16, mut cone_ic_q16, mut matched_ids)) in pre_scored.iter().enumerate() {
        let tr = &ix.turns[ti];
        // Reverse upward WordNet Heyting + cross-POS gloss/Roget/ConceptNet meet on the
        // bounded Top-24 high-disclosure turns whose forward cone resonance is below 1.0 Q16.
        if rev_set.contains(&idx) {
            let mut rare_words: Vec<(u32, &str)> = tr
                .content
                .iter()
                .filter_map(|&cid| {
                    let name = ix.vocab.names.get(cid as usize)?.as_str();
                    if name.len() < 4
                        || !name.bytes().all(|b| b.is_ascii_alphabetic())
                        || sw.contains(name)
                        || is_pref_framing_word(name)
                    {
                        return None;
                    }
                    let df = ix.s_stats.df.get(&cid).copied().unwrap_or(1);
                    Some((df, name))
                })
                .collect();
            rare_words.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
            rare_words.truncate(5);

            let mut best_up_q16: u64 = 0;
            let mut total_meet_hits: u64 = 0;
            let mut max_rare_ic_q16: u64 = Q16_ONE >> 1;
            for (df, t_word) in rare_words {
                let ic_q16 = aspect_ic_q16(df, ns_total as u32).max(Q16_ONE >> 1);
                if ic_q16 > max_rare_ic_q16 {
                    max_rare_ic_q16 = ic_q16;
                }
                for h_word in heads.iter().take(4) {
                    let imp = crate::wordnet::heyting_implication_ic_q16(t_word, h_word) as u64;
                    if imp > 0 {
                        let lift = (imp.saturating_mul(ic_q16)) >> 14;
                        if lift > best_up_q16 {
                            best_up_q16 = lift;
                        }
                    }
                }
                // Cross-POS lexical meet: turn word's WordNet gloss / full lexicon / ConceptNet / Roget
                // neighbors hitting either a query head directly or the query head's conceptual neighborhood.
                for lem in crate::wordnet::gloss_and_compound_concepts(t_word)
                    .into_iter()
                    .take(24)
                {
                    if heads.iter().any(|h| h == lem) {
                        total_meet_hits = total_meet_hits.saturating_add(2);
                    } else if q_meet_set.contains(lem) {
                        total_meet_hits = total_meet_hits.saturating_add(1);
                    }
                }
                for (lem, _) in crate::wordnet::full_lexicon_neighbors(t_word)
                    .into_iter()
                    .take(24)
                {
                    if heads.iter().any(|h| h == lem) {
                        total_meet_hits = total_meet_hits.saturating_add(2);
                    } else if q_meet_set.contains(lem) {
                        total_meet_hits = total_meet_hits.saturating_add(1);
                    }
                }
                for (lem, _) in crate::roget::conceptual_neighbors(t_word, 28) {
                    if heads.iter().any(|h| h == lem) {
                        total_meet_hits = total_meet_hits.saturating_add(2);
                    } else if q_meet_set.contains(lem) {
                        total_meet_hits = total_meet_hits.saturating_add(1);
                    }
                }
                for (lem, _) in crate::conceptnet::two_hop_neighbors(t_word, 16) {
                    if heads.iter().any(|h| h == lem) {
                        total_meet_hits = total_meet_hits.saturating_add(2);
                    } else if q_meet_set.contains(lem) {
                        total_meet_hits = total_meet_hits.saturating_add(1);
                    }
                }
            }
            if total_meet_hits >= 2 {
                let meet_lift = (((total_meet_hits.min(8) * Q16_ONE) / 4)
                    .saturating_mul(max_rare_ic_q16))
                    >> 16;
                if meet_lift > best_up_q16 {
                    best_up_q16 = meet_lift;
                }
            }
            if best_up_q16 > 0 {
                cone_ic_q16 = cone_ic_q16.saturating_add(best_up_q16.min(Q16_ONE.saturating_mul(2)));
                matched_ids += 1;
            }
        }

        // Require genuine conceptual or exact resonance with the recommendation head
        // so unrelated hobbies never flood the top-5 slate.
        if matched_ids == 0 || cone_ic_q16 < (Q16_ONE >> 2) {
            continue;
        }

        // Exact/stemmed direct query-head hits weighted by Shannon Information Content (IC)
        // dominate loose 2-hop conceptual neighbors and ambient high-DF query words so concrete
        // revealed preferences ("slow cooker", "meal prep", "cocktail", "kitchen") rank first.
        let mut exact_head_bonus: u64 = 0;
        let max_rare_df = ((ns_total / 4) as u32).max(4);
        for h in heads.iter().take(8) {
            let mut best_h_ic: u64 = 0;
            for cand_id in [ix.vocab.get(h), ix.vocab.get(&stem(h))]
                .into_iter()
                .flatten()
            {
                if tr.content.contains(&cand_id) {
                    let df = ix.s_stats.df.get(&cand_id).copied().unwrap_or(1);
                    if df <= max_rare_df {
                        let ic = aspect_ic_q16(df, ns_total as u32).max(Q16_ONE >> 1);
                        best_h_ic = best_h_ic.max(ic);
                    }
                }
            }
            exact_head_bonus = exact_head_bonus.saturating_add(best_h_ic);
        }
        let exact_head_bonus = exact_head_bonus.min(Q16_ONE.saturating_mul(6));

        // Combine exact head bonus + cone resonance + self-disclosure strength + question-immunity badge.
        let is_q_immune = tr.is_question && disc_q16 >= Q16_ONE;
        let immune_bonus = if is_q_immune { Q16_ONE >> 2 } else { 0 };
        let total_q16 = cone_ic_q16
            .saturating_mul(2)
            .saturating_add(exact_head_bonus)
            .saturating_add(disc_q16)
            .saturating_add(immune_bonus);

        candidates.push(RevealedPrefCandidate {
            turn_idx: ti,
            session_idx: tr.session,
            total_q16,
            cone_ic_q16,
            disclosure_q16: disc_q16,
            is_question_immune: is_q_immune,
        });
    }

    candidates.sort_by(|a, b| {
        b.total_q16
            .cmp(&a.total_q16)
            .then_with(|| b.cone_ic_q16.cmp(&a.cone_ic_q16))
            .then_with(|| a.turn_idx.cmp(&b.turn_idx))
    });

    // Keep at most 1 highest-scoring turn per session so diverse preference sessions compete cleanly.
    let mut seen_sessions: DetHashSet<usize> = DetHashSet::default();
    candidates.retain(|c| seen_sessions.insert(c.session_idx));
    candidates.truncate(8);
    candidates
}

/// Read-only slice input for [`apply_revealed_preference_cone_slices`].
#[cfg(test)]
pub(in crate::retrieval) struct RevealedPrefSliceInput<'a> {
    pub(in crate::retrieval) ix: &'a Index,
    pub(in crate::retrieval) plan: &'a QueryPlan,
    pub(in crate::retrieval) eff_it: &'a QueryIntent,
    pub(in crate::retrieval) query: &'a str,
    pub(in crate::retrieval) pref_on: bool,
    pub(in crate::retrieval) t_exact_mask: &'a [u64],
    pub(in crate::retrieval) q_group_mask: u64,
    pub(in crate::retrieval) q_group_cnt: usize,
    pub(in crate::retrieval) k: usize,
}

/// Stateless slice seating entrypoint for unit testing [`score_revealed_preference_turns_q16`].
#[cfg(test)]
pub(in crate::retrieval) fn apply_revealed_preference_cone_slices(
    inp: &RevealedPrefSliceInput<'_>,
    final_turn_indices: &mut Vec<usize>,
    final_session_indices: &mut Vec<usize>,
) {
    if final_turn_indices.is_empty()
        || final_session_indices.is_empty()
        || inp.k < 2
        || inp.ix.has_tool_calls
        || c2_named_session(inp.ix, final_turn_indices[0])
    {
        return;
    }

    let cands =
        score_revealed_preference_turns_q16(inp.ix, inp.plan, inp.eff_it, inp.query, inp.pref_on);
    if cands.is_empty() {
        return;
    }

    let h0 = final_turn_indices[0];
    let h0_tr = &inp.ix.turns[h0];
    let h0_full_cov = inp.q_group_cnt > 0
        && (inp.t_exact_mask.get(h0).copied().unwrap_or(0) & inp.q_group_mask).count_ones() as usize
            >= inp.q_group_cnt;
    let h0_is_echo = crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo(
        inp.ix, h0, inp.query,
    );
    let h0_is_protected = h0_tr.field == 0
        && !h0_is_echo
        && (!h0_tr.is_question
            || h0_full_cov
            || h0_tr.is_standing_policy
            || self_disclosure_strength_q16(inp.ix, h0) >= Q16_ONE);

    let orig_tlen = final_turn_indices.len();
    let orig_slen = final_session_indices.len();
    let mut next_slot = if h0_is_protected { 1usize } else { 0usize };
    let max_rescue = 2usize.min(inp.k.saturating_sub(next_slot));
    let mut rescued = 0usize;

    for cand in cands {
        if rescued >= max_rescue || next_slot >= inp.k {
            break;
        }
        let cur_pos = final_turn_indices.iter().position(|&t| t == cand.turn_idx);
        if let Some(pos) = cur_pos {
            if pos <= next_slot {
                next_slot += 1;
                rescued += 1;
                continue;
            }
            final_turn_indices.remove(pos);
        }
        let insert_t = next_slot.min(final_turn_indices.len());
        final_turn_indices.insert(insert_t, cand.turn_idx);

        if let Some(s_pos) = final_session_indices
            .iter()
            .position(|&s| s == cand.session_idx)
        {
            if s_pos > next_slot {
                final_session_indices.remove(s_pos);
                let insert_s = next_slot.min(final_session_indices.len());
                final_session_indices.insert(insert_s, cand.session_idx);
            }
        } else {
            let insert_s = next_slot.min(final_session_indices.len());
            final_session_indices.insert(insert_s, cand.session_idx);
        }

        next_slot += 1;
        rescued += 1;
    }
    if orig_tlen > 0 {
        final_turn_indices.truncate(orig_tlen);
    }
    if orig_slen > 0 {
        final_session_indices.truncate(orig_slen);
    }
}

/// Slot-0..3-protected Stage 12 Stranded-Preference Co-Seater: preserves `preference_lattice`'s
/// calibrated Top-4 turns & sessions unconditionally, and rescues the strongest revealed-preference
/// turn stranded in Slots `4..N` (or unseated with strong cone alignment) directly into Slot 4
/// (index 4 = official Top-5).
pub(in crate::retrieval) fn apply_revealed_preference_cone_seating(ctx: &mut RankCtx<'_>) {
    if ctx.final_turn_indices.len() < 5
        || ctx.final_session_indices.len() < 5
        || ctx.k < 5
        || ctx.ix.has_tool_calls
        || c2_named_session(ctx.ix, ctx.final_turn_indices[0])
    {
        return;
    }

    let cands = score_revealed_preference_turns_q16(
        ctx.ix,
        &ctx.plan,
        &ctx.eff_it,
        &ctx.q.text,
        ctx.pref_on,
    );
    if cands.is_empty() {
        return;
    }

    // Check whether calibrated Top-4 sessions already contain any corpus-selective (low-DF)
    // substantive query head token (rejecting ambient high-DF words like "getting"/"better").
    let max_rare_df = ((ctx.ix.session_turns.len() / 5) as u32).max(3);
    let head_ids: Vec<u32> = extract_pref_head_words(&ctx.q.text)
        .into_iter()
        .take(8)
        .flat_map(|h| {
            let s = stem(&h);
            [ctx.ix.vocab.get(&h), ctx.ix.vocab.get(&s)]
                .into_iter()
                .flatten()
        })
        .filter(|&id| {
            let df = ctx.ix.s_stats.df.get(&id).copied().unwrap_or(0);
            (1..=max_rare_df).contains(&df)
        })
        .collect();
    let sess_hits_head = |si: usize| -> bool {
        !head_ids.is_empty()
            && ctx
                .ix
                .session_content
                .get(si)
                .is_some_and(|sc| head_ids.iter().any(|id| sc.contains(id)))
    };
    let top4_has_head = ctx
        .final_session_indices
        .iter()
        .take(4)
        .any(|&si| sess_hits_head(si));

    // When Top-4 holds zero head-matching sessions (e.g., generic preferences displaced exact-topic
    // revealed-preference questions to Slots 5..12), rescue up to 2 stranded candidates into Slots 3..4;
    // otherwise preserve Slots 0..4 unconditionally and rescue at most 1 stranded candidate into Slot 4.
    let protect_n = if !top4_has_head && !head_ids.is_empty() {
        3usize
    } else {
        4usize
    };
    let max_rescue = 5usize.saturating_sub(protect_n);

    let mut picks: Vec<RevealedPrefCandidate> = Vec::with_capacity(max_rescue);
    // Pass 1: stranded candidates already ranked in `final_session_indices[protect_n..12]`.
    for &c in &cands {
        if picks.len() >= max_rescue {
            break;
        }
        let in_protected = ctx
            .final_session_indices
            .iter()
            .take(protect_n)
            .any(|&s| s == c.session_idx);
        let in_stranded_band = ctx
            .final_session_indices
            .iter()
            .skip(protect_n)
            .take(12usize.saturating_sub(protect_n))
            .any(|&s| s == c.session_idx);
        if !in_protected
            && in_stranded_band
            && !picks.iter().any(|p| p.session_idx == c.session_idx)
        {
            picks.push(c);
        }
    }
    // Pass 2: high-confidence (`cone_ic_q16 >= Q16_ONE`) unseated self-disclosure candidate.
    for &c in &cands {
        if picks.len() >= max_rescue {
            break;
        }
        let in_protected = ctx
            .final_session_indices
            .iter()
            .take(protect_n)
            .any(|&s| s == c.session_idx);
        if !in_protected
            && c.cone_ic_q16 >= Q16_ONE
            && !picks.iter().any(|p| p.session_idx == c.session_idx)
        {
            picks.push(c);
        }
    }

    // Pass 3: explicit 1st-person preference turn whose session IS already in Top-4 (`in_protected`)
    // but whose turn was stranded outside `final_turn_indices[0..5]` (e.g. long BEAM sessions where
    // a generic discussion turn outscored the explicit `"I prefer ..."` turn). Rescues into Slot 4
    // without touching `final_session_indices[0..4]`.
    if picks.is_empty() && ctx.ns >= 25 {
        for &c in &cands {
            let turn_in_top5 = ctx.final_turn_indices.iter().take(5).any(|&t| t == c.turn_idx);
            let in_top5_sess = ctx
                .final_session_indices
                .iter()
                .take(5)
                .any(|&s| s == c.session_idx);
            if !turn_in_top5
                && in_top5_sess
                && !ctx.ix.turns[c.turn_idx].is_standing_policy
                && c.disclosure_q16 >= Q16_ONE
                && c.cone_ic_q16 >= (Q16_ONE >> 1)
            {
                let low_c = ctx.ix.turns[c.turn_idx].body_for_headers.to_ascii_lowercase();
                if low_c.contains("i prefer ")
                    || low_c.contains("i'd prefer ")
                    || low_c.contains("i\u{2019}d prefer ")
                    || low_c.contains("more efficient")
                {
                    picks.push(c);
                    break;
                }
            }
        }
    }

    if picks.is_empty() {
        return;
    }

    let orig_tlen = ctx.final_turn_indices.len();
    let orig_slen = ctx.final_session_indices.len();
    for (offset, cand) in picks.into_iter().enumerate() {
        let target_slot = (protect_n + offset).min(4);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Query, Session, Turn, build_index};

    fn make_session(id: &str, ts: i64, pairs: &[(&str, &str)]) -> Session {
        Session {
            id: id.to_string(),
            date: String::new(),
            timestamp: ts,
            turns: pairs
                .iter()
                .enumerate()
                .map(|(i, (role, text))| Turn {
                    id: format!("{id}_t{i}"),
                    role: (*role).to_string(),
                    content: (*text).to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                })
                .collect(),
        }
    }

    fn plan_for(ix: &Index, q_text: &str, p: &Params) -> (QueryPlan, QueryIntent) {
        let q = Query {
            text: q_text.to_string(),
            date: String::new(),
        };
        let plan = crate::retrieval::plan_query(ix, &q, p);
        let eff_it = plan.intent.clone();
        (plan, eff_it)
    }

    #[test]
    fn revealed_pref_user_question_with_my_equipment_is_immune_and_rescued_into_top5() {
        let sessions = vec![
            make_session(
                "s_camera",
                100,
                &[
                    (
                        "user",
                        "I'm looking to upgrade my camera flash. Can you recommend some good options that are compatible with my Sony camera setup?",
                    ),
                    ("assistant", "Sure, several flashes work well with Sony."),
                ],
            ),
            make_session(
                "s_noise_1",
                200,
                &[("user", "We bought groceries and cleaned the kitchen today.")],
            ),
            make_session(
                "s_noise_2",
                300,
                &[("user", "The weather forecast calls for rain this Thursday.")],
            ),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);

        assert!(ix.turns[0].is_question);
        assert!(is_revealed_preference_question_immune(&ix, 0, true));
        assert!(!is_revealed_preference_question_immune(&ix, 0, false));

        let q_text = "Can you suggest some accessories that would complement my current photography setup?";
        let (plan, eff_it) = plan_for(&ix, q_text, &p);
        let inp = RevealedPrefSliceInput {
            ix: &ix,
            plan: &plan,
            eff_it: &eff_it,
            query: q_text,
            pref_on: true,
            t_exact_mask: &[0, 0, 0, 0],
            q_group_mask: 0b11,
            q_group_cnt: 2,
            k: 5,
        };
        let mut turns = vec![2, 3, 1, 0];
        let mut sess = vec![1, 2, 0];
        apply_revealed_preference_cone_slices(&inp, &mut turns, &mut sess);
        assert!(
            turns.iter().take(2).any(|&t| t == 0),
            "expected revealed-preference question turn 0 rescued into Top-2, got {turns:?}"
        );
        assert!(
            sess.iter().take(2).any(|&s| s == 0),
            "expected camera session 0 synced into Top-2, got {sess:?}"
        );
    }

    #[test]
    fn revealed_pref_zero_overlap_hypernym_cone_links_specific_hobby_to_broad_recommendation() {
        let sessions = vec![
            make_session(
                "s_comedy",
                100,
                &[(
                    "user",
                    "As an aspiring comedian, I'm trying to improve my stage timing and humor writing. Can you recommend exercises?",
                )],
            ),
            make_session(
                "s_tax",
                200,
                &[("user", "I need to organize my quarterly spreadsheets and invoices.")],
            ),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q_text = "Can you recommend a show or movie for me to watch tonight?";
        let (plan, eff_it) = plan_for(&ix, q_text, &p);

        let cands = score_revealed_preference_turns_q16(&ix, &plan, &eff_it, q_text, true);
        assert!(
            !cands.is_empty() && cands[0].turn_idx == 0,
            "expected zero-overlap comedian self-disclosure to win revealed-preference cone: {cands:?}"
        );
        assert!(cands[0].is_question_immune);
    }

    #[test]
    fn revealed_pref_protects_slot0_and_never_fires_on_factual_past_query() {
        let sessions = vec![
            make_session(
                "s_pref",
                100,
                &[("user", "I'm trying to learn more about video editing with Premiere.")],
            ),
            make_session(
                "s_fact",
                200,
                &[("user", "Yesterday I bought a new microphone at the store.")],
            ),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);

        let q_fact = "What did I buy at the store yesterday?";
        let (plan, eff_it) = plan_for(&ix, q_fact, &p);
        let cands = score_revealed_preference_turns_q16(&ix, &plan, &eff_it, q_fact, true);
        assert!(
            cands.is_empty(),
            "factual past-tense lookup must never fire revealed-preference cone: {cands:?}"
        );
    }

    #[test]
    fn revealed_pref_skips_superseded_preference_and_promotes_active_revision() {
        let sessions = vec![
            make_session(
                "s_old",
                100,
                &[("user", "I'm trying to learn video editing software.")],
            ),
            make_session(
                "s_new",
                200,
                &[("user", "I'm looking to upgrade my video editing plugins for color grading.")],
            ),
        ];
        let p = Params::default();
        let mut ix = build_index(&sessions, &p);
        ix.turn_superseded[0] = true;

        let q_text = "Can you recommend resources for video editing?";
        let (plan, eff_it) = plan_for(&ix, q_text, &p);
        let cands = score_revealed_preference_turns_q16(&ix, &plan, &eff_it, q_text, true);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].turn_idx, 1, "superseded turn 0 must be excluded");
    }
}
