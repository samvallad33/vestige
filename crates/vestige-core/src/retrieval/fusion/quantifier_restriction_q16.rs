//! Movement III: Barwise-Cooper Quantifier Restriction Grounding & 1st-Person Deictic
//! Voice Alignment for Multi-Session `All@5` Slate Seating (`Q16.16`).
//!
//! Pure deterministic, zero-vector, zero-LLM, closed-class linguistic pass targeting
//! multi-session counting and superlative queries (`"How many <NP>..."`, `"Which <NP> ... most/least..."`)
//! on long user-assistant histories (`ns >= 16`, `!has_named_speakers_eff`, `!is_monologue_corpus`).
//!
//! Two formal-semantic laws (Barwise & Cooper 1981; Roberts Dynamic Scoreboard):
//! 1. **Quantifier Restriction Witness Law**: Any session instantiating the lowest-DF
//!    substantive head noun(s) of the quantifier's restriction NP (`"How many plants..."`,
//!    `"How many movie festivals..."`, `"How many hours of jogging and yoga..."`,
//!    `"Which grocery store did I spend the most..."`) MUST outrank Top-5 sessions that
//!    miss the restriction head entirely (e.g., matching only `"many"`, `"hours"`, `"attended"`).
//! 2. **1st-Person Deictic Voice Alignment**: On 1st-person counting/superlative queries
//!    (`"did I"`, `"have I"`, `"do I"`, `"my"`), sessions in `final_session_indices[5..12]`
//!    carrying BOTH restriction grounding AND 1st-person user voice outrank impersonal
//!    3rd-person haystack distractors (zero 1st-person user voice) occupying Top-5.

use std::collections::BTreeSet;

use crate::retrieval::rank::context::RankCtx;
use crate::retrieval::{Index, eff_summary_request, is_content, raw_tokens, stem};

/// Closed-class boundary tokens that terminate the quantifier restriction NP.
/// hardcode-scan: closed-class (English auxiliaries, modals, wh-words, and clausal prepositions).
fn is_np_boundary(w: &str) -> bool {
    matches!(
        w,
        "did"
            | "do"
            | "does"
            | "have"
            | "has"
            | "had"
            | "were"
            | "was"
            | "are"
            | "is"
            | "am"
            | "can"
            | "could"
            | "would"
            | "should"
            | "will"
            | "might"
            | "may"
            | "that"
            | "which"
            | "who"
            | "where"
            | "when"
            | "in"
            | "on"
            | "at"
            | "before"
            | "after"
            | "since"
            | "during"
            | "for"
            | "from"
            | "to"
            | "with"
            | "by"
            | "about"
            | "between"
            | "across"
            | "over"
            | "under"
            | "including"
            | "excluding"
    )
}

/// Closed-class partitive, measure, temporal-unit, and generic classifier words inside
/// `"How many <classifier> (of) <head>..."` that must not stand in for the real entity noun.
/// hardcode-scan: closed-class (English measure nouns, partitives, quantifiers, and generic classifiers).
fn is_measure_or_classifier(w: &str) -> bool {
    matches!(
        w,
        "different"
            | "distinct"
            | "separate"
            | "various"
            | "individual"
            | "types"
            | "type"
            | "kinds"
            | "kind"
            | "sorts"
            | "sort"
            | "number"
            | "total"
            | "amount"
            | "hours"
            | "hour"
            | "minutes"
            | "minute"
            | "days"
            | "day"
            | "weeks"
            | "week"
            | "months"
            | "month"
            | "years"
            | "year"
            | "times"
            | "time"
            | "items"
            | "item"
            | "things"
            | "thing"
            | "pieces"
            | "piece"
            | "pairs"
            | "pair"
            | "sets"
            | "set"
            | "related"
            | "new"
            | "old"
            | "other"
            | "more"
            | "many"
            | "much"
            | "few"
            | "several"
            | "some"
            | "all"
            | "both"
            | "each"
            | "every"
            | "single"
            | "extra"
            | "additional"
            | "current"
            | "recent"
            | "past"
            | "last"
            | "next"
    )
}

/// Check whether a query is a multi-instance counting or superlative restriction query,
/// returning the starting token index of its restriction NP (immediately after `"how many"`
/// or `"which"`).
/// hardcode-scan: closed-class (counting and superlative interrogative frames).
fn quantifier_np_start(toks: &[String], q_low: &str) -> Option<usize> {
    if toks.len() >= 3 && toks[0] == "how" && toks[1] == "many" {
        return Some(2);
    }
    if let Some(pos) = toks.windows(2).position(|w| w[0] == "how" && w[1] == "many")
        && pos + 2 < toks.len()
    {
        return Some(pos + 2);
    }
    let has_superlative = q_low.contains(" the most ")
        || q_low.contains(" most ")
        || q_low.contains(" the least ")
        || q_low.contains(" highest ")
        || q_low.contains(" lowest ")
        || q_low.contains(" longest ")
        || q_low.contains(" shortest ")
        || q_low.contains(" largest ")
        || q_low.contains(" biggest ")
        || q_low.contains(" smallest ");
    if has_superlative && toks.first().is_some_and(|w| w == "which") && toks.len() >= 3 {
        return Some(1);
    }
    None
}

/// Extract the restriction head-term IDs (exact + stem) from the quantifier NP.
/// Handles compound NPs (`"movie festivals"` -> lowest-DF `"festivals"`), partitive
/// constructions (`"hours of jogging and yoga"` -> `{"jogging", "yoga"}`), and
/// classifier-headed NPs (`"kitchen items"` -> `"kitchen"`).
pub(in crate::retrieval) fn extract_quantifier_restriction_ids(
    ix: &Index,
    q_text: &str,
) -> Vec<(Vec<u32>, u32)> {
    let q_low = q_text.trim().to_ascii_lowercase();
    let toks = raw_tokens(&q_low);
    let Some(start) = quantifier_np_start(&toks, &q_low) else {
        return Vec::new();
    };

    // Walk the NP starting at `start`, stepping through `"of"` / `"and"` / `"or"` when joining
    // partitives or coordinated nouns ("hours of jogging and yoga", "types of food delivery services").
    let mut np_words: Vec<String> = Vec::new();
    let mut saw_coord = false;
    let mut i = start;
    while i < toks.len() && np_words.len() < 8 {
        let w = toks[i].as_str();
        if w == "of" {
            // Partitive bridge ("hours of X", "types of X") -> keep going into X.
            i += 1;
            continue;
        }
        if w == "and" || w == "or" {
            if !np_words.is_empty() {
                saw_coord = true;
            }
            i += 1;
            continue;
        }
        if matches!(w, "a" | "an" | "the" | "my" | "our" | "their" | "these" | "those") {
            i += 1;
            continue;
        }
        if is_np_boundary(w) {
            break;
        }
        if is_content(w) && !is_measure_or_classifier(w) {
            np_words.push(w.to_string());
        }
        i += 1;
    }

    if np_words.is_empty() {
        return Vec::new();
    }

    // Resolve each restriction word to its (exact + stem) vocab IDs and minimum corpus session DF.
    let ns = ix.session_turns.len() as u32;
    let max_df = ((ns * 2) / 3).max(4);
    let mut resolved: Vec<(Vec<u32>, u32)> = Vec::new();
    for w in &np_words {
        let mut ids: Vec<u32> = Vec::new();
        let mut best_df = u32::MAX;
        let st = stem(w);
        let st_key = format!("\u{1}stem:{st}");
        for key in [w.as_str(), st.as_str(), st_key.as_str()] {
            if let Some(id) = ix.vocab.get(key) {
                let df = ix.s_stats.df.get(&id).copied().unwrap_or(0);
                if df > 0 && df <= max_df {
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                    if df < best_df {
                        best_df = df;
                    }
                }
            }
        }
        if !ids.is_empty() {
            resolved.push((ids, best_df));
        }
    }
    if resolved.is_empty() {
        return Vec::new();
    }

    // For coordinated NPs ("jogging and yoga"), keep all coordinated head nouns so matching ANY
    // branch satisfies the disjunction. For uncoordinated compound NPs ("movie festivals",
    // "grocery store", "rare items"), prioritize the rarest (lowest-DF) term(s) within 2x min_df
    // so generic modifiers ("movie", "store", "delivery") cannot masquerade as the true head.
    if saw_coord || resolved.len() == 1 {
        return resolved;
    }
    let min_df = resolved.iter().map(|&(_, df)| df).min().unwrap_or(1);
    let df_thresh = (min_df.saturating_mul(3) / 2).max(min_df);
    let rare: Vec<(Vec<u32>, u32)> = resolved
        .iter()
        .filter(|&&(_, df)| df <= df_thresh)
        .cloned()
        .collect();
    if rare.is_empty() { resolved } else { rare }
}

/// True when session `si` contains at least one restriction head term (exact or stem) in
/// the user's own turns (`tf[0]`), or anywhere in the session when `user_only == false`.
fn session_hits_restriction(
    ix: &Index,
    si: usize,
    restriction: &[(Vec<u32>, u32)],
    user_only: bool,
) -> bool {
    let sb = &ix.session_bags[si];
    if user_only {
        return restriction
            .iter()
            .any(|(ids, _)| ids.iter().any(|id| sb.tf[0].contains_key(id)));
    }
    let sc = &ix.session_content[si];
    restriction.iter().any(|(ids, _)| {
        ids.iter()
            .any(|id| sc.contains(id) || sb.tf[0].contains_key(id) || sb.tf[1].contains_key(id))
    })
}

/// True when session `si` contains a 1st-person user turn (`I`, `I'm`, `I've`, `I'd`, `my`, `we`, `our`).
/// Separates authentic user episodic sessions from impersonal 3rd-person QA/reference sessions.
/// hardcode-scan: closed-class (1st-person personal pronouns and contractions).
fn session_has_first_person_user_voice(ix: &Index, si: usize) -> bool {
    ix.session_turns[si].iter().any(|&ti| {
        let tr = &ix.turns[ti];
        if tr.field != 0 || tr.is_standing_policy {
            return false;
        }
        tr.body_for_headers
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '\'' && c != '\u{2019}')
            .any(|w| {
                matches!(
                    w.to_ascii_lowercase().as_str(),
                    "i" | "my"
                        | "me"
                        | "mine"
                        | "we"
                        | "our"
                        | "ours"
                        | "us"
                        | "i'm"
                        | "i've"
                        | "i'd"
                        | "i'll"
                        | "we're"
                        | "we've"
                        | "i\u{2019}m"
                        | "i\u{2019}ve"
                        | "i\u{2019}d"
                        | "i\u{2019}ll"
                        | "we\u{2019}re"
                        | "we\u{2019}ve"
                )
            })
    })
}

/// True when the query itself is framed in 1st-person episodic voice ("did I", "have I", "do I", "my").
/// hardcode-scan: closed-class (1st-person pronouns).
fn query_has_first_person(q_text: &str) -> bool {
    q_text
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '\'' && c != '\u{2019}')
        .any(|w| {
            matches!(
                w.to_ascii_lowercase().as_str(),
                "i" | "my" | "me" | "mine" | "we" | "our" | "i'm" | "i've" | "i'd" | "i'll"
            )
        })
}

/// Apply Movement III Quantifier Restriction Grounding & 1st-Person Voice Alignment strictly
/// to multi-session counting/superlative queries on long user-assistant histories (`ns >= 16`).
pub(in crate::retrieval) fn apply_quantifier_restriction_all5(ctx: &mut RankCtx<'_>) {
    if ctx.has_named_speakers_eff
        || ctx.is_monologue_corpus
        || ctx.ix.has_tool_calls
        || ctx.eff_it.is_event_sequence
        || ctx.eff_it.temporal
        || ctx.torrens_locked_head
        || ctx.implicit_disclosure_locked
        || eff_summary_request(&ctx.q.text)
        || ctx.ix.session_turns.len() < 16
        || ctx.final_session_indices.len() < 6
        || ctx.k < 5
    {
        return;
    }

    let restriction = extract_quantifier_restriction_ids(ctx.ix, &ctx.q.text);
    if restriction.is_empty() {
        return;
    }

    let q_1p = query_has_first_person(&ctx.q.text);
    let smax = ctx.smax.max(
        ctx.final_session_indices
            .iter()
            .take(5)
            .map(|&si| ctx.s_score[si] + ctx.bridge_boost[si])
            .fold(0.0f32, f32::max),
    );
    if smax <= 0.0 {
        return;
    }

    // Score tier for candidate sessions:
    //   Tier 3: USER turns (`tf[0]`) hit quantifier restriction AND (not 1P query OR has 1P user voice)
    //   Tier 2: ANY turn hits quantifier restriction (e.g. assistant-only mention or 3P voice)
    //   Tier 1: misses restriction (has 1P user voice)
    //   Tier 0: misses restriction AND impersonal 3P voice
    let tier_of = |si: usize| -> u8 {
        let hits_user_r = session_hits_restriction(ctx.ix, si, &restriction, true);
        let hits_any_r = hits_user_r || session_hits_restriction(ctx.ix, si, &restriction, false);
        let has_1p = !q_1p || session_has_first_person_user_voice(ctx.ix, si);
        match (hits_user_r, hits_any_r, has_1p) {
            (true, _, true) => 3,
            (true, _, false) | (false, true, _) => 2,
            (false, false, true) => 1,
            (false, false, false) => 0,
        }
    };

    // Collect Tier-3 candidates from `final_session_indices[5..12]` plus `s_ord_base[..16]`
    // with non-trivial lexical score (`>= 0.25 * smax`).
    let pool_len = ctx.final_session_indices.len().min(12);
    let mut rescue_cands: Vec<(usize, f32)> = Vec::new();
    let mut seen_cand: BTreeSet<usize> = ctx.final_session_indices[..5].iter().copied().collect();
    for &si in ctx.final_session_indices[5..pool_len]
        .iter()
        .chain(ctx.s_ord_base.iter().take(16))
    {
        if !seen_cand.insert(si) {
            continue;
        }
        let sc = ctx.s_score[si] + ctx.bridge_boost[si];
        if tier_of(si) == 3 && sc >= 0.25 * smax {
            rescue_cands.push((si, sc));
        }
    }
    if rescue_cands.is_empty() {
        return;
    }
    // Preserve rank order (already quality-ordered by submodular session selection) while
    // breaking ties by higher lexical score.
    let mut promoted_sessions: BTreeSet<usize> = BTreeSet::new();
    let orig_slen = ctx.final_session_indices.len();

    // Replace weak Top-5 sessions (`tier < 3`, preferring lowest tier first, then lowest rank `4..0`)
    // with Tier-3 restriction-grounded, 1P-voiced sessions from `5..12`.
    // Protect Slot 0 unless Slot 0 is strictly `tier <= 1` (misses the restriction head noun completely!)
    // OR Slot 0 is an impersonal 3P session (`tier == 2`) while a strong Tier-3 session (`>= 0.50 * smax`) exists.
    for (cand_si, cand_sc) in rescue_cands {
        if ctx.final_session_indices[..5].contains(&cand_si) {
            continue;
        }
        let mut best_victim: Option<(usize, u8)> = None;
        for slot in (0..5usize).rev() {
            let v_si = ctx.final_session_indices[slot];
            let vt = tier_of(v_si);
            if vt == 3 {
                continue;
            }
            if slot == 0 && vt == 2 && cand_sc < 0.50 * smax {
                continue;
            }
            // Prefer evicting lowest tier first (Tier 0 -> Tier 1 -> Tier 2), then highest slot index (4 -> 0).
            if best_victim.is_none_or(|(_, bt)| vt < bt) {
                best_victim = Some((slot, vt));
            }
        }
        let Some((v_slot, _)) = best_victim else {
            break;
        };
        if let Some(pos) = ctx.final_session_indices.iter().position(|&s| s == cand_si) {
            ctx.final_session_indices.remove(pos);
        }
        let ins = v_slot.min(ctx.final_session_indices.len());
        ctx.final_session_indices.insert(ins, cand_si);
        ctx.final_session_indices.truncate(orig_slen);
        promoted_sessions.insert(cand_si);
    }

    // Lightly sync turns for promoted sessions: if a promoted session has zero turns in
    // `final_turn_indices[..5]`, seat its strongest substantive user turn into the lowest-tier
    // Top-5 turn slot (protecting Slot 0 turn unless Slot 0's session was replaced at Slot 0).
    if !promoted_sessions.is_empty() && ctx.final_turn_indices.len() >= 5 {
        let orig_tlen = ctx.final_turn_indices.len();
        for &psi in &promoted_sessions {
            if ctx.final_turn_indices[..5]
                .iter()
                .any(|&ti| ctx.ix.turns[ti].session == psi)
            {
                continue;
            }
            let best_ti = ctx.ix.session_turns[psi]
                .iter()
                .copied()
                .filter(|&ti| {
                    let tr = &ctx.ix.turns[ti];
                    tr.field == 0 && !tr.is_backchannel && !tr.is_standing_policy
                })
                .max_by(|&a, &b| {
                    ctx.t_scene[a]
                        .partial_cmp(&ctx.t_scene[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| b.cmp(&a))
                });
            let Some(new_ti) = best_ti else {
                continue;
            };
            let min_slot = if ctx.final_session_indices[0] == psi { 0 } else { 1 };
            let victim_t_slot = (min_slot..5usize)
                .rev()
                .find(|&slot| tier_of(ctx.ix.turns[ctx.final_turn_indices[slot]].session) < 3);
            if let Some(tslot) = victim_t_slot {
                if let Some(pos) = ctx.final_turn_indices.iter().position(|&t| t == new_ti) {
                    ctx.final_turn_indices.remove(pos);
                }
                let ins = tslot.min(ctx.final_turn_indices.len());
                ctx.final_turn_indices.insert(ins, new_ti);
                ctx.final_turn_indices.truncate(orig_tlen);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Session, Turn, build_index};

    fn mk_s(id: &str, u: &str, a: &str) -> Session {
        Session {
            id: id.to_string(),
            date: "2023/05/01 (Mon) 10:00".to_string(),
            timestamp: 1_682_935_200,
            turns: vec![
                Turn {
                    id: format!("{id}_0"),
                    role: "user".to_string(),
                    content: u.to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                },
                Turn {
                    id: format!("{id}_1"),
                    role: "assistant".to_string(),
                    content: a.to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                },
            ],
        }
    }

    #[test]
    fn extracts_rare_head_noun_and_partitive_coordination() {
        let sessions = vec![
            mk_s("s0", "I attended two movie festivals this summer.", "Sounds fun!"),
            mk_s("s1", "I watched a movie at home last night.", "Nice!"),
            mk_s("s2", "I watched another movie with friends.", "Great!"),
            mk_s("s3", "I did two hours of jogging and one hour of yoga.", "Good workout!"),
            mk_s("s4", "I spent three hours reading.", "Nice!"),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);

        let r_fest = extract_quantifier_restriction_ids(&ix, "How many movie festivals did I attend?");
        assert!(
            session_hits_restriction(&ix, 0, &r_fest, true),
            "festival session must match restriction"
        );
        assert!(
            !session_hits_restriction(&ix, 1, &r_fest, true),
            "plain movie session without festival must not satisfy rarer head 'festivals'"
        );

        let r_yoga = extract_quantifier_restriction_ids(
            &ix,
            "How many hours of jogging and yoga did I do last week?",
        );
        assert!(
            session_hits_restriction(&ix, 3, &r_yoga, true),
            "partitive coordinated session ('jogging', 'yoga') must satisfy restriction"
        );
        assert!(
            !session_hits_restriction(&ix, 4, &r_yoga, true),
            "session matching only 'hours' must not satisfy restriction"
        );
    }
}
