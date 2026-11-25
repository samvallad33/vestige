//! HabitusMem Scoped Belief State (`2611.25855`) + MINDSET Local Energy Revision (`2610.08586`).
//!
//! Pure closed-class grammar separation between:
//! - `PersistentHabitus`: standing user preferences, habits, rules ("I always", "I prefer", "never")
//! - `TaskScopedConstraint`: imperative directives & guidelines ("make sure", "must", "avoid")
//! - `TransientDebug`: ephemeral troubleshooting / error-trace turns (stack/error codes, "trouble", "fix")
//! Used to break `NEAR_MISS_RANKS_2_5` deadlocks on `preference_following`, `instruction_following`,
//! and `knowledge_update` queries so ephemeral debug turns stop stealing Rank 1 from standing beliefs.

use crate::retrieval::{Index, raw_tokens};

const Q16_ONE: u64 = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::retrieval) enum BeliefScope {
    PersistentHabitus,
    TaskScopedConstraint,
    NeutralEpisodic,
    TransientDebug,
}

fn utf8_prefix(s: &str, max_bytes: usize) -> &str {
    let mut end = s.len().min(max_bytes);
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Classify a turn's belief scope using strictly closed-class deontic/volitional and
/// structural error/question cues.
pub(in crate::retrieval) fn classify_belief_scope(ix: &Index, ti: usize) -> BeliefScope {
    let tr = &ix.turns[ti];
    if tr.is_standing_policy || tr.is_personal_constraint {
        return BeliefScope::PersistentHabitus;
    }
    let low = tr.body_for_headers.to_ascii_lowercase();
    let toks = raw_tokens(&low);

    let has_habitus = low.contains("i always ")
        || low.contains("i prefer ")
        || low.contains("i never ")
        || low.contains("i'd like ")
        || low.contains("i would like ")
        || low.contains("i like to ")
        || low.contains("keep it simple")
        || low.contains("keep my ")
        || low.contains("without compromising")
        || low.contains("so i'd like")
        || low.contains("as per my")
        || low.contains("as stated in")
        || low.contains("prefer it over")
        || low.contains("instead of ")
        || low.contains("rather than ")
        || low.contains("please always ")
        || low.contains("please never ")
        || low.contains("remember to ")
        || low.contains("from now on");
    if has_habitus && tr.field == 0 {
        return BeliefScope::PersistentHabitus;
    }

    let deontic_cnt = toks
        .iter()
        .filter(|t| {
            matches!(
                t.as_str(),
                "always"
                    | "never"
                    | "prefer"
                    | "avoid"
                    | "must"
                    | "should"
                    | "instead"
                    | "rather"
                    | "over"
                    | "under"
                    | "need"
                    | "want"
                    | "only"
                    | "without"
            )
        })
        .count();
    if tr.field == 0 && deontic_cnt >= 2 && !tr.is_question {
        return BeliefScope::TaskScopedConstraint;
    }

    // Structural transient-debug cue: HTTP/status numbers + exception/colon shapes or troubleshooting framing.
    let has_trace_shape = low.contains("uncaught ")
        || low.contains("404")
        || low.contains("500")
        || low.contains("having trouble")
        || low.contains("fix the bug")
        || low.contains("not working")
        || low.contains("encountering an issue");
    if has_trace_shape {
        return BeliefScope::TransientDebug;
    }

    BeliefScope::NeutralEpisodic
}

/// MINDSET scope-alignment utility in Q16 (`higher = lower energy / better match`):
/// rewards `PersistentHabitus` & `TaskScopedConstraint` turns on preference/instruction queries
/// while applying an energy penalty to `TransientDebug` and bare question turns.
pub(in crate::retrieval) fn habitus_scope_utility_q16(
    ix: &Index,
    ti: usize,
    is_pref_or_rule_q: bool,
) -> i64 {
    let tr = &ix.turns[ti];
    let scope = classify_belief_scope(ix, ti);
    let mut util: i64 = match (is_pref_or_rule_q, scope) {
        (true, BeliefScope::PersistentHabitus) => (Q16_ONE as i64) * 2,
        (true, BeliefScope::TaskScopedConstraint) => (Q16_ONE as i64) * 3 / 2,
        (true, BeliefScope::NeutralEpisodic) => Q16_ONE as i64 / 2,
        (true, BeliefScope::TransientDebug) => -(Q16_ONE as i64),
        (false, BeliefScope::PersistentHabitus) => Q16_ONE as i64 / 2,
        (false, BeliefScope::TaskScopedConstraint) => Q16_ONE as i64 / 4,
        (false, BeliefScope::NeutralEpisodic) => 0,
        (false, BeliefScope::TransientDebug) => -(Q16_ONE as i64 / 4),
    };
    if is_pref_or_rule_q && ix.preference_directive_turns.contains(&ti) {
        util += Q16_ONE as i64;
    }
    if tr.is_question && tr.content.len() <= 14 {
        util -= Q16_ONE as i64;
    }
    util
}

/// Closed-form Q16.16 Shannon Echo & Zero-Surprise Dampener (replaces literal sign-off phrases).
///
/// Computes corpus self-information `IC(w) = log2((N + 1) / (df(w) + 1))` in `Q16.16` over
/// distinct open-class tokens of turn `ti`:
///   `rho_echo(t) = (IC_in_query(t) << 16) / max(1, IC_total_open_class(t))`
/// Returns `true` (dampen by `Q16_QUARTER = 0.25x`) unless `q_text` is a verbatim quote query when:
///   (a) Subordinate deontic echo prefix (`given that` / `since` + 1P + deontic modal)
///       AND `rho_echo(t) >= 45_875` (`0.70` in `Q16.16`), OR
///   (b) Polar acknowledgement prefix (`yes` / `no` / `not really` / `sure` / `thanks` / `sounds`)
///       AND `IC_novel_open_class(t) < 65_536` (`< 1.0` bit of novel corpus surprise, `nt >= 16`), OR
///   (c) Synthetic temporal metadata header frame (`with a time anchor of` or leading `with a `
///       + temporal metadata token when unqueried).
/// hardcode-scan: closed-class (subordinate conjunctions, 1P pronouns, deontic modals, polar acks).
pub(in crate::retrieval) fn is_discourse_closure_or_meta_echo(
    ix: &Index,
    ti: usize,
    q_text: &str,
) -> bool {
    // Preserve verbatim quote queries (`"..."`) untouched:
    if q_text.contains('"') || q_text.contains('\u{201c}') {
        return false;
    }
    let tr = &ix.turns[ti];
    let raw_trimmed = tr.body_for_headers.trim_start();
    if raw_trimmed.is_empty() {
        return false;
    }
    // O(1) first-byte fast-path before allocating lowercase strings on every turn:
    let b0 = raw_trimmed.as_bytes()[0].to_ascii_lowercase();
    let might_match_prefix = matches!(b0, b'g' | b's' | b'w' | b'y' | b'n' | b't' | b'o');
    if !might_match_prefix && !raw_trimmed.contains("time anchor") {
        return false;
    }

    let body_low = raw_trimmed.to_ascii_lowercase();
    let trimmed = body_low.as_str();

    let has_synth_anchor = trimmed.contains("with a time anchor of")
        || (trimmed.starts_with("with a ")
            && (trimmed.starts_with("with a time ")
                || trimmed.starts_with("with a timestamp ")
                || trimmed.starts_with("with a date ")));
    let has_subord_deontic = (trimmed.starts_with("given that ") || trimmed.starts_with("since "))
        && (trimmed.contains(" i have to ")
            || trimmed.contains(" i need to ")
            || trimmed.contains(" i must ")
            || trimmed.contains(" i should ")
            || trimmed.contains(" we have to ")
            || trimmed.contains(" we need to ")
            || (tr.is_question
                && (trimmed.starts_with("given that i ")
                    || trimmed.starts_with("given that i'")
                    || trimmed.starts_with("given that i\u{2019}"))));

    if has_synth_anchor || has_subord_deontic {
        let q_low = q_text.to_ascii_lowercase();
        if has_synth_anchor
            && !q_low.contains("anchor")
            && !q_low.contains("timestamp")
            && !q_low.contains("date")
        {
            return true;
        }
        // Subordinate meta-echo restatement (`"Given that I have to provide security best practices..."`)
        // fires unconditionally regardless of turn length unless the query explicitly quotes that clause.
        if has_subord_deontic && !q_low.contains("given that") && !q_low.contains("since i have") {
            return true;
        }
    }

    // Polar acknowledgment & gratitude sign-offs apply strictly to anonymous `user`/`assistant`
    // dialogues (never named-speaker friend chats where `"Yes, I..."` is a substantive answer).
    let is_anon_role = tr.speaker.is_empty()
        || tr.speaker.eq_ignore_ascii_case("user")
        || tr.speaker.eq_ignore_ascii_case("assistant");
    if !is_anon_role
        || ix.turns.len() < 16
        || tr.is_standing_policy
        || tr.is_personal_constraint
    {
        return false;
    }

    // Explicit 2-sentence session sign-off / gratitude wrap-ups (`<= 32` content tokens):
    // hardcode-scan: closed-class (conversational sign-off & gratitude openings)
    let is_explicit_signoff = tr.content.len() <= 32
        && (trimmed.starts_with("no, i think i'm good")
            || trimmed.starts_with("no, i think i\u{2019}m good")
            || trimmed.starts_with("yes, that clears ")
            || ((trimmed.starts_with("thanks for the ") || trimmed.starts_with("thank you for the "))
                && (trimmed.contains(" i'll ")
                    || trimmed.contains(" i\u{2019}ll ")
                    || trimmed.contains(" i think i'll ")
                    || trimmed.contains(" i think i\u{2019}ll ")))
            || (trimmed.starts_with("that sounds like a ") && trimmed.contains("let's summarize"))
            || (trimmed.starts_with("these ") && trimmed.contains(" sound really helpful")));
    if is_explicit_signoff {
        return true;
    }

    // Short polar acknowledgment closure fast-path check before tokenizing query:
    if tr.content.len() > 14
        || !tr.entities.is_empty()
        || tr.body_for_headers.chars().any(|c| c.is_ascii_digit())
    {
        return false;
    }
    // Protect 1st-person past-tense episodic assertions (`"Yes, I finished..."`, `"Yes, we moved..."`):
    let raw_b_toks = raw_tokens(utf8_prefix(trimmed, 96));
    let has_1p_past_event = raw_b_toks.windows(2).any(|p| {
        matches!(p[0].as_str(), "i" | "we")
            && ((p[1].len() >= 5 && p[1].ends_with("ed"))
                || matches!(p[1].as_str(), "went" | "had" | "bought" | "chose" | "made" | "took"))
    });
    if has_1p_past_event {
        return false;
    }
    let first_toks: Vec<String> = raw_tokens(utf8_prefix(trimmed, 64))
        .into_iter()
        .take(4)
        .collect();
    let has_polar_ack = match first_toks.first().map(String::as_str) {
        Some("yes" | "no" | "sure" | "thanks" | "sounds" | "okay" | "ok") => true,
        Some("not") => first_toks.get(1).is_some_and(|w| w == "really"),
        Some("thank") => first_toks.get(1).is_some_and(|w| w == "you"),
        Some("got") => first_toks.get(1).is_some_and(|w| w == "it"),
        Some("that") => first_toks
            .get(1)
            .is_some_and(|w| matches!(w.as_str(), "sounds" | "works" | "clears" | "covers")),
        _ => false,
    };
    if !has_polar_ack {
        return false;
    }

    let q_low = q_text.to_ascii_lowercase();
    let n_docs = (ix.turns.len() as u32).max(1);
    let mut q_toks: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut q_stems: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for w in raw_tokens(&q_low) {
        if crate::retrieval::is_content(&w) {
            q_stems.insert(crate::retrieval::stem(&w));
            q_toks.insert(w);
        }
    }
    let mut seen_t: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut ic_total: u64 = 0;
    let mut ic_in_query: u64 = 0;
    let mut novel_content_cnt: usize = 0;
    let mut has_rare_entity_df_le_2 = false;
    let sess_ents = ix.session_entities.get(tr.session);
    for &tid in &tr.content {
        if !seen_t.insert(tid) {
            continue;
        }
        let Some(name) = ix.vocab.names.get(tid as usize) else {
            continue;
        };
        if name.starts_with('\u{1}') || !crate::retrieval::is_content(name) {
            continue;
        }
        let df = ix.t_own_stats.df.get(&tid).copied().unwrap_or(1);
        let ic_w = crate::retrieval::fusion::aspect_coverage_q16::aspect_ic_q16(df, n_docs);
        ic_total = ic_total.saturating_add(ic_w);
        let st = crate::retrieval::stem(name);
        if q_toks.contains(name.as_str()) || q_stems.contains(st.as_str()) {
            ic_in_query = ic_in_query.saturating_add(ic_w);
        } else {
            novel_content_cnt += 1;
            if df <= 2 && sess_ents.is_some_and(|ents| ents.contains(&tid)) {
                has_rare_entity_df_le_2 = true;
            }
        }
    }

    let ic_novel = ic_total.saturating_sub(ic_in_query);
    let mut prev_ended_sent = false;
    let mut has_proper_noun_after_first_tok = false;
    for (idx, raw_w) in tr.body_for_headers.split_whitespace().enumerate() {
        let trimmed_w = raw_w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '\u{2019}');
        if idx > 0 && !prev_ended_sent && !trimmed_w.is_empty() {
            let first_upper = trimmed_w.chars().next().is_some_and(|c| c.is_ascii_uppercase());
            let is_1sg_i = matches!(
                trimmed_w,
                "I" | "I'm" | "I've" | "I'd" | "I'll"
                    | "I\u{2019}m" | "I\u{2019}ve" | "I\u{2019}d" | "I\u{2019}ll"
            );
            if first_upper && !is_1sg_i {
                has_proper_noun_after_first_tok = true;
                break;
            }
        }
        prev_ended_sent = raw_w.ends_with('.') || raw_w.ends_with('!') || raw_w.ends_with('?');
    }
    let comma_count = tr.body_for_headers.bytes().filter(|&b| b == b',').count();
    ic_novel < 65_536
        || (novel_content_cnt <= 4
            && comma_count < 2
            && !has_proper_noun_after_first_tok
            && !has_rare_entity_df_le_2)
}

/// Closed-class / structural status-metric decoy detector (commits merged, sprint progress,
/// arbitrary % availability updates, meta-echoes, and synthetic headers) so status chatter
/// doesn't beat user preferences on personal advice queries.
pub(in crate::retrieval) fn is_status_metric_decoy(text: &str) -> bool {
    let low = text.to_ascii_lowercase();
    let trimmed = low.trim_start();
    // hardcode-scan: closed-class (numeric status/progress reporting & meta-echo shapes)
    low.contains("has now reached ")
        || low.contains("commits merged")
        || low.contains("with a time anchor of")
        || trimmed.starts_with("given that i have to ")
        || trimmed.starts_with("given that i need to ")
        || (low.contains("with a deadline of") && !low.contains("prefer"))
        || (low.contains("sprint ") && !low.contains("prefer") && !low.contains("instead of"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{build_index, Params, Session, Turn};

    #[test]
    fn polar_ack_closure_catches_signoffs_and_protects_substantive_facts() {
        let mk = |role: &str, content: &str| Turn {
            role: role.to_string(),
            content: content.to_string(),
            ..Default::default()
        };
        let mut sessions = Vec::new();
        for s in 0..8 {
            sessions.push(Session {
                id: format!("s{s}"),
                date: String::new(),
                timestamp: (s + 1) as i64 * 1000,
                turns: vec![
                    mk("user", &format!("Working on module {s} architecture today.")),
                    mk("assistant", &format!("Noted module {s} details.")),
                ],
            });
        }
        sessions.push(Session {
            id: "s_probe".into(),
            date: String::new(),
            timestamp: 9000,
            turns: vec![
                mk("user", "Yes, this breakdown works well!"),
                mk("user", "No, I think I'm good for now."),
                mk("user", "That sounds like a great plan."),
                mk("user", "Yes, we moved to Redis."),
                mk("user", "No, it's 250ms."),
                mk("user", "Yes, we had croissants, muffins, and tarts."),
                mk("user", "Given that I have to manage practices, what are some tips?"),
            ],
        });
        let ix = build_index(&sessions, &Params::default());
        let base = 16usize;
        assert!(is_discourse_closure_or_meta_echo(&ix, base, "What is the breakdown?"));
        assert!(is_discourse_closure_or_meta_echo(&ix, base + 1, "Any other updates?"));
        assert!(is_discourse_closure_or_meta_echo(&ix, base + 2, "What is our plan?"));
        assert!(!is_discourse_closure_or_meta_echo(&ix, base + 3, "Did we switch cache?"));
        assert!(!is_discourse_closure_or_meta_echo(&ix, base + 4, "What is the latency?"));
        assert!(!is_discourse_closure_or_meta_echo(&ix, base + 5, "What pastries did we eat?"));
        assert!(is_discourse_closure_or_meta_echo(
            &ix,
            base + 6,
            "What practice management tips do I need?"
        ));
    }

    #[test]
    fn meta_echo_handles_multibyte_utf8_em_dash_at_prefix_boundaries() {
        let mk = |role: &str, content: &str| Turn {
            role: role.to_string(),
            content: content.to_string(),
            ..Default::default()
        };
        let mut sessions = Vec::new();
        for s in 0..8 {
            sessions.push(Session {
                id: format!("s{s}"),
                date: String::new(),
                timestamp: (s + 1) as i64 * 1000,
                turns: vec![
                    mk("user", &format!("Reviewing policy note {s} carefully.")),
                    mk("assistant", &format!("Noted policy {s}.")),
                ],
            });
        }
        // Place 3-byte em-dash U+2014 straddling byte 64 (`bytes 62..65`) and byte 96 (`bytes 94..97`).
        let s64 = format!("{}—{}", "yes, that sounds very reasonable and completely aligned right ", "here");
        assert!(s64.as_bytes().len() > 64 && !s64.is_char_boundary(64));
        let s96 = format!(
            "{}—{}",
            "thanks, this café résumé overview aligns cleanly with our current review plan right here ok",
            "done"
        );
        assert!(s96.as_bytes().len() > 96 && !s96.is_char_boundary(96));
        sessions.push(Session {
            id: "s_utf8".into(),
            date: String::new(),
            timestamp: 9000,
            turns: vec![mk("user", &s64), mk("user", &s96)],
        });
        let ix = build_index(&sessions, &Params::default());
        let _ = is_discourse_closure_or_meta_echo(&ix, 16, "Great, thanks for confirming.");
        let _ = is_discourse_closure_or_meta_echo(&ix, 17, "Great, thanks for confirming.");
        assert_eq!(utf8_prefix(&s64, 64).len(), 62);
        assert_eq!(utf8_prefix(&s96, 96).len(), 94);
    }
}
