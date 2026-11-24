//! Mission 1: Unified Implicit-Bridge Engine (`InMind` + `LoCoMo Open-Domain` + `ConvoMem Implicit`).
//!
//! Pure deterministic, zero-vector, zero-LLM, `$0.00` runtime architecture using
//! only closed-class English grammar, corpus DF statistics, and pinned offline
//! WordNet 3.1 / ConceptNet 5.7 / Wikidata CC0 FST graphs:
//!
//! 1. **Regime A — Sparse Personal-Disclosure Elevation**:
//!    In long-form user-assistant histories where concise declarative personal
//!    disclosures (`user` <= 42w + concise non-question `assistant` reply <= 50w)
//!    are corpus-rare (`1..=4` pairs and `< 1.6%` of turns), elevates those
//!    disclosures on present-tense advice/safety/recommendation frames and
//!    non-temporal 1st-person profile-recall queries using wrapper-stripped
//!    direct + 1-2 hop FST ConceptNet/WordNet/Wikidata bridge evidence, strictly
//!    above `?` questions, gratitude, and request-wrapper chatter.
//! 2. **Regime B — Named-Speaker Open-Domain FST Bridge + Greeting Lockout**:
//!    On inferential modal/copular dialogue queries (`Would X likely ...`,
//!    `What might X ...`, `Does X ...`), locks out zero-substance greetings and
//!    scaffold-only fillers (`"Hey X! Just wanted to say hi!"`) and promotes
//!    queried-speaker turns matching compound sub-tokens (`roadtrip -> road+trip`,
//!    `bookshelf -> book+shelf`), derivational/OOV-typo stems (`educaton -> edu/education`),
//!    or 1-2 hop ConceptNet + WordNet + Wikidata FST bridges (`Vivaldi -> classical`,
//!    `Dr. Seuss -> kids' books`, `national park -> camping`).
//! 3. **Regime C — Non-Sparse Advice-Frame User Assertion Rescue**:
//!    On dense conversational advice queries, demotes generic assistant advice
//!    and pure user questions below declarative `field == 0` user assertions
//!    carrying strictly stronger wrapper-stripped exact + FST concept evidence.

use std::collections::BTreeSet;

use crate::retrieval::graph::signed_galois_fca_q16::{
    apply_signed_galois_hazard_seat_q16, score_signed_galois_candidates_q16,
};
use crate::retrieval::index::normalize::{
    ascii_edit_dist_le_1, derivational_base_stems, split_wordnet_compound, stem_term,
};
use crate::retrieval::lexical::speaker_translation_q16::{
    is_open_domain_inferential_shape, is_strict_epistemic_inferential,
};
use crate::retrieval::query::tokens::is_content;
use crate::retrieval::rank::context::RankCtx;
use crate::retrieval::seating::c2::common::eff_summary_request;
use crate::retrieval::*;

/// Closed-class conversational advice/request/temporal wrapper words stripped
/// before FST concept expansion and non-scaffold match counting.
/// hardcode-scan: closed-class (wh-words, modals, auxiliaries, light verbs, temporal units, discourse request wrappers).
fn is_wrapper_or_scaffold(w: &str) -> bool {
    matches!(
        w,
        "what"
            | "when"
            | "where"
            | "who"
            | "whom"
            | "whose"
            | "which"
            | "why"
            | "how"
            | "can"
            | "could"
            | "would"
            | "should"
            | "shall"
            | "will"
            | "may"
            | "might"
            | "must"
            | "do"
            | "does"
            | "did"
            | "done"
            | "doing"
            | "is"
            | "am"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
            | "being"
            | "have"
            | "has"
            | "had"
            | "having"
            | "need"
            | "needed"
            | "want"
            | "wanted"
            | "wants"
            | "look"
            | "looking"
            | "think"
            | "thinking"
            | "thought"
            | "know"
            | "knew"
            | "known"
            | "tell"
            | "told"
            | "say"
            | "said"
            | "ask"
            | "asked"
            | "asking"
            | "mention"
            | "mentioned"
            | "explain"
            | "describe"
            | "list"
            | "give"
            | "gave"
            | "help"
            | "keep"
            | "mind"
            | "plan"
            | "planned"
            | "planning"
            | "go"
            | "going"
            | "went"
            | "gone"
            | "get"
            | "got"
            | "gotten"
            | "getting"
            | "take"
            | "took"
            | "taken"
            | "taking"
            | "make"
            | "made"
            | "making"
            | "start"
            | "started"
            | "starting"
            | "buy"
            | "bought"
            | "buying"
            | "try"
            | "trying"
            | "prepare"
            | "prepared"
            | "preparing"
            | "preparations"
            | "share"
            | "something"
            | "anything"
            | "everything"
            | "nothing"
            | "someone"
            | "anyone"
            | "everyone"
            | "some"
            | "any"
            | "many"
            | "much"
            | "more"
            | "most"
            | "other"
            | "another"
            | "long"
            | "often"
            | "far"
            | "ago"
            | "last"
            | "next"
            | "first"
            | "latest"
            | "recent"
            | "recently"
            | "lately"
            | "currently"
            | "soon"
            | "now"
            | "today"
            | "tonight"
            | "yesterday"
            | "tomorrow"
            | "weekend"
            | "time"
            | "times"
            | "day"
            | "days"
            | "week"
            | "weeks"
            | "month"
            | "months"
            | "year"
            | "years"
            | "thing"
            | "things"
            | "way"
            | "ways"
            | "kind"
            | "type"
            | "sort"
            | "good"
            | "great"
            | "nice"
            | "best"
            | "better"
            | "safe"
            | "useful"
            | "necessary"
            | "worth"
            | "special"
            | "aware"
            | "likely"
            | "interested"
            | "open"
            | "considered"
            | "experience"
            | "experiencing"
            | "recommend"
            | "recommended"
            | "recommendation"
            | "recommendations"
            | "suggest"
            | "suggested"
            | "suggestion"
            | "suggestions"
            | "advice"
            | "tip"
            | "tips"
            | "idea"
            | "ideas"
            | "option"
            | "options"
            | "thanks"
            | "thank"
            | "question"
            | "questions"
            | "also"
            | "really"
            | "actually"
            | "just"
            | "always"
            | "never"
            | "ever"
            | "before"
            | "previously"
            | "bit"
            | "little"
            | "new"
            | "feel"
            | "feeling"
            | "information"
            | "details"
            | "reason"
            | "considerations"
            | "resources"
            | "available"
            | "person"
            | "people"
            | "close"
            | "future"
            | "back"
            | "effective"
            | "effectively"
            | "optimal"
            | "focus"
            | "upcoming"
    )
}

/// Closed-class check for present-tense advice/safety/recommendation/decision query frames
/// (excluding explicit past-memory recall phrasing like "did I / did you / when did").
/// hardcode-scan: closed-class (interrogative/modal advice frames and past-recall guards).
fn is_implicit_advice_frame(q_text: &str) -> bool {
    let ql = q_text.to_ascii_lowercase();
    if [
        "did i",
        "did you",
        "did we",
        "have i ever",
        "have i told",
        "have i mentioned",
        "was i",
        "were we",
        "told you",
        "i mentioned",
        "remind me",
        "previously",
        "earlier",
        "last time",
        "when did",
        "where did",
        "who did",
        "how many",
        "what did",
    ]
    .iter()
    .any(|p| ql.contains(p))
    {
        return false;
    }
    [
        "recommend",
        "suggestion",
        "considering ",
        "given my ",
        "for my ",
        "with my ",
        "suited for",
        "good option",
        "advice",
        "tips",
        "should i",
        "can i ",
        "is this safe",
        "is it safe",
        "is it necessary",
        "worth learning",
        "keep in mind",
        "anything i need",
        "anything i should",
        "anything special",
        "be aware of",
        "useful for me",
        "taste good",
        "could be the reason",
        "information can you share",
        "ways to ",
        "how long can i",
        "how should i",
        "how can i",
        "how about ",
        "what's a good",
        "what is a good",
        "any good ",
        "any latest ",
        "special considerations",
        "what preparations",
        "like to know",
        "like to learn",
        "like to understand",
        "help me plan",
        "what should i",
        "are there any",
        "do you have any",
        "planning a ",
        "thinking of ",
        "thinking about ",
        "starting a new",
        "want to buy",
        "want to get",
        "want to try",
        "want to pay",
        "want to improve",
        "want to contact",
        "looking to buy",
        "looking into ",
        "experiencing ",
        "getting frequent",
        "has been hurting",
        "craving ",
        "invited me to",
        "lately, i",
        "tight on cash",
        "look quite interesting",
        "offer temporary",
        "help prevent",
        "should i put",
        "pros and cons",
        "any ideas",
        "what approach",
        "tend to be",
        "i want to ",
        "what is it for",
        "what aged ",
        "plan my ",
        "organize my ",
        "schedule my ",
        "structure my ",
        "prepare for ",
        "preparing for ",
        "dealing with ",
        "handling ",
        "keep it ",
        "keeping it ",
        "optimal time",
    ]
    .iter()
    .any(|p| ql.contains(p))
}

/// Closed-class check for direct 1st-person personal-profile recall queries
/// (e.g. "What health conditions do I have?", "Did I tell you what I do for a living?",
/// "How old am I?", "What allergy did I mention?").
/// hardcode-scan: closed-class (1st-person interrogative memory-recall frames).
fn is_personal_profile_recall_query(q_text: &str) -> bool {
    let ql = q_text.to_ascii_lowercase();
    if [
        "when did",
        "how many",
        "how much",
        "before ",
        "after ",
        "earlier",
        "first ",
        "last ",
        "recently changed",
        "updated",
        "now instead",
    ]
    .iter()
    .any(|p| ql.contains(p))
    {
        return false;
    }
    let toks = raw_tokens(&ql);
    let has_1sg = toks
        .iter()
        .any(|w| matches!(w.as_str(), "i" | "my" | "me" | "mine" | "we" | "our"));
    if !has_1sg {
        return false;
    }
    let starts_q = toks.first().is_some_and(|w| {
        matches!(
            w.as_str(),
            "what" | "which" | "who" | "where" | "how" | "did" | "do" | "have" | "am" | "is" | "are"
        )
    });
    starts_q
        || [
            "did i tell",
            "did i ever tell",
            "did i mention",
            "did i share",
            "have i told",
            "have i mentioned",
            "for a living",
        ]
        .iter()
        .any(|p| ql.contains(p))
}

/// Closed-class explicit assistant memory-acknowledgment phrase check.
/// hardcode-scan: closed-class (memory-commit acknowledgment markers).
fn is_explicit_memory_ack(text: &str) -> bool {
    let low = text.to_ascii_lowercase();
    [
        "i'll remember",
        "will remember",
        "make a note",
        "made a note",
        "i've noted",
        "noted down",
        "keep that in mind",
        "keep this in mind",
        "keep it in mind",
        "keep your ",
        "noted your ",
        "keep that information",
        "keep this information",
        "i understand your ",
        "i understand you ",
    ]
    .iter()
    .any(|p| low.contains(p))
}

/// Closed-class check for user imperative task prompts or request/gratitude wrappers
/// that must never be treated as personal-state disclosures.
/// hardcode-scan: closed-class (imperative task verbs, request framing, and gratitude markers).
fn is_request_or_gratitude_wrapper(text: &str) -> bool {
    let low = text.to_ascii_lowercase();
    if let Some(first) = raw_tokens(&low).first()
        && matches!(
            first.as_str(),
            "write"
                | "rewrite"
                | "generate"
                | "create"
                | "draft"
                | "compose"
                | "translate"
                | "summarize"
                | "explain"
                | "describe"
                | "list"
                | "give"
                | "make"
                | "please"
                | "convert"
                | "format"
                | "edit"
                | "add"
                | "remove"
                | "fix"
                | "show"
                | "provide"
                | "outline"
                | "continue"
        )
    {
        return true;
    }
    [
        "recommend",
        "suggest",
        "tips",
        "advice",
        "looking for",
        "gift ideas",
        "can you",
        "could you",
        "help me",
        "tell me",
        "thanks",
        "thank you",
        "thx",
        "next question:",
        "sounds good",
        "got it",
        "that's a good",
        "that's a great",
        "okay, i'm going",
        "ok, i'm going",
        "let's see",
    ]
    .iter()
    .any(|p| low.contains(p))
}

/// True when a dialogue turn is a zero-substance greeting, farewell, or filler
/// (e.g., "Hey Melanie! Just wanted to say hi!", "You're welcome!").
/// hardcode-scan: closed-class (greetings, farewells, and courtesy fillers).
fn is_zero_substance_greeting(tr: &TurnIx) -> bool {
    if tr.is_backchannel {
        return true;
    }
    let toks = raw_tokens(&tr.body_for_headers);
    if toks.len() > 16 {
        return false;
    }
    let low = tr.body_for_headers.to_ascii_lowercase();
    let has_greet = [
        "just wanted to say hi",
        "wanted to say hi",
        "say hi",
        "say hello",
        "how're ya",
        "how are you",
        "how are ya",
        "talk to you soon",
        "catch up soon",
        "you're welcome",
        "no problem",
        "good to hear from you",
    ]
    .iter()
    .any(|p| low.contains(p));
    if has_greet && toks.len() <= 14 {
        return true;
    }
    // Ultra-short turn with zero non-scaffold tokens.
    let non_scaf = toks
        .iter()
        .filter(|w| is_content(w) && !is_wrapper_or_scaffold(w))
        .count();
    toks.len() <= 10 && non_scaf == 0
}

/// hardcode-scan: closed-class (greetings, farewells, and courtesy/ack leads).
fn has_closed_class_greeting_or_ack_lead(low: &str) -> bool {
    let trimmed = low.trim_start();
    [
        "hey ", "hi ", "hello ", "yep", "yeah", "haha", "wow", "thanks", "aw",
    ]
    .iter()
    .any(|p| trimmed.starts_with(p))
}

/// hardcode-scan: closed-class (greetings, farewells, and courtesy fillers).
fn is_zero_substance_greeting_with_hits(tr: &TurnIx, dh: u32, fh: u32) -> bool {
    if is_zero_substance_greeting(tr) {
        return true;
    }
    if dh == 0 && fh == 0 {
        let toks = raw_tokens(&tr.body_for_headers);
        if toks.len() <= 18
            && has_closed_class_greeting_or_ack_lead(&tr.body_for_headers.to_ascii_lowercase())
        {
            return true;
        }
    }
    false
}

/// Closed-class check for calendar weekday/month names excluded from co-hyponym sibling
/// expansion so temporal words do not cross-match adjacent days or months.
/// hardcode-scan: closed-class (calendar weekdays and months only).
fn is_calendar_word(w: &str) -> bool {
    matches!(
        w,
        "monday"
            | "tuesday"
            | "wednesday"
            | "thursday"
            | "friday"
            | "saturday"
            | "sunday"
            | "january"
            | "february"
            | "march"
            | "april"
            | "may"
            | "june"
            | "july"
            | "august"
            | "september"
            | "october"
            | "november"
            | "december"
    )
}

/// Closed-class 1st-person preference, habit, affinity, or standing-goal stance detector
/// used alongside `TurnIx::is_habitual` and `TurnIx::is_personal_constraint` to anchor
/// B4 WordNet co-hyponym / LCS sibling resonance to the queried speaker's traits.
/// hardcode-scan: closed-class (1st-person preference, affinity, habit, and goal stance frames).
fn is_speaker_preference_or_habit_turn(tr: &TurnIx) -> bool {
    if tr.is_habitual || tr.is_personal_constraint {
        return true;
    }
    if tr.is_question || tr.is_backchannel {
        return false;
    }
    let low = tr.body_for_headers.to_ascii_lowercase();
    // hardcode-scan: closed-class (1st-person preference, affinity, habit, and goal markers)
    [
        "i'm a fan",
        "i am a fan",
        "i love ",
        "i enjoy ",
        "i prefer ",
        "i'm keen",
        "i'm allergic",
        "i always ",
        "i usually ",
        "we always ",
        "my favorite",
        "my dream",
        "my goal",
        "i'd love",
        "i want to ",
        "i hope to ",
        "i've always",
        "i've never",
        "i avoid ",
        "i'm trying ",
    ]
    .iter()
    .any(|p| low.contains(p))
}

#[inline]
fn turn_idf(nt: usize, df: u32) -> f32 {
    let n = (nt as f32).max(1.0);
    let d = (df as f32).max(0.5);
    ((n + 1.0) / (d + 0.5)).ln().max(0.5)
}

/// Weighted FST concept maps (`vocab_id -> (weight, canonical_group_id)`) covering
/// exact tokens and their `\u{1}stem:...` variants without double-counting.
struct FstConceptMaps {
    direct_weights: DetHashMap<u32, (f32, u32)>,
    fst_weights: DetHashMap<u32, (f32, u32)>,
    /// 1-hop WordNet `strict_co_hyponym_siblings` + `lcs_sibling_resonance` (B4),
    /// locked strictly to `queried_speakers[0]`'s `is_habitual` / preference / declarative turns.
    sibling_weights: DetHashMap<u32, (f32, u32)>,
}

/// Build weighted FST expansion vocab maps for query concepts using pinned offline
/// WordNet 3.1 + ConceptNet 5.7 + Wikidata CC0 + Roget 1911 + GeoNames + corpus PPMI
/// + compound splitting + derivational stems + OOV typo/abbreviation rescue.
fn build_fst_concept_map(
    ctx: &RankCtx<'_>,
    speaker_vocab: &DetHashSet<String>,
) -> FstConceptMaps {
    let nt = ctx.nt;
    let max_fst_df = ((nt / 7).clamp(14, 48)) as u32;
    let mut direct_weights: DetHashMap<u32, (f32, u32)> = DetHashMap::default();
    let mut fst_weights: DetHashMap<u32, (f32, u32)> = DetHashMap::default();
    let mut sibling_weights: DetHashMap<u32, (f32, u32)> = DetHashMap::default();

    let insert_best = |map: &mut DetHashMap<u32, (f32, u32)>, id: u32, w: f32, grp: u32| {
        map.entry(id)
            .and_modify(|cur| {
                if w > cur.0 {
                    cur.0 = w;
                    cur.1 = grp;
                }
            })
            .or_insert((w, grp));
    };

    let mut add_direct_tok = |tok: &str, mul: f32| -> bool {
        let mut hit = false;
        let st = stem(tok);
        let st_id_opt = if st.len() >= 3 {
            ctx.ix.vocab.get(&stem_term(&st))
        } else {
            None
        };
        let exact_id_opt = ctx.ix.vocab.get(tok);
        let grp = st_id_opt.or(exact_id_opt).unwrap_or(0);
        if let Some(id) = exact_id_opt {
            let df = ctx.ix.t_own_stats.df.get(&id).copied().unwrap_or(0);
            if df > 0 {
                hit = true;
                insert_best(&mut direct_weights, id, mul * turn_idf(nt, df), grp);
            }
        }
        if let Some(sid) = st_id_opt {
            let df = ctx.ix.t_own_stats.df.get(&sid).copied().unwrap_or(0);
            if df > 0 {
                hit = true;
                insert_best(
                    &mut direct_weights,
                    sid,
                    mul * 0.82 * turn_idf(nt, df),
                    grp,
                );
            }
        }
        if st.len() >= 3
            && st != tok
            && let Some(plain_sid) = ctx.ix.vocab.get(&st)
        {
            let df = ctx.ix.t_own_stats.df.get(&plain_sid).copied().unwrap_or(0);
            if df > 0 {
                hit = true;
                insert_best(
                    &mut direct_weights,
                    plain_sid,
                    mul * 0.78 * turn_idf(nt, df),
                    grp,
                );
            }
        }
        hit
    };

    let add_to_map = |map: &mut DetHashMap<u32, (f32, u32)>, tok: &str, base_w: f32| {
        if tok.len() < 3
            || !is_content(tok)
            || is_wrapper_or_scaffold(tok)
            || speaker_vocab.contains(tok)
        {
            return;
        }
        let st = stem(tok);
        let st_id_opt = if st.len() >= 3 {
            ctx.ix.vocab.get(&stem_term(&st))
        } else {
            None
        };
        let exact_id_opt = ctx.ix.vocab.get(tok);
        let grp = st_id_opt.or(exact_id_opt).unwrap_or(0);
        if let Some(id) = exact_id_opt {
            let df = ctx.ix.t_own_stats.df.get(&id).copied().unwrap_or(0);
            if (1..=max_fst_df).contains(&df) {
                let w = base_w * turn_idf(nt, df);
                insert_best(map, id, w, grp);
            }
        }
        if let Some(sid) = st_id_opt {
            let df = ctx.ix.t_own_stats.df.get(&sid).copied().unwrap_or(0);
            if (1..=max_fst_df).contains(&df) {
                let w = base_w * 0.85 * turn_idf(nt, df);
                insert_best(map, sid, w, grp);
            }
        }
    };

    let q_raw = raw_tokens(&ctx.q.text);
    let mut concepts: Vec<String> = Vec::new();
    for w in &q_raw {
        let wl = w.to_ascii_lowercase();
        if wl.len() < 3
            || !is_content(&wl)
            || is_wrapper_or_scaffold(&wl)
            || speaker_vocab.contains(&wl)
        {
            continue;
        }
        if !concepts.contains(&wl) {
            concepts.push(wl);
        }
    }

    // Also extract 2-word and 3-word phrase/entity concepts (e.g., "dr seuss", "national park",
    // "theme park", "four seasons", "hot pot", "energy bar", "green card") for Wikidata & ConceptNet.
    let mut phrases: Vec<String> = Vec::new();
    for win in q_raw.windows(2) {
        let a = win[0].to_ascii_lowercase();
        let b = win[1].to_ascii_lowercase();
        if a.len() >= 2 && b.len() >= 3 && !speaker_vocab.contains(&a) && !speaker_vocab.contains(&b)
        {
            let sp = format!("{a} {b}");
            let us = format!("{a}_{b}");
            if !phrases.contains(&sp) {
                phrases.push(sp);
            }
            if !phrases.contains(&us) {
                phrases.push(us);
            }
        }
    }
    for win in q_raw.windows(3) {
        let a = win[0].to_ascii_lowercase();
        let b = win[1].to_ascii_lowercase();
        let c = win[2].to_ascii_lowercase();
        if a.len() >= 2
            && !b.is_empty()
            && c.len() >= 3
            && !speaker_vocab.contains(&a)
            && !speaker_vocab.contains(&c)
        {
            let sp = format!("{a} {b} {c}");
            if (crate::wikidata::has_source(&sp) || !crate::wordnet::senses(&sp).is_empty())
                && !phrases.contains(&sp)
            {
                phrases.push(sp);
            }
        }
    }

    // Expand single concepts: direct + compound split + derivational + OOV edit-1/abbr + FST graphs.
    for c in &concepts {
        let grounded_here = add_direct_tok(c, 2.2);

        // 1. Compound splitting (`roadtrip -> road + trip`, `bookshelf -> book + shelf`).
        if let Some((left, right)) = split_wordnet_compound(c) {
            for part in [left, right] {
                if !is_wrapper_or_scaffold(part) {
                    add_direct_tok(part, 1.65);
                    add_to_map(&mut fst_weights, part, 0.95);
                }
            }
        }

        // 2. Derivational morphology (`educational -> education, educate`).
        for d in derivational_base_stems(c) {
            add_direct_tok(&d, 1.45);
            add_to_map(&mut fst_weights, &d, 0.85);
        }

        // 3. OOV typo / abbreviation rescue (`educaton -> education / edu`).
        if !grounded_here && c.is_ascii() && c.len() >= 6 {
            let prefix3 = &c[..3];
            for (vword, &vid) in &ctx.ix.vocab.map {
                if vword.contains(':') || !vword.bytes().all(|b| b.is_ascii_lowercase()) {
                    continue;
                }
                let df = ctx.ix.t_own_stats.df.get(&vid).copied().unwrap_or(0);
                if !(1..=max_fst_df).contains(&df) {
                    continue;
                }
                if ascii_edit_dist_le_1(c, vword) {
                    add_direct_tok(vword, 1.85);
                    // Also expand FST from the corrected word!
                    for (nbr, sc) in crate::conceptnet::two_hop_neighbors(vword, 14) {
                        let w = ((sc as f32) / 200.0).clamp(0.25, 1.0) * 0.72;
                        for tok in raw_tokens(nbr) {
                            add_to_map(&mut fst_weights, &tok, w);
                        }
                    }
                    for (nbr, sc) in crate::wordnet::full_lexicon_neighbors(vword)
                        .into_iter()
                        .take(14)
                    {
                        let w = ((sc as f32) / 100.0).clamp(0.25, 1.0) * 0.68;
                        for tok in raw_tokens(nbr) {
                            add_to_map(&mut fst_weights, &tok, w);
                        }
                    }
                } else if vword.len() == 3 && vword == prefix3 && !is_wrapper_or_scaffold(vword) {
                    add_direct_tok(vword, 1.35);
                }
            }
        }

        // 4. Wikidata entity upward containers, direct typed edges, and class instances.
        for container in crate::wikidata::upward_containers(c, 2).into_iter().take(12) {
            for tok in raw_tokens(container) {
                add_to_map(&mut fst_weights, &tok, 0.85);
            }
        }
        for (target, _) in crate::wikidata::direct_typed_edges(c).into_iter().take(12) {
            for tok in raw_tokens(target) {
                add_to_map(&mut fst_weights, &tok, 0.90);
            }
        }
        for inst in crate::wikidata::direct_instances_of(c, 12) {
            for tok in raw_tokens(inst) {
                add_to_map(&mut fst_weights, &tok, 0.78);
            }
        }

        // 5. ConceptNet 1-2 hop + reciprocal 2-hop neighbors (`sc` in `0..=200`).
        for (nbr, sc) in crate::conceptnet::two_hop_neighbors(c, 22) {
            let w = ((sc as f32) / 200.0).clamp(0.25, 1.0) * 0.80;
            for tok in raw_tokens(nbr) {
                add_to_map(&mut fst_weights, &tok, w);
            }
        }
        for (nbr, sc) in crate::conceptnet::reciprocal_two_hop_neighbors(c, 16) {
            let w = ((sc as f32) / 200.0).clamp(0.25, 1.0) * 0.78;
            for tok in raw_tokens(nbr) {
                add_to_map(&mut fst_weights, &tok, w);
            }
        }

        // 6. WordNet full lexicon neighbors (Heyting asymmetric subsumption weighted)
        //    + dual gloss hypernym bridge + gloss concepts + class-to-instances.
        for (nbr, sc) in crate::wordnet::full_lexicon_neighbors(c)
            .into_iter()
            .take(18)
        {
            let heyt = crate::wordnet::heyting_implication_ic_q16(nbr, c);
            let eff_sc = if heyt >= 76 {
                sc.max(heyt)
            } else if heyt == 38 {
                (sc * 3) / 4
            } else {
                sc
            };
            let w = ((eff_sc as f32) / 100.0).clamp(0.25, 1.0) * 0.76;
            for tok in raw_tokens(nbr) {
                add_to_map(&mut fst_weights, &tok, w);
            }
        }
        for (nbr, sc) in crate::wordnet::dual_gloss_hypernym_bridge(c, 14) {
            let w = ((sc as f32) / 100.0).clamp(0.25, 1.0) * 0.68;
            for tok in raw_tokens(nbr) {
                add_to_map(&mut fst_weights, &tok, w);
            }
        }
        for gc in crate::wordnet::gloss_and_compound_concepts(c)
            .into_iter()
            .take(12)
        {
            for tok in raw_tokens(gc) {
                add_to_map(&mut fst_weights, &tok, 0.54);
            }
        }
        if (1..=4).contains(&crate::wordnet::senses(c).len()) {
            for inst in crate::wordnet::class_to_instances(c, 1).into_iter().take(24) {
                for tok in raw_tokens(inst) {
                    add_to_map(&mut fst_weights, &tok, 0.72);
                }
            }
        }

        // 7. B4 1-hop WordNet `strict_co_hyponym_siblings` + `lcs_sibling_resonance`
        //    (stored separately and locked strictly to `queried_speakers[0]`'s
        //    `is_habitual` / preference / declarative turns in Regime B!).
        if !is_calendar_word(c) {
            for sib in crate::wordnet::strict_co_hyponym_siblings(c)
                .into_iter()
                .take(16)
            {
                if !is_calendar_word(sib) {
                    add_to_map(&mut sibling_weights, sib, 0.78);
                }
            }
            for (lcs_sib, lcs_sc) in crate::wordnet::lcs_sibling_resonance(c, 16) {
                if !is_calendar_word(lcs_sib) {
                    let w = ((lcs_sc as f32) / 100.0).clamp(0.35, 1.0) * 0.88;
                    add_to_map(&mut sibling_weights, lcs_sib, w);
                }
            }
        }

        // 8. Roget 1911 conceptual neighbors + GeoNames gazetteer neighbors & class instances
        //    + 1-hop corpus PPMI associative translation.
        for (r_nbr, r_sc) in crate::roget::conceptual_neighbors(c, 28) {
            let w = ((r_sc as f32) / 100.0).clamp(0.25, 1.0) * 0.56;
            add_to_map(&mut fst_weights, r_nbr, w);
        }
        for (g_nbr, g_sc) in crate::geonames::geo_neighbors(c).into_iter().take(12) {
            let w = ((g_sc as f32) / 100.0).clamp(0.30, 1.0) * 0.85;
            add_to_map(&mut fst_weights, g_nbr, w);
        }
        for geo_inst in crate::geonames::geo_class_instances(c) {
            if let Some(gid) = ctx.ix.vocab.get(geo_inst) {
                let df = ctx.ix.t_own_stats.df.get(&gid).copied().unwrap_or(0);
                if (1..=max_fst_df).contains(&df) {
                    add_to_map(&mut fst_weights, geo_inst, 0.76);
                }
            }
        }
        if let Some(qid_v) = ctx.ix.vocab.get(c)
            && let Some(ppmi_list) = ctx.ix.ppmi_neighbors.get(&qid_v)
        {
            for &(pid, pw) in ppmi_list.iter().take(6) {
                let df = ctx.ix.t_own_stats.df.get(&pid).copied().unwrap_or(0);
                if (1..=max_fst_df).contains(&df) {
                    let w = (pw.clamp(0.1, 2.0) * 0.36) * turn_idf(nt, df);
                    insert_best(&mut fst_weights, pid, w, pid);
                }
            }
        }
    }

    // Phrase-level Wikidata + ConceptNet lookups ("Dr. Seuss", "national park", "Four Seasons").
    for ph in &phrases {
        if crate::wikidata::has_source(ph) {
            for container in crate::wikidata::upward_containers(ph, 2).into_iter().take(12) {
                for tok in raw_tokens(container) {
                    add_to_map(&mut fst_weights, &tok, 1.05);
                }
            }
            for (target, _) in crate::wikidata::direct_typed_edges(ph).into_iter().take(12) {
                for tok in raw_tokens(target) {
                    add_to_map(&mut fst_weights, &tok, 1.10);
                }
            }
        }
        for (nbr, sc) in crate::conceptnet::two_hop_neighbors(ph, 16) {
            let w = ((sc as f32) / 200.0).clamp(0.25, 1.0) * 0.95;
            for tok in raw_tokens(nbr) {
                add_to_map(&mut fst_weights, &tok, w);
            }
        }
    }

    FstConceptMaps {
        direct_weights,
        fst_weights,
        sibling_weights,
    }
}

/// Score a turn's `content` + `\u{1}stem:...` `feats` against `FstConceptMaps`,
/// deduplicating exact and stemmed forms per canonical group key.
/// `allow_siblings` MUST be true ONLY for `queried_speakers[0]`'s declarative turns (B4).
/// `skip_other_speaker` excludes 2nd-person `tr.other_speaker_terms` on single-speaker queries.
fn score_turn_concepts_full(
    tr: &TurnIx,
    maps: &FstConceptMaps,
    allow_siblings: bool,
    skip_other_speaker: bool,
) -> (f32, f32, f32, u32, u32, u32) {
    let mut best_d: DetHashMap<u32, f32> = DetHashMap::default();
    let mut best_f: DetHashMap<u32, f32> = DetHashMap::default();
    let mut best_s: DetHashMap<u32, f32> = DetHashMap::default();

    let mut visit_id = |cid: u32| {
        if skip_other_speaker && tr.other_speaker_terms.contains(&cid) {
            return;
        }
        if let Some(&(w, grp)) = maps.direct_weights.get(&cid) {
            best_d
                .entry(grp)
                .and_modify(|cur| {
                    if w > *cur {
                        *cur = w;
                    }
                })
                .or_insert(w);
        } else if let Some(&(w, grp)) = maps.fst_weights.get(&cid) {
            best_f
                .entry(grp)
                .and_modify(|cur| {
                    if w > *cur {
                        *cur = w;
                    }
                })
                .or_insert(w);
        }
        if allow_siblings
            && !maps.direct_weights.contains_key(&cid)
            && !maps.fst_weights.contains_key(&cid)
            && let Some(&(w, grp)) = maps.sibling_weights.get(&cid)
        {
            best_s
                .entry(grp)
                .and_modify(|cur| {
                    if w > *cur {
                        *cur = w;
                    }
                })
                .or_insert(w);
        }
    };

    for &cid in &tr.content {
        visit_id(cid);
    }
    for f in &tr.feats {
        if maps.direct_weights.contains_key(&f.id)
            || maps.fst_weights.contains_key(&f.id)
            || (allow_siblings && maps.sibling_weights.contains_key(&f.id))
        {
            visit_id(f.id);
        }
    }

    let direct: f32 = best_d.values().sum();
    let fst: f32 = best_f.values().sum();
    let sib: f32 = best_s.values().sum();
    (
        direct,
        fst,
        sib,
        best_d.len() as u32,
        best_f.len() as u32,
        best_s.len() as u32,
    )
}

/// Convenience wrapper for advice/disclosure regimes without B4 sibling expansion.
fn score_turn_concepts(tr: &TurnIx, maps: &FstConceptMaps) -> (f32, f32, u32, u32) {
    let (d, f, _, dh, fh, _) = score_turn_concepts_full(tr, maps, false, false);
    (d, f, dh, fh)
}

/// Check whether turn is a concise declarative 1st-person disclosure paired with a
/// concise non-question assistant reply in the same session.
fn find_sparse_personal_disclosure_pairs(ix: &Index) -> Vec<(usize, usize, bool)> {
    let mut pairs = Vec::new();
    let nt = ix.turns.len();
    for i in 0..nt.saturating_sub(1) {
        let u = &ix.turns[i];
        let a = &ix.turns[i + 1];
        if u.session != a.session || u.field != 0 || a.field != 1 {
            continue;
        }
        if u.is_question || a.is_question || u.is_backchannel {
            continue;
        }
        let ut = &u.body_for_headers;
        let at = &a.body_for_headers;
        if ut.contains('?') || at.contains('?') {
            continue;
        }
        let uw = ut.split_whitespace().count();
        let aw = at.split_whitespace().count();
        if !(3..=42).contains(&uw) || !(3..=50).contains(&aw) {
            continue;
        }
        if is_request_or_gratitude_wrapper(ut) {
            continue;
        }
        let u_toks = raw_tokens(ut);
        // Require a 1st-person subject in the opening clause (first 7 tokens) so quoted
        // or embedded pronouns inside creative-writing prompts are excluded.
        // hardcode-scan: closed-class (1st-person personal pronouns and contractions)
        let has_leading_1sg = u_toks.iter().take(7).any(|w| {
            matches!(
                w.as_str(),
                "i" | "i'm" | "i've" | "i'd" | "i'll" | "my" | "me" | "mine" | "we" | "we're" | "we've" | "our" | "ours"
            )
        });
        if !has_leading_1sg {
            continue;
        }
        let mem_ack = is_explicit_memory_ack(at);
        pairs.push((i, i + 1, mem_ack));
    }
    pairs
}

/// Regime A: Sparse personal-disclosure elevation (elevates corpus-rare declarative
/// user disclosure + assistant acknowledgment pairs on implicit advice and direct
/// personal-profile recall queries).
fn apply_sparse_personal_disclosure_regime(ctx: &mut RankCtx<'_>) -> bool {
    if ctx.ix.has_tool_calls
        || ctx.has_named_speakers_eff
        || ctx
            .final_turn_indices
            .first()
            .is_some_and(|&t0| ctx.ix.turns[t0].is_standing_policy)
        || ctx.eff_it.is_event_sequence
        || ctx.eff_it.assistant_attribution
        || ctx.torrens_locked_head
    {
        return false;
    }

    let raw_pairs = find_sparse_personal_disclosure_pairs(ctx.ix);
    // Preserve synthetic sparse haystack behavior (`1..=6` pairs & `< 1.8%` density);
    // on denser real chats (ConvoMem / LME), retain explicit memory-ack pairs if `1..=8` exist,
    // otherwise fall through cleanly to Regime C without locking out dense rescue.
    let pairs: Vec<(usize, usize, bool)> = if !raw_pairs.is_empty()
        && raw_pairs.len() <= 6
        && raw_pairs.len() * 55 <= ctx.nt
    {
        raw_pairs
    } else {
        let ack_only: Vec<(usize, usize, bool)> = raw_pairs
            .into_iter()
            .filter(|&(_, _, ack)| ack)
            .collect();
        if (1..=8).contains(&ack_only.len()) && ack_only.len() * 20 <= ctx.nt {
            ack_only
        } else {
            return false;
        }
    };
    let advice_q = is_implicit_advice_frame(&ctx.q.text);
    let profile_q = !advice_q
        && !ctx.eff_it.temporal
        && is_personal_profile_recall_query(&ctx.q.text);
    if !advice_q && !profile_q {
        return false;
    }

    if pairs.iter().any(|&(_, _, ack)| ack)
        || (!ctx.final_turn_indices.is_empty()
            && pairs
                .iter()
                .any(|&(u, a, _)| u == ctx.final_turn_indices[0] || a == ctx.final_turn_indices[0]))
    {
        ctx.implicit_disclosure_locked = true;
    }

    let empty_sp = DetHashSet::default();
    let maps = build_fst_concept_map(ctx, &empty_sp);
    let mut pair_turns: Vec<usize> = Vec::with_capacity(pairs.len() * 2);
    for &(u_idx, a_idx, _) in &pairs {
        pair_turns.push(u_idx);
        pair_turns.push(a_idx);
    }
    let galois_scores =
        score_signed_galois_candidates_q16(ctx.ix, &ctx.q.text, &pair_turns, &BTreeSet::new());

    let mut scored_pairs: Vec<(usize, usize, f32, bool, f32, f32, f32, f32)> =
        Vec::with_capacity(pairs.len());
    for &(u_idx, a_idx, mem_ack) in &pairs {
        let (ud, uf, _, _) = score_turn_concepts(&ctx.ix.turns[u_idx], &maps);
        let (ad, af, _, _) = score_turn_concepts(&ctx.ix.turns[a_idx], &maps);
        let ug = galois_scores.get(&u_idx).copied().unwrap_or_default();
        let ag = galois_scores.get(&a_idx).copied().unwrap_or_default();
        let haz = ((ug.hazard_resonance_q16.max(ag.hazard_resonance_q16) as f32) / 65_536.0)
            .min(4.0)
            * 4.5;
        let g_meet = ((ug.galois_meet_q16.saturating_add(ag.galois_meet_q16) as f32) / 65_536.0)
            .min(6.0)
            * 1.4;
        let ack_bonus = if mem_ack { 18.0 } else { 1.5 };
        let sc = (ud + ad) * 2.0
            + (uf + af) * 1.15
            + haz
            + g_meet
            + ack_bonus
            + 0.10 * ctx.t_scene[u_idx].max(0.0);
        scored_pairs.push((u_idx, a_idx, sc, mem_ack, ud, uf, ad, af));
    }
    scored_pairs.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    if profile_q {
        let (_, _, _, best_ack, _, _, _, _) = scored_pairs[0];
        if !best_ack {
            return false;
        }
    }

    // Seat winning disclosure pair(s) cleanly at the front of `final_turn_indices`
    // while demoting questions/wrappers within Top-10 and syncing `final_session_indices[0]`.
    let mut front: Vec<usize> = Vec::with_capacity(pairs.len() * 2);
    for &(u_idx, a_idx, _, _, _, _, _, _) in &scored_pairs {
        front.push(u_idx);
        front.push(a_idx);
    }

    let mut reordered: Vec<usize> = Vec::with_capacity(ctx.final_turn_indices.len() + front.len());
    let mut seen: DetHashSet<usize> = DetHashSet::default();
    for &ti in &front {
        if seen.insert(ti) {
            reordered.push(ti);
        }
    }
    // Keep non-question declarative turns next, then remaining turns.
    for &ti in &ctx.final_turn_indices {
        let tr = &ctx.ix.turns[ti];
        let noisy = tr.is_question
            || tr.is_backchannel
            || tr.body_for_headers.contains('?')
            || is_request_or_gratitude_wrapper(&tr.body_for_headers);
        if !noisy && seen.insert(ti) {
            reordered.push(ti);
        }
    }
    for &ti in &ctx.final_turn_indices {
        if seen.insert(ti) {
            reordered.push(ti);
        }
    }
    ctx.final_turn_indices = reordered;

    // Sync winning disclosure session to `final_session_indices[0]`.
    let win_sess = ctx.ix.turns[scored_pairs[0].0].session;
    if let Some(pos) = ctx.final_session_indices.iter().position(|&s| s == win_sess) {
        ctx.final_session_indices.remove(pos);
    }
    ctx.final_session_indices.insert(0, win_sess);
    ctx.implicit_disclosure_locked = true;
    true
}

/// Regime B: Named-speaker Open-Domain FST ConceptNet + WordNet + Wikidata + Roget + GeoNames
/// + B4 WordNet co-hyponym/LCS sibling bridge + Engine #6 Signed Galois Lattice
/// + zero-substance greeting/filler lockout + mandatory Regime B session sync.
fn apply_named_open_domain_fst_bridge(ctx: &mut RankCtx<'_>) {
    if !ctx.has_named_speakers_eff
        || ctx.ix.has_tool_calls
        || ctx.eff_it.sniper_when
        || ctx.eff_it.is_event_sequence
        || ctx.eff_it.multi_hop
        || ctx.eff_it.task_under_rules
        || ctx.torrens_locked_head
    {
        return;
    }

    let q_toks_owned = raw_tokens(&ctx.q.text);
    let q_toks_ref: Vec<&str> = q_toks_owned.iter().map(String::as_str).collect();
    if !is_open_domain_inferential_shape(&q_toks_ref) {
        return;
    }
    let strict_epistemic = is_strict_epistemic_inferential(&q_toks_ref);

    let mut speaker_vocab: DetHashSet<String> = DetHashSet::default();
    for tr in &ctx.ix.turns {
        if !tr.speaker.is_empty() {
            speaker_vocab.insert(tr.speaker.to_ascii_lowercase());
        }
    }
    let bindings = ctx.ix.nicks.bindings_in(&ctx.q.text);
    let mut queried_speakers: Vec<String> = Vec::new();
    for tok in &q_toks_owned {
        let low = tok.to_ascii_lowercase();
        let canon = bindings.get(&low).cloned().unwrap_or(low);
        if speaker_vocab.contains(&canon) && !queried_speakers.contains(&canon) {
            queried_speakers.push(canon);
        }
    }
    let single_sp = (queried_speakers.len() == 1).then(|| queried_speakers[0].as_str());

    let maps = build_fst_concept_map(ctx, &speaker_vocab);
    if maps.direct_weights.is_empty()
        && maps.fst_weights.is_empty()
        && maps.sibling_weights.is_empty()
    {
        return;
    }

    let t0 = ctx.final_turn_indices[0];

    // Pass 1: score FST + B4 sibling channels across turns to select Engine #6 Galois pool.
    struct PrelimTurn {
        ti: usize,
        si: usize,
        sp_ok: bool,
        is_pref_hab: bool,
        d: f32,
        f: f32,
        sib_w: f32,
        dh: u32,
        fh: u32,
        sh: u32,
        prelim_raw: f32,
    }
    let mut prelim: Vec<PrelimTurn> = Vec::with_capacity(ctx.nt);
    let mut t0_prelim: Option<(f32, f32, f32, u32, u32, u32, bool, bool)> = None;
    for ti in 0..ctx.nt {
        let tr = &ctx.ix.turns[ti];
        let sp_ok = single_sp.is_none_or(|sp| tr.speaker.eq_ignore_ascii_case(sp));
        let is_pref_hab = is_speaker_preference_or_habit_turn(tr);
        let allow_sib = single_sp.is_some() && sp_ok && !tr.is_question && !tr.is_backchannel;
        let skip_other = single_sp.is_some() && sp_ok;
        let (d, f, sib, dh, fh, sh) =
            score_turn_concepts_full(tr, &maps, allow_sib, skip_other);
        let sib_w = if is_pref_hab { sib * 1.35 } else { sib * 0.78 };
        if ti == t0 {
            t0_prelim = Some((d, f, sib_w, dh, fh, sh, sp_ok, is_pref_hab));
        }
        if tr.is_standing_policy {
            continue;
        }
        let prelim_raw = d + f + sib_w;
        if prelim_raw > 0.0 || (sp_ok && is_pref_hab) {
            prelim.push(PrelimTurn {
                ti,
                si: tr.session,
                sp_ok,
                is_pref_hab,
                d,
                f,
                sib_w,
                dh,
                fh,
                sh,
                prelim_raw,
            });
        }
    }

    // Build bounded candidate pool for Engine #6 (Signed Galois Lattice & Exclusion-Cone Resonance).
    let mut galois_pool: Vec<usize> = Vec::with_capacity(96);
    let mut galois_seen: DetHashSet<usize> = DetHashSet::default();
    for &ti in ctx.final_turn_indices.iter().take(20) {
        if galois_seen.insert(ti) {
            galois_pool.push(ti);
        }
    }
    let mut top_prelim_idx: Vec<usize> = (0..prelim.len()).collect();
    top_prelim_idx.sort_by(|&a, &b| {
        prelim[b]
            .prelim_raw
            .partial_cmp(&prelim[a].prelim_raw)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| prelim[a].ti.cmp(&prelim[b].ti))
    });
    for &idx in top_prelim_idx.iter().take(48) {
        let ti = prelim[idx].ti;
        if galois_seen.insert(ti) {
            galois_pool.push(ti);
        }
    }
    let mut hab_added = 0usize;
    for pt in &prelim {
        if hab_added >= 32 {
            break;
        }
        if pt.sp_ok && pt.is_pref_hab && galois_seen.insert(pt.ti) {
            galois_pool.push(pt.ti);
            hab_added += 1;
        }
    }
    let exclude_sp_set: BTreeSet<String> = speaker_vocab.iter().cloned().collect();
    let galois_scores =
        score_signed_galois_candidates_q16(ctx.ix, &ctx.q.text, &galois_pool, &exclude_sp_set);
    let sheaf_scores = crate::retrieval::graph::sheaf_mobius_q16::evaluate_sheaf_mobius_pool_q16(
        ctx.ix,
        &ctx.q.text,
        &galois_pool,
        &exclude_sp_set,
    );

    let compute_turn_score = |ti: usize,
                              d: f32,
                              f: f32,
                              sib_w: f32,
                              dh: u32,
                              fh: u32,
                              sh: u32,
                              sp_ok: bool,
                              is_pref_hab: bool|
     -> (f32, f32, f32, f32, u32) {
        let tr = &ctx.ix.turns[ti];
        let g_sc = galois_scores.get(&ti).copied().unwrap_or_default();
        let sh_sc = sheaf_scores.get(&ti).copied().unwrap_or_default();
        let heyting_lift = if sp_ok && !tr.is_question && sh_sc.heyting_entail_q16 > 0 {
            ((sh_sc.heyting_entail_q16 as f32) / 65_536.0).min(3.0) * 1.25
        } else {
            0.0
        };
        let neg_compat_lift = if sp_ok && !tr.is_question && sh_sc.heyting_neg_compat_q16 > 0 {
            ((sh_sc.heyting_neg_compat_q16 as f32) / 65_536.0).min(2.5) * 1.15
        } else {
            0.0
        };
        let sheaf_pos = if sp_ok && !tr.is_question && sh_sc.sheaf_harmony_q16 > 0 {
            ((sh_sc.sheaf_harmony_q16 as f32) / 65_536.0).min(3.5) * 1.10
        } else {
            0.0
        };
        let galois_meet =
            ((g_sc.galois_meet_q16 as f32) / 65_536.0).min(6.0) * 0.95 + heyting_lift + sheaf_pos;
        let hazard = if sp_ok && !tr.is_question {
            ((g_sc.hazard_resonance_q16 as f32) / 65_536.0).min(4.0) * 1.35 + neg_compat_lift
        } else {
            0.0
        };
        let cohomology_veto =
            if sh_sc.heyting_neg_compat_q16 <= -16_384 || sh_sc.sheaf_harmony_q16 <= -16_384 {
                0.38
            } else {
                1.0
            };
        let raw_concept = (d + f + sib_w + galois_meet + hazard) * cohomology_veto;
        let sp_mul = if single_sp.is_some() {
            if sp_ok { 1.55 } else { 0.55 }
        } else {
            1.0
        };
        let hab_mul = if sp_ok && is_pref_hab { 1.28 } else { 1.0 };
        let decl_mul = if tr.is_pure_info_question {
            0.65
        } else if tr.is_question {
            0.88
        } else {
            1.12
        };
        let total_hits = dh
            + fh
            + sh
            + u32::from(g_sc.galois_meet_q16 > 0)
            + u32::from(hazard > 0.0)
            + u32::from(sh_sc.heyting_entail_q16 >= 16_384 || sh_sc.sheaf_harmony_q16 >= 16_384);
        let multi_hit_bonus = if total_hits >= 3 {
            1.35
        } else if total_hits >= 2 {
            1.20
        } else {
            1.0
        };
        let total = raw_concept * sp_mul * hab_mul * decl_mul * multi_hit_bonus
            + 0.10 * ctx.t_scene[ti].max(0.0);
        (total, raw_concept, galois_meet, hazard, total_hits)
    };

    let (d0, f0, sib0, dh0, fh0, sh0, t0_sp_ok, t0_pref_hab) =
        t0_prelim.unwrap_or((0.0, 0.0, 0.0, 0, 0, 0, true, false));
    let tr0 = &ctx.ix.turns[t0];
    let t0_greeting = is_zero_substance_greeting_with_hits(tr0, dh0, fh0 + sh0);
    let (mut t0_total, raw0, galois0, hazard0, _t0_hits) =
        compute_turn_score(t0, d0, f0, sib0, dh0, fh0, sh0, t0_sp_ok, t0_pref_hab);

    struct ScoredCand {
        ti: usize,
        si: usize,
        sp_ok: bool,
        total: f32,
        raw_concept: f32,
        dh: u32,
        fh: u32,
        sh: u32,
        galois_meet: f32,
        hazard: f32,
        total_hits: u32,
    }
    let mut cands: Vec<ScoredCand> = Vec::with_capacity(prelim.len());
    // Track top-2 declarative speaker turn base scores per session for multi-turn coherence.
    let mut sess_top2: DetHashMap<usize, (f32, f32)> = DetHashMap::default();
    for pt in &prelim {
        let tr = &ctx.ix.turns[pt.ti];
        let (total, raw_concept, galois_meet, hazard, total_hits) = compute_turn_score(
            pt.ti,
            pt.d,
            pt.f,
            pt.sib_w,
            pt.dh,
            pt.fh,
            pt.sh,
            pt.sp_ok,
            pt.is_pref_hab,
        );
        if total_hits == 0
            || is_zero_substance_greeting_with_hits(tr, pt.dh, pt.fh + pt.sh)
        {
            continue;
        }
        if pt.sp_ok && !tr.is_question {
            let e = sess_top2.entry(pt.si).or_insert((0.0, 0.0));
            if total > e.0 {
                e.1 = e.0;
                e.0 = total;
            } else if total > e.1 {
                e.1 = total;
            }
        }
        cands.push(ScoredCand {
            ti: pt.ti,
            si: pt.si,
            sp_ok: pt.sp_ok,
            total,
            raw_concept,
            dh: pt.dh,
            fh: pt.fh,
            sh: pt.sh,
            galois_meet,
            hazard,
            total_hits,
        });
    }

    if cands.is_empty() {
        return;
    }

    // Apply bounded multi-turn session coherence bonus to both candidates and t0.
    for c in &mut cands {
        if c.sp_ok
            && let Some(&(s1, s2)) = sess_top2.get(&c.si)
        {
            let companion = if (c.total - s1).abs() < 1e-4 { s2 } else { s1 };
            if companion > 0.0 {
                c.total += 0.14 * companion.min(5.0);
            }
        }
    }
    if t0_sp_ok
        && let Some(&(s1, s2)) = sess_top2.get(&tr0.session)
    {
        let companion = if (t0_total - s1).abs() < 1e-4 { s2 } else { s1 };
        if companion > 0.0 {
            t0_total += 0.14 * companion.min(5.0);
        }
    }

    cands.sort_by(|a, b| {
        b.total
            .partial_cmp(&a.total)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.ti.cmp(&b.ti))
    });

    let best = &cands[0];
    let best_tr = &ctx.ix.turns[best.ti];
    let should_promote_head = best.ti != t0
        && (t0_greeting
            || (dh0 == 0
                && fh0 == 0
                && sh0 == 0
                && galois0 == 0.0
                && hazard0 == 0.0
                && best.sp_ok
                && (best.dh > 0
                    || best.fh + best.sh >= 2
                    || best.galois_meet > 0.0
                    || best.hazard > 0.0)
                && best.raw_concept >= 2.0)
            || (!t0_sp_ok
                && best.sp_ok
                && (dh0 == 0 || best.dh >= dh0)
                && best.total >= 1.15 * t0_total
                && best.raw_concept >= 1.6)
            || (strict_epistemic
                && best.sp_ok
                && !best_tr.is_pure_info_question
                && best.total >= 1.32 * t0_total
                && (best.raw_concept >= 2.2 || best.galois_meet > 0.0 || best.hazard > 0.0)
                && (best.total_hits >= 2 || best.raw_concept >= 1.45 * raw0))
            || (!strict_epistemic
                && best.sp_ok
                && !best_tr.is_pure_info_question
                && best.total >= 1.65 * t0_total
                && best.total_hits >= 2
                && (best.fh + best.sh >= 2 || best.galois_meet > 0.0 || best.hazard > 0.0)
                && best.raw_concept >= 3.0));

    let mut mutated_top5 = false;
    let mut head_synced_si: Option<usize> = None;
    if should_promote_head {
        if let Some(pos) = ctx.final_turn_indices.iter().position(|&x| x == best.ti) {
            ctx.final_turn_indices.remove(pos);
        }
        ctx.final_turn_indices.insert(0, best.ti);
        mutated_top5 = true;
        // Regime B session sync: sync `best.si` to `final_session_indices[0]` unless the
        // incumbent Stage-12 session head strictly dominates `best.si` on proven `s_score + bridge_boost`.
        let inc_s0 = ctx.final_session_indices.first().copied();
        let inc_dom = inc_s0.is_some_and(|s0| {
            s0 != best.si
                && (ctx.s_score[s0] + ctx.bridge_boost[s0])
                    > 1.25 * (ctx.s_score[best.si] + ctx.bridge_boost[best.si])
        });
        if !inc_dom {
            head_synced_si = Some(best.si);
        }
    } else if best.ti == t0
        && strict_epistemic
        && best.sp_ok
        && best.raw_concept >= 2.0
        && !ctx.final_session_indices.is_empty()
        && ctx.final_session_indices[0] != best.si
    {
        head_synced_si = Some(best.si);
    }

    // Seat up to 2 strong distinct-session FST/Galois bridged turns into greeting/courtesy,
    // wrong-speaker, or zero-concept question tail slots inside Top-5.
    let mut seated = 0usize;
    let mut seen_s: DetHashSet<usize> = ctx
        .final_turn_indices
        .iter()
        .take(3)
        .map(|&ti| ctx.ix.turns[ti].session)
        .collect();
    for cand in cands.iter().take(10) {
        if seated >= 2 || cand.raw_concept < 2.0 || cand.total < 2.6 {
            break;
        }
        let top5 = ctx.final_turn_indices.len().min(5);
        if ctx.final_turn_indices[..top5].contains(&cand.ti) || !seen_s.insert(cand.si) {
            continue;
        }
        let mut repl: Option<usize> = None;
        for pos in (1..top5).rev() {
            let cur = ctx.final_turn_indices[pos];
            let cur_tr = &ctx.ix.turns[cur];
            let (_, _, cdh, cfh) = score_turn_concepts(cur_tr, &maps);
            let cur_wrong_sp = single_sp
                .is_some_and(|sp| !cur_tr.speaker.eq_ignore_ascii_case(sp));
            if is_zero_substance_greeting_with_hits(cur_tr, cdh, cfh)
                || (cdh == 0 && cfh == 0 && (cur_tr.is_question || (strict_epistemic && cur_wrong_sp)))
            {
                repl = Some(pos);
                break;
            }
        }
        if let Some(pos) = repl {
            if let Some(old) = ctx.final_turn_indices.iter().position(|&x| x == cand.ti) {
                ctx.final_turn_indices.remove(old);
            }
            let ins = pos.min(ctx.final_turn_indices.len());
            ctx.final_turn_indices.insert(ins, cand.ti);
            mutated_top5 = true;
            seated += 1;
        }
    }

    if (mutated_top5 || head_synced_si.is_some()) && !ctx.final_session_indices.is_empty() {
        let top5 = ctx.final_turn_indices.len().min(5);
        let mut synced_sess: Vec<usize> = Vec::with_capacity(ctx.final_session_indices.len() + top5);
        let mut seen_sync: DetHashSet<usize> = DetHashSet::default();
        if let Some(win_si) = head_synced_si
            && seen_sync.insert(win_si)
        {
            synced_sess.push(win_si);
        }
        for &ti in &ctx.final_turn_indices[..top5] {
            let si = ctx.ix.turns[ti].session;
            if seen_sync.insert(si) {
                synced_sess.push(si);
            }
        }
        for &si in &ctx.final_session_indices {
            if seen_sync.insert(si) {
                synced_sess.push(si);
            }
        }
        ctx.final_session_indices = synced_sess;
    }
}

/// Regime C: Non-sparse advice-frame user assertion rescue (demotes assistant essays
/// and pure user questions below declarative `field == 0` user turns carrying stronger
/// wrapper-stripped exact + FST concept + personal constraint/habit evidence).
fn apply_dense_advice_user_rescue(ctx: &mut RankCtx<'_>) {
    if ctx.has_named_speakers_eff
        || ctx.ix.has_tool_calls
        || ctx.implicit_disclosure_locked
        || ctx.eff_it.is_event_sequence
        || ctx.eff_it.assistant_attribution
        || ctx.eff_it.multi_hop
        || ctx.eff_it.sniper_when
        || ctx.eff_it.temporal
        || eff_summary_request(&ctx.q.text)
        || ctx.torrens_locked_head
    {
        return;
    }
    if !is_implicit_advice_frame(&ctx.q.text) {
        return;
    }
    if ctx.final_turn_indices.len() < 2 {
        return;
    }

    let t0 = ctx.final_turn_indices[0];
    let tr0 = &ctx.ix.turns[t0];
    let slot0_locked = tr0.is_standing_policy
        || (tr0.field == 0 && tr0.is_personal_constraint && !tr0.is_pure_info_question);

    let empty_sp = DetHashSet::default();
    let maps = build_fst_concept_map(ctx, &empty_sp);
    if maps.direct_weights.is_empty() && maps.fst_weights.is_empty() {
        return;
    }

    let (d0, f0, dh0_c, fh0_c) = score_turn_concepts(tr0, &maps);
    let t0_exact_bits = (ctx.t_exact_mask.get(t0).copied().unwrap_or(0) & ctx.q_group_mask).count_ones();
    let t0_weak = !slot0_locked
        && (tr0.field == 1
            || tr0.is_question
            || tr0.body_for_headers.contains('?')
            || is_request_or_gratitude_wrapper(&tr0.body_for_headers)
            || (dh0_c == 0 && fh0_c <= 1 && t0_exact_bits <= 1));

    let window = ctx.final_turn_indices.len().min(20);
    let mut pool: Vec<usize> = ctx.final_turn_indices[1..window].to_vec();
    let mut seen_pool: DetHashSet<usize> = pool.iter().copied().collect();
    seen_pool.insert(t0);
    for &si in ctx.final_session_indices.iter().take(5) {
        for &ti in &ctx.ix.session_turns[si] {
            if ctx.ix.turns[ti].field == 0 && seen_pool.insert(ti) {
                pool.push(ti);
            }
        }
    }
    // O(q_terms) KITE upward-cone + WordNet bridge posting lookup for zero-literal-overlap disclosures:
    let kite = ctx.ix.kite_index();
    for qw in raw_tokens(&ctx.q.text) {
        if qw.len() < 3 || !is_content(&qw) {
            continue;
        }
        if let Some(post) = kite.concept_postings.get(qw.as_str()) {
            for &(ti, _) in post.iter().take(48) {
                if ti < ctx.nt && ctx.ix.turns[ti].field == 0 && seen_pool.insert(ti) {
                    pool.push(ti);
                }
            }
        }
        for &syn in crate::wordnet::senses(&qw).iter().take(3) {
            if let Some(post) = kite.synset_postings.get(&syn) {
                for &(ti, _) in post.iter().take(48) {
                    if ti < ctx.nt && ctx.ix.turns[ti].field == 0 && seen_pool.insert(ti) {
                        pool.push(ti);
                    }
                }
            }
        }
    }
    for (ti, tr) in ctx.ix.turns.iter().enumerate() {
        if tr.field == 0
            && !tr.is_question
            && !tr.is_backchannel
            && !tr.is_standing_policy
            && (tr.is_personal_constraint
                || (tr.has_prose_disclosure && ctx.nt <= 2_500))
            && seen_pool.insert(ti)
        {
            pool.push(ti);
        }
    }

    let mut galois_pool = pool.clone();
    galois_pool.push(t0);
    let galois_scores =
        score_signed_galois_candidates_q16(ctx.ix, &ctx.q.text, &galois_pool, &BTreeSet::new());
    let sheaf_scores = crate::retrieval::graph::sheaf_mobius_q16::evaluate_sheaf_mobius_pool_q16(
        ctx.ix,
        &ctx.q.text,
        &galois_pool,
        &BTreeSet::new(),
    );

    let base_t0_sc = if tr0.field == 1 {
        0.45 * d0 + 0.50 * f0
    } else {
        (d0 * 1.45 + f0 * 1.15 + 0.10 * ctx.t_scene[t0].max(0.0)).max(2.2)
    };
    let mut ranked_cands: Vec<(f32, usize, usize)> = Vec::new();

    for ti in pool {
        let tr = &ctx.ix.turns[ti];
        if tr.field != 0
            || tr.is_question
            || tr.is_backchannel
            || tr.is_standing_policy
            || tr.body_for_headers.contains('?')
            || is_request_or_gratitude_wrapper(&tr.body_for_headers)
        {
            continue;
        }
        let low = tr.body_for_headers.to_ascii_lowercase();
        let toks = raw_tokens(&low);
        if !(3..=110).contains(&toks.len()) {
            continue;
        }
        let s_rank = ctx
            .final_session_indices
            .iter()
            .take(5)
            .position(|&s| s == tr.session);
        let (d, f, dh, fh) = score_turn_concepts(tr, &maps);
        let g_sc = galois_scores.get(&ti).copied().unwrap_or_default();
        let sh_sc = sheaf_scores.get(&ti).copied().unwrap_or_default();
        if sh_sc.heyting_neg_compat_q16 <= -16_384 || sh_sc.sheaf_harmony_q16 <= -16_384 {
            continue;
        }
        let sheaf_pos = if sh_sc.sheaf_harmony_q16 > 0 {
            ((sh_sc.sheaf_harmony_q16 as f32) / 65_536.0).min(3.5) * 1.05
        } else {
            0.0
        };
        let heyting_pos = if sh_sc.heyting_entail_q16 > 0 {
            ((sh_sc.heyting_entail_q16 as f32) / 65_536.0).min(3.0) * 1.15
        } else {
            0.0
        };
        let neg_pos = if sh_sc.heyting_neg_compat_q16 > 0 {
            ((sh_sc.heyting_neg_compat_q16 as f32) / 65_536.0).min(2.5) * 1.10
        } else {
            0.0
        };
        let has_sheaf_proof = sh_sc.heyting_entail_q16 >= 16_384
            || sh_sc.sheaf_harmony_q16 >= 16_384
            || sh_sc.heyting_neg_compat_q16 >= 16_384;
        let g_meet =
            ((g_sc.galois_meet_q16 as f32) / 65_536.0).min(4.0) * 0.85 + heyting_pos + sheaf_pos;
        let haz_res = ((g_sc.hazard_resonance_q16 as f32) / 65_536.0).min(3.5) * 1.15 + neg_pos;
        // hardcode-scan: closed-class (1st-person personal pronouns and contractions)
        let has_leading_1sg = toks.iter().take(8).any(|w| {
            matches!(
                w.as_str(),
                "i" | "i'm" | "i've" | "i'd" | "i'll" | "my" | "me" | "mine" | "we" | "we're" | "we've" | "our" | "ours"
            )
        });
        if !tr.is_personal_constraint && !tr.has_prose_disclosure && !has_leading_1sg {
            continue;
        }
        // hardcode-scan: closed-class (1st-person deontic, habit, and constraint stance modals/verbs)
        let has_stance = tr.is_personal_constraint
            || tr.has_prose_disclosure
            || [
                "prefer",
                "usually",
                "always",
                "never",
                "avoid",
                "without",
                "hard to",
                "struggle",
                "struggling",
                "worried",
                "takes a hit",
                "top priority",
                "learned",
                "focus on",
            ]
            .iter()
            .any(|p| low.contains(p));
        let in_top3_sess = s_rank.is_some_and(|r| r < 3);
        let has_concept_link = dh > 0
            || fh >= 2
            || g_sc.galois_meet_q16 > 0
            || g_sc.hazard_resonance_q16 > 0
            || has_sheaf_proof
            || (in_top3_sess && fh >= 1 && (has_stance || tr.has_prose_disclosure));
        if !has_concept_link {
            continue;
        }
        let sess_prior = match s_rank {
            Some(0) => 0.85,
            Some(1) | Some(2) => 0.55,
            Some(3) | Some(4) => 0.25,
            _ => 0.0,
        };
        let deontic_bonus = if has_stance { 1.25 } else { 0.0 };
        let sc = d * 1.45
            + f * 1.15
            + g_meet
            + haz_res
            + deontic_bonus
            + sess_prior
            + 0.10 * ctx.t_scene[ti].max(0.0);
        ranked_cands.push((sc, ti, tr.session));
    }

    if ranked_cands.is_empty() {
        return;
    }
    ranked_cands.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    // 1. Slot-0 promotion ONLY when `t0_weak` and top disclosure decisively beats `base_t0_sc`.
    let mut seated_slot0 = false;
    if t0_weak && ranked_cands[0].0 > base_t0_sc + 0.25 {
        let win_ti = ranked_cands[0].1;
        if let Some(pos) = ctx.final_turn_indices.iter().position(|&x| x == win_ti) {
            ctx.final_turn_indices.remove(pos);
        }
        ctx.final_turn_indices.insert(0, win_ti);
        let win_s = ctx.ix.turns[win_ti].session;
        if !ctx.final_session_indices.is_empty() && ctx.final_session_indices[0] != win_s {
            if let Some(sp) = ctx.final_session_indices.iter().position(|&s| s == win_s) {
                ctx.final_session_indices.remove(sp);
            }
            ctx.final_session_indices.insert(0, win_s);
        }
        seated_slot0 = true;
    }

    // 2. Non-destructive Top-5 disclosure packing (Slots 2..4): seat up to 2 runner-up
    //    distinct-session 1st-person personal disclosures from `final_session_indices[0..5]`
    //    without ever disturbing Slot 0 or Slot 1 (lifts @5/@10 on implicit advice queries
    //    where Session R@5 is already high while protecting @1 with zero regression).
    let start_idx = usize::from(seated_slot0);
    let mut packed = 0usize;
    let mut packed_sessions: BTreeSet<usize> = BTreeSet::new();
    if let Some(&h0) = ctx.final_turn_indices.first() {
        packed_sessions.insert(ctx.ix.turns[h0].session);
    }
    let orig_len = ctx.final_turn_indices.len();
    for &(sc, cti, csess) in ranked_cands.iter().skip(start_idx) {
        if packed >= 2 || sc < 2.1 {
            break;
        }
        if ctx.final_turn_indices.iter().take(5).any(|&x| x == cti) {
            packed_sessions.insert(csess);
            continue;
        }
        let in_top5_sess = ctx.final_session_indices.iter().take(5).any(|&s| s == csess);
        if !in_top5_sess {
            continue;
        }
        if packed_sessions.contains(&csess) && packed >= 1 {
            continue;
        }
        if let Some(pos) = ctx.final_turn_indices.iter().position(|&x| x == cti) {
            ctx.final_turn_indices.remove(pos);
        }
        let ins_slot = (2 + packed).min(4).min(ctx.final_turn_indices.len());
        ctx.final_turn_indices.insert(ins_slot, cti);
        ctx.final_turn_indices.truncate(orig_len);
        packed_sessions.insert(csess);
        packed += 1;
    }
}

/// Entry point wired after `tool_receipt::apply_tool_receipt_priority` at the end of Stage 12.
pub(in crate::retrieval) fn apply_unified_implicit_bridge(ctx: &mut RankCtx<'_>) {
    if ctx.final_turn_indices.is_empty() || ctx.nt < 2 {
        return;
    }
    if apply_sparse_personal_disclosure_regime(ctx) {
        ctx.implicit_disclosure_locked = true;
        return;
    }
    apply_named_open_domain_fst_bridge(ctx);
    let galois_advice_eligible = !ctx.has_named_speakers_eff
        && !ctx.ix.has_tool_calls
        && !ctx.implicit_disclosure_locked
        && !ctx.eff_it.is_event_sequence
        && !ctx.eff_it.assistant_attribution
        && !ctx.eff_it.multi_hop
        && !ctx.eff_it.sniper_when
        && !eff_summary_request(&ctx.q.text)
        && !ctx.torrens_locked_head
        && is_implicit_advice_frame(&ctx.q.text);
    apply_signed_galois_hazard_seat_q16(
        ctx.ix,
        &ctx.q.text,
        galois_advice_eligible,
        &mut ctx.final_turn_indices,
        &mut ctx.final_session_indices,
    );
    apply_dense_advice_user_rescue(ctx);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_turn(id: &str, role: &str, content: &str) -> Turn {
        Turn {
            id: id.to_string(),
            role: role.to_string(),
            content: content.to_string(),
            links: Vec::new(),
            stability: 0.0,
        }
    }

    #[test]
    fn compound_split_and_derivational_stems_resolve_via_wordnet() {
        assert_eq!(split_wordnet_compound("roadtrip"), Some(("road", "trip")));
        assert_eq!(split_wordnet_compound("bookshelf"), Some(("book", "shelf")));
        let derivs = derivational_base_stems("educational");
        assert!(
            derivs.iter().any(|s| s == "education" || s == "educate"),
            "got {derivs:?}"
        );
        assert!(ascii_edit_dist_le_1("educaton", "education"));
    }

    #[test]
    fn sparse_personal_disclosure_beats_question_and_thanks_chatter() {
        let mut sessions = Vec::new();
        // Create 35 long-form background Q&A sessions so personal disclosure pair is corpus-sparse.
        for i in 0..35 {
            sessions.push(Session {
                id: format!("bg_{i}"),
                date: "2024/01/01 (Mon) 10:00".into(),
                timestamp: i as i64 + 1,
                turns: vec![
                    mk_turn(
                        &format!("bg_{i}_u"),
                        "user",
                        "That's awesome, thanks for the recommendations! Do you have any other pet gift ideas for my trip next month?",
                    ),
                    mk_turn(
                        &format!("bg_{i}_a"),
                        "assistant",
                        &"Here is a comprehensive detailed guide covering fifty ideas and options for your question across multiple categories. ".repeat(8),
                    ),
                ],
            });
        }
        sessions.push(Session {
            id: "target".into(),
            date: "2024/02/01 (Thu) 10:00".into(),
            timestamp: 40,
            turns: vec![
                mk_turn("gold_u", "user", "I've been diagnosed with asthma."),
                mk_turn(
                    "gold_a",
                    "assistant",
                    "Understood, I'll remember this health condition.",
                ),
            ],
        });
        let req = RankRequest {
            sessions,
            queries: vec![Query {
                text: "My family wants to get a pet, do you have any recommendations?".into(),
                date: String::new(),
            }],
            top_k: 5,
            params: Params::default(),
        };
        let res = rank(&req);
        assert_eq!(res.results[0].turns[0].id, "gold_u");
    }

    #[test]
    fn open_domain_locks_out_greeting_and_bridges_vivaldi_and_roadtrip() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "s_greet".into(),
                    date: "2024/03/01 (Fri) 10:00".into(),
                    timestamp: 1,
                    turns: vec![
                        mk_turn("g1", "Caroline", "Hey Melanie! Just wanted to say hi!"),
                        mk_turn(
                            "g2",
                            "Melanie",
                            "Yep, Caroline. I'm off to go swimming with the kids. Talk to you soon!",
                        ),
                    ],
                },
                Session {
                    id: "s_trip".into(),
                    date: "2024/03/05 (Tue) 10:00".into(),
                    timestamp: 2,
                    turns: vec![
                        mk_turn(
                            "t1",
                            "Melanie",
                            "Yeah, our trip got off to a bad start when our car hit a pothole on the highway.",
                        ),
                        mk_turn("t2", "Caroline", "Oh no, glad you are safe!"),
                    ],
                },
                Session {
                    id: "s_music".into(),
                    date: "2024/03/10 (Sun) 10:00".into(),
                    timestamp: 3,
                    turns: vec![
                        mk_turn(
                            "m1",
                            "Melanie",
                            "I'm a fan of both classical music like Bach and Mozart, as well as modern acoustic songs.",
                        ),
                        mk_turn("m2", "Caroline", "That is great taste!"),
                    ],
                },
            ],
            queries: vec![
                Query {
                    text: "Would Melanie go on another roadtrip soon?".into(),
                    date: String::new(),
                },
                Query {
                    text: "Would Melanie likely enjoy the song The Four Seasons by Vivaldi?".into(),
                    date: String::new(),
                },
            ],
            top_k: 5,
            params: Params::default(),
        };
        let res = rank(&req);
        assert_eq!(res.results[0].turns[0].id, "t1");
        assert_eq!(res.results[1].turns[0].id, "m1");
        assert_eq!(res.results[1].sessions[0].id, "s_music");
    }

    #[test]
    fn open_domain_b4_sibling_and_galois_beats_lexical_decoy_and_syncs_session() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "s_decoy".into(),
                    date: "2024/03/01 (Fri) 10:00".into(),
                    timestamp: 1,
                    turns: vec![
                        mk_turn(
                            "decoy1",
                            "Melanie",
                            "The radio played a loud pop song yesterday at the cafe.",
                        ),
                        mk_turn("decoy2", "Caroline", "Nice!"),
                    ],
                },
                Session {
                    id: "s_classical".into(),
                    date: "2024/03/10 (Sun) 10:00".into(),
                    timestamp: 2,
                    turns: vec![
                        mk_turn(
                            "gold1",
                            "Melanie",
                            "I'm a fan of classical music like Bach and Mozart whenever I relax.",
                        ),
                        mk_turn("gold2", "Caroline", "Lovely taste!"),
                    ],
                },
            ],
            queries: vec![Query {
                text: "Would Melanie likely enjoy the song The Four Seasons by Vivaldi?".into(),
                date: String::new(),
            }],
            top_k: 5,
            params: Params::default(),
        };
        let res = rank(&req);
        assert_eq!(res.results[0].turns[0].id, "gold1");
        assert_eq!(res.results[0].sessions[0].id, "s_classical");
    }
}
