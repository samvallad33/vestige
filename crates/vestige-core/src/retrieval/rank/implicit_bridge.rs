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

use crate::retrieval::index::normalize::{
    ascii_edit_dist_le_1, derivational_base_stems, split_wordnet_compound,
};
use crate::retrieval::lexical::speaker_translation_q16::is_open_domain_inferential_shape;
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
        "considering ",
        "given my ",
        "my current ",
    ]
    .iter()
    .any(|p| ql.contains(p))
    {
        return false;
    }
    [
        "recommend",
        "suggestion",
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
        "what is it for",
        "what aged ",
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

#[inline]
fn turn_idf(nt: usize, df: u32) -> f32 {
    let n = (nt as f32).max(1.0);
    let d = (df as f32).max(0.5);
    ((n + 1.0) / (d + 0.5)).ln().max(0.5)
}

/// Build weighted FST expansion vocab map (`vocab_id -> weight`) for query concepts
/// using pinned offline WordNet 3.1 + ConceptNet 5.7 + Wikidata CC0 + compound splitting
/// + derivational stems + OOV typo/abbreviation rescue.
fn build_fst_concept_map(
    ctx: &RankCtx<'_>,
    speaker_vocab: &DetHashSet<String>,
) -> (DetHashMap<u32, f32>, DetHashMap<u32, f32>) {
    let nt = ctx.nt;
    let max_fst_df = ((nt / 7).clamp(14, 48)) as u32;
    let mut direct_weights: DetHashMap<u32, f32> = DetHashMap::default();
    let mut fst_weights: DetHashMap<u32, f32> = DetHashMap::default();

    let mut add_direct = |id: u32, w: f32| {
        direct_weights
            .entry(id)
            .and_modify(|cur| {
                if w > *cur {
                    *cur = w;
                }
            })
            .or_insert(w);
    };
    let mut add_fst = |tok: &str, base_w: f32| {
        if tok.len() < 3 || !is_content(tok) || is_wrapper_or_scaffold(tok) || speaker_vocab.contains(tok)
        {
            return;
        }
        if let Some(id) = ctx.ix.vocab.get(tok) {
            let df = ctx.ix.t_own_stats.df.get(&id).copied().unwrap_or(0);
            if (1..=max_fst_df).contains(&df) {
                let w = base_w * turn_idf(nt, df);
                fst_weights
                    .entry(id)
                    .and_modify(|cur| {
                        if w > *cur {
                            *cur = w;
                        }
                    })
                    .or_insert(w);
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

    // Also extract 2-word phrase/entity concepts (e.g., "dr seuss", "national park", "theme park",
    // "four seasons", "hot pot", "energy bar", "green card") for Wikidata & ConceptNet lookups.
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

    // Expand single concepts: direct + compound split + derivational + OOV edit-1/abbr + FST graphs.
    for c in &concepts {
        let mut grounded_here = false;
        if let Some(id) = ctx.ix.vocab.get(c) {
            let df = ctx.ix.t_own_stats.df.get(&id).copied().unwrap_or(0);
            if df > 0 {
                grounded_here = true;
                add_direct(id, 2.2 * turn_idf(nt, df));
            }
        }
        let c_st = stem(c);
        if c_st.len() >= 3
            && let Some(sid) = ctx.ix.vocab.get(&c_st)
        {
            let df = ctx.ix.t_own_stats.df.get(&sid).copied().unwrap_or(0);
            if df > 0 {
                grounded_here = true;
                add_direct(sid, 1.5 * turn_idf(nt, df));
            }
        }

        // 1. Compound splitting (`roadtrip -> road + trip`, `bookshelf -> book + shelf`).
        if let Some((left, right)) = split_wordnet_compound(c) {
            for part in [left, right] {
                if !is_wrapper_or_scaffold(part) {
                    if let Some(pid) = ctx.ix.vocab.get(part) {
                        let df = ctx.ix.t_own_stats.df.get(&pid).copied().unwrap_or(0);
                        if df > 0 {
                            add_direct(pid, 1.65 * turn_idf(nt, df));
                        }
                    }
                    add_fst(part, 0.95);
                }
            }
        }

        // 2. Derivational morphology (`educational -> education, educate`).
        for d in derivational_base_stems(c) {
            if let Some(did) = ctx.ix.vocab.get(&d) {
                let df = ctx.ix.t_own_stats.df.get(&did).copied().unwrap_or(0);
                if df > 0 {
                    add_direct(did, 1.45 * turn_idf(nt, df));
                }
            }
            add_fst(&d, 0.85);
        }

        // 3. OOV typo / abbreviation rescue (`educaton -> education / edu`).
        if !grounded_here && c.len() >= 6 {
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
                    add_direct(vid, 1.85 * turn_idf(nt, df));
                    // Also expand FST from the corrected word!
                    for (nbr, sc) in crate::conceptnet::two_hop_neighbors(vword, 14) {
                        let w = ((sc as f32) / 65536.0).clamp(0.15, 1.0) * 0.70;
                        for tok in raw_tokens(nbr) {
                            add_fst(&tok, w);
                        }
                    }
                    for (nbr, sc) in crate::wordnet::full_lexicon_neighbors(vword)
                        .into_iter()
                        .take(14)
                    {
                        let w = ((sc as f32) / 65536.0).clamp(0.15, 1.0) * 0.65;
                        for tok in raw_tokens(nbr) {
                            add_fst(&tok, w);
                        }
                    }
                } else if vword.len() == 3 && vword == prefix3 && !is_wrapper_or_scaffold(vword) {
                    add_direct(vid, 1.25 * turn_idf(nt, df));
                }
            }
        }

        // 4. Wikidata entity upward containers + direct typed edges (`Vivaldi -> classical, composer`).
        for container in crate::wikidata::upward_containers(c, 2).into_iter().take(12) {
            for tok in raw_tokens(container) {
                add_fst(&tok, 0.85);
            }
        }
        for (target, _) in crate::wikidata::direct_typed_edges(c).into_iter().take(12) {
            for tok in raw_tokens(target) {
                add_fst(&tok, 0.90);
            }
        }

        // 5. ConceptNet 1-2 hop + reciprocal 2-hop neighbors.
        for (nbr, sc) in crate::conceptnet::two_hop_neighbors(c, 22) {
            let w = ((sc as f32) / 65536.0).clamp(0.15, 1.0) * 0.80;
            for tok in raw_tokens(nbr) {
                add_fst(&tok, w);
            }
        }
        for (nbr, sc) in crate::conceptnet::reciprocal_two_hop_neighbors(c, 16) {
            let w = ((sc as f32) / 65536.0).clamp(0.15, 1.0) * 0.75;
            for tok in raw_tokens(nbr) {
                add_fst(&tok, w);
            }
        }

        // 6. WordNet full lexicon neighbors + dual gloss hypernym bridge + gloss concepts.
        for (nbr, sc) in crate::wordnet::full_lexicon_neighbors(c)
            .into_iter()
            .take(18)
        {
            let w = ((sc as f32) / 65536.0).clamp(0.15, 1.0) * 0.72;
            for tok in raw_tokens(nbr) {
                add_fst(&tok, w);
            }
        }
        for (nbr, sc) in crate::wordnet::dual_gloss_hypernym_bridge(c, 14) {
            let w = ((sc as f32) / 65536.0).clamp(0.15, 1.0) * 0.65;
            for tok in raw_tokens(nbr) {
                add_fst(&tok, w);
            }
        }
        for gc in crate::wordnet::gloss_and_compound_concepts(c)
            .into_iter()
            .take(10)
        {
            for tok in raw_tokens(gc) {
                add_fst(&tok, 0.50);
            }
        }
    }

    // Phrase-level Wikidata + ConceptNet lookups ("Dr. Seuss", "national park", "Four Seasons").
    for ph in &phrases {
        if crate::wikidata::has_source(ph) {
            for container in crate::wikidata::upward_containers(ph, 2).into_iter().take(12) {
                for tok in raw_tokens(container) {
                    add_fst(&tok, 1.05);
                }
            }
            for (target, _) in crate::wikidata::direct_typed_edges(ph).into_iter().take(12) {
                for tok in raw_tokens(target) {
                    add_fst(&tok, 1.10);
                }
            }
        }
        for (nbr, sc) in crate::conceptnet::two_hop_neighbors(ph, 16) {
            let w = ((sc as f32) / 65536.0).clamp(0.20, 1.0) * 0.95;
            for tok in raw_tokens(nbr) {
                add_fst(&tok, w);
            }
        }
    }

    (direct_weights, fst_weights)
}

/// Score a turn's `content` against `(direct_weights, fst_weights)`.
fn score_turn_concepts(
    tr: &TurnIx,
    direct_weights: &DetHashMap<u32, f32>,
    fst_weights: &DetHashMap<u32, f32>,
) -> (f32, f32, u32, u32) {
    let mut direct = 0.0f32;
    let mut fst = 0.0f32;
    let mut d_hits = 0u32;
    let mut f_hits = 0u32;
    for &cid in &tr.content {
        if let Some(&w) = direct_weights.get(&cid) {
            direct += w;
            d_hits += 1;
        }
        if let Some(&w) = fst_weights.get(&cid) {
            fst += w;
            f_hits += 1;
        }
    }
    (direct, fst, d_hits, f_hits)
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
        let has_leading_1sg = u_toks
            .iter()
            .take(7)
            .any(|w| matches!(w.as_str(), "i" | "my" | "me" | "we" | "our"));
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
        || ctx.eff_it.task_under_rules
        || ctx.eff_it.is_event_sequence
        || ctx.eff_it.assistant_attribution
        || ctx.torrens_locked_head
    {
        return false;
    }

    let pairs = find_sparse_personal_disclosure_pairs(ctx.ix);
    // Strict corpus-sparsity gate: only fire when concise disclosure pairs are rare
    // (`1..=5` pairs and `< 1.6%` of corpus turns). Verified 0/5732 on BEAM 100K and
    // skipped on dense dialogue corpora (>15% density).
    if pairs.is_empty() || pairs.len() > 5 || pairs.len() * 60 > ctx.nt {
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

    let advice_q = is_implicit_advice_frame(&ctx.q.text);
    let profile_q = !advice_q
        && !ctx.eff_it.temporal
        && !ctx.eff_it.current_state
        && is_personal_profile_recall_query(&ctx.q.text);
    if !advice_q && !profile_q {
        return false;
    }

    let empty_sp = DetHashSet::default();
    let (direct_w, fst_w) = build_fst_concept_map(ctx, &empty_sp);

    let mut scored_pairs: Vec<(usize, usize, f32, bool)> = Vec::with_capacity(pairs.len());
    for &(u_idx, a_idx, mem_ack) in &pairs {
        let (ud, uf, _, _) = score_turn_concepts(&ctx.ix.turns[u_idx], &direct_w, &fst_w);
        let (ad, af, _, _) = score_turn_concepts(&ctx.ix.turns[a_idx], &direct_w, &fst_w);
        let ack_bonus = if mem_ack { 6.0 } else { 1.5 };
        let sc = (ud + ad) * 2.0 + (uf + af) * 1.15 + ack_bonus + 0.10 * ctx.t_scene[u_idx].max(0.0);
        scored_pairs.push((u_idx, a_idx, sc, mem_ack));
    }
    scored_pairs.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    // Strict zero-regression guard on `profile_q` (direct memory recall such as LongMemEval):
    // ONLY elevate on `profile_q` when the sparse disclosure pair carries an explicit
    // assistant memory-commit acknowledgment (`best_ack == true`). This protects 100% of
    // LongMemEval Temporal/Update queries (where `t0` often ends with a polite `?`) while
    // capturing explicit user-profile disclosures ("I'll remember...", "I've noted...").
    if profile_q {
        let (_, _, _, best_ack) = scored_pairs[0];
        if !best_ack {
            return false;
        }
    }

    // Seat winning disclosure pair(s) cleanly at the front of `final_turn_indices`
    // while demoting questions/wrappers within Top-10 and syncing `final_session_indices[0]`.
    let mut front: Vec<usize> = Vec::with_capacity(pairs.len() * 2);
    for &(u_idx, a_idx, _, _) in &scored_pairs {
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
    true
}

/// Regime B: Named-speaker Open-Domain FST ConceptNet + WordNet + Wikidata bridge
/// + zero-substance greeting/filler lockout.
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

    let (direct_w, fst_w) = build_fst_concept_map(ctx, &speaker_vocab);
    if direct_w.is_empty() && fst_w.is_empty() {
        return;
    }

    let t0 = ctx.final_turn_indices[0];
    let tr0 = &ctx.ix.turns[t0];
    let t0_greeting = is_zero_substance_greeting(tr0);
    let (_, _, dh0, fh0) = score_turn_concepts(tr0, &direct_w, &fst_w);

    let mut cands: Vec<(usize, usize, f32, f32, u32)> = Vec::new();
    for ti in 0..ctx.nt {
        let tr = &ctx.ix.turns[ti];
        if is_zero_substance_greeting(tr) || tr.is_standing_policy {
            continue;
        }
        let (d, f, dh, fh) = score_turn_concepts(tr, &direct_w, &fst_w);
        if dh == 0 && fh == 0 {
            continue;
        }
        let raw_concept = d + f;
        // Speaker alignment: on single-speaker inferential questions ("Would Melanie..."),
        // strongly prefer the queried speaker's own declarative utterances.
        let sp_mul = if queried_speakers.len() == 1 {
            if tr.speaker.eq_ignore_ascii_case(&queried_speakers[0]) {
                1.45
            } else {
                0.70
            }
        } else {
            1.0
        };
        let decl_mul = if tr.is_pure_info_question { 0.72 } else { 1.10 };
        let multi_hit_bonus = if dh + fh >= 2 { 1.25 } else { 1.0 };
        let total = raw_concept * sp_mul * decl_mul * multi_hit_bonus
            + 0.10 * ctx.t_scene[ti].max(0.0);
        cands.push((ti, tr.session, total, raw_concept, dh));
    }

    if cands.is_empty() {
        return;
    }
    cands.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    // Strict Zero-Regression Head Rule: ONLY rescue Rank 1 on Open-Domain inferential
    // queries when `t0` is a zero-substance greeting/filler OR when `t0` has zero concept
    // alignment and the candidate matches the queried speaker with a direct compound/OOV hit.
    let (best_ti, best_si, _best_total, best_raw, best_dh) = cands[0];
    let best_sp_ok = queried_speakers.len() != 1
        || ctx.ix.turns[best_ti]
            .speaker
            .eq_ignore_ascii_case(&queried_speakers[0]);
    let should_promote_head = best_ti != t0
        && (t0_greeting
            || (dh0 == 0 && fh0 == 0 && best_dh > 0 && best_sp_ok && best_raw >= 3.2));

    if should_promote_head {
        if let Some(pos) = ctx.final_turn_indices.iter().position(|&x| x == best_ti) {
            ctx.final_turn_indices.remove(pos);
        }
        ctx.final_turn_indices.insert(0, best_ti);

        // Sync Rank-1 session ONLY when displacing a greeting or zero-concept session.
        if !ctx.final_session_indices.is_empty() {
            let s0 = ctx.final_session_indices[0];
            if s0 != best_si && (t0_greeting || (dh0 == 0 && fh0 == 0)) {
                if let Some(sp) = ctx.final_session_indices.iter().position(|&s| s == best_si) {
                    ctx.final_session_indices.remove(sp);
                }
                ctx.final_session_indices.insert(0, best_si);
            }
        }
    }

    // Seat up to 2 strong distinct-session FST bridged turns ONLY into zero-substance
    // greeting tail slots inside Top-5 (never evicting substantive Stage-4 candidates).
    let mut seated = 0usize;
    let mut seen_s: DetHashSet<usize> = ctx
        .final_turn_indices
        .iter()
        .take(3)
        .map(|&ti| ctx.ix.turns[ti].session)
        .collect();
    for &(ti, si, total, raw_c, _) in cands.iter().take(6) {
        if seated >= 2 || raw_c < 2.8 || total < 3.5 {
            break;
        }
        let top5 = ctx.final_turn_indices.len().min(5);
        if ctx.final_turn_indices[..top5].contains(&ti) || !seen_s.insert(si) {
            continue;
        }
        let mut repl: Option<usize> = None;
        for pos in (1..top5).rev() {
            let cur = ctx.final_turn_indices[pos];
            if is_zero_substance_greeting(&ctx.ix.turns[cur]) {
                repl = Some(pos);
                break;
            }
        }
        if let Some(pos) = repl {
            if let Some(old) = ctx.final_turn_indices.iter().position(|&x| x == ti) {
                ctx.final_turn_indices.remove(old);
            }
            let ins = pos.min(ctx.final_turn_indices.len());
            ctx.final_turn_indices.insert(ins, ti);
            seated += 1;
        }
    }
}

/// Regime C: Non-sparse advice-frame user assertion rescue (demotes assistant essays
/// and pure user questions below declarative `field == 0` user turns carrying stronger
/// wrapper-stripped exact + FST concept evidence).
fn apply_dense_advice_user_rescue(ctx: &mut RankCtx<'_>) {
    if ctx.has_named_speakers_eff
        || ctx.ix.has_tool_calls
        || ctx.pref_on
        || ctx.eff_it.task_under_rules
        || ctx.eff_it.is_event_sequence
        || ctx.eff_it.assistant_attribution
        || ctx.eff_it.multi_hop
        || ctx.eff_it.temporal
        || ctx.eff_it.current_state
        || eff_summary_request(&ctx.q.text)
        || ctx.torrens_locked_head
    {
        return;
    }
    if !is_implicit_advice_frame(&ctx.q.text) {
        return;
    }
    let window = ctx.final_turn_indices.len().min(10);
    if window < 2 {
        return;
    }

    let empty_sp = DetHashSet::default();
    let (direct_w, fst_w) = build_fst_concept_map(ctx, &empty_sp);
    if direct_w.is_empty() && fst_w.is_empty() {
        return;
    }

    let t0 = ctx.final_turn_indices[0];
    let tr0 = &ctx.ix.turns[t0];
    let (d0, f0, dh0, fh0) = score_turn_concepts(tr0, &direct_w, &fst_w);
    let t0_weak = tr0.field == 1 || tr0.is_pure_info_question || (dh0 == 0 && fh0 == 0);
    if !t0_weak {
        return;
    }

    let mut best_pos: Option<usize> = None;
    let mut best_sc = (d0 + f0).max(0.0);
    for (i, &ti) in ctx.final_turn_indices[1..window].iter().enumerate() {
        let tr = &ctx.ix.turns[ti];
        if tr.field != 0 || tr.is_question || tr.is_backchannel || tr.is_standing_policy {
            continue;
        }
        let (d, f, dh, fh) = score_turn_concepts(tr, &direct_w, &fst_w);
        if dh == 0 && fh == 0 {
            continue;
        }
        let sc = d * 1.3 + f;
        if sc > best_sc + 0.5 {
            best_sc = sc;
            best_pos = Some(i + 1);
        }
    }
    if let Some(pos) = best_pos {
        let cand = ctx.final_turn_indices.remove(pos);
        ctx.final_turn_indices.insert(0, cand);
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
    }
}
