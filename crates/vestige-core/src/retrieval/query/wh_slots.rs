//! WH-slot structural classification and answer-form checks.

use crate::retrieval::*;


pub(in crate::retrieval) const WH_TIME: u8 = 1;
pub(in crate::retrieval) const WH_QTY: u8 = 2;
pub(in crate::retrieval) const WH_PERSON: u8 = 4;
pub(in crate::retrieval) const WH_PLACE: u8 = 8;
pub(in crate::retrieval) const WH_CAUSE: u8 = 16;

/// Closed class of English interrogative function words. Not a topical list.
pub(in crate::retrieval) fn wh_slots(query: &str) -> u8 {
    let lower = query.to_ascii_lowercase();
    let tokens: Vec<String> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect();
    let mut slots = 0u8;
    let has = |w: &str| tokens.iter().any(|t| t == w);
    let phrase = |a: &str, b: &str| {
        tokens.windows(2).any(|p| p[0] == a && p[1] == b)
    };
    // "when" is interrogative only before a closed-class auxiliary ("when did"),
    // not before a lexical verb ("when building"). Auxiliaries are the same
    // function words already in STOPWORDS.
    let when_interrogative = tokens.windows(2).any(|p| {
        p[0] == "when"
            && matches!(
                p[1].as_str(),
                "did" | "didnt"
                    | "does"
                    | "doesnt"
                    | "do"
                    | "dont"
                    | "is"
                    | "isnt"
                    | "are"
                    | "arent"
                    | "was"
                    | "wasnt"
                    | "were"
                    | "werent"
                    | "will"
                    | "wont"
                    | "can"
                    | "cant"
                    | "could"
                    | "couldnt"
                    | "would"
                    | "wouldnt"
                    | "should"
                    | "shouldnt"
                    | "shall"
                    | "has"
                    | "hasnt"
                    | "have"
                    | "havent"
                    | "had"
                    | "hadnt"
                    | "may"
                    | "might"
                    | "must"
            )
    });
    if phrase("what", "time")
        || phrase("which", "year")
        || phrase("how", "long")
        || when_interrogative
    {
        slots |= WH_TIME;
    }
    if phrase("how", "many") || phrase("how", "much") || phrase("how", "often") {
        slots |= WH_QTY;
    }
    if has("who") || has("whom") || has("whose") {
        slots |= WH_PERSON;
    }
    if has("where") {
        slots |= WH_PLACE;
    }
    if has("why") {
        slots |= WH_CAUSE;
    }
    slots
}

pub(in crate::retrieval) fn has_iso_or_hhmm(body: &str) -> bool {
    let b = body.as_bytes();
    if b.len() >= 10 {
        for i in 0..=b.len() - 10 {
            if b[i].is_ascii_digit()
                && b[i + 1].is_ascii_digit()
                && b[i + 2].is_ascii_digit()
                && b[i + 3].is_ascii_digit()
                && b[i + 4] == b'-'
                && b[i + 5].is_ascii_digit()
                && b[i + 6].is_ascii_digit()
                && b[i + 7] == b'-'
                && b[i + 8].is_ascii_digit()
                && b[i + 9].is_ascii_digit()
            {
                return true;
            }
        }
    }
    for i in 0..b.len() {
        if b[i] == b':'
            && i > 0
            && i + 2 < b.len()
            && b[i - 1].is_ascii_digit()
            && b[i + 1].is_ascii_digit()
            && b[i + 2].is_ascii_digit()
        {
            return true;
        }
    }
    false
}

pub(in crate::retrieval) fn has_ascii_digit(body: &str) -> bool {
    body.bytes().any(|c| c.is_ascii_digit())
}

pub(in crate::retrieval) fn has_handle_or_path(body: &str) -> bool {
    let b = body.as_bytes();
    for i in 0..b.len() {
        if b[i] == b'@' && i + 1 < b.len() && b[i + 1].is_ascii_alphanumeric() {
            return true;
        }
        if i + 2 < b.len() && b[i] == b':' && b[i + 1] == b'/' && b[i + 2] == b'/' {
            return true;
        }
    }
    body.split_whitespace().any(|tok| tok.contains('/') && tok.len() > 1)
}

/// A capitalized word of at least two letters that is not the first word of a sentence.
pub(in crate::retrieval) fn has_mid_sentence_name(body: &str) -> bool {
    let chars: Vec<char> = body.chars().collect();
    let mut sentence_start = true;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '.' || c == '!' || c == '?' || c == '\n' {
            sentence_start = true;
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_uppercase() {
            let start = i;
            while i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
            }
            if !sentence_start && i - start >= 2 {
                return true;
            }
            sentence_start = false;
            continue;
        }
        sentence_start = false;
        i += 1;
    }
    false
}

pub(in crate::retrieval) fn wh_satisfied(ix: &Index, ti: usize, slots: u8, has_cause_edge: bool) -> bool {
    if slots == 0 {
        return true;
    }
    let turn = &ix.turns[ti];
    let ts = ix.session_ts.get(turn.session).copied().unwrap_or(0);
    if slots & WH_TIME != 0 && ts == 0 && !turn.form_clock {
        return false;
    }
    if slots & WH_QTY != 0 && !turn.form_digit {
        return false;
    }
    if slots & WH_PERSON != 0 && !turn.form_handle && !turn.form_midcap {
        return false;
    }
    if slots & WH_PLACE != 0 && !turn.form_midcap && !turn.form_handle {
        return false;
    }
    if slots & WH_CAUSE != 0 && !has_cause_edge {
        return false;
    }
    true
}
