//! Explicit source-time annotations. Original text is never rewritten.
use chrono::{Datelike, Days, NaiveDate};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    pub from: String,
    pub to: String,
    pub days: i64,
    pub whole_weeks: i64,
    pub remainder_days: i64,
}
pub fn date(text: &str) -> Option<NaiveDate> {
    let short: String = text.chars().take(10).collect();
    NaiveDate::parse_from_str(&short.replace('/', "-"), "%Y-%m-%d").ok()
}
pub fn delta(from: &str, to: &str) -> Option<Delta> {
    let days = (date(to)? - date(from)?).num_days();
    Some(Delta {
        from: from.into(),
        to: to.into(),
        days,
        whole_weeks: days / 7,
        remainder_days: days % 7,
    })
}
pub fn canonicalize(
    text: &str,
    origin: Option<&str>,
    bindings: &BTreeMap<String, String>,
) -> String {
    // Bind only explicitly supplied, unambiguous speaker/identity references.
    // Surface matching respects token boundaries and preserves source quotes.
    let mut out = String::new();
    let tokens: Vec<&str> = text.split_inclusive(|c: char| c.is_whitespace()).collect();
    for token in tokens {
        let word = token.trim().trim_matches(|c: char| !c.is_alphanumeric());
        if let Some(name) = bindings.get(&word.to_lowercase()) {
            out.push_str(&token.replacen(word, &format!("{word} [= {name}]"), 1));
        } else {
            out.push_str(token);
        }
    }
    for (phrase, absolute) in resolved_dates(&out, origin) {
        let mut cursor = 0;
        let mut next = String::new();
        let lower = out.to_ascii_lowercase();
        for (at, _) in lower.match_indices(&phrase) {
            if boundary(&lower, at, phrase.len()) {
                next.push_str(&out[cursor..at + phrase.len()]);
                next.push_str(&format!(" [= {absolute}]"));
                cursor = at + phrase.len();
            }
        }
        next.push_str(&out[cursor..]);
        out = next;
    }
    out
}
pub fn resolved_dates(text: &str, origin: Option<&str>) -> Vec<(String, String)> {
    let Some(anchor) = origin.and_then(date) else {
        return Vec::new();
    };
    let lower = text.to_ascii_lowercase();
    let mut annotations = Vec::new();
    for (phrase, days) in [
        ("yesterday", 1),
        ("two weeks ago", 14),
        ("a couple days ago", 2),
        ("a couple of days ago", 2),
    ] {
        if whole(&lower, phrase)
            && let Some(value) = anchor.checked_sub_days(Days::new(days))
        {
            annotations.push((phrase.to_string(), value.to_string()));
        }
    }
    for (index, day) in [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    .iter()
    .enumerate()
    {
        let phrase = format!("last {day}");
        if whole(&lower, &phrase) {
            let days =
                (anchor.weekday().num_days_from_monday() as i64 - index as i64).rem_euclid(7);
            if let Some(value) =
                anchor.checked_sub_days(Days::new(if days == 0 { 7 } else { days as u64 }))
            {
                annotations.push((phrase, value.to_string()));
            }
        }
    }
    annotations
}
pub(crate) fn boundary(text: &str, at: usize, len: usize) -> bool {
    text[..at]
        .chars()
        .next_back()
        .is_none_or(|c| !c.is_alphanumeric())
        && text[at + len..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphanumeric())
}
pub(crate) fn whole(text: &str, needle: &str) -> bool {
    text.match_indices(needle)
        .any(|(at, _)| boundary(text, at, needle.len()))
}
