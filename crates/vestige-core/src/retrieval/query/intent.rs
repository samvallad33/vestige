//! Query classification, question shape detection, and query-plan types.

use crate::retrieval::*;

// =========================================================================
// QUERY ANALYSIS
// =========================================================================

#[derive(Clone)]
pub(in crate::retrieval) struct QTerm {
    pub(in crate::retrieval) id: u32,
    pub(in crate::retrieval) weight: f32,
    pub(in crate::retrieval) group: Option<usize>,
    pub(in crate::retrieval) exact_hit: bool,
}

pub(in crate::retrieval) struct QueryPlan {
    pub(in crate::retrieval) intent: QueryIntent,
    pub(in crate::retrieval) terms: Vec<QTerm>,
    pub(in crate::retrieval) groups: Vec<String>,
    /// term id -> group, for proximity scanning
    pub(in crate::retrieval) group_of: DetHashMap<u32, usize>,
    pub(in crate::retrieval) focal_ids: DetHashSet<u32>,
    /// Closed-class excluded NP term/stem IDs ("other than X", "besides X", "apart from X", "except X")
    pub(in crate::retrieval) excluded_ids: DetHashSet<u32>,
    /// Corpus-rare (`1 <= s_df <= 2`) exact proper/hyphenated entity IDs (`Flask-Login`, `\u{1}ent:*`)
    pub(in crate::retrieval) rare_entity_ids: Vec<u32>,
}

/// Normalize punctuation (preserving internal apostrophes in contractions like `i'm` / `don't`)
/// into single-space-padded lowercase tokens so word-boundary cue matching never misfires on
/// substrings ("Army" != "my", "server" != "serve", "Miami's" != "i'", "claim" != "aim")
/// and never strips literal "$0".
pub(in crate::retrieval) fn word_padded(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 4);
    out.push(' ');
    let chars: Vec<char> = text.chars().collect();
    let mut prev_space = true;
    for (idx, &ch) in chars.iter().enumerate() {
        let lower = ch.to_ascii_lowercase();
        let is_apos = (lower == '\'' || lower == '’')
            && idx > 0
            && idx + 1 < chars.len()
            && chars[idx - 1].is_alphanumeric()
            && chars[idx + 1].is_alphanumeric();
        // Glue a currency marker to the amount so "$0" stays the literal token "$0".
        let is_currency = ch == '$' && idx + 1 < chars.len() && chars[idx + 1].is_ascii_digit();
        if lower.is_alphanumeric() || is_currency {
            out.push(if is_currency { '$' } else { lower });
            prev_space = false;
        } else if is_apos {
            out.push('\'');
            prev_space = false;
        } else if !prev_space {
            out.push(' ');
            prev_space = true;
        }
    }
    if !prev_space {
        out.push(' ');
    }
    out
}

pub fn classify(text: &str) -> QueryIntent {
    let raw = raw_tokens(text);
    let slots = wh_slots(text);
    let dates = inline_date_tokens(text);
    let year_count = raw
        .iter()
        .filter(|w| w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()))
        .count();
    let digit_groups = raw.iter().filter(|w| w.chars().any(|c| c.is_ascii_digit())).count();
    let temporal = (slots & WH_TIME) != 0
        || !dates.is_empty()
        || year_count > 0
        || raw.iter().any(|w| token_is_temporal_word(w));
    let distinct_years = dates
        .iter()
        .filter(|d| d.starts_with("__YEAR__"))
        .collect::<DetHashSet<_>>()
        .len();
    let distinct_months = dates
        .iter()
        .filter(|d| d.starts_with("__MONTH__"))
        .collect::<DetHashSet<_>>()
        .len();
    let distinct_dates = dates
        .iter()
        .filter(|d| d.starts_with("__DATE__"))
        .collect::<DetHashSet<_>>()
        .len();
    let distinct_seasons = dates
        .iter()
        .filter(|d| d.starts_with("__SEASON__"))
        .collect::<DetHashSet<_>>()
        .len();
    let content_groups = raw.iter().filter(|w| is_content(w)).collect::<DetHashSet<_>>().len();
    QueryIntent {
        // Public Skeptic-Audit contract: `classify` leaves `preference` false and sets
        // `is_event_sequence` strictly from >=2 distinct explicit calendar markers; internal effective
        // sequence/preference modes are derived in `rank_one` (`eff_it`) without mutating `RankResult.intent`.
        preference: false,
        current_state: false,
        assistant_attribution: false,
        user_attribution: false,
        multi_hop: content_groups >= 3 && (digit_groups >= 2 || slots.count_ones() >= 2),
        inference: false,
        temporal,
        sniper_when: (slots & WH_TIME) != 0 || year_count > 0,
        sniper_how_many: (slots & WH_QTY) != 0,
        sniper_where: (slots & WH_PLACE) != 0,
        sniper_why: (slots & WH_CAUSE) != 0,
        is_event_sequence: year_count >= 2
            || distinct_years >= 2
            || distinct_months >= 2
            || distinct_dates >= 2
            || distinct_seasons >= 2,
        task_under_rules: false,
    }
}


pub(in crate::retrieval) fn detect_question(content: &str, raw_len: usize) -> (bool, bool) {
    let prose = content.split("```").next().unwrap_or(content).trim_end();
    if let Some(q_pos) = prose.rfind('?') {
        let tail = prose[q_pos + 1..].trim();
        let ends_with_q = tail.is_empty() || tail.chars().all(|c| !c.is_ascii_alphabetic());
        let is_q = ends_with_q || prose.len() <= 350;
        let is_short_q = ends_with_q && raw_len < 10;
        return (is_q, is_short_q);
    }
    (false, false)
}

/// Closed-class interrogatives and auxiliaries. Not a topical vocabulary.
// hardcode-scan: closed-class (question words)
pub(in crate::retrieval) const INFO_Q_WORDS: &[&str] = &[
    "who", "whom", "whose", "what", "where", "when", "why", "which", "how",
    "did", "do", "does", "is", "are", "was", "were",
    "have", "has", "had", "can", "could", "would", "should", "will",
];

pub(in crate::retrieval) fn take_word(text: &str) -> Option<(&str, &str)> {
    let text = text.trim_start();
    let end = text
        .char_indices()
        .find(|(_, c)| !c.is_ascii_alphanumeric())
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    if end == 0 { None } else { Some((&text[..end], &text[end..])) }
}

pub(in crate::retrieval) fn skip_comma(text: &str) -> &str {
    let s = text.trim_start();
    s.strip_prefix(',').unwrap_or(s).trim_start()
}

pub(in crate::retrieval) fn is_speaker_alias(tok: &str, names: &DetHashSet<String>) -> bool {
    if names.contains(tok) {
        return true;
    }
    if tok.len() < 3 || stopwords().contains(tok) {
        return false;
    }
    names.iter().any(|n| n.len() >= tok.len() + 3 && n.starts_with(tok))
}

pub(in crate::retrieval) fn strip_one_question_lead<'a>(text: &'a str, names: &DetHashSet<String>) -> Option<&'a str> {
    let s = text.trim_start();
    let (tok, rest) = take_word(s)?;
    let low = tok.to_ascii_lowercase();
    // A corpus-attested speaker name followed by a comma is a vocative, not content.
    if is_speaker_alias(&low, names) && rest.trim_start().starts_with(',') {
        return Some(skip_comma(rest));
    }
    None
}

pub(in crate::retrieval) fn info_question_start(clause: &str) -> bool {
    let s = clause.trim_start();
    let Some((tok, rest)) = take_word(s) else { return false };
    let low = tok.to_ascii_lowercase();
    if low == "what" {
        if let Some((next, _)) = take_word(skip_comma(rest)) {
            let n = next.to_ascii_lowercase();
            if n == "a" || n == "an" {
                return false;
            }
        }
    }
    INFO_Q_WORDS.contains(&low.as_str())
}

pub(in crate::retrieval) fn stripped_question_clause<'a>(clause: &'a str, names: &DetHashSet<String>) -> &'a str {
    let mut s = clause.trim();
    for _ in 0..8 {
        if info_question_start(s) {
            return s;
        }
        match strip_one_question_lead(s, names) {
            Some(next) => {
                let next = next.trim_start();
                if next.is_empty() || next.len() == s.len() {
                    return s;
                }
                s = next;
            }
            None => return s,
        }
    }
    s
}

pub(in crate::retrieval) fn last_sentence_boundary(head: &str) -> Option<usize> {
    let bytes = head.as_bytes();
    let mut end = head.len();
    while end > 0 {
        let rel = head[..end].rfind(|c: char| matches!(c, '.' | '!' | ';' | '\n'))?;
        let ok = match bytes[rel] {
            b'.' => {
                let prev_digit = rel > 0 && bytes[rel - 1].is_ascii_digit();
                let next_ok = head[rel + 1..].chars().next().is_none_or(|c| c.is_whitespace());
                !prev_digit && next_ok
            }
            _ => true,
        };
        if ok {
            return Some(rel);
        }
        end = rel;
    }
    None
}

pub(in crate::retrieval) fn terminal_question_spans(prose: &str) -> Option<(&str, &str)> {
    let prose = prose.split("```").next().unwrap_or(prose);
    let q_pos = prose.rfind('?')?;
    let tail = prose[q_pos + 1..].trim();
    if !(tail.is_empty() || tail.chars().all(|c| !c.is_ascii_alphabetic())) {
        return None;
    }
    let head = &prose[..q_pos];
    match last_sentence_boundary(head) {
        Some(i) => Some((&head[..i], head[i + 1..].trim())),
        None => Some(("", head.trim())),
    }
}

pub(in crate::retrieval) fn token_is_temporal_word(w: &str) -> bool {
    MONTHS.contains(&w)
        || WEEKDAYS.contains(&w)
        || [
            "yesterday", "tomorrow", "ago", "today", "tonight",
            "week", "weeks", "weekend", "month", "months", "year", "years",
            "days", "spring", "summer", "autumn", "winter",
            "mon", "tue", "wed", "thu", "fri", "sat", "sun",
        ]
        .contains(&w)
        || (w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()))
}

pub(in crate::retrieval) fn pre_has_answer_payload(pre: &str) -> bool {
    if pre.chars().any(|c| c.is_ascii_digit()) {
        return true;
    }
    raw_tokens(pre).iter().any(|w| token_is_temporal_word(w))
}

pub(in crate::retrieval) fn content_term_ids(vocab: &mut Vocab, text: &str, stemming: bool) -> DetHashSet<u32> {
    let mut ids = DetHashSet::default();
    for w in raw_tokens(text).into_iter().filter(|w| is_content(w)) {
        ids.insert(vocab.intern(&w));
        if stemming {
            ids.insert(vocab.intern(&stem_term(&stem(&w))));
        }
    }
    ids
}

pub(in crate::retrieval) struct QuestionShape {
    pub(in crate::retrieval) pre_ids: DetHashSet<u32>,
    pub(in crate::retrieval) clause_ids: DetHashSet<u32>,
    pub(in crate::retrieval) is_pure: bool,
}

pub(in crate::retrieval) fn analyze_question(
    vocab: &mut Vocab,
    prose: &str,
    is_question: bool,
    stemming: bool,
    names: &DetHashSet<String>,
) -> QuestionShape {
    let lower = prose.to_ascii_lowercase();
    let legacy = is_question && info_question_start(lower.trim_start());
    let tag = {
        let t = lower.trim();
        t.ends_with("right?") || t.ends_with("yeah?") || t.ends_with("huh?")
    };
    let Some((pre, clause)) = terminal_question_spans(prose) else {
        return QuestionShape {
            pre_ids: DetHashSet::default(),
            clause_ids: DetHashSet::default(),
            is_pure: is_question && legacy && !tag,
        };
    };
    let stripped = stripped_question_clause(clause, names);
    let terminal_info = info_question_start(stripped);
    let payload = pre_has_answer_payload(pre);
    QuestionShape {
        pre_ids: content_term_ids(vocab, pre, stemming),
        clause_ids: content_term_ids(vocab, clause, stemming),
        is_pure: is_question && !tag && !payload && (legacy || terminal_info),
    }
}
