//! Tokenization, stemming, UK/US normalization, proper-entity tokens and date/deictic compilers.
//! Split out of the former single-file `retrieval.rs` with no logic change.

use crate::retrieval::*;


// =========================================================================
// LEXICONS (fixed, deterministic, domain-general)
// =========================================================================

/// Closed-class function words only. Open-class and domain tokens are not listed;
/// a token that is merely frequent is demoted from the corpus document frequency.
pub(in crate::retrieval) const STOPWORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "arent", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cant", "could", "couldnt", "did", "didnt", "do", "does", "doesnt",
    "doing", "dont", "down", "during", "each", "few", "for", "from", "further", "had", "hadnt",
    "has", "hasnt", "have", "havent", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "id", "if", "ill", "im", "in", "into", "is", "isnt", "it",
    "its", "itself", "ive", "just", "me", "more", "most", "my", "myself", "no", "nor", "not",
    "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "shouldnt", "so", "some", "such", "than", "that",
    "thats", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "theyre", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasnt", "we", "were", "werent", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "wont", "would", "wouldnt", "you", "your", "youre", "yours",
    "yourself", "yourselves", "also", "may", "might", "must", "shall", "us", "let",
];

// hardcode-scan: closed-class (pronouns/anaphors)
pub(in crate::retrieval) const ANAPHORS: &[&str] = &[
    "it", "that", "they", "them", "he", "she", "him", "her", "there", "one", "this", "those",
    "these", "its", "their",
];

pub(in crate::retrieval) const MONTHS: [&str; 12] = [
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december",
];
pub(in crate::retrieval) const WEEKDAYS: [&str; 7] = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
];

pub(in crate::retrieval) fn stopwords() -> &'static DetHashSet<&'static str> {
    static S: OnceLock<DetHashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| STOPWORDS.iter().copied().collect())
}

/// Corpus frequency, not a word list. A token is framing when it occurs in at
/// least half the turns and the index is large enough for that rate to be a
/// frequency rather than a one-document accident.
pub(in crate::retrieval) fn corpus_frame(n_docs: usize, df: u32) -> bool {
    n_docs >= 32 && (df as usize) * 2 >= n_docs
}

// =========================================================================
// TOKENISATION + PROPER ENTITIES + DETERMINISTIC STEMMER
// =========================================================================

pub(in crate::retrieval) fn clean_contraction(raw_lower: &str) -> &str {
    let trimmed = raw_lower.trim_matches(|c: char| c == '\'' || c == '’');
    if let Some(stem_part) = trimmed.strip_suffix("'s").or_else(|| trimmed.strip_suffix("’s")) {
        return stem_part;
    }
    if let Some(stem_part) = trimmed.strip_suffix("'ve").or_else(|| trimmed.strip_suffix("’ve")) {
        return stem_part;
    }
    if let Some(stem_part) = trimmed.strip_suffix("'re").or_else(|| trimmed.strip_suffix("’re")) {
        return stem_part;
    }
    if let Some(stem_part) = trimmed.strip_suffix("'ll").or_else(|| trimmed.strip_suffix("’ll")) {
        return stem_part;
    }
    if let Some(stem_part) = trimmed.strip_suffix("'m").or_else(|| trimmed.strip_suffix("’m")) {
        return stem_part;
    }
    if let Some(stem_part) = trimmed.strip_suffix("'d").or_else(|| trimmed.strip_suffix("’d")) {
        return stem_part;
    }
    trimmed
}

pub(in crate::retrieval) fn collapse_thousands_commas(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    for (i, ch) in text.char_indices() {
        if ch == ','
            && i > 0
            && bytes[i - 1].is_ascii_digit()
            && i + 3 < bytes.len()
            && bytes[i + 1].is_ascii_digit()
            && bytes[i + 2].is_ascii_digit()
            && bytes[i + 3].is_ascii_digit()
            && (i + 4 == bytes.len() || !bytes[i + 4].is_ascii_digit())
        {
            continue;
        }
        out.push(ch);
    }
    out
}

/// Strip an ASCII `v` only when it introduces a dotted version (`v5.3.0` -> `5.3.0`,
/// `(v2.1)` -> `(2.1)`). The `v` must be at the start or after a non-alphanumeric,
/// and must be followed by digits, a dot, and a digit. Bare `v8` / `v2` / `v3` stay
/// intact, and `dev5.3` is left alone because its `v` is inside a word.
pub(in crate::retrieval) fn strip_dotted_version_v(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut prev_non_alnum = true;
    for (i, ch) in text.char_indices() {
        if ch == 'v' && prev_non_alnum {
            let rest = &bytes[i + 1..];
            let mut j = 0;
            while j < rest.len() && rest[j].is_ascii_digit() {
                j += 1;
            }
            if j > 0 && j + 1 < rest.len() && rest[j] == b'.' && rest[j + 1].is_ascii_digit() {
                continue;
            }
        }
        out.push(ch);
        prev_non_alnum = !ch.is_alphanumeric();
    }
    out
}

pub(in crate::retrieval) fn emit_token(out: &mut Vec<String>, tok: &str) {
    out.push(tok.to_string());
}

pub(in crate::retrieval) fn raw_tokens(text: &str) -> Vec<String> {
    let lower = strip_dotted_version_v(&collapse_thousands_commas(&text.to_lowercase()));
    let mut out = Vec::new();
    for piece in lower.split(|c: char| !c.is_alphanumeric() && c != '\'' && c != '’') {
        if piece.is_empty() {
            continue;
        }
        let cleaned = clean_contraction(piece);
        if cleaned.contains('\'') || cleaned.contains('’') {
            if cleaned.ends_with("n't") || cleaned.ends_with("n’t") {
                let joined: String = cleaned.chars().filter(|&c| c != '\'' && c != '’').collect();
                if !joined.is_empty() {
                    emit_token(&mut out, &joined);
                }
            } else {
                for sub in cleaned.split(|c: char| c == '\'' || c == '’') {
                    if !sub.is_empty() {
                        emit_token(&mut out, sub);
                    }
                }
            }
        } else if !cleaned.is_empty() {
            emit_token(&mut out, cleaned);
        }
    }
    // Preserve compact single-digit ratio / meeting tokens (`1:1`, `2:1`, `3:2`) that would
    // otherwise split into 1-char digits dropped by `is_content`.
    for ws in lower.split_whitespace() {
        let trimmed = ws.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != ':');
        let bytes = trimmed.as_bytes();
        if bytes.len() == 3
            && bytes[0].is_ascii_digit()
            && bytes[1] == b':'
            && bytes[2].is_ascii_digit()
        {
            emit_token(&mut out, trimmed);
        }
    }
    out
}

pub(in crate::retrieval) fn is_content(w: &str) -> bool {
    w.chars().count() > 1 && !stopwords().contains(w)
}

/// Closed-class words that must not become proper-name tokens, plus the
/// date deictics that are calendar words rather than names.
// hardcode-scan: closed-class (question words and function words)
pub(in crate::retrieval) const STARTER_STOP: &[&str] = &[
    "what", "when", "where", "why", "which", "who", "whom", "whose", "how",
    "are", "is", "was", "were", "do", "does", "did", "can", "could", "will",
    "would", "should", "shall", "may", "might", "have", "has", "had", "the",
    "this", "that", "these", "those", "there", "here", "no",
    "if", "so", "but", "and", "or",
    "my", "your", "our", "his", "her", "their", "its", "i", "we", "you", "they", "he", "she", "it",
    "also", "just", "now", "then", "still", "even",
    "however", "though", "although", "because", "since", "after", "before", "during", "while",
    "today", "yesterday", "tomorrow", "tonight", "always", "never", "let",
    "not", "only", "very",
    "another", "other", "more", "some", "many", "most", "every", "each", "both", "all", "any",
    "from", "with", "into", "over", "under", "about", "again", "once", "twice",
];

pub(in crate::retrieval) const TITLE_CONNECTORS: &[&str] = &["and", "of", "the", "for", "in", "on", "at"];

/// Extract leading document title/header line when a multi-line passage starts with a compact
/// header (`"{Title}\n{Body}"`, excluding trailing bracketed attachment/caption blocks), stripping
/// trailing parenthetical disambiguators (`"Armada (novel)"` -> `"Armada"`).
pub(in crate::retrieval) fn prose_without_blip(text: &str) -> &str {
    split_attachment_caption_raw(text).0
}

/// Strip a trailing structured provenance notation (`" ->-> 1,25"`) without touching prose text.
/// Only strips when the suffix after `"->->"` consists solely of digits, commas, periods, or whitespace.
pub(in crate::retrieval) fn strip_trailing_provenance_tag(text: &str) -> &str {
    if let Some((head, tail)) = text.rsplit_once("->->") {
        let trimmed_tail = tail.trim();
        if !trimmed_tail.is_empty()
            && trimmed_tail.len() <= 24
            && trimmed_tail
                .chars()
                .all(|c| c.is_ascii_digit() || matches!(c, ',' | '.' | '-' | ' ' | '\t'))
            && trimmed_tail.chars().any(|c| c.is_ascii_digit())
        {
            return head.trim_end();
        }
    }
    text
}

// hardcode-scan: closed-class (generic multimodal attachment/caption block headers)
fn split_attachment_caption_raw(text: &str) -> (&str, Option<&str>) {
    let mut search_from = 0usize;
    while let Some(rel) = text[search_from..].find("\n[") {
        let nl_idx = search_from + rel;
        let after_bracket = &text[nl_idx + 2..];
        if let Some(close_rel) = after_bracket.find(']') {
            let tag = after_bracket[..close_rel].trim();
            if !tag.is_empty() && tag.len() <= 32 {
                let tag_low = tag.to_ascii_lowercase();
                let is_media_tag = matches!(
                    tag_low.as_str(),
                    "caption" | "image" | "photo" | "attachment" | "alt"
                ) || tag_low.ends_with(" caption")
                    || tag_low.ends_with(" alt")
                    || tag_low.ends_with(" attachment");
                if is_media_tag {
                    let body = &text[..nl_idx];
                    let cap = after_bracket[close_rel + 1..].trim();
                    return (body, (!cap.is_empty()).then_some(cap));
                }
            }
        }
        search_from = nl_idx + 2;
    }
    (text, None)
}

pub(in crate::retrieval) fn split_blip_caption(text: &str) -> (&str, Option<&str>) {
    let (body, cap) = split_attachment_caption_raw(text);
    (strip_trailing_provenance_tag(body), cap)
}

pub(in crate::retrieval) fn extract_header_titles(text: &str) -> Option<(&str, &str)> {
    let prose = prose_without_blip(text);
    let (first_line, rest) = prose.split_once('\n')?;
    if rest.trim().is_empty() {
        return None;
    }
    let header = first_line.trim();
    if header.is_empty()
        || header.len() > 96
        || header.starts_with('[')
        || header.starts_with("```")
        || header.ends_with(['.', '?', '!', ';'])
    {
        return None;
    }
    let word_count = header.split_whitespace().count();
    if !(1..=10).contains(&word_count) {
        return None;
    }
    if !header.chars().next().is_some_and(|c| c.is_ascii_uppercase() || c.is_ascii_digit()) {
        return None;
    }
    let core = match header.find(" (") {
        Some(idx) if header.ends_with(')') && idx > 0 => header[..idx].trim(),
        _ => header,
    };
    if core.is_empty() { None } else { Some((header, core)) }
}

pub(in crate::retrieval) fn extract_header_title(text: &str) -> Option<&str> {
    extract_header_titles(text).map(|(_, core)| core)
}

pub(in crate::retrieval) fn norm_span_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

pub(in crate::retrieval) fn hash_u32_slice(tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &tok in tokens {
        h ^= u64::from(tok);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Extract contiguous capitalized proper-noun entity phrases ("\u{1}ent:local_h", "\u{1}ent:kiss_and_tell")
/// plus mid-clause single-token proper names ("\u{1}ent:ghana", "\u{1}ent:melanie") and header titles.
/// Preserves short single-letter entities ("H" in "Local H"), capitalized prepositions ("For Against"),
/// and title-case connectors ("Kiss and Tell", "Lord of the Rings").
pub(in crate::retrieval) fn proper_entity_tokens(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if let Some(core_title) = extract_header_title(text) {
        let parts: Vec<String> = core_title
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect();
        if !parts.is_empty() && parts.len() <= 6 {
            out.push(format!("\u{1}ent:{}", parts.join("_")));
        }
    }

    let mut run: Vec<String> = Vec::new();
    let mut run_started_clause = true;
    let mut at_clause_start;

    let flush_run = |run: &mut Vec<String>, started_clause: bool, out: &mut Vec<String>| {
        // Trim trailing lowercase connectors if the run ended on a preposition like "Kiss and"
        while run.last().is_some_and(|s| s.starts_with('\u{2}')) {
            run.pop();
        }
        if run.len() >= 2 {
            for part in run.iter() {
                if !part.starts_with('\u{2}')
                    && part.len() >= 3
                    && !STARTER_STOP.contains(&part.as_str())
                    && !stopwords().contains(part.as_str())
                    && !MONTHS.contains(&part.as_str())
                    && !WEEKDAYS.contains(&part.as_str())
                {
                    out.push(format!("\u{1}ent:{part}"));
                }
            }
            for win in run.windows(2) {
                if !win[0].starts_with('\u{2}') && !win[1].starts_with('\u{2}') {
                    out.push(format!("\u{1}ent:{}_{}", win[0], win[1]));
                }
            }
            if run.len() <= 4 {
                let clean_parts: Vec<&str> = run.iter().map(|s| s.trim_start_matches('\u{2}')).collect();
                out.push(format!("\u{1}ent:{}", clean_parts.join("_")));
            }
        } else if run.len() == 1 && !started_clause {
            let tok = &run[0];
            if tok.len() >= 3
                && !STARTER_STOP.contains(&tok.as_str())
                && !stopwords().contains(tok.as_str())
                && !MONTHS.contains(&tok.as_str())
                && !WEEKDAYS.contains(&tok.as_str())
            {
                out.push(format!("\u{1}ent:{tok}"));
            }
        }
        run.clear();
    };

    for line in text.lines() {
        at_clause_start = true;
        let mut in_mid_quote = false;
        for raw_word in line.split_whitespace() {
            let ends_hard_punct = raw_word.ends_with(['.', '?', '!', ';', ':']);
            let ends_comma = raw_word.ends_with(',');
            let starts_quote = raw_word.starts_with(['"', '\'', '“', '‘', '(', '[']);
            let closes_quote = raw_word.len() >= 2
                && raw_word
                    .trim_end_matches([',', '.', ';', ':', '!', '?'])
                    .ends_with(['"', '\'', '”', '’', ')', ']']);
            if starts_quote && !at_clause_start {
                in_mid_quote = true;
            }
            let ends_clause = ends_hard_punct || (ends_comma && !in_mid_quote);
            let w = raw_word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '’' && c != '-');
            if w.is_empty() {
                flush_run(&mut run, run_started_clause, &mut out);
                in_mid_quote = false;
                at_clause_start = true;
                continue;
            }
            let clean = clean_contraction(w);
            let first_upper = clean.chars().next().is_some_and(|c| c.is_ascii_uppercase());
            let all_upper_acronym = clean.len() >= 2
                && clean.len() <= 6
                && clean.chars().all(|c| c.is_ascii_uppercase());
            let lower = clean.to_lowercase();
            let norm: String = lower.chars().filter(|c| c.is_alphanumeric()).collect();
            if first_upper && !norm.is_empty() && !STARTER_STOP.contains(&norm.as_str()) {
                if run.is_empty() {
                    run_started_clause = at_clause_start && !all_upper_acronym;
                }
                run.push(norm);
            } else if !run.is_empty() && TITLE_CONNECTORS.contains(&norm.as_str()) {
                // Allow up to 2 title connectors inside an active proper-noun chain (e.g. "Lord of the Rings").
                let trailing_conns = run.iter().rev().take_while(|s| s.starts_with('\u{2}')).count();
                if trailing_conns < 2 {
                    run.push(format!("\u{2}{norm}"));
                } else {
                    flush_run(&mut run, run_started_clause, &mut out);
                }
            } else {
                flush_run(&mut run, run_started_clause, &mut out);
            }
            if closes_quote {
                in_mid_quote = false;
            }
            if ends_clause || closes_quote {
                flush_run(&mut run, run_started_clause, &mut out);
                at_clause_start = ends_hard_punct;
            } else {
                at_clause_start = false;
            }
        }
        flush_run(&mut run, run_started_clause, &mut out);
    }
    out
}
