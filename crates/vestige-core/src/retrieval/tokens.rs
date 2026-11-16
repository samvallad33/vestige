//! Tokenization, stemming, UK/US normalization, proper-entity tokens and date/deictic compilers.
//! Split out of the former single-file `retrieval.rs` with no logic change.

use super::*;

// =========================================================================
// LEXICONS (fixed, deterministic, domain-general)
// =========================================================================

/// Closed-class function words only. Open-class and domain tokens are not listed;
/// a token that is merely frequent is demoted from the corpus document frequency.
pub(super) const STOPWORDS: &[&str] = &[
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
pub(super) const ANAPHORS: &[&str] = &[
    "it", "that", "they", "them", "he", "she", "him", "her", "there", "one", "this", "those",
    "these", "its", "their",
];

pub(super) const MONTHS: [&str; 12] = [
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december",
];
pub(super) const WEEKDAYS: [&str; 7] = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
];

pub(super) fn stopwords() -> &'static DetHashSet<&'static str> {
    static S: OnceLock<DetHashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| STOPWORDS.iter().copied().collect())
}

/// Corpus frequency, not a word list. A token is framing when it occurs in at
/// least half the turns and the index is large enough for that rate to be a
/// frequency rather than a one-document accident.
pub(super) fn corpus_frame(n_docs: usize, df: u32) -> bool {
    n_docs >= 32 && (df as usize) * 2 >= n_docs
}

// =========================================================================
// TOKENISATION + PROPER ENTITIES + DETERMINISTIC STEMMER
// =========================================================================

pub(super) fn clean_contraction(raw_lower: &str) -> &str {
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

pub(super) fn collapse_thousands_commas(text: &str) -> String {
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
pub(super) fn strip_dotted_version_v(text: &str) -> String {
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

pub(super) fn emit_token(out: &mut Vec<String>, tok: &str) {
    out.push(tok.to_string());
}

pub(super) fn raw_tokens(text: &str) -> Vec<String> {
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
    out
}

pub(super) fn is_content(w: &str) -> bool {
    w.chars().count() > 1 && !stopwords().contains(w)
}

/// Closed-class words that must not become proper-name tokens, plus the
/// date deictics that are calendar words rather than names.
// hardcode-scan: closed-class (question words and function words)
pub(super) const STARTER_STOP: &[&str] = &[
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

pub(super) const TITLE_CONNECTORS: &[&str] = &["and", "of", "the", "for", "in", "on", "at"];

/// Extract leading document title/header line when a multi-line passage starts with a compact
/// header (`"{Title}\n{Body}"`, excluding `[BLIP caption]`), stripping trailing parenthetical
/// disambiguators (`"Armada (novel)"` -> `"Armada"`).
pub(super) fn prose_without_blip(text: &str) -> &str {
    text.split_once("\n[BLIP caption]").map(|(body, _)| body).unwrap_or(text)
}

pub(super) fn split_blip_caption(text: &str) -> (&str, Option<&str>) {
    if let Some((body, cap)) = text.split_once("\n[BLIP caption]") {
        let trimmed = cap.trim();
        (body, (!trimmed.is_empty()).then_some(trimmed))
    } else {
        (text, None)
    }
}

pub(super) fn extract_header_titles(text: &str) -> Option<(&str, &str)> {
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

pub(super) fn extract_header_title(text: &str) -> Option<&str> {
    extract_header_titles(text).map(|(_, core)| core)
}

pub(super) fn norm_span_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

pub(super) fn hash_u32_slice(tokens: &[u32]) -> u64 {
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
pub(super) fn proper_entity_tokens(text: &str) -> Vec<String> {
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
        for raw_word in line.split_whitespace() {
            let ends_clause = raw_word.ends_with(['.', '?', '!', ',', ';', ':']);
            let starts_quote = raw_word.starts_with(['"', '\'', '“', '‘', '(', '[']);
            let w = raw_word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '’' && c != '-');
            if w.is_empty() {
                flush_run(&mut run, run_started_clause, &mut out);
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
                    run_started_clause = (at_clause_start || starts_quote) && !all_upper_acronym;
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
            if ends_clause {
                flush_run(&mut run, run_started_clause, &mut out);
                at_clause_start = raw_word.ends_with(['.', '?', '!', ';', ':']);
            } else {
                at_clause_start = false;
            }
        }
        flush_run(&mut run, run_started_clause, &mut out);
    }
    out
}

// UK -> US orthography is closed-class spelling normalization, kept on purpose.
pub(super) fn uk_exact(w: &str) -> Option<&'static str> {
    Some(match w {
        "travelling" => "traveling",
        "travelled" => "traveled",
        "traveller" => "traveler",
        "travellers" => "travelers",
        "cancelled" => "canceled",
        "cancelling" => "canceling",
        "modelling" => "modeling",
        "modelled" => "modeled",
        "labelled" => "labeled",
        "labelling" => "labeling",
        "jewellery" => "jewelry",
        "aluminium" => "aluminum",
        "programme" => "program",
        "programmes" => "programs",
        "grey" => "gray",
        "greys" => "grays",
        "greyed" => "grayed",
        "greying" => "graying",
        "greyer" => "grayer",
        "greyest" => "grayest",
        "greyish" => "grayish",
        "catalogue" => "catalog",
        "catalogues" => "catalogs",
        "catalogued" => "cataloged",
        "cataloguing" => "cataloging",
        "dialogue" => "dialog",
        "dialogues" => "dialogs",
        "dialogued" => "dialoged",
        "dialoguing" => "dialoging",
        "practise" => "practice",
        "practises" => "practices",
        "practised" => "practiced",
        "practising" => "practicing",
        "licencing" => "licensing",
        _ => return None,
    })
}

pub(super) fn uk_suffix(w: &str) -> Option<String> {
    // hardcode-scan: closed-class (UK/US spelling normalization)
    const PAIRS: &[(&str, &str)] = &[
        ("centring", "centering"),
        ("centred", "centered"),
        ("centres", "centers"),
        ("centre", "center"),
        ("theatres", "theaters"),
        ("theatre", "theater"),
        ("metres", "meters"),
        ("metre", "meter"),
        ("litres", "liters"),
        ("litre", "liter"),
        ("fibres", "fibers"),
        ("fibre", "fiber"),
        ("defencelessly", "defenselessly"),
        ("defenceless", "defenseless"),
        ("defences", "defenses"),
        ("defence", "defense"),
        ("offencelessly", "offenselessly"),
        ("offenceless", "offenseless"),
        ("offences", "offenses"),
        ("offence", "offense"),
        ("licences", "licenses"),
        ("licenced", "licensed"),
        ("licencee", "licensee"),
        ("licence", "license"),
    ];
    for (uk, us) in PAIRS {
        if let Some(prefix) = w.strip_suffix(uk) {
            if prefix.is_empty()
                || (prefix.len() <= 12 && prefix.bytes().all(|b| b.is_ascii_alphabetic()))
            {
                return Some(format!("{prefix}{us}"));
            }
        }
    }
    None
}

pub(super) fn our_tail(tail: &str) -> bool {
    matches!(
        tail,
        "" | "s"
            | "ed"
            | "ing"
            | "ings"
            | "er"
            | "ers"
            | "able"
            | "ite"
            | "ites"
            | "ful"
            | "fully"
            | "less"
            | "al"
            | "ally"
            | "ation"
            | "hood"
            | "hoods"
    )
}

pub(super) fn map_our(w: &str) -> Option<String> {
    // hardcode-scan: closed-class (UK/US spelling normalization)
    const ROOTS: &[&str] = &[
        "neighbour", "behaviour", "flavour", "honour", "colour", "labour", "humour", "favour",
    ];
    let mut best: Option<(usize, String)> = None;
    for root in ROOTS {
        let mut from = 0;
        while from < w.len() {
            let Some(rel) = w[from..].find(root) else { break };
            let i = from + rel;
            let prefix = &w[..i];
            let tail = &w[i + root.len()..];
            let prefix_ok = prefix.is_empty() || prefix.bytes().all(|b| b.is_ascii_alphabetic());
            if prefix_ok && our_tail(tail) && best.as_ref().is_none_or(|(n, _)| tail.len() > *n) {
                let head = &root[..root.len() - 3];
                best = Some((tail.len(), format!("{prefix}{head}or{tail}")));
            }
            from = i + 1;
        }
    }
    best.map(|(_, s)| s)
}

pub(super) fn map_yse(w: &str) -> Option<String> {
    if w.len() < 6 {
        return None;
    }
    let (base, us) = if let Some(b) = w.strip_suffix("ysing") {
        (b, "yzing")
    } else if let Some(b) = w.strip_suffix("ysed") {
        (b, "yzed")
    } else if let Some(b) = w.strip_suffix("yser") {
        (b, "yzer")
    } else if let Some(b) = w.strip_suffix("yse") {
        (b, "yze")
    } else {
        return None;
    };
    // hardcode-scan: closed-class (UK/US spelling normalization)
    const BASES: &[&str] = &[
        "psychoanal", "electrol", "breathal", "hydrol", "paral", "catal", "pyrol", "anal", "dial",
    ];
    let ok = BASES.iter().any(|b| {
        base == *b
            || (base.ends_with(b)
                && base.len() > b.len()
                && base[..base.len() - b.len()].bytes().all(|c| c.is_ascii_alphabetic()))
    });
    if base.is_empty() || !ok {
        return None;
    }
    Some(format!("{base}{us}"))
}

pub(super) fn ise_blocked(lemma: &str) -> bool {
    if lemma.len() < 4 {
        return true;
    }
    if lemma.ends_with("wise")
        || lemma.ends_with("prise")
        || lemma.ends_with("vise")
        || lemma.ends_with("oise")
        || lemma.ends_with("uise")
        || lemma.ends_with("aise")
    {
        return true;
    }
    matches!(
        lemma,
        "promise"
            | "surprise"
            | "exercise"
            | "advise"
            | "revise"
            | "compromise"
            | "enterprise"
            | "supervise"
            | "improvise"
            | "paradise"
            | "otherwise"
            | "clockwise"
            | "likewise"
            | "franchise"
            | "merchandise"
            | "disguise"
            | "bruise"
            | "cruise"
            | "raise"
            | "praise"
            | "noise"
            | "poise"
            | "choose"
            | "chose"
            | "house"
            | "mouse"
            | "practise"
            | "advertise"
            | "apprise"
            | "arise"
            | "chastise"
            | "circumcise"
            | "comprise"
            | "demise"
            | "despise"
            | "devise"
            | "disfranchise"
            | "enfranchise"
            | "excise"
            | "incise"
            | "premise"
            | "reprise"
            | "surmise"
            | "televise"
            | "precise"
            | "concise"
            | "expertise"
            | "treatise"
            | "sunrise"
            | "moonrise"
            | "rise"
            | "uprise"
            | "anise"
            | "chemise"
            | "valise"
            | "cerise"
            | "mortise"
            | "tortoise"
            | "porpoise"
            | "liaise"
    )
}

pub(super) fn map_ise(w: &str) -> Option<String> {
    if w.len() < 6 || !w.is_ascii() {
        return None;
    }
    let (base, us) = if let Some(b) = w.strip_suffix("isations") {
        (b, "izations")
    } else if let Some(b) = w.strip_suffix("isation") {
        (b, "ization")
    } else if let Some(b) = w.strip_suffix("ising") {
        (b, "izing")
    } else if let Some(b) = w.strip_suffix("ised") {
        (b, "ized")
    } else if let Some(b) = w.strip_suffix("ises") {
        // "irises" / "crises" are Latin plurals (base "ir" / "cr"), not UK -ise verbs.
        if b.len() < 4 {
            return None;
        }
        (b, "izes")
    } else if let Some(b) = w.strip_suffix("ise") {
        (b, "ize")
    } else {
        return None;
    };
    if base.is_empty() || ise_blocked(&format!("{base}ise")) {
        return None;
    }
    Some(format!("{base}{us}"))
}

pub(super) fn uk_to_us(w: &str) -> Option<String> {
    if let Some(exact) = uk_exact(w) {
        return Some(exact.to_string());
    }
    if let Some(hit) = uk_suffix(w) {
        return Some(hit);
    }
    if let Some(hit) = map_our(w) {
        return Some(hit);
    }
    if let Some(hit) = map_yse(w) {
        return Some(hit);
    }
    map_ise(w)
}

pub(super) fn classic_stem(w: &str) -> String {
    if w.len() <= 3 || !w.is_ascii() {
        return w.to_string();
    }
    // hardcode-scan: closed-class (inflectional suffixes)
    let rules: &[(&str, &str, usize)] = &[
        ("ingly", "", 4),
        ("edly", "", 4),
        ("ments", "", 4),
        ("ment", "", 4),
        ("ness", "", 4),
        ("tions", "t", 3),
        ("tion", "t", 3),
        ("ings", "", 3),
        ("ing", "", 3),
        ("ies", "y", 2),
        ("ied", "y", 2),
        ("ers", "", 4),
        ("er", "", 4),
        ("est", "", 4),
        ("ed", "", 3),
        ("es", "", 3),
        ("ly", "", 4),
        ("s", "", 3),
    ];
    let mut out = w.to_string();
    for (suf, rep, min) in rules {
        if let Some(base) = w.strip_suffix(suf) {
            if base.len() >= *min {
                if *suf == "s" && (base.ends_with('s') || base.ends_with('u') || base.ends_with('i')) {
                    break;
                }
                out = format!("{base}{rep}");
                break;
            }
        }
    }
    let b = out.as_bytes();
    if b.len() >= 4 {
        let (x, y) = (b[b.len() - 1], b[b.len() - 2]);
        if x == y && !b"aeiouylsz".contains(&x) {
            out.pop();
        }
    }
    if out.len() > 3 && out.ends_with('e') {
        out.pop();
    }
    out
}

/// Suffix stem plus closed-class UK -> US spelling normalization. No
/// abbreviation map and no irregular-verb vocabulary.
pub fn stem(word: &str) -> String {
    let lower = word.to_lowercase();
    let normalized = uk_to_us(&lower).unwrap_or(lower);
    classic_stem(&normalized)
}

// =========================================================================
// DATE PARSING + DEICTIC TIME COMPILER
// =========================================================================

/// Parse the free-form date strings used by real conversation logs into Unix seconds (UTC).
pub fn parse_date(s: &str) -> i64 {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return dt.timestamp();
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return dt.and_utc().timestamp();
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return d.and_hms_opt(0, 0, 0).map(|dt| dt.and_utc().timestamp()).unwrap_or(0);
    }
    let cleaned: String = s
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ':' { c.to_ascii_lowercase() } else { ' ' })
        .collect();
    let toks: Vec<&str> = cleaned.split_whitespace().collect();
    let (mut year, mut month, mut hour, mut minute) = (None, None, 0u32, 0u32);
    let mut nums: Vec<u32> = Vec::new();
    for (i, t) in toks.iter().enumerate() {
        if let Some((h, m)) = t.split_once(':') {
            let mut hh: u32 = h.parse().unwrap_or(0);
            minute = m.parse().unwrap_or(0);
            match toks.get(i + 1) {
                Some(&"pm") if hh < 12 => hh += 12,
                Some(&"am") if hh == 12 => hh = 0,
                _ => {}
            }
            hour = hh;
        } else if t.len() == 4 && t.chars().all(|c| c.is_ascii_digit()) {
            year = t.parse::<i32>().ok();
        } else if t.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(n) = t.parse::<u32>() {
                nums.push(n);
            }
        } else if t.len() >= 3 {
            if let Some(m) = MONTHS.iter().position(|mn| mn.starts_with(&t[..3]) && mn.starts_with(t.get(..t.len().min(mn.len())).unwrap_or(""))) {
                month = Some(m as u32 + 1);
            }
        }
    }
    let (m, d) = match month {
        Some(m) => (m, nums.first().copied().unwrap_or(1)),
        None => (nums.first().copied().unwrap_or(1), nums.get(1).copied().unwrap_or(1)),
    };
    let Some(y) = year else { return 0 };
    chrono::NaiveDate::from_ymd_opt(y, m, d)
        .and_then(|date| date.and_hms_opt(hour.min(23), minute.min(59), 0))
        .map(|dt| dt.and_utc().timestamp())
        .unwrap_or(0)
}

pub(super) fn season_code(month: u32) -> &'static str {
    match month {
        3..=5 => "SP",
        6..=8 => "SU",
        9..=11 => "FA",
        _ => "WI",
    }
}

pub(super) fn calendar_tokens(ts: i64) -> Vec<String> {
    use chrono::Datelike;
    let Some(dt) = chrono::DateTime::from_timestamp(ts, 0) else { return Vec::new() };
    let d = dt.date_naive();
    let sc = season_code(d.month());
    let sy = if d.month() == 12 { d.year() + 1 } else { d.year() };
    vec![
        d.day().to_string(),
        MONTHS[d.month0() as usize].to_string(),
        d.year().to_string(),
        WEEKDAYS[d.weekday().num_days_from_monday() as usize].to_string(),
        format!("__DATE__{}_{:02}_{:02}", d.year(), d.month(), d.day()),
        format!("__YM__{}_{:02}", d.year(), d.month()),
        format!("__MONTH__{:02}", d.month()),
        format!("__MD__{:02}_{:02}", d.month(), d.day()),
        format!("__{}__{:02}", month_band(d.day()), d.month()),
        format!("__YEAR__{}", d.year()),
        format!("__SEASON__{sc}"),
        format!("__SY__{}_{sc}", d.year()),
        format!("__SY__{sy}_{sc}"),
    ]
}

pub(super) fn month_band(day: u32) -> &'static str {
    if day <= 10 { "EARLY" } else if day >= 21 { "LATE" } else { "MID" }
}

pub(super) fn month_index(tok: &str) -> Option<(u32, bool)> {
    match tok {
        "january" | "jan" => Some((1, false)),
        "february" | "feb" => Some((2, false)),
        "march" => Some((3, true)),
        "mar" => Some((3, false)),
        "april" | "apr" => Some((4, false)),
        "may" => Some((5, true)),
        "june" | "jun" => Some((6, false)),
        "july" | "jul" => Some((7, false)),
        "august" | "aug" => Some((8, false)),
        "september" | "sep" | "sept" => Some((9, false)),
        "october" | "oct" => Some((10, false)),
        "november" | "nov" => Some((11, false)),
        "december" | "dec" => Some((12, false)),
        _ => None,
    }
}

pub(super) fn parse_day_token(tok: &str) -> Option<u32> {
    let stripped = tok
        .strip_suffix("st")
        .or_else(|| tok.strip_suffix("nd"))
        .or_else(|| tok.strip_suffix("rd"))
        .or_else(|| tok.strip_suffix("th"))
        .unwrap_or(tok);
    if let Ok(n) = stripped.parse::<u32>() {
        if (1..=31).contains(&n) {
            return Some(n);
        }
    }
    match tok {
        "first" => Some(1),
        "second" => Some(2),
        "third" => Some(3),
        "fourth" | "forth" => Some(4),
        "fifth" => Some(5),
        "sixth" => Some(6),
        "seventh" => Some(7),
        "eighth" => Some(8),
        "ninth" => Some(9),
        "tenth" => Some(10),
        "eleventh" => Some(11),
        "twelfth" => Some(12),
        "thirteenth" => Some(13),
        "fourteenth" => Some(14),
        "fifteenth" => Some(15),
        "sixteenth" => Some(16),
        "seventeenth" => Some(17),
        "eighteenth" => Some(18),
        "nineteenth" => Some(19),
        "twentieth" => Some(20),
        "thirtieth" => Some(30),
        _ => None,
    }
}

/// Extract explicit calendar dates written directly inside prose/queries
/// (e.g. "1 May, 2022", "May 1st, 2022", "the 15th of March", "in May", "summer of 2023", "2022-05-01").
pub(super) fn inline_date_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let lower = text.to_lowercase();
    // Match ISO YYYY-MM-DD or YYYY/MM/DD
    for w in lower.split_whitespace() {
        let trimmed = w.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '/');
        if trimmed.len() == 10 {
            let b = trimmed.as_bytes();
            if (b.get(4) == Some(&b'-') && b.get(7) == Some(&b'-'))
                || (b.get(4) == Some(&b'/') && b.get(7) == Some(&b'/'))
            {
                let iso = trimmed.replace('/', "-");
                if let Ok(d) = chrono::NaiveDate::parse_from_str(&iso, "%Y-%m-%d") {
                    if let Some(dt) = d.and_hms_opt(0, 0, 0) {
                        out.extend(calendar_tokens(dt.and_utc().timestamp()));
                    }
                }
            }
        }
    }
    let orig_toks: Vec<&str> = text
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();
    let toks: Vec<String> = orig_toks.iter().map(|s| s.to_lowercase()).collect();

    // Closed-class temporal scaffold. Not a domain vocabulary: these are the
    // function words a calendar expression uses to attach a month name.
    // hardcode-scan: closed-class (temporal prepositions)
    const TEMP_PREPS: &[&str] = &[
        "in", "on", "of", "during", "since", "before", "after", "until", "by", "from", "through",
        "around", "this", "last", "next", "past", "every", "each", "late", "early", "mid", "middle",
    ];

    for i in 0..toks.len() {
        // A 4-digit token in the calendar range is a year. No domain blocklist.
        if toks[i].len() == 4 && toks[i].chars().all(|c| c.is_ascii_digit()) {
            if let Ok(y) = toks[i].parse::<i32>() {
                if (1900..=2099).contains(&y) {
                    out.push(format!("__YEAR__{y}"));
                }
            }
        }

        // Seasonal anchors ("summer of 2023", "in the winter", "last spring")
        let season = match toks[i].as_str() {
            "summer" => Some(("SU", false)),
            "winter" => Some(("WI", false)),
            "autumn" => Some(("FA", false)),
            "spring" => Some(("SP", true)),
            "fall" => Some(("FA", true)),
            _ => None,
        };
        if let Some((sc, ambig_season)) = season {
            let prev1 = i.checked_sub(1).map(|k| toks[k].as_str()).unwrap_or("");
            let prev2 = i.checked_sub(2).map(|k| toks[k].as_str()).unwrap_or("");
            let y_after = if toks.get(i + 1).map(String::as_str) == Some("of") {
                toks.get(i + 2)
            } else {
                toks.get(i + 1)
            }
            .and_then(|s| (s.len() == 4).then(|| s.parse::<i32>().ok()).flatten())
            .filter(|&y| (1900..=2099).contains(&y));

            if let Some(y) = y_after {
                out.push(format!("__SY__{y}_{sc}"));
                out.push(format!("__SEASON__{sc}"));
                out.push(format!("__YEAR__{y}"));
            } else if !ambig_season
                || matches!(prev1, "this" | "last" | "next" | "past" | "every" | "early" | "late" | "mid")
                || (prev1 == "the" && matches!(prev2, "in" | "during" | "since" | "before" | "after"))
            {
                out.push(format!("__SEASON__{sc}"));
            }
        }

        let Some((m, ambig_month)) = month_index(&toks[i]) else { continue };
        let band = i
            .checked_sub(1)
            .and_then(|k| match toks[k].as_str() {
                "early" | "earlier" => Some("EARLY"),
                "late" | "later" => Some("LATE"),
                "mid" | "middle" => Some("MID"),
                "of" if k >= 1 => match toks[k - 1].as_str() {
                    "beginning" | "start" => Some("EARLY"),
                    "middle" => Some("MID"),
                    "end" => Some("LATE"),
                    _ => None,
                },
                _ => None,
            });
        if let Some(b) = band {
            out.push(format!("__{b}__{m:02}"));
            out.push(format!("__MONTH__{m:02}"));
        }

        // "<day> [of] <month> [of] <year>" OR "<month> [the] <day> [of] <year>"
        let prev_num = i.checked_sub(1).and_then(|k| {
            if toks[k] == "of" && k >= 1 {
                parse_day_token(&toks[k - 1])
            } else {
                parse_day_token(&toks[k])
            }
        });
        let (next_num, day_end_idx) = if toks.get(i + 1).map(String::as_str) == Some("the") {
            (toks.get(i + 2).and_then(|s| parse_day_token(s)), i + 2)
        } else {
            (toks.get(i + 1).and_then(|s| parse_day_token(s)), i + 1)
        };
        let year_at = |idx: usize| -> Option<i32> {
            let cand = if toks.get(idx).map(String::as_str) == Some("of") {
                toks.get(idx + 1)
            } else {
                toks.get(idx)
            };
            cand.and_then(|s| (s.len() == 4).then(|| s.parse::<i32>().ok()).flatten())
                .filter(|&y| (1900..=2099).contains(&y))
        };
        let year_after_month = year_at(i + 1);
        let year_after_next = year_at(day_end_idx + 1);

        if let (Some(day), Some(y)) = (prev_num, year_after_month) {
            if let Some(dt) = chrono::NaiveDate::from_ymd_opt(y, m, day).and_then(|d| d.and_hms_opt(0, 0, 0)) {
                out.extend(calendar_tokens(dt.and_utc().timestamp()));
                continue;
            }
        }
        if let (Some(day), Some(y)) = (next_num, year_after_next) {
            if let Some(dt) = chrono::NaiveDate::from_ymd_opt(y, m, day).and_then(|d| d.and_hms_opt(0, 0, 0)) {
                out.extend(calendar_tokens(dt.and_utc().timestamp()));
                continue;
            }
        }
        if let Some(y) = year_after_month {
            out.push(format!("__YM__{y}_{m:02}"));
            out.push(format!("__MONTH__{m:02}"));
            out.push(format!("__YEAR__{y}"));
        } else if let Some(day) = prev_num.or(next_num) {
            out.push(format!("__MD__{m:02}_{day:02}"));
            out.push(format!("__MONTH__{m:02}"));
        } else if band.is_none() {
            // Bare month in prose/query ("in May", "during July", "last October").
            // Guard ambiguous "may"/"march" with preposition or capitalization + non-modal context.
            let prev1 = i.checked_sub(1).map(|k| toks[k].as_str()).unwrap_or("");
            let is_cap = orig_toks
                .get(i)
                .and_then(|s| s.chars().next())
                .is_some_and(|c| c.is_ascii_uppercase());
            if !ambig_month || TEMP_PREPS.contains(&prev1) || (is_cap && i > 0) {
                out.push(format!("__MONTH__{m:02}"));
            }
        }
    }
    out
}

pub(super) fn parse_qty_word(q: &str) -> i64 {
    match q {
        "a" | "an" | "one" => 1,
        "two" | "couple" => 2,
        "three" | "few" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        other => other.parse().unwrap_or(1),
    }
}

/// Resolve relative time expressions against the session timestamp into (token, relative_weight).
pub(super) fn deictic_weighted_tokens(raw: &[String], ts: i64) -> Vec<(String, f32)> {
    if ts == 0 {
        return Vec::new();
    }
    const DAY: i64 = 86_400;
    // (offset_seconds, weight, is_coarse_window)
    let mut specs: Vec<(i64, f32, bool)> = Vec::new();
    for (i, w) in raw.iter().enumerate() {
        let prev = if i > 0 { raw[i - 1].as_str() } else { "" };
        let next = raw.get(i + 1).map(String::as_str).unwrap_or("");
        match w.as_str() {
            "yesterday" => specs.push((-DAY, 1.0, false)),
            "night" if prev == "last" => specs.push((-DAY, 1.0, false)),
            "before" | "earlier" if prev == "day" => specs.push((-DAY, 0.9, false)),
            "before" if prev == "week" => specs.push((-7 * DAY, 0.9, true)),
            "before" if prev == "month" => specs.push((-30 * DAY, 0.9, true)),
            "before" if prev == "year" => specs.push((-365 * DAY, 0.9, true)),
            "after" | "later" if prev == "day" && (i < 2 || raw[i - 2] == "the" || raw[i - 2] == "a") => {
                specs.push((DAY, 0.9, false));
            }
            "after" | "following" if prev == "week" && (i < 2 || raw[i - 2] == "the" || raw[i - 2] == "a") => {
                specs.push((7 * DAY, 0.9, true));
            }
            "tomorrow" => specs.push((DAY, 1.0, false)),
            "today" | "tonight" => specs.push((0, 1.0, false)),
            "week" if matches!(prev, "last" | "past" | "previous") => specs.push((-7 * DAY, 1.0, true)),
            "week" if matches!(prev, "next" | "coming" | "following") => specs.push((7 * DAY, 1.0, true)),
            "week" if prev == "this" && i >= 2 && matches!(raw[i - 2].as_str(), "earlier" | "early") => {
                specs.push((-2 * DAY, 0.9, true));
            }
            "weekend" if matches!(prev, "last" | "this" | "past" | "previous") => specs.push((-2 * DAY, 1.0, true)),
            "weekend" if matches!(prev, "next" | "coming") => specs.push((5 * DAY, 0.9, true)),
            "month" if matches!(prev, "last" | "past" | "previous") => {
                specs.push((-30 * DAY, 1.0, true));
                specs.push((-15 * DAY, 0.85, false));
            }
            "month" if prev == "this" && i >= 2 && matches!(raw[i - 2].as_str(), "earlier" | "early") => {
                specs.push((-7 * DAY, 0.9, true));
            }
            "month" if matches!(prev, "next" | "coming" | "following") => specs.push((30 * DAY, 1.0, true)),
            "year" if matches!(prev, "last" | "past" | "previous") => specs.push((-365 * DAY, 1.0, true)),
            "year" if matches!(prev, "next" | "coming" | "following") => specs.push((365 * DAY, 1.0, true)),
            "ago" | "earlier" | "later" => {
                if matches!(prev, "days" | "weeks" | "months" | "years" | "day" | "week" | "month" | "year") {
                    let q = if i >= 3 && raw[i - 2] == "of" {
                        raw[i - 3].as_str()
                    } else if i >= 2 {
                        raw[i - 2].as_str()
                    } else {
                        ""
                    };
                    let n = parse_qty_word(q);
                    let (unit, coarse) = match prev.trim_end_matches('s') {
                        "day" => (1, n > 3),
                        "week" => (7, true),
                        "month" => (30, true),
                        _ => (365, true),
                    };
                    let dir = if w == "later" { 1 } else { -1 };
                    if n > 0 {
                        specs.push((dir * n * unit * DAY, 1.0, coarse));
                        if q == "few" && unit >= 7 {
                            specs.push((dir * (n - 1) * unit * DAY, 0.65, true));
                            specs.push((dir * (n + 1) * unit * DAY, 0.65, true));
                        }
                    }
                }
            }
            "days" | "weeks" | "months" | "years" if !matches!(next, "ago" | "earlier" | "later") => {
                // Handle "for 3 years now", "past 2 weeks", "last 3 months"
                let prep = if i >= 2 { raw[i - 2].as_str() } else { "" };
                if matches!(prep, "for" | "past" | "last" | "in") || next == "now" {
                    let n = parse_qty_word(prev);
                    let unit = match w.trim_end_matches('s') {
                        "day" => 1,
                        "week" => 7,
                        "month" => 30,
                        _ => 365,
                    };
                    if n > 0 && n <= 120 {
                        specs.push((-n * unit * DAY, 0.9, true));
                    }
                }
            }
            _ => {
                if matches!(prev, "last" | "this" | "next" | "past" | "previous" | "coming" | "following")
                    && WEEKDAYS.contains(&w.as_str())
                {
                    use chrono::Datelike;
                    if let Some(dt) = chrono::DateTime::from_timestamp(ts, 0) {
                        let today = dt.weekday().num_days_from_monday() as i64;
                        let target = WEEKDAYS.iter().position(|d| d == w).unwrap_or(0) as i64;
                        let is_future = matches!(prev, "next" | "coming" | "following");
                        let is_strict_past = matches!(prev, "last" | "past" | "previous");
                        let raw_back = (today - target + 7) % 7;
                        let raw_fwd = (target - today + 7) % 7;
                        let back = if is_strict_past && raw_back == 0 { 7 } else { raw_back };
                        let fwd = if is_future && raw_fwd == 0 { 7 } else { raw_fwd };
                        let delta = if is_future { fwd } else { -back };
                        specs.push((delta * DAY, 1.0, false));
                    }
                }
            }
        }
    }
    let mut out = Vec::new();
    for (off, w, coarse) in specs {
        for tok in calendar_tokens(ts + off) {
            out.push((tok, w));
        }
        // Soft +-2d halo on coarse multi-week / multi-month expressions (handles Feb 28 / Mar 1 boundary).
        if coarse {
            for d_off in [-2 * DAY, -DAY, DAY, 2 * DAY] {
                for tok in calendar_tokens(ts + off + d_off) {
                    if tok.starts_with("__DATE__")
                        || tok.starts_with("__YM__")
                        || tok.starts_with("__MONTH__")
                        || tok.starts_with("__MD__")
                    {
                        out.push((tok, w * 0.45));
                    }
                }
            }
        }
    }
    out
}
