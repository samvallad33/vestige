//! Normalization, surface tokenization, typed slots, and speaker nickname index.

use crate::retrieval::*;


pub(in crate::retrieval) fn stem_term(s: &str) -> String {
    format!("\u{1}stem:{s}")
}

/// Object tokens of a personal constraint. The marker is not enough; the thing
/// constrained has to be in the index or a later question cannot hit it.
/// Parse both compact `key:value` tokens and structured `Key: Value` clauses
/// (excluding URL schemes `http:`/`https:` and `[BLIP caption]`), emitting exact
/// `(slot_key, slot_value)` pairs so typed attributes affect ranking and supersession.
pub(in crate::retrieval) fn typed_slot_tokens(text: &str) -> Vec<(String, String)> {
    // URI schemes are protocol structure, not a topical vocabulary.
    const SKIP_KEYS: &[&str] = &["http", "https", "mailto", "file"];
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed_line = line.trim();
        if trimmed_line.starts_with('[') || trimmed_line.starts_with("```") {
            continue;
        }
        // Check structured `Key: Value` at line/clause head (up to 2 words for key, 1-2 words for value).
        if let Some((lhs, rhs)) = trimmed_line.split_once(':') {
            let key_words: Vec<&str> = lhs.split_whitespace().collect();
            if (1..=2).contains(&key_words.len())
                && key_words.iter().all(|w| w.len() >= 2 && w.chars().all(|c| c.is_ascii_alphabetic() || c == '_' || c == '-'))
            {
                let norm_key = key_words.join("_").to_lowercase();
                if !SKIP_KEYS.contains(&norm_key.as_str()) {
                    for val in rhs
                        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-' && c != '$')
                        .filter(|w| !w.is_empty())
                        .take(2)
                    {
                        let clean_val = val.trim_matches('$').to_lowercase();
                        if clean_val.len() >= 2 && !stopwords().contains(clean_val.as_str()) {
                            out.push((norm_key.clone(), clean_val));
                        }
                    }
                }
            }
        }
        // Also check inline `key:value` tokens without spaces.
        for raw in trimmed_line.split_whitespace() {
            let tok = raw.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != ':' && c != '_' && c != '-');
            let Some((key, value)) = tok.split_once(':') else { continue };
            let kl = key.to_lowercase();
            if kl.len() >= 2
                && kl.chars().all(|c| c.is_ascii_alphabetic())
                && !SKIP_KEYS.contains(&kl.as_str())
                && !value.is_empty()
                && value.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            {
                out.push((kl, value.to_lowercase()));
            }
        }
    }
    out
}

/// Pure byte-exact technical handle extractor for Retroactive Salience Backfill (RSB):
/// URLs, emails, commit SHAs (7..=40 hex with >=1 alpha + >=1 digit), issue refs (`PROJ-123`),
/// version tags (`v1.2.3`), and file paths (`dir/file.ext`). Zero word similarity.
pub(in crate::retrieval) fn extract_exact_technical_handles(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = DetHashSet::default();
    for raw_tok in text.split_whitespace() {
        let tok = raw_tok.trim_matches(|c: char| matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | ',' | ';' | '"' | '\'' | '`'));
        let tok_clean = tok.trim_end_matches('.');
        if tok_clean.len() < 3 {
            continue;
        }
        // 1. URL
        if tok_clean.starts_with("http://") || tok_clean.starts_with("https://") {
            let h = format!("\u{1}h:url:{}", tok_clean.to_lowercase());
            if seen.insert(h.clone()) {
                out.push(h);
            }
            continue;
        }
        // 2. Email
        if let Some((local, dom)) = tok_clean.split_once('@') {
            if !local.is_empty()
                && dom.contains('.')
                && !dom.starts_with('.')
                && !dom.ends_with('.')
                && local.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '%' | '+' | '-'))
                && dom.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-'))
            {
                let h = format!("\u{1}h:email:{}", tok_clean.to_lowercase());
                if seen.insert(h.clone()) {
                    out.push(h);
                }
                continue;
            }
        }
        // 3. IssueRef (`ABC-123`)
        if let Some((proj, num)) = tok_clean.split_once('-') {
            if (2..=10).contains(&proj.len())
                && proj.chars().next().is_some_and(|c| c.is_ascii_uppercase())
                && proj.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
                && !num.is_empty()
                && num.len() <= 7
                && num.chars().all(|c| c.is_ascii_digit())
            {
                let h = format!("\u{1}h:issue:{}", tok_clean.to_lowercase());
                if seen.insert(h.clone()) {
                    out.push(h);
                }
                continue;
            }
        }
        // 4. CommitSha (7..=40 hex with >=1 hex letter and >=1 digit)
        if (7..=40).contains(&tok_clean.len())
            && tok_clean.chars().all(|c| c.is_ascii_hexdigit())
            && tok_clean.chars().any(|c| c.is_ascii_digit())
            && tok_clean.chars().any(|c| matches!(c.to_ascii_lowercase(), 'a'..='f'))
        {
            let h = format!("\u{1}h:sha:{}", &tok_clean.to_lowercase()[..7.min(tok_clean.len())]);
            if seen.insert(h.clone()) {
                out.push(h);
            }
            continue;
        }
        // 5. Version (`v1.2` or `1.2.3`)
        let ver_cand = tok_clean.strip_prefix('v').or_else(|| tok_clean.strip_prefix('V')).unwrap_or(tok_clean);
        let ver_parts: Vec<&str> = ver_cand.split('.').collect();
        if ver_parts.len() >= 2
            && (tok_clean.starts_with('v') || tok_clean.starts_with('V') || ver_parts.len() >= 3)
            && ver_parts.iter().all(|p| !p.is_empty() && p.len() <= 5 && p.chars().all(|c| c.is_ascii_digit()))
        {
            let h = format!("\u{1}h:ver:{ver_cand}");
            if seen.insert(h.clone()) {
                out.push(h);
            }
            continue;
        }
        // 6. FilePath (`dir/file.ext` or `src\file.rs`)
        let norm_path = tok_clean.replace('\\', "/");
        if norm_path.contains('/')
            && !norm_path.contains("://")
            && norm_path.chars().any(|c| c.is_ascii_alphabetic())
            && norm_path.rsplit('/').next().is_some_and(|leaf| leaf.contains('.') && leaf.len() >= 3)
        {
            let h = format!("\u{1}h:path:{}", norm_path.to_lowercase());
            if seen.insert(h.clone()) {
                out.push(h);
            }
        }
    }
    out
}

/// Technical-handle term ids (`\u{1}h:…`) are linked only by backward RSB edges.
/// Undirected PPR must not reactivate them, or a later zero-overlap session is pulled forward.
pub(in crate::retrieval) fn is_technical_handle(vocab: &Vocab, id: u32) -> bool {
    vocab.names.get(id as usize).is_some_and(|name| name.starts_with("\u{1}h:"))
}

/// Episode prefix before `_s{digits}` when sub-sessions belong to one contiguous batch,
/// else empty string (never inspects benchmark labels).
pub(in crate::retrieval) fn session_episode_key<'a>(session_id: &'a str, date: &'a str) -> String {
    if let Some((prefix, suffix)) = session_id.rsplit_once("_s") {
        if !prefix.is_empty() && !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            return format!("ep:{prefix}");
        }
    }
    let trimmed = date.trim();
    if !trimmed.is_empty() {
        return format!("dt:{trimmed}");
    }
    String::new()
}

pub(in crate::retrieval) fn camel_case_parts(ident: &str) -> Vec<String> {
    let chars: Vec<char> = ident.chars().collect();
    let mut seen_lower = false;
    let mut gated = false;
    for &c in &chars {
        if c.is_ascii_lowercase() {
            seen_lower = true;
        } else if seen_lower && c.is_ascii_uppercase() {
            gated = true;
            break;
        }
    }
    if !gated {
        return Vec::new();
    }
    let mut raw_parts: Vec<String> = Vec::new();
    let mut start = 0usize;
    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let cur = chars[i];
        let hump = (prev.is_ascii_lowercase() || prev.is_ascii_digit()) && cur.is_ascii_uppercase();
        let acronym_tail = prev.is_ascii_uppercase()
            && cur.is_ascii_uppercase()
            && chars.get(i + 1).is_some_and(|n| n.is_ascii_lowercase());
        if hump || acronym_tail {
            raw_parts.push(chars[start..i].iter().collect());
            start = i;
        }
    }
    raw_parts.push(chars[start..].iter().collect());
    if raw_parts.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for part in raw_parts {
        let lower = part.to_lowercase();
        if lower.chars().count() < 3 || !is_content(&lower) || out.contains(&lower) {
            continue;
        }
        out.push(lower);
    }
    out
}

pub(in crate::retrieval) fn surface_aux_tokens(word: &str) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    if word.contains('_') || word.contains('-') {
        let joined: String = word
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        if (4..=48).contains(&joined.len()) && joined.chars().any(|c| c.is_ascii_alphabetic()) {
            out.push((joined, 1.0));
        }
    }
    for piece in word.split(|c: char| !c.is_ascii_alphanumeric()) {
        if piece.is_empty() {
            continue;
        }
        for part in camel_case_parts(piece) {
            if !out.iter().any(|(p, _)| p == &part) {
                out.push((part, 0.85));
            }
        }
        // Alphanumeric versioned base split (`HTML5` -> `html`, `CSS3` -> `css`, `OAuth2` -> `oauth`).
        let alpha_end = piece
            .bytes()
            .take_while(|b| b.is_ascii_alphabetic())
            .count();
        let digit_tail = piece.len().saturating_sub(alpha_end);
        if alpha_end >= 3
            && (1..=2).contains(&digit_tail)
            && piece.as_bytes()[alpha_end..].iter().all(u8::is_ascii_digit)
        {
            let base = piece[..alpha_end].to_ascii_lowercase();
            if is_content(&base) && !out.iter().any(|(p, _)| p == &base) {
                out.push((base, 0.85));
            }
        }
    }
    out
}

pub(in crate::retrieval) const NICK_TAIL: usize = 3;

pub(in crate::retrieval) struct SpeakerNickIndex {
    pub(in crate::retrieval) prefix_of: DetHashMap<String, String>,
    pub(in crate::retrieval) witnessed: DetHashMap<String, DetHashSet<String>>,
    pub(in crate::retrieval) lower_attested: DetHashSet<String>,
}

pub(in crate::retrieval) fn common_speaker_tokens() -> &'static DetHashSet<String> {
    static S: OnceLock<DetHashSet<String>> = OnceLock::new();
    S.get_or_init(|| {
        let mut s = DetHashSet::default();
        for w in STOPWORDS
            .iter()
            .chain(STARTER_STOP)
            .chain(MONTHS.iter())
            .chain(WEEKDAYS.iter())
        {
            s.insert((*w).to_string());
        }
        for word in MONTHS.iter().chain(WEEKDAYS.iter()) {
            for len in 2..word.len() {
                s.insert(word[..len].to_string());
            }
        }
        s
    })
}

pub(in crate::retrieval) fn common_speaker_token(w: &str) -> bool {
    common_speaker_tokens().contains(w)
}

pub(in crate::retrieval) fn inflectional_tail(nick: &str, full: &str) -> bool {
    matches!(
        full.strip_prefix(nick),
        Some("s" | "es" | "ed" | "ing" | "er" | "ers" | "ly")
    )
}

pub(in crate::retrieval) struct SurfaceTok {
    pub(in crate::retrieval) lower: String,
    pub(in crate::retrieval) title_case: bool,
    pub(in crate::retrieval) sentence_initial: bool,
}

pub(in crate::retrieval) fn surface_tokens(text: &str) -> Vec<SurfaceTok> {
    let mut out = Vec::new();
    let mut sentence_initial = true;
    let mut current = String::new();
    let flush = |current: &mut String, sentence_initial: bool, out: &mut Vec<SurfaceTok>| {
        if current.is_empty() {
            return;
        }
        let piece = std::mem::take(current);
        let owned = piece.clone();
        let lower_piece = owned.to_lowercase();
        let cleaned_lower = clean_contraction(&lower_piece);
        let cleaned_raw = if cleaned_lower.len() == lower_piece.len() {
            owned
        } else {
            let keep = cleaned_lower.chars().count();
            owned.chars().take(keep).collect()
        };
        if cleaned_raw.is_empty() {
            return;
        }
        let mut chars = cleaned_raw.chars();
        let first_upper = chars.next().is_some_and(|c| c.is_ascii_uppercase());
        let rest_lower = chars.all(|c| !c.is_ascii_uppercase());
        let lower: String = cleaned_raw.chars().flat_map(|c| c.to_lowercase()).collect();
        if lower.is_empty() {
            return;
        }
        out.push(SurfaceTok {
            lower,
            title_case: first_upper && rest_lower,
            sentence_initial,
        });
    };
    for ch in text.chars() {
        let keep = ch.is_alphanumeric() || ch == '\'' || ch == '’';
        if keep {
            current.push(ch);
            continue;
        }
        let was_empty = current.is_empty();
        flush(&mut current, sentence_initial, &mut out);
        if matches!(ch, '.' | '?' | '!' | '\n') {
            sentence_initial = true;
        } else if !was_empty {
            sentence_initial = false;
        }
    }
    flush(&mut current, sentence_initial, &mut out);
    out
}

impl SpeakerNickIndex {
    pub(in crate::retrieval) fn build(sessions: &[Session]) -> Self {
        let mut exact_roles: DetHashSet<String> = DetHashSet::default();
        let mut targets: DetHashSet<String> = DetHashSet::default();
        let mut lower_attested: DetHashSet<String> = DetHashSet::default();
        let mut texts: Vec<&str> = Vec::new();
        for s in sessions {
            for t in &s.turns {
                texts.push(t.content.as_str());
                let role = t.role.trim();
                if role.is_empty()
                    || role.eq_ignore_ascii_case("user")
                    || role.eq_ignore_ascii_case("assistant")
                {
                    continue;
                }
                for w in raw_tokens(role) {
                    exact_roles.insert(w.clone());
                    if w.len() >= 4
                        && !STARTER_STOP.contains(&w.as_str())
                        && !stopwords().contains(w.as_str())
                    {
                        targets.insert(w);
                    }
                }
            }
        }
        for text in &texts {
            for tok in surface_tokens(text) {
                if !tok.title_case {
                    lower_attested.insert(tok.lower);
                }
            }
        }
        let mut hits: DetHashMap<String, Vec<String>> = DetHashMap::default();
        let mut ordered: Vec<&String> = targets.iter().collect();
        ordered.sort();
        for name in ordered {
            for len in 2..=5 {
                if name.len() < len + NICK_TAIL || !name.is_char_boundary(len) {
                    continue;
                }
                let p = &name[..len];
                if exact_roles.contains(p) || common_speaker_token(p) || inflectional_tail(p, name) {
                    continue;
                }
                hits.entry(p.to_string()).or_default().push(name.clone());
            }
        }
        let mut prefix_of: DetHashMap<String, String> = DetHashMap::default();
        for (p, mut names) in hits {
            names.sort();
            names.dedup();
            if names.len() == 1 {
                prefix_of.insert(p, names.pop().unwrap());
            }
        }
        let mut witnessed: DetHashMap<String, DetHashSet<String>> = DetHashMap::default();
        for text in &texts {
            for tok in surface_tokens(text) {
                if !tok.title_case {
                    continue;
                }
                let Some(full) = prefix_of.get(&tok.lower) else { continue };
                if tok.sentence_initial && lower_attested.contains(&tok.lower) {
                    continue;
                }
                witnessed.entry(full.clone()).or_default().insert(tok.lower);
            }
        }
        Self { prefix_of, witnessed, lower_attested }
    }

    pub(in crate::retrieval) fn bind_token(&self, tok: &SurfaceTok, sentence_initial: bool) -> Option<&str> {
        if !tok.title_case || common_speaker_token(&tok.lower) {
            return None;
        }
        let full = self.prefix_of.get(&tok.lower)?;
        if sentence_initial && self.lower_attested.contains(&tok.lower) {
            return None;
        }
        let seen = self.witnessed.get(full);
        let witnessed_this = seen.is_some_and(|s| s.contains(&tok.lower));
        if !(3..=4).contains(&tok.lower.len()) && !witnessed_this {
            return None;
        }
        if seen.is_some_and(|s| !s.is_empty() && !s.contains(&tok.lower)) {
            return None;
        }
        Some(full.as_str())
    }

    pub(in crate::retrieval) fn bindings_in(&self, text: &str) -> DetHashMap<String, String> {
        let mut out = DetHashMap::default();
        for tok in surface_tokens(text) {
            if let Some(full) = self.bind_token(&tok, tok.sentence_initial) {
                out.insert(tok.lower, full.to_string());
            }
        }
        out
    }
}

pub(in crate::retrieval) fn digit_token_set(text: &str) -> DetHashSet<String> {
    raw_tokens(text)
        .into_iter()
        .filter(|w| w.chars().any(|c| c.is_ascii_digit()))
        .collect()
}

pub(in crate::retrieval) fn closed_locative(w: &str) -> bool {
    matches!(w, "in" | "at" | "to" | "from" | "on" | "into" | "over" | "by")
}

/// Split an ASCII lowercase compound word (`len >= 7`) into two WordNet-attested
/// content words (`left`, `right`) of length >= 3 (e.g. `roadtrip -> ("road", "trip")`,
/// `bookshelf -> ("book", "shelf")`), skipping regular derivational/inflectional endings.
/// hardcode-scan: closed-class (English derivational/inflectional suffixes only).
pub(in crate::retrieval) fn split_wordnet_compound(w: &str) -> Option<(&str, &str)> {
    let bytes = w.as_bytes();
    if bytes.len() < 7 || !bytes.iter().all(|b| b.is_ascii_lowercase()) {
        return None;
    }
    if [
        "ing", "tion", "sion", "ment", "ness", "able", "ible", "less", "ship", "hood", "ward",
        "ical", "ious", "eous", "ally", "ated", "ting", "ized", "ised",
    ]
    .iter()
    .any(|suf| w.ends_with(suf))
    {
        return None;
    }
    let mut best: Option<(&str, &str, usize)> = None;
    for k in 3..=(bytes.len() - 3) {
        let left = &w[..k];
        let right = &w[k..];
        if !crate::retrieval::query::tokens::is_content(left)
            || !crate::retrieval::query::tokens::is_content(right)
        {
            continue;
        }
        if crate::wordnet::senses(left).is_empty() || crate::wordnet::senses(right).is_empty() {
            continue;
        }
        // Prefer balanced splits (larger min(left.len, right.len)).
        let balance = left.len().min(right.len());
        if best.is_none_or(|(_, _, b)| balance > b) {
            best = Some((left, right, balance));
        }
    }
    best.map(|(l, r, _)| (l, r))
}

/// Deterministic English derivational stem normalization (`-tion/-sion/-ment/-ical/-al/-ity/-ness/-ive/-ous`)
/// verified strictly against Princeton WordNet 3.1 (`wordnet::senses`), returning
/// attested base lemmas (e.g. `educational -> ["education", "educate"]`).
/// hardcode-scan: closed-class (English derivational morphology suffixes only).
pub(in crate::retrieval) fn derivational_base_stems(w: &str) -> Vec<String> {
    let bytes = w.as_bytes();
    if bytes.len() < 6 || !bytes.iter().all(|b| b.is_ascii_lowercase()) {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut try_push = |cand: String| {
        if cand.len() >= 4
            && cand != w
            && !crate::wordnet::senses(&cand).is_empty()
            && !out.contains(&cand)
        {
            out.push(cand);
        }
    };
    // hardcode-scan: closed-class (English derivational morphology suffix rules only).
    const SUFFIX_RULES: &[(&str, &[&str])] = &[
        ("ational", &["ation", "ate"]),
        ("ation", &["ate", ""]),
        ("ition", &["ite", ""]),
        ("tion", &["te", "t", ""]),
        ("sion", &["de", "d", "se", ""]),
        ("ment", &[""]),
        ("ical", &["ic", "y", ""]),
        ("ally", &["al", ""]),
        ("al", &["", "e"]),
        ("ity", &["", "e"]),
        ("ness", &[""]),
        ("ive", &["", "e"]),
        ("ous", &["", "e", "y"]),
        ("ist", &["", "ism", "e"]),
    ];
    for &(suf, repls) in SUFFIX_RULES {
        if let Some(base) = w.strip_suffix(suf)
            && base.len() >= 3
        {
            for &rep in repls {
                try_push(format!("{base}{rep}"));
            }
        }
    }
    out
}

/// True when two lowercase ASCII words differ by at most 1 single-character edit
/// (insertion, deletion, substitution, or adjacent transposition), used strictly
/// to rescue OOV query typos (`df == 0`, `len >= 6`, e.g. `educaton <-> education`).
pub(in crate::retrieval) fn ascii_edit_dist_le_1(a: &str, b: &str) -> bool {
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    let la = ab.len();
    let lb = bb.len();
    if la.abs_diff(lb) > 1 || la < 5 || lb < 5 {
        return false;
    }
    if la == lb {
        let mut diffs = Vec::new();
        for i in 0..la {
            if ab[i] != bb[i] {
                diffs.push(i);
                if diffs.len() > 2 {
                    return false;
                }
            }
        }
        return diffs.len() == 1
            || (diffs.len() == 2
                && diffs[1] == diffs[0] + 1
                && ab[diffs[0]] == bb[diffs[1]]
                && ab[diffs[1]] == bb[diffs[0]]);
    }
    let (short, long) = if la < lb { (ab, bb) } else { (bb, ab) };
    let (mut i, mut j, mut skipped) = (0usize, 0usize, false);
    while i < short.len() && j < long.len() {
        if short[i] == long[j] {
            i += 1;
            j += 1;
        } else if !skipped {
            skipped = true;
            j += 1;
        } else {
            return false;
        }
    }
    true
}
