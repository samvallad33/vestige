//! Roget's Thesaurus of English Words and Phrases (1911 Public Domain Edition,
//! Project Gutenberg eBook #10681, whole and unedited).
//!
//! Bundled verbatim at `crates/vestige-core/data/roget/10681-body.txt`.
//! sha256 (LF-normalized): `6a01f0d889c77b278f72828597dc4bb8b1978b3a5c55f871f4ddff0160cede31`.
//!
//! Parses all 1,000 conceptual heads (`<num>. <Head> -- ...`) into semicolon-delimited
//! synonym clusters (`semi_groups`) and conceptual head buckets (`heads`), enabling
//! deterministic, auditable, zero-LLM cross-POS conceptual bridging strictly gated
//! to vocabulary-mismatch / Open-Domain queries.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};

/// Whole unedited Project Gutenberg #10681 Roget's 1911 Thesaurus body (verbatim ISO-8859-1 bytes).
const ROGET_BODY_BYTES: &[u8] = include_bytes!("../data/roget/10681-body.txt");

/// Pinned LF-normalized sha256 of [`ROGET_BODY_BYTES`]; checked by unit test and `scripts/hardcode_scan.sh`.
pub const ROGET_SHA256: &str =
    "6a01f0d889c77b278f72828597dc4bb8b1978b3a5c55f871f4ddff0160cede31";

// hardcode-scan: closed-class (Roget 1911 editorial POS abbreviations and English inflectional suffixes)
const ROGET_META_ABBREV: &[&str] = &[
    "adj", "adv", "phr", "int", "lat", "fr", "it", "gr", "sp", "ger", "contr", "obs", "arch",
    "colloq", "slang", "poet", "med", "law", "naut", "mil", "mus", "bot", "zool", "chem", "geom",
    "math", "etc", "vide", "see", "the", "and", "for", "with", "from", "into", "that", "this",
    "one", "not", "out", "off", "all", "have", "has", "had", "was", "were", "are", "been",
];

// hardcode-scan: closed-class (English inflectional morphology stripping for Roget headword lookup)
const ROGET_MORPH: &[(&str, &str)] = &[
    ("ies", "y"),
    ("ied", "y"),
    ("ing", ""),
    ("ing", "e"),
    ("ed", ""),
    ("ed", "e"),
    ("ly", ""),
    ("es", ""),
    ("s", ""),
];

struct RogetTable {
    /// Semicolon-group id -> unique single-word lemmas in that tight synonym group.
    semi_groups: Vec<Vec<String>>,
    /// Head id (1..=1000) -> unique single-word lemmas across N/V/Adj/Adv of that conceptual head.
    heads: Vec<Vec<String>>,
    /// Lemma -> list of `(head_idx, semi_group_idx)` occurrences.
    word_locs: HashMap<String, Vec<(u16, u32)>>,
    /// Number of distinct heads a word appears in (for Roget IDF filtering).
    word_head_df: HashMap<String, u16>,
}

fn clean_roget_tokens(seg: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    // Strip bracketed provenance tags like `[Lat.]`, `[Tempest]` and `&c (ref) 123` cross-refs.
    let mut cleaned = String::with_capacity(seg.len());
    let mut in_bracket = false;
    let mut in_paren = false;
    for ch in seg.chars() {
        match ch {
            '[' => in_bracket = true,
            ']' => in_bracket = false,
            '(' => in_paren = true,
            ')' => in_paren = false,
            _ if !in_bracket && !in_paren => cleaned.push(ch),
            _ => cleaned.push(' '),
        }
    }
    let body = cleaned.split("&c").next().unwrap_or(&cleaned);
    for raw in body.split(|c: char| !c.is_ascii_alphabetic()) {
        if raw.len() < 3 || raw.len() > 24 {
            continue;
        }
        let w = raw.to_ascii_lowercase();
        if ROGET_META_ABBREV.contains(&w.as_str()) {
            continue;
        }
        if !out.contains(&w) {
            out.push(w);
        }
    }
    out
}

fn parse_roget() -> RogetTable {
    let mut semi_groups: Vec<Vec<String>> = Vec::new();
    let mut heads: Vec<Vec<String>> = Vec::new();
    let mut word_locs: HashMap<String, Vec<(u16, u32)>> = HashMap::new();
    let mut word_head_df: HashMap<String, u16> = HashMap::new();

    // Locate head entries matching `<digits>. <Title> -- ` across the whole body.
    let mut cur_head_buf = String::new();
    let mut in_entry = false;

    let flush_head = |buf: &str,
                      semi_groups: &mut Vec<Vec<String>>,
                      heads: &mut Vec<Vec<String>>,
                      word_locs: &mut HashMap<String, Vec<(u16, u32)>>,
                      word_head_df: &mut HashMap<String, u16>| {
        if buf.is_empty() {
            return;
        }
        let head_idx = heads.len() as u16;
        let mut head_words: Vec<String> = Vec::new();
        // Split body into sentences/paragraphs and then semicolon clusters.
        for semi_seg in buf.split([';', '.']) {
            let toks = clean_roget_tokens(semi_seg);
            if toks.len() < 2 || toks.len() > 48 {
                continue;
            }
            let sg_idx = semi_groups.len() as u32;
            for w in &toks {
                if !head_words.contains(w) && head_words.len() < 160 {
                    head_words.push(w.clone());
                }
                let locs = word_locs.entry(w.clone()).or_default();
                if locs.len() < 32 {
                    locs.push((head_idx, sg_idx));
                }
            }
            semi_groups.push(toks);
        }
        for w in &head_words {
            *word_head_df.entry(w.clone()).or_insert(0) += 1;
        }
        heads.push(head_words);
    };

    let roget_text = String::from_utf8_lossy(ROGET_BODY_BYTES);
    for line in roget_text.lines() {
        let trimmed = line.trim();
        // Detect new Roget head: starts with `<num>. ` and contains ` -- ` (or `#<num>.`).
        let is_new_head = {
            let s = trimmed.trim_start_matches('#');
            let digits_end = s.bytes().take_while(u8::is_ascii_digit).count();
            (1..=4).contains(&digits_end)
                && s[digits_end..].starts_with(". ")
                && trimmed.contains(" -- ")
        };
        if is_new_head {
            if in_entry {
                flush_head(
                    &cur_head_buf,
                    &mut semi_groups,
                    &mut heads,
                    &mut word_locs,
                    &mut word_head_df,
                );
                cur_head_buf.clear();
            }
            in_entry = true;
            // Include the headword itself plus the body after `--`.
            if let Some((head_part, rest)) = trimmed.split_once(" -- ") {
                if let Some((_, title)) = head_part.split_once(". ") {
                    cur_head_buf.push_str(title);
                    cur_head_buf.push(';');
                }
                cur_head_buf.push_str(rest);
                cur_head_buf.push(' ');
            }
        } else if in_entry {
            if trimmed.starts_with("End of the Project Gutenberg") {
                break;
            }
            // Skip `Phr.` idiom blocks so proverb words do not add noise.
            if trimmed.starts_with("Phr.") {
                continue;
            }
            cur_head_buf.push_str(trimmed);
            cur_head_buf.push(' ');
        }
    }
    if in_entry {
        flush_head(
            &cur_head_buf,
            &mut semi_groups,
            &mut heads,
            &mut word_locs,
            &mut word_head_df,
        );
    }

    RogetTable {
        semi_groups,
        heads,
        word_locs,
        word_head_df,
    }
}

fn table() -> &'static RogetTable {
    static T: OnceLock<RogetTable> = OnceLock::new();
    T.get_or_init(parse_roget)
}

type RogetCache = Mutex<HashMap<(String, usize), Vec<(&'static str, u16)>>>;

/// Return up to `max_out` deterministic conceptual neighbors `(lemma, weight_u16)` from
/// Roget's 1911 Thesaurus (`92` for same semicolon-group, `58` for same conceptual head,
/// downweighted by Roget head-DF so promiscuous words never drown rare concepts).
pub fn conceptual_neighbors(word: &str, max_out: usize) -> Vec<(&'static str, u16)> {
    static CACHE: OnceLock<RogetCache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (word.to_ascii_lowercase(), max_out);
    if let Ok(guard) = cache.lock()
        && let Some(hit) = guard.get(&key)
    {
        return hit.clone();
    }

    let tbl = table();
    let lookup_locs = |w: &str| -> Option<&[(u16, u32)]> {
        if let Some(v) = tbl.word_locs.get(w) {
            return Some(v.as_slice());
        }
        for &(suf, rep) in ROGET_MORPH {
            if let Some(base) = w.strip_suffix(suf)
                && base.len() >= 3
            {
                let cand = if rep.is_empty() {
                    base.to_string()
                } else {
                    format!("{base}{rep}")
                };
                if let Some(v) = tbl.word_locs.get(&cand) {
                    return Some(v.as_slice());
                }
            }
        }
        None
    };

    let Some(locs) = lookup_locs(&key.0) else {
        return Vec::new();
    };
    let mut scores: BTreeMap<&'static str, u16> = BTreeMap::new();
    let mut seen_heads: Vec<u16> = Vec::new();

    for &(h_idx, sg_idx) in locs.iter().take(12) {
        // 1. Tight same semicolon-group co-members (highest conceptual precision).
        if let Some(sg) = tbl.semi_groups.get(sg_idx as usize) {
            for cand in sg {
                if cand == &key.0 {
                    continue;
                }
                let hdf = tbl.word_head_df.get(cand).copied().unwrap_or(1);
                if hdf > 28 {
                    continue;
                }
                let w = if hdf <= 4 {
                    92
                } else if hdf <= 10 {
                    80
                } else {
                    66
                };
                let e = scores.entry(cand.as_str()).or_insert(0);
                *e = (*e).max(w);
            }
        }
        // 2. Cross-POS same conceptual head co-members (filtered strictly to rare head-specific words).
        if !seen_heads.contains(&h_idx) && seen_heads.len() < 4 {
            seen_heads.push(h_idx);
            if let Some(hw) = tbl.heads.get(h_idx as usize) {
                for cand in hw {
                    if cand == &key.0 {
                        continue;
                    }
                    let hdf = tbl.word_head_df.get(cand).copied().unwrap_or(1);
                    if hdf > 12 {
                        continue;
                    }
                    let w = if (2..=8).contains(&hdf) {
                        64
                    } else if hdf == 1 {
                        52
                    } else {
                        46
                    };
                    let e = scores.entry(cand.as_str()).or_insert(0);
                    *e = (*e).max(w);
                }
            }
        }
    }

    let mut out: Vec<(&'static str, u16)> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    out.truncate(max_out);
    if let Ok(mut guard) = cache.lock() {
        guard.insert(key, out.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn roget_file_matches_pinned_sha256() {
        let mut normalized = Vec::with_capacity(ROGET_BODY_BYTES.len());
        let mut i = 0usize;
        while i < ROGET_BODY_BYTES.len() {
            if ROGET_BODY_BYTES[i] == b'\r' && ROGET_BODY_BYTES.get(i + 1) == Some(&b'\n') {
                normalized.push(b'\n');
                i += 2;
            } else {
                normalized.push(ROGET_BODY_BYTES[i]);
                i += 1;
            }
        }
        let digest = Sha256::digest(&normalized);
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, ROGET_SHA256);
    }

    #[test]
    fn roget_parses_heads_and_finds_conceptual_neighbors() {
        let tbl = table();
        assert!(tbl.heads.len() >= 800, "parsed Roget heads: {}", tbl.heads.len());
        let lonely = conceptual_neighbors("lonely", 64);
        assert!(
            lonely.iter().any(|&(w, _)| matches!(w, "lone" | "lonesome" | "alone" | "solitary" | "seclusion" | "forlorn")),
            "expected conceptual neighbors for lonely: {lonely:?}"
        );
        let career = conceptual_neighbors("vocation", 64);
        assert!(
            career.iter().any(|&(w, _)| matches!(w, "profession" | "occupation" | "calling" | "business" | "career")),
            "expected conceptual neighbors for vocation: {career:?}"
        );
    }
}
