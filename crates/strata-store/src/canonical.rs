//! Canonical content identity for the proof-carrying write path.
//!
//! Two submissions are the *same memory* when their canonical forms byte
//! compare equal, regardless of Unicode composition, zero-width joinery, or
//! whitespace run widths. Case is identity: `API_TIMEOUT` and `api_timeout`
//! are different text, the way `Vestige` and `vestige` are different tags. The pipeline is pure: a function of
//! the submitted bytes only — no clock, no randomness, no network, no
//! HashMap-iteration-order input. Same bytes in, same bytes out, forever:
//! [`CANONICAL_PIPELINE_VERSION`] pins the behavior, and every response that
//! names a canonical hash also names the pipeline so a reader can tell which
//! normalization produced it.
//!
//! Determinism law (spec INGEST V5): sort or compare by bytes, then ids.
//! Everything here walks `char`s in order and emits one canonical `String`.

use unicode_normalization::UnicodeNormalization;

/// Version stamp of the canonicalization pipeline. Change it only with a
/// pipeline change; digests are comparable only within one version.
pub const CANONICAL_PIPELINE_VERSION: &str = "nfc-zwstrip-wscollapse-v1";

/// `source.system` value that marks a duplicate-echo node. Echo nodes are
/// created by the reinforcer after a canonical duplicate was found; they are
/// never indexed as dedup targets, so a duplicate of a duplicate still points
/// at the original.
pub const DUPLICATE_SOURCE: &str = "duplicate";

/// Canonical form of `content`:
///
/// 1. Unicode NFC (compose combining marks),
/// 2. strip the zero-width/format characters `U+200B..=U+200D` and `U+FEFF`
///    (none of them are Unicode whitespace, hence the explicit pass),
/// 3. collapse every run of Unicode whitespace to a single ASCII space,
/// 4. trim leading/trailing whitespace.
///
/// Case is kept. Folding it would make two memories that differ only in the
/// case of an identifier, an environment variable or a path the same memory,
/// and the second would never be stored.
///
/// Steps run in exactly this order; the version constant above names them.
pub fn canonicalize(content: &str) -> String {
    // 1: NFC. 2: zero-width / BOM strip.
    let mut stripped = String::with_capacity(content.len());
    for ch in content.nfc() {
        if matches!(ch, '\u{200B}'..='\u{200D}' | '\u{FEFF}') {
            continue;
        }
        stripped.push(ch);
    }
    // 3: collapse whitespace runs. `char::is_whitespace` is the Unicode
    // White_Space property (space, tab, NBSP, ideographic space, ...).
    let mut collapsed = String::with_capacity(stripped.len());
    let mut in_run = false;
    for ch in stripped.chars() {
        if ch.is_whitespace() {
            if !in_run {
                collapsed.push(' ');
                in_run = true;
            }
        } else {
            collapsed.push(ch);
            in_run = false;
        }
    }
    // 4: after collapse only ASCII spaces can remain at the ends.
    collapsed.trim().to_string()
}

/// blake3 over the canonical form of `content`, as bytes.
pub fn canonical_hash(content: &str) -> [u8; 32] {
    *blake3::hash(canonicalize(content).as_bytes()).as_bytes()
}

/// Lowercase hex of [`canonical_hash`]. This is the `canonicalHash` response
/// field and the input to [`intent_digest`].
pub fn canonical_hash_hex(content: &str) -> String {
    hex_lower(&canonical_hash(content))
}

/// The v1 intent (idempotency) digest: lowercase hex of blake3 over the
/// canonical JSON object
/// `{"content":<canonical_hash_hex>,"source":<source>,"tags":[...]}` with
/// `tags` sorted and deduplicated first.
///
/// The digest is over the canonical content *hash*, never the content bytes,
/// so a store never has to keep the original submission to recompute it, and
/// over the sorted tag list, so tag order never leaks into identity. The
/// JSON here is hand-serialized in a fixed key order (the crate keeps its
/// dependency footprint at borsh + blake3 + the strata siblings); the
/// escaping is standard minimal JSON (RFC 8259), so any conformant
/// serializer produces identical bytes for these shapes.
pub fn intent_digest(canonical_hash_hex: &str, source: &str, tags: &[String]) -> String {
    let mut tags: Vec<&str> = tags.iter().map(String::as_str).collect();
    tags.sort_unstable();
    tags.dedup();
    let mut json = String::with_capacity(
        64 + source.len() + tags.iter().map(|tag| tag.len() + 2).sum::<usize>(),
    );
    json.push_str("{\"content\":\"");
    json.push_str(canonical_hash_hex);
    json.push_str("\",\"source\":");
    push_json_string(&mut json, source);
    json.push_str(",\"tags\":[");
    for (index, tag) in tags.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        push_json_string(&mut json, tag);
    }
    json.push_str("]}");
    hex_lower(blake3::hash(json.as_bytes()).as_bytes())
}

/// Append `value` as a minimal-escaped JSON string (with quotes).
fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Lowercase hex, fixed-width two digits per byte.
fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nfc_composes_combining_marks() {
        let decomposed = "cafe\u{301}";
        let precomposed = "caf\u{e9}";
        assert_eq!(canonicalize(decomposed), canonicalize(precomposed));
        assert_eq!(canonicalize(precomposed), "café");
    }

    #[test]
    fn case_is_identity() {
        assert_eq!(canonicalize("Vestige MEMORY Store"), "Vestige MEMORY Store");
        assert_ne!(canonicalize("VÉSTÍGE"), canonicalize("véstíge"));
        // The pair that must never collapse: the same words, opposite meaning.
        assert_ne!(
            canonical_hash_hex("Set API_TIMEOUT in prod, never api_timeout."),
            canonical_hash_hex("set api_timeout in prod, never API_TIMEOUT.")
        );
    }

    #[test]
    fn zero_width_characters_are_stripped() {
        for zw in ['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}'] {
            let marked = format!("ab{zw}cd");
            assert_eq!(
                canonicalize(&marked),
                "abcd",
                "zero-width {zw:?} not stripped"
            );
        }
        // Inside multibyte content too.
        assert_eq!(canonicalize("日\u{200D}本"), "日本");
    }

    #[test]
    fn whitespace_runs_collapse_to_one_space() {
        assert_eq!(canonicalize("a \t \n b"), "a b");
        // NBSP and ideographic space are Unicode whitespace.
        assert_eq!(canonicalize("a\u{00a0}\u{3000}b"), "a b");
        assert_eq!(
            canonicalize("   leading and trailing   "),
            "leading and trailing"
        );
        assert_eq!(canonicalize("\u{2028}x\u{2029}"), "x");
    }

    #[test]
    fn multibyte_content_survives_byte_safe() {
        let input = "重要 fix in 🚒 src/store.py";
        assert_eq!(canonicalize(input), "重要 fix in 🚒 src/store.py");
        // The pipeline never splits a multibyte char: every byte offset of
        // the canonical form is a char boundary, and chars survive intact.
        let canon = canonicalize(input);
        for (offset, _) in canon.char_indices() {
            assert!(canon.is_char_boundary(offset));
        }
        assert!(canon.contains('重') && canon.contains('🚒'));
        // Deterministic across calls.
        assert_eq!(canonical_hash_hex(input), canonical_hash_hex(input));
    }

    #[test]
    fn empty_and_whitespace_only_inputs() {
        assert_eq!(canonicalize(""), "");
        assert_eq!(canonicalize(" \t\u{200b}\r\n"), "");
        assert_eq!(canonical_hash_hex(""), canonical_hash_hex("   "));
    }

    #[test]
    fn golden_vector_hex_is_pinned() {
        // Print-once pinned vector. Input mixes every pipeline stage:
        // composition, case, zero-width joinery, mixed whitespace runs.
        // Zero-width chars are stripped (not replaced by a space), so
        // "Memory\u{200D}Store" canonicalizes to "memorystore".
        let input = "  Véstí\u{200B}ge\tMemory\u{200D}Store \u{FEFF} ";
        let canonical = canonicalize(input);
        assert_eq!(canonical, "Véstíge MemoryStore");
        let hex = canonical_hash_hex(input);
        assert_eq!(hex, GOLDEN_CANONICAL_HEX);
        assert_eq!(hex.len(), 64);
        assert!(hex
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        // The same text via decomposed accents, a different zero-width char
        // and a whitespace run instead of the tab.
        assert_eq!(
            canonical_hash_hex("V\u{65}\u{301}st\u{69}\u{301}ge  Memory\u{200B}Store"),
            GOLDEN_CANONICAL_HEX
        );
    }

    /// blake3("Véstíge MemoryStore") hex, pinned so any pipeline change is
    /// a visible test failure, not a silent identity migration.
    const GOLDEN_CANONICAL_HEX: &str =
        "2657027573014a9f9f8db96feb6ce1fc19ebf42da8b33ba896641998eef44b26";

    #[test]
    fn intent_digest_is_tag_order_independent_and_pinned() {
        let hash = canonical_hash_hex("intent fixture");
        let sorted = intent_digest(&hash, "mcp", &["b".into(), "a".into(), "b".into()]);
        let canonical_input = intent_digest(&hash, "mcp", &["a".into(), "b".into()]);
        assert_eq!(sorted, canonical_input, "tags must be sorted+deduped first");
        // Empty tags, and escaping of quotes/backslashes in source and tags.
        let bare = intent_digest(&hash, "mcp", &[]);
        assert_ne!(bare, canonical_input);
        let escaped = intent_digest(&hash, "sy\"s\\t", &["ta\"g".into()]);
        assert_eq!(escaped.len(), 64);
        assert!(escaped.chars().all(|c| c.is_ascii_hexdigit()));
        // Pinned golden vector: intent_digest over
        // {"content":<hash of "intent fixture">,"source":"mcp","tags":["a","b"]}.
        assert_eq!(
            canonical_input,
            "52de12c863b82ca8ee081c804171af4428072a663426778264c1220b7c82f56c"
        );
    }
}
