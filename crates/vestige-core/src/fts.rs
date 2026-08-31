//! FTS5 Query Sanitization
//!
//! Always-available utilities for SQLite FTS5 full-text search.
//! Separated from the `search` module (which requires the `vector-search` feature)
//! because FTS5 keyword search is a core capability that works without embeddings.

/// Dangerous FTS5 operators that could be used for injection or DoS
const FTS5_OPERATORS: &[&str] = &["OR", "AND", "NOT", "NEAR"];

/// Sanitize input for FTS5 MATCH queries using individual term matching.
///
/// Unlike `sanitize_fts5_query` which wraps in quotes for a phrase search,
/// this function produces individual terms joined with implicit AND.
/// This matches documents that contain ALL the query words in any order.
///
/// Use this when you want "find all records containing these words" rather
/// than "find records with this exact phrase".
pub fn sanitize_fts5_terms(query: &str) -> Option<String> {
    let limited: String = query.chars().take(1000).collect();
    let mut sanitized = limited;

    // Blank EVERY non-alphanumeric character rather than a hand-picked deny list.
    // The old list missed 14 characters still meaningful to the FTS5 query
    // grammar -- most damagingly the apostrophe, which opens a string literal:
    // recall("cargo can't find crate") returned zero keyword hits while the same
    // query without the apostrophe matched. A deny list has to be exhaustively
    // right; an allow list only has to be conservative.
    //
    // Unicode-aware (`is_alphanumeric`, not `is_ascii_alphanumeric`) to mirror
    // the index's `unicode61` tokenizer as of migration V30 -- accented and
    // non-Latin words are real tokens and must survive sanitization.
    sanitized = sanitized
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();

    // Drop any token that IS a bare FTS5 boolean operator (AND/OR/NOT/NEAR),
    // case-insensitively. Token-level filtering is complete and repetition-proof.
    // The previous string-replace approach was single-pass and non-overlapping,
    // so a doubled operator ("foo AND AND AND") left a bare operator at the edge
    // or interior that leaked into the raw MATCH — either aborting with an FTS5
    // syntax error, or silently flipping implicit-AND to boolean-OR
    // ("foo OR bar"). Filtering tokens cannot leak an operator regardless of how
    // many are chained.
    let terms: Vec<&str> = sanitized
        .split_whitespace()
        .filter(|t| !FTS5_OPERATORS.iter().any(|op| t.eq_ignore_ascii_case(op)))
        .collect();
    if terms.is_empty() {
        return None;
    }
    // Join with space: FTS5 implicit AND — all terms must appear
    Some(terms.join(" "))
}

/// Build a RECALL-friendly FTS5 query that matches rows containing ANY of the
/// query's tokens, each quoted as a phrase literal so punctuation/operators are
/// neutralized. Produces e.g. `"500" OR "internal" OR "server" OR "error"`.
///
/// This is the correct default for natural-language similarity search: implicit
/// AND (the old behavior) requires every word — including "on"/"the" — to appear,
/// which silently drops near-matches; wrapping the whole string in one phrase
/// (the prior `sanitize_fts5_query`) requires the tokens to be adjacent and in
/// order, which drops nearly everything. OR + `ORDER BY rank` (BM25) ranks the
/// row sharing the most distinctive tokens first — true lexical resemblance.
///
/// Per https://sqlite.org/fts5.html an embedded `"` is escaped by doubling it.
///
/// Tokenization MUST mirror the index's `tokenize='porter ascii'` (migration V7):
/// the `ascii` tokenizer treats every non-ASCII-alphanumeric byte as a separator,
/// including `_` and any non-ASCII letter. So we split on `!is_ascii_alphanumeric`
/// — otherwise a query token like `API_TIMEOUT` or `café` becomes a single phrase
/// (`"api_timeout"` / `"café"`) that can NEVER match the index (which stored them
/// as `api`+`timeout` / `caf`). Per-token length is capped at 64 (the ascii
/// tokenizer's effective max token length) and token count at 64 to bound the
/// OR-chain. ASCII lowercasing mirrors the tokenizer's case-folding.
pub fn sanitize_fts5_or_query(query: &str) -> Option<String> {
    let limited: String = query.chars().take(1000).collect();
    let q: String = limited
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .take(64) // bound the OR-chain length (DoS hardening)
        .map(|t| {
            // mirror the unicode61 tokenizer (migration V30): lowercase, cap length
            let tok: String = t.chars().take(64).collect::<String>().to_lowercase();
            format!("\"{}\"", tok.replace('"', "\"\""))
        })
        .collect::<Vec<_>>()
        .join(" OR ");
    if q.is_empty() { None } else { Some(q) }
}

/// Sanitize input for FTS5 MATCH queries
///
/// Prevents:
/// - Boolean operator injection (OR, AND, NOT, NEAR)
/// - Column targeting attacks (content:secret)
/// - Prefix/suffix wildcards for data extraction
/// - DoS via complex query patterns
pub fn sanitize_fts5_query(query: &str) -> String {
    // Limit query length to prevent DoS (char-aware to avoid UTF-8 boundary issues)
    let limited: String = query.chars().take(1000).collect();

    // Remove FTS5 special characters and operators
    let mut sanitized = limited.to_string();

    // Remove special characters: * : ^ - " ( ) and common identifier/path
    // punctuation that FTS5 otherwise treats as syntax.
    sanitized = sanitized
        .chars()
        .map(|c| match c {
            '*' | ':' | '^' | '-' | '"' | '(' | ')' | '{' | '}' | '[' | ']' | '.' | '/' | '\\'
            | '=' | '@' => ' ',
            _ => c,
        })
        .collect();

    // Remove FTS5 boolean operators (case-insensitive)
    for op in FTS5_OPERATORS {
        // Use word boundary replacement to avoid partial matches
        let pattern = format!(" {} ", op);
        sanitized = sanitized.replace(&pattern, " ");
        sanitized = sanitized.replace(&pattern.to_lowercase(), " ");

        // Handle operators at start/end (using char-aware operations)
        let upper = sanitized.to_uppercase();
        let start_pattern = format!("{} ", op);
        if upper.starts_with(&start_pattern) {
            sanitized = sanitized.chars().skip(op.len()).collect();
        }
        let end_pattern = format!(" {}", op);
        if upper.ends_with(&end_pattern) {
            let char_count = sanitized.chars().count();
            sanitized = sanitized
                .chars()
                .take(char_count.saturating_sub(op.len()))
                .collect();
        }
    }

    // Collapse multiple spaces and trim
    let sanitized = sanitized.split_whitespace().collect::<Vec<_>>().join(" ");

    // If empty after sanitization, return a safe default
    if sanitized.is_empty() {
        return "\"\"".to_string(); // Empty phrase - matches nothing safely
    }

    // Wrap in quotes to treat as literal phrase search
    format!("\"{}\"", sanitized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_fts5_query_basic() {
        assert_eq!(sanitize_fts5_query("hello world"), "\"hello world\"");
    }

    #[test]
    fn test_sanitize_fts5_query_operators() {
        assert_eq!(sanitize_fts5_query("hello OR world"), "\"hello world\"");
        assert_eq!(sanitize_fts5_query("hello AND world"), "\"hello world\"");
        assert_eq!(sanitize_fts5_query("NOT hello"), "\"hello\"");
    }

    #[test]
    fn test_sanitize_fts5_query_special_chars() {
        assert_eq!(sanitize_fts5_query("hello* world"), "\"hello world\"");
        assert_eq!(sanitize_fts5_query("content:secret"), "\"content secret\"");
        assert_eq!(sanitize_fts5_query("^boost"), "\"boost\"");
    }

    #[test]
    fn test_sanitize_fts5_query_empty() {
        assert_eq!(sanitize_fts5_query(""), "\"\"");
        assert_eq!(sanitize_fts5_query("   "), "\"\"");
        assert_eq!(sanitize_fts5_query("* : ^"), "\"\"");
    }

    #[test]
    fn test_sanitize_fts5_query_length_limit() {
        let long_query = "a".repeat(2000);
        let sanitized = sanitize_fts5_query(&long_query);
        assert!(sanitized.len() <= 1004);
    }

    // --- sanitize_fts5_or_query (rotation-audit-hardened) -------------------

    #[test]
    fn or_query_splits_like_ascii_tokenizer() {
        // The index uses tokenize='porter ascii': '_' and non-ASCII are separators.
        // API_TIMEOUT must become two tokens, lowercased — NOT one phrase that
        // could never match the index. (Consensus finding, DeepSeek + MiniMax.)
        let q = sanitize_fts5_or_query("API_TIMEOUT failed").unwrap();
        assert_eq!(q, "\"api\" OR \"timeout\" OR \"failed\"");
    }

    #[test]
    fn or_query_preserves_non_ascii_tokens() {
        // As of migration V30 the index uses `unicode61 remove_diacritics 2`, so
        // an accented word is a REAL token, not a truncated ASCII prefix. The old
        // `ascii` tokenizer indexed "café" as "caf", and this test asserted that
        // lossy behavior; preserving the token is now the correct expectation
        // (FTS5 applies the same tokenizer to query terms, so diacritic folding
        // happens on both sides and still matches).
        let q = sanitize_fts5_or_query("café").unwrap();
        assert_eq!(q, "\"café\"");
    }

    #[test]
    fn or_query_neutralizes_fts5_operators_and_injection() {
        // Operators/columns/wildcards are all separators -> stripped, then quoted.
        let q = sanitize_fts5_or_query("title:secret OR a* -b \"x\"").unwrap();
        // every token is a quoted phrase literal; no bare operator survives except
        // our own joining OR. An embedded quote is doubled.
        assert!(q.contains("\"title\""));
        assert!(q.contains("\"secret\""));
        assert!(!q.contains("title:"));
        assert!(!q.contains("a*"));
        assert!(!q.contains("-b"));
    }

    #[test]
    fn or_query_empty_and_punctuation_only() {
        assert_eq!(sanitize_fts5_or_query(""), None);
        assert_eq!(sanitize_fts5_or_query("   "), None);
        assert_eq!(sanitize_fts5_or_query(":-*^()"), None);
    }

    #[test]
    fn or_query_bounds_token_count_and_length() {
        // DoS hardening: <=64 arms, each token <=64 chars.
        let many = (0..500)
            .map(|i| format!("t{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let q = sanitize_fts5_or_query(&many).unwrap();
        assert!(q.matches(" OR ").count() <= 63, "OR-chain must be bounded");
        let longtok = "a".repeat(200);
        let q2 = sanitize_fts5_or_query(&longtok).unwrap();
        assert!(
            q2.len() <= 66,
            "single token capped at 64 + quotes, got {}",
            q2.len()
        );
    }

    #[test]
    fn terms_strips_all_bare_operators_even_when_doubled() {
        // Regression: doubled/chained operators must never leak a bare operator
        // into the raw MATCH (which aborts with a syntax error or silently flips
        // implicit-AND to boolean-OR).
        for op in FTS5_OPERATORS {
            let doubled = format!("foo {op} {op} {op} bar");
            let out = sanitize_fts5_terms(&doubled).unwrap();
            for tok in out.split_whitespace() {
                assert!(
                    !FTS5_OPERATORS.iter().any(|o| tok.eq_ignore_ascii_case(o)),
                    "operator {op} leaked into output {out:?}"
                );
            }
            assert_eq!(out, "foo bar", "only real terms should survive: {out:?}");
        }
        // Lowercase + leading/trailing operators are stripped too.
        assert_eq!(sanitize_fts5_terms("or foo and bar or").unwrap(), "foo bar");
        // Operator-only input yields nothing.
        assert_eq!(sanitize_fts5_terms("AND OR NOT").map(|_| ()), None);
    }

    #[test]
    fn terms_strips_all_bare_operators_even_when_doubled() {
        // Regression: doubled/chained operators must never leak a bare operator
        // into the raw MATCH (which aborts with a syntax error or silently flips
        // implicit-AND to boolean-OR).
        for op in FTS5_OPERATORS {
            let doubled = format!("foo {op} {op} {op} bar");
            let out = sanitize_fts5_terms(&doubled).unwrap();
            for tok in out.split_whitespace() {
                assert!(
                    !FTS5_OPERATORS.iter().any(|o| tok.eq_ignore_ascii_case(o)),
                    "operator {op} leaked into output {out:?}"
                );
            }
            assert_eq!(out, "foo bar", "only real terms should survive: {out:?}");
        }
        // Lowercase + leading/trailing operators are stripped too.
        assert_eq!(sanitize_fts5_terms("or foo and bar or").unwrap(), "foo bar");
        // Operator-only input yields nothing.
        assert_eq!(sanitize_fts5_terms("AND OR NOT").map(|_| ()), None);
    }
}
