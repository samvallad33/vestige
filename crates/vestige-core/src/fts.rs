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

    sanitized = sanitized
        .chars()
        .map(|c| match c {
            '*' | ':' | '^' | '-' | '"' | '(' | ')' | '{' | '}' | '[' | ']' => ' ',
            _ => c,
        })
        .collect();

    for op in FTS5_OPERATORS {
        let pattern = format!(" {} ", op);
        sanitized = sanitized.replace(&pattern, " ");
        sanitized = sanitized.replace(&pattern.to_lowercase(), " ");
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

    let terms: Vec<&str> = sanitized.split_whitespace().collect();
    if terms.is_empty() {
        return None;
    }
    // Join with space: FTS5 implicit AND — all terms must appear
    Some(terms.join(" "))
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

    // Remove special characters: * : ^ - " ( )
    sanitized = sanitized
        .chars()
        .map(|c| match c {
            '*' | ':' | '^' | '-' | '"' | '(' | ')' | '{' | '}' | '[' | ']' => ' ',
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
}
