//! # Handle Resolver (handle-based recall)
//!
//! EXACT or PREFIX ONLY. No lexical ranking, no fuzzy matching, no FTS, no
//! embeddings. A *handle* is a globally unique anchor: a memory id (uuid), a
//! commit sha (full 40-hex or a >=7-char prefix), a file path, a symbol
//! (normalized snake/camel), a test name, a run id, a tool-call id, or a tag.
//!
//! Resolution order (first hit wins, per the owner-approved design):
//!
//! 1. uuid-exact over `knowledge_nodes.id`
//! 2. sha-exact/prefix over git-commit-tagged records (content line 1 is
//!    `commit <sha> ...`; a hex-looking query shorter than 7 chars is an
//!    ambiguity error, not a match attempt)
//! 3. boundary-exact token match over node content + tags (kind `File` for
//!    path-shaped queries, `Test` for test-shaped queries — both EXACT only)
//! 4. symbol exact/prefix over extracted entities (query normalized with the
//!    same camel_to_snake normalization the extractor uses; Code-tier entities
//!    only, so a word like `pyvenv` can never prefix-match `pyvenv.cfg`)
//! 5. run id / tool-call id exact over `agent_traces` (V18)
//! 6. tag exact over node tags
//!
//! Prefix matching is granted to sha and symbol ONLY. Everything else is
//! exact. Candidate lists are capped at [`MAX_CANDIDATES`].
//!
//! The resolver is read-only: it never strengthens, writes edges, or mutates
//! FSRS state. It is scope-agnostic on purpose — a handle is globally unique,
//! so resolving it must not depend on the caller's active scope.

use rusqlite::params;

use super::sqlite::{Result, SqliteMemoryStore, StorageError};
use crate::advanced::git_records::COMMIT_TAG;
use crate::advanced::retroactive_backfill::{IdentifierTier, extract_entities, normalized_tier};

// `MAX_CANDIDATES`, `HANDLE_REQUIRED_DETAIL`, `HandleKind`, and
// `HandleResolution` are defined in (and re-exported from)
// `crate::storage::types`.
pub use crate::storage::types::{
    HANDLE_REQUIRED_DETAIL, HandleKind, HandleResolution, MAX_CANDIDATES,
};

impl HandleResolution {
    fn unresolved() -> Self {
        Self {
            kind: HandleKind::Unknown,
            ids: Vec::new(),
            exact: false,
            candidates: Vec::new(),
            handle_required: Some(HANDLE_REQUIRED_DETAIL.to_string()),
            proofs: Vec::new(),
        }
    }

    fn resolved(kind: HandleKind, ids: Vec<String>, exact: bool) -> Self {
        Self {
            kind,
            ids,
            exact,
            candidates: Vec::new(),
            handle_required: None,
            proofs: Vec::new(),
        }
    }

    fn ambiguous(kind: HandleKind, candidates: Vec<(String, HandleKind)>) -> Self {
        Self {
            kind,
            ids: Vec::new(),
            exact: false,
            candidates,
            handle_required: None,
            proofs: Vec::new(),
        }
    }
}

// ============================================================================
// QUERY SHAPE TESTS
// ============================================================================

fn is_hex(query: &str) -> bool {
    !query.is_empty() && query.chars().all(|c| c.is_ascii_hexdigit())
}

/// A hex-shaped query: a sha prefix attempt, or a too-short one (error).
enum ShaShape {
    /// 7..=40 hex chars: a legal prefix (or full sha at 40).
    Prefix,
    /// 4..=6 hex chars containing a digit: almost certainly a truncated sha —
    /// the digit requirement keeps English hex words ("face", "added") out.
    /// Reported as an ambiguity error, never matched.
    TooShort,
}

fn sha_shape(query: &str) -> Option<ShaShape> {
    if !is_hex(query) || query.len() > 40 {
        return None;
    }
    if query.len() >= 7 {
        Some(ShaShape::Prefix)
    } else if query.len() >= 4 && query.chars().any(|c| c.is_ascii_digit()) {
        Some(ShaShape::TooShort)
    } else {
        None
    }
}

/// Path-shaped queries contain a separator or extension: `src/main.rs`,
/// `pyvenv.cfg`. These resolve File-kind, exact only.
fn is_path_shaped(query: &str) -> bool {
    query.contains('/') || query.contains('.')
}

/// Test-shaped queries: a path under tests/, a `_test`-suffixed name, or a
/// Rust `test_...` fn name. These resolve Test-kind, exact only.
fn is_test_shaped(query: &str) -> bool {
    query.contains("tests/") || query.contains("_test") || query.starts_with("test_")
}

/// Normalize a single-token identifier query with the SAME normalization the
/// entity extractor applies at write/scan time (camelCase -> snake_case, then
/// lowercase), so `isAbortError` joins a stored `is_abort_error` entity.
/// Returns None for multi-token queries (a symbol handle is exactly one
/// identifier) and for queries that are not identifier-shaped at all.
fn normalize_identifier(query: &str) -> Option<String> {
    // Exactly one identifier-shaped token, no separators.
    let mut tokens = query
        .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.' || c == '/'))
        .filter(|t| !t.is_empty());
    let first = tokens.next()?;
    if tokens.next().is_some() {
        return None;
    }
    // extract_entities camel-normalizes, lowercases, and shape-tests in one
    // pass; for one clean token the result is exactly that token's form.
    extract_entities(first, &[]).into_iter().next()
}

/// Escape a string for a SQLite LIKE pattern with `ESCAPE '\'`.
fn escape_like(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        if matches!(c, '%' | '_' | '\\') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Split text into boundary-delimited tokens exactly the way
/// [`extract_entities`] does (same separator set, same edge trim), but WITHOUT
/// normalization — file paths must match as written, case-sensitively.
fn boundary_tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.' || c == '/' || c == '-'))
        .map(|raw| raw.trim_matches(|c: char| c == '.' || c == '/' || c == '-'))
        .filter(|t| !t.is_empty())
}

fn parse_tags(raw: &str) -> Vec<String> {
    serde_json::from_str::<Vec<String>>(raw).unwrap_or_default()
}

/// Full sha of a git-commit record: content line 1 is `commit <sha> <subject>`.
fn commit_sha(content: &str) -> Option<&str> {
    let first = content.lines().next()?;
    let rest = first.strip_prefix("commit ")?;
    rest.split_whitespace().next()
}

// ============================================================================
// RESOLUTION
// ============================================================================

impl SqliteMemoryStore {
    /// Resolve one handle query. EXACT or PREFIX only — see the module docs
    /// for the resolution order and the exact/prefix rules per kind.
    pub fn resolve_handle(&self, query: &str) -> HandleResolution {
        let query = query.trim();
        if query.is_empty() {
            return HandleResolution::unresolved();
        }

        // 1. uuid-exact: a parseable uuid can only ever be a memory id.
        if uuid::Uuid::parse_str(query).is_ok() {
            return match self.node_ids_where("id = ?1", params![query]) {
                Ok(ids) if !ids.is_empty() => {
                    HandleResolution::resolved(HandleKind::Memory, ids, true)
                }
                _ => HandleResolution::unresolved(),
            };
        }

        // 2. commit sha exact/prefix over git-commit-tagged records.
        match sha_shape(query) {
            Some(ShaShape::TooShort) => {
                let mut r = HandleResolution::unresolved();
                r.kind = HandleKind::Commit;
                r.handle_required = Some(format!(
                    "commit sha prefix must be at least 7 hex characters; '{query}' is too short to resolve unambiguously"
                ));
                return r;
            }
            Some(ShaShape::Prefix) => {
                if let Some(hit) = self.resolve_commit_sha(query) {
                    return hit;
                }
                // No commit matched: fall through (a hex string is never a
                // file/symbol/tag worth retrying, but the remaining rules are
                // cheap and exact, so let them run).
            }
            None => {}
        }

        // 3. boundary-exact token match over content + tags.
        //    File for path-shaped queries, Test for test-shaped ones; both
        //    EXACT ONLY (no prefix: `pyvenv` must not resolve `pyvenv.cfg`).
        if is_path_shaped(query) || is_test_shaped(query) {
            let kind = if is_test_shaped(query) {
                HandleKind::Test
            } else {
                HandleKind::File
            };
            if let Ok(ids) = self.boundary_match_ids(query)
                && !ids.is_empty()
            {
                return HandleResolution::resolved(kind, ids, true);
            }
        }

        // 4. symbol exact/prefix over extracted entities (Code tier only).
        if let Some(normalized) = normalize_identifier(query)
            && matches!(normalized_tier(&normalized), IdentifierTier::Code)
            && let Some(hit) = self.resolve_symbol(&normalized)
        {
            return hit;
        }

        // 5. run id / tool-call id over agent_traces (V18). Exact only.
        if let Ok(found) = self.run_id_exists(query)
            && found
        {
            return HandleResolution::resolved(HandleKind::Run, vec![query.to_string()], true);
        }
        if let Ok(found) = self.tool_call_id_exists(query)
            && found
        {
            return HandleResolution::resolved(HandleKind::ToolCall, vec![query.to_string()], true);
        }

        // 6. tag exact over node tags.
        if let Ok(ids) = self.tag_match_ids(query)
            && !ids.is_empty()
        {
            return HandleResolution::resolved(HandleKind::Tag, ids, true);
        }

        HandleResolution::unresolved()
    }

    /// SELECT id FROM knowledge_nodes WHERE <clause>.
    fn node_ids_where(&self, clause: &str, params: impl rusqlite::Params) -> Result<Vec<String>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let sql = format!("SELECT id FROM knowledge_nodes WHERE {clause}");
        let mut stmt = reader.prepare(&sql)?;
        let ids = stmt
            .query_map(params, |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(ids)
    }

    /// Commit sha resolution: a full 40-hex match is exact; a >=7-char prefix
    /// resolves when unique and reports capped candidates when ambiguous.
    fn resolve_commit_sha(&self, query: &str) -> Option<HandleResolution> {
        let needle = query.to_ascii_lowercase();
        let reader = self.reader.lock().ok()?;
        let mut stmt = reader
            .prepare(
                "SELECT id, content FROM knowledge_nodes \
                 WHERE tags LIKE ?1 ESCAPE '\\'",
            )
            .ok()?;
        let rows = stmt
            .query_map(
                params![format!("%\"{}\"%", escape_like(COMMIT_TAG))],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .ok()?;

        let mut matches: Vec<String> = Vec::new();
        for row in rows.flatten() {
            let (id, content) = row;
            if let Some(sha) = commit_sha(&content) {
                let sha = sha.to_ascii_lowercase();
                if sha == needle || sha.starts_with(&needle) {
                    matches.push(id);
                }
            }
        }
        match matches.len() {
            0 => None,
            1 => Some(HandleResolution::resolved(
                HandleKind::Commit,
                matches,
                needle.len() == 40,
            )),
            // A duplicated full sha still resolves exactly (both rows).
            _ if needle.len() == 40 => Some(HandleResolution::resolved(
                HandleKind::Commit,
                matches,
                true,
            )),
            _ => {
                matches.truncate(MAX_CANDIDATES);
                Some(HandleResolution::ambiguous(
                    HandleKind::Commit,
                    matches
                        .into_iter()
                        .map(|id| (id, HandleKind::Commit))
                        .collect(),
                ))
            }
        }
    }

    /// Boundary-exact token match: the query must appear as a WHOLE token in
    /// node content, or as a whole tag. Coarse SQL LIKE prefilter, then exact
    /// Rust-side boundary verification (LIKE alone is too weak — it would
    /// match `pyvenv.cfg` inside `pyvenv.cfg.bak`).
    fn boundary_match_ids(&self, query: &str) -> Result<Vec<String>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let pattern = format!("%{}%", escape_like(query));
        let mut stmt = reader.prepare(
            "SELECT id, content, tags FROM knowledge_nodes \
             WHERE content LIKE ?1 ESCAPE '\\' OR tags LIKE ?1 ESCAPE '\\'",
        )?;
        let rows = stmt.query_map(params![pattern], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;

        let mut ids = Vec::new();
        for row in rows.flatten() {
            let (id, content, tags) = row;
            let in_content = boundary_tokens(&content).any(|tok| tok == query);
            let in_tags = parse_tags(&tags)
                .iter()
                .any(|tag| tag == query || boundary_tokens(tag).any(|tok| tok == query));
            if in_content || in_tags {
                ids.push(id);
            }
        }
        Ok(ids)
    }

    /// Symbol resolution over extracted entities. Exact match on >=1 node
    /// resolves (exact=true); otherwise unique Code-tier prefix resolves
    /// (exact=false); multiple prefixes report capped candidates. Entities are
    /// restricted to Code tier so word/path queries never leak in.
    fn resolve_symbol(&self, normalized: &str) -> Option<HandleResolution> {
        let reader = self.reader.lock().ok()?;
        let mut stmt = reader
            .prepare("SELECT id, content, tags FROM knowledge_nodes")
            .ok()?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .ok()?;

        let mut exact_ids: Vec<String> = Vec::new();
        let mut prefix_ids: Vec<String> = Vec::new();
        for row in rows.flatten() {
            let (id, content, tags) = row;
            let tags = parse_tags(&tags);
            let entities = extract_entities(&content, &tags);
            let mut exact = false;
            let mut prefix = false;
            for entity in &entities {
                if matches!(normalized_tier(entity), IdentifierTier::Code) {
                    if entity == normalized {
                        exact = true;
                        break;
                    }
                    if entity.starts_with(normalized) {
                        prefix = true;
                    }
                }
            }
            if exact {
                exact_ids.push(id);
            } else if prefix {
                prefix_ids.push(id);
            }
        }

        if !exact_ids.is_empty() {
            return Some(HandleResolution::resolved(
                HandleKind::Symbol,
                exact_ids,
                true,
            ));
        }
        match prefix_ids.len() {
            0 => None,
            1 => Some(HandleResolution::resolved(
                HandleKind::Symbol,
                prefix_ids,
                false,
            )),
            _ => {
                prefix_ids.truncate(MAX_CANDIDATES);
                Some(HandleResolution::ambiguous(
                    HandleKind::Symbol,
                    prefix_ids
                        .into_iter()
                        .map(|id| (id, HandleKind::Symbol))
                        .collect(),
                ))
            }
        }
    }

    /// Does this run id exist in the black-box trace tables (V18)?
    fn run_id_exists(&self, run_id: &str) -> Result<bool> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let in_runs: bool = reader.query_row(
            "SELECT EXISTS(SELECT 1 FROM agent_runs WHERE run_id = ?1)",
            params![run_id],
            |row| row.get(0),
        )?;
        if in_runs {
            return Ok(true);
        }
        let in_traces: bool = reader.query_row(
            "SELECT EXISTS(SELECT 1 FROM agent_traces WHERE run_id = ?1)",
            params![run_id],
            |row| row.get(0),
        )?;
        Ok(in_traces)
    }

    /// Does this tool-call (trace event) id exist in agent_traces?
    fn tool_call_id_exists(&self, id: &str) -> Result<bool> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let found: bool = reader.query_row(
            "SELECT EXISTS(SELECT 1 FROM agent_traces WHERE id = ?1)",
            params![id],
            |row| row.get(0),
        )?;
        Ok(found)
    }

    /// Exact tag match: coarse quoted-LIKE prefilter, then exact Rust-side
    /// comparison against the parsed tag list (LIKE is case-insensitive for
    /// ASCII, so the prefilter is only a superset narrowser).
    fn tag_match_ids(&self, query: &str) -> Result<Vec<String>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let pattern = format!("%\"{}\"%", escape_like(query));
        let mut stmt = reader
            .prepare("SELECT id, tags FROM knowledge_nodes WHERE tags LIKE ?1 ESCAPE '\\'")?;
        let rows = stmt.query_map(params![pattern], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut ids = Vec::new();
        for row in rows.flatten() {
            let (id, tags) = row;
            if parse_tags(&tags).iter().any(|tag| tag == query) {
                ids.push(id);
            }
        }
        Ok(ids)
    }
}

// ============================================================================
// TESTS — exact/prefix/ambiguous/uuid paths, and the NO-fuzzy receipts.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn store() -> (SqliteMemoryStore, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = SqliteMemoryStore::new(Some(dir.path().join("resolver.db"))).unwrap();
        (storage, dir)
    }

    fn insert_node(storage: &SqliteMemoryStore, id: &str, content: &str, tags: &[&str]) {
        let now = chrono::Utc::now().to_rfc3339();
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "INSERT INTO knowledge_nodes
                    (id, content, node_type, created_at, updated_at, last_accessed, tags, scope)
                 VALUES (?1, ?2, 'fact', ?3, ?3, ?3, ?4, 'user')",
                params![id, content, &now, serde_json::json!(tags).to_string()],
            )
            .unwrap();
    }

    fn insert_trace(storage: &SqliteMemoryStore, id: &str, run_id: &str) {
        let now = chrono::Utc::now().to_rfc3339();
        let writer = storage.writer.lock().unwrap();
        writer
            .execute(
                "INSERT INTO agent_traces (id, run_id, seq, event_type, tool, payload, at, created_at)
                 VALUES (?1, ?2, 1, 'mcp.call', 'recall', '{}', 0, ?3)",
                params![id, run_id, &now],
            )
            .unwrap();
        writer
            .execute(
                "INSERT OR IGNORE INTO agent_runs
                    (run_id, first_tool, event_count, started_at, last_at, created_at)
                 VALUES (?1, 'recall', 1, 0, 0, ?2)",
                params![run_id, &now],
            )
            .unwrap();
    }

    const SHA_A: &str = "0123456789abcdef0123456789abcdef01234567";
    const SHA_B: &str = "0123456789fffffffedcba9876543210fedcba98";

    fn seeded() -> (SqliteMemoryStore, TempDir) {
        let (storage, dir) = store();
        // A git-commit record in the canonical git_records::record_content
        // shape: line 1 `commit <sha> <subject>`, then files/symbols lines.
        insert_node(
            &storage,
            "commit-a",
            &format!(
                "commit {SHA_A} speed up cold starts\nfiles: src/main.rs, pyvenv.cfg\nsymbols: src/main.rs/write_file"
            ),
            &["git-commit"],
        );
        // A second commit sharing the first 10 sha chars (prefix-ambiguous).
        insert_node(
            &storage,
            "commit-b",
            &format!("commit {SHA_B} fix the thing\nfiles: src/other.rs"),
            &["git-commit"],
        );
        // A plain memory with an env-var entity and a camel symbol in prose.
        insert_node(
            &storage,
            "mem-env",
            "Set API_TIMEOUT=2 in the deploy env; isAbortError handled in worker",
            &["deploy-env"],
        );
        (storage, dir)
    }

    #[test]
    fn uuid_exact_resolves_memory() {
        let (storage, _dir) = store();
        let id = uuid::Uuid::new_v4().to_string();
        insert_node(&storage, &id, "a plain memory", &[]);
        let r = storage.resolve_handle(&id);
        assert_eq!(r.kind, HandleKind::Memory);
        assert!(r.exact);
        assert_eq!(r.ids, vec![id.clone()]);
        assert!(r.handle_required.is_none());
        // A well-formed uuid with no node behind it resolves to nothing.
        let missing = uuid::Uuid::new_v4().to_string();
        let r = storage.resolve_handle(&missing);
        assert!(r.ids.is_empty() && r.candidates.is_empty());
        assert!(r.handle_required.is_some());
    }

    #[test]
    fn full_sha_is_exact_and_prefix_resolves_unique() {
        let (storage, _dir) = seeded();
        // Full 40-hex: exact.
        let r = storage.resolve_handle(SHA_A);
        assert_eq!(r.kind, HandleKind::Commit);
        assert!(r.exact);
        assert_eq!(r.ids, vec!["commit-a".to_string()]);

        // The two seeded shas share the first 10 chars but diverge at char 11:
        // a 7-prefix ending where they still agree... they agree on
        // "0123456789" then differ, so a prefix strictly shorter than 10 chars
        // is ambiguous, and one covering the difference is unique.
        let shared = &SHA_A[..7]; // 0123456 — both commits start with this
        let r = storage.resolve_handle(shared);
        assert_eq!(r.kind, HandleKind::Commit);
        assert!(r.ids.is_empty(), "shared prefix must be ambiguous");
        let cand_ids: Vec<&str> = r.candidates.iter().map(|(id, _)| id.as_str()).collect();
        assert!(cand_ids.contains(&"commit-a") && cand_ids.contains(&"commit-b"));

        // 12 chars covers the divergence (index 10 differs: 'a' vs 'f'... the
        // shas differ first at position 10; a 12-char prefix of SHA_A excludes
        // SHA_B) -> unique prefix, exact=false.
        let unique = &SHA_A[..12];
        assert_ne!(&SHA_B[..12], unique);
        let r = storage.resolve_handle(unique);
        assert_eq!(r.kind, HandleKind::Commit);
        assert!(!r.exact, "prefix match is not exact");
        assert_eq!(r.ids, vec!["commit-a".to_string()]);
    }

    #[test]
    fn short_hex_is_an_ambiguity_error_not_a_match() {
        let (storage, _dir) = seeded();
        let r = storage.resolve_handle("abc12"); // 5 hex chars, has a digit
        assert_eq!(r.kind, HandleKind::Commit);
        assert!(r.ids.is_empty() && r.candidates.is_empty());
        let msg = r
            .handle_required
            .expect("too-short sha must carry an error");
        assert!(
            msg.contains("7 hex"),
            "message should name the minimum: {msg}"
        );
        // English hex words without digits are NOT sha attempts.
        let r = storage.resolve_handle("face");
        assert_eq!(r.kind, HandleKind::Unknown);
    }

    #[test]
    fn file_exact_resolves_and_no_fuzzy_no_prefix() {
        let (storage, _dir) = seeded();
        // Exact path token in the commit record's files line.
        let r = storage.resolve_handle("pyvenv.cfg");
        assert_eq!(r.kind, HandleKind::File);
        assert!(r.exact);
        assert_eq!(r.ids, vec!["commit-a".to_string()]);
        let r = storage.resolve_handle("src/main.rs");
        assert_eq!(r.kind, HandleKind::File);
        assert_eq!(r.ids, vec!["commit-a".to_string()]);

        // NO fuzzy: "pyvenv" is not the token "pyvenv.cfg" and files are
        // exact-only, so it must resolve nothing.
        let r = storage.resolve_handle("pyvenv");
        assert_eq!(
            r.kind,
            HandleKind::Unknown,
            "word must not fuzzy-match a path"
        );
        assert!(r.ids.is_empty());
        assert!(r.handle_required.is_some());

        // NO prefix on files: "pyvenv.c" is a prefix of "pyvenv.cfg" but files
        // do not prefix-match.
        let r = storage.resolve_handle("pyvenv.c");
        assert_eq!(r.kind, HandleKind::Unknown);
        assert!(r.ids.is_empty());
    }

    #[test]
    fn symbols_resolve_exact_and_prefix_with_normalization() {
        let (storage, _dir) = seeded();
        // camelCase query joins the stored snake entity via normalization.
        let r = storage.resolve_handle("isAbortError");
        assert_eq!(r.kind, HandleKind::Symbol);
        assert!(r.exact);
        assert_eq!(r.ids, vec!["mem-env".to_string()]);
        // Same entity asked for in snake form.
        let r = storage.resolve_handle("is_abort_error");
        assert_eq!(r.kind, HandleKind::Symbol);
        assert_eq!(r.ids, vec!["mem-env".to_string()]);
        // UPPER_SNAKE env var, lowercased on both sides.
        let r = storage.resolve_handle("API_TIMEOUT");
        assert_eq!(r.kind, HandleKind::Symbol);
        assert_eq!(r.ids, vec!["mem-env".to_string()]);

        // Unique prefix resolves, flagged not-exact.
        let r = storage.resolve_handle("is_abort");
        assert_eq!(r.kind, HandleKind::Symbol);
        assert!(!r.exact);
        assert_eq!(r.ids, vec!["mem-env".to_string()]);
        // A prefix that names nothing does not resolve.
        let r = storage.resolve_handle("nonexistent_thing");
        assert!(r.ids.is_empty() && r.candidates.is_empty());
    }

    #[test]
    fn test_names_and_tags_resolve_exact() {
        let (storage, _dir) = seeded();
        insert_node(
            &storage,
            "mem-test",
            "flaky run noted in tests/parser_test.rs; test_extract_versions caught it",
            &[],
        );
        // Test path.
        let r = storage.resolve_handle("tests/parser_test.rs");
        assert_eq!(r.kind, HandleKind::Test);
        assert!(r.exact);
        assert_eq!(r.ids, vec!["mem-test".to_string()]);
        // Test fn name.
        let r = storage.resolve_handle("test_extract_versions");
        assert_eq!(r.kind, HandleKind::Test);
        assert_eq!(r.ids, vec!["mem-test".to_string()]);

        // Tag exact (case-sensitive).
        let r = storage.resolve_handle("git-commit");
        assert_eq!(r.kind, HandleKind::Tag);
        assert!(r.exact);
        let ids = r.ids.clone();
        assert!(ids.contains(&"commit-a".to_string()));
        assert!(ids.contains(&"commit-b".to_string()));
        // Near-miss tag does not resolve.
        let r = storage.resolve_handle("git-commits");
        assert!(r.ids.is_empty());
    }

    #[test]
    fn run_and_tool_call_ids_resolve_exact() {
        let (storage, _dir) = seeded();
        insert_trace(&storage, "call-42", "run-alpha");
        let r = storage.resolve_handle("run-alpha");
        assert_eq!(r.kind, HandleKind::Run);
        assert!(r.exact);
        assert_eq!(r.ids, vec!["run-alpha".to_string()]);
        let r = storage.resolve_handle("call-42");
        assert_eq!(r.kind, HandleKind::ToolCall);
        assert!(r.exact);
        assert_eq!(r.ids, vec!["call-42".to_string()]);
        // Unknown run resolves nothing.
        let r = storage.resolve_handle("run-omega");
        assert!(r.ids.is_empty());
        assert!(r.handle_required.is_some());
    }

    #[test]
    fn free_prose_resolves_nothing() {
        let (storage, _dir) = seeded();
        let r = storage.resolve_handle("how did the build break");
        assert_eq!(r.kind, HandleKind::Unknown);
        assert!(r.ids.is_empty() && r.candidates.is_empty());
        assert_eq!(r.handle_required.as_deref(), Some(HANDLE_REQUIRED_DETAIL));
        // Whitespace-only is empty.
        let r = storage.resolve_handle("   ");
        assert_eq!(r.kind, HandleKind::Unknown);
    }
}
