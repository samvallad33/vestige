//! Session-scoped queries backing the `session_start` context packet, plus
//! the local-only lookups the `source_sync` `closed_by` chain-linking needs.
//!
//! The `session_start` tool answers "what do I need before the first
//! substantive call?" in one shot. Two of its sections need joins the generic
//! search paths cannot express:
//!
//! - **open failures touching changed files** — failure-like memories
//!   ([`crate::advanced::retroactive_backfill::looks_like_failure`]) whose
//!   recorded files (source anchors in `code_memory_anchors`, or the `files:`
//!   line of a git-commit record) intersect the caller's changed set by
//!   EXACT path equality. A prefix or near-match (`src/pool.rs` vs
//!   `src/pool.rs.bak`) deliberately does not hit: the section exists to say
//!   "this failure lives in a file you are editing right now", and a fuzzy
//!   match would turn that into noise.
//! - **last session failed calls** — `mcp.call` trace rows whose payload
//!   carries `success: false`, from the latest (or a given) run. Rows whose
//!   payload has no `success` field are not failed calls; today's recorder
//!   writes plain `mcp.call` events, so this section stays empty until a
//!   recorder adopts [`SqliteMemoryStore::append_mcp_call_outcome`]. That
//!   outcome row keeps the exact `mcp.call` JSON shape (extra fields are
//!   ignored by the event's deserializer), so Black Box replay still parses
//!   it and `agent_traces` ordering is preserved.
//!
//! `source_sync`'s `closed_by` linking is local-only and deterministic: the
//! GitHub connector payload does not carry the closing PR (it fetches
//! issues with their comments but no timeline events), so the link is built
//! from the rows the store already holds: closed issue nodes and
//! git-commit records.  These queries hand the raw pairs to the caller;
//! the keyword matcher and edge writing live in the tool.

use rusqlite::{OptionalExtension, params};

use super::sqlite::SqliteMemoryStore;
use super::{Result, StorageError};

/// Page size for the bounded failure-node scan.
const FAILURE_SCAN_PAGE: i32 = 500;
/// Hard cap on nodes scanned per `open_failures_touching` call. The scan is
/// chunked `get_all_nodes` pages filtered in Rust; the cap bounds worst-case
/// work on a huge store while covering every realistic session-start store.
const FAILURE_SCAN_MAX_NODES: usize = 10_000;
/// Preview cap for failure content (chars, cut on a UTF-8 boundary).
const PREVIEW_MAX_CHARS: usize = 120;
/// Failed calls returned per query (the "last 20").
pub const FAILED_CALLS_MAX: usize = 20;
/// Trace rows examined (newest first) while hunting for failed calls, so a
/// pathological run cannot make session_start scan the whole black box.
const FAILED_CALLS_SCAN_ROWS: i64 = 2_000;
/// Cap for the error excerpt (chars, cut on a UTF-8 boundary).
const ERROR_EXCERPT_MAX_CHARS: usize = 160;
/// Bound on commit records considered by the `closed_by` lookup.
const COMMIT_LOOKUP_LIMIT: i64 = 1_000;

// `OpenFailureTouching`, `FailedToolCall`, `ClosedIssueNode`, and
// `GitCommitNode` are defined in (and re-exported from)
// `crate::storage::types`.
pub use crate::storage::types::{ClosedIssueNode, FailedToolCall, GitCommitNode, OpenFailureTouching};

/// Cut `text` to at most `max` chars on a UTF-8 boundary.
fn cap_chars(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let end = text.floor_char_boundary(max);
    text[..end].to_string()
}

/// First-line preview of a memory's content, capped.
fn content_preview(content: &str) -> String {
    let first_line = content.lines().next().unwrap_or("").trim();
    cap_chars(first_line, PREVIEW_MAX_CHARS)
}

/// Paths listed on a git-commit record's `files:` line (see
/// [`crate::advanced::git_records::record_content`]). The ` (+N more)`
/// truncation suffix rides on the last entry and is stripped; it names a
/// count, not a path.
fn files_line_paths(content: &str) -> Vec<&str> {
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("files: ") {
            let listed = rest.split(" (+").next().unwrap_or(rest);
            return listed
                .split(',')
                .map(str::trim)
                .filter(|path| !path.is_empty())
                .collect();
        }
    }
    Vec::new()
}

/// Parse a stored tags column (JSON array text) into owned tags. Returns an
/// empty vec on a malformed value — a bad tags blob must not break the query.
fn parse_tags(raw: &str) -> Vec<String> {
    serde_json::from_str::<Vec<String>>(raw).unwrap_or_default()
}

impl SqliteMemoryStore {
    /// Failure-like memories ("open": not superseded, not suppressed,
    /// currently valid) whose recorded files intersect `changed_files` by
    /// EXACT path equality.
    ///
    /// A failure's recorded files are, in priority order:
    /// 1. its source anchors (`code_memory_anchors.file_path`);
    /// 2. the entries of a `files:` line in its content (git-commit records).
    ///
    /// The returned `anchor` names the first matching location in the
    /// anchor's deterministic order (file_path, then start_line), or the
    /// matching `files:` entry when no anchor matched. Results keep the
    /// scan's newest-first order.
    pub fn open_failures_touching(
        &self,
        changed_files: &[String],
    ) -> Result<Vec<OpenFailureTouching>> {
        let wanted: std::collections::HashSet<&str> = changed_files
            .iter()
            .map(|f| f.trim())
            .filter(|f| !f.is_empty())
            .collect();
        if wanted.is_empty() {
            return Ok(Vec::new());
        }

        let superseded = self.superseded_node_ids()?;
        let now = chrono::Utc::now();
        let mut out = Vec::new();
        let mut scanned = 0usize;
        let mut offset = 0i32;

        loop {
            let page = self.get_all_nodes(FAILURE_SCAN_PAGE, offset)?;
            let page_len = page.len();
            if page.is_empty() {
                break;
            }

            // One batched anchor query per page, for the failures on it only.
            let failure_ids: Vec<String> = page
                .iter()
                .filter(|n| {
                    crate::advanced::retroactive_backfill::looks_like_failure(
                        &n.content,
                        &n.tags,
                    )
                })
                .map(|n| n.id.clone())
                .collect();
            let anchors = self.code_anchors_for_nodes(&failure_ids)?;

            for node in &page {
                if !crate::advanced::retroactive_backfill::looks_like_failure(
                    &node.content,
                    &node.tags,
                ) {
                    continue;
                }
                // "Open": still the live version of what it knows.
                if superseded.contains(&node.id)
                    || node.suppression_count > 0
                    || node.valid_from.is_some_and(|t| t > now)
                    || node.valid_until.is_some_and(|t| t <= now)
                {
                    continue;
                }

                let mut matched: Option<String> = None;
                if let Some(node_anchors) = anchors.get(&node.id) {
                    for anchor in node_anchors {
                        if wanted.contains(anchor.file_path.trim()) {
                            matched = Some(match anchor.symbol.as_deref() {
                                Some(symbol) if !symbol.is_empty() => {
                                    format!("{}:{}", anchor.file_path, symbol)
                                }
                                _ => anchor.file_path.clone(),
                            });
                            break;
                        }
                    }
                }
                if matched.is_none() {
                    for path in files_line_paths(&node.content) {
                        if wanted.contains(path) {
                            matched = Some(path.to_string());
                            break;
                        }
                    }
                }

                if let Some(anchor) = matched {
                    out.push(OpenFailureTouching {
                        id: node.id.clone(),
                        content_preview: content_preview(&node.content),
                        anchor: Some(anchor),
                    });
                }
            }

            scanned += page_len;
            offset += FAILURE_SCAN_PAGE;
            if page_len < FAILURE_SCAN_PAGE as usize || scanned >= FAILURE_SCAN_MAX_NODES {
                break;
            }
        }
        Ok(out)
    }

    /// Failed tool calls of a run: `agent_traces` rows whose serialized
    /// `mcp.call` payload carries `success: false`.
    ///
    /// `run_id = None` selects the latest run by `agent_runs.last_at`. The
    /// last [`FAILED_CALLS_MAX`] failed calls are returned in chronological
    /// order (oldest first), so the section reads like the run unfolded.
    pub fn last_session_failed_calls(
        &self,
        run_id: Option<&str>,
    ) -> Result<Vec<FailedToolCall>> {
        let run = match run_id {
            Some(given) => given.to_string(),
            None => {
                let reader = self
                    .reader
                    .lock()
                    .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
                match reader
                    .query_row(
                        "SELECT run_id FROM agent_runs ORDER BY last_at DESC LIMIT 1",
                        [],
                        |row| row.get::<_, String>(0),
                    )
                    .optional()?
                {
                    Some(latest) => latest,
                    None => return Ok(Vec::new()),
                }
            }
        };

        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT tool, payload, at FROM agent_traces WHERE run_id = ?1 \
             ORDER BY seq DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![run, FAILED_CALLS_SCAN_ROWS], |row| {
            Ok((
                row.get::<_, Option<String>>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })?;

        let mut failed: Vec<FailedToolCall> = Vec::new();
        for row in rows {
            let (tool_column, payload, at) = row?;
            if failed.len() >= FAILED_CALLS_MAX {
                break;
            }
            if let Some(call) = failed_call_from_payload(&run, tool_column, &payload, at) {
                failed.push(call);
            }
        }
        failed.reverse();
        Ok(failed)
    }

    /// Append a `mcp.call` trace row that carries its outcome
    /// (`success`, and `error` when it failed).
    ///
    /// The payload is the standard `MemoryTraceEvent::McpCall` JSON shape
    /// extended with `success` / `error` fields; the event deserializer
    /// ignores unknown fields, so Black Box replay (`get_trace`) still
    /// parses these rows. `argsHash` is the opaque constant `"outcome"` —
    /// it is never derived from raw args, so no prompt content or secret
    /// can leak through it. Ordering and the run roll-up mirror
    /// [`SqliteMemoryStore::append_trace_event`].
    pub fn append_mcp_call_outcome(
        &self,
        run_id: &str,
        tool: &str,
        success: bool,
        error: Option<&str>,
        at_ms: i64,
    ) -> Result<()> {
        let mut payload = serde_json::json!({
            "type": "mcp.call",
            "runId": run_id,
            "tool": tool,
            "argsHash": "outcome",
            "at": at_ms,
            "success": success,
        });
        if let Some(error) = error.filter(|e| !e.is_empty()) {
            payload["error"] = serde_json::Value::String(error.to_string());
        }

        let now = chrono::Utc::now().to_rfc3339();
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let seq: i64 = writer.query_row(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM agent_traces WHERE run_id = ?1",
            params![run_id],
            |r| r.get(0),
        )?;
        writer.execute(
            "INSERT INTO agent_traces (id, run_id, seq, event_type, tool, payload, at, created_at)
             VALUES (?1, ?2, ?3, 'mcp.call', ?4, ?5, ?6, ?7)",
            params![
                uuid::Uuid::new_v4().to_string(),
                run_id,
                seq,
                tool,
                payload.to_string(),
                at_ms,
                now,
            ],
        )?;
        writer.execute(
            "INSERT INTO agent_runs (run_id, first_tool, event_count, retrieved_count,
                 suppressed_count, write_count, veto_count, started_at, last_at, created_at)
             VALUES (?1, ?2, 1, 0, 0, 0, 0, ?3, ?3, ?4)
             ON CONFLICT(run_id) DO UPDATE SET
                 first_tool = COALESCE(agent_runs.first_tool, excluded.first_tool),
                 event_count = agent_runs.event_count + 1,
                 last_at = MAX(agent_runs.last_at, ?3)",
            params![run_id, tool, at_ms, now],
        )?;
        Ok(())
    }

    /// Live closed-issue nodes of one external source scope (e.g.
    /// `source_system = "github"`, `scope = "owner/repo"`). "Closed" is the
    /// connector-recorded `state:closed` tag, matched exactly after parsing
    /// the tags JSON (the SQL `LIKE` is only a pre-filter).
    pub fn closed_issue_nodes(
        &self,
        source_system: &str,
        scope: &str,
    ) -> Result<Vec<ClosedIssueNode>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT id, source_id, tags FROM knowledge_nodes \
             WHERE source_system = ?1 AND source_project = ?2 AND valid_until IS NULL \
             ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map(params![source_system, scope], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (node_id, source_id, tags_raw) = row?;
            let Some(issue_number) = source_id.filter(|s| !s.trim().is_empty()) else {
                continue;
            };
            if parse_tags(&tags_raw)
                .iter()
                .any(|t| t == "state:closed")
            {
                out.push(ClosedIssueNode {
                    node_id,
                    issue_number: issue_number.trim().to_string(),
                });
            }
        }
        Ok(out)
    }

    /// Locally ingested git-commit records (tag `git-commit`, matched exactly
    /// after parsing the tags JSON), newest first, bounded.
    pub fn git_commit_nodes(&self, limit: usize) -> Result<Vec<GitCommitNode>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT id, content, tags FROM knowledge_nodes \
             WHERE tags LIKE '%git-commit%' ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit.min(COMMIT_LOOKUP_LIMIT as usize) as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (node_id, content, tags_raw) = row?;
            if parse_tags(&tags_raw).iter().any(|t| t == "git-commit") {
                out.push(GitCommitNode { node_id, content });
            }
        }
        Ok(out)
    }
}

/// Build a [`FailedToolCall`] from one trace row's payload, or `None` when
/// the row is not a failed call (payload unparseable, or no
/// `success: false`).
fn failed_call_from_payload(
    run_id: &str,
    tool_column: Option<String>,
    payload: &str,
    at: i64,
) -> Option<FailedToolCall> {
    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    if value.get("success") != Some(&serde_json::Value::Bool(false)) {
        return None;
    }
    let tool = value
        .get("tool")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .or(tool_column)
        .unwrap_or_default();
    let error_excerpt = value
        .get("error")
        .map(|error| match error {
            serde_json::Value::String(text) => cap_chars(text, ERROR_EXCERPT_MAX_CHARS),
            serde_json::Value::Object(_) => error
                .get("message")
                .or_else(|| error.get("detail"))
                .and_then(serde_json::Value::as_str)
                .map(|text| cap_chars(text, ERROR_EXCERPT_MAX_CHARS))
                .unwrap_or_default(),
            _ => String::new(),
        })
        .unwrap_or_default();
    Some(FailedToolCall {
        run_id: run_id.to_string(),
        tool,
        at,
        error_excerpt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IngestInput;
    use crate::codebase::CodeAnchor;
    use crate::memory::SourceEnvelope;

    fn store() -> (tempfile::TempDir, SqliteMemoryStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteMemoryStore::new(Some(dir.path().join("session_queries.db"))).unwrap();
        (dir, store)
    }

    fn ingest(store: &SqliteMemoryStore, content: &str, tags: &[&str]) -> String {
        store
            .ingest(IngestInput {
                content: content.to_string(),
                tags: tags.iter().map(|t| t.to_string()).collect(),
                ..Default::default()
            })
            .unwrap()
            .id
    }

    fn anchor(node_id: &str, file_path: &str, symbol: Option<&str>) -> CodeAnchor {
        CodeAnchor {
            id: format!("anchor-{file_path}"),
            node_id: node_id.to_string(),
            file_path: file_path.to_string(),
            symbol: symbol.map(str::to_string),
            symbol_kind: None,
            start_line: Some(1),
            end_line: Some(2),
            span_lines: Some(2),
            content_hash: None,
            captured_at: chrono::Utc::now(),
            last_verified_at: None,
            last_status: None,
        }
    }

    #[test]
    fn files_line_parses_paths_and_strips_the_more_suffix() {
        assert_eq!(
            files_line_paths("commit abc fix: thing\nfiles: a.rs, src/b.rs (+2 more)\nsymbols: x"),
            vec!["a.rs", "src/b.rs"]
        );
        assert!(files_line_paths("no files line here").is_empty());
    }

    #[test]
    fn open_failures_match_anchors_and_files_lines_exactly() {
        let (_dir, store) = store();
        let failure = ingest(
            &store,
            "Deploy failed: connection pool saturated at 100%.",
            &[],
        );
        store.record_code_anchors(&[anchor(&failure, "src/pool.rs", Some("acquire"))]).unwrap();

        let commit_failure = ingest(
            &store,
            "commit 1111111111111111111111111111111111111111 cleanup after crash\nfiles: src/other.rs",
            &["git-commit"],
        );

        // Anchor match (with symbol), files-line match, and a near-miss that
        // must NOT match: exact equality only.
        let hits = store
            .open_failures_touching(&["src/pool.rs".to_string()])
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, failure);
        assert_eq!(hits[0].anchor.as_deref(), Some("src/pool.rs:acquire"));
        assert!(hits[0].content_preview.contains("Deploy failed"));

        let hits = store
            .open_failures_touching(&["src/other.rs".to_string()])
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, commit_failure);
        assert_eq!(hits[0].anchor.as_deref(), Some("src/other.rs"));

        // Prefix/suffix near-misses are noise, not matches.
        assert!(store
            .open_failures_touching(&["src/pool.rs.bak".to_string(), "pool.rs".to_string()])
            .unwrap()
            .is_empty());
        // Absent/empty changed set: nothing, not noise.
        assert!(store.open_failures_touching(&[]).unwrap().is_empty());
    }

    #[test]
    fn open_failures_skip_non_failures_and_closed_memories() {
        let (_dir, store) = store();
        let quiet = ingest(&store, "Prefer Rust for systems work.", &[]);
        store.record_code_anchors(&[anchor(&quiet, "src/pool.rs", None)]).unwrap();
        assert!(store
            .open_failures_touching(&["src/pool.rs".to_string()])
            .unwrap()
            .is_empty(), "a non-failure memory anchored to the file must not surface");

        let failure = ingest(&store, "Build broke on CI.", &[]);
        store.record_code_anchors(&[anchor(&failure, "src/ci.rs", None)]).unwrap();
        store.suppress_memory(&failure).unwrap();
        assert!(store
            .open_failures_touching(&["src/ci.rs".to_string()])
            .unwrap()
            .is_empty(), "a suppressed failure is not open");
    }

    #[test]
    fn failed_calls_latest_run_last_twenty_chronological() {
        let (_dir, store) = store();
        store.append_mcp_call_outcome("run_a", "recall", true, None, 100).unwrap();
        store.append_mcp_call_outcome("run_a", "backfill", false, Some("scope must be non-empty"), 110).unwrap();
        store.append_mcp_call_outcome("run_b", "memory", false, Some("NotFound: abc"), 200).unwrap();

        // Latest run is run_b (last_at 200): only its failed call appears.
        let latest = store.last_session_failed_calls(None).unwrap();
        assert_eq!(latest.len(), 1);
        assert_eq!(latest[0].run_id, "run_b");
        assert_eq!(latest[0].tool, "memory");
        assert_eq!(latest[0].error_excerpt, "NotFound: abc");
        assert_eq!(latest[0].at, 200);

        // Explicit run: run_a's successful call is filtered out.
        let run_a = store.last_session_failed_calls(Some("run_a")).unwrap();
        assert_eq!(run_a.len(), 1);
        assert_eq!(run_a[0].tool, "backfill");
        assert!(run_a[0].error_excerpt.contains("scope must be non-empty"));

        // Unknown run: empty, not an error.
        assert!(store.last_session_failed_calls(Some("run_zz")).unwrap().is_empty());

        // The outcome rows still replay as plain mcp.call events.
        let events = store.get_trace("run_a").unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind(), "mcp.call");
    }

    #[test]
    fn failed_calls_cap_at_twenty() {
        let (_dir, store) = store();
        for i in 0..25 {
            store.append_mcp_call_outcome("run_c", &format!("tool_{i}"), false, Some("boom"), 1000 + i).unwrap();
        }
        let calls = store.last_session_failed_calls(Some("run_c")).unwrap();
        assert_eq!(calls.len(), FAILED_CALLS_MAX);
        // The LAST 20 in chronological order: tool_5..tool_24.
        assert_eq!(calls.first().unwrap().tool, "tool_5");
        assert_eq!(calls.last().unwrap().tool, "tool_24");
    }

    #[test]
    fn closed_issue_and_commit_lookups_are_exact() {
        let (_dir, store) = store();
        let envelope = SourceEnvelope {
            source_system: Some("github".to_string()),
            source_id: Some("42".to_string()),
            source_url: Some("https://example/42".to_string()),
            content_hash: Some("h42".to_string()),
            source_project: Some("o/r".to_string()),
            source_type: Some("issue".to_string()),
            ..Default::default()
        };
        let closed = store
            .upsert_by_source(IngestInput {
                content: "[o/r#42] Thing".to_string(),
                tags: vec!["github".into(), "state:closed".into()],
                source_envelope: Some(envelope),
                ..Default::default()
            })
            .unwrap();
        // Same scope, but open — must not be returned.
        store
            .upsert_by_source(IngestInput {
                content: "[o/r#43] Other".to_string(),
                tags: vec!["github".into(), "state:open".into()],
                source_envelope: Some(SourceEnvelope {
                    source_system: Some("github".to_string()),
                    source_id: Some("43".to_string()),
                    content_hash: Some("h43".to_string()),
                    source_project: Some("o/r".to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .unwrap();

        let issues = store.closed_issue_nodes("github", "o/r").unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue_number, "42");
        assert_eq!(issues[0].node_id, closed.node_id);

        let commit = ingest(
            &store,
            "commit 2222222222222222222222222222222222222222 fix: closes #42\nfiles: src/x.rs",
            &["git-commit"],
        );
        let commits = store.git_commit_nodes(10).unwrap();
        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].node_id, commit);
        assert!(commits[0].content.contains("closes #42"));
    }
}
