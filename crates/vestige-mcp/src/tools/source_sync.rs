//! `source_sync` MCP tool (#57) — index an external system into Vestige.
//!
//! Turns Vestige into a durable, offline, provenance-linked retrieval layer
//! over a long-lived external system. GitHub Issues and Redmine are the first
//! reference connectors: Vestige indexes issues, comments/journals, and source
//! metadata as source-aware memories you can search semantically and cite back
//! to the canonical issue URL — re-runnable idempotently (no duplicates) and
//! able to tombstone issues that vanish upstream.
//!
//! Unlike the official GitHub MCP server (a stateless live API proxy), this
//! keeps a local index: searchable offline, embedded for semantic recall,
//! joinable with the rest of your memory, and temporally versioned.
//!
//! ## Auth (security)
//!
//! Tokens are read from environment variables (`GITHUB_TOKEN` /
//! `VESTIGE_GITHUB_TOKEN`, `REDMINE_API_KEY` / `VESTIGE_REDMINE_API_KEY`) and
//! never from tool arguments, so credentials are not logged in the conversation.
//! Public GitHub repositories and anonymous Redmine instances can work without a
//! token/key at lower capability.

use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use vestige_core::ConnectionRecord;
use vestige_core::storage::Storage;

/// JSON schema for the `source_sync` tool.
pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "enum": ["github", "redmine"],
                "description": "'github' (issues) or 'redmine' (a project).",
                "default": "github"
            },
            "repo": {
                "type": "string",
                "description": "GitHub: repository as 'owner/name'."
            },
            "project": {
                "type": "string",
                "description": "Redmine: project slug or id. Host from REDMINE_URL."
            },
            "reconcile": {
                "type": "boolean",
                "description": "Also tombstone local memories for issues gone upstream (full enumeration). Default false.",
                "default": false
            },
            "max_pages": {
                "type": "integer",
                "description": "Max API pages per run (100 issues each); resume a large first sync across calls. Default 10.",
                "default": 10,
                "minimum": 1,
                "maximum": 1000
            }
        },
        "required": []
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SourceSyncArgs {
    #[serde(default = "default_source")]
    source: String,
    #[serde(default)]
    repo: Option<String>,
    #[serde(default)]
    project: Option<String>,
    #[serde(default)]
    reconcile: bool,
    #[serde(default, alias = "max_pages")]
    max_pages: Option<usize>,
}

fn default_source() -> String {
    "github".to_string()
}

/// Read the GitHub token from the environment (never from tool args).
fn github_token() -> Option<String> {
    std::env::var("GITHUB_TOKEN")
        .or_else(|_| std::env::var("VESTIGE_GITHUB_TOKEN"))
        .ok()
        .filter(|s| !s.trim().is_empty())
}

/// Read the Redmine API key from the environment (never from tool args).
fn redmine_api_key() -> Option<String> {
    std::env::var("REDMINE_API_KEY")
        .or_else(|_| std::env::var("VESTIGE_REDMINE_API_KEY"))
        .ok()
        .filter(|s| !s.trim().is_empty())
}

/// Read the Redmine base URL from the environment.
fn redmine_url() -> Option<String> {
    std::env::var("REDMINE_URL")
        .or_else(|_| std::env::var("VESTIGE_REDMINE_URL"))
        .ok()
        .filter(|s| !s.trim().is_empty())
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: SourceSyncArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {e}"))?,
        None => return Err("Missing arguments".to_string()),
    };

    // Clamp to the schema's advertised maximum (1000). The JSON-schema `maximum`
    // is advisory only — a client can still send a larger value — so enforce it
    // here to bound the paginated fetch loop.
    let max_pages = args.max_pages.unwrap_or(10).clamp(1, 1000);

    match args.source.as_str() {
        "github" => {
            let repo = args
                .repo
                .as_deref()
                .ok_or_else(|| "github requires a 'repo' ('owner/name')".to_string())?;
            let (owner, repo) = repo
                .split_once('/')
                .filter(|(o, r)| !o.is_empty() && !r.is_empty())
                .ok_or_else(|| {
                    "repo must be in 'owner/name' form, e.g. 'samvallad33/vestige'".to_string()
                })?;
            execute_github(storage, owner, repo, args.reconcile, max_pages).await
        }
        "redmine" => {
            let project = args
                .project
                .as_deref()
                .filter(|p| !p.trim().is_empty())
                .ok_or_else(|| "redmine requires a 'project' identifier".to_string())?;
            let base_url = redmine_url().ok_or_else(|| {
                "set the REDMINE_URL env var to the Redmine host (e.g. https://redmine.example.com)"
                    .to_string()
            })?;
            execute_redmine(storage, &base_url, project, args.reconcile, max_pages).await
        }
        other => Err(format!(
            "Unsupported source '{other}'. Supported: 'github', 'redmine'."
        )),
    }
}

/// Connectors are feature-gated; surface a clear message when the build omits
/// them rather than failing obscurely.
#[cfg(not(feature = "connectors"))]
async fn execute_github(
    _storage: &Arc<Storage>,
    _owner: &str,
    _repo: &str,
    _reconcile: bool,
    _max_pages: usize,
) -> Result<Value, String> {
    Err(NO_CONNECTORS_MSG.to_string())
}

#[cfg(not(feature = "connectors"))]
async fn execute_redmine(
    _storage: &Arc<Storage>,
    _base_url: &str,
    _project: &str,
    _reconcile: bool,
    _max_pages: usize,
) -> Result<Value, String> {
    Err(NO_CONNECTORS_MSG.to_string())
}

#[cfg(not(feature = "connectors"))]
const NO_CONNECTORS_MSG: &str = "This Vestige build was compiled without the 'connectors' feature. \
     Rebuild with --features connectors to enable source_sync.";

#[cfg(feature = "connectors")]
async fn execute_github(
    storage: &Arc<Storage>,
    owner: &str,
    repo: &str,
    reconcile: bool,
    max_pages: usize,
) -> Result<Value, String> {
    use vestige_core::connectors::github::{GithubConfig, GithubConnector};
    use vestige_core::connectors::run_sync;

    let config = GithubConfig::new(owner, repo).with_token(github_token());
    let connector =
        GithubConnector::new(config).map_err(|e| format!("connector init failed: {e}"))?;

    let report = run_sync(storage.as_ref(), &connector, reconcile, max_pages)
        .await
        .map_err(|e| format!("sync failed: {e}"))?;

    let scope = format!("{owner}/{repo}");
    let total = report.created + report.updated + report.unchanged;
    let authed = github_token().is_some();

    // closed_by chain linking: local-only, deterministic, best-effort. The
    // connector payload carries no closing-PR reference (issues + comments
    // only; no timeline events), so the link is built purely from what is
    // already ingested. Failures degrade to zero links, never a sync error.
    let closed_by_links = link_closed_by_from_local_commits(storage, &scope);

    let summary = format!(
        "Synced {scope}: {} created, {} updated, {} unchanged{}{} ({total} records seen{}).",
        report.created,
        report.updated,
        report.unchanged,
        if report.reconciled {
            format!(", {} tombstoned", report.tombstoned)
        } else {
            String::new()
        },
        if closed_by_links > 0 {
            format!(", {closed_by_links} closed_by links ensured")
        } else {
            String::new()
        },
        if authed { "" } else { ", unauthenticated" },
    );

    Ok(json!({
        "ok": true,
        "summary": summary,
        "source": "github",
        "scope": scope,
        "created": report.created,
        "updated": report.updated,
        "unchanged": report.unchanged,
        "tombstoned": report.tombstoned,
        "reconciled": report.reconciled,
        "closedByLinks": closed_by_links,
        "cursor": report.new_cursor.map(|d| d.to_rfc3339()),
        "authenticated": authed,
        "warnings": report.warnings,
        "hint": if total == 0 && !authed {
            "No records returned. For private repos or higher rate limits, set GITHUB_TOKEN in the server environment."
        } else if report.new_cursor.is_some() && total >= 100 {
            "More may remain — run source_sync again to continue from the saved cursor."
        } else {
            "Search these with the normal search tools; results cite the GitHub issue URL."
        }
    }))
}

// ============================================================================
// closed_by CHAIN LINKING (github issues → closing commits)
// ============================================================================

/// GitHub's issue-closing keywords ("Linking a pull request to an issue"),
/// matched case-insensitively as whole words.
const CLOSING_KEYWORDS: &[&str] = &[
    "close", "closes", "closed", "fix", "fixes", "fixed", "resolve", "resolves", "resolved",
];

/// Does `line` contain any closing keyword as a whole word? ("discloses"
/// contains "closes" but is not one.)
fn line_has_closing_keyword(line_lower: &str) -> bool {
    for keyword in CLOSING_KEYWORDS {
        let mut from = 0usize;
        while let Some(pos) = line_lower[from..].find(keyword) {
            let start = from + pos;
            let end = start + keyword.len();
            let before_ok = line_lower[..start]
                .chars()
                .next_back()
                .is_none_or(|c| !(c.is_alphanumeric() || c == '_'));
            let after_ok = line_lower[end..]
                .chars()
                .next()
                .is_none_or(|c| !(c.is_alphanumeric() || c == '_'));
            if before_ok && after_ok {
                return true;
            }
            from = start + 1;
        }
    }
    false
}

/// Does `line` reference `#<issue_number>` EXACTLY? `#420` must not satisfy
/// issue 42, and the `#` must start the reference (not `##42` or `abc#42`).
fn line_references_issue(line: &str, issue_number: &str) -> bool {
    let target = format!("#{issue_number}");
    let mut from = 0usize;
    while let Some(pos) = line[from..].find(&target) {
        let start = from + pos;
        let end = start + target.len();
        let leading_ok = line[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_alphanumeric() || c == '_' || c == '#'));
        let trailing_ok = line[end..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_ascii_digit());
        if leading_ok && trailing_ok {
            return true;
        }
        from = start + 1;
    }
    false
}

/// Does this commit-record content close `#<issue_number>`?
///
/// Exact and line-scoped: a closing keyword and the bare `#<number>` must
/// appear on the SAME line, so a keyword in the subject never pairs with a
/// reference three lines down in the diff body, and near-miss numbers
/// (`#420` for #42) never match.
fn commit_closes_issue(content: &str, issue_number: &str) -> bool {
    if issue_number.is_empty() || !issue_number.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    content.lines().any(|line| {
        line_has_closing_keyword(&line.to_lowercase()) && line_references_issue(line, issue_number)
    })
}

/// Ensure a `closed_by` edge (issue node → closing commit node) for every
/// closed github issue in `scope` that a locally ingested git-commit record
/// claims to close. Local-only and deterministic: no remote fetch, no new
/// API calls — the connector payload carries no closing-PR reference, so the
/// link is built from `memory_connections` neighbors that already exist in
/// the store. Edges are written through the existing `save_connection`
/// (`INSERT OR REPLACE`), so re-running a sync re-ensures the same edge
/// instead of duplicating it. Returns the number of edges ensured.
pub(crate) fn link_closed_by_from_local_commits(storage: &Arc<Storage>, scope: &str) -> usize {
    let Ok(issues) = storage.closed_issue_nodes("github", scope) else {
        return 0;
    };
    if issues.is_empty() {
        return 0;
    }
    let Ok(commits) = storage.git_commit_nodes(1_000) else {
        return 0;
    };
    let now = chrono::Utc::now();
    let mut linked = 0usize;
    for issue in &issues {
        for commit in &commits {
            if !commit_closes_issue(&commit.content, &issue.issue_number) {
                continue;
            }
            let edge = ConnectionRecord {
                source_id: issue.node_id.clone(),
                target_id: commit.node_id.clone(),
                strength: 1.0,
                link_type: "closed_by".to_string(),
                created_at: now,
                last_activated: now,
                activation_count: 0,
            };
            if storage.save_connection(&edge).is_ok() {
                linked += 1;
            }
        }
    }
    linked
}

#[cfg(feature = "connectors")]
async fn execute_redmine(
    storage: &Arc<Storage>,
    base_url: &str,
    project: &str,
    reconcile: bool,
    max_pages: usize,
) -> Result<Value, String> {
    use vestige_core::connectors::redmine::{RedmineConfig, RedmineConnector};
    use vestige_core::connectors::run_sync;

    let config = RedmineConfig::new(base_url, project).with_api_key(redmine_api_key());
    let connector =
        RedmineConnector::new(config).map_err(|e| format!("connector init failed: {e}"))?;

    let report = run_sync(storage.as_ref(), &connector, reconcile, max_pages)
        .await
        .map_err(|e| format!("sync failed: {e}"))?;

    let total = report.created + report.updated + report.unchanged;
    let authed = redmine_api_key().is_some();

    let summary = format!(
        "Synced redmine project '{project}': {} created, {} updated, {} unchanged{} ({total} records seen{}).",
        report.created,
        report.updated,
        report.unchanged,
        if report.reconciled {
            format!(", {} tombstoned", report.tombstoned)
        } else {
            String::new()
        },
        if authed { "" } else { ", anonymous" },
    );

    Ok(json!({
        "ok": true,
        "summary": summary,
        "source": "redmine",
        "scope": project,
        "created": report.created,
        "updated": report.updated,
        "unchanged": report.unchanged,
        "tombstoned": report.tombstoned,
        "reconciled": report.reconciled,
        "cursor": report.new_cursor.map(|d| d.to_rfc3339()),
        "authenticated": authed,
        "warnings": report.warnings,
        "hint": if total == 0 && !authed {
            "No records returned. Set REDMINE_API_KEY (and confirm the REST API is enabled on the instance) for private projects."
        } else if report.new_cursor.is_some() && total >= 100 {
            "More may remain — run source_sync again to continue from the saved cursor."
        } else {
            "Search these with the normal search tools; results cite the Redmine issue URL."
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use vestige_core::{IngestInput, SourceEnvelope};

    fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage =
            vestige_core::open_storage(Some(dir.path().join("source_sync_test.db"))).unwrap();
        (storage, dir)
    }

    fn ingest_commit(storage: &Arc<Storage>, sha: &str, subject: &str, files: &str) -> String {
        let content = format!("commit {sha} {subject}\nfiles: {files}");
        storage
            .ingest(IngestInput {
                content,
                node_type: "fact".to_string(),
                tags: vec!["git-commit".to_string()],
                ..Default::default()
            })
            .unwrap()
            .id
    }

    fn upsert_issue(storage: &Arc<Storage>, number: u64, state: &str, scope: &str) -> String {
        // SourceEnvelope is #[non_exhaustive]; build via Default + field set.
        let mut envelope = SourceEnvelope::default();
        envelope.source_system = Some("github".to_string());
        envelope.source_id = Some(number.to_string());
        envelope.source_url = Some(format!("https://github.com/{scope}/issues/{number}"));
        envelope.content_hash = Some(format!("h-{number}-{state}"));
        envelope.source_project = Some(scope.to_string());
        envelope.source_type = Some("issue".to_string());
        storage
            .upsert_by_source(IngestInput {
                content: format!("[{scope}#{number}] Something broke"),
                node_type: "event".to_string(),
                tags: vec!["github".to_string(), format!("state:{state}")],
                source_envelope: Some(envelope),
                ..Default::default()
            })
            .unwrap()
            .node_id
    }

    // ========================================================================
    // MATCHER TESTS
    // ========================================================================

    #[test]
    fn source_sync_matcher_accepts_closing_keyword_forms() {
        for subject in [
            "fix: closes #42",
            "Fixes #42",
            "fixed #42 in the pool",
            "close #42",
            "Closed #42",
            "resolve: resolves #42",
            "resolved #42 and #43",
        ] {
            assert!(
                commit_closes_issue(&format!("commit {subject}\nfiles: x.rs"), "42"),
                "'{subject}' must close #42"
            );
        }
    }

    #[test]
    fn source_sync_matcher_rejects_near_misses() {
        // Wrong number with a digit tail: #420 is not #42.
        assert!(!commit_closes_issue(
            "commit fix: closes #420\nfiles: x.rs",
            "42"
        ));
        assert!(!commit_closes_issue(
            "commit fix: closes #4\nfiles: x.rs",
            "42"
        ));
        // Keyword hidden inside another word.
        assert!(!commit_closes_issue(
            "commit discloses #42\nfiles: x.rs",
            "42"
        ));
        assert!(!commit_closes_issue(
            "commit prefixes #42\nfiles: x.rs",
            "42"
        ));
        // Reference without a closing keyword on the line.
        assert!(!commit_closes_issue(
            "commit see #42 for context\nfiles: x.rs",
            "42"
        ));
        // Keyword and reference on DIFFERENT lines never pair.
        assert!(!commit_closes_issue(
            "commit fixes the thing\nsee #42 for context",
            "42"
        ));
        // Glued references are not bare references.
        assert!(!commit_closes_issue(
            "commit fix: abc#42\nfiles: x.rs",
            "42"
        ));
        assert!(!commit_closes_issue("commit fix: ##42\nfiles: x.rs", "42"));
        // Non-numeric / empty issue numbers never match.
        assert!(!commit_closes_issue("commit fix: closes #42", "abc"));
        assert!(!commit_closes_issue("commit fix: closes #42", ""));
    }

    // ========================================================================
    // LINKING TESTS
    // ========================================================================

    #[tokio::test]
    async fn source_sync_closed_by_edge_appears_after_linking() {
        let (storage, _dir) = test_storage();
        let issue_node = upsert_issue(&storage, 42, "closed", "o/r");
        let commit_node =
            ingest_commit(&storage, &"a".repeat(40), "fix: closes #42", "src/pool.rs");
        // Decoys: open issue, unrelated commit.
        let open_node = upsert_issue(&storage, 43, "open", "o/r");
        let unrelated = ingest_commit(&storage, &"b".repeat(40), "docs: readme", "README.md");

        let linked = link_closed_by_from_local_commits(&storage, "o/r");
        assert_eq!(linked, 1, "exactly the closes-#42 pair links");

        let edges = storage.get_connections_for_memory(&issue_node).unwrap();
        let closed_by: Vec<&ConnectionRecord> = edges
            .iter()
            .filter(|e| e.link_type == "closed_by")
            .collect();
        assert_eq!(closed_by.len(), 1, "one closed_by edge, not duplicates");
        assert_eq!(
            closed_by[0].source_id, issue_node,
            "source is the issue node"
        );
        assert_eq!(
            closed_by[0].target_id, commit_node,
            "target is the commit node"
        );

        // Negative: the open issue and the unrelated commit carry no edges.
        assert!(
            storage
                .get_connections_for_memory(&open_node)
                .unwrap()
                .is_empty(),
            "an open issue is never closed_by-linked"
        );
        assert!(
            storage
                .get_connections_for_memory(&unrelated)
                .unwrap()
                .is_empty(),
            "a commit that closes nothing gets no edge"
        );

        // Idempotent: re-running re-ensures the same edge, never a second one.
        let again = link_closed_by_from_local_commits(&storage, "o/r");
        assert_eq!(again, 1);
        let edges = storage.get_connections_for_memory(&issue_node).unwrap();
        assert_eq!(
            edges.iter().filter(|e| e.link_type == "closed_by").count(),
            1,
            "INSERT OR REPLACE must not duplicate the edge"
        );
    }

    #[tokio::test]
    async fn source_sync_closed_by_silent_without_candidates() {
        let (storage, _dir) = test_storage();
        // Closed issue but no commit records at all: no links, no noise.
        upsert_issue(&storage, 7, "closed", "o/r");
        assert_eq!(link_closed_by_from_local_commits(&storage, "o/r"), 0);
        // Wrong scope: the issue is not this connector instance's.
        let commit_node = ingest_commit(&storage, &"c".repeat(40), "fix: closes #7", "src/x.rs");
        assert_eq!(
            link_closed_by_from_local_commits(&storage, "other/scope"),
            0
        );
        assert!(
            storage
                .get_connections_for_memory(&commit_node)
                .unwrap()
                .is_empty()
        );
    }
}
