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

use vestige_core::storage::Storage;

/// JSON schema for the `source_sync` tool.
pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "enum": ["github", "redmine"],
                "description": "External system to sync: 'github' (GitHub Issues) or 'redmine' (a Redmine project).",
                "default": "github"
            },
            "repo": {
                "type": "string",
                "description": "GitHub only: repository as 'owner/name', e.g. 'samvallad33/vestige'."
            },
            "project": {
                "type": "string",
                "description": "Redmine only: project identifier (slug or numeric id) to sync. The Redmine host comes from the REDMINE_URL env var."
            },
            "reconcile": {
                "type": "boolean",
                "description": "Also tombstone local memories for issues no longer visible upstream (an extra full enumeration pass). Default false on incremental syncs.",
                "default": false
            },
            "max_pages": {
                "type": "integer",
                "description": "Max API pages to fetch this run (each page is up to 100 issues). Lets a first sync of a large project be resumed across calls. Default 10.",
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

    let summary = format!(
        "Synced {scope}: {} created, {} updated, {} unchanged{} ({total} records seen{}).",
        report.created,
        report.updated,
        report.unchanged,
        if report.reconciled {
            format!(", {} tombstoned", report.tombstoned)
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
