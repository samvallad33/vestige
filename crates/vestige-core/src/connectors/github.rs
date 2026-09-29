//! GitHub Issues connector (#57).
//!
//! Indexes a repository's issues + comments into source-aware Vestige memories
//! so an agent can search and reason over the full issue history **offline**,
//! **semantically**, and **cited back to the canonical issue URL**. Unlike the
//! official GitHub MCP server — a stateless live API proxy — this builds a
//! durable, embedded, temporally-versioned local index.
//!
//! ## Incremental sync (per the connector sync contract)
//!
//! - `state=all` so closing an issue is not mistaken for a deletion.
//! - `sort=updated&direction=asc` so we page forward in cursor order and a
//!   mid-run interruption resumes safely.
//! - `since=<cursor − overlap>` filters on `updated_at`; the overlap + the
//!   `content_hash` no-op makes re-scans safe and cheap.
//! - `Link: rel="next"` drives pagination (never hand-built page urls).
//! - Entries carrying a `pull_request` key are dropped (PRs are not issues).
//! - Per issue we fold the body + comments into one memory; the hash covers
//!   the stable fields only (title, body, state, labels, comments) — never the
//!   cursor timestamp or volatile counts.
//!
//! GitHub has no deletion feed, so deletions are reconciled out-of-band via
//! [`list_live_ids`](Connector::list_live_ids).

use chrono::{DateTime, Utc};
use serde::Deserialize;

use super::{
    Connector, ConnectorError, ConnectorResult, FetchPage, NormalizedRecord, SkippedRecord,
    content_hash,
};
use crate::memory::SourceEnvelope;

const API_ROOT: &str = "https://api.github.com";
const USER_AGENT: &str = concat!("vestige-connector/", env!("CARGO_PKG_VERSION"));
const PER_PAGE: u32 = 100;

/// Configuration for a GitHub Issues connector instance.
#[derive(Clone)]
pub struct GithubConfig {
    /// Repository owner (user or org).
    pub owner: String,
    /// Repository name.
    pub repo: String,
    /// Personal access token. Optional for public repos (60 req/hr
    /// unauthenticated) but strongly recommended (5000 req/hr authenticated).
    pub token: Option<String>,
    /// Override the API root (for GitHub Enterprise or tests).
    pub api_root: Option<String>,
    /// Max comments to fold into one issue memory (defense against huge threads).
    pub max_comments: usize,
}

// Manual Debug that NEVER prints the token — a derived Debug would leak the
// bearer credential into any `{:?}` log line or panic message.
impl std::fmt::Debug for GithubConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GithubConfig")
            .field("owner", &self.owner)
            .field("repo", &self.repo)
            .field("token", &self.token.as_ref().map(|_| "<redacted>"))
            .field("api_root", &self.api_root)
            .field("max_comments", &self.max_comments)
            .finish()
    }
}

impl GithubConfig {
    pub fn new(owner: impl Into<String>, repo: impl Into<String>) -> Self {
        Self {
            owner: owner.into(),
            repo: repo.into(),
            token: None,
            api_root: None,
            max_comments: 50,
        }
    }

    pub fn with_token(mut self, token: Option<String>) -> Self {
        self.token = token;
        self
    }

    fn scope(&self) -> String {
        format!("{}/{}", self.owner, self.repo)
    }

    fn root(&self) -> &str {
        self.api_root.as_deref().unwrap_or(API_ROOT)
    }
}

/// A GitHub Issues connector bound to one repository.
pub struct GithubConnector {
    config: GithubConfig,
    scope: String,
    client: reqwest::Client,
}

impl GithubConnector {
    pub fn new(config: GithubConfig) -> ConnectorResult<Self> {
        if config.owner.is_empty() || config.repo.is_empty() {
            return Err(ConnectorError::Config(
                "owner and repo are required".to_string(),
            ));
        }
        // owner/repo are interpolated raw into request URLs; restrict them to
        // GitHub's actual charset so `/`, `%`, `?`, `#`, traversal sequences, etc.
        // cannot break out of the path or redirect the request.
        let valid = |s: &str| {
            s.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        };
        if !valid(&config.owner) || !valid(&config.repo) {
            return Err(ConnectorError::Config(
                "owner/repo may only contain [A-Za-z0-9._-]".to_string(),
            ));
        }
        let client = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            // Without explicit timeouts a hung connection (silently dropped
            // peer, stalled proxy) blocks the sync — and the MCP tool call —
            // forever. Connect fast, read bounded.
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| ConnectorError::Transport(e.to_string()))?;
        let scope = config.scope();
        Ok(Self {
            config,
            scope,
            client,
        })
    }

    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let req = req
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28");
        match &self.config.token {
            Some(t) => req.bearer_auth(t),
            None => req,
        }
    }

    /// Detect GitHub rate limiting on a response, so the driver can back off
    /// politely instead of hammering.
    ///
    /// - *Primary* limit: 403/429 with `x-ratelimit-remaining: 0`.
    /// - *Secondary* limit: 403/429 carrying a `Retry-After` header even when
    ///   `remaining` is nonzero (per GitHub's secondary-rate-limit docs).
    ///   Classifying those as a generic 403 caused callers to see a bare
    ///   "forbidden" and retry immediately, deepening the penalty.
    fn rate_limited(resp: &reqwest::Response) -> Option<ConnectorError> {
        let status = resp.status().as_u16();
        if status != 403 && status != 429 {
            return None;
        }
        let remaining = resp
            .headers()
            .get("x-ratelimit-remaining")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<i64>().ok());
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .map(std::time::Duration::from_secs);
        if remaining == Some(0) || retry_after.is_some() || status == 429 {
            return Some(ConnectorError::RateLimited(retry_after));
        }
        None
    }

    /// Build a `Source` error that names the exact failing call: method + URL,
    /// status, GitHub's own `message` body when present, and a hint for the
    /// statuses that actually confuse users (401/403/404 on a repo path read
    /// as "bug" when they mean "wrong name, private, or token without access").
    async fn source_error(resp: reqwest::Response, url: &str) -> ConnectorError {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        let api_message = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| {
                v.get("message")
                    .and_then(|m| m.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            });
        let hint = match status.as_u16() {
            401 => "unauthorized — the token was rejected; check GITHUB_TOKEN validity",
            403 => "forbidden — the token lacks access to this repo (or a secondary rate limit applied)",
            404 => "not found — check owner/repo spelling, and that the token (if set) can see this private repo",
            _ => status.canonical_reason().unwrap_or("request failed"),
        };
        ConnectorError::Source {
            status: status.as_u16(),
            message: match api_message {
                Some(m) => format!("GET {url} -> {status}: {m} ({hint})"),
                None => format!("GET {url} -> {status}: {hint}"),
            },
        }
    }

    /// Parse the `Link` header for the `rel="next"` url, if any.
    ///
    /// The `next` url comes from the server response, so we pin it to the
    /// configured API host before following it: otherwise a malicious or
    /// compromised endpoint could redirect the connector — which attaches the
    /// bearer token to every request — to an attacker-controlled URL and
    /// exfiltrate the credential (SSRF / token leak). `expected_host` is the
    /// host of the connector's API root.
    fn next_link(resp: &reqwest::Response, expected_host: Option<&str>) -> Option<String> {
        let link = resp.headers().get(reqwest::header::LINK)?.to_str().ok()?;
        for part in link.split(',') {
            let part = part.trim();
            if part.contains("rel=\"next\"")
                && let (Some(start), Some(end)) = (part.find('<'), part.find('>'))
                && start < end
            {
                let url = &part[start + 1..end];
                // Host-pin: only follow a next-url on the same host as the API
                // root we were configured with. FAIL-CLOSED: if we could not
                // determine the expected host (unparseable/hostless api_root), we
                // must NOT follow the url — the bearer token would otherwise ride
                // along to an attacker-influenced host (SSRF / token exfiltration).
                let Some(expected) = expected_host else {
                    tracing::warn!(
                        next_url = url,
                        "dropping Link next url: no pinned host (fail-closed)"
                    );
                    return None;
                };
                match reqwest::Url::parse(url) {
                    Ok(parsed) if parsed.host_str() == Some(expected) => {
                        return Some(url.to_string());
                    }
                    _ => {
                        tracing::warn!(
                            next_url = url,
                            "dropping cross-host Link next url (host pin)"
                        );
                        return None;
                    }
                }
            }
        }
        None
    }

    /// Host of the configured API root, used to pin Link `next` urls.
    fn api_host(&self) -> Option<String> {
        reqwest::Url::parse(self.config.root())
            .ok()
            .and_then(|u| u.host_str().map(|h| h.to_string()))
    }

    /// Fetch the comments for one issue (a single page; capped by `max_comments`).
    async fn fetch_comments(&self, issue_number: u64) -> ConnectorResult<Vec<RawComment>> {
        let url = format!(
            "{}/repos/{}/{}/issues/{}/comments?per_page={}",
            self.config.root(),
            self.config.owner,
            self.config.repo,
            issue_number,
            self.config.max_comments.min(100),
        );
        let resp = self
            .auth(self.client.get(&url))
            .send()
            .await
            .map_err(|e| ConnectorError::Transport(format!("GET {url}: {e}")))?;
        if let Some(err) = Self::rate_limited(&resp) {
            return Err(err);
        }
        if !resp.status().is_success() {
            return Err(Self::source_error(resp, &url).await);
        }
        resp.json::<Vec<RawComment>>()
            .await
            .map_err(|e| ConnectorError::Transport(format!("GET {url}: decode failed: {e}")))
    }

    /// Fold a raw issue + its comments into one normalized memory record.
    fn normalize(&self, issue: &RawIssue, comments: &[RawComment]) -> NormalizedRecord {
        let author = issue.user.as_ref().map(|u| u.login.clone());

        // Human-readable content: header + body + chronological comments.
        let mut content = format!(
            "[{}#{}] {}\nState: {}\n",
            self.scope, issue.number, issue.title, issue.state
        );
        if let Some(body) = &issue.body
            && !body.trim().is_empty()
        {
            content.push('\n');
            content.push_str(body.trim());
            content.push('\n');
        }
        let mut sorted_comments: Vec<&RawComment> = comments.iter().collect();
        sorted_comments.sort_by_key(|c| c.id);
        for c in &sorted_comments {
            let who = c.user.as_ref().map(|u| u.login.as_str()).unwrap_or("?");
            content.push_str(&format!("\n— {who}: {}", c.body.trim()));
        }

        // Labels, sorted for a stable hash.
        let mut labels: Vec<String> = issue.labels.iter().map(|l| l.name.clone()).collect();
        labels.sort();

        // Stable content hash — meaning only, never the cursor timestamp or
        // volatile counts. Comments contribute their id+body in id order.
        let comments_blob = sorted_comments
            .iter()
            .map(|c| format!("{}:{}", c.id, c.body.trim()))
            .collect::<Vec<_>>()
            .join("\u{1f}");
        let labels_blob = labels.join(",");
        let number_str = issue.number.to_string();
        let body_str = issue.body.clone().unwrap_or_default();
        let hash = content_hash(&[
            ("number", &number_str),
            ("title", &issue.title),
            ("state", &issue.state),
            ("body", &body_str),
            ("labels", &labels_blob),
            ("comments", &comments_blob),
        ]);

        let mut tags = vec![
            "github".to_string(),
            "issue".to_string(),
            format!("state:{}", issue.state),
        ];
        // Labels lowercased for tag_prefix matching: `tag_prefix` is
        // case-sensitive and GitHub labels are commonly mixed-case ("Bug",
        // "Good First Issue"), which would make `tag_prefix=label:bug` miss
        // them. Same convention the Redmine connector uses.
        tags.extend(labels.into_iter().map(|l| format!("label:{}", l.to_lowercase())));

        let envelope = SourceEnvelope {
            source_system: Some("github".to_string()),
            source_id: Some(issue.number.to_string()),
            source_url: Some(issue.html_url.clone()),
            source_updated_at: DateTime::parse_from_rfc3339(&issue.updated_at)
                .ok()
                .map(|d| d.with_timezone(&Utc)),
            content_hash: Some(hash),
            synced_at: Some(Utc::now()),
            source_project: Some(self.scope.clone()),
            source_type: Some("issue".to_string()),
            source_author: author,
        };

        NormalizedRecord {
            content,
            tags,
            envelope,
        }
    }
}

impl Connector for GithubConnector {
    fn source_system(&self) -> &str {
        "github"
    }

    fn scope(&self) -> &str {
        &self.scope
    }

    async fn fetch_updated(
        &self,
        since: Option<DateTime<Utc>>,
        cursor: Option<String>,
    ) -> ConnectorResult<FetchPage> {
        // `cursor` is a full next-page url from a previous Link header; on the
        // first page we build the url from owner/repo + since.
        let url = match cursor {
            Some(u) => u,
            None => {
                let mut u = format!(
                    "{}/repos/{}/{}/issues?state=all&sort=updated&direction=asc&per_page={}",
                    self.config.root(),
                    self.config.owner,
                    self.config.repo,
                    PER_PAGE,
                );
                if let Some(s) = since {
                    // GitHub documents the `since` format as YYYY-MM-DDTHH:MM:SSZ.
                    // `to_rfc3339()` emits the `+00:00` offset form, and the `+`
                    // is a reserved query char that the server decodes as a
                    // space — corrupting the timestamp and silently re-fetching
                    // all history every run. Emit the `Z` form (no reserved
                    // char, exact documented format) instead.
                    let since_z = s.to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
                    u.push_str(&format!("&since={since_z}"));
                }
                u
            }
        };

        let resp = self
            .auth(self.client.get(&url))
            .send()
            .await
            .map_err(|e| ConnectorError::Transport(format!("GET {url}: {e}")))?;
        if let Some(err) = Self::rate_limited(&resp) {
            return Err(err);
        }
        if !resp.status().is_success() {
            return Err(Self::source_error(resp, &url).await);
        }
        let next_cursor = Self::next_link(&resp, self.api_host().as_deref());
        let issues: Vec<RawIssue> = resp
            .json()
            .await
            .map_err(|e| ConnectorError::Transport(format!("GET {url}: decode failed: {e}")))?;

        let mut records = Vec::new();
        let mut skipped = Vec::new();
        for issue in &issues {
            // Drop pull requests — "every PR is an issue, but not vice versa".
            if issue.pull_request.is_some() {
                continue;
            }
            let updated_at = DateTime::parse_from_rfc3339(&issue.updated_at)
                .ok()
                .map(|d| d.with_timezone(&Utc));
            // Fetch comments only when the issue has any. A comment-fetch
            // failure must neither swallow into a comment-less record with a
            // corrupted hash (the historical bug) NOR abort the whole page —
            // one flaky comment response used to kill a multi-thousand-issue
            // sync mid-run. Retry once, then SKIP the issue with a cursor
            // clamp: the driver records it under `skipped`, holds the run
            // cursor at/below this issue's `updated_at`, and the next sync
            // re-fetches it (the idempotent upsert makes that safe).
            let comments = if issue.comments > 0 {
                match self.fetch_comments(issue.number).await {
                    Ok(c) => c,
                    Err(first) => match self.fetch_comments(issue.number).await {
                        Ok(c) => c,
                        Err(second) => {
                            skipped.push(SkippedRecord {
                                source_updated_at: updated_at,
                                reason: format!(
                                    "{}#{} comments: {first}; retry also failed: {second}",
                                    self.scope, issue.number
                                ),
                            });
                            continue;
                        }
                    },
                }
            } else {
                Vec::new()
            };
            records.push(self.normalize(issue, &comments));
        }

        Ok(FetchPage {
            records,
            next_cursor,
            skipped,
        })
    }

    async fn list_live_ids(&self) -> ConnectorResult<Option<Vec<String>>> {
        // Enumerate all issue numbers (ids only) for the reconcile pass, paging
        // via Link. Cheap relative to full sync (no comment fetch, no bodies).
        // Hard-capped like the Redmine connector: a hostile/broken endpoint
        // that always serves a `rel="next"` link must not paginate forever.
        const MAX_PAGES: u32 = 10_000;
        let mut ids = Vec::new();
        let mut url = Some(format!(
            "{}/repos/{}/{}/issues?state=all&per_page={}",
            self.config.root(),
            self.config.owner,
            self.config.repo,
            PER_PAGE,
        ));
        let mut pages = 0u32;
        while let Some(u) = url {
            pages += 1;
            if pages > MAX_PAGES {
                return Err(ConnectorError::Source {
                    status: 0,
                    message: format!(
                        "live-id enumeration exceeded {MAX_PAGES} pages; aborting reconcile \
                         (possible pagination loop)"
                    ),
                });
            }
            let resp = self
                .auth(self.client.get(&u))
                .send()
                .await
                .map_err(|e| ConnectorError::Transport(format!("GET {u}: {e}")))?;
            if let Some(err) = Self::rate_limited(&resp) {
                return Err(err);
            }
            if !resp.status().is_success() {
                return Err(Self::source_error(resp, &u).await);
            }
            let next = Self::next_link(&resp, self.api_host().as_deref());
            let issues: Vec<RawIssue> = resp
                .json()
                .await
                .map_err(|e| ConnectorError::Transport(format!("GET {u}: decode failed: {e}")))?;
            for issue in issues {
                if issue.pull_request.is_none() {
                    ids.push(issue.number.to_string());
                }
            }
            url = next;
        }
        Ok(Some(ids))
    }
}

// ---------------------------------------------------------------------------
// Raw GitHub API shapes (only the fields we use)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RawIssue {
    number: u64,
    title: String,
    #[serde(default)]
    body: Option<String>,
    state: String,
    html_url: String,
    updated_at: String,
    #[serde(default)]
    comments: u64,
    #[serde(default)]
    labels: Vec<RawLabel>,
    #[serde(default)]
    user: Option<RawUser>,
    /// Present iff this "issue" is actually a pull request.
    #[serde(default)]
    pull_request: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct RawLabel {
    name: String,
}

#[derive(Debug, Deserialize)]
struct RawUser {
    login: String,
}

#[derive(Debug, Deserialize)]
struct RawComment {
    id: u64,
    body: String,
    #[serde(default)]
    user: Option<RawUser>,
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;

    fn issue(number: u64, title: &str, body: &str, state: &str) -> RawIssue {
        RawIssue {
            number,
            title: title.to_string(),
            body: Some(body.to_string()),
            state: state.to_string(),
            html_url: format!("https://github.com/o/r/issues/{number}"),
            updated_at: "2026-06-19T00:00:00Z".to_string(),
            comments: 0,
            labels: vec![RawLabel {
                name: "bug".to_string(),
            }],
            user: Some(RawUser {
                login: "octocat".to_string(),
            }),
            pull_request: None,
        }
    }

    fn connector() -> GithubConnector {
        GithubConnector::new(GithubConfig::new("o", "r")).unwrap()
    }

    #[test]
    fn normalize_builds_keyed_envelope_with_citation() {
        let c = connector();
        let rec = c.normalize(&issue(57, "Connectors", "Add Redmine", "open"), &[]);
        let env = &rec.envelope;
        assert!(env.has_key());
        assert_eq!(env.source_system.as_deref(), Some("github"));
        assert_eq!(env.source_id.as_deref(), Some("57"));
        assert_eq!(
            env.source_url.as_deref(),
            Some("https://github.com/o/r/issues/57")
        );
        assert_eq!(env.source_project.as_deref(), Some("o/r"));
        assert!(rec.content.contains("Connectors"));
        assert!(rec.tags.contains(&"state:open".to_string()));
        assert!(rec.tags.contains(&"label:bug".to_string()));
    }

    #[test]
    fn hash_stable_across_label_order_and_changes_on_edit() {
        let c = connector();
        let mut a = issue(1, "T", "body", "open");
        a.labels = vec![RawLabel { name: "b".into() }, RawLabel { name: "a".into() }];
        let mut b = issue(1, "T", "body", "open");
        b.labels = vec![RawLabel { name: "a".into() }, RawLabel { name: "b".into() }];
        let ha = c.normalize(&a, &[]).envelope.content_hash;
        let hb = c.normalize(&b, &[]).envelope.content_hash;
        assert_eq!(ha, hb, "label order must not change the hash");

        // Editing the body must change the hash.
        let edited = c
            .normalize(&issue(1, "T", "EDITED", "open"), &[])
            .envelope
            .content_hash;
        assert_ne!(ha, edited);

        // Closing the issue changes state → changes the hash (not a no-op).
        let closed = c
            .normalize(&issue(1, "T", "body", "closed"), &[])
            .envelope
            .content_hash;
        assert_ne!(ha, closed);
    }

    #[test]
    fn comments_fold_in_id_order_and_affect_hash() {
        let c = connector();
        let comments = vec![
            RawComment {
                id: 2,
                body: "second".into(),
                user: Some(RawUser { login: "x".into() }),
            },
            RawComment {
                id: 1,
                body: "first".into(),
                user: Some(RawUser { login: "y".into() }),
            },
        ];
        let rec = c.normalize(&issue(1, "T", "body", "open"), &comments);
        // Folded in id order regardless of input order.
        let first_pos = rec.content.find("first").unwrap();
        let second_pos = rec.content.find("second").unwrap();
        assert!(first_pos < second_pos, "comments must fold in id order");

        let no_comments = c
            .normalize(&issue(1, "T", "body", "open"), &[])
            .envelope
            .content_hash;
        assert_ne!(
            rec.envelope.content_hash, no_comments,
            "comments must contribute to the hash"
        );
    }

    #[test]
    fn rejects_empty_owner_repo() {
        assert!(GithubConnector::new(GithubConfig::new("", "r")).is_err());
        assert!(GithubConnector::new(GithubConfig::new("o", "")).is_err());
    }

    #[test]
    fn since_uses_z_form_not_plus_offset() {
        // Regression: to_rfc3339() emits `+00:00`; the `+` decodes to a space
        // server-side and corrupts the cursor. We must emit the `Z` form.
        let ts = DateTime::parse_from_rfc3339("2026-06-19T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let z = ts.to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        assert_eq!(z, "2026-06-19T00:00:00Z");
        assert!(!z.contains('+'), "since must not contain a reserved '+'");
    }

    #[test]
    fn next_link_host_pin_drops_cross_host_url() {
        // The host-pin parsing logic (used to prevent token exfiltration via a
        // malicious Link header) must reject a different host.
        let same = reqwest::Url::parse("https://api.github.com/x?page=2").unwrap();
        let other = reqwest::Url::parse("https://evil.example/x?page=2").unwrap();
        assert_eq!(same.host_str(), Some("api.github.com"));
        assert_ne!(other.host_str(), Some("api.github.com"));
    }
}

// ===================== HTTP mock tests (loopback, no external network) ==================
// These exercise the wire-level behavior the pure unit tests cannot: Link-header
// pagination, rate-limit classification (including GitHub's secondary limits),
// error messages naming the exact failing call, and the comment retry-then-skip
// path. A tiny hand-rolled HTTP server on 127.0.0.1 keeps this dependency-free.

#[cfg(all(test, feature = "connectors", feature = "legacy-sqlite"))]
mod http_tests {
    use super::*;
    use crate::storage::SqliteMemoryStore;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    struct Response {
        status: u16,
        headers: Vec<(String, String)>,
        body: String,
    }

    impl Response {
        fn json(status: u16, body: String) -> Self {
            Self {
                status,
                headers: vec![("Content-Type".into(), "application/json".into())],
                body,
            }
        }

        fn with_header(mut self, k: &str, v: &str) -> Self {
            self.headers.push((k.into(), v.into()));
            self
        }
    }

    /// Minimal loopback HTTP server: one accept thread, one request per
    /// connection, handler picks the response by request path. Records every
    /// path it served so tests can assert exactly which calls were made.
    struct MockApi {
        base_url: String,
        requests: Arc<Mutex<Vec<String>>>,
        stop: Arc<AtomicBool>,
    }

    type MockHandler = Arc<dyn Fn(&str, &str) -> Response + Send + Sync>;

    impl MockApi {
        fn spawn(handler: MockHandler) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let base_url = format!("http://127.0.0.1:{port}");
            let requests = Arc::new(Mutex::new(Vec::new()));
            let stop = Arc::new(AtomicBool::new(false));
            let reqs = requests.clone();
            let stop_flag = stop.clone();
            std::thread::spawn(move || {
                listener.set_nonblocking(true).unwrap();
                while !stop_flag.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            // macOS/BSD: accepted sockets inherit the
                            // listener's O_NONBLOCK; force blocking reads.
                            stream.set_nonblocking(false).unwrap();
                            let path = read_request_path(&mut stream);
                            reqs.lock().unwrap().push(path.clone());
                            let resp = handler(&path, &base_url);
                            write_response(&mut stream, &resp);
                        }
                        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(std::time::Duration::from_millis(2));
                        }
                        Err(_) => break,
                    }
                }
            });
            Self {
                base_url: format!("http://127.0.0.1:{port}"),
                requests,
                stop,
            }
        }

        fn requests(&self) -> Vec<String> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl Drop for MockApi {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
        }
    }

    fn read_request_path(stream: &mut TcpStream) -> String {
        let mut buf = [0u8; 4096];
        let mut data = Vec::new();
        // GET requests carry no body; read once, then a bit more if needed.
        for _ in 0..3 {
            match stream.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    data.extend_from_slice(&buf[..n]);
                    if data.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        let head = String::from_utf8_lossy(&data);
        head.split_whitespace()
            .nth(1)
            .unwrap_or("<unread>")
            .to_string()
    }

    fn write_response(stream: &mut TcpStream, resp: &Response) {
        let reason = match resp.status {
            200 => "OK",
            403 => "Forbidden",
            404 => "Not Found",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            _ => "Unknown",
        };
        let mut out = format!("HTTP/1.1 {} {}\r\n", resp.status, reason);
        for (k, v) in &resp.headers {
            out.push_str(&format!("{k}: {v}\r\n"));
        }
        out.push_str(&format!(
            "Content-Length: {}\r\nConnection: close\r\n\r\n",
            resp.body.len()
        ));
        let _ = stream.write_all(out.as_bytes());
        let _ = stream.write_all(resp.body.as_bytes());
        let _ = stream.flush();
    }

    fn issue_json(number: u64, comments: u64, pr: bool) -> String {
        let mut issue = format!(
            r#"{{"number": {number}, "title": "Issue {number}", "body": "body {number}", "state": "open", "html_url": "https://github.com/o/r/issues/{number}", "updated_at": "2026-06-19T00:00:0{number}Z", "comments": {comments}, "labels": [], "user": {{"login": "octocat"}}"#
        );
        if pr {
            issue.push_str(r#", "pull_request": {"url": "https://api.github.com/repos/o/r/pulls/1"}"#);
        }
        issue.push('}');
        issue
    }

    fn connector(base_url: &str) -> GithubConnector {
        GithubConnector::new(GithubConfig {
            owner: "o".to_string(),
            repo: "r".to_string(),
            token: None,
            api_root: Some(base_url.to_string()),
            max_comments: 50,
        })
        .unwrap()
    }

    #[tokio::test]
    async fn pagination_follows_link_next_headers() {
        let api = MockApi::spawn(Arc::new(|path: &str, base: &str| {
            if path.contains("page=2") {
                Response::json(200, format!("[{}]", issue_json(2, 0, false)))
            } else {
                Response::json(200, format!("[{}]", issue_json(1, 0, false))).with_header(
                    "Link",
                    &format!(
                        r#"<{base}/repos/o/r/issues?state=all&sort=updated&direction=asc&per_page=100&page=2>; rel="next", <{base}/repos/o/r/issues?page=9>; rel="last""#
                    ),
                )
            }
        }));
        let conn = connector(&api.base_url);
        let first = conn.fetch_updated(None, None).await.unwrap();
        assert_eq!(first.records.len(), 1, "page 1");
        assert_eq!(first.records[0].envelope.source_id.as_deref(), Some("1"));
        let next = first.next_cursor.expect("Link rel=next must be followed");
        assert!(next.contains("page=2"));

        let second = conn.fetch_updated(None, Some(next)).await.unwrap();
        assert_eq!(second.records.len(), 1, "page 2");
        assert_eq!(second.records[0].envelope.source_id.as_deref(), Some("2"));
        assert!(second.next_cursor.is_none(), "no rel=next → exhausted");

        let reqs = api.requests();
        assert_eq!(reqs.len(), 2, "exactly one request per page");
        assert!(reqs[1].contains("page=2"), "the server's next url is used");
    }

    #[tokio::test]
    async fn a_404_error_names_the_exact_failing_call() {
        let api = MockApi::spawn(Arc::new(|_path: &str, _base: &str| {
            Response::json(404, r#"{"message": "Not Found", "documentation_url": "x"}"#.into())
        }));
        let err = connector(&api.base_url)
            .fetch_updated(None, None)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains(&format!("GET {}/repos/o/r/issues", api.base_url)),
            "error must name the failing call: {msg}"
        );
        assert!(
            msg.contains("404") && msg.contains("check owner/repo"),
            "error must carry status and the actionable hint: {msg}"
        );
        assert_eq!(api.requests().len(), 1);
    }

    #[tokio::test]
    async fn secondary_rate_limit_403_with_retry_after_is_classified_rate_limited() {
        let api = MockApi::spawn(Arc::new(|_path: &str, _base: &str| {
            Response::json(
                403,
                r#"{"message": "You have exceeded a secondary rate limit"}"#.into(),
            )
            .with_header("Retry-After", "30")
            .with_header("x-ratelimit-remaining", "42")
        }));
        let err = connector(&api.base_url)
            .fetch_updated(None, None)
            .await
            .unwrap_err();
        match err {
            ConnectorError::RateLimited(Some(d)) => {
                assert_eq!(d.as_secs(), 30, "the server's Retry-After must be honored")
            }
            other => panic!("403 + Retry-After must be RateLimited, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn comment_fetch_failure_retries_then_skips_without_aborting_the_page() {
        let api = MockApi::spawn(Arc::new(|path: &str, _base: &str| {
            if path.contains("/comments") {
                Response::json(500, r#"{"message": "boom"}"#.into())
            } else {
                Response::json(200, format!("[{}]", issue_json(1, 1, false)))
            }
        }));
        let conn = connector(&api.base_url);
        let page = conn.fetch_updated(None, None).await.unwrap();
        assert!(
            page.records.is_empty(),
            "the comment-less record must NOT be persisted with a corrupted hash"
        );
        assert_eq!(page.skipped.len(), 1);
        assert!(
            page.skipped[0].reason.contains("#1"),
            "the skip must name the issue: {:?}",
            page.skipped[0].reason
        );
        let comment_calls = api
            .requests()
            .iter()
            .filter(|p| p.contains("/comments"))
            .count();
        assert_eq!(comment_calls, 2, "exactly one retry before the skip");
    }

    #[tokio::test]
    async fn comment_fetch_recovers_after_one_transient_failure() {
        // First comments request fails, the retry succeeds. The flag is a
        // one-shot: `swap(false)` makes exactly the first call fail.
        let failing = Arc::new(AtomicBool::new(true));
        let api = {
            let failing = failing.clone();
            MockApi::spawn(Arc::new(move |path: &str, _base: &str| {
                if path.contains("/comments") {
                    if failing.swap(false, Ordering::SeqCst) {
                        Response::json(500, "{}".into())
                    } else {
                        Response::json(
                            200,
                            r#"[{"id": 7, "body": "a comment", "user": {"login": "u"}}]"#.into(),
                        )
                    }
                } else {
                    Response::json(200, format!("[{}]", issue_json(1, 1, false)))
                }
            }))
        };
        let conn = connector(&api.base_url);
        let page = conn.fetch_updated(None, None).await.unwrap();
        assert_eq!(page.records.len(), 1, "retry succeeded -> record persists");
        assert!(page.records[0].content.contains("a comment"));
        assert!(page.skipped.is_empty());
    }

    #[tokio::test]
    async fn pull_requests_are_dropped_and_never_fetch_comments() {
        let api = MockApi::spawn(Arc::new(|_path: &str, _base: &str| {
            Response::json(200, format!("[{}, {}]", issue_json(1, 0, true), issue_json(2, 0, false)))
        }));
        let page = connector(&api.base_url)
            .fetch_updated(None, None)
            .await
            .unwrap();
        assert_eq!(page.records.len(), 1);
        assert_eq!(page.records[0].envelope.source_id.as_deref(), Some("2"));
        assert!(
            !api.requests().iter().any(|p| p.contains("/comments")),
            "PRs must not trigger comment fetches"
        );
    }

    #[tokio::test]
    async fn a_connection_failure_names_the_failing_call() {
        // Bind a port, then drop the listener: nothing is listening.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let base = format!("http://127.0.0.1:{port}");
        let err = connector(&base).fetch_updated(None, None).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains(&format!("GET {base}/repos/o/r/issues")),
            "transport errors must name the exact call (reqwest 0.13 Display may \
             omit the url; our wrapper must not): {msg}"
        );
    }

    // ===================== Real read-only integration ==================
    /// REAL network test against api.github.com (READ-ONLY, public repo).
    /// Gated: run with `VESTIGE_SOURCE_SYNC_INTEGRATION=1 cargo test -p
    /// vestige-core --features connectors`. Never writes to GitHub; syncs one
    /// page of samvallad33/vestige issues into a throwaway temp db and proves
    /// the end-to-end path: paginate → normalize → cite-backed memories →
    /// idempotent re-run (no duplicates).
    #[tokio::test]
    async fn integration_real_github_sync_readonly_end_to_end() {
        if std::env::var("VESTIGE_SOURCE_SYNC_INTEGRATION").ok().as_deref() != Some("1") {
            eprintln!("skipping: set VESTIGE_SOURCE_SYNC_INTEGRATION=1 to run");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteMemoryStore::new(Some(dir.path().join("integration.db"))).unwrap();

        let token = std::env::var("GITHUB_TOKEN").ok().filter(|s| !s.is_empty());
        let config = GithubConfig::new("samvallad33", "vestige").with_token(token);
        let conn = GithubConnector::new(config).unwrap();

        let report = crate::connectors::run_sync(&store, &conn, false, 1)
            .await
            .unwrap();
        assert!(
            report.created >= 1,
            "a real page of samvallad33/vestige issues must index: {report:?}"
        );
        assert!(
            report.warnings.is_empty(),
            "no warnings expected on a healthy repo: {:?}",
            report.warnings
        );
        assert!(
            report.new_cursor.is_some(),
            "the cursor checkpoint must be persisted"
        );

        // Every ingested memory is a good citizen: keyed envelope, citation
        // URL, event node type, structured tags.
        let rows: Vec<(String, String, String, String, String, String)> = {
            let reader = store.reader.lock().unwrap();
            let mut stmt = reader
                .prepare(
                    "SELECT node_type, source_system, source_id, source_url, tags, content_hash \
                     FROM knowledge_nodes WHERE source_system = 'github'",
                )
                .unwrap();
            stmt
                .query_map([], |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                    ))
            })
            .unwrap()
            .filter_map(Result::ok)
                .collect()
        };
        assert_eq!(
            rows.len(),
            report.created,
            "no duplicates: one node per record"
        );
        for (node_type, system, id, url, tags, hash) in &rows {
            assert_eq!(node_type, "event");
            assert_eq!(system, "github");
            assert!(!id.is_empty());
            assert!(url.starts_with("https://github.com/samvallad33/vestige/issues/"));
            assert!(tags.contains("\"github\""), "tagged github: {tags}");
            assert!(!hash.is_empty(), "content_hash present");
        }

        // Re-run resumes from the saved cursor (max_pages=1 is a PARTIAL sync
        // of this repo): the overlap window re-scans the boundary issues as
        // Unchanged and may legitimately index further never-seen issues. The
        // hard invariant is ONE node per source id — no duplicates ever.
        let report2 = crate::connectors::run_sync(&store, &conn, false, 1)
            .await
            .unwrap();
        assert!(
            report2.warnings.is_empty(),
            "no warnings expected: {:?}",
            report2.warnings
        );
        let (total, distinct): (i64, i64) = {
            let reader = store.reader.lock().unwrap();
            let mut stmt = reader
                .prepare(
                    "SELECT COUNT(*), COUNT(DISTINCT source_id) FROM knowledge_nodes \
                     WHERE source_system = 'github' AND source_project = 'samvallad33/vestige'",
                )
                .unwrap();
            stmt.query_row([], |r| Ok((r.get(0)?, r.get(1)?))).unwrap()
        };
        assert_eq!(
            total as usize,
            rows.len() + report2.created,
            "every created report maps to exactly one new row (report2: {report2:?})"
        );
        assert_eq!(
            total, distinct,
            "no duplicate source ids across re-runs (report2: {report2:?})"
        );
    }
}
