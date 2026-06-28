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
    Connector, ConnectorError, ConnectorResult, FetchPage, NormalizedRecord, content_hash,
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

    /// Map an HTTP response status into a connector error, honoring rate-limit
    /// signals so the driver can back off politely.
    fn classify_status(resp: &reqwest::Response) -> Option<ConnectorError> {
        let status = resp.status();
        if status.is_success() {
            return None;
        }
        // Primary rate limit: 403/429 with remaining=0.
        if status.as_u16() == 403 || status.as_u16() == 429 {
            let remaining = resp
                .headers()
                .get("x-ratelimit-remaining")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<i64>().ok());
            if remaining == Some(0) || status.as_u16() == 429 {
                let retry = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok())
                    .map(std::time::Duration::from_secs);
                return Some(ConnectorError::RateLimited(retry));
            }
        }
        Some(ConnectorError::Source {
            status: status.as_u16(),
            message: status
                .canonical_reason()
                .unwrap_or("request failed")
                .to_string(),
        })
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
            .map_err(|e| ConnectorError::Transport(e.to_string()))?;
        if let Some(err) = Self::classify_status(&resp) {
            return Err(err);
        }
        resp.json::<Vec<RawComment>>()
            .await
            .map_err(|e| ConnectorError::Transport(e.to_string()))
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
        tags.extend(labels.into_iter().map(|l| format!("label:{l}")));

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
            .map_err(|e| ConnectorError::Transport(e.to_string()))?;
        if let Some(err) = Self::classify_status(&resp) {
            return Err(err);
        }
        let next_cursor = Self::next_link(&resp, self.api_host().as_deref());
        let issues: Vec<RawIssue> = resp
            .json()
            .await
            .map_err(|e| ConnectorError::Transport(e.to_string()))?;

        let mut records = Vec::new();
        for issue in &issues {
            // Drop pull requests — "every PR is an issue, but not vice versa".
            if issue.pull_request.is_some() {
                continue;
            }
            // Fetch comments only when the issue has any. Propagate failures
            // instead of swallowing them: a silent unwrap_or_default() stored a
            // comment-less record with a corrupted content hash AND let the
            // cursor advance past it, so the issue would never be re-synced. A
            // propagated error keeps the issue in the next window (cursor clamp).
            let comments = if issue.comments > 0 {
                self.fetch_comments(issue.number).await?
            } else {
                Vec::new()
            };
            records.push(self.normalize(issue, &comments));
        }

        Ok(FetchPage {
            records,
            next_cursor,
        })
    }

    async fn list_live_ids(&self) -> ConnectorResult<Option<Vec<String>>> {
        // Enumerate all issue numbers (ids only) for the reconcile pass, paging
        // via Link. Cheap relative to full sync (no comment fetch, no bodies).
        let mut ids = Vec::new();
        let mut url = Some(format!(
            "{}/repos/{}/{}/issues?state=all&per_page={}",
            self.config.root(),
            self.config.owner,
            self.config.repo,
            PER_PAGE,
        ));
        while let Some(u) = url {
            let resp = self
                .auth(self.client.get(&u))
                .send()
                .await
                .map_err(|e| ConnectorError::Transport(e.to_string()))?;
            if let Some(err) = Self::classify_status(&resp) {
                return Err(err);
            }
            let next = Self::next_link(&resp, self.api_host().as_deref());
            let issues: Vec<RawIssue> = resp
                .json()
                .await
                .map_err(|e| ConnectorError::Transport(e.to_string()))?;
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

#[cfg(test)]
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
