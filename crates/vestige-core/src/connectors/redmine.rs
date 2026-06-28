//! Redmine connector (#57).
//!
//! Indexes a Redmine project's issues + journals (comments and status/assignment
//! history) into source-aware Vestige memories so an investigative agent can
//! search and reason over years of ticket history **offline**, **semantically**,
//! and **cited back to the canonical issue URL**. Redmine stays the system of
//! record; Vestige indexes, connects, retrieves, and links back.
//!
//! ## Incremental sync (per the connector sync contract)
//!
//! Redmine's REST API has three traps this connector handles explicitly (all
//! confirmed against the official wiki + canonical defects):
//!
//! - **`status_id=*` is mandatory.** The list endpoint returns *open issues
//!   only* by default, so without it closing an issue looks like a deletion and
//!   closed issues are never synced (Defect #19088). We pass it on both the
//!   incremental pull and the reconcile enumeration.
//! - **`include=journals` is silently ignored on the list endpoint.** Journals
//!   come back only on the per-issue detail endpoint `GET /issues/{id}.json`
//!   (Defect #35242), so each changed issue costs one extra round-trip.
//! - **Filter operators must be hex-encoded** in the compact form
//!   (`updated_on=>=…` → `updated_on=%3E%3D…`). We build the query with
//!   `reqwest`'s `.query(&[…])` and pass the raw `>=…` value so it is encoded
//!   exactly once (no double-encoding).
//!
//! `sort=updated_on:asc` pages forward in cursor order so a mid-run interruption
//! resumes safely; the `since = cursor − overlap` window + the `content_hash`
//! no-op make the re-scan free. Redmine has no deletion feed, so deletions are
//! reconciled out-of-band via [`list_live_ids`](Connector::list_live_ids).

use chrono::{DateTime, Utc};
use serde::Deserialize;

use super::{
    Connector, ConnectorError, ConnectorResult, FetchPage, NormalizedRecord, content_hash,
};
use crate::memory::SourceEnvelope;

const USER_AGENT: &str = concat!("vestige-connector/", env!("CARGO_PKG_VERSION"));
const PAGE_LIMIT: u32 = 100;

/// Configuration for a Redmine connector instance bound to one project.
#[derive(Clone)]
pub struct RedmineConfig {
    /// Base URL of the Redmine instance, e.g. `https://redmine.example.com`.
    pub base_url: String,
    /// Project identifier to scope the sync to. May be the numeric id or the
    /// project identifier slug — used as `project_id` and stored as
    /// `source_project`. (Note: Redmine's `project_id` list filter wants the
    /// numeric id; the slug works as the human-readable scope label.)
    pub project: String,
    /// API access key. Optional only if the instance allows anonymous REST.
    pub api_key: Option<String>,
    /// Max journals to fold into one issue memory (defense against huge threads).
    pub max_journals: usize,
}

// Manual Debug that NEVER prints the api_key — a derived Debug would leak the
// credential into any `{:?}` log line or panic message.
impl std::fmt::Debug for RedmineConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedmineConfig")
            .field("base_url", &self.base_url)
            .field("project", &self.project)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("max_journals", &self.max_journals)
            .finish()
    }
}

impl RedmineConfig {
    pub fn new(base_url: impl Into<String>, project: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            project: project.into(),
            api_key: None,
            max_journals: 100,
        }
    }

    pub fn with_api_key(mut self, key: Option<String>) -> Self {
        self.api_key = key;
        self
    }

    /// Base URL with any trailing slash removed.
    fn root(&self) -> String {
        self.base_url.trim_end_matches('/').to_string()
    }
}

/// A Redmine connector bound to one project.
pub struct RedmineConnector {
    config: RedmineConfig,
    scope: String,
    client: reqwest::Client,
}

impl RedmineConnector {
    pub fn new(config: RedmineConfig) -> ConnectorResult<Self> {
        if config.base_url.trim().is_empty() {
            return Err(ConnectorError::Config("base_url is required".to_string()));
        }
        if config.project.trim().is_empty() {
            return Err(ConnectorError::Config("project is required".to_string()));
        }
        // SSRF guard: require http(s) and reject internal/reserved hosts so a
        // misconfigured/hostile base_url cannot point the authenticated client at
        // localhost, link-local metadata endpoints (169.254.169.254), or private
        // ranges. Gated off only when explicitly allowed (for local tests).
        match reqwest::Url::parse(&config.root()) {
            Ok(url) => {
                let scheme = url.scheme();
                if scheme != "http" && scheme != "https" {
                    return Err(ConnectorError::Config(format!(
                        "base_url scheme must be http or https, got {scheme}"
                    )));
                }
                if std::env::var("VESTIGE_ALLOW_PRIVATE_CONNECTOR_HOSTS").is_err() {
                    match url.host() {
                        None => {
                            return Err(ConnectorError::Config(
                                "base_url has no host".to_string(),
                            ));
                        }
                        Some(url::Host::Ipv4(ip))
                            if ip.is_loopback()
                                || ip.is_private()
                                || ip.is_link_local()
                                || ip.is_unspecified() =>
                        {
                            return Err(ConnectorError::Config(format!(
                                "base_url host {ip} is a reserved/internal address (SSRF guard)"
                            )));
                        }
                        Some(url::Host::Ipv6(ip)) if ip.is_loopback() || ip.is_unspecified() => {
                            return Err(ConnectorError::Config(format!(
                                "base_url host {ip} is a reserved/internal address (SSRF guard)"
                            )));
                        }
                        Some(url::Host::Domain(d))
                            if d.eq_ignore_ascii_case("localhost") =>
                        {
                            return Err(ConnectorError::Config(
                                "base_url host localhost is blocked (SSRF guard)".to_string(),
                            ));
                        }
                        _ => {}
                    }
                }
            }
            Err(_) => {
                return Err(ConnectorError::Config(format!(
                    "base_url is not a valid URL: {}",
                    config.base_url
                )));
            }
        }
        let client = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .build()
            .map_err(|e| ConnectorError::Transport(e.to_string()))?;
        let scope = config.project.clone();
        Ok(Self {
            config,
            scope,
            client,
        })
    }

    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let req = req.header("Accept", "application/json");
        match &self.config.api_key {
            // The key goes in the header (not the URL) so it stays out of proxy
            // and access logs.
            Some(k) => req.header("X-Redmine-API-Key", k),
            None => req,
        }
    }

    fn classify_status(resp: &reqwest::Response) -> Option<ConnectorError> {
        let status = resp.status();
        if status.is_success() {
            return None;
        }
        if status.as_u16() == 429 {
            let retry = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .map(std::time::Duration::from_secs);
            return Some(ConnectorError::RateLimited(retry));
        }
        let message = match status.as_u16() {
            // A valid key against an instance with REST disabled 401/403s; make
            // that distinguishable from "no results".
            401 | 403 => {
                "unauthorized — check REDMINE_API_KEY and that the instance has the REST API enabled (Administration → Settings → API)"
                    .to_string()
            }
            _ => status
                .canonical_reason()
                .unwrap_or("request failed")
                .to_string(),
        };
        Some(ConnectorError::Source {
            status: status.as_u16(),
            message,
        })
    }

    /// Fetch the journals + relations for one issue (the detail endpoint —
    /// journals are not returned on the list endpoint).
    async fn fetch_detail(&self, issue_id: u64) -> ConnectorResult<RawIssue> {
        let url = format!("{}/issues/{}.json", self.config.root(), issue_id);
        let resp = self
            .auth(self.client.get(&url))
            .query(&[("include", "journals,relations")])
            .send()
            .await
            .map_err(|e| ConnectorError::Transport(e.to_string()))?;
        if let Some(err) = Self::classify_status(&resp) {
            return Err(err);
        }
        let wrapper: IssueWrapper = resp
            .json()
            .await
            .map_err(|e| ConnectorError::Transport(e.to_string()))?;
        Ok(wrapper.issue)
    }

    /// Fold a raw issue (with journals) into one normalized memory record.
    fn normalize(&self, issue: &RawIssue) -> NormalizedRecord {
        let status_name = issue.status.as_ref().map(|s| s.name.clone());
        let tracker_name = issue.tracker.as_ref().map(|t| t.name.clone());
        let author = issue.author.as_ref().map(|a| a.name.clone());

        // Journals sorted by id for a stable order + stable hash. Keep notes
        // and field changes so status/assignment history remains searchable.
        let mut journals: Vec<&RawJournal> = issue
            .journals
            .iter()
            .filter(|j| {
                j.notes
                    .as_deref()
                    .map(|n| !n.trim().is_empty())
                    .unwrap_or(false)
                    || !j.details.is_empty()
            })
            .collect();
        journals.sort_by_key(|j| j.id);
        journals.truncate(self.config.max_journals);

        // Human-readable content.
        let mut content = format!("[{}#{}] {}\n", self.scope, issue.id, issue.subject);
        if let Some(s) = &status_name {
            content.push_str(&format!("Status: {s}\n"));
        }
        if let Some(t) = &tracker_name {
            content.push_str(&format!("Tracker: {t}\n"));
        }
        if let Some(desc) = &issue.description
            && !desc.trim().is_empty()
        {
            content.push('\n');
            content.push_str(desc.trim());
            content.push('\n');
        }
        for j in &journals {
            let who = j.user.as_ref().map(|u| u.name.as_str()).unwrap_or("?");
            let note = j.notes.as_deref().unwrap_or("").trim();
            if !note.is_empty() {
                content.push_str(&format!("\n- {who}: {note}"));
            }
            for detail in &j.details {
                content.push_str(&format!(
                    "\n- {who} changed {}{}: {} -> {}",
                    detail.property.as_deref().unwrap_or("field"),
                    detail
                        .name
                        .as_deref()
                        .map(|n| format!(".{n}"))
                        .unwrap_or_default(),
                    detail.old_value.as_deref().unwrap_or(""),
                    detail.new_value.as_deref().unwrap_or("")
                ));
            }
        }
        if !issue.relations.is_empty() {
            content.push_str("\n\nRelations:");
            let mut relations: Vec<&RawRelation> = issue.relations.iter().collect();
            relations.sort_by_key(|r| r.id);
            for relation in relations {
                let related = relation.related_issue_id(issue.id);
                content.push_str(&format!(
                    "\n- #{} ({})",
                    related,
                    relation.relation_type.as_deref().unwrap_or("relates")
                ));
                if let Some(delay) = relation.delay {
                    content.push_str(&format!(", delay {delay}"));
                }
            }
        }

        // Stable content hash — meaning only, never the cursor (`updated_on`) or
        // volatile counts. Journals and relations contribute stable fields in id
        // order.
        let journals_blob = journals
            .iter()
            .map(|j| {
                let details = j
                    .details
                    .iter()
                    .map(|d| {
                        format!(
                            "{}:{}:{}:{}",
                            d.property.as_deref().unwrap_or(""),
                            d.name.as_deref().unwrap_or(""),
                            d.old_value.as_deref().unwrap_or(""),
                            d.new_value.as_deref().unwrap_or("")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\u{1e}");
                format!(
                    "{}:{}:{}",
                    j.id,
                    j.notes.as_deref().unwrap_or("").trim(),
                    details
                )
            })
            .collect::<Vec<_>>()
            .join("\u{1f}");
        let relations_blob = {
            let mut relations: Vec<&RawRelation> = issue.relations.iter().collect();
            relations.sort_by_key(|r| r.id);
            relations
                .iter()
                .map(|r| {
                    format!(
                        "{}:{}:{}:{}",
                        r.id,
                        r.issue_id.unwrap_or_default(),
                        r.issue_to_id.unwrap_or_default(),
                        r.relation_type.as_deref().unwrap_or("")
                    )
                })
                .collect::<Vec<_>>()
                .join("\u{1f}")
        };
        let id_str = issue.id.to_string();
        let status_id_str = issue
            .status
            .as_ref()
            .map(|s| s.id.to_string())
            .unwrap_or_default();
        let tracker_id_str = issue
            .tracker
            .as_ref()
            .map(|t| t.id.to_string())
            .unwrap_or_default();
        let done_ratio_str = issue.done_ratio.unwrap_or(0).to_string();
        let desc_str = issue.description.clone().unwrap_or_default();
        let hash = content_hash(&[
            ("id", &id_str),
            ("subject", &issue.subject),
            ("description", &desc_str),
            ("status_id", &status_id_str),
            ("tracker_id", &tracker_id_str),
            ("done_ratio", &done_ratio_str),
            ("journals", &journals_blob),
            ("relations", &relations_blob),
        ]);

        // Tags, lowercased — `tag_prefix` matching is case-sensitive, and
        // Redmine status/tracker names are mixed-case.
        let mut tags = vec!["redmine".to_string(), "issue".to_string()];
        if let Some(s) = &status_name {
            tags.push(format!("status:{}", s.to_lowercase()));
        }
        if let Some(t) = &tracker_name {
            tags.push(format!("tracker:{}", t.to_lowercase()));
        }
        if let Some(p) = &issue.priority {
            tags.push(format!("priority:{}", p.name.to_lowercase()));
        }

        let envelope = SourceEnvelope {
            source_system: Some("redmine".to_string()),
            source_id: Some(issue.id.to_string()),
            source_url: Some(format!("{}/issues/{}", self.config.root(), issue.id)),
            source_updated_at: issue
                .updated_on
                .as_deref()
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
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

impl Connector for RedmineConnector {
    fn source_system(&self) -> &str {
        "redmine"
    }

    fn scope(&self) -> &str {
        &self.scope
    }

    async fn fetch_updated(
        &self,
        since: Option<DateTime<Utc>>,
        cursor: Option<String>,
    ) -> ConnectorResult<FetchPage> {
        // The cursor carries the next offset (Redmine pages by offset, not an
        // opaque url). First page = offset 0.
        let offset: u32 = cursor.as_deref().and_then(|c| c.parse().ok()).unwrap_or(0);

        let url = format!("{}/issues.json", self.config.root());
        let limit_str = PAGE_LIMIT.to_string();
        let offset_str = offset.to_string();
        // Build params; reqwest percent-encodes each value exactly once, so we
        // pass the RAW `>=…` operator (it becomes %3E%3D on the wire). Do not
        // pre-encode here or it would be double-encoded.
        let mut params: Vec<(&str, String)> = vec![
            ("status_id", "*".to_string()),
            ("sort", "updated_on:asc".to_string()),
            ("project_id", self.config.project.clone()),
            ("limit", limit_str),
            ("offset", offset_str),
        ];
        if let Some(s) = since {
            let since_z = s.to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
            params.push(("updated_on", format!(">={since_z}")));
        }

        let resp = self
            .auth(self.client.get(&url))
            .query(&params)
            .send()
            .await
            .map_err(|e| ConnectorError::Transport(e.to_string()))?;
        if let Some(err) = Self::classify_status(&resp) {
            return Err(err);
        }
        let page: IssueListResponse = resp
            .json()
            .await
            .map_err(|e| ConnectorError::Transport(e.to_string()))?;

        // Per-issue detail fetch for journals (list endpoint omits them).
        let mut records = Vec::new();
        for summary in &page.issues {
            let detailed = match self.fetch_detail(summary.id).await {
                Ok(d) => d,
                // A single issue failing detail-fetch should not abort the page;
                // fall back to the list-level fields (no journals).
                Err(_) => summary.clone(),
            };
            records.push(self.normalize(&detailed));
        }

        // Advance the offset cursor until we've walked total_count.
        let next_offset = offset + page.issues.len() as u32;
        let next_cursor = if (next_offset as u64) < page.total_count && !page.issues.is_empty() {
            Some(next_offset.to_string())
        } else {
            None
        };

        Ok(FetchPage {
            records,
            next_cursor,
        })
    }

    async fn list_live_ids(&self) -> ConnectorResult<Option<Vec<String>>> {
        // Enumerate all issue ids (open AND closed) for the reconcile pass.
        // status_id=* is mandatory here too, or closed issues read as deleted.
        let mut ids = Vec::new();
        // u64 offset (a u32 could wrap on a huge/compromised total_count, turning
        // the loop infinite + allocating unboundedly). Also hard-cap pages.
        let mut offset: u64 = 0;
        const MAX_PAGES: u32 = 10_000;
        let mut pages = 0u32;
        loop {
            let url = format!("{}/issues.json", self.config.root());
            let resp = self
                .auth(self.client.get(&url))
                .query(&[
                    ("status_id", "*".to_string()),
                    ("project_id", self.config.project.clone()),
                    ("limit", PAGE_LIMIT.to_string()),
                    ("offset", offset.to_string()),
                ])
                .send()
                .await
                .map_err(|e| ConnectorError::Transport(e.to_string()))?;
            if let Some(err) = Self::classify_status(&resp) {
                return Err(err);
            }
            let page: IssueListResponse = resp
                .json()
                .await
                .map_err(|e| ConnectorError::Transport(e.to_string()))?;
            if page.issues.is_empty() {
                break;
            }
            for issue in &page.issues {
                ids.push(issue.id.to_string());
            }
            let new_offset = offset + page.issues.len() as u64;
            // Defensive: a non-advancing page would loop forever.
            if new_offset <= offset {
                break;
            }
            offset = new_offset;
            pages += 1;
            if offset >= page.total_count || pages >= MAX_PAGES {
                break;
            }
        }
        Ok(Some(ids))
    }
}

// ---------------------------------------------------------------------------
// Raw Redmine API shapes (only the fields we use)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct IssueListResponse {
    #[serde(default)]
    issues: Vec<RawIssue>,
    #[serde(default)]
    total_count: u64,
}

#[derive(Debug, Deserialize)]
struct IssueWrapper {
    issue: RawIssue,
}

#[derive(Debug, Clone, Deserialize)]
struct RawIssue {
    id: u64,
    #[serde(default)]
    subject: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    status: Option<NamedRef>,
    #[serde(default)]
    tracker: Option<NamedRef>,
    #[serde(default)]
    priority: Option<NamedRef>,
    #[serde(default)]
    author: Option<NamedRef>,
    #[serde(default)]
    done_ratio: Option<i64>,
    #[serde(default)]
    updated_on: Option<String>,
    #[serde(default)]
    journals: Vec<RawJournal>,
    #[serde(default)]
    relations: Vec<RawRelation>,
}

/// Redmine `{id, name}` reference (status, tracker, priority, user, …).
#[derive(Debug, Clone, Deserialize)]
struct NamedRef {
    #[serde(default)]
    id: i64,
    #[serde(default)]
    name: String,
}

#[derive(Debug, Clone, Deserialize)]
struct RawJournal {
    id: u64,
    #[serde(default)]
    notes: Option<String>,
    #[serde(default)]
    user: Option<NamedRef>,
    #[serde(default)]
    details: Vec<RawJournalDetail>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawJournalDetail {
    #[serde(default)]
    property: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    old_value: Option<String>,
    #[serde(default)]
    new_value: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawRelation {
    #[serde(default)]
    id: u64,
    #[serde(default)]
    issue_id: Option<u64>,
    #[serde(default)]
    issue_to_id: Option<u64>,
    #[serde(default)]
    relation_type: Option<String>,
    #[serde(default)]
    delay: Option<i64>,
}

impl RawRelation {
    fn related_issue_id(&self, current_issue_id: u64) -> u64 {
        match (self.issue_id, self.issue_to_id) {
            (Some(from), Some(to)) if from == current_issue_id => to,
            (Some(from), Some(to)) if to == current_issue_id => from,
            (_, Some(to)) => to,
            (Some(from), _) => from,
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn issue(id: u64, subject: &str, desc: &str, status: (i64, &str)) -> RawIssue {
        RawIssue {
            id,
            subject: subject.to_string(),
            description: Some(desc.to_string()),
            status: Some(NamedRef {
                id: status.0,
                name: status.1.to_string(),
            }),
            tracker: Some(NamedRef {
                id: 1,
                name: "Bug".to_string(),
            }),
            priority: Some(NamedRef {
                id: 2,
                name: "Normal".to_string(),
            }),
            author: Some(NamedRef {
                id: 7,
                name: "Jane Dev".to_string(),
            }),
            done_ratio: Some(0),
            updated_on: Some("2026-06-19T00:00:00Z".to_string()),
            journals: vec![],
            relations: vec![],
        }
    }

    fn connector() -> RedmineConnector {
        RedmineConnector::new(RedmineConfig::new("https://redmine.example.com", "infra")).unwrap()
    }

    #[test]
    fn rejects_empty_and_bad_config() {
        assert!(RedmineConnector::new(RedmineConfig::new("", "p")).is_err());
        assert!(RedmineConnector::new(RedmineConfig::new("https://r.example", "")).is_err());
        assert!(RedmineConnector::new(RedmineConfig::new("not a url", "p")).is_err());
    }

    #[test]
    fn normalize_builds_keyed_envelope_with_citation() {
        let c = connector();
        let rec = c.normalize(&issue(123, "Disk full", "df -h shows 100%", (1, "New")));
        let env = &rec.envelope;
        assert!(env.has_key());
        assert_eq!(env.source_system.as_deref(), Some("redmine"));
        assert_eq!(env.source_id.as_deref(), Some("123"));
        assert_eq!(
            env.source_url.as_deref(),
            Some("https://redmine.example.com/issues/123")
        );
        assert_eq!(env.source_project.as_deref(), Some("infra"));
        assert_eq!(env.source_author.as_deref(), Some("Jane Dev"));
        assert!(rec.content.contains("Disk full"));
        // Tags lowercased so the case-sensitive tag_prefix filter matches.
        assert!(rec.tags.contains(&"status:new".to_string()));
        assert!(rec.tags.contains(&"tracker:bug".to_string()));
        assert!(rec.tags.contains(&"priority:normal".to_string()));
    }

    #[test]
    fn status_change_changes_hash() {
        let c = connector();
        let new = c
            .normalize(&issue(1, "T", "body", (1, "New")))
            .envelope
            .content_hash;
        let closed = c
            .normalize(&issue(1, "T", "body", (5, "Closed")))
            .envelope
            .content_hash;
        assert_ne!(
            new, closed,
            "a status change must change the hash → re-embed"
        );
    }

    #[test]
    fn journals_fold_in_id_order_and_affect_hash() {
        let c = connector();
        let mut iss = issue(1, "T", "body", (1, "New"));
        iss.journals = vec![
            RawJournal {
                id: 20,
                notes: Some("second".to_string()),
                user: Some(NamedRef {
                    id: 1,
                    name: "B".to_string(),
                }),
                details: vec![],
            },
            RawJournal {
                id: 10,
                notes: Some("first".to_string()),
                user: Some(NamedRef {
                    id: 2,
                    name: "A".to_string(),
                }),
                details: vec![],
            },
            // Pure empty journal must be dropped, not folded.
            RawJournal {
                id: 30,
                notes: None,
                user: Some(NamedRef {
                    id: 3,
                    name: "C".to_string(),
                }),
                details: vec![],
            },
        ];
        let rec = c.normalize(&iss);
        let first = rec.content.find("first").unwrap();
        let second = rec.content.find("second").unwrap();
        assert!(first < second, "journals fold in id order");

        let no_journals = c
            .normalize(&issue(1, "T", "body", (1, "New")))
            .envelope
            .content_hash;
        assert_ne!(
            rec.envelope.content_hash, no_journals,
            "journals must contribute to the hash"
        );
    }

    #[test]
    fn journal_details_and_relations_are_searchable_and_hashed() {
        let c = connector();
        let mut iss = issue(1, "T", "body", (1, "New"));
        iss.journals = vec![RawJournal {
            id: 1,
            notes: None,
            user: Some(NamedRef {
                id: 2,
                name: "A".to_string(),
            }),
            details: vec![RawJournalDetail {
                property: Some("attr".to_string()),
                name: Some("status_id".to_string()),
                old_value: Some("1".to_string()),
                new_value: Some("5".to_string()),
            }],
        }];
        iss.relations = vec![RawRelation {
            id: 9,
            issue_id: Some(1),
            issue_to_id: Some(42),
            relation_type: Some("relates".to_string()),
            delay: None,
        }];

        let rec = c.normalize(&iss);
        assert!(rec.content.contains("changed attr.status_id: 1 -> 5"));
        assert!(rec.content.contains("#42 (relates)"));

        let no_history = c.normalize(&issue(1, "T", "body", (1, "New")));
        assert_ne!(
            rec.envelope.content_hash, no_history.envelope.content_hash,
            "field-change journals and relations must affect idempotent updates"
        );
    }
}
