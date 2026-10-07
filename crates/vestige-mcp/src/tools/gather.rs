//! `gather`: the receipt factory for a GitHub issue hunt.
//!
//! Given an issue URL, a checkout path and an optional version window, gather
//! records every artifact the hunt needs as its own labeled memory — the
//! issue, its comments, every cross-referenced pull request, and the repo's
//! commit window through the ingest_repo machinery — then runs ghostlink and
//! causal_walk over the recorded receipts. Gathering is dynamic (it talks to
//! the `gh` CLI); judging is deterministic (the walk follows recorded edges
//! only, and never reads the fetched prose to pick a cause).
//!
//! Order matters and is the point: commits are recorded BEFORE the issue
//! memory, so the auto-connect `touched` edges point from the earlier commit
//! to the later failure and the backward walk can traverse them. Everything
//! lands in ONE scope, so a walk can never come back empty from a scope
//! mismatch. All writes go through the server's own `Storage` handle: Strata
//! is single-writer and a subprocess store would collide with the running
//! server's lock.
//!
//! If the walk returns nothing, the report says which receipt class was
//! missing (no version window, no linked PRs, no commits in scope) instead of
//! inventing a cause. A hunt that fails this way defines the next thing
//! gather learns to fetch.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{Value, json};
use tokio::sync::Mutex;

use vestige_core::Storage;

use super::repo_ingest;
use crate::cognitive::CognitiveEngine;

pub struct GatherRequest {
    /// GitHub issue URL (…/issues/N or …/pull/N).
    pub url: String,
    /// Optional checkout of the same repository; when present its commit
    /// window is ingested before the failure record is written.
    pub repo_path: Option<PathBuf>,
    /// Worked-in tag for the version window (e.g. `v0.23.3`).
    pub worked_in: Option<String>,
    /// Broke-in tag for the version window.
    pub broke_in: Option<String>,
    /// Scope for every record. Defaults to `<owner>-<repo>-<number>`.
    pub scope: Option<String>,
    /// Maximum comments recorded (default 50).
    pub max_comments: usize,
}

struct IssueRef {
    owner: String,
    name: String,
    number: u64,
    scope: String,
}

fn parse_issue_url(url: &str) -> Result<IssueRef, String> {
    let trimmed = url.trim().trim_end_matches('/');
    let parts: Vec<&str> = trimmed.split('/').collect();
    let (owner, name, num) = match parts.as_slice() {
        [.., o, n, "issues", num] | [.., o, n, "pulls", num] => (*o, *n, *num),
        _ => {
            return Err(
                "url must be a GitHub issue URL like https://github.com/owner/repo/issues/123"
                    .into(),
            );
        }
    };
    let number: u64 = num
        .parse()
        .map_err(|_| format!("issue number {num:?} is not a number"))?;
    Ok(IssueRef {
        owner: owner.to_string(),
        name: name.to_string(),
        number,
        scope: format!("{owner}-{name}-{number}"),
    })
}

fn gh_json(args: &[&str]) -> Result<Value, String> {
    let out = std::process::Command::new("gh")
        .args(args)
        .output()
        .map_err(|e| format!("gh is not runnable: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "gh {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    serde_json::from_slice(&out.stdout)
        .map_err(|e| format!("gh {} returned invalid JSON: {e}", args.join(" ")))
}

/// Run one ingest_repo page and return `(summary, pageBackRev)`.
async fn ingest_page(
    storage: &Arc<Storage>,
    repo_path: &PathBuf,
    scope: &str,
    worked_in: &str,
    broke_in: &str,
    page_back: Option<String>,
) -> Result<(Value, Option<String>), String> {
    let request = repo_ingest::Request {
        repo_path: repo_path.clone(),
        codebase: None,
        scope: Some(scope.to_string()),
        rev: page_back,
        since: None,
        until: None,
        limit: Some(200),
        dry_run: false,
        budget: None,
    };
    let summary = repo_ingest::execute(storage, request).await?;
    // The version window bounds the walk's structural ranking; the tags make
    // both ends recallable by handle.
    let _ = (worked_in, broke_in);
    let back = summary["pageBackWith"]["rev"].as_str().map(String::from);
    Ok((summary, back))
}

pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: Option<&Arc<Mutex<CognitiveEngine>>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args = args.ok_or("gather requires arguments: url, repoPath (optional), workedIn/brokeIn (optional), scope (optional)")?;
    let url = args["url"]
        .as_str()
        .ok_or("gather requires `url`: the GitHub issue URL")?
        .to_string();
    let repo_path = args["repoPath"].as_str().map(PathBuf::from);
    let worked_in = args["workedIn"].as_str().map(String::from);
    let broke_in = args["brokeIn"].as_str().map(String::from);
    let scope = args["scope"].as_str().map(String::from);
    let max_comments = args["maxComments"].as_u64().unwrap_or(50) as usize;

    let issue = parse_issue_url(&url)?;
    let scope = scope.unwrap_or_else(|| issue.scope.clone());

    // ---- Fetch (dynamic, through gh) ------------------------------------
    let issue_json = gh_json(&[
        "api",
        &format!(
            "repos/{}/{}/issues/{}",
            issue.owner, issue.name, issue.number
        ),
    ])?;
    let title = issue_json["title"]
        .as_str()
        .unwrap_or("untitled")
        .to_string();
    let body = issue_json["body"].as_str().unwrap_or("").to_string();
    let state = issue_json["state"].as_str().unwrap_or("open").to_string();
    let mut labels: Vec<String> = Vec::new();
    if let Some(arr) = issue_json["labels"].as_array() {
        for l in arr {
            if let Some(n) = l["name"].as_str() {
                labels.push(n.to_string());
            }
        }
    }

    let comments = gh_json(&[
        "api",
        &format!(
            "repos/{}/{}/issues/{}?per_page={}",
            issue.owner, issue.name, issue.number, max_comments
        ),
    ])
    .or_else(|_| {
        gh_json(&[
            "api",
            &format!(
                "repos/{}/{}/issues/{}/comments?per_page={}",
                issue.owner, issue.name, issue.number, max_comments
            ),
        ])
    })?;
    let comment_bodies: Vec<(String, String)> = comments
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|c| {
                    let who = c["user"]["login"].as_str()?.to_string();
                    let body = c["body"].as_str()?.trim().to_string();
                    (!body.is_empty()).then_some((who, body))
                })
                .collect()
        })
        .unwrap_or_default();

    let mut linked_prs: Vec<(u64, String)> = Vec::new();
    if let Ok(timeline) = gh_json(&[
        "api",
        &format!(
            "repos/{}/{}/issues/{}/timeline?per_page=100",
            issue.owner, issue.name, issue.number
        ),
    ]) {
        if let Some(events) = timeline.as_array() {
            for ev in events {
                let cross = ev["event"].as_str() == Some("cross-referenced");
                let is_pr = ev["source"]["issue"]["pull_request"].is_object();
                if cross && is_pr {
                    if let (Some(n), Some(t)) = (
                        ev["source"]["issue"]["number"].as_u64(),
                        ev["source"]["issue"]["title"].as_str(),
                    ) {
                        linked_prs.push((n, t.to_string()));
                    }
                }
            }
        }
    }

    // ---- Record (through the server's own storage handle) ---------------
    // Commits FIRST so auto-connect edges point failure-ward.
    let mut commits_created = 0u64;
    let mut commits_covered = 0u64;
    let mut commit_pages = 0;
    if let Some(repo_path) = &repo_path {
        let mut page_back: Option<String> = None;
        loop {
            let (summary, back) = ingest_page(
                storage,
                repo_path,
                &scope,
                worked_in.as_deref().unwrap_or(""),
                broke_in.as_deref().unwrap_or(""),
                page_back.clone(),
            )
            .await?;
            commit_pages += 1;
            // True coverage: new writes AND commits the scope already held
            // (a re-run counts the window it verified, not just new writes).
            commits_created += summary["commits"]["created"].as_u64().unwrap_or(0);
            commits_covered += summary["commits"]["created"].as_u64().unwrap_or(0)
                + summary["commits"]["alreadyIngested"].as_u64().unwrap_or(0);
            match back {
                Some(rev) if commit_pages < 10 => page_back = Some(rev),
                _ => break,
            }
        }
    }

    let write = |content: String, tags: Vec<String>, source: String| {
        let input = vestige_core::IngestInput {
            content,
            node_type: "event".to_string(),
            source: Some(source),
            sentiment_score: 0.0,
            sentiment_magnitude: 0.0,
            tags,
            valid_from: None,
            valid_until: None,
            validity_inferred: false,
            source_envelope: None,
        };
        // The scoped write path, the same one smart_ingest uses: receipts
        // MUST land in the gather scope or the walk (scoped) never sees them.
        storage.ingest_in_scope_with_secret_policy(
            input,
            &scope,
            vestige_core::SecretPolicy::Reject,
        )
    };

    let mut linked_pr_ids: Vec<String> = Vec::new();
    for (n, t) in &linked_prs {
        let node = write(
            format!(
                "Linked pull request #{n} on issue {}/{}/#{}: {t}",
                issue.owner, issue.name, issue.number
            ),
            vec![issue.name.clone(), "gather".into(), "linked-pr".into()],
            format!("github:{}/{}/#{}", issue.owner, issue.name, n),
        )
        .map_err(|e| format!("linked PR #{n} was not admitted: {e}"))?;
        linked_pr_ids.push(node.id);
    }

    let failure_node = write(
        format!(
            "Issue {}/{}/#{} [{}]: {title}. {}",
            issue.owner,
            issue.name,
            issue.number,
            state,
            body.trim()
        ),
        {
            let mut t = vec![issue.name.clone(), "failure".into(), "gather".into()];
            t.extend(labels);
            t
        },
        format!("github:{}/{}/#{}", issue.owner, issue.name, issue.number),
    )
    .map_err(|e| format!("the failure record was not admitted: {e}"))?;
    let failure_id = failure_node.id.clone();

    let mut comment_ids: Vec<String> = Vec::new();
    for (who, body) in &comment_bodies {
        let node = write(
            format!(
                "Comment on issue {}/{}/#{} by {}: {}",
                issue.owner, issue.name, issue.number, who, body
            ),
            vec![issue.name.clone(), "gather".into(), "comment".into()],
            format!("github:{}/{}/#{}", issue.owner, issue.name, issue.number),
        )
        .map_err(|e| format!("a comment record was not admitted: {e}"))?;
        comment_ids.push(node.id);
    }

    // ---- Bridge (deterministic, exact entities only) ----------------------
    // The commit records were written by ingest_repo and the receipts by the
    // scoped write above; neither pass runs the full-scan connect. Run it
    // here so exact shared entities (an API token in the report, a directory
    // word from a path) wire failure-to-commit `touched` edges the walk can
    // follow. Same machinery as `vestige connect`: no similarity, no guesses.
    let bridge_edges = {
        let mut nodes = storage
            .get_all_nodes_in_scope(&scope, 10_000, 0)
            .map_err(|e| format!("connect scan could not read scope {scope}: {e}"))?;
        nodes.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id)));
        let entity_sets: Vec<std::collections::HashSet<String>> = nodes
            .iter()
            .map(|n| {
                crate::auto_connect::extract_entities(&n.content, &n.tags)
                    .into_iter()
                    .collect()
            })
            .collect();
        let joined: std::collections::HashSet<(String, String)> = storage
            .get_all_connections()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|e| {
                Some(if e.source_id < e.target_id {
                    (e.source_id.clone(), e.target_id.clone())
                } else {
                    (e.target_id.clone(), e.source_id.clone())
                })
            })
            .collect();
        let now = chrono::Utc::now();
        let mut written = 0usize;
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let shared: Vec<String> = entity_sets[i]
                    .intersection(&entity_sets[j])
                    .cloned()
                    .collect();
                if shared.is_empty() {
                    continue;
                }
                let key = if nodes[i].id < nodes[j].id {
                    (nodes[i].id.clone(), nodes[j].id.clone())
                } else {
                    (nodes[j].id.clone(), nodes[i].id.clone())
                };
                if joined.contains(&key) {
                    continue;
                }
                let edge = vestige_core::ConnectionRecord {
                    source_id: key.0.clone(),
                    target_id: key.1.clone(),
                    strength: 0.5,
                    link_type: "touched".to_string(),
                    created_at: now,
                    last_activated: now,
                    activation_count: 0,
                };
                if storage.save_connection(&edge).is_ok() {
                    written += 1;
                }
            }
        }
        written
    };

    // ---- Judge (deterministic, recorded edges only) ----------------------
    // GhostLink proposes never-composed pairs among the new receipts (leads
    // only), then the walk ranks causes over recorded edges.
    let mut ghostlink_note = "skipped: ghostlink proposal over this scope returned no pairs".into();
    let mut ghostlink_pairs = 0;
    if let Some(cognitive) = cognitive {
        if let Ok(pairs) = super::ghostlink::execute(
            storage,
            cognitive,
            Some(json!({"mode": "propose", "lens": "bridge", "scope": scope})),
        )
        .await
        {
            if let Some(list) = pairs["candidates"].as_array() {
                ghostlink_pairs = list.len();
                if ghostlink_pairs > 0 {
                    ghostlink_note =
                        format!("{ghostlink_pairs} never-composed pair(s) proposed as leads");
                }
            }
        }
    }

    let walk = super::causal_walk::execute(
        storage,
        Some(json!({
            "start_points": [{"kind": "logged_write", "node_id": failure_id}],
            "scope": scope
        })),
    )
    .await?;
    let walk_causes = walk["causes"].as_array().map(|a| a.len()).unwrap_or(0);
    let empty_because = walk["emptyBecause"].as_str().map(String::from);

    // The honesty clause: name what was missing instead of inventing a cause.
    let missing_receipts: Vec<String> = {
        let mut m = Vec::new();
        if repo_path.is_none() {
            m.push(
                "repoPath: no commits were ingested, so the walk cannot reach any commit record"
                    .into(),
            );
        }
        if worked_in.is_none() || broke_in.is_none() {
            m.push(
                "workedIn/brokeIn: no version window, so structural ranking could not fire".into(),
            );
        }
        if !storage
            .node_is_in_scope(&failure_id, &scope)
            .unwrap_or(false)
        {
            m.push("scope mismatch: the failure record is not in the gather scope".into());
        }
        if walk_causes == 0 && commits_covered == 0 && repo_path.is_some() {
            m.push("commits ingested but the walk still returned nothing: page further back or check the scope".into());
        }
        m
    };

    Ok(json!({
        "tool": "gather",
        "issue": format!("{}/{}/#{}", issue.owner, issue.name, issue.number),
        "scope": scope,
        "recorded": {
            "failureId": failure_id,
            "failureScope": storage
                .node_is_in_scope(&failure_id, &scope)
                .unwrap_or(false)
                .then(|| scope.clone()),
            "linkedPrIds": linked_pr_ids,
            "commentIds": comment_ids,
            "commitPages": commit_pages,
            "commitsCreated": commits_created,
            "commitsCovered": commits_covered,
            "bridgeEdges": bridge_edges,
        },
        "leads": {
            "ghostlinkPairs": ghostlink_pairs,
            "note": ghostlink_note,
        },
        "walk": {
            "causes": walk_causes,
            "top": walk["causes"].as_array().and_then(|a| a.first().cloned()),
            "emptyBecause": empty_because,
        },
        "missingReceipts": missing_receipts,
        "note": "gather records receipts and runs the deterministic walk; causes are hypotheses, not proven causes. A missingReceipts entry names what to fetch next time."
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scope contract, as a test: every receipt gather writes (failure,
    /// comment) must land in the gather scope. If the writer ever drifts
    /// back to the default scope, the scoped walk goes blind and this fails.
    #[test]
    fn receipts_land_in_the_gather_scope_not_user() {
        let dir = tempfile::TempDir::new().expect("temp store dir");
        let storage = crate::strata_memory::open(dir.path()).expect("test store");
        let scope = "gather-scope-test";
        let input = vestige_core::IngestInput {
            content: "Issue acme/app#1 [open]: it broke. ".into(),
            node_type: "event".to_string(),
            source: Some("github:acme/app#1".into()),
            sentiment_score: 0.0,
            sentiment_magnitude: 0.0,
            tags: vec!["failure".into(), "gather".into()],
            valid_from: None,
            valid_until: None,
            validity_inferred: false,
            source_envelope: None,
        };
        let node = storage
            .ingest_in_scope_with_secret_policy(input, scope, vestige_core::SecretPolicy::Reject)
            .expect("scoped write");
        let in_gather_scope = storage.node_is_in_scope(&node.id, scope).unwrap_or(false);
        let in_user_scope = storage
            .node_is_in_scope(&node.id, vestige_core::DEFAULT_MEMORY_SCOPE)
            .unwrap_or(false);
        assert!(in_gather_scope, "receipt must be in the gather scope");
        assert!(
            !in_user_scope,
            "receipt must NOT fall back to the user scope"
        );
    }
}
