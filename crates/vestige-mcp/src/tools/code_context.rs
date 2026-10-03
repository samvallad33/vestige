//! Shared current-code retrieval and evidence contract for MCP delivery paths.
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use vestige_core::codebase::{AnchorStatus, AnchorVerification, CodeAnchor, verify_anchor};
use vestige_core::{KnowledgeNode, Storage};

pub(super) fn is_code_memory(node: &KnowledgeNode) -> bool {
    matches!(node.node_type.as_str(), "pattern" | "decision")
        && node
            .tags
            .iter()
            .any(|tag| tag == "codebase" || tag.starts_with("codebase:"))
}

pub(super) fn current_nodes(
    storage: &Arc<Storage>,
    kind: &str,
    codebase: Option<&str>,
    scope: &str,
    limit: i32,
) -> Result<Vec<KnowledgeNode>, String> {
    let tag = codebase.map(|c| format!("codebase:{c}"));
    storage
        .current_code_context_nodes(kind, tag.as_deref(), scope, limit)
        .map_err(|e| format!("Cannot read current code context: {e}"))
}

/// Code memories held in one scope: the advice `get_context` lists (patterns
/// and decisions) and the change records `verify` also checks (events).
pub(super) struct ScopeCounts {
    pub(super) scope: String,
    pub(super) patterns: usize,
    pub(super) decisions: usize,
    /// `event` memories tagged `codebase:<name>`, such as `ingest_repo` change
    /// records. Counted only for a named codebase.
    pub(super) events: usize,
}

impl ScopeCounts {
    /// Any pattern or decision, the kinds `get_context` returns.
    pub(super) fn has_advice(&self) -> bool {
        self.patterns + self.decisions > 0
    }

    /// The row as `get_context` and `session_start` list it.
    pub(super) fn json(&self) -> Value {
        json!({
            "scope": self.scope,
            "patterns": self.patterns,
            "decisions": self.decisions,
            "events": self.events,
        })
    }
}

/// Every scope that holds code memories for `codebase`, with exact totals,
/// ordered by scope name. Events are counted only when a codebase is named:
/// without its exact tag, `event` would match every event in the store.
/// `Err` only when the backend cannot list scopes; callers report that
/// instead of an empty list.
pub(super) fn code_scopes(
    storage: &Arc<Storage>,
    codebase: Option<&str>,
) -> Result<Vec<ScopeCounts>, String> {
    let tag = codebase.map(|c| format!("codebase:{c}"));
    let kinds: &[&str] = if tag.is_some() {
        &["pattern", "decision", "event"]
    } else {
        &["pattern", "decision"]
    };
    let mut by_scope: std::collections::BTreeMap<String, ScopeCounts> =
        std::collections::BTreeMap::new();
    for kind in kinds {
        let counts = storage
            .current_code_context_scope_counts(kind, tag.as_deref())
            .map_err(|e| format!("scope listing is unavailable on this storage backend: {e}"))?;
        for (scope, count) in counts {
            let row = by_scope.entry(scope.clone()).or_insert(ScopeCounts {
                scope,
                patterns: 0,
                decisions: 0,
                events: 0,
            });
            match *kind {
                "pattern" => row.patterns = count,
                "decision" => row.decisions = count,
                _ => row.events = count,
            }
        }
    }
    Ok(by_scope.into_values().collect())
}

/// Skip the generated title so a startup summary contains the actual advice.
pub(super) fn summary(content: &str) -> String {
    let body = content
        .lines()
        .find(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
        .unwrap_or(content)
        .trim();
    let end = body
        .find(". ")
        .map(|i| i + 1)
        .unwrap_or(body.len())
        .min(240);
    body[..body.floor_char_boundary(end)].to_string()
}

/// Reads require an explicit checkout. A missing root never means deleted code.
pub(super) fn annotate(
    storage: &Arc<Storage>,
    items: &mut [Value],
    repo_path: Option<&str>,
    enabled: bool,
) -> Result<Value, String> {
    annotate_with(storage, items, repo_path, enabled, true)
}

/// [`annotate`], choosing whether fresh verdicts are persisted. A read-only
/// tool (`session_start`) passes `persist: false`: it reports the live
/// verdicts and writes nothing.
pub(super) fn annotate_with(
    storage: &Arc<Storage>,
    items: &mut [Value],
    repo_path: Option<&str>,
    enabled: bool,
    persist: bool,
) -> Result<Value, String> {
    let root = repo_path
        .filter(|s| !s.trim().is_empty())
        .and_then(|s| std::fs::canonicalize(s.trim()).ok())
        .filter(|p| p.is_dir());
    let root = if enabled { root } else { None };
    let Some(root) = root else {
        let reason = if enabled {
            "Pass repoPath for an available checkout; source evidence was not checked"
        } else {
            "Source verification disabled by caller"
        };
        for item in items {
            item["anchorStatus"] = json!("unverifiable");
            item["evidence"] = json!({"state":"unavailable", "reason":reason, "checkedAnchors":0, "totalAnchors":null, "claimVerified":false});
        }
        return Ok(json!({"enabled":false, "reason":reason, "repoPath":null}));
    };
    let ids: Vec<String> = items
        .iter()
        .filter_map(|v| v["id"].as_str().map(str::to_owned))
        .collect();
    let verified = verify_nodes_with(storage, &root, &ids, persist)?;
    annotate_items(items, &verified);
    let fresh = verified.values().filter(|(s, _)| s.is_fresh()).count();
    let stale = verified.values().filter(|(s, _)| s.is_stale()).count();
    // Canonical checkout identity is explicit; no shared verification cache is used.
    let repo: PathBuf = root;
    Ok(
        json!({"enabled":true,"repoPath":repo,"checked":ids.len(),"fresh":fresh,"stale":stale,
        "unverifiable":ids.len() - fresh - stale,
        "warning": if stale > 0 { Some(format!("{stale} code memories need rechecking; inspect staleReason before acting")) } else { None },
        "basis":"live checkout; source-span matches do not prove natural-language claims"}),
    )
}

/// One anchor verdict, rendered for the tool response.
pub(super) fn verification_json(v: &AnchorVerification) -> Value {
    serde_json::json!({
        "anchorId": v.anchor_id,
        "checkedAt": v.checked_at.to_rfc3339(),
        "path": v.file_path,
        "symbol": v.symbol,
        "status": v.status.as_str(),
        "detail": v.detail,
        "recordedLine": v.recorded_line,
        "currentLine": v.current_line,
    })
}

/// Roll several anchor verdicts for one memory into a single status.
///
/// Staleness wins over freshness: if any anchor of a memory no longer matches,
/// the memory is flagged. A memory whose anchors are all unverifiable is
/// "unverifiable", never "stale" - this is the legacy path, and accusing a
/// correct memory of being wrong would be worse than the bug being fixed.
fn worst_status(verifications: &[AnchorVerification]) -> AnchorStatus {
    if verifications
        .iter()
        .any(|v| v.status == AnchorStatus::Missing)
    {
        AnchorStatus::Missing
    } else if verifications
        .iter()
        .any(|v| v.status == AnchorStatus::Drifted)
    {
        AnchorStatus::Drifted
    } else if !verifications.is_empty() && verifications.iter().all(|v| v.status.is_fresh()) {
        // Some anchor positively matched and none contradicted it.
        if verifications
            .iter()
            .any(|v| v.status == AnchorStatus::Moved)
        {
            AnchorStatus::Moved
        } else {
            AnchorStatus::Verified
        }
    } else {
        AnchorStatus::Unverifiable
    }
}

/// Verify every anchor belonging to `node_ids` and return, per node, the rolled
/// up status plus the individual verdicts.
///
/// Each fresh verdict is also persisted as the anchor's last-known state
/// (`last_status` / `last_verified_at`). The retrieval path still re-verifies
/// against the live tree on every read - the cached column is informational,
/// feeding recall's `codeEvidence` hint between explicit checks. Writes are
/// change-only so read paths do not churn the writer lock.
pub(super) fn verify_nodes(
    storage: &Arc<Storage>,
    repo_root: &std::path::Path,
    node_ids: &[String],
) -> Result<std::collections::HashMap<String, (AnchorStatus, Vec<AnchorVerification>)>, String> {
    verify_nodes_with(storage, repo_root, node_ids, true)
}

/// [`verify_nodes`], choosing whether changed verdicts are persisted.
pub(super) fn verify_nodes_with(
    storage: &Arc<Storage>,
    repo_root: &std::path::Path,
    node_ids: &[String],
    persist: bool,
) -> Result<std::collections::HashMap<String, (AnchorStatus, Vec<AnchorVerification>)>, String> {
    let mut out = std::collections::HashMap::new();
    let by_node = storage
        .code_anchors_for_nodes(node_ids)
        .map_err(|e| format!("Cannot read code anchors: {e}"))?;
    for (node_id, anchors) in by_node {
        let mut verdicts: Vec<AnchorVerification> = Vec::with_capacity(anchors.len());
        for anchor in &anchors {
            let v = verify_anchor(anchor, repo_root);
            if persist
                && (anchor.last_status != Some(v.status) || anchor.last_verified_at.is_none())
            {
                let _ = storage.record_anchor_verification(&anchor.id, v.status, v.checked_at);
            }
            verdicts.push(v);
        }
        out.insert(node_id, (worst_status(&verdicts), verdicts));
    }
    Ok(out)
}

/// Roll the *persisted* per-anchor statuses of one memory into a single
/// last-known status. Staleness wins over freshness, mirroring
/// [`worst_status`]; `None` means the memory has no anchors at all.
pub(super) fn worst_persisted_status(anchors: &[CodeAnchor]) -> Option<AnchorStatus> {
    if anchors.is_empty() {
        return None;
    }
    let statuses: Vec<AnchorStatus> = anchors.iter().filter_map(|a| a.last_status).collect();
    if statuses.is_empty() {
        return None;
    }
    if statuses.iter().any(|s| s == &AnchorStatus::Missing) {
        Some(AnchorStatus::Missing)
    } else if statuses.iter().any(|s| s == &AnchorStatus::Drifted) {
        Some(AnchorStatus::Drifted)
    } else if statuses.len() < anchors.len() {
        // At least one anchor has never been checked: the honest roll-up is
        // "we last looked and saw X, but not everything has been looked at".
        Some(AnchorStatus::Unverifiable)
    } else if statuses.iter().any(|s| s == &AnchorStatus::Moved) {
        Some(AnchorStatus::Moved)
    } else {
        Some(AnchorStatus::Verified)
    }
}

/// Annotate already-formatted memory items with their verification verdict.
/// Returns the ids of the memories that are visibly stale.
pub(super) fn annotate_items(
    items: &mut [Value],
    verified: &std::collections::HashMap<String, (AnchorStatus, Vec<AnchorVerification>)>,
) -> Vec<String> {
    let mut stale_ids = Vec::new();
    for item in items.iter_mut() {
        let Some(id) = item.get("id").and_then(|v| v.as_str()).map(str::to_string) else {
            continue;
        };
        let Some(obj) = item.as_object_mut() else {
            continue;
        };

        match verified.get(&id) {
            Some((status, verdicts)) => {
                obj.insert(
                    "anchorStatus".to_string(),
                    Value::String(status.as_str().to_string()),
                );
                obj.insert(
                    "anchors".to_string(),
                    Value::Array(verdicts.iter().map(verification_json).collect()),
                );
                let matching = verdicts.iter().filter(|v| v.status.is_fresh()).count();
                let checked = verdicts
                    .iter()
                    .filter(|v| v.status != AnchorStatus::Unverifiable)
                    .count();
                obj.insert("evidence".into(), serde_json::json!({
                    "state": if status.is_stale() { "needs_recheck" }
                        else if matching == verdicts.len() && matching > 0 { "unchanged_evidence" }
                        else if checked > 0 { "partial" } else { "unavailable" },
                    "checkedAnchors": checked, "matchingAnchors": matching,
                    "totalAnchors": verdicts.len(),
                    "claimVerified": false,
                }));
                if status.is_stale() {
                    stale_ids.push(id);
                    let reason = verdicts
                        .iter()
                        .filter(|v| v.is_stale())
                        .map(|v| v.detail.clone())
                        .collect::<Vec<_>>()
                        .join(" ");
                    obj.insert("stale".to_string(), Value::Bool(true));
                    obj.insert("staleReason".to_string(), Value::String(reason));
                }
            }
            None => {
                obj.insert(
                    "evidence".into(),
                    serde_json::json!({
                        "state": "unanchored", "checkedAnchors": 0, "matchingAnchors": 0,
                        "totalAnchors": 0, "claimVerified": false,
                    }),
                );
                // No anchor row at all: every memory written before anchoring
                // existed lands here. Unverifiable, explicitly not stale.
                obj.insert(
                    "anchorStatus".to_string(),
                    Value::String("unanchored".to_string()),
                );
                obj.insert(
                    "anchorNote".to_string(),
                    Value::String(
                        "This memory has no source anchor, so Vestige cannot check it against the code. It may be perfectly correct - it just cannot prove it.".to_string(),
                    ),
                );
            }
        }
    }
    stale_ids
}
