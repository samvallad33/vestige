//! # Retroactive Salience Backfill — MCP tool
//!
//! Propose earlier memories related to a recorded failure through shared
//! entities and chronology. These are investigation candidates, not proven
//! causes. The default preview does not persist edges or reinforce memories.
//! Explicit promote=true records candidate edges and reinforces eligible nodes.

use chrono::Utc;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

use vestige_core::advanced::git_records;
use vestige_core::advanced::retroactive_backfill::{
    self, BackfillCandidate, BackfillResult, ExcludedCandidate, FailureEvent, RetroactiveBackfill,
};
use vestige_core::{ConnectionRecord, KnowledgeNode, Storage};

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "scope": {"type": "string", "default": "user", "description": "Exact project namespace for failure and candidates. Defaults to user; cross-project inference is not performed."},
            "failure_id": {
                "type": "string",
                "description": "Failure memory to backfill from. Omitted: the most recent failure-like memory."
            },
            "manual": {
                "type": "boolean",
                "description": "Force the backfill when the event is not auto-detected as salient. Default false.",
                "default": false
            },
            "lookback_days": {
                "type": "integer",
                "description": "How many days back to reach for the cause. Default 30.",
                "minimum": 1,
                "maximum": 365,
                "default": 30
            },
            "promote": {
                "type": "boolean",
                "description": "Explicitly reinforce candidates and record candidate edges after review. Default false: preview only, with no graph or strength mutation. Promotion does not verify causality.",
                "default": false
            },
            "scan_limit": {
                "type": "integer",
                "description": "Max memories to scan as candidate causes. Default 500.",
                "minimum": 10,
                "maximum": 5000,
                "default": 500
            },
            "why_not": {
                "type": "string",
                "description": "Memory id (or commit sha prefix): report the exact rule that excluded this record, or its rank among the surfaced causes."
            },
            "git_repo": {
                "type": "string",
                "description": "Path to the git repository backing this scope's commit records. Enables version-range mapping: 'broke in X, worked in Y' in the failure text (or explicit worked_in/broke_in tags) restricts commit candidates to that range."
            },
            "worked_in": {
                "type": "string",
                "description": "Last-known-good tag for the version range. Requires git_repo."
            },
            "broke_in": {
                "type": "string",
                "description": "First-bad tag for the version range. Requires git_repo."
            }
        }
    })
}

#[derive(Deserialize, Default)]
struct Args {
    failure_id: Option<String>,
    scope: Option<String>,
    #[serde(default)]
    manual: bool,
    lookback_days: Option<i64>,
    promote: Option<bool>,
    scan_limit: Option<i32>,
    why_not: Option<String>,
    git_repo: Option<String>,
    worked_in: Option<String>,
    broke_in: Option<String>,
}

/// Pull entities out of a memory: its tags, plus heuristic code-ish tokens from
/// content (UPPER_SNAKE env vars, dotted/slashed file paths). These are the
/// shared-entity join keys the backward reach follows.
///
/// Thin `&KnowledgeNode` adapter over the single core definition
/// [`retroactive_backfill::extract_entities`] so the MCP tool, CLI, and the
/// offline consolidation pass all extract entities identically (no drift).
fn extract_entities(node: &KnowledgeNode) -> Vec<String> {
    retroactive_backfill::extract_entities(&node.content, &node.tags)
}

/// Heuristic: does this memory read like a failure/"aversive event"? Checks both
/// content AND tags against the full FAILURE_MARKERS list. Public so the CLI and
/// any caller share ONE failure-detection definition (no drifting subsets).
///
/// Thin `&KnowledgeNode` adapter over [`retroactive_backfill::looks_like_failure`].
pub fn looks_like_failure(node: &KnowledgeNode) -> bool {
    retroactive_backfill::looks_like_failure(&node.content, &node.tags)
}

type VersionRange = (String, String, std::collections::HashSet<String>);

/// Map "broke in X, worked in Y" to the commit set between the two tags.
/// Explicit tags win; otherwise semver tokens in the failure text are matched
/// against the repo's tags (lowest = worked in, highest = broke in).
///
/// Explicit tags that fail to resolve (typo, shallow clone without the tag) or
/// resolve to an EMPTY range (worked/broke swapped) are ERRORS: silently
/// dropping a range the caller explicitly asked for would widen the candidate
/// set without a hint. Auto-detected ranges degrade silently to no-range.
fn resolve_version_range(
    repo: &str,
    worked_in: Option<&str>,
    broke_in: Option<&str>,
    failure_text: &str,
) -> Result<Option<VersionRange>, String> {
    let explicit = worked_in.is_some() && broke_in.is_some();
    let (worked, broke) = match (worked_in, broke_in) {
        (Some(w), Some(b)) => (w.to_string(), b.to_string()),
        _ => {
            let Some(tags) = git_lines(repo, &["tag", "--list"]) else {
                return Ok(None);
            };
            let versions = git_records::extract_versions(failure_text);
            let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
            let matched = git_records::match_version_tags(&tag_refs, &versions);
            let Some((w, b)) = git_records::version_range(&matched) else {
                return Ok(None);
            };
            (w, b)
        }
    };
    let Some(shas) = git_lines(repo, &["rev-list", &format!("{worked}..{broke}")]) else {
        if explicit {
            return Err(format!(
                "could not resolve the version range {worked}..{broke} in {repo} (tag missing or not fetched?)"
            ));
        }
        return Ok(None);
    };
    if shas.is_empty() && explicit {
        return Err(format!(
            "version range {worked}..{broke} is empty — broke_in must come after worked_in"
        ));
    }
    Ok(Some((
        worked,
        broke,
        git_records::parse_rev_list(&shas.join("\n")),
    )))
}

fn git_lines(repo: &str, git_args: &[&str]) -> Option<Vec<String>> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(git_args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
    )
}

/// The full sha a commit record carries (first line: `commit <sha> ...`).
fn commit_sha(content: &str) -> Option<String> {
    let mut words = content.split_whitespace();
    if words.next() != Some("commit") {
        return None;
    }
    let sha = words.next()?;
    (sha.len() >= 7 && sha.chars().all(|c| c.is_ascii_hexdigit())).then(|| sha.to_ascii_lowercase())
}

/// Pure candidate assembly so the supersession and version-range rules stay
/// unit-testable without a store. Older memories become candidates; a
/// superseded record passes its candidacy to its replacement, dated by the
/// superseded record (the fact's origin); commit records outside the mapped
/// version range are excluded with a why-not reason.
fn build_candidates(
    failure: &FailureEvent,
    failure_created: chrono::DateTime<Utc>,
    all: &[KnowledgeNode],
    supersession: &std::collections::HashMap<String, String>,
    range: Option<&VersionRange>,
) -> (Vec<BackfillCandidate>, Vec<ExcludedCandidate>, usize) {
    let mut candidates: Vec<BackfillCandidate> = Vec::new();
    let mut excluded: Vec<ExcludedCandidate> = Vec::new();
    let mut commit_records = 0usize;

    for node in all {
        if node.id == failure.id || node.suppression_count > 0 {
            continue;
        }
        // follow supersession to the current belief (transitive, cycle-capped)
        let mut terminal = node.id.clone();
        for _ in 0..5 {
            match supersession.get(&terminal) {
                Some(next) if next != &terminal => terminal = next.clone(),
                _ => break,
            }
        }
        let (origin, current): (&KnowledgeNode, &KnowledgeNode) = if terminal == node.id {
            (node, node)
        } else {
            match all.iter().find(|n| n.id == terminal) {
                Some(rep) => (node, rep),
                None => continue, // replacement outside scope: trail ends here
            }
        };
        if current.id == failure.id {
            continue;
        }

        let age = (failure_created - origin.created_at).num_seconds() as f64 / 86_400.0;
        let mut entities = retroactive_backfill::extract_entities(&current.content, &current.tags);
        if current.id != origin.id {
            for e in retroactive_backfill::extract_entities(&origin.content, &origin.tags) {
                if !entities.contains(&e) {
                    entities.push(e);
                }
            }
        }
        let is_commit = current
            .tags
            .iter()
            .chain(origin.tags.iter())
            .any(|t| t == git_records::COMMIT_TAG);
        let candidate = BackfillCandidate {
            id: current.id.clone(),
            content: current.content.clone(),
            entities,
            age_days_before_failure: age,
            stability: current.stability,
            via_supersession_of: if current.id != origin.id {
                Some(origin.id.clone())
            } else {
                None
            },
            is_change_record: is_commit,
        };

        if is_commit {
            commit_records += 1;
            if let Some((worked, broke, allowed)) = range {
                let in_range = commit_sha(&candidate.content)
                    .map(|sha| allowed.contains(&sha))
                    .unwrap_or(false);
                if !in_range {
                    let reason = match commit_sha(&candidate.content) {
                        Some(_) => format!("outside version range {worked}..{broke}"),
                        None => "commit sha unknown; cannot place in the version range".to_string(),
                    };
                    excluded.push(ExcludedCandidate { candidate, reason });
                    continue;
                }
            }
        }

        // the origin timeline is the fact's timeline: a traversal candidate
        // replaces a direct candidate for the same replacement record
        match candidates.iter_mut().find(|c| c.id == candidate.id) {
            Some(existing) if candidate.via_supersession_of.is_some() => *existing = candidate,
            Some(_) => {}
            None => candidates.push(candidate),
        }
    }
    (candidates, excluded, commit_records)
}

/// One-line answer to "why not X?" for a memory id or commit sha prefix.
///
/// The rule is DERIVED from the candidate itself rather than read from
/// `result.rejected`: the rejected list is truncated to the strongest few, so a
/// truncated rejection would otherwise be mislabeled "ranked below the top 3".
/// Caller-side exclusions (version range) are answered from `excluded`.
fn explain_why_not(
    target: &str,
    failure: &FailureEvent,
    result: &BackfillResult,
    candidates: &[BackfillCandidate],
    excluded: &[ExcludedCandidate],
    lookback_days: i64,
    contents: &std::collections::HashMap<String, String>,
) -> Value {
    let lowered = target.trim().to_lowercase();
    let is_hexish = lowered.chars().all(|c| c.is_ascii_hexdigit());
    if lowered.len() < 7 && is_hexish && !lowered.is_empty() {
        return json!({
            "target": target,
            "verdict": "ambiguous",
            "detail": "sha prefix too short; use at least 7 characters or the memory id",
        });
    }
    // caller-excluded records (e.g. outside the version range) are not in the
    // candidate pool; resolve them first so their reason is not lost
    if let Some(ex) = excluded
        .iter()
        .find(|ex| ex.candidate.id.to_lowercase() == lowered)
        .or_else(|| {
            excluded.iter().find(|ex| {
                contents
                    .get(&ex.candidate.id)
                    .and_then(|c| commit_sha(c))
                    .is_some_and(|sha| sha.starts_with(&lowered))
            })
        })
    {
        return json!({
            "target": target,
            "verdict": "rejected",
            "detail": ex.reason,
        });
    }
    let id = candidates
        .iter()
        .map(|c| &c.id)
        .find(|id| id.to_lowercase() == lowered)
        .cloned()
        .or_else(|| {
            // sha prefix: match the record whose content carries that commit
            candidates.iter().find_map(|c| {
                let content = contents.get(&c.id)?;
                let sha = commit_sha(content)?;
                sha.starts_with(&lowered).then(|| c.id.clone())
            })
        });
    let Some(id) = id else {
        return json!({
            "target": target,
            "verdict": "not a candidate",
            "detail": "not in this scope, suppressed, or outside the scanned window entirely",
        });
    };
    if let Some((rank, cause)) = result
        .causes
        .iter()
        .enumerate()
        .find(|(_, c)| c.memory_id == id)
    {
        return json!({
            "target": target,
            "verdict": "surfaced",
            "detail": format!("cause candidate #{}: {}", rank + 1, cause.reason),
        });
    }
    // derive the rule from the candidate, not the (truncated) rejected list
    if let Some(c) = candidates.iter().find(|c| c.id == id) {
        let shared = c
            .entities
            .iter()
            .filter(|e| failure.entities.contains(e))
            .count();
        let rule = if c.age_days_before_failure <= 0.0 {
            "record is newer than the failure".to_string()
        } else if c.age_days_before_failure > lookback_days as f64 {
            format!("outside the {lookback_days}d lookback window")
        } else if shared == 0 {
            "shares no entity with the failure".to_string()
        } else {
            format!(
                "ranked below the top {}: scored lower than the surfaced causes (shares {shared} entit{} with the failure)",
                result.causes.len().max(1),
                if shared == 1 { "y" } else { "ies" }
            )
        };
        return json!({
            "target": target,
            "verdict": "rejected",
            "detail": rule,
        });
    }
    json!({
        "target": target,
        "verdict": "not a candidate",
        "detail": "not in this scope, suppressed, or outside the scanned window entirely",
    })
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        return Err("pending_strata: backfill entity overlap is not a recorded edge".into());
    }
    let args: Args = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| e.to_string())?,
        None => Args::default(),
    };
    // Clamp numeric inputs to the documented schema bounds. The MCP dispatch
    // layer does NOT enforce the JSON-schema min/max, so a caller can send
    // scan_limit=-1 (SQLite treats a negative LIMIT as unbounded => full-table
    // fetch = DoS) or values above the 5000 cap. Clamp rather than trust.
    let lookback = args.lookback_days.unwrap_or(30).clamp(1, 365);
    let promote = args.promote.unwrap_or(false);
    let scan_limit = args.scan_limit.unwrap_or(500).clamp(10, 5000);

    let scope = args.scope.as_deref().unwrap_or("user").trim();
    if scope.is_empty() {
        return Err("scope must not be empty".into());
    }

    // 1. Resolve the failure event within the requested namespace.
    let failure_node = match &args.failure_id {
        Some(id) => {
            if !storage
                .node_is_in_scope(id, scope)
                .map_err(|e| e.to_string())?
            {
                return Err("failure memory not found in requested scope".into());
            }
            storage
                .get_node(id)
                .map_err(|e| e.to_string())?
                .ok_or_else(|| "failure memory not found in requested scope".to_string())?
        }
        None => {
            // most recent memory that looks like a failure
            let recent = storage
                .get_all_nodes_in_scope(scope, scan_limit, 0)
                .map_err(|e| e.to_string())?;
            recent.into_iter().find(looks_like_failure).ok_or_else(|| {
                "no failure-like memory found to backfill from; pass failure_id or manual=true"
                    .to_string()
            })?
        }
    };

    let failure_entities = extract_entities(&failure_node);

    // surprise/prediction-error proxy: a failure-marked memory is treated as
    // high-salience; otherwise fall back to a neutral value (manual can force).
    let pe = if looks_like_failure(&failure_node) {
        0.9_f32
    } else {
        0.3_f32
    };

    let failure = FailureEvent {
        id: failure_node.id.clone(),
        content: failure_node.content.clone(),
        entities: failure_entities.clone(),
        tags: failure_node.tags.clone(),
        prediction_error: pe,
        manual: args.manual,
    };

    // 2. Build candidate causes from all OTHER memories (older than the failure).
    // Supersession is followed, not fatal: a superseded record passes its
    // candidacy to its replacement, dated by the superseded record.
    let all = storage
        .get_all_nodes_in_scope(scope, scan_limit, 0)
        .map_err(|e| e.to_string())?;
    let supersession: std::collections::HashMap<String, String> = storage
        .supersession_pairs()
        .map_err(|e| e.to_string())?
        .into_iter()
        .collect();
    let range = match &args.git_repo {
        Some(repo) => resolve_version_range(
            repo,
            args.worked_in.as_deref(),
            args.broke_in.as_deref(),
            &failure_node.content,
        )?,
        None => None,
    };
    let (candidates, excluded, commit_records) = build_candidates(
        &failure,
        failure_node.created_at,
        &all,
        &supersession,
        range.as_ref(),
    );
    let contents: std::collections::HashMap<String, String> = all
        .iter()
        .map(|n| (n.id.clone(), n.content.clone()))
        .collect();

    // 3. Run the backward reach.
    let backfill = RetroactiveBackfill {
        lookback_days: lookback,
        ..RetroactiveBackfill::new()
    };
    let result = backfill.run_trail(&failure, &candidates, &excluded);

    if !result.triggered {
        return Ok(json!({
            "tool": "backfill",
            "triggered": false,
            "reason": "the event was not salient (not a detected failure and manual=false). Pass manual=true to force.",
            "failure_id": failure.id,
            // answer the why-not question even here: there is no ranking to
            // interrogate until the reach actually runs
            "why_not": args.why_not.as_deref().map(|t| json!({
                "target": t,
                "verdict": "not triggered",
                "detail": "the event was not salient, so no candidate was ranked or excluded; pass manual=true to force the reach",
            })),
        }));
    }

    // 4. Promote the surfaced cause(s) so they stop decaying and resurface.
    let mut promoted = Vec::new();
    for cause in &result.causes {
        let content_preview = candidates
            .iter()
            .find(|c| c.id == cause.memory_id)
            .map(|c| c.content.chars().take(140).collect::<String>())
            .unwrap_or_default();
        // Preview never records a relationship. Explicit promotion preserves
        // the weaker candidate edge type; it does not create a causal fact.
        let link_type = "backfill_candidate".to_string();
        let already_linked = storage
            .get_connections_for_memory(&cause.memory_id)
            .map(|connections| {
                connections.iter().any(|connection| {
                    connection.source_id == cause.memory_id
                        && connection.target_id == failure_node.id
                        && connection.link_type == link_type
                })
            })
            .unwrap_or(false);
        let candidate_edge_persisted = already_linked
            || (promote
                && storage
                    .save_connection(&ConnectionRecord {
                        source_id: cause.memory_id.clone(),
                        target_id: failure_node.id.clone(),
                        strength: cause.score.clamp(0.0, 1.0),
                        link_type,
                        created_at: Utc::now(),
                        last_activated: Utc::now(),
                        activation_count: 0,
                    })
                    .is_ok());
        let now = Utc::now();
        let eligible_for_promotion = all
            .iter()
            .find(|node| node.id == cause.memory_id)
            .is_some_and(|node| {
                node.valid_from.is_none_or(|date| date <= now)
                    && node.valid_until.is_none_or(|date| date > now)
            });
        let did_promote = if promote && candidate_edge_persisted && eligible_for_promotion {
            // promote_memory_backfill boosts retrieval strength + reps (the
            // FSRS knob) with a bounded stability multiply.
            storage.promote_memory_backfill(&cause.memory_id).is_ok()
        } else {
            false
        };
        promoted.push(json!({
            "memory_id": cause.memory_id,
            "content_preview": content_preview,
            "shared_entities": cause.shared_entities,
            "age_days_before_failure": (cause.age_days * 10.0).round() / 10.0,
            "backfill_score": (cause.score * 100.0).round() / 100.0,
            "promoted": did_promote,
            "eligible_for_promotion": eligible_for_promotion,
            "candidate_edge_persisted": candidate_edge_persisted,
            "reason": cause.reason,
            "evidence_status": "hypothesis",
            "causality_verified": false,
        }));
    }

    // why-not-X: one line per named suspect
    let why_not = args.why_not.as_deref().map(|t| {
        explain_why_not(
            t,
            &failure,
            &result,
            &candidates,
            &excluded,
            lookback,
            &contents,
        )
    });

    // gap suggestion: name the concrete missing record kind
    let gap_value = result.gap.clone().map(|gap| {
        let suggestion = if commit_records == 0 {
            Some("no commit records in this scope — `vestige ingest-git <repo>` adds them so the trail can reach code".to_string())
        } else if !gap.missing_entities.is_empty() {
            Some(format!(
                "a commit or note touching one of: {}",
                gap.missing_entities.iter().take(3).cloned().collect::<Vec<_>>().join(", ")
            ))
        } else {
            None
        };
        json!({
            "missing_entities": gap.missing_entities,
            "note": gap.note,
            "suggestion": suggestion,
        })
    });

    Ok(json!({
        "tool": "backfill",
        "triggered": true,
        "lookback_days": lookback,
        "scope": scope,
        "preview": !promote,
        "evidence_status": "hypothesis",
        "causality_verified": false,
        "scan_limit": scan_limit,
        "scan_limit_reached": all.len() == scan_limit as usize,
        "headline": format!(
            "Found {} earlier candidate memor{} linked by shared entities; investigate before attributing cause.",
            result.causes.len(),
            if result.causes.len() == 1 { "y" } else { "ies" }
        ),
        "failure": {
            "id": failure.id,
            "content_preview": failure.content.chars().take(160).collect::<String>(),
            "entities": failure_entities,
        },
        "scanned": result.scanned,
        // why-not-X: strongest rejections with the rule that excluded each
        "rejected": result.rejected.iter().map(|r| json!({
            "memory_id": r.memory_id,
            "content_preview": contents.get(&r.memory_id).map(|c| c.chars().take(140).collect::<String>()).unwrap_or_default(),
            "reason": r.reason,
        })).collect::<Vec<_>>(),
        // trail-break report: which link is missing and what would close it
        "gap": gap_value,
        "version_range": range.as_ref().map(|(w, b, shas)| json!({
            "worked_in": w,
            "broke_in": b,
            "commits_in_range": shas.len(),
        })),
        "commit_records_in_scope": commit_records,
        "why_not": why_not,
        // Explicit direct evidence edge from the highest-ranked candidate to
        // the failure — not a topology inferred by the renderer.
        "path_ids": promoted.iter().find(|candidate| candidate["candidate_edge_persisted"] == true)
            .and_then(|cause| cause.get("memory_id"))
            .and_then(|id| id.as_str())
            .map(|id| vec![id.to_string(), failure.id.clone()])
            .unwrap_or_default(),
        "causes": promoted,
        "note": "The legacy causes field contains candidates ranked by shared entities and chronology. Scores are ranking heuristics, not probabilities of causation. Similarity rank compares scanned candidates only; no counterfactual search result is proven. Preview does not write graph edges or reinforce memory.",
    }))
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use vestige_core::IngestInput;

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        (storage, dir)
    }

    #[tokio::test]
    async fn backfill_scopes_before_limit_and_checks_explicit_failure() {
        let (storage, _dir) = test_storage().await;
        let own = storage
            .ingest(IngestInput {
                content: "Failure in FIXTURE_SERVICE".into(),
                ..Default::default()
            })
            .unwrap();
        let mut foreign_id = String::new();
        for _ in 0..15 {
            foreign_id = storage
                .ingest_in_scope(
                    IngestInput {
                        content: "Failure in FOREIGN_SERVICE".into(),
                        ..Default::default()
                    },
                    "other-project",
                )
                .unwrap()
                .id;
        }
        let out = execute(&storage, Some(json!({"scan_limit": 10})))
            .await
            .unwrap();
        assert_eq!(out["failure"]["id"], own.id);
        assert_eq!(out["scope"], "user");
        assert!(
            execute(&storage, Some(json!({"failure_id": foreign_id})))
                .await
                .is_err()
        );
        let out = execute(
            &storage,
            Some(json!({"failure_id": foreign_id, "scope": "other-project"})),
        )
        .await
        .unwrap();
        assert_eq!(out["scope"], "other-project");
        assert!(
            execute(&storage, Some(json!({"scope": " "})))
                .await
                .is_err()
        );
    }

    /// LIVE end-to-end: plant a quiet env-var cause, a semantic distractor, and a
    /// failure into a REAL SQLite store, then run the backfill MCP tool and assert
    /// it surfaces the causal env-var memory by the shared API_TIMEOUT entity —
    /// the root cause a vector search would never rank first. This is the
    /// reproducible receipt behind the demo.
    #[tokio::test]
    async fn live_backfill_surfaces_root_cause_through_storage() {
        let (storage, _dir) = test_storage().await;

        // 1) The quiet cause: an env-var edit (no failure words; not "similar" to a crash).
        //    Backdated 3 days so the backward reach can find it (the demo scenario).
        let cause = storage
            .ingest(IngestInput {
                content: "Set API_TIMEOUT=2 in the deploy env to speed up cold starts".to_string(),
                node_type: "decision".to_string(),
                tags: vec!["API_TIMEOUT".to_string(), "deploy-env".to_string()],
                ..Default::default()
            })
            .unwrap();
        storage
            .set_created_at(&cause.id, chrono::Utc::now() - chrono::Duration::days(3))
            .unwrap();

        // 2) A semantic distractor: looks like a crash report, but shares NO
        //    entity with the failure. Wording matters under the tiered
        //    extractor (w1c): reusing the failure's own words ("service",
        //    "internal", "server", "error") would hand the distractor four
        //    word-tier shared entities, and four rare-ish words can outscore
        //    one code-tier join. "Outage/billing/system" keep the resemblance
        //    for a human while joining on nothing.
        //    Backdated 20 days (also in the past, so only the entity link decides).
        let distractor = storage
            .ingest(IngestInput {
                content: "An outage hit the billing system last month".to_string(),
                node_type: "event".to_string(),
                tags: vec!["billing-system".to_string()],
                ..Default::default()
            })
            .unwrap();
        storage
            .set_created_at(
                &distractor.id,
                chrono::Utc::now() - chrono::Duration::days(20),
            )
            .unwrap();

        // 3) The failure, recorded last (most recent) — the "aversive event".
        let failure = storage
            .ingest(IngestInput {
                content: "Service crashed: 500 Internal Server Error on the auth endpoint"
                    .to_string(),
                node_type: "event".to_string(),
                tags: vec![
                    "auth-service".to_string(),
                    "API_TIMEOUT".to_string(),
                    "crash".to_string(),
                ],
                ..Default::default()
            })
            .unwrap();

        // Default discovery is a true preview, including graph state.
        let before = storage.get_node(&cause.id).unwrap().unwrap();
        for _ in 0..2 {
            let preview = execute(&storage, Some(json!({"failure_id": failure.id})))
                .await
                .unwrap();
            assert_eq!(preview["preview"], true);
            assert_eq!(preview["causality_verified"], false);
            assert_eq!(preview["causes"][0]["promoted"], false);
            assert_eq!(preview["causes"][0]["candidate_edge_persisted"], false);
            assert_eq!(preview["path_ids"], json!([]));
        }
        let after = storage.get_node(&cause.id).unwrap().unwrap();
        assert_eq!(before.reps, after.reps);
        assert_eq!(before.stability, after.stability);
        assert!(
            storage
                .get_connections_for_memory(&cause.id)
                .unwrap()
                .is_empty()
        );

        // Run the backfill tool against the real store (auto-finds the failure).
        let out = execute(&storage, Some(json!({ "promote": true, "manual": false })))
            .await
            .expect("backfill must run");

        assert_eq!(
            out["triggered"],
            json!(true),
            "the crash must trigger a backfill"
        );
        let causes = out["causes"].as_array().expect("causes array");
        assert!(!causes.is_empty(), "must surface at least one cause");

        // The top cause is the env-var memory, surfaced by the shared API_TIMEOUT entity.
        let top = &causes[0];
        let content = top["content_preview"].as_str().unwrap_or("");
        assert!(
            content.contains("API_TIMEOUT") && content.contains("deploy"),
            "top cause must be the env-var edit, got: {content}"
        );
        let shared = top["shared_entities"].as_array().unwrap();
        assert!(
            shared.iter().any(|e| e.as_str() == Some("api_timeout")),
            "must link via the shared API_TIMEOUT entity, got: {shared:?}"
        );
        assert_eq!(
            top["candidate_edge_persisted"],
            json!(true),
            "the candidate evidence edge must exist before a receipt can claim it"
        );
        assert_eq!(
            out["path_ids"],
            json!([cause.id, failure.id]),
            "the tool must publish the exact backend-authored candidate-to-failure path"
        );
        let edges = storage.get_connections_for_memory(&cause.id).unwrap();
        assert!(
            edges.iter().any(|edge| {
                edge.source_id == cause.id
                    && edge.target_id == failure.id
                    && edge.link_type == "backfill_candidate"
            }),
            "candidate evidence edge must be persisted"
        );
        // It was actually promoted in the real store.
        assert_eq!(
            top["promoted"],
            json!(true),
            "the cause must be promoted in storage"
        );
        // Sanity: the failure we ingested is the one that fired.
        assert_eq!(out["failure"]["id"], json!(failure.id));
    }

    // ---- candidate assembly: supersession traversal + version-range rules ----

    fn node(id: &str, content: &str, tags: &[&str], days_ago: i64) -> KnowledgeNode {
        // non_exhaustive: default-then-mutate is the only cross-crate construction
        let mut n = KnowledgeNode::default();
        n.id = id.into();
        n.content = content.into();
        n.tags = tags.iter().map(|t| t.to_string()).collect();
        n.created_at = Utc::now() - chrono::Duration::days(days_ago);
        n.suppression_count = 0;
        n
    }

    fn failure_event() -> (FailureEvent, chrono::DateTime<Utc>) {
        (
            FailureEvent {
                id: "fail".into(),
                content: "crash in events/local.py".into(),
                entities: vec!["events/local.py".into()],
                tags: vec![],
                prediction_error: 0.9,
                manual: true,
            },
            Utc::now(),
        )
    }

    fn commit_record_content(sha: &str) -> String {
        git_records::record_content(&git_records::GitCommit {
            sha: sha.into(),
            time: Utc::now(),
            subject: "fix local write".into(),
            files: vec!["events/local.py".into()],
            symbols: vec![],
            extra_files: 0,
            mentions: vec![],
            hunks: vec![],
            extra_hunks: 0,
            imports: vec![],
            parents: vec![],
            reverts: None,
            lock_bumps: vec![],
        })
    }

    #[test]
    fn superseded_record_passes_its_candidacy_to_the_replacement() {
        // old record (pre-failure, carries the join key) was superseded by a
        // newer belief created AFTER the failure; the trail must follow.
        let (failure, failure_created) = failure_event();
        let all = vec![
            node("fail", "crash in events/local.py", &[], 0),
            node(
                "new-belief",
                "events/local.py uses a ring buffer now",
                &[],
                -3,
            ), // post-failure
            node(
                "old-fact",
                "events/local.py writes via LocalFileStore",
                &[],
                10,
            ),
        ];
        let mut supersession = std::collections::HashMap::new();
        supersession.insert("old-fact".to_string(), "new-belief".to_string());

        let (candidates, excluded, _) =
            build_candidates(&failure, failure_created, &all, &supersession, None);
        assert!(excluded.is_empty());
        let cand = candidates
            .iter()
            .find(|c| c.id == "new-belief")
            .expect("trail must reach the replacement");
        assert_eq!(cand.via_supersession_of.as_deref(), Some("old-fact"));
        // dated by the superseded record: 10 days before the failure, not -3
        assert!((cand.age_days_before_failure - 10.0).abs() < 0.01);
        // join keys are the union of old and new wording
        assert!(cand.entities.iter().any(|e| e.contains("local.py")));
    }

    #[test]
    fn commits_outside_the_version_range_are_excluded_with_a_reason() {
        let (failure, failure_created) = failure_event();
        let good = "1111111111111111111111111111111111111111";
        let bad = "2222222222222222222222222222222222222222";
        let all = vec![
            node("fail", "crash in events/local.py after upgrade", &[], 0),
            node(
                "c-good",
                &commit_record_content(good),
                &[git_records::COMMIT_TAG],
                5,
            ),
            node(
                "c-bad",
                &commit_record_content(bad),
                &[git_records::COMMIT_TAG],
                20,
            ),
        ];
        let range = (
            "1.41.0".to_string(),
            "1.42.1".to_string(),
            [good.to_string()].into_iter().collect(),
        );
        let (candidates, excluded, commit_records) = build_candidates(
            &failure,
            failure_created,
            &all,
            &Default::default(),
            Some(&range),
        );
        assert_eq!(commit_records, 2);
        assert!(candidates.iter().any(|c| c.id == "c-good"));
        // commit records carry the change-record flag the scorer bonuses
        assert!(
            candidates
                .iter()
                .find(|c| c.id == "c-good")
                .unwrap()
                .is_change_record
        );
        assert!(!candidates.iter().any(|c| c.id == "c-bad"));
        assert_eq!(excluded.len(), 1);
        assert_eq!(excluded[0].candidate.id, "c-bad");
        assert_eq!(excluded[0].reason, "outside version range 1.41.0..1.42.1");
    }

    fn temp_git_repo() -> tempfile::TempDir {
        use std::process::Command;
        let dir = tempfile::TempDir::new().unwrap();
        let git = |args: &[&str]| {
            let out = Command::new("git")
                .arg("-C")
                .arg(dir.path())
                .args(["-c", "user.email=t@t", "-c", "user.name=t"])
                .args(args)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        };
        git(&["init", "-q"]);
        git(&["commit", "--allow-empty", "-q", "-m", "one"]);
        git(&["tag", "v1.41.0"]);
        git(&["commit", "--allow-empty", "-q", "-m", "two"]);
        git(&["tag", "v1.42.1"]);
        dir
    }

    #[test]
    fn explicit_range_failures_are_errors_not_silent_fallbacks() {
        let dir = temp_git_repo();
        let repo = dir.path().display().to_string();
        // typo'd tag: explicit range must error instead of silently widening
        let err = resolve_version_range(&repo, Some("v9.9.9"), Some("v1.42.1"), "").unwrap_err();
        assert!(err.contains("could not resolve"), "{err}");
        // swapped order resolves to an empty range: also an error
        let err = resolve_version_range(&repo, Some("v1.42.1"), Some("v1.41.0"), "").unwrap_err();
        assert!(err.contains("empty"), "{err}");
        // the valid explicit range resolves
        let ok = resolve_version_range(&repo, Some("v1.41.0"), Some("v1.42.1"), "")
            .unwrap()
            .expect("range");
        assert_eq!(ok.0, "v1.41.0");
        assert_eq!(ok.1, "v1.42.1");
        assert_eq!(ok.2.len(), 1);
        // auto-detection with no versions in the text degrades silently
        assert!(
            resolve_version_range(&repo, None, None, "no versions here")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn why_not_answers_for_range_excluded_commits_and_short_prefixes() {
        let (failure, failure_created) = failure_event();
        let good = "4444444444444444444444444444444444444444";
        let all = vec![
            node("fail", "crash in events/local.py", &[], 0),
            node(
                "c1",
                &commit_record_content(good),
                &[git_records::COMMIT_TAG],
                5,
            ),
        ];
        let (candidates, excluded, _) =
            build_candidates(&failure, failure_created, &all, &Default::default(), None);
        let result = RetroactiveBackfill::new().run_trail(&failure, &candidates, &excluded);
        let contents: std::collections::HashMap<String, String> = all
            .iter()
            .map(|n| (n.id.clone(), n.content.clone()))
            .collect();

        // a range-excluded commit answers from the excluded list, not "not a candidate"
        let excluded_c = vec![ExcludedCandidate {
            candidate: candidates[0].clone(),
            reason: "outside version range 1.41.0..1.42.1".into(),
        }];
        let out = explain_why_not(
            "4444444",
            &failure,
            &result,
            &[],
            &excluded_c,
            30,
            &contents,
        );
        assert_eq!(out["verdict"], "rejected");
        assert_eq!(out["detail"], "outside version range 1.41.0..1.42.1");

        // hex prefixes shorter than 7 chars are refused, not guessed
        let out = explain_why_not("444", &failure, &result, &candidates, &[], 30, &contents);
        assert_eq!(out["verdict"], "ambiguous");
    }

    #[test]
    fn why_not_answers_for_every_disposition() {
        let (failure, failure_created) = failure_event();
        let newer = node("later", "touched local.py yesterday", &[], -1);
        let all = vec![node("fail", "crash in events/local.py", &[], 0), newer];
        let (candidates, _excluded, _) =
            build_candidates(&failure, failure_created, &all, &Default::default(), None);
        let result = RetroactiveBackfill::new().run_trail(&failure, &candidates, &[]);
        let contents: std::collections::HashMap<String, String> = all
            .iter()
            .map(|n| (n.id.clone(), n.content.clone()))
            .collect();

        // a rejected record gets its rule
        let out = explain_why_not("later", &failure, &result, &candidates, &[], 30, &contents);
        assert_eq!(out["verdict"], "rejected");
        assert_eq!(out["detail"], "record is newer than the failure");
        // an unknown id says so
        let out = explain_why_not("nope", &failure, &result, &candidates, &[], 30, &contents);
        assert_eq!(out["verdict"], "not a candidate");
        // a commit sha prefix resolves to its record
        let sha = "9999998888777766665555444433332222221111";
        let all2 = vec![
            node("fail", "crash in events/local.py", &[], 0),
            node(
                "c1",
                &commit_record_content(sha),
                &[git_records::COMMIT_TAG],
                4,
            ),
        ];
        let (candidates2, _, _) =
            build_candidates(&failure, failure_created, &all2, &Default::default(), None);
        let result2 = RetroactiveBackfill::new().run_trail(&failure, &candidates2, &[]);
        let contents2: std::collections::HashMap<String, String> = all2
            .iter()
            .map(|n| (n.id.clone(), n.content.clone()))
            .collect();
        let out = explain_why_not(
            "9999998",
            &failure,
            &result2,
            &candidates2,
            &[],
            30,
            &contents2,
        );
        assert_eq!(out["verdict"], "surfaced", "{out}");
    }
}
