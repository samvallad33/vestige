//! Retroactive Salience Backfill: bounded hindsight over exact recorded identities.
//!
//! `causal_walk` remains the investigation tool. This planner reuses its recorded
//! backward BFS, then considers direct exact-entity predecessors without declaring
//! those co-mentions to be causes. Every score change has an admitted receipt.

use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::sync::Arc;
use vestige_core::advanced::retroactive_backfill::{
    self, SalienceChange, SalienceMemory, plan_salience,
};
use vestige_core::{KnowledgeNode, Storage};

pub fn schema() -> Value {
    json!({"type":"object","properties":{
        "action":{"type":"string","enum":["run","undo","receipts"],"default":"run"},
        "trigger_id":{"type":"string","description":"Exact newly saved memory id. failure_id is a legacy alias."},
        "failure_id":{"type":"string"},
        "receipt_id":{"type":"string","description":"Exact salience receipt for undo."},
        "scope":{"type":"string","default":"user"},
        "manual":{"type":"boolean","default":false},
        "dry_run":{"type":"boolean","default":false,"description":"Preview only; append nothing, including no tool trace."},
        "lookback_days":{"type":"integer","minimum":1,"maximum":365,"default":30},
        "scan_limit":{"type":"integer","minimum":1,"maximum":500,"default":500}
    }})
}

#[derive(Default, Deserialize)]
struct Args {
    action: Option<String>,
    trigger_id: Option<String>,
    failure_id: Option<String>,
    receipt_id: Option<String>,
    scope: Option<String>,
    #[serde(default)]
    manual: bool,
    #[serde(default)]
    dry_run: bool,
    lookback_days: Option<i64>,
    scan_limit: Option<usize>,
    // Old clients explicitly requested promote=false for a preview.
    promote: Option<bool>,
}

pub fn looks_like_failure(node: &KnowledgeNode) -> bool {
    retroactive_backfill::looks_like_failure(&node.content, &node.tags)
}

pub(crate) fn autofire_enabled() -> bool {
    !std::env::var("VESTIGE_BACKFILL_AUTOFIRE")
        .ok()
        .is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
}

fn salience_view(node: &KnowledgeNode) -> SalienceMemory {
    SalienceMemory {
        id: node.id.clone(),
        content: node.content.clone(),
        tags: retroactive_backfill::salience_metadata_tags(&node.tags),
        created_at_ms: node.created_at.timestamp_millis(),
        stability: node.stability,
    }
}

/// Cheap mirror of the planner's trigger bounds, before graph/store scans.
/// Retrieval indexing retains every original tag.
pub(crate) fn salience_eligible(node: &KnowledgeNode) -> bool {
    let tags = retroactive_backfill::salience_metadata_tags(&node.tags);
    node.content.len() <= retroactive_backfill::MAX_SALIENCE_CONTENT_BYTES
        && tags.len() <= retroactive_backfill::MAX_SALIENCE_TAGS_PER_MEMORY
        && tags.iter().all(|tag| tag.len() <= 512)
}

fn preview(change: &SalienceChange) -> Value {
    json!({"memory_id":change.memory_id,"trigger_id":change.trigger_id,
        "shared_entity":change.shared_entity,"reason":change.reason,
        "old_score":change.old_stability,"new_score":change.new_stability,
        "direction":if change.new_stability > change.old_stability {"raise"} else {"lower"},
        "receipt_id":Value::Null})
}

fn run(storage: &Storage, args: &Args, trigger_id: &str, scope: &str) -> Result<Value, String> {
    if !storage
        .node_is_in_scope(trigger_id, scope)
        .map_err(|e| e.to_string())?
    {
        return Err("trigger_id does not name a live memory in this scope".into());
    }
    let trigger = storage
        .get_node(trigger_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "trigger memory not found".to_string())?;
    let limit = args.scan_limit.unwrap_or(500);
    if !(1..=500).contains(&limit) {
        return Err("scan_limit must be between 1 and 500".into());
    }
    let lookback = args.lookback_days.unwrap_or(30).clamp(1, 365);
    let manual_eligible = args.manual || looks_like_failure(&trigger);
    let dry_run = args.dry_run || args.promote == Some(false);
    // Every save evaluates eligibility, but an ordinary unrelated save need
    // not walk or clone the entire log merely to produce an empty plan.
    if !manual_eligible && !retroactive_backfill::has_salient_personal_attribute(&trigger.tags) {
        return Ok(
            json!({"trigger_id":trigger_id,"scope":scope,"dry_run":dry_run,
            "changes":[],"changed":0,"would_change":0,"scanned":0,"truncated":false,
            "reason":"no salient personal attribute or recorded failure","causality_verified":false}),
        );
    }
    if !salience_eligible(&trigger) {
        return Ok(
            json!({"trigger_id":trigger_id,"scope":scope,"dry_run":dry_run,
            "changes":[],"changed":0,"would_change":0,"scanned":0,"truncated":false,
            "reason":"trigger exceeds native salience bounds","causality_verified":false}),
        );
    }
    let receipts = storage.salience_receipts().map_err(|e| e.to_string())?;
    let used: HashSet<String> = receipts
        .iter()
        .filter(|row| row["trigger_id"] == trigger_id)
        .filter_map(|row| row["memory_id"].as_str().map(str::to_owned))
        .collect();
    // The same BFS as causal_walk, including its one-touch and depth bounds.
    let walked: HashSet<String> =
        super::causal_walk::salience_candidate_ids(storage, scope, trigger_id, limit)?
            .into_iter()
            .collect();
    let mut predecessors = storage
        .get_all_nodes_in_scope(scope, i32::MAX, 0)
        .map_err(|e| e.to_string())?;
    let identities: HashSet<&str> = trigger
        .tags
        .iter()
        .map(String::as_str)
        .filter(|tag| {
            ["person:", "place:", "food:", "file:", "service:", "symbol:"]
                .iter()
                .any(|prefix| tag.starts_with(prefix))
        })
        .collect();
    predecessors.retain(|node| {
        let age = trigger.created_at.signed_duration_since(node.created_at);
        node.id != trigger_id
            && !used.contains(&node.id)
            && age.num_milliseconds() > 0
            && age.num_days() <= lookback
            && node
                .tags
                .iter()
                .any(|tag| identities.contains(tag.as_str()))
    });
    predecessors.sort_by(|a, b| {
        walked
            .contains(&b.id)
            .cmp(&walked.contains(&a.id))
            .then_with(|| b.created_at.cmp(&a.created_at))
            .then_with(|| a.id.cmp(&b.id))
    });
    let truncated = predecessors.len() > limit;
    predecessors.truncate(limit);
    let candidates: Vec<_> = predecessors.iter().map(salience_view).collect();
    // Validate candidates independently: one malformed old record cannot
    // silence valid neighbors. Consumed pairs were removed before this cap,
    // so a resumed pass progresses instead of spending its budget on them.
    let trigger_view = salience_view(&trigger);
    let changes: Vec<_> = candidates
        .iter()
        .flat_map(|candidate| {
            plan_salience(
                &trigger_view,
                std::slice::from_ref(candidate),
                manual_eligible,
            )
        })
        .take(retroactive_backfill::MAX_PLANNED_SALIENCE_CHANGES)
        .collect();
    let mut rows = Vec::new();
    for change in &changes {
        if dry_run {
            rows.push(preview(change));
        } else if let Some(receipt) = storage
            .apply_salience_change(change)
            .map_err(|e| e.to_string())?
        {
            rows.push(receipt);
        }
    }
    Ok(
        json!({"trigger_id":trigger_id,"scope":scope,"dry_run":dry_run,
        "changes":rows,"changed":if dry_run {0} else {rows.len()},
        "would_change":rows.len(),"scanned":candidates.len(),"truncated":truncated,
        "change_budget_reached":changes.len()==retroactive_backfill::MAX_PLANNED_SALIENCE_CHANGES,
        "causality_verified":false,"ranking_signal":"bounded_salience_from_exact_shared_entities"}),
    )
}

/// Central every-save/consolidation entry. Manual overrides ignore the optout;
/// neither entry may override chronology, exact identity or stability bounds.
pub(crate) fn autofire(storage: &Storage, trigger_id: &str, scope: &str) -> Result<(), String> {
    if !autofire_enabled() {
        return Ok(());
    }
    run(storage, &Args::default(), trigger_id, scope).map(|_| ())
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: Args = serde_json::from_value(args.unwrap_or_else(|| json!({})))
        .map_err(|e| format!("invalid backfill arguments: {e}"))?;
    let scope = args.scope.as_deref().unwrap_or("user").trim();
    if scope.is_empty() {
        return Err("scope must not be empty".into());
    }
    match args.action.as_deref().unwrap_or("run") {
        "run" => {
            if args.trigger_id.is_some()
                && args.failure_id.is_some()
                && args.trigger_id != args.failure_id
            {
                return Err("trigger_id and failure_id disagree".into());
            }
            let trigger = args
                .trigger_id
                .as_deref()
                .or(args.failure_id.as_deref())
                .filter(|s| !s.is_empty())
                .ok_or("trigger_id is required; Backfill never guesses the trigger")?;
            run(storage.as_ref(), &args, trigger, scope)
        }
        "receipts" => {
            let rows: Vec<_> = storage
                .salience_receipts()
                .map_err(|e| e.to_string())?
                .into_iter()
                .filter(|r| {
                    r["memory_id"]
                        .as_str()
                        .is_some_and(|id| storage.node_is_in_scope(id, scope).unwrap_or(false))
                })
                .collect();
            Ok(json!({"scope":scope,"receipts":rows}))
        }
        "undo" => {
            let id = args
                .receipt_id
                .as_deref()
                .ok_or("receipt_id is required for undo")?;
            let rows = storage.salience_receipts().map_err(|e| e.to_string())?;
            let receipt = rows
                .iter()
                .find(|r| r["receipt_id"] == id)
                .ok_or("salience receipt not found")?;
            let target = receipt["memory_id"]
                .as_str()
                .ok_or("invalid receipt target")?;
            if !storage
                .node_is_in_scope(target, scope)
                .map_err(|e| e.to_string())?
            {
                return Err("salience receipt is outside scope".into());
            }
            if args.dry_run {
                return Ok(json!({"dry_run":true,"would_undo":receipt,"changed":0}));
            }
            storage.undo_salience_change(id).map_err(|e| e.to_string())
        }
        _ => Err("unknown backfill action".into()),
    }
}
