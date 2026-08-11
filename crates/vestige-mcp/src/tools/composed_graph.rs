//! composed_graph tool — durable composition history and bounty-mode lane queue.

use chrono::Utc;
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use uuid::Uuid;
use vestige_core::{CompositionOutcomeRecord, Storage};

pub(crate) const OUTCOME_TYPES: &[&str] = &[
    "helpful",
    "dead_end",
    "submitted",
    "accepted",
    "rejected",
    "duplicate_risk",
    "needs_poc",
    "bad_severity",
    "user_promoted",
    "user_demoted",
    "closed_by_scope",
    "closed_by_duplicate",
    "closed_by_false_assumption",
    "closed_by_user",
    "expired_lane",
];

pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["recent", "get", "memory", "neighbors", "never_composed", "bounty_mode", "label"],
                "description": "ComposedGraph action to run."
            },
            "event_id": {
                "type": "string",
                "description": "Composition event id for get/label actions."
            },
            "memory_id": {
                "type": "string",
                "description": "Memory id for memory/neighbors actions."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum rows to return (default 10, max 100).",
                "default": 10,
                "minimum": 1,
                "maximum": 100
            },
            "tags": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Optional tag filter for never_composed and bounty_mode."
            },
            "outcome_type": {
                "type": "string",
                "enum": ["helpful", "dead_end", "submitted", "accepted", "rejected", "duplicate_risk", "needs_poc", "bad_severity", "user_promoted", "user_demoted", "closed_by_scope", "closed_by_duplicate", "closed_by_false_assumption", "closed_by_user", "expired_lane"],
                "description": "Outcome label for label action."
            },
            "notes": {
                "type": "string",
                "description": "Optional outcome notes."
            },
            "label_source": {
                "type": "string",
                "description": "Where the outcome label came from (default: user)."
            },
            "confidence_delta": {
                "type": "number",
                "description": "Optional confidence adjustment for this outcome."
            }
        },
        "required": ["action"]
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct ComposedGraphArgs {
    action: String,
    event_id: Option<String>,
    memory_id: Option<String>,
    limit: Option<i32>,
    tags: Option<Vec<String>>,
    outcome_type: Option<String>,
    notes: Option<String>,
    label_source: Option<String>,
    confidence_delta: Option<f64>,
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: ComposedGraphArgs = match args {
        Some(value) => {
            serde_json::from_value(value).map_err(|e| format!("Invalid arguments: {}", e))?
        }
        None => return Err("Missing arguments".to_string()),
    };
    let limit = args.limit.unwrap_or(10).clamp(1, 100);

    match args.action.as_str() {
        "recent" => recent(storage, limit),
        "get" => {
            let event_id = args
                .event_id
                .as_deref()
                .ok_or_else(|| "event_id is required for get".to_string())?;
            get(storage, event_id)
        }
        "memory" => {
            let memory_id = args
                .memory_id
                .as_deref()
                .ok_or_else(|| "memory_id is required for memory".to_string())?;
            memory(storage, memory_id, limit)
        }
        "neighbors" => {
            let memory_id = args
                .memory_id
                .as_deref()
                .ok_or_else(|| "memory_id is required for neighbors".to_string())?;
            neighbors(storage, memory_id, limit)
        }
        "never_composed" => never_composed(storage, limit, args.tags.as_deref()),
        "bounty_mode" => bounty_mode(storage, limit, args.tags.as_deref()),
        "label" => label(storage, &args),
        other => Err(format!("Unknown composed_graph action: {}", other)),
    }
}

fn recent(storage: &Storage, limit: i32) -> Result<Value, String> {
    let events = storage
        .get_recent_composition_events(limit)
        .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "action": "recent",
        "events": events,
    }))
}

fn get(storage: &Storage, event_id: &str) -> Result<Value, String> {
    let event = storage
        .get_composition_event(event_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("composition event not found: {}", event_id))?;
    let members = storage
        .get_composition_members(event_id)
        .map_err(|e| e.to_string())?;
    let outcomes = storage
        .get_composition_outcomes(event_id)
        .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "action": "get",
        "event": event,
        "members": members,
        "outcomes": outcomes,
    }))
}

fn memory(storage: &Storage, memory_id: &str, limit: i32) -> Result<Value, String> {
    let events = storage
        .get_compositions_for_memory(memory_id, limit)
        .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "action": "memory",
        "memoryId": memory_id,
        "events": events,
    }))
}

fn neighbors(storage: &Storage, memory_id: &str, limit: i32) -> Result<Value, String> {
    let neighbors = storage
        .get_composition_neighbors(memory_id, limit)
        .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "action": "neighbors",
        "memoryId": memory_id,
        "neighbors": neighbors,
    }))
}

fn never_composed(storage: &Storage, limit: i32, tags: Option<&[String]>) -> Result<Value, String> {
    let candidates = storage
        .get_never_composed_candidates(limit, tags)
        .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "action": "never_composed",
        "candidates": candidates,
    }))
}

fn bounty_mode(storage: &Storage, limit: i32, tags: Option<&[String]>) -> Result<Value, String> {
    const PAGE_SIZE: i32 = 100;
    const MAX_SCAN_EVENTS: i32 = 1_000;

    let mut offset = 0;
    let mut scanned = 0;
    let mut already_composed = Vec::new();
    let mut closed_doors = Vec::new();
    let mut duplicate_risk_lanes = Vec::new();
    let mut needs_poc_lanes = Vec::new();

    loop {
        let events = storage
            .get_recent_composition_events_page(PAGE_SIZE, offset)
            .map_err(|e| e.to_string())?;
        if events.is_empty() {
            break;
        }
        scanned += events.len() as i32;

        for event in events {
            let outcomes = storage
                .get_composition_outcomes(&event.id)
                .map_err(|e| e.to_string())?;
            let members = storage
                .get_composition_members(&event.id)
                .map_err(|e| e.to_string())?;
            if !composition_matches_tags(storage, &event, &members, tags)? {
                continue;
            }
            let item = serde_json::json!({
                "event": event,
                "members": members,
                "outcomes": outcomes,
            });
            let outcome_types = item["outcomes"]
                .as_array()
                .map(|values| {
                    values
                        .iter()
                        .filter_map(|value| value.get("outcomeType").and_then(|v| v.as_str()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            if outcome_types.iter().any(|kind| {
                matches!(
                    *kind,
                    "dead_end"
                        | "rejected"
                        | "bad_severity"
                        | "closed_by_scope"
                        | "closed_by_duplicate"
                        | "closed_by_false_assumption"
                        | "closed_by_user"
                        | "expired_lane"
                )
            }) {
                push_limited(&mut closed_doors, item.clone(), limit);
            }
            if outcome_types
                .iter()
                .any(|kind| matches!(*kind, "duplicate_risk" | "closed_by_duplicate"))
            {
                push_limited(&mut duplicate_risk_lanes, item.clone(), limit);
            }
            if outcome_types.contains(&"needs_poc") {
                push_limited(&mut needs_poc_lanes, item.clone(), limit);
            }
            if already_composed.len() < limit as usize {
                already_composed.push(item);
            }
            if bounty_mode_lanes_full(
                limit,
                &already_composed,
                &closed_doors,
                &duplicate_risk_lanes,
                &needs_poc_lanes,
            ) {
                break;
            }
        }

        if bounty_mode_lanes_full(
            limit,
            &already_composed,
            &closed_doors,
            &duplicate_risk_lanes,
            &needs_poc_lanes,
        ) || scanned >= MAX_SCAN_EVENTS
        {
            break;
        }
        offset += PAGE_SIZE;
    }

    let never = storage
        .get_never_composed_candidates(limit, tags)
        .map_err(|e| e.to_string())?;
    let top_weird_combinations = never.iter().take(3).cloned().collect::<Vec<_>>();

    Ok(serde_json::json!({
        "action": "bounty_mode",
        "alreadyComposedLanes": already_composed,
        "neverComposedLanes": never,
        "closedDoors": closed_doors,
        "duplicateRiskLanes": duplicate_risk_lanes,
        "needsPocLanes": needs_poc_lanes,
        "topWeirdCombinations": top_weird_combinations,
        "guardrails": [
            "never-composed lane is not a finding",
            "composition score is not severity",
            "submit/reportable still needs source refs, scope fit, and PoC evidence"
        ]
    }))
}

fn push_limited(items: &mut Vec<Value>, item: Value, limit: i32) {
    if items.len() < limit as usize {
        items.push(item);
    }
}

fn bounty_mode_lanes_full(
    limit: i32,
    already_composed: &[Value],
    closed_doors: &[Value],
    duplicate_risk_lanes: &[Value],
    needs_poc_lanes: &[Value],
) -> bool {
    let limit = limit as usize;
    already_composed.len() >= limit
        && closed_doors.len() >= limit
        && duplicate_risk_lanes.len() >= limit
        && needs_poc_lanes.len() >= limit
}

fn composition_matches_tags(
    storage: &Storage,
    event: &vestige_core::CompositionEventRecord,
    members: &[vestige_core::CompositionMemberRecord],
    tags: Option<&[String]>,
) -> Result<bool, String> {
    let Some(tags) = tags else {
        return Ok(true);
    };
    if tags.is_empty() {
        return Ok(true);
    }

    if json_value_has_tag(&event.metadata, tags) {
        return Ok(true);
    }

    for member in members {
        if json_value_has_tag(&member.metadata, tags) {
            return Ok(true);
        }
        if let Some(node) = storage
            .get_node(&member.memory_id)
            .map_err(|e| e.to_string())?
            && node.tags.iter().any(|tag| tag_matches_filter(tag, tags))
        {
            return Ok(true);
        }
    }

    Ok(false)
}

fn json_value_has_tag(value: &Value, tags: &[String]) -> bool {
    value
        .get("tags")
        .and_then(|tags_value| tags_value.as_array())
        .is_some_and(|values| {
            values.iter().any(|value| {
                value
                    .as_str()
                    .is_some_and(|tag| tag_matches_filter(tag, tags))
            })
        })
}

fn tag_matches_filter(tag: &str, filters: &[String]) -> bool {
    filters
        .iter()
        .any(|wanted| tag == wanted || tag.starts_with(&format!("{wanted}:")))
}

fn label(storage: &Storage, args: &ComposedGraphArgs) -> Result<Value, String> {
    let event_id = args
        .event_id
        .as_deref()
        .ok_or_else(|| "event_id is required for label".to_string())?;
    let outcome_type = args
        .outcome_type
        .as_deref()
        .ok_or_else(|| "outcome_type is required for label".to_string())?;
    if !OUTCOME_TYPES.contains(&outcome_type) {
        return Err(format!("unsupported outcome_type: {}", outcome_type));
    }
    if storage
        .get_composition_event(event_id)
        .map_err(|e| e.to_string())?
        .is_none()
    {
        return Err(format!("composition event not found: {}", event_id));
    }

    let outcome = CompositionOutcomeRecord {
        id: Uuid::new_v4().to_string(),
        event_id: event_id.to_string(),
        outcome_type: outcome_type.to_string(),
        labeled_at: Utc::now(),
        label_source: args
            .label_source
            .clone()
            .unwrap_or_else(|| "user".to_string()),
        confidence_delta: args.confidence_delta,
        notes: args.notes.clone(),
        metadata: serde_json::json!({}),
    };
    storage
        .record_composition_outcome(&outcome)
        .map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "action": "label",
        "eventId": event_id,
        "outcome": outcome,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use vestige_core::{
        CompositionEventRecord, CompositionMemberRecord, CompositionOutcomeRecord, IngestInput,
    };

    fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    fn ingest(storage: &Storage, content: &str, tags: &[&str]) -> String {
        storage
            .ingest(IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                tags: tags.iter().map(|tag| tag.to_string()).collect(),
                ..Default::default()
            })
            .unwrap()
            .id
    }

    #[tokio::test]
    async fn test_composed_graph_get_label_and_bounty_mode() {
        let (storage, _dir) = test_storage();
        let first = ingest(
            &storage,
            "Oracle drift bounty lane",
            &["protocolgate", "boundary-oracle", "settlement"],
        );
        let second = ingest(
            &storage,
            "Withdrawal queue bounty lane",
            &["protocolgate", "boundary-queue", "settlement"],
        );
        let third = ingest(
            &storage,
            "Keeper role bounty lane",
            &["protocolgate", "boundary-role", "settlement"],
        );

        let event = CompositionEventRecord {
            id: "composed-graph-test".to_string(),
            created_at: Utc::now(),
            tool: "deep_reference".to_string(),
            mode: "bounty".to_string(),
            query: Some("oracle withdrawal".to_string()),
            query_hash: Some("test".to_string()),
            confidence: Some(0.8),
            status: Some("resolved".to_string()),
            output_preview: Some("compose oracle and withdrawal queue".to_string()),
            metadata: serde_json::json!({}),
        };
        storage
            .save_composition(
                &event,
                &[
                    CompositionMemberRecord {
                        event_id: event.id.clone(),
                        memory_id: first.clone(),
                        role: "primary".to_string(),
                        rank: 0,
                        trust: Some(0.8),
                        score: Some(0.9),
                        preview: None,
                        metadata: serde_json::json!({}),
                    },
                    CompositionMemberRecord {
                        event_id: event.id.clone(),
                        memory_id: second.clone(),
                        role: "supporting".to_string(),
                        rank: 1,
                        trust: Some(0.7),
                        score: Some(0.8),
                        preview: None,
                        metadata: serde_json::json!({}),
                    },
                ],
                &[],
            )
            .unwrap();

        let unrelated = ingest(&storage, "Personal planning lane", &["personal"]);
        storage
            .save_composition(
                &CompositionEventRecord {
                    id: "unrelated-composed-graph-test".to_string(),
                    created_at: Utc::now() + chrono::Duration::seconds(10),
                    tool: "deep_reference".to_string(),
                    mode: "planning".to_string(),
                    query: Some("personal planning".to_string()),
                    query_hash: Some("unrelated".to_string()),
                    confidence: Some(0.4),
                    status: Some("resolved".to_string()),
                    output_preview: Some("unrelated composition".to_string()),
                    metadata: serde_json::json!({}),
                },
                &[CompositionMemberRecord {
                    event_id: "unrelated-composed-graph-test".to_string(),
                    memory_id: unrelated,
                    role: "primary".to_string(),
                    rank: 0,
                    trust: Some(0.4),
                    score: Some(0.2),
                    preview: None,
                    metadata: serde_json::json!({}),
                }],
                &[CompositionOutcomeRecord {
                    id: "unrelated-composed-graph-outcome".to_string(),
                    event_id: "unrelated-composed-graph-test".to_string(),
                    outcome_type: "needs_poc".to_string(),
                    labeled_at: Utc::now(),
                    label_source: "test".to_string(),
                    confidence_delta: None,
                    notes: None,
                    metadata: serde_json::json!({}),
                }],
            )
            .unwrap();

        let get_result = execute(
            &storage,
            Some(serde_json::json!({
                "action": "get",
                "event_id": event.id
            })),
        )
        .await
        .unwrap();
        assert_eq!(get_result["members"].as_array().unwrap().len(), 2);

        let label_result = execute(
            &storage,
            Some(serde_json::json!({
                "action": "label",
                "event_id": "composed-graph-test",
                "outcome_type": "submitted",
                "notes": "submitted in test"
            })),
        )
        .await
        .unwrap();
        assert_eq!(
            label_result["outcome"]["outcomeType"].as_str(),
            Some("submitted")
        );
        let closed_label_result = execute(
            &storage,
            Some(serde_json::json!({
                "action": "label",
                "event_id": "composed-graph-test",
                "outcome_type": "closed_by_scope",
                "notes": "closed in test"
            })),
        )
        .await
        .unwrap();
        assert_eq!(
            closed_label_result["outcome"]["outcomeType"].as_str(),
            Some("closed_by_scope")
        );
        let duplicate_label_result = execute(
            &storage,
            Some(serde_json::json!({
                "action": "label",
                "event_id": "composed-graph-test",
                "outcome_type": "closed_by_duplicate",
                "notes": "duplicate family in test"
            })),
        )
        .await
        .unwrap();
        assert_eq!(
            duplicate_label_result["outcome"]["outcomeType"].as_str(),
            Some("closed_by_duplicate")
        );

        let bounty = execute(
            &storage,
            Some(serde_json::json!({
                "action": "bounty_mode",
                "tags": ["protocolgate"],
                "limit": 1
            })),
        )
        .await
        .unwrap();
        let already = bounty["alreadyComposedLanes"].as_array().unwrap();
        assert_eq!(already.len(), 1);
        assert!(
            already[0]["event"]["id"].as_str() == Some("composed-graph-test"),
            "tag-scoped bounty_mode should skip newer unrelated events before truncating"
        );
        assert_eq!(bounty["closedDoors"].as_array().unwrap().len(), 1);
        assert_eq!(bounty["duplicateRiskLanes"].as_array().unwrap().len(), 1);
        assert!(bounty["needsPocLanes"].as_array().unwrap().is_empty());
        assert!(
            bounty["neverComposedLanes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|candidate| {
                    let first_id = candidate["firstId"].as_str().unwrap_or_default();
                    let second_id = candidate["secondId"].as_str().unwrap_or_default();
                    [first_id, second_id].contains(&third.as_str())
                })
        );
    }

    #[tokio::test]
    async fn test_bounty_mode_paginates_tag_filter_and_matches_namespaced_tags() {
        let (storage, _dir) = test_storage();
        let tagged = ingest(
            &storage,
            "Older tagged composition lane",
            &["project:vestige", "composition"],
        );
        let unrelated = ingest(&storage, "Newer unrelated lane", &["unrelated"]);
        let base_time = Utc::now();

        storage
            .save_composition(
                &CompositionEventRecord {
                    id: "older-tagged-composition".to_string(),
                    created_at: base_time,
                    tool: "deep_reference".to_string(),
                    mode: "research".to_string(),
                    query: Some("older tagged lane".to_string()),
                    query_hash: Some("fnv1a64:older".to_string()),
                    confidence: Some(0.8),
                    status: Some("resolved".to_string()),
                    output_preview: None,
                    metadata: serde_json::json!({}),
                },
                &[CompositionMemberRecord {
                    event_id: "older-tagged-composition".to_string(),
                    memory_id: tagged,
                    role: "primary".to_string(),
                    rank: 0,
                    trust: Some(0.8),
                    score: Some(0.9),
                    preview: None,
                    metadata: serde_json::json!({}),
                }],
                &[],
            )
            .unwrap();

        for idx in 0..101 {
            let event_id = format!("newer-unrelated-composition-{idx}");
            storage
                .save_composition(
                    &CompositionEventRecord {
                        id: event_id.clone(),
                        created_at: base_time + chrono::Duration::seconds(i64::from(idx + 1)),
                        tool: "deep_reference".to_string(),
                        mode: "planning".to_string(),
                        query: Some(format!("newer unrelated lane {idx}")),
                        query_hash: Some(format!("fnv1a64:newer-{idx}")),
                        confidence: Some(0.3),
                        status: Some("resolved".to_string()),
                        output_preview: None,
                        metadata: serde_json::json!({}),
                    },
                    &[CompositionMemberRecord {
                        event_id,
                        memory_id: unrelated.clone(),
                        role: "primary".to_string(),
                        rank: 0,
                        trust: Some(0.3),
                        score: Some(0.2),
                        preview: None,
                        metadata: serde_json::json!({}),
                    }],
                    &[],
                )
                .unwrap();
        }

        let bounty = execute(
            &storage,
            Some(serde_json::json!({
                "action": "bounty_mode",
                "tags": ["project"],
                "limit": 1
            })),
        )
        .await
        .unwrap();
        let already = bounty["alreadyComposedLanes"].as_array().unwrap();
        assert_eq!(already.len(), 1);
        assert_eq!(
            already[0]["event"]["id"].as_str(),
            Some("older-tagged-composition"),
            "tag-filtered bounty_mode should page past newer unrelated events and match namespaced tags"
        );
    }

    #[tokio::test]
    async fn test_purge_removes_composition_event_that_retains_member_data() {
        let (storage, _dir) = test_storage();
        let tagged = ingest(
            &storage,
            "Tagged member that will be purged",
            &["project:vestige", "composition"],
        );

        storage
            .save_composition(
                &CompositionEventRecord {
                    id: "purged-tagged-member-composition".to_string(),
                    created_at: Utc::now(),
                    tool: "deep_reference".to_string(),
                    mode: "research".to_string(),
                    query: Some("purged tagged lane".to_string()),
                    query_hash: Some("fnv1a64:purged".to_string()),
                    confidence: Some(0.6),
                    status: Some("closed".to_string()),
                    output_preview: None,
                    metadata: serde_json::json!({}),
                },
                &[CompositionMemberRecord {
                    event_id: "purged-tagged-member-composition".to_string(),
                    memory_id: tagged.clone(),
                    role: "primary".to_string(),
                    rank: 0,
                    trust: Some(0.7),
                    score: Some(0.8),
                    preview: Some("Tagged member that will be purged".to_string()),
                    metadata: serde_json::json!({}),
                }],
                &[CompositionOutcomeRecord {
                    id: "purged-tagged-member-outcome".to_string(),
                    event_id: "purged-tagged-member-composition".to_string(),
                    outcome_type: "closed_by_scope".to_string(),
                    labeled_at: Utc::now(),
                    label_source: "test".to_string(),
                    confidence_delta: Some(-0.2),
                    notes: None,
                    metadata: serde_json::json!({}),
                }],
            )
            .unwrap();

        storage
            .purge_node(&tagged, Some("test purge"))
            .expect("purge should succeed");

        let get_error = execute(
            &storage,
            Some(serde_json::json!({
                "action": "get",
                "event_id": "purged-tagged-member-composition"
            })),
        )
        .await
        .unwrap_err();
        assert!(
            get_error.contains("composition event not found"),
            "purge must remove a composed event rather than leave a source-data-bearing projection"
        );

        let bounty = execute(
            &storage,
            Some(serde_json::json!({
                "action": "bounty_mode",
                "tags": ["project"],
                "limit": 1
            })),
        )
        .await
        .unwrap();
        let already = bounty["alreadyComposedLanes"].as_array().unwrap();
        assert!(
            already.is_empty(),
            "bounty mode must not surface a composition event that retained the purged member"
        );
        assert!(bounty["closedDoors"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_bounty_mode_guardrail_buckets_are_not_truncated_by_already_limit() {
        let (storage, _dir) = test_storage();
        let neutral = ingest(&storage, "Neutral release lane", &["project:vestige"]);
        let closed = ingest(&storage, "Closed release lane", &["project:vestige"]);
        let base_time = Utc::now();

        storage
            .save_composition(
                &CompositionEventRecord {
                    id: "older-closed-lane".to_string(),
                    created_at: base_time,
                    tool: "deep_reference".to_string(),
                    mode: "release".to_string(),
                    query: Some("older closed lane".to_string()),
                    query_hash: Some("fnv1a64:older-closed".to_string()),
                    confidence: Some(0.3),
                    status: Some("closed".to_string()),
                    output_preview: None,
                    metadata: serde_json::json!({}),
                },
                &[CompositionMemberRecord {
                    event_id: "older-closed-lane".to_string(),
                    memory_id: closed,
                    role: "primary".to_string(),
                    rank: 0,
                    trust: Some(0.5),
                    score: Some(0.4),
                    preview: None,
                    metadata: serde_json::json!({}),
                }],
                &[CompositionOutcomeRecord {
                    id: "older-closed-outcome".to_string(),
                    event_id: "older-closed-lane".to_string(),
                    outcome_type: "closed_by_false_assumption".to_string(),
                    labeled_at: base_time,
                    label_source: "test".to_string(),
                    confidence_delta: Some(-0.3),
                    notes: None,
                    metadata: serde_json::json!({}),
                }],
            )
            .unwrap();

        storage
            .save_composition(
                &CompositionEventRecord {
                    id: "newer-neutral-lane".to_string(),
                    created_at: base_time + chrono::Duration::seconds(1),
                    tool: "deep_reference".to_string(),
                    mode: "release".to_string(),
                    query: Some("newer neutral lane".to_string()),
                    query_hash: Some("fnv1a64:newer-neutral".to_string()),
                    confidence: Some(0.7),
                    status: Some("resolved".to_string()),
                    output_preview: None,
                    metadata: serde_json::json!({}),
                },
                &[CompositionMemberRecord {
                    event_id: "newer-neutral-lane".to_string(),
                    memory_id: neutral,
                    role: "primary".to_string(),
                    rank: 0,
                    trust: Some(0.8),
                    score: Some(0.8),
                    preview: None,
                    metadata: serde_json::json!({}),
                }],
                &[],
            )
            .unwrap();

        let bounty = execute(
            &storage,
            Some(serde_json::json!({
                "action": "bounty_mode",
                "tags": ["project"],
                "limit": 1
            })),
        )
        .await
        .unwrap();

        assert_eq!(
            bounty["alreadyComposedLanes"][0]["event"]["id"].as_str(),
            Some("newer-neutral-lane")
        );
        assert_eq!(
            bounty["closedDoors"][0]["event"]["id"].as_str(),
            Some("older-closed-lane"),
            "guardrail buckets should keep scanning after alreadyComposedLanes reaches limit"
        );
    }
}
