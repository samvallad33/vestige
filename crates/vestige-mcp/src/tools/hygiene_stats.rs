//! Full-store, agent-facing memory hygiene statistics.
//!
//! Aggregates the metadata-only snapshot supplied by `vestige-core`. Aggregate
//! counts always cover every stored row in the selected scope, including rows
//! outside their validity window and superseded rows. Only the two detail lists
//! are bounded.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use vestige_core::{DEFAULT_MEMORY_SCOPE, HygieneNodeSummary, Storage, scan_secrets};

const DEFAULT_DETAIL_LIMIT: usize = 50;
const MAX_DETAIL_LIMIT: usize = 200;
const TAG_AUDIT_WINDOW: usize = 50;

#[derive(Debug, Clone)]
struct TagAuditSummary {
    operation_id: String,
    operation_type: String,
    status: String,
    reason: Option<String>,
    affected_count: usize,
    created_at: String,
    reverted_at: Option<String>,
    scope: Option<String>,
    all_scopes: bool,
    source_tags: Vec<String>,
    target_tag: Option<String>,
}

#[derive(Debug, Clone)]
struct TagAuditWindow {
    operations: Vec<TagAuditSummary>,
    scanned_operations: usize,
    truncated: bool,
}

/// Return complete hygiene aggregates and bounded diagnostic lists.
pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args = args.unwrap_or_else(|| json!({}));
    let limit = parse_limit(&args)?;
    let all_scopes = parse_bool(&args, "all_scopes")?.unwrap_or(false);
    let requested_scope = parse_string(&args, "scope")?.map(str::trim);
    let selected_scope = if all_scopes {
        None
    } else {
        Some(match requested_scope {
            Some("") | None => DEFAULT_MEMORY_SCOPE,
            Some(scope) => scope,
        })
    };

    let nodes = storage
        .hygiene_snapshot(selected_scope)
        .map_err(|error| format!("Failed to build hygiene snapshot: {error}"))?;
    let tag_audit = load_tag_audit(storage, selected_scope)?;

    Ok(build_response(
        nodes,
        selected_scope,
        limit,
        Utc::now(),
        tag_audit,
    ))
}

fn parse_limit(args: &Value) -> Result<usize, String> {
    let Some(value) = args.get("limit") else {
        return Ok(DEFAULT_DETAIL_LIMIT);
    };
    let Some(limit) = value.as_u64() else {
        return Err("limit must be a positive integer".to_string());
    };
    if limit == 0 {
        return Err("limit must be at least 1".to_string());
    }
    Ok((limit as usize).min(MAX_DETAIL_LIMIT))
}

fn parse_bool(args: &Value, field: &str) -> Result<Option<bool>, String> {
    match args.get(field) {
        Some(value) => value
            .as_bool()
            .map(Some)
            .ok_or_else(|| format!("{field} must be a boolean")),
        None => Ok(None),
    }
}

fn parse_string<'a>(args: &'a Value, field: &str) -> Result<Option<&'a str>, String> {
    match args.get(field) {
        Some(value) => value
            .as_str()
            .map(Some)
            .ok_or_else(|| format!("{field} must be a string")),
        None => Ok(None),
    }
}

fn load_tag_audit(
    storage: &Arc<Storage>,
    selected_scope: Option<&str>,
) -> Result<TagAuditWindow, String> {
    // Read tag operations directly (rather than filtering a general reflog
    // window) so unrelated merge activity cannot bury durable tag audits.
    let mut reflog = storage
        .list_tag_operations(TAG_AUDIT_WINDOW + 1, selected_scope)
        .map_err(|error| format!("Failed to read tag-operation audit log: {error}"))?;
    let truncated = reflog.len() > TAG_AUDIT_WINDOW;
    reflog.truncate(TAG_AUDIT_WINDOW);
    let scanned_operations = reflog.len();
    let operations = reflog
        .into_iter()
        .map(|operation| {
            let source_tags = operation
                .signals
                .as_ref()
                .and_then(|signals| signals.get("sourceTags"))
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            let target_tag = operation
                .signals
                .as_ref()
                .and_then(|signals| signals.get("targetTag"))
                .and_then(Value::as_str)
                .map(str::to_string);
            let scope = operation
                .signals
                .as_ref()
                .and_then(|signals| signals.get("scope"))
                .and_then(Value::as_str)
                .map(str::to_string);
            let all_scopes = operation
                .signals
                .as_ref()
                .and_then(|signals| signals.get("allScopes"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            TagAuditSummary {
                operation_id: operation.id,
                operation_type: operation.op_type,
                status: operation.status,
                reason: operation.reason,
                affected_count: operation.affected_ids.len(),
                created_at: operation.created_at,
                reverted_at: operation.reverted_at,
                scope,
                all_scopes,
                source_tags,
                target_tag,
            }
        })
        .collect();

    Ok(TagAuditWindow {
        operations,
        scanned_operations,
        truncated,
    })
}

fn build_response(
    nodes: Vec<HygieneNodeSummary>,
    selected_scope: Option<&str>,
    limit: usize,
    now: DateTime<Utc>,
    tag_audit: TagAuditWindow,
) -> Value {
    let total = nodes.len();
    let mut type_counts = BTreeMap::<String, usize>::new();
    let mut tag_counts = BTreeMap::<String, usize>::new();
    let mut age_counts = age_buckets();
    let mut retention_counts = retention_buckets();
    let mut untagged = 0usize;
    let mut within_validity_window = 0usize;
    let mut not_yet_valid = 0usize;
    let mut expired = 0usize;
    let mut invalid_temporal_bounds = 0usize;
    let mut superseded = 0usize;

    for node in &nodes {
        *type_counts.entry(node.node_type.clone()).or_default() += 1;

        if node.tags.is_empty() {
            untagged += 1;
        }
        // A count means "memories carrying this exact tag", not raw duplicate
        // array entries in a malformed legacy row.
        for tag in node.tags.iter().collect::<BTreeSet<_>>() {
            *tag_counts.entry(tag.clone()).or_default() += 1;
        }

        let age_days = now.signed_duration_since(node.created_at).num_days();
        let age_bucket = if node.created_at > now {
            "futureDated"
        } else if age_days <= 7 {
            "0-7d"
        } else if age_days <= 30 {
            "8-30d"
        } else if age_days <= 90 {
            "31-90d"
        } else if age_days <= 180 {
            "91-180d"
        } else {
            "181d+"
        };
        *age_counts.get_mut(age_bucket).expect("fixed age bucket") += 1;

        let retention_bucket = retention_bucket(node.retention_strength);
        *retention_counts
            .get_mut(retention_bucket)
            .expect("fixed retention bucket") += 1;

        if node
            .valid_from
            .zip(node.valid_until)
            .is_some_and(|(from, until)| from > until)
        {
            invalid_temporal_bounds += 1;
        }
        if node.valid_until.is_some_and(|until| until < now) {
            expired += 1;
        } else if node.valid_from.is_some_and(|from| from > now) {
            not_yet_valid += 1;
        } else {
            within_validity_window += 1;
        }
        if node.superseded {
            superseded += 1;
        }
    }

    let mut never_accessed: Vec<&HygieneNodeSummary> =
        nodes.iter().filter(|node| node.never_accessed).collect();
    never_accessed.sort_by(|left, right| {
        left.created_at
            .cmp(&right.created_at)
            .then_with(|| left.id.cmp(&right.id))
    });
    let never_accessed_total = never_accessed.len();
    let never_accessed_items: Vec<Value> = never_accessed
        .into_iter()
        .take(limit)
        .map(detail_item)
        .collect();

    let mut largest: Vec<&HygieneNodeSummary> = nodes.iter().collect();
    largest.sort_by(|left, right| {
        right
            .content_bytes
            .cmp(&left.content_bytes)
            .then_with(|| left.id.cmp(&right.id))
    });
    let largest_items: Vec<Value> = largest.into_iter().take(limit).map(detail_item).collect();

    let tag_operations: Vec<Value> = tag_audit
        .operations
        .into_iter()
        .map(|operation| {
            json!({
                "operationId": operation.operation_id,
                "operationType": operation.operation_type,
                "status": operation.status,
                "reason": operation.reason,
                "affectedCount": operation.affected_count,
                "createdAt": operation.created_at,
                "revertedAt": operation.reverted_at,
                "scope": operation.scope,
                "allScopes": operation.all_scopes,
                "sourceTags": operation.source_tags,
                "targetTag": operation.target_tag,
            })
        })
        .collect();

    json!({
        "success": true,
        "view": "stats",
        "computedAt": now.to_rfc3339(),
        "scope": match selected_scope {
            Some(scope) => json!({ "mode": "single", "value": scope }),
            None => json!({ "mode": "all" }),
        },
        "semantics": {
            "population": "Every stored knowledge_nodes row in the selected scope, including expired, not-yet-valid, invalid-bound, and superseded rows.",
            "tagCounts": "Number of memories carrying each exact tag; duplicate entries within one memory count once.",
            "age": "Whole elapsed days from createdAt to computedAt. Future-created rows are reported separately.",
            "validity": "Temporal buckets are mutually exclusive: expired first, then not-yet-valid, then within-window. Superseded and invalid-bound counts are independent overlays.",
            "neverAccessed": "No durable memory_access_log row exists for the memory.",
            "contentSize": "UTF-8 bytes. Returned content is a bounded preview, never the full body. Blocking credential shapes are redacted without echoing the matched bytes."
        },
        "population": {
            "total": total,
            "includesInvalid": true,
            "includesSuperseded": true,
        },
        "counts": {
            "byMemoryType": type_counts,
            "byTag": tag_counts,
            "untagged": untagged,
            "byAge": age_counts,
        },
        "retentionDistribution": {
            "buckets": retention_counts,
        },
        "lifecycle": {
            "withinValidityWindow": within_validity_window,
            "notYetValid": not_yet_valid,
            "expired": expired,
            "invalidTemporalBounds": invalid_temporal_bounds,
            "superseded": superseded,
        },
        "neverAccessed": {
            "total": never_accessed_total,
            "returned": never_accessed_items.len(),
            "limit": limit,
            "truncated": never_accessed_total > never_accessed_items.len(),
            "memories": never_accessed_items,
        },
        "largestNodes": {
            "total": total,
            "returned": largest_items.len(),
            "limit": limit,
            "truncated": total > largest_items.len(),
            "memories": largest_items,
        },
        "recentTagOperations": {
            "source": "merge_operations memory reflog",
            "windowLimit": TAG_AUDIT_WINDOW,
            "scannedOperations": tag_audit.scanned_operations,
            "returned": tag_operations.len(),
            "truncated": tag_audit.truncated,
            "note": "Tag rename/merge operations are read directly from merge_operations so unrelated merge/supersede activity cannot bury them. Single-scope stats expose only operations recorded for that exact scope; all-scopes stats expose every tag scope. A reverted operation remains visible with status and revertedAt.",
            "operations": tag_operations,
        },
    })
}

fn detail_item(node: &HygieneNodeSummary) -> Value {
    let redacted = scan_secrets(&node.content_preview)
        .iter()
        .any(vestige_core::SecretFinding::blocks_ingestion);
    json!({
        "id": node.id,
        "memoryType": node.node_type,
        "createdAt": node.created_at.to_rfc3339(),
        "contentBytes": node.content_bytes,
        "contentPreview": if redacted {
            "[redacted — probable credential in stored content]"
        } else {
            node.content_preview.as_str()
        },
        "contentPreviewRedacted": redacted,
        "tags": node.tags,
        "superseded": node.superseded,
    })
}

fn age_buckets() -> BTreeMap<&'static str, usize> {
    ["0-7d", "8-30d", "31-90d", "91-180d", "181d+", "futureDated"]
        .into_iter()
        .map(|bucket| (bucket, 0))
        .collect()
}

fn retention_buckets() -> BTreeMap<&'static str, usize> {
    [
        "below0",
        "0-20%",
        ">20-40%",
        ">40-60%",
        ">60-80%",
        ">80-100%",
        "above100%",
        "nonFinite",
    ]
    .into_iter()
    .map(|bucket| (bucket, 0))
    .collect()
}

fn retention_bucket(value: f64) -> &'static str {
    if !value.is_finite() {
        "nonFinite"
    } else if value < 0.0 {
        "below0"
    } else if value <= 0.2 {
        "0-20%"
    } else if value <= 0.4 {
        ">20-40%"
    } else if value <= 0.6 {
        ">40-60%"
    } else if value <= 0.8 {
        ">60-80%"
    } else if value <= 1.0 {
        ">80-100%"
    } else {
        "above100%"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};
    use vestige_core::IngestInput;

    fn empty_audit() -> TagAuditWindow {
        TagAuditWindow {
            operations: Vec::new(),
            scanned_operations: 0,
            truncated: false,
        }
    }

    fn summary(
        id: &str,
        node_type: &str,
        created_at: DateTime<Utc>,
        retention_strength: f64,
        tags: &[&str],
        content_bytes: usize,
    ) -> HygieneNodeSummary {
        HygieneNodeSummary {
            id: id.to_string(),
            node_type: node_type.to_string(),
            created_at,
            retention_strength,
            tags: tags.iter().map(|tag| (*tag).to_string()).collect(),
            valid_from: None,
            valid_until: None,
            superseded: false,
            content_bytes,
            content_preview: format!("preview-{id}"),
            never_accessed: false,
        }
    }

    #[tokio::test]
    async fn empty_store_has_all_zero_buckets_and_default_scope() {
        let directory = tempfile::tempdir().expect("temporary database");
        let storage =
            Arc::new(Storage::new(Some(directory.path().join("stats.db"))).expect("test storage"));

        let response = execute(&storage, Some(json!({ "view": "stats" })))
            .await
            .expect("empty hygiene stats");

        assert_eq!(response["scope"]["value"], DEFAULT_MEMORY_SCOPE);
        assert_eq!(response["population"]["total"], 0);
        assert_eq!(response["counts"]["byAge"]["futureDated"], 0);
        assert_eq!(response["retentionDistribution"]["buckets"][">80-100%"], 0);
        assert_eq!(response["neverAccessed"]["total"], 0);
        assert_eq!(response["largestNodes"]["total"], 0);
        assert_eq!(response["recentTagOperations"]["returned"], 0);

        let blank_scope = execute(&storage, Some(json!({ "view": "stats", "scope": "   " })))
            .await
            .expect("blank scope uses the legacy-compatible default");
        assert_eq!(blank_scope["scope"]["value"], DEFAULT_MEMORY_SCOPE);
    }

    #[tokio::test]
    async fn tag_audit_is_agent_visible_and_scope_filtered() {
        let directory = tempfile::tempdir().expect("temporary database");
        let storage =
            Arc::new(Storage::new(Some(directory.path().join("audit.db"))).expect("test storage"));
        storage
            .ingest_in_scope(
                IngestInput {
                    content: "scoped audit fixture".into(),
                    tags: vec!["old-tag".into()],
                    ..Default::default()
                },
                "user",
            )
            .expect("seed scoped memory");
        let sources = vec!["old-tag".to_string()];
        let preview = storage
            .preview_tag_mutation(&sources, "new-tag", Some("user"))
            .expect("tag preview");
        let preview_token = preview["previewToken"].as_str().expect("preview token");
        storage
            .apply_tag_mutation(
                &sources,
                "new-tag",
                Some("user"),
                preview_token,
                "tag_rename",
                "verify the agent-visible tag audit",
            )
            .expect("apply tag rename");

        let user_stats = execute(&storage, Some(json!({ "view": "stats" })))
            .await
            .expect("user-scope stats");
        assert_eq!(user_stats["recentTagOperations"]["returned"], 1);
        assert_eq!(
            user_stats["recentTagOperations"]["operations"][0]["operationType"],
            "tag_rename"
        );
        assert_eq!(
            user_stats["recentTagOperations"]["operations"][0]["scope"],
            "user"
        );
        assert_eq!(
            user_stats["recentTagOperations"]["operations"][0]["sourceTags"],
            json!(["old-tag"])
        );
        assert_eq!(
            user_stats["recentTagOperations"]["operations"][0]["targetTag"],
            "new-tag"
        );
        assert_eq!(
            user_stats["recentTagOperations"]["operations"][0]["reason"],
            "verify the agent-visible tag audit"
        );

        let other_stats = execute(
            &storage,
            Some(json!({ "view": "stats", "scope": "other-project" })),
        )
        .await
        .expect("other-scope stats");
        assert_eq!(other_stats["recentTagOperations"]["returned"], 0);
    }

    #[test]
    fn deterministic_all_category_fixture_covers_full_agent_response() {
        let now = Utc.with_ymd_and_hms(2026, 8, 19, 12, 0, 0).unwrap();
        let mut nodes = vec![
            summary("n0", "fact", now + Duration::days(1), -0.1, &["alpha"], 10),
            summary(
                "n1",
                "decision",
                now - Duration::days(3),
                0.1,
                &["alpha", "alpha"],
                20,
            ),
            summary("n2", "fact", now - Duration::days(8), 0.3, &["beta"], 30),
            summary("n3", "procedure", now - Duration::days(31), 0.5, &[], 40),
            summary(
                "n4",
                "concept",
                now - Duration::days(91),
                0.7,
                &["gamma"],
                50,
            ),
            summary("n5", "fact", now - Duration::days(181), 0.9, &["beta"], 60),
            summary(
                "a-tie",
                "fact",
                now - Duration::days(30),
                1.1,
                &["delta"],
                99,
            ),
            summary(
                "b-tie",
                "fact",
                now - Duration::days(90),
                f64::NAN,
                &["delta"],
                99,
            ),
        ];
        nodes[0].valid_from = Some(now + Duration::days(1));
        nodes[1].valid_until = Some(now - Duration::seconds(1));
        nodes[2].valid_from = Some(now + Duration::days(2));
        nodes[2].valid_until = Some(now - Duration::days(2));
        nodes[3].superseded = true;
        nodes[5].never_accessed = true;
        nodes[4].never_accessed = true;
        nodes[3].never_accessed = true;

        let response = build_response(nodes, None, 2, now, empty_audit());

        assert_eq!(response["scope"]["mode"], "all");
        assert_eq!(response["population"]["total"], 8);
        assert_eq!(response["counts"]["byMemoryType"]["fact"], 5);
        assert_eq!(response["counts"]["byTag"]["alpha"], 2);
        assert_eq!(response["counts"]["untagged"], 1);
        for bucket in ["0-7d", "8-30d", "31-90d", "91-180d", "181d+", "futureDated"] {
            assert!(response["counts"]["byAge"][bucket].as_u64().unwrap() > 0);
        }
        for bucket in [
            "below0",
            "0-20%",
            ">20-40%",
            ">40-60%",
            ">60-80%",
            ">80-100%",
            "above100%",
            "nonFinite",
        ] {
            assert_eq!(response["retentionDistribution"]["buckets"][bucket], 1);
        }
        assert_eq!(response["lifecycle"]["notYetValid"], 1);
        assert_eq!(response["lifecycle"]["expired"], 2);
        assert_eq!(response["lifecycle"]["invalidTemporalBounds"], 1);
        assert_eq!(response["lifecycle"]["superseded"], 1);
        assert_eq!(response["neverAccessed"]["total"], 3);
        assert_eq!(response["neverAccessed"]["returned"], 2);
        assert_eq!(response["neverAccessed"]["truncated"], true);
        assert_eq!(response["neverAccessed"]["memories"][0]["id"], "n5");
        assert_eq!(response["neverAccessed"]["memories"][1]["id"], "n4");
        assert_eq!(response["largestNodes"]["memories"][0]["id"], "a-tie");
        assert_eq!(response["largestNodes"]["memories"][1]["id"], "b-tie");
        assert_eq!(response["largestNodes"]["truncated"], true);
        assert_eq!(
            response["neverAccessed"]["memories"][0]["contentPreviewRedacted"],
            false
        );
    }

    #[test]
    fn detail_lists_redact_blocking_credentials_without_echoing_them() {
        let now = Utc.with_ymd_and_hms(2026, 8, 19, 12, 0, 0).unwrap();
        let credential = format!("ghp_{}", "A".repeat(36));
        let mut node = summary(
            "secret-node",
            "fact",
            now - Duration::days(181),
            0.9,
            &["ops"],
            60,
        );
        node.never_accessed = true;
        node.content_preview = format!("rotated token {credential}");

        let response = build_response(vec![node], Some("user"), 50, now, empty_audit());
        let encoded = serde_json::to_string(&response).expect("encode stats response");
        assert!(
            !encoded.contains(&credential),
            "hygiene stats must not echo a stored credential"
        );
        assert_eq!(
            response["neverAccessed"]["memories"][0]["contentPreview"],
            "[redacted — probable credential in stored content]"
        );
        assert_eq!(
            response["neverAccessed"]["memories"][0]["contentPreviewRedacted"],
            true
        );
        assert_eq!(
            response["largestNodes"]["memories"][0]["contentPreviewRedacted"],
            true
        );
    }

    #[tokio::test]
    async fn stats_population_stays_inside_the_requested_scope() {
        let directory = tempfile::tempdir().expect("temporary database");
        let storage =
            Arc::new(Storage::new(Some(directory.path().join("scope.db"))).expect("test storage"));
        storage
            .ingest_in_scope(
                IngestInput {
                    content: "user-scope hygiene count".into(),
                    tags: vec!["user-tag".into()],
                    ..Default::default()
                },
                "user",
            )
            .expect("seed user memory");
        storage
            .ingest_in_scope(
                IngestInput {
                    content: "project-scope hygiene count".into(),
                    tags: vec!["project-tag".into()],
                    ..Default::default()
                },
                "project-a",
            )
            .expect("seed project memory");

        let user_stats = execute(&storage, Some(json!({ "view": "stats" })))
            .await
            .expect("user-scope stats");
        assert_eq!(user_stats["population"]["total"], 1);
        assert_eq!(user_stats["counts"]["byTag"]["user-tag"], 1);
        assert!(user_stats["counts"]["byTag"]["project-tag"].is_null());

        let project_stats = execute(
            &storage,
            Some(json!({ "view": "stats", "scope": "project-a" })),
        )
        .await
        .expect("project-scope stats");
        assert_eq!(project_stats["population"]["total"], 1);
        assert_eq!(project_stats["counts"]["byTag"]["project-tag"], 1);

        let all_stats = execute(
            &storage,
            Some(json!({ "view": "stats", "all_scopes": true })),
        )
        .await
        .expect("all-scopes stats");
        assert_eq!(all_stats["population"]["total"], 2);
        assert_eq!(all_stats["scope"]["mode"], "all");
    }

    #[test]
    fn detail_limit_is_validated_and_capped() {
        assert_eq!(parse_limit(&json!({})).unwrap(), DEFAULT_DETAIL_LIMIT);
        assert_eq!(
            parse_limit(&json!({ "limit": 500 })).unwrap(),
            MAX_DETAIL_LIMIT
        );
        assert!(parse_limit(&json!({ "limit": 0 })).is_err());
        assert!(parse_limit(&json!({ "limit": "many" })).is_err());
    }
}
