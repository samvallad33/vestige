//! Unified `memory_status` Tool (v2.2 — Tool Consolidation)
//!
//! Folds four read-only status/health/temporal tools into one
//! view-dispatched surface:
//!
//!   view = health (default) | retention | timeline | changelog | stats
//!
//! - `health` → full system health + statistics (the former `system_status`).
//!   Returns the byte-for-byte `system_status` shape (audit scripts parse it),
//!   including `schema_introspection` passthrough.
//! - `retention` → the lightweight retention dashboard (former `memory_health`).
//! - `timeline` → chronological browse (former `memory_timeline`).
//! - `changelog` → audit trail of memory changes (former `memory_changelog`).
//! - `stats` → full-store hygiene aggregates plus bounded diagnostic lists.
//! - `tools` → server-resolved progressive discovery from the actual tools/list.
//!
//! This is a thin facade: each view forwards the *same* args envelope to the
//! existing handler. None of the underlying arg structs use
//! `deny_unknown_fields`, so the discriminator `view` is simply ignored by each
//! handler — no lossy re-scoping required, and per-view fields validate as
//! before. The `cognitive` lock is never held across a forwarded call.

use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::{OutputConfig, Storage};

use crate::cognitive::CognitiveEngine;

/// Progressive discovery from the `tools/list` result. Since #212 the
/// catalog on the wire is compact; a selected tool's `inputSchema` here comes
/// from the full-schema registry in `tools::compact`, so the complete shape
/// stays available one call deeper. The parity guard test in `server.rs`
/// keeps that registry and the catalog name-for-name identical. Tool-level
/// annotations remain tool-level hints; they are not permission for
/// individual actions.
pub fn tool_guide(catalog: &Value, args: &Value) -> Result<Value, String> {
    let selected = match args.get("tool") {
        None => None,
        Some(Value::String(name)) if !name.trim().is_empty() => Some(name.as_str()),
        Some(_) => return Err("tool must be a non-empty advertised tool name".into()),
    };
    let definitions = catalog["tools"]
        .as_array()
        .ok_or("Tool catalog unavailable")?;
    let mut entries = Vec::new();
    for definition in definitions {
        if selected.is_some_and(|name| definition["name"] != name) {
            continue;
        }
        let schema = &definition["inputSchema"];
        let mut selectors = serde_json::Map::new();
        for field in ["action", "mode", "view"] {
            if let Some(values) = schema["properties"][field]["enum"].as_array() {
                selectors.insert(
                    field.into(),
                    serde_json::json!({
                        "values": values,
                        "default": schema["properties"][field]["default"],
                    }),
                );
            }
        }
        let mut entry = serde_json::json!({
            "name": definition["name"],
            "description": definition["description"],
            "selectors": selectors,
            "toolAnnotations": definition["annotations"],
        });
        if let Some(name) = selected {
            // #212: the catalog schema is compact; the selected tool gets the
            // full registry schema, falling back to the catalog form only if
            // the registry ever missed a name (the parity test forbids that).
            entry["inputSchema"] =
                crate::tools::compact::full_schema(name).unwrap_or_else(|| schema.clone());
        }
        entries.push(entry);
    }
    if entries.is_empty() {
        return Err(format!(
            "Unknown advertised tool '{}'; omit tool to list available names",
            selected.unwrap_or_default()
        ));
    }
    Ok(serde_json::json!({
        "view": "tools",
        "catalogVersion": env!("CARGO_PKG_VERSION"),
        "source": "tools/list",
        "compiledFeatures": {
            "embeddings": cfg!(vestige_embeddings_removed),
            "vectorSearch": cfg!(vestige_embeddings_removed),
            "connectors": cfg!(feature = "connectors"),
            "cloudSync": cfg!(feature = "cloud-sync"),
        },
        "availabilityNote": "Compiled features are not runtime readiness. Merge candidates, merge plans and plan apply need embeddings, which this build does not ship; the exact-equality dedup scan, tag maintenance and undo work without them. Inspect health for runtime state. Connector calls additionally require configured upstream access.",
        "tools": entries,
        "guidance": "Choose tools for the task; there is no call quota. To inspect arguments, call memory_status with view='tools' and tool='<name>'. Descriptions and schemas describe capabilities, not authorization. Mixed tools include both reads and writes; inspect the selected action and confirmation requirements. Memory matches, graph links, retention scores and receipt replay do not establish truth or causality."
    }))
}

/// Discriminated-union schema for the unified `memory_status` tool.
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "view": {
                "type": "string",
                "enum": ["health", "retention", "timeline", "changelog", "provenance", "coverage", "stats", "tools"],
                "default": "health",
                "description": "'tools': current tool/action inventory, or one full schema with tool. 'health' (default): system health, stats, decay preview, warnings, structured diagnostics (with entity IDs and next actions), recommendations. 'retention': average, distribution, trend. 'timeline': memories by date, bounded by limit (returned/truncated are explicit). 'changelog': audit trail of consolidations, dreams, state transitions, and merge/supersede/undo/tag operations; with memory_id it is legacy engine only (on Strata use 'provenance' with memoryId). 'coverage': anchor coverage (nodes with code_memory_anchors / total), memory edge counts by link type, and index freshness (newest git-commit record age, newest agent trace age). 'stats': hygiene counts by type, tag, age, retention, and lifecycle, bounded detail lists, recommended actions with example IDs, and recent tag operations."
            },
            "tool": {
                "type": "string",
                "description": "[tools view] Exact advertised tool name. Omit for a compact inventory of every tool and action; specify a name for its complete input schema. This is discovery, not tool execution."
            },
            // --- [health view] ---
            "schema_introspection": {
                "type": "boolean",
                "description": "[health] Include the response-schema description."
            },
            // --- [timeline view] ---
            "start": { "type": "string", "description": "[timeline, changelog] Range start (ISO 8601)." },
            "end": { "type": "string", "description": "[timeline, changelog] Range end (ISO 8601)." },
            "node_type": { "type": "string", "description": "[timeline] Filter by node type." },
            "tags": { "type": "array", "items": { "type": "string" }, "description": "[timeline view] Filter by tags (ANY match)." },
            "detail_level": {
                "type": "string", "enum": ["brief", "summary", "full"],
                "description": "[timeline] Detail level (default 'summary')."
            },
            // --- [changelog view] ---
            "memory_id": { "type": "string", "description": "[changelog] State transitions for this memory only. Legacy engine only; on Strata use view 'provenance' with memoryId." },
            // --- [stats view] ---
            "scope": {
                "type": "string",
                "default": "user",
                "description": "[stats] Scope to aggregate (default 'user'); ignored when all_scopes."
            },
            "all_scopes": {
                "type": "boolean",
                "default": false,
                "description": "[stats] Aggregate every scope. Default false."
            },
            // --- shared: limit (per-view ranges differ; clamped internally) ---
            "limit": {
                "type": "integer",
                "description": "Max results: timeline 50 (max 200), changelog 20 (max 100), stats lists 50 (max 200). Ignored by health and retention.",
                "minimum": 1, "maximum": 200
            }
        }
    })
}

/// Unified dispatcher for `memory_status`. Routes on `view` (default `health`).
pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    output_config: &OutputConfig,
    args: Option<Value>,
) -> Result<Value, String> {
    let view = args
        .as_ref()
        .and_then(|a| a.get("view"))
        .and_then(|v| v.as_str())
        .unwrap_or("health")
        .to_string();

    match view.as_str() {
        // Byte-for-byte system_status shape (incl. schema_introspection passthrough).
        "health" => super::maintenance::execute_system_status(storage, cognitive, args).await,
        "retention" => super::health::execute(storage, args).await,
        "timeline" => super::timeline::execute(storage, output_config, args).await,
        "changelog" => super::changelog::execute(storage, args).await,
        "stats" => super::hygiene_stats::execute(storage, args).await,
        // #252 Phase A: "who said what" — endorsement events with their role
        // resolution, plus the operator policy snapshot in force.
        "provenance" => execute_provenance(storage, args),
        // Coverage: anchor coverage, edge counts by type, index freshness.
        "coverage" => execute_coverage(storage),
        other => Err(format!(
            "Unknown memory_status view '{other}'. Use health|retention|timeline|changelog|provenance|coverage|stats|tools."
        )),
    }
}

/// The `coverage` view: exact store aggregates for how much of the knowledge
/// graph is anchored to source, how the edge population breaks down by link
/// type, and how fresh the indexes that feed code memory are. Read-only, no
/// lists — counts and ages only, so the output shape is constant.
pub fn execute_coverage(storage: &Arc<Storage>) -> Result<Value, String> {
    let snapshot = storage.coverage_snapshot().map_err(|e| e.to_string())?;

    let mut edge_counts = serde_json::Map::new();
    for (link_type, count) in &snapshot.edge_counts_by_type {
        edge_counts.insert(link_type.clone(), json!(count));
    }

    // Deterministic staleness note derived from the two age probes.
    let commit_age = snapshot.newest_git_commit_record_age_days;
    let trace_age = snapshot.newest_agent_trace_age_hours;
    let staleness_note = match (commit_age, trace_age) {
        (None, None) => {
            "no git-commit records and no agent traces recorded yet; index freshness is unknown"
                .to_string()
        }
        (None, Some(_)) => {
            "no git-commit records in this store; commit coverage is unknown".to_string()
        }
        (Some(_), None) => "no agent trace events yet; Black Box freshness is unknown".to_string(),
        (Some(days), Some(hours)) => {
            let commit_side = if days > 30 {
                format!("git-commit records are {days} days old")
            } else {
                "git-commit records are fresh".to_string()
            };
            let trace_side = if hours > 72.0 {
                format!("agent traces are {hours} hours old")
            } else {
                "agent traces are fresh".to_string()
            };
            format!("{commit_side}; {trace_side}")
        }
    };

    Ok(json!({
        "view": "coverage",
        "anchorCoveragePct": snapshot.anchor_coverage_pct,
        "anchoredNodes": snapshot.anchored_nodes,
        "totalNodes": snapshot.total_nodes,
        "edgeCountsByType": Value::Object(edge_counts),
        "indexFreshness": {
            "newestGitCommitRecord": snapshot.newest_git_commit_record,
            "newestGitCommitRecordAgeDays": commit_age,
            "newestAgentTraceAt": snapshot.newest_agent_trace_at,
            "newestAgentTraceAgeHours": trace_age,
            "stalenessNote": staleness_note,
        },
        "claimBoundary": "Coverage counts anchors, edges, and record ages. It measures how much of the store is anchored and how fresh the indexes are; it says nothing about memory truth or causal linkage.",
    }))
}

/// The `provenance` view (#252 Phase A): actor-attributed endorsement events
/// ("who said what"), optionally filtered by `memoryId` or `actorId`, with
/// the operator-controlled role/weight policy snapshot and the process actor
/// identity. Read-only; every list is bounded.
pub fn execute_provenance(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        return crate::strata_memory::execute_node_provenance(storage.as_ref(), args.as_ref());
    }
    const MAX_PROVENANCE_RESULTS: usize = 50;
    let memory_id = args
        .as_ref()
        .and_then(|a| a.get("memoryId"))
        .and_then(|v| v.as_str());
    let actor_id = args
        .as_ref()
        .and_then(|a| a.get("actorId"))
        .and_then(|v| v.as_str());
    let limit = args
        .as_ref()
        .and_then(|a| a.get("limit"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(MAX_PROVENANCE_RESULTS);

    let events = storage
        .list_endorsement_events(memory_id, actor_id, limit)
        .map_err(|e| e.to_string())?;
    let snapshot = storage.actor_policy_snapshot().map_err(|e| e.to_string())?;
    let process_actor = storage.process_actor_did();

    let events_json: Vec<Value> = events
        .iter()
        .map(|event| {
            serde_json::json!({
                "eventId": event.event_id,
                "memoryId": event.memory_id,
                "actorId": event.actor_did,
                "claimedRole": event.claimed_role,
                "effectiveRole": event.effective_role,
                "resolvedWeight": event.resolved_weight,
                "resolutionDisposition": event.resolution_disposition,
                "policyVersion": event.policy_version,
                "kind": event.endorsement_kind,
                "revisionDigest": event.revision_digest,
                "independentPrior": event.independent_prior,
                "tool": event.tool,
                "receiptId": event.receipt_id,
                "createdAt": event.created_at,
            })
        })
        .collect();

    let weights_json: serde_json::Map<String, Value> = snapshot
        .weights
        .iter()
        .map(|(role, weight)| (role.clone(), json!(weight)))
        .collect();
    let memberships_json: serde_json::Map<String, Value> = snapshot
        .memberships
        .iter()
        .map(|(actor, roles)| (actor.clone(), json!(roles)))
        .collect();

    Ok(serde_json::json!({
        "view": "provenance",
        "processActor": process_actor,
        "processActorBound": process_actor.is_some(),
        "policy": {
            "version": snapshot.policy_version,
            "roleWeights": weights_json,
            "membership": memberships_json,
            "note": "Operator-controlled. No tool call can grant roles or change weights; edits happen directly in the store and bump the version.",
        },
        "endorsements": events_json,
        "count": events_json.len(),
        "claimBoundary": "Provenance records who made or endorsed a claim and at what resolved authority. It never establishes that a claim is true. Unregistered actors are neutral at 1.0; self-support carries zero independent prior.",
    }))
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;
    use crate::cognitive::CognitiveEngine;

    fn test_storage() -> Arc<Storage> {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        // Keep the tempdir alive for the duration of the process by leaking it;
        // these are short-lived unit tests.
        std::mem::forget(dir);
        storage
    }

    #[test]
    fn test_schema_views() {
        let s = schema();
        let views = s["properties"]["view"]["enum"].as_array().unwrap();
        assert_eq!(views.len(), 8);
        assert_eq!(s["properties"]["view"]["default"], "health");
        assert_eq!(s["properties"]["scope"]["default"], "user");
        assert_eq!(s["properties"]["all_scopes"]["default"], false);
    }

    #[tokio::test]
    async fn test_default_view_is_health() {
        let storage = test_storage();
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let oc = OutputConfig::default();
        // No args → health view → must match system_status output exactly.
        let unified = execute(&storage, &cognitive, &oc, None).await.unwrap();
        let direct = super::super::maintenance::execute_system_status(&storage, &cognitive, None)
            .await
            .unwrap();
        assert_eq!(
            unified, direct,
            "memory_status view=health must equal system_status byte-for-byte"
        );
    }

    #[tokio::test]
    async fn test_all_views_resolve() {
        let storage = test_storage();
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let oc = OutputConfig::default();
        for view in [
            "health",
            "retention",
            "timeline",
            "changelog",
            "provenance",
            "coverage",
            "stats",
        ] {
            let args = Some(serde_json::json!({ "view": view }));
            let r = execute(&storage, &cognitive, &oc, args).await;
            assert!(r.is_ok(), "view={view} should resolve, got {r:?}");
        }
    }

    #[tokio::test]
    async fn test_coverage_view_math_on_seeded_store() {
        let storage = test_storage();
        // Empty store: zero coverage, no NaN, unknown freshness.
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let oc = OutputConfig::default();
        let empty = execute(
            &storage,
            &cognitive,
            &oc,
            Some(serde_json::json!({ "view": "coverage" })),
        )
        .await
        .unwrap();
        assert_eq!(empty["view"], "coverage");
        assert_eq!(empty["anchorCoveragePct"], 0.0);
        assert_eq!(empty["totalNodes"], 0);
        assert_eq!(empty["anchoredNodes"], 0);
        assert_eq!(empty["edgeCountsByType"], serde_json::json!({}));
        assert_eq!(
            empty["indexFreshness"]["stalenessNote"],
            "no git-commit records and no agent traces recorded yet; index freshness is unknown"
        );

        // Seed: 1 anchored node of 4 others, two edge types, a git-commit
        // record, and one trace event. memory_connections is keyed
        // (source_id, target_id), so distinct pairs are needed per edge.
        let anchored = storage
            .ingest(vestige_core::IngestInput {
                content: "pattern: prefer canonical json for digests".to_string(),
                node_type: "pattern".to_string(),
                ..Default::default()
            })
            .unwrap();
        let mut facts = Vec::new();
        for content in ["plain fact one", "plain fact two", "plain fact three"] {
            facts.push(
                storage
                    .ingest(vestige_core::IngestInput {
                        content: content.to_string(),
                        ..Default::default()
                    })
                    .unwrap()
                    .id,
            );
        }
        let commit = storage
            .ingest(vestige_core::IngestInput {
                content: "commit aabbcc0 bump schema".to_string(),
                tags: vec!["git-commit".to_string()],
                ..Default::default()
            })
            .unwrap();
        use chrono::Utc;
        use vestige_core::ConnectionRecord;
        use vestige_core::codebase::CodeAnchor;
        storage
            .record_code_anchors(&[CodeAnchor {
                id: "anc_cov".into(),
                node_id: anchored.id.clone(),
                file_path: "src/main.rs".into(),
                symbol: None,
                symbol_kind: None,
                start_line: Some(1),
                end_line: Some(1),
                span_lines: Some(1),
                content_hash: Some("b3:deadbeef".into()),
                captured_at: Utc::now(),
                last_verified_at: None,
                last_status: None,
            }])
            .unwrap();
        for (target, link) in [
            (commit.id.as_str(), "backfill_candidate"),
            (facts[0].as_str(), "backfill_candidate"),
            (facts[1].as_str(), "semantic"),
        ] {
            storage
                .save_connection(&ConnectionRecord {
                    source_id: anchored.id.clone(),
                    target_id: target.to_string(),
                    strength: 0.4,
                    link_type: link.to_string(),
                    created_at: Utc::now(),
                    last_activated: Utc::now(),
                    activation_count: 0,
                })
                .unwrap();
        }
        storage
            .append_trace_event(&vestige_core::MemoryTraceEvent::McpCall {
                run_id: "run_cov".into(),
                tool: "memory_status".into(),
                args_hash: "0".into(),
                at: Utc::now().timestamp_millis(),
            })
            .unwrap();

        let coverage = execute(
            &storage,
            &cognitive,
            &oc,
            Some(serde_json::json!({ "view": "coverage" })),
        )
        .await
        .unwrap();
        assert_eq!(coverage["totalNodes"], 5);
        assert_eq!(coverage["anchoredNodes"], 1);
        assert_eq!(coverage["anchorCoveragePct"], 20.0);
        assert_eq!(
            coverage["edgeCountsByType"],
            serde_json::json!({"backfill_candidate": 2, "semantic": 1})
        );
        let freshness = &coverage["indexFreshness"];
        assert!(
            freshness["newestGitCommitRecordAgeDays"].as_i64().unwrap() <= 1,
            "seeded commit is seconds old"
        );
        assert!(
            freshness["newestAgentTraceAgeHours"].as_f64().unwrap() < 1.0,
            "seeded trace is seconds old"
        );
        assert_eq!(
            freshness["stalenessNote"],
            "git-commit records are fresh; agent traces are fresh"
        );
    }

    #[tokio::test]
    async fn test_provenance_view_lists_endorsements_and_policy() {
        let storage = test_storage();
        // Ingest before binding: the later support is plain "support".
        let node = storage
            .ingest(vestige_core::IngestInput {
                content: "provenance view target".to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec![],
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap();
        let did = vestige_core::actor::ProcessActor::mint().did().to_string();
        storage.set_process_actor(&did).unwrap();
        storage
            .promote_memory_as_actor(&node.id, Some("qa"), "memory")
            .unwrap();

        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let oc = OutputConfig::default();
        let args = Some(serde_json::json!({ "view": "provenance", "memoryId": node.id }));
        let value = execute(&storage, &cognitive, &oc, args).await.unwrap();
        assert_eq!(value["view"], "provenance");
        assert_eq!(value["processActor"], did);
        assert_eq!(value["count"], 1);
        assert_eq!(value["endorsements"][0]["kind"], "support");
        assert_eq!(value["endorsements"][0]["actorId"], did);
        assert_eq!(value["endorsements"][0]["effectiveRole"], "unattributed");
        assert_eq!(value["policy"]["version"], 1);
        assert_eq!(value["policy"]["roleWeights"]["operator"], 1.50);
        assert_eq!(value["policy"]["roleWeights"]["destructive-tester"], 1.30);
        assert_eq!(value["policy"]["roleWeights"]["architect"], 1.25);
        assert_eq!(value["policy"]["roleWeights"]["functional-tester"], 1.15);
        assert_eq!(value["policy"]["roleWeights"]["qa"], 1.10);
        assert_eq!(value["policy"]["roleWeights"]["dev"], 1.00);
        assert!(
            value["claimBoundary"]
                .as_str()
                .unwrap()
                .contains("never establishes that a claim is true")
        );

        // Filter by actor id.
        let args = Some(serde_json::json!({ "view": "provenance", "actorId": did }));
        let value = execute(&storage, &cognitive, &oc, args).await.unwrap();
        assert_eq!(value["count"], 1);

        // Unknown filters return an empty, well-formed list.
        let args =
            Some(serde_json::json!({ "view": "provenance", "actorId": "did:key:z6MkNobody" }));
        let value = execute(&storage, &cognitive, &oc, args).await.unwrap();
        assert_eq!(value["count"], 0);
    }
}

#[cfg(test)]
mod strata_stdio {
    use super::*;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};

    fn init_line() -> String {
        json!({
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "provenance-probe", "version": "1"}
            }
        })
        .to_string()
            + "\n"
    }

    fn hex32(bytes: &[u8; 32]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(64);
        for byte in bytes {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
        out
    }

    async fn drive(storage: Arc<Storage>, input: String) -> Vec<Value> {
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let server = crate::server::McpServer::new(storage, cognitive);
        let (mut client_w, server_r) = tokio::io::duplex(1 << 16);
        let (server_w, mut client_r) = tokio::io::duplex(1 << 20);
        let handle = tokio::spawn(async move {
            crate::protocol::stdio::run_io(server, None, BufReader::new(server_r), server_w).await
        });
        client_w.write_all(input.as_bytes()).await.unwrap();
        drop(client_w);
        let mut buf = String::new();
        client_r.read_to_string(&mut buf).await.unwrap();
        handle.await.unwrap().unwrap();
        buf.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str::<Value>(line).expect("json"))
            .collect()
    }

    #[tokio::test]
    async fn provenance_stdio_returns_the_planted_origin() {
        const CREATED_MS: i64 = 1_700_000_000_000;
        let dir = tempfile::TempDir::new().unwrap();
        let (origin_id, successor_id, frame_hash, frame_seq, hop_hash) = {
            let mut store = strata_store::StrataStore::open_with_policy(
                dir.path(),
                strata_store::permissive_policy(),
            )
            .unwrap();
            let origin_id = store
                .ingest_in_scope(
                    strata_store::IngestInput {
                        content: "planted origin fact".into(),
                        source: None,
                        source_updated_at_ms: None,
                        node_type: "decision".into(),
                        tags: vec!["prov-seed".into()],
                        created_at_ms: Some(CREATED_MS),
                        valid_from_ms: Some(CREATED_MS),
                        valid_until_ms: None,
                    },
                    "user",
                )
                .unwrap();
            let successor_id = store
                .ingest_in_scope(
                    strata_store::IngestInput {
                        content: "planted successor fact".into(),
                        source: None,
                        source_updated_at_ms: None,
                        node_type: "fact".into(),
                        tags: vec!["prov-next".into()],
                        created_at_ms: Some(CREATED_MS + 111_000),
                        valid_from_ms: None,
                        valid_until_ms: None,
                    },
                    "user",
                )
                .unwrap();
            store.supersede(&origin_id, &successor_id).unwrap();
            let origin = store.recorded_origin(&origin_id).unwrap().unwrap();
            assert_eq!(origin.record.content, "planted origin fact");
            assert_eq!(origin.supersede_chain.len(), 1);
            (
                origin_id,
                successor_id,
                hex32(&origin.frame_hash),
                origin.frame_seq,
                hex32(&origin.supersede_chain[0].frame_hash),
            )
        };

        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let input = init_line()
            + &format!(
                "{}\n{}\n{}\n",
                json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
                json!({
                    "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {
                        "name": "memory_status",
                        "arguments": {"view": "provenance", "memoryId": origin_id}
                    }
                }),
                json!({
                    "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {
                        "name": "memory_status",
                        "arguments": {"view": "provenance", "memoryId": "mem-00000000000000ff"}
                    }
                })
            );
        let out = drive(storage, input).await;
        let listed = out
            .iter()
            .find(|v| v["id"] == json!(1))
            .expect("tools/list");
        let tools = listed["result"]["tools"].as_array().unwrap();
        let advertises: Vec<&str> = tools
            .iter()
            .filter(|tool| {
                tool["inputSchema"]["properties"]["view"]["enum"]
                    .as_array()
                    .is_some_and(|values| values.iter().any(|value| value == "provenance"))
            })
            .filter_map(|tool| tool["name"].as_str())
            .collect();
        assert_eq!(advertises, vec!["memory_status"]);

        let call = out
            .iter()
            .find(|v| v["id"] == json!(2))
            .expect("tools/call");
        assert_ne!(call["result"]["isError"], json!(true), "{call}");
        let body = &call["result"]["structuredContent"];
        assert_eq!(body["view"], "provenance");
        assert_eq!(body["status"], "completed");
        assert_eq!(body["found"], true);
        assert_eq!(body["memoryId"], origin_id);
        assert_eq!(body["origin"]["record"]["content"], "planted origin fact");
        assert_eq!(body["origin"]["record"]["nodeType"], "decision");
        assert_eq!(body["origin"]["record"]["tags"], json!(["prov-seed"]));
        assert_eq!(body["origin"]["record"]["scope"], "user");
        assert_eq!(body["origin"]["record"]["id"], origin_id);
        assert_eq!(body["origin"]["timestamps"]["createdAtMs"], CREATED_MS);
        assert_eq!(body["origin"]["timestamps"]["validFromMs"], CREATED_MS);
        assert!(body["origin"]["timestamps"]["validUntilMs"].is_null());
        assert!(body["origin"]["actor"].is_null());
        assert!(body["origin"]["source"].is_null());
        assert_eq!(body["origin"]["frame"]["kind"], 32);
        assert_eq!(body["origin"]["frame"]["kindName"], "STORE_WRITE");
        assert_eq!(body["origin"]["frame"]["op"], "UpsertNode");
        assert_eq!(body["origin"]["frame"]["seq"], frame_seq);
        assert_eq!(body["origin"]["frame"]["frameHash"], frame_hash);
        assert_eq!(
            body["origin"]["frame"]["effectSeq"].as_u64().unwrap() + 1,
            frame_seq
        );
        assert_eq!(body["supersedeChain"].as_array().unwrap().len(), 1);
        assert_eq!(body["supersedeChain"][0]["id"], origin_id);
        assert_eq!(body["supersedeChain"][0]["supersededBy"], successor_id);
        assert_eq!(body["supersedeChain"][0]["recordedAs"], "SupersedeNode");
        assert_eq!(body["supersedeChain"][0]["frameHash"], hop_hash);

        let missing = out.iter().find(|v| v["id"] == json!(3)).expect("missing");
        assert_ne!(missing["result"]["isError"], json!(true), "{missing}");
        let missing_body = &missing["result"]["structuredContent"];
        assert_eq!(missing_body["status"], "completed");
        assert_eq!(missing_body["found"], false);
    }
}
