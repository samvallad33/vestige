//! Unified `recall` Tool (v2.2 — Tool Consolidation, HOT PATH)
//!
//! Folds the four retrieval/reasoning tools into one mode-dispatched surface:
//!
//!   mode = lookup (DEFAULT) | reason | contradictions
//!
//! - `lookup` (default) → hybrid search (the former `search`). This is the hot
//!   path: with no `mode` set, `recall` is a ZERO-overhead pass-through to
//!   `search_unified::execute` — it must never pay the cost of the reasoning
//!   path. (`deep_reference`/`reason` runs spreading activation + FSRS trust
//!   scoring + contradiction analysis and is 5–10× slower.)
//! - `reason` → deep cognitive reasoning across memories (former
//!   `deep_reference` / `cross_reference`).
//! - `contradictions` → trust-weighted disagreement pairs (former
//!   `contradictions`).
//!
//! ## Handle mode (handle-based recall — EXACT/PREFIX only, no fuzzy)
//!
//! When the args contain a `handle` key (even empty), the tool switches to the
//! handle flow BEFORE any mode dispatch: the handle (memory id, commit sha,
//! file path, symbol, test name, run id, tool-call id, or tag) is resolved
//! exactly (or by unique sha/symbol prefix) and the response is the resolved
//! node payloads plus one-hop `memory_connections` neighbors. Free text with
//! no handle inside handle mode returns the `handle_required` error payload
//! with top exact/prefix candidates. Args without a `handle` key keep the
//! legacy mode dispatch untouched — this is the flip point for making recall
//! handle-only by default later.
//!
//! The schema is derived from `search_unified::schema()` (so every lookup
//! parameter stays available and documented) plus the `mode` discriminator and
//! the reason/contradictions fields. `query` is NOT globally required because
//! the contradictions mode is scoped by `topic`; per-mode requirements are
//! validated at runtime.

use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::storage::{HandleKind, HandleResolution, MAX_CANDIDATES};
use vestige_core::{KnowledgeNode, OutputConfig, Storage};

use crate::cognitive::CognitiveEngine;

/// Discriminated-union schema for the unified `recall` tool.
///
/// Built on top of `search_unified::schema()` so all lookup parameters carry
/// through verbatim; the `required: ["query"]` constraint is dropped (validated
/// per-mode at runtime) and the mode/reason/contradictions fields are added.
pub fn schema() -> Value {
    let mut schema = super::search_unified::schema();

    if let Some(obj) = schema.as_object_mut() {
        obj.insert(
            "description".to_string(),
            serde_json::json!(
                "Pass handle: a full memory id or an exact tag. Every other field belongs to the legacy engine's lookup, reason and contradictions modes; Strata returns similarity_disabled for them. reason takes query, depth, limit, scope/includeCrossScope, the retention/score/type/tag/validAt/source filters and token_budget, and rejects lookup-only controls. contradictions takes topic, since, min_trust, limit and scope/includeCrossScope, and rejects token_budget and the lookup/reason filters instead of silently ignoring them."
            ),
        );
        // Drop the global `query` requirement — contradictions uses `topic`.
        obj.remove("required");

        if let Some(props) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) {
            props.insert("id".into(), serde_json::json!({"type":"string", "description":"Exact memory id; alias for handle."}));
            // Handle mode (handle-based recall). Presence of the key — even
            // empty — activates the handle flow at entry, before mode dispatch.
            props.insert(
                "handle".to_string(),
                serde_json::json!({
                    "type": "string",
                            "description": "[handle mode] Exact memory id or case-sensitive tag, including explicit person:Maya entities; a unique id prefix of 8+ characters also resolves. No fuzzy or lexical matching. Multi-memory results rank by updated salience, then recency, before limit, with recorded-edge neighbors."
                }),
            );
            props.insert(
                "mode".to_string(),
                serde_json::json!({
                    "type": "string",
                    "enum": ["lookup", "reason", "contradictions"],
                    "default": "lookup",
                    "description": "'lookup' (default): by handle; a free-text 'query' is legacy engine only. 'reason': legacy engine only; scoped supersession and contradiction analysis of 'query'; verify material claims against current sources. 'contradictions': legacy engine only; disagreement pairs for a 'topic', or recent memories."
                }),
            );
            // reason (deep_reference) extra field.
            props.insert(
                "depth".to_string(),
                serde_json::json!({
                    "type": "integer",
                    "description": "[reason mode] How many memories to analyze (default 20, max 50).",
                    "minimum": 5, "maximum": 50
                }),
            );
            // contradictions extra fields.
            props.insert(
                "topic".to_string(),
                serde_json::json!({
                    "type": "string",
                    "description": "[contradictions mode] Topic to scope contradiction detection. If omitted, scans recent memories."
                }),
            );
            props.insert(
                "since".to_string(),
                serde_json::json!({
                    "type": "string",
                    "description": "[contradictions mode] RFC3339 timestamp; only memories updated after this are considered."
                }),
            );
            props.insert(
                "min_trust".to_string(),
                serde_json::json!({
                    "type": "number",
                    "minimum": 0.0, "maximum": 1.0,
                    "description": "[contradictions mode] Minimum trust for both sides of a contradiction (default 0.3)."
                }),
            );

            // These lookup-pipeline controls have no honest equivalent in the
            // reason pipeline. Mark them mode-specific in the model-facing
            // schema; runtime validation returns a precise error if supplied.
            for field in [
                "detail_level",
                "context_topics",
                "retrieval_mode",
                "concrete",
                "rank_native_fusion",
            ] {
                if let Some(property) = props.get_mut(field).and_then(Value::as_object_mut) {
                    let prior = property
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    property.insert(
                        "description".to_string(),
                        serde_json::json!(format!(
                            "[lookup mode only; reason returns an unsupported-field error] {prior}"
                        )),
                    );
                }
            }

            for field in [
                "query",
                "min_retention",
                "min_similarity",
                "exclude_types",
                "include_types",
                "token_budget",
                "tag_prefix",
                "validAt",
                "source_system",
                "source_project",
                "source_id",
                "source_type",
                "source_author",
                "source_updated_after",
                "source_updated_before",
                "source_status",
            ] {
                if let Some(property) = props.get_mut(field).and_then(Value::as_object_mut) {
                    let prior = property
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    property.insert(
                        "description".to_string(),
                        serde_json::json!(format!(
                            "[lookup and reason modes; contradictions returns an unsupported-field error] {prior}"
                        )),
                    );
                }
            }
        }
    }

    schema
}

/// Unified dispatcher for `recall`. Routes on `mode` (default `lookup`).
///
/// HOT-PATH INVARIANT: `mode` absent ⇒ `lookup` ⇒ direct pass-through to
/// `search_unified::execute`, no extra work.
///
/// HANDLE INVARIANT: a `handle` key in the args short-circuits every mode at
/// entry (see the module docs). This gate is the single flip point for making
/// recall handle-only by default: route the no-handle branch through
/// [`handle_required_payload`] instead of the mode match.
pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    output_config: &OutputConfig,
    args: Option<Value>,
) -> Result<Value, String> {
    if let Some(handled) = handle_flow(storage, &args) {
        return handled;
    }

    let mode = args
        .as_ref()
        .and_then(|a| a.get("mode"))
        .and_then(|v| v.as_str())
        .unwrap_or("lookup");

    match mode {
        // Zero-overhead default: straight to hybrid search.
        "lookup" => {
            if let Some(object) = args.as_ref().and_then(Value::as_object) {
                let wrong_mode_fields: Vec<&str> =
                    ["depth", "topic", "since", "min_trust", "minTrust"]
                        .into_iter()
                        .filter(|field| object.contains_key(*field))
                        .collect();
                if !wrong_mode_fields.is_empty() {
                    return Err(format!(
                        "Unsupported recall lookup-mode field(s): {}.",
                        wrong_mode_fields.join(", ")
                    ));
                }
            }
            super::search_unified::execute(storage, cognitive, output_config, args).await
        }
        // Deep reasoning (deep_reference / cross_reference share this handler).
        "reason" => super::cross_reference::execute(storage, cognitive, args).await,
        // Trust-weighted contradiction pairs (storage-only).
        "contradictions" => super::contradictions::execute(storage, args).await,
        other => Err(format!(
            "Unknown recall mode '{other}'. Use lookup|reason|contradictions."
        )),
    }
}

// ============================================================================
// HANDLE MODE — exact/prefix handle resolution, no fuzzy, no ranking.
// ============================================================================

/// Max one-hop neighbor edges reported per handle resolution.
const MAX_NEIGHBOR_EDGES: usize = 20;
/// Memories a tag returns when the caller sets no limit. An exact id returns
/// its one memory whole; a tag returns this many at summary detail, with
/// `totalNodes` and a `next` call for the rest.
const DEFAULT_TAG_LIMIT: usize = 20;

/// Entry gate for handle mode. Returns `None` when the args carry no `handle`
/// key (legacy mode dispatch proceeds unchanged); otherwise the complete
/// handle-mode response.
/// Rank nodes in storage using the Total-Recall deterministic lexical retrieval engine.
pub fn rank_nodes_lexical(
    storage: &Storage,
    query: &str,
    top_k: usize,
) -> Vec<(String, f32)> {
    rank_nodes_lexical_scoped(storage, None, false, None, query, top_k)
}

/// Scoped, lifecycle-filtered Total-Recall ranking over `Storage`.
/// Enforces scope isolation (when `scope` is `Some`), excludes retired/superseded nodes
/// unless `include_superseded` is true, filters `valid_from` / `valid_until` against `valid_at`,
/// and applies Anderson (2025) `ActiveForgettingSystem` suppression decay.
pub fn rank_nodes_lexical_scoped(
    storage: &Storage,
    scope: Option<&str>,
    include_superseded: bool,
    valid_at: Option<chrono::DateTime<chrono::Utc>>,
    query: &str,
    top_k: usize,
) -> Vec<(String, f32)> {
    let mut nodes = match scope {
        Some(sc) => storage.get_all_nodes_in_scope(sc, i32::MAX, 0).unwrap_or_default(),
        None => storage.get_all_nodes(i32::MAX, 0).unwrap_or_default(),
    };
    if nodes.is_empty() || query.trim().is_empty() {
        return Vec::new();
    }
    if !include_superseded && let Ok(retired) = storage.superseded_node_ids() {
        nodes.retain(|n| !retired.contains(&n.id));
    }
    let check_ts = valid_at.unwrap_or_else(chrono::Utc::now);
    nodes.retain(|n| {
        n.valid_from.is_none_or(|t| t <= check_ts)
            && n.valid_until.is_none_or(|t| t > check_ts)
    });
    if nodes.is_empty() {
        return Vec::new();
    }
    rank_loaded_nodes_total_recall(storage, &nodes, query, top_k)
}

/// Core Total-Recall bridge shared by `recall` (`rank_nodes_lexical`) and the live
/// Reflex hook (`memory_packet::packet`). Wires recorded graph connections, active
/// (non-undone) Retroactive Salience Backfill receipts, typed identity tags, and
/// `ActiveForgettingSystem` suppression penalties into `vestige_core::retrieval::rank`.
pub(crate) fn rank_loaded_nodes_total_recall(
    storage: &Storage,
    raw_nodes: &[KnowledgeNode],
    query: &str,
    top_k: usize,
) -> Vec<(String, f32)> {
    if raw_nodes.is_empty() || query.trim().is_empty() {
        return Vec::new();
    }
    let mut nodes: Vec<&KnowledgeNode> = raw_nodes.iter().collect();
    nodes.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id)));
    let forgetting = vestige_core::neuroscience::ActiveForgettingSystem::default();
    let penalty_by_id: std::collections::HashMap<&str, f32> = nodes
        .iter()
        .filter(|n| n.suppression_count > 0)
        .map(|n| (n.id.as_str(), forgetting.retrieval_penalty(n.suppression_count) as f32))
        .collect();

    // Exact edges the ranker can follow: recorded graph links, backfill receipts,
    // and typed tags (`person:`, `place:`). Built from ids, not from words.
    let mut links_of: std::collections::HashMap<String, Vec<vestige_core::retrieval::TurnLink>> =
        std::collections::HashMap::new();
    let mut tag_members: std::collections::BTreeMap<String, Vec<String>> = std::collections::BTreeMap::new();
    for node in &nodes {
        if let Ok(conns) = storage.get_connections_for_memory(&node.id) {
            for edge in conns {
                let other = if edge.source_id == node.id {
                    edge.target_id
                } else if edge.target_id == node.id {
                    edge.source_id
                } else {
                    continue;
                };
                if other.is_empty() || other == node.id {
                    continue;
                }
                links_of.entry(node.id.clone()).or_default().push(
                    vestige_core::retrieval::TurnLink {
                        kind: edge.link_type,
                        to: other,
                        degree: 2,
                    },
                );
            }
        }
        for tag in &node.tags {
            if tag.contains(':') {
                tag_members.entry(tag.clone()).or_default().push(node.id.clone());
            }
        }
    }
    if let Ok(receipts) = storage.salience_receipts() {
        let undone: std::collections::HashSet<String> = receipts
            .iter()
            .filter_map(|row| {
                row.get("undo_of")
                    .and_then(|v| v.as_str())
                    .filter(|id| !id.is_empty())
                    .map(str::to_string)
            })
            .collect();
        for row in receipts {
            if row.get("undo_of").is_some_and(|v| !v.is_null()) {
                continue;
            }
            if row
                .get("receipt_id")
                .and_then(|v| v.as_str())
                .is_some_and(|id| undone.contains(id))
            {
                continue;
            }
            let memory = row.get("memory_id").and_then(|v| v.as_str()).unwrap_or("");
            let trigger = row.get("trigger_id").and_then(|v| v.as_str()).unwrap_or("");
            if memory.is_empty() || trigger.is_empty() || memory == trigger {
                continue;
            }
            links_of.entry(trigger.to_string()).or_default().push(
                vestige_core::retrieval::TurnLink {
                    kind: "backfill".into(),
                    to: memory.to_string(),
                    degree: 2,
                },
            );
        }
    }
    for ids in tag_members.values_mut() {
        ids.sort();
        ids.dedup();
        if ids.len() < 2 {
            continue;
        }
        let degree = ids.len() as u32;
        let canonical = ids[0].clone();
        for id in ids.iter().skip(1) {
            links_of.entry(id.clone()).or_default().push(
                vestige_core::retrieval::TurnLink {
                    kind: "tag".into(),
                    to: canonical.clone(),
                    degree,
                },
            );
        }
    }

    // One session per UTC day. Same-day turns stay a micro-scene; distinct
    // days are the sessions the chrono-lasso and event-sequence diversity
    // penalties actually separate.
    let mut by_day: std::collections::BTreeMap<String, (i64, String, Vec<vestige_core::retrieval::Turn>)> =
        std::collections::BTreeMap::new();
    for node in nodes {
        let day = node.created_at.format("%Y-%m-%d").to_string();
        let role = if node.tags.iter().any(|t| t == "assistant" || t == "system") {
            "assistant".to_string()
        } else if let Some(person) = node.tags.iter().find_map(|t| {
            t.strip_prefix("person:")
                .or_else(|| t.strip_prefix("speaker:"))
                .filter(|s| !s.trim().is_empty())
        }) {
            person.trim().to_string()
        } else {
            "user".to_string()
        };
        let entry = by_day.entry(day).or_insert_with(|| {
            (
                node.created_at.timestamp(),
                node.created_at.to_rfc3339(),
                Vec::new(),
            )
        });
        let links = links_of.remove(&node.id).unwrap_or_default();
        entry.2.push(vestige_core::retrieval::Turn {
            id: node.id.clone(),
            role,
            content: format!("{} {}", node.content, node.tags.join(" ")),
            links,
            stability: node.stability as f32,
        });
    }
    let sessions = by_day
        .into_iter()
        .map(|(day, (timestamp, date, turns))| vestige_core::retrieval::Session {
            id: day,
            date,
            timestamp,
            turns,
        })
        .collect();

    let now = chrono::Utc::now();
    let req = vestige_core::retrieval::RankRequest {
        sessions,
        queries: vec![vestige_core::retrieval::Query {
            text: query.to_string(),
            date: now.to_rfc3339(),
        }],
        top_k: top_k.max(1),
        params: vestige_core::retrieval::Params::default(),
    };

    let resp = vestige_core::retrieval::rank(&req);
    let mut scored: Vec<(String, f32)> = resp
        .results
        .into_iter()
        .next()
        .map(|q_res| {
            q_res
                .turns
                .into_iter()
                .filter_map(|hit| {
                    let mult = 1.0 - penalty_by_id.get(hit.id.as_str()).copied().unwrap_or(0.0);
                    let adj = hit.score * mult;
                    (adj > 0.0).then_some((hit.id, adj))
                })
                .collect()
        })
        .unwrap_or_default();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

fn handle_flow(storage: &Arc<Storage>, args: &Option<Value>) -> Option<Result<Value, String>> {
    let object = args.as_ref()?.as_object()?;
    let limit = object
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|n| (n as usize).max(1));
    let detail_level = object
        .get("detail_level")
        .or_else(|| object.get("detailLevel"))
        .and_then(|v| v.as_str());
    let offset = object
        .get("offset")
        .and_then(|v| v.as_u64())
        .map_or(0, |n| n as usize);
    if !object.contains_key("handle") {
        // Check if an id was passed directly (absorbing memory(action="get", id=...))
        if let Some(id) = object.get("id").and_then(|v| v.as_str()).map(str::trim)
            && !id.is_empty()
        {
            let resolution = storage.resolve_handle(id);
            return Some(Ok(handle_resolution_payload(
                storage,
                id,
                resolution,
                limit,
                detail_level,
                offset,
            )));
        }
        // A Strata log has no free-text search, so a `query` there can only be
        // an exact handle. Resolve it as one instead of refusing an id or a tag
        // because it arrived under the other parameter name.
        let query = object
            .get("query")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .unwrap_or("");
        if query.is_empty() || !crate::strata_memory::is_strata_backend(storage.as_ref()) {
            return None;
        }
        let resolution = storage.resolve_handle(query);
        if resolution.ids.is_empty() && resolution.candidates.is_empty() {
            return Some(Ok(handle_required_payload(storage, query)));
        }
        return Some(Ok(handle_resolution_payload(
            storage,
            query,
            resolution,
            limit,
            detail_level,
            offset,
        )));
    }
    let handle = object
        .get("handle")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .unwrap_or("");
    if !handle.is_empty() {
        let resolution = storage.resolve_handle(handle);
        return Some(Ok(handle_resolution_payload(
            storage,
            handle,
            resolution,
            limit,
            detail_level,
            offset,
        )));
    }
    // `handle` present but empty/non-string: free text with no handle — the
    // handle_required payload, with candidates mined from the free text.
    let free_text = object
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    Some(Ok(handle_required_payload(storage, free_text)))
}

/// Response payload for a resolved (or ambiguous, or unresolved) handle.
fn handle_resolution_payload(
    storage: &Arc<Storage>,
    handle: &str,
    resolution: HandleResolution,
    limit: Option<usize>,
    detail_level: Option<&str>,
    offset: usize,
) -> Value {
    // One memory asked for by id comes back whole; a tag comes back as a
    // page of summaries unless the caller asks for more.
    let single = resolution.ids.len() == 1;
    let detail_level = detail_level.unwrap_or(if single { "full" } else { "summary" });
    let limit = limit.or((!single).then_some(DEFAULT_TAG_LIMIT));
    if !resolution.ids.is_empty() {
        let mut indexed_nodes: Vec<(usize, String, KnowledgeNode)> = resolution
            .ids
            .iter()
            .enumerate()
            .filter_map(|(idx, id)| {
                storage
                    .get_node(id)
                    .ok()
                    .flatten()
                    .map(|node| (idx, id.clone(), node))
            })
            .collect();
        let total_nodes = indexed_nodes.len();
        if indexed_nodes.len() > 1 {
            let salience = super::memory_packet::scores(storage.as_ref());
            indexed_nodes
                .sort_by(|(_, _, a), (_, _, b)| super::memory_packet::compare(a, b, &salience));
        }
        if offset > 0 {
            indexed_nodes.drain(..offset.min(indexed_nodes.len()));
        }
        if let Some(l) = limit {
            indexed_nodes.truncate(l);
        }
        let active_ids: Vec<String> = indexed_nodes.iter().map(|(_, id, _)| id.clone()).collect();
        let nodes: Vec<Value> = indexed_nodes
            .into_iter()
            .map(|(_, _, n)| node_payload_with_detail(n, detail_level))
            .collect();
        let neighbors = one_hop_neighbors(storage, &active_ids, detail_level);
        let mut out = serde_json::json!({
            "handle": handle,
            "kind": resolution.kind.as_str(),
            "exact": resolution.exact,
            "nodes": nodes,
            "neighbors": neighbors,
        });
        if nodes.len() < total_nodes {
            out["totalNodes"] = serde_json::json!(total_nodes);
            let shown_through = offset + nodes.len();
            if shown_through < total_nodes {
                out["next"] = serde_json::json!({
                    "handle": handle,
                    "offset": shown_through,
                    "limit": limit,
                    "detail_level": detail_level,
                });
            }
        }
        if !single && detail_level == "summary" {
            out["detailLevel"] = serde_json::json!("summary");
        }
        return out;
    }
    if !resolution.candidates.is_empty() {
        let cands = candidate_json(&resolution);
        let discovered = cands.clone();
        return serde_json::json!({
            "error": "ambiguous",
            "detail": format!(
                "handle prefix matched {} record(s); pass a longer prefix or the full id",
                resolution.candidates.len()
            ),
            "handle": handle,
            "kind": resolution.kind.as_str(),
            "candidates": cands,
            "discovered_handles": discovered,
        });
    }
    // Nothing matched. The resolver's own message is more specific than the
    // generic detail when it was a too-short sha, so prefer it when present.
    let detail = resolution
        .handle_required
        .unwrap_or_else(|| vestige_core::storage::HANDLE_REQUIRED_DETAIL.to_string());
    serde_json::json!({
        "error": "handle_required",
        "detail": detail,
        "candidates": [],
        "discovered_handles": [],
    })
}

/// The `handle_required` payload for free text.
///
/// On a Strata log the candidates are what the whole string resolves to as a
/// handle, and nothing else: the text is never split into words to look for a
/// tag or an id inside it, because a word that happens to equal a tag is a
/// keyword match, not a handle the caller declared. The legacy engine also
/// tries each identifier-shaped token through its resolver.
fn handle_required_payload(storage: &Arc<Storage>, free_text: &str) -> Value {
    let mut candidates: Vec<(String, HandleKind)> = Vec::new();
    let text = free_text.trim();
    if !text.is_empty() {
        let mut queries: Vec<&str> = vec![text];
        if !crate::strata_memory::is_strata_backend(storage.as_ref()) {
            // Legacy engine: each identifier-shaped token too (bounded).
            queries.extend(
                text.split(|c: char| {
                    !(c.is_alphanumeric() || c == '_' || c == '.' || c == '/' || c == '-')
                })
                .filter(|t| t.len() >= 3)
                .take(8),
            );
        }
        for q in queries {
            let r = storage.resolve_handle(q);
            for id in &r.ids {
                push_candidate(&mut candidates, id.clone(), r.kind);
            }
            for (id, kind) in &r.candidates {
                push_candidate(&mut candidates, id.clone(), *kind);
            }
            if candidates.len() >= MAX_CANDIDATES {
                break;
            }
        }
        candidates.truncate(MAX_CANDIDATES);
    }
    let discovered: Vec<serde_json::Value> = candidates
        .iter()
        .map(|(id, kind)| {
            serde_json::json!({
                "id": id,
                "kind": kind.as_str(),
                "handle": id,
            })
        })
        .collect();
    serde_json::json!({
        "error": "handle_required",
        "detail": vestige_core::storage::HANDLE_REQUIRED_DETAIL,
        "candidates": candidates
            .iter()
            .map(|(id, kind)| serde_json::json!({"id": id, "kind": kind.as_str()}))
            .collect::<Vec<_>>(),
        "discovered_handles": discovered,
    })
}

fn push_candidate(candidates: &mut Vec<(String, HandleKind)>, id: String, kind: HandleKind) {
    if !candidates.iter().any(|(existing, _)| *existing == id) {
        candidates.push((id, kind));
    }
}

fn candidate_json(resolution: &HandleResolution) -> Vec<Value> {
    resolution
        .candidates
        .iter()
        .map(|(id, kind)| serde_json::json!({"id": id, "kind": kind.as_str()}))
        .collect()
}

fn truncate_content(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let mut s: String = text.chars().take(max_chars).collect();
        s.push('…');
        s
    }
}

/// Lean node payload for handle responses. `createdAt` lets a caller order
/// what one tag returns (the SessionStart rules hook lists newest first).
fn node_payload_with_detail(node: KnowledgeNode, detail_level: &str) -> Value {
    match detail_level {
        "brief" => serde_json::json!({
            "id": node.id,
            "stability": node.stability,
            "type": node.node_type,
            "tags": node.tags,
            "createdAt": node.created_at.to_rfc3339(),
        }),
        "summary" => serde_json::json!({
            "id": node.id,
            "stability": node.stability,
            "type": node.node_type,
            "content": truncate_content(&node.content, 240),
            "tags": node.tags,
            "createdAt": node.created_at.to_rfc3339(),
        }),
        _ => serde_json::json!({
            "id": node.id,
            "stability": node.stability,
            "type": node.node_type,
            "content": node.content,
            "tags": node.tags,
            "createdAt": node.created_at.to_rfc3339(),
        }),
    }
}

fn neighbor_node_payload(node: KnowledgeNode, max_chars: Option<usize>) -> Value {
    let content = match max_chars {
        Some(limit) => truncate_content(&node.content, limit),
        None => node.content,
    };
    serde_json::json!({
        "id": node.id,
        "type": node.node_type,
        "content": content,
        "tags": node.tags,
        "createdAt": node.created_at.to_rfc3339(),
    })
}

/// One hop over `memory_connections` typed edges, strongest first, deduped,
/// capped. Each neighbor entry carries the edge (link_type + strength +
/// direction) and a bounded neighbor node preview (unless `detail_level` is
/// `"full_neighbors"`) so multi-KB bodies are not duplicated across up to 20
/// edges.
fn one_hop_neighbors(storage: &Arc<Storage>, ids: &[String], detail_level: &str) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    let mut seen_edges: Vec<(String, String)> = Vec::new();
    let mut seen_nodes: std::collections::HashSet<String> = ids.iter().cloned().collect();
    let full_neighbors = detail_level == "full_neighbors";
    for id in ids {
        let connections = match storage.get_connections_for_memory(id) {
            Ok(conns) => conns,
            Err(_) => continue,
        };
        for edge in connections {
            if out.len() >= MAX_NEIGHBOR_EDGES {
                return out;
            }
            let key = (edge.source_id.clone(), edge.target_id.clone());
            if seen_edges.contains(&key) {
                continue;
            }
            seen_edges.push(key);
            let other = if edge.source_id == *id {
                edge.target_id.clone()
            } else {
                edge.source_id.clone()
            };
            let first_seen = seen_nodes.insert(other.clone());
            let max_chars = if full_neighbors {
                None
            } else if first_seen {
                Some(200)
            } else {
                Some(80)
            };
            let node = storage
                .get_node(&other)
                .ok()
                .flatten()
                .map(|n| neighbor_node_payload(n, max_chars));
            out.push(serde_json::json!({
                "from": edge.source_id,
                "to": edge.target_id,
                "link_type": edge.link_type,
                "strength": edge.strength,
                "direction": if edge.source_id == *id { "outgoing" } else { "incoming" },
                "node": node,
            }));
        }
    }
    // Also surface active (non-undone) Retroactive Salience Backfill edges on handle lookup
    // so inspecting either the later trigger or the earlier memory reveals the causal RSB bond.
    if out.len() < MAX_NEIGHBOR_EDGES && let Ok(receipts) = storage.salience_receipts() {
        let undone: std::collections::HashSet<String> = receipts
            .iter()
            .filter_map(|r| r.get("undo_of").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string))
            .collect();
        let id_set: std::collections::HashSet<&str> = ids.iter().map(String::as_str).collect();
        for row in receipts {
            if out.len() >= MAX_NEIGHBOR_EDGES {
                break;
            }
            if row.get("undo_of").is_some_and(|v| !v.is_null()) {
                continue;
            }
            if row.get("receipt_id").and_then(Value::as_str).is_some_and(|rid| undone.contains(rid)) {
                continue;
            }
            let mem = row.get("memory_id").and_then(Value::as_str).unwrap_or("");
            let trig = row.get("trigger_id").and_then(Value::as_str).unwrap_or("");
            if mem.is_empty() || trig.is_empty() || mem == trig {
                continue;
            }
            let (matched_self, other, direction) = if id_set.contains(trig) {
                (trig, mem, "outgoing")
            } else if id_set.contains(mem) {
                (mem, trig, "incoming")
            } else {
                continue;
            };
            let key = (trig.to_string(), mem.to_string());
            if seen_edges.contains(&key) {
                continue;
            }
            seen_edges.push(key);
            let first_seen = seen_nodes.insert(other.to_string());
            let max_chars = if full_neighbors {
                None
            } else if first_seen {
                Some(200)
            } else {
                Some(80)
            };
            let node = storage
                .get_node(other)
                .ok()
                .flatten()
                .map(|n| neighbor_node_payload(n, max_chars));
            let _ = matched_self;
            out.push(serde_json::json!({
                "from": trig,
                "to": mem,
                "link_type": "backfill",
                "strength": 1.0,
                "direction": direction,
                "receipt_id": row.get("receipt_id").cloned().unwrap_or(Value::Null),
                "shared_entity": row.get("shared_entity").cloned().unwrap_or(Value::Null),
                "node": node,
            }));
        }
    }
    out
}

/// `query` on a Strata log: there is no free-text search there, so an exact
/// id or tag passed as `query` resolves as a handle.
#[cfg(test)]
mod strata_query {
    use super::*;
    use vestige_core::IngestInput;

    fn store() -> (Arc<Storage>, tempfile::TempDir, String, String) {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let put = |content: &str, tags: &[&str]| {
            storage
                .ingest(IngestInput {
                    content: content.to_string(),
                    tags: tags.iter().map(|tag| (*tag).to_string()).collect(),
                    ..Default::default()
                })
                .unwrap()
                .id
        };
        let tagged = put(
            "RECALL_QUERY deploy timeout is two seconds",
            &["deploy-env"],
        );
        let other = put("RECALL_QUERY unrelated note", &[]);
        (storage, dir, tagged, other)
    }

    async fn recall(storage: &Arc<Storage>, args: Value) -> Result<Value, String> {
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        execute(storage, &cognitive, &OutputConfig::default(), Some(args)).await
    }

    #[tokio::test]
    async fn an_exact_id_passed_as_query_resolves_that_memory() {
        let (storage, _dir, tagged, _other) = store();
        let out = recall(&storage, serde_json::json!({ "query": tagged }))
            .await
            .unwrap();
        assert_eq!(out["kind"], "memory", "{out}");
        assert_eq!(out["exact"], true, "{out}");
        let nodes = out["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 1, "{out}");
        assert_eq!(nodes[0]["id"], serde_json::json!(tagged));
        // The same value under `handle` gives the same answer.
        let by_handle = recall(&storage, serde_json::json!({ "handle": tagged }))
            .await
            .unwrap();
        assert_eq!(out, by_handle);
    }

    #[tokio::test]
    async fn an_exact_tag_passed_as_query_resolves_the_tagged_memories() {
        let (storage, _dir, tagged, _other) = store();
        // Surrounding whitespace is trimmed, as it is for `handle`.
        let out = recall(&storage, serde_json::json!({ "query": "  deploy-env " }))
            .await
            .unwrap();
        assert_eq!(out["kind"], "tag", "{out}");
        assert_eq!(out["exact"], true, "{out}");
        let nodes = out["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 1, "{out}");
        assert_eq!(nodes[0]["id"], serde_json::json!(tagged));
        // Tags are exact and case-sensitive: another spelling is a refusal,
        // not a resolved handle.
        let wrong_case = handle_flow(
            &storage,
            &Some(serde_json::json!({ "query": "Deploy-Env" })),
        )
        .expect("a query on Strata stays in handle mode");
        let wrong_case = wrong_case.expect("case mismatch is a structured refusal");
        assert_eq!(wrong_case["error"], "handle_required", "{wrong_case}");
        assert!(wrong_case["nodes"].as_array().is_none_or(|nodes| nodes.is_empty()), "{wrong_case}");
    }

    #[tokio::test]
    async fn free_text_is_never_split_into_words_to_find_a_tag_inside_it() {
        let (storage, _dir, tagged, _other) = store();
        // One word of this sentence equals a stored tag. Before 4.2.0 that
        // word was looked up on its own and the tagged memory came back as a
        // candidate: a keyword match.
        let sentence = serde_json::json!({
            "handle": "", "query": "what changed in deploy-env yesterday",
        });
        let out = recall(&storage, sentence.clone()).await.unwrap();
        assert_eq!(out["error"], "handle_required", "{out}");
        assert_eq!(out["candidates"], serde_json::json!([]), "{out}");
        assert!(!out.to_string().contains(&tagged), "{out}");

        // The whole string as an exact tag is a declared handle.
        let whole = recall(
            &storage,
            serde_json::json!({"handle": "", "query": " deploy-env "}),
        )
        .await
        .unwrap();
        assert_eq!(
            whole["candidates"],
            serde_json::json!([{"id": tagged, "kind": "tag"}]),
            "{whole}"
        );

        // Same store, same call, same bytes.
        let again = recall(&storage, sentence).await.unwrap();
        assert_eq!(out.to_string(), again.to_string());
    }

    #[tokio::test]
    async fn free_text_passed_as_query_is_still_not_answered_as_a_handle() {
        let (storage, _dir, _tagged, _other) = store();
        let args = Some(serde_json::json!({ "query": "what is the deploy timeout" }));
        let flow = handle_flow(&storage, &args).expect("free text on Strata stays in handle mode");
        let flow = flow.expect("free text is a structured refusal");
        assert_eq!(flow["error"], "handle_required", "{flow}");
        let out = recall(
            &storage,
            serde_json::json!({ "query": "what is the deploy timeout" }),
        )
        .await
        .expect("free text returns the handle refusal");
        let text = out.to_string();
        assert!(text.contains("handle_required"), "{text}");
        assert!(!text.contains("RECALL_QUERY"), "{text}");
    }

    /// The refusal is a refusal. Before 4.2.1 it was wrapped in the store's
    /// `Init` error and read "Initialization error: similarity_disabled: ...",
    /// as if the store had failed to start.
    #[tokio::test]
    async fn free_text_refusal_starts_with_its_code_and_is_not_an_init_failure() {
        let (storage, _dir, _tagged, _other) = store();
        let out = recall(
            &storage,
            serde_json::json!({ "query": "what is the deploy timeout" }),
        )
        .await
        .expect("free text is a structured refusal on a Strata log");
        assert_eq!(out["error"], "handle_required", "{out}");
        let text = out.to_string();
        assert!(
            !text.to_lowercase().contains("initialization error"),
            "{text}"
        );
        assert!(text.contains("pass a memory id") || text.contains("exact tag"), "{text}");
    }

    #[tokio::test]
    async fn recall_respects_limit_and_detail_level_on_tag_handles() {
        let (storage, _dir, _tagged, _other) = store();
        let second = storage
            .ingest(IngestInput {
                content: "second deploy-env memory with extra payload".to_string(),
                tags: vec!["deploy-env".to_string()],
                ..Default::default()
            })
            .unwrap()
            .id;
        let out = recall(
            &storage,
            serde_json::json!({
                "handle": "deploy-env",
                "limit": 1,
                "detail_level": "brief"
            }),
        )
        .await
        .unwrap();
        let nodes = out["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(out["totalNodes"], 2);
        assert_eq!(
            nodes[0]["id"], second,
            "tag recall must sort newest first before cutting to limit"
        );
        assert!(nodes[0].get("content").is_none());
    }

    #[tokio::test]
    async fn recall_neighbor_preview_truncates_by_default_and_full_neighbors_expands() {
        let (storage, _dir, tagged, _other) = store();
        let long_neighbor_text = "N".repeat(320);
        let neighbor_id = storage
            .ingest(IngestInput {
                content: long_neighbor_text.clone(),
                tags: vec!["neighbor-tag".to_string()],
                ..Default::default()
            })
            .unwrap()
            .id;
        let now = chrono::Utc::now();
        storage
            .save_connection(&vestige_core::ConnectionRecord {
                source_id: tagged.clone(),
                target_id: neighbor_id.clone(),
                strength: 0.9,
                link_type: "derived_from".to_string(),
                created_at: now,
                last_activated: now,
                activation_count: 1,
            })
            .unwrap();

        let default_out = recall(&storage, serde_json::json!({ "handle": tagged }))
            .await
            .unwrap();
        let preview = default_out["neighbors"][0]["node"]["content"]
            .as_str()
            .unwrap();
        assert_eq!(preview.chars().count(), 201, "200 chars + ellipsis");
        assert!(preview.ends_with('…'));

        let full_out = recall(
            &storage,
            serde_json::json!({
                "handle": tagged,
                "detail_level": "full_neighbors"
            }),
        )
        .await
        .unwrap();
        let full_neighbor = full_out["neighbors"][0]["node"]["content"]
            .as_str()
            .unwrap();
        assert_eq!(full_neighbor, long_neighbor_text);
    }

    #[test]
    fn recall_splits_memories_by_utc_day() {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let put = |content: &str| {
            storage
                .ingest(IngestInput {
                    content: content.to_string(),
                    ..Default::default()
                })
                .unwrap()
                .id
        };
        let jan_a = put("We shipped the ledger format to production.");
        let jan_b = put("The ledger format deploy finished before noon.");
        let mar = put("We rolled the ledger format back after the audit.");
        let jun = put("We restored the ledger format with signed receipts.");
        drop(storage);

        {
            let mut store = strata_store::StrataStore::open(dir.path()).unwrap();
            let ms = |stamp: &str| {
                chrono::DateTime::parse_from_rfc3339(stamp)
                    .unwrap()
                    .timestamp_millis()
            };
            store
                .set_created_at(&jan_a, ms("2024-01-04T10:00:00Z"))
                .unwrap();
            store
                .set_created_at(&jan_b, ms("2024-01-04T18:00:00Z"))
                .unwrap();
            store
                .set_created_at(&mar, ms("2024-03-02T11:00:00Z"))
                .unwrap();
            store
                .set_created_at(&jun, ms("2024-06-09T11:00:00Z"))
                .unwrap();
        }

        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let ranked = rank_nodes_lexical(
            storage.as_ref(),
            "What is the chronological order of the ledger format changes?",
            3,
        );
        assert_eq!(ranked.len(), 3, "{ranked:?}");
        let days: Vec<String> = ranked
            .iter()
            .map(|(id, _)| {
                storage
                    .get_node(id)
                    .unwrap()
                    .unwrap()
                    .created_at
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();
        let unique: std::collections::HashSet<_> = days.iter().cloned().collect();
        assert_eq!(
            unique.len(),
            3,
            "top hits collapsed onto one UTC day: {days:?} ranked={ranked:?}"
        );
        let january = ranked.iter().filter(|(id, _)| id == &jan_a || id == &jan_b).count();
        assert_eq!(january, 1, "same-day pair both survived top-3: {ranked:?}");
    }
}
