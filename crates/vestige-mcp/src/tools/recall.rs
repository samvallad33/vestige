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
        if let Some(id) = object.get("id").and_then(|v| v.as_str()).map(str::trim) {
            if !id.is_empty() {
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
            // Strata returns structured handle discovery refusal on free-text query
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
        .map(|(id, kind)| serde_json::json!({
            "id": id,
            "kind": kind.as_str(),
            "handle": id,
        }))
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
            "type": node.node_type,
            "tags": node.tags,
            "createdAt": node.created_at.to_rfc3339(),
        }),
        "summary" => serde_json::json!({
            "id": node.id,
            "type": node.node_type,
            "content": truncate_content(&node.content, 240),
            "tags": node.tags,
            "createdAt": node.created_at.to_rfc3339(),
        }),
        _ => serde_json::json!({
            "id": node.id,
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
        // Tags are exact and case-sensitive: another spelling is not a handle.
        assert!(
            handle_flow(
                &storage,
                &Some(serde_json::json!({ "query": "Deploy-Env" }))
            )
            .is_none()
        );
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
        assert!(
            handle_flow(&storage, &args).is_none(),
            "free text must fall through to the search path"
        );
        let out = recall(
            &storage,
            serde_json::json!({ "query": "what is the deploy timeout" }),
        )
        .await;
        let text = match out {
            Ok(value) => value.to_string(),
            Err(err) => err,
        };
        assert!(text.contains("similarity_disabled"), "{text}");
        assert!(!text.contains("RECALL_QUERY"), "{text}");
    }

    /// The refusal is a refusal. Before 4.2.1 it was wrapped in the store's
    /// `Init` error and read "Initialization error: similarity_disabled: ...",
    /// as if the store had failed to start.
    #[tokio::test]
    async fn free_text_refusal_starts_with_its_code_and_is_not_an_init_failure() {
        let (storage, _dir, _tagged, _other) = store();
        let err = recall(
            &storage,
            serde_json::json!({ "query": "what is the deploy timeout" }),
        )
        .await
        .expect_err("free text is refused on a Strata log");
        assert!(err.starts_with("similarity_disabled: "), "{err}");
        assert!(
            !err.to_lowercase().contains("initialization error"),
            "{err}"
        );
        assert!(err.contains("pass an exact handle"), "{err}");
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
}
