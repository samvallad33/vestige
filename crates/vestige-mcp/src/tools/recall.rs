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
                    "description": "[handle mode] Exact handle: a full memory id or an exact tag (case-sensitive); a unique id prefix of 8+ characters also resolves. Commit sha, file path, symbol, test name, run id and tool-call id handles resolve on the legacy engine only. No fuzzy or lexical matching. Passing this key (even empty) switches recall to handle mode: a resolvable handle returns the node payloads plus one-hop recorded-edge neighbors; anything else returns the handle_required error with exact candidates."
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

/// Entry gate for handle mode. Returns `None` when the args carry no `handle`
/// key (legacy mode dispatch proceeds unchanged); otherwise the complete
/// handle-mode response.
fn handle_flow(storage: &Arc<Storage>, args: &Option<Value>) -> Option<Result<Value, String>> {
    let object = args.as_ref()?.as_object()?;
    if !object.contains_key("handle") {
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
            return None; // not a handle: the search path answers similarity_disabled as before
        }
        return Some(Ok(handle_resolution_payload(storage, query, resolution)));
    }
    let handle = object
        .get("handle")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .unwrap_or("");
    if !handle.is_empty() {
        let resolution = storage.resolve_handle(handle);
        return Some(Ok(handle_resolution_payload(storage, handle, resolution)));
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
) -> Value {
    if !resolution.ids.is_empty() {
        let nodes: Vec<Value> = resolution
            .ids
            .iter()
            .filter_map(|id| storage.get_node(id).ok().flatten())
            .map(node_payload)
            .collect();
        let neighbors = one_hop_neighbors(storage, &resolution.ids);
        return serde_json::json!({
            "handle": handle,
            "kind": resolution.kind.as_str(),
            "exact": resolution.exact,
            "nodes": nodes,
            "neighbors": neighbors,
        });
    }
    if !resolution.candidates.is_empty() {
        return serde_json::json!({
            "error": "ambiguous",
            "detail": format!(
                "handle prefix matched {} record(s); pass a longer prefix or the full id",
                resolution.candidates.len()
            ),
            "handle": handle,
            "kind": resolution.kind.as_str(),
            "candidates": candidate_json(&resolution),
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
    })
}

/// The `handle_required` payload for free text: candidates mined from the
/// text's identifier-shaped tokens via the resolver (exact/prefix only).
fn handle_required_payload(storage: &Arc<Storage>, free_text: &str) -> Value {
    let mut candidates: Vec<(String, HandleKind)> = Vec::new();
    let text = free_text.trim();
    if !text.is_empty() {
        // Whole string first, then each identifier-shaped token (bounded).
        let mut queries: Vec<&str> = vec![text];
        queries.extend(
            text.split(|c: char| {
                !(c.is_alphanumeric() || c == '_' || c == '.' || c == '/' || c == '-')
            })
            .filter(|t| t.len() >= 3)
            .take(8),
        );
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
    serde_json::json!({
        "error": "handle_required",
        "detail": vestige_core::storage::HANDLE_REQUIRED_DETAIL,
        "candidates": candidates
            .iter()
            .map(|(id, kind)| serde_json::json!({"id": id, "kind": kind.as_str()}))
            .collect::<Vec<_>>(),
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

/// Lean node payload for handle responses.
fn node_payload(node: KnowledgeNode) -> Value {
    serde_json::json!({
        "id": node.id,
        "type": node.node_type,
        "content": node.content,
        "tags": node.tags,
    })
}

/// One hop over `memory_connections` typed edges, strongest first, deduped,
/// capped. Each neighbor entry carries the edge (link_type + strength +
/// direction) and the neighbor node payload.
fn one_hop_neighbors(storage: &Arc<Storage>, ids: &[String]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    let mut seen_edges: Vec<(String, String)> = Vec::new();
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
            let node = storage.get_node(&other).ok().flatten().map(node_payload);
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

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;

    #[test]
    fn test_schema_has_mode_and_no_required() {
        let s = schema();
        let modes = s["properties"]["mode"]["enum"].as_array().unwrap();
        assert_eq!(modes.len(), 3);
        assert_eq!(s["properties"]["mode"]["default"], "lookup");
        // query must NOT be globally required (contradictions uses topic).
        assert!(
            s.get("required").is_none(),
            "recall must not globally require 'query'"
        );
        // lookup params carried over from search schema.
        assert!(s["properties"]["limit"].is_object());
        assert!(s["properties"]["detail_level"].is_object());
    }

    #[tokio::test]
    async fn test_lookup_is_default_and_resolves() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let oc = OutputConfig::default();
        // No mode → lookup → behaves like search (query required by search).
        let args = Some(serde_json::json!({ "query": "anything" }));
        let r = execute(&storage, &cognitive, &oc, args).await;
        assert!(r.is_ok(), "default lookup should resolve: {r:?}");
    }

    #[tokio::test]
    async fn test_contradictions_mode_resolves_without_query() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let oc = OutputConfig::default();
        // contradictions uses topic, not query — must resolve with no query.
        let args = Some(serde_json::json!({ "mode": "contradictions" }));
        let r = execute(&storage, &cognitive, &oc, args).await;
        assert!(r.is_ok(), "contradictions mode should resolve: {r:?}");
    }

    // ---- handle mode ----

    use vestige_core::{ConnectionRecord, IngestInput};

    fn connect(storage: &Arc<Storage>, from: &str, to: &str, link_type: &str, strength: f64) {
        let now = chrono::Utc::now();
        storage
            .save_connection(&ConnectionRecord {
                source_id: from.to_string(),
                target_id: to.to_string(),
                strength,
                link_type: link_type.to_string(),
                created_at: now,
                last_activated: now,
                activation_count: 0,
            })
            .unwrap();
    }

    const SHA_A: &str = "0123456789abcdef0123456789abcdef01234567";
    const SHA_B: &str = "0123456789fffffffedcba9876543210fedcba98";

    async fn handle_store() -> (Arc<Storage>, tempfile::TempDir, String) {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("handle.db"))).unwrap();
        let memory = storage
            .ingest(IngestInput {
                content: "Set API_TIMEOUT=2 in the deploy env".into(),
                tags: vec!["deploy-env".into()],
                ..Default::default()
            })
            .unwrap();
        let cause = storage
            .ingest(IngestInput {
                content: "commit flipping API_TIMEOUT default".into(),
                ..Default::default()
            })
            .unwrap();
        let commit_a = storage
            .ingest(IngestInput {
                content: format!("commit {SHA_A} speed up cold starts\nfiles: src/main.rs"),
                tags: vec!["git-commit".into()],
                ..Default::default()
            })
            .unwrap();
        let _commit_b = storage
            .ingest(IngestInput {
                content: format!("commit {SHA_B} fix the thing"),
                tags: vec!["git-commit".into()],
                ..Default::default()
            })
            .unwrap();
        connect(&storage, &memory.id, &cause.id, "causal", 0.9);
        connect(&storage, &commit_a.id, &memory.id, "temporal", 0.4);
        let memory_id = memory.id.clone();
        (storage, dir, memory_id)
    }

    #[tokio::test]
    async fn handle_resolves_uuid_and_returns_one_hop_neighbors() {
        let (storage, _dir, memory_id) = handle_store().await;
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let oc = OutputConfig::default();
        let out = execute(
            &storage,
            &cognitive,
            &oc,
            Some(serde_json::json!({ "handle": memory_id })),
        )
        .await
        .unwrap();
        assert_eq!(out["kind"], "memory");
        assert_eq!(out["exact"], true);
        assert_eq!(out["nodes"][0]["id"], serde_json::json!(memory_id));
        let neighbors = out["neighbors"].as_array().unwrap();
        assert_eq!(neighbors.len(), 2, "one hop, both directions: {out}");
        // Strongest edge first (causal 0.9 before temporal 0.4).
        assert_eq!(neighbors[0]["link_type"], "causal");
        assert_eq!(neighbors[0]["direction"], "outgoing");
        assert!(
            neighbors[0]["node"].is_object(),
            "neighbor node payload included"
        );
        assert_eq!(neighbors[1]["link_type"], "temporal");
        assert_eq!(neighbors[1]["direction"], "incoming");
    }

    #[tokio::test]
    async fn handle_commit_sha_prefix_and_ambiguity() {
        let (storage, _dir, _memory_id) = handle_store().await;
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let oc = OutputConfig::default();
        // Full sha: exact commit resolution.
        let out = execute(
            &storage,
            &cognitive,
            &oc,
            Some(serde_json::json!({ "handle": SHA_A })),
        )
        .await
        .unwrap();
        assert_eq!(out["kind"], "commit");
        assert_eq!(out["exact"], true);
        // Shared 7-char prefix across both commits: ambiguous candidates.
        let out = execute(
            &storage,
            &cognitive,
            &oc,
            Some(serde_json::json!({ "handle": &SHA_A[..7] })),
        )
        .await
        .unwrap();
        assert_eq!(out["error"], "ambiguous");
        assert_eq!(out["candidates"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn free_text_without_handle_returns_handle_required_with_candidates() {
        let (storage, _dir, _memory_id) = handle_store().await;
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let oc = OutputConfig::default();
        // `handle` key present but empty: free text -> handle_required, with
        // candidates mined from the text (API_TIMEOUT resolves the memory).
        let out = execute(
            &storage,
            &cognitive,
            &oc,
            Some(serde_json::json!({ "handle": "", "query": "what did API_TIMEOUT change" })),
        )
        .await
        .unwrap();
        assert_eq!(out["error"], "handle_required");
        assert!(
            out["detail"]
                .as_str()
                .unwrap()
                .contains("recall is handle-based"),
            "detail must be the canonical guidance: {out}"
        );
        let candidates = out["candidates"].as_array().unwrap();
        assert!(
            !candidates.is_empty(),
            "identifier token must yield candidates: {out}"
        );
        assert!(
            candidates
                .iter()
                .all(|c| c["kind"] == "symbol" || c["kind"] == "memory")
        );
        // Pure prose with no identifiers: the same error, empty candidates.
        let out = execute(
            &storage,
            &cognitive,
            &oc,
            Some(serde_json::json!({ "handle": "", "query": "how did the build break" })),
        )
        .await
        .unwrap();
        assert_eq!(out["error"], "handle_required");
        assert_eq!(out["candidates"].as_array().unwrap().len(), 0);
        // An unresolvable non-empty handle: same payload shape.
        let out = execute(
            &storage,
            &cognitive,
            &oc,
            Some(serde_json::json!({ "handle": "no_such_symbol_anywhere" })),
        )
        .await
        .unwrap();
        assert_eq!(out["error"], "handle_required");
    }

    #[tokio::test]
    async fn no_handle_key_keeps_legacy_mode_dispatch() {
        let (storage, _dir, _memory_id) = handle_store().await;
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let oc = OutputConfig::default();
        // No `handle` key: the legacy hot path still answers a plain query.
        let out = execute(
            &storage,
            &cognitive,
            &oc,
            Some(serde_json::json!({ "query": "API_TIMEOUT" })),
        )
        .await;
        assert!(out.is_ok(), "legacy lookup must keep working: {out:?}");
    }
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
}
