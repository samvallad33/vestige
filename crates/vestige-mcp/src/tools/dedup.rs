//! Find Duplicates Tool
//!
//! Detects duplicate and near-duplicate memory clusters using
//! cosine similarity on stored embeddings. Uses union-find for
//! efficient clustering.

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
use serde::Deserialize;
use serde_json::Value;
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
use std::collections::HashMap;
use std::sync::Arc;

use vestige_core::Storage;
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
use vestige_core::cosine_similarity;

/// Input schema for find_duplicates tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "similarity_threshold": {
                "type": "number",
                "description": "Minimum cosine similarity to consider as duplicate (0.0-1.0, default: 0.80)",
                "default": 0.80,
                "minimum": 0.5,
                "maximum": 1.0
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of duplicate clusters to return (default: 20)",
                "default": 20,
                "minimum": 1,
                "maximum": 100
            },
            "tags": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Optional: only check memories with these tags (ANY match)"
            }
        }
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
struct DedupArgs {
    #[serde(alias = "similarity_threshold")]
    similarity_threshold: Option<f64>,
    limit: Option<usize>,
    tags: Option<Vec<String>>,
}

/// Simple union-find for clustering
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
    }
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    {
        let args: DedupArgs = match args {
            Some(v) => {
                serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?
            }
            None => DedupArgs {
                similarity_threshold: None,
                limit: None,
                tags: None,
            },
        };
        let threshold = args.similarity_threshold.unwrap_or(0.80) as f32;
        let limit = args.limit.unwrap_or(20);
        let tag_filter = args.tags.unwrap_or_default();

        // Load all embeddings
        let all_embeddings = storage
            .get_all_embeddings()
            .map_err(|e| format!("Failed to load embeddings: {}", e))?;

        if all_embeddings.is_empty() {
            return Ok(serde_json::json!({
                "clusters": [],
                "totalMemories": 0,
                "totalWithEmbeddings": 0,
                "message": "No embeddings found. Run consolidation first."
            }));
        }

        // Load nodes for metadata (content preview, retention, tags)
        let mut all_nodes = Vec::new();
        let mut offset = 0;
        loop {
            let batch = storage
                .get_all_nodes(500, offset)
                .map_err(|e| format!("Failed to load nodes: {}", e))?;
            let batch_len = batch.len();
            all_nodes.extend(batch);
            if batch_len < 500 {
                break;
            }
            offset += 500;
        }

        // Build node lookup
        let node_map: HashMap<String, &vestige_core::KnowledgeNode> =
            all_nodes.iter().map(|n| (n.id.clone(), n)).collect();

        // Filter by tags if specified
        let filtered_embeddings: Vec<(usize, &String, &Vec<f32>)> = all_embeddings
            .iter()
            .enumerate()
            .filter(|(_, (id, _))| {
                if tag_filter.is_empty() {
                    return true;
                }
                if let Some(node) = node_map.get(id) {
                    tag_filter.iter().any(|t| node.tags.contains(t))
                } else {
                    false
                }
            })
            .map(|(i, (id, vec))| (i, id, vec))
            .collect();

        let n = filtered_embeddings.len();

        if n > 2000 {
            return Ok(serde_json::json!({
                "warning": format!("Too many memories to scan ({} with embeddings). Filter by tags to reduce scope.", n),
                "totalMemories": all_nodes.len(),
                "totalWithEmbeddings": n
            }));
        }

        // O(n^2) pairwise similarity + union-find clustering
        let mut uf = UnionFind::new(n);
        let mut similarities: Vec<(usize, usize, f32)> = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine_similarity(filtered_embeddings[i].2, filtered_embeddings[j].2);
                if sim >= threshold {
                    uf.union(i, j);
                    similarities.push((i, j, sim));
                }
            }
        }

        // Group into clusters
        let mut cluster_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = uf.find(i);
            cluster_map.entry(root).or_default().push(i);
        }

        // Only keep clusters with >1 member, sorted by size descending
        let mut clusters: Vec<Vec<usize>> =
            cluster_map.into_values().filter(|c| c.len() > 1).collect();
        clusters.sort_by_key(|b| std::cmp::Reverse(b.len()));
        clusters.truncate(limit);

        // Build similarity lookup for formatting
        let mut sim_lookup: HashMap<(usize, usize), f32> = HashMap::new();
        for &(i, j, sim) in &similarities {
            sim_lookup.insert((i, j), sim);
            sim_lookup.insert((j, i), sim);
        }

        // Format output
        let cluster_results: Vec<Value> = clusters
            .iter()
            .enumerate()
            .map(|(ci, members)| {
                let anchor = members[0];
                let member_results: Vec<Value> = members
                    .iter()
                    .map(|&idx| {
                        let id = &filtered_embeddings[idx].1;
                        let node = node_map.get(id.as_str());
                        let content_preview = node
                            .map(|n| {
                                let c = n.content.replace('\n', " ");
                                if c.len() > 120 {
                                    format!("{}...", &c[..c.floor_char_boundary(120)])
                                } else {
                                    c
                                }
                            })
                            .unwrap_or_default();

                        let sim_to_anchor = if idx == anchor {
                            1.0
                        } else {
                            sim_lookup
                                .get(&(anchor, idx))
                                .copied()
                                .unwrap_or(0.0)
                        };

                        serde_json::json!({
                            "id": id,
                            "contentPreview": content_preview,
                            "retention": node.map(|n| n.retention_strength).unwrap_or(0.0),
                            "createdAt": node.map(|n| n.created_at.to_rfc3339()).unwrap_or_default(),
                            "tags": node.map(|n| &n.tags).unwrap_or(&vec![]),
                            "similarityToAnchor": format!("{:.3}", sim_to_anchor)
                        })
                    })
                    .collect();

                serde_json::json!({
                    "clusterId": ci,
                    "size": members.len(),
                    "members": member_results,
                    "suggestedAction": if members.len() > 3 { "review" } else { "merge" }
                })
            })
            .collect();

        Ok(serde_json::json!({
            "clusters": cluster_results,
            "totalClusters": cluster_results.len(),
            "totalMemories": all_nodes.len(),
            "totalWithEmbeddings": n,
            "threshold": threshold,
            "pairsChecked": n * (n - 1) / 2
        }))
    }

    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    {
        let _ = storage;
        let _ = args;
        Ok(serde_json::json!({
            "error": "Embeddings feature not enabled. Cannot compute similarities.",
            "clusters": []
        }))
    }
}

// ============================================================================
// UNIFIED `dedup` TOOL (v2.2 — Tool Consolidation)
//
// Folds the 8 former dedup/merge tools into a single action-dispatched surface:
//   action = scan (default) | plan_merge | plan_supersede | apply | undo
//          | tag_rename | tag_merge | protect | policy
//
// `scan` combines cosine-similarity duplicate clusters (this module's
// `execute`) with Fellegi-Sunter merge candidates (`merge::merge_candidates`),
// returning both in separate fields. The mutate/preview/reverse actions delegate
// to `super::merge::execute` verbatim, preserving plan_id → apply → undo,
// confirm-gating, and bitemporal-never-delete byte-for-byte.
// ============================================================================

/// Discriminated-union schema for the unified `dedup` tool.
pub fn unified_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["scan", "plan_merge", "plan_supersede", "apply", "undo", "tag_rename", "tag_merge", "protect", "policy"],
                "default": "scan",
                "description": "What to do. 'scan' (default): surface duplicate clusters (cosine) AND merge candidates (Fellegi-Sunter), read-only. 'plan_merge'/'plan_supersede': preview a reversible memory plan. 'apply': execute a plan_id. 'undo': reverse a prior memory or tag operation (omit operation_id to list the reflog). 'tag_rename'/'tag_merge': exact, scoped, preview-token-gated tag maintenance. 'protect': pin a memory. 'policy': get/set Fellegi-Sunter thresholds."
            },
            "similarity_threshold": {
                "type": "number",
                "description": "[scan] Minimum cosine similarity for duplicate clusters (0.5-1.0, default 0.80).",
                "minimum": 0.5, "maximum": 1.0
            },
            "limit": {
                "type": "integer",
                "description": "[scan] Max clusters/candidates to return (default 20).",
                "minimum": 1, "maximum": 100
            },
            "tags": {
                "type": "array", "items": { "type": "string" },
                "description": "[scan] Optional: only consider memories with these tags (ANY match)."
            },
            "member_ids": {
                "type": "array", "items": { "type": "string" },
                "description": "[plan_merge] IDs of memories to merge (>= 2). Survivor kept; rest bitemporally invalidated."
            },
            "survivor_id": { "type": "string", "description": "[plan_merge] Optional: which member to keep (defaults to highest-retention)." },
            "old_id": { "type": "string", "description": "[plan_supersede] Memory being superseded (kept, marked invalid)." },
            "new_id": { "type": "string", "description": "[plan_supersede] Memory that supersedes the old one." },
            "plan_id": { "type": "string", "description": "[apply] ID of a plan produced by plan_merge/plan_supersede." },
            "confirm": { "type": "boolean", "default": false, "description": "[apply/tag_*] Explicit mutation confirmation. Tag actions always preview when false and require the returned preview_token when true." },
            "operation_id": { "type": "string", "description": "[undo] Operation to reverse. Omit to list the reflog." },
            "source_tag": { "type": "string", "description": "[tag_rename] Exact source tag to rename." },
            "source_tags": { "type": "array", "items": { "type": "string" }, "minItems": 2, "maxItems": 50, "description": "[tag_merge] Two or more exact source tags to merge." },
            "target_tag": { "type": "string", "description": "[tag_rename/tag_merge] Exact destination tag." },
            "scope": { "type": "string", "default": "user", "description": "[tag_rename/tag_merge] Project scope. Defaults to the isolated 'user' scope." },
            "all_scopes": { "type": "boolean", "default": false, "description": "[tag_rename/tag_merge] Explicitly operate across every scope. Default false." },
            "preview_token": { "type": "string", "description": "[tag_rename/tag_merge confirm=true] Token returned by the immediately preceding matching preview." },
            "reason": { "type": "string", "minLength": 1, "maxLength": 1000, "description": "[tag_rename/tag_merge confirm=true] Required durable audit reason." },
            "id": { "type": "string", "description": "[protect] Memory id to protect/unprotect." },
            "protected": { "type": "boolean", "default": true, "description": "[protect] true to pin, false to unpin." },
            "match_threshold": { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "[policy] Score >= this => 'match'." },
            "possible_threshold": { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "[policy] Score in [possible, match) => review." },
            "auto_apply": { "type": "boolean", "description": "[policy] Allow 'match' plans to apply without confirm. Default false." }
        }
    })
}

/// Unified dispatcher for the `dedup` tool. Routes on `action` (default `scan`).
pub async fn execute_unified(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let action = args
        .as_ref()
        .and_then(|a| a.get("action"))
        .and_then(|v| v.as_str())
        .unwrap_or("scan")
        .to_string();

    match action.as_str() {
        "scan" => {
            // Cosine-similarity duplicate clusters (this module).
            let clusters = execute(storage, args.clone()).await?;
            // Fellegi-Sunter merge candidates (merge module, name-dispatched).
            let candidates =
                super::merge::execute(storage, "merge_candidates", args.clone()).await?;
            Ok(serde_json::json!({
                "action": "scan",
                "duplicateClusters": clusters,
                "mergeCandidates": candidates,
                "nextStep": "Use action='plan_merge' (member_ids) or action='plan_supersede' (old_id,new_id) to preview a reversible plan, then action='apply' (plan_id)."
            }))
        }
        "plan_merge" => super::merge::execute(storage, "plan_merge", args).await,
        "plan_supersede" => super::merge::execute(storage, "plan_supersede", args).await,
        "apply" => super::merge::execute(storage, "apply_plan", args).await,
        "undo" => super::merge::execute(storage, "merge_undo", args).await,
        "tag_rename" => execute_tag_mutation(storage, args, false),
        "tag_merge" => execute_tag_mutation(storage, args, true),
        "protect" => super::merge::execute(storage, "protect", args).await,
        "policy" => super::merge::execute(storage, "merge_policy", args).await,
        other => Err(format!(
            "Unknown dedup action '{other}'. Use scan|plan_merge|plan_supersede|apply|undo|tag_rename|tag_merge|protect|policy."
        )),
    }
}

fn execute_tag_mutation(
    storage: &Arc<Storage>,
    args: Option<Value>,
    merge: bool,
) -> Result<Value, String> {
    let args = args
        .as_ref()
        .and_then(Value::as_object)
        .ok_or("tag action arguments must be an object")?;
    let target_tag = args
        .get("target_tag")
        .and_then(Value::as_str)
        .ok_or("target_tag is required")?;
    let source_tags = if merge {
        let tags = args
            .get("source_tags")
            .and_then(Value::as_array)
            .ok_or("source_tags is required for tag_merge")?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or("source_tags must contain only strings")
            })
            .collect::<Result<Vec<_>, _>>()?;
        if tags.len() < 2 {
            return Err("tag_merge requires at least two source_tags".into());
        }
        tags
    } else {
        vec![
            args.get("source_tag")
                .and_then(Value::as_str)
                .ok_or("source_tag is required for tag_rename")?
                .to_string(),
        ]
    };
    let all_scopes = args
        .get("all_scopes")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let scope = if all_scopes {
        None
    } else {
        Some(
            args.get("scope")
                .and_then(Value::as_str)
                .unwrap_or(vestige_core::DEFAULT_MEMORY_SCOPE),
        )
    };
    let confirm = args
        .get("confirm")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let op_type = if merge { "tag_merge" } else { "tag_rename" };

    if !confirm {
        let mut preview = storage
            .preview_tag_mutation(&source_tags, target_tag, scope)
            .map_err(|error| error.to_string())?;
        if let Some(object) = preview.as_object_mut() {
            object.insert("action".into(), Value::String(op_type.into()));
            object.insert(
                "nextStep".into(),
                Value::String(format!(
                    "Review this preview, then call dedup action='{op_type}' with confirm=true, this preview_token, and a nonempty reason."
                )),
            );
        }
        return Ok(preview);
    }

    let preview_token = args
        .get("preview_token")
        .and_then(Value::as_str)
        .ok_or("preview_token is required when confirm=true")?;
    let reason = args
        .get("reason")
        .and_then(Value::as_str)
        .ok_or("reason is required when confirm=true")?;
    let operation = storage
        .apply_tag_mutation(
            &source_tags,
            target_tag,
            scope,
            preview_token,
            op_type,
            reason,
        )
        .map_err(|error| error.to_string())?;
    let operation_id = operation.id.clone();
    let operation_scope = operation
        .signals
        .as_ref()
        .and_then(|signals| signals.get("scope"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let operation_all_scopes = operation
        .signals
        .as_ref()
        .and_then(|signals| signals.get("allScopes"))
        .and_then(Value::as_bool)
        .unwrap_or(all_scopes);
    let operation_source_tags = operation
        .signals
        .as_ref()
        .and_then(|signals| signals.get("sourceTags"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!(source_tags));
    let operation_target_tag = operation
        .signals
        .as_ref()
        .and_then(|signals| signals.get("targetTag"))
        .cloned()
        .unwrap_or_else(|| Value::String(target_tag.to_string()));
    Ok(serde_json::json!({
        "action": op_type,
        "status": "applied",
        "operationId": operation_id,
        "affectedMemoryCount": operation.affected_ids.len(),
        "affectedMemoryIds": operation.affected_ids,
        "scope": operation_scope,
        "allScopes": operation_all_scopes,
        "sourceTags": operation_source_tags,
        "targetTag": operation_target_tag,
        "reason": operation.reason,
        "reversible": true,
        "nextStep": format!("To reverse this operation without overwriting later tag edits, call dedup action='undo' with operation_id='{operation_id}'."),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vestige_core::IngestInput;

    #[test]
    fn test_schema() {
        let schema = schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["similarity_threshold"].is_object());
    }

    #[test]
    fn test_unified_schema() {
        let schema = unified_schema();
        assert_eq!(schema["type"], "object");
        let actions = schema["properties"]["action"]["enum"].as_array().unwrap();
        assert_eq!(actions.len(), 9);
        assert_eq!(schema["properties"]["action"]["default"], "scan");
    }

    #[tokio::test]
    async fn test_unified_scan_empty_storage() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        let storage = Arc::new(storage);
        // Default action (scan) on empty storage must not error.
        let result = execute_unified(&storage, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn tag_rename_requires_preview_token_and_supports_agent_visible_undo() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = Arc::new(Storage::new(Some(dir.path().join("test.db"))).unwrap());
        let node = storage
            .ingest(IngestInput {
                content: "MCP tag rename fixture".to_string(),
                tags: vec!["old".to_string(), "keep".to_string()],
                ..Default::default()
            })
            .unwrap();

        let preview = execute_unified(
            &storage,
            Some(serde_json::json!({
                "action": "tag_rename",
                "source_tag": "old",
                "target_tag": "new",
                "scope": " user "
            })),
        )
        .await
        .unwrap();
        assert_eq!(preview["requiresConfirmation"], true);
        assert_eq!(preview["affectedMemoryCount"], 1);
        assert_eq!(
            storage.get_node(&node.id).unwrap().unwrap().tags,
            vec!["old", "keep"]
        );

        let missing_token = execute_unified(
            &storage,
            Some(serde_json::json!({
                "action": "tag_rename",
                "source_tag": "old",
                "target_tag": "new",
                "scope": " user ",
                "confirm": true,
                "reason": "normalize tag"
            })),
        )
        .await;
        assert!(missing_token.unwrap_err().contains("preview_token"));

        let applied = execute_unified(
            &storage,
            Some(serde_json::json!({
                "action": "tag_rename",
                "source_tag": "old",
                "target_tag": "new",
                "scope": " user ",
                "confirm": true,
                "preview_token": preview["previewToken"],
                "reason": "normalize tag"
            })),
        )
        .await
        .unwrap();
        assert_eq!(applied["status"], "applied");
        assert_eq!(applied["scope"], "user");
        assert_eq!(applied["sourceTags"], serde_json::json!(["old"]));
        assert_eq!(applied["targetTag"], "new");
        assert_eq!(
            storage.get_node(&node.id).unwrap().unwrap().tags,
            vec!["new", "keep"]
        );

        let undone = execute_unified(
            &storage,
            Some(serde_json::json!({
                "action": "undo",
                "operation_id": applied["operationId"]
            })),
        )
        .await
        .unwrap();
        assert_eq!(undone["status"], "reverted");
        assert_eq!(
            storage.get_node(&node.id).unwrap().unwrap().tags,
            vec!["old", "keep"]
        );
    }

    #[tokio::test]
    async fn tag_merge_requires_multiple_sources() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = Arc::new(Storage::new(Some(dir.path().join("test.db"))).unwrap());
        let error = execute_unified(
            &storage,
            Some(serde_json::json!({
                "action": "tag_merge",
                "source_tags": ["one"],
                "target_tag": "target"
            })),
        )
        .await
        .unwrap_err();
        assert!(error.contains("at least two"));
    }

    #[test]
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn test_union_find() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(2, 3);
        uf.union(1, 3);
        assert_eq!(uf.find(0), uf.find(3));
        assert_ne!(uf.find(0), uf.find(4));
    }

    #[tokio::test]
    async fn test_empty_storage() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        let storage = Arc::new(storage);
        let result = execute(&storage, None).await;
        assert!(result.is_ok());
    }
}
