//! `curate` MCP tool — Graph & State Mutation.
//!
//! Absorbs: `memory(edit | promote | demote | state)` + `suppress` + `graph` + `ghostlink` + CLI `vestige connect`.
//! Actions: "promote" | "demote" | "edit" | "link" | "bridge" | "connect_identities" | "suppress".

use std::sync::Arc;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;
use vestige_core::Storage;

use crate::cognitive::CognitiveEngine;
use crate::tools;

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["promote", "demote", "edit", "link", "bridge", "connect_identities", "suppress"],
                "description": "'promote'/'demote': FSRS-6 review updates; 'edit': replace node content with successor; 'link': add typed edge; 'bridge': ghostlink cross-domain connection; 'connect_identities': unify identifiers; 'suppress': hide node from reads."
            },
            "id": { "type": "string", "description": "Target memory ID." },
            "source_id": { "type": "string", "description": "Source memory ID for link/bridge." },
            "target_id": { "type": "string", "description": "Target memory ID for link/bridge." },
            "link_type": { "type": "string", "default": "relates_to", "description": "Relationship type for 'link' (e.g. caused_by, fixed_by, derives_from)." },
            "content": { "type": "string", "description": "New content for 'edit'." },
            "reason": { "type": "string", "description": "Reason for promote/demote/suppress." },
            "role": { "type": "string", "description": "Claimed role for promote/demote provenance." },
            "identity_a": { "type": "string", "description": "For 'connect_identities': first identity tag/handle." },
            "identity_b": { "type": "string", "description": "For 'connect_identities': second identity tag/handle." },
            "confirm": { "type": "boolean", "default": true, "description": "Required confirmation for destructive mutations like suppress." }
        },
        "required": ["action"]
    })
}

#[derive(Debug, Deserialize)]
struct CurateArgs {
    action: String,
    id: Option<String>,
    source_id: Option<String>,
    target_id: Option<String>,
    link_type: Option<String>,
    content: Option<String>,
    reason: Option<String>,
    role: Option<String>,
    identity_a: Option<String>,
    identity_b: Option<String>,
    #[serde(default = "default_confirm")]
    confirm: bool,
}

fn default_confirm() -> bool {
    true
}

pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: CurateArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid curate arguments: {e}"))?,
        None => return Err("Missing curate arguments".to_string()),
    };

    match args.action.as_str() {
        "promote" => {
            let id = args.id.ok_or("promote requires 'id'")?;
            tools::memory_unified::execute(
                storage,
                cognitive,
                Some(json!({
                    "action": "promote",
                    "id": id,
                    "reason": args.reason,
                    "role": args.role,
                })),
            ).await
        }
        "demote" => {
            let id = args.id.ok_or("demote requires 'id'")?;
            tools::memory_unified::execute(
                storage,
                cognitive,
                Some(json!({
                    "action": "demote",
                    "id": id,
                    "reason": args.reason,
                    "role": args.role,
                })),
            ).await
        }
        "edit" => {
            let id = args.id.ok_or("edit requires 'id'")?;
            let content = args.content.ok_or("edit requires 'content'")?;
            tools::memory_unified::execute(
                storage,
                cognitive,
                Some(json!({
                    "action": "edit",
                    "id": id,
                    "content": content,
                })),
            ).await
        }
        "suppress" => {
            let id = args.id.ok_or("suppress requires 'id'")?;
            tools::suppress::execute(
                storage,
                Some(json!({
                    "memory_id": id,
                    "reason": args.reason,
                    "confirm": args.confirm,
                })),
            ).await
        }
        "link" => {
            let src = args.source_id.or(args.id).ok_or("link requires 'source_id' or 'id'")?;
            let tgt = args.target_id.ok_or("link requires 'target_id'")?;
            let ltype = args.link_type.unwrap_or_else(|| "relates_to".to_string());
            tools::graph_unified::execute(
                storage,
                cognitive,
                Some(json!({
                    "action": "link",
                    "source_id": src,
                    "target_id": tgt,
                    "link_type": ltype,
                })),
            ).await
        }
        "bridge" => {
            let src = args.source_id.or(args.id);
            let tgt = args.target_id;
            tools::ghostlink::execute(
                storage,
                cognitive,
                Some(json!({
                    "mode": "propose",
                    "lens": "bridge",
                    "from": src,
                    "to": tgt,
                })),
            ).await
        }
        "connect_identities" => {
            let id_a = args.identity_a.ok_or("connect_identities requires 'identity_a'")?;
            let id_b = args.identity_b.ok_or("connect_identities requires 'identity_b'")?;
            // Add a bidirectional identity connection edge in Strata
            let storage_guard = storage.clone();
            tokio::task::spawn_blocking(move || {
                storage_guard.record_identity_connection(&id_a, &id_b)
            })
            .await
            .map_err(|e| format!("Join error: {e}"))?
            .map_err(|e| e.to_string())?;

            Ok(json!({
                "action": "connect_identities",
                "identity_a": id_a,
                "identity_b": id_b,
                "success": true,
                "note": "Identities unified across memory graph."
            }))
        }
        other => Err(format!("unknown curate action '{other}'; use promote, demote, edit, link, bridge, connect_identities, or suppress")),
    }
}
