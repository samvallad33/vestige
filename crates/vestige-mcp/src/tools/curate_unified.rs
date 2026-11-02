//! `curate` MCP tool — Graph & State Mutation.
//!
//! Absorbs: `memory(edit | promote | demote | state)` + `suppress` + `graph` + `ghostlink` + CLI `vestige connect`.
//! Actions: "promote" | "demote" | "edit" | "link" | "bridge" | "connect_identities" | "suppress".

use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::Mutex;
use vestige_core::Storage;

use crate::cognitive::CognitiveEngine;
use crate::tools;

pub fn schema() -> Value {
    let mut schema = json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["promote", "demote", "edit", "link", "bridge", "connect_identities", "suppress", "memory", "graph", "ghostlink", "dream", "maintain"],
                "description": "'promote'/'demote': FSRS-6 review updates; 'edit': replace node content with successor; 'link': add typed edge; 'bridge': ghostlink cross-domain connection; 'connect_identities': unify identifiers; 'suppress': hide node from reads."
            },
            "params": {"type":"object", "description":"Complete native payload for memory, graph, ghostlink, dream, or maintain, including its own action/mode."},
            "scope": {"type":"string", "default":"user", "description":"Both link endpoints must be live in this namespace."},
            "id": { "type": "string", "description": "Target memory ID." },
            "source_id": { "type": "string", "description": "Source memory ID for link/bridge." },
            "target_id": { "type": "string", "description": "Target memory ID for link/bridge." },
            "link_type": { "type": "string", "enum":["derived_from", "evidence_of", "closes", "supersedes"], "description": "Native declared edge kind. 'closes' records target -closed_by-> source. An edge alone does not retire a memory." },
            "content": { "type": "string", "description": "New content for 'edit'." },
            "reason": { "type": "string", "description": "Reason for promote/demote/suppress." },
            "role": { "type": "string", "description": "Claimed role for promote/demote provenance." },
            "identity_a": { "type": "string", "description": "For 'connect_identities': first identity tag/handle." },
            "identity_b": { "type": "string", "description": "For 'connect_identities': second identity tag/handle." },
            "confirm": { "type": "boolean", "default": true, "description": "Required confirmation for destructive mutations like suppress." }
        },
        "required": ["action"]
    });
    schema["allOf"] = json!([
        {"if":{"properties":{"action":{"const":"memory"}},"required":["action"]},
         "then":{"properties":{"params":tools::session_unified::embedded_schema(tools::memory_unified::schema(), "/allOf/0/then/properties/params")},"required":["params"]}},
        {"if":{"properties":{"action":{"const":"graph"}},"required":["action"]},
         "then":{"properties":{"params":tools::session_unified::embedded_schema(tools::graph_unified::schema(), "/allOf/1/then/properties/params")},"required":["params"]}},
        {"if":{"properties":{"action":{"const":"ghostlink"}},"required":["action"]},
         "then":{"properties":{"params":tools::session_unified::embedded_schema(tools::ghostlink::schema(), "/allOf/2/then/properties/params")},"required":["params"]}},
        {"if":{"properties":{"action":{"const":"dream"}},"required":["action"]},
         "then":{"properties":{"params":tools::session_unified::embedded_schema(tools::dream::schema(), "/allOf/3/then/properties/params")},"required":["params"]}},
        {"if":{"properties":{"action":{"const":"maintain"}},"required":["action"]},
         "then":{"properties":{"params":tools::session_unified::embedded_schema(tools::maintain::schema(), "/allOf/4/then/properties/params")},"required":["params"]}}
    ]);
    schema
}

#[derive(Debug, Deserialize)]
struct CurateArgs {
    action: String,
    id: Option<String>,
    source_id: Option<String>,
    target_id: Option<String>,
    link_type: Option<String>,
    scope: Option<String>,
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
        Some(v) => {
            serde_json::from_value(v).map_err(|e| format!("Invalid curate arguments: {e}"))?
        }
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
            )
            .await
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
            )
            .await
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
            )
            .await
        }
        "suppress" => {
            let id = args.id.ok_or("suppress requires 'id'")?;
            tools::suppress::execute(
                storage,
                Some(json!({
                    "id": id,
                    "reason": args.reason,
                    "confirm": args.confirm,
                })),
            )
            .await
        }
        "link" => {
            let src = args
                .source_id
                .or(args.id)
                .ok_or("link requires 'source_id' or 'id'")?;
            let tgt = args.target_id.ok_or("link requires 'target_id'")?;
            let ltype = args.link_type.ok_or("link requires 'link_type'")?;
            let kind = crate::strata_memory::DeclaredLink::parse(&ltype)
                .ok_or("link_type must be derived_from, evidence_of, closes, or supersedes")?;
            if src == tgt {
                return Err("a memory cannot link to itself".into());
            }
            let scope = args.scope.as_deref().unwrap_or("user");
            // Validate BOTH endpoints before writing, including retirement and
            // scope boundaries, then use the existing native gated edge path.
            crate::strata_memory::check_links(
                storage.as_ref(),
                scope,
                &[(kind, src.clone()), (kind, tgt.clone())],
            )?;
            let links = crate::strata_memory::save_links(storage.as_ref(), &src, &[(kind, tgt)])?;
            Ok(json!({"action":"link", "success":true, "links":links}))
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
            )
            .await
        }
        "connect_identities" => {
            let id_a = args
                .identity_a
                .ok_or("connect_identities requires 'identity_a'")?;
            let id_b = args
                .identity_b
                .ok_or("connect_identities requires 'identity_b'")?;
            Err(format!(
                "connect_identities is unavailable: no storage identity-unification API for {id_a:?} and {id_b:?}"
            ))
        }
        other => Err(format!(
            "unknown curate action '{other}'; use promote, demote, edit, link, bridge, connect_identities, or suppress"
        )),
    }
}
