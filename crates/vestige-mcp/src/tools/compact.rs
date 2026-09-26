//! Compact wire schemas for `tools/list` (#212).
//!
//! `tools/list` serves a budgeted compact view of every tool schema:
//! discriminator enums, types, and required arrays survive; deep variant
//! trees, per-field prose, and defaults do not. The FULL schema for any
//! tool stays on the wire through `memory_status` with `view='tools'` and
//! `tool='<name>'` (see [`full_schema`]), so compaction loses no
//! discoverability — it moves it one call deeper, which is the progressive
//! disclosure the server already documents in its instructions string.
//!
//! Budget: the serialized `tools/list` payload must stay under 20 KiB. The
//! guard test in `server.rs` fails the build when a schema change pushes
//! the catalog over.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

/// Cap for tool-level and inputSchema-root descriptions.
const ROOT_DESCRIPTION_CAP: usize = 60;
/// Cap for descriptions on discriminator properties (`action`/`view`/`mode`).
const SELECTOR_DESCRIPTION_CAP: usize = 50;
/// Properties this deep in the tree keep their `items` shape only as a type.
const ITEMS_FLATTEN_DEPTH: usize = 2;

const VARIANT_POINTER: &str =
    "Exact per-variant parameters: memory_status with view='tools' and tool='<name>'.";

/// Filter fields grouped into objects at the root of a schema. Keeps flat
/// filter lists from dominating the wire budget while staying one level deep
/// instead of one property each. Handler parsing is unchanged: a group is a
/// normal object property, and nothing here moves a required field.
const FOLD_GROUPS: &[(&str, &[&str])] = &[
    (
        "source",
        &[
            "source_author",
            "source_id",
            "source_project",
            "source_status",
            "source_system",
            "source_type",
            "source_updated_after",
            "source_updated_before",
        ],
    ),
    (
        "filters",
        &[
            "include_types",
            "exclude_types",
            "tag_prefix",
            "min_retention",
            "min_similarity",
            "concrete",
            "validAt",
            "rank_native_fusion",
            "context_packet",
            "known_packet_id",
            "token_budget",
            "retrieval_mode",
        ],
    ),
];

fn truncate(s: &str, limit: usize) -> String {
    if s.len() <= limit {
        return s.to_string();
    }
    let cut = &s[..limit];
    if let Some(i) = cut.rfind(". ")
        && i > limit / 2
    {
        return cut[..i + 1].to_string();
    }
    match cut.rfind(' ') {
        Some(i) if i > 0 => cut[..i].to_string(),
        _ => cut.to_string(),
    }
}

/// The `action`/`view`/`mode` const a oneOf variant discriminates on, if any.
fn variant_discriminator(variant: &Value) -> Option<Value> {
    let props = variant.get("properties")?;
    for key in ["action", "view", "mode"] {
        if let Some(c) = props.get(key).and_then(|p| p.get("const")) {
            return Some(c.clone());
        }
    }
    None
}

/// A oneOf/anyOf tree becomes one object whose action enum carries the
/// variant discriminators. Variants without a discriminator collapse into a
/// pointer rather than an empty list, so the text never reads "variants: .".
fn compact_variants(map: &Map<String, Value>) -> Value {
    let variants = map
        .get("oneOf")
        .or_else(|| map.get("anyOf"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let discs: Vec<Value> = variants.iter().filter_map(variant_discriminator).collect();
    let action = if discs.is_empty() {
        json!({"type": "string"})
    } else {
        json!({"type": "string", "enum": discs})
    };
    let summary = if discs.is_empty() {
        format!("Several request shapes; {VARIANT_POINTER}")
    } else {
        let names: Vec<String> = discs
            .iter()
            .map(|d| d.as_str().unwrap_or_default().to_string())
            .collect();
        format!("One action per call; variants: {}. {VARIANT_POINTER}", names.join(", "))
    };
    json!({
        "type": "object",
        "properties": { "action": action },
        "description": summary,
    })
}

fn compact(node: &Value, depth: usize, keep_desc: bool) -> Value {
    match node {
        Value::Object(map) => {
            if map.contains_key("oneOf") || map.contains_key("anyOf") {
                return compact_variants(map);
            }
            let mut out = Map::new();
            for (key, value) in map {
                match key.as_str() {
                    "description" => {
                        if let Some(text) = value.as_str() {
                            let cap = if depth == 0 {
                                ROOT_DESCRIPTION_CAP
                            } else if keep_desc {
                                SELECTOR_DESCRIPTION_CAP
                            } else {
                                0
                            };
                            let truncated = truncate(text, cap);
                            if !truncated.is_empty() {
                                out.insert(key.clone(), Value::String(truncated));
                            }
                        }
                    }
                    // Defaults, examples, formats, and shared definition
                    // blocks are full-schema material. A `$ref` into a dropped
                    // `$defs` block becomes a plain object; the full schema
                    // keeps the real shape.
                    "default" | "examples" | "format" | "$defs" => {}
                    "$ref" => {
                        out.insert("type".into(), json!("object"));
                    }
                    "properties" if depth == 0 => {
                        out.insert(key.clone(), compact_root_properties(value));
                    }
                    "items" if depth >= ITEMS_FLATTEN_DEPTH => {
                        let item_type = value.get("type").cloned().unwrap_or(json!("object"));
                        out.insert(key.clone(), json!({ "type": item_type }));
                    }
                    _ => {
                        let child_keeps = keep_desc && matches!(key.as_str(), "action" | "view" | "mode");
                        out.insert(key.clone(), compact(value, depth + 1, child_keeps));
                    }
                }
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| compact(item, depth + 1, keep_desc))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Root `properties` get the #212 fold: the long tail of investigation
/// filters moves into grouped objects, everything else compacts in place
/// with descriptions kept only on discriminators.
fn compact_root_properties(properties: &Value) -> Value {
    let Value::Object(props) = properties else {
        return compact(properties, 1, false);
    };
    let mut out = Map::new();
    let mut groups: BTreeMap<String, Map<String, Value>> = BTreeMap::new();
    for (name, schema) in props {
        let mut placed = false;
        for (group, members) in FOLD_GROUPS {
            if members.contains(&name.as_str()) {
                let entry = json!({
                    "type": schema.get("type").cloned().unwrap_or(json!("string")),
                    "description": truncate(schema.get("description").and_then(Value::as_str).unwrap_or_default(), 60),
                });
                groups
                    .entry((*group).to_string())
                    .or_default()
                    .insert(name.clone(), entry);
                placed = true;
                break;
            }
        }
        if !placed {
            let is_selector = matches!(name.as_str(), "action" | "view" | "mode");
            out.insert(name.clone(), compact(schema, 1, is_selector));
        }
    }
    for (group, members) in groups {
        let names: Vec<&str> = {
            let mut v: Vec<&str> = members.keys().map(String::as_str).collect();
            v.sort_unstable();
            v
        };
        out.insert(
            group,
            json!({
                "type": "object",
                "description": format!("Grouped filters ({}).", names.join(", ")),
                "properties": Value::Object(members),
            }),
        );
    }
    Value::Object(out)
}

/// Compact one full tool schema for the `tools/list` wire. Idempotent on
/// already-compact input; the full schema passed in is never mutated.
pub fn of(full: &Value) -> Value {
    let mut compacted = compact(full, 0, true);
    // A `required` entry that named a field now living inside a fold group
    // would make the compact schema unsatisfiable. Prune to what the compact
    // root still declares.
    let root_props: Vec<Value> = compacted
        .get("properties")
        .and_then(Value::as_object)
        .map(|props| props.keys().map(|key| Value::String(key.clone())).collect())
        .unwrap_or_default();
    if let Some(required) = compacted.get_mut("required").and_then(Value::as_array_mut) {
        required.retain(|entry| root_props.iter().any(|prop| prop == entry));
    }
    compacted
}

/// Full schemas by advertised tool name. `tools/list` serves the compact
/// form; this registry is how `memory_status` `view='tools'` hands back the
/// complete schema for a selected tool, so no detail is lost — it lives one
/// call deeper. The parity guard test in `server.rs` fails the build if this
/// registry and the catalog ever disagree on a name.
pub fn full_schema(name: &str) -> Option<Value> {
    use super::*;
    Some(match name {
        "recall" => recall::schema(),
        "receipt" => receipt::schema(),
        "memory" => memory_unified::schema(),
        "codebase" => codebase_unified::schema(),
        "project" => project::schema(),
        "intention" => intention_graph::schema(),
        "smart_ingest" => smart_ingest::schema(),
        "source_sync" => source_sync::schema(),
        "memory_status" => memory_status::schema(),
        "maintain" => maintain::schema(),
        "dedup" => dedup::unified_schema(),
        "graph" => graph_unified::schema(),
        "session_start" => session_context::schema(),
        "suppress" => suppress::schema(),
        "backfill" => backfill::schema(),
        "purge" => memory_unified::purge_schema(),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_oneof_with_discriminators_becomes_an_enum_and_a_pointer() {
        let full = json!({
            "type": "object",
            "oneOf": [
                {"properties": {"action": {"const": "scan"}}, "required": ["action"]},
                {"properties": {"action": {"const": "apply"}}, "required": ["action"]}
            ]
        });
        let compact = of(&full);
        let enum_values = compact["properties"]["action"]["enum"]
            .as_array()
            .expect("enum survives");
        assert_eq!(enum_values.len(), 2);
        assert!(compact.to_string().contains(VARIANT_POINTER));
    }

    #[test]
    fn a_oneof_without_discriminators_never_reads_variants_empty() {
        let full = json!({
            "type": "object",
            "oneOf": [
                {"properties": {"at": {"type": "string"}}},
                {"properties": {"in_minutes": {"type": "integer"}}}
            ]
        });
        let compact = of(&full);
        let text = compact.to_string();
        assert!(!text.contains("variants: ."), "{text}");
        assert!(text.contains(VARIANT_POINTER));
    }

    #[test]
    fn folded_filter_fields_leave_required_and_land_in_their_group() {
        let full = json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "source_author": {"type": "string", "description": "Only this source author (not assignee)."},
                "token_budget": {"type": "integer", "description": "Max response tokens."}
            },
            "required": ["query", "source_author"]
        });
        let compact = of(&full);
        assert!(compact["properties"]["query"].is_object());
        assert!(compact["properties"]["source"]["properties"]["source_author"].is_object());
        assert!(compact["properties"]["filters"]["properties"]["token_budget"].is_object());
        let required = compact["required"].as_array().unwrap();
        assert_eq!(required, &vec![json!("query")], "folded names leave required");
    }

    #[test]
    fn a_dropped_defs_block_turns_refs_into_plain_objects() {
        let full = json!({
            "type": "object",
            "$defs": {"trigger": {"type": "object", "properties": {"at": {"type": "string"}}}},
            "properties": {"trigger": {"$ref": "#/$defs/trigger"}}
        });
        let compact = of(&full);
        assert!(compact.get("$defs").is_none());
        assert_eq!(compact["properties"]["trigger"]["type"], json!("object"));
        assert!(compact["properties"]["trigger"].get("$ref").is_none());
    }

    #[test]
    fn long_field_descriptions_drop_while_discriminators_keep_a_cap() {
        let full = json!({
            "type": "object",
            "description": "Tool-level description that is long enough to need truncation and then some more text beyond the cap.",
            "properties": {
                "action": {"type": "string", "enum": ["a"], "description": "Discriminator description that is also fairly long and will be capped rather than dropped entirely."},
                "obscure_field": {"type": "string", "description": "A long per-field description that should disappear entirely from the compact form."}
            }
        });
        let compact = of(&full);
        assert!(compact["description"].as_str().unwrap().len() <= 200);
        let action_desc = compact["properties"]["action"]["description"]
            .as_str()
            .unwrap();
        assert!(!action_desc.is_empty() && action_desc.len() <= 120);
        assert!(compact["properties"]["obscure_field"].get("description").is_none());
    }
}
