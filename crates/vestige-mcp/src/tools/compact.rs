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
//! Budget: the serialized `tools/list` payload must stay under 22 KiB. The
//! guard test in `server.rs` fails the build when a schema change pushes
//! the catalog over.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

/// Cap for tool-level and inputSchema-root descriptions.
const ROOT_DESCRIPTION_CAP: usize = 60;
/// Cap for descriptions on discriminator properties (`action`/`view`/`mode`).
const SELECTOR_DESCRIPTION_CAP: usize = 50;
/// Properties this deep in the tree keep their `items` shape only as a type.
const ITEMS_FLATTEN_DEPTH: usize = 2;
/// An object this deep keeps only its type. Every field a call names at the
/// root or one level down (a variant's field, a grouped filter) is above it.
const OBJECT_FLATTEN_DEPTH: usize = 3;

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

/// Keywords a union node keeps from its own schema.
const UNION_OWN_KEYWORDS: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "description",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
];

/// JSON type name of a discriminator value.
fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
        Value::Number(_) => "number",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
        Value::Null => "null",
        Value::String(_) => "string",
    }
}

/// A oneOf/anyOf node on the wire. Scalar alternatives (a value that is a
/// string, integer or boolean) become a type list. Object alternatives keep
/// the node's own properties and gain every variant's properties, so no
/// field a call may send is missing from the wire: a client that drops
/// undeclared fields would otherwise strip real arguments (smart_ingest's
/// `content`, causal_walk's start point `kind`). A property the variants pin
/// with `const` (the discriminator: `action`, `kind`, `view`, ...) becomes an
/// enum of those values. Which field goes with which variant stays
/// full-schema detail, one call deeper.
fn compact_union(map: &Map<String, Value>, depth: usize, keep_desc: bool) -> Value {
    let variants: Vec<Value> = map
        .get("oneOf")
        .or_else(|| map.get("anyOf"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let object_like = map.contains_key("properties")
        || variants
            .iter()
            .any(|v| v.get("properties").is_some() || v.get("type") == Some(&json!("object")));
    if !object_like {
        let mut types: Vec<Value> = Vec::new();
        for variant in &variants {
            let listed = match variant.get("type") {
                Some(Value::Array(list)) => list.clone(),
                Some(single) => vec![single.clone()],
                None => Vec::new(),
            };
            for kind in listed {
                if !types.contains(&kind) {
                    types.push(kind);
                }
            }
        }
        return match types.len() {
            0 => json!({}),
            1 => json!({ "type": types[0].clone() }),
            _ => json!({ "type": types }),
        };
    }

    // The node's own schema keywords only. Annotations some unions carry
    // (`$schema`, `algorithm_version`, `limits`, ...) make strict JSON Schema
    // compilers reject the whole tool.
    let mut own = map.clone();
    own.retain(|key, _| UNION_OWN_KEYWORDS.contains(&key.as_str()));
    let mut out = match compact(&Value::Object(own), depth, keep_desc) {
        Value::Object(out) => out,
        _ => Map::new(),
    };
    out.insert("type".into(), json!("object"));

    let mut discriminators: Vec<(String, Vec<Value>)> = Vec::new();
    let mut variant_fields: Vec<(String, Value)> = Vec::new();
    let mut required_sets: Vec<Vec<Value>> = Vec::new();
    for variant in &variants {
        if let Some(props) = variant.get("properties").and_then(Value::as_object) {
            for (name, schema) in props {
                match schema.get("const") {
                    Some(value) => match discriminators.iter_mut().find(|(key, _)| key == name) {
                        Some((_, values)) if !values.contains(value) => values.push(value.clone()),
                        Some(_) => {}
                        None => discriminators.push((name.clone(), vec![value.clone()])),
                    },
                    None => variant_fields.push((name.clone(), schema.clone())),
                }
            }
        }
        required_sets.push(
            variant
                .get("required")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
        );
    }

    let props = out
        .entry("properties")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("properties is an object");
    for (name, values) in &discriminators {
        let kind = values.first().map(json_type).unwrap_or("string");
        props.insert(name.clone(), json!({ "type": kind, "enum": values }));
    }
    for (name, schema) in variant_fields {
        if !props.contains_key(&name) {
            props.insert(name, compact(&schema, depth + 1, false));
        }
    }

    // A field every variant requires is required whichever variant is sent.
    if let Some((first, rest)) = required_sets.split_first() {
        let mut always: Vec<Value> = first
            .iter()
            .filter(|name| rest.iter().all(|set| set.contains(name)))
            .filter(|name| name.as_str().is_some_and(|n| props.contains_key(n)))
            .cloned()
            .collect();
        if !always.is_empty() {
            let required = out
                .entry("required")
                .or_insert_with(|| json!([]))
                .as_array_mut()
                .expect("required is an array");
            always.retain(|name| !required.contains(name));
            required.extend(always);
        }
    }

    // The enum already lists the variants; the text only says where their
    // exact fields are.
    // Which field goes with which variant is one call deeper; memory_status
    // says so in its own description, so no pointer is repeated here.
    out.remove("description");
    Value::Object(out)
}

fn compact(node: &Value, depth: usize, keep_desc: bool) -> Value {
    match node {
        Value::Object(map) => {
            if depth >= OBJECT_FLATTEN_DEPTH
                && (map.contains_key("properties")
                    || map.contains_key("oneOf")
                    || map.contains_key("anyOf"))
            {
                return json!({ "type": "object" });
            }
            if map.contains_key("oneOf") || map.contains_key("anyOf") {
                return compact_union(map, depth, keep_desc);
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
                        let child_keeps =
                            keep_desc && matches!(key.as_str(), "action" | "view" | "mode");
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
                // Type only: a 60-character cut of the field's prose read as
                // noise, and the full description is one call deeper.
                let entry = json!({
                    "type": schema.get("type").cloned().unwrap_or(json!("string")),
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
    // A group's name says what it holds (`filters`, `source`); every member
    // is optional, so the group carries no text of its own on the wire.
    for (group, members) in groups {
        out.insert(
            group,
            json!({
                "type": "object",
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

/// Undo the #212 fold on an incoming call. `tools/list` advertises the long
/// tail of filter fields inside `filters` and `source` objects, but the
/// handlers read them at the top level, so a call shaped like the advertised
/// schema had those fields silently ignored. Fields of the tool's own root
/// schema move back to the top level; an explicit top-level value wins. A
/// tool with a real root property of the group's name (smart_ingest's
/// `source`) is left alone.
pub fn unfold_arguments(tool: &str, args: &mut Value) {
    let Some(obj) = args.as_object_mut() else {
        return;
    };
    if !FOLD_GROUPS
        .iter()
        .any(|(group, _)| obj.get(*group).is_some_and(Value::is_object))
    {
        return;
    }
    let Some(full) = full_schema(tool) else {
        return;
    };
    let Some(root) = full.get("properties").and_then(Value::as_object) else {
        return;
    };
    for (group, members) in FOLD_GROUPS {
        if root.contains_key(*group) {
            continue;
        }
        let Some(Value::Object(inner)) = obj.remove(*group) else {
            continue;
        };
        let mut left = Map::new();
        for (name, value) in inner {
            if members.contains(&name.as_str()) && root.contains_key(&name) {
                obj.entry(name).or_insert(value);
            } else {
                left.insert(name, value);
            }
        }
        if !left.is_empty() {
            obj.insert((*group).to_string(), Value::Object(left));
        }
    }
}

/// Full schemas by advertised tool name. `tools/list` serves the compact
/// form; this registry is how `memory_status` `view='tools'` hands back the
/// complete schema for a selected tool, so no detail is lost — it lives one
/// call deeper. The parity guard test in `server.rs` fails the build if this
/// registry and the catalog ever disagree on a name.
pub fn full_schema(name: &str) -> Option<Value> {
    use super::*;
    Some(match name {
        "session" => session_unified::schema(),
        "remember" => remember_unified::schema(),
        "curate" => curate_unified::schema(),
        "recall" => recall::schema(),
        "receipt" => receipt::schema(),
        "memory" => memory_unified::schema(),
        "codebase" => codebase_unified::schema(),
        "project" => project::schema(),
        "intention" => intention_graph::schema(),
        "smart_ingest" => smart_ingest::schema(),
        #[cfg(feature = "connectors")]
        "source_sync" => source_sync::schema(),
        "memory_status" => memory_status::schema(),
        "maintain" => maintain::schema(),
        "dedup" => dedup::unified_schema(),
        "ghostlink" => ghostlink::schema(),
        "session_start" => session_context::schema(),
        "suppress" => suppress::schema(),
        "backfill" => backfill::schema(),
        "causal_walk" => causal_walk::schema(),
        "selftest" => selftest::schema(),
        "forgotten_lesson" => forgotten_lesson::schema(),
        "precedent" => precedent::schema(),
        "gather" => gather::schema(),
        "purge" => memory_unified::purge_schema(),
        _ => return None,
    })
}

/// Every tool name `full_schema` knows, for the guards below.
#[cfg(test)]
const ALL_TOOLS: &[&str] = &[
    "session",
    "remember",
    "curate",
    "recall",
    "receipt",
    "memory",
    "codebase",
    "project",
    "intention",
    "smart_ingest",
    "memory_status",
    "maintain",
    "dedup",
    "ghostlink",
    "session_start",
    "suppress",
    "backfill",
    "causal_walk",
    "selftest",
    "forgotten_lesson",
    "precedent",
    "gather",
];

#[cfg(test)]
mod wire_tests {
    use super::*;

    /// Every property name at `node` (root and variants), recursively
    /// through object properties and array items.
    fn field_paths(node: &Value, path: &str, out: &mut Vec<String>) {
        let Some(map) = node.as_object() else {
            return;
        };
        let mut shapes = vec![node.clone()];
        for comb in ["oneOf", "anyOf", "allOf"] {
            if let Some(variants) = map.get(comb).and_then(Value::as_array) {
                shapes.extend(variants.iter().cloned());
            }
        }
        for shape in shapes {
            if let Some(props) = shape.get("properties").and_then(Value::as_object) {
                for (name, schema) in props {
                    let child = format!("{path}.{name}");
                    out.push(child.clone());
                    field_paths(schema, &child, out);
                }
            }
            if let Some(items) = shape.get("items") {
                field_paths(items, &format!("{path}[]"), out);
            }
        }
    }

    /// The fold groups move root fields one level down; everything else
    /// keeps its path.
    fn folded(path: &str) -> String {
        for (group, members) in FOLD_GROUPS {
            for member in *members {
                if path == format!(".{member}") {
                    return format!(".{group}.{member}");
                }
            }
        }
        path.to_string()
    }

    /// No field a call may send is missing from `tools/list`, down to the
    /// depth where compaction deliberately stops at a type.
    #[test]
    fn every_root_and_variant_field_reaches_the_wire() {
        for tool in ALL_TOOLS {
            let full = full_schema(tool).expect("known tool");
            let compact = of(&full);
            let mut wanted = Vec::new();
            field_paths(&full, "", &mut wanted);
            let mut present = Vec::new();
            field_paths(&compact, "", &mut present);
            for path in wanted {
                // Items two levels down flatten to a type on purpose.
                if path.matches('.').count() > 2 || path.matches("[]").count() > 1 {
                    continue;
                }
                let want = folded(&path);
                assert!(
                    present.contains(&want),
                    "{tool}: {want} is in the full schema but not on the wire"
                );
            }
        }
    }

    #[test]
    fn smart_ingest_advertises_its_real_fields_and_no_action() {
        let compact = of(&full_schema("smart_ingest").unwrap());
        let props = compact["properties"].as_object().unwrap();
        for field in ["content", "items", "tags", "forceCreate"] {
            assert!(props.contains_key(field), "missing {field}: {compact}");
        }
        assert!(!props.contains_key("action"), "{compact}");
    }

    #[test]
    fn a_const_discriminator_other_than_action_becomes_an_enum() {
        let full = json!({
            "type": "object",
            "properties": {"start_points": {"type": "array", "items": {"oneOf": [
                {"type": "object", "properties": {"kind": {"const": "test"}, "name": {"type": "string"}}, "required": ["kind", "name"]},
                {"type": "object", "properties": {"kind": {"const": "ci_run"}, "run_id": {"type": "string"}}, "required": ["kind", "run_id"]}
            ]}}}
        });
        let compact = of(&full);
        let items = &compact["properties"]["start_points"]["items"];
        assert_eq!(
            items["properties"]["kind"]["enum"],
            json!(["test", "ci_run"])
        );
        assert!(items["properties"]["name"].is_object());
        assert!(items["properties"]["run_id"].is_object());
        assert!(items["properties"].get("action").is_none());
        assert_eq!(items["required"], json!(["kind"]));
    }

    #[test]
    fn a_union_keeps_only_schema_keywords_of_its_own() {
        let full = json!({"type": "object", "properties": {"command": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Intention graph command",
            "algorithm_version": 3,
            "limits": {"max": 1},
            "type": "object",
            "oneOf": [
                {"type": "object", "properties": {"action": {"const": "plan"}, "goal": {"type": "string"}}},
                {"type": "object", "properties": {"action": {"const": "cancel"}, "id": {"type": "string"}}}
            ]
        }}});
        let command = &of(&full)["properties"]["command"];
        for key in ["$schema", "title", "algorithm_version", "limits"] {
            assert!(command.get(key).is_none(), "{key} leaked: {command}");
        }
        assert_eq!(
            command["properties"]["action"]["enum"],
            json!(["plan", "cancel"])
        );
        assert!(command["properties"]["goal"].is_object());
    }

    #[test]
    fn scalar_alternatives_become_a_type_list() {
        let full = json!({"type": "object", "properties": {"value": {"oneOf": [
            {"type": "boolean"}, {"type": "integer"}, {"type": "string"}
        ]}}});
        assert_eq!(
            of(&full)["properties"]["value"]["type"],
            json!(["boolean", "integer", "string"])
        );
    }

    #[test]
    fn unfold_moves_grouped_filters_back_and_keeps_explicit_values() {
        let mut args = json!({"filters": {"token_budget": 60}, "queries": ["x"]});
        unfold_arguments("session_start", &mut args);
        assert_eq!(args, json!({"token_budget": 60, "queries": ["x"]}));

        let mut args = json!({"token_budget": 10, "filters": {"token_budget": 60}});
        unfold_arguments("session_start", &mut args);
        assert_eq!(args, json!({"token_budget": 10}));

        let mut args = json!({"handle": "t", "source": {"source_id": "a"}, "filters": {"tag_prefix": "p", "bogus": 1}});
        unfold_arguments("recall", &mut args);
        assert_eq!(args["source_id"], "a");
        assert_eq!(args["tag_prefix"], "p");
        assert_eq!(args["filters"], json!({"bogus": 1}));

        // smart_ingest has a real `source` field; it is not a fold group there.
        let mut args = json!({"content": "c", "source": {"source_id": "a"}});
        unfold_arguments("smart_ingest", &mut args);
        assert_eq!(args["source"], json!({"source_id": "a"}));
    }
}
