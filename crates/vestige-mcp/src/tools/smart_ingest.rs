//! Smart Ingest Tool
//!
//! Intelligent memory ingestion with Prediction Error Gating.
//! Automatically decides whether to create, update, or supersede memories
//! based on semantic similarity to existing content.
//!
//! This solves the "bad vs good similar memory" problem by:
//! - Detecting when new content is similar to existing memories
//! - Updating existing memories when appropriate (low prediction error)
//! - Creating new memories when content is substantially different (high PE)
//! - Superseding demoted/outdated memories with better alternatives
//!
//! v1.5.0: Enhanced with cognitive pipeline:
//!   Pre-ingest: importance scoring (4-channel) + intent detection → auto-tag
//!   Post-ingest: synaptic tagging + novelty model update + hippocampal indexing
//!
//! INGEST-V5 (feat/ingest-proof): the proof-carrying write path. Every write
//! is gated by caller idempotency (`intent_id` replay, zero writes) and by
//! canonical-byte identity (an exact duplicate records an echo node plus an
//! `evidence_of` edge — decision `reinforce` — and NEVER merges or mutates
//! the original), and every create response carries proof fields
//! (`receiptId`, `canonicalHash`, `entities`, `importance`) derivable from
//! log facts or the submitted bytes alone. No clock reads and no randomness
//! decide anything on this path.

use crate::cognitive::CognitiveEngine;
use crate::strata_memory;
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;
use vestige_core::{
    DEFAULT_MEMORY_SCOPE, ImportanceContext, IngestInput, SecretPolicy, Storage, StorageError,
    SynapticSignalSnapshot, scan_secrets,
};

/// The `intent_id` property schema, shared by the single and batch-item
/// forms (INGEST-V5 B1). Built in its own function so the big `schema()`
/// literal stays under `json_internal!`'s expansion depth.
fn intent_id_property() -> Value {
    serde_json::json!({
        "type": "string",
        "maxLength": 128,
        "pattern": "^[A-Za-z0-9._:-]+$",
        "description": "Idempotency key for this write. Replaying an intent_id already recorded in this scope returns the first outcome (decision 'replay', replayOf set) and performs no write of any kind, whatever the new bytes say; the response's requestCanonicalHash makes a divergent replay visible."
    })
}

/// Input schema for smart_ingest tool
///
/// Supports two modes:
/// - **Single mode**: provide `content` (required) + optional fields
/// - **Batch mode**: provide `items` array (max 20), each with full cognitive pipeline
pub fn schema() -> Value {
    let mut s = serde_json::json!({
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "What to remember (single mode). Strata admits it as a new memory; the legacy engine compares it with existing memories."
            },
            "node_type": {
                "type": "string",
                "description": "fact, concept, event, person, place, note, pattern, decision, or state. 'state' snapshots expire after VESTIGE_STATE_TTL_DAYS (default 30) unless validUntil is set, then rank last with currentlyValid=false.",
                "default": "fact"
            },
            "tags": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Tags, stored exactly as given. Legacy engine only: near-miss same-scope tag suggestions come back, never auto-applied."
            },
            "source": {
                "type": "string",
                "description": "Source or reference for this knowledge"
            },
            "scope": {
                "type": "string",
                "description": "Project namespace (default 'user')."
            },
            "validFrom": {
                "type": "string",
                "description": "When the fact becomes true (RFC3339 or YYYY-MM-DD). Omitted: one unambiguous 'as of DATE' phrase is inferred."
            },
            "validUntil": {
                "type": "string",
                "description": "When the fact stops being true (RFC3339 or YYYY-MM-DD, after validFrom)."
            },
            "forceCreate": {
                "type": "boolean",
                "description": "Legacy engine only: create even when a merge would apply. Strata always creates.",
                "default": false
            },
            "allowSecrets": {
                "type": "boolean",
                "description": "Store a detected credential for this item. Dangerous; redact or store a reference instead.",
                "default": false
            },
            "previewTagSuggestions": {
                "type": "boolean",
                "description": "Read-only preflight: inferred validity, plus tag suggestions on the legacy engine only; nothing stored.",
                "default": false
            },
            "acceptedTagSuggestions": {
                "type": "object",
                "additionalProperties": { "type": "string" },
                "description": "Legacy engine only: accepted input-tag to existing-tag mappings from a preflight; revalidated before ingest."
            },
            "batchMergePolicy": {
                "type": "string",
                "enum": ["force_create", "smart"],
                "description": "Batch only. 'smart' (default): an item whose text already exists in the scope reinforces it with an echo. 'force_create' stores every item as a new memory.",
                "default": "smart"
            },
            "role": {
                "type": "string",
                "description": "CLAIMED role for provenance (e.g. 'qa', 'architect'). Claims never override the process identity and never self-grant authority: the operator-controlled policy resolves the effective role, and unregistered claims stay neutral at 1.0. The response 'actor' block reports the resolution."
            },
            "intent_id": intent_id_property(),
            "links": {
                "type": "array",
                "maxItems": 16,
                "description": "Typed links from this memory to existing memories in the same scope, each written through the gate with its own receipt. derived_from: this memory derives from it. evidence_of: this memory is evidence about it. closes: this memory closes it (an issue, a failure). supersedes: this memory fully replaces it (for corrections prefer the review-gated corrects path). Strata only.",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["derived_from", "evidence_of", "closes", "supersedes"]},
                        "to": {"type": "string", "description": "Exact id of an existing memory"}
                    },
                    "required": ["kind", "to"]
                }
            },
            "items": {
                "type": "array",
                "description": "Batch: up to 20 items with the same fields as single mode, each force-created unless batchMergePolicy='smart'. For session end or before compaction.",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string"
                        },
                        "links": {
                            "type": "array",
                            "maxItems": 16,
                            "description": "Typed links from this item, as in single mode",
                            "items": {"type": "object"}
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" }
                        },
                        "node_type": {
                            "type": "string",
                            "default": "fact"
                        },
                        "source": {
                            "type": "string"
                        },
                        "scope": {
                            "type": "string"
                        },
                        "validFrom": {
                            "type": "string"
                        },
                        "validUntil": {
                            "type": "string"
                        },
                        "forceCreate": {
                            "type": "boolean",
                            "default": false
                        },
                        "allowSecrets": {
                            "type": "boolean",
                            "default": false
                        },
                        "previewTagSuggestions": {
                            "type": "boolean",
                            "default": false
                        },
                        "acceptedTagSuggestions": {
                            "type": "object",
                            "additionalProperties": { "type": "string" }
                        },
                        "intent_id": intent_id_property()
                    },
                    "required": ["content"]
                }
            }
        },
        "oneOf": [
            {"required":["content"], "not":{"required":["items"]}},
            {"required":["items"], "not":{"required":["content"]}}
        ]
    });
    s["properties"]["outputProfile"] = serde_json::json!({
        "type": "string",
        "enum": ["compact", "standard", "full"],
        "description": "'compact' (default): proof fields only ({decision, nodeId, receiptId, canonicalHash, scope}). 'standard': proof fields plus entities and importance."
    });
    for field in ["entities", "attributes"] {
        let description = if field == "entities" {
            "Explicit case-sensitive identities, e.g. person:Maya or place:Bangkok Garden. Stored as exact tags; never inferred from prose."
        } else {
            "Explicit facts, e.g. allergy:peanut or preference:ambience=quiet. Stored as exact tags."
        };
        let property = serde_json::json!({"type":"array","maxItems":64,"items":{"type":"string","maxLength":512},"description":description});
        s["properties"][field] = property.clone();
        s["properties"]["items"]["items"]["properties"][field] = property;
    }
    s
}

/// Fold caller-declared identities and attributes into the existing exact tag
/// representation. Validate every batch item before its first write.
fn personal_metadata(args: Option<Value>) -> Result<Option<Value>, String> {
    fn normalize(item: &mut Value) -> Result<(), String> {
        let object = item
            .as_object_mut()
            .ok_or("ingest item must be an object")?;
        let mut metadata = Vec::new();
        for field in ["entities", "attributes"] {
            if let Some(value) = object.remove(field) {
                let values = value
                    .as_array()
                    .ok_or_else(|| format!("{field} must be an array"))?;
                if values.len() > 64 {
                    return Err(format!("at most 64 {field} per memory"));
                }
                for value in values {
                    let text = value
                        .as_str()
                        .ok_or_else(|| format!("{field} must contain strings"))?;
                    let (kind, name) = text
                        .split_once(':')
                        .ok_or_else(|| format!("{field} need typed values"))?;
                    if name.trim().is_empty()
                        || text.len() > 512
                        || text.chars().any(char::is_control)
                    {
                        return Err(format!("invalid {field} value"));
                    }
                    if field == "entities"
                        && !matches!(
                            kind,
                            "person"
                                | "place"
                                | "food"
                                | "file"
                                | "service"
                                | "symbol"
                                | "path"
                                | "commit"
                                | "issue"
                                | "url"
                        )
                    {
                        return Err("unsupported explicit entity kind".into());
                    }
                    metadata.push(Value::String(text.to_string()));
                }
            }
        }
        if !metadata.is_empty() {
            let tags = object
                .entry("tags")
                .or_insert_with(|| serde_json::json!([]));
            let tags = tags.as_array_mut().ok_or("tags must be an array")?;
            for value in metadata {
                if !tags.contains(&value) {
                    tags.push(value);
                }
            }
        }
        Ok(())
    }
    let Some(mut args) = args else {
        return Ok(None);
    };
    normalize(&mut args)?;
    if let Some(items) = args.get_mut("items").and_then(Value::as_array_mut) {
        for item in items {
            normalize(item)?;
        }
    }
    Ok(Some(args))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SmartIngestArgs {
    content: Option<String>,
    #[serde(alias = "node_type")]
    node_type: Option<String>,
    tags: Option<Vec<String>>,
    source: Option<String>,
    scope: Option<String>,
    valid_from: Option<String>,
    valid_until: Option<String>,
    force_create: Option<bool>,
    allow_secrets: Option<bool>,
    preview_tag_suggestions: Option<bool>,
    accepted_tag_suggestions: Option<std::collections::BTreeMap<String, String>>,
    batch_merge_policy: Option<String>,
    items: Option<Vec<BatchItem>>,
    /// #252 Phase A: the CALLER's claimed role. This never overrides the
    /// process identity and never self-grants authority — the operator
    /// policy decides the effective role and weight, and unknown claims
    /// stay neutral at 1.0.
    #[serde(alias = "claimed_role")]
    role: Option<String>,
}

/// A single item in batch mode
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BatchItem {
    content: String,
    tags: Option<Vec<String>>,
    #[serde(alias = "node_type")]
    node_type: Option<String>,
    source: Option<String>,
    scope: Option<String>,
    valid_from: Option<String>,
    valid_until: Option<String>,
    force_create: Option<bool>,
    allow_secrets: Option<bool>,
    preview_tag_suggestions: Option<bool>,
    accepted_tag_suggestions: Option<std::collections::BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Copy)]
struct ValidityRange {
    from: Option<DateTime<Utc>>,
    until: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
struct ValidityResolution {
    range: ValidityRange,
    source: &'static str,
    inferred_phrase: Option<String>,
    ambiguous_phrases: Vec<String>,
}

fn parse_validity_timestamp(
    raw: Option<&str>,
    field: &str,
) -> Result<Option<DateTime<Utc>>, String> {
    let Some(raw) = raw else { return Ok(None) };
    if raw.trim() != raw || raw.is_empty() {
        return Err(format!(
            "Invalid {field}: expected RFC3339 or YYYY-MM-DD without surrounding whitespace"
        ));
    }
    if let Ok(timestamp) = DateTime::parse_from_rfc3339(raw) {
        return Ok(Some(timestamp.with_timezone(&Utc)));
    }
    if let Ok(date) = NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
        return Ok(Some(Utc.from_utc_datetime(
            &date.and_hms_opt(0, 0, 0).expect("midnight is valid"),
        )));
    }
    Err(format!("Invalid {field}: expected RFC3339 or YYYY-MM-DD"))
}

fn parse_validity_range(
    valid_from: Option<&str>,
    valid_until: Option<&str>,
) -> Result<ValidityRange, String> {
    let valid_from = parse_validity_timestamp(valid_from, "validFrom")?;
    let valid_until = parse_validity_timestamp(valid_until, "validUntil")?;
    if let (Some(from), Some(until)) = (valid_from, valid_until)
        && until <= from
    {
        return Err("Invalid validity range: validUntil must be after validFrom".to_string());
    }
    Ok(ValidityRange {
        from: valid_from,
        until: valid_until,
    })
}

fn infer_as_of_dates(content: &str) -> Vec<(DateTime<Utc>, String)> {
    let lowered = content.to_ascii_lowercase();
    let bytes = content.as_bytes();
    let mut inferred = std::collections::BTreeMap::new();
    for (phrase_start, _) in lowered.match_indices("as of ") {
        if phrase_start > 0
            && content[..phrase_start]
                .chars()
                .next_back()
                .is_some_and(|character| {
                    character.is_alphanumeric() || character == '_' || character == '-'
                })
        {
            continue;
        }
        let date_start = phrase_start + "as of ".len();
        let date_end = date_start + 10;
        if date_end > bytes.len() {
            continue;
        }
        let candidate = &bytes[date_start..date_end];
        if !candidate.iter().enumerate().all(|(index, byte)| {
            matches!(index, 4 | 7) && *byte == b'-'
                || !matches!(index, 4 | 7) && byte.is_ascii_digit()
        }) {
            continue;
        }
        if bytes
            .get(date_end)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'-')
        {
            continue;
        }
        let Ok(raw_date) = std::str::from_utf8(candidate) else {
            continue;
        };
        let Ok(date) = NaiveDate::parse_from_str(raw_date, "%Y-%m-%d") else {
            continue;
        };
        let timestamp = Utc.from_utc_datetime(
            &date
                .and_hms_opt(0, 0, 0)
                .expect("a parsed date always has a valid midnight"),
        );
        let phrase = std::str::from_utf8(&bytes[phrase_start..date_end])
            .unwrap_or("as of <date>")
            .to_string();
        inferred.entry(timestamp).or_insert(phrase);
    }
    inferred.into_iter().collect()
}

fn resolve_validity_range(
    content: &str,
    valid_from: Option<&str>,
    valid_until: Option<&str>,
) -> Result<ValidityResolution, String> {
    let mut range = parse_validity_range(valid_from, valid_until)?;
    let explicit = valid_from.is_some() || valid_until.is_some();
    let inferred = infer_as_of_dates(content);
    let mut source = if explicit { "explicit" } else { "none" };
    let mut inferred_phrase = None;
    let mut ambiguous_phrases = Vec::new();

    if range.from.is_none() {
        if inferred.len() == 1 {
            let (timestamp, phrase) = inferred[0].clone();
            if range.until.is_some_and(|until| until <= timestamp) {
                // An incidental prose date at or after the explicit validUntil
                // is not caller intent; abstain from inference instead of
                // failing the whole ingest, and surface the skipped phrase.
                ambiguous_phrases.push(phrase);
                source = "inferred_as_of_conflicts_with_explicit_validity_ignored";
            } else {
                range.from = Some(timestamp);
                inferred_phrase = Some(phrase);
                source = if explicit {
                    "explicit_and_inferred_as_of"
                } else {
                    "inferred_as_of"
                };
            }
        } else if inferred.len() > 1 {
            ambiguous_phrases = inferred.into_iter().map(|(_, phrase)| phrase).collect();
            source = if explicit {
                "explicit_with_ambiguous_as_of_ignored"
            } else {
                "ambiguous_as_of_not_applied"
            };
        }
    }
    Ok(ValidityResolution {
        range,
        source,
        inferred_phrase,
        ambiguous_phrases,
    })
}

/// Default lifetime of a `state` memory when the caller gives no `validUntil`.
/// Overridable with `VESTIGE_STATE_TTL_DAYS`; `0` disables the default.
const DEFAULT_STATE_TTL_DAYS: i64 = 30;

fn state_ttl_days() -> i64 {
    std::env::var("VESTIGE_STATE_TTL_DAYS")
        .ok()
        .and_then(|raw| raw.trim().parse::<i64>().ok())
        .filter(|days| *days >= 0)
        .unwrap_or(DEFAULT_STATE_TTL_DAYS)
}

/// "Current state" memories (version numbers, progress percentages,
/// inventories) are the rot class that pollutes recall worst: they never stop
/// being true in the store long after they stopped being true in the world.
/// A `state` node therefore expires by default instead of by discipline.
/// An explicit `validUntil` always wins; other node types are untouched.
fn apply_state_ttl(
    resolution: &mut ValidityResolution,
    node_type: Option<&str>,
    now: DateTime<Utc>,
) {
    if !node_type.is_some_and(|kind| kind.eq_ignore_ascii_case("state")) {
        return;
    }
    if resolution.range.until.is_some() {
        return;
    }
    let ttl = state_ttl_days();
    if ttl == 0 {
        return;
    }
    let until = now + chrono::Duration::days(ttl);
    if resolution.range.from.is_some_and(|from| from >= until) {
        return;
    }
    resolution.range.until = Some(until);
    if resolution.source == "none" {
        resolution.source = "state_default_ttl";
    }
}

fn validity_response(resolution: &ValidityResolution) -> Value {
    serde_json::json!({
        "validFrom": resolution.range.from.map(|value| value.to_rfc3339()),
        "validUntil": resolution.range.until.map(|value| value.to_rfc3339()),
        "source": resolution.source,
        "inferredPhrase": resolution.inferred_phrase,
        "ambiguousPhrases": resolution.ambiguous_phrases,
    })
}

#[derive(Debug, Clone)]
struct TagSuggestionReport {
    suggestions: Vec<Value>,
    status: Value,
}

fn similar_tag_suggestions(
    storage: &Arc<Storage>,
    scope: &str,
    requested: &[String],
) -> TagSuggestionReport {
    if requested.is_empty() {
        return TagSuggestionReport {
            suggestions: Vec::new(),
            status: serde_json::json!({
                "status": "complete",
                "scope": scope,
                "vocabularyScanned": false,
                "vocabularyCount": 0,
                "maximumVocabulary": 10_000,
                "requestedTagsTruncated": false,
                "ignoredOverlongInputTags": 0,
                "ignoredOverlongVocabularyTags": 0,
                "ignoredSecretShapedVocabularyTags": 0,
                "unicodeNormalization": "NFKC plus Unicode lowercase",
            }),
        };
    }
    let status = match storage.tag_vocabulary(Some(scope)) {
        Ok(vocabulary) => serde_json::json!({
            "status": "complete",
            "scope": scope,
            "vocabularyScanned": true,
            "vocabularyCount": vocabulary.tags.len(),
            "maximumVocabulary": 10_000,
            "requestedTagsTruncated": requested.len() > 50,
            "ignoredOverlongInputTags": 0,
            "ignoredOverlongVocabularyTags": vocabulary.skipped_overlong,
            "ignoredSecretShapedVocabularyTags": 0,
            "unicodeNormalization": "NFKC plus Unicode lowercase",
        }),
        Err(error) => {
            let text = error.to_string();
            if text.contains(crate::tools::unavailable::SIMILARITY_DISABLED) {
                crate::tools::unavailable::tag_suggestions_on_strata(scope)
            } else {
                serde_json::json!({
                    "status": "unavailable",
                    "reason": text,
                    "scope": scope,
                })
            }
        }
    };
    TagSuggestionReport {
        suggestions: Vec::new(),
        status,
    }
}

fn apply_accepted_tag_suggestions(
    tags: Vec<String>,
    report: &TagSuggestionReport,
    accepted: Option<&std::collections::BTreeMap<String, String>>,
) -> Result<Vec<String>, String> {
    let Some(accepted) = accepted else {
        return Ok(tags);
    };
    if accepted.iter().any(|(input, existing)| {
        [input.as_str(), existing.as_str()]
            .into_iter()
            .any(|value| {
                scan_secrets(value)
                    .into_iter()
                    .any(|finding| finding.blocks_ingestion())
            })
    }) {
        return Err(
            "Refused acceptedTagSuggestions containing a probable credential; secret bytes were not stored, logged, or returned"
                .to_string(),
        );
    }
    for (input, existing) in accepted {
        let valid = report.suggestions.iter().any(|suggestion| {
            suggestion["inputTag"].as_str() == Some(input.as_str())
                && suggestion["similarExistingTag"].as_str() == Some(existing.as_str())
        });
        if !valid {
            return Err(format!(
                "acceptedTagSuggestions mapping '{input}' -> '{existing}' is not a current same-scope suggestion; run previewTagSuggestions again"
            ));
        }
        if !tags.iter().any(|tag| tag == input) {
            return Err(format!(
                "acceptedTagSuggestions source '{input}' is not present in tags"
            ));
        }
    }

    let mut rewritten = Vec::with_capacity(tags.len());
    let mut seen = std::collections::HashSet::new();
    for tag in tags {
        let canonical = accepted.get(&tag).cloned().unwrap_or(tag);
        if seen.insert(canonical.clone()) {
            rewritten.push(canonical);
        }
    }
    Ok(rewritten)
}

/// Longest `mergePreview` shipped in a response. The full merged text is the
/// memory now stored under `nodeId`, one `memory(action='get')` away.
const MERGE_PREVIEW_CHARS: usize = 240;

/// Entity spans a lean create response still carries (spec B4: keep top 8).
const LEAN_ENTITY_SPANS: usize = 8;

pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args = personal_metadata(args)?;
    // Agents get the proof fields by default; `standard` and `full` add the
    // entities, importance and auto-connect detail on request.
    let compact_profile = args
        .as_ref()
        .and_then(|v| v.get("outputProfile").or_else(|| v.get("output_profile")))
        .and_then(Value::as_str)
        .is_none_or(|p| p.eq_ignore_ascii_case("compact"));
    let (args, links) = take_links(storage, args)?;
    let (args, intents) = take_intent_ids(args)?;
    let mut value = execute_verbose(storage, cognitive, args, &intents).await?;
    write_links(storage, &mut value, &links);
    auto_connect_saved(storage, &mut value);
    lean_response(&mut value);
    if compact_profile {
        compact_proof_response(&mut value);
    }
    if let Some(results) = value.get_mut("results").and_then(Value::as_array_mut) {
        for item in results {
            lean_response(item);
            if compact_profile {
                compact_proof_response(item);
            }
        }
    }
    Ok(value)
}

/// Declared links per saved memory: `None` for the single form, `Some(i)` for
/// batch item `i`.
type PendingLinks = Vec<(Option<usize>, Vec<(strata_memory::DeclaredLink, String)>)>;

/// Parse and check `links` (top-level and per batch item) before anything is
/// written, and strip them from the arguments the ingest path sees.
fn take_links(
    storage: &Arc<Storage>,
    args: Option<Value>,
) -> Result<(Option<Value>, PendingLinks), String> {
    let Some(mut args) = args else {
        return Ok((None, Vec::new()));
    };
    let default_scope = args
        .get("scope")
        .and_then(Value::as_str)
        .unwrap_or(vestige_core::DEFAULT_MEMORY_SCOPE)
        .to_string();
    let mut pending: PendingLinks = Vec::new();
    if let Some(raw) = args.as_object_mut().and_then(|map| map.remove("links")) {
        pending.push((None, parse_links(&raw)?));
    }
    if let Some(items) = args.get_mut("items").and_then(Value::as_array_mut) {
        for (index, item) in items.iter_mut().enumerate() {
            if let Some(raw) = item.as_object_mut().and_then(|map| map.remove("links")) {
                pending.push((Some(index), parse_links(&raw)?));
            }
        }
    }
    pending.retain(|(_, links)| !links.is_empty());
    if pending.is_empty() {
        return Ok((Some(args), pending));
    }
    if !strata_memory::is_strata_backend(storage.as_ref()) {
        return Err("links need a Strata log; this store has none".into());
    }
    for (index, links) in &pending {
        let scope = index
            .and_then(|i| args["items"][i].get("scope").and_then(Value::as_str))
            .unwrap_or(&default_scope);
        strata_memory::check_links(storage.as_ref(), scope, links)?;
    }
    Ok((Some(args), pending))
}

fn parse_links(raw: &Value) -> Result<Vec<(strata_memory::DeclaredLink, String)>, String> {
    let Some(entries) = raw.as_array() else {
        return Err("links must be an array of {kind, to}".into());
    };
    if entries.len() > MAX_DECLARED_LINKS {
        return Err(format!("at most {MAX_DECLARED_LINKS} links per memory"));
    }
    entries
        .iter()
        .map(|entry| {
            let kind = entry.get("kind").and_then(Value::as_str).unwrap_or("");
            let kind = strata_memory::DeclaredLink::parse(kind).ok_or_else(|| {
                format!(
                    "link kind '{kind}' is not declarable; use one of {}",
                    strata_memory::DeclaredLink::NAMES.join(", ")
                )
            })?;
            let to = entry
                .get("to")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|to| !to.is_empty())
                .ok_or("each link needs `to`: the id of an existing memory")?;
            Ok((kind, to.to_string()))
        })
        .collect()
}

/// Caller-declared idempotency ids (`intent_id`), taken from the arguments
/// before the ingest pipeline sees them — exactly the way [`take_links`]
/// strips `links` — so a replayed call short-circuits to the first outcome
/// with zero writes (INGEST-V5 spec B2.1/B2.2, B3.3).
#[derive(Debug, Default)]
struct TakenIntents {
    /// The single-mode `intent_id`.
    single: Option<String>,
    /// Per batch-item `intent_id` by caller index; `None` where undeclared.
    items: Vec<Option<String>>,
}

/// Take `intent_id` (preferred) or its `intentId` alias, removing BOTH
/// spellings so neither spelling leaks past this gate.
fn remove_intent_id(map: &mut serde_json::Map<String, Value>) -> Option<Value> {
    let primary = map.remove("intent_id");
    let alias = map.remove("intentId");
    primary.or(alias)
}

/// Take and validate `intent_id` (top level and per batch item) before
/// anything is written. An invalid pattern fails the whole call up front,
/// the same contract `take_links` gives `links`.
fn take_intent_ids(args: Option<Value>) -> Result<(Option<Value>, TakenIntents), String> {
    let Some(mut args) = args else {
        return Ok((None, TakenIntents::default()));
    };
    let mut intents = TakenIntents::default();
    if let Some(raw) = args.as_object_mut().and_then(remove_intent_id) {
        intents.single = Some(validate_intent_id(raw)?);
    }
    if let Some(items) = args.get_mut("items").and_then(Value::as_array_mut) {
        for item in items.iter_mut() {
            let raw = item.as_object_mut().and_then(remove_intent_id);
            intents.items.push(raw.map(validate_intent_id).transpose()?);
        }
    }
    Ok((Some(args), intents))
}

/// `intent_id` is a capability-scoped id: 1..=128 chars of
/// `[A-Za-z0-9._:-]` (the schema's pattern, enforced here so the store's
/// intent index only ever sees one canonical shape).
fn validate_intent_id(raw: Value) -> Result<String, String> {
    let Some(id) = raw.as_str() else {
        return Err("intent_id must be a string".to_string());
    };
    if id.is_empty() || id.len() > 128 {
        return Err("intent_id must be 1 to 128 characters".to_string());
    }
    if !id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-'))
    {
        return Err("intent_id may only contain the characters [A-Za-z0-9._:-]".to_string());
    }
    Ok(id.to_string())
}

/// Write the declared links onto the memories the ingest just saved. The
/// memory is already admitted, so a refused link is reported on the response
/// (`linkError`) instead of failing the call.
fn write_links(storage: &Arc<Storage>, value: &mut Value, pending: &PendingLinks) {
    for (index, links) in pending {
        let slot = match index {
            None => Some(&mut *value),
            Some(i) => value
                .get_mut("results")
                .and_then(Value::as_array_mut)
                .and_then(|results| {
                    results.iter_mut().find(|result| {
                        result.get("index").and_then(Value::as_u64) == Some(*i as u64)
                    })
                }),
        };
        let Some(slot) = slot else { continue };
        // A repeat is recorded as an echo node, and the links the caller
        // declared with it are written from that echo.
        let Some(node_id) = slot
            .get("nodeId")
            .or_else(|| slot.get("echoNodeId"))
            .and_then(Value::as_str)
            .map(str::to_owned)
        else {
            slot["linkError"] = serde_json::json!("nothing was saved, so no link was written");
            continue;
        };
        match strata_memory::save_links(storage.as_ref(), &node_id, links) {
            Ok(written) => {
                let mut all = slot
                    .get("links")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                if let Some(written) = serde_json::json!(written).as_array() {
                    all.extend(written.iter().cloned());
                }
                slot["links"] = Value::Array(all);
            }
            Err(err) => slot["linkError"] = serde_json::json!(err),
        }
    }
}

/// Most declared links one memory may carry.
const MAX_DECLARED_LINKS: usize = 16;

/// Tags an echo node may carry, including the literal `duplicate` tag.
/// The rest are the submission's tags, trimmed, emptied, deduped, and cut
/// in byte order so the cap does not depend on the order they were sent.
const MAX_ECHO_TAGS: usize = 16;

/// `duplicate` once, then at most [`MAX_ECHO_TAGS`] - 1 other tags, in order
/// that is the same for every permutation of `stored`.
fn bounded_echo_tags(stored: &[String]) -> Vec<String> {
    let mut extras: Vec<String> = stored
        .iter()
        .map(|tag| tag.trim().to_string())
        .filter(|tag| !tag.is_empty() && tag != "duplicate")
        .collect();
    extras.sort_unstable();
    extras.dedup();
    extras.truncate(MAX_ECHO_TAGS.saturating_sub(1));
    let mut tags = Vec::with_capacity(1 + extras.len());
    tags.push("duplicate".to_string());
    tags.extend(extras);
    tags.sort_unstable();
    tags.dedup();
    tags
}

/// Auto-connect every memory this ingest saved: the ingest-time share of
/// `vestige connect`, joined onto the saved nodes through the shared
/// `crate::auto_connect` module. Walks the same response slots `write_links`
/// does — the single form's `nodeId`, then every batch result — so one hook
/// covers every save path. Like declared links, a refused edge is reported on
/// the slot (`autoConnectError`) instead of failing an ingest that already
/// succeeded; `autoConnect` itself ships only when edges were written or a
/// tag was skipped (each skipped tag is named with its reason).
fn auto_connect_saved(storage: &Arc<Storage>, value: &mut Value) {
    if let Some(node_id) = value
        .get("nodeId")
        .and_then(Value::as_str)
        .map(str::to_owned)
    {
        let scope = value
            .get("scope")
            .and_then(Value::as_str)
            .unwrap_or(DEFAULT_MEMORY_SCOPE)
            .to_string();
        auto_connect_slot(storage, value, &node_id, &scope);
    }
    if let Some(results) = value.get_mut("results").and_then(Value::as_array_mut) {
        for slot in results {
            let Some(node_id) = slot
                .get("nodeId")
                .and_then(Value::as_str)
                .map(str::to_owned)
            else {
                continue;
            };
            let scope = slot
                .get("scope")
                .and_then(Value::as_str)
                .unwrap_or(DEFAULT_MEMORY_SCOPE)
                .to_string();
            auto_connect_slot(storage, slot, &node_id, &scope);
        }
    }
}

/// Most joined pairs one save's `autoConnect` block names (strongest first).
/// `pairsTotal` counts them all, and `receiptId` lists every edge written.
const AUTO_CONNECT_PAIRS_SHOWN: usize = 5;

/// Auto-connect one saved memory and attach the report to its response slot.
/// The node is read back from the store so the memory's own (post-gate)
/// content and tags drive the identity extraction. Each written edge is
/// reported with the exact identities that joined it.
fn auto_connect_slot(storage: &Arc<Storage>, slot: &mut Value, node_id: &str, scope: &str) {
    let node = match storage.get_node(node_id) {
        Ok(node) => node,
        Err(err) => {
            slot["autoConnectError"] =
                serde_json::json!(format!("auto-connect could not read {node_id} back: {err}"));
            return;
        }
    };
    let Some(node) = node else {
        return;
    };
    match crate::auto_connect::auto_connect_new_memory(
        storage.as_ref(),
        node_id,
        scope,
        &node.content,
        &node.tags,
    ) {
        // A pass that wrote nothing but skipped a tag still reports: a tag is
        // never skipped silently.
        Ok(report) if report.edges > 0 || !report.skipped.is_empty() => {
            let mut auto_obj = serde_json::json!({
                "edges": report.edges,
                "sharedIdentities": report.shared_identities,
                // The strongest pairs only: one save can join up to
                // MAX_AUTO_EDGES memories. Every edge is written either way;
                // the receipt below lists them all.
                "pairsTotal": report.pairs.len(),
                "pairsShown": report.pairs.len().min(AUTO_CONNECT_PAIRS_SHOWN),
                "pairs": report
                    .pairs
                    .iter()
                    .take(AUTO_CONNECT_PAIRS_SHOWN)
                    .map(|pair| serde_json::json!({
                        "source": pair.source_id,
                        "target": pair.target_id,
                        "joinedOn": pair.identities,
                    }))
                    .collect::<Vec<_>>(),
                // The one write that admitted every edge above.
                "receiptId": report.receipt_id,
            });
            if !report.skipped_common_tags.is_empty() {
                auto_obj["skippedCommonTags"] = serde_json::json!(report.skipped_common_tags);
            }
            if !report.skipped.is_empty() {
                auto_obj["skipped"] = serde_json::json!(
                    report
                        .skipped
                        .iter()
                        .map(|skipped| serde_json::json!({
                            "tag": skipped.tag,
                            "carriers": skipped.carriers,
                            "scopeSize": skipped.scope_size,
                            "reason": skipped.reason.to_string(),
                        }))
                        .collect::<Vec<_>>()
                );
            }
            if report.not_linked > 0 {
                auto_obj["notLinked"] = serde_json::json!(report.not_linked);
            }
            slot["autoConnect"] = auto_obj;
        }
        Ok(_) => {}
        Err(err) => slot["autoConnectError"] = serde_json::json!(err),
    }
}

// ============================================================================
// The proof-carrying write path.
// Determinism law: nothing below reads a clock or any randomness to decide
// anything; every response field is a log fact or a pure function of the
// submitted bytes.
// ============================================================================

/// Most entity spans a create response carries (spec B2.4). The extractor's
/// own cap is higher; responses trim further under [`lean_response`].
const MAX_RESPONSE_ENTITY_SPANS: usize = 32;

/// The zero-write replay response (spec B2.2). `requestCanonicalHash` is the
/// canonical hash of the CURRENT submission, so comparing it with the
/// `canonicalHash` the original create returned makes a divergent replay
/// visible instead of hidden (spec B5.5) — idempotency wins, but it never
/// pretends the bytes matched.
fn intent_replay_response(
    scope: &str,
    intent_id: &str,
    replay_of: &str,
    effect_seq: u64,
    digest: &str,
    submitted: &str,
) -> Value {
    serde_json::json!({
        "success": true,
        "decision": "replay",
        "replayed": true,
        "replayOf": replay_of,
        "effectSeq": effect_seq,
        "intentDigest": digest,
        "intentId": intent_id,
        "requestCanonicalHash": strata_store::canonical::canonical_hash_hex(submitted),
        "scope": scope,
        "message": "intent_id is already recorded for this scope: the first outcome is replayed and nothing was written",
    })
}

/// Proof enrichment for every create (spec B2.4): `receiptId` is a log fact;
/// `canonicalHash`, `entities` and `importance` are pure functions of the
/// submitted bytes (plus the tags actually stored, for importance).
fn enrich_create_response(
    storage: &Arc<Storage>,
    response: &mut Value,
    node_id: &str,
    content: &str,
    tags: &[String],
) {
    if let Some(receipt) = storage.latest_receipt_id_for_node(node_id) {
        response["receiptId"] = serde_json::json!(receipt);
    }
    response["canonicalHash"] =
        serde_json::json!(strata_store::canonical::canonical_hash_hex(content));
    let spans = crate::intake::entities::extract_typed_spans(content);
    response["entities"] = entity_spans_json(&spans, MAX_RESPONSE_ENTITY_SPANS);
    response["importance"] = crate::intake::importance::compute_and_format(content, &spans, tags);
}

/// Entity spans serialized as `{kind, surface, byteStart, byteEnd}`, capped.
/// Byte offsets refer to the submitted content exactly as received.
fn entity_spans_json(spans: &[crate::intake::entities::EntitySpan], cap: usize) -> Value {
    serde_json::Value::Array(
        spans
            .iter()
            .take(cap)
            .map(|span| {
                serde_json::json!({
                    "kind": span.kind.as_str(),
                    "surface": span.surface,
                    "byteStart": span.byte_start,
                    "byteEnd": span.byte_end,
                })
            })
            .collect(),
    )
}

/// What one admitted write recorded, as [`attach_intent_record`] needs it
/// to register the write's `intent_id`.
struct IntentRecord<'a> {
    intent_id: Option<&'a str>,
    node_id: &'a str,
    effect_seq: Option<u64>,
    submitted_content: &'a str,
    submitted_source: &'a str,
    stored_tags: &'a [String],
}

/// Record the intent entry for a node this call just created (spec B2.4 /
/// B2.3-echo). The digest is always computed from the SUBMITTED bytes (the
/// caller's request is what idempotency keys on), whether the node it
/// resolves to is a regular create or a duplicate echo. The node is already
/// admitted, so a failed recording is reported on the response
/// (`intentError`, the same contract `write_links` gives `linkError`)
/// instead of failing an ingest that already succeeded.
fn attach_intent_record(
    storage: &Arc<Storage>,
    response: &mut Value,
    scope: &str,
    record: IntentRecord<'_>,
) {
    let IntentRecord {
        intent_id,
        node_id,
        effect_seq,
        submitted_content,
        submitted_source,
        stored_tags,
    } = record;
    let Some(intent_id) = intent_id else {
        return;
    };
    let digest = strata_store::canonical::intent_digest(
        &strata_store::canonical::canonical_hash_hex(submitted_content),
        submitted_source,
        stored_tags,
    );
    match storage.record_intent_entry(scope, intent_id, node_id, effect_seq.unwrap_or(0), &digest) {
        Ok(()) => response["intentId"] = serde_json::json!(intent_id),
        Err(err) => {
            response["intentError"] = serde_json::json!(format!(
                "memory saved but intent {intent_id} was not recorded, so a retry will not replay: {err}"
            ));
        }
    }
}

/// The canonical duplicate path (spec B2.3). Law 1 (append-only): the
/// original is NEVER merged, mutated, or returned in full. The submission
/// becomes a tiny echo node — `source: "duplicate"` so the store never
/// indexes it as a dedup target — plus one `evidence_of` edge to the
/// original, each with its own effect receipt, and the caller is told
/// `reinforce` with proof fields for the whole pipeline.
#[allow(clippy::too_many_arguments)]
fn reinforce_duplicate(
    storage: &Arc<Storage>,
    scope: &str,
    canonical_hash: &str,
    content: &str,
    intent_id: Option<&str>,
    original: &str,
    validity: &ValidityResolution,
    tag_suggestions: &TagSuggestionReport,
    accepted_tag_suggestions: &std::collections::BTreeMap<String, String>,
    submitted_source: &str,
    stored_tags: &[String],
) -> Result<Value, String> {
    // The echo carries ids and counters only — never a copy of the original
    // or the submitted bytes (law 5). It carries a bounded tag set: the
    // literal `duplicate` tag once, plus the submission's tags trimmed,
    // deduped and capped. A tag is the only handle a memory is found by, so
    // a repeat sent with a new tag must be findable under it until the cap.
    let echo_tags = bounded_echo_tags(stored_tags);
    let echo_content = format!(
        "duplicate of {original}\ncanonical_hash: {canonical_hash}\nsubmitted_line_count: {}",
        content.lines().count()
    );
    // Same echo content and the same bounded tags reuse the live echo.
    // Past the per-original cap, the earliest live echo is reused and
    // nothing is written. Either way the original stays untouched.
    let reused = strata_memory::reuse_duplicate_echo(
        storage.as_ref(),
        scope,
        original,
        &echo_content,
        &echo_tags,
    );
    let (echo_id, echo_receipt, echo_seq) = if let Some(echo_id) = reused {
        (
            echo_id.clone(),
            storage.latest_receipt_id_for_node(&echo_id),
            storage.node_effect_seq(&echo_id),
        )
    } else {
        let echo = storage
            .ingest_in_scope_with_secret_policy(
                IngestInput {
                    content: echo_content,
                    node_type: "note".to_string(),
                    source: Some("duplicate".to_string()),
                    sentiment_score: 0.0,
                    sentiment_magnitude: 0.0,
                    tags: echo_tags,
                    valid_from: None,
                    valid_until: None,
                    validity_inferred: false,
                    source_envelope: None,
                },
                scope,
                SecretPolicy::Reject,
            )
            .map_err(|e| e.to_string())?;
        let echo_id = echo.id;
        // Resolve the echo's own create proof BEFORE the edge lands, so
        // `receiptId` and the intent's effect_seq name the node's create, not
        // the edge that follows it.
        let echo_receipt = storage.latest_receipt_id_for_node(&echo_id);
        let echo_seq = storage.node_effect_seq(&echo_id);
        (echo_id, echo_receipt, echo_seq)
    };
    let edge_exists = storage
        .get_connections_for_memory(&echo_id)
        .map(|edges| {
            edges.iter().any(|edge| {
                edge.link_type == "evidence_of"
                    && edge.source_id == echo_id
                    && edge.target_id == original
            })
        })
        .unwrap_or(false);
    let edge = if edge_exists {
        Ok(Vec::new())
    } else {
        strata_memory::save_links(
            storage.as_ref(),
            &echo_id,
            &[(
                strata_memory::DeclaredLink::EvidenceOf,
                original.to_string(),
            )],
        )
    };
    let mut response = serde_json::json!({
        "success": true,
        "decision": "reinforce",
        "scope": scope,
        "duplicateOf": original,
        "echoNodeId": echo_id.clone(),
        "canonicalHash": canonical_hash,
        "rawBytes": content.len(),
        "pipeline": strata_store::canonical::CANONICAL_PIPELINE_VERSION,
        "receiptId": echo_receipt,
        "message": "Canonically identical memory already exists: an echo node and an evidence_of edge were recorded and the original was left untouched",
        "validity": validity_response(validity),
        "tagSuggestions": tag_suggestions.suggestions,
        "tagSuggestionStatus": tag_suggestions.status,
        "acceptedTagSuggestions": accepted_tag_suggestions,
    });
    match edge {
        Ok(written) => response["links"] = serde_json::json!(written),
        Err(err) => response["linkError"] = serde_json::json!(err),
    }
    attach_intent_record(
        storage,
        &mut response,
        scope,
        IntentRecord {
            intent_id,
            node_id: &echo_id,
            effect_seq: echo_seq,
            submitted_content: content,
            submitted_source,
            stored_tags,
        },
    );
    Ok(response)
}

/// Trim one ingest response object to what the caller can act on.
///
/// On the real store an update decision echoed the full merged memory twice
/// (`previousContent` and `mergePreview`), about 13 KB for one save, and every
/// response carried a 311-byte tag-status block that said nothing. The rules:
///
/// - `mergePreview` is the content now stored under `nodeId`, so it becomes a
///   240-character preview plus `mergedContentLength`; `memory(action='get')`
///   has the rest.
/// - `previousContent` stays whole. A merge replaces the old text and nothing
///   else keeps a copy, so this field is the only way to recover it.
/// - `tagSuggestionStatus` ships only when it has something to say (a status
///   other than complete, a truncation, an ignored tag, or a suggestion), or
///   when the response is a `previewTagSuggestions` preflight, whose contract
///   in docs/MEMORY_HYGIENE.md includes it.
/// - Empty suggestion lists, an empty `validity` block and null decision
///   fields are dropped; absent and null read the same to every caller.
fn lean_response(value: &mut Value) {
    let Some(obj) = value.as_object_mut() else {
        return;
    };
    for key in [
        "similarity",
        "predictionError",
        "supersededId",
        "mergedFrom",
        "mergePreview",
        "previousContent",
        "autoClosedUntil",
        "reconsolidation",
    ] {
        if obj.get(key).is_some_and(Value::is_null) {
            obj.remove(key);
        }
    }
    if let Some(preview) = obj.get("mergePreview").and_then(Value::as_str) {
        let total = preview.chars().count();
        if total > MERGE_PREVIEW_CHARS {
            let short: String = preview.chars().take(MERGE_PREVIEW_CHARS).collect();
            obj.insert("mergePreview".to_string(), Value::String(short));
            obj.insert("mergePreviewTruncated".to_string(), Value::Bool(true));
        }
        obj.insert("mergedContentLength".to_string(), Value::from(total));
    }
    // INGEST-V5 (spec B4) proof-field policy, matching how the optional
    // fields above are treated: `receiptId` and `canonicalHash` always
    // survive — they ARE the proof a caller verifies. `entities` trims to
    // the first 8 spans and an emptied array is dropped like the other
    // say-nothing collections (the full set stays derivable from the
    // submitted bytes via the version-pinned extractor). `importance`
    // keeps only `score` + `weightsVersion`: the factors block is
    // recomputable from the same bytes and never acted on by a caller.
    if let Some(entities) = obj.get_mut("entities").and_then(Value::as_array_mut) {
        entities.truncate(LEAN_ENTITY_SPANS);
        if entities.is_empty() {
            obj.remove("entities");
        }
    }
    if let Some(importance) = obj.get_mut("importance").and_then(Value::as_object_mut) {
        let score = importance.get("score").cloned();
        let weights_version = importance.get("weightsVersion").cloned();
        importance.clear();
        if let Some(score) = score {
            importance.insert("score".to_string(), score);
        }
        if let Some(weights_version) = weights_version {
            importance.insert("weightsVersion".to_string(), weights_version);
        }
    }
    if obj
        .get("tagSuggestions")
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty)
    {
        obj.remove("tagSuggestions");
    }
    if obj
        .get("acceptedTagSuggestions")
        .and_then(Value::as_object)
        .is_some_and(|m| m.is_empty())
    {
        obj.remove("acceptedTagSuggestions");
    }
    let is_preflight = obj.contains_key("wouldWrite");
    let has_suggestions = obj.contains_key("tagSuggestions");
    if !is_preflight
        && !has_suggestions
        && obj
            .get("tagSuggestionStatus")
            .is_some_and(tag_status_reports_nothing)
    {
        obj.remove("tagSuggestionStatus");
    }
    if obj.get("validity").is_some_and(validity_is_empty) {
        obj.remove("validity");
    }
    if obj.get("synapticCapture").is_some_and(Value::is_null) {
        obj.remove("synapticCapture");
    }
}

fn compact_proof_response(value: &mut Value) {
    let Some(obj) = value.as_object_mut() else {
        return;
    };
    for key in [
        "entities",
        "importance",
        "importanceScore",
        "predictionError",
        "hasEmbedding",
        "embeddingsCompiledIn",
        "dedup",
        "message",
        "reason",
        "tagSuggestionStatus",
        "synapticCapture",
        "actor",
    ] {
        obj.remove(key);
    }
    if let Some(auto) = obj.get_mut("autoConnect").and_then(Value::as_object_mut) {
        let edges = auto.get("edges").cloned();
        let receipt_id = auto.get("receiptId").cloned();
        auto.clear();
        if let Some(edges) = edges {
            auto.insert("edges".to_string(), edges);
        }
        if let Some(receipt_id) = receipt_id {
            auto.insert("receiptId".to_string(), receipt_id);
        }
    }
}

fn tag_status_reports_nothing(status: &Value) -> bool {
    status["status"] == "complete"
        && status["requestedTagsTruncated"] == false
        && [
            "ignoredOverlongInputTags",
            "ignoredOverlongVocabularyTags",
            "ignoredSecretShapedVocabularyTags",
        ]
        .iter()
        .all(|key| status[*key].as_u64() == Some(0))
}

fn validity_is_empty(validity: &Value) -> bool {
    validity["source"] == "none"
        && validity["inferredPhrase"].is_null()
        && validity["validFrom"].is_null()
        && validity["validUntil"].is_null()
        && validity["ambiguousPhrases"]
            .as_array()
            .is_some_and(Vec::is_empty)
}

/// A scope is echoed by every write and preview response and names the
/// namespace reads are filtered by, so a credential-shaped scope is refused
/// before anything runs, under every secret policy. Reports kinds only.
fn scope_credential_refusal(scope: &str) -> Option<String> {
    let kinds: Vec<String> = scan_secrets(scope)
        .into_iter()
        .filter(|finding| finding.blocks_ingestion())
        .map(|finding| finding.kind.to_string())
        .collect();
    (!kinds.is_empty()).then(|| StorageError::SecretDetected { kinds }.to_string())
}

async fn execute_verbose(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
    intents: &TakenIntents,
) -> Result<Value, String> {
    if args
        .as_ref()
        .is_some_and(|value| value.get("content").is_some() && value.get("items").is_some())
    {
        return Err("Provide either content or items, never both; no memories were stored".into());
    }
    let args: SmartIngestArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };
    let scope = args
        .scope
        .unwrap_or_else(|| DEFAULT_MEMORY_SCOPE.to_string());
    if let Some(refusal) = scope_credential_refusal(&scope) {
        return Err(refusal);
    }
    // #252 Phase A: the claimed role. Trimmed here once; it can never
    // override the process identity — the operator policy resolves it.
    let claimed_role = args
        .role
        .as_deref()
        .map(str::trim)
        .filter(|role| !role.is_empty())
        .map(str::to_string);

    // Detect mode: batch (items present) vs single (content present)
    if let Some(items) = args.items {
        // INGEST-V5: the default batch policy is "smart". The canonical gate
        // makes it safe — a duplicate reinforces through an echo node and an
        // evidence_of edge, never a merge — so an unspecified policy no
        // longer skips dedup. Callers keep "force_create" (or an explicit
        // forceCreate) to bypass the gate on purpose.
        let batch_merge_policy = args
            .batch_merge_policy
            .unwrap_or_else(|| "smart".to_string());
        let default_force_create = match batch_merge_policy.as_str() {
            "force_create" => true,
            "smart" => false,
            other => {
                return Err(format!(
                    "Invalid batchMergePolicy '{}'. Must be 'force_create' or 'smart'.",
                    other
                ));
            }
        };
        let global_force = match args.force_create {
            // An EXPLICIT forceCreate is authoritative and must be honored in both
            // policies. Previously `Some(false)` under the default 'force_create'
            // policy fell through to `default_force_create` (= true), silently
            // inverting the caller's explicit false into a force-create.
            Some(explicit) => explicit,
            None => default_force_create,
        };
        return execute_batch(
            storage,
            cognitive,
            items,
            &intents.items,
            BatchSettings {
                force_create: global_force,
                merge_policy: &batch_merge_policy,
                default_scope: &scope,
            },
            claimed_role.as_deref(),
        )
        .await;
    }

    // Single mode: content is required
    let content = args.content.ok_or(
        "Missing 'content' field. Provide 'content' for single mode or 'items' for batch mode.",
    )?;

    // ====================================================================
    // INGEST-V5 B2.2 — idempotent intent replay. A recorded intent_id for
    // this scope replays the FIRST outcome and performs no write of any
    // kind. This short-circuits before the canonical gate and before any
    // storage write, and it outranks validation of the other fields: the
    // same logical call must always answer the same way, so a divergent
    // replay (different bytes) still replays — visibly, via
    // `requestCanonicalHash`.
    // ====================================================================
    if let Some(intent_id) = intents.single.as_deref()
        && let Some((replay_of, effect_seq, digest)) = storage
            .find_intent_record(&scope, intent_id)
            .map_err(|e| e.to_string())?
    {
        return Ok(intent_replay_response(
            &scope, intent_id, &replay_of, effect_seq, &digest, &content,
        ));
    }

    let mut validity = resolve_validity_range(
        &content,
        args.valid_from.as_deref(),
        args.valid_until.as_deref(),
    )?;
    apply_state_ttl(&mut validity, args.node_type.as_deref(), Utc::now());
    let secret_policy = if args.allow_secrets.unwrap_or(false) {
        SecretPolicy::AllowExplicitly
    } else {
        SecretPolicy::Reject
    };

    // Validate content
    if content.trim().is_empty() {
        return Err("Content cannot be empty".to_string());
    }

    if content.len() > 1_000_000 {
        return Err("Content too large (max 1MB)".to_string());
    }

    // ====================================================================
    // COGNITIVE PRE-INGEST: importance scoring + intent detection + content analysis
    // ====================================================================
    let mut importance_composite = 0.0_f64;
    let mut importance_snapshot = SynapticSignalSnapshot {
        novelty: 0.0,
        arousal: 0.0,
        reward: 0.0,
        attention: 0.0,
        composite: 0.0,
    };
    let requested_tags = args.tags.clone().unwrap_or_default();
    let tag_suggestions = similar_tag_suggestions(storage, &scope, &requested_tags);
    if args.preview_tag_suggestions.unwrap_or(false) {
        return Ok(serde_json::json!({
            "success": true,
            "decision": "preflight",
            "wouldWrite": false,
            "scope": scope,
            "validity": validity_response(&validity),
            "tagSuggestions": tag_suggestions.suggestions,
            "tagSuggestionStatus": tag_suggestions.status,
            "nextStep": "Accept any mapping explicitly with acceptedTagSuggestions, or ingest unchanged without it. No tags were auto-applied and no memory was stored."
        }));
    }
    let accepted_tag_suggestions = args.accepted_tag_suggestions.clone().unwrap_or_default();
    let mut tags = apply_accepted_tag_suggestions(
        args.tags.unwrap_or_default(),
        &tag_suggestions,
        (!accepted_tag_suggestions.is_empty()).then_some(&accepted_tag_suggestions),
    )?;

    if let Ok(cog) = cognitive.try_lock() {
        // 4A. Full 4-channel importance scoring
        let context = ImportanceContext::current();
        let importance = cog
            .importance_signals
            .compute_importance(&content, &context);
        importance_composite = importance.composite;
        importance_snapshot = SynapticSignalSnapshot {
            novelty: importance.novelty,
            arousal: importance.arousal,
            reward: importance.reward,
            attention: importance.attention,
            composite: importance.composite,
        };

        // 4B. Intent detection → auto-tag
        let intent_result = cog.intent_detector.detect_intent();
        if intent_result.confidence > 0.5 {
            let intent_tag = format!("intent:{:?}", intent_result.primary_intent);
            // Truncate long intent tags
            let intent_tag = if intent_tag.len() > 50 {
                format!("{}...", &intent_tag[..intent_tag.floor_char_boundary(47)])
            } else {
                intent_tag
            };
            tags.push(intent_tag);
        }
    }

    // INGEST-V5: the submitted source and the canonical hash of the
    // submitted bytes feed the intent digest and the response proof fields;
    // captured here before `input` consumes them. Pure byte functions — no
    // clock, no randomness.
    let submitted_source = args.source.clone().unwrap_or_default();
    let canonical_hash = strata_store::canonical::canonical_hash_hex(&content);

    let input = IngestInput {
        content: content.clone(),
        node_type: args.node_type.unwrap_or_else(|| "fact".to_string()),
        source: args.source,
        sentiment_score: 0.0,
        // Store importance composite as sentiment_magnitude for FSRS encoding boost
        sentiment_magnitude: importance_composite,
        tags,
        valid_from: validity.range.from,
        valid_until: validity.range.until,
        validity_inferred: validity.inferred_phrase.is_some(),
        source_envelope: None,
    };

    // ====================================================================
    // INGEST (storage lock)
    // ====================================================================

    let hook_tags = input.tags.clone();

    // Check if force_create is enabled
    if args.force_create.unwrap_or(false) {
        let node = storage
            .ingest_in_scope_with_secret_policy(input, &scope, secret_policy)
            .map_err(|e| e.to_string())?;
        let node_id = node.id.clone();
        let node_content = node.content.clone();
        let node_type = node.node_type.clone();
        // INGEST-V5: resolved immediately after the create so the intent
        // entry points at THIS effect even if a post-hook records more.
        let effect_seq = storage.node_effect_seq(&node_id);

        // Post-ingest cognitive side effects
        let synaptic_capture = run_post_ingest_with_snapshot(
            storage,
            cognitive,
            &node_id,
            &node_content,
            &node_type,
            importance_composite,
            importance_snapshot.clone(),
        );

        let failure_hooks =
            run_failure_hooks(storage, &node_id, &node_content, &hook_tags, &scope).await;
        let mut response = serde_json::json!({
            "success": true,
            "decision": "create",
            "nodeId": node_id,
            "scope": scope,
            "synapticCapture": synaptic_capture,
            "validity": validity_response(&validity),
            "tagSuggestions": tag_suggestions.suggestions,
            "tagSuggestionStatus": tag_suggestions.status,
            "acceptedTagSuggestions": accepted_tag_suggestions,
        });
        // INGEST-V5 (B2.4/B2.5): forceCreate skips only the canonical gate;
        // the create still carries the proof fields and records the intent.
        enrich_create_response(storage, &mut response, &node_id, &content, &hook_tags);
        attach_intent_record(
            storage,
            &mut response,
            &scope,
            IntentRecord {
                intent_id: intents.single.as_deref(),
                node_id: &node_id,
                effect_seq,
                submitted_content: &content,
                submitted_source: &submitted_source,
                stored_tags: &hook_tags,
            },
        );
        if claimed_role.is_some() {
            crate::actor_surface::attach_actor_block(
                &mut response,
                storage,
                claimed_role.as_deref(),
            );
        }
        attach_failure_hooks(&mut response, failure_hooks);
        return Ok(response);
    }

    // Use smart ingest with prediction error gating
    // Tags are consulted after `input` is consumed by the ingest below, by the
    // failure hooks (backfill and failure feedback).
    let hook_tags: Vec<String> = input.tags.clone();

    {
        // ================================================================
        // INGEST-V5 B2.3 — canonical duplicate gate. A canonically identical
        // memory never merges or mutates the original (law 1): the write
        // becomes an echo node + evidence_of edge and reports `reinforce`.
        // This branch is only reached when forceCreate is false (B2.5), so
        // the gate is armed exactly where the caller did not opt out.
        // ================================================================
        if let Some(original) = storage
            .find_duplicate_by_canonical_hash(&scope, &content)
            .map_err(|e| e.to_string())?
        {
            return reinforce_duplicate(
                storage,
                &scope,
                &canonical_hash,
                &content,
                intents.single.as_deref(),
                &original,
                &validity,
                &tag_suggestions,
                &accepted_tag_suggestions,
                &submitted_source,
                &hook_tags,
            );
        }
        let node = storage
            .ingest_in_scope_with_secret_policy(input, &scope, secret_policy)
            .map_err(|e| e.to_string())?;
        let node_id = node.id.clone();
        let node_content = node.content.clone();
        let node_type = node.node_type.clone();
        // INGEST-V5: resolved immediately after the create so the intent
        // entry points at THIS effect even if a post-hook records more.
        let effect_seq = storage.node_effect_seq(&node_id);

        let synaptic_capture = run_post_ingest_with_snapshot(
            storage,
            cognitive,
            &node_id,
            &node_content,
            &node_type,
            importance_composite,
            importance_snapshot,
        );

        let failure_hooks =
            run_failure_hooks(storage, &node_id, &node_content, &hook_tags, &scope).await;
        let mut response = serde_json::json!({
            "success": true,
            "decision": "create",
            "nodeId": node_id,
            "scope": scope,
            "synapticCapture": synaptic_capture,
            "validity": validity_response(&validity),
            "tagSuggestions": tag_suggestions.suggestions,
            "tagSuggestionStatus": tag_suggestions.status,
            "acceptedTagSuggestions": accepted_tag_suggestions,
        });
        // INGEST-V5 (B2.4): proof fields for the create.
        enrich_create_response(storage, &mut response, &node_id, &content, &hook_tags);
        attach_intent_record(
            storage,
            &mut response,
            &scope,
            IntentRecord {
                intent_id: intents.single.as_deref(),
                node_id: &node_id,
                effect_seq,
                submitted_content: &content,
                submitted_source: &submitted_source,
                stored_tags: &hook_tags,
            },
        );
        if claimed_role.is_some() {
            crate::actor_surface::attach_actor_block(
                &mut response,
                storage,
                claimed_role.as_deref(),
            );
        }
        attach_failure_hooks(&mut response, failure_hooks);
        Ok(response)
    }
}

/// A kill switch read the way the other Vestige gates are: unset, empty or
/// anything unrecognised means ON; only `0`, `false`, `off`, `no` turn it off.
/// An on-by-default lever: unset means on, and only an explicit
/// `0`/`false`/`off`/`no` turns it off.
fn env_flag_enabled(name: &str) -> bool {
    flag_enabled_from(std::env::var(name).ok().as_deref())
}

/// An opt-in lever: unset means off, and only an explicit
/// `1`/`true`/`on`/`yes` turns it on. Typos stay off.
fn env_flag_opt_in(name: &str) -> bool {
    flag_opt_in_from(std::env::var(name).ok().as_deref())
}

fn flag_enabled_from(value: Option<&str>) -> bool {
    match value {
        Some(value) => {
            let value = value.trim();
            !(value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("off")
                || value.eq_ignore_ascii_case("no")
                || value == "0")
        }
        None => true,
    }
}

fn flag_opt_in_from(value: Option<&str>) -> bool {
    match value {
        Some(value) => {
            let value = value.trim();
            value == "1"
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("on")
                || value.eq_ignore_ascii_case("yes")
        }
        None => false,
    }
}

/// Post-ingest hooks that fire only when the new memory reads as a failure
/// (crash, regression, outage, the same detector `backfill` uses).
///
/// 1. Retroactive Salience Backfill runs LIVE through the same pipeline as the
///    `backfill` tool, receipts included. Zaki et al. (Nature 2025), the paper
///    the mechanism is ported from, found offline co-reactivation linking is
///    stronger during wake than sleep, so deferring it to the consolidation
///    pass alone was the weaker schedule. Honours `VESTIGE_BACKFILL_AUTOFIRE`.
/// 2. Post-retrieval failure feedback: the memories retrieved in the last
///    thirty minutes of receipts lose accessibility by their reactivation
///    strength (Heinbockel et al., eLife 2025). Opt-in through
///    `VESTIGE_FAILURE_FEEDBACK=1` for now: it demotes memories on its own,
///    so it becomes the default only once real-store data shows the
///    demotions land on the right ones. Every delta is ledgered and
///    reversible either way.
///
/// Neither hook can fail the ingest; problems are logged and the memory is
/// already saved. Returns the hook results for the response, or None when the
/// memory is not a failure.
async fn run_failure_hooks(
    storage: &Arc<Storage>,
    node_id: &str,
    content: &str,
    tags: &[String],
    scope: &str,
) -> Option<Value> {
    if !vestige_core::advanced::retroactive_backfill::looks_like_failure(content, tags) {
        return None;
    }
    let mut hooks = serde_json::Map::new();
    if env_flag_opt_in("VESTIGE_FAILURE_FEEDBACK") {
        match storage.apply_failure_feedback(node_id, chrono::Duration::minutes(30)) {
            Ok(report) => {
                hooks.insert(
                    "failureFeedback".to_string(),
                    serde_json::to_value(report).unwrap_or(Value::Null),
                );
            }
            Err(error) => {
                tracing::warn!(%error, node_id, "post-retrieval failure feedback skipped");
            }
        }
    }
    if env_flag_enabled("VESTIGE_BACKFILL_AUTOFIRE") {
        match super::backfill::execute(
            storage,
            Some(serde_json::json!({ "failure_id": node_id, "scope": scope, "promote": false })),
        )
        .await
        {
            Ok(result) => {
                let promoted = result
                    .get("causes")
                    .and_then(Value::as_array)
                    .map(|causes| {
                        causes
                            .iter()
                            .filter(|c| c.get("promoted") == Some(&Value::Bool(true)))
                            .count()
                    })
                    .unwrap_or(0);
                hooks.insert(
                    "backfill".to_string(),
                    serde_json::json!({
                        "triggered": result.get("triggered").cloned().unwrap_or(Value::Bool(false)),
                        "causesPromoted": promoted,
                        "candidatesFound": result.get("causes").and_then(Value::as_array).map_or(0, Vec::len),
                        "preview": true,
                        "scope": scope,
                        "evidenceStatus": "hypothesis",
                        "causalityVerified": false,
                    }),
                );
            }
            Err(error) => {
                tracing::warn!(%error, node_id, "live backfill skipped");
            }
        }
    }
    if hooks.is_empty() {
        None
    } else {
        Some(Value::Object(hooks))
    }
}

fn attach_failure_hooks(response: &mut Value, hooks: Option<Value>) {
    if let (Some(object), Some(hooks)) = (response.as_object_mut(), hooks) {
        object.insert("failureHooks".to_string(), hooks);
    }
}

/// The settings one `items` batch shares across all of its items.
struct BatchSettings<'a> {
    force_create: bool,
    merge_policy: &'a str,
    default_scope: &'a str,
}

/// Execute batch mode: process up to 20 items, each with full cognitive pipeline.
///
/// Unlike the old `session_checkpoint` tool, batch mode runs the full cognitive
/// pre-ingest (importance scoring, intent detection) and post-ingest (synaptic
/// tagging, novelty update, hippocampal indexing) pipelines per item.
async fn execute_batch(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    items: Vec<BatchItem>,
    item_intents: &[Option<String>],
    settings: BatchSettings<'_>,
    claimed_role: Option<&str>,
) -> Result<Value, String> {
    let BatchSettings {
        force_create: global_force_create,
        merge_policy: batch_merge_policy,
        default_scope,
    } = settings;
    if items.is_empty() {
        return Err("Items array cannot be empty".to_string());
    }
    if items.len() > 20 {
        return Err("Maximum 20 items per batch".to_string());
    }

    // ====================================================================
    // INGEST-V5 B3.1/B3.2 — deterministic processing order. Canonical hash
    // per item, then indices sorted by (canonical_hash_hex, original index).
    // Canonically identical items collapse onto the lowest original index,
    // and intra-batch duplicates dedup against the earlier-sorted identical
    // item because it is indexed first. Items with invalid or missing
    // fields keep their current per-item handling; hashing them is free and
    // keeps the order total. Results are re-sorted into caller order before
    // returning, so the response is identical whatever order the caller
    // used — no arrival-order dependence anywhere.
    // ====================================================================
    let canonical_hashes: Vec<String> = items
        .iter()
        .map(|item| strata_store::canonical::canonical_hash_hex(&item.content))
        .collect();
    // Imported history is indexed oldest first (salience-pro 0.3.7 import):
    // an item with a valid validFrom goes before an undated one, earlier
    // moments first, so a later fact meets the older ones already on the log.
    // The key is the item's own data, so the order stays arrival-independent.
    let moments: Vec<Option<DateTime<Utc>>> = items
        .iter()
        .map(|item| {
            parse_validity_timestamp(item.valid_from.as_deref(), "validFrom")
                .ok()
                .flatten()
        })
        .collect();
    let mut processing_order: Vec<usize> = (0..items.len()).collect();
    processing_order.sort_by(|&left, &right| {
        moments[left]
            .is_none()
            .cmp(&moments[right].is_none())
            .then(moments[left].cmp(&moments[right]))
            .then(canonical_hashes[left].cmp(&canonical_hashes[right]))
            .then(left.cmp(&right))
    });
    let mut items: Vec<Option<BatchItem>> = items.into_iter().map(Some).collect();

    let mut results = Vec::new();
    let mut created = 0u32;
    let updated = 0u32;
    let mut skipped = 0u32;
    let mut errors = 0u32;
    let mut replayed = 0u32;
    let mut reinforced = 0u32;
    let mut batch_created_node_ids: Vec<String> = Vec::new();

    for &i in &processing_order {
        let Some(item) = items[i].take() else {
            continue;
        };
        let intent_id = item_intents.get(i).cloned().flatten();
        let scope = item
            .scope
            .clone()
            .unwrap_or_else(|| default_scope.to_string());
        if let Some(refusal) = scope_credential_refusal(&scope) {
            errors += 1;
            results.push(serde_json::json!({
                "index": i,
                "status": "rejected",
                "reason": refusal
            }));
            continue;
        }
        // ====================================================================
        // INGEST-V5 B3.3 — per-item intent replay, same flow as single mode:
        // zero writes, first outcome returned.
        // ====================================================================
        if let Some(intent_id) = intent_id.as_deref() {
            match storage.find_intent_record(&scope, intent_id) {
                Ok(Some((replay_of, effect_seq, digest))) => {
                    replayed += 1;
                    let mut result = intent_replay_response(
                        &scope,
                        intent_id,
                        &replay_of,
                        effect_seq,
                        &digest,
                        &item.content,
                    );
                    result["index"] = serde_json::json!(i);
                    result["status"] = serde_json::json!("replayed");
                    results.push(result);
                    continue;
                }
                Ok(None) => {}
                Err(reason) => {
                    errors += 1;
                    results.push(serde_json::json!({
                        "index": i,
                        "status": "error",
                        "reason": reason.to_string()
                    }));
                    continue;
                }
            }
        }
        let mut validity = match resolve_validity_range(
            &item.content,
            item.valid_from.as_deref(),
            item.valid_until.as_deref(),
        ) {
            Ok(range) => range,
            Err(reason) => {
                errors += 1;
                results.push(serde_json::json!({
                    "index": i,
                    "status": "error",
                    "reason": reason
                }));
                continue;
            }
        };
        apply_state_ttl(&mut validity, item.node_type.as_deref(), Utc::now());
        // Skip empty content
        if item.content.trim().is_empty() {
            results.push(serde_json::json!({
                "index": i,
                "status": "skipped",
                "reason": "Empty content"
            }));
            skipped += 1;
            continue;
        }

        // Skip content > 1MB
        if item.content.len() > 1_000_000 {
            results.push(serde_json::json!({
                "index": i,
                "status": "skipped",
                "reason": "Content too large (max 1MB)"
            }));
            skipped += 1;
            continue;
        }

        // Extract per-item force_create before consuming other fields
        let item_force_create = item.force_create.unwrap_or(false);
        let secret_policy = if item.allow_secrets.unwrap_or(false) {
            SecretPolicy::AllowExplicitly
        } else {
            SecretPolicy::Reject
        };

        // ================================================================
        // COGNITIVE PRE-INGEST (per item)
        // ================================================================
        let mut importance_composite = 0.0_f64;
        let mut importance_snapshot = SynapticSignalSnapshot {
            novelty: 0.0,
            arousal: 0.0,
            reward: 0.0,
            attention: 0.0,
            composite: 0.0,
        };
        let requested_tags = item.tags.clone().unwrap_or_default();
        let tag_suggestions = similar_tag_suggestions(storage, &scope, &requested_tags);
        if item.preview_tag_suggestions.unwrap_or(false) {
            skipped += 1;
            results.push(serde_json::json!({
                "index": i,
                "status": "previewed",
                "decision": "preflight",
                "wouldWrite": false,
                "scope": scope,
                "validity": validity_response(&validity),
                "tagSuggestions": tag_suggestions.suggestions,
                "tagSuggestionStatus": tag_suggestions.status,
            }));
            continue;
        }
        let accepted_tag_suggestions = item.accepted_tag_suggestions.clone().unwrap_or_default();
        let mut tags = match apply_accepted_tag_suggestions(
            item.tags.unwrap_or_default(),
            &tag_suggestions,
            (!accepted_tag_suggestions.is_empty()).then_some(&accepted_tag_suggestions),
        ) {
            Ok(tags) => tags,
            Err(reason) => {
                errors += 1;
                results.push(serde_json::json!({
                    "index": i,
                    "status": "error",
                    "reason": reason,
                }));
                continue;
            }
        };

        if let Ok(cog) = cognitive.try_lock() {
            let context = ImportanceContext::current();
            let importance = cog
                .importance_signals
                .compute_importance(&item.content, &context);
            importance_composite = importance.composite;
            importance_snapshot = SynapticSignalSnapshot {
                novelty: importance.novelty,
                arousal: importance.arousal,
                reward: importance.reward,
                attention: importance.attention,
                composite: importance.composite,
            };

            let intent_result = cog.intent_detector.detect_intent();
            if intent_result.confidence > 0.5 {
                let intent_tag = format!("intent:{:?}", intent_result.primary_intent);
                let intent_tag = if intent_tag.len() > 50 {
                    format!("{}...", &intent_tag[..intent_tag.floor_char_boundary(47)])
                } else {
                    intent_tag
                };
                tags.push(intent_tag);
            }
        }

        // INGEST-V5: submitted source (intent digest input) captured before
        // `input` consumes it.
        let submitted_source = item.source.clone().unwrap_or_default();
        let input = IngestInput {
            content: item.content.clone(),
            node_type: item.node_type.unwrap_or_else(|| "fact".to_string()),
            source: item.source,
            sentiment_score: 0.0,
            sentiment_magnitude: importance_composite,
            tags,
            valid_from: validity.range.from,
            valid_until: validity.range.until,
            validity_inferred: validity.inferred_phrase.is_some(),
            source_envelope: None,
        };
        // INGEST-V5: the tags as stored (post-acceptance, post-intent-tag)
        // feed the proof enrichment and the intent digest.
        let hook_tags: Vec<String> = input.tags.clone();

        // ================================================================
        // INGEST (storage lock per item)
        // ================================================================

        // Check force_create: global flag OR per-item flag
        let item_force = global_force_create || item_force_create;
        if item_force {
            match storage.ingest_in_scope_with_secret_policy(input, &scope, secret_policy) {
                Ok(node) => {
                    let node_id = node.id.clone();
                    let node_content = node.content.clone();
                    let node_type = node.node_type.clone();
                    // INGEST-V5: this item's create effect, resolved before
                    // any post-hook can append more.
                    let effect_seq = storage.node_effect_seq(&node_id);

                    created += 1;
                    batch_created_node_ids.push(node_id.clone());
                    let synaptic_capture = run_post_ingest_with_snapshot(
                        storage,
                        cognitive,
                        &node_id,
                        &node_content,
                        &node_type,
                        importance_composite,
                        importance_snapshot.clone(),
                    );

                    let mut result = serde_json::json!({
                        "index": i,
                        "status": "saved",
                        "decision": "create",
                        "nodeId": node_id,
                        "scope": scope,
                        "synapticCapture": synaptic_capture,
                        "validity": validity_response(&validity),
                        "tagSuggestions": tag_suggestions.suggestions,
                        "tagSuggestionStatus": tag_suggestions.status,
                        "acceptedTagSuggestions": accepted_tag_suggestions,
                    });
                    // INGEST-V5 (B2.4/B2.5): forceCreate skips only the
                    // canonical gate; proof fields and the intent entry
                    // still apply.
                    enrich_create_response(
                        storage,
                        &mut result,
                        &node_id,
                        &item.content,
                        &hook_tags,
                    );
                    attach_intent_record(
                        storage,
                        &mut result,
                        &scope,
                        IntentRecord {
                            intent_id: intent_id.as_deref(),
                            node_id: &node_id,
                            effect_seq,
                            submitted_content: &item.content,
                            submitted_source: &submitted_source,
                            stored_tags: &hook_tags,
                        },
                    );
                    results.push(result);
                }
                Err(e) => {
                    errors += 1;
                    results.push(serde_json::json!({
                        "index": i,
                        "status": if matches!(&e, StorageError::SecretDetected { .. }) { "rejected" } else { "error" },
                        "reason": e.to_string()
                    }));
                }
            }
            continue;
        }

        {
            // ================================================================
            // INGEST-V5 B3 — canonical duplicate gate per item (the B2.3
            // flow). Only reached when this item is not force-created.
            // Identical items earlier in the sorted processing order were
            // indexed first, so intra-batch duplicates resolve to the
            // lowest original index of the group.
            // ================================================================
            let duplicate_of = match storage.find_duplicate_by_canonical_hash(&scope, &item.content)
            {
                Ok(found) => found,
                Err(reason) => {
                    errors += 1;
                    results.push(serde_json::json!({
                        "index": i,
                        "status": "error",
                        "reason": reason.to_string()
                    }));
                    continue;
                }
            };
            if let Some(original) = duplicate_of {
                reinforced += 1;
                match reinforce_duplicate(
                    storage,
                    &scope,
                    &canonical_hashes[i],
                    &item.content,
                    intent_id.as_deref(),
                    &original,
                    &validity,
                    &tag_suggestions,
                    &accepted_tag_suggestions,
                    &submitted_source,
                    &hook_tags,
                ) {
                    Ok(mut result) => {
                        result["index"] = serde_json::json!(i);
                        result["status"] = serde_json::json!("saved");
                        results.push(result);
                    }
                    Err(reason) => {
                        errors += 1;
                        results.push(serde_json::json!({
                            "index": i,
                            "status": "error",
                            "reason": reason
                        }));
                    }
                }
                continue;
            }
            match storage.ingest_in_scope_with_secret_policy(input, &scope, secret_policy) {
                Ok(node) => {
                    let node_id = node.id.clone();
                    let node_content = node.content.clone();
                    let node_type = node.node_type.clone();
                    // INGEST-V5: this item's create effect, resolved before
                    // any post-hook can append more.
                    let effect_seq = storage.node_effect_seq(&node_id);

                    created += 1;
                    batch_created_node_ids.push(node_id.clone());
                    let synaptic_capture = run_post_ingest_with_snapshot(
                        storage,
                        cognitive,
                        &node_id,
                        &node_content,
                        &node_type,
                        importance_composite,
                        importance_snapshot.clone(),
                    );

                    let mut result = serde_json::json!({
                        "index": i,
                        "status": "saved",
                        "decision": "create",
                        "nodeId": node_id,
                        "scope": scope,
                        "synapticCapture": synaptic_capture,
                        "validity": validity_response(&validity),
                        "tagSuggestions": tag_suggestions.suggestions,
                        "tagSuggestionStatus": tag_suggestions.status,
                        "acceptedTagSuggestions": accepted_tag_suggestions,
                    });
                    // INGEST-V5 (B2.4): proof fields for the create.
                    enrich_create_response(
                        storage,
                        &mut result,
                        &node_id,
                        &item.content,
                        &hook_tags,
                    );
                    attach_intent_record(
                        storage,
                        &mut result,
                        &scope,
                        IntentRecord {
                            intent_id: intent_id.as_deref(),
                            node_id: &node_id,
                            effect_seq,
                            submitted_content: &item.content,
                            submitted_source: &submitted_source,
                            stored_tags: &hook_tags,
                        },
                    );
                    results.push(result);
                }
                Err(e) => {
                    errors += 1;
                    results.push(serde_json::json!({
                        "index": i,
                        "status": if matches!(&e, StorageError::SecretDetected { .. }) { "rejected" } else { "error" },
                        "reason": e.to_string()
                    }));
                }
            }
        }
    }

    // INGEST-V5 B3.2 — results return in the caller's order. Indexes are
    // unique per item, so the sort is total and stable whatever order the
    // items were processed in.
    results.sort_by_key(|result| result["index"].as_u64().unwrap_or(0));

    let mut envelope = serde_json::json!({
        "success": errors == 0,
        "mode": "batch",
        "atomic": false,
        "batchOutcome": if errors > 0 {
            if created + updated + reinforced > 0 { "partial" } else { "failed" }
        } else if created + updated + reinforced == 0 { "no_changes" } else { "applied" },
        "batchMergePolicy": batch_merge_policy,
        "summary": {
            "total": results.len(),
            "created": created,
            "updated": updated,
            "replayed": replayed,
            "reinforced": reinforced,
            "skipped": skipped,
            "errors": errors
        },
        "results": results
    });
    if claimed_role.is_some() {
        crate::actor_surface::attach_actor_block(&mut envelope, storage, claimed_role);
    }
    Ok(envelope)
}

#[allow(clippy::too_many_arguments)]
fn run_post_ingest_with_snapshot(
    _storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    node_id: &str,
    content: &str,
    _node_type: &str,
    _importance_composite: f64,
    _importance_snapshot: SynapticSignalSnapshot,
) -> Option<Value> {
    if let Ok(mut cog) = cognitive.try_lock() {
        cog.importance_signals.learn_content(content);
        cog.cross_project
            .record_project_memory(node_id, "default", None);
    }
    None
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod echo_bound_tests {
    #[test]
    fn echo_tags_are_capped_and_independent_of_input_order() {
        let mut forward: Vec<String> = (0..100).map(|i| format!("tag-{i:03}")).collect();
        forward.push("  duplicate  ".into());
        forward.push("   ".into());
        let mut backward = forward.clone();
        backward.reverse();
        let tags = super::bounded_echo_tags(&forward);
        assert_eq!(tags, super::bounded_echo_tags(&backward));
        assert_eq!(tags.len(), super::MAX_ECHO_TAGS);
        assert_eq!(
            tags.iter()
                .filter(|tag| tag.as_str() == "duplicate")
                .count(),
            1
        );
        assert!(tags.iter().any(|tag| tag == "tag-000"));
        assert!(tags.iter().any(|tag| tag == "tag-014"));
        assert!(!tags.iter().any(|tag| tag == "tag-015"));
    }
}

/// The duplicate gate on the default Strata opener. `legacy-sqlite` tests
/// open a different store that does not keep the in-process echo index.
#[cfg(test)]
mod echo_reuse_tests {
    use std::sync::Arc;

    use tokio::sync::Mutex;

    use crate::cognitive::CognitiveEngine;

    fn test_cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    async fn test_storage() -> (Arc<vestige_core::Storage>, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        (storage, dir)
    }

    #[tokio::test]
    async fn identical_reinforces_reuse_one_echo_and_distinct_tags_stop_at_the_cap() {
        let (storage, _dir) = test_storage().await;
        let created = super::execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "content": "the cache ttl is 60 seconds",
                "tags": ["cache"]
            })),
        )
        .await
        .unwrap();
        let original = created["nodeId"].as_str().unwrap().to_string();
        let mut echo = String::new();
        for _ in 0..20 {
            let again = super::execute(
                &storage,
                &test_cognitive(),
                Some(serde_json::json!({
                    "content": "  the   cache ttl is 60 seconds  "
                })),
            )
            .await
            .unwrap();
            assert_eq!(again["decision"], "reinforce", "{again}");
            assert_eq!(again["duplicateOf"], original);
            let id = again["echoNodeId"].as_str().unwrap().to_string();
            if echo.is_empty() {
                echo = id.clone();
            }
            assert_eq!(id, echo, "the same echo content and tags reuse one node");
        }
        let live = storage
            .get_all_nodes_in_scope("user", i32::MAX, 0)
            .unwrap()
            .len();
        let other_case = super::execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({ "content": "THE CACHE TTL IS 60 SECONDS" })),
        )
        .await
        .unwrap();
        assert_eq!(
            other_case["decision"], "create",
            "case is a different memory: {other_case}"
        );
        assert_eq!(live, 2, "one original and one echo");
        let echo_node = storage.get_node(&echo).unwrap().unwrap();
        assert!(echo_node.tags.iter().any(|tag| tag == "duplicate"));
        assert!(echo_node.tags.len() <= super::MAX_ECHO_TAGS);

        let capped = super::execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({ "content": "cap the echo carriers" })),
        )
        .await
        .unwrap();
        let capped_original = capped["nodeId"].as_str().unwrap().to_string();
        let mut earliest = String::new();
        let mut last = String::new();
        for n in 0..17 {
            let again = super::execute(
                &storage,
                &test_cognitive(),
                Some(serde_json::json!({
                    "content": "cap the echo carriers",
                    "tags": [format!("only-{n:02}")]
                })),
            )
            .await
            .unwrap();
            assert_eq!(again["decision"], "reinforce");
            assert_eq!(again["duplicateOf"], capped_original);
            let id = again["echoNodeId"].as_str().unwrap().to_string();
            if n == 0 {
                earliest = id.clone();
            }
            last = id;
        }
        assert_eq!(last, earliest, "past 16 live echoes the earliest is reused");
        let echoes = storage
            .get_all_nodes_in_scope("user", i32::MAX, 0)
            .unwrap()
            .into_iter()
            .filter(|node| node.tags.iter().any(|tag| tag == "duplicate"))
            .count();
        assert_eq!(
            echoes,
            16 + 1,
            "the first fixture's echo plus 16 capped echoes"
        );
    }
}

/// The `autoConnect` block of a save names at most `AUTO_CONNECT_PAIRS_SHOWN`
/// pairs and counts the rest; every edge is still written. Before 4.2.1 one
/// save listed every pair it joined, up to `MAX_AUTO_EDGES` (100).
#[cfg(test)]
mod auto_connect_response_tests {
    use std::sync::Arc;

    use tokio::sync::Mutex;

    use crate::cognitive::CognitiveEngine;

    async fn save(
        storage: &Arc<vestige_core::Storage>,
        content: &str,
        tag: &str,
    ) -> serde_json::Value {
        super::execute(
            storage,
            &Arc::new(Mutex::new(CognitiveEngine::new())),
            // The pair list is part of the standard profile; compact is the default.
            Some(serde_json::json!({ "content": content, "node_type": "fact", "tags": [tag], "outputProfile": "standard" })),
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn a_save_that_joins_many_memories_lists_five_pairs_and_writes_every_edge() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        for n in 0..8 {
            save(
                &storage,
                &format!("case {n} edited src/hot.rs"),
                &format!("case-{n}"),
            )
            .await;
        }
        let last = save(&storage, "one more edit to src/hot.rs", "final").await;
        let report = &last["autoConnect"];
        assert_eq!(report["edges"], 8, "{last}");
        assert_eq!(report["pairs"].as_array().unwrap().len(), 5, "{last}");
        assert_eq!(report["pairsTotal"], 8, "{last}");
        assert_eq!(report["pairsShown"], 5, "{last}");
        assert_eq!(
            report["sharedIdentities"],
            serde_json::json!(["path:src/hot.rs"])
        );
        let id = last["nodeId"].as_str().unwrap();
        let edges = storage.get_connections_for_memory(id).unwrap();
        assert_eq!(
            edges.len(),
            8,
            "every edge is written, only the response is cut"
        );
    }

    #[tokio::test]
    async fn a_save_with_few_pairs_lists_them_all() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        save(&storage, "first edit of src/cold.rs", "a").await;
        let second = save(&storage, "second edit of src/cold.rs", "b").await;
        let report = &second["autoConnect"];
        assert_eq!(report["edges"], 1, "{second}");
        assert_eq!(report["pairs"].as_array().unwrap().len(), 1, "{second}");
        assert_eq!(report["pairsTotal"], 1, "{second}");
        assert_eq!(report["pairsShown"], 1, "{second}");
    }

    #[tokio::test]
    async fn compact_output_profile_returns_minimal_proof_receipt() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        let out = super::execute(
            &storage,
            &Arc::new(Mutex::new(CognitiveEngine::new())),
            Some(serde_json::json!({
                "content": "compact profile test on src/compact.rs",
                "node_type": "fact",
                "tags": ["compact-test"],
                "outputProfile": "compact"
            })),
        )
        .await
        .unwrap();
        assert_eq!(out["success"], true);
        assert_eq!(out["decision"], "create");
        assert!(out["nodeId"].is_string());
        assert!(out["receiptId"].is_string());
        assert!(out["canonicalHash"].is_string());
        assert!(out.get("importance").is_none());
        assert!(out.get("tagSuggestionStatus").is_none());
    }
}
