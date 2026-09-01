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

use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;
use unicode_normalization::UnicodeNormalization;

use crate::cognitive::CognitiveEngine;
use vestige_core::{
    ContentType, DEFAULT_MEMORY_SCOPE, ImportanceContext, IngestInput, SecretPolicy, Storage,
    StorageError, SynapticCapturePolicy, SynapticImportanceEvent, SynapticIngestRequest,
    SynapticSignalSnapshot, SynapticTag, SynapticTaggingConfig, scan_secrets,
};

/// Input schema for smart_ingest tool
///
/// Supports two modes:
/// - **Single mode**: provide `content` (required) + optional fields
/// - **Batch mode**: provide `items` array (max 20), each with full cognitive pipeline
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The content to remember. Will be compared against existing memories. (Single mode)"
            },
            "node_type": {
                "type": "string",
                "description": "Type of knowledge: fact, concept, event, person, place, note, pattern, decision, state. 'state' is for current-state snapshots (version numbers, progress, inventories) that rot: unless validUntil is given, it expires VESTIGE_STATE_TTL_DAYS (default 30) after ingest; expired memories are down-ranked to the bottom of recall and marked currentlyValid=false (still retrievable for audit via validAt).",
                "default": "fact"
            },
            "tags": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Tags for categorization. The response non-destructively suggests close existing tags in the same scope; suggestions are never auto-applied."
            },
            "source": {
                "type": "string",
                "description": "Source or reference for this knowledge"
            },
            "scope": {
                "type": "string",
                "description": "Project namespace for this memory. Defaults to 'user' for backward compatibility. Recall searches this namespace unless includeCrossScope=true."
            },
            "validFrom": {
                "type": "string",
                "description": "When this fact becomes true. Use RFC3339 or an exact YYYY-MM-DD date. When omitted, one unambiguous 'as of YYYY-MM-DD' phrase in content is inferred as validFrom and reported in the response."
            },
            "validUntil": {
                "type": "string",
                "description": "When this fact stops being true. Use RFC3339 or an exact YYYY-MM-DD date; must be after validFrom."
            },
            "forceCreate": {
                "type": "boolean",
                "description": "Force creation of a new memory even if similar content exists",
                "default": false
            },
            "allowSecrets": {
                "type": "boolean",
                "description": "Allow a detected credential to be stored for this single item. Dangerous: normally redact the value or store a secret-manager reference instead.",
                "default": false
            },
            "previewTagSuggestions": {
                "type": "boolean",
                "description": "Read-only preflight. Returns same-scope similar-tag suggestions and inferred validity without storing anything.",
                "default": false
            },
            "acceptedTagSuggestions": {
                "type": "object",
                "additionalProperties": { "type": "string" },
                "description": "Explicitly accepted input-tag to existing-tag mappings from a preflight response. Each mapping is revalidated against current same-scope suggestions before ingest."
            },
            "batchMergePolicy": {
                "type": "string",
                "enum": ["force_create", "smart"],
                "description": "Batch mode only. Defaults to 'force_create' so caller-separated items stay separate. Use 'smart' to allow Prediction Error Gating against existing memories.",
                "default": "force_create"
            },
            "items": {
                "type": "array",
                "description": "Batch mode: array of items to save (max 20). Defaults to force-creating each caller-separated item; set batchMergePolicy='smart' to allow Prediction Error Gating against existing memories. Use at session end or before context compaction.",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content to remember"
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Tags for categorization. Similar existing same-scope tags are returned as non-mutating suggestions."
                        },
                        "node_type": {
                            "type": "string",
                            "description": "Type: fact, concept, event, person, place, note, pattern, decision",
                            "default": "fact"
                        },
                        "source": {
                            "type": "string",
                            "description": "Source reference"
                        },
                        "scope": {
                            "type": "string",
                            "description": "Project namespace for this item. Overrides the batch scope when supplied."
                        },
                        "validFrom": {
                            "type": "string",
                            "description": "When this item becomes true (RFC3339 or YYYY-MM-DD). If omitted, one unambiguous 'as of YYYY-MM-DD' phrase is inferred."
                        },
                        "validUntil": {
                            "type": "string",
                            "description": "When this item stops being true (RFC3339 or YYYY-MM-DD; after validFrom)."
                        },
                        "forceCreate": {
                            "type": "boolean",
                            "description": "Force creation of this item even if similar content exists",
                            "default": false
                        },
                        "allowSecrets": {
                            "type": "boolean",
                            "description": "Allow a detected credential for this item only. Defaults to false; do not use for ordinary session summaries.",
                            "default": false
                        },
                        "previewTagSuggestions": {
                            "type": "boolean",
                            "description": "Read-only per-item tag/validity preflight; this item is not stored.",
                            "default": false
                        },
                        "acceptedTagSuggestions": {
                            "type": "object",
                            "additionalProperties": { "type": "string" },
                            "description": "Explicitly accepted tag mappings from a prior preflight."
                        }
                    },
                    "required": ["content"]
                }
            }
        }
    })
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

fn normalized_tag_fingerprint(tag: &str) -> String {
    tag.nfkc()
        .flat_map(char::to_lowercase)
        .filter(|character| character.is_alphanumeric())
        .collect()
}

fn normalized_tag_text(tag: &str) -> String {
    tag.nfkc().flat_map(char::to_lowercase).collect()
}

fn bounded_levenshtein(left: &str, right: &str, maximum: usize) -> Option<usize> {
    let left: Vec<char> = left.chars().collect();
    let right: Vec<char> = right.chars().collect();
    if left.len().abs_diff(right.len()) > maximum {
        return None;
    }

    let outside_band = maximum + 1;
    let mut previous: Vec<usize> = (0..=right.len())
        .map(|index| {
            if index <= maximum {
                index
            } else {
                outside_band
            }
        })
        .collect();
    // Two rows are swapped instead of allocating a fresh row per left char.
    // Each row only rewrites its banded window, so the cells just outside the
    // band are pinned to `outside_band` to keep off-band reads unchanged.
    let mut current = vec![outside_band; right.len() + 1];
    for (left_index, left_char) in left.iter().enumerate() {
        let row = left_index + 1;
        current[0] = if row <= maximum { row } else { outside_band };
        let start = row.saturating_sub(maximum).max(1);
        let end = (row + maximum).min(right.len());
        if start > 1 {
            current[start - 1] = outside_band;
        }
        if end < right.len() {
            current[end + 1] = outside_band;
        }
        for column in start..=end {
            current[column] = std::cmp::min(
                std::cmp::min(current[column - 1] + 1, previous[column] + 1),
                previous[column - 1] + usize::from(*left_char != right[column - 1]),
            );
        }
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[right.len()] <= maximum).then_some(previous[right.len()])
}

#[derive(Debug, Clone)]
struct TagSuggestionReport {
    suggestions: Vec<Value>,
    status: Value,
}

fn similar_tag_suggestions(
    storage: &Storage,
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
    // Overlong STORED tags are skipped and counted by the storage layer
    // (mirroring overlong-input and secret-shaped handling), so one legacy
    // tag cannot disable suggestions for the whole scope. Only the hard
    // 10,000-tag vocabulary bound still surfaces as "unavailable".
    let (vocabulary, ignored_overlong_vocabulary_tags) = match storage.tag_vocabulary(Some(scope)) {
        Ok(vocabulary) => (vocabulary.tags, vocabulary.skipped_overlong),
        Err(error) => {
            return TagSuggestionReport {
                suggestions: Vec::new(),
                status: serde_json::json!({
                    "status": "unavailable",
                    "reason": error.to_string(),
                    "scope": scope,
                }),
            };
        }
    };
    let mut ignored_secret_shaped_vocabulary_tags = 0usize;
    let normalized_vocabulary: Vec<_> = vocabulary
        .iter()
        .filter_map(|existing| {
            if scan_secrets(existing)
                .into_iter()
                .any(|finding| finding.blocks_ingestion())
            {
                ignored_secret_shaped_vocabulary_tags += 1;
                return None;
            }
            Some((
                existing,
                normalized_tag_text(existing),
                normalized_tag_fingerprint(existing),
            ))
        })
        .collect();
    let mut suggestions = Vec::new();
    let mut ignored_overlong = 0usize;
    for input in requested.iter().take(50) {
        if input.chars().count() > 200 {
            ignored_overlong += 1;
            continue;
        }
        if !scan_secrets(input).is_empty() {
            continue;
        }
        let input_lower = normalized_tag_text(input);
        let input_fingerprint = normalized_tag_fingerprint(input);
        if input_fingerprint.chars().count() < 4 {
            continue;
        }
        // The input-side suffix fingerprint is loop-invariant; computing it
        // per vocabulary entry repeats the NFKC normalization up to the full
        // 10k-entry bound on the hot ingest path.
        let input_is_namespaced = input.rsplit_once(':').is_some();
        let input_suffix_fingerprint = normalized_tag_fingerprint(
            input
                .rsplit_once(':')
                .map_or(input.as_str(), |(_, suffix)| suffix),
        );
        let input_suffix_is_comparable = input_suffix_fingerprint.chars().count() >= 4;
        let mut candidates = Vec::new();
        for (existing, existing_lower, existing_fingerprint) in &normalized_vocabulary {
            if existing.as_str() == input.as_str() {
                continue;
            }
            let existing_suffix = existing.rsplit_once(':').map(|(_, suffix)| suffix);
            let exactly_one_namespaced = input_is_namespaced ^ existing_suffix.is_some();
            let existing_suffix = existing_suffix.unwrap_or(existing);
            let namespace_variant = exactly_one_namespaced
                && input_suffix_is_comparable
                && input_suffix_fingerprint == normalized_tag_fingerprint(existing_suffix);
            let (score, reason) = if existing_lower.as_str() == input_lower.as_str() {
                (1.0, "casing_variant")
            } else if existing_fingerprint.as_str() == input_fingerprint.as_str() {
                (1.0, "punctuation_variant")
            } else if namespace_variant {
                (0.95, "namespace_variant")
            } else {
                let longest = input_lower
                    .chars()
                    .count()
                    .max(existing_lower.chars().count());
                let allowed_distance = if longest <= 5 { 1 } else { 2 };
                let Some(distance) =
                    bounded_levenshtein(&input_lower, existing_lower, allowed_distance)
                else {
                    continue;
                };
                let score = 1.0 - distance as f64 / longest.max(1) as f64;
                if score < 0.75 {
                    continue;
                }
                (score, "edit_distance")
            };
            candidates.push((score, *existing, reason));
        }
        candidates.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.1.cmp(right.1))
        });
        for (score, existing, reason) in candidates.into_iter().take(3) {
            suggestions.push(serde_json::json!({
                "inputTag": input,
                "similarExistingTag": existing,
                "similarity": score,
                "reason": reason,
                "appliedAutomatically": false,
            }));
        }
    }
    TagSuggestionReport {
        suggestions,
        status: serde_json::json!({
            "status": "complete",
            "scope": scope,
            "vocabularyScanned": true,
            "vocabularyCount": vocabulary.len(),
            "maximumVocabulary": 10_000,
            "requestedTagsTruncated": requested.len() > 50,
            "ignoredOverlongInputTags": ignored_overlong,
            "ignoredOverlongVocabularyTags": ignored_overlong_vocabulary_tags,
            "ignoredSecretShapedVocabularyTags": ignored_secret_shaped_vocabulary_tags,
            "unicodeNormalization": "NFKC plus Unicode lowercase",
        }),
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

pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: SmartIngestArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };
    let scope = args
        .scope
        .unwrap_or_else(|| DEFAULT_MEMORY_SCOPE.to_string());

    // Detect mode: batch (items present) vs single (content present)
    if let Some(items) = args.items {
        let batch_merge_policy = args
            .batch_merge_policy
            .unwrap_or_else(|| "force_create".to_string());
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
            global_force,
            &batch_merge_policy,
            &scope,
        )
        .await;
    }

    // Single mode: content is required
    let content = args.content.ok_or(
        "Missing 'content' field. Provide 'content' for single mode or 'items' for batch mode.",
    )?;
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
    let input_has_secret_finding = !scan_secrets(&content).is_empty();

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

        // 4D. Adaptive embedding — detect content type for logging
        let _content_type = ContentType::detect(&content);
    }

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

    // Check if force_create is enabled
    if args.force_create.unwrap_or(false) {
        let node = storage
            .ingest_in_scope_with_secret_policy(input, &scope, secret_policy)
            .map_err(|e| e.to_string())?;
        let node_id = node.id.clone();
        let node_content = node.content.clone();
        let node_type = node.node_type.clone();
        let has_embedding = node.has_embedding.unwrap_or(false);

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

        return Ok(serde_json::json!({
            "success": true,
            "decision": "create",
            "nodeId": node_id,
            "scope": scope,
            "message": "Memory created (force_create=true)",
            "hasEmbedding": has_embedding,
            "predictionError": 1.0,
            "importanceScore": importance_composite,
            "synapticCapture": synaptic_capture,
            "reason": "Forced creation - skipped similarity check",
            "validity": validity_response(&validity),
            "tagSuggestions": tag_suggestions.suggestions,
            "tagSuggestionStatus": tag_suggestions.status,
            "acceptedTagSuggestions": accepted_tag_suggestions,
        }));
    }

    // Use smart ingest with prediction error gating
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    {
        let result = storage
            .smart_ingest_in_scope_with_secret_policy(input, &scope, secret_policy)
            .map_err(|e| e.to_string())?;
        let node_id = result.node.id.clone();
        let node_content = result.node.content.clone();
        let node_type = result.node.node_type.clone();
        let has_embedding = result.node.has_embedding.unwrap_or(false);
        let previous_content = if input_has_secret_finding {
            Some("[redacted: credential-bearing ingest]".to_string())
        } else {
            result.previous_content.clone()
        };
        let merge_preview = if input_has_secret_finding {
            Some("[redacted: credential-bearing ingest]".to_string())
        } else {
            result.merge_preview.clone()
        };

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

        Ok(serde_json::json!({
            "success": true,
            "decision": result.decision,
            "nodeId": node_id,
            "scope": scope,
            "message": format!("Smart ingest complete: {}", result.reason),
            "hasEmbedding": has_embedding,
            "similarity": result.similarity,
            "predictionError": result.prediction_error,
            "supersededId": result.superseded_id,
            "previousContent": previous_content,
            "mergedFrom": result.merged_from,
            "mergePreview": merge_preview,
            "autoClosedUntil": result.auto_closed_until.map(|value| value.to_rfc3339()),
            "importanceScore": importance_composite,
            "synapticCapture": synaptic_capture,
            "reason": result.reason,
            "validity": validity_response(&validity),
            "tagSuggestions": tag_suggestions.suggestions,
            "tagSuggestionStatus": tag_suggestions.status,
            "acceptedTagSuggestions": accepted_tag_suggestions,
            "explanation": match result.decision.as_str() {
                "create" => "Created new memory - content was different enough from existing memories",
                "update" => "Updated existing memory - content was similar to an existing memory",
                "reinforce" => "Reinforced existing memory - content was nearly identical",
                "supersede" => "Superseded old memory - new content is an improvement/correction",
                "merge" => "Merged with related memories - content connects multiple topics",
                "replace" => "Replaced existing memory content entirely",
                "add_context" => "Added new content as context to existing memory",
                _ => "Memory processed successfully"
            }
        }))
    }

    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    {
        let node = storage
            .ingest_in_scope_with_secret_policy(input, &scope, secret_policy)
            .map_err(|e| e.to_string())?;
        let node_id = node.id.clone();
        let node_content = node.content.clone();
        let node_type = node.node_type.clone();

        let synaptic_capture = run_post_ingest_with_snapshot(
            storage,
            cognitive,
            &node_id,
            &node_content,
            &node_type,
            importance_composite,
            importance_snapshot,
        );

        Ok(serde_json::json!({
            "success": true,
            "decision": "create",
            "nodeId": node_id,
            "scope": scope,
            "message": "Memory created (smart ingest requires embeddings feature)",
            "hasEmbedding": false,
            "predictionError": 1.0,
            "importanceScore": importance_composite,
            "synapticCapture": synaptic_capture,
            "reason": "Embeddings not available - used regular ingest",
            "validity": validity_response(&validity),
            "tagSuggestions": tag_suggestions.suggestions,
            "tagSuggestionStatus": tag_suggestions.status,
            "acceptedTagSuggestions": accepted_tag_suggestions,
        }))
    }
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
    global_force_create: bool,
    batch_merge_policy: &str,
    default_scope: &str,
) -> Result<Value, String> {
    if items.is_empty() {
        return Err("Items array cannot be empty".to_string());
    }
    if items.len() > 20 {
        return Err("Maximum 20 items per batch".to_string());
    }

    let mut results = Vec::new();
    let mut created = 0u32;
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    let mut updated = 0u32;
    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    let updated = 0u32;
    let mut skipped = 0u32;
    let mut errors = 0u32;
    let mut batch_created_node_ids: Vec<String> = Vec::new();

    for (i, item) in items.into_iter().enumerate() {
        let scope = item
            .scope
            .clone()
            .unwrap_or_else(|| default_scope.to_string());
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
        let input_has_secret_finding = !scan_secrets(&item.content).is_empty();

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

            let _content_type = ContentType::detect(&item.content);
        }

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

                    results.push(serde_json::json!({
                        "index": i,
                        "status": "saved",
                        "decision": "create",
                        "nodeId": node_id,
                        "scope": scope,
                        "importanceScore": importance_composite,
                        "synapticCapture": synaptic_capture,
                        "reason": "Forced creation - skipped similarity check",
                        "validity": validity_response(&validity),
                        "tagSuggestions": tag_suggestions.suggestions,
                        "tagSuggestionStatus": tag_suggestions.status,
                        "acceptedTagSuggestions": accepted_tag_suggestions,
                    }));
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

        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        {
            match storage.smart_ingest_excluding_in_scope_with_secret_policy(
                input,
                &scope,
                &batch_created_node_ids,
                secret_policy,
            ) {
                Ok(result) => {
                    let node_id = result.node.id.clone();
                    let node_content = result.node.content.clone();
                    let node_type = result.node.node_type.clone();
                    let previous_content = if input_has_secret_finding {
                        Some("[redacted: credential-bearing ingest]".to_string())
                    } else {
                        result.previous_content.clone()
                    };
                    let merge_preview = if input_has_secret_finding {
                        Some("[redacted: credential-bearing ingest]".to_string())
                    } else {
                        result.merge_preview.clone()
                    };

                    match result.decision.as_str() {
                        "create" | "supersede" | "merge" => {
                            created += 1;
                            batch_created_node_ids.push(node_id.clone());
                        }
                        "update" | "reinforce" | "replace" | "add_context" => updated += 1,
                        _ => created += 1,
                    }

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

                    results.push(serde_json::json!({
                        "index": i,
                        "status": "saved",
                        "decision": result.decision,
                        "nodeId": node_id,
                        "scope": scope,
                        "similarity": result.similarity,
                        "predictionError": result.prediction_error,
                        "supersededId": result.superseded_id,
                        "previousContent": previous_content,
                        "mergedFrom": result.merged_from,
                        "mergePreview": merge_preview,
                        "autoClosedUntil": result.auto_closed_until.map(|value| value.to_rfc3339()),
                        "importanceScore": importance_composite,
                        "synapticCapture": synaptic_capture,
                        "reason": result.reason,
                        "validity": validity_response(&validity),
                        "tagSuggestions": tag_suggestions.suggestions,
                        "tagSuggestionStatus": tag_suggestions.status,
                        "acceptedTagSuggestions": accepted_tag_suggestions,
                    }));
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

        #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
        {
            match storage.ingest_in_scope_with_secret_policy(input, &scope, secret_policy) {
                Ok(node) => {
                    let node_id = node.id.clone();
                    let node_content = node.content.clone();
                    let node_type = node.node_type.clone();

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

                    results.push(serde_json::json!({
                        "index": i,
                        "status": "saved",
                        "decision": "create",
                        "nodeId": node_id,
                        "scope": scope,
                        "importanceScore": importance_composite,
                        "synapticCapture": synaptic_capture,
                        "reason": "Embeddings not available - used regular ingest",
                        "validity": validity_response(&validity),
                        "tagSuggestions": tag_suggestions.suggestions,
                        "tagSuggestionStatus": tag_suggestions.status,
                        "acceptedTagSuggestions": accepted_tag_suggestions,
                    }));
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

    Ok(serde_json::json!({
        "success": errors == 0,
        "mode": "batch",
        "batchMergePolicy": batch_merge_policy,
        "summary": {
            "total": results.len(),
            "created": created,
            "updated": updated,
            "skipped": skipped,
            "errors": errors
        },
        "results": results
    }))
}

/// Cognitive post-ingest side effects: synaptic tagging, novelty update, hippocampal indexing.
///
/// A high-salience ingest can capture nearby *previously tagged* memories. Each
/// actual capture is promoted and persisted as a normal, fetchable receipt. The
/// receipt contains ids and measured strength changes only — never copied memory
/// content — so purge remains authoritative and suppressed memories stay out.
///
/// Uses try_lock() for non-blocking access. If cognitive is locked, side effects are skipped.
#[cfg(test)]
fn run_post_ingest(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    node_id: &str,
    content: &str,
    node_type: &str,
    importance_composite: f64,
) -> Option<Value> {
    let result = run_post_ingest_with_snapshot(
        storage,
        cognitive,
        node_id,
        content,
        node_type,
        importance_composite,
        SynapticSignalSnapshot {
            novelty: importance_composite,
            arousal: 0.0,
            reward: 0.0,
            attention: 0.0,
            composite: importance_composite,
        },
    );
    // Preserve the pre-V22 private helper contract used by focused tests:
    // ordinary ingests persist a tag but do not claim a capture result.
    if importance_composite <= 0.7 {
        None
    } else {
        result
    }
}

#[allow(clippy::too_many_arguments)]
fn run_post_ingest_with_snapshot(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    node_id: &str,
    content: &str,
    node_type: &str,
    importance_composite: f64,
    importance_snapshot: SynapticSignalSnapshot,
) -> Option<Value> {
    let config = cognitive
        .try_lock()
        .map(|cog| cog.synaptic_tagging.config().clone())
        .unwrap_or_else(|_| SynapticTaggingConfig::default());
    let node = storage.get_node(node_id).ok().flatten();
    let trigger_is_eligible = node.as_ref().is_some_and(|node| {
        node.suppression_count == 0 && node.valid_until.is_none_or(|until| until > Utc::now())
    });
    let mut synaptic_capture = None;
    let mut persisted_tag = None;
    let mut tag_persisted = false;
    let mut tag_active_after_commit = false;

    // 4C. Durable synaptic tagging for retroactive capture. The SQLite
    // transaction is authoritative; CognitiveEngine is only a live projection.
    // Evaluate the trigger before recording its own tag so self-capture is
    // impossible even when smart ingest updates an existing node id.
    if importance_composite > 0.3
        && trigger_is_eligible
        && let Some(node) = &node
    {
        let tag = SynapticTag {
            memory_id: node_id.to_string(),
            created_at: node.updated_at,
            tag_strength: 1.0,
            initial_strength: 1.0,
            captured: false,
            capture_event: None,
            captured_at: None,
            encoding_context: Some(node.node_type.clone()),
        };
        let (event_type, radius) = dominant_importance_event(&importance_snapshot);
        let policy = SynapticCapturePolicy {
            backward_hours: config.capture_window.backward_hours * radius,
            forward_hours: config.capture_window.forward_hours * radius,
            tag_lifetime_hours: config.tag_lifetime_hours,
            minimum_tag_strength: config.min_tag_strength,
            maximum_captures: config.max_cluster_size,
            decay_function: config.capture_window.decay_function,
        };
        let event = (importance_composite > 0.7).then(|| SynapticImportanceEvent {
            event_type: event_type.into(),
            occurred_at: node.updated_at,
            strength: importance_composite,
            policy,
            signal_snapshot: importance_snapshot,
        });
        match storage.process_synaptic_ingest(&SynapticIngestRequest {
            memory_id: node_id.to_string(),
            tag: Some(tag.clone()),
            event,
        }) {
            Ok(outcome) => {
                persisted_tag = Some(tag);
                tag_persisted = outcome.tag_persisted;
                tag_active_after_commit = outcome.tag_active;
                let forward_receipts: Vec<Value> = outcome
                    .forward_receipts
                    .into_iter()
                    .map(|pair| {
                        serde_json::json!({
                            "receiptId": pair.receipt.receipt_id,
                            "receipt": pair.receipt,
                            "eventId": pair.event_id,
                            "disposition": pair.disposition,
                            "reusedExisting": pair.reused_existing
                        })
                    })
                    .collect();
                let mut capture = serde_json::Map::from_iter([
                    ("durable".into(), Value::Bool(true)),
                    ("tagPersisted".into(), Value::Bool(outcome.tag_persisted)),
                    ("tagActive".into(), Value::Bool(outcome.tag_active)),
                    (
                        "algorithmVersion".into(),
                        Value::String("vestige.synaptic_capture.v2".into()),
                    ),
                    ("forwardReceipts".into(), Value::Array(forward_receipts)),
                ]);
                if let Some(root) = outcome.event {
                    capture.insert(
                        "receiptId".into(),
                        Value::String(root.receipt.receipt_id.clone()),
                    );
                    capture.insert(
                        "receipt".into(),
                        serde_json::to_value(root.receipt).unwrap_or(Value::Null),
                    );
                    capture.insert("eventId".into(), Value::String(root.event_id));
                    capture.insert(
                        "capturedCount".into(),
                        serde_json::json!(root.captured_count),
                    );
                    capture.insert("reusedExisting".into(), Value::Bool(root.reused_existing));
                }
                synaptic_capture = Some(Value::Object(capture));
            }
            Err(error) => {
                tracing::warn!(%error, "atomic synaptic ingest transaction failed");
                synaptic_capture = Some(serde_json::json!({
                    "durable": false,
                    "tagPersisted": false,
                    "algorithmVersion": "vestige.synaptic_capture.v2",
                    "error": "Synaptic event/tag transaction was not committed; no capture or restart-safe tag is claimed."
                }));
            }
        }
    }

    if let Ok(mut cog) = cognitive.try_lock() {
        if let Some(capture) = &synaptic_capture {
            if let Some(retrieved) = capture["receipt"]["retrieved"].as_array() {
                for id in retrieved.iter().filter_map(Value::as_str) {
                    cog.synaptic_tagging.remove_tag(id);
                }
            }
            if let Some(forward) = capture["forwardReceipts"].as_array() {
                for pair in forward {
                    if let Some(retrieved) = pair["receipt"]["retrieved"].as_array() {
                        for id in retrieved.iter().filter_map(Value::as_str) {
                            cog.synaptic_tagging.remove_tag(id);
                        }
                    }
                }
            }
        }
        if tag_persisted && let Some(tag) = persisted_tag {
            // The SQLite result is authoritative. A new tag may have been
            // atomically captured by an event that was already open, so never
            // restore merely because its row was persisted.
            if tag_active_after_commit {
                cog.synaptic_tagging.restore_tag(tag);
            } else {
                cog.synaptic_tagging.remove_tag(&tag.memory_id);
            }
        }

        // 4E. Update novelty model with new content
        cog.importance_signals.learn_content(content);

        // 4F. Record in hippocampal index
        let _ = cog.hippocampal_index.index_memory(
            node_id,
            content,
            node_type,
            Utc::now(),
            None, // semantic_embedding — generated separately
        );

        // 4G. Cross-project pattern recording
        cog.cross_project
            .record_project_memory(node_id, "default", None);
    }

    synaptic_capture
}

fn dominant_importance_event(snapshot: &SynapticSignalSnapshot) -> (&'static str, f64) {
    let channels = [
        ("novelty_spike", snapshot.novelty, 0.7),
        ("emotional_content", snapshot.arousal, 1.5),
        ("reward_signal", snapshot.reward, 1.0),
        ("attention_spike", snapshot.attention, 1.0),
    ];
    channels
        .into_iter()
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(kind, _, radius)| (kind, radius))
        .unwrap_or(("novelty_spike", 0.7))
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::CognitiveEngine;
    use tempfile::TempDir;

    fn test_cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    /// Create a test storage instance with a temporary database
    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    #[tokio::test]
    async fn test_smart_ingest_empty_content_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "content": "" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_smart_ingest_basic_content_succeeds() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "This is a test fact to remember."
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert!(value["nodeId"].is_string());
        assert!(value["decision"].is_string());
    }

    #[tokio::test]
    async fn test_smart_ingest_force_create() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "Force create test content.",
            "forceCreate": true
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert_eq!(value["decision"], "create");
        assert!(
            value["reason"].as_str().unwrap().contains("Forced")
                || value["reason"]
                    .as_str()
                    .unwrap()
                    .contains("Embeddings not available")
        );
    }

    #[tokio::test]
    async fn smart_ingest_persists_the_requested_project_scope() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "content": "The web-app backup worker keeps six daily snapshots.",
                "scope": "web-app",
                "forceCreate": true,
            })),
        )
        .await
        .expect("scoped ingest succeeds");

        let id = result["nodeId"].as_str().expect("node id");
        assert_eq!(result["scope"], "web-app");
        assert!(storage.node_is_in_scope(id, "web-app").unwrap());
        assert!(!storage.node_is_in_scope(id, DEFAULT_MEMORY_SCOPE).unwrap());
    }

    #[tokio::test]
    async fn high_salience_ingest_promotes_earlier_tag_and_saves_capture_receipt() {
        let (storage, _dir) = test_storage().await;
        let cognitive = test_cognitive();
        let earlier = storage
            .ingest(IngestInput {
                content: "We selected the retry policy for the deploy worker.".to_string(),
                node_type: "decision".to_string(),
                ..Default::default()
            })
            .unwrap();
        assert!(
            run_post_ingest(
                &storage,
                &cognitive,
                &earlier.id,
                &earlier.content,
                &earlier.node_type,
                0.5,
            )
            .is_none(),
            "an ordinary decision should be tagged but not self-promoted"
        );
        let before = storage.demote_memory(&earlier.id).unwrap();

        let trigger = storage
            .ingest(IngestInput {
                content: "Production deployment failed after the retry policy exhausted."
                    .to_string(),
                node_type: "event".to_string(),
                ..Default::default()
            })
            .unwrap();
        let capture = run_post_ingest(
            &storage,
            &cognitive,
            &trigger.id,
            &trigger.content,
            &trigger.node_type,
            0.9,
        )
        .expect("high-salience trigger captures the earlier tagged decision");

        let receipt_id = capture["receiptId"].as_str().expect("receipt id");
        let saved = storage
            .get_receipt(receipt_id)
            .unwrap()
            .expect("persisted receipt");
        assert_eq!(saved.retrieved, vec![earlier.id.clone()]);
        assert_eq!(saved.mutations[0].kind, "synaptic_capture");
        assert!(
            saved.activation_path[0].contains(&trigger.id),
            "receipt records the trigger reference"
        );
        let after = storage.get_node(&earlier.id).unwrap().unwrap();
        assert!(after.retrieval_strength > before.retrieval_strength);
        assert_eq!(capture["receipt"]["retrieved"][0], earlier.id);
        assert_eq!(
            capture["receipt"]["evidence"]["predicate"]["trigger"]["memoryId"],
            trigger.id
        );
        assert_eq!(
            capture["receipt"]["evidence"]["predicate"]["claimBoundary"],
            "Evidence-backed temporal association with a measured memory-state change; not proof that the trigger caused the earlier memory or a downstream outcome."
        );
        assert!(
            saved.evidence.is_some(),
            "the complete typed capture predicate must be fetchable"
        );
    }

    #[tokio::test]
    async fn suppressed_tag_is_not_promoted_or_exposed_in_a_capture_receipt() {
        let (storage, _dir) = test_storage().await;
        let cognitive = test_cognitive();
        let earlier = storage
            .ingest(IngestInput {
                content: "An ordinary approval was recorded before the incident.".to_string(),
                node_type: "decision".to_string(),
                ..Default::default()
            })
            .unwrap();
        run_post_ingest(
            &storage,
            &cognitive,
            &earlier.id,
            &earlier.content,
            &earlier.node_type,
            0.5,
        );
        let suppressed = storage.suppress_memory(&earlier.id).unwrap();

        let trigger = storage
            .ingest(IngestInput {
                content: "A high-salience incident arrived after the approval.".to_string(),
                node_type: "event".to_string(),
                ..Default::default()
            })
            .unwrap();
        let receipt = run_post_ingest(
            &storage,
            &cognitive,
            &trigger.id,
            &trigger.content,
            &trigger.node_type,
            0.9,
        )
        .expect("a suppressed-only decision remains auditable");
        assert_eq!(receipt["capturedCount"], 0);
        assert!(
            !serde_json::to_string(&receipt)
                .unwrap()
                .contains(&earlier.id),
            "suppressed evidence uses a receipt-local opaque slot"
        );
        assert_eq!(
            receipt["receipt"]["evidence"]["predicate"]["candidates"][0]["disposition"],
            "withheld_suppressed"
        );
        let after = storage.get_node(&earlier.id).unwrap().unwrap();
        assert_eq!(after.suppression_count, suppressed.suppression_count);
        assert_eq!(after.retrieval_strength, suppressed.retrieval_strength);
    }

    #[tokio::test]
    async fn restart_between_tag_and_trigger_preserves_capture_eligibility() {
        let (storage, _dir) = test_storage().await;
        let first_process = test_cognitive();
        let earlier = storage
            .ingest(IngestInput {
                content: "A restart-safe decision preceded the incident.".to_string(),
                node_type: "decision".to_string(),
                ..Default::default()
            })
            .unwrap();
        run_post_ingest(
            &storage,
            &first_process,
            &earlier.id,
            &earlier.content,
            &earlier.node_type,
            0.5,
        );
        let before = storage.demote_memory(&earlier.id).unwrap();

        // Simulate a fresh process. Hydration is verified separately; capture
        // itself must use SQLite as its source of truth.
        drop(first_process);
        let second_process = test_cognitive();
        second_process.lock().await.hydrate(&storage);
        let trigger = storage
            .ingest(IngestInput {
                content: "A later high-salience incident arrived after restart.".to_string(),
                node_type: "event".to_string(),
                ..Default::default()
            })
            .unwrap();
        let capture = run_post_ingest(
            &storage,
            &second_process,
            &trigger.id,
            &trigger.content,
            &trigger.node_type,
            0.9,
        )
        .expect("persisted tag captures after restart");

        assert_eq!(capture["receipt"]["retrieved"][0], earlier.id);
        let after = storage.get_node(&earlier.id).unwrap().unwrap();
        assert!(after.retrieval_strength > before.retrieval_strength);
    }

    #[tokio::test]
    async fn live_projection_does_not_restore_tag_consumed_by_open_forward_event() {
        let (storage, _dir) = test_storage().await;
        let cognitive = test_cognitive();
        let trigger = storage
            .ingest(IngestInput {
                content: "Database retry policy caused a production incident.".to_string(),
                node_type: "event".to_string(),
                ..Default::default()
            })
            .unwrap();
        storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: trigger.id,
                tag: None,
                event: Some(SynapticImportanceEvent {
                    event_type: "novelty_spike".into(),
                    occurred_at: Utc::now() - chrono::Duration::seconds(1),
                    strength: 0.95,
                    policy: SynapticCapturePolicy {
                        backward_hours: 9.0,
                        forward_hours: 2.0,
                        tag_lifetime_hours: 12.0,
                        minimum_tag_strength: 0.3,
                        maximum_captures: 50,
                        decay_function: vestige_core::DecayFunction::Exponential,
                    },
                    signal_snapshot: SynapticSignalSnapshot {
                        novelty: 0.95,
                        arousal: 0.1,
                        reward: 0.1,
                        attention: 0.8,
                        composite: 0.95,
                    },
                }),
            })
            .unwrap();

        let candidate = storage
            .ingest(IngestInput {
                content: "Database retry policy decision recorded for deployment.".to_string(),
                node_type: "decision".to_string(),
                ..Default::default()
            })
            .unwrap();
        assert!(
            run_post_ingest(
                &storage,
                &cognitive,
                &candidate.id,
                &candidate.content,
                &candidate.node_type,
                0.5,
            )
            .is_none(),
            "ordinary ingests retain their pre-V22 helper contract"
        );

        assert!(
            !storage
                .load_active_synaptic_tags()
                .unwrap()
                .iter()
                .any(|tag| tag.memory_id == candidate.id),
            "the durable V22 tag was captured by the already-open event"
        );
        assert!(
            !cognitive
                .lock()
                .await
                .synaptic_tagging
                .has_active_tag(&candidate.id),
            "the live projection must use the authoritative post-commit tag state"
        );
    }

    #[tokio::test]
    async fn test_smart_ingest_rejects_secret_even_when_force_created() {
        let (storage, _dir) = test_storage().await;
        let secret = format!("ghp_{}", "A".repeat(36));
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "content": format!("Store this token: {secret}"),
                "forceCreate": true
            })),
        )
        .await;

        let err = result.unwrap_err();
        assert!(err.contains("Refused to store probable credential"));
        assert!(
            !err.contains(&secret),
            "MCP errors must not echo the rejected credential"
        );
        assert_eq!(
            storage.get_stats().unwrap().total_nodes,
            0,
            "forceCreate must not bypass the credential guard"
        );
    }

    #[tokio::test]
    async fn test_explicit_secret_override_does_not_echo_content_in_response() {
        let (storage, _dir) = test_storage().await;
        let secret = format!("ghp_{}", "A".repeat(36));
        let response = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "content": format!("intentional local credential: {secret}"),
                "forceCreate": true,
                "allowSecrets": true
            })),
        )
        .await
        .unwrap();

        assert_eq!(response["success"], true);
        assert!(
            !serde_json::to_string(&response).unwrap().contains(&secret),
            "the override response must not copy the credential into an MCP transcript"
        );
    }

    #[test]
    fn test_schema_has_required_fields() {
        let schema_value = schema();
        assert_eq!(schema_value["type"], "object");
        assert!(schema_value["properties"]["content"].is_object());
        assert!(schema_value["properties"]["forceCreate"].is_object());
        assert!(schema_value["properties"]["batchMergePolicy"].is_object());
        assert!(schema_value["properties"]["items"].is_object());
        assert!(schema_value["properties"]["validFrom"].is_object());
        assert!(schema_value["properties"]["validUntil"].is_object());
        // v1.7: no top-level required — content for single mode, items for batch mode
        assert!(schema_value.get("required").is_none() || schema_value["required"].is_null());
    }

    #[tokio::test]
    async fn smart_ingest_persists_strict_single_item_validity() {
        let (storage, _dir) = test_storage().await;
        let response = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "content": "The deployment policy applies during Q3.",
                "forceCreate": true,
                "validFrom": "2026-07-01",
                "validUntil": "2026-10-01T00:00:00Z"
            })),
        )
        .await
        .unwrap();
        let node = storage
            .get_node(response["nodeId"].as_str().unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(
            node.valid_from.unwrap().to_rfc3339(),
            "2026-07-01T00:00:00+00:00"
        );
        assert_eq!(
            node.valid_until.unwrap().to_rfc3339(),
            "2026-10-01T00:00:00+00:00"
        );
    }

    #[tokio::test]
    async fn smart_ingest_infers_one_strict_as_of_date_and_reports_provenance() {
        let (storage, _dir) = test_storage().await;
        let response = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "content": "Current version is 4.2 as of 2026-03-04.",
                "forceCreate": true
            })),
        )
        .await
        .unwrap();
        let node = storage
            .get_node(response["nodeId"].as_str().unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(
            node.valid_from.unwrap().to_rfc3339(),
            "2026-03-04T00:00:00+00:00"
        );
        assert_eq!(response["validity"]["source"], "inferred_as_of");
        assert_eq!(response["validity"]["inferredPhrase"], "as of 2026-03-04");
    }

    #[test]
    fn as_of_inference_is_case_insensitive_deduplicated_and_boundary_safe() {
        let dates =
            infer_as_of_dates("Inventory AS OF 2026-03-04, repeated as of 2026-03-04 for clarity.");
        assert_eq!(dates.len(), 1, "the same date repeated is unambiguous");
        assert_eq!(dates[0].0.to_rfc3339(), "2026-03-04T00:00:00+00:00");

        assert!(
            infer_as_of_dates("phaseas of 2026-03-04 must not match").is_empty(),
            "an embedded word is not an 'as of' phrase boundary"
        );
        assert!(
            infer_as_of_dates("éas of 2026-03-04 must not match").is_empty(),
            "a preceding Unicode letter is not an 'as of' phrase boundary"
        );
        assert!(
            infer_as_of_dates("invalid as of 2026-02-30").is_empty(),
            "invalid calendar dates are ignored"
        );
        assert_eq!(
            infer_as_of_dates("Résumé — as of 2026-03-04").len(),
            1,
            "a Unicode phrase before the ASCII marker must not break byte boundaries"
        );

        assert!(
            infer_as_of_dates("as of 2026-03-04T09:15:00Z must not match").is_empty(),
            "a trailing timestamp component is not a bare date"
        );
        assert!(
            infer_as_of_dates("as of 2026-03-04abc must not match").is_empty(),
            "a trailing letter means the token is not a date"
        );
        assert!(
            infer_as_of_dates("save_as of 2026-03-04 must not match").is_empty(),
            "a preceding underscore is not an 'as of' phrase boundary"
        );
        assert!(
            infer_as_of_dates("x-as of 2026-03-04 must not match").is_empty(),
            "a preceding hyphen is not an 'as of' phrase boundary"
        );
        assert_eq!(
            infer_as_of_dates("Reported as of 2026-03-04.").len(),
            1,
            "trailing punctuation is a valid right boundary"
        );
        assert_eq!(
            infer_as_of_dates("Inventory (as of 2026-03-04)").len(),
            1,
            "surrounding parentheses are valid boundaries"
        );
        assert!(
            infer_as_of_dates("as of 2026-03-041 must not match").is_empty(),
            "a trailing digit means the token is not a date"
        );
        assert!(
            infer_as_of_dates("as of 2026-03-04-05 must not match").is_empty(),
            "a trailing hyphen means the token is not a date"
        );
    }

    #[test]
    fn state_nodes_expire_by_default_and_only_state_nodes() {
        let now = Utc::now();
        let mut state = resolve_validity_range("version is 2.6.0", None, None).unwrap();
        apply_state_ttl(&mut state, Some("state"), now);
        let until = state
            .range
            .until
            .expect("state node gets a default validUntil");
        assert_eq!(until, now + chrono::Duration::days(DEFAULT_STATE_TTL_DAYS));
        assert_eq!(state.source, "state_default_ttl");

        // Case-insensitive on the type, and an explicit validUntil always wins.
        let mut explicit =
            resolve_validity_range("progress 40%", None, Some("2030-01-01")).unwrap();
        apply_state_ttl(&mut explicit, Some("State"), now);
        assert_eq!(
            explicit.range.until.unwrap(),
            Utc.with_ymd_and_hms(2030, 1, 1, 0, 0, 0).unwrap()
        );
        assert_eq!(explicit.source, "explicit");

        // Every other node type is untouched.
        for kind in [Some("fact"), Some("decision"), None] {
            let mut other = resolve_validity_range("durable lesson", None, None).unwrap();
            apply_state_ttl(&mut other, kind, now);
            assert!(
                other.range.until.is_none(),
                "{kind:?} must not expire by default"
            );
            assert_eq!(other.source, "none");
        }
    }

    #[test]
    fn inferred_as_of_must_not_cross_an_explicit_valid_until() {
        // An incidental prose date at or after the explicit validUntil is not
        // caller intent: the inference abstains instead of failing the ingest.
        let resolution =
            resolve_validity_range("Policy as of 2026-04-01", None, Some("2026-03-01"))
                .expect("a conflicting inferred date abstains instead of erroring");
        assert!(resolution.range.from.is_none());
        assert_eq!(
            resolution.range.until.unwrap().to_rfc3339(),
            "2026-03-01T00:00:00+00:00"
        );
        assert_eq!(
            resolution.source,
            "inferred_as_of_conflicts_with_explicit_validity_ignored"
        );
        assert!(resolution.inferred_phrase.is_none());
        assert_eq!(resolution.ambiguous_phrases, vec!["as of 2026-04-01"]);
    }

    #[tokio::test]
    async fn conflicting_inferred_as_of_abstains_and_the_ingest_still_stores() {
        let (storage, _dir) = test_storage().await;
        let response = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "content": "Retrospective written as of 2026-04-01 about the old policy.",
                "validUntil": "2026-03-01",
                "forceCreate": true
            })),
        )
        .await
        .expect("the conflicting phrase must not hard-fail the whole ingest");
        assert_eq!(
            response["validity"]["source"],
            "inferred_as_of_conflicts_with_explicit_validity_ignored"
        );
        assert_eq!(
            response["validity"]["ambiguousPhrases"][0],
            "as of 2026-04-01"
        );
        assert!(response["validity"]["inferredPhrase"].is_null());
        let node = storage
            .get_node(response["nodeId"].as_str().unwrap())
            .unwrap()
            .unwrap();
        assert!(node.valid_from.is_none(), "the inference was dropped");
        assert_eq!(
            node.valid_until.unwrap().to_rfc3339(),
            "2026-03-01T00:00:00+00:00",
            "the explicit bound is kept"
        );
    }

    #[test]
    fn tag_similarity_uses_nfkc_before_case_and_punctuation_comparison() {
        assert_eq!(
            normalized_tag_fingerprint("Ｐｒｉｘ－Ｓｉｘ"),
            normalized_tag_fingerprint("prix-six")
        );
        assert_eq!(bounded_levenshtein("colour", "colur", 1), Some(1));
        assert_eq!(bounded_levenshtein("unrelated", "banana", 2), None);
    }

    #[test]
    fn bounded_levenshtein_matches_the_naive_distance_at_band_edges() {
        fn naive(left: &str, right: &str) -> usize {
            let left: Vec<char> = left.chars().collect();
            let right: Vec<char> = right.chars().collect();
            let mut previous: Vec<usize> = (0..=right.len()).collect();
            for (row, left_char) in left.iter().enumerate() {
                let mut current = vec![row + 1];
                for (column, right_char) in right.iter().enumerate() {
                    current.push(std::cmp::min(
                        std::cmp::min(current[column] + 1, previous[column + 1] + 1),
                        previous[column] + usize::from(left_char != right_char),
                    ));
                }
                previous = current;
            }
            previous[right.len()]
        }
        for (left, right) in [
            ("", ""),
            ("", "ab"),
            ("ab", ""),
            ("kitten", "sitting"),
            ("sitting", "kitten"),
            ("colour", "colur"),
            ("abcdef", "abcdef"),
            ("axcdef", "abcdef"),
            ("abc", "abcde"),
            ("abcde", "abc"),
            ("résumé", "resume"),
            ("prix-six", "prixsix"),
        ] {
            for maximum in 0..=3 {
                let expected = Some(naive(left, right)).filter(|distance| *distance <= maximum);
                assert_eq!(
                    bounded_levenshtein(left, right, maximum),
                    expected,
                    "left={left:?} right={right:?} maximum={maximum}"
                );
            }
        }
    }

    #[tokio::test]
    async fn smart_ingest_does_not_guess_between_multiple_as_of_dates() {
        let (storage, _dir) = test_storage().await;
        let response = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "content": "Compared inventories as of 2026-03-04 and as of 2026-03-10.",
                "forceCreate": true
            })),
        )
        .await
        .unwrap();
        let node = storage
            .get_node(response["nodeId"].as_str().unwrap())
            .unwrap()
            .unwrap();
        assert!(node.valid_from.is_none());
        assert_eq!(
            response["validity"]["source"],
            "ambiguous_as_of_not_applied"
        );
        assert_eq!(
            response["validity"]["ambiguousPhrases"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn explicit_valid_from_overrides_as_of_inference() {
        let (storage, _dir) = test_storage().await;
        let response = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "content": "Reported as of 2026-03-04.",
                "validFrom": "2026-04-01",
                "forceCreate": true
            })),
        )
        .await
        .unwrap();
        let node = storage
            .get_node(response["nodeId"].as_str().unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(
            node.valid_from.unwrap().to_rfc3339(),
            "2026-04-01T00:00:00+00:00"
        );
        assert_eq!(response["validity"]["source"], "explicit");
    }

    #[tokio::test]
    async fn smart_ingest_suggests_similar_same_scope_tags_without_rewriting() {
        let (storage, _dir) = test_storage().await;
        storage
            .ingest_in_scope(
                IngestInput {
                    content: "existing user vocabulary".to_string(),
                    tags: vec!["prixsix".to_string()],
                    ..Default::default()
                },
                "user",
            )
            .unwrap();
        storage
            .ingest_in_scope(
                IngestInput {
                    content: "other project vocabulary".to_string(),
                    tags: vec!["project-only".to_string()],
                    ..Default::default()
                },
                "other-project",
            )
            .unwrap();

        let response = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "content": "new project note",
                "tags": ["prix-six", "project_onli"],
                "scope": "user",
                "forceCreate": true
            })),
        )
        .await
        .unwrap();
        assert_eq!(response["tagSuggestions"].as_array().unwrap().len(), 1);
        assert_eq!(
            response["tagSuggestions"][0]["similarExistingTag"],
            "prixsix"
        );
        assert_eq!(response["tagSuggestions"][0]["appliedAutomatically"], false);
        let node = storage
            .get_node(response["nodeId"].as_str().unwrap())
            .unwrap()
            .unwrap();
        assert!(node.tags.contains(&"prix-six".to_string()));
        assert!(!node.tags.contains(&"prixsix".to_string()));
    }

    #[tokio::test]
    async fn tag_suggestion_preflight_is_no_write_and_explicit_acceptance_is_revalidated() {
        let (storage, _dir) = test_storage().await;
        storage
            .ingest(IngestInput {
                content: "existing tag vocabulary".to_string(),
                tags: vec!["prixsix".to_string()],
                ..Default::default()
            })
            .unwrap();
        let before = storage.get_stats().unwrap().total_nodes;
        let preview = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "content": "preflight-only memory",
                "tags": ["prix-six"],
                "previewTagSuggestions": true,
                "forceCreate": true
            })),
        )
        .await
        .unwrap();
        assert_eq!(preview["decision"], "preflight");
        assert_eq!(preview["wouldWrite"], false);
        assert_eq!(storage.get_stats().unwrap().total_nodes, before);

        let accepted = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "content": "accepted tag memory",
                "tags": ["prix-six"],
                "acceptedTagSuggestions": {"prix-six": "prixsix"},
                "forceCreate": true
            })),
        )
        .await
        .unwrap();
        let node = storage
            .get_node(accepted["nodeId"].as_str().unwrap())
            .unwrap()
            .unwrap();
        assert!(node.tags.contains(&"prixsix".to_string()));
        assert!(!node.tags.contains(&"prix-six".to_string()));

        let rejected = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "content": "invalid acceptance must not write",
                "tags": ["prix-six"],
                "acceptedTagSuggestions": {"prix-six": "not-current"},
                "forceCreate": true
            })),
        )
        .await;
        assert!(rejected.is_err());
        assert_eq!(storage.get_stats().unwrap().total_nodes, before + 1);
    }

    #[tokio::test]
    async fn accepted_tag_suggestions_canonicalize_duplicates_in_first_seen_order() {
        let (storage, _dir) = test_storage().await;
        storage
            .ingest(IngestInput {
                content: "existing canonical tag".to_string(),
                tags: vec!["prixsix".to_string()],
                ..Default::default()
            })
            .unwrap();
        let report = similar_tag_suggestions(&storage, "user", &["prix-six".to_string()]);
        let accepted =
            std::collections::BTreeMap::from([("prix-six".to_string(), "prixsix".to_string())]);

        for tags in [
            vec![
                "prix-six".to_string(),
                "prixsix".to_string(),
                "tail".to_string(),
            ],
            vec![
                "prixsix".to_string(),
                "prix-six".to_string(),
                "tail".to_string(),
            ],
        ] {
            assert_eq!(
                apply_accepted_tag_suggestions(tags, &report, Some(&accepted)).unwrap(),
                vec!["prixsix", "tail"]
            );
        }
    }

    #[tokio::test]
    async fn accepted_tag_suggestions_reject_secret_keys_and_values_without_echo_or_write() {
        let (storage, _dir) = test_storage().await;
        let credential = format!("ghp_{}", "a".repeat(36));
        let secret_key = serde_json::to_value(std::collections::BTreeMap::from([(
            credential.clone(),
            "safe".to_string(),
        )]))
        .unwrap();
        let secret_value = serde_json::to_value(std::collections::BTreeMap::from([(
            "safe".to_string(),
            credential.clone(),
        )]))
        .unwrap();
        for (tags, accepted) in [
            (vec![credential.clone()], secret_key),
            (vec!["safe".to_string()], secret_value),
        ] {
            let result = execute(
                &storage,
                &test_cognitive(),
                Some(serde_json::json!({
                    "content": "accepted suggestion secret guard fixture",
                    "tags": tags,
                    "acceptedTagSuggestions": accepted,
                    "forceCreate": true,
                })),
            )
            .await
            .unwrap_err();
            assert!(result.contains("probable credential"));
            assert!(!result.contains(&credential));
            assert_eq!(storage.get_stats().unwrap().total_nodes, 0);
        }
    }

    #[tokio::test]
    async fn tag_suggestion_cost_bounds_are_explicit_and_tagless_ingest_skips_vocabulary() {
        let (storage, _dir) = test_storage().await;
        storage
            .ingest(IngestInput {
                content: "legacy overlong vocabulary fixture".to_string(),
                tags: vec!["x".repeat(201)],
                ..Default::default()
            })
            .unwrap();

        let tagless = similar_tag_suggestions(&storage, "user", &[]);
        assert_eq!(tagless.status["status"], "complete");
        assert_eq!(tagless.status["vocabularyScanned"], false);

        // One overlong stored tag degrades gracefully: it is skipped and
        // counted instead of disabling suggestions for the whole scope.
        let bounded = similar_tag_suggestions(&storage, "user", &["candidate".to_string()]);
        assert_eq!(bounded.status["status"], "complete");
        assert_eq!(bounded.status["vocabularyScanned"], true);
        assert_eq!(bounded.status["vocabularyCount"], 0);
        assert_eq!(bounded.status["ignoredOverlongVocabularyTags"], 1);
        assert!(bounded.suggestions.is_empty());
    }

    #[tokio::test]
    async fn vocabulary_beyond_ten_thousand_tags_is_reported_unavailable() {
        let (storage, _dir) = test_storage().await;
        // 21 memories x 500 distinct tags = 10,500 eligible vocabulary tags.
        for batch in 0..21 {
            let tags: Vec<String> = (0..500)
                .map(|index| format!("bulk-{batch:02}-{index:03}"))
                .collect();
            storage
                .ingest(IngestInput {
                    content: format!("vocabulary bound fixture {batch}"),
                    tags,
                    ..Default::default()
                })
                .unwrap();
        }

        let report = similar_tag_suggestions(&storage, "user", &["candidate".to_string()]);
        assert_eq!(report.status["status"], "unavailable");
        assert!(
            report.status["reason"]
                .as_str()
                .unwrap()
                .contains("exceeds the 10000-tag"),
            "the 10,000-tag bound stays a hard error and is not masked by overlong skipping"
        );
        assert!(report.suggestions.is_empty());
    }

    #[tokio::test]
    async fn requested_tags_beyond_the_50_bound_are_reported_truncated_and_skipped() {
        let (storage, _dir) = test_storage().await;
        storage
            .ingest(IngestInput {
                content: "vocabulary for the truncation bound".to_string(),
                tags: vec!["colour".to_string()],
                ..Default::default()
            })
            .unwrap();

        // 50 requested tags stay within the bound: the last one is scanned.
        let mut within: Vec<String> = (0..49)
            .map(|index| format!("filler-tag-{index:02}"))
            .collect();
        within.push("colur".to_string());
        let report = similar_tag_suggestions(&storage, "user", &within);
        assert_eq!(report.status["status"], "complete");
        assert_eq!(report.status["requestedTagsTruncated"], false);
        assert!(
            report
                .suggestions
                .iter()
                .any(|suggestion| suggestion["inputTag"] == "colur"),
            "the 50th requested tag is still scanned"
        );

        // A 51st tag crosses the bound: it is reported truncated and skipped.
        let mut truncated: Vec<String> = (0..50)
            .map(|index| format!("filler-tag-{index:02}"))
            .collect();
        truncated.push("colur".to_string());
        let report = similar_tag_suggestions(&storage, "user", &truncated);
        assert_eq!(report.status["status"], "complete");
        assert_eq!(report.status["requestedTagsTruncated"], true);
        assert!(
            !report
                .suggestions
                .iter()
                .any(|suggestion| suggestion["inputTag"] == "colur"),
            "the 51st requested tag must not be scanned"
        );
    }

    #[tokio::test]
    async fn tag_suggestions_never_echo_secret_shaped_existing_vocabulary() {
        let (storage, _dir) = test_storage().await;
        let credential = format!("ghp_{}", "a".repeat(36));
        storage
            .ingest_with_secret_policy(
                IngestInput {
                    content: "intentional local credential vocabulary fixture".to_string(),
                    tags: vec![credential.clone()],
                    ..Default::default()
                },
                SecretPolicy::AllowExplicitly,
            )
            .unwrap();

        let near_match = format!("ghq_{}", "a".repeat(36));
        let report = similar_tag_suggestions(&storage, "user", &[near_match]);
        assert_eq!(report.status["status"], "complete");
        assert_eq!(report.status["ignoredSecretShapedVocabularyTags"], 1);
        assert!(report.suggestions.is_empty());
        assert!(
            !serde_json::to_string(&report.status)
                .unwrap()
                .contains(&credential)
        );
    }

    #[tokio::test]
    async fn tag_suggestions_cover_issue_variants_and_suppress_noise_deterministically() {
        let (storage, _dir) = test_storage().await;
        for tag in [
            "prixsix",
            "codebase:prix-six",
            "Prix-Six",
            "colour",
            "coolur",
            "exact",
            "ab",
        ] {
            storage
                .ingest(IngestInput {
                    content: format!("vocabulary {tag}"),
                    tags: vec![tag.to_string()],
                    ..Default::default()
                })
                .unwrap();
        }
        let report = similar_tag_suggestions(
            &storage,
            "user",
            &[
                "prix-six".to_string(),
                "colur".to_string(),
                "exact".to_string(),
                "ac".to_string(),
                "banana".to_string(),
            ],
        );
        assert_eq!(report.status["status"], "complete");
        let prix: Vec<_> = report
            .suggestions
            .iter()
            .filter(|suggestion| suggestion["inputTag"] == "prix-six")
            .collect();
        assert!(prix.iter().any(|suggestion| {
            suggestion["similarExistingTag"] == "prixsix"
                && suggestion["reason"] == "punctuation_variant"
        }));
        assert!(prix.iter().any(|suggestion| {
            suggestion["similarExistingTag"] == "codebase:prix-six"
                && suggestion["reason"] == "namespace_variant"
        }));
        assert!(report.suggestions.iter().any(|suggestion| {
            suggestion["inputTag"] == "colur" && suggestion["reason"] == "edit_distance"
        }));
        assert!(!report.suggestions.iter().any(|suggestion| {
            matches!(
                suggestion["inputTag"].as_str(),
                Some("exact" | "ac" | "banana")
            )
        }));

        let colur: Vec<&str> = report
            .suggestions
            .iter()
            .filter(|suggestion| suggestion["inputTag"] == "colur")
            .filter_map(|suggestion| suggestion["similarExistingTag"].as_str())
            .collect();
        let mut sorted = colur.clone();
        sorted.sort();
        assert_eq!(colur, sorted, "equal-score ties are lexicographic");
    }

    #[tokio::test]
    async fn batch_preflight_has_as_of_and_tag_suggestion_parity_without_writes() {
        let (storage, _dir) = test_storage().await;
        storage
            .ingest(IngestInput {
                content: "batch vocabulary".to_string(),
                tags: vec!["prixsix".to_string()],
                ..Default::default()
            })
            .unwrap();
        let before = storage.get_stats().unwrap().total_nodes;
        let response = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [{
                    "content": "Batch inventory as of 2026-03-10",
                    "tags": ["prix-six"],
                    "previewTagSuggestions": true
                }]
            })),
        )
        .await
        .unwrap();
        assert_eq!(response["results"][0]["status"], "previewed");
        assert_eq!(
            response["results"][0]["validity"]["source"],
            "inferred_as_of"
        );
        assert_eq!(
            response["results"][0]["tagSuggestions"][0]["similarExistingTag"],
            "prixsix"
        );
        assert_eq!(storage.get_stats().unwrap().total_nodes, before);
    }

    #[tokio::test]
    async fn batch_tag_suggestion_acceptance_and_rejection_match_single_mode() {
        let (storage, _dir) = test_storage().await;
        storage
            .ingest(IngestInput {
                content: "batch acceptance vocabulary".to_string(),
                tags: vec!["prixsix".to_string()],
                ..Default::default()
            })
            .unwrap();
        let before = storage.get_stats().unwrap().total_nodes;

        let response = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    {
                        "content": "batch accepted suggestion",
                        "tags": ["prix-six"],
                        "acceptedTagSuggestions": {"prix-six": "prixsix"},
                        "forceCreate": true
                    },
                    {
                        "content": "batch rejected suggestion",
                        "tags": ["prix-six"],
                        "acceptedTagSuggestions": {"prix-six": "not-current"},
                        "forceCreate": true
                    }
                ]
            })),
        )
        .await
        .unwrap();

        assert_eq!(response["results"][0]["status"], "saved");
        let accepted = storage
            .get_node(response["results"][0]["nodeId"].as_str().unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(accepted.tags, vec!["prixsix"]);
        assert_eq!(response["results"][1]["status"], "error");
        assert!(
            response["results"][1]["reason"]
                .as_str()
                .unwrap()
                .contains("not a current same-scope suggestion")
        );
        assert_eq!(storage.get_stats().unwrap().total_nodes, before + 1);
    }

    #[tokio::test]
    async fn batch_as_of_inference_persists_when_the_item_is_saved() {
        let (storage, _dir) = test_storage().await;
        let response = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [{
                    "content": "Saved batch inventory As Of 2026-03-10",
                    "forceCreate": true
                }]
            })),
        )
        .await
        .unwrap();
        let item = &response["results"][0];
        let node = storage
            .get_node(item["nodeId"].as_str().unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(
            node.valid_from.unwrap().to_rfc3339(),
            "2026-03-10T00:00:00+00:00"
        );
        assert_eq!(item["validity"]["source"], "inferred_as_of");
    }

    #[tokio::test]
    async fn smart_ingest_rejects_invalid_dates_and_reversed_ranges() {
        let (storage, _dir) = test_storage().await;
        for args in [
            serde_json::json!({"content": "bad date", "validFrom": "07/01/2026"}),
            serde_json::json!({
                "content": "bad range",
                "validFrom": "2026-10-01",
                "validUntil": "2026-07-01"
            }),
        ] {
            assert!(
                execute(&storage, &test_cognitive(), Some(args))
                    .await
                    .is_err()
            );
        }
        assert_eq!(storage.get_stats().unwrap().total_nodes, 0);
    }

    #[tokio::test]
    async fn batch_validity_is_per_item_and_invalid_items_do_not_write() {
        let (storage, _dir) = test_storage().await;
        let response = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    {"content": "dated item", "validFrom": "2026-08-01"},
                    {"content": "invalid item", "validUntil": "yesterday"}
                ]
            })),
        )
        .await
        .unwrap();
        assert_eq!(response["success"], false);
        assert_eq!(response["results"][1]["status"], "error");
        let node = storage
            .get_node(response["results"][0]["nodeId"].as_str().unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(
            node.valid_from.unwrap().to_rfc3339(),
            "2026-08-01T00:00:00+00:00"
        );
        assert_eq!(storage.get_stats().unwrap().total_nodes, 1);
    }

    #[tokio::test]
    async fn test_smart_ingest_missing_args_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, &test_cognitive(), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_smart_ingest_whitespace_only_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "content": "   \t\n  " });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_smart_ingest_too_large_fails() {
        let (storage, _dir) = test_storage().await;
        let large = "x".repeat(1_000_001);
        let args = serde_json::json!({ "content": large });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too large"));
    }

    #[tokio::test]
    async fn test_smart_ingest_exactly_1mb_succeeds() {
        let (storage, _dir) = test_storage().await;
        let content = "x".repeat(1_000_000);
        let args = serde_json::json!({ "content": content });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_smart_ingest_with_node_type() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "A concept to remember",
            "node_type": "concept"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_smart_ingest_with_tags_and_source() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "Tagged and sourced memory",
            "tags": ["test", "smart-ingest"],
            "source": "unit-test"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["success"], true);
    }

    #[tokio::test]
    async fn test_smart_ingest_response_has_importance_score() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "content": "Important memory content" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        let value = result.unwrap();
        assert!(value["importanceScore"].is_number());
    }

    #[tokio::test]
    async fn test_smart_ingest_missing_content_field_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "tags": ["test"] });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("content"));
    }

    // ========================================================================
    // TESTS PORTED FROM ingest.rs (v1.7.0 merge)
    // ========================================================================

    #[tokio::test]
    async fn test_smart_ingest_with_all_optional_fields() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "Complex memory with all metadata.",
            "node_type": "decision",
            "tags": ["architecture", "design"],
            "source": "team meeting notes"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert!(value["nodeId"].is_string());
    }

    #[tokio::test]
    async fn test_smart_ingest_default_node_type_is_fact() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "content": "Default type test content." });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let node_id = result.unwrap()["nodeId"].as_str().unwrap().to_string();
        let node = storage.get_node(&node_id).unwrap().unwrap();
        assert_eq!(node.node_type, "fact");
    }

    #[test]
    fn test_schema_has_optional_fields() {
        let schema_value = schema();
        assert!(schema_value["properties"]["node_type"].is_object());
        assert!(schema_value["properties"]["tags"].is_object());
        assert!(schema_value["properties"]["source"].is_object());
    }

    #[tokio::test]
    async fn test_smart_ingest_with_source() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "MCP protocol version 2024-11-05 is the current standard.",
            "source": "https://modelcontextprotocol.io/spec"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["success"], true);
    }

    // ========================================================================
    // BATCH MODE TESTS (ported from checkpoint.rs, v1.7.0 merge)
    // ========================================================================

    #[tokio::test]
    async fn test_batch_empty_items_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({ "items": [] })),
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_batch_ingest() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    { "content": "First batch item", "tags": ["test"] },
                    { "content": "Second batch item", "tags": ["test"] }
                ]
            })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["mode"], "batch");
        assert_eq!(value["batchMergePolicy"], "force_create");
        assert_eq!(value["summary"]["total"], 2);
    }

    #[tokio::test]
    async fn test_batch_ingest_saves_safe_item_and_rejects_secret_item() {
        let (storage, _dir) = test_storage().await;
        let secret = format!("ghp_{}", "A".repeat(36));
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    { "content": "safe batch memory" },
                    { "content": format!("batch credential: {secret}") }
                ]
            })),
        )
        .await
        .unwrap();

        assert_eq!(result["mode"], "batch");
        assert_eq!(result["results"][0]["status"], "saved");
        assert_eq!(result["results"][1]["status"], "rejected");
        assert!(
            !result["results"][1]["reason"]
                .as_str()
                .unwrap()
                .contains(&secret),
            "batch result must not echo the rejected credential"
        );
        assert_eq!(result["summary"]["created"], 1);
        assert_eq!(result["summary"]["errors"], 1);
        assert_eq!(
            storage.get_stats().unwrap().total_nodes,
            1,
            "safe batch entries may persist while rejected entries never do"
        );
    }

    #[tokio::test]
    async fn test_batch_defaults_to_force_create_for_caller_separated_items() {
        // Default policy (no explicit forceCreate) force-creates each
        // caller-separated item so they stay separate.
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    { "content": "Jira tickets should not auto-assign sprint fields." },
                    { "content": "Sprint planning summaries should not append Jira status labels." }
                ]
            })),
        )
        .await;

        let value = result.unwrap();
        assert_eq!(value["batchMergePolicy"], "force_create");
        assert_eq!(value["summary"]["created"], 2);
        assert_eq!(value["summary"]["updated"], 0);
        for item in value["results"].as_array().unwrap() {
            assert_eq!(item["decision"], "create");
            assert!(item["reason"].as_str().unwrap().contains("Forced creation"));
        }
    }

    #[tokio::test]
    async fn test_batch_explicit_force_create_false_is_honored() {
        // Regression (#130): an EXPLICIT forceCreate:false must NOT be silently
        // inverted to force-create by the default policy. Distinct/novel items are
        // still created (PE gating creates novel content), but NOT via the
        // "Forced creation" path.
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "forceCreate": false,
                "items": [
                    { "content": "Jira tickets should not auto-assign sprint fields." },
                    { "content": "Sprint planning summaries should not append Jira status labels." }
                ]
            })),
        )
        .await;

        let value = result.unwrap();
        assert_eq!(value["summary"]["created"], 2, "novel items still created");
        for item in value["results"].as_array().unwrap() {
            assert!(
                !item["reason"].as_str().unwrap().contains("Forced creation"),
                "explicit forceCreate:false must not force-create"
            );
        }
    }

    #[tokio::test]
    async fn test_batch_rejects_invalid_merge_policy() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "batchMergePolicy": "merge_everything",
                "items": [{ "content": "Invalid policy should fail." }]
            })),
        )
        .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("batchMergePolicy"));
    }

    #[tokio::test]
    async fn test_batch_skips_empty_content() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    { "content": "Valid item" },
                    { "content": "" },
                    { "content": "Another valid item" }
                ]
            })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["summary"]["skipped"], 1);
    }

    #[tokio::test]
    async fn test_batch_missing_args_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, &test_cognitive(), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_batch_exceeds_20_items_fails() {
        let (storage, _dir) = test_storage().await;
        let items: Vec<serde_json::Value> = (0..21)
            .map(|i| serde_json::json!({ "content": format!("Item {}", i) }))
            .collect();
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({ "items": items })),
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Maximum 20 items"));
    }

    #[tokio::test]
    async fn test_batch_exactly_20_items_succeeds() {
        let (storage, _dir) = test_storage().await;
        let items: Vec<serde_json::Value> = (0..20)
            .map(|i| serde_json::json!({ "content": format!("Item {}", i) }))
            .collect();
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({ "items": items })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["summary"]["total"], 20);
    }

    #[tokio::test]
    async fn test_batch_skips_whitespace_only_content() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    { "content": "   \t\n  " },
                    { "content": "Valid content" }
                ]
            })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["summary"]["skipped"], 1);
        assert_eq!(value["summary"]["created"], 1);
    }

    #[tokio::test]
    async fn test_batch_single_item_succeeds() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [{ "content": "Single item" }]
            })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["summary"]["total"], 1);
        assert_eq!(value["success"], true);
    }

    #[tokio::test]
    async fn test_batch_items_with_all_fields() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [{
                    "content": "Full fields item",
                    "tags": ["test", "batch"],
                    "node_type": "decision",
                    "source": "test-suite"
                }]
            })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["summary"]["created"], 1);
    }

    #[tokio::test]
    async fn test_batch_results_array_matches_items() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    { "content": "First" },
                    { "content": "" },
                    { "content": "Third" }
                ]
            })),
        )
        .await;
        let value = result.unwrap();
        let results = value["results"].as_array().unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0]["index"], 0);
        assert_eq!(results[1]["index"], 1);
        assert_eq!(results[1]["status"], "skipped");
        assert_eq!(results[2]["index"], 2);
    }

    #[tokio::test]
    async fn test_batch_success_true_when_only_skipped() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    { "content": "" },
                    { "content": "   " }
                ]
            })),
        )
        .await;
        let value = result.unwrap();
        assert_eq!(value["success"], true); // skipped ≠ errors
        assert_eq!(value["summary"]["errors"], 0);
        assert_eq!(value["summary"]["skipped"], 2);
    }

    #[tokio::test]
    async fn test_batch_has_importance_scores() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [{ "content": "Important batch memory content" }]
            })),
        )
        .await;
        let value = result.unwrap();
        let results = value["results"].as_array().unwrap();
        assert!(results[0]["importanceScore"].is_number());
    }

    #[tokio::test]
    async fn test_batch_force_create_global() {
        let (storage, _dir) = test_storage().await;
        // Three items with very similar content + global forceCreate
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "forceCreate": true,
                "items": [
                    { "content": "Physics question about quantum mechanics and wave functions" },
                    { "content": "Physics question about quantum mechanics and wave equations" },
                    { "content": "Physics question about quantum mechanics and wave behavior" }
                ]
            })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["mode"], "batch");
        // All 3 should be created separately, not merged
        assert_eq!(value["summary"]["created"], 3);
        assert_eq!(value["summary"]["updated"], 0);
        // Each result should say "Forced creation"
        let results = value["results"].as_array().unwrap();
        for r in results {
            assert_eq!(r["decision"], "create");
            assert!(r["reason"].as_str().unwrap().contains("Forced"));
        }
    }

    #[tokio::test]
    async fn test_batch_force_create_per_item() {
        let (storage, _dir) = test_storage().await;
        // Mix of forced and non-forced items
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    { "content": "Forced item one", "forceCreate": true },
                    { "content": "Normal item two" },
                    { "content": "Forced item three", "forceCreate": true }
                ]
            })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        let results = value["results"].as_array().unwrap();
        // Forced items should say "Forced creation"
        assert_eq!(results[0]["decision"], "create");
        assert!(results[0]["reason"].as_str().unwrap().contains("Forced"));
        // Non-forced item gets normal processing
        assert_eq!(results[1]["status"], "saved");
        // Third forced item
        assert_eq!(results[2]["decision"], "create");
        assert!(results[2]["reason"].as_str().unwrap().contains("Forced"));
    }

    #[tokio::test]
    async fn test_no_content_no_items_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "tags": ["orphan"] });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("content"));
    }
}
