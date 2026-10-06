//! GhostLink: the causal proof engine's composition surface.
//!
//! The ghost is the never-composed: a pairing the recorded graph already
//! implies (or forces) that nobody has summoned. Eight modes:
//!
//! * `propose` — never-composed pairs. `lens='bridge'` (default): pairs
//!   within 3 undirected hops over recorded touched / derived_from /
//!   closed_by edges. `lens='divergent'`: pairs joined by no recorded edge,
//!   scored `min(Path_min, 7) x typed divergence`, with forced
//!   juxtapositions for memories that have no typed profile. Every
//!   candidate carries its proof.
//! * `bounty` — lanes from woven compositions by exact outcome type, plus
//!   the bridge lens as the never-composed lane.
//! * `weave` — record a composition outcome (a write): on a Strata log a
//!   composition record plus `derived_from` edges to both memories, each
//!   with a receipt; on a legacy SQLite store an outcome on `event_id`.
//! * `map` — recorded subgraph around an exact center for visualization.
//! * `inspect` — `view` recent / get / memory / neighbors over woven
//!   compositions.
//! * `explore` — `kind` chain / associations / bridges over recorded
//!   typed edges.
//! * `predict` — context-ahead memories; on Strata only from exact handles
//!   (`current_file` code anchors). Free-text topics are refused.
//! * `harden` — seed invariant bug-class laws (a write), idempotent by law id.
//!
//! Binding principle: ZERO vector / strings / RAG. Nothing here admits,
//! ranks, pairs or explains through embeddings, text-vector cosine,
//! BM25/FTS, keyword or shared-term overlap, tag-name overlap, or free-text
//! retrieval. `graph` stays a hidden, deprecated alias of this tool.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use super::composed_graph::OUTCOME_TYPES;
use crate::cognitive::CognitiveEngine;
use crate::strata_memory::ghostlink::{self as engine, HardenSeed, Lens, ProposeRequest};
use vestige_core::Storage;

/// Every GhostLink mode, in schema order.
pub const MODES: &[&str] = &[
    "propose", "bounty", "weave", "map", "inspect", "explore", "predict", "harden",
];
/// `inspect` views.
pub const VIEWS: &[&str] = &["recent", "get", "memory", "neighbors"];
/// `explore` kinds.
pub const KINDS: &[&str] = &["chain", "associations", "bridges"];
/// `propose` lenses.
pub const LENSES: &[&str] = &["bridge", "divergent"];
/// Law file name, read from the data dir, then `~/.vestige/`.
pub const LAWS_FILE: &str = "ghostlink-laws.json";

/// GhostLink tool schema.
pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": MODES,
                "description": "propose: never-composed pairs with proofs (lens bridge|divergent). bounty: lanes from woven outcomes. weave: record a composition outcome (write). map: recorded subgraph. inspect: woven compositions (view). explore: typed paths (kind). predict: memories anchored to an exact file path (context.current_file), ordered by FSRS retention. harden: seed invariant laws (write)."
            },
            "lens": {
                "type": "string",
                "enum": LENSES,
                "default": "bridge",
                "description": "[propose] 'bridge' (default): pairs within 3 recorded touched/derived_from/closed_by hops. 'divergent': pairs no recorded edge joins, scored min(Path_min,7) x typed divergence; cold pairs are forced juxtapositions. [weave] lens the pair came from."
            },
            "cursor": { "type": "string", "description": "[propose divergent] nextCursor from the previous page (same log head and filter)." },
            "view": {
                "type": "string",
                "enum": VIEWS,
                "description": "[inspect] Which composition view to read."
            },
            "kind": {
                "type": "string",
                "enum": KINDS,
                "description": "[explore] Which recorded-path read to run."
            },
            "from": { "type": "string", "description": "[explore] Source memory ID." },
            "to": { "type": "string", "description": "[explore:chain/bridges] Target memory ID." },
            "context": { "type": "object", "description": "[predict] current_file (exact path), codebase. current_topics is free text and is refused on Strata." },
            "center_id": { "type": "string", "description": "[map] Exact center node id." },
            "query": { "type": "string", "description": "[map] Legacy SQLite only: pick the center by search. Refused on Strata (similarity_disabled)." },
            "depth": { "type": "integer", "minimum": 1, "maximum": 3, "description": "[map] Traversal depth (1-3, default 2)." },
            "max_nodes": { "type": "integer", "description": "[map] Max nodes (default 50, capped 200)." },
            "first_id": { "type": "string", "description": "[weave] One memory of the composed pair." },
            "second_id": { "type": "string", "description": "[weave] The other memory of the pair." },
            "event_id": { "type": "string", "description": "[inspect:get] Composition record id. [weave] Legacy SQLite composition event id." },
            "memory_id": { "type": "string", "description": "[inspect:memory/neighbors] Memory id." },
            "tags": { "type": "array", "items": { "type": "string" }, "description": "[propose/bounty] Exact tags; both members must carry one." },
            "outcome_type": {
                "type": "string",
                "enum": OUTCOME_TYPES,
                "description": "[weave] Outcome to record."
            },
            "evidence": {
                "type": "array",
                "maxItems": 8,
                "description": "[weave] External findings for this pair, e.g. from a web search on its composition question: url, sha256 of the fetched content, retrievedAt (RFC 3339), optional note. Recorded on the composition record and tagged evidence:<sha256>. Vestige never fetches the URL.",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": { "type": "string" },
                        "sha256": { "type": "string" },
                        "retrievedAt": { "type": "string" },
                        "note": { "type": "string" }
                    },
                    "required": ["url", "sha256", "retrievedAt"]
                }
            },
            "scope": { "type": "string", "default": "user", "description": "[propose/bounty] Exact project namespace." },
            "includeCrossScope": { "type": "boolean", "default": false, "description": "[propose/bounty] Consider every namespace." },
            "limit": { "type": "integer", "description": "Max results (per-mode defaults, clamped).", "minimum": 1, "maximum": 100 }
        },
        "required": ["mode"]
    })
}

/// The legacy `graph` action a GhostLink call corresponds to, for surfaces
/// keyed by graph action (Strata withholding). `harden` has none.
pub fn graph_action_for(args: Option<&Value>) -> Option<String> {
    let field = |name: &str| args.and_then(|a| a.get(name)).and_then(Value::as_str);
    Some(
        match field("mode")? {
            "propose" => "never_composed",
            "bounty" => "bounty_mode",
            "weave" => "label",
            "map" => "memory_graph",
            "predict" => "predict",
            "inspect" => field("view")?,
            "explore" => field("kind")?,
            _ => return None,
        }
        .to_string(),
    )
}

/// The GhostLink `(mode, view|kind)` for a legacy `graph` action.
pub fn mode_for_graph_action(
    action: &str,
) -> Option<(&'static str, Option<(&'static str, &'static str)>)> {
    Some(match action {
        "never_composed" => ("propose", None),
        "bounty_mode" => ("bounty", None),
        "label" => ("weave", None),
        "memory_graph" => ("map", None),
        "predict" => ("predict", None),
        "recent" => ("inspect", Some(("view", "recent"))),
        "get" => ("inspect", Some(("view", "get"))),
        "memory" => ("inspect", Some(("view", "memory"))),
        "neighbors" => ("inspect", Some(("view", "neighbors"))),
        "chain" => ("explore", Some(("kind", "chain"))),
        "associations" => ("explore", Some(("kind", "associations"))),
        "bridges" => ("explore", Some(("kind", "bridges"))),
        _ => return None,
    })
}

// ----------------------------------------------------------------------
// Invariant laws
// ----------------------------------------------------------------------

/// One invariant bug-class law.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InvariantLaw {
    /// Stable id, e.g. `LAW-REPLAY`. Becomes a tag and the idempotency key.
    pub id: String,
    /// Short name.
    pub name: String,
    /// The law.
    pub law: String,
    /// Words that tend to appear where the law applies. Reading material
    /// only: never used for matching anywhere.
    #[serde(default)]
    pub signals: Vec<String>,
    /// Severity when the invariant is absent.
    #[serde(default = "default_severity")]
    pub severity_if_absent: String,
}

fn default_severity() -> String {
    "Critical".to_string()
}

#[derive(Debug, Deserialize)]
struct LawFile {
    ghostlink_invariant_laws: Vec<InvariantLaw>,
}

/// Where the seeded laws came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LawSource {
    /// A `ghostlink-laws.json` file.
    File(PathBuf),
    /// The six laws built into this binary.
    Embedded,
}

fn law(id: &str, name: &str, text: &str, signals: &[&str]) -> InvariantLaw {
    InvariantLaw {
        id: id.to_string(),
        name: name.to_string(),
        law: text.to_string(),
        signals: signals.iter().map(|s| s.to_string()).collect(),
        severity_if_absent: default_severity(),
    }
}

/// The six laws distilled from four root-caused bugs across four frameworks
/// (the 2026-09-25 harden set).
pub fn embedded_laws() -> Vec<InvariantLaw> {
    vec![
        law(
            "LAW-REPLAY",
            "Claim-Before-Execute (Replay Class)",
            "Wherever retries and side-effecting tools co-occur without a claim-before-execute ledger, duplicate execution exists. Fix: durable action ledger keyed on (run_id, operation, normalized_args_hash), claimed BEFORE execution, settled AFTER, returning prior receipt on any retry.",
            &[
                "retry",
                "re-execut",
                "idempoten",
                "nonce",
                "claim",
                "ledger",
                "replay",
                "duplicate",
            ],
        ),
        law(
            "LAW-EPOCH",
            "Attempt Epoch + Fencing Token",
            "Wherever a supervisor re-dispatches work from a checkpoint, the task identity must include an attempt epoch. Without it, a duplicate is byte-identical to the original and storage cannot distinguish them. Heartbeat must be bidirectional: the worker must be able to learn it lost the lease.",
            &[
                "checkpoint",
                "resume",
                "heartbeat",
                "fencing",
                "lease",
                "epoch",
                "attempt",
                "sweep",
                "timeout",
            ],
        ),
        law(
            "LAW-ANCHOR",
            "Ground-Truth State Anchor",
            "Any state the agent believes (cwd, balance, nonce, context) must be re-verified against ground truth before destructive operations. Belief stored in conversational transcript is lossy: compaction/restart loses it.",
            &["cwd", "compaction", "context loss", "anchor", "reset"],
        ),
        law(
            "LAW-SUPERSEDE",
            "Supersession + Validity Windows",
            "Wherever new facts are added without marking old contradicting facts as superseded, the context becomes contradictory. Fix: every write must emit ADD/UPDATE/DELETE/NONE events; every stored fact carries validFrom/validUntil; superseded facts are demoted and excluded from recall.",
            &[
                "supersede",
                "contradict",
                "stale",
                "valid_from",
                "valid_until",
                "accumulat",
                "pollut",
            ],
        ),
        law(
            "LAW-RECEIPT",
            "Success Receipts Bound to Verified Outcomes",
            "Every success signal must be a receipt bound to a verified outcome, not a status flag. An index that reports success but has unembedded chunks is a green-badge failure. Fix: coverage watermarks and verify-anchors.",
            &["coverage", "watermark", "verify", "receipt"],
        ),
        law(
            "LAW-SETTLE",
            "Settlement Identity Uniqueness",
            "Wherever multiple flows can produce the same settlement (mint, redeem, claim, distribution), each settlement must carry a unique identity that prevents duplicates.",
            &[
                "settle",
                "settlement",
                "identity",
                "duplicate",
                "double",
                "mint",
                "redeem",
                "distribution",
            ],
        ),
    ]
}

fn validate_laws(laws: &[InvariantLaw]) -> Result<(), String> {
    if laws.is_empty() {
        return Err("ghostlink_invariant_laws is empty".into());
    }
    let mut seen = std::collections::HashSet::new();
    for law in laws {
        let id = law.id.trim();
        if id.is_empty()
            || id.len() > 64
            || !id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(format!(
                "law id '{}' must be 1-64 ASCII letters, digits, '-' or '_'",
                law.id
            ));
        }
        if !seen.insert(id.to_string()) {
            return Err(format!("law id '{id}' appears twice"));
        }
        if law.name.trim().is_empty() || law.law.trim().is_empty() {
            return Err(format!("law '{id}' needs a non-empty name and law"));
        }
        if law.severity_if_absent.trim().is_empty() {
            return Err(format!("law '{id}' has an empty severity_if_absent"));
        }
    }
    Ok(())
}

/// Parse and validate one laws file.
pub fn parse_laws_file(path: &Path) -> Result<Vec<InvariantLaw>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| format!("ghostlink harden: cannot read {}: {err}", path.display()))?;
    let file: LawFile = serde_json::from_str(&text)
        .map_err(|err| format!("ghostlink harden: {} is malformed: {err}", path.display()))?;
    validate_laws(&file.ghostlink_invariant_laws)
        .map_err(|err| format!("ghostlink harden: {} is malformed: {err}", path.display()))?;
    Ok(file.ghostlink_invariant_laws)
}

/// `<data_dir>/ghostlink-laws.json`, else `<home>/.vestige/ghostlink-laws.json`,
/// else the embedded six. A file that exists but does not parse is an error
/// naming that file; it never falls through to a quieter source.
pub fn load_laws(
    data_dir: &Path,
    home: Option<&Path>,
) -> Result<(Vec<InvariantLaw>, LawSource), String> {
    let candidates = std::iter::once(data_dir.join(LAWS_FILE))
        .chain(home.map(|home| home.join(".vestige").join(LAWS_FILE)));
    for path in candidates {
        if path.is_file() {
            let laws = parse_laws_file(&path)?;
            return Ok((laws, LawSource::File(path)));
        }
    }
    Ok((embedded_laws(), LawSource::Embedded))
}

fn home_dir() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|dirs| dirs.home_dir().to_path_buf())
}

/// The memory a law becomes: its name, text, severity and signals as plain
/// reading material, tagged by exact identity.
pub fn seed_for(law: &InvariantLaw) -> HardenSeed {
    let id = law.id.trim().to_string();
    let signals = if law.signals.is_empty() {
        "none listed".to_string()
    } else {
        law.signals.join(", ")
    };
    HardenSeed {
        content: format!(
            "INVARIANT LAW [{id}]: {}\n{}\nSeverity if absent: {}\nSignals (reading material only, never used for matching): {signals}",
            law.name.trim(),
            law.law.trim(),
            law.severity_if_absent.trim(),
        ),
        tags: vec![
            "ghostlink".to_string(),
            "pattern-neuron".to_string(),
            "invariant-law".to_string(),
            id.clone(),
        ],
        law_id: id,
        name: law.name.trim().to_string(),
    }
}

/// Harden over the `Storage` trait (the legacy SQLite path). Same
/// idempotency key and counts as the Strata path.
fn harden_via_storage(storage: &Arc<Storage>, seeds: &[HardenSeed]) -> Result<Vec<Value>, String> {
    let mut present: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut offset = 0;
    loop {
        let page = storage
            .get_all_nodes(500, offset)
            .map_err(|err| err.to_string())?;
        if page.is_empty() {
            break;
        }
        for node in &page {
            if let Some(source) = &node.source
                && source.starts_with(engine::HARDEN_SOURCE_PREFIX)
                && node.suppression_count == 0
            {
                present
                    .entry(source.clone())
                    .or_insert_with(|| node.id.clone());
            }
        }
        offset += page.len() as i32;
    }
    let mut out = Vec::new();
    for seed in seeds {
        let source = format!("{}{}", engine::HARDEN_SOURCE_PREFIX, seed.law_id);
        if let Some(id) = present.get(&source) {
            out.push(json!({
                "lawId": seed.law_id, "name": seed.name,
                "decision": "already_present", "id": id,
            }));
            continue;
        }
        let written = storage.ingest(vestige_core::IngestInput {
            content: seed.content.clone(),
            node_type: "pattern".to_string(),
            source: Some(source),
            tags: seed.tags.clone(),
            ..Default::default()
        });
        out.push(match written {
            Ok(node) => json!({
                "lawId": seed.law_id, "name": seed.name, "decision": "seeded", "id": node.id,
                "receiptId": storage.get_receipt(&node.id).ok().flatten().map(|r| r.receipt_id),
            }),
            Err(err) => json!({
                "lawId": seed.law_id, "name": seed.name,
                "decision": "failed", "error": err.to_string(),
            }),
        });
    }
    Ok(out)
}

async fn harden(storage: &Arc<Storage>, strata: bool) -> Result<Value, String> {
    let (laws, source) = load_laws(storage.data_dir(), home_dir().as_deref())?;
    let seeds: Vec<HardenSeed> = laws.iter().map(seed_for).collect();
    let results = if strata {
        engine::harden(storage.as_ref(), &seeds)?
    } else {
        harden_via_storage(storage, &seeds)?
    };
    let count = |decision: &str| {
        results
            .iter()
            .filter(|r| r["decision"] == json!(decision))
            .count()
    };
    let (seeded, present, failed) = (count("seeded"), count("already_present"), count("failed"));
    let source = match source {
        LawSource::File(path) => json!({ "kind": "file", "path": path.display().to_string() }),
        LawSource::Embedded => json!({ "kind": "embedded", "path": Value::Null }),
    };
    Ok(json!({
        "mode": "harden",
        "lawsSource": source,
        "laws": seeds.len(),
        "seeded": seeded,
        "already_present": present,
        "failed": failed,
        "results": results,
        "note": "Each law is a pattern memory tagged ghostlink, pattern-neuron, invariant-law and its id. Its signals are reading material: nothing matches on them. Re-running harden is idempotent by law id.",
    }))
}

// ----------------------------------------------------------------------
// Dispatch
// ----------------------------------------------------------------------

fn str_arg<'a>(args: &'a Value, name: &str) -> Option<&'a str> {
    args.get(name)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn limit_arg(args: &Value, default: usize) -> usize {
    args.get("limit")
        .and_then(Value::as_u64)
        .map(|limit| limit.clamp(1, 100) as usize)
        .unwrap_or(default)
}

fn tags_arg(args: &Value) -> Vec<String> {
    args.get("tags")
        .and_then(Value::as_array)
        .map(|tags| {
            tags.iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|tag| !tag.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Exact scope from `scope` / `includeCrossScope` (default `user`).
fn scope_arg(args: &Value) -> Result<Option<String>, String> {
    if args
        .get("includeCrossScope")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Ok(None);
    }
    match args.get("scope") {
        None | Some(Value::Null) => Ok(Some(vestige_core::DEFAULT_MEMORY_SCOPE.to_string())),
        Some(Value::String(scope)) => {
            let scope = scope.trim();
            if scope.is_empty() {
                return Err("scope must not be empty".into());
            }
            Ok(Some(scope.to_string()))
        }
        Some(_) => Err("scope must be a string".into()),
    }
}

fn propose_request(args: &Value) -> Result<ProposeRequest, String> {
    Ok(ProposeRequest {
        lens: Lens::parse(args.get("lens").and_then(Value::as_str))?,
        scope: scope_arg(args)?,
        tags: tags_arg(args),
        limit: limit_arg(args, 10),
        cursor: str_arg(args, "cursor").map(str::to_string),
    })
}

fn check_mode_fields(mode: &str, args: &Value) -> Result<(), String> {
    if !matches!(mode, "propose" | "bounty")
        && (args.get("scope").is_some() || args.get("includeCrossScope").is_some())
    {
        return Err("scope and includeCrossScope apply only to modes propose and bounty".into());
    }
    if !matches!(mode, "propose" | "weave") && args.get("lens").is_some() {
        return Err("lens applies only to modes propose and weave".into());
    }
    if mode != "propose" && args.get("cursor").is_some() {
        return Err("cursor applies only to mode propose with lens 'divergent'".into());
    }
    Ok(())
}

fn sub_selector(mode: &str, args: &Value) -> Result<String, String> {
    let (key, allowed) = if mode == "inspect" {
        ("view", VIEWS)
    } else {
        ("kind", KINDS)
    };
    let sub = args
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("Missing '{key}' for mode '{mode}'."))?;
    if !allowed.contains(&sub) {
        return Err(format!(
            "Invalid {key} '{sub}' for mode '{mode}'. Allowed: {}.",
            allowed.join(", ")
        ));
    }
    Ok(sub.to_string())
}

/// Run one GhostLink call.
pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args = args.unwrap_or_else(|| json!({}));
    let mode = args
        .get("mode")
        .and_then(Value::as_str)
        .ok_or("Missing 'mode'. Use propose|bounty|weave|map|inspect|explore|predict|harden.")?
        .to_string();
    if !MODES.contains(&mode.as_str()) {
        return Err(format!(
            "Unknown mode '{mode}'. Use propose|bounty|weave|map|inspect|explore|predict|harden."
        ));
    }
    check_mode_fields(&mode, &args)?;
    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        return execute_strata(storage, &mode, &args).await;
    }
    execute_legacy(storage, cognitive, &mode, args).await
}

/// Strata dispatch: every mode answers from recorded structure.
async fn execute_strata(storage: &Arc<Storage>, mode: &str, args: &Value) -> Result<Value, String> {
    match mode {
        "propose" => engine::propose(storage.as_ref(), &propose_request(args)?),
        "bounty" => {
            let mut request = propose_request(args)?;
            request.lens = Lens::Bridge;
            request.cursor = None;
            engine::bounty(storage.as_ref(), &request)
        }
        "weave" => {
            let outcome =
                str_arg(args, "outcome_type").ok_or("outcome_type is required for weave")?;
            match (str_arg(args, "first_id"), str_arg(args, "second_id")) {
                (Some(first), Some(second)) => {
                    let evidence = engine::parse_evidence(args.get("evidence"))?;
                    engine::weave_with_evidence(
                        storage.as_ref(),
                        first,
                        second,
                        outcome,
                        args.get("lens").and_then(Value::as_str),
                        &evidence,
                    )
                }
                _ => Err("weave on a Strata log takes first_id and second_id (the two composed memories) and outcome_type; event_id names a legacy SQLite composition event".into()),
            }
        }
        "map" => {
            let mut forwarded = args.clone();
            if let Some(object) = forwarded.as_object_mut() {
                object.remove("mode");
            }
            super::graph::execute(storage, Some(forwarded)).await
        }
        "inspect" => {
            let view = sub_selector(mode, args)?;
            engine::inspect(
                storage.as_ref(),
                &view,
                str_arg(args, "event_id"),
                str_arg(args, "memory_id"),
                limit_arg(args, 10),
            )
        }
        "explore" => {
            let kind = sub_selector(mode, args)?;
            let from = str_arg(args, "from").ok_or("Missing 'from'")?;
            engine::explore(
                storage.as_ref(),
                &kind,
                from,
                str_arg(args, "to"),
                limit_arg(args, 10),
            )
        }
        "predict" => engine::predict(storage.as_ref(), args.get("context")),
        "harden" => harden(storage, true).await,
        other => Err(format!("Unknown mode '{other}'.")),
    }
}

/// Legacy SQLite dispatch through the unified graph engine.
async fn execute_legacy(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    mode: &str,
    mut args: Value,
) -> Result<Value, String> {
    let action = match mode {
        "harden" => return harden(storage, false).await,
        "propose" => {
            if Lens::parse(args.get("lens").and_then(Value::as_str))? == Lens::Divergent {
                return Err("lens 'divergent' needs a Strata log (Vestige 4.0); a legacy SQLite store answers lens 'bridge' only".into());
            }
            "never_composed".to_string()
        }
        "bounty" => "bounty_mode".to_string(),
        "weave" => "label".to_string(),
        "map" => "memory_graph".to_string(),
        "predict" => "predict".to_string(),
        "inspect" | "explore" => sub_selector(mode, &args)?,
        other => return Err(format!("Unknown mode '{other}'.")),
    };
    if let Some(object) = args.as_object_mut() {
        for key in [
            "mode",
            "view",
            "kind",
            "lens",
            "cursor",
            "first_id",
            "second_id",
        ] {
            object.remove(key);
        }
        object.insert("action".into(), Value::String(action));
    }
    super::graph_unified::execute(storage, cognitive, Some(args)).await
}

/// A legacy `graph` / `composed_graph` action on a Strata log: answered by
/// the same engine as the matching GhostLink mode, so the hidden aliases
/// can never disagree with `ghostlink`.
pub async fn execute_graph_action_on_strata(
    storage: &Arc<Storage>,
    action: &str,
    args: Option<Value>,
) -> Result<Value, String> {
    let (mode, sub) = mode_for_graph_action(action).ok_or_else(|| {
        format!(
            "Unknown graph action '{action}'. Use chain|associations|bridges|predict|memory_graph|recent|get|memory|neighbors|never_composed|bounty_mode|label."
        )
    })?;
    let mut args = args.unwrap_or_else(|| json!({}));
    if let Some(object) = args.as_object_mut() {
        object.remove("action");
        object.insert("mode".into(), json!(mode));
        if let Some((key, value)) = sub {
            object.insert(key.into(), json!(value));
        }
    }
    let mut result = execute_strata(storage, mode, &args).await?;
    if let Some(object) = result.as_object_mut() {
        object
            .entry("action".to_string())
            .or_insert_with(|| json!(action));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_lists_eight_modes_and_two_lenses() {
        let s = schema();
        let modes: Vec<&str> = s["properties"]["mode"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(
            modes,
            [
                "propose", "bounty", "weave", "map", "inspect", "explore", "predict", "harden"
            ]
        );
        assert_eq!(
            s["properties"]["lens"]["enum"],
            json!(["bridge", "divergent"])
        );
        assert_eq!(s["properties"]["lens"]["default"], json!("bridge"));
        assert!(
            s["properties"].get("data_dir").is_none(),
            "harden reads the store's own data dir"
        );
        for field in [
            "first_id",
            "second_id",
            "cursor",
            "view",
            "kind",
            "outcome_type",
        ] {
            assert!(s["properties"].get(field).is_some(), "{field} missing");
        }
        let doc = include_str!("ghostlink.rs");
        for mode in MODES {
            assert!(
                doc.contains(&format!("//! * `{mode}`")),
                "doc comment misses {mode}"
            );
        }
    }

    #[test]
    fn every_legacy_graph_action_is_reachable_and_maps_back() {
        let legacy = [
            "chain",
            "associations",
            "bridges",
            "predict",
            "memory_graph",
            "recent",
            "get",
            "memory",
            "neighbors",
            "never_composed",
            "bounty_mode",
            "label",
        ];
        for action in legacy {
            let (mode, sub) = mode_for_graph_action(action).expect(action);
            let mut args = json!({ "mode": mode });
            if let Some((key, value)) = sub {
                args[key] = json!(value);
            }
            assert_eq!(
                graph_action_for(Some(&args)).as_deref(),
                Some(action),
                "{action} does not round-trip through mode {mode}"
            );
        }
        assert_eq!(graph_action_for(Some(&json!({"mode": "harden"}))), None);
    }

    #[test]
    fn laws_load_from_data_dir_then_home_then_embedded() {
        let data = tempfile::tempdir().unwrap();
        let home = tempfile::tempdir().unwrap();
        let (laws, source) = load_laws(data.path(), Some(home.path())).unwrap();
        assert_eq!(source, LawSource::Embedded);
        assert_eq!(laws.len(), 6);

        std::fs::create_dir_all(home.path().join(".vestige")).unwrap();
        let home_file = home.path().join(".vestige").join(LAWS_FILE);
        std::fs::write(
            &home_file,
            json!({"ghostlink_invariant_laws": [
                {"id": "LAW-HOME", "name": "Home law", "law": "From home.", "signals": ["x"], "severity_if_absent": "High"}
            ], "version": "1", "date": "2026-09-29"})
            .to_string(),
        )
        .unwrap();
        let (laws, source) = load_laws(data.path(), Some(home.path())).unwrap();
        assert_eq!(source, LawSource::File(home_file.clone()));
        assert_eq!(laws[0].id, "LAW-HOME");

        let data_file = data.path().join(LAWS_FILE);
        std::fs::write(
            &data_file,
            json!({"ghostlink_invariant_laws": [
                {"id": "LAW-DATA", "name": "Data law", "law": "From the data dir."}
            ]})
            .to_string(),
        )
        .unwrap();
        let (laws, source) = load_laws(data.path(), Some(home.path())).unwrap();
        assert_eq!(source, LawSource::File(data_file.clone()));
        assert_eq!(laws[0].severity_if_absent, "Critical");

        std::fs::write(&data_file, "{ not json").unwrap();
        let err = load_laws(data.path(), Some(home.path())).unwrap_err();
        assert!(err.contains(&data_file.display().to_string()), "{err}");
        assert!(err.contains("malformed"), "{err}");

        std::fs::write(
            &data_file,
            json!({"ghostlink_invariant_laws": [
                {"id": "A", "name": "a", "law": "a"}, {"id": "A", "name": "b", "law": "b"}
            ]})
            .to_string(),
        )
        .unwrap();
        let err = load_laws(data.path(), Some(home.path())).unwrap_err();
        assert!(err.contains("appears twice"), "{err}");
    }

    #[test]
    fn a_law_seed_is_reading_material_tagged_by_identity() {
        let seed = seed_for(&embedded_laws()[0]);
        assert_eq!(seed.law_id, "LAW-REPLAY");
        assert_eq!(
            seed.tags,
            ["ghostlink", "pattern-neuron", "invariant-law", "LAW-REPLAY"]
        );
        assert!(seed.content.contains("Severity if absent: Critical"));
        assert!(seed.content.contains("never used for matching"));
    }
}
