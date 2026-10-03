//! GhostLink on a Strata log: the causal proof engine surface.
//!
//! Both propose lenses, weave, inspect, bounty, explore, predict and harden,
//! answered from the log's recorded structure: ids, exact scope / node_type
//! / tag identity, typed edges, woven composition records, FSRS state and
//! log sequence numbers. No content text, embedding, keyword, BM25/FTS or
//! tag-name overlap admits, ranks, pairs or explains anything here; content
//! previews ride along as reading material only. Every write goes through
//! the store's PROPOSE -> GATE -> EFFECT path and reports its receipt.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use serde_json::{Value, json};
use strata_store::{
    BridgeCandidate, BridgeReport, CompositionRecord, DivergentEval, GhostSnapshot, Lane, PathStep,
    PoolFilter, StrataStore,
};
use vestige_core::Storage;
use vestige_core::composition::{
    BridgeScore, OUTCOME_TYPES, bridge_score, composition_novelty, composition_trust,
    outcome_score_adjustment, outcome_signal,
};
use vestige_core::storage::NeverComposedCandidate;

use super::{StrataMemory, blocking_secrets, live_memory, map_store, receipt_id_for, retrievable};

/// What no lens ever uses to admit, rank, pair or explain.
pub const NEVER_USES: &[&str] = &[
    "embeddings",
    "text-vector cosine",
    "BM25/FTS",
    "keyword or shared-term overlap",
    "tag-name overlap",
    "free-text retrieval",
];

/// Source-system prefix of a seeded invariant law.
pub const HARDEN_SOURCE_PREFIX: &str = "ghostlink-harden:";

const PREVIEW_CHARS: usize = 160;
const EXPLORE_MAX_HOPS: u32 = 6;

const CLAIM_BOUNDARY: &str = "Pairs have no recorded joint composition in this log. A bridge pair is connected by recorded typed edges; a divergent pair is joined by no recorded edge at all. Neither is proof of causality, correctness or worldwide novelty: weave the outcome once tested.";
const JUXTAPOSITION_NOTE: &str = "Juxtaposition pairs are forced, not found: no recorded relation joins them and at least one member has no typed profile, so nothing measured their divergence. Treat each as an experiment to run, then weave the outcome; weaving gives both members a typed profile.";

/// Which propose lens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lens {
    /// Pairs within three recorded typed-edge hops.
    Bridge,
    /// Pairs no recorded edge joins.
    Divergent,
}

impl Lens {
    /// Parse the wire value; absent is the bridge lens.
    pub fn parse(value: Option<&str>) -> Result<Lens, String> {
        match value.map(str::trim) {
            None | Some("bridge") => Ok(Lens::Bridge),
            Some("divergent") => Ok(Lens::Divergent),
            Some(other) => Err(format!(
                "unknown lens '{other}'; use 'bridge' or 'divergent'"
            )),
        }
    }

    /// Wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Lens::Bridge => "bridge",
            Lens::Divergent => "divergent",
        }
    }
}

/// A propose call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposeRequest {
    /// Lens.
    pub lens: Lens,
    /// Exact scope; `None` considers every scope.
    pub scope: Option<String>,
    /// Exact tags; both members must carry one. Empty is no filter.
    pub tags: Vec<String>,
    /// Page size.
    pub limit: usize,
    /// Divergent-lens cursor from the previous page.
    pub cursor: Option<String>,
}

/// One invariant law to seed through `harden`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardenSeed {
    /// Law id (a tag and the idempotency key).
    pub law_id: String,
    /// Law name.
    pub name: String,
    /// Plain reading-material content.
    pub content: String,
    /// Exact tags.
    pub tags: Vec<String>,
}

fn memory(storage: &Storage) -> Result<Arc<StrataMemory>, String> {
    live_memory(storage)
        .ok_or_else(|| "ghostlink: the Strata log is not open in this process".to_string())
}

/// `mem-` ids keep their last 8 hex digits; other ids their first 8 chars.
pub fn short_id(id: &str) -> String {
    match id.strip_prefix("mem-") {
        Some(rest) if rest.len() > 8 => rest[rest.len() - 8..].to_string(),
        _ => id.chars().take(8).collect(),
    }
}

fn preview(content: &str) -> String {
    content
        .replace('\n', " ")
        .chars()
        .take(PREVIEW_CHARS)
        .collect()
}

fn node_type_of(store: &StrataStore, id: &str) -> String {
    store
        .get_node(id)
        .map(|record| record.node_type)
        .unwrap_or_else(|| "artifact".to_string())
}

fn path_json(path: &[PathStep]) -> Vec<Value> {
    path.iter()
        .map(|step| {
            json!({
                "from": step.from,
                "kind": step.kind,
                "to": step.to,
                "reversed": step.reversed,
            })
        })
        .collect()
}

fn kinds_arrow(path: &[PathStep]) -> String {
    path.iter()
        .map(|step| step.kind.as_str())
        .collect::<Vec<_>>()
        .join(" -> ")
}

fn plural(count: u32, one: &str, many: &str) -> String {
    format!("{count} {}", if count == 1 { one } else { many })
}

fn record_json(store: &StrataStore, record: &CompositionRecord) -> Value {
    json!({
        "id": record.id,
        "firstId": record.first_id,
        "secondId": record.second_id,
        "outcomeType": record.outcome_type,
        "lens": record.lens,
        "scope": record.scope,
        "createdAtMs": record.created_at_ms,
        "receiptId": record.origin_seq.map(receipt_id_for),
        "memberEdges": record.member_edges,
        "complete": record.complete,
        "firstNodeType": node_type_of(store, &record.first_id),
        "secondNodeType": node_type_of(store, &record.second_id),
    })
}

fn newest_first(records: &mut [&CompositionRecord]) {
    records.sort_by(|a, b| {
        b.origin_seq
            .cmp(&a.origin_seq)
            .then_with(|| b.id.cmp(&a.id))
    });
}

fn filter_of(request: &ProposeRequest) -> PoolFilter {
    PoolFilter {
        scope: request.scope.clone(),
        tags: request.tags.clone(),
    }
}

// ----------------------------------------------------------------------
// Bridge lens
// ----------------------------------------------------------------------

/// A bridge candidate with its score parts.
struct ScoredBridge {
    candidate: BridgeCandidate,
    score: BridgeScore,
    novelty: f64,
    trust: f64,
    adjustment: f64,
    prior_outcomes: Vec<String>,
}

/// The best `limit` bridge pairs with their score parts, best first, plus
/// the walk's report and how many pairs it admitted. Pairs are scored as
/// the bounded walk finds them (`GhostSnapshot::bridge_top`), with
/// retention read at the log's head clock, so a page is a function of the
/// log head and memory stays bounded by the page however many pairs a hub
/// admits.
fn rank_bridges(
    store: &StrataStore,
    snapshot: &GhostSnapshot<'_>,
    limit: usize,
) -> (BridgeReport, usize, Vec<ScoredBridge>) {
    let clock = store.head_clock_ms();
    let retention_of = |id: &str| {
        store
            .retrievability_at(id, clock)
            .ok()
            .flatten()
            .unwrap_or(0.0)
    };
    let mut retention: HashMap<String, f64> = HashMap::new();
    let mut outcomes: HashMap<String, Vec<String>> = HashMap::new();
    let (report, admitted) = snapshot.bridge_top(limit, |first, second, hops| {
        let mut member = |id: &str| {
            if !retention.contains_key(id) {
                retention.insert(id.to_string(), retention_of(id));
                outcomes.insert(id.to_string(), snapshot.prior_outcomes(id, id));
            }
            retention[id]
        };
        let trust = composition_trust(member(first), member(second));
        let novelty =
            composition_novelty(snapshot.weave_degree(first), snapshot.weave_degree(second));
        let adjustment = if outcomes[first].is_empty() && outcomes[second].is_empty() {
            0.0
        } else {
            outcome_score_adjustment(&snapshot.prior_outcomes(first, second))
        };
        bridge_score(hops, novelty, trust, adjustment).score
    });
    let scored = report
        .candidates
        .iter()
        .cloned()
        .map(|candidate| {
            let novelty = composition_novelty(
                snapshot.weave_degree(&candidate.first_id),
                snapshot.weave_degree(&candidate.second_id),
            );
            let trust = composition_trust(
                retention_of(&candidate.first_id),
                retention_of(&candidate.second_id),
            );
            let prior_outcomes = snapshot.prior_outcomes(&candidate.first_id, &candidate.second_id);
            let adjustment = outcome_score_adjustment(&prior_outcomes);
            let score = bridge_score(candidate.hops, novelty, trust, adjustment);
            ScoredBridge {
                candidate,
                score,
                novelty,
                trust,
                adjustment,
                prior_outcomes,
            }
        })
        .collect();
    (report, admitted, scored)
}

fn bridge_question(store: &StrataStore, candidate: &BridgeCandidate) -> String {
    format!(
        "{} {} reaches {} {} through {}, but they were never composed. What does the first establish about the second along this path? State one testable claim and the observation that would falsify it, then weave the outcome.",
        node_type_of(store, &candidate.first_id),
        short_id(&candidate.first_id),
        node_type_of(store, &candidate.second_id),
        short_id(&candidate.second_id),
        kinds_arrow(&candidate.path),
    )
}

fn bridge_reason(hops: u32) -> String {
    format!(
        "Connected by {} but never composed",
        plural(hops, "typed-edge hop", "typed-edge hops")
    )
}

fn content_preview(store: &StrataStore, id: &str) -> String {
    store
        .get_node(id)
        .map(|record| preview(&record.content))
        .unwrap_or_default()
}

/// Bridge candidates as the `MemoryStore` trait returns them (no proof).
pub(super) fn bridge_trait_candidates(
    memory: &StrataMemory,
    scope: Option<&str>,
    tags: Option<&[String]>,
    limit: usize,
) -> Vec<NeverComposedCandidate> {
    let store = memory.lock();
    let snapshot = store.ghost_snapshot(PoolFilter {
        scope: scope.map(str::to_string),
        tags: tags.map(<[String]>::to_vec).unwrap_or_default(),
    });
    let (_, _, scored) = rank_bridges(&store, &snapshot, limit);
    scored
        .into_iter()
        .map(|item| {
            let first = store.get_node(&item.candidate.first_id);
            let second = store.get_node(&item.candidate.second_id);
            NeverComposedCandidate {
                first_id: item.candidate.first_id.clone(),
                second_id: item.candidate.second_id.clone(),
                score: item.score.score,
                novelty_score: item.novelty,
                bridge_score: item.score.bridge,
                trust_score: item.trust,
                outcome_score_adjustment: item.adjustment,
                shared_tags: Vec::new(),
                boundary_tags: Vec::new(),
                shared_terms: Vec::new(),
                outcome_signal: outcome_signal(&item.prior_outcomes),
                prior_outcomes: item.prior_outcomes,
                first_node_type: first
                    .as_ref()
                    .map(|r| r.node_type.clone())
                    .unwrap_or_default(),
                second_node_type: second
                    .as_ref()
                    .map(|r| r.node_type.clone())
                    .unwrap_or_default(),
                first_preview: first
                    .as_ref()
                    .map(|r| preview(&r.content))
                    .unwrap_or_default(),
                second_preview: second
                    .as_ref()
                    .map(|r| preview(&r.content))
                    .unwrap_or_default(),
                reason: bridge_reason(item.candidate.hops),
                composition_question: bridge_question(&store, &item.candidate),
            }
        })
        .collect()
}

fn pool_json(request: &ProposeRequest, size: usize) -> Value {
    json!({
        "size": size,
        "scope": request.scope,
        "includeCrossScope": request.scope.is_none(),
        "tags": request.tags,
        "excludes": "retired (suppressed, undone, superseded) memories and composition records",
    })
}

fn propose_bridge(memory: &StrataMemory, request: &ProposeRequest) -> Value {
    let store = memory.lock();
    let snapshot = store.ghost_snapshot(filter_of(request));
    let (report, admitted, scored) = rank_bridges(&store, &snapshot, request.limit);
    let head = report.head_seq;
    let (pool_size, with_edges, touching, woven) = (
        report.pool_size,
        report.pool_nodes_with_admitting_edges,
        report.admitting_edges_touching_pool,
        report.woven_pairs_excluded,
    );
    let candidates: Vec<Value> = scored
        .iter()
        .map(|item| {
            let c = &item.candidate;
            json!({
                "firstId": c.first_id,
                "secondId": c.second_id,
                "lens": "bridge",
                "score": item.score.score,
                "noveltyScore": item.novelty,
                "bridgeScore": item.score.bridge,
                "trustScore": item.trust,
                "outcomeScoreAdjustment": item.adjustment,
                "sharedTags": [],
                "boundaryTags": [],
                "sharedTerms": [],
                "priorOutcomes": item.prior_outcomes,
                "outcomeSignal": outcome_signal(&item.prior_outcomes),
                "firstNodeType": node_type_of(&store, &c.first_id),
                "secondNodeType": node_type_of(&store, &c.second_id),
                "firstPreview": content_preview(&store, &c.first_id),
                "secondPreview": content_preview(&store, &c.second_id),
                "hops": c.hops,
                "reason": bridge_reason(c.hops),
                "compositionQuestion": bridge_question(&store, c),
                "proof": {
                    "lens": "bridge",
                    "headSeq": head,
                    "hops": c.hops,
                    "path": path_json(&c.path),
                    "neverWoven": true,
                },
            })
        })
        .collect();
    let mut admission = json!({
        "lens": "bridge",
        "admits": "pool pairs that reach each other within 3 undirected hops over recorded touched / derived_from / closed_by edges and were never woven",
        "scoring": "(1.5 + 1/hops) + (1/hops)*2 + novelty*1.5 + trust + outcomeAdjustment; novelty = mean(1/(1+weaveDegree)), trust = mean FSRS retention",
        "pool": pool_json(request, pool_size),
        "poolNodesWithAdmittingEdges": with_edges,
        "admittingEdgesTouchingPool": touching,
        "admittedPairs": admitted,
        "wovenPairsExcluded": woven,
        "headSeq": head,
        "neverUses": NEVER_USES,
    });
    if candidates.is_empty() {
        admission["emptyBecause"] = json!(if pool_size < 2 {
            "the pool has fewer than two live memories for this scope and tag filter"
        } else if with_edges == 0 {
            "no pool memory has a recorded touched, derived_from or closed_by edge; record typed edges or weave compositions, or ask lens='divergent' for pairs no recorded edge joins"
        } else if woven > 0 && admitted == 0 {
            "every pair within 3 typed hops is already woven"
        } else {
            "no two pool memories are within 3 recorded typed-edge hops"
        });
    }
    json!({
        "mode": "propose",
        "lens": "bridge",
        "candidates": candidates,
        "scope": request.scope,
        "includeCrossScope": request.scope.is_none(),
        "headSeq": head,
        "admission": admission,
        "evidenceStatus": "hypothesis",
        "globalNoveltyVerified": false,
        "claimBoundary": CLAIM_BOUNDARY,
    })
}

// ----------------------------------------------------------------------
// Divergent lens
// ----------------------------------------------------------------------

fn path_min_json(eval: &DivergentEval) -> Value {
    match eval.path_min {
        Some(hops) => json!(hops),
        None => json!("beyond_6"),
    }
}

fn nearest_path(eval: &DivergentEval) -> String {
    match eval.path_min {
        None => "none within 6 hops".to_string(),
        Some(hops) => format!(
            "{} via {}",
            plural(hops, "hop", "hops"),
            kinds_arrow(&eval.path)
        ),
    }
}

fn divergent_json(store: &StrataStore, head: u64, as_of_ms: i64, eval: &DivergentEval) -> Value {
    let (first, second) = (&eval.first_id, &eval.second_id);
    let first_record = store.get_node(first);
    let second_record = store.get_node(second);
    let scope_of = |record: &Option<strata_store::NodeRecord>| {
        record.as_ref().map(|r| r.scope.clone()).unwrap_or_default()
    };
    let retention = |id: &str| {
        store
            .retrievability_at(id, as_of_ms)
            .ok()
            .flatten()
            .unwrap_or(0.0)
    };
    let (first_type, second_type) = (node_type_of(store, first), node_type_of(store, second));
    let reason = match (eval.lane, eval.divergence) {
        (Lane::Measured, Some(divergence)) => format!(
            "No recorded relation joins them; Path_min {} over every recorded edge, typed divergence {divergence:.2}",
            eval.path_min
                .map(|hops| hops.to_string())
                .unwrap_or_else(|| "beyond 6".to_string())
        ),
        _ => "No recorded relation joins them and a member has no typed profile: a forced juxtaposition, not a finding".to_string(),
    };
    let question = format!(
        "No recorded relation connects {first_type} {} and {second_type} {} (verified at log seq {head}; nearest path: {}). Force a composition: name the mechanism that would connect them, one prediction it makes, and the observation that would kill it. If no mechanism survives, weave dead_end.",
        short_id(first),
        short_id(second),
        nearest_path(eval),
    );
    let mut proof = json!({
        "lens": "divergent",
        "headSeq": head,
        "noEdgeVerified": true,
        "neverWoven": true,
        "pathMin": path_min_json(eval),
        "pathVia": eval.path_via.as_str(),
        "divergenceMeasured": eval.lane == Lane::Measured,
        "typedNeighborCounts": eval.typed_neighbor_counts,
        "sharedTypedNeighbors": eval.shared_typed_neighbors,
    });
    if eval.path_min.is_some() {
        proof["path"] = json!(path_json(&eval.path));
    }
    json!({
        "firstId": first,
        "secondId": second,
        "lens": "divergent",
        "lane": eval.lane.as_str(),
        "score": eval.score,
        "scoreBasis": if eval.lane == Lane::Measured { "min(pathMin, 7) x (1 - typedOverlap)" } else { "unmeasured" },
        "pathMin": path_min_json(eval),
        "divergence": eval.divergence,
        "firstNodeType": first_type,
        "secondNodeType": second_type,
        "firstScope": scope_of(&first_record),
        "secondScope": scope_of(&second_record),
        "firstRetention": retention(first),
        "secondRetention": retention(second),
        "firstPreview": first_record.as_ref().map(|r| preview(&r.content)).unwrap_or_default(),
        "secondPreview": second_record.as_ref().map(|r| preview(&r.content)).unwrap_or_default(),
        "sharedTags": [],
        "boundaryTags": [],
        "sharedTerms": [],
        "reason": reason,
        "compositionQuestion": question,
        "proof": proof,
    })
}

fn propose_divergent(memory: &StrataMemory, request: &ProposeRequest) -> Result<Value, String> {
    let store = memory.lock();
    let snapshot = store.ghost_snapshot(filter_of(request));
    let page = snapshot
        .divergent_page(request.cursor.as_deref(), request.limit)
        .map_err(|err| err.to_string())?;
    let head = page.head_seq;
    let as_of = page.retention_as_of_ms;
    let candidates: Vec<Value> = page
        .measured
        .iter()
        .chain(page.juxtaposition.iter())
        .map(|eval| divergent_json(&store, head, as_of, eval))
        .collect();
    let s = &page.summary;
    let mut admission = json!({
        "lens": "divergent",
        "admits": "pool pairs joined by no recorded edge of any kind in either direction (typed, legacy_inferred, anchored_to, anything) and never woven",
        "pathMin": "shortest undirected path over every recorded edge, radius 6; beyond it counts as 7",
        "divergence": "1 - |Nt(a) & Nt(b)| / sqrt(|Nt(a)| * |Nt(b)|), Nt = neighbor ids over touched / derived_from / closed_by edges",
        "lanes": "measured (both typed profiles exist): score = min(pathMin, 7) x divergence, best first. juxtaposition (a profile is empty): no score; a deterministic spread sampler picks pairs, preferring pairs beyond the radius",
        "legacyEdges": "legacy_inferred edges only shorten pathMin: they can dampen a score or remove the one pair they join, never add a candidate or raise a score",
        "pool": pool_json(request, s.pool_size),
        "eligiblePairs": s.eligible_pairs,
        "headSeq": head,
        "neverUses": NEVER_USES,
    });
    if candidates.is_empty() {
        admission["emptyBecause"] = json!(if s.pool_size < 2 {
            "the pool has fewer than two live memories for this scope and tag filter"
        } else if s.eligible_pairs == 0 {
            "every pool pair is already joined by a recorded edge or woven"
        } else {
            "this cursor has walked the whole schedule; omit it to start again"
        });
    }
    Ok(json!({
        "mode": "propose",
        "lens": "divergent",
        "candidates": candidates,
        "laneCounts": {
            "measured": page.measured.len(),
            "juxtaposition": page.juxtaposition.len(),
        },
        "eligiblePairs": {
            "total": s.eligible_pairs,
            "measured": s.measured_eligible_pairs,
            "juxtaposition": s.juxtaposition_eligible_pairs,
        },
        "juxtapositionNote": JUXTAPOSITION_NOTE,
        "summary": {
            "poolSize": s.pool_size,
            "nodesWithTypedProfile": s.nodes_with_typed_profile,
            "isolatedNodes": s.isolated_nodes,
            "legacyEdgesInPool": s.legacy_edges_in_pool,
            "legacyOnlyPairsInPool": s.legacy_only_pairs_in_pool,
            "typedEdgesInPool": s.typed_edges_in_pool,
            "measuredMembersEvaluated": s.measured_members_evaluated,
            "measuredMembersBeyondCap": s.measured_members_beyond_cap,
            "positionsScanned": s.positions_scanned,
        },
        "nextCursor": page.next_cursor,
        "scope": request.scope,
        "includeCrossScope": request.scope.is_none(),
        "headSeq": head,
        "retentionAsOfMs": as_of,
        "admission": admission,
        "evidenceStatus": "hypothesis",
        "globalNoveltyVerified": false,
        "claimBoundary": CLAIM_BOUNDARY,
    }))
}

/// `propose` on a Strata log.
pub fn propose(storage: &Storage, request: &ProposeRequest) -> Result<Value, String> {
    let memory = memory(storage)?;
    propose_with(&memory, request)
}

/// `propose` against an open [`StrataMemory`].
pub fn propose_with(memory: &StrataMemory, request: &ProposeRequest) -> Result<Value, String> {
    match request.lens {
        Lens::Bridge => Ok(propose_bridge(memory, request)),
        Lens::Divergent => propose_divergent(memory, request),
    }
}

// ----------------------------------------------------------------------
// Weave
// ----------------------------------------------------------------------

/// Record a composition outcome for a pair: one composition record through
/// the gated ingest path, then `derived_from` edges from the record to each
/// member. Every write reports its receipt. Re-weaving appends a record.
pub fn weave(
    storage: &Storage,
    first_id: &str,
    second_id: &str,
    outcome_type: &str,
    lens: Option<&str>,
) -> Result<Value, String> {
    weave_with_evidence(storage, first_id, second_id, outcome_type, lens, &[])
}

/// One external finding woven into a pair: where it came from, the sha256 of
/// the content that was fetched, and when. Vestige never fetches the URL; the
/// hash lets anyone who fetches it later prove it is the same content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalEvidence {
    pub url: String,
    pub sha256: String,
    pub retrieved_at_ms: i64,
    pub note: Option<String>,
}

/// Most external findings one weave may carry.
pub const MAX_WEAVE_EVIDENCE: usize = 8;

/// Parse `evidence` from a weave call. Everything is checked before anything
/// is written: http(s) URLs only, a 64-hex sha256, an RFC 3339 `retrievedAt`
/// that is not in the future, and a short optional note.
pub fn parse_evidence(raw: Option<&Value>) -> Result<Vec<ExternalEvidence>, String> {
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };
    let entries = raw
        .as_array()
        .ok_or("evidence must be an array of {url, sha256, retrievedAt, note}")?;
    if entries.len() > MAX_WEAVE_EVIDENCE {
        return Err(format!(
            "at most {MAX_WEAVE_EVIDENCE} evidence entries per weave"
        ));
    }
    let now_ms = Utc::now().timestamp_millis();
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let url = entry
            .get("url")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        let scheme_ok = url.starts_with("https://") || url.starts_with("http://");
        if !scheme_ok || url.len() > 2048 || url.chars().any(char::is_whitespace) {
            return Err(format!(
                "evidence url must be an http(s) URL of at most 2048 characters: {url:?}"
            ));
        }
        let sha256 = entry
            .get("sha256")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!(
                "evidence sha256 must be 64 hex characters (the hash of the fetched content) for {url}"
            ));
        }
        let retrieved = entry
            .get("retrievedAt")
            .and_then(Value::as_str)
            .unwrap_or("");
        let retrieved_at_ms = chrono::DateTime::parse_from_rfc3339(retrieved)
            .map_err(|_| format!("evidence retrievedAt must be an RFC 3339 time for {url}"))?
            .timestamp_millis();
        if retrieved_at_ms > now_ms + 300_000 {
            return Err(format!("evidence retrievedAt is in the future for {url}"));
        }
        let note = entry
            .get("note")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|n| !n.is_empty());
        if note.is_some_and(|n| n.chars().count() > 500) {
            return Err(format!(
                "evidence note must be at most 500 characters for {url}"
            ));
        }
        if out.iter().any(|e: &ExternalEvidence| e.sha256 == sha256) {
            return Err(format!("evidence sha256 {sha256} is listed twice"));
        }
        out.push(ExternalEvidence {
            url: url.to_string(),
            sha256,
            retrieved_at_ms,
            note: note.map(str::to_owned),
        });
    }
    Ok(out)
}

/// [`weave`] carrying external evidence for the pair.
pub fn weave_with_evidence(
    storage: &Storage,
    first_id: &str,
    second_id: &str,
    outcome_type: &str,
    lens: Option<&str>,
    evidence: &[ExternalEvidence],
) -> Result<Value, String> {
    let memory = memory(storage)?;
    weave_with(&memory, first_id, second_id, outcome_type, lens, evidence)
}

/// Exact tag prefix for a woven finding: `evidence:<sha256>`, so `recall`
/// finds the composition record by the content hash alone.
pub const EVIDENCE_TAG_PREFIX: &str = "evidence:";

/// The composition record's text: the pair, the outcome, and one line per
/// external finding.
fn composition_content(
    a: &str,
    b: &str,
    outcome_type: &str,
    lens: &str,
    evidence: &[ExternalEvidence],
) -> String {
    let mut content =
        format!("GhostLink composition of {a} and {b}. Outcome: {outcome_type}. Lens: {lens}.");
    if !evidence.is_empty() {
        content.push_str("\nEvidence:");
        for e in evidence {
            let when = chrono::DateTime::from_timestamp_millis(e.retrieved_at_ms)
                .map(|t| t.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
                .unwrap_or_default();
            content.push_str(&format!(
                "\n- {} sha256:{} retrieved {when}",
                e.url, e.sha256
            ));
            if let Some(note) = &e.note {
                content.push_str(&format!(" ({note})"));
            }
        }
    }
    content
}

fn evidence_json(evidence: &[ExternalEvidence]) -> Value {
    Value::Array(
        evidence
            .iter()
            .map(|e| {
                json!({
                    "url": e.url,
                    "sha256": e.sha256,
                    "retrievedAtMs": e.retrieved_at_ms,
                    "note": e.note,
                    "tag": format!("{EVIDENCE_TAG_PREFIX}{}", e.sha256),
                })
            })
            .collect(),
    )
}

/// [`weave`] against an open [`StrataMemory`].
pub fn weave_with(
    memory: &StrataMemory,
    first_id: &str,
    second_id: &str,
    outcome_type: &str,
    lens: Option<&str>,
    evidence: &[ExternalEvidence],
) -> Result<Value, String> {
    if !OUTCOME_TYPES.contains(&outcome_type) {
        return Err(format!("unsupported outcome_type: {outcome_type}"));
    }
    let lens = match lens.map(str::trim) {
        None | Some("") => "unknown",
        Some("bridge") => "bridge",
        Some("divergent") => "divergent",
        Some(other) => {
            return Err(format!(
                "unknown lens '{other}'; use 'bridge' or 'divergent'"
            ));
        }
    };
    let (first_id, second_id) = (first_id.trim(), second_id.trim());
    if first_id.is_empty() || second_id.is_empty() {
        return Err("weave needs first_id and second_id".into());
    }
    if first_id == second_id {
        return Err("weave needs two distinct memories".into());
    }
    let (a, b) = if first_id <= second_id {
        (first_id, second_id)
    } else {
        (second_id, first_id)
    };
    let mut store = memory.lock();
    for id in [a, b] {
        let record = store
            .get_node(id)
            .filter(retrievable)
            .ok_or_else(|| format!("memory not found or retired: {id}"))?;
        if strata_store::composition_pair(&record).is_some() {
            return Err(format!(
                "{id} is a composition record; weave composes two memories"
            ));
        }
    }
    let scope = store
        .get_node(a)
        .map(|record| record.scope)
        .unwrap_or_else(|| vestige_core::DEFAULT_MEMORY_SCOPE.to_string());
    let now = Utc::now().timestamp_millis();
    let (record_id, record_effect) = store
        .ingest_in_scope_with_receipt(
            strata_store::IngestInput {
                content: composition_content(a, b, outcome_type, lens, evidence),
                source: Some(strata_store::SourceKey {
                    system: strata_store::weave_source(a, b),
                    project: String::new(),
                    id: String::new(),
                }),
                source_updated_at_ms: None,
                node_type: strata_store::COMPOSITION_NODE_TYPE.to_string(),
                tags: [
                    strata_store::GHOSTLINK_TAG.to_string(),
                    strata_store::WEAVE_TAG.to_string(),
                    format!("{}{outcome_type}", strata_store::OUTCOME_TAG_PREFIX),
                    format!("{}{lens}", strata_store::LENS_TAG_PREFIX),
                ]
                .into_iter()
                .chain(
                    evidence
                        .iter()
                        .map(|e| format!("{EVIDENCE_TAG_PREFIX}{}", e.sha256)),
                )
                .collect(),
                created_at_ms: Some(now),
                valid_from_ms: None,
                valid_until_ms: None,
            },
            &scope,
        )
        .map_err(|err| map_store(err).to_string())?;
    let mut receipts = vec![json!({
        "write": "composition_record",
        "id": record_id,
        "receiptId": receipt_id_for(record_effect),
    })];
    for member in [a, b] {
        let edge = strata_store::ConnectionRecord {
            source_id: record_id.clone(),
            target_id: member.to_string(),
            strength_milli: 1000,
            link_type: strata_store::EdgeKind::DerivedFrom.as_str().to_string(),
            meta_sha: None,
            created_at_ms: now,
            activation_count: 0,
        };
        match store.save_connection(&edge) {
            Ok(effect) => receipts.push(json!({
                "write": "derived_from",
                "source": record_id,
                "target": member,
                "receiptId": receipt_id_for(effect),
            })),
            Err(err) => {
                return Err(format!(
                    "weave recorded {record_id} ({}) but its derived_from edge to {member} was not admitted: {}; the record weaves nothing until the pair is woven again",
                    receipt_id_for(record_effect),
                    map_store(err)
                ));
            }
        }
    }
    let snapshot = store.ghost_snapshot(PoolFilter::default());
    let mut pair_records: Vec<&CompositionRecord> = snapshot
        .records()
        .iter()
        .filter(|record| record.complete && record.first_id == a && record.second_id == b)
        .collect();
    newest_first(&mut pair_records);
    let mut pair_outcomes: Vec<String> = pair_records
        .iter()
        .filter_map(|record| record.outcome_type.clone())
        .collect();
    pair_outcomes.sort();
    pair_outcomes.dedup();
    let records_json: Vec<Value> = pair_records
        .iter()
        .map(|record| record_json(&store, record))
        .collect();
    Ok(json!({
        "mode": "weave",
        "decision": "woven",
        "evidence": evidence_json(evidence),
        "nodeId": record_id,
        "recordId": record_id,
        "firstId": a,
        "secondId": b,
        "outcomeType": outcome_type,
        "lens": lens,
        "scope": scope,
        "receipts": receipts,
        "pairRecords": records_json,
        "pairOutcomes": pair_outcomes,
        "outcomeSignal": outcome_signal(&pair_outcomes),
        "headSeq": snapshot.head_seq(),
        "note": "The pair is woven: both lenses now exclude it, and each member's typed profile gains this record through derived_from. Suppress or undo the record to unweave it.",
    }))
}

// ----------------------------------------------------------------------
// Inspect
// ----------------------------------------------------------------------

/// `inspect` on a Strata log: composition records and the members they wove.
pub fn inspect(
    storage: &Storage,
    view: &str,
    event_id: Option<&str>,
    memory_id: Option<&str>,
    limit: usize,
) -> Result<Value, String> {
    let memory = memory(storage)?;
    let store = memory.lock();
    let snapshot = store.ghost_snapshot(PoolFilter::default());
    let records = snapshot.records();
    match view {
        "recent" => {
            let mut all: Vec<&CompositionRecord> = records.iter().collect();
            newest_first(&mut all);
            let events: Vec<Value> = all
                .iter()
                .take(limit)
                .map(|record| record_json(&store, record))
                .collect();
            Ok(json!({ "view": "recent", "events": events, "headSeq": snapshot.head_seq() }))
        }
        "get" => {
            let id = event_id
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .ok_or("event_id (a composition record id) is required for inspect view 'get'")?;
            let record = records
                .iter()
                .find(|record| record.id == id)
                .ok_or_else(|| format!("composition record not found: {id}"))?;
            let members: Vec<Value> = [&record.first_id, &record.second_id]
                .iter()
                .map(|member| {
                    let node = store.get_node(member).filter(retrievable);
                    json!({
                        "memoryId": member,
                        "live": node.is_some(),
                        "nodeType": node.as_ref().map(|r| r.node_type.clone()),
                        "scope": node.as_ref().map(|r| r.scope.clone()),
                        "preview": node.as_ref().map(|r| preview(&r.content)),
                        "edge": "derived_from",
                    })
                })
                .collect();
            Ok(json!({
                "view": "get",
                "event": record_json(&store, record),
                "members": members,
                "outcomes": record.outcome_type.iter().map(|o| json!({ "outcomeType": o })).collect::<Vec<_>>(),
                "headSeq": snapshot.head_seq(),
            }))
        }
        "memory" | "neighbors" => {
            let id = memory_id
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| format!("memory_id is required for inspect view '{view}'"))?;
            let mut involving: Vec<&CompositionRecord> = records
                .iter()
                .filter(|record| record.first_id == id || record.second_id == id)
                .collect();
            newest_first(&mut involving);
            if view == "memory" {
                let events: Vec<Value> = involving
                    .iter()
                    .take(limit)
                    .map(|record| record_json(&store, record))
                    .collect();
                return Ok(json!({
                    "view": "memory",
                    "memoryId": id,
                    "events": events,
                    "headSeq": snapshot.head_seq(),
                }));
            }
            // Memories woven with `id`, from complete records only.
            let mut by_partner: std::collections::BTreeMap<String, PartnerTally> =
                std::collections::BTreeMap::new();
            for record in involving.iter().filter(|record| record.complete) {
                let partner = if record.first_id == id {
                    &record.second_id
                } else {
                    &record.first_id
                };
                let entry = by_partner
                    .entry(partner.clone())
                    .or_insert_with(|| (0, Vec::new(), record.id.clone()));
                entry.0 += 1;
                if let Some(outcome) = &record.outcome_type
                    && !entry.1.contains(outcome)
                {
                    entry.1.push(outcome.clone());
                }
            }
            let mut neighbors: Vec<(String, PartnerTally)> = by_partner.into_iter().collect();
            neighbors.sort_by(|a, b| b.1.0.cmp(&a.1.0).then_with(|| a.0.cmp(&b.0)));
            let neighbors: Vec<Value> = neighbors
                .into_iter()
                .take(limit)
                .map(|(partner, (count, mut outcomes, latest))| {
                    outcomes.sort();
                    json!({
                        "memoryId": partner,
                        "composedCount": count,
                        "outcomes": outcomes,
                        "latestRecordId": latest,
                        "nodeType": node_type_of(&store, &partner),
                    })
                })
                .collect();
            Ok(json!({
                "view": "neighbors",
                "memoryId": id,
                "neighbors": neighbors,
                "headSeq": snapshot.head_seq(),
            }))
        }
        other => Err(format!(
            "Invalid view '{other}' for mode 'inspect'. Allowed: recent, get, memory, neighbors."
        )),
    }
}

/// Per woven partner: composition count, distinct outcomes, latest record id.
type PartnerTally = (usize, Vec<String>, String);

// ----------------------------------------------------------------------
// Bounty
// ----------------------------------------------------------------------

/// `bounty` on a Strata log: lanes from woven records by their exact
/// outcome types, plus the bridge lens as the never-composed lane.
pub fn bounty(storage: &Storage, request: &ProposeRequest) -> Result<Value, String> {
    let memory = memory(storage)?;
    let (lanes, head) = {
        let store = memory.lock();
        let snapshot = store.ghost_snapshot(PoolFilter::default());
        let member_tagged = |record: &CompositionRecord| {
            request.tags.is_empty()
                || [&record.first_id, &record.second_id].iter().any(|member| {
                    store
                        .get_node(member)
                        .is_some_and(|node| node.tags.iter().any(|tag| request.tags.contains(tag)))
                })
        };
        let in_scope = |record: &CompositionRecord| {
            request
                .scope
                .as_deref()
                .is_none_or(|scope| record.scope == scope)
        };
        let mut records: Vec<&CompositionRecord> = snapshot
            .records()
            .iter()
            .filter(|record| record.complete && member_tagged(record) && in_scope(record))
            .collect();
        newest_first(&mut records);
        let lane = |keep: &dyn Fn(&str) -> bool| -> Vec<Value> {
            records
                .iter()
                .filter(|record| record.outcome_type.as_deref().is_some_and(keep))
                .take(request.limit)
                .map(|record| record_json(&store, record))
                .collect()
        };
        let already: Vec<Value> = records
            .iter()
            .take(request.limit)
            .map(|record| record_json(&store, record))
            .collect();
        // Same closed-door set as the SQLite bounty lanes.
        let closed = lane(&|outcome| {
            matches!(
                outcome,
                "dead_end"
                    | "rejected"
                    | "bad_severity"
                    | "closed_by_scope"
                    | "closed_by_duplicate"
                    | "closed_by_false_assumption"
                    | "closed_by_user"
                    | "expired_lane"
            )
        });
        let duplicate =
            lane(&|outcome| matches!(outcome, "duplicate_risk" | "closed_by_duplicate"));
        let needs_poc = lane(&|outcome| outcome == "needs_poc");
        ((already, closed, duplicate, needs_poc), snapshot.head_seq())
    };
    let never = propose_bridge(
        &memory,
        &ProposeRequest {
            lens: Lens::Bridge,
            ..request.clone()
        },
    );
    let never_lanes = never["candidates"].as_array().cloned().unwrap_or_default();
    let top: Vec<Value> = never_lanes.iter().take(3).cloned().collect();
    let (already, closed, duplicate, needs_poc) = lanes;
    Ok(json!({
        "mode": "bounty",
        "alreadyComposedLanes": already,
        "neverComposedLanes": never_lanes,
        "closedDoors": closed,
        "duplicateRiskLanes": duplicate,
        "needsPocLanes": needs_poc,
        "topWeirdCombinations": top,
        "admission": never["admission"],
        "headSeq": head,
        "guardrails": [
            "never-composed lane is not a finding",
            "composition score is not severity",
            "submit/reportable still needs source refs, scope fit, and PoC evidence"
        ],
    }))
}

// ----------------------------------------------------------------------
// Explore (structural)
// ----------------------------------------------------------------------

/// `explore` on a Strata log: recorded structure only. `chain` is the
/// shortest path over typed vocabulary edges, `associations` the recorded
/// typed neighbors, `bridges` the nodes on every shortest typed path.
/// `legacy_inferred` links are not causal hops and are never walked.
pub fn explore(
    storage: &Storage,
    kind: &str,
    from: &str,
    to: Option<&str>,
    limit: usize,
) -> Result<Value, String> {
    let memory = memory(storage)?;
    let store = memory.lock();
    let from = from.trim();
    let live = |id: &str| {
        store
            .get_node(id)
            .is_some_and(|record| retrievable(&record))
    };
    let snapshot = store.ghost_snapshot(PoolFilter::default());
    let describe = |id: &str| -> Value {
        match store.get_node(id).filter(retrievable) {
            Some(record) => json!({
                "memory_id": id,
                "nodeType": record.node_type,
                "memory_preview": preview(&record.content),
            }),
            None => json!({ "memory_id": id, "nodeType": Value::Null, "artifact": true }),
        }
    };
    let basis = "recorded typed vocabulary edges only (legacy_inferred excluded); shortest paths break ties by id";
    match kind {
        "chain" => {
            let to = to
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .ok_or("'to' is required for chain")?;
            let path = if live(from) && live(to) {
                snapshot.recorded_path(from, to, EXPLORE_MAX_HOPS)
            } else {
                None
            };
            match path {
                Some(path) => {
                    // `steps` lists every memory on the chain, origin first,
                    // as the legacy chain contract does. The origin arrives
                    // by no edge; each later step names the recorded edge it
                    // arrived by. `path` carries the edges themselves.
                    // connection_strength is the recorded edge's own strength
                    // and confidence their geometric mean (the legacy
                    // formula), read as written: no score is derived.
                    let strengths: Vec<f64> = path
                        .iter()
                        .map(|step| {
                            let milli = snapshot
                                .recorded_strengths(&step.from)
                                .get(&(step.to.clone(), step.kind.clone()))
                                .copied()
                                .unwrap_or(1000);
                            milli as f64 / 1000.0
                        })
                        .collect();
                    let confidence = if strengths.is_empty() {
                        1.0
                    } else {
                        strengths
                            .iter()
                            .product::<f64>()
                            .powf(1.0 / strengths.len() as f64)
                    };
                    let mut origin = describe(from);
                    origin["connection_type"] = json!("origin");
                    origin["reversed"] = json!(false);
                    origin["connection_strength"] = json!(1.0);
                    origin["reasoning"] = json!("origin of the chain");
                    let steps: Vec<Value> = std::iter::once(origin)
                        .chain(path.iter().zip(&strengths).map(|(step, strength)| {
                            let mut item = describe(&step.to);
                            item["connection_type"] = json!(step.kind);
                            item["reversed"] = json!(step.reversed);
                            item["connection_strength"] = json!(strength);
                            item["reasoning"] = json!(if step.reversed {
                                format!("recorded {} edge, walked from its target", step.kind)
                            } else {
                                format!("recorded {} edge", step.kind)
                            });
                            item
                        }))
                        .collect();
                    Ok(json!({
                        "action": "chain",
                        "from": from,
                        "to": to,
                        "steps": steps,
                        "path": path_json(&path),
                        "total_hops": path.len(),
                        "confidence": confidence,
                        "basis": basis,
                        "headSeq": snapshot.head_seq(),
                    }))
                }
                None => {
                    // `message` is the stable no-chain contract; `reason`
                    // says which way the walk ended.
                    let reason = if !live(from) {
                        format!("'from' {from} is not a live memory")
                    } else if !live(to) {
                        format!("'to' {to} is not a live memory")
                    } else {
                        format!(
                            "no recorded typed path within {EXPLORE_MAX_HOPS} hops between these memories"
                        )
                    };
                    Ok(json!({
                        "action": "chain",
                        "from": from,
                        "to": to,
                        "steps": [],
                        "message": "No chain found between these memories",
                        "reason": reason,
                        "basis": basis,
                        "headSeq": snapshot.head_seq(),
                    }))
                }
            }
        }
        "associations" => {
            let (typed, legacy) = if live(from) {
                let all = snapshot.recorded_neighbors(from, true);
                let legacy = all
                    .iter()
                    .filter(|(_, kind, _)| kind == strata_store::LEGACY_INFERRED)
                    .count();
                (snapshot.recorded_neighbors(from, false), legacy)
            } else {
                (Vec::new(), 0)
            };
            let strengths = snapshot.recorded_strengths(from);
            let associations: Vec<Value> = typed
                .iter()
                .take(limit)
                .map(|(neighbor, kind, forward)| {
                    let mut item = describe(neighbor);
                    let milli = strengths
                        .get(&(neighbor.clone(), kind.clone()))
                        .copied()
                        .unwrap_or(1000);
                    item["strength"] = json!(milli as f64 / 1000.0);
                    item["link_type"] = json!(kind);
                    item["direction"] = json!(if *forward { "outgoing" } else { "incoming" });
                    item["source"] = json!("recorded_edge");
                    item
                })
                .collect();
            Ok(json!({
                "action": "associations",
                "from": from,
                "associations": associations,
                "count": associations.len(),
                "legacyInferredExcluded": legacy,
                "basis": basis,
                "headSeq": snapshot.head_seq(),
            }))
        }
        "bridges" => {
            let to = to
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .ok_or("'to' is required for bridges")?;
            let bridges = if live(from) && live(to) {
                snapshot.recorded_bridges(from, to, EXPLORE_MAX_HOPS)
            } else {
                Vec::new()
            };
            // `bridges` stays the legacy list of memory ids; each described
            // bridge rides along in `bridgeDetails`, in the same order.
            let details: Vec<Value> = bridges
                .iter()
                .take(limit)
                .map(|(id, hops)| {
                    let mut item = describe(id);
                    item["hopsFromSource"] = json!(hops);
                    item
                })
                .collect();
            let ids: Vec<&str> = bridges
                .iter()
                .take(limit)
                .map(|(id, _)| id.as_str())
                .collect();
            Ok(json!({
                "action": "bridges",
                "from": from,
                "to": to,
                "bridges": ids,
                "bridgeDetails": details,
                "count": ids.len(),
                "basis": basis,
                "headSeq": snapshot.head_seq(),
            }))
        }
        other => Err(format!(
            "Invalid kind '{other}' for mode 'explore'. Allowed: chain, associations, bridges."
        )),
    }
}

// ----------------------------------------------------------------------
// Predict (structural)
// ----------------------------------------------------------------------

/// `predict` on a Strata log. `current_topics` is free text, so it is
/// refused with `similarity_disabled`. `current_file` is an exact handle:
/// memories with a code anchor on exactly that path, by retention.
pub fn predict(storage: &Storage, context: Option<&Value>) -> Result<Value, String> {
    let topics = context
        .and_then(|c| c.get("current_topics"))
        .and_then(Value::as_array)
        .is_some_and(|topics| {
            topics
                .iter()
                .any(|topic| topic.as_str().is_some_and(|t| !t.trim().is_empty()))
        });
    if topics {
        return Err(super::similarity("predict current_topics").to_string());
    }
    let memory = memory(storage)?;
    let store = memory.lock();
    let file = context
        .and_then(|c| c.get("current_file"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|file| !file.is_empty());
    let mut predictions: Vec<(f64, String, Value)> = Vec::new();
    if let Some(file) = file {
        for record in store.nodes().into_iter().filter(retrievable) {
            for anchor in store.anchors_for(&record.id) {
                if anchor.file_path != file {
                    continue;
                }
                let retention = store
                    .retrievability(&record.id)
                    .ok()
                    .flatten()
                    .unwrap_or(0.0);
                predictions.push((
                    retention,
                    record.id.clone(),
                    json!({
                        "memory_id": record.id,
                        "content_preview": preview(&record.content),
                        "confidence": retention,
                        "reasoning": "code anchor on current_file (exact path)",
                        "anchorId": anchor.id,
                        "symbol": anchor.symbol,
                    }),
                ));
                break;
            }
        }
    }
    predictions.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let predictions: Vec<Value> = predictions
        .into_iter()
        .take(20)
        .map(|(_, _, v)| v)
        .collect();
    Ok(json!({
        "predictions": predictions,
        "suggestions": [],
        "speculative": [],
        "top_interests": [],
        "prediction_accuracy": Value::Null,
        "predict_degraded": false,
        "basis": "Strata: exact code anchors on context.current_file, ranked by FSRS retention. The learned interest model (query text) is not consulted, and current_topics returns similarity_disabled.",
    }))
}

// ----------------------------------------------------------------------
// Harden
// ----------------------------------------------------------------------

/// Seed invariant laws through the gated ingest path. Idempotent by the
/// live source `ghostlink-harden:<ID>`: a law already present is reported
/// with its memory id and not written again.
pub fn harden(storage: &Storage, seeds: &[HardenSeed]) -> Result<Vec<Value>, String> {
    let memory = memory(storage)?;
    Ok(harden_with(&memory, seeds))
}

/// [`harden`] against an open [`StrataMemory`].
pub fn harden_with(memory: &StrataMemory, seeds: &[HardenSeed]) -> Vec<Value> {
    let mut store = memory.lock();
    let now = Utc::now().timestamp_millis();
    // Live laws by exact source key, first by id. Law ids are unique, so a
    // law seeded in this call cannot be asked for again in it.
    let mut present: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for record in store.nodes().into_iter().filter(retrievable) {
        if let Some(key) = &record.source
            && key.system.starts_with(HARDEN_SOURCE_PREFIX)
            && key.project.is_empty()
            && key.id.is_empty()
        {
            present
                .entry(key.system.clone())
                .or_insert(record.id.clone());
        }
    }
    let mut out = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let source = format!("{HARDEN_SOURCE_PREFIX}{}", seed.law_id);
        if let Some(id) = present.get(&source) {
            out.push(json!({
                "lawId": seed.law_id,
                "name": seed.name,
                "decision": "already_present",
                "id": id,
                "receiptId": store.origin_seq(id).map(receipt_id_for),
            }));
            continue;
        }
        let secrets = blocking_secrets(&seed.content);
        if !secrets.is_empty() {
            out.push(json!({
                "lawId": seed.law_id,
                "name": seed.name,
                "decision": "failed",
                "error": format!("law text looks like a secret ({}); not written", secrets.join(", ")),
            }));
            continue;
        }
        let written = store.ingest_in_scope_with_receipt(
            strata_store::IngestInput {
                content: seed.content.clone(),
                source: Some(strata_store::SourceKey {
                    system: source.clone(),
                    project: String::new(),
                    id: String::new(),
                }),
                source_updated_at_ms: None,
                node_type: "pattern".to_string(),
                tags: seed.tags.clone(),
                created_at_ms: Some(now),
                valid_from_ms: None,
                valid_until_ms: None,
            },
            vestige_core::DEFAULT_MEMORY_SCOPE,
        );
        out.push(match written {
            Ok((id, effect)) => json!({
                "lawId": seed.law_id,
                "name": seed.name,
                "decision": "seeded",
                "id": id,
                "receiptId": receipt_id_for(effect),
            }),
            Err(err) => json!({
                "lawId": seed.law_id,
                "name": seed.name,
                "decision": "failed",
                "error": map_store(err).to_string(),
            }),
        });
    }
    out
}
