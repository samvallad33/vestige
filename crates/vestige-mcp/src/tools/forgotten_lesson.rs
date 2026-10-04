//! `forgotten_lesson` MCP tool — decayed corrective-memory detection.
//!
//! Given a failure memory, find earlier FIX/LESSON memories that
//!
//! 1. share at least ONE exact anchor with the failure (the same
//!    [`retroactive_backfill::extract_entities`] join key the backward reach
//!    uses — an env var, file path, or code identifier, matched by exact
//!    normalized string equality, never by similarity), and
//! 2. had FSRS-6 retrievability BELOW 0.5 AT FAILURE TIME, computed with the
//!    crate's own forgetting curve from the node's STORED stability and the
//!    time elapsed between `last_accessed` and the failure.
//!
//! That combination is a lesson the system once recorded but can no longer
//! retrieve: the exact condition under which the same root cause bites twice.
//! Read-only: no graph, strength, or FSRS state is modified.
//!
//! On a Strata log the entity-overlap scan is not a recorded edge, so it is
//! not used. The log path walks backward from the failure along recorded
//! causal edges only (`corrects`, `derived_from`, `evidence_of`, `closed_by`;
//! cause → effect) and returns earlier nodes ranked by the store's existing
//! FSRS retrievability, lowest first, each with the edge path that reached it.

use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use vestige_core::KnowledgeNode;
use vestige_core::Storage;
use vestige_core::advanced::retroactive_backfill::extract_entities;
use vestige_core::fsrs::retrievability;

/// Retrievability at failure time below this marks the lesson as forgotten.
/// 0.5 is the midpoint of the FSRS probability-of-recall scale.
pub const FORGOTTEN_THRESHOLD: f64 = 0.5;

/// Exact tag tokens (case-insensitive equality, whole tag) that mark a memory
/// as a recorded lesson/fix. Not substring matches: a `prefix-fix` tag is a
/// different tag.
const LESSON_TAGS: &[&str] = &["fix", "fixed", "lesson", "lessons", "resolved", "learned"];

/// Exact past-tense fix markers searched in content with word boundaries, so
/// "fixed" never matches "unfixed" or "prefix-fixed".
const FIX_MARKERS: &[&str] = &[
    "fixed",
    "resolved",
    "solved",
    "patched",
    "mitigated",
    "repaired",
    "reverted",
];

/// Whole-word marker match on already-lowercased text (same boundary rule as
/// the core `contains_marker_word`: alphanumeric or `_` on either side means
/// the match is embedded in a larger identifier and does not count).
fn contains_marker_word(hay: &str, marker: &str) -> bool {
    let mut from = 0usize;
    while let Some(pos) = hay[from..].find(marker) {
        let start = from + pos;
        let end = start + marker.len();
        let before_ok = hay[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_alphanumeric() || c == '_'));
        let after_ok = hay[end..]
            .chars()
            .next()
            .is_none_or(|c| !(c.is_alphanumeric() || c == '_'));
        if before_ok && after_ok {
            return true;
        }
        from = start + 1;
    }
    false
}

/// Does this tag set carry an exact lesson/fix tag?
pub fn lesson_tagged(tags: &[String]) -> bool {
    tags.iter()
        .any(|t| LESSON_TAGS.contains(&t.trim().to_lowercase().as_str()))
}

/// Does this content carry an exact past-tense fix marker (whole word)?
pub fn has_fix_marker(content: &str) -> bool {
    let hay = content.to_lowercase();
    FIX_MARKERS.iter().any(|m| contains_marker_word(&hay, m))
}

/// One detected forgotten lesson.
#[derive(Debug, Clone, PartialEq)]
pub struct ForgottenLesson {
    pub lesson_id: String,
    /// When the lesson was recorded (`created_at`).
    pub recorded_at: chrono::DateTime<chrono::Utc>,
    /// FSRS retrievability at failure time (0.0..1.0, below the threshold).
    pub retention: f64,
    /// The first exact anchor shared with the failure.
    pub shared_anchor: String,
    /// All exact anchors shared with the failure.
    pub shared_anchors: Vec<String>,
}

/// Pure detection: is this node a lesson that (a) shares at least one exact
/// anchor with the failure's anchor set and (b) was below the retrievability
/// threshold at `failure_at`? Exposed so tests and future callers share ONE
/// detection rule (same discipline as `backfill::looks_like_failure`).
pub fn detect_lesson(
    node: &KnowledgeNode,
    failure_anchors: &HashSet<String>,
    failure_at: chrono::DateTime<chrono::Utc>,
) -> Option<ForgottenLesson> {
    // Lesson-shaped: exact lesson tag OR exact past-tense fix marker.
    if !lesson_tagged(&node.tags) && !has_fix_marker(&node.content) {
        return None;
    }
    // Anchor-set intersection: >= 1 EXACT anchor (normalized entity string).
    let node_anchors: HashSet<String> = extract_entities(&node.content, &node.tags)
        .into_iter()
        .collect();
    let mut shared: Vec<String> = failure_anchors
        .intersection(&node_anchors)
        .cloned()
        .collect();
    if shared.is_empty() {
        return None;
    }
    shared.sort();
    // FSRS retrievability at failure time: stored stability + elapsed since
    // last_accessed, through the crate's own forgetting curve. The curve
    // returns 1.0 for non-positive elapsed (accessed at/after the failure),
    // which correctly reads as "was retrievable then" and never flags.
    let elapsed_days = (failure_at - node.last_accessed).num_seconds() as f64 / 86_400.0;
    let retention = retrievability(node.stability, elapsed_days);
    if retention >= FORGOTTEN_THRESHOLD {
        return None;
    }
    Some(ForgottenLesson {
        lesson_id: node.id.clone(),
        recorded_at: node.created_at,
        retention,
        shared_anchor: shared[0].clone(),
        shared_anchors: shared,
    })
}

/// Recorded cause → effect edges. `touched` and `anchored_to` point at
/// files, not causes. Projections and supersession are not causal either.
const CAUSAL_EDGE_TYPES: &[&str] = &["closed_by", "corrects", "derived_from", "evidence_of"];

/// Hop cap for the backward walk. The failure is depth 0.
const CAUSAL_WALK_MAX_DEPTH: u32 = 8;

/// How many ranked lessons the tool returns.
const MAX_LESSONS: usize = 20;

#[derive(Clone)]
struct CausalStep {
    source_id: String,
    target_id: String,
    link_type: String,
}

struct WalkHit {
    node: KnowledgeNode,
    retention: f64,
    path: Vec<CausalStep>,
}

fn is_causal_edge(link_type: &str) -> bool {
    CAUSAL_EDGE_TYPES.contains(&link_type)
}

/// A node recorded before the failure. Equal timestamps break on id, which
/// is the log order for `mem-` ids.
fn is_earlier(node: &KnowledgeNode, failure: &KnowledgeNode) -> bool {
    match node.created_at.cmp(&failure.created_at) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Equal => node.id < failure.id,
        std::cmp::Ordering::Greater => false,
    }
}

/// Bounded BFS from `failure` against the reverse of recorded causal edges.
///
/// Edges are stored cause → effect (`source` caused `target`). Backward means
/// `target == current`, predecessor `source`. Expansion order is
/// `(link_type, source_id, target_id)`. The first path wins. Nodes outside
/// `scope`, dangling targets, and non-causal edges are skipped. Only nodes
/// earlier than the failure are hits, and the walk does not continue through
/// a later node.
fn walk_recorded_causes(
    storage: &Storage,
    failure: &KnowledgeNode,
    scope: &str,
    scan_limit: usize,
) -> Result<(Vec<WalkHit>, usize), String> {
    struct Frame {
        id: String,
        depth: u32,
        path: Vec<CausalStep>,
    }

    let mut queue = VecDeque::from([Frame {
        id: failure.id.clone(),
        depth: 0,
        path: Vec::new(),
    }]);
    let mut seen = HashSet::from([failure.id.clone()]);
    // The failure itself consumes one slot of the scan budget.
    let mut scanned = 1usize;
    let mut hits = Vec::new();

    while let Some(frame) = queue.pop_front() {
        if scanned >= scan_limit || frame.depth >= CAUSAL_WALK_MAX_DEPTH {
            continue;
        }

        // Each causal edge with the memory at its earlier end. An edge is
        // stored the way its writer names it: `A derived_from B` runs from
        // the derivative to its origin, so the earlier record is the target;
        // every other causal kind runs from the earlier record to the later.
        let mut edges: Vec<(String, vestige_core::ConnectionRecord)> = storage
            .get_connections_for_memory(&frame.id)
            .map_err(|e| e.to_string())?
            .into_iter()
            .filter_map(|edge| {
                if edge.source_id == edge.target_id || !is_causal_edge(&edge.link_type) {
                    return None;
                }
                let earlier = if edge.link_type == "derived_from" {
                    (edge.source_id == frame.id).then(|| edge.target_id.clone())?
                } else {
                    (edge.target_id == frame.id).then(|| edge.source_id.clone())?
                };
                Some((earlier, edge))
            })
            .collect();
        edges.sort_by(|a, b| a.1.link_type.cmp(&b.1.link_type).then(a.0.cmp(&b.0)));

        for (earlier, edge) in edges {
            if !seen.insert(earlier.clone()) {
                continue;
            }
            let Some(node) = storage.get_node(&earlier).map_err(|e| e.to_string())? else {
                continue;
            };
            if !storage
                .node_is_in_scope(&node.id, scope)
                .map_err(|e| e.to_string())?
            {
                continue;
            }
            if scanned >= scan_limit {
                break;
            }
            scanned += 1;
            let mut path = Vec::with_capacity(frame.path.len() + 1);
            path.push(CausalStep {
                source_id: edge.source_id.clone(),
                target_id: edge.target_id.clone(),
                link_type: edge.link_type.clone(),
            });
            path.extend(frame.path.iter().cloned());
            // A later node is not a cause of this failure, and the walk does
            // not continue through it.
            if !is_earlier(&node, failure) {
                continue;
            }
            let id = node.id.clone();
            hits.push(WalkHit {
                retention: node.retrieval_strength,
                node,
                path: path.clone(),
            });
            queue.push_back(Frame {
                id,
                depth: frame.depth + 1,
                path,
            });
        }
    }

    hits.sort_by(|a, b| {
        a.retention
            .total_cmp(&b.retention)
            .then_with(|| a.node.id.cmp(&b.node.id))
    });
    hits.truncate(MAX_LESSONS);
    Ok((hits, scanned))
}

fn recorded_causes(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: Args = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| e.to_string())?,
        None => Args::default(),
    };
    let scan_limit =
        usize::try_from(args.scan_limit.unwrap_or(1000).clamp(10, 5000)).unwrap_or(1000);
    let scope = args.scope.as_deref().unwrap_or("user").trim();
    if scope.is_empty() {
        return Err("scope must not be empty".into());
    }
    let failure_id = args
        .failure_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| "failure_id is required".to_string())?;

    if !storage
        .node_is_in_scope(failure_id, scope)
        .map_err(|e| e.to_string())?
    {
        return Err("failure memory not found in requested scope".into());
    }
    let failure = storage
        .get_node(failure_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "failure memory not found in requested scope".to_string())?;

    let (hits, scanned) = walk_recorded_causes(storage.as_ref(), &failure, scope, scan_limit)?;
    let lessons: Vec<Value> = hits
        .iter()
        .map(|hit| {
            json!({
                "lesson_id": hit.node.id,
                "recorded_at": hit.node.created_at.to_rfc3339(),
                "retention_pct": (hit.retention * 1000.0).round() / 10.0,
                "edge_path": hit.path.iter().map(|step| json!({
                    "source_id": step.source_id,
                    "target_id": step.target_id,
                    "link_type": step.link_type,
                })).collect::<Vec<_>>(),
                "content_preview": hit.node.content.chars().take(140).collect::<String>(),
            })
        })
        .collect();

    Ok(json!({
        "tool": "forgotten_lesson",
        "failure_id": failure.id,
        "scope": scope,
        "threshold": FORGOTTEN_THRESHOLD,
        "scanned": scanned,
        "count": lessons.len(),
        "forgotten_lessons": lessons,
        "note": "Backward walk over recorded causal edges only: from a memory to what it is derived_from, and to the records that are evidence_of it, that correct it, or that it closed. Ranked by existing FSRS retrievability, lowest first. No entity overlap, keyword or FTS search, embeddings, or inferred edges.",
    }))
}

pub fn schema() -> Value {
    json!({
        "type": "object",
        "description": "Recorded causal edges from a failure, lowest FSRS R first. Legacy engine only: exact-anchor fix/lesson memories with R below 0.5.",
        "properties": {
            "failure_id": {
                "type": "string",
                "description": "Failure memory to inspect."
            },
            "scope": {
                "type": "string",
                "default": "user",
                "description": "Exact project namespace of the failure. Defaults to user."
            },
            "scan_limit": {
                "type": "integer",
                "minimum": 10,
                "maximum": 5000,
                "default": 1000,
                "description": "Max memories scanned in the walk."
            }
        },
        "required": ["failure_id"]
    })
}

#[derive(Deserialize, Default)]
struct Args {
    failure_id: Option<String>,
    scope: Option<String>,
    scan_limit: Option<i32>,
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        return recorded_causes(storage, args);
    }
    let args: Args = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| e.to_string())?,
        None => Args::default(),
    };
    let scan_limit = args.scan_limit.unwrap_or(1000).clamp(10, 5000);

    let scope = args.scope.as_deref().unwrap_or("user").trim();
    if scope.is_empty() {
        return Err("scope must not be empty".into());
    }
    let failure_id = args
        .failure_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| "failure_id is required".to_string())?;

    // Resolve the failure within the requested namespace (same contract as
    // the backfill tool: a foreign-scope id is an error, not a silent empty).
    if !storage
        .node_is_in_scope(failure_id, scope)
        .map_err(|e| e.to_string())?
    {
        return Err("failure memory not found in requested scope".into());
    }
    let failure = storage
        .get_node(failure_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "failure memory not found in requested scope".to_string())?;

    let failure_anchors: HashSet<String> = extract_entities(&failure.content, &failure.tags)
        .into_iter()
        .collect();

    // Scan the namespace in bounded pages; only records OLDER than the
    // failure can be forgotten lessons for it.
    let mut detected: Vec<ForgottenLesson> = Vec::new();
    let mut scanned = 0usize;
    let mut offset = 0;
    'scan: loop {
        let page = storage
            .get_all_nodes_in_scope(scope, 500, offset)
            .map_err(|e| e.to_string())?;
        let page_len = page.len();
        for node in &page {
            scanned += 1;
            if node.id == failure.id || node.created_at >= failure.created_at {
                continue;
            }
            if let Some(lesson) = detect_lesson(node, &failure_anchors, failure.created_at) {
                detected.push(lesson);
            }
        }
        if page_len < 500 || scanned >= scan_limit as usize {
            break 'scan;
        }
        offset += 500;
    }

    // Most-forgotten first.
    detected.sort_by(|a, b| {
        a.retention
            .partial_cmp(&b.retention)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    detected.truncate(20);

    let lessons: Vec<Value> = detected
        .iter()
        .map(|l| {
            let node = storage.get_node(&l.lesson_id).ok().flatten();
            json!({
                "lesson_id": l.lesson_id,
                "recorded_at": l.recorded_at.to_rfc3339(),
                "retention_pct": (l.retention * 1000.0).round() / 10.0,
                "shared_anchor": l.shared_anchor,
                "shared_anchors": l.shared_anchors,
                "content_preview": node
                    .map(|n| n.content.chars().take(140).collect::<String>())
                    .unwrap_or_default(),
            })
        })
        .collect();

    Ok(json!({
        "tool": "forgotten_lesson",
        "failure_id": failure.id,
        "scope": scope,
        "threshold": FORGOTTEN_THRESHOLD,
        "scanned": scanned,
        "count": lessons.len(),
        "forgotten_lessons": lessons,
        "note": "Each entry shares >=1 exact anchor with the failure and had FSRS retrievability < 0.5 at failure time (stored stability, elapsed since last_accessed). A decayed corrective memory, not a proven cause.",
    }))
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use vestige_core::IngestInput;

    fn test_storage() -> (Arc<Storage>, TempDir, std::path::PathBuf) {
        let dir = TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let storage = vestige_core::open_storage(Some(db.clone())).unwrap();
        (storage, dir, db)
    }

    /// Fabricate FSRS decay directly in the seeded store: the storage API has
    /// no "set stability/last_accessed to arbitrary values" knob (by design),
    /// and the values this tool reads live in the `knowledge_nodes` row. A
    /// second SQLite connection to the same temp file is the same trick the
    /// upgrade dry-run uses for snapshots, in write mode.
    fn fabricate_fsrs(
        db: &std::path::Path,
        id: &str,
        stability: f64,
        last_accessed: chrono::DateTime<chrono::Utc>,
    ) {
        let conn = rusqlite::Connection::open(db).unwrap();
        let changed = conn
            .execute(
                "UPDATE knowledge_nodes SET stability = ?1, last_accessed = ?2 WHERE id = ?3",
                rusqlite::params![stability, last_accessed.to_rfc3339(), id],
            )
            .unwrap();
        assert_eq!(changed, 1, "fabrication must hit the seeded row");
    }

    fn ingest(storage: &Arc<Storage>, content: &str, tags: &[&str]) -> vestige_core::KnowledgeNode {
        storage
            .ingest(IngestInput {
                content: content.to_string(),
                tags: tags.iter().map(|t| t.to_string()).collect(),
                ..Default::default()
            })
            .unwrap()
    }

    #[test]
    fn schema_is_valid_and_requires_failure_id() {
        let s = schema();
        assert_eq!(s["type"], "object");
        assert_eq!(s["required"][0], "failure_id");
    }

    #[test]
    fn marker_matching_is_whole_word_not_substring() {
        assert!(has_fix_marker("the leak was fixed yesterday"));
        assert!(has_fix_marker("Resolved: pager duty quiet"));
        // embedded in a larger identifier (alphanumeric on either side): no match.
        // A hyphen IS a boundary — same rule as the core marker matcher.
        assert!(!has_fix_marker("the autofixed widget"));
        assert!(has_fix_marker("the prefix-fixed widget"));
        assert!(!has_fix_marker("nothing happened here"));
        assert!(lesson_tagged(&["Fix".into(), "deploy".into()]));
        assert!(!lesson_tagged(&["prefix-fix".into(), "deploy".into()]));
    }

    /// Seeded store with a fabricated DECAYED stability: the lesson is
    /// detected with the shared env-var anchor and a retention below 50%.
    #[tokio::test]
    async fn detects_decayed_lesson_in_seeded_store() {
        let (storage, _dir, db) = test_storage();

        let decayed = ingest(
            &storage,
            "Fixed the API_TIMEOUT overflow in src/auth/login.rs",
            &["fix"],
        );
        storage
            .set_created_at(&decayed.id, chrono::Utc::now() - chrono::Duration::days(10))
            .unwrap();
        // stability 0.5d, last touched 60d before the failure lands.
        fabricate_fsrs(
            &db,
            &decayed.id,
            0.5,
            chrono::Utc::now() - chrono::Duration::days(60),
        );

        // Fresh, well-reviewed lesson sharing the SAME anchors: must NOT flag.
        let fresh = ingest(
            &storage,
            "Fixed the API_TIMEOUT overflow in src/auth/login.rs (verified)",
            &["fix"],
        );
        storage
            .set_created_at(&fresh.id, chrono::Utc::now() - chrono::Duration::days(10))
            .unwrap();
        fabricate_fsrs(&db, &fresh.id, 100.0, chrono::Utc::now());

        // Decayed but NOT a lesson (no tag, no fix marker): must NOT flag.
        let noise = ingest(
            &storage,
            "Noted API_TIMEOUT behaviour in src/auth/login.rs",
            &["observation"],
        );
        storage
            .set_created_at(&noise.id, chrono::Utc::now() - chrono::Duration::days(10))
            .unwrap();
        fabricate_fsrs(
            &db,
            &noise.id,
            0.5,
            chrono::Utc::now() - chrono::Duration::days(60),
        );

        let failure = ingest(
            &storage,
            "Outage: crash loop in src/auth/login.rs after API_TIMEOUT change",
            &["incident"],
        );

        let out = execute(&storage, Some(json!({"failure_id": failure.id})))
            .await
            .unwrap();
        assert_eq!(out["count"], json!(1), "exactly the decayed lesson: {out}");
        let entry = &out["forgotten_lessons"][0];
        assert_eq!(entry["lesson_id"], json!(decayed.id));
        assert_eq!(entry["shared_anchor"], json!("api_timeout"));
        let retention = entry["retention_pct"].as_f64().unwrap();
        assert!(
            (0.0..50.0).contains(&retention),
            "retention must be below the 50% threshold, got {retention}"
        );

        // Scope + validation contract mirrors backfill.
        assert!(
            execute(
                &storage,
                Some(json!({"failure_id": failure.id, "scope": " "}))
            )
            .await
            .is_err()
        );
        assert!(
            execute(&storage, Some(json!({}))).await.is_err(),
            "failure_id is required"
        );
    }

    /// A lesson recorded AFTER the failure is never a forgotten lesson for it.
    #[tokio::test]
    async fn newer_than_failure_records_are_ignored() {
        let (storage, _dir, db) = test_storage();
        let failure = ingest(&storage, "Outage: crash in billing", &["incident"]);
        let later = ingest(&storage, "Fixed the billing crash for good", &["fix"]);
        fabricate_fsrs(
            &db,
            &later.id,
            0.5,
            chrono::Utc::now() - chrono::Duration::days(120),
        );
        let out = execute(&storage, Some(json!({"failure_id": failure.id})))
            .await
            .unwrap();
        assert_eq!(out["count"], json!(0));
    }

    /// Property: FSRS retrievability strictly decreases as elapsed time since
    /// last access grows, for every stability — so "older access => more
    /// forgotten" holds across the whole curve the tool reads.
    #[test]
    fn retention_decreases_with_elapsed_time() {
        let ladder = [1.0_f64, 10.0, 60.0, 365.0, 1000.0];
        for stability in [0.5_f64, 1.0, 5.0, 20.0, 100.0] {
            let mut prev = f64::INFINITY;
            for (i, &elapsed) in ladder.iter().enumerate() {
                let r = retrievability(stability, elapsed);
                assert!(
                    (0.0..=1.0).contains(&r),
                    "retrievability must stay in [0,1]: s={stability} t={elapsed}"
                );
                if i > 0 {
                    assert!(
                        r < prev,
                        "retention must decrease with elapsed time: s={stability} {prev} !> {r}"
                    );
                }
                prev = r;
            }
        }
        // And through the tool's own detection rule: the SAME lesson row
        // flips from not-forgotten (fresh access) to forgotten (long-stale
        // access). Stability 5d: R ~ 1.0 at ~0 elapsed, R ~ 0.44 at 730d.
        let anchors: HashSet<String> = ["api_timeout".to_string()].into_iter().collect();
        let at = chrono::Utc::now();
        let mut node = KnowledgeNode::default();
        node.content = "Fixed the api_timeout overflow".to_string();
        node.tags = vec!["fix".to_string()];
        node.stability = 5.0;
        node.created_at = at - chrono::Duration::days(3650);
        let mut fresh = node.clone();
        fresh.last_accessed = at;
        assert!(detect_lesson(&fresh, &anchors, at).is_none());
        let mut stale = node;
        stale.last_accessed = at - chrono::Duration::days(730);
        let detected = detect_lesson(&stale, &anchors, at).expect("stale lesson detected");
        assert!(detected.retention < FORGOTTEN_THRESHOLD);
        assert_eq!(detected.shared_anchor, "api_timeout");
    }
}

#[cfg(test)]
mod strata_walk_tests {
    use super::*;
    use tempfile::TempDir;
    use vestige_core::IngestInput;

    fn open() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        (storage, dir)
    }

    fn ingest(storage: &Arc<Storage>, content: &str) -> KnowledgeNode {
        storage
            .ingest(IngestInput {
                content: content.to_string(),
                ..Default::default()
            })
            .unwrap()
    }

    fn link(storage: &Arc<Storage>, source: &str, target: &str, link_type: &str) {
        let now = chrono::Utc::now();
        storage
            .save_connection(&vestige_core::ConnectionRecord {
                source_id: source.to_string(),
                target_id: target.to_string(),
                strength: 1.0,
                link_type: link_type.to_string(),
                created_at: now,
                last_activated: now,
                activation_count: 0,
            })
            .unwrap();
    }

    #[tokio::test]
    async fn walks_recorded_edges_and_ignores_overlap_and_noncausal_edges() {
        let (storage, dir) = open();
        let lesson = ingest(&storage, "Recorded correction for the pool ceiling");
        for i in 0..6 {
            ingest(&storage, &format!("filler {i}"));
        }
        let corrected = ingest(&storage, "Earlier correction of the pool ceiling");
        let touched = ingest(&storage, "Commit that touched the pool ceiling");
        let decoy = ingest(
            &storage,
            "Recorded correction for the pool ceiling, unlinked copy",
        );
        let cause = ingest(&storage, "Prior adjustment of the pool ceiling");
        let failure = ingest(&storage, "Outage crash loop after the pool ceiling change");
        let forward_only = ingest(&storage, "Later note that the failure points at");
        link(&storage, &failure.id, &cause.id, "derived_from");
        link(&storage, &lesson.id, &cause.id, "evidence_of");
        link(&storage, &corrected.id, &failure.id, "corrects");
        link(&storage, &touched.id, &failure.id, "touched");
        link(&storage, &decoy.id, &failure.id, "anchored_to");
        // Forward edge: failure is the source. The walk must not follow it.
        link(&storage, &forward_only.id, &failure.id, "derived_from");
        drop(storage);

        {
            let mut store = strata_store::StrataStore::open(dir.path()).unwrap();
            // Easy reviews reset the older cards' review clocks and raise
            // their stability. The untouched cause then has the lower
            // retrievability even though `corrected` was ingested first, so
            // the rank is the FSRS score and not ingest order.
            for _ in 0..8 {
                store.review(&lesson.id, 4).unwrap();
            }
            store.review(&corrected.id, 4).unwrap();
            // The untouched cause was written a month ago and has decayed
            // with elapsed time.
            store
                .set_created_at(
                    &cause.id,
                    chrono::Utc::now().timestamp_millis() - 30 * 86_400_000,
                )
                .unwrap();
        }
        let storage = crate::strata_memory::open(dir.path()).unwrap();

        let out = execute(&storage, Some(json!({"failure_id": failure.id})))
            .await
            .unwrap();
        let again = execute(&storage, Some(json!({"failure_id": failure.id})))
            .await
            .unwrap();
        assert_eq!(out, again, "the walk is deterministic");
        let body = serde_json::to_string(&out).unwrap();
        assert!(!body.contains("pending_strata"), "{out}");
        assert!(!body.contains("shared_anchor"), "{out}");

        let lessons = out["forgotten_lessons"].as_array().unwrap();
        let ids: Vec<&str> = lessons
            .iter()
            .map(|entry| entry["lesson_id"].as_str().unwrap())
            .collect();
        assert!(ids.contains(&lesson.id.as_str()), "{out}");
        assert!(ids.contains(&cause.id.as_str()), "{out}");
        assert!(ids.contains(&corrected.id.as_str()), "{out}");
        assert!(
            !ids.contains(&touched.id.as_str()),
            "touched is not a cause edge: {out}"
        );
        assert!(
            !ids.contains(&decoy.id.as_str()),
            "anchored_to is not causal: {out}"
        );
        assert!(
            !ids.contains(&forward_only.id.as_str()),
            "forward edges are not a backward walk: {out}"
        );

        let cause_r = lessons
            .iter()
            .find(|entry| entry["lesson_id"] == cause.id)
            .unwrap()["retention_pct"]
            .as_f64()
            .unwrap();
        let lesson_r = lessons
            .iter()
            .find(|entry| entry["lesson_id"] == lesson.id)
            .unwrap()["retention_pct"]
            .as_f64()
            .unwrap();
        assert!(
            cause_r < lesson_r,
            "lapsed cause must rank as more forgotten: cause {cause_r} lesson {lesson_r}"
        );
        assert_eq!(lessons[0]["lesson_id"], json!(cause.id), "{out}");

        let cause_path = &lessons.iter().find(|e| e["lesson_id"] == cause.id).unwrap()["edge_path"];
        assert_eq!(cause_path.as_array().unwrap().len(), 1, "{cause_path}");
        assert_eq!(cause_path[0]["link_type"], "derived_from");
        // recorded as `failure derived_from cause`
        assert_eq!(cause_path[0]["source_id"], json!(failure.id));
        assert_eq!(cause_path[0]["target_id"], json!(cause.id));

        let lesson_path = &lessons
            .iter()
            .find(|e| e["lesson_id"] == lesson.id)
            .unwrap()["edge_path"];
        assert_eq!(lesson_path.as_array().unwrap().len(), 2, "{lesson_path}");
        assert_eq!(lesson_path[0]["source_id"], json!(lesson.id));
        assert_eq!(lesson_path[0]["target_id"], json!(cause.id));
        assert_eq!(lesson_path[0]["link_type"], "evidence_of");
        assert_eq!(lesson_path[1]["source_id"], json!(failure.id));

        assert!(
            execute(
                &storage,
                Some(json!({"failure_id": failure.id, "scope": " "}))
            )
            .await
            .is_err()
        );
        assert!(execute(&storage, Some(json!({}))).await.is_err());
    }

    #[tokio::test]
    async fn depth_and_scan_bounds_stop_the_walk() {
        let (storage, _dir) = open();
        let mut ids = Vec::new();
        for i in 0..12 {
            ids.push(ingest(&storage, &format!("cause {i}")).id);
        }
        let failure = ingest(&storage, "the failure");
        // ids[0] is earliest. Chain ids[11] → … → ids[0] → failure would put
        // the earliest node deepest. Link each earlier node as the cause of
        // the next newer one, then the newest cause to the failure.
        for pair in ids.windows(2) {
            link(&storage, &pair[0], &pair[1], "closed_by");
        }
        link(&storage, ids.last().unwrap(), &failure.id, "closed_by");

        let out = execute(
            &storage,
            Some(json!({"failure_id": failure.id, "scan_limit": 10})),
        )
        .await
        .unwrap();
        let returned: Vec<&str> = out["forgotten_lessons"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["lesson_id"].as_str().unwrap())
            .collect();
        assert!(
            returned.contains(&ids[11].as_str()),
            "the direct cause is in range: {out}"
        );
        assert!(
            !returned.contains(&ids[0].as_str()),
            "the far end of a 12-hop chain is past depth {CAUSAL_WALK_MAX_DEPTH}: {out}"
        );
        assert!(out["scanned"].as_u64().unwrap() <= 10, "{out}");
    }

    #[tokio::test]
    async fn scan_limit_stops_a_wide_backward_frontier() {
        let (storage, _dir) = open();
        let mut causes = Vec::new();
        for i in 0..15 {
            causes.push(ingest(&storage, &format!("direct cause {i}")).id);
        }
        let failure = ingest(&storage, "wide failure");
        for id in &causes {
            link(&storage, &failure.id, id, "derived_from");
        }
        let out = execute(
            &storage,
            Some(json!({"failure_id": failure.id, "scan_limit": 10})),
        )
        .await
        .unwrap();
        let returned = out["forgotten_lessons"].as_array().unwrap();
        assert_eq!(out["scanned"], json!(10), "{out}");
        assert_eq!(
            returned.len(),
            9,
            "failure plus nine causes, then stop: {out}"
        );
        let ids: Vec<&str> = returned
            .iter()
            .map(|entry| entry["lesson_id"].as_str().unwrap())
            .collect();
        assert!(ids.contains(&causes[0].as_str()), "{out}");
        assert!(
            !ids.contains(&causes[14].as_str()),
            "the last cause is past the scan budget: {out}"
        );
    }
}
