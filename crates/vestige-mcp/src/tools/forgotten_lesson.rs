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
//! On a Strata log the anchor join above is not a recorded edge, so it is
//! not used. Cause is the edge source and effect is the target, so the walk
//! follows incoming `corrects`, `derived_from`, and `evidence_of` edges only.
//! Earlier in-scope memories come back ranked by the store's existing FSRS
//! retrievability, lowest first, each with the edge path that reached it.
//! Artifact edges (`touched`, `anchored_to`, `projected_to`) are not causes.
//! The legacy SQLite body is unchanged and is not the stdio backend.

use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use vestige_core::advanced::retroactive_backfill::extract_entities;
use vestige_core::fsrs::retrievability;
use vestige_core::{ConnectionRecord, KnowledgeNode, Storage};

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

pub fn schema() -> Value {
    json!({
        "type": "object",
        "description": "Fix/lesson memories decayed below R 0.5 at failure time.",
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
                "description": "Max lesson-candidate memories to scan."
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

/// Hop cap for the recorded-edge walk. A node this many causal hops from the
/// failure is kept; the next hop is not expanded.
const RECORDED_WALK_MAX_DEPTH: usize = 8;

/// Response cap after ranking. The walk may see more; the lowest
/// retrievability wins the seats.
const RECORDED_RESULT_CAP: usize = 20;

/// One recorded causal edge on the path from the failure to a lesson.
#[derive(Clone)]
struct RecordedHop {
    source_id: String,
    target_id: String,
    link_type: String,
}

struct WalkHit {
    lesson_id: String,
    recorded_at: chrono::DateTime<chrono::Utc>,
    retrievability: f64,
    path: Vec<RecordedHop>,
    preview: String,
}

struct WalkFrame {
    id: String,
    depth: usize,
    path: Vec<RecordedHop>,
}

/// Recorded causal edges. The source is the cause; the target is the effect.
/// Artifact anchors (`touched`, `anchored_to`, `projected_to`) are not causes.
fn is_causal_link(link_type: &str) -> bool {
    matches!(link_type, "corrects" | "derived_from" | "evidence_of")
}

/// Source of an incoming causal edge. Walking only incoming edges is the
/// backward step: from an effect back to the recorded cause.
fn causal_predecessor(from: &str, edge: &ConnectionRecord) -> Option<String> {
    if !is_causal_link(&edge.link_type) || edge.target_id != from {
        return None;
    }
    if edge.source_id.is_empty() || edge.source_id == from {
        return None;
    }
    Some(edge.source_id.clone())
}

/// Bounded backward walk from `failure` over recorded causal edges.
///
/// Every earlier in-scope node reached by an incoming `corrects`,
/// `derived_from`, or `evidence_of` edge is a hit. The caller ranks those
/// hits by existing FSRS retrievability (`KnowledgeNode::retrieval_strength`,
/// the kernel value projected by the store), lowest first. No content, tag,
/// or embedding signal is read.
fn walk_recorded_causes(
    storage: &Arc<Storage>,
    failure: &KnowledgeNode,
    scope: &str,
    scan_limit: usize,
) -> Result<(Vec<WalkHit>, usize), String> {
    let mut hits = Vec::new();
    let mut seen = HashSet::new();
    seen.insert(failure.id.clone());
    let mut queue = VecDeque::from([WalkFrame {
        id: failure.id.clone(),
        depth: 0,
        path: Vec::new(),
    }]);
    let mut scanned = 0usize;

    while let Some(frame) = queue.pop_front() {
        if scanned >= scan_limit {
            break;
        }
        scanned += 1;
        if frame.depth >= RECORDED_WALK_MAX_DEPTH {
            continue;
        }
        let edges = storage
            .get_connections_for_memory(&frame.id)
            .map_err(|e| e.to_string())?;
        let mut steps: Vec<(String, RecordedHop)> = Vec::new();
        for edge in &edges {
            let Some(next) = causal_predecessor(&frame.id, edge) else {
                continue;
            };
            if seen.contains(&next) {
                continue;
            }
            steps.push((
                next,
                RecordedHop {
                    source_id: edge.source_id.clone(),
                    target_id: edge.target_id.clone(),
                    link_type: edge.link_type.clone(),
                },
            ));
        }
        steps.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then(left.1.link_type.cmp(&right.1.link_type))
                .then(left.1.source_id.cmp(&right.1.source_id))
                .then(left.1.target_id.cmp(&right.1.target_id))
        });
        for (next, hop) in steps {
            if !seen.insert(next.clone()) {
                continue;
            }
            let Some(node) = storage.get_node(&next).map_err(|e| e.to_string())? else {
                continue;
            };
            if !storage
                .node_is_in_scope(&next, scope)
                .map_err(|e| e.to_string())?
            {
                continue;
            }
            if node.created_at >= failure.created_at {
                continue;
            }
            let mut path = frame.path.clone();
            path.push(hop);
            hits.push(WalkHit {
                lesson_id: node.id.clone(),
                recorded_at: node.created_at,
                retrievability: node.retrieval_strength,
                path: path.clone(),
                preview: node.content.chars().take(140).collect(),
            });
            queue.push_back(WalkFrame {
                id: node.id,
                depth: frame.depth + 1,
                path,
            });
        }
    }

    hits.sort_by(|left, right| {
        left.retrievability
            .total_cmp(&right.retrievability)
            .then_with(|| left.lesson_id.cmp(&right.lesson_id))
    });
    hits.truncate(RECORDED_RESULT_CAP);
    Ok((hits, scanned))
}

fn forgotten_from_recorded_edges(
    storage: &Arc<Storage>,
    args: Option<Value>,
) -> Result<Value, String> {
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

    let (found, scanned) = walk_recorded_causes(
        storage,
        &failure,
        scope,
        usize::try_from(scan_limit).unwrap_or(1000),
    )?;
    let lessons: Vec<Value> = found
        .iter()
        .map(|hit| {
            json!({
                "lesson_id": hit.lesson_id,
                "recorded_at": hit.recorded_at.to_rfc3339(),
                "retention_pct": (hit.retrievability * 1000.0).round() / 10.0,
                "retrievability": hit.retrievability,
                "edge_path": hit.path.iter().map(|hop| json!({
                    "source_id": hop.source_id,
                    "target_id": hop.target_id,
                    "link_type": hop.link_type,
                })).collect::<Vec<_>>(),
                "content_preview": hit.preview,
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
        "note": "Backward walk of recorded causal edges (corrects, derived_from, evidence_of). Ranked by existing FSRS retrievability, lowest first. The edge is the record.",
    }))
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        return forgotten_from_recorded_edges(storage, args);
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
mod recorded_edge_tests {
    use super::*;

    struct Graph {
        failure: String,
        low: String,
        high: String,
        mid: String,
        twin: String,
        bystander: String,
        forward: String,
        later: String,
        other: String,
        low_r: f64,
        high_r: f64,
    }

    fn mem(content: &str, at: i64) -> strata_store::IngestInput {
        strata_store::IngestInput {
            content: content.to_string(),
            created_at_ms: Some(at),
            ..Default::default()
        }
    }

    fn link(src: &str, dst: &str, kind: strata_store::EdgeKind) -> strata_store::ConnectionRecord {
        strata_store::ConnectionRecord {
            source_id: src.to_string(),
            target_id: dst.to_string(),
            link_type: kind.as_str().to_string(),
            created_at_ms: 1,
            ..Default::default()
        }
    }

    fn kernel_r(store: &strata_store::StrataStore, id: &str) -> f64 {
        store
            .retrievability(id)
            .expect("retrievability")
            .expect("card")
    }

    /// Easy-rate the later lesson, Again-rate the earlier one, then advance
    /// the log until the Again card's existing retrievability is lower.
    fn separate_retrievability(store: &mut strata_store::StrataStore, low: &str, high: &str) {
        store.review(high, 4).expect("easy review");
        for step in 0..8 {
            store
                .ingest_in_scope(mem(&format!("pre-pad {step}"), 8_000 + step), "user")
                .expect("pre pad");
        }
        store.review(low, 1).expect("again review");
        for step in 1..=80 {
            let low_r = kernel_r(store, low);
            let high_r = kernel_r(store, high);
            if low_r < high_r {
                return;
            }
            assert!(
                step < 80,
                "retrievability did not separate: low={low_r} high={high_r}"
            );
            store
                .ingest_in_scope(mem(&format!("pad {step}"), 9_000 + step), "user")
                .expect("pad");
        }
    }

    fn seed_graph(dir: &std::path::Path) -> Graph {
        let mut store = strata_store::StrataStore::open(dir).expect("open strata");
        let low = store
            .ingest_in_scope(mem("again-rated cause", 1_000), "user")
            .unwrap();
        let twin = store
            .ingest_in_scope(
                mem("fixed the API_TIMEOUT overflow in the same outage", 1_100),
                "user",
            )
            .unwrap();
        let other = store
            .ingest_in_scope(mem("other-scope cause", 1_200), "other")
            .unwrap();
        let bystander = store
            .ingest_in_scope(mem("file anchor only", 1_300), "user")
            .unwrap();
        let forward = store
            .ingest_in_scope(mem("downstream of the failure", 1_400), "user")
            .unwrap();
        let mid = store
            .ingest_in_scope(mem("intermediate cause", 2_000), "user")
            .unwrap();
        let high = store
            .ingest_in_scope(mem("easy-rated cause", 3_000), "user")
            .unwrap();
        let failure = store
            .ingest_in_scope(mem("outage after API_TIMEOUT change", 5_000), "user")
            .unwrap();
        let later = store
            .ingest_in_scope(mem("later note", 9_500), "user")
            .unwrap();

        store
            .save_connection(&link(&mid, &failure, strata_store::EdgeKind::DerivedFrom))
            .unwrap();
        store
            .save_connection(&link(&low, &mid, strata_store::EdgeKind::Corrects))
            .unwrap();
        store
            .save_connection(&link(&high, &failure, strata_store::EdgeKind::EvidenceOf))
            .unwrap();
        store
            .save_connection(&link(&bystander, &failure, strata_store::EdgeKind::Touched))
            .unwrap();
        store
            .save_connection(&link(&later, &failure, strata_store::EdgeKind::DerivedFrom))
            .unwrap();
        store
            .save_connection(&link(&other, &failure, strata_store::EdgeKind::DerivedFrom))
            .unwrap();
        store
            .save_connection(&link(
                &failure,
                &forward,
                strata_store::EdgeKind::DerivedFrom,
            ))
            .unwrap();

        separate_retrievability(&mut store, &low, &high);
        let low_r = kernel_r(&store, &low);
        let high_r = kernel_r(&store, &high);
        drop(store);
        Graph {
            failure,
            low,
            high,
            mid,
            twin,
            bystander,
            forward,
            later,
            other,
            low_r,
            high_r,
        }
    }

    fn lesson<'a>(out: &'a Value, id: &str) -> &'a Value {
        out["forgotten_lessons"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["lesson_id"] == id)
            .unwrap_or_else(|| panic!("missing {id} in {out}"))
    }

    #[tokio::test]
    async fn recorded_edges_rank_forgotten_causes_and_ignore_overlap() {
        let dir = tempfile::tempdir().unwrap();
        let graph = seed_graph(dir.path());
        let storage = crate::strata_memory::open(dir.path()).unwrap();

        let out = execute(&storage, Some(json!({"failure_id": graph.failure})))
            .await
            .unwrap();
        let again = execute(&storage, Some(json!({"failure_id": graph.failure})))
            .await
            .unwrap();
        assert_eq!(out, again, "the walk is deterministic");
        assert_eq!(out["tool"], "forgotten_lesson");
        assert!(!out.to_string().contains("pending_strata"));

        let lessons = out["forgotten_lessons"].as_array().unwrap();
        let ids: Vec<&str> = lessons
            .iter()
            .map(|entry| entry["lesson_id"].as_str().unwrap())
            .collect();
        for banned in [
            graph.twin.as_str(),
            graph.bystander.as_str(),
            graph.forward.as_str(),
            graph.later.as_str(),
            graph.other.as_str(),
            graph.failure.as_str(),
        ] {
            assert!(
                !ids.contains(&banned),
                "{banned} is not a recorded causal predecessor"
            );
        }
        let low_at = ids.iter().position(|id| *id == graph.low).expect("low");
        let high_at = ids.iter().position(|id| *id == graph.high).expect("high");
        assert!(
            low_at < high_at,
            "lowest retrievability ranks first: {ids:?}"
        );

        let reported: Vec<f64> = lessons
            .iter()
            .map(|entry| entry["retrievability"].as_f64().unwrap())
            .collect();
        for window in reported.windows(2) {
            assert!(
                window[0] <= window[1],
                "not ranked lowest-first: {reported:?}"
            );
        }
        let low = lesson(&out, &graph.low);
        let high = lesson(&out, &graph.high);
        assert!((low["retrievability"].as_f64().unwrap() - graph.low_r).abs() < 1e-9);
        assert!((high["retrievability"].as_f64().unwrap() - graph.high_r).abs() < 1e-9);
        assert!(graph.low_r < graph.high_r);

        let low_path = low["edge_path"].as_array().unwrap();
        assert_eq!(low_path.len(), 2);
        assert_eq!(low_path[0]["link_type"], "derived_from");
        assert_eq!(low_path[0]["source_id"], graph.mid);
        assert_eq!(low_path[0]["target_id"], graph.failure);
        assert_eq!(low_path[1]["link_type"], "corrects");
        assert_eq!(low_path[1]["source_id"], graph.low);
        assert_eq!(low_path[1]["target_id"], graph.mid);
        let high_path = high["edge_path"].as_array().unwrap();
        assert_eq!(high_path.len(), 1);
        assert_eq!(high_path[0]["link_type"], "evidence_of");
        assert_eq!(high_path[0]["source_id"], graph.high);
        assert_eq!(high_path[0]["target_id"], graph.failure);
        assert_eq!(low["content_preview"], "again-rated cause");

        let missing = execute(
            &storage,
            Some(json!({"failure_id": "mem-0000000000000000"})),
        )
        .await
        .unwrap_err();
        assert!(missing.contains("not found"), "{missing}");
        let empty = execute(
            &storage,
            Some(json!({"failure_id": graph.failure, "scope": " "})),
        )
        .await
        .unwrap_err();
        assert!(empty.contains("empty"), "{empty}");
    }

    #[tokio::test]
    async fn walk_stops_at_the_depth_bound() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = strata_store::StrataStore::open(dir.path()).unwrap();
        let hops = RECORDED_WALK_MAX_DEPTH + 1;
        let mut nodes = Vec::with_capacity(hops);
        for index in 0..hops {
            nodes.push(
                store
                    .ingest_in_scope(mem(&format!("hop {index}"), 1_000 + index as i64), "user")
                    .unwrap(),
            );
        }
        let failure = store
            .ingest_in_scope(mem("bounded failure", 5_000), "user")
            .unwrap();
        // nodes[0] is the far end. nodes[hops-1] points at the failure.
        store
            .save_connection(&link(
                &nodes[hops - 1],
                &failure,
                strata_store::EdgeKind::DerivedFrom,
            ))
            .unwrap();
        for index in 0..hops - 1 {
            store
                .save_connection(&link(
                    &nodes[index],
                    &nodes[index + 1],
                    strata_store::EdgeKind::DerivedFrom,
                ))
                .unwrap();
        }
        let reachable = nodes[1].clone();
        let too_far = nodes[0].clone();
        drop(store);

        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let out = execute(&storage, Some(json!({"failure_id": failure})))
            .await
            .unwrap();
        let ids: Vec<&str> = out["forgotten_lessons"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["lesson_id"].as_str().unwrap())
            .collect();
        assert!(
            ids.contains(&reachable.as_str()),
            "depth {RECORDED_WALK_MAX_DEPTH} is in range: {ids:?}"
        );
        assert!(
            !ids.contains(&too_far.as_str()),
            "one hop past the cap is excluded: {ids:?}"
        );
        let reached = lesson(&out, &reachable);
        assert_eq!(
            reached["edge_path"].as_array().unwrap().len(),
            RECORDED_WALK_MAX_DEPTH
        );
        assert_eq!(reached["edge_path"][0]["target_id"], failure);
        assert_eq!(
            reached["edge_path"][RECORDED_WALK_MAX_DEPTH - 1]["source_id"],
            reachable
        );
    }

    #[tokio::test]
    async fn stdio_returns_the_recorded_lesson() {
        let dir = tempfile::tempdir().unwrap();
        let graph = seed_graph(dir.path());
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let cognitive = Arc::new(tokio::sync::Mutex::new(
            crate::cognitive::CognitiveEngine::new(),
        ));
        let server = crate::server::McpServer::new(storage, cognitive);
        let (mut client_w, server_r) = tokio::io::duplex(1 << 16);
        let (server_w, mut client_r) = tokio::io::duplex(1 << 20);
        let handle = tokio::spawn(async move {
            crate::protocol::stdio::run_io(
                server,
                None,
                tokio::io::BufReader::new(server_r),
                server_w,
            )
            .await
        });
        let input = format!(
            "{}\n{}\n{}\n",
            json!({
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "forgotten-lesson", "version": "1"}
                }
            }),
            json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "forgotten_lesson",
                    "arguments": {"failure_id": graph.failure}
                }
            })
        );
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        client_w.write_all(input.as_bytes()).await.unwrap();
        drop(client_w);
        let mut buf = String::new();
        tokio::time::timeout(
            std::time::Duration::from_secs(60),
            client_r.read_to_string(&mut buf),
        )
        .await
        .expect("stdio answer")
        .unwrap();
        handle.await.unwrap().unwrap();

        let lines: Vec<Value> = buf
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("json line"))
            .collect();
        let response = lines
            .iter()
            .find(|line| line["id"] == json!(1))
            .unwrap_or_else(|| panic!("no tools/call response: {lines:?}"));
        assert!(response.get("error").is_none(), "{response}");
        let body = &response["result"];
        assert_eq!(body["isError"], json!(false), "{body}");
        let text = body["content"][0]["text"].as_str().unwrap_or("");
        assert!(!text.contains("pending_strata"), "{text}");
        let structured = &body["structuredContent"];
        assert_eq!(structured["tool"], "forgotten_lesson");
        let ids: Vec<&str> = structured["forgotten_lessons"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|lesson| lesson["lesson_id"].as_str())
            .collect();
        assert!(ids.contains(&graph.low.as_str()), "{structured}");
        assert!(!ids.contains(&graph.twin.as_str()), "{structured}");
        assert!(!structured.to_string().contains("pending_strata"));
    }
}
