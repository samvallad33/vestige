//! `forgotten_lesson` MCP tool — decayed corrective-memory detection.
//!
//! Given a failure memory, find earlier FIX/LESSON memories that
//!
//! 1. are causally relevant: they share an exact code, path, env, or version
//!    entity with the failure, or the log recorded an edge between them (or
//!    to the same endpoint), and
//! 2. had FSRS-6 retrievability BELOW 0.5 AT FAILURE TIME, computed with
//!    [`vestige_core::fsrs::retrievability`] from the node's stored stability
//!    and the elapsed days between `last_accessed` and the failure.
//!
//! Prose-word overlap is not relevance. Read-only: no graph, strength, or
//! FSRS state is modified.

use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::sync::Arc;

use vestige_core::advanced::retroactive_backfill::{
    IdentifierTier, extract_entities, normalized_tier,
};
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

/// Code, path, env, and version entities only. A bare prose word is not a cause.
fn is_causal_entity(entity: &str) -> bool {
    normalized_tier(entity) != IdentifierTier::Word
}

/// Recorded edges that tie `lesson_id` to `failure_id`: a direct edge's
/// `link_type`, or an endpoint both edges already name.
fn causal_anchors(lesson_id: &str, failure_id: &str, edges: &[ConnectionRecord]) -> Vec<String> {
    let mut direct = Vec::new();
    let mut lesson_ends = HashSet::new();
    let mut failure_ends = HashSet::new();
    for edge in edges {
        let direct_hit = (edge.source_id == lesson_id && edge.target_id == failure_id)
            || (edge.source_id == failure_id && edge.target_id == lesson_id);
        if direct_hit {
            direct.push(edge.link_type.clone());
        }
        if edge.source_id == lesson_id {
            lesson_ends.insert(edge.target_id.clone());
        } else if edge.target_id == lesson_id {
            lesson_ends.insert(edge.source_id.clone());
        }
        if edge.source_id == failure_id {
            failure_ends.insert(edge.target_id.clone());
        } else if edge.target_id == failure_id {
            failure_ends.insert(edge.source_id.clone());
        }
    }
    lesson_ends.remove(failure_id);
    failure_ends.remove(lesson_id);
    let mut shared: Vec<String> = lesson_ends.intersection(&failure_ends).cloned().collect();
    shared.append(&mut direct);
    shared.sort();
    shared.dedup();
    shared
}

/// Pure detection. `links` are recorded-edge anchors (link type or shared
/// endpoint) and count even when no entity overlaps.
pub fn detect_lesson(
    node: &KnowledgeNode,
    failure_anchors: &HashSet<String>,
    failure_at: chrono::DateTime<chrono::Utc>,
) -> Option<ForgottenLesson> {
    detect_lesson_with_links(node, failure_anchors, failure_at, &[])
}

fn detect_lesson_with_links(
    node: &KnowledgeNode,
    failure_anchors: &HashSet<String>,
    failure_at: chrono::DateTime<chrono::Utc>,
    links: &[String],
) -> Option<ForgottenLesson> {
    if !lesson_tagged(&node.tags) && !has_fix_marker(&node.content) {
        return None;
    }
    let node_anchors: HashSet<String> = extract_entities(&node.content, &node.tags)
        .into_iter()
        .filter(|entity| is_causal_entity(entity))
        .collect();
    let mut shared: Vec<String> = failure_anchors
        .iter()
        .filter(|entity| is_causal_entity(entity) && node_anchors.contains(*entity))
        .cloned()
        .collect();
    for link in links {
        if !shared.iter().any(|anchor| anchor == link) {
            shared.push(link.clone());
        }
    }
    if shared.is_empty() {
        return None;
    }
    shared.sort();
    // Stored stability + elapsed days since last_accessed, exact FSRS-6 curve.
    // Non-positive elapsed returns 1.0 (accessed at or after the failure).
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

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
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
        .filter(|entity| is_causal_entity(entity))
        .collect();
    let edges = storage.get_all_connections().map_err(|e| e.to_string())?;

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
            let links = causal_anchors(&node.id, &failure.id, &edges);
            if let Some(lesson) =
                detect_lesson_with_links(node, &failure_anchors, failure.created_at, &links)
            {
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
        "note": "Each entry is an older lesson with FSRS-6 retrievability < 0.5 at failure time (stored stability, elapsed since last_accessed) and either an exact code/path/env/version entity or a recorded edge. A decayed corrective memory, not a proven cause.",
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
mod strata_tests {
    use super::*;
    use strata_store::{ConnectionRecord, EdgeKind, IngestInput, StrataStore};

    fn no_sqlite(dir: &std::path::Path) -> bool {
        let mut stack = vec![dir.to_path_buf()];
        while let Some(path) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&path) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                if name.ends_with(".sqlite")
                    || name.ends_with(".sqlite3")
                    || name.ends_with(".db")
                    || name.ends_with(".db-wal")
                    || name.ends_with(".db-shm")
                {
                    return false;
                }
                if path.is_dir() {
                    stack.push(path);
                }
            }
        }
        true
    }

    fn input(content: &str, tags: &[&str], created_at_ms: i64) -> IngestInput {
        IngestInput {
            content: content.to_string(),
            tags: tags.iter().map(|tag| tag.to_string()).collect(),
            created_at_ms: Some(created_at_ms),
            ..IngestInput::default()
        }
    }

    fn digest(dir: &std::path::Path) -> [u8; 32] {
        StrataStore::open(dir).unwrap().state_digest()
    }

    /// Decayed lesson sharing `API_TIMEOUT` is returned. A decayed lesson on
    /// another entity is not. A fresh lesson on the same entity is not. A
    /// decayed lesson with no shared entity but a recorded `corrects` edge is.
    /// The blake3 state digest is unchanged, and no SQLite file appears.
    #[tokio::test]
    async fn surfaces_decayed_shared_entity_not_unrelated() {
        let dir = tempfile::TempDir::new().unwrap();
        let failure_ms = 1_750_000_000_000_i64;
        let day_ms = 86_400_000_i64;
        let decayed_id;
        let unrelated_id;
        let linked_id;
        let fresh_id;
        let failure_id;
        {
            let mut store = StrataStore::open(dir.path()).unwrap();
            decayed_id = store
                .ingest_in_scope(
                    input(
                        "Fixed the API_TIMEOUT overflow in src/auth/login.rs",
                        &["fix"],
                        failure_ms - 500 * day_ms,
                    ),
                    "user",
                )
                .unwrap();
            // Shares the prose word "crash" with the failure and nothing else.
            unrelated_id = store
                .ingest_in_scope(
                    input(
                        "Fixed the crash after the billing close",
                        &["fix"],
                        failure_ms - 500 * day_ms,
                    ),
                    "user",
                )
                .unwrap();
            linked_id = store
                .ingest_in_scope(
                    input(
                        "Fixed the ledger drift in the nightly close",
                        &["fix"],
                        failure_ms - 500 * day_ms,
                    ),
                    "user",
                )
                .unwrap();
            fresh_id = store
                .ingest_in_scope(
                    input(
                        "Fixed the API_TIMEOUT overflow in src/auth/login.rs again",
                        &["fix"],
                        failure_ms - day_ms,
                    ),
                    "user",
                )
                .unwrap();
            failure_id = store
                .ingest_in_scope(
                    input(
                        "Outage: crash loop in src/auth/login.rs after API_TIMEOUT change",
                        &["incident"],
                        failure_ms,
                    ),
                    "user",
                )
                .unwrap();
            store
                .save_connection(&ConnectionRecord {
                    source_id: linked_id.clone(),
                    target_id: failure_id.clone(),
                    link_type: EdgeKind::Corrects.as_str().to_string(),
                    created_at_ms: failure_ms,
                    ..ConnectionRecord::default()
                })
                .unwrap();
        }
        // Open once so a later reopen is comparable. state_digest is blake3.
        drop(crate::strata_memory::open(dir.path()).unwrap());
        let before = digest(dir.path());

        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let out = execute(&storage, Some(json!({"failure_id": failure_id})))
            .await
            .unwrap();
        drop(storage);

        let after = digest(dir.path());
        assert_eq!(before, after, "forgotten_lesson must not write the log");
        assert!(no_sqlite(dir.path()));

        let ids: Vec<&str> = out["forgotten_lessons"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["lesson_id"].as_str().unwrap())
            .collect();
        assert!(ids.contains(&decayed_id.as_str()), "{out}");
        assert!(ids.contains(&linked_id.as_str()), "{out}");
        assert!(!ids.contains(&unrelated_id.as_str()), "{out}");
        assert!(!ids.contains(&fresh_id.as_str()), "{out}");
        assert_eq!(out["count"], json!(2), "{out}");

        let decayed = out["forgotten_lessons"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["lesson_id"] == decayed_id)
            .unwrap();
        assert!(
            decayed["shared_anchors"]
                .as_array()
                .unwrap()
                .iter()
                .any(|anchor| anchor == "api_timeout"),
            "{decayed}"
        );
        let retention = decayed["retention_pct"].as_f64().unwrap();
        assert!(
            (0.0..50.0).contains(&retention),
            "retention must be below 50%, got {retention}"
        );
        let linked = out["forgotten_lessons"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["lesson_id"] == linked_id)
            .unwrap();
        assert!(
            linked["shared_anchors"]
                .as_array()
                .unwrap()
                .iter()
                .any(|anchor| anchor == "corrects"),
            "{linked}"
        );
    }

    #[tokio::test]
    async fn rejects_invalid_input_on_strata() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let missing = execute(&storage, Some(json!({}))).await.unwrap_err();
        assert!(missing.contains("failure_id is required"), "{missing}");
        let blank = execute(&storage, Some(json!({"failure_id": " "})))
            .await
            .unwrap_err();
        assert!(blank.contains("failure_id is required"), "{blank}");
        let scope = execute(
            &storage,
            Some(json!({"failure_id": "mem-0000000000000001", "scope": " "})),
        )
        .await
        .unwrap_err();
        assert!(scope.contains("scope must not be empty"), "{scope}");
        let unknown = execute(
            &storage,
            Some(json!({"failure_id": "mem-0000000000000001"})),
        )
        .await
        .unwrap_err();
        assert!(unknown.contains("failure memory not found"), "{unknown}");
        assert!(no_sqlite(dir.path()));
    }
}
