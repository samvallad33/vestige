//! `forgotten_lesson` MCP tool — decayed corrective-memory detection.
//!
//! Given a failure memory, find earlier FIX/LESSON memories that
//!
//! 1. the log joined to the failure with a recorded edge of type `touched`,
//!    `anchored_to`, `derived_from`, `supersedes`, `corrects`, `closed_by`,
//!    `projected_to`, or `evidence_of`, and
//! 2. had FSRS-6 retrievability BELOW 0.5 AT FAILURE TIME, computed with
//!    [`vestige_core::fsrs::retrievability`] from the node's stored stability
//!    and the elapsed days since the card's last review.
//!
//! Name overlap is not relevance. `legacy_inferred` is not a cause. On a
//! Strata log the decay clock is `created_at_ms` of the upsert frame that
//! folded the last review. An explicit review frame has a log sequence and
//! no timestamp; those lessons are counted and not scored. Read-only.

use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;

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
    /// The first recorded link type tying this lesson to the failure.
    pub shared_anchor: String,
    /// Recorded link types tying this lesson to the failure.
    pub shared_anchors: Vec<String>,
}

/// Owner vocabulary, plus an explicit skip. The live edge has no
/// `legacy_inferred` field; a migrated inferred link is stored as
/// `derived_from` and cannot be told apart from a declared one.
fn is_recorded_cause(link_type: &str) -> bool {
    link_type != "legacy_inferred" && strata_store::TYPED_EDGE_VOCABULARY.contains(&link_type)
}

fn other_end<'a>(edge: &'a ConnectionRecord, node: &str) -> Option<&'a str> {
    if edge.source_id == node {
        Some(edge.target_id.as_str())
    } else if edge.target_id == node {
        Some(edge.source_id.as_str())
    } else {
        None
    }
}

/// Link types of recorded edges that tie `lesson_id` to `failure_id`.
///
/// A direct edge contributes its `link_type`. Two allowed edges that name the
/// same third endpoint contribute their link types. Endpoint ids are not
/// anchors: a path or symbol stored as an endpoint is causal structure only
/// when both edges were recorded, and it is not echoed.
fn causal_link_types(lesson_id: &str, failure_id: &str, edges: &[ConnectionRecord]) -> Vec<String> {
    let mut types = Vec::new();
    let mut lesson_ends: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut failure_ends: HashMap<&str, Vec<&str>> = HashMap::new();
    for edge in edges {
        if !is_recorded_cause(&edge.link_type) {
            continue;
        }
        let lesson_side = other_end(edge, lesson_id);
        let failure_side = other_end(edge, failure_id);
        if lesson_side.is_some() && failure_side.is_some() {
            types.push(edge.link_type.as_str());
            continue;
        }
        if let Some(end) = lesson_side.filter(|end| *end != lesson_id && *end != failure_id) {
            lesson_ends
                .entry(end)
                .or_default()
                .push(edge.link_type.as_str());
        }
        if let Some(end) = failure_side.filter(|end| *end != failure_id && *end != lesson_id) {
            failure_ends
                .entry(end)
                .or_default()
                .push(edge.link_type.as_str());
        }
    }
    for (end, lesson_types) in &lesson_ends {
        if let Some(failure_types) = failure_ends.get(end) {
            types.extend(lesson_types.iter().copied());
            types.extend(failure_types.iter().copied());
        }
    }
    let mut owned: Vec<String> = types.into_iter().map(str::to_string).collect();
    owned.sort();
    owned.dedup();
    owned
}

/// Pure detection. `link_types` are recorded edges. `reviewed_at` is the
/// wall-clock time of the card's last review. Callers that have no timestamp
/// for that review must not call this with `created_at` as a stand-in.
pub fn detect_lesson(
    node: &KnowledgeNode,
    link_types: &[String],
    failure_at: chrono::DateTime<chrono::Utc>,
    reviewed_at: chrono::DateTime<chrono::Utc>,
) -> Option<ForgottenLesson> {
    if link_types.is_empty() || (!lesson_tagged(&node.tags) && !has_fix_marker(&node.content)) {
        return None;
    }
    // Stored stability + elapsed days since the last review, exact FSRS-6.
    // Non-positive elapsed returns 1.0 (reviewed at or after the failure).
    let elapsed_days = (failure_at - reviewed_at).num_seconds() as f64 / 86_400.0;
    let retention = retrievability(node.stability, elapsed_days);
    if retention >= FORGOTTEN_THRESHOLD {
        return None;
    }
    let mut shared = link_types.to_vec();
    shared.sort();
    shared.dedup();
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

fn review_instant(
    storage: &Storage,
    node: &KnowledgeNode,
) -> Option<chrono::DateTime<chrono::Utc>> {
    match crate::strata_memory::lesson_review_clock(storage, &node.id) {
        crate::strata_memory::LessonClock::NotStrata => Some(node.last_accessed),
        crate::strata_memory::LessonClock::Mapped(at) => Some(at),
        crate::strata_memory::LessonClock::Unmapped => None,
    }
}

pub async fn execute(
    storage: &std::sync::Arc<Storage>,
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

    let edges = storage.get_all_connections().map_err(|e| e.to_string())?;

    // Scan the namespace in bounded pages; only records OLDER than the
    // failure can be forgotten lessons for it.
    let mut detected: Vec<ForgottenLesson> = Vec::new();
    let mut unmapped_reviews = 0usize;
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
            let links = causal_link_types(&node.id, &failure.id, &edges);
            if links.is_empty() || (!lesson_tagged(&node.tags) && !has_fix_marker(&node.content)) {
                continue;
            }
            let Some(reviewed_at) = review_instant(storage.as_ref(), node) else {
                unmapped_reviews += 1;
                continue;
            };
            if let Some(lesson) = detect_lesson(node, &links, failure.created_at, reviewed_at) {
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
        "unmapped_reviews": unmapped_reviews,
        "forgotten_lessons": lessons,
        "note": "Each entry is an older lesson joined by a recorded edge (touched, anchored_to, derived_from, supersedes, corrects, closed_by, projected_to, evidence_of) with FSRS-6 retrievability < 0.5 at failure time. The decay clock is created_at_ms on the upsert frame of the card's last review. An explicit review frame stores a log sequence and no timestamp, so those lessons are counted in unmapped_reviews and are not scored. Name overlap is not a cause. The live edge has no legacy_inferred field; that link type is skipped, and a migrated inferred edge stored as derived_from cannot be told apart.",
    }))
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use vestige_core::IngestInput;

    fn test_storage() -> (std::sync::Arc<Storage>, TempDir, std::path::PathBuf) {
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

    fn ingest(storage: &std::sync::Arc<Storage>, content: &str, tags: &[&str]) -> KnowledgeNode {
        storage
            .ingest(IngestInput {
                content: content.to_string(),
                tags: tags.iter().map(|t| t.to_string()).collect(),
                ..Default::default()
            })
            .unwrap()
    }

    fn link(storage: &std::sync::Arc<Storage>, source: &str, target: &str, link_type: &str) {
        let now = chrono::Utc::now();
        storage
            .save_connection(&ConnectionRecord {
                source_id: source.to_string(),
                target_id: target.to_string(),
                strength: 1.0,
                link_type: link_type.to_string(),
                created_at: now,
                last_activated: now,
                activation_count: 1,
            })
            .unwrap();
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
    /// detected through a recorded `corrects` edge. A decayed lesson that
    /// only shares a name, with no edge, is not.
    #[tokio::test]
    async fn detects_decayed_lesson_in_seeded_store() {
        let (storage, _dir, db) = test_storage();

        let decayed = ingest(
            &storage,
            "Patched the ledger drift during the nightly close",
            &["fix"],
        );
        storage
            .set_created_at(&decayed.id, chrono::Utc::now() - chrono::Duration::days(10))
            .unwrap();
        fabricate_fsrs(
            &db,
            &decayed.id,
            0.5,
            chrono::Utc::now() - chrono::Duration::days(60),
        );

        // Shares API_TIMEOUT and the login path with the failure. No edge.
        let named = ingest(
            &storage,
            "Fixed the API_TIMEOUT overflow in src/auth/login.rs",
            &["fix"],
        );
        storage
            .set_created_at(&named.id, chrono::Utc::now() - chrono::Duration::days(10))
            .unwrap();
        fabricate_fsrs(
            &db,
            &named.id,
            0.5,
            chrono::Utc::now() - chrono::Duration::days(60),
        );

        // Fresh, well-reviewed lesson on the same edge: must NOT flag.
        let fresh = ingest(
            &storage,
            "Repaired the journal checksum before dawn",
            &["fix"],
        );
        storage
            .set_created_at(&fresh.id, chrono::Utc::now() - chrono::Duration::days(10))
            .unwrap();
        fabricate_fsrs(&db, &fresh.id, 100.0, chrono::Utc::now());

        // Decayed edge, but NOT a lesson (no tag, no fix marker): must NOT flag.
        let noise = ingest(
            &storage,
            "Noted the ledger drift during the nightly close",
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
        link(&storage, &decayed.id, &failure.id, "corrects");
        link(&storage, &fresh.id, &failure.id, "corrects");
        link(&storage, &noise.id, &failure.id, "corrects");

        let out = execute(&storage, Some(json!({"failure_id": failure.id})))
            .await
            .unwrap();
        assert_eq!(out["count"], json!(1), "exactly the decayed lesson: {out}");
        let entry = &out["forgotten_lessons"][0];
        assert_eq!(entry["lesson_id"], json!(decayed.id));
        assert_eq!(entry["shared_anchor"], json!("corrects"));
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
        link(&storage, &later.id, &failure.id, "corrects");
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
        // The cause is the recorded link type, not the words in the content.
        let links = vec!["corrects".to_string()];
        let at = chrono::Utc::now();
        let mut node = KnowledgeNode::default();
        node.content = "Patched the ledger drift during the nightly close".to_string();
        node.tags = vec!["fix".to_string()];
        node.stability = 5.0;
        node.created_at = at - chrono::Duration::days(3650);
        let mut fresh = node.clone();
        fresh.last_accessed = at;
        assert!(detect_lesson(&fresh, &links, at, fresh.last_accessed).is_none());
        let mut stale = node;
        stale.last_accessed = at - chrono::Duration::days(730);
        assert!(detect_lesson(&stale, &[], at, stale.last_accessed).is_none());
        let detected =
            detect_lesson(&stale, &links, at, stale.last_accessed).expect("stale lesson detected");
        assert!(detected.retention < FORGOTTEN_THRESHOLD);
        assert_eq!(detected.shared_anchor, "corrects");
    }
}

#[cfg(test)]
mod strata_tests {
    use super::*;
    use std::collections::HashSet;
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

    fn names(text: &str) -> HashSet<String> {
        text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .filter(|token| !token.is_empty())
            .map(|token| token.to_ascii_lowercase())
            .collect()
    }

    fn vestige_edge(source: &str, target: &str, link_type: &str) -> vestige_core::ConnectionRecord {
        let now = chrono::Utc::now();
        vestige_core::ConnectionRecord {
            source_id: source.to_string(),
            target_id: target.to_string(),
            strength: 1.0,
            link_type: link_type.to_string(),
            created_at: now,
            last_activated: now,
            activation_count: 0,
        }
    }

    #[test]
    fn recorded_edges_ignore_names_and_legacy_inferred() {
        let direct = causal_link_types(
            "lesson",
            "failure",
            &[vestige_edge("lesson", "failure", "corrects")],
        );
        assert_eq!(direct, vec!["corrects".to_string()]);

        let shared_end = causal_link_types(
            "lesson",
            "failure",
            &[
                vestige_edge("lesson", "sym-1", "touched"),
                vestige_edge("failure", "sym-1", "anchored_to"),
            ],
        );
        assert_eq!(
            shared_end,
            vec!["anchored_to".to_string(), "touched".to_string()]
        );
        assert!(shared_end.iter().all(|anchor| anchor != "sym-1"));

        let named = causal_link_types(
            "lesson",
            "failure",
            &[
                vestige_edge("lesson", "failure", "legacy_inferred"),
                vestige_edge("lesson", "API_TIMEOUT", "touched"),
                vestige_edge("failure", "API_TIMEOUT", "legacy_inferred"),
                vestige_edge("lesson", "failure", "semantic"),
            ],
        );
        assert!(named.is_empty(), "{named:?}");
    }

    /// Decayed lesson linked by `corrects`, sharing no name with the failure,
    /// is returned. A decayed lesson that shares `API_TIMEOUT` and the login
    /// path, with no edge, is not. A fresh lesson on a `corrects` edge is not.
    /// The blake3 state digest is unchanged, and no SQLite file appears.
    #[tokio::test]
    async fn surfaces_decayed_lesson_only_through_a_recorded_edge() {
        let dir = tempfile::TempDir::new().unwrap();
        let failure_ms = 1_750_000_000_000_i64;
        let day_ms = 86_400_000_i64;
        let lesson = "Patched the ledger drift during the nightly close";
        let failure_text = "Outage: crash loop in src/auth/login.rs after API_TIMEOUT change";
        let named = "Fixed the API_TIMEOUT overflow in src/auth/login.rs";
        assert!(
            names(lesson).is_disjoint(&names(failure_text)),
            "the linked lesson and the failure must share no name"
        );
        assert!(!names(named).is_disjoint(&names(failure_text)));

        let lesson_id;
        let named_id;
        let fresh_id;
        let failure_id;
        {
            let mut store = StrataStore::open(dir.path()).unwrap();
            lesson_id = store
                .ingest_in_scope(input(lesson, &["fix"], failure_ms - 500 * day_ms), "user")
                .unwrap();
            named_id = store
                .ingest_in_scope(input(named, &["fix"], failure_ms - 500 * day_ms), "user")
                .unwrap();
            fresh_id = store
                .ingest_in_scope(
                    input(
                        "Repaired the journal checksum before dawn",
                        &["fix"],
                        failure_ms - day_ms,
                    ),
                    "user",
                )
                .unwrap();
            failure_id = store
                .ingest_in_scope(input(failure_text, &["incident"], failure_ms), "user")
                .unwrap();
            store
                .save_connection(&ConnectionRecord {
                    source_id: lesson_id.clone(),
                    target_id: failure_id.clone(),
                    link_type: EdgeKind::Corrects.as_str().to_string(),
                    created_at_ms: failure_ms,
                    ..ConnectionRecord::default()
                })
                .unwrap();
            store
                .save_connection(&ConnectionRecord {
                    source_id: fresh_id.clone(),
                    target_id: failure_id.clone(),
                    link_type: EdgeKind::Corrects.as_str().to_string(),
                    created_at_ms: failure_ms,
                    ..ConnectionRecord::default()
                })
                .unwrap();
        }
        let before = digest(dir.path());

        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let out = execute(&storage, Some(json!({"failure_id": failure_id})))
            .await
            .unwrap();
        drop(storage);

        let after = digest(dir.path());
        assert_eq!(before, after, "forgotten_lesson must not write the log");
        assert!(no_sqlite(dir.path()));
        assert_eq!(out["count"], json!(1), "{out}");
        assert_eq!(out["unmapped_reviews"], json!(0), "{out}");

        let entry = &out["forgotten_lessons"][0];
        assert_eq!(entry["lesson_id"], json!(lesson_id));
        assert_eq!(entry["shared_anchor"], json!("corrects"));
        assert_eq!(entry["shared_anchors"], json!(["corrects"]));
        let retention = entry["retention_pct"].as_f64().unwrap();
        assert!(
            (0.0..50.0).contains(&retention),
            "retention must be below 50%, got {retention}"
        );
        let ids: Vec<&str> = out["forgotten_lessons"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["lesson_id"].as_str().unwrap())
            .collect();
        assert!(!ids.contains(&named_id.as_str()), "{out}");
        assert!(!ids.contains(&fresh_id.as_str()), "{out}");
        let note = out["note"].as_str().unwrap();
        assert!(note.contains("no timestamp"), "{note}");
        assert!(note.contains("Name overlap is not a cause"), "{note}");
    }

    /// An explicit review frame has no wall-clock time. The lesson stays
    /// uncounted even though its upsert `created_at_ms` is 500 days old.
    #[tokio::test]
    async fn explicit_review_with_no_timestamp_is_not_scored() {
        let dir = tempfile::TempDir::new().unwrap();
        let failure_ms = 1_750_000_000_000_i64;
        let day_ms = 86_400_000_i64;
        let failure_id;
        {
            let mut store = StrataStore::open(dir.path()).unwrap();
            let lesson_id = store
                .ingest_in_scope(
                    input(
                        "Patched the ledger drift during the nightly close",
                        &["fix"],
                        failure_ms - 500 * day_ms,
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
                    source_id: lesson_id.clone(),
                    target_id: failure_id.clone(),
                    link_type: EdgeKind::Corrects.as_str().to_string(),
                    created_at_ms: failure_ms,
                    ..ConnectionRecord::default()
                })
                .unwrap();
            store.review(&lesson_id, 3).unwrap();
            assert_eq!(
                store.review_clock(&lesson_id),
                Some(strata_store::ReviewClock::Unmapped)
            );
        }
        let before = digest(dir.path());

        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let out = execute(&storage, Some(json!({"failure_id": failure_id})))
            .await
            .unwrap();
        drop(storage);

        let after = digest(dir.path());
        assert_eq!(before, after, "forgotten_lesson must not write the log");
        assert_eq!(out["count"], json!(0), "{out}");
        assert_eq!(out["forgotten_lessons"], json!([]));
        assert_eq!(out["unmapped_reviews"], json!(1), "{out}");
        assert!(
            out["note"].as_str().unwrap().contains("no timestamp"),
            "{}",
            out["note"]
        );
        assert!(no_sqlite(dir.path()));
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
