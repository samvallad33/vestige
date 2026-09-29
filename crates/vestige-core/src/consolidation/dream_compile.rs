//! Dream Compile — the DreamEngine wired to durable state, review-gated.
//!
//! The 4-phase [`DreamEngine`] (NREM1 triage → NREM3 consolidation → REM
//! creative → Integration) was written and tested but had no way to touch a
//! real store. This module is that wiring, with one product rule at its
//! center: **sleep never rewrites memory on its own.** Everything it produces
//! lands as a reviewable, diff-able, evidence-signed Memory PR
//! ([`MemoryPrKind::DreamConsolidation`]); the only direct writes are
//! `memory_connections` strength deltas (replay strengthens co-hit edges,
//! unreplayed weak edges take the Tononi-Cirelli 0.95 downscale).
//!
//! Pipeline:
//! 1. Select the top `memory_count` memories by retention strength in scope.
//! 2. Run the DreamEngine phases over them.
//! 3. Replay: walk existing `memory_connections` among co-hit (both-endpoints
//!    replayed) memories and strengthen them; weak edges not covered by the
//!    replay take the 0.95 synaptic-downscaling factor.
//! 4. Scan the replayed set with the shared contradiction detector.
//! 5. File one Memory PR per contradiction/pattern found (bounded by
//!    `max_prs`; contradictions file first, then REM insights by
//!    confidence x novelty — the Integration phase already sorts them).
//! 6. Record a `dream_history` row so the automation triggers see the run.
//!
//! References: same as `phases.rs` — Tononi & Cirelli (2006) for the 0.95
//! downscale; Diekelmann & Born (2010) for replay-driven consolidation.

use std::collections::HashSet;

use chrono::Utc;
use serde::Serialize;
use uuid::Uuid;

use crate::advanced::contradiction::{SubjectIdentity, appears_contradictory};
use crate::neuroscience::emotional_memory::EmotionalMemory;
use crate::neuroscience::importance_signals::ImportanceSignals;
use crate::neuroscience::synaptic_tagging::SynapticTaggingSystem;
use crate::storage::DreamHistoryRecord;
use crate::trace::{MemoryPr, MemoryPrKind, MemoryPrStatus, RiskSignal};

use super::phases::{DreamEngine, DreamInsight, DreamPhase};

/// Replay strengthens a co-hit edge by this much (capped at 1.0 by the
/// storage layer). Small by design: a single dream is one vote, not a verdict.
pub const REPLAY_STRENGTHEN_BOOST: f64 = 0.1;

/// An edge weaker than this is "weak" for synaptic downscaling. Strong edges
/// are protected even when unreplayed — downscaling targets unused synapses,
/// not established ones.
pub const WEAK_EDGE_FLOOR: f64 = 0.5;

/// Pairs examined by the contradiction scanner per run, mirroring the default
/// `max_pairs` bound of the `dream` action (50 choose 2 = 1225).
const CONTRADICTION_SCAN_MEMORIES: usize = 50;

/// Configuration for [`run_dream_compile`].
#[derive(Debug, Clone)]
pub struct DreamCompileConfig {
    /// Namespace to dream over.
    pub scope: String,
    /// How many top-retention memories enter the dream (5..=500).
    pub memory_count: usize,
    /// Review-PR budget per run. 0 runs the phases but files nothing.
    pub max_prs: usize,
}

impl Default for DreamCompileConfig {
    fn default() -> Self {
        Self {
            scope: "user".to_string(),
            memory_count: 50,
            max_prs: 20,
        }
    }
}

/// One phase of the executed dream, for the report.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DreamCompilePhase {
    pub phase: String,
    pub duration_ms: u64,
    pub memories_processed: usize,
    pub actions: Vec<String>,
}

/// A Memory PR filed by the run.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DreamCompilePr {
    pub id: String,
    pub title: String,
    pub kind: String,
    pub source_ids: Vec<String>,
}

/// What one dream compile did. Serialized straight onto the MCP wire.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DreamCompileReport {
    /// `"compiled"` or `"insufficient_memories"`.
    pub status: String,
    pub scope: String,
    pub memories_selected: usize,
    pub memories_replayed: usize,
    pub phases: Vec<DreamCompilePhase>,
    pub edges_strengthened: usize,
    pub edges_downscaled: usize,
    pub contradictions_found: usize,
    pub insights_generated: usize,
    pub prs_filed: Vec<DreamCompilePr>,
    pub dream_history_recorded: bool,
    pub duration_ms: u64,
}

/// Run the full review-gated dream compile against a store.
///
/// Errors only on storage or configuration failure; a dream that finds
/// nothing is a successful run with zero PRs.
pub fn run_dream_compile(
    storage: &crate::storage::Storage,
    config: &DreamCompileConfig,
) -> Result<DreamCompileReport, String> {
    let run_started = std::time::Instant::now();
    if config.scope.trim().is_empty() {
        return Err("scope must not be empty".into());
    }

    // 1. Select the dream set: top memories by retention strength in scope.
    let mut nodes = storage
        .dream_compile_candidates(&config.scope, config.memory_count)
        .map_err(|e| e.to_string())?;
    // Validity windows are checked in Rust, matching the `dream` action.
    nodes.retain(|n| n.is_currently_valid());

    if nodes.len() < 5 {
        return Ok(DreamCompileReport {
            status: "insufficient_memories".to_string(),
            scope: config.scope.clone(),
            memories_selected: nodes.len(),
            memories_replayed: 0,
            phases: Vec::new(),
            edges_strengthened: 0,
            edges_downscaled: 0,
            contradictions_found: 0,
            insights_generated: 0,
            prs_filed: Vec::new(),
            dream_history_recorded: false,
            duration_ms: run_started.elapsed().as_millis() as u64,
        });
    }

    let run_id = format!("dream_{}", Uuid::new_v4());

    // 2. Run the 4-phase DreamEngine over the selected set.
    let engine = DreamEngine::new();
    let mut emotional = EmotionalMemory::new();
    let importance = ImportanceSignals::new();
    let mut synaptic = SynapticTaggingSystem::new();
    let result = engine.run(&nodes, &mut emotional, &importance, &mut synaptic);

    let phases: Vec<DreamCompilePhase> = result
        .phases
        .iter()
        .map(|p| DreamCompilePhase {
            phase: p.phase.as_str().to_string(),
            duration_ms: p.duration_ms,
            memories_processed: p.memories_processed,
            actions: p.actions.clone(),
        })
        .collect();

    let content_by_id: std::collections::HashMap<&str, &str> =
        nodes.iter().map(|n| (n.id.as_str(), n.content.as_str())).collect();

    // 3. Replay over memory_connections. Each unordered pair is visited once
    // even though get_connections_for_memory returns it from both endpoints.
    let replay_set: HashSet<&str> = result.replay_queue.iter().map(|s| s.as_str()).collect();
    let mut visited_pairs: HashSet<(String, String)> = HashSet::new();
    let mut edges_strengthened = 0usize;
    let mut edges_downscaled = 0usize;
    for id in &result.replay_queue {
        let connections = storage
            .get_connections_for_memory(id)
            .map_err(|e| format!("failed to read connections for replay: {e}"))?;
        for conn in connections {
            if !visited_pairs.insert(ordered_pair(&conn.source_id, &conn.target_id)) {
                continue;
            }
            let both_replayed = replay_set.contains(conn.source_id.as_str())
                && replay_set.contains(conn.target_id.as_str());
            if both_replayed {
                if storage
                    .strengthen_connection(
                        &conn.source_id,
                        &conn.target_id,
                        REPLAY_STRENGTHEN_BOOST,
                    )
                    .map_err(|e| e.to_string())?
                {
                    edges_strengthened += 1;
                }
            } else if conn.strength < WEAK_EDGE_FLOOR {
                // Synaptic downscaling (Tononi & Cirelli): an edge the dream
                // did not replay — one endpoint outside the replay set —
                // takes the 0.95 factor. Written through the existing
                // connection upsert so created_at / activation_count survive.
                let mut updated = conn.clone();
                updated.strength *= 0.95;
                storage
                    .save_connection(&updated)
                    .map_err(|e| e.to_string())?;
                edges_downscaled += 1;
            }
        }
    }

    // 4+5. Contradictions first, then REM insights — everything as PRs.
    let mut prs: Vec<MemoryPr> = Vec::new();
    let mut seen_pairs: HashSet<(String, String)> = HashSet::new();

    // 4. Contradiction scan over the replayed set (bounded).
    let scan: Vec<&String> = result
        .replay_queue
        .iter()
        .take(CONTRADICTION_SCAN_MEMORIES)
        .collect();
    let mut contradictions_found = 0usize;
    for (i, a_id) in scan.iter().enumerate() {
        for b_id in scan.iter().skip(i + 1) {
            let (Some(a_text), Some(b_text)) = (
                content_by_id.get(a_id.as_str()),
                content_by_id.get(b_id.as_str()),
            ) else {
                continue;
            };
            if !appears_contradictory(a_text, b_text, SubjectIdentity::FromTextOverlap) {
                continue;
            }
            contradictions_found += 1;
            if !seen_pairs.insert(ordered_pair(a_id, b_id)) {
                continue;
            }
            if prs.len() >= config.max_prs {
                continue;
            }
            prs.push(contradiction_pr(
                &run_id, nodes.len(), a_id, b_id, a_text, b_text,
            ));
        }
    }

    // REM insights (Integration already validated, sorted, and capped them).
    let insights_generated = result.insights.len();
    for insight in &result.insights {
        if prs.len() >= config.max_prs {
            break;
        }
        let Some(first) = insight.source_memory_ids.first() else {
            continue;
        };
        let second = insight.source_memory_ids.get(1).unwrap_or(first);
        if !seen_pairs.insert(ordered_pair(first, second)) {
            continue;
        }
        prs.push(insight_pr(&run_id, &config.scope, nodes.len(), insight));
    }

    // File the PRs. Each is a reviewable proposal; nothing here mutates a
    // memory row.
    let mut prs_filed = Vec::with_capacity(prs.len());
    for pr in &prs {
        match storage.save_memory_pr(pr) {
            Ok(()) => prs_filed.push(DreamCompilePr {
                id: pr.id.clone(),
                title: pr.title.clone(),
                kind: pr.kind.as_str().to_string(),
                source_ids: pr_source_ids(pr),
            }),
            Err(e) => {
                // A failed PR write must not fail the whole dream; the run is
                // still recorded and the remaining PRs still file.
                tracing::warn!(run_id = %run_id, error = %e, "failed to file dream PR");
            }
        }
    }

    // 6. Dream history row — feeds the automation triggers (`needsDream`).
    let mut dream_history_recorded = false;
    let phase_ms = |want: DreamPhase| {
        result
            .phases
            .iter()
            .find(|p| p.phase == want)
            .map(|p| p.duration_ms as i64)
    };
    let record = DreamHistoryRecord {
        dreamed_at: Utc::now(),
        duration_ms: run_started.elapsed().as_millis() as i64,
        memories_replayed: result.memories_replayed as i32,
        connections_found: (edges_strengthened + edges_downscaled) as i32,
        insights_generated: insights_generated as i32,
        memories_strengthened: result.memories_strengthened as i32,
        memories_compressed: 0,
        phase_nrem1_ms: phase_ms(DreamPhase::Nrem1),
        phase_nrem3_ms: phase_ms(DreamPhase::Nrem3),
        phase_rem_ms: phase_ms(DreamPhase::Rem),
        phase_integration_ms: phase_ms(DreamPhase::Integration),
        summaries_generated: None,
        emotional_memories_processed: Some(result.emotional_processed as i32),
        creative_connections_found: Some(result.creative_connections.len() as i32),
    };
    match storage.save_dream_history(&record) {
        Ok(_) => dream_history_recorded = true,
        Err(e) => {
            tracing::warn!(run_id = %run_id, error = %e, "failed to record dream history");
        }
    }

    Ok(DreamCompileReport {
        status: "compiled".to_string(),
        scope: config.scope.clone(),
        memories_selected: nodes.len(),
        memories_replayed: result.memories_replayed,
        phases,
        edges_strengthened,
        edges_downscaled,
        contradictions_found,
        insights_generated,
        prs_filed,
        dream_history_recorded,
        duration_ms: run_started.elapsed().as_millis() as u64,
    })
}

fn ordered_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

/// A bounded claim excerpt for the PR diff.
fn claim_excerpt(text: &str) -> String {
    if text.len() <= 200 {
        text.to_string()
    } else {
        text.get(..200).unwrap_or(text).to_string()
    }
}

fn dream_signal(run_id: &str, memories: usize) -> RiskSignal {
    RiskSignal {
        code: "dream_consolidation".to_string(),
        detail: format!(
            "Proposed by dream compile run {run_id} (4-phase DreamEngine over {memories} memories). Machine-generated memory change, held for review."
        ),
    }
}

fn contradiction_pr(
    run_id: &str,
    memories: usize,
    a_id: &str,
    b_id: &str,
    a_text: &str,
    b_text: &str,
) -> MemoryPr {
    let short_a = &a_id[..8.min(a_id.len())];
    let short_b = &b_id[..8.min(b_id.len())];
    MemoryPr {
        id: Uuid::new_v4().to_string(),
        kind: MemoryPrKind::DreamConsolidation,
        status: MemoryPrStatus::Pending,
        title: format!("Dream: contradiction between {short_a} and {short_b}"),
        diff: serde_json::json!({
            "change": "contradiction_resolution",
            "before": [
                {"id": a_id, "claim": claim_excerpt(a_text)},
                {"id": b_id, "claim": claim_excerpt(b_text)},
            ],
            "proposed": "Resolve the conflict: supersede the outdated claim, or record a conditional (context-dependent) version of both. Reviewer decides; the dream only flags.",
            "evidence": {
                "detector": "appears_contradictory (SubjectIdentity::FromTextOverlap)",
                "replay_set": "both memories were co-replayed in the NREM3 consolidation wave",
            },
        }),
        signals: vec![
            dream_signal(run_id, memories),
            RiskSignal {
                code: "contradiction_detected".to_string(),
                detail: "The replayed memories assert opposing positions on a shared subject."
                    .to_string(),
            },
            RiskSignal {
                code: "replay_evidence".to_string(),
                detail: "Both sides of the conflict were in the same replay set — co-retrieval evidence they compete for the same slot.".to_string(),
            },
        ],
        subject_id: Some(a_id.to_string()),
        run_id: Some(run_id.to_string()),
        created_at: Utc::now().to_rfc3339(),
        decided_at: None,
        decision: None,
    }
}

fn insight_pr(run_id: &str, scope: &str, memories: usize, insight: &DreamInsight) -> MemoryPr {
    MemoryPr {
        id: Uuid::new_v4().to_string(),
        kind: MemoryPrKind::DreamConsolidation,
        status: MemoryPrStatus::Pending,
        title: format!(
            "Dream: {} insight (confidence {:.2})",
            insight.insight_type, insight.confidence
        ),
        diff: serde_json::json!({
            "change": "pattern_principle",
            "proposed": insight.insight,
            "sources": insight.source_memory_ids,
            "confidence": insight.confidence,
            "novelty": insight.novelty,
            "insight_type": insight.insight_type,
            "evidence": {
                "origin": format!("REM_Creative phase, scope '{scope}'"),
                "validation": "Integration phase kept this above the validation threshold",
            },
        }),
        signals: vec![
            dream_signal(run_id, memories),
            RiskSignal {
                code: "pattern_evidence".to_string(),
                detail: format!(
                    "Cross-memory pattern at confidence {:.2} and novelty {:.2}, sourced from {} memory ids.",
                    insight.confidence, insight.novelty, insight.source_memory_ids.len()
                ),
            },
        ],
        subject_id: insight.source_memory_ids.first().cloned(),
        run_id: Some(run_id.to_string()),
        created_at: Utc::now().to_rfc3339(),
        decided_at: None,
        decision: None,
    }
}

/// Recover the source memory ids from the stored diff (contradiction PRs
/// carry them under `before[].id`, insight PRs under `sources`).
fn pr_source_ids(pr: &MemoryPr) -> Vec<String> {
    if let Some(sources) = pr.diff.get("sources").and_then(|v| v.as_array()) {
        return sources
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
    }
    pr.diff
        .get("before")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|o| o.get("id").and_then(|v| v.as_str().map(String::from)))
                .collect()
        })
        .unwrap_or_default()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::ConnectionRecord;
    use chrono::Duration;

    fn test_storage() -> (std::sync::Arc<crate::storage::SqliteMemoryStore>, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = std::sync::Arc::new(crate::storage::SqliteMemoryStore::new(Some(dir.path().join("test.db"))).unwrap());
        (storage, dir)
    }

    fn ingest(storage: &crate::storage::SqliteMemoryStore, content: &str, tags: &[&str]) -> String {
        storage
            .ingest(crate::IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: tags.iter().map(|t| t.to_string()).collect(),
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap()
            .id
    }

    fn seed_connection(storage: &crate::storage::SqliteMemoryStore, a: &str, b: &str, strength: f64) {
        let now = Utc::now();
        storage
            .save_connection(&ConnectionRecord {
                source_id: a.to_string(),
                target_id: b.to_string(),
                strength,
                link_type: "semantic".to_string(),
                created_at: now,
                last_activated: now,
                activation_count: 1,
            })
            .unwrap();
    }

    fn edge(storage: &crate::storage::SqliteMemoryStore, a: &str, b: &str) -> Option<ConnectionRecord> {
        storage
            .get_connections_for_memory(a)
            .unwrap()
            .into_iter()
            .find(|c| c.source_id == a && c.target_id == b)
    }

    fn config(scope: &str) -> DreamCompileConfig {
        DreamCompileConfig {
            scope: scope.to_string(),
            ..DreamCompileConfig::default()
        }
    }

    /// Ten related memories: a replay-able corpus.
    fn store_with_corpus() -> (std::sync::Arc<crate::storage::SqliteMemoryStore>, tempfile::TempDir, Vec<String>) {
        let (storage, dir) = test_storage();
        let mut ids = Vec::new();
        for i in 0..10 {
            ids.push(ingest(
                &storage,
                &format!("Dream compile test memory number {i} about the deploy pipeline"),
                &["dream-compile-test"],
            ));
        }
        (storage, dir, ids)
    }

    /// An anchor memory that can never enter the dream set: expired before
    /// the run. Deterministic way to place an edge OUTSIDE the replay set
    /// (fresh memories all tie at retention 1.0, so tiny stores would select
    /// everything and co-replay every edge).
    fn expired_anchor(storage: &crate::storage::SqliteMemoryStore, content: &str) -> String {
        storage
            .ingest(crate::IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec!["expired-anchor".to_string()],
                valid_from: Some(Utc::now() - Duration::days(10)),
                valid_until: Some(Utc::now() - Duration::days(1)),
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap()
            .id
    }

    #[test]
    fn empty_store_reports_insufficient_memories() {
        let (storage, _dir) = test_storage();
        let report = run_dream_compile(&*storage, &config("user")).unwrap();
        assert_eq!(report.status, "insufficient_memories");
        assert!(report.prs_filed.is_empty());
        assert!(!report.dream_history_recorded);
    }

    #[test]
    fn compile_runs_all_four_phases_and_records_history() {
        let (storage, _dir, _ids) = store_with_corpus();

        let report = run_dream_compile(&*storage, &config("user")).unwrap();
        assert_eq!(report.status, "compiled");
        assert_eq!(report.memories_selected, 10);
        assert_eq!(report.memories_replayed, 10);
        let names: Vec<&str> = report.phases.iter().map(|p| p.phase.as_str()).collect();
        assert_eq!(
            names,
            [
                "NREM1_Triage",
                "NREM3_Consolidation",
                "REM_Creative",
                "Integration"
            ]
        );
        assert!(report.dream_history_recorded, "dream_history row recorded");
        // The history row is what resets the needsDream automation trigger.
        assert!(storage.get_last_dream().unwrap().is_some());
    }

    #[test]
    fn replay_strengthens_co_hit_edges() {
        let (storage, _dir, ids) = store_with_corpus();
        seed_connection(&storage, &ids[0], &ids[1], 0.6);
        let before = edge(&storage, &ids[0], &ids[1]).unwrap().strength;

        let report = run_dream_compile(&*storage, &config("user")).unwrap();
        assert_eq!(report.edges_strengthened, 1);
        let after = edge(&storage, &ids[0], &ids[1]).unwrap().strength;
        assert!(after > before, "co-replayed edge must strengthen: {before} -> {after}");
        assert!(after <= 1.0, "strengthening must stay capped at 1.0");
    }

    #[test]
    fn unreplayed_weak_edges_take_the_downscale() {
        let (storage, _dir, ids) = store_with_corpus();
        // A weak edge to an anchor OUTSIDE the dream set: it cannot be
        // co-replayed, so it takes the synaptic downscale.
        let outside = expired_anchor(&storage, "An expired stray note far from the dream");
        seed_connection(&storage, &ids[2], &outside, 0.40);
        // A strong edge reaching outside: protected despite being unreplayed.
        let outside_2 = expired_anchor(&storage, "Another expired strong anchor note");
        seed_connection(&storage, &ids[3], &outside_2, 0.90);

        let report = run_dream_compile(&*storage, &config("user")).unwrap();
        assert_eq!(report.edges_downscaled, 1, "only the weak edge downscales");
        let weak = edge(&storage, &ids[2], &outside).unwrap();
        assert!((weak.strength - 0.40 * 0.95).abs() < 1e-9);
        let strong = edge(&storage, &ids[3], &outside_2).unwrap();
        assert!((strong.strength - 0.90).abs() < 1e-9, "strong edge untouched");
    }

    #[test]
    fn contradiction_between_replayed_memories_files_a_pr() {
        let (storage, _dir) = test_storage();
        // Shared substantive vocabulary + a ("never", "always") polarity flip.
        let _a = ingest(
            &storage,
            "Deployments to production always use the blue pipeline on friday",
            &["deploys"],
        );
        let _b = ingest(
            &storage,
            "Deployments to production never use the blue pipeline on friday",
            &["deploys"],
        );
        // Corpus padding so the 5-memory floor passes.
        for i in 0..4 {
            ingest(
                &storage,
                &format!("Deployment pipeline note {i} about staging checks"),
                &["deploys"],
            );
        }

        assert!(
            appears_contradictory(
                "Deployments to production always use the blue pipeline on friday",
                "Deployments to production never use the blue pipeline on friday",
                SubjectIdentity::FromTextOverlap,
            ),
            "fixture must be a detected contradiction"
        );

        let report = run_dream_compile(&*storage, &config("user")).unwrap();
        assert_eq!(report.status, "compiled");
        assert!(report.contradictions_found >= 1, "{report:?}");
        assert!(report.prs_filed.iter().any(|p| p.kind == "dream_consolidation"));

        // The PR is queryable, pending, and evidence-signed.
        let pending = storage
            .list_memory_prs(Some(MemoryPrStatus::Pending), 100)
            .unwrap();
        let dream_prs: Vec<_> = pending
            .iter()
            .filter(|p| p.kind == MemoryPrKind::DreamConsolidation)
            .collect();
        assert!(!dream_prs.is_empty());
        let contradiction_pr = dream_prs
            .iter()
            .find(|p| p.diff.get("change").and_then(|c| c.as_str()) == Some("contradiction_resolution"))
            .expect("a contradiction PR must be stored");
        assert!(contradiction_pr.signals.iter().any(|s| s.code == "dream_consolidation"));
        assert!(contradiction_pr.signals.iter().any(|s| s.code == "contradiction_detected"));
        assert!(contradiction_pr.run_id.as_deref().unwrap_or("").starts_with("dream_"));
    }

    #[test]
    fn pattern_insight_files_a_reviewable_pr() {
        let (storage, _dir) = test_storage();
        // A shared "error handling" bigram across 4+ memories gives the REM
        // pattern extractor (bigrams shared by 3-10 memories) something to
        // find; varied first tags give cross-domain pairing room.
        let topics = [
            ("Rust error handling with Result types", vec!["rust", "safety"]),
            ("TypeScript error handling with try catch blocks", vec!["typescript"]),
            ("Python error handling with exception classes", vec!["python"]),
            ("Go error handling with error values everywhere", vec!["go"]),
            ("Java error handling with checked exceptions", vec!["java"]),
            ("A cooking recipe for sourdough bread at home", vec!["cooking"]),
        ];
        for (content, tags) in &topics {
            ingest(&storage, content, tags);
        }

        let report = run_dream_compile(&*storage, &config("user")).unwrap();
        assert_eq!(report.status, "compiled");
        assert!(
            report.prs_filed.iter().any(|p| p.kind == "dream_consolidation"),
            "REM patterns must land as PRs, got: {report:?}"
        );
        // Insight PRs carry a structured proposed change, not a memory write.
        let pending = storage
            .list_memory_prs(Some(MemoryPrStatus::Pending), 100)
            .unwrap();
        assert!(pending
            .iter()
            .any(|p| p.diff.get("change").and_then(|c| c.as_str()) == Some("pattern_principle")));
    }

    #[test]
    fn no_direct_memory_mutation_only_prs_and_edge_strengths() {
        let (storage, _dir, ids) = store_with_corpus();
        let outside = expired_anchor(&storage, "An expired weak anchor note");
        seed_connection(&storage, &ids[0], &ids[1], 0.6);
        seed_connection(&storage, &ids[2], &outside, 0.40);

        // Snapshot every memory row before the dream.
        #[derive(PartialEq, Debug, Clone)]
        struct NodeSnapshot {
            id: String,
            content: String,
            retention_strength: f64,
            suppression_count: i32,
            tags: Vec<String>,
            created_at: chrono::DateTime<Utc>,
            updated_at: chrono::DateTime<Utc>,
        }
        let snapshot = |storage: &crate::storage::SqliteMemoryStore| -> Vec<NodeSnapshot> {
            storage
                .get_all_nodes(1000, 0)
                .unwrap()
                .into_iter()
                .map(|n| NodeSnapshot {
                    id: n.id,
                    content: n.content,
                    retention_strength: n.retention_strength,
                    suppression_count: n.suppression_count,
                    tags: n.tags,
                    created_at: n.created_at,
                    updated_at: n.updated_at,
                })
                .collect()
        };
        let before = snapshot(&storage);

        let report = run_dream_compile(&*storage, &config("user")).unwrap();

        let after = snapshot(&storage);
        assert_eq!(
            before, after,
            "dream compile must not mutate any memory row — changes land as PRs only"
        );

        // And the produced surface is exactly what the contract allows:
        // pending PRs plus memory_connections strength deltas.
        assert!(report.dream_history_recorded);
        let pending = storage
            .list_memory_prs(Some(MemoryPrStatus::Pending), 100)
            .unwrap();
        assert!(
            pending.iter().all(|p| p.kind == MemoryPrKind::DreamConsolidation),
            "every filed PR must be a DreamConsolidation proposal"
        );
        let strengthened = edge(&storage, &ids[0], &ids[1]).unwrap();
        assert!(strengthened.strength > 0.6);
        let downscaled = edge(&storage, &ids[2], &outside).unwrap();
        assert!((downscaled.strength - 0.40 * 0.95).abs() < 1e-9);
    }

    #[test]
    fn max_prs_budget_is_respected() {
        let (storage, _dir) = test_storage();
        let _a = ingest(
            &storage,
            "Deployments to production always use the blue pipeline on friday",
            &["deploys"],
        );
        let _b = ingest(
            &storage,
            "Deployments to production never use the blue pipeline on friday",
            &["deploys"],
        );
        for i in 0..4 {
            ingest(
                &storage,
                &format!("Deployment pipeline note {i} about staging checks"),
                &["deploys"],
            );
        }

        let report = run_dream_compile(
            &*storage,
            &DreamCompileConfig {
                max_prs: 1,
                ..config("user")
            },
        )
        .unwrap();
        assert!(report.prs_filed.len() <= 1, "{report:?}");
        // The budget caps what FILES, not what the dream FOUND.
        assert!(report.contradictions_found >= 1 || report.insights_generated >= 1);

        let zero = run_dream_compile(
            &*storage,
            &DreamCompileConfig {
                max_prs: 0,
                ..config("user")
            },
        )
        .unwrap();
        assert!(zero.prs_filed.is_empty());
    }

    #[test]
    fn report_names_the_dreamed_namespace() {
        let (storage, _dir) = test_storage();
        for i in 0..10 {
            ingest(&storage, &format!("Scoped item {i} about the deploy pipeline"), &["work"]);
        }
        let report = run_dream_compile(&*storage, &config("user")).unwrap();
        assert_eq!(report.scope, "user");
        assert!(report.memories_selected >= 10, "{report:?}");
    }

    #[test]
    fn stale_and_expired_never_enter_the_dream() {
        let (storage, _dir) = test_storage();
        // One memory already expired (valid_until in the past).
        storage
            .ingest(crate::IngestInput {
                content: "An expired dream note about the deploy pipeline".to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec!["expired".to_string()],
                valid_from: Some(Utc::now() - Duration::days(10)),
                valid_until: Some(Utc::now() - Duration::days(1)),
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap();
        for i in 0..10 {
            ingest(
                &storage,
                &format!("Current dream note {i} about the deploy pipeline"),
                &["current"],
            );
        }

        let report = run_dream_compile(&*storage, &config("user")).unwrap();
        assert_eq!(report.status, "compiled");
        // 11 ingested, 1 expired: the dream set must be the 10 current ones.
        assert_eq!(report.memories_selected, 10, "{report:?}");
    }

    #[test]
    fn empty_scope_is_rejected() {
        let (storage, _dir) = test_storage();
        let err = run_dream_compile(
            &*storage,
            &DreamCompileConfig {
                scope: "  ".to_string(),
                ..DreamCompileConfig::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("scope"));
    }
}
