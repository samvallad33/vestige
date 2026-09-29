//! Dream compile — the review-gated DreamEngine action.
//!
//! Runs the 4-phase DreamEngine (NREM1 triage → NREM3 consolidation → REM
//! creative → Integration) over the top memories by retention strength in a
//! scope, then lands EVERYTHING it found as reviewable Memory PRs
//! (`kind=dream_consolidation`). The only direct writes are
//! `memory_connections` strength deltas: co-hit edges strengthen, weak
//! unreplayed edges take the 0.95 synaptic-downscaling factor. No memory row
//! is ever mutated — sleep proposes, the reviewer disposes.
//!
//! Integrated, not duplicated: the existing `maintain(action="dream")`
//! (MemoryDreamer) stays as-is; this action is the 4-phase engine's surface.

use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use vestige_core::{
    ConnectionRecord, DreamCompileConfig, DreamCompilePhase, DreamCompileReport, KnowledgeNode,
    Storage, run_dream_compile,
};

const BOOST_MILLI: i64 = 100;
const CAP_MILLI: i64 = 1000;
const WEAK_MILLI: i64 = 500;

pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "scope":{"type":"string","default":"user","description":"Namespace to dream over."},
            "memory_count":{"type":"integer","minimum":5,"maximum":500,"default":50,"description":"Top memories by retention strength entering the dream."},
            "max_prs":{"type":"integer","minimum":0,"maximum":50,"default":20,"description":"Review-PR budget; contradictions file first, then REM insights. No memory is written autonomously either way."}
        }
    })
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    #[derive(serde::Deserialize, Default)]
    struct Args {
        scope: Option<String>,
        memory_count: Option<usize>,
        max_prs: Option<usize>,
    }
    let parsed: Args = serde_json::from_value(args.unwrap_or_else(|| serde_json::json!({})))
        .map_err(|e| e.to_string())?;
    let memory_count = parsed.memory_count.unwrap_or(50);
    let max_prs = parsed.max_prs.unwrap_or(20);
    if !(5..=500).contains(&memory_count) || max_prs > 50 {
        return Err("invalid memory_count (5..=500) or max_prs (0..=50)".into());
    }
    let config = DreamCompileConfig {
        scope: parsed.scope.unwrap_or_else(|| "user".into()),
        memory_count,
        max_prs,
    };

    // Synchronous store work goes to the blocking pool, matching consolidate.
    let storage = Arc::clone(storage);
    let report = tokio::task::spawn_blocking(move || {
        if crate::strata_memory::is_strata_backend(storage.as_ref()) {
            run_strata_dream_compile(storage.as_ref(), &config)
        } else {
            run_dream_compile(storage.as_ref(), &config)
        }
    })
    .await
    .map_err(|e| format!("dream compile task failed: {e}"))??;

    serde_json::to_value(report).map_err(|e| e.to_string())
}

/// Rank live memories by FSRS retrievability and replay recorded edges.
///
/// Co-replayed edges gain 0.1 (cap 1.0). A weak edge (under 0.5) with one end
/// outside the replay set is scaled by 0.95. A `corrects` edge whose ends are
/// both replayed is a contradiction; `derived_from` / `evidence_of` edges in
/// that set are the insight count. The log has no memory-pr or dream-history
/// op, so those stay unfiled. Decisions never read content.
fn run_strata_dream_compile(
    storage: &Storage,
    config: &DreamCompileConfig,
) -> Result<DreamCompileReport, String> {
    let started = Instant::now();
    let mut nodes = live_nodes(storage, &config.scope)?;
    nodes.sort_by(|a, b| {
        b.retention_strength
            .total_cmp(&a.retention_strength)
            .then_with(|| a.id.cmp(&b.id))
    });
    nodes.truncate(config.memory_count);
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
            duration_ms: elapsed_ms(started),
        });
    }

    let replay: HashSet<&str> = nodes.iter().map(|node| node.id.as_str()).collect();
    let selected = nodes.len();
    let triage_at = Instant::now();
    let triage = phase(
        "NREM1_Triage",
        triage_at,
        selected,
        vec![format!("ranked {selected} by fsrs retrievability")],
    );

    let edges = latest_edges(storage, &replay)?;
    let mut edges_strengthened = 0usize;
    let mut edges_downscaled = 0usize;
    let mut contradictions_found = 0usize;
    let mut insights_generated = 0usize;
    let consolidate_at = Instant::now();
    for edge in &edges {
        let both =
            replay.contains(edge.source_id.as_str()) && replay.contains(edge.target_id.as_str());
        if both && edge.link_type == "corrects" {
            contradictions_found += 1;
        }
        if both && (edge.link_type == "derived_from" || edge.link_type == "evidence_of") {
            insights_generated += 1;
        }
        let current = strength_milli(edge.strength);
        if both {
            let mut updated = edge.clone();
            updated.strength = i64_milli((current + BOOST_MILLI).min(CAP_MILLI));
            updated.activation_count = updated.activation_count.saturating_add(1);
            storage
                .save_connection(&updated)
                .map_err(|e| e.to_string())?;
            edges_strengthened += 1;
        } else if current < WEAK_MILLI {
            let next = current * 95 / 100;
            if next != current {
                let mut updated = edge.clone();
                updated.strength = i64_milli(next);
                storage
                    .save_connection(&updated)
                    .map_err(|e| e.to_string())?;
                edges_downscaled += 1;
            }
        }
    }
    let consolidation = phase(
        "NREM3_Consolidation",
        consolidate_at,
        selected,
        vec![format!(
            "strengthened {edges_strengthened}; downscaled {edges_downscaled}"
        )],
    );
    let rem_at = Instant::now();
    let rem = phase(
        "REM_Creative",
        rem_at,
        edges.len(),
        vec![format!(
            "recorded edges: {insights_generated} derived_from/evidence_of, {contradictions_found} corrects"
        )],
    );
    let integrate_at = Instant::now();
    let integration = phase(
        "Integration",
        integrate_at,
        selected,
        vec!["no memory row rewritten".to_string()],
    );

    Ok(DreamCompileReport {
        status: "compiled".to_string(),
        scope: config.scope.clone(),
        memories_selected: selected,
        memories_replayed: selected,
        phases: vec![triage, consolidation, rem, integration],
        edges_strengthened,
        edges_downscaled,
        contradictions_found,
        insights_generated,
        prs_filed: Vec::new(),
        dream_history_recorded: false,
        duration_ms: elapsed_ms(started),
    })
}

fn live_nodes(storage: &Storage, scope: &str) -> Result<Vec<KnowledgeNode>, String> {
    let superseded = storage.superseded_node_ids().map_err(|e| e.to_string())?;
    let mut nodes = Vec::new();
    let mut offset = 0i32;
    loop {
        let page = storage
            .get_all_nodes_in_scope(scope, 1000, offset)
            .map_err(|e| e.to_string())?;
        if page.is_empty() {
            break;
        }
        let count = i32::try_from(page.len()).unwrap_or(i32::MAX);
        for node in page {
            if !superseded.contains(&node.id) && node.is_currently_valid() {
                nodes.push(node);
            }
        }
        offset = offset.saturating_add(count);
        if count < 1000 {
            break;
        }
    }
    Ok(nodes)
}

/// Latest admitted frame per directed `(source, target, link_type)`.
/// `SaveEdge` appends; an earlier frame is history, not the live strength.
fn latest_edges(
    storage: &Storage,
    replay: &HashSet<&str>,
) -> Result<Vec<ConnectionRecord>, String> {
    let all = storage.get_all_connections().map_err(|e| e.to_string())?;
    let mut latest: BTreeMap<(String, String, String), ConnectionRecord> = BTreeMap::new();
    for edge in all {
        if !replay.contains(edge.source_id.as_str()) && !replay.contains(edge.target_id.as_str()) {
            continue;
        }
        let key = (
            edge.source_id.clone(),
            edge.target_id.clone(),
            edge.link_type.clone(),
        );
        latest.insert(key, edge);
    }
    Ok(latest.into_values().collect())
}

fn strength_milli(strength: f64) -> i64 {
    if !strength.is_finite() || strength <= 0.0 {
        return 0;
    }
    (strength * 1000.0).round().clamp(0.0, CAP_MILLI as f64) as i64
}

fn i64_milli(milli: i64) -> f64 {
    milli as f64 / 1000.0
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn phase(
    name: &str,
    started: Instant,
    processed: usize,
    actions: Vec<String>,
) -> DreamCompilePhase {
    DreamCompilePhase {
        phase: name.to_string(),
        duration_ms: elapsed_ms(started),
        memories_processed: processed,
        actions,
    }
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        (storage, dir)
    }

    async fn ingest_n(storage: &Arc<Storage>, n: usize) {
        for i in 0..n {
            storage
                .ingest(vestige_core::IngestInput {
                    content: format!(
                        "Dream compile wire test memory {i} about the deploy pipeline"
                    ),
                    node_type: "fact".to_string(),
                    source: None,
                    sentiment_score: 0.0,
                    sentiment_magnitude: 0.0,
                    tags: vec!["dream-compile-wire".to_string()],
                    valid_from: None,
                    valid_until: None,
                    validity_inferred: false,
                    source_envelope: None,
                })
                .unwrap();
        }
    }

    #[test]
    fn schema_is_compact_and_bounded() {
        let s = schema();
        assert_eq!(s["type"], "object");
        assert_eq!(s["properties"]["memory_count"]["default"], 50);
        assert_eq!(s["properties"]["memory_count"]["maximum"], 500);
        assert_eq!(s["properties"]["max_prs"]["default"], 20);
    }

    #[tokio::test]
    async fn insufficient_memories_is_a_clean_response() {
        let (storage, _dir) = test_storage().await;
        let value = execute(&storage, None).await.unwrap();
        assert_eq!(value["status"], "insufficient_memories");
        assert_eq!(value["memoriesSelected"], 0);
        assert_eq!(value["prsFiled"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn compiles_a_small_store_end_to_end() {
        let (storage, _dir) = test_storage().await;
        ingest_n(&storage, 10).await;

        let value = execute(&storage, None).await.unwrap();
        assert_eq!(value["status"], "compiled");
        assert_eq!(value["memoriesSelected"], 10);
        assert_eq!(value["dreamHistoryRecorded"], true);
        let phases = value["phases"].as_array().unwrap();
        assert_eq!(phases.len(), 4);
        assert_eq!(phases[0]["phase"], "NREM1_Triage");
        assert_eq!(phases[3]["phase"], "Integration");
        assert!(value["edgesStrengthened"].is_number());
        assert!(value["edgesDownscaled"].is_number());
        assert!(value["contradictionsFound"].is_number());
        assert!(value["prsFiled"].is_array());
        // The history row resets the needsDream automation trigger.
        assert!(storage.get_last_dream().unwrap().is_some());
    }

    #[tokio::test]
    async fn rejects_out_of_range_arguments() {
        let (storage, _dir) = test_storage().await;
        for args in [
            serde_json::json!({"memory_count": 4}),
            serde_json::json!({"memory_count": 501}),
            serde_json::json!({"max_prs": 51}),
            serde_json::json!({"scope": " "}),
        ] {
            assert!(execute(&storage, Some(args)).await.is_err());
        }
    }
}

#[cfg(test)]
mod strata_tests {
    use super::*;

    fn seed() -> (tempfile::TempDir, Vec<String>) {
        let dir = tempfile::tempdir().unwrap();
        let ids = {
            let mut store = strata_store::StrataStore::open(dir.path()).unwrap();
            let mut ids = Vec::new();
            for (index, content) in [
                "cache is always enabled",
                "cache is never enabled",
                "cache is always enabled",
                "deploy timestamp note",
                "cache is never enabled",
                "shipping checklist",
            ]
            .into_iter()
            .enumerate()
            {
                ids.push(
                    store
                        .ingest_in_scope(
                            strata_store::IngestInput {
                                content: format!("{content} {index}"),
                                source: None,
                                source_updated_at_ms: None,
                                node_type: "fact".into(),
                                tags: Vec::new(),
                                created_at_ms: Some(1_700_000_000_000),
                                valid_from_ms: None,
                                valid_until_ms: None,
                            },
                            "user",
                        )
                        .unwrap(),
                );
            }
            // Easy review moves the oldest id to the front of the FSRS order.
            store.review(&ids[0], 4).unwrap();
            store
                .save_connection(&strata_edge(&ids[0], &ids[5], "derived_from", 600))
                .unwrap();
            store
                .save_connection(&strata_edge(&ids[0], &ids[1], "derived_from", 400))
                .unwrap();
            store
                .save_connection(&strata_edge(&ids[2], &ids[3], "corrects", 800))
                .unwrap();
            let mut ranked: Vec<(String, f64)> = ids
                .iter()
                .map(|id| (id.clone(), store.retrievability(id).unwrap().unwrap()))
                .collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            assert_eq!(ranked[0].0, ids[0], "reviewed card leads: {ranked:?}");
            assert_eq!(
                ranked[5].0, ids[1],
                "unreviewed second ingest is last: {ranked:?}"
            );
            ids
        };
        (dir, ids)
    }

    fn strata_edge(
        source: &str,
        target: &str,
        link_type: &str,
        milli: i64,
    ) -> strata_store::ConnectionRecord {
        strata_store::ConnectionRecord {
            source_id: source.to_string(),
            target_id: target.to_string(),
            strength_milli: milli,
            link_type: link_type.to_string(),
            meta_sha: None,
            created_at_ms: 1_700_000_000_000,
            activation_count: 1,
        }
    }

    fn latest_milli(dir: &std::path::Path, source: &str, target: &str, link_type: &str) -> i64 {
        strata_store::StrataStore::open(dir)
            .unwrap()
            .edges()
            .into_iter()
            .rfind(|edge| {
                edge.source_id == source && edge.target_id == target && edge.link_type == link_type
            })
            .unwrap()
            .strength_milli
    }

    #[tokio::test]
    async fn strata_compile_follows_fsrs_and_recorded_edges() {
        let (dir, ids) = seed();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let value = execute(
            &storage,
            Some(serde_json::json!({"memory_count": 5, "max_prs": 20})),
        )
        .await
        .unwrap();
        drop(storage);

        assert_eq!(value["status"], "compiled");
        assert_eq!(value["memoriesSelected"], 5);
        assert_eq!(value["memoriesReplayed"], 5);
        assert_eq!(value["edgesStrengthened"], 2);
        assert_eq!(value["edgesDownscaled"], 1);
        assert_eq!(value["contradictionsFound"], 1);
        assert_eq!(value["insightsGenerated"], 1);
        assert_eq!(value["prsFiled"].as_array().unwrap().len(), 0);
        assert_eq!(value["dreamHistoryRecorded"], false);
        let phases = value["phases"].as_array().unwrap();
        assert_eq!(
            phases
                .iter()
                .map(|phase| phase["phase"].as_str().unwrap())
                .collect::<Vec<_>>(),
            [
                "NREM1_Triage",
                "NREM3_Consolidation",
                "REM_Creative",
                "Integration"
            ]
        );
        // Text pairs "always"/"never" are not contradictions. Only the corrects edge is.
        assert_eq!(
            latest_milli(dir.path(), &ids[0], &ids[5], "derived_from"),
            700
        );
        assert_eq!(
            latest_milli(dir.path(), &ids[0], &ids[1], "derived_from"),
            380
        );
        assert_eq!(latest_milli(dir.path(), &ids[2], &ids[3], "corrects"), 900);

        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let again = execute(&storage, Some(serde_json::json!({"memory_count": 5})))
            .await
            .unwrap();
        drop(storage);
        assert_eq!(again["edgesStrengthened"], 2);
        assert_eq!(again["edgesDownscaled"], 1);
        assert_eq!(
            latest_milli(dir.path(), &ids[0], &ids[5], "derived_from"),
            800
        );
        assert_eq!(
            latest_milli(dir.path(), &ids[0], &ids[1], "derived_from"),
            361
        );
    }

    #[tokio::test]
    async fn strata_compile_needs_five_live_memories() {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let value = execute(&storage, None).await.unwrap();
        assert_eq!(value["status"], "insufficient_memories");
        assert_eq!(value["memoriesSelected"], 0);
        assert_eq!(value["phases"].as_array().unwrap().len(), 0);
        assert!(
            execute(&storage, Some(serde_json::json!({"memory_count": 4})))
                .await
                .is_err()
        );
    }
}
