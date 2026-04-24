//! Autopilot — v2.0.9 event-subscriber task.
//!
//! Subscribes to the shared `VestigeEvent` broadcast bus and routes every
//! live event into the cognitive modules that already have trigger methods
//! implemented. Without this layer, Vestige's 30 cognitive modules are a
//! passive library that only responds to MCP tool queries — the event bus
//! emits 20 event types but every one of them terminates at the dashboard.
//!
//! This module closes that gap. It turns Vestige from "fast retrieval with
//! neuroscience modules" into "self-managing cognitive surface that acts
//! without being asked." See `docs/VESTIGE_STATE_AND_PLAN.md` §15 for the
//! full architectural rationale.
//!
//! ## What fires autonomously after v2.0.9
//!
//! - **`MemoryCreated`**  → `synaptic_tagging.trigger_prp()` (9h retroactive
//!   PRP window on every save) + `predictive_memory.record_memory_access()`
//!   (pattern learning for `predict` tool).
//! - **`SearchPerformed`** → `predictive_memory.record_query()` (keeps the
//!   query-interest model warm without waiting for the next `predict` call).
//! - **`MemoryPromoted`** → `activation_network.activate()` (spreads a small
//!   reinforcement ripple from the promoted node to its neighbors).
//! - **`MemorySuppressed`** → emits the previously-declared-never-emitted
//!   `Rac1CascadeSwept` event so the dashboard can render the cascade wave.
//! - **`ImportanceScored` with `composite_score > 0.85`** → auto-`promote`
//!   when the score refers to a stored memory.
//! - **`Heartbeat` with `memory_count > DUPLICATES_THRESHOLD`** →
//!   opportunistic `find_duplicates` sweep (rate-limited).
//!
//! ## What polls on a timer
//!
//! A 60-second `tokio::interval` calls `prospective_memory.check_triggers()`
//! with the best context we can infer from recent WebSocket activity.
//! Matched intentions are logged at `info!` level today; v2.5 "Autonomic"
//! will promote this to MCP sampling/createMessage notifications that
//! actually reach the agent mid-session.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, broadcast};
use tracing::{debug, info, warn};
use vestige_core::Storage;
use vestige_core::neuroscience::prospective_memory::Context as ProspectiveContext;
use vestige_core::neuroscience::synaptic_tagging::{ImportanceEvent, ImportanceEventType};

use crate::cognitive::CognitiveEngine;
use crate::dashboard::events::VestigeEvent;

/// Composite-score threshold above which `ImportanceScored` auto-promotes
/// the referenced memory. Conservative default — tune in telemetry.
const AUTO_PROMOTE_THRESHOLD: f64 = 0.85;

/// Memory-count threshold above which a `Heartbeat` triggers a
/// `find_duplicates` sweep. Matches the CLAUDE.md guidance ("totalMemories > 700").
const DUPLICATES_THRESHOLD: usize = 700;

/// Minimum interval between autopilot-triggered `find_duplicates` sweeps,
/// regardless of Heartbeat cadence. Prevents sweep-storms when the count
/// hovers near the threshold.
const DUPLICATES_SWEEP_COOLDOWN_SECS: u64 = 6 * 3600; // 6 hours

/// Interval for polling `prospective_memory.check_triggers()`.
const PROSPECTIVE_POLL_SECS: u64 = 60;

/// Launch the Autopilot event-subscriber task + prospective-memory poller.
///
/// Both tasks live for the entire process lifetime and gracefully handle
/// broadcast lag (warn + resume) and closure (exit). The event loop holds
/// the `CognitiveEngine` mutex only for the duration of a single handler,
/// and never inside an `await`, so it never starves MCP tool dispatch.
pub fn spawn(
    cognitive: Arc<Mutex<CognitiveEngine>>,
    storage: Arc<Storage>,
    event_tx: broadcast::Sender<VestigeEvent>,
) {
    let rx = event_tx.subscribe();

    // Event-subscriber task — routes every emitted event into cognitive hooks.
    {
        let cognitive = cognitive.clone();
        let storage = storage.clone();
        let event_tx = event_tx.clone();
        tokio::spawn(async move {
            run_event_subscriber(rx, cognitive, storage, event_tx).await;
        });
    }

    // Prospective memory poller — a separate task so a long-running event
    // handler can't starve intention checks.
    {
        let cognitive = cognitive.clone();
        tokio::spawn(async move {
            run_prospective_poller(cognitive).await;
        });
    }

    info!("Autopilot spawned (event-subscriber + prospective poller)");
}

async fn run_event_subscriber(
    mut rx: broadcast::Receiver<VestigeEvent>,
    cognitive: Arc<Mutex<CognitiveEngine>>,
    storage: Arc<Storage>,
    event_tx: broadcast::Sender<VestigeEvent>,
) {
    // Last-time cache for Heartbeat-triggered auto-sweeps — prevents the
    // same 5-second heartbeat from firing a dedup sweep on every tick.
    let mut last_dedup_sweep: Option<Instant> = None;

    loop {
        match rx.recv().await {
            Ok(event) => {
                handle_event(
                    event,
                    &cognitive,
                    &storage,
                    &event_tx,
                    &mut last_dedup_sweep,
                )
                .await;
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                warn!("Autopilot lagged {n} events — increase channel capacity if this persists");
            }
            Err(broadcast::error::RecvError::Closed) => {
                info!("Autopilot event bus closed — subscriber exiting");
                break;
            }
        }
    }
}

async fn handle_event(
    event: VestigeEvent,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    storage: &Arc<Storage>,
    event_tx: &broadcast::Sender<VestigeEvent>,
    last_dedup_sweep: &mut Option<Instant>,
) {
    match event {
        VestigeEvent::MemoryCreated {
            id,
            content_preview,
            tags,
            timestamp,
            ..
        } => {
            // Synaptic tagging: every save is a CrossReference event candidate
            // for Frey & Morris 1997 PRP (retroactive importance within a 9h
            // window). The system dedups internally, so firing per-save is safe.
            let ev = ImportanceEvent {
                event_type: ImportanceEventType::CrossReference,
                memory_id: Some(id.clone()),
                timestamp,
                strength: 0.5,
                context: None,
            };
            let tag_outcome = {
                let mut cog = cognitive.lock().await;
                let outcome = cog.synaptic_tagging.trigger_prp(ev);
                // Predictive memory learns the ingested tags for pattern-match
                // against future `predict` queries. Method is `&self` (interior
                // RwLock), so we keep the cognitive mutex guard for ordering
                // but don't actually need &mut on this call.
                let _ = cog
                    .predictive_memory
                    .record_memory_access(&id, &content_preview, &tags);
                outcome
            };
            debug!(
                memory_id = %id,
                captured = ?tag_outcome,
                "Autopilot: MemoryCreated routed to synaptic_tagging + predictive_memory"
            );
        }

        VestigeEvent::SearchPerformed {
            query, result_ids, ..
        } => {
            // Feed the search into the predictive-retrieval model so the
            // speculative prefetch path warms up for the NEXT query. The
            // event doesn't carry per-result content, so we record with an
            // empty preview — the model only needs the id + tag signal.
            let cog = cognitive.lock().await;
            let empty_tags_str: [&str; 0] = [];
            let empty_tags_string: [String; 0] = [];
            let _ = cog.predictive_memory.record_query(&query, &empty_tags_str);
            for mid in result_ids.iter().take(10) {
                let _ = cog
                    .predictive_memory
                    .record_memory_access(mid, "", &empty_tags_string);
            }
            debug!(
                query = %query,
                n_results = result_ids.len(),
                "Autopilot: SearchPerformed routed to predictive_memory"
            );
        }

        VestigeEvent::MemoryPromoted { id, .. } => {
            // Spread a small activation ripple from the promoted node. The
            // ActivationNetwork internally handles decay (0.7/hop) so this
            // cannot over-amplify.
            let mut cog = cognitive.lock().await;
            let spread = cog.activation_network.activate(&id, 0.3);
            debug!(
                memory_id = %id,
                n_activated = spread.len(),
                "Autopilot: MemoryPromoted triggered activation spread"
            );
        }

        VestigeEvent::MemorySuppressed {
            id,
            estimated_cascade,
            timestamp,
            ..
        } => {
            // Surface the previously-declared-never-emitted Rac1CascadeSwept
            // event so the dashboard's cascade animation actually fires. The
            // per-suppress work happens synchronously inside `suppress_memory`
            // on the handler path; this is the observable shadow for the UI.
            let _ = event_tx.send(VestigeEvent::Rac1CascadeSwept {
                seeds: 1,
                neighbors_affected: estimated_cascade,
                timestamp,
            });
            debug!(
                memory_id = %id,
                cascade_size = estimated_cascade,
                "Autopilot: MemorySuppressed → Rac1CascadeSwept emitted"
            );
        }

        VestigeEvent::ImportanceScored {
            memory_id,
            composite_score,
            ..
        } => {
            // Auto-promote only when the score refers to a stored memory AND
            // exceeds the threshold. None means "score was computed for
            // arbitrary content via the importance tool" — nothing to promote.
            if let Some(mid) = memory_id
                && composite_score > AUTO_PROMOTE_THRESHOLD
            {
                match storage.promote_memory(&mid) {
                    Ok(node) => {
                        info!(
                            memory_id = %mid,
                            composite_score,
                            new_retention = node.retention_strength,
                            "Autopilot: auto-promoted memory with composite > {AUTO_PROMOTE_THRESHOLD}"
                        );
                        let _ = event_tx.send(VestigeEvent::MemoryPromoted {
                            id: node.id,
                            new_retention: node.retention_strength,
                            timestamp: chrono::Utc::now(),
                        });
                    }
                    Err(e) => {
                        warn!(
                            memory_id = %mid,
                            error = %e,
                            "Autopilot: auto-promote failed"
                        );
                    }
                }
            }
        }

        VestigeEvent::Heartbeat { memory_count, .. } => {
            if memory_count <= DUPLICATES_THRESHOLD {
                return;
            }
            let now = Instant::now();
            let cooldown_elapsed = last_dedup_sweep
                .map(|t| now.duration_since(t).as_secs() >= DUPLICATES_SWEEP_COOLDOWN_SECS)
                .unwrap_or(true);
            if !cooldown_elapsed {
                return;
            }
            *last_dedup_sweep = Some(now);

            // Fire the find_duplicates tool with conservative defaults.
            // Running on the heartbeat task keeps this off the critical
            // MCP-dispatch path. Result is logged only — the user's client
            // can still call the tool explicitly for an interactive run.
            let storage = storage.clone();
            tokio::spawn(async move {
                let args = serde_json::json!({
                    "similarity_threshold": 0.85,
                    "limit": 50,
                });
                match crate::tools::dedup::execute(&storage, Some(args)).await {
                    Ok(result) => {
                        let clusters = result
                            .get("duplicate_clusters")
                            .and_then(|v| v.as_array())
                            .map(|a| a.len())
                            .unwrap_or(0);
                        if clusters > 0 {
                            info!(
                                memory_count,
                                clusters,
                                "Autopilot: Heartbeat-triggered find_duplicates surfaced clusters"
                            );
                        }
                    }
                    Err(e) => {
                        warn!(
                            memory_count,
                            error = %e,
                            "Autopilot: Heartbeat-triggered find_duplicates failed"
                        );
                    }
                }
            });
        }

        // Events that carry no autopilot work today. Explicit pass-through so
        // adding a new event variant upstream produces a non_exhaustive_match
        // compiler nudge here.
        VestigeEvent::MemoryUpdated { .. }
        | VestigeEvent::MemoryDeleted { .. }
        | VestigeEvent::MemoryDemoted { .. }
        | VestigeEvent::MemoryUnsuppressed { .. }
        | VestigeEvent::Rac1CascadeSwept { .. }
        | VestigeEvent::DeepReferenceCompleted { .. }
        | VestigeEvent::DreamStarted { .. }
        | VestigeEvent::DreamProgress { .. }
        | VestigeEvent::DreamCompleted { .. }
        | VestigeEvent::ConsolidationStarted { .. }
        | VestigeEvent::ConsolidationCompleted { .. }
        | VestigeEvent::RetentionDecayed { .. }
        | VestigeEvent::ConnectionDiscovered { .. }
        | VestigeEvent::ActivationSpread { .. } => {}
    }
}

/// Background task that polls `prospective_memory.check_triggers()` every
/// `PROSPECTIVE_POLL_SECS` seconds. Today triggers are logged at info!
/// level; v2.5 "Autonomic" upgrades this to fire MCP sampling/createMessage
/// notifications so the agent sees intentions mid-conversation.
async fn run_prospective_poller(cognitive: Arc<Mutex<CognitiveEngine>>) {
    // Short delay on startup so hydration + other init settles first.
    tokio::time::sleep(Duration::from_secs(10)).await;

    let mut ticker = tokio::time::interval(Duration::from_secs(PROSPECTIVE_POLL_SECS));
    // Skip the immediate first tick that `interval` fires.
    ticker.tick().await;

    loop {
        ticker.tick().await;

        let context = ProspectiveContext {
            timestamp: chrono::Utc::now(),
            ..Default::default()
        };

        let triggered = {
            let cog = cognitive.lock().await;
            cog.prospective_memory.check_triggers(&context)
        };

        match triggered {
            Ok(intentions) if !intentions.is_empty() => {
                info!(
                    n_triggered = intentions.len(),
                    ids = ?intentions.iter().map(|i| i.id.as_str()).collect::<Vec<_>>(),
                    "Autopilot: prospective memory triggered intentions"
                );
                // v2.5 "Autonomic" will emit MCP sampling/createMessage here
                // so the agent actually sees the intention mid-conversation.
            }
            Ok(_) => {
                // No triggers — silent. This runs every 60s and the common
                // case is no work to do.
            }
            Err(e) => {
                warn!(error = %e, "Autopilot: prospective check_triggers failed");
            }
        }
    }
}
