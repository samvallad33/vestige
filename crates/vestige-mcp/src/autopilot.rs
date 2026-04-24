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

/// Backoff between supervisor restarts after a panicked child task. Short
/// enough that a single bad memory doesn't meaningfully degrade the system,
/// long enough to avoid a tight crash loop if the panic source is persistent.
const SUPERVISOR_RESTART_BACKOFF_SECS: u64 = 5;

/// Tracks an in-flight Heartbeat-triggered dedup sweep so the next Heartbeat
/// can skip spawning a second sweep while the first is still running. The
/// previous implementation stored only the *start* time, which allowed two
/// concurrent scans on databases where `find_duplicates` exceeds the 6h
/// cooldown window.
struct DedupSweepState {
    last_fired: Option<Instant>,
    in_flight: Option<tokio::task::JoinHandle<()>>,
}

impl DedupSweepState {
    fn new() -> Self {
        Self {
            last_fired: None,
            in_flight: None,
        }
    }

    /// True if a previous sweep is still running. Drops a finished handle so
    /// a long-dead sweep doesn't keep us from firing the next one.
    fn is_running(&mut self) -> bool {
        match &self.in_flight {
            Some(h) if !h.is_finished() => true,
            _ => {
                self.in_flight = None;
                false
            }
        }
    }
}

/// Launch the Autopilot event-subscriber task + prospective-memory poller.
///
/// Both tasks are supervised: if the inner loop panics on a single bad
/// memory, the supervisor logs the panic and restarts it after a short
/// backoff. This turns a permanent silent-failure mode ("task dies, every
/// future cognitive event lost") into a transient hiccup ("one bad memory
/// skipped, subsystem resumes"). The event loop holds the `CognitiveEngine`
/// mutex only for the duration of a single handler, and never inside an
/// `await`, so it never starves MCP tool dispatch.
pub fn spawn(
    cognitive: Arc<Mutex<CognitiveEngine>>,
    storage: Arc<Storage>,
    event_tx: broadcast::Sender<VestigeEvent>,
) {
    // Opt-out: users upgrading in place from v2.0.8 may want to keep the
    // "passive library" contract. Set VESTIGE_AUTOPILOT_ENABLED=0 to skip
    // spawning both background tasks. Anything else (unset, "1", "true", etc.)
    // enables the default v2.0.9 Autopilot behavior.
    match std::env::var("VESTIGE_AUTOPILOT_ENABLED").as_deref() {
        Ok("0") | Ok("false") | Ok("no") | Ok("off") => {
            info!(
                "Autopilot disabled via VESTIGE_AUTOPILOT_ENABLED — \
                 cognitive modules remain passive (v2.0.8 behavior)"
            );
            return;
        }
        _ => {}
    }

    // Event-subscriber supervisor.
    {
        let cognitive = cognitive.clone();
        let storage = storage.clone();
        let event_tx = event_tx.clone();
        tokio::spawn(async move {
            loop {
                let rx = event_tx.subscribe();
                let cog = cognitive.clone();
                let sto = storage.clone();
                let etx = event_tx.clone();
                let handle = tokio::spawn(async move {
                    run_event_subscriber(rx, cog, sto, etx).await;
                });
                match handle.await {
                    Ok(()) => {
                        info!("Autopilot event subscriber exited cleanly");
                        break;
                    }
                    Err(e) if e.is_panic() => {
                        warn!(
                            error = ?e,
                            backoff_secs = SUPERVISOR_RESTART_BACKOFF_SECS,
                            "Autopilot event subscriber panicked — supervisor restarting"
                        );
                        tokio::time::sleep(Duration::from_secs(
                            SUPERVISOR_RESTART_BACKOFF_SECS,
                        ))
                        .await;
                    }
                    Err(e) => {
                        warn!(error = ?e, "Autopilot event subscriber join error — exiting");
                        break;
                    }
                }
            }
        });
    }

    // Prospective-memory poller supervisor — symmetric restart semantics.
    {
        let cognitive = cognitive.clone();
        tokio::spawn(async move {
            loop {
                let cog = cognitive.clone();
                let handle = tokio::spawn(async move {
                    run_prospective_poller(cog).await;
                });
                match handle.await {
                    Ok(()) => {
                        info!("Autopilot prospective poller exited cleanly");
                        break;
                    }
                    Err(e) if e.is_panic() => {
                        warn!(
                            error = ?e,
                            backoff_secs = SUPERVISOR_RESTART_BACKOFF_SECS,
                            "Autopilot prospective poller panicked — supervisor restarting"
                        );
                        tokio::time::sleep(Duration::from_secs(
                            SUPERVISOR_RESTART_BACKOFF_SECS,
                        ))
                        .await;
                    }
                    Err(e) => {
                        warn!(error = ?e, "Autopilot prospective poller join error — exiting");
                        break;
                    }
                }
            }
        });
    }

    info!("Autopilot spawned (event-subscriber + prospective poller, supervised)");
}

async fn run_event_subscriber(
    mut rx: broadcast::Receiver<VestigeEvent>,
    cognitive: Arc<Mutex<CognitiveEngine>>,
    storage: Arc<Storage>,
    event_tx: broadcast::Sender<VestigeEvent>,
) {
    // Tracks Heartbeat-triggered auto-sweeps so the next Heartbeat skips
    // spawning a second sweep while the first is still running — essential
    // on large DBs where `find_duplicates` can outrun the cooldown.
    let mut dedup_state = DedupSweepState::new();

    loop {
        match rx.recv().await {
            Ok(event) => {
                handle_event(
                    event,
                    &cognitive,
                    &storage,
                    &event_tx,
                    &mut dedup_state,
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
    dedup_state: &mut DedupSweepState,
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
            // If a prior sweep is still running (possible on very large DBs
            // where `find_duplicates` exceeds the 6h cooldown), skip this
            // tick rather than spawn a concurrent second scan.
            if dedup_state.is_running() {
                debug!(
                    memory_count,
                    "Autopilot: dedup sweep already in flight — skipping Heartbeat tick"
                );
                return;
            }
            let now = Instant::now();
            let cooldown_elapsed = dedup_state
                .last_fired
                .map(|t| now.duration_since(t).as_secs() >= DUPLICATES_SWEEP_COOLDOWN_SECS)
                .unwrap_or(true);
            if !cooldown_elapsed {
                return;
            }
            dedup_state.last_fired = Some(now);

            // Fire the find_duplicates tool with conservative defaults.
            // Running on the heartbeat task keeps this off the critical
            // MCP-dispatch path. Result is logged only — the user's client
            // can still call the tool explicitly for an interactive run.
            let storage = storage.clone();
            let handle = tokio::spawn(async move {
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
            dedup_state.in_flight = Some(handle);
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
