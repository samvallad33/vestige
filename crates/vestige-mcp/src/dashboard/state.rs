//! Dashboard shared state

use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex, broadcast, watch};
use vestige_core::Storage;

use super::events::VestigeEvent;
use crate::cognitive::CognitiveEngine;

/// Broadcast channel capacity — how many events can buffer before old ones drop.
// 4096 events of headroom before a slow dashboard subscriber lags. A lagging
// subscriber is told how many events it missed (see websocket.rs) instead of
// silently resuming mid-stream.
pub const EVENT_CHANNEL_CAPACITY: usize = 4096;
// Public so the two servers that build their own channel (the MCP binary
// and `vestige-cli serve`) cannot drift away from the dashboard's value:
// a hardcoded 4096 beside a tunable constant silently stops matching the
// moment anyone tunes it.

/// Shared application state for the dashboard
#[derive(Clone)]
pub struct AppState {
    pub storage: Arc<Storage>,
    pub cognitive: Option<Arc<Mutex<CognitiveEngine>>>,
    pub event_tx: broadcast::Sender<VestigeEvent>,
    pub start_time: Instant,
    /// Flips to `true` once when this dashboard stops serving: the listener
    /// closes and every open WebSocket ends.
    stopped: Arc<watch::Sender<bool>>,
}

impl AppState {
    /// Create a new AppState with event broadcasting.
    pub fn new(storage: Arc<Storage>, cognitive: Option<Arc<Mutex<CognitiveEngine>>>) -> Self {
        let (event_tx, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        Self::with_event_tx(storage, cognitive, event_tx)
    }

    /// Get a new event receiver (for WebSocket connections).
    pub fn subscribe(&self) -> broadcast::Receiver<VestigeEvent> {
        self.event_tx.subscribe()
    }

    /// Create a new AppState sharing an external event broadcast channel.
    pub fn with_event_tx(
        storage: Arc<Storage>,
        cognitive: Option<Arc<Mutex<CognitiveEngine>>>,
        event_tx: broadcast::Sender<VestigeEvent>,
    ) -> Self {
        Self {
            storage,
            cognitive,
            event_tx,
            start_time: Instant::now(),
            stopped: Arc::new(watch::channel(false).0),
        }
    }

    /// Stop serving: the listener closes and open WebSockets end.
    pub fn stop(&self) {
        self.stopped.send_replace(true);
    }

    /// Resolves once [`AppState::stop`] has been called (at once if it was).
    pub async fn stopped(&self) {
        let mut stopped = self.stopped.subscribe();
        // `wait_for` only errs when the sender is gone, and `self` holds it.
        let _ = stopped.wait_for(|stopped| *stopped).await;
    }

    /// Emit an event to all connected clients.
    pub fn emit(&self, event: VestigeEvent) {
        // Ignore send errors (no receivers connected)
        let _ = self.event_tx.send(event);
    }
}
