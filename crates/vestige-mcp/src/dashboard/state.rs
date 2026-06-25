//! Dashboard shared state

use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex, broadcast};
use vestige_core::Storage;

use super::events::VestigeEvent;
use crate::cognitive::CognitiveEngine;

/// Broadcast channel capacity — how many events can buffer before old ones drop.
const EVENT_CHANNEL_CAPACITY: usize = 1024;

/// Shared application state for the dashboard
#[derive(Clone)]
pub struct AppState {
    pub storage: Arc<Storage>,
    pub cognitive: Option<Arc<Mutex<CognitiveEngine>>>,
    pub event_tx: broadcast::Sender<VestigeEvent>,
    pub start_time: Instant,
}

impl AppState {
    /// Create a new AppState with event broadcasting.
    pub fn new(storage: Arc<Storage>, cognitive: Option<Arc<Mutex<CognitiveEngine>>>) -> Self {
        let (event_tx, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        Self {
            storage,
            cognitive,
            event_tx,
            start_time: Instant::now(),
        }
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
        }
    }

    /// Emit an event to all connected clients.
    pub fn emit(&self, event: VestigeEvent) {
        // Ignore send errors (no receivers connected)
        let _ = self.event_tx.send(event);
    }
}
