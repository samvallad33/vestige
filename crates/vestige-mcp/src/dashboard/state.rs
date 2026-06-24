//! Dashboard shared state

use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex, broadcast};
use tracing::warn;
use vestige_core::Storage;

use super::events::VestigeEvent;
use crate::cognitive::CognitiveEngine;
use subtle::ConstantTimeEq;

/// Broadcast channel capacity — how many events can buffer before old ones drop.
const EVENT_CHANNEL_CAPACITY: usize = 1024;

/// Shared application state for the dashboard
#[derive(Clone)]
pub struct AppState {
    pub storage: Arc<Storage>,
    pub cognitive: Option<Arc<Mutex<CognitiveEngine>>>,
    pub event_tx: broadcast::Sender<VestigeEvent>,
    pub start_time: Instant,
    dashboard_token: Arc<str>,
    dashboard_allowed_origins: Arc<Vec<String>>,
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
            dashboard_token: load_dashboard_token().into(),
            dashboard_allowed_origins: Arc::new(Vec::new()),
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
            dashboard_token: load_dashboard_token().into(),
            dashboard_allowed_origins: Arc::new(Vec::new()),
        }
    }

    /// Attach the exact origins allowed to drive the dashboard API and WS.
    pub fn with_dashboard_allowed_origins(mut self, origins: Vec<String>) -> Self {
        self.dashboard_allowed_origins = Arc::new(origins);
        self
    }

    /// Return true when a dashboard token matches the shared Vestige auth token.
    pub fn is_valid_dashboard_token(&self, token: &str) -> bool {
        let expected = self.dashboard_token.as_bytes();
        let candidate = token.as_bytes();
        candidate.len() == expected.len() && candidate.ct_eq(expected).unwrap_u8() == 1
    }

    /// Return true when an Origin header exactly matches the configured dashboard origins.
    pub fn is_allowed_dashboard_origin(&self, origin: &str) -> bool {
        self.dashboard_allowed_origins
            .iter()
            .any(|allowed| allowed == origin)
    }

    /// Percent-encode the dashboard token for use inside a URL fragment.
    pub fn dashboard_token_fragment_value(&self) -> String {
        percent_encode_fragment_value(self.dashboard_token.as_ref())
    }

    /// Emit an event to all connected clients.
    pub fn emit(&self, event: VestigeEvent) {
        // Ignore send errors (no receivers connected)
        let _ = self.event_tx.send(event);
    }
}

fn load_dashboard_token() -> String {
    match crate::protocol::auth::get_or_create_auth_token() {
        Ok(token) => token,
        Err(err) => {
            warn!(
                "Could not load persisted auth token for dashboard; using process-local token: {}",
                err
            );
            uuid::Uuid::new_v4().to_string()
        }
    }
}

fn percent_encode_fragment_value(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                encoded.push(byte as char);
            }
            _ => {
                encoded.push('%');
                encoded.push_str(&format!("{:02X}", byte));
            }
        }
    }
    encoded
}
