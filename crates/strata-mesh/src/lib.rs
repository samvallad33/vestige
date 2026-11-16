//! Strata Mesh transport.
//!
//! The fold lives in `strata-store`. This crate moves chain bytes between two
//! stores and admits them with that fold. iroh supplies the QUIC crypto.
//! Nothing here mints a signature scheme, and a replica never sees `strata.key`.
//!
//! An unpaired store does not construct an endpoint. [`IrohSession::bind`] is
//! the only place a socket opens, and only a pair or a sync with peers calls it.

#![forbid(unsafe_code)]

mod pairing;
mod protocol;
mod scheduler;
mod transport;

pub use pairing::{
    accept_response, accept_ticket, finish_offer, joiner_message, load_peers, offer_ticket,
    respond_ticket, save_peers, test_confirm, ticket_pake, unpair, PairingOffer, PairingReply,
    PeerRecord, BINDING_DOMAIN,
};
pub use protocol::{
    apply_export, exchange, pack_export, serve, sync_over, ImportSide, MeshError, SyncReport,
};
pub use scheduler::{
    index_lies, schedule, segment_contains, Demand, IndexHint, MissingRange, ScheduleDecision,
};
pub use transport::{peer_addr, IrohSession, MemoryPipe, MeshTransport, RelaySetting};
