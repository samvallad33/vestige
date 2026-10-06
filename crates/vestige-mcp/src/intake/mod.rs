//! Pure intake modules for the proof-carrying write path (INGEST V5, Lane C).
//!
//! THE LAW: every function in this module tree is a pure function of bytes.
//! No clocks, no randomness, no network, no storage access, no globals, no
//! HashMap-iteration-order-dependent output. Same input bytes in, same
//! output bytes out, forever. All outputs are version-pinned —
//! [`entities::EXTRACTOR_VERSION`] for entity extraction and
//! [`importance::WEIGHTS_VERSION`] for importance scoring; any observable
//! behavior change requires a new version string, never a silent edit.

pub mod entities;
pub mod importance;
