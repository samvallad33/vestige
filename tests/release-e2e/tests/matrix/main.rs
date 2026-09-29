//! v4.0.0 Strata release gate.
//!
//! Every test is a real process. A feature that is not on this head fails
//! with a message that starts `MISSING:` and names the absent piece. Nothing
//! is skipped or weakened to go green. The one ignored row is
//! `purge_canary_absent_from_segment_bytes` (`deferred_4_1`): segment-byte
//! erasure is not a 4.0 assertion and that test is not run.

mod cli;
mod engine;
mod gate;
mod mcp;
mod support;
mod upgrade;
