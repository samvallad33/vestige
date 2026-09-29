//! v4.0.0 Strata release gate.
//!
//! Every test is a real process. A feature that is not on this head fails
//! with a message that starts `MISSING:` and names the absent piece. Nothing
//! is skipped, ignored, or weakened to go green.

mod cli;
mod engine;
mod mcp;
mod support;
mod upgrade;
