//! Extension point for chronological event-ordering passes.

use super::context::RankCtx;

/// Applies event-ordering passes after session late-sync and diversity coverage.
pub(in crate::retrieval) fn apply_event_ordering_stack(_ctx: &mut RankCtx<'_>) {}
