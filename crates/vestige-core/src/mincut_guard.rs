//! Min-cut episode islands. Every number here was fixed before the card.
//!
//! Sources, not a test-set search:
//!
//! - L. R. Ford and D. R. Fulkerson, "Maximal flow through a network",
//!   Canadian Journal of Mathematics 8 (1956). A cut of value 0 is a
//!   disconnection. Connectivity 1 is one unit edge.
//! - Mechthild Stoer and Frank Wagner, "A simple min-cut algorithm",
//!   Journal of the ACM 44(4), 1997. The global min-cut used to certify
//!   each island.
//! - R. E. Gomory and T. C. Hu, "Multi-terminal network flows", Journal of
//!   the SIAM 9(4), 1961. The all-pairs cut tree. Islands here are the
//!   sets whose Stoer-Wagner cut stays at least 1, which is the
//!   connectivity encoded by a Gomory-Hu tree after every 0-weight tree
//!   edge is removed.
//! - NetworkX `stoer_wagner` (the documented 8-vertex example has cut value
//!   4): <https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.connectivity.stoerwagner.stoer_wagner.html>
//! - LEMON `MinCut` / Gomory-Hu: <https://lemon.cs.elte.hu/pub/doc/1.3.1/a00616.html>
//!
//! The deep-pool width is the turn-level PPR seed width already in
//! `rank_one` (`order(&t_scene, &turn_ts).take(16)`). Those 16 seats are
//! the head's cross-session neighbors: policy backlinks, supersession,
//! exact edges, exact name/handle tokens, and one turn of a
//! `session_episode_prev` session, in that same scene order. Adjacent
//! turns are not edges. The local RSB reply stays in the degree ballot
//! and is not a second island edge. The token session range is the
//! ingest-time exact-handle gate (`2..=6`) and the header-anchor session
//! degree cap (`1..=6`). A token is the capitalized form on that turn.
//! The known-proper backfill is not an identity edge.

use std::sync::atomic::{AtomicU64, Ordering};

/// Turns taken from below the post-RSB window and placed in the tail pool.
pub const MINCUT_DEEP_POOL: usize = 16;

/// Session cardinality of a capitalized name or a handle that may add one
/// unit identity edge. Below 2 the token never leaves its session. Above 6
/// it is a hub, the same ceiling as the handle gate and the header anchors.
pub const MINCUT_TOKEN_SESSION_DF_MIN: u32 = 2;
pub const MINCUT_TOKEN_SESSION_DF_MAX: u32 = 6;

/// Recursive Stoer-Wagner stops splitting when the global min-cut reaches
/// this value. 1 means a single unit edge keeps the component one island.
pub const MINCUT_ISLAND_CONNECTIVITY: i64 = 1;

static FIRES: AtomicU64 = AtomicU64::new(0);

/// One query whose tail differed from the gatekeeper without this vote.
pub fn record_fire() {
    FIRES.fetch_add(1, Ordering::Relaxed);
}

/// Fires since process start. The bench prints this once, on exit.
pub fn fires() -> u64 {
    FIRES.load(Ordering::Relaxed)
}
