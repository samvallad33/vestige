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
//! `rank_one` (`order(&t_scene, &turn_ts).take(16)`). Those 16 candidates
//! are the head's cross-session neighbors: policy backlinks, supersession,
//! exact edges, exact name/handle tokens, and one turn of a
//! `session_episode_prev` session, in that same scene order. Adjacent
//! turns already in the pool share one edge, so a below-window answer
//! beside a named turn stays on that turn's island. The local RSB reply
//! stays in the degree ballot and is not a second island edge. The token
//! session range is the ingest-time exact-handle gate (`2..=6`) and the
//! header-anchor session degree cap (`1..=6`). A token is the capitalized
//! form on that turn. The known-proper backfill is not an identity edge.
//!
//! Entry, fixed before the v-next card and not fit to a slice. The seated
//! tail is ordered by the gatekeeper without this vote. A deep turn is
//! seated only when its margin against every seated tail turn is strictly
//! positive on the same five ballots (scene, structural degree, stability,
//! wh-slot, island). Qualifiers are ordered by Kemeny among themselves and
//! evict from the bottom of the tail. When none qualify, the tail stays in
//! that baseline order. The island ballot is one of the five votes. It is
//! not a second pass over the window.

use std::sync::atomic::{AtomicU64, Ordering};

/// Cross-session neighbors of the head considered for a tail seat.
/// A neighbor is seated only when it beats every seated tail turn.
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
