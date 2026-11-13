//! Steiner-bridge parameters. Frozen before any benchmark of this version.
//!
//! Every edge the solver sees has cost 1. Nothing here is a rank weight, a
//! bonus, or a veto. Changing a value is a new version and has to land in a
//! commit before that version is benchmarked.

/// A token that sits on fewer than two turns is not an edge.
/// Same lower bound as the ingest-time exact-handle postings
/// (`2..=6` in `build_index`).
pub const STEINER_TOKEN_DF_MIN: u32 = 2;

/// Digits and technical handles (`\u{1}h:`) above this document frequency are
/// not edges. Copied from the ingest-time exact-handle window `2..=6`.
/// A calendar year or a repeated URL would otherwise clique the window and
/// the tree would have no Steiner node.
/// Capitalized-name tokens are not capped by this constant. Speaker-field
/// identities are dropped instead, because a recurring proper name is the
/// multi-hop bridge and a speaker name is structural metadata.
pub const STEINER_HUB_DF_MAX: u32 = 6;

/// `is_content` already drops one-character tokens, so a one-digit string
/// is not a stored content id. Two is the shortest digit token the index
/// can actually hold.
pub const STEINER_DIGIT_MIN_CHARS: usize = 2;

/// Compute bound on below-window turns admitted to the graph.
/// Admission order is structural (a turn that touches two terminals the
/// direct graph has not already joined, head-touching first, then turn
/// index). It is not a score.
/// 10 terminals + 128 candidates = 138 nodes, which sits under
/// [`STEINER_EXACT_MAX_NODES`].
pub const STEINER_BELOW_CAP: usize = 128;

/// Dreyfus-Wagner is exact and is used when both this and
/// [`STEINER_EXACT_MAX_NODES`] hold. 3^12 = 531441. Past that the
/// component uses the Kou-Markowsky-Berman distance-network heuristic.
/// Benchmarks ask for top_k 10, so the card runs the exact solver.
pub const STEINER_EXACT_MAX_TERMINALS: usize = 12;

/// Dreyfus-Wagner node bound. Pairs with [`STEINER_BELOW_CAP`]:
/// a top-10 window plus the below cap is 138 nodes.
pub const STEINER_EXACT_MAX_NODES: usize = 160;

/// A length-2 candidate is seated only when it is the only below-window
/// turn joined to that disconnected terminal pair by a stored edge or an
/// exact token.
///
/// v1 broke the tie by turn index. On the measured card that seated an
/// arbitrary neighbor: LoCoMo changed 1547 queries, core turn R@5 moved
/// 1067/1536 (69.47%) to 1047/1536 (68.16%), Multi-Hop stayed 138/282,
/// Open-Domain moved 36/92 to 31/92, and LongMemEval turn Any@5 moved
/// 376/470 to 375/470. An index is not a vote.
///
/// Session membership and equal timestamps still connect two terminals, so
/// a same-session pair needs no bridge. They do not qualify a below-window
/// turn as touching a terminal. Counting that co-session edge as a touch
/// made every pair look supported and the unique-pair rule fired 0 times
/// on BEAM and on MemoryAgentBench.
pub const STEINER_UNIQUE_PAIR: bool = true;

/// The seated bridge's session is absent from the window.
///
/// v3 (unique stored/token pair, co-session edge ignored) fired on 713
/// LoCoMo queries and moved Multi-Hop turn R@5 from 138/282 to 135/282 and
/// core turn R@5 from 1067/1536 (69.47%) to 1057/1536 (68.82%). The bridge
/// usually lived in a session the window already held, so the insert
/// evicted a top-5 turn without adding a session. A bridge is new evidence
/// only when its session is not already a terminal.
pub const STEINER_NEW_SESSION: bool = true;

/// Do not push the only top-5 turn of a session out of the top 5.
///
/// Inserting at seat s < 5 shifts the old index 4 to index 5. v4 fired
/// 259 LoCoMo queries and moved Multi-Hop turn R@5 from 138/282 to
/// 137/282 and core turn R@5 from 1067/1536 (69.47%) to 1060/1536
/// (69.01%). The turn that left the top 5 was often the only representative
/// of its session. A duplicate-session turn may leave the top 5. A sole
/// representative stays.
pub const STEINER_KEEP_SOLE_SESSION: bool = true;

/// Non-terminal Steiner nodes seated per query.
/// One node, the one nearest the frozen head on the tree.
/// Seating it directly behind that head (slot 1 when the head is an
/// endpoint) swaps a single below-window turn into the window. The head
/// stays. A second seat would evict another top-5 member on a k=10 list.
pub const STEINER_MAX_BRIDGES: usize = 1;
