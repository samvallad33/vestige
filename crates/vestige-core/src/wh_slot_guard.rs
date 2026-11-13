//! Wh-slot tail ballot. Locked before the Step 1 card. Nothing here was fit to a score.
//!
//! Four ballots, each at most one vote: scene, structural degree, stability,
//! and the wh residual. Scene is a ballot and the tie-break (a zero margin
//! keeps the input order). Wh is one vote, +1, -1, or 0. On a query with no
//! wh class that vote is 0 and the other three ballots still run.
//!
//! WH_NONE keeps the PR 715 ballots: scene is the sign of the score gap,
//! scaled to one ballot, and structural degree still counts `rsb_antecedent`
//! so the WH_NONE list stays byte-identical to 715. On a closed wh class,
//! scene is the continuous relative margin
//!
//! `round(1000 * (scene_a - scene_b) / max(scene_a, scene_b))`
//!
//! with a zero denominator yielding 0. A near-tie casts a small scene vote,
//! so the wh residual can break it. A large t_scene gap still casts nearly
//! one full ballot and blocks deg=1 noise.
//!
//! Closed-class structural degree is the PR 715 / WH_NONE degree, including
//! `rsb_antecedent` (the dump field `degree_none`). That is the offline 217
//! stack's locked substitution: bit G plus `degree_none` on every class.
//! Graph-only degree (dropping antecedent) was not part of the 217 card.
//!
//! The wh weight is the scarcity of the slot inside the seated window, not
//! the deeper pool. This is the same formula locked in WhV6, kept here:
//!
//! `round(1000 * ln((n + 1) / (p + 1)) / ln(n + 1))`
//!
//! clamped to 0..=1000. This is 1000 when no seated turn passes and 0 when
//! every seated turn passes. Two of the other ballots beat wh. A 2-1 of those
//! ballots is not overturned when wh joins the dissenter. The continuous
//! scene margin, not a change to this weight, is what lets wh win a near-tie.
//!
//! `rsb_seat_behind` runs on the first k turns with the caller's k, the same
//! call PR 715 makes. A full window does not grow. Only after that seat does
//! the ranker append positive-score turns, deduped, up to `GATEKEEPER_POOL`.
//! Those deeper turns never pass through `rsb_seat_behind`. A query with no
//! wh class does not admit them, so its list is the 715 window.
//!
//! A below-window turn is seated only when it passes the novel residual, its
//! t_scene is at least `WH_CLOSED_SCENE_FLOOR` of the last seated turn, and
//! its four-ballot margin is strictly positive against at least one seated
//! tail turn (beats_worst). The returned tail keeps the seated length. Slot
//! 0 is the RSB head and stays there. There is no second pass and no swap.
//! `wh_guarded_swap` stays off.
//!
//! Pair binding (choice, locked here): after `seat_wh_tail` rebuilds a
//! closed-class window, the ranker re-applies `rsb_seat_behind` to that
//! window with the same k. Kemeny-Young may reorder the tail, and this
//! placement puts each cause back immediately after its symptom, the same
//! way PR 715 seats a cause. The head is never moved ahead of slot 0.
//! WH_NONE does not re-apply, so its list stays the 715 order. This was
//! picked over treating the pair as a Kemeny super-node, because the seat
//! function is already the 715 binding and cannot undo RSB.
//!
//! The class order is how long, then how many or how much, then when, then
//! who, whom, or whose, then where. Anything else is no class. A residual is
//! a character form read after the query's own tokens are blanked. When is a
//! 4-digit calendar year in `WH_YEAR_MIN`..=`WH_YEAR_MAX`, an ISO date, a
//! slash date, or a digit-colon clock. How long is a digit or that clock.
//! How many is a digit. Who and where take a novel TitleCase name or an
//! @handle, URL, or path. A TitleCase name is length at least
//! `WH_MIN_NAME_LEN`, first letter upper, the rest lower. Mid-sentence names
//! always count. A sentence-initial name counts only when the conversation
//! index never attests that token in lowercase (`df_lower == 0`) and it is
//! not in the query's excluded tokens. All-caps tokens are rejected. The
//! session timestamp is not a clock.
//!
//! Later steps, formulas locked before any card. They are not wired in the
//! Step 1 commit.
//!
//! - Step 2: `IC_novel(t \ Q)` is the sum of `ln((N+1)/(df(w)+1))` over
//!   tokens in the turn that are not in the query. When the RSB head has
//!   lower `IC_novel` than its same-session `response_turn` or `rsb_cause`,
//!   and that neighbor passes `wh_residual`, the neighbor is seated at
//!   slot 1. The head stays in slot 0. No extra threshold.
//! - Step 3A (wired in the 217 stack): on `is_event_sequence` only, two
//!   equal milli-votes join the ballot, both scaled by `WH_MARGIN_SCALE`:
//!   proximity `sign(|ts_b - ts_anchor| - |ts_a - ts_anchor|)` and order
//!   `sign(ts_a - ts_b)`. The anchor is the RSB head's session timestamp.
//!   If the anchor or either turn's session timestamp is <= 0, both votes
//!   are 0. `WH_WHEN` alone does not fire this. Digit-shape clocks stay on
//!   the wh residual and are not counted a second time.
//! - Slot-4 session replacement and idea 18 anti-monopoly (wired in the
//!   217 stack): after the tail is seated, if slot 4's session already
//!   appears in slots 0-3, slot 4 is replaced by the highest-t_scene
//!   non-superseded positive-scene turn from an unseen session in the
//!   50-turn pool union per-session best turns, and the displaced turn
//!   moves to slot 5. Then, when that pool spans at least
//!   `WH_ANTIMONOPOLY_MIN_SESSIONS` sessions, any session is capped at
//!   `WH_SESSION_CAP_TOP5` turns in the top 5 (slot 0 is kept). Freed
//!   slots are filled from next-best unseen sessions. The RSB head stays
//!   in slot 0.
//! - Step 4: `temporal_quantile_tail`. Slot 0 stays the RSB head. The best
//!   positive-t_scene, non-superseded turn is taken from each session except
//!   the head's, then the cumulative t_scene energy is split into `k-1`
//!   equal-mass quantiles and the highest t_scene turn in each quantile is
//!   seated. Applied only when `is_event_sequence` is set or the query is
//!   multi-session by structure (more than one session in the seated window
//!   before the tail pass). This is used instead of the `1/(1+c_s)` session
//!   redundancy penalty. Bipartite IC diffusion (Fix 4) is not built.
//! - Step 5: a 1-hop walk over `exact_edges` and `policy_backlinks` only.
//!   No co-occurrence, no bipartite token graph, no IC diffusion.
//! - Step 6: `certify_premise_coverage` returns VerifiedLock, Supported,
//!   PartialPremise, or Abstain. Coverage is
//!   `Cov(Q, R) = sum_{w in Q and R} IC(w)^2 / sum_{w in Q} IC(w)^2` over
//!   the top turns plus each of their `response_turn`s.
//!   Abstain if the highest-IC query token has `df == 0`.
//!   PartialPremise if that token is missing from R_k, or if `cov_k` is
//!   below `WH_COV_PARTIAL` (0.35). VerifiedLock if `cov_1` is at least
//!   `WH_COV_LOCK` (0.85). Otherwise Supported. Those two cutoffs were
//!   fixed by Sam before any card, not estimated on a test set.
//!
//! Provenance, fixed before the run. None of these constants was fit to
//! LoCoMo, LME, BEAM, or MAB.
//!
//! - Scarcity weight: Sparck Jones, "A statistical interpretation of term
//!   specificity and its application in retrieval," Journal of Documentation
//!   28(1), 1972. Normalized by `ln(n+1)` so the weight is one ballot.
//!   Manning, Raghavan, and Schütze, Introduction to Information Retrieval,
//!   Cambridge, 2008, chapter 6.
//! - Continuous scene margin: a parameter-free relative difference, the
//!   same normalization that maps a unique lead onto one ballot. No
//!   threshold is estimated from a test set.
//! - Sentence-initial names: the conversation's own lowercase document
//!   frequency. No starter list and no stopword list.
//! - Calendar year range: the 19xx and 20xx century digits, character
//!   geometry on a 4-digit token. Not a month or weekday list.
//! - Order: Kemeny, "Mathematics without numbers," Daedalus 88(4), 1959, and
//!   Young, "Condorcet's Theory of Voting," American Political Science Review
//!   82(4), 1988. Exact branch-and-bound through 8!. Equal scores keep the
//!   RSB v4 order. Cause-symptom binding is the RSB v4 `rsb_seat_behind`
//!   placement, reapplied after the tail order on closed wh classes.
//! - Below-window entry: Condorcet pairwise majority against the worst
//!   seated tail turn, the suffix-automaton v2 admission rule, the novel
//!   residual, and the t_scene floor.
//! - `GATEKEEPER_POOL` 50, `WH_UNINFORMATIVE` 0.65, `WH_MARGIN_SCALE` 1000,
//!   `WH_MIN_NAME_LEN` 3, `WH_SCENE_FLOOR` 0.50, `WH_YEAR_MIN` 1900,
//!   `WH_YEAR_MAX` 2099: locked before the card. Not estimated on a test set.
//! - `WH_CLOSED_SCENE_FLOOR` 0.85, `WH_SESSION_CAP_TOP5` 2, and
//!   `WH_ANTIMONOPOLY_MIN_SESSIONS` 3: locked from the offline 217 BEAM
//!   stack (idea 24 closed-class floor, then idea 18). Not estimated on a
//!   test set. `WH_MARGIN_SCALE` is also the Step 3A vote scale.
//! - RSB uses the caller's k, as in PR 715 (`5abd62a0`).

/// Deeper pool appended after RSB seats into k. The returned list stays k.
pub const GATEKEEPER_POOL: usize = 50;

/// When true, WH_HOW_MANY uses the PR 715 ballots: sign scene, degree
/// including rsb_antecedent, no residual vote, no below-window admission,
/// and no second rsb_seat_behind. Locked from the offline factorial
/// (bit G). Not estimated on a test set.
pub const WH_HOW_MANY_FOLLOWS_715: bool = true;

/// When true, closed-class order_pool locks each (symptom, cause) pair
/// present in the list as an adjacent supernode (symptom then cause),
/// first symptom claims the cause, skipped when the cause already sits
/// at or before the symptom in the input list. Never applied on a 715
/// ballot class (WH_NONE and, when WH_HOW_MANY_FOLLOWS_715, WH_HOW_MANY).
/// The offline 217 stack used only bit G (no bit B), so this stays false
/// and pair binding is left to the post-window `rsb_seat_behind` pass,
/// matching the replayer. Not estimated on a test set.
pub const WH_CAUSE_BIND_IN_ORDER_POOL: bool = false;


/// A pass rate above this carries no information, so the wh ballot is withheld.
pub const WH_UNINFORMATIVE: f64 = 0.65;

/// One ballot, in milli-votes. Scene, degree, and stability use this weight.
/// The wh ballot uses [`wh_weight_milli`], which is at most this value.
/// Step 3A also uses this scale for the proximity and order votes
/// (`1000 * sign(...)`) locked from the offline 3A / 217 stack. Not
/// estimated on a test set.
pub const WH_MARGIN_SCALE: i64 = 1000;

/// Minimum length of a novel TitleCase name. All-caps tokens fail the
/// "rest lower" test on their own. Fixed before the card.
pub const WH_MIN_NAME_LEN: usize = 3;

/// A below-window candidate must have at least this fraction of the last
/// seated turn's t_scene. The original lock, kept for WH_NONE documentation.
/// Closed-class admission uses `WH_CLOSED_SCENE_FLOOR`.
pub const WH_SCENE_FLOOR: f32 = 0.50;

/// Closed-class below-window admission floor. Offline idea 24
/// (`24_closed_085`) on the 217 BEAM stack: Stage B raised Any@5 from 215
/// to 216 with this floor on closed classes only. WH_NONE seating is
/// unchanged (no admission). Not estimated on a test set.
pub const WH_CLOSED_SCENE_FLOOR: f32 = 0.85;

/// Idea 18 anti-monopoly: cap any session at this many turns in the top 5.
/// Slot 0 is always kept. Locked from the Stage B card that raised Any@5
/// from 216 to 217. Not estimated on a test set.
pub const WH_SESSION_CAP_TOP5: usize = 2;

/// Idea 18 fires only when the 50-turn candidate pool spans at least this
/// many distinct sessions. Locked from the Stage B card (217). Not
/// estimated on a test set.
pub const WH_ANTIMONOPOLY_MIN_SESSIONS: usize = 3;

/// Inclusive lower bound of a 4-digit calendar year. Character geometry
/// on 19xx. Fixed before the card.
pub const WH_YEAR_MIN: u32 = 1900;

/// Inclusive upper bound of a 4-digit calendar year. Character geometry
/// on 20xx. Fixed before the card.
pub const WH_YEAR_MAX: u32 = 2099;

/// Scarcity of a wh slot, in milli-votes, capped at one ballot.
///
/// `n` is the seated window. `passing` is how many of those turns carry the
/// novel residual. Parameter-free: 1000 when `passing` is 0, 0 when `passing`
/// is `n`. This is the WhV6 formula. The continuous scene margin is what
/// lets a residual break a near-tie; this weight is not retuned.
pub fn wh_weight_milli(n: usize, passing: usize) -> i64 {
    if n == 0 {
        return 0;
    }
    let denom = (n as f64 + 1.0).ln();
    if denom == 0.0 {
        return 0;
    }
    let numer = ((n as f64 + 1.0) / (passing as f64 + 1.0)).ln();
    let raw = (WH_MARGIN_SCALE as f64 * numer / denom).round() as i64;
    raw.clamp(0, WH_MARGIN_SCALE)
}

/// Parameter-free relative scene margin, in milli-votes.
///
/// `round(1000 * (scene_a - scene_b) / max(scene_a, scene_b))`. A non-positive
/// denominator yields 0, so two empty scores are a tie. Used only on a closed
/// wh class. WH_NONE keeps the sign ballot from PR 715.
pub fn scene_margin_milli(scene_a: f32, scene_b: f32) -> i64 {
    let denom = scene_a.max(scene_b);
    if denom <= 0.0 {
        return 0;
    }
    (WH_MARGIN_SCALE as f64 * (scene_a as f64 - scene_b as f64) / denom as f64).round() as i64
}

/// Information content of a token from the conversation's own counts.
/// Locked for Step 2 (`IC_novel`) and Step 6 (`Cov`). Not used in Step 1.
pub fn token_ic(n: usize, df: usize) -> f64 {
    ((n as f64 + 1.0) / (df as f64 + 1.0)).ln()
}

/// Temporal proximity used as an equal vote in Step 3.
/// Locked before the card. Not used in Step 1.
pub fn temporal_proximity(ts: i64, anchor: i64) -> f64 {
    1.0 / (1.0 + (ts - anchor).unsigned_abs() as f64)
}

/// Coverage floor for PartialPremise in Step 6. Fixed by Sam before any card.
/// Not estimated on LoCoMo, LME, BEAM, or MAB. Not used in Step 1.
pub const WH_COV_PARTIAL: f64 = 0.35;

/// Coverage floor for VerifiedLock in Step 6. Fixed by Sam before any card.
/// Not estimated on LoCoMo, LME, BEAM, or MAB. Not used in Step 1.
pub const WH_COV_LOCK: f64 = 0.85;
