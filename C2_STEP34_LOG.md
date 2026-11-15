# C2 Step 34 log — Bertillonage / Locard point threshold

Branch `perf/c2-step34-bertillonage`, cut from `d121d063` (Step 3A). Applied `steps7_10a/*.patch`, then `algos44/*.patch`, then `wins/*.patch` with `git am -3`. Algo-44 tip on this branch is `b1e4509e` (the bd2693e2 tree: its bench matches the published board exactly). Combined Step 31 + Step 32 tip is `3345f79f`.

Author of the Step 34 commit: Sam Valladares `<samvalladares21@gmail.com>`.

Datasets verified before any ranker edit:

- LoCoMo sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- LongMemEval_S sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`

## Dump fields (read from `write_one_replay_dump` before any edit)

`pre_ids` is the seated turn list (the official final order) followed by a fill to 50 of turns with `t_scene > 0`, in `order()` (higher `t_scene`, then higher session timestamp, then earlier turn index). `candidates[].rank_pre` is that pool index. So `rank_pre == 0` is final rank 0 whenever the seated list is non-empty, and `pre_ids[:5]` is the final top 5 whenever at least five turns are seated. `final_ids` and `seated_ids` are the official turn order. `t_scene` is the scene score copied into the dump; the seat rules do not rewrite it. `candidates[].role` is `TurnIx.speaker`. `qtok_present` is a dump-only probe of the turn body and content ids. Ranking never reads `TurnIx.body`.

The informative pre-seat number is therefore argmax `t_scene` over the pool (scene rank), not `rank_pre`.

## Stage funnel, combined tip, before any Step 34 edit

Turn gold. Measured by replaying `VESTIGE_REPLAY_DUMP` on `/tmp/bin/retrieval_bench_tip` (`3345f79f`).

| Bench | n | corpus | pool@50 | pre_rank@5 | pre_rank @1 | final @5 (post-Kemeny) | final @1 | scene @5 | scene @1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LME_S | 470 | 470 | 463 | 428 | 352 | 428 | 352 | 399 | 265 |
| LoCoMo | 1536 | 1535 | 1415 | 1127 | 801 | 1127 | 801 | 1110 | 708 |
| BEAM-100K | 355 | 355 | 328 | 239 | 159 | 238 | 159 | 218 | 124 |

LME and LoCoMo `pre_rank@5` equal final @5, as the pool construction requires when the seated list has at least 5 turns. BEAM `pre_rank@5` is 239 and final @5 is 238: one question has a short seated list, so the t_scene fill occupies a pool slot inside the first five. One LoCoMo gold id is absent from that conversation's turn ids (corpus 1535/1536). Scene @1 is far below final @1 on every bench: the existing seats already lift gold above the raw scene leader. The remaining target is final top-5 but not #1: LME 76, LoCoMo 326, BEAM 79.

## Baseline scoreboard (measured, before the edit)

`b1e4509e` (bd2693e2 tree) versus `3345f79f` (Step 31 + Step 32 together). Per-question losses of the tip against bd2693e2: **0** on every coordinate.

| | bd2693e2 | tip 31+32 | tip minus bd |
| --- | --- | --- | --- |
| LME Session @1/@5/@10, All@5 | 430/463/468, 424 | same | 0 |
| LME Turn @1/@5/@10, All@5 | 350/428/437, 314 | 352/428/437, 314 | T@1 +2 |
| LoCoMo Turn @1/@5/@10, Sess @1/@5 | 796/1127/1226, 1133/1416 | 801/1127/1226, 1133/1416 | T@1 +5 |
| BEAM Turn @1/@5/@10, All@5, Sess @1/@5 | 156/238/267, 118, 133/249 | 159/238/267, 118, 133/249 | T@1 +3 |

Tip gains, and they are still hits after Step 34: LME `e8a79c70`, `c4a1ceb8`; LoCoMo `2:106`, `3:143`, `5:13`, `6:25`, `9:147`; BEAM `9:temporal_reasoning:1`, `11:knowledge_update:0`, `20:temporal_reasoning:1`.

## What the dumps showed

On the six questions the shipped rule flips, slot 1 is a question whose card is a strict subset of a same-session top-5 turn that is strictly more on-scene:

- LME `ccb36322` and BEAM `19:multi_session_reasoning:0`: slot 1 posts no query anchor and is not in the strongest session. The same-session turn that posts an anchor is gold.
- LoCoMo `1:42`, `5:105`, `9:9`, `9:36`: the query names a speaker. Slot 1 is the other speaker and posts an anchor. The named speaker's reply in that session posts the anchor and the speaker and outscores slot 1. A cross-session echo with a higher scene is the wrong witness.

Counting session inside the score, or promoting any richer card with no same-session and scene gate, demotes gold heads that paraphrase. Those structures were measured on the replay rows and not committed.

## Attempts

### Rejected on the replay grid (no ranker commit)

Same dump rows as the funnel (2,361 scored questions). A candidate replaced slot 1 only when slot 1 was under N, the candidate was inside the top 5, its kind set contained slot 1's, and its kind count met N. N was the number of query-side kinds present. Predicted per-question Turn @1:

| Structure | Predicted |
| --- | --- |
| Any/all/majority anchors, entities, numerals, with or without a session bit, unique or max | Best cells about +26/−58. No zero-loss cell. |
| Partitioned tokens (a token feeds one kind) plus "higher scene and same session", session bit inside the score | +2/−0, but only LME `ccb36322` and BEAM `19:multi_session_reasoning:0`. The four LoCoMo gains die because slot 1's session bit already meets N. |
| Question head, same speaker or the named speaker, no blank-card arm | +5/−0 (the LME question and four LoCoMo). Misses the BEAM gain. |
| Same gates with no question-head cut and no role cut | +6/−3 (`gpt4_194be4b3`, `73d42213`, LoCoMo `9:147`). |

`9:147` is the same shape as the LoCoMo gains except slot 1 is not a question, and the named-speaker turn is not gold. `gpt4_194be4b3` is a gold question with an empty anchor card sitting in the strongest session; the assistant echo is not gold. Both have to stay.

### Attempt 1 — shipped. WIN

Commit: the Step 34 commit on this branch (the only ranker change).

Files:

- `crates/vestige-core/src/algos_c2.rs` 3480–3709 (`apply_c2_step34_bertillonage` and the card helpers), test 7221–7363
- `crates/vestige-core/src/retrieval.rs` 6270–6273 (one call after `apply_c2_step32_trial_balance`)

Rule. Five kinds, each query token in at most one of the first four:

1. Numeral shape: the token contains an ASCII digit, and the turn's content posts that surface id.
2. Speaker: the query mentions an attested speaker name (not the `user` / `assistant` role), and the turn's speaker is that name.
3. Entity: a title-case proper token from `proper_entity_tokens` on the query, excluding speaker tokens, numerals, and closed-class words, posted in the turn's content.
4. Anchor: any other content token, posted in the turn's content. One hit is enough for the kind.
5. Session: the turn's session is the session of the strongest positive `t_scene` (the `order()` tie break).

N is how many of kinds 1–4 the query actually has in the vocabulary. Session is not part of N. Slot 1 is kept when its card's popcount is already at least N. A turn under N yields to the unique top-5 turn that meets N, strictly contains slot 1's kinds, shares slot 1's session, and has a strictly higher `t_scene`, and is admissible in one of two ways:

- Slot 1 is a question and the witness is not, and the witness is the same speaker or a speaker the query names.
- Slot 1's card is empty and slot 1 is not in the strongest session (the session measurement blocks this arm, which is what saves `gpt4_194be4b3`).

Two witnesses with the same popcount are not an identification. The top-5 set does not change, so Turn @5/@10 and All@5 cannot move. The session list is not rewritten. `rsb_seat_behind` is not touched. No turn body is read.

Funnel for this attempt. Pool membership and `t_scene` are untouched, and the scoreboard shows Turn @5 and every session coordinate unchanged, so pool@50, scene @1, and scene @5 stay on the table above. Final @5 stays 428 / 1127 / 238. Final @1 (and therefore `pre_rank` @1, because the pool is seated-first) becomes the Step 34 Turn @1 column: LME 353, LoCoMo 805, BEAM 160.

Scoreboard versus the tip: see `SCOREBOARD_STEP34.md`. Gains: LME `ccb36322`; LoCoMo `1:42`, `5:105`, `9:9`, `9:36`; BEAM `19:multi_session_reasoning:0`. Losses: 0. These are the six questions the replay grid predicted.

## Gates

- `cargo test -p vestige-core`: 562 passed, 0 failed. Includes `step34_bertillonage_yields_to_the_richer_same_session_card`.
- `cargo clippy -p vestige-core --all-targets -- -D warnings`: clean.
- `cargo check -p vestige-mcp`: clean.
- `c2_hashcheck.py` on the release `retrieval_bench`: ranked ids 0/500, full responses 0/500.
- LME Session: 430/463/468 and All@5 424, identical to the tip and to bd2693e2. Per-question session coordinates: +0/−0.
