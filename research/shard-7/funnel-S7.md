# Shard 7 funnel

Base SHA `e2ecd1da037974c1835f563ad8b046f4a4ca3629`. Attached scoreboard `perq-bd2693e2` (`bin` `/workspace/c2-algos-bins/t51b-attr-unseated-opener-t10`, sha256 `e93411ac566c4c83f92a83204e70ea85ae743445a3256dc5cb4eacc2523e6f84`). That file is the Step 31 / pre-Step-33 board. On this branch Step 33 moves LoCoMo Turn@1 from 796 to 799 and Open-Domain Turn@1 from 18 to 19. Turn@5 stays 35. LME and BEAM are unchanged (C2_STEP33_LOG.md).

The per-question writer is not in this repo. Bit layouts were checked by summing the arrays against the header in the same JSON.

- LoCoMo 5-vector sums to `[1133, 1416, 796, 1127, 1226]`, which is `locomo.all` in order Session@1, Session@5, Turn@1, Turn@5, Turn@10.
- BEAM 7-vector sums to `[156, 238, 267, 118, 133, 249, 137]`, which is Turn@1, Turn@5, Turn@10, All@5, Session@1, Session@5, Session-All@5. It is not Session@10 and not All@10. Preference sums `[7, 14, 16, 11, 4, 14, 14]` match the preference header on those seven fields. Header fields `all10`, `s10`, and `sall10` are not in the per-question vector. Preference n is 39 because conversation 9 has `preference_following:0` and no `:1`.
- LME is not a primary slice here. Its header turn block is Any@1 350, Any@5 428, Any@10 437, All@5 314.

## What the dump fields are

`write_one_replay_dump` (`crates/vestige-core/src/retrieval.rs`, from line 7426) runs only when `VESTIGE_REPLAY_DUMP` is set. `maybe_dump_replay` returns before copying anything when the variable is unset. The dump is off the ranking path.

- `final_ids` / `seated_ids`: official order after the gatekeeper and every C2 seat, including Step 33. Rank is the index in that list.
- `pre_ids` (lines 7429-7440): the seated list first, then turns with `t_scene > 0` in scene order until 50. `rank_pre` is the index in `pre_ids`. `pre_ids[0]` is `final_ids[0]` by construction. Gold at pre_rank 0 is the same count as gold at final rank 0. It is not a pre-Kemeny rank and it is not scene rank.
- `t_scene` is the scene-pooled score. Seating does not rewrite it. Scene-rank 0 is the argmax of `t_scene`, tie broken by the lower index. For Open-Domain that count is 14, which is not pre_rank 0.
- There is no post-Kemeny snapshot. `rank1_gatekeeper_keep_rsb_head` Kemeny-orders slots 1..k with slot 0 frozen, then the C2 seats run. The dump sees the final seated list.
- `t_own`, `t_turn_raw`, `t_scene_sess`, `s_score`, and `bridge_boost` are copies of scores `rank_one` already computed. They are not read back into ranking. `t_own` is uncarried BM25 plus the contiguous-bigram bonus. `s_score + bridge_boost` is the official session rank score.
- `qtok_present` (from line 7490) and `turn_ic_novel` read `TurnIx.body`. They are dump-only. Ranking does not.

`tools/c2_funnel.py` is cited by C2_STEP33_LOG.md and is not in this checkout.

## Open-Domain, five lines

Attached JSON plus the Step 33 log's pool@50 on this same scoreboard. Miss-map stages are misses at Turn@1: `no_pool` 14, `pool_not_k` 35, `k_not5` 8, `t5_not1` 17, sum 74, and 92-18=74.

1. Gold in corpus: 92. C2_STEP33_LOG on the Step 31 binary says 92. The per-question file has 92 Open-Domain rows and no evidence ids, so the attachment alone cannot prove the evidence id is in the haystack. The log can.
2. Gold in pool@50: 63. The attachment cannot compute this per slice. Miss-map `no_pool` is absence from the broader pool (LoCoMo pool 1492, not pool@50 1415: the category `no_pool` counts sum to 44, which is 1536-1492). Open-Domain in the broader pool is 92-14=78. The 63 figure is the Step 33 log.
3. Gold at pre_rank 0: 18 on the attached board, 19 after Step 33. Equal to final rank 0 by the dump construction above.
4. Gold in top 5: 35. Unchanged by Step 33.
5. Gold at final rank 0: 18 attached, 19 on this branch.

Scene-rank 0 is 14 and is a different stage. Session@1 is 36 and Session@5 is 65. Those are not turn ranks.

A slot-1 reorder inside the current top 5 can reach 35/92 (38%). It cannot reach 83/92. Perfect use of pool@50 reaches 63/92 and is still 20 short. The 14 `no_pool` questions and part of the 35 `pool_not_k` questions need a recall or index change. Ordering rules in this shard are capped at the top-5 number. Recall rules are the ones that can move the ceiling.

## preference_following, five lines

Header: n=39, Turn@1 7, Turn@5 14, Turn@10 16, All@5 11, Session@1 4, Session@5 14, Session-All@5 14. Miss-map: `no_pool` 9, `pool_not_k` 14, `k_not5` 2, `t5_not1` 7, sum 32, and 39-7=32.

1. Gold in corpus: 39, inferred from BEAM corpus 355/355 in C2_STEP33_LOG. The attachment has no evidence ids, so it cannot show this by itself.
2. Gold in pool@50: not computable per slice. Broader pool is 30. BEAM pool is 329 and pool@50 is 328, so one question somewhere in BEAM is in the pool and not in pool@50. Preference pool@50 is 29 or 30.
3. Gold at pre_rank 0: 7. Equal to final rank 0 by construction. Step 33 does not change BEAM, so this branch is also 7.
4. Gold in top 5: 14.
5. Gold at final rank 0: 7.

Scene-rank 0 for this slice is not in the attachment. The log's BEAM scene-rank@1 is 124/355 for the whole bench, not for preference.

A slot-1 reorder can reach 14/39 (36%). It cannot reach 36/39. The broader pool holds 30, which is still 6 short of 36, so at least 6 of the 9 `no_pool` questions have to enter the pool, and nearly every retrieved gold has to be seated at slot 1. Many existing C2 seats return immediately on `c2_text_preference` (`algos_c2.rs` line 951). A new preference rule has to be gated by that existing test rather than by a new word list. `PREF_QUERY_CUES`, `HARD_CONSTRAINT_MARKERS`, and `constraint_object_tokens` at weight 1.8 are already in `retrieval.rs` and are not extended here.

## Hooks

- Ordering seat: `apply_c2_step33_triangulation` is called at `retrieval.rs` 6265-6272, and `maybe_dump_replay` begins at 6273. A new seat sits between those, after Step 31 (`apply_c2_step31_exchequer_split` at 6259) and after Step 33. The seat shape is `algos_c2.rs` 2543-2593. `C2_BULLAE_WINDOW` is 5 (`algos_c2.rs` line 7). Top-5 membership does not change, so Turn@5 cannot move.
- Index feats: the content-token loop is `retrieval.rs` 3121-3161. New inheritable feats belong there. `HYPERNYM_LATTICE` is not removed.
- Pool admission: submodular turn selection starts at `retrieval.rs` 5628. The candidate filter is `t_scene > 0` at line 5633. A recall rule that only reorders the top 5 cannot change pool@50. Admitting a turn with `t_scene == 0` is a change at that filter.
- `rsb_seat_behind` is called at line 6083 and defined at 6702. No rule in this shard edits it.

## What a slot-1 reorder cannot do

| Slice | Now at slot 1 (attached / this branch) | Top-5 ceiling | Pool ceiling | 90% target |
| --- | ---: | ---: | ---: | ---: |
| Open-Domain | 18 / 19 | 35/92 | 63/92 pool@50 | 83/92 |
| preference_following | 7 / 7 | 14/39 | 30/39 broader pool; pool@50 is 29 or 30 | 36/39 |

Recall and index rules are tagged `RECALL/INDEX`. Slot-1 rules that do not add pool members are tagged `ORDERING`.
