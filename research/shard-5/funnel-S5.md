# Shard 5 funnel

Base SHA `e2ecd1da037974c1835f563ad8b046f4a4ca3629`. Attached scores are bin bd2693e2 (perq JSON `t51b-attr-unseated-opener-t10`). That scoreboard is the Step 33 baseline: LoCoMo Turn 796/1127/1226 and BEAM Turn 156/238/267. Step 33, which is the checked-out HEAD, moved LoCoMo Open-Domain Turn@1 from 18 to 19 and left BEAM byte-identical. The ceilings below use the attached bin and then say what HEAD changed.

The brief's "19/92" is the post-Step-33 count. The attached JSON's Open-Domain array is 18 hits. 18/92 is 19.57%, which is why the rate is sometimes written 19. The hit count on the attached bin is 18. HEAD is 19. Top-5, top-10, and pool@50 did not move.

## What the dump fields mean

Read from `maybe_dump_replay` (retrieval.rs 7262-7297) and `write_one_replay_dump` (7426-7538) on this SHA.

- `final_ids` / `seated_ids` / `rank`: the official turn order after the gatekeeper and every seat that ran, including Step 33. `maybe_dump_replay` stores `seated` from `final_turn_indices` and returns immediately when `VESTIGE_REPLAY_DUMP` is unset. The dump is not on the ranking path.
- `pre_ids` / `rank_pre`: `pre_ids` is that seated list, then turns with `t_scene > 0` in scene order until 50. `rank_pre` is the index in `pre_ids`. For every seated turn, `rank_pre` equals `rank`. Gold at pre_rank 0 is therefore the same count as gold at final rank 0. There is no separate pre-Kemeny snapshot in the dump.
- `t_scene`: the scene-pooled turn score on the snapshot. Seats do not rewrite it. Scene-rank 0 is the argmax.
- `t_own`: the turn's own lexical score. `t_turn_raw` and `t_scene_sess` are the other score copies. `s_score + bridge_boost` is the session rank score. None of these are read back into ranking.
- `qtok_present` reads `TurnIx.body` and is dump-only. Ranking does not read `body`. On this SHA `body` is empty unless the dump variable is set (retrieval.rs 3379-3383).

Miss-map stages, reconciled to the headers:

- `no_pool`: gold not in the scored pool. Bench arithmetic: LoCoMo 1536-1492=44, BEAM 355-329=26, and the category sums match.
- `pool_not_k`: in the scored pool, not in official k (top 10).
- `k_not5`: in official k, not in the top 5.
- `t5_not1`: in the top 5, not at rank 0.
- `pool@50`: gold inside `pre_ids`. It is a subset of the scored pool. The 77 LoCoMo golds and the 1 BEAM gold that are in the scored pool but outside pool@50 sit inside `pool_not_k`, because anything in official k is inside pool@50 (the dump fills pool@50 with the seated list first).

Per-question JSON layouts, checked by summing to the category totals:

- BEAM 7-tuple `[t1, t5, t10, all5, s1, s5, sall5]`. `all10` and `s10` are not in the vector. event_ordering is `[7, 14, 22, 0, 7, 22, 0]`.
- LoCoMo 5-tuple `[s1, s5, t1, t5, t10]`. Keys are `conv:qid` with no category label, so Open-Domain question ids cannot be listed from the JSON. The category totals are under `locomo.cats.Open-Domain`.

## Open-Domain

LoCoMo Open-Domain n=92. Attached JSON cats: s1=36, s5=65, t1=18, t5=35, t10=43. Miss map: pool_not_k 35, t5_not1 17, no_pool 14, k_not5 8 (18+74=92). gold in corpus: 92, from tools/c2_funnel.py in C2_STEP33_LOG on this same scoreboard (the JSON has no corpus field; bench LoCoMo corpus is 1535/1536 because one other question's evidence id is absent, and that miss is not this slice). gold in pool@50: 63, from that same per-slice funnel. The attached miss map gives only the bench pool@50 of 1415, so the slice figure is the log, not the JSON. gold at pre_rank 0: 18, equal to final rank 0 because pre_ids is seated-first. gold in top5: 35. gold at final rank 0: 18 on the attached bin. Checked-out Step 33 moved this slice from 18 to 19 and did not change top5, t10, or pool@50. Scored pool = 92-14 = 78, of which 63 are inside pool@50 (20 of those 63 are outside official k) and 15 are scored but outside pool@50. Slot-1 ceiling 35/92. Official-k ceiling 43/92. pool@50 ceiling 63/92. Scored-pool ceiling 78/92. 78 is below 83, so 90% cannot be reached by ordering or by pool fill. At least 5 of the 14 no_pool golds must gain a match, and the pooled misses must then be seated. The attached data does not say whether a no_pool gold has zero token overlap or only a dropped date, ordinal, or header token.

Five-line funnel on the attached bin:

1. gold in corpus: 92
2. gold in pool@50: 63
3. gold at pre_rank 0: 18 (equals final rank 0; not an independent stage)
4. gold in top5: 35
5. gold at final rank 0: 18 (HEAD: 19)

Stages the attached JSON and miss map cannot compute by themselves: per-slice pool@50, per-slice corpus, and per-question Open-Domain ids. Pool@50 63 and corpus 92 are taken from `C2_STEP33_LOG.md`, whose baseline Turn@1/Turn@5/Turn@10 match this JSON. Scene-rank 0 for the slice is 14 in that log; the brief's five lines do not ask for it.

What a slot-1 reorder can reach: 35/92 if it may permute the current top 5, 43/92 if it may permute official k, 63/92 if it may pull any pool@50 gold into slot 1, 78/92 if it may pull any scored-pool gold. All four are below 83/92.

What needs recall/index changes: at least 5 of the 14 no_pool golds. Those golds are indexed and have no scored-pool membership, so a selector that only sees `t_scene > 0` never considers them. Session@1 is 36 against Turn@1 of 18, so on many hits the right session is already known and the wrong turn is seated. That part is ordering. It cannot, even if perfect, pass 78/92.

Open-Domain questions are not `is_event_sequence`. The C2 seat stack runs on them, including Step 33, and the misses remain. New Open-Domain ordering has to be a different integer gate, not another score-ratio swap. The Step 33 log's pairwise census found no clean score swap.

## Event ordering

BEAM event_ordering n=40 (attached miss map and perq array [7,14,22,0,7,22,0]). gold in corpus: bench BEAM corpus is 355/355 in C2_STEP33_LOG, so this slice's golds are indexed; the 1 no_pool gold is unscored, not a missing document. gold in pool@50: 38 or 39. Bench pool is 329 and pool@50 is 328; that one gold sits outside the dump's 50 and the attached data does not say which BEAM category owns it. gold at pre_rank 0: 7. write_one_replay_dump builds pre_ids as the seated list first, so pre_rank 0 equals final rank 0 by construction and is not a pre-Kemeny rank. gold in top5: 14. gold at final rank 0: 7. Slot-1 reorder of the current top 5 ceilings at 14/40. Reorder of official k ceilings at 22/40. Reselection inside the scored pool ceilings at 39/40, which can reach 36/40. Turn@1 of 90% therefore requires the event-sequence selector to admit scored-pool golds into k. A post-list swap of the seated top 5 cannot. New index features are required only for the 1 no_pool gold.

Five-line funnel:

1. gold in corpus: indexed (bench 355/355). The slice-level corpus flag is not a separate column; `no_pool=1` is the unscored case.
2. gold in pool@50: 38 or 39. Not separable from the attached data.
3. gold at pre_rank 0: 7 (equals final rank 0)
4. gold in top5: 14
5. gold at final rank 0: 7

All@5 is 0/40 and the published All@10 is 0/40. These questions have more than one gold turn. The goal used here is Turn@1, meaning at least one gold at rank 0, not every gold inside the top 5.

`is_event_sequence` is set in `classify` (retrieval.rs 3969-3972) by a closed list of order phrases. Almost every C2 seat is inside `if !it.is_event_sequence` (6017, and `rsb_seat_behind` at 6082-6083). Step 33 at 6261-6272 is outside that guard and returns immediately unless the dialogue is named-speaker, which BEAM user/assistant chats are not. An event-ordering seat has to run when `is_event_sequence` is true.

The misses that decide 90%:

- `pool_not_k` 17 and `k_not5` 8 are already scored. The turn selector's `remaining` list is every turn with `t_scene > 0` (5633). They lose the seat to `topic_repeat` (jaccard at least 0.45, lines 5647-5651), to `same_date_seen` redundancy (5521-5533), to the one-per-session cap while the top 5 is filling (`max_per_session_top5 = 1`, lines 5619-5625 and 5675-5679), or to the episode boost pulling a neighbor (5458-5471). Changing those integer gates is ordering. Changing the constants 0.45, 0.25, 0.15, 0.52 is a forbidden retune and is not what these designs do.
- `t5_not1` 7 can be fixed by a seat at 6261-6272 and cannot by itself reach 36/40.
- `no_pool` 1 needs an index feature.

Official k for the slice is 22 (t10). 22+17 pool_not_k = 39 scored. 39+1 no_pool = 40.

## Other slices

This shard does not retarget Multi-Hop, instruction_following, summarization, or preference_following. A date-head or ordinal feature can move them only as a side effect. Their ceilings were not recomputed beyond the miss-map header.
