# Shard 8 funnel and per-metric ceilings

Base SHA `e2ecd1da037974c1835f563ad8b046f4a4ca3629`. Dump writer `write_one_replay_dump` in `crates/vestige-core/src/retrieval.rs` lines 7426-7621. Rank path: `build_index` 2969-3872, `plan_query` 3991, `rank_one` 4490, session submodular 5480-5599, `max_per_session_top5` 5618-5625, turn submodular ending 5781, post-gatekeeper seats through Step 33 triangulation 6261-6272, `maybe_dump_replay` 6273. `rsb_seat_behind` is at 6702 and is not a hook for any design in this shard.

Numbers are from `perq-bd2693e2` unless a cell is marked step 34. Step 34 (`62041ef5`) published bench Turn@1 only: LME 353, LoCoMo 805, BEAM 160. It did not publish a new Turn@5, Turn@10, or per-category split. Category Turn@1 below is the bd2693e2 value. Need is `ceil(0.9 * n)`.

## What the dump fields are

`pool` in the replay JSON starts as `snap.seated` and then fills with `t_scene>0` turns up to 50. `pre_rank` is the index in that pool. `final_rank` is the index in `snap.seated`. `final_ids` equals `seated_ids`. Therefore `pre_ids[0] == final_ids[0] ==` the seated head. "Gold at pre_rank 0" and "gold at final rank 0" are the same bit. They are Turn@1. A reorder hook cannot make those two lines differ.

The dump also writes `t_scene`, `t_own`, `t_turn_raw`, `s_score`, `bridge_boost`, `session_order`, `role`, `is_question`, and `qtok_present`. The dump path reads `ix.turns[ti].body` for `qtok_present`. Ranking hooks in this shard do not. They read index token stats, masks, form flags, `list_item_count`, `form_clock`, `form_digit`, `form_midcap`, and the answers edge.

`TurnIx.body` is dump-only (struct field around 1898-1900). `Query` carries `text` and `date`.

A rank_one hook can see every turn with `t_scene>0`. The dump truncates observation to 50. The operative in-engine ceiling for a pool selection is the positive pool, not the dumped 50. Where the attachment gives only bench pool@50, the category split of `pool_not_k` into inside-50 and outside-50 is not computable. The table writes that interval as `unknown in [t10, pool]`.

## Bit layouts, checked by summing perq

LME, 7 bits `[s1, s5, s10, t1, t5, t10, ta5]`, sums `[430, 463, 468, 350, 428, 437, 314]`. Session All@5=424 and Session All@10=447 are aggregate-only. There is no LME Turn All@10 bit.

LoCoMo, 5 bits `[s1, s5, t1, t5, t10]`, sums `[1133, 1416, 796, 1127, 1226]`. Keys are `conversation:question`, not category. No s10 bit. No Turn All@10 bit. Missmap header All@5=954 is not in the bits; it is an extra row below.

BEAM, 7 bits `[t1, t5, t10, all5, s1, s5, sall5]`, sums `[156, 238, 267, 118, 133, 249, 137]`. `all10`, `s10`, and `sall10` are on the category object and are not in the 7-bit vector. Missmap `k=266` and `T@10=267` differ by one; ceilings use T@10.

## Stage words

`no_pool`: the primary gold is outside the scored pool. `pool_not_k`: in the pool, not in the kept list. `k_not5`: in the kept list, not in top 5. `t5_not1`: in top 5, not in slot 1. For every category that has its own missmap line, `n - t1` equals the sum of those four counts. That identity was checked while this file was generated. BEAM `contradiction_resolution` has no own stage line. Its `t10` equals `n` (40), so every gold is inside the top 10 and the pool@50 ceiling is 40 by that fact, not by a stage line.

## Five-line funnels

### LME bench
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=470.
2. gold in pool@50: 463 (missmap header).
3. gold at pre_rank 0: 350. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 428.
5. gold at final rank 0: 350. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
Step 34 Turn@1 is 353. Category Turn@1 below stays at bd2693e2 because the step-34 per-category split is not attached. Session All@5=424 and Session All@10=447 are aggregate-only; they are not in the 7 perq bits.

### LME single-session-user
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=64.
2. gold in pool@50: not separable; bounds [63, 64] because t10 <= pool@50 <= pool.
3. gold at pre_rank 0: 54. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 62.
5. gold at final rank 0: 54. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'t5_not1': 8, 'pool_not_k': 1, 'k_not5': 1}. Turn All@5=62. No category Turn All@10 bit.

### LME multi-session
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=121.
2. gold in pool@50: not separable; bounds [113, 121] because t10 <= pool@50 <= pool.
3. gold at pre_rank 0: 88. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 113.
5. gold at final rank 0: 88. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'t5_not1': 25, 'pool_not_k': 8}. Turn All@5=53. No category Turn All@10 bit.

### LME single-session-preference
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=30.
2. gold in pool@50: not separable; bounds [19, 30] because t10 <= pool@50 <= pool.
3. gold at pre_rank 0: 12. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 19.
5. gold at final rank 0: 12. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'pool_not_k': 11, 't5_not1': 7}. Turn All@5=14. No category Turn All@10 bit.

### LME temporal-reasoning
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=127.
2. gold in pool@50: not separable; bounds [116, 125] because t10 <= pool@50 <= pool.
3. gold at pre_rank 0: 93. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 111.
5. gold at final rank 0: 93. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'pool_not_k': 9, 't5_not1': 18, 'k_not5': 5, 'no_pool': 2}. Turn All@5=74. No category Turn All@10 bit.

### LME knowledge-update
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=72.
2. gold in pool@50: not separable; bounds [72, 72] because t10 <= pool@50 <= pool.
3. gold at pre_rank 0: 62. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 71.
5. gold at final rank 0: 62. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'t5_not1': 9, 'k_not5': 1}. Turn All@5=59. No category Turn All@10 bit.

### LME single-session-assistant
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=56.
2. gold in pool@50: not separable; bounds [54, 56] because t10 <= pool@50 <= pool.
3. gold at pre_rank 0: 41. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 52.
5. gold at final rank 0: 41. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'t5_not1': 11, 'k_not5': 2, 'pool_not_k': 2}. Turn All@5=52. No category Turn All@10 bit.

### LoCoMo bench
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=1536.
2. gold in pool@50: 1415 (missmap header).
3. gold at pre_rank 0: 796. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 1127.
5. gold at final rank 0: 796. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
Step 34 Turn@1 is 805. Missmap All@5=954 is not a perq bit. No LoCoMo s10 bit and no Turn All@10 bit. k header=1226 equals T@10.

### LoCoMo Multi-Hop
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=282.
2. gold in pool@50: not separable; bounds [186, 269].
3. gold at pre_rank 0: 97. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 158.
5. gold at final rank 0: 97. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'k_not5': 28, 'no_pool': 13, 't5_not1': 61, 'pool_not_k': 83}. No category All@5 bit.

### LoCoMo Temporal
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=321.
2. gold in pool@50: not separable; bounds [260, 313].
3. gold at pre_rank 0: 187. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 248.
5. gold at final rank 0: 187. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'pool_not_k': 53, 't5_not1': 61, 'k_not5': 12, 'no_pool': 8}. No category All@5 bit.

### LoCoMo Open-Domain
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=92.
2. gold in pool@50: not separable; bounds [43, 78].
3. gold at pre_rank 0: 18. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 35.
5. gold at final rank 0: 18. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'pool_not_k': 35, 't5_not1': 17, 'no_pool': 14, 'k_not5': 8}. No category All@5 bit.

### LoCoMo Single-Hop
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=841.
2. gold in pool@50: not separable; bounds [737, 832].
3. gold at pre_rank 0: 494. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 686.
5. gold at final rank 0: 494. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'t5_not1': 192, 'pool_not_k': 95, 'k_not5': 51, 'no_pool': 9}. No category All@5 bit.

### BEAM bench
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=355.
2. gold in pool@50: 328 (missmap header).
3. gold at pre_rank 0: 156. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 238.
5. gold at final rank 0: 156. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
Step 34 Turn@1 is 160. Missmap k=266 and T@10=267 differ by 1; the funnel uses T@10=267. Pool=329.

### BEAM contradiction_resolution
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=40.
2. gold in pool@50: 40, derived because t10=n so every gold is inside top 10 and therefore inside pool@50. The missmap has no separate stage line for this category..
3. gold at pre_rank 0: 32. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 40.
5. gold at final rank 0: 32. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
no stage line of its own. Turn All@5=22, Turn All@10=27.

### BEAM event_ordering
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=40.
2. gold in pool@50: not separable; bounds [22, 39].
3. gold at pre_rank 0: 7. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 14.
5. gold at final rank 0: 7. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'t5_not1': 7, 'pool_not_k': 17, 'k_not5': 8, 'no_pool': 1}. Turn All@5=0, Turn All@10=0.

### BEAM information_extraction
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=40.
2. gold in pool@50: not separable; bounds [29, 32].
3. gold at pre_rank 0: 20. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 27.
5. gold at final rank 0: 20. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'t5_not1': 7, 'pool_not_k': 3, 'no_pool': 8, 'k_not5': 2}. Turn All@5=20, Turn All@10=21.

### BEAM instruction_following
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=40.
2. gold in pool@50: not separable; bounds [21, 35].
3. gold at pre_rank 0: 6. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 18.
5. gold at final rank 0: 6. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'t5_not1': 12, 'pool_not_k': 15, 'no_pool': 5, 'k_not5': 2}. Turn All@5=13, Turn All@10=16.

### BEAM knowledge_update
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=40.
2. gold in pool@50: not separable; bounds [39, 39].
3. gold at pre_rank 0: 30. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 37.
5. gold at final rank 0: 30. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'t5_not1': 7, 'k_not5': 2, 'no_pool': 1}. Turn All@5=24, Turn All@10=27.

### BEAM multi_session_reasoning
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=40.
2. gold in pool@50: not separable; bounds [36, 40].
3. gold at pre_rank 0: 21. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 34.
5. gold at final rank 0: 21. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'t5_not1': 13, 'pool_not_k': 4, 'k_not5': 2}. Turn All@5=4, Turn All@10=7.

### BEAM preference_following
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=39.
2. gold in pool@50: not separable; bounds [16, 30].
3. gold at pre_rank 0: 7. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 14.
5. gold at final rank 0: 7. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'k_not5': 2, 'pool_not_k': 14, 't5_not1': 7, 'no_pool': 9}. Turn All@5=11, Turn All@10=13.

### BEAM summarization
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=36.
2. gold in pool@50: not separable; bounds [25, 34].
3. gold at pre_rank 0: 6. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 17.
5. gold at final rank 0: 6. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'t5_not1': 11, 'k_not5': 8, 'pool_not_k': 9, 'no_pool': 2}. Turn All@5=0, Turn All@10=0.

### BEAM temporal_reasoning
1. gold in corpus: not computable. The attachment has no corpus column. no_pool means outside the scored pool, not proven absent from the index. n=40.
2. gold in pool@50: not separable; bounds [39, 40].
3. gold at pre_rank 0: 27. pre_ids[0] is the seated head by construction of write_one_replay_dump (retrieval.rs 7426-7621), so this line equals Turn@1.
4. gold in top5: 37.
5. gold at final rank 0: 27. final_ids[0] is the same seated head, so line 5 equals line 3. They cannot differ in this dump.
stages {'t5_not1': 10, 'k_not5': 2, 'pool_not_k': 1}. Turn All@5=24, Turn All@10=27.


## Per-metric ceiling table

`ceiling_slot1_or_set` is the measured top-5 value for Turn@1 and Session@1 (a reorder of who occupies slot 1), and the metric's own current value for every set metric (Turn@5, Turn@10, All@5, All@10, Session@5, Session@10, Session All@*). Reordering a set does not change the set.

`ceiling_top10` is Turn@10 for turn-slot metrics, Session@10 where that bit exists, and All@10 where that field exists. LoCoMo has no s10 and no Turn All@10. LME has no Turn All@10.

`ceiling_pool50` is the missmap header at bench grain. Per category it is an interval, except BEAM contradiction_resolution, where t10=n forces pool@50=n.

`ceiling_pool` is `n - no_pool` where the stage line exists. It is the primary gold's pool, not the pool of every member of an All-set. All@5 and All@10 therefore have an unknown pool ceiling: the header "pool" is the primary turn gold.

`shortfall` uses step-34 Turn@1 on the three bench Turn@1 rows and the bd2693e2 value everywhere else. A shortfall of 0 means the measured value already meets `ceil(0.9*n)`.

`class` is the cheapest ceiling that can reach the need. It is a ceiling class, not a claim that a design in this shard attains it.

| bench | row | metric | n | now_bd2693e2 | now_step34 | need | shortfall | ceiling_slot1_or_set | slot_reaches_90 | ceiling_top10 | top10_reaches_90 | ceiling_pool50 | pool50_reaches_90 | ceiling_pool | pool_reaches_90 | class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LME | BENCH | Session@1 | 470 | 430 |  | 423 | 0 | 463 | yes | 468 | yes | unknown | unknown | unknown | unknown | session-rank reorder inside the measured session top 5 |
| LME | BENCH | Session@5 | 470 | 463 |  | 423 | 0 | 463 | yes | 468 | yes | unknown | unknown | unknown | unknown | already met; reordering the current set does not raise it |
| LME | BENCH | Session@10 | 470 | 468 |  | 423 | 0 | 468 | yes | 468 | yes | unknown | unknown | unknown | unknown | already met; the reorder ceiling is the current value |
| LME | BENCH | Session All@5 | 470 | 424 |  | 423 | 0 | 424 | yes | 447 | yes | unknown | unknown | unknown | unknown | already met; reordering slot 1 does not change the set |
| LME | BENCH | Session All@10 | 470 | 447 |  | 423 | 0 | 447 | yes | 447 | yes | unknown | unknown | unknown | unknown | already met; the reorder ceiling is the current value |
| LME | BENCH | Turn@1 | 470 | 350 | 353 | 423 | 70 | 428 | yes | 437 | yes | 463 | yes | 468 | yes | slot-1 reorder inside the measured top 5 |
| LME | BENCH | Turn@5 | 470 | 428 |  | 423 | 0 | 428 | yes | 437 | yes | 463 | yes | 468 | yes | already met; reordering the current set does not raise it |
| LME | BENCH | Turn@10 | 470 | 437 |  | 423 | 0 | 437 | yes | 437 | yes | 463 | yes | 468 | yes | already met; the reorder ceiling is the current value |
| LME | BENCH | Turn All@5 | 470 | 314 |  | 423 | 109 | 314 | no | unknown | unknown | unknown | unknown | unknown | unknown | set ceiling past the measured value is unknown (no All@10 bit, and full-set pool membership is not in the attachment) |
| LME | single-session-user | Turn@1 | 64 | 54 |  | 58 | 4 | 62 | yes | 63 | yes | unknown in [63, 64] | yes | 64 | yes | slot-1 reorder inside the measured top 5 |
| LME | single-session-user | Turn@5 | 64 | 62 |  | 58 | 0 | 62 | yes | 63 | yes | unknown in [63, 64] | yes | 64 | yes | already met; reordering the current set does not raise it |
| LME | single-session-user | Turn@10 | 64 | 63 |  | 58 | 0 | 63 | yes | 63 | yes | unknown in [63, 64] | yes | 64 | yes | already met; the reorder ceiling is the current value |
| LME | single-session-user | Turn All@5 | 64 | 62 |  | 58 | 0 | 62 | yes | unknown | unknown | unknown | unknown | unknown | unknown | already met; reordering slot 1 does not change the set |
| LME | single-session-user | Session@1 | 64 | 62 |  | 58 | 0 | 64 | yes | 64 | yes | unknown | unknown | unknown | unknown | session-rank reorder inside the measured session top 5 |
| LME | single-session-user | Session@5 | 64 | 64 |  | 58 | 0 | 64 | yes | 64 | yes | unknown | unknown | unknown | unknown | already met; reordering the current set does not raise it |
| LME | single-session-user | Session@10 | 64 | 64 |  | 58 | 0 | 64 | yes | 64 | yes | unknown | unknown | unknown | unknown | already met; the reorder ceiling is the current value |
| LME | multi-session | Turn@1 | 121 | 88 |  | 109 | 21 | 113 | yes | 113 | yes | unknown in [113, 121] | yes | 121 | yes | slot-1 reorder inside the measured top 5 |
| LME | multi-session | Turn@5 | 121 | 113 |  | 109 | 0 | 113 | yes | 113 | yes | unknown in [113, 121] | yes | 121 | yes | already met; reordering the current set does not raise it |
| LME | multi-session | Turn@10 | 121 | 113 |  | 109 | 0 | 113 | yes | 113 | yes | unknown in [113, 121] | yes | 121 | yes | already met; the reorder ceiling is the current value |
| LME | multi-session | Turn All@5 | 121 | 53 |  | 109 | 56 | 53 | no | unknown | unknown | unknown | unknown | unknown | unknown | set ceiling past the measured value is unknown (no All@10 bit, and full-set pool membership is not in the attachment) |
| LME | multi-session | Session@1 | 121 | 110 |  | 109 | 0 | 121 | yes | 121 | yes | unknown | unknown | unknown | unknown | session-rank reorder inside the measured session top 5 |
| LME | multi-session | Session@5 | 121 | 121 |  | 109 | 0 | 121 | yes | 121 | yes | unknown | unknown | unknown | unknown | already met; reordering the current set does not raise it |
| LME | multi-session | Session@10 | 121 | 121 |  | 109 | 0 | 121 | yes | 121 | yes | unknown | unknown | unknown | unknown | already met; the reorder ceiling is the current value |
| LME | single-session-preference | Turn@1 | 30 | 12 |  | 27 | 15 | 19 | no | 19 | no | unknown in [19, 30] | unknown | 30 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| LME | single-session-preference | Turn@5 | 30 | 19 |  | 27 | 8 | 19 | no | 19 | no | unknown in [19, 30] | unknown | 30 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| LME | single-session-preference | Turn@10 | 30 | 19 |  | 27 | 8 | 19 | no | 19 | no | unknown in [19, 30] | unknown | 30 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| LME | single-session-preference | Turn All@5 | 30 | 14 |  | 27 | 13 | 14 | no | unknown | unknown | unknown | unknown | unknown | unknown | set ceiling past the measured value is unknown (no All@10 bit, and full-set pool membership is not in the attachment) |
| LME | single-session-preference | Session@1 | 30 | 17 |  | 27 | 10 | 26 | no | 30 | yes | unknown | unknown | unknown | unknown | session-rank reorder inside the measured session top 10 |
| LME | single-session-preference | Session@5 | 30 | 26 |  | 27 | 1 | 26 | no | 30 | yes | unknown | unknown | unknown | unknown | admit members from the measured top 10 into the top 5 |
| LME | single-session-preference | Session@10 | 30 | 30 |  | 27 | 0 | 30 | yes | 30 | yes | unknown | unknown | unknown | unknown | already met; the reorder ceiling is the current value |
| LME | temporal-reasoning | Turn@1 | 127 | 93 |  | 115 | 22 | 111 | no | 116 | yes | unknown in [116, 125] | yes | 125 | yes | pull slot 1 from the measured top 10 (top 5 is short) |
| LME | temporal-reasoning | Turn@5 | 127 | 111 |  | 115 | 4 | 111 | no | 116 | yes | unknown in [116, 125] | yes | 125 | yes | admit members from the measured top 10 into the top 5 |
| LME | temporal-reasoning | Turn@10 | 127 | 116 |  | 115 | 0 | 116 | yes | 116 | yes | unknown in [116, 125] | yes | 125 | yes | already met; the reorder ceiling is the current value |
| LME | temporal-reasoning | Turn All@5 | 127 | 74 |  | 115 | 41 | 74 | no | unknown | unknown | unknown | unknown | unknown | unknown | set ceiling past the measured value is unknown (no All@10 bit, and full-set pool membership is not in the attachment) |
| LME | temporal-reasoning | Session@1 | 127 | 114 |  | 115 | 1 | 124 | yes | 125 | yes | unknown | unknown | unknown | unknown | session-rank reorder inside the measured session top 5 |
| LME | temporal-reasoning | Session@5 | 127 | 124 |  | 115 | 0 | 124 | yes | 125 | yes | unknown | unknown | unknown | unknown | already met; reordering the current set does not raise it |
| LME | temporal-reasoning | Session@10 | 127 | 125 |  | 115 | 0 | 125 | yes | 125 | yes | unknown | unknown | unknown | unknown | already met; the reorder ceiling is the current value |
| LME | knowledge-update | Turn@1 | 72 | 62 |  | 65 | 3 | 71 | yes | 72 | yes | unknown in [72, 72] | yes | 72 | yes | slot-1 reorder inside the measured top 5 |
| LME | knowledge-update | Turn@5 | 72 | 71 |  | 65 | 0 | 71 | yes | 72 | yes | unknown in [72, 72] | yes | 72 | yes | already met; reordering the current set does not raise it |
| LME | knowledge-update | Turn@10 | 72 | 72 |  | 65 | 0 | 72 | yes | 72 | yes | unknown in [72, 72] | yes | 72 | yes | already met; the reorder ceiling is the current value |
| LME | knowledge-update | Turn All@5 | 72 | 59 |  | 65 | 6 | 59 | no | unknown | unknown | unknown | unknown | unknown | unknown | set ceiling past the measured value is unknown (no All@10 bit, and full-set pool membership is not in the attachment) |
| LME | knowledge-update | Session@1 | 72 | 71 |  | 65 | 0 | 72 | yes | 72 | yes | unknown | unknown | unknown | unknown | session-rank reorder inside the measured session top 5 |
| LME | knowledge-update | Session@5 | 72 | 72 |  | 65 | 0 | 72 | yes | 72 | yes | unknown | unknown | unknown | unknown | already met; reordering the current set does not raise it |
| LME | knowledge-update | Session@10 | 72 | 72 |  | 65 | 0 | 72 | yes | 72 | yes | unknown | unknown | unknown | unknown | already met; the reorder ceiling is the current value |
| LME | single-session-assistant | Turn@1 | 56 | 41 |  | 51 | 10 | 52 | yes | 54 | yes | unknown in [54, 56] | yes | 56 | yes | slot-1 reorder inside the measured top 5 |
| LME | single-session-assistant | Turn@5 | 56 | 52 |  | 51 | 0 | 52 | yes | 54 | yes | unknown in [54, 56] | yes | 56 | yes | already met; reordering the current set does not raise it |
| LME | single-session-assistant | Turn@10 | 56 | 54 |  | 51 | 0 | 54 | yes | 54 | yes | unknown in [54, 56] | yes | 56 | yes | already met; the reorder ceiling is the current value |
| LME | single-session-assistant | Turn All@5 | 56 | 52 |  | 51 | 0 | 52 | yes | unknown | unknown | unknown | unknown | unknown | unknown | already met; reordering slot 1 does not change the set |
| LME | single-session-assistant | Session@1 | 56 | 56 |  | 51 | 0 | 56 | yes | 56 | yes | unknown | unknown | unknown | unknown | session-rank reorder inside the measured session top 5 |
| LME | single-session-assistant | Session@5 | 56 | 56 |  | 51 | 0 | 56 | yes | 56 | yes | unknown | unknown | unknown | unknown | already met; reordering the current set does not raise it |
| LME | single-session-assistant | Session@10 | 56 | 56 |  | 51 | 0 | 56 | yes | 56 | yes | unknown | unknown | unknown | unknown | already met; the reorder ceiling is the current value |
| LoCoMo | BENCH | Turn@1 | 1536 | 796 | 805 | 1383 | 578 | 1127 | no | 1226 | no | 1415 | yes | 1492 | yes | select from the measured pool@50 (top 10 is short) |
| LoCoMo | BENCH | Turn@5 | 1536 | 1127 |  | 1383 | 256 | 1127 | no | 1226 | no | 1415 | yes | 1492 | yes | select from the measured pool@50 (top 10 is short) |
| LoCoMo | BENCH | Turn@10 | 1536 | 1226 |  | 1383 | 157 | 1226 | no | 1226 | no | 1415 | yes | 1492 | yes | select from the measured pool@50 (top 10 is short) |
| LoCoMo | BENCH | Turn All@5 (missmap, not a perq bit) | 1536 | 954 |  | 1383 | 429 | 954 | no | unknown | unknown | unknown | unknown | unknown | unknown | set ceiling past the measured value is unknown (no All@10 bit, and full-set pool membership is not in the attachment) |
| LoCoMo | BENCH | Session@1 | 1536 | 1133 |  | 1383 | 250 | 1416 | yes | unknown | unknown | unknown | unknown | unknown | unknown | session-rank reorder inside the measured session top 5 |
| LoCoMo | BENCH | Session@5 | 1536 | 1416 |  | 1383 | 0 | 1416 | yes | unknown | unknown | unknown | unknown | unknown | unknown | already met; reordering the current set does not raise it |
| LoCoMo | Multi-Hop | Turn@1 | 282 | 97 |  | 254 | 157 | 158 | no | 186 | no | unknown in [186, 269] | unknown | 269 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| LoCoMo | Multi-Hop | Turn@5 | 282 | 158 |  | 254 | 96 | 158 | no | 186 | no | unknown in [186, 269] | unknown | 269 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| LoCoMo | Multi-Hop | Turn@10 | 282 | 186 |  | 254 | 68 | 186 | no | 186 | no | unknown in [186, 269] | unknown | 269 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| LoCoMo | Multi-Hop | Session@1 | 282 | 156 |  | 254 | 98 | 246 | no | unknown | unknown | unknown | unknown | unknown | unknown | session top 5 is short and session@10 was not recorded, so the next ceiling is unknown |
| LoCoMo | Multi-Hop | Session@5 | 282 | 246 |  | 254 | 8 | 246 | no | unknown | unknown | unknown | unknown | unknown | unknown | session top 5 is short and session@10 was not recorded, so the next ceiling is unknown |
| LoCoMo | Temporal | Turn@1 | 321 | 187 |  | 289 | 102 | 248 | no | 260 | no | unknown in [260, 313] | unknown | 313 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| LoCoMo | Temporal | Turn@5 | 321 | 248 |  | 289 | 41 | 248 | no | 260 | no | unknown in [260, 313] | unknown | 313 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| LoCoMo | Temporal | Turn@10 | 321 | 260 |  | 289 | 29 | 260 | no | 260 | no | unknown in [260, 313] | unknown | 313 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| LoCoMo | Temporal | Session@1 | 321 | 242 |  | 289 | 47 | 299 | yes | unknown | unknown | unknown | unknown | unknown | unknown | session-rank reorder inside the measured session top 5 |
| LoCoMo | Temporal | Session@5 | 321 | 299 |  | 289 | 0 | 299 | yes | unknown | unknown | unknown | unknown | unknown | unknown | already met; reordering the current set does not raise it |
| LoCoMo | Open-Domain | Turn@1 | 92 | 18 |  | 83 | 65 | 35 | no | 43 | no | unknown in [43, 78] | no | 78 | no | past the positive pool: at least one no_pool gold has to be admitted before 90% is possible |
| LoCoMo | Open-Domain | Turn@5 | 92 | 35 |  | 83 | 48 | 35 | no | 43 | no | unknown in [43, 78] | no | 78 | no | past the positive pool: at least one no_pool gold has to be admitted before 90% is possible |
| LoCoMo | Open-Domain | Turn@10 | 92 | 43 |  | 83 | 40 | 43 | no | 43 | no | unknown in [43, 78] | no | 78 | no | past the positive pool: at least one no_pool gold has to be admitted before 90% is possible |
| LoCoMo | Open-Domain | Session@1 | 92 | 36 |  | 83 | 47 | 65 | no | unknown | unknown | unknown | unknown | unknown | unknown | session top 5 is short and session@10 was not recorded, so the next ceiling is unknown |
| LoCoMo | Open-Domain | Session@5 | 92 | 65 |  | 83 | 18 | 65 | no | unknown | unknown | unknown | unknown | unknown | unknown | session top 5 is short and session@10 was not recorded, so the next ceiling is unknown |
| LoCoMo | Single-Hop | Turn@1 | 841 | 494 |  | 757 | 263 | 686 | no | 737 | no | unknown in [737, 832] | unknown | 832 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| LoCoMo | Single-Hop | Turn@5 | 841 | 686 |  | 757 | 71 | 686 | no | 737 | no | unknown in [737, 832] | unknown | 832 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| LoCoMo | Single-Hop | Turn@10 | 841 | 737 |  | 757 | 20 | 737 | no | 737 | no | unknown in [737, 832] | unknown | 832 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| LoCoMo | Single-Hop | Session@1 | 841 | 699 |  | 757 | 58 | 806 | yes | unknown | unknown | unknown | unknown | unknown | unknown | session-rank reorder inside the measured session top 5 |
| LoCoMo | Single-Hop | Session@5 | 841 | 806 |  | 757 | 0 | 806 | yes | unknown | unknown | unknown | unknown | unknown | unknown | already met; reordering the current set does not raise it |
| BEAM | BENCH | Turn@1 | 355 | 156 | 160 | 320 | 160 | 238 | no | 267 | no | 328 | yes | 329 | yes | select from the measured pool@50 (top 10 is short) |
| BEAM | BENCH | Turn@5 | 355 | 238 |  | 320 | 82 | 238 | no | 267 | no | 328 | yes | 329 | yes | select from the measured pool@50 (top 10 is short) |
| BEAM | BENCH | Turn@10 | 355 | 267 |  | 320 | 53 | 267 | no | 267 | no | 328 | yes | 329 | yes | select from the measured pool@50 (top 10 is short) |
| BEAM | BENCH | Turn All@5 | 355 | 118 |  | 320 | 202 | 118 | no | 138 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | BENCH | Turn All@10 | 355 | 138 |  | 320 | 182 | 138 | no | 138 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | BENCH | Session@1 | 355 | 133 |  | 320 | 187 | 249 | no | 275 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | BENCH | Session@5 | 355 | 249 |  | 320 | 71 | 249 | no | 275 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | BENCH | Session@10 | 355 | 275 |  | 320 | 45 | 275 | no | 275 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | BENCH | Session All@5 | 355 | 137 |  | 320 | 183 | 137 | no | 178 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | BENCH | Session All@10 | 355 | 178 |  | 320 | 142 | 178 | no | 178 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | contradiction_resolution | Turn@1 | 40 | 32 |  | 36 | 4 | 40 | yes | 40 | yes | 40 | yes | 40 | yes | slot-1 reorder inside the measured top 5 |
| BEAM | contradiction_resolution | Turn@5 | 40 | 40 |  | 36 | 0 | 40 | yes | 40 | yes | 40 | yes | 40 | yes | already met; reordering the current set does not raise it |
| BEAM | contradiction_resolution | Turn@10 | 40 | 40 |  | 36 | 0 | 40 | yes | 40 | yes | 40 | yes | 40 | yes | already met; the reorder ceiling is the current value |
| BEAM | contradiction_resolution | Turn All@5 | 40 | 22 |  | 36 | 14 | 22 | no | 27 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | contradiction_resolution | Turn All@10 | 40 | 27 |  | 36 | 9 | 27 | no | 27 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | contradiction_resolution | Session@1 | 40 | 29 |  | 36 | 7 | 40 | yes | 40 | yes | unknown | unknown | unknown | unknown | session-rank reorder inside the measured session top 5 |
| BEAM | contradiction_resolution | Session@5 | 40 | 40 |  | 36 | 0 | 40 | yes | 40 | yes | unknown | unknown | unknown | unknown | already met; reordering the current set does not raise it |
| BEAM | contradiction_resolution | Session@10 | 40 | 40 |  | 36 | 0 | 40 | yes | 40 | yes | unknown | unknown | unknown | unknown | already met; the reorder ceiling is the current value |
| BEAM | contradiction_resolution | Session All@5 | 40 | 20 |  | 36 | 16 | 20 | no | 30 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | contradiction_resolution | Session All@10 | 40 | 30 |  | 36 | 6 | 30 | no | 30 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | event_ordering | Turn@1 | 40 | 7 |  | 36 | 29 | 14 | no | 22 | no | unknown in [22, 39] | unknown | 39 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| BEAM | event_ordering | Turn@5 | 40 | 14 |  | 36 | 22 | 14 | no | 22 | no | unknown in [22, 39] | unknown | 39 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| BEAM | event_ordering | Turn@10 | 40 | 22 |  | 36 | 14 | 22 | no | 22 | no | unknown in [22, 39] | unknown | 39 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| BEAM | event_ordering | Turn All@5 | 40 | 0 |  | 36 | 36 | 0 | no | 0 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | event_ordering | Turn All@10 | 40 | 0 |  | 36 | 36 | 0 | no | 0 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | event_ordering | Session@1 | 40 | 7 |  | 36 | 29 | 22 | no | 27 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | event_ordering | Session@5 | 40 | 22 |  | 36 | 14 | 22 | no | 27 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | event_ordering | Session@10 | 40 | 27 |  | 36 | 9 | 27 | no | 27 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | event_ordering | Session All@5 | 40 | 0 |  | 36 | 36 | 0 | no | 0 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | event_ordering | Session All@10 | 40 | 0 |  | 36 | 36 | 0 | no | 0 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | information_extraction | Turn@1 | 40 | 20 |  | 36 | 16 | 27 | no | 29 | no | unknown in [29, 32] | no | 32 | no | past the positive pool: at least one no_pool gold has to be admitted before 90% is possible |
| BEAM | information_extraction | Turn@5 | 40 | 27 |  | 36 | 9 | 27 | no | 29 | no | unknown in [29, 32] | no | 32 | no | past the positive pool: at least one no_pool gold has to be admitted before 90% is possible |
| BEAM | information_extraction | Turn@10 | 40 | 29 |  | 36 | 7 | 29 | no | 29 | no | unknown in [29, 32] | no | 32 | no | past the positive pool: at least one no_pool gold has to be admitted before 90% is possible |
| BEAM | information_extraction | Turn All@5 | 40 | 20 |  | 36 | 16 | 20 | no | 21 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | information_extraction | Turn All@10 | 40 | 21 |  | 36 | 15 | 21 | no | 21 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | information_extraction | Session@1 | 40 | 15 |  | 36 | 21 | 25 | no | 27 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | information_extraction | Session@5 | 40 | 25 |  | 36 | 11 | 25 | no | 27 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | information_extraction | Session@10 | 40 | 27 |  | 36 | 9 | 27 | no | 27 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | information_extraction | Session All@5 | 40 | 25 |  | 36 | 11 | 25 | no | 27 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | information_extraction | Session All@10 | 40 | 27 |  | 36 | 9 | 27 | no | 27 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | instruction_following | Turn@1 | 40 | 6 |  | 36 | 30 | 18 | no | 21 | no | unknown in [21, 35] | no | 35 | no | past the positive pool: at least one no_pool gold has to be admitted before 90% is possible |
| BEAM | instruction_following | Turn@5 | 40 | 18 |  | 36 | 18 | 18 | no | 21 | no | unknown in [21, 35] | no | 35 | no | past the positive pool: at least one no_pool gold has to be admitted before 90% is possible |
| BEAM | instruction_following | Turn@10 | 40 | 21 |  | 36 | 15 | 21 | no | 21 | no | unknown in [21, 35] | no | 35 | no | past the positive pool: at least one no_pool gold has to be admitted before 90% is possible |
| BEAM | instruction_following | Turn All@5 | 40 | 13 |  | 36 | 23 | 13 | no | 16 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | instruction_following | Turn All@10 | 40 | 16 |  | 36 | 20 | 16 | no | 16 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | instruction_following | Session@1 | 40 | 4 |  | 36 | 32 | 16 | no | 17 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | instruction_following | Session@5 | 40 | 16 |  | 36 | 20 | 16 | no | 17 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | instruction_following | Session@10 | 40 | 17 |  | 36 | 19 | 17 | no | 17 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | instruction_following | Session All@5 | 40 | 15 |  | 36 | 21 | 15 | no | 17 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | instruction_following | Session All@10 | 40 | 17 |  | 36 | 19 | 17 | no | 17 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | knowledge_update | Turn@1 | 40 | 30 |  | 36 | 6 | 37 | yes | 39 | yes | unknown in [39, 39] | yes | 39 | yes | slot-1 reorder inside the measured top 5 |
| BEAM | knowledge_update | Turn@5 | 40 | 37 |  | 36 | 0 | 37 | yes | 39 | yes | unknown in [39, 39] | yes | 39 | yes | already met; reordering the current set does not raise it |
| BEAM | knowledge_update | Turn@10 | 40 | 39 |  | 36 | 0 | 39 | yes | 39 | yes | unknown in [39, 39] | yes | 39 | yes | already met; the reorder ceiling is the current value |
| BEAM | knowledge_update | Turn All@5 | 40 | 24 |  | 36 | 12 | 24 | no | 27 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | knowledge_update | Turn All@10 | 40 | 27 |  | 36 | 9 | 27 | no | 27 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | knowledge_update | Session@1 | 40 | 24 |  | 36 | 12 | 37 | yes | 38 | yes | unknown | unknown | unknown | unknown | session-rank reorder inside the measured session top 5 |
| BEAM | knowledge_update | Session@5 | 40 | 37 |  | 36 | 0 | 37 | yes | 38 | yes | unknown | unknown | unknown | unknown | already met; reordering the current set does not raise it |
| BEAM | knowledge_update | Session@10 | 40 | 38 |  | 36 | 0 | 38 | yes | 38 | yes | unknown | unknown | unknown | unknown | already met; the reorder ceiling is the current value |
| BEAM | knowledge_update | Session All@5 | 40 | 29 |  | 36 | 7 | 29 | no | 31 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | knowledge_update | Session All@10 | 40 | 31 |  | 36 | 5 | 31 | no | 31 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | multi_session_reasoning | Turn@1 | 40 | 21 |  | 36 | 15 | 34 | no | 36 | yes | unknown in [36, 40] | yes | 40 | yes | pull slot 1 from the measured top 10 (top 5 is short) |
| BEAM | multi_session_reasoning | Turn@5 | 40 | 34 |  | 36 | 2 | 34 | no | 36 | yes | unknown in [36, 40] | yes | 40 | yes | admit members from the measured top 10 into the top 5 |
| BEAM | multi_session_reasoning | Turn@10 | 40 | 36 |  | 36 | 0 | 36 | yes | 36 | yes | unknown in [36, 40] | yes | 40 | yes | already met; the reorder ceiling is the current value |
| BEAM | multi_session_reasoning | Turn All@5 | 40 | 4 |  | 36 | 32 | 4 | no | 7 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | multi_session_reasoning | Turn All@10 | 40 | 7 |  | 36 | 29 | 7 | no | 7 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | multi_session_reasoning | Session@1 | 40 | 20 |  | 36 | 16 | 33 | no | 36 | yes | unknown | unknown | unknown | unknown | session-rank reorder inside the measured session top 10 |
| BEAM | multi_session_reasoning | Session@5 | 40 | 33 |  | 36 | 3 | 33 | no | 36 | yes | unknown | unknown | unknown | unknown | admit members from the measured top 10 into the top 5 |
| BEAM | multi_session_reasoning | Session@10 | 40 | 36 |  | 36 | 0 | 36 | yes | 36 | yes | unknown | unknown | unknown | unknown | already met; the reorder ceiling is the current value |
| BEAM | multi_session_reasoning | Session All@5 | 40 | 8 |  | 36 | 28 | 8 | no | 15 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | multi_session_reasoning | Session All@10 | 40 | 15 |  | 36 | 21 | 15 | no | 15 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | preference_following | Turn@1 | 39 | 7 |  | 36 | 29 | 14 | no | 16 | no | unknown in [16, 30] | no | 30 | no | past the positive pool: at least one no_pool gold has to be admitted before 90% is possible |
| BEAM | preference_following | Turn@5 | 39 | 14 |  | 36 | 22 | 14 | no | 16 | no | unknown in [16, 30] | no | 30 | no | past the positive pool: at least one no_pool gold has to be admitted before 90% is possible |
| BEAM | preference_following | Turn@10 | 39 | 16 |  | 36 | 20 | 16 | no | 16 | no | unknown in [16, 30] | no | 30 | no | past the positive pool: at least one no_pool gold has to be admitted before 90% is possible |
| BEAM | preference_following | Turn All@5 | 39 | 11 |  | 36 | 25 | 11 | no | 13 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | preference_following | Turn All@10 | 39 | 13 |  | 36 | 23 | 13 | no | 13 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | preference_following | Session@1 | 39 | 4 |  | 36 | 32 | 14 | no | 24 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | preference_following | Session@5 | 39 | 14 |  | 36 | 22 | 14 | no | 24 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | preference_following | Session@10 | 39 | 24 |  | 36 | 12 | 24 | no | 24 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | preference_following | Session All@5 | 39 | 14 |  | 36 | 22 | 14 | no | 24 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | preference_following | Session All@10 | 39 | 24 |  | 36 | 12 | 24 | no | 24 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | summarization | Turn@1 | 36 | 6 |  | 33 | 27 | 17 | no | 25 | no | unknown in [25, 34] | unknown | 34 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| BEAM | summarization | Turn@5 | 36 | 17 |  | 33 | 16 | 17 | no | 25 | no | unknown in [25, 34] | unknown | 34 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| BEAM | summarization | Turn@10 | 36 | 25 |  | 33 | 8 | 25 | no | 25 | no | unknown in [25, 34] | unknown | 34 | yes | positive pool reaches the need; whether those golds sit inside the dumped 50 is not separable. rank_one sees every t_scene>0 turn, so the operative ceiling is the positive pool |
| BEAM | summarization | Turn All@5 | 36 | 0 |  | 33 | 33 | 0 | no | 0 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | summarization | Turn All@10 | 36 | 0 |  | 33 | 33 | 0 | no | 0 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | summarization | Session@1 | 36 | 6 |  | 33 | 27 | 24 | no | 26 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | summarization | Session@5 | 36 | 24 |  | 33 | 9 | 24 | no | 26 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | summarization | Session@10 | 36 | 26 |  | 33 | 7 | 26 | no | 26 | no | unknown | unknown | unknown | unknown | outside the current session top 10; session-pool membership is not in the attachment |
| BEAM | summarization | Session All@5 | 36 | 0 |  | 33 | 33 | 0 | no | 1 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | summarization | Session All@10 | 36 | 1 |  | 33 | 32 | 1 | no | 1 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | temporal_reasoning | Turn@1 | 40 | 27 |  | 36 | 9 | 37 | yes | 39 | yes | unknown in [39, 40] | yes | 40 | yes | slot-1 reorder inside the measured top 5 |
| BEAM | temporal_reasoning | Turn@5 | 40 | 37 |  | 36 | 0 | 37 | yes | 39 | yes | unknown in [39, 40] | yes | 40 | yes | already met; reordering the current set does not raise it |
| BEAM | temporal_reasoning | Turn@10 | 40 | 39 |  | 36 | 0 | 39 | yes | 39 | yes | unknown in [39, 40] | yes | 40 | yes | already met; the reorder ceiling is the current value |
| BEAM | temporal_reasoning | Turn All@5 | 40 | 24 |  | 36 | 12 | 24 | no | 27 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | temporal_reasoning | Turn All@10 | 40 | 27 |  | 36 | 9 | 27 | no | 27 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | temporal_reasoning | Session@1 | 40 | 24 |  | 36 | 12 | 38 | yes | 40 | yes | unknown | unknown | unknown | unknown | session-rank reorder inside the measured session top 5 |
| BEAM | temporal_reasoning | Session@5 | 40 | 38 |  | 36 | 0 | 38 | yes | 40 | yes | unknown | unknown | unknown | unknown | already met; reordering the current set does not raise it |
| BEAM | temporal_reasoning | Session@10 | 40 | 40 |  | 36 | 0 | 40 | yes | 40 | yes | unknown | unknown | unknown | unknown | already met; the reorder ceiling is the current value |
| BEAM | temporal_reasoning | Session All@5 | 40 | 26 |  | 36 | 10 | 26 | no | 33 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |
| BEAM | temporal_reasoning | Session All@10 | 40 | 33 |  | 36 | 3 | 33 | no | 33 | no | unknown | unknown | unknown | unknown | outside the current top 10; the attachment does not locate the missing gold members of the set |

## How to read the big gaps

LoCoMo Turn@1, bench shortfall 578 against step-34 805 (need 1383). Measured top 5 is 1127 and top 10 is 1226, both short. pool@50 is 1415, which clears 1383. The bench is a pool@50 selection problem (157 of the golds that are in the 50 and not in the top 10). S8-001 is that selection. It does not clear every category: Open-Domain pool is 78 against need 83, so at least 5 of 14 no_pool golds have to be admitted (S8-010, S8-062). Multi-Hop pool is 269 against need 254, but top 10 is 186 and category pool@50 is only known to lie in [186, 269]. Single-Hop pool is 832 against need 757, top 10 is 737. Temporal pool is 313 against need 289, top 10 is 260.

LoCoMo Turn@10, bench shortfall 157. pool@50 1415 clears 1383. Open-Domain still sits past its pool.

LoCoMo Session@1, bench shortfall 250. Session top 5 is 1416, which clears 1383, so the bench is a session reorder (S8-007, S8-040, S8-055, S8-078). Multi-Hop session top 5 is 246 against need 254, and there is no s10 bit. Open-Domain session top 5 is 65 against need 83, and there is no s10 bit. Those two categories are not a bench-level reorder.

BEAM Turn@1, bench shortfall 160 against step-34 160 (need 320). Top 5 is 238 and top 10 is 267, both short. pool@50 is 328 and the positive pool is 329, so the bench clears inside the positive pool. Three categories do not: information_extraction pool 32 against need 36 (no_pool 8), instruction_following pool 35 against need 36 (no_pool 5), preference_following pool 30 against need 36 (no_pool 9). event_ordering pool is 39 against need 36, but top 10 is 22. summarization pool is 34 against need 33, top 10 is 25. multi_session_reasoning top 5 is 34 and top 10 is 36, so that category is a top-10 pull.

BEAM Turn All@5 is 118 and Turn All@10 is 138, need 320. Slot-1 reorder leaves All@5 at 118. Every category's All@10 is below its own need (contradiction 27, event_ordering 0, information_extraction 21, instruction_following 16, knowledge_update 27, multi_session_reasoning 7, preference_following 13, summarization 0, temporal_reasoning 27, against need 36 or, for summarization, 33). 90% All@5 or All@10 is impossible inside the current top 10. The attachment does not say where the missing set-members sit. Designs S8-003, S8-005, S8-025, S8-031, S8-033, S8-051, S8-057, S8-060, S8-064, S8-069, S8-077, S8-079, S8-080, S8-081, and S8-084 are admission rules whose ceiling class is "outside the current top 10". None of them is a proof that 320 is reached.

BEAM Session@1 is 133, session top 5 is 249, session top 10 is 275, need 320. A perfect reorder of the session top 10 is still short 45. Session@5's top-10 ceiling is 275. Session@10's reorder ceiling is 275. Categories already inside session top 5: contradiction_resolution 40, knowledge_update 37, temporal_reasoning 38. multi_session_reasoning reaches 36 at session top 10. event_ordering session top 10 is 27, information_extraction 27, instruction_following 17, preference_following 24, summarization 26. S8-006, S8-017, S8-018, S8-047, S8-053, S8-058, S8-065, S8-071, S8-073, and S8-075 insert a session the current top 10 does not hold. The attachment does not identify those sessions, so the designs are the ceiling class, not a measured path to 320.

LME Turn@1, bench shortfall 70 against step-34 353 (need 423). Measured top 5 is 428, which clears 423, except two categories. single-session-preference top 5 and top 10 are both 19 against need 27, pool is 30, no_pool is 0: the category has to be pulled from the positive pool outside the top 10. temporal-reasoning top 5 is 111 against need 115 and top 10 is 116, so that category is a top-10 membership problem, not a deep-recall problem. LME session metrics are already met (Session@1 430, Session@5 463, Session@10 468, Session All@5 424, Session All@10 447, need 423). None of the session rules is allowed to spend those.

LME Turn All@5 is 314, need 423. Slot-1 reorder leaves it at 314. There is no Turn All@10 bit, so the top-10 set ceiling is unknown. Session All@5 is already 424, so the missing mass is turns inside sessions the session metric already found (S8-005 and the other intra-session fills). Whether every gold member of those questions sits in the current top 10 is not computable.

## The event-ordering cap

`max_per_session_top5` is 1 when `is_event_sequence` is set, 2 on multi_hop, otherwise 3 (`retrieval.rs` 5618-5625). On a question where that flag is set, a same-session second gold cannot enter the top 5 through the existing cap. event_ordering All@5 and All@10 are both 0, so no top-10 set edit creates even one All@5 hit. S8-003 admits later same-session stations after the submodular result (hook 5781-5783). It does not change the constant. The flag's own cues ("order of", "chronological", and the other closed strings at 3969-3972) are narrow; event_ordering designs are not gated only on that flag.

## What 90% cannot mean from this attachment

- Open-Domain Turn@1, BEAM information_extraction Turn@1, BEAM instruction_following Turn@1, and BEAM preference_following Turn@1 each have a positive pool below the 90% need.
- BEAM Turn All@5, Turn All@10, Session@1, Session@5, and Session@10 each have a measured top-10 ceiling below the bench need.
- LoCoMo Multi-Hop Session@1 and Open-Domain Session@1 are short of 90% at session top 5, and session top 10 was not recorded.
- LME Turn All@5's top-10 set ceiling was not recorded.
- No rule in S8-001..S8-090 is a measured proof of BEAM All@5 = 320 or BEAM Session@1 = 320. Those need golds the attachment does not locate.

## Hook classes used by the designs

ORDERING hooks sit at 6272-6273 (after Step 33 triangulation, before the dump) or at 5590-5599 (session reorder of an already built list). They permute or select among turns and sessions that already have a positive score.

RECALL/INDEX hooks sit at 5781-5783 (after turn submodular, before the Q4 block), at 5480-5599 (session admission), at 4852-4856 (a zero `t_scene` lifted once), or at 3370-3377 (a new index flag consumed when `t_exact_mask` is written, around 4737-4738). Admitting a turn or session with `s_score==0` or `t_scene==0` is RECALL/INDEX. Pulling a `t_scene>0` turn from the pool into the kept list is ORDERING relative to the pool, and the slot-1-of-top-5 ceiling does not apply to it.
