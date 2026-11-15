# Shard 2 funnel — LoCoMo Open-Domain and BEAM summarization

Base checkout: `perf/c2-step33-triangulation` at `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.
Attached measurements: `perq-bd2693e2.json` (binary `/workspace/c2-algos-bins/t51b-attr-unseated-opener-t10`, sha256 `e93411ac566c4c83f92a83204e70ea85ae743445a3256dc5cb4eacc2523e6f84`) and `missmap-bd2693e2.txt`. That binary is the pre-Step-33 bd2693e2 dump. Step 33 is already on this HEAD and is not in the attached bits.

## What the replay dump fields mean

`write_one_replay_dump` (`crates/vestige-core/src/retrieval.rs` 7426–7448) builds the pool after seating:

- `seated` / `final_ids` / `final_rank` is the official turn list after the gatekeeper and every C2 seat, including Step 33. Rank 0 is `seated[0]`.
- The dump pool starts as that seated list, then appends turns with `t_scene > 0` in scene order until 50. `pre_rank` is the index in that list. `pre_ids[0]` is `seated[0]` by construction, so gold at pre_rank 0 and gold at final rank 0 are the same number. There is no separate post-Kemeny snapshot.
- `t_scene` is the scene-pooled turn score. Seating does not rewrite it. Scene-rank 0 is the argmax, tie broken toward the lower index.
- `t_own` is uncarried BM25 plus the contiguous-bigram bonus. `s_score + bridge_boost` is the official session rank score. Both are dump copies. Ranking does not read them back.
- `qtok_present` and `turn_ic_novel` read `TurnIx.body`. That field is filled only when `VESTIGE_REPLAY_DUMP` is set (retrieval.rs 3379–3383). A ranking rule must not read it.
- A turn enters the wide pool only with `t_scene > 0`. `no_pool` in the miss map is absence from that wide pool. Bench LoCoMo `pool − n = 1492 − 1536 = −44`, and the four category `no_pool` counts sum to 44. `pool@50` is the 50-long `pre_ids` list. Bench LoCoMo `pool − pool@50 = 77`. Those 77 are not broken out by category in the attachment.

Per-question bit layouts, checked by summing the arrays against the aggregates in the same JSON:

- LoCoMo, 5 bits: `[s1, s5, t1, t5, t10]`. Sums `[1133, 1416, 796, 1127, 1226]`. Keys are `conv:qidx` with no category, so Open-Domain cannot be filtered out of `perq.locomo`. The category totals below come from `locomo.cats`.
- BEAM, 7 bits: `[t1, t5, t10, all5, s1, s5, sall5]`. Sums `[156, 238, 267, 118, 133, 249, 137]`. `all10`, `s10`, and `sall10` are not in the array. They are in `beam.cats`. Keys include the category (`1:summarization:0`).

## Open-Domain, from the attachment

`locomo.cats.Open-Domain`: n=92, s1=36, s5=65, t1=18, t5=35, t10=43.

Miss map, misses@1: `pool_not_k` 35, `t5_not1` 17, `no_pool` 14, `k_not5` 8. Those sum to 74. Hits are 18. Check: 18+17=35 (top5), 35+8=43 (official k / t10), 43+35=78 (wide pool), 78+14=92.

Five-line funnel:

1. Gold in corpus: the attachment has no corpus bit. Not computable here. `C2_STEP33_LOG.md` measured the same binary (its bench pool@50 matches the miss map) and records Open-Domain corpus 92/92. One LoCoMo question overall, not this slice, is missing from the index (bench corpus 1535/1536).
2. Gold in pool@50: not in the attachment. The same log records 63. Wide-pool gold from the miss map is 78, so on that measurement 15 pooled golds sit outside the 50.
3. Gold at pre_rank 0: 18. This equals final rank 0 because `pre_ids` is the seated list followed by the scene fill.
4. Gold in top5: 35.
5. Gold at final rank 0: 18.

Step 33 on this HEAD reordered inside that top 5 and moved Open-Domain final rank 0 from 18 to 19 (`4:28` only). Top5, t10, and the pool did not move (`SCOREBOARD_STEP33.md`, `C2_STEP33_LOG.md`). The live residual inside the top 5 is 16 questions, and the top-5 membership ceiling is still 35/92.

Ceilings against the attached 18/92, and against the brief's 83/92:

- Slot-1 reorder of the current top 5: 35/92. After the shipped Step 33 seat the attachment's 18 is 19 on this HEAD, and 35 is unchanged.
- Promote any official-k gold (t10) to rank 1: 43/92.
- Promote any wide-pool gold to rank 1: 78/92. 78 is still short of 83.
- Promote any pool@50 gold, using the same-binary log rather than the attachment: 63/92.
- 83/92 requires at least 5 of the 14 `no_pool` golds to be admitted by an index or ingest change, and then nearly every recovered gold has to sit at rank 1. A pure ordering pass cannot reach 90% on this slice. The designs below are tagged ORDERING when they only permute the current top 5, and RECALL/INDEX when they admit a turn that is outside that top 5 or change what ingest stores.

The attached miss that Step 33 already fixed is the shape the remaining ordering rules still have to catch. `4:28` asks which popular composer's tunes Tim plays. The gold turns were already in the top 5. The query is a descriptive paraphrase. The legal signals are orthographic shape, closed-class grammar, df, and session co-occurrence. A new domain word list is not a legal signal. `SUMMARY_REQUEST_CUES` (retrieval.rs 376–378) stays as it is.

## BEAM summarization, from the attachment

`beam.cats.summarization`: n=36, t1=6, t5=17, t10=25, all5=0, all10=0, s1=6, s5=24, s10=26, sall5=0, sall10=1.

Miss map, misses@1: `t5_not1` 11, `k_not5` 8, `pool_not_k` 9, `no_pool` 2. Sum 30. Hits 6. Check: 6+11=17, 17+8=25, 25+9=34, 34+2=36.

Five-line funnel:

1. Gold in corpus: not in the attachment. The same-binary log records BEAM corpus 355/355 for the bench, not a per-slice count. This slice's corpus membership is not computable from the attachment.
2. Gold in pool@50: not per slice. Bench pool is 329 and pool@50 is 328, so exactly one BEAM question differs. This slice's wide-pool gold is 34, so its pool@50 gold is 33 or 34. Which one is not in the attachment.
3. Gold at pre_rank 0: 6, equal to final rank 0 by the writer above. Step 33 changed no BEAM bit.
4. Gold in top5: 17.
5. Gold at final rank 0: 6.

Ceilings against 33/36 Turn@1:

- Slot-1 reorder of the current top 5: 17/36. Short of 33.
- Promote any official-k gold: 25/36. Still short of 33.
- Promote any wide-pool gold: 34/36. That clears 33 if 33 of the 34 pooled questions put one gold at rank 1. The 2 `no_pool` questions are not required for the Turn@1 bar. They are required if the pool seat misses even two of the 34.
- All@5 is 0 and All@10 is 0. No summarization question currently has every gold turn inside the top 5 or the top 10. Slot-1 does not move All@5. A section quota that admits one gold per block is what moves All@5. The brief's 33/36 bar is Turn@1, so the first summarization seats are RECALL/INDEX from outside the top 5, then an ORDERING rule that picks which admitted gold takes slot 1.

Session@1 on this slice is 6/36, the same as Turn@1. Session@5 is 24/36. A large part of the Turn@1 gap is the wrong turn inside a session that is already near the session list, which is the pool_not_k and k_not5 mass, not a missing corpus.

## What that implies for the tag

ORDERING is a permutation of the current top 5. Its Open-Domain ceiling is 35/92 and its summarization ceiling is 17/36. Neither is 90%.

RECALL/INDEX is a change to who enters the pool or the seated list, or to the integer features stored at the `TurnIx` push (retrieval.rs 3334–3384). Open-Domain needs it to pass 78/92 and to have any path to 83/92. Summarization Turn@1 needs it to pass 25/36 and to have a path to 33/36. The new seat, if one is later implemented, attaches after `apply_c2_step33_triangulation` and before `maybe_dump_replay` (retrieval.rs 6265–6273), and does not write `rsb_seat_behind`.

## Target change — 90% on every metric

Sam's target of 15 Nov 2026 asks for 90% on every metric of LME, LoCoMo, and BEAM, and on every category. Need is ceil(0.9 × n). Turn@1 floors are Step 34 (`62041ef5`): LME 353, LoCoMo 805, BEAM 160. Every other cell is the attached bd2693e2 dump. Step 34's extra Turn@1 hits are +3, +9, and +4, and they are not broken out by category, so category Turn@1 floors below are the attachment.

A slot-1 reorder permutes the current top 5. Its Turn@1 ceiling is Turn@5. It does not change Turn@5, Turn@10, All@5, All@10, or the session list. The session list is built at `retrieval.rs` 5480–5600 and is not rewritten by the turn seats at 6265–6272. A session reorder's Session@1 ceiling is Session@5. Admitting a turn the pool does not hold is the only way past the wide-pool column.

The five-line Open-Domain and summarization funnels above still hold. They are the category rows, not the bench. Open-Domain's pool of 78 cannot reach 83. The bench LoCoMo pool of 1492, and pool@50 of 1415, both can reach 1383, because Single-Hop, Multi-Hop, and Temporal have spare.

### Per-metric ceiling, every bench row and every category row

| Bench | Slice | Metric | n | Floor | Need | Gap | Met | What 90% requires | Source |
|---|---|---|---:|---:|---:|---:|---|---|---|
| LME | ALL | Session@1 | 470 | 430 | 423 | 0 | yes | already met (430) | bd2693e2 |
| LME | ALL | Session@5 | 470 | 463 | 423 | 0 | yes | already met (463) | bd2693e2 |
| LME | ALL | Session@10 | 470 | 468 | 423 | 0 | yes | already met (468) | bd2693e2 |
| LME | ALL | Session All@5 | 470 | 424 | 423 | 0 | yes | already met (424) | bd2693e2 |
| LME | ALL | Session All@10 | 470 | 447 | 423 | 0 | yes | already met (447) | bd2693e2 |
| LME | ALL | Turn@1 | 470 | 353 | 423 | 70 | no | slot-1 reorder of the current top 5 (ceiling 428) | Step 34 floor 353; Turn@5 428 and pool 468 are the attachment |
| LME | ALL | Turn@5 | 470 | 428 | 423 | 0 | yes | already at Turn@5 (428) | bd2693e2 and the target table |
| LME | ALL | Turn@10 | 470 | 437 | 423 | 0 | yes | already at Turn@10 (437) | bd2693e2 and the target table |
| LME | ALL | Turn All@5 | 470 | 314 | 423 | 109 | no | slot-1 ceiling stays 314, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 470. | bd2693e2 |
| LME | ALL | Turn All@10 | 470 | not recorded | 423 |  | no | not recorded | absent from lme.turn and from the target table |
| LoCoMo | ALL | Turn@1 | 1536 | 805 | 1383 | 578 | no | promote from the wide pool (ceiling 1492). pool@50 is 1415, which also clears 1383 | Step 34 floor 805; Turn@5 1127, Turn@10 1226, pool 1492 are the attachment |
| LoCoMo | ALL | Turn@5 | 1536 | 1127 | 1383 | 256 | no | promote from the wide pool (ceiling 1492). pool@50 1415 clears 1383 | bd2693e2 |
| LoCoMo | ALL | Turn@10 | 1536 | 1226 | 1383 | 157 | no | promote from the wide pool (ceiling 1492). pool@50 1415 clears 1383 | bd2693e2 |
| LoCoMo | ALL | Turn All@5 | 1536 | 954 | 1383 | 429 | no | slot-1 ceiling stays 954, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 1536. | miss-map header. Not in the target table and not in the per-question bits |
| LoCoMo | ALL | Turn All@10 | 1536 | not recorded | 1383 |  | no | not recorded | absent from the attachment and from the target table |
| LoCoMo | ALL | Session@1 | 1536 | 1133 | 1383 | 250 | no | reorder sessions already in the top 5 (ceiling 1416) | bd2693e2 |
| LoCoMo | ALL | Session@5 | 1536 | 1416 | 1383 | 0 | yes | already met (1416) | bd2693e2 |
| LoCoMo | ALL | Session@10 | 1536 | not recorded | 1383 |  | no | not recorded | absent from perq.locomo and from locomo.all |
| BEAM | ALL | Turn@1 | 355 | 160 | 320 | 160 | no | promote from the wide pool (ceiling 329). pool@50 is 328, which also clears 320 | Step 34 floor 160; Turn@5 238, Turn@10 267, pool 329 are the attachment |
| BEAM | ALL | Turn@5 | 355 | 238 | 320 | 82 | no | promote from the wide pool (ceiling 329) | bd2693e2 |
| BEAM | ALL | Turn@10 | 355 | 267 | 320 | 53 | no | promote from the wide pool (ceiling 329) | bd2693e2 |
| BEAM | ALL | Turn All@5 | 355 | 118 | 320 | 202 | no | slot-1 ceiling stays 118, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 355. | bd2693e2 |
| BEAM | ALL | Turn All@10 | 355 | 138 | 320 | 182 | no | slot-1 ceiling stays 138, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 355. | bd2693e2 beam.all. All@10 is not a per-question bit |
| BEAM | ALL | Session@1 | 355 | 133 | 320 | 187 | no | admit sessions outside the top 10 (Session@10 275, short 45) | bd2693e2 |
| BEAM | ALL | Session@5 | 355 | 249 | 320 | 71 | no | admit sessions outside the top 10 (Session@10 275, short 45) | bd2693e2 |
| BEAM | ALL | Session@10 | 355 | 275 | 320 | 45 | no | admit sessions outside the top 10 (now 275, short 45). Wider ceiling is not recorded. Upper bound 355. | bd2693e2 |
| BEAM | ALL | Session All@5 | 355 | 137 | 320 | 183 | no | slot-1 ceiling stays 137, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 355. | bd2693e2 beam.all.sall5. Not a row of the target table |
| BEAM | ALL | Session All@10 | 355 | 178 | 320 | 142 | no | slot-1 ceiling stays 178, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 355. | bd2693e2 beam.all.sall10. Not a row of the target table |
| LME | single-session-user | Session@1 | 64 | 62 | 58 | 0 | yes | already met (62) | lme.by_type |
| LME | single-session-user | Session@5 | 64 | 64 | 58 | 0 | yes | already met (64) | lme.by_type |
| LME | single-session-user | Session@10 | 64 | 64 | 58 | 0 | yes | already met (64) | lme.by_type |
| LME | single-session-user | Turn@1 | 64 | 54 | 58 | 4 | no | slot-1 reorder of the current top 5 (ceiling 62) | attachment. Step 34's +3 Turn@1 hits are not broken out by category |
| LME | single-session-user | Turn@5 | 64 | 62 | 58 | 0 | yes | already at Turn@5 (62) | lme.by_type |
| LME | single-session-user | Turn@10 | 64 | 63 | 58 | 0 | yes | already at Turn@10 (63) | lme.by_type |
| LME | single-session-user | Turn All@5 | 64 | 62 | 58 | 0 | yes | already met (62) | lme.by_type ta5 |
| LME | single-session-user | Session All@5 | 64 | not recorded | 58 |  | no | not recorded | by_type has no session-all cell |
| LME | single-session-user | Session All@10 | 64 | not recorded | 58 |  | no | not recorded | by_type has no session-all cell |
| LME | single-session-user | Turn All@10 | 64 | not recorded | 58 |  | no | not recorded | by_type has no turn-all@10 cell |
| LME | multi-session | Session@1 | 121 | 110 | 109 | 0 | yes | already met (110) | lme.by_type |
| LME | multi-session | Session@5 | 121 | 121 | 109 | 0 | yes | already met (121) | lme.by_type |
| LME | multi-session | Session@10 | 121 | 121 | 109 | 0 | yes | already met (121) | lme.by_type |
| LME | multi-session | Turn@1 | 121 | 88 | 109 | 21 | no | slot-1 reorder of the current top 5 (ceiling 113) | attachment. Step 34's +3 Turn@1 hits are not broken out by category |
| LME | multi-session | Turn@5 | 121 | 113 | 109 | 0 | yes | already at Turn@5 (113) | lme.by_type |
| LME | multi-session | Turn@10 | 121 | 113 | 109 | 0 | yes | already at Turn@10 (113) | lme.by_type |
| LME | multi-session | Turn All@5 | 121 | 53 | 109 | 56 | no | slot-1 ceiling stays 53, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 121. | lme.by_type ta5 |
| LME | multi-session | Session All@5 | 121 | not recorded | 109 |  | no | not recorded | by_type has no session-all cell |
| LME | multi-session | Session All@10 | 121 | not recorded | 109 |  | no | not recorded | by_type has no session-all cell |
| LME | multi-session | Turn All@10 | 121 | not recorded | 109 |  | no | not recorded | by_type has no turn-all@10 cell |
| LME | single-session-preference | Session@1 | 30 | 17 | 27 | 10 | no | promote a session already inside the top 10 (ceiling 30) | lme.by_type |
| LME | single-session-preference | Session@5 | 30 | 26 | 27 | 1 | no | promote from session top 10 into the top 5 (ceiling 30) | lme.by_type |
| LME | single-session-preference | Session@10 | 30 | 30 | 27 | 0 | yes | already met (30) | lme.by_type |
| LME | single-session-preference | Turn@1 | 30 | 12 | 27 | 15 | no | promote from the wide pool (ceiling 30) | attachment. Step 34's +3 Turn@1 hits are not broken out by category |
| LME | single-session-preference | Turn@5 | 30 | 19 | 27 | 8 | no | promote from the wide pool (ceiling 30) | lme.by_type |
| LME | single-session-preference | Turn@10 | 30 | 19 | 27 | 8 | no | promote from the wide pool (ceiling 30) | lme.by_type |
| LME | single-session-preference | Turn All@5 | 30 | 14 | 27 | 13 | no | slot-1 ceiling stays 14, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 30. | lme.by_type ta5 |
| LME | single-session-preference | Session All@5 | 30 | not recorded | 27 |  | no | not recorded | by_type has no session-all cell |
| LME | single-session-preference | Session All@10 | 30 | not recorded | 27 |  | no | not recorded | by_type has no session-all cell |
| LME | single-session-preference | Turn All@10 | 30 | not recorded | 27 |  | no | not recorded | by_type has no turn-all@10 cell |
| LME | temporal-reasoning | Session@1 | 127 | 114 | 115 | 1 | no | reorder sessions already in the top 5 (ceiling 124) | lme.by_type |
| LME | temporal-reasoning | Session@5 | 127 | 124 | 115 | 0 | yes | already met (124) | lme.by_type |
| LME | temporal-reasoning | Session@10 | 127 | 125 | 115 | 0 | yes | already met (125) | lme.by_type |
| LME | temporal-reasoning | Turn@1 | 127 | 93 | 115 | 22 | no | promote a gold already inside official k (ceiling 116) | attachment. Step 34's +3 Turn@1 hits are not broken out by category |
| LME | temporal-reasoning | Turn@5 | 127 | 111 | 115 | 4 | no | reorder official k into the top 5 (ceiling 116) | lme.by_type |
| LME | temporal-reasoning | Turn@10 | 127 | 116 | 115 | 0 | yes | already at Turn@10 (116) | lme.by_type |
| LME | temporal-reasoning | Turn All@5 | 127 | 74 | 115 | 41 | no | slot-1 ceiling stays 74, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 127. | lme.by_type ta5 |
| LME | temporal-reasoning | Session All@5 | 127 | not recorded | 115 |  | no | not recorded | by_type has no session-all cell |
| LME | temporal-reasoning | Session All@10 | 127 | not recorded | 115 |  | no | not recorded | by_type has no session-all cell |
| LME | temporal-reasoning | Turn All@10 | 127 | not recorded | 115 |  | no | not recorded | by_type has no turn-all@10 cell |
| LME | knowledge-update | Session@1 | 72 | 71 | 65 | 0 | yes | already met (71) | lme.by_type |
| LME | knowledge-update | Session@5 | 72 | 72 | 65 | 0 | yes | already met (72) | lme.by_type |
| LME | knowledge-update | Session@10 | 72 | 72 | 65 | 0 | yes | already met (72) | lme.by_type |
| LME | knowledge-update | Turn@1 | 72 | 62 | 65 | 3 | no | slot-1 reorder of the current top 5 (ceiling 71) | attachment. Step 34's +3 Turn@1 hits are not broken out by category |
| LME | knowledge-update | Turn@5 | 72 | 71 | 65 | 0 | yes | already at Turn@5 (71) | lme.by_type |
| LME | knowledge-update | Turn@10 | 72 | 72 | 65 | 0 | yes | already at Turn@10 (72) | lme.by_type |
| LME | knowledge-update | Turn All@5 | 72 | 59 | 65 | 6 | no | slot-1 ceiling stays 59, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 72. | lme.by_type ta5 |
| LME | knowledge-update | Session All@5 | 72 | not recorded | 65 |  | no | not recorded | by_type has no session-all cell |
| LME | knowledge-update | Session All@10 | 72 | not recorded | 65 |  | no | not recorded | by_type has no session-all cell |
| LME | knowledge-update | Turn All@10 | 72 | not recorded | 65 |  | no | not recorded | by_type has no turn-all@10 cell |
| LME | single-session-assistant | Session@1 | 56 | 56 | 51 | 0 | yes | already met (56) | lme.by_type |
| LME | single-session-assistant | Session@5 | 56 | 56 | 51 | 0 | yes | already met (56) | lme.by_type |
| LME | single-session-assistant | Session@10 | 56 | 56 | 51 | 0 | yes | already met (56) | lme.by_type |
| LME | single-session-assistant | Turn@1 | 56 | 41 | 51 | 10 | no | slot-1 reorder of the current top 5 (ceiling 52) | attachment. Step 34's +3 Turn@1 hits are not broken out by category |
| LME | single-session-assistant | Turn@5 | 56 | 52 | 51 | 0 | yes | already at Turn@5 (52) | lme.by_type |
| LME | single-session-assistant | Turn@10 | 56 | 54 | 51 | 0 | yes | already at Turn@10 (54) | lme.by_type |
| LME | single-session-assistant | Turn All@5 | 56 | 52 | 51 | 0 | yes | already met (52) | lme.by_type ta5 |
| LME | single-session-assistant | Session All@5 | 56 | not recorded | 51 |  | no | not recorded | by_type has no session-all cell |
| LME | single-session-assistant | Session All@10 | 56 | not recorded | 51 |  | no | not recorded | by_type has no session-all cell |
| LME | single-session-assistant | Turn All@10 | 56 | not recorded | 51 |  | no | not recorded | by_type has no turn-all@10 cell |
| LoCoMo | Single-Hop | Turn@1 | 841 | 494 | 757 | 263 | no | promote from the wide pool (ceiling 832) | locomo.cats plus miss-map no_pool. Step 34's +9 Turn@1 hits are not broken out |
| LoCoMo | Single-Hop | Turn@5 | 841 | 686 | 757 | 71 | no | promote from the wide pool (ceiling 832) | locomo.cats |
| LoCoMo | Single-Hop | Turn@10 | 841 | 737 | 757 | 20 | no | promote from the wide pool (ceiling 832) | locomo.cats |
| LoCoMo | Single-Hop | Session@1 | 841 | 699 | 757 | 58 | no | reorder sessions already in the top 5 (ceiling 806) | locomo.cats. No Session@10 |
| LoCoMo | Single-Hop | Session@5 | 841 | 806 | 757 | 0 | yes | already met (806) | locomo.cats |
| LoCoMo | Single-Hop | Turn All@5 | 841 | not recorded | 757 |  | no | not recorded | no per-category All@5 in the attachment |
| LoCoMo | Single-Hop | Session@10 | 841 | not recorded | 757 |  | no | not recorded | no per-category Session@10 |
| LoCoMo | Multi-Hop | Turn@1 | 282 | 97 | 254 | 157 | no | promote from the wide pool (ceiling 269) | locomo.cats plus miss-map no_pool. Step 34's +9 Turn@1 hits are not broken out |
| LoCoMo | Multi-Hop | Turn@5 | 282 | 158 | 254 | 96 | no | promote from the wide pool (ceiling 269) | locomo.cats |
| LoCoMo | Multi-Hop | Turn@10 | 282 | 186 | 254 | 68 | no | promote from the wide pool (ceiling 269) | locomo.cats |
| LoCoMo | Multi-Hop | Session@1 | 282 | 156 | 254 | 98 | no | top-5 session reorder stops at 246, short 8. Session@10 is not in the attachment. | locomo.cats. No Session@10 |
| LoCoMo | Multi-Hop | Session@5 | 282 | 246 | 254 | 8 | no | not past 246. Session@10 is not in the attachment. | locomo.cats |
| LoCoMo | Multi-Hop | Turn All@5 | 282 | not recorded | 254 |  | no | not recorded | no per-category All@5 in the attachment |
| LoCoMo | Multi-Hop | Session@10 | 282 | not recorded | 254 |  | no | not recorded | no per-category Session@10 |
| LoCoMo | Temporal | Turn@1 | 321 | 187 | 289 | 102 | no | promote from the wide pool (ceiling 313) | locomo.cats plus miss-map no_pool. Step 34's +9 Turn@1 hits are not broken out |
| LoCoMo | Temporal | Turn@5 | 321 | 248 | 289 | 41 | no | promote from the wide pool (ceiling 313) | locomo.cats |
| LoCoMo | Temporal | Turn@10 | 321 | 260 | 289 | 29 | no | promote from the wide pool (ceiling 313) | locomo.cats |
| LoCoMo | Temporal | Session@1 | 321 | 242 | 289 | 47 | no | reorder sessions already in the top 5 (ceiling 299) | locomo.cats. No Session@10 |
| LoCoMo | Temporal | Session@5 | 321 | 299 | 289 | 0 | yes | already met (299) | locomo.cats |
| LoCoMo | Temporal | Turn All@5 | 321 | not recorded | 289 |  | no | not recorded | no per-category All@5 in the attachment |
| LoCoMo | Temporal | Session@10 | 321 | not recorded | 289 |  | no | not recorded | no per-category Session@10 |
| LoCoMo | Open-Domain | Turn@1 | 92 | 18 | 83 | 65 | no | admit turns outside the pool (pool 78, short 5) | locomo.cats plus miss-map no_pool. Step 34's +9 Turn@1 hits are not broken out |
| LoCoMo | Open-Domain | Turn@5 | 92 | 35 | 83 | 48 | no | admit turns outside the pool (pool 78, short 5) | locomo.cats |
| LoCoMo | Open-Domain | Turn@10 | 92 | 43 | 83 | 40 | no | admit turns outside the pool (pool 78, short 5) | locomo.cats |
| LoCoMo | Open-Domain | Session@1 | 92 | 36 | 83 | 47 | no | top-5 session reorder stops at 65, short 18. Session@10 is not in the attachment. | locomo.cats. No Session@10 |
| LoCoMo | Open-Domain | Session@5 | 92 | 65 | 83 | 18 | no | not past 65. Session@10 is not in the attachment. | locomo.cats |
| LoCoMo | Open-Domain | Turn All@5 | 92 | not recorded | 83 |  | no | not recorded | no per-category All@5 in the attachment |
| LoCoMo | Open-Domain | Session@10 | 92 | not recorded | 83 |  | no | not recorded | no per-category Session@10 |
| BEAM | contradiction_resolution | Turn@1 | 40 | 32 | 36 | 4 | no | slot-1 reorder of the current top 5 (ceiling 40) | beam.cats. The miss map has no contradiction line. no_pool is 0 because t5 = t10 = n. Step 34's +4 Turn@1 hits are not broken out |
| BEAM | contradiction_resolution | Turn@5 | 40 | 40 | 36 | 0 | yes | already at Turn@5 (40) | beam.cats. The miss map has no contradiction line. no_pool is 0 because t5 = t10 = n |
| BEAM | contradiction_resolution | Turn@10 | 40 | 40 | 36 | 0 | yes | already at Turn@10 (40) | beam.cats. The miss map has no contradiction line. no_pool is 0 because t5 = t10 = n |
| BEAM | contradiction_resolution | Turn All@5 | 40 | 22 | 36 | 14 | no | slot-1 ceiling stays 22, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats. The miss map has no contradiction line. no_pool is 0 because t5 = t10 = n |
| BEAM | contradiction_resolution | Turn All@10 | 40 | 27 | 36 | 9 | no | slot-1 ceiling stays 27, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats. The miss map has no contradiction line. no_pool is 0 because t5 = t10 = n |
| BEAM | contradiction_resolution | Session@1 | 40 | 29 | 36 | 7 | no | reorder sessions already in the top 5 (ceiling 40) | beam.cats. The miss map has no contradiction line. no_pool is 0 because t5 = t10 = n |
| BEAM | contradiction_resolution | Session@5 | 40 | 40 | 36 | 0 | yes | already met (40) | beam.cats. The miss map has no contradiction line. no_pool is 0 because t5 = t10 = n |
| BEAM | contradiction_resolution | Session@10 | 40 | 40 | 36 | 0 | yes | already met (40) | beam.cats. The miss map has no contradiction line. no_pool is 0 because t5 = t10 = n |
| BEAM | contradiction_resolution | Session All@5 | 40 | 20 | 36 | 16 | no | slot-1 ceiling stays 20, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats. The miss map has no contradiction line. no_pool is 0 because t5 = t10 = n |
| BEAM | contradiction_resolution | Session All@10 | 40 | 30 | 36 | 6 | no | slot-1 ceiling stays 30, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats. The miss map has no contradiction line. no_pool is 0 because t5 = t10 = n |
| BEAM | event_ordering | Turn@1 | 40 | 7 | 36 | 29 | no | promote from the wide pool (ceiling 39) | beam.cats. Step 34's +4 Turn@1 hits are not broken out |
| BEAM | event_ordering | Turn@5 | 40 | 14 | 36 | 22 | no | promote from the wide pool (ceiling 39) | beam.cats |
| BEAM | event_ordering | Turn@10 | 40 | 22 | 36 | 14 | no | promote from the wide pool (ceiling 39) | beam.cats |
| BEAM | event_ordering | Turn All@5 | 40 | 0 | 36 | 36 | no | slot-1 ceiling stays 0, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | event_ordering | Turn All@10 | 40 | 0 | 36 | 36 | no | slot-1 ceiling stays 0, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | event_ordering | Session@1 | 40 | 7 | 36 | 29 | no | admit sessions outside the top 10 (Session@10 27, short 9) | beam.cats |
| BEAM | event_ordering | Session@5 | 40 | 22 | 36 | 14 | no | admit sessions outside the top 10 (Session@10 27, short 9) | beam.cats |
| BEAM | event_ordering | Session@10 | 40 | 27 | 36 | 9 | no | admit sessions outside the top 10 (now 27, short 9). Wider ceiling is not recorded. Upper bound 40. | beam.cats |
| BEAM | event_ordering | Session All@5 | 40 | 0 | 36 | 36 | no | slot-1 ceiling stays 0, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | event_ordering | Session All@10 | 40 | 0 | 36 | 36 | no | slot-1 ceiling stays 0, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | information_extraction | Turn@1 | 40 | 20 | 36 | 16 | no | admit turns outside the pool (pool 32, short 4) | beam.cats. Step 34's +4 Turn@1 hits are not broken out |
| BEAM | information_extraction | Turn@5 | 40 | 27 | 36 | 9 | no | admit turns outside the pool (pool 32, short 4) | beam.cats |
| BEAM | information_extraction | Turn@10 | 40 | 29 | 36 | 7 | no | admit turns outside the pool (pool 32, short 4) | beam.cats |
| BEAM | information_extraction | Turn All@5 | 40 | 20 | 36 | 16 | no | slot-1 ceiling stays 20, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | information_extraction | Turn All@10 | 40 | 21 | 36 | 15 | no | slot-1 ceiling stays 21, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | information_extraction | Session@1 | 40 | 15 | 36 | 21 | no | admit sessions outside the top 10 (Session@10 27, short 9) | beam.cats |
| BEAM | information_extraction | Session@5 | 40 | 25 | 36 | 11 | no | admit sessions outside the top 10 (Session@10 27, short 9) | beam.cats |
| BEAM | information_extraction | Session@10 | 40 | 27 | 36 | 9 | no | admit sessions outside the top 10 (now 27, short 9). Wider ceiling is not recorded. Upper bound 40. | beam.cats |
| BEAM | information_extraction | Session All@5 | 40 | 25 | 36 | 11 | no | slot-1 ceiling stays 25, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | information_extraction | Session All@10 | 40 | 27 | 36 | 9 | no | slot-1 ceiling stays 27, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | instruction_following | Turn@1 | 40 | 6 | 36 | 30 | no | admit turns outside the pool (pool 35, short 1) | beam.cats. Miss-map stage sum is 1 off the category totals. Pool is n minus no_pool 5, which is 35. Step 34's +4 Turn@1 hits are not broken out |
| BEAM | instruction_following | Turn@5 | 40 | 18 | 36 | 18 | no | admit turns outside the pool (pool 35, short 1) | beam.cats. Miss-map stage sum is 1 off the category totals. Pool is n minus no_pool 5, which is 35 |
| BEAM | instruction_following | Turn@10 | 40 | 21 | 36 | 15 | no | admit turns outside the pool (pool 35, short 1) | beam.cats. Miss-map stage sum is 1 off the category totals. Pool is n minus no_pool 5, which is 35 |
| BEAM | instruction_following | Turn All@5 | 40 | 13 | 36 | 23 | no | slot-1 ceiling stays 13, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats. Miss-map stage sum is 1 off the category totals. Pool is n minus no_pool 5, which is 35 |
| BEAM | instruction_following | Turn All@10 | 40 | 16 | 36 | 20 | no | slot-1 ceiling stays 16, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats. Miss-map stage sum is 1 off the category totals. Pool is n minus no_pool 5, which is 35 |
| BEAM | instruction_following | Session@1 | 40 | 4 | 36 | 32 | no | admit sessions outside the top 10 (Session@10 17, short 19) | beam.cats. Miss-map stage sum is 1 off the category totals. Pool is n minus no_pool 5, which is 35 |
| BEAM | instruction_following | Session@5 | 40 | 16 | 36 | 20 | no | admit sessions outside the top 10 (Session@10 17, short 19) | beam.cats. Miss-map stage sum is 1 off the category totals. Pool is n minus no_pool 5, which is 35 |
| BEAM | instruction_following | Session@10 | 40 | 17 | 36 | 19 | no | admit sessions outside the top 10 (now 17, short 19). Wider ceiling is not recorded. Upper bound 40. | beam.cats. Miss-map stage sum is 1 off the category totals. Pool is n minus no_pool 5, which is 35 |
| BEAM | instruction_following | Session All@5 | 40 | 15 | 36 | 21 | no | slot-1 ceiling stays 15, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats. Miss-map stage sum is 1 off the category totals. Pool is n minus no_pool 5, which is 35 |
| BEAM | instruction_following | Session All@10 | 40 | 17 | 36 | 19 | no | slot-1 ceiling stays 17, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats. Miss-map stage sum is 1 off the category totals. Pool is n minus no_pool 5, which is 35 |
| BEAM | knowledge_update | Turn@1 | 40 | 30 | 36 | 6 | no | slot-1 reorder of the current top 5 (ceiling 37) | beam.cats. Step 34's +4 Turn@1 hits are not broken out |
| BEAM | knowledge_update | Turn@5 | 40 | 37 | 36 | 0 | yes | already at Turn@5 (37) | beam.cats |
| BEAM | knowledge_update | Turn@10 | 40 | 39 | 36 | 0 | yes | already at Turn@10 (39) | beam.cats |
| BEAM | knowledge_update | Turn All@5 | 40 | 24 | 36 | 12 | no | slot-1 ceiling stays 24, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | knowledge_update | Turn All@10 | 40 | 27 | 36 | 9 | no | slot-1 ceiling stays 27, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | knowledge_update | Session@1 | 40 | 24 | 36 | 12 | no | reorder sessions already in the top 5 (ceiling 37) | beam.cats |
| BEAM | knowledge_update | Session@5 | 40 | 37 | 36 | 0 | yes | already met (37) | beam.cats |
| BEAM | knowledge_update | Session@10 | 40 | 38 | 36 | 0 | yes | already met (38) | beam.cats |
| BEAM | knowledge_update | Session All@5 | 40 | 29 | 36 | 7 | no | slot-1 ceiling stays 29, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | knowledge_update | Session All@10 | 40 | 31 | 36 | 5 | no | slot-1 ceiling stays 31, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | multi_session_reasoning | Turn@1 | 40 | 21 | 36 | 15 | no | promote a gold already inside official k (ceiling 36) | beam.cats. Step 34's +4 Turn@1 hits are not broken out |
| BEAM | multi_session_reasoning | Turn@5 | 40 | 34 | 36 | 2 | no | reorder official k into the top 5 (ceiling 36) | beam.cats |
| BEAM | multi_session_reasoning | Turn@10 | 40 | 36 | 36 | 0 | yes | already at Turn@10 (36) | beam.cats |
| BEAM | multi_session_reasoning | Turn All@5 | 40 | 4 | 36 | 32 | no | slot-1 ceiling stays 4, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | multi_session_reasoning | Turn All@10 | 40 | 7 | 36 | 29 | no | slot-1 ceiling stays 7, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | multi_session_reasoning | Session@1 | 40 | 20 | 36 | 16 | no | promote a session already inside the top 10 (ceiling 36) | beam.cats |
| BEAM | multi_session_reasoning | Session@5 | 40 | 33 | 36 | 3 | no | promote from session top 10 into the top 5 (ceiling 36) | beam.cats |
| BEAM | multi_session_reasoning | Session@10 | 40 | 36 | 36 | 0 | yes | already met (36) | beam.cats |
| BEAM | multi_session_reasoning | Session All@5 | 40 | 8 | 36 | 28 | no | slot-1 ceiling stays 8, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | multi_session_reasoning | Session All@10 | 40 | 15 | 36 | 21 | no | slot-1 ceiling stays 15, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | preference_following | Turn@1 | 39 | 7 | 36 | 29 | no | admit turns outside the pool (pool 30, short 6) | beam.cats. Step 34's +4 Turn@1 hits are not broken out |
| BEAM | preference_following | Turn@5 | 39 | 14 | 36 | 22 | no | admit turns outside the pool (pool 30, short 6) | beam.cats |
| BEAM | preference_following | Turn@10 | 39 | 16 | 36 | 20 | no | admit turns outside the pool (pool 30, short 6) | beam.cats |
| BEAM | preference_following | Turn All@5 | 39 | 11 | 36 | 25 | no | slot-1 ceiling stays 11, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 39. | beam.cats |
| BEAM | preference_following | Turn All@10 | 39 | 13 | 36 | 23 | no | slot-1 ceiling stays 13, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 39. | beam.cats |
| BEAM | preference_following | Session@1 | 39 | 4 | 36 | 32 | no | admit sessions outside the top 10 (Session@10 24, short 12) | beam.cats |
| BEAM | preference_following | Session@5 | 39 | 14 | 36 | 22 | no | admit sessions outside the top 10 (Session@10 24, short 12) | beam.cats |
| BEAM | preference_following | Session@10 | 39 | 24 | 36 | 12 | no | admit sessions outside the top 10 (now 24, short 12). Wider ceiling is not recorded. Upper bound 39. | beam.cats |
| BEAM | preference_following | Session All@5 | 39 | 14 | 36 | 22 | no | slot-1 ceiling stays 14, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 39. | beam.cats |
| BEAM | preference_following | Session All@10 | 39 | 24 | 36 | 12 | no | slot-1 ceiling stays 24, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 39. | beam.cats |
| BEAM | summarization | Turn@1 | 36 | 6 | 33 | 27 | no | promote from the wide pool (ceiling 34) | beam.cats. Step 34's +4 Turn@1 hits are not broken out |
| BEAM | summarization | Turn@5 | 36 | 17 | 33 | 16 | no | promote from the wide pool (ceiling 34) | beam.cats |
| BEAM | summarization | Turn@10 | 36 | 25 | 33 | 8 | no | promote from the wide pool (ceiling 34) | beam.cats |
| BEAM | summarization | Turn All@5 | 36 | 0 | 33 | 33 | no | slot-1 ceiling stays 0, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 36. | beam.cats |
| BEAM | summarization | Turn All@10 | 36 | 0 | 33 | 33 | no | slot-1 ceiling stays 0, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 36. | beam.cats |
| BEAM | summarization | Session@1 | 36 | 6 | 33 | 27 | no | admit sessions outside the top 10 (Session@10 26, short 7) | beam.cats |
| BEAM | summarization | Session@5 | 36 | 24 | 33 | 9 | no | admit sessions outside the top 10 (Session@10 26, short 7) | beam.cats |
| BEAM | summarization | Session@10 | 36 | 26 | 33 | 7 | no | admit sessions outside the top 10 (now 26, short 7). Wider ceiling is not recorded. Upper bound 36. | beam.cats |
| BEAM | summarization | Session All@5 | 36 | 0 | 33 | 33 | no | slot-1 ceiling stays 0, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 36. | beam.cats |
| BEAM | summarization | Session All@10 | 36 | 1 | 33 | 32 | no | slot-1 ceiling stays 1, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 36. | beam.cats |
| BEAM | temporal_reasoning | Turn@1 | 40 | 27 | 36 | 9 | no | slot-1 reorder of the current top 5 (ceiling 37) | beam.cats. Step 34's +4 Turn@1 hits are not broken out |
| BEAM | temporal_reasoning | Turn@5 | 40 | 37 | 36 | 0 | yes | already at Turn@5 (37) | beam.cats |
| BEAM | temporal_reasoning | Turn@10 | 40 | 39 | 36 | 0 | yes | already at Turn@10 (39) | beam.cats |
| BEAM | temporal_reasoning | Turn All@5 | 40 | 24 | 36 | 12 | no | slot-1 ceiling stays 24, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | temporal_reasoning | Turn All@10 | 40 | 27 | 36 | 9 | no | slot-1 ceiling stays 27, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | temporal_reasoning | Session@1 | 40 | 24 | 36 | 12 | no | reorder sessions already in the top 5 (ceiling 38) | beam.cats |
| BEAM | temporal_reasoning | Session@5 | 40 | 38 | 36 | 0 | yes | already met (38) | beam.cats |
| BEAM | temporal_reasoning | Session@10 | 40 | 40 | 36 | 0 | yes | already met (40) | beam.cats |
| BEAM | temporal_reasoning | Session All@5 | 40 | 26 | 36 | 10 | no | slot-1 ceiling stays 26, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |
| BEAM | temporal_reasoning | Session All@10 | 40 | 33 | 36 | 3 | no | slot-1 ceiling stays 33, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. | beam.cats |

### Largest gaps, bench and category

Gap is need minus the floor. A category can miss its own bar after the bench has cleared.

| Gap | Bench | Slice | Metric | Floor | Need | What 90% requires |
|---:|---|---|---|---:|---:|---|
| 578 | LoCoMo | ALL | Turn@1 | 805 | 1383 | promote from the wide pool (ceiling 1492). pool@50 is 1415, which also clears 1383 |
| 429 | LoCoMo | ALL | Turn All@5 | 954 | 1383 | slot-1 ceiling stays 954, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 1536. |
| 263 | LoCoMo | Single-Hop | Turn@1 | 494 | 757 | promote from the wide pool (ceiling 832) |
| 256 | LoCoMo | ALL | Turn@5 | 1127 | 1383 | promote from the wide pool (ceiling 1492). pool@50 1415 clears 1383 |
| 250 | LoCoMo | ALL | Session@1 | 1133 | 1383 | reorder sessions already in the top 5 (ceiling 1416) |
| 202 | BEAM | ALL | Turn All@5 | 118 | 320 | slot-1 ceiling stays 118, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 355. |
| 187 | BEAM | ALL | Session@1 | 133 | 320 | admit sessions outside the top 10 (Session@10 275, short 45) |
| 183 | BEAM | ALL | Session All@5 | 137 | 320 | slot-1 ceiling stays 137, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 355. |
| 182 | BEAM | ALL | Turn All@10 | 138 | 320 | slot-1 ceiling stays 138, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 355. |
| 160 | BEAM | ALL | Turn@1 | 160 | 320 | promote from the wide pool (ceiling 329). pool@50 is 328, which also clears 320 |
| 157 | LoCoMo | ALL | Turn@10 | 1226 | 1383 | promote from the wide pool (ceiling 1492). pool@50 1415 clears 1383 |
| 157 | LoCoMo | Multi-Hop | Turn@1 | 97 | 254 | promote from the wide pool (ceiling 269) |
| 142 | BEAM | ALL | Session All@10 | 178 | 320 | slot-1 ceiling stays 178, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 355. |
| 109 | LME | ALL | Turn All@5 | 314 | 423 | slot-1 ceiling stays 314, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 470. |
| 102 | LoCoMo | Temporal | Turn@1 | 187 | 289 | promote from the wide pool (ceiling 313) |
| 98 | LoCoMo | Multi-Hop | Session@1 | 156 | 254 | top-5 session reorder stops at 246, short 8. Session@10 is not in the attachment. |
| 96 | LoCoMo | Multi-Hop | Turn@5 | 158 | 254 | promote from the wide pool (ceiling 269) |
| 82 | BEAM | ALL | Turn@5 | 238 | 320 | promote from the wide pool (ceiling 329) |
| 71 | BEAM | ALL | Session@5 | 249 | 320 | admit sessions outside the top 10 (Session@10 275, short 45) |
| 71 | LoCoMo | Single-Hop | Turn@5 | 686 | 757 | promote from the wide pool (ceiling 832) |
| 70 | LME | ALL | Turn@1 | 353 | 423 | slot-1 reorder of the current top 5 (ceiling 428) |
| 68 | LoCoMo | Multi-Hop | Turn@10 | 186 | 254 | promote from the wide pool (ceiling 269) |
| 65 | LoCoMo | Open-Domain | Turn@1 | 18 | 83 | admit turns outside the pool (pool 78, short 5) |
| 58 | LoCoMo | Single-Hop | Session@1 | 699 | 757 | reorder sessions already in the top 5 (ceiling 806) |
| 56 | LME | multi-session | Turn All@5 | 53 | 109 | slot-1 ceiling stays 53, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 121. |
| 53 | BEAM | ALL | Turn@10 | 267 | 320 | promote from the wide pool (ceiling 329) |
| 48 | LoCoMo | Open-Domain | Turn@5 | 35 | 83 | admit turns outside the pool (pool 78, short 5) |
| 47 | LoCoMo | Temporal | Session@1 | 242 | 289 | reorder sessions already in the top 5 (ceiling 299) |
| 47 | LoCoMo | Open-Domain | Session@1 | 36 | 83 | top-5 session reorder stops at 65, short 18. Session@10 is not in the attachment. |
| 45 | BEAM | ALL | Session@10 | 275 | 320 | admit sessions outside the top 10 (now 275, short 45). Wider ceiling is not recorded. Upper bound 355. |
| 41 | LME | temporal-reasoning | Turn All@5 | 74 | 115 | slot-1 ceiling stays 74, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 127. |
| 41 | LoCoMo | Temporal | Turn@5 | 248 | 289 | promote from the wide pool (ceiling 313) |
| 40 | LoCoMo | Open-Domain | Turn@10 | 43 | 83 | admit turns outside the pool (pool 78, short 5) |
| 36 | BEAM | event_ordering | Turn All@5 | 0 | 36 | slot-1 ceiling stays 0, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. |
| 36 | BEAM | event_ordering | Turn All@10 | 0 | 36 | slot-1 ceiling stays 0, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. |
| 36 | BEAM | event_ordering | Session All@5 | 0 | 36 | slot-1 ceiling stays 0, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. |
| 36 | BEAM | event_ordering | Session All@10 | 0 | 36 | slot-1 ceiling stays 0, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 40. |
| 33 | BEAM | summarization | Turn All@5 | 0 | 33 | slot-1 ceiling stays 0, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 36. |
| 33 | BEAM | summarization | Turn All@10 | 0 | 33 | slot-1 ceiling stays 0, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 36. |
| 33 | BEAM | summarization | Session All@5 | 0 | 33 | slot-1 ceiling stays 0, because a slot-1 reorder does not change the top-5 set. How many of the remaining questions have every gold inside k or inside the pool is not in the bits. Upper bound 36. |

### What the attachment cannot compute

- LoCoMo Session@10, LoCoMo Turn All@10, LME Turn All@10, and every LoCoMo per-category All@5. The bits are not there.

- All@5 and All@10 ceilings above the current value. The per-question bits say whether some gold is inside k. They do not say whether every gold is inside k, or how many golds a question has. A question with six gold turns can never reach All@5. That count is not in the dump.

- Which of the Step 34 Turn@1 gains sit in which category. The ordering ceilings use Turn@5 from the attachment, and the target table's Turn@5 matches that attachment, so the extra hits are inside a top 5 the dump already counted.

- instruction_following's miss-map stages disagree with `beam.cats` by 1. The table uses the category totals and `n - no_pool`.

- Per-slice pool@50. Bench pool@50 is LoCoMo 1415, BEAM 328, LME 463. Open-Domain pool@50 is not in the attachment. The same-binary log used in the funnel above records 63.


### Which of the 90 rules can move which ceiling

The mechanisms are the ones in `algos-S2.md`. This table only says which ceiling each one can touch. SLOT1 permutes the current top 5. ADMIT can change pool membership and, when the admitted turn's session is absent, inserts that one session at `retrieval.rs` 5480–5600. MULTI admits more than one turn, which is the only class that can move All@5 or All@10.

| id | class | metrics it can move | ceiling it is allowed to claim |
|---|---|---|---|
| S2-001 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-002 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-003 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-004 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-005 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-006 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-007 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-008 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-009 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-010 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; LoCoMo Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-011 | ADMIT | BEAM summarization; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-012 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-013 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-014 | SLOT1 | LoCoMo Open-Domain; LME Turn@1; LoCoMo Session@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-015 | MULTI | BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1 | All@5 slot-1 ceiling is the current value, LME 314 and BEAM 118, both short of 423 and 320. Wide-pool All@5 is not computable. Turn@1 can rise to the wide pool: LoCoMo 1492, BEAM 329. Full table in funnel-S2.md. |
| S2-016 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-017 | SLOT1 | BEAM summarization; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-018 | SLOT1 | BEAM summarization; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-019 | MULTI | BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1 | All@5 slot-1 ceiling is the current value, LME 314 and BEAM 118, both short of 423 and 320. Wide-pool All@5 is not computable. Turn@1 can rise to the wide pool: LoCoMo 1492, BEAM 329. Full table in funnel-S2.md. |
| S2-020 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-021 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-022 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-023 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-024 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-025 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-026 | ADMIT | BEAM summarization; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-027 | SLOT1 | BEAM summarization; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-028 | MULTI | BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1 | All@5 slot-1 ceiling is the current value, LME 314 and BEAM 118, both short of 423 and 320. Wide-pool All@5 is not computable. Turn@1 can rise to the wide pool: LoCoMo 1492, BEAM 329. Full table in funnel-S2.md. |
| S2-029 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-030 | SLOT1 | BEAM summarization; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-031 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-032 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-033 | ADMIT | BEAM summarization; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-034 | ADMIT | BEAM summarization; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-035 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-036 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-037 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-038 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-039 | MULTI | BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1 | All@5 slot-1 ceiling is the current value, LME 314 and BEAM 118, both short of 423 and 320. Wide-pool All@5 is not computable. Turn@1 can rise to the wide pool: LoCoMo 1492, BEAM 329. Full table in funnel-S2.md. |
| S2-040 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-041 | SLOT1 | BEAM summarization; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-042 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-043 | SLOT1 | BEAM summarization; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-044 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-045 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-046 | MULTI | BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1 | All@5 slot-1 ceiling is the current value, LME 314 and BEAM 118, both short of 423 and 320. Wide-pool All@5 is not computable. Turn@1 can rise to the wide pool: LoCoMo 1492, BEAM 329. Full table in funnel-S2.md. |
| S2-047 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-048 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-049 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-050 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-051 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-052 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-053 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-054 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-055 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-056 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-057 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-058 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-059 | SLOT1 | BEAM summarization; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-060 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-061 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-062 | SLOT1 | BEAM summarization; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-063 | SLOT1 | BEAM summarization; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-064 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-065 | MULTI | BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1 | All@5 slot-1 ceiling is the current value, LME 314 and BEAM 118, both short of 423 and 320. Wide-pool All@5 is not computable. Turn@1 can rise to the wide pool: LoCoMo 1492, BEAM 329. Full table in funnel-S2.md. |
| S2-066 | SLOT1 | BEAM summarization; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-067 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-068 | MULTI | BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1 | All@5 slot-1 ceiling is the current value, LME 314 and BEAM 118, both short of 423 and 320. Wide-pool All@5 is not computable. Turn@1 can rise to the wide pool: LoCoMo 1492, BEAM 329. Full table in funnel-S2.md. |
| S2-069 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-070 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-071 | SLOT1 | BEAM summarization; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-072 | MULTI | BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1 | All@5 slot-1 ceiling is the current value, LME 314 and BEAM 118, both short of 423 and 320. Wide-pool All@5 is not computable. Turn@1 can rise to the wide pool: LoCoMo 1492, BEAM 329. Full table in funnel-S2.md. |
| S2-073 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; LoCoMo Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-074 | SLOT1 | BEAM summarization; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-075 | MULTI | BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1; LoCoMo Session@1 | All@5 slot-1 ceiling is the current value, LME 314 and BEAM 118, both short of 423 and 320. Wide-pool All@5 is not computable. Turn@1 can rise to the wide pool: LoCoMo 1492, BEAM 329. Full table in funnel-S2.md. |
| S2-076 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-077 | SLOT1 | BEAM summarization; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-078 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-079 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-080 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-081 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-082 | ADMIT | BEAM summarization; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-083 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-084 | MULTI | BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1 | All@5 slot-1 ceiling is the current value, LME 314 and BEAM 118, both short of 423 and 320. Wide-pool All@5 is not computable. Turn@1 can rise to the wide pool: LoCoMo 1492, BEAM 329. Full table in funnel-S2.md. |
| S2-085 | SLOT1 | BEAM summarization; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-086 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-087 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; LoCoMo Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-088 | ADMIT | LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
| S2-089 | SLOT1 | LoCoMo Open-Domain; LME Turn@1 | Slot-1 ceiling equals Turn@5: LME 428 (need 423), LoCoMo 1127 (need 1383), BEAM 238 (need 320). All@5 stays 314 / 118. Session@1 stays 1133 / 133. Full table in funnel-S2.md. |
| S2-090 | ADMIT | BEAM summarization; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1 | Turn@1 inside-k ceiling is Turn@10: LoCoMo 1226 (need 1383), BEAM 267 (need 320). Wide pool clears both (1492 and 329). Open-Domain pool 78 is short of 83. Full table in funnel-S2.md. |
