# Shard 10 funnel — bd2693e2 attachment, HEAD `e2ecd1da037974c1835f563ad8b046f4a4ca3629`

Scoreboard file: `perq-bd2693e2.json` (sha256 `e93411ac566c4c83f92a83204e70ea85ae743445a3256dc5cb4eacc2523e6f84` of the uploaded copy). Miss stages: `missmap-bd2693e2.txt`. Both were read locally. No engine code was run and no dump was recomputed. Where this file and the Step 33 log disagree, the attachment is what the funnel uses.

## What the replay dump writer actually emits

`write_one_replay_dump` in `crates/vestige-core/src/retrieval.rs` (7426–7620), called from `maybe_dump_replay` after Step 33 (6273) and flushed at the end of `rank` (7628). It is off the ranking path. Fields:

| Field | Meaning |
|---|---|
| `pre_ids` | `snap.seated` first, then turns with `t_scene > 0` in scene-score order, capped at 50 (7429–7440). |
| `rank_pre` | Index of that turn inside `pre_ids`. |
| `final_ids`, `seated_ids`, `rank` | The seated list after every seat, including Step 33. `pre_ids[0]` is `seated[0]`, so gold at pre_rank 0 and gold at final rank 0 are the same fact whenever this writer produced the row. There is no separate post-Kemeny snapshot. |
| `head_id` | `seated[0]`. |
| `t_scene`, `t_own`, `t_turn_raw`, `t_scene_sess` | Scene score, uncarried BM25, raw turn score, session scene score. |
| `s_score`, `bridge_boost` | Official session score and its bridge term. |
| `qtok_present`, `turn_ic_novel`, `ends_with_question` | Dump-only. They read `turn.body`. Ranking must not. |
| `wh_slots`, `wh_class`, `is_event_sequence`, `temporal`, `preference`, `assistant_attribution` | Query-plan flags already on the snap. |
| `k`, `pool_size`, `candidates`, `session_turns` | `k` is the seated width. `pool_size` is the dump pool (≤ 50), not the broad pool. `session_turns` is every turn of sessions that placed in the official top 10. |

## What the per-question JSON actually stores

Bits were checked by summing them against the scoreboard totals.

- LoCoMo, 5 bits `[s1, s5, t1, t5, t10]`. Sums `[1133, 1416, 796, 1127, 1226]`. Keys are `conv:q` with no category, so a category funnel cannot be recomputed from the bits alone. Category counts below are the miss map, and they add back to these sums.
- BEAM, 7 bits `[t1, t5, t10, all5, s1, s5, sall5]`. Sums `[156, 238, 267, 118, 133, 249, 137]`. The 7th bit is `sall5` (137). It is not `s10`.
- LME, 7 bits `[s1, s5, s10, t1, t5, t10, ta5]`. Sums `[430, 463, 468, 350, 428, 437, 314]`. The aggregate object also carries session All@5 424 and All@10 447, which are not a seventh bit.

Miss-map stage names, checked against the header totals:

- `no_pool` — gold is outside the **broad** pool. It is not a pool@50 miss. LoCoMo category `no_pool` sums to 44 = 1536 − 1492. BEAM category `no_pool` sums to 26 = 355 − 329.
- `pool_not_k` — in the broad pool, not in the official top-k.
- `k_not5` — in the official top-k, outside top 5.
- `t5_not1` — in top 5, not at rank 0. `t1 + t5_not1` equals per-question `t5` on all six slices (18+17, 97+61, 6+12, 6+11, 7+7, 7+7).

`pool` and `pool@50` are different globally (LoCoMo 1492 vs 1415, BEAM 329 vs 328). The miss map does not say which of a slice's in-pool gold fall in that gap. **Per-slice pool@50 cannot be computed from the attachment.**

Gold-in-corpus cannot be separated from scored `n`. The Step 33 log says one LoCoMo evidence id is missing from indexed turns (1535 vs 1536) and does not say which category. This attachment has no corpus-membership bit.

The attachment's LoCoMo Turn@1 is **796**, and Open-Domain Turn@1 is **18** (92 − 74 misses). The brief's "19/92" is the Step 33 log after triangulation (gains on questions 4:28, 4:44, 8:124). Multi-Hop stays 97 in that log. This funnel does not borrow the log's pool@50 or scene-rank numbers.

## Per-slice funnel

Need is `ceil(0.9 * n)`. "Reorder ceiling" is gold already in top 5, which is the most a slot-1 swap can seat. "Broad-pool ceiling" is `n − no_pool`.

### Open-Domain (LoCoMo) — n=92, need 83

1. Gold in corpus: not separable from n=92.
2. Gold in pool@50: not computable. Broad pool = 78.
3. Gold at pre_rank 0: same event as final rank 0 under this dump writer. Final rank 0 = 18.
4. Gold in top 5: 35.
5. Gold at final rank 0: 18.

Misses of the 74: `pool_not_k` 35, `t5_not1` 17, `no_pool` 14, `k_not5` 8.

Broad-pool ceiling 78 is **below** 83. A perfect reorder of the current top 5 stops at 35. Reaching 90% requires pulling at least 5 of the 14 `no_pool` golds into the pool and then converting essentially every in-pool miss (35 + 8 + 17, plus those 5). Slot-1 code alone cannot get there.

### Multi-Hop (LoCoMo) — n=282, need 254

1. Gold in corpus: not separable from n=282.
2. Gold in pool@50: not computable. Broad pool = 269.
3. Gold at pre_rank 0: same event as final rank 0. Final rank 0 = 97.
4. Gold in top 5: 158.
5. Gold at final rank 0: 97.

Misses of the 185: `pool_not_k` 83, `t5_not1` 61, `k_not5` 28, `no_pool` 13.

Broad-pool ceiling 269 ≥ 254, so `no_pool` recovery is not strictly required. Reorder ceiling 158 is far below 254. The gap is pool→k and k→top5: 111 of those 172 in-pool misses have to become top-5 hits, and ordering still has to finish the job. Tag the work RECALL/INDEX unless it only swaps inside an existing top 5.

### instruction_following (BEAM) — n=40, need 36

1. Gold in corpus: not separable from n=40.
2. Gold in pool@50: not computable. Broad pool = 35.
3. Gold at pre_rank 0: same event as final rank 0. Final rank 0 = 6.
4. Gold in top 5: 18. `all5` = 13 (multi-gold all seated in the top 5).
5. Gold at final rank 0: 6.

Misses of the 34: `pool_not_k` 15, `t5_not1` 12, `no_pool` 5, `k_not5` 2.

Broad-pool ceiling 35 is **below** 36. Reorder ceiling 18. Need at least 1 of 5 `no_pool` golds, and all 29 in-pool misses, before a reorder can finish.

### summarization (BEAM) — n=36, need 33

1. Gold in corpus: not separable from n=36.
2. Gold in pool@50: not computable. Broad pool = 34.
3. Gold at pre_rank 0: same event as final rank 0. Final rank 0 = 6.
4. Gold in top 5: 17. `all5` = 0 and the miss map's all10 column is 0: a summary question's gold set is never fully seated.
5. Gold at final rank 0: 6. Turn@1 here is any-gold at rank 0, not the full set.

Misses of the 30: `t5_not1` 11, `pool_not_k` 9, `k_not5` 8, `no_pool` 2.

Broad-pool ceiling 34 ≥ 33. Reorder ceiling 17. Need 27 of 28 in-pool misses fixed. `no_pool` recovery is not strictly required for the Turn@1 count and is required if the empty `all5` is the thing being fixed.

### event_ordering (BEAM) — n=40, need 36

1. Gold in corpus: not separable from n=40.
2. Gold in pool@50: not computable. Broad pool = 39.
3. Gold at pre_rank 0: same event as final rank 0. Final rank 0 = 7.
4. Gold in top 5: 14. `all5` = 0.
5. Gold at final rank 0: 7.

Misses of the 33: `pool_not_k` 17, `k_not5` 8, `t5_not1` 7, `no_pool` 1.

Broad-pool ceiling 39 ≥ 36. Reorder ceiling 14. Need 29 of 32 in-pool misses fixed.

### preference_following (BEAM) — n=39, need 36

1. Gold in corpus: not separable from n=39.
2. Gold in pool@50: not computable. Broad pool = 30.
3. Gold at pre_rank 0: same event as final rank 0. Final rank 0 = 7.
4. Gold in top 5: 14. `all5` = 11.
5. Gold at final rank 0: 7.

Misses of the 32: `pool_not_k` 14, `no_pool` 9, `t5_not1` 7, `k_not5` 2.

Broad-pool ceiling 30 is **below** 36. Reorder ceiling 14. Need at least 6 of 9 `no_pool` golds and all 23 in-pool misses.

## What that forces on the designs

| Slice | Reorder ceiling | Broad-pool ceiling | 90% requires |
|---|---|---|---|
| Open-Domain | 35 | 78 | recall of ≥5 `no_pool`, then almost every in-pool miss |
| Multi-Hop | 158 | 269 | pool→k and k→top5 on 157 of 172 in-pool misses; ordering after that |
| instruction_following | 18 | 35 | recall of ≥1 `no_pool`, then every in-pool miss |
| summarization | 17 | 34 | 27 of 28 in-pool misses; `all5` stays 0 until a set is filled |
| event_ordering | 14 | 39 | 29 of 32 in-pool misses; `all5` stays 0 until a set is filled |
| preference_following | 14 | 30 | recall of ≥6 `no_pool`, then every in-pool miss |

All six are recall-bound. An ORDERING tag is honest only for a swap inside an existing top 5 (`t5_not1`: 17, 61, 12, 11, 7, 7). Everything aimed at 90% is RECALL/INDEX: new atomic keys the splitter currently destroys, aliases copied off the turn's own second name, and set-fill that puts a sibling into the pool. No design in this shard retunes an existing float.

The six slices above are the original brief. The 90% floor on every metric is the next section. LME is part of that floor.

## Floor after the 90% target

Sam's target is 90% of every metric on LME, LoCoMo, and BEAM, and 90% of every category inside them. Need is `ceil(0.9 * n)`. The target note's Turn@1 bench floors are Step 34 (`62041ef5`): LME 353, LoCoMo 805, BEAM 160. Every category cell and every other bench cell below is the bd2693e2 attachment (LME Turn@1 350, LoCoMo 796, BEAM 156). The +3 / +9 / +4 Turn@1 hits are not split by category, so a ceiling that replaces a category starts from the attachment. Turn@5 and Turn@10 in the target note match this attachment.

Two facts in `retrieval.rs` bound every row:

- `final_session_indices` is chosen at line 5480 from `s_base` over `session_bags` (lines 4536–4543). Those bags are the turn feats (lines 3630–3632). A seat at 6265–6272 runs after that list is frozen, so it cannot move Session@1, Session@5, or Session@10. An ingest feat can, because the feat is already in the bag.
- Turn All@5 is a set. BEAM All@10 is 138. Rearranging the current top 10 cannot push All@5 above 138, and 138 is 182 short of 320. A set moves when a fill rule inserts the other golds, or when an inherit rule copies feats onto them. A slot-1 swap does neither. How many golds share an accession stem is not in the attachment, so a fill cap is an upper bound, not a count of questions the rule will hit.

Numbers in parentheses are the cap if the new feat is also what the `no_pool` golds were missing. The bare number keeps those golds out. Neither number is an expected gain.

### Bench, and what kind of change can reach 90%

| Bench | Metric | Attachment | Step 34 | n | Need | Seat cap | In-pool cap of the whole attachment |
|---|---|---:|---:|---:|---:|---:|---:|
| LoCoMo | Turn@1 | 796 | 805 | 1536 | 1383 | 1127 | 1492. Bench 90% is inside the broad pool. Open-Domain's own pool is 78, under its need of 83. |
| LoCoMo | Turn@5 | 1127 | 1127 | 1536 | 1383 | 1127 | 1492 |
| LoCoMo | Turn@10 | 1226 | 1226 | 1536 | 1383 | 1226 | 1492 |
| LoCoMo | Session@1 | 1133 | — | 1536 | 1383 | 1416 | Session@5 already holds the bench need. Open-Domain Session@5 is 65 (need 83) and Multi-Hop Session@5 is 246 (need 254). |
| LoCoMo | Session@5 | 1416 | — | 1536 | 1383 | already met | The same two categories are not met. |
| BEAM | Turn@1 | 156 | 160 | 355 | 320 | 238 | 329, the broad pool. Category instruction_following pool is 35 (need 36) and preference_following pool is 30 (need 36). |
| BEAM | Turn@5 | 238 | 238 | 355 | 320 | 238 | 329 |
| BEAM | Turn@10 | 267 | 267 | 355 | 320 | 267 | 329 |
| BEAM | All@5 | 118 | — | 355 | 320 | ≤138 | All@10 is 138. event_ordering and summarization are 0 at both. |
| BEAM | All@10 | 138 | — | 355 | 320 | 138 | 138 |
| BEAM | Session@1 | 133 | — | 355 | 320 | 249 | Session@5 is 249 and Session@10 is 275. Both are under 320. |
| BEAM | Session@5 | 249 | — | 355 | 320 | 249 | 249 |
| BEAM | Session@10 | 275 | — | 355 | 320 | 275 | 275 |
| LME | Turn@1 | 350 | 353 | 470 | 423 | 428 | 468, the broad pool. Bench 90% is inside Turn@5 except preference (Turn@5 19, need 27) and temporal (Turn@5 111, need 115). |
| LME | Turn@5 / Turn@10 | 428 / 437 | — | 470 | 423 | already met | The two category holes above. |
| LME | Turn All@5 | 314 | — | 470 | 423 | not a reorder | multi-session is 53 (need 109). temporal-reasoning is 74 (need 115). |
| LME | Session@1 / @5 / @10 | 430 / 463 / 468 | — | 470 | 423 | already met | preference Session@1 is 17 (need 27) and Session@5 is 26 (need 27). temporal Session@1 is 114 (need 115). |
| LME | Session All@5 / All@10 | 424 / 447 | — | 470 | 423 | already met | No per-category session-all figure is in the aggregate. |

LoCoMo Turn@5 is 256 short of 1383. That gap is not in the one-line gap list and it is real: a reorder cannot raise Turn@5. Category 90% is stricter than bench 90% because each category rounds up on its own.

### Category ceilings

`Seat cap` is Turn@5 or Session@5. `Pool cap` is `n - no_pool` from the miss map. contradiction_resolution is not given per-stage counts; its Turn@5 equals n, so those 8 misses are already inside the top 5.

| Bench | Category | n | Need | Turn@1 | Seat | Pool | Session@1 | Session seat | All@5 | All@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo | Single-Hop | 841 | 757 | 494 | 686 | 832 | 699 | 806 | — | — |
| LoCoMo | Multi-Hop | 282 | 254 | 97 | 158 | 269 | 156 | 246 | — | — |
| LoCoMo | Temporal | 321 | 289 | 187 | 248 | 313 | 242 | 299 | — | — |
| LoCoMo | Open-Domain | 92 | 83 | 18 | 35 | 78 | 36 | 65 | — | — |
| LME | single-session-user | 64 | 58 | 54 | 62 | 64 | 62 | 64 | 62 | — |
| LME | multi-session | 121 | 109 | 88 | 113 | 121 | 110 | 121 | 53 | — |
| LME | single-session-preference | 30 | 27 | 12 | 19 | 30 | 17 | 26 | 14 | — |
| LME | temporal-reasoning | 127 | 115 | 93 | 111 | 125 | 114 | 124 | 74 | — |
| LME | knowledge-update | 72 | 65 | 62 | 71 | 72 | 71 | 72 | 59 | — |
| LME | single-session-assistant | 56 | 51 | 41 | 52 | 56 | 56 | 56 | 52 | — |
| BEAM | contradiction_resolution | 40 | 36 | 32 | 40 | 40 | 29 | 40 | 22 | 27 |
| BEAM | event_ordering | 40 | 36 | 7 | 14 | 39 | 7 | 22 | 0 | 0 |
| BEAM | information_extraction | 40 | 36 | 20 | 27 | 32 | 15 | 25 | 20 | 21 |
| BEAM | instruction_following | 40 | 36 | 6 | 18 | 35 | 4 | 16 | 13 | 16 |
| BEAM | knowledge_update | 40 | 36 | 30 | 37 | 39 | 24 | 37 | 24 | 27 |
| BEAM | multi_session_reasoning | 40 | 36 | 21 | 34 | 40 | 20 | 33 | 4 | 7 |
| BEAM | preference_following | 39 | 36 | 7 | 14 | 30 | 4 | 14 | 11 | 13 |
| BEAM | summarization | 36 | 33 | 6 | 17 | 34 | 6 | 24 | 0 | 0 |
| BEAM | temporal_reasoning | 40 | 36 | 27 | 37 | 40 | 24 | 38 | 24 | 27 |

### What the ninety can reach together

A category takes the best cap of any one row that names it. Two rows do not add inside the same category. Two rows do add when they name different categories. The parenthetical is the cap if `no_pool` golds carry the shape.

- LoCoMo Turn@1, categories at their best in-pool cap: Single-Hop 832, Multi-Hop 269, Temporal 313, Open-Domain 78. Sum 1492. With `no_pool` recovered, 1536. Need 1383. The in-pool sum clears the bench. Open-Domain's in-pool cap is 78, and its need is 83, so the category does not clear unless at least 5 of its 14 `no_pool` golds carry the key or the alias.
- LoCoMo Session@1, ingest rows only, structural cap n on each named category: Single-Hop 841; Multi-Hop 282; Temporal 321; Open-Domain 92. Sum 1536. Need 1383. That cap clears the bench and every LoCoMo category. It is the cap if the feat puts the gold session at rank 0 whenever the shape is present. The attachment does not count the shape.
- BEAM Turn@1, in-pool: contradiction_resolution 32; event_ordering 39; information_extraction 32; instruction_following 35; knowledge_update 39; multi_session_reasoning 40; preference_following 30; summarization 34; temporal_reasoning 40. Sum 321. With `no_pool` recovered, 347. Need 320. instruction_following in-pool is 35 (need 36) and preference_following in-pool is 30 (need 36). Both clear only in the parenthetical. contradiction_resolution stays at 32 (need 36) because no row's test is a contradiction test. The bench sum clears.
- BEAM All@5: contradiction_resolution 22; event_ordering 40; information_extraction 20; instruction_following 13; knowledge_update 24; multi_session_reasoning 40; preference_following 39; summarization 36; temporal_reasoning 24. Sum 258. Need 320. Short 62. All@10, with the same categories raised and the others left at their current All@10: contradiction_resolution 27; event_ordering 40; information_extraction 21; instruction_following 16; knowledge_update 27; multi_session_reasoning 40; preference_following 39; summarization 36; temporal_reasoning 27. Sum 273. Short 47. The untouched categories are contradiction_resolution, information_extraction, instruction_following, knowledge_update, and temporal_reasoning. Their missing golds are not established to share a stem, so no row's cap covers them.
- BEAM Session@1, ingest feats plus a fill stem written into `session_bags`: contradiction_resolution 29; event_ordering 40; information_extraction 40; instruction_following 40; knowledge_update 40; multi_session_reasoning 40; preference_following 39; summarization 36; temporal_reasoning 40. Sum 344. Need 320. contradiction_resolution stays 29 (need 36). Every other category's structural cap is its n. Without the stem written into the bag, multi_session_reasoning stays at its current Session@1, because a pool insert at 5628 does not change `s_base`.
- LME Turn@1: single-session-user 64; multi-session 121; single-session-preference 30; temporal-reasoning 125 (127); knowledge-update 72; single-session-assistant 56. In-pool sum 468, full sum 470. Need 423. preference and temporal are the two categories whose Turn@5 is under the need; alias, inherit, and year rows are what move them. The other four categories are already inside Turn@5.
- LME Turn All@5: single-session-user 62; multi-session 121; single-session-preference 30; temporal-reasoning 127; knowledge-update 59; single-session-assistant 52. Sum 451. Need 423. knowledge-update stays 59 (need 65). The bench clears on the fill of multi-session and temporal-reasoning. The knowledge-update category does not.

### Per-row ceiling

Each cell is the **bench** score after this row alone, if its test succeeds on every question in the categories in `applies`. Other categories stay at the attachment. `—` means this hook cannot move that metric. A parenthetical on a turn cell is the cap with `no_pool` included. A parenthetical on a fill row's session cell is the cap if the same stem is interned as a feat at ingest so it enters `session_bags` before line 5480. The pool hook by itself does not do that.

Session All@5 and Session All@10 on LME are 424 and 447, both already at or above 423. No row is a session-all rule. LoCoMo has no All@5, All@10, or Session@10 in the target table or in this aggregate.

#### LoCoMo

| id | group | lever | applies | Turn@1 | Turn@5 | Turn@10 | Session@1 | Session@5 |
|---|---|---|---|---:|---:|---:|---:|---:|
| S10-001 | exact | ingest | LOC MH+OD+SH · BEAM IE+PF · LME asst+multi+user | 1366 (1402) | 1427 (1463) | 1439 (1475) | 1457 | 1514 |
| S10-002 | exact | ingest | LOC MH+SH · BEAM EO+IE · LME asst+multi+user | 1306 (1328) | 1384 (1406) | 1404 (1426) | 1401 | 1487 |
| S10-003 | exact | ingest | LOC MH+SH · BEAM IE+SUM · LME asst+multi+user | 1306 (1328) | 1384 (1406) | 1404 (1426) | 1401 | 1487 |
| S10-004 | year | ingest | LOC OD+TM · BEAM EO+TR · LME temp | 982 (1004) | 1235 (1257) | 1314 (1336) | 1268 | 1465 |
| S10-005 | year | seat | LOC OD+TM · BEAM EO+TR · LME temp | 874 | — | — | — | — |
| S10-006 | short | pool | LOC MH+OD+SH · BEAM — · LME — | 1366 (1402) | 1427 (1463) | 1439 (1475) | — | — |
| S10-007 | fill | pool | LOC MH · BEAM EO+MS+PF+SUM · LME multi+temp | 968 (981) | 1238 (1251) | 1309 (1322) | — | — |
| S10-008 | exact | ingest | LOC MH+OD+SH · BEAM IE · LME asst+multi+user | 1366 (1402) | 1427 (1463) | 1439 (1475) | 1457 | 1514 |
| S10-009 | short | ingest | LOC OD+SH · BEAM SUM · LME — | 1194 (1217) | 1316 (1339) | 1356 (1379) | 1331 | 1478 |
| S10-010 | short | seat | LOC OD+SH · BEAM SUM · LME — | 1005 | — | — | — | — |
| S10-011 | year | ingest | LOC OD+TM · BEAM EO+TR · LME temp | 982 (1004) | 1235 (1257) | 1314 (1336) | 1268 | 1465 |
| S10-012 | inherit | ingest | LOC OD · BEAM PF+SUM · LME pref | 856 (870) | 1170 (1184) | 1261 (1275) | 1189 | 1443 |
| S10-013 | card | ingest | LOC MH · BEAM SUM · LME — | 968 (981) | 1238 (1251) | 1309 (1322) | 1259 | 1452 |
| S10-014 | year | ingest | LOC OD+TM · BEAM EO+TR · LME temp | 982 (1004) | 1235 (1257) | 1314 (1336) | 1268 | 1465 |
| S10-015 | alias | ingest | LOC MH+OD+SH · BEAM PF · LME pref | 1366 (1402) | 1427 (1463) | 1439 (1475) | 1457 | 1514 |
| S10-016 | short | ingest | LOC MH+SH · BEAM — · LME — | 1306 (1328) | 1384 (1406) | 1404 (1426) | 1401 | 1487 |
| S10-017 | short | pool | LOC — · BEAM IF · LME asst | — | — | — | — | — |
| S10-018 | year | ingest | LOC MH+TM · BEAM EO+TR · LME temp | 1094 (1115) | 1303 (1324) | 1362 (1383) | 1338 | 1474 |
| S10-019 | short | seat | LOC MH+SH · BEAM SUM · LME — | 1049 | — | — | — | — |
| S10-020 | year | ingest | LOC TM · BEAM EO+IF+TR · LME asst+temp | 922 (930) | 1192 (1200) | 1279 (1287) | 1212 | 1438 |
| S10-021 | year | ingest | LOC OD+TM · BEAM EO+TR · LME temp | 982 (1004) | 1235 (1257) | 1314 (1336) | 1268 | 1465 |
| S10-022 | alias | ingest | LOC OD+SH · BEAM PF · LME pref | 1194 (1217) | 1316 (1339) | 1356 (1379) | 1331 | 1478 |
| S10-023 | exact | ingest | LOC OD+SH · BEAM IE · LME asst+multi+user | 1194 (1217) | 1316 (1339) | 1356 (1379) | 1331 | 1478 |
| S10-024 | short | seat | LOC MH+OD+SH · BEAM — · LME — | 1066 | — | — | — | — |
| S10-025 | exact | ingest | LOC MH+SH · BEAM IE · LME asst+multi+user | 1306 (1328) | 1384 (1406) | 1404 (1426) | 1401 | 1487 |
| S10-026 | fill | pool | LOC MH · BEAM EO+MS+PF+SUM · LME multi+temp | 968 (981) | 1238 (1251) | 1309 (1322) | — | — |
| S10-027 | exact | pool | LOC MH+SH · BEAM IE · LME asst+multi+user | 1306 (1328) | 1384 (1406) | 1404 (1426) | — | — |
| S10-028 | exact | ingest | LOC OD+SH · BEAM IE+SUM · LME asst+multi+user | 1194 (1217) | 1316 (1339) | 1356 (1379) | 1331 | 1478 |
| S10-029 | year | seat | LOC TM · BEAM EO+IF+TR · LME asst+temp | 857 | — | — | — | — |
| S10-030 | year | pool | LOC OD+TM · BEAM EO+TR · LME temp | 982 (1004) | 1235 (1257) | 1314 (1336) | — | — |
| S10-031 | year | ingest | LOC OD+TM · BEAM EO+TR · LME temp | 982 (1004) | 1235 (1257) | 1314 (1336) | 1268 | 1465 |
| S10-032 | short | ingest | LOC OD+SH · BEAM SUM · LME — | 1194 (1217) | 1316 (1339) | 1356 (1379) | 1331 | 1478 |
| S10-033 | exact | ingest | LOC MH+SH · BEAM EO+IE · LME asst+multi+user | 1306 (1328) | 1384 (1406) | 1404 (1426) | 1401 | 1487 |
| S10-034 | card | ingest | LOC MH · BEAM SUM · LME — | 968 (981) | 1238 (1251) | 1309 (1322) | 1259 | 1452 |
| S10-035 | year | ingest | LOC TM · BEAM EO+TR · LME temp | 922 (930) | 1192 (1200) | 1279 (1287) | 1212 | 1438 |
| S10-036 | card | seat | LOC MH+OD · BEAM SUM · LME — | 874 | — | — | — | — |
| S10-037 | alias | ingest | LOC SH · BEAM PF · LME pref | 1134 (1143) | 1273 (1282) | 1321 (1330) | 1275 | 1451 |
| S10-038 | year | ingest | LOC TM · BEAM EO+KU+TR · LME ku+temp | 922 (930) | 1192 (1200) | 1279 (1287) | 1212 | 1438 |
| S10-039 | exact | ingest | LOC MH+SH · BEAM IE · LME asst+multi+user | 1306 (1328) | 1384 (1406) | 1404 (1426) | 1401 | 1487 |
| S10-040 | exact | ingest | LOC MH+OD+SH · BEAM IE · LME asst+multi+user | 1366 (1402) | 1427 (1463) | 1439 (1475) | 1457 | 1514 |
| S10-041 | inherit | ingest | LOC OD · BEAM PF · LME pref | 856 (870) | 1170 (1184) | 1261 (1275) | 1189 | 1443 |
| S10-042 | alias | ingest | LOC OD+SH · BEAM PF · LME pref | 1194 (1217) | 1316 (1339) | 1356 (1379) | 1331 | 1478 |
| S10-043 | short | seat | LOC MH+SH · BEAM SUM · LME — | 1049 | — | — | — | — |
| S10-044 | short | pool | LOC — · BEAM IF · LME asst | — | — | — | — | — |
| S10-045 | year | ingest | LOC TM · BEAM EO+TR · LME temp | 922 (930) | 1192 (1200) | 1279 (1287) | 1212 | 1438 |
| S10-046 | year | ingest | LOC TM · BEAM EO+IF+TR · LME asst+temp | 922 (930) | 1192 (1200) | 1279 (1287) | 1212 | 1438 |
| S10-047 | year | ingest | LOC OD+TM · BEAM EO+TR · LME temp | 982 (1004) | 1235 (1257) | 1314 (1336) | 1268 | 1465 |
| S10-048 | card | seat | LOC MH · BEAM SUM · LME — | 857 | — | — | — | — |
| S10-049 | year | seat | LOC OD+TM · BEAM EO+TR · LME temp | 874 | — | — | — | — |
| S10-050 | fill | pool | LOC MH · BEAM EO+MS+PF+SUM · LME multi+temp | 968 (981) | 1238 (1251) | 1309 (1322) | — | — |
| S10-051 | short | seat | LOC MH+OD+SH · BEAM — · LME — | 1066 | — | — | — | — |
| S10-052 | exact | pool | LOC MH+OD+SH · BEAM IE · LME asst+multi+user | 1366 (1402) | 1427 (1463) | 1439 (1475) | — | — |
| S10-053 | exact | ingest | LOC OD+SH · BEAM IE+SUM · LME asst+multi+user | 1194 (1217) | 1316 (1339) | 1356 (1379) | 1331 | 1478 |
| S10-054 | year | ingest | LOC OD+TM · BEAM EO+TR · LME temp | 982 (1004) | 1235 (1257) | 1314 (1336) | 1268 | 1465 |
| S10-055 | short | seat | LOC OD+SH · BEAM IF+SUM · LME asst | 1005 | — | — | — | — |
| S10-056 | fill | pool | LOC MH+OD · BEAM EO+MS+PF+SUM · LME multi+temp | 1028 (1055) | 1281 (1308) | 1344 (1371) | — | — |
| S10-057 | fill | pool | LOC MH+OD · BEAM EO+MS+PF+SUM · LME multi+temp | 1028 (1055) | 1281 (1308) | 1344 (1371) | — | — |
| S10-058 | fill | pool | LOC MH · BEAM EO+MS+PF+SUM · LME multi+temp | 968 (981) | 1238 (1251) | 1309 (1322) | — | — |
| S10-059 | card | ingest | LOC MH · BEAM SUM · LME — | 968 (981) | 1238 (1251) | 1309 (1322) | 1259 | 1452 |
| S10-060 | year | ingest | LOC TM · BEAM EO+TR · LME temp | 922 (930) | 1192 (1200) | 1279 (1287) | 1212 | 1438 |
| S10-061 | alias | ingest | LOC OD+SH · BEAM PF · LME pref | 1194 (1217) | 1316 (1339) | 1356 (1379) | 1331 | 1478 |
| S10-062 | year | seat | LOC MH+TM · BEAM EO+TR · LME temp | 918 | — | — | — | — |
| S10-063 | year | ingest | LOC OD+TM · BEAM EO+SUM+TR · LME temp | 982 (1004) | 1235 (1257) | 1314 (1336) | 1268 | 1465 |
| S10-064 | exact | ingest | LOC MH+SH · BEAM IE · LME asst+multi+user | 1306 (1328) | 1384 (1406) | 1404 (1426) | 1401 | 1487 |
| S10-065 | year | seat | LOC OD+TM · BEAM EO+TR · LME temp | 874 | — | — | — | — |
| S10-066 | exact | ingest | LOC MH+OD+SH · BEAM IE · LME asst+multi+user | 1366 (1402) | 1427 (1463) | 1439 (1475) | 1457 | 1514 |
| S10-067 | exact | ingest | LOC MH+OD+SH · BEAM IE · LME asst+multi+user | 1366 (1402) | 1427 (1463) | 1439 (1475) | 1457 | 1514 |
| S10-068 | year | ingest | LOC TM · BEAM EO+TR · LME temp | 922 (930) | 1192 (1200) | 1279 (1287) | 1212 | 1438 |
| S10-069 | fill | pool | LOC MH · BEAM EO+MS+PF+SUM · LME multi+temp | 968 (981) | 1238 (1251) | 1309 (1322) | — | — |
| S10-070 | year | seat | LOC TM · BEAM EO+TR · LME temp | 857 | — | — | — | — |
| S10-071 | alias | ingest | LOC OD+SH · BEAM PF · LME pref | 1194 (1217) | 1316 (1339) | 1356 (1379) | 1331 | 1478 |
| S10-072 | year | ingest | LOC TM · BEAM EO+SUM+TR · LME temp | 922 (930) | 1192 (1200) | 1279 (1287) | 1212 | 1438 |
| S10-073 | short | seat | LOC MH+OD+SH · BEAM — · LME — | 1066 | — | — | — | — |
| S10-074 | short | pool | LOC — · BEAM IF · LME asst | — | — | — | — | — |
| S10-075 | exact | pool | LOC OD+SH · BEAM IE · LME asst+multi+user | 1194 (1217) | 1316 (1339) | 1356 (1379) | — | — |
| S10-076 | exact | ingest | LOC MH+OD+SH · BEAM IE · LME asst+multi+user | 1366 (1402) | 1427 (1463) | 1439 (1475) | 1457 | 1514 |
| S10-077 | year | ingest | LOC MH+TM · BEAM EO+TR · LME temp | 1094 (1115) | 1303 (1324) | 1362 (1383) | 1338 | 1474 |
| S10-078 | short | ingest | LOC OD+SH · BEAM IF · LME asst | 1194 (1217) | 1316 (1339) | 1356 (1379) | 1331 | 1478 |
| S10-079 | alias | ingest | LOC OD+SH · BEAM EO+PF · LME pref | 1194 (1217) | 1316 (1339) | 1356 (1379) | 1331 | 1478 |
| S10-080 | fill | pool | LOC MH · BEAM EO+MS+PF+SUM · LME multi+temp | 968 (981) | 1238 (1251) | 1309 (1322) | — | — |
| S10-081 | year | seat | LOC TM · BEAM EO+TR · LME temp | 857 | — | — | — | — |
| S10-082 | year | seat | LOC OD+TM · BEAM EO+TR · LME temp | 874 | — | — | — | — |
| S10-083 | year | seat | LOC TM · BEAM EO+IF+TR · LME asst+temp | 857 | — | — | — | — |
| S10-084 | fill | ingest | LOC MH · BEAM EO+MS+PF+SUM · LME multi+temp | 968 (981) | 1238 (1251) | 1309 (1322) | 1259 | 1452 |
| S10-085 | fill | seat | LOC MH · BEAM EO+MS+PF+SUM · LME multi+temp | 857 | — | — | — | — |
| S10-086 | pin | seat | LOC OD+SH · BEAM IF · LME asst | 1005 | — | — | — | — |
| S10-087 | year | ingest | LOC TM · BEAM EO+SUM+TR · LME temp | 922 (930) | 1192 (1200) | 1279 (1287) | 1212 | 1438 |
| S10-088 | inherit | ingest | LOC OD · BEAM PF · LME pref | 856 (870) | 1170 (1184) | 1261 (1275) | 1189 | 1443 |
| S10-089 | pin | seat | LOC — · BEAM IF · LME asst | — | — | — | — | — |
| S10-090 | year | seat | LOC OD+TM · BEAM EO+TR · LME temp | 874 | — | — | — | — |

#### BEAM

| id | group | lever | Turn@1 | Turn@5 | Turn@10 | All@5 | All@10 | Session@1 | Session@5 | Session@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S10-001 | exact | ingest | 191 (208) | 259 (276) | 284 (301) | — | — | 193 | 289 | 303 |
| S10-002 | exact | ingest | 200 (209) | 268 (277) | 287 (296) | — | — | 191 | 282 | 301 |
| S10-003 | exact | ingest | 196 (206) | 260 (270) | 279 (289) | — | — | 188 | 276 | 298 |
| S10-004 | year | ingest | 201 (202) | 266 (267) | 285 (286) | — | — | 182 | 269 | 288 |
| S10-005 | year | seat | 173 | — | — | — | — | — | — | — |
| S10-006 | short | pool | — | — | — | — | — | — | — | — |
| S10-007 | fill | pool | 258 (270) | 302 (314) | 311 (323) | 258 | 273 | — (251 if the stem is also a feat) | — (311 if the stem is also a feat) | — (317 if the stem is also a feat) |
| S10-008 | exact | ingest | 168 (176) | 243 (251) | 270 (278) | — | — | 158 | 264 | 288 |
| S10-009 | short | ingest | 184 (186) | 255 (257) | 276 (278) | — | — | 163 | 261 | 285 |
| S10-010 | short | seat | 167 | — | — | — | — | — | — | — |
| S10-011 | year | ingest | 201 (202) | 266 (267) | 285 (286) | — | — | 182 | 269 | 288 |
| S10-012 | inherit | ingest | 207 (218) | 271 (282) | 290 (301) | 182 | 200 | 198 | 286 | 300 |
| S10-013 | card | ingest | 184 (186) | 255 (257) | 276 (278) | — | — | 163 | 261 | 285 |
| S10-014 | year | ingest | 201 (202) | 266 (267) | 285 (286) | — | — | 182 | 269 | 288 |
| S10-015 | alias | ingest | 179 (188) | 254 (263) | 281 (290) | — | — | 168 | 274 | 290 |
| S10-016 | short | ingest | — | — | — | — | — | — | — | — |
| S10-017 | short | pool | 185 (190) | 255 (260) | 281 (286) | — | — | — | — | — |
| S10-018 | year | ingest | 201 (202) | 266 (267) | 285 (286) | — | — | 182 | 269 | 288 |
| S10-019 | short | seat | 167 | — | — | — | — | — | — | — |
| S10-020 | year | ingest | 230 (236) | 283 (289) | 299 (305) | — | — | 218 | 293 | 311 |
| S10-021 | year | ingest | 201 (202) | 266 (267) | 285 (286) | — | — | 182 | 269 | 288 |
| S10-022 | alias | ingest | 179 (188) | 254 (263) | 281 (290) | — | — | 168 | 274 | 290 |
| S10-023 | exact | ingest | 168 (176) | 243 (251) | 270 (278) | — | — | 158 | 264 | 288 |
| S10-024 | short | seat | — | — | — | — | — | — | — | — |
| S10-025 | exact | ingest | 168 (176) | 243 (251) | 270 (278) | — | — | 158 | 264 | 288 |
| S10-026 | fill | pool | 258 (270) | 302 (314) | 311 (323) | 258 | 273 | — (251 if the stem is also a feat) | — (311 if the stem is also a feat) | — (317 if the stem is also a feat) |
| S10-027 | exact | pool | 168 (176) | 243 (251) | 270 (278) | — | — | — | — | — |
| S10-028 | exact | ingest | 196 (206) | 260 (270) | 279 (289) | — | — | 188 | 276 | 298 |
| S10-029 | year | seat | 185 | — | — | — | — | — | — | — |
| S10-030 | year | pool | 201 (202) | 266 (267) | 285 (286) | — | — | — | — | — |
| S10-031 | year | ingest | 201 (202) | 266 (267) | 285 (286) | — | — | 182 | 269 | 288 |
| S10-032 | short | ingest | 184 (186) | 255 (257) | 276 (278) | — | — | 163 | 261 | 285 |
| S10-033 | exact | ingest | 200 (209) | 268 (277) | 287 (296) | — | — | 191 | 282 | 301 |
| S10-034 | card | ingest | 184 (186) | 255 (257) | 276 (278) | — | — | 163 | 261 | 285 |
| S10-035 | year | ingest | 201 (202) | 266 (267) | 285 (286) | — | — | 182 | 269 | 288 |
| S10-036 | card | seat | 167 | — | — | — | — | — | — | — |
| S10-037 | alias | ingest | 179 (188) | 254 (263) | 281 (290) | — | — | 168 | 274 | 290 |
| S10-038 | year | ingest | 210 (212) | 268 (270) | 285 (287) | — | — | 198 | 272 | 290 |
| S10-039 | exact | ingest | 168 (176) | 243 (251) | 270 (278) | — | — | 158 | 264 | 288 |
| S10-040 | exact | ingest | 168 (176) | 243 (251) | 270 (278) | — | — | 158 | 264 | 288 |
| S10-041 | inherit | ingest | 179 (188) | 254 (263) | 281 (290) | 146 | 164 | 168 | 274 | 290 |
| S10-042 | alias | ingest | 179 (188) | 254 (263) | 281 (290) | — | — | 168 | 274 | 290 |
| S10-043 | short | seat | 167 | — | — | — | — | — | — | — |
| S10-044 | short | pool | 185 (190) | 255 (260) | 281 (286) | — | — | — | — | — |
| S10-045 | year | ingest | 201 (202) | 266 (267) | 285 (286) | — | — | 182 | 269 | 288 |
| S10-046 | year | ingest | 230 (236) | 283 (289) | 299 (305) | — | — | 218 | 293 | 311 |
| S10-047 | year | ingest | 201 (202) | 266 (267) | 285 (286) | — | — | 182 | 269 | 288 |
| S10-048 | card | seat | 167 | — | — | — | — | — | — | — |
| S10-049 | year | seat | 173 | — | — | — | — | — | — | — |
| S10-050 | fill | pool | 258 (270) | 302 (314) | 311 (323) | 258 | 273 | — (251 if the stem is also a feat) | — (311 if the stem is also a feat) | — (317 if the stem is also a feat) |
| S10-051 | short | seat | — | — | — | — | — | — | — | — |
| S10-052 | exact | pool | 168 (176) | 243 (251) | 270 (278) | — | — | — | — | — |
| S10-053 | exact | ingest | 196 (206) | 260 (270) | 279 (289) | — | — | 188 | 276 | 298 |
| S10-054 | year | ingest | 201 (202) | 266 (267) | 285 (286) | — | — | 182 | 269 | 288 |
| S10-055 | short | seat | 179 | — | — | — | — | — | — | — |
| S10-056 | fill | pool | 258 (270) | 302 (314) | 311 (323) | 258 | 273 | — (251 if the stem is also a feat) | — (311 if the stem is also a feat) | — (317 if the stem is also a feat) |
| S10-057 | fill | pool | 258 (270) | 302 (314) | 311 (323) | 258 | 273 | — (251 if the stem is also a feat) | — (311 if the stem is also a feat) | — (317 if the stem is also a feat) |
| S10-058 | fill | pool | 258 (270) | 302 (314) | 311 (323) | 258 | 273 | — (251 if the stem is also a feat) | — (311 if the stem is also a feat) | — (317 if the stem is also a feat) |
| S10-059 | card | ingest | 184 (186) | 255 (257) | 276 (278) | — | — | 163 | 261 | 285 |
| S10-060 | year | ingest | 201 (202) | 266 (267) | 285 (286) | — | — | 182 | 269 | 288 |
| S10-061 | alias | ingest | 179 (188) | 254 (263) | 281 (290) | — | — | 168 | 274 | 290 |
| S10-062 | year | seat | 173 | — | — | — | — | — | — | — |
| S10-063 | year | ingest | 229 (232) | 283 (286) | 294 (297) | — | — | 212 | 281 | 298 |
| S10-064 | exact | ingest | 168 (176) | 243 (251) | 270 (278) | — | — | 158 | 264 | 288 |
| S10-065 | year | seat | 173 | — | — | — | — | — | — | — |
| S10-066 | exact | ingest | 168 (176) | 243 (251) | 270 (278) | — | — | 158 | 264 | 288 |
| S10-067 | exact | ingest | 168 (176) | 243 (251) | 270 (278) | — | — | 158 | 264 | 288 |
| S10-068 | year | ingest | 201 (202) | 266 (267) | 285 (286) | — | — | 182 | 269 | 288 |
| S10-069 | fill | pool | 258 (270) | 302 (314) | 311 (323) | 258 | 273 | — (251 if the stem is also a feat) | — (311 if the stem is also a feat) | — (317 if the stem is also a feat) |
| S10-070 | year | seat | 173 | — | — | — | — | — | — | — |
| S10-071 | alias | ingest | 179 (188) | 254 (263) | 281 (290) | — | — | 168 | 274 | 290 |
| S10-072 | year | ingest | 229 (232) | 283 (286) | 294 (297) | — | — | 212 | 281 | 298 |
| S10-073 | short | seat | — | — | — | — | — | — | — | — |
| S10-074 | short | pool | 185 (190) | 255 (260) | 281 (286) | — | — | — | — | — |
| S10-075 | exact | pool | 168 (176) | 243 (251) | 270 (278) | — | — | — | — | — |
| S10-076 | exact | ingest | 168 (176) | 243 (251) | 270 (278) | — | — | 158 | 264 | 288 |
| S10-077 | year | ingest | 201 (202) | 266 (267) | 285 (286) | — | — | 182 | 269 | 288 |
| S10-078 | short | ingest | 185 (190) | 255 (260) | 281 (286) | — | — | 169 | 273 | 298 |
| S10-079 | alias | ingest | 211 (221) | 279 (289) | 298 (308) | — | — | 201 | 292 | 303 |
| S10-080 | fill | pool | 258 (270) | 302 (314) | 311 (323) | 258 | 273 | — (251 if the stem is also a feat) | — (311 if the stem is also a feat) | — (317 if the stem is also a feat) |
| S10-081 | year | seat | 173 | — | — | — | — | — | — | — |
| S10-082 | year | seat | 173 | — | — | — | — | — | — | — |
| S10-083 | year | seat | 185 | — | — | — | — | — | — | — |
| S10-084 | fill | ingest | 258 (270) | 302 (314) | 311 (323) | 258 | 273 | 251 | 311 | 317 |
| S10-085 | fill | seat | 194 | — | — | — | — | — (251 if the stem is also a feat) | — (311 if the stem is also a feat) | — (317 if the stem is also a feat) |
| S10-086 | pin | seat | 168 | — | — | — | — | — | — | — |
| S10-087 | year | ingest | 229 (232) | 283 (286) | 294 (297) | — | — | 212 | 281 | 298 |
| S10-088 | inherit | ingest | 179 (188) | 254 (263) | 281 (290) | 146 | 164 | 168 | 274 | 290 |
| S10-089 | pin | seat | 168 | — | — | — | — | — | — | — |
| S10-090 | year | seat | 173 | — | — | — | — | — | — | — |

#### LME

| id | group | lever | Turn@1 | Turn@5 | Turn@10 | Turn All@5 | Session@1 | Session@5 | Session@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| S10-001 | exact | ingest | 408 | 442 | 448 | — | 443 | 463 | 468 |
| S10-002 | exact | ingest | 408 | 442 | 448 | — | 443 | 463 | 468 |
| S10-003 | exact | ingest | 408 | 442 | 448 | — | 443 | 463 | 468 |
| S10-004 | year | ingest | 382 (384) | 442 (444) | 446 (448) | — | 443 | 466 | 470 |
| S10-005 | year | seat | 368 | — | — | — | — | — | — |
| S10-006 | short | pool | — | — | — | — | — | — | — |
| S10-007 | fill | pool | 415 (417) | 450 (452) | 454 (456) | 435 | — | — | — |
| S10-008 | exact | ingest | 408 | 442 | 448 | — | 443 | 463 | 468 |
| S10-009 | short | ingest | — | — | — | — | — | — | — |
| S10-010 | short | seat | — | — | — | — | — | — | — |
| S10-011 | year | ingest | 382 (384) | 442 (444) | 446 (448) | — | 443 | 466 | 470 |
| S10-012 | inherit | ingest | 368 | 439 | 448 | 330 | 443 | 467 | 468 |
| S10-013 | card | ingest | — | — | — | — | — | — | — |
| S10-014 | year | ingest | 382 (384) | 442 (444) | 446 (448) | — | 443 | 466 | 470 |
| S10-015 | alias | ingest | 368 | 439 | 448 | — | 443 | 467 | 468 |
| S10-016 | short | ingest | — | — | — | — | — | — | — |
| S10-017 | short | pool | 365 | 432 | 439 | — | — | — | — |
| S10-018 | year | ingest | 382 (384) | 442 (444) | 446 (448) | — | 443 | 466 | 470 |
| S10-019 | short | seat | — | — | — | — | — | — | — |
| S10-020 | year | ingest | 397 (399) | 446 (448) | 448 (450) | — | 443 | 466 | 470 |
| S10-021 | year | ingest | 382 (384) | 442 (444) | 446 (448) | — | 443 | 466 | 470 |
| S10-022 | alias | ingest | 368 | 439 | 448 | — | 443 | 467 | 468 |
| S10-023 | exact | ingest | 408 | 442 | 448 | — | 443 | 463 | 468 |
| S10-024 | short | seat | — | — | — | — | — | — | — |
| S10-025 | exact | ingest | 408 | 442 | 448 | — | 443 | 463 | 468 |
| S10-026 | fill | pool | 415 (417) | 450 (452) | 454 (456) | 435 | — | — | — |
| S10-027 | exact | pool | 408 | 442 | 448 | — | — | — | — |
| S10-028 | exact | ingest | 408 | 442 | 448 | — | 443 | 463 | 468 |
| S10-029 | year | seat | 379 | — | — | — | — | — | — |
| S10-030 | year | pool | 382 (384) | 442 (444) | 446 (448) | — | — | — | — |
| S10-031 | year | ingest | 382 (384) | 442 (444) | 446 (448) | — | 443 | 466 | 470 |
| S10-032 | short | ingest | — | — | — | — | — | — | — |
| S10-033 | exact | ingest | 408 | 442 | 448 | — | 443 | 463 | 468 |
| S10-034 | card | ingest | — | — | — | — | — | — | — |
| S10-035 | year | ingest | 382 (384) | 442 (444) | 446 (448) | — | 443 | 466 | 470 |
| S10-036 | card | seat | — | — | — | — | — | — | — |
| S10-037 | alias | ingest | 368 | 439 | 448 | — | 443 | 467 | 468 |
| S10-038 | year | ingest | 392 (394) | 443 (445) | 446 (448) | — | 444 | 466 | 470 |
| S10-039 | exact | ingest | 408 | 442 | 448 | — | 443 | 463 | 468 |
| S10-040 | exact | ingest | 408 | 442 | 448 | — | 443 | 463 | 468 |
| S10-041 | inherit | ingest | 368 | 439 | 448 | 330 | 443 | 467 | 468 |
| S10-042 | alias | ingest | 368 | 439 | 448 | — | 443 | 467 | 468 |
| S10-043 | short | seat | — | — | — | — | — | — | — |
| S10-044 | short | pool | 365 | 432 | 439 | — | — | — | — |
| S10-045 | year | ingest | 382 (384) | 442 (444) | 446 (448) | — | 443 | 466 | 470 |
| S10-046 | year | ingest | 397 (399) | 446 (448) | 448 (450) | — | 443 | 466 | 470 |
| S10-047 | year | ingest | 382 (384) | 442 (444) | 446 (448) | — | 443 | 466 | 470 |
| S10-048 | card | seat | — | — | — | — | — | — | — |
| S10-049 | year | seat | 368 | — | — | — | — | — | — |
| S10-050 | fill | pool | 415 (417) | 450 (452) | 454 (456) | 435 | — | — | — |
| S10-051 | short | seat | — | — | — | — | — | — | — |
| S10-052 | exact | pool | 408 | 442 | 448 | — | — | — | — |
| S10-053 | exact | ingest | 408 | 442 | 448 | — | 443 | 463 | 468 |
| S10-054 | year | ingest | 382 (384) | 442 (444) | 446 (448) | — | 443 | 466 | 470 |
| S10-055 | short | seat | 361 | — | — | — | — | — | — |
| S10-056 | fill | pool | 415 (417) | 450 (452) | 454 (456) | 435 | — | — | — |
| S10-057 | fill | pool | 415 (417) | 450 (452) | 454 (456) | 435 | — | — | — |
| S10-058 | fill | pool | 415 (417) | 450 (452) | 454 (456) | 435 | — | — | — |
| S10-059 | card | ingest | — | — | — | — | — | — | — |
| S10-060 | year | ingest | 382 (384) | 442 (444) | 446 (448) | — | 443 | 466 | 470 |
| S10-061 | alias | ingest | 368 | 439 | 448 | — | 443 | 467 | 468 |
| S10-062 | year | seat | 368 | — | — | — | — | — | — |
| S10-063 | year | ingest | 382 (384) | 442 (444) | 446 (448) | — | 443 | 466 | 470 |
| S10-064 | exact | ingest | 408 | 442 | 448 | — | 443 | 463 | 468 |
| S10-065 | year | seat | 368 | — | — | — | — | — | — |
| S10-066 | exact | ingest | 408 | 442 | 448 | — | 443 | 463 | 468 |
| S10-067 | exact | ingest | 408 | 442 | 448 | — | 443 | 463 | 468 |
| S10-068 | year | ingest | 382 (384) | 442 (444) | 446 (448) | — | 443 | 466 | 470 |
| S10-069 | fill | pool | 415 (417) | 450 (452) | 454 (456) | 435 | — | — | — |
| S10-070 | year | seat | 368 | — | — | — | — | — | — |
| S10-071 | alias | ingest | 368 | 439 | 448 | — | 443 | 467 | 468 |
| S10-072 | year | ingest | 382 (384) | 442 (444) | 446 (448) | — | 443 | 466 | 470 |
| S10-073 | short | seat | — | — | — | — | — | — | — |
| S10-074 | short | pool | 365 | 432 | 439 | — | — | — | — |
| S10-075 | exact | pool | 408 | 442 | 448 | — | — | — | — |
| S10-076 | exact | ingest | 408 | 442 | 448 | — | 443 | 463 | 468 |
| S10-077 | year | ingest | 382 (384) | 442 (444) | 446 (448) | — | 443 | 466 | 470 |
| S10-078 | short | ingest | 365 | 432 | 439 | — | 430 | 463 | 468 |
| S10-079 | alias | ingest | 368 | 439 | 448 | — | 443 | 467 | 468 |
| S10-080 | fill | pool | 415 (417) | 450 (452) | 454 (456) | 435 | — | — | — |
| S10-081 | year | seat | 368 | — | — | — | — | — | — |
| S10-082 | year | seat | 368 | — | — | — | — | — | — |
| S10-083 | year | seat | 379 | — | — | — | — | — | — |
| S10-084 | fill | ingest | 415 (417) | 450 (452) | 454 (456) | 435 | 454 | 466 | 470 |
| S10-085 | fill | seat | 393 | — | — | — | — | — | — |
| S10-086 | pin | seat | 361 | — | — | — | — | — | — |
| S10-087 | year | ingest | 382 (384) | 442 (444) | 446 (448) | — | 443 | 466 | 470 |
| S10-088 | inherit | ingest | 368 | 439 | 448 | 330 | 443 | 467 | 468 |
| S10-089 | pin | seat | 361 | — | — | — | — | — | — |
| S10-090 | year | seat | 368 | — | — | — | — | — | — |

Reading a row against the need, without adding it to any other row:

- S10-001 is the LoCoMo Turn@1 row whose applies set is Single-Hop, Multi-Hop, and Open-Domain together. Its in-pool cap is 1366, 17 short of 1383. The parenthetical 1402 clears the bench, and only if the key is also what those three categories' 36 `no_pool` golds lack. An exact row that leaves one of the three categories out stays under 1383 even in the parenthetical. Single-Hop is 263 of the attachment Turn@1 gap of 587 (796 up to the need of 1383). Step 34 has already taken 9 of those 587, and the attachment does not say which category received them.
- A year ingest (S10-004, S10-020, S10-035, and the other ingest rows in that group) is the Temporal lever, the LME temporal-reasoning lever, and the BEAM event_ordering and temporal_reasoning lever. It does not, alone, clear LoCoMo Turn@1, because Single-Hop is untouched.
- A fill row on a pool or ingest hook (S10-026, S10-050, S10-058, S10-069, S10-080, S10-084) is the BEAM All@5 / All@10 lever and the LME Turn All@5 lever. S10-085 is the same group on a seat hook, so its All@5 cell stays `—`: rotating a run that is already seated does not pull the missing golds in. The All@5 cell is the bench cap if every gold set in the named categories shares the stem. That cap is under 320. The LME All@5 cell clears 423. A pool fill shows `—` on Session until the stem is also a feat. An ingest fill already has the feat, so its Session cell moves.
- An inherit ingest (S10-012, S10-088) copies the closing leaf onto earlier leaves of the same stem, at weight 1/2, and stops the copy when the query already names the leaf. That feat is in the session bag, so it can move preference_following and LME single-session-preference. All@5 moves for preference_following, and for summarization on the row that names it (S10-012).
- A seat (S10-010, S10-049, S10-055, S10-082, S10-083, S10-086, S10-089, and the other 6265 rows) can finish a category whose Turn@5 already meets the need. It cannot finish LoCoMo Turn@1, BEAM Turn@1, BEAM All@5, or any Session metric.

Build order that follows from the caps: S10-001 (exact key, LoCoMo Turn@1 and Session@1), S10-026 (set fill, BEAM All@5 and LME Turn All@5), S10-004 (year class, Temporal and both temporal categories), S10-012 (inherit, preference session and LME preference), S10-015 (quoted alias, Open-Domain `no_pool`), S10-038 (line-0 year, the only row that names BEAM knowledge_update), S10-017 (command shape, instruction_following `no_pool`), S10-010 (shorter exact unit, the Single-Hop residue that is already inside Turn@5), S10-088 (stop the inherit when the key is exact), S10-050 (exact suffix first, then fill the rest under the quantifier). contradiction_resolution is not named by any row. Its Turn@1 need is a seat inside an already-complete top 5, and its Session@1 and All@5 needs are not.

