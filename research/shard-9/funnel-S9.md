# Funnel S9

Attachment identity: `perq-bd2693e2.json` field `bin` is `/workspace/c2-algos-bins/t51b-attr-unseated-opener-t10`, `bin_sha256` `e93411ac566c4c83f92a83204e70ea85ae743445a3256dc5cb4eacc2523e6f84`. ALGOS_LOG names that binary WIN #44 `bd2693e2` (attribution unseated-session opener, Turn@10). The miss map header matches that binary. The JSON is therefore bd2693e2, not a post-Step-32 or post-Step-33 rescore. `C2_STEP33_LOG.md` on this branch records one later Open-Domain Turn@1 gain, 18 to 19, with top 5 unchanged. The shard's "19/92" is that later log. Ceilings below use the attachment, and the +1 does not change them: top 5 is still 35, and 35 is far under 83.

`tools/c2_funnel.py` is not in this checkout. No replay dumps are in the workspace. Scene-rank and per-slice pool@50 were not recomputed.

## What the replay dump fields mean

`write_one_replay_dump` (`crates/vestige-core/src/retrieval.rs`, function at line 7426) writes one JSON object per query.

`snap.seated` is the official turn list after the gatekeeper and the C2 seats, including `apply_c2_step33_triangulation` at lines 6265-6272. `maybe_dump_replay` is the next call, line 6273.

- `final_ids` and `seated_ids` are `snap.seated`, turn ids in official order. `head_id` is `seated[0]`.
- `rank` on a candidate is the index of that turn in `seated`. Missing when the turn is not seated.
- `pre_ids` is built at lines 7429-7440: start with `seated`, then append turns in `t_scene` descending order while `t_scene > 0`, until 50. So `pre_ids[0]` is `final_ids[0]` whenever the seated list is nonempty. **Gold at pre_rank 0 and gold at final rank 0 are the same number by construction.** A scene-rank 0, the argmax of `t_scene` before seating, is a different quantity and is not a field in the attached JSON or miss map.
- `rank_pre` is the index in that 50-list.
- `t_scene` is the scene-pooled score. Seating does not rewrite it.
- `t_own` is the turn's own lexical score (uncarried BM25 plus the contiguous-bigram bonus), the first sightline of Step 33.
- `t_turn_raw` is the raw turn score before that pooling.
- `t_scene_sess` is the scene score aggregated at session grain.
- `s_score` and `bridge_boost` are the official session rank score and its bridge term. Step 33's second sightline is their sum.
- `k` is the official cutoff. `pool_size` is `len(pre_ids)`, at most 50.
- `stability` is the stored turn stability. `session` / `session_id` / `session_order` place the turn. `ts` is the session timestamp.
- `role` is `t.speaker`. Designs in this shard do not branch on it.
- `ends_with_question`, `is_question`, `turn_ic_novel`, and `qtok_present` are dump-only and may read `TurnIx.body`. Ranking must not.
- `superseded`, `response_turn_id`, `parent_user_turn_id` are structural links.
- `rsb_seated` is membership in `seated`. `rsb_cause_id`, `rsb_cause_rank`, `rsb_symptom_ids` are the existing RSB links. These designs do not write `rsb_seat_behind`.
- `exact_edges`, `policy_backlinks`, `degree_closed`, `degree_none` are graph degrees on the pool.
- `candidates` is one object per `pre_ids` turn. `session_turns` dumps every turn of sessions that appear in the official turn top 10 or session top 10.
- Query-side fields: `query_text`, `query_date`, `query_ts`, `wh_slots`, `wh_class`, `is_event_sequence`, `temporal`, `preference`, `assistant_attribution`, `final_session_ids`, `query_tokens`, `n_sessions`, `n_turns`, `first_session_id`.

The wide scored pool used by the miss map is every turn with `t_scene > 0`, built at `retrieval.rs:5633` (`remaining`). That pool is larger than `pre_ids`. The miss-map header "pool" is the wide pool. "pool@50" is the dump's 50-list. They are not the same set.

## Per-question bit layout

There is no per-question writer in this checkout. Layouts are the ones whose column sums equal the aggregate objects exactly.

LoCoMo, n=1536, five bits `[s1, s5, t1, t5, t10]`. Sums: s1 1133, s5 1416, t1 796, t5 1127, t10 1226.

BEAM, n=355, seven bits `[t1, t5, t10, all5, s1, s5, sall5]`. Sums: t1 156, t5 238, t10 267, all5 118, s1 133, s5 249, sall5 137. `all10` 138, `s10` 275, and `sall10` 178 are in the aggregate object and not in the bit vector.

LME, n=470, seven bits `[s1, s5, s10, t1, t5, t10, ta5]` summing to 430, 463, 468, 350, 428, 437, 314. Session all5 424 and all10 447 are not in the vector.

## Miss-map stages

Stages partition Turn@1 misses. `no_pool`: gold not in the wide scored pool. `pool_not_k`: in the wide pool, not in the official k. `k_not5`: in k, not in top 5. `t5_not1`: in top 5, not at rank 0.

Bench headers:

- LME n=470 pool=468 pool@50=463 k=437 T@5=428 T@1=350. no_pool sums to 2, which is 470-468.
- LOC n=1536 pool=1492 pool@50=1415 k=1226 T@5=1127 T@1=796. Category no_pool sums to 44 = 1536-1492. Category pool_not_k sums to 266 = 1492-1226. The 77 turns in the wide pool but outside pool@50 are not assigned to a slice by the attachment.
- BEAM header k=266, JSON t10=267. The difference is WIN #44's one Turn@10 gain, logged as `14:instruction_following:0`. Category stage sums still match Turn@1 miss totals. For instruction following, JSON t10 is 21 while the stage arithmetic gives k=20, so one former pool_not_k is now inside k.

## Five-line funnels

Gold in pool@50, per slice, cannot be computed from the attachments. Gold at a pre_rank distinct from final rank 0 cannot be computed, because the dump writer forces them to be equal. Both lines are printed, with that limit on them.

### Open-Domain (LoCoMo, n=92)

1. Gold in corpus: 92.
2. Gold in pool@50: cannot be computed per slice. Bench LOC pool@50 is 1415 against a wide pool of 1492.
3. Gold at pre_rank 0: 18, and only because pre_rank 0 is final rank 0. A distinct pre-seat scene-rank is not in the attachment.
4. Gold in top 5: 35 (JSON t5).
5. Gold at final rank 0: 18 (JSON t1). Step 33's log later records 19, top 5 unchanged.

Stages: no_pool 14, pool_not_k 35, k_not5 8, t5_not1 17. Misses 74 = 92-18. Wide pool 78. Official k 43.

Slot-1 ceiling (top 5 frozen): 35, under the bar of 83. Wide-pool ceiling: 78, still under 83. At least 5 of the 14 no_pool golds need a new posting before any reordering can reach 90%.

### Multi-Hop (LoCoMo, n=282)

1. Gold in corpus: 282.
2. Gold in pool@50: cannot be computed per slice.
3. Gold at pre_rank 0: 97, identical to final rank 0 by the dump writer.
4. Gold in top 5: 158.
5. Gold at final rank 0: 97.

Stages: no_pool 13, pool_not_k 83, k_not5 28, t5_not1 61. Misses 185 = 282-97. Wide pool 269. Official k 186.

Slot-1 ceiling: 158, under 254. Wide-pool ceiling: 269, which clears 254. Pool reselection among turns with `t_scene > 0` can reach the bar. The 13 no_pool golds still need a posting if the target is the whole slice rather than the bar.

### instruction_following (BEAM, n=40)

1. Gold in corpus: 40.
2. Gold in pool@50: cannot be computed per slice. Bench BEAM pool@50 is 328 against a wide pool of 329, a gap of 1, not assigned to a category.
3. Gold at pre_rank 0: 6, identical to final rank 0.
4. Gold in top 5: 18 (JSON t5).
5. Gold at final rank 0: 6 (JSON t1).

Stages: no_pool 5, pool_not_k 15, k_not5 2, t5_not1 12, summing to 34 = 40-6. Wide pool 35. Stage k = 20. JSON t10 = 21. all5 = 13, all10 = 16. The bar is Turn@1, 36.

Slot-1 ceiling: 18, under 36. Wide-pool ceiling: 35, under 36. At least 1 of 5 no_pool golds needs a new posting.

### summarization (BEAM, n=36)

1. Gold in corpus: 36.
2. Gold in pool@50: cannot be computed per slice.
3. Gold at pre_rank 0: 6, identical to final rank 0.
4. Gold in top 5: 17.
5. Gold at final rank 0: 6.

Stages: no_pool 2, pool_not_k 9, k_not5 8, t5_not1 11. Wide pool 34. Official k 25. all5 = 0 and all10 = 0 (multi-gold). The shard's bar is Turn@1, 33.

Slot-1 ceiling: 17, under 33. Wide-pool ceiling: 34, which clears 33.

### event_ordering (BEAM, n=40)

1. Gold in corpus: 40.
2. Gold in pool@50: cannot be computed per slice.
3. Gold at pre_rank 0: 7, identical to final rank 0.
4. Gold in top 5: 14.
5. Gold at final rank 0: 7.

Stages: no_pool 1, pool_not_k 17, k_not5 8, t5_not1 7. Wide pool 39. Official k 22.

Slot-1 ceiling: 14, under 36. Wide-pool ceiling: 39, which clears 36.

### preference_following (BEAM, n=39)

1. Gold in corpus: 39.
2. Gold in pool@50: cannot be computed per slice.
3. Gold at pre_rank 0: 7, identical to final rank 0.
4. Gold in top 5: 14.
5. Gold at final rank 0: 7.

Stages: no_pool 9, pool_not_k 14, k_not5 2, t5_not1 7. Wide pool 30. Official k 16.

Slot-1 ceiling: 14, under 36. Wide-pool ceiling: 30, under 36. At least 6 of 9 no_pool golds need a new posting.

## Where a rule has to hook

Recall, a new posting so a `t_scene == 0` turn can become positive: the feat pushes in the ingest loop, `retrieval.rs` 3125-3274. `Feat` is defined at 1794-1800 (`id`, `w`, `counts_len`, `inheritable`). A witness must use `counts_len: false`.

Ordering among turns already scored: `remaining` is built at line 5633 from `t_scene > 0`, and the submodular pick runs from 5638. Hook range 5628-5638.

The post-triangulation call at 6265-6272 only reorders inside the top 5. Its ceiling is the top-5 line above, which misses the bar on every target slice. None of the ninety rules is hooked there.

## Tag counts

Filled by the generator at write time.

ORDERING 39. RECALL/INDEX 51. Combined mechanisms 47 of 90.

## Target: 90% of every metric

Sam's target of 2026-11-15 replaces the six-slice bar. Floor: Step 34 `62041ef5` for the three bench Turn@1 cells (LME 353, LoCoMo 805, BEAM 160), and bd2693e2 for every other cell. Step 34's Turn@1 gains are +3 / +9 / +4 and are not split by category, so category Turn@1 below is the bd2693e2 count. Turn@5 and Turn@10 in the target match the JSON, so those did not move. `need = ceil(0.9 * n)`. A hook ceiling is the most that class of rule can reach. It is not a promise that the rule attains it.

LoCoMo has no All@5, All@10, or Session@10 field in the attached aggregate. LME has no Turn All@10 field. Those metrics cannot be given a number.

### Bench hook ceilings

| Bench metric | Now | Need | Gap | Hook ceiling |
|---|---:|---:|---:|---|
| LME Session@1 | 430 | 423 | 0 | met |
| LME Session@5 | 463 | 423 | 0 | met |
| LME Session@10 | 468 | 423 | 0 | met |
| LME Session All@5 | 424 | 423 | 0 | met |
| LME Session All@10 | 447 | 423 | 0 | met |
| LME Turn@1 | 353 | 423 | 70 | turn top 5 (t5, unchanged by Step 34) ceiling 428 clears 423 |
| LME Turn@5 | 428 | 423 | 0 | met |
| LME Turn@10 | 437 | 423 | 0 | met |
| LME Turn All@5 | 314 | 423 | 109 | Turn All@10 is not in the JSON, so no all-gold bound above 314 |
| LoCoMo Turn@1 | 805 | 1383 | 578 | pool@50 (wide pool 1492 is higher) ceiling 1415 clears 1383 |
| LoCoMo Turn@5 | 1127 | 1383 | 256 | pool@50 ceiling 1415 clears 1383 |
| LoCoMo Turn@10 | 1226 | 1383 | 157 | pool@50 ceiling 1415 clears 1383 |
| LoCoMo Session@1 | 1133 | 1383 | 250 | session top 5 ceiling 1416 clears 1383 |
| LoCoMo Session@5 | 1416 | 1383 | 0 | met |
| BEAM Turn@1 | 160 | 320 | 160 | wide scored pool (header pool=329) ceiling 329 clears 320 |
| BEAM Turn@5 | 238 | 320 | 82 | wide scored pool ceiling 329 clears 320 |
| BEAM Turn@10 | 267 | 320 | 53 | wide scored pool ceiling 329 clears 320 |
| BEAM Turn All@5 | 118 | 320 | 202 | All@10, and only for questions with at most 5 golds ceiling 138 short of 320 by 182 |
| BEAM Turn All@10 | 138 | 320 | 182 | full gold set inside the wide pool is not measured |
| BEAM Session@1 | 133 | 320 | 187 | session top 10 ceiling 275 short of 320 by 45 |
| BEAM Session@5 | 249 | 320 | 71 | session top 10 ceiling 275 short of 320 by 45 |
| BEAM Session@10 | 275 | 320 | 45 | gold sessions outside the top 10 are not counted |
| BEAM Session All@5 | 137 | 320 | 183 | Session All@10, if the gold-session count is at most 5 ceiling 178 short of 320 by 142 |
| BEAM Session All@10 | 178 | 320 | 142 | gold sessions outside the top 10 are not counted |

Reading the gaps that miss a measured ceiling: LoCoMo Turn@1/+5/+10 clear from pool@50 (1415 >= 1383) without new postings, at bench grain. LME Turn@1 clears from the current top 5 (428 >= 423). LoCoMo Session@1 clears from the current session top 5 (1416 >= 1383). BEAM Turn@1/@5/@10 clear from the wide pool (329 >= 320) at bench grain. BEAM Session@1 does not: session top 10 is 275, short of 320 by 45, and the attachment does not count gold sessions outside that 10. BEAM All@5 does not: the best measured bound is All@10 = 138, short of 320 by 182, and only when each of those questions has at most 5 golds. LME Turn All@5 has no All@10 to bound it. Event ordering and summarization All@5 and All@10 are 0, so no question in those categories currently holds every gold inside the top 10.

Category bars still force recall where the bench surplus is not the category bar. Open-Domain wide pool 78 < 83. BEAM instruction following 35 < 36, preference following 30 < 36, information extraction 32 < 36. LoCoMo Multi-Hop session top 5 is 246 < 254 and Open-Domain session top 5 is 65 < 83.

### Category ceilings

Wide pool = n minus miss-map `no_pool` where that stage is present. `pool@50` is not per category. Session top 10 exists for LME and BEAM only.

| Category | n | Metric | Now | Need | Gap | Ceiling |
|---|---:|---|---:|---:|---:|---|
| LME single-session-user | 64 | Session@1 | 62 | 58 | 0 | met |
| LME single-session-user | 64 | Turn@1 | 54 | 58 | 4 | t5 62 clears 58 |
| LME single-session-user | 64 | Turn All@5 | 62 | 58 | 0 | met |
| LME multi-session | 121 | Session@1 | 110 | 109 | 0 | met |
| LME multi-session | 121 | Turn@1 | 88 | 109 | 21 | t5 113 clears 109 |
| LME multi-session | 121 | Turn All@5 | 53 | 109 | 56 | Turn All@10 absent; ta5 53 vs need 109 |
| LME single-session-preference | 30 | Session@1 | 17 | 27 | 10 | s5 26 short by 1; s10 30 clears 27 |
| LME single-session-preference | 30 | Session@5 | 26 | 27 | 1 | s10 30 clears 27 |
| LME single-session-preference | 30 | Turn@1 | 12 | 27 | 15 | wide pool (no_pool 0, t10 19) 30 clears 27 |
| LME single-session-preference | 30 | Turn@5 | 19 | 27 | 8 | wide pool 30 clears 27 |
| LME single-session-preference | 30 | Turn@10 | 19 | 27 | 8 | wide pool 30 clears 27 |
| LME single-session-preference | 30 | Turn All@5 | 14 | 27 | 13 | Turn All@10 absent |
| LME temporal-reasoning | 127 | Session@1 | 114 | 115 | 1 | s5 124 clears 115 |
| LME temporal-reasoning | 127 | Turn@1 | 93 | 115 | 22 | t10 116 clears 115 |
| LME temporal-reasoning | 127 | Turn@5 | 111 | 115 | 4 | t10 116 clears 115 |
| LME temporal-reasoning | 127 | Turn All@5 | 74 | 115 | 41 | Turn All@10 absent; wide pool 125 is any-gold, not all-gold |
| LME knowledge-update | 72 | Session@1 | 71 | 65 | 0 | met |
| LME knowledge-update | 72 | Turn@1 | 62 | 65 | 3 | t5 71 clears 65 |
| LME knowledge-update | 72 | Turn All@5 | 59 | 65 | 6 | Turn All@10 absent; t10 72 is any-gold |
| LME single-session-assistant | 56 | Session@1 | 56 | 51 | 0 | met |
| LME single-session-assistant | 56 | Turn@1 | 41 | 51 | 10 | t5 52 clears 51 |
| LME single-session-assistant | 56 | Turn All@5 | 52 | 51 | 0 | met |
| LoCoMo Single-Hop | 841 | Turn@1 | 494 | 757 | 263 | wide pool (no_pool 9); t10 737 is short 832 clears 757 |
| LoCoMo Single-Hop | 841 | Turn@5 | 686 | 757 | 71 | wide pool; t10 737 is short 832 clears 757 |
| LoCoMo Single-Hop | 841 | Turn@10 | 737 | 757 | 20 | wide pool 832 clears 757 |
| LoCoMo Single-Hop | 841 | Session@1 | 699 | 757 | 58 | s5 806 clears 757 |
| LoCoMo Multi-Hop | 282 | Turn@1 | 97 | 254 | 157 | wide pool 269 clears 254 |
| LoCoMo Multi-Hop | 282 | Turn@5 | 158 | 254 | 96 | wide pool; t10 186 is short 269 clears 254 |
| LoCoMo Multi-Hop | 282 | Turn@10 | 186 | 254 | 68 | wide pool 269 clears 254 |
| LoCoMo Multi-Hop | 282 | Session@1 | 156 | 254 | 98 | s5 246 short by 8 |
| LoCoMo Temporal | 321 | Turn@1 | 187 | 289 | 102 | wide pool; t10 260 is short 313 clears 289 |
| LoCoMo Temporal | 321 | Turn@5 | 248 | 289 | 41 | wide pool 313 clears 289 |
| LoCoMo Temporal | 321 | Turn@10 | 260 | 289 | 29 | wide pool 313 clears 289 |
| LoCoMo Temporal | 321 | Session@1 | 242 | 289 | 47 | s5 299 clears 289 |
| LoCoMo Open-Domain | 92 | Turn@1 | 18 | 83 | 65 | wide pool 78 short by 5 |
| LoCoMo Open-Domain | 92 | Turn@5 | 35 | 83 | 48 | wide pool 78 short by 5 |
| LoCoMo Open-Domain | 92 | Turn@10 | 43 | 83 | 40 | wide pool 78 short by 5 |
| LoCoMo Open-Domain | 92 | Session@1 | 36 | 83 | 47 | s5 65 short by 18 |
| BEAM contradiction_resolution | 40 | Turn@1 | 32 | 36 | 4 | t5 40 clears 36 |
| BEAM contradiction_resolution | 40 | Turn All@5 | 22 | 36 | 14 | all10, if at most 5 golds 27 short by 9 |
| BEAM contradiction_resolution | 40 | Turn All@10 | 27 | 36 | 9 | full gold set in the wide pool is not measured |
| BEAM contradiction_resolution | 40 | Session@1 | 29 | 36 | 7 | s5 40 clears 36 |
| BEAM contradiction_resolution | 40 | Session All@5 | 20 | 36 | 16 | sall10, if at most 5 gold sessions 30 short by 6 |
| BEAM event_ordering | 40 | Turn@1 | 7 | 36 | 29 | wide pool (no_pool 1) 39 clears 36 |
| BEAM event_ordering | 40 | Turn All@5 | 0 | 36 | 36 | all10 0 short by 36 |
| BEAM event_ordering | 40 | Turn All@10 | 0 | 36 | 36 | full gold set in the wide pool is not measured |
| BEAM event_ordering | 40 | Session@1 | 7 | 36 | 29 | s10 27 short by 9 |
| BEAM event_ordering | 40 | Session All@5 | 0 | 36 | 36 | sall10 0 short by 36 |
| BEAM event_ordering | 40 | Session All@10 | 0 | 36 | 36 | gold sessions outside the top 10 are not counted |
| BEAM information_extraction | 40 | Turn@1 | 20 | 36 | 16 | wide pool (no_pool 8) 32 short by 4 |
| BEAM information_extraction | 40 | Turn All@5 | 20 | 36 | 16 | all10 21 short by 15 |
| BEAM information_extraction | 40 | Session@1 | 15 | 36 | 21 | s10 27 short by 9 |
| BEAM information_extraction | 40 | Session All@5 | 25 | 36 | 11 | sall10 27 short by 9 |
| BEAM instruction_following | 40 | Turn@1 | 6 | 36 | 30 | wide pool (no_pool 5); JSON t10 21 35 short by 1 |
| BEAM instruction_following | 40 | Turn All@5 | 13 | 36 | 23 | all10 16 short by 20 |
| BEAM instruction_following | 40 | Session@1 | 4 | 36 | 32 | s10 17 short by 19 |
| BEAM instruction_following | 40 | Session All@5 | 15 | 36 | 21 | sall10 17 short by 19 |
| BEAM knowledge_update | 40 | Turn@1 | 30 | 36 | 6 | t5 37 clears 36 |
| BEAM knowledge_update | 40 | Turn All@5 | 24 | 36 | 12 | all10 27 short by 9 |
| BEAM knowledge_update | 40 | Session@1 | 24 | 36 | 12 | s5 37 clears 36 |
| BEAM knowledge_update | 40 | Session All@5 | 29 | 36 | 7 | sall10 31 short by 5 |
| BEAM multi_session_reasoning | 40 | Turn@1 | 21 | 36 | 15 | wide pool 40 clears 36; t5 34 is short; t10 36 meets |
| BEAM multi_session_reasoning | 40 | Turn All@5 | 4 | 36 | 32 | all10 7 short by 29 |
| BEAM multi_session_reasoning | 40 | Session@1 | 20 | 36 | 16 | s10 36 clears 36 |
| BEAM multi_session_reasoning | 40 | Session All@5 | 8 | 36 | 28 | sall10 15 short by 21 |
| BEAM preference_following | 39 | Turn@1 | 7 | 36 | 29 | wide pool (no_pool 9) 30 short by 6 |
| BEAM preference_following | 39 | Turn All@5 | 11 | 36 | 25 | all10 13 short by 23 |
| BEAM preference_following | 39 | Session@1 | 4 | 36 | 32 | s10 24 short by 12 |
| BEAM preference_following | 39 | Session All@5 | 14 | 36 | 22 | sall10 24 short by 12 |
| BEAM summarization | 36 | Turn@1 | 6 | 33 | 27 | wide pool (no_pool 2) 34 clears 33 |
| BEAM summarization | 36 | Turn All@5 | 0 | 33 | 33 | all10 0 short by 33 |
| BEAM summarization | 36 | Turn All@10 | 0 | 33 | 33 | full gold set in the wide pool is not measured |
| BEAM summarization | 36 | Session@1 | 6 | 33 | 27 | s10 26 short by 7 |
| BEAM summarization | 36 | Session All@10 | 1 | 33 | 32 | gold sessions outside the top 10 are not counted |
| BEAM temporal_reasoning | 40 | Turn@1 | 27 | 36 | 9 | t5 37 clears 36 |
| BEAM temporal_reasoning | 40 | Turn All@5 | 24 | 36 | 12 | all10 27 short by 9 |
| BEAM temporal_reasoning | 40 | Session@1 | 24 | 36 | 12 | s5 38 clears 36 |
| BEAM temporal_reasoning | 40 | Session All@5 | 26 | 36 | 10 | sall10 33 short by 3 |

Contradiction resolution has no miss-map stage line. Its Turn@5 is 40 of 40, so the wide pool is the whole category and Turn@1 is a top-5 reorder.

Single-session-preference Turn@1: miss-map stages are pool_not_k 11 and t5_not1 7, so no_pool is 0 and the wide pool is all 30. The top 10 holds 19. The bar of 27 needs turns that are scored and outside k.

A question with more than 5 gold turns cannot reach Turn All@5 under any ranking, and one with more than 10 cannot reach Turn All@10. Gold cardinality is not in the attachment, so the All@5 bar of 320 on BEAM and 423 on LME may be partly structural. Event ordering and summarization, both at All@10 = 0, are where that risk is concentrated.

### Per-row hook ceiling

Each design keeps its mechanism. The ceiling is the ceiling of its hook. Turn ordering is `retrieval.rs` 5628-5638 (`t_scene > 0`). Turn recall is 3125-3274. The session twin of a list reorder is the session candidate set at 5489-5491 (`s_score > 0` or `bridge_boost > 0`). A new length-free turn feat can lift a session from score 0 into that set, which is the only route past BEAM session@10 = 275.

| id | tag | Turn@1 hook ceiling | All@5 hook ceiling | Session@1 hook ceiling | Big gaps this row can move |
|---|---|---|---|---|---|
| S9-001 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275, All@5 unscored gold |
| S9-002 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275, MH/OD Session@1, All@5 unscored gold |
| S9-003 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-004 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275, All@5 unscored gold |
| S9-005 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | coverage loop over scored turns. Measured bound BEAM All@10=138, LME Turn All@10 absent. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1, BEAM All@5/All@10, LME All@5 |
| S9-006 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | coverage loop over scored turns. Measured bound BEAM All@10=138, LME Turn All@10 absent. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1, BEAM All@5/All@10, LME All@5 |
| S9-007 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | session top 5: LoCoMo 1416, BEAM 249, LME 463 (met). Twin hook 5480-5491. | LoCoMo T@1, LME T@1, BEAM T@1, LoCoMo Session@1 |
| S9-008 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-009 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275, All@5 unscored gold |
| S9-010 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-011 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-012 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-013 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-014 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-015 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-016 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | session top 5: LoCoMo 1416, BEAM 249, LME 463 (met). Twin hook 5480-5491. | LoCoMo T@1, LME T@1, BEAM T@1, LoCoMo Session@1 |
| S9-017 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-018 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-019 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-020 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-021 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-022 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-023 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-024 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-025 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-026 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-027 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-028 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-029 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-030 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-031 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-032 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-033 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-034 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-035 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-036 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-037 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-038 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-039 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-040 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-041 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-042 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-043 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-044 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-045 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275, All@5 unscored gold |
| S9-046 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-047 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-048 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-049 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-050 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-051 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-052 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-053 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-054 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-055 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-056 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-057 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-058 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-059 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-060 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275, All@5 unscored gold |
| S9-061 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-062 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-063 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275, All@5 unscored gold |
| S9-064 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275, MH/OD Session@1 |
| S9-065 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-066 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-067 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-068 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-069 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-070 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-071 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-072 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-073 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-074 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-075 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-076 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | session top 5: LoCoMo 1416, BEAM 249, LME 463 (met). Twin hook 5480-5491. | LoCoMo T@1, LME T@1, BEAM T@1, LoCoMo Session@1 |
| S9-077 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275, MH/OD Session@1 |
| S9-078 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-079 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-080 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-081 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-082 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-083 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-084 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-085 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-086 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-087 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-088 | ORDERING | wide pool: LME 468, LoCoMo 1492 (pool@50 1415), BEAM 329. Also LME top5 428. | single-seat ordering. All@5 stays at the current count for this rule alone. | turn list only. Session@1 unchanged. | LoCoMo T@1, LME T@1, BEAM T@1 |
| S9-089 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |
| S9-090 | RECALL/INDEX | new postings. Required where wide pool is short: OD 78, IF 35, PF 30, IE 32. | can bring an unscored gold into the pool. Does not by itself place every gold inside 5 slots. | can lift s_score 0 into the set at 5490. Needed past BEAM s10=275 and for LoCoMo MH s5=246, OD s5=65. | OD/IF/PF/IE no_pool, BEAM Session@1 past 275 |

### Which rows carry the big gaps

- **LoCoMo Turn@1, +578 (805 to 1383).** Bench ceiling is pool@50 = 1415. Every ORDERING row, hooked at 5628-5638, sits under that ceiling. The category residuals that 1415 does not cover are Open-Domain, wide pool 78 against a bar of 83. Those five questions are the RECALL rows aimed at Open-Domain, led by S9-001, S9-063, S9-064, S9-002, S9-060.

- **LoCoMo Session@1, +250 (1133 to 1383).** Session top 5 is already 1416. S9-007, S9-016, and S9-076 are the list reorders whose math (inversion count, rank sum, fixed point of the two ranks) applies to the session candidate list at 5480-5491 without a new constant. Multi-Hop still needs 8 sessions outside that top 5 (246 vs 254) and Open-Domain needs 18 (65 vs 83). Those are S9-002, S9-064, and S9-077, which post a session key.

- **BEAM All@5, +202 (118 to 320), All@10, +182 (138 to 320).** S9-005 and S9-006 change the gain inside the loop that fills all k seats, so a scored gold outside the current top 5 can enter. The measured ceiling of that loop, if it may only reshuffle the current top 10, is All@10 = 138. Everything above 138 requires golds that are scored and outside the top 10, which the same loop can take when `t_scene > 0`, or golds with `t_scene = 0`, which only a RECALL row can add. Event ordering and summarization are at All@10 = 0. Gold counts per question are not in the attachment; a question with more than 5 golds cannot reach All@5.

- **BEAM Session@1, +187 (133 to 320), and Session@5/+71, Session@10/+45.** Reordering the current session top 10 stops at 275. The remaining 45 questions need a session that is outside that 10, which means `s_score = 0` and `bridge_boost = 0` or a session the selector never reached. Every RECALL row can raise a session score by a new length-free feat and thereby enter the set at line 5490. S9-002, S9-064, and S9-077 are the ones that post the session itself.

- **LME Turn@1, +70 (353 to 423).** Top 5 already holds 428. Any ORDERING row can promote a gold that is sitting in slots 2 through 5. The category that top 5 cannot clear is single-session-preference (t5 = 19, bar 27, wide pool 30): that one needs the pool hook, not a new posting.

- **LME Turn All@5, +109 (314 to 423).** The hole is multi-session (53 vs 109) and temporal-reasoning (74 vs 115). S9-005 and S9-006 are the coverage loop. There is no Turn All@10 in the JSON, so the ceiling above 314 is unmeasured. Single-session-user and single-session-assistant All@5 are already at the bar.

