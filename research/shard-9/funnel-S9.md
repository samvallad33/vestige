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
