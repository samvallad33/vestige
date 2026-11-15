# Shard 6 funnel

Base SHA `e2ecd1da037974c1835f563ad8b046f4a4ca3629` (`perf(retrieval): Step 33 Triangulation`). Score file `perq-bd2693e2.json`: bin `/workspace/c2-algos-bins/t51b-attr-unseated-opener-t10`, sha256 `e93411ac566c4c83f92a83204e70ea85ae743445a3256dc5cb4eacc2523e6f84`. Miss map `missmap-bd2693e2.txt`. This shard's slices are BEAM `instruction_following` (6/40 Turn@1, 90% means 36) and BEAM `summarization` (6/36 Turn@1, 90% means 33).

## What writes the replay dump

`maybe_dump_replay` (`crates/vestige-core/src/retrieval.rs` 7262-7298) returns immediately when `VESTIGE_REPLAY_DUMP` is unset. The call sits at 6273-6286, after `apply_c2_step33_triangulation` (6265-6272) and after every earlier seat. `flush_replay_dumps` (7300) writes the snaps. `write_one_replay_dump` (7426-7621) emits one JSON line.

The dump is off the ranking path. Nothing in `write_one_replay_dump` is read back into `rank_one`.

| Field | Meaning |
| --- | --- |
| `final_ids`, `seated_ids` | The official turn list after the gatekeeper and every C2 seat, including Step 33. Same vector written twice (7612-7613). `rank` is the index in that list. `head_id` is `final_ids[0]`. |
| `pre_ids` | Built at 7429-7440. It starts as a clone of `seated`, then appends turns in `t_scene` order while `t_scene > 0`, stopping at 50. `rank_pre` is the index in that vector. |
| `t_scene` | Scene-pooled score copied into the snap at dump time (7290). Seats reorder `final_turn_indices`. They do not rewrite `t_scene`. Scene-rank 0 is the argmax of `t_scene` among turns with `t_scene > 0` (tie: earlier ingest index, via `order`). |
| `t_own`, `t_turn_raw`, `t_scene_sess`, `s_score`, `bridge_boost` | Copies of scores `rank_one` already computed. Dump-only. |
| `qtok_present` | 7490-7514. Reads `TurnIx.body` when the dump is on. Ranking must not. |
| `turn_ic_novel` | Dump-only information residual. Ranking must not read `body` to recompute it. |
| `role` | `TurnIx.speaker`. These designs do not read it. |
| Live k | Submodular selection at 5628-5638 keeps a turn only when `t_scene > 0`. The dump's 50-wide `pre_ids` pool is not that candidate set. |

`pre_ids[0]` is `seated[0]` by construction. In this dump, gold at pre_rank 0 and gold at final rank 0 are the same fact. There is no pre-seat snapshot in the file.

## What the per-question JSON stores

Top-level keys: `bin`, `bin_sha256`, `lme`, `locomo`, `beam`, `perq`. `perq.beam` has 355 keys. A BEAM value is 7 bits. Summing each bit across a category reproduces `beam.cats.<name>`:

| Bit | Aggregate field | instruction_following sum | summarization sum |
| --- | --- | --- | --- |
| 0 | Turn@1 `t1` | 6 | 6 |
| 1 | Turn@5 `t5` | 18 | 17 |
| 2 | Turn@10 `t10` | 21 | 25 |
| 3 | All@5 `all5` | 13 | 0 |
| 4 | Session@1 `s1` | 4 | 6 |
| 5 | Session@5 `s5` | 16 | 24 |
| 6 | Session All@5 `sall5` | 15 | 0 |

Bit 6 is session All@5. It is not session@10 and it is not turn All@10. Those live only on the category object (`s10`, `all10`, `sall10`).

Category objects, copied from the file:

- instruction_following, n=40: t1=6, t5=18, t10=21, all5=13, all10=16, s1=4, s5=16, s10=17, sall5=15, sall10=17.
- summarization, n=36: t1=6, t5=17, t10=25, all5=0, all10=0, s1=6, s5=24, s10=26, sall5=0, sall10=1.

There is no corpus bit. A question absent from `perq.beam` cannot be separated from a question whose seven bits are zero. Both slices are fully present (40 and 36 keys).

## Miss-map stages

BEAM header: n=355, pool=329, pool@50=328, k=266, T@5=238, T@1=156, All@5=118. Per-question JSON Turn@10 for all of BEAM is 267. The one-question gap is entirely inside instruction_following. Every other BEAM category printed in the miss map has `t10` equal to `n - (pool_not_k + no_pool)`.

Stage names, used as this shard's tags:

- `t5_not1`: gold is in the top 5 and not at slot 1. A reorder inside the top 5 can fix it. Tag ORDERING.
- `k_not5`: gold is in the official top 10 and outside the top 5. Membership of the top 5 has to change. Tag RECALL/INDEX.
- `pool_not_k`: gold is in the pool@50 and outside the official k. Tag RECALL/INDEX.
- `no_pool`: gold is outside pool@50 (`t_scene <= 0` and not seated). Tag RECALL/INDEX.

## Instruction following, five lines

1. Gold in corpus: cannot be computed. The attachment has no corpus bit.
2. Gold in pool@50: 35 if the miss map's `no_pool=5` is right (40 − 5). See the one-question gap below. It may be 36.
3. Gold at pre_rank 0: cannot be computed as its own stage. Dump `pre_ids[0]` is the seated head, so this number equals final rank 0 whenever the dump exists. The attachment has no `pre_ids`.
4. Gold in top 5: 18 (per-question bit 1; category `t5`).
5. Gold at final rank 0: 6 (bit 0; category `t1`).

Miss-map misses@1, sum 34 = 40 − 6: `t5_not1` 12, `pool_not_k` 15, `no_pool` 5, `k_not5` 2.

Per-question bits: not in top 5 = 22, of which 3 are in top 10 (`t10 − t5`) and 19 are outside top 10. The miss map says `k_not5` 2 and `no_pool + pool_not_k` 20. One question is in top 10 according to the bits and outside k according to the miss map. Moving that question from `no_pool` into `k_not5` makes the two files agree (`no_pool` 4, `pool_not_k` 15, `k_not5` 3). Moving it from `pool_not_k` into `k_not5` leaves `no_pool` at 5. The attachment does not say which.

Ceilings:

- Slot-1 reorder of the current top 5: 18/40 (45%).
- Reorder inside the current top 10: 21/40 (52.5%).
- Every pooled gold seated at slot 1: 35/40 on the printed `no_pool`, or 36/40 if the disputed question is actually pooled.

36 is the 90% line. On the printed miss map the pool ceiling is 35, one short, so at least one gold now marked `no_pool` has to enter the ranked list, and then none of the 35 pooled golds can be missed. On the reconciliation that sets `no_pool` to 4, the pool ceiling is exactly 36 and there is still no slack: every pooled gold, including the 15 outside k and the 3 in k outside the top 5, has to reach slot 1. Either reading, ordering the current top 5 or top 10 cannot reach 90%. Recall/index that changes who is in the list below rank 10 is mandatory. The designs tagged `no_pool` cover the harsher reading.

## Summarization, five lines

The miss map and the bits agree. `t10 − t5` = 8 = `k_not5`. `n − t10` = 11 = `pool_not_k + no_pool`.

1. Gold in corpus: cannot be computed. No corpus bit.
2. Gold in pool@50: 34 (36 − `no_pool` 2).
3. Gold at pre_rank 0: same fact as final rank 0 in the dump, and not present as a separate column in the attachment.
4. Gold in top 5: 17.
5. Gold at final rank 0: 6.

Misses@1, sum 30: `t5_not1` 11, `k_not5` 8, `pool_not_k` 9, `no_pool` 2.

Ceilings:

- Slot-1 reorder of the current top 5: 17/36 (47%).
- Reorder inside the current top 10: 25/36 (69%).
- Every pooled gold seated at slot 1: 34/36 (94%). That meets 33. At most one of the 34 pooled golds can be missed. Both `no_pool` golds can stay missed.

Turn@1 at 90% is reachable without new corpus recall. It is not reachable by a top-5 or top-10 reorder. Nine golds outside k and eight golds in k but outside the top 5 have to move.

All@5 is 0/36 and All@10 is 0/36. A slot-1 rule raises Turn@1 and leaves All@5 at 0 when the other golds of a multi-gold question stay outside the top 5. Session@1 equals Turn@1 (both 6). Session@5 is 24, above Turn@5 at 17: the right session is often already in the session top 5 while the wrong turn is seated.

## Where a new rule hooks

- ORDERING: new function in `algos_c2.rs`, called from `retrieval.rs` between line 6272 and line 6273. Top-5 membership stays. Rotate a turn that is already in `list[..5]`.
- RECALL/INDEX that inserts or swaps: same call site. Inserting evicts the tail. Swapping from index `>= 5` into index 4 changes the top 5. A `no_pool` rule may insert a turn whose `t_scene` is 0, because the submodular filter at 5633 would otherwise drop it.
- Ingest flags (digit positions, clause offsets, quote spans, colon heads, pecia numbers, seal marks, all-capitals counts): `TurnIx` push at 3334-3377, computed from the prose the indexer already scans for `form_digit` (3371). The flag does not change BM25 weights. Case folding at line 658 has already happened by the time a token id exists, so shape is stored from the raw token.

`apply_c2_step33_triangulation` (algos_c2.rs 2543-2593) returns at 2554-2556 unless `c2_named_session` is true. BEAM user/assistant questions do not enter it. These designs do not retune that function, `C2_BULLAE_WINDOW`, or `c2_summary_request` (934-939).

## Risk that belongs to the summary gate

Summary designs fire only when `c2_summary_request` is true, and instruction designs fire only when it is false. That reuses the existing cue lists and adds none. This shard did not measure how many of the 36 summarization questions match that detector. A summarization question the detector misses gets no summary seat from this set. Widening the cue list would be a new word list, which these designs do not do.
