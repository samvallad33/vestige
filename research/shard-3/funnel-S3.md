# Shard 3 funnel at bd2693e2, read against HEAD e2ecd1da

Base for every hook below is `e2ecd1da037974c1835f563ad8b046f4a4ca3629` on `perf/c2-step33-triangulation` (Step 33 triangulation). The attached per-question file is an earlier binary: `/workspace/c2-algos-bins/t51b-attr-unseated-opener-t10`, sha256 `e93411ac566c4c83f92a83204e70ea85ae743445a3256dc5cb4eacc2523e6f84`. That is ALGOS_LOG win #44. Step 33 is not in those bits. `C2_STEP33_LOG.md` records the only movement after this JSON: LoCoMo turn@1 796 to 799, Open-Domain 18 to 19, Temporal 187 to 188, Single-Hop 494 to 495. Multi-Hop stayed 97. LME and BEAM bits did not move. Funnels below are computed from the attachment. The Step 33 deltas are cited only where that log states them.

## What the dump fields are

`write_one_replay_dump` (`crates/vestige-core/src/retrieval.rs`, starting at line 7426) runs only when `VESTIGE_REPLAY_DUMP` is set. It is off the ranking path. `maybe_dump_replay` is called at line 6273, after Step 33.

The dump pool is built at lines 7429-7440. It starts as `snap.seated` (the official post-gatekeeper order) and then appends other turns with `t_scene > 0`, in scene order, until 50. `pre_rank` is the index in that list. `final_rank` is the index in `snap.seated`. Because the pool is seeded with `seated`, `pool[0]` is `seated[0]` whenever the official list is non-empty. Gold at pre-rank 0 and gold at final rank 0 are the same fact. They are not two stages.

`t_scene` is the lexical scene score and is not rewritten by seats. `t_own` is the turn's own lexical score. `t_turn_raw` is the pre-bridge turn score. `s_score` is the session score and `bridge_boost` is the session bridge. `qtok_present` (line 7490) and `turn_ic_novel` read `TurnIx.body`. That read is dump-only. None of the designs below read `body` at rank time.

The per-question JSON writer is not in this repository. The file is self-describing because the bit sums equal the summary object:

- BEAM length 7, in order: turn@1, turn@5, turn@10, all-at-5, session@1, session@5, session-all-at-5. Sums `[156, 238, 267, 118, 133, 249, 137]`. The summary also has turn@10-all, session@10, and session-all-at-10. Those are not in the per-question bits.
- LoCoMo length 5: session@1, session@5, turn@1, turn@5, turn@10. Sums `[1133, 1416, 796, 1127, 1226]`.
- LME length 7: session@1, session@5, session@10, turn@1, turn@5, turn@10, turn-all-at-5. Sums `[430, 463, 468, 350, 428, 437, 314]`.

`preference_following` has n=39 because key `9:preference_following:1` is absent.

## What the attachment cannot compute

Corpus membership is not in the attachment. `n` is the question count. It is not proof that the gold id was indexed. `C2_STEP33_LOG.md` says the later binary has LoCoMo 1535/1536 indexed and BEAM/LME corpus equal to `n`. Those figures are not computed from this JSON.

Per-slice pool versus pool@50 cannot be split. Aggregates: LME pool 468 of 470, pool@50 463. LoCoMo pool 1492 of 1536, pool@50 1415. BEAM pool 329 of 355, pool@50 328. BEAM `no_pool` across slices sums to 26, which is 355-329, so the one BEAM question that is in the pool but not in the top 50 is not assigned to a slice.

Gold at pre-rank 0 cannot be computed as its own stage. The dump defines it as final rank 0, and the JSON has no pre-rank field.

## Miss-map offset on instruction following

The miss map header is `k=266` for BEAM. The JSON turn@10 sum is 267. Win #44 seated `14:instruction_following:0`, whose bits are `[0,0,1,0,0,0,0]`: in the top 10, not in the top 5. The miss map still has that question in `pool_not_k`. After that one seat the working instruction-following partition is: `no_pool` 5, `pool_not_k` 14, `k_not5` 3, `t5_not1` 12, turn@1 6. Check: 5+14+3+12+6 = 40, and turn@5 = 18, turn@10 = 21, matching the JSON. Other slices match the miss map without that adjustment. Preference following: 9+14+2+7+7 = 39.

## Five-line funnels

Counts are gold-in-stage / questions. "Pool" is `n - no_pool` from the miss map (with the one instruction-following adjustment above). "Pool@50" is not available per slice. "Pre-rank 0" is not an independent stage.

### BEAM instruction_following (primary)

1. Gold in corpus: not computable from the attachment. Question count 40.
2. Gold in the pool: 35/40. Gold in pool@50: not computable per slice.
3. Gold at pre-rank 0: same fact as final rank 0, 6/40. Not an independent stage.
4. Gold in the top 5: 18/40. Gold in the official top 10: 21/40.
5. Gold at final rank 0: 6/40.

Slot-1 reorder of the current top 5 ceilings at 18/40 (45%). Reordering the current top 10 ceilings at 21/40 (52.5%). Seating every gold that is already in the pool ceilings at 35/40 (87.5%), one short of 36/40. A slot-1 reorder cannot reach 90%. At least one of the five `no_pool` golds has to be indexed, and essentially every pooled gold has to reach rank 0.

### BEAM preference_following (primary)

1. Gold in corpus: not computable. Question count 39 (one key absent).
2. Gold in the pool: 30/39. Gold in pool@50: not computable per slice.
3. Gold at pre-rank 0: same fact as final rank 0, 7/39.
4. Gold in the top 5: 14/39. Gold in the official top 10: 16/39.
5. Gold at final rank 0: 7/39.

Slot-1 reorder of the current top 5 ceilings at 14/39 (35.9%). Top 10 ceilings at 16/39 (41.0%). The whole current pool ceilings at 30/39 (76.9%), six short of 36/39. A slot-1 reorder cannot reach 90%. At least six of the nine `no_pool` golds have to be indexed.

### LoCoMo Open-Domain

Attachment turn@1 is 18/92. Step 33's log says a later reorder made it 19/92. The pool did not gain a document from that reorder.

1. Gold in corpus: not computable from the attachment.
2. Gold in the pool: 78/92 (`no_pool` 14). Gold in pool@50: not computable per slice.
3. Gold at pre-rank 0: same fact as final rank 0. Attachment 18/92. Step 33 log: 19/92.
4. Gold in the top 5: 35/92 (18+17). Gold in the top 10: 43/92 (35+8).
5. Gold at final rank 0: 18/92 in the attachment.

90% is 83/92. The current pool holds 78, so at least five of the fourteen `no_pool` golds have to be recovered. Slot-1 reorder of the current top 5 ceilings at 35/92 (38.0%).

### LoCoMo Multi-Hop

1. Gold in corpus: not computable from the attachment.
2. Gold in the pool: 269/282 (`no_pool` 13). Gold in pool@50: not computable per slice.
3. Gold at pre-rank 0: same fact as final rank 0, 97/282. Step 33 did not move this category.
4. Gold in the top 5: 158/282. Gold in the top 10: 186/282 (158+28).
5. Gold at final rank 0: 97/282.

90% is 254/282. The pool already holds 269, fifteen above that line, so 90% does not require a `no_pool` recovery. It does require pulling `pool_not_k` (83) and fixing `k_not5` (28) and `t5_not1` (61). Slot-1 reorder of the current top 5 ceilings at 158/282 (56.0%).

### BEAM summarization

1. Gold in corpus: not computable. Question count 36.
2. Gold in the pool: 34/36 (`no_pool` 2).
3. Gold at pre-rank 0: same fact as final rank 0, 6/36.
4. Gold in the top 5: 17/36. Gold in the top 10: 25/36 (17+8).
5. Gold at final rank 0: 6/36.

90% is 33/36. The pool holds 34, one above that line. A reorder that can reach past the current top 5 into `pool_not_k` (9) and `k_not5` (8) can get there. A slot-1 reorder of the current top 5 ceilings at 17/36 (47.2%) and cannot.

### BEAM event_ordering

1. Gold in corpus: not computable. Question count 40.
2. Gold in the pool: 39/40 (`no_pool` 1).
3. Gold at pre-rank 0: same fact as final rank 0, 7/40.
4. Gold in the top 5: 14/40. Gold in the top 10: 22/40 (14+8).
5. Gold at final rank 0: 7/40.

90% is 36/40. The pool holds 39. Slot-1 reorder of the current top 5 ceilings at 14/40 (35%) and cannot. A pool-wide reorder can.

## Where a new rule has to hook

Recall and index rules that add feats hook the token loop in `build_index`, `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Rules that pull a non-adjacent turn into the candidate seed hook the scene-positive seed at lines 5028-5032. Ordering rules that change slot 1 hook immediately after `apply_c2_step33_triangulation` returns and before `maybe_dump_replay`, lines 6272-6273. They must not call `rsb_seat_behind` (the call is at 6083; the function starts at 6702) and must not edit `HYPERNYM_LATTICE` (line 460), `rungs.rs`, or `benchmarks/real_engine/`.

The tag on each algorithm follows these ceilings. `RECALL/INDEX` means the gold is outside the pool or outside k and needs a new feat or a new seed. `ORDERING` means the gold is already in the top 5 or in k and the rule only reseats. Ordering is the second wave. On the two primary slices it cannot, by itself, reach 90%.
