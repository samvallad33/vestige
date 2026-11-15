# Shard 4 funnel (bd2693e2 per-question bits + miss map)

Base SHA for every hook below: `e2ecd1da037974c1835f563ad8b046f4a4ca3629` on `research/shard-4`, cut from `perf/c2-step33-triangulation`.

## What writes the two artifacts

### Replay dump

`VESTIGE_REPLAY_DUMP` is off the ranking path. `replay_dump_path` (`crates/vestige-core/src/retrieval.rs` 7230–7235) returns the path only when that variable is a non-empty string. `maybe_dump_replay` (7262–7298) returns immediately when the path is absent, then otherwise stores a `ReplaySnap` of the already-computed seated list and scores. `rank` calls `flush_replay_dumps` (7300–7319) after every query in the request has been ranked. `write_one_replay_dump` (7426–7621) is the writer.

Field meanings, from that function:

- `final_ids` and `seated_ids` are the official turn order after the gatekeeper and every C2 seat, including Step 33. `candidates[].rank` is the index in that list.
- `pre_ids` is that seated list first, then turns with `t_scene > 0` in `t_scene` order, capped at 50 (7429–7440). `rank_pre` is the index in `pre_ids`. Because the seated list is prefixed, `pre_ids[0]` is `final_ids[0]`. Gold at pre_rank 0 is the same count as gold at final rank 0. There is no earlier pre-seat snapshot in the dump.
- `t_scene` is the scene-pooled turn score copied onto the snap. Seats do not write it back. Scene-rank 0 would be the argmax of `t_scene`. The attached per-question file does not store it.
- `t_own` is the turn's uncarried BM25 (plus the contiguous-bigram bonus already folded in before the snap). `t_turn_raw` is the carried turn score. `t_scene_sess` is the session-pooled scene score. `s_score` and `bridge_boost` are the session-rank components. Their sum is the official session score. The dump copies them. Ranking does not read the dump.
- `qtok_present` and `turn_ic_novel` read `TurnIx.body` and are dump-only (7490–7515, 7362–7379). Ranking is forbidden from doing that.
- `answers` on `TurnIx` (retrieval.rs 1871–1873) is the previous turn in the session with a different role, backward only. The dump exposes it as `parent_user_turn_id`. It is not a semantic entailment edge.
- `k` is the requested official list length. `pool_size` is `pre_ids.len()`.

### Per-question JSON

`crates/vestige-core/src/bin/retrieval_bench.rs` writes one `RankResponse` JSON line per request. It does not write the compact bit file. No file in this checkout emits `perq`. The attached file's arrays were identified by summing each column and matching the sibling aggregate objects in the same JSON:

- LoCoMo, 1536 rows, width 5, sums `[1133, 1416, 796, 1127, 1226]` = `[session@1, session@5, turn@1, turn@5, turn@10]`. Keys are `conv:question`. Category is not in the key.
- BEAM, 355 rows, width 7, sums `[156, 238, 267, 118, 133, 249, 137]` = `[turn@1, turn@5, turn@10, turn all@5, session@1, session@5, session all@5]`. Keys are `chat:category:index`. Per-category sums match `beam.cats` exactly, including summarization `[6, 17, 25, 0, 6, 24, 0]`.
- LME, 470 rows, width 7, sums `[430, 463, 468, 350, 428, 437, 314]` = `[session any@1, session any@5, session any@10, turn any@1, turn any@5, turn any@10, turn all@5]`.

Bits are 0/1. A 1 means that any-hit (or all-hit, for the all@5 columns) succeeded. The file has no pool bit, no pre_rank bit, and no scene-rank bit.

### Miss-map stages

`missmap-bd2693e2.txt` partitions Turn@1 misses. For a slice, the stage counts sum to `n - turn@1`. Where the arithmetic is a pure prefix of the any-hit bits:

- `t5_not1 = turn@5 - turn@1` (gold already in the official top 5, not in slot 1)
- `k_not5 = turn@10 - turn@5` (gold in the official top 10, not in the top 5)
- `pool_not_k` = gold in the wider pool, not in the official top 10
- `no_pool` = gold in neither

The header separates `pool` from `pool@50` (LoCoMo 1492 vs 1415, BEAM 329 vs 328, LME 468 vs 463). The stage name `no_pool` lines up with the wider pool, not with pool@50. The attached bits cannot say which `pool_not_k` rows sit inside pool@50.

`C2_STEP33_LOG.md` in this checkout measured pool@50 on the Step 31 binary whose LoCoMo Turn@1 is the attached 796 (Multi-Hop 97, Open-Domain 18, Temporal 187, Single-Hop 494). Those pool@50 figures are cited below as that log, not as a column of the attached JSON. For Multi-Hop the log's pool@50=243 reconciles with the miss map: top10 186, so 57 are in pool@50 but not top 10; wide pool 269, so 26 are in the wide pool but not pool@50; 57+26=83=`pool_not_k`.

## Five-line funnels

Gold-at-pre_rank-0 is gold-at-final-rank-0, by the dump prefix. The attached file cannot compute pool@50 or scene-rank. Corpus below is the question count in the attached aggregate. The Step 33 log says one LoCoMo evidence id is absent from the indexed turns (corpus 1535 vs n 1536) and that the missing id is not in Multi-Hop (that slice's corpus line in the log is 282).

### LoCoMo Multi-Hop (primary)

```
gold in corpus          282   (question count; log corpus agrees)
gold in pool@50         243   (NOT in the attached bits; C2_STEP33_LOG on this T@1)
gold at pre_rank 0       97   (equals final rank 0)
gold in top5            158
gold at final rank 0     97
```

Attached stage split of the 185 misses: t5_not1 61, k_not5 28, pool_not_k 83, no_pool 13. 97+61=158, 158+28=186=turn@10, 186+83=269 wide pool, 269+13=282.

Slot-1 reorder ceiling = 158/282 (56%). Need 254/282 (90%). 158 and 186 (top-10 ceiling, 66%) and 243 (pool@50 ceiling, 86%) are all below 254. Wide-pool ceiling is 269 (95%), which clears 254. Reaching 254 needs at least 11 golds admitted into pool@50 (254-243) and those, plus golds already in the pool but not at rank 1, seated at rank 1. A slot-1 swap inside the current top 5 cannot get there. Tag the admission rules RECALL/INDEX and the top-5 swaps ORDERING.

### BEAM summarization (primary)

```
gold in corpus           36   (question count; per-slice pool@50 is not in the attachment or the Step 33 log)
gold in pool@50          not computable per slice
gold at pre_rank 0        6
gold in top5             17
gold at final rank 0      6
```

Turn@10=25, all@5=0, all@10=0, session@1=6, session@5=24, session all@5=0. Stages: t5_not1 11, k_not5 8, pool_not_k 9, no_pool 2. 6+11=17, 17+8=25, 25+9=34 wide pool, 34+2=36.

Need 33/36. Slot-1 ceiling 17 (47%). Top-10 ceiling 25 (69%). Wide-pool ceiling 34 (94%), so 33 is reachable if 33 of the 34 pooled golds can be placed at rank 1. Bench-level BEAM pool is 329 and pool@50 is 328, one question apart; the attachment cannot say whether that one question is a summarization row. All@5 is 0 on all 36 questions: the set of gold turns is never covered by the top 5, including the 17 questions that already have one gold in the top 5. Summarization needs a quota that pulls further pooled golds into the five seats, not only a new slot-1 winner.

### LoCoMo Open-Domain

```
gold in corpus           92
gold in pool@50          63   (C2_STEP33_LOG, same T@1=18; not in the attached bits)
gold at pre_rank 0       18
gold in top5             35
gold at final rank 0     18
```

Turn@10=43. Stages: t5_not1 17, k_not5 8, pool_not_k 35, no_pool 14. Wide pool = 43+35 = 78. Need 83. Slot-1 ceiling 35, top-10 ceiling 43, pool@50 ceiling 63, wide-pool ceiling 78. All four sit below 83. At least 5 of the 14 no_pool questions need a new index hit, and the pooled golds still have to be seated at rank 1. This slice is not a seat-reorder problem.

### BEAM instruction_following

```
gold in corpus           40
gold in pool@50          not computable per slice
gold at pre_rank 0        6
gold in top5             18
gold at final rank 0      6
```

Turn@10=21, all@5=13. Miss-map stages of the 34 misses: t5_not1 12, k_not5 2, pool_not_k 15, no_pool 5. Those four sum to 34. `turn@10 - turn@5 = 3`, which is not the stage's `k_not5 = 2`. One question's stage label is not a pure prefix of the any-hit bits; the attached files do not say which. Need 36. Slot-1 ceiling 18. Top-10 ceiling 21. Wide pool implied by the stages is 21+15=36 if k_not5 is ignored, or 40-5=35 if no_pool is the complement. Either reading sits at the line of 36 only if no_pool is fixed too. Under the all-metric floor this category is in scope: Turn@1, All@5 (13 against 36), and Session@1 (4 against 36) all sit under it.

### BEAM event_ordering

```
gold in corpus           40
gold in pool@50          not computable per slice
gold at pre_rank 0        7
gold in top5             14
gold at final rank 0      7
```

Turn@10=22, all@5=0, all@10=0. Stages: t5_not1 7, k_not5 8, pool_not_k 17, no_pool 1 (sums to 33=40-7; 14-7=7; 22-14=8; 22+17+1=40). Need 36. Slot-1 ceiling 14. Top-10 ceiling 22. Wide-pool ceiling 39. Ordering of the current top 5 cannot reach 36.

### BEAM preference_following

```
gold in corpus           39
gold in pool@50          not computable per slice
gold at pre_rank 0        7
gold in top5             14
gold at final rank 0      7
```

Turn@10=16, all@5=11. Stages: t5_not1 7, k_not5 2, pool_not_k 14, no_pool 9 (sums to 32=39-7; 16+14+9=39). Need 36. Slot-1 ceiling 14. Top-10 ceiling 16. Wide-pool ceiling 30, which is below 36. At least 6 of the 9 no_pool questions need an index hit. The wide pool itself is short of 90%, so this category is a FEAT problem. The all-metric table below is the full set.

## What a new rule is allowed to touch

ORDERING hooks the call immediately after `apply_c2_step33_triangulation` (`retrieval.rs` 6265–6272) and before `maybe_dump_replay` (6273). The pattern is `apply_c2_step33_triangulation` in `algos_c2.rs` 2543–2593: it reorders `list` and does not read `TurnIx.body`.

RECALL/INDEX either posts new feats inside the ingest loop (`retrieval.rs` 3121–3144, the content-token loop that pushes `Feat` at 3125 and the stem feat at 3143) or changes who enters the official list. The official list's candidate set is `remaining`, built at `retrieval.rs` 5633 by `(0..nt).filter(|&ti| t_scene[ti] > 0.0)`. The submodular gain is computed at 5710 and the chosen turn is pushed at 5726–5730. A rule that only swaps inside the current top 5 is ORDERING. A rule that posts a feat, admits a turn that filter drops, or replaces a top-10 seat with a pooled turn outside that list is RECALL/INDEX.

Neither hook reads `ix.turns[ti].body`, edits `rsb_seat_behind`, deletes `HYPERNYM_LATTICE`, or touches `rungs.rs` or `benchmarks/real_engine/`.

## All-metric 90% ceilings

Sam's floor is 90% on every metric of every benchmark and every category. Need is `(9*n + 9) // 10`. Turn@1 after Step 34 (`62041ef5`) is LME 353, LoCoMo 805, BEAM 160. Every other cell in `TARGET_90_ALL.md` matches this bd2693e2 file. This checkout has no per-question bits at `62041ef5`, so category Turn@1 ceilings below use the attached bits (LME 350, LoCoMo 796, BEAM 156). The Step 34 increments are +3 / +9 / +4 at bench level and do not change a ceiling that was already short or already clear by more than that.

Three ceilings, and they are not interchangeable:

- **Slot ceiling.** Gold already in the top 5 (Turn@5, or Session@5). A permute of the current five, which is what hook `6265–6272` can do when it only rewrites `list[..5]`, cannot pass this number. All@5 does not move at all under a permute: the set is unchanged, so the slot ceiling of All@5 is the current All@5.
- **Top-10 ceiling.** Gold already in the official top 10 (Turn@10, Session@10, All@10). Reaching it means pulling a gold from positions 6–10 into the five or into slot 1. LoCoMo has no Session@10 and no All@10 in the attached bits. LME has no turn All@10.
- **Pool ceiling.** For turns, pool@50 where `C2_STEP33_LOG.md` or the missmap header measured it on this Turn@1, otherwise the wide pool (`Turn@10 + pool_not_k`). `no_pool` is outside that. For sessions, Session@10 is the furthest figure the bits contain; there is no session-pool column. There is no bit for "all gold turns are in the pool", so All@5 has no pool ceiling.

`clears` means the ceiling is at least the 90% need. `short` means it is not. A metric with gap 0 is already at the floor.

| Bench | Category | Metric | Now | n | 90% | Gap | Slot ceiling | Top-10 ceiling | Pool ceiling | Slot | Top-10 | Pool | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| LME | all | Session@1 | 430 | 470 | 423 | 0 | 463 | 468 | 468 | clears | clears | clears | already met at Session@1 |
| LME | all | Session@5 | 463 | 470 | 423 | 0 | 463 | 468 | 468 | clears | clears | clears | met |
| LME | all | Session@10 | 468 | 470 | 423 | 0 | 468 | 468 | 468 | clears | clears | clears | met |
| LME | all | Session All@5 | 424 | 470 | 423 | 0 | 424 | 447 | 447 | clears | clears | clears | met; All@5 reorder ceiling is the current All@5 |
| LME | all | Session All@10 | 447 | 470 | 423 | 0 | 447 | 447 | — | clears | clears | unknown | met |
| LME | all | Turn@1 | 350 | 470 | 423 | 73 | 428 | 437 | 468 | clears | clears | clears | Step 34 Turn@1 is 353; top-5 ceiling 428 still clears 423 |
| LME | all | Turn@5 | 428 | 470 | 423 | 0 | 428 | 437 | 468 | clears | clears | clears | met |
| LME | all | Turn@10 | 437 | 470 | 423 | 0 | 437 | 437 | 468 | clears | clears | clears | met |
| LME | all | Turn All@5 | 314 | 470 | 423 | 109 | 314 | — | — | short | unknown | unknown | no turn All@10 in the attached bits; a top-5 permute does not raise All@5 |
| LME | single-session-user | Session@1 | 62 | 64 | 58 | 0 | 64 | 64 | 64 | clears | clears | clears |  |
| LME | single-session-user | Session@5 | 64 | 64 | 58 | 0 | 64 | 64 | 64 | clears | clears | clears |  |
| LME | single-session-user | Session@10 | 64 | 64 | 58 | 0 | 64 | 64 | 64 | clears | clears | clears |  |
| LME | single-session-user | Turn@1 | 54 | 64 | 58 | 4 | 62 | 63 | 64 | clears | clears | clears | missmap {'t5_not1': 8, 'pool_not_k': 1, 'k_not5': 1} |
| LME | single-session-user | Turn@5 | 62 | 64 | 58 | 0 | 62 | 63 | 64 | clears | clears | clears |  |
| LME | single-session-user | Turn@10 | 63 | 64 | 58 | 0 | 63 | 63 | 64 | clears | clears | clears |  |
| LME | single-session-user | Turn All@5 | 62 | 64 | 58 | 0 | 62 | — | — | clears | unknown | unknown | no per-type turn All@10 |
| LME | multi-session | Session@1 | 110 | 121 | 109 | 0 | 121 | 121 | 121 | clears | clears | clears |  |
| LME | multi-session | Session@5 | 121 | 121 | 109 | 0 | 121 | 121 | 121 | clears | clears | clears |  |
| LME | multi-session | Session@10 | 121 | 121 | 109 | 0 | 121 | 121 | 121 | clears | clears | clears |  |
| LME | multi-session | Turn@1 | 88 | 121 | 109 | 21 | 113 | 113 | 121 | clears | clears | clears | missmap {'t5_not1': 25, 'pool_not_k': 8} |
| LME | multi-session | Turn@5 | 113 | 121 | 109 | 0 | 113 | 113 | 121 | clears | clears | clears |  |
| LME | multi-session | Turn@10 | 113 | 121 | 109 | 0 | 113 | 113 | 121 | clears | clears | clears |  |
| LME | multi-session | Turn All@5 | 53 | 121 | 109 | 56 | 53 | — | — | short | unknown | unknown | no per-type turn All@10 |
| LME | single-session-preference | Session@1 | 17 | 30 | 27 | 10 | 26 | 30 | 30 | short | clears | clears |  |
| LME | single-session-preference | Session@5 | 26 | 30 | 27 | 1 | 26 | 30 | 30 | short | clears | clears |  |
| LME | single-session-preference | Session@10 | 30 | 30 | 27 | 0 | 30 | 30 | 30 | clears | clears | clears |  |
| LME | single-session-preference | Turn@1 | 12 | 30 | 27 | 15 | 19 | 19 | 30 | short | short | clears | missmap {'pool_not_k': 11, 't5_not1': 7} |
| LME | single-session-preference | Turn@5 | 19 | 30 | 27 | 8 | 19 | 19 | 30 | short | short | clears |  |
| LME | single-session-preference | Turn@10 | 19 | 30 | 27 | 8 | 19 | 19 | 30 | short | short | clears |  |
| LME | single-session-preference | Turn All@5 | 14 | 30 | 27 | 13 | 14 | — | — | short | unknown | unknown | no per-type turn All@10 |
| LME | temporal-reasoning | Session@1 | 114 | 127 | 115 | 1 | 124 | 125 | 125 | clears | clears | clears |  |
| LME | temporal-reasoning | Session@5 | 124 | 127 | 115 | 0 | 124 | 125 | 125 | clears | clears | clears |  |
| LME | temporal-reasoning | Session@10 | 125 | 127 | 115 | 0 | 125 | 125 | 125 | clears | clears | clears |  |
| LME | temporal-reasoning | Turn@1 | 93 | 127 | 115 | 22 | 111 | 116 | 125 | short | clears | clears | missmap {'pool_not_k': 9, 't5_not1': 18, 'k_not5': 5, 'no_pool': 2} |
| LME | temporal-reasoning | Turn@5 | 111 | 127 | 115 | 4 | 111 | 116 | 125 | short | clears | clears |  |
| LME | temporal-reasoning | Turn@10 | 116 | 127 | 115 | 0 | 116 | 116 | 125 | clears | clears | clears |  |
| LME | temporal-reasoning | Turn All@5 | 74 | 127 | 115 | 41 | 74 | — | — | short | unknown | unknown | no per-type turn All@10 |
| LME | knowledge-update | Session@1 | 71 | 72 | 65 | 0 | 72 | 72 | 72 | clears | clears | clears |  |
| LME | knowledge-update | Session@5 | 72 | 72 | 65 | 0 | 72 | 72 | 72 | clears | clears | clears |  |
| LME | knowledge-update | Session@10 | 72 | 72 | 65 | 0 | 72 | 72 | 72 | clears | clears | clears |  |
| LME | knowledge-update | Turn@1 | 62 | 72 | 65 | 3 | 71 | 72 | 72 | clears | clears | clears | missmap {'t5_not1': 9, 'k_not5': 1} |
| LME | knowledge-update | Turn@5 | 71 | 72 | 65 | 0 | 71 | 72 | 72 | clears | clears | clears |  |
| LME | knowledge-update | Turn@10 | 72 | 72 | 65 | 0 | 72 | 72 | 72 | clears | clears | clears |  |
| LME | knowledge-update | Turn All@5 | 59 | 72 | 65 | 6 | 59 | — | — | short | unknown | unknown | no per-type turn All@10 |
| LME | single-session-assistant | Session@1 | 56 | 56 | 51 | 0 | 56 | 56 | 56 | clears | clears | clears |  |
| LME | single-session-assistant | Session@5 | 56 | 56 | 51 | 0 | 56 | 56 | 56 | clears | clears | clears |  |
| LME | single-session-assistant | Session@10 | 56 | 56 | 51 | 0 | 56 | 56 | 56 | clears | clears | clears |  |
| LME | single-session-assistant | Turn@1 | 41 | 56 | 51 | 10 | 52 | 54 | 56 | clears | clears | clears | missmap {'t5_not1': 11, 'k_not5': 2, 'pool_not_k': 2} |
| LME | single-session-assistant | Turn@5 | 52 | 56 | 51 | 0 | 52 | 54 | 56 | clears | clears | clears |  |
| LME | single-session-assistant | Turn@10 | 54 | 56 | 51 | 0 | 54 | 54 | 56 | clears | clears | clears |  |
| LME | single-session-assistant | Turn All@5 | 52 | 56 | 51 | 0 | 52 | — | — | clears | unknown | unknown | no per-type turn All@10 |
| LoCoMo | all | Turn@1 | 796 | 1536 | 1383 | 587 | 1127 | 1226 | 1415 | short | short | clears | Step 34 Turn@1 is 805; pool@50 1415 is from C2_STEP33_LOG on the attached Turn@1, wide pool 1492 |
| LoCoMo | all | Turn@5 | 1127 | 1536 | 1383 | 256 | 1127 | 1226 | 1415 | short | short | clears |  |
| LoCoMo | all | Turn@10 | 1226 | 1536 | 1383 | 157 | 1226 | 1226 | 1415 | short | short | clears |  |
| LoCoMo | all | Session@1 | 1133 | 1536 | 1383 | 250 | 1416 | — | — | clears | unknown | unknown | no Session@10 in the attached LoCoMo bits |
| LoCoMo | all | Session@5 | 1416 | 1536 | 1383 | 0 | 1416 | — | — | clears | unknown | unknown | met |
| LoCoMo | all | Turn All@5 | 954 | 1536 | 1383 | 429 | 954 | — | — | short | unknown | unknown | missmap only; not in the per-question vector or the target table |
| LoCoMo | Multi-Hop | Turn@1 | 97 | 282 | 254 | 157 | 158 | 186 | 243 | short | short | short | wide pool 269; missmap {'k_not5': 28, 'no_pool': 13, 't5_not1': 61, 'pool_not_k': 83} |
| LoCoMo | Multi-Hop | Turn@5 | 158 | 282 | 254 | 96 | 158 | 186 | 243 | short | short | short | wide 269 |
| LoCoMo | Multi-Hop | Turn@10 | 186 | 282 | 254 | 68 | 186 | 186 | 243 | short | short | short | wide 269 |
| LoCoMo | Multi-Hop | Session@1 | 156 | 282 | 254 | 98 | 246 | — | — | short | unknown | unknown | no Session@10 |
| LoCoMo | Multi-Hop | Session@5 | 246 | 282 | 254 | 8 | 246 | — | — | short | unknown | unknown |  |
| LoCoMo | Temporal | Turn@1 | 187 | 321 | 289 | 102 | 248 | 260 | 299 | short | short | clears | wide pool 313; missmap {'pool_not_k': 53, 't5_not1': 61, 'k_not5': 12, 'no_pool': 8} |
| LoCoMo | Temporal | Turn@5 | 248 | 321 | 289 | 41 | 248 | 260 | 299 | short | short | clears | wide 313 |
| LoCoMo | Temporal | Turn@10 | 260 | 321 | 289 | 29 | 260 | 260 | 299 | short | short | clears | wide 313 |
| LoCoMo | Temporal | Session@1 | 242 | 321 | 289 | 47 | 299 | — | — | clears | unknown | unknown | no Session@10 |
| LoCoMo | Temporal | Session@5 | 299 | 321 | 289 | 0 | 299 | — | — | clears | unknown | unknown |  |
| LoCoMo | Open-Domain | Turn@1 | 18 | 92 | 83 | 65 | 35 | 43 | 63 | short | short | short | wide pool 78; missmap {'pool_not_k': 35, 't5_not1': 17, 'no_pool': 14, 'k_not5': 8} |
| LoCoMo | Open-Domain | Turn@5 | 35 | 92 | 83 | 48 | 35 | 43 | 63 | short | short | short | wide 78 |
| LoCoMo | Open-Domain | Turn@10 | 43 | 92 | 83 | 40 | 43 | 43 | 63 | short | short | short | wide 78 |
| LoCoMo | Open-Domain | Session@1 | 36 | 92 | 83 | 47 | 65 | — | — | short | unknown | unknown | no Session@10 |
| LoCoMo | Open-Domain | Session@5 | 65 | 92 | 83 | 18 | 65 | — | — | short | unknown | unknown |  |
| LoCoMo | Single-Hop | Turn@1 | 494 | 841 | 757 | 263 | 686 | 737 | 810 | short | short | clears | wide pool 832; missmap {'t5_not1': 192, 'pool_not_k': 95, 'k_not5': 51, 'no_pool': 9} |
| LoCoMo | Single-Hop | Turn@5 | 686 | 841 | 757 | 71 | 686 | 737 | 810 | short | short | clears | wide 832 |
| LoCoMo | Single-Hop | Turn@10 | 737 | 841 | 757 | 20 | 737 | 737 | 810 | short | short | clears | wide 832 |
| LoCoMo | Single-Hop | Session@1 | 699 | 841 | 757 | 58 | 806 | — | — | clears | unknown | unknown | no Session@10 |
| LoCoMo | Single-Hop | Session@5 | 806 | 841 | 757 | 0 | 806 | — | — | clears | unknown | unknown |  |
| BEAM | all | Turn@1 | 156 | 355 | 320 | 164 | 238 | 267 | 328 | short | short | clears | Step 34 Turn@1 is 160; wide pool 329 |
| BEAM | all | Turn@5 | 238 | 355 | 320 | 82 | 238 | 267 | 328 | short | short | clears |  |
| BEAM | all | Turn@10 | 267 | 355 | 320 | 53 | 267 | 267 | 328 | short | short | clears |  |
| BEAM | all | Turn All@5 | 118 | 355 | 320 | 202 | 118 | 138 | — | short | short | unknown | All@10 is the ceiling of a set-cover inside the current top 10; no all-golds-in-pool bit |
| BEAM | all | Turn All@10 | 138 | 355 | 320 | 182 | 138 | 138 | — | short | short | unknown |  |
| BEAM | all | Session@1 | 133 | 355 | 320 | 187 | 249 | 275 | — | short | short | unknown |  |
| BEAM | all | Session@5 | 249 | 355 | 320 | 71 | 249 | 275 | — | short | short | unknown |  |
| BEAM | all | Session@10 | 275 | 355 | 320 | 45 | 275 | 275 | — | short | short | unknown |  |
| BEAM | all | Session All@5 | 137 | 355 | 320 | 183 | 137 | 178 | — | short | short | unknown |  |
| BEAM | all | Session All@10 | 178 | 355 | 320 | 142 | 178 | 178 | — | short | short | unknown |  |
| BEAM | contradiction_resolution | Turn@1 | 32 | 40 | 36 | 4 | 40 | 40 | — | clears | clears | unknown | no missmap stages (contradiction_resolution is not in the stage list) |
| BEAM | contradiction_resolution | Turn@5 | 40 | 40 | 36 | 0 | 40 | 40 | — | clears | clears | unknown |  |
| BEAM | contradiction_resolution | Turn@10 | 40 | 40 | 36 | 0 | 40 | 40 | — | clears | clears | unknown |  |
| BEAM | contradiction_resolution | Turn All@5 | 22 | 40 | 36 | 14 | 22 | 27 | — | short | short | unknown | no all-in-pool bit |
| BEAM | contradiction_resolution | Turn All@10 | 27 | 40 | 36 | 9 | 27 | 27 | — | short | short | unknown |  |
| BEAM | contradiction_resolution | Session@1 | 29 | 40 | 36 | 7 | 40 | 40 | — | clears | clears | unknown |  |
| BEAM | contradiction_resolution | Session@5 | 40 | 40 | 36 | 0 | 40 | 40 | — | clears | clears | unknown |  |
| BEAM | contradiction_resolution | Session@10 | 40 | 40 | 36 | 0 | 40 | 40 | — | clears | clears | unknown |  |
| BEAM | contradiction_resolution | Session All@5 | 20 | 40 | 36 | 16 | 20 | 30 | — | short | short | unknown |  |
| BEAM | contradiction_resolution | Session All@10 | 30 | 40 | 36 | 6 | 30 | 30 | — | short | short | unknown |  |
| BEAM | event_ordering | Turn@1 | 7 | 40 | 36 | 29 | 14 | 22 | 39 | short | short | clears | wide 39; missmap {'t5_not1': 7, 'pool_not_k': 17, 'k_not5': 8, 'no_pool': 1} |
| BEAM | event_ordering | Turn@5 | 14 | 40 | 36 | 22 | 14 | 22 | 39 | short | short | clears |  |
| BEAM | event_ordering | Turn@10 | 22 | 40 | 36 | 14 | 22 | 22 | 39 | short | short | clears |  |
| BEAM | event_ordering | Turn All@5 | 0 | 40 | 36 | 36 | 0 | 0 | — | short | short | unknown | no all-in-pool bit |
| BEAM | event_ordering | Turn All@10 | 0 | 40 | 36 | 36 | 0 | 0 | — | short | short | unknown |  |
| BEAM | event_ordering | Session@1 | 7 | 40 | 36 | 29 | 22 | 27 | — | short | short | unknown |  |
| BEAM | event_ordering | Session@5 | 22 | 40 | 36 | 14 | 22 | 27 | — | short | short | unknown |  |
| BEAM | event_ordering | Session@10 | 27 | 40 | 36 | 9 | 27 | 27 | — | short | short | unknown |  |
| BEAM | event_ordering | Session All@5 | 0 | 40 | 36 | 36 | 0 | 0 | — | short | short | unknown |  |
| BEAM | event_ordering | Session All@10 | 0 | 40 | 36 | 36 | 0 | 0 | — | short | short | unknown |  |
| BEAM | information_extraction | Turn@1 | 20 | 40 | 36 | 16 | 27 | 29 | 32 | short | short | short | wide 32; missmap {'t5_not1': 7, 'pool_not_k': 3, 'no_pool': 8, 'k_not5': 2} |
| BEAM | information_extraction | Turn@5 | 27 | 40 | 36 | 9 | 27 | 29 | 32 | short | short | short |  |
| BEAM | information_extraction | Turn@10 | 29 | 40 | 36 | 7 | 29 | 29 | 32 | short | short | short |  |
| BEAM | information_extraction | Turn All@5 | 20 | 40 | 36 | 16 | 20 | 21 | — | short | short | unknown | no all-in-pool bit |
| BEAM | information_extraction | Turn All@10 | 21 | 40 | 36 | 15 | 21 | 21 | — | short | short | unknown |  |
| BEAM | information_extraction | Session@1 | 15 | 40 | 36 | 21 | 25 | 27 | — | short | short | unknown |  |
| BEAM | information_extraction | Session@5 | 25 | 40 | 36 | 11 | 25 | 27 | — | short | short | unknown |  |
| BEAM | information_extraction | Session@10 | 27 | 40 | 36 | 9 | 27 | 27 | — | short | short | unknown |  |
| BEAM | information_extraction | Session All@5 | 25 | 40 | 36 | 11 | 25 | 27 | — | short | short | unknown |  |
| BEAM | information_extraction | Session All@10 | 27 | 40 | 36 | 9 | 27 | 27 | — | short | short | unknown |  |
| BEAM | instruction_following | Turn@1 | 6 | 40 | 36 | 30 | 18 | 21 | 36 | short | short | clears | wide 36; missmap {'t5_not1': 12, 'pool_not_k': 15, 'no_pool': 5, 'k_not5': 2}; t10-t5=3 but k_not5=2 |
| BEAM | instruction_following | Turn@5 | 18 | 40 | 36 | 18 | 18 | 21 | 36 | short | short | clears |  |
| BEAM | instruction_following | Turn@10 | 21 | 40 | 36 | 15 | 21 | 21 | 36 | short | short | clears |  |
| BEAM | instruction_following | Turn All@5 | 13 | 40 | 36 | 23 | 13 | 16 | — | short | short | unknown | no all-in-pool bit |
| BEAM | instruction_following | Turn All@10 | 16 | 40 | 36 | 20 | 16 | 16 | — | short | short | unknown |  |
| BEAM | instruction_following | Session@1 | 4 | 40 | 36 | 32 | 16 | 17 | — | short | short | unknown |  |
| BEAM | instruction_following | Session@5 | 16 | 40 | 36 | 20 | 16 | 17 | — | short | short | unknown |  |
| BEAM | instruction_following | Session@10 | 17 | 40 | 36 | 19 | 17 | 17 | — | short | short | unknown |  |
| BEAM | instruction_following | Session All@5 | 15 | 40 | 36 | 21 | 15 | 17 | — | short | short | unknown |  |
| BEAM | instruction_following | Session All@10 | 17 | 40 | 36 | 19 | 17 | 17 | — | short | short | unknown |  |
| BEAM | knowledge_update | Turn@1 | 30 | 40 | 36 | 6 | 37 | 39 | 39 | clears | clears | clears | wide 39; missmap {'t5_not1': 7, 'k_not5': 2, 'no_pool': 1} |
| BEAM | knowledge_update | Turn@5 | 37 | 40 | 36 | 0 | 37 | 39 | 39 | clears | clears | clears |  |
| BEAM | knowledge_update | Turn@10 | 39 | 40 | 36 | 0 | 39 | 39 | 39 | clears | clears | clears |  |
| BEAM | knowledge_update | Turn All@5 | 24 | 40 | 36 | 12 | 24 | 27 | — | short | short | unknown | no all-in-pool bit |
| BEAM | knowledge_update | Turn All@10 | 27 | 40 | 36 | 9 | 27 | 27 | — | short | short | unknown |  |
| BEAM | knowledge_update | Session@1 | 24 | 40 | 36 | 12 | 37 | 38 | — | clears | clears | unknown |  |
| BEAM | knowledge_update | Session@5 | 37 | 40 | 36 | 0 | 37 | 38 | — | clears | clears | unknown |  |
| BEAM | knowledge_update | Session@10 | 38 | 40 | 36 | 0 | 38 | 38 | — | clears | clears | unknown |  |
| BEAM | knowledge_update | Session All@5 | 29 | 40 | 36 | 7 | 29 | 31 | — | short | short | unknown |  |
| BEAM | knowledge_update | Session All@10 | 31 | 40 | 36 | 5 | 31 | 31 | — | short | short | unknown |  |
| BEAM | multi_session_reasoning | Turn@1 | 21 | 40 | 36 | 15 | 34 | 36 | 40 | short | clears | clears | wide 40; missmap {'t5_not1': 13, 'pool_not_k': 4, 'k_not5': 2} |
| BEAM | multi_session_reasoning | Turn@5 | 34 | 40 | 36 | 2 | 34 | 36 | 40 | short | clears | clears |  |
| BEAM | multi_session_reasoning | Turn@10 | 36 | 40 | 36 | 0 | 36 | 36 | 40 | clears | clears | clears |  |
| BEAM | multi_session_reasoning | Turn All@5 | 4 | 40 | 36 | 32 | 4 | 7 | — | short | short | unknown | no all-in-pool bit |
| BEAM | multi_session_reasoning | Turn All@10 | 7 | 40 | 36 | 29 | 7 | 7 | — | short | short | unknown |  |
| BEAM | multi_session_reasoning | Session@1 | 20 | 40 | 36 | 16 | 33 | 36 | — | short | clears | unknown |  |
| BEAM | multi_session_reasoning | Session@5 | 33 | 40 | 36 | 3 | 33 | 36 | — | short | clears | unknown |  |
| BEAM | multi_session_reasoning | Session@10 | 36 | 40 | 36 | 0 | 36 | 36 | — | clears | clears | unknown |  |
| BEAM | multi_session_reasoning | Session All@5 | 8 | 40 | 36 | 28 | 8 | 15 | — | short | short | unknown |  |
| BEAM | multi_session_reasoning | Session All@10 | 15 | 40 | 36 | 21 | 15 | 15 | — | short | short | unknown |  |
| BEAM | preference_following | Turn@1 | 7 | 39 | 36 | 29 | 14 | 16 | 30 | short | short | short | wide 30; missmap {'k_not5': 2, 'pool_not_k': 14, 't5_not1': 7, 'no_pool': 9} |
| BEAM | preference_following | Turn@5 | 14 | 39 | 36 | 22 | 14 | 16 | 30 | short | short | short |  |
| BEAM | preference_following | Turn@10 | 16 | 39 | 36 | 20 | 16 | 16 | 30 | short | short | short |  |
| BEAM | preference_following | Turn All@5 | 11 | 39 | 36 | 25 | 11 | 13 | — | short | short | unknown | no all-in-pool bit |
| BEAM | preference_following | Turn All@10 | 13 | 39 | 36 | 23 | 13 | 13 | — | short | short | unknown |  |
| BEAM | preference_following | Session@1 | 4 | 39 | 36 | 32 | 14 | 24 | — | short | short | unknown |  |
| BEAM | preference_following | Session@5 | 14 | 39 | 36 | 22 | 14 | 24 | — | short | short | unknown |  |
| BEAM | preference_following | Session@10 | 24 | 39 | 36 | 12 | 24 | 24 | — | short | short | unknown |  |
| BEAM | preference_following | Session All@5 | 14 | 39 | 36 | 22 | 14 | 24 | — | short | short | unknown |  |
| BEAM | preference_following | Session All@10 | 24 | 39 | 36 | 12 | 24 | 24 | — | short | short | unknown |  |
| BEAM | summarization | Turn@1 | 6 | 36 | 33 | 27 | 17 | 25 | 34 | short | short | clears | wide 34; missmap {'t5_not1': 11, 'k_not5': 8, 'pool_not_k': 9, 'no_pool': 2} |
| BEAM | summarization | Turn@5 | 17 | 36 | 33 | 16 | 17 | 25 | 34 | short | short | clears |  |
| BEAM | summarization | Turn@10 | 25 | 36 | 33 | 8 | 25 | 25 | 34 | short | short | clears |  |
| BEAM | summarization | Turn All@5 | 0 | 36 | 33 | 33 | 0 | 0 | — | short | short | unknown | no all-in-pool bit |
| BEAM | summarization | Turn All@10 | 0 | 36 | 33 | 33 | 0 | 0 | — | short | short | unknown |  |
| BEAM | summarization | Session@1 | 6 | 36 | 33 | 27 | 24 | 26 | — | short | short | unknown |  |
| BEAM | summarization | Session@5 | 24 | 36 | 33 | 9 | 24 | 26 | — | short | short | unknown |  |
| BEAM | summarization | Session@10 | 26 | 36 | 33 | 7 | 26 | 26 | — | short | short | unknown |  |
| BEAM | summarization | Session All@5 | 0 | 36 | 33 | 33 | 0 | 1 | — | short | short | unknown |  |
| BEAM | summarization | Session All@10 | 1 | 36 | 33 | 32 | 1 | 1 | — | short | short | unknown |  |
| BEAM | temporal_reasoning | Turn@1 | 27 | 40 | 36 | 9 | 37 | 39 | 40 | clears | clears | clears | wide 40; missmap {'t5_not1': 10, 'k_not5': 2, 'pool_not_k': 1} |
| BEAM | temporal_reasoning | Turn@5 | 37 | 40 | 36 | 0 | 37 | 39 | 40 | clears | clears | clears |  |
| BEAM | temporal_reasoning | Turn@10 | 39 | 40 | 36 | 0 | 39 | 39 | 40 | clears | clears | clears |  |
| BEAM | temporal_reasoning | Turn All@5 | 24 | 40 | 36 | 12 | 24 | 27 | — | short | short | unknown | no all-in-pool bit |
| BEAM | temporal_reasoning | Turn All@10 | 27 | 40 | 36 | 9 | 27 | 27 | — | short | short | unknown |  |
| BEAM | temporal_reasoning | Session@1 | 24 | 40 | 36 | 12 | 38 | 40 | — | clears | clears | unknown |  |
| BEAM | temporal_reasoning | Session@5 | 38 | 40 | 36 | 0 | 38 | 40 | — | clears | clears | unknown |  |
| BEAM | temporal_reasoning | Session@10 | 40 | 40 | 36 | 0 | 40 | 40 | — | clears | clears | unknown |  |
| BEAM | temporal_reasoning | Session All@5 | 26 | 40 | 36 | 10 | 26 | 33 | — | short | short | unknown |  |
| BEAM | temporal_reasoning | Session All@10 | 33 | 40 | 36 | 3 | 33 | 33 | — | short | short | unknown |  |

### What that says about the large gaps

- **LoCoMo Turn@1, gap 578 from Step 34's 805 (587 from the attached 796) to 1383.** Slot ceiling 1127 is short by 256. Top-10 ceiling 1226 is short by 157. Pool@50 ceiling 1415 clears by 32, and the wide pool 1492 clears by 109. The bench is a pool reorder plus the admissions that make those 1415 contain the right golds. It is not a top-5 swap. By category, pool@50 clears Single-Hop (810≥757) and Temporal (299≥289) and misses Multi-Hop (243<254, need 11 more admissions) and Open-Domain (63<83; the wide pool is 78, so at least 5 of 14 no_pool questions need a new posting).
- **BEAM Turn All@5, gap 202 (118 to 320), and All@10, gap 182 (138 to 320).** The top-10 set-cover ceiling is 138. A quota inside today's ten cannot get near 320. event_ordering and summarization are All@5 0 and All@10 0. The other golds are not in the official list. Admission first, then the once-only quota.
- **BEAM Session@1, gap 187 (133 to 320).** Session@5 is 249, Session@10 is 275. Both are short of 320. Categories already clear on Session@5: contradiction_resolution (40), knowledge_update (37), temporal_reasoning (38). Short at Session@10: event_ordering 27, information_extraction 27, instruction_following 17, preference_following 24, summarization 26, all against a need of 36 (33 for summarization). Those five need a session that is not in the current ten.
- **LoCoMo Session@1, gap 250 (1133 to 1383).** Session@5 is 1416 and clears. The bench is an ordering problem. The categories are not: Multi-Hop Session@5 is 246 against 254, Open-Domain Session@5 is 65 against 83, and LoCoMo has no Session@10 bit, so those two categories need a new session inside the five. Single-Hop (806≥757) and Temporal (299≥289) clear on Session@5.
- **LME Turn@1, gap 70 from Step 34's 353 (73 from the attached 350) to 423.** Top-5 ceiling 428 clears. Preference does not: Turn@5 and Turn@10 are both 19 against 27, and the wide pool is 30, so 8 of its 11 pool_not_k questions have to enter the ten and then slot 1. Temporal Turn@5 is 111 against 115 and Turn@10 is 116, so it needs a pull from positions 6–10, not a new posting.
- **LME Turn All@5, gap 109 (314 to 423).** A top-5 permute does not raise it. There is no turn All@10. The deficit is multi-session 53 against 109, temporal 74 against 115, preference 14 against 27, and knowledge-update 59 against 65. single-session-user (62≥58) and single-session-assistant (52≥51) are already there.

Metrics already at the floor, left alone by these rules: LME Session@1/5/10 (430/463/468), LME Session All@5/All@10 (424/447), LME Turn@5/10 (428/437), LoCoMo Session@5 (1416), and the per-category cells whose slot ceiling already clears (LME single-session-user and knowledge-update Turn@1, BEAM contradiction_resolution Turn@1, BEAM knowledge_update and temporal_reasoning Turn@1).

### Per-rule ceiling class

The 90 mechanisms are unchanged. Each one is labeled with the ceiling it can actually move. SLOT is hook `6265–6272` permuting the current five. K_PULL is the same hook inserting a pool turn into the five. ADMIT is hook `5633–5730` (`t_scene[ti] > 0.0`). FEAT is hook `3121–3144`.

| Id | Name | Class | Hook | What 90% it can move | What it cannot reach |
|---|---|---|---|---|---|
| S4-001 | Adhikāra frame stack | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-002 | Vipratiṣedha rightmost group | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-003 | Pūrvatrāsiddha snapshot | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-004 | Ekaśeṣa counted residue | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-005 | It-saṃjñā marker class | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-006 | Lopa place-gap | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-007 | Bahuvrīhi external referent | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-008 | Dvandva both-or-zero | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-009 | Parāṅgavat head gate | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-010 | Avyayībhāva left-head order | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-011 | Trairūpya three-count gate | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. Also Session@1, up to Session@5 (LoCoMo bench clears) or Session@10 where it exists (BEAM bench does not). | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-012 | Anupalabdhi under conditions | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. Also Session@1, up to Session@5 (LoCoMo bench clears) or Session@10 where it exists (BEAM bench does not). | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-013 | Kārya-hetu empty successor | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-014 | Svabhāva subset gate | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-015 | Asiddha-hetu rejection | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-016 | Viruddha-hetu majority | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-017 | Anaikāntika width | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-018 | Pakṣābhāsa self-negation | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-019 | Avacchedaka clause lock | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-020 | Abhāva triple posting | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-021 | Prāgabhāva interval open | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-022 | Pradhvaṃsa destruction point | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-023 | Anyonyābhāva exactly-one | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-024 | Saptabhaṅgī class quota | K_PULL | `6265-6272` | All@5, up to the questions whose other golds are already in the pool. | BEAM All@5 to 320 while All@10 is 138. LME All@5 has no All@10 ceiling in the bits. |
| S4-025 | Nikṣepa mention zero | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-026 | Yogavibhāga clause length | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-027 | Gaṇapāṭha list object | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-028 | Pratyāhāra end-pair | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-029 | Anukramaṇī five-count card | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. Also Session@1, up to Session@5 (LoCoMo bench clears) or Session@10 where it exists (BEAM bench does not). | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-030 | Grantha extent checksum | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. Also Session@1, up to Session@5 (LoCoMo bench clears) or Session@10 where it exists (BEAM bench does not). | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-031 | Pecha consecutive leaf | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-032 | Explicit scope close | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-033 | Mūla containment seat | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-034 | Sūtra/bhāṣya length alternation | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-035 | Vārttika anukta field | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-036 | Ṭīkā depth quota | K_PULL | `6265-6272` | All@5, up to the questions whose other golds are already in the pool. | BEAM All@5 to 320 while All@10 is 138. LME All@5 has no All@10 ceiling in the bits. |
| S4-037 | Adhikaraṇa limb quota | K_PULL | `6265-6272` | All@5, up to the questions whose other golds are already in the pool. | BEAM All@5 to 320 while All@10 is 138. LME All@5 has no All@10 ceiling in the bits. |
| S4-038 | Pañcāvayava example link | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-039 | Dūṣaṇa missing-rūpa | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-040 | Natthi change-point | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-041 | Adhikāra plus adhyāhāra | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-042 | Bahuvrīhi plus kārya-hetu | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-043 | Trairūpya plus anupalabdhi | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-044 | Avacchedaka plus dvandva | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-045 | Ekaśeṣa plus gaṇapāṭha | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-046 | Asiddha plus vipratiṣedha | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-047 | It-marker plus lopa gap | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-048 | Yogavibhāga plus aṅga | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-049 | Pratyāhāra plus nigamana | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-050 | Anukramaṇī plus grantha | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. Also Session@1, up to Session@5 (LoCoMo bench clears) or Session@10 where it exists (BEAM bench does not). | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-051 | Alternation plus mūla containment | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-052 | Anukta plus ṭīkā depth | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-053 | Limb quota plus saptabhaṅgī | K_PULL | `6265-6272` | All@5, up to the questions whose other golds are already in the pool. | BEAM All@5 to 320 while All@10 is 138. LME All@5 has no All@10 ceiling in the bits. |
| S4-054 | Udāharaṇa plus tātparya | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-055 | Nikṣepa plus it-marker | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-056 | Ekādeśa plus udātta | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-057 | Pecha leaf min-cover | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-058 | Scope close plus frame pop | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-059 | Three hetvābhāsa gates | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-060 | Pakṣābhāsa plus dūṣaṇa | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-061 | Abhāva triple plus anyonya | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-062 | Prāg plus pradhvaṃsa interval | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-063 | Natthi at distance one | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-064 | Kośa bits plus anukramaṇī | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. Also Session@1, up to Session@5 (LoCoMo bench clears) or Session@10 where it exists (BEAM bench does not). | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-065 | Lakṣaṇā under blocked literal | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-066 | Sakṛd plus grantha quota | K_PULL | `6265-6272` | All@5, up to the questions whose other golds are already in the pool. | BEAM All@5 to 320 while All@10 is 138. LME All@5 has no All@10 ceiling in the bits. |
| S4-067 | Gaṇa plus dvandva | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-068 | Bahuvrīhi plus avacchedaka | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-069 | Trairūpya plus kārya-hetu | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-070 | Anvita within and across | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-071 | Adhikāra plus pratyāhāra | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-072 | Vipratiṣedha plus aṅga | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-073 | Ekaśeṣa plus saptabhaṅgī | K_PULL | `6265-6272` | All@5, up to the questions whose other golds are already in the pool. | BEAM All@5 to 320 while All@10 is 138. LME All@5 has no All@10 ceiling in the bits. |
| S4-074 | Yogavibhāga plus simile field | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-075 | Alternation plus hrasva/dīrgha | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-076 | Correction plus viruddha | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-077 | Ṭīkā depth plus bahuvrīhi | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-078 | Limb plus avayava agreement | K_PULL | `6265-6272` | All@5, up to the questions whose other golds are already in the pool. | BEAM All@5 to 320 while All@10 is 138. LME All@5 has no All@10 ceiling in the bits. |
| S4-079 | Vyāpaka-anupalabdhi | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. Also Session@1, up to Session@5 (LoCoMo bench clears) or Session@10 where it exists (BEAM bench does not). | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-080 | Ekādeśa plus lopa place | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-081 | Volume boundary plus asiddha | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. Also Session@1, up to Session@5 (LoCoMo bench clears) or Session@10 where it exists (BEAM bench does not). | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-082 | Mūla plus ekaśeṣa | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-083 | Change-point plus trairūpya | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-084 | Sphoṭa plus dvandva | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-085 | Lakṣaṇā plus anupalabdhi | ADMIT | `5633-5730` | Turn@1 up to the pool. LoCoMo pool@50 1415 and BEAM pool@50 328 clear the bench lines. | Multi-Hop pool@50 243<254 and Open-Domain pool@50 63<83. All@5, which needs every gold, not one. |
| S4-086 | Sakṛd plus vipratiṣedha | K_PULL | `6265-6272` | All@5, up to the questions whose other golds are already in the pool. | BEAM All@5 to 320 while All@10 is 138. LME All@5 has no All@10 ceiling in the bits. |
| S4-087 | Gaṇa plus pratyāhāra | FEAT | `3121-3144` | no_pool, which is the only way past a short wide pool. | A guarantee of 90%. The posting either retrieves the missing gold or it does not; the bits cannot say which questions move. |
| S4-088 | Saṃśaya yields to siddhānta | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-089 | Hrasva count plus grantha | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. Also Session@1, up to Session@5 (LoCoMo bench clears) or Session@10 where it exists (BEAM bench does not). | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |
| S4-090 | Dūṣaṇa plus anaikāntika | SLOT | `6265-6272` | LME Turn@1 up to 428. The Turn@5-not-1 mass on every bench. | LoCoMo Turn@1 (1127<1383), BEAM Turn@1 (238<320), every All@5. |

Rows in the ceiling table: 177. Rows still short at every computable ceiling: 80.
