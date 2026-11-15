# Funnel S1 — what the attached scores can and cannot say

Base under design: `e2ecd1da037974c1835f563ad8b046f4a4ca3629` on `perf/c2-step33-triangulation`. The attached per-question file is the 44-algo tip `bd2693e2` (`/workspace/c2-algos-bins/t51b-attr-unseated-opener-t10`, sha256 `e93411ac566c4c83f92a83204e70ea85ae743445a3256dc5cb4eacc2523e6f84`). Step 31 (Exchequer) skips named-speaker dialogues, event-sequence queries, temporal queries, and preference queries (`apply_c2_step31_exchequer_split`). Step 33 runs for every query but returns before any swap unless `c2_named_session` is true, and `C2_STEP33_LOG.md` records Multi-Hop Turn@1 still 97 and BEAM byte-identical. The attached counts are therefore the current counts for these two slices.

## What each dump field means

Verified in `write_one_replay_dump` (`crates/vestige-core/src/retrieval.rs` 7426-7443) and `C2_STEP33_LOG.md`.

`VESTIGE_REPLAY_DUMP` is off the ranking path. `maybe_dump_replay` (7262) returns immediately when the variable is unset. The dump is written after Step 33, at the call on line 6273.

- `final_ids` / `seated_ids`: official order after the gatekeeper and every C2 seat, including Step 33. Rank 0 is slot 1.
- `pre_ids`: that seated list first, then turns with `t_scene > 0` in `t_scene` order until 50 (lines 7429-7440). `rank_pre` is the index in `pre_ids` (7441-7443).
- `pre_ids[0] == final_ids[0]` by construction. "Gold at pre_rank 0" is the same count as "gold at final rank 0". It is not an independent stage.
- `t_scene` is not rewritten by seating. There is no post-Kemeny snapshot. The gatekeeper Kemeny-orders slots `1..k` with slot 0 frozen, then the C2 seats run, then the dump sees the final list.
- `qtok_present` and `turn_ic_novel` read `TurnIx.body` and are dump-only. Ranking does not read that field.

## What each per-question bit means

Checked by summing the attached `perq` arrays against the official `cats` in the same file.

LoCoMo value length 5, order `[s1, s5, t1, t5, t10]`. Bit sums over 1536 questions are `[1133, 1416, 796, 1127, 1226]`, which match `locomo.all`. Per-question keys are `conv:qid` and do not carry the category, so the slice totals below are the `cats` object, not a re-sum of bits by slice. The global sums confirm the bit order.

BEAM value length 7, order `[t1, t5, t10, all5, s1, s5, sall5]`. `event_ordering` bits sum to `[7, 14, 22, 0, 7, 22, 0]`, which match `cats` `t1=7, t5=14, t10=22, all5=0, s1=7, s5=22, sall5=0`. `all10`, `s10`, and `sall10` exist only in `cats` (event_ordering `all10=0`, `s10=27`, `sall10=0`).

LME value length 7 is `[session any1, any5, any10, turn any1, any5, any10, turn all5]`. Session `all5` (424) and `all10` (447) are not in the bits. This shard does not target LME.

## What the miss-map stages mean

`missmap-bd2693e2.txt` partitions Turn@1 misses. The header's `pool` is wider than `pool@50` (LoCoMo 1492 vs 1415, BEAM 329 vs 328). `C2_STEP33_LOG.md` identifies `pool@50` with gold membership in `pre_ids`. Gold in official k is inside `pre_ids`, because `pre_ids` starts with the seated list.

Among misses:

- `t5_not1`: gold is in the official top 5 and is not rank 0. A slot-1 reorder can fix these and nothing else.
- `k_not5`: gold is in official k (Turn@10) and not in the top 5.
- `pool_not_k`: gold is in the wider pool and not in official k.
- `no_pool`: gold is not in that wider pool. In the ranker, the submodular pool is `t_scene > 0` (line 5633). A gold with `t_scene <= 0` cannot be selected. Fixing it is an ingest / index change.

Multi-Hop misses sum `61+28+83+13 = 185 = 282-97`. Event-ordering misses sum `7+8+17+1 = 33 = 40-7`.

## Five-line funnel

The brief asks for gold in corpus, gold in pool@50, gold at pre_rank 0, gold in top 5, gold at final rank 0.

### LoCoMo Multi-Hop (n=282)

| stage | count | source |
| --- | ---: | --- |
| gold in corpus | 282 | `C2_STEP33_LOG.md` lines 27-31, step31 binary. The attachment has no per-slice corpus bit. LoCoMo overall corpus is 1535/1536 in that same log; the missing evidence id is not attributed to a slice by the attachment. |
| gold in pool@50 | 243 | same log. The attachment cannot compute per-slice pool@50. Global LoCoMo pool@50 is 1415 in both the miss map and that log, and Step 33 does not write `t_scene`. |
| gold at pre_rank 0 | 97 | not a separate measurement. Equals final rank 0 by lines 7429-7443. The perq file has no pre_rank field. |
| gold in top 5 | 158 | perq `cats` `t5`, and the same log's final top5 |
| gold at final rank 0 | 97 | perq `cats` `t1`. Step 33 left this number at 97 |

Miss-map pool (wider than pool@50) = 282 − 13 = 269. Of those, 243 are in pool@50 on the step31 dump, so 26 golds have scene mass and still miss the dumped 50. Official k holds 186. `pool_not_k` 83 = (269−243) + (243−186).

### BEAM event_ordering (n=40)

| stage | count | source |
| --- | ---: | --- |
| gold in corpus | not in the attachment, and the Step 33 log does not break BEAM corpus by category. BEAM overall corpus is 355/355 in that log. | |
| gold in pool@50 | 38 or 39 | attachment bound, not a direct cell. Miss-map pool = 40−1 = 39. Global BEAM pool 329 vs pool@50 328, so at most one question in the whole bench is in the wider pool and outside pool@50. Gold in k (22) is inside pool@50. |
| gold at pre_rank 0 | 7 | equals final rank 0 by the dump writer. Not independently stored. |
| gold in top 5 | 14 | perq bits `[7,14,22,0,7,22,0]` and `cats.t5` |
| gold at final rank 0 | 7 | perq `cats.t1` |

`all5 = 0` and `all10 = 0`. The target in the brief is Turn@1 36/40, not All@5. All@5 stays zero while any gold of a multi-gold question is outside the top 5, which the one-per-session cap (lines 5619-5625) forces whenever two golds share a chat.

## Ceilings

A slot-1 reorder that does not change top-5 membership:

- Multi-Hop: 158/282 (56%). Target 254. Short by 96.
- event_ordering: 14/40 (35%). Target 36. Short by 22.

A reorder of official k, which current C2 seats do not do (they swap inside the top 5 only; Step 33 says top-5 membership is unchanged):

- Multi-Hop: 186/282 (66%). Still 68 short of 254.
- event_ordering: 22/40 (55%). Still 14 short of 36.

Recall ceiling if every in-pool gold can be brought into slot 1 and the `no_pool` golds cannot:

- Multi-Hop: 269/282 (95.4%), which clears 254. The 13 `no_pool` golds are not required if 68 of the 83 `pool_not_k` golds enter k and every in-k miss is fixed (186+68=254).
- event_ordering: 39/40 (97.5%), which clears 36. The single `no_pool` gold is not required if 14 of the 17 `pool_not_k` golds enter k and every in-k miss is fixed (22+14=36).

Pure ordering cannot reach 90% on either slice. Algorithms tagged `ORDERING` spend `t5_not1` (61 and 7) and, if they reorder inside k rather than only inside the top 5, `k_not5` (28 and 8). Algorithms tagged `RECALL/INDEX` are the ones that can touch `pool_not_k` and `no_pool`.

## Why the existing seats do not already spend the event-ordering top 5

`is_event_sequence` is set in `classify` (lines 3969-3972) by a closed list of phrases (`order of`, `order in which`, `first to last`, `earliest to latest`, `chronological`, `order the`, `order these`, `timeline of`, `i brought up`). The C2 seat block that ends just before Step 31 is inside `if !it.is_event_sequence`. Those seats never run when the classifier fires. Step 33 is called outside that guard (lines 6265-6272) and then returns false on user/assistant transcripts. BEAM event ordering is therefore unseated by the whole C2 stack. A new ordering rule has to be invoked at that call site, before `maybe_dump_replay` on line 6273, and it must not return on `is_event_sequence`. Adding phrases to `classify` is a domain list and is out of bounds. The admission change is a new predicate on the `session_capped` continue at lines 5676-5679. The literal `1` at lines 5619-5625 stays.

## Stages the attachment cannot compute

- Per-slice gold-in-corpus.
- Per-slice gold in pool@50, except the event-ordering bound 38 or 39 above. The Multi-Hop figure 243 is the repo funnel log, cited as such.
- An independent pre_rank. The perq file has none. Equality with final rank 0 is the dump-writer invariant, not a second measurement.
- Which of the 17 event-ordering `pool_not_k` misses are inside pool@50 versus only in the wider pool. At most one question in all of BEAM is in that gap.

## Per-metric ceilings, one row each

Target note of 2026-11-15: 90% on every metric of every benchmark and every category. Step 34 (`62041ef5`) is the Turn@1 floor in that note (LME 353, LoCoMo 805, BEAM 160). Every other cell in the note is bd2693e2. The stage counts below are the bd2693e2 miss map, which is the only per-category partition. Step 34's Turn@1 lifts (LME +3, LoCoMo +9, BEAM +4) sit inside those stages, so they do not raise a ceiling built from the stages.

Reading a cell `BEAM T@1 156→188`: if this rule alone repaired every question in the stages it claims, the bench metric would reach 188. That is an upper bound. It is not a forecast. Two bounds are doing the work. The miss map does not say how often the orthographic pattern occurs. For All@5 and for session recall, the miss map also does not say where the other golds sit, so those ceilings use `n` or `n − no_pool`.

What a stage claim is allowed to move:

- `t5`: Turn@1, up to the current Turn@5. Turn@5, Turn@10, All@5 do not move.

- `k`: golds in official k but outside the top 5 enter the top 5. Turn@1 and Turn@5 move. Turn@10 does not.

- `pool`: golds in the wider pool but outside k enter k. Turn@1, Turn@5, and Turn@10 move.

- `none`: golds with no pool mass. Ingest. Same three turn metrics, and it raises the All@5 cap from `n − no_pool` to `n` when `all5` is also claimed.

- `s_re`: Session@1 can reach Session@5. The session list does not grow.

- `s_k`: Session@1 can reach Session@10, and Session@5 can reach Session@10, by reordering the current session list (the swap after line 5602). The list does not grow.

- `s_rec`: a session with `s_score == 0` can enter at line 5489. Upper bound is the category `n`.

- `all5` / `all10`: upper bound is `n − no_pool`, or `n` when `none` is also claimed.

Ordering rules do not claim `pool`, `none`, `all5`, or `s_rec`. The hook on each row is the one in the CSV.

### Bench union

| bench | metric | bd2693e2 now | 90% need | union ceiling | clears |
| --- | --- | ---: | ---: | ---: | --- |
| LME | Turn@1 | 350 | 423 | 461 | yes |
| LME | Turn@5 | 428 | 423 | 461 | yes |
| LME | Turn@10 | 437 | 423 | 465 | yes |
| LME | Turn All@5 | 314 | 423 | 462 | yes |
| LME | Session@1 | 430 | 423 | 453 | yes |
| LME | Session@5 | 463 | 423 | 467 | yes |
| LME | Session@10 | 468 | 423 | 468 | yes |
| LOC | Turn@1 | 796 | 1383 | 1519 | yes |
| LOC | Turn@5 | 1127 | 1383 | 1519 | yes |
| LOC | Turn@10 | 1226 | 1383 | 1519 | yes |
| LOC | Session@1 | 1133 | 1383 | 1479 | yes |
| LOC | Session@5 | 1416 | 1383 | 1479 | yes |
| BEAM | Turn@1 | 156 | 320 | 351 | yes |
| BEAM | Turn@5 | 238 | 320 | 351 | yes |
| BEAM | Turn@10 | 267 | 320 | 351 | yes |
| BEAM | Turn All@5 | 118 | 320 | 351 | yes |
| BEAM | Turn All@10 | 138 | 320 | 351 | yes |
| BEAM | Session@1 | 133 | 320 | 349 | yes |
| BEAM | Session@5 | 249 | 320 | 349 | yes |
| BEAM | Session@10 | 275 | 320 | 349 | yes |

LME Session All@5 is 424 and Session All@10 is 447, both already at or above 423. No row claims them. LoCoMo has no All@5 metric in the published cats. LoCoMo Session@5 is 1416, which clears the bench line 1383 and does not clear Multi-Hop (246, need 254) or Open-Domain (65, need 83). Those two category shortfalls are in the category table.

### Category union, gaps only

Category union ceilings. `need` is ceil(0.9·n). A ceiling is an upper bound, not a predicted gain.

| bench | category | metric | now | need | union ceiling | clears |
| --- | --- | --- | ---: | ---: | ---: | --- |
| LOC | Single-Hop | Turn@1 | 494 | 757 | 832 | yes |
| LOC | Single-Hop | Turn@5 | 686 | 757 | 832 | yes |
| LOC | Single-Hop | Turn@10 | 737 | 757 | 832 | yes |
| LOC | Single-Hop | S@1 | 699 | 757 | 806 | yes |
| LOC | Multi-Hop | Turn@1 | 97 | 254 | 282 | yes |
| LOC | Multi-Hop | Turn@5 | 158 | 254 | 282 | yes |
| LOC | Multi-Hop | Turn@10 | 186 | 254 | 282 | yes |
| LOC | Multi-Hop | S@1 | 156 | 254 | 282 | yes |
| LOC | Multi-Hop | S@5 | 246 | 254 | 282 | yes |
| LOC | Temporal | Turn@1 | 187 | 289 | 313 | yes |
| LOC | Temporal | Turn@5 | 248 | 289 | 313 | yes |
| LOC | Temporal | Turn@10 | 260 | 289 | 313 | yes |
| LOC | Temporal | S@1 | 242 | 289 | 299 | yes |
| LOC | Open-Domain | Turn@1 | 18 | 83 | 92 | yes |
| LOC | Open-Domain | Turn@5 | 35 | 83 | 92 | yes |
| LOC | Open-Domain | Turn@10 | 43 | 83 | 92 | yes |
| LOC | Open-Domain | S@1 | 36 | 83 | 92 | yes |
| LOC | Open-Domain | S@5 | 65 | 83 | 92 | yes |
| BEAM | contradiction_resolution | Turn@1 | 32 | 36 | 40 | yes |
| BEAM | contradiction_resolution | All@5 | 22 | 36 | 40 | yes |
| BEAM | contradiction_resolution | All@10 | 27 | 36 | 40 | yes |
| BEAM | contradiction_resolution | S@1 | 29 | 36 | 40 | yes |
| BEAM | event_ordering | Turn@1 | 7 | 36 | 39 | yes |
| BEAM | event_ordering | Turn@5 | 14 | 36 | 39 | yes |
| BEAM | event_ordering | Turn@10 | 22 | 36 | 39 | yes |
| BEAM | event_ordering | All@5 | 0 | 36 | 39 | yes |
| BEAM | event_ordering | All@10 | 0 | 36 | 39 | yes |
| BEAM | event_ordering | S@1 | 7 | 36 | 40 | yes |
| BEAM | event_ordering | S@5 | 22 | 36 | 40 | yes |
| BEAM | event_ordering | S@10 | 27 | 36 | 40 | yes |
| BEAM | information_extraction | Turn@1 | 20 | 36 | 40 | yes |
| BEAM | information_extraction | Turn@5 | 27 | 36 | 40 | yes |
| BEAM | information_extraction | Turn@10 | 29 | 36 | 40 | yes |
| BEAM | information_extraction | All@5 | 20 | 36 | 40 | yes |
| BEAM | information_extraction | All@10 | 21 | 36 | 40 | yes |
| BEAM | information_extraction | S@1 | 15 | 36 | 40 | yes |
| BEAM | information_extraction | S@5 | 25 | 36 | 40 | yes |
| BEAM | information_extraction | S@10 | 27 | 36 | 40 | yes |
| BEAM | instruction_following | Turn@1 | 6 | 36 | 40 | yes |
| BEAM | instruction_following | Turn@5 | 18 | 36 | 40 | yes |
| BEAM | instruction_following | Turn@10 | 21 | 36 | 40 | yes |
| BEAM | instruction_following | All@5 | 13 | 36 | 40 | yes |
| BEAM | instruction_following | All@10 | 16 | 36 | 40 | yes |
| BEAM | instruction_following | S@1 | 4 | 36 | 40 | yes |
| BEAM | instruction_following | S@5 | 16 | 36 | 40 | yes |
| BEAM | instruction_following | S@10 | 17 | 36 | 40 | yes |
| BEAM | knowledge_update | Turn@1 | 30 | 36 | 39 | yes |
| BEAM | knowledge_update | All@5 | 24 | 36 | 39 | yes |
| BEAM | knowledge_update | All@10 | 27 | 36 | 39 | yes |
| BEAM | knowledge_update | S@1 | 24 | 36 | 38 | yes |
| BEAM | multi_session_reasoning | Turn@1 | 21 | 36 | 40 | yes |
| BEAM | multi_session_reasoning | Turn@5 | 34 | 36 | 40 | yes |
| BEAM | multi_session_reasoning | All@5 | 4 | 36 | 40 | yes |
| BEAM | multi_session_reasoning | All@10 | 7 | 36 | 40 | yes |
| BEAM | multi_session_reasoning | S@1 | 20 | 36 | 36 | yes |
| BEAM | multi_session_reasoning | S@5 | 33 | 36 | 36 | yes |
| BEAM | preference_following | Turn@1 | 7 | 36 | 39 | yes |
| BEAM | preference_following | Turn@5 | 14 | 36 | 39 | yes |
| BEAM | preference_following | Turn@10 | 16 | 36 | 39 | yes |
| BEAM | preference_following | All@5 | 11 | 36 | 39 | yes |
| BEAM | preference_following | All@10 | 13 | 36 | 39 | yes |
| BEAM | preference_following | S@1 | 4 | 36 | 39 | yes |
| BEAM | preference_following | S@5 | 14 | 36 | 39 | yes |
| BEAM | preference_following | S@10 | 24 | 36 | 39 | yes |
| BEAM | summarization | Turn@1 | 6 | 33 | 34 | yes |
| BEAM | summarization | Turn@5 | 17 | 33 | 34 | yes |
| BEAM | summarization | Turn@10 | 25 | 33 | 34 | yes |
| BEAM | summarization | All@5 | 0 | 33 | 34 | yes |
| BEAM | summarization | All@10 | 0 | 33 | 34 | yes |
| BEAM | summarization | S@1 | 6 | 33 | 36 | yes |
| BEAM | summarization | S@5 | 24 | 33 | 36 | yes |
| BEAM | summarization | S@10 | 26 | 33 | 36 | yes |
| BEAM | temporal_reasoning | Turn@1 | 27 | 36 | 40 | yes |
| BEAM | temporal_reasoning | All@5 | 24 | 36 | 40 | yes |
| BEAM | temporal_reasoning | All@10 | 27 | 36 | 40 | yes |
| BEAM | temporal_reasoning | S@1 | 24 | 36 | 40 | yes |
| LME | single-session-user | Turn@1 | 54 | 58 | 62 | yes |
| LME | multi-session | Turn@1 | 88 | 109 | 121 | yes |
| LME | multi-session | All@5 | 53 | 109 | 121 | yes |
| LME | single-session-preference | Turn@1 | 12 | 27 | 30 | yes |
| LME | single-session-preference | Turn@5 | 19 | 27 | 30 | yes |
| LME | single-session-preference | Turn@10 | 19 | 27 | 30 | yes |
| LME | single-session-preference | All@5 | 14 | 27 | 30 | yes |
| LME | single-session-preference | S@1 | 17 | 27 | 30 | yes |
| LME | single-session-preference | S@5 | 26 | 27 | 30 | yes |
| LME | temporal-reasoning | Turn@1 | 93 | 115 | 125 | yes |
| LME | temporal-reasoning | Turn@5 | 111 | 115 | 125 | yes |
| LME | temporal-reasoning | All@5 | 74 | 115 | 125 | yes |
| LME | temporal-reasoning | S@1 | 114 | 115 | 124 | yes |
| LME | knowledge-update | Turn@1 | 62 | 65 | 71 | yes |
| LME | knowledge-update | All@5 | 59 | 65 | 72 | yes |
| LME | single-session-assistant | Turn@1 | 41 | 51 | 52 | yes |

Every category metric that is below 90% today has a union ceiling at or above its need. That is an upper bound under the stage claims, not a statement that the patterns occur on those questions.

### Each rule

Unchanged bench metrics are omitted on the row. The full claim string is the `affect` field used to build the row.

- S1-001 RECALL/INDEX: BEAM T@1 156→188; BEAM T@5 238→263; BEAM T@10 267→284; BEAM All@5 118→157; BEAM All@10 138→177
  - claims: `BEAM/event_ordering:t5+k+pool+all5+all10`
- S1-002 RECALL/INDEX: LOC T@1 796→1066; LOC T@5 1127→1397; LOC T@10 1226→1417
  - claims: `LOC/Multi-Hop:pool+k+none; LOC/Single-Hop:pool+k`
- S1-003 RECALL/INDEX: LOC T@1 796→907; LOC T@5 1127→1238; LOC T@10 1226→1309
  - claims: `LOC/Multi-Hop:pool+k`
- S1-004 RECALL/INDEX: LOC T@1 796→974; LOC T@5 1127→1305; LOC T@10 1226→1404
  - claims: `LOC/Multi-Hop:pool; LOC/Single-Hop:pool`
- S1-005 RECALL/INDEX: LOC T@1 796→974; LOC T@5 1127→1305; LOC T@10 1226→1404
  - claims: `LOC/Single-Hop:pool; LOC/Multi-Hop:pool`
- S1-006 RECALL/INDEX: LOC T@1 796→1025; LOC T@5 1127→1356; LOC T@10 1226→1404
  - claims: `LOC/Single-Hop:pool+k; LOC/Multi-Hop:pool`
- S1-007 ORDERING: LME T@1 350→368; LOC T@1 796→857; BEAM T@1 156→166
  - claims: `LOC/Temporal:t5; LME/temporal-reasoning:t5; BEAM/temporal_reasoning:t5`
- S1-008 ORDERING: LME T@1 350→368; LOC T@1 796→857; BEAM T@1 156→163
  - claims: `LOC/Temporal:t5; LME/temporal-reasoning:t5; BEAM/event_ordering:t5`
- S1-009 ORDERING: LME T@1 350→386; LOC T@1 796→988; BEAM T@1 156→167
  - claims: `LOC/Single-Hop:t5; LME/single-session-assistant:t5; LME/multi-session:t5; BEAM/summarization:t5`
- S1-010 RECALL/INDEX: LOC T@1 796→879; LOC T@5 1127→1210; LOC T@10 1226→1309; BEAM T@1 156→159; BEAM T@5 238→241; BEAM T@10 267→270
  - claims: `LOC/Multi-Hop:pool; BEAM/information_extraction:pool`
- S1-011 RECALL/INDEX: LOC T@1 796→1132; LOC T@5 1127→1210; LOC T@10 1226→1309
  - claims: `LOC/Multi-Hop:t5+pool; LOC/Single-Hop:t5`
- S1-012 RECALL/INDEX: LME T@1 350→375; LOC T@1 796→1049
  - claims: `LOC/Multi-Hop:t5; LOC/Single-Hop:t5; LME/multi-session:t5`
- S1-013 ORDERING: LOC T@1 796→988; BEAM T@1 156→163
  - claims: `BEAM/event_ordering:t5; LOC/Single-Hop:t5`
- S1-014 ORDERING: LME T@1 350→368; LOC T@1 796→857; BEAM T@1 156→163
  - claims: `BEAM/event_ordering:t5; LOC/Temporal:t5; LME/temporal-reasoning:t5`
- S1-015 ORDERING: LOC T@1 796→988; BEAM T@1 156→163
  - claims: `BEAM/event_ordering:t5; LOC/Single-Hop:t5`
- S1-016 ORDERING: BEAM T@1 156→170
  - claims: `BEAM/event_ordering:t5; BEAM/knowledge_update:t5`
- S1-017 RECALL/INDEX: LOC T@1 796→974; LOC T@5 1127→1305; LOC T@10 1226→1404
  - claims: `LOC/Multi-Hop:pool; LOC/Single-Hop:pool`
- S1-018 RECALL/INDEX: LOC T@1 796→907; LOC T@5 1127→1238; LOC T@10 1226→1309
  - claims: `LOC/Multi-Hop:pool+k`
- S1-019 ORDERING: LME T@1 350→357; BEAM T@1 156→163
  - claims: `BEAM/preference_following:t5; LME/single-session-preference:t5`
- S1-020 RECALL/INDEX: LOC T@1 796→879; LOC T@5 1127→1210; LOC T@10 1226→1309
  - claims: `LOC/Multi-Hop:pool`
- S1-021 RECALL/INDEX: LOC T@1 796→974; LOC T@5 1127→1305; LOC T@10 1226→1404
  - claims: `LOC/Multi-Hop:pool; LOC/Single-Hop:pool`
- S1-022 RECALL/INDEX: LOC T@1 796→891; LOC T@5 1127→1222; LOC T@10 1226→1321
  - claims: `LOC/Single-Hop:pool`
- S1-023 ORDERING: LOC T@1 796→1005
  - claims: `LOC/Single-Hop:t5; LOC/Open-Domain:t5`
- S1-024 RECALL/INDEX: LOC T@1 796→891; LOC T@5 1127→1222; LOC T@10 1226→1321; BEAM T@1 156→169; BEAM T@5 238→251; BEAM T@10 267→278
  - claims: `BEAM/information_extraction:pool+k+none; LOC/Single-Hop:pool`
- S1-025 ORDERING: LOC T@1 796→857; BEAM T@1 156→163
  - claims: `LOC/Temporal:t5; BEAM/event_ordering:t5`
- S1-026 ORDERING: LME T@1 350→368; LOC T@1 796→857
  - claims: `LOC/Temporal:t5; LME/temporal-reasoning:t5`
- S1-027 RECALL/INDEX: LOC T@1 796→941; LOC T@5 1127→1272; LOC T@10 1226→1371; LOC S@1 1133→1315
  - claims: `LOC/Open-Domain:none+pool+s_rec; LOC/Multi-Hop:none+pool+s_rec`
- S1-028 RECALL/INDEX: LOC T@1 796→926; LOC T@5 1127→1257; LOC T@10 1226→1356
  - claims: `LOC/Single-Hop:pool; LOC/Open-Domain:pool`
- S1-029 ORDERING: LOC T@1 796→988
  - claims: `LOC/Single-Hop:t5`
- S1-030 ORDERING: LOC T@1 796→857; BEAM T@1 156→163
  - claims: `BEAM/event_ordering:t5; LOC/Temporal:t5`
- S1-031 ORDERING: LOC T@1 796→988; BEAM T@1 156→167
  - claims: `LOC/Single-Hop:t5; BEAM/summarization:t5`
- S1-032 RECALL/INDEX: LOC T@1 796→879; LOC T@5 1127→1210; LOC T@10 1226→1309; BEAM T@1 156→169; BEAM T@5 238→251; BEAM T@10 267→278; BEAM All@5 118→138; BEAM All@10 138→157
  - claims: `BEAM/information_extraction:pool+k+none+all5; LOC/Multi-Hop:pool`
- S1-033 RECALL/INDEX: LOC T@1 796→891; LOC T@5 1127→1222; LOC T@10 1226→1321; BEAM T@1 156→167; BEAM T@5 238→249; BEAM T@10 267→278
  - claims: `BEAM/information_extraction:pool+none; LOC/Single-Hop:pool`
- S1-034 ORDERING: LME T@1 350→361; LOC T@1 796→1005
  - claims: `LOC/Single-Hop:t5; LOC/Open-Domain:t5; LME/single-session-assistant:t5`
- S1-035 RECALL/INDEX: LOC T@1 796→928; LOC T@5 1127→1259; LOC T@10 1226→1358
  - claims: `LOC/Open-Domain:pool+none; LOC/Multi-Hop:pool`
- S1-036 RECALL/INDEX: LOC T@1 796→848; LOC T@5 1127→1162; LOC T@10 1226→1261; BEAM T@1 156→168
  - claims: `LOC/Open-Domain:t5+pool; BEAM/instruction_following:t5`
- S1-037 RECALL/INDEX: LME T@1 350→359; LME All@5 314→327; LOC T@1 796→870; LOC T@5 1127→1184; LOC T@10 1226→1275; BEAM T@1 156→190; BEAM T@5 238→260; BEAM T@10 267→286; BEAM All@5 118→145; BEAM All@10 138→162
  - claims: `LOC/Open-Domain:none+pool+k+t5; BEAM/instruction_following:none+pool+k+t5+all5; LME/knowledge-update:t5+all5`
- S1-038 RECALL/INDEX: LME T@1 350→364; LOC T@1 796→861; LOC T@5 1127→1192; LOC T@10 1226→1279; BEAM T@1 156→157; BEAM T@5 238→239; BEAM T@10 267→268
  - claims: `LOC/Temporal:pool+k; LME/temporal-reasoning:pool+k; BEAM/temporal_reasoning:pool`
- S1-039 ORDERING: LME T@1 350→368; LOC T@1 796→857
  - claims: `LOC/Temporal:t5; LME/temporal-reasoning:t5`
- S1-040 ORDERING: LME T@1 350→368; LOC T@1 796→857; BEAM T@1 156→166
  - claims: `LOC/Temporal:t5; BEAM/temporal_reasoning:t5; LME/temporal-reasoning:t5`
- S1-041 RECALL/INDEX: BEAM T@1 156→188; BEAM T@5 238→263; BEAM T@10 267→284; BEAM All@5 118→157; BEAM All@10 138→177
  - claims: `BEAM/event_ordering:t5+k+pool+all5+all10`
- S1-042 RECALL/INDEX: LOC T@1 796→1066; LOC T@5 1127→1397; LOC T@10 1226→1417
  - claims: `LOC/Multi-Hop:pool+k+none; LOC/Single-Hop:pool+k`
- S1-043 RECALL/INDEX: LOC T@1 796→974; LOC T@5 1127→1305; LOC T@10 1226→1404
  - claims: `LOC/Single-Hop:pool; LOC/Multi-Hop:pool`
- S1-044 RECALL/INDEX: LME T@1 350→359; LOC T@1 796→932; LOC T@5 1127→1263; LOC T@10 1226→1362
  - claims: `LOC/Temporal:pool; LOC/Multi-Hop:pool; LME/temporal-reasoning:pool`
- S1-045 ORDERING: LOC T@1 796→1049
  - claims: `LOC/Single-Hop:t5; LOC/Multi-Hop:t5`
- S1-046 ORDERING: LME T@1 350→394; LOC T@1 796→988; BEAM T@1 156→174
  - claims: `LOC/Single-Hop:t5; LME/single-session-assistant:t5; LME/single-session-user:t5; LME/multi-session:t5; BEAM/summarization:t5; BEAM/information_extraction:t5`
- S1-047 ORDERING: LME T@1 350→361; LOC T@1 796→988; BEAM T@1 156→168
  - claims: `LOC/Single-Hop:t5; BEAM/instruction_following:t5; LME/single-session-assistant:t5`
- S1-048 RECALL/INDEX: LOC T@1 796→1114; LOC T@5 1127→1384; LOC T@10 1226→1404
  - claims: `LOC/Multi-Hop:t5+k+pool; LOC/Single-Hop:k+pool`
- S1-049 RECALL/INDEX: LME T@1 350→368; LME All@5 314→330; LOC T@1 796→879; LOC T@5 1127→1210; LOC T@10 1226→1309; BEAM T@1 156→188; BEAM T@5 238→263; BEAM T@10 267→290; BEAM All@5 118→146; BEAM All@10 138→164; BEAM S@1 133→168; BEAM S@5 249→274; BEAM S@10 275→290
  - claims: `BEAM/preference_following:t5+k+pool+none+s_rec+all5+all10; LME/single-session-preference:t5+pool+s_rec+all5; LOC/Multi-Hop:pool`
- S1-050 RECALL/INDEX: LOC T@1 796→977; LOC T@5 1127→1308; LOC T@10 1226→1356
  - claims: `LOC/Single-Hop:k+pool; LOC/Open-Domain:pool`
- S1-051 ORDERING: LOC T@1 796→1005
  - claims: `LOC/Single-Hop:t5; LOC/Open-Domain:t5`
- S1-052 RECALL/INDEX: LOC T@1 796→891; LOC T@5 1127→1222; LOC T@10 1226→1321; BEAM T@1 156→176; BEAM T@5 238→251; BEAM T@10 267→278; BEAM All@5 118→138; BEAM All@10 138→157
  - claims: `BEAM/information_extraction:t5+k+pool+none+all5+all10; LOC/Single-Hop:pool`
- S1-053 RECALL/INDEX: LME T@1 350→364; LOC T@1 796→944; LOC T@5 1127→1275; LOC T@10 1226→1362
  - claims: `LOC/Temporal:k+pool; LOC/Multi-Hop:pool; LME/temporal-reasoning:k+pool`
- S1-054 ORDERING: LME T@1 350→368; LOC T@1 796→857; BEAM T@1 156→166
  - claims: `LOC/Temporal:t5; LME/temporal-reasoning:t5; BEAM/temporal_reasoning:t5`
- S1-055 RECALL/INDEX: LOC T@1 796→966; LOC T@5 1127→1280; LOC T@10 1226→1371; LOC S@1 1133→1315; BEAM T@1 156→161; BEAM T@5 238→243; BEAM T@10 267→272
  - claims: `LOC/Open-Domain:t5+k+pool+none+s_rec; LOC/Multi-Hop:pool+none+s_rec; BEAM/instruction_following:none`
- S1-056 RECALL/INDEX: LOC S@1 1133→1479; BEAM S@1 133→347; BEAM S@5 249→347; BEAM S@10 275→349
  - claims: `LOC/Single-Hop:s_re; LOC/Multi-Hop:s_rec; LOC/Temporal:s_re; LOC/Open-Domain:s_rec; BEAM/event_ordering:s_rec; BEAM/information_extraction:s_rec; BEAM/instruction_following:s_rec; BEAM/preference_following:s_rec; BEAM/summarization:s_rec; BEAM/contradiction_resolution:s_k; BEAM/knowledge_update:s_k; BEAM/temporal_reasoning:s_re; BEAM/multi_session_reasoning:s_k; LME/single-session-preference:s_k; LME/temporal-reasoning:s_re`
- S1-057 RECALL/INDEX: LOC T@1 796→1025; LOC T@5 1127→1356; LOC T@10 1226→1404
  - claims: `LOC/Single-Hop:k+pool; LOC/Multi-Hop:pool`
- S1-058 RECALL/INDEX: LME T@1 350→375; LOC T@1 796→988; BEAM T@1 156→164; BEAM All@5 118→136; BEAM All@10 138→151
  - claims: `LOC/Single-Hop:t5; LME/multi-session:t5; BEAM/contradiction_resolution:t5+all5`
- S1-059 ORDERING: LME T@1 350→368; LOC T@1 796→857; BEAM T@1 156→163
  - claims: `BEAM/event_ordering:t5; LME/temporal-reasoning:t5; LOC/Temporal:t5`
- S1-060 RECALL/INDEX: LOC T@1 796→891; LOC T@5 1127→1222; LOC T@10 1226→1321; BEAM T@1 156→176; BEAM T@5 238→251; BEAM T@10 267→278; BEAM All@5 118→138; BEAM All@10 138→157
  - claims: `BEAM/information_extraction:t5+k+pool+none+all5+all10; LOC/Single-Hop:pool`
- S1-061 ORDERING: LME T@1 350→361; LOC T@1 796→1005; BEAM T@1 156→167
  - claims: `LOC/Single-Hop:t5; LOC/Open-Domain:t5; LME/single-session-assistant:t5; BEAM/summarization:t5`
- S1-062 RECALL/INDEX: LME T@1 350→359; LME All@5 314→327; LOC T@1 796→870; LOC T@5 1127→1184; LOC T@10 1226→1275; BEAM T@1 156→190; BEAM T@5 238→260; BEAM T@10 267→286; BEAM All@5 118→145; BEAM All@10 138→162
  - claims: `LOC/Open-Domain:none+pool+k+t5; BEAM/instruction_following:none+pool+k+t5+all5+all10; LME/knowledge-update:t5+all5`
- S1-063 RECALL/INDEX: LME T@1 350→375; LOC T@1 796→988; BEAM T@1 156→181
  - claims: `LOC/Single-Hop:t5; BEAM/instruction_following:t5; LME/multi-session:t5; BEAM/multi_session_reasoning:t5`
- S1-064 RECALL/INDEX: LME T@1 350→382; LME All@5 314→365; LOC T@1 796→922; LOC T@5 1127→1192; LOC T@10 1226→1279; BEAM T@1 156→159; BEAM T@5 238→241; BEAM T@10 267→268; BEAM All@5 118→134; BEAM All@10 138→151
  - claims: `LOC/Temporal:k+pool+t5; LME/temporal-reasoning:k+pool+t5+all5; BEAM/temporal_reasoning:k+pool+all5`
- S1-065 ORDERING: LME T@1 350→368; LOC T@1 796→857; BEAM T@1 156→173
  - claims: `LOC/Temporal:t5; LME/temporal-reasoning:t5; BEAM/temporal_reasoning:t5; BEAM/knowledge_update:t5`
- S1-066 ORDERING: LME T@1 350→368; LOC T@1 796→857; BEAM T@1 156→173
  - claims: `LOC/Temporal:t5; BEAM/temporal_reasoning:t5; LME/temporal-reasoning:t5; BEAM/event_ordering:t5`
- S1-067 RECALL/INDEX: LOC T@1 796→995; LOC T@5 1127→1326; LOC T@10 1226→1374
  - claims: `LOC/Single-Hop:pool+k; LOC/Temporal:pool`
- S1-068 RECALL/INDEX: LOC T@1 796→1002; LOC T@5 1127→1333; LOC T@10 1226→1404
  - claims: `LOC/Multi-Hop:pool+k; LOC/Single-Hop:pool`
- S1-069 RECALL/INDEX: LOC T@1 796→991; LOC T@5 1127→1261; LOC T@10 1226→1309
  - claims: `LOC/Multi-Hop:pool+t5; LOC/Single-Hop:k`
- S1-070 RECALL/INDEX: LOC T@1 796→891; LOC T@5 1127→1222; LOC T@10 1226→1321; BEAM T@1 156→167; BEAM T@5 238→249; BEAM T@10 267→278
  - claims: `LOC/Single-Hop:pool; BEAM/information_extraction:pool+none`
- S1-071 ORDERING: LME T@1 350→394; LOC T@1 796→988; BEAM T@1 156→174
  - claims: `LOC/Single-Hop:t5; LME/single-session-assistant:t5; LME/single-session-user:t5; LME/multi-session:t5; BEAM/summarization:t5; BEAM/preference_following:t5`
- S1-072 RECALL/INDEX: LOC T@1 796→941; LOC T@5 1127→1272; LOC T@10 1226→1371; BEAM T@1 156→165; BEAM T@5 238→247; BEAM T@10 267→276; BEAM All@5 118→152; BEAM All@10 138→172
  - claims: `LOC/Open-Domain:none+pool; LOC/Multi-Hop:none+pool; BEAM/summarization:pool+all5+all10`
- S1-073 RECALL/INDEX: LME T@1 350→369; LME All@5 314→462; BEAM T@1 156→254; BEAM T@5 238→305; BEAM T@10 267→312; BEAM All@5 118→264; BEAM All@10 138→275
  - claims: `BEAM/event_ordering:t5+k+pool+all5+all10; BEAM/summarization:t5+k+pool+all5+all10; BEAM/multi_session_reasoning:t5+k+pool+all5+all10; BEAM/instruction_following:k+pool+all5+all10; BEAM/knowledge_update:k+all5+all10; LME/multi-session:pool+all5; LME/temporal-reasoning:all5; LME/knowledge-update:all5; LME/single-session-preference:pool+all5`
- S1-074 RECALL/INDEX: LOC T@1 796→1053; LOC T@5 1127→1384; LOC T@10 1226→1404
  - claims: `LOC/Single-Hop:pool+k; LOC/Multi-Hop:pool+k`
- S1-075 RECALL/INDEX: LOC T@1 796→810; LOC T@5 1127→1141; LOC T@10 1226→1240; LOC S@1 1133→1189; BEAM T@1 156→161; BEAM T@5 238→243; BEAM T@10 267→272; BEAM S@1 133→169; BEAM S@5 249→273; BEAM S@10 275→298
  - claims: `LOC/Open-Domain:s_rec+none; BEAM/instruction_following:s_rec+none`
- S1-076 ORDERING: LME T@1 350→368; LOC T@1 796→857; BEAM T@1 156→163
  - claims: `LOC/Temporal:t5; BEAM/event_ordering:t5; LME/temporal-reasoning:t5`
- S1-077 RECALL/INDEX: LOC T@1 796→879; LOC T@5 1127→1210; LOC T@10 1226→1309; BEAM T@1 156→169; BEAM T@5 238→251; BEAM T@10 267→278
  - claims: `BEAM/information_extraction:pool+k+none; LOC/Multi-Hop:pool`
- S1-078 RECALL/INDEX: LOC T@1 796→845; LOC T@5 1127→1176; LOC T@10 1226→1275; BEAM T@1 156→176; BEAM T@5 238→258; BEAM T@10 267→287; BEAM All@5 118→138; BEAM All@10 138→157
  - claims: `LOC/Open-Domain:none+pool; BEAM/information_extraction:none+pool+all5; BEAM/preference_following:none`
- S1-079 RECALL/INDEX: LOC T@1 796→861; LOC T@5 1127→1192; LOC T@10 1226→1279; LOC S@1 1133→1259; BEAM T@1 156→157; BEAM T@5 238→239; BEAM T@10 267→268; BEAM All@5 118→134; BEAM All@10 138→151; BEAM S@1 133→149; BEAM S@5 249→251
  - claims: `LOC/Temporal:pool+k; LOC/Multi-Hop:s_rec; BEAM/temporal_reasoning:pool+s_rec+all5`
- S1-080 ORDERING: LOC T@1 796→1005; BEAM T@1 156→163
  - claims: `LOC/Single-Hop:t5; LOC/Open-Domain:t5; BEAM/information_extraction:t5`
- S1-081 RECALL/INDEX: LME T@1 350→361; LME All@5 314→330; LOC T@1 796→849; LOC T@5 1127→1180; LOC T@10 1226→1279; BEAM T@1 156→181; BEAM T@5 238→263; BEAM T@10 267→290; BEAM All@5 118→146; BEAM All@10 138→164
  - claims: `BEAM/preference_following:pool+none+all5+all10+k; LME/single-session-preference:pool+all5; LOC/Temporal:pool`
- S1-082 RECALL/INDEX: LOC T@1 796→940; LOC T@5 1127→1210; LOC T@10 1226→1309; BEAM T@1 156→164; BEAM All@5 118→136; BEAM All@10 138→151
  - claims: `LOC/Multi-Hop:pool+t5; BEAM/contradiction_resolution:t5+all5`
- S1-083 ORDERING: LOC T@1 796→857; BEAM T@1 156→163
  - claims: `BEAM/event_ordering:t5; LOC/Temporal:t5`
- S1-084 ORDERING: LOC T@1 796→988; BEAM T@1 156→163
  - claims: `BEAM/event_ordering:t5; LOC/Single-Hop:t5`
- S1-085 RECALL/INDEX: LOC T@1 796→1088; LOC T@5 1127→1419; LOC T@10 1226→1439
  - claims: `LOC/Multi-Hop:pool+k; LOC/Single-Hop:pool+k; LOC/Open-Domain:pool`
- S1-086 ORDERING: LME T@1 350→358; LOC T@1 796→988; BEAM T@1 156→167
  - claims: `LOC/Single-Hop:t5; LME/single-session-user:t5; BEAM/summarization:t5`
- S1-087 RECALL/INDEX: LME T@1 350→364; LOC T@1 796→861; LOC T@5 1127→1192; LOC T@10 1226→1279; BEAM T@1 156→159; BEAM T@5 238→241; BEAM T@10 267→268
  - claims: `LOC/Temporal:pool+k; LME/temporal-reasoning:pool+k; BEAM/temporal_reasoning:pool+k`
- S1-088 RECALL/INDEX: LOC T@1 796→1217; LOC T@5 1127→1356; LOC T@10 1226→1404
  - claims: `LOC/Single-Hop:pool+k+t5; LOC/Multi-Hop:pool`
- S1-089 ORDERING: LME T@1 350→368; LOC T@1 796→857; BEAM T@1 156→163
  - claims: `LOC/Temporal:t5; BEAM/event_ordering:t5; LME/temporal-reasoning:t5`
- S1-090 RECALL/INDEX: LME T@1 350→364; LME All@5 314→365; LOC T@1 796→861; LOC T@5 1127→1192; LOC T@10 1226→1279; BEAM T@1 156→189; BEAM T@5 238→264; BEAM T@10 267→285; BEAM All@5 118→173; BEAM All@10 138→190
  - claims: `BEAM/event_ordering:pool+k+t5+all5+all10; BEAM/temporal_reasoning:pool+all5; LOC/Temporal:pool+k; LME/temporal-reasoning:pool+k+all5`
