# C2 Step 33 — Triangulation

Branch `perf/c2-step33-triangulation`. Base is `d121d063be3f22c59e40b6f39d7a14c905569a43`. `git am -3` applied `steps7_10a` (5), then `algos44` (44), then the Step 31 Exchequer patch. No conflicts. Author on every commit is Sam Valladares `<samvalladares21@gmail.com>`.

The am tip before this step is `99e9d44738d440aabab5d1dc6198f523a2f277a5` (tree `ed6d9d0b8cd9b606f6bbe8f5b7cb65d97c95e0a3`). Its parent tree is `4612c06f74b115c06606306e9e84abeacb6de956`, the same tree the Step 31 log records for the 44-algo tip. The Step 31 patch's `From` line is `776d8adc8e44af3ff8f643a67881f0736ddd8ec6`. That object is not in this clone; the patch applied cleanly onto the matching parent tree, and the scoreboard below matches the published Step 31 numbers on every metric.

Datasets, hashes checked before any bench:

- `longmemeval_s_cleaned.json` sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- `locomo10.json` sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- BEAM-100K: 21 conversations under `BEAM/chats/100K`

## Dump fields (verified in `write_one_replay_dump`, retrieval.rs)

`VESTIGE_REPLAY_DUMP` is off the ranking path. `maybe_dump_replay` returns before it copies anything when the variable is unset. `write_one_replay_dump` then builds:

- `final_ids` / `seated_ids`: official turn order after the gatekeeper and every C2 seat, including Step 33. `rank` is the index in that list.
- `pre_ids`: that seated list first, then `t_scene`-descending turns with `t_scene > 0` until 50. `rank_pre` is the index in `pre_ids`. So `pre_ids[0]` is `final_ids[0]` by construction. A literal "gold at pre_rank 0" is the same count as "gold at final rank 0".
- `t_scene`: scene-pooled turn score on the snapshot. Seating does not rewrite it. Scene-rank 0 is `argmax t_scene` (tie break lower `idx`). `tools/c2_funnel.py` prints this as "gold at scene rank0".
- There is no post-Kemeny snapshot. `rank1_gatekeeper_keep_rsb_head` Kemeny-orders slots `1..k` with slot 0 frozen, then the C2 seats and Step 33 run. The dump sees only the final seated list.
- `t_own`, `t_turn_raw`, `t_scene_sess`, `s_score`, `bridge_boost`: dump-only copies of the scores `rank_one` already computed. `t_own` is the turn's uncarried BM25 (plus the contiguous-bigram bonus). `s_score + bridge_boost` is the official session rank score (`RankedItem.score` for sessions). They are not read back into ranking.
- `qtok_present` and `turn_ic_novel` read `TurnIx.body` and are dump-only. Ranking does not.

Baseline 5-line funnel from `tools/c2_funnel.py` on `/home/ubuntu/c2-bins/step31-baseline` (sha256 `c29cb6bb921913adc1cae961d6bcece5c086f1fbddd0fba851857aa48db6b227`), before the Step 33 seat:

```
[LoCoMo Multi-Hop] n=282
  gold in corpus     282
  gold in pool@50    243
  gold at scene rank0 72
  gold at final rank0 97
  gold in final top5 158
[LoCoMo Temporal] n=321
  gold in corpus     320
  gold in pool@50    299
  gold at scene rank0 182
  gold at final rank0 187
  gold in final top5 248
[LoCoMo Open-Domain] n=92
  gold in corpus     92
  gold in pool@50    63
  gold at scene rank0 14
  gold at final rank0 18
  gold in final top5 35
[LoCoMo Single-Hop] n=841
  gold in corpus     841
  gold in pool@50    810
  gold at scene rank0 440
  gold at final rank0 494
  gold in final top5 686
```

Same binary, all three benches, gold-in-field counts (pool is `pre_ids`, pre_rank@1 is `pre_ids[0]`, which equals final@1):

| bench | n | corpus | pool@50 | pre_rank@1 | pre_ids@5 | scene-rank@1 | final@1 | final@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LME_S | 470 | 470 | 463 | 351 | 428 | 265 | 351 | 428 |
| LoCoMo | 1536 | 1535 | 1415 | 796 | 1127 | 708 | 796 | 1127 |
| BEAM-100K | 355 | 355 | 328 | 156 | 239 | 124 | 156 | 238 |

BEAM `pre_ids@5` is 239 and final@5 is 238 because one seated list is shorter than 5, so the pool fill occupies a `pre_ids` slot the official list does not have. LME pool@50 463 and BEAM pool@50 328 match `missmap-bd2693e2.txt`. LoCoMo corpus is 1535 because one scored question's evidence id is not in the indexed turns.

## Baseline scoreboard

`retrieval_bench` of the Step 31 tip, before any Step 33 edit. Reproduced exactly.

- LME_S n=470: Session 430/463/468, All@5 424, All@10 447. Turn 351/428/437, All@5 314.
- LoCoMo n=1536: Turn 796/1127/1226, Session 1133/1416.
- BEAM-100K n=355: Turn 156/238/267, All@5 118, All@10 138, Session 133/249/275.

## What the misses are

LoCoMo has 198 top-5-not-#1 misses in the bd2693e2 miss map (plus the Step 31 seat, which does not touch LoCoMo). On those questions the gold turn is already in the official top 5 and a different turn holds slot 1. The two scores that still disagree with that seat are the turn's own lexical score (`t_own`) and the session score (`s_score + bridge_boost`). Slot 1 is often a scene-pooled neighbor (`t_scene` high, `t_own` low) from a session that is not the session-score leader.

## Simulations that were not shipped

Scored on replay dumps of the Step 31 binary. A swap was counted only inside the already-seated top 5. None of these was committed.

- Unique argmax of `t_own` and of `t_scene_sess` must agree, any dialogue. LME +1/−26, LoCoMo +7/−23, BEAM +4/−17. The losses are assistant echoes (or a later user question) that outscore the user or speaker who actually stated the fact. Promoting the raw score-winner undoes seats that were right.
- Same pair restricted to the head's session, to a cross-session challenger, to "not preference / not temporal / not assistant-attribution", and to same-role non-questions. Every variant still had at least one loss, or it had zero swaps.
- Session-winner internal (the lexical best turn must sit in the unique best session, and the head must be another turn of that session). LME +8/−35 and worse. That is a different rule from "best on both orderings of the turns", and it loses.
- Content-token strict cover with no score agreement: never reached a zero-loss gain by itself. The cover is the guard, not the seat.

The zero-loss simulation that matched the spec was: unique strict argmax of `t_own` and of `s_score + bridge_boost` agree on one top-5 turn, the dialogue is named-speaker, and that turn's content-token set strictly covers the head (every content token the head attests, plus one the head does not). Predicted LoCoMo +5/−0 (`2:84`, `4:28`, `4:44`, `8:124`, `8:148`) and 0 swaps on LME and BEAM.

## Attempt 1 — group cover — not committed

`apply_c2_step33_triangulation` with the agreed turn required to strictly cover the head's **query groups** (`c2_turn_groups`), not the head's terms. Binary sha256 `a188c0f592f53f0046f5803c5fcc8d4faf28ee202057206136fe3e35f56d3a83`. No commit.

LoCoMo Turn@1 796 → 798. Per-question T@1 +3/−1. Gains `1:41`, `4:28`, `8:124`. Loss `5:118` ("How often does Audrey meet up with other dog owners for tips and playdates?"). The promoted turn `D4:21` was best on both scores but did not attest `tips`, which the gold head `D27:4` did. Group collapse had dropped that term, so the group cover passed. Session bits and Turn@5 did not move. Failed the zero-loss gate. Not a threshold change: the attestation was the wrong object (groups instead of terms).

## Attempt 2 — term cover — WIN

`c2_term_ids` in `crates/vestige-core/src/algos_c2.rs` lines 95–106. `c2_unique_max` lines 2510–2528. `apply_c2_step33_triangulation` lines 2530–2593. Hook in `crates/vestige-core/src/retrieval.rs` lines 6261–6272, after the Step 31 block, before `maybe_dump_replay`. Unit test `triangulation_seats_dual_best_richer_mark_in_named_dialogue` lines 6676–6766. `rsb_seat_behind` is not called and is not edited. Turn bodies are not read.

Rule. Named-speaker dialogues only (`c2_named_session`). Over the official top 5, sightline A is `t_own` and sightline B is `s_score[session] + bridge_boost[session]`. Both sightlines must have a strict unique maximum, and it must be the same turn. Ties fix nothing. If that turn is already slot 1, it stays: a turn that is already best on both is not displaced. Otherwise it takes slot 1 only when its attested query term ids (`TurnIx.content` or `TurnIx.feats`, via `plan.terms`) include every id the head attests and at least one id the head does not. Top-5 membership is unchanged, so Turn@5, Turn@10, All@5, and the session list cannot move. User/assistant transcripts return before any swap, which is why LME and BEAM are byte-identical per question.

Real run of `/home/ubuntu/c2-bins/step33-a2` (sha256 `9cc1a7ac610adc7f5e0ad7996d74ebd659512df3f5703326bdcb262a6c0a8b0b`):

- LME_S Session 430/463/468, All@5 424, All@10 447. Turn 351/428/437, All@5 314.
- LoCoMo Turn **799**/1127/1226, Session 1133/1416. Temporal Turn@1 187 → 188. Open-Domain Turn@1 18 → 19. Single-Hop Turn@1 494 → 495. Multi-Hop Turn@1 97 unchanged.
- BEAM-100K Turn 156/238/267, All@5 118, All@10 138, Session 133/249/275.

Per-question bits vs the Step 31 json: LoCoMo Turn@1 +3 −0 (`4:28`, `4:44`, `8:124`). Every other LoCoMo bit, all 470 LME bits, and all 355 BEAM bits: +0 −0. LME Session drops: 0.

The seat only reorders inside the top 5 and does not write `t_scene`, so pool@50, scene-rank, and final@5 stay on the baseline funnel. final@1 / pre_rank@1 on LoCoMo is 799 (Temporal 188, Open-Domain 19, Single-Hop 495, Multi-Hop 97). LME and BEAM funnels are unchanged.

Gates on this tree:

- `cargo clippy -p vestige-core --all-targets -- -D warnings` exit 0
- `cargo check -p vestige-mcp` exit 0
- `cargo test -p vestige-core` 559 passed, 0 failed
- `tools/c2_hashcheck.py` ranked ids 0/500, full response including f32 scores 0/500

Winning commit is the one commit on top of `99e9d44738d440aabab5d1dc6198f523a2f277a5` that adds `apply_c2_step33_triangulation`. Its hash is `git rev-parse HEAD` of the commit that contains this file.
